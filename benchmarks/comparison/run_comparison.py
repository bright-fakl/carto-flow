"""
Compare flow cartogram implementations on US census data.

Tools: carto-flow, go-cart (cartogram-cpp), R ``cartogram`` (``cartogram_cont``),
R ``cartogramR`` (``gsm`` and ``gn``) and the Python ``cartogram`` package.

Every output is measured here with the same metrics, from the output
geometries and the target values, whatever the tool reports itself:

* area error: relative deviation of each region's share of the total area from
  its share of the total value (mean and maximum, in %);
* validity: the number of output regions that are not valid polygons.

A tool does not decide when it is accurate enough. The iterative tools run with
a stopping tolerance that cannot be reached and an increasing iteration cap, so
each run stops at its cap and the measured error of that run is the result. The
time to a target is the shortest time among the runs that measure within it.
go-cart has no iteration cap; it runs with a ladder of maximum area errors
(its stopping rule is a maximum area error, like the target).

Run it with the Docker image (see README.md) or, with all tools installed,
``python run_comparison.py --dataset states --output results.json``.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import geopandas as gpd
import numpy as np

HERE = Path(__file__).resolve().parent

# Iteration caps of the iterative tools.
ITERATIONS = {
    "carto-flow": [5, 10, 20, 40, 80, 160, 320],
    "Python cartogram": [5, 10, 20, 50, 100, 200],
    "R cartogram": [5, 10, 20, 50, 100],
    "cartogramR gsm": [2, 5, 10, 20, 50],
    "cartogramR gn": [2, 5, 10, 20, 50],
}
# Maximum area errors go-cart is asked to reach (fractions).
GO_CART_TOLERANCES = [0.05, 0.02, 0.01, 0.005, 0.002]
# carto-flow validity modes run as separate tools; plain "carto-flow" only checks.
CARTO_FLOW_VARIANTS = {
    "carto-flow": "check",
    "carto-flow (repair-crossings)": "repair-crossings",
    "carto-flow (repair-quadtree)": "repair-quadtree",
}
# Accuracy targets for the time-to-target summary (%).
TARGETS_PCT = [5.0, 1.0, 0.5]

R_TOOLS = ("R cartogram", "cartogramR gsm", "cartogramR gn")
PY_TOOLS = (*CARTO_FLOW_VARIANTS, "Python cartogram")
ALL_TOOLS = (*PY_TOOLS, *R_TOOLS, "go-cart")


# ---------------------------------------------------------------------------
# Data and metrics
# ---------------------------------------------------------------------------


def load_dataset(name: str) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame with columns ``ID`` (non-numeric), ``value`` and geometry."""
    from carto_flow.data import load_us_census

    if name == "states":
        gdf = load_us_census(population=True, contiguous_only=True)
    elif name == "districts":
        gdf = load_us_census(level="congressional_district", population=True, contiguous_only=True, simplify=1000)
    elif name == "counties":
        gdf = load_us_census(level="county", population=True, contiguous_only=True, simplify=1000)
    else:
        raise ValueError(f"unknown dataset {name!r}")
    gdf = gdf.reset_index(drop=True)
    return gpd.GeoDataFrame(
        {"ID": [f"R{i}" for i in range(len(gdf))], "value": gdf["Population"].to_numpy()},
        geometry=gdf.geometry,
        crs=gdf.crs,
    )


def area_errors(geoms, values) -> tuple[float, float]:
    """Mean and maximum relative error (%) of the area shares against the value shares."""
    area = np.asarray(geoms.area, dtype=float)
    share = area / area.sum()
    target = np.asarray(values, dtype=float) / np.sum(values)
    err = np.abs(share / target - 1.0) * 100
    return float(err.mean()), float(err.max())


def record(gdf, out_geoms, seconds: list[float], **extra) -> dict:
    mean_err, max_err = area_errors(out_geoms, gdf["value"])
    return {
        "time_s": statistics.median(seconds),
        "times_s": seconds,
        "mean_area_error_pct": mean_err,
        "max_area_error_pct": max_err,
        "n_invalid": int((~out_geoms.is_valid).sum()),
        **extra,
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def timed(fn, repeats: int):
    seconds, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        seconds.append(time.perf_counter() - t0)
    return seconds, out


def run_carto_flow(gdf, repeats, validity):
    from carto_flow.flow_cartogram import MorphOptions, create_flow_cartogram

    def call(n_iter):
        # Tolerances that cannot be reached and no stall rule: the run stops at n_iter.
        options = MorphOptions(
            n_iter=n_iter, mean_tol=1e-6, max_tol=1e-6, stall_patience=None, validity=validity, show_progress=False
        )
        return create_flow_cartogram(gdf, "value", options=options)

    # Compile the numba kernels outside the timed runs.
    create_flow_cartogram(gdf, "value", options=MorphOptions(grid=64, n_iter=2, show_progress=False))
    rows = []
    for n_iter in ITERATIONS["carto-flow"]:
        seconds, res = timed(lambda n_iter=n_iter: call(n_iter), repeats)
        out = res.to_geodataframe(include_errors=False, include_density=False).geometry
        rows.append(record(gdf, out, seconds, max_iterations=n_iter))
    return rows


def run_python_cartogram(gdf, repeats):
    from cartogram import Cartogram

    # The package reads the attribute named in the call and returns a GeoDataFrame.
    data = gdf[["value", "geometry"]]
    rows = []
    for n_iter in ITERATIONS["Python cartogram"]:
        # A mean error of 0 is never reached, so the run stops at max_iterations.
        seconds, res = timed(
            lambda n_iter=n_iter: Cartogram(data, "value", max_iterations=n_iter, max_average_error=0.0), repeats
        )
        rows.append(record(gdf, res.geometry, seconds, max_iterations=n_iter))
    return rows


def run_go_cart(gdf, repeats, binary, workdir):
    import pandas as pd

    geojson, csv = workdir / "go_cart.geojson", workdir / "go_cart.csv"
    gdf[["ID", "geometry"]].to_crs(4326).to_file(geojson, driver="GeoJSON")
    pd.DataFrame({"ID": gdf["ID"], "Population (people)": gdf["value"]}).to_csv(csv, index=False)
    rows = []
    for tol in GO_CART_TOLERANCES:
        outdir = workdir / f"go_cart_{tol}"
        outdir.mkdir(exist_ok=True)

        def call(tol=tol, outdir=outdir):
            subprocess.run(  # noqa: S603
                [str(binary), str(geojson), str(csv), "--max_permitted_area_error", str(tol)],
                cwd=outdir,
                check=True,
                capture_output=True,
            )

        seconds, _ = timed(call, repeats)
        # go-cart writes coordinates in its own planar projection, not EPSG:4326.
        out = gpd.read_file(next(outdir.glob("*_cartogram.geojson"))).set_index("ID").loc[gdf["ID"]].geometry
        out = out.set_crs(None, allow_override=True)
        rows.append(record(gdf, out.reset_index(drop=True), seconds, tolerance=tol))
    return rows


def run_r_tools(gdf, repeats, workdir):
    """Run the three R tools in one R process; outputs come back as GeoPackages."""
    gdf.to_file(workdir / "input.gpkg", driver="GPKG")
    (workdir / "r_ladders.json").write_text(json.dumps({t: ITERATIONS[t] for t in R_TOOLS}))
    subprocess.run(  # noqa: S603
        ["Rscript", str(HERE / "run_r.R"), str(workdir), str(repeats)],  # noqa: S607
        check=True,
    )
    timings = json.loads((workdir / "r_times.json").read_text())
    results = {tool: [] for tool in R_TOOLS}
    for item in timings:
        out = gpd.read_file(workdir / item["file"]).geometry
        results[item["tool"]].append(record(gdf, out, item["times_s"], max_iterations=item["max_iterations"]))
    return results


# ---------------------------------------------------------------------------
# Summary and entry point
# ---------------------------------------------------------------------------


def time_to_target(rows: list[dict]) -> dict:
    """Shortest median time among the runs whose measured maximum area error is within each target."""
    out = {}
    for target in TARGETS_PCT:
        times = [r["time_s"] for r in rows if r["max_area_error_pct"] <= target]
        out[f"{target:g}%"] = min(times) if times else None
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["states", "districts", "counties"], default="states")
    parser.add_argument("--tools", nargs="+", choices=ALL_TOOLS, default=list(ALL_TOOLS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--go-cart", default=shutil.which("cartogram") or "cartogram", help="path of the go-cart binary"
    )
    parser.add_argument("--output", type=Path, default=Path("comparison_results.json"))
    args = parser.parse_args()

    gdf = load_dataset(args.dataset)
    results: dict[str, list[dict]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for name, validity in CARTO_FLOW_VARIANTS.items():
            if name in args.tools:
                results[name] = run_carto_flow(gdf, args.repeats, validity)
        if "Python cartogram" in args.tools:
            results["Python cartogram"] = run_python_cartogram(gdf, args.repeats)
        if "go-cart" in args.tools:
            results["go-cart"] = run_go_cart(gdf, args.repeats, args.go_cart, workdir)
        if any(t in args.tools for t in R_TOOLS):
            r = run_r_tools(gdf, args.repeats, workdir)
            results.update({t: v for t, v in r.items() if t in args.tools})

    payload = {
        "dataset": args.dataset,
        "n_regions": len(gdf),
        "repeats": args.repeats,
        "iterations": ITERATIONS,
        "go_cart_tolerances": GO_CART_TOLERANCES,
        "machine": {"platform": platform.platform(), "processor": platform.processor()},
        "tools": {tool: {"runs": rows, "time_to_target_s": time_to_target(rows)} for tool, rows in results.items()},
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")
    for tool, entry in payload["tools"].items():
        print(f"{tool:18s}", {k: (None if v is None else round(v, 2)) for k, v in entry["time_to_target_s"].items()})


if __name__ == "__main__":
    main()
