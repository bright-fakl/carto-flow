"""Draw the example cartograms of a comparison run into one image.

Reads ``comparison_<dataset>.json`` and ``comparison_<dataset>.gpkg`` written by ``run_comparison.py`` and writes
``comparison_<dataset>_examples.png`` next to them.

    python render_examples.py RESULTS_DIR [OUTPUT_DIR]
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


def render(results: Path, dataset: str, output: Path) -> None:
    with open(results / f"comparison_{dataset}.json") as f:
        data = json.load(f)
    path = results / f"comparison_{dataset}.gpkg"
    panels = [("input", None)] + [(tool, e["example"]) for tool, e in data["tools"].items() if "example" in e]
    ncols = 3
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.9 * nrows))
    lw = 0.3 if dataset == "states" else 0.1
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, (name, ex) in zip(axes.ravel(), panels, strict=False):
        gdf = gpd.read_file(path, layer=name)
        gdf.plot(ax=ax, column="value", cmap="viridis", norm=mpl.colors.LogNorm(), edgecolor="white", linewidth=lw)
        if ex is None:
            title = "input"
        else:
            setting = f"{ex['max_iterations']} it." if "max_iterations" in ex else f"tol. {ex['tolerance']:g}"
            title = (
                f"{name}\n{setting}, {ex['time_s']:.1f} s, max error {ex['max_area_error_pct']:.2g} %, "
                f"{ex['n_invalid']} invalid"
            )
        ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / f"comparison_{dataset}_examples.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    results = Path(sys.argv[1])
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else results
    for ds in ("states", "districts", "counties"):
        if (results / f"comparison_{ds}.gpkg").exists():
            render(results, ds, output)
