#!/usr/bin/env python3
"""Build the bundled ACS 2020 census snapshot used by ``carto_flow.data.load_us_census``.

``api.census.gov`` now rejects keyless requests, so ``load_us_census`` can no
longer hit the live API at runtime (see the CI docs build failure noted in
the v2.0 plan, Phase 0.5). Instead, the package bundles a snapshot of the ACS
2020 5-year estimates, downloaded once with this script and shipped as
GeoParquet under ``src/carto_flow/data/``.

This script requires the optional ``censusdis`` dependency (``pip install
carto-flow[data]``) and a Census API key. ``censusdis`` reads the key
automatically from ``~/.censusdis/api_key.txt`` (or the ``CENSUS_API_KEY``
environment variable) -- see https://www.census.gov/data/developers.html for
how to request one.

Usage
-----
    uv run python scripts/build_census_snapshot.py

Output
------
- ``src/carto_flow/data/us_census_2020_state.parquet``
- ``src/carto_flow/data/us_census_2020_congressional_district.parquet``

Both files include every geography the loader can subset later (all states,
territories, and DC; all congressional districts, including placeholder
``"ZZ"`` districts) so that ``load_us_census(contiguous_only=False)` still
works. Geometries are projected to ESRI:102008 (Albers equal-area) and
coverage-simplified at 1000 m, then densified so no segment exceeds 5 km --
see ``carto_flow.geo_utils.simplify_coverage`` docs for why densification is
needed (the flow cartogram cannot bend a straight segment with no interior
vertices).

To regenerate with a different vintage or resolution, edit the constants
below and re-run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

VINTAGE = 2020
SIMPLIFY_TOLERANCE = 1000.0
MAX_SEGMENT_LENGTH = 5000.0
CRS = "ESRI:102008"

# All variables the loader can request, keyed by raw ACS variable code.
VARIABLES = {
    "NAME": None,  # human-readable name, handled specially below
    "B01003_001E": "Population",
    "B16009_001E": "Total Poverty",
    "B16009_002E": "Below Poverty Level",
    "B16009_015E": "Above Poverty Level",
    "B03002_001E": "Total Race",
    "B03002_003E": "White",
    "B03002_004E": "Black or African American",
    "B03002_006E": "Asian",
    "B03002_013E": "Hispanic or Latino",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "carto_flow" / "data"


def _build_level(level: str) -> None:
    import censusdis.data as ced
    import censusdis.states

    from carto_flow.geo_utils import simplify_coverage

    variable_codes = list(VARIABLES.keys())

    print(f"Downloading level={level!r} vintage={VINTAGE} ...")
    if level == "state":
        gdf = ced.download("acs/acs5", VINTAGE, variable_codes, state="*", with_geometry=True)
    elif level == "congressional_district":
        gdf = ced.download(
            "acs/acs5",
            VINTAGE,
            variable_codes,
            state="*",
            congressional_district="*",
            with_geometry=True,
        )
    else:
        raise ValueError(level)

    gdf = gdf.to_crs(CRS)

    # State abbreviation / name mappings computed once here so the runtime
    # loader never needs to import censusdis.
    gdf["State Abbreviation"] = gdf["STATE"].map(censusdis.states.ABBREVIATIONS_FROM_IDS)
    gdf["State Name"] = gdf["STATE"].map(censusdis.states.NAMES_FROM_IDS)

    print(f"  {len(gdf)} rows before simplification")

    # A handful of "ZZ" placeholder districts (non-geographic, e.g. water-only
    # remainders) come back with null geometry. They carry no shape to
    # simplify and load_us_census always drops CONGRESSIONAL_DISTRICT == "ZZ"
    # rows anyway, so drop them here rather than feeding nulls to
    # coverage_simplify (which cannot handle them).
    n_before = len(gdf)
    gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty].copy()
    if len(gdf) != n_before:
        print(f"  dropped {n_before - len(gdf)} rows with null/empty geometry")

    gdf = simplify_coverage(gdf, tolerance=SIMPLIFY_TOLERANCE, max_segment_length=MAX_SEGMENT_LENGTH)

    # Keep raw variable codes as column names (loader renames them), plus
    # STATE / CONGRESSIONAL_DISTRICT identifiers and the derived name columns.
    keep_cols = [*variable_codes, "STATE", "State Abbreviation", "State Name", "geometry"]
    if level == "congressional_district":
        keep_cols.insert(-3, "CONGRESSIONAL_DISTRICT")
    gdf = gdf[keep_cols]

    out_path = DATA_DIR / f"us_census_2020_{level}.parquet"
    gdf.to_parquet(out_path, compression="zstd")

    size_kb = out_path.stat().st_size / 1024
    print(f"  wrote {out_path} ({len(gdf)} rows, {size_kb:.1f} KiB)")


def main() -> None:
    start = time.perf_counter()
    for level in ("state", "congressional_district"):
        _build_level(level)
    print(f"Done in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
