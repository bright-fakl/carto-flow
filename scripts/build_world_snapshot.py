#!/usr/bin/env python3
"""Build the bundled world countries snapshot used by ``carto_flow.data.load_world``.

Source data
-----------
Natural Earth 1:110m Admin 0 - Countries (public domain,
https://www.naturalearthdata.com/), as redistributed by GeoPandas <= 0.14 under
``geopandas.datasets.get_path("naturalearth_lowres")``. That dataset was removed
from GeoPandas in 1.0, so the copy under ``src/carto_flow/data/world.parquet``
is the archived source: 177 rows in EPSG:4326 with columns ``name``,
``continent``, ``pop_est``, ``gdp_md_est`` and ``geometry``.

What this script does
---------------------
It repairs the geometries so the dataset is valid as shipped and stays valid
when reprojected, and rewrites the parquet in place. Three steps, in order:

1. ``set_precision(1e-7)`` -- snaps coordinates to a ~1.1 cm grid, removing
   near-duplicate vertices that are distinct in degrees but collapse into
   degenerate spikes once scaled to metres (Mozambique had three vertices
   within 0.5 m of each other).
2. ``make_valid`` -- resolves self-intersections that are present in the source
   regardless of CRS (Sudan's exterior ring doubles back on itself, enclosing a
   zero-area spike).
3. ``segmentize(1.0)`` (degrees) -- densifies long straight edges. Reprojection
   maps vertices, not edges, so a long chord in EPSG:4326 can cut across its
   own polygon under a curved projection. Natural Earth clips rings at the
   antimeridian with a single straight closing edge, and in Mollweide that edge
   crosses Chukotka's coastline, making Russia self-intersecting. Densifying is
   exact in EPSG:4326 -- it only inserts collinear vertices -- but it does
   change reprojected areas, because the densified outline follows the
   projected curve instead of cutting the chord.

The operations are idempotent: running this script on an already-repaired file
leaves the geometries unchanged.

Usage
-----
    uv run python scripts/build_world_snapshot.py [--source PATH] [--check]

``--source`` reads from somewhere other than the bundled file (for rebuilding
from a freshly downloaded Natural Earth extract). ``--check`` reports what the
repair would change without writing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PRECISION_GRID = 1e-7  # degrees, ~1.1 cm
MAX_SEGMENT_LENGTH = 1.0  # degrees

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "carto_flow" / "data"
OUT_PATH = DATA_DIR / "world.parquet"


def repair(geometry):
    """Apply the repair pipeline to an array of EPSG:4326 geometries."""
    import shapely

    geometry = shapely.set_precision(geometry, PRECISION_GRID)
    geometry = shapely.make_valid(geometry)
    return shapely.segmentize(geometry, MAX_SEGMENT_LENGTH)


def main(argv: list[str] | None = None) -> int:
    import geopandas as gpd
    import shapely

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=OUT_PATH, help="source GeoParquet (default: the bundled file)")
    parser.add_argument("--check", action="store_true", help="report changes without writing")
    args = parser.parse_args(argv)

    gdf = gpd.read_parquet(args.source)
    before = gdf.geometry.values
    after = repair(before)

    invalid_before = [n for n, g in zip(gdf["name"], before, strict=True) if not shapely.is_valid(g)]
    invalid_after = [n for n, g in zip(gdf["name"], after, strict=True) if not shapely.is_valid(g)]
    print(f"read {args.source} ({len(gdf)} rows)")
    print(f"  invalid before: {invalid_before or 'none'}")
    print(f"  invalid after:  {invalid_after or 'none'}")
    print(f"  vertices: {shapely.get_num_coordinates(before).sum()} -> {shapely.get_num_coordinates(after).sum()}")
    changed_type = [
        (n, a.geom_type, b.geom_type)
        for n, a, b in zip(gdf["name"], before, after, strict=True)
        if a.geom_type != b.geom_type
    ]
    print(f"  geometry type changes: {changed_type or 'none'}")

    if args.check:
        return 0

    gdf = gdf.copy()
    gdf["geometry"] = after
    gdf.to_parquet(OUT_PATH, compression="zstd")
    print(f"  wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
