"""Grid and mosaic layouts on the US states and the congressional districts, against the tactile-based tilings.

``tests/data/layouts.json`` holds the placement metrics of the layouts with the tilings of carto-flow 2.0. The layouts
change on purpose with the tilings (the lattice is anchored at the center of the region and the tiles are listed in
another order), so the tile assignments are not compared. What is compared:

* every region receives its tiles (49 states with one tile each, 154 tiles for the states in the mosaic layout, one tile
  per district grouped by state),
* at most 3 regions are split over more than one block of tiles,
* the mean distance between the centroid of a region and the centroid of its tiles, in tile sizes, is within the larger
  of 25 percent and the range that the same code gives when only the position of the lattice relative to the data
  changes (``phase_range``, six shifts of the lattice for the old and the new code).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import shapely
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components

from carto_flow.data import load_us_census
from carto_flow.symbol_cartogram import create_layout
from carto_flow.symbol_cartogram.layouts.grid import GridBasedLayout
from carto_flow.symbol_cartogram.layouts.mosaic import MosaicLayout

FIXTURE = json.loads((Path(__file__).parent / "data" / "layouts.json").read_text())

# name, input, tiling, layout, tile count column, group_by. The district grid layouts take minutes and are not run here.
CASES = [
    ("states grid hexagon", "states", "hexagon", "grid", None, None),
    ("states grid square", "states", "square", "grid", None, None),
    ("states grid triangle", "states", "triangle", "grid", None, None),
    ("states grid quadrilateral", "states", "quadrilateral", "grid", None, None),
    ("states grid scalloped_hexagon", "states", "scalloped_hexagon", "grid", None, None),
    ("states mosaic hexagon", "states", "hexagon", "mosaic", "tiles", None),
    ("states mosaic square", "states", "square", "mosaic", "tiles", None),
    ("states mosaic triangle", "states", "triangle", "mosaic", "tiles", None),
    ("states mosaic scalloped_hexagon", "states", "scalloped_hexagon", "mosaic", "tiles", None),
    ("districts mosaic hexagon", "districts", "hexagon", "mosaic", None, "State Name"),
]


@pytest.fixture(scope="module")
def inputs():
    states = load_us_census(population=True).reset_index(drop=True)
    population = states["Population"].to_numpy(dtype=float)
    states["tiles"] = np.maximum(1, np.round(population / population.sum() * 150)).astype(int)
    districts = load_us_census(population=True, level="congressional_district").reset_index(drop=True)
    return {"states": states, "districts": districts}


def _metrics(gdf, result) -> dict:
    tiling_result = result.tiling_result
    tiles = np.asarray(result.assignments, dtype=np.intp)
    source = result.source_indices
    regions = np.asarray(source, dtype=np.intp) if source is not None else np.arange(len(tiles), dtype=np.intp)
    per_region: dict[int, list[int]] = {}
    for tile, region in zip(tiles.tolist(), regions.tolist(), strict=True):
        per_region.setdefault(region, []).append(tile)
    adjacency = csr_array(np.asarray(tiling_result.adjacency))
    split = 0
    distance = []
    for region, region_tiles in per_region.items():
        if len(region_tiles) > 1:
            sub = adjacency[region_tiles][:, region_tiles]
            split += connected_components(sub, directed=False)[0] > 1
        footprint = shapely.union_all([tiling_result.polygons[t] for t in region_tiles]).centroid
        centroid = gdf.geometry.iloc[region].centroid
        distance.append(np.hypot(footprint.x - centroid.x, footprint.y - centroid.y) / tiling_result.tile_size)
    return {
        "assigned": len(tiles),
        "regions": len(per_region),
        "split": int(split),
        "disp_mean": float(np.mean(distance)),
        "disp_max": float(np.max(distance)),
    }


@pytest.mark.parametrize(
    ("name", "source", "tiling", "layout", "tile_count", "group_by"), CASES, ids=[c[0] for c in CASES]
)
def test_layout_metrics_stay_within_the_effect_of_the_lattice_position(
    inputs, name, source, tiling, layout, tile_count, group_by
):
    gdf = inputs[source]
    algorithm = GridBasedLayout(tiling=tiling) if layout == "grid" else MosaicLayout(tiling=tiling, morph=False)
    result = create_layout(gdf, tile_count=tile_count, group_by=group_by, layout=algorithm, show_progress=False)
    expected = FIXTURE["cases"][name]
    got = _metrics(gdf, result)

    assert got["assigned"] == expected["assigned"]
    assert got["regions"] == expected["regions"] == len(gdf)
    assert got["split"] <= 3

    spread = 0.0
    if layout == "grid" and tiling in FIXTURE["phase_range"]:
        low, high = FIXTURE["phase_range"][tiling]
        spread = high - low
    tolerance = max(0.25 * expected["disp_mean"], spread)
    assert abs(got["disp_mean"] - expected["disp_mean"]) <= tolerance, (
        got["disp_mean"],
        expected["disp_mean"],
        tolerance,
    )
