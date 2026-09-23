"""Geometry validity of the bundled datasets, as loaded and after reprojection.

A dataset that is valid in its own CRS can still become self-intersecting once
reprojected, because reprojection maps vertices and leaves the edges between
them straight. Both properties are checked here for every bundled dataset.

``ESRI:102003`` is deliberately not among the target CRSs: it is a US-centred
Albers projection, and applying it worldwide puts geometries far outside its
domain of validity, where self-intersections are a property of the projection
rather than of the data.
"""

import pytest
import shapely

# Equal-area CRSs with worldwide domains.
EQUAL_AREA_CRS = ["ESRI:54009", "EPSG:6933"]


def _load(name):
    import carto_flow.data as data

    loaders = {
        "world": data.load_world,
        "us_states": data.load_us_states,
        "cities": data.load_sample_cities,
        "census_state": lambda: data.load_us_census(level="state"),
        "census_state_all": lambda: data.load_us_census(level="state", contiguous_only=False),
        "census_state_simplified": lambda: data.load_us_census(level="state", simplify=2000),
        "census_district": lambda: data.load_us_census(level="congressional_district"),
        "census_district_all": lambda: data.load_us_census(level="congressional_district", contiguous_only=False),
        "census_district_simplified": lambda: data.load_us_census(level="congressional_district", simplify=2000),
    }
    return loaders[name]()


DATASETS = [
    "world",
    "us_states",
    "cities",
    "census_state",
    "census_state_all",
    "census_state_simplified",
    "census_district",
    "census_district_all",
    "census_district_simplified",
]


def _invalid(gdf):
    """Return ``(index, reason)`` for every invalid geometry in ``gdf``."""
    return [(i, shapely.is_valid_reason(g)) for i, g in gdf.geometry.items() if not shapely.is_valid(g)]


@pytest.mark.parametrize("dataset", DATASETS)
def test_bundled_dataset_valid_as_loaded(dataset):
    gdf = _load(dataset)
    assert _invalid(gdf) == []


@pytest.mark.parametrize("crs", EQUAL_AREA_CRS)
@pytest.mark.parametrize("dataset", DATASETS)
def test_bundled_dataset_valid_after_equal_area_reprojection(dataset, crs):
    gdf = _load(dataset).to_crs(crs)
    assert _invalid(gdf) == []


def test_world_union_after_equal_area_reprojection():
    """A study union over the reprojected world must not hit a GEOS topology error."""
    gdf = _load("world")
    gdf = gdf[(gdf["name"] != "Antarctica") & (gdf["pop_est"] > 0)].to_crs("ESRI:54009")
    union = gdf.geometry.union_all()
    assert union.is_valid
    assert union.area > 0


def test_world_repair_is_idempotent():
    """The shipped file is already repaired, so re-running the repair changes no shape.

    ``set_precision`` normalizes its output, so a second pass can reorder rings;
    the geometries are compared normalized for that reason.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    try:
        from build_world_snapshot import repair
    finally:
        sys.path.pop(0)

    gdf = _load("world")
    repaired = repair(gdf.geometry.values)
    for original, again in zip(gdf.geometry.values, repaired, strict=True):
        assert shapely.normalize(original).equals_exact(shapely.normalize(again), 0.0)
