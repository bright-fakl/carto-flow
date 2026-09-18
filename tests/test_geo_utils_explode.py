"""Tests for carto_flow.geo_utils.explode."""

import geopandas as gpd
import pytest
from shapely.geometry import box

from carto_flow.geo_utils.explode import explode_geodataframe


def test_touching_squares_are_separated():
    # Two unit squares sharing an edge at x=1.
    gdf = gpd.GeoDataFrame(
        {"name": ["a", "b"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:3857",
    )
    distance = 0.5
    areas_before = gdf.area.to_numpy()

    result = explode_geodataframe(gdf, distance=distance, max_iter=500)

    gap = result.geometry.iloc[0].distance(result.geometry.iloc[1])
    assert gap >= distance - 1e-2

    areas_after = result.area.to_numpy()
    assert areas_after == pytest.approx(areas_before, rel=1e-9)


def test_far_apart_geometries_essentially_unchanged():
    gdf = gpd.GeoDataFrame(
        {"name": ["a", "b"]},
        geometry=[box(0, 0, 1, 1), box(1000, 1000, 1001, 1001)],
        crs="EPSG:3857",
    )
    result = explode_geodataframe(gdf, distance=1.0, max_iter=200)

    for orig, moved in zip(gdf.geometry, result.geometry, strict=False):
        cx0, cy0 = orig.centroid.x, orig.centroid.y
        cx1, cy1 = moved.centroid.x, moved.centroid.y
        assert abs(cx1 - cx0) < 1e-6
        assert abs(cy1 - cy0) < 1e-6


def test_preserves_index_crs_and_columns():
    gdf = gpd.GeoDataFrame(
        {"name": ["a", "b"], "value": [1, 2]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:3857",
        index=[10, 20],
    )
    result = explode_geodataframe(gdf, distance=0.5, max_iter=100)

    assert list(result.index) == [10, 20]
    assert result.crs == gdf.crs
    assert list(result["name"]) == ["a", "b"]
    assert list(result["value"]) == [1, 2]


def test_invalid_group_by_column_raises():
    gdf = gpd.GeoDataFrame(
        {"name": ["a", "b"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:3857",
    )
    with pytest.raises(KeyError):
        explode_geodataframe(gdf, distance=0.5, group_by="does_not_exist")
