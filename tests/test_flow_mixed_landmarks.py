"""Tests for landmarks of mixed geometry types in the flow cartogram."""

import geopandas as gpd
import pytest
import shapely
from shapely.geometry import LineString, MultiPoint, Point, Polygon

import carto_flow.data as data
from carto_flow.flow_cartogram import MorphOptions, create_flow_cartogram
from carto_flow.geo_utils.geometry import GeometryCoordinates


@pytest.fixture(scope="module")
def states():
    return data.load_us_census(population=True).reset_index(drop=True)


def _mixed(gdf):
    c = gdf.geometry.iloc[0].representative_point()
    x, y = c.x, c.y
    ring = Polygon([(x - 1e4, y - 1e4), (x + 1e4, y - 1e4), (x + 1e4, y + 1e4), (x - 1e4, y + 1e4)])
    hole = Polygon(
        ring.exterior.coords,
        [[(x - 3e3, y - 3e3), (x + 3e3, y - 3e3), (x + 3e3, y + 3e3), (x - 3e3, y + 3e3)]],
    )
    return [
        Point(x, y),
        LineString([(x, y), (x + 1e4, y + 1e4)]),
        hole,
        MultiPoint([(x, y), (x + 5e3, y)]),
        Point(),
    ]


def test_reconstruct_keeps_structure():
    geoms = [Point(1, 2, 3), Polygon([(0, 0), (4, 0), (4, 4), (0, 4)], [[(1, 1), (2, 1), (2, 2)]]), Point()]
    flat = GeometryCoordinates(geoms)
    assert flat.coords.shape == (1 + 5 + 4, 2)
    flat.coords = flat.coords + 1.0
    out = flat.reconstruct()
    assert [g.geom_type for g in out] == ["Point", "Polygon", "Point"]
    assert out[2].is_empty
    assert len(out[1].interiors) == 1
    assert (out[0].x, out[0].y) == (2.0, 3.0)
    assert not out[0].has_z
    assert geoms[0].has_z  # input untouched


def test_mixed_landmarks_through_create_flow_cartogram(states):
    landmarks = _mixed(states)
    lm = gpd.GeoDataFrame(geometry=landmarks, crs=states.crs)
    result = create_flow_cartogram(
        states,
        "Population",
        landmarks=lm,
        options=MorphOptions(grid=64, n_iter=30, show_progress=False),
    )
    out = result.get_landmarks()
    assert [g.geom_type for g in out] == [g.geom_type for g in landmarks]
    assert len(out[2].interiors) == 1
    assert out[4].is_empty
    assert not shapely.equals(out[0], landmarks[0])
