"""Tests for carto_flow.geo_utils.adjacency."""

from shapely.geometry import GeometryCollection, LineString, box

from carto_flow.geo_utils.adjacency import find_adjacent_pairs


def test_adjacent_squares_are_found():
    pairs = find_adjacent_pairs([box(0, 0, 1, 1), box(1, 0, 2, 1), box(5, 5, 6, 6)])
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]


def test_geometry_collection_does_not_crash():
    # Clipping Voronoi cells to a boundary can leave a GeometryCollection with a
    # zero-area LineString part; shapely.boundary of a collection is None.
    gc = GeometryCollection([box(0, 0, 1, 1), LineString([(0, 1), (0, 2)])])
    pairs = find_adjacent_pairs([gc, box(1, 0, 2, 1)])
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]
