"""Tests for carto_flow.geo_utils.adjacency."""

from shapely.geometry import GeometryCollection, LineString, Point, box

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


def test_point_cell_is_adjacent_to_its_neighbours():
    # A cell that degraded to a Point has an EMPTY boundary (not None), so the
    # shared-length computation must fall back to the buffered intersection.
    squares = [box(0, 0, 1, 1), box(1, 0, 2, 1)]
    point = Point(1.0, 0.5)
    pairs = find_adjacent_pairs([*squares, point], distance_tolerance=0.1)
    assert (0, 2) in [(i, j) for i, j, _ in pairs]
    assert (1, 2) in [(i, j) for i, j, _ in pairs]
    assert all(length > 0 for _, _, length in pairs)
