"""Tests for carto_flow.geo_utils.adjacency.find_bordering_pairs."""

from shapely.geometry import Point, box

from carto_flow.geo_utils.adjacency import find_adjacent_pairs, find_bordering_pairs


def _pairs(geoms, **kw):
    return [(i, j) for i, j, _ in find_bordering_pairs(geoms, **kw)]


def test_shared_edge_is_a_border():
    pairs = find_bordering_pairs([box(0, 0, 1, 1), box(1, 0, 2, 1), box(5, 5, 6, 6)])
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]
    assert abs(pairs[0][2] - 1.0) < 0.05


def test_corner_contact_is_not_a_border():
    assert _pairs([box(0, 0, 1, 1), box(1, 1, 2, 2)]) == []


def test_narrow_gap_is_not_a_border_but_distance_tolerance_would_join_it():
    a, b = box(0, 0, 1, 1), box(1.02, 0, 2, 1)
    assert _pairs([a, b]) == []
    assert len(find_adjacent_pairs([a, b], distance_tolerance=0.05)) == 1


def test_short_shared_segment_is_below_threshold():
    big = box(0, 0, 10, 10)
    small = box(10, 4.99, 11, 5.0 + 0.01)  # 0.02 long shared edge, tile ~ 11
    assert _pairs([big, small, box(20, 20, 30, 30)]) == []
    assert _pairs([big, box(10, 3, 11, 7)]) == [(0, 1)]


def test_explicit_tile_size_and_sorted_output():
    geoms = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)]
    assert _pairs(geoms, tile_size=1.0) == [(0, 1), (1, 2)]
    assert _pairs(geoms, tile_size=1.0, min_length_fraction=2.0) == []


def test_degenerate_input():
    assert find_bordering_pairs([]) == []
    assert find_bordering_pairs([box(0, 0, 1, 1)]) == []
    assert find_bordering_pairs([Point(0, 0), Point(1, 1)]) == []
