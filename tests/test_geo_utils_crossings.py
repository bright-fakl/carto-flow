"""Crossing detection for rings (carto_flow.geo_utils.crossings)."""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon

from carto_flow.geo_utils import find_crossing_pairs, find_crossing_segments, unpack_geometries
from carto_flow.geo_utils.crossings import SegmentGrid, segment_starts


def _rings(*rings):
    """Flat coordinates and ring offsets of closed rings given as vertex lists."""
    coords = np.concatenate([np.asarray(r, float) for r in rings])
    offsets = np.concatenate([[0], np.cumsum([len(r) for r in rings])])
    return coords, offsets


def _geometry_rings(geometries):
    info = unpack_geometries(geometries)
    return info.coords, np.asarray(info.ragged_offsets[0], dtype=np.int64)


class TestSegments:
    def test_segment_starts_follow_the_rings(self):
        starts = segment_starts(np.array([0, 4, 9]))
        assert starts.tolist() == [0, 1, 2, 4, 5, 6, 7]

    def test_a_grid_needs_rings_or_starts(self):
        with pytest.raises(ValueError, match="ring_offsets or starts"):
            SegmentGrid(np.zeros((3, 2)))

    def test_coordinates_must_be_finite(self):
        with pytest.raises(ValueError, match="finite"):
            SegmentGrid(np.array([[0.0, 0.0], [np.nan, 1.0], [0.0, 0.0]]), np.array([0, 3]))


class TestCrossings:
    def test_bowtie_has_one_crossing_pair(self):
        coords, offsets = _rings([(0, 0), (4, 4), (4, 0), (0, 4), (0, 0)])
        assert find_crossing_segments(coords, offsets).tolist() == [True, False, True, False]
        assert find_crossing_pairs(coords, offsets).tolist() == [[0, 2]]

    def test_valid_square_has_no_crossing(self):
        coords, offsets = _rings([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        assert not find_crossing_segments(coords, offsets).any()
        assert find_crossing_pairs(coords, offsets).shape == (0, 2)

    def test_crossing_between_two_rings(self):
        a = [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]
        b = [(2, 2), (6, 2), (6, 6), (2, 6), (2, 2)]
        coords, offsets = _rings(a, b)
        pairs = find_crossing_pairs(coords, offsets)
        assert pairs.tolist() == [[1, 4], [2, 7]]

    def test_touching_and_shared_end_points_are_not_reported(self):
        # the second square touches the first in a corner and along a point of an edge
        a = [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]
        b = [(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)]
        c = [(1, 2), (1, 3), (0, 3), (1, 2)]
        coords, offsets = _rings(a, b, c)
        assert not find_crossing_segments(coords, offsets).any()

    def test_collinear_overlap_is_not_reported(self):
        a = [(0, 0), (3, 0), (3, 1), (0, 1), (0, 0)]
        b = [(1, 0), (4, 0), (4, -1), (1, -1), (1, 0)]
        coords, offsets = _rings(a, b)
        assert not find_crossing_segments(coords, offsets).any()

    def test_shared_boundary_with_identical_segments_is_not_reported(self):
        lower = Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (0, 1)])
        upper = Polygon([(0, 1), (1, 1), (2, 1), (2, 2), (0, 2)])
        coords, offsets = _geometry_rings([lower, upper])
        assert not find_crossing_segments(coords, offsets).any()
        # the same edge listed in the opposite direction by the two rings
        upper = Polygon([(2, 1), (1, 1), (0, 1), (0, 2), (2, 2)])
        coords, offsets = _geometry_rings([lower, upper])
        assert not find_crossing_segments(coords, offsets).any()

    def test_rings_with_holes(self):
        outer = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        hole = [(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]
        coords, offsets = _geometry_rings([Polygon(outer, [hole])])
        assert not find_crossing_segments(coords, offsets).any()
        # a hole that leaves the exterior crosses its boundary
        hole = [(8, 8), (8, 12), (12, 12), (12, 8), (8, 8)]
        coords, offsets = _rings(outer, hole)
        assert find_crossing_segments(coords, offsets).sum() == 4

    def test_multipolygon_parts(self):
        parts = MultiPolygon([Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])])
        coords, offsets = _geometry_rings([parts])
        assert len(offsets) == 3
        assert not find_crossing_segments(coords, offsets).any()
        overlapping = [(0.5, 0.5), (3, 0.5), (3, 1.5), (0.5, 1.5), (0.5, 0.5)]
        coords, offsets = _rings(*[r.exterior.coords[:] for r in parts.geoms], overlapping)
        assert find_crossing_segments(coords, offsets).any()

    def test_agrees_with_shapely_on_random_rings(self):
        rng = np.random.default_rng(3)
        for _ in range(60):
            n = int(rng.integers(4, 12))
            ring = rng.uniform(0, 10, (n, 2))
            ring = np.vstack([ring, ring[:1]])
            coords, offsets = _rings(ring)
            crossing = bool(find_crossing_segments(coords, offsets).any())
            simple = shapely.LinearRing(ring).is_simple
            assert crossing == (not simple)

    def test_pairs_match_brute_force(self):
        rng = np.random.default_rng(5)
        rings = []
        for _ in range(6):
            center = rng.uniform(0, 6, 2)
            angle = np.sort(rng.uniform(0, 2 * np.pi, 9))
            radius = rng.uniform(1, 3, 9)
            ring = center + np.c_[radius * np.cos(angle), radius * np.sin(angle)]
            rings.append(np.vstack([ring, ring[:1]]))
        coords, offsets = _rings(*rings)
        starts = segment_starts(offsets)
        lines = shapely.linestrings(np.stack([coords[starts], coords[starts + 1]], axis=1))
        expected = {
            (i, j) for i in range(len(lines)) for j in range(i + 1, len(lines)) if shapely.crosses(lines[i], lines[j])
        }
        assert {tuple(p) for p in find_crossing_pairs(coords, offsets).tolist()} == expected
        mask = find_crossing_segments(coords, offsets)
        assert set(np.flatnonzero(mask)) == {k for pair in expected for k in pair}


class TestQueries:
    def test_identical_segments_of_a_flagged_segment_are_included(self):
        lower = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
        upper = Polygon([(2, 1), (0, 1), (0, 2), (2, 2)])
        coords, offsets = _geometry_rings([lower, upper])
        grid = SegmentGrid(coords, offsets)
        flag = np.zeros(len(grid.starts), bool)
        flag[2] = True  # the edge (2, 1)-(0, 1) of the lower ring
        both = grid.with_identical(flag)
        assert np.flatnonzero(both).tolist() == [2, 4]

    def test_neighborhood_contains_the_flagged_segment_and_nearby_ones(self):
        coords, offsets = _rings(
            [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)], [(100, 100), (101, 100), (101, 101), (100, 101), (100, 100)]
        )
        grid = SegmentGrid(coords, offsets)
        flag = np.zeros(len(grid.starts), bool)
        flag[0] = True
        near = grid.neighborhood(flag)
        assert near[:4].all()
        assert not near[4:].any()
