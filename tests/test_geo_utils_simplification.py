"""Tests for carto_flow.geo_utils.simplification (simplify_coverage, densify_coverage)."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from carto_flow.geo_utils import densify_coverage, simplify_coverage


def _ring_segment_lengths(geom):
    """Return an array of segment lengths for all rings in geom."""
    polys = geom.geoms if hasattr(geom, "geoms") else [geom]
    lengths = []
    for poly in polys:
        for ring in (poly.exterior, *poly.interiors):
            coords = np.asarray(ring.coords)
            deltas = np.diff(coords, axis=0)
            lengths.append(np.hypot(deltas[:, 0], deltas[:, 1]))
    return np.concatenate(lengths)


def _two_squares_gdf():
    return gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)])


def test_densify_coverage_respects_max_segment_length():
    gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)])
    densified = densify_coverage(gdf, max_segment_length=2.0)
    lengths = _ring_segment_lengths(densified.geometry.iloc[0])
    assert lengths.max() <= 2.0 + 1e-9


def test_densify_coverage_keeps_shared_edge_identical():
    gdf = _two_squares_gdf()
    densified = densify_coverage(gdf, max_segment_length=2.0)

    left, right = densified.geometry
    # Shared edge is x=10, from y=0 to y=10.
    left_shared = {c for c in left.exterior.coords if c[0] == 10}
    right_shared = {c for c in right.exterior.coords if c[0] == 10}
    assert left_shared == right_shared


def test_densify_coverage_preserves_crs_and_columns():
    gdf = _two_squares_gdf()
    gdf["value"] = [1, 2]
    gdf = gdf.set_crs("EPSG:3857")

    densified = densify_coverage(gdf, max_segment_length=2.0)
    assert densified.crs == gdf.crs
    assert list(densified["value"]) == [1, 2]


def test_simplify_coverage_with_max_segment_length_composes():
    gdf = _two_squares_gdf()
    result = simplify_coverage(gdf, tolerance=0.01, max_segment_length=1.0)
    for geom in result.geometry:
        lengths = _ring_segment_lengths(geom)
        assert lengths.max() <= 1.0 + 1e-9


def test_simplify_coverage_without_max_segment_length_unchanged_behavior():
    gdf = _two_squares_gdf()
    result = simplify_coverage(gdf, tolerance=0.01)
    # No densification requested: squares stay coarse (4 vertices per ring).
    for geom in result.geometry:
        assert len(geom.exterior.coords) <= 6


@pytest.mark.parametrize("max_segment_length", [None, 5.0])
def test_simplify_coverage_preserves_union(max_segment_length):
    gdf = _two_squares_gdf()
    result = simplify_coverage(gdf, tolerance=0.01, max_segment_length=max_segment_length)
    assert result.union_all().equals(gdf.union_all())
