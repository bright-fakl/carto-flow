"""Tests for flow cartogram module."""

import warnings

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from carto_flow.flow_cartogram import MorphOptions, morph_gdf, morph_geometries
from carto_flow.geo_utils import densify_coverage


def make_grid_gdf(rows: int = 3, cols: int = 3, seed: int = 42) -> gpd.GeoDataFrame:
    """Create a grid of adjacent squares with population data."""
    rng = np.random.default_rng(seed)
    geoms = []
    for r in range(rows):
        for c in range(cols):
            geoms.append(box(c, r, c + 1, r + 1))
    n = rows * cols
    return gpd.GeoDataFrame(
        {"population": rng.integers(100, 10000, size=n).astype(float)},
        geometry=geoms,
    )


@pytest.fixture
def gdf():
    return make_grid_gdf()


class TestLongSegmentWarning:
    """morph_geometries warns about long straight segments (Phase 0.5)."""

    def test_warns_for_coarse_square(self):
        # A single large square has 1000-unit edges. Default grid_size=256
        # over these bounds gives cells far smaller than 1000/4, so the
        # warning should fire.
        square = [box(0, 0, 1000, 1000)]
        values = [100]
        options = MorphOptions(n_iter=1, recompute_every=1)

        with pytest.warns(UserWarning, match="straight segment"):
            morph_geometries(square, values, options=options)

    def test_no_warning_for_densified_square(self):
        gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1000, 1000)])
        densified = densify_coverage(gdf, max_segment_length=10.0)
        values = [100]
        options = MorphOptions(n_iter=1, recompute_every=1)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            morph_geometries(list(densified.geometry), values, options=options)


class TestFlowCartogram:
    """Tests for flow cartogram functionality."""

    def test_basic_flow_cartogram(self, gdf):
        """Test creating a basic flow cartogram."""
        cartogram = morph_gdf(gdf, "population")
        assert cartogram is not None
        assert len(cartogram.get_geometry()) == len(gdf)

    def test_flow_cartogram_with_parameters(self, gdf):
        """Test flow cartogram with custom parameters."""
        options = MorphOptions(
            grid_size=64,
            n_iter=100,
            mean_tol=1e-3,
        )
        cartogram = morph_gdf(gdf, "population", options=options)
        assert cartogram is not None
        assert len(cartogram.get_geometry()) == len(gdf)

    def test_flow_cartogram_different_grid_sizes(self, gdf):
        """Test flow cartogram with different grid sizes."""
        for grid_size in [32, 64, 128]:
            options = MorphOptions(grid_size=grid_size)
            cartogram = morph_gdf(gdf, "population", options=options)
            assert cartogram is not None
            assert len(cartogram.get_geometry()) == len(gdf)

    def test_flow_cartogram_convergence(self, gdf):
        """Test flow cartogram convergence behavior."""
        options = MorphOptions(
            n_iter=10,
            mean_tol=1e-1,
        )
        cartogram = morph_gdf(gdf, "population", options=options)
        assert cartogram is not None
        assert len(cartogram.get_geometry()) == len(gdf)


class TestZeroValueGeometry:
    """A zero sizing value must not break convergence for the whole morph."""

    def test_zero_value_does_not_emit_divide_by_zero_warnings(self, gdf):
        gdf = gdf.copy()
        gdf.loc[gdf.index[0], "population"] = 0.0
        options = MorphOptions(n_iter=20, grid_size=32)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            cartogram = morph_gdf(gdf, "population", options=options)

        assert cartogram is not None

    def test_zero_value_region_shrinks_and_convergence_is_reachable(self, gdf):
        """The zero-value region's area shrinks, and mean/max error stay finite."""
        gdf = gdf.copy()
        zero_idx = gdf.index[0]
        gdf.loc[zero_idx, "population"] = 0.0
        options = MorphOptions(n_iter=100, grid_size=64, mean_tol=1e-2, max_tol=1e-1)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            cartogram = morph_gdf(gdf, "population", options=options)

        final = cartogram.latest
        errors = final.errors
        assert np.isfinite(errors.mean_log_error)
        assert np.isfinite(errors.max_log_error)

        original_area = gdf.geometry.iloc[0].area
        final_area = final.geometry[0].area
        assert final_area < original_area

    def test_collapsed_current_area_does_not_diverge(self):
        """Mirror case: a region whose *current* area collapses toward zero.

        A tiny sliver of positive value next to a much larger neighbour has
        current area << target area early in the morph; this must not blow
        up the reported error to -inf.
        """
        geoms = [box(0, 0, 1, 1), box(1, 0, 1.001, 1)]
        values = [1.0, 1000.0]
        options = MorphOptions(n_iter=5, recompute_every=1, grid_size=32, show_progress=False)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            cartogram = morph_geometries(geoms, values, options=options)

        for snapshot in cartogram.snapshots:
            assert np.all(np.isfinite(snapshot.errors.log_errors))


if __name__ == "__main__":
    pytest.main([__file__])
