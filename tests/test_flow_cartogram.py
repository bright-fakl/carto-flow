"""Tests for flow cartogram module."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from carto_flow.flow_cartogram import MorphOptions, morph_gdf


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


if __name__ == "__main__":
    pytest.main([__file__])
