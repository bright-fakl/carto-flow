"""Tests for proportional cartogram module."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from carto_flow.proportional_cartogram import (
    generate_dot_density,
    partition_geometries,
    shrink,
    split,
)


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


class TestProportionalCartogram:
    """Tests for proportional cartogram functionality."""

    def test_basic_split(self, gdf):
        """Test basic splitting functionality."""
        # Test with a single geometry
        geom = gdf.geometry.iloc[0]
        parts = split(geom, fractions=[0.3, 0.3, 0.4])
        assert len(parts) == 3
        # Check that the sum of areas is approximately the original area
        total_area = sum(part.area for part in parts)
        assert total_area == pytest.approx(geom.area, rel=1e-6)

    def test_basic_shrink(self, gdf):
        """Test basic shrinking functionality."""
        geom = gdf.geometry.iloc[0]
        shells = shrink(geom, fractions=[0.3, 0.3, 0.4])
        assert len(shells) == 3

    def test_partition_geometries(self, gdf):
        """Test partitioning geometries in a GeoDataFrame."""
        result = partition_geometries(
            gdf,
            columns=["population"],
            method="split",
            normalization="sum",
        )
        assert result is not None
        assert len(result) > 0

    def test_dot_density(self, gdf):
        """Test dot density generation."""
        result = generate_dot_density(
            gdf,
            "population",
            n_dots=100,
        )
        assert result is not None
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__])
