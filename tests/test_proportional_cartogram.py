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


class TestShrinkAreaTolerance:
    """`tol` bounds the relative area error of the shrunken part."""

    @pytest.fixture(scope="class")
    def states(self):
        from carto_flow.data import load_us_census

        gdf = load_us_census(population=True)
        return gdf.set_index("State Abbreviation").geometry

    @pytest.mark.parametrize(
        ("state", "fraction"),
        [("SD", 0.01), ("MT", 0.0063), ("ID", 0.018), ("NM", 0.015), ("OH", 0.25)],
    )
    def test_default_tol_bounds_area_error(self, states, state, fraction):
        geom = states[state]
        tol = 0.01
        core, shell = shrink(geom, fraction)
        assert abs(core.area / (fraction * geom.area) - 1) < tol
        assert core.area + shell.area == pytest.approx(geom.area, rel=1e-6)

    @pytest.mark.parametrize("tol", [0.05, 0.01, 1e-4])
    def test_tol_sets_the_achieved_error(self, states, tol):
        geom = states["SD"]
        core = shrink(geom, 0.01, tol=tol)[0]
        assert abs(core.area / (0.01 * geom.area) - 1) < tol

    def test_multipolygon_with_hole(self):
        geom = box(0, 0, 10, 10).difference(box(4, 4, 6, 6)).union(box(20, 0, 24, 4))
        core = shrink(geom, 0.02, tol=1e-4)[0]
        assert abs(core.area / (0.02 * geom.area) - 1) < 1e-4

    def test_fraction_bounds(self):
        square = box(0, 0, 10, 10)
        assert shrink(square, 1.0)[0].equals(square)
        assert shrink(square, 0.0)[0].is_empty
        with pytest.raises(ValueError, match="fraction"):
            shrink(square, 1.5)

    def test_tiny_fraction(self):
        square = box(0, 0, 10, 10)
        core = shrink(square, 1e-6, tol=1e-3)[0]
        assert abs(core.area / (1e-6 * square.area) - 1) < 1e-3
