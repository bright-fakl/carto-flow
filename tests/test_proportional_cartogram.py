"""Tests for proportional cartogram module."""

import math

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


class TestShrinkIsotropic:
    """`isotropic=True` keeps the proportions of an elongated geometry."""

    @staticmethod
    def _extent(geom):
        xmin, ymin, xmax, ymax = geom.bounds
        return xmax - xmin, ymax - ymin

    def test_rectangle_keeps_its_proportions(self):
        from shapely.geometry import box

        rectangle = box(0, 0, 10, 1)
        plain = shrink(rectangle, 0.2)[0]
        isotropic = shrink(rectangle, 0.2, isotropic=True)[0]

        width, height = self._extent(isotropic)
        assert width / height == pytest.approx(10.0, rel=0.05)
        assert self._extent(plain)[0] / self._extent(plain)[1] > 30
        assert isotropic.area == pytest.approx(2.0, rel=0.01)

    def test_rotated_rectangle(self):
        from shapely import affinity
        from shapely.geometry import box

        rectangle = affinity.rotate(box(0, 0, 10, 1), 30, origin=(0, 0))
        core = shrink(rectangle, 0.2, isotropic=True)[0]
        corners = list(core.minimum_rotated_rectangle.exterior.coords)
        sides = sorted(math.dist(corners[i], corners[i + 1]) for i in range(2))
        assert sides[1] / sides[0] == pytest.approx(10.0, rel=0.05)
        assert rectangle.buffer(1e-9).contains(core)

    def test_outer_boundary_is_unchanged(self):
        from shapely.geometry import Polygon

        polygon = Polygon([(0, 0), (12, 0), (12, 2), (7, 3), (0, 2)])
        core, shell = shrink(polygon, 0.3, isotropic=True)
        assert polygon.buffer(1e-9).contains(core)
        assert shell.union(core).symmetric_difference(polygon).area < 1e-9 * polygon.area
        assert core.area / polygon.area == pytest.approx(0.3, abs=0.01 * 0.3)

    def test_states_stay_inside_with_target_area(self):
        from carto_flow.data import load_us_census

        states = load_us_census(population=True).set_index("State Abbreviation").geometry
        for state in ("WY", "TN", "OK", "FL"):
            geom = states[state]
            core = shrink(geom, 0.05, isotropic=True)[0]
            assert abs(core.area / (0.05 * geom.area) - 1) < 0.01
            assert geom.buffer(1e-6 * geom.length).contains(core)

    def test_large_coordinates_do_not_change_the_result(self):
        from shapely import affinity
        from shapely.geometry import box

        near = box(0, 0, 1000, 100)
        far = affinity.translate(near, 1e8, 1e8)
        core_near = shrink(near, 0.2, isotropic=True)[0]
        core_far = shrink(far, 0.2, isotropic=True)[0]
        moved_back = affinity.translate(core_far, -1e8, -1e8)
        assert moved_back.symmetric_difference(core_near).area < 1e-6 * core_near.area

    def test_multiple_shells(self):
        from shapely.geometry import box

        parts = shrink(box(0, 0, 10, 1), [0.25, 0.25, 0.5], isotropic=True)
        assert [round(p.area, 1) for p in parts] == [2.5, 2.5, 5.0]

    def test_default_is_unchanged(self):
        from shapely.geometry import box

        rectangle = box(0, 0, 10, 1)
        assert shrink(rectangle, 0.2)[0].equals(shrink(rectangle, 0.2, isotropic=False)[0])

    def test_degenerate_geometry_falls_back(self):
        from shapely.geometry import Polygon

        sliver = Polygon([(0, 0), (1000, 0), (1000, 1e-9), (0, 1e-9)])
        assert shrink(sliver, 0.5, isotropic=True)[0].area >= 0.0


class TestPartitionIsotropic:
    def _frame(self):
        return gpd.GeoDataFrame({"share": [0.2, 0.2]}, geometry=[box(0, 0, 10, 1), box(0, 5, 10, 6)])

    def test_isotropic_is_passed_to_shrink(self):
        gdf = self._frame()
        plain = partition_geometries(gdf, "share", method="shrink")
        isotropic = partition_geometries(gdf, "share", method="shrink", isotropic=True)

        def aspect(geom):
            xmin, ymin, xmax, ymax = geom.bounds
            return (xmax - xmin) / (ymax - ymin)

        assert aspect(plain["geometry_share"].iloc[0]) > 30
        assert aspect(isotropic["geometry_share"].iloc[0]) == pytest.approx(10.0, rel=0.05)
        expected = shrink(box(0, 0, 10, 1), 0.2, isotropic=True)[0]
        assert isotropic["geometry_share"].iloc[0].equals_exact(expected, 1e-9)

    def test_split_ignores_isotropic(self):
        gdf = self._frame()
        a = partition_geometries(gdf, "share", method="split")
        b = partition_geometries(gdf, "share", method="split", isotropic=True)
        assert a["geometry_share"].iloc[0].equals(b["geometry_share"].iloc[0])
