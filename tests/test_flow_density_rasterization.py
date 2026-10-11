"""Exact (area-weighted) density rasterization of the flow cartogram."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from carto_flow import data
from carto_flow.flow_cartogram import MorphOptions, MorphStatus, morph_gdf
from carto_flow.flow_cartogram.density import compute_density_field_from_geometries
from carto_flow.flow_cartogram.grid import Grid


def _grid(n=16, bounds=(0.0, 0.0, 16.0, 16.0)):
    return Grid(bounds, size=n)


def _reference_coverage(geom, grid):
    """Fraction of every cell covered by ``geom``, by clipping the cell with the geometry."""
    xe, ye = grid.x_edges, grid.y_edges
    out = np.zeros(grid.X.shape)
    for i in range(len(ye) - 1):
        for j in range(len(xe) - 1):
            cell = box(xe[j], ye[i], xe[j + 1], ye[i + 1])
            out[i, j] = geom.intersection(cell).area / cell.area
    return out


class TestExactCoverage:
    @pytest.mark.parametrize(
        "geom",
        [
            box(2.3, 3.1, 9.7, 6.4),
            Polygon([(1.2, 1.1), (13.7, 2.9), (7.4, 12.6)]),
            Polygon([(2, 2), (12, 2), (12, 12), (2, 12)], holes=[[(5.5, 5.5), (9.1, 5.5), (9.1, 8.2), (5.5, 8.2)]]),
            Polygon([(0.3, 0.2), (15.8, 0.4), (15.6, 15.5), (8.0, 9.9), (0.5, 15.7)]),
        ],
        ids=["rectangle", "triangle", "hole", "concave"],
    )
    def test_matches_clipping_every_cell(self, geom):
        grid = _grid()
        value = 7.0
        rho = compute_density_field_from_geometries([geom], [value], grid, mean_density=0.0, rasterization="exact")
        expected = _reference_coverage(geom, grid) * value / geom.area
        np.testing.assert_allclose(rho, expected, atol=1e-12)

    def test_integral_of_the_density_is_the_total_value(self):
        grid = _grid(24, (0.0, 0.0, 24.0, 24.0))
        geoms = [box(1.2, 1.7, 8.9, 11.3), Polygon([(10, 2), (22.5, 3.4), (15.2, 20.1)]), box(2.2, 13.1, 9.4, 22.6)]
        values = [3.0, 5.0, 2.0]
        rho = compute_density_field_from_geometries(geoms, values, grid, mean_density=0.0, rasterization="exact")
        assert rho.sum() * grid.dx * grid.dy == pytest.approx(sum(values), rel=1e-12)

    def test_region_smaller_than_a_cell_takes_part(self):
        """A region with no cell center inside is invisible to center sampling and counted by exact coverage."""
        grid = _grid()
        tiny = box(3.1, 3.1, 3.4, 3.4)  # cell centers are at x.5
        centre = compute_density_field_from_geometries([tiny], [1.0], grid, mean_density=0.0)
        exact = compute_density_field_from_geometries([tiny], [1.0], grid, mean_density=0.0, rasterization="exact")
        assert centre.sum() == 0.0
        assert exact.sum() * grid.dx * grid.dy == pytest.approx(1.0, rel=1e-12)

    def test_uncovered_part_gets_the_mean_density(self):
        grid = _grid()
        geom = box(0.0, 0.0, 8.0, 16.0)
        rho = compute_density_field_from_geometries([geom], [8.0], grid, mean_density=2.0, rasterization="exact")
        assert rho[:, :8] == pytest.approx(8.0 / geom.area)
        assert rho[:, 8:] == pytest.approx(2.0)

    def test_overlapping_geometries_do_not_lower_the_mean_density_term(self):
        grid = _grid()
        a, b = box(2, 2, 10, 10), box(6, 6, 14, 14)
        rho = compute_density_field_from_geometries([a, b], [4.0, 4.0], grid, mean_density=1.0, rasterization="exact")
        assert np.all(rho >= 0.0)

    def test_geometry_mask_marks_the_dominant_geometry(self):
        grid = _grid()
        a, b = box(0, 0, 8, 16), box(8, 0, 16, 16)
        _, mask = compute_density_field_from_geometries(
            [a, b], [1.0, 1.0], grid, 1.0, return_geometry_mask=True, rasterization="exact"
        )
        assert (mask[:, :8] == 0).all() and (mask[:, 8:] == 1).all()

    def test_rejects_an_unknown_mode(self):
        with pytest.raises(ValueError, match="rasterization"):
            compute_density_field_from_geometries([box(0, 0, 1, 1)], [1.0], _grid(), 1.0, rasterization="fast")


class TestOption:
    def test_default_is_center(self):
        assert MorphOptions().density_rasterization == "center"

    @pytest.mark.parametrize("value", ["fast", None, True, 2])
    def test_invalid_value_is_rejected(self, value):
        with pytest.raises(ValueError, match="density_rasterization"):
            MorphOptions(density_rasterization=value)

    def test_survives_copy_with(self):
        assert MorphOptions().copy_with(density_rasterization="exact").density_rasterization == "exact"


@pytest.fixture(scope="module")
def states():
    return data.load_us_census(population=True).reset_index(drop=True)


class TestMorph:
    def test_center_is_unchanged_by_default(self, states):
        """The default run is the same with the option spelled out."""
        a = morph_gdf(states, "Population", options=MorphOptions(show_progress=False, grid_size=64, n_iter=20))
        b = morph_gdf(
            states,
            "Population",
            options=MorphOptions(show_progress=False, grid_size=64, n_iter=20, density_rasterization="center"),
        )
        assert [g.wkb for g in a.get_geometry()] == [g.wkb for g in b.get_geometry()]

    def test_exact_lowers_the_area_error(self, states):
        """Center sampling quantizes every region to whole cells; exact coverage removes that (grid 256, states)."""
        base = MorphOptions(show_progress=False, grid_size=256, n_iter=300)
        center = morph_gdf(states, "Population", options=base)
        exact = morph_gdf(states, "Population", options=base.copy_with(density_rasterization="exact"))
        assert exact.status == MorphStatus.CONVERGED
        assert exact.get_errors().max_error_pct < 0.5 * center.get_errors().max_error_pct
