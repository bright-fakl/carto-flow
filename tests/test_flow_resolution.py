"""Automatic grid resolution and the resolution warnings of the multiresolution morph."""

import warnings

import numpy as np
import pytest

from carto_flow import data
from carto_flow.flow_cartogram import MorphOptions, multiresolution_morph
from carto_flow.flow_cartogram.resolution import (
    MAX_RESOLUTION,
    MAX_START,
    auto_levels,
    auto_start,
    check_resolution,
    smallest_resolution,
)

BOUNDS = (0.0, 0.0, 1.0, 1.0)  # with margin 0.5 the grid spans 2 x 2 units
CELL64 = (2 / 64) ** 2


def _targets(smallest_cells, n=100, resolution=64):
    """n regions of 10 cells each except the first, which covers smallest_cells cells at the resolution."""
    t = np.full(n, 10.0 * (2 / resolution) ** 2)
    t[0] = smallest_cells * (2 / resolution) ** 2
    return t


@pytest.fixture(scope="module")
def states():
    return data.load_us_census(population=True)


@pytest.fixture(scope="module")
def districts():
    return data.load_us_census(population=True, level="congressional_district")


def _targets_of(gdf):
    values = gdf["Population"].to_numpy(float)
    return gdf.area.sum() * values / values.sum()


def test_smallest_resolution_is_first_power_of_two_with_enough_cells():
    assert smallest_resolution(BOUNDS, _targets(3.0)) == 64
    assert smallest_resolution(BOUNDS, _targets(2.0)) == 128  # 2 cells at 64 is 8 at 128
    assert smallest_resolution(BOUNDS, _targets(0.1)) == 512  # 0.1 cells at 64 is 6.4 at 512


def test_zero_targets_are_ignored():
    t = np.concatenate([_targets(3.0), [0.0]])
    assert smallest_resolution(BOUNDS, t) == 64


def test_empty_targets_have_no_resolution():
    assert smallest_resolution(BOUNDS, np.zeros(3)) is None


def test_auto_start_is_limited_to_the_maximum():
    assert auto_start(BOUNDS, _targets(0.001)) == MAX_START
    assert auto_start(BOUNDS, np.zeros(3)) == MAX_START


def test_auto_levels_double_up_to_the_finest_grid():
    assert auto_levels(128) == 5
    assert auto_levels(512) == 3
    assert auto_levels(MAX_RESOLUTION) == 1
    assert auto_levels(4096) == 1


def test_bundled_data_start_grids(states, districts):
    assert auto_start(states.total_bounds, _targets_of(states)) == 128
    assert auto_start(districts.total_bounds, _targets_of(districts)) == 128


def test_warning_when_coarsest_grid_is_too_coarse():
    with pytest.warns(
        UserWarning, match=r"smallest target region covers 1 cells at the coarsest grid \(64\).*min_resolution=128"
    ):
        check_resolution(BOUNDS, _targets(1.0), 64, 1)


def test_warning_names_the_regions_below_one_cell_at_the_finest_grid():
    t = _targets(0.5)
    t[:5] = 0.5 * CELL64
    with pytest.warns(
        UserWarning, match=r"5 of 100 target regions \(5 %\) are smaller than one cell at the finest grid \(64\)"
    ):
        check_resolution(BOUNDS, t, 64, 1)


def test_no_warning_when_resolved():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_resolution(BOUNDS, _targets(3.0), 64, 3)


def test_few_sub_cell_regions_at_the_finest_grid_do_not_warn_about_it():
    # one region of 0.5 cells at 64 is 8 cells at 128, so only the coarsest-grid message appears
    with pytest.warns(UserWarning) as record:
        check_resolution(BOUNDS, _targets(0.5), 64, 2)
    assert len(record) == 1
    assert "coarsest grid" in str(record[0].message)
    assert "finest grid" not in str(record[0].message)


def test_morph_with_automatic_resolution_converges_without_a_resolution_warning(states):
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Grid resolution")
        cartogram = multiresolution_morph(
            states, "Population", min_resolution="auto", levels="auto", options=MorphOptions(show_progress=False)
        )
    assert cartogram.status.value == "converged"
    assert cartogram.options.grid.sx == 128


def test_morph_warns_for_a_too_coarse_explicit_resolution(districts):
    with pytest.warns(UserWarning, match="Grid resolution"):
        multiresolution_morph(
            districts, "Population", min_resolution=64, levels=2, options=MorphOptions(show_progress=False)
        )


def test_levels_auto_with_explicit_resolution(states):
    cartogram = multiresolution_morph(
        states, "Population", min_resolution=256, levels="auto", options=MorphOptions(show_progress=False)
    )
    assert cartogram.options.grid.sx == 256


def test_warning_when_no_grid_up_to_the_maximum_resolves_the_smallest_region():
    with pytest.warns(UserWarning, match=r"no grid up to 2048 does"):
        check_resolution(BOUNDS, _targets(0.001), 512, 3)
