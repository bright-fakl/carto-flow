"""GridSpec and LadderSpec in MorphOptions.grid_size, create_flow_cartogram and the deprecated entry points."""

import warnings

import pytest

from carto_flow import data
from carto_flow.flow_cartogram import (
    CartogramWorkflow,
    GridSpec,
    LadderSpec,
    MorphOptions,
    create_flow_cartogram,
    morph_gdf,
    multiresolution_morph,
)
from carto_flow.flow_cartogram.grid import Grid
from carto_flow.flow_cartogram.options import MorphOptionsValidationError

BOUNDS = (0.0, 0.0, 10.0, 5.0)


@pytest.fixture(scope="module")
def states():
    return data.load_us_census(population=True)


def _quiet(**kwargs):
    return MorphOptions(show_progress=False, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"size": 0}, {"size": (0, 4)}, {"size": "x"}, {"size": 64, "margin": -1}, {"size": 64, "square": 1}],
)
def test_grid_spec_rejects_invalid_values(kwargs):
    with pytest.raises((ValueError, TypeError)):
        GridSpec(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"min_resolution": 0}, {"levels": "x"}, {"levels": -1}, {"margin": -0.1}, {"square": "yes"}],
)
def test_ladder_spec_rejects_invalid_values(kwargs):
    with pytest.raises((ValueError, TypeError)):
        LadderSpec(**kwargs)


def test_grid_spec_builds_the_grid_of_its_size_margin_and_square():
    options = _quiet(grid_size=GridSpec(64, margin=0.2, square=True))
    expected = Grid.from_bounds(BOUNDS, 64, 0.2, True)
    grid = options.get_grid(BOUNDS)
    assert (grid.sx, grid.sy, grid.dx, grid.dy) == (expected.sx, expected.sy, expected.dx, expected.dy)


def test_int_grid_size_keeps_margin_half_and_rectangular_cells():
    grid = _quiet(grid_size=64).get_grid(BOUNDS)
    expected = Grid.from_bounds(BOUNDS, 64, 0.5, False)
    assert (grid.sx, grid.sy) == (expected.sx, expected.sy)


def test_auto_is_a_ladder_with_automatic_levels():
    assert _quiet(grid_size="auto").grid_spec == LadderSpec()


def test_other_strings_are_rejected():
    with pytest.raises(MorphOptionsValidationError):
        _quiet(grid_size="fine")


def test_legacy_margin_and_square_are_deprecated_but_honored():
    with pytest.warns(DeprecationWarning, match="grid_margin is deprecated"):
        options = _quiet(grid_size=64, grid_margin=0.1)
    assert options.grid_spec == GridSpec(64, 0.1, False)
    with pytest.warns(DeprecationWarning, match="grid_square is deprecated"):
        options = _quiet(grid_size=64, grid_square=True)
    assert options.grid_spec == GridSpec(64, 0.5, True)


def test_defaults_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _quiet()
        _quiet(grid_size=GridSpec(64))


def test_explicit_grid_cannot_be_combined_with_a_ladder():
    grid = Grid.from_bounds(BOUNDS, 64)
    with pytest.raises(MorphOptionsValidationError, match="ladder"):
        _quiet(grid=grid, grid_size="auto")
    with pytest.raises(MorphOptionsValidationError, match="ladder"):
        _quiet(grid=grid, grid_size=LadderSpec())
    assert _quiet(grid=grid, grid_size=64).get_grid(BOUNDS) is grid


def test_a_ladder_has_no_single_grid():
    with pytest.raises(ValueError, match="ladder"):
        _quiet(grid_size=LadderSpec()).get_grid(BOUNDS)


def test_create_flow_cartogram_runs_a_ladder(states):
    options = _quiet(grid_size=LadderSpec(min_resolution=64, levels=2, square=True), n_iter=30, recompute_every=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cartogram = create_flow_cartogram(states, "Population", options=options)
    assert cartogram.options.grid.sx in (64, 128)


def test_workflow_morph_runs_a_ladder_and_stores_each_level(states):
    options = _quiet(grid_size=LadderSpec(min_resolution=64, levels=2), n_iter=20, recompute_every=5)
    workflow = CartogramWorkflow(states, "Population", options=options)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        workflow.morph()
    assert 2 <= len(workflow) <= 3
    assert workflow[1].options.grid.sx == 64


def test_entry_point_takes_keywords_after_the_column(states):
    with pytest.raises(TypeError):
        create_flow_cartogram(states, "Population", None)


def test_morph_gdf_is_deprecated_and_equivalent(states):
    options = _quiet(n_iter=10, recompute_every=5)
    with pytest.warns(DeprecationWarning, match="create_flow_cartogram"):
        old = morph_gdf(states, "Population", options=options)
    new = create_flow_cartogram(states, "Population", options=options)
    assert old.niterations == new.niterations
    assert old.get_errors().mean_error_pct == pytest.approx(new.get_errors().mean_error_pct)


def test_multiresolution_morph_is_deprecated(states):
    options = _quiet(n_iter=10, recompute_every=5)
    with pytest.warns(DeprecationWarning, match="create_flow_cartogram"):
        cartogram = multiresolution_morph(states, "Population", min_resolution=64, levels=2, options=options)
    assert cartogram.options.grid.sx in (64, 128)
