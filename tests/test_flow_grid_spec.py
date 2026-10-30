"""GridSpec and LadderSpec in MorphOptions.grid, create_flow_cartogram and the deprecated entry points."""

import warnings

import numpy as np
import pytest

from carto_flow import data
from carto_flow.flow_cartogram import (
    CartogramWorkflow,
    GridSpec,
    LadderSpec,
    MorphOptions,
    create_flow_cartogram,
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
    options = _quiet(grid=GridSpec(64, margin=0.2, square=True))
    expected = Grid.from_bounds(BOUNDS, 64, 0.2, True)
    grid = options.get_grid(BOUNDS)
    assert (grid.sx, grid.sy, grid.dx, grid.dy) == (expected.sx, expected.sy, expected.dx, expected.dy)


def test_int_grid_size_keeps_margin_half_and_rectangular_cells():
    grid = _quiet(grid=64).get_grid(BOUNDS)
    expected = Grid.from_bounds(BOUNDS, 64, 0.5, False)
    assert (grid.sx, grid.sy) == (expected.sx, expected.sy)


def test_auto_is_a_ladder_with_automatic_levels():
    assert _quiet(grid="auto").grid_spec == LadderSpec()


def test_other_strings_are_rejected():
    with pytest.raises(MorphOptionsValidationError):
        _quiet(grid="fine")


def test_legacy_margin_and_square_are_deprecated_but_honored():
    with pytest.warns(DeprecationWarning, match="grid_margin is deprecated"):
        options = _quiet(grid=64, grid_margin=0.1)
    assert options.grid_spec == GridSpec(64, 0.1, False)
    with pytest.warns(DeprecationWarning, match="grid_square is deprecated"):
        options = _quiet(grid=64, grid_square=True)
    assert options.grid_spec == GridSpec(64, 0.5, True)


def test_defaults_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _quiet()
        _quiet(grid=GridSpec(64))


def test_prebuilt_grid_is_returned_as_is():
    grid = Grid.from_bounds(BOUNDS, 64)
    options = _quiet(grid=grid)
    assert options.get_grid(BOUNDS) is grid
    assert options.grid_spec is None


def test_grid_size_is_a_deprecated_alias_of_grid():
    with pytest.warns(DeprecationWarning, match="grid_size is deprecated"):
        options = _quiet(grid_size=64)
    assert options.grid == 64
    assert options.grid_size is None
    with pytest.warns(DeprecationWarning, match="grid_size is deprecated"):
        options.grid_size = LadderSpec()
    assert options.grid == LadderSpec()
    with pytest.warns(DeprecationWarning, match="grid_size is deprecated"):
        copy = options.copy_with(grid_size=128)
    assert copy.grid == 128
    assert options.grid == LadderSpec()


def test_invalid_grid_size_alias_is_rejected():
    with pytest.warns(DeprecationWarning), pytest.raises(MorphOptionsValidationError):
        _quiet(grid_size=-1)


def test_a_ladder_has_no_single_grid():
    with pytest.raises(ValueError, match="ladder"):
        _quiet(grid=LadderSpec()).get_grid(BOUNDS)


def test_create_flow_cartogram_runs_a_ladder(states):
    options = _quiet(grid=LadderSpec(min_resolution=64, levels=2, square=True), n_iter=30, recompute_every=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cartogram = create_flow_cartogram(states, "Population", options=options)
    assert cartogram.options.grid.sx in (64, 128)


def test_workflow_morph_runs_a_ladder_and_stores_each_level(states):
    options = _quiet(grid=LadderSpec(min_resolution=64, levels=2), n_iter=20, recompute_every=5)
    workflow = CartogramWorkflow(states, "Population", options=options)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        workflow.morph()
    assert 2 <= len(workflow) <= 3
    assert workflow[1].options.grid.sx == 64


def test_entry_point_takes_keywords_after_the_column(states):
    with pytest.raises(TypeError):
        create_flow_cartogram(states, "Population", None)


def test_n_iter_is_the_total_of_a_ladder(states):
    rng = np.random.default_rng(0)
    gdf = states[["geometry"]].copy()
    gdf["v"] = gdf.area * rng.lognormal(0, 0.2, len(gdf))
    options = _quiet(grid=LadderSpec(min_resolution=64, levels=3), n_iter=300, mean_tol=0.01, max_tol=0.03)
    workflow = CartogramWorkflow(gdf, "v", options=options)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        workflow.morph()
    levels = list(workflow)[1:]
    assert len(levels) >= 2
    assert sum(r.niterations for r in levels) <= 300
    # a later level is given what the earlier levels left
    assert levels[1].options.n_iter == 300 - levels[0].niterations


def test_a_list_of_options_keeps_n_iter_per_level(states):
    options = [_quiet(n_iter=20, recompute_every=10, mean_tol=1e-4, max_tol=1e-3) for _ in range(2)]
    workflow = CartogramWorkflow(states, "Population", options=options[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        workflow.morph_multiresolution(min_resolution=64, levels=2, options=options)
    assert [r.options.n_iter for r in list(workflow)[1:]] == [20, 20]


def test_remaining_iterations_below_recompute_every_are_allowed(states):
    options = _quiet(
        grid=LadderSpec(min_resolution=64, levels=3),
        n_iter=25,
        recompute_every=10,
        mean_tol=1e-4,
        max_tol=1e-3,
        stall_patience=None,
    )
    workflow = CartogramWorkflow(states, "Population", options=options)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        workflow.morph()
    assert sum(r.niterations for r in list(workflow)[1:]) == 25
