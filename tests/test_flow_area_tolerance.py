"""Absolute area tolerance (area_tol_cells): regions of a few cells count as on target."""

import warnings

import numpy as np
import pytest

import carto_flow.data as data
from carto_flow.flow_cartogram import CartogramWorkflow, MorphOptions, MorphStatus, create_flow_cartogram
from carto_flow.flow_cartogram.errors import count_unresolved, effective_log_errors
from carto_flow.flow_cartogram.options import MorphOptionsValidationError


@pytest.fixture(scope="module")
def states():
    return data.load_us_census(population=True).reset_index(drop=True)


OPTIONS = {"grid": 64, "n_iter": 30, "area_scale": 1e-6, "show_progress": False}


class TestHelpers:
    def test_regions_within_the_tolerance_are_zeroed(self):
        current = np.array([10.0, 10.0, 10.0])
        target = np.array([10.4, 12.0, 5.0])
        log_errors = np.log2(current / target)
        effective = effective_log_errors(log_errors, current, target, abs_tol=0.5)
        assert effective[0] == 0.0
        np.testing.assert_array_equal(effective[1:], log_errors[1:])

    def test_zero_tolerance_changes_nothing(self):
        log_errors = np.array([0.3, -0.2])
        assert effective_log_errors(log_errors, np.ones(2), np.ones(2), 0.0) is log_errors
        assert count_unresolved(log_errors, np.ones(2), np.ones(2), 0.0, 0.1) == 0

    def test_unresolved_regions_are_excused_with_a_relative_error_above_the_tolerance(self):
        current = np.array([1.0, 1.0, 100.0])
        target = np.array([2.0, 1.02, 150.0])
        log_errors = np.log2(current / target)
        # region 0 is within 1.5 area units and off by 50 %; region 1 is within and almost on target; region 2 is outside
        assert count_unresolved(log_errors, current, target, 1.5, np.log2(1.1)) == 1


class TestOption:
    def test_default_is_off(self):
        assert MorphOptions().area_tol_cells == 0.0

    @pytest.mark.parametrize("value", [-1, "1", True])
    def test_invalid_values_raise(self, value):
        with pytest.raises(MorphOptionsValidationError, match="area_tol_cells"):
            MorphOptions(area_tol_cells=value)


class TestMorph:
    def test_off_reports_no_unresolved_regions(self, states):
        result = create_flow_cartogram(states, "Population", options=MorphOptions(**OPTIONS))
        assert result.n_unresolved == 0

    def test_a_large_tolerance_converges_at_once_and_the_errors_stay_true(self, states):
        options = MorphOptions(**OPTIONS, area_tol_cells=1e9)
        result = create_flow_cartogram(states, "Population", options=options)
        assert result.status == MorphStatus.CONVERGED
        assert result.niterations == 1
        errors = result.get_errors()
        assert errors.max_error_pct > options.max_tol * 100  # the reported error is the relative error
        assert result.n_unresolved == int(np.count_nonzero(np.abs(errors.errors_pct) >= options.max_tol * 100 - 1e-9))

    def test_a_tolerance_never_makes_the_run_worse_than_without_it(self, states):
        plain = create_flow_cartogram(states, "Population", options=MorphOptions(**OPTIONS))
        tolerant = create_flow_cartogram(states, "Population", options=MorphOptions(**OPTIONS, area_tol_cells=0.5))
        # excused regions only make the convergence test easier: it cannot end later
        assert tolerant.niterations <= plain.niterations
        assert 0 <= tolerant.n_unresolved <= len(states)


class TestStallWarning:
    def test_the_warning_names_the_continuation(self, states):
        result = create_flow_cartogram(states, "Population", options=MorphOptions(**OPTIONS))
        result.status = MorphStatus.STALLED
        with pytest.warns(UserWarning, match=r"stall_patience=None, n_iter=N"):
            assert CartogramWorkflow._warn_if_stalled(result) is result

    def test_no_warning_without_a_stall(self, states):
        result = create_flow_cartogram(states, "Population", options=MorphOptions(**OPTIONS))
        result.status = MorphStatus.COMPLETED
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            CartogramWorkflow._warn_if_stalled(result)


class TestLadder:
    def test_the_tolerance_is_the_same_area_at_every_level(self, states):
        workflow = CartogramWorkflow(states, "Population", validate=False)
        options = MorphOptions(n_iter=4, area_scale=1e-6, show_progress=False, area_tol_cells=2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            workflow.morph_multiresolution(min_resolution=32, levels=2, options=[options, options])
        coarse, fine = list(workflow)[1:3]
        area = [c.options.area_tol_cells * c.grid.dx * c.grid.dy for c in (coarse, fine)]
        assert area[0] == pytest.approx(area[1])
        assert fine.options.area_tol_cells == pytest.approx(2.0)
