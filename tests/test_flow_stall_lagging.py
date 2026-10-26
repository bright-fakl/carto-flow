"""The lagging error: a stall component that follows the regions still above the tolerance."""

import warnings
from typing import ClassVar

import numpy as np
import pytest

import carto_flow.data as data
from carto_flow.flow_cartogram import CartogramWorkflow, MorphOptions, morph_geometries
from carto_flow.flow_cartogram.errors import lagging_error
from carto_flow.flow_cartogram.options import MorphOptionsValidationError
from carto_flow.flow_cartogram.stall import StallMonitor

TOL = np.log2(1.10)
MEAN_TOL = np.log2(1.05)


@pytest.fixture(scope="module")
def states():
    return data.load_us_census(population=True).reset_index(drop=True)


def stall_step(monitor, lags, mean=0.5, max_=20.0, cycle=10):
    """Iteration (1-based) at which the monitor stalls on cycles of constant mean and max ratios, or None."""
    for cycle_index, lag in enumerate(lags):
        for i in range(cycle):
            step = cycle_index * cycle + i
            if monitor.update(step, mean, max_, 0.1, lag):
                return step + 1
    return None


class TestLaggingError:
    def test_zero_when_everything_is_within_the_tolerances(self):
        assert lagging_error(np.array([0.01, -0.02, 0.0]), TOL, MEAN_TOL) == 0.0

    def test_sums_the_excess_over_the_regions_above_the_tolerance(self):
        errors = np.array([0.5, -0.4, 0.01])
        mean_excess = max(0.0, np.abs(errors).mean() - MEAN_TOL)
        expected = (0.5 - TOL) + (0.4 - TOL) + 3 * mean_excess
        assert lagging_error(errors, TOL, MEAN_TOL) == pytest.approx(expected)

    def test_does_not_depend_on_the_sign(self):
        errors = np.array([0.3, -0.2, 0.05])
        assert lagging_error(errors, TOL, MEAN_TOL) == lagging_error(-errors, TOL, MEAN_TOL)

    def test_falls_when_a_lagging_region_improves_even_if_the_max_stays(self):
        before = np.array([1.0, 0.5, 0.4, 0.0])
        after = np.array([1.0, 0.3, 0.2, 0.0])
        assert lagging_error(after, TOL, MEAN_TOL) < lagging_error(before, TOL, MEAN_TOL)


class TestStallMonitorLag:
    def test_falling_lag_is_progress_when_mean_and_max_are_flat(self):
        lags = [10.0 * 0.8**k for k in range(12)]
        assert stall_step(StallMonitor(3, 0.05, 10), lags) is None
        assert stall_step(StallMonitor(3, 0.05, 10), [None] * 12) is not None  # without the component: stalled

    def test_a_flat_lag_stalls_after_the_patience(self):
        assert stall_step(StallMonitor(3, 0.05, 10), [5.0] * 12) == 40  # one first cycle counts, then three without

    def test_a_creeping_lag_stalls(self):
        lags = [5.0 * 0.999**k for k in range(12)]
        assert stall_step(StallMonitor(3, 0.05, 10), lags) is not None

    def test_a_lag_of_zero_does_not_count(self):
        assert stall_step(StallMonitor(2, 0.05, 10), [0.0] * 6, mean=0.5, max_=0.5) is not None

    def test_a_rise_in_the_lag_is_not_progress(self):
        lags = [5.0 * 1.1**k for k in range(8)]
        assert stall_step(StallMonitor(2, 0.05, 10), lags) is not None


class TestOption:
    def test_default_is_on(self):
        assert MorphOptions().stall_lagging is True

    @pytest.mark.parametrize("value", [1, "yes", None])
    def test_invalid_values_raise(self, value):
        with pytest.raises(MorphOptionsValidationError, match="stall_lagging"):
            MorphOptions(stall_lagging=value)


def lag_of(snapshot, options):
    return lagging_error(snapshot.errors.log_errors, np.log2(1 + options.max_tol), np.log2(1 + options.mean_tol))


class TestRun:
    OPTIONS: ClassVar[dict] = {
        "grid": 64,
        "n_iter": 120,
        "area_scale": 1e-6,
        "show_progress": False,
        "snapshot_every": 1,
    }

    def test_the_returned_iterate_has_the_lowest_lagging_error_of_the_recorded_ones(self, states):
        options = MorphOptions(**self.OPTIONS, mean_tol=0.001, max_tol=0.002)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = morph_geometries(list(states.geometry), states["Population"].to_numpy(float), options=options)
        lags = {s.iteration: lag_of(s, options) for s in result.snapshots.snapshots}
        assert lags[result.best_iteration] == pytest.approx(min(lags.values()))

    def test_the_run_is_bounded_by_n_iter_and_never_ends_earlier_than_without_the_component(self, states):
        kwargs = {**self.OPTIONS, "snapshot_every": None, "mean_tol": 0.001, "max_tol": 0.002, "n_iter": 150}
        results = {}
        for lagging in (False, True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results[lagging] = morph_geometries(
                    list(states.geometry),
                    states["Population"].to_numpy(float),
                    options=MorphOptions(**kwargs, stall_lagging=lagging),
                )
        assert results[True].niterations <= 150
        assert results[True].niterations >= results[False].niterations  # the same until the old rule would stall

    def test_only_the_finest_level_of_a_ladder_follows_the_lagging_regions(self, states, monkeypatch):
        import carto_flow.flow_cartogram.workflow as workflow_module

        seen = []
        real = workflow_module.morph_geometries

        def short(*args, options, **kwargs):
            seen.append(options.stall_lagging)
            return real(*args, options=options.copy_with(n_iter=2), **kwargs)

        monkeypatch.setattr(workflow_module, "morph_geometries", short)
        workflow = CartogramWorkflow(states, "Population", validate=False)
        options = MorphOptions(n_iter=20, area_scale=1e-6, show_progress=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            workflow.morph_multiresolution(min_resolution=32, levels=3, options=options)
        assert seen == [False, False, True]

    def test_levels_with_their_own_options_keep_them(self, states):
        workflow = CartogramWorkflow(states, "Population", validate=False)
        options = MorphOptions(n_iter=5, area_scale=1e-6, show_progress=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            workflow.morph_multiresolution(min_resolution=32, levels=2, options=[options, options])
        assert [c.options.stall_lagging for c in list(workflow)[1:]] == [True, True]
