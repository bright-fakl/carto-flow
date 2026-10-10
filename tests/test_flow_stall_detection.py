"""Tests for stall detection, best-iterate return and stop reason of the flow cartogram."""

from typing import ClassVar

import numpy as np
import pytest

import carto_flow.data as data
from carto_flow.flow_cartogram import CartogramWorkflow, MorphOptions, MorphStatus, StopReason, morph_gdf
from carto_flow.flow_cartogram.anisotropy import DirectionalTensor
from carto_flow.flow_cartogram.refresh import score_rose, should_refresh
from carto_flow.flow_cartogram.stall import MIN_CYCLE_LENGTH, StallMonitor, cycle_length


@pytest.fixture(scope="module")
def states():
    return data.load_us_census(population=True).reset_index(drop=True)


def _score(conv, options):
    """Combined convergence score per iteration (below 1 means converged)."""
    return np.maximum(
        conv.mean_log_errors / np.log2(1 + options.mean_tol),
        conv.max_log_errors / np.log2(1 + options.max_tol),
    )


def _strong_anisotropy_options(modulator):
    return MorphOptions.preset_balanced().copy_with(
        area_scale=1e-6, n_iter=400, show_progress=False, anisotropy=modulator
    )


class TestStrongAnisotropyConverges:
    """Strong anisotropy has a long plateau of the max error and an early non-monotone phase."""

    @pytest.mark.parametrize(
        "modulator",
        [
            DirectionalTensor(theta=0, Dpar=4, Dperp=0.3),
            DirectionalTensor.tangential(Dpar=4, Dperp=0.3),
            DirectionalTensor(theta=np.pi / 6, Dpar=4, Dperp=0.3),
        ],
        ids=["horizontal", "tangential", "tilted"],
    )
    def test_converges_within_400_iterations(self, states, modulator):
        result = morph_gdf(states, "Population", options=_strong_anisotropy_options(modulator))
        assert result.status == MorphStatus.CONVERGED
        assert result.stop_reason == StopReason.CONVERGED
        assert result.niterations <= 400
        assert result.best_iteration == result.niterations


class TestConvergedRunsUnchanged:
    def test_default_options(self, states):
        result = morph_gdf(states, "Population", options=MorphOptions(show_progress=False))
        assert result.status == MorphStatus.CONVERGED
        assert result.niterations == 113
        assert result.best_iteration == 113
        assert result.stop_reason == StopReason.CONVERGED
        assert result.snapshots.get_iterations() == [113]

    def test_grid_512(self, states):
        result = morph_gdf(states, "Population", options=MorphOptions(show_progress=False, grid_size=512))
        assert result.status == MorphStatus.CONVERGED
        assert result.niterations == 214
        assert result.best_iteration == result.niterations


class TestBestIterate:
    """A run that does not converge returns the iterate with the lowest score."""

    # A large step with tight tolerances: the error is lowest well before the end.
    OPTIONS: ClassVar[dict] = {
        "show_progress": False,
        "n_iter": 60,
        "dt": 0.6,
        "mean_tol": 0.001,
        "max_tol": 0.002,
        "stall_patience": None,
    }

    def test_iteration_cap_returns_best_iterate(self, states):
        options = MorphOptions(**self.OPTIONS)
        result = morph_gdf(states, "Population", options=options)
        score = _score(result.convergence, options)
        best = int(np.argmin(score)) + 1

        assert result.status == MorphStatus.COMPLETED
        assert result.stop_reason == StopReason.ITERATION_LIMIT
        assert result.niterations == 60
        assert result.best_iteration == best
        assert best < 60
        # The returned state is the best iterate; the final iterate stays available.
        assert result.latest.iteration == best
        assert result.snapshots.get_snapshot(60) is not None
        assert result.get_errors().max_log_error == result.convergence.get_by_iteration(best).max_log_error
        assert result.get_errors().max_log_error < result.get_errors(60).max_log_error
        assert result.to_geodataframe().geometry.iloc[0].equals(result.get_geometry(best)[0])
        assert len(result.snapshots.get_iterations()) == len(set(result.snapshots.get_iterations()))

    def test_best_iterate_with_periodic_snapshots_is_not_duplicated(self, states):
        options = MorphOptions(**{**self.OPTIONS, "snapshot_every": 1})
        result = morph_gdf(states, "Population", options=options)
        iterations = result.snapshots.get_iterations()
        assert len(iterations) == len(set(iterations))
        assert result.latest.iteration == result.best_iteration

    def test_geometry_landmarks_and_coords_belong_to_the_same_iterate(self, states):
        landmarks = states.copy()
        coords = np.array([[x, y] for x in np.linspace(-2.5e6, 2.5e6, 5) for y in np.linspace(-1.5e6, 1.5e6, 5)])
        kwargs = {"landmarks": landmarks, "displacement_coords": coords}
        full = morph_gdf(states, "Population", options=MorphOptions(**self.OPTIONS), **kwargs)
        assert full.best_iteration < full.niterations

        # Running exactly to the best iteration gives the same state.
        reference_options = MorphOptions(**{**self.OPTIONS, "n_iter": full.best_iteration})
        reference = morph_gdf(states, "Population", options=reference_options, **kwargs)
        assert reference.niterations == full.best_iteration

        for got, want in zip(full.get_geometry(), reference.get_geometry(), strict=True):
            assert got.equals_exact(want, tolerance=1e-3)
        for got, want in zip(full.get_landmarks(), reference.get_landmarks(), strict=True):
            assert got.equals_exact(want, tolerance=1e-3)
        np.testing.assert_allclose(full.get_coords(), reference.get_coords(), atol=1e-3)
        assert not np.allclose(full.get_coords(), full.get_coords(full.niterations), atol=1e-3)

    def test_serialization_round_trip(self, states, tmp_path):
        result = morph_gdf(states, "Population", options=MorphOptions(**self.OPTIONS))
        result.save(tmp_path / "c.json")
        loaded = type(result).load(tmp_path / "c.json")
        assert loaded.best_iteration == result.best_iteration
        assert loaded.stop_reason == result.stop_reason
        assert loaded.latest.iteration == result.best_iteration


class TestWorkflowContinuation:
    def test_refinement_continues_from_the_best_iterate(self, states):
        options = MorphOptions(**TestBestIterate.OPTIONS)
        workflow = CartogramWorkflow(states, "Population", options=options)
        first = workflow.morph()
        assert first.best_iteration < first.niterations
        best_geometry = first.latest.geometry

        # The exported geometry is the best iterate, and it is the start of the next run.
        assert workflow.to_geodataframe().geometry.iloc[0].equals(best_geometry[0])
        second = workflow.morph(options=MorphOptions(**{**TestBestIterate.OPTIONS, "n_iter": 1, "recompute_every": 1}))
        start_error = first.get_errors().mean_log_error
        final_error = first.get_errors(first.niterations).mean_log_error
        assert abs(second.convergence.mean_log_errors[0] - start_error) < abs(
            second.convergence.mean_log_errors[0] - final_error
        )


def _run_monitor(monitor, mean, mx):
    """Feed per-iteration ratios; return the 1-based iteration at which the monitor stalls, or None."""
    for i, (m, x) in enumerate(zip(mean, mx, strict=True)):
        if monitor.update(i, m, x):
            return i + 1
    return None


class TestStallMonitor:
    """The cycle-level stall decision, on synthetic score traces."""

    @staticmethod
    def _sawtooth(minima, cycle=10, amplitude=3.0):
        """Per-iteration ratios: each cycle starts at its minimum and rises toward the next refresh."""
        trace = []
        for low in minima:
            trace.extend(low + amplitude * np.linspace(0.0, 1.0, cycle))
        return np.array(trace)

    def test_cycle_length_has_a_minimum(self):
        assert cycle_length(10) == 10
        assert cycle_length(25) == 25
        assert cycle_length(1) == MIN_CYCLE_LENGTH
        assert cycle_length(None) == MIN_CYCLE_LENGTH

    def test_sawtooth_with_falling_minima_is_progress(self):
        trace = self._sawtooth([20, 15, 11, 8, 6, 4.5, 3.4, 2.5])
        assert _run_monitor(StallMonitor(2, 0.02, 10), trace, trace) is None

    def test_creeping_minima_stall(self):
        # Each cycle minimum is 0.5% below the previous one: below the 2% threshold.
        minima = [10.0 * 0.995**k for k in range(10)]
        trace = self._sawtooth(minima)
        assert _run_monitor(StallMonitor(3, 0.02, 10), trace, trace) == 40  # 1 progress cycle + 3 without

    def test_creeping_minima_count_as_progress_without_threshold(self):
        minima = [10.0 * 0.995**k for k in range(10)]
        trace = self._sawtooth(minima)
        assert _run_monitor(StallMonitor(3, 0.0, 10), trace, trace) is None

    def test_flat_max_with_falling_mean_is_progress(self):
        mean = self._sawtooth([20, 15, 11, 8, 6, 4.5, 3.4, 2.5])
        flat_max = np.full_like(mean, 28.0)
        assert _run_monitor(StallMonitor(2, 0.02, 10), mean, flat_max) is None

    def test_creep_of_a_satisfied_component_does_not_count(self):
        # The mean component is below 1 and keeps improving; the max component is stuck above 1.
        mean = self._sawtooth([0.9 * 0.9**k for k in range(8)], amplitude=0.05)
        flat_max = np.full_like(mean, 10.0)
        assert _run_monitor(StallMonitor(3, 0.02, 10), mean, flat_max) == 40

    def test_violated_plateau_with_falling_mean_counts(self):
        # Same flat max, but the mean component is still above 1 and falling.
        mean = self._sawtooth([20, 15, 11, 8, 6, 4.5, 3.4, 2.5])
        flat_max = np.full_like(mean, 10.0)
        assert _run_monitor(StallMonitor(3, 0.02, 10), mean, flat_max) is None

    def test_progress_ends_when_the_falling_mean_becomes_satisfied(self):
        mean = self._sawtooth([4, 3, 2, 1.5, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4], amplitude=0.05)
        flat_max = np.full_like(mean, 10.0)
        # Cycles 1 to 4 have a violated, improving mean (cycle 4 minimum 1.5); then 3 cycles without progress.
        assert _run_monitor(StallMonitor(3, 0.02, 10), mean, flat_max) == 70

    def test_regression_above_one_after_being_satisfied_does_not_count(self):
        mean = self._sawtooth([0.8, 3.0, 2.5, 2.0, 1.6])
        flat_max = self._sawtooth([20, 20, 20, 20, 20], amplitude=0.0)
        assert _run_monitor(StallMonitor(3, 0.02, 10), mean, flat_max) == 40

    def test_first_cycle_counts_when_a_component_is_violated(self):
        trace = np.full(40, 5.0)
        assert _run_monitor(StallMonitor(1, 0.02, 10), trace, trace) == 20

    def test_flat_everything_stalls_after_patience_cycles(self):
        flat = np.full(100, 5.0)
        assert _run_monitor(StallMonitor(4, 0.02, 10), flat, flat) == 50

    def test_incomplete_cycle_is_not_judged(self):
        flat = np.full(49, 5.0)
        assert _run_monitor(StallMonitor(4, 0.02, 10), flat, flat) is None

    def test_none_disables(self):
        flat = np.full(200, 5.0)
        assert _run_monitor(StallMonitor(None, 0.02, 10), flat, flat) is None

    def test_a_rise_within_a_cycle_does_not_count(self):
        # The score rises in the second half of every cycle but the minima fall.
        trace = self._sawtooth([20, 16, 12, 9, 7, 5], amplitude=10.0)
        assert _run_monitor(StallMonitor(1, 0.02, 10), trace, trace) is None


class TestStallPatience:
    OPTIONS: ClassVar[dict] = {"show_progress": False, "n_iter": 60, "dt": 0.6, "mean_tol": 0.001, "max_tol": 0.002}

    def test_preset_balanced_stalls_on_a_diverging_step(self, states):
        options = MorphOptions.preset_balanced().copy_with(dt=0.6, show_progress=False, refresh_on_rise=None)
        result = morph_gdf(states, "Population", options=options)
        score = _score(result.convergence, options)

        assert result.status == MorphStatus.STALLED
        assert result.stop_reason == StopReason.STALL_PATIENCE
        assert result.niterations < options.n_iter
        # Stops at the end of a cycle, and returns the best iterate.
        assert result.niterations % options.recompute_every == 0
        assert result.best_iteration == int(np.argmin(score)) + 1
        assert result.best_iteration < result.niterations
        assert result.latest.iteration == result.best_iteration

    def test_none_disables_stall_detection(self, states):
        options = MorphOptions(**self.OPTIONS, stall_patience=None)
        result = morph_gdf(states, "Population", options=options)
        assert result.niterations == 60
        assert result.stop_reason == StopReason.ITERATION_LIMIT

    @pytest.mark.parametrize("recompute_every", [1, 2, 5, 10])
    def test_horizontal_anisotropy_converges_for_any_refresh_interval(self, states, recompute_every):
        options = _strong_anisotropy_options(DirectionalTensor(theta=0, Dpar=4, Dperp=0.3)).copy_with(
            recompute_every=recompute_every
        )
        result = morph_gdf(states, "Population", options=options)
        assert result.status == MorphStatus.CONVERGED

    def test_negative_patience_is_rejected(self):
        with pytest.raises(ValueError, match="stall_patience"):
            MorphOptions(stall_patience=-1)

    @pytest.mark.parametrize("value", [-0.1, 1.0, "x"])
    def test_invalid_min_improvement_is_rejected(self, value):
        with pytest.raises(ValueError, match="stall_min_improvement"):
            MorphOptions(stall_min_improvement=value)


class TestRefreshOnRise:
    def test_score_rose_uses_relative_tolerance(self):
        assert score_rose(10.5, 10.0, 0.0)
        assert not score_rose(10.0, 10.0, 0.0)
        assert score_rose(10.2, 10.0, 0.01)
        assert not score_rose(10.05, 10.0, 0.01)
        assert not score_rose(100.0, 10.0, None)

    def test_fixed_schedule_without_refresh_on_rise(self):
        refreshes = [i for i in range(25) if should_refresh(i, i % 10, True, 10, None)]
        assert refreshes == [0, 10, 20]
        assert not should_refresh(5, 5, True, None, None)

    def test_refreshes_after_a_rise(self):
        assert should_refresh(0, 0, False, 10, 0.0)
        assert should_refresh(4, 3, True, 10, 0.0)
        assert not should_refresh(4, 3, False, 10, 0.0)

    def test_field_is_used_at_least_one_iteration(self):
        assert not should_refresh(4, 0, True, 10, 0.0)

    def test_maximum_interval_is_respected(self):
        assert should_refresh(14, 10, False, 10, 0.0)
        assert not should_refresh(14, 9, False, 10, 0.0)
        # Without a maximum interval only rises refresh.
        assert not should_refresh(300, 299, False, None, 0.0)
        assert should_refresh(300, 299, True, None, 0.0)

    def test_stall_window_does_not_depend_on_refresh_times(self):
        # The monitor judges fixed windows from iteration 1, whatever the refresh times.
        flat = np.full(100, 5.0)
        monitor = StallMonitor(4, 0.02, cycle_length(10))
        assert _run_monitor(monitor, flat, flat) == 50

    def test_fewer_rises_than_the_fixed_schedule(self, states):
        options = _strong_anisotropy_options(DirectionalTensor(theta=0, Dpar=4, Dperp=0.3)).copy_with(benchmark=True)

        def rises(result):
            s = _score(result.convergence, options)
            return int(np.sum((s[1:] > s[:-1]) & (s[:-1] < 20)))

        off = morph_gdf(states, "Population", options=options.copy_with(refresh_on_rise=None))
        on = morph_gdf(states, "Population", options=options.copy_with(refresh_on_rise=0.01))
        assert off.status == on.status == MorphStatus.CONVERGED
        assert rises(on) < rises(off)
        assert on.benchmark.density_calls != off.benchmark.density_calls

    def test_no_change_when_the_score_never_rises(self, states):
        off = morph_gdf(states, "Population", options=MorphOptions(show_progress=False, refresh_on_rise=None))
        on = morph_gdf(states, "Population", options=MorphOptions(show_progress=False))
        assert on.niterations == off.niterations == 113

    def test_default_is_one_percent(self):
        assert MorphOptions().refresh_on_rise == 0.01

    @pytest.mark.parametrize("value", [-0.1, "x", True])
    def test_invalid_value_is_rejected(self, value):
        with pytest.raises(ValueError, match="refresh_on_rise"):
            MorphOptions(refresh_on_rise=value)
