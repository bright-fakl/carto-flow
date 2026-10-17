"""Tests for stall detection, best-iterate return and stop reason of the flow cartogram."""

from typing import ClassVar

import numpy as np
import pytest

import carto_flow.data as data
from carto_flow.flow_cartogram import CartogramWorkflow, MorphOptions, MorphStatus, create_flow_cartogram
from carto_flow.flow_cartogram.anisotropy import DirectionalTensor
from carto_flow.flow_cartogram.recompute import score_rose, should_recompute
from carto_flow.flow_cartogram.stall import MIN_CYCLE_LENGTH, StallMonitor, cycle_length, region_size


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
        result = create_flow_cartogram(states, "Population", options=_strong_anisotropy_options(modulator))
        assert result.status == MorphStatus.CONVERGED
        assert result.niterations <= 400
        assert result.best_iteration == result.niterations


class TestConvergedRunsUnchanged:
    def test_default_options(self, states):
        result = create_flow_cartogram(states, "Population", options=MorphOptions(show_progress=False))
        assert result.status == MorphStatus.CONVERGED
        assert result.niterations == 112
        assert result.best_iteration == 112
        assert result.snapshots.get_iterations() == [112]

    def test_grid_512(self, states):
        result = create_flow_cartogram(states, "Population", options=MorphOptions(show_progress=False, grid=512))
        assert result.status == MorphStatus.CONVERGED
        assert result.niterations == 214
        assert result.best_iteration == result.niterations


class TestBestIterate:
    """A run that does not converge returns the iterate with the lowest score."""

    # A large step with tight tolerances: the error is lowest well before the end. The step control
    # would settle this run, so it is off here.
    OPTIONS: ClassVar[dict] = {
        "adaptive_dt": False,
        "show_progress": False,
        "n_iter": 60,
        "dt": 0.6,
        "mean_tol": 0.001,
        "max_tol": 0.002,
        "stall_patience": None,
    }

    def test_iteration_cap_returns_best_iterate(self, states):
        options = MorphOptions(**self.OPTIONS)
        result = create_flow_cartogram(states, "Population", options=options)
        score = _score(result.convergence, options)
        best = int(np.argmin(score)) + 1

        assert result.status == MorphStatus.COMPLETED
        assert result.niterations == 60
        assert result.best_iteration == best
        assert best < 60
        # The returned state is the best iterate; the final iterate stays available.
        assert result.best.iteration == best
        assert result.snapshots.get_snapshot(60) is not None
        assert result.get_errors().max_log_error == result.convergence.get_by_iteration(best).max_log_error
        assert result.get_errors().max_log_error < result.get_errors(60).max_log_error
        assert result.to_geodataframe().geometry.iloc[0].equals(result.get_geometry(best)[0])
        assert len(result.snapshots.get_iterations()) == len(set(result.snapshots.get_iterations()))

    def test_best_iterate_with_periodic_snapshots_is_not_duplicated(self, states):
        options = MorphOptions(**{**self.OPTIONS, "snapshot_every": 1})
        result = create_flow_cartogram(states, "Population", options=options)
        iterations = result.snapshots.get_iterations()
        assert len(iterations) == len(set(iterations))
        assert result.best.iteration == result.best_iteration

    def test_geometry_landmarks_and_coords_belong_to_the_same_iterate(self, states):
        landmarks = states.copy()
        coords = np.array([[x, y] for x in np.linspace(-2.5e6, 2.5e6, 5) for y in np.linspace(-1.5e6, 1.5e6, 5)])
        kwargs = {"landmarks": landmarks, "displacement_coords": coords}
        full = create_flow_cartogram(states, "Population", options=MorphOptions(**self.OPTIONS), **kwargs)
        assert full.best_iteration < full.niterations

        # Running exactly to the best iteration gives the same state.
        reference_options = MorphOptions(**{**self.OPTIONS, "n_iter": full.best_iteration})
        reference = create_flow_cartogram(states, "Population", options=reference_options, **kwargs)
        assert reference.niterations == full.best_iteration

        for got, want in zip(full.get_geometry(), reference.get_geometry(), strict=True):
            assert got.equals_exact(want, tolerance=1e-3)
        for got, want in zip(full.get_landmarks(), reference.get_landmarks(), strict=True):
            assert got.equals_exact(want, tolerance=1e-3)
        np.testing.assert_allclose(full.get_coords(), reference.get_coords(), atol=1e-3)
        assert not np.allclose(full.get_coords(), full.get_coords(full.niterations), atol=1e-3)

    def test_serialization_round_trip(self, states, tmp_path):
        result = create_flow_cartogram(states, "Population", options=MorphOptions(**self.OPTIONS))
        result.save(tmp_path / "c.json")
        loaded = type(result).load(tmp_path / "c.json")
        assert loaded.best_iteration == result.best_iteration
        assert loaded.best.iteration == result.best_iteration


class TestWorkflowContinuation:
    def test_refinement_continues_from_the_best_iterate(self, states):
        # The margin of the final comparison is small and tuned on center sampling.
        scenario = {**TestBestIterate.OPTIONS, "density_rasterization": "center"}
        options = MorphOptions(**scenario)
        workflow = CartogramWorkflow(states, "Population", options=options)
        first = workflow.morph()
        assert first.best_iteration < first.niterations
        best_geometry = first.best.geometry

        # The exported geometry is the best iterate, and it is the start of the next run.
        assert workflow.to_geodataframe().geometry.iloc[0].equals(best_geometry[0])
        second = workflow.morph(options=MorphOptions(**{**scenario, "n_iter": 1, "recompute_every": 1}))
        start_error = first.get_errors().mean_log_error
        final_error = first.get_errors(first.niterations).mean_log_error
        assert abs(second.convergence.mean_log_errors[0] - start_error) < abs(
            second.convergence.mean_log_errors[0] - final_error
        )


def _run_monitor(monitor, mean, mx, moved=0.1):
    """Feed per-iteration ratios; return the 1-based iteration at which the monitor stalls, or None.

    With ``moved`` 0.1 a cycle of 10 iterations moves 1, so the efficiency is the improvement per cycle.
    """
    for i, (m, x) in enumerate(zip(mean, mx, strict=True)):
        if monitor.update(i, m, x, moved):
            return i + 1
    return None


class TestStallMonitor:
    """The cycle-level stall decision, on synthetic score traces."""

    @staticmethod
    def _sawtooth(minima, cycle=10, amplitude=3.0):
        """Per-iteration ratios: each cycle starts at its minimum and rises toward the next recompute."""
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
    OPTIONS: ClassVar[dict] = {
        "adaptive_dt": False,
        "show_progress": False,
        "n_iter": 60,
        "dt": 0.6,
        "mean_tol": 0.001,
        "max_tol": 0.002,
    }

    def test_preset_balanced_stalls_on_a_diverging_step(self, states):
        # The diverging step is tuned on center sampling; this test is about the stall rule, not the rasterizer.
        options = MorphOptions.preset_balanced().copy_with(
            dt=0.6, show_progress=False, recompute_on_rise=None, density_rasterization="center", adaptive_dt=False
        )
        result = create_flow_cartogram(states, "Population", options=options)
        score = _score(result.convergence, options)

        assert result.status == MorphStatus.STALLED
        assert result.niterations < options.n_iter
        # Stops at the end of a cycle, and returns the best iterate.
        assert result.niterations % options.recompute_every == 0
        assert result.best_iteration == int(np.argmin(score)) + 1
        assert result.best_iteration < result.niterations
        assert result.best.iteration == result.best_iteration

    def test_none_disables_stall_detection(self, states):
        options = MorphOptions(**self.OPTIONS, stall_patience=None)
        result = create_flow_cartogram(states, "Population", options=options)
        assert result.niterations == 60

    @pytest.mark.parametrize("recompute_every", [1, 2, 5, 10])
    def test_horizontal_anisotropy_converges_for_any_recompute_interval(self, states, recompute_every):
        options = _strong_anisotropy_options(DirectionalTensor(theta=0, Dpar=4, Dperp=0.3)).copy_with(
            recompute_every=recompute_every
        )
        result = create_flow_cartogram(states, "Population", options=options)
        assert result.status == MorphStatus.CONVERGED

    @pytest.mark.parametrize("value", [-1, True, 2.5])
    def test_invalid_patience_is_rejected(self, value):
        with pytest.raises(ValueError, match="stall_patience"):
            MorphOptions(stall_patience=value)

    @pytest.mark.parametrize("value", [-0.1, True, "x"])
    def test_invalid_min_efficiency_is_rejected(self, value):
        with pytest.raises(ValueError, match="stall_min_efficiency"):
            MorphOptions(stall_min_efficiency=value)


class TestRecomputeOnRise:
    def test_score_rose_uses_relative_tolerance(self):
        assert score_rose(10.5, 10.0, 0.0)
        assert not score_rose(10.0, 10.0, 0.0)
        assert score_rose(10.2, 10.0, 0.01)
        assert not score_rose(10.05, 10.0, 0.01)
        assert not score_rose(100.0, 10.0, None)

    def test_fixed_schedule_without_recompute_on_rise(self):
        recomputes = [i for i in range(25) if should_recompute(i, i % 10, True, 10, None)]
        assert recomputes == [0, 10, 20]
        assert not should_recompute(5, 5, True, None, None)

    def test_recomputes_after_a_rise(self):
        assert should_recompute(0, 0, False, 10, 0.0)
        assert should_recompute(4, 3, True, 10, 0.0)
        assert not should_recompute(4, 3, False, 10, 0.0)

    def test_field_is_used_at_least_one_iteration(self):
        assert not should_recompute(4, 0, True, 10, 0.0)

    def test_maximum_interval_is_respected(self):
        assert should_recompute(14, 10, False, 10, 0.0)
        assert not should_recompute(14, 9, False, 10, 0.0)
        # Without a maximum interval only rises recompute.
        assert not should_recompute(300, 299, False, None, 0.0)
        assert should_recompute(300, 299, True, None, 0.0)

    def test_stall_window_does_not_depend_on_recompute_times(self):
        # The monitor judges fixed windows from iteration 1, whatever the recompute times.
        flat = np.full(100, 5.0)
        monitor = StallMonitor(4, 0.02, cycle_length(10))
        assert _run_monitor(monitor, flat, flat) == 50

    def test_fewer_rises_than_the_fixed_schedule(self, states):
        options = _strong_anisotropy_options(DirectionalTensor(theta=0, Dpar=4, Dperp=0.3)).copy_with(
            benchmark=True, adaptive_dt=False
        )

        def rises(result):
            s = _score(result.convergence, options)
            return int(np.sum((s[1:] > s[:-1]) & (s[:-1] < 20)))

        off = create_flow_cartogram(states, "Population", options=options.copy_with(recompute_on_rise=None))
        on = create_flow_cartogram(states, "Population", options=options.copy_with(recompute_on_rise=0.01))
        assert off.status == on.status == MorphStatus.CONVERGED
        assert rises(on) < rises(off)
        assert on.benchmark.density_calls != off.benchmark.density_calls

    def test_no_change_when_the_score_never_rises(self, states):
        off = create_flow_cartogram(
            states, "Population", options=MorphOptions(show_progress=False, recompute_on_rise=None)
        )
        on = create_flow_cartogram(states, "Population", options=MorphOptions(show_progress=False))
        assert on.niterations == off.niterations == 112

    def test_default_is_one_percent(self):
        assert MorphOptions().recompute_on_rise == 0.01

    @pytest.mark.parametrize("value", [-0.1, "x", True])
    def test_invalid_value_is_rejected(self, value):
        with pytest.raises(ValueError, match="recompute_on_rise"):
            MorphOptions(recompute_on_rise=value)


class TestBestAndLast:
    """`best` is the returned state, `last` the last stored snapshot; histories stay in iteration order."""

    @pytest.fixture(scope="class")
    def result(self, states):
        return create_flow_cartogram(
            states, "Population", options=MorphOptions(**TestBestIterate.OPTIONS, snapshot_every=20)
        )

    def test_snapshots_are_in_iteration_order(self, result):
        iterations = result.snapshots.get_iterations()
        assert iterations == sorted(iterations)
        assert iterations[-1] == result.niterations
        assert result.best_iteration in iterations
        assert result.best_iteration < result.niterations

    def test_best_and_last(self, result):
        assert result.best.iteration == result.best_iteration
        assert result.last.iteration == result.niterations
        assert result.get_errors().max_error_pct < result.get_errors(iteration=result.niterations).max_error_pct

    def test_defaults_read_the_best_snapshot(self, result):
        gdf = result.to_geodataframe()
        assert list(gdf.geometry) == list(result.best.geometry)
        assert list(result.get_geometry()) == list(result.best.geometry)

    def test_latest_is_deprecated_and_means_best(self, result):
        with pytest.warns(DeprecationWarning, match="best"):
            assert result.latest is result.best

    def test_converged_run_has_the_same_best_and_last(self, states):
        result = create_flow_cartogram(states, "Population", options=MorphOptions(show_progress=False))
        assert result.best is result.last

    def test_internals_end_at_the_best_iterate(self, states):
        result = create_flow_cartogram(
            states, "Population", options=MorphOptions(**TestBestIterate.OPTIONS, save_internals=True)
        )
        internals = result.internals
        assert internals.best_iteration <= result.best_iteration
        assert internals.best().iteration == internals.best_iteration
        assert all(s.iteration <= result.best_iteration for s in internals.through_best())

    def test_workflow_last_replaces_latest(self, states):
        workflow = CartogramWorkflow(states, "Population", options=MorphOptions(show_progress=False))
        workflow.morph()
        assert workflow.last is workflow[-1]
        with pytest.warns(DeprecationWarning, match="last"):
            assert workflow.latest is workflow.last

    @pytest.mark.filterwarnings("ignore:Animation was deleted")
    def test_animation_stops_at_the_best_iteration(self, result):
        from carto_flow.flow_cartogram.animation import animate_morph_history

        seen = []

        def mapper(progress, variables):
            seen.append(list(variables["iteration"]))
            return 0.0

        animate_morph_history(result, position_mapper=mapper, duration=0.2, fps=5)
        animate_morph_history(result, position_mapper=mapper, duration=0.2, fps=5, through="last")
        assert seen[0][-1] == result.best_iteration
        assert seen[-1][-1] == result.niterations
        with pytest.raises(ValueError, match="through"):
            animate_morph_history(result, through="never")

    def test_repr_names_the_best_iteration(self, result):
        assert f"iter={result.niterations} (best {result.best_iteration})" in repr(result)

    @pytest.mark.filterwarnings("ignore:Animation was deleted")
    def test_workflow_animation_title_uses_the_best_iteration(self, states):
        from carto_flow.flow_cartogram.animation import animate_workflow

        workflow = CartogramWorkflow(states, "Population", options=MorphOptions(**TestBestIterate.OPTIONS))
        workflow.morph()
        run = workflow.last
        anim = animate_workflow(workflow, title="{iteration:.0f}|{n_iterations:.0f}", duration=1.0, fps=4)
        anim._func(anim._save_count - 1 if hasattr(anim, "_save_count") else 3)
        shown, total = anim._fig.axes[0].get_title().split("|")
        assert int(shown) == run.best_iteration
        assert int(total) == run.niterations


class TestRequiredImprovementFollowsTheStep:
    """The improvement a cycle has to make is the efficiency times the distance moved relative to the region size."""

    @staticmethod
    def _slow_trace():
        # the ratio falls 1 % per cycle of 10 iterations
        return np.array([50.0 * 0.99 ** (step // 10) for step in range(400)])

    def test_region_size_is_the_area_weighted_root(self):
        assert region_size(np.array([4.0, 4.0])) == pytest.approx(2.0)
        assert region_size(np.array([1.0, 3.0])) == pytest.approx(np.sqrt((1 + 9) / 4))

    def test_steady_slow_progress_stalls_when_much_is_moved_and_not_when_little(self):
        trace = self._slow_trace()
        assert _run_monitor(StallMonitor(4, 0.05, 10), trace, trace, moved=0.1) is not None  # 5 % required per cycle
        assert _run_monitor(StallMonitor(4, 0.05, 10), trace, trace, moved=0.01) is None  # 0.5 % required per cycle

    def test_zero_efficiency_never_stalls_on_decreasing_minima(self):
        trace = self._slow_trace()
        assert _run_monitor(StallMonitor(4, 0.0, 10), trace, trace) is None

    def test_distance_is_proportional_to_the_cell_size(self, states, monkeypatch):
        from carto_flow.flow_cartogram import algorithm

        first = {}

        class Spy(StallMonitor):
            def update(self, step, mean_ratio, max_ratio, moved):
                first.setdefault(self.cycle, moved)
                return super().update(step, mean_ratio, max_ratio, moved)

        monkeypatch.setattr(algorithm, "StallMonitor", Spy)
        moved = {}
        for grid in (128, 512):
            first.clear()
            options = MorphOptions(grid=grid, area_scale=1e-6, n_iter=1, show_progress=False, adaptive_dt=False)
            create_flow_cartogram(states, "Population", options=options)
            moved[grid] = next(iter(first.values()))
        assert moved[128] / moved[512] == pytest.approx(4.0, rel=0.05)
