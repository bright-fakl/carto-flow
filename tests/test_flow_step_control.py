"""Tests for the step control of the flow cartogram (``MorphOptions.adaptive_dt``)."""

from itertools import pairwise
from typing import ClassVar

import numpy as np
import pytest
import shapely

import carto_flow.data as data
import carto_flow.flow_cartogram.algorithm as algorithm
from carto_flow.flow_cartogram import CartogramWorkflow, MorphOptions, MorphStatus, create_flow_cartogram
from carto_flow.flow_cartogram.stepsize import (
    DEAD_BAND_FRACTION,
    FLOOR,
    RESTART_SCORE,
    STRENGTH,
    THRESHOLD,
    WINDOW,
    StepController,
)


@pytest.fixture(scope="module")
def states():
    return data.load_us_census(population=True).reset_index(drop=True)


@pytest.fixture(scope="module")
def districts():
    return data.load_us_census(population=True, level="congressional_district").reset_index(drop=True)


def _options(adaptive, **kwargs):
    return MorphOptions(**{"grid_size": 128, "show_progress": False, "adaptive_dt": adaptive, **kwargs})


def _score(conv, options):
    return np.maximum(
        conv.mean_log_errors / np.log2(1 + options.mean_tol),
        conv.max_log_errors / np.log2(1 + options.max_tol),
    )


def _true_max_errors(result, values):
    """True maximum area error in percent of every recorded iteration."""
    shares = values / values.sum()
    out = []
    for snapshot in result.snapshots.snapshots:
        areas = shapely.area(np.asarray(list(snapshot.geometry), dtype=object))
        out.append(np.max(np.abs(areas / areas.sum() / shares - 1.0)) * 100)
    return np.array(out)


class TestOption:
    def test_default_is_on(self):
        assert MorphOptions().adaptive_dt is True

    @pytest.mark.parametrize("value", [None, 1, "yes"])
    def test_invalid_value_is_rejected(self, value):
        with pytest.raises(ValueError, match="adaptive_dt"):
            MorphOptions(adaptive_dt=value)

    def test_copy_with(self):
        assert MorphOptions().copy_with(adaptive_dt=False).adaptive_dt is False


def _alternating(n_regions, n_flipping, sign):
    """Log errors of which the first ``n_flipping`` regions have the given sign and the rest are positive."""
    e = np.full(n_regions, 0.05)
    e[:n_flipping] *= sign
    return e


BAND = 1e-3


def _controller(window, score, enabled=True):
    return StepController(window, score, BAND, enabled=enabled)


class TestStepController:
    def test_disabled_always_uses_the_full_step(self):
        controller = _controller(10, 0.1, enabled=False)
        for sign in (1, -1, 1, -1):
            controller.update(_alternating(20, 20, sign), 5.0)
        assert [controller.current(step) for step in (0, 1, 50)] == [1.0, 1.0, 1.0]

    def test_errors_that_keep_their_sign_never_shrink_the_step(self):
        controller = _controller(0, 100.0)
        for k in range(20):
            controller.update(np.linspace(0.5, 0.05, 20) * (0.9**k), 100.0)
        assert controller.current(5) == 1.0

    def test_alternating_errors_shrink_the_step_and_it_does_not_recover(self):
        controller = _controller(0, 100.0)
        fractions = []
        for k in range(12):
            controller.update(_alternating(20, 20, 1 if k % 2 == 0 else -1), 100.0)
            fractions.append(controller.current(5))
        assert fractions[0] == 1.0  # nothing to compare with yet
        assert fractions[1] == pytest.approx(1 - STRENGTH)  # all regions inverted
        assert all(b <= a for a, b in pairwise(fractions))
        assert fractions[-1] == FLOOR
        # Errors that stop alternating do not bring the step back.
        for _ in range(10):
            controller.update(np.full(20, 0.05), 1.0)
        assert controller.current(5) == FLOOR

    def test_a_sign_change_is_remembered_for_the_window(self):
        controller = _controller(0, 100.0)
        controller.update(_alternating(20, 20, 1), 100.0)
        controller.update(_alternating(20, 20, -1), 100.0)
        shrunk = controller.current(5)
        assert shrunk < 1.0
        # Steady errors afterwards: the share stays above the threshold while the change is in the window,
        # so the step shrinks further, and it stops once the change has left the window.
        values = []
        for _ in range(WINDOW + 3):
            controller.update(_alternating(20, 20, -1), 100.0)
            values.append(controller.current(5))
        assert values[-1] == values[-2] == values[-3]

    def test_share_below_the_threshold_is_ignored(self):
        n = 100
        controller = _controller(0, 100.0)
        controller.update(_alternating(n, 0, 1), 100.0)
        controller.update(_alternating(n, int(THRESHOLD * n), -1), 100.0)  # exactly the threshold: not above
        assert controller.current(5) == 1.0

    def test_errors_inside_the_dead_band_have_no_sign(self):
        controller = _controller(0, 100.0)
        for k in range(10):
            controller.update(np.full(20, 5e-4 * (1 if k % 2 == 0 else -1)), 100.0)
        assert controller.current(5) == 1.0

    def test_restart_window_scales_with_the_score(self):
        controller = _controller(10, 0.5 * RESTART_SCORE)
        assert controller.current(0) == pytest.approx(0.5)
        assert controller.current(9) == pytest.approx(0.5)
        assert controller.current(10) == 1.0  # after the window
        controller.update(np.full(20, 0.05), 0.0)
        assert controller.current(1) == FLOOR  # not below the floor

    def test_far_from_the_target_the_step_is_full(self):
        controller = _controller(10, 1000.0)
        assert controller.current(0) == 1.0


class TestDeadBand:
    @staticmethod
    def _band(states, monkeypatch, max_tol):
        seen = []
        original = algorithm.StepController

        def recording(window, score, dead_band, enabled=True):
            seen.append(dead_band)
            return original(window, score, dead_band, enabled)

        monkeypatch.setattr(algorithm, "StepController", recording)
        create_flow_cartogram(
            states, "Population", options=_options(True, n_iter=2, recompute_every=2, mean_tol=0.002, max_tol=max_tol)
        )
        return seen[0]

    def test_band_follows_the_max_tolerance(self, states, monkeypatch):
        default = self._band(states, monkeypatch, 0.1)
        tight = self._band(states, monkeypatch, 0.004)
        assert default == pytest.approx(DEAD_BAND_FRACTION * np.log2(1.1))
        assert tight == pytest.approx(DEAD_BAND_FRACTION * np.log2(1.004))
        assert tight < default

    def test_a_tight_band_sees_signs_that_a_wide_band_ignores(self):
        errors = [np.full(20, 0.004 * sign) for sign in (1, -1, 1, -1, 1, -1)]
        for band, shrinks in ((1e-3, True), (1e-2, False)):
            controller = StepController(0, 100.0, band)
            for e in errors:
                controller.update(e, 100.0)
            assert (controller.current(5) < 1.0) is shrinks


class TestRunsWithoutOvershoot:
    def test_default_tolerances_are_not_slower(self, states):
        off = create_flow_cartogram(states, "Population", options=_options(False))
        on = create_flow_cartogram(states, "Population", options=_options(True))
        assert off.status == on.status == MorphStatus.CONVERGED
        assert on.niterations <= 1.1 * off.niterations


class TestLatePhase:
    def test_amplitude_of_the_oscillation_drops(self, states):
        """Tolerances that cannot be met: the true maximum error keeps cycling around the target."""
        values = states["Population"].to_numpy(float)
        kwargs = {"n_iter": 150, "mean_tol": 1e-6, "max_tol": 1e-6, "stall_patience": None, "snapshot_every": 1}
        highest = {}
        for adaptive in (False, True):
            result = create_flow_cartogram(states, "Population", options=_options(adaptive, **kwargs))
            highest[adaptive] = _true_max_errors(result, values)[-40:].max()
        assert highest[False] > 1.0
        assert highest[True] < 0.5 * highest[False]

    @pytest.mark.parametrize("recompute_on_rise", [0.01, None])
    def test_tight_tolerances_converge_where_constant_dt_stalls(self, states, recompute_on_rise):
        """The step control works the same with and without recompute_on_rise."""
        kwargs = {"dt": 0.4, "mean_tol": 0.005, "max_tol": 0.01, "recompute_on_rise": recompute_on_rise}
        off = create_flow_cartogram(states, "Population", options=_options(False, **kwargs))
        on = create_flow_cartogram(states, "Population", options=_options(True, **kwargs))
        assert on.status == MorphStatus.CONVERGED
        assert off.status != MorphStatus.CONVERGED
        assert on.niterations < off.niterations


class TestRestart:
    TIGHT: ClassVar[dict] = {"mean_tol": 0.005, "max_tol": 0.01}

    @pytest.mark.parametrize("adaptive", [False, True])
    def test_refinement_of_a_converged_cartogram(self, states, adaptive):
        options = _options(adaptive, **self.TIGHT)
        workflow = CartogramWorkflow(states, "Population", options=options)
        workflow.morph()
        assert workflow.last.status == MorphStatus.CONVERGED
        second = workflow.morph()
        score = _score(second.convergence, second.options)
        if adaptive:
            # The first step is capped, so the converged state stays converged.
            assert second.niterations == 1
            assert score[0] < 1.0
        else:
            # A full step moves the fastest point dt cells and undoes the convergence.
            assert score[0] > 1.0
            assert second.niterations > 1

    def test_fresh_run_is_not_capped(self, states):
        """A run that starts far from the target takes the same first steps with and without the control."""
        scores = {}
        for adaptive in (False, True):
            result = create_flow_cartogram(
                states, "Population", options=_options(adaptive, n_iter=10, stall_patience=None)
            )
            scores[adaptive] = _score(result.convergence, result.options)
        np.testing.assert_allclose(scores[True], scores[False], rtol=1e-9)

    def test_districts_multiresolution_converges(self, districts):
        """The coarse level of the districts must not stall at tight tolerances (0.5 % / 1 %)."""
        options = _options(True, n_iter=300, mean_tol=0.005, max_tol=0.01)
        workflow = CartogramWorkflow(districts, "Population", options=options)
        workflow.morph_multiresolution(min_resolution=128, levels=3)
        assert workflow.last.status == MorphStatus.CONVERGED

    def test_multiresolution_converges_on_the_coarse_level(self, states):
        iterations = {}
        for adaptive in (False, True):
            workflow = CartogramWorkflow(states, "Population", options=_options(adaptive, n_iter=300, **self.TIGHT))
            workflow.morph_multiresolution(min_resolution=128, levels=3)
            assert workflow.last.status == MorphStatus.CONVERGED
            iterations[adaptive] = sum(result.niterations for result in workflow.results[1:])
        assert iterations[True] < iterations[False]
