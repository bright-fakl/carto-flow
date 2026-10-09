"""Tests for stall detection, best-iterate return and stop reason of the flow cartogram."""

from typing import ClassVar

import numpy as np
import pytest

import carto_flow.data as data
from carto_flow.flow_cartogram import CartogramWorkflow, MorphOptions, MorphStatus, StopReason, morph_gdf
from carto_flow.flow_cartogram.anisotropy import DirectionalTensor


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


class TestStallPatience:
    OPTIONS: ClassVar[dict] = {"show_progress": False, "n_iter": 60, "dt": 0.6, "mean_tol": 0.001, "max_tol": 0.002}

    @pytest.mark.parametrize("patience", [0, 3, 8])
    def test_stops_after_patience_iterations_without_a_new_best(self, states, patience):
        options = MorphOptions(**self.OPTIONS, stall_patience=patience)
        result = morph_gdf(states, "Population", options=options)
        score = _score(result.convergence, options)

        assert result.status == MorphStatus.STALLED
        assert result.stop_reason == StopReason.STALL_PATIENCE
        assert result.niterations < 60
        assert result.best_iteration == int(np.argmin(score)) + 1
        assert result.niterations - result.best_iteration == patience + 1
        assert result.latest.iteration == result.best_iteration

    def test_none_disables_stall_detection(self, states):
        options = MorphOptions(**self.OPTIONS, stall_patience=None)
        result = morph_gdf(states, "Population", options=options)
        assert result.niterations == 60
        assert result.stop_reason == StopReason.ITERATION_LIMIT

    def test_negative_patience_is_rejected(self):
        with pytest.raises(ValueError, match="stall_patience"):
            MorphOptions(stall_patience=-1)
