"""The error floor is a fraction of the mean target area, not of a grid cell."""

from __future__ import annotations

import numpy as np
import pytest

from carto_flow import data
from carto_flow.flow_cartogram import CartogramWorkflow, MorphOptions, MorphStatus, morph_gdf


@pytest.fixture(scope="module")
def districts():
    return data.load_us_census(population=True, level="congressional_district").reset_index(drop=True)


@pytest.fixture(scope="module")
def states():
    return data.load_us_census(population=True).reset_index(drop=True)


class TestOption:
    def test_default_is_a_thousandth_of_the_mean_target_area(self):
        assert MorphOptions().error_floor == 1e-3

    @pytest.mark.parametrize("value", [-0.1, 1.0, "x", True, None])
    def test_invalid_value_is_rejected(self, value):
        with pytest.raises(ValueError, match="error_floor"):
            MorphOptions(error_floor=value)

    def test_zero_is_allowed(self):
        assert MorphOptions(error_floor=0.0).error_floor == 0.0


class TestGridIndependence:
    def test_initial_errors_do_not_depend_on_the_grid(self, districts):
        """The same geometry and targets give the same reported error on every grid (issue 87)."""
        results = [
            CartogramWorkflow(
                districts, "Population", options=MorphOptions(grid_size=n, show_progress=False)
            ).original.get_errors()
            for n in (64, 128, 256, 512)
        ]
        means = [e.mean_error_pct for e in results]
        assert max(means) == pytest.approx(min(means), rel=1e-12)
        np.testing.assert_allclose(results[0].log_errors, results[-1].log_errors, rtol=1e-12)

    def test_the_floor_follows_the_option(self, districts):
        small = CartogramWorkflow(districts, "Population", options=MorphOptions(error_floor=1e-6, show_progress=False))
        large = CartogramWorkflow(districts, "Population", options=MorphOptions(error_floor=0.05, show_progress=False))
        assert large.original.get_errors().mean_error_pct < small.original.get_errors().mean_error_pct

    def test_a_zero_valued_region_has_a_finite_error(self, states):
        gdf = states.copy()
        gdf.loc[0, "Population"] = 0.0
        errors = CartogramWorkflow(gdf, "Population", options=MorphOptions(show_progress=False)).original.get_errors()
        assert np.isfinite(errors.mean_error_pct) and np.isfinite(errors.max_error_pct)

    def test_no_floor_gives_an_infinite_error_for_a_zero_value(self, states):
        gdf = states.copy()
        gdf.loc[0, "Population"] = 0.0
        with np.errstate(divide="ignore"):
            errors = CartogramWorkflow(
                gdf, "Population", options=MorphOptions(error_floor=0.0, show_progress=False)
            ).original.get_errors()
        assert np.isinf(errors.max_error_pct)


class TestMorphUnchangedForNormalRegions:
    def test_states_converge_in_the_same_iterations(self, states):
        """No state is below the floor, so the run is the one the one-cell floor gave."""
        result = morph_gdf(states, "Population", options=MorphOptions(show_progress=False))
        assert result.status == MorphStatus.CONVERGED
        assert result.niterations == 112
