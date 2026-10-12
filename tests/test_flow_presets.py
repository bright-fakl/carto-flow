"""The MorphOptions presets converge on the bundled states and congressional districts."""

from __future__ import annotations

import pytest

from carto_flow import data
from carto_flow.flow_cartogram import MorphOptions, MorphStatus, morph_gdf


@pytest.fixture(scope="module")
def inputs():
    return {
        "states": data.load_us_census(population=True).reset_index(drop=True),
        "districts": data.load_us_census(population=True, level="congressional_district").reset_index(drop=True),
    }


PRESETS = {
    "fast": MorphOptions.preset_fast,
    "balanced": MorphOptions.preset_balanced,
    "high_quality": MorphOptions.preset_high_quality,
}


class TestPresets:
    @pytest.mark.parametrize("name", ["states", "districts"])
    @pytest.mark.parametrize("preset", list(PRESETS))
    def test_the_preset_converges_within_its_tolerances(self, inputs, preset, name):
        options = PRESETS[preset]().copy_with(show_progress=False)
        result = morph_gdf(inputs[name], "Population", options=options)
        errors = result.get_errors()
        assert result.status == MorphStatus.CONVERGED
        assert result.niterations < options.n_iter
        assert errors.mean_error_pct <= 100 * options.mean_tol
        assert errors.max_error_pct <= 100 * options.max_tol

    def test_balanced_has_the_default_values(self):
        balanced, default = MorphOptions.preset_balanced(), MorphOptions()
        for field in ("grid_size", "n_iter", "dt", "mean_tol", "max_tol"):
            assert getattr(balanced, field) == getattr(default, field)

    def test_the_presets_get_finer_and_stricter(self):
        fast, balanced, high = (PRESETS[k]() for k in ("fast", "balanced", "high_quality"))
        assert fast.grid_size < balanced.grid_size < high.grid_size
        assert fast.mean_tol > balanced.mean_tol > high.mean_tol
        assert fast.max_tol > balanced.max_tol > high.max_tol
