"""Tests for carto_flow.flow_cartogram.errors.compute_error_metrics."""

import numpy as np
import pytest

from carto_flow.flow_cartogram.errors import compute_error_metrics


class TestEpsFloor:
    """The ``eps`` floor makes zero-area regions measurable instead of infinite."""

    def test_default_eps_reproduces_unclamped_ratio(self):
        """With eps=0 (the default) behaviour is exactly the historical, unclamped ratio."""
        current = np.array([4.0, 2.0, 1.0])
        target = np.array([2.0, 2.0, 2.0])
        result = compute_error_metrics(current, target)
        np.testing.assert_allclose(result.log_errors, np.log2(current / target))

    def test_zero_target_area_is_inf_without_eps(self):
        current = np.array([1.0])
        target = np.array([0.0])
        with np.errstate(divide="ignore"):
            result = compute_error_metrics(current, target)
        assert np.isinf(result.max_log_error)

    def test_zero_target_area_is_finite_with_eps(self):
        """A zero target area no longer produces inf once floored by a cell area."""
        current = np.array([1.0])
        target = np.array([0.0])
        result = compute_error_metrics(current, target, eps=0.01)
        assert np.isfinite(result.log_errors[0])
        assert np.isfinite(result.max_log_error)
        assert np.isfinite(result.mean_log_error)

    def test_zero_target_error_shrinks_as_current_shrinks(self):
        """A zero-target region's error falls as it shrinks, reaching zero below eps."""
        eps = 0.01
        target = np.array([0.0])
        big = compute_error_metrics(np.array([1.0]), target, eps=eps).log_errors[0]
        small = compute_error_metrics(np.array([0.02]), target, eps=eps).log_errors[0]
        tiny = compute_error_metrics(np.array([0.005]), target, eps=eps).log_errors[0]
        assert big > small > 0
        # Once current area is below eps, both sides are floored to eps -> zero error.
        assert tiny == 0.0

    def test_mirror_case_zero_current_area_is_finite_with_eps(self):
        """A region whose *current* area has collapsed no longer gives -inf."""
        current = np.array([0.0])
        target = np.array([1.0])
        result = compute_error_metrics(current, target, eps=0.01)
        assert np.isfinite(result.log_errors[0])
        assert result.log_errors[0] < 0

    def test_mirror_case_is_minus_inf_without_eps(self):
        current = np.array([0.0])
        target = np.array([1.0])
        with np.errstate(divide="ignore"):
            result = compute_error_metrics(current, target)
        assert np.isneginf(result.log_errors[0])

    def test_no_op_for_areas_above_eps(self):
        """Areas well above the eps floor are completely unaffected by clamping."""
        current = np.array([100.0, 50.0, 10.0])
        target = np.array([80.0, 60.0, 12.0])
        eps = 0.01
        clamped = compute_error_metrics(current, target, eps=eps)
        unclamped = compute_error_metrics(current, target, eps=0.0)
        np.testing.assert_array_equal(clamped.log_errors, unclamped.log_errors)
        assert clamped.max_log_error == unclamped.max_log_error
        assert clamped.mean_log_error == unclamped.mean_log_error

    def test_eps_boundary_is_exact_no_op(self):
        """An area exactly at eps is unaffected (maximum is a no-op at the boundary)."""
        current = np.array([0.01])
        target = np.array([0.02])
        result = compute_error_metrics(current, target, eps=0.01)
        assert result.log_errors[0] == pytest.approx(np.log2(0.01 / 0.02))
