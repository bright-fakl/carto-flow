"""
Error metrics for morphing computation.

Classes
-------
MorphErrors
    Structured error metrics from morphing computation.

Functions
---------
compute_error_metrics
    Compute error metrics based on log2 ratio of current to target areas.
"""

from dataclasses import dataclass

import numpy as np

__all__ = [
    "MorphErrors",
    "compute_error_metrics",
]


@dataclass
class MorphErrors:
    """Structured error metrics from morphing computation.

    Attributes
    ----------
    log_errors : np.ndarray
        Array of log2 area errors for each geometry.
        Computed as log2(current_area / target_area).
        Positive values indicate oversized regions, negative values indicate
        undersized regions. This representation is symmetric: a value of +1
        means 2x too large, -1 means 2x too small.
    mean_log_error : float
        Mean of absolute log2 errors across all geometries.
    max_log_error : float
        Maximum of absolute log2 errors across all geometries.
    errors_pct : np.ndarray
        Approximate percentage error for each geometry.
        Computed as sign(log_error) * (2^|log_error| - 1) * 100.
    mean_error_pct : float
        Mean approximate percentage error across all geometries.
    max_error_pct : float
        Maximum approximate percentage error across all geometries.
    """

    log_errors: np.ndarray
    mean_log_error: float
    max_log_error: float
    errors_pct: np.ndarray
    mean_error_pct: float
    max_error_pct: float

    def __repr__(self) -> str:
        """Concise string representation for terminal display."""
        return f"MorphErrors(mean_log={self.mean_log_error:.4f}, max_log={self.max_log_error:.4f}, mean_pct={self.mean_error_pct:.1f}%, max_pct={self.max_error_pct:.1f}%)"


def compute_error_metrics(current_areas: np.ndarray, target_areas: np.ndarray) -> MorphErrors:
    """Compute error metrics based on log2 ratio of current to target areas.

    Parameters
    ----------
    current_areas : np.ndarray
        Current areas of geometries
    target_areas : np.ndarray
        Target areas

    Returns
    -------
    MorphErrors
        Structured error metrics object containing all error fields.
    """
    log_errors = np.log2(current_areas / target_areas)
    max_log_error = float(np.max(np.abs(log_errors)))
    mean_log_error = float(np.mean(np.abs(log_errors)))

    errors_pct = np.sign(log_errors) * (2 ** np.abs(log_errors) - 1) * 100
    max_error_pct = float((2**max_log_error - 1) * 100)
    mean_error_pct = float((2**mean_log_error - 1) * 100)

    return MorphErrors(
        log_errors=log_errors,
        mean_log_error=mean_log_error,
        max_log_error=max_log_error,
        errors_pct=errors_pct,
        mean_error_pct=mean_error_pct,
        max_error_pct=max_error_pct,
    )
