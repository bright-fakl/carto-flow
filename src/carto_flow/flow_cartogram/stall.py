"""Stall detection on windows of iterations, judged by their minima."""

import numpy as np

__all__ = ["MIN_CYCLE_LENGTH", "StallMonitor", "cycle_length", "region_size"]

MIN_CYCLE_LENGTH = 10
"""Shortest cycle, in iterations, used for stall detection."""


def region_size(target_areas: np.ndarray) -> float:
    """Typical linear size of the target regions: the square root of their area-weighted mean area."""
    return float(np.sqrt(np.sum(target_areas**2) / np.sum(target_areas)))


def cycle_length(recompute_every: int | None) -> int:
    """Number of iterations in one stall-detection cycle.

    A cycle is a window of ``recompute_every`` iterations, counted from the first
    iteration, but never shorter than ``MIN_CYCLE_LENGTH`` iterations, so that
    frequent recomputes do not make a cycle a single noisy iteration. Without a
    recompute interval (``None``) cycles have the minimum length. Cycles do not
    depend on when recomputes actually happen.
    """
    return max(recompute_every or 0, MIN_CYCLE_LENGTH)


class StallMonitor:
    """Counts consecutive cycles without progress.

    Each cycle of ``cycle`` iterations records the minimum of the mean-error
    ratio (``mean_error / mean_tol``) and of the max-error ratio
    (``max_error / max_tol``); a ratio below 1 means that component is
    satisfied. A component counts toward progress in a cycle only if its cycle
    minimum is still above 1 and lower than that component's best over the
    earlier cycles by at least the fraction ``min_efficiency * moved`` of that best, where ``moved`` is the distance
    the boundaries moved in the cycle relative to the size of the regions.
    A cycle is progress if any component counts. Improvements of a satisfied
    component, and a component that regressed above 1 after having been
    satisfied, do not count. The run is stalled after ``patience`` consecutive
    cycles without progress. A final incomplete cycle is not judged.

    Parameters
    ----------
    patience : int or None
        Consecutive cycles without progress that make the run stalled. None
        disables stall detection.
    min_efficiency : float
        Required relative improvement of a cycle minimum per unit of ``moved``.
    cycle : int
        Iterations per cycle.
    """

    def __init__(self, patience: int | None, min_efficiency: float, cycle: int):
        self.patience = patience
        self.min_efficiency = min_efficiency
        self.cycle = cycle
        self.best_mean = float("inf")
        self.best_max = float("inf")
        self.cycles_without_progress = 0
        self._cur_mean = float("inf")
        self._cur_max = float("inf")
        self._cur_moved = 0.0

    def update(self, step: int, mean_ratio: float, max_ratio: float, moved: float) -> bool:
        """Record iteration ``step`` (0-based), in which the boundaries moved ``moved``; return True if the run is stalled.

        ``moved`` is the distance of the iteration divided by the size of the regions (see ``region_size``).
        """
        self._cur_mean = min(self._cur_mean, mean_ratio)
        self._cur_max = min(self._cur_max, max_ratio)
        self._cur_moved += moved
        if (step + 1) % self.cycle != 0:
            return False

        keep = 1.0 - self.min_efficiency * self._cur_moved
        progress = (self._cur_mean > 1.0 and self._cur_mean < self.best_mean * keep) or (
            self._cur_max > 1.0 and self._cur_max < self.best_max * keep
        )
        self.best_mean = min(self.best_mean, self._cur_mean)
        self.best_max = min(self.best_max, self._cur_max)
        self._cur_mean = self._cur_max = float("inf")
        self._cur_moved = 0.0
        self.cycles_without_progress = 0 if progress else self.cycles_without_progress + 1
        return (
            self.patience is not None
            and self.cycles_without_progress > 0
            and self.cycles_without_progress >= self.patience
        )
