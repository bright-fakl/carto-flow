"""Step-size control: shrink ``dt`` when the area errors alternate in sign."""

from collections import deque

import numpy as np

__all__ = ["DEAD_BAND", "FLOOR", "RESTART_SCORE", "STRENGTH", "THRESHOLD", "WINDOW", "StepController"]

DEAD_BAND = 1e-3
"""Smallest ``|log2(area / target)|`` that counts as an error with a sign."""

WINDOW = 3
"""Number of iterations over which a sign change of a region is remembered."""

THRESHOLD = 0.3
"""Share of regions with a recent sign change above which the step shrinks."""

STRENGTH = 0.6
"""The step is multiplied by ``1 - STRENGTH * share`` (share of regions with a recent sign change)."""

FLOOR = 0.125
"""Smallest fraction of ``dt``."""

RESTART_SCORE = 10.0
"""Score below which the first steps of a run are scaled in proportion to the score."""


class StepController:
    """Fraction of ``dt`` to use, decreasing whenever the step overshoots the target.

    The step moves the fastest point ``dt`` grid cells whatever the error. Near the target that is
    more than the correction asks for: the area of each region then alternates around its target
    and the sign of its error changes from one iteration to the next. A region counts as inverted
    if the sign of its error changed in any of the last ``WINDOW`` iterations; a change is only
    recorded between two iterations in which the region's error exceeds ``DEAD_BAND``. Far from
    the target almost no region inverts. When the share of inverted regions exceeds ``THRESHOLD``
    the fraction is multiplied by ``1 - STRENGTH * share`` (not below ``FLOOR``); one shrink is
    followed by ``WINDOW`` iterations without another, so a sign change shrinks the step once. It never grows
    back, and it does not depend on the tolerances or on when the field is recomputed.

    A run that starts from an existing geometry near the target (a refinement, a multiresolution
    level) would take a first step that moves the fastest point ``dt`` cells and undo it, before any
    sign change can be seen. During the first ``window`` iterations the fraction is therefore also
    limited to the score of the current state divided by ``RESTART_SCORE`` (not below ``FLOOR``);
    a run that starts far from the target (score above ``RESTART_SCORE``) is not affected.

    Parameters
    ----------
    window : int
        Number of iterations of the restart limit.
    score : float
        Score of the starting geometries (below 1 is converged).
    enabled : bool
        If False, the fraction is always 1.
    """

    def __init__(self, window: int, score: float, enabled: bool = True):
        self.window = window
        self.score = score
        self.enabled = enabled
        self.fraction = 1.0
        self._previous: np.ndarray | None = None
        self._recent: deque[np.ndarray] = deque(maxlen=WINDOW)
        self._pause = 0

    def current(self, step: int) -> float:
        """Fraction of ``dt`` for iteration ``step`` (0-based)."""
        if not self.enabled:
            return 1.0
        if step < self.window:
            return min(self.fraction, max(self.score / RESTART_SCORE, FLOOR))
        return self.fraction

    def update(self, log_errors: np.ndarray, score: float) -> None:
        """Record the signed log2 errors (area over target) and the score of the state reached by the last iteration."""
        if not self.enabled:
            return
        self.score = score
        if self._previous is None:
            self._recent.append(np.zeros(len(log_errors), dtype=bool))
        else:
            significant = (np.abs(log_errors) > DEAD_BAND) & (np.abs(self._previous) > DEAD_BAND)
            self._recent.append(significant & (np.sign(log_errors) != np.sign(self._previous)))
        self._previous = log_errors
        if self._pause > 0:
            self._pause -= 1
            return
        share = float(np.any(self._recent, axis=0).mean())
        if share > THRESHOLD:
            self.fraction = max(self.fraction * (1.0 - STRENGTH * min(share, 1.0)), FLOOR)
            self._pause = WINDOW
