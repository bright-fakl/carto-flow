"""Decision of when the velocity field is recomputed."""

__all__ = ["score_rose", "should_refresh"]


def score_rose(score: float, previous_score: float, refresh_on_rise: float | None) -> bool:
    """Whether ``score`` exceeds ``previous_score`` by more than the relative amount ``refresh_on_rise``."""
    return refresh_on_rise is not None and score > previous_score * (1.0 + refresh_on_rise)


def should_refresh(
    step: int,
    iterations_since_refresh: int,
    rose: bool,
    recompute_every: int | None,
    refresh_on_rise: float | None,
) -> bool:
    """Whether the velocity field is recomputed before iteration ``step`` (0-based).

    Without ``refresh_on_rise`` the field is refreshed on the fixed schedule
    (``step % recompute_every == 0``). With it, the field is refreshed at the first
    iteration, after ``recompute_every`` iterations with the same field (never, if
    None), and as soon as the field has been used for one iteration and the score
    rose (``rose``) in the last one.
    """
    if step == 0:
        return True
    if refresh_on_rise is None:
        return recompute_every is not None and step % recompute_every == 0
    if recompute_every is not None and iterations_since_refresh >= recompute_every:
        return True
    return rose and iterations_since_refresh >= 1
