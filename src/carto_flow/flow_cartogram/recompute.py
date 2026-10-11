"""Decision of when the velocity field is recomputed."""

__all__ = ["score_rose", "should_recompute"]


def score_rose(score: float, previous_score: float, recompute_on_rise: float | None) -> bool:
    """Whether ``score`` exceeds ``previous_score`` by more than the relative amount ``recompute_on_rise``."""
    return recompute_on_rise is not None and score > previous_score * (1.0 + recompute_on_rise)


def should_recompute(
    step: int,
    iterations_since_recompute: int,
    rose: bool,
    recompute_every: int | None,
    recompute_on_rise: float | None,
) -> bool:
    """Whether the velocity field is recomputed before iteration ``step`` (0-based).

    Without ``recompute_on_rise`` the field is recomputed on the fixed schedule
    (``step % recompute_every == 0``). With it, the field is recomputed at the first
    iteration, after ``recompute_every`` iterations with the same field (never, if
    None), and as soon as the field has been used for one iteration and the score
    rose (``rose``) in the last one.
    """
    if step == 0:
        return True
    if recompute_on_rise is None:
        return recompute_every is not None and step % recompute_every == 0
    if recompute_every is not None and iterations_since_recompute >= recompute_every:
        return True
    return rose and iterations_since_recompute >= 1
