"""Plot of the allocated tile set against the assigned tiles of a grid layout."""

from __future__ import annotations

import numpy as np


def plot_tile_sets(result, ax=None, labels=None):
    """Draw the lattice with the allocated tiles shaded and the assigned tiles outlined.

    The allocated tiles are those the assignment may use
    (``result.metrics.algorithm.allowed_tiles``); the assigned tiles are
    ``result.assignments``. Allocated tiles without a region are filled red.

    Parameters
    ----------
    result : GridLayoutResult
        Result of ``create_layout`` with a ``GridBasedLayout``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure by default.
    labels : sequence of str, optional
        One label per region, written on its tile.

    Returns
    -------
    matplotlib.axes.Axes

    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    ax = ax if ax is not None else plt.subplots()[1]
    tr = result.tiling_result
    allowed = np.zeros(tr.n_tiles, dtype=bool)
    allowed[result.metrics.algorithm.allowed_tiles] = True
    assigned = np.zeros(tr.n_tiles, dtype=bool)
    assigned[np.asarray(result.assignments)] = True
    for t, poly in enumerate(tr.polygons):
        color = "#f3f3f3" if not allowed[t] else ("#9ecae1" if assigned[t] else "#d62728")
        ax.add_patch(
            Polygon(
                np.asarray(poly.exterior.coords),
                fc=color,
                ec="k" if assigned[t] else "#cccccc",
                lw=1.0 if assigned[t] else 0.3,
            )
        )
    if labels is not None:
        labels = list(labels)
        for i, t in enumerate(np.asarray(result.assignments)):
            ax.text(*tr.centers[t], str(labels[i]), ha="center", va="center", fontsize=5)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    return ax
