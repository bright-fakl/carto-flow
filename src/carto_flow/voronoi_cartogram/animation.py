"""Animation utilities for Voronoi cartogram results.

Functions
---------
animate_voronoi_history
    Animate Lloyd relaxation through per-iteration snapshots.
save_animation
    Save a matplotlib animation to file (GIF, MP4, etc.).

Examples
--------
>>> from carto_flow.voronoi_cartogram import create_voronoi_cartogram, VoronoiOptions
>>> from carto_flow.voronoi_cartogram.animation import animate_voronoi_history, save_animation
>>> options = VoronoiOptions(record_history=True, record_cells=True, n_iter=50)
>>> result = create_voronoi_cartogram(gdf, options=options)
>>> anim = animate_voronoi_history(result, fps=10)
>>> save_animation(anim, "voronoi.gif", fps=10)
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .._plot_params import deprecated_params

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation

    from .result import VoronoiCartogram

__all__ = [
    "animate_voronoi_history",
    "save_animation",
]


@deprecated_params(cell_kwargs="cell_style", point_kwargs="point_style")
def animate_voronoi_history(
    result: VoronoiCartogram,
    *,
    show_cells: bool = True,
    show_points: bool = True,
    show_convergence_title: bool = True,
    cell_kwargs: dict | None = None,
    point_kwargs: dict | None = None,
    figsize: tuple[float, float] = (8, 6),
    fps: int = 15,
    repeat: bool = True,
) -> FuncAnimation:
    """Animate Lloyd relaxation through per-iteration snapshots.

    Requires ``VoronoiOptions(record_history=True)`` when running
    :func:`~carto_flow.voronoi_cartogram.create_voronoi_cartogram`.  Set
    ``record_cells=True`` to also animate cell polygons.

    Parameters
    ----------
    result : VoronoiCartogram
        The cartogram with a populated :attr:`~VoronoiCartogram.history`.
    show_cells : bool
        Animate cell polygons if they were recorded (``record_cells=True``).
        Falls back to point scatter only when cells are unavailable.
        Default ``True``.
    show_points : bool
        Overlay generator point positions.  Default ``True``.
    show_convergence_title : bool
        Include iteration number and CV(area) in the title.  Default ``True``.
    cell_kwargs : dict or None
        Styling for cell polygons, merged over defaults
        ``{"facecolor": "#d0e4f7", "edgecolor": "#1a5276", "linewidth": 0.4, "alpha": 0.7}``.
        All keys forwarded to ``GeoDataFrame.plot()``.
    point_kwargs : dict or None
        Styling for generator points, merged over defaults
        ``{"s": 14, "c": "#e74c3c", "zorder": 3, "linewidths": 0}``.
        All keys forwarded to ``ax.scatter()``.
    figsize : tuple of float
        Figure size.
    fps : int
        Frames per second (used as a hint for the interval; pass to
        ``save_animation`` separately to control output fps).
    repeat : bool
        Whether the animation loops.  Default ``True``.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Raises
    ------
    RuntimeError
        If ``result.history`` is ``None`` (history was not recorded).
    """
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if result.history is None:
        raise RuntimeError("result.history is None — re-run with VoronoiOptions(record_history=True)")

    snapshots = list(result.history)
    if not snapshots:
        raise RuntimeError("result.history is empty — no snapshots were recorded")

    # Merge style defaults
    _cell_kw: dict = {
        "facecolor": "#d0e4f7",
        "edgecolor": "#1a5276",
        "linewidth": 0.4,
        "alpha": 0.7,
        **(cell_kwargs or {}),
    }
    _pt_kw: dict = {
        "s": 14,
        "c": "#e74c3c",
        "zorder": 3,
        "linewidths": 0,
        **(point_kwargs or {}),
    }

    # Determine whether cell animation is possible
    has_cells = show_cells and any(s.cells is not None for s in snapshots)

    # Compute axis bounds from final positions + small margin
    all_pos = np.vstack([s.positions for s in snapshots])
    x_min, y_min = all_pos.min(axis=0)
    x_max, y_max = all_pos.max(axis=0)
    pad_x = (x_max - x_min) * 0.05 or 1.0
    pad_y = (y_max - y_min) * 0.05 or 1.0

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)

    # Get CRS from source GDF if available
    crs = getattr(getattr(result, "_source_gdf", None), "crs", None)

    # Artists updated each frame
    _scatter: list[Any] = [None]
    _collection_list: list = []

    def _draw_frame(snap: Any) -> None:
        # Clear previous cell patches
        for coll in _collection_list:
            with contextlib.suppress(Exception):
                coll.remove()
        _collection_list.clear()

        # Draw cells
        if has_cells and snap.cells is not None:
            gdf_cells = gpd.GeoDataFrame(geometry=list(snap.cells), crs=crs)
            gdf_cells.plot(ax=ax, **_cell_kw)
            _collection_list.extend(ax.collections[-len(gdf_cells) :])

        # Draw / update scatter
        if show_points:
            if _scatter[0] is not None:
                _scatter[0].remove()
            _scatter[0] = ax.scatter(snap.positions[:, 0], snap.positions[:, 1], **_pt_kw)

        # Title
        if show_convergence_title:
            ax.set_title(
                f"Iteration {snap.iteration}  —  CV(area) = {snap.area_cv:.4f}",
                fontsize=10,
            )

    # Draw first frame immediately so the figure is not blank
    _draw_frame(snapshots[0])

    def _update(frame_idx: int) -> None:
        _draw_frame(snapshots[frame_idx])

    interval_ms = max(1, round(1000 / fps))
    anim = FuncAnimation(
        fig,
        _update,  # type: ignore[arg-type]
        frames=len(snapshots),
        interval=interval_ms,
        repeat=repeat,
        blit=False,
    )
    return anim


def save_animation(
    anim: FuncAnimation,
    path: str | Path,
    *,
    fps: int = 15,
    dpi: int = 100,
    writer: str | None = None,
    **kwargs: Any,
) -> None:
    """Save a matplotlib animation to file.

    Parameters
    ----------
    anim : FuncAnimation
        The animation to save (e.g. from :func:`animate_voronoi_history`).
    path : str or Path
        Output path.  The format is inferred from the file extension:
        ``.gif`` uses Pillow, ``.mp4`` / ``.mov`` use ffmpeg,
        ``.html`` produces an HTML widget.
    fps : int
        Frames per second.  Default ``15``.
    dpi : int
        Dots per inch.  Default ``100``.
    writer : str or None
        Explicit matplotlib writer name (e.g. ``"pillow"``, ``"ffmpeg"``).
        ``None`` (default) infers from the file extension.
    **kwargs
        Additional keyword arguments forwarded to
        ``FuncAnimation.save()``.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if writer is None:
        if suffix == ".gif":
            writer = "pillow"
        elif suffix in (".mp4", ".mov", ".avi"):
            writer = "ffmpeg"
        elif suffix == ".html":
            writer = "html"
        # else: let matplotlib decide

    save_kwargs: dict = {"fps": fps, "dpi": dpi}
    if writer is not None:
        save_kwargs["writer"] = writer
    save_kwargs.update(kwargs)

    anim.save(str(path), **save_kwargs)
