"""Drawing the fill of a plot of geometries with the palette of a style.

Shared by the plot functions of the cartogram types that draw ``GeoDataFrame.plot``: the data role of the fill
(sequential, diverging or categorical) selects the palette, the normalization and the colorbar of the style.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = ["Fill", "plot_fill"]


@dataclass
class Fill:
    """The resolved fill of a plot: the arguments for ``GeoDataFrame.plot``, and the data role if it is data."""

    kwargs: dict[str, Any]
    role: str | None = None
    column: str | None = None
    label: str | None = None


def plot_fill(
    ax: Axes,
    gdf: Any,
    fill: Fill,
    style: Any,
    *,
    cmap: Any,
    default_cmap: str | None,
    norm: Any,
    center: float | None,
    legend: bool,
    legend_kwargs: dict | None,
    legend_label: str | None,
    categorical: bool | None,
    values: Any = None,
    edge_default: Any = None,
    linewidth_default: Any = None,
    kwargs: dict[str, Any],
) -> None:
    """Draw ``gdf`` with its fill: the palette of the data role comes from the call, then the style, then the default.

    The edge color and width come from the call (``kwargs``), then the style, then ``edge_default`` and
    ``linewidth_default``. The colorbar is left to ``GeoDataFrame.plot`` unless the style or a ``norm`` of the call
    changes it.
    """
    from matplotlib.colors import ListedColormap, Normalize

    kw = dict(kwargs)
    for channel, key, default in (
        ("edge", "edgecolor", edge_default),
        ("linewidth", "linewidth", linewidth_default),
    ):
        value = style.channel_value(channel, default)
        if value is not None:
            kw.setdefault(key, value)
    if fill.role is None:
        gdf.plot(ax=ax, **fill.kwargs, **kw)
        return

    legend_kw = {"legend_kwds": legend_kwargs} if legend_kwargs else {}
    if fill.role == "categorical":
        colors = None
        if cmap is None:
            colors = style.categorical_colors(list(np.unique(gdf[fill.column].dropna())))
        plot_cmap = ListedColormap(colors) if colors is not None else (cmap if cmap is not None else default_cmap)
        extra: dict[str, Any] = {"categorical": True} if categorical else {}
        if plot_cmap is not None:
            extra["cmap"] = plot_cmap
        gdf.plot(ax=ax, legend=legend, **fill.kwargs, **extra, **legend_kw, **kw)
        return

    role = fill.role
    plot_cmap = cmap if cmap is not None else (style.colormap(role) or default_cmap)
    data = np.asarray(gdf[fill.column] if values is None else values, dtype=float)
    finite = data[np.isfinite(data)]
    limits = (kw.get("vmin"), kw.get("vmax"))
    norm_obj = style.resolve_norm(norm, role, data, center=center, limits=limits)
    if norm_obj is not None:  # the normalization replaces the limits of the call
        kw.pop("vmin", None)
        kw.pop("vmax", None)
    extra = {}
    if plot_cmap is not None:
        extra["cmap"] = plot_cmap
    if norm_obj is not None:
        extra["norm"] = norm_obj
    own_guide = legend and (style.draws_guide or norm_obj is not None)
    if own_guide:
        gdf.plot(ax=ax, legend=False, **fill.kwargs, **extra, **kw)
        lo = limits[0] if limits[0] is not None else finite.min()
        hi = limits[1] if limits[1] is not None else finite.max()
        guide = dict(legend_kwargs or {})
        label = guide.pop("label", legend_label if legend_label is not None else fill.label)
        style.add_colorbar(
            ax.figure, ax, plot_cmap, norm_obj if norm_obj is not None else Normalize(vmin=lo, vmax=hi), label, **guide
        )
    else:
        gdf.plot(ax=ax, legend=legend, **fill.kwargs, **extra, **legend_kw, **kw)
