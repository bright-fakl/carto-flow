"""Plot of a non-contiguous cartogram."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._fill import Fill, plot_fill
from .._plot_params import resolve_fill
from ..style import resolve_style, styled
from .result import NoncontiguousPlotResult

if TYPE_CHECKING:
    from .result import NoncontiguousCartogram

__all__ = ["plot_cartogram"]


def _fill_plan(
    gdf: Any, facecolor: Any, categorical: bool | None, center: float | None, default: str, order: Any
) -> tuple[Any, Fill]:
    """The regions in drawing order (``order`` is a permutation of the rows) and their ``Fill``: a color, or data with
    its role."""
    import pandas as pd

    kind, value = resolve_fill(facecolor, set(gdf.columns), (), n=len(gdf), stacklevel=5)
    if kind == "none":
        return gdf.iloc[order], Fill({"facecolor": default})
    if kind == "color":
        if not isinstance(value, str) and np.ndim(value) > 0 and len(value) == len(gdf):  # one color per region
            value = [value[i] for i in order]
        return gdf.iloc[order], Fill({"facecolor": value})
    column, label = (value, value)
    if kind == "values":
        gdf = gdf.copy()
        gdf["_facecolor"] = value
        column, label = "_facecolor", None
    gdf = gdf.iloc[order]
    numeric = pd.api.types.is_numeric_dtype(gdf[column]) and not pd.api.types.is_bool_dtype(gdf[column])
    if categorical is True or (categorical is None and not numeric):
        role = "categorical"
    else:
        role = "diverging" if center is not None else "sequential"
    return gdf, Fill({"column": column}, role, column, label)


@styled
def plot_cartogram(
    cartogram: NoncontiguousCartogram,
    facecolor: Any = None,
    cmap: Any = None,
    show_original: bool = True,
    legend: bool = True,
    legend_kwargs: dict | None = None,
    legend_label: str | None = None,
    norm: Any = None,
    center: float | None = None,
    categorical: bool | None = None,
    ax: Any | None = None,
    **kwargs: Any,
) -> NoncontiguousPlotResult:
    """Plot the scaled regions over the outlines of the original regions.

    Parameters
    ----------
    cartogram : NoncontiguousCartogram
        The cartogram to plot.
    facecolor : str, array-like, or None
        Fill of the regions. ``None`` (default) fills them with the fill value of the style. A column of the
        input (or ``"_ncont_scale"``, the scale factor of the regions) gives a choropleth, a numeric array with one
        value per region is colored by value, and a color (a string or an array of colors) fills all regions. A string
        that is a column and also a color is read as the column, and a warning is issued.
    cmap : str, Colormap, dict, or None
        Colormap of the data. Defaults to the palette of the style for the data role (categorical colors for text
        data, sequential for numbers, diverging with ``center``), else ``"viridis"``.
    show_original : bool
        Draw the original regions under the scaled ones, with the reference fill and edge of the style. Default
        ``True``.
    legend : bool
        Show a legend or colorbar for data. Default ``True``.
    legend_kwargs : dict, optional
        Keyword arguments of the legend or colorbar. A ``"label"`` key sets the label of a colorbar.
    legend_label : str, optional
        Label of the colorbar. Defaults to the name of the data.
    norm : Normalize, {"linear", "sqrt", "log", "auto"}, optional
        Normalization of continuous data: an instance, or the scale of the style's normalization.
    center : float, optional
        Treat the data as diverging around this value.
    categorical : bool, optional
        Force (True) or prevent (False) the categorical treatment of the data; by default the data type decides.
    ax : matplotlib Axes, optional
        Axes to draw on. A new figure is created if ``None``.
    **kwargs
        Passed to ``GeoDataFrame.plot()`` for the scaled regions (for example ``edgecolor`` and ``linewidth``, which
        default to the edge and line width of the style, else dark gray and 0.5).

    Notes
    -----
    The regions are drawn in order of decreasing area, so that a region that grew over its neighbors (with a
    ``max_scale`` above 1) lies under the smaller ones and does not hide them.

    Returns
    -------
    NoncontiguousPlotResult
        Dataclass with ``ax``, ``collections`` and ``original_collections``.
    """
    import matplotlib.pyplot as plt

    st = resolve_style()
    if ax is None:
        _, ax = plt.subplots()

    original_collections: list = []
    if show_original:
        before = len(ax.collections)
        cartogram.source.plot(
            ax=ax,
            facecolor=st.pick("reference_facecolor", "#f0f0f0"),
            edgecolor=st.pick("reference_edgecolor", "#bbbbbb"),
            linewidth=st.pick("reference_linewidth", 0.5),
            alpha=st.pick("reference_alpha", 1.0),
            zorder=0,
        )
        original_collections = list(ax.collections[before:])

    before = len(ax.collections)
    # the largest regions are drawn first, so that a region that grew over its neighbors does not hide them
    order = np.argsort(-np.asarray(cartogram.geometries.geometry.area, dtype=float), kind="stable")
    gdf, fill = _fill_plan(
        cartogram.geometries, facecolor, categorical, center, st.channel_value("fill", "steelblue"), order
    )
    plot_fill(
        ax,
        gdf,
        fill,
        st,
        cmap=cmap,
        default_cmap="viridis",
        norm=norm,
        center=center,
        legend=legend,
        legend_kwargs=legend_kwargs,
        legend_label=legend_label,
        categorical=categorical,
        edge_default="#333333",
        linewidth_default=0.5,
        kwargs=dict(kwargs),
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    return NoncontiguousPlotResult(
        ax=ax,
        collections=list(ax.collections[before:]),
        original_collections=original_collections,
    )
