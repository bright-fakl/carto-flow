"""Visualization utilities for Voronoi cartogram results.

Functions
---------
plot_cartogram
    Plot Voronoi cells, optionally colored by a data column.
plot_comparison
    Side-by-side comparison of original geometries vs Voronoi cells.
plot_convergence
    Plot CV(area) over Lloyd iterations.
plot_displacement
    Centroids and displacement arrows on a single axis.
plot_topology
    Topology issues (satellites, violated adjacency, misaligned orientation)
    overlaid on the Voronoi cells.
plot_topology_repair
    Before/after comparison of a topology repair operation.
plot_compactness
    Per-cell inertia (distance to group centroid) as a choropleth — shows
    which cells the compactness repair considers outliers.

Examples
--------
>>> from carto_flow.voronoi_cartogram import create_voronoi_cartogram
>>> from carto_flow.voronoi_cartogram.visualization import plot_comparison, plot_convergence
>>> result = create_voronoi_cartogram(gdf)
>>> plot_comparison(gdf, result)
>>> plot_convergence(result)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    import geopandas as gpd

    from .result import TopologyAnalysis, TopologyRepairReport, VoronoiCartogram, VoronoiPlotResult

__all__ = [
    "plot_cartogram",
    "plot_compactness",
    "plot_comparison",
    "plot_convergence",
    "plot_displacement",
    "plot_topology",
    "plot_topology_repair",
]


# ---------------------------------------------------------------------------
# plot_cartogram
# ---------------------------------------------------------------------------


def plot_cartogram(
    result: VoronoiCartogram,
    column: str | None = None,
    cmap: str | None = None,
    show_edges: bool = True,
    legend: bool = False,
    labels: bool | str | list[str] | None = None,
    label_fontsize: int = 8,
    label_color: str = "black",
    label_location: str = "centroid",
    ax: Any | None = None,
    **kwargs: Any,
) -> VoronoiPlotResult:
    """Plot Voronoi cells.

    Parameters
    ----------
    result : VoronoiCartogram
        The cartogram to plot.
    column : str or None
        Column from the source GeoDataFrame to use for choropleth coloring.
        ``None`` (default) colors each cell with a distinct categorical
        color derived from its index.  Pass ``"area_error_pct"`` to plot the
        signed per-cell area error as a diverging choropleth.
    cmap : str or None
        Colormap name.  Defaults to ``"tab20"`` for categorical coloring
        (``column=None``), ``"RdBu_r"`` for ``"area_error_pct"``, and
        ``"viridis"`` for other numeric columns.
    show_edges : bool
        Draw cell borders.  Default ``True``.
    legend : bool
        Show a legend or colorbar.  Default ``False``.
    labels : bool, str, list of str, or None
        Annotate each cell with a text label.

        * ``True`` — use the source GeoDataFrame index values.
        * ``str`` — column name in the source GeoDataFrame.
        * ``list[str]`` — explicit label strings (length must equal number of
          cells).
        * ``None`` / ``False`` — no labels (default).
    label_fontsize : int
        Font size for cell labels.  Default ``8``.
    label_color : str
        Text color for cell labels.  Default ``"black"``.
    label_location : str
        Where to place labels within each cell.  Default ``"centroid"`` (the
        geometric centroid).  Other options are ``"representative"`` (a
        representative point guaranteed to be inside the cell), or ``"generator"``
        (the Voronoi cell generator point).
    ax : matplotlib Axes or None
        Axes to draw on.  A new figure is created if ``None``.
    **kwargs
        Passed to ``GeoDataFrame.plot()``.

    Returns
    -------
    VoronoiPlotResult
        Dataclass with ``ax`` and ``collections``.
    """
    import geopandas as gpd
    import matplotlib.pyplot as plt

    from .result import VoronoiPlotResult

    if ax is None:
        _, ax = plt.subplots()

    edge_color = "black" if show_edges else "none"
    linewidth = 0.4 if show_edges else 0

    gdf_cells = gpd.GeoDataFrame(
        {"_idx": np.arange(len(result.cells))},
        geometry=list(result.cells),
        crs=result._source_gdf.crs if result._source_gdf is not None else None,
    )

    if column is None:
        gdf_cells.plot(
            ax=ax,
            column="_idx",
            categorical=True,
            cmap=cmap or "tab20",
            edgecolor=edge_color,
            linewidth=linewidth,
            legend=legend,
            **kwargs,
        )
    elif column == "area_error_pct" and result.area_errors is not None:
        gdf_cells["area_error_pct"] = result.area_errors
        gdf_cells.plot(
            ax=ax,
            column="area_error_pct",
            cmap=cmap or "RdBu_r",
            edgecolor=edge_color,
            linewidth=linewidth,
            legend=legend,
            **kwargs,
        )
    else:
        if result._source_gdf is None or column not in result._source_gdf.columns:
            raise ValueError(f"Column {column!r} not found in source GeoDataFrame.")
        gdf_cells[column] = result._source_gdf[column].values
        gdf_cells.plot(
            ax=ax,
            column=column,
            cmap=cmap or "viridis",
            edgecolor=edge_color,
            linewidth=linewidth,
            legend=legend,
            **kwargs,
        )

    if labels is not None and labels is not False:
        if labels is True:
            label_texts = (
                [str(x) for x in result._source_gdf.index]
                if result._source_gdf is not None
                else [str(i) for i in range(len(result.cells))]
            )
        elif isinstance(labels, str):
            if result._source_gdf is None or labels not in result._source_gdf.columns:
                raise ValueError(f"labels column {labels!r} not found in source GeoDataFrame.")
            label_texts = [str(x) for x in result._source_gdf[labels]]
        else:
            label_texts = [str(x) for x in labels]
        for idx, (cell, txt) in enumerate(zip(result.cells, label_texts, strict=False)):
            pt: tuple | np.ndarray
            if label_location == "centroid":
                pt = (cell.centroid.x, cell.centroid.y)
            elif label_location == "representative":
                _rp = cell.representative_point()
                pt = (_rp.x, _rp.y)
            elif label_location == "generator":
                pt = result.positions[idx, :]
            ax.annotate(
                txt,
                xy=pt,  # type: ignore[arg-type]
                ha="center",
                va="center",
                fontsize=label_fontsize,
                color=label_color,
            )

    ax.set_axis_off()
    return VoronoiPlotResult(ax=ax, collections=list(ax.collections))


# ---------------------------------------------------------------------------
# plot_comparison
# ---------------------------------------------------------------------------


def plot_comparison(
    gdf: gpd.GeoDataFrame,
    result: VoronoiCartogram,
    *,
    column: str | None = None,
    cmap: str | None = None,
    axes: Any | None = None,
    figsize: tuple[float, float] = (14, 6),
    title: str | None = None,
) -> VoronoiPlotResult:
    """Side-by-side comparison of original geometries vs Voronoi cells.

    Parameters
    ----------
    gdf : GeoDataFrame
        Original input geometries.
    result : VoronoiCartogram
        The cartogram to compare against.
    column : str or None
        Column to use for choropleth coloring on both panels.  ``None``
        uses categorical index coloring.
    cmap : str or None
        Colormap name.  Defaults to ``"tab20"`` (categorical) or
        ``"viridis"`` (numeric).
    axes : sequence of two Axes, or None
        Existing axes to draw on.  A new ``(1, 2)`` figure is created when
        ``None``.
    figsize : tuple of float
        Figure size when ``axes`` is ``None``.
    title : str or None
        Figure suptitle.  Defaults to ``"Original vs Voronoi Cartogram"``.

    Returns
    -------
    VoronoiPlotResult
        ``ax`` is a tuple ``(ax_original, ax_cartogram)``.
    """
    import matplotlib.pyplot as plt

    from .result import VoronoiPlotResult

    if axes is not None:
        ax_orig, ax_cart = next(iter(axes)), list(axes)[1]
    else:
        _, (ax_orig, ax_cart) = plt.subplots(1, 2, figsize=figsize)

    edge_kw = {"edgecolor": "black", "linewidth": 0.4}

    if column is None:
        gdf_plot = gdf.copy()
        gdf_plot["_idx"] = np.arange(len(gdf))
        gdf_plot.plot(ax=ax_orig, column="_idx", categorical=True, cmap=cmap or "tab20", **edge_kw)
    else:
        gdf.plot(ax=ax_orig, column=column, cmap=cmap or "viridis", **edge_kw)

    ax_orig.set_title("Original", fontsize=10)
    ax_orig.set_axis_off()
    ax_orig.set_aspect("equal")

    plot_cartogram(result, column=column, cmap=cmap, ax=ax_cart)
    ax_cart.set_title("Voronoi Cartogram", fontsize=10)
    ax_cart.set_aspect("equal")

    if title is None:
        title = "Original vs Voronoi Cartogram"
    ax_orig.figure.suptitle(title, fontsize=12)

    return VoronoiPlotResult(ax=(ax_orig, ax_cart), collections=list(ax_cart.collections))


# ---------------------------------------------------------------------------
# plot_convergence
# ---------------------------------------------------------------------------


def plot_convergence(
    result: VoronoiCartogram,
    *,
    ax: Any | None = None,
    figsize: tuple[float, float] = (7, 3),
    title: str = "Lloyd relaxation convergence",
    log_scale: bool = False,
) -> VoronoiPlotResult:
    """Plot CV(area) over Lloyd iterations.

    Parameters
    ----------
    result : VoronoiCartogram
        The cartogram whose :attr:`~VoronoiCartogram.convergence_history`
        is plotted.
    ax : matplotlib Axes or None
        Axes to draw on.  A new figure is created if ``None``.
    figsize : tuple of float
        Figure size when ``ax`` is ``None``.
    title : str
        Axis title.
    log_scale : bool
        Use a log scale on the y-axis.  Default ``False``.

    Returns
    -------
    VoronoiPlotResult
        Dataclass with ``ax`` and ``collections``.
    """
    import matplotlib.pyplot as plt

    from .result import VoronoiPlotResult

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    history = result.convergence_history
    iters = list(range(1, len(history) + 1))
    ax.plot(iters, history, color="#2c7bb6", linewidth=1.5)
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("CV(area)", fontsize=9)
    ax.set_title(title, fontsize=10)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, linewidth=0.4, alpha=0.5)

    # Annotate final value
    if history:
        ax.axhline(history[-1], color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.text(
            iters[-1],
            history[-1],
            f"  {history[-1]:.4f}",
            va="bottom",
            ha="left",
            fontsize=8,
            color="#e74c3c",
        )

    # Mark convergence point if converged before n_iter
    metrics = result.metrics
    if metrics.get("converged") and len(history) < metrics.get("n_iterations", len(history)) + 1:
        ax.axvline(len(history), color="#27ae60", linewidth=0.8, linestyle=":", alpha=0.7)

    return VoronoiPlotResult(ax=ax, collections=list(ax.collections))


# ---------------------------------------------------------------------------
# plot_displacement
# ---------------------------------------------------------------------------


def plot_displacement(
    result: VoronoiCartogram,
    *,
    state: str = "final",
    show_geometries: bool = True,
    show_centroids: bool = True,
    show_adjacency: bool = True,
    show_positions: bool = False,
    show_displacement: bool = True,
    legend: bool = True,
    geometry_style: dict | None = None,
    adjacency_style: dict | None = None,
    centroid_style: dict | None = None,
    position_style: dict | None = None,
    displacement_style: dict | None = None,
    ax: Any | None = None,
    figsize: tuple[float, float] = (10, 8),
    title: str = "Voronoi cartogram — centroid redistribution",
) -> VoronoiPlotResult:
    """Plot centroids and displacement arrows on a single axis.

    Parameters
    ----------
    result : VoronoiCartogram
        The cartogram to visualize.
    state : {"original", "final"}
        Which state to visualize.  ``"original"`` draws the source
        geometries and their centroids; ``"final"`` (default) draws the
        Voronoi cells and their centroids.  Displacement arrows always run
        original → final regardless of this setting.
    show_geometries : bool
        Draw geometry polygons behind the centroid scatter.  Default ``True``.
    show_centroids : bool
        Draw centroid scatter markers.  Default ``True``.
    show_adjacency : bool
        Draw adjacency graph edges.  Default ``True``.
    show_positions : bool
        Also scatter ``result.positions`` (the Lloyd generator points) as a
        secondary marker (only used when ``state="final"``).  Default ``False``.
    show_displacement : bool
        Draw colored arrows from original to final cell-centroid positions.
        Default ``True``.
    legend : bool
        Show the displacement-magnitude colorbar (and any other legends).
        Default ``True``.
    geometry_style : dict or None
        Styling for geometry polygons, merged over defaults
        ``{"facecolor": "#e8eef4", "edgecolor": "#9aafc4", "linewidth": 0.4, "alpha": 0.6}``.
        All keys forwarded to ``GeoDataFrame.plot()``.

        The ``"color"`` key supports multi-type resolution:

        - ``str`` matching a column → choropleth; use ``"cmap"`` (default
          ``"Blues"``) to set the colormap.
        - any other ``str`` → solid fill color.
        - 1-D numeric array → per-geometry values mapped via ``"cmap"``.
        - list/array of color strings → per-geometry colors.
    adjacency_style : dict or None
        Styling for adjacency graph edges, merged over defaults
        ``{"color": "#6b8fa8", "linewidth": 0.6, "alpha": 0.5, "zorder": 2}``.
        All keys forwarded to ``ax.plot()``.
    centroid_style : dict or None
        Styling for centroid scatter markers, merged over defaults
        ``{"s": 18, "c": "#1a5276", "zorder": 3, "linewidths": 0}``.
        All keys forwarded to ``ax.scatter()``.

        The ``"c"`` (or ``"color"``) key supports the same multi-type
        resolution as ``geometry_style["color"]``; use ``"cmap"`` (default
        ``"viridis"``) for column/numeric cases.
    position_style : dict or None
        Styling for the ``result.positions`` generator-point overlay (only
        used when ``show_positions=True``), merged over defaults
        ``{"s": 14, "marker": "x", "c": "#e74c3c", "zorder": 5, "linewidths": 0.8}``.
        All keys forwarded to ``ax.scatter()``.
    displacement_style : dict or None
        Styling for displacement arrows, merged over defaults
        ``{"cmap": "plasma", "lw": 0.9, "mutation_scale": 6, "arrowstyle": "-|>", "min_frac": 0.01}``.

        - ``"cmap"`` — arrow/colorbar colormap.
        - ``"min_frac"`` — skip arrows whose magnitude is below this
          fraction of the maximum.
        - remaining keys forwarded into ``arrowprops``.
    ax : matplotlib Axes or None
        Axes to draw on.  A new figure is created if ``None``.
    figsize : tuple of float
        Figure size used when ``ax`` is ``None``.
    title : str
        Axis title.

    Returns
    -------
    VoronoiPlotResult
        Dataclass with ``ax`` and ``collections``.

    Raises
    ------
    RuntimeError
        If no source GeoDataFrame is stored on the result.
    ValueError
        If ``state`` is not ``"original"`` or ``"final"``.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    from carto_flow.geo_utils.adjacency import find_adjacent_pairs

    from .result import VoronoiPlotResult

    if result._source_gdf is None:
        raise RuntimeError("No source GeoDataFrame stored; cannot plot displacement.")
    if state not in ("original", "final"):
        raise ValueError(f"state must be 'original' or 'final', got {state!r}")

    # --- resolve style dicts -------------------------------------------------
    def _norm_scatter(user: dict | None) -> dict:
        """Normalize 'color' → 'c' so scatter never receives both."""
        d = dict(user or {})
        if "color" in d:
            d.setdefault("c", d.pop("color"))
        return d

    geo_kw: dict = {
        "facecolor": "#e8eef4",
        "edgecolor": "#9aafc4",
        "linewidth": 0.4,
        "alpha": 0.6,
        **(geometry_style or {}),
    }
    adj_kw: dict = {
        "color": "#6b8fa8",
        "linewidth": 0.6,
        "alpha": 0.5,
        "zorder": 2,
        **(adjacency_style or {}),
    }
    cen_kw: dict = {
        "s": 18,
        "c": "#1a5276",
        "zorder": 3,
        "linewidths": 0,
        **_norm_scatter(centroid_style),
    }
    pos_kw: dict = {
        "s": 14,
        "marker": "x",
        "c": "#e74c3c",
        "zorder": 5,
        "linewidths": 0.8,
        **_norm_scatter(position_style),
    }
    disp_kw: dict = {
        "cmap": "plasma",
        "lw": 0.9,
        "mutation_scale": 6,
        "arrowstyle": "-|>",
        "min_frac": 0.01,
        **(displacement_style or {}),
    }

    # --- helpers -------------------------------------------------------------
    def _is_column(val: Any) -> bool:
        return (
            isinstance(val, str) and val in result._source_gdf.columns  # type: ignore[union-attr]
        )

    def _is_numeric_array(val: Any) -> bool:
        return not isinstance(val, str) and np.issubdtype(np.asarray(val).dtype, np.number)

    def _resolve_c_for_scatter(kw: dict) -> dict:
        """Resolve 'c' column/array/color and return scatter kwargs."""
        out = dict(kw)
        c_val = out.get("c")
        cmap_val = out.pop("cmap", "viridis")
        if _is_column(c_val):
            vals: Any = result._source_gdf[c_val].values  # type: ignore[index]
            if not np.issubdtype(np.asarray(vals).dtype, np.number):
                _, vals = np.unique(vals, return_inverse=True)
                vals = vals.astype(float)
            out["c"] = vals
            out["cmap"] = cmap_val
        elif _is_numeric_array(c_val):
            out["c"] = np.asarray(c_val, dtype=float)
            out["cmap"] = cmap_val
        elif c_val is not None and not isinstance(c_val, str):
            # list/array of color strings
            out["c"] = list(c_val)
        # else: plain color string or None — pass through as-is
        return out

    # --- geometry color resolution ------------------------------------------
    geo_color_val = geo_kw.pop("color", None)
    geo_cmap_val = geo_kw.pop("cmap", "Blues")

    geo_col_values: np.ndarray | None = None
    if _is_column(geo_color_val):
        geo_col_values = result._source_gdf[geo_color_val].values

    # --- main data -----------------------------------------------------------
    geometries = list(result._source_gdf.geometry)
    old_pos = np.array([[g.centroid.x, g.centroid.y] for g in geometries], dtype=np.float64)
    cell_centroids = np.array([[c.centroid.x, c.centroid.y] for c in result.cells], dtype=np.float64)
    edges = [(int(a), int(b)) for a, b, _ in find_adjacent_pairs(geometries)] if show_adjacency else []
    displacement = cell_centroids - old_pos
    disp_mag = np.linalg.norm(displacement, axis=1)

    cen_scatter_kw = _resolve_c_for_scatter(cen_kw)

    # --- axis ----------------------------------------------------------------
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # --- choose data for the selected state ----------------------------------
    gdf_cells = result.to_geodataframe()
    if state == "original":
        panel_gdf = result._source_gdf
        positions = old_pos
    else:
        panel_gdf = gdf_cells
        positions = cell_centroids

    # --- geometry layer ------------------------------------------------------
    if show_geometries:
        if geo_color_val is None:
            panel_gdf.plot(ax=ax, **geo_kw)
        elif geo_col_values is not None:
            tmp = panel_gdf.copy()
            tmp["_gc"] = geo_col_values
            tmp.plot(ax=ax, column="_gc", cmap=geo_cmap_val, **geo_kw)
        elif _is_numeric_array(geo_color_val):
            tmp = panel_gdf.copy()
            tmp["_gc"] = np.asarray(geo_color_val, dtype=float)
            tmp.plot(ax=ax, column="_gc", cmap=geo_cmap_val, **geo_kw)
        elif isinstance(geo_color_val, str):
            panel_gdf.plot(ax=ax, color=geo_color_val, **geo_kw)
        else:
            panel_gdf.plot(ax=ax, color=list(geo_color_val), **geo_kw)

    # --- adjacency layer -----------------------------------------------------
    if show_adjacency:
        for i, j in edges:
            ax.plot(
                [positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                **adj_kw,
            )

    # --- centroid scatter ----------------------------------------------------
    if show_centroids:
        ax.scatter(positions[:, 0], positions[:, 1], **cen_scatter_kw)

    # --- generator-point overlay (final state only) --------------------------
    if show_positions and state == "final":
        ax.scatter(result.positions[:, 0], result.positions[:, 1], **pos_kw)

    # --- displacement arrows -------------------------------------------------
    if show_displacement:
        min_frac = disp_kw.pop("min_frac", 0.01)
        disp_cmap = plt.get_cmap(disp_kw.pop("cmap", "plasma"))
        arrow_kw = disp_kw

        max_mag = disp_mag.max() or 1.0
        norm = mcolors.Normalize(vmin=0, vmax=max_mag)
        for i in range(len(geometries)):
            if disp_mag[i] < max_mag * min_frac:
                continue
            ax.annotate(
                "",
                xy=(cell_centroids[i, 0], cell_centroids[i, 1]),
                xytext=(old_pos[i, 0], old_pos[i, 1]),
                arrowprops={"color": disp_cmap(norm(disp_mag[i])), **arrow_kw},
                zorder=4,
            )
        if legend:
            sm = plt.cm.ScalarMappable(cmap=disp_cmap, norm=norm)
            sm.set_array([])
            cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("Displacement magnitude", fontsize=9)

    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.axis("off")
    return VoronoiPlotResult(ax=ax, collections=list(ax.collections))


# ---------------------------------------------------------------------------
# _plot_topology_issues  (internal helper)
# ---------------------------------------------------------------------------


def _plot_topology_issues(
    analysis: TopologyAnalysis,
    cartogram: VoronoiCartogram,
    ax: Any,
    *,
    show_base: bool = True,
    show_contiguity: bool = True,
    show_adjacency: bool = True,
    show_orientation: bool = True,
) -> None:
    """Draw topology-issue layers onto *ax* (no axis formatting)."""
    import geopandas as gpd
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    cells = list(cartogram.cells)
    n = len(cells)
    labels = list(cartogram._source_gdf.index) if cartogram._source_gdf is not None else list(range(n))
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    # --- 1. Base cells (optional — skip when overlaying on an existing plot) ---
    if show_base:
        gdf_base = gpd.GeoDataFrame(geometry=cells, crs=getattr(cartogram._source_gdf, "crs", None))
        gdf_base.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#888888", linewidth=0.4, zorder=1)

    legend_handles: list[Any] = []

    # --- 2. Satellite cells (contiguity) ---
    if show_contiguity and analysis.discontiguous_groups:
        sat_labels: set = set()
        for _grp, lbl_list in analysis.discontiguous_groups:
            sat_labels.update(lbl_list)
        sat_idx = [label_to_idx[lbl] for lbl in sat_labels if lbl in label_to_idx]
        if sat_idx:
            sat_geoms = [cells[i] for i in sat_idx]
            gdf_sat = gpd.GeoDataFrame(
                geometry=sat_geoms,
                crs=getattr(cartogram._source_gdf, "crs", None),
            )
            gdf_sat.plot(
                ax=ax,
                facecolor="none",
                edgecolor="#e67e00",
                linewidth=1.5,
                zorder=2,
            )
            cx = [cells[i].centroid.x for i in sat_idx]
            cy = [cells[i].centroid.y for i in sat_idx]
            ax.scatter(cx, cy, s=12, c="#e67e00", marker="s", zorder=6, linewidths=0)
            legend_handles.append(
                mpatches.Patch(
                    facecolor="none",
                    edgecolor="#e67e00",
                    linewidth=1.5,
                    label=f"Satellite cells ({len(sat_idx)})",
                )
            )

    need_cell_xy = (show_adjacency and bool(analysis.violated_adjacency)) or (
        show_orientation and bool(analysis.misaligned_orientation)
    )
    if need_cell_xy:
        cell_xy = {i: (cells[i].centroid.x, cells[i].centroid.y) for i in range(n)}

    # --- 3. Violated adjacency ---
    if show_adjacency and analysis.violated_adjacency:
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D

        segments = [
            [cell_xy[label_to_idx[li]], cell_xy[label_to_idx[lj]]]
            for li, lj in analysis.violated_adjacency
            if li in label_to_idx and lj in label_to_idx
        ]
        if segments:
            lc = LineCollection(
                segments,
                colors="#e74c3c",
                linewidths=1.0,
                linestyles="--",
                alpha=0.8,
                zorder=3,
            )
            ax.add_collection(lc)
            pts = np.array([p for seg in segments for p in seg])
            ax.scatter(pts[:, 0], pts[:, 1], s=8, c="#e74c3c", zorder=3, linewidths=0)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#e74c3c",
                linewidth=1.2,
                linestyle="--",
                label=f"Violated adjacency ({len(analysis.violated_adjacency)}/{analysis.n_adjacency_pairs})",
            )
        )

    # --- 4. Misaligned orientation ---
    if show_orientation and analysis.misaligned_orientation:
        norm = mcolors.Normalize(vmin=-1.0, vmax=0.0)
        cmap = plt.get_cmap("RdYlGn_r")
        X, Y, U, V, C = [], [], [], [], []
        for li, lj, cos_val in analysis.misaligned_orientation:
            if li not in label_to_idx or lj not in label_to_idx:
                continue
            xi, yi = cell_xy[label_to_idx[li]]
            xj, yj = cell_xy[label_to_idx[lj]]
            X.append(xi)
            Y.append(yi)
            U.append(xj - xi)
            V.append(yj - yi)
            C.append(norm(cos_val))
        if X:
            ax.quiver(
                X,
                Y,
                U,
                V,
                color=cmap(np.array(C)),
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.003,
                headwidth=4,
                headlength=5,
                zorder=4,
            )
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Orientation cosine", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        legend_handles.append(
            mpatches.Patch(
                facecolor=cmap(0.5),
                alpha=0.8,
                label=f"Misaligned orientation ({len(analysis.misaligned_orientation)})",
            )
        )

    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=8, loc="best", framealpha=0.8)


# ---------------------------------------------------------------------------
# plot_topology
# ---------------------------------------------------------------------------


def plot_topology(
    analysis: TopologyAnalysis,
    cartogram: VoronoiCartogram,
    *,
    show_base: bool | None = None,
    show_contiguity: bool = True,
    show_adjacency: bool = True,
    show_orientation: bool = True,
    ax: Any | None = None,
    figsize: tuple[float, float] = (10, 8),
    title: str = "Topology Analysis",
) -> VoronoiPlotResult:
    """Plot topology issues overlaid on the Voronoi cells.

    Parameters
    ----------
    analysis : TopologyAnalysis
        The analysis result to visualize.
    cartogram : VoronoiCartogram
        The cartogram whose cells provide the spatial context.
    show_base : bool or None
        Draw all cells as a gray background layer.  ``None`` (default) draws
        the base only when *ax* is ``None`` (i.e. a new figure is created);
        when an existing *ax* is supplied the base is skipped so that any
        prior plot on that axis is not hidden.  Pass ``True`` or ``False`` to
        override the automatic behavior.
    show_contiguity : bool
        Highlight satellite cells (stage 1 issues).  Default ``True``.
    show_adjacency : bool
        Draw lines between centroids of violated adjacency pairs (stage 2
        issues).  Default ``True``.
    show_orientation : bool
        Draw arrows for misaligned Voronoi-adjacent pairs (stage 3 issues).
        Default ``True``.
    ax : matplotlib Axes or None
        Axes to draw on.  A new figure is created if ``None``.
    figsize : tuple of float
        Figure size when ``ax`` is ``None``.
    title : str
        Axis title.

    Returns
    -------
    VoronoiPlotResult
        Dataclass with ``ax`` and ``collections``.
    """
    import matplotlib.pyplot as plt

    from .result import VoronoiPlotResult

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if show_base is None:
        show_base = ax is None or len(ax.collections) == 0

    _plot_topology_issues(
        analysis,
        cartogram,
        ax,
        show_base=show_base,
        show_contiguity=show_contiguity,
        show_adjacency=show_adjacency,
        show_orientation=show_orientation,
    )
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.axis("off")
    return VoronoiPlotResult(ax=ax, collections=list(ax.collections))


# ---------------------------------------------------------------------------
# plot_topology_repair
# ---------------------------------------------------------------------------


def plot_topology_repair(
    report: TopologyRepairReport,
    *,
    show_base: bool | None = None,
    show_issues: bool = True,
    show_reassigned: bool = True,
    axes: Any | None = None,
    figsize: tuple[float, float] = (16, 8),
    title: str | None = None,
) -> VoronoiPlotResult:
    """Plot a before/after comparison of the topology repair.

    Parameters
    ----------
    report : TopologyRepairReport
        The repair report to visualize.
    show_base : bool or None
        Draw all cells as a gray background layer in each panel.  ``None``
        (default) draws the base only when *axes* is ``None`` (new figure);
        when existing axes are supplied the base is skipped so prior content
        is not hidden.  Pass ``True`` or ``False`` to override.
    show_issues : bool
        Overlay topology-issue highlights (satellite cells, violated
        adjacency lines, misaligned orientation arrows) on both panels.
        Default ``True``.
    show_reassigned : bool
        Highlight reassigned cells in the *after* panel with a thick green
        border.  Default ``True``.
    axes : sequence of two Axes, or None
        Existing axes to draw on.  A new ``(1, 2)`` figure is created when
        ``None``.
    figsize : tuple of float
        Figure size when ``axes`` is ``None``.  Default ``(16, 8)``.
    title : str or None
        Figure suptitle.  Defaults to
        ``"Topology Repair — {stages_run}"``.

    Returns
    -------
    VoronoiPlotResult
        ``ax`` is a tuple ``(ax_before, ax_after)``.

    Notes
    -----
    When :attr:`~TopologyRepairReport._original_cartogram` is ``None``
    (e.g., the report was reconstructed from disk), only the *after* panel
    is rendered and a ``UserWarning`` is emitted.
    """
    import warnings

    import geopandas as gpd
    import matplotlib.pyplot as plt

    from .result import VoronoiPlotResult

    # ---- axis setup ----
    only_after = report._original_cartogram is None
    if only_after:
        warnings.warn(
            "TopologyRepairReport._original_cartogram is None — rendering only the 'after' panel.",
            UserWarning,
            stacklevel=2,
        )

    if axes is not None:
        ax_sequence = list(axes)
        if only_after:
            ax_after = ax_sequence[0]
            ax_before = None
        else:
            ax_before, ax_after = ax_sequence[0], ax_sequence[1]
        fig = ax_after.figure
    else:
        if only_after:
            fig, ax_after = plt.subplots(figsize=(figsize[0] // 2, figsize[1]))
            ax_before = None
        else:
            fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=figsize)

    if show_base is None:
        show_base = axes is None

    # ---- helper: draw base cells ----
    def _draw_base(cartogram: VoronoiCartogram, ax: Any) -> None:
        cells = list(cartogram.cells)
        crs = getattr(getattr(cartogram, "_source_gdf", None), "crs", None)
        gdf_base = gpd.GeoDataFrame(geometry=cells, crs=crs)
        gdf_base.plot(
            ax=ax,
            facecolor="#f0f0f0",
            edgecolor="#888888",
            linewidth=0.4,
            zorder=1,
        )

    # ---- left panel: before ----
    if ax_before is not None:
        _orig_cart = cast("VoronoiCartogram", report._original_cartogram)  # not None when ax_before is not None
        if show_base:
            _draw_base(_orig_cart, ax_before)
        if show_issues:
            _plot_topology_issues(
                report.before,
                _orig_cart,
                ax_before,
                show_base=False,
                show_contiguity=True,
                show_adjacency=True,
                show_orientation=True,
            )
        n_before = (
            report.before.n_discontiguous_groups
            + report.before.n_violated_adjacency
            + report.before.n_misaligned_orientation
        )
        ax_before.set_title(f"Before — {n_before} issue(s)", fontsize=10)
        ax_before.set_aspect("equal")
        ax_before.axis("off")

    # ---- right panel: after ----
    if show_base:
        _draw_base(report.cartogram, ax_after)
    if show_issues:
        _plot_topology_issues(
            report.after,
            report.cartogram,
            ax_after,
            show_base=False,
            show_contiguity=True,
            show_adjacency=True,
            show_orientation=True,
        )
    if show_reassigned and report.reassigned:
        cells_after = list(report.cartogram.cells)
        labels_after = (
            list(report.cartogram._source_gdf.index)
            if report.cartogram._source_gdf is not None
            else list(range(len(cells_after)))
        )
        label_to_idx_after = {lbl: i for i, lbl in enumerate(labels_after)}
        reassigned_idx = [label_to_idx_after[lbl] for lbl in report.reassigned if lbl in label_to_idx_after]
        if reassigned_idx:
            crs = getattr(getattr(report.cartogram, "_source_gdf", None), "crs", None)
            gdf_reassigned = gpd.GeoDataFrame(geometry=[cells_after[i] for i in reassigned_idx], crs=crs)
            gdf_reassigned.plot(
                ax=ax_after,
                facecolor="none",
                edgecolor="#27ae60",
                linewidth=2.0,
                zorder=5,
            )
            from matplotlib.lines import Line2D

            handle = Line2D(
                [0],
                [0],
                color="#27ae60",
                linewidth=2.0,
                label=f"Reassigned ({len(reassigned_idx)})",
            )
            existing = ax_after.get_legend()
            handles = list(existing.legend_handles) if existing else []
            handles.append(handle)
            ax_after.legend(handles=handles, fontsize=8, loc="best", framealpha=0.8)

    n_after = (
        report.after.n_discontiguous_groups + report.after.n_violated_adjacency + report.after.n_misaligned_orientation
    )
    ax_after.set_title(f"After — {n_after} issue(s)", fontsize=10)
    ax_after.set_aspect("equal")
    ax_after.axis("off")

    # ---- suptitle ----
    if title is None:
        stages = ", ".join(report.stages_run) if report.stages_run else "none"
        title = f"Topology Repair — {stages}"
    fig.suptitle(title, fontsize=12)

    ax_result = ax_after if only_after else (ax_before, ax_after)
    return VoronoiPlotResult(ax=ax_result, collections=list(ax_after.collections))


# ---------------------------------------------------------------------------
# plot_compactness
# ---------------------------------------------------------------------------


def plot_compactness(
    cartogram: VoronoiCartogram,
    group_by: str,
    *,
    normalize: bool = True,
    cmap: str = "YlOrRd",
    show_centroids: bool = True,
    show_edges: bool = True,
    legend: bool = True,
    ax: Any | None = None,
    figsize: tuple[float, float] = (10, 8),
    title: str = "Compactness",
) -> VoronoiPlotResult:
    """Plot per-cell inertia — the compactness measure used by topology repair.

    For each district *d* in group *g* the inertia is
    ``||cell_centroid[d] - group_centroid[g]||²``, matching the internal
    measure used by ``_repair_compactness``.  By default the value is
    normalized within each group (divided by the group mean) so that cells
    in different-sized groups can be compared on the same scale: values above
    1 are outliers relative to their group, values below 1 are compact.

    Parameters
    ----------
    cartogram : VoronoiCartogram
        The cartogram to analyze.
    group_by : str
        Column in the source GeoDataFrame identifying groups (same argument
        as ``repair_topology``).
    normalize : bool
        Divide each cell's inertia by the group mean.  Default ``True``.
    cmap : str
        Matplotlib colormap name.  Default ``"YlOrRd"``.
    show_centroids : bool
        Overlay group centroids as small black markers.  Default ``True``.
    show_edges : bool
        Draw cell borders.  Default ``True``.
    legend : bool
        Show a colorbar.  Default ``True``.
    ax : matplotlib Axes or None
        Axes to draw on.  A new figure is created if ``None``.
    figsize : tuple of float
        Figure size when ``ax`` is ``None``.
    title : str
        Axis title.

    Returns
    -------
    VoronoiPlotResult
        Dataclass with ``ax`` and ``collections``.
    """
    import geopandas as gpd
    import matplotlib.pyplot as plt

    from .result import VoronoiPlotResult

    if cartogram._source_gdf is None or group_by not in cartogram._source_gdf.columns:
        raise ValueError(f"group_by column {group_by!r} not found in source GeoDataFrame.")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    cells = list(cartogram.cells)
    n = len(cells)
    cell_xy = np.array([[c.centroid.x, c.centroid.y] for c in cells])
    group_labels = list(cartogram._source_gdf[group_by])

    # Build group → [district indices] mapping.
    from collections import defaultdict

    group_members: dict = defaultdict(list)
    for d, g in enumerate(group_labels):
        group_members[g].append(d)

    # Compute per-cell inertia.
    raw = np.zeros(n)
    centroids_xy: list[tuple[float, float]] = []
    for _g, members in group_members.items():
        pts = cell_xy[members]
        c_g = pts.mean(axis=0)
        centroids_xy.append((float(c_g[0]), float(c_g[1])))
        diffs = pts - c_g
        inertia = np.sum(diffs**2, axis=1)
        for i, d in enumerate(members):
            raw[d] = float(inertia[i])

    if normalize:
        score = np.zeros(n)
        for members in group_members.values():
            vals = raw[members]
            mean_val = float(vals.mean())
            if mean_val > 0:
                score[members] = vals / mean_val
            # else leave as zero (all cells at centroid)
        col_label = "Normalized inertia (relative to group mean)"
    else:
        score = raw
        col_label = "Inertia (m²)"

    gdf_cells = gpd.GeoDataFrame(
        {"_score": score},
        geometry=cells,
        crs=cartogram._source_gdf.crs,
    )
    edge_color = "black" if show_edges else "none"
    linewidth = 0.4 if show_edges else 0
    gdf_cells.plot(
        ax=ax,
        column="_score",
        cmap=cmap,
        edgecolor=edge_color,
        linewidth=linewidth,
        legend=legend,
        legend_kwds={"label": col_label, "shrink": 0.6} if legend else {},
    )

    if show_centroids and centroids_xy:
        cx, cy = zip(*centroids_xy, strict=False)
        ax.scatter(cx, cy, s=8, c="black", zorder=5, linewidths=0)

    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.axis("off")
    return VoronoiPlotResult(ax=ax, collections=list(ax.collections))
