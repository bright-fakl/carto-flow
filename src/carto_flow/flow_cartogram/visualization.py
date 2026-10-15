"""
Visualization utilities for cartogram results.

Plotting and animation utilities for morphing results and algorithm analysis.

Functions
---------
plot_cartogram
    Plot morphed geometries from a Cartogram.
plot_comparison
    Side-by-side comparison of original vs morphed.
plot_convergence
    Plot error metrics over iterations.
plot_density_field
    Visualize density field as heatmap.
plot_velocity_field
    Visualize velocity field as quiver plot.

Examples
--------
>>> from carto_flow.flow_cartogram import create_flow_cartogram, MorphOptions
>>> from carto_flow.flow_cartogram.visualization import plot_comparison, plot_convergence
>>> cartogram = create_flow_cartogram(gdf, "population", options=MorphOptions.preset_fast())
>>> plot_comparison(gdf, cartogram)
>>> plot_convergence(cartogram.snapshots)
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from .._plot_params import classify_string, deprecated_init_params, deprecated_params

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.image import AxesImage
    from matplotlib.lines import Line2D

    from .cartogram import Cartogram
    from .grid import Grid
    from .plot_results import (
        CartogramComparisonResult,
        CartogramPlotResult,
        ConvergencePlotResult,
        DensityFieldResult,
        VelocityFieldResult,
        WorkflowConvergencePlotResult,
    )
    from .workflow import CartogramWorkflow

__all__ = [
    "DensityPlotOptions",
    "VelocityPlotOptions",
    "plot_cartogram",
    "plot_comparison",
    "plot_convergence",
    "plot_density_field",
    "plot_velocity_field",
    "plot_workflow_convergence",
]


@deprecated_init_params(legend_kwargs="colorbar_kwargs")
@dataclass
class DensityPlotOptions:
    """Options for density field rendering in static plots and animations.

    Passing the same ``DensityPlotOptions`` instance to both
    ``plot_density_field`` and ``animate_fields`` guarantees identical rendering.

    Parameters
    ----------
    normalize : str, optional
        Density normalization: ``None`` (absolute values), ``"difference"``
        (rho - target), or ``"ratio"`` (rho / target, log scale).
    cmap : str, optional
        Colormap name. Defaults based on ``normalize``.
    alpha : float, default=1.0
        Transparency of the density heatmap.
    clip_percentile : float, optional
        Clip color range to exclude outliers (difference mode only).
        Value in [0, 100] specifies percentile from each end.
    max_scale : float, optional
        Fixed maximum for symmetric color scale. For difference mode sets
        range to [-max_scale, max_scale]; for ratio mode sets
        [1/max_scale, max_scale].
    vmin : float, optional
        Override color-scale minimum.
    vmax : float, optional
        Override color-scale maximum.
    legend_kwargs : dict, optional
        Extra keyword arguments forwarded to ``fig.colorbar()``.
    imshow_kwargs : dict, optional
        Extra keyword arguments forwarded to ``ax.imshow()``.
    """

    normalize: str | None = None
    cmap: str | None = None
    alpha: float = 1.0
    clip_percentile: float | None = None
    max_scale: float | None = None
    vmin: float | None = None
    vmax: float | None = None
    legend_kwargs: dict = field(default_factory=dict)
    imshow_kwargs: dict = field(default_factory=dict)


@deprecated_init_params(legend="colorbar", legend_kwargs="colorbar_kwargs")
@dataclass
class VelocityPlotOptions:
    """Options for velocity field rendering in static plots and animations.

    Passing the same ``VelocityPlotOptions`` instance to both
    ``plot_velocity_field`` and ``animate_fields`` guarantees identical rendering.

    Parameters
    ----------
    skip : int, default=4
        Plot every nth velocity arrow to reduce clutter.
    velocity_scale : float, optional
        Arrow length as a fraction of grid cell size at the reference magnitude.
        If None, matplotlib auto-scales.
    ref_magnitude : float, optional
        Reference velocity magnitude for arrow scaling. Defaults to the
        maximum magnitude in the field. Pass the same value to multiple calls
        for consistent scaling.
    color : str, default="white"
        Base arrow color when ``color_by`` is None.
    alpha : float, default=0.8
        Base arrow transparency.
    color_by : str, optional
        Color arrows by ``"magnitude"`` or ``"direction"``. None for solid color.
    cmap : str, optional
        Colormap when ``color_by`` is set.
    legend : bool, default=True
        Whether to show a colorbar when ``color_by`` is set.
    alpha_by_magnitude : bool, default=False
        Vary arrow transparency with velocity magnitude.
    alpha_range : tuple[float, float], default=(0.2, 1.0)
        Min/max alpha when ``alpha_by_magnitude`` is True.
    legend_kwargs : dict, optional
        Extra keyword arguments forwarded to ``fig.colorbar()``.
    quiver_kwargs : dict, optional
        Extra keyword arguments forwarded to ``ax.quiver()``.
    """

    skip: int = 4
    velocity_scale: float | None = None
    ref_magnitude: float | None = None
    color: str = "white"
    alpha: float = 0.8
    color_by: str | None = None
    cmap: str | None = None
    legend: bool = True
    alpha_by_magnitude: bool = False
    alpha_range: tuple = (0.2, 1.0)
    legend_kwargs: dict = field(default_factory=dict)
    quiver_kwargs: dict = field(default_factory=dict)


def _resolve_bounds(
    bounds: Any,
    grid: "Grid",
) -> tuple[float, float, float, float] | None:
    """Resolve bounds parameter to (xmin, ymin, xmax, ymax) tuple.

    Parameters
    ----------
    bounds : str, float, tuple, or None
        Bounds specification:
        - None: No clipping (returns None)
        - "data": Use grid.data_bounds (original bounds without margin)
        - float: Data bounds with margin as fraction (e.g., 0.1 = 10% margin)
        - tuple: (xmin, ymin, xmax, ymax) explicit bounds
    grid : Grid
        Grid object for data_bounds lookup

    Returns
    -------
    tuple or None
        Resolved bounds as (xmin, ymin, xmax, ymax), or None for no clipping
    """
    if bounds is None:
        return None
    elif bounds == "data":
        return grid.data_bounds
    elif isinstance(bounds, (int, float)) and not isinstance(bounds, bool):
        # Float/int: data bounds + margin as fraction
        xmin, ymin, xmax, ymax = grid.data_bounds
        w = xmax - xmin
        h = ymax - ymin
        margin = float(bounds)
        return (
            xmin - margin * w,
            ymin - margin * h,
            xmax + margin * w,
            ymax + margin * h,
        )
    elif isinstance(bounds, (tuple, list)) and len(bounds) == 4:
        return tuple(bounds)
    else:
        raise ValueError(
            f"bounds must be None, 'data', a float (margin), or (xmin, ymin, xmax, ymax) tuple, got {bounds!r}"
        )


def _prepare_density_display(
    rho: np.ndarray,
    target_density: float | None,
    normalize: str | None,
    clip_percentile: float | None = None,
    max_scale: float | None = None,
    geometry_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float | None, float | None, str, str, Any | None]:
    """Prepare density field for display with optional normalization.

    This internal helper is used by both plot_density_field and animate_fields
    to ensure consistent normalization behavior.

    Parameters
    ----------
    rho : np.ndarray
        Density field array (already in display units, i.e. values / display-area).
    target_density : float
        Target equilibrium density for normalization (same units as rho).
    normalize : str or None
        Normalization mode: None, "difference", or "ratio".
    clip_percentile : float, optional
        Percentile for clipping outliers (difference mode only).
    max_scale : float, optional
        Fixed maximum for symmetric color scale.
    geometry_mask : np.ndarray, optional
        Integer mask for cells inside/outside geometries.

    Returns
    -------
    tuple
        (display_array, vmin, vmax, cmap, colorbar_label, norm)
        - display_array: Normalized density values for display
        - vmin, vmax: Color scale limits
        - cmap: Colormap name
        - colorbar_label: Label for colorbar
        - norm: Matplotlib Normalize or TwoSlopeNorm object (or None)
    """
    from matplotlib.colors import TwoSlopeNorm

    if normalize is None:
        return rho, None, None, "viridis", "Density", None

    if normalize not in ("difference", "ratio"):
        raise ValueError(f"normalize must be 'difference', 'ratio', or None, got {normalize!r}")

    if target_density is None or target_density <= 0:
        raise ValueError(
            f"target_density must be positive, got {target_density}. Ensure the cartogram was computed successfully."
        )

    if normalize == "difference":
        rho_display = rho - target_density
        colorbar_label = "Density - Mean"

        if max_scale is not None:
            max_deviation = max_scale
        elif clip_percentile is not None:
            inside_values = rho_display[geometry_mask != -1] if geometry_mask is not None else rho_display.ravel()
            p_low = np.percentile(inside_values, clip_percentile)
            p_high = np.percentile(inside_values, 100 - clip_percentile)
            max_deviation = max(abs(p_low), abs(p_high))
        else:
            data_min = rho_display.min()
            data_max = rho_display.max()
            max_deviation = max(abs(data_min), abs(data_max))

        # Handle case where all values are exactly target density
        if max_deviation == 0:
            vmin = -1e-10
            vmax = 1e-10
        else:
            vmin = -max_deviation
            vmax = max_deviation

        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cmap = "RdBu_r"

    else:  # ratio
        ratio = rho / target_density
        ratio_clipped = np.clip(ratio, 1e-6, None)
        rho_display = np.log2(ratio_clipped)
        colorbar_label = "Density / Mean"

        if max_scale is not None:
            max_log_deviation = np.log2(max_scale)
        else:
            data_min = rho_display.min()
            data_max = rho_display.max()
            max_log_deviation = max(abs(data_min), abs(data_max))

        vmin = -max_log_deviation
        vmax = max_log_deviation
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cmap = "RdBu_r"

    return rho_display, vmin, vmax, cmap, colorbar_label, norm


def _compute_quiver_scale(
    mag_max: float,
    skip: int,
    dx: float,
    scale: float | None,
    ref_magnitude: float | None = None,
) -> dict:
    """Return quiver kwargs with scale/scale_units set when scale is given.

    Parameters
    ----------
    mag_max : float
        Maximum velocity magnitude across all arrows (used when ref_magnitude
        is None).
    skip : int
        Subsampling stride (every nth grid point is plotted).
    dx : float
        Grid cell size.
    scale : float or None
        Desired arrow length as a fraction of one grid cell for an arrow at
        the reference magnitude. If None, returns empty dict (matplotlib
        auto-scale).
    ref_magnitude : float, optional
        Reference magnitude for arrow scaling. Defaults to ``mag_max``.

    Returns
    -------
    dict
        Keyword arguments to pass to ``ax.quiver()``.
    """
    if scale is None:
        return {}
    effective_ref = ref_magnitude if ref_magnitude is not None else mag_max
    if effective_ref <= 0:
        return {}
    return {"scale": effective_ref / (scale * skip * dx), "scale_units": "xy"}


def _render_density_on_ax(
    ax: "Axes",
    rho: np.ndarray,
    target_density: float | None,
    extent: list,
    opts: "DensityPlotOptions",
    override_vmin: float | None = None,
    override_vmax: float | None = None,
    geometry_mask: np.ndarray | None = None,
) -> "tuple[AxesImage, str]":
    """Render a density field on an axes and return the artist and colorbar label.

    Combines :func:`_prepare_density_display` with ``ax.imshow``. The
    ``override_vmin``/``override_vmax`` parameters allow animation functions to
    pass globally pre-computed limits while still applying the correct
    ``TwoSlopeNorm`` for diverging colormaps. ``opts.vmin``/``opts.vmax``
    override per-field computed limits but are themselves overridden by
    ``override_vmin``/``override_vmax``.

    Parameters
    ----------
    ax : Axes
        Axes to draw on.
    rho : np.ndarray
        Density field array.
    target_density : float
        Target equilibrium density (same units as rho).
    extent : list
        ``[xmin, xmax, ymin, ymax]`` passed to ``ax.imshow``.
    opts : DensityPlotOptions
        Rendering options (normalization, colormap, alpha, clip settings, etc.).
    override_vmin : float, optional
        Animation global minimum — takes precedence over ``opts.vmin``.
    override_vmax : float, optional
        Animation global maximum — takes precedence over ``opts.vmax``.
    geometry_mask : np.ndarray, optional
        Mask for percentile clipping passed to :func:`_prepare_density_display`.

    Returns
    -------
    tuple[AxesImage, str]
        The image artist and the colorbar label string.
    """
    from matplotlib.colors import TwoSlopeNorm

    rho_display, local_vmin, local_vmax, default_cmap, label, norm = _prepare_density_display(
        rho,
        target_density,
        opts.normalize,
        clip_percentile=opts.clip_percentile,
        max_scale=opts.max_scale,
        geometry_mask=geometry_mask,
    )
    effective_cmap = opts.cmap if opts.cmap is not None else default_cmap

    # Precedence: override_vmin/vmax > opts.vmin/vmax > per-field local_vmin/vmax
    if override_vmin is not None:
        effective_vmin = override_vmin
    elif opts.vmin is not None:
        effective_vmin = opts.vmin
    else:
        effective_vmin = local_vmin  # type: ignore[assignment]

    if override_vmax is not None:
        effective_vmax = override_vmax
    elif opts.vmax is not None:
        effective_vmax = opts.vmax
    else:
        effective_vmax = local_vmax  # type: ignore[assignment]

    if opts.normalize is not None and norm is not None:
        norm = TwoSlopeNorm(vmin=effective_vmin, vcenter=0.0, vmax=effective_vmax)

    imshow_kw: dict = {
        "origin": "lower",
        "extent": extent,
        "aspect": "equal",
        "cmap": effective_cmap,
        "norm": norm,
        "alpha": opts.alpha,
        **opts.imshow_kwargs,
    }
    if opts.normalize is None:
        imshow_kw["vmin"] = effective_vmin
        imshow_kw["vmax"] = effective_vmax

    im = ax.imshow(rho_display, **imshow_kw)
    return im, label


def _prepare_velocity_colors(
    vx: np.ndarray,
    vy: np.ndarray,
    color_by: str | None = None,
    alpha_by_magnitude: bool = False,
    alpha_range: tuple[float, float] = (0.2, 1.0),
    cmap: str | None = None,
    base_color: str = "white",
    base_alpha: float = 0.8,
    mag_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, str | None, str | None]:
    """Prepare velocity field colors based on magnitude/direction.

    This internal helper is used by both plot_velocity_field and animate_fields
    to ensure consistent color/alpha computation.

    Parameters
    ----------
    vx, vy : np.ndarray
        Velocity components (already subsampled if needed).
    color_by : str, optional
        Color mode: None, "magnitude", or "direction".
    alpha_by_magnitude : bool, default=False
        Whether to vary transparency with magnitude.
    alpha_range : tuple, default=(0.2, 1.0)
        Min and max alpha when alpha_by_magnitude is True.
    cmap : str, optional
        Colormap name. Defaults based on color_by.
    base_color : str, default="white"
        Base color when not using color_by.
    base_alpha : float, default=0.8
        Base alpha when not using alpha_by_magnitude.
    mag_range : tuple, optional
        (min, max) magnitude for normalization. If None, computed from data.

    Returns
    -------
    tuple
        (colors, colorbar_label, effective_cmap)
        - colors: RGBA array of shape (n, 4) for quiver
        - colorbar_label: Label for colorbar (or None)
        - effective_cmap: Colormap name used (or None)
    """
    from matplotlib.colors import to_rgba

    magnitude = np.sqrt(vx**2 + vy**2)

    if mag_range is not None:
        mag_min, mag_max = mag_range
    else:
        mag_min, mag_max = magnitude.min(), magnitude.max()

    if mag_max > mag_min:
        magnitude_norm = np.clip((magnitude - mag_min) / (mag_max - mag_min), 0, 1)
    else:
        magnitude_norm = np.ones_like(magnitude)

    if color_by is None and not alpha_by_magnitude:
        # Simple solid color
        base_rgba = to_rgba(base_color, alpha=base_alpha)
        colors = np.tile(base_rgba, (vx.size, 1))
        return colors, None, None

    # Determine colormap and color values
    if color_by == "magnitude":
        C_norm = magnitude_norm
        colorbar_label = "Magnitude"
        effective_cmap = cmap if cmap is not None else "viridis"
    elif color_by == "direction":
        direction = np.arctan2(vy, vx)
        C_norm = (direction + np.pi) / (2 * np.pi)
        colorbar_label = "Direction"
        effective_cmap = cmap if cmap is not None else "twilight"
    else:
        C_norm = np.zeros_like(magnitude)
        colorbar_label = None
        effective_cmap = cmap if cmap is not None else "viridis"

    # Get colors from colormap
    cmap_obj = plt.get_cmap(effective_cmap)
    colors = cmap_obj(C_norm.ravel())

    # Apply alpha by magnitude if requested
    if alpha_by_magnitude:
        alpha_min, alpha_max = alpha_range
        alphas = alpha_min + magnitude_norm.ravel() * (alpha_max - alpha_min)
        colors = np.array(colors)  # Ensure mutable
        colors[:, 3] = alphas
    elif color_by is None:
        # No color_by but we computed colors for alpha purposes
        base_rgba = to_rgba(base_color, alpha=base_alpha)
        colors = np.tile(base_rgba, (vx.size, 1))
        alpha_min, alpha_max = alpha_range
        alphas = alpha_min + magnitude_norm.ravel() * (alpha_max - alpha_min)
        colors[:, 3] = alphas

    return colors, colorbar_label, effective_cmap


# Built-in quantities of a cartogram, by their name in the vocabulary and the column that holds them
_BUILTIN_COLUMNS = {"errors_pct": "_morph_error_pct", "density": "_morph_density"}


def _fill_arguments(gdf: Any, facecolor: Any) -> tuple[Any, dict[str, Any]]:
    """Resolve ``facecolor`` against a GeoDataFrame into the data and the arguments for ``GeoDataFrame.plot``.

    Returns the GeoDataFrame (a copy when a column was added) and ``{"column": name}`` for data or
    ``{"facecolor": value}`` for a literal or an array of colors. A string is resolved by
    :func:`carto_flow._plot_params.classify_string`.
    """
    if facecolor is None:
        return gdf, {}
    if isinstance(facecolor, str):
        kind = classify_string(facecolor, set(gdf.columns), _BUILTIN_COLUMNS, stacklevel=4)
        if kind == "column":
            return gdf, {"column": facecolor}
        if kind == "builtin":
            column = _BUILTIN_COLUMNS[facecolor]
            if column not in gdf.columns:
                raise ValueError(f"'{facecolor}' is not available for this cartogram (no column '{column}')")
            return gdf, {"column": column}
        return gdf, {"facecolor": facecolor}
    values = np.asarray(facecolor)
    if values.ndim == 1 and len(values) == len(gdf) and values.dtype.kind in "iufb":
        gdf = gdf.copy()
        gdf["_facecolor"] = values
        return gdf, {"column": "_facecolor"}
    return gdf, {"facecolor": facecolor}


@deprecated_params(facecolor="column", legend_kwargs="legend_kwds")
def plot_cartogram(
    cartogram: "Cartogram",
    ax: Optional["Axes"] = None,
    facecolor: Any = None,
    iteration: int | None = None,
    cmap: str = "RdYlGn_r",
    legend: bool = True,
    legend_kwargs: dict | None = None,
    **kwargs: Any,
) -> "CartogramPlotResult":
    """Plot morphed geometries from a Cartogram.

    Parameters
    ----------
    cartogram : Cartogram
        Cartogram containing morphed geometries to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    facecolor : str or array-like, optional
        Fill of the geometries. A column of the source GeoDataFrame, the built-in
        quantity ``"errors_pct"`` (percentage errors, the default if available) or
        ``"density"``, a numeric array with one value per geometry (all color the
        geometries by value), or a color (a string or an array of colors, drawn as is).
        A string that is a column and also a built-in quantity or a color is read as the
        column, then as the built-in quantity, and a warning is issued.
    iteration : int, optional
        Which iteration snapshot to plot (default: latest).
    cmap : str, default='RdYlGn_r'
        Matplotlib colormap name, used when the fill is data.
    legend : bool, default=True
        Show colorbar legend when plotting by data.
    legend_kwargs : dict, optional
        Keyword arguments of the legend of the fill (passed to ``GeoDataFrame.plot`` as ``legend_kwds``).
    **kwargs
        Additional arguments passed to GeoDataFrame.plot()

    Returns
    -------
    CartogramPlotResult
        Result with the axes and named artist references.

    Examples
    --------
    >>> from carto_flow import create_flow_cartogram
    >>> from carto_flow.flow_cartogram.visualization import plot_cartogram
    >>> cartogram = create_flow_cartogram(gdf, 'population')
    >>> plot_cartogram(cartogram)
    >>> plot_cartogram(cartogram, column='population', cmap='Blues')
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 7))

    gdf = cartogram.to_geodataframe(iteration=iteration, include_errors=True, include_density=True)

    # Default to the error column if no fill is given and errors are available
    if facecolor is None and "_morph_error_pct" in gdf.columns:
        facecolor = "errors_pct"

    gdf, fill = _fill_arguments(gdf, facecolor)

    before = len(ax.collections)
    if "column" in fill:
        legend_kw = {"legend_kwds": legend_kwargs} if legend_kwargs else {}
        gdf.plot(ax=ax, cmap=cmap, legend=legend, **fill, **legend_kw, **kwargs)
    else:
        gdf.plot(ax=ax, **fill, **kwargs)
    collections = list(ax.collections[before:])

    ax.set_aspect("equal")
    ax.set_title("Cartogram Result")
    from .plot_results import CartogramPlotResult

    return CartogramPlotResult(ax=ax, collections=collections)


@deprecated_params(facecolor="column")
def plot_comparison(
    left: Any,
    right: "Cartogram",
    facecolor: Any = None,
    iteration: int | None = None,
    left_iteration: int | None = None,
    figsize: tuple[float, float] = (14, 6),
    **kwargs: Any,
) -> "CartogramComparisonResult":
    """Side-by-side comparison of two geometries.

    Supports comparing:
    - A GeoDataFrame (original) with a Cartogram (result)
    - Two Cartogram objects (e.g., different parameters or iterations)

    Parameters
    ----------
    left : GeoDataFrame or Cartogram
        Left panel: original GeoDataFrame or a Cartogram to compare.
    right : Cartogram
        Right panel: Cartogram containing transformed geometries.
    facecolor : str or array-like, optional
        Fill of the geometries in both panels, resolved as in :func:`plot_cartogram`.
        A column that only the right panel has is drawn there only.
    iteration : int, optional
        Which iteration snapshot to plot for `right` (default: latest).
    left_iteration : int, optional
        Which iteration snapshot to plot for `left` when it's a Cartogram
        (default: latest). Ignored if `left` is a GeoDataFrame.
    figsize : tuple, default=(14, 6)
        Figure size (width, height).
    **kwargs
        Additional arguments passed to GeoDataFrame.plot()

    Returns
    -------
    CartogramComparisonResult
        Result with the figure, both axes, and named artist references.

    Examples
    --------
    >>> from carto_flow import create_flow_cartogram
    >>> from carto_flow.flow_cartogram.visualization import plot_comparison
    >>> cartogram = create_flow_cartogram(gdf, 'population')
    >>> # Compare original GeoDataFrame with cartogram result
    >>> plot_comparison(gdf, cartogram)
    >>> plot_comparison(gdf, cartogram, facecolor='population')
    >>> # Compare two cartograms (e.g., different parameters)
    >>> cartogram_fast = create_flow_cartogram(gdf, 'population', options=MorphOptions.preset_fast())
    >>> cartogram_quality = create_flow_cartogram(gdf, 'population', options=MorphOptions.preset_quality())
    >>> plot_comparison(cartogram_fast, cartogram_quality)
    """
    from .cartogram import Cartogram

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Determine if left is a Cartogram or GeoDataFrame
    left_is_cartogram = isinstance(left, Cartogram)

    # Plot left panel
    if left_is_cartogram:
        left_gdf = left.to_geodataframe(iteration=left_iteration)
        left_title = "Cartogram (left)"
        if left.status is not None:
            status_str = left.status.value if hasattr(left.status, "value") else str(left.status)
            left_title = f"Cartogram (status: {status_str})"
    else:
        left_gdf = left
        left_title = "Original"

    right_gdf = right.to_geodataframe(iteration=iteration, include_errors=True, include_density=True)
    right_gdf, right_fill = _fill_arguments(right_gdf, facecolor)
    left_fill: dict[str, Any] = {}
    if (
        hasattr(left_gdf, "plot")
        and facecolor is not None
        and ("column" not in right_fill or right_fill["column"] in left_gdf.columns)
    ):
        left_gdf, left_fill = _fill_arguments(left_gdf, facecolor)

    before_left = len(ax1.collections)
    if hasattr(left_gdf, "plot"):
        if "column" in left_fill:
            left_gdf.plot(ax=ax1, legend=True, **left_fill, **kwargs)
        else:
            left_gdf.plot(ax=ax1, **left_fill, **kwargs)
    left_collections = list(ax1.collections[before_left:])
    ax1.set_aspect("equal")
    ax1.set_title(left_title)

    # Plot right panel (always a Cartogram)
    before_right = len(ax2.collections)
    if hasattr(right_gdf, "plot"):
        if "column" in right_fill:
            right_gdf.plot(ax=ax2, legend=True, **right_fill, **kwargs)
        else:
            right_gdf.plot(ax=ax2, **right_fill, **kwargs)
    right_collections = list(ax2.collections[before_right:])
    ax2.set_aspect("equal")
    status_str = right.status.value if hasattr(right.status, "value") else str(right.status)
    ax2.set_title(f"Cartogram (status: {status_str})")

    plt.tight_layout()
    from .plot_results import CartogramComparisonResult

    return CartogramComparisonResult(
        fig=fig,
        ax_left=ax1,
        ax_right=ax2,
        left_collections=left_collections,
        right_collections=right_collections,
    )


def plot_convergence(
    cartogram: "Cartogram",
    ax: Optional["Axes"] = None,
    show_both: bool = True,
    use_pct: bool = True,
    show_tolerance: bool = True,
    show_recompute: bool = False,
    **kwargs: Any,
) -> "ConvergencePlotResult":
    """Plot convergence metrics (mean/max error) over iterations.

    Parameters
    ----------
    cartogram : Cartogram
        Cartogram result object. Uses convergence history if available
        (all iterations), otherwise falls back to snapshots (sparse).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    show_both : bool, default=True
        If True, shows both mean and max error on separate y-axes.
        If False, only shows mean error.
    use_pct : bool, default=True
        If True, shows percentage errors (more interpretable).
        If False, shows log2 errors (raw algorithm metric).
    show_tolerance : bool, default=True
        If True, shows horizontal lines indicating error tolerances
        from the MorphOptions used for computation.
    show_recompute : bool, default=False
        If True, shows vertical lines indicating iterations where
        density fields were recomputed.
    **kwargs
        Additional arguments passed to plt.plot()

    Returns
    -------
    ConvergencePlotResult
        Result with the axes and named artist references.

    Examples
    --------
    >>> from carto_flow import create_flow_cartogram
    >>> from carto_flow.flow_cartogram.visualization import plot_convergence
    >>> cartogram = create_flow_cartogram(gdf, 'population')
    >>> plot_convergence(cartogram)
    >>> plot_convergence(cartogram, show_recompute=True)
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Prefer convergence history (all iterations) over snapshots (sparse)
    if cartogram.convergence is not None and len(cartogram.convergence) > 0:
        iterations = cartogram.convergence.iterations
        if use_pct:
            mean_errors = cartogram.convergence.mean_errors_pct
            max_errors = cartogram.convergence.max_errors_pct
        else:
            mean_errors = cartogram.convergence.mean_log_errors
            max_errors = cartogram.convergence.max_log_errors
    else:
        # Fall back to extracting from snapshots (legacy behavior)
        iterations = []  # type: ignore[assignment]
        mean_errors = []  # type: ignore[assignment]
        max_errors = []  # type: ignore[assignment]

        for snapshot in cartogram.snapshots.snapshots:
            iterations.append(snapshot.iteration)  # type: ignore[attr-defined]
            if snapshot.errors is not None:
                if use_pct:
                    mean_errors.append(snapshot.errors.mean_error_pct)  # type: ignore[attr-defined]
                    max_errors.append(snapshot.errors.max_error_pct)  # type: ignore[attr-defined]
                else:
                    mean_errors.append(snapshot.errors.mean_log_error)  # type: ignore[attr-defined]
                    max_errors.append(snapshot.errors.max_log_error)  # type: ignore[attr-defined]

    # Colors for the two error types
    max_color = "C0"  # Blue
    mean_color = "C1"  # Orange

    # Track returned artists
    max_line: Line2D | None = None
    mean_line: Line2D | None = None
    ax_twin: Axes | None = None
    tolerance_lines: list[Line2D] = []
    recompute_lines: list[Line2D] = []

    # Plot max error on primary axis
    if show_both and len(max_errors) > 0:
        lines = ax.plot(
            iterations[: len(max_errors)],
            max_errors,
            label="Max Error",
            color=max_color,
            marker="o",
            markersize=3,
            **kwargs,
        )
        max_line = lines[0]
        ax.set_ylabel("Max Error (%)" if use_pct else "Max Error (log2)", color=max_color)
        ax.tick_params(axis="y", labelcolor=max_color)

        # Create secondary y-axis for mean error
        if len(mean_errors) > 0:
            ax_twin = ax.twinx()
            lines = ax_twin.plot(
                iterations[: len(mean_errors)],
                mean_errors,
                label="Mean Error",
                color=mean_color,
                marker="o",
                markersize=3,
                **kwargs,
            )
            mean_line = lines[0]
            ax_twin.set_ylabel("Mean Error (%)" if use_pct else "Mean Error (log2)", color=mean_color)
            ax_twin.tick_params(axis="y", labelcolor=mean_color)

            # Add tolerance lines
            if show_tolerance and cartogram.options is not None:
                mean_tol = cartogram.options.mean_tol
                if mean_tol is not None:
                    # Convert to percentage if using percentage mode
                    tol_value = mean_tol * 100 if use_pct else mean_tol
                    tol_line = ax_twin.axhline(
                        y=tol_value,
                        color=mean_color,
                        linestyle="-",
                        linewidth=1,
                        alpha=0.7,
                        label=f"Mean Tol ({mean_tol * 100:.1f}%)",
                    )
                    tolerance_lines.append(tol_line)
                max_tol = cartogram.options.max_tol
                if max_tol is not None:
                    tol_value = max_tol * 100 if use_pct else max_tol
                    tol_line = ax.axhline(
                        y=tol_value,
                        color=max_color,
                        linestyle="-",
                        linewidth=1,
                        alpha=0.7,
                        label=f"Max Tol ({max_tol * 100:.1f}%)",
                    )
                    tolerance_lines.append(tol_line)

            # Combined legend from both axes
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_twin.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    elif len(mean_errors) > 0:
        # Only mean error requested
        lines = ax.plot(
            iterations[: len(mean_errors)],
            mean_errors,
            label="Mean Error",
            color=mean_color,
            marker="o",
            markersize=3,
            **kwargs,
        )
        mean_line = lines[0]
        ax.set_ylabel("Mean Error (%)" if use_pct else "Mean Error (log2)")

        # Add tolerance line for mean error
        if show_tolerance and cartogram.options is not None:
            mean_tol = cartogram.options.mean_tol
            if mean_tol is not None:
                tol_value = mean_tol * 100 if use_pct else mean_tol
                tol_line = ax.axhline(
                    y=tol_value,
                    color=mean_color,
                    linestyle="-",
                    linewidth=1,
                    alpha=0.7,
                    label=f"Mean Tol ({mean_tol * 100:.1f}%)",
                )
                tolerance_lines.append(tol_line)
        ax.legend(loc="upper right")

    # Add vertical lines for density recomputation iterations
    if show_recompute and cartogram.options is not None:
        recompute_every = cartogram.options.recompute_every
        if recompute_every is not None and len(iterations) > 0:
            max_iter = int(np.max(iterations))
            recompute_iters = list(range(0, max_iter + 1, recompute_every))
            for i, recomp_iter in enumerate(recompute_iters):
                vline = ax.axvline(
                    x=recomp_iter,
                    color="gray",
                    linestyle="--",
                    alpha=0.4,
                    label="Density recompute" if i == 0 else None,
                )
                recompute_lines.append(vline)

    ax.set_xlabel("Iteration")
    ax.set_title("Convergence History")
    ax.grid(True, alpha=0.3)

    from .plot_results import ConvergencePlotResult

    return ConvergencePlotResult(
        ax=ax,
        max_line=max_line,
        mean_line=mean_line,
        ax_mean=ax_twin,
        tolerance_lines=tolerance_lines,
        recompute_lines=recompute_lines,
    )


def plot_density_field(
    cartogram: "Cartogram",
    *,
    iteration: int | None = None,
    ax: Optional["Axes"] = None,
    bounds: Any | None = None,
    density: Optional["DensityPlotOptions"] = None,
) -> "DensityFieldResult":
    """Visualize density field as a heatmap.

    Parameters
    ----------
    cartogram : Cartogram
        Cartogram containing internals with density field data.
        Requires morphing with `save_internals=True` in MorphOptions.
    iteration : int, optional
        Which iteration snapshot to visualize (default: latest).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    bounds : str, float, or tuple, optional
        Clip view to specified bounds (removes grid margin):
        - None: Show full grid extent (default)
        - "data": Clip to original data bounds (no margin)
        - float: Data bounds + margin as fraction (e.g., 0.1 = 10% margin)
        - tuple: (xmin, ymin, xmax, ymax) custom bounds
    density : DensityPlotOptions, optional
        Rendering options for the density field. Key fields:

        - ``normalize``: None (absolute), ``"difference"``, or ``"ratio"``.
        - ``clip_percentile``: Clip outliers for difference mode.
        - ``max_scale``: Fixed symmetric color scale maximum.
        - ``cmap``: Override colormap.
        - ``vmin``/``vmax``: Override color-scale limits.
        - ``legend_kwargs``: Passed to ``plt.colorbar()``.
        - ``imshow_kwargs``: Passed to ``ax.imshow()``.

    Returns
    -------
    DensityFieldResult
        Result with the axes and named artist references.

    Examples
    --------
    >>> from carto_flow import create_flow_cartogram, MorphOptions
    >>> from carto_flow.flow_cartogram.visualization import plot_density_field, DensityPlotOptions
    >>> cartogram = create_flow_cartogram(gdf, 'pop', options=MorphOptions(save_internals=True))
    >>> plot_density_field(cartogram)

    >>> # Clip to original data bounds (no margin)
    >>> plot_density_field(cartogram, bounds="data")

    >>> # Show density difference from equilibrium (centered at 0)
    >>> plot_density_field(cartogram, density=DensityPlotOptions(normalize="difference"))

    >>> # Clip outliers to [2nd, 98th] percentiles for difference mode
    >>> plot_density_field(cartogram, density=DensityPlotOptions(normalize="difference", clip_percentile=2))

    >>> # Show density ratio to equilibrium (centered at 1)
    >>> plot_density_field(cartogram, density=DensityPlotOptions(normalize="ratio"))

    >>> # Compare first and last iterations with consistent color scale
    >>> fig, (ax1, ax2) = plt.subplots(1, 2)
    >>> opts = DensityPlotOptions(normalize="ratio", max_scale=4)
    >>> plot_density_field(cartogram, iteration=0, ax=ax1, density=opts)
    >>> plot_density_field(cartogram, ax=ax2, density=opts)
    """
    opts = density or DensityPlotOptions()

    # Extract internals from cartogram
    if cartogram.internals is None:
        raise ValueError(
            "Cartogram does not contain internals data. Re-run morphing with MorphOptions(save_internals=True)."
        )

    snapshot = cartogram.internals.best() if iteration is None else cartogram.internals.get_snapshot(iteration)

    if snapshot is None:
        raise ValueError(f"No internals snapshot found for iteration {iteration}")

    grid = cartogram.grid
    if grid is None:
        raise ValueError("Cartogram does not contain grid data.")

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 8))

    if not hasattr(snapshot, "rho") or snapshot.rho is None:
        raise ValueError("Snapshot does not contain density field (rho)")

    rho = snapshot.rho
    extent = [grid.xmin, grid.xmax, grid.ymin, grid.ymax]
    target_density = cartogram.target_density
    geometry_mask = getattr(snapshot, "geometry_mask", None)

    im, colorbar_label = _render_density_on_ax(
        ax,
        rho,
        target_density,
        extent,
        opts,
        geometry_mask=geometry_mask,
    )

    cbar_kw = {"label": colorbar_label, **opts.legend_kwargs}
    cbar = plt.colorbar(im, ax=ax, **cbar_kw)

    # Apply custom tick formatter for ratio mode (colorbar shows ratio, not log2)
    if opts.normalize == "ratio":
        from matplotlib.ticker import FuncFormatter

        def _ratio_fmt(x, pos):
            ratio_val = 2**x
            if ratio_val >= 1:
                return f"{int(ratio_val)}" if ratio_val == int(ratio_val) else f"{ratio_val:.1f}"
            return f"{ratio_val:.2f}" if ratio_val >= 0.1 else f"{ratio_val:.3f}"

        cbar.ax.yaxis.set_major_formatter(FuncFormatter(_ratio_fmt))

    # Determine title suffix
    if opts.normalize == "difference":
        title_suffix = f" (difference, clipped {opts.clip_percentile}%)" if opts.clip_percentile else " (difference)"
    elif opts.normalize == "ratio":
        title_suffix = " (ratio, log scale)"
    else:
        title_suffix = ""

    ax.set_title(f"Density Field (iteration {snapshot.iteration}){title_suffix}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Apply bounds clipping if specified
    resolved_bounds = _resolve_bounds(bounds, grid)
    if resolved_bounds is not None:
        xmin, ymin, xmax, ymax = resolved_bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    from .plot_results import DensityFieldResult

    return DensityFieldResult(ax=ax, image=im, colorbar=cbar)


def plot_velocity_field(
    cartogram: "Cartogram",
    *,
    iteration: int | None = None,
    ax: Optional["Axes"] = None,
    bounds: Any | None = None,
    velocity: Optional["VelocityPlotOptions"] = None,
) -> "VelocityFieldResult":
    """Visualize velocity field as a quiver plot.

    Parameters
    ----------
    cartogram : Cartogram
        Cartogram containing internals with velocity field data.
        Requires morphing with `save_internals=True` in MorphOptions.
    iteration : int, optional
        Which iteration snapshot to visualize (default: latest).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    bounds : str, float, or tuple, optional
        Clip view to specified bounds (removes grid margin):
        - None: Show full grid extent (default)
        - "data": Clip to original data bounds (no margin)
        - float: Data bounds + margin as fraction (e.g., 0.1 = 10% margin)
        - tuple: (xmin, ymin, xmax, ymax) custom bounds
    velocity : VelocityPlotOptions, optional
        Rendering options for the velocity field. Key fields:

        - ``skip``: Plot every nth arrow (default 4).
        - ``velocity_scale``: Arrow length as fraction of grid cell spacing.
        - ``ref_magnitude``: Reference magnitude for consistent scaling.
        - ``color`` / ``alpha``: Base arrow color and transparency.
        - ``color_by``: None, ``"magnitude"``, or ``"direction"``.
        - ``cmap``: Colormap when ``color_by`` is set.
        - ``legend``: Whether to show a colorbar (default True).
        - ``alpha_by_magnitude``: Vary alpha with velocity magnitude.
        - ``alpha_range``: Min/max alpha for magnitude-based transparency.
        - ``legend_kwargs``: Passed to ``plt.colorbar()``.
        - ``quiver_kwargs``: Passed to ``ax.quiver()``.

    Returns
    -------
    VelocityFieldResult
        Result with the axes and named artist references.

    Examples
    --------
    >>> from carto_flow import create_flow_cartogram, MorphOptions
    >>> from carto_flow.flow_cartogram.visualization import plot_velocity_field, VelocityPlotOptions
    >>> cartogram = create_flow_cartogram(gdf, 'pop', options=MorphOptions(save_internals=True))
    >>> plot_velocity_field(cartogram)

    >>> # Color by magnitude
    >>> plot_velocity_field(cartogram, velocity=VelocityPlotOptions(color_by="magnitude", cmap="plasma"))

    >>> # Color by direction (uses cyclic colormap by default)
    >>> plot_velocity_field(cartogram, velocity=VelocityPlotOptions(color_by="direction"))

    >>> # Transparency by magnitude (fade out weak velocities)
    >>> plot_velocity_field(cartogram, velocity=VelocityPlotOptions(alpha_by_magnitude=True))

    >>> # Compare two velocity fields with consistent arrow scaling
    >>> fig, (ax1, ax2) = plt.subplots(1, 2)
    >>> opts = VelocityPlotOptions(velocity_scale=1.0, ref_magnitude=0.5)
    >>> plot_velocity_field(cartogram, iteration=0, ax=ax1, velocity=opts)
    >>> plot_velocity_field(cartogram, ax=ax2, velocity=opts)
    """
    opts = velocity or VelocityPlotOptions()

    # Extract internals from cartogram
    if cartogram.internals is None:
        raise ValueError(
            "Cartogram does not contain internals data. Re-run morphing with MorphOptions(save_internals=True)."
        )

    snapshot = cartogram.internals.best() if iteration is None else cartogram.internals.get_snapshot(iteration)

    if snapshot is None:
        raise ValueError(f"No internals snapshot found for iteration {iteration}")

    grid = cartogram.grid
    if grid is None:
        raise ValueError("Cartogram does not contain grid data.")

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 8))

    if not hasattr(snapshot, "vx") or snapshot.vx is None:
        raise ValueError("Snapshot does not contain velocity field (vx)")
    if not hasattr(snapshot, "vy") or snapshot.vy is None:
        raise ValueError("Snapshot does not contain velocity field (vy)")

    vx = snapshot.vx
    vy = snapshot.vy

    # Create coordinate meshgrid
    X, Y = grid.X, grid.Y

    # Subsample for clearer visualization
    X_sub = X[:: opts.skip, :: opts.skip]
    Y_sub = Y[:: opts.skip, :: opts.skip]
    vx_sub = vx[:: opts.skip, :: opts.skip]
    vy_sub = vy[:: opts.skip, :: opts.skip]

    # Compute magnitude for arrow scaling and colorbar
    magnitude = np.sqrt(vx_sub**2 + vy_sub**2)
    mag_min, mag_max = magnitude.min(), magnitude.max()

    # Build quiver kwargs for consistent arrow scaling
    quiver_kw = {
        **opts.quiver_kwargs,
        **_compute_quiver_scale(mag_max, opts.skip, grid.dx, opts.velocity_scale, opts.ref_magnitude),
    }

    # Use shared helper for velocity colors
    colors, colorbar_label, effective_cmap = _prepare_velocity_colors(
        vx_sub,
        vy_sub,
        color_by=opts.color_by,
        alpha_by_magnitude=opts.alpha_by_magnitude,
        alpha_range=opts.alpha_range,
        cmap=opts.cmap,
        base_color=opts.color,
        base_alpha=opts.alpha,
    )

    quiver_art = ax.quiver(X_sub, Y_sub, vx_sub, vy_sub, color=colors, **quiver_kw)

    colorbar_art: Colorbar | None = None
    if opts.legend and opts.color_by is not None:
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize

        norm = (
            Normalize(vmin=mag_min, vmax=mag_max)
            if opts.color_by == "magnitude"
            else Normalize(vmin=-np.pi, vmax=np.pi)
        )
        cmap_obj = plt.get_cmap(effective_cmap)
        sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar_kw = {"label": colorbar_label, **opts.legend_kwargs}
        colorbar_art = plt.colorbar(sm, ax=ax, **cbar_kw)

    ax.set_aspect("equal")
    ax.set_title(f"Velocity Field (iteration {snapshot.iteration})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Apply bounds clipping if specified
    resolved_bounds = _resolve_bounds(bounds, grid)
    if resolved_bounds is not None:
        xmin, ymin, xmax, ymax = resolved_bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    from .plot_results import VelocityFieldResult

    return VelocityFieldResult(ax=ax, arrows=quiver_art, colorbar=colorbar_art)


def plot_workflow_convergence(
    workflow: "CartogramWorkflow",
    ax: Optional["Axes"] = None,
    use_pct: bool = True,
    show_tolerance: bool = True,
    show_run_boundaries: bool = True,
    metric: str = "both",
    color_by_run: bool = True,
    cmap: str = "tab10",
    figsize: tuple[float, float] = (12, 6),
    **kwargs: Any,
) -> "WorkflowConvergencePlotResult":
    """Plot convergence history across all cartograms in a workflow.

    Shows the sequential convergence of multiple morphing runs on a single plot,
    with cumulative iterations on the x-axis and run boundaries indicated.
    Useful for visualizing multi-resolution or iterative refinement workflows.

    Parameters
    ----------
    workflow : CartogramWorkflow
        Workflow containing multiple cartogram results.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    use_pct : bool, default=True
        Show percentage errors (True) or log2 errors (False).
    show_tolerance : bool, default=True
        Show horizontal tolerance lines from MorphOptions.
    show_run_boundaries : bool, default=True
        Show vertical dashed lines at run boundaries.
    metric : str, default="both"
        Which error metric to show: "mean", "max", or "both".
        When "both", uses dual y-axes (max on left, mean on right).
    color_by_run : bool, default=True
        Use different colors for each run. If False, uses single color.
    cmap : str, default="tab10"
        Colormap for distinguishing runs when color_by_run=True.
    figsize : tuple, default=(12, 6)
        Figure size (width, height) if creating new figure.
    **kwargs
        Additional arguments passed to plt.plot().

    Returns
    -------
    WorkflowConvergencePlotResult
        Result with the axes and named artist references.

    Examples
    --------
    >>> workflow = CartogramWorkflow(gdf, 'population')
    >>> workflow.morph_multiresolution(levels=3)
    >>> plot_workflow_convergence(workflow)

    >>> # Show only mean error with custom colors
    >>> plot_workflow_convergence(workflow, metric="mean", cmap="Set2")

    >>> # Show log2 errors instead of percentage
    >>> plot_workflow_convergence(workflow, use_pct=False)
    """
    if metric not in ("mean", "max", "both"):
        raise ValueError(f"metric must be 'mean', 'max', or 'both', got {metric!r}")

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    # Get colormap for runs
    cmap_obj = plt.get_cmap(cmap)
    n_runs = len(workflow)

    # Collect convergence data from all cartograms with cumulative iteration offset
    run_data: list[dict[str, Any]] = []
    iter_offset = 0
    run_boundaries: list[int] = [0]

    for run_idx, cartogram in enumerate(workflow):
        run_info: dict[str, Any] = {
            "run_idx": run_idx,
            "iter_offset": iter_offset,
            "iterations": np.array([]),
            "mean_errors": np.array([]),
            "max_errors": np.array([]),
            "options": cartogram.options,
            "status": cartogram.status,
        }

        # Prefer convergence history (all iterations) over snapshots (sparse)
        if cartogram.convergence is not None and len(cartogram.convergence) > 0:
            raw_iters = cartogram.convergence.iterations
            if use_pct:
                run_info["mean_errors"] = cartogram.convergence.mean_errors_pct
                run_info["max_errors"] = cartogram.convergence.max_errors_pct
            else:
                run_info["mean_errors"] = cartogram.convergence.mean_log_errors
                run_info["max_errors"] = cartogram.convergence.max_log_errors
            run_info["iterations"] = raw_iters + iter_offset
            if len(raw_iters) > 0:
                iter_offset = run_info["iterations"][-1]
        else:
            # Fall back to snapshots
            iterations = []
            mean_errors = []
            max_errors = []
            for snap in cartogram.snapshots.snapshots:
                iterations.append(snap.iteration)
                if snap.errors is not None:
                    if use_pct:
                        mean_errors.append(snap.errors.mean_error_pct)
                        max_errors.append(snap.errors.max_error_pct)
                    else:
                        mean_errors.append(snap.errors.mean_log_error)
                        max_errors.append(snap.errors.max_log_error)
            if iterations:
                run_info["iterations"] = np.array(iterations) + iter_offset
                run_info["mean_errors"] = np.array(mean_errors)
                run_info["max_errors"] = np.array(max_errors)
                iter_offset = run_info["iterations"][-1]

        run_data.append(run_info)
        run_boundaries.append(iter_offset)

    # Colors for metrics
    max_color_base = "C0"  # Blue for max error
    mean_color_base = "C1"  # Orange for mean error

    # Track returned artists
    max_lines: list[Line2D] = []
    mean_lines: list[Line2D] = []
    boundary_lines: list[Line2D] = []
    tolerance_lines: list[Line2D] = []

    # Create secondary axis for mean error if showing both
    ax2: Axes | None = None
    if metric == "both":
        ax2 = ax.twinx()

    # Plot each run's convergence
    for run_info in run_data:
        run_idx = run_info["run_idx"]
        iterations = run_info["iterations"]
        mean_errors = run_info["mean_errors"]
        max_errors = run_info["max_errors"]

        if len(iterations) == 0:
            continue

        # Determine color for this run
        run_color = (cmap_obj(run_idx / max(1, n_runs - 1)) if n_runs > 1 else cmap_obj(0.5)) if color_by_run else None

        label_suffix = f" (Run {run_idx})" if n_runs > 1 else ""

        # Plot max error on primary axis
        if metric in ("max", "both") and len(max_errors) > 0:
            color = run_color if color_by_run else max_color_base
            lines = ax.plot(
                iterations[: len(max_errors)],
                max_errors,
                label=f"Max Error{label_suffix}" if run_idx == 0 or not color_by_run else None,
                color=color,
                marker="o" if len(iterations) < 50 else None,
                markersize=3,
                linestyle="-",
                **kwargs,
            )
            if lines:
                max_lines.append(lines[0])

        # Plot mean error (on secondary axis if showing both)
        if metric in ("mean", "both") and len(mean_errors) > 0:
            target_ax = ax2 if metric == "both" else ax
            color = run_color if color_by_run else mean_color_base
            linestyle = "--" if metric == "both" and not color_by_run else "-"
            lines = target_ax.plot(  # type: ignore[union-attr]
                iterations[: len(mean_errors)],
                mean_errors,
                label=f"Mean Error{label_suffix}" if run_idx == 0 or not color_by_run else None,
                color=color,
                marker="o" if len(iterations) < 50 else None,
                markersize=3,
                linestyle=linestyle,
                **kwargs,
            )
            if lines:
                mean_lines.append(lines[0])

    # Configure y-axis labels
    y_label = "Error (%)" if use_pct else "Error (log2)"
    if metric == "both":
        ax.set_ylabel(f"Max {y_label}", color=max_color_base if not color_by_run else "black")
        if not color_by_run:
            ax.tick_params(axis="y", labelcolor=max_color_base)
        if ax2 is not None:
            ax2.set_ylabel(f"Mean {y_label}", color=mean_color_base if not color_by_run else "black")
            if not color_by_run:
                ax2.tick_params(axis="y", labelcolor=mean_color_base)
    elif metric == "max":
        ax.set_ylabel(f"Max {y_label}")
    else:
        ax.set_ylabel(f"Mean {y_label}")

    # Add run boundary lines
    if show_run_boundaries and len(run_boundaries) > 2:
        for i, boundary in enumerate(run_boundaries[1:-1], start=1):
            vline = ax.axvline(
                x=boundary,
                color="gray",
                linestyle="--",
                alpha=0.5,
                linewidth=1,
                label="Run boundary" if i == 1 else None,
            )
            boundary_lines.append(vline)

    # Add tolerance lines from latest options
    if show_tolerance:
        latest_options = workflow.last.options
        if latest_options is not None:
            if metric in ("mean", "both") and latest_options.mean_tol is not None:
                tol_value = latest_options.mean_tol * 100 if use_pct else latest_options.mean_tol
                target_ax = ax2 if metric == "both" else ax
                tol_line = target_ax.axhline(  # type: ignore[union-attr]
                    y=tol_value,
                    color=mean_color_base,
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.7,
                    label=f"Mean Tol ({latest_options.mean_tol * 100:.1f}%)",
                )
                tolerance_lines.append(tol_line)
            if metric in ("max", "both") and latest_options.max_tol is not None:
                tol_value = latest_options.max_tol * 100 if use_pct else latest_options.max_tol
                tol_line = ax.axhline(
                    y=tol_value,
                    color=max_color_base,
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.7,
                    label=f"Max Tol ({latest_options.max_tol * 100:.1f}%)",
                )
                tolerance_lines.append(tol_line)

    # Legend
    if metric == "both" and ax2 is not None:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax.legend(loc="upper right")

    ax.set_xlabel("Cumulative Iteration")
    ax.set_title("Workflow Convergence History")
    ax.grid(True, alpha=0.3)

    from .plot_results import WorkflowConvergencePlotResult

    return WorkflowConvergencePlotResult(
        ax=ax,
        ax_mean=ax2,
        max_lines=max_lines,
        mean_lines=mean_lines,
        boundary_lines=boundary_lines,
        tolerance_lines=tolerance_lines,
    )
