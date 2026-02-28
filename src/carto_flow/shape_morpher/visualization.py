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
>>> from carto_flow.shape_morpher import morph_gdf, MorphOptions
>>> from carto_flow.shape_morpher.visualization import plot_comparison, plot_convergence
>>> cartogram = morph_gdf(gdf, "population", options=MorphOptions.preset_fast())
>>> plot_comparison(gdf, cartogram)
>>> plot_convergence(cartogram.snapshots)
"""

from typing import TYPE_CHECKING, Any, Optional

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .cartogram import Cartogram
    from .grid import Grid
    from .workflow import CartogramWorkflow

__all__ = [
    "plot_cartogram",
    "plot_comparison",
    "plot_convergence",
    "plot_density_field",
    "plot_velocity_field",
    "plot_workflow_convergence",
]


def _resolve_bounds(
    bounds: Any,
    grid: "Grid",
) -> Optional[tuple[float, float, float, float]]:
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
    target_density: float,
    normalize: Optional[str],
    area_scale: float = 1.0,
    clip_percentile: Optional[float] = None,
    max_scale: Optional[float] = None,
    outside_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[float], Optional[float], str, str, Optional[Any]]:
    """Prepare density field for display with optional normalization.

    This internal helper is used by both plot_density_field and animate_fields
    to ensure consistent normalization behavior.

    Parameters
    ----------
    rho : np.ndarray
        Raw density field array (in CRS units).
    target_density : float
        Target equilibrium density for normalization.
    normalize : str or None
        Normalization mode: None, "difference", or "ratio".
    area_scale : float, default=1.0
        Scale factor for converting density units.
    clip_percentile : float, optional
        Percentile for clipping outliers (difference mode only).
    max_scale : float, optional
        Fixed maximum for symmetric color scale.
    outside_mask : np.ndarray, optional
        Boolean mask for cells outside geometries.

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

    # Scale rho to user-friendly units (matching target_density)
    rho_scaled = rho / area_scale

    if normalize is None:
        return rho_scaled, None, None, "viridis", "Density", None

    if normalize not in ("difference", "ratio"):
        raise ValueError(f"normalize must be 'difference', 'ratio', or None, got {normalize!r}")

    if target_density is None or target_density <= 0:
        raise ValueError(
            f"target_density must be positive, got {target_density}. Ensure the cartogram was computed successfully."
        )

    if normalize == "difference":
        rho_display = rho_scaled - target_density
        colorbar_label = "Density - Mean"

        if max_scale is not None:
            max_deviation = max_scale
        elif clip_percentile is not None:
            inside_values = rho_display[~outside_mask] if outside_mask is not None else rho_display.ravel()
            p_low = np.percentile(inside_values, clip_percentile)
            p_high = np.percentile(inside_values, 100 - clip_percentile)
            max_deviation = max(abs(p_low), abs(p_high))
        else:
            data_min = rho_display.min()
            data_max = rho_display.max()
            max_deviation = max(abs(data_min), abs(data_max))

        vmin = -max_deviation
        vmax = max_deviation
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cmap = "RdBu_r"

    else:  # ratio
        ratio = rho_scaled / target_density
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


def _prepare_velocity_colors(
    vx: np.ndarray,
    vy: np.ndarray,
    color_by: Optional[str] = None,
    alpha_by_magnitude: bool = False,
    alpha_range: tuple[float, float] = (0.2, 1.0),
    cmap: Optional[str] = None,
    base_color: str = "white",
    base_alpha: float = 0.8,
    mag_range: Optional[tuple[float, float]] = None,
) -> tuple[np.ndarray, Optional[str], Optional[str]]:
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


def plot_cartogram(
    cartogram: "Cartogram",
    ax: Optional["Axes"] = None,
    column: Optional[str] = None,
    iteration: Optional[int] = None,
    cmap: str = "RdYlGn_r",
    legend: bool = True,
    **kwargs: Any,
) -> "Axes":
    """Plot morphed geometries from a Cartogram.

    Parameters
    ----------
    cartogram : Cartogram
        Cartogram containing morphed geometries to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    column : str, optional
        Column to use for coloring geometries. Special values:
        - '_morph_error_pct': Percentage errors (default if available)
        - '_morph_density': Density values
        - Any column from source GeoDataFrame
        If None, uses '_morph_error_pct' if available, otherwise no coloring.
    iteration : int, optional
        Which iteration snapshot to plot (default: latest).
    cmap : str, default='RdYlGn_r'
        Matplotlib colormap name.
    legend : bool, default=True
        Show colorbar legend when plotting by column.
    **kwargs
        Additional arguments passed to GeoDataFrame.plot()

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot

    Examples
    --------
    >>> from carto_flow import morph_gdf
    >>> from carto_flow.shape_morpher.visualization import plot_cartogram
    >>> cartogram = morph_gdf(gdf, 'population')
    >>> plot_cartogram(cartogram)
    >>> plot_cartogram(cartogram, column='population', cmap='Blues')
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Determine which columns to include based on requested column
    include_errors = column == "_morph_error_pct" or column is None
    include_density = column == "_morph_density"

    gdf = cartogram.to_geodataframe(
        iteration=iteration,
        include_errors=include_errors,
        include_density=include_density,
    )

    # Default to error column if no column specified and errors are available
    if column is None and "_morph_error_pct" in gdf.columns:
        column = "_morph_error_pct"

    if column is not None and column in gdf.columns:
        gdf.plot(ax=ax, column=column, cmap=cmap, legend=legend, **kwargs)
    else:
        gdf.plot(ax=ax, **kwargs)

    ax.set_aspect("equal")
    ax.set_title("Cartogram Result")
    return ax


def plot_comparison(
    left: Any,
    right: "Cartogram",
    column: Optional[str] = None,
    iteration: Optional[int] = None,
    left_iteration: Optional[int] = None,
    figsize: tuple[float, float] = (14, 6),
    **kwargs: Any,
) -> tuple["Figure", tuple["Axes", "Axes"]]:
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
    column : str, optional
        Column to use for coloring geometries.
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
    tuple[Figure, tuple[Axes, Axes]]
        The figure and a tuple of (left_ax, right_ax) axes

    Examples
    --------
    >>> from carto_flow import morph_gdf
    >>> from carto_flow.shape_morpher.visualization import plot_comparison
    >>> cartogram = morph_gdf(gdf, 'population')
    >>> # Compare original GeoDataFrame with cartogram result
    >>> plot_comparison(gdf, cartogram)
    >>> plot_comparison(gdf, cartogram, column='population')
    >>> # Compare two cartograms (e.g., different parameters)
    >>> cartogram_fast = morph_gdf(gdf, 'population', options=MorphOptions.preset_fast())
    >>> cartogram_quality = morph_gdf(gdf, 'population', options=MorphOptions.preset_quality())
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

    if hasattr(left_gdf, "plot"):
        if column and column in left_gdf.columns:
            left_gdf.plot(ax=ax1, column=column, legend=True, **kwargs)
        else:
            left_gdf.plot(ax=ax1, **kwargs)
    ax1.set_aspect("equal")
    ax1.set_title(left_title)

    # Plot right panel (always a Cartogram)
    right_gdf = right.to_geodataframe(iteration=iteration)
    if hasattr(right_gdf, "plot"):
        if column and column in right_gdf.columns:
            right_gdf.plot(ax=ax2, column=column, legend=True, **kwargs)
        else:
            right_gdf.plot(ax=ax2, **kwargs)
    ax2.set_aspect("equal")
    status_str = right.status.value if hasattr(right.status, "value") else str(right.status)
    ax2.set_title(f"Cartogram (status: {status_str})")

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_convergence(
    cartogram: "Cartogram",
    ax: Optional["Axes"] = None,
    show_both: bool = True,
    use_pct: bool = True,
    show_tolerance: bool = True,
    show_recompute: bool = False,
    **kwargs: Any,
) -> "Axes":
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
    matplotlib.axes.Axes
        The axes with the convergence plot (primary axis for max error)

    Examples
    --------
    >>> from carto_flow import morph_gdf
    >>> from carto_flow.shape_morpher.visualization import plot_convergence
    >>> cartogram = morph_gdf(gdf, 'population')
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
        iterations = []
        mean_errors = []
        max_errors = []

        for snapshot in cartogram.snapshots.snapshots:
            iterations.append(snapshot.iteration)
            if snapshot.errors is not None:
                if use_pct:
                    mean_errors.append(snapshot.errors.mean_error_pct)
                    max_errors.append(snapshot.errors.max_error_pct)
                else:
                    mean_errors.append(snapshot.errors.mean_log_error)
                    max_errors.append(snapshot.errors.max_log_error)

    # Colors for the two error types
    max_color = "C0"  # Blue
    mean_color = "C1"  # Orange

    # Plot max error on primary axis
    if show_both and len(max_errors) > 0:
        ax.plot(
            iterations[: len(max_errors)],
            max_errors,
            label="Max Error",
            color=max_color,
            marker="o",
            markersize=3,
            **kwargs,
        )
        ax.set_ylabel("Max Error (%)" if use_pct else "Max Error (log2)", color=max_color)
        ax.tick_params(axis="y", labelcolor=max_color)

        # Create secondary y-axis for mean error
        if len(mean_errors) > 0:
            ax2 = ax.twinx()
            ax2.plot(
                iterations[: len(mean_errors)],
                mean_errors,
                label="Mean Error",
                color=mean_color,
                marker="o",
                markersize=3,
                **kwargs,
            )
            ax2.set_ylabel("Mean Error (%)" if use_pct else "Mean Error (log2)", color=mean_color)
            ax2.tick_params(axis="y", labelcolor=mean_color)

            # Add tolerance lines
            if show_tolerance and cartogram.options is not None:
                mean_tol = cartogram.options.mean_tol
                if mean_tol is not None:
                    # Convert to percentage if using percentage mode
                    tol_value = mean_tol * 100 if use_pct else mean_tol
                    ax2.axhline(
                        y=tol_value,
                        color=mean_color,
                        linestyle="-",
                        linewidth=1,
                        alpha=0.7,
                        label=f"Mean Tol ({mean_tol * 100:.1f}%)",
                    )
                max_tol = cartogram.options.max_tol
                if max_tol is not None:
                    tol_value = max_tol * 100 if use_pct else max_tol
                    ax.axhline(
                        y=tol_value,
                        color=max_color,
                        linestyle="-",
                        linewidth=1,
                        alpha=0.7,
                        label=f"Max Tol ({max_tol * 100:.1f}%)",
                    )

            # Combined legend from both axes
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    elif len(mean_errors) > 0:
        # Only mean error requested
        ax.plot(
            iterations[: len(mean_errors)],
            mean_errors,
            label="Mean Error",
            color=mean_color,
            marker="o",
            markersize=3,
            **kwargs,
        )
        ax.set_ylabel("Mean Error (%)" if use_pct else "Mean Error (log2)")

        # Add tolerance line for mean error
        if show_tolerance and cartogram.options is not None:
            mean_tol = cartogram.options.mean_tol
            if mean_tol is not None:
                tol_value = mean_tol * 100 if use_pct else mean_tol
                ax.axhline(
                    y=tol_value,
                    color=mean_color,
                    linestyle="-",
                    linewidth=1,
                    alpha=0.7,
                    label=f"Mean Tol ({mean_tol * 100:.1f}%)",
                )
        ax.legend(loc="upper right")

    # Add vertical lines for density recomputation iterations
    if show_recompute and cartogram.options is not None:
        recompute_every = cartogram.options.recompute_every
        if recompute_every is not None and len(iterations) > 0:
            max_iter = int(np.max(iterations))
            recompute_iters = list(range(0, max_iter + 1, recompute_every))
            for i, recomp_iter in enumerate(recompute_iters):
                ax.axvline(
                    x=recomp_iter,
                    color="gray",
                    linestyle="--",
                    alpha=0.4,
                    label="Density recompute" if i == 0 else None,
                )

    ax.set_xlabel("Iteration")
    ax.set_title("Convergence History")
    ax.grid(True, alpha=0.3)

    return ax


def plot_density_field(
    cartogram: "Cartogram",
    iteration: Optional[int] = None,
    ax: Optional["Axes"] = None,
    bounds: Optional[Any] = None,
    normalize: Optional[str] = None,
    clip_percentile: Optional[float] = None,
    max_scale: Optional[float] = None,
    **kwargs: Any,
) -> "Axes":
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
    normalize : str, optional
        Show density relative to target_density (the target equilibrium):
        - None: Show absolute density values (default)
        - "difference": Show (rho - target_density), centered at 0.
          Positive values indicate denser regions, negative indicate sparser.
        - "ratio": Show (rho / target_density), centered at 1.0.
          Values >1 indicate denser regions, <1 indicate sparser.
        Both use a diverging colormap (RdBu_r).
    clip_percentile : float, optional
        Clip color range to exclude outliers (for difference mode only).
        Value in range [0, 100] specifies percentile from each end.
        E.g., clip_percentile=2 clips to [2nd, 98th] percentiles.
        Helps when extreme outliers dominate the color scale.
        Ignored for ratio mode (which uses log scale for outlier robustness).
    max_scale : float, optional
        Fixed maximum for symmetric color scale. Useful for comparing multiple
        snapshots with consistent coloring.
        - For difference mode: sets range to [-max_scale, max_scale]
        - For ratio mode: sets range to [1/max_scale, max_scale]
          E.g., max_scale=4 gives range [0.25, 4] (4x from equilibrium)
        Overrides automatic scaling and clip_percentile.
    **kwargs
        Additional arguments passed to plt.imshow()

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the density field plot

    Examples
    --------
    >>> from carto_flow import morph_gdf, MorphOptions
    >>> from carto_flow.shape_morpher.visualization import plot_density_field
    >>> cartogram = morph_gdf(gdf, 'pop', options=MorphOptions(save_internals=True))
    >>> plot_density_field(cartogram)

    >>> # Clip to original data bounds (no margin)
    >>> plot_density_field(cartogram, bounds="data")

    >>> # Clip with 10% margin around data bounds
    >>> plot_density_field(cartogram, bounds=0.1)

    >>> # Show density difference from equilibrium (centered at 0)
    >>> plot_density_field(cartogram, normalize="difference")

    >>> # Clip outliers to [2nd, 98th] percentiles for difference mode
    >>> plot_density_field(cartogram, normalize="difference", clip_percentile=2)

    >>> # Show density ratio to equilibrium (centered at 1)
    >>> plot_density_field(cartogram, normalize="ratio")

    >>> # Compare first and last iterations with consistent color scale
    >>> fig, (ax1, ax2) = plt.subplots(1, 2)
    >>> plot_density_field(cartogram, iteration=0, ax=ax1, normalize="ratio", max_scale=4)
    >>> plot_density_field(cartogram, ax=ax2, normalize="ratio", max_scale=4)
    """
    # Extract internals from cartogram
    if cartogram.internals is None:
        raise ValueError(
            "Cartogram does not contain internals data. Re-run morphing with MorphOptions(save_internals=True)."
        )

    snapshot = cartogram.internals.latest() if iteration is None else cartogram.internals.get_snapshot(iteration)

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

    # Get area_scale for unit conversion
    area_scale = cartogram.options.area_scale if cartogram.options else 1.0
    target_density = cartogram.target_density

    # Get outside_mask for percentile clipping
    outside_mask = getattr(snapshot, "outside_mask", None)

    # Use helper for density normalization
    rho_display, vmin, vmax, cmap_name, colorbar_label, norm = _prepare_density_display(
        rho,
        target_density,
        normalize,
        area_scale=area_scale,
        clip_percentile=clip_percentile,
        max_scale=max_scale,
        outside_mask=outside_mask,
    )

    # Allow user overrides via kwargs
    vmin = kwargs.pop("vmin", vmin)
    vmax = kwargs.pop("vmax", vmax)
    cmap = kwargs.pop("cmap", cmap_name)

    # Update norm if user provided custom vmin/vmax
    if normalize is not None and norm is not None:
        from matplotlib.colors import TwoSlopeNorm

        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    # Determine title suffix and colorbar formatter
    if normalize == "difference":
        title_suffix = f" (difference, clipped {clip_percentile}%)" if clip_percentile else " (difference)"
        colorbar_formatter = None
    elif normalize == "ratio":
        title_suffix = " (ratio, log scale)"

        # Custom formatter to show ratio values on colorbar
        def colorbar_formatter(x, pos):
            ratio_val = 2**x
            if ratio_val >= 1:
                if ratio_val == int(ratio_val):
                    return f"{int(ratio_val)}"
                return f"{ratio_val:.1f}"
            else:
                if ratio_val >= 0.1:
                    return f"{ratio_val:.2f}"
                return f"{ratio_val:.3f}"

    else:
        title_suffix = ""
        colorbar_formatter = None

    im = ax.imshow(
        rho_display,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=cmap,
        norm=norm,
        **kwargs,
    )
    cbar = plt.colorbar(im, ax=ax, label=colorbar_label)

    # Apply custom tick formatter for ratio mode
    if colorbar_formatter is not None:
        from matplotlib.ticker import FuncFormatter

        cbar.ax.yaxis.set_major_formatter(FuncFormatter(colorbar_formatter))
    ax.set_title(f"Density Field (iteration {snapshot.iteration}){title_suffix}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Apply bounds clipping if specified
    resolved_bounds = _resolve_bounds(bounds, grid)
    if resolved_bounds is not None:
        xmin, ymin, xmax, ymax = resolved_bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    return ax


def plot_velocity_field(
    cartogram: "Cartogram",
    iteration: Optional[int] = None,
    ax: Optional["Axes"] = None,
    skip: int = 4,
    arrow_scale: Optional[float] = None,
    ref_magnitude: Optional[float] = None,
    color_by: Optional[str] = None,
    cmap: Optional[str] = None,
    colorbar: bool = True,
    alpha_by_magnitude: bool = False,
    alpha_range: tuple[float, float] = (0.2, 1.0),
    bounds: Optional[Any] = None,
    **kwargs: Any,
) -> "Axes":
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
    skip : int, default=4
        Plot every nth arrow to reduce clutter.
    arrow_scale : float, optional
        Arrow length as a fraction of the grid cell spacing (after skip).
        For example, arrow_scale=1.0 means the reference velocity arrow
        spans one grid cell. If None, matplotlib auto-scales.
    ref_magnitude : float, optional
        Reference velocity magnitude for arrow scaling. An arrow with this
        magnitude will have length `arrow_scale * cell_size`. If None,
        uses the maximum magnitude in the current field. To compare
        multiple fields on the same scale, pass the same ref_magnitude
        to all plots (e.g., the global maximum across all fields).
    color_by : str, optional
        Color arrows by a property:
        - None: No coloring (default)
        - "magnitude": Color by velocity magnitude (sqrt(vx² + vy²))
        - "direction": Color by direction angle (-π to π)
    cmap : str, optional
        Colormap to use when color_by is specified.
        Defaults to "viridis" for magnitude, "twilight" for direction.
    colorbar : bool, default=True
        Whether to show colorbar when color_by is specified.
    alpha_by_magnitude : bool, default=False
        If True, arrow transparency varies with magnitude (stronger = more opaque).
    alpha_range : tuple[float, float], default=(0.2, 1.0)
        Min and max alpha values when alpha_by_magnitude is True.
    bounds : str, float, or tuple, optional
        Clip view to specified bounds (removes grid margin):
        - None: Show full grid extent (default)
        - "data": Clip to original data bounds (no margin)
        - float: Data bounds + margin as fraction (e.g., 0.1 = 10% margin)
        - tuple: (xmin, ymin, xmax, ymax) custom bounds
    **kwargs
        Additional arguments passed to plt.quiver()

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the velocity field plot

    Examples
    --------
    >>> from carto_flow import morph_gdf, MorphOptions
    >>> from carto_flow.shape_morpher.visualization import plot_velocity_field
    >>> cartogram = morph_gdf(gdf, 'pop', options=MorphOptions(save_internals=True))
    >>> plot_velocity_field(cartogram)

    >>> # Color by magnitude
    >>> plot_velocity_field(cartogram, color_by="magnitude", cmap="plasma")

    >>> # Color by direction (uses cyclic colormap by default)
    >>> plot_velocity_field(cartogram, color_by="direction")

    >>> # Transparency by magnitude (fade out weak velocities)
    >>> plot_velocity_field(cartogram, alpha_by_magnitude=True)

    >>> # Combine direction coloring with magnitude transparency
    >>> plot_velocity_field(cartogram, color_by="direction", alpha_by_magnitude=True)

    >>> # Compare two velocity fields with consistent arrow scaling
    >>> fig, (ax1, ax2) = plt.subplots(1, 2)
    >>> max_vel = 0.5  # Use same reference for both
    >>> plot_velocity_field(cartogram, iteration=0, ax=ax1, arrow_scale=1.0, ref_magnitude=max_vel)
    >>> plot_velocity_field(cartogram, ax=ax2, arrow_scale=1.0, ref_magnitude=max_vel)

    >>> # Clip to original data bounds (no margin)
    >>> plot_velocity_field(cartogram, bounds="data")

    >>> # Clip with 10% margin around data bounds
    >>> plot_velocity_field(cartogram, bounds=0.1)
    """
    # Extract internals from cartogram
    if cartogram.internals is None:
        raise ValueError(
            "Cartogram does not contain internals data. Re-run morphing with MorphOptions(save_internals=True)."
        )

    snapshot = cartogram.internals.latest() if iteration is None else cartogram.internals.get_snapshot(iteration)

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
    X_sub = X[::skip, ::skip]
    Y_sub = Y[::skip, ::skip]
    vx_sub = vx[::skip, ::skip]
    vy_sub = vy[::skip, ::skip]

    # Compute magnitude for arrow scaling
    magnitude = np.sqrt(vx_sub**2 + vy_sub**2)
    mag_min, mag_max = magnitude.min(), magnitude.max()

    # Build quiver kwargs for consistent arrow scaling
    quiver_kwargs = dict(kwargs)
    if arrow_scale is not None:
        cell_size = skip * grid.dx
        effective_ref = ref_magnitude if ref_magnitude is not None else mag_max
        if effective_ref > 0:
            quiver_kwargs["scale"] = effective_ref / (arrow_scale * cell_size)
            quiver_kwargs["scale_units"] = "xy"

    # Use shared helper for velocity colors
    colors, colorbar_label, effective_cmap = _prepare_velocity_colors(
        vx_sub,
        vy_sub,
        color_by=color_by,
        alpha_by_magnitude=alpha_by_magnitude,
        alpha_range=alpha_range,
        cmap=cmap,
        base_alpha=0.8,
    )

    ax.quiver(X_sub, Y_sub, vx_sub, vy_sub, color=colors, **quiver_kwargs)

    if colorbar and color_by is not None:
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize

        norm = Normalize(vmin=mag_min, vmax=mag_max) if color_by == "magnitude" else Normalize(vmin=-np.pi, vmax=np.pi)

        cmap_obj = plt.get_cmap(effective_cmap)
        sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=colorbar_label)

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

    return ax


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
) -> "Axes":
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
    matplotlib.axes.Axes
        The axes with the convergence plot (primary axis for max error).

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

    # Create secondary axis for mean error if showing both
    ax2 = None
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
            ax.plot(
                iterations[: len(max_errors)],
                max_errors,
                label=f"Max Error{label_suffix}" if run_idx == 0 or not color_by_run else None,
                color=color,
                marker="o" if len(iterations) < 50 else None,
                markersize=3,
                linestyle="-",
                **kwargs,
            )

        # Plot mean error (on secondary axis if showing both)
        if metric in ("mean", "both") and len(mean_errors) > 0:
            target_ax = ax2 if metric == "both" else ax
            color = run_color if color_by_run else mean_color_base
            linestyle = "--" if metric == "both" and not color_by_run else "-"
            target_ax.plot(
                iterations[: len(mean_errors)],
                mean_errors,
                label=f"Mean Error{label_suffix}" if run_idx == 0 or not color_by_run else None,
                color=color,
                marker="o" if len(iterations) < 50 else None,
                markersize=3,
                linestyle=linestyle,
                **kwargs,
            )

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
            ax.axvline(
                x=boundary,
                color="gray",
                linestyle="--",
                alpha=0.5,
                linewidth=1,
                label="Run boundary" if i == 1 else None,
            )

    # Add tolerance lines from latest options
    if show_tolerance:
        latest_options = workflow.latest.options
        if latest_options is not None:
            if metric in ("mean", "both") and latest_options.mean_tol is not None:
                tol_value = latest_options.mean_tol * 100 if use_pct else latest_options.mean_tol
                target_ax = ax2 if metric == "both" else ax
                target_ax.axhline(
                    y=tol_value,
                    color=mean_color_base,
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.7,
                    label=f"Mean Tol ({latest_options.mean_tol * 100:.1f}%)",
                )
            if metric in ("max", "both") and latest_options.max_tol is not None:
                tol_value = latest_options.max_tol * 100 if use_pct else latest_options.max_tol
                ax.axhline(
                    y=tol_value,
                    color=max_color_base,
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.7,
                    label=f"Max Tol ({latest_options.max_tol * 100:.1f}%)",
                )

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

    return ax
