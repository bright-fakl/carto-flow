"""
===========================================================================
Visualization utilities for cartogram results
===========================================================================

Plotting and animation utilities for morphing results and algorithm analysis.

Main Components
---------------
- plot_result: Plot morphed geometries
- plot_comparison: Side-by-side comparison of original vs morphed
- plot_convergence: Plot error metrics over iterations
- plot_density_field: Visualize density field as heatmap
- plot_velocity_field: Visualize velocity field as quiver plot

Example
-------
>>> from carto_flow.shape_morpher import morph_gdf, MorphOptions
>>> from carto_flow.shape_morpher.visualization import plot_comparison, plot_convergence
>>>
>>> result = morph_gdf(gdf, 'population', options=MorphOptions.preset_fast())
>>> plot_comparison(gdf, result)
>>> plot_convergence(result.history)
"""

from typing import TYPE_CHECKING, Any, Optional

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .grid import Grid
    from .history import CartogramInternalsSnapshot, History
    from .result import MorphResult

__all__ = [
    "plot_comparison",
    "plot_convergence",
    "plot_density_field",
    "plot_result",
    "plot_velocity_field",
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


def plot_result(
    result: "MorphResult",
    ax: Optional["Axes"] = None,
    column: Optional[str] = None,
    **kwargs: Any,
) -> "Axes":
    """Plot morphed geometries.

    Parameters
    ----------
    result : MorphResult
        Morphing result containing geometries to plot
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    column : str, optional
        Column to use for coloring geometries
    **kwargs
        Additional arguments passed to GeoDataFrame.plot()

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 8))

    gdf = result.geometries
    if hasattr(gdf, "plot"):
        if column and column in gdf.columns:
            gdf.plot(ax=ax, column=column, legend=True, **kwargs)
        else:
            gdf.plot(ax=ax, **kwargs)
    else:
        raise TypeError("result.geometries must be a GeoDataFrame with plot method")

    ax.set_aspect("equal")
    ax.set_title("Cartogram Result")
    return ax


def plot_comparison(
    original: Any,
    result: "MorphResult",
    column: Optional[str] = None,
    figsize: tuple[float, float] = (14, 6),
    **kwargs: Any,
) -> "Figure":
    """Side-by-side comparison of original vs morphed geometries.

    Parameters
    ----------
    original : GeoDataFrame
        Original GeoDataFrame before morphing
    result : MorphResult
        Morphing result containing transformed geometries
    column : str, optional
        Column to use for coloring geometries
    figsize : tuple, default=(14, 6)
        Figure size (width, height)
    **kwargs
        Additional arguments passed to GeoDataFrame.plot()

    Returns
    -------
    matplotlib.figure.Figure
        The figure with comparison plots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot original
    if hasattr(original, "plot"):
        if column and column in original.columns:
            original.plot(ax=ax1, column=column, legend=True, **kwargs)
        else:
            original.plot(ax=ax1, **kwargs)
    ax1.set_aspect("equal")
    ax1.set_title("Original")

    # Plot result
    gdf = result.geometries
    if hasattr(gdf, "plot"):
        if column and column in gdf.columns:
            gdf.plot(ax=ax2, column=column, legend=True, **kwargs)
        else:
            gdf.plot(ax=ax2, **kwargs)
    ax2.set_aspect("equal")
    ax2.set_title(f"Cartogram (status: {result.status})")

    plt.tight_layout()
    return fig


def plot_convergence(
    history: "History",
    ax: Optional["Axes"] = None,
    show_both: bool = True,
    **kwargs: Any,
) -> "Axes":
    """Plot convergence metrics (mean/max error) over iterations.

    Parameters
    ----------
    history : History
        History object containing iteration snapshots
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    show_both : bool, default=True
        If True, shows both mean and max error. If False, only mean error.
    **kwargs
        Additional arguments passed to plt.plot()

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the convergence plot
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 6))

    iterations = []
    mean_errors = []
    max_errors = []

    for snapshot in history.snapshots:
        iterations.append(snapshot.iteration)
        if hasattr(snapshot, "mean_error") and snapshot.mean_error is not None:
            mean_errors.append(snapshot.mean_error)
        if hasattr(snapshot, "max_error") and snapshot.max_error is not None:
            max_errors.append(snapshot.max_error)

    if mean_errors:
        ax.plot(iterations[: len(mean_errors)], mean_errors, label="Mean Error", **kwargs)
    if show_both and max_errors:
        ax.plot(iterations[: len(max_errors)], max_errors, label="Max Error", linestyle="--", **kwargs)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error")
    ax.set_title("Convergence History")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_density_field(
    snapshot: "CartogramInternalsSnapshot",
    grid: "Grid",
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
    snapshot : CartogramInternalsSnapshot
        Snapshot containing density field (rho attribute).
        Get from result.history_internals.snapshots (requires save_internals=True).
    grid : Grid
        Grid object for coordinate information.
        Available from result.grid after morphing.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    bounds : str, float, or tuple, optional
        Clip view to specified bounds (removes grid margin):
        - None: Show full grid extent (default)
        - "data": Clip to original data bounds (no margin)
        - float: Data bounds + margin as fraction (e.g., 0.1 = 10% margin)
        - tuple: (xmin, ymin, xmax, ymax) custom bounds
    normalize : str, optional
        Show density relative to mean_density (the target equilibrium):
        - None: Show absolute density values (default)
        - "difference": Show (rho - mean_density), centered at 0.
          Positive values indicate denser regions, negative indicate sparser.
        - "ratio": Show (rho / mean_density), centered at 1.0.
          Values >1 indicate denser regions, <1 indicate sparser.
        Both use a diverging colormap (RdBu_r).
        Requires snapshot to have mean_density attribute.
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
    >>> result = morph_gdf(gdf, 'pop', options=MorphOptions(save_internals=True))
    >>> snapshot = result.history_internals.snapshots[-1]
    >>> plot_density_field(snapshot, result.grid)

    >>> # Clip to original data bounds (no margin)
    >>> plot_density_field(snapshot, result.grid, bounds="data")

    >>> # Clip with 10% margin around data bounds
    >>> plot_density_field(snapshot, result.grid, bounds=0.1)

    >>> # Show density difference from equilibrium (centered at 0)
    >>> plot_density_field(snapshot, result.grid, normalize="difference")

    >>> # Clip outliers to [2nd, 98th] percentiles for difference mode
    >>> plot_density_field(snapshot, result.grid, normalize="difference", clip_percentile=2)

    >>> # Show density ratio to equilibrium (centered at 1)
    >>> plot_density_field(snapshot, result.grid, normalize="ratio")

    >>> # Compare two snapshots with consistent color scale
    >>> snap1 = result.history_internals.snapshots[0]
    >>> snap2 = result.history_internals.snapshots[-1]
    >>> fig, (ax1, ax2) = plt.subplots(1, 2)
    >>> plot_density_field(snap1, result.grid, ax=ax1, normalize="ratio", max_scale=4)
    >>> plot_density_field(snap2, result.grid, ax=ax2, normalize="ratio", max_scale=4)
    """
    from matplotlib.colors import TwoSlopeNorm

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 8))

    if not hasattr(snapshot, "rho") or snapshot.rho is None:
        raise ValueError("Snapshot does not contain density field (rho)")

    rho = snapshot.rho
    extent = [grid.xmin, grid.xmax, grid.ymin, grid.ymax]

    # Handle normalization
    if normalize is not None:
        if normalize not in ("difference", "ratio"):
            raise ValueError(f"normalize must be 'difference', 'ratio', or None, got {normalize!r}")
        if not hasattr(snapshot, "mean_density") or snapshot.mean_density is None:
            raise ValueError(
                "Snapshot does not contain mean_density. "
                "Re-run morphing with save_internals=True using the latest version."
            )
        mean_density = snapshot.mean_density
        if mean_density <= 0:
            raise ValueError(f"mean_density must be positive, got {mean_density}")

        if normalize == "difference":
            # Subtraction: centered at 0, symmetric range
            rho_display = rho - mean_density
            colorbar_label = "Density - Mean"
            title_suffix = " (difference)"

            # Determine data range
            if max_scale is not None:
                # User-specified fixed scale
                max_deviation = max_scale
            elif clip_percentile is not None:
                # Get values inside geometries only (exclude background)
                if hasattr(snapshot, "outside_mask") and snapshot.outside_mask is not None:
                    inside_values = rho_display[~snapshot.outside_mask]
                else:
                    # Fallback: use all values if mask not available
                    inside_values = rho_display.ravel()
                # Clip to [p, 100-p] percentiles to exclude outliers
                p_low = np.percentile(inside_values, clip_percentile)
                p_high = np.percentile(inside_values, 100 - clip_percentile)
                # Symmetric around 0
                max_deviation = max(abs(p_low), abs(p_high))
                title_suffix = f" (difference, clipped {clip_percentile}%)"
            else:
                data_min = rho_display.min()
                data_max = rho_display.max()
                max_deviation = max(abs(data_min), abs(data_max))

            sym_vmin = -max_deviation
            sym_vmax = max_deviation

            # Allow user override via kwargs
            vmin = kwargs.pop("vmin", sym_vmin)
            vmax = kwargs.pop("vmax", sym_vmax)

            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            cmap = kwargs.pop("cmap", "RdBu_r")
            colorbar_formatter = None
        else:  # ratio
            # Use log2 scale for perceptually uniform mapping
            # This handles extreme values (like DC population density) gracefully
            ratio = rho / mean_density
            # Clip to avoid log(0); minimum ratio of 1e-6 (one millionth of mean)
            ratio_clipped = np.clip(ratio, 1e-6, None)
            rho_display = np.log2(ratio_clipped)
            colorbar_label = "Density / Mean"
            title_suffix = " (ratio, log scale)"

            # Determine range in log space
            if max_scale is not None:
                # User-specified fixed scale: max_scale=4 means range [0.25, 4]
                max_log_deviation = np.log2(max_scale)
            else:
                # Symmetric range in log space (centered at 0 = ratio of 1)
                data_min = rho_display.min()
                data_max = rho_display.max()
                max_log_deviation = max(abs(data_min), abs(data_max))

            # Allow user override via kwargs (in log2 units)
            vmin = kwargs.pop("vmin", -max_log_deviation)
            vmax = kwargs.pop("vmax", max_log_deviation)

            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            cmap = kwargs.pop("cmap", "RdBu_r")

            # Custom formatter to show ratio values on colorbar
            def colorbar_formatter(x, pos):
                ratio_val = 2**x
                if ratio_val >= 1:
                    if ratio_val == int(ratio_val):
                        return f"{int(ratio_val)}"
                    return f"{ratio_val:.1f}"
                else:
                    # Show as fraction for values < 1
                    if ratio_val >= 0.1:
                        return f"{ratio_val:.2f}"
                    return f"{ratio_val:.3f}"

    else:
        rho_display = rho
        colorbar_label = "Density"
        title_suffix = ""
        norm = None
        cmap = kwargs.pop("cmap", "viridis")
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
    snapshot: "CartogramInternalsSnapshot",
    grid: "Grid",
    ax: Optional["Axes"] = None,
    skip: int = 4,
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
    snapshot : CartogramInternalsSnapshot
        Snapshot containing velocity fields (vx, vy attributes).
        Get from result.history_internals.snapshots (requires save_internals=True).
    grid : Grid
        Grid object for coordinate information.
        Available from result.grid after morphing.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    skip : int, default=4
        Plot every nth arrow to reduce clutter
    color_by : str, optional
        Color arrows by a property:
        - None: No coloring (default)
        - "magnitude": Color by velocity magnitude (sqrt(vx² + vy²))
        - "direction": Color by direction angle (-π to π)
    cmap : str, optional
        Colormap to use when color_by is specified.
        Defaults to "viridis" for magnitude, "twilight" for direction.
    colorbar : bool, default=True
        Whether to show colorbar when color_by is specified
    alpha_by_magnitude : bool, default=False
        If True, arrow transparency varies with magnitude (stronger = more opaque)
    alpha_range : tuple[float, float], default=(0.2, 1.0)
        Min and max alpha values when alpha_by_magnitude is True
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
    >>> result = morph_gdf(gdf, 'pop', options=MorphOptions(save_internals=True))
    >>> snapshot = result.history_internals.snapshots[-1]
    >>> plot_velocity_field(snapshot, result.grid)

    >>> # Color by magnitude
    >>> plot_velocity_field(snapshot, result.grid, color_by="magnitude", cmap="plasma")

    >>> # Color by direction (uses cyclic colormap by default)
    >>> plot_velocity_field(snapshot, result.grid, color_by="direction")

    >>> # Transparency by magnitude (fade out weak velocities)
    >>> plot_velocity_field(snapshot, result.grid, alpha_by_magnitude=True)

    >>> # Combine direction coloring with magnitude transparency
    >>> plot_velocity_field(snapshot, result.grid, color_by="direction", alpha_by_magnitude=True)

    >>> # Clip to original data bounds (no margin)
    >>> plot_velocity_field(snapshot, result.grid, bounds="data")

    >>> # Clip with 10% margin around data bounds
    >>> plot_velocity_field(snapshot, result.grid, bounds=0.1)
    """
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

    # Compute magnitude (needed for alpha and possibly coloring)
    magnitude = np.sqrt(vx_sub**2 + vy_sub**2)

    # Normalize magnitude for alpha mapping
    mag_min, mag_max = magnitude.min(), magnitude.max()
    if mag_max > mag_min:
        magnitude_norm = (magnitude - mag_min) / (mag_max - mag_min)
    else:
        magnitude_norm = np.ones_like(magnitude)

    # Compute color values if requested
    if color_by is not None or alpha_by_magnitude:
        # Determine colormap
        if color_by == "magnitude":
            C_norm = magnitude_norm
            colorbar_label = "Magnitude"
            effective_cmap = cmap if cmap is not None else "viridis"
        elif color_by == "direction":
            direction = np.arctan2(vy_sub, vx_sub)
            # Normalize direction from [-π, π] to [0, 1]
            C_norm = (direction + np.pi) / (2 * np.pi)
            colorbar_label = "Direction (radians)"
            # Use cyclic colormap for direction by default
            effective_cmap = cmap if cmap is not None else "twilight"
        elif color_by is None:
            # No color_by but alpha_by_magnitude is True
            C_norm = magnitude_norm
            colorbar_label = None
            effective_cmap = cmap if cmap is not None else "viridis"
        else:
            raise ValueError(f"color_by must be 'magnitude', 'direction', or None, got '{color_by}'")

        # Get colormap and compute RGBA colors
        cmap_obj = plt.get_cmap(effective_cmap)
        colors = cmap_obj(C_norm.ravel())

        # Apply alpha by magnitude if requested
        if alpha_by_magnitude:
            alpha_values = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * magnitude_norm
            colors[:, 3] = alpha_values.ravel()

        ax.quiver(X_sub, Y_sub, vx_sub, vy_sub, color=colors, **kwargs)

        if colorbar and color_by is not None:
            # Create a ScalarMappable for colorbar
            import matplotlib.cm as cm
            from matplotlib.colors import Normalize

            if color_by == "magnitude":
                norm = Normalize(vmin=mag_min, vmax=mag_max)
            else:  # direction
                norm = Normalize(vmin=-np.pi, vmax=np.pi)

            sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label=colorbar_label)
    else:
        ax.quiver(X_sub, Y_sub, vx_sub, vy_sub, **kwargs)

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
