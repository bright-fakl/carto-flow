"""
===========================================================================
Animation utilities for cartogram results
===========================================================================

Animation functions for visualizing morphing results, including geometry
transitions, density field evolution, and velocity field dynamics.

Main Components
---------------
- animate_morph_history: Animate through algorithm snapshots
- animate_geometry_keyframes: Animate between arbitrary geometry states
- animate_density_field: Animate density field evolution
- animate_velocity_field: Animate velocity field evolution
- save_animation: Save animations to various formats

Example
-------
>>> from carto_flow.shape_morpher import morph_gdf, MorphOptions
>>> from carto_flow.shape_morpher.animation import animate_morph_history, save_animation
>>>
>>> result = morph_gdf(gdf, 'population', options=MorphOptions(save_internals=True))
>>> anim = animate_morph_history(result, duration=5.0, fps=15)
>>> save_animation(anim, "morph.gif", fps=15)
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

if TYPE_CHECKING:
    from geopandas import GeoDataFrame

    from .result import MorphResult

__all__ = [
    "animate_density_field",
    "animate_fields",
    "animate_geometry_keyframes",
    "animate_morph_history",
    "animate_velocity_field",
    "linear_over",
    "save_animation",
    "weights_to_position_mapper",
]

# Type alias for position mapper functions
# Signature: (progress: float, variables: dict[str, np.ndarray]) -> float
PositionMapper = Callable[[float, dict[str, np.ndarray]], float]


# =============================================================================
# Built-in position mappers and factories
# =============================================================================


def _linear_over_values(
    progress: float,
    values: np.ndarray,
    decreasing: bool = False,
) -> float:
    """Core implementation: map progress to position by linearizing over values.

    Parameters
    ----------
    progress : float
        Animation progress in [0, 1]
    values : np.ndarray
        Array of values to linearize over
    decreasing : bool, default=False
        If True, values are expected to decrease (e.g., error values)

    Returns
    -------
    float
        Position in [0, n-1]
    """
    if len(values) <= 1:
        return 0.0

    if decreasing:
        # For decreasing values (like error), interpolate from max to min
        max_val = values[0]
        min_val = values[-1]
        target = max_val - progress * (max_val - min_val)

        # Find position (searching for decreasing values)
        idx = 0
        for i in range(len(values) - 1):
            if values[i] >= target >= values[i + 1]:
                idx = i
                break
        else:
            idx = len(values) - 2

        idx = np.clip(idx, 0, len(values) - 2)

        # Interpolate within segment
        val_before = values[idx]
        val_after = values[idx + 1]
        t = (val_before - target) / (val_before - val_after) if val_before != val_after else 0.0
    else:
        # For increasing values (like iteration), interpolate from min to max
        min_val = values[0]
        max_val = values[-1]
        target = min_val + progress * (max_val - min_val)

        # Find position
        idx = np.searchsorted(values, target, side="right") - 1
        idx = np.clip(idx, 0, len(values) - 2)

        # Interpolate within segment
        val_before = values[idx]
        val_after = values[idx + 1]
        t = (target - val_before) / (val_after - val_before) if val_after > val_before else 0.0

    return idx + np.clip(t, 0.0, 1.0)


def linear_over(
    key_or_values: Union[str, np.ndarray],
    decreasing: Optional[bool] = None,
) -> PositionMapper:
    """Create a position mapper that linearizes animation progress over values.

    This factory creates a position mapper where animation progress [0, 1] maps
    linearly to the specified values. Use this to control which variable
    determines animation pacing.

    Parameters
    ----------
    key_or_values : str or array
        - str: Variable name to extract from the variables dict.
          Common keys for animate_morph_history: "iteration", "mean_error", "max_error"
          Common keys for animate_geometry_keyframes: "index", "time"
        - array: Values to linearize over directly
    decreasing : bool, optional
        Whether values decrease over time (e.g., error values).
        If None (default), auto-detected: True for keys containing "error",
        False otherwise.

    Returns
    -------
    PositionMapper
        Position mapper function with signature (progress, variables) -> position

    Examples
    --------
    >>> # Linearize over iteration (default for morph_history)
    >>> anim = animate_morph_history(result, position_mapper=linear_over("iteration"))

    >>> # Linearize over mean error
    >>> anim = animate_morph_history(result, position_mapper=linear_over("mean_error"))

    >>> # Linearize over max error
    >>> anim = animate_morph_history(result, position_mapper=linear_over("max_error"))

    >>> # Custom values
    >>> times = np.array([0, 0.2, 0.5, 1.0])
    >>> anim = animate_geometry_keyframes(keyframes, position_mapper=linear_over(times))
    """
    if isinstance(key_or_values, str):
        key = key_or_values
        # Auto-detect decreasing based on key name
        is_decreasing = decreasing if decreasing is not None else ("error" in key.lower())

        def mapper(progress: float, variables: dict[str, np.ndarray]) -> float:
            if key not in variables:
                available = list(variables.keys())
                raise KeyError(f"Variable '{key}' not available. Available variables: {available}")
            return _linear_over_values(progress, variables[key], decreasing=is_decreasing)

        return mapper
    else:
        values = np.asarray(key_or_values)
        # Auto-detect decreasing: True if values generally decrease
        is_decreasing = decreasing if decreasing is not None else (values[-1] < values[0])

        def mapper(progress: float, variables: dict[str, np.ndarray]) -> float:
            return _linear_over_values(progress, values, decreasing=is_decreasing)

        return mapper


def weights_to_position_mapper(
    weight_fn: Callable[[dict[str, np.ndarray]], np.ndarray],
) -> PositionMapper:
    """Convert a weight-based function to a position mapper.

    This allows users to define mappers using the simpler weight-based interface
    (where higher weight = more time at that snapshot) while using the unified
    position mapper internally.

    Parameters
    ----------
    weight_fn : callable
        Function with signature (variables: dict) -> weights array.
        Higher weights mean more animation time allocated to that item.

    Returns
    -------
    PositionMapper
        Position mapper function with signature (progress, variables) -> position

    Examples
    --------
    >>> def slow_at_changes(variables):
    ...     errors = variables["mean_error"]
    ...     d_error = np.abs(np.gradient(errors))
    ...     return 1 + 5 * (d_error / d_error.max())
    >>>
    >>> anim = animate_morph_history(
    ...     result,
    ...     position_mapper=weights_to_position_mapper(slow_at_changes)
    ... )
    """
    # Cache for computed cumulative weights
    _cache: dict = {}

    def position_mapper(
        progress: float,
        variables: dict[str, np.ndarray],
    ) -> float:
        # Compute weights and cumulative distribution (cached)
        cache_key = id(variables)
        if cache_key not in _cache:
            weights = weight_fn(variables)
            weights = np.maximum(weights, 0)
            cumsum = np.cumsum(weights)
            cumsum = cumsum / cumsum[-1]  # Normalize to [0, 1]
            _cache[cache_key] = cumsum

        cumsum = _cache[cache_key]

        # Find position using inverse CDF lookup
        if progress <= 0:
            return 0.0
        if progress >= 1:
            return len(cumsum) - 1.0

        # Find which segment we're in
        idx = np.searchsorted(cumsum, progress, side="right")
        idx = np.clip(idx, 1, len(cumsum) - 1)

        # Interpolate within segment
        cumsum_before = cumsum[idx - 1] if idx > 0 else 0.0
        cumsum_after = cumsum[idx]
        t = (progress - cumsum_before) / (cumsum_after - cumsum_before) if cumsum_after > cumsum_before else 0.0

        return (idx - 1) + np.clip(t, 0.0, 1.0)

    return position_mapper


# Default position mapper (iteration-linear)
_default_position_mapper = linear_over("iteration")


# =============================================================================
# Internal helper functions
# =============================================================================


def _extract_geometries(source: Any) -> "GeoDataFrame":
    """Extract GeoDataFrame from various source types.

    Parameters
    ----------
    source : GeoDataFrame or MorphResult
        Geometry source to extract from

    Returns
    -------
    GeoDataFrame
        The extracted geometries

    Raises
    ------
    TypeError
        If source type is not supported
    """
    # Check for MorphResult first (has .geometries attribute)
    if hasattr(source, "geometries"):
        return source.geometries
    # Check for GeoDataFrame (has .geometry attribute)
    if hasattr(source, "geometry"):
        return source
    raise TypeError(f"Expected GeoDataFrame or MorphResult, got {type(source).__name__}")


def _interpolate_geometry(geom_a: Any, geom_b: Any, t: float) -> Any:
    """Linearly interpolate between two geometries.

    Parameters
    ----------
    geom_a : shapely geometry
        Starting geometry
    geom_b : shapely geometry
        Ending geometry
    t : float
        Interpolation parameter in [0, 1]

    Returns
    -------
    shapely geometry
        Interpolated geometry
    """
    from shapely.geometry import (
        LineString,
        MultiPolygon,
        Point,
        Polygon,
    )

    if t <= 0:
        return geom_a
    if t >= 1:
        return geom_b

    def interpolate_coords(coords_a, coords_b, t):
        """Interpolate coordinate sequences."""
        arr_a = np.array(coords_a)
        arr_b = np.array(coords_b)

        # Handle mismatched lengths by resampling
        if len(arr_a) != len(arr_b):
            # Resample to the larger size
            n = max(len(arr_a), len(arr_b))
            if len(arr_a) < n:
                indices = np.linspace(0, len(arr_a) - 1, n)
                arr_a = np.array([np.interp(indices, range(len(arr_a)), arr_a[:, i]) for i in range(arr_a.shape[1])]).T
            if len(arr_b) < n:
                indices = np.linspace(0, len(arr_b) - 1, n)
                arr_b = np.array([np.interp(indices, range(len(arr_b)), arr_b[:, i]) for i in range(arr_b.shape[1])]).T

        return arr_a * (1 - t) + arr_b * t

    def interpolate_polygon(poly_a, poly_b, t):
        """Interpolate between two polygons."""
        # Interpolate exterior
        ext_coords = interpolate_coords(list(poly_a.exterior.coords), list(poly_b.exterior.coords), t)
        exterior = ext_coords.tolist()

        # Interpolate interiors (holes)
        interiors = []
        n_holes = max(len(poly_a.interiors), len(poly_b.interiors))
        for i in range(n_holes):
            if i < len(poly_a.interiors) and i < len(poly_b.interiors):
                hole_coords = interpolate_coords(
                    list(poly_a.interiors[i].coords),
                    list(poly_b.interiors[i].coords),
                    t,
                )
                interiors.append(hole_coords.tolist())
            elif i < len(poly_a.interiors):
                # Hole only in a - fade out by shrinking to centroid
                centroid = poly_a.interiors[i].centroid
                hole_a = np.array(poly_a.interiors[i].coords)
                hole_b = np.full_like(hole_a, [centroid.x, centroid.y])
                hole_coords = hole_a * (1 - t) + hole_b * t
                interiors.append(hole_coords.tolist())
            else:
                # Hole only in b - fade in by growing from centroid
                centroid = poly_b.interiors[i].centroid
                hole_a = np.full((len(poly_b.interiors[i].coords), 2), [centroid.x, centroid.y])
                hole_b = np.array(poly_b.interiors[i].coords)
                hole_coords = hole_a * (1 - t) + hole_b * t
                interiors.append(hole_coords.tolist())

        return Polygon(exterior, interiors)

    # Handle different geometry types
    geom_type_a = geom_a.geom_type
    geom_type_b = geom_b.geom_type

    if geom_type_a != geom_type_b:
        # Fall back to simple linear interpolation if types don't match
        # This is a simplification - could be improved
        return geom_a if t < 0.5 else geom_b

    if geom_type_a == "Point":
        coords = interpolate_coords([geom_a.coords[0]], [geom_b.coords[0]], t)
        return Point(coords[0])

    elif geom_type_a == "LineString":
        coords = interpolate_coords(list(geom_a.coords), list(geom_b.coords), t)
        return LineString(coords)

    elif geom_type_a == "Polygon":
        return interpolate_polygon(geom_a, geom_b, t)

    elif geom_type_a == "MultiPolygon":
        n_polys = max(len(geom_a.geoms), len(geom_b.geoms))
        polys = []
        for i in range(n_polys):
            if i < len(geom_a.geoms) and i < len(geom_b.geoms):
                polys.append(interpolate_polygon(geom_a.geoms[i], geom_b.geoms[i], t))
            elif i < len(geom_a.geoms):
                # Polygon only in a - keep it (could fade out)
                polys.append(geom_a.geoms[i])
            else:
                # Polygon only in b - add it (could fade in)
                polys.append(geom_b.geoms[i])
        return MultiPolygon(polys)

    else:
        # Fallback for unsupported types
        return geom_a if t < 0.5 else geom_b


def _interpolate_geodataframe(
    gdf_a: "GeoDataFrame",
    gdf_b: "GeoDataFrame",
    t: float,
    color_values_a: Optional[np.ndarray] = None,
    color_values_b: Optional[np.ndarray] = None,
    color_column: str = "_color",
) -> "GeoDataFrame":
    """Interpolate between two GeoDataFrames.

    Parameters
    ----------
    gdf_a : GeoDataFrame
        Starting geometries
    gdf_b : GeoDataFrame
        Ending geometries
    t : float
        Interpolation parameter in [0, 1]
    color_values_a : np.ndarray, optional
        Color values for gdf_a (one per geometry)
    color_values_b : np.ndarray, optional
        Color values for gdf_b (one per geometry)
    color_column : str, default="_color"
        Column name to store interpolated color values

    Returns
    -------
    GeoDataFrame
        Interpolated geometries (using gdf_a's non-geometry columns).
        If color values provided, includes interpolated colors in color_column.
    """

    if t <= 0:
        result = gdf_a.copy()
        if color_values_a is not None:
            result[color_column] = color_values_a
        return result
    if t >= 1:
        result = gdf_b.copy()
        if color_values_b is not None:
            result[color_column] = color_values_b
        return result

    if len(gdf_a) != len(gdf_b):
        raise ValueError(f"GeoDataFrames must have same length for interpolation ({len(gdf_a)} vs {len(gdf_b)})")

    # Interpolate each geometry
    interpolated_geoms = [
        _interpolate_geometry(geom_a, geom_b, t) for geom_a, geom_b in zip(gdf_a.geometry, gdf_b.geometry)
    ]

    # Create result with interpolated geometries and original attributes
    result = gdf_a.copy()
    result.geometry = interpolated_geoms

    # Interpolate color values if provided
    if color_values_a is not None and color_values_b is not None:
        interp_colors = color_values_a * (1 - t) + color_values_b * t
        result[color_column] = interp_colors

    return result


# =============================================================================
# Save animation function
# =============================================================================


def save_animation(
    anim: FuncAnimation,
    path: Union[str, Path],
    fps: int = 15,
    dpi: int = 100,
    **kwargs: Any,
) -> None:
    """Save an animation to file.

    Parameters
    ----------
    anim : FuncAnimation
        Animation object to save
    path : str or Path
        Output path. Format determined by extension:
        - .gif: Uses pillow writer
        - .mp4: Uses ffmpeg writer
        - Directory path: Saves individual PNG frames
    fps : int, default=15
        Frames per second
    dpi : int, default=100
        Resolution in dots per inch
    **kwargs
        Additional arguments passed to the writer

    Raises
    ------
    ValueError
        If the output format is not supported or required writer is unavailable
    """
    path = Path(path)

    if path.suffix.lower() == ".gif":
        try:
            anim.save(str(path), writer="pillow", fps=fps, dpi=dpi, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to save GIF. Ensure pillow is installed: {e}") from e

    elif path.suffix.lower() == ".mp4":
        try:
            anim.save(str(path), writer="ffmpeg", fps=fps, dpi=dpi, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to save MP4. Ensure ffmpeg is installed: {e}") from e

    elif path.suffix == "" or path.is_dir():
        # Save as individual PNG frames
        path.mkdir(parents=True, exist_ok=True)

        # Get the figure from the animation
        fig = anim._fig

        # Save each frame
        for i, _ in enumerate(range(len(anim._save_seq) if hasattr(anim, "_save_seq") else anim.save_count)):
            anim._draw_frame(i)
            fig.savefig(path / f"frame_{i:04d}.png", dpi=dpi)

    else:
        raise ValueError(
            f"Unsupported output format: {path.suffix}. Use .gif, .mp4, or a directory path for PNG frames."
        )


# =============================================================================
# Geometry animation functions
# =============================================================================


def animate_morph_history(
    result: "MorphResult",
    duration: float = 5.0,
    fps: int = 15,
    interpolation: str = "nearest",
    position_mapper: Optional[PositionMapper] = None,
    column: Optional[str] = None,
    color_values: Optional[list[np.ndarray]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    colorbar: bool = False,
    colorbar_label: Optional[str] = None,
    figsize: tuple[float, float] = (10, 8),
    **kwargs: Any,
) -> FuncAnimation:
    """Animate through algorithm snapshots from a MorphResult's history.

    Parameters
    ----------
    result : MorphResult
        Morphing result with history containing geometry snapshots
    duration : float, default=5.0
        Animation duration in seconds
    fps : int, default=15
        Frames per second
    interpolation : str, default="nearest"
        Interpolation method between snapshots:
        - "nearest": Show the nearest snapshot (no interpolation)
        - "linear": Linearly interpolate geometry vertices between snapshots
    position_mapper : callable, optional
        Function to control animation timing. Maps progress [0,1] to position
        in snapshot space [0, n_snapshots-1]. Use linear_over() factory:
        - linear_over("iteration"): Linear time-to-iteration (default)
        - linear_over("mean_error"): Linear time-to-mean-error
        - linear_over("max_error"): Linear time-to-max-error
        - linear_over(custom_values): Linear over any custom array
        Use weights_to_position_mapper() for weight-based functions.
    column : str, optional
        Column to use for coloring geometries. Values are read from each
        snapshot's GeoDataFrame and color limits are computed globally.
        Special value: "area_errors" extracts per-geometry errors from
        each snapshot's area_errors attribute.
    color_values : list of arrays, optional
        Per-snapshot color values. Each array has one value per geometry.
        When provided, takes precedence over column. Color values are
        interpolated when using linear geometry interpolation.
    vmin : float, optional
        Minimum value for color scale. If None, computed from all values.
    vmax : float, optional
        Maximum value for color scale. If None, computed from all values.
    cmap : str, default="viridis"
        Colormap name for coloring geometries.
    colorbar : bool, default=False
        Whether to show a colorbar.
    colorbar_label : str, optional
        Label for the colorbar. Defaults to column name if using column.
    figsize : tuple, default=(10, 8)
        Figure size (width, height)
    **kwargs
        Additional arguments passed to GeoDataFrame.plot()

    Returns
    -------
    FuncAnimation
        Matplotlib animation object

    Raises
    ------
    ValueError
        If result.history doesn't contain geometry snapshots

    Notes
    -----
    Available variables for position_mapper (extracted from snapshots):
    - "iteration": Iteration numbers
    - "mean_error": Mean area errors (if available)
    - "max_error": Maximum area errors (if available)
    - "index": Snapshot indices [0, 1, 2, ...]

    Color limits are computed globally across all snapshots to ensure
    consistent coloring throughout the animation.

    Special column handling:
    - ``column="area_errors"``: Extracts per-geometry area errors from each
      snapshot's ``area_errors`` attribute (not from the GeoDataFrame columns).
      This allows coloring geometries by their individual area errors.

    Examples
    --------
    >>> result = morph_gdf(gdf, 'pop', options=MorphOptions(save_history=True))
    >>> anim = animate_morph_history(result, duration=5.0, fps=15)
    >>> save_animation(anim, "morph.gif", fps=15)

    >>> # Color by per-geometry area error (from snapshot.area_errors)
    >>> anim = animate_morph_history(
    ...     result, column="area_errors", cmap="RdYlGn_r",
    ...     colorbar=True, colorbar_label="Area Error"
    ... )

    >>> # Custom per-snapshot values
    >>> densities = [compute_density(snap.geometry) for snap in result.history]
    >>> anim = animate_morph_history(result, color_values=densities, cmap="plasma")

    >>> # Smooth animation with linear interpolation (colors also interpolated)
    >>> anim = animate_morph_history(
    ...     result, interpolation="linear", column="area_errors"
    ... )
    """
    if interpolation not in ("nearest", "linear"):
        raise ValueError(f"interpolation must be 'nearest' or 'linear', got {interpolation!r}")

    # Extract geometry snapshots from history
    if not hasattr(result, "history") or result.history is None:
        raise ValueError("result.history is required for animate_morph_history")

    snapshots = result.history.snapshots
    if not snapshots:
        raise ValueError("result.history contains no snapshots")

    # Extract geometries and build variables dict from snapshots
    geometries = []
    snapshot_area_errors: list[Optional[np.ndarray]] = []
    variables: dict[str, list] = {
        "iteration": [],
        "mean_error": [],
        "max_error": [],
    }

    for snap in snapshots:
        if hasattr(snap, "geometry") and snap.geometry is not None:
            geometries.append(snap.geometry)
            variables["iteration"].append(snap.iteration)
            variables["mean_error"].append(snap.mean_error if hasattr(snap, "mean_error") else None)
            variables["max_error"].append(snap.max_error if hasattr(snap, "max_error") else None)
            # Extract area_errors from snapshot (stored as attribute, not in GeoDataFrame)
            snapshot_area_errors.append(snap.area_errors if hasattr(snap, "area_errors") else None)

    if not geometries:
        raise ValueError(
            "No geometry snapshots found in history. "
            "Ensure morphing was run with save_history=True and geometry saving enabled."
        )

    n_snapshots = len(geometries)
    n_frames = int(duration * fps)

    # Check if column refers to a snapshot attribute (e.g., "area_errors")
    # and use snapshot area_errors as color_values
    if column == "area_errors" and all(e is not None for e in snapshot_area_errors):
        color_values = snapshot_area_errors

    # Validate color_values if provided
    if color_values is not None:
        if len(color_values) != n_snapshots:
            raise ValueError(
                f"color_values length ({len(color_values)}) must match number of snapshots ({n_snapshots})"
            )
        # Convert to numpy arrays
        color_values = [np.asarray(cv) for cv in color_values]

    # Determine coloring mode and gather all color values for global limits
    use_coloring = column is not None or color_values is not None
    all_color_values: list[np.ndarray] = []

    if color_values is not None:
        all_color_values = color_values
    elif column is not None:
        for gdf in geometries:
            if column in gdf.columns:
                all_color_values.append(gdf[column].values)

    # Compute global color limits
    computed_vmin = None
    computed_vmax = None
    if all_color_values:
        computed_vmin = min(np.nanmin(cv) for cv in all_color_values)
        computed_vmax = max(np.nanmax(cv) for cv in all_color_values)

    color_vmin = vmin if vmin is not None else computed_vmin
    color_vmax = vmax if vmax is not None else computed_vmax

    # Determine colorbar label
    if colorbar_label is None and column is not None:
        colorbar_label = column

    # Convert lists to arrays, removing variables that are all None
    variables_arrays: dict[str, np.ndarray] = {}
    variables_arrays["index"] = np.arange(n_snapshots)
    for key, values in variables.items():
        if all(v is not None for v in values):
            variables_arrays[key] = np.array(values)

    # Use default position mapper if none provided
    mapper = position_mapper if position_mapper is not None else _default_position_mapper

    # Convenience references
    iterations = variables_arrays.get("iteration", np.arange(n_snapshots))
    errors = variables_arrays.get("mean_error")

    # Set up figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Get consistent bounds from all geometries
    all_bounds = [gdf.total_bounds for gdf in geometries]
    xmin = min(b[0] for b in all_bounds)
    ymin = min(b[1] for b in all_bounds)
    xmax = max(b[2] for b in all_bounds)
    ymax = max(b[3] for b in all_bounds)

    # Track colorbar (only add once)
    cbar_added = [False]

    def init():
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        return []

    def get_color_values_for_snapshot(idx: int) -> Optional[np.ndarray]:
        """Get color values for a specific snapshot."""
        if color_values is not None:
            return color_values[idx]
        elif column is not None and column in geometries[idx].columns:
            return geometries[idx][column].values
        return None

    def get_frame_data(frame: int) -> tuple["GeoDataFrame", int, float, Optional[float], Optional[np.ndarray]]:
        """Get the geometry and color values for a given frame.

        Returns (gdf, snapshot_idx, iteration, error, colors) where gdf and colors
        may be interpolated.
        """
        # Map frame to progress [0, 1]
        progress = frame / max(1, n_frames - 1)

        # Map progress to position in snapshot space [0, n_snapshots-1]
        position = mapper(progress, variables_arrays)
        position = np.clip(position, 0, n_snapshots - 1)

        # Extract integer index and fractional part
        idx_before = int(position)
        idx_after = min(idx_before + 1, n_snapshots - 1)
        t = position - idx_before

        if interpolation == "nearest":
            # Snap to nearest snapshot
            idx = idx_before if t < 0.5 else idx_after
            error = errors[idx] if errors is not None else None
            colors = get_color_values_for_snapshot(idx)
            return geometries[idx], idx, iterations[idx], error, colors

        else:  # linear interpolation
            if idx_before == idx_after or t <= 0:
                gdf = geometries[idx_before]
                iteration = iterations[idx_before]
                error = errors[idx_before] if errors is not None else None
                colors = get_color_values_for_snapshot(idx_before)
            elif t >= 1:
                gdf = geometries[idx_after]
                iteration = iterations[idx_after]
                error = errors[idx_after] if errors is not None else None
                colors = get_color_values_for_snapshot(idx_after)
            else:
                # Get color values for interpolation
                colors_before = get_color_values_for_snapshot(idx_before)
                colors_after = get_color_values_for_snapshot(idx_after)

                gdf = _interpolate_geodataframe(
                    geometries[idx_before],
                    geometries[idx_after],
                    t,
                    color_values_a=colors_before,
                    color_values_b=colors_after,
                    color_column="_color",
                )

                # Interpolate iteration and error for display
                iteration = iterations[idx_before] + t * (iterations[idx_after] - iterations[idx_before])
                if errors is not None:
                    error = errors[idx_before] + t * (errors[idx_after] - errors[idx_before])
                else:
                    error = None

                # Get interpolated colors from the GeoDataFrame
                colors = gdf["_color"].values if "_color" in gdf.columns else None

            return gdf, idx_before, iteration, error, colors

    def update(frame):
        ax.clear()
        gdf, snapshot_idx, iteration, error, colors = get_frame_data(frame)

        # Determine how to plot
        if use_coloring and colors is not None and color_vmin is not None:
            # Add color values to a temporary column for plotting
            plot_gdf = gdf.copy()
            plot_gdf["_plot_color"] = colors
            mappable = plot_gdf.plot(
                ax=ax, column="_plot_color", legend=False, vmin=color_vmin, vmax=color_vmax, cmap=cmap, **kwargs
            )
        elif use_coloring and column is not None and column in gdf.columns and color_vmin is not None:
            mappable = gdf.plot(
                ax=ax, column=column, legend=False, vmin=color_vmin, vmax=color_vmax, cmap=cmap, **kwargs
            )
        else:
            gdf.plot(ax=ax, **kwargs)
            mappable = None

        # Add colorbar on first frame if requested
        if colorbar and not cbar_added[0] and mappable is not None and color_vmin is not None:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=color_vmin, vmax=color_vmax))
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label=colorbar_label)
            cbar_added[0] = True

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")

        error_str = ""
        if error is not None:
            error_str = f", error={error:.4f}"
        if isinstance(iteration, float) and not iteration.is_integer():
            ax.set_title(f"Iteration {iteration:.1f}{error_str}")
        else:
            ax.set_title(f"Iteration {int(iteration)}{error_str}")

        return []

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        blit=False,
        interval=1000 / fps,
    )

    return anim


# Default position mapper for keyframes (index-linear)
_default_keyframe_position_mapper = linear_over("index")


def animate_geometry_keyframes(
    keyframes: list[Any],
    duration: float = 5.0,
    fps: int = 15,
    interpolation: str = "linear",
    position_mapper: Optional[PositionMapper] = None,
    keyframe_times: Optional[np.ndarray] = None,
    hold_frames: int = 0,
    column: Optional[str] = None,
    color_values: Optional[list[np.ndarray]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    colorbar: bool = False,
    colorbar_label: Optional[str] = None,
    figsize: tuple[float, float] = (10, 8),
    **kwargs: Any,
) -> FuncAnimation:
    """Animate between arbitrary geometry states (keyframes).

    Parameters
    ----------
    keyframes : list
        List of geometry sources. Each can be:
        - GeoDataFrame
        - MorphResult (uses .geometries)
    duration : float, default=5.0
        Animation duration in seconds
    fps : int, default=15
        Frames per second
    interpolation : str, default="linear"
        Interpolation method between keyframes:
        - "nearest": Jump between keyframes (no interpolation)
        - "linear": Linearly interpolate geometry vertices between keyframes
    position_mapper : callable, optional
        Function to control animation timing. Maps progress [0,1] to position
        in keyframe space [0, n_keyframes-1]. Use linear_over() factory:
        - linear_over("index"): Linear time-to-keyframe index (default)
        - linear_over("time"): Linear time-to-time (requires keyframe_times)
        - linear_over(custom_values): Linear over any custom array
        Use weights_to_position_mapper() for weight-based functions.
    keyframe_times : array-like, optional
        Time values for each keyframe. Must have same length as keyframes.
        Enables linear_over("time") position mapper. Values should be
        monotonically increasing.
    hold_frames : int, default=0
        Number of frames to hold at each keyframe before transitioning.
        Note: When using a position_mapper, hold_frames is ignored.
    column : str, optional
        Column to use for coloring geometries. Values are read from each
        keyframe's GeoDataFrame and color limits are computed globally.
    color_values : list of arrays, optional
        Per-keyframe color values. Each array has one value per geometry.
        When provided, takes precedence over column. Color values are
        interpolated when using linear geometry interpolation.
    vmin : float, optional
        Minimum value for color scale. If None, computed from all values.
    vmax : float, optional
        Maximum value for color scale. If None, computed from all values.
    cmap : str, default="viridis"
        Colormap name for coloring geometries.
    colorbar : bool, default=False
        Whether to show a colorbar.
    colorbar_label : str, optional
        Label for the colorbar. Defaults to column name if using column.
    figsize : tuple, default=(10, 8)
        Figure size (width, height)
    **kwargs
        Additional arguments passed to GeoDataFrame.plot()

    Returns
    -------
    FuncAnimation
        Matplotlib animation object

    Notes
    -----
    Available variables for position_mapper:
    - "index": Keyframe indices [0, 1, 2, ...]
    - "time": Keyframe times (if keyframe_times is provided)

    Color limits are computed globally across all keyframes to ensure
    consistent coloring throughout the animation.

    Examples
    --------
    >>> # Animate with smooth interpolation (default: linear over index)
    >>> anim = animate_geometry_keyframes(
    ...     keyframes=[gdf, result1, result2],
    ...     duration=6.0,
    ...     interpolation="linear",
    ... )
    >>> save_animation(anim, "keyframes.gif", fps=15)

    >>> # Animate with changing colors (e.g., population over years)
    >>> anim = animate_geometry_keyframes(
    ...     keyframes=[gdf_2020, gdf_2021, gdf_2022],
    ...     color_values=[pop_2020, pop_2021, pop_2022],
    ...     interpolation="linear",
    ...     cmap="plasma",
    ...     colorbar=True,
    ...     colorbar_label="Population"
    ... )

    >>> # Use column from GeoDataFrames with consistent limits
    >>> anim = animate_geometry_keyframes(
    ...     keyframes=[gdf1, gdf2, gdf3],
    ...     column="value",
    ...     cmap="RdYlGn",
    ... )

    >>> # Linear over custom times
    >>> from carto_flow.shape_morpher.animation import linear_over
    >>> anim = animate_geometry_keyframes(
    ...     keyframes=[gdf, result1, result2],
    ...     keyframe_times=[0.0, 0.3, 1.0],
    ...     position_mapper=linear_over("time"),
    ... )

    >>> # Hold at each keyframe before transitioning (legacy mode)
    >>> anim = animate_geometry_keyframes(
    ...     keyframes=[gdf, result1, result2],
    ...     hold_frames=5,
    ... )
    """
    if interpolation not in ("nearest", "linear"):
        raise ValueError(f"interpolation must be 'nearest' or 'linear', got {interpolation!r}")
    if len(keyframes) < 2:
        raise ValueError("At least 2 keyframes are required for animation")

    # Extract GeoDataFrames from all keyframes
    gdfs = [_extract_geometries(kf) for kf in keyframes]

    # Validate all have same length
    lengths = [len(gdf) for gdf in gdfs]
    if len(set(lengths)) > 1:
        raise ValueError(f"All keyframes must have the same number of geometries. Got lengths: {lengths}")

    n_keyframes = len(gdfs)
    n_frames = int(duration * fps)

    # Validate keyframe_times if provided
    if keyframe_times is not None:
        keyframe_times = np.asarray(keyframe_times)
        if len(keyframe_times) != n_keyframes:
            raise ValueError(
                f"keyframe_times length ({len(keyframe_times)}) must match number of keyframes ({n_keyframes})"
            )

    # Validate color_values if provided
    if color_values is not None:
        if len(color_values) != n_keyframes:
            raise ValueError(
                f"color_values length ({len(color_values)}) must match number of keyframes ({n_keyframes})"
            )
        # Convert to numpy arrays
        color_values = [np.asarray(cv) for cv in color_values]

    # Determine coloring mode and gather all color values for global limits
    use_coloring = column is not None or color_values is not None
    all_color_values: list[np.ndarray] = []

    if color_values is not None:
        all_color_values = color_values
    elif column is not None:
        for gdf in gdfs:
            if column in gdf.columns:
                all_color_values.append(gdf[column].values)

    # Compute global color limits
    computed_vmin = None
    computed_vmax = None
    if all_color_values:
        computed_vmin = min(np.nanmin(cv) for cv in all_color_values)
        computed_vmax = max(np.nanmax(cv) for cv in all_color_values)

    color_vmin = vmin if vmin is not None else computed_vmin
    color_vmax = vmax if vmax is not None else computed_vmax

    # Determine colorbar label
    if colorbar_label is None and column is not None:
        colorbar_label = column

    # Build variables dict for position mapper
    variables_arrays: dict[str, np.ndarray] = {
        "index": np.arange(n_keyframes),
    }
    if keyframe_times is not None:
        variables_arrays["time"] = keyframe_times

    # Determine whether to use position_mapper mode or legacy hold_frames mode
    use_position_mapper = position_mapper is not None or hold_frames == 0

    if use_position_mapper:
        # Use position mapper approach
        mapper = position_mapper if position_mapper is not None else _default_keyframe_position_mapper
        actual_n_frames = n_frames
    else:
        # Legacy hold_frames mode (for backward compatibility)
        n_transitions = n_keyframes - 1
        total_hold = hold_frames * n_keyframes
        transition_frames = max(1, (n_frames - total_hold) // n_transitions)
        actual_n_frames = transition_frames * n_transitions + total_hold
        mapper = None  # Not used in legacy mode

    # Set up figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Get consistent bounds from all geometries
    all_bounds = [gdf.total_bounds for gdf in gdfs]
    xmin = min(b[0] for b in all_bounds)
    ymin = min(b[1] for b in all_bounds)
    xmax = max(b[2] for b in all_bounds)
    ymax = max(b[3] for b in all_bounds)

    # Track colorbar (only add once)
    cbar_added = [False]

    def init():
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        return []

    def get_color_values_for_keyframe(idx: int) -> Optional[np.ndarray]:
        """Get color values for a specific keyframe."""
        if color_values is not None:
            return color_values[idx]
        elif column is not None and column in gdfs[idx].columns:
            return gdfs[idx][column].values
        return None

    def get_frame_data_position_mapper(frame: int) -> tuple["GeoDataFrame", int, float, Optional[np.ndarray]]:
        """Get the GeoDataFrame and colors using position mapper approach."""
        # Map frame to progress [0, 1]
        progress = frame / max(1, actual_n_frames - 1)

        # Map progress to position in keyframe space [0, n_keyframes-1]
        position = mapper(progress, variables_arrays)
        position = np.clip(position, 0, n_keyframes - 1)

        # Extract integer index and fractional part
        idx_before = int(position)
        idx_after = min(idx_before + 1, n_keyframes - 1)
        t = position - idx_before

        if interpolation == "nearest":
            # Snap to nearest keyframe
            idx = idx_before if t < 0.5 else idx_after
            colors = get_color_values_for_keyframe(idx)
            return gdfs[idx], idx, 0.0 if t < 0.5 else 1.0, colors

        else:  # linear interpolation
            if idx_before == idx_after or t <= 0:
                colors = get_color_values_for_keyframe(idx_before)
                return gdfs[idx_before], idx_before, 0.0, colors
            elif t >= 1:
                colors = get_color_values_for_keyframe(idx_after)
                return gdfs[idx_after], idx_after, 0.0, colors
            else:
                # Get color values for interpolation
                colors_before = get_color_values_for_keyframe(idx_before)
                colors_after = get_color_values_for_keyframe(idx_after)

                gdf = _interpolate_geodataframe(
                    gdfs[idx_before],
                    gdfs[idx_after],
                    t,
                    color_values_a=colors_before,
                    color_values_b=colors_after,
                    color_column="_color",
                )

                # Get interpolated colors from the GeoDataFrame
                colors = gdf["_color"].values if "_color" in gdf.columns else None

                return gdf, idx_before, t, colors

    def get_frame_data_legacy(frame: int) -> tuple["GeoDataFrame", int, float, Optional[np.ndarray]]:
        """Get the GeoDataFrame and colors using legacy hold_frames approach."""
        # Check if we're in a hold period
        frames_per_segment = transition_frames + hold_frames
        segment_idx = frame // frames_per_segment
        frame_in_segment = frame % frames_per_segment

        if segment_idx >= n_keyframes - 1:
            # Last keyframe
            colors = get_color_values_for_keyframe(n_keyframes - 1)
            return gdfs[-1], n_keyframes - 1, 0.0, colors

        if frame_in_segment < hold_frames:
            # Holding at keyframe
            colors = get_color_values_for_keyframe(segment_idx)
            return gdfs[segment_idx], segment_idx, 0.0, colors

        # Transitioning
        t = (frame_in_segment - hold_frames) / transition_frames

        if interpolation == "nearest":
            # Jump to nearest keyframe
            nearest_idx = segment_idx if t < 0.5 else segment_idx + 1
            colors = get_color_values_for_keyframe(nearest_idx)
            return gdfs[nearest_idx], nearest_idx, 0.0 if t < 0.5 else 1.0, colors

        # Linear interpolation
        colors_before = get_color_values_for_keyframe(segment_idx)
        colors_after = get_color_values_for_keyframe(segment_idx + 1)

        gdf = _interpolate_geodataframe(
            gdfs[segment_idx],
            gdfs[segment_idx + 1],
            t,
            color_values_a=colors_before,
            color_values_b=colors_after,
            color_column="_color",
        )

        # Get interpolated colors from the GeoDataFrame
        colors = gdf["_color"].values if "_color" in gdf.columns else None

        return gdf, segment_idx, t, colors

    # Select the appropriate frame getter
    get_frame_data = get_frame_data_position_mapper if use_position_mapper else get_frame_data_legacy

    def update(frame):
        ax.clear()
        gdf, keyframe_idx, t, colors = get_frame_data(frame)

        # Determine how to plot
        if use_coloring and colors is not None and color_vmin is not None:
            # Add color values to a temporary column for plotting
            plot_gdf = gdf.copy()
            plot_gdf["_plot_color"] = colors
            mappable = plot_gdf.plot(
                ax=ax, column="_plot_color", legend=False, vmin=color_vmin, vmax=color_vmax, cmap=cmap, **kwargs
            )
        elif use_coloring and column is not None and column in gdf.columns and color_vmin is not None:
            mappable = gdf.plot(
                ax=ax, column=column, legend=False, vmin=color_vmin, vmax=color_vmax, cmap=cmap, **kwargs
            )
        else:
            gdf.plot(ax=ax, **kwargs)
            mappable = None

        # Add colorbar on first frame if requested
        if colorbar and not cbar_added[0] and mappable is not None and color_vmin is not None:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=color_vmin, vmax=color_vmax))
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label=colorbar_label)
            cbar_added[0] = True

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")

        if t == 0.0:
            ax.set_title(f"Keyframe {keyframe_idx + 1}/{n_keyframes}")
        else:
            ax.set_title(f"Transition {keyframe_idx + 1} -> {keyframe_idx + 2} ({t:.0%})")

        return []

    anim = FuncAnimation(
        fig,
        update,
        frames=actual_n_frames,
        init_func=init,
        blit=False,
        interval=1000 / fps,
    )

    return anim


# =============================================================================
# Field animation functions
# =============================================================================


def animate_fields(
    result: "MorphResult",
    duration: float = 5.0,
    fps: int = 15,
    interpolation: str = "nearest",
    position_mapper: Optional[PositionMapper] = None,
    bounds: Optional[Any] = None,
    # What to show
    show_density: bool = True,
    show_velocity: bool = False,
    # Density options
    normalize: Optional[str] = None,
    density_cmap: Optional[str] = None,
    density_alpha: float = 1.0,
    # Velocity options
    velocity_skip: int = 4,
    velocity_color: str = "white",
    velocity_alpha: float = 0.8,
    velocity_scale: Optional[float] = None,
    velocity_color_by: Optional[str] = None,
    velocity_cmap: Optional[str] = None,
    velocity_colorbar: bool = False,
    velocity_alpha_by_magnitude: bool = False,
    velocity_alpha_range: tuple[float, float] = (0.2, 1.0),
    velocity_kwargs: Optional[dict[str, Any]] = None,
    figsize: tuple[float, float] = (10, 8),
    **kwargs: Any,
) -> FuncAnimation:
    """Animate density and/or velocity fields over iterations.

    Parameters
    ----------
    result : MorphResult
        Morphing result with history_internals (requires save_internals=True)
    duration : float, default=5.0
        Animation duration in seconds
    fps : int, default=15
        Frames per second
    interpolation : str, default="nearest"
        Interpolation method: "nearest" or "linear"
    position_mapper : callable, optional
        Function to control animation timing. Use linear_over() factory.
    bounds : str, float, or tuple, optional
        Clip view to specified bounds
    show_density : bool, default=True
        Whether to show density field heatmap
    show_velocity : bool, default=False
        Whether to show velocity field arrows
    normalize : str, optional
        Density normalization: None, "difference", or "ratio"
    density_cmap : str, optional
        Colormap for density field. Defaults based on normalize setting.
    density_alpha : float, default=1.0
        Alpha (transparency) for density heatmap
    velocity_skip : int, default=4
        Plot every nth velocity arrow
    velocity_color : str, default="white"
        Color for velocity arrows (when not using velocity_color_by)
    velocity_alpha : float, default=0.8
        Alpha for velocity arrows
    velocity_scale : float, optional
        Scale factor for arrow lengths
    velocity_color_by : str, optional
        Color arrows by: None, "magnitude", or "direction"
    velocity_cmap : str, optional
        Colormap for velocity coloring
    velocity_colorbar : bool, default=False
        Whether to show colorbar for velocity
    velocity_alpha_by_magnitude : bool, default=False
        If True, arrow transparency varies with magnitude
    velocity_alpha_range : tuple[float, float], default=(0.2, 1.0)
        Min and max alpha when velocity_alpha_by_magnitude is True
    velocity_kwargs : dict, optional
        Additional arguments passed to quiver (for velocity arrows)
    figsize : tuple, default=(10, 8)
        Figure size (width, height)
    **kwargs
        Additional arguments passed to imshow (for density)

    Returns
    -------
    FuncAnimation
        Matplotlib animation object

    Examples
    --------
    >>> # Density field only (default)
    >>> anim = animate_fields(result, normalize="ratio")

    >>> # Velocity field only
    >>> anim = animate_fields(result, show_density=False, show_velocity=True)

    >>> # Combined: density with velocity overlay
    >>> anim = animate_fields(
    ...     result, show_velocity=True, normalize="ratio",
    ...     velocity_color="black", velocity_alpha_by_magnitude=True
    ... )
    """
    from matplotlib.colors import to_rgba

    from .visualization import _resolve_bounds

    if interpolation not in ("nearest", "linear"):
        raise ValueError(f"interpolation must be 'nearest' or 'linear', got {interpolation!r}")

    if not show_density and not show_velocity:
        raise ValueError("At least one of show_density or show_velocity must be True")

    # Validate inputs
    if not hasattr(result, "history_internals") or result.history_internals is None:
        raise ValueError("result.history_internals is required. Run morphing with save_internals=True.")

    if result.grid is None:
        raise ValueError("result.grid is required for field animation")

    snapshots = result.history_internals.snapshots
    if not snapshots:
        raise ValueError("result.history_internals contains no snapshots")

    # Filter snapshots based on what we need to show
    def has_density(s: Any) -> bool:
        return hasattr(s, "rho") and s.rho is not None

    def has_velocity(s: Any) -> bool:
        return hasattr(s, "vx") and s.vx is not None and hasattr(s, "vy") and s.vy is not None

    if show_density and show_velocity:
        valid_snapshots = [s for s in snapshots if has_density(s) and has_velocity(s)]
        if not valid_snapshots:
            raise ValueError("No snapshots with both density and velocity fields found")
    elif show_density:
        valid_snapshots = [s for s in snapshots if has_density(s)]
        if not valid_snapshots:
            raise ValueError("No density field snapshots found in history_internals")
    else:  # show_velocity only
        valid_snapshots = [s for s in snapshots if has_velocity(s)]
        if not valid_snapshots:
            raise ValueError("No velocity field snapshots found in history_internals")

    grid = result.grid
    n_snapshots = len(valid_snapshots)
    n_frames = int(duration * fps)

    # Build variables dict from snapshots
    iterations = [s.iteration for s in valid_snapshots]
    variables: dict[str, list] = {
        "iteration": iterations,
        "mean_error": [],
        "max_error": [],
    }

    # Get errors if available (from main history), interpolated to match iterations
    if hasattr(result, "history") and result.history is not None:
        history_iterations = result.history.get_variable_history("iteration")
        mean_errors = result.history.get_variable_history("mean_error")
        max_errors = result.history.get_variable_history("max_error")

        if history_iterations and mean_errors:
            valid_pairs = [
                (it, err) for it, err in zip(history_iterations, mean_errors) if it is not None and err is not None
            ]
            if valid_pairs:
                hist_iters = np.array([p[0] for p in valid_pairs])
                hist_errors = np.array([p[1] for p in valid_pairs])
                variables["mean_error"] = list(np.interp(iterations, hist_iters, hist_errors))

        if history_iterations and max_errors:
            valid_pairs = [
                (it, err) for it, err in zip(history_iterations, max_errors) if it is not None and err is not None
            ]
            if valid_pairs:
                hist_iters = np.array([p[0] for p in valid_pairs])
                hist_errors = np.array([p[1] for p in valid_pairs])
                variables["max_error"] = list(np.interp(iterations, hist_iters, hist_errors))

    # Convert lists to arrays
    variables_arrays: dict[str, np.ndarray] = {}
    variables_arrays["index"] = np.arange(n_snapshots)
    for key, values in variables.items():
        if values and all(v is not None for v in values):
            variables_arrays[key] = np.array(values)

    mapper = position_mapper if position_mapper is not None else _default_position_mapper

    # Set up figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    extent = [grid.xmin, grid.xmax, grid.ymin, grid.ymax]

    # Compute global ranges for density
    density_vmin, density_vmax, effective_density_cmap = None, None, None
    if show_density:
        if normalize == "difference":
            all_diffs = []
            for snap in valid_snapshots:
                if hasattr(snap, "mean_density") and snap.mean_density is not None:
                    all_diffs.append(np.abs(snap.rho - snap.mean_density).max())
            density_vmax = max(all_diffs) if all_diffs else 1.0
            density_vmin = -density_vmax
            effective_density_cmap = density_cmap or "RdBu_r"
        elif normalize == "ratio":
            all_ratios = []
            for snap in valid_snapshots:
                if hasattr(snap, "mean_density") and snap.mean_density is not None:
                    ratio = snap.rho / snap.mean_density
                    ratio_clipped = np.clip(ratio, 1e-6, None)
                    all_ratios.append(np.abs(np.log2(ratio_clipped)).max())
            max_log = max(all_ratios) if all_ratios else 1.0
            density_vmin, density_vmax = -max_log, max_log
            effective_density_cmap = density_cmap or "RdBu_r"
        else:
            all_rho = [snap.rho for snap in valid_snapshots]
            density_vmin = min(r.min() for r in all_rho)
            density_vmax = max(r.max() for r in all_rho)
            effective_density_cmap = density_cmap or "viridis"

    # Compute global ranges for velocity
    mag_min, mag_max = 0.0, 1.0
    X_sub, Y_sub = None, None
    if show_velocity:
        X, Y = grid.X, grid.Y
        X_sub = X[::velocity_skip, ::velocity_skip]
        Y_sub = Y[::velocity_skip, ::velocity_skip]

        all_mags = []
        for snap in valid_snapshots:
            vx_sub = snap.vx[::velocity_skip, ::velocity_skip]
            vy_sub = snap.vy[::velocity_skip, ::velocity_skip]
            mag = np.sqrt(vx_sub**2 + vy_sub**2)
            all_mags.extend([mag.min(), mag.max()])
        mag_min, mag_max = min(all_mags), max(all_mags)

    # Determine velocity colormap
    if velocity_color_by == "magnitude":
        effective_velocity_cmap = velocity_cmap or "viridis"
        velocity_colorbar_label = "Magnitude"
    elif velocity_color_by == "direction":
        effective_velocity_cmap = velocity_cmap or "twilight"
        velocity_colorbar_label = "Direction"
    else:
        effective_velocity_cmap = velocity_cmap
        velocity_colorbar_label = None

    base_arrow_rgba = to_rgba(velocity_color, alpha=velocity_alpha)
    density_cbar_added = [False]
    velocity_cbar_added = [False]

    def init():
        ax.set_aspect("equal")
        return []

    def apply_normalization(rho: np.ndarray, snap: Any) -> tuple[np.ndarray, str]:
        if normalize == "difference":
            if hasattr(snap, "mean_density") and snap.mean_density is not None:
                return rho - snap.mean_density, "Density - Mean"
            return rho, "Density - Mean"
        elif normalize == "ratio":
            if hasattr(snap, "mean_density") and snap.mean_density is not None:
                ratio = rho / snap.mean_density
                return np.log2(np.clip(ratio, 1e-6, None)), "log2(Density / Mean)"
            return rho, "log2(Density / Mean)"
        return rho, "Density"

    def get_frame_data(frame: int) -> tuple[Any, Any, Any, float]:
        """Get fields for a frame. Returns (rho, vx, vy, iteration)."""
        progress = frame / max(1, n_frames - 1)
        position = mapper(progress, variables_arrays)
        position = np.clip(position, 0, n_snapshots - 1)

        idx_before = int(position)
        idx_after = min(idx_before + 1, n_snapshots - 1)
        t = position - idx_before

        iters = variables_arrays.get("iteration", np.arange(n_snapshots))

        if interpolation == "nearest":
            idx = idx_before if t < 0.5 else idx_after
            snap = valid_snapshots[idx]
            rho = snap.rho if show_density else None
            vx = snap.vx if show_velocity else None
            vy = snap.vy if show_velocity else None
            return rho, vx, vy, iters[idx]

        else:  # linear
            if idx_before == idx_after or t <= 0:
                snap = valid_snapshots[idx_before]
                rho = snap.rho if show_density else None
                vx = snap.vx if show_velocity else None
                vy = snap.vy if show_velocity else None
                return rho, vx, vy, iters[idx_before]
            elif t >= 1:
                snap = valid_snapshots[idx_after]
                rho = snap.rho if show_density else None
                vx = snap.vx if show_velocity else None
                vy = snap.vy if show_velocity else None
                return rho, vx, vy, iters[idx_after]
            else:
                snap_before = valid_snapshots[idx_before]
                snap_after = valid_snapshots[idx_after]
                rho = None
                if show_density:
                    rho = snap_before.rho * (1 - t) + snap_after.rho * t
                vx, vy = None, None
                if show_velocity:
                    vx = snap_before.vx * (1 - t) + snap_after.vx * t
                    vy = snap_before.vy * (1 - t) + snap_after.vy * t
                iteration = iters[idx_before] + t * (iters[idx_after] - iters[idx_before])
                return rho, vx, vy, iteration

    def update(frame):
        ax.clear()
        rho, vx, vy, iteration = get_frame_data(frame)

        idx = min(int(frame / max(1, n_frames - 1) * (n_snapshots - 1)), n_snapshots - 1)
        snap = valid_snapshots[idx]

        # Draw density field
        if show_density and rho is not None:
            rho_display, label = apply_normalization(rho, snap)
            im = ax.imshow(
                rho_display,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap=effective_density_cmap,
                vmin=density_vmin,
                vmax=density_vmax,
                alpha=density_alpha,
                **kwargs,
            )
            if not density_cbar_added[0]:
                plt.colorbar(im, ax=ax, label=label)
                density_cbar_added[0] = True

        # Draw velocity field
        if show_velocity and vx is not None and vy is not None:
            vx_sub = vx[::velocity_skip, ::velocity_skip]
            vy_sub = vy[::velocity_skip, ::velocity_skip]

            if velocity_color_by is not None or velocity_alpha_by_magnitude:
                magnitude = np.sqrt(vx_sub**2 + vy_sub**2)
                if mag_max > mag_min:
                    magnitude_norm = np.clip((magnitude - mag_min) / (mag_max - mag_min), 0, 1)
                else:
                    magnitude_norm = np.ones_like(magnitude)

                if velocity_color_by == "magnitude":
                    C_norm = magnitude_norm
                elif velocity_color_by == "direction":
                    direction = np.arctan2(vy_sub, vx_sub)
                    C_norm = (direction + np.pi) / (2 * np.pi)
                else:
                    C_norm = np.zeros_like(magnitude)

                if effective_velocity_cmap is not None:
                    cmap_obj = plt.get_cmap(effective_velocity_cmap)
                    colors = cmap_obj(C_norm.ravel())
                else:
                    colors = np.tile(base_arrow_rgba, (C_norm.size, 1))

                if velocity_alpha_by_magnitude:
                    alpha_min, alpha_max = velocity_alpha_range
                    alphas = alpha_min + magnitude_norm.ravel() * (alpha_max - alpha_min)
                    colors = np.array(colors)  # Ensure mutable
                    colors[:, 3] = alphas

                ax.quiver(X_sub, Y_sub, vx_sub, vy_sub, color=colors, scale=velocity_scale, **(velocity_kwargs or {}))

                if velocity_colorbar and not velocity_cbar_added[0] and velocity_color_by is not None:
                    sm = plt.cm.ScalarMappable(cmap=effective_velocity_cmap, norm=plt.Normalize(vmin=0, vmax=1))
                    if velocity_color_by == "magnitude":
                        sm.set_clim(mag_min, mag_max)
                    elif velocity_color_by == "direction":
                        sm.set_clim(-np.pi, np.pi)
                    sm.set_array([])
                    fig.colorbar(sm, ax=ax, label=velocity_colorbar_label)
                    velocity_cbar_added[0] = True
            else:
                ax.quiver(
                    X_sub,
                    Y_sub,
                    vx_sub,
                    vy_sub,
                    color=velocity_color,
                    alpha=velocity_alpha,
                    scale=velocity_scale,
                    **(velocity_kwargs or {}),
                )

        ax.set_aspect("equal")

        # Title based on what we're showing
        if show_density and show_velocity:
            title = "Density + Velocity"
        elif show_density:
            title = "Density Field"
        else:
            title = "Velocity Field"

        if isinstance(iteration, float) and not float(iteration).is_integer():
            ax.set_title(f"{title} (iteration {iteration:.1f})")
        else:
            ax.set_title(f"{title} (iteration {int(iteration)})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        resolved_bounds = _resolve_bounds(bounds, grid)
        if resolved_bounds is not None:
            xmin, ymin, xmax, ymax = resolved_bounds
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        return []

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        blit=False,
        interval=1000 / fps,
    )

    return anim


def animate_density_field(
    result: "MorphResult",
    duration: float = 5.0,
    fps: int = 15,
    interpolation: str = "nearest",
    position_mapper: Optional[PositionMapper] = None,
    bounds: Optional[Any] = None,
    normalize: Optional[str] = None,
    figsize: tuple[float, float] = (10, 8),
    **kwargs: Any,
) -> FuncAnimation:
    """Animate density field evolution. Convenience wrapper for animate_fields.

    See animate_fields for full documentation.
    """
    return animate_fields(
        result,
        duration=duration,
        fps=fps,
        interpolation=interpolation,
        position_mapper=position_mapper,
        bounds=bounds,
        show_density=True,
        show_velocity=False,
        normalize=normalize,
        density_cmap=kwargs.pop("cmap", None),
        figsize=figsize,
        **kwargs,
    )


def animate_velocity_field(
    result: "MorphResult",
    duration: float = 5.0,
    fps: int = 15,
    interpolation: str = "nearest",
    position_mapper: Optional[PositionMapper] = None,
    bounds: Optional[Any] = None,
    skip: int = 4,
    color_by: Optional[str] = None,
    cmap: Optional[str] = None,
    colorbar: bool = True,
    alpha_by_magnitude: bool = False,
    alpha_range: tuple[float, float] = (0.2, 1.0),
    figsize: tuple[float, float] = (10, 8),
    **kwargs: Any,
) -> FuncAnimation:
    """Animate velocity field evolution. Convenience wrapper for animate_fields.

    See animate_fields for full documentation.
    """
    return animate_fields(
        result,
        duration=duration,
        fps=fps,
        interpolation=interpolation,
        position_mapper=position_mapper,
        bounds=bounds,
        show_density=False,
        show_velocity=True,
        velocity_skip=skip,
        velocity_color_by=color_by,
        velocity_cmap=cmap,
        velocity_colorbar=colorbar,
        velocity_alpha_by_magnitude=alpha_by_magnitude,
        velocity_alpha_range=alpha_range,
        velocity_kwargs=kwargs,
        figsize=figsize,
    )
