"""
Animation utilities for cartogram results.

Animation functions for visualizing morphing results, including geometry
transitions, density field evolution, and velocity field dynamics.

Functions
---------
animate_morph_history
    Animate through algorithm snapshots.
animate_geometry_keyframes
    Animate between arbitrary geometry states.
animate_density_field
    Animate density field evolution.
animate_velocity_field
    Animate velocity field evolution.
save_animation
    Save animations to various formats.

Examples
--------
>>> from carto_flow import CartogramWorkflow, MorphOptions
>>> from carto_flow.flow_cartogram.animation import animate_morph_history, save_animation
>>>
>>> workflow = CartogramWorkflow(gdf, 'population')
>>> cartogram = workflow.morph(options=MorphOptions(save_history=True))
>>> anim = animate_morph_history(cartogram, duration=5.0, fps=15)
>>> save_animation(anim, "morph.gif", fps=15)
"""

import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

from .._plot_params import classify_string, deprecated_params
from ..style import PlotStyle, rc_wrapped, resolve_style

if TYPE_CHECKING:
    from geopandas import GeoDataFrame

    from .cartogram import Cartogram
    from .visualization import DensityPlotOptions, VelocityPlotOptions
    from .workflow import CartogramWorkflow

__all__ = [
    "animate_density_field",
    "animate_fields",
    "animate_geometry_keyframes",
    "animate_morph_history",
    "animate_velocity_field",
    "animate_workflow",
    "animate_workflow_fields",
    "linear_over",
    "morph_timeline",
    "save_animation",
    "weights_to_position_mapper",
]

# Type alias for position mapper functions
# Signature: (progress: float, variables: dict[str, np.ndarray]) -> float
PositionMapper = Callable[[float, dict[str, np.ndarray]], float]


def _geometry_to_mpl_path(geom: Any) -> Any:
    """Convert a shapely Polygon or MultiPolygon to a matplotlib Path."""
    from matplotlib.path import Path as MplPath
    from shapely.geometry import MultiPolygon

    if isinstance(geom, MultiPolygon):
        sub = [
            MplPath.make_compound_path(
                MplPath(np.asarray(p.exterior.coords)[:, :2]),
                *[MplPath(np.asarray(r.coords)[:, :2]) for r in p.interiors],
            )
            for p in geom.geoms
        ]
        verts = np.concatenate([p.vertices for p in sub])
        codes = np.concatenate([p.codes for p in sub])
        return MplPath(verts, codes)
    else:
        return MplPath.make_compound_path(
            MplPath(np.asarray(geom.exterior.coords)[:, :2]),
            *[MplPath(np.asarray(r.coords)[:, :2]) for r in geom.interiors],
        )


def _extract_geometry_rings(geom: Any) -> list[np.ndarray]:
    """Extract coordinate rings from a Polygon or MultiPolygon as numpy arrays.

    Returns a flat list of arrays (exterior first, then holes) suitable for
    passing to `_rings_to_mpl_path`. The ring structure is preserved so that
    `Path.make_compound_path` renders filled polygons with holes correctly.
    """
    from shapely.geometry import MultiPolygon

    if isinstance(geom, MultiPolygon):
        rings: list[np.ndarray] = []
        for p in geom.geoms:
            rings.append(np.asarray(p.exterior.coords)[:, :2])
            for r in p.interiors:
                rings.append(np.asarray(r.coords)[:, :2])
        return rings
    else:
        rings = [np.asarray(geom.exterior.coords)[:, :2]]
        for r in geom.interiors:
            rings.append(np.asarray(r.coords)[:, :2])
        return rings


def _rings_to_mpl_path(rings: list[np.ndarray]) -> Any:
    """Build a matplotlib Path from a list of coordinate ring arrays."""
    from matplotlib.path import Path as MplPath

    return MplPath.make_compound_path(*[MplPath(ring) for ring in rings])


def _interpolate_rings_to_path(rings_a: list[np.ndarray], rings_b: list[np.ndarray], t: float) -> Any:
    """Interpolate between two ring lists and return a matplotlib Path.

    Performs pure numpy linear interpolation — no shapely object creation.
    Assumes rings_a and rings_b have the same structure (same ring count and
    vertex counts), which holds when both come from the same base geometry.
    """
    return _rings_to_mpl_path([a * (1.0 - t) + b * t for a, b in zip(rings_a, rings_b, strict=False)])


def _fast_precompute_paths(
    gdfs: list,
    keyframe_color_arrays: list,
    actual_n_frames: int,
    n_keyframes: int,
    position_fn: Callable[[int], tuple[int, int, float]],
    interpolation: str,
) -> tuple[list, list, list]:
    """Fast precompute for polygon-geometry animations.

    Extracts coordinate rings from keyframes once, then interpolates with pure
    numpy for each frame — no shapely Polygon construction, no GeoDataFrame copying.

    Parameters
    ----------
    gdfs : list of GeoDataFrame
        One GeoDataFrame per keyframe. All geometries must be Polygon/MultiPolygon.
    keyframe_color_arrays : list of ndarray or None
        Per-keyframe color arrays (one entry per keyframe, may be None).
    actual_n_frames : int
        Total number of frames to precompute.
    n_keyframes : int
        Number of keyframes.
    position_fn : callable
        ``position_fn(frame) -> (idx_a, idx_b, t)`` where idx_a/idx_b are the
        surrounding keyframe indices and t is the interpolation fraction in [0, 1].
    interpolation : str
        ``"nearest"`` or ``"linear"``.

    Returns
    -------
    precomputed_paths : list of list of Path
        Per-frame list of matplotlib Paths (one per geometry).
    precomputed_colors : list of ndarray or None
        Per-frame interpolated color arrays.
    precomputed_positions : list of (int, float)
        Per-frame ``(idx_a, t)`` for use in title/metadata computation.
    """
    _keyframe_rings = [[_extract_geometry_rings(g) for g in gdf.geometry] for gdf in gdfs]

    precomputed_paths: list = []
    precomputed_colors: list = []
    precomputed_positions: list = []

    for _f in range(actual_n_frames):
        idx_a, idx_b, _t = position_fn(_f)

        if interpolation == "nearest":
            idx = idx_a if _t < 0.5 else idx_b
            paths = [_rings_to_mpl_path(rings) for rings in _keyframe_rings[idx]]
            colors = keyframe_color_arrays[idx]
            precomputed_positions.append((idx, 0.0))
        elif _t <= 0 or idx_a == idx_b:
            paths = [_rings_to_mpl_path(rings) for rings in _keyframe_rings[idx_a]]
            colors = keyframe_color_arrays[idx_a]
            precomputed_positions.append((idx_a, 0.0))
        elif _t >= 1:
            paths = [_rings_to_mpl_path(rings) for rings in _keyframe_rings[idx_b]]
            colors = keyframe_color_arrays[idx_b]
            precomputed_positions.append((idx_b, 0.0))
        else:
            paths = [
                _interpolate_rings_to_path(ra, rb, _t)
                for ra, rb in zip(_keyframe_rings[idx_a], _keyframe_rings[idx_b], strict=False)
            ]
            ca = keyframe_color_arrays[idx_a]
            cb = keyframe_color_arrays[idx_b]
            colors = (ca * (1.0 - _t) + cb * _t) if (ca is not None and cb is not None) else ca
            precomputed_positions.append((idx_a, _t))

        precomputed_paths.append(paths)
        precomputed_colors.append(colors)

    return precomputed_paths, precomputed_colors, precomputed_positions


def _create_patch_collection(
    frame0_paths: list,
    frame0_colors: Any,
    norm: Any,
    cmap: str,
    coll_kwargs: dict,
    legend: bool,
    legend_label: str | None,
    legend_kwargs: dict | None,
    ax: Any,
    fig: Any,
    cbar_added: list,
    style: PlotStyle | None = None,
) -> Any:
    """Create a PatchCollection from the first frame and add it to the axes.

    Also adds a colorbar if requested and marks ``cbar_added[0] = True``.
    Returns the PatchCollection.
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import PathPatch

    patch_coll = PatchCollection(
        [PathPatch(p) for p in frame0_paths],
        norm=norm,
        cmap=cmap,
        **coll_kwargs,
    )
    if frame0_colors is not None:
        patch_coll.set_array(np.asarray(frame0_colors).ravel())
    ax.add_collection(patch_coll)
    if legend:
        with style.rc_context() if style is not None else contextlib.nullcontext():
            (style or PlotStyle()).add_colorbar(fig, ax, cmap, norm, legend_label, **(legend_kwargs or {}))
    cbar_added[0] = True
    return patch_coll


class _SafeFormatDict(dict):
    """Format dict that returns NaN for unknown keys instead of raising KeyError."""

    def __missing__(self, key: str) -> float:
        return float("nan")


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
    key_or_values: str | np.ndarray,
    decreasing: bool | None = None,
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
    source : GeoDataFrame or Cartogram
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
    # Check for Cartogram first (has .best property with .geometry)
    if hasattr(source, "best") and source.best is not None:
        snapshot = source.best
        if hasattr(snapshot, "geometry") and snapshot.geometry is not None:
            # Return geometry as a GeoDataFrame-like object
            import geopandas as gpd

            return gpd.GeoDataFrame(geometry=list(snapshot.geometry))
    # Check for GeoDataFrame (has .geometry attribute)
    if hasattr(source, "geometry"):
        return source
    raise TypeError(f"Expected GeoDataFrame or Cartogram, got {type(source).__name__}")


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
    color_values_a: np.ndarray | None = None,
    color_values_b: np.ndarray | None = None,
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
        _interpolate_geometry(geom_a, geom_b, t) for geom_a, geom_b in zip(gdf_a.geometry, gdf_b.geometry, strict=False)
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
    path: str | Path,
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

    # The saved frames have the surface of the figure (the style's), not the default white of the writer
    figure_facecolor = anim._fig.get_facecolor()  # type: ignore[attr-defined]
    savefig_kwargs = dict(kwargs.pop("savefig_kwargs", None) or {})
    savefig_kwargs.setdefault("facecolor", figure_facecolor)
    kwargs["savefig_kwargs"] = savefig_kwargs

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
        fig = anim._fig  # type: ignore[attr-defined]

        # Save each frame
        for i, _ in enumerate(range(len(anim._save_seq) if hasattr(anim, "_save_seq") else anim.save_count)):  # type: ignore[attr-defined]
            anim._draw_frame(i)  # type: ignore[attr-defined]
            fig.savefig(path / f"frame_{i:04d}.png", dpi=dpi, facecolor=savefig_kwargs["facecolor"])

    else:
        raise ValueError(
            f"Unsupported output format: {path.suffix}. Use .gif, .mp4, or a directory path for PNG frames."
        )


# =============================================================================
# Geometry animation functions
# =============================================================================


# Built-in quantities of a cartogram that ``facecolor`` accepts in the animations
_FILL_QUANTITIES = ("errors_pct", "density", "original")


def _resolve_fill(facecolor: Any, label: str | None, sources: list, builtins: Any, kwargs: dict) -> tuple[Any, Any]:
    """Resolve a string ``facecolor`` of an animation: a color becomes a constant fill in ``kwargs``.

    A string that is a column and also a built-in quantity is read as the column (as a static array per source), so
    the rule of :func:`carto_flow._plot_params.classify_string` holds. Returns the resolved ``facecolor`` and label.
    """
    if not isinstance(facecolor, str):
        return facecolor, label
    sources = [src for src in sources if src is not None]
    columns = {c for src in sources for c in src.columns}
    kind = classify_string(facecolor, columns, builtins, stacklevel=4)
    if kind == "color":
        kwargs.setdefault("facecolor", facecolor)
        return None, label
    if kind == "column" and facecolor in builtins:
        arrays = [np.asarray(src[facecolor].values) for src in sources if facecolor in src.columns]
        return (arrays[0] if len(arrays) == 1 else arrays), (label if label is not None else facecolor)
    return facecolor, label


@deprecated_params(
    facecolor="color_by", legend="colorbar", legend_label="colorbar_label", legend_kwargs="colorbar_kwargs"
)
def animate_morph_history(
    cartogram: "Cartogram",
    duration: float = 5.0,
    fps: int = 15,
    interpolation: str = "nearest",
    position_mapper: PositionMapper | None = None,
    facecolor: str | np.ndarray | list | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: Any = None,
    legend: bool = False,
    legend_label: str | None = None,
    show_axes: bool = True,
    legend_kwargs: dict | None = None,
    title: str | bool | None = None,
    precompute: bool = True,
    figsize: tuple[float, float] = (10, 8),
    through: Literal["best", "last"] = "best",
    norm: Any = None,
    center: float | None = None,
    style: Any = None,
    **kwargs: Any,
) -> FuncAnimation:
    """Animate through algorithm snapshots from a Cartogram's history.

    Parameters
    ----------
    cartogram : Cartogram
        Cartogram result with snapshots containing geometry history
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
    facecolor : str, np.ndarray, or list of np.ndarray, optional
        Fill of the geometries:
        - a column name looked up in ``cartogram._source_gdf`` (static across snapshots)
        - ``"errors_pct"``: per-geometry percentage error at each snapshot
        - ``"density"``: per-geometry density (values/area) at each snapshot
        - ``"original"``: values from the column used to build the cartogram
          (static across snapshots; requires ``cartogram._value_column``)
        - a color (``"red"``, ``"#d4e6f1"``): constant fill
        - np.ndarray: single array broadcast to all snapshots (static)
        - list of np.ndarray: one array per snapshot

        A string that is a column and also a built-in quantity or a color is read as
        the column, then as the built-in quantity, and a warning is issued.
    vmin : float, optional
        Minimum value for color scale. If None, computed from all values.
    vmax : float, optional
        Maximum value for color scale. If None, computed from all values.
    cmap : str or Colormap, optional
        Colormap of the colors. Defaults to the palette of the style for the data role, else ``"viridis"``.
    legend : bool, default=False
        Whether to show a colorbar.
    legend_label : str, optional
        Label for the colorbar. Auto-set from ``facecolor`` if not provided.
    show_axes : bool, default=True
        If False, hides the axes frame, ticks, and labels (``ax.axis("off")``).
        Useful for clean map-style animations.
    legend_kwargs : dict, optional
        Additional keyword arguments passed to ``fig.colorbar()``.
        Example: ``dict(shrink=0.8, pad=0.02)``.
        The ``"label"`` key overrides the auto-set label.
    title : str, bool, or None, default=None
        Controls the axes title each frame.
        - ``None``: auto-generated title showing iteration and error (default)
        - ``False`` or ``""``: no title
        - str: format template rendered with ``str.format_map()`` each frame.
          Available keys: ``{iteration}``, ``{mean_error}``, ``{max_error}``,
          ``{mean_error_pct}``, ``{max_error_pct}``, ``{t}``.
          Unavailable keys (e.g. when no error history was saved) yield ``nan``.
    precompute : bool, default=True
        If True, all frames are computed before the animation starts.
        Eliminates per-frame geometry interpolation and GeoDataFrame copying,
        giving a significant speedup especially for ``interpolation="linear"``.
        Set to False to reduce peak memory usage for very long animations.
    figsize : tuple, default=(10, 8)
        Figure size (width, height)
    through : {"best", "last"}, default="best"
        Where the animation ends. ``"best"`` stops at the returned state
        (``best_iteration``); ``"last"`` continues through the iterations run after it.
    **kwargs
        Additional arguments passed to GeoDataFrame.plot()

    norm : Normalize, {"linear", "log", "auto"}, optional
        Normalization of the colors: an instance, or the scale of the style's normalization. A ``vmin`` or ``vmax``
        of the call fixes a linear normalization.
    center : float, optional
        Treat the colors as diverging around this value.
    style : PlotStyle or str, optional
        Style of the animation (default: the style of the block or session, else ``"default"``).
    Returns
    -------
    FuncAnimation
        Matplotlib animation object

    Raises
    ------
    ValueError
        If cartogram.snapshots doesn't contain geometry snapshots

    Notes
    -----
    Available variables for position_mapper (extracted from snapshots):
    - "iteration": Iteration numbers
    - "mean_error": Mean log2 errors (if available)
    - "max_error": Maximum log2 errors (if available)
    - "index": Snapshot indices [0, 1, 2, ...]

    Color limits are computed globally across all snapshots to ensure
    consistent coloring throughout the animation.

    Examples
    --------
    >>> workflow = CartogramWorkflow(gdf, 'pop')
    >>> cartogram = workflow.morph(options=MorphOptions(save_history=True))
    >>> anim = animate_morph_history(cartogram, duration=5.0, fps=15)
    >>> save_animation(anim, "morph.gif", fps=15)

    >>> # Color by per-geometry error percentage
    >>> anim = animate_morph_history(
    ...     cartogram, facecolor="errors_pct", cmap="RdYlGn_r", legend=True
    ... )

    >>> # Color by per-snapshot density
    >>> anim = animate_morph_history(cartogram, facecolor="density", cmap="plasma", legend=True)

    >>> # Color by original data column (static)
    >>> anim = animate_morph_history(cartogram, facecolor="original", legend=True)
    >>> anim = animate_morph_history(cartogram, facecolor="population", legend=True)

    >>> # Single static array
    >>> anim = animate_morph_history(cartogram, facecolor=gdf["population"].values)

    >>> # Per-snapshot arrays
    >>> anim = animate_morph_history(cartogram, facecolor=[snap.density for snap in cartogram.snapshots])
    """
    if interpolation not in ("nearest", "linear"):
        raise ValueError(f"interpolation must be 'nearest' or 'linear', got {interpolation!r}")

    # Extract geometry snapshots from cartogram
    if not hasattr(cartogram, "snapshots") or cartogram.snapshots is None:
        raise ValueError("cartogram.snapshots is required for animate_morph_history")

    if through not in ("best", "last"):
        raise ValueError(f"through must be 'best' or 'last', got {through!r}")
    snapshots = cartogram.snapshots.through_best() if through == "best" else cartogram.snapshots.snapshots
    if not snapshots:
        raise ValueError("cartogram.snapshots contains no snapshots")

    # Extract geometries and build variables dict from snapshots
    geometries = []
    snapshot_errors_pct: list[np.ndarray | None] = []
    snapshot_densities: list[np.ndarray | None] = []
    variables: dict[str, list] = {
        "iteration": [],
        "mean_error": [],
        "max_error": [],
        "mean_error_pct": [],
        "max_error_pct": [],
    }

    for snap in snapshots:
        if hasattr(snap, "geometry") and snap.geometry is not None:
            geometries.append(snap.geometry)
            variables["iteration"].append(snap.iteration)
            # Extract errors from the MorphErrors object
            if snap.errors is not None:
                variables["mean_error"].append(snap.errors.mean_log_error)
                variables["max_error"].append(snap.errors.max_log_error)
                variables["mean_error_pct"].append(snap.errors.mean_error_pct)
                variables["max_error_pct"].append(snap.errors.max_error_pct)
                snapshot_errors_pct.append(snap.errors.errors_pct)
            else:
                variables["mean_error"].append(None)
                variables["max_error"].append(None)
                variables["mean_error_pct"].append(None)
                variables["max_error_pct"].append(None)
                snapshot_errors_pct.append(None)
            # Extract per-geometry density
            if hasattr(snap, "density") and snap.density is not None:
                snapshot_densities.append(np.asarray(snap.density))
            else:
                snapshot_densities.append(None)

    if not geometries:
        raise ValueError(
            "No geometry snapshots found in history. "
            "Ensure morphing was run with save_history=True and geometry saving enabled."
        )

    n_snapshots = len(geometries)
    n_frames = int(duration * fps)

    facecolor, legend_label = _resolve_fill(
        facecolor, legend_label, [getattr(cartogram, "_source_gdf", None)], _FILL_QUANTITIES, kwargs
    )

    # Resolve facecolor to a list of per-snapshot arrays and a default label
    resolved_legend_label = legend_label
    color_values: list[np.ndarray] | None = None

    if isinstance(facecolor, np.ndarray):
        # Single static array — broadcast to all snapshots
        color_values = [np.asarray(facecolor)] * n_snapshots

    elif isinstance(facecolor, list):
        # Per-snapshot arrays
        if len(facecolor) != n_snapshots:
            raise ValueError(f"facecolor list length ({len(facecolor)}) must match number of snapshots ({n_snapshots})")
        color_values = [np.asarray(cv) for cv in facecolor]

    elif isinstance(facecolor, str):
        if facecolor == "errors_pct":
            if all(e is not None for e in snapshot_errors_pct):
                color_values = snapshot_errors_pct  # type: ignore[assignment]
            if resolved_legend_label is None:
                resolved_legend_label = "Error %"

        elif facecolor == "density":
            if all(d is not None for d in snapshot_densities):
                color_values = snapshot_densities  # type: ignore[assignment]
            if resolved_legend_label is None:
                resolved_legend_label = "Density"

        elif facecolor == "original":
            col = getattr(cartogram, "_value_column", None)
            src = getattr(cartogram, "_source_gdf", None)
            if col is None or src is None or col not in src.columns:
                raise ValueError(
                    "facecolor='original' requires cartogram._value_column and _source_gdf "
                    "(set automatically when using CartogramWorkflow)."
                )
            vals = np.asarray(src[col].values)
            color_values = [vals] * n_snapshots
            if resolved_legend_label is None:
                resolved_legend_label = col

        else:
            # Any other column name — look up in source GDF (static)
            src = getattr(cartogram, "_source_gdf", None)
            if src is not None and facecolor in src.columns:
                vals = np.asarray(src[facecolor].values)
                color_values = [vals] * n_snapshots
                if resolved_legend_label is None:
                    resolved_legend_label = facecolor
            else:
                available = list(src.columns) if src is not None else []
                raise ValueError(f"Column '{facecolor}' not found in source GeoDataFrame. Available: {available}")

    # Convert geometry arrays to GeoDataFrames for plotting
    import geopandas as gpd

    gdfs = [gpd.GeoDataFrame(geometry=list(geom)) for geom in geometries]

    # Determine coloring mode and compute global color limits
    use_coloring = color_values is not None

    computed_vmin = None
    computed_vmax = None
    if color_values is not None:
        computed_vmin = min(np.nanmin(cv) for cv in color_values)
        computed_vmax = max(np.nanmax(cv) for cv in color_values)

    color_vmin = vmin if vmin is not None else computed_vmin
    color_vmax = vmax if vmax is not None else computed_vmax

    legend_label = resolved_legend_label

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
    st = resolve_style(style)
    _role = (
        "diverging"
        if ((isinstance(facecolor, str) and facecolor == "errors_pct") or center is not None)
        else "sequential"
    )
    cmap = cmap if cmap is not None else (st.colormap(_role) or "viridis")
    _norm_obj = (
        st.resolve_norm(norm, _role, np.array([color_vmin, color_vmax]), center=center, limits=(vmin, vmax))
        or Normalize(vmin=color_vmin, vmax=color_vmax)
        if color_vmin is not None and color_vmax is not None
        else None
    )
    with st.rc_context():
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
        if not show_axes:
            ax.axis("off")
        return []

    def get_color_values_for_snapshot(idx: int) -> np.ndarray | None:
        """Get color values for a specific snapshot."""
        if color_values is not None:
            return color_values[idx]
        return None

    def get_frame_data(frame: int) -> tuple["GeoDataFrame", int, float, float | None, np.ndarray | None, float]:
        """Get the geometry and color values for a given frame.

        Returns (gdf, snapshot_idx, iteration, error, colors, t) where gdf and colors
        may be interpolated and t is the inter-snapshot fractional position.
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
            return gdfs[idx], idx, iterations[idx], error, colors, 0.0

        else:  # linear interpolation
            if idx_before == idx_after or t <= 0:
                gdf = gdfs[idx_before]
                iteration = iterations[idx_before]
                error = errors[idx_before] if errors is not None else None
                colors = get_color_values_for_snapshot(idx_before)
                return gdf, idx_before, iteration, error, colors, 0.0
            elif t >= 1:
                gdf = gdfs[idx_after]
                iteration = iterations[idx_after]
                error = errors[idx_after] if errors is not None else None
                colors = get_color_values_for_snapshot(idx_after)
                return gdf, idx_before, iteration, error, colors, 1.0
            else:
                # Get color values for interpolation
                colors_before = get_color_values_for_snapshot(idx_before)
                colors_after = get_color_values_for_snapshot(idx_after)

                gdf = _interpolate_geodataframe(
                    gdfs[idx_before],
                    gdfs[idx_after],
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

                return gdf, idx_before, iteration, error, colors, t

    # Determine if the fast PatchCollection path can be used
    _use_patch_coll = False
    _patch_coll = None
    if precompute and use_coloring and color_vmin is not None:
        from shapely.geometry import MultiPolygon, Polygon

        if all(isinstance(g, (Polygon, MultiPolygon)) for g in gdfs[0].geometry):
            _use_patch_coll = True

    # Pre-compute all frames
    _precomputed: list = []
    _precomputed_paths: list = []
    _precomputed_colors: list = []
    _precomputed_meta: list = []

    if _use_patch_coll:

        def _pos_fn(frame: int) -> tuple[int, int, float]:
            progress = frame / max(1, n_frames - 1)
            position = float(np.clip(mapper(progress, variables_arrays), 0, n_snapshots - 1))
            idx_a = int(position)
            return idx_a, min(idx_a + 1, n_snapshots - 1), position - idx_a

        _snap_colors = [get_color_values_for_snapshot(i) for i in range(n_snapshots)]
        _precomputed_paths, _precomputed_colors, _precomputed_meta = _fast_precompute_paths(
            gdfs, _snap_colors, n_frames, n_snapshots, _pos_fn, interpolation
        )
        _norm = _norm_obj
        _coll_kwargs = {k: v for k, v in kwargs.items() if k not in ("markersize", "marker")}
        _patch_coll = _create_patch_collection(
            _precomputed_paths[0],
            _precomputed_colors[0],
            _norm,
            cmap,
            _coll_kwargs,
            legend,
            legend_label,
            legend_kwargs,
            ax,
            fig,
            cbar_added,
            style=st,
        )

    elif precompute:
        for _f in range(n_frames):
            _gdf, _snap_idx, _iteration, _error, _colors, _t_val = get_frame_data(_f)
            if use_coloring and _colors is not None and color_vmin is not None:
                _plot_gdf = _gdf.assign(_plot_color=_colors)
                _has_colors = True
            else:
                _plot_gdf = _gdf
                _has_colors = False
            _precomputed.append((_plot_gdf, _snap_idx, _iteration, _error, _t_val, _has_colors))

    def update(frame):
        if _use_patch_coll:
            if _patch_coll is None:
                return []
            snapshot_idx, _t_val = _precomputed_meta[frame]
            idx_b = min(snapshot_idx + 1, n_snapshots - 1)
            from matplotlib.collections import Collection as _Collection

            _Collection.set_paths(_patch_coll, _precomputed_paths[frame])
            if _precomputed_colors[frame] is not None:
                _patch_coll.set_array(np.asarray(_precomputed_colors[frame]).ravel())
            iteration = float(iterations[snapshot_idx]) + _t_val * (
                float(iterations[idx_b]) - float(iterations[snapshot_idx])
            )
            mean_error = (
                float(errors[snapshot_idx]) + _t_val * (float(errors[idx_b]) - float(errors[snapshot_idx]))
                if errors is not None
                else None
            )
        else:
            ax.clear()
            if not show_axes:
                ax.axis("off")

            if precompute:
                plot_gdf, snapshot_idx, iteration, mean_error, _t_val, _has_colors = _precomputed[frame]
            else:
                _gdf, snapshot_idx, iteration, mean_error, _colors, _t_val = get_frame_data(frame)
                if use_coloring and _colors is not None and color_vmin is not None:
                    plot_gdf = _gdf.assign(_plot_color=_colors)
                    _has_colors = True
                else:
                    plot_gdf = _gdf
                    _has_colors = False

            # Determine how to plot
            if use_coloring and _has_colors and color_vmin is not None:
                mappable = plot_gdf.plot(ax=ax, column="_plot_color", legend=False, norm=_norm_obj, cmap=cmap, **kwargs)
            else:
                plot_gdf.plot(ax=ax, **kwargs)
                mappable = None

            # Add colorbar on first frame if requested
            if legend and not cbar_added[0] and mappable is not None and _norm_obj is not None:
                st.add_colorbar(fig, ax, cmap, _norm_obj, legend_label, **(legend_kwargs or {}))
                cbar_added[0] = True

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal")

        if title is None:
            error_str = ""
            if mean_error is not None:
                error_str = f", error={mean_error:.4f}"
            if isinstance(iteration, float) and not iteration.is_integer():
                ax.set_title(f"Iteration {iteration:.1f}{error_str}")
            else:
                ax.set_title(f"Iteration {int(iteration)}{error_str}")
        elif isinstance(title, str):
            _max_err_arr = variables_arrays.get("max_error")
            _mean_pct_arr = variables_arrays.get("mean_error_pct")
            _max_pct_arr = variables_arrays.get("max_error_pct")
            _nan = float("nan")
            _n_iters = variables["iteration"][-1] if variables["iteration"] else _nan
            _t_elapsed = frame * duration / max(1, n_frames - 1)
            ctx = _SafeFormatDict(
                iteration=iteration,
                n_iterations=_n_iters,
                t=_t_elapsed,
                duration=duration,
                mean_error=mean_error if mean_error is not None else _nan,
                max_error=_max_err_arr[snapshot_idx] if _max_err_arr is not None else _nan,
                mean_error_pct=_mean_pct_arr[snapshot_idx] if _mean_pct_arr is not None else _nan,
                max_error_pct=_max_pct_arr[snapshot_idx] if _max_pct_arr is not None else _nan,
            )
            ax.set_title(title.format_map(ctx))
        else:
            ax.set_title("")

        return [_patch_coll] if _use_patch_coll else []

    anim = FuncAnimation(
        fig,
        rc_wrapped(st, update),
        frames=n_frames,
        init_func=rc_wrapped(st, init),
        blit=False,
        interval=1000 / fps,
    )

    return anim


def morph_timeline(
    snapshots: Any,
    *,
    hold_start: float = 1.5,
    morph: float = 7.0,
    hold_end: float = 2.5,
    back: float = 5.0,
    easing: Literal["cosine", "linear"] = "cosine",
) -> tuple[list[Any], np.ndarray]:
    """Keyframes and times of a morph that holds, morphs, holds and returns, for ``animate_geometry_keyframes``.

    Parameters
    ----------
    snapshots : Cartogram or sequence
        A cartogram (its saved snapshots are used) or the geometry states of the morph, in order.
    hold_start, morph, hold_end, back : float
        Seconds of the hold on the first state, of the morph, of the hold on the last state and of the return to the
        first state. ``back=0`` leaves out the return (a morph that plays forward only).
    easing : {"cosine", "linear"}
        ``"cosine"`` plays the morph slowly, fast, slowly; ``"linear"`` at constant speed.

    Returns
    -------
    keyframes : list
        The states, repeated for the holds.
    times : np.ndarray
        The time in seconds of every keyframe, non-decreasing.

    Examples
    --------
    >>> keyframes, times = morph_timeline(cartogram)
    >>> anim = animate_geometry_keyframes(
    ...     keyframes, keyframe_times=times, position_mapper=linear_over("time"), duration=times[-1]
    ... )
    """
    if hasattr(snapshots, "snapshots") and hasattr(snapshots, "to_geodataframe"):
        frames = [snapshots.to_geodataframe(iteration=snap.iteration) for snap in snapshots.snapshots]
    else:
        frames = list(snapshots)
    n = len(frames)
    if n < 2:
        raise ValueError("morph_timeline needs at least 2 states")
    if easing == "cosine":

        def ease(p: float) -> float:
            return float(np.arccos(np.clip(1.0 - 2.0 * p, -1.0, 1.0)) / np.pi)

    elif easing == "linear":

        def ease(p: float) -> float:
            return p

    else:
        raise ValueError(f"easing must be 'cosine' or 'linear', got {easing!r}")
    keyframes = [frames[0]]
    times = [0.0]
    for i, frame in enumerate(frames):
        keyframes.append(frame)
        times.append(hold_start + morph * ease(i / (n - 1)))
    keyframes.append(frames[-1])
    times.append(hold_start + morph + hold_end)
    if back > 0:
        for i in range(n - 2, -1, -1):
            keyframes.append(frames[i])
            times.append(hold_start + morph + hold_end + back * ease(1.0 - i / (n - 1)))
    return keyframes, np.asarray(times, dtype=float)


# Default position mapper for keyframes (index-linear)
_default_keyframe_position_mapper = linear_over("index")


@deprecated_params(
    facecolor=("column", "color_values"),
    legend="colorbar",
    legend_label="colorbar_label",
    legend_kwargs="colorbar_kwargs",
)
def animate_geometry_keyframes(
    keyframes: list[Any],
    duration: float = 5.0,
    fps: int = 15,
    interpolation: str = "linear",
    position_mapper: PositionMapper | None = None,
    keyframe_times: np.ndarray | None = None,
    hold_frames: int = 0,
    facecolor: str | list[np.ndarray] | np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: Any = None,
    legend: bool = False,
    legend_label: str | None = None,
    show_axes: bool = True,
    legend_kwargs: dict | None = None,
    title: str | bool | Callable[..., str] | None = None,
    precompute: bool = True,
    figsize: tuple[float, float] = (10, 8),
    norm: Any = None,
    center: float | None = None,
    style: Any = None,
    **kwargs: Any,
) -> FuncAnimation:
    """Animate between arbitrary geometry states (keyframes).

    Parameters
    ----------
    keyframes : list
        List of geometry sources. Each can be:
        - GeoDataFrame
        - Cartogram (uses latest snapshot's geometry)
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
    facecolor : str, list of arrays, or array, optional
        Fill of the geometries:

        - a column name: values are read from each keyframe's GeoDataFrame and
          color limits are computed globally.
        - a color (``"red"``, ``"#d4e6f1"``): constant fill.
        - a list of arrays: per-keyframe color values, each with one value per
          geometry. Color values are interpolated when using linear geometry
          interpolation.
        - an array: one array used for every keyframe.

        A string that is a column and also a color is read as the column, and a
        warning is issued.
    vmin : float, optional
        Minimum value for color scale. If None, computed from all values.
    vmax : float, optional
        Maximum value for color scale. If None, computed from all values.
    cmap : str or Colormap, optional
        Colormap of the colors. Defaults to the palette of the style for the data role, else ``"viridis"``.
    legend : bool, default=False
        Whether to show a colorbar.
    legend_label : str, optional
        Label for the colorbar. Defaults to the column name if ``facecolor`` is a column.
    show_axes : bool, default=True
        If False, hides the axes frame, ticks, and labels (``ax.axis("off")``).
        Useful for clean map-style animations.
    legend_kwargs : dict, optional
        Additional keyword arguments passed to ``fig.colorbar()``.
        Example: ``dict(shrink=0.8, pad=0.02)``.
        The ``"label"`` key overrides the auto-set label.
    title : str, bool, callable, or None, default=None
        Controls the axes title each frame.

        - ``None``: auto-title ("Keyframe X/N" or transition label).
        - ``False`` or ``""``: no title.
        - ``str``: format-string expanded with ``str.format_map(ctx)`` each
          frame. Available keys: ``{keyframe}`` (1-based), ``{n_keyframes}``,
          ``{t}`` (inter-keyframe fraction in [0, 1]).
        - callable: called each frame as ``title(keyframe, n_keyframes, t)``
          where ``keyframe`` is 1-based, ``n_keyframes`` is total keyframe
          count, and ``t`` is the inter-keyframe fraction in [0, 1]. Must
          return a string.
    precompute : bool, default=True
        If True, all frames are computed before the animation starts.
        Eliminates per-frame geometry interpolation and GeoDataFrame copying,
        giving a significant speedup especially for ``interpolation="linear"``.
        Set to False to reduce peak memory usage for very long animations.
    figsize : tuple, default=(10, 8)
        Figure size (width, height)
    **kwargs
        Additional arguments passed to GeoDataFrame.plot()

    norm : Normalize, {"linear", "log", "auto"}, optional
        Normalization of the colors: an instance, or the scale of the style's normalization. A ``vmin`` or ``vmax``
        of the call fixes a linear normalization.
    center : float, optional
        Treat the colors as diverging around this value.
    style : PlotStyle or str, optional
        Style of the animation (default: the style of the block or session, else ``"default"``).
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
    ...     facecolor=[pop_2020, pop_2021, pop_2022],
    ...     interpolation="linear",
    ...     cmap="plasma",
    ...     legend=True,
    ...     legend_label="Population"
    ... )

    >>> # Use column from GeoDataFrames with consistent limits
    >>> anim = animate_geometry_keyframes(
    ...     keyframes=[gdf1, gdf2, gdf3],
    ...     facecolor="value",
    ...     cmap="RdYlGn",
    ... )

    >>> # Linear over custom times
    >>> from carto_flow.flow_cartogram.animation import linear_over
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

    column: str | None = None
    color_values: list[np.ndarray] | None = None
    if isinstance(facecolor, str):
        kind = classify_string(facecolor, {c for gdf in gdfs for c in gdf.columns}, (), stacklevel=3)
        if kind == "column":
            column = facecolor
        else:
            kwargs.setdefault("facecolor", facecolor)
    elif isinstance(facecolor, np.ndarray):
        color_values = [facecolor] * n_keyframes
    elif facecolor is not None:
        color_values = list(facecolor)

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
    if legend_label is None and column is not None:
        legend_label = column

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
    st = resolve_style(style)
    _role = "diverging" if center is not None else "sequential"
    cmap = cmap if cmap is not None else (st.colormap(_role) or "viridis")
    _norm_obj = (
        st.resolve_norm(norm, _role, np.array([color_vmin, color_vmax]), center=center, limits=(vmin, vmax))
        or Normalize(vmin=color_vmin, vmax=color_vmax)
        if color_vmin is not None and color_vmax is not None
        else None
    )
    with st.rc_context():
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
        if not show_axes:
            ax.axis("off")
        return []

    def get_color_values_for_keyframe(idx: int) -> np.ndarray | None:
        """Get color values for a specific keyframe."""
        if color_values is not None:
            return color_values[idx]
        elif column is not None and column in gdfs[idx].columns:
            return gdfs[idx][column].values
        return None

    def get_frame_data_position_mapper(frame: int) -> tuple["GeoDataFrame", int, float, np.ndarray | None]:
        """Get the GeoDataFrame and colors using position mapper approach."""
        # Map frame to progress [0, 1]
        progress = frame / max(1, actual_n_frames - 1)

        # Map progress to position in keyframe space [0, n_keyframes-1]
        position = mapper(progress, variables_arrays)  # type: ignore[misc]
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

    def get_frame_data_legacy(frame: int) -> tuple["GeoDataFrame", int, float, np.ndarray | None]:
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

    # Determine if the fast PatchCollection path can be used.
    # Requires: precompute, active coloring, position-mapper mode, and all-polygon geometry.
    # When eligible, the precompute loop bypasses GeoDataFrame/shapely entirely.
    _use_patch_coll = False
    _patch_coll = None
    if precompute and use_coloring and color_vmin is not None and use_position_mapper:
        from shapely.geometry import MultiPolygon, Polygon

        if all(isinstance(g, (Polygon, MultiPolygon)) for g in gdfs[0].geometry):
            _use_patch_coll = True

    # Pre-compute all frames
    _precomputed: list = []  # used by slow path
    _precomputed_paths: list[list] = []  # used by fast path
    _precomputed_meta: list[tuple] = []  # used by fast path
    _precomputed_colors: list = []  # used by fast path

    if _use_patch_coll:
        # Fast path: extract rings from each keyframe once, then interpolate
        # with pure numpy — no shapely Polygon construction, no GeoDataFrame copying.
        def _pos_fn(frame: int) -> tuple[int, int, float]:
            progress = frame / max(1, actual_n_frames - 1)
            position = float(np.clip(mapper(progress, variables_arrays), 0, n_keyframes - 1))  # type: ignore[misc]
            idx_a = int(position)
            return idx_a, min(idx_a + 1, n_keyframes - 1), position - idx_a

        _kf_colors = [get_color_values_for_keyframe(i) for i in range(n_keyframes)]
        _precomputed_paths, _precomputed_colors, _precomputed_meta = _fast_precompute_paths(
            gdfs, _kf_colors, actual_n_frames, n_keyframes, _pos_fn, interpolation
        )
        _norm = _norm_obj
        _coll_kwargs = {k: v for k, v in kwargs.items() if k not in ("markersize", "marker")}
        _patch_coll = _create_patch_collection(
            _precomputed_paths[0],
            _precomputed_colors[0],
            _norm,
            cmap,
            _coll_kwargs,
            legend,
            legend_label,
            legend_kwargs,
            ax,
            fig,
            cbar_added,
            style=st,
        )

    elif precompute:
        # Slow path: existing GeoDataFrame-based precomputation (unchanged)
        for _f in range(actual_n_frames):
            _gdf, _keyframe_idx, _t, _colors = get_frame_data(_f)
            if use_coloring and _colors is not None and color_vmin is not None:
                _plot_gdf = _gdf.assign(_plot_color=_colors)
                _has_colors = True
            elif use_coloring and column is not None and column in _gdf.columns and color_vmin is not None:
                _plot_gdf = _gdf
                _has_colors = False  # use column directly
            else:
                _plot_gdf = _gdf
                _has_colors = False
            _precomputed.append((_plot_gdf, _keyframe_idx, _t, _has_colors))

    def update(frame):
        if _use_patch_coll:
            if _patch_coll is None:
                return []
            keyframe_idx, t = _precomputed_meta[frame]
            # Bypass PatchCollection.set_paths (which expects Patch objects) and
            # call the base Collection.set_paths directly with Path objects.
            from matplotlib.collections import Collection as _Collection

            _Collection.set_paths(_patch_coll, _precomputed_paths[frame])
            _colors = _precomputed_colors[frame]
            if _colors is not None:
                _patch_coll.set_array(np.asarray(_colors).ravel())
        else:
            ax.clear()
            if not show_axes:
                ax.axis("off")

            if precompute:
                plot_gdf, keyframe_idx, t, _has_colors = _precomputed[frame]
            else:
                _gdf, keyframe_idx, t, _colors = get_frame_data(frame)
                if use_coloring and _colors is not None and color_vmin is not None:
                    plot_gdf = _gdf.assign(_plot_color=_colors)
                    _has_colors = True
                else:
                    plot_gdf = _gdf
                    _has_colors = False

            # Determine how to plot
            if use_coloring and _has_colors and color_vmin is not None:
                mappable = plot_gdf.plot(ax=ax, column="_plot_color", legend=False, norm=_norm_obj, cmap=cmap, **kwargs)
            elif use_coloring and column is not None and column in plot_gdf.columns and color_vmin is not None:
                mappable = plot_gdf.plot(ax=ax, column=column, legend=False, norm=_norm_obj, cmap=cmap, **kwargs)
            else:
                plot_gdf.plot(ax=ax, **kwargs)
                mappable = None

            # Add colorbar on first frame if requested
            if legend and not cbar_added[0] and mappable is not None and _norm_obj is not None:
                st.add_colorbar(fig, ax, cmap, _norm_obj, legend_label, **(legend_kwargs or {}))
                cbar_added[0] = True

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal")

        if title is None:
            if t == 0.0:
                ax.set_title(f"Keyframe {keyframe_idx + 1}/{n_keyframes}")
            else:
                ax.set_title(f"Transition {keyframe_idx + 1} -> {keyframe_idx + 2} ({t:.0%})")
        elif isinstance(title, str):
            _t_elapsed = frame * duration / max(1, actual_n_frames - 1)
            ctx = _SafeFormatDict(keyframe=keyframe_idx + 1, n_keyframes=n_keyframes, t=_t_elapsed, duration=duration)
            ax.set_title(title.format_map(ctx))
        elif callable(title):
            ax.set_title(title(keyframe_idx + 1, n_keyframes, t))
        else:
            ax.set_title("")

        return [_patch_coll] if _use_patch_coll else []

    anim = FuncAnimation(
        fig,
        rc_wrapped(st, update),
        frames=actual_n_frames,
        init_func=rc_wrapped(st, init),
        blit=False,
        interval=1000 / fps,
    )

    return anim


# =============================================================================
# Field animation functions
# =============================================================================


def animate_fields(
    cartogram: "Cartogram",
    *,
    duration: float = 5.0,
    fps: int = 15,
    interpolation: str = "nearest",
    position_mapper: PositionMapper | None = None,
    bounds: Any | None = None,
    show_density: bool = True,
    show_velocity: bool = False,
    density: Optional["DensityPlotOptions"] = None,
    velocity: Optional["VelocityPlotOptions"] = None,
    show_axes: bool = True,
    title: str | bool | None = None,
    figsize: tuple[float, float] = (10, 8),
    through: Literal["best", "last"] = "best",
    style: Any = None,
) -> FuncAnimation:
    """Animate density and/or velocity fields over iterations.

    Parameters
    ----------
    cartogram : Cartogram
        Cartogram result with internals (requires save_internals=True)
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
    density : DensityPlotOptions, optional
        Rendering options for the density field. Key fields:

        - ``normalize``: None (absolute), ``"difference"``, or ``"ratio"``.
        - ``clip_percentile`` / ``max_scale``: outlier control for difference mode.
        - ``cmap`` / ``alpha``: colormap and transparency.
        - ``vmin`` / ``vmax``: override the global color-scale limits.
        - ``colorbar_kwargs``: passed to ``fig.colorbar()``.
        - ``imshow_kwargs``: passed to ``ax.imshow()``.
    velocity : VelocityPlotOptions, optional
        Rendering options for the velocity field. Key fields:

        - ``skip``: plot every nth arrow (default 4).
        - ``velocity_scale`` / ``ref_magnitude``: arrow length control.
        - ``color`` / ``alpha``: base arrow color and transparency.
        - ``color_by``: None, ``"magnitude"``, or ``"direction"``.
        - ``colorbar``: whether to show a colorbar (default False in animations).
        - ``colorbar_kwargs``: passed to ``fig.colorbar()``.
        - ``quiver_kwargs``: passed to ``ax.quiver()``.
    show_axes : bool, default=True
        If False, hides the axes frame, ticks, and labels (``ax.axis("off")``).
        Useful for clean map-style animations.
    title : str, bool, or None, default=None
        Controls the axes title each frame.

        - ``None``: auto-title ("Density Field (iteration N)").
        - ``False`` or ``""``: no title.
        - ``str``: format-string expanded each frame. Available keys:
          ``{iteration}``, ``{field}`` (e.g. "Density Field"), ``{t}``.
    figsize : tuple, default=(10, 8)
        Figure size (width, height)
    through : {"best", "last"}, default="best"
        Where the animation ends. ``"best"`` stops at the returned state
        (``best_iteration``); ``"last"`` continues through the iterations run after it.

    style : PlotStyle or str, optional
        Style of the animation (default: the style of the block or session, else ``"default"``).
    Returns
    -------
    FuncAnimation
        Matplotlib animation object

    Examples
    --------
    >>> from carto_flow.flow_cartogram.visualization import DensityPlotOptions, VelocityPlotOptions
    >>> # Density field only (default)
    >>> anim = animate_fields(cartogram, density=DensityPlotOptions(normalize="ratio"))

    >>> # Velocity field only
    >>> anim = animate_fields(cartogram, show_density=False, show_velocity=True)

    >>> # Combined: density with velocity overlay
    >>> anim = animate_fields(
    ...     cartogram, show_velocity=True,
    ...     density=DensityPlotOptions(normalize="ratio"),
    ...     velocity=VelocityPlotOptions(color="black", alpha_by_magnitude=True),
    ... )
    """
    from .visualization import (
        DensityPlotOptions,
        VelocityPlotOptions,
        _compute_quiver_scale,
        _prepare_density_display,
        _prepare_velocity_colors,
        _render_density_on_ax,
        _resolve_bounds,
        _with_arrow_color,
    )

    density_opts = density or DensityPlotOptions()
    velocity_opts = _with_arrow_color(velocity or VelocityPlotOptions(legend=False), style, "white")

    if interpolation not in ("nearest", "linear"):
        raise ValueError(f"interpolation must be 'nearest' or 'linear', got {interpolation!r}")

    if not show_density and not show_velocity:
        raise ValueError("At least one of show_density or show_velocity must be True")

    # Validate inputs
    if not hasattr(cartogram, "internals") or cartogram.internals is None:
        raise ValueError("cartogram.internals is required. Run morphing with save_internals=True.")

    if cartogram.grid is None:
        raise ValueError("cartogram.grid is required for field animation")

    if through not in ("best", "last"):
        raise ValueError(f"through must be 'best' or 'last', got {through!r}")
    snapshots = cartogram.internals.through_best() if through == "best" else cartogram.internals.snapshots
    if not snapshots:
        raise ValueError("cartogram.internals contains no snapshots")

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

    grid = cartogram.grid
    n_snapshots = len(valid_snapshots)
    n_frames = int(duration * fps)

    # Build variables dict from snapshots
    iterations = [s.iteration for s in valid_snapshots]
    variables: dict[str, list] = {
        "iteration": iterations,
        "mean_error": [],
        "max_error": [],
    }

    # Get errors if available (from snapshots), interpolated to match internals iterations
    if hasattr(cartogram, "snapshots") and cartogram.snapshots is not None:
        history_iterations = cartogram.snapshots.get_variable_history("iteration")

        # Extract errors from MorphErrors objects in snapshots
        mean_errors: list[float | None] = []
        max_errors: list[float | None] = []
        for snap in cartogram.snapshots.snapshots:
            if snap.errors is not None:
                mean_errors.append(snap.errors.mean_log_error)
                max_errors.append(snap.errors.max_log_error)
            else:
                mean_errors.append(None)
                max_errors.append(None)

        if history_iterations and mean_errors:
            valid_pairs = [
                (it, err)
                for it, err in zip(history_iterations, mean_errors, strict=False)
                if it is not None and err is not None
            ]
            if valid_pairs:
                hist_iters = np.array([p[0] for p in valid_pairs])
                hist_errors = np.array([p[1] for p in valid_pairs])
                variables["mean_error"] = list(np.interp(iterations, hist_iters, hist_errors))

        if history_iterations and max_errors:
            valid_pairs = [
                (it, err)
                for it, err in zip(history_iterations, max_errors, strict=False)
                if it is not None and err is not None
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
    st = resolve_style(style)
    with st.rc_context():
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    extent = [grid.xmin, grid.xmax, grid.ymin, grid.ymax]

    # Get target density for normalization
    target_density = cartogram.target_density

    # Compute global density ranges using _prepare_density_display per snapshot
    density_vmin, density_vmax = None, None
    if show_density:
        if density_opts.normalize is not None:
            per_snap = [
                _prepare_density_display(
                    snap.rho,  # type: ignore[arg-type]
                    target_density,
                    density_opts.normalize,
                    clip_percentile=density_opts.clip_percentile,
                    max_scale=density_opts.max_scale,
                )
                for snap in valid_snapshots
            ]
            vmins = [r[1] for r in per_snap if r[1] is not None]
            vmaxes = [r[2] for r in per_snap if r[2] is not None]
            density_vmin = min(vmins) if vmins else -1.0
            density_vmax = max(vmaxes) if vmaxes else 1.0
        else:
            all_rho = [snap.rho for snap in valid_snapshots]
            density_vmin = min(r.min() for r in all_rho)  # type: ignore[union-attr]
            density_vmax = max(r.max() for r in all_rho)  # type: ignore[union-attr]

        # opts.vmin/vmax take precedence over computed global limits
        if density_opts.vmin is not None:
            density_vmin = density_opts.vmin
        if density_opts.vmax is not None:
            density_vmax = density_opts.vmax

    # Compute global ranges for velocity
    mag_min, mag_max = 0.0, 1.0
    X_sub, Y_sub = None, None
    if show_velocity:
        X, Y = grid.X, grid.Y
        X_sub = X[:: velocity_opts.skip, :: velocity_opts.skip]
        Y_sub = Y[:: velocity_opts.skip, :: velocity_opts.skip]

        all_mags = []
        for int_snap in valid_snapshots:
            vx_sub = int_snap.vx[:: velocity_opts.skip, :: velocity_opts.skip]  # type: ignore[index]
            vy_sub = int_snap.vy[:: velocity_opts.skip, :: velocity_opts.skip]  # type: ignore[index]
            mag = np.sqrt(vx_sub**2 + vy_sub**2)
            all_mags.extend([mag.min(), mag.max()])
        mag_min, mag_max = min(all_mags), max(all_mags)

        quiver_kw = {
            **velocity_opts.quiver_kwargs,
            **_compute_quiver_scale(
                mag_max,
                velocity_opts.skip,
                grid.dx,
                velocity_opts.velocity_scale,
                velocity_opts.ref_magnitude,
            ),
        }

    from matplotlib.colors import to_rgba

    to_rgba(velocity_opts.color or "white", alpha=velocity_opts.alpha)
    density_cbar_added = [False]
    velocity_cbar_added = [False]

    def init():
        ax.set_aspect("equal")
        return []

    def get_frame_data(frame: int) -> tuple[Any, Any, Any, float, float]:
        """Get fields for a frame. Returns (rho, vx, vy, iteration, t)."""
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
            return rho, vx, vy, iters[idx], 0.0

        else:  # linear
            if idx_before == idx_after or t <= 0:
                snap = valid_snapshots[idx_before]
                rho = snap.rho if show_density else None
                vx = snap.vx if show_velocity else None
                vy = snap.vy if show_velocity else None
                return rho, vx, vy, iters[idx_before], 0.0
            elif t >= 1:
                snap = valid_snapshots[idx_after]
                rho = snap.rho if show_density else None
                vx = snap.vx if show_velocity else None
                vy = snap.vy if show_velocity else None
                return rho, vx, vy, iters[idx_after], 0.0
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
                return rho, vx, vy, iteration, t

    def update(frame: int) -> list:
        ax.clear()
        if not show_axes:
            ax.axis("off")
        rho, vx, vy, iteration, _t_val = get_frame_data(frame)

        # Draw density field
        if show_density and rho is not None:
            im, label = _render_density_on_ax(
                ax,
                rho,
                target_density,
                extent,
                density_opts,
                override_vmin=density_vmin,
                override_vmax=density_vmax,
            )
            if not density_cbar_added[0]:
                _cbar_kw = {"label": label, **density_opts.legend_kwargs}
                fig.colorbar(im, ax=ax, **_cbar_kw)
                density_cbar_added[0] = True

        # Draw velocity field
        if show_velocity and vx is not None and vy is not None:
            vx_sub = vx[:: velocity_opts.skip, :: velocity_opts.skip]
            vy_sub = vy[:: velocity_opts.skip, :: velocity_opts.skip]

            colors, _colorbar_label, _vcmap = _prepare_velocity_colors(
                vx_sub,
                vy_sub,
                color_by=velocity_opts.color_by,
                alpha_by_magnitude=velocity_opts.alpha_by_magnitude,
                alpha_range=velocity_opts.alpha_range,
                cmap=velocity_opts.cmap,
                base_color=velocity_opts.color or "white",
                base_alpha=velocity_opts.alpha,
                mag_range=(mag_min, mag_max),
            )

            ax.quiver(X_sub, Y_sub, vx_sub, vy_sub, color=colors, **quiver_kw)

            if velocity_opts.legend and not velocity_cbar_added[0] and velocity_opts.color_by is not None:
                sm = plt.cm.ScalarMappable(cmap=_vcmap, norm=plt.Normalize(vmin=0, vmax=1))
                if velocity_opts.color_by == "magnitude":
                    sm.set_clim(mag_min, mag_max)
                elif velocity_opts.color_by == "direction":
                    sm.set_clim(-np.pi, np.pi)
                sm.set_array([])
                _cbar_kw = {"label": _colorbar_label, **velocity_opts.legend_kwargs}
                fig.colorbar(sm, ax=ax, **_cbar_kw)
                velocity_cbar_added[0] = True

        ax.set_aspect("equal")

        # Title based on what we're showing
        if show_density and show_velocity:
            _field_label = "Density + Velocity"
        elif show_density:
            _field_label = "Density Field"
        else:
            _field_label = "Velocity Field"

        if title is None:
            if isinstance(iteration, float) and not float(iteration).is_integer():
                ax.set_title(f"{_field_label} (iteration {iteration:.1f})")
            else:
                ax.set_title(f"{_field_label} (iteration {int(iteration)})")
        elif isinstance(title, str):
            _n_iters = valid_snapshots[-1].iteration if valid_snapshots else float("nan")
            _t_elapsed = frame * duration / max(1, n_frames - 1)
            ctx = _SafeFormatDict(
                iteration=iteration, n_iterations=_n_iters, field=_field_label, t=_t_elapsed, duration=duration
            )
            ax.set_title(title.format_map(ctx))
        else:
            ax.set_title("")
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
        rc_wrapped(st, update),
        frames=n_frames,
        init_func=rc_wrapped(st, init),
        blit=False,
        interval=1000 / fps,
    )

    return anim


def animate_density_field(
    cartogram: "Cartogram",
    *,
    duration: float = 5.0,
    fps: int = 15,
    interpolation: str = "nearest",
    position_mapper: PositionMapper | None = None,
    bounds: Any | None = None,
    density: Optional["DensityPlotOptions"] = None,
    show_axes: bool = True,
    title: str | bool | None = None,
    figsize: tuple[float, float] = (10, 8),
    style: Any = None,
) -> FuncAnimation:
    """Animate density field evolution. Convenience wrapper for animate_fields.

    See animate_fields for full documentation.
    """
    return animate_fields(
        cartogram,
        duration=duration,
        fps=fps,
        interpolation=interpolation,
        position_mapper=position_mapper,
        bounds=bounds,
        show_density=True,
        show_velocity=False,
        density=density,
        show_axes=show_axes,
        title=title,
        figsize=figsize,
        style=style,
    )


def animate_velocity_field(
    cartogram: "Cartogram",
    *,
    duration: float = 5.0,
    fps: int = 15,
    interpolation: str = "nearest",
    position_mapper: PositionMapper | None = None,
    bounds: Any | None = None,
    velocity: Optional["VelocityPlotOptions"] = None,
    show_axes: bool = True,
    title: str | bool | None = None,
    figsize: tuple[float, float] = (10, 8),
    style: Any = None,
) -> FuncAnimation:
    """Animate velocity field evolution. Convenience wrapper for animate_fields.

    See animate_fields for full documentation.
    """
    return animate_fields(
        cartogram,
        duration=duration,
        fps=fps,
        interpolation=interpolation,
        position_mapper=position_mapper,
        bounds=bounds,
        show_density=False,
        show_velocity=True,
        velocity=velocity,
        show_axes=show_axes,
        title=title,
        figsize=figsize,
        style=style,
    )


# =============================================================================
# Workflow animation functions
# =============================================================================


@deprecated_params(
    facecolor="color_by", legend="colorbar", legend_label="colorbar_label", legend_kwargs="colorbar_kwargs"
)
def animate_workflow(
    workflow: "CartogramWorkflow",
    duration: float = 8.0,
    fps: int = 15,
    interpolation: str = "linear",
    hold_at_keyframes: float = 0.5,
    facecolor: str | np.ndarray | list | None = "errors_pct",
    cmap: Any = None,
    vmin: float | None = None,
    vmax: float | None = None,
    legend: bool = True,
    legend_label: str | None = None,
    show_run_info: bool = True,
    show_axes: bool = True,
    legend_kwargs: dict | None = None,
    title: str | bool | None = None,
    precompute: bool = True,
    figsize: tuple[float, float] = (10, 8),
    norm: Any = None,
    center: float | None = None,
    style: Any = None,
    **kwargs: Any,
) -> FuncAnimation:
    """Animate through all cartograms in a workflow.

    Shows smooth transitions from original through all morphing results,
    with optional hold time at each keyframe (final state of each run).

    Parameters
    ----------
    workflow : CartogramWorkflow
        Workflow containing cartogram results.
    duration : float, default=8.0
        Total animation duration in seconds.
    fps : int, default=15
        Frames per second.
    interpolation : str, default="linear"
        Interpolation between keyframes: "nearest" or "linear".
    hold_at_keyframes : float, default=0.5
        Seconds to pause at each cartogram's final state.
    facecolor : str, array, list, or None, default="errors_pct"
        Fill of the geometries. Accepted values:

        - ``"errors_pct"`` (default): per-geometry percentage error from each
          cartogram's ``get_errors().errors_pct``.
        - ``"density"``: per-geometry density from each cartogram's
          ``get_density()``.
        - ``"original"``: static values from the column used to build the
          cartogram (requires ``_value_column`` + ``_source_gdf``; set
          automatically by ``CartogramWorkflow``).
        - A column name: looked up in ``_source_gdf`` for every cartogram.
        - A color (``"red"``, ``"#d4e6f1"``): constant fill.
        - ``np.ndarray``: single array broadcast to every keyframe.
        - ``list`` of arrays: one array per cartogram in the workflow.
        - ``None``: no coloring.

        A string that is a column and also a built-in quantity or a color is read as the
        column, then as the built-in quantity, and a warning is issued. A string that is
        none of these raises ``ValueError``.
    cmap : str or Colormap, optional
        Colormap of the colors. Defaults to the palette of the style for the data role, else ``"RdYlGn_r"``.
    vmin : float, optional
        Minimum value for color scale. If None, computed from all values.
    vmax : float, optional
        Maximum value for color scale. If None, computed from all values.
    legend : bool, default=True
        Whether to show a colorbar.
    legend_label : str, optional
        Label for the colorbar. Auto-set from ``facecolor`` if not provided.
    show_run_info : bool, default=True
        Show "Run X/N" in title during animation.
    show_axes : bool, default=True
        If False, hides the axes frame, ticks, and labels (``ax.axis("off")``).
        Useful for clean map-style animations.
    legend_kwargs : dict, optional
        Additional keyword arguments passed to ``fig.colorbar()``.
        Example: ``dict(shrink=0.8, pad=0.02)``.
        The ``"label"`` key overrides the auto-set label.
    title : str, bool, or None, default=None
        Controls the axes title each frame.

        - ``None``: auto-title (uses ``show_run_info`` logic).
        - ``False`` or ``""``: no title.
        - ``str``: format-string expanded each frame. Available keys:
          ``{run}`` (0-based keyframe index), ``{n_runs}``,
          ``{t}`` (elapsed seconds), ``{duration}`` (total seconds),
          ``{iteration}`` (iteration of the shown state, ``best_iteration``),
          ``{n_iterations}`` (iterations run, summed over the runs), ``{mean_error}``,
          ``{max_error}``, ``{mean_error_pct}``, ``{max_error_pct}``.
    precompute : bool, default=True
        If True, all frames are computed before the animation starts.
        Eliminates per-frame geometry interpolation and GeoDataFrame copying,
        giving a significant speedup especially for ``interpolation="linear"``.
        Set to False to reduce peak memory usage for very long animations.
    figsize : tuple, default=(10, 8)
        Figure size (width, height).
    **kwargs
        Additional arguments passed to GeoDataFrame.plot().

    norm : Normalize, {"linear", "log", "auto"}, optional
        Normalization of the colors: an instance, or the scale of the style's normalization. A ``vmin`` or ``vmax``
        of the call fixes a linear normalization.
    center : float, optional
        Treat the colors as diverging around this value.
    style : PlotStyle or str, optional
        Style of the animation (default: the style of the block or session, else ``"default"``).
    Returns
    -------
    FuncAnimation
        Matplotlib animation object.

    Examples
    --------
    >>> workflow = CartogramWorkflow(gdf, 'population')
    >>> workflow.morph_multiresolution(levels=3)
    >>> anim = animate_workflow(workflow, duration=10.0)
    >>> save_animation(anim, "workflow.gif")

    >>> # Color by density with custom colormap
    >>> anim = animate_workflow(workflow, facecolor="density", cmap="viridis")

    >>> # No coloring, just geometry transitions
    >>> anim = animate_workflow(workflow, facecolor=None)
    """
    import geopandas as gpd

    if interpolation not in ("nearest", "linear"):
        raise ValueError(f"interpolation must be 'nearest' or 'linear', got {interpolation!r}")

    if len(workflow) < 1:
        raise ValueError("Workflow must contain at least one cartogram")

    facecolor, legend_label = _resolve_fill(
        facecolor, legend_label, [getattr(c, "_source_gdf", None) for c in workflow], _FILL_QUANTITIES, kwargs
    )

    # Extract keyframes (final geometry from each cartogram)
    keyframes: list[GeoDataFrame] = []
    color_values: list[np.ndarray | None] = []
    kf_mean_error: list[float] = []
    kf_max_error: list[float] = []
    kf_mean_error_pct: list[float] = []
    kf_max_error_pct: list[float] = []
    kf_iterations: list[float] = []  # iteration of the shown (best) state
    kf_iterations_run: list[float] = []  # iterations run, including those after the best one

    for cartogram in workflow:
        latest = cartogram.best
        if latest is None or latest.geometry is None:
            raise ValueError("Cartogram has no geometry snapshot")

        gdf = gpd.GeoDataFrame(geometry=list(latest.geometry))
        keyframes.append(gdf)

        # Collect per-keyframe error scalars for title format context
        errs = cartogram.get_errors()
        _nan = float("nan")
        kf_mean_error.append(errs.mean_log_error if errs else _nan)
        kf_max_error.append(errs.max_log_error if errs else _nan)
        kf_mean_error_pct.append(errs.mean_error_pct if errs else _nan)
        kf_max_error_pct.append(errs.max_error_pct if errs else _nan)
        _best_iter = cartogram.best_iteration
        kf_iterations.append(float(_best_iter) if _best_iter is not None else _nan)
        kf_iterations_run.append(float(cartogram.niterations) if hasattr(cartogram, "niterations") else _nan)

        # Extract color values based on facecolor
        if isinstance(facecolor, str):
            if facecolor == "errors_pct":
                _errs = cartogram.get_errors()
                color_values.append(_errs.errors_pct if _errs else None)
            elif facecolor == "density":
                color_values.append(cartogram.get_density())
            elif facecolor == "original":
                _col = getattr(cartogram, "_value_column", None)
                _src = getattr(cartogram, "_source_gdf", None)
                if _col and _src is not None and _col in _src.columns:
                    color_values.append(np.asarray(_src[_col].values))
                else:
                    color_values.append(None)
            else:
                _src = getattr(cartogram, "_source_gdf", None)
                if _src is not None and facecolor in _src.columns:
                    color_values.append(np.asarray(_src[facecolor].values))
                else:
                    color_values.append(None)
        elif isinstance(facecolor, np.ndarray):
            color_values.append(np.asarray(facecolor))
        elif isinstance(facecolor, list):
            pass  # handled after loop
        else:
            color_values.append(None)

    n_keyframes = len(keyframes)
    n_frames = int(duration * fps)

    # Handle list facecolor: validate and assign
    if isinstance(facecolor, list):
        if len(facecolor) != n_keyframes:
            raise ValueError(
                f"facecolor list length ({len(facecolor)}) must match number of workflow cartograms ({n_keyframes})"
            )
        color_values = [np.asarray(cv) for cv in facecolor]

    # Raise for arbitrary column not found in any cartogram
    if (
        isinstance(facecolor, str)
        and facecolor not in ("errors_pct", "density", "original")
        and not any(cv is not None for cv in color_values)
    ):
        _available: list = []
        for _c in workflow:
            _src2 = getattr(_c, "_source_gdf", None)
            if _src2 is not None:
                _available = list(_src2.columns)
                break
        raise ValueError(f"Column '{facecolor}' not found in source GeoDataFrame. Available: {_available}")

    # Compute global color limits
    use_coloring = facecolor is not None and any(cv is not None for cv in color_values)
    computed_vmin, computed_vmax = None, None
    if use_coloring:
        valid_colors = [cv for cv in color_values if cv is not None]
        if valid_colors:
            computed_vmin = min(np.nanmin(cv) for cv in valid_colors)
            computed_vmax = max(np.nanmax(cv) for cv in valid_colors)

    color_vmin = vmin if vmin is not None else computed_vmin
    color_vmax = vmax if vmax is not None else computed_vmax

    # Determine colorbar label
    resolved_legend_label = legend_label
    if resolved_legend_label is None and facecolor is not None:
        if facecolor == "errors_pct":
            resolved_legend_label = "Error %"
        elif facecolor == "density":
            resolved_legend_label = "Density"
        elif facecolor == "original":
            for _c in workflow:
                _col2 = getattr(_c, "_value_column", None)
                if _col2:
                    resolved_legend_label = _col2
                    break
        elif isinstance(facecolor, str):
            resolved_legend_label = facecolor

    # Compute timing with hold at keyframes
    # Total time = hold_time * n_keyframes + transition_time * (n_keyframes - 1)
    total_hold_time = hold_at_keyframes * n_keyframes
    if n_keyframes > 1:
        transition_time = (duration - total_hold_time) / (n_keyframes - 1)
        transition_time = max(0.1, transition_time)  # Minimum transition time
    else:
        transition_time = 0.0

    # Build keyframe times for position mapping
    keyframe_times = []
    current_time = 0.0
    for i in range(n_keyframes):
        keyframe_times.append(current_time)
        current_time += hold_at_keyframes
        if i < n_keyframes - 1:
            current_time += transition_time

    # Normalize to [0, 1]
    total_time = current_time
    keyframe_positions = np.array(keyframe_times) / total_time if total_time > 0 else np.zeros(n_keyframes)

    # Set up figure
    st = resolve_style(style)
    _role = (
        "diverging"
        if ((isinstance(facecolor, str) and facecolor == "errors_pct") or center is not None)
        else "sequential"
    )
    cmap = cmap if cmap is not None else (st.colormap(_role) or "RdYlGn_r")
    _norm_obj = (
        st.resolve_norm(norm, _role, np.array([color_vmin, color_vmax]), center=center, limits=(vmin, vmax))
        or Normalize(vmin=color_vmin, vmax=color_vmax)
        if color_vmin is not None and color_vmax is not None
        else None
    )
    with st.rc_context():
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Get consistent bounds from all geometries
    all_bounds = [gdf.total_bounds for gdf in keyframes]
    xmin = min(b[0] for b in all_bounds)
    ymin = min(b[1] for b in all_bounds)
    xmax = max(b[2] for b in all_bounds)
    ymax = max(b[3] for b in all_bounds)

    # Track colorbar (only add once)
    cbar_added = [False]

    def init() -> list:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        if not show_axes:
            ax.axis("off")
        return []

    def get_frame_data(frame: int) -> tuple["GeoDataFrame", int, float, np.ndarray | None]:
        """Get the GeoDataFrame and colors for a given frame."""
        # Map frame to progress [0, 1]
        progress = frame / max(1, n_frames - 1)

        # Find which segment we're in based on keyframe_positions
        keyframe_idx = 0
        t = 0.0

        for i in range(n_keyframes - 1):
            pos_before = keyframe_positions[i]
            pos_after = keyframe_positions[i + 1]

            if progress <= pos_after:
                keyframe_idx = i
                t = (progress - pos_before) / (pos_after - pos_before) if pos_after > pos_before else 0.0
                break
        else:
            keyframe_idx = n_keyframes - 1
            t = 0.0

        # Handle hold period at keyframe start
        hold_fraction = hold_at_keyframes / (hold_at_keyframes + transition_time) if transition_time > 0 else 1.0
        actual_t = 0.0 if t < hold_fraction else (t - hold_fraction) / (1 - hold_fraction) if hold_fraction < 1 else 0.0

        if interpolation == "nearest":
            idx = keyframe_idx if actual_t < 0.5 else min(keyframe_idx + 1, n_keyframes - 1)
            return keyframes[idx], idx, 0.0, color_values[idx]

        else:  # linear interpolation
            if actual_t <= 0 or keyframe_idx >= n_keyframes - 1:
                return keyframes[keyframe_idx], keyframe_idx, 0.0, color_values[keyframe_idx]
            elif actual_t >= 1:
                next_idx = min(keyframe_idx + 1, n_keyframes - 1)
                return keyframes[next_idx], next_idx, 0.0, color_values[next_idx]
            else:
                next_idx = min(keyframe_idx + 1, n_keyframes - 1)
                colors_before = color_values[keyframe_idx]
                colors_after = color_values[next_idx]

                gdf = _interpolate_geodataframe(
                    keyframes[keyframe_idx],
                    keyframes[next_idx],
                    actual_t,
                    color_values_a=colors_before,
                    color_values_b=colors_after,
                    color_column="_color",
                )

                # Get interpolated colors
                colors = gdf["_color"].values if "_color" in gdf.columns else None

                return gdf, keyframe_idx, actual_t, colors

    # Determine if the fast PatchCollection path can be used
    _use_patch_coll = False
    _patch_coll = None
    if precompute and use_coloring and color_vmin is not None:
        from shapely.geometry import MultiPolygon, Polygon

        if all(isinstance(g, (Polygon, MultiPolygon)) for g in keyframes[0].geometry):
            _use_patch_coll = True

    # Pre-compute all frames
    _precomputed: list = []
    _precomputed_paths: list = []
    _precomputed_colors: list = []
    _precomputed_meta: list = []

    if _use_patch_coll:

        def _pos_fn(frame: int) -> tuple[int, int, float]:
            progress = frame / max(1, n_frames - 1)
            kf_idx = 0
            raw_t = 0.0
            for i in range(n_keyframes - 1):
                pos_before = keyframe_positions[i]
                pos_after = keyframe_positions[i + 1]
                if progress <= pos_after:
                    kf_idx = i
                    raw_t = (progress - pos_before) / (pos_after - pos_before) if pos_after > pos_before else 0.0
                    break
            else:
                kf_idx = n_keyframes - 1
                raw_t = 0.0
            hold_fraction = hold_at_keyframes / (hold_at_keyframes + transition_time) if transition_time > 0 else 1.0
            actual_t = (
                0.0
                if raw_t < hold_fraction
                else (raw_t - hold_fraction) / (1 - hold_fraction)
                if hold_fraction < 1
                else 0.0
            )
            return kf_idx, min(kf_idx + 1, n_keyframes - 1), actual_t

        _precomputed_paths, _precomputed_colors, _precomputed_meta = _fast_precompute_paths(
            keyframes, color_values, n_frames, n_keyframes, _pos_fn, interpolation
        )
        _norm = _norm_obj
        _coll_kwargs = {k: v for k, v in kwargs.items() if k not in ("markersize", "marker")}
        _patch_coll = _create_patch_collection(
            _precomputed_paths[0],
            _precomputed_colors[0],
            _norm,
            cmap,
            _coll_kwargs,
            legend,
            resolved_legend_label,
            legend_kwargs,
            ax,
            fig,
            cbar_added,
            style=st,
        )

    elif precompute:
        for _f in range(n_frames):
            _gdf, _keyframe_idx, _t, _colors = get_frame_data(_f)
            if use_coloring and _colors is not None and color_vmin is not None:
                _plot_gdf = _gdf.assign(_plot_color=_colors)
                _has_colors = True
            else:
                _plot_gdf = _gdf
                _has_colors = False
            _precomputed.append((_plot_gdf, _keyframe_idx, _t, _has_colors))

    def _interp_kf(arr: list, i: int, frac: float) -> float:
        if not arr:
            return float("nan")
        if i >= len(arr) - 1:
            return arr[min(i, len(arr) - 1)]
        return arr[i] * (1 - frac) + arr[i + 1] * frac

    def update(frame: int) -> list:
        if _use_patch_coll:
            if _patch_coll is None:
                return []
            keyframe_idx, t = _precomputed_meta[frame]
            from matplotlib.collections import Collection as _Collection

            _Collection.set_paths(_patch_coll, _precomputed_paths[frame])
            if _precomputed_colors[frame] is not None:
                _patch_coll.set_array(np.asarray(_precomputed_colors[frame]).ravel())
        else:
            ax.clear()
            if not show_axes:
                ax.axis("off")

            if precompute:
                plot_gdf, keyframe_idx, t, _has_colors = _precomputed[frame]
            else:
                _gdf, keyframe_idx, t, _colors = get_frame_data(frame)
                if use_coloring and _colors is not None and color_vmin is not None:
                    plot_gdf = _gdf.assign(_plot_color=_colors)
                    _has_colors = True
                else:
                    plot_gdf = _gdf
                    _has_colors = False

            # Determine how to plot
            if use_coloring and _has_colors and color_vmin is not None:
                plot_gdf.plot(ax=ax, column="_plot_color", legend=False, norm=_norm_obj, cmap=cmap, **kwargs)
            else:
                plot_gdf.plot(ax=ax, **kwargs)

            # Add colorbar on first frame if requested
            if legend and not cbar_added[0] and use_coloring and _norm_obj is not None:
                st.add_colorbar(fig, ax, cmap, _norm_obj, resolved_legend_label, **(legend_kwargs or {}))
                cbar_added[0] = True

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal")

        # Build title
        if title is None:
            if show_run_info:
                if t == 0.0:
                    _auto_title = f"Run {keyframe_idx}/{n_keyframes - 1}"
                    if keyframe_idx == 0:
                        _auto_title = "Original"
                else:
                    _auto_title = f"Run {keyframe_idx} → {keyframe_idx + 1} ({t:.0%})"
            else:
                _auto_title = "Cartogram Workflow"
            ax.set_title(_auto_title)
        elif isinstance(title, str):
            _t_elapsed = frame * duration / max(1, n_frames - 1)
            _n_iters = sum(i for i in kf_iterations_run if not np.isnan(i))
            ctx = _SafeFormatDict(
                run=keyframe_idx,
                n_runs=n_keyframes,
                t=_t_elapsed,
                duration=duration,
                iteration=_interp_kf(kf_iterations, keyframe_idx, t),
                n_iterations=_n_iters,
                mean_error=_interp_kf(kf_mean_error, keyframe_idx, t),
                max_error=_interp_kf(kf_max_error, keyframe_idx, t),
                mean_error_pct=_interp_kf(kf_mean_error_pct, keyframe_idx, t),
                max_error_pct=_interp_kf(kf_max_error_pct, keyframe_idx, t),
            )
            ax.set_title(title.format_map(ctx))
        else:
            ax.set_title("")
        return [_patch_coll] if _use_patch_coll else []

    anim = FuncAnimation(
        fig,
        rc_wrapped(st, update),
        frames=n_frames,
        init_func=rc_wrapped(st, init),
        blit=False,
        interval=1000 / fps,
    )

    return anim


def animate_workflow_fields(
    workflow: "CartogramWorkflow",
    *,
    duration: float = 8.0,
    fps: int = 15,
    interpolation: str = "nearest",
    bounds: Any | None = None,
    show_density: bool = True,
    show_velocity: bool = False,
    density: Optional["DensityPlotOptions"] = None,
    velocity: Optional["VelocityPlotOptions"] = None,
    show_axes: bool = True,
    title: str | bool | None = None,
    figsize: tuple[float, float] = (10, 8),
    through: Literal["best", "last"] = "best",
    style: Any = None,
) -> FuncAnimation:
    """Animate density and/or velocity fields across all cartograms in a workflow.

    Shows how the fields evolve through multiple morphing runs. Each run
    displays at its native grid resolution, allowing visualization of
    multi-resolution workflows where grid detail increases progressively.

    Parameters
    ----------
    workflow : CartogramWorkflow
        Workflow containing cartogram results with internals (requires
        save_internals=True in MorphOptions for each run).
    duration : float, default=8.0
        Animation duration in seconds.
    fps : int, default=15
        Frames per second.
    interpolation : str, default="nearest"
        Interpolation method: "nearest" or "linear".
    bounds : str, float, or tuple, optional
        Clip view to specified bounds (same as animate_fields).
    show_density : bool, default=True
        Whether to show density field heatmap.
    show_velocity : bool, default=False
        Whether to show velocity field arrows.
    density : DensityPlotOptions, optional
        Rendering options for the density field. Defaults to
        ``DensityPlotOptions(normalize="ratio")`` for workflow animations.
        See :class:`~carto_flow.flow_cartogram.visualization.DensityPlotOptions`.
    velocity : VelocityPlotOptions, optional
        Rendering options for the velocity field.
        See :class:`~carto_flow.flow_cartogram.visualization.VelocityPlotOptions`.
    show_axes : bool, default=True
        If False, hides the axes frame, ticks, and labels (``ax.axis("off")``).
        Useful for clean map-style animations.
    title : str, bool, or None, default=None
        Controls the axes title each frame.

        - ``None``: auto-title ("Density Field (Run X/N, iter Y)").
        - ``False`` or ``""``: no title.
        - ``str``: format-string expanded each frame. Available keys:
          ``{iteration}``, ``{run}`` (0-based), ``{n_runs}``,
          ``{field}`` (e.g. "Density Field"), ``{t}``.
    figsize : tuple, default=(10, 8)
        Figure size (width, height).
    through : {"best", "last"}, default="best"
        Where the animation ends. ``"best"`` stops at the returned state
        (``best_iteration``); ``"last"`` continues through the iterations run after it.

    style : PlotStyle or str, optional
        Style of the animation (default: the style of the block or session, else ``"default"``).
    Returns
    -------
    FuncAnimation
        Matplotlib animation object.

    Examples
    --------
    >>> from carto_flow.flow_cartogram.visualization import DensityPlotOptions, VelocityPlotOptions
    >>> workflow = CartogramWorkflow(gdf, 'population')
    >>> workflow.morph(options=MorphOptions(save_internals=True))
    >>> workflow.morph(options=MorphOptions(save_internals=True))
    >>> anim = animate_workflow_fields(workflow, density=DensityPlotOptions(normalize="ratio"))
    >>> save_animation(anim, "workflow_fields.gif")

    >>> # Show velocity overlaid on density
    >>> anim = animate_workflow_fields(
    ...     workflow, show_velocity=True,
    ...     density=DensityPlotOptions(normalize="ratio"),
    ...     velocity=VelocityPlotOptions(color="black", alpha_by_magnitude=True),
    ... )
    """
    from .visualization import (
        DensityPlotOptions,
        VelocityPlotOptions,
        _compute_quiver_scale,
        _prepare_density_display,
        _prepare_velocity_colors,
        _render_density_on_ax,
        _resolve_bounds,
        _with_arrow_color,
    )

    # Default: ratio normalization for workflow animations
    density_opts = density if density is not None else DensityPlotOptions(normalize="ratio")
    velocity_opts = _with_arrow_color(velocity or VelocityPlotOptions(legend=False), style, "white")

    if interpolation not in ("nearest", "linear"):
        raise ValueError(f"interpolation must be 'nearest' or 'linear', got {interpolation!r}")

    if not show_density and not show_velocity:
        raise ValueError("At least one of show_density or show_velocity must be True")

    # Collect all internals snapshots with run and grid metadata
    # Format: (run_idx, grid, snapshot, cumulative_iteration)
    all_snapshots: list[tuple[int, Any, Any, int]] = []
    run_boundaries: list[int] = [0]
    cumulative_iter = 0

    # Track global bounds across all grids
    global_xmin, global_ymin = float("inf"), float("inf")
    global_xmax, global_ymax = float("-inf"), float("-inf")

    def has_density(s: Any) -> bool:
        return hasattr(s, "rho") and s.rho is not None

    def has_velocity(s: Any) -> bool:
        return hasattr(s, "vx") and s.vx is not None and hasattr(s, "vy") and s.vy is not None

    for run_idx, cartogram in enumerate(workflow):
        if cartogram.internals is None or len(cartogram.internals.snapshots) == 0:
            continue  # Skip runs without internals

        grid = cartogram.grid
        if grid is None:
            continue

        # Update global bounds
        global_xmin = min(global_xmin, grid.xmin)
        global_ymin = min(global_ymin, grid.ymin)
        global_xmax = max(global_xmax, grid.xmax)
        global_ymax = max(global_ymax, grid.ymax)

        run_snapshots = cartogram.internals.through_best() if through == "best" else cartogram.internals.snapshots
        for snap in run_snapshots:
            # Filter based on what we need to show
            if show_density and show_velocity:
                if not (has_density(snap) and has_velocity(snap)):
                    continue
            elif show_density:
                if not has_density(snap):
                    continue
            else:  # show_velocity only
                if not has_velocity(snap):
                    continue

            all_snapshots.append((run_idx, grid, snap, cumulative_iter + snap.iteration))

        # Update cumulative iteration based on last snapshot
        if run_snapshots:
            cumulative_iter += run_snapshots[-1].iteration

        run_boundaries.append(len(all_snapshots))

    if not all_snapshots:
        raise ValueError(
            "No field snapshots found in workflow. Ensure morphing was run with MorphOptions(save_internals=True)."
        )

    n_snapshots = len(all_snapshots)
    n_frames = int(duration * fps)

    # Build variables dict for position mapping
    iterations = np.array([snap[3] for snap in all_snapshots])
    variables_arrays: dict[str, np.ndarray] = {
        "index": np.arange(n_snapshots),
        "iteration": iterations,
    }

    mapper = _default_position_mapper

    # Compute global density/velocity ranges across all runs
    density_vmin, density_vmax = None, None
    if show_density:
        # Get first cartogram's target_density for normalization
        first_morphed = None
        for cartogram in workflow:
            if cartogram.target_density is not None:
                first_morphed = cartogram
                break

        target_density = first_morphed.target_density if first_morphed else None

        if density_opts.normalize is not None:
            per_snap = [
                _prepare_density_display(
                    snap[2].rho,
                    target_density,
                    density_opts.normalize,
                    clip_percentile=density_opts.clip_percentile,
                    max_scale=density_opts.max_scale,
                )
                for snap in all_snapshots
            ]
            vmins = [r[1] for r in per_snap if r[1] is not None]
            vmaxes = [r[2] for r in per_snap if r[2] is not None]
            density_vmin = min(vmins) if vmins else -1.0
            density_vmax = max(vmaxes) if vmaxes else 1.0
        else:
            all_rho = [snap[2].rho for snap in all_snapshots]
            density_vmin = min(r.min() for r in all_rho)
            density_vmax = max(r.max() for r in all_rho)

        # opts.vmin/vmax take precedence over computed global limits
        if density_opts.vmin is not None:
            density_vmin = density_opts.vmin
        if density_opts.vmax is not None:
            density_vmax = density_opts.vmax

    # Compute global velocity magnitude range
    mag_min, mag_max = 0.0, 1.0
    if show_velocity:
        all_mags = []
        for _, _, snap, _ in all_snapshots:
            vx_sub = snap.vx[:: velocity_opts.skip, :: velocity_opts.skip]
            vy_sub = snap.vy[:: velocity_opts.skip, :: velocity_opts.skip]
            mag = np.sqrt(vx_sub**2 + vy_sub**2)
            all_mags.extend([mag.min(), mag.max()])
        mag_min, mag_max = min(all_mags), max(all_mags)

    from matplotlib.colors import to_rgba

    to_rgba(velocity_opts.color or "white", alpha=velocity_opts.alpha)
    density_cbar_added = [False]
    velocity_cbar_added = [False]

    # Compute n_runs once (used for title format context)
    n_runs = max(snap[0] for snap in all_snapshots) + 1

    # Set up figure
    st = resolve_style(style)
    with st.rc_context():
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    def init() -> list:
        ax.set_aspect("equal")
        return []

    def get_frame_data(frame: int) -> tuple[int, Any, Any, int, float]:
        """Get snapshot data for a frame. Returns (run_idx, grid, snapshot, iteration, t)."""
        progress = frame / max(1, n_frames - 1)
        position = mapper(progress, variables_arrays)
        position = np.clip(position, 0, n_snapshots - 1)

        idx_before = int(position)
        idx_after = min(idx_before + 1, n_snapshots - 1)
        t = position - idx_before

        if interpolation == "nearest":
            idx = idx_before if t < 0.5 else idx_after
            return (*all_snapshots[idx], 0.0)

        else:  # linear
            if idx_before == idx_after or t <= 0:
                return (*all_snapshots[idx_before], 0.0)
            elif t >= 1:
                return (*all_snapshots[idx_after], 0.0)
            else:
                # For simplicity, use nearest for linear mode too
                # (interpolating between different grid sizes is complex)
                idx = idx_before if t < 0.5 else idx_after
                return (*all_snapshots[idx], t)

    def update(frame: int) -> list:
        ax.clear()
        if not show_axes:
            ax.axis("off")
        run_idx, grid, snap, _cum_iter, _t_val = get_frame_data(frame)

        extent = [grid.xmin, grid.xmax, grid.ymin, grid.ymax]

        # Draw density field
        if show_density and snap.rho is not None:
            im, label = _render_density_on_ax(
                ax,
                snap.rho,
                target_density,
                extent,
                density_opts,
                override_vmin=density_vmin,
                override_vmax=density_vmax,
            )
            if not density_cbar_added[0]:
                _cbar_kw = {"label": label, **density_opts.legend_kwargs}
                fig.colorbar(im, ax=ax, **_cbar_kw)
                density_cbar_added[0] = True

        # Draw velocity field
        if show_velocity and snap.vx is not None and snap.vy is not None:
            X, Y = grid.X, grid.Y
            X_sub = X[:: velocity_opts.skip, :: velocity_opts.skip]
            Y_sub = Y[:: velocity_opts.skip, :: velocity_opts.skip]
            vx_sub = snap.vx[:: velocity_opts.skip, :: velocity_opts.skip]
            vy_sub = snap.vy[:: velocity_opts.skip, :: velocity_opts.skip]

            colors, _colorbar_label, _vcmap = _prepare_velocity_colors(
                vx_sub,
                vy_sub,
                color_by=velocity_opts.color_by,
                alpha_by_magnitude=velocity_opts.alpha_by_magnitude,
                alpha_range=velocity_opts.alpha_range,
                cmap=velocity_opts.cmap,
                base_color=velocity_opts.color or "white",
                base_alpha=velocity_opts.alpha,
                mag_range=(mag_min, mag_max),
            )

            # Quiver scale uses per-frame grid.dx (grid changes across runs)
            quiver_kw = {
                **velocity_opts.quiver_kwargs,
                **_compute_quiver_scale(
                    mag_max,
                    velocity_opts.skip,
                    grid.dx,
                    velocity_opts.velocity_scale,
                    velocity_opts.ref_magnitude,
                ),
            }

            ax.quiver(X_sub, Y_sub, vx_sub, vy_sub, color=colors, **quiver_kw)

            if velocity_opts.legend and not velocity_cbar_added[0] and velocity_opts.color_by is not None:
                sm = plt.cm.ScalarMappable(cmap=_vcmap, norm=plt.Normalize(vmin=0, vmax=1))
                if velocity_opts.color_by == "magnitude":
                    sm.set_clim(mag_min, mag_max)
                elif velocity_opts.color_by == "direction":
                    sm.set_clim(-np.pi, np.pi)
                sm.set_array([])
                _cbar_kw = {"label": _colorbar_label, **velocity_opts.legend_kwargs}
                fig.colorbar(sm, ax=ax, **_cbar_kw)
                velocity_cbar_added[0] = True

        ax.set_aspect("equal")

        # Title with run info
        if show_density and show_velocity:
            _field_label = "Density + Velocity"
        elif show_density:
            _field_label = "Density Field"
        else:
            _field_label = "Velocity Field"

        if title is None:
            ax.set_title(f"{_field_label} (Run {run_idx}/{n_runs - 1}, iter {snap.iteration})")
        elif isinstance(title, str):
            _n_iters = all_snapshots[-1][3] if all_snapshots else float("nan")
            _t_elapsed = frame * duration / max(1, n_frames - 1)
            ctx = _SafeFormatDict(
                iteration=snap.iteration,
                n_iterations=_n_iters,
                run=run_idx,
                n_runs=n_runs,
                field=_field_label,
                t=_t_elapsed,
                duration=duration,
            )
            ax.set_title(title.format_map(ctx))
        else:
            ax.set_title("")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Apply bounds clipping
        resolved_bounds = _resolve_bounds(bounds, grid)
        if resolved_bounds is not None:
            bxmin, bymin, bxmax, bymax = resolved_bounds
            ax.set_xlim(bxmin, bxmax)
            ax.set_ylim(bymin, bymax)
        else:
            # Use global bounds for consistent view
            ax.set_xlim(global_xmin, global_xmax)
            ax.set_ylim(global_ymin, global_ymax)

        return []

    anim = FuncAnimation(
        fig,
        rc_wrapped(st, update),
        frames=n_frames,
        init_func=rc_wrapped(st, init),
        blit=False,
        interval=1000 / fps,
    )

    return anim
