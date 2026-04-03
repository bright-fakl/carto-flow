"""
Core morphing algorithm for cartogram generation.

Low-level cartogram algorithm working directly with shapely geometries.

Functions
---------
morph_geometries
    Core morphing function for shapely geometries.

Examples
--------
>>> from carto_flow.flow_cartogram import morph_geometries, MorphOptions
>>> from shapely.geometry import Polygon
>>>
>>> polygons = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
>>> values = [100]
>>> options = MorphOptions(n_iter=50)
>>>
>>> result = morph_geometries(polygons, values, options=options)
>>> morphed = result.snapshots.latest().geometry
"""

import time

import numpy as np
import shapely
import tqdm

from ..geo_utils.geometry import (
    reconstruct_geometries,
    unpack_geometries,
)
from .cartogram import Cartogram
from .density import DensityModulator, compute_density_field_from_geometries
from .displacement import displace_coords_numba, max_abs_velocity, max_velocity_magnitude
from .errors import compute_error_metrics
from .history import (
    CartogramInternalsSnapshot,
    CartogramSnapshot,
    ConvergenceHistory,
    History,
)
from .options import MorphOptions, MorphStatus
from .velocity import VelocityComputerFFTW

__all__ = [
    "morph_geometries",
]


# ============================================================================
# Coordinate Format Utilities
# ============================================================================


def _detect_coordinate_format(coords):
    """
    Automatically detect the format of displacement coordinates.

    Parameters
    ----------
    coords : array-like
        Input coordinates to analyze

    Returns
    -------
    (str, sz)
        Detected format ('points', 'grid', or 'mesh') and size of coordinate set

    Raises
    ------
    ValueError
        If format cannot be determined or is ambiguous
    """
    coords_array = np.asarray(coords)

    # Check for points format: (N, 2)
    if coords_array.ndim == 2 and coords_array.shape[1] == 2:
        # Additional check: ensure values look like coordinates (not too large)
        if np.any(np.abs(coords_array) > 1e10):
            raise ValueError("Coordinates appear to contain unreasonably large values")
        return "points", coords_array.shape

    # Check for grid format: tuple/list of (X, Y) arrays
    elif isinstance(coords, (tuple, list)) and len(coords) == 2:
        try:
            X, Y = coords
            X_array = np.asarray(X)
            Y_array = np.asarray(Y)

            # Check if they have the same shape
            if X_array.shape != Y_array.shape:
                raise ValueError("X and Y arrays in grid format must have the same shape")

            # Check for reasonable dimensions (not too large)
            if X_array.size > 1e6:
                raise ValueError("Grid format coordinates appear to be too large (>1M points)")
        except (ValueError, TypeError):
            pass
        else:
            return "grid", (X_array.shape, Y_array.shape)

    # Check for mesh format: (M, N, 2)
    elif coords_array.ndim == 3 and coords_array.shape[2] == 2:
        # Check for reasonable size
        if coords_array.size > 2e6:  # 1M coordinates * 2 values
            raise ValueError("Mesh format coordinates appear to be too large (>1M points)")
        return "mesh", coords_array.shape

    # If we get here, format is unclear
    raise ValueError(
        "Could not determine displacement_coords format. "
        "Expected: (N, 2) points, (X, Y) grid tuple, or (M, N, 2) mesh. "
        f"Got shape: {coords_array.shape if hasattr(coords_array, 'shape') else type(coords)}"
    )


def _convert_coords_to_input_format(coords, format_type, sz):
    """
    Convert displacement field back to the same format as input coordinates.

    Parameters
    ----------
    displacement_field : (N,2) array
        (Ux, Uy) displacement arrays as (N,2) array
    format_type : str
        The detected format of input coordinates
    sz : tuple
        The size of the original input coordinates (used for reshaping)

    Returns
    -------
    array-like
        Coordinates in same format as input coordinates
    """
    if format_type == "points":
        # Input was (N, 2), return (N, 2) displacement
        return coords
    elif format_type == "grid":
        # Input was (X, Y) tuple, return (X, Y) displacement tuple
        return (coords[:, 0].reshape(sz[0]), coords[:, 1].reshape(sz[1]))
    elif format_type == "mesh":
        # Input was (M, N, 2), return (M, N, 2) displacement
        return coords.reshape(sz)
    else:
        raise ValueError(f"Unknown format: {format_type}")


def _normalize_coordinates(coords, format_type=None):
    """
    Normalize displacement field coordinates to (N, 2) format.

    Parameters
    ----------
    coords : array-like
        Input coordinates in various formats
    format_type : str, optional
        Format of input coordinates. If None, will be auto-detected:
        - 'points': (N, 2) array of [x, y] coordinates
        - 'grid': Tuple of (X, Y) meshgrid arrays
        - 'mesh': (M, N, 2) array of coordinates

    Returns
    -------
    np.ndarray
        Normalized coordinates as (N, 2) array
    """
    if coords is None:
        return None

    # Auto-detect format if not provided
    if format_type is None:
        format_type, _ = _detect_coordinate_format(coords)

    if format_type == "points":
        # Already in correct format - create copy to prevent in-place modification
        return np.asarray(coords).copy()
    elif format_type == "grid":
        # Convert meshgrid to point coordinates - create copy
        X, Y = coords
        return np.column_stack([X.ravel(), Y.ravel()]).copy()
    elif format_type == "mesh":
        # Convert mesh format to points - create copy
        coords_array = np.asarray(coords)
        return coords_array.reshape(-1, 2).copy()
    else:
        raise ValueError(f"Unknown displacement_coords_format: {format_type}")


# ============================================================================
# Core Algorithm
# ============================================================================


def morph_geometries(
    geometries,
    values,
    target_density=None,
    landmarks=None,
    coords=None,
    options: MorphOptions | None = None,
) -> Cartogram:
    """
    Core morphing algorithm working with shapely geometries.

    This function implements the fundamental cartogram algorithm that works
    directly with shapely geometries, making it suitable for users who want
    to work without dataframes.

    Parameters
    ----------
    geometries : List[Geometry]
        Shapely geometries to morph
    values : np.ndarray
        Data values driving the morphing (e.g., population sizes)
    target_density : float, optional
        Target density for refinement mode. If None, computed as the mean
        density from values and areas of geometries.
    landmarks : Any, optional
        Optional landmarks geometries for tracking reference points
    coords : array-like, optional
        User-defined coordinates, e.g. for displacement field computation, in various formats:
        - (N, 2) array for point coordinates
        - (X, Y) tuple for meshgrid coordinates
        - (M, N, 2) array for mesh format
        Format is automatically detected from the coordinate structure.
    options : MorphOptions, optional
        Algorithm options (dt, n_iter, density_smooth, etc.)

    Returns
    -------
    result : Cartogram
        Complete cartogram result containing:
        - snapshots: History of CartogramSnapshot objects with algorithm state
        - status: MorphStatus enum (CONVERGED, STALLED, COMPLETED, RUNNING, ORIGINAL)
        - niterations: Number of iterations completed
        - duration: Computation time in seconds
        - options: MorphOptions used for computation
        - internals: History of internal state (if save_internals=True)
        - grid: Grid used for computation
        - target_density: Target equilibrium density

        Access final results via result.latest or result.snapshots.latest():
        - .geometry: List of morphed geometries
        - .landmarks: Morphed landmarks (if provided)
        - .coords: Displaced coordinates (if coords provided)
        - .errors: MorphErrors with log and percentage error metrics
        - .density: Current density values

    Examples
    --------
    >>> from carto_flow.flow_cartogram import morph_geometries, MorphOptions
    >>> from shapely.geometry import Polygon
    >>>
    >>> # Simple morphing
    >>> polygons = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    >>> values = [100]
    >>> options = MorphOptions(n_iter=50)
    >>>
    >>> # Get complete result object
    >>> result = morph_geometries(polygons, values, options=options)
    >>> morphed = result.snapshots.latest().geometry
    >>> print(f"Status: {result.status}")
    >>> print(f"Converged in {result.niterations} iterations")

    >>> # Compute displacement field
    >>> # Create regular grid of coordinates for displacement field
    >>> x = np.linspace(0, 100, 50)
    >>> y = np.linspace(0, 80, 40)
    >>> X, Y = np.meshgrid(x, y)
    >>> coords = np.column_stack([X.ravel(), Y.ravel()])
    >>>
    >>> # Run morphing with displacement field computation (auto-detects format)
    >>> result = morph_geometries(
    ...     polygons, values,
    ...     coords=coords,
    ...     options=options
    ... )
    >>>
    >>> # Access displaced coordinates (same format as input)
    >>> final = result.snapshots.latest()
    >>> displaced_coords = final.coords
    >>>
    >>> # Refine with displaced coordinates (format auto-detected)
    >>> refined_result = morph_geometries(
    ...     final.geometry, values,
    ...     coords=final.coords,
    ...     options=options
    ... )
    """

    # --- Start timing ---
    start_time = time.perf_counter()

    # --- Handle options ---
    if options is None:
        options = MorphOptions()

    # 1. Setup and validation
    if len(geometries) != len(values):
        raise ValueError("geometries and values must have same length")

    # 2. Compute areas and targets
    current_areas = np.array([g.area for g in geometries])

    # 3. Compute algorithm inputs
    values_array = np.asarray(values)
    # Compute target density - algorithm uses unscaled (original CRS units), user sees scaled
    if target_density is None:
        # Initial morph: compute from current areas
        unscaled_target_density = float(np.sum(values_array) / np.sum(current_areas))
        target_density = unscaled_target_density / options.area_scale
    else:
        # Refinement: target_density provided in scaled units, derive unscaled for algorithm
        unscaled_target_density = target_density * options.area_scale
    target_areas = values_array / unscaled_target_density

    # PRE-SCALING (Feature 1 — remove this block to disable)
    if options.prescale_components:
        from .prescale import prescale_connected_components

        geometries = list(prescale_connected_components(geometries, values_array, unscaled_target_density))

    # 4. Initialize algorithm state
    flat_geoms = unpack_geometries(geometries)
    # Resolve grid using options
    grid = options.get_grid(options._calculate_bounds_from_geometries(geometries))

    # Resolve parallel settings
    import multiprocessing as _mp

    if options.parallel_fft is True:
        _nfft = max(_mp.cpu_count() - 1, 1)
    elif options.parallel_fft is False or options.parallel_fft == 1:
        _nfft = 1
    else:  # positive int > 1
        _nfft = min(options.parallel_fft, _mp.cpu_count() - 1)

    if options.parallel_density is True:
        _parallel_density = True
        _ndensity = max(_mp.cpu_count() - 1, 1)
    elif options.parallel_density is False or options.parallel_density == 1:
        _parallel_density = False
        _ndensity = 1
    else:  # positive int > 1
        _parallel_density = True
        _ndensity = min(options.parallel_density, _mp.cpu_count() - 1)

    velocity_computer = VelocityComputerFFTW(grid, Dx=options.Dx, Dy=options.Dy, threads=_nfft)

    # Handle landmarks
    flat_landmarks_geoms = unpack_geometries(landmarks) if landmarks is not None else None

    # Handle displacement coordinates
    flat_coords = None
    coords_format = None
    coords_sz = None
    if coords is not None:
        # Detect and normalize input format to (N, 2) array
        coords_format, coords_sz = _detect_coordinate_format(coords)
        flat_coords = _normalize_coordinates(coords, coords_format)

    # 5. Main algorithm loop with snapshotting
    snapshots: History[CartogramSnapshot] = History()
    convergence = ConvergenceHistory(capacity=options.n_iter)
    internals: History[CartogramInternalsSnapshot] | None = History() if options.save_internals else None

    # Pre-compute log2 thresholds from user-specified percentage tolerances
    # User specifies tolerance as percentage (e.g., 0.02 for 2%)
    # Internally we check if |log2(current/target)| < log2(1 + tolerance)
    log2_mean_tol = np.log2(1.0 + options.mean_tol)
    log2_max_tol = np.log2(1.0 + options.max_tol)

    status = MorphStatus.ORIGINAL

    last_mean_error = np.inf
    stalled_acc = 0

    msg = "Morph geometries" if options.progress_message is None else options.progress_message
    pbar = tqdm.trange(
        options.n_iter, desc=msg, disable=not options.show_progress, miniters=20, mininterval=2.0, maxinterval=5.0
    )

    for step in pbar:
        # 1. Compute density field
        if (options.recompute_every is not None and step % options.recompute_every == 0) or step == 0:
            # 1a. Reconstruct Shapely geometries from the current displaced
            # coordinates and compute the density field.
            # from_ragged_array rebuilds all geometries in a single C-level
            # call (~1 ms) vs the Python-loop reconstruct_geometries (~13 ms).
            # Offsets (topology) were computed once in unpack_geometries and
            # remain valid because morphing changes coordinate values only.
            current_geoms = list(
                shapely.from_ragged_array(
                    flat_geoms.ragged_geometry_type,
                    flat_geoms.coords,
                    flat_geoms.ragged_offsets,
                )
            )

            rho, geom_mask = compute_density_field_from_geometries(
                current_geoms,
                values,
                grid,
                unscaled_target_density,
                return_geometry_mask=True,
                use_parallel=_parallel_density,
                n_jobs=_ndensity,
            )
            rho /= options.area_scale  # convert to display units (values / display-area)

            # 1b. Optional density modulation (e.g., border extension)
            if isinstance(options.density_mod, DensityModulator):
                rho = options.density_mod(rho, grid, geom_mask, target_density)

            # 1c. Compute baseline velocity field
            vx, vy = velocity_computer.compute(rho)

            # 1d. Normalize for stability
            vmax = max_velocity_magnitude(vx, vy)
            if vmax > 1e-12:
                vx /= vmax
                vy /= vmax

            if step == 0:
                vmax_initial = vmax

            vmax_scale = vmax / vmax_initial if vmax_initial > 1e-12 else 1.0

            # 1d. Optional velocity modulation + re-normalize
            if options.anisotropy is not None:
                vx, vy = options.anisotropy(vx, vy, grid, geom_mask)
                vmax = max_velocity_magnitude(vx, vy)
                if vmax > 1e-12:
                    vx /= vmax
                    vy /= vmax

            vx *= vmax_scale
            vy *= vmax_scale

            # 1f. Cache internal state if requested
            if options.save_internals and internals is not None:
                snapshot = CartogramInternalsSnapshot(
                    iteration=step + 1,
                    rho=rho.copy(),
                    vx=vx.copy(),
                    vy=vy.copy(),
                    geometry_mask=geom_mask.copy(),
                )
                internals.add_snapshot(snapshot)

        # 2. Displace geometries
        max_v = max(max_abs_velocity(vx, vy), 1e-8)
        dt_prime = options.dt * min(grid.dx, grid.dy) / max_v

        # 2a. Displace geometry coordinates using the velocity field
        flat_geoms.coords = displace_coords_numba(
            flat_geoms.coords,
            grid.x_coords,
            grid.y_coords,
            vx,
            vy,
            dt_prime,
            grid.dx,
            grid.dy,
            use_parallel=False,
        )
        flat_geoms.invalidate_cache()

        # 2b. Displace landmarks if provided (using same velocity field)
        if landmarks is not None and flat_landmarks_geoms is not None:
            flat_landmarks_geoms.coords = displace_coords_numba(
                flat_landmarks_geoms.coords,
                grid.x_coords,
                grid.y_coords,
                vx,
                vy,
                dt_prime,
                grid.dx,
                grid.dy,
                use_parallel=False,
            )
            # no need to invalidate cache, because we do not compute the areas

        # 2c. Displace displacement field coordinates
        if flat_coords is not None:
            flat_coords[:] = displace_coords_numba(
                flat_coords,
                grid.x_coords,
                grid.y_coords,
                vx,
                vy,
                dt_prime,
                grid.dx,
                grid.dy,
                use_parallel=False,
            )

        # 3. Convergence stats
        current_areas = flat_geoms.compute_areas(use_parallel=False)

        # Compute error metrics using the structured MorphErrors object
        error_metrics = compute_error_metrics(current_areas, target_areas)

        # Record scalar error metrics for every iteration (lightweight)
        convergence.add(step + 1, error_metrics)

        # Use structured errors for convergence check
        max_error = error_metrics.max_log_error
        mean_error = error_metrics.mean_log_error

        # Check convergence using log2-converted thresholds
        converged = mean_error < log2_mean_tol and max_error < log2_max_tol
        stalled_acc += mean_error > last_mean_error
        stalled = options.stall_patience is not None and stalled_acc > options.stall_patience

        last_mean_error = mean_error

        status = (
            MorphStatus.CONVERGED
            if converged
            else MorphStatus.STALLED
            if stalled
            else MorphStatus.COMPLETED
            if (step + 1) == options.n_iter
            else MorphStatus.RUNNING
        )

        # 4. Create snapshot data at appropriate iterations
        # We save snapshots at the final iteration (when the algorithm has converged, stalled, or reached the maximum number
        # of iterations), or at user-defined intervals.
        if (
            converged
            or stalled
            or (step + 1) == options.n_iter
            or (options.snapshot_every is not None and step % options.snapshot_every == 0)
        ):
            snapshot_data = CartogramSnapshot(
                iteration=step + 1,
                geometry=reconstruct_geometries(flat_geoms),
                landmarks=reconstruct_geometries(flat_landmarks_geoms) if flat_landmarks_geoms is not None else None,
                coords=_convert_coords_to_input_format(flat_coords, coords_format, coords_sz)
                if flat_coords is not None
                else None,
                errors=error_metrics,
                density=values_array / (current_areas * options.area_scale),
            )
            snapshots.add_snapshot(snapshot_data)

        # 5. Display progress using structured error metrics
        pbar.set_postfix_str(
            f"max={error_metrics.max_error_pct:.1f}%, mean={error_metrics.mean_error_pct:.1f}% - {status.value}"
        )

        if converged or stalled:
            pbar.update()
            pbar.close()
            break

    # Return final geometries and snapshot data
    # Finalize convergence history (convert to arrays, free list memory)
    convergence.finalize()

    # Create result object
    result = Cartogram(
        snapshots=snapshots,
        convergence=convergence,
        status=status,
        niterations=snapshots.latest().iteration,  # type: ignore[union-attr]
        duration=time.perf_counter() - start_time,
        options=options,
        grid=grid,
        target_density=target_density,
        internals=internals,
    )
    return result
