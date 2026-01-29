"""
===========================================================================
Core morphing algorithm for cartogram generation
===========================================================================

Low-level cartogram algorithm working directly with shapely geometries.

Main Components
---------------
- morph_geometries: Core morphing function for shapely geometries
- Coordinate format detection and conversion utilities

Example
-------
>>> from carto_flow.shape_morpher import morph_geometries, MorphOptions
>>> from shapely.geometry import Polygon
>>>
>>> polygons = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
>>> values = [100]
>>> options = MorphOptions(n_iter=50)
>>>
>>> result = morph_geometries(polygons, values, options=options)
>>> morphed = result.geometries
"""

from typing import Optional

import numpy as np
import tqdm
from scipy.ndimage import gaussian_filter

from ..optimizations.geometry import (
    reconstruct_geometries,
    unpack_geometries,
)
from .density import compute_density_field_from_geometries
from .displacement import displace_coords_numba
from .history import (
    CartogramInternalsSnapshot,
    CartogramSnapshot,
    History,
)
from .options import MorphOptions, MorphStatus
from .result import MorphResult
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
    str
        Detected format: 'points', 'grid', or 'mesh'

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
        return "points"

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

            return "grid"
        except (ValueError, TypeError):
            pass

    # Check for mesh format: (M, N, 2)
    elif coords_array.ndim == 3 and coords_array.shape[2] == 2:
        # Check for reasonable size
        if coords_array.size > 2e6:  # 1M coordinates * 2 values
            raise ValueError("Mesh format coordinates appear to be too large (>1M points)")
        return "mesh"

    # If we get here, format is unclear
    raise ValueError(
        "Could not determine displacement_coords format. "
        "Expected: (N, 2) points, (X, Y) grid tuple, or (M, N, 2) mesh. "
        f"Got shape: {coords_array.shape if hasattr(coords_array, 'shape') else type(coords)}"
    )


def _convert_displacement_to_input_format(displacement_field, original_coords, detected_format):
    """
    Convert displacement field back to the same format as input coordinates.

    Parameters
    ----------
    displacement_field : (N,2) array
        (Ux, Uy) displacement arrays as (N,2) array
    original_coords : array-like
        Original input coordinates to match format
    detected_format : str
        The detected format of input coordinates

    Returns
    -------
    array-like
        Displacement field in same format as input coordinates
    """
    if detected_format == "points":
        # Input was (N, 2), return (N, 2) displacement
        return displacement_field
    elif detected_format == "grid":
        # Input was (X, Y) tuple, return (X, Y) displacement tuple
        X, Y = original_coords
        return (displacement_field[:, 0].reshape(X.shape), displacement_field[:, 1].reshape(Y.shape))
    elif detected_format == "mesh":
        # Input was (M, N, 2), return (M, N, 2) displacement
        coords_array = np.asarray(original_coords)
        return displacement_field.reshape(coords_array.shape)
    else:
        raise ValueError(f"Unknown format: {detected_format}")


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
        format_type = _detect_coordinate_format(coords)

    if format_type == "points":
        # Already in correct format
        return np.asarray(coords)
    elif format_type == "grid":
        # Convert meshgrid to point coordinates
        X, Y = coords
        return np.column_stack([X.ravel(), Y.ravel()])
    elif format_type == "mesh":
        # Convert mesh format to points
        coords_array = np.asarray(coords)
        return coords_array.reshape(-1, 2)
    else:
        raise ValueError(f"Unknown displacement_coords_format: {format_type}")


# ============================================================================
# Core Algorithm
# ============================================================================


def morph_geometries(
    geometries,
    column_values,
    original_areas=None,
    landmarks=None,
    # New options parameter for simplified API
    options: Optional[MorphOptions] = None,
    # Displacement field computation
    displacement_coords=None,
    previous_displaced_coords=None,
) -> MorphResult:
    """
    Core morphing algorithm working with shapely geometries.

    This function implements the fundamental cartogram algorithm that works
    directly with shapely geometries, making it suitable for users who want
    to work without dataframes.

    Parameters
    ----------
    geometries : List[Geometry]
        Shapely geometries to morph
    column_values : np.ndarray
        Data values driving the morphing (e.g., population values)
    original_areas : np.ndarray, optional
        Original areas for refinement mode. If None, treats as initial morphing
    landmarks : Any, optional
        Optional landmarks GeoDataFrame for tracking reference points
    options : MorphOptions, optional
        Algorithm options (dt, n_iter, density_smooth, etc.)
    displacement_coords : array-like, optional
        Coordinates for displacement field computation in various formats:
        - (N, 2) array for point coordinates
        - (X, Y) tuple for meshgrid coordinates
        - (M, N, 2) array for mesh format
        Format is automatically detected from the coordinate structure.
    previous_displaced_coords : np.ndarray, optional
        Previously displaced coordinates for refinement mode

    Returns
    -------
    result : MorphResult
        Complete morphing results containing:
        - geometries: List of morphed geometries
        - landmarks: List of morphed landmark geometries (if provided)
        - history: List of snapshot data for each saved iteration
        - status: Computation status ("converged", "stalled", or "completed")
        - displacement_field: Displacement field in same format as input coordinates (if displacement_coords provided)
        - displaced_coords: Final displaced coordinates for refinement (if displacement_coords provided)

    Examples
    --------
    >>> from carto_flow.shape_morpher import morph_geometries, MorphResult
    >>> from shapely.geometry import Polygon
    >>>
    >>> # Simple morphing
    >>> polygons = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    >>> values = [100]
    >>>
    >>> # Get complete result object
    >>> result = morph_geometries(polygons, values, options=options)
    >>> morphed = result.geometries
    >>> snapshots = result.history
    >>> print(f"Status: {result.status}")
    >>> print(f"Converged in {len(snapshots)} iterations")

    >>> # Compute displacement field
    >>> # Create regular grid of coordinates for displacement field
    >>> x = np.linspace(0, 100, 50)
    >>> y = np.linspace(0, 80, 40)
    >>> X, Y = np.meshgrid(x, y)
    >>> displacement_coords = np.column_stack([X.ravel(), Y.ravel()])
    >>>
    >>> # Run morphing with displacement field computation (auto-detects format)
    >>> result = morph_geometries(
    ...     polygons, values,
    ...     displacement_coords=displacement_coords,
    ...     options=options
    ... )
    >>>
    >>> # Access displacement field results (same format as input)
    >>> displacement_field = result.displacement_field  # Format matches displacement_coords
    >>> displaced_coords = result.displaced_coords      # For refinement
    >>>
    >>> # Refine with displacement field (format auto-detected)
    >>> refined_result = morph_geometries(
    ...     result.geometries, values,
    ...     displacement_coords=displacement_coords,
    ...     previous_displaced_coords=result.displaced_coords,
    ...     options=options
    ... )
    """

    # --- Handle options ---
    if options is None:
        # Create options from explicit parameters (backward compatibility)
        options = MorphOptions()

    # 1. Setup and validation
    # Handle different input types for geometries
    if hasattr(geometries, "values"):
        # pandas Series or numpy array
        geom_list = geometries.values
    else:
        # List of geometries
        geom_list = geometries

    if len(geom_list) != len(column_values):
        raise ValueError("geometries and column_values must have same length")

    # 2. Compute areas and targets
    current_areas = np.array([g.area for g in geom_list])

    if original_areas is None:
        original_areas = current_areas

    # 3. Compute algorithm inputs
    column_values_array = np.asarray(column_values)
    mean_density = float(np.sum(column_values_array) / np.sum(original_areas))
    target_areas = np.sum(original_areas) * column_values_array / np.sum(column_values_array)

    # 4. Initialize algorithm state
    flat_geoms = unpack_geometries(geom_list)
    # Resolve grid using options
    grid = options.get_grid(options._calculate_bounds_from_geometries(geom_list))
    velocity_computer = VelocityComputerFFTW(grid, Dx=options.Dx, Dy=options.Dy)

    # Handle landmarks
    flat_landmarks_geoms = unpack_geometries(landmarks) if landmarks is not None else None

    # Handle displacement field coordinates
    displacement_coords_original = None
    displacement_coords_current = None
    detected_format = None
    if displacement_coords is not None:
        # Detect and normalize input format to (N, 2) array
        detected_format = _detect_coordinate_format(displacement_coords)
        displacement_coords_original = _normalize_coordinates(displacement_coords)

        if previous_displaced_coords is not None:
            # Validate that displacement coordinates match in size
            original_size = len(displacement_coords_original)
            previous_size = len(previous_displaced_coords)
            if original_size != previous_size:
                raise ValueError(
                    f"displacement_coords and previous_displaced_coords must have the same number of points. "
                    f"Got {original_size} and {previous_size} points respectively."
                )

            # Refinement mode - use previously displaced coordinates
            displacement_coords_current = previous_displaced_coords.copy()
        else:
            # Initial mode - start with original coordinates
            displacement_coords_current = displacement_coords_original.copy()

    # 5. Main algorithm loop with snapshotting
    history = History() if options.save_history else None
    history_internals = History() if options.save_internals else None

    # --- Initial snapshot ---
    if options.save_history:
        current_areas = flat_geoms.compute_areas(use_parallel=True)
        errors = (current_areas - target_areas) / target_areas
        max_error = np.max(np.abs(errors))
        mean_error = np.mean(np.abs(errors))

        snapshot = CartogramSnapshot(
            iteration=0,
            geometry=geom_list,
            area_errors=errors.copy(),
            mean_error=mean_error,
            max_error=max_error,
        )
        history.add_snapshot(snapshot)

    last_mean_error = np.inf
    stalled_acc = 0

    msg = "Morph geometries" if options.progress_message is None else options.progress_message
    pbar = tqdm.trange(options.n_iter, desc=msg, disable=not options.show_progress, miniters=1, mininterval=0)

    for step in pbar:
        # 1. Compute density field
        if (options.recompute_every is not None and step % options.recompute_every == 0) or step == 0:
            # Compute density field directly from geometries (no dataframe dependency)
            current_geoms = reconstruct_geometries(flat_geoms)
            if options.save_internals:
                rho, outside_mask = compute_density_field_from_geometries(
                    current_geoms, column_values, grid, mean_density, options.density_smooth, return_outside_mask=True
                )
            else:
                rho = compute_density_field_from_geometries(
                    current_geoms, column_values, grid, mean_density, options.density_smooth
                )
                outside_mask = None

            # 2. Compute baseline velocity field
            vx, vy = velocity_computer.compute(rho)

            # 3. Velocity modulation
            if options.anisotropy is not None:
                fx, fy = options.anisotropy(grid)
                vx_mod = vx * fx
                vy_mod = vy * fy
            else:
                vx_mod, vy_mod = vx, vy

            # 4. Normalize for stability
            vmax = np.nanmax(np.sqrt(vx_mod**2 + vy_mod**2))
            if vmax > 1e-12:
                vx_mod /= vmax
                vy_mod /= vmax

            # Smooth velocity
            if options.vsmooth is not None and options.vsmooth > 0.0:
                vx_mod = gaussian_filter(vx_mod, sigma=options.vsmooth, mode="reflect")
                vy_mod = gaussian_filter(vy_mod, sigma=options.vsmooth, mode="reflect")

            if options.save_internals:
                snapshot = CartogramInternalsSnapshot(
                    iteration=step + 1,
                    rho=rho.copy(),
                    vx=vx.copy(),
                    vy=vy.copy(),
                    vx_mod=vx_mod.copy() if options.anisotropy is not None else None,
                    vy_mod=vy_mod.copy() if options.anisotropy is not None else None,
                    mean_density=mean_density,
                    outside_mask=outside_mask.copy() if outside_mask is not None else None,
                )
                history_internals.add_snapshot(snapshot)

        # 5. Displace geometries
        max_v = max(np.max(np.abs(vx_mod)), np.max(np.abs(vy_mod)), 1e-8)
        dt_prime = options.dt * min(grid.dx, grid.dy) / max_v

        flat_geoms.coords = displace_coords_numba(
            flat_geoms.coords, grid.x_coords, grid.y_coords, vx_mod, vy_mod, dt_prime, grid.dx, grid.dy
        )
        flat_geoms.invalidate_cache()

        if landmarks is not None:
            flat_landmarks_geoms.coords = displace_coords_numba(
                flat_landmarks_geoms.coords, grid.x_coords, grid.y_coords, vx_mod, vy_mod, dt_prime, grid.dx, grid.dy
            )
            # no need to invalidate cache, because we do not compute the areas

        # Displace displacement field coordinates
        if displacement_coords is not None:
            displacement_coords_current[:] = displace_coords_numba(
                displacement_coords_current, grid.x_coords, grid.y_coords, vx_mod, vy_mod, dt_prime, grid.dx, grid.dy
            )

        # 6. Convergence stats
        current_areas = flat_geoms.compute_areas(use_parallel=True)
        errors = (current_areas - target_areas) / target_areas
        max_error = np.max(np.abs(errors))
        mean_error = np.mean(np.abs(errors))

        converged = mean_error < options.mean_tol and max_error < options.max_tol
        stalled_acc += mean_error > last_mean_error
        stalled = stalled_acc > 5

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

        # 7. Create snapshot data at appropriate iterations
        if options.save_history and (
            options.snapshot_every is None
            or step % options.snapshot_every == 0
            or converged
            or stalled
            or (step + 1) == options.n_iter
        ):
            snapshot_data = CartogramSnapshot(
                iteration=step + 1,
                geometry=reconstruct_geometries(flat_geoms),
                area_errors=errors.copy(),
                mean_error=mean_error,
                max_error=max_error,
            )
            history.add_snapshot(snapshot_data)

        pbar.set_postfix_str(f"max={100 * max_error:.1f}%, mean={100 * mean_error:.1f}% - {status}")

        if converged or stalled:
            pbar.update()
            pbar.close()
            break

    # 8. Return final geometries and snapshot data
    final_geometries = reconstruct_geometries(flat_geoms)

    # Handle landmarks if provided
    if landmarks is not None:
        landmarks = reconstruct_geometries(flat_landmarks_geoms)

    # Compute displacement field results
    displacement_field = None
    displaced_coords_result = None
    if displacement_coords is not None:
        # Compute displacement field (displaced - original)
        displacement_deltas = displacement_coords_current - displacement_coords_original

        # Convert displacement field back to input format
        displacement_field = _convert_displacement_to_input_format(
            displacement_deltas, displacement_coords, detected_format
        )
        # Return displaced coordinates for future refinement (no copy needed)
        displaced_coords_result = displacement_coords_current

    # Extract final error metrics from history
    final_snapshot = history.latest()
    final_mean_error = final_snapshot.mean_error if final_snapshot else None
    final_max_error = final_snapshot.max_error if final_snapshot else None
    final_area_errors = final_snapshot.area_errors if final_snapshot else None
    iterations_completed = final_snapshot.iteration if final_snapshot else None

    # Create result object
    result = MorphResult(
        geometries=final_geometries,
        landmarks=landmarks,
        history=history,
        status=status,
        history_internals=history_internals,
        iterations_completed=iterations_completed,
        final_mean_error=final_mean_error,
        final_max_error=final_max_error,
        final_area_errors=final_area_errors,
        displacement_field=displacement_field,
        displaced_coords=displaced_coords_result,
        grid=grid,
    )
    return result
