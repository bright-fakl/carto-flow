"""
Displacement computation utilities for flow-based cartography.

This module provides functions for applying displacement fields to geometric
objects, including polygon displacement with bilinear interpolation and
displacement field upsampling. These utilities are essential for the iterative
deformation process in flow-based cartogram algorithms.

Functions
---------
displace_coords
    Basic coordinate displacement with bilinear interpolation.
displace_coords_numba
    Optimized Numba-compiled version for uniform grids.
apply_displacement_to_polygons_vectorized
    Vectorized polygon displacement.
upsample_displacement
    Upsample displacement fields to higher resolution.

Examples
--------
>>> from carto_flow.flow_cartogram.displacement import displace_coords
>>> from carto_flow.flow_cartogram.grid import Grid
>>> import numpy as np
>>> grid = Grid.from_bounds((0, 0, 100, 80), size=100)
>>> coords = np.array([[10, 20], [30, 40], [50, 60]])
>>> displaced_coords = displace_coords(coords, grid, vx, vy, dt=0.1)
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from scipy.ndimage import gaussian_filter, zoom
from shapely.geometry import MultiPolygon, Polygon

# Import grid utilities
from .grid import Grid

# Module-level exports - Public API
__all__ = [
    "apply_displacement_to_polygons_vectorized",
    "displace_coords",
    "displace_coords_numba",
    "upsample_displacement",
]


def displace_coords(coords: np.ndarray, grid: Grid, vx: np.ndarray, vy: np.ndarray, dt: float) -> np.ndarray:
    """
    Displace coordinates using bilinear interpolation of velocity field.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (n, 2) containing [x, y] coordinates to displace
    grid : Grid
        Grid information containing coordinate arrays and spatial discretization
    vx, vy : np.ndarray
        Velocity field components with shape (ny, nx) matching grid dimensions
    dt : float
        Time step for displacement integration

    Returns
    -------
    coords : np.ndarray
        Displaced coordinates with same shape as input
    """
    # Compute grid indices
    ix = np.clip(np.searchsorted(grid.x_coords, coords[:, 0]) - 1, 0, vx.shape[1] - 2)
    iy = np.clip(np.searchsorted(grid.y_coords, coords[:, 1]) - 1, 0, vy.shape[0] - 2)

    # Bilinear interpolation
    tx = (coords[:, 0] - grid.x_coords[ix]) / (grid.x_coords[ix + 1] - grid.x_coords[ix])
    ty = (coords[:, 1] - grid.y_coords[iy]) / (grid.y_coords[iy + 1] - grid.y_coords[iy])

    vx_interp = (
        (1 - tx) * (1 - ty) * vx[iy, ix]
        + tx * (1 - ty) * vx[iy, ix + 1]
        + (1 - tx) * ty * vx[iy + 1, ix]
        + tx * ty * vx[iy + 1, ix + 1]
    )

    vy_interp = (
        (1 - tx) * (1 - ty) * vy[iy, ix]
        + tx * (1 - ty) * vy[iy, ix + 1]
        + (1 - tx) * ty * vy[iy + 1, ix]
        + tx * ty * vy[iy + 1, ix + 1]
    )

    # Apply displacement
    coords[:, 0] += vx_interp * dt
    coords[:, 1] += vy_interp * dt

    return coords


# Option 1: Numba JIT compilation with uniform grid
@njit(parallel=True, fastmath=True, cache=True)
def displace_coords_numba(
    coords: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
) -> np.ndarray:
    """
    Numba-optimized version with parallel execution for uniform grids
    dx, dy: constant grid spacing
    """
    n_points = coords.shape[0]
    new_coords = coords.copy()

    # Precompute reciprocals (division is slower than multiplication)
    dx_inv = 1.0 / dx
    dy_inv = 1.0 / dy

    x_min = x_coords[0]
    y_min = y_coords[0]
    max_ix = vx.shape[1] - 2
    max_iy = vy.shape[0] - 2

    for i in prange(n_points):  # Parallel loop
        x, y = coords[i, 0], coords[i, 1]

        # Direct index calculation (much faster than searchsorted)
        ix = int((x - x_min) * dx_inv)
        iy = int((y - y_min) * dy_inv)

        # Clipping
        ix = max(0, min(ix, max_ix))
        iy = max(0, min(iy, max_iy))

        # Interpolation weights (simplified with uniform grid)
        tx = (x - x_coords[ix]) * dx_inv
        ty = (y - y_coords[iy]) * dy_inv

        # Precompute weight complements
        tx1 = 1.0 - tx
        ty1 = 1.0 - ty

        # Bilinear interpolation
        w00 = tx1 * ty1
        w10 = tx * ty1
        w01 = tx1 * ty
        w11 = tx * ty

        vx_interp = w00 * vx[iy, ix] + w10 * vx[iy, ix + 1] + w01 * vx[iy + 1, ix] + w11 * vx[iy + 1, ix + 1]

        vy_interp = w00 * vy[iy, ix] + w10 * vy[iy, ix + 1] + w01 * vy[iy + 1, ix] + w11 * vy[iy + 1, ix + 1]

        # Apply displacement
        new_coords[i, 0] += vx_interp * dt
        new_coords[i, 1] += vy_interp * dt

    return new_coords


def apply_displacement_to_polygons_vectorized(
    polygons: list[Polygon | MultiPolygon], vx: np.ndarray, vy: np.ndarray, grid: Grid, dt: float
) -> list[Polygon | MultiPolygon]:
    """
    Apply vectorized displacement to polygon geometries using velocity field interpolation.

    This function efficiently displaces multiple polygon geometries by interpolating
    velocity values from a regular grid and applying the displacement over a time step.
    The algorithm handles both individual polygons and multi-polygons while preserving
    their topological structure.

    The displacement process:
    1. **Flattening**: Convert all polygons to individual Polygon objects
    2. **Coordinate extraction**: Collect all vertex coordinates into single array
    3. **Grid indexing**: Find grid cell indices for each coordinate using binary search
    4. **Bilinear interpolation**: Interpolate velocity values at exact coordinate locations
    5. **Displacement application**: Apply velocity * dt to each coordinate
    6. **Reconstruction**: Rebuild polygons maintaining MultiPolygon structure

    Parameters
    ----------
    polygons : List[Union[Polygon, MultiPolygon]]
        List of shapely polygon geometries to displace. Can contain both
        individual Polygon and MultiPolygon objects.
    vx : np.ndarray
        X-component of velocity field with shape (ny, nx) matching grid dimensions.
        Represents horizontal flow velocities at grid cell centers.
    vy : np.ndarray
        Y-component of velocity field with shape (ny, nx) matching grid dimensions.
        Represents vertical flow velocities at grid cell centers.
    grid : Grid
        Grid information containing coordinate arrays and spatial discretization.
        Must provide x_coords, y_coords arrays for interpolation.
    dt : float
        Time step for displacement integration. Controls the magnitude
        of displacement applied to polygon vertices.

    Returns
    -------
    displaced_polygons : List[Union[Polygon, MultiPolygon]]
        List of displaced polygon geometries with same structure as input.
        Individual Polygons remain as Polygons, MultiPolygons are reconstructed
        from their constituent displaced Polygon components.

    Notes
    -----
    **Performance characteristics**:
    - **Vectorized operations**: O(n) complexity for n polygon vertices
    - **Binary search indexing**: O(log n) per coordinate for grid lookup
    - **Bilinear interpolation**: Smooth velocity field approximation
    - **Memory efficient**: Processes all coordinates in single vectorized operations

    **Boundary handling**:
    - Clips grid indices to valid range to handle edge cases
    - Preserves polygon topology and MultiPolygon structure
    - Maintains coordinate ordering for proper polygon reconstruction

    **Use cases**:
    - Flow-based cartography for dynamic shape deformation
    - Physics-based animation of polygon boundaries
    - Time integration of velocity fields on geometric objects

    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> import numpy as np
    >>> from carto_flow.displacement import apply_displacement_to_polygons_vectorized
    >>> from carto_flow.grid import Grid
    >>>
    >>> # Create sample polygons
    >>> poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    >>> poly2 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
    >>> polygons = [poly1, poly2]
    >>>
    >>> # Set up velocity field (rightward flow)
    >>> bounds = (0, 0, 4, 2)
    >>> grid = Grid.from_bounds(bounds, size=20)
    >>> vx = np.ones((grid.sy, grid.sx)) * 0.1  # Rightward velocity
    >>> vy = np.zeros((grid.sy, grid.sx))       # No vertical velocity
    >>>
    >>> # Apply displacement
    >>> dt = 1.0
    >>> displaced = apply_displacement_to_polygons_vectorized(
    ...     polygons, vx, vy, grid, dt
    ... )
    >>>
    >>> # Polygons have moved right by 0.1 units
    >>> print(f"Original: {polygons[0]}")
    >>> print(f"Displaced: {displaced[0]}")
    """
    # Flatten all polygons into a single list of Polygon objects
    flat_polys = []
    for poly in polygons:
        if isinstance(poly, Polygon):
            flat_polys.append(poly)
        elif isinstance(poly, MultiPolygon):
            flat_polys.extend(poly.geoms)
        else:
            raise TypeError("Geometry must be Polygon or MultiPolygon")

    # Flatten all coordinates
    coord_list: list[np.ndarray] = []
    poly_sizes = []
    for p in flat_polys:
        c = np.array(p.exterior.coords)
        coord_list.append(c)
        poly_sizes.append(len(c))
    coords = np.vstack(coord_list)

    # Apply displacement using optimized Numba version
    coords = displace_coords_numba(coords, grid.x_coords, grid.y_coords, vx, vy, dt, grid.dx, grid.dy)

    # Reconstruct polygons
    displaced_flat_polys = []
    start = 0
    for n in poly_sizes:
        displaced_flat_polys.append(Polygon(coords[start : start + n]))
        start += n

    # Recombine MultiPolygons if needed
    result_polys = []
    idx = 0
    for poly in polygons:
        if isinstance(poly, Polygon):
            result_polys.append(displaced_flat_polys[idx])
            idx += 1
        elif isinstance(poly, MultiPolygon):
            mp_geoms = displaced_flat_polys[idx : idx + len(poly.geoms)]
            result_polys.append(MultiPolygon(mp_geoms))
            idx += len(poly.geoms)

    return result_polys


def upsample_displacement(
    u_field: np.ndarray,
    v_field: np.ndarray,
    new_shape: tuple[int, int],
    order: int = 3,
    sigma_smooth: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Upsample a displacement field (u, v) to a higher-resolution grid.

    The displacement fields are defined on a regular grid, representing
    the displacement of each grid cell center in x and y directions.
    When moving to a higher resolution, we upsample smoothly using
    spline interpolation and optional Gaussian smoothing to avoid
    high-frequency artifacts.

    Parameters
    ----------
    u_field, v_field : np.ndarray
        2D displacement arrays (shape: [ny, nx]).
    new_shape : tuple[int, int]
        Target shape (ny_new, nx_new).
    order : int, default=3
        Spline interpolation order (0=nearest, 1=linear, 3=cubic).
    sigma_smooth : float, default=0.6
        Standard deviation for optional Gaussian smoothing applied after
        interpolation (in pixels). Set to 0 to disable.

    Returns
    -------
    (u_up, v_up) : tuple[np.ndarray, np.ndarray]
        Upsampled displacement fields matching the target resolution.
    """

    zoom_factors = np.array(new_shape, dtype=float) / np.array(u_field.shape, dtype=float)

    # Interpolate to higher resolution
    u_up = zoom(u_field, zoom_factors, order=order, mode="reflect", grid_mode=True)
    v_up = zoom(v_field, zoom_factors, order=order, mode="reflect", grid_mode=True)

    # Optional smoothing to avoid aliasing artifacts
    if sigma_smooth and sigma_smooth > 0:
        u_up = gaussian_filter(u_up, sigma=sigma_smooth)
        v_up = gaussian_filter(v_up, sigma=sigma_smooth)

    return u_up, v_up
