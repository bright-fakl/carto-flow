"""
===========================================================================
Density field computation utilities for cartographic applications
===========================================================================

This module provides functions for computing density fields from geospatial data,
which are essential for flow-based cartography algorithms. The main functionality
includes rasterizing polygon geometries onto regular grids to create density
distributions that drive cartogram deformation processes.

Main components
---------------
- compute_density_field: Core function for rasterizing polygon data to density grids
- Support for various smoothing and background density options
- Integration with shapely geometries and numpy arrays

Example
-------
    >>> from carto_flow.density import compute_density_field
    >>> from carto_flow.grid import Grid
    >>>
    >>> # Set up grid and compute density
    >>> grid = Grid.from_bounds((0, 0, 100, 80), size=100)
    >>> density = compute_density_field(gdf, 'population', grid)
"""

from typing import Any, Optional

import numpy as np
import shapely
from scipy.ndimage import gaussian_filter

# Import grid utilities
from .grid import Grid

# Module-level exports - Public API
__all__ = [
    "compute_density_field",
    "compute_density_field_from_geometries",
]


def compute_density_field(
    gdf: Any, column: str, grid: Grid, mean_density: Optional[float] = None, smooth: Optional[float] = None
) -> np.ndarray:
    """
    Rasterize polygon geometries to a density grid.

    This function computes a density field by rasterizing polygon geometries onto
    a regular grid. Each polygon's density contribution is calculated as its
    attribute value divided by its area.

    Parameters
    ----------
    gdf : Any
        GeoDataFrame-like object containing polygon geometries. Must have:
        - 'geometry' column with polygon geometries
        - Column specified by 'column' parameter with values to use for densities
    column : str
        Name of the column in gdf containing the variable of interest (e.g., 'population').
    grid : Grid
        Grid information object containing X, Y meshgrid arrays and grid properties.
    mean_density : float, optional
        Background density value for areas outside polygons. If None,
        computed as total column sum divided by total geometry area sum.
    smooth : float, optional
        Standard deviation for Gaussian smoothing. If provided, applies
        gaussian_filter with this sigma value and preserves global mean.

    Returns
    -------
    rho : np.ndarray
        2D density array with same shape as grid.X and grid.Y. Contains density
        values (attribute per unit area) at each grid cell center.

    Notes
    -----
    The function:
    - Assigns density = polygon_value / polygon_area to cells inside each polygon
    - Fills exterior cells with mean_density (computed or provided)
    - Optionally applies Gaussian smoothing while preserving the global mean

    Examples
    --------
    >>> import geopandas as gpd
    >>> from carto_flow.density import compute_density_field
    >>> from carto_flow.grid import Grid
    >>>
    >>> # Create sample data
    >>> gdf = gpd.GeoDataFrame({
    ...     'geometry': [polygon1, polygon2],
    ...     'population': [1000, 2000]
    ... })
    >>>
    >>> # Set up grid
    >>> bounds = (0, 0, 10, 10)
    >>> grid = Grid.from_bounds(bounds, size=50)
    >>>
    >>> # Compute density field
    >>> density = compute_density_field(gdf, 'population', grid)
    >>> print(f"Density grid shape: {density.shape}")
    """

    rho = np.zeros_like(grid.X)

    # Assign density based on polygon population / area
    for _idx, row in gdf.iterrows():
        poly = row.geometry
        if poly.is_empty:
            continue

        pop_density = row[column] / poly.area

        # Create a mask of inside points
        # mask = np.array([poly.contains(pt) for pt in points]).reshape(grid.X.shape)
        mask = shapely.contains_xy(poly, grid.X, grid.Y)
        rho[mask] = pop_density

    outside = rho == 0

    if mean_density is None:
        mean_density = float(gdf[column].sum() / gdf.area.sum())

    rho[outside] = mean_density

    if smooth is not None:
        mu_rho = np.mean(rho)
        rho = gaussian_filter(rho, sigma=smooth)
        # preserve the global mean
        rho *= mu_rho / np.mean(rho)

    return rho


# Helper function for density computation from geometries
def compute_density_field_from_geometries(
    geometries, column_values, grid, mean_density=None, smooth=None, return_outside_mask=False
):
    """Compute density field directly from geometries (no dataframe dependency).

    Parameters
    ----------
    geometries : list of shapely geometries
        The geometries to compute density for.
    column_values : array-like
        Values associated with each geometry.
    grid : Grid
        Grid object with X, Y coordinate arrays.
    mean_density : float, optional
        Background density for cells outside geometries.
    smooth : float, optional
        Gaussian smoothing sigma.
    return_outside_mask : bool, default=False
        If True, return tuple (rho, outside_mask) where outside_mask is a boolean
        array indicating cells outside all geometries.

    Returns
    -------
    rho : np.ndarray
        Density field array.
    outside_mask : np.ndarray (only if return_outside_mask=True)
        Boolean array where True indicates cells outside all geometries.
    """
    rho = np.zeros_like(grid.X)

    # Assign density based on geometry values / areas
    for geom, value in zip(geometries, column_values):
        if geom.is_empty:
            continue

        geom_density = value / geom.area

        # Create mask of grid cells inside geometry
        mask = shapely.contains_xy(geom, grid.X, grid.Y)
        rho[mask] = geom_density

    # Fill exterior cells with mean_density
    outside = rho == 0

    if mean_density is None:
        # Compute mean density from provided areas
        total_value = np.sum(column_values)
        total_area = np.sum([g.area for g in geometries])
        mean_density = float(total_value / total_area) if total_area > 0 else 0.0

    rho[outside] = mean_density

    # Apply smoothing if requested
    if smooth is not None:
        mu_rho = np.mean(rho)
        rho = gaussian_filter(rho, sigma=smooth)
        # Preserve the global mean
        rho *= mu_rho / np.mean(rho)

    if return_outside_mask:
        return rho, outside
    return rho
