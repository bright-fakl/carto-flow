"""
================================================================================
Grid Construction and Management Utilities for Flow-Based Cartography
================================================================================

This module provides functions for creating and managing spatial grids used
in flow-based cartogram algorithms. It includes utilities for constructing
regular grids, multi-resolution grids, and grid information containers.

Main Components
--------------
Grid : Structured grid information container with lazy computation
    Class providing flexible grid construction with multiple class methods
    and lazy computation with caching of coordinate arrays for optimal performance.

build_multilevel_grids : Multi-resolution grid generator
    Creates dyadically scaled grids for multi-resolution cartogram algorithms.

Examples
--------
Basic grid construction:

>>> from carto_flow.grid import Grid, build_multilevel_grids
>>> grid = Grid.from_bounds((0, 0, 100, 80), size=200)
>>> print(f"Grid shape: {grid.shape}")
Grid shape: (100, 200)

Multi-resolution grid construction:

>>> grids = build_multilevel_grids((0, 0, 100, 80), 256, n_levels=3)
>>> print(f"Number of resolution levels: {len(grids)}")
Number of resolution levels: 3

Notes
-----
All grid functions return Grid objects that provide both array-based
and property-based access to grid coordinates and metadata. The grids are
designed to work seamlessly with FFT-based algorithms and maintain proper
aspect ratios for cartographic applications.
"""

import math
from typing import Optional, Union

import numpy as np

# Module-level exports - Public API
__all__ = [
    "Grid",
    "build_multilevel_grids",
]


class Grid:
    """
    Grid class with lazy computation and caching of coordinate arrays.

    This class provides flexible grid construction with multiple class methods
    and lazy computation with caching of coordinate arrays for optimal performance.

    Construction Methods (Preferred)
    -------------------
    1. From bounds and size:
       Grid.from_bounds(bounds, size, margin=0.0, square=False)

    2. From bounds and spacing:
       Grid.from_bounds_and_spacing(bounds, spacing, margin=0.0, strict=False)

    3. From size and spacing:
       Grid.from_size_and_spacing(size, spacing)

    Parameters (Internal)
    ----------
    xmin : float
        Minimum x-coordinate of the grid bounds.
    ymin : float
        Minimum y-coordinate of the grid bounds.
    xmax : float
        Maximum x-coordinate of the grid bounds.
    ymax : float
        Maximum y-coordinate of the grid bounds.
    sx : int
        Number of grid cells in x-direction (columns).
    sy : int
        Number of grid cells in y-direction (rows).
    dx : float
        Grid cell width in coordinate units.
    dy : float
        Grid cell height in coordinate units.

    Properties
    ----------
    bounds : tuple of float
        Grid bounds as (xmin, ymin, xmax, ymax).
    size : tuple of int
        Grid size as (sx, sy).
    spacing : tuple of float
        Grid spacing as (dx, dy).
    shape : tuple of int
        Grid shape as (sy, sx).

    Examples
    --------
    >>> from carto_flow.grid import Grid
    >>> grid = Grid.from_bounds((0, 0, 10, 5), size=100)
    >>> print(f"Grid shape: {grid.shape}")
    Grid shape: (50, 100)
    >>> print(f"Grid bounds: {grid.bounds}")
    Grid bounds: (0.0, 0.0, 10.0, 5.0)
    >>> x_coords = grid.x_coords  # Computed and cached on first access
    >>> X = grid.X  # Computed and cached on first access

    Notes
    -----
    Use the class methods (from_bounds, from_bounds_and_spacing, from_size_and_spacing)
    for construction rather than the direct constructor to ensure parameter consistency.
    The class uses lazy computation with caching for coordinate arrays.
    Arrays are computed once on first access and then cached for performance.
    """

    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        size: Union[int, tuple[Optional[int], Optional[int]]],
        margin: float = 0.0,
        square: bool = False,
    ):
        """
        Create a Grid from bounds and size specification.

        Parameters
        ----------
        bounds : tuple of float
            Grid bounds as (xmin, ymin, xmax, ymax).
        size : int or tuple
            Grid size specification:
            - int: number of points along longest edge
            - (sx, sy): exact grid dimensions
            - (sx, None): sx points in x, computed sy for similar dx/dy
            - (None, sy): computed sx, sy points in y for similar dx/dy
        margin : float, optional
            Margin to add around bounds as fraction. Default is 0.0.
        square : bool, optional
            If True, adjust bounds to ensure dx == dy. Default is False.
        """
        xmin, ymin, xmax, ymax = bounds

        # Store original data bounds before margin expansion
        self._data_bounds = (xmin, ymin, xmax, ymax)

        # Apply margin
        if margin > 0:
            w = xmax - xmin
            h = ymax - ymin
            xmin -= margin * w
            xmax += margin * w
            ymin -= margin * h
            ymax += margin * h

        w = xmax - xmin
        h = ymax - ymin

        # Handle size specification
        if isinstance(size, int):
            # Scalar size: points along longest edge
            w = xmax - xmin
            h = ymax - ymin

            if w > h:
                sx = size
                sy = max(1, int(h * size / w))
            else:
                sx = max(1, int(w * size / h))
                sy = size
        else:
            # Tuple size specification
            sx, sy = size

            if sx is None and sy is None:
                raise ValueError("At least one dimension must be specified")
            elif sx is None:
                # Only sy specified: compute sx for similar dx/dy
                h = ymax - ymin
                target_dy = h / sy
                w = xmax - xmin
                sx = max(1, int(w / target_dy))
            elif sy is None:
                # Only sx specified: compute sy for similar dx/dy
                w = xmax - xmin
                target_dx = w / sx
                h = ymax - ymin
                sy = max(1, int(h / target_dx))
            # else: both sx and sy specified

        # Compute cell sizes
        dx = (xmax - xmin) / sx
        dy = (ymax - ymin) / sy

        # Handle square requirement
        if square:
            # Adjust bounds to make dx == dy
            avg_spacing = (dx + dy) / 2
            if w > h:
                # Adjust height to match width at average spacing
                new_h = avg_spacing * sy
                ymin = ymax - new_h
            else:
                # Adjust width to match height at average spacing
                new_w = avg_spacing * sx
                xmin = xmax - new_w
            dx = dy = avg_spacing

        # Store computed parameters
        self._xmin = xmin
        self._ymin = ymin
        self._xmax = xmax
        self._ymax = ymax
        self._sx = sx
        self._sy = sy
        self._dx = dx
        self._dy = dy

        # Cache storage for lazy properties
        self._x_coords = None
        self._y_coords = None
        self._x_edges = None
        self._y_edges = None
        self._X = None
        self._Y = None

    @property
    def xmin(self) -> float:
        """Minimum x-coordinate of the grid bounds."""
        return self._xmin

    @property
    def ymin(self) -> float:
        """Minimum y-coordinate of the grid bounds."""
        return self._ymin

    @property
    def xmax(self) -> float:
        """Maximum x-coordinate of the grid bounds."""
        return self._xmax

    @property
    def ymax(self) -> float:
        """Maximum y-coordinate of the grid bounds."""
        return self._ymax

    @property
    def sx(self) -> int:
        """Number of grid cells in x-direction (columns)."""
        return self._sx

    @property
    def sy(self) -> int:
        """Number of grid cells in y-direction (rows)."""
        return self._sy

    @property
    def dx(self) -> float:
        """Grid cell width in coordinate units."""
        return self._dx

    @property
    def dy(self) -> float:
        """Grid cell height in coordinate units."""
        return self._dy

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Grid bounds as (xmin, ymin, xmax, ymax), including any margin."""
        return (self._xmin, self._ymin, self._xmax, self._ymax)

    @property
    def data_bounds(self) -> tuple[float, float, float, float]:
        """Original data bounds as (xmin, ymin, xmax, ymax), without margin."""
        return self._data_bounds

    @property
    def size(self) -> tuple[int, int]:
        """Grid size as (sx, sy)."""
        return (self._sx, self._sy)

    @property
    def spacing(self) -> tuple[float, float]:
        """Grid spacing as (dx, dy)."""
        return (self._dx, self._dy)

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape as (sy, sx)."""
        return (self._sy, self._sx)

    @property
    def x_coords(self) -> np.ndarray:
        """1D array of x-coordinates for grid columns (computed and cached)."""
        if self._x_coords is None:
            self._x_coords = np.linspace(self._xmin + self._dx / 2, self._xmax - self._dx / 2, self._sx)
        return self._x_coords

    @property
    def y_coords(self) -> np.ndarray:
        """1D array of y-coordinates for grid rows (computed and cached)."""
        if self._y_coords is None:
            self._y_coords = np.linspace(self._ymin + self._dy / 2, self._ymax - self._dy / 2, self._sy)
        return self._y_coords

    @property
    def X(self) -> np.ndarray:
        """2D array of x-coordinates in meshgrid format (computed and cached)."""
        if self._X is None:
            self._X = np.meshgrid(self.x_coords, self.y_coords)[0]
        return self._X

    @property
    def Y(self) -> np.ndarray:
        """2D array of y-coordinates in meshgrid format (computed and cached)."""
        if self._Y is None:
            self._Y = np.meshgrid(self.x_coords, self.y_coords)[1]
        return self._Y

    @property
    def x_edges(self) -> np.ndarray:
        """1D array of x-coordinates for cell edges (computed and cached)."""
        if self._x_edges is None:
            self._x_edges = np.linspace(self._xmin, self._xmax, self._sx + 1)
        return self._x_edges

    @property
    def y_edges(self) -> np.ndarray:
        """1D array of y-coordinates for cell edges (computed and cached)."""
        if self._y_edges is None:
            self._y_edges = np.linspace(self._ymin, self._ymax, self._sy + 1)
        return self._y_edges

    @classmethod
    def from_bounds(
        cls,
        bounds: tuple[float, float, float, float],
        size: Union[int, tuple[Optional[int], Optional[int]]],
        margin: float = 0.0,
        square: bool = False,
    ) -> "Grid":
        """
        Create a Grid from bounds and size specification.

        Parameters
        ----------
        bounds : tuple of float
            Grid bounds as (xmin, ymin, xmax, ymax).
        size : int or tuple
            Grid size specification:
            - int: number of points along longest edge
            - (sx, sy): exact grid dimensions
            - (sx, None): sx points in x, computed sy for similar dx/dy
            - (None, sy): computed sx, sy points in y for similar dx/dy
        margin : float, optional
            Margin to add around bounds as fraction. Default is 0.0.
        square : bool, optional
            If True, adjust bounds to ensure dx == dy. Default is False.

        Returns
        -------
        Grid
            New Grid instance.

        Examples
        --------
        >>> bounds = (0, 0, 10, 5)
        >>> grid = Grid.from_bounds(bounds, size=100)  # 100 points along longest edge
        >>> grid = Grid.from_bounds(bounds, size=(50, 25))  # Exact dimensions
        >>> grid = Grid.from_bounds(bounds, size=(50, None))  # 50 in x, computed in y
        """

        return cls(bounds, size, margin, square)

    @classmethod
    def from_bounds_and_spacing(
        cls,
        bounds: tuple[float, float, float, float],
        spacing: Union[float, tuple[float, float]],
        margin: float = 0.0,
        strict: bool = False,
    ) -> "Grid":
        """
        Create a Grid from bounds and spacing specification.

        Parameters
        ----------
        bounds : tuple of float
            Grid bounds as (xmin, ymin, xmax, ymax).
        spacing : float or tuple
            Grid spacing:
            - float: same spacing in both directions
            - (dx, dy): different spacing in each direction
        margin : float, optional
            Margin to add around bounds as fraction. Default is 0.0.
        strict : bool, optional
            If True, adjust bounds to ensure exact spacing. Default is False.

        Returns
        -------
        Grid
            New Grid instance.

        Examples
        --------
        >>> bounds = (0, 0, 10, 5)
        >>> grid = Grid.from_bounds_and_spacing(bounds, spacing=1.0)  # Same spacing
        >>> grid = Grid.from_bounds_and_spacing(bounds, spacing=(1.0, 0.5))  # Different spacing
        """
        xmin, ymin, xmax, ymax = bounds

        # Apply margin
        if margin > 0:
            w = xmax - xmin
            h = ymax - ymin
            xmin -= margin * w
            xmax += margin * w
            ymin -= margin * h
            ymax += margin * h

        # Handle spacing specification
        if isinstance(spacing, (int, float)):
            dx = dy = float(spacing)
        else:
            dx, dy = spacing

        # Compute grid dimensions
        sx = int((xmax - xmin) / dx)
        sy = int((ymax - ymin) / dy)

        # Handle strict spacing requirement
        if strict:
            # Adjust bounds to ensure exact spacing
            xmax = xmin + sx * dx
            ymax = ymin + sy * dy

        return cls(bounds, (sx, sy), margin, False)

    @classmethod
    def from_size_and_spacing(
        cls,
        size: Union[int, tuple[int, int]],
        spacing: Union[float, tuple[float, float]],
        center: tuple[float, float] = (0.0, 0.0),
    ) -> "Grid":
        """
        Create a Grid from size and spacing specification.

        Parameters
        ----------
        size : int or tuple
            Grid size:
            - int: same number of points in both directions
            - (sx, sy): different size in each direction
        spacing : float or tuple
            Grid spacing:
            - float: same spacing in both directions
            - (dx, dy): different spacing in each direction
        center : tuple of float, optional
            Center coordinates of the grid as (center_x, center_y).
            Default is (0.0, 0.0).

        Returns
        -------
        Grid
            New Grid instance with bounds computed around the specified center.

        Examples
        --------
        >>> grid = Grid.from_size_and_spacing(size=100, spacing=1.0)  # Centered at origin
        >>> grid = Grid.from_size_and_spacing(size=(50, 25), spacing=(1.0, 2.0))  # Centered at origin
        >>> grid = Grid.from_size_and_spacing(size=100, spacing=1.0, center=(10.0, 5.0))  # Custom center
        """
        # Handle size specification
        if isinstance(size, int):
            sx = sy = size
        else:
            sx, sy = size

        # Handle spacing specification
        if isinstance(spacing, (int, float)):
            dx = dy = float(spacing)
        else:
            dx, dy = spacing

        # Compute bounds centered at specified center coordinates
        w = sx * dx
        h = sy * dy
        center_x, center_y = center
        xmin = center_x - w / 2
        ymin = center_y - h / 2
        xmax = center_x + w / 2
        ymax = center_y + h / 2

        return cls((xmin, ymin, xmax, ymax), (sx, sy), 0.0, False)

    def __repr__(self) -> str:
        """Concise string representation for terminal display."""
        # Show dimensions and spacing in a compact format
        sx, sy = self.size
        dx, dy = self.spacing

        # Format bounds concisely
        xmin, ymin, xmax, ymax = self.bounds
        bounds_str = f"({xmin:.1f}, {ymin:.1f}) to ({xmax:.1f}, {ymax:.1f})"

        return f"Grid({sx}x{sy}, dx={dx:.3f}, dy={dy:.3f}, bounds={bounds_str})"


def build_multilevel_grids(bounds, N, n_levels=3, margin=0.5, square=False) -> list:
    """
    Build dyadically scaled FFT-friendly grids for multi-resolution cartography.

    Creates a hierarchy of grids with dyadically decreasing resolution levels,
    starting from the highest resolution and working downward. Each level has
    dimensions exactly half of the previous level, making them ideal for
    multi-resolution cartogram algorithms and FFT-based computations.

    Parameters
    ----------
    bounds : tuple of float
        The bounding box as (xmin, ymin, xmax, ymax) coordinates that define
        the spatial extent for all resolution levels.
    N : int
        Number of grid points along the longest bounding box edge at the
        highest resolution level. Will be adjusted to be divisible by
        2^(n_levels-1) for exact halving at each level.
    n_levels : int, optional
        Number of resolution levels to create. Each level halves the
        dimensions of the previous level. Default is 3.
    margin : float, optional
        Margin to add around the bounds as a fraction of the bounding box
        dimensions before computing grid sizes. Default is 0.5.

    Returns
    -------
    list of Grid
        List of Grid objects ordered from lowest to highest resolution:

        - grids[0] : Grid
            Lowest resolution grid with dimensions divided by 2^(n_levels-1)
        - grids[1] : Grid
            Second resolution level with dimensions doubled from previous
        - grids[n_levels-1] : Grid
            Highest resolution grid with approximately N x N points

        Each Grid object contains coordinate arrays and metadata for
        its respective resolution level.

    Examples
    --------
    >>> bounds = (0, 0, 100, 80)  # Rectangular bounding box
    >>> grids = build_multilevel_grids(bounds, N=256, n_levels=3)
    >>> print(f"Number of levels: {len(grids)}")
    Number of levels: 3
    >>> print(f"Level 0 (lowest res) shape: {grids[0].shape}")
    Level 0 (lowest res) shape: (64, 128)
    >>> print(f"Level 1 shape: {grids[1].shape}")
    Level 1 shape: (128, 256)
    >>> print(f"Level 2 (highest res) shape: {grids[2].shape}")
    Level 2 (highest res) shape: (256, 512)

    Notes
    -----
    The algorithm ensures FFT-friendly dimensions by:

    1. Adjusting N to be divisible by 2^(n_levels-1)
    2. Computing dimensions that maintain aspect ratio
    3. Finding the best short-axis resolution to minimize aspect distortion
    4. Creating grids where each level has exactly double the dimensions of the previous

    The returned grids are ordered from lowest to highest resolution, making
    them suitable for algorithms that progressively refine solutions from
    coarse to fine scales. The lowest resolution grid (grids[0]) provides
    a smoothed overview, while the highest resolution grid (grids[-1])
    captures fine details.
    """

    xmin, ymin, xmax, ymax = bounds

    w = xmax - xmin
    h = ymax - ymin
    xmin -= margin * w
    xmax += margin * w
    ymin -= margin * h
    ymax += margin * h
    Lx = xmax - xmin
    Ly = ymax - ymin

    # Ensure N is divisible by 2^(n_levels-1)
    pow2 = 2 ** (n_levels - 1)
    N = int(math.ceil(N / pow2) * pow2)

    # Determine long vs short side
    if Lx >= Ly:
        nx_long = N
        # compute ny that gives dx≈dy
        ny_target = round((Ly / Lx) * nx_long)
    else:
        ny_long = N
        nx_target = round((Lx / Ly) * ny_long)

    # Adjust the short axis resolution to nearest multiple of 2^(n_levels-1)
    if Lx >= Ly:
        # start from target, adjust to maintain dx≈dy
        ny_candidates = [ny_target + k for k in range(-4, 5)]
        ny_candidates = [ny for ny in ny_candidates if ny > 4]
        ny_best = min(ny_candidates, key=lambda n: abs((Lx / nx_long) / (Ly / n) - 1))
        ny_best = int(math.ceil(ny_best / pow2) * pow2)
        nx, ny = nx_long, ny_best
    else:
        nx_candidates = [nx_target + k for k in range(-4, 5)]
        nx_candidates = [nx for nx in nx_candidates if nx > 4]
        nx_best = min(nx_candidates, key=lambda n: abs((Lx / n) / (Ly / ny_long) - 1))
        nx_best = int(math.ceil(nx_best / pow2) * pow2)
        nx, ny = nx_best, ny_long

    # Now compute the base grid and derive lower levels
    grids = []
    for level in range(n_levels):
        scale = 2 ** (n_levels - 1 - level)
        grids.append(Grid((xmin, ymin, xmax, ymax), size=(nx // scale, ny // scale), margin=0.0, square=square))

    return grids
