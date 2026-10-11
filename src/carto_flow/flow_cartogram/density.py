"""
Density field computation utilities for cartographic applications.

This module provides functions for computing density fields from geospatial
data, which are essential for flow-based cartography algorithms. The main
functionality includes rasterizing polygon geometries onto regular grids to
create density distributions that drive cartogram deformation processes.

Functions
---------
compute_density_field
    Core function for rasterizing polygon data to density grids.

Examples
--------
>>> from carto_flow.flow_cartogram.density import compute_density_field
>>> from carto_flow.flow_cartogram.grid import Grid
>>> grid = Grid.from_bounds((0, 0, 100, 80), size=100)
>>> density = compute_density_field(gdf, "population", grid)
"""

from typing import Any

import numpy as np
import shapely
from numba import njit
from scipy.ndimage import distance_transform_edt, gaussian_filter

# Import grid utilities
from .grid import Grid

# Module-level exports - Public API
__all__ = [
    "DensityBorderExtension",
    "DensityModulator",
    "DensityPipeline",
    "Smooth",
    "compute_density_field",
    "compute_density_field_from_geometries",
    "preview_modulator",
]


def compute_density_field(
    gdf: Any,
    column: str,
    grid: Grid,
    mean_density: float | None = None,
    smooth: float | None = None,
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

    x_1d = grid.x_coords
    y_1d = grid.y_coords

    # Assign density based on polygon population / area
    for _idx, row in gdf.iterrows():
        poly = row.geometry
        if poly.is_empty:
            continue

        pop_density = row[column] / poly.area

        minx, miny, maxx, maxy = poly.bounds
        c0 = int(np.searchsorted(x_1d, minx))
        c1 = int(np.searchsorted(x_1d, maxx, side="right"))
        r0 = int(np.searchsorted(y_1d, miny))
        r1 = int(np.searchsorted(y_1d, maxy, side="right"))
        if r0 >= r1 or c0 >= c1:
            continue

        contains = shapely.contains_xy(poly, grid.X[r0:r1, c0:c1], grid.Y[r0:r1, c0:c1])
        rho[r0:r1, c0:c1][contains] = pop_density

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
    geometries,
    column_values,
    grid,
    mean_density=None,
    smooth=None,
    return_geometry_mask=False,
    return_outside_mask=False,  # Deprecated for backward compatibility
    use_parallel=False,
    n_jobs=None,
    rasterization="center",
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
        Gaussian smoothing sigma applied to the entire density field (both
        inside and outside). Preserves the global mean.
    return_geometry_mask : bool, default=False
        If True, return tuple (rho, geometry_mask) where geometry_mask is an integer
        array indicating which geometry each cell belongs to:
        - -1: Outside all geometries
        - k: Inside geometry k (0 <= k < number of geometries)
    return_outside_mask : bool, default=False
        (Deprecated) If True, return tuple (rho, outside_mask) where outside_mask is a boolean
        array indicating cells outside all geometries.
    use_parallel : bool, default=False
        Whether to use parallel computation for density field calculation.
    n_jobs : int, optional
        Number of parallel jobs to use for density field calculation. If None,
        uses all available cores.
    rasterization : {"center", "exact"}, default="center"
        ``"center"``: each cell takes the density of the geometry that contains its center, and
        the mean density outside all geometries. ``"exact"``: each cell takes the area-weighted
        mean of the geometry densities over the cell (the uncovered part at the mean density), so
        a geometry smaller than a cell contributes in proportion to its area and the integral of
        the density over the grid equals the total value. With ``"exact"``, ``geometry_mask`` holds
        the geometry that covers most of the cell, or -1 if that geometry covers less than half of
        it, and ``use_parallel`` and ``n_jobs`` are ignored.

    Returns
    -------
    rho : np.ndarray
        Density field array.
    geometry_mask : np.ndarray (only if return_geometry_mask=True)
        Integer array where each element indicates which geometry the cell belongs to.
    outside_mask : np.ndarray (only if return_outside_mask=True)
        Boolean array where True indicates cells outside all geometries.
    """
    if rasterization not in ("center", "exact"):
        raise ValueError(f"rasterization must be 'center' or 'exact', got {rasterization!r}")
    if rasterization == "exact":
        return _exact_density(
            geometries, column_values, grid, mean_density, smooth, return_geometry_mask, return_outside_mask
        )

    rho = np.zeros_like(grid.X)
    # Initialize geometry mask with -1 (outside all geometries)
    geom_mask = np.full(grid.X.shape, -1, dtype=int)

    # Precompute 1D coordinate arrays for searchsorted-based bbox slicing.
    # grid.X[i,j] = x_coords[j] and grid.Y[i,j] = y_coords[i], so row bounds
    # map to y and column bounds map to x.
    x_1d = grid.x_coords  # shape (sx,), sorted ascending
    y_1d = grid.y_coords  # shape (sy,), sorted ascending

    def _slice_bounds(geom):
        """Return (r0, r1, c0, c1) integer slice bounds for geom's bbox, or None."""
        minx, miny, maxx, maxy = geom.bounds
        c0 = int(np.searchsorted(x_1d, minx))
        c1 = int(np.searchsorted(x_1d, maxx, side="right"))
        r0 = int(np.searchsorted(y_1d, miny))
        r1 = int(np.searchsorted(y_1d, maxy, side="right"))
        if r0 >= r1 or c0 >= c1:
            return None
        return r0, r1, c0, c1

    # Assign density and geometry indices
    if use_parallel:
        import os

        from joblib import Parallel, delayed

        if n_jobs is None or n_jobs == -1:
            n_jobs = os.cpu_count() or 4

        # Phase 1: serial precomputation — bbox bounds as integers only (trivially fast).
        precomputed = []
        for idx, (geom, value) in enumerate(zip(geometries, column_values, strict=False)):
            if geom.is_empty:
                continue
            bounds = _slice_bounds(geom)
            if bounds is None:
                continue
            precomputed.append((idx, geom, value / geom.area, bounds))

        # Phase 2: parallel contains_xy — each task works on a small bbox sub-array.
        # contains_xy releases the GIL (C/GEOS), so threads run truly in parallel.
        def run_pip(idx, geom, geom_density, bounds):
            r0, r1, c0, c1 = bounds
            contains = shapely.contains_xy(geom, grid.X[r0:r1, c0:c1], grid.Y[r0:r1, c0:c1])
            return idx, geom_density, bounds, contains

        results = Parallel(n_jobs=n_jobs, backend="threading")(delayed(run_pip)(*entry) for entry in precomputed)
        del precomputed

        # Phase 3: serial result assembly — write into bbox sub-array slices.
        for idx, geom_density, bounds, contains in results:
            r0, r1, c0, c1 = bounds
            rho[r0:r1, c0:c1][contains] = geom_density
            if return_geometry_mask:
                geom_mask[r0:r1, c0:c1][contains] = idx
    else:
        # Serial implementation
        for idx, (geom, value) in enumerate(zip(geometries, column_values, strict=False)):
            if geom.is_empty:
                continue

            bounds = _slice_bounds(geom)
            if bounds is None:
                continue

            r0, r1, c0, c1 = bounds
            geom_density = value / geom.area
            contains = shapely.contains_xy(geom, grid.X[r0:r1, c0:c1], grid.Y[r0:r1, c0:c1])
            rho[r0:r1, c0:c1][contains] = geom_density
            geom_mask[r0:r1, c0:c1][contains] = idx

    # Identify exterior cells (where geom_mask is still -1)
    outside = geom_mask == -1

    if mean_density is None:
        # Compute mean density from provided areas
        total_value = np.sum(column_values)
        total_area = np.sum([g.area for g in geometries])
        mean_density = float(total_value / total_area) if total_area > 0 else 0.0

    # Default: fill outside cells with uniform mean_density
    rho[outside] = mean_density

    # Apply smoothing if requested (blurs the entire field, preserving global mean)
    if smooth is not None:
        mu_rho = np.mean(rho)
        rho = gaussian_filter(rho, sigma=smooth)
        # Preserve the global mean
        rho *= mu_rho / np.mean(rho)

    if return_geometry_mask:
        return rho, geom_mask
    if return_outside_mask:
        return rho, outside
    return rho


@njit(cache=True)
def _coverage_kernel(u, v, starts, nr, nc, cu, cv):
    """Exact fraction of each cell covered by oriented rings, by signed area accumulation.

    ``u, v`` are ring vertices in cell units (cell (i, j) spans [j, j+1] x [i, i+1] after subtracting
    the offset ``cu, cv`` of the first cell); ``starts`` gives the ring boundaries. Each edge adds its signed area contribution to the cells it crosses; a
    cumulative sum along each row then gives the coverage (the method used by scanline font
    rasterizers).
    """
    acc = np.zeros((nr, nc + 2))
    for k in range(len(starts) - 1):
        for i in range(starts[k], starts[k + 1] - 1):
            x0, y0, x1, y1 = u[i] - cu, v[i] - cv, u[i + 1] - cu, v[i + 1] - cv
            if y0 == y1:
                continue
            d = 1.0
            if y0 > y1:
                x0, y0, x1, y1 = x1, y1, x0, y0
                d = -1.0
            dxdy = (x1 - x0) / (y1 - y0)
            x = x0
            if y0 < 0.0:
                x -= y0 * dxdy
                y0 = 0.0
            yend = min(nr, int(np.ceil(y1)))
            for r in range(int(y0), yend):
                dy = min(r + 1.0, y1) - max(float(r), y0)
                xn = x + dxdy * dy
                w = dy * d
                xa = min(max(min(x, xn), 0.0), float(nc))
                xb = min(max(max(x, xn), 0.0), float(nc))
                xaf = np.floor(xa)
                ia = int(xaf)
                ib = int(np.ceil(xb))
                if ib <= ia + 1:
                    xm = 0.5 * (xa + xb) - xaf
                    acc[r, ia] += w - w * xm
                    acc[r, ia + 1] += w * xm
                else:
                    sl = 1.0 / (xb - xa)
                    xa_f = xa - xaf
                    a0 = 0.5 * sl * (1.0 - xa_f) * (1.0 - xa_f)
                    xb_f = xb - ib + 1.0
                    am = 0.5 * sl * xb_f * xb_f
                    acc[r, ia] += w * a0
                    if ib == ia + 2:
                        acc[r, ia + 1] += w * (1.0 - a0 - am)
                    else:
                        a1 = sl * (1.5 - xa_f)
                        acc[r, ia + 1] += w * (a1 - a0)
                        for xi in range(ia + 2, ib - 1):
                            acc[r, xi] += w * sl
                        a2 = a1 + (ib - ia - 3) * sl
                        acc[r, ib - 1] += w * (1.0 - a2 - am)
                    acc[r, ib] += w * am
                x = xn
    out = np.zeros((nr, nc))
    for r in range(nr):
        run = 0.0
        for c in range(nc):
            run += acc[r, c]
            f = -run
            out[r, c] = 0.0 if f < 1e-12 else (1.0 if f > 1.0 else f)
    return out


@njit(cache=True)
def _exact_fields(u, v, ring_starts, geom_rings, density, nx, ny, rho, cover, best, mask):
    """Accumulate exact coverage of every geometry into ``rho``, ``cover``, ``best`` and ``mask``."""
    for g in range(len(geom_rings) - 1):
        ra, rb = geom_rings[g], geom_rings[g + 1]
        if rb <= ra:
            continue
        pa, pb = ring_starts[ra], ring_starts[rb]
        c0, r0, c1, r1 = nx, ny, 0, 0
        for i in range(pa, pb):
            c0 = min(c0, int(np.floor(u[i])))
            c1 = max(c1, int(np.ceil(u[i])))
            r0 = min(r0, int(np.floor(v[i])))
            r1 = max(r1, int(np.ceil(v[i])))
        c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(max(c1, c0 + 1), nx), min(max(r1, r0 + 1), ny)
        if c1 <= c0 or r1 <= r0:
            continue
        frac = _coverage_kernel(
            u[pa:pb], v[pa:pb], ring_starts[ra : rb + 1] - pa, r1 - r0, c1 - c0, float(c0), float(r0)
        )
        for i in range(r1 - r0):
            for j in range(c1 - c0):
                f = frac[i, j]
                if f > 0.0:
                    rho[r0 + i, c0 + j] += f * density[g]
                    cover[r0 + i, c0 + j] += f
                    if f > best[r0 + i, c0 + j]:
                        best[r0 + i, c0 + j] = f
                        mask[r0 + i, c0 + j] = g


def _exact_coverage_fields(geometries, density, grid, rho, cover, best, geom_mask):
    """Exact coverage of all geometries on the grid (polygon edges integrated per cell, in numba)."""
    arr = np.empty(len(geometries), dtype=object)
    arr[:] = list(geometries)
    parts, part_g = shapely.get_parts(shapely.orient_polygons(arr), return_index=True)
    rings, ring_p = shapely.get_rings(parts, return_index=True)
    pts, pt_r = shapely.get_coordinates(rings, return_index=True)
    ring_starts = np.searchsorted(pt_r, np.arange(len(rings) + 1))
    geom_rings = np.searchsorted(part_g[ring_p], np.arange(len(arr) + 1))
    u = (pts[:, 0] - (grid.x_coords[0] - 0.5 * grid.dx)) / grid.dx
    v = (pts[:, 1] - (grid.y_coords[0] - 0.5 * grid.dy)) / grid.dy
    ny, nx = rho.shape
    _exact_fields(u, v, ring_starts, geom_rings, density, nx, ny, rho, cover, best, geom_mask)


def _exact_density(geometries, column_values, grid, mean_density, smooth, return_geometry_mask, return_outside_mask):
    """Area-weighted density field (``rasterization="exact"`` of ``compute_density_field_from_geometries``)."""
    rho = np.zeros_like(grid.X)
    cover = np.zeros_like(grid.X)
    best = np.zeros_like(grid.X)
    geom_mask = np.full(grid.X.shape, -1, dtype=int)

    areas = shapely.area(np.asarray(list(geometries), dtype=object)) if len(geometries) else np.zeros(0)
    valid = np.array([not g.is_empty for g in geometries], dtype=bool) & (areas > 0)
    density = np.where(valid, np.asarray(column_values, dtype=float) / np.where(valid, areas, 1.0), 0.0)
    total_area = float(areas[valid].sum())

    _exact_coverage_fields(
        [g if ok else shapely.Polygon() for g, ok in zip(geometries, valid, strict=False)],
        density,
        grid,
        rho,
        cover,
        best,
        geom_mask,
    )

    if mean_density is None:
        total_value = np.sum(column_values)
        mean_density = float(total_value / total_area) if total_area > 0 else 0.0
    rho += np.clip(1.0 - cover, 0.0, 1.0) * mean_density
    outside = best < 0.5
    geom_mask[outside] = -1

    if smooth is not None:
        mu_rho = np.mean(rho)
        rho = gaussian_filter(rho, sigma=smooth)
        rho *= mu_rho / np.mean(rho)

    if return_geometry_mask:
        return rho, geom_mask
    if return_outside_mask:
        return rho, outside
    return rho


class DensityModulator:
    """Abstract base class for density field modulators.

    Subclasses implement ``__call__(density, grid, mask, target_density)``
    and return a modified density array.  Modulators can be chained with
    ``+`` into a :class:`DensityPipeline` that applies them left-to-right::

        mod = DensityBorderExtension(extension_width=10) + DensitySmooth(sigma=2)
    """

    def __call__(self, density: np.ndarray, grid: Grid, mask: np.ndarray, target_density: float) -> np.ndarray:
        """Apply modulation to density field."""
        raise NotImplementedError

    def __add__(self, other: "DensityModulator") -> "DensityPipeline":
        """Chain modulators in sequence (self followed by other)."""
        if isinstance(self, DensityPipeline):
            return DensityPipeline([*self.modulators, other])
        else:
            return DensityPipeline([self, other])


class DensityPipeline(DensityModulator):
    """Sequence of density modulators applied left-to-right.

    Constructed automatically when two modulators are combined with ``+``.
    Further ``+`` calls append to the same pipeline rather than nesting::

        pipe = DensityBorderExtension() + DensitySmooth(sigma=2)
    """

    def __init__(self, modulators: list):
        self.modulators = modulators

    def __call__(self, density: np.ndarray, grid: Grid, mask: np.ndarray, target_density: float) -> np.ndarray:
        """Apply all modulators in sequence."""
        for mod in self.modulators:
            density = mod(density, grid, mask, target_density)
        return density

    def __add__(self, other: "DensityModulator") -> "DensityPipeline":
        """Chain another modulator after this pipeline."""
        return DensityPipeline([*self.modulators, other])

    def __repr__(self) -> str:
        return f"Pipeline({self.modulators})"


class DensityBorderExtension(DensityModulator):
    """Extend interior density values outward, blending toward the target density.

    By default, outside cells are assigned ``target_density`` (the
    mean equalized density).  This can cause a sharp density step at the
    outer boundary, which may distort boundary geometries during morphing.

    ``DensityBorderExtension`` replaces outside-cell densities with a
    smooth outward extrapolation of the nearest interior values.  Over
    ``extension_width`` units outside the boundary the interior density is
    copied as-is; beyond that it transitions toward ``target_density`` over
    ``transition_width`` units via linear blending.

    Parameters
    ----------
    extension_width : float, default 10.0
        Distance in world/data-coordinate units over which interior
        densities are propagated outward without blending.
    transition_width : float, default 10.0
        Distance in world/data-coordinate units over which the density
        blends from the extrapolated interior value to ``target_density``.
        Set to 0 for a hard cutoff at ``extension_width``.
    smooth : float or None, default None
        Optional Gaussian smoothing sigma in world-coordinate units
        applied to the final density field to remove small artifacts.

    Examples
    --------
    >>> DensityBorderExtension(extension_width=5, transition_width=20)
    >>> DensityBorderExtension(extension_width=10) + DensitySmooth(sigma=2)
    """

    def __init__(self, extension_width: float = 10.0, transition_width: float = 10.0, smooth: float | None = None):
        self.extension_width = extension_width
        self.transition_width = transition_width
        self.smooth = smooth

    def __call__(self, density: np.ndarray, grid: Grid, mask: np.ndarray, target_density: float) -> np.ndarray:
        """Extend density values from the interior outward and blend to target."""
        outside = mask == -1

        # EDT with cell-size sampling returns distances in world/data-coordinate units
        dist, indices = distance_transform_edt(outside, return_indices=True, sampling=(grid.dy, grid.dx))

        density_mod = density.copy()

        # copy interior densities outward
        density_mod[outside] = density[indices[0][outside], indices[1][outside]]

        if self.transition_width > 0:
            s = np.clip((dist[outside] - self.extension_width) / self.transition_width, 0, 1)
            density_mod[outside] = (1 - s) * density_mod[outside] + s * target_density
        else:
            density_mod[outside & (dist > self.extension_width)] = target_density

        # optional smoothing to remove small artifacts
        if self.smooth is not None and self.smooth > 0:
            sigma_yx = (self.smooth / grid.dy, self.smooth / grid.dx)
            density_mod = gaussian_filter(density_mod, sigma_yx, mode="reflect")

        return density_mod

    def __repr__(self) -> str:
        return f"DensityBorderExtension(extension_width={self.extension_width}, transition_width={self.transition_width}, smooth={self.smooth})"


class Smooth(DensityModulator):
    """Gaussian smoothing of the density field, preserving the global mean.

    Convolves the density field with an isotropic Gaussian kernel, then
    rescales the result so the spatial mean is unchanged.  This softens
    sharp density boundaries between geometries without shifting the
    overall target density.  Exported from the package as ``DensitySmooth``
    to avoid name collision with the velocity :class:`~anisotropy.Smooth`.

    Parameters
    ----------
    sigma : float, default 1.0
        Standard deviation of the Gaussian kernel in world/data-coordinate
        units.  Converted to pixels at call time.

    Examples
    --------
    >>> DensitySmooth(sigma=3)
    """

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def __call__(self, density: np.ndarray, grid: Grid, mask: np.ndarray, target_density: float) -> np.ndarray:
        mu = np.mean(density)
        sigma_yx = (self.sigma / grid.dy, self.sigma / grid.dx)
        smoothed = gaussian_filter(density, sigma=sigma_yx, mode="reflect")
        smoothed *= mu / np.mean(smoothed)
        return smoothed

    def __repr__(self) -> str:
        return f"DensitySmooth(sigma={self.sigma})"


def preview_modulator(
    modulator: DensityModulator,
    gdf,
    grid_size: int = 64,
    column: str | None = None,
    values=None,
    margin: float = 0.1,
    show_geometry: bool = True,
    show: str = "output",
    ax=None,
    cmap: str = "RdBu_r",
    show_colorbar: bool = True,
    colorbar_kwargs: dict | None = None,
    image_kwargs: dict | None = None,
    area_scale: float = 1.0,
):
    """Preview a density modulator applied to a computed density field.

    Rasterizes the input density from *gdf* (or uniform values if *column* and
    *values* are both omitted), applies the modulator, and displays the result
    as a heatmap.

    Parameters
    ----------
    modulator : DensityModulator
        The modulator to preview.
    gdf : GeoDataFrame
        Geometries used to derive the spatial extent, the geometry mask, and
        (when *column* is given) the density values.
    grid_size : int, default 64
        Number of grid cells along the longer axis.
    column : str, optional
        Column in *gdf* to use as per-geometry values.  Takes precedence over
        *values*.
    values : array-like, optional
        Per-geometry values (same length as *gdf*).  Ignored when *column* is
        given.  If both are ``None``, uniform values (all 1.0) are used so the
        input density equals 1 / geometry_area everywhere.
    margin : float, default 0.1
        Fractional margin added around the bounds before building the grid.
    show_geometry : bool, default True
        Draw geometry outlines as a light background layer.
    show : str, default 'output'
        Which field to display.  One of:

        - ``'output'`` — post-modulation density field (default).
        - ``'input'`` — pre-modulation density field.
        - ``'ratio'`` — log₁₀ of the element-wise ratio ``rho_out / rho_in``,
          displayed with a symmetric linear norm (equal multiplicative changes
          occupy equal visual space); tick labels show actual ratio values.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created if ``None``.
    cmap : str or Colormap, default 'RdBu_r'
        Colormap used for all ``show`` modes.  A diverging colormap is
        recommended: for ``'input'`` and ``'output'`` the center is anchored
        at *target_density* (white = equilibrium, blue = below, red = above);
        for ``'ratio'`` the center is anchored at 1 (no change).
    show_colorbar : bool, default True
        When ``False``, the colorbar is omitted.
    colorbar_kwargs : dict, optional
        Extra keyword arguments for ``plt.colorbar``.
    image_kwargs : dict, optional
        Extra keyword arguments merged into the ``ax.imshow`` call.
    area_scale : float, default 1.0
        Multiplier applied to areas when computing density, matching
        ``MorphOptions.area_scale``.  For example, ``1e-6`` converts m²
        grid coordinates to km² density units.

    Returns
    -------
    DensityModulatorPreviewResult
        Named container with the produced artists:

        - ``ax`` — the axes
        - ``image`` — the :class:`~matplotlib.image.AxesImage` heatmap
        - ``colorbar`` — :class:`~matplotlib.colorbar.Colorbar`, or ``None``
        - ``geometry_collections`` — list of collections from ``gdf.plot()``

    Examples
    --------
    >>> from carto_flow.flow_cartogram import preview_density_modulator, DensityBorderExtension
    >>> mod = DensityBorderExtension(extension_width=50_000, transition_width=30_000)
    >>> result = preview_density_modulator(mod, gdf, column="population")

    >>> # Inspect where the modulator changed the density (ratio centered at 1):
    >>> result = preview_density_modulator(mod, gdf, column="population", show="ratio")
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    from .grid import Grid
    from .plot_results import DensityModulatorPreviewResult

    if show not in ("input", "output", "ratio"):
        raise ValueError(f"show must be 'input', 'output', or 'ratio', got {show!r}")

    bounds = gdf.total_bounds  # (xmin, ymin, xmax, ymax)
    grid = Grid.from_bounds(bounds, size=grid_size, margin=margin)

    # Rasterize geometries to produce a geometry mask (-1 = outside, k = inside geometry k)
    geom_mask = np.full(grid.shape, -1, dtype=int)
    for idx, geom in enumerate(gdf.geometry):
        if not geom.is_empty:
            inside = shapely.contains_xy(geom, grid.X, grid.Y)
            geom_mask[inside] = idx

    # Resolve values
    if column is not None:
        vals = gdf[column].to_numpy(dtype=float)
    elif values is not None:
        vals = np.asarray(values, dtype=float)
    else:
        vals = np.ones(len(gdf), dtype=float)

    # Compute input density field (rasterized in grid coordinate units)
    rho_in = compute_density_field_from_geometries(gdf.geometry, vals, grid)
    rho_in = rho_in / area_scale

    # target_density = total value / total area, scaled to display units (matches algorithm convention)
    total_area = sum(g.area for g in gdf.geometry if not g.is_empty)
    target_density = float(vals.sum() / total_area / area_scale) if total_area > 0 else 0.0

    # Apply modulator
    rho_out = modulator(rho_in.copy(), grid, geom_mask, target_density)

    # Choose display field
    gxmin, gymin, gxmax, gymax = grid.bounds
    extent = [gxmin, gxmax, gymin, gymax]

    if show == "ratio":
        with np.errstate(divide="ignore", invalid="ignore"):
            display = np.log10(np.where(rho_in > 0, rho_out / rho_in, np.nan))
        max_log = float(np.nanmax(np.abs(display)))
        # Symmetric linear norm on log10 data: equal multiplicative factors occupy equal space
        norm = mcolors.Normalize(vmin=-max_log, vmax=max_log) if max_log > 0 else None
        cb_label = "Density ratio (output / input)"
    elif show == "input":
        with np.errstate(divide="ignore", invalid="ignore"):
            display = np.log10(np.where(rho_in > 0, rho_in / target_density, np.nan))
        max_log = float(np.nanmax(np.abs(display[np.isfinite(display)]))) if target_density > 0 else 0.0
        norm = mcolors.Normalize(vmin=-max_log, vmax=max_log) if max_log > 0 else None
        cb_label = "Input density"
    else:  # "output"
        with np.errstate(divide="ignore", invalid="ignore"):
            display = np.log10(np.where(rho_out > 0, rho_out / target_density, np.nan))
        max_log = float(np.nanmax(np.abs(display[np.isfinite(display)]))) if target_density > 0 else 0.0
        norm = mcolors.Normalize(vmin=-max_log, vmax=max_log) if max_log > 0 else None
        cb_label = "Output density"

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    # Optional geometry outlines
    geom_collections = []
    if show_geometry:
        before = len(ax.collections)
        gdf.plot(ax=ax, facecolor="none", edgecolor="0.6", linewidth=0.6, zorder=1)
        geom_collections = list(ax.collections[before:])

    _img_kw = {
        "origin": "lower",
        "extent": extent,
        "cmap": cmap,
        "zorder": 0,
        **({"norm": norm} if norm is not None else {}),
        **(image_kwargs or {}),
    }
    img = ax.imshow(display, **_img_kw)

    cb = None
    if show_colorbar:
        _cb_kw = {"label": cb_label, **(colorbar_kwargs or {})}
        cb = plt.colorbar(img, ax=ax, **_cb_kw)

    if cb is not None:
        if show == "ratio" and norm is not None:
            # Decade candidates: symmetric pairs selected within ±max_log
            _log_cands = np.log10([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
            ticks_pos = _log_cands[(_log_cands > 0) & (_log_cands <= norm.vmax + 1e-9)]  # type: ignore[operator]
            ticks_log = np.concatenate([-ticks_pos[::-1], [0.0], ticks_pos])
            cb.set_ticks(ticks_log)  # type: ignore[arg-type]
            cb.set_ticklabels([f"{10**t:.3g}" for t in ticks_log])
            cb.ax.axhline(0.0, color="0.15", linewidth=1.5)  # log10(1) = 0
        elif show in ("input", "output") and norm is not None and target_density > 0:
            _log_cands = np.log10([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
            ticks_pos = _log_cands[(_log_cands > 0) & (_log_cands <= norm.vmax + 1e-9)]  # type: ignore[operator]
            ticks_log = np.concatenate([-ticks_pos[::-1], [0.0], ticks_pos])
            cb.set_ticks(ticks_log)  # type: ignore[arg-type]
            cb.set_ticklabels([f"{target_density:.3g}" if t == 0.0 else f"{10**t:.3g}x" for t in ticks_log])
            cb.ax.axhline(0.0, color="0.15", linewidth=1.5)  # log10(1) = 0 = target density

    ax.set_aspect("equal")
    ax.set_title(repr(modulator))
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return DensityModulatorPreviewResult(
        ax=ax,
        image=img,
        colorbar=cb,
        geometry_collections=geom_collections,
    )
