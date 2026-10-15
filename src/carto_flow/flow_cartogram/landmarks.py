"""Uniform point landmarks after a flow morph.

Points that are uniform in a region before the morph are not uniform in the morphed region: the flow
scales the area less near the border of the region than inside (the density field is a blend of the region and
its neighbors there), so points of an expanding region crowd the border and points of a shrinking region leave
it. The area scaling of the flow along the path of every landmark (``Cartogram.landmark_area_scaling``)
gives the density of the landmarks relative to a uniform one, and the distance of a landmark to the border of
its region is remapped so that the density becomes uniform again, along the direction to the nearest border
point.

The remap needs the distribution of the distance to the border of a uniform sample of the region. It is computed
deterministically from a distance transform of the polygon and does not depend on a random sample, so the
correction changes smoothly from one iteration of a morph to the next.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import shapely
from scipy import ndimage

__all__ = ["distance_distributions", "probe_points", "region_of_points", "uniform_landmarks"]

# Bins of the distance to the border, in units of sqrt(area of the region), for the density ratio of the landmarks
_EDGES = np.array([0.0, 0.01, 0.02, 0.04, 0.06, 0.1, 0.2, 0.4, 0.6])
# Distances at which the distribution of the distance to the border is tabulated, in the same units
_SGRID = np.array([0.0, 0.005, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
_PIXEL = 0.0075  # pixel of the raster of a region, in units of sqrt(area)
_NEAR = 0.03  # below the distance _NEAR the distribution is extrapolated linearly from its value there
_MAX_PIXELS = 400_000


def region_of_points(regions: Any, points: np.ndarray) -> np.ndarray:
    """Index of the first region that contains each point, -1 for a point outside all regions."""
    tree = shapely.STRtree(np.asarray(regions, dtype=object))
    hit_i, hit_r = tree.query(shapely.points(points), predicate="within")
    owner = np.full(len(points), -1)
    owner[hit_i[::-1]] = hit_r[::-1]
    return owner


def _distribution(poly: Any) -> np.ndarray | None:
    """Share of the area of the polygon within each distance of ``_SGRID`` (x sqrt(area)) of its border."""
    if poly.is_empty or poly.area <= 0:
        return None
    sc = np.sqrt(poly.area)
    minx, miny, maxx, maxy = poly.bounds
    pixel = max(_PIXEL * sc, np.sqrt(max((maxx - minx) * (maxy - miny), 0.0) / _MAX_PIXELS))
    nx, ny = int((maxx - minx) / pixel) + 3, int((maxy - miny) / pixel) + 3
    x = minx - pixel + (np.arange(nx) + 0.5) * pixel
    y = miny - pixel + (np.arange(ny) + 0.5) * pixel
    X, Y = np.meshgrid(x, y)
    mask = shapely.contains_xy(poly, X.ravel(), Y.ravel()).reshape(X.shape)
    if not mask.any():
        return None
    # distance of a pixel center to the nearest outside pixel center, less half a pixel
    dist = np.maximum(ndimage.distance_transform_edt(mask)[mask] - 0.5, 0.0) * pixel / sc
    F = np.searchsorted(np.sort(dist), _SGRID, side="right") / dist.size
    # Within a few pixels of the border the raster is too coarse: the share of the area within a small
    # distance is proportional to the distance (the perimeter times the distance, to first order).
    near = _SGRID <= _NEAR
    F[near] = F[np.searchsorted(_SGRID, _NEAR)] * _SGRID[near] / _NEAR
    return F


def distance_distributions(regions: Any) -> list[np.ndarray | None]:
    """The distribution of the distance to the border of every region (None for an empty region)."""
    return [_distribution(shapely.buffer(g, 0)) for g in regions]


def _uniform_in(poly: Any, n: int, rng: np.random.Generator) -> np.ndarray:
    minx, miny, maxx, maxy = poly.bounds
    out = np.empty((0, 2))
    while len(out) < n:
        c = np.column_stack([rng.uniform(minx, maxx, 4 * n), rng.uniform(miny, maxy, 4 * n)])
        out = np.vstack([out, c[shapely.contains_xy(poly, c[:, 0], c[:, 1])]])
    return out[:n]


def probe_points(regions: Any, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``n`` points per region, uniform in the region: their coordinates and the index of their region."""
    rng = np.random.default_rng(seed)
    xy, region = [], []
    for r, geom in enumerate(regions):
        poly = shapely.buffer(geom, 0)
        if poly.is_empty or poly.area <= 0:
            continue
        xy.append(_uniform_in(poly, n, rng))
        region.append(np.full(n, r))
    if not xy:
        return np.empty((0, 2)), np.empty(0, dtype=int)
    return np.vstack(xy), np.concatenate(region)


def uniform_landmarks(
    final_regions: Any,
    points: np.ndarray,
    owner: np.ndarray,
    area_ratio: np.ndarray,
    distributions: list[np.ndarray | None],
    profile_points: np.ndarray,
    profile_scaling: np.ndarray,
    profile_owner: np.ndarray,
) -> np.ndarray:
    """Positions of points after the remap of their distance to the border of their region.

    Parameters
    ----------
    final_regions : array-like of polygons
        The regions at the iteration of the points.
    points : np.ndarray, shape (n, 2)
        The points as carried by the flow at that iteration: landmark vertices or displacement coordinates.
    owner : np.ndarray, shape (n,)
        The region of every point before the morph (-1: none, the point is not moved).
    area_ratio : np.ndarray, shape (n_regions,)
        The area of every region at that iteration divided by its area before the morph.
    distributions : list of np.ndarray
        The distribution of the distance to the border of every region (``distance_distributions``); it
        can be computed once and used for every iteration.
    profile_points, profile_scaling, profile_owner : np.ndarray
        The points that measure the crowding near the borders, as carried by the flow at that iteration, the area
        scaling along their paths and their regions before the morph. They must have been uniform in their region
        before the morph: probe points (``probe_points``), or point landmarks that were uniform.

    Returns
    -------
    np.ndarray, shape (n, 2)
        The corrected positions. Points outside every original region, outside their morphed region, deeper than
        the tabulated distances, or in a region with too few profile points are not moved.
    """
    out = np.array(points, dtype=float)
    centers = 0.5 * (_EDGES[:-1] + _EDGES[1:])
    for r in np.unique(owner[owner >= 0]):
        sel = np.where(owner == r)[0]
        psel = np.where(profile_owner == r)[0]
        F = distributions[r]
        poly = shapely.buffer(final_regions[r], 0)
        if psel.size < 3 or F is None or poly.is_empty:
            continue
        sc = np.sqrt(poly.area)
        ppts = shapely.points(profile_points[psel])
        pins = shapely.contains(poly, ppts)
        pd = shapely.distance(ppts, poly.boundary) / sc
        c = area_ratio[r] / profile_scaling[psel]  # density relative to uniform
        chat = np.full(len(centers), np.nan)
        for b in range(len(centers)):
            m = pins & (pd >= _EDGES[b]) & (pd < _EDGES[b + 1])
            if m.any():
                chat[b] = 1.0 / np.mean(1.0 / c[m])  # area average of the density in the bin
        ok = np.isfinite(chat)
        if ok.sum() < 2:
            continue
        cfun = np.interp(_SGRID, centers[ok], chat[ok])
        G = np.concatenate([[0.0], np.cumsum(np.diff(F) * 0.5 * (cfun[1:] + cfun[:-1]))])
        if G[-1] <= 0 or F[-1] <= 0:
            continue
        G = G / G[-1] * F[-1]
        keep = np.concatenate([[True], np.diff(F) > 1e-9])  # F must be strictly increasing to be inverted
        pts = shapely.points(out[sel])
        inside = shapely.contains(poly, pts)
        d = np.where(inside, 1.0, -1.0) * shapely.distance(pts, poly.boundary) / sc
        move = inside & (d <= _SGRID[-1])  # the deepest part of the region is left where it is
        if not move.any():
            continue
        new_d = np.interp(np.interp(d[move], _SGRID, G), F[keep], _SGRID[keep])
        line = shapely.shortest_line(pts[move], poly.boundary)
        xy = shapely.get_coordinates(line).reshape(-1, 2, 2)
        v = xy[:, 0, :] - xy[:, 1, :]
        u = v / np.maximum(np.linalg.norm(v, axis=1), 1e-300)[:, None]
        out[sel[move]] += ((new_d - d[move]) * sc)[:, None] * u
    return out
