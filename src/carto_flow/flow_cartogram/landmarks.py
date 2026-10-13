"""Uniform point landmarks after a flow morph.

Points that are uniform in a region before the morph are not uniform in the morphed region: the flow
scales the area less near the border of the region than inside (the density field is a blend of the region and
its neighbors there), so points of an expanding region crowd the border and points of a shrinking region leave
it. The area scaling of the flow along the path of every landmark (``Cartogram.landmark_area_scaling``)
gives the density of the landmarks relative to a uniform one, and the distance of a landmark to the border of
its region is remapped so that the density becomes uniform again, along the direction to the nearest border
point.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import shapely

__all__ = ["uniform_landmarks"]

# Bins of the distance to the border, in units of sqrt(area of the region)
_EDGES = np.array([0.0, 0.01, 0.02, 0.04, 0.06, 0.1, 0.2, 0.4, 0.6])


def _uniform_in(poly: Any, n: int, rng: np.random.Generator) -> np.ndarray:
    minx, miny, maxx, maxy = poly.bounds
    out = np.empty((0, 2))
    while len(out) < n:
        c = np.column_stack([rng.uniform(minx, maxx, 4 * n), rng.uniform(miny, maxy, 4 * n)])
        out = np.vstack([out, c[shapely.contains_xy(poly, c[:, 0], c[:, 1])]])
    return out[:n]


def uniform_landmarks(
    original_regions: Any,
    final_regions: Any,
    original_points: np.ndarray,
    final_points: np.ndarray,
    area_scaling: np.ndarray,
    n_ref: int = 400,
    seed: int = 0,
) -> np.ndarray:
    """Positions of point landmarks after the remap of their distance to the border of their region.

    Parameters
    ----------
    original_regions, final_regions : array-like of polygons
        The regions before and after the morph, in the same order.
    original_points, final_points : np.ndarray, shape (n, 2)
        The landmarks before the morph and as carried by the flow.
    area_scaling : np.ndarray, shape (n,)
        The area scaling of the flow along the path of every landmark.
    n_ref : int, default=400
        Number of uniform sample points per region used for the distribution of the distance to the border.
    seed : int, default=0
        Seed of the sample points.

    Returns
    -------
    np.ndarray, shape (n, 2)
        The corrected positions. Landmarks outside every original region, outside their morphed region, or in a
        region with too few landmarks are not moved.
    """
    rng = np.random.default_rng(seed)
    original_regions = np.asarray(original_regions, dtype=object)
    final = [shapely.buffer(g, 0) for g in final_regions]
    tree = shapely.STRtree(original_regions)
    pts0 = shapely.points(original_points)
    hit_i, hit_r = tree.query(pts0, predicate="within")
    owner = np.full(len(original_points), -1)
    owner[hit_i[::-1]] = hit_r[::-1]  # the first region that contains the point
    out = np.array(final_points, dtype=float)
    centers = 0.5 * (_EDGES[:-1] + _EDGES[1:])
    for r in np.unique(owner[owner >= 0]):
        sel = np.where(owner == r)[0]
        poly = final[r]
        if sel.size < 3 or poly.is_empty or original_regions[r].area <= 0:
            continue
        sc = np.sqrt(poly.area)
        pts = shapely.points(out[sel])
        inside = shapely.contains(poly, pts)
        d = np.where(inside, 1.0, -1.0) * shapely.distance(pts, poly.boundary) / sc
        c = (poly.area / original_regions[r].area) / area_scaling[sel]  # density relative to uniform
        chat = np.full(len(centers), np.nan)
        for b in range(len(centers)):
            m = inside & (d >= _EDGES[b]) & (d < _EDGES[b + 1])
            if m.any():
                chat[b] = 1.0 / np.mean(1.0 / c[m])  # area average of the density in the bin
        ok = np.isfinite(chat)
        if ok.sum() < 2:
            continue
        ref = np.sort(shapely.distance(shapely.points(_uniform_in(poly, n_ref, rng)), poly.boundary) / sc)
        sgrid = np.linspace(0.0, ref[int(0.98 * n_ref)], 60)
        F = np.searchsorted(ref, sgrid, side="right") / n_ref
        cfun = np.interp(sgrid, centers[ok], chat[ok])
        G = np.concatenate([[0.0], np.cumsum(np.diff(F) * 0.5 * (cfun[1:] + cfun[:-1]))])
        if G[-1] <= 0 or F[-1] <= 0:
            continue
        G = G / G[-1] * F[-1]
        keep = np.concatenate([[True], np.diff(F) > 0])
        move = inside & (d <= sgrid[-1])  # the deepest 2 % of the area are left where they are
        new_d = np.interp(np.interp(d[move], sgrid, G), F[keep], sgrid[keep])
        line = shapely.shortest_line(pts[move], poly.boundary)
        xy = shapely.get_coordinates(line).reshape(-1, 2, 2)
        v = xy[:, 0, :] - xy[:, 1, :]
        u = v / np.maximum(np.linalg.norm(v, axis=1), 1e-300)[:, None]
        out[sel[move]] += ((new_d - d[move]) * sc)[:, None] * u
    return out
