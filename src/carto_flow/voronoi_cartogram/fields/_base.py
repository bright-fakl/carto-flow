"""Shared Voronoi field base class and exact-cell extraction utility."""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

# ---------------------------------------------------------------------------
# Boundary hygiene
# ---------------------------------------------------------------------------


def drop_sliver_holes(geom):
    """Remove numerically degenerate interior rings from a (Multi)Polygon.

    Unioning a polygonal coverage (the usual way the outer boundary is built)
    leaves behind interior rings with an area at the noise floor of the
    coordinate magnitudes -- typically ``1e-11`` to ``1e-6`` square metres for
    projected metre coordinates.  They are invisible, but GEOS overlay is not
    robust against them: ``intersection(cell, boundary)`` can collapse a
    perfectly ordinary cell that happens to contain such a ring to a
    zero-area ``LineString``, even though ``boundary.covers(cell)`` is
    ``True``.  Dropping the slivers up front avoids that.

    A ring is dropped when ``area <= 1e-9 * scale * perimeter``, i.e. when it
    is thinner than the coordinate resolution at this magnitude (*scale* is the
    largest coordinate magnitude in play).  That is a wide margin: on the US
    districts the 273 sliver rings sit at most ``1.3e-8`` of the threshold and
    the 14 real holes (down to 559 m2) at least ``855`` times above it.
    """
    if geom is None or geom.is_empty or geom.geom_type not in ("Polygon", "MultiPolygon"):
        return geom
    minx, miny, maxx, maxy = geom.bounds
    # Rounding noise scales with the coordinate magnitude, not just the extent.
    scale = max(maxx - minx, maxy - miny, abs(minx), abs(maxx), abs(miny), abs(maxy))
    if not np.isfinite(scale) or scale <= 0.0:
        return geom
    width = 1e-9 * scale
    parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    cleaned = []
    changed = False
    for part in parts:
        keep = [r for r in part.interiors if Polygon(r).area > width * r.length]
        if len(keep) != len(part.interiors):
            changed = True
            cleaned.append(Polygon(part.exterior, keep))
        else:
            cleaned.append(part)
    if not changed:
        return geom
    if len(cleaned) == 1:
        return cleaned[0]
    from shapely.geometry import MultiPolygon

    return MultiPolygon(cleaned)


# ---------------------------------------------------------------------------
# BaseField
# ---------------------------------------------------------------------------


class BaseField:
    """Shared infrastructure for all Voronoi field backends.

    Parameters
    ----------
    arr : np.ndarray, shape (n, 2)
        Initial point positions.
    boundary : shapely.Geometry
        Outer boundary polygon (or MultiPolygon).
    adj_pairs : list of (int, int) or None
        Adjacent point index pairs for topology spring.
    boundary_mask : array-like of bool or None
        ``True`` marks centroids to snap to the outer boundary line.
    adhesion_boundary : shapely.Geometry or None
        Geometry whose boundary is used for snapping.  Defaults to
        ``boundary``.  Pass the un-simplified polygon when ``boundary``
        has been simplified for speed.
    adhesion_strength : float
        Snap strength in ``[0, 1]``.  ``1.0`` = full snap; ``0`` = off.
    weights : array-like of float or None
        Positive per-point weights.  Normalised internally.
    """

    def __init__(
        self,
        arr: np.ndarray,
        boundary,
        *,
        adj_pairs: list[tuple[int, int]] | None = None,
        intra_adj_pairs: list[tuple[int, int]] | None = None,
        boundary_mask=None,
        adhesion_boundary=None,
        adhesion_strength: float = 1.0,
        weights=None,
    ) -> None:
        if not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("arr must be a numpy array of shape (n, 2)")
        import shapely as sh

        self.points = arr.astype(float).copy()
        # Degenerate interior rings left over from the coverage union break
        # GEOS overlay at cell-clipping time (see drop_sliver_holes).
        boundary = drop_sliver_holes(boundary)
        self.boundary = boundary
        n = len(arr)
        self._cell_radius = np.sqrt(float(boundary.area) / n)
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64).ravel()
            if w.shape != (n,):
                raise ValueError(f"weights must have length {n}, got {w.shape}")
            if (w <= 0).any():
                raise ValueError("all weights must be positive")
            self._weights = w / w.mean()
        else:
            self._weights = None
        self.interior_pt = boundary.representative_point()
        sh.prepare(boundary)
        # Topology spring
        if adj_pairs:
            ij = np.asarray(adj_pairs, dtype=np.intp)
            self._adj_i: np.ndarray | None = ij[:, 0]
            self._adj_j: np.ndarray | None = ij[:, 1]
        else:
            self._adj_i = self._adj_j = None
        # Pre-compute per-edge area-derived rest-length proportionals.
        # Used when weights are available; alpha calibration is applied at call time.
        if self._adj_i is not None and self._weights is not None:
            sqrt_w = np.sqrt(self._weights)
            self._adj_rest_raw = sqrt_w[self._adj_i] + sqrt_w[self._adj_j]
        else:
            self._adj_rest_raw = None
        # Intra-group spring (separate pair set with its own rest-length precompute)
        if intra_adj_pairs:
            ij2 = np.asarray(intra_adj_pairs, dtype=np.intp)
            self._intra_adj_i: np.ndarray | None = ij2[:, 0]
            self._intra_adj_j: np.ndarray | None = ij2[:, 1]
        else:
            self._intra_adj_i = self._intra_adj_j = None
        if self._intra_adj_i is not None and self._weights is not None:
            sqrt_w = np.sqrt(self._weights)
            self._intra_adj_rest_raw = sqrt_w[self._intra_adj_i] + sqrt_w[self._intra_adj_j]
        else:
            self._intra_adj_rest_raw = None
        # Boundary adhesion
        if not (0.0 <= adhesion_strength <= 1.0):
            raise ValueError(f"adhesion_strength must be in [0, 1], got {adhesion_strength}")
        self._adhesion_strength = float(adhesion_strength)
        if boundary_mask is not None:
            self._boundary_idx: np.ndarray | None = np.where(np.asarray(boundary_mask, dtype=bool))[0]
            snap_geom = adhesion_boundary if adhesion_boundary is not None else boundary
            self._boundary_line = snap_geom.boundary
        else:
            self._boundary_idx = None
        # Current boundary (may be mutated by elastic subclasses)
        self._original_boundary = boundary
        self._current_boundary = boundary

    # ------------------------------------------------------------------
    # Internal helpers (shared)
    # ------------------------------------------------------------------

    def _constrain_points(self) -> None:
        """Hard-snap any point outside the active boundary back to its edge."""
        import shapely

        check = self._current_boundary
        inside = shapely.contains_xy(check, self.points[:, 0], self.points[:, 1])
        _parts = list(check.geoms) if check.geom_type == "MultiPolygon" else None
        for i in np.where(~inside)[0]:
            p = Point(self.points[i])
            nearest = nearest_points(check.boundary, p)[0]
            nx, ny = nearest.x, nearest.y
            if _parts is not None:
                # Use the representative point of the sub-polygon that owns the
                # snapped edge, so the nudge points inward for that component.
                ref = min(_parts, key=lambda g: nearest.distance(g)).representative_point()
            else:
                ref = self.interior_pt
            inward = np.array([ref.x - nx, ref.y - ny])
            d = np.linalg.norm(inward)
            nudge = inward / d * 1e-6 if d > 1e-10 else np.zeros(2)
            self.points[i] = np.array([nx, ny]) + nudge

    def _apply_boundary_adhesion(self) -> None:
        """Snap boundary centroids to the nearest point on the outer boundary ring."""
        if self._boundary_idx is None or self._adhesion_strength == 0.0:
            return
        s = self._adhesion_strength
        for i in self._boundary_idx:
            p = Point(self.points[i])
            nearest = nearest_points(self._boundary_line, p)[0]
            if s == 1.0:
                self.points[i] = [nearest.x, nearest.y]
            else:
                self.points[i, 0] += s * (nearest.x - self.points[i, 0])
                self.points[i, 1] += s * (nearest.y - self.points[i, 1])

    def _compute_spring(
        self,
        adj_i: np.ndarray,
        adj_j: np.ndarray,
        rest_raw: np.ndarray | None,
        weight: float,
    ) -> np.ndarray:
        """One-sided tension spring for a set of adjacent pairs.

        Uses per-edge area-derived rest lengths when *rest_raw* is provided
        (weights available), otherwise falls back to a global median rest length.
        """
        n = len(self.points)
        vecs = self.points[adj_j] - self.points[adj_i]
        dist = np.linalg.norm(vecs, axis=1, keepdims=True)
        dist = np.maximum(dist, 1e-10)
        rest: np.ndarray | float
        if rest_raw is not None:
            alpha = float(dist.mean()) / float(rest_raw.mean())
            rest = (alpha * rest_raw)[:, None]
        else:
            rest = float(np.median(dist))
        stretch = np.maximum(dist - rest, 0.0) / dist
        force = stretch * vecs
        delta = np.zeros_like(self.points)
        np.add.at(delta, adj_i, force)
        np.add.at(delta, adj_j, -force)
        degree = np.bincount(np.concatenate([adj_i, adj_j]), minlength=n).clip(min=1)
        delta /= degree[:, None]
        return weight * delta

    def _topology_spring(self, weight: float, intra_weight: float = 0.0) -> np.ndarray:
        """Compute adjacency spring displacement (cross-group + intra-group)."""
        result = np.zeros_like(self.points)
        if self._adj_i is not None and weight != 0.0:
            result += self._compute_spring(self._adj_i, cast(np.ndarray, self._adj_j), self._adj_rest_raw, weight)
        if self._intra_adj_i is not None and intra_weight != 0.0:
            result += self._compute_spring(
                self._intra_adj_i, cast(np.ndarray, self._intra_adj_j), self._intra_adj_rest_raw, intra_weight
            )
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_points(self) -> np.ndarray:
        """Return current point positions, shape (n, 2)."""
        return self.points.copy()

    def area_cv(self) -> float:
        """Coefficient of variation of cell areas (std / mean).

        Raises ``RuntimeError`` if called before the first ``relax()``.
        """
        raise NotImplementedError

    def relax(self, **kwargs) -> None:
        raise NotImplementedError

    def get_cells(self) -> np.ndarray:
        """Return clipped Voronoi cell polygons, shape (n,)."""
        raise NotImplementedError

    def get_cells_for_contiguity(self) -> np.ndarray:
        """Return cells for mid-loop adjacency/contiguity checks.

        Defaults to ``get_cells()``.  Raster fields override this to use
        computation-resolution raster cells (faster; topology-equivalent).
        """
        return self.get_cells()

    def _on_points_changed(self) -> None:
        """Called after generator positions are modified externally (e.g. repair swap).

        Subclasses that cache derived state from ``self.points`` must override
        this to rebuild that cache.
        """
        pass


# ---------------------------------------------------------------------------
# Shared geometry normalisation
# ---------------------------------------------------------------------------


def _keep_polygonal(geom):
    """Reduce *geom* to its polygonal parts.

    Clipping a Voronoi cell to the boundary can yield a ``GeometryCollection``
    with zero-area line/point parts, or collapse the cell entirely to a
    ``LineString``/``MultiLineString``. Non-polygonal geometry breaks
    downstream consumers (``shapely.boundary`` returns None/empty, and
    ``shapely.coverage_simplify`` raises), so keep the polygonal parts and
    return an empty polygon when there are none. Polygonal input is returned
    unchanged.
    """
    import shapely as sh

    geom_type = geom.geom_type
    if geom_type in ("Polygon", "MultiPolygon"):
        return geom
    if sh.is_empty(geom):
        return sh.Polygon()
    parts = [g for g in getattr(geom, "geoms", ()) if g.geom_type in ("Polygon", "MultiPolygon")]
    return sh.union_all(parts) if parts else sh.Polygon()


# ---------------------------------------------------------------------------
# Shared exact-cell extraction (used by both ExactField and RasterField)
# ---------------------------------------------------------------------------


def _extract_exact_cells(points: np.ndarray, boundary) -> np.ndarray:
    """Compute exact Voronoi cells from *points* clipped to *boundary*.

    Implements the same algorithm as :meth:`ExactField.get_cells` but as a
    standalone function so that :class:`RasterField` can reuse it for
    high-quality final cell extraction without changing the iterative
    labeling used during relaxation.

    Parameters
    ----------
    points : ndarray, shape (G, 2)
        Generator positions.
    boundary : shapely Geometry
        Clip polygon (may be the original or a deformed elastic boundary).

    Returns
    -------
    ndarray of shapely Geometry, shape (G,)
        Clipped Voronoi cell polygons.
    """
    import shapely as sh

    minx, miny, maxx, maxy = boundary.bounds
    far = max(maxx - minx, maxy - miny) * 10
    ghosts = np.array([
        [minx - far, miny - far],
        [maxx + far, miny - far],
        [maxx + far, maxy + far],
        [minx - far, maxy + far],
    ])
    all_pts = np.vstack([points, ghosts])
    vor = Voronoi(all_pts, qhull_options="Qbb Qc Qx")
    verts_2d = vor.vertices

    G = len(points)
    cell_polys = np.empty(G, dtype=object)
    for i in range(G):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        verts = [v for v in region if v != -1]
        if len(verts) < 3:
            cell_polys[i] = Point(points[i])
            continue
        try:
            coords = verts_2d[verts]
            cx, cy = coords.mean(axis=0)
            order = np.argsort(np.arctan2(coords[:, 1] - cy, coords[:, 0] - cx))
            cell_polys[i] = Polygon(coords[order])
        except Exception:
            cell_polys[i] = Point(points[i])

    # Clip to boundary
    clip_geom = boundary
    boundary_area = clip_geom.area
    oversized = sh.area(cell_polys) > boundary_area
    if oversized.any():
        cell_polys = cell_polys.copy()
        bbox = clip_geom.envelope
        cell_polys[oversized] = sh.intersection(cell_polys[oversized], bbox)
    inside = sh.within(cell_polys, clip_geom)
    clipped = cell_polys.copy()
    need_clip = ~inside
    if need_clip.any():
        clipped[need_clip] = sh.intersection(cell_polys[need_clip], clip_geom)
    clipped[:] = [_keep_polygonal(c) for c in clipped]
    bad = sh.is_empty(clipped) | (sh.area(clipped) == 0.0)
    if bad.any():
        radius = np.sqrt(float(clip_geom.area) / len(points)) * 0.5
        approx = sh.intersection(sh.buffer(sh.points(points[bad]), radius), clip_geom)
        clipped[bad] = np.array([_keep_polygonal(a) for a in np.atleast_1d(approx)], dtype=object)
    return clipped
