"""Exact scipy Voronoi + shapely clipping field."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Point, Polygon

from ._base import BaseField, _extract_exact_cells, _keep_polygonal


class ExactField(BaseField):
    """Exact scipy Voronoi + shapely clipping (Lloyd relaxation).

    Parameters
    ----------
    arr, boundary, adj_pairs, boundary_mask, adhesion_boundary,
    adhesion_strength, weights
        See :class:`BaseField`.
    """

    def __init__(
        self,
        arr: np.ndarray,
        boundary,
        *,
        adj_pairs=None,
        intra_adj_pairs=None,
        boundary_mask=None,
        adhesion_boundary=None,
        adhesion_strength: float = 1.0,
        weights=None,
    ) -> None:
        super().__init__(
            arr,
            boundary,
            adj_pairs=adj_pairs,
            intra_adj_pairs=intra_adj_pairs,
            boundary_mask=boundary_mask,
            adhesion_boundary=adhesion_boundary,
            adhesion_strength=adhesion_strength,
            weights=weights,
        )
        if self._weights is not None:
            import warnings

            warnings.warn(
                "ExactBackend: weights are not supported and will be ignored; use RasterBackend for weighted Voronoi",
                stacklevel=4,
            )
        self._build_voronoi()

    def _build_voronoi(self) -> None:
        minx, miny, maxx, maxy = self.boundary.bounds
        far = max(maxx - minx, maxy - miny) * 10
        ghosts = np.array([
            [minx - far, miny - far],
            [maxx + far, miny - far],
            [maxx + far, maxy + far],
            [minx - far, maxy + far],
        ])
        all_pts = np.vstack([self.points, ghosts])
        self.voronoi = Voronoi(all_pts, qhull_options="Qbb Qc Qx")
        self._vor_verts_2d = self.voronoi.vertices
        self._n_real = len(self.points)

    def _build_cell_polys(self) -> np.ndarray:
        G = len(self.points)
        cell_polys = np.empty(G, dtype=object)
        for i in range(G):
            region_idx = self.voronoi.point_region[i]
            region = self.voronoi.regions[region_idx]
            verts = [v for v in region if v != -1]
            if len(verts) < 3:
                cell_polys[i] = Point(self.points[i])
                continue
            try:
                coords = self._vor_verts_2d[verts]
                cx, cy = coords.mean(axis=0)
                order = np.argsort(np.arctan2(coords[:, 1] - cy, coords[:, 0] - cx))
                cell_polys[i] = Polygon(coords[order])
            except Exception:
                cell_polys[i] = Point(self.points[i])
        return cell_polys

    def _clip_to_boundary(self, cell_polys: np.ndarray) -> np.ndarray:
        import shapely as sh

        clip_geom = self._current_boundary
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
        clipped = np.array([_keep_polygonal(c) for c in clipped], dtype=object)
        bad = sh.is_empty(clipped) | (sh.area(clipped) == 0.0)
        if bad.any():
            radius = np.sqrt(float(clip_geom.area) / len(self.points)) * 0.5
            approx = sh.intersection(sh.buffer(sh.points(self.points[bad]), radius), clip_geom)
            clipped[bad] = np.array([_keep_polygonal(a) for a in np.atleast_1d(approx)], dtype=object)
        return clipped

    def relax(
        self,
        relaxation_factor: float = 1.0,
        topology_weight: float = 0.0,
        intra_topology_weight: float = 0.0,
        **_kwargs: Any,
    ) -> None:
        """Run one exact-Voronoi Lloyd step.

        Parameters
        ----------
        relaxation_factor : float
            Over-relaxation factor (SOR).
        topology_weight : float
            Adjacency spring strength (cross-group pairs, or all pairs when
            ``intra_topology_weight=0``).
        intra_topology_weight : float
            Spring strength for intra-group pairs (0 = disabled).
        """
        import shapely as sh

        cell_polys = self._build_cell_polys()
        clipped = self._clip_to_boundary(cell_polys)
        centroids = sh.centroid(clipped)
        target = np.stack([sh.get_x(centroids), sh.get_y(centroids)], axis=1)
        current_areas = sh.area(clipped)
        self.points = (
            self.points
            + relaxation_factor * (target - self.points)
            + self._topology_spring(topology_weight, intra_topology_weight)
        )
        self._apply_boundary_adhesion()
        self._constrain_points()
        self._build_voronoi()
        self._last_areas = current_areas

    def area_cv(self) -> float:
        if not hasattr(self, "_last_areas"):
            raise RuntimeError("call relax() at least once before area_cv()")
        areas = self._last_areas
        mean = float(np.mean(areas))
        return 0.0 if mean == 0 else float(np.std(areas) / mean)

    def get_cells(self) -> np.ndarray:
        return _extract_exact_cells(self.points, self._current_boundary)

    def _on_points_changed(self) -> None:
        # Rebuild the cached scipy Voronoi structure so the next relax() call
        # uses the updated generator positions.
        self._build_voronoi()
