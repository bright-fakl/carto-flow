"""Raster nearest-neighbour Lloyd relaxation field."""

from __future__ import annotations

import math
import warnings
from typing import Any, cast

import numpy as np
from shapely.errors import ShapelyError
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from ._base import BaseField, _extract_exact_cells, _keep_polygonal, drop_sliver_holes

# Keep every power-diagram cell non-empty when extracting the final cells
# (see ``RasterField._nonempty_offsets``).  Module-level constants rather than
# options: set to False to restore the unguarded solver.
ENSURE_NONEMPTY_POWER_CELLS = True

# Apply the same guard during relaxation, to seeds that lost all their pixels.
# Off by default: on top of the guard above it removes further degenerate cells
# at coarse resolutions (districts + 5000 m: 46 -> 35 at res 32, 9 -> 7 at 48,
# 1 -> 0 at 128) but it perturbs the Lloyd trajectory, and at res 256 it raised
# the satellite-component count from 16 to 19.  See the PR for the measurement.
ENSURE_NONEMPTY_POWER_CELLS_IN_RELAXATION = False


class RasterField(BaseField):
    """Raster nearest-neighbour Lloyd relaxation.

    Parameters
    ----------
    arr, boundary
        See :class:`BaseField`.
    resolution : int
        Grid resolution (longest dimension).
    boundary_elasticity : float
        Elastic boundary deformation strength ``[0, 1]``.  ``0`` = rigid
        hard-clip.  ``> 0`` = the boundary is deformed each iteration by an
        FFT velocity field driven by cell-area pressure.  Setting
        ``boundary_elasticity > 0`` with ``relaxation_factor=0`` and
        ``relaxation_factor_min=0`` disables Lloyd relaxation entirely,
        reproducing the pure FFT-flow algorithm.
    boundary_step_scale : float
        Boundary advection speed as a fraction of ``boundary_elasticity``,
        in ``(0, 1]``.
    density_smooth : float or None
        Gaussian sigma for density field smoothing (requires
        ``boundary_elasticity > 0``).
    labeling : {"euclidean", "geodesic"}
        Pixel labeling algorithm.
    area_eq_weight : float
        Power-diagram area equalizer learning rate.
    debug_geodesic : bool
        When ``True`` and ``labeling="geodesic"``, emit warnings for
        misplaced BFS seeds (cross-water labeling diagnostics).
    adj_pairs, boundary_mask, adhesion_boundary, adhesion_strength, weights
        See :class:`BaseField`.
    """

    def __init__(
        self,
        arr: np.ndarray,
        boundary,
        *,
        resolution: int = 300,
        final_resolution: int | None = None,
        boundary_elasticity: float = 0.0,
        boundary_step_scale: float = 1.0,
        density_smooth: float | None = None,
        min_boundary_points: int | None = None,
        labeling: str = "euclidean",
        area_eq_weight: float = 0.0,
        weight_ramp_iters: int = 0,
        debug_geodesic: bool = False,
        adj_pairs=None,
        intra_adj_pairs=None,
        boundary_mask=None,
        adhesion_boundary=None,
        adhesion_strength: float = 1.0,
        weights=None,
    ) -> None:
        BaseField.__init__(
            self,
            arr,
            boundary,
            adj_pairs=adj_pairs,
            intra_adj_pairs=intra_adj_pairs,
            boundary_mask=boundary_mask,
            adhesion_boundary=adhesion_boundary,
            adhesion_strength=adhesion_strength,
            weights=weights,
        )
        self._raster_resolution = resolution
        self._final_resolution = final_resolution
        self._boundary_elasticity = float(boundary_elasticity)
        self._boundary_step_scale = float(boundary_step_scale)
        self._density_smooth = float(density_smooth) if density_smooth else None
        self._elastic_densify_verts = int(min_boundary_points) if min_boundary_points is not None else None
        self._geodesic_voronoi = labeling == "geodesic"
        self._area_eq_weight = float(area_eq_weight)
        self._weight_ramp_iters = int(weight_ramp_iters)
        self._power_offsets = np.zeros(len(arr), dtype=np.float64)
        self._debug_geodesic = bool(debug_geodesic)
        if self._geodesic_voronoi and self._weights is not None:
            import warnings

            warnings.warn(
                "RasterBackend: labeling='geodesic' does not support weights; "
                "weights will be ignored (use labeling='euclidean' with "
                "area_eq_weight for weighted area-equalised Voronoi)",
                stacklevel=4,
            )

        if self._boundary_elasticity > 0.0:
            self._init_elastic_state(boundary)
        self._precompute_grid()

    # -- dt helpers ---------------------------------------------------------

    def _boundary_dt(self) -> float:
        return self._boundary_elasticity * self._boundary_step_scale * self._cell_radius

    def _centroid_dt(self) -> float:
        """Co-advect centroids at the same rate as the boundary.

        When ``boundary_elasticity > 0``, the elastic boundary deforms each
        iteration via an FFT velocity field.  Without centroid co-advection the
        boundary can pinch inward and split a cell's active-pixel territory into
        disconnected components.  Moving centroids with the boundary keeps them
        inside their territory; Lloyd then refines positions toward pixel
        centroids as usual.
        """
        return self._boundary_elasticity * self._cell_radius

    # -- Grid setup ---------------------------------------------------------

    def _precompute_grid(self) -> None:
        import shapely as sh

        res = self._raster_resolution
        elastic = self._boundary_elasticity > 0.0
        minx, miny, maxx, maxy = self._original_boundary.bounds
        if elastic:
            pad_x = 0.5 * (maxx - minx)
            pad_y = 0.5 * (maxy - miny)
            minx -= pad_x
            maxx += pad_x
            miny -= pad_y
            maxy += pad_y
        width = maxx - minx
        height = maxy - miny
        if width >= height:
            nx = res
            ny = max(1, round(res * height / width))
        else:
            ny = res
            nx = max(1, round(res * width / height))
        xs = np.linspace(minx, maxx, nx)
        ys = np.linspace(miny, maxy, ny)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        if elastic:
            x_flat = pts[:, 0].astype(np.float32)
            y_flat = pts[:, 1].astype(np.float32)
            self._elastic_active_mask = sh.contains_xy(self._original_boundary, x_flat, y_flat)
        else:
            inside = sh.contains_xy(self.boundary, pts[:, 0], pts[:, 1])
            x_flat = pts[inside, 0].astype(np.float32)
            y_flat = pts[inside, 1].astype(np.float32)
            self._initial_active_mask = inside
        self._grid_pts_x = x_flat
        self._grid_pts_y = y_flat
        self._grid_pts_f32 = np.column_stack([x_flat, y_flat])
        self._pixel_area = float(width * height) / (nx * ny)
        self._grid_nx = nx
        self._grid_ny = ny
        self._grid_dx = float(xs[1] - xs[0]) if nx > 1 else float(width)
        self._grid_dy = float(ys[1] - ys[0]) if ny > 1 else float(height)
        self._grid_x_coords = xs.copy()
        self._grid_y_coords = ys.copy()
        if elastic:
            self._vel_computer = self._make_vel_computer()

    def _make_vel_computer(self):
        try:
            from carto_flow.flow_cartogram.velocity import VelocityComputerFFTW

            nx, ny = self._grid_nx, self._grid_ny
            dx, dy = self._grid_dx, self._grid_dy

            class _G:
                sx = nx
                sy = ny
                pass

            g = _G()
            g.dx = dx  # type: ignore[attr-defined]
            g.dy = dy  # type: ignore[attr-defined]
            return VelocityComputerFFTW(g)  # type: ignore[arg-type]
        except (ImportError, Exception):
            return None

    def _init_elastic_state(self, boundary) -> None:
        # Simplify boundary to grid resolution before storing vertices.
        # Sub-pixel detail is meaningless for elastic deformation and causes
        # expensive make_valid + unary_union calls each iteration.
        minx, miny, maxx, maxy = boundary.bounds
        extent = max(maxx - minx, maxy - miny)
        tol = extent / self._raster_resolution * 0.5
        boundary = boundary.simplify(tol, preserve_topology=True)
        # Optional: densify to a caller-specified minimum point count.  Must
        # happen after simplify so the new points are not immediately stripped.
        # Needed for simple shapes (bbox, circle, convex hull) whose sides are
        # straight and end up with very few vertices after simplification.
        if self._elastic_densify_verts is not None:
            import shapely as _sh

            seg_len = boundary.length / self._elastic_densify_verts
            boundary = _sh.segmentize(boundary, seg_len)

        def _ring_state(poly):
            return np.array(poly.exterior.coords)[:-1]

        self._elastic_verts: list | None = None
        self._elastic_current_verts: list | None = None
        if boundary.geom_type == "Polygon":
            v = _ring_state(boundary)
            self._elastic_verts = [v]
            self._elastic_current_verts = [v.copy()]
        elif boundary.geom_type == "MultiPolygon":
            self._elastic_verts, self._elastic_current_verts = [], []
            for poly in boundary.geoms:
                v = _ring_state(poly)
                self._elastic_verts.append(v)
                self._elastic_current_verts.append(v.copy())

    # -- Boundary deformation -----------------------------------------------

    def _deform_boundary(self) -> None:
        """Deform boundary vertices (and optionally centroids) via FFT velocity.

        No-op when ``_boundary_dt() == 0`` and ``_centroid_dt() == 0``,
        or when ``_elastic_verts`` is not initialised, or before the first
        raster labeling step (``_last_labels_2d`` not set).
        """
        bdt = self._boundary_dt()
        cdt = self._centroid_dt()
        if bdt == 0.0 and cdt == 0.0:
            return
        if getattr(self, "_elastic_verts", None) is None:
            return
        if not hasattr(self, "_last_labels_2d"):
            return

        import shapely as sh

        from carto_flow.flow_cartogram.displacement import displace_coords_numba

        G = len(self.points)
        mask = self._elastic_active_mask

        # Re-compute true Voronoi areas (without power offsets) for density field.
        gx_active = self._grid_pts_x[mask]
        gy_active = self._grid_pts_y[mask]
        m_active = int(mask.sum())
        if self._geodesic_voronoi:
            from ..geodesic import geodesic_label_active

            _result = geodesic_label_active(
                self.points,
                gx_active,
                gy_active,
                mask,
                self._grid_nx,
                self._grid_ny,
                return_debug=self._debug_geodesic,
                grid_x_coords=self._grid_x_coords,
                grid_y_coords=self._grid_y_coords,
            )
            if self._debug_geodesic:
                labels_true, _di = _result
                self._debug_geodesic_last = _di
                if _di["n_misplaced"] > 0:
                    import warnings

                    warnings.warn(
                        f"geodesic labeling (_deform_boundary): "
                        f"{_di['n_misplaced']} seed(s) placed in wrong "
                        f"connected component (cross-water labeling likely)",
                        stacklevel=3,
                    )
            else:
                labels_true = _result
        elif self._weights is not None:
            pts32 = self.points.astype(np.float32)
            w32 = self._weights.astype(np.float32)
            chunk = 4096
            labels_true = np.empty(m_active, dtype=np.intp)
            for start in range(0, m_active, chunk):
                end = min(start + chunk, m_active)
                dx = gx_active[start:end, None] - pts32[None, :, 0]
                dy = gy_active[start:end, None] - pts32[None, :, 1]
                labels_true[start:end] = ((dx * dx + dy * dy) / w32[None, :]).argmin(axis=1)
        else:
            from scipy.spatial import cKDTree

            gf32_active = self._grid_pts_f32[mask]
            _, labels_true = cKDTree(self.points).query(gf32_active, workers=-1)

        counts_true = np.bincount(labels_true, minlength=G).astype(float)
        areas = counts_true * self._pixel_area

        total_area = float(self.boundary.area)
        use_weights = self._weights is not None and not self._geodesic_voronoi
        target_density = float(self._weights.sum()) / total_area if use_weights else G / total_area

        min_area = self._pixel_area
        density = self._weights / np.maximum(areas, min_area) if use_weights else 1.0 / np.maximum(areas, min_area)

        full_labels_true = np.full(self._grid_nx * self._grid_ny, -1, dtype=np.intp)
        active_idx = np.where(mask)[0]
        full_labels_true[active_idx] = labels_true
        labels_2d_true = full_labels_true.reshape(self._grid_ny, self._grid_nx)
        rho_2d = np.full((self._grid_ny, self._grid_nx), target_density, dtype=np.float64)
        valid_pix = labels_2d_true >= 0
        rho_2d[valid_pix] = density[labels_2d_true[valid_pix]]

        if self._density_smooth:
            from scipy.ndimage import gaussian_filter

            mu = rho_2d.mean()
            rho_2d = gaussian_filter(rho_2d, sigma=self._density_smooth)
            rho_2d *= mu / rho_2d.mean()

        if getattr(self, "_vel_computer", None) is not None:
            vx, vy = self._vel_computer.compute(rho_2d)
        else:
            from carto_flow.flow_cartogram.velocity import compute_velocity_anisotropic_rfft

            class _G:
                dx = self._grid_dx
                dy = self._grid_dy

            vx, vy = compute_velocity_anisotropic_rfft(rho_2d, _G())  # type: ignore[arg-type]

        vmax = float(np.nanmax(np.sqrt(vx**2 + vy**2)))
        if vmax > 1e-12:
            vx = vx / vmax
            vy = vy / vmax

        # Advect boundary vertices
        _elastic_current_verts = cast(list, self._elastic_current_verts)
        if bdt > 0.0:
            for i, cur_verts in enumerate(_elastic_current_verts):
                _elastic_current_verts[i] = displace_coords_numba(
                    cur_verts,
                    self._grid_x_coords,
                    self._grid_y_coords,
                    vx,
                    vy,
                    bdt,
                    self._grid_dx,
                    self._grid_dy,
                )

        # Advect centroids
        if cdt > 0.0:
            self.points = displace_coords_numba(
                self.points,
                self._grid_x_coords,
                self._grid_y_coords,
                vx,
                vy,
                cdt,
                self._grid_dx,
                self._grid_dy,
            )

        # Debug state
        orig_cat = np.vstack(cast(list, self._elastic_verts))
        cur_cat = np.vstack(_elastic_current_verts)
        self._debug_area_pressure = density / target_density - 1.0
        self._debug_rho = rho_2d.copy()
        self._debug_vx = vx.copy()
        self._debug_vy = vy.copy()
        self._debug_vertex_xy = cur_cat.copy()
        self._debug_vertex_disp = np.linalg.norm(cur_cat - orig_cat, axis=1)
        self._debug_vertex_disp_vec = cur_cat - orig_cat

        # Rebuild boundary from displaced vertices.
        # Fast path: skip make_valid + unary_union when all polygons are already
        # valid (common for small displacements with boundary_elasticity << 1).
        new_polys = [Polygon(np.vstack([v, v[:1]])) for v in _elastic_current_verts]
        if len(new_polys) == 1 and sh.is_valid(new_polys[0]):
            new_geom = new_polys[0]
        elif len(new_polys) > 1 and all(sh.is_valid(p) for p in new_polys):
            new_geom = unary_union(new_polys)
        else:
            # Slow path: fix self-intersections introduced by displacement.
            valid_parts: list = []
            for p in new_polys:
                vp = sh.make_valid(p)
                if vp.geom_type == "Polygon":
                    valid_parts.append(vp)
                else:
                    polys = [g for g in getattr(vp, "geoms", [vp]) if g.geom_type == "Polygon"]
                    valid_parts.extend(polys)
            new_geom = unary_union(valid_parts) if valid_parts else self._current_boundary
        self._current_boundary = drop_sliver_holes(new_geom)
        sh.prepare(self._current_boundary)
        self._elastic_active_mask = sh.contains_xy(self._current_boundary, self._grid_pts_x, self._grid_pts_y)
        # Keep adhesion snap target aligned with the deformed boundary so that
        # _apply_boundary_adhesion() snaps centroids to the current shape.
        if self._adhesion_strength > 0.0 and self._boundary_idx is not None:
            self._boundary_line = self._current_boundary.boundary

    # -- Cell construction helpers ------------------------------------------

    def _fill_label_flat(
        self,
        labels_active: np.ndarray,
        active_mask: np.ndarray,
    ) -> np.ndarray:
        """Build a full (nx*ny,) int32 label array from labels for active pixels.

        Inactive (water/outside) pixels are filled by nearest-centroid NN so
        that the returned array can be reshaped to (ny, nx) and passed directly
        to ``_label_2d_to_cell_polys``.
        """
        from scipy.spatial import cKDTree

        nx, ny = self._grid_nx, self._grid_ny
        x_coords = self._grid_x_coords
        y_coords = self._grid_y_coords
        label_flat = np.full(nx * ny, -1, dtype=np.int32)
        label_flat[np.where(active_mask)[0]] = labels_active
        inactive_flat = np.where(~active_mask)[0]
        if len(inactive_flat) > 0:
            inactive_x = x_coords[inactive_flat % nx].astype(np.float64)
            inactive_y = y_coords[inactive_flat // nx].astype(np.float64)
            _, nn = cKDTree(self.points).query(np.column_stack([inactive_x, inactive_y]))
            label_flat[inactive_flat] = nn.astype(np.int32)
        return label_flat

    def _label_2d_to_cell_polys(
        self,
        label_2d: np.ndarray,
        *,
        nx: int | None = None,
        ny: int | None = None,
        dx: float | None = None,
        dy: float | None = None,
        x_coords=None,
        y_coords=None,
        boundary=None,
    ) -> np.ndarray:
        """Convert a (ny, nx) label grid to gap-free cell polygons via polygonize.

        Detects label-boundary transitions in the grid (horizontal and vertical
        edges between differently-labeled pixels), converts them to line
        segments, and uses ``shapely.ops.polygonize`` to close them into
        water-tight cell polygons.

        After polygon construction, ``shapely.coverage_simplify`` is applied
        with a ``sqrt(dx * dy / 2)`` distance tolerance (about 0.7 pixel) and
        ``simplify_boundary=False`` to smooth out pixel staircases while
        preserving the outer coverage boundary.

        Optional keyword overrides (*nx*, *ny*, *dx*, *dy*, *x_coords*,
        *y_coords*, *boundary*) replace the corresponding ``self._grid_*``
        attributes, allowing callers to supply a different grid without
        modifying field state (e.g. for upsampled final-cell extraction).
        """
        import shapely as sh
        from shapely.ops import polygonize
        from shapely.ops import unary_union as ops_unary_union

        G = len(self.points)
        nx = nx if nx is not None else self._grid_nx
        ny = ny if ny is not None else self._grid_ny
        dx = dx if dx is not None else self._grid_dx
        dy = dy if dy is not None else self._grid_dy
        x_coords = x_coords if x_coords is not None else self._grid_x_coords
        y_coords = y_coords if y_coords is not None else self._grid_y_coords
        boundary = boundary if boundary is not None else self._current_boundary
        half_dx = dx / 2.0
        half_dy = dy / 2.0
        # A cell that clips to less than a millionth of a pixel is not a cell.
        min_cell_area = 1e-6 * dx * dy
        grid_size = 1e-6 * min(dx, dy)

        x_edges = np.empty(nx + 1)
        x_edges[0] = x_coords[0] - half_dx
        x_edges[1:] = x_coords + half_dx
        y_edges = np.empty(ny + 1)
        y_edges[0] = y_coords[0] - half_dy
        y_edges[1:] = y_coords + half_dy

        cell_polys = np.empty(G, dtype=object)
        for i in range(G):
            mask = label_2d == i
            if not mask.any():
                cell_polys[i] = Point(self.points[i])
                continue
            MP = np.zeros((ny + 2, nx + 2), dtype=bool)
            MP[1:-1, 1:-1] = mask
            h = MP[:-1, 1:-1] != MP[1:, 1:-1]
            h_r, h_c = np.where(h)
            v = MP[1:-1, :-1] != MP[1:-1, 1:]
            v_r, v_c = np.where(v)
            h_coords = np.stack(
                [
                    np.column_stack([x_edges[h_c], y_edges[h_r]]),
                    np.column_stack([x_edges[h_c + 1], y_edges[h_r]]),
                ],
                axis=1,
            )
            v_coords = np.stack(
                [
                    np.column_stack([x_edges[v_c], y_edges[v_r]]),
                    np.column_stack([x_edges[v_c], y_edges[v_r + 1]]),
                ],
                axis=1,
            )
            lines = sh.linestrings(np.concatenate([h_coords, v_coords], axis=0))
            polys = list(polygonize(lines))
            if not polys:
                cell_polys[i] = Point(self.points[i])
                continue
            poly = polys[0] if len(polys) == 1 else ops_unary_union(polys)
            clipped = sh.intersection(poly, boundary)
            # Clipping along the boundary can leave zero-area line/point parts,
            # or collapse the cell to a bare Line/MultiLineString; keep only the
            # polygonal parts so downstream code (shapely.boundary in
            # find_adjacent_pairs, coverage_simplify below) sees a Polygon.
            clipped = _keep_polygonal(clipped)
            if clipped.area <= min_cell_area < poly.area:
                # The clip lost a cell that owns pixels.  This should not happen
                # now that the boundary is cleaned of degenerate rings (see
                # drop_sliver_holes), but never report a phantom point cell for
                # a cell that really is inside the boundary: retry on a coarse
                # precision grid, and otherwise keep the raw pixel union.
                # (A cell whose pixels genuinely lie outside the boundary -- the
                # NN fill labels those too -- is left as the empty clip.)
                fallback = how = None
                retry = _keep_polygonal(
                    sh.intersection(sh.set_precision(poly, grid_size), sh.set_precision(boundary, grid_size))
                )
                if retry.area > min_cell_area:
                    fallback, how = retry, "reduced-precision clip"
                elif sh.covers(boundary, poly):
                    fallback, how = poly, "raw pixel union"
                if fallback is not None:
                    warnings.warn(
                        f"raster cell extraction failed for seed {i} ({int(mask.sum())} pixel(s)): "
                        f"clipping to the boundary collapsed the cell to a zero-area "
                        f"{clipped.geom_type.lower()}; falling back to the {how}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    clipped = fallback
            cell_polys[i] = clipped if not sh.is_empty(clipped) else Point(self.points[i])

        # Smooth pixel staircases: coverage_simplify on shared interior edges only.
        # `tol` is a distance tolerance in shapely (max vertex displacement),
        # not an area: use sqrt(dx*dy/2) (~0.7 px) rather than dx*dy/2, which
        # would be ~1e8x too large at typical grid resolutions and over-smooths.
        tol = math.sqrt(dx * dy / 2.0)
        valid = np.array([c.geom_type in ("Polygon", "MultiPolygon") for c in cell_polys])
        if valid.any():
            try:
                cell_polys[valid] = sh.coverage_simplify(cell_polys[valid], tol, simplify_boundary=False)
            except (ShapelyError, ValueError) as exc:  # degenerate coverage
                warnings.warn(
                    f"coverage_simplify failed; cell boundaries are left unsmoothed ({exc})",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return cell_polys

    def _get_weighted_cells_from_grid(self) -> np.ndarray:
        gx, gy = self._grid_pts_x, self._grid_pts_y
        pts32 = self.points.astype(np.float32)
        w32 = self._weights.astype(np.float32)
        m = len(gx)
        labels = np.empty(m, dtype=np.intp)
        chunk = 4096
        for start in range(0, m, chunk):
            end = min(start + chunk, m)
            dx = gx[start:end, None] - pts32[None, :, 0]
            dy = gy[start:end, None] - pts32[None, :, 1]
            labels[start:end] = ((dx * dx + dy * dy) / w32[None, :]).argmin(axis=1)

        # For elastic grids, _grid_pts_x covers all nx*ny pixels (no fill needed).
        # For rigid grids, _grid_pts_x covers only active pixels; fill the rest.
        if self._boundary_elasticity > 0.0:
            label_flat = labels.astype(np.int32)
        else:
            label_flat = self._fill_label_flat(labels.astype(np.int32), self._initial_active_mask)
        return self._label_2d_to_cell_polys(label_flat.reshape(self._grid_ny, self._grid_nx))

    def _get_geodesic_raster_cells(self) -> np.ndarray:
        from ..geodesic import geodesic_label_active

        if self._boundary_elasticity > 0.0:
            active = self._elastic_active_mask
            gx = self._grid_pts_x[active]
            gy = self._grid_pts_y[active]
        else:
            active = self._initial_active_mask
            gx = self._grid_pts_x
            gy = self._grid_pts_y

        _result = geodesic_label_active(
            self.points,
            gx,
            gy,
            active,
            self._grid_nx,
            self._grid_ny,
            return_debug=self._debug_geodesic,
            grid_x_coords=self._grid_x_coords,
            grid_y_coords=self._grid_y_coords,
        )
        if self._debug_geodesic:
            labels, _di = _result
            self._debug_geodesic_last = _di
            if _di["n_misplaced"] > 0:
                import warnings

                warnings.warn(
                    f"geodesic labeling (get_cells): "
                    f"{_di['n_misplaced']} seed(s) placed in wrong "
                    f"connected component (cross-water labeling likely)",
                    stacklevel=3,
                )
        else:
            labels = _result
        label_flat = self._fill_label_flat(labels.astype(np.int32), active)
        return self._label_2d_to_cell_polys(label_flat.reshape(self._grid_ny, self._grid_nx))

    def _build_cells_upsampled(self, res: int) -> np.ndarray:
        """Re-run NN (or geodesic) labeling at resolution *res* and return cell polys.

        Builds a fresh grid from the current boundary at the requested
        resolution, labels every pixel to its nearest generator, converts the
        label grid to polygons, and simplifies pixel staircases — without
        modifying any field state.
        """
        import shapely as sh
        from scipy.spatial import cKDTree

        boundary = self._current_boundary
        minx, miny, maxx, maxy = boundary.bounds
        width = maxx - minx
        height = maxy - miny
        if width >= height:
            nx = res
            ny = max(1, round(res * height / width))
        else:
            ny = res
            nx = max(1, round(res * width / height))
        xs = np.linspace(minx, maxx, nx)
        ys = np.linspace(miny, maxy, ny)
        gx_grid, gy_grid = np.meshgrid(xs, ys)  # (ny, nx) each
        px = gx_grid.ravel().astype(np.float32)
        py = gy_grid.ravel().astype(np.float32)
        active = sh.contains_xy(boundary, px, py)

        if self._geodesic_voronoi:
            from ..geodesic import geodesic_label_active

            _result = geodesic_label_active(
                self.points,
                px[active].astype(np.float64),
                py[active].astype(np.float64),
                active,
                nx,
                ny,
                return_debug=False,
                grid_x_coords=xs,
                grid_y_coords=ys,
            )
            labels_active = np.asarray(_result, dtype=np.int32)
        else:
            gx_active = px[active].astype(np.float32)
            gy_active = py[active].astype(np.float32)
            pts32 = self.points.astype(np.float32)
            use_weights = False  # weights affect target_counts only, not distance formula
            use_offsets = self._area_eq_weight > 0.0 and self._power_offsets is not None

            if use_weights or use_offsets:
                m_active = int(active.sum())
                chunk = 4096
                labels_active = np.empty(m_active, dtype=np.int32)
                w32 = self._weights.astype(np.float32) if use_weights else None
                lam = self._nonempty_offsets() if ENSURE_NONEMPTY_POWER_CELLS else self._power_offsets
                lam32 = lam.astype(np.float32) if use_offsets else None
                for start in range(0, m_active, chunk):
                    end = min(start + chunk, m_active)
                    dx = gx_active[start:end, None] - pts32[None, :, 0]
                    dy = gy_active[start:end, None] - pts32[None, :, 1]
                    dist = dx * dx + dy * dy
                    if use_weights:
                        dist = dist / cast(np.ndarray, w32)[None, :]
                    if use_offsets:
                        dist = dist - cast(np.ndarray, lam32)[None, :]
                    labels_active[start:end] = dist.argmin(axis=1).astype(np.int32)
            else:
                pts_active = np.column_stack([gx_active.astype(np.float64), gy_active.astype(np.float64)])
                _, labels_nn = cKDTree(self.points).query(pts_active, workers=-1)
                labels_active = labels_nn.astype(np.int32)

        label_flat = np.zeros(nx * ny, dtype=np.int32)
        label_flat[np.where(active)[0]] = labels_active
        # Fill inactive pixels with nearest generator for a gapless label grid
        inactive_idx = np.where(~active)[0]
        if len(inactive_idx):
            inactive_pts = np.column_stack([
                px[inactive_idx].astype(np.float64),
                py[inactive_idx].astype(np.float64),
            ])
            _, nn = cKDTree(self.points).query(inactive_pts)
            label_flat[inactive_idx] = nn.astype(np.int32)

        dx = float(xs[1] - xs[0]) if nx > 1 else float(width)
        dy = float(ys[1] - ys[0]) if ny > 1 else float(height)
        return self._label_2d_to_cell_polys(
            label_flat.reshape(ny, nx),
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            x_coords=xs,
            y_coords=ys,
            boundary=boundary,
        )

    def get_cells_for_contiguity(self) -> np.ndarray:
        # Use computation-resolution raster cells: faster than exact Voronoi,
        # and correctly reflects the topology used during Lloyd relaxation.
        return self._build_cells_upsampled(self._raster_resolution)

    # -- Power-offset guard -------------------------------------------------

    def _nonempty_offsets(self, max_passes: int = 5) -> np.ndarray:
        """Raise power offsets until every seed owns at least its own position.

        A seed's power cell is ``{x : |x - p_i|^2 - lambda_i <= |x - p_j|^2 - lambda_j}``.
        When ``lambda_i`` falls far enough below a neighbour's, that set becomes
        empty: the cell degrades to a Point, and because a pixel-less seed takes
        its own position as the Lloyd target, it is frozen there and can never
        win territory back.  ``lambda_i >= max_j(lambda_j - d_ij^2)`` is the
        weakest condition that keeps ``p_i`` itself inside cell *i*, so clamping
        the offsets from below by that value is enough to keep every power cell
        non-empty.  Raising one offset raises other seeds' floors, hence the
        (cheap, usually single-pass) repetition.
        """
        pts = self.points
        sq = np.einsum("ij,ij->i", pts, pts)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T)
        np.fill_diagonal(d2, np.inf)
        offsets = self._power_offsets
        for _ in range(max_passes):
            floor = (offsets[None, :] - d2).max(axis=1)
            raised = np.maximum(offsets, floor)
            if np.array_equal(raised, offsets):
                break
            offsets = raised
        return offsets

    # -- Lloyd step ---------------------------------------------------------

    def _relax_raster(
        self,
        relaxation_factor: float,
        topology_weight: float,
        area_eq_weight: float = 0.0,
        iteration: int = 0,
        intra_topology_weight: float = 0.0,
    ) -> None:
        from scipy.spatial import cKDTree

        # Deform boundary first so that labeling and centroid updates use the
        # updated active region.  This eliminates the boundary-Lloyd ordering
        # mismatch that caused satellite pixel regions (cell-inside-cell).
        # Adhesion runs after so that centroids snap to the freshly deformed
        # boundary (for non-elastic cases _deform_boundary() is a no-op).
        self._deform_boundary()
        self._apply_boundary_adhesion()

        if self._boundary_elasticity > 0.0:
            mask = self._elastic_active_mask
            gx = self._grid_pts_x[mask]
            gy = self._grid_pts_y[mask]
            gf32 = self._grid_pts_f32[mask]
        else:
            gx = self._grid_pts_x
            gy = self._grid_pts_y
            gf32 = self._grid_pts_f32

        G = len(self.points)
        m = len(gx)
        pts32 = self.points.astype(np.float32)
        chunk = 4096
        labels: np.ndarray = np.empty(m, dtype=np.intp)

        if self._geodesic_voronoi:
            from ..geodesic import geodesic_label_active

            active_flat = self._elastic_active_mask if self._boundary_elasticity > 0.0 else self._initial_active_mask
            _result = geodesic_label_active(
                self.points,
                gx,
                gy,
                active_flat,
                self._grid_nx,
                self._grid_ny,
                return_debug=self._debug_geodesic,
                grid_x_coords=self._grid_x_coords,
                grid_y_coords=self._grid_y_coords,
            )
            if self._debug_geodesic:
                labels, _di = _result
                self._debug_geodesic_last = _di
                if _di["n_misplaced"] > 0:
                    import warnings

                    warnings.warn(
                        f"geodesic labeling (relax): "
                        f"{_di['n_misplaced']} seed(s) placed in wrong "
                        f"connected component (cross-water labeling likely)",
                        stacklevel=3,
                    )
            else:
                labels = np.asarray(_result, dtype=np.intp)
        elif self._weights is not None:
            if self._weight_ramp_iters > 0:
                ramp = min(iteration / self._weight_ramp_iters, 1.0)
                w_ramp = (1.0 - ramp) + ramp * self._weights
                w32 = w_ramp.astype(np.float32)
            else:
                w32 = self._weights.astype(np.float32)
            for start in range(0, m, chunk):
                end = min(start + chunk, m)
                dx = gx[start:end, None] - pts32[None, :, 0]
                dy = gy[start:end, None] - pts32[None, :, 1]
                labels[start:end] = ((dx * dx + dy * dy) / w32[None, :]).argmin(axis=1)
        elif area_eq_weight > 0.0:
            lam32 = self._power_offsets.astype(np.float32)
            for start in range(0, m, chunk):
                end = min(start + chunk, m)
                dx = gx[start:end, None] - pts32[None, :, 0]
                dy = gy[start:end, None] - pts32[None, :, 1]
                labels[start:end] = (dx * dx + dy * dy - lam32[None, :]).argmin(axis=1)
        else:
            _, labels = cKDTree(self.points).query(gf32, workers=-1)

        counts = np.bincount(labels, minlength=G)
        sum_x = np.bincount(labels, weights=gx, minlength=G)
        sum_y = np.bincount(labels, weights=gy, minlength=G)
        safe = np.maximum(counts, 1)
        target = np.column_stack([sum_x / safe, sum_y / safe])
        target[counts == 0] = self.points[counts == 0]

        if area_eq_weight > 0.0 and self._weights is None and not self._geodesic_voronoi:
            target_counts = float(m) / G
            self._power_offsets *= 1.0 - area_eq_weight
            self._power_offsets += area_eq_weight * 2.0 * self._pixel_area * (target_counts - counts)
            if ENSURE_NONEMPTY_POWER_CELLS_IN_RELAXATION:
                starved = counts == 0
                if starved.any():
                    self._power_offsets[starved] = self._nonempty_offsets()[starved]

        self.points = (
            self.points
            + relaxation_factor * (target - self.points)
            + self._topology_spring(topology_weight, intra_topology_weight)
        )

        if self._boundary_elasticity > 0.0:
            full_labels = np.full(self._grid_nx * self._grid_ny, -1, dtype=np.intp)
            active_idx = np.where(self._elastic_active_mask)[0]
            full_labels[active_idx] = labels
            self._last_labels_2d = full_labels.reshape(self._grid_ny, self._grid_nx)

        self._last_counts = counts
        self._constrain_points()

    def relax(
        self,
        relaxation_factor: float = 1.0,
        topology_weight: float = 0.0,
        area_eq_weight: float = 0.0,
        iteration: int = 0,
        intra_topology_weight: float = 0.0,
        **_kwargs: Any,
    ) -> None:
        """Run one raster Lloyd step."""
        self._relax_raster(relaxation_factor, topology_weight, area_eq_weight, iteration, intra_topology_weight)

    def area_cv(self) -> float:
        if not hasattr(self, "_last_counts"):
            raise RuntimeError("call relax() at least once before area_cv()")
        counts = self._last_counts.astype(float)
        # Weighted target: count[i] should be proportional to weight[i].
        # Normalize so that convergence -> CV -> 0 regardless of weight magnitude.
        normalized = counts / self._weights if self._weights is not None else counts
        mean = float(np.mean(normalized))
        return 0.0 if mean == 0 else float(np.std(normalized) / mean)

    def get_cells(self) -> np.ndarray:
        # Plain euclidean (no weights, no power-diagram): exact Voronoi is pixel-perfect.
        # Geodesic, weighted, or area_eq_weight>0 all require raster upsampling because
        # the cell boundaries are not standard Voronoi bisectors.
        if not self._geodesic_voronoi and self._area_eq_weight == 0.0 and self._weights is None:
            return _extract_exact_cells(self.points, self._current_boundary)
        # Geodesic / power-weighted: exact Voronoi gives wrong cell shapes;
        # use upsampled raster NN (default 4x computation resolution).
        res = self._final_resolution or (4 * self._raster_resolution)
        return self._build_cells_upsampled(res)
