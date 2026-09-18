"""Convenience wrapper: create_voronoi_cartogram."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
from shapely.ops import unary_union

from carto_flow._history import History

from .backends import AdhesiveBoundary, ElasticBoundary, ExactBackend, RasterBackend, _resolve_relaxation
from .history import VoronoiSnapshot
from .options import VoronoiOptions
from .result import VoronoiCartogram


def _make_outer_boundary(geometries, spec, *, _union=None):
    """Compute the outer boundary polygon from a boundary specification.

    Parameters
    ----------
    geometries : list of shapely.Geometry
        Input polygons (after any pre-scaling).
    spec : str or shapely.Geometry
        ``"union"`` / ``None`` — union of input geometries (default).
        ``"bbox"`` — axis-aligned bounding box of the union.
        ``"convex_hull"`` — convex hull of the union.
        ``"circle"`` — minimum enclosing circle (discretised with 64 segments).
        shapely Geometry — used directly.
    _union : shapely.Geometry or None
        Pre-computed union, to avoid recomputing when the caller already has it.
    """
    union = _union if _union is not None else unary_union(geometries)
    if spec is None or spec == "union":
        return union
    if spec == "bbox":
        from shapely.geometry import box

        return box(*union.bounds)
    if spec == "convex_hull":
        return union.convex_hull
    if spec == "circle":
        coords = np.array(union.convex_hull.exterior.coords)
        cx, cy = coords.mean(axis=0)
        r = float(np.linalg.norm(coords - [cx, cy], axis=1).max())
        from shapely.geometry import Point

        return Point(cx, cy).buffer(r, resolution=64)
    if hasattr(spec, "geom_type"):
        return spec
    raise ValueError(f"boundary must be 'union', 'bbox', 'convex_hull', 'circle', or a shapely Geometry; got {spec!r}")


def create_voronoi_cartogram(
    gdf: gpd.GeoDataFrame,
    *,
    weights: str | np.ndarray | None = None,
    backend: ExactBackend | RasterBackend | None = None,
    options: VoronoiOptions | None = None,
    group_by: str | None = None,
    boundary: str | Any = "union",
) -> VoronoiCartogram:
    """Run Lloyd relaxation (or FFT flow) on GeoDataFrame centroids.

    Distributes the centroid of each geometry in *gdf* so that every point
    claims an equal-area (or proportionally weighted) Voronoi cell within the
    outer union boundary (CVT — Centroidal Voronoi Tessellation).

    Parameters
    ----------
    gdf : GeoDataFrame
        Input polygons.  One point is placed at each geometry's centroid.
    weights : str, array-like of float, or None
        Per-geometry weights.  Pass a column name (str) to look up values
        from *gdf*, or a numeric array-like of length ``len(gdf)``.  A
        heavier point claims territory proportional to its weight.
        ``None`` = uniform weights.
    backend : ExactBackend, RasterBackend, or None
        Algorithm backend.  Each backend class exposes only the parameters
        relevant to its algorithm:

        * :class:`~carto_flow.voronoi_cartogram.backends.RasterBackend`
          (default) — raster NN Lloyd; 10-50x faster than exact.  Set
          ``boundary=ElasticBoundary(...)`` with ``relaxation=0.0`` for pure
          FFT-flow mode (no Lloyd).
        * :class:`~carto_flow.voronoi_cartogram.backends.ExactBackend`
          — exact scipy Voronoi + shapely clipping.

        ``None`` → ``RasterBackend()`` (resolution=300).
    options : VoronoiOptions or None
        Run-control settings (iteration count, stopping criteria, history
        recording).  Defaults to :class:`VoronoiOptions` with all defaults.
    group_by : str or None
        Column in *gdf* identifying a grouping of rows (e.g. state for
        congressional districts).  When set, each group is expected to form a
        contiguous region in the Voronoi tessellation.  Topology repair (if
        enabled via ``options.fix_topology``) will use this grouping for stage
        1 (group contiguity).  ``None`` = no contiguity tracking.
    boundary : str or shapely Geometry
        Outer clipping boundary for the Voronoi tessellation.  Accepted forms:

        * ``"union"`` *(default)* — union of all input geometries.
        * ``"bbox"`` — axis-aligned bounding box of the union.
        * ``"convex_hull"`` — convex hull of the union.
        * ``"circle"`` — minimum enclosing circle of the union (64-segment
          approximation).
        * shapely Geometry — used directly as the outer boundary.

        When using :class:`~carto_flow.voronoi_cartogram.backends.ElasticBoundary`,
        simple shapes (e.g. ``"bbox"``) are automatically densified to roughly
        ``backend.resolution`` vertices before the elastic deformation is
        initialised, ensuring smooth FFT-driven boundary deformation.

    Returns
    -------
    VoronoiCartogram
        Result object with:

        - ``positions`` — final centroid coordinates, shape ``(G, 2)``
        - ``cells`` — clipped Voronoi cell polygons, shape ``(G,)``
        - ``metrics`` — convergence summary
        - ``convergence_history`` — list of area_cv values per iteration
        - ``history`` — per-iteration snapshots when
          ``options.record_history`` is set, else ``None``
        - ``to_geodataframe()`` — source GDF with Voronoi cells as geometry
        - ``plot()`` — choropleth of the cells

    Examples
    --------
    >>> import geopandas as gpd
    >>> from carto_flow import (
    ...     create_voronoi_cartogram,
    ...     ExactBackend, RasterBackend, VoronoiOptions,
    ...     RelaxationSchedule,
    ... )

    Default (raster Lloyd, resolution=300):

    >>> result = create_voronoi_cartogram(gdf)

    Exact backend with custom schedule:

    >>> result = create_voronoi_cartogram(
    ...     gdf,
    ...     backend=ExactBackend(relaxation=RelaxationSchedule(start=1.9, decay=0.95)),
    ...     options=VoronoiOptions(n_iter=100),
    ... )

    Raster + geodesic distance:

    >>> result = create_voronoi_cartogram(
    ...     gdf, backend=RasterBackend(resolution=300, distance_mode="geodesic")
    ... )

    Pure FFT-flow (no Lloyd):

    >>> from carto_flow import ElasticBoundary
    >>> result = create_voronoi_cartogram(
    ...     gdf,
    ...     backend=RasterBackend(
    ...         resolution=300,
    ...         relaxation=0.0,
    ...         boundary=ElasticBoundary(strength=0.3, step_scale=0.8),
    ...         distance_mode="geodesic",
    ...     ),
    ... )
    """
    from carto_flow.geo_utils.adjacency import find_adjacent_pairs

    if backend is None:
        backend = RasterBackend()
    if options is None:
        options = VoronoiOptions()

    if group_by is not None and group_by not in gdf.columns:
        raise ValueError(f"group_by column {group_by!r} not found in GeoDataFrame")
    if backend.intra_group_spring is not None and group_by is None:
        import warnings

        warnings.warn(
            "intra_group_spring is set but group_by is None — intra_group_spring will be ignored",
            UserWarning,
            stacklevel=2,
        )

    weights_arr: np.ndarray | None
    if isinstance(weights, str):
        if weights not in gdf.columns:
            raise ValueError(f"weights column {weights!r} not found in GeoDataFrame")
        weights_arr = gdf[weights].to_numpy(dtype=float)
    else:
        weights_arr = weights

    geometries = list(gdf.geometry)
    areas = np.array([g.area for g in geometries], dtype=np.float64)
    _mean = areas.mean()
    initial_cv: float = float(np.std(areas) / _mean) if _mean > 0 else 0.0

    if options.prescale_components:
        from carto_flow.geo_utils.prescale import prescale_connected_components

        w_for_prescale = weights_arr if weights_arr is not None else np.ones(len(geometries), dtype=np.float64)
        total_area = float(areas.sum())
        target_density = float(np.sum(w_for_prescale)) / total_area if total_area > 0 else 1.0
        geometries = prescale_connected_components(geometries, w_for_prescale, target_density)

    union_boundary = unary_union(geometries)  # always the data union, for AdhesiveBoundary mask
    outer_full = _make_outer_boundary(geometries, boundary, _union=union_boundary)
    if options.simplify_tol is not None:
        outer = outer_full.simplify(options.simplify_tol, preserve_topology=True)
    else:
        outer = outer_full

    _pos_list: list[list[float]] = []
    for g in geometries:
        c = g.centroid
        if not g.contains(c):
            c = g.representative_point()
        _pos_list.append([c.x, c.y])
    positions: np.ndarray = np.array(_pos_list, dtype=np.float64)

    topo = options._topology_repair()
    adj_pairs: list[tuple[int, int]] | None = None
    intra_adj_pairs: list[tuple[int, int]] | None = None
    need_adj = backend.adjacency_spring > 0 or backend.intra_group_spring is not None or topo is not None
    if need_adj:
        adj_pairs = [
            (i, j) for i, j, _ in find_adjacent_pairs(geometries, min_shared_length=options.adj_min_shared_length)
        ]
    # Split adj_pairs into cross-group and intra-group when intra_group_spring is active.
    # adjacency_spring then applies only to cross-group pairs; intra_group_spring to same-group.
    if group_by is not None and backend.intra_group_spring is not None and adj_pairs:
        _grp = list(gdf[group_by])
        intra_adj_pairs = [(i, j) for i, j in adj_pairs if _grp[i] == _grp[j]]
        adj_pairs = [(i, j) for i, j in adj_pairs if _grp[i] != _grp[j]]

    boundary_mask: np.ndarray | None = None
    adhesion_boundary = None
    _is_adhesive = isinstance(backend.boundary, AdhesiveBoundary)
    _is_elastic_adhesive = isinstance(backend.boundary, ElasticBoundary) and backend.boundary.adhesion_strength > 0
    if _is_adhesive or _is_elastic_adhesive:
        if _is_adhesive:
            # Snap to data union regardless of the outer clipping boundary.
            # A geometry touching only a bbox edge (not the data edge) should
            # not be snapped.
            snap_ref = union_boundary
            adhesion_boundary = union_boundary
        else:
            # Elastic + adhesion: snap to outer (= initial elastic boundary).
            # _boundary_line is updated dynamically in _deform_boundary() each
            # step so centroids always track the current deformed shape.
            snap_ref = outer
            adhesion_boundary = None  # BaseField uses boundary (= outer) as snap target
        boundary_mask = np.array([bool(g.intersects(snap_ref.boundary)) for g in geometries])

    build_kwargs: dict = {
        "weights": weights_arr,
        "adj_pairs": adj_pairs,
        "intra_adj_pairs": intra_adj_pairs,
        "boundary_mask": boundary_mask,
        "adhesion_boundary": adhesion_boundary,
    }
    if isinstance(backend, RasterBackend):
        build_kwargs["debug"] = options.debug

    field = backend.build_field(positions, outer, **build_kwargs)

    _schedule = _resolve_relaxation(backend.relaxation)
    errors: list[float] = []
    history_obj: History | None = History() if options.record_history is not False else None
    converged = False

    _groups_array = list(gdf[group_by]) if group_by is not None else None
    _gdf_labels = gdf.index.tolist()

    for i in range(options.n_iter):
        x_prev = field.get_points()
        factor = _schedule(i)
        backend.relax_step(field, factor, i)  # type: ignore[arg-type]

        if topo is not None and (i + 1) % topo.every == 0:
            from .contiguity import _compose_topology_permutation

            _cells_list = list(field.get_cells_for_contiguity())
            _slot_of, _ = _compose_topology_permutation(
                _cells_list,
                _groups_array,
                adj_pairs,
                group_contiguity=topo.group_contiguity,
                adjacency=topo.adjacency,
                orientation=topo.orientation,
                max_passes=topo.max_passes,
                geom_positions=positions,
            )
            # Rebuild _cells_list in permuted order for intersection-midpoint relocation.
            _cells_list = [_cells_list[s] for s in _slot_of]

            # Apply the composed permutation. For each cycle, relocate each district
            # via intersection-midpoint mirroring to reduce overshoot on large↔small swaps.
            changed = np.where(_slot_of != np.arange(len(_slot_of)))[0]
            if len(changed) > 0:
                import shapely as sh
                from shapely.geometry import LineString

                orig_pts = field.points.copy()
                new_pts = field.points[_slot_of].copy()  # default: simple swap

                _debug_cycles: list[list[int]] = []
                relocated: set[int] = set()

                visited = np.zeros(len(_slot_of), dtype=bool)
                for start in changed:
                    if visited[start]:
                        continue
                    cycle: list[int] = []
                    cur = int(start)
                    while not visited[cur]:
                        visited[cur] = True
                        cycle.append(cur)
                        cur = int(_slot_of[cur])
                    if options.debug:
                        _debug_cycles.append(cycle)
                    k = len(cycle)
                    for ci, d in enumerate(cycle):
                        prev_d = cycle[(ci - 1) % k]
                        next_d = cycle[(ci + 1) % k]
                        cell_d_old = _cells_list[prev_d]  # d's current cell
                        cell_next_old = _cells_list[d]  # next_d's current cell
                        pd = orig_pts[d]
                        pnext = orig_pts[next_d]
                        diff = pd - pnext
                        dist = float(np.linalg.norm(diff))
                        if dist < 1e-12:
                            continue  # degenerate; keep simple swap
                        dir_u = diff / dist
                        scale = (
                            max(
                                cell_d_old.bounds[2] - cell_d_old.bounds[0],
                                cell_d_old.bounds[3] - cell_d_old.bounds[1],
                                cell_next_old.bounds[2] - cell_next_old.bounds[0],
                                cell_next_old.bounds[3] - cell_next_old.bounds[1],
                            )
                            + dist
                        )
                        ray_d = LineString([
                            (pd[0], pd[1]),
                            (pd[0] + dir_u[0] * scale, pd[1] + dir_u[1] * scale),
                        ])
                        ray_n = LineString([
                            (pnext[0], pnext[1]),
                            (pnext[0] - dir_u[0] * scale, pnext[1] - dir_u[1] * scale),
                        ])
                        inter_d = sh.intersection(cell_d_old.boundary, ray_d)
                        inter_n = sh.intersection(cell_next_old.boundary, ray_n)
                        if sh.is_empty(inter_d) or sh.is_empty(inter_n):
                            continue  # generator outside cell; keep simple swap
                        pts_d = sh.get_coordinates(inter_d)
                        pts_n = sh.get_coordinates(inter_n)
                        fa = pts_d[np.argmax(pts_d @ dir_u - pnext @ dir_u)]
                        fb = pts_n[np.argmax(pts_n @ -dir_u - pd @ -dir_u)]
                        pivot = (fa + fb) / 2.0
                        new_pts[d] = 2.0 * pivot - pd
                        if options.debug:
                            relocated.add(d)

                field.points[:] = new_pts
                field._on_points_changed()

                if options.debug and _debug_cycles:
                    print(f"[topology repair] iter {i + 1}: {len(changed)} districts in {len(_debug_cycles)} cycles")
                    for cycle in _debug_cycles:
                        k = len(cycle)
                        lbls = [str(_gdf_labels[d]) for d in cycle]
                        if k == 2:
                            print(f"  2-cycle  [{lbls[0]} ↔ {lbls[1]}]")
                        else:
                            print(f"  {k}-chain  [{' → '.join(lbls)} → ...]")
                        for d in cycle:
                            ox, oy = orig_pts[d]
                            nx, ny = new_pts[d]
                            method = "intersection-midpoint" if d in relocated else "simple-swap (degenerate)"
                            print(
                                f"    {_gdf_labels[d]!s:<20s}  "
                                f"({ox:+.4f}, {oy:+.4f}) → ({nx:+.4f}, {ny:+.4f})"
                                f"  [{method}]"
                            )

        curr = field.get_points()
        max_disp = float(np.linalg.norm(curr - x_prev, axis=1).max())
        cv = field.area_cv()
        errors.append(cv)

        if history_obj is not None:
            record = options.record_history
            should_snap = record is True or (
                isinstance(record, int) and not isinstance(record, bool) and i % record == 0
            )
            if should_snap:
                snap_cells = field.get_cells() if options.record_cells else None
                history_obj.add_snapshot(
                    VoronoiSnapshot(
                        iteration=i,
                        positions=field.get_points(),
                        area_cv=cv,
                        cells=snap_cells,
                    )
                )

        if options.show_progress:
            print(f"  iter {i:3d}  max_disp={max_disp:.4g}  area_cv={cv:.4f}  factor={factor:.3f}")

        if options.tol is not None and max_disp < options.tol:
            if options.show_progress:
                print(f"  converged at iter {i} (displacement tol)")
            converged = True
            break

        if options.area_cv_tol is not None and cv < options.area_cv_tol:
            if options.show_progress:
                print(f"  converged at iter {i} (area_cv tol)")
            converged = True
            break

    final_cv = errors[-1] if errors else None
    cells = field.get_cells()

    # Per-geometry signed area errors (% deviation from target)
    boundary_area = outer.area
    _n = len(cells)
    _w = weights_arr
    _total_w = float(_w.sum()) if _w is not None else float(_n)
    target_areas = (
        _w / _total_w * boundary_area if _w is not None else np.full(_n, boundary_area / _n, dtype=np.float64)
    )
    actual_areas = np.array([c.area for c in cells], dtype=np.float64)
    area_errors = (actual_areas / target_areas - 1.0) * 100.0  # signed %, shape (G,)

    metrics = {
        "n_iterations": len(errors),
        "converged": converged,
        "initial_area_cv": initial_cv,
        "final_area_cv": final_cv,
        "mean_area_error_pct": float(np.mean(np.abs(area_errors))),
        "max_area_error_pct": float(np.max(np.abs(area_errors))),
    }

    return VoronoiCartogram(
        positions=field.get_points(),
        cells=cells,
        metrics=metrics,
        options=options,
        _source_gdf=gdf,
        convergence_history=errors,
        history=history_obj,
        _field=field,
        area_errors=area_errors,
        _weighted=weights is not None,
    )
