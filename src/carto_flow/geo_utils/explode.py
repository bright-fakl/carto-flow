"""Physics-based region separation for GeoDataFrames."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import geopandas as gpd

__all__ = ["explode_geodataframe"]


def explode_geodataframe(
    gdf: gpd.GeoDataFrame,
    distance: float,
    *,
    group_by: str | None = None,
    max_iter: int = 200,
    k_repulse: float = 1.0,
    pre_displace: float = 0.0,
    max_step: float = 0.2,
    simplify_tolerance: float = 0.05,
    rebuild_every: int = 5,
    return_proxies: bool = False,
) -> gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Separate polygons in a GeoDataFrame so that neighboring bodies are spaced apart.

    Uses pure repulsion: bodies that are closer than ``distance`` apart push each
    other away proportionally to the gap deficit.  Forces go to exactly zero once
    all gaps reach ``distance``, giving clean convergence.  The global centroid of
    the layout is preserved (Newton's third law).

    Performance: simplified proxy geometries are used for all in-loop distance
    computations; the STRtree is rebuilt every ``rebuild_every`` iterations using
    bounding-box expansion (no buffer); centroids are tracked as numpy arrays.
    The original geometries are translated only once for the final output.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input geometries to separate.
    distance : float
        Target spacing (gap) between neighboring bodies, in the same units as
        the input geometries.  A good starting value is one tile size.
    group_by : str or None
        Column name whose values define rigid bodies.  All rows with the same
        value move together as a single unit.  ``None`` (default) treats each
        row independently.  Use this when ``gdf`` contains individual tiles but
        you want whole regions to move together
        (e.g. ``group_by="STATE"`` on a tile-level GeoDataFrame).
    max_iter : int
        Maximum number of simulation steps.  Default 200.
    k_repulse : float
        Repulsion strength per pair.  Force in map units =
        ``k_repulse x (distance - gap)``, so at full contact (gap = 0) each
        pair contributes ``k_repulse x distance`` of displacement per step
        before clamping.  Default 1.0.
    pre_displace : float
        Outward displacement applied to each body before the simulation starts,
        as a multiple of ``distance``.  Each body is pushed away from the
        collective centroid by ``pre_displace x distance`` (regardless of how
        far it is from the centroid).  This pre-separates bodies in dense chains
        so the repulsion loop converges faster.  0.0 = no pre-displacement
        (default).  Try 1-3 for layouts where many bodies are initially
        overlapping.
    max_step : float
        Maximum displacement per body per iteration as a fraction of
        ``distance``.  Default 0.2.  Increase (e.g. 0.4) to make pressure
        propagate faster through neighbor chains at the cost of potential
        overshoot; decrease for smoother convergence.
    simplify_tolerance : float
        Fraction of ``distance`` used as the Douglas-Peucker simplification
        tolerance for proxy geometries.  Also capped at 10 % of each body's
        effective radius so small bodies are not over-simplified.
        Default 0.05 (5 %).  Increase for faster computation at the cost of
        slightly less accurate spacing; decrease for more faithful geometry.
    rebuild_every : int
        Rebuild the active pair list every this many iterations.  Default 5.
        Use 1 for maximum correctness (bodies pushed sideways are caught
        immediately); increase for speed when the layout is well-behaved.
    return_proxies : bool
        If True, return a tuple ``(result, proxies_gdf)`` where ``proxies_gdf``
        contains the simplified proxy geometries used for distance computation
        (one row per body, not per input row).  Useful for debugging spacing
        accuracy.  Default False.

    Returns
    -------
    GeoDataFrame or tuple[GeoDataFrame, GeoDataFrame]
        Copy of ``gdf`` with geometry replaced by translated polygons.
        All non-geometry columns and the index are preserved unchanged.
        When ``return_proxies=True``, returns ``(result, proxies_gdf)``.

    Examples
    --------
    Separate region blobs directly:

    >>> exploded = explode_geodataframe(result.regions, distance=tile_size)
    >>> exploded.plot()

    Separate regions while keeping individual tile boundaries visible:

    >>> tiles_gdf = result.to_geodataframe()
    >>> exploded = explode_geodataframe(tiles_gdf, distance=tile_size, group_by="STATE")
    >>> exploded.plot()
    """
    import math

    import numpy as np
    from shapely.affinity import translate
    from shapely.ops import unary_union
    from shapely.strtree import STRtree

    geoms = list(gdf.geometry)
    row_n = len(geoms)

    # Build bodies: each body has a combined geometry and a list of row indices
    if group_by is not None:
        groups = list(dict.fromkeys(gdf[group_by]))  # ordered unique values
        body_rows: dict[Any, list[int]] = {gv: [] for gv in groups}
        for i, gv in enumerate(gdf[group_by]):
            body_rows[gv].append(i)
        body_geoms: list[Any] = []
        body_row_lists: list[list[int]] = []
        for gv in groups:
            rows = body_rows[gv]
            valid_g = [geoms[i] for i in rows if geoms[i] is not None and not geoms[i].is_empty]
            body_geoms.append(unary_union(valid_g) if valid_g else None)
            body_row_lists.append(rows)
    else:
        body_geoms = geoms
        body_row_lists = [[i] for i in range(row_n)]

    n = len(body_geoms)
    valid_idx = [i for i, g in enumerate(body_geoms) if g is not None and not g.is_empty]
    if not valid_idx:
        return gdf.copy()

    # --- One-time setup ---

    # Simplified proxy geometries: per-body adaptive tolerance
    # Capped at 10% of body radius to protect small bodies from over-simplification.
    # preserve_topology=True prevents protrusions from being collapsed, which would
    # make the proxy smaller than the original and cause under-separation.
    proxy_geoms: list[Any] = []
    for g in body_geoms:
        if g is None or g.is_empty:
            proxy_geoms.append(None)
        else:
            tol = min(distance * simplify_tolerance, math.sqrt(max(g.area, 0)) * 0.1)
            proxy_geoms.append(g.simplify(tol, preserve_topology=True))

    # Initial centroids tracked as numpy arrays — no .centroid calls inside the loop
    cx0 = np.array([body_geoms[i].centroid.x for i in valid_idx])
    cy0 = np.array([body_geoms[i].centroid.y for i in valid_idx])
    pos_of = {i: p for p, i in enumerate(valid_idx)}  # body index → position in valid_idx

    mean_diam = float(np.mean([math.sqrt(max(body_geoms[i].area, 0)) for i in valid_idx])) or 1.0
    min_dist = mean_diam * 0.01  # prevents division by zero when centroids coincide
    max_step_abs = max_step * distance  # fraction → map units
    tol = 1e-3 * distance  # converged when step < 0.1% of target spacing

    from shapely.geometry import box as _box

    dx, dy = np.zeros(n), np.zeros(n)

    # Pre-displace: push each body outward by pre_displace * distance
    if pre_displace > 0.0:
        gcx = float(np.mean(cx0))
        gcy = float(np.mean(cy0))
        for p, i in enumerate(valid_idx):
            ddx = cx0[p] - gcx
            ddy = cy0[p] - gcy
            mag = math.hypot(ddx, ddy)
            if mag > 0:
                dx[i] = ddx / mag * pre_displace * distance
                dy[i] = ddy / mag * pre_displace * distance
    active_pairs: list[tuple[int, int]] = []
    last_rebuild = -(rebuild_every + 1)  # force rebuild on first iteration

    for iteration in range(max_iter):
        # Translate proxy geometries at current positions (few vertices — fast)
        proxy_t = [translate(proxy_geoms[i], dx[i], dy[i]) if proxy_geoms[i] is not None else None for i in range(n)]

        # Rebuild active pair list every rebuild_every iterations using a cheap
        # bounding-box expansion query (no buffer geometry created).
        # Bodies can move sideways toward initially-distant bodies, so the list
        # must be refreshed; rebuild_every > 1 amortises the STRtree cost.
        if (iteration - last_rebuild) >= rebuild_every:
            # proxy_t[i] is None only when body_geoms[i] is None/empty, which
            # is already excluded from valid_idx above.
            valid_pt = [(i, proxy_t[i]) for i in valid_idx if proxy_t[i] is not None]
            tree = STRtree([t for _, t in valid_pt])
            imap = [i for i, _ in valid_pt]
            active_pairs = []
            for i, pi in valid_pt:
                bx0, by0, bx1, by1 = cast("Any", pi).bounds
                qbox = _box(bx0 - distance / 2, by0 - distance / 2, bx1 + distance / 2, by1 + distance / 2)
                for lj in tree.query(qbox):  # bbox-only filter, no predicate overhead
                    j = imap[lj]
                    if j <= i:
                        continue
                    active_pairs.append((i, j))
            last_rebuild = iteration

        fdx = np.zeros(n)
        fdy = np.zeros(n)

        for i, j in active_pairs:
            pi, pj = proxy_t[i], proxy_t[j]
            if pi is None or pj is None:
                continue
            gap = pi.distance(pj)
            if gap >= distance:
                continue
            # Centroid direction from tracked arrays — no .centroid call
            p_i, p_j = pos_of[i], pos_of[j]
            ddx = (cx0[p_i] + dx[i]) - (cx0[p_j] + dx[j])
            ddy = (cy0[p_i] + dy[i]) - (cy0[p_j] + dy[j])
            cdist = max(math.hypot(ddx, ddy), min_dist)
            f = k_repulse * (distance - gap)
            fdx[i] += f * ddx / cdist
            fdy[i] += f * ddy / cdist
            fdx[j] -= f * ddx / cdist
            fdy[j] -= f * ddy / cdist

        step_norm = max(float(np.abs(fdx).max()), float(np.abs(fdy).max()), 1e-12)
        if step_norm > max_step_abs:
            fdx *= max_step_abs / step_norm
            fdy *= max_step_abs / step_norm
            step_norm = max_step_abs

        dx += fdx
        dy += fdy
        if step_norm < tol:
            break

    # Map body displacements back to row level
    row_dx, row_dy = np.zeros(row_n), np.zeros(row_n)
    for bi, rows in enumerate(body_row_lists):
        for ri in rows:
            row_dx[ri] = dx[bi]
            row_dy[ri] = dy[bi]

    # Translate original (full-resolution) geometries once
    result = gdf.copy()
    result = result.set_geometry([
        translate(g, float(row_dx[i]), float(row_dy[i])) if (g is not None and not g.is_empty) else g
        for i, g in enumerate(geoms)
    ])

    if return_proxies:
        import geopandas as gpd_mod

        proxies_gdf = gpd_mod.GeoDataFrame(
            geometry=[
                translate(proxy_geoms[i], float(dx[i]), float(dy[i])) if proxy_geoms[i] is not None else None
                for i in range(n)
            ],
            crs=gdf.crs,
        )
        return result, proxies_gdf

    return result
