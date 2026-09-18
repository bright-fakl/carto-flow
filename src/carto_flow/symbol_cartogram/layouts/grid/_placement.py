"""Grid placement computation: hole filling, island fixing, Hungarian assignment."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def fill_internal_holes(
    assignments: NDArray[np.intp],
    tile_adjacency: NDArray[np.bool_],
    centroids: NDArray[np.floating],
    grid_centers: NDArray[np.floating],
    original_polygons: list,
    *,
    min_hole_fraction: float = 0.5,
    verbose: bool = False,
) -> NDArray[np.intp]:
    """Post-process grid assignments to fill internal holes.

    An internal hole is an unoccupied tile surrounded by occupied tiles that
    cannot reach the grid boundary through other unoccupied tiles.  Geographic
    gaps (e.g. internal lakes) present in the original geometries are preserved
    by comparing the number of grid holes to the number of geographic holes.

    Parameters
    ----------
    assignments : (n,) int array
        Current region-to-tile assignments (indices into *grid_centers*).
    tile_adjacency : (m, m) bool array
        Edge-based tile adjacency matrix.  Used for the flood-fill that
        classifies exterior vs. internal tiles, for grouping holes into
        connected components, and for shift chain BFS.  Should use strict
        edge adjacency (shared vertices >= 2) so that vertex-only touches
        don't leak the flood-fill through gaps.
    centroids : (n, 2) array
        Original region centroids.
    grid_centers : (m, 2) array
        Tile center positions.
    original_polygons : list of Polygon
        Original region geometries, used to detect expected geographic gaps.
    min_hole_fraction : float
        Minimum area of a geographic interior ring, as a fraction of one
        tile's area, for it to count as a genuine geographic gap.  Rings
        smaller than this are treated as boundary artifacts and ignored.
    verbose : bool
        If True, print diagnostic information about the hole-filling process.

    Returns
    -------
    (n,) int array
        Updated assignments with internal holes filled.

    """
    from collections import deque

    from shapely import Polygon as ShapelyPolygon
    from shapely.ops import unary_union

    assignments = assignments.copy()
    m = len(grid_centers)
    n = len(assignments)

    if verbose:
        assigned_set = set(assignments.tolist())
        print(f"[fill_holes] {n} regions assigned to {m} tiles")
        print(f"[fill_holes] {m - len(assigned_set)} unassigned tiles")

    # --- Estimate minimum area for a geographic hole to be significant ---
    # Approximate one tile's area from the mean distance between adjacent
    # tile centers, then scale by min_hole_fraction.  Interior rings
    # smaller than this threshold are treated as boundary artifacts.
    adj_pairs = np.argwhere(tile_adjacency)
    if len(adj_pairs) > 0:
        dists = np.linalg.norm(
            grid_centers[adj_pairs[:, 0]] - grid_centers[adj_pairs[:, 1]],
            axis=1,
        )
        tile_area_est = float(np.mean(dists)) ** 2
    else:
        tile_area_est = 0.0
    min_hole_area = min_hole_fraction * tile_area_est

    # --- Count expected geographic holes in original geometries ---
    union_geom = unary_union(original_polygons)
    n_geo_holes = 0
    n_total_rings = 0
    if union_geom.geom_type == "Polygon":
        for ring in union_geom.interiors:
            n_total_rings += 1
            if ShapelyPolygon(ring).area >= min_hole_area:
                n_geo_holes += 1
    elif union_geom.geom_type == "MultiPolygon":
        for poly in union_geom.geoms:
            for ring in poly.interiors:
                n_total_rings += 1
                if ShapelyPolygon(ring).area >= min_hole_area:
                    n_geo_holes += 1

    if verbose:
        print(f"[fill_holes] Union geometry type: {union_geom.geom_type}")
        print(
            f"[fill_holes] Estimated tile area: {tile_area_est:.4g}, "
            f"min_hole_fraction: {min_hole_fraction}, "
            f"min_hole_area: {min_hole_area:.4g}",
        )
        print(
            f"[fill_holes] Interior rings in union: {n_total_rings} total, "
            f"{n_geo_holes} significant (area >= {min_hole_area:.4g})",
        )

    # --- Iteratively detect and fill holes ---
    # We loop because each shift chain changes the assignment set.
    max_outer = len(assignments)  # safety limit
    for _ in range(max_outer):
        assigned_set = set(assignments.tolist())
        unassigned = set(range(m)) - assigned_set

        if not unassigned:
            break

        # -- Detect boundary tiles (fewer neighbors than interior max) --
        neighbor_counts = tile_adjacency.sum(axis=1)
        max_neighbors = int(neighbor_counts.max())

        if verbose:
            n_unassigned_boundary = sum(1 for t in unassigned if int(neighbor_counts[t]) < max_neighbors)
            print(
                f"[fill_holes] Max tile neighbors: {max_neighbors}, unassigned boundary seeds: {n_unassigned_boundary}",
            )

        # -- Flood-fill from ALL unoccupied boundary tiles --
        exterior: set[int] = set()
        queue: deque[int] = deque()
        for t in unassigned:
            if int(neighbor_counts[t]) < max_neighbors:
                exterior.add(t)
                queue.append(t)

        while queue:
            t = queue.popleft()
            for j in np.where(tile_adjacency[t])[0]:
                j_int = int(j)
                if j_int in unassigned and j_int not in exterior:
                    exterior.add(j_int)
                    queue.append(j_int)

        # -- Group internal holes into connected components --
        internal = unassigned - exterior
        if verbose:
            print(f"[fill_holes] Exterior tiles: {len(exterior)}, internal hole tiles: {len(internal)}")
        if not internal:
            if verbose:
                print("[fill_holes] No internal holes found — done.")
            break

        visited: set[int] = set()
        components: list[list[int]] = []
        for seed in internal:
            if seed in visited:
                continue
            comp: list[int] = []
            q: deque[int] = deque([seed])
            visited.add(seed)
            while q:
                t = q.popleft()
                comp.append(t)
                for j in np.where(tile_adjacency[t])[0]:
                    j_int = int(j)
                    if j_int in internal and j_int not in visited:
                        visited.add(j_int)
                        q.append(j_int)
            components.append(comp)

        n_grid_holes = len(components)
        n_to_fill = max(0, n_grid_holes - n_geo_holes)

        if verbose:
            comp_sizes = [len(c) for c in components]
            print(f"[fill_holes] Grid hole components: {n_grid_holes} (sizes: {sorted(comp_sizes)})")
            print(f"[fill_holes] Geographic holes: {n_geo_holes} → filling {n_to_fill} component(s)")

        if n_to_fill == 0:
            if verbose:
                print("[fill_holes] All grid holes accounted for by geographic gaps — done.")
            break

        # Sort by component size (ascending) — fill smallest first
        components.sort(key=len)
        to_fill = components[:n_to_fill]

        # Precompute boundary tile set for shift chain BFS.
        boundary_tiles = {int(t) for t in range(m) if int(neighbor_counts[t]) < max_neighbors}

        # Fill one hole per outer iteration so that hole detection is
        # re-run with fresh topology after each shift chain.
        filled_any = False
        for comp in to_fill:
            if filled_any:
                break
            for h in comp:
                # BFS from hole *h* through assigned tiles to reach one
                # that can be safely vacated: either it borders an
                # exterior unoccupied tile, or it is itself a boundary
                # tile (fewer neighbors than grid interior max).
                parent: dict[int, int] = {h: -1}
                bfs_q: deque[int] = deque([h])
                end_tile: int | None = None

                while bfs_q:
                    cur = bfs_q.popleft()
                    for nb in np.where(tile_adjacency[cur])[0]:
                        nb_int = int(nb)
                        if nb_int in parent:
                            continue
                        if nb_int not in assigned_set:
                            # Unassigned neighbor: if it's exterior and
                            # cur is not the hole itself, cur can be the
                            # chain end (its vacated spot borders exterior).
                            if nb_int in exterior and cur != h:
                                end_tile = cur
                                break
                            continue
                        # nb_int is assigned — check if it's safe to vacate
                        if nb_int in boundary_tiles:
                            # Boundary tile: vacating it won't create
                            # an internal hole (it's on the grid edge).
                            parent[nb_int] = cur
                            end_tile = nb_int
                            break
                        parent[nb_int] = cur
                        bfs_q.append(nb_int)
                    if end_tile is not None:
                        break

                if end_tile is None:
                    if verbose:
                        print(
                            f"[fill_holes]   Hole tile {h}: no path to "
                            f"exterior found (BFS explored {len(parent)} tiles)",
                        )
                    continue  # no path found for this hole tile

                # Reconstruct path from end_tile back to h
                path = []
                node = end_tile
                while node != -1:
                    path.append(node)
                    node = parent[node]
                path.reverse()  # path[0] = h, path[-1] = end_tile

                # Shift assignments along the path:
                # region at path[1] → path[0], path[2] → path[1], etc.
                # path[-1] becomes unoccupied (borders exterior).
                for k in range(len(path) - 1):
                    dst_tile = path[k]
                    src_tile = path[k + 1]
                    region = int(np.where(assignments == src_tile)[0][0])
                    assignments[region] = dst_tile

                if verbose:
                    print(f"[fill_holes]   Hole tile {h}: shift chain length {len(path)} → vacating tile {path[-1]}")

                # The vacated tile (path[-1]) now borders the exterior.
                vacated = path[-1]
                assigned_set.discard(vacated)
                assigned_set.add(h)
                exterior.add(vacated)
                filled_any = True
                break  # restart outer loop to re-detect holes

        if not filled_any:
            if verbose:
                print("[fill_holes] No holes filled in this iteration — done.")
            break

    if verbose:
        final_assigned = set(assignments.tolist())
        final_unassigned = set(range(m)) - final_assigned
        print(f"[fill_holes] Final: {len(final_unassigned)} unassigned tiles")

    return assignments


def _fix_island_assignments(
    assignments: NDArray[np.intp],
    tile_adjacency: NDArray[np.bool_],
    centroids: NDArray[np.floating],
    grid_centers: NDArray[np.floating],
    region_adjacency: NDArray[np.floating] | None = None,
    *,
    verbose: bool = False,
) -> NDArray[np.intp]:
    """Fix assignment connectivity: connect non-islands, disconnect true islands.

    Performs two corrections:

    1. **Connect non-islands**: regions that are NOT true geographic islands
       but are assigned to tiles disconnected from the main cluster are
       reassigned to tiles adjacent to the main cluster.
    2. **Disconnect true islands**: regions that ARE true geographic islands
       but are assigned to tiles connected to the main cluster are reassigned
       to tiles NOT adjacent to the main cluster.

    Parameters
    ----------
    assignments : (n,) int array
        Current region-to-tile assignments.
    tile_adjacency : (m, m) bool array
        Edge-based tile adjacency matrix.
    centroids : (n, 2) array
        Original region centroids.
    grid_centers : (m, 2) array
        Tile center positions.
    region_adjacency : (n, n) float array, optional
        Original region adjacency matrix.  Used to identify true geographic
        islands.  If *None*, all regions are assumed connected.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    (n,) int array
        Updated assignments with connectivity fixed.

    """
    from collections import deque

    assignments = assignments.copy()
    n = len(assignments)
    m = len(grid_centers)

    # --- Identify true geographic islands (connected components of regions) ---
    if region_adjacency is not None:
        region_visited: set[int] = set()
        region_components: list[set[int]] = []
        for r in range(n):
            if r in region_visited:
                continue
            comp: set[int] = set()
            q: deque[int] = deque([r])
            region_visited.add(r)
            while q:
                cur = q.popleft()
                comp.add(cur)
                for nb in range(n):
                    if nb not in region_visited and region_adjacency[cur, nb] > 0:
                        region_visited.add(nb)
                        q.append(nb)
            region_components.append(comp)
        mainland_regions = max(region_components, key=len)
        true_island_regions = set(range(n)) - mainland_regions
    else:
        true_island_regions = set()
        region_components = [set(range(n))]

    if verbose:
        n_island_comps = (
            sum(1 for c in region_components if c != mainland_regions) if region_adjacency is not None else 0
        )
        print(
            f"[fix_islands] {n} regions, "
            f"{len(true_island_regions)} true island region(s) "
            f"in {n_island_comps} island group(s)",
        )

    # ---------------------------------------------------------------
    # Step 1: Connect non-island regions that are disconnected
    # ---------------------------------------------------------------
    max_iters = n
    for _iteration in range(max_iters):
        assigned_set = set(assignments.tolist())

        # Build assignment connected components via tile adjacency
        tile_to_regions: dict[int, list[int]] = {}
        for r in range(n):
            t = int(assignments[r])
            tile_to_regions.setdefault(t, []).append(r)

        tile_visited: set[int] = set()
        assign_components: list[list[int]] = []
        for r in range(n):
            t = int(assignments[r])
            if t in tile_visited:
                continue
            comp_regions: list[int] = []
            tile_queue: deque[int] = deque([t])
            tile_visited.add(t)
            while tile_queue:
                cur_t = tile_queue.popleft()
                comp_regions.extend(tile_to_regions.get(cur_t, []))
                for nb in np.where(tile_adjacency[cur_t])[0]:
                    nb_int = int(nb)
                    if nb_int not in tile_visited and nb_int in assigned_set:
                        tile_visited.add(nb_int)
                        tile_queue.append(nb_int)
            assign_components.append(comp_regions)

        if len(assign_components) <= 1:
            break

        main_comp = max(assign_components, key=len)
        main_tiles = {int(assignments[r]) for r in main_comp}

        # Non-main components that should be connected (not all true islands)
        non_main = [c for c in assign_components if c is not main_comp and not all(r in true_island_regions for r in c)]

        if not non_main:
            break

        if verbose:
            sizes = sorted(len(c) for c in non_main)
            print(f"[fix_islands] Connect: {len(non_main)} disconnected component(s) to fix (sizes: {sizes})")

        fixed_any = False
        for comp_regions in non_main:
            for region in comp_regions:
                candidate_tiles = set()
                for mt in main_tiles:
                    for nb in np.where(tile_adjacency[mt])[0]:
                        nb_int = int(nb)
                        if nb_int not in assigned_set:
                            candidate_tiles.add(nb_int)

                if not candidate_tiles:
                    if verbose:
                        print(f"[fix_islands]   Region {region}: no unoccupied tile adjacent to main cluster")
                    continue

                candidate_list = list(candidate_tiles)

                # Prefer tiles near geographic neighbors in the main cluster
                main_comp_set = set(main_comp)
                geo_neighbor_tiles = []
                if region_adjacency is not None:
                    for nb_r in range(n):
                        if region_adjacency[region, nb_r] > 0 and nb_r in main_comp_set:
                            geo_neighbor_tiles.append(int(assignments[nb_r]))

                target = grid_centers[geo_neighbor_tiles].mean(axis=0) if geo_neighbor_tiles else centroids[region]

                dists = np.linalg.norm(grid_centers[candidate_list] - target, axis=1)
                best_tile = candidate_list[int(np.argmin(dists))]

                old_tile = int(assignments[region])
                assignments[region] = best_tile
                assigned_set.discard(old_tile)
                assigned_set.add(best_tile)
                main_tiles.add(best_tile)
                fixed_any = True

                if verbose:
                    print(f"[fix_islands]   Connect region {region}: tile {old_tile} → {best_tile}")

        if not fixed_any:
            break

    # ---------------------------------------------------------------
    # Step 2: Disconnect true island regions that are attached to main
    # ---------------------------------------------------------------
    if not true_island_regions:
        return assignments

    # Recompute main cluster tiles after step 1
    assigned_set = set(assignments.tolist())

    # Find which tiles belong to the main cluster (mainland regions)
    mainland_tiles: set[int] = set()
    for r in range(n):
        if r not in true_island_regions:
            mainland_tiles.add(int(assignments[r]))

    # Expand mainland tiles to include all tiles reachable from them
    # via tile adjacency through assigned tiles (the full main cluster)
    main_cluster: set[int] = set()
    q_init: deque[int] = deque(mainland_tiles)
    main_cluster.update(mainland_tiles)
    while q_init:
        t = q_init.popleft()
        for nb in np.where(tile_adjacency[t])[0]:
            nb_int = int(nb)
            if nb_int not in main_cluster and nb_int in assigned_set:
                main_cluster.add(nb_int)
                q_init.append(nb_int)

    # For each true island region group, check if they're attached
    island_groups = [c for c in region_components if c != mainland_regions] if region_adjacency is not None else []

    for island_group in island_groups:
        # Check if any region in this island group is on a tile
        # that is edge-adjacent to the main cluster
        island_tiles = {int(assignments[r]) for r in island_group}
        attached = any(tile_adjacency[it, mt] for it in island_tiles for mt in main_cluster if it != mt)

        if not attached:
            continue

        if verbose:
            print(f"[fix_islands] Disconnect: island group {sorted(island_group)} is attached to main cluster")

        # Reassign each island region to an unoccupied tile NOT adjacent
        # to the main cluster, closest to its centroid
        for region in island_group:
            # Find unoccupied tiles not adjacent to main cluster
            isolated_tiles: list[int] = []
            for t in range(m):
                if t in assigned_set:
                    continue
                # Check tile is not edge-adjacent to any main cluster tile
                if not any(tile_adjacency[t, mt] for mt in main_cluster):
                    isolated_tiles.append(t)

            if not isolated_tiles:
                if verbose:
                    print(f"[fix_islands]   Region {region}: no isolated tile available")
                continue

            centroid = centroids[region]
            dists = np.linalg.norm(grid_centers[isolated_tiles] - centroid, axis=1)
            best_tile = isolated_tiles[int(np.argmin(dists))]

            old_tile = int(assignments[region])
            assignments[region] = best_tile
            assigned_set.discard(old_tile)
            assigned_set.add(best_tile)

            if verbose:
                print(f"[fix_islands]   Disconnect region {region}: tile {old_tile} → {best_tile}")

    return assignments


def _bfs_distance_matrix(adjacency: NDArray[np.bool_]) -> NDArray[np.int32]:
    """Compute all-pairs shortest-path distances on an adjacency graph via BFS.

    Parameters
    ----------
    adjacency : (m, m) boolean array
        Adjacency matrix.

    Returns
    -------
    (m, m) int32 array
        Shortest-path distances. Disconnected pairs get distance m (max).

    """
    from collections import deque

    m = len(adjacency)
    dists = np.full((m, m), m, dtype=np.int32)
    np.fill_diagonal(dists, 0)

    for source in range(m):
        queue = deque([source])
        visited = np.zeros(m, dtype=bool)
        visited[source] = True
        while queue:
            node = queue.popleft()
            d = dists[source, node]
            neighbors = np.where(adjacency[node])[0]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    dists[source, nb] = d + 1
                    queue.append(nb)

    return dists


def bfs_expand_assignments(
    anchor_assignments: NDArray[np.intp],
    counts: NDArray[np.int32],
    tile_adjacency: NDArray[np.bool_],
    grid_centers: NDArray[np.floating] | None = None,
) -> list[list[int]]:
    """Expand G-level anchor assignments to fill counts[g] tiles per geometry via BFS.

    Parameters
    ----------
    anchor_assignments : (G,) int array
        One anchor tile per geometry from a G-level Hungarian assignment.
    counts : (G,) int array
        Number of tiles each geometry needs (>= 1).
    tile_adjacency : (M, M) bool array
        Edge-based tile adjacency matrix.
    grid_centers : (M, 2) array, optional
        Tile center positions.  Used in the fallback when BFS cannot find
        enough adjacent free tiles: remaining tiles are taken from the
        globally nearest unoccupied ones.  When None, unoccupied tiles are
        taken in index order.

    Returns
    -------
    list of lists
        ``geom_tiles[g]`` is the list of tile indices assigned to geometry g,
        of length ``counts[g]``.
    """
    from collections import deque

    G = len(anchor_assignments)
    M = tile_adjacency.shape[0]

    # Precompute adjacency lists once
    adj_lists: list[list[int]] = [[] for _ in range(M)]
    rows, cols = np.where(tile_adjacency)
    for r, c in zip(rows.tolist(), cols.tolist(), strict=False):
        adj_lists[r].append(c)

    occupied: dict[int, int] = {}  # tile_idx -> geom_idx
    geom_tiles: list[list[int]] = [[] for _ in range(G)]
    for g in range(G):
        t = int(anchor_assignments[g])
        occupied[t] = g
        geom_tiles[g].append(t)

    # Process geometries largest-first so they claim contiguous space first
    order = np.argsort(counts)[::-1].tolist()
    for g in order:
        need = int(counts[g]) - 1  # anchor already claimed
        if need == 0:
            continue
        frontier: deque[int] = deque(geom_tiles[g])
        while need > 0 and frontier:
            t = frontier.popleft()
            for nb in adj_lists[t]:
                if nb not in occupied:
                    occupied[nb] = g
                    geom_tiles[g].append(nb)
                    frontier.append(nb)
                    need -= 1
                    if need == 0:
                        break

        # Fallback: geometry is fully surrounded — grab nearest free tiles globally
        if need > 0:
            free = np.array([t for t in range(M) if t not in occupied], dtype=np.intp)
            if len(free) > 0:
                if grid_centers is not None:
                    anchor_center = grid_centers[int(anchor_assignments[g])]
                    dists = np.linalg.norm(grid_centers[free] - anchor_center, axis=1)
                    free = free[np.argsort(dists)]
                for ft in free[:need].tolist():
                    occupied[ft] = g
                    geom_tiles[g].append(ft)

    return geom_tiles


def assign_to_grid_hungarian(
    centroids: NDArray[np.floating],
    grid_centers: NDArray[np.floating],
    adjacency: NDArray[np.floating] | None = None,
    tile_adjacency: NDArray[np.bool_] | None = None,
    vertex_adjacency: NDArray[np.bool_] | None = None,
    *,
    origin_weight: float = 1.0,
    neighbor_weight: float = 0.3,
    topology_weight: float = 0.0,
    compactness: float = 0.0,
    source_indices: NDArray[np.intp] | None = None,
) -> NDArray[np.intp]:
    """Optimal assignment of regions to grid cells using Hungarian algorithm.

    The cost function combines four terms (all normalized to [0, 1]):

    - **Origin cost**: Squared Euclidean distance from each region's centroid
      to each grid cell. Weight: ``origin_weight``.
    - **Neighbor cost**: Squared BFS hop distance for each adjacent pair
      (shifted by 1 so edge neighbors cost 0, hop 2 costs 1, hop 3
      costs 4, etc.). Vertex-adjacent tiles get a reduced penalty of
      0.2. Not normalized by max — ``neighbor_weight`` directly scales
      the raw squared-hop cost.
    - **Topology cost**: For each adjacent pair, penalizes assignments that
      reverse the relative direction between neighbors. Weight:
      ``topology_weight``.
    - **Compactness cost**: Penalizes assignments far from the grid center.
      Weight: ``compactness``.

    Parameters
    ----------
    centroids : np.ndarray of shape (n, 2)
        Original region centroids.
    grid_centers : np.ndarray of shape (m, 2)
        Grid cell centers (m >= n).
    adjacency : np.ndarray of shape (n, n), optional
        Region adjacency matrix. Required for neighbor and topology costs.
    tile_adjacency : np.ndarray of shape (m, m), optional
        Tile edge adjacency matrix.
    vertex_adjacency : np.ndarray of shape (m, m), optional
        Tile vertex-only adjacency matrix (tiles sharing exactly one vertex
        but no edge).
    origin_weight : float
        Weight for origin (centroid distance) cost. Default: 1.0.
    neighbor_weight : float
        Weight for neighbor cost. Default: 0.3.
    topology_weight : float
        Weight for neighbor orientation cost. Default: 0.0.
    compactness : float
        Weight for compactness cost. Default: 0.0.

    Returns
    -------
    np.ndarray of shape (n,)
        Indices into grid_centers for each centroid.

    """
    n = len(centroids)
    m = len(grid_centers)

    # --- Origin cost: distance from centroids to grid cells ---
    origin_cost = cdist(centroids, grid_centers, metric="sqeuclidean")
    max_dist = np.max(origin_cost)
    if max_dist > 0:
        origin_cost = origin_cost / max_dist

    # --- Compactness cost: distance from grid center ---
    if compactness > 0:
        grid_centroid = grid_centers.mean(axis=0)
        compact_dists = np.sum((grid_centers - grid_centroid) ** 2, axis=1)
        max_compact = np.max(compact_dists)
        if max_compact > 0:
            compact_dists = compact_dists / max_compact
        # Broadcast to (n, m) — same for all regions
        compactness_cost = np.broadcast_to(compact_dists[np.newaxis, :], (n, m))
    else:
        compactness_cost = np.zeros((n, m))

    # --- Neighbor costs (require adjacency and iterative refinement) ---
    has_neighbor_costs = adjacency is not None and (neighbor_weight > 0 or topology_weight > 0)

    if has_neighbor_costs:
        assert adjacency is not None  # noqa: S101 — guaranteed by has_neighbor_costs condition
        # Pre-compute BFS hop distances for distance-scaled neighbor cost
        bfs_dists = _bfs_distance_matrix(tile_adjacency) if tile_adjacency is not None and neighbor_weight > 0 else None

        # Pre-compute original direction angles for orientation cost
        if topology_weight > 0:
            orig_angles = np.full((n, n), np.nan)
            for i in range(n):
                for j in range(n):
                    if i != j and adjacency[i, j] > 0:
                        # Skip within-group pairs: same source geometry → zero diff → degenerate angle
                        if source_indices is not None and source_indices[i] == source_indices[j]:
                            continue
                        diff = centroids[j] - centroids[i]
                        orig_angles[i, j] = np.arctan2(diff[1], diff[0])

        # Initial assignment: always use origin cost for a geographically
        # sensible starting point, even when origin_weight=0 in the final
        # cost. Without this, the initial assignment is essentially random
        # and iterative refinement can't bootstrap.
        init_origin_w = max(origin_weight, 1.0)
        base_cost = init_origin_w * origin_cost + compactness * compactness_cost
        _, assignment = linear_sum_assignment(base_cost)

        # Iterative refinement (up to 10 passes, stop if converged)
        for _iter in range(10):
            neighbor_cost = np.zeros((n, m))
            orientation_cost = np.zeros((n, m))

            for i in range(n):
                for j in range(n):
                    if i == j or adjacency[i, j] <= 0:
                        continue
                    j_cell = assignment[j]
                    w = adjacency[i, j]

                    # Distance-scaled neighbor penalty (squared hops):
                    # Edge neighbors (hop 1) cost 0, hop 2 costs 1,
                    # hop 3 costs 4, hop 5 costs 16, etc.
                    # Not normalized by max — neighbor_weight directly
                    # controls the cost scale.
                    if neighbor_weight > 0:
                        if bfs_dists is not None:
                            raw = np.maximum(bfs_dists[:, j_cell] - 1, 0).astype(float)
                            penalty = raw * raw
                        else:
                            penalty = np.ones(m)
                        if vertex_adjacency is not None:
                            penalty[vertex_adjacency[:, j_cell]] = 0.2
                        if tile_adjacency is not None:
                            penalty[tile_adjacency[:, j_cell]] = 0.0
                        penalty[j_cell] = 0.0
                        neighbor_cost[i, :] += penalty * w

                    # Orientation: penalize angular error (skip within-group pairs — nan angle)
                    if topology_weight > 0:
                        orig_angle = orig_angles[i, j]
                        if np.isnan(orig_angle):
                            continue
                        for k in range(m):
                            diff = grid_centers[j_cell] - grid_centers[k]
                            d = np.linalg.norm(diff)
                            if d < 1e-12:
                                # Same cell → max angular penalty
                                orientation_cost[i, k] += w
                            else:
                                grid_angle = np.arctan2(diff[1], diff[0])
                                # Angular error in [0, pi] → [0, 1]
                                angle_err = abs(orig_angle - grid_angle)
                                angle_err = min(angle_err, 2 * np.pi - angle_err)
                                orientation_cost[i, k] += (angle_err / np.pi) * w

            # Reverse neighbor penalty: for each candidate tile k, count
            # how many of k's tile-neighbors are occupied by regions
            # that are NOT geographic neighbors of region i.
            if neighbor_weight > 0 and tile_adjacency is not None:
                reverse_cost = np.zeros((n, m))
                # Build tile→region lookup
                tile_to_region = np.full(m, -1, dtype=int)
                for r_idx in range(n):
                    tile_to_region[assignment[r_idx]] = r_idx
                for k in range(m):
                    for t_nb in np.where(tile_adjacency[k])[0]:
                        j_region = tile_to_region[int(t_nb)]
                        if j_region < 0:
                            continue
                        # All regions not adjacent to j_region get penalty
                        non_adj = adjacency[:, j_region] <= 0
                        non_adj[j_region] = False
                        reverse_cost[non_adj, k] += 1.0
            else:
                reverse_cost = np.zeros((n, m))

            # Normalize orientation cost to [0, 1].
            # neighbor_cost and reverse_cost are NOT normalized — their
            # values are directly scaled by neighbor_weight.
            max_oc = np.max(orientation_cost)
            if max_oc > 0:
                orientation_cost /= max_oc

            combined = (
                origin_weight * origin_cost
                + neighbor_weight * (neighbor_cost + reverse_cost)
                + topology_weight * orientation_cost
                + compactness * compactness_cost
            )
            prev_assignment = assignment
            _, assignment = linear_sum_assignment(combined)

            # Converged if assignment didn't change
            if np.array_equal(assignment, prev_assignment):
                break

        return assignment

    # No neighbor costs — solve with origin + compactness only
    combined = origin_weight * origin_cost + compactness * compactness_cost
    _, col_ind = linear_sum_assignment(combined)
    return col_ind
