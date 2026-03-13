"""Placement algorithms for symbol cartograms."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm


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


def resolve_circle_overlaps(
    positions: NDArray[np.floating],
    radii: NDArray[np.floating],
    *,
    spacing: float = 0.05,
    max_iterations: int = 20,
    overlap_tolerance: float = 1e-4,
    global_step_fraction: float = 0.5,
    local_step_fraction: float = 0.5,
    max_expansion_factor: float = 2.0,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.floating], dict[str, Any]]:
    """Resolve circle overlaps via global expansion + local separation.

    Alternates partial global expansion from the weighted centroid with
    partial local pairwise separation until all overlaps are resolved.

    This is a standalone function extracted from TopologyPreservingSimulator
    for reuse in simple layout algorithms that only need overlap resolution.

    Parameters
    ----------
    positions : (n, 2) array
        Circle center positions.
    radii : (n,) array
        Circle radii.
    spacing : float
        Minimum gap as fraction of average radius. Default: 0.05
    max_iterations : int
        Maximum outer iterations. Default: 20
    overlap_tolerance : float
        Convergence tolerance as fraction of average radius. Default: 1e-4
    global_step_fraction : float
        Fraction of global expansion to apply per iteration (0-1]. Default: 0.5
    local_step_fraction : float
        Fraction of local separation to apply per iteration (0-1]. Default: 0.5
    max_expansion_factor : float
        Maximum expansion factor clamp per iteration. Must be > 1.0. Default: 2.0
    rng : np.random.Generator, optional
        Random number generator for coincident circle handling.
        If None, a default generator with seed 42 is used.

    Returns
    -------
    positions : (n, 2) array
        Resolved positions with no overlaps (or minimal remaining overlap).
    info : dict
        Statistics with keys:
        - "iterations": Number of iterations performed
        - "final_max_overlap": Maximum remaining overlap as fraction of avg radius

    Examples
    --------
    >>> positions = np.array([[0, 0], [0.5, 0], [1, 0]])
    >>> radii = np.array([0.4, 0.4, 0.4])
    >>> resolved, info = resolve_circle_overlaps(positions, radii, spacing=0.1)
    >>> info["iterations"]
    3

    """
    n = len(positions)
    if n == 0:
        return positions.copy(), {"iterations": 0, "final_max_overlap": 0.0}

    # Use provided RNG or create default
    if rng is None:
        rng = np.random.default_rng(42)

    # Work on a copy
    positions = positions.astype(float).copy()
    radii = radii.astype(float).copy()

    # Compute scale factor to normalize to unit-ish coordinates
    all_mins = positions - radii[:, None]
    all_maxs = positions + radii[:, None]
    min_coords = all_mins.min(axis=0)
    max_coords = all_maxs.max(axis=0)
    center = (min_coords + max_coords) / 2
    extent = np.max(max_coords - min_coords)
    scale = extent if extent > 0 else 1.0

    # Normalize positions and radii
    positions_norm = (positions - center) / scale
    radii_norm = radii / scale

    # Compute spacing in normalized coordinates
    avg_radius = float(np.mean(radii_norm))
    spacing_norm = spacing * avg_radius
    tol_overlap = overlap_tolerance * avg_radius

    # Precompute upper triangular indices for vectorized operations
    i_idx, j_idx = np.triu_indices(n, k=1)
    all_radii_sum = radii_norm[i_idx] + radii_norm[j_idx]

    def weighted_centroid() -> NDArray[np.floating]:
        """Compute area-weighted centroid."""
        weights = radii_norm**2
        return (positions_norm * weights[:, None]).sum(axis=0) / weights.sum()

    def global_expansion_factor() -> float:
        """Compute the exact global expansion factor to resolve all overlaps."""
        diff = positions_norm[j_idx] - positions_norm[i_idx]
        d = np.linalg.norm(diff, axis=1)
        d_min = all_radii_sum + spacing_norm

        valid = (d > 1e-10) & (d < d_min)
        if not np.any(valid):
            return 1.0

        s_pairs = d_min[valid] / d[valid]
        return float(np.max(s_pairs))

    def separate_overlapping_pairs_partial(step_fraction: float = 0.5) -> float:
        """Single pass pushing overlapping circles apart with partial steps."""
        diff = positions_norm[j_idx] - positions_norm[i_idx]
        d = np.linalg.norm(diff, axis=1)
        d_min = all_radii_sum + spacing_norm
        overlap = d_min - d

        overlap_mask = overlap > 0
        if not np.any(overlap_mask):
            return 0.0

        max_overlap = float(np.max(overlap[overlap_mask]))

        idx = np.where(overlap_mask)[0]
        diff_overlap = diff[idx]
        d_overlap = d[idx]
        overlap_vals = overlap[idx]

        valid_dist = d_overlap > 1e-10
        directions = np.zeros((len(idx), 2))

        if np.any(valid_dist):
            directions[valid_dist] = diff_overlap[valid_dist] / d_overlap[valid_dist, None]

        if np.any(~valid_dist):
            n_coincident = int(np.sum(~valid_dist))
            random_dirs = rng.standard_normal((n_coincident, 2))
            random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
            directions[~valid_dist] = random_dirs

        shifts = step_fraction * 0.5 * overlap_vals[:, None] * directions

        np.add.at(positions_norm, i_idx[idx], -shifts)
        np.add.at(positions_norm, j_idx[idx], shifts)

        return max_overlap

    # Main iteration loop
    C = weighted_centroid()
    max_overlap = 0.0
    iterations_done = 0

    for _ in range(max_iterations):
        iterations_done += 1

        # 1. Compute exact global expansion factor
        s_global = global_expansion_factor()

        # 2. Apply partial global expansion (clamped)
        s_global = min(s_global, max_expansion_factor)
        if s_global > 1.0:
            s_applied = 1.0 + global_step_fraction * (s_global - 1.0)
            positions_norm = C + s_applied * (positions_norm - C)

        # 3. Single-pass partial local separation
        max_overlap = separate_overlapping_pairs_partial(step_fraction=local_step_fraction)

        # 4. Check convergence
        if max_overlap < tol_overlap:
            break

    # Convert back to original coordinate system
    final_positions = positions_norm * scale + center

    return final_positions, {
        "iterations": iterations_done,
        "final_max_overlap": max_overlap,
    }


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

                    # Orientation: penalize angular error
                    if topology_weight > 0:
                        orig_angle = orig_angles[i, j]
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


class CirclePhysicsSimulator:
    """Physics-based simulator for resolving circle overlaps.

    Uses a two-phase approach:
    1. Separation phase: Strong repulsion only (no attraction) until overlaps resolved
    2. Settling phase: Gentle attraction while maintaining separation

    Internally normalizes to unit scale for numerical stability.
    """

    def __init__(
        self,
        positions: NDArray[np.floating],
        radii: NDArray[np.floating],
        original_positions: NDArray[np.floating] | None = None,
        adjacency: NDArray[np.floating] | None = None,
        spacing: float = 0.05,
        compactness: float = 0.5,
        topology_weight: float = 0.3,
        damping: float = 0.85,
        dt: float = 0.15,
        max_velocity: float = 3.0,
        k_repel: float = 15.0,
        k_attract: float = 2.0,
    ):
        """Initialize simulator.

        Parameters
        ----------
        compactness : float
            Balance between centroid-attraction and neighbor-attraction (0-1).
            0 = symbols attracted only to original centroids
            1 = symbols attracted only to neighbors (tight cluster)
        damping : float
            Velocity damping factor (0-1). Default: 0.85
        dt : float
            Integration timestep. Default: 0.15
        max_velocity : float
            Maximum velocity magnitude. Default: 3.0
        k_repel : float
            Repulsion force coefficient. Default: 15.0
        k_attract : float
            Base attraction force coefficient. Default: 2.0

        """
        self.n = len(positions)
        self.adjacency = adjacency
        self.damping = damping
        self.compactness = compactness
        self.topology_weight = topology_weight

        # Handle None original_positions
        if original_positions is None:
            original_positions = positions.copy()

        # Compute scale factor to normalize to unit-ish coordinates
        # Use bounding box that includes full circle extent (position ± radius)
        all_mins = np.vstack([positions - radii[:, None], original_positions - radii[:, None]])
        all_maxs = np.vstack([positions + radii[:, None], original_positions + radii[:, None]])
        min_coords = all_mins.min(axis=0)
        max_coords = all_maxs.max(axis=0)
        self.center = (min_coords + max_coords) / 2
        extent = np.max(max_coords - min_coords)
        self.scale = extent if extent > 0 else 1.0

        # Store normalized positions
        self.positions = (positions.astype(float) - self.center) / self.scale
        self.original_positions = (original_positions.astype(float) - self.center) / self.scale
        self.radii = radii.astype(float) / self.scale
        self.velocities = np.zeros_like(self.positions)

        # Compute spacing in normalized coordinates
        self.avg_radius = float(np.mean(self.radii))
        self.spacing = spacing * self.avg_radius

        # Physics parameters
        self.dt = dt
        self.max_velocity = max_velocity
        self.k_repel = k_repel
        self.k_attract = k_attract

    def _count_overlaps(self) -> int:
        """Count current number of overlapping pairs."""
        count = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                diff = self.positions[j] - self.positions[i]
                dist = float(np.linalg.norm(diff))
                min_dist = self.radii[i] + self.radii[j] + self.spacing
                if dist < min_dist:
                    count += 1
        return count

    def step(self, apply_attraction: bool = False) -> tuple[int, float, float]:
        """Perform one simulation step.

        Parameters
        ----------
        apply_attraction : bool
            Whether to apply attraction toward original positions.

        Returns
        -------
        n_overlaps : int
            Number of overlapping pairs.
        max_velocity : float
            Maximum velocity magnitude.
        max_gap_ratio : float
            Maximum gap ratio (how far neighbors are from touching).

        """
        forces = np.zeros_like(self.positions)

        # 1. Repulsive forces from overlaps (always applied)
        n_overlaps = self._compute_repulsion(forces)

        # 2. Attractive forces (only when requested and no overlaps)
        # Always compute gap_ratio for tracking, but only apply forces when no overlaps
        gap_ratio = self._compute_gap_ratio() if self.adjacency is not None else 0.0

        if apply_attraction and n_overlaps == 0:
            self._compute_origin_attraction(forces)
            if self.adjacency is not None and self.compactness > 0:
                self._compute_neighbor_attraction(forces, gap_ratio)

        # 3. Clamp forces
        force_magnitudes = np.linalg.norm(forces, axis=1, keepdims=True)
        max_force = 30.0  # High limit to allow fast overlap resolution
        scale_factors = np.where(force_magnitudes > max_force, max_force / (force_magnitudes + 1e-10), 1.0)
        forces *= scale_factors

        # 4. Integration with velocity clamping
        self.velocities += forces * self.dt

        # Adaptive damping based on phase
        if n_overlaps > 0:
            # Separation phase: low damping for fast overlap resolution
            effective_damping = 0.5
        elif gap_ratio >= 0:
            if gap_ratio < 0.02:
                # Very close to target - higher damping to allow final approach
                effective_damping = 0.7
            elif gap_ratio < 0.1:
                # Close - moderate damping to prevent oscillation
                effective_damping = 0.5
            else:
                # Far from target - low damping for fast movement
                effective_damping = 0.5
        else:
            effective_damping = self.damping

        self.velocities *= effective_damping

        vel_magnitudes = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        scale_factors = np.where(vel_magnitudes > self.max_velocity, self.max_velocity / (vel_magnitudes + 1e-10), 1.0)
        self.velocities *= scale_factors

        self.positions += self.velocities * self.dt

        # Check for NaN/Inf
        if not np.all(np.isfinite(self.positions)):
            self.positions = self.original_positions.copy()
            self.velocities = np.zeros_like(self.positions)
            return n_overlaps, 0.0, gap_ratio

        max_velocity = float(np.max(np.linalg.norm(self.velocities, axis=1)))
        return n_overlaps, max_velocity, gap_ratio

    def _compute_repulsion(self, forces: NDArray[np.floating]) -> int:
        """Compute repulsive forces between overlapping circles."""
        n_overlaps = 0

        for i in range(self.n):
            for j in range(i + 1, self.n):
                diff = self.positions[j] - self.positions[i]
                dist = float(np.linalg.norm(diff))
                min_dist = self.radii[i] + self.radii[j] + self.spacing

                if dist < min_dist:
                    if dist > 1e-10:
                        n_overlaps += 1
                        overlap = min_dist - dist

                        # Non-linear force for deep overlaps
                        overlap_ratio = overlap / min_dist
                        force_multiplier = 1.0 + overlap_ratio * 3.0

                        direction = diff / dist
                        # Ensure minimum force even for tiny overlaps
                        min_repel_force = 0.5 * self.k_repel * self.avg_radius
                        force_mag = max(self.k_repel * overlap * force_multiplier, min_repel_force)
                        force = force_mag * direction
                        forces[i] -= force
                        forces[j] += force
                    else:
                        # Exactly overlapping - strong random push
                        random_dir = np.random.randn(2)
                        norm = float(np.linalg.norm(random_dir))
                        if norm > 1e-10:
                            random_dir /= norm
                        else:
                            random_dir = np.array([1.0, 0.0])
                        force = self.k_repel * min_dist * 2.0 * random_dir  # Stronger push
                        forces[i] -= force
                        forces[j] += force
                        n_overlaps += 1

        return n_overlaps

    def _compute_origin_attraction(self, forces: NDArray[np.floating]) -> None:
        """Attractive force toward original positions (weighted by 1 - compactness)."""
        displacement = self.original_positions - self.positions
        weight = (1 - self.compactness) * self.k_attract
        forces += weight * displacement

    def _compute_gap_ratio(self) -> float:
        """Compute max gap ratio between adjacent symbols (without applying forces)."""
        if self.adjacency is None:
            return 0.0

        max_gap_ratio = 0.0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.adjacency[i, j] > 0:
                    diff = self.positions[j] - self.positions[i]
                    dist = float(np.linalg.norm(diff))
                    target_dist = self.radii[i] + self.radii[j] + self.spacing
                    if dist > 1e-10:
                        gap_ratio = (dist - target_dist) / target_dist
                        max_gap_ratio = max(max_gap_ratio, gap_ratio)
        return max_gap_ratio

    def _compute_neighbor_attraction(self, forces: NDArray[np.floating], current_gap_ratio: float) -> None:
        """Attractive forces between original neighbors (weighted by compactness)."""
        if self.adjacency is None:
            return

        for i in range(self.n):
            for j in range(i + 1, self.n):
                adj_weight = self.adjacency[i, j]
                if adj_weight > 0:
                    diff = self.positions[j] - self.positions[i]
                    dist = float(np.linalg.norm(diff))

                    if dist > 1e-10:
                        direction = diff / dist
                        target_dist = self.radii[i] + self.radii[j] + self.spacing

                        # Attract if farther than touching distance
                        if dist > target_dist:
                            gap = dist - target_dist
                            local_gap_ratio = gap / target_dist

                            # Scale force based on gap: stronger when far, min force when close
                            if local_gap_ratio > 0.5:
                                # Far apart: use strong constant force to pull together quickly
                                force_mag = self.compactness * self.k_attract * adj_weight * 0.5
                            elif local_gap_ratio < 0.1:
                                # Close: use minimum force to prevent stalling
                                min_force = 0.1 * self.k_attract
                                force_mag = max(
                                    self.compactness * self.k_attract * adj_weight * gap,
                                    min_force * self.compactness * adj_weight,
                                )
                            else:
                                # Medium range: proportional force
                                force_mag = self.compactness * self.k_attract * adj_weight * gap
                            force = force_mag * direction
                            forces[i] += force
                            forces[j] -= force

    def run(
        self,
        max_iterations: int = 500,
        tolerance: float = 1e-4,
        show_progress: bool = True,
        save_history: bool = False,
    ) -> tuple[NDArray[np.floating], dict[str, Any], list[NDArray[np.floating]] | None]:
        """Run simulation until convergence or max iterations.

        Returns
        -------
        positions : np.ndarray
            Final symbol positions (in original coordinate system).
        info : dict
            Simulation statistics.
        history : list or None
            Position history if save_history=True (in original coordinates).

        """
        history: list[NDArray[np.floating]] | None = [] if save_history else None
        velocity_history: list[float] = []
        overlap_history: list[int] = []

        iterator: Any = range(max_iterations)
        if show_progress:
            iterator = tqdm(iterator, desc="Resolving overlaps", leave=False)

        converged = False
        final_overlaps = 0
        final_velocity = 0.0
        iterations_run = 0
        overlap_free_streak = 0

        for _ in iterator:
            iterations_run += 1
            if history is not None:
                history.append(self.positions * self.scale + self.center)

            # Always pass apply_attraction=True - step() internally only applies
            # attraction when n_overlaps == 0
            n_overlaps, max_velocity, gap_ratio = self.step(apply_attraction=True)
            velocity_history.append(max_velocity)
            overlap_history.append(n_overlaps)

            if overlap_free_streak >= 10:
                pass
            final_overlaps = n_overlaps
            final_velocity = max_velocity

            # Track overlap-free iterations
            if n_overlaps == 0:
                overlap_free_streak += 1
            else:
                overlap_free_streak = 0

            if show_progress:
                # Phase is "settling" when no overlaps (attraction was applied)
                phase = "settling" if n_overlaps == 0 else "separating"
                postfix = {"phase": phase, "overlaps": n_overlaps}
                if n_overlaps == 0:
                    postfix["gap"] = f"{gap_ratio:.1%}"
                postfix["vel"] = f"{max_velocity:.2e}"
                iterator.set_postfix(postfix)

            # Check convergence
            if n_overlaps == 0:
                if self.adjacency is not None and self.compactness > 0:
                    # With neighbor attraction: converge when neighbors are close to touching
                    # gap_ratio < 0.02 means within 2% of target distance
                    if gap_ratio < 0.02:
                        converged = True
                        break
                # No neighbor attraction - converge on velocity
                elif max_velocity < tolerance:
                    converged = True
                    break

        info = {
            "iterations": iterations_run,
            "converged": converged,
            "final_overlaps": final_overlaps,
            "final_max_velocity": final_velocity,
            "velocity_history": velocity_history,
            "overlap_history": overlap_history,
        }

        final_positions = self.positions * self.scale + self.center
        return final_positions, info, history


class ExponentialMovingStats:
    """EMA tracker for mean and std of vector-valued observations.

    Maintains per-element exponential moving averages of mean and variance
    for vector-valued time series. Used for steady-state detection by
    tracking displacement vectors over iterations.

    Parameters
    ----------
    n : int
        Number of elements (e.g., circles).
    dim : int
        Vector dimension (e.g., 2 for 2D displacement).
    n_eff : int
        Effective window size. Alpha = 2 / (n_eff + 1).
    adaptive : bool
        If True, alpha starts at 1/k and decays to the fixed value,
        providing faster initial convergence.

    """

    def __init__(self, n: int, dim: int, n_eff: int = 20, *, adaptive: bool = True):
        self.alpha = 2.0 / (n_eff + 1)
        self.mean = np.zeros((n, dim))
        self.var = np.zeros((n, dim))
        self._initialized = False
        self._adaptive = adaptive
        self._k = 0

    def update(self, x: NDArray[np.floating]) -> None:
        """Update with new observation x of shape (n, dim)."""
        if not self._initialized:
            self.mean[:] = x
            self._initialized = True
            self._k = 1
            return
        self._k += 1
        a = max(self.alpha, 1.0 / self._k) if self._adaptive else self.alpha
        delta = x - self.mean
        self.mean += a * delta
        self.var[:] = (1 - a) * (self.var + a * delta**2)

    @property
    def mean_magnitude(self) -> NDArray[np.floating]:
        """Per-element magnitude of mean vector: ||mu_i||."""
        return np.linalg.norm(self.mean, axis=1)

    @property
    def std_magnitude(self) -> NDArray[np.floating]:
        """Per-element std magnitude: sqrt(sum of component variances)."""
        return np.sqrt(np.sum(self.var, axis=1))


class ScalarEMA:
    """EMA tracker for a single scalar value.

    Parameters
    ----------
    n_eff : int
        Effective window size. Alpha = 2 / (n_eff + 1).
    adaptive : bool
        If True, alpha starts at 1/k and decays to the fixed value.
    initial_value : float
        Value before the first update. Default: 0.0

    """

    def __init__(
        self,
        n_eff: int = 20,
        *,
        adaptive: bool = True,
        initial_value: float = 0.0,
    ):
        self.alpha = 2.0 / (n_eff + 1)
        self.value: float = initial_value
        self._initialized = False
        self._adaptive = adaptive
        self._k = 0

    def update(self, x: float) -> None:
        """Update with new scalar observation."""
        if not self._initialized:
            self.value = x
            self._initialized = True
            self._k = 1
            return
        self._k += 1
        a = max(self.alpha, 1.0 / self._k) if self._adaptive else self.alpha
        self.value += a * (x - self.value)


class TopologyPreservingSimulator:
    """Two-phase force-based simulator with topology preservation.

    Unlike CirclePhysicsSimulator which uses velocity-based physics,
    this simulator uses:

    - **Overlap Resolution Phase**: Global expansion + overlap projection
      to reach a non-overlapping state
    - **Packing Phase**: Force-based refinement with:
      - Distance-gated angular topology force (preserves neighbor directions)
      - Neighbor tangency spring (pulls separated neighbors together)
      - Global centroid attraction force (pulls toward original centroid)
      - Origin attraction force (pulls each circle toward its original position)
      - Iterative contact reaction (handles compressive forces at contacts)

    The contact reaction constraint allows circles to slide along each
    other without penetrating, enabling tighter packing while maintaining
    topology.

    Parameters
    ----------
    positions : NDArray[np.floating]
        Initial symbol positions, shape (n, 2).
    radii : NDArray[np.floating]
        Symbol radii, shape (n,).
    original_positions : NDArray[np.floating] | None
        Original centroid positions for topology reference.
    adjacency : NDArray[np.floating] | None
        Adjacency matrix of shape (n, n). Required for topology forces.
    spacing : float
        Minimum gap as fraction of average radius. Default: 0.05
    compactness : float
        Global compaction strength (0-1). Default: 0.5
    topology_weight : float
        Topology preservation strength (0-1). Default: 0.3
    overlap_tolerance : float
        Overlap tolerance for overlap resolution convergence, as fraction of
        average radius. Default: 1e-4
    expansion_max_iterations : int
        Maximum outer iterations for overlap resolution. Default: 20
    max_expansion_factor : float
        Maximum expansion factor clamp per iteration. Default: 2.0
    topology_gate_distance : float
        Topology force gate distance (in sum of radii). Default: 2.5
    neighbor_weight : float
        Neighbor tangency force coefficient. Default: 0.5
    origin_weight : float
        Origin attraction force strength. Pulls each circle toward its
        original position. Set to 0 to disable. Default: 0.0
        - 0: No origin attraction (current behavior)
        - 0.1-0.5: Gentle pull toward original positions
        - > 1.0: Strong pull, may interfere with topology preservation
    force_mode : str
        How attraction force magnitude is computed. Applies to both the
        global centroid attraction force and the origin attraction force.
        Default: "direction"
        - "direction": Constant magnitude with drop-off near target
        - "linear": Force proportional to distance (spring)
        - "normalized": Force proportional to distance / radius
    contact_tolerance : float
        Contact detection tolerance (fraction of sum of radii). Default: 0.02
    contact_iterations : int
        Number of contact reaction passes per packing step. Default: 3
    max_step : float
        Maximum step size (fraction of avg radius). Default: 0.3
    contact_transfer_ratio : float
        Balance between cancel (0) and transfer (1) of compressive forces
        at contact points. Default: 0.5
    contact_elasticity : float
        Controls net compression vs bounce behavior (-1 to 1). Default: 0.0
    size_sensitivity : float
        Controls how step size scales with circle radius. Default: 0.0
    overlap_projection_iters : int
        Overlap projection iterations per packing step. Default: 5
    step_smoothing_window : int
        EMA window for step smoothing. Default: 20
    convergence_window : int
        EMA window for displacement convergence tracking. Default: 50
    adaptive_ema : bool
        Whether EMA uses adaptive warmup (alpha starts at 1/k). Default: True

    Notes
    -----
    The topology force uses distance gating: it only acts when circles
    are within ``topology_gate_distance * (r_i + r_j)`` distance. This prevents distant
    circles from exerting topology forces, focusing preservation on
    local relationships.

    """

    def __init__(
        self,
        positions: NDArray[np.floating],
        radii: NDArray[np.floating],
        original_positions: NDArray[np.floating] | None = None,
        adjacency: NDArray[np.floating] | None = None,
        spacing: float = 0.05,
        compactness: float = 0.5,
        topology_weight: float = 0.3,
        overlap_tolerance: float = 1e-4,
        expansion_max_iterations: int = 20,
        max_expansion_factor: float = 2.0,
        topology_gate_distance: float = 2.5,
        neighbor_weight: float = 0.5,
        origin_weight: float = 0.0,
        force_mode: str = "direction",
        contact_tolerance: float = 0.02,
        contact_iterations: int = 3,
        max_step: float = 0.3,
        contact_transfer_ratio: float = 0.5,
        contact_elasticity: float = 0.0,
        size_sensitivity: float = 0.0,
        global_step_fraction: float = 0.5,
        local_step_fraction: float = 0.5,
        overlap_projection_iters: int = 5,
        step_smoothing_window: int = 20,
        convergence_window: int = 50,
        adaptive_ema: bool = True,
    ):
        self.n = len(positions)

        # Handle None original_positions
        if original_positions is None:
            original_positions = positions.copy()

        # Compute scale factor to normalize to unit-ish coordinates
        # Use bounding box that includes full circle extent (position ± radius)
        all_mins = np.vstack([positions - radii[:, None], original_positions - radii[:, None]])
        all_maxs = np.vstack([positions + radii[:, None], original_positions + radii[:, None]])
        min_coords = all_mins.min(axis=0)
        max_coords = all_maxs.max(axis=0)
        self.center = (min_coords + max_coords) / 2
        extent = np.max(max_coords - min_coords)
        self.scale = extent if extent > 0 else 1.0

        # Store normalized positions
        self.positions = (positions.astype(float) - self.center) / self.scale
        self.original_positions = (original_positions.astype(float) - self.center) / self.scale
        self.radii = radii.astype(float) / self.scale

        # Compute spacing in normalized coordinates
        self.avg_radius = float(np.mean(self.radii))
        self.spacing = spacing * self.avg_radius

        # Store adjacency and extract pairs with per-pair weights
        self.adjacency = adjacency
        self.adjacency_pairs: list[tuple[int, int]] = []
        self._adj_weight_list: list[float] = []
        if adjacency is not None:
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    w_ij = adjacency[i, j]
                    w_ji = adjacency[j, i]
                    if w_ij > 0 or w_ji > 0:
                        self.adjacency_pairs.append((i, j))
                        # Symmetric max for asymmetric matrices
                        self._adj_weight_list.append(max(w_ij, w_ji))

        # Store parameters
        self.compactness = compactness
        self.topology_weight = topology_weight
        self.overlap_tolerance = overlap_tolerance
        self.expansion_max_iterations = expansion_max_iterations
        self.max_expansion_factor = max_expansion_factor
        self.topology_gate_distance = topology_gate_distance
        self.neighbor_weight = neighbor_weight
        self.origin_weight = origin_weight
        self.force_mode = force_mode
        self.contact_tolerance = contact_tolerance
        self.contact_iterations = contact_iterations
        self.max_step = max_step
        self.contact_transfer_ratio = np.clip(contact_transfer_ratio, 0, 1)
        self.contact_elasticity = np.clip(contact_elasticity, -1, 1)
        self.size_sensitivity = size_sensitivity
        self.global_step_fraction = global_step_fraction
        self.local_step_fraction = local_step_fraction
        self.overlap_projection_iters = overlap_projection_iters
        self.step_smoothing_window = step_smoothing_window
        self.convergence_window = convergence_window
        self.adaptive_ema = adaptive_ema

        # EMA accumulators (reset at start of each run)
        self._reset_ema_state()
        # Use local RNG for reproducibility without global state
        self._rng = np.random.default_rng(42)

        # Precomputed data for vectorized force computation
        self.adj_pairs = np.array(self.adjacency_pairs, dtype=np.intp)  # (m, 2)
        self.adj_weights = np.array(self._adj_weight_list, dtype=float)  # (m,)
        self.radii_sum = self.radii[self.adj_pairs[:, 0]] + self.radii[self.adj_pairs[:, 1]]  # (m,)

        # Precompute static topology data (original_positions never changes)
        if self.topology_weight > 0 and len(self.adj_pairs) > 0:
            v0 = self.original_positions[self.adj_pairs[:, 1]] - self.original_positions[self.adj_pairs[:, 0]]  # (m, 2)
            d0 = np.linalg.norm(v0, axis=1)  # (m,)
            self.topo_valid = d0 > 1e-10  # (m,) boolean mask for non-incident circles
            self.u0 = np.empty((len(self.adj_pairs), 2))  # (m, 2)
            self.u0[self.topo_valid] = v0[self.topo_valid] / d0[self.topo_valid, None]  # original unit vectors
        else:
            self.topo_valid = np.zeros(len(self.adj_pairs), dtype=bool)
            self.u0 = np.empty((len(self.adj_pairs), 2))

        # Precompute original centroid for stable global compaction
        weights = self.radii**2
        self.original_centroid = (self.original_positions * weights[:, None]).sum(axis=0) / weights.sum()

        # Precompute upper triangular indices for vectorized overlap projection and contact reaction
        i_idx, j_idx = np.triu_indices(self.n, k=1)
        self.all_pairs_i = i_idx  # (m,)
        self.all_pairs_j = j_idx  # (m,)
        self.all_radii_sum = self.radii[i_idx] + self.radii[j_idx]  # (m,)

    def _reset_ema_state(self) -> None:
        """Reset EMA accumulators for a fresh run."""
        self._step_smooth = ExponentialMovingStats(
            self.n,
            2,
            n_eff=self.step_smoothing_window,
            adaptive=self.adaptive_ema,
        )
        self._disp_stats: ExponentialMovingStats | None = None

    def _weighted_centroid(self) -> NDArray[np.floating]:
        """Compute area-weighted centroid."""
        weights = self.radii**2
        return (self.positions * weights[:, None]).sum(axis=0) / weights.sum()

    def _count_overlaps(self) -> int:
        """Count number of overlapping circle pairs."""
        diff = self.positions[self.all_pairs_j] - self.positions[self.all_pairs_i]
        dist = np.linalg.norm(diff, axis=1)
        min_dist = self.all_radii_sum + self.spacing
        return int(np.sum(dist < min_dist))

    def _separate_overlapping_pairs(self, max_iter: int = 30) -> float:
        """Iteratively push overlapping circles apart.

        Returns the maximum remaining overlap after projection.
        """
        for _ in range(max_iter):
            # Compute pairwise differences using precomputed indices
            diff = self.positions[self.all_pairs_j] - self.positions[self.all_pairs_i]  # (m, 2)
            dist = np.linalg.norm(diff, axis=1)  # (m,)

            # Compute min distances and overlaps using precomputed radii_sum
            min_dist = self.all_radii_sum + self.spacing
            overlap = min_dist - dist  # (m,)

            # Find overlapping pairs
            overlap_mask = overlap > 0
            if not np.any(overlap_mask):
                return 0.0

            # Get max overlap
            max_overlap = float(np.max(overlap[overlap_mask]))

            # Early exit if remaining overlap is tiny
            tol = self.overlap_tolerance * self.avg_radius
            if max_overlap < tol:
                return max_overlap

            # Get indices of overlapping pairs
            idx = np.where(overlap_mask)[0]
            diff_overlap = diff[idx]  # (k, 2)
            dist_overlap = dist[idx]  # (k,)
            overlap_vals = overlap[idx]  # (k,)

            # Compute directions for overlapping pairs
            valid_dist = dist_overlap > 1e-10
            directions = np.zeros((len(idx), 2))

            # For pairs with valid distance, use computed direction
            if np.any(valid_dist):
                directions[valid_dist] = diff_overlap[valid_dist] / dist_overlap[valid_dist, None]

            # For coincident pairs, use random directions
            if np.any(~valid_dist):
                n_coincident = np.sum(~valid_dist)
                random_dirs = self._rng.standard_normal((n_coincident, 2))
                random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
                directions[~valid_dist] = random_dirs

            # Compute shifts
            shifts = 0.5 * overlap_vals[:, None] * directions  # (k, 2)

            # Apply shifts using np.add.at
            np.add.at(self.positions, self.all_pairs_i[idx], -shifts)
            np.add.at(self.positions, self.all_pairs_j[idx], shifts)

        return max_overlap

    def _global_expansion_factor(self) -> float:
        """Compute the exact global expansion factor to resolve all overlaps.

        For each overlapping pair, the expansion factor needed is d_min / d.
        The global factor is the maximum over all pairs: the minimum uniform
        scaling from the centroid that would resolve every overlap.

        Returns 1.0 if no overlaps exist.
        """
        diff = self.positions[self.all_pairs_j] - self.positions[self.all_pairs_i]
        d = np.linalg.norm(diff, axis=1)
        d_min = self.all_radii_sum + self.spacing

        valid = (d > 1e-10) & (d < d_min)
        if not np.any(valid):
            return 1.0

        s_pairs = d_min[valid] / d[valid]
        return float(np.max(s_pairs))

    def _separate_overlapping_pairs_partial(self, step_fraction: float = 0.5) -> float:
        """Single pass pushing overlapping circles apart with partial steps.

        Called once per outer iteration to maintain balance with global
        expansion. The ratio between global_step_fraction and step_fraction
        directly controls relative correction weight.

        Parameters
        ----------
        step_fraction : float
            Fraction of full separation to apply (0-1].

        Returns
        -------
        max_overlap : float
            Maximum remaining overlap after projection.

        """
        diff = self.positions[self.all_pairs_j] - self.positions[self.all_pairs_i]
        d = np.linalg.norm(diff, axis=1)
        d_min = self.all_radii_sum + self.spacing
        overlap = d_min - d

        overlap_mask = overlap > 0
        if not np.any(overlap_mask):
            return 0.0

        max_overlap = float(np.max(overlap[overlap_mask]))

        idx = np.where(overlap_mask)[0]
        diff_overlap = diff[idx]
        d_overlap = d[idx]
        overlap_vals = overlap[idx]

        valid_dist = d_overlap > 1e-10
        directions = np.zeros((len(idx), 2))

        if np.any(valid_dist):
            directions[valid_dist] = diff_overlap[valid_dist] / d_overlap[valid_dist, None]

        if np.any(~valid_dist):
            n_coincident = int(np.sum(~valid_dist))
            random_dirs = self._rng.standard_normal((n_coincident, 2))
            random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
            directions[~valid_dist] = random_dirs

        shifts = step_fraction * 0.5 * overlap_vals[:, None] * directions

        np.add.at(self.positions, self.all_pairs_i[idx], -shifts)
        np.add.at(self.positions, self.all_pairs_j[idx], shifts)

        return max_overlap

    def run_overlap_resolution(self) -> dict[str, Any]:
        """Overlap Resolution Phase: Reach non-overlapping state via balanced global + local separation.

        Alternates partial global expansion (using the exact expansion factor)
        with partial local pairwise separation. The global_step_fraction and
        local_step_fraction parameters control the relative weight of each.

        Can be called independently or as part of :meth:`run`.

        Returns
        -------
        info : dict
            ``{"iterations": int, "final_max_overlap": float}``

        """
        # Convert normalized positions to original coordinates for the standalone function
        original_positions = self.positions * self.scale + self.center
        original_radii = self.radii * self.scale

        # Use the standalone function for overlap resolution
        resolved_positions, info = resolve_circle_overlaps(
            positions=original_positions,
            radii=original_radii,
            spacing=self.spacing / self.avg_radius,  # Convert back to fraction
            max_iterations=self.expansion_max_iterations,
            overlap_tolerance=self.overlap_tolerance,
            global_step_fraction=self.global_step_fraction,
            local_step_fraction=self.local_step_fraction,
            max_expansion_factor=self.max_expansion_factor,
            rng=self._rng,
        )

        # Update internal state with resolved positions (normalized)
        self.positions = (resolved_positions - self.center) / self.scale

        return info

    def _compute_forces(self) -> NDArray[np.floating]:
        """Compute packing forces: topology, neighbor tangency, global centroid attraction, origin attraction.

        Returns the total force array of shape (n, 2).
        """
        F = np.zeros_like(self.positions)

        # --- Distance-gated angular topology force ---
        if self.topology_weight > 0 and np.any(self.topo_valid):
            # Compute current vectors for all pairs
            v = self.positions[self.adj_pairs[:, 1]] - self.positions[self.adj_pairs[:, 0]]  # (m, 2)
            d = np.linalg.norm(v, axis=1)  # (m,)

            # Valid current distances mask (combine with precomputed topo_valid)
            valid = self.topo_valid & (d > 1e-10)
            if np.any(valid):
                # Compute gap (same definition as neighbor force: distance - sum of radii - spacing)
                gap = d[valid] - self.radii_sum[valid] - self.spacing

                # Smooth decay weight: w=1 for gap<=0, w=0 for gap>topology_gate_distance, smooth decay in between
                w = np.clip(1 - gap / self.topology_gate_distance, 0, 1) ** 2  # (k,)

                # Get indices of pairs with non-zero weight
                idx = np.where(valid)[0][w > 0]
                if len(idx) > 0:
                    # Current unit vectors for pairs with non-zero weight
                    u = v[idx] / d[idx, None]  # (p, 2)

                    # Topology force (u0 precomputed), weighted by adjacency
                    Ft = (
                        self.topology_weight * w[w > 0, None] * self.adj_weights[idx, None] * (self.u0[idx] - u)
                    )  # (p, 2)

                    # Accumulate forces using np.add.at for unbuffered accumulation
                    np.add.at(F, self.adj_pairs[idx, 0], -Ft)
                    np.add.at(F, self.adj_pairs[idx, 1], Ft)

        # --- Neighbor tangency force ---
        if len(self.adj_pairs) > 0:
            # Compute all pairwise differences
            dx = self.positions[self.adj_pairs[:, 1]] - self.positions[self.adj_pairs[:, 0]]  # (m, 2)
            d = np.linalg.norm(dx, axis=1)  # (m,)

            # Compute target distance and gap
            target = self.radii_sum + self.spacing
            gap = d - target

            # Positive gap mask (pull together) - also handles d <= 1e-10 since gap would be negative
            gap_mask = gap > 0
            if np.any(gap_mask):
                # Get indices of pairs with positive gap
                idx = np.where(gap_mask)[0]

                # Unit direction
                n_ij = dx[idx] / d[idx, None]  # (p, 2)

                # Strength
                strength = np.minimum(gap[idx] / target[idx], 1.0)  # (p,)

                # Neighbor force, weighted by adjacency
                Fn = self.neighbor_weight * self.adj_weights[idx, None] * strength[:, None] * n_ij  # (p, 2)

                # Accumulate
                np.add.at(F, self.adj_pairs[idx, 0], Fn)
                np.add.at(F, self.adj_pairs[idx, 1], -Fn)

        # --- Global centroid attraction force ---
        if self.compactness > 0:
            # Use precomputed original centroid for stability
            d_vec = self.original_centroid - self.positions  # (n, 2)
            dn = np.linalg.norm(d_vec, axis=1)  # (n,)

            # Valid distances mask (needed to avoid division by zero)
            valid = dn > 1e-10
            if np.any(valid):
                # Normalize directions
                d_vec_valid = d_vec[valid]
                dn_valid = dn[valid]
                directions = d_vec_valid / dn_valid[:, None]  # (k, 2)

                # Compute force magnitude based on mode
                if self.force_mode == "direction":
                    # Default: constant magnitude with drop-off near centroid
                    w_compact = np.clip(dn_valid / self.radii[valid], 0, 1)  # (k,)
                    force_mag = self.compactness * w_compact  # (k,)

                elif self.force_mode == "linear":
                    # Linear spring: force proportional to distance
                    force_mag = self.compactness * dn_valid  # (k,)

                elif self.force_mode == "normalized":
                    # Normalized: force proportional to distance / radius
                    force_mag = self.compactness * (dn_valid / self.radii[valid])  # (k,)
                else:
                    # Fallback to direction mode
                    w_compact = np.clip(dn_valid / self.radii[valid], 0, 1)
                    force_mag = self.compactness * w_compact

                # Apply to valid indices
                F[valid] += force_mag[:, None] * directions

        # --- Origin attraction force ---
        if self.origin_weight > 0:
            # Vector from current position to original position
            displacement = self.original_positions - self.positions  # (n, 2)

            # Compute distance
            dist = np.linalg.norm(displacement, axis=1)  # (n,)

            # Valid mask for non-zero distances
            valid = dist > 1e-10
            if np.any(valid):
                # Direction toward original position
                direction = displacement[valid] / dist[valid, None]  # (k, 2)

                # Compute force magnitude based on mode
                if self.force_mode == "direction":
                    # Default: constant magnitude with drop-off near origin
                    w_origin = np.clip(dist[valid] / self.radii[valid], 0, 1)  # (k,)
                    force_mag = self.origin_weight * w_origin  # (k,)

                elif self.force_mode == "linear":
                    # Linear spring: force proportional to distance
                    force_mag = self.origin_weight * dist[valid]  # (k,)

                elif self.force_mode == "normalized":
                    # Normalized: force proportional to distance / radius
                    force_mag = self.origin_weight * (dist[valid] / self.radii[valid])  # (k,)
                else:
                    # Fallback to direction mode
                    w_origin = np.clip(dist[valid] / self.radii[valid], 0, 1)
                    force_mag = self.origin_weight * w_origin

                # Apply force
                F[valid] += force_mag[:, None] * direction

        return F

    def _resolve_contacts(self, F: NDArray[np.floating]) -> None:
        """Apply contact reaction with configurable transfer and elasticity.

        This allows circles to slide along each other without penetrating.
        Modifies F in place.

        The behavior is controlled by contact_transfer_ratio and contact_elasticity:
        - contact_transfer_ratio: Balance between cancel (0) and transfer (1)
        - contact_elasticity: Controls net compression vs bounce (-1 to 1)
        """
        # Precompute a and b from parameters
        s = (1 - self.contact_elasticity) / (1 + self.contact_elasticity + 1e-10)
        a = (1 - self.contact_transfer_ratio) ** s
        b = self.contact_transfer_ratio**s

        for _ in range(self.contact_iterations):
            # Compute pairwise differences using precomputed indices
            dx = self.positions[self.all_pairs_j] - self.positions[self.all_pairs_i]  # (m, 2)
            d = np.linalg.norm(dx, axis=1)  # (m,)

            # Valid distances mask
            valid = d > 1e-10
            if not np.any(valid):
                continue

            # Compute radii sums and contact tolerance using precomputed radii_sum
            r_sum = self.all_radii_sum + self.spacing
            tol = self.contact_tolerance * r_sum

            # Contact mask
            contact_mask = valid & (d < r_sum + tol)
            if not np.any(contact_mask):
                continue

            # Get indices of contacting pairs
            idx = np.where(contact_mask)[0]
            dx_contact = dx[idx]  # (k, 2)
            d_contact = d[idx]  # (k,)

            # Unit directions
            n_ij = dx_contact / d_contact[:, None]  # (k, 2)
            n_ji = -n_ij  # Direction from j to i

            # Compute compressive components
            F_i = F[self.all_pairs_i[idx]]
            F_j = F[self.all_pairs_j[idx]]
            comp_i = np.sum(F_i * n_ij, axis=1)
            comp_j = np.sum(F_j * n_ji, axis=1)

            # Apply reaction (remove compressive components)
            comp_i_mask = comp_i > 0
            comp_j_mask = comp_j > 0

            if np.any(comp_i_mask):
                idx_i = idx[comp_i_mask]
                compressive_i = comp_i[comp_i_mask, None] * n_ij[comp_i_mask]
                np.add.at(F, self.all_pairs_i[idx_i], -a * compressive_i)

            if np.any(comp_j_mask):
                idx_j = idx[comp_j_mask]
                compressive_j = comp_j[comp_j_mask, None] * n_ji[comp_j_mask]
                np.add.at(F, self.all_pairs_j[idx_j], -a * compressive_j)

            # Transfer compressive forces
            if np.any(comp_i_mask):
                idx_i = idx[comp_i_mask]
                transfer_i = b * comp_i[comp_i_mask, None] * n_ij[comp_i_mask]
                np.add.at(F, self.all_pairs_j[idx_i], transfer_i)

            if np.any(comp_j_mask):
                idx_j = idx[comp_j_mask]
                transfer_j = b * comp_j[comp_j_mask, None] * n_ji[comp_j_mask]
                np.add.at(F, self.all_pairs_i[idx_j], transfer_j)

    def packing_step(self) -> tuple[float, float]:
        """Perform one packing refinement iteration.

        Returns
        -------
        drift : float
            Mean relative magnitude of smoothed displacement vectors.
            Trends to zero at steady state (symmetric jitter cancels).
        jitter : float
            Mean relative std of displacement vectors. Measures
            oscillation amplitude around equilibrium.

        """
        # Compute forces and resolve contacts
        F = self._compute_forces()
        self._resolve_contacts(F)

        # Integrate with fixed step clamping
        norms = np.linalg.norm(F, axis=1)
        scale = np.minimum(1.0, self.max_step / (norms + 1e-8))
        effective_radius = self.avg_radius * (self.radii / self.avg_radius) ** self.size_sensitivity
        step = F * scale[:, None] * effective_radius[:, None]

        # Smooth step via EMA
        self._step_smooth.update(step)

        old_positions = self.positions.copy()
        self.positions += self._step_smooth.mean

        # Hard non-overlap projection
        self._separate_overlapping_pairs(max_iter=self.overlap_projection_iters)

        # Actual displacement after projection (true net movement)
        displacement = self.positions - old_positions

        # Update displacement statistics (vector EMA for convergence)
        if self._disp_stats is None:
            self._disp_stats = ExponentialMovingStats(
                self.n,
                2,
                n_eff=self.convergence_window,
                adaptive=self.adaptive_ema,
            )
        self._disp_stats.update(displacement)

        drift = float(np.mean(self._disp_stats.mean_magnitude / self.radii))
        jitter = float(np.mean(self._disp_stats.std_magnitude / self.radii))

        return drift, jitter

    def run_circle_packing(
        self,
        max_iterations: int = 500,
        tolerance: float = 0.025,
        show_progress: bool = True,
        save_history: bool = False,
    ) -> tuple[NDArray[np.floating], dict[str, Any], list[NDArray[np.floating]] | None]:
        """Run circle packing (Stage 2) until convergence or max iterations.

        Resets EMA state so this method can be called multiple times or
        after :meth:`run_overlap_resolution`. Uses :meth:`packing_step`
        internally.

        Parameters
        ----------
        max_iterations : int
            Maximum number of packing iterations. Default: 500
        tolerance : float
            Convergence threshold for mean step size. Default: 1e-4
        show_progress : bool
            Whether to display progress bar. Default: True
        save_history : bool
            Whether to save position history. Default: False

        Returns
        -------
        positions : NDArray[np.floating]
            Final symbol positions in original coordinate system.
        info : dict
            Simulation statistics including:
            - "converged": Whether convergence criteria met
            - "final_overlaps": Remaining overlap count
            - "stage2_iterations": Packing iteration count
            - "final_drift": Final smoothed drift metric
            - "final_jitter": Final jitter metric
            - "drift_history": Per-iteration drift values
            - "jitter_history": Per-iteration jitter values
        history : list[NDArray[np.floating]] | None
            Position history if save_history=True, else None.

        """
        self._reset_ema_state()
        history: list[NDArray[np.floating]] | None = [] if save_history else None

        iterator: Any = range(max_iterations)
        if show_progress:
            iterator = tqdm(iterator, desc="Refining topology", leave=True)

        converged = False
        stage2_iters = 0
        drift_history: list[float] = []
        jitter_history: list[float] = []
        drift_rate_history: list[float] = []
        drift_rate_neg_frac_history: list[float] = []
        overlap_history: list[int] = []
        prev_drift: float | None = None
        drift_rate_ema = ScalarEMA(n_eff=20, adaptive=self.adaptive_ema)
        neg_frac_ema = ScalarEMA(n_eff=20, adaptive=self.adaptive_ema, initial_value=0.5)

        for _ in iterator:
            stage2_iters += 1

            if history is not None:
                history.append(self.positions * self.scale + self.center)

            drift, jitter = self.packing_step()
            drift_history.append(drift)
            jitter_history.append(jitter)

            if prev_drift is None:
                drift_rate_history.append(float("nan"))
                drift_rate_neg_frac_history.append(float("nan"))
            else:
                drift_rate = drift - prev_drift
                drift_rate_ema.update(drift_rate)
                neg_frac_ema.update(1.0 if drift_rate <= 0 else 0.0)

                # drift_rate_history.append(drift_rate_ema.value)
                drift_rate_history.append(drift_rate)
                drift_rate_neg_frac_history.append(neg_frac_ema.value)
            prev_drift = drift

            overlaps = self._count_overlaps()
            overlap_history.append(overlaps)

            if show_progress:
                iterator.set_postfix({"drift": f"{drift:.2e}", "jitter": f"{jitter:.2e}", "overlaps": overlaps})

            # Steady state: smoothed net displacement below tolerance
            if neg_frac_ema.value < 0.5 and drift < tolerance:
                converged = True
                if show_progress:
                    iterator.update()
                    iterator.close()
                break

        final_overlaps = self._count_overlaps()

        info = {
            "converged": converged,
            "final_overlaps": final_overlaps,
            "stage2_iterations": stage2_iters,
            "final_drift": drift_history[-1] if drift_history else 0.0,
            "final_jitter": jitter_history[-1] if jitter_history else 0.0,
            "drift_history": drift_history,
            "jitter_history": jitter_history,
            "drift_rate_history": drift_rate_history,
            "drift_rate_neg_frac_history": drift_rate_neg_frac_history,
            "overlap_history": overlap_history,
        }

        final_positions = self.positions * self.scale + self.center
        return final_positions, info, history

    def run(
        self,
        max_iterations: int = 500,
        tolerance: float = 0.025,
        show_progress: bool = True,
        save_history: bool = False,
    ) -> tuple[NDArray[np.floating], dict[str, Any], list[NDArray[np.floating]] | None]:
        """Run both stages: overlap resolution then circle packing.

        Equivalent to calling :meth:`run_overlap_resolution` followed by
        :meth:`run_circle_packing`. Can be called multiple times; EMA
        state is reset at the start of each packing phase.

        Parameters
        ----------
        max_iterations : int
            Maximum number of packing iterations. Default: 500
        tolerance : float
            Convergence threshold for mean step size. Default: 1e-4
        show_progress : bool
            Whether to display progress bar. Default: True
        save_history : bool
            Whether to save position history. Default: False

        Returns
        -------
        positions : NDArray[np.floating]
            Final symbol positions in original coordinate system.
        info : dict
            Simulation statistics including:
            - "iterations": Total iterations (stage1 + stage2)
            - "converged": Whether convergence criteria met
            - "final_overlaps": Remaining overlap count
            - "stage1_iterations": Overlap resolution iteration count
            - "stage2_iterations": Packing iteration count
            - "final_drift": Final smoothed drift metric
            - "final_jitter": Final jitter metric
            - "drift_history": Per-iteration drift values
            - "jitter_history": Per-iteration jitter values
        history : list[NDArray[np.floating]] | None
            Position history if save_history=True, else None.

        """
        stage1_info = self.run_overlap_resolution()

        positions, stage2_info, history = self.run_circle_packing(
            max_iterations=max_iterations,
            tolerance=tolerance,
            show_progress=show_progress,
            save_history=save_history,
        )

        # Merge info dicts
        info = {
            "iterations": stage1_info["iterations"] + stage2_info["stage2_iterations"],
            "stage1_iterations": stage1_info["iterations"],
            **stage2_info,
        }

        return positions, info, history
