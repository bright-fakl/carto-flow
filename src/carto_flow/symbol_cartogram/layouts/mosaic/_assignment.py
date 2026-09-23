"""Tile assignment for mosaic layout."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ....geo_utils.adjacency import find_adjacent_pairs

if TYPE_CHECKING:
    from ...tiling import TilingResult

__all__ = [
    "hungarian_morphed_assignment",
]

# Per-iteration cost reduction applied to inter-region gap bridge candidates in
# the connectivity-repair loop, as a multiple of the static cost maximum.
_GAP_BRIDGE_MULT = 5.0

# How many inter-region gap pairs a single disconnected tile counts as in the
# tie-breaking score of the connectivity-repair loop.  Only the ratio to the
# gap weight (fixed at 1) matters.
_DISCONNECTED_SCORE_WEIGHT = 100

# Per-iteration cost raise applied to intra-region disconnected tiles in the
# connectivity-repair loop, as a multiple of the static cost maximum.  Any
# multiple above 1 puts the tile out of contention outright, so the size of it
# is not a knob -- see ``HungarianOptions.penalize_disconnected``.
_DISCONNECTED_PENALTY_MULT = 10.0


# ---------------------------------------------------------------------------
# Cost matrix and connectivity helpers
# ---------------------------------------------------------------------------


def _build_cost_matrix(
    tiling_result: TilingResult,
    geometries: list,
    counts: np.ndarray,
    valid_tile_indices: list[int],
    adj_list: list[list[int]],
    working_union,
    *,
    distance_weight: float,
    outside_penalty: float,
    interior_bonus: float,
) -> tuple:
    """Build the (n_slots x n_valid) cost matrix for Hungarian assignment.

    Returns
    -------
    cost_G : np.ndarray, shape (G, n_valid)
    cost : np.ndarray, shape (n_slots, n_valid)
    slot_to_geom : list of int
    connectivity : np.ndarray, shape (n_valid,)
    """
    G = len(geometries)
    valid_set = set(valid_tile_indices)

    tile_polys = [tiling_result.polygons[t] for t in valid_tile_indices]
    tile_area = tile_polys[0].area

    geom_centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in geometries], dtype=np.float64)
    tile_centroids = np.array([[p.centroid.x, p.centroid.y] for p in tile_polys], dtype=np.float64)

    # Normalized pairwise distances G x n_valid
    dx = tile_centroids[:, 0][None, :] - geom_centroids[:, 0][:, None]
    dy = tile_centroids[:, 1][None, :] - geom_centroids[:, 1][:, None]
    dist_matrix = dx**2 + dy**2
    max_dist = dist_matrix.max() or 1.0
    dist_norm = dist_matrix / max_dist

    # Fraction of each tile outside the study union
    import shapely as _shapely

    tile_inside_area = _shapely.area(_shapely.intersection(np.asarray(tile_polys), working_union))
    tile_inside_area = np.where(tile_inside_area > 1e-12, tile_inside_area, tile_area)
    outside_frac = 1.0 - tile_inside_area / tile_area  # (n_valid,)

    # Connectivity: fraction of neighbors that are valid tiles
    connectivity = np.array(
        [sum(1 for nb in adj_list[t] if nb in valid_set) / max(len(adj_list[t]), 1) for t in valid_tile_indices],
        dtype=np.float64,
    )

    cost_G = (
        distance_weight * dist_norm + outside_penalty * outside_frac[None, :] - interior_bonus * connectivity[None, :]
    )

    slot_to_geom: list[int] = []
    for g in range(G):
        slot_to_geom.extend([g] * int(counts[g]))

    cost = cost_G[slot_to_geom, :]  # n_slots x n_valid
    return cost_G, cost, slot_to_geom, connectivity


def _find_group_disconnected_tiles(
    assignment: np.ndarray,
    valid_tile_indices: list[int],
    adj_list: list[list[int]],
    group_labels: np.ndarray,
) -> list[tuple[int, int]]:
    """Return (tile, geom) pairs disconnected from their group's main component.

    Unlike :func:`_find_disconnected_tiles` which checks per-geometry connectivity,
    this checks per-group connectivity: all tiles assigned to geometries in the
    same group must form a connected subgraph.
    """
    from collections import deque

    n_groups = int(group_labels.max()) + 1
    group_to_tiles: dict[int, list[int]] = {gr: [] for gr in range(n_groups)}
    for t in valid_tile_indices:
        g = int(assignment[t])
        if g >= 0:
            group_to_tiles[int(group_labels[g])].append(t)

    disconnected: list[tuple[int, int]] = []
    for gr_tiles in group_to_tiles.values():
        if len(gr_tiles) <= 1:
            continue
        tile_set = set(gr_tiles)
        visited: set[int] = {gr_tiles[0]}
        queue: deque[int] = deque([gr_tiles[0]])
        while queue:
            t = queue.popleft()
            for nb in adj_list[t]:
                if nb in tile_set and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        disconnected.extend((t, int(assignment[t])) for t in gr_tiles if t not in visited)
    return disconnected


def _find_disconnected_tiles(
    assignment: np.ndarray,
    valid_tile_indices: list[int],
    adj_list: list[list[int]],
    G: int,
) -> list[tuple[int, int]]:
    """Return (tile, geom) pairs disconnected from their geometry's main component."""
    from collections import deque

    disconnected: list[tuple[int, int]] = []
    for g in range(G):
        g_tiles = [t for t in valid_tile_indices if assignment[t] == g]
        if len(g_tiles) <= 1:
            continue
        tile_set = set(g_tiles)
        visited: set[int] = {g_tiles[0]}
        queue: deque[int] = deque([g_tiles[0]])
        while queue:
            t = queue.popleft()
            for nb in adj_list[t]:
                if nb in tile_set and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        disconnected.extend((t, g) for t in g_tiles if t not in visited)
    return disconnected


def hungarian_morphed_assignment(
    tiling_result: TilingResult,
    geometries: list,
    counts: np.ndarray,
    valid_tile_indices: list[int],
    adj_list: list[list[int]],
    *,
    core_set: set[int],
    working_union=None,
    options=None,
    group_labels: np.ndarray | None = None,
    show_progress: bool = False,
    stats: dict | None = None,
) -> np.ndarray:
    """Hungarian assignment in the morphed coordinate space.

    Assigns exactly ``counts[g]`` tiles to each geometry by solving a
    slot-expanded linear assignment problem.  An iterative repair loop
    raises the cost of intra-state disconnected tiles and reduces cost
    for tiles that could bridge inter-state gaps, then re-solves — until
    all regions are contiguous or ``options.max_connectivity_iters`` is
    reached.  Passes are ranked by how many regions (or groups, in group
    mode) end up split over more than one block, with the disconnected-tile
    and gap counts only breaking ties, and the best pass is kept; a pass
    that splits more regions than the incumbent never replaces it and stops
    the loop.

    Parameters
    ----------
    tiling_result : TilingResult
        Tile grid generated over the morphed bounds.
    geometries : list of shapely.Geometry
        Morphed geometries (``working_geometries`` from ``api.py``).
    counts : np.ndarray, shape (G,)
        Target tile counts.  ``sum(counts) <= len(valid_tile_indices)``.
    valid_tile_indices : list of int
        Core tiles (centroid inside union) + overlap extras.
    adj_list : list of list of int
        Global tile adjacency lists.
    core_set : set of int
        Tile indices whose centroid is inside the study union.
    working_union : shapely.Geometry or None
        Union of all morphed geometries.  Computed if None.
    options : HungarianOptions or None
        Cost-function and iteration parameters.  None → defaults.
    group_labels : np.ndarray of int32, shape (G,), or None
        When provided, connectivity repair operates at the group level:
        all tiles assigned to geometries in the same group must form a
        connected subgraph.  Use when ``counts`` are all 1 and geometries
        belong to a grouping variable (e.g. congressional districts grouped
        by state).  Disconnected sub-components (islands) within a group
        are handled automatically — tiles are already split into separate
        effective groups before calling this function.  None → per-geometry
        connectivity (default, existing behaviour).
    stats : dict or None
        When given, filled in place with diagnostics of the repair loop:
        ``passes`` (number of linear-assignment solves actually run) and
        ``split_units`` (region-level score of the kept assignment: number of
        regions — or groups, in group mode — whose tiles are not one connected
        component).

    Returns
    -------
    assignment : np.ndarray of int32, shape (T_global,)
    """
    from scipy.optimize import linear_sum_assignment

    from . import HungarianOptions

    if options is None:
        options = HungarianOptions()

    if working_union is None:
        from shapely.ops import unary_union

        working_union = unary_union(geometries)

    G = len(geometries)
    T_global = len(tiling_result.polygons)
    tile_to_local = {t: i for i, t in enumerate(valid_tile_indices)}

    # Precompute BFS hop matrix over valid tiles (VxV int16, ~2 bytes x V²).
    bfs_dists: np.ndarray | None = None
    if options.neighbor_bfs and options.neighbor_weight > 0:
        from collections import deque as _deque

        V = len(valid_tile_indices)
        bfs_dists = np.full((V, V), V, dtype=np.int16)
        np.fill_diagonal(bfs_dists, 0)
        local_adj = [
            [tile_to_local[nb] for nb in adj_list[valid_tile_indices[i]] if nb in tile_to_local] for i in range(V)
        ]
        for src in range(V):
            queue = _deque([src])
            while queue:
                node = queue.popleft()
                d = int(bfs_dists[src, node])
                for nb in local_adj[node]:
                    if bfs_dists[src, nb] == V:
                        bfs_dists[src, nb] = d + 1
                        queue.append(nb)

    # Geometry adjacency for inter-state gap detection
    geom_adj_pairs = find_adjacent_pairs(geometries)
    geom_neighbors: dict[int, set[int]] = {g: set() for g in range(G)}
    for g1, g2, _ in geom_adj_pairs:
        geom_neighbors[g1].add(g2)
        geom_neighbors[g2].add(g1)

    _, cost_iter, slot_to_geom, _ = _build_cost_matrix(
        tiling_result,
        geometries,
        counts,
        valid_tile_indices,
        adj_list,
        working_union,
        distance_weight=options.distance_weight,
        outside_penalty=options.outside_penalty,
        interior_bonus=options.interior_bonus,
    )

    geom_to_slots: dict[int, list[int]] = {g: [] for g in range(G)}
    for s, g in enumerate(slot_to_geom):
        geom_to_slots[g].append(s)

    # Capture initial cost scale before any repair penalties are added.
    # Used to keep cost_neighbor comparable to the static terms regardless of
    # how large repair penalties grow in later iterations.
    cost_static_max = max(float(cost_iter.max()), 1.0)

    # Pre-compute tile centroids for neighbor cost
    tile_centroids = np.array(
        [tiling_result.polygons[t].centroid.coords[0] for t in range(T_global)],
        dtype=np.float64,
    )
    tile_local_centroids = tile_centroids[valid_tile_indices]  # (n_valid, 2)

    # Warm-start cost_neighbor from geometry centroids so iteration 0 already
    # has a neighbor signal (important when the loop converges in one pass).
    cost_neighbor = np.zeros_like(cost_iter)
    if options.neighbor_weight > 0:
        geom_centroids_arr = np.array([[geom.centroid.x, geom.centroid.y] for geom in geometries], dtype=np.float64)
        if bfs_dists is not None:
            # BFS warm-start: use nearest valid tile to each geometry centroid as proxy.
            diffs = tile_local_centroids[:, None, :] - geom_centroids_arr[None, :, :]
            nearest_local = np.argmin((diffs**2).sum(axis=2), axis=0)  # (G,)
            for g, neighbors in geom_neighbors.items():
                if not neighbors or not geom_to_slots[g]:
                    continue
                for s in geom_to_slots[g]:
                    for g2 in neighbors:
                        j = int(nearest_local[g2])
                        raw = np.maximum(bfs_dists[:, j].astype(np.float32) - 1, 0)
                        cost_neighbor[s] += raw * raw
        else:
            for g, neighbors in geom_neighbors.items():
                if not neighbors or not geom_to_slots[g]:
                    continue
                target_pts = geom_centroids_arr[list(neighbors)]
                diffs = tile_local_centroids[:, None, :] - target_pts
                dists = np.sqrt((diffs**2).sum(axis=2)).sum(axis=1)
                for s in geom_to_slots[g]:
                    cost_neighbor[s] += dists
        max_nc = cost_neighbor.max()
        if max_nc > 0:
            cost_neighbor *= options.neighbor_weight / max_nc

    # Pre-compute group-level neighbor sets when group_labels are provided
    if group_labels is not None:
        n_groups = int(group_labels.max()) + 1
        group_neighbors: dict[int, set[int]] = {gr: set() for gr in range(n_groups)}
        for g1, neighbors in geom_neighbors.items():
            for g2 in neighbors:
                gr1, gr2 = int(group_labels[g1]), int(group_labels[g2])
                if gr1 != gr2:
                    group_neighbors[gr1].add(gr2)
                    group_neighbors[gr2].add(gr1)

    assignment = np.full(T_global, -1, dtype=np.int32)
    best_assignment = assignment.copy()
    # Lexicographic score: (split units, tile/gap score).  The first term is
    # what users see — a region (or, in group mode, a group) whose tiles are
    # not one connected block — and it decides on its own; the tile/gap score
    # only breaks ties between assignments that split the same number of units.
    best_score: tuple[int, int] = (2**31, 2**31)
    passes_run = 0

    for iteration in range(options.max_connectivity_iters + 1):
        passes_run = iteration + 1
        row_ind, col_ind = linear_sum_assignment(cost_iter + cost_neighbor)
        assignment[:] = -1
        for s, i in zip(row_ind, col_ind, strict=False):
            assignment[valid_tile_indices[i]] = slot_to_geom[s]

        # Refine cost_neighbor from current tile pool (replaces warm-start
        # after iteration 0; always computed before break so next iter benefits even
        # when stopping early due to connectivity convergence).
        if options.neighbor_weight > 0:
            cost_neighbor[:] = 0.0
            if bfs_dists is not None:
                g_local_tiles: dict[int, list[int]] = {}
                for g in range(G):
                    locs = [tile_to_local[t] for t in valid_tile_indices if assignment[t] == g]
                    if locs:
                        g_local_tiles[g] = locs
                for g, neighbors in geom_neighbors.items():
                    valid_nbrs = [g2 for g2 in neighbors if g2 in g_local_tiles]
                    if not valid_nbrs or not geom_to_slots[g]:
                        continue
                    for s in geom_to_slots[g]:
                        for g2 in valid_nbrs:
                            min_hops = bfs_dists[:, g_local_tiles[g2]].min(axis=1).astype(np.float32)
                            raw = np.maximum(min_hops - 1, 0)
                            cost_neighbor[s] += raw * raw
            else:
                pool_centroids = np.zeros((G, 2))
                pool_assigned = np.zeros(G, dtype=bool)
                for g in range(G):
                    g_tiles = [t for t in valid_tile_indices if assignment[t] == g]
                    if g_tiles:
                        pool_centroids[g] = tile_centroids[g_tiles].mean(axis=0)
                        pool_assigned[g] = True
                for g, neighbors in geom_neighbors.items():
                    valid_nbrs = [g2 for g2 in neighbors if pool_assigned[g2]]
                    if not valid_nbrs or not geom_to_slots[g]:
                        continue
                    target_pts = pool_centroids[valid_nbrs]
                    diffs = tile_local_centroids[:, None, :] - target_pts
                    dists = np.sqrt((diffs**2).sum(axis=2)).sum(axis=1)
                    for s in geom_to_slots[g]:
                        cost_neighbor[s] += dists
            max_nc = cost_neighbor.max()
            if max_nc > 0:
                cost_neighbor *= options.neighbor_weight / max_nc

        if group_labels is not None:
            # Intra-group: find tiles disconnected from their group's main component
            disconnected = _find_group_disconnected_tiles(assignment, valid_tile_indices, adj_list, group_labels)

            # Inter-group: find adjacent groups with no shared tile edge
            group_tile_sets: dict[int, set[int]] = {gr: set() for gr in range(n_groups)}
            for t in valid_tile_indices:
                g = int(assignment[t])
                if g >= 0:
                    group_tile_sets[int(group_labels[g])].add(t)

            inter_state_gaps: list[tuple[int, int]] = []
            for gr1, gr_neighbors in group_neighbors.items():
                for gr2 in gr_neighbors:
                    if gr2 <= gr1:
                        continue
                    tiles_gr1 = group_tile_sets[gr1]
                    tiles_gr2 = group_tile_sets[gr2]
                    if not tiles_gr1 or not tiles_gr2:
                        continue
                    if not any(nb in tiles_gr2 for t in tiles_gr1 for nb in adj_list[t]):
                        inter_state_gaps.append((gr1, gr2))
        else:
            # Intra-geometry: find disconnected tiles
            disconnected = _find_disconnected_tiles(assignment, valid_tile_indices, adj_list, G)

            # Inter-geometry: find adjacent geometry pairs with no shared tile edge
            geom_tile_sets: dict[int, set[int]] = {g: set() for g in range(G)}
            for t in valid_tile_indices:
                g = int(assignment[t])
                if g >= 0:
                    geom_tile_sets[g].add(t)

            inter_state_gaps = []
            for g1, neighbors in geom_neighbors.items():
                for g2 in neighbors:
                    if g2 <= g1:
                        continue
                    tiles_g1 = geom_tile_sets[g1]
                    tiles_g2 = geom_tile_sets[g2]
                    if not tiles_g1 or not tiles_g2:
                        continue
                    if not any(nb in tiles_g2 for t in tiles_g1 for nb in adj_list[t]):
                        inter_state_gaps.append((g1, g2))

        # Region-level score: how many units (geometries, or groups in group
        # mode) end up in more than one block.  A unit is split exactly when it
        # owns at least one disconnected tile, so this is free to derive.
        if group_labels is not None:
            split_units = len({int(group_labels[g]) for _, g in disconnected})
        else:
            split_units = len({g for _, g in disconnected})
        tile_gap_score = len(disconnected) * _DISCONNECTED_SCORE_WEIGHT + len(inter_state_gaps)
        score = (split_units, tile_gap_score)

        if show_progress:
            print(
                f"[mosaic]   Hungarian iter {iteration}: "
                f"split={split_units}  score={tile_gap_score}  "
                f"disconnected={len(disconnected)}  gaps={len(inter_state_gaps)}"
            )

        improved = score < best_score
        if improved:
            best_score = score
            best_assignment = assignment.copy()

        if score == (0, 0):
            if show_progress:
                print(f"[mosaic]   Hungarian converged at iteration {iteration}")
            break

        if iteration == options.max_connectivity_iters or not improved:
            if show_progress:
                print(f"[mosaic]   Hungarian stopped — best split={best_score[0]} score={best_score[1]}")
            break

        disc_penalty = cost_static_max * _DISCONNECTED_PENALTY_MULT if options.penalize_disconnected else 0.0
        bridge_bonus = cost_static_max * _GAP_BRIDGE_MULT

        # Raise cost for disconnected tiles
        for t, g in disconnected:
            i = tile_to_local[t]
            for s in geom_to_slots[g]:
                cost_iter[s, i] += disc_penalty

        # Reduce cost for bridge-candidate tiles at gaps
        if group_labels is not None:
            for gr1, gr2 in inter_state_gaps:
                for gr_src, gr_dst in ((gr1, gr2), (gr2, gr1)):
                    for t in group_tile_sets[gr_src]:
                        if any(nb in group_tile_sets[gr_dst] for nb in adj_list[t]):
                            i = tile_to_local.get(t)
                            if i is not None:
                                g = int(assignment[t])
                                for s in geom_to_slots[g]:
                                    cost_iter[s, i] -= bridge_bonus
        else:
            for g1, g2 in inter_state_gaps:
                for g_src, g_dst in ((g1, g2), (g2, g1)):
                    for t in geom_tile_sets[g_src]:
                        if any(nb in geom_tile_sets[g_dst] for nb in adj_list[t]):
                            i = tile_to_local.get(t)
                            if i is not None:
                                for s in geom_to_slots[g_src]:
                                    cost_iter[s, i] -= bridge_bonus

    if stats is not None:
        stats["passes"] = passes_run
        stats["split_units"] = 0 if best_score[0] >= 2**31 else best_score[0]

    return best_assignment
