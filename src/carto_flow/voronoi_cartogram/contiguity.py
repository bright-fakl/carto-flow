"""Group-contiguity repair for Voronoi cartogram results."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import geopandas as gpd
import numpy as np

__all__ = ["make_groups_contiguous"]

# _compose_topology_permutation is intentionally private (leading underscore)
# but imported by api.py and result.py.


from ..geo_utils.contiguity import _swap_preserves_contiguity
from ..geo_utils.contiguity import repair_adjacency as _repair_adjacency
from ..geo_utils.contiguity import repair_compactness as _repair_compactness
from ..geo_utils.contiguity import repair_contiguity as _repair_contiguity


def _repair_orientation(
    cells: list,
    geom_positions: np.ndarray,
    adj_pairs: list[tuple[int, int]],
    slot_of: np.ndarray,
    *,
    max_passes: int = 3,
    groups: list | None = None,
    min_shared_length: float | None = None,
) -> np.ndarray:
    """Permute slots to align output cell directions with input geometry directions.

    For each pair in *adj_pairs* that is currently Voronoi-adjacent, computes the
    cosine similarity between the input direction vector (geom j - geom i) and the
    output direction vector (cell[slot_of[j]].centroid - cell[slot_of[i]].centroid).
    Swaps are applied only when (1) the net adjacency count does not decrease and
    (2) the total orientation score (sum of cosines for adjacent pairs) strictly
    improves.

    Parameters
    ----------
    cells : list
        Shapely geometry objects — the Voronoi cells, one per slot.
    geom_positions : np.ndarray, shape (G, 2)
        Input geometry centroids (x, y), district-indexed.
    adj_pairs : list of (i, j)
        Pairs of district indices adjacent in the input geometries.
    slot_of : np.ndarray[int], shape (G,)
        Current permutation array (district → slot), typically from
        ``_repair_adjacency()``.  Modified in-place and returned.
    max_passes : int
        Maximum repair passes.

    Returns
    -------
    slot_of : np.ndarray[int], shape (G,)
        Updated permutation array.
    """
    from carto_flow.geo_utils.adjacency import find_adjacent_pairs

    n = len(cells)
    dist_at = list(np.argsort(slot_of))  # slot → district (inverse of slot_of)
    # Recompute dist_at from slot_of (slot_of may differ from identity)
    dist_at = [-1] * n
    for d, s in enumerate(slot_of):
        dist_at[s] = d

    if not adj_pairs:
        return slot_of

    # Build Voronoi adjacency set and dict (for contiguity guard)
    raw = find_adjacent_pairs(cells, min_shared_length=min_shared_length)
    voronoi_adj: set[tuple[int, int]] = set()
    voronoi_adj_dict: dict[int, set[int]] = defaultdict(set)
    for s1, s2, _ in raw:
        voronoi_adj.add((min(s1, s2), max(s1, s2)))
        voronoi_adj_dict[s1].add(s2)
        voronoi_adj_dict[s2].add(s1)

    def _v_adj(s1: int, s2: int) -> bool:
        return (min(s1, s2), max(s1, s2)) in voronoi_adj

    # Precompute cell centroids (slot-indexed) and input direction vectors
    cell_xy = np.array([[c.centroid.x, c.centroid.y] for c in cells])
    adj_i = np.array([i for i, j in adj_pairs], dtype=np.intp)
    adj_j = np.array([j for i, j in adj_pairs], dtype=np.intp)
    d_in = geom_positions[adj_j] - geom_positions[adj_i]  # fixed

    # adj_pair index lists per district
    district_pairs: defaultdict[int, list[int]] = defaultdict(list)
    for k, (i, j) in enumerate(adj_pairs):
        district_pairs[i].append(k)
        district_pairs[j].append(k)

    def _cosine(s_i: int, s_j: int, k: int) -> float:
        d_out = cell_xy[s_j] - cell_xy[s_i]
        n_in = float(np.linalg.norm(d_in[k]))
        n_out = float(np.linalg.norm(d_out))
        if n_in < 1e-10 or n_out < 1e-10:
            return 1.0  # degenerate → treat as satisfied
        return float(np.dot(d_in[k], d_out) / (n_in * n_out))

    def _current_score() -> float:
        total = 0.0
        for k, (pi, pj) in enumerate(adj_pairs):
            si, sj = slot_of[pi], slot_of[pj]
            if _v_adj(si, sj):
                total += _cosine(si, sj, k)
        return total

    def _delta_swap(a: int, b: int):
        """Return (adjacency_net_gain, orientation_delta) for swapping a and b."""
        sa, sb = slot_of[a], slot_of[b]
        adj_gain = 0
        ori_delta = 0.0
        seen: set[int] = set()
        for d in (a, b):
            for k in district_pairs[d]:
                if k in seen:
                    continue
                seen.add(k)
                pi, pj = adj_pairs[k]
                si, sj = slot_of[pi], slot_of[pj]
                new_si = sb if pi == a else (sa if pi == b else si)
                new_sj = sb if pj == a else (sa if pj == b else sj)
                was_adj = _v_adj(si, sj)
                now_adj = _v_adj(new_si, new_sj)
                adj_gain += int(now_adj) - int(was_adj)
                if was_adj:
                    ori_delta -= _cosine(si, sj, k)
                if now_adj:
                    ori_delta += _cosine(new_si, new_sj, k)
        return adj_gain, ori_delta

    def _do_swap(a: int, b: int) -> None:
        sa, sb = slot_of[a], slot_of[b]
        slot_of[a], slot_of[b] = sb, sa
        dist_at[sa], dist_at[sb] = b, a

    # Working copy of groups for contiguity guard
    cur_groups = list(groups) if groups is not None else None

    for _ in range(max_passes):
        # Find pairs with negative cosine that are currently adjacent
        violated: set[int] = set()
        for k, (pi, pj) in enumerate(adj_pairs):
            si, sj = slot_of[pi], slot_of[pj]
            if _v_adj(si, sj) and _cosine(si, sj, k) <= 0:
                violated.add(pi)
                violated.add(pj)
        if not violated:
            break

        locked: set[int] = set()
        improved = False
        for d in list(violated):
            if d in locked:
                continue
            best_ori, best_b = 0.0, -1
            for b in range(n):
                if b == d or b in locked:
                    continue
                adj_gain, ori_delta = _delta_swap(d, b)
                if adj_gain >= 0 and ori_delta > best_ori:
                    if cur_groups is not None and not _swap_preserves_contiguity(d, b, cur_groups, voronoi_adj_dict):
                        continue
                    best_ori, best_b = ori_delta, b
            if best_b >= 0:
                _do_swap(d, best_b)
                if cur_groups is not None:
                    cur_groups[d], cur_groups[best_b] = cur_groups[best_b], cur_groups[d]
                locked.add(d)
                locked.add(best_b)
                improved = True
        if not improved:
            break

    return slot_of


def make_groups_contiguous(
    gdf: gpd.GeoDataFrame,
    group_by: str,
    *,
    max_passes: int = 20,
    show_progress: bool = False,
    debug: bool = False,
) -> tuple[gpd.GeoDataFrame, list[tuple[Any, list[int]]]]:
    """Permute rows of *gdf* so that each group forms a contiguous Voronoi region.

    After lloyd relaxation, rows belonging to the same group (e.g. congressional
    districts within a state) may occupy non-adjacent Voronoi slots.  This
    function permutes the row-to-slot mapping — without moving any cell geometry
    — so that each group's slots form a connected subgraph of the Voronoi
    adjacency graph.

    The returned GeoDataFrame has the same rows as *gdf* but reordered:
    ``result.iloc[i]`` is the row that should be placed at slot *i* (i.e. the
    geometry at ``gdf.geometry.iloc[i]`` is now owned by a potentially different
    original row).

    Algorithm
    ---------
    1. Build a slot-adjacency graph from ``gdf.geometry`` using
       :func:`carto_flow.geo_utils.adjacency.find_adjacent_pairs`.
    2. Maintain a permutation ``slot_of[d]`` (district → slot) and its inverse
       ``dist_at[s]`` (slot → district).
    3. Repeat up to *max_passes* times:

       a. For each multi-district group, find connected components of its
          current slots (BFS restricted to that group's slots).
       b. For each satellite component (all but the largest), BFS through
          other-group slots to find the shortest path to the main body.
          Slots that changed hands earlier in the same pass are forbidden,
          preventing within-pass oscillation.
       c. Walk the path: swap each intermediate district with the one moving
          forward along the path.  Before each swap, check that the swap would
          not disconnect the displaced district's group (articulation-point
          check).  If it would, skip this path.
       d. Stop when all groups are contiguous or no swaps were made in a pass.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame whose ``geometry`` column contains the Voronoi cells (one
        cell per row).  All other columns are carried along unchanged.
    group_by : str
        Column in *gdf* that identifies which group each row belongs to.
    max_passes : int
        Maximum number of repair passes (default 20).
    show_progress : bool
        If True, print a summary of discontiguous groups before the first pass
        and a status line after each pass.
    debug : bool
        If True, print one line per satellite repair attempt showing the group,
        satellite slots, and outcome (SWAPPED / REJECTED / NO PATH).

    Returns
    -------
    GeoDataFrame
        Copy of *gdf* with only the geometry column permuted.  Row *i* of the
        result keeps all original attribute values but is assigned the Voronoi
        cell geometry that was determined to belong to it.
    list[tuple[Any, list[int]]]
        One entry per remaining satellite component: ``(group_id, row_indices)``
        where *group_id* is the value from the *group_by* column and *row_indices*
        is the sorted list of original gdf row indices belonging to that satellite.
        The main body of each group is excluded.  Empty when fully converged.
    """
    cells = list(gdf.geometry)
    groups = list(gdf[group_by])
    slot_of, discontiguous = _repair_contiguity(
        cells,
        groups,
        max_passes=max_passes,
        show_progress=show_progress,
        debug=debug,
    )
    result = gdf.copy()
    result = result.set_geometry([cells[s] for s in slot_of])
    return result, discontiguous


def _compose_topology_permutation(
    cells: list,
    groups: list | None,
    adj_pairs: list[tuple[int, int]] | None,
    *,
    group_contiguity: bool | None = None,
    compactness: bool | None = None,
    adjacency: bool | None = None,
    orientation: bool | None = None,
    max_passes: int = 3,
    geom_positions: np.ndarray | None = None,
    min_shared_length: float | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Run topology repair stages and return the composed permutation.

    This is the shared core used by both the online ``api.py`` pipeline and
    the post-hoc :meth:`~carto_flow.voronoi_cartogram.result.VoronoiCartogram.repair_topology`
    method.

    Parameters
    ----------
    cells : list
        Voronoi cell geometries (one per district/slot).
    groups : list or None
        Group label for each district.  Required for stage 1; ignored when
        ``group_contiguity=False`` or ``None``.
    adj_pairs : list of (i, j) or None
        Input-adjacent district index pairs.  Required for stages 2 & 3;
        those stages are skipped when ``None``.
    group_contiguity : bool
        Enable stage 1 (group contiguity repair).
    compactness : bool or None
        Enable stage 1.5 (compactness enhancement).  ``None`` → ``True`` when
        *groups* is not ``None``, ``False`` otherwise.
    adjacency : bool or None
        Enable stage 2 (adjacency permutation).  ``None`` → ``True`` when
        *groups* is ``None``, ``False`` otherwise.
    orientation : bool or None
        Enable stage 3 (orientation alignment).  ``None`` → ``True`` when
        *groups* is ``None``, ``False`` otherwise.
    max_passes : int
        Maximum repair passes per active stage.
    geom_positions : np.ndarray of shape (n, 2) or None
        Input geometry centroids (x, y), district-indexed.  Required for
        stage 3; stage 3 is skipped when ``None``.

    Returns
    -------
    slot_of : np.ndarray[int], shape (n,)
        Composed permutation across all active stages.
        ``slot_of[d]`` is the original slot whose cell district *d* should
        now occupy.  Identity array when no stages were run or no swaps
        were made.
    stages_run : list[str]
        Names of stages that were executed (subset of
        ``["group_contiguity", "compactness", "adjacency", "orientation"]``).
    """
    # Auto-resolve stage flags
    if group_contiguity is None:
        group_contiguity = groups is not None
    if compactness is None:
        compactness = groups is not None
    if adjacency is None:
        adjacency = groups is None
    if orientation is None:
        orientation = groups is None

    n = len(cells)
    slot_of = np.arange(n, dtype=np.intp)
    stages_run: list[str] = []

    # Stage 1: group contiguity
    if group_contiguity and groups is not None:
        _s, _ = _repair_contiguity(cells, groups, max_passes=max_passes, min_shared_length=min_shared_length)
        slot_of = slot_of[_s]
        cells = [cells[s] for s in _s]
        stages_run.append("group_contiguity")

    # Stage 1.5: compactness enhancement — boundary swaps between adjacent groups
    # to reduce inertia without introducing satellites.  Runs on the already-permuted
    # (district-indexed) cells produced by stage 1.
    if compactness and groups is not None:
        _s = _repair_compactness(cells, groups, max_passes=max_passes, min_shared_length=min_shared_length)
        slot_of = slot_of[_s]
        cells = [cells[s] for s in _s]
        stages_run.append("compactness")

    # Stage 2: adjacency
    if adjacency and adj_pairs:
        _s = _repair_adjacency(
            cells, adj_pairs, max_passes=max_passes, groups=groups, min_shared_length=min_shared_length
        )
        slot_of = slot_of[_s]
        cells = [cells[s] for s in _s]
        stages_run.append("adjacency")

    # Stage 3: orientation
    if orientation and adj_pairs and geom_positions is not None:
        _s = _repair_orientation(
            cells,
            geom_positions,
            adj_pairs,
            np.arange(len(cells), dtype=np.intp),
            max_passes=max_passes,
            groups=groups,
            min_shared_length=min_shared_length,
        )
        slot_of = slot_of[_s]
        stages_run.append("orientation")

    return slot_of, stages_run
