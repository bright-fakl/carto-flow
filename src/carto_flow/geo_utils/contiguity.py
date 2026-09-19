"""Group-contiguity and adjacency repair algorithms for tile/cell assignments."""

from __future__ import annotations

import heapq
from collections import defaultdict, deque
from typing import Any

import numpy as np

__all__ = ["repair_adjacency", "repair_compactness", "repair_contiguity", "repair_group_assignment"]


def repair_contiguity(
    cells: list | None,
    groups: list,
    *,
    max_passes: int = 20,
    show_progress: bool = False,
    debug: bool = False,
    min_shared_length: float | None = None,
    adjacency: list[set[int]] | None = None,
    max_candidate_paths: int = 20,
) -> tuple[np.ndarray, list[tuple[Any, list[int]]]]:
    """Make each group's cells form a connected subgraph via local swaps.

    Parameters
    ----------
    cells : list or None
        Shapely geometry objects — one per slot.  May be None when
        *adjacency* is supplied.
    groups : list
        Group label for each slot (same length as *cells*).
    max_passes : int
        Maximum number of repair passes (default 20).
    show_progress : bool
        Print per-pass progress summary.
    debug : bool
        Print one line per satellite repair attempt.
    min_shared_length : float or None
        Minimum shared border length for adjacency.
    adjacency : list of set of int, or None
        Precomputed slot adjacency (``adjacency[i]`` = slots sharing an edge
        with slot *i*).  When given, *cells* is not used and no geometric
        adjacency is computed.  Callers that already hold an exact tile
        adjacency graph should pass it so the repair sees the same topology
        the caller's metrics are computed on.
    max_candidate_paths : int
        How many shortest satellite-to-main-body chains to try before giving
        up on a satellite.  Each candidate may be rejected because rerouting
        it would split another group, so a low value can abandon a satellite
        that a slightly longer search would have repaired.  Default 20.

    Returns
    -------
    slot_of : np.ndarray of int, shape (n,)
        Permutation array: ``slot_of[d]`` is the slot that district *d*
        should occupy.
    discontiguous : list[tuple[Any, list[int]]]
        Remaining satellite components that could not be repaired.
    """
    n = len(groups)

    if adjacency is not None:
        adj: list[set[int]] = [set(a) for a in adjacency]
    else:
        from .adjacency import find_adjacent_pairs

        if cells is None:
            raise ValueError("repair_contiguity needs either cells or adjacency")
        raw_pairs = find_adjacent_pairs(cells, min_shared_length=min_shared_length)
        adj = [set() for _ in range(n)]
        for i, j, _ in raw_pairs:
            adj[i].add(j)
            adj[j].add(i)

    slot_of = np.arange(n, dtype=np.intp)
    dist_at = list(range(n))

    group_districts: dict = defaultdict(list)
    for d, g in enumerate(groups):
        group_districts[g].append(d)

    def _connected_components(slots: list[int]) -> list[list[int]]:
        slot_set = set(slots)
        visited: set[int] = set()
        components: list[list[int]] = []
        for s in slots:
            if s in visited:
                continue
            comp: list[int] = []
            q: deque[int] = deque([s])
            visited.add(s)
            while q:
                cur = q.popleft()
                comp.append(cur)
                for nb in adj[cur]:
                    if nb in slot_set and nb not in visited:
                        visited.add(nb)
                        q.append(nb)
            components.append(comp)
        return components

    def _enumerate_paths(
        src_slots: set[int],
        dst_slots: set[int],
        forbidden_nodes: set[int],
        max_len: int = 10,
        max_k: int = max_candidate_paths,
    ):
        heap: list[tuple[int, tuple[int, ...]]] = []
        for s in src_slots:
            if s not in forbidden_nodes:
                heapq.heappush(heap, (1, (s,)))
        k = 0
        while heap and k < max_k:
            length, path_tup = heapq.heappop(heap)
            cur = path_tup[-1]
            if cur in dst_slots:
                yield list(path_tup)
                k += 1
                continue
            if length >= max_len:
                continue
            path_set = set(path_tup)
            for nb in adj[cur]:
                if nb not in path_set and nb not in forbidden_nodes:
                    heapq.heappush(heap, (length + 1, (*path_tup, nb)))

    def _do_swap(d1: int, d2: int) -> None:
        s1, s2 = slot_of[d1], slot_of[d2]
        slot_of[d1], slot_of[d2] = s2, s1
        dist_at[s1], dist_at[s2] = d2, d1

    if show_progress:
        n_discontig = 0
        n_satellites = 0
        for dists in group_districts.values():
            if len(dists) < 2:
                continue
            comps = _connected_components([slot_of[d] for d in dists])
            if len(comps) > 1:
                n_discontig += 1
                n_satellites += len(comps) - 1
        print(f"[contiguity]  {n_discontig} discontiguous groups, {n_satellites} satellites total")

    passes_done = 0
    for _pass in range(max_passes):
        any_swap = False
        pass_swaps = 0
        pass_locked: set[int] = set()

        for grp, dists in group_districts.items():
            if len(dists) < 2:
                continue

            current_slots = [slot_of[d] for d in dists]
            comps = _connected_components(current_slots)
            if len(comps) == 1:
                continue

            comps.sort(key=lambda c: -len(c))
            main_slots = set(comps[0])

            def _sat_dist(sat_comp: list[int], _main_slots: set[int] = main_slots) -> int:
                visited: set[int] = set(sat_comp)
                frontier: list[int] = list(sat_comp)
                dist = 0
                while frontier:
                    dist += 1
                    nxt: list[int] = []
                    for s in frontier:
                        for nb in adj[s]:
                            if nb in _main_slots:
                                return dist
                            if nb not in visited:
                                visited.add(nb)
                                nxt.append(nb)
                    frontier = nxt
                return dist

            sat_comps_sorted = sorted(comps[1:], key=_sat_dist)

            for sat_comp in sat_comps_sorted:
                sat_slots = set(sat_comp)

                if sat_slots & pass_locked:
                    if debug:
                        print(
                            f"[contiguity debug] pass {_pass + 1}  group={grp}"
                            f"  sat={set(sat_comp)}  SKIPPED (created this pass)"
                        )
                    continue

                other_sat_slots: set[int] = set()
                for c in comps[2:]:
                    other_sat_slots.update(c)

                open_main = main_slots - pass_locked

                path = None
                for candidate in _enumerate_paths(sat_slots, open_main, other_sat_slots):
                    n_cand = len(candidate)
                    affected_final: dict[int, int] = {}
                    for i in range(n_cand - 1):
                        d = dist_at[candidate[i]]
                        affected_final[d] = candidate[i - 1] if i > 0 else candidate[n_cand - 2]

                    bad_slot = -1
                    bad_grp = None
                    checked: set = set()
                    for i in range(n_cand - 1):
                        grp_b = groups[dist_at[candidate[i]]]
                        if grp_b in checked:
                            continue
                        checked.add(grp_b)
                        grp_b_dists = group_districts[grp_b]
                        if len(grp_b_dists) < 2:
                            continue
                        current_b_slots = [slot_of[dd] for dd in grp_b_dists]
                        if len(_connected_components(current_b_slots)) > 1:
                            continue
                        final_b_slots = [affected_final.get(dd, slot_of[dd]) for dd in grp_b_dists]
                        if len(_connected_components(final_b_slots)) > 1:
                            for k in range(1, n_cand - 1):
                                if groups[dist_at[candidate[k]]] == grp_b:
                                    bad_slot = candidate[k]
                                    break
                            else:
                                bad_slot = candidate[n_cand - 2]
                            bad_grp = grp_b
                            break

                    if bad_slot != -1:
                        if debug:
                            print(
                                f"[contiguity debug] pass {_pass + 1}  group={grp}"
                                f"  sat={set(sat_comp)}  SKIP"
                                f"  sim-blocked at slot {bad_slot} (group={bad_grp})"
                            )
                        continue

                    path = candidate
                    break
                else:
                    if debug:
                        lock_note = f"  locked={len(pass_locked)}" if pass_locked else ""
                        print(
                            f"[contiguity debug] pass {_pass + 1}  group={grp}"
                            f"  sat={set(sat_comp)}  NO PATH"
                            f"  (candidates exhausted{lock_note})"
                        )

                if path is None:
                    continue

                for k in range(1, len(path) - 1):
                    d_moving = dist_at[path[k - 1]]
                    d_displaced = dist_at[path[k]]
                    _do_swap(d_moving, d_displaced)
                    any_swap = True
                    pass_swaps += 1

                pass_locked.update(path[:-1])

                if debug:
                    print(
                        f"[contiguity debug] pass {_pass + 1}  group={grp}"
                        f"  sat={set(sat_comp)}  SWAPPED {len(path) - 2}"
                        f"  path={path}"
                    )

                current_slots = [slot_of[d] for d in dists]
                comps = _connected_components(current_slots)
                comps.sort(key=lambda c: -len(c))
                main_slots = set(comps[0])

        passes_done = _pass + 1
        if show_progress and pass_swaps > 0:
            remaining = sum(
                1
                for dists in group_districts.values()
                if len(dists) >= 2 and len(_connected_components([slot_of[d] for d in dists])) > 1
            )
            print(f"[contiguity]  pass {_pass + 1:2d}  swaps={pass_swaps:<4d}  discontiguous={remaining}")

        if not any_swap:
            break

    if show_progress:
        remaining = sum(
            1
            for dists in group_districts.values()
            if len(dists) >= 2 and len(_connected_components([slot_of[d] for d in dists])) > 1
        )
        if remaining == 0:
            print(f"[contiguity]  converged after {passes_done} pass{'es' if passes_done != 1 else ''}")
        else:
            print(f"[contiguity]  stopped at max_passes={max_passes}  discontiguous={remaining}")

    discontiguous: list[tuple[Any, list[int]]] = []
    for grp, dists in group_districts.items():
        if len(dists) < 2:
            continue
        comps = _connected_components([slot_of[d] for d in dists])
        if len(comps) <= 1:
            continue
        comps.sort(key=lambda c: -len(c))
        for sat_comp in comps[1:]:
            discontiguous.append((grp, sorted(sat_comp)))

    return slot_of, discontiguous


def _swap_preserves_contiguity(
    d1: int,
    d2: int,
    groups: list,
    adj_dict: dict,
) -> bool:
    g1, g2 = groups[d1], groups[d2]
    if g1 == g2:
        return True

    for target_d, source_d, grp in ((d1, d2, g1), (d2, d1, g2)):
        members = {i for i, g in enumerate(groups) if g == grp}
        new_members = (members - {target_d}) | {source_d}
        if len(new_members) <= 1:
            continue
        start = next(iter(new_members))
        visited = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nb in adj_dict.get(node, ()):
                if nb in new_members and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        if visited != new_members:
            return False
    return True


def repair_adjacency(
    cells: list,
    adj_pairs: list[tuple[int, int]],
    *,
    max_passes: int = 3,
    groups: list | None = None,
    min_shared_length: float | None = None,
) -> np.ndarray:
    """Permute slots so that input-adjacent geometry pairs share an edge.

    Parameters
    ----------
    cells : list
        Shapely geometry objects — one per slot.
    adj_pairs : list of (i, j)
        Pairs of district indices that are adjacent in the input geometries.
    max_passes : int
        Maximum repair passes.
    groups : list or None
        Optional group labels per district for contiguity guarding.
    min_shared_length : float or None
        Minimum shared border length for adjacency.

    Returns
    -------
    slot_of : np.ndarray[int], shape (n,)
        Permutation array. Identity if no improvement was found.
    """
    from .adjacency import find_adjacent_pairs

    n = len(cells)
    slot_of = np.arange(n, dtype=np.intp)
    dist_at = list(range(n))

    if not adj_pairs:
        return slot_of

    raw = find_adjacent_pairs(cells, min_shared_length=min_shared_length)
    voronoi_adj: set[tuple[int, int]] = set()
    voronoi_adj_dict: dict[int, set[int]] = defaultdict(set)
    for s1, s2, _ in raw:
        voronoi_adj.add((s1, s2))
        voronoi_adj.add((s2, s1))
        voronoi_adj_dict[s1].add(s2)
        voronoi_adj_dict[s2].add(s1)

    def _v_adj(s1: int, s2: int) -> bool:
        return (s1, s2) in voronoi_adj

    district_pairs: defaultdict[int, list[int]] = defaultdict(list)
    for k, (i, j) in enumerate(adj_pairs):
        district_pairs[i].append(k)
        district_pairs[j].append(k)

    def _net_gain_swap(a: int, b: int) -> int:
        sa, sb = slot_of[a], slot_of[b]
        gain = 0
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
                before = _v_adj(si, sj)
                after = _v_adj(new_si, new_sj)
                gain += int(after) - int(before)
        return gain

    def _do_swap(a: int, b: int) -> None:
        sa, sb = slot_of[a], slot_of[b]
        slot_of[a], slot_of[b] = sb, sa
        dist_at[sa], dist_at[sb] = b, a

    cur_groups = list(groups) if groups is not None else None

    for _ in range(max_passes):
        violated_districts: set[int] = set()
        for i, j in adj_pairs:
            if not _v_adj(slot_of[i], slot_of[j]):
                violated_districts.add(i)
                violated_districts.add(j)
        if not violated_districts:
            break

        # For each violated district d, compute which slots would satisfy at least
        # one of its required adjacency pairs — i.e. slots neighbouring the current
        # slot of each required partner.  The districts currently at those slots are
        # the only candidates that can produce a positive net gain for d, so we
        # restrict the inner loop to this small set (typically ≤30) instead of all n.
        desirable_candidates: dict[int, set[int]] = {}
        for d in violated_districts:
            cands: set[int] = set()
            for k in district_pairs[d]:
                pi, pj = adj_pairs[k]
                partner = pj if pi == d else pi
                for nb_slot in voronoi_adj_dict[slot_of[partner]]:
                    if nb_slot != slot_of[d]:
                        cands.add(dist_at[nb_slot])
            desirable_candidates[d] = cands - {d}

        locked: set[int] = set()
        improved = False
        for d in list(violated_districts):
            if d in locked:
                continue
            best_gain, best_b = 0, -1
            # First try the small set of geometrically useful candidates.
            # Fall back to all violated districts only if nothing is found.
            for b in desirable_candidates.get(d, ()):
                if b in locked:
                    continue
                gain = _net_gain_swap(d, b)
                if gain > best_gain:
                    if cur_groups is not None and not _swap_preserves_contiguity(d, b, cur_groups, voronoi_adj_dict):
                        continue
                    best_gain, best_b = gain, b
            if best_gain <= 0:
                for b in violated_districts:
                    if b == d or b in locked or b in desirable_candidates.get(d, ()):
                        continue
                    gain = _net_gain_swap(d, b)
                    if gain > best_gain:
                        if cur_groups is not None and not _swap_preserves_contiguity(
                            d, b, cur_groups, voronoi_adj_dict
                        ):
                            continue
                        best_gain, best_b = gain, b
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


def repair_compactness(
    cells: list,
    groups: list,
    *,
    max_passes: int = 10,
    min_shared_length: float | None = None,
) -> np.ndarray:
    """Permute slots to improve spatial compactness of each group.

    For each pair of adjacent slots belonging to different groups, a swap is
    accepted when the following dimensionless criterion is negative:

        delta_inertia / I_0  -  delta_shared / S_0  <  0

    where *I_0* is the total inertia of the two groups (sum of squared distances
    to group centroid) and *S_0* is the total intra-group shared edge length.
    Contiguity of both groups is verified before each swap is applied.
    A BFS-chain pass then tries to pull in outlying cells via short displacement
    chains.

    Parameters
    ----------
    cells : list
        Shapely geometry objects — district-indexed (``cells[d]`` is the cell
        currently assigned to district *d*).
    groups : list
        Group label for each district (same length as *cells*).
    max_passes : int
        Maximum improvement passes (default 10).
    min_shared_length : float or None
        Minimum shared border length for adjacency.

    Returns
    -------
    slot_of : np.ndarray[int], shape (n,)
        Permutation array. Identity when no improvement was found.
    """
    from .adjacency import find_adjacent_pairs

    n = len(cells)
    slot_of = np.arange(n, dtype=np.intp)
    dist_at = list(range(n))

    raw_pairs = find_adjacent_pairs(cells, min_shared_length=min_shared_length)
    adj: list[set[int]] = [set() for _ in range(n)]
    edge_len: dict[tuple[int, int], float] = {}
    for s1, s2, length in raw_pairs:
        adj[s1].add(s2)
        adj[s2].add(s1)
        edge_len[(min(s1, s2), max(s1, s2))] = length

    group_districts: dict = defaultdict(list)
    for d, g in enumerate(groups):
        group_districts[g].append(d)

    cell_xy = np.array([[c.centroid.x, c.centroid.y] for c in cells])
    cell_area = np.array([c.area for c in cells], dtype=np.float64)

    def _connected(slots: set[int]) -> bool:
        if len(slots) <= 1:
            return True
        start = next(iter(slots))
        visited = {start}
        q: deque[int] = deque([start])
        while q:
            cur = q.popleft()
            for nb in adj[cur]:
                if nb in slots and nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        return len(visited) == len(slots)

    def _connected_components(slots: list[int]) -> list[list[int]]:
        slot_set = set(slots)
        visited: set[int] = set()
        components: list[list[int]] = []
        for s in slots:
            if s in visited:
                continue
            comp: list[int] = []
            q: deque[int] = deque([s])
            visited.add(s)
            while q:
                cur = q.popleft()
                comp.append(cur)
                for nb in adj[cur]:
                    if nb in slot_set and nb not in visited:
                        visited.add(nb)
                        q.append(nb)
            components.append(comp)
        return components

    def _enum_paths(
        src_slots: set[int],
        forbidden_nodes: set[int],
        min_len: int = 3,
        max_len: int = 6,
        max_k: int = 50,
    ):
        heap: list[tuple[int, tuple[int, ...]]] = []
        for s in src_slots:
            if s not in forbidden_nodes:
                heapq.heappush(heap, (1, (s,)))
        k = 0
        while heap and k < max_k:
            length, path_tup = heapq.heappop(heap)
            if length >= min_len:
                yield list(path_tup)
                k += 1
            if length < max_len:
                path_set = set(path_tup)
                cur = path_tup[-1]
                for nb in adj[cur]:
                    if nb not in path_set and nb not in forbidden_nodes:
                        heapq.heappush(heap, (length + 1, (*path_tup, nb)))

    def _elen(a: int, b: int) -> float:
        return edge_len.get((min(a, b), max(a, b)), 0.0)

    for _pass in range(max_passes):
        centroid: dict = {}
        for g, dists in group_districts.items():
            slots_g = [slot_of[d] for d in dists]
            pts = cell_xy[slots_g]
            w_g = cell_area[slots_g]
            w_sum = w_g.sum()
            centroid[g] = np.average(pts, axis=0, weights=w_g) if w_sum > 0 else pts.mean(axis=0)

        locked: set[int] = set()
        improved = False

        for s1 in range(n):
            if s1 in locked:
                continue
            d1 = dist_at[s1]
            g1 = groups[d1]
            for s2 in adj[s1]:
                if s2 <= s1 or s2 in locked:
                    continue
                d2 = dist_at[s2]
                g2 = groups[d2]
                if g1 == g2:
                    continue

                delta_inertia = 2.0 * float(np.dot(cell_xy[s2] - cell_xy[s1], centroid[g2] - centroid[g1]))

                g1_slots = {slot_of[d] for d in group_districts[g1]}
                g2_slots = {slot_of[d] for d in group_districts[g2]}
                g1_rem = g1_slots - {s1}
                g2_rem = g2_slots - {s2}
                delta_shared = (
                    sum(_elen(s2, nb) for nb in adj[s2] if nb in g1_rem)
                    - sum(_elen(s1, nb) for nb in adj[s1] if nb in g1_rem)
                    + sum(_elen(s1, nb) for nb in adj[s1] if nb in g2_rem)
                    - sum(_elen(s2, nb) for nb in adj[s2] if nb in g2_rem)
                )

                g1_list = list(g1_slots)
                g2_list = list(g2_slots)
                g1_arr = cell_xy[g1_list]
                g2_arr = cell_xy[g2_list]
                w1 = cell_area[g1_list]
                w1 = w1 / w1.sum() if w1.sum() > 0 else np.ones(len(g1_list)) / len(g1_list)
                w2 = cell_area[g2_list]
                w2 = w2 / w2.sum() if w2.sum() > 0 else np.ones(len(g2_list)) / len(g2_list)
                i0_g1 = float(np.sum(w1 * np.sum((g1_arr - centroid[g1]) ** 2, axis=1)))
                i0_g2 = float(np.sum(w2 * np.sum((g2_arr - centroid[g2]) ** 2, axis=1)))
                I_0 = i0_g1 + i0_g2
                if I_0 == 0.0:
                    continue
                S_0 = sum(_elen(a, b) for a in g1_slots for b in adj[a] if b in g1_slots and b > a) + sum(
                    _elen(a, b) for a in g2_slots for b in adj[a] if b in g2_slots and b > a
                )
                criterion = delta_inertia / I_0 - (delta_shared / S_0 if S_0 > 0.0 else 0.0)
                if criterion >= 0.0:
                    continue

                if not _connected(g1_rem | {s2}):
                    continue
                if not _connected(g2_rem | {s1}):
                    continue

                slot_of[d1], slot_of[d2] = s2, s1
                dist_at[s1], dist_at[s2] = d2, d1
                locked.add(s1)
                locked.add(s2)
                improved = True
                break

        for g0, g0_dists in group_districts.items():
            if len(g0_dists) < 2:
                continue
            cg0 = centroid[g0]

            d0 = max(
                (d for d in g0_dists if slot_of[d] not in locked),
                key=lambda d: cell_area[slot_of[d]] * float(np.sum((cell_xy[slot_of[d]] - cg0) ** 2)),
                default=None,
            )
            if d0 is None:
                continue
            s0 = slot_of[d0]
            r_sq0 = cell_area[s0] * float(np.sum((cell_xy[s0] - cg0) ** 2))
            if r_sq0 == 0.0:
                continue

            for candidate in _enum_paths({s0}, locked, min_len=3, max_len=5, max_k=20):
                n_cand = len(candidate)

                if any(groups[dist_at[candidate[i]]] == g0 for i in range(1, n_cand - 1)):
                    continue

                s_final = candidate[n_cand - 2]
                if cell_area[s_final] * float(np.sum((cell_xy[s_final] - centroid[g0]) ** 2)) >= r_sq0:
                    continue

                affected_final: dict[int, int] = {}
                for i in range(n_cand - 1):
                    d = dist_at[candidate[i]]
                    affected_final[d] = candidate[i - 1] if i > 0 else candidate[n_cand - 2]

                bad = False
                checked_grps: set = set()
                for i in range(n_cand - 1):
                    grp_b = groups[dist_at[candidate[i]]]
                    if grp_b in checked_grps:
                        continue
                    checked_grps.add(grp_b)
                    grp_b_dists = group_districts[grp_b]
                    if len(grp_b_dists) < 2:
                        continue
                    cur_slots_b = [slot_of[dd] for dd in grp_b_dists]
                    if len(_connected_components(cur_slots_b)) > 1:
                        continue
                    fin_slots_b = [affected_final.get(dd, slot_of[dd]) for dd in grp_b_dists]
                    if len(_connected_components(fin_slots_b)) > 1:
                        bad = True
                        break
                if bad:
                    continue

                affected_groups = {groups[d] for d in affected_final}
                delta_I = 0.0
                delta_S = 0.0
                I_0_total = 0.0
                S_0_total = 0.0
                for g in affected_groups:
                    g_dists = group_districts[g]
                    cur_slots_g = [slot_of[dd] for dd in g_dists]
                    fin_slots_g = [affected_final.get(dd, slot_of[dd]) for dd in g_dists]
                    cg = centroid[g]
                    w_ag = cell_area[cur_slots_g]
                    w_ag = w_ag / w_ag.sum() if w_ag.sum() > 0 else np.ones(len(cur_slots_g)) / len(cur_slots_g)
                    w_ag_fin = cell_area[fin_slots_g]
                    w_ag_fin = (
                        w_ag_fin / w_ag_fin.sum()
                        if w_ag_fin.sum() > 0
                        else np.ones(len(fin_slots_g)) / len(fin_slots_g)
                    )
                    i_cur = float(np.sum(w_ag * np.sum((cell_xy[cur_slots_g] - cg) ** 2, axis=1)))
                    i_fin = float(np.sum(w_ag_fin * np.sum((cell_xy[fin_slots_g] - cg) ** 2, axis=1)))
                    delta_I += i_fin - i_cur
                    I_0_total += i_cur
                    cur_set = set(cur_slots_g)
                    fin_set = set(fin_slots_g)
                    s_cur_g = sum(_elen(a, b) for a in cur_set for b in adj[a] if b in cur_set and b > a)
                    s_fin_g = sum(_elen(a, b) for a in fin_set for b in adj[a] if b in fin_set and b > a)
                    delta_S += s_fin_g - s_cur_g
                    S_0_total += s_cur_g

                if I_0_total == 0.0:
                    continue
                criterion = delta_I / I_0_total - (delta_S / S_0_total if S_0_total > 0.0 else 0.0)
                if criterion >= 0.0:
                    continue

                for k in range(1, n_cand - 1):
                    d_mv = dist_at[candidate[k - 1]]
                    d_dp = dist_at[candidate[k]]
                    sa, sb = slot_of[d_mv], slot_of[d_dp]
                    slot_of[d_mv], slot_of[d_dp] = sb, sa
                    dist_at[sa], dist_at[sb] = d_dp, d_mv
                locked.update(candidate[:-1])
                improved = True
                break

        if not improved:
            break

    return slot_of


def repair_group_assignment(
    cells: list,
    groups: list,
    adj_pairs: list[tuple[int, int]],
    *,
    max_passes: int = 10,
    min_shared_length: float | None = None,
) -> np.ndarray:
    """Run Stage 1 (contiguity) then Stage 2 (adjacency) and return composed permutation.

    Parameters
    ----------
    cells : list
        Shapely geometry objects — one per slot.
    groups : list
        Group label for each slot (same length as *cells*).
    adj_pairs : list of (i, j)
        Pairs of geometry indices adjacent in the input.
    max_passes : int
        Maximum passes for each repair stage.
    min_shared_length : float or None
        Minimum shared border length for adjacency.

    Returns
    -------
    slot_of : np.ndarray[int], shape (n,)
        Composed permutation from both repair stages.
    """
    slot1, _ = repair_contiguity(cells, groups, max_passes=max_passes, min_shared_length=min_shared_length)
    cells_r = [cells[slot1[d]] for d in range(len(cells))]
    slot2 = repair_adjacency(
        cells_r,
        adj_pairs,
        max_passes=max_passes,
        groups=groups,
        min_shared_length=min_shared_length,
    )
    slot12 = slot1[slot2]
    cells_r3 = [cells[slot12[d]] for d in range(len(cells))]
    slot3 = repair_compactness(cells_r3, groups, max_passes=max_passes, min_shared_length=min_shared_length)
    return slot12[slot3]
