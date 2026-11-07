"""Assignment: capacitated optimal transport of tiles to groups, and contiguity repair."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_array
from scipy.spatial import cKDTree

#: Cost of a tile that a group may not use (a tile of another land part).
FORBIDDEN = 1.0e6

#: Label of an allowed tile that no group uses, in a label array over the lattice.
RESERVE = -1
#: Label of a tile that is not allowed.
UNUSABLE = -2


def tile_step(centers: NDArray[np.floating]) -> float:
    """Median distance between neighboring tile centers."""
    d, _ = cKDTree(centers).query(centers, k=2)
    return float(np.median(d[:, 1]))


def neighbor_lists(adjacency: csr_array) -> list[list[int]]:
    """Neighbors of each tile as lists."""
    adj = adjacency.tocsr()
    return [adj.indices[adj.indptr[t] : adj.indptr[t + 1]].tolist() for t in range(adj.shape[0])]


def components_of(tiles: Sequence[int], nbrs: list[list[int]]) -> list[list[int]]:
    """Connected components of a set of tiles, in order of first tile."""
    members = set(tiles)
    seen: set[int] = set()
    out: list[list[int]] = []
    for t in tiles:
        if t in seen:
            continue
        seen.add(t)
        stack, comp = [t], [t]
        while stack:
            x = stack.pop()
            for u in nbrs[x]:
                if u in members and u not in seen:
                    seen.add(u)
                    stack.append(u)
                    comp.append(u)
        out.append(comp)
    return out


def group_cost(
    geometries: Sequence,
    centers: NDArray[np.floating],
    step: float,
    centroid_weight: float = 0.1,
) -> NDArray[np.floating]:
    """Cost of giving each tile to each group.

    The squared distance (in tile steps) from the tile center to the geometry
    of the group, plus *centroid_weight* times the squared distance to its
    centroid.

    Parameters
    ----------
    geometries : sequence of shapely geometries
        One geometry per group.
    centers : NDArray[np.floating]
        Tile centers, shape ``(k, 2)``.
    step : float
        Distance between neighboring tile centers.
    centroid_weight : float
        Weight of the centroid term. Default: 0.1.

    Returns
    -------
    NDArray[np.floating]
        Cost, shape ``(len(geometries), k)``.
    """
    points = shapely.points(centers)
    cost = np.zeros((len(geometries), len(centers)))
    for i, g in enumerate(geometries):
        to_geometry = shapely.distance(points, g) / step
        centroid = np.asarray(g.centroid.coords[0], dtype=float)
        to_centroid = np.hypot(*(centers - centroid).T) / step
        cost[i] = to_geometry**2 + centroid_weight * to_centroid**2
    return cost


def hull_cost(centers: NDArray[np.floating], step: float) -> NDArray[np.floating]:
    """Distance (in tile steps) from each center to the boundary of the convex hull of the centers."""
    hull = shapely.convex_hull(shapely.multipoints(centers))
    target = hull.boundary if hull.geom_type == "Polygon" else hull
    return np.asarray(shapely.distance(shapely.points(centers), target)) / step


def solve_transport(
    cost: NDArray[np.floating], capacity: NDArray[np.intp], reserve_cost: NDArray[np.floating]
) -> NDArray[np.intp]:
    """Exact assignment of tiles to groups with capacities and a reserve sink.

    Group g takes exactly ``capacity[g]`` tiles, the remaining tiles go to the
    reserve at the tile's reserve cost. The problem is solved as a square
    assignment on rows replicated by capacity, so memory and time grow with
    the square of the number of usable tiles.

    Parameters
    ----------
    cost : NDArray[np.floating]
        Group-by-tile cost, shape ``(G, k)``.
    capacity : NDArray[np.intp]
        Tiles per group, shape ``(G,)``; sums to at most *k*.
    reserve_cost : NDArray[np.floating]
        Cost of leaving each tile unused, shape ``(k,)``.

    Returns
    -------
    NDArray[np.intp]
        Group of each tile, ``RESERVE`` for unused tiles.
    """
    n_groups, k = cost.shape
    n_reserve = k - int(capacity.sum())
    blocks = [np.repeat(cost, capacity, axis=0)]
    if n_reserve > 0:
        blocks.append(np.tile(reserve_cost, (n_reserve, 1)))
    square = np.vstack(blocks).T
    rows, cols = linear_sum_assignment(square)
    owner = np.concatenate([np.repeat(np.arange(n_groups), capacity), np.full(n_reserve, RESERVE)])
    label = np.empty(k, dtype=np.intp)
    label[rows] = owner[cols]
    return label


def transport_labels(
    cost: NDArray[np.floating],
    capacity: NDArray[np.intp],
    allowed: NDArray[np.intp],
    centers: NDArray[np.floating],
    nbrs: list[list[int]],
    step: float,
    neighbor_iterations: int,
    beta: float = 1.0,
) -> tuple[NDArray[np.intp], NDArray[np.floating]]:
    """Assign the allowed tiles to groups, with unused tiles pushed to the hull.

    A tile left unused costs *beta* times its distance to the boundary of the
    convex hull of the allowed tiles, so surplus tiles end up at the edge of
    the allowed set instead of as holes inside it. With *neighbor_iterations*
    the assignment is solved again with an added cost of 1 per used neighbor
    of an unused tile, from the previous result.

    Parameters
    ----------
    cost : NDArray[np.floating]
        Group-by-lattice-tile cost, shape ``(G, m)``.
    capacity : NDArray[np.intp]
        Tiles per group.
    allowed : NDArray[np.intp]
        Usable tiles.
    centers : NDArray[np.floating]
        Tile centers, shape ``(m, 2)``.
    nbrs : list of list of int
        Tile neighbors.
    step : float
        Distance between neighboring tile centers.
    neighbor_iterations : int
        Number of additional solves with the neighbor term.
    beta : float
        Weight of the hull term. Default: 1.

    Returns
    -------
    label : NDArray[np.intp]
        Group of each lattice tile, ``RESERVE`` or ``UNUSABLE`` otherwise.
    reserve_cost : NDArray[np.floating]
        Cost of leaving each lattice tile unused (zero for unusable tiles).
    """
    m = len(centers)
    base = beta * hull_cost(centers[allowed], step)
    local_cost = cost[:, allowed]
    reserve = base
    label = np.full(m, UNUSABLE, dtype=np.intp)
    label[allowed] = solve_transport(local_cost, capacity, reserve)
    for _ in range(neighbor_iterations):
        used = (label >= 0).astype(float)
        count = np.array([sum(used[u] for u in nbrs[t]) for t in allowed])
        reserve = base + count
        label[allowed] = solve_transport(local_cost, capacity, reserve)
    reserve_full = np.zeros(m)
    reserve_full[allowed] = reserve
    return label, reserve_full


def _extra_components(tiles: Iterable[int], nbrs: list[list[int]]) -> int:
    return max(len(components_of(list(tiles), nbrs)) - 1, 0)


def swap_search(
    label: NDArray[np.intp],
    cost: NDArray[np.floating],
    reserve_cost: NDArray[np.floating],
    nbrs: list[list[int]],
    max_passes: int = 60,
) -> NDArray[np.intp]:
    """Swap tiles between neighboring groups while that reduces split groups.

    For each pair of neighboring groups (a group and the reserve count as
    well) the best exchange of one tile of each, from tiles that border the
    other group, is taken when it lowers the number of extra components of the
    two groups, or leaves it unchanged and lowers the cost. Group sizes do not
    change, and a group never receives a forbidden tile.

    Parameters
    ----------
    label : NDArray[np.intp]
        Group of each lattice tile (``RESERVE``, ``UNUSABLE`` otherwise).
    cost : NDArray[np.floating]
        Group-by-lattice-tile cost, shape ``(G, m)``.
    reserve_cost : NDArray[np.floating]
        Cost of each tile in the reserve, shape ``(m,)``.
    nbrs : list of list of int
        Tile neighbors.
    max_passes : int
        Upper bound on passes over all neighboring pairs. Default: 60.

    Returns
    -------
    NDArray[np.intp]
        Updated labels.
    """
    label = label.copy()
    n_groups = cost.shape[0]
    members: dict[int, set[int]] = {g: set(np.flatnonzero(label == g).tolist()) for g in range(RESERVE, n_groups)}

    def price(g: int, t: int) -> float:
        return float(reserve_cost[t] if g < 0 else cost[g, t])

    for _ in range(max_passes):
        improved = False
        pairs: set[tuple[int, int]] = set()
        for g in range(n_groups):
            for t in members[g]:
                for u in nbrs[t]:
                    b = int(label[u])
                    if b != g and b != UNUSABLE:
                        pairs.add((g, b) if b < 0 else (min(g, b), max(g, b)))
        for a, b in sorted(pairs):
            tiles_a = [t for t in sorted(members[a]) if any(label[u] == b for u in nbrs[t])]
            tiles_b = [u for u in sorted(members[b]) if any(label[v] == a for v in nbrs[u])]
            if not tiles_a or not tiles_b:
                continue
            extra_a = _extra_components(members[a], nbrs)
            extra_b = _extra_components(members[b], nbrs) if b >= 0 else 0
            best = None
            for t in tiles_a:
                if b >= 0 and price(b, t) >= FORBIDDEN / 2:
                    continue
                for u in tiles_b:
                    if price(a, u) >= FORBIDDEN / 2:
                        continue
                    d_cost = price(b, t) + price(a, u) - price(a, t) - price(b, u)
                    if extra_a == 0 and extra_b == 0 and d_cost >= -1e-9:
                        continue
                    after = _extra_components((members[a] - {t}) | {u}, nbrs)
                    if b >= 0:
                        after += _extra_components((members[b] - {u}) | {t}, nbrs)
                    key = (after - extra_a - extra_b, d_cost)
                    if (key[0] < 0 or (key[0] == 0 and key[1] < -1e-9)) and (best is None or key < best[0]):
                        best = (key, t, u)
            if best is not None:
                _, t, u = best
                members[a].remove(t)
                members[a].add(u)
                members[b].remove(u)
                members[b].add(t)
                label[t], label[u] = b, a
                improved = True
        if not improved:
            break
    return label


def _extend_chain(
    stray: int,
    groups: list[int],
    tiles: list[int],
    limit: int,
    label: NDArray[np.intp],
    nbrs: list[list[int]],
    price: Callable[[int, int], float],
    out: list[tuple[int, list[int], list[int]]],
) -> None:
    """Collect the chains that start with *groups* and *tiles* and end by taking *stray*.

    ``tiles[i]`` is the tile that ``groups[i]`` takes; the last group takes
    *stray*. A chain ends at the reserve or after *limit* groups.
    """
    last = groups[-1]
    if len(groups) - 1 == limit or last == RESERVE:
        if last == RESERVE or (any(label[v] == last for v in nbrs[stray]) and price(last, stray) < FORBIDDEN / 2):
            out.append((stray, list(groups), list(tiles)))
        return
    reach = sorted({x for t in np.flatnonzero(label == last).tolist() for x in nbrs[t]})
    for u in reach:
        h = int(label[u])
        if h == UNUSABLE or h in groups or u == stray or u in tiles or price(last, u) >= FORBIDDEN / 2:
            continue
        _extend_chain(stray, [*groups, h], [*tiles, u], limit, label, nbrs, price, out)


def chain_search(
    label: NDArray[np.intp],
    cost: NDArray[np.floating],
    reserve_cost: NDArray[np.floating],
    nbrs: list[list[int]],
    depth: int = 3,
    max_iter: int = 20,
) -> NDArray[np.intp]:
    """Reconnect split groups by shifting tiles along a chain of up to *depth* groups.

    A stray tile of a split group moves to the last group of a chain, each
    group of the chain takes one tile from the next group, and the first
    group takes a tile next to its main piece. A chain is applied when no
    group on it gains a component, the total number of components drops, and
    no group receives a forbidden tile; the cheapest chain of the shortest
    length wins.

    Parameters
    ----------
    label : NDArray[np.intp]
        Group of each lattice tile.
    cost : NDArray[np.floating]
        Group-by-lattice-tile cost, shape ``(G, m)``.
    reserve_cost : NDArray[np.floating]
        Cost of each tile in the reserve, shape ``(m,)``.
    nbrs : list of list of int
        Tile neighbors.
    depth : int
        Maximum number of groups after the split group. Default: 3.
    max_iter : int
        Upper bound on passes over all groups. Default: 20.

    Returns
    -------
    NDArray[np.intp]
        Updated labels.
    """
    label = label.copy()
    n_groups = cost.shape[0]

    def price(g: int, t: int) -> float:
        return float(reserve_cost[t] if g < 0 else cost[g, t])

    def tiles_of(g: int, lab: NDArray[np.intp] | None = None) -> list[int]:
        return np.flatnonzero((label if lab is None else lab) == g).tolist()

    for _ in range(max_iter):
        progress = False
        for a in range(n_groups):
            pieces = components_of(tiles_of(a), nbrs)
            if len(pieces) < 2:
                continue
            pieces.sort(key=len, reverse=True)
            main = set(pieces[0])
            strays = [t for piece in pieces[1:] for t in piece]
            firsts = sorted({u for t in main for u in nbrs[t] if label[u] not in (a, UNUSABLE)})
            found = None
            for limit in range(1, depth + 1):
                candidates: list[tuple[int, list[int], list[int]]] = []

                for s in strays:
                    for t1 in firsts:
                        if price(a, t1) < FORBIDDEN / 2:
                            _extend_chain(s, [a, int(label[t1])], [t1], limit, label, nbrs, price, candidates)
                best = None
                for s, groups, tiles in candidates:
                    new = label.copy()
                    for g, t in zip(groups, tiles, strict=False):
                        new[t] = g
                    new[s] = groups[-1]
                    involved = [g for g in groups if g >= 0]
                    before = [len(components_of(tiles_of(g), nbrs)) for g in involved]
                    after = [len(components_of(tiles_of(g, new), nbrs)) for g in involved]
                    if sum(after) >= sum(before) or any(x > y for x, y in zip(after, before, strict=True)):
                        continue
                    d_cost = sum(price(g, t) - price(int(label[t]), t) for g, t in zip(groups, tiles, strict=False))
                    d_cost += price(groups[-1], s) - price(a, s)
                    if best is None or (sum(after) - sum(before), d_cost) < best[0]:
                        best = ((sum(after) - sum(before), d_cost), new)
                if best is not None:
                    found = best[1]
                    break
            if found is not None:
                label = found
                progress = True
        if not progress:
            break
    return label


def split_groups(label: NDArray[np.intp], n_groups: int, nbrs: list[list[int]]) -> tuple[int, int]:
    """Number of groups in more than one piece, and the tiles outside their largest piece."""
    split = stray = 0
    for g in range(n_groups):
        tiles = np.flatnonzero(label == g).tolist()
        if len(tiles) < 2:
            continue
        sizes = [len(c) for c in components_of(tiles, nbrs)]
        if len(sizes) > 1:
            split += 1
            stray += len(tiles) - max(sizes)
    return split, stray
