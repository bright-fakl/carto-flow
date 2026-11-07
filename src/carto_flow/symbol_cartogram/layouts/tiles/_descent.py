"""Contact descent: swap search for regions that hold one tile each."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array
from scipy.sparse.csgraph import shortest_path

_HOP_CAP = 9.0  # cap of the squared hop gap of a neighboring pair that does not touch
_RELOCATE_REACH = 2  # hops a region may move to a free tile
_SWAP_REACH = 3  # hops between two regions that may exchange their tiles


def contact_descent(
    tile_of: NDArray[np.intp],
    allowed: NDArray[np.intp],
    centers: NDArray[np.floating],
    nbrs: list[list[int]],
    targets: NDArray[np.floating],
    neighbor_pairs: Iterable[tuple[int, int]],
    step: float,
    tile_ok: Callable[[int, int], bool] | None = None,
    protect_contacts: bool = False,
    pocket_weight: float = 1.0,
    max_move: float = 2.0,
    max_total: float = 1.2,
    max_sweeps: int = 30,
) -> NDArray[np.intp]:
    """Improve the contacts of regions that hold one tile each by swaps and moves.

    The energy is 0.5 times the displacement of each region from its target
    (in tile steps), plus 0.5 times the squared hop gap (capped) of each pair of
    neighboring regions that do not touch, plus 0.5 for each touching pair of
    regions that are not neighbors, plus *pocket_weight* per pocket. A pocket
    is an unused usable tile with more than half of the largest number of
    neighbors used. Two regions within three hops exchange tiles, and a region
    moves to a free usable tile within two hops, whenever that lowers the
    energy and keeps every displacement within *max_move* of its start and
    the total within *max_total* times its start.

    Parameters
    ----------
    tile_of : NDArray[np.intp]
        Tile of each region, shape ``(G,)``.
    allowed : NDArray[np.intp]
        Usable tiles.
    centers : NDArray[np.floating]
        Tile centers, shape ``(m, 2)``.
    nbrs : list of list of int
        Tile neighbors.
    targets : NDArray[np.floating]
        Position each region is drawn to, shape ``(G, 2)``.
    neighbor_pairs : iterable of (int, int)
        Pairs of regions that border each other, ``a < b``.
    step : float
        Distance between neighboring tile centers.
    tile_ok : callable, optional
        ``tile_ok(tile, region)`` is False when the region may not use the tile.
    protect_contacts : bool
        Reject a move or swap that lowers the number of touching neighbor
        pairs among the regions it moves.
    pocket_weight : float
        Weight of a pocket. Default: 1.
    max_move : float
        Largest increase of a region's displacement over its start, in tile
        steps. Default: 2.
    max_total : float
        Largest total displacement as a multiple of the start. Default: 1.2.
    max_sweeps : int
        Upper bound on sweeps. Default: 30.

    Returns
    -------
    NDArray[np.intp]
        Tile of each region.
    """
    m = len(centers)
    n = len(tile_of)
    tile_of = tile_of.copy()
    occupant = np.full(m, -2, dtype=np.intp)
    occupant[allowed] = -1
    occupant[tile_of] = np.arange(n)
    k_pocket = max(len(x) for x in nbrs) // 2 + 1

    pairs = sorted({(int(a), int(b)) for a, b in neighbor_pairs})
    neighbors_of: dict[int, list[int]] = defaultdict(list)
    for a, b in pairs:
        neighbors_of[a].append(b)
        neighbors_of[b].append(a)
    pair_set = set(pairs)

    rows = [t for t in range(m) for _ in nbrs[t]]
    cols = [u for t in range(m) for u in nbrs[t]]
    hops = shortest_path(csr_array((np.ones(len(rows)), (rows, cols)), shape=(m, m)), unweighted=True, directed=False)

    def displacement(g: int) -> float:
        return float(np.hypot(*(centers[tile_of[g]] - targets[g])) / step)

    nbr_sets = [set(x) for x in nbrs]

    def local_energy(regions: Iterable[int]) -> tuple[float, int]:
        """Energy of the terms involving *regions*, and the number of touching neighbor pairs among them."""
        regions = sorted(set(regions))
        energy = 0.0
        pairs_seen: set[tuple[int, int]] = set()
        n_true = 0
        for g in regions:
            energy += 0.5 * displacement(g)
            others = {int(occupant[u]) for u in nbrs[tile_of[g]] if occupant[u] >= 0}
            others.update(neighbors_of[g])
            for h in others:
                p = (min(g, h), max(g, h))
                if p in pairs_seen:
                    continue
                pairs_seen.add(p)
                touch = tile_of[h] in nbr_sets[tile_of[g]]
                if p in pair_set:
                    if touch:
                        n_true += 1
                    else:
                        gap = hops[tile_of[g], tile_of[h]] - 1
                        energy += 0.5 * min(gap * gap, _HOP_CAP)
                elif touch:
                    energy += 0.5
        tiles = set(tile_of[regions].tolist())
        for t in list(tiles):
            tiles.update(nbrs[t])
        n_pockets = sum(
            1 for y in tiles if occupant[y] == -1 and sum(1 for q in nbrs[y] if occupant[q] >= 0) >= k_pocket
        )
        return energy + pocket_weight * n_pockets, n_true

    start = np.array([displacement(g) for g in range(n)])
    total_start = float(start.sum())
    state = {"total": total_start}

    def place(g: int, tile: int) -> None:
        occupant[tile_of[g]] = -1
        tile_of[g] = tile
        occupant[tile] = g

    def attempt(moves: list[tuple[int, int]]) -> bool:
        """Move regions to tiles, keep the result when the energy drops."""
        regions = [g for g, _ in moves]
        if tile_ok is not None and any(not tile_ok(t, g) for g, t in moves):
            return False
        old = [(g, int(tile_of[g])) for g in regions]
        before, true_before = local_energy(regions)
        total_before = sum(displacement(g) for g in regions)
        # swap or relocation: release all tiles first so that exchanges work
        for g, _ in moves:
            occupant[tile_of[g]] = -1
        for g, t in moves:
            tile_of[g] = t
            occupant[t] = g
        ok = all(displacement(g) <= start[g] + max_move for g in regions)
        if ok:
            total_after = sum(displacement(g) for g in regions)
            ok = state["total"] + total_after - total_before <= max_total * total_start
        if ok:
            after, true_after = local_energy(regions)
            ok = after < before - 1e-9 and not (protect_contacts and true_after < true_before)
            if ok:
                state["total"] += total_after - total_before
        if ok:
            return True
        for g, _ in moves:
            occupant[tile_of[g]] = -1
        for g, t in old:
            tile_of[g] = t
            occupant[t] = g
        return False

    for _ in range(max_sweeps):
        moved = 0
        for t in range(m):
            a = occupant[t]
            if a < 0:
                continue
            candidates = np.flatnonzero((hops[t] <= _SWAP_REACH) & (hops[t] > 0))
            for u in candidates.tolist():
                b = occupant[u]
                if b >= 0 and b > a and attempt([(int(a), u), (int(b), t)]):
                    moved += 1
                    break
        for g in range(n):
            reach = np.flatnonzero(hops[tile_of[g]] <= _RELOCATE_REACH)
            for u in reach.tolist():
                if occupant[u] == -1 and attempt([(g, u)]):
                    moved += 1
        if moved == 0:
            break
    return tile_of
