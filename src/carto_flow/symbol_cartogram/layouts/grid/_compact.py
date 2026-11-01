"""Candidate tiles of the grid assignment, restricted by ``compactness``."""

from __future__ import annotations

import heapq

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array
from scipy.spatial.distance import cdist

from ._placement import _canonical

_N_LOOSE = 20  # chosen tiles tried per pocket, fewest chosen neighbors first


def packed_order(
    centroids: NDArray[np.floating],
    grid_centers: NDArray[np.floating],
    tile_adjacency: csr_array,
) -> NDArray[np.intp]:
    """Tiles in the order of a packed growth from the middle of the centroids.

    The growth starts at the tile nearest the mean centroid and repeatedly
    adds the tile, among those sharing an edge with the grown set, that shares
    edges with the most grown tiles. Ties go to the tile nearest any centroid,
    then to the lower index. Filling the most enclosed tile first keeps every
    prefix of the order connected and free of notches, while the tie rule
    makes it follow the outline of the centroids.

    Tiles that the growth cannot reach come last, nearest centroid first.
    """
    m = len(grid_centers)
    if m == 0:
        return np.zeros(0, dtype=np.intp)
    adj = _canonical(tile_adjacency)
    adj = _canonical(adj.maximum(adj.T))
    nbrs = [adj.indices[adj.indptr[t] : adj.indptr[t + 1]].tolist() for t in range(m)]
    near = cdist(grid_centers, centroids).min(axis=1)
    seed = int(np.argmin(np.linalg.norm(grid_centers - centroids.mean(axis=0), axis=1)))

    count = np.zeros(m, dtype=np.intp)
    taken = np.zeros(m, dtype=bool)
    order: list[int] = []
    heap: list[tuple[int, float, int]] = [(0, float(near[seed]), seed)]
    while heap:
        neg, _, t = heapq.heappop(heap)
        if taken[t] or -neg != count[t]:
            continue  # taken already, or an outdated entry
        taken[t] = True
        order.append(t)
        for u in nbrs[t]:
            if not taken[u]:
                count[u] += 1
                heapq.heappush(heap, (-int(count[u]), float(near[u]), u))
    rest = np.flatnonzero(~taken)
    rest = rest[np.argsort(near[rest], kind="stable")]
    return np.concatenate([np.asarray(order, dtype=np.intp), rest])


def candidate_tiles(
    centroids: NDArray[np.floating],
    grid_centers: NDArray[np.floating],
    tile_adjacency: csr_array,
    compactness: float,
) -> NDArray[np.intp]:
    """Sorted indices of the tiles the assignment may use.

    The first ``n + round((1 - compactness) * (m - n))`` tiles of the packed
    order: all ``m`` tiles at 0, the ``n`` most packed tiles at 1.
    """
    n, m = len(centroids), len(grid_centers)
    size = n + round((1.0 - compactness) * (m - n))
    if size >= m:
        return np.arange(m, dtype=np.intp)
    order = packed_order(centroids, grid_centers, tile_adjacency)
    chosen = np.zeros(m, dtype=bool)
    chosen[order[:size]] = True
    return np.flatnonzero(_close_pockets(chosen, tile_adjacency))


def _close_pockets(chosen: NDArray[np.bool_], tile_adjacency: csr_array) -> NDArray[np.bool_]:
    """Swap tiles out of and into *chosen* while that lowers the number of pockets.

    A pocket is an unchosen tile with more than half of the largest number of
    edge neighbors chosen. Each swap moves a chosen tile with few chosen
    neighbors into the pocket with the most.
    """
    adj = _canonical(tile_adjacency)
    adj = csr_array((adj.maximum(adj.T) != 0).astype(float))
    kk = int(np.diff(adj.indptr).max()) // 2 + 1
    chosen = chosen.copy()

    def pockets(c: NDArray[np.bool_]) -> tuple[int, NDArray[np.floating]]:
        cnt = adj @ c.astype(float)
        return int((~c & (cnt >= kk)).sum()), cnt

    n_p, cnt = pockets(chosen)
    for _ in range(int(chosen.sum())):
        if n_p == 0:
            break
        pocket = int(np.argmax(np.where(~chosen & (cnt >= kk), cnt, -1)))
        for loose in np.argsort(np.where(chosen, cnt, np.inf), kind="stable")[:_N_LOOSE].tolist():
            trial = chosen.copy()
            trial[pocket], trial[loose] = True, False
            n_t, cnt_t = pockets(trial)
            if n_t < n_p:
                chosen, n_p, cnt = trial, n_t, cnt_t
                break
        else:
            break
    return chosen
