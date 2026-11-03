"""Candidate tiles of the grid assignment, restricted by ``compactness``."""

from __future__ import annotations

import heapq
from collections.abc import Sequence

import numpy as np
import shapely
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


_N_SCALES = 120  # scales of the outline tried


def _silhouette_rank(
    grid_centers: NDArray[np.floating], outline: shapely.geometry.base.BaseGeometry, anchor: NDArray[np.floating]
) -> NDArray[np.intp]:
    """Tiles in the order of the smallest scale of *outline* from which on it contains their center.

    The outline is scaled about its area centroid, which is placed at *anchor*.
    A tile counts only once it stays inside at every larger scale, so notches
    of the outline (lakes, bays) are not filled early by a shrunken copy.
    Ties, and tiles outside at every scale, go by distance to *anchor*.
    """
    c = np.asarray(outline.centroid.coords[0], dtype=float)
    first = np.zeros(len(grid_centers))
    for k in np.geomspace(0.03, 4.0, _N_SCALES):
        q = c + (grid_centers - anchor) / k
        first[~shapely.contains_xy(outline, q[:, 0], q[:, 1])] = k
    dist = np.hypot(*(grid_centers - anchor).T)
    return np.lexsort((np.arange(len(grid_centers)), dist, first)).astype(np.intp)


def _land_parts(
    geometries: Sequence, centroids: NDArray[np.floating]
) -> tuple[list[shapely.geometry.base.BaseGeometry], NDArray[np.intp]]:
    """Land parts of the union of *geometries* and the part of each region.

    Parts joined by one region (the islands of a multipart region) are one
    part, so a small part is always a whole region or several, such as an
    island state, which keeps its own patch.
    """
    geoms = shapely.make_valid(np.asarray(geometries, dtype=object))
    union = shapely.union_all(geoms).buffer(0)
    polys = list(getattr(union, "geoms", [union]))
    parent = list(range(len(polys)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    tree = shapely.STRtree(polys)
    for g in geoms:
        h = sorted(int(j) for j in tree.query(g, predicate="intersects"))
        for j in h[1:]:
            parent[find(j)] = find(h[0])
    roots = sorted({find(i) for i in range(len(polys))})
    groups = [shapely.union_all([polys[i] for i in range(len(polys)) if find(i) == r]) for r in roots]
    parts = groups
    label = np.zeros(len(geoms), dtype=np.intp)
    for i, g in enumerate(geoms):
        ref = shapely.Point(centroids[i]) if g.is_empty else g
        label[i] = int(np.argmin([part.distance(ref) for part in parts]))
    return parts, label


def _silhouette_set(
    centroids: NDArray[np.floating],
    grid_centers: NDArray[np.floating],
    geometries: Sequence,
    size: int,
    components: str,
    tile_adjacency: csr_array,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Mask of *size* tiles chosen by the silhouette of the land parts, and of those of the largest part."""
    n, m = len(centroids), len(grid_centers)
    parts, label = _land_parts(geometries, centroids)
    counts = np.bincount(label, minlength=len(parts))
    quota = counts + ((size - n) * counts) // n
    quota[int(np.argmax(counts))] += size - int(quota.sum())
    chosen = np.zeros(m, dtype=bool)
    sample = cdist(grid_centers[: min(m, 50)], grid_centers)
    step = float(np.where(sample > 0, sample, np.inf).min()) if m > 1 else 1.0
    adj = _canonical(tile_adjacency)
    adj = csr_array((adj.maximum(adj.T) != 0).astype(float))
    first, main = True, None
    for j in np.argsort(-counts, kind="stable"):
        anchor = np.asarray(parts[j].centroid.coords[0], dtype=float)
        take = int(quota[j])
        free = ~chosen
        if components != "separate" and not first:
            gap = components == "spaced"
            anchor, seed, free = _attach_anchor(anchor, chosen, grid_centers, step, take, adj, gap)
            if seed is not None:
                chosen[seed], take = True, take - 1
                free[seed] = False
        first = False
        free = np.flatnonzero(free)
        if quota[j] == 1 and components == "separate":
            # smaller than a tile: the free tile nearest the part
            d = shapely.distance(shapely.points(grid_centers[free]), parts[j])
            order = free[np.lexsort((np.hypot(*(grid_centers[free] - anchor).T), d))]
        else:
            order = free[_silhouette_rank(grid_centers[free], parts[j], anchor)]
        chosen[order[:take]] = True
        if main is None:
            main = chosen.copy()
    return chosen, main


def _attach_anchor(
    anchor: NDArray[np.floating],
    chosen: NDArray[np.bool_],
    grid_centers: NDArray[np.floating],
    step: float,
    quota: int,
    adj: csr_array,
    gap: bool,
) -> tuple[NDArray[np.floating], int | None, NDArray[np.bool_]]:
    """Anchor, first tile and free tiles that put a part against the chosen block.

    The first tile is the free tile nearest *anchor* among those next to the
    block (two edge steps from it with *gap*, which leaves one tile free); the
    anchor lies one patch radius outside it, away from the block. The free
    tiles exclude the one-tile border of the block with *gap*. *step* is the
    distance between neighboring tile centers.
    """
    c = chosen.astype(float)
    ring1 = (adj @ c > 0) & ~chosen
    if gap:
        ring2 = (adj @ ring1.astype(float) > 0) & ~chosen & ~ring1
        edge, free = np.flatnonzero(ring2), ~chosen & ~ring1
    else:
        edge, free = np.flatnonzero(ring1), ~chosen
    if len(edge) == 0:
        return anchor, None, ~chosen
    seed = int(edge[np.argmin(np.hypot(*(grid_centers[edge] - anchor).T))])
    direction = grid_centers[seed] - grid_centers[chosen].mean(axis=0)
    direction = direction / max(float(np.hypot(*direction)), 1e-12)
    return grid_centers[seed] + direction * step * np.sqrt(quota / np.pi), seed, free


def candidate_tiles(
    centroids: NDArray[np.floating],
    grid_centers: NDArray[np.floating],
    tile_adjacency: csr_array,
    compactness: float,
    geometries: Sequence | None = None,
    components: str = "separate",
) -> NDArray[np.intp]:
    """Sorted indices of the tiles the assignment may use.

    ``n + round((1 - compactness) * (m - n))`` tiles: all ``m`` at 0, ``n`` at 1.
    With the region *geometries*, the tiles whose centers lie in the smallest
    scaled copy of the outline of the map: each land part is scaled about its
    area centroid and gets tiles in proportion to its number of regions,
    the largest part first. ``components="connected"`` places the other parts
    against the block of the largest part, on the side of their real position;
    ``"spaced"`` does so with one free tile between each part and the block.
    Pockets are then closed. Without usable geometries the first tiles of the
    packed growth are taken.
    """
    n, m = len(centroids), len(grid_centers)
    size = n + round((1.0 - compactness) * (m - n))
    if size >= m:
        return np.arange(m, dtype=np.intp)
    chosen = movable = None
    if geometries is not None and len(geometries) == n:
        try:
            chosen, movable = _silhouette_set(
                centroids, grid_centers, list(geometries), size, components, tile_adjacency
            )
        except Exception:
            chosen = None
    if chosen is None:
        order = packed_order(centroids, grid_centers, tile_adjacency)
        chosen = np.zeros(m, dtype=bool)
        chosen[order[:size]] = True
    return np.flatnonzero(_close_pockets(chosen, tile_adjacency, movable))


def _close_pockets(
    chosen: NDArray[np.bool_], tile_adjacency: csr_array, movable: NDArray[np.bool_] | None = None
) -> NDArray[np.bool_]:
    """Swap tiles out of and into *chosen* while that lowers the number of pockets.

    A pocket is an unchosen tile with more than half of the largest number of
    edge neighbors chosen. Each swap moves a chosen tile with few chosen
    neighbors into the pocket with the most. Only tiles in *movable* (default:
    all chosen tiles) are moved, so the patches of small parts stay where placed.
    """
    adj = _canonical(tile_adjacency)
    adj = csr_array((adj.maximum(adj.T) != 0).astype(float))
    kk = int(np.diff(adj.indptr).max()) // 2 + 1
    chosen = chosen.copy()
    movable = chosen.copy() if movable is None else movable.copy()

    def pockets(c: NDArray[np.bool_]) -> tuple[int, NDArray[np.floating]]:
        cnt = adj @ c.astype(float)
        return int((~c & (cnt >= kk)).sum()), cnt

    n_p, cnt = pockets(chosen)
    for _ in range(int(chosen.sum())):
        if n_p == 0:
            break
        pocket = int(np.argmax(np.where(~chosen & (cnt >= kk), cnt, -1)))
        for loose in np.argsort(np.where(chosen & movable, cnt, np.inf), kind="stable")[:_N_LOOSE].tolist():
            if not (chosen[loose] and movable[loose]):
                continue
            trial = chosen.copy()
            trial[pocket], trial[loose] = True, False
            movable_t = movable.copy()
            movable_t[pocket], movable_t[loose] = True, False
            n_t, cnt_t = pockets(trial)
            if n_t < n_p:
                chosen, n_p, cnt, movable = trial, n_t, cnt_t, movable_t
                break
        else:
            break
    return chosen
