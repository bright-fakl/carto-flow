"""Allocation: which lattice tiles the tile layout may use."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

_N_SCALES = 120  # scales of the outline tried when ranking tiles
_N_LOOSE = 20  # chosen tiles tried per pocket, fewest chosen neighbors first


@dataclass
class Allocation:
    """The tiles the assignment may use.

    Attributes
    ----------
    tiles : NDArray[np.intp]
        Sorted indices of the usable tiles.
    part_of_tile : NDArray[np.intp]
        Land part each lattice tile belongs to, shape ``(m,)``; -1 for tiles
        that are not usable.
    """

    tiles: NDArray[np.intp]
    part_of_tile: NDArray[np.intp]


def allocation_size(n: int, m: int, compactness: float) -> int:
    """Number of usable tiles on the geometric compactness scale.

    ``round(n * (m / n) ** (1 - compactness))``: *n* tiles at compactness 1,
    all *m* at 0, and a constant ratio of usable to needed tiles per step of
    compactness in between.

    Parameters
    ----------
    n : int
        Tiles needed.
    m : int
        Tiles of the lattice in the window.
    compactness : float
        Compactness in [0, 1].

    Returns
    -------
    int
        Number of usable tiles, between *n* and *m*.
    """
    if m <= n:
        return m
    return int(min(m, max(n, round(n * (m / n) ** (1.0 - compactness)))))


def land_parts(geometries: Sequence) -> tuple[list, NDArray[np.intp]]:
    """Land parts of a set of geometries and the part of each geometry.

    A land part is a connected component of the union of the geometries.
    Components joined by one geometry (the islands of a multipart region) form
    one land part.

    Parameters
    ----------
    geometries : sequence of shapely geometries
        Geometries whose union is split.

    Returns
    -------
    parts : list of shapely geometries
        One geometry per land part, largest area first.
    labels : NDArray[np.intp]
        Index into *parts* for each geometry.
    """
    geoms = shapely.make_valid(np.asarray(geometries, dtype=object))
    union = shapely.union_all(geoms).buffer(0)
    pieces = list(getattr(union, "geoms", [union]))
    parent = list(range(len(pieces)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    tree = shapely.STRtree(pieces)
    for g in geoms:
        hit = sorted(int(j) for j in tree.query(g, predicate="intersects"))
        for j in hit[1:]:
            parent[find(j)] = find(hit[0])
    roots = sorted({find(i) for i in range(len(pieces))})
    merged = [shapely.union_all([pieces[i] for i in range(len(pieces)) if find(i) == r]) for r in roots]
    parts = sorted(merged, key=lambda p: (-round(p.area, 6), round(p.centroid.x, 6), round(p.centroid.y, 6)))
    labels = np.empty(len(geoms), dtype=np.intp)
    for i, g in enumerate(geoms):
        labels[i] = int(np.argmin([p.distance(g) for p in parts]))
    return parts, labels


def _silhouette_rank(
    centers: NDArray[np.floating], outline: shapely.Geometry, anchor: NDArray[np.floating]
) -> NDArray[np.intp]:
    """Order of *centers* by the smallest scale of *outline* that contains them.

    The outline is scaled about its area centroid, which is placed at *anchor*.
    Ties, and centers outside the outline at every scale, go by distance to
    *anchor*, then by index.
    """
    c = np.asarray(outline.centroid.coords[0], dtype=float)
    first = np.full(len(centers), np.inf)
    for k in np.geomspace(0.03, 4.0, _N_SCALES)[::-1]:
        q = c + (centers - anchor) / k
        first[shapely.contains_xy(outline, q[:, 0], q[:, 1])] = k
    dist = np.hypot(*(centers - anchor).T)
    return np.lexsort((np.arange(len(centers)), dist, first)).astype(np.intp)


def _symmetric(adjacency: csr_array) -> csr_array:
    return csr_array((adjacency.maximum(adjacency.T) != 0).astype(float))


def _attach_anchor(
    anchor: NDArray[np.floating],
    chosen: NDArray[np.bool_],
    centers: NDArray[np.floating],
    step: float,
    quota: int,
    adjacency: csr_array,
    gap: bool,
) -> tuple[NDArray[np.floating], int | None, NDArray[np.bool_]]:
    """Anchor, first tile and free tiles that put a part against the chosen block.

    The first tile is the free tile nearest *anchor* among those next to the
    block (two edge steps from it with *gap*, which leaves one tile free); the
    anchor lies one patch radius outside it, away from the block. With *gap*
    the free tiles exclude the one-tile border of the block.
    """
    block = chosen.astype(float)
    ring1 = (adjacency @ block > 0) & ~chosen
    if gap:
        ring2 = (adjacency @ ring1.astype(float) > 0) & ~chosen & ~ring1
        edge, free = np.flatnonzero(ring2), ~chosen & ~ring1
    else:
        edge, free = np.flatnonzero(ring1), ~chosen
    if len(edge) == 0:
        return anchor, None, ~chosen
    seed = int(edge[np.argmin(np.hypot(*(centers[edge] - anchor).T))])
    direction = centers[seed] - centers[chosen].mean(axis=0)
    direction = direction / max(float(np.hypot(*direction)), 1e-12)
    return centers[seed] + direction * step * np.sqrt(quota / np.pi), seed, free


def _connect_patch(
    chosen: NDArray[np.bool_],
    owner: NDArray[np.intp],
    part: int,
    tiles: NDArray[np.intp],
    order: NDArray[np.intp],
    adjacency: csr_array,
) -> None:
    """Make the tiles of one land part a single piece, in place.

    The tiles of the smaller pieces move to the free tiles next to the largest
    piece that come first in the part's *order*.
    """
    if len(tiles) < 2:
        return
    label = connected_components(adjacency[tiles][:, tiles], directed=False)[1]
    if label.max() == 0:
        return
    keep = label == np.argmax(np.bincount(label))
    chosen[tiles[~keep]] = False
    owner[tiles[~keep]] = -1
    rank = np.full(len(chosen), len(chosen) + 1)
    rank[order] = np.arange(len(order))
    block = np.zeros(len(chosen), dtype=bool)
    block[tiles[keep]] = True
    for _ in range(int((~keep).sum())):
        edge = np.flatnonzero((adjacency @ block.astype(float) > 0) & ~chosen)
        if len(edge) == 0:
            break
        t = int(edge[np.argmin(rank[edge])])
        chosen[t] = block[t] = True
        owner[t] = part


def _close_pockets(
    chosen: NDArray[np.bool_],
    owner: NDArray[np.intp],
    movable: NDArray[np.bool_],
    adjacency: csr_array,
) -> None:
    """Swap chosen tiles into pockets while that lowers the number of pockets, in place.

    A pocket is an unchosen tile with more than half of the largest number of
    edge neighbors chosen. Each swap moves a movable chosen tile with few
    chosen neighbors into the pocket with the most; the pocket takes over its
    land part.
    """
    k_min = int(np.diff(adjacency.indptr).max()) // 2 + 1

    def pockets(c: NDArray[np.bool_]) -> tuple[int, NDArray[np.floating]]:
        count = adjacency @ c.astype(float)
        return int((~c & (count >= k_min)).sum()), count

    n_pockets, count = pockets(chosen)
    for _ in range(int(chosen.sum())):
        if n_pockets == 0:
            break
        pocket = int(np.argmax(np.where(~chosen & (count >= k_min), count, -1)))
        loose_order = np.argsort(np.where(chosen & movable, count, np.inf), kind="stable")[:_N_LOOSE]
        for loose in loose_order.tolist():
            if not (chosen[loose] and movable[loose]):
                continue
            trial = chosen.copy()
            trial[pocket], trial[loose] = True, False
            n_trial, count_trial = pockets(trial)
            if n_trial < n_pockets:
                chosen[:] = trial
                n_pockets, count = n_trial, count_trial
                owner[pocket], owner[loose] = owner[loose], -1
                movable[pocket], movable[loose] = True, False
                break
        else:
            break


def allocate_tiles(
    outlines: Sequence,
    need: NDArray[np.intp],
    centers: NDArray[np.floating],
    adjacency: csr_array,
    size: int,
    components: str,
) -> Allocation:
    """Choose *size* tiles by the silhouette of the land parts.

    Each land part is scaled about its area centroid and receives tiles in
    proportion to the number it needs, taken in the order of the smallest scale
    of its outline that contains their centers; the first part receives the
    remainder, and with ``"separate"`` chooses its tiles last. ``components="connected"`` puts the other parts against the
    block of the first part on the side of their real position, ``"spaced"``
    with one free tile between each part and the block. Each part is then made
    one piece and pockets in the result are closed by moving tiles of the first
    part.

    Parameters
    ----------
    outlines : sequence of shapely geometries
        Outline of each land part, the part that needs most tiles first.
    need : NDArray[np.intp]
        Tiles needed in each land part.
    centers : NDArray[np.floating]
        Tile centers, shape ``(m, 2)``.
    adjacency : csr_array
        Tile adjacency (shared edge).
    size : int
        Number of tiles to choose; at least ``need.sum()``.
    components : {"separate", "connected", "spaced"}
        How the land parts are positioned relative to each other.

    Returns
    -------
    Allocation
        The usable tiles and the land part of each.
    """
    m = len(centers)
    n = int(need.sum())
    adj = _symmetric(adjacency)
    quota = need + ((size - n) * need) // n
    quota[0] += size - int(quota.sum())
    sample = cdist(centers[: min(m, 50)], centers)
    step = float(np.where(sample > 0, sample, np.inf).min()) if m > 1 else 1.0

    chosen = np.zeros(m, dtype=bool)
    owner = np.full(m, -1, dtype=np.intp)
    patches: list[tuple[int, NDArray[np.intp], NDArray[np.intp]]] = []
    # Separate parts are placed at their own position: the smaller parts choose their tiles first, so the first
    # part flows around them and a small part is not pushed away by the tiles of a large neighbor.
    sequence = [*range(1, len(outlines)), 0] if components == "separate" else range(len(outlines))
    for j in sequence:
        outline = outlines[j]
        anchor = np.asarray(outline.centroid.coords[0], dtype=float)
        take = int(quota[j])
        free: NDArray[np.bool_] = ~chosen
        seeds: list[int] = []
        if components != "separate" and j > 0:
            anchor, seed, free = _attach_anchor(anchor, chosen, centers, step, take, adj, components == "spaced")
            if seed is not None:
                chosen[seed], owner[seed], take = True, j, take - 1
                seeds.append(seed)
                free[seed] = False
        free_idx = np.flatnonzero(free)
        if quota[j] == 1 and components == "separate":
            # smaller than a tile: the free tile nearest the part
            near = shapely.distance(shapely.points(centers[free_idx]), outline)
            order = free_idx[np.lexsort((np.hypot(*(centers[free_idx] - anchor).T), near))]
        else:
            order = free_idx[_silhouette_rank(centers[free_idx], outline, anchor)]
        chosen[order[:take]] = True
        owner[order[:take]] = j
        patches.append((j, np.concatenate([seeds, order[:take]]).astype(np.intp), order))
    for j, tiles, order in patches:
        _connect_patch(chosen, owner, j, tiles, order, adj)
    _close_pockets(chosen, owner, owner == 0, adj)
    return Allocation(tiles=np.flatnonzero(chosen).astype(np.intp), part_of_tile=owner)
