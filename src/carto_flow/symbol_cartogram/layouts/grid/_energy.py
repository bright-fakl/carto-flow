"""Grid assignment as the minimization of one explicit energy.

The energy of an assignment ``t`` (region ``i`` on tile ``t[i]``) is measured in
tile units: lengths in tile steps (the median distance between neighboring tile
centers), so it does not depend on the extent of the map or of the lattice::

    E(t) = distance_weight * sum_i d(i, t[i])
         + contact_weight * (sum_{i<j neighbors}     A[i, j] * p(t[i], t[j])
                             + sum_{i<j not neighbors} q(t[i], t[j]))
         + connectivity_weight * K(t)

- ``d(i, k)``: distance from the centroid of region ``i`` to the center of tile
  ``k``, in tile steps. It is linear, so every tile step costs the same.
- ``p(a, b)``: 0 for tiles sharing an edge, 0.2 for tiles sharing only a vertex,
  ``(h - 1)**2`` for tiles ``h`` steps apart otherwise (one missed contact
  between neighbors costs 1).
- ``q(a, b)``: 1 for tiles sharing an edge, else 0 (two regions that are not
  neighbors touch on the grid).
- ``K(t)``: number of connectivity defects. The occupied tiles of each group of
  geographically connected regions should form one connected set of their own,
  and the free tiles should have no component enclosed by occupied tiles beyond
  the number of geographic holes. ``K`` counts every extra piece of a group,
  every pair of groups merged into one piece, and every extra enclosed hole.

Unused tiles enter the energy only through the distance of a region placed on
them, so tiles far from every centroid are not chosen unless that saves
contacts or defects worth more than the distance.

The minimizer is a descent from a given assignment. It applies a move only when
the move lowers ``E``, so ``E`` decreases with every move and the descent ends.
The moves are: one region to a free tile, a swap of two regions, and, while
there are defects, the filling of a hole by shift chains and the move of a
detached piece next to the rest of its group.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

from ._placement import _bfs_distance_matrix, _canonical

_EPS = 1e-9
_N_CHAIN_ENDS = 6
_N_REPAIR = 20


def tile_step(grid_centers: NDArray[np.floating]) -> float:
    """Median distance from a tile center to the nearest other tile center."""
    if len(grid_centers) < 2:
        return 1.0
    dd = cdist(grid_centers, grid_centers)
    np.fill_diagonal(dd, np.inf)
    return float(np.median(dd.min(axis=1)))


class GridEnergy:
    """The assignment energy for one set of regions on one tile lattice.

    Parameters
    ----------
    centroids : (n, 2) array
        Region centroids.
    grid_centers : (m, 2) array
        Tile centers.
    adjacency : (n, n) array
        Region adjacency; positive entries are neighbors, weighted by value.
    tile_adjacency : (m, m) sparse matrix
        Tiles sharing an edge.
    vertex_adjacency : (m, m) sparse matrix, optional
        Tiles sharing only a vertex.
    distance_weight : float
        Energy of one tile step of distance between a region and its tile.
    contact_weight : float
        Energy of one missed contact: the factor of the pair terms.
    connectivity_weight : float
        Energy of one connectivity defect.
    allowed_holes : int
        Number of enclosed free components that are not defects (geographic
        holes of the input).

    """

    def __init__(
        self,
        centroids: NDArray[np.floating],
        grid_centers: NDArray[np.floating],
        adjacency: NDArray[np.floating],
        tile_adjacency: csr_array,
        vertex_adjacency: csr_array | None = None,
        *,
        distance_weight: float = 1.0,
        contact_weight: float = 1.0,
        connectivity_weight: float = 5.0,
        allowed_holes: int = 0,
    ) -> None:
        n, m = len(centroids), len(grid_centers)
        self.n, self.m = n, m
        step = tile_step(grid_centers)
        self.unary = distance_weight * cdist(centroids, grid_centers) / step

        edge = _canonical(tile_adjacency)
        edge = _canonical(edge.maximum(edge.T))
        edge_dense = edge.toarray() > 0
        hops = _bfs_distance_matrix(edge).astype(float)
        p = np.maximum(hops - 1.0, 0.0) ** 2
        if vertex_adjacency is not None:
            vert = _canonical(vertex_adjacency)
            vert = vert.maximum(vert.T).toarray() > 0
            p[vert] = 0.2
        p[edge_dense] = 0.0
        np.fill_diagonal(p, 0.0)
        self.p = p
        self.q = edge_dense.astype(float)
        np.fill_diagonal(self.q, 0.0)

        w = np.asarray(adjacency, dtype=float).copy()
        np.fill_diagonal(w, 0.0)
        self.n_groups, self.group = connected_components(csr_array(w > 0), directed=False)
        self._contact = contact_weight
        self._nb_ind_dense = w > 0
        self._nb_ind = csr_array(self._nb_ind_dense.astype(float))
        self._w_adj_sp = csr_array(contact_weight * np.where(w > 0, w, 0.0))
        self._pair_i, self._pair_j = np.nonzero(np.triu(w > 0, 1))
        self._pair_w = contact_weight * w[self._pair_i, self._pair_j]
        self.w_adj = contact_weight * np.where(w > 0, w, 0.0)
        self.w_non = contact_weight * (w <= 0).astype(float)
        np.fill_diagonal(self.w_non, 0.0)

        self.connectivity_weight = connectivity_weight
        self.allowed_holes = allowed_holes
        coo = edge.tocoo()
        self._rows, self._cols = coo.row, coo.col
        counts = np.diff(edge.indptr)
        self._boundary = counts < (counts.max() if m else 0)
        self._nbrs = [edge.indices[edge.indptr[x] : edge.indptr[x + 1]].tolist() for x in range(m)]

    # --- terms -------------------------------------------------------------------------

    def pair_field(self, t: NDArray[np.intp]) -> NDArray[np.floating]:
        """(n, m) pair energy of each region on each tile, with all other regions fixed."""
        qt = self.q[t]
        # w_non = c * (1 - I - [neighbors]), so its product needs no dense (n, n) matrix
        non = self._contact * (qt.sum(axis=0)[None, :] - qt - self._nb_ind @ qt)
        return self._w_adj_sp @ self.p[t] + non

    def _labels(self, occ: NDArray[np.bool_]) -> NDArray[np.intp]:
        """Component label of every tile; edges join two occupied or two free tiles."""
        keep = occ[self._rows] == occ[self._cols]
        g = csr_array(
            (np.ones(int(keep.sum())), (self._rows[keep], self._cols[keep])),
            shape=(self.m, self.m),
        )
        return connected_components(g, directed=False)[1]

    def defects(self, t: NDArray[np.intp]) -> int:
        """Number of connectivity defects ``K`` of an assignment."""
        occ = np.zeros(self.m, dtype=bool)
        occ[t] = True
        lab = self._labels(occ)
        occ_lab = lab[t]
        pieces = len(np.unique(occ_lab))
        incid = len(np.unique(self.group * self.m + occ_lab))
        free_lab = lab[~occ]
        holes = len(np.unique(free_lab)) - len(np.unique(lab[~occ & self._boundary]))
        return (2 * incid - self.n_groups - pieces) + max(0, holes - self.allowed_holes)

    def terms(self, t: NDArray[np.intp]) -> tuple[float, float, int]:
        """(distance energy, pair energy, defect count) of an assignment."""
        t = np.asarray(t)
        r = np.arange(self.n)
        dist = float(self.unary[r, t].sum())
        pair = float((self._pair_w * self.p[t[self._pair_i], t[self._pair_j]]).sum())
        owner = np.full(self.m, -1, dtype=np.intp)
        owner[t] = r
        ro, co = owner[self._rows], owner[self._cols]
        both = (ro >= 0) & (co >= 0) & (self._rows < self._cols)
        pair += self._contact * float((~self._nb_ind_dense[ro[both], co[both]]).sum())
        return dist, pair, self.defects(t)

    def energy(self, t: NDArray[np.intp]) -> float:
        """Total energy ``E`` of an assignment."""
        dist, pair, k = self.terms(t)
        return dist + pair + self.connectivity_weight * k

    # --- descent ------------------------------------------------------------------------

    def descend(
        self,
        t: NDArray[np.intp],
        *,
        max_sweeps: int = 100,
        n_candidates: int = 3,
        trace: list[float] | None = None,
    ) -> NDArray[np.intp]:
        """Lower the energy by moves until no move lowers it.

        Each sweep first removes defects: while there are defects, it applies
        the defect-removing move (see ``_best_repair``) that lowers ``E`` most.
        It then visits the regions in index order. For a region it ranks the
        moves to a free tile and the swaps with another region by their change
        of distance and pair energy, computes the exact change of ``E``
        (connectivity included) for the best *n_candidates* of each kind that
        lower those terms, and applies the one that lowers ``E`` most. The
        descent ends after a sweep without a move.

        Parameters
        ----------
        t : (n,) int array
            Starting assignment, one distinct tile per region.
        max_sweeps : int
            Upper limit on the number of sweeps.
        n_candidates : int
            Number of relocations and of swaps per region whose connectivity
            change is evaluated.
        trace : list of float, optional
            When given, the energy after the start and after every applied move
            is appended.

        Returns
        -------
        (n,) int array
            The assignment at which the descent stopped.

        """
        n, m = self.n, self.m
        t = np.asarray(t, dtype=np.intp).copy()
        owner = np.full(m, -1, dtype=np.intp)
        owner[t] = np.arange(n)
        field = self.pair_field(t)
        k_cur = self.defects(t)
        mu = self.connectivity_weight
        ar = np.arange(n)
        if trace is not None:
            trace.append(self.energy(t))

        def _move(r: int, old: int, new: int) -> None:
            nonlocal field
            wa, wn = self.w_adj[:, r], self.w_non[:, r]
            if wa.any():
                field += np.outer(wa, self.p[new] - self.p[old])
            field += np.outer(wn, self.q[new] - self.q[old])

        def _reset(new_t: NDArray[np.intp]) -> None:
            nonlocal t, field, k_cur
            t = new_t
            owner[:] = -1
            owner[t] = ar
            field = self.pair_field(t)
            k_cur = self.defects(t)
            if trace is not None:
                trace.append(self.energy(t))

        for _sweep in range(max_sweeps):
            # Repairs first: the cheapest defect-removing move, as long as one lowers E
            repaired = False
            while k_cur > 0:
                fixed = self._best_repair(t, owner, field, k_cur)
                if fixed is None:
                    break
                _reset(fixed)
                repaired = True
            improved = False
            for i in range(n):
                a = int(t[i])
                # Relocations of i to every free tile
                d_rel = self.unary[i] - self.unary[i, a] + field[i] - field[i, a]
                d_rel[owner >= 0] = np.inf
                # Swaps of i with every other region j
                tj = t
                d_swp = (
                    self.unary[i, tj]
                    - self.unary[i, a]
                    + self.unary[ar, a]
                    - self.unary[ar, tj]
                    + field[i, tj]
                    - field[i, a]
                    + field[ar, a]
                    - field[ar, tj]
                    + 2.0 * (self.w_adj[i] * self.p[a, tj] + self.w_non[i] * self.q[a, tj])
                )
                d_swp[i] = np.inf

                cands: list[tuple[str, int, float]] = []
                order_r = np.argsort(d_rel, kind="stable")[:n_candidates]
                cands += [("r", int(x), float(d_rel[x])) for x in order_r if d_rel[x] < -_EPS]
                order_s = np.argsort(d_swp, kind="stable")[:n_candidates]
                cands += [("s", int(x), float(d_swp[x])) for x in order_s if d_swp[x] < -_EPS]

                best: tuple[str, int, float, int] | None = None
                for kind, x, d in cands:
                    if best is not None and d - mu * k_cur >= best[2] - _EPS:
                        continue  # cannot beat the best even if it removes every defect
                    t2 = t.copy()
                    if kind == "r":
                        t2[i] = x
                    else:
                        t2[i], t2[x] = t[x], a
                    # a swap keeps the occupied tiles: no change with one group
                    k2 = k_cur if kind == "s" and self.n_groups == 1 else self.defects(t2)
                    total = d + mu * (k2 - k_cur)
                    if total < -_EPS and (best is None or total < best[2]):
                        best = (kind, x, total, k2)
                if best is None:
                    continue
                kind, x, _, k2 = best
                if kind == "r":
                    _move(i, a, x)
                    owner[a], owner[x] = -1, i
                    t[i] = x
                else:
                    b = int(t[x])
                    _move(i, a, b)
                    _move(x, b, a)
                    t[i], t[x] = b, a
                    owner[a], owner[b] = x, i
                k_cur = k2
                improved = True
                if trace is not None:
                    trace.append(self.energy(t))
            if not improved and not repaired:
                break
        return t

    def _best_repair(
        self, t: NDArray[np.intp], owner: NDArray[np.intp], field: NDArray[np.floating], k_cur: int
    ) -> NDArray[np.intp] | None:
        """The defect-removing move with the lowest energy, if it lowers ``E``.

        The moves are: one region to a free tile that touches two pieces or lies
        in a hole (the cheapest few by distance and pair energy), the filling of
        a hole by shift chains, and the move of a detached piece next to the
        largest piece of its group.
        """
        n = self.n
        mu = self.connectivity_weight
        ar = np.arange(n)
        repair = np.array(sorted(self._repair_tiles(owner)), dtype=np.intp)
        e0 = self.energy(t)
        best: tuple[float, NDArray[np.intp]] | None = None
        if len(repair):
            d = self.unary[:, repair] - self.unary[ar, t][:, None] + field[:, repair] - field[ar, t][:, None]
            flat = np.argsort(d, axis=None, kind="stable")[:_N_REPAIR]
            for f in flat.tolist():
                r, x = divmod(f, len(repair))
                if d[r, x] - mu * k_cur >= (np.inf if best is None else best[0] - e0) - _EPS:
                    break
                t2 = t.copy()
                t2[r] = repair[x]
                e2 = self.energy(t2)
                if e2 < e0 - _EPS and (best is None or e2 < best[0]):
                    best = (e2, t2)
        for move in (self._best_chain(t, owner), self._best_regroup(t, owner)):
            if move is not None:
                e2 = self.energy(move)
                if e2 < e0 - _EPS and (best is None or e2 < best[0]):
                    best = (e2, move)
        return None if best is None else best[1]

    def _best_chain(self, t: NDArray[np.intp], owner: NDArray[np.intp]) -> NDArray[np.intp] | None:
        """Best filling of one enclosed hole by shift chains, if it lowers ``E``.

        Each tile of the hole is filled in turn by a chain: a shortest path from
        the tile through occupied tiles to an occupied tile next to a free tile
        outside every hole or on the edge of the lattice; each region on the
        path moves one tile toward the hole. Of the first few such paths the one
        with the lowest energy is taken. The filling of every hole is tried.
        """
        from collections import deque

        occ = owner >= 0
        lab = self._labels(occ)
        open_labels = np.unique(lab[~occ & self._boundary])
        is_open = ~occ & np.isin(lab, open_labels)
        hole_labels = np.unique(lab[~occ & ~is_open])
        nbrs = self._nbrs
        e0 = self.energy(t)
        best: tuple[float, NDArray[np.intp]] | None = None
        for hl in hole_labels.tolist():
            t2, owner2, occ2, open2 = t.copy(), owner.copy(), occ.copy(), is_open.copy()
            e2 = e0
            for h in np.flatnonzero(lab == hl).tolist():
                if occ2[h] or open2[h]:
                    continue
                parent = {h: -1}
                queue = deque([h])
                ends: list[int] = []
                while queue and len(ends) < _N_CHAIN_ENDS:
                    cur = queue.popleft()
                    for nb in nbrs[cur]:
                        if nb in parent or not occ2[nb]:
                            continue
                        parent[nb] = cur
                        queue.append(nb)
                        if self._boundary[nb] or any(open2[x] for x in nbrs[nb]):
                            ends.append(nb)
                options = []
                for end in ends:
                    t3 = t2.copy()
                    node = end
                    while parent[node] != -1:
                        t3[owner2[node]] = parent[node]
                        node = parent[node]
                    options.append((self.energy(t3), end, t3))
                if not options:
                    e2 = np.inf
                    break
                e2, end, t2 = min(options, key=lambda o: (o[0], o[1]))
                owner2[:] = -1
                owner2[t2] = np.arange(self.n)
                occ2[h], occ2[end] = True, False
                # the vacated tile and the free tiles it reaches are outside every hole
                flood = deque([end])
                open2[end] = True
                while flood:
                    cur = flood.popleft()
                    for nb in nbrs[cur]:
                        if not occ2[nb] and not open2[nb]:
                            open2[nb] = True
                            flood.append(nb)
            if e2 < e0 - _EPS and (best is None or e2 < best[0]):
                best = (e2, t2)
        return None if best is None else best[1]

    def _best_regroup(self, t: NDArray[np.intp], owner: NDArray[np.intp]) -> NDArray[np.intp] | None:
        """Best move of all regions of a detached piece next to the largest piece of their group, if it lowers ``E``.

        The regions of the piece move one at a time, in index order, each to the
        free tile next to the largest piece that is closest to its centroid.
        """
        occ = owner >= 0
        lab = self._labels(occ)
        e0 = self.energy(t)
        best: tuple[float, NDArray[np.intp]] | None = None
        for grp in range(self.n_groups):
            members = np.flatnonzero(self.group == grp)
            labels, sizes = np.unique(lab[t[members]], return_counts=True)
            if len(labels) < 2:
                continue
            main = labels[np.argmax(sizes)]  # the first label among the largest
            for piece in labels:
                if piece == main:
                    continue
                t2 = t.copy()
                occ2 = occ.copy()
                body = occ & (lab == main)
                movers = members[lab[t[members]] == piece]
                for r in movers.tolist():
                    near = np.zeros(self.m, dtype=bool)
                    near[self._cols[body[self._rows]]] = True
                    near &= ~occ2
                    if not near.any():
                        break
                    cand = np.flatnonzero(near)
                    x = int(cand[np.argmin(self.unary[r, cand])])
                    occ2[t2[r]] = False
                    occ2[x] = True
                    body[x] = True
                    t2[r] = x
                else:
                    e2 = self.energy(t2)
                    if e2 < e0 - _EPS and (best is None or e2 < best[0]):
                        best = (e2, t2)
        return None if best is None else best[1]

    def _repair_tiles(self, owner: NDArray[np.intp]) -> set[int]:
        """Free tiles next to two or more occupied pieces, and free tiles of enclosed holes."""
        occ = owner >= 0
        lab = self._labels(occ)
        out: set[int] = set()
        open_labels = set(np.unique(lab[~occ & self._boundary]).tolist())
        for x in np.flatnonzero(~occ).tolist():
            if int(lab[x]) not in open_labels:
                out.add(x)
        # free tiles touching two distinct occupied pieces
        rows, cols = self._rows, self._cols
        sel = ~occ[rows] & occ[cols]
        pieces: dict[int, set[int]] = {}
        for r_, c_ in zip(rows[sel].tolist(), cols[sel].tolist(), strict=False):
            pieces.setdefault(r_, set()).add(int(lab[c_]))
        out.update(x for x, s in pieces.items() if len(s) > 1)
        return out
