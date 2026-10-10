"""Power diagrams clipped to a boundary, and the offset solve that fits their areas.

A power diagram assigns a point ``x`` to the generator ``i`` that minimizes
``|x - p_i|^2 - lambda_i``.  Every cell is the intersection of half-planes,
so it is a convex polygon with straight edges (before clipping to the
boundary), and adjacent cells share their edges exactly.  Raising
``lambda_i`` grows cell *i*.

For fixed generators and positive target areas that sum to the boundary
area, offsets that realize the targets exist and are unique up to a common
constant.  :meth:`ClippedPowerDiagram.fit` finds them with a damped Newton
method on the exact clipped cell areas: the derivative of area *i* with
respect to ``lambda_j`` is ``-L_ij / (2 |p_i - p_j|)``, where ``L_ij`` is the
length of the shared edge inside the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely as sh
from scipy.sparse import coo_matrix
from scipy.sparse import identity as sparse_identity
from scipy.sparse.linalg import spsolve
from scipy.spatial import ConvexHull, QhullError

from ._base import _keep_polygonal


def nonempty_offsets(points: np.ndarray, offsets: np.ndarray, max_passes: int = 5) -> np.ndarray:
    """Raise offsets until every generator lies inside its own power cell.

    A generator's cell becomes empty once its offset falls far enough below a
    neighbor's.  ``lambda_i >= max_j(lambda_j - |p_i - p_j|^2)`` keeps ``p_i``
    inside cell *i*; raising one offset raises other generators' floors, hence
    the (usually single-pass) repetition.
    """
    sq = np.einsum("ij,ij->i", points, points)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (points @ points.T)
    np.fill_diagonal(d2, np.inf)
    for _ in range(max_passes):
        floor = (offsets[None, :] - d2).max(axis=1)
        raised = np.maximum(offsets, floor)
        if np.array_equal(raised, offsets):
            break
        offsets = raised
    return offsets


@dataclass
class PowerFit:
    """Result of :meth:`ClippedPowerDiagram.fit`.

    Attributes
    ----------
    offsets : ndarray, shape (G,)
        Power offsets ``lambda_i`` in squared map units.
    cells : ndarray of shapely Geometry, shape (G,)
        Power cells clipped to the boundary.
    areas : ndarray, shape (G,)
        Cell areas.
    converged : bool
        ``True`` when every cell is within the relative tolerance of its target.
    """

    offsets: np.ndarray
    cells: np.ndarray
    areas: np.ndarray
    converged: bool


@dataclass
class _Diagram:
    raw: np.ndarray  # unclipped convex cells (empty Polygon for an empty cell)
    areas: np.ndarray  # areas inside the boundary
    edge_i: np.ndarray
    edge_j: np.ndarray
    edge_len: np.ndarray  # shared-edge lengths inside the boundary


class ClippedPowerDiagram:
    """Power diagrams of varying generators and offsets inside one fixed boundary.

    Areas and edge lengths inside the boundary are measured on a grid of
    boundary tiles, so a cell is only intersected with the few tiles it
    overlaps; cells and edges that lie inside the boundary are measured
    directly.
    """

    def __init__(self, boundary) -> None:
        self.boundary = boundary
        sh.prepare(boundary)
        minx, miny, maxx, maxy = boundary.bounds
        self._center = np.array([(minx + maxx) / 2.0, (miny + maxy) / 2.0])
        self._scale = max(maxx - minx, maxy - miny)
        k = max(4, int(np.ceil(np.sqrt(sh.get_num_coordinates(boundary) / 50.0))))
        xs = np.linspace(minx, maxx, k + 1)
        ys = np.linspace(miny, maxy, k + 1)
        tiles = [sh.clip_by_rect(boundary, xs[a], ys[b], xs[a + 1], ys[b + 1]) for a in range(k) for b in range(k)]
        self._tiles = np.array([t for t in tiles if not t.is_empty and t.area > 0.0], dtype=object)
        self._tree = sh.STRtree(self._tiles)

    # -- geometry ---------------------------------------------------------

    def _measure(self, geoms: np.ndarray, measure) -> np.ndarray:
        """Area (or length) of each geometry inside the boundary."""
        inside = sh.contains_properly(self.boundary, geoms)
        out = np.where(inside, measure(geoms), 0.0)
        idx = np.where(~inside)[0]
        if len(idx):
            gi, ti = self._tree.query(geoms[idx], predicate="intersects")
            parts = measure(sh.intersection(geoms[idx][gi], self._tiles[ti]))
            out[idx] += np.bincount(gi, weights=parts, minlength=len(idx))
        return out

    def _diagram(self, points: np.ndarray, offsets: np.ndarray) -> _Diagram:
        """Power diagram via the lower convex hull of the lifted generators."""
        n = len(points)
        s = self._scale
        p = (points - self._center) / s
        lam = offsets / (s * s)
        lam = lam - lam.mean()
        # Four far-away generators bound the outer cells.
        far = 10.0
        ghosts = np.array([[-far, -far], [far, -far], [far, far], [-far, far]])
        allp = np.vstack([p, ghosts])
        z = np.einsum("ij,ij->i", allp, allp) - np.concatenate([lam, np.zeros(4)])
        try:
            hull = ConvexHull(np.column_stack([allp, z]))
        except QhullError:
            hull = ConvexHull(np.column_stack([allp, z]), qhull_options="QJ")
        eq = hull.equations
        lower = eq[:, 2] < 0.0
        simplices = hull.simplices
        # Each lower facet is dual to a power-diagram vertex.
        vertices = -eq[:, :2] / (2.0 * eq[:, 2:3]) * s + self._center

        lower_idx = np.where(lower)[0]
        owner = simplices[lower_idx].ravel()
        facet = np.repeat(lower_idx, 3)
        order = np.argsort(owner, kind="stable")
        owner, facet = owner[order], facet[order]
        starts = np.searchsorted(owner, np.arange(n + 1))
        raw = np.empty(n, dtype=object)
        for i in range(n):
            f = facet[starts[i] : starts[i + 1]]
            if len(f) < 3:
                raw[i] = sh.Polygon()
                continue
            v = vertices[f]
            m = v.mean(axis=0)
            raw[i] = sh.Polygon(v[np.argsort(np.arctan2(v[:, 1] - m[1], v[:, 0] - m[0]))])

        # Edge between generators a and b: the segment joining the vertices of
        # the two lower facets that share the hull edge (a, b).
        nb = hull.neighbors[lower_idx]  # (F, 3): facet opposite each vertex
        f_idx = np.repeat(lower_idx, 3)
        g_idx = nb.ravel()
        k_idx = np.tile(np.arange(3), len(lower_idx))
        keep = lower[g_idx] & (g_idx > f_idx)
        f_idx, g_idx, k_idx = f_idx[keep], g_idx[keep], k_idx[keep]
        tri = simplices[f_idx]
        a = tri[np.arange(len(tri)), (k_idx + 1) % 3]
        b = tri[np.arange(len(tri)), (k_idx + 2) % 3]
        real = (a < n) & (b < n)
        a, b, f_idx, g_idx = a[real], b[real], f_idx[real], g_idx[real]
        if len(a):
            lines = sh.linestrings(np.stack([vertices[f_idx], vertices[g_idx]], axis=1))
            edge_len = self._measure(lines, sh.length)
        else:
            edge_len = np.zeros(0)
        return _Diagram(raw, self._measure(raw, sh.area), a, b, edge_len)

    def _clip(self, raw: np.ndarray) -> np.ndarray:
        inside = sh.contains_properly(self.boundary, raw)
        cells = raw.copy()
        need = np.where(~inside)[0]
        if len(need):
            cells[need] = [_keep_polygonal(c) for c in sh.intersection(raw[need], self.boundary)]
        return cells

    # -- solve ------------------------------------------------------------

    def _separate_duplicates(self, points: np.ndarray) -> np.ndarray:
        """Move coincident generators apart by a negligible distance.

        Two generators at the same position cannot be told apart by any
        offsets, so one of them would keep an empty cell.
        """
        _, counts = np.unique(points, axis=0, return_counts=True)
        if (counts == 1).all():
            return points
        points = points.copy()
        eps = 1e-9 * self._scale
        seen: dict[tuple[float, float], int] = {}
        for i, xy in enumerate(map(tuple, points)):
            k = seen.get(xy, 0)
            if k:
                points[i] += eps * k * np.array([np.cos(k), np.sin(k)])
            seen[xy] = k + 1
        return points

    def _newton_step(self, points: np.ndarray, d: _Diagram, residual: np.ndarray) -> np.ndarray:
        n = len(points)
        dist = np.linalg.norm(points[d.edge_i] - points[d.edge_j], axis=1)
        c = d.edge_len / (2.0 * np.maximum(dist, 1e-300))
        rows = np.concatenate([d.edge_i, d.edge_j, d.edge_i, d.edge_j])
        cols = np.concatenate([d.edge_j, d.edge_i, d.edge_i, d.edge_j])
        vals = np.concatenate([-c, -c, c, c])
        hess = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        diag = hess.diagonal()
        # A cell without an edge inside the boundary cannot change its area
        # by a small offset change; leave its offset alone.
        free = diag > 0.0
        step = np.zeros(n)
        if free.any():
            h = hess[free][:, free]
            # The Laplacian is singular (a common offset shift changes
            # nothing); a tiny ridge selects one solution.
            ridge = 1e-9 * float(diag[free].mean())
            step[free] = spsolve((h + ridge * sparse_identity(int(free.sum()))).tocsc(), residual[free])
        return step - step.mean()

    def fit(
        self,
        points: np.ndarray,
        targets: np.ndarray,
        offsets: np.ndarray | None = None,
        *,
        rtol: float = 0.01,
        max_iter: int = 50,
        want_cells: bool = True,
    ) -> PowerFit:
        """Solve the offsets whose clipped power cells have the target areas.

        Parameters
        ----------
        points : ndarray, shape (G, 2)
            Generators, held fixed.
        targets : ndarray, shape (G,)
            Positive target areas; they should sum to the boundary area.
        offsets : ndarray or None
            Starting offsets (warm start).  ``None`` starts from zero.
        rtol : float
            Stop once every cell is within this relative error of its target.
        max_iter : int
            Maximum number of Newton steps.

        Returns
        -------
        PowerFit
        """
        lam = np.zeros(len(points)) if offsets is None else np.asarray(offsets, dtype=float).copy()
        points = self._separate_duplicates(points)
        d = self._diagram(points, lam)
        # Newton needs every cell to have positive area.  Raise the offset of
        # each empty cell just enough that its generator lies inside a small
        # cell of its own (``lambda_i > lambda_j - |p_i - p_j|^2`` for all j);
        # the damped steps below then keep every area positive.
        # A generator just outside the boundary may need a larger margin before
        # its cell reaches into the boundary, hence the growing margin.
        margin = 0.01
        for _ in range(12):
            empty = np.where(d.areas <= 0.0)[0]
            if not len(empty):
                break
            d2 = np.sum((points[empty, None, :] - points[None, :, :]) ** 2, axis=2)
            d2[np.arange(len(empty)), empty] = np.inf
            lam[empty] = np.maximum(lam[empty], (lam[None, :] - d2).max(axis=1) + margin * d2.min(axis=1))
            d = self._diagram(points, lam)
            margin *= 2.0
        converged = False
        for _ in range(max_iter + 1):
            residual = targets - d.areas
            if np.max(np.abs(residual) / targets) <= rtol:
                converged = True
                break
            if _ == max_iter:
                break
            step = self._newton_step(points, d, residual)
            # Damped Newton: halve the step until no non-empty cell shrinks
            # below half of the smallest current or target area and the
            # residual drops.
            alive = d.areas > 0.0
            floor = 0.5 * min(float(targets.min()), float(d.areas[alive].min()))
            norm = float(np.linalg.norm(residual[alive]))
            t = 1.0
            accepted = None
            while t > 1e-4:
                trial = self._diagram(points, lam + t * step)
                if (
                    trial.areas[alive].min() > floor
                    and np.linalg.norm((targets - trial.areas)[alive]) <= (1.0 - t / 2.0) * norm
                ):
                    accepted = trial
                    break
                t *= 0.5
            if accepted is None:
                break
            lam = lam + t * step
            d = accepted
        cells = self._clip(d.raw) if want_cells else np.empty(0, dtype=object)
        return PowerFit(offsets=lam, cells=cells, areas=d.areas, converged=converged)
