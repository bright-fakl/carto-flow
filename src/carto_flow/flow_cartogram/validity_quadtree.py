"""
Quadtree windows: output polygons that do not self-intersect, from a piecewise-affine map inside each window.

A window is the span of iterations between two recomputes of the velocity field. At the start of a window a graded
quadtree is built from the density (:mod:`carto_flow.flow_cartogram.quadtree_mesh`), every leaf is triangulated as a
fan, and vertices are inserted where the boundaries of the regions cross an edge of the mesh, so that every segment lies
inside one triangle. During the window only the nodes of the mesh are advected through the velocity field; a vertex
keeps its barycentric coordinates in its triangle and moves with the affine map of the triangle. A segment inside one
triangle stays a straight segment inside the image of the triangle, and the map is injective as long as no triangle
inverts, so no two segments cross during the window.

The mesh is kept for the following windows and rebuilt when a fan inverted and was repaired, or when the area of a
triangle fell below a fraction of its area at construction.

A leaf that is squeezed into a sliver with a reflex corner has its center outside its kernel, and its fan inverts. The
center of such a leaf is moved to the centroid of the kernel. The nodes of a leaf that stays inverted are limited: they
move a half, a quarter, ... and finally none of the step. An inversion that survives is counted
(:attr:`QuadtreeWindows.n_unresolved`) and reported when the run ends.

At each rebuild, inserted vertices that are no longer needed are removed. Only inserted vertices are candidates; a
vertex is a candidate if it lies within a fraction of a cell of the chord of its two neighbors and its coordinate hash
is larger than the hash of both neighbors (so no two adjacent vertices are removed, and the two rings of a shared
boundary decide alike). A candidate is removed unless its new chord crosses a segment or another chord, or a foreign
vertex lies within the tolerance of the chord. The foreign-vertex test (it catches a small ring that lies in the sliver
between the path and the chord, which no crossing test sees) is restricted to rings of at most ``max_ring`` vertices.
That limit is a heuristic: a ring with more vertices that lies entirely inside a sliver narrower than the tolerance
would be missed.

Every insertion and every removal is recorded as a round of the :class:`~carto_flow.flow_cartogram.validity.VertexLineage`.
The best iterate of the run keeps its own vertex set: when the run returns it, the vertex set and the lineage are reset
to the state of that iterate.

Classes
-------
QuadtreeSettings
    The constants of the method.
QuadtreeWindows
    The state of the method, owned by one call of the morph loop.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from ..geo_utils.crossings import SegmentGrid, _cell, _cross
from .displacement import displace_coords_numba
from .quadtree_mesh import QuadtreeMesh, apply_map, build_quadtree
from .validity import InsertionRound, RemovalRound, ValidityMethod, VertexLineage

if TYPE_CHECKING:
    from ..geo_utils.geometry import GeometryCoordinateInfo
    from .grid import Grid

__all__ = ["QuadtreeSettings", "QuadtreeWindows"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuadtreeSettings:
    """
    The constants of the method.

    Attributes
    ----------
    leaf_fraction : float
        Number of quadtree leaves as a fraction of the number of grid cells. More leaves follow the flow more
        closely and cost vertices and time.
    prune_tolerance : float
        An inserted vertex is removed only if it lies within this many cells of the chord of its neighbors; the same
        distance is the clearance required from foreign vertices.
    area_ratio : float
        The mesh is rebuilt when the area of a triangle falls below this fraction of its area at construction. A
        small ratio keeps the mesh longer, at the price of distorted triangles, which the repair of inverted fans
        handles; a large ratio rebuilds more often.
    max_ring : int
        The foreign-vertex test of the pruning looks at rings with at most this many vertices.
    limiter_rounds : int
        Rounds of the limiter for the nodes of a leaf whose fan stays inverted: the nodes move 1/2, 1/4, ... of
        the step in the rounds, and none of it in the last.
    """

    leaf_fraction: float = 2.0**-6
    prune_tolerance: float = 0.02
    area_ratio: float = 0.001
    max_ring: int = 64
    limiter_rounds: int = 6


_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)


def _splitmix(z: np.ndarray) -> np.ndarray:
    z = (z ^ (z >> np.uint64(30))) * _MIX1
    z = (z ^ (z >> np.uint64(27))) * _MIX2
    return z ^ (z >> np.uint64(31))


def _coordinate_hash(xy: np.ndarray) -> np.ndarray:
    """Hash of the bit pattern of each coordinate pair; equal coordinates have equal hashes."""
    bits = np.ascontiguousarray(xy + 0.0).view(np.uint64)  # + 0.0: -0.0 and 0.0 are the same coordinate
    return _splitmix(_splitmix(bits[:, 0]) + _GOLDEN * bits[:, 1])


@njit(cache=True, parallel=True)
def _candidates(c, interior, is_input, h, tol):
    """Inserted vertices within tol of the chord of their neighbors that are local maxima of the hash."""
    n = c.shape[0]
    out = np.zeros(n, np.bool_)
    for i in prange(1, n - 1):
        if not interior[i] or is_input[i]:
            continue
        ux = c[i - 1, 0]
        uy = c[i - 1, 1]
        lx = c[i + 1, 0] - ux
        ly = c[i + 1, 1] - uy
        length2 = lx * lx + ly * ly
        if length2 < 1e-300:
            continue
        dev = abs((c[i, 0] - ux) * ly - (c[i, 1] - uy) * lx) / np.sqrt(length2)
        along = ((c[i, 0] - ux) * lx + (c[i, 1] - uy) * ly) / length2
        if dev < tol and 0.0 < along < 1.0 and h[i] > h[i - 1] and h[i] > h[i + 1]:
            out[i] = True
    return out


@njit(cache=True, inline="always")
def _point_segment_distance2(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    length2 = dx * dx + dy * dy
    t = 0.0
    if length2 > 0.0:
        t = min(max(((px - ax) * dx + (py - ay) * dy) / length2, 0.0), 1.0)
    qx = ax + t * dx - px
    qy = ay + t * dy - py
    return qx * qx + qy * qy


@njit(cache=True, inline="always")
def _same(c, i, x, y):
    return c[i, 0] == x and c[i, 1] == y


@njit(cache=True, parallel=True)
def _blocked_chords(c, starts, cnt, items, h, x0, y0, nx, ny, ring_size, candidates, tol, max_ring):
    """Candidates whose chord crosses a segment or has a foreign vertex within tol.

    The chord of candidate i joins the vertices i - 1 and i + 1. The own path (the segments between i - 1 and i + 1)
    and the path of the twin ring of a shared boundary have vertices equal to those of the run i - 1, i, i + 1 and are
    skipped. Segments that touch an end point of the chord are tested for crossings only.
    """
    blocked = np.zeros(candidates.shape[0], np.bool_)
    tol2 = tol * tol
    for k in prange(candidates.shape[0]):
        i = candidates[k]
        a = i - 1
        b = i + 1
        ax = c[a, 0]
        ay = c[a, 1]
        bx = c[b, 0]
        by = c[b, 1]
        mx = c[i, 0]
        my = c[i, 1]
        lo_x = min(ax, bx) - tol
        hi_x = max(ax, bx) + tol
        lo_y = min(ay, by) - tol
        hi_y = max(ay, by) + tol
        i0 = _cell(lo_x, x0, h, nx)
        i1 = _cell(hi_x, x0, h, nx)
        j0 = _cell(lo_y, y0, h, ny)
        j1 = _cell(hi_y, y0, h, ny)
        for cj in range(j0, j1 + 1):
            for ci in range(i0, i1 + 1):
                cell = cj * nx + ci
                for u in range(cnt[cell], cnt[cell + 1]):
                    s = items[u]
                    p = starts[s]
                    q = p + 1
                    if max(c[p, 0], c[q, 0]) < lo_x or min(c[p, 0], c[q, 0]) > hi_x:
                        continue
                    if max(c[p, 1], c[q, 1]) < lo_y or min(c[p, 1], c[q, 1]) > hi_y:
                        continue
                    p_on_run = _same(c, p, ax, ay) or _same(c, p, mx, my) or _same(c, p, bx, by)
                    q_on_run = _same(c, q, ax, ay) or _same(c, q, mx, my) or _same(c, q, bx, by)
                    if p_on_run and q_on_run:
                        continue
                    if _cross(c, a, b, p, q):
                        blocked[k] = True
                        break
                    touches = _same(c, p, ax, ay) or _same(c, p, bx, by) or _same(c, q, ax, ay) or _same(c, q, bx, by)
                    if touches or ring_size[s] > max_ring:
                        continue
                    if not p_on_run and _point_segment_distance2(c[p, 0], c[p, 1], ax, ay, bx, by) < tol2:
                        blocked[k] = True
                        break
                    if not q_on_run and _point_segment_distance2(c[q, 0], c[q, 1], ax, ay, bx, by) < tol2:
                        blocked[k] = True
                        break
                if blocked[k]:
                    break
            if blocked[k]:
                break
    return blocked


def _prune_mask(coords: np.ndarray, offsets: np.ndarray, is_input: np.ndarray, tol: float, max_ring: int):
    """Mask of the vertices to keep, or None if every vertex is kept."""
    n = len(coords)
    interior = np.ones(n, np.bool_)
    interior[offsets[:-1]] = False
    interior[offsets[1:] - 1] = False
    candidates = np.flatnonzero(_candidates(coords, interior, is_input, _coordinate_hash(coords), tol))
    if len(candidates) == 0:
        return None
    sizes = np.diff(offsets)
    grid = SegmentGrid(coords, offsets)
    ring_size = np.repeat(sizes, np.maximum(sizes - 1, 0))
    blocked = _blocked_chords(*grid.arrays(), ring_size, candidates, tol, max_ring)
    # the chords against each other
    chord_ends = np.empty((2 * len(candidates), 2))
    chord_ends[0::2] = coords[candidates - 1]
    chord_ends[1::2] = coords[candidates + 1]
    blocked |= SegmentGrid(chord_ends, starts=2 * np.arange(len(candidates))).crossing_mask()
    removed = candidates[~blocked]
    if len(removed) == 0:
        return None
    keep = np.ones(n, np.bool_)
    keep[removed] = False
    return keep


def _required(value):
    """The value of a state that the hooks of the loop set before it is used."""
    if value is None:
        raise RuntimeError("the hooks of the morph loop were called out of order")
    return value


class QuadtreeWindows(ValidityMethod):
    """
    The validity method of a morph with ``validity="quadtree"``.

    One instance belongs to one call of the morph loop (one level of a ladder). It holds the mesh of the current
    window and the per-vertex triangle and barycentric coordinates (36 bytes per vertex), the flags of the input
    vertices and the lineage. The geometry coordinates are displaced by the instance (:meth:`displace`) instead of
    by the loop.

    Parameters
    ----------
    n_vertices : int
        Number of vertices of the input geometries.
    settings : QuadtreeSettings, optional
        The constants of the method.

    Attributes
    ----------
    n_rebuilds : int
        Number of meshes built.
    n_fixed : int
        Iterations in which a fan inverted with the center at the mean of the corners.
    n_limited : int
        Iterations in which the nodes of a leaf had to be limited.
    n_unresolved : int
        Iterations that ended with an inverted triangle. The guarantee does not hold for the window of such an
        iteration; the output may contain invalid polygons (see ``Cartogram.invalid_regions``).
    n_pruned : int
        Number of inserted vertices removed.
    n_inserted : int
        Number of vertices inserted.
    """

    displaces_geometry = True

    def __init__(self, n_vertices: int, settings: QuadtreeSettings | None = None) -> None:
        self.settings = QuadtreeSettings() if settings is None else settings
        self._n_input = n_vertices
        self._events: list[InsertionRound | RemovalRound] = []
        self._is_input: np.ndarray | None = None
        self._geoms: GeometryCoordinateInfo | None = None
        self._grid: Grid | None = None
        self._iteration = 0
        self._mesh: QuadtreeMesh | None = None
        self._nodes = np.empty((0, 2))
        self._area0 = np.empty(0)
        self._tri = np.empty((0, 3), np.int32)
        self._bary = np.empty((0, 3))
        self._dirty = False
        self._best: tuple[np.ndarray, int] | None = None
        self.n_rebuilds = 0
        self.n_fixed = 0
        self.n_limited = 0
        self.n_unresolved = 0
        self.n_unresolved_leaves = 0
        self.n_pruned = 0
        self.n_inserted = 0

    # ------------------------------------------------------------------
    # Hooks of the morph loop
    # ------------------------------------------------------------------

    def begin_window(
        self, geoms: GeometryCoordinateInfo, grid: Grid, best_state: tuple | None, iteration: int
    ) -> tuple | None:
        """Start of a window: remember the geometry; the mesh is built when the density is known."""
        self._geoms = geoms
        self._grid = grid
        self._iteration = iteration
        return best_state

    def set_density(self, rho: np.ndarray) -> None:
        """Build the mesh of the window from the density, unless the mesh of the last window can be kept."""
        if self._mesh is not None and not self._dirty and self._min_area_ratio() >= self.settings.area_ratio:
            return
        self._rebuild(rho)

    def displace(self, grid: Grid, vx: np.ndarray, vy: np.ndarray, dt: float) -> np.ndarray:
        """Advect the nodes of the mesh one step and return the new coordinates of the vertices."""
        mesh = _required(self._mesh)
        previous = self._nodes
        nodes = displace_coords_numba(
            previous, grid.x_coords, grid.y_coords, vx, vy, dt, grid.dx, grid.dy, use_parallel=False
        )
        mesh.place_centers(nodes)
        inverted = mesh.inverted_leaves(nodes)
        if len(inverted):
            self._dirty = True  # the mesh is rebuilt at the next window
            self.n_fixed += 1
            unresolved = inverted[~mesh.fix_centers(nodes, inverted)]
            if len(unresolved):
                self.n_limited += 1
                nodes = self._limit(previous, nodes)
                inverted = mesh.inverted_leaves(nodes)
                unresolved = inverted[~mesh.fix_centers(nodes, inverted)] if len(inverted) else inverted
                if len(unresolved):
                    self.n_unresolved += 1
                    self.n_unresolved_leaves += len(unresolved)
                    logger.debug(
                        "iteration %d: %d leaves keep an inverted fan triangle", self._iteration, len(unresolved)
                    )
        self._nodes = nodes
        return apply_map(self._tri, self._bary, nodes)

    def mark_best(self) -> None:
        """The state after the last step is the best iterate: remember its vertex set."""
        self._best = (np.array(_required(self._geoms).ragged_offsets[0], dtype=np.int64), len(self._events))

    def restore_best(self, geoms: GeometryCoordinateInfo, coords: np.ndarray) -> None:
        """Reset the vertex set and the lineage to the state of the best iterate, whose coordinates are ``coords``."""
        if self._best is None:
            return
        offsets, n_events = self._best
        del self._events[n_events:]
        if not np.array_equal(offsets, np.asarray(geoms.ragged_offsets[0])) or len(coords) != len(geoms.coords):  # type: ignore[index]
            geoms.set_vertices(coords, offsets)

    def finish(
        self, geoms: GeometryCoordinateInfo, grid: Grid, best_state: tuple | None, iteration: int
    ) -> tuple | None:
        """End of the run: report inversions that the repair could not remove."""
        if self.n_unresolved:
            warnings.warn(
                f"validity='quadtree': {self.n_unresolved} iterations ended with an inverted mesh triangle that "
                f"could not be repaired; the output may contain invalid polygons.",
                UserWarning,
                stacklevel=2,
            )
        return best_state

    def lineage(self) -> VertexLineage:
        """The insertions and removals of the run."""
        return VertexLineage(self._n_input, tuple(self._events))

    # ------------------------------------------------------------------
    # The mesh
    # ------------------------------------------------------------------

    def _min_area_ratio(self) -> float:
        """Smallest ratio of the area of a triangle to its area at construction (non-positive if one inverted)."""
        return float((_required(self._mesh).triangle_areas(self._nodes) / self._area0).min())

    def _rebuild(self, rho: np.ndarray) -> None:
        """Build the mesh from the density, prune and insert vertices, and locate the vertices in the triangles."""
        geoms, grid, s = _required(self._geoms), _required(self._grid), self.settings
        frame = (float(grid.x_coords[0] - grid.dx / 2), float(grid.y_coords[0] - grid.dy / 2), grid.dx, grid.dy)
        mesh = build_quadtree(rho, frame, s.leaf_fraction)
        coords = geoms.coords
        offsets = np.asarray(geoms.ragged_offsets[0], dtype=np.int64)
        if self._is_input is None:
            self._is_input = np.ones(len(coords), np.bool_)
        is_input = self._is_input
        changed = False

        keep = _prune_mask(coords, offsets, is_input, s.prune_tolerance * min(grid.dx, grid.dy), s.max_ring)
        if keep is not None:
            kept = np.flatnonzero(keep)
            ring = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
            offsets = np.concatenate([[0], np.cumsum(np.bincount(ring[kept], minlength=len(offsets) - 1))])
            self.n_pruned += len(coords) - len(kept)
            coords = coords[kept]
            is_input = is_input[kept]
            self._events.append(RemovalRound(kept, self._iteration))
            changed = True

        inserted = mesh.insert(coords, offsets)
        if inserted is not None:
            grown = np.zeros(len(inserted.coords), np.bool_)
            old = np.ones(len(grown), np.bool_)
            old[inserted.new_index] = False
            grown[old] = is_input
            is_input = grown
            coords = inserted.coords
            offsets = inserted.ring_offsets
            self.n_inserted += len(inserted.new_index)
            self._events.append(InsertionRound(inserted.new_index, inserted.left, inserted.fraction, self._iteration))
            changed = True

        if changed:
            geoms.set_vertices(coords, offsets)
            self._is_input = is_input
        self._tri, self._bary = mesh.locate(geoms.coords)
        self._nodes = mesh.node_xy()
        self._area0 = mesh.triangle_areas(self._nodes)
        self._mesh = mesh
        self._dirty = False
        self.n_rebuilds += 1

    def _unresolved(self, nodes: np.ndarray) -> np.ndarray:
        """Leaves whose fan stays inverted at these node positions, without changing them."""
        mesh = _required(self._mesh)
        inverted = mesh.inverted_leaves(nodes)
        if len(inverted) == 0:
            return inverted
        return inverted[~mesh.fix_centers(nodes.copy(), inverted)]

    def _limit(self, previous: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Node positions after a step whose nodes in unresolved leaves are limited to a fraction of the step."""
        mesh = _required(self._mesh)
        step = target - previous
        limited = np.zeros(len(previous), np.bool_)
        current = target
        rounds = self.settings.limiter_rounds
        for k in range(rounds):
            leaves = self._unresolved(current)
            if len(leaves) == 0:
                break
            limited[mesh.leaf_nodes(leaves)] = True
            fraction = 0.5 ** (k + 1) if k < rounds - 1 else 0.0
            current = previous + np.where(limited[:, None], fraction, 1.0) * step
            mesh.place_centers(current)
        return current
