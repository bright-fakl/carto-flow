"""
Detection of crossing segments in a set of rings with a uniform grid (Numba).

The input is a flat coordinate array and the start of every ring in it. The segments are the consecutive vertex pairs
inside a ring (rings are closed lists: first vertex equal to last). Two segments cross if the orientations of the
end points of each against the other are strictly opposite. Segments that share an end point, identical segments of
two rings (a shared boundary) and touching or collinear segments have an orientation of exactly zero and are not
reported. All rings are tested against each other, so overlapping neighbors are found as well as self-intersections.

Functions
---------
find_crossing_segments
    Boolean mask of the segments that cross another segment.
find_crossing_pairs
    The pairs of crossing segments.
segment_starts
    Index of the first vertex of every segment.

Classes
-------
SegmentGrid
    Uniform grid of segments, with crossing test, neighborhood and identical-segment queries.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

__all__ = ["SegmentGrid", "find_crossing_pairs", "find_crossing_segments", "segment_starts"]


@njit(cache=True, inline="always")
def _orient(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


@njit(cache=True)
def _cross(c, a, b, p, q):
    ax, ay, bx, by = c[a, 0], c[a, 1], c[b, 0], c[b, 1]
    px, py, qx, qy = c[p, 0], c[p, 1], c[q, 0], c[q, 1]
    if not (_orient(ax, ay, bx, by, px, py) * _orient(ax, ay, bx, by, qx, qy) < 0):
        return False
    return _orient(px, py, qx, qy, ax, ay) * _orient(px, py, qx, qy, bx, by) < 0


@njit(cache=True, inline="always")
def _cell(x, x0, h, n):
    return max(0, min(int((x - x0) / h), n - 1))


@njit(cache=True)
def _segment_starts(offsets):
    n = 0
    for r in range(offsets.shape[0] - 1):
        n += max(offsets[r + 1] - offsets[r] - 1, 0)
    first = np.empty(n, np.int64)
    k = 0
    for r in range(offsets.shape[0] - 1):
        for i in range(offsets[r], offsets[r + 1] - 1):
            first[k] = i
            k += 1
    return first


@njit(cache=True)
def _grid_params(c, first, factor, max_cells_per_segment):
    n = first.shape[0]
    mean = 0.0
    xmin = 1e300
    ymin = 1e300
    xmax = -1e300
    ymax = -1e300
    for s in range(n):
        a = first[s]
        mean += np.hypot(c[a + 1, 0] - c[a, 0], c[a + 1, 1] - c[a, 1])
        xmin = min(xmin, c[a, 0], c[a + 1, 0])
        xmax = max(xmax, c[a, 0], c[a + 1, 0])
        ymin = min(ymin, c[a, 1], c[a + 1, 1])
        ymax = max(ymax, c[a, 1], c[a + 1, 1])
    mean /= max(n, 1)
    h = max(mean * factor, 1e-12)
    x0 = xmin - h
    y0 = ymin - h
    nx = int((xmax - x0) / h) + 2
    ny = int((ymax - y0) / h) + 2
    while nx * ny > max_cells_per_segment * n + 1000:
        h *= 1.25
        nx = int((xmax - x0) / h) + 2
        ny = int((ymax - y0) / h) + 2
    return h, x0, y0, nx, ny


@njit(cache=True)
def _build_grid(c, first, h, x0, y0, nx, ny):
    """Segments per cell in compressed form: segments of cell k are items[cnt[k]:cnt[k + 1]]."""
    n = first.shape[0]
    cnt = np.zeros(nx * ny + 1, np.int64)
    for s in range(n):
        a = first[s]
        b = a + 1
        i0 = _cell(min(c[a, 0], c[b, 0]), x0, h, nx)
        i1 = _cell(max(c[a, 0], c[b, 0]), x0, h, nx)
        j0 = _cell(min(c[a, 1], c[b, 1]), y0, h, ny)
        j1 = _cell(max(c[a, 1], c[b, 1]), y0, h, ny)
        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                cnt[j * nx + i + 1] += 1
    for k in range(nx * ny):
        cnt[k + 1] += cnt[k]
    items = np.empty(cnt[nx * ny], np.int64)
    pos = cnt[:-1].copy()
    for s in range(n):
        a = first[s]
        b = a + 1
        i0 = _cell(min(c[a, 0], c[b, 0]), x0, h, nx)
        i1 = _cell(max(c[a, 0], c[b, 0]), x0, h, nx)
        j0 = _cell(min(c[a, 1], c[b, 1]), y0, h, ny)
        j1 = _cell(max(c[a, 1], c[b, 1]), y0, h, ny)
        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                items[pos[j * nx + i]] = s
                pos[j * nx + i] += 1
    return cnt, items


@njit(cache=True)
def _owns_pair(c, a, b, p, q, ci, cj, x0, y0, h, nx, ny):
    """A pair is tested only in the cell that holds the min corner of the intersection of the two boxes."""
    mx = max(min(c[a, 0], c[b, 0]), min(c[p, 0], c[q, 0]))
    my = max(min(c[a, 1], c[b, 1]), min(c[p, 1], c[q, 1]))
    return _cell(mx, x0, h, nx) == ci and _cell(my, y0, h, ny) == cj


@njit(cache=True, parallel=True)
def _scan_mask(c, first, cnt, items, h, x0, y0, nx, ny, flag):
    for cell in prange(nx * ny):
        lo = cnt[cell]
        hi = cnt[cell + 1]
        if hi - lo < 2:
            continue
        ci = cell % nx
        cj = cell // nx
        for u in range(lo, hi):
            a = first[items[u]]
            for w in range(u + 1, hi):
                p = first[items[w]]
                if _owns_pair(c, a, a + 1, p, p + 1, ci, cj, x0, y0, h, nx, ny) and _cross(c, a, a + 1, p, p + 1):
                    flag[items[u]] = True
                    flag[items[w]] = True


@njit(cache=True)
def _scan_pairs(c, first, cnt, items, h, x0, y0, nx, ny, out):
    """Number of crossing pairs; the first out.shape[0] of them are written to out."""
    k = 0
    for cell in range(nx * ny):
        lo = cnt[cell]
        hi = cnt[cell + 1]
        ci = cell % nx
        cj = cell // nx
        for u in range(lo, hi):
            a = first[items[u]]
            for w in range(u + 1, hi):
                p = first[items[w]]
                if _owns_pair(c, a, a + 1, p, p + 1, ci, cj, x0, y0, h, nx, ny) and _cross(c, a, a + 1, p, p + 1):
                    if k < out.shape[0]:
                        out[k, 0] = min(items[u], items[w])
                        out[k, 1] = max(items[u], items[w])
                    k += 1
    return k


@njit(cache=True, parallel=True)
def _identical(c, first, cnt, items, h, x0, y0, nx, ny, flag, out):
    for s in prange(flag.shape[0]):
        if not flag[s]:
            continue
        a = first[s]
        b = a + 1
        out[s] = True
        # identical segments cover the cell of the first end point too
        cell = _cell(min(c[a, 1], c[b, 1]), y0, h, ny) * nx + _cell(min(c[a, 0], c[b, 0]), x0, h, nx)
        for u in range(cnt[cell], cnt[cell + 1]):
            p = first[items[u]]
            q = p + 1
            same = c[a, 0] == c[p, 0] and c[a, 1] == c[p, 1] and c[b, 0] == c[q, 0] and c[b, 1] == c[q, 1]
            swapped = c[a, 0] == c[q, 0] and c[a, 1] == c[q, 1] and c[b, 0] == c[p, 0] and c[b, 1] == c[p, 1]
            if same or swapped:
                out[items[u]] = True


@njit(cache=True)
def _neighborhood(c, first, cnt, items, h, x0, y0, nx, ny, flag, margin):
    marked = np.zeros(nx * ny, np.bool_)
    for s in range(flag.shape[0]):
        if not flag[s]:
            continue
        a = first[s]
        b = a + 1
        i0 = max(_cell(min(c[a, 0], c[b, 0]), x0, h, nx) - margin, 0)
        i1 = min(_cell(max(c[a, 0], c[b, 0]), x0, h, nx) + margin, nx - 1)
        j0 = max(_cell(min(c[a, 1], c[b, 1]), y0, h, ny) - margin, 0)
        j1 = min(_cell(max(c[a, 1], c[b, 1]), y0, h, ny) + margin, ny - 1)
        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                marked[j * nx + i] = True
    near = np.zeros(flag.shape[0], np.bool_)
    for cell in range(nx * ny):
        if marked[cell]:
            for u in range(cnt[cell], cnt[cell + 1]):
                near[items[u]] = True
    return near


def segment_starts(ring_offsets: np.ndarray) -> np.ndarray:
    """
    Index of the first vertex of every segment.

    The segments of a ring are its consecutive vertex pairs, in ring order. A segment ``k`` joins the vertices
    ``starts[k]`` and ``starts[k] + 1``.

    Parameters
    ----------
    ring_offsets : np.ndarray
        Start of every ring in the coordinate array and the total length, shape (n_rings + 1,).

    Returns
    -------
    np.ndarray
        Vertex indices, shape (n_segments,), dtype int64.
    """
    return _segment_starts(np.ascontiguousarray(ring_offsets, dtype=np.int64))


class SegmentGrid:
    """
    Uniform grid of segments for crossing queries.

    Build it once for a set of coordinates and query the crossing segments, their neighborhood and the segments that
    coincide with given ones. The segment indices of all results refer to ``starts``.

    Parameters
    ----------
    coords : np.ndarray
        Vertex coordinates, shape (n_points, 2).
    ring_offsets : np.ndarray, optional
        Start of every ring and the total length. The segments are the consecutive vertex pairs of the rings.
    starts : np.ndarray, optional
        Instead of ``ring_offsets``: the index of the first vertex of every segment, to grid a subset of the
        segments.
    cell_factor : float, default=6.0
        Cell size as a multiple of the mean segment length.

    Raises
    ------
    ValueError
        If neither ``ring_offsets`` nor ``starts`` is given, or the coordinates are not finite.
    """

    def __init__(
        self,
        coords: np.ndarray,
        ring_offsets: np.ndarray | None = None,
        *,
        starts: np.ndarray | None = None,
        cell_factor: float = 6.0,
    ) -> None:
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        if not np.isfinite(coords).all():
            raise ValueError("the coordinates must be finite")
        if starts is None:
            if ring_offsets is None:
                raise ValueError("give ring_offsets or starts")
            starts = segment_starts(ring_offsets)
        self.coords = coords
        self.starts = np.ascontiguousarray(starts, dtype=np.int64)
        self._params = _grid_params(coords, self.starts, cell_factor, 4)
        self._cells = _build_grid(coords, self.starts, *self._params)

    def _args(self):
        return (self.coords, self.starts, *self._cells, *self._params)

    def crossing_mask(self) -> np.ndarray:
        """Boolean mask over the segments: True for a segment that crosses another one."""
        flag = np.zeros(len(self.starts), np.bool_)
        _scan_mask(*self._args(), flag)
        return flag

    def crossing_pairs(self) -> np.ndarray:
        """Crossing pairs of segments as an array of shape (k, 2), the lower segment index first, sorted."""
        empty = np.empty((0, 2), np.int64)
        count = _scan_pairs(*self._args(), empty)
        out = np.empty((count, 2), np.int64)
        _scan_pairs(*self._args(), out)
        return out[np.lexsort((out[:, 1], out[:, 0]))]

    def neighborhood(self, mask: np.ndarray, margin: int = 1) -> np.ndarray:
        """Segments that share a grid cell, or a cell within ``margin`` cells, with a segment of ``mask``."""
        return _neighborhood(*self._args(), np.ascontiguousarray(mask, dtype=np.bool_), margin)

    def with_identical(self, mask: np.ndarray) -> np.ndarray:
        """``mask`` plus every segment with the same two end points (in either order) as a segment of ``mask``."""
        out = np.zeros(len(self.starts), np.bool_)
        _identical(*self._args(), np.ascontiguousarray(mask, dtype=np.bool_), out)
        return out


def find_crossing_segments(coords: np.ndarray, ring_offsets: np.ndarray) -> np.ndarray:
    """
    Mask of the segments that cross another segment.

    Parameters
    ----------
    coords : np.ndarray
        Vertex coordinates of all rings, shape (n_points, 2).
    ring_offsets : np.ndarray
        Start of every ring in ``coords`` and the total length, shape (n_rings + 1,). Rings are closed.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (n_segments,), indexed like :func:`segment_starts`.

    Examples
    --------
    >>> import numpy as np
    >>> from carto_flow.geo_utils import find_crossing_segments
    >>> bowtie = np.array([[0, 0], [4, 4], [4, 0], [0, 4], [0, 0]], float)
    >>> find_crossing_segments(bowtie, np.array([0, 5])).tolist()
    [True, False, True, False]
    """
    return SegmentGrid(coords, ring_offsets).crossing_mask()


def find_crossing_pairs(coords: np.ndarray, ring_offsets: np.ndarray) -> np.ndarray:
    """
    Pairs of segments that cross.

    Parameters
    ----------
    coords, ring_offsets
        As in :func:`find_crossing_segments`.

    Returns
    -------
    np.ndarray
        Segment indices (see :func:`segment_starts`), shape (n_pairs, 2) with the lower index first, sorted.
    """
    return SegmentGrid(coords, ring_offsets).crossing_pairs()
