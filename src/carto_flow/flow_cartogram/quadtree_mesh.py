"""
Graded quadtree mesh with a fan triangulation, and the piecewise-affine map of its nodes (Numba).

The mesh covers the grid of the flow cartogram, padded to a square with a power of two cells per side. It is built from
a scalar field: the leaf with the largest range of the field is split first, until the number of leaves reaches a
budget, and the tree is then graded, so that neighboring leaves differ by at most one level. The mesh is stored on the
lattice of its finest leaves: a lattice square is ``block`` grid cells wide.

Every leaf is triangulated as a fan from a center node to its boundary nodes: the four corners and, on a side whose
neighbor is one level deeper, the midpoint of the side (a hanging node, shared with the two smaller leaves). A leaf has
at most eight fan triangles. A point of the lattice lies in one triangle and has barycentric coordinates in it.
Moving the nodes moves the points with the piecewise-affine map of the triangles; this map is injective while no
triangle is inverted, so two segments that lie inside different triangles, or inside one triangle, cannot cross.

The module knows nothing about cartograms: it builds the mesh, locates points, applies the map, inserts vertices where
polygon boundaries cross mesh edges, and repairs fans that invert.

Classes
-------
QuadtreeMesh
    The mesh, with point location, vertex insertion and the checks of the fan triangles.
Insertion
    Result of :meth:`QuadtreeMesh.insert`.

Functions
---------
build_quadtree
    Graded quadtree from a scalar field.
apply_map
    Piecewise-affine map of located points for given node positions.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

__all__ = ["Insertion", "QuadtreeMesh", "apply_map", "build_quadtree"]

MAX_BOUNDARY = 8
"""Boundary nodes of a leaf: four corners and at most one hanging node per side."""

_T_EPS = 1e-9
"""Crossings closer than this (as a fraction of a segment) to an end point or to each other are not separate."""


@njit(cache=True)
def _grade(depth):
    """Raise the depth of leaves until neighboring leaves differ by at most one level."""
    n = depth.shape[0]
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(n):
                d = depth[i, j]
                for k in range(2):
                    ni = i + 1 if k == 0 else i
                    nj = j + 1 if k == 1 else j
                    if ni >= n or nj >= n:
                        continue
                    e = depth[ni, nj]
                    if e > d + 1:
                        s = n >> d
                        i0 = i // s * s
                        j0 = j // s * s
                        depth[i0 : i0 + s, j0 : j0 + s] = d + 1
                        changed = True
                    elif d > e + 1:
                        s = n >> e
                        i0 = ni // s * s
                        j0 = nj // s * s
                        depth[i0 : i0 + s, j0 : j0 + s] = e + 1
                        changed = True


@njit(cache=True)
def _leaves(depth):
    """Leaf index of every cell, the leaves with their fans, and the nodes (lattice coordinates, x = column)."""
    n = depth.shape[0]
    n_leaves = 0
    for i in range(n):
        for j in range(n):
            s = n >> depth[i, j]
            if i % s == 0 and j % s == 0:
                n_leaves += 1
    leaf_of_square = np.empty((n, n), np.int32)
    box = np.empty((n_leaves, 3), np.int32)  # row, column and size
    k = 0
    for i in range(n):
        for j in range(n):
            s = n >> depth[i, j]
            if i % s == 0 and j % s == 0:
                box[k, 0] = i
                box[k, 1] = j
                box[k, 2] = s
                leaf_of_square[i : i + s, j : j + s] = k
                k += 1
    node_at = -np.ones((n + 1) * (n + 1), np.int32)
    nodes = np.empty((n_leaves * (MAX_BOUNDARY + 1), 2))
    boundary = -np.ones((n_leaves, MAX_BOUNDARY), np.int32)
    n_boundary = np.zeros(n_leaves, np.int32)
    center = np.empty(n_leaves, np.int32)
    corners = np.empty((n_leaves, 4), np.int32)
    n_nodes = 0
    for k in range(n_leaves):
        i0 = box[k, 0]
        j0 = box[k, 1]
        s = box[k, 2]
        d = depth[i0, j0]
        count = 0
        # counterclockwise in (x = column, y = row): bottom side left to right, right side, top side right to left,
        # left side top to bottom; a side contributes its first corner and, if the neighbor across it is deeper, its
        # midpoint
        for side in range(4):
            if side == 0:
                cx, cy, ex, ey = j0, i0, s, 0
                ni, nj = i0 - 1, j0
            elif side == 1:
                cx, cy, ex, ey = j0 + s, i0, 0, s
                ni, nj = i0, j0 + s
            elif side == 2:
                cx, cy, ex, ey = j0 + s, i0 + s, -s, 0
                ni, nj = i0 + s, j0 + s - 1
            else:
                cx, cy, ex, ey = j0, i0 + s, 0, -s
                ni, nj = i0 + s - 1, j0 - 1
            for part in range(2):
                if part == 1:
                    if ni < 0 or nj < 0 or ni >= n or nj >= n or depth[ni, nj] <= d:
                        continue
                    px = cx + ex // 2
                    py = cy + ey // 2
                else:
                    px = cx
                    py = cy
                key = py * (n + 1) + px
                if node_at[key] < 0:
                    node_at[key] = n_nodes
                    nodes[n_nodes, 0] = px
                    nodes[n_nodes, 1] = py
                    n_nodes += 1
                boundary[k, count] = node_at[key]
                if part == 0:
                    corners[k, side] = node_at[key]
                count += 1
        n_boundary[k] = count
        nodes[n_nodes, 0] = j0 + s / 2.0
        nodes[n_nodes, 1] = i0 + s / 2.0
        center[k] = n_nodes
        n_nodes += 1
    return leaf_of_square, box, boundary, n_boundary, center, corners, nodes[:n_nodes].copy()


@njit(cache=True, parallel=True)
def _locate(u, v, leaf_of_square, boundary, n_boundary, center, nodes):
    """Triangle (center, a, b) of the fan that holds each point, and the barycentric coordinates in it.

    A point on an edge may be assigned to either triangle; the triangle with the largest smallest coordinate is taken.
    """
    n = u.shape[0]
    size = leaf_of_square.shape[0]
    tri = np.empty((n, 3), np.int32)
    bary = np.empty((n, 3))
    for p in prange(n):
        col = min(max(int(np.floor(u[p])), 0), size - 1)
        row = min(max(int(np.floor(v[p])), 0), size - 1)
        k = leaf_of_square[row, col]
        c = center[k]
        cx = nodes[c, 0]
        cy = nodes[c, 1]
        px = u[p] - cx
        py = v[p] - cy
        best = -1e300
        m = n_boundary[k]
        for t in range(m):
            a = boundary[k, t]
            b = boundary[k, (t + 1) % m]
            ax = nodes[a, 0] - cx
            ay = nodes[a, 1] - cy
            bx = nodes[b, 0] - cx
            by = nodes[b, 1] - cy
            det = ax * by - ay * bx
            wa = (px * by - py * bx) / det
            wb = (ax * py - ay * px) / det
            wc = 1.0 - wa - wb
            low = min(wa, wb, wc)
            if low > best:
                best = low
                tri[p, 0] = c
                tri[p, 1] = a
                tri[p, 2] = b
                bary[p, 0] = wc
                bary[p, 1] = wa
                bary[p, 2] = wb
    return tri, bary


@njit(cache=True, parallel=True)
def _apply(tri, bary, nodes):
    n = tri.shape[0]
    out = np.empty((n, 2))
    for p in prange(n):
        a = tri[p, 0]
        b = tri[p, 1]
        c = tri[p, 2]
        out[p, 0] = bary[p, 0] * nodes[a, 0] + bary[p, 1] * nodes[b, 0] + bary[p, 2] * nodes[c, 0]
        out[p, 1] = bary[p, 0] * nodes[a, 1] + bary[p, 1] * nodes[b, 1] + bary[p, 2] * nodes[c, 1]
    return out


@njit(cache=True)
def _place_centers(nodes, corners, center):
    for k in range(center.shape[0]):
        x = 0.0
        y = 0.0
        for s in range(4):
            x += nodes[corners[k, s], 0]
            y += nodes[corners[k, s], 1]
        nodes[center[k], 0] = 0.25 * x
        nodes[center[k], 1] = 0.25 * y


@njit(cache=True)
def _inverted_leaves(nodes, boundary, n_boundary, center):
    """Leaves with a fan triangle of non-positive area, in ascending order."""
    flag = np.zeros(boundary.shape[0], np.bool_)
    for k in range(boundary.shape[0]):
        c = center[k]
        m = n_boundary[k]
        for t in range(m):
            a = boundary[k, t]
            b = boundary[k, (t + 1) % m]
            cross = (nodes[a, 0] - nodes[c, 0]) * (nodes[b, 1] - nodes[c, 1]) - (nodes[a, 1] - nodes[c, 1]) * (
                nodes[b, 0] - nodes[c, 0]
            )
            if cross <= 0.0:
                flag[k] = True
                break
    return np.flatnonzero(flag)


@njit(cache=True)
def _triangle_areas(nodes, boundary, n_boundary, center):
    total = 0
    for k in range(boundary.shape[0]):
        total += n_boundary[k]
    out = np.empty(total)
    i = 0
    for k in range(boundary.shape[0]):
        c = center[k]
        m = n_boundary[k]
        for t in range(m):
            a = boundary[k, t]
            b = boundary[k, (t + 1) % m]
            out[i] = 0.5 * (
                (nodes[a, 0] - nodes[c, 0]) * (nodes[b, 1] - nodes[c, 1])
                - (nodes[a, 1] - nodes[c, 1]) * (nodes[b, 0] - nodes[c, 0])
            )
            i += 1
    return out


@njit(cache=True)
def _kernel_centroid(nodes, boundary, m, k, out):
    """Centroid of the kernel of leaf k (the points that see all of its boundary); False if the kernel is empty.

    The kernel is the intersection of the left half planes of the directed boundary edges, computed by clipping the
    bounding box of the leaf with one half plane after the other.
    """
    xs = np.empty(2 * MAX_BOUNDARY + 4)
    ys = np.empty(2 * MAX_BOUNDARY + 4)
    xt = np.empty(2 * MAX_BOUNDARY + 4)
    yt = np.empty(2 * MAX_BOUNDARY + 4)
    x0 = 1e300
    y0 = 1e300
    x1 = -1e300
    y1 = -1e300
    for i in range(m):
        a = boundary[k, i]
        x0 = min(x0, nodes[a, 0])
        x1 = max(x1, nodes[a, 0])
        y0 = min(y0, nodes[a, 1])
        y1 = max(y1, nodes[a, 1])
    xs[0] = x0
    ys[0] = y0
    xs[1] = x1
    ys[1] = y0
    xs[2] = x1
    ys[2] = y1
    xs[3] = x0
    ys[3] = y1
    n = 4
    for e in range(m):
        a = boundary[k, e]
        b = boundary[k, (e + 1) % m]
        ax = nodes[a, 0]
        ay = nodes[a, 1]
        ex = nodes[b, 0] - ax
        ey = nodes[b, 1] - ay
        nn = 0
        for i in range(n):
            j = (i + 1) % n
            si = ex * (ys[i] - ay) - ey * (xs[i] - ax)
            sj = ex * (ys[j] - ay) - ey * (xs[j] - ax)
            if si >= 0.0:
                xt[nn] = xs[i]
                yt[nn] = ys[i]
                nn += 1
            if (si >= 0.0) != (sj >= 0.0):
                t = si / (si - sj)
                xt[nn] = xs[i] + t * (xs[j] - xs[i])
                yt[nn] = ys[i] + t * (ys[j] - ys[i])
                nn += 1
        if nn < 3:
            return False
        for i in range(nn):
            xs[i] = xt[i]
            ys[i] = yt[i]
        n = nn
    area = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        j = (i + 1) % n
        w = xs[i] * ys[j] - xs[j] * ys[i]
        area += w
        cx += (xs[i] + xs[j]) * w
        cy += (ys[i] + ys[j]) * w
    if area <= 0.0:
        return False
    out[0] = cx / (3.0 * area)
    out[1] = cy / (3.0 * area)
    return True


@njit(cache=True)
def _fan_valid(nodes, boundary, m, k, cx, cy, min_cross):
    for t in range(m):
        a = boundary[k, t]
        b = boundary[k, (t + 1) % m]
        cross = (nodes[a, 0] - cx) * (nodes[b, 1] - cy) - (nodes[a, 1] - cy) * (nodes[b, 0] - cx)
        if cross <= min_cross:
            return False
    return True


@njit(cache=True, parallel=True)
def _fix_centers(nodes, boundary, n_boundary, center, leaves, fixed):
    for u in prange(leaves.shape[0]):
        k = leaves[u]
        m = n_boundary[k]
        out = np.empty(2)
        fixed[u] = False
        if _kernel_centroid(nodes, boundary, m, k, out):
            x0 = 1e300
            x1 = -1e300
            y0 = 1e300
            y1 = -1e300
            for i in range(m):
                a = boundary[k, i]
                x0 = min(x0, nodes[a, 0])
                x1 = max(x1, nodes[a, 0])
                y0 = min(y0, nodes[a, 1])
                y1 = max(y1, nodes[a, 1])
            if _fan_valid(nodes, boundary, m, k, out[0], out[1], 1e-9 * (x1 - x0) * (y1 - y0)):
                nodes[center[k], 0] = out[0]
                nodes[center[k], 1] = out[1]
                fixed[u] = True


@njit(cache=True)
def _crossings(ax, ay, bx, by, leaf_of_square, boundary, n_boundary, center, nodes, buf):
    """Parameters t (ascending, in (0, 1)) where the segment a-b crosses an edge of the mesh.

    The edges are the sides of the leaves, where the leaf changes, and the fan edges from the center of a leaf to its
    boundary nodes. Returns the number of parameters written to buf.
    """
    size = leaf_of_square.shape[0]
    capacity = buf.shape[0]
    n = 0
    dx = bx - ax
    dy = by - ay
    if dx != 0.0:
        for k in range(int(np.ceil(min(ax, bx))), int(np.floor(max(ax, bx))) + 1):
            if k <= 0 or k >= size:
                continue
            t = (k - ax) / dx
            if t <= 1e-12 or t >= 1 - 1e-12:
                continue
            row = int(np.floor(ay + t * dy))
            if row < 0 or row >= size:
                continue
            if leaf_of_square[row, k - 1] != leaf_of_square[row, k] and n < capacity - 1:
                buf[n] = t
                n += 1
    if dy != 0.0:
        for k in range(int(np.ceil(min(ay, by))), int(np.floor(max(ay, by))) + 1):
            if k <= 0 or k >= size:
                continue
            t = (k - ay) / dy
            if t <= 1e-12 or t >= 1 - 1e-12:
                continue
            col = int(np.floor(ax + t * dx))
            if col < 0 or col >= size:
                continue
            if leaf_of_square[k - 1, col] != leaf_of_square[k, col] and n < capacity - 1:
                buf[n] = t
                n += 1
    _sort(buf, n)
    total = n
    t0 = 0.0
    for piece in range(n + 1):
        t1 = buf[piece] if piece < n else 1.0
        tm = 0.5 * (t0 + t1)
        col = min(max(int(np.floor(ax + tm * dx)), 0), size - 1)
        row = min(max(int(np.floor(ay + tm * dy)), 0), size - 1)
        k = leaf_of_square[row, col]
        c = center[k]
        cx = nodes[c, 0]
        cy = nodes[c, 1]
        for e in range(n_boundary[k]):
            b = boundary[k, e]
            sx = nodes[b, 0] - cx
            sy = nodes[b, 1] - cy
            den = dx * sy - dy * sx
            if abs(den) < 1e-14:
                continue
            s = ((cx - ax) * sy - (cy - ay) * sx) / den
            r = ((cx - ax) * dy - (cy - ay) * dx) / den
            if 0.0 <= r <= 1.0 and t0 + _T_EPS < s < t1 - _T_EPS and total < capacity:
                buf[total] = s
                total += 1
        t0 = t1
    _sort(buf, total)
    out = 0
    last = 0.0
    for i in range(total):
        if buf[i] - last > _T_EPS and 1.0 - buf[i] > _T_EPS:
            buf[out] = buf[i]
            out += 1
            last = buf[i]
    return out


@njit(cache=True)
def _sort(buf, n):
    for i in range(1, n):
        value = buf[i]
        j = i - 1
        while j >= 0 and buf[j] > value:
            buf[j + 1] = buf[j]
            j -= 1
        buf[j + 1] = value


@njit(cache=True, inline="always")
def _ordered(lat, p, q):
    """True if the segment p-q must be walked from q to p to have its end points in lexicographic order."""
    return lat[p, 0] > lat[q, 0] or (lat[p, 0] == lat[q, 0] and lat[p, 1] > lat[q, 1])


@njit(cache=True, parallel=True)
def _count(lat, has_segment, capacity, leaf_of_square, boundary, n_boundary, center, nodes):
    """Number of vertices to insert into the segment that starts at each vertex."""
    n = lat.shape[0]
    counts = np.zeros(n, np.int64)
    for index in prange(n):
        p = np.int64(index)
        if not has_segment[p]:
            continue
        buf = np.empty(capacity)
        q = p + 1
        a = p
        b = q
        if _ordered(lat, p, q):
            a = q
            b = p
        counts[p] = _crossings(
            lat[a, 0], lat[a, 1], lat[b, 0], lat[b, 1], leaf_of_square, boundary, n_boundary, center, nodes, buf
        )
    return counts


@njit(cache=True, parallel=True)
def _fill(
    lat,
    xy,
    has_segment,
    counts,
    position,
    capacity,
    frame,
    leaf_of_square,
    boundary,
    n_boundary,
    center,
    nodes,
    out,
    left,
    fraction,
):
    """Write the vertices and the inserted ones into out; left and fraction describe each inserted vertex.

    The end points of a segment are ordered lexicographically before the crossings are computed, so identical
    segments of two rings (a shared boundary) get bit-identical vertices.
    """
    n = lat.shape[0]
    x0, y0, dx, dy = frame
    for index in prange(n):
        p = np.int64(index)
        o = position[p]
        out[o, 0] = xy[p, 0]
        out[o, 1] = xy[p, 1]
        left[o] = -1
        if not has_segment[p] or counts[p] == 0:
            continue
        buf = np.empty(capacity)
        q = p + 1
        swap = _ordered(lat, p, q)
        a = p
        b = q
        if swap:
            a = q
            b = p
        ax = lat[a, 0]
        ay = lat[a, 1]
        bx = lat[b, 0]
        by = lat[b, 1]
        m = _crossings(ax, ay, bx, by, leaf_of_square, boundary, n_boundary, center, nodes, buf)
        for i in range(m):
            t = buf[m - 1 - i] if swap else buf[i]
            out[o + 1 + i, 0] = x0 + (ax + t * (bx - ax)) * dx
            out[o + 1 + i, 1] = y0 + (ay + t * (by - ay)) * dy
            left[o + 1 + i] = p
            fraction[o + 1 + i] = 1.0 - t if swap else t


@njit(cache=True)
def _segment_extent(lat, has_segment):
    longest = 0.0
    for p in range(lat.shape[0]):
        if has_segment[p]:
            longest = max(longest, abs(lat[p + 1, 0] - lat[p, 0]) + abs(lat[p + 1, 1] - lat[p, 1]))
    return longest


@dataclass(frozen=True)
class Insertion:
    """
    Vertices inserted where polygon boundaries cross mesh edges.

    Attributes
    ----------
    coords : np.ndarray
        Vertex coordinates after the insertion, shape (n + k, 2).
    ring_offsets : np.ndarray
        Start of every ring in ``coords`` and the total length.
    new_index : np.ndarray
        Positions of the k inserted vertices in ``coords``, ascending.
    left : np.ndarray
        For each inserted vertex, the index in the input of the first end point of the segment it lies on; the second
        end point is ``left + 1``, in the same ring.
    fraction : np.ndarray
        The vertex lies at ``(1 - fraction) * input[left] + fraction * input[left + 1]``.
    """

    coords: np.ndarray
    ring_offsets: np.ndarray
    new_index: np.ndarray
    left: np.ndarray
    fraction: np.ndarray


@dataclass(frozen=True)
class QuadtreeMesh:
    """
    A graded quadtree with fan triangles, on the lattice of its finest leaves.

    Create it with :func:`build_quadtree`. The lattice has ``size`` squares on each side and a square is ``block``
    grid cells wide (the size of the finest leaf). A point (x, y) has the lattice coordinates
    ``((x - x0) / dx, (y - y0) / dy)`` with ``frame = (x0, y0, dx, dy)``, the lower left corner of the lattice and the
    size of a lattice square, so the square of column j and row i spans [j, j + 1] by [i, i + 1]. Node positions are
    given in the coordinates of the frame, not on the lattice.

    Attributes
    ----------
    size : int
        Squares per side of the lattice (a power of two).
    block : int
        Grid cells per side of a lattice square (a power of two).
    frame : tuple of float
        ``(x0, y0, dx, dy)``: lower left corner of the lattice and size of a lattice square.
    leaf_of_square : np.ndarray
        Leaf index of every lattice square, shape (size, size).
    boundary : np.ndarray
        For each leaf, the node indices of its boundary in counterclockwise order, padded with -1, shape (n_leaves, 8).
    n_boundary : np.ndarray
        Number of boundary nodes of each leaf (the number of fan triangles).
    center : np.ndarray
        Node index of the fan center of each leaf.
    corners : np.ndarray
        Node indices of the four corners of each leaf.
    lattice_nodes : np.ndarray
        Node positions on the lattice, shape (n_nodes, 2).
    """

    size: int
    block: int
    frame: tuple[float, float, float, float]
    leaf_of_square: np.ndarray
    boundary: np.ndarray
    n_boundary: np.ndarray
    center: np.ndarray
    corners: np.ndarray
    lattice_nodes: np.ndarray

    @property
    def n_leaves(self) -> int:
        """Number of leaves."""
        return len(self.center)

    @property
    def n_triangles(self) -> int:
        """Number of fan triangles."""
        return int(self.n_boundary.sum())

    def _tables(self):
        return self.boundary, self.n_boundary, self.center

    def node_xy(self) -> np.ndarray:
        """Node positions in the coordinates of the frame, shape (n_nodes, 2)."""
        x0, y0, dx, dy = self.frame
        return np.column_stack([x0 + self.lattice_nodes[:, 0] * dx, y0 + self.lattice_nodes[:, 1] * dy])

    def _lattice(self, xy: np.ndarray) -> np.ndarray:
        x0, y0, dx, dy = self.frame
        return np.ascontiguousarray(np.column_stack([(xy[:, 0] - x0) / dx, (xy[:, 1] - y0) / dy]))

    def locate(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Triangle and barycentric coordinates of points.

        Points outside the lattice are assigned to the nearest square. A point on an edge belongs to one of the
        triangles that share the edge, and the same point is always assigned to the same triangle.

        Parameters
        ----------
        xy : np.ndarray
            Point coordinates, shape (n, 2).

        Returns
        -------
        tri : np.ndarray
            Node indices ``(center, a, b)`` of the triangle of each point, shape (n, 3), int32.
        bary : np.ndarray
            Barycentric coordinates in the order of ``tri``, shape (n, 3).
        """
        lat = self._lattice(xy)
        return _locate(
            np.ascontiguousarray(lat[:, 0]),
            np.ascontiguousarray(lat[:, 1]),
            self.leaf_of_square,
            *self._tables(),
            self.lattice_nodes,
        )

    def place_centers(self, node_xy: np.ndarray) -> None:
        """Put the fan center of every leaf at the mean of its four corners (in place)."""
        _place_centers(node_xy, self.corners, self.center)

    def inverted_leaves(self, node_xy: np.ndarray) -> np.ndarray:
        """Indices of the leaves with a fan triangle of non-positive area."""
        return _inverted_leaves(node_xy, *self._tables())

    def triangle_areas(self, node_xy: np.ndarray) -> np.ndarray:
        """Signed areas of all fan triangles, in leaf order."""
        return _triangle_areas(node_xy, *self._tables())

    def fix_centers(self, node_xy: np.ndarray, leaves: np.ndarray) -> np.ndarray:
        """
        Move the fan center of the given leaves to the centroid of the kernel of the leaf polygon (in place).

        A leaf that is squeezed into a sliver with a reflex corner has its center (the mean of the corners) outside
        its kernel, and its fan inverts. The kernel is the set of points that see the whole boundary, so a fan from
        its centroid has no inverted triangle. A leaf with an empty kernel, or whose fan still inverts, is left
        unchanged.

        Returns
        -------
        np.ndarray
            Boolean mask over ``leaves``: True where the center was moved and the fan is valid.
        """
        leaves = np.ascontiguousarray(leaves, dtype=np.int64)
        fixed = np.zeros(len(leaves), np.bool_)
        _fix_centers(node_xy, *self._tables(), leaves, fixed)
        return fixed

    def leaf_nodes(self, leaves: np.ndarray) -> np.ndarray:
        """Indices of the nodes (boundary and center) of the given leaves."""
        nodes = self.boundary[leaves]
        return np.concatenate([nodes[nodes >= 0], self.center[leaves]])

    def insert(self, xy: np.ndarray, ring_offsets: np.ndarray) -> Insertion | None:
        """
        Insert vertices where the segments of the rings cross an edge of the mesh.

        Afterwards every segment lies inside one triangle (up to rounding), so a vertex and its neighbors move with
        one affine map. The two rings of a shared boundary get bit-identical vertices: the end points of a segment
        are ordered lexicographically before the crossings are computed.

        Parameters
        ----------
        xy : np.ndarray
            Vertex coordinates of all rings, shape (n, 2).
        ring_offsets : np.ndarray
            Start of every ring and the total length. Rings are closed; segments are the consecutive vertex pairs
            of a ring.

        Returns
        -------
        Insertion or None
            ``None`` if no vertex is inserted.
        """
        n = len(xy)
        lat = self._lattice(xy)
        has_segment = np.ones(n, np.bool_)
        has_segment[np.asarray(ring_offsets)[1:] - 1] = False
        # a segment crosses at most nine mesh edges per lattice cell it passes through
        capacity = int(9 * _segment_extent(lat, has_segment)) + 64
        counts = _count(lat, has_segment, capacity, self.leaf_of_square, *self._tables(), self.lattice_nodes)
        total = int(counts.sum())
        if total == 0:
            return None
        before = np.cumsum(counts) - counts
        position = np.arange(n, dtype=np.int64) + before
        out = np.empty((n + total, 2))
        left = np.empty(n + total, np.int64)
        fraction = np.empty(n + total)
        _fill(
            lat,
            np.ascontiguousarray(xy),
            has_segment,
            counts,
            position,
            capacity,
            self.frame,
            self.leaf_of_square,
            *self._tables(),
            self.lattice_nodes,
            out,
            left,
            fraction,
        )
        new_index = np.flatnonzero(left >= 0)
        offsets = np.asarray(ring_offsets, dtype=np.int64)
        new_offsets = offsets + np.concatenate([before, [total]])[offsets]
        return Insertion(out, new_offsets, new_index, left[new_index], fraction[new_index])


def apply_map(tri: np.ndarray, bary: np.ndarray, node_xy: np.ndarray) -> np.ndarray:
    """
    Positions of located points for given node positions.

    Parameters
    ----------
    tri, bary
        Triangles and barycentric coordinates from :meth:`QuadtreeMesh.locate`.
    node_xy : np.ndarray
        Node positions, shape (n_nodes, 2).

    Returns
    -------
    np.ndarray
        The points ``bary @ node_xy[tri]``, shape (n, 2).
    """
    return _apply(tri, bary, np.ascontiguousarray(node_xy))


@njit(cache=True, parallel=True)
def _reduce(high, low):
    """Maximum of ``high`` and minimum of ``low`` over 2 by 2 blocks."""
    n = high.shape[0] // 2
    out_high = np.empty((n, n), high.dtype)
    out_low = np.empty((n, n), low.dtype)
    for i in prange(n):
        for j in range(n):
            out_high[i, j] = max(
                max(high[2 * i, 2 * j], high[2 * i, 2 * j + 1]), max(high[2 * i + 1, 2 * j], high[2 * i + 1, 2 * j + 1])
            )
            out_low[i, j] = min(
                min(low[2 * i, 2 * j], low[2 * i, 2 * j + 1]), min(low[2 * i + 1, 2 * j], low[2 * i + 1, 2 * j + 1])
            )
    return out_high, out_low


@njit(cache=True)
def _fill_depth(depth, leaf_depth, leaf_row, leaf_col):
    finest = depth.shape[0]
    for k in range(leaf_depth.shape[0]):
        # a leaf of depth d covers finest >> d squares on each side
        s = finest >> leaf_depth[k]
        depth[leaf_row[k] * s : (leaf_row[k] + 1) * s, leaf_col[k] * s : (leaf_col[k] + 1) * s] = leaf_depth[k]


def build_quadtree(
    field: np.ndarray, frame: tuple[float, float, float, float], leaf_fraction: float = 2.0**-6
) -> QuadtreeMesh:
    """
    Graded quadtree from a scalar field on a regular grid.

    The root is the square of the grid padded to a power of two (the padding holds the mean of the field). The leaf
    whose cells have the largest range of the field (maximum minus minimum) is split first, until the number of
    leaves reaches ``leaf_fraction`` times the number of grid cells or the field is constant in every leaf. The tree
    is then graded: neighboring leaves differ by at most one level.

    Parameters
    ----------
    field : np.ndarray
        Field values, shape (ny, nx), row i at y index i.
    frame : tuple of float
        ``(x0, y0, dx, dy)``: the lower left corner of the cell (0, 0) of the grid and the cell size.
    leaf_fraction : float, default=2**-6
        Leaf budget as a fraction of the number of grid cells (at least four leaves).

    Returns
    -------
    QuadtreeMesh
        The frame of the mesh is that of its lattice: the cell size is multiplied by ``block``.
    """
    ny, nx = field.shape
    levels = max(int(np.ceil(np.log2(max(nx, ny)))), 1)
    size = 1 << levels
    padded = np.full((size, size), float(np.mean(field)), np.float32)
    padded[:ny, :nx] = field
    high = [padded]
    low = [padded]
    for _ in range(levels):
        h, lo = _reduce(high[-1], low[-1])
        high.append(h)
        low.append(lo)
    high.reverse()
    low.reverse()  # index = depth
    # split the leaf with the largest range first; a leaf is (depth, row, column) in units of its own size
    target = max(int(nx * ny * leaf_fraction), 4)
    heap = [(-float(high[0][0, 0] - low[0][0, 0]), 0, 0, 0)]
    leaves = []  # leaves of the finest level, which cannot be split
    n_leaves = 1
    while heap and n_leaves < target and heap[0][0] != 0:
        _, d, i, j = heapq.heappop(heap)
        n_leaves += 3
        for ci in (2 * i, 2 * i + 1):
            for cj in (2 * j, 2 * j + 1):
                if d + 1 < levels:
                    heapq.heappush(heap, (-float(high[d + 1][ci, cj] - low[d + 1][ci, cj]), d + 1, ci, cj))
                else:
                    leaves.append((d + 1, ci, cj))
    leaves.extend(item[1:] for item in heap)
    leaf = np.array(leaves, dtype=np.int64)
    finest = int(leaf[:, 0].max())
    depth = np.zeros((1 << finest, 1 << finest), np.int8)
    _fill_depth(depth, leaf[:, 0], leaf[:, 1], leaf[:, 2])
    _grade(depth)
    leaf_of_square, _, boundary, n_boundary, center, corners, nodes = _leaves(depth)
    block = size >> finest
    x0, y0, dx, dy = frame
    return QuadtreeMesh(
        1 << finest,
        block,
        (x0, y0, dx * block, dy * block),
        leaf_of_square,
        boundary,
        n_boundary,
        center,
        corners,
        nodes,
    )
