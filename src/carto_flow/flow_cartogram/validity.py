"""
Validated windows: output polygons that do not self-intersect.

Vertices move individually through the velocity field, so a segment between two vertices is not the image of the
segment, and where a polygon is squeezed into a strip narrower than the vertex spacing the segments cross. Inside a
window (the iterations between two recomputes of the field) the field is fixed, so every vertex moves independently of
all others. At the end of a span of ``validity_check_every`` windows, :class:`ValidatedWindows`

1. finds the crossing segments (:mod:`carto_flow.geo_utils.crossings`),
2. splits each of them at the midpoint of its chord at the start of the span (the two rings of a shared boundary get
   identical vertices),
3. moves the new vertices through the recorded steps of the span (replay), so they land on the path the flow gives
   them,
4. tests the new segments again around the crossings, for at most ``MAX_ROUNDS`` rounds.

No input vertex is moved or removed. Every insertion round is recorded in a :class:`VertexLineage`, which lifts
coordinates of an earlier state to the vertex set of the final geometry.

The morph loop talks to a validity method through the hooks of :class:`ValidityMethod`; the other method,
:class:`carto_flow.flow_cartogram.validity_quadtree.QuadtreeWindows`, implements the same hooks and records insertions
and removals in the same lineage.

Classes
-------
InsertionRound
    The vertices inserted in one round.
RemovalRound
    The vertices kept in one round that removed vertices.
VertexLineage
    The insertion and removal rounds of a morph, with the lift of earlier coordinates.
ValidityMethod
    The hooks of the morph loop; the defaults do nothing.
ValidatedWindows
    The state of the check, owned by one call of the morph loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from ..geo_utils.crossings import SegmentGrid, segment_starts
from .displacement import displace_coords_numba

if TYPE_CHECKING:
    from ..geo_utils.geometry import GeometryCoordinateInfo
    from .grid import Grid

__all__ = ["InsertionRound", "RemovalRound", "ValidatedWindows", "ValidityMethod", "VertexLineage"]

MAX_ROUNDS = 14
"""Rounds of insertion and re-test of one check."""

GROWTH_ROUNDS = 3
"""A check stops when the number of crossing segments has grown by ``GROWTH_FACTOR`` in each of the last rounds."""

GROWTH_FACTOR = 1.4

MIN_SPLIT_CELLS = 2e-7
"""Segments whose chord at the start of the span is shorter than this many grid cells are not split."""


@dataclass(frozen=True)
class InsertionRound:
    """
    The vertices inserted in one round.

    Attributes
    ----------
    new_index : np.ndarray
        Indices of the inserted vertices in the vertex set after the round, ascending.
    left : np.ndarray
        For each inserted vertex, the index in the vertex set before the round of the left end point of the segment
        it splits. The right end point is ``left + 1``, in the same ring.
    fraction : float or np.ndarray
        The vertex lies at ``(1 - fraction) * coords[left] + fraction * coords[left + 1]`` in every state of the
        morph before the round. One value for all vertices of the round, or one per vertex.
    iteration : int
        Number of iterations done when the round ran.
    """

    new_index: np.ndarray
    left: np.ndarray
    fraction: float | np.ndarray
    iteration: int


@dataclass(frozen=True)
class RemovalRound:
    """
    The vertices removed in one round.

    Attributes
    ----------
    kept_index : np.ndarray
        Indices in the vertex set before the round of the vertices that remain, ascending. The others are removed;
        a removed vertex is never an input vertex.
    iteration : int
        Number of iterations done when the round ran.
    """

    kept_index: np.ndarray
    iteration: int


@dataclass(frozen=True)
class VertexLineage:
    """
    The vertices inserted and removed by a validity method, in the order of the rounds.

    A morph with ``validity="windows"`` inserts vertices into the segments of the geometries. With
    ``validity="quadtree"`` it also removes inserted vertices that lie on the chord of their neighbors. None of the
    input vertices moves or is removed, and the order of the vertices of a ring is kept. The lineage lifts the
    coordinates of a state of the run (the input, or a snapshot recorded earlier) to the vertex set of the returned
    geometry.

    Attributes
    ----------
    n_input : int
        Number of vertices (counting the closing vertex of every ring) of the geometries the morph started from.
    rounds : tuple of InsertionRound or RemovalRound
        The rounds in order. A round changes the number of vertices.
    """

    n_input: int
    rounds: tuple[InsertionRound | RemovalRound, ...] = field(default_factory=tuple)

    def _counts(self) -> list[int]:
        """Number of vertices of the input and after each round."""
        counts = [self.n_input]
        for r in self.rounds:
            counts.append(counts[-1] + len(r.new_index) if isinstance(r, InsertionRound) else len(r.kept_index))
        return counts

    @property
    def n_vertices(self) -> int:
        """Number of vertices after the last round."""
        return self._counts()[-1]

    def state_at(self, n_vertices: int, iteration: int) -> int:
        """
        The state of a snapshot with ``n_vertices`` vertices that was recorded after ``iteration`` iterations.

        The number of vertices identifies the state, unless removals gave two states the same number: then it is
        the last of them whose rounds all ran at or before the iteration. The result is the argument ``state`` of
        :meth:`lift`.

        Raises
        ------
        ValueError
            If no state of the run has that many vertices (at that iteration).
        """
        states = [s for s, count in enumerate(self._counts()) if count == n_vertices]
        if len(states) > 1:
            states = [s for s in states if all(r.iteration <= iteration for r in self.rounds[:s])][-1:]
        if not states:
            raise ValueError(f"{n_vertices} vertices is not a state of this run at iteration {iteration}")
        return states[0]

    def lift(self, values: np.ndarray, state: int | None = None) -> np.ndarray:
        """
        Lift values at the vertices of an earlier state to the vertices after the last round.

        Parameters
        ----------
        values : np.ndarray
            Array whose first axis runs over the vertices of an earlier state, in the order of
            ``shapely.get_coordinates`` of the geometries (a ring list with the closing vertex of every ring),
            for example coordinates of shape (n, 2). The number of vertices must be ``n_input`` or the number after
            one of the rounds.
        state : int, optional
            Number of rounds that the state of ``values`` has seen (0 for the input). Only needed when the number of
            vertices does not identify the state: after removals, two states of the run can have the same number of
            vertices.

        Returns
        -------
        np.ndarray
            The values at all vertices after the last round: the old values at the old vertices and, at each
            inserted vertex, the interpolation between the values at the end points of the segment it lies on
            (recursively when a round splits a segment that an earlier round created). Vertices removed by a round
            are dropped.

        Raises
        ------
        ValueError
            If the number of vertices is not that of a state of the run, or identifies more than one state and
            ``state`` is not given.
        """
        values = np.asarray(values)
        counts = self._counts()
        n = len(values)
        if state is None:
            states = [i for i, count in enumerate(counts) if count == n]
            if not states:
                raise ValueError(
                    f"{n} vertices is not a state of this run (states have {sorted(set(counts))} vertices)"
                )
            if len(states) > 1:
                raise ValueError(f"{n} vertices is the size of the states {states} of this run: pass state")
            state = states[0]
        elif not 0 <= state < len(counts) or counts[state] != n:
            raise ValueError(f"{n} vertices is not the state {state} of this run")
        for r in self.rounds[state:]:
            if isinstance(r, RemovalRound):
                values = values[r.kept_index]
                continue
            lifted = np.empty((len(values) + len(r.new_index), *values.shape[1:]), dtype=np.result_type(values, float))
            is_new = np.zeros(len(lifted), bool)
            is_new[r.new_index] = True
            lifted[~is_new] = values
            fraction = np.asarray(r.fraction, dtype=float)
            if fraction.ndim:
                fraction = fraction.reshape(-1, *([1] * (values.ndim - 1)))
            lifted[r.new_index] = (1.0 - fraction) * values[r.left] + fraction * values[r.left + 1]
            values = lifted
        return values


@njit(cache=True)
def _split_segments(start, end, best, offsets, split, near):
    """Insert the midpoint of the start chord into the segments with split True.

    The end points of a chord are ordered lexicographically before the midpoint is taken, so identical segments of
    two rings get bit-identical vertices. The inserted vertex holds the midpoint in all three coordinate arrays and
    ``near`` is inherited by both halves of a split segment.
    """
    n_rings = offsets.shape[0] - 1
    n_split = 0
    for k in range(split.shape[0]):
        if split[k]:
            n_split += 1
    n_new = start.shape[0] + n_split
    start_o = np.empty((n_new, 2))
    end_o = np.empty((n_new, 2))
    best_o = np.empty((n_new, 2))
    offsets_o = np.empty(n_rings + 1, np.int64)
    new_index = np.empty(n_split, np.int64)
    left = np.empty(n_split, np.int64)
    near_o = np.zeros(split.shape[0] + n_split, np.bool_)
    o = 0
    k = 0
    kn = 0
    m = 0
    for r in range(n_rings):
        offsets_o[r] = o
        for i in range(offsets[r], offsets[r + 1] - 1):
            start_o[o] = start[i]
            end_o[o] = end[i]
            best_o[o] = best[i]
            o += 1
            if split[k]:
                ax, ay, bx, by = start[i, 0], start[i, 1], start[i + 1, 0], start[i + 1, 1]
                if ax > bx or (ax == bx and ay > by):
                    ax, ay, bx, by = bx, by, ax, ay
                mx = ax + 0.5 * (bx - ax)
                my = ay + 0.5 * (by - ay)
                for arr in (start_o, end_o, best_o):
                    arr[o, 0] = mx
                    arr[o, 1] = my
                new_index[m] = o
                left[m] = i
                near_o[kn] = near[k]
                near_o[kn + 1] = near[k]
                kn += 2
                m += 1
                o += 1
            else:
                near_o[kn] = near[k]
                kn += 1
            k += 1
        last = offsets[r + 1] - 1
        start_o[o] = start[last]
        end_o[o] = end[last]
        best_o[o] = best[last]
        o += 1
    offsets_o[n_rings] = o
    return start_o, end_o, best_o, offsets_o, new_index, left, near_o


@njit(cache=True)
def _long_enough(start, offsets, min_length2):
    n = 0
    for r in range(offsets.shape[0] - 1):
        n += max(offsets[r + 1] - offsets[r] - 1, 0)
    keep = np.zeros(n, np.bool_)
    k = 0
    for r in range(offsets.shape[0] - 1):
        for i in range(offsets[r], offsets[r + 1] - 1):
            dx = start[i + 1, 0] - start[i, 0]
            dy = start[i + 1, 1] - start[i, 1]
            keep[k] = dx * dx + dy * dy > min_length2
            k += 1
    return keep


def _diverging(flagged: list[int]) -> bool:
    """A fold of the flow itself is flagged by every sub-segment, so splitting doubles the crossings in each round."""
    last = flagged[-GROWTH_ROUNDS - 1 :]
    return len(last) > GROWTH_ROUNDS and all(b >= GROWTH_FACTOR * a for a, b in pairwise(last))


class ValidityMethod:
    """
    The hooks that the morph loop calls on a validity method.

    The defaults do nothing, so a method implements the hooks it needs. The loop calls, in this order, once per
    window: :meth:`begin_window` (before the density is computed), :meth:`set_density` and :meth:`add_field`; once
    per iteration: :meth:`record_step` and, if :attr:`displaces_geometry`, :meth:`displace` in place of the
    displacement of the geometry coordinates; :meth:`mark_best` when an iteration becomes the best iterate;
    :meth:`restore_best` when the run returns an earlier best iterate; :meth:`finish` when the run ends.
    """

    displaces_geometry = False
    """True if :meth:`displace` moves the geometry coordinates instead of the loop."""

    def begin_window(
        self, geoms: GeometryCoordinateInfo, grid: Grid, best_state: tuple | None, iteration: int
    ) -> tuple | None:
        """Start of a window, before the density is computed; returns the best state."""
        return best_state

    def set_density(self, rho: np.ndarray) -> None:
        """The density the field of the window is computed from."""

    def add_field(self, vx: np.ndarray, vy: np.ndarray) -> None:
        """The field of the window that :meth:`begin_window` opened."""

    def record_step(self, dt: float) -> None:
        """The step size of an iteration taken with the current field."""

    def displace(self, grid: Grid, vx: np.ndarray, vy: np.ndarray, dt: float) -> np.ndarray:
        """New geometry coordinates after one step with the field (when :attr:`displaces_geometry`)."""
        raise NotImplementedError

    def mark_best(self) -> None:
        """The state after the last step is the best iterate."""

    def restore_best(self, geoms: GeometryCoordinateInfo, coords: np.ndarray) -> None:
        """The run returns the best iterate, whose coordinates are ``coords``: put its vertex set on ``geoms``."""

    def finish(
        self, geoms: GeometryCoordinateInfo, grid: Grid, best_state: tuple | None, iteration: int
    ) -> tuple | None:
        """End of the run; returns the best state."""
        return best_state

    def lineage(self) -> VertexLineage:
        """The insertions and removals of the run."""
        raise NotImplementedError


class _Window:
    """The field of one window (float32 by default) and the step sizes taken with it."""

    __slots__ = ("dts", "vx", "vy")

    def __init__(self, vx: np.ndarray, vy: np.ndarray) -> None:
        self.vx = vx
        self.vy = vy
        self.dts: list[float] = []

    @property
    def nbytes(self) -> int:
        return self.vx.nbytes + self.vy.nbytes


class ValidatedWindows(ValidityMethod):
    """
    The validity check of a morph with ``validity="windows"``.

    One instance belongs to one call of the morph loop (one level of a ladder); it holds the geometry at the start of
    the current span, the fields and step sizes of the windows of the span, and the insertion rounds. The loop calls
    :meth:`begin_window`, :meth:`add_field`, :meth:`record_step`, :meth:`mark_best` and :meth:`finish`.

    Parameters
    ----------
    check_every : int
        Windows per span.
    memory_mb : float
        A span also ends when its stored fields reach this size.
    n_vertices : int
        Number of vertices of the input geometries.
    field_dtype : numpy dtype, default=float32
        Storage type of the fields.
    max_inserted : int, optional
        Cap on the vertices inserted in the whole run, defaults to ``max(5000, n_vertices // 100)``. A crossing that
        the flow itself creates (a fold) is flagged again by every check and splitting does not remove it, so the
        number of inserted vertices would grow without bound. A check stops before the round that would exceed the
        cap and disables the checks of the rest of the run (the fields are no longer stored); the regions that
        are still invalid are reported by the morph as usual.
    """

    def __init__(
        self,
        check_every: int,
        memory_mb: float,
        n_vertices: int,
        field_dtype: type = np.float32,
        max_inserted: int | None = None,
    ) -> None:
        self.check_every = check_every
        self.memory_bytes = int(memory_mb * 2**20)
        self.field_dtype = field_dtype
        self.max_inserted = max(5000, n_vertices // 100) if max_inserted is None else max_inserted
        self._n_input = n_vertices
        self._rounds: list[InsertionRound] = []
        self._start: np.ndarray | None = None  # coordinates at the start of the span
        self._windows: list[_Window] = []
        self._span = 0
        self._best_span = -1
        self._best_steps = 0
        self._exhausted = False
        self._inserted = 0
        self.n_checks = 0
        self.n_unresolved = 0
        self.peak_field_bytes = 0

    # ------------------------------------------------------------------
    # Hooks of the morph loop
    # ------------------------------------------------------------------

    def begin_window(
        self, geoms: GeometryCoordinateInfo, grid: Grid, best_state: tuple | None, iteration: int
    ) -> tuple | None:
        """
        Start of a window, before the density is computed: end the span if it is due, then open a window.

        Returns the best state, whose coordinates are lifted to the vertex set of ``geoms`` if vertices were inserted.
        """
        if self._exhausted:
            return best_state
        if self._start is not None and (
            len(self._windows) >= self.check_every or self._field_bytes() >= self.memory_bytes
        ):
            best_state = self._check(geoms, grid, best_state, iteration)
        if self._start is None and not self._exhausted:
            self._start = geoms.coords.copy()
            self._span += 1
        return best_state

    def add_field(self, vx: np.ndarray, vy: np.ndarray) -> None:
        """Store the field of the window that :meth:`begin_window` opened."""
        if self._exhausted:
            return
        self._windows.append(_Window(vx.astype(self.field_dtype), vy.astype(self.field_dtype)))
        self.peak_field_bytes = max(self.peak_field_bytes, self._field_bytes())

    def record_step(self, dt: float) -> None:
        """Record the step size of an iteration taken with the current field."""
        if self._exhausted:
            return
        self._windows[-1].dts.append(float(dt))

    def mark_best(self) -> None:
        """The state after the last recorded step is the best iterate."""
        self._best_span = self._span
        self._best_steps = self._n_steps()

    def finish(
        self, geoms: GeometryCoordinateInfo, grid: Grid, best_state: tuple | None, iteration: int
    ) -> tuple | None:
        """End of the run: check the current state and the best iterate, whatever the position in the span."""
        if self._exhausted or self._start is None or self._n_steps() == 0:
            return best_state
        return self._check(geoms, grid, best_state, iteration)

    def lineage(self) -> VertexLineage:
        """The insertion rounds of the run."""
        return VertexLineage(self._n_input, tuple(self._rounds))

    # ------------------------------------------------------------------
    # The check
    # ------------------------------------------------------------------

    def _n_steps(self) -> int:
        return sum(len(w.dts) for w in self._windows)

    def _field_bytes(self) -> int:
        return sum(w.nbytes for w in self._windows)

    def _replay(self, points: np.ndarray, grid: Grid, keep: set[int]) -> dict[int, np.ndarray]:
        """Move points through the recorded steps of the span; the positions after the steps in ``keep``."""
        out = {}
        k = 0
        for window in self._windows:
            for dt in window.dts:
                points = displace_coords_numba(
                    points, grid.x_coords, grid.y_coords, window.vx, window.vy, dt, grid.dx, grid.dy, use_parallel=False
                )
                k += 1
                if k in keep:
                    out[k] = points
        return out

    def _check(
        self, geoms: GeometryCoordinateInfo, grid: Grid, best_state: tuple | None, iteration: int
    ) -> tuple | None:
        """Test the state (and the best iterate if it lies in the span) and repair the crossings; ends the span."""
        self.n_checks += 1
        end = geoms.coords
        offsets = np.asarray(geoms.ragged_offsets[0], dtype=np.int64)  # type: ignore[index]
        best_open = best_state is not None and self._best_span == self._span
        best = best_state[0] if best_state is not None else end

        end_grid = SegmentGrid(end, offsets)
        flag = end_grid.crossing_mask()
        if best_open:
            flag |= SegmentGrid(best, offsets).crossing_mask()
        if flag.any() and self._n_steps() > 0:
            end, best, offsets = self._repair(end_grid, best, offsets, flag, grid, best_open, iteration)
            geoms.set_vertices(end, offsets)
            if best_state is not None:
                best_state = (best, *best_state[1:])
        self._start = None
        self._windows = []
        return best_state

    def _repair(self, end_grid, best, offsets, flag, grid, best_open, iteration):
        """Rounds of insertion at the crossing segments; returns the end state, the best state and the offsets."""
        start = self._start
        end = end_grid.coords
        n_steps = self._n_steps()
        min_length2 = (MIN_SPLIT_CELLS * min(grid.dx, grid.dy)) ** 2
        near = end_grid.neighborhood(flag)
        local_grid = end_grid
        local = None  # indices of the segments of the neighborhood in the current structure
        flagged = [int(flag.sum())]  # crossing segments before the first round and after each round
        keep_steps = {n_steps, self._best_steps} if best_open else {n_steps}
        for _ in range(MAX_ROUNDS):
            identical = local_grid.with_identical(flag)
            if local is None:
                split = identical
            else:
                split = np.zeros(len(near), np.bool_)
                split[local[identical]] = True
            split &= _long_enough(start, offsets, min_length2)
            if not split.any():
                break
            if self._inserted + int(split.sum()) > self.max_inserted:
                self._exhausted = True
                break
            start, end, best, offsets, new_index, left, near = _split_segments(start, end, best, offsets, split, near)
            positions = self._replay(np.ascontiguousarray(start[new_index]), grid, keep_steps)
            end[new_index] = positions[n_steps]
            # the end points of the split segment are the neighbors of the new vertex
            best[new_index] = (
                positions[self._best_steps] if best_open else 0.5 * (best[new_index - 1] + best[new_index + 1])
            )
            self._rounds.append(InsertionRound(new_index, left, 0.5, iteration))
            self._inserted += len(new_index)
            local = np.flatnonzero(near)
            local_grid = SegmentGrid(end, starts=segment_starts(offsets)[local])
            flag = local_grid.crossing_mask()
            if best_open:
                flag |= SegmentGrid(best, starts=local_grid.starts).crossing_mask()
            if not flag.any():
                break
            flagged.append(int(flag.sum()))
            if _diverging(flagged):
                break
        self.n_unresolved += int(flag.sum())
        return end, best, offsets
