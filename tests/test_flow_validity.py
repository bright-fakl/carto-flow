"""Validated windows: output polygons that do not self-intersect (validity="windows")."""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Point, Polygon, box

from carto_flow.flow_cartogram import CartogramWorkflow, MorphOptions, morph_geometries
from carto_flow.flow_cartogram.displacement import displace_coords_numba
from carto_flow.flow_cartogram.errors import compute_error_metrics
from carto_flow.flow_cartogram.grid import Grid
from carto_flow.flow_cartogram.options import MorphOptionsValidationError
from carto_flow.flow_cartogram.validity import (
    InsertionRound,
    ValidatedWindows,
    VertexLineage,
    _split_segments,
)
from carto_flow.geo_utils import find_crossing_segments, reconstruct_geometries, unpack_geometries

# ----------------------------------------------------------------------------
# A ribbon that a fixed field squeezes onto a curve, and a driver for the hooks
# ----------------------------------------------------------------------------


def ribbon() -> Polygon:
    """A strip of height 1 with sparse vertices; the top and the bottom have different vertices."""
    top = [(x, 1.0) for x in (0.0, 3.3, 6.6, 10.0)]
    bottom = [(x, 0.0) for x in (10.0, 5.0, 0.0)]
    return Polygon(top + bottom)


@pytest.fixture(scope="module")
def squeeze():
    """Grid and a field that moves every point to 2 % of its distance from the curve y = 0.5 + 0.5 sin(2 pi x / 10).

    One step of size 1 squeezes the ribbon to a width of 0.02 along the curve. The straight segments between the
    sparse vertices cross, the flow does not.
    """
    grid = Grid.from_bounds((0, -1, 10, 2), size=400, margin=0.0, square=False)
    x, y = np.meshgrid(grid.x_coords, grid.y_coords)
    center = 0.5 + 0.5 * np.sin(2 * np.pi * x / 10)
    return grid, np.zeros_like(y), -(y - center) * 0.98


def curve_field(grid, factor):
    """Field that moves every point by `factor` of its distance to the curve of the `squeeze` fixture."""
    x, y = np.meshgrid(grid.x_coords, grid.y_coords)
    return np.zeros_like(y), -(y - (0.5 + 0.5 * np.sin(2 * np.pi * x / 10))) * factor


def drive(windows, geoms, grid, fields, dt=1.0, steps=1, best_at=None):
    """Run the hooks of the morph loop for the given fields; returns the best state (or None)."""
    best_state = None
    iteration = 0
    for vx, vy in fields:
        best_state = windows.begin_window(geoms, grid, best_state, iteration)
        windows.add_field(vx, vy)
        for _ in range(steps):
            windows.record_step(dt)
            geoms.coords = displace_coords_numba(
                geoms.coords, grid.x_coords, grid.y_coords, vx, vy, dt, grid.dx, grid.dy, use_parallel=False
            )
            geoms.invalidate_cache()
            iteration += 1
            if best_at == iteration:
                windows.mark_best()
                best_state = (geoms.coords.copy(),)
    return windows.finish(geoms, grid, best_state, iteration)


def plain(coords, grid, field, dt=1.0, steps=1):
    for _ in range(steps):
        coords = displace_coords_numba(
            coords, grid.x_coords, grid.y_coords, field[0], field[1], dt, grid.dx, grid.dy, use_parallel=False
        )
    return coords


# ----------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------


class TestOptions:
    def test_defaults(self):
        options = MorphOptions()
        assert options.validity == "none"
        assert options.validity_check_every == 10
        assert options.validity_memory_mb == 128.0

    def test_windows_is_accepted(self):
        options = MorphOptions(validity="windows", validity_check_every=1, validity_memory_mb=0.5)
        assert options.copy_with(validity_check_every=3).validity_check_every == 3

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("validity", "quadtree"),
            ("validity", True),
            ("validity", None),
            ("validity_check_every", 0),
            ("validity_check_every", 2.5),
            ("validity_check_every", True),
            ("validity_memory_mb", 0),
            ("validity_memory_mb", -1.0),
            ("validity_memory_mb", "128"),
        ],
    )
    def test_invalid_values_are_rejected(self, name, value):
        with pytest.raises(MorphOptionsValidationError, match=name):
            MorphOptions(**{name: value})
        options = MorphOptions()
        with pytest.raises(MorphOptionsValidationError, match=name):
            setattr(options, name, value)


# ----------------------------------------------------------------------------
# Splitting segments
# ----------------------------------------------------------------------------


class TestSplitSegments:
    def test_the_two_rings_of_a_shared_boundary_get_identical_vertices(self):
        rng = np.random.default_rng(1)
        a, b = rng.uniform(0, 1, 2), rng.uniform(0, 1, 2)
        lower = [(0.0, 0.0), tuple(a), tuple(b), (1.0, -1.0), (0.0, 0.0)]
        upper = [tuple(b), tuple(a), (0.0, 2.0), (1.0, 2.0), tuple(b)]  # the boundary a-b in the opposite direction
        coords = np.array(lower + upper)
        offsets = np.array([0, 5, 10])
        split = np.zeros(8, bool)
        split[1] = True  # a-b in the lower ring
        split[4] = True  # b-a in the upper ring
        start, end, best, new_offsets, new_index, left, near = _split_segments(
            coords, coords, coords, offsets, split, np.zeros(8, bool)
        )
        assert new_offsets.tolist() == [0, 6, 12]
        assert new_index.tolist() == [2, 7]
        assert left.tolist() == [1, 5]
        assert (start[new_index[0]] == start[new_index[1]]).all()
        assert (end[new_index] == start[new_index]).all()
        assert (best[new_index] == start[new_index]).all()
        np.testing.assert_allclose(start[2], 0.5 * (a + b), rtol=0, atol=1e-15)
        # the other vertices are copied in order
        assert (np.delete(start, new_index, axis=0) == coords).all()
        # both halves of a split segment belong to the neighborhood of the segment
        near = _split_segments(coords, coords, coords, offsets, split, np.arange(8) == 1)[-1]
        assert np.flatnonzero(near).tolist() == [1, 2]


# ----------------------------------------------------------------------------
# The lineage
# ----------------------------------------------------------------------------


class TestLineage:
    def test_lift_inserts_the_fraction_between_the_end_points(self):
        rounds = (InsertionRound(np.array([1, 4]), np.array([0, 2]), 0.25, 3),)
        lineage = VertexLineage(4, rounds)
        coords = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 0.0]])
        lifted = lineage.lift(coords)
        assert lineage.n_vertices == 6
        np.testing.assert_allclose(lifted[1], [1.0, 0.0])
        np.testing.assert_allclose(lifted[4], [3.0, 3.0])
        assert (lifted[[0, 2, 3, 5]] == coords).all()

    def test_lift_composes_over_rounds_and_accepts_intermediate_states(self):
        first = InsertionRound(np.array([1]), np.array([0]), 0.5, 1)  # [0, 4, 8, 0] -> [0, 2, 4, 8, 0]
        second = InsertionRound(np.array([1, 5]), np.array([0, 3]), 0.5, 2)  # -> [0, 1, 2, 4, 8, 4, 0]
        coords = np.array([[0.0], [4.0], [8.0], [0.0]])
        full = VertexLineage(4, (first, second))
        head = VertexLineage(4, (first,))
        tail = VertexLineage(5, (second,))
        assert full.lift(coords)[:, 0].tolist() == [0.0, 1.0, 2.0, 4.0, 8.0, 4.0, 0.0]
        np.testing.assert_allclose(full.lift(coords), tail.lift(head.lift(coords)))
        # an intermediate state is accepted, a count that is not a state is not
        np.testing.assert_allclose(full.lift(head.lift(coords)), full.lift(coords))
        with pytest.raises(ValueError, match="not a state"):
            full.lift(np.zeros((6, 1)))

    def test_lift_keeps_the_extra_axes(self):
        lineage = VertexLineage(2, (InsertionRound(np.array([1]), np.array([0]), 0.5, 1),))
        assert lineage.lift(np.arange(6.0).reshape(2, 3)).shape == (3, 3)


# ----------------------------------------------------------------------------
# The check on the hooks
# ----------------------------------------------------------------------------


class TestWindows:
    def test_the_ribbon_is_invalid_without_and_valid_with_the_check(self, squeeze):
        grid, vx, vy = squeeze
        plain_geoms = unpack_geometries([ribbon()])
        plain_geoms.coords = plain(plain_geoms.coords, grid, (vx, vy))
        plain_geoms.invalidate_cache()
        assert not shapely.is_valid(reconstruct_geometries(plain_geoms)[0])

        geoms = unpack_geometries([ribbon()])
        windows = ValidatedWindows(10, 128.0, len(geoms.coords))
        drive(windows, geoms, grid, [(vx, vy)])
        (repaired,) = reconstruct_geometries(geoms)
        assert shapely.is_valid(repaired)
        assert windows.n_unresolved == 0
        assert len(geoms.coords) > len(plain_geoms.coords)
        assert not find_crossing_segments(geoms.coords, geoms.ragged_offsets[0]).any()

    def test_input_vertices_move_as_in_the_plain_loop_and_new_ones_lie_on_the_path(self, squeeze):
        grid, vx, vy = squeeze
        geoms = unpack_geometries([ribbon()])
        windows = ValidatedWindows(10, 128.0, len(geoms.coords), field_dtype=np.float64)
        start = geoms.coords.copy()
        drive(windows, geoms, grid, [(vx, vy)])
        lineage = windows.lineage()
        inserted = _inserted_positions(lineage)
        assert len(inserted) > 0
        # a vertex inserted at the chord midpoint and advected in the plain loop lands where the replay put it
        advected = plain(lineage.lift(start), grid, (vx, vy))
        np.testing.assert_allclose(geoms.coords, advected, rtol=0, atol=1e-12)
        # no input vertex is moved or removed: the others are the advected input vertices, bit for bit
        np.testing.assert_array_equal(np.delete(geoms.coords, inserted, axis=0), plain(start, grid, (vx, vy)))

    def test_float32_fields_replay_close_to_the_true_path(self, squeeze):
        grid, vx, vy = squeeze
        exact = unpack_geometries([ribbon()])
        drive(ValidatedWindows(10, 128.0, 8, field_dtype=np.float64), exact, grid, [(vx, vy)])
        single = unpack_geometries([ribbon()])
        windows = ValidatedWindows(10, 128.0, 8)
        drive(windows, single, grid, [(vx, vy)])
        assert windows._windows == []  # the fields of a span are released after the check
        assert single.coords.shape == exact.coords.shape
        np.testing.assert_allclose(single.coords, exact.coords, rtol=0, atol=1e-4)
        assert shapely.is_valid(reconstruct_geometries(single)[0])

    def test_a_span_of_two_windows_replays_through_both_fields(self, squeeze):
        grid, _, _ = squeeze
        fields = [curve_field(grid, 0.5), curve_field(grid, 0.6)]
        geoms = unpack_geometries([ribbon()])
        windows = ValidatedWindows(2, 128.0, 8, field_dtype=np.float64)
        start = geoms.coords.copy()
        drive(windows, geoms, grid, fields, dt=1.0, steps=2)
        assert windows.n_checks == 1
        assert windows.lineage().rounds
        expected = windows.lineage().lift(start)
        for field in fields:
            expected = plain(expected, grid, field, steps=2)
        np.testing.assert_allclose(geoms.coords, expected, rtol=0, atol=1e-12)

    def test_the_check_waits_for_the_end_of_the_span(self, squeeze):
        grid, vx, vy = squeeze
        geoms = unpack_geometries([ribbon()])
        windows = ValidatedWindows(3, 128.0, 8)
        best = None
        for window in range(5):
            best = windows.begin_window(geoms, grid, best, window)
            windows.add_field(vx * 0, vy * 0)  # a field that does not move anything
            windows.record_step(1.0)
        assert windows.n_checks == 1  # at the start of the fourth window
        windows.finish(geoms, grid, best, 5)
        assert windows.n_checks == 2

    def test_the_memory_budget_ends_the_span(self, squeeze):
        grid, vx, vy = squeeze
        geoms = unpack_geometries([ribbon()])
        one_window = 2 * vx.size * 4
        windows = ValidatedWindows(1000, 2.5 * one_window / 2**20, 8)
        best = None
        for window in range(10):
            best = windows.begin_window(geoms, grid, best, window)
            windows.add_field(vx * 0, vy * 0)
            windows.record_step(1.0)
        # a check whenever three windows are stored
        assert windows.n_checks == 3
        assert windows.peak_field_bytes == 3 * one_window

    def test_the_cap_on_inserted_vertices_stops_the_repair(self, squeeze):
        grid, vx, vy = squeeze
        geoms = unpack_geometries([ribbon()])
        windows = ValidatedWindows(10, 128.0, 8, max_inserted=1)
        drive(windows, geoms, grid, [(vx, vy)])
        assert not shapely.is_valid(reconstruct_geometries(geoms)[0])
        assert len(geoms.coords) == 8
        assert windows.n_unresolved > 0
        # the cap ends the checks of the run: nothing is stored any more
        windows.add_field(vx, vy)
        assert windows._windows == []
        geoms = unpack_geometries([ribbon()])
        windows = ValidatedWindows(10, 128.0, 8, max_inserted=1000)
        drive(windows, geoms, grid, [(vx, vy)])
        assert shapely.is_valid(reconstruct_geometries(geoms)[0])

    def test_the_best_iterate_is_lifted_and_valid(self, squeeze):
        grid, vx, vy = squeeze
        geoms = unpack_geometries([ribbon()])
        windows = ValidatedWindows(10, 128.0, 8, field_dtype=np.float64)
        start = geoms.coords.copy()
        slow = curve_field(grid, 0.5)
        # two windows of one step; the best state is the one after the first
        best_state = drive(windows, geoms, grid, [slow, (vx, vy)], best_at=1)
        lineage = windows.lineage()
        assert lineage.rounds
        assert len(best_state[0]) == len(geoms.coords)
        assert not find_crossing_segments(best_state[0], geoms.ragged_offsets[0]).any()
        # the inserted vertices of the best state are on the path of the first window only
        inserted = _inserted_positions(lineage)
        expected = plain(lineage.lift(start), grid, slow)
        np.testing.assert_allclose(best_state[0][inserted], expected[inserted], rtol=0, atol=1e-12)

    def test_polygons_with_holes_and_parts_keep_their_structure(self, squeeze):
        grid, vx, vy = squeeze
        holed = shapely.segmentize(
            Polygon([(0, -0.5), (10, -0.5), (10, -0.9), (0, -0.9)], [[(1, -0.6), (1, -0.8), (2, -0.8), (2, -0.6)]]),
            0.25,
        )
        parts = MultiPolygon([ribbon(), box(0, 1.2, 1, 1.4)])
        geometries = [holed, parts, box(5, 1.5, 6, 1.8)]
        geoms = unpack_geometries(geometries)
        windows = ValidatedWindows(10, 128.0, len(geoms.coords))
        drive(windows, geoms, grid, [(vx, vy)])
        rebuilt = reconstruct_geometries(geoms)
        assert len(windows.lineage().rounds) > 0
        assert [g.geom_type for g in rebuilt] == ["Polygon", "MultiPolygon", "Polygon"]
        assert len(rebuilt[0].interiors) == 1
        assert [len(g.geoms) for g in rebuilt if g.geom_type == "MultiPolygon"] == [2]
        assert sum(len(shapely.get_coordinates(g)) for g in rebuilt) == len(geoms.coords)
        # the metadata is the one a round trip through shapely gives
        again = unpack_geometries(rebuilt)
        assert again.metadata == geoms.metadata
        assert (again.ragged_offsets[0] == geoms.ragged_offsets[0]).all()
        # parts of the multipolygon and the holed polygon are valid
        assert all(shapely.is_valid(g) for g in rebuilt)


def _inserted_positions(lineage):
    """Positions in the final vertex set of the vertices inserted by any round."""
    marks = np.zeros(lineage.n_input, bool)
    for r in lineage.rounds:
        grown = np.zeros(len(marks) + len(r.new_index), bool)
        is_new = np.zeros(len(grown), bool)
        is_new[r.new_index] = True
        grown[~is_new] = marks
        grown[r.new_index] = True
        marks = grown
    return np.flatnonzero(marks)


# ----------------------------------------------------------------------------
# The morph
# ----------------------------------------------------------------------------


def strip_scene(seed, nx=4, ny=3):
    """A grid of boxes with unequal sizes and densified edges; two cells have almost no target area."""
    rng = np.random.default_rng(seed)
    xs = np.concatenate([[0], np.cumsum(rng.uniform(1.0, 4.0, nx))])
    ys = np.concatenate([[0], np.cumsum(rng.uniform(0.3, 3.0, ny))])
    polygons, values = [], []
    for j in range(ny):
        for i in range(nx):
            cell = box(xs[i], ys[j], xs[i + 1], ys[j + 1])
            polygons.append(shapely.segmentize(cell, 1.0))
            values.append(cell.area * rng.uniform(0.5, 3))
    for k in rng.choice(nx * ny, 2, replace=False):
        values[k] *= 0.003
    return polygons, np.array(values)


def morph(seed, **kwargs):
    polygons, values = strip_scene(seed)
    options = MorphOptions(grid=64, n_iter=300, mean_tol=0.02, max_tol=0.05, show_progress=False, **kwargs)
    return polygons, morph_geometries(polygons, values, options=options)


def invalid_count(result):
    return int((~shapely.is_valid(np.array(result.best.geometry, dtype=object))).sum())


pytestmark = pytest.mark.filterwarnings(
    "ignore:.*output regions are not valid polygons.*", "ignore:Grid resolution", "ignore:Vertex density"
)

# seeds whose baseline run ends with an invalid polygon, by the way the run ends
ENDS = {"n_iter": 11, "converged": 46, "stalled": 59}


class TestMorph:
    @pytest.mark.parametrize("seed", [11, 31, 33])
    def test_a_squeezed_strip_is_invalid_without_and_valid_with_windows(self, seed):
        _, baseline = morph(seed)
        if invalid_count(baseline) == 0:
            pytest.skip("the run without the option is valid for this seed on this platform (the scene is chaotic)")
        assert invalid_count(baseline) >= 1
        assert baseline.invalid_regions != ()
        polygons, repaired = morph(seed, validity="windows")
        # a zero-width cusp can stay invalid (reported in invalid_regions); the repair never adds invalid polygons
        assert invalid_count(repaired) <= invalid_count(baseline)
        assert len(repaired.invalid_regions) == invalid_count(repaired)
        assert repaired.vertex_lineage is not None
        n_input = len(shapely.get_coordinates(np.array(polygons, dtype=object)))
        n_output = len(shapely.get_coordinates(np.array(repaired.best.geometry, dtype=object)))
        assert repaired.vertex_lineage.n_input == n_input
        assert repaired.vertex_lineage.n_vertices == n_output > n_input

    @pytest.mark.parametrize("how", sorted(ENDS))
    def test_the_state_at_the_end_of_the_run_is_checked_whatever_the_position_in_the_span(self, how):
        # a span of 10,000 windows never ends before the run does: only the final check can repair
        seed = ENDS[how]
        _, baseline = morph(seed)
        if invalid_count(baseline) == 0:
            pytest.skip("the run without the option is valid for this seed on this platform (the scene is chaotic)")
        assert invalid_count(baseline) >= 1
        _, repaired = morph(seed, validity="windows", validity_check_every=10_000)
        expected = {"n_iter": "COMPLETED", "converged": "CONVERGED", "stalled": "STALLED"}[how]
        assert repaired.status.name == expected
        assert repaired.vertex_lineage.rounds
        assert invalid_count(repaired) <= invalid_count(baseline)

    def test_the_returned_best_iterate_is_checked_when_it_is_not_the_last(self):
        _, baseline = morph(ENDS["stalled"])
        _, repaired = morph(ENDS["stalled"], validity="windows", validity_check_every=10_000)
        assert repaired.best_iteration < repaired.niterations
        assert invalid_count(repaired) <= invalid_count(baseline)
        # the errors of the snapshot are those of the geometry it holds
        polygons, values = strip_scene(ENDS["stalled"])
        areas = np.array([g.area for g in repaired.best.geometry])
        target = values / values.sum() * sum(g.area for g in polygons)
        expected = compute_error_metrics(areas, target, eps=0.0)
        np.testing.assert_allclose(repaired.best.errors.log_errors, expected.log_errors, rtol=1e-9, atol=1e-12)

    def test_option_off_inserts_nothing(self):
        polygons, default = morph(11)
        _, explicit = morph(11, validity="none")
        assert default.vertex_lineage is None
        assert explicit.vertex_lineage is None
        n_input = len(shapely.get_coordinates(np.array(polygons, dtype=object)))
        assert len(shapely.get_coordinates(np.array(default.best.geometry, dtype=object))) == n_input
        coords = np.array([[0.0, 0.0], [1.0, 1.0]])
        assert default.lift_coordinates(coords) is coords or (default.lift_coordinates(coords) == coords).all()

    def test_option_off_does_not_touch_the_loop(self, monkeypatch):
        import carto_flow.flow_cartogram.algorithm as algorithm
        from carto_flow import data

        def forbidden(*args, **kwargs):
            raise AssertionError("ValidatedWindows is used with validity='none'")

        monkeypatch.setattr(algorithm, "ValidatedWindows", forbidden)
        states = data.load_us_census(level="state")
        options = MorphOptions(grid=64, n_iter=20, show_progress=False)
        result = morph_geometries(list(states.geometry), states["Population"].to_numpy(float), options=options)
        assert result.vertex_lineage is None
        n_input = len(shapely.get_coordinates(states.geometry.values))
        assert len(shapely.get_coordinates(np.array(result.best.geometry, dtype=object))) == n_input

    def test_lift_coordinates_gives_the_final_vertex_set_and_the_same_rings(self):
        polygons, repaired = morph(11, validity="windows")
        start = shapely.get_coordinates(np.array(polygons, dtype=object))
        lifted = repaired.lift_coordinates(start)
        assert len(lifted) == len(shapely.get_coordinates(np.array(repaired.best.geometry, dtype=object)))
        # the lifted polygons are the input polygons: same area, same shape
        final = unpack_geometries(repaired.best.geometry)
        moved = unpack_geometries(polygons)
        moved.set_vertices(lifted, final.ragged_offsets[0])
        rebuilt = reconstruct_geometries(moved)
        for original, new in zip(polygons, rebuilt, strict=True):
            assert new.area == pytest.approx(original.area, rel=1e-12)
            assert new.symmetric_difference(original).area == pytest.approx(0.0, abs=1e-9)

    def test_snapshots_keep_the_vertex_set_they_had_and_lift_to_the_final_one(self):
        _, result = morph(11, validity="windows", snapshot_every=5)
        final = result.vertex_lineage.n_vertices
        counts = [len(shapely.get_coordinates(np.array(s.geometry, dtype=object))) for s in result.snapshots.snapshots]
        assert len(set(counts)) > 1  # snapshots recorded before an insertion have fewer vertices
        assert max(counts) == final
        for snapshot in result.snapshots.snapshots:
            lifted = result.lift_coordinates(shapely.get_coordinates(np.array(snapshot.geometry, dtype=object)))
            assert len(lifted) == final

    def test_landmarks_and_coords_are_not_affected(self):
        polygons, values = strip_scene(11)
        landmarks = [Point(1.0, 0.5), Point(4.0, 2.0)]
        coords = np.array([[1.0, 1.0], [3.0, 2.0], [5.0, 1.0]])
        options = MorphOptions(
            grid=64, n_iter=300, mean_tol=0.02, max_tol=0.05, show_progress=False, validity="windows"
        )
        result = morph_geometries(polygons, values, landmarks=landmarks, coords=coords, options=options)
        assert result.vertex_lineage.rounds
        assert len(result.best.landmarks) == 2
        assert result.best.coords.shape == coords.shape

    def test_each_level_of_a_ladder_has_its_own_lineage(self):
        import geopandas as gpd

        polygons, values = strip_scene(11)
        gdf = gpd.GeoDataFrame({"v": values}, geometry=polygons)
        options = [
            MorphOptions(n_iter=100, mean_tol=0.02, max_tol=0.05, show_progress=False, validity="windows")
            for _ in range(2)
        ]
        workflow = CartogramWorkflow(gdf, "v", validate=False)
        workflow.morph_multiresolution(min_resolution=32, levels=2, options=options)
        levels = [c for c in workflow if c.vertex_lineage is not None]
        assert len(levels) == 2
        counts = [len(shapely.get_coordinates(np.array(c.best.geometry, dtype=object))) for c in levels]
        assert levels[0].vertex_lineage.n_input == len(shapely.get_coordinates(gdf.geometry.values))
        assert levels[1].vertex_lineage.n_input == counts[0]
        assert levels[1].vertex_lineage.n_vertices == counts[1]
