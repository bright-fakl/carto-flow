"""Animation of flow cartograms with inserted vertices: lift_geometries and align_vertices (validity="repair-crossings")."""

from __future__ import annotations

import itertools
import json

import geopandas as gpd
import matplotlib
import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon, box
from test_flow_validity import strip_scene

import carto_flow.flow_cartogram.animation as animation_module
from carto_flow.flow_cartogram import (
    Cartogram,
    CartogramWorkflow,
    MorphOptions,
    align_vertices,
    morph_geometries,
)
from carto_flow.flow_cartogram.animation import animate_geometry_keyframes, animate_morph_history
from carto_flow.flow_cartogram.history import CartogramSnapshot, History
from carto_flow.flow_cartogram.validity import InsertionRound, RemovalRound, VertexLineage

matplotlib.use("Agg")

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*output regions are not valid polygons.*", "ignore:Grid resolution", "ignore:Vertex density"
)


# one vertex at the middle of the first segment of the shell, the hole and the second part
INSERTION = InsertionRound(new_index=np.array([1, 7, 18]), left=np.array([0, 5, 15]), fraction=0.5, iteration=3)


def count(geometries) -> int:
    return len(shapely.get_coordinates(np.asarray(geometries, dtype=object)))


def run(seed, values=None, **kwargs):
    """Morph of the squeezed-strip scene of `seed` (optionally with other values)."""
    polygons, scene_values = strip_scene(seed)
    options = MorphOptions(
        grid=64, n_iter=300, mean_tol=0.02, max_tol=0.05, show_progress=False, stall_lagging=False, **kwargs
    )
    return polygons, morph_geometries(polygons, scene_values if values is None else values, options=options)


@pytest.fixture(scope="module")
def grown():
    """A windows morph whose snapshots have different numbers of vertices, and its input and values."""
    polygons, result = run(11, validity="repair-crossings", snapshot_every=5)
    return polygons, strip_scene(11)[1], result


@pytest.fixture(scope="module")
def pair():
    """Two windows morphs of one input with different values (different inserted vertices) and one without lineage."""
    polygons, values = strip_scene(11)
    _, a = run(11, validity="repair-crossings")
    _, b = run(11, values[::-1].copy(), validity="repair-crossings")
    _, plain = run(11, values[::-1].copy())
    return polygons, a, b, plain


def synthetic():
    """A cartogram of an input with a holed polygon and a multipolygon, with one vertex inserted in three rings."""
    shell, hole = box(0, 0, 4, 4).exterior.coords[:], box(1, 1, 3, 3).exterior.coords[:]
    part1, part2 = box(6, 0, 8, 2), box(10, 0, 12, 2)
    start = np.array([Polygon(shell, [hole]), MultiPolygon([part1, part2])], dtype=object)  # 5 + 5 + 5 + 5 vertices

    def midpoint(ring):
        return [ring[0], tuple((np.array(ring[0]) + ring[1]) / 2), *ring[1:]]

    inserted = np.array(
        [
            Polygon(midpoint(shell), [midpoint(hole)]),
            MultiPolygon([part1, Polygon(midpoint(part2.exterior.coords[:]))]),
        ],
        dtype=object,
    )
    lineage = VertexLineage(n_input=20, rounds=(INSERTION,))
    final = shapely.transform(inserted, lambda c: c * 2.0 + 1.0)
    cart = Cartogram(snapshots=History([CartogramSnapshot(iteration=5, geometry=final)]), vertex_lineage=lineage)
    return start, cart


class TestLiftGeometries:
    def test_snapshots_lift_to_the_vertices_of_the_returned_geometries(self, grown):
        _, _, result = grown
        counts = {count(s.geometry) for s in result.snapshots.snapshots}
        assert len(counts) > 1
        for snapshot in result.snapshots.snapshots:
            lifted = result.lift_geometries(snapshot.geometry)
            assert isinstance(lifted, np.ndarray)
            assert count(lifted) == result.vertex_lineage.n_vertices == count(result.get_geometry())
            np.testing.assert_allclose(shapely.area(lifted), shapely.area(np.asarray(snapshot.geometry)), rtol=1e-12)

    def test_the_input_lifts_with_the_same_shape(self, grown):
        polygons, _, result = grown
        lifted = result.lift_geometries(polygons)
        assert count(lifted) == count(result.get_geometry())
        for original, new in zip(polygons, lifted, strict=True):
            assert new.symmetric_difference(original).area == pytest.approx(0.0, abs=1e-9)

    def test_the_returned_geometries_are_unchanged(self, grown):
        _, _, result = grown
        final = np.asarray(result.get_geometry(), dtype=object)
        lifted = result.lift_geometries(final)
        assert shapely.equals_exact(lifted, final, tolerance=0).all()

    def test_holes_and_parts_are_kept(self):
        start, cart = synthetic()
        lifted = cart.lift_geometries(start)
        assert count(lifted) == 23
        assert [g.geom_type for g in lifted] == ["Polygon", "MultiPolygon"]
        assert len(lifted[0].interiors) == 1
        assert len(lifted[1].geoms) == 2
        np.testing.assert_allclose(shapely.area(lifted), shapely.area(start), rtol=1e-12)
        # the ring of the hole holds the inserted vertex
        assert len(lifted[0].interiors[0].coords) == 6
        np.testing.assert_allclose(lifted[0].interiors[0].coords[1], [3.0, 2.0], atol=1e-12)

    def test_without_a_lineage_the_geometries_are_returned_unchanged(self):
        polygons, result = run(11)
        assert result.vertex_lineage is None
        lifted = result.lift_geometries(polygons)
        assert shapely.equals_exact(lifted, np.array(polygons, dtype=object), tolerance=0).all()

    def test_errors(self, grown):
        polygons, _, result = grown
        with pytest.raises(ValueError, match="geometries given"):
            result.lift_geometries(polygons[:-1])
        with pytest.raises(ValueError, match="not a state"):
            result.lift_geometries(shapely.segmentize(np.array(polygons, dtype=object), 0.1))
        start, cart = synthetic()
        with pytest.raises(ValueError, match="parts and rings"):
            cart.lift_geometries(np.array([start[1], start[0]], dtype=object))


class TestAnimateMorphHistory:
    @pytest.mark.parametrize("facecolor", [True, False])
    def test_linear_lifts_the_snapshots(self, grown, monkeypatch, facecolor):
        _, values, result = grown
        n_final = result.vertex_lineage.n_vertices
        seen = []
        fast = animation_module._fast_precompute_paths
        slow = animation_module._interpolate_geometry
        monkeypatch.setattr(
            animation_module,
            "_fast_precompute_paths",
            lambda gdfs, *args: seen.extend(count(g.geometry) for g in gdfs) or fast(gdfs, *args),
        )
        monkeypatch.setattr(
            animation_module,
            "_interpolate_geometry",
            lambda a, b, t: seen.append((count([a]), count([b]))) or slow(a, b, t),
        )
        animate_morph_history(
            result, interpolation="linear", duration=1, fps=5, facecolor=values if facecolor else None
        )
        assert seen
        if facecolor:
            assert set(seen) == {n_final}
        else:  # one geometry at a time: the vertices of the geometries of the returned state
            per_geometry = set(shapely.get_num_coordinates(np.asarray(result.get_geometry(), dtype=object)))
            assert {n for pair_ in seen for n in pair_} <= per_geometry
            assert all(a == b for a, b in seen)

    def test_nearest_shows_the_snapshots_as_recorded(self, grown, monkeypatch):
        _, values, result = grown
        seen = []
        fast = animation_module._fast_precompute_paths
        monkeypatch.setattr(
            animation_module,
            "_fast_precompute_paths",
            lambda gdfs, *args: seen.extend(count(g.geometry) for g in gdfs) or fast(gdfs, *args),
        )
        animate_morph_history(result, interpolation="nearest", duration=1, fps=5, facecolor=values)
        assert len(set(seen)) > 1

    def test_a_snapshot_that_cannot_be_lifted_names_itself(self, grown, monkeypatch):
        _, values, result = grown
        snapshots = result.snapshots.snapshots
        broken = snapshots[1]
        original = result.lift_geometries
        calls = []

        def fail(*args):
            calls.append(args)
            if len(calls) == 2:  # the second snapshot
                raise ValueError("no such state")
            return original(*args)

        monkeypatch.setattr(result, "lift_geometries", fail)
        with pytest.raises(ValueError, match=rf"iteration {broken.iteration}.*interpolation='nearest'"):
            animate_morph_history(result, interpolation="linear", duration=1, fps=5, facecolor=values)

    def test_equal_vertex_counts_are_not_lifted(self, monkeypatch):
        _, result = run(11)
        monkeypatch.setattr(
            Cartogram, "lift_geometries", lambda *args: pytest.fail("lift_geometries is used without a lineage")
        )
        animate_morph_history(result, interpolation="linear", duration=1, fps=5, facecolor=np.arange(12.0))


class TestKeyframesWithDifferentVertices:
    @pytest.fixture
    def frames(self):
        a = box(0, 0, 1, 1)
        b = shapely.segmentize(box(0, 0, 2, 1), 0.5)
        return [gpd.GeoDataFrame(geometry=[g]) for g in (a, b)]

    def test_a_mismatch_with_colors_raises_a_clear_error(self, frames):
        with pytest.raises(ValueError, match=r"different numbers of vertices.*align_vertices.*lift_geometries"):
            animate_geometry_keyframes(frames, interpolation="linear", facecolor=np.array([1.0]), duration=1, fps=5)

    def test_equal_vertices_still_animate(self, frames):
        animate_geometry_keyframes([frames[0], frames[0]], facecolor=np.array([1.0]), duration=1, fps=5)

    def test_nearest_needs_no_alignment(self, frames):
        animate_geometry_keyframes(frames, interpolation="nearest", facecolor=np.array([1.0]), duration=1, fps=5)


def check_aligned(outputs, aligned):
    """Equal structure, and every output keeps its shape and validity."""
    assert len({count(g) for g in aligned}) == 1
    for first, other in itertools.pairwise(aligned):
        assert [g.geom_type for g in first] == [g.geom_type for g in other]
        assert [shapely.get_num_geometries(g) for g in first] == [shapely.get_num_geometries(g) for g in other]
        assert [len(shapely.get_rings(g)) for g in first] == [len(shapely.get_rings(g)) for g in other]
    for output, new in zip(outputs, aligned, strict=True):
        output = np.asarray(output, dtype=object)
        assert count(new) >= count(output)
        np.testing.assert_allclose(shapely.area(new), shapely.area(output), rtol=1e-12)
        assert shapely.hausdorff_distance(output, new).max() < 1e-9
        assert (shapely.is_valid(output) == shapely.is_valid(new)).all()
        # the output vertices are among the vertices of the result
        have = {tuple(c) for c in shapely.get_coordinates(new)}
        assert all(tuple(c) in have for c in shapely.get_coordinates(output))


class TestAlignVertices:
    def test_two_morphs_with_different_inserted_vertices(self, pair):
        _, a, b, _ = pair
        assert count(a.get_geometry()) != count(b.get_geometry())
        aligned = align_vertices([a, b])
        check_aligned([a.get_geometry(), b.get_geometry()], aligned)
        assert count(aligned[0]) > max(count(a.get_geometry()), count(b.get_geometry())) - 1
        assert (shapely.is_valid(aligned[0])).all()
        assert (shapely.is_valid(aligned[1])).all()

    def test_the_result_is_a_pre_processing_step_for_the_keyframe_animation(self, pair):
        polygons, a, b, _ = pair
        frames = [
            gpd.GeoDataFrame({"v": np.arange(len(polygons), dtype=float)}, geometry=list(g))
            for g in align_vertices([a, b])
        ]
        animate_geometry_keyframes(frames, interpolation="linear", facecolor="v", duration=1, fps=5)

    def test_a_morph_without_lineage_mixes_with_one_with_lineage(self, pair):
        _, a, _, plain = pair
        assert plain.vertex_lineage is None
        aligned = align_vertices([plain, a])
        check_aligned([plain.get_geometry(), a.get_geometry()], aligned)
        assert count(aligned[0]) == a.vertex_lineage.n_vertices  # only the inserted vertices of `a` are added

    def test_two_morphs_without_lineage_are_returned_with_their_vertices(self, pair):
        polygons, _, _, plain = pair
        _, other = run(11)
        aligned = align_vertices([plain, other])
        check_aligned([plain.get_geometry(), other.get_geometry()], aligned)
        assert count(aligned[0]) == count(polygons)

    def test_three_sources_and_the_order_of_the_sources(self, pair):
        _, a, b, plain = pair
        aligned = align_vertices([a, plain, b])
        check_aligned([a.get_geometry(), plain.get_geometry(), b.get_geometry()], aligned)
        reordered = align_vertices([b, a])
        assert shapely.equals_exact(reordered[1], aligned[0], tolerance=0).all()

    def test_a_ladder_workflow_is_chained_through_its_levels(self):
        polygons, values = strip_scene(31)
        gdf = gpd.GeoDataFrame({"v": values, "w": values[::-1].copy()}, geometry=polygons)
        options = MorphOptions(
            grid=64, n_iter=60, mean_tol=0.02, max_tol=0.05, show_progress=False, validity="repair-crossings"
        )
        workflows = []
        for column in ("v", "w"):
            workflow = CartogramWorkflow(gdf, column, validate=False)
            workflow.morph(options=options)
            workflow.morph()
            workflows.append(workflow)
        levels = [c.vertex_lineage for c in workflows[0]][1:]
        assert levels[1].n_input == levels[0].n_vertices  # the second level starts from the first
        aligned = align_vertices(workflows)
        check_aligned([w[-1].get_geometry() for w in workflows], aligned)
        # a plain Cartogram of a single morph can be mixed with a workflow
        _, single = run(31, values[::-1].copy(), validity="repair-crossings")
        mixed = align_vertices([workflows[0], single])
        check_aligned([workflows[0][-1].get_geometry(), single.get_geometry()], mixed)

    def test_errors(self, pair):
        _, a, _, _ = pair
        with pytest.raises(ValueError, match="at least two"):
            align_vertices([a])
        with pytest.raises(TypeError, match="Cartogram or CartogramWorkflow"):
            align_vertices([a, object()])
        _, other_input = run(31, validity="repair-crossings")
        with pytest.raises(ValueError, match="same input"):
            align_vertices([a, other_input])
        _, cart = synthetic()
        with pytest.raises(ValueError, match="same input"):
            align_vertices([a, cart])

    def test_positions_must_increase_along_the_rings(self):
        _, cart = synthetic()
        other = Cartogram(snapshots=cart.snapshots, vertex_lineage=cart.vertex_lineage)
        bad_round = InsertionRound(new_index=INSERTION.new_index, left=INSERTION.left, fraction=1.5, iteration=3)
        other.vertex_lineage = VertexLineage(n_input=20, rounds=(bad_round,))
        with pytest.raises(ValueError, match="not increasing"):
            align_vertices([cart, other])

    def test_the_synthetic_input_with_holes_and_parts_aligns(self):
        start, cart = synthetic()
        moved = Cartogram(
            snapshots=History([
                CartogramSnapshot(
                    iteration=5, geometry=shapely.transform(np.asarray(start, dtype=object), lambda c: c * 3.0)
                )
            ])
        )
        aligned = align_vertices([cart, moved])
        check_aligned([cart.get_geometry(), moved.get_geometry()], aligned)
        assert [g.geom_type for g in aligned[1]] == ["Polygon", "MultiPolygon"]
        assert count(aligned[0]) == 23


# ----------------------------------------------------------------------------
# validity="repair-quadtree": removals, states shared by two vertex counts, the lineage ends at the best iterate
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def quadtree():
    """A quadtree morph with snapshots before and after the best iterate, and a second one with other values."""
    polygons, values = strip_scene(11)
    _, first = run(11, validity="repair-quadtree", snapshot_every=5)
    _, second = run(11, values[::-1].copy(), validity="repair-quadtree")
    _, plain = run(11, values[::-1].copy())
    return polygons, values, first, second, plain


class TestStateAt:
    @pytest.fixture
    def lineage(self):
        # 3 vertices, one inserted at iteration 5 (4), then removed again at iteration 8 (3)
        insertion = InsertionRound(new_index=np.array([1]), left=np.array([0]), fraction=0.5, iteration=5)
        removal = RemovalRound(kept_index=np.array([0, 2, 3]), iteration=8)
        return VertexLineage(n_input=3, rounds=(insertion, removal))

    def test_a_shared_count_is_resolved_by_the_iteration(self, lineage):
        assert [lineage.state_at(3, it) for it in (0, 4, 6, 7, 8, 20)] == [0, 0, 0, 0, 2, 2]

    def test_a_unique_count_gives_its_state(self, lineage):
        assert lineage.state_at(4, 5) == 1
        assert lineage.state_at(4, 100) == 1

    def test_unknown_counts_raise(self, lineage):
        with pytest.raises(ValueError, match="not a state"):
            lineage.state_at(5, 10)


class TestQuadtreeLifting:
    def test_the_lineage_has_insertions_and_removals(self, quadtree):
        _, _, first, second, _ = quadtree
        for result in (first, second):
            kinds = {type(r) for r in result.vertex_lineage.rounds}
            assert kinds == {InsertionRound, RemovalRound}

    def test_snapshots_up_to_the_best_iterate_lift_to_the_returned_vertex_set(self, quadtree):
        result = quadtree[2]
        before = [s for s in result.snapshots.snapshots if s.iteration <= result.best_iteration]
        assert len({count(s.geometry) for s in before}) > 1
        for snapshot in before:
            state = result.vertex_lineage.state_at(count(snapshot.geometry), snapshot.iteration)
            lifted = result.lift_geometries(snapshot.geometry, state)
            assert count(lifted) == count(result.get_geometry())
            # the vertices that were removed later lie within the removal tolerance of the chord of their neighbors
            area = shapely.area(np.asarray(snapshot.geometry, dtype=object))
            assert np.abs(shapely.area(lifted) - area).max() < 1e-3 * area.sum()

    def test_a_snapshot_after_the_best_iterate_is_not_a_state_of_the_lineage(self, quadtree):
        result = quadtree[2]
        after = [s for s in result.snapshots.snapshots if s.iteration > result.best_iteration]
        counts = set(result.vertex_lineage._counts())
        # a snapshot whose vertex count is that of no state cannot be lifted (one that happens to have the count of a
        # state is not detectable by its count, so the scene decides whether there is any)
        for snapshot in after:
            if count(snapshot.geometry) not in counts:
                with pytest.raises(ValueError, match="not a state"):
                    result.lift_geometries(snapshot.geometry)

    def test_a_state_that_does_not_match_raises(self, quadtree):
        polygons, _, result, _, _ = quadtree
        with pytest.raises(ValueError, match="not the state"):
            result.lift_geometries(polygons, state=1)

    def test_the_input_lifts_with_the_same_shape(self, quadtree):
        polygons, _, result, _, _ = quadtree
        lifted = result.lift_geometries(polygons, state=0)
        assert count(lifted) == count(result.get_geometry())
        for original, new in zip(polygons, lifted, strict=True):
            assert new.symmetric_difference(original).area == pytest.approx(0.0, abs=1e-9)


class TestQuadtreeAnimation:
    def collect(self, monkeypatch):
        seen = []
        fast = animation_module._fast_precompute_paths
        monkeypatch.setattr(
            animation_module,
            "_fast_precompute_paths",
            lambda gdfs, *args: seen.append([count(g.geometry) for g in gdfs]) or fast(gdfs, *args),
        )
        return seen

    def test_linear_ends_at_the_best_iterate_and_has_equal_vertex_sets(self, quadtree, monkeypatch):
        _, values, result, _, _ = quadtree
        seen = self.collect(monkeypatch)
        animate_morph_history(result, interpolation="linear", duration=1, fps=5, facecolor=values)
        n_final = count(result.get_geometry())
        (counts,) = seen
        assert counts == [n_final] * len(result.snapshots.through_best())
        assert len(counts) < len(result.snapshots.snapshots)

    def test_linear_without_colors_builds_the_animation(self, quadtree):
        animate_morph_history(quadtree[2], interpolation="linear", duration=1, fps=5)

    def test_snapshots_after_the_best_iterate_cannot_be_lifted(self, quadtree):
        _, values, result, _, _ = quadtree
        first_after = next(s for s in result.snapshots.snapshots if s.iteration > result.best_iteration)
        with pytest.raises(ValueError, match=rf"iteration {first_after.iteration}.*lineage ends.*through='best'"):
            animate_morph_history(result, interpolation="linear", through="last", duration=1, fps=5, facecolor=values)

    def test_nearest_can_continue_after_the_best_iterate(self, quadtree, monkeypatch):
        _, values, result, _, _ = quadtree
        seen = self.collect(monkeypatch)
        animate_morph_history(result, interpolation="nearest", through="last", duration=1, fps=5, facecolor=values)
        assert len(seen[0]) == len(result.snapshots.snapshots)


class TestQuadtreeAlignment:
    def test_two_morphs_with_different_inserted_and_removed_vertices(self, quadtree):
        _, _, first, second, _ = quadtree
        assert count(first.get_geometry()) != count(second.get_geometry())
        aligned = align_vertices([first, second])
        check_aligned([first.get_geometry(), second.get_geometry()], aligned)
        assert shapely.is_valid(aligned[0]).all()
        assert shapely.is_valid(aligned[1]).all()
        frames = [gpd.GeoDataFrame({"v": np.ones(len(g))}, geometry=list(g)) for g in aligned]
        animate_geometry_keyframes(frames, interpolation="linear", facecolor="v", duration=1, fps=5)

    def test_a_morph_without_lineage_and_a_windows_morph_mix_with_quadtree(self, quadtree):
        _, _, first, _, plain = quadtree
        _, windows = run(11, strip_scene(11)[1][::-1].copy(), validity="repair-crossings")
        aligned = align_vertices([plain, first, windows])
        check_aligned([plain.get_geometry(), first.get_geometry(), windows.get_geometry()], aligned)

    def test_a_ladder_workflow_is_chained_through_its_levels(self):
        polygons, values = strip_scene(31)
        gdf = gpd.GeoDataFrame({"v": values, "w": values[::-1].copy()}, geometry=polygons)
        options = MorphOptions(
            grid=64, n_iter=60, mean_tol=0.02, max_tol=0.05, show_progress=False, validity="repair-quadtree"
        )
        workflows = []
        for column in ("v", "w"):
            workflow = CartogramWorkflow(gdf, column, validate=False)
            workflow.morph(options=options)
            workflow.morph()
            workflows.append(workflow)
        levels = [c.vertex_lineage for c in workflows[0]][1:]
        assert levels[1].n_input == levels[0].n_vertices
        aligned = align_vertices(workflows)
        check_aligned([w[-1].get_geometry() for w in workflows], aligned)


# ----------------------------------------------------------------------------
# save_cartogram / load_cartogram keep the vertex lineage
# ----------------------------------------------------------------------------


def same_rounds(a, b):
    assert a.n_input == b.n_input
    assert [type(r) for r in a.rounds] == [type(r) for r in b.rounds]
    for x, y in zip(a.rounds, b.rounds, strict=True):
        assert x.iteration == y.iteration
        if isinstance(x, InsertionRound):
            np.testing.assert_array_equal(x.new_index, y.new_index)
            np.testing.assert_array_equal(x.left, y.left)
            np.testing.assert_array_equal(x.fraction, y.fraction)
        else:
            np.testing.assert_array_equal(x.kept_index, y.kept_index)


class TestSerialization:
    def test_the_lineage_round_trips_through_json(self, quadtree):
        # a quadtree lineage has insertion and removal rounds; a scalar and an array fraction
        lineage = quadtree[2].vertex_lineage
        mixed = VertexLineage(
            n_input=lineage.n_input,
            rounds=(
                *lineage.rounds,
                InsertionRound(
                    new_index=np.array([0, 3]), left=np.array([0, 1]), fraction=np.array([0.25, 0.5]), iteration=999
                ),
            ),
        )
        restored = VertexLineage.from_dict(json.loads(json.dumps(mixed.to_dict())))
        same_rounds(mixed, restored)
        assert restored.n_vertices == mixed.n_vertices

    @pytest.mark.parametrize("name", ["grown", "quadtree"])
    def test_a_loaded_cartogram_lifts_and_aligns_like_the_original(self, name, grown, quadtree, tmp_path):
        polygons, result = (grown[0], grown[2]) if name == "grown" else (quadtree[0], quadtree[2])
        path = tmp_path / "c.json"
        result.save(path)
        loaded = type(result).load(path)
        same_rounds(result.vertex_lineage, loaded.vertex_lineage)
        coords = shapely.get_coordinates(np.asarray(polygons, dtype=object))
        np.testing.assert_array_equal(loaded.lift_coordinates(coords), result.lift_coordinates(coords))

    def test_a_loaded_cartogram_aligns_with_another(self, pair, tmp_path):
        _, a, b, _ = pair
        for name, cartogram in (("a", a), ("b", b)):
            cartogram.save(tmp_path / f"{name}.json")
        loaded = [type(a).load(tmp_path / f"{name}.json") for name in "ab"]
        got = align_vertices(loaded)
        want = align_vertices([a, b])
        for g, w in zip(got, want, strict=True):
            assert count(g) == count(w)

    def test_files_without_a_lineage_load_without_one(self, pair, tmp_path):
        plain = pair[3]
        plain.save(tmp_path / "p.json")
        assert plain.vertex_lineage is None
        assert type(plain).load(tmp_path / "p.json").vertex_lineage is None

    def test_an_unknown_version_raises(self):
        with pytest.raises(ValueError, match="version"):
            VertexLineage.from_dict({"version": 2, "n_input": 3, "rounds": []})
