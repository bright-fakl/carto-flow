"""Quadtree windows: output polygons that do not self-intersect (validity="quadtree")."""

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
from carto_flow.flow_cartogram.quadtree_mesh import apply_map, build_quadtree
from carto_flow.flow_cartogram.validity import InsertionRound, RemovalRound, VertexLineage
from carto_flow.flow_cartogram.validity_quadtree import QuadtreeWindows, _coordinate_hash, _prune_mask
from carto_flow.geo_utils import find_crossing_segments, reconstruct_geometries, unpack_geometries

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*output regions are not valid polygons.*", "ignore:Grid resolution", "ignore:Vertex density"
)

# ----------------------------------------------------------------------------
# The mesh
# ----------------------------------------------------------------------------

FRAME = (3.0, -2.0, 0.5, 0.25)  # origin and cell size of the grid; not square, not at the origin


def bumpy_field(ny=37, nx=50, seed=0):
    """A field with a few bumps and noise, so that the tree has leaves of many sizes."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:ny, 0:nx]
    field = np.exp(-((x - 12) ** 2 + (y - 10) ** 2) / 30.0) + 0.6 * np.exp(-((x - 38) ** 2 + (y - 25) ** 2) / 12.0)
    return field + 0.02 * rng.random((ny, nx))


@pytest.fixture(scope="module")
def mesh():
    return build_quadtree(bumpy_field(), FRAME, leaf_fraction=2.0**-4)


def leaf_boxes(mesh):
    """Lower left corner and side of every leaf on the lattice, from its corner nodes."""
    corners = mesh.lattice_nodes[mesh.corners]  # (leaves, 4, 2): bottom left, bottom right, top right, top left
    side = corners[:, 1, 0] - corners[:, 0, 0]
    return corners[:, 0, 0], corners[:, 0, 1], side


def barycentric(points, a, b, c):
    """Barycentric coordinates of points in the triangles (a, b, c); all arrays of shape (n, 2)."""
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    wb = ((points[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (points[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) / det
    wc = ((b[:, 0] - a[:, 0]) * (points[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (points[:, 0] - a[:, 0])) / det
    return np.column_stack([1 - wb - wc, wb, wc])


class TestMesh:
    def test_leaves_tile_the_padded_domain(self, mesh):
        x, y, side = leaf_boxes(mesh)
        assert mesh.size == 2 ** int(np.log2(mesh.size))
        assert mesh.size * mesh.block == 64  # 50 columns, padded to a power of two
        assert (side == side.astype(int)).all()
        assert (side * side).sum() == mesh.size**2
        # every lattice square belongs to the leaf that contains it
        rows, cols = np.indices(mesh.leaf_of_square.shape)
        leaf = mesh.leaf_of_square
        assert (
            (x[leaf] <= cols) & (cols < x[leaf] + side[leaf]) & (y[leaf] <= rows) & (rows < y[leaf] + side[leaf])
        ).all()
        assert np.bincount(leaf.ravel(), minlength=mesh.n_leaves).tolist() == (side**2).astype(int).tolist()

    def test_neighboring_leaves_differ_by_at_most_one_level(self, mesh):
        _, _, side = leaf_boxes(mesh)
        leaf = mesh.leaf_of_square
        for a, b in ((leaf[:, :-1], leaf[:, 1:]), (leaf[:-1, :], leaf[1:, :])):
            different = a != b
            ratio = side[a[different]] / side[b[different]]
            assert set(np.unique(ratio)) <= {0.5, 1.0, 2.0}
        assert len(np.unique(side)) >= 3

    def test_hanging_nodes_are_shared_by_the_leaves_on_both_sides(self, mesh):
        x, y, side = leaf_boxes(mesh)
        nodes = mesh.lattice_nodes
        pairs = set()
        leaf = mesh.leaf_of_square
        for a, b in ((leaf[:, :-1], leaf[:, 1:]), (leaf[:-1, :], leaf[1:, :])):
            different = a != b
            pairs.update(zip(a[different].tolist(), b[different].tolist(), strict=True))

        def on_line(k, axis, value, lo, hi):
            ids = mesh.boundary[k, : mesh.n_boundary[k]]
            p = nodes[ids]
            keep = (p[:, axis] == value) & (p[:, 1 - axis] >= lo) & (p[:, 1 - axis] <= hi)
            return {tuple(q) for q in p[keep]}

        for a, b in pairs:
            for axis in (0, 1):
                # a leaf whose upper (right) side lies on the lower (left) side of the other
                lines = x if axis == 0 else y
                u = a if lines[a] + side[a] == lines[b] else b if lines[b] + side[b] == lines[a] else None
                if u is None:
                    continue
                v = b if u == a else a
                other = 1 - axis
                coord = (x, y)[other]
                lo = max(coord[u], coord[v])
                hi = min(coord[u] + side[u], coord[v] + side[v])
                if hi <= lo:
                    continue
                line = lines[u] + side[u]
                first = on_line(u, axis, line, lo, hi)
                second = on_line(v, axis, line, lo, hi)
                assert first == second
        assert (mesh.n_boundary > 4).any()  # the tree has leaves of different sizes next to each other

    def test_fan_triangles_are_positive_and_tile_the_lattice(self, mesh):
        areas = mesh.triangle_areas(mesh.lattice_nodes)
        assert len(areas) == mesh.n_triangles
        assert (areas > 0).all()
        assert areas.sum() == pytest.approx(mesh.size**2)
        assert len(mesh.inverted_leaves(mesh.lattice_nodes)) == 0
        assert mesh.n_boundary.max() <= 8

    def test_a_point_is_in_exactly_one_triangle_and_apply_at_the_start_is_the_identity(self, mesh):
        rng = np.random.default_rng(2)
        x0, y0, dx, dy = mesh.frame
        extent = mesh.size
        points = np.column_stack([x0 + rng.uniform(0, extent, 4000) * dx, y0 + rng.uniform(0, extent, 4000) * dy])
        tri, bary = mesh.locate(points)
        assert bary.min() >= -1e-12
        np.testing.assert_allclose(bary.sum(axis=1), 1.0, rtol=0, atol=1e-12)
        nodes = mesh.node_xy()
        np.testing.assert_allclose(apply_map(tri, bary, nodes), points, rtol=0, atol=1e-9)
        # no other triangle of the leaf contains the point (random points are not on an edge)
        lat = np.column_stack([(points[:, 0] - x0) / dx, (points[:, 1] - y0) / dy])
        leaf = mesh.leaf_of_square[lat[:, 1].astype(int), lat[:, 0].astype(int)]
        for p in range(0, len(points), 40):
            k = leaf[p]
            m = mesh.n_boundary[k]
            ids = mesh.boundary[k, :m]
            a = np.repeat(nodes[mesh.center[k]][None], m, axis=0)
            b = nodes[ids]
            c = nodes[np.roll(ids, -1)]
            inside = (barycentric(np.repeat(points[p][None], m, axis=0), a, b, c) > 1e-12).all(axis=1)
            assert inside.sum() == 1
            assert set(tri[p]) == {mesh.center[k], *ids[inside].tolist(), *np.roll(ids, -1)[inside].tolist()}

    def test_the_map_is_piecewise_affine_in_the_nodes(self, mesh):
        rng = np.random.default_rng(3)
        x0, y0, dx, dy = mesh.frame
        points = np.column_stack([x0 + rng.uniform(0, mesh.size, 500) * dx, y0 + rng.uniform(0, mesh.size, 500) * dy])
        tri, bary = mesh.locate(points)
        nodes = mesh.node_xy()
        moved = nodes @ np.array([[0.0, 2.0], [-1.0, 0.0]]) + np.array([5.0, 7.0])  # rotation, scaling, shift
        expected = points @ np.array([[0.0, 2.0], [-1.0, 0.0]]) + np.array([5.0, 7.0])
        np.testing.assert_allclose(apply_map(tri, bary, moved), expected, rtol=0, atol=1e-9)

    def test_a_constant_field_gives_one_leaf(self):
        mesh = build_quadtree(np.ones((20, 20)), (0.0, 0.0, 1.0, 1.0))
        assert mesh.n_leaves == 1
        assert mesh.size == 1
        assert mesh.block == 32


def random_rings(seed, n_rings=6, n=40, extent=(0.0, 0.0, 25.0, 9.0)):
    """Closed random polygons with segments much longer than the cells of the mesh."""
    rng = np.random.default_rng(seed)
    x0, y0, w, h = extent
    rings = []
    for _ in range(n_rings):
        center = rng.uniform([x0 + 2, y0 + 1], [x0 + w - 2, y0 + h - 1])
        angle = np.sort(rng.uniform(0, 2 * np.pi, n))
        radius = rng.uniform(0.5, 2.0, n)
        ring = center + np.column_stack([radius * np.cos(angle), 0.5 * radius * np.sin(angle)])
        rings.append(np.vstack([ring, ring[:1]]))
    offsets = np.concatenate([[0], np.cumsum([len(r) for r in rings])])
    return np.vstack(rings), offsets


class TestInsertion:
    def test_every_segment_lies_in_one_triangle_and_the_new_vertices_lie_on_the_segments(self, mesh):
        coords, offsets = random_rings(0)
        result = mesh.insert(coords, offsets)
        assert result is not None
        new = result.new_index
        assert len(new) > 0
        # the input vertices keep their order and values
        np.testing.assert_array_equal(np.delete(result.coords, new, axis=0), coords)
        assert result.ring_offsets[0] == 0
        assert result.ring_offsets[-1] == len(result.coords)
        np.testing.assert_array_equal(result.coords[result.ring_offsets[:-1]], coords[offsets[:-1]])
        # the new vertices lie at `fraction` on the segment they split
        expected = (1 - result.fraction[:, None]) * coords[result.left] + result.fraction[:, None] * coords[
            result.left + 1
        ]
        np.testing.assert_allclose(result.coords[new], expected, rtol=0, atol=1e-12)
        assert ((result.fraction > 0) & (result.fraction < 1)).all()
        # the new vertices of a segment are ordered along it
        for k in np.unique(result.left):
            f = result.fraction[result.left == k]
            assert (np.diff(f) > 0).all()
        # both end points of every sub-segment are in the triangle of its midpoint
        c = result.coords
        a, b = c[:-1], c[1:]
        inner = np.ones(len(c) - 1, bool)
        inner[result.ring_offsets[1:-1] - 1] = False
        a, b = a[inner], b[inner]
        tri, _ = mesh.locate(0.5 * (a + b))
        nodes = mesh.node_xy()
        for end in (a, b):
            w = barycentric(end, nodes[tri[:, 0]], nodes[tri[:, 1]], nodes[tri[:, 2]])
            assert w.min() > -1e-7

    def test_the_two_rings_of_a_shared_boundary_get_identical_vertices(self, mesh):
        rng = np.random.default_rng(4)
        shared = np.column_stack([np.linspace(2.0, 20.0, 6), 4.0 + rng.uniform(-1, 1, 6)])
        lower = np.vstack([shared, [[20.0, 0.5], [2.0, 0.5], shared[0]]])
        upper = np.vstack([shared[::-1], [[2.0, 7.5], [20.0, 7.5], shared[-1]]])
        coords = np.vstack([lower, upper])
        offsets = np.array([0, len(lower), len(coords)])
        result = mesh.insert(coords, offsets)
        c = result.coords
        first = c[result.ring_offsets[0] : result.ring_offsets[1]]
        second = c[result.ring_offsets[1] : result.ring_offsets[2]]
        # the boundary runs from vertex 0 to vertex 5 of `lower`; the vertices on it are the same in both rings
        end_lower = int(np.flatnonzero((first == shared[-1]).all(axis=1))[0])
        start_upper = int(np.flatnonzero((second == shared[-1]).all(axis=1))[0])
        end_upper = int(np.flatnonzero((second == shared[0]).all(axis=1))[0])
        boundary_lower = first[: end_lower + 1]
        boundary_upper = second[start_upper : end_upper + 1][::-1]
        assert len(boundary_lower) > len(shared)
        np.testing.assert_array_equal(boundary_lower, boundary_upper)

    def test_nothing_is_inserted_when_no_segment_crosses_an_edge(self):
        mesh = build_quadtree(np.ones((8, 8)), (0.0, 0.0, 1.0, 1.0))  # one leaf with four fan triangles
        coords = np.array([[1.0, 1.0], [1.5, 1.0], [1.5, 1.5], [1.0, 1.0]])
        assert mesh.insert(coords, np.array([0, 4])) is None


# ----------------------------------------------------------------------------
# The repair of inverted fans
# ----------------------------------------------------------------------------

# A leaf of the 2 x 2 tree squeezed into a sliver with a reflex corner (a failing leaf of a county run): the mean of
# its corners lies outside its kernel.
SLIVER = np.array([
    [-781881.0461652172, 545578.5063385332],
    [-780897.9576160266, 550021.5595253885],
    [-774463.3302028578, 565893.9776579767],
    [-778611.4078712035, 563253.5728460771],
])
DART = np.array([[0.0, 0.0], [6.0, 2.0], [1.0, 1.0], [0.0, 6.0]])
BOWTIE = np.array([[0.0, 0.0], [4.0, 4.0], [4.0, 0.0], [0.0, 4.0]])


@pytest.fixture(scope="module")
def four_leaves():
    return build_quadtree(np.array([[0.0, 1.0], [2.0, 3.0]]), (0.0, 0.0, 1.0, 1.0), 2.0**-6)


def deformed(mesh, corners):
    nodes = mesh.node_xy()
    nodes[mesh.corners[0]] = corners
    mesh.place_centers(nodes)
    return nodes


def in_kernel(corners, point):
    edges = np.roll(corners, -1, axis=0) - corners
    rel = point - corners
    return (edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0] > 0).all()


class TestKernelCenter:
    @pytest.mark.parametrize("corners", [DART, SLIVER], ids=["dart", "county sliver"])
    def test_the_mean_of_the_corners_inverts_the_fan_and_the_kernel_centroid_repairs_it(self, four_leaves, corners):
        nodes = deformed(four_leaves, corners)
        assert not in_kernel(corners, corners.mean(axis=0))
        assert 0 in four_leaves.inverted_leaves(nodes)
        fixed = four_leaves.fix_centers(nodes, np.array([0]))
        assert fixed.tolist() == [True]
        center = nodes[four_leaves.center[0]]
        assert in_kernel(corners, center)
        assert 0 not in four_leaves.inverted_leaves(nodes)
        assert (four_leaves.triangle_areas(nodes)[: four_leaves.n_boundary[0]] > 0).all()

    def test_a_leaf_without_a_kernel_is_left_alone(self, four_leaves):
        nodes = deformed(four_leaves, BOWTIE)
        before = nodes.copy()
        assert four_leaves.fix_centers(nodes, np.array([0])).tolist() == [False]
        np.testing.assert_array_equal(nodes, before)
        assert 0 in four_leaves.inverted_leaves(nodes)

    def test_a_valid_leaf_keeps_a_valid_fan_when_its_center_is_moved(self, four_leaves):
        nodes = deformed(four_leaves, np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]))
        assert four_leaves.fix_centers(nodes, np.array([0])).tolist() == [True]
        np.testing.assert_allclose(nodes[four_leaves.center[0]], [1.0, 1.0])


# ----------------------------------------------------------------------------
# Pruning of inserted vertices
# ----------------------------------------------------------------------------


def hash_maximum_at(x_values, y, left, right):
    """The first x of x_values for which the vertex (x, y) has a larger hash than both of its neighbors."""
    for x in x_values:
        h = _coordinate_hash(np.array([[x, y], left, right]))
        if h[0] > h[1] and h[0] > h[2]:
            return x
    raise AssertionError("no hash maximum found")


def prune(coords, offsets, is_input, tol=0.02, max_ring=64):
    keep = _prune_mask(
        np.asarray(coords, float), np.asarray(offsets, np.int64), np.asarray(is_input, bool), tol, max_ring
    )
    return np.ones(len(coords), bool) if keep is None else keep


class TestPruning:
    def test_only_inserted_vertices_are_removed_and_none_of_two_adjacent_ones(self):
        x = np.linspace(0, 10, 60)
        wobble = 0.004 * np.sin(40 * x)
        line = np.column_stack([x, wobble])
        coords = np.vstack([line, [[10.0, 5.0], [0.0, 5.0], line[0]]])
        offsets = np.array([0, len(coords)])
        is_input = np.zeros(len(coords), bool)
        is_input[[0, 59, 60, 61, 62]] = True
        is_input[30] = True  # an input vertex in the run
        keep = prune(coords, offsets, is_input)
        removed = np.flatnonzero(~keep)
        assert len(removed) > 10
        assert is_input[keep == 0].sum() == 0
        assert keep[is_input].all()
        assert (np.diff(removed) > 1).all()

    def test_a_run_of_collinear_inserted_vertices_keeps_the_end_points_and_the_shape(self):
        line = np.column_stack([np.linspace(0, 8, 33), np.zeros(33)])
        coords = np.vstack([line, [[8.0, 3.0], [0.0, 3.0], line[0]]])
        offsets = np.array([0, len(coords)])
        is_input = np.zeros(len(coords), bool)
        is_input[[0, 32, 33, 34, 35]] = True
        keep = prune(coords, offsets, is_input)
        assert 5 < (~keep).sum() < 17  # at most every other vertex
        kept = coords[keep]
        assert shapely.Polygon(kept).area == pytest.approx(shapely.Polygon(coords).area, rel=1e-12)
        assert shapely.is_valid(shapely.Polygon(kept))
        assert not find_crossing_segments(kept, np.array([0, len(kept)])).any()

    def test_the_chord_is_not_used_where_it_crosses_another_ring(self):
        # the lower ring A has a vertex m 0.015 above the chord u-w; a thin triangle B below pokes its apex up between
        # the chord and the vertex
        m_x = hash_maximum_at(np.linspace(4.9, 5.1, 400), 0.015, (0.0, 0.0), (10.0, 0.0))
        ring_a = np.array([[0.0, 0.0], [m_x, 0.015], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0], [0.0, 0.0]])
        ring_b = np.array([[m_x - 0.3, -1.0], [m_x + 0.3, -1.0], [m_x, 0.005], [m_x - 0.3, -1.0]])
        coords = np.vstack([ring_a, ring_b])
        offsets = np.array([0, 6, 10])
        is_input = np.ones(len(coords), bool)
        is_input[1] = False
        # without the apex the vertex is removed
        assert (~prune(ring_a, [0, 6], is_input[:6])).tolist() == [False, True, False, False, False, False]
        for max_ring in (64, 0):  # by the crossing alone (max_ring 0 skips the foreign-vertex test) and by both
            keep = prune(coords, offsets, is_input, max_ring=max_ring)
            assert keep.all()
        # B is a valid polygon below the chord, and removing the vertex would make the two rings overlap
        merged = Polygon(ring_a[[0, 2, 3, 4, 5]])
        assert merged.intersects(Polygon(ring_b))

    def test_a_small_ring_inside_the_sliver_blocks_the_removal_without_a_crossing(self):
        m_x = hash_maximum_at(np.linspace(4.9, 5.1, 400), 0.015, (0.0, 0.0), (10.0, 0.0))
        ring_a = np.array([[0.0, 0.0], [m_x, 0.015], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0], [0.0, 0.0]])
        island = np.array([[m_x - 0.001, 0.004], [m_x + 0.001, 0.004], [m_x, 0.006], [m_x - 0.001, 0.004]])
        coords = np.vstack([ring_a, island])
        offsets = np.array([0, 6, 10])
        is_input = np.ones(len(coords), bool)
        is_input[1] = False
        assert not find_crossing_segments(coords, offsets).any()
        assert prune(coords, offsets, is_input).all()
        # the test looks at rings of at most `max_ring` vertices only (a heuristic): the same island is not seen as a
        # ring of 4 vertices when the limit is 2
        assert not prune(coords, offsets, is_input, max_ring=2)[1]

    def test_the_two_rings_of_a_shared_boundary_decide_alike(self):
        rng = np.random.default_rng(5)
        x = np.linspace(0, 12, 80)
        shared = np.column_stack([x, 0.006 * rng.standard_normal(80)])
        shared[[0, -1], 1] = 0.0
        below = np.vstack([shared, [[12.0, -3.0], [0.0, -3.0], shared[0]]])
        above = np.vstack([shared[::-1], [[0.0, 3.0], [12.0, 3.0], shared[-1]]])
        coords = np.vstack([below, above])
        offsets = np.array([0, len(below), len(coords)])
        is_input = (np.abs(coords[:, 1]) > 1.0) | np.isin(coords[:, 0], [0.0, 12.0])  # all but the shared interior
        keep = prune(coords, offsets, is_input)
        assert (~keep).sum() > 4
        kept = coords[keep]
        on_boundary = np.abs(kept[:, 1]) < 1.0
        n_below = keep[: len(below)].sum()
        assert {tuple(p) for p in kept[:n_below][on_boundary[:n_below]]} == {
            tuple(p) for p in kept[n_below:][on_boundary[n_below:]]
        }
        assert not find_crossing_segments(kept, np.array([0, n_below, len(kept)])).any()

    def test_pruning_never_creates_a_crossing_and_keeps_the_input_vertices(self):
        rng = np.random.default_rng(6)
        # two neighbors with a shared wiggly boundary of inserted vertices, and a narrow third region close to it
        x = np.linspace(0, 20, 400)
        shared = np.column_stack([x, 0.012 * np.sin(1.3 * x) + 0.004 * rng.standard_normal(400)])
        shared[[0, -1], 1] = 0.0
        below = np.vstack([shared, [[20.0, -4.0], [0.0, -4.0], shared[0]]])
        above = np.vstack([shared[::-1], [[0.0, 0.03], [20.0, 0.03], shared[-1]]])  # a strip of thickness 0.03
        coords = np.vstack([below, above])
        offsets = np.array([0, len(below), len(coords)])
        is_input = np.zeros(len(coords), bool)
        for start, size in ((0, len(below)), (len(below), len(above))):
            is_input[start + np.array([0, size - 1])] = True
        is_input[len(below) - 3 : len(below)] = True
        is_input[len(below) + 400 : len(below) + 403] = True
        assert not find_crossing_segments(coords, offsets).any()
        keep = prune(coords, offsets, is_input, tol=0.02)
        assert keep[is_input].all()
        new_offsets = np.cumsum([0, keep[: len(below)].sum(), keep[len(below) :].sum()])
        assert not find_crossing_segments(coords[keep], new_offsets).any()


# ----------------------------------------------------------------------------
# The lineage
# ----------------------------------------------------------------------------


class TestLineage:
    def test_lift_with_removals_and_per_vertex_fractions(self):
        # [0, 4, 8, 0] -> insert at 1/4 and 3/4 of the first segment -> remove the second inserted vertex
        insert = InsertionRound(np.array([1, 2]), np.array([0, 0]), np.array([0.25, 0.75]), 1)
        remove = RemovalRound(np.array([0, 1, 3, 4, 5]), 2)
        lineage = VertexLineage(4, (insert, remove))
        coords = np.array([[0.0], [4.0], [8.0], [0.0]])
        assert lineage._counts() == [4, 6, 5]
        assert lineage.n_vertices == 5
        assert lineage.lift(coords)[:, 0].tolist() == [0.0, 1.0, 4.0, 8.0, 0.0]
        # the intermediate state has all inserted vertices; lifting it applies the removal only
        middle = np.array([[0.0], [1.0], [3.0], [4.0], [8.0], [0.0]])
        np.testing.assert_allclose(lineage.lift(middle), [[0.0], [1.0], [4.0], [8.0], [0.0]])
        with pytest.raises(ValueError, match="not a state"):
            lineage.lift(np.zeros((3, 1)))

    def test_a_vertex_count_that_identifies_two_states_must_be_disambiguated(self):
        insert = InsertionRound(np.array([1]), np.array([0]), np.array([0.5]), 1)
        remove = RemovalRound(np.array([0, 2, 3, 4]), 2)  # removes the vertex that the first round inserted
        lineage = VertexLineage(4, (insert, remove))
        coords = np.array([[0.0], [4.0], [8.0], [0.0]])
        with pytest.raises(ValueError, match="pass state"):
            lineage.lift(coords)
        np.testing.assert_allclose(lineage.lift(coords, state=0), coords)
        np.testing.assert_allclose(lineage.lift(coords, state=2), coords)
        with pytest.raises(ValueError, match="not the state"):
            lineage.lift(coords, state=1)

    def test_removals_keep_the_input_vertices_and_their_order_in_a_morph(self):
        polygons, result = morph(11, validity="quadtree")
        lineage = result.vertex_lineage
        assert any(isinstance(r, RemovalRound) for r in lineage.rounds)
        assert any(isinstance(r, InsertionRound) for r in lineage.rounds)
        # follow the position of every input vertex through the rounds: none is removed, the order is kept
        start = shapely.get_coordinates(np.array(polygons, dtype=object))
        position = np.arange(len(start))
        count = len(start)
        for r in lineage.rounds:
            if isinstance(r, InsertionRound):
                count += len(r.new_index)
                position = np.setdiff1d(np.arange(count), r.new_index)[position]
            else:
                kept_position = np.searchsorted(r.kept_index, position)
                assert (r.kept_index[kept_position] == position).all()
                position = kept_position
                count = len(r.kept_index)
        assert (np.diff(position) > 0).all()
        assert position.max() < lineage.n_vertices
        # the lifted input polygons are the input polygons
        final = unpack_geometries(result.best.geometry)
        lifted = result.lift_coordinates(start)
        assert lifted.shape == (lineage.n_vertices, 2)
        np.testing.assert_array_equal(lifted[position], start)
        moved = unpack_geometries(polygons)
        moved.set_vertices(lifted, final.ragged_offsets[0])
        for original, new in zip(polygons, reconstruct_geometries(moved), strict=True):
            assert new.area == pytest.approx(original.area, rel=1e-12)


# ----------------------------------------------------------------------------
# The hooks on a window
# ----------------------------------------------------------------------------


def ribbon() -> Polygon:
    """A strip of height 1 with sparse vertices; the top and the bottom have different vertices."""
    top = [(x, 1.0) for x in (0.0, 3.3, 6.6, 10.0)]
    bottom = [(x, 0.0) for x in (10.0, 5.0, 0.0)]
    return Polygon(top + bottom)


@pytest.fixture(scope="module")
def squeeze():
    """Grid, a random density and a field that moves every point to 2 % of its distance from a curve in one step."""
    grid = Grid.from_bounds((0, -1, 10, 2), size=400, margin=0.0, square=False)
    x, y = np.meshgrid(grid.x_coords, grid.y_coords)
    center = 0.5 + 0.5 * np.sin(2 * np.pi * x / 10)
    rho = bumpy_field(len(grid.y_coords), len(grid.x_coords), seed=1)
    return grid, rho, np.zeros_like(y), -(y - center) * 0.98


def drive(method, geoms, grid, rho, fields, dt=1.0, steps=1):
    """Run the hooks of the morph loop for the given fields."""
    best_state = None
    iteration = 0
    for vx, vy in fields:
        best_state = method.begin_window(geoms, grid, best_state, iteration)
        method.set_density(rho)
        method.add_field(vx, vy)
        for _ in range(steps):
            method.record_step(dt)
            geoms.coords = method.displace(grid, vx, vy, dt)
            geoms.invalidate_cache()
            iteration += 1
    return method.finish(geoms, grid, best_state, iteration)


class TestWindow:
    def test_the_ribbon_is_valid_after_a_squeeze_that_crosses_its_segments(self, squeeze):
        grid, rho, vx, vy = squeeze
        plain = unpack_geometries([ribbon()])
        plain.coords = displace_coords_numba(
            plain.coords, grid.x_coords, grid.y_coords, vx, vy, 1.0, grid.dx, grid.dy, use_parallel=False
        )
        assert not shapely.is_valid(reconstruct_geometries(plain)[0])
        geoms = unpack_geometries([ribbon()])
        method = QuadtreeWindows(len(geoms.coords))
        start = geoms.coords.copy()
        drive(method, geoms, grid, rho, [(vx, vy)])
        assert shapely.is_valid(reconstruct_geometries(geoms)[0])
        assert not find_crossing_segments(geoms.coords, geoms.ragged_offsets[0]).any()
        assert method.n_unresolved == 0
        lineage = method.lineage()
        assert lineage.rounds and lineage.n_vertices == len(geoms.coords)
        assert len(geoms.coords) > len(start)

    def test_the_vertices_before_the_window_end_are_valid_at_every_iteration(self, squeeze):
        grid, rho, _, _ = squeeze
        x, y = np.meshgrid(grid.x_coords, grid.y_coords)
        center = 0.5 + 0.5 * np.sin(2 * np.pi * x / 10)
        vx, vy = np.zeros_like(y), -(y - center) * 0.2
        geoms = unpack_geometries([ribbon()])
        method = QuadtreeWindows(len(geoms.coords))
        method.begin_window(geoms, grid, None, 0)
        method.set_density(rho)
        method.add_field(vx, vy)
        for _ in range(12):
            method.record_step(1.0)
            geoms.coords = method.displace(grid, vx, vy, 1.0)
            assert not find_crossing_segments(geoms.coords, geoms.ragged_offsets[0]).any()
        assert method.n_rebuilds == 1

    def test_polygons_with_holes_and_parts_keep_their_structure(self, squeeze):
        grid, rho, vx, vy = squeeze
        holed = shapely.segmentize(
            Polygon([(0, -0.5), (10, -0.5), (10, -0.9), (0, -0.9)], [[(1, -0.6), (1, -0.8), (2, -0.8), (2, -0.6)]]),
            0.25,
        )
        parts = MultiPolygon([ribbon(), box(0, 1.2, 1, 1.4)])
        geometries = [holed, parts, box(5, 1.5, 6, 1.8)]
        geoms = unpack_geometries(geometries)
        method = QuadtreeWindows(len(geoms.coords))
        drive(method, geoms, grid, rho, [(vx, vy)])
        rebuilt = reconstruct_geometries(geoms)
        assert [g.geom_type for g in rebuilt] == ["Polygon", "MultiPolygon", "Polygon"]
        assert len(rebuilt[0].interiors) == 1
        assert [len(g.geoms) for g in rebuilt if g.geom_type == "MultiPolygon"] == [2]
        assert sum(len(shapely.get_coordinates(g)) for g in rebuilt) == len(geoms.coords)
        again = unpack_geometries(rebuilt)
        assert again.metadata == geoms.metadata
        assert (again.ragged_offsets[0] == geoms.ragged_offsets[0]).all()
        assert all(shapely.is_valid(g) for g in rebuilt)

    def test_the_mesh_is_kept_until_a_fix_or_the_area_rule_asks_for_a_new_one(self, squeeze):
        grid, rho, _, _ = squeeze
        zero = np.zeros_like(rho)
        geoms = unpack_geometries([ribbon()])
        method = QuadtreeWindows(len(geoms.coords))
        drive(method, geoms, grid, rho, [(zero, zero)] * 4)
        assert method.n_rebuilds == 1  # nothing moved
        method._dirty = True
        drive(method, geoms, grid, rho, [(zero, zero)])
        assert method.n_rebuilds == 2
        # squeezing a window much further than the area rule allows rebuilds at the next window
        _, y = np.meshgrid(grid.x_coords, grid.y_coords)
        vy = -(y - 0.5) * 0.9999
        drive(method, geoms, grid, rho, [(zero, vy), (zero, zero)])
        assert method.n_rebuilds >= 3

    def test_an_inverted_fan_is_repaired_or_limited_and_counted_when_it_survives(self):
        grid = Grid.from_bounds((0, 0, 10, 10), size=128, margin=0.0, square=True)
        y, x = np.meshgrid(grid.y_coords, grid.x_coords, indexing="ij")
        rho = 1 + np.sin(x) * np.cos(y)

        def run(seed, amplitude):
            rng = np.random.default_rng(seed)
            k = rng.uniform(0.3, 1.5, 4)
            phase = rng.uniform(0, 6.28, 4)
            scale = amplitude * grid.dx * 60
            vx = scale * np.sin(k[0] * x + phase[0]) * np.cos(k[1] * y + phase[1])
            vy = scale * np.sin(k[2] * x + phase[2]) * np.cos(k[3] * y + phase[3])
            geoms = unpack_geometries([shapely.segmentize(box(1, 1, 9, 9), 0.3)])
            method = QuadtreeWindows(len(geoms.coords))
            method.begin_window(geoms, grid, None, 0)
            method.set_density(rho)
            method.add_field(vx, vy)
            for _ in range(10):
                method.record_step(1.0)
                geoms.coords = method.displace(grid, vx, vy, 1.0)
            return method, geoms

        method, geoms = run(1, 0.05)  # leaves are repaired or limited
        assert method.n_fixed > 0
        assert method.n_unresolved == 0
        assert len(method._mesh.inverted_leaves(method._nodes)) == 0
        assert method._dirty
        assert shapely.is_valid(reconstruct_geometries(geoms)[0])
        # a stronger field leaves inversions behind: they are counted, logged and reported at the end
        method, geoms = run(3, 0.1)
        assert method.n_unresolved > 0
        assert method.n_unresolved_leaves >= method.n_unresolved
        with pytest.warns(UserWarning, match="inverted mesh triangle"):
            method.finish(geoms, None, None, 10)

    def test_the_best_iterate_goes_back_to_its_own_vertex_set(self, squeeze):
        grid, rho, _, _ = squeeze
        x, y = np.meshgrid(grid.x_coords, grid.y_coords)
        vx, vy = np.zeros_like(y), -(y - (0.5 + 0.5 * np.sin(2 * np.pi * x / 10))) * 0.3
        geoms = unpack_geometries([ribbon()])
        method = QuadtreeWindows(len(geoms.coords))
        method.begin_window(geoms, grid, None, 0)
        method.set_density(rho)
        method.add_field(vx, vy)
        method.record_step(1.0)
        geoms.coords = method.displace(grid, vx, vy, 1.0)
        method.mark_best()
        best_coords = geoms.coords.copy()
        best_offsets = geoms.ragged_offsets[0].copy()
        n_rounds = len(method.lineage().rounds)
        # a second window prunes and inserts again
        method._dirty = True
        method.begin_window(geoms, grid, None, 1)
        method.set_density(rho)
        method.add_field(vx, vy)
        method.record_step(1.0)
        geoms.coords = method.displace(grid, vx, vy, 1.0)
        assert len(method.lineage().rounds) > n_rounds or len(geoms.coords) != len(best_coords)
        method.restore_best(geoms, best_coords)
        np.testing.assert_array_equal(geoms.coords, best_coords)
        np.testing.assert_array_equal(geoms.ragged_offsets[0], best_offsets)
        assert len(method.lineage().rounds) == n_rounds
        assert method.lineage().n_vertices == len(best_coords)


# ----------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------


class TestOptions:
    def test_quadtree_is_accepted_next_to_windows_and_none(self):
        for value in ("none", "windows", "quadtree"):
            assert MorphOptions(validity=value).validity == value
        assert MorphOptions(validity="quadtree").copy_with(validity="windows").validity == "windows"

    @pytest.mark.parametrize("value", ["both", "quadtree,windows", "Quadtree", True, None])
    def test_other_values_are_rejected(self, value):
        with pytest.raises(MorphOptionsValidationError, match="validity"):
            MorphOptions(validity=value)
        options = MorphOptions()
        with pytest.raises(MorphOptionsValidationError, match='"quadtree"'):
            options.validity = value

    def test_each_value_selects_one_method(self, monkeypatch):
        import carto_flow.flow_cartogram.algorithm as algorithm

        def forbidden(name):
            def raise_(*args, **kwargs):
                raise AssertionError(f"{name} is used")

            return raise_

        monkeypatch.setattr(algorithm, "ValidatedWindows", forbidden("ValidatedWindows"))
        polygons, values = strip_scene(11)
        options = MorphOptions(grid=64, n_iter=20, show_progress=False, validity="quadtree")
        assert morph_geometries(polygons, values, options=options).vertex_lineage is not None
        monkeypatch.undo()
        monkeypatch.setattr(algorithm, "QuadtreeWindows", forbidden("QuadtreeWindows"))
        for validity in ("none", "windows"):
            options = MorphOptions(grid=64, n_iter=20, show_progress=False, validity=validity)
            morph_geometries(polygons, values, options=options)


# ----------------------------------------------------------------------------
# The morph
# ----------------------------------------------------------------------------

# seeds whose baseline run ends with an invalid polygon, by the way the run ends
ENDS = {"n_iter": 11, "converged": 46, "stalled": 59}


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


def n_vertices_of(geometries):
    return len(shapely.get_coordinates(np.array(list(geometries), dtype=object)))


class TestMorph:
    @pytest.mark.parametrize("seed", [11, 31, 33])
    def test_a_squeezed_strip_is_invalid_without_and_valid_with_quadtree_windows(self, seed):
        _, baseline = morph(seed)
        assert invalid_count(baseline) >= 1
        polygons, result = morph(seed, validity="quadtree")
        assert invalid_count(result) == 0
        assert result.invalid_regions == ()
        assert all(shapely.is_valid(s.geometry).all() for s in result.snapshots.snapshots)
        lineage = result.vertex_lineage
        assert lineage.n_input == n_vertices_of(polygons)
        assert lineage.n_vertices == n_vertices_of(result.best.geometry) > lineage.n_input

    @pytest.mark.parametrize("how", sorted(ENDS))
    def test_the_best_iterate_and_the_final_state_are_valid_whatever_the_end_of_the_run(self, how):
        seed = ENDS[how]
        _, baseline = morph(seed)
        assert invalid_count(baseline) >= 1
        _, result = morph(seed, validity="quadtree")
        expected = {"n_iter": "COMPLETED", "converged": "CONVERGED", "stalled": "STALLED"}[how]
        assert result.status.name == expected
        assert invalid_count(result) == 0
        assert result.vertex_lineage.n_vertices == n_vertices_of(result.best.geometry)
        # the last stored snapshot is the final iterate; it is valid as well
        assert shapely.is_valid(np.array(result.last.geometry, dtype=object)).all()

    def test_the_returned_best_iterate_has_its_own_vertex_set_and_errors(self):
        _, result = morph(ENDS["stalled"], validity="quadtree")
        assert result.best_iteration < result.niterations
        polygons, values = strip_scene(ENDS["stalled"])
        areas = np.array([g.area for g in result.best.geometry])
        target = values / values.sum() * sum(g.area for g in polygons)
        expected = compute_error_metrics(areas, target, eps=0.0)
        np.testing.assert_allclose(result.best.errors.log_errors, expected.log_errors, rtol=1e-9, atol=1e-12)
        # the returned geometry and the lineage describe the same vertex set; the final iterate has another one
        assert result.vertex_lineage.n_vertices == n_vertices_of(result.best.geometry)

    def test_snapshots_before_the_best_iterate_lift_to_the_returned_vertex_set(self):
        _, result = morph(11, validity="quadtree", snapshot_every=5)
        final = n_vertices_of(result.best.geometry)
        lifted = 0
        for snapshot in result.snapshots.snapshots:
            if snapshot.iteration > result.best_iteration:
                continue
            coords = shapely.get_coordinates(np.array(snapshot.geometry, dtype=object))
            try:
                out = result.lift_coordinates(coords)
            except ValueError:  # the number of vertices is shared by two states of the run
                continue
            assert len(out) == final
            lifted += 1
        assert lifted > 3

    def test_landmarks_probes_and_coords_are_not_affected(self):
        polygons, values = strip_scene(11)
        landmarks = [Point(1.0, 0.5), Point(4.0, 2.0)]
        coords = np.array([[1.0, 1.0], [3.0, 2.0], [5.0, 1.0]])
        options = MorphOptions(
            grid=64, n_iter=300, mean_tol=0.02, max_tol=0.05, show_progress=False, validity="quadtree"
        )
        result = morph_geometries(polygons, values, landmarks=landmarks, coords=coords, options=options)
        assert result.vertex_lineage.rounds
        assert len(result.best.landmarks) == 2
        assert result.best.coords.shape == coords.shape
        # the coordinates moved with the field
        assert not np.allclose(result.best.coords, coords)

    def test_each_level_of_a_ladder_has_its_own_lineage(self):
        import geopandas as gpd

        polygons, values = strip_scene(11)
        gdf = gpd.GeoDataFrame({"v": values}, geometry=polygons)
        options = [
            MorphOptions(n_iter=100, mean_tol=0.02, max_tol=0.05, show_progress=False, validity="quadtree")
            for _ in range(2)
        ]
        workflow = CartogramWorkflow(gdf, "v", validate=False)
        workflow.morph_multiresolution(min_resolution=32, levels=2, options=options)
        levels = [c for c in workflow if c.vertex_lineage is not None]
        assert len(levels) == 2
        counts = [n_vertices_of(c.best.geometry) for c in levels]
        assert levels[0].vertex_lineage.n_input == len(shapely.get_coordinates(gdf.geometry.values))
        assert levels[1].vertex_lineage.n_input == counts[0]
        assert levels[1].vertex_lineage.n_vertices == counts[1]

    def test_polygons_with_holes_and_parts_morph_to_valid_polygons(self):
        outer = shapely.segmentize(box(0, 0, 6, 4), 0.7)
        holed = Polygon(outer.exterior.coords, [shapely.segmentize(box(2, 1, 3, 2), 0.3).exterior.coords])
        parts = MultiPolygon([shapely.segmentize(box(6, 0, 9, 2), 0.7), shapely.segmentize(box(6, 2.5, 8, 4), 0.7)])
        single = shapely.segmentize(box(9, 0, 12, 4), 0.7)
        geometries = [holed, parts, single]
        options = MorphOptions(
            grid=64, n_iter=150, mean_tol=0.02, max_tol=0.05, show_progress=False, validity="quadtree"
        )
        result = morph_geometries(geometries, np.array([5.0, 0.4, 4.0]), options=options)
        assert result.invalid_regions == ()
        rebuilt = list(result.best.geometry)
        assert [g.geom_type for g in rebuilt] == ["Polygon", "MultiPolygon", "Polygon"]
        assert len(rebuilt[0].interiors) == 1
        assert len(rebuilt[1].geoms) == 2
        assert result.vertex_lineage.n_vertices == n_vertices_of(rebuilt)
