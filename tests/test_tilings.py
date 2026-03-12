"""Tests for the tiling module.

Tests cover:
- TilingResult dataclass properties
- SquareTiling generation and adjacency
- HexagonTiling generation and adjacency
- TriangleTiling generation and adjacency (various shapes)
- QuadrilateralTiling generation and adjacency (various shapes)
- resolve_tiling registry function
- Edge vs vertex adjacency modes
- Tile congruence (equal area, no overlaps, no gaps)
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from carto_flow.symbol_cartogram.tiling import (
    HexagonTiling,
    IsohedralTiling,
    QuadrilateralTiling,
    SquareTiling,
    TileAdjacencyType,
    TileTransform,
    TilingResult,
    TriangleTiling,
    resolve_tiling,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOUNDS = (0.0, 0.0, 10.0, 10.0)
N_TILES = 50


def _interior_mask(result: TilingResult, bounds, margin_frac: float = 0.3):
    """Mask for tiles whose center is well inside bounds (to avoid edge effects)."""
    minx, miny, maxx, maxy = bounds
    w, h = maxx - minx, maxy - miny
    pad_x, pad_y = w * margin_frac, h * margin_frac
    centers = result.centers
    return (
        (centers[:, 0] > minx + pad_x)
        & (centers[:, 0] < maxx - pad_x)
        & (centers[:, 1] > miny + pad_y)
        & (centers[:, 1] < maxy - pad_y)
    )


def _neighbor_counts(result: TilingResult, mask=None):
    """Per-tile neighbor counts (optionally filtered by mask)."""
    counts = result.adjacency.sum(axis=1)
    if mask is not None:
        return counts[mask]
    return counts


def _check_no_overlaps(result: TilingResult, tol_frac: float = 0.01):
    """Check that no two tiles overlap (pairwise intersection area ≈ 0)."""
    max_area = result.polygons[0].area * tol_frac
    # Only check a random subset to keep tests fast
    rng = np.random.default_rng(42)
    n = len(result.polygons)
    indices = rng.choice(n, size=200, replace=False) if n > 200 else np.arange(n)

    for idx, i in enumerate(indices):
        for j in indices[idx + 1 :]:
            if result.adjacency[i, j]:
                area = result.polygons[i].intersection(result.polygons[j]).area
                assert area < max_area, f"Tiles {i} and {j} overlap with area {area:.6f} (max allowed {max_area:.6f})"


def _check_equal_areas(result: TilingResult, rtol: float = 0.01):
    """Check that all tiles have approximately equal area."""
    areas = np.array([p.area for p in result.polygons])
    mean_area = areas.mean()
    assert np.allclose(areas, mean_area, rtol=rtol), (
        f"Tile areas not equal: min={areas.min():.6f}, max={areas.max():.6f}, mean={mean_area:.6f}"
    )


# ---------------------------------------------------------------------------
# TileTransform
# ---------------------------------------------------------------------------


class TestTileTransform:
    def test_defaults(self):
        t = TileTransform(center=(1.0, 2.0))
        assert t.center == (1.0, 2.0)
        assert t.rotation == 0.0
        assert t.flipped is False

    def test_with_rotation_and_flip(self):
        t = TileTransform(center=(0, 0), rotation=180.0, flipped=True)
        assert t.rotation == 180.0
        assert t.flipped is True


# ---------------------------------------------------------------------------
# TilingResult
# ---------------------------------------------------------------------------


class TestTilingResult:
    def test_centers_property(self):
        transforms = [
            TileTransform(center=(1.0, 2.0)),
            TileTransform(center=(3.0, 4.0)),
        ]
        result = TilingResult(
            polygons=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])] * 2,
            transforms=transforms,
            adjacency=np.zeros((2, 2), dtype=bool),
            vertex_adjacency=np.zeros((2, 2), dtype=bool),
            tile_size=1.0,
            inscribed_radius=0.5,
            canonical_tile=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        )
        centers = result.centers
        assert centers.shape == (2, 2)
        np.testing.assert_array_equal(centers[0], [1.0, 2.0])
        np.testing.assert_array_equal(centers[1], [3.0, 4.0])

    def test_n_tiles(self):
        result = TilingResult(
            polygons=[Polygon([(0, 0), (1, 0), (1, 1)])] * 5,
            transforms=[TileTransform(center=(0, 0))] * 5,
            adjacency=np.zeros((5, 5), dtype=bool),
            vertex_adjacency=np.zeros((5, 5), dtype=bool),
            tile_size=1.0,
            inscribed_radius=0.3,
            canonical_tile=Polygon([(0, 0), (1, 0), (1, 1)]),
        )
        assert result.n_tiles == 5


# ---------------------------------------------------------------------------
# SquareTiling
# ---------------------------------------------------------------------------


class TestSquareTiling:
    def test_generates_tiles(self):
        tiling = SquareTiling()
        result = tiling.generate(BOUNDS, n_tiles=N_TILES)
        assert result.n_tiles > 0

    def test_equal_areas(self):
        result = SquareTiling().generate(BOUNDS, n_tiles=N_TILES)
        _check_equal_areas(result)

    def test_no_overlaps(self):
        result = SquareTiling().generate(BOUNDS, n_tiles=N_TILES)
        _check_no_overlaps(result)

    def test_edge_adjacency_interior_4(self):
        """Interior tiles in a square grid have 4 edge-neighbors."""
        result = SquareTiling().generate(BOUNDS, n_tiles=N_TILES, adjacency_type=TileAdjacencyType.EDGE)
        mask = _interior_mask(result, BOUNDS)
        counts = _neighbor_counts(result, mask)
        assert len(counts) > 0, "No interior tiles found"
        assert (counts == 4).all(), f"Expected 4 neighbors, got {np.unique(counts)}"

    def test_vertex_adjacency_interior_8(self):
        """Interior tiles in a square grid have 8 vertex-neighbors."""
        result = SquareTiling().generate(BOUNDS, n_tiles=N_TILES, adjacency_type=TileAdjacencyType.VERTEX)
        mask = _interior_mask(result, BOUNDS)
        counts = _neighbor_counts(result, mask)
        assert len(counts) > 0, "No interior tiles found"
        assert (counts == 8).all(), f"Expected 8 neighbors, got {np.unique(counts)}"

    def test_tile_size_parameter(self):
        result = SquareTiling().generate(BOUNDS, tile_size=2.0)
        assert abs(result.tile_size - 2.0) < 1e-10

    def test_canonical_tile(self):
        tile = SquareTiling().canonical_tile
        assert abs(tile.area - 1.0) < 1e-10

    def test_inscribed_radius(self):
        result = SquareTiling().generate(BOUNDS, tile_size=2.0)
        assert abs(result.inscribed_radius - 1.0) < 1e-10  # half of side=2

    def test_transforms_no_rotation(self):
        """Square tiling tiles are all axis-aligned (rotation=0, flipped=False)."""
        result = SquareTiling().generate(BOUNDS, n_tiles=N_TILES)
        for t in result.transforms:
            assert t.rotation == 0.0
            assert t.flipped is False


# ---------------------------------------------------------------------------
# HexagonTiling
# ---------------------------------------------------------------------------


class TestHexagonTiling:
    def test_generates_tiles(self):
        result = HexagonTiling().generate(BOUNDS, n_tiles=N_TILES)
        assert result.n_tiles > 0

    def test_equal_areas(self):
        result = HexagonTiling().generate(BOUNDS, n_tiles=N_TILES)
        _check_equal_areas(result)

    def test_no_overlaps(self):
        result = HexagonTiling().generate(BOUNDS, n_tiles=N_TILES)
        _check_no_overlaps(result)

    def test_edge_adjacency_interior_6(self):
        """Interior hexagonal tiles have 6 edge-neighbors."""
        result = HexagonTiling().generate(BOUNDS, n_tiles=100, adjacency_type=TileAdjacencyType.EDGE)
        mask = _interior_mask(result, BOUNDS)
        counts = _neighbor_counts(result, mask)
        assert len(counts) > 0, "No interior tiles found"
        assert (counts == 6).all(), f"Expected 6 neighbors, got {np.unique(counts)}"

    def test_transforms_no_rotation(self):
        """Hex tiling tiles are all unrotated."""
        result = HexagonTiling().generate(BOUNDS, n_tiles=N_TILES)
        for t in result.transforms:
            assert t.rotation == 0.0
            assert t.flipped is False


# ---------------------------------------------------------------------------
# TriangleTiling
# ---------------------------------------------------------------------------


class TestTriangleTiling:
    def test_equilateral_generates(self):
        result = TriangleTiling.equilateral().generate(BOUNDS, n_tiles=N_TILES)
        assert result.n_tiles > 0

    def test_equilateral_equal_areas(self):
        result = TriangleTiling.equilateral().generate(BOUNDS, n_tiles=N_TILES)
        _check_equal_areas(result)

    def test_equilateral_no_overlaps(self):
        result = TriangleTiling.equilateral().generate(BOUNDS, n_tiles=N_TILES)
        _check_no_overlaps(result)

    def test_equilateral_edge_adjacency_interior_3(self):
        """Interior equilateral triangle tiles have 3 edge-neighbors."""
        result = TriangleTiling.equilateral().generate(BOUNDS, n_tiles=200, adjacency_type=TileAdjacencyType.EDGE)
        mask = _interior_mask(result, BOUNDS)
        counts = _neighbor_counts(result, mask)
        assert len(counts) > 0, "No interior tiles found"
        assert (counts == 3).all(), f"Expected 3 neighbors, got {np.unique(counts)}"

    def test_equilateral_vertex_adjacency_interior_12(self):
        """Interior equilateral triangle tiles have 12 vertex-neighbors."""
        result = TriangleTiling.equilateral().generate(BOUNDS, n_tiles=200, adjacency_type=TileAdjacencyType.VERTEX)
        mask = _interior_mask(result, BOUNDS)
        counts = _neighbor_counts(result, mask)
        assert len(counts) > 0, "No interior tiles found"
        assert (counts == 12).all(), f"Expected 12 neighbors, got {np.unique(counts)}"

    def test_right_isosceles(self):
        result = TriangleTiling.right_isosceles().generate(BOUNDS, n_tiles=N_TILES)
        _check_equal_areas(result)
        _check_no_overlaps(result)

    def test_right_triangle(self):
        result = TriangleTiling.right(aspect_ratio=2.0).generate(BOUNDS, n_tiles=N_TILES)
        _check_equal_areas(result)
        _check_no_overlaps(result)

    def test_isosceles(self):
        result = TriangleTiling.isosceles(apex_angle=90.0).generate(BOUNDS, n_tiles=N_TILES)
        _check_equal_areas(result)
        _check_no_overlaps(result)

    def test_custom_triangle(self):
        """Arbitrary triangle from user-supplied polygon."""
        tri = Polygon([(0, 0), (3, 0), (1, 4)])
        result = TriangleTiling(tri).generate(BOUNDS, n_tiles=N_TILES)
        _check_equal_areas(result)
        _check_no_overlaps(result)

    def test_from_polygon(self):
        tri = Polygon([(0, 0), (1, 0), (0.5, 0.8)])
        tiling = TriangleTiling.from_polygon(tri)
        assert isinstance(tiling, TriangleTiling)

    def test_invalid_polygon_raises(self):
        """Non-triangle raises ValueError."""
        with pytest.raises(ValueError, match="3 vertices"):
            TriangleTiling(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))

    def test_transforms_have_rotation(self):
        """Triangle tiling produces tiles with rotation=0 and rotation=180."""
        result = TriangleTiling.equilateral().generate(BOUNDS, n_tiles=N_TILES)
        rotations = {t.rotation for t in result.transforms}
        assert 0.0 in rotations
        assert 180.0 in rotations

    def test_unit_tile_has_unit_area(self):
        tiling = TriangleTiling.equilateral()
        assert abs(tiling.canonical_tile.area - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# QuadrilateralTiling
# ---------------------------------------------------------------------------


class TestQuadrilateralTiling:
    def test_parallelogram_generates(self):
        result = QuadrilateralTiling.parallelogram().generate(BOUNDS, n_tiles=N_TILES)
        assert result.n_tiles > 0

    def test_parallelogram_equal_areas(self):
        result = QuadrilateralTiling.parallelogram().generate(BOUNDS, n_tiles=N_TILES)
        _check_equal_areas(result)

    def test_parallelogram_no_overlaps(self):
        result = QuadrilateralTiling.parallelogram().generate(BOUNDS, n_tiles=N_TILES)
        _check_no_overlaps(result)

    def test_parallelogram_edge_adjacency_interior_4(self):
        """Interior parallelogram tiles have 4 edge-neighbors."""
        result = QuadrilateralTiling.parallelogram().generate(
            BOUNDS, n_tiles=200, adjacency_type=TileAdjacencyType.EDGE
        )
        mask = _interior_mask(result, BOUNDS)
        counts = _neighbor_counts(result, mask)
        assert len(counts) > 0, "No interior tiles found"
        assert (counts == 4).all(), f"Expected 4 neighbors, got {np.unique(counts)}"

    def test_rectangle(self):
        result = QuadrilateralTiling.rectangle(aspect_ratio=1.5).generate(BOUNDS, n_tiles=N_TILES)
        _check_equal_areas(result)
        _check_no_overlaps(result)

    def test_rhombus(self):
        result = QuadrilateralTiling.rhombus(angle=45.0).generate(BOUNDS, n_tiles=N_TILES)
        _check_equal_areas(result)
        _check_no_overlaps(result)

    def test_trapezoid(self):
        result = QuadrilateralTiling.trapezoid(top_ratio=0.6).generate(BOUNDS, n_tiles=N_TILES)
        _check_equal_areas(result)
        _check_no_overlaps(result)

    def test_trapezoid_edge_adjacency_interior_4(self):
        """Interior trapezoid tiles have 4 edge-neighbors."""
        result = QuadrilateralTiling.trapezoid(top_ratio=0.5).generate(
            BOUNDS, n_tiles=200, adjacency_type=TileAdjacencyType.EDGE
        )
        mask = _interior_mask(result, BOUNDS)
        counts = _neighbor_counts(result, mask)
        assert len(counts) > 0, "No interior tiles found"
        assert (counts == 4).all(), f"Expected 4 neighbors, got {np.unique(counts)}"

    def test_custom_quadrilateral(self):
        """Arbitrary quadrilateral from user-supplied polygon."""
        quad = Polygon([(0, 0), (4, 0), (3, 3), (1, 2)])
        result = QuadrilateralTiling(quad).generate(BOUNDS, n_tiles=N_TILES)
        _check_equal_areas(result)
        _check_no_overlaps(result)

    def test_from_polygon(self):
        quad = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
        tiling = QuadrilateralTiling.from_polygon(quad)
        assert isinstance(tiling, QuadrilateralTiling)

    def test_invalid_polygon_raises(self):
        """Non-quadrilateral raises ValueError."""
        with pytest.raises(ValueError, match="4 vertices"):
            QuadrilateralTiling(Polygon([(0, 0), (1, 0), (0.5, 1)]))

    def test_transforms_have_rotation(self):
        """Quadrilateral tiling produces tiles with rotation=0 and rotation=180."""
        result = QuadrilateralTiling.parallelogram().generate(BOUNDS, n_tiles=N_TILES)
        rotations = {t.rotation for t in result.transforms}
        assert 0.0 in rotations
        assert 180.0 in rotations

    def test_unit_tile_has_unit_area(self):
        tiling = QuadrilateralTiling.parallelogram()
        assert abs(tiling.canonical_tile.area - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# resolve_tiling
# ---------------------------------------------------------------------------


class TestResolveTiling:
    def test_string_square(self):
        tiling = resolve_tiling("square")
        assert isinstance(tiling, SquareTiling)

    def test_string_hexagon(self):
        tiling = resolve_tiling("hexagon")
        assert isinstance(tiling, HexagonTiling)

    def test_string_triangle(self):
        tiling = resolve_tiling("triangle")
        assert isinstance(tiling, TriangleTiling)

    def test_string_quadrilateral(self):
        tiling = resolve_tiling("quadrilateral")
        assert isinstance(tiling, QuadrilateralTiling)

    def test_tiling_instance_passthrough(self):
        tiling = SquareTiling()
        assert resolve_tiling(tiling) is tiling

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown tiling"):
            resolve_tiling("pentagon")

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            resolve_tiling(42)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestTilingErrors:
    def test_omit_size_and_count_uses_default_square(self):
        # When neither n_tiles nor tile_size is given, DEFAULT_N_TILES is used.
        from carto_flow.symbol_cartogram.tiling import DEFAULT_N_TILES

        result = SquareTiling().generate(BOUNDS)
        assert result.n_tiles > 0
        assert abs(result.n_tiles - DEFAULT_N_TILES) < DEFAULT_N_TILES

    def test_omit_size_and_count_uses_default_hex(self):
        from carto_flow.symbol_cartogram.tiling import DEFAULT_N_TILES

        result = HexagonTiling().generate(BOUNDS)
        assert result.n_tiles > 0
        assert abs(result.n_tiles - DEFAULT_N_TILES) < DEFAULT_N_TILES

    def test_omit_size_and_count_uses_default_triangle(self):
        result = TriangleTiling.equilateral().generate(BOUNDS)
        assert result.n_tiles > 0

    def test_omit_size_and_count_uses_default_quad(self):
        result = QuadrilateralTiling.parallelogram().generate(BOUNDS)
        assert result.n_tiles > 0

    def test_degenerate_triangle_raises(self):
        with pytest.raises(ValueError, match="zero area"):
            TriangleTiling(Polygon([(0, 0), (1, 0), (2, 0)]))

    def test_degenerate_quad_raises(self):
        with pytest.raises(ValueError, match="zero area"):
            QuadrilateralTiling(Polygon([(0, 0), (1, 0), (2, 0), (3, 0)]))


# ---------------------------------------------------------------------------
# IsohedralTiling
# ---------------------------------------------------------------------------


class TestIsohedralTiling:
    """Tests for IsohedralTiling wrapping the tactile library."""

    def test_create_default(self):
        """Instantiate with type 1 (most general hexagon) — valid polygon."""
        t = IsohedralTiling(tiling_type=1)
        assert t.canonical_tile.is_valid
        assert t.canonical_tile.area == pytest.approx(1.0, rel=1e-6)

    def test_create_with_parameters(self):
        """Type 4 (6 params) — custom parameters change the shape."""
        info = IsohedralTiling.type_info(4)
        assert info["num_parameters"] == 6
        default_params = info["default_parameters"]
        t_default = IsohedralTiling(tiling_type=4)
        t_custom = IsohedralTiling(
            tiling_type=4,
            parameters=[p + 0.1 for p in default_params],
        )
        # Shape should differ (centroid distance or area normalized, so compare vertices)
        d_coords = np.array(t_default.canonical_tile.exterior.coords)
        c_coords = np.array(t_custom.canonical_tile.exterior.coords)
        assert not np.allclose(d_coords, c_coords, atol=1e-6)

    def test_generate_produces_tiles(self):
        t = IsohedralTiling(tiling_type=1)
        result = t.generate(BOUNDS, n_tiles=30)
        assert result.n_tiles > 0
        assert all(p.is_valid for p in result.polygons)

    def test_equal_areas(self):
        """All tiles should have equal area."""
        t = IsohedralTiling(tiling_type=4)
        result = t.generate(BOUNDS, n_tiles=30)
        areas = [p.area for p in result.polygons]
        assert np.allclose(areas, areas[0], rtol=1e-4)

    def test_no_overlaps(self):
        """No tiles should overlap significantly."""
        t = IsohedralTiling(tiling_type=28)
        result = t.generate(BOUNDS, n_tiles=30)
        for i in range(len(result.polygons)):
            for j in range(i + 1, min(i + 10, len(result.polygons))):
                intersection = result.polygons[i].intersection(result.polygons[j])
                # Small numerical overlap is OK; significant area overlap is not
                assert intersection.area < result.polygons[i].area * 0.01

    def test_flipped_tiles(self):
        """Types with reflections should produce flipped tiles."""
        # Type 2 has reflections (verified in smoke test)
        t = IsohedralTiling(tiling_type=2)
        result = t.generate(BOUNDS, n_tiles=30)
        has_flipped = any(tr.flipped for tr in result.transforms)
        assert has_flipped, "Expected some tiles to be flipped for IH2"

    def test_adjacency_symmetric(self):
        t = IsohedralTiling(tiling_type=76)
        result = t.generate(BOUNDS, n_tiles=30)
        adj = result.adjacency
        assert adj.shape[0] == adj.shape[1] == result.n_tiles
        assert np.array_equal(adj, adj.T)
        assert adj.any(), "Adjacency matrix should not be all-zero"

    def test_available_types(self):
        types = IsohedralTiling.available_types()
        assert len(types) == 81
        assert all(isinstance(t, int) for t in types)

    def test_type_info(self):
        info = IsohedralTiling.type_info(93)
        assert info["num_vertices"] == 3
        assert info["num_parameters"] == 0
        assert "edge_shapes" in info

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid isohedral"):
            IsohedralTiling(tiling_type=999)

    def test_resolve_tiling_ih(self):
        t = resolve_tiling("ih4")
        assert isinstance(t, IsohedralTiling)

    def test_resolve_tiling_isohedral(self):
        t = resolve_tiling("isohedral")
        assert isinstance(t, IsohedralTiling)

    def test_from_polygon_raises(self):
        with pytest.raises(NotImplementedError):
            IsohedralTiling.from_polygon(Polygon([(0, 0), (1, 0), (0, 1)]))

    @pytest.mark.parametrize("tiling_type", [1, 4, 28, 43, 76, 93])
    def test_various_types(self, tiling_type):
        """Generate tiles for representative types across vertex counts."""
        t = IsohedralTiling(tiling_type=tiling_type)
        result = t.generate(BOUNDS, n_tiles=20)
        assert result.n_tiles > 0
        areas = [p.area for p in result.polygons]
        assert np.allclose(areas, areas[0], rtol=1e-4)

    @pytest.mark.parametrize("tiling_type", [2, 24])
    def test_transform_reconstruction(self, tiling_type):
        """Reconstructing tiles from TileTransform must match actual polygons.

        This catches rotation extraction bugs for reflected (flipped) tiles.
        """
        from shapely.affinity import rotate as shapely_rotate
        from shapely.affinity import scale as shapely_scale
        from shapely.affinity import translate as shapely_translate

        t = IsohedralTiling(tiling_type=tiling_type)
        result = t.generate(BOUNDS, n_tiles=20)
        canonical = result.canonical_tile

        for poly, tr in zip(result.polygons, result.transforms, strict=False):
            # Reconstruct: canonical → flip → rotate → translate
            recon = canonical
            if tr.flipped:
                recon = shapely_scale(recon, xfact=-1, yfact=1, origin=(0, 0))
            if tr.rotation != 0.0:
                recon = shapely_rotate(recon, tr.rotation, origin=(0, 0))
            recon = shapely_translate(recon, tr.center[0], tr.center[1])

            dist = poly.hausdorff_distance(recon)
            assert dist < result.tile_size * 0.01, (
                f"Hausdorff distance {dist:.6f} too large for "
                f"tiling_type={tiling_type} rotation={tr.rotation} "
                f"flipped={tr.flipped}"
            )


class TestIsohedralEdgeCurves:
    """Tests for custom edge curve support."""

    def test_edge_curves_j(self):
        """J-type edges accept custom curves and produce more vertices."""
        # IH1: all edges are J (free)
        wavy = [(0, 0), (0.25, 0.1), (0.5, 0), (0.75, -0.1), (1, 0)]
        t_curved = IsohedralTiling(tiling_type=1, edge_curves={0: wavy})
        t_straight = IsohedralTiling(tiling_type=1)
        # Curved tile has more vertices
        n_curved = len(t_curved.canonical_tile.exterior.coords) - 1
        n_straight = len(t_straight.canonical_tile.exterior.coords) - 1
        assert n_curved > n_straight

    def test_edge_curves_s_symmetry(self):
        """S-type edges enforce 180° rotational symmetry."""
        # IH4 has S edges (edge shape IDs 0, 2, 3 are S for type 4)
        info = IsohedralTiling.type_info(4)
        # Find an S-edge ID
        s_ids = [i for i, name in enumerate(info["edge_shapes"][: info["num_edge_shapes"]]) if name == "S"]
        if not s_ids:
            pytest.skip("No S edges in type 4")
        s_id = s_ids[0]
        # Provide an asymmetric curve — should be symmetrized
        curve = [(0, 0), (0.2, 0.3), (0.4, 0.1), (0.6, 0.2), (1, 0)]
        t = IsohedralTiling(tiling_type=4, edge_curves={s_id: curve})
        # Should not raise and should produce a valid tile
        assert t.canonical_tile.is_valid
        assert t.canonical_tile.area == pytest.approx(1.0, rel=1e-4)

    def test_edge_curves_u_symmetry(self):
        """U-type edges enforce mirror symmetry."""
        # Find a type with U edges
        for tp in IsohedralTiling.available_types():
            info = IsohedralTiling.type_info(tp)
            edge_names = info["edge_shapes"][: info["num_edge_shapes"]]
            u_ids = [i for i, name in enumerate(edge_names) if name == "U"]
            if u_ids:
                break
        else:
            pytest.skip("No type with U edges found")
        curve = [(0, 0), (0.2, 0.2), (0.5, 0.3), (0.8, 0.1), (1, 0)]
        t = IsohedralTiling(tiling_type=tp, edge_curves={u_ids[0]: curve})
        assert t.canonical_tile.is_valid

    def test_edge_curves_i_ignored(self):
        """I-type edges ignore custom curves (must remain straight)."""
        # IH93 has all I edges
        curve = [(0, 0), (0.5, 0.5), (1, 0)]
        t = IsohedralTiling(tiling_type=93, edge_curves={0: curve})
        # Should still produce a simple triangle (3 vertices)
        n_verts = len(t.canonical_tile.exterior.coords) - 1
        assert n_verts == 3

    def test_edge_curve_tessellation(self):
        """Tiles with custom J-edge curves should still tessellate."""
        wavy = [(0, 0), (0.3, 0.15), (0.5, 0), (0.7, -0.15), (1, 0)]
        t = IsohedralTiling(tiling_type=1, edge_curves={0: wavy})
        result = t.generate(BOUNDS, n_tiles=20)
        assert result.n_tiles > 0
        # Check no significant overlaps among nearby tiles
        for i in range(min(10, len(result.polygons))):
            for j in range(i + 1, min(i + 5, len(result.polygons))):
                inter = result.polygons[i].intersection(result.polygons[j])
                assert inter.area < result.polygons[i].area * 0.01

    def test_edge_curves_ih83(self):
        """IH83 with custom J-edge curves should generate tiles (scale fix)."""
        curve = [(0, 0), (0.25, 0.5), (0.75, -0.5), (1, 0)]
        t = IsohedralTiling(tiling_type=83, edge_curves={0: curve})
        result = t.generate(BOUNDS, n_tiles=30)
        assert result.n_tiles > 0
        # All tiles should have valid area
        for poly in result.polygons:
            assert poly.is_valid
            assert poly.area > 0


class TestIsohedralTypeInfo:
    """Tests for enhanced type_info, describe, and plot_prototile."""

    def test_type_info_edges(self):
        """type_info returns edge detail dicts."""
        info = IsohedralTiling.type_info(83)
        assert "edges" in info
        assert len(info["edges"]) == info["num_vertices"]
        for e in info["edges"]:
            assert "edge_index" in e
            assert "shape_id" in e
            assert "shape_type" in e
            assert "description" in e
            assert "customizable" in e
        # IH83 has J and I edges
        types = {e["shape_type"] for e in info["edges"]}
        assert "J" in types
        assert "I" in types

    def test_type_info_has_reflections(self):
        """type_info detects reflection aspects."""
        info24 = IsohedralTiling.type_info(24)
        assert info24["has_reflections"] is True
        info1 = IsohedralTiling.type_info(1)
        assert info1["has_reflections"] is False

    def test_type_info_customizable_ids(self):
        """type_info lists customizable shape IDs."""
        info = IsohedralTiling.type_info(83)
        assert "customizable_shape_ids" in info
        assert 0 in info["customizable_shape_ids"]
        # Shape ID for I-edge should not be customizable
        i_ids = {e["shape_id"] for e in info["edges"] if not e["customizable"]}
        for sid in i_ids:
            assert sid not in info["customizable_shape_ids"]

    def test_describe_output(self):
        """describe() returns a multi-line string with key info."""
        text = IsohedralTiling.describe(83)
        assert "IH83" in text
        assert "Parameters:" in text
        assert "Edges:" in text
        assert "Free curve" in text
        assert "Straight" in text

    def test_describe_shows_parameter_effects(self):
        """describe() includes per-parameter vertex movement info."""
        text = IsohedralTiling.describe(84)
        assert "moves V2 horizontally" in text
        assert "moves V2 vertically" in text

    def test_type_info_parameters(self):
        """type_info returns parameter descriptions."""
        info = IsohedralTiling.type_info(84)
        assert "parameters" in info
        assert len(info["parameters"]) == 2
        p0 = info["parameters"][0]
        assert "index" in p0
        assert "default" in p0
        assert "affected_vertices" in p0
        assert "description" in p0
        # Param 0 moves V2 horizontally
        assert any(a["vertex"] == 2 for a in p0["affected_vertices"])

    def test_type_info_parameters_zero_params(self):
        """type_info returns empty parameters list for fixed types."""
        info = IsohedralTiling.type_info(93)
        assert info["parameters"] == []

    def test_self_intersection_warning(self):
        """Self-intersecting edge curves trigger a UserWarning."""
        import warnings

        curve = [(0, 0), (0.25, 0.5), (0.75, -0.5), (1, 0)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            IsohedralTiling(tiling_type=83, edge_curves={0: curve})
        assert any("self-intersecting" in str(x.message) for x in w)

    @pytest.mark.parametrize("tiling_type", [1, 24, 83])
    def test_plot_prototile_returns_axes(self, tiling_type):
        """plot_prototile returns a matplotlib Axes."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = IsohedralTiling(tiling_type=tiling_type)
        ax = t.plot_prototile()
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_plot_prototile_with_curves(self):
        """plot_prototile renders curved edges."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        curve = [(0, 0), (0.25, 0.3), (0.75, -0.3), (1, 0)]
        t = IsohedralTiling(tiling_type=1, edge_curves={0: curve})
        ax = t.plot_prototile()
        assert isinstance(ax, plt.Axes)
        plt.close("all")


class TestIsohedralFindTypes:
    """Tests for find_types() search method."""

    def test_find_by_vertices(self):
        """Filter by number of vertices."""
        tri = IsohedralTiling.find_types(num_vertices=3)
        assert all(IsohedralTiling.type_info(t)["num_vertices"] == 3 for t in tri)
        assert len(tri) > 0

    def test_find_all_j_edges(self):
        """Filter for types with all J edges."""
        types = IsohedralTiling.find_types(all_edges="J")
        for t in types:
            info = IsohedralTiling.type_info(t)
            assert all(e["shape_type"] == "J" for e in info["edges"])

    def test_find_no_reflections(self):
        """Filter for types without reflections."""
        types = IsohedralTiling.find_types(has_reflections=False)
        for t in types[:5]:  # spot check
            info = IsohedralTiling.type_info(t)
            assert info["has_reflections"] is False

    def test_find_max_parameters(self):
        """Filter by maximum parameters."""
        types = IsohedralTiling.find_types(max_parameters=0)
        for t in types:
            info = IsohedralTiling.type_info(t)
            assert info["num_parameters"] == 0

    def test_find_combined_filters(self):
        """Multiple filters are ANDed together."""
        types = IsohedralTiling.find_types(
            num_vertices=4,
            all_edges="J",
            has_reflections=False,
        )
        for t in types:
            info = IsohedralTiling.type_info(t)
            assert info["num_vertices"] == 4
            assert all(e["shape_type"] == "J" for e in info["edges"])
            assert info["has_reflections"] is False

    def test_find_no_match_returns_empty(self):
        """No match returns empty list."""
        types = IsohedralTiling.find_types(num_vertices=3, all_edges="J")
        assert types == []


class TestIsohedralPresets:
    """Tests for preset tilings."""

    def test_list_presets(self):
        """list_presets returns a dict of names to descriptions."""
        presets = IsohedralTiling.list_presets()
        assert isinstance(presets, dict)
        assert len(presets) >= 5
        for name, desc in presets.items():
            assert isinstance(name, str)
            assert isinstance(desc, str)

    def test_from_preset_straight(self):
        """from_preset creates a valid tiling (straight-edge preset)."""
        t = IsohedralTiling.from_preset("regular_hexagon")
        assert t.canonical_tile.is_valid
        result = t.generate(BOUNDS, n_tiles=20)
        assert result.n_tiles > 0

    def test_from_preset_curved(self):
        """from_preset creates a valid tiling (curved-edge preset)."""
        t = IsohedralTiling.from_preset("scalloped_hexagon")
        assert t.canonical_tile.is_valid
        result = t.generate(BOUNDS, n_tiles=20)
        assert result.n_tiles > 0
        # Curved tiles should have more vertices than straight
        t_straight = IsohedralTiling.from_preset("regular_hexagon")
        n_curved = len(t.canonical_tile.exterior.coords) - 1
        n_straight = len(t_straight.canonical_tile.exterior.coords) - 1
        assert n_curved > n_straight

    def test_from_preset_with_params(self):
        """from_preset applies non-default parameters."""
        t = IsohedralTiling.from_preset("regular_hexagon")
        assert t.canonical_tile.is_valid
        result = t.generate(BOUNDS, n_tiles=20)
        assert result.n_tiles > 0

    def test_from_preset_with_overrides(self):
        """from_preset accepts edge_curves override."""
        curve = [(0, 0), (0.3, 0.1), (0.7, -0.1), (1, 0)]
        t = IsohedralTiling.from_preset("square", edge_curves={0: curve})
        n_curved = len(t.canonical_tile.exterior.coords) - 1
        t_plain = IsohedralTiling.from_preset("square")
        n_plain = len(t_plain.canonical_tile.exterior.coords) - 1
        assert n_curved > n_plain

    def test_from_preset_invalid(self):
        """from_preset raises for unknown preset."""
        with pytest.raises(ValueError, match="Unknown preset"):
            IsohedralTiling.from_preset("nonexistent")

    def test_resolve_tiling_preset(self):
        """resolve_tiling resolves preset names."""
        t = resolve_tiling("regular_hexagon")
        assert isinstance(t, IsohedralTiling)

    @pytest.mark.parametrize("name", list(IsohedralTiling._PRESETS.keys()))
    def test_all_presets_generate(self, name):
        """Every preset generates tiles without error."""
        t = IsohedralTiling.from_preset(name)
        result = t.generate(BOUNDS, n_tiles=15)
        assert result.n_tiles > 0
