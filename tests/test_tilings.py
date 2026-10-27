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
from shapely.affinity import scale as shapely_scale
from shapely.affinity import translate as shapely_translate
from shapely.geometry import Polygon

from carto_flow.symbol_cartogram.tiling import (
    HexagonTiling,
    IsohedralTiling,
    QuadrilateralTiling,
    SquareTiling,
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
        result = SquareTiling().generate(BOUNDS, n_tiles=N_TILES, adjacency_type="edge")
        mask = _interior_mask(result, BOUNDS)
        counts = _neighbor_counts(result, mask)
        assert len(counts) > 0, "No interior tiles found"
        assert (counts == 4).all(), f"Expected 4 neighbors, got {np.unique(counts)}"

    def test_vertex_adjacency_interior_8(self):
        """Interior tiles in a square grid have 8 vertex-neighbors."""
        result = SquareTiling().generate(BOUNDS, n_tiles=N_TILES, adjacency_type="vertex")
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
        result = HexagonTiling().generate(BOUNDS, n_tiles=100, adjacency_type="edge")
        mask = _interior_mask(result, BOUNDS)
        counts = _neighbor_counts(result, mask)
        assert len(counts) > 0, "No interior tiles found"
        assert (counts == 6).all(), f"Expected 6 neighbors, got {np.unique(counts)}"

    def test_transforms_no_rotation(self):
        """Hex tiling tiles are all unrotated."""
        result = HexagonTiling().generate(BOUNDS, n_tiles=N_TILES)
        for t in result.transforms:
            assert t.rotation == pytest.approx(0.0, abs=1e-9)
            assert t.flipped is False

    def test_flat_top_has_vertex_at_angle_zero(self):
        """pointy_top=False gives flat-top tiles: a vertex on the horizontal axis of the canonical tile."""
        flat = HexagonTiling(pointy_top=False).canonical_tile
        pointy = HexagonTiling(pointy_top=True).canonical_tile
        flat_angles = np.degrees(np.arctan2(*np.array(flat.exterior.coords[:-1]).T[::-1])) % 60
        pointy_angles = np.degrees(np.arctan2(*np.array(pointy.exterior.coords[:-1]).T[::-1])) % 60
        np.testing.assert_allclose((flat_angles + 30.0) % 60.0 - 30.0, 0.0, atol=1e-9)
        np.testing.assert_allclose(pointy_angles, 30.0, atol=1e-9)

    def test_flat_top_tiles_match_the_symbol(self):
        """The tiles of HexagonTiling(pointy_top=False) are the tiles that HexagonSymbol(pointy_top=False) fills."""
        tiling = HexagonTiling(pointy_top=False)
        result = tiling.generate(BOUNDS, tile_size=1.0)
        unit = tiling.canonical_symbol().unit_polygon()
        # the symbol has bounding radius 0.5, the tile has circumradius tile_size
        symbol_tile = shapely_scale(unit, xfact=2.0, yfact=2.0, origin=(0, 0))
        placed = shapely_translate(symbol_tile, *result.transforms[0].center)
        assert placed.symmetric_difference(result.polygons[0]).area < 1e-9 * result.polygons[0].area

    def test_hexagon_tile_size_is_the_circumradius(self):
        result = HexagonTiling().generate(BOUNDS, tile_size=1.0)
        radii = np.linalg.norm(
            np.array(result.polygons[0].exterior.coords[:-1]) - np.array(result.transforms[0].center), axis=1
        )
        np.testing.assert_allclose(radii, 1.0, atol=1e-9)
        assert result.tile_size == pytest.approx(1.0)


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
        result = TriangleTiling.equilateral().generate(BOUNDS, n_tiles=200, adjacency_type="edge")
        mask = _interior_mask(result, BOUNDS)
        counts = _neighbor_counts(result, mask)
        assert len(counts) > 0, "No interior tiles found"
        assert (counts == 3).all(), f"Expected 3 neighbors, got {np.unique(counts)}"

    def test_equilateral_vertex_adjacency_interior_12(self):
        """Interior equilateral triangle tiles have 12 vertex-neighbors."""
        result = TriangleTiling.equilateral().generate(BOUNDS, n_tiles=200, adjacency_type="vertex")
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
        result = QuadrilateralTiling.parallelogram().generate(BOUNDS, n_tiles=200, adjacency_type="edge")
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
        result = QuadrilateralTiling.trapezoid(top_ratio=0.5).generate(BOUNDS, n_tiles=200, adjacency_type="edge")
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
    """Tests for IsohedralTiling, the Tiling over isotile."""

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

    @pytest.mark.parametrize("shape", ["S", "U"])
    def test_symmetric_edge_curves_reach_the_canonical_tile(self, shape):
        """Curves of S and U edges are applied to the canonical tile and keep it a simple unit-area polygon."""
        for number in IsohedralTiling.available_types():
            info = IsohedralTiling.type_info(number)
            ids = [i for i, name in enumerate(info["edge_shapes"]) if name == shape]
            if ids:
                break
        curve = [(0, 0), (0.2, 0.1), (0.4, 0.05), (0.6, -0.05), (0.8, -0.1), (1, 0)]
        curved = IsohedralTiling(tiling_type=number, edge_curves={ids[0]: curve})
        straight = IsohedralTiling(tiling_type=number)
        assert curved.canonical_tile.is_valid
        assert curved.canonical_tile.area == pytest.approx(1.0, rel=1e-4)
        assert len(curved.canonical_tile.exterior.coords) > len(straight.canonical_tile.exterior.coords)

    def test_edge_curve_on_a_straight_edge_raises(self):
        """A curve for an I edge cannot be applied and is an error."""
        # IH93 has only I edges
        with pytest.raises(ValueError, match="straight"):
            IsohedralTiling(tiling_type=93, edge_curves={0: [(0, 0), (0.5, 0.5), (1, 0)]})

    def test_self_intersecting_edge_curves_raise(self):
        """Edge curves that make the prototile self-intersect are rejected."""
        curve = [(0, 0), (0.25, 0.5), (0.75, -0.5), (1, 0)]
        with pytest.raises(ValueError, match="invalid prototile"):
            IsohedralTiling(tiling_type=83, edge_curves={0: curve})

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
        """IH83 with a large custom J-edge curve generates tiles of the same size as without it."""
        curve = [(0, 0), (0.25, 0.3), (0.75, 0.3), (1, 0)]
        t = IsohedralTiling(tiling_type=83, edge_curves={0: curve})
        result = t.generate(BOUNDS, n_tiles=30)
        assert result.n_tiles > 0
        # All tiles should have valid area
        for poly in result.polygons:
            assert poly.is_valid
            assert poly.area > 0


class TestIsohedralDelegates:
    """The type metadata, presets and serialization of IsohedralTiling are isotile's."""

    def test_type_info_on_class_and_instance(self):
        import isotile

        info = isotile.get_type(83).info()
        assert IsohedralTiling.type_info(83) == info
        assert IsohedralTiling(83).type_info() == info
        assert "IH83" in IsohedralTiling.describe(83)
        assert IsohedralTiling(83).describe() == IsohedralTiling.describe(83)

    def test_type_info_needs_a_number_on_the_class(self):
        with pytest.raises(TypeError, match="type_info"):
            IsohedralTiling.type_info()
        with pytest.raises(TypeError, match="describe"):
            IsohedralTiling.describe()

    def test_find_types(self):
        import isotile

        assert IsohedralTiling.find_types(num_vertices=3) == isotile.find_types(num_vertices=3)
        assert IsohedralTiling.find_types(num_vertices=3, all_edges="J") == []
        assert IsohedralTiling.find_types(num_vertices=6, max_parameters=0) == [10, 11, 18, 20]

    def test_list_presets(self):
        import isotile

        assert IsohedralTiling.list_presets() == isotile.list_presets()
        assert len(IsohedralTiling.list_presets()) == 15

    def test_dict_and_file_round_trip(self, tmp_path):
        original = IsohedralTiling.from_preset("scalloped_hexagon")
        for restored in (
            IsohedralTiling.from_dict(original.to_dict()),
            _saved_and_loaded(original, tmp_path / "t.json"),
        ):
            assert isinstance(restored, IsohedralTiling)
            assert restored.isotile.to_dict() == original.isotile.to_dict()
            np.testing.assert_allclose(
                np.array(restored.canonical_tile.exterior.coords), np.array(original.canonical_tile.exterior.coords)
            )

    def test_isotile_property(self):
        import isotile

        tiling = IsohedralTiling(4)
        assert isinstance(tiling.isotile, isotile.IsohedralTiling)
        assert tiling.isotile.type.number == 4


def _saved_and_loaded(tiling: IsohedralTiling, path) -> IsohedralTiling:
    tiling.save(path)
    return IsohedralTiling.load(path)


class TestIsohedralPlotPrototile:
    """``plot_prototile`` draws with isotile."""

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


class TestIsohedralPresets:
    """Tests for preset tilings."""

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
        with pytest.raises(ValueError, match="unknown preset"):
            IsohedralTiling.from_preset("nonexistent")

    def test_resolve_tiling_preset(self):
        """resolve_tiling resolves preset names."""
        t = resolve_tiling("regular_hexagon")
        assert isinstance(t, IsohedralTiling)

    @pytest.mark.parametrize("name", list(IsohedralTiling.list_presets()))
    def test_all_presets_generate(self, name):
        """Every preset generates tiles without error."""
        t = IsohedralTiling.from_preset(name)
        result = t.generate(BOUNDS, n_tiles=15)
        assert result.n_tiles > 0


# ---------------------------------------------------------------------------
# Covering a region
# ---------------------------------------------------------------------------

FAMILIES = [
    pytest.param(SquareTiling, 0.2, id="square"),
    pytest.param(HexagonTiling, 0.12, id="hexagon"),
    pytest.param(TriangleTiling.equilateral, 0.2, id="triangle"),
    pytest.param(QuadrilateralTiling.parallelogram, 0.2, id="quadrilateral"),
]


def _sample_hits(result: TilingResult, region, n: int = 2000, seed: int = 5) -> np.ndarray:
    """Number of tiles that contain each of ``n`` random points of the region."""
    import shapely
    from shapely import STRtree

    minx, miny, maxx, maxy = region
    rng = np.random.default_rng(seed)
    points = shapely.points(np.column_stack([rng.uniform(minx, maxx, n), rng.uniform(miny, maxy, n)]))
    hit = STRtree(result.polygons).query(points, predicate="within")
    return np.bincount(hit[0], minlength=n)


class TestCoverage:
    """The tiles cover the region exactly once, wherever the region is."""

    @pytest.mark.parametrize("center", [0.0, 10.0, 1e3, 1e6])
    @pytest.mark.parametrize(("factory", "size"), FAMILIES)
    def test_region_far_from_the_origin(self, factory, size, center):
        bounds = (center - 1.0, center - 1.0, center + 1.0, center + 1.0)
        result = factory().generate(bounds, tile_size=size, margin=0.0)
        assert result.n_tiles * result.canonical_tile.area >= 4.0 * (1 - 1e-6)
        assert (_sample_hits(result, bounds) == 1).all()

    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("sides", [3, 4])
    def test_random_shapes_leave_no_hole(self, sides, seed):
        rng = np.random.default_rng(seed)
        while True:
            points = rng.uniform(-1.0, 1.0, (sides, 2))
            center = points.mean(axis=0)
            order = np.argsort(np.arctan2(*(points - center).T[::-1]))  # a star-shaped polygon is simple
            shape = Polygon(points[order])
            if shape.is_valid and shape.area > 0.3:
                break
        cls = TriangleTiling if sides == 3 else QuadrilateralTiling
        bounds = (-1.0, -1.0, 1.0, 1.0)
        result = cls(shape).generate(bounds, tile_size=0.25, margin=0.0)
        assert (_sample_hits(result, bounds) == 1).all()

    @pytest.mark.parametrize(("factory", "size"), FAMILIES)
    def test_tiles_overlap_the_region(self, factory, size):
        from shapely.geometry import box

        bounds = (0.0, 0.0, 2.0, 1.0)
        result = factory().generate(bounds, tile_size=size, margin=0.0)
        region = box(*bounds)
        assert all(p.intersection(region).area > 1e-6 * p.area for p in result.polygons)

    @pytest.mark.parametrize(("factory", "size"), FAMILIES)
    def test_lattice_is_kept_when_the_bounds_expand_symmetrically(self, factory, size):
        """Every tile of a region is a tile of the region expanded by the same amount on all sides."""
        from scipy.spatial import cKDTree

        tiling = factory()
        bounds = (-1.0, -1.0, 1.0, 1.0)
        inner = tiling.generate(bounds, tile_size=size, margin=0.0)
        buffer = 1.5 * size
        outer = tiling.generate(
            (bounds[0] - buffer, bounds[1] - buffer, bounds[2] + buffer, bounds[3] + buffer),
            tile_size=size,
            margin=0.0,
        )
        distance, _ = cKDTree(outer.centers).query(inner.centers)
        assert (distance < 1e-9).all()

    @pytest.mark.parametrize(("factory", "size"), FAMILIES)
    def test_n_tiles_sets_the_tile_size(self, factory, size):
        result = factory().generate((0.0, 0.0, 4.0, 3.0), n_tiles=100, margin=0.0)
        assert result.polygons[0].area == pytest.approx(12.0 / 100, rel=1e-9)
        assert result.n_tiles >= 100

    def test_margin_expands_the_region(self):
        plain = SquareTiling().generate((0.0, 0.0, 4.0, 4.0), tile_size=1.0, margin=0.0)
        padded = SquareTiling().generate((0.0, 0.0, 4.0, 4.0), tile_size=1.0, margin=0.5)
        assert plain.n_tiles == 16
        assert padded.n_tiles == 64

    def test_invalid_arguments_raise(self):
        with pytest.raises(ValueError):
            SquareTiling().generate(BOUNDS, tile_size=-1.0)
        with pytest.raises(ValueError):
            SquareTiling().generate(BOUNDS, n_tiles=0)
        with pytest.raises(ValueError, match="margin"):
            SquareTiling().generate(BOUNDS, n_tiles=10, margin=-0.5)
        with pytest.raises(ValueError):
            SquareTiling().generate((1.0, 0.0, 1.0, 2.0), n_tiles=10)

    def test_self_intersecting_quadrilateral_raises(self):
        with pytest.raises(ValueError):
            QuadrilateralTiling(Polygon([(0, 0), (1, 1), (1, 0), (0, 1)]))


# ---------------------------------------------------------------------------
# Adjacency kinds and the isotile tile set
# ---------------------------------------------------------------------------


class TestAdjacencyKinds:
    def test_neighbor_counts_of_squares(self):
        counts = {}
        for kind in ("edge", "vertex", "vertex-only"):
            result = SquareTiling().generate(BOUNDS, n_tiles=100, adjacency_type=kind)
            mask = _interior_mask(result, BOUNDS)
            counts[kind] = set(np.unique(_neighbor_counts(result, mask)).tolist())
        assert counts == {"edge": {4}, "vertex": {8}, "vertex-only": {4}}

    def test_vertex_adjacency_field_does_not_depend_on_the_kind(self):
        tiling = TriangleTiling.equilateral()
        reference = tiling.generate(BOUNDS, n_tiles=100).vertex_adjacency
        assert reference.any()
        for kind in ("edge", "vertex", "vertex-only"):
            np.testing.assert_array_equal(
                tiling.generate(BOUNDS, n_tiles=100, adjacency_type=kind).vertex_adjacency, reference
            )

    def test_vertex_only_kind_equals_the_vertex_adjacency_field(self):
        result = TriangleTiling.equilateral().generate(BOUNDS, n_tiles=100, adjacency_type="vertex-only")
        np.testing.assert_array_equal(result.adjacency, result.vertex_adjacency)

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            SquareTiling().generate(BOUNDS, n_tiles=10, adjacency_type="corner")

    def test_hexagons_have_no_vertex_only_neighbors(self):
        result = HexagonTiling().generate(BOUNDS, n_tiles=100)
        assert not result.vertex_adjacency.any()


class TestTileSet:
    def test_tile_set_describes_the_tiles(self):
        result = TriangleTiling.equilateral().generate(BOUNDS, n_tiles=50)
        tile_set = result.tile_set
        assert tile_set is not None
        assert len(tile_set) == result.n_tiles
        assert tile_set.tiling.type.number == 84
        assert set(tile_set.aspect.tolist()) == {0, 1}
        np.testing.assert_allclose(tile_set.polygons()[3].exterior.coords, result.polygons[3].exterior.coords)

    def test_rotate_drops_the_tile_set(self):
        result = SquareTiling().generate(BOUNDS, n_tiles=20)
        assert result.rotate(30.0).tile_set is None

    def test_tile_set_is_optional(self):
        result = TilingResult(
            polygons=[Polygon([(0, 0), (1, 0), (1, 1)])],
            transforms=[TileTransform(center=(0, 0))],
            adjacency=np.zeros((1, 1), dtype=bool),
            vertex_adjacency=None,
            tile_size=1.0,
            inscribed_radius=0.3,
            canonical_tile=Polygon([(0, 0), (1, 0), (1, 1)]),
        )
        assert result.tile_set is None

    def test_isohedral_tile_set_keeps_the_cells(self):
        result = IsohedralTiling(2).generate(BOUNDS, n_tiles=50)
        assert result.tile_set.cell.shape == (result.n_tiles, 2)
        assert result.n_base_vertices == 6


class TestSerialization:
    def test_files_of_the_former_format_load(self):
        data = {
            "version": "1.0",
            "tiling_type": 4,
            "parameters": IsohedralTiling.type_info(4)["default_parameters"],
            "edge_curves": {},
        }
        tiling = IsohedralTiling.from_dict(data)
        assert tiling.isotile.type.number == 4

    def test_format_1_1_is_written(self):
        written = IsohedralTiling.from_preset("wavy_square").to_dict()
        assert written["version"] == "1.1"
        assert set(written["edge_curve_modes"]) == set(written["edge_curves"])
        assert "edge_curves" not in IsohedralTiling(4).to_dict()
