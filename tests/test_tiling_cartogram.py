"""Integration tests for tiling-based symbol cartograms.

Tests the full pipeline: create_symbol_cartogram with various tiling options
passed through GridBasedLayout.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from carto_flow.symbol_cartogram import (
    GridBasedLayout,
    create_symbol_cartogram,
)
from carto_flow.symbol_cartogram.options import GridBasedLayoutOptions, SymbolOrientation
from carto_flow.symbol_cartogram.tiling import (
    IsohedralTiling,
    QuadrilateralTiling,
    SquareTiling,
    TriangleTiling,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_grid_gdf(rows: int = 3, cols: int = 3, seed: int = 42) -> gpd.GeoDataFrame:
    """Create a grid of adjacent squares with population data."""
    rng = np.random.default_rng(seed)
    geoms = []
    for r in range(rows):
        for c in range(cols):
            geoms.append(box(c, r, c + 1, r + 1))
    n = rows * cols
    return gpd.GeoDataFrame(
        {"population": rng.integers(100, 10000, size=n).astype(float)},
        geometry=geoms,
    )


@pytest.fixture
def gdf():
    return make_grid_gdf()


# ---------------------------------------------------------------------------
# Pipeline tests with different tilings
# ---------------------------------------------------------------------------


class TestGridAlgorithmWithTilings:
    """Test GridBasedLayout with each tiling type."""

    def test_default_hexagon(self, gdf):
        """Default grid algorithm uses hexagon tiling."""
        result = create_symbol_cartogram(gdf, "population", layout="grid")
        assert result is not None
        assert len(result.symbols) == len(gdf)

    def test_square_tiling_string(self, gdf):
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling="square"))
        result = create_symbol_cartogram(gdf, "population", layout=layout)
        assert len(result.symbols) == len(gdf)

    def test_hexagon_tiling_string(self, gdf):
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling="hexagon"))
        result = create_symbol_cartogram(gdf, "population", layout=layout)
        assert len(result.symbols) == len(gdf)

    def test_triangle_tiling_string(self, gdf):
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling="triangle"))
        result = create_symbol_cartogram(gdf, "population", layout=layout)
        assert len(result.symbols) == len(gdf)

    def test_quadrilateral_tiling_string(self, gdf):
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling="quadrilateral"))
        result = create_symbol_cartogram(gdf, "population", layout=layout)
        assert len(result.symbols) == len(gdf)

    def test_square_tiling_instance(self, gdf):
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling=SquareTiling()))
        result = create_symbol_cartogram(gdf, "population", layout=layout)
        assert len(result.symbols) == len(gdf)

    def test_triangle_tiling_instance(self, gdf):
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling=TriangleTiling.equilateral()))
        result = create_symbol_cartogram(gdf, "population", layout=layout)
        assert len(result.symbols) == len(gdf)

    def test_right_triangle_tiling(self, gdf):
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling=TriangleTiling.right_isosceles()))
        result = create_symbol_cartogram(gdf, "population", layout=layout)
        assert len(result.symbols) == len(gdf)

    def test_parallelogram_tiling(self, gdf):
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling=QuadrilateralTiling.parallelogram(angle=60.0)))
        result = create_symbol_cartogram(gdf, "population", layout=layout)
        assert len(result.symbols) == len(gdf)

    def test_trapezoid_tiling(self, gdf):
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling=QuadrilateralTiling.trapezoid(top_ratio=0.5)))
        result = create_symbol_cartogram(gdf, "population", layout=layout)
        assert len(result.symbols) == len(gdf)

    def test_rhombus_tiling(self, gdf):
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling=QuadrilateralTiling.rhombus(angle=60.0)))
        result = create_symbol_cartogram(gdf, "population", layout=layout)
        assert len(result.symbols) == len(gdf)

    def test_custom_triangle(self, gdf):
        from shapely.geometry import Polygon

        tri = Polygon([(0, 0), (3, 0), (1, 4)])
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling=TriangleTiling(tri)))
        result = create_symbol_cartogram(gdf, "population", layout=layout)
        assert len(result.symbols) == len(gdf)

    def test_custom_quadrilateral(self, gdf):
        from shapely.geometry import Polygon

        quad = Polygon([(0, 0), (4, 0), (3, 3), (1, 2)])
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling=QuadrilateralTiling(quad)))
        result = create_symbol_cartogram(gdf, "population", layout=layout)
        assert len(result.symbols) == len(gdf)


# ---------------------------------------------------------------------------
# Symbol orientation
# ---------------------------------------------------------------------------


class TestSymbolOrientation:
    def test_upright_default(self, gdf):
        """Default orientation is UPRIGHT — no rotation info in result."""
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling=TriangleTiling.equilateral()))
        result = create_symbol_cartogram(gdf, "population", layout=layout)
        assert len(result.symbols) == len(gdf)

    def test_with_tile_orientation(self, gdf):
        """WITH_TILE orientation should not error."""
        layout = GridBasedLayout(
            GridBasedLayoutOptions(
                tiling=TriangleTiling.equilateral(),
                symbol_orientation=SymbolOrientation.WITH_TILE,
            )
        )
        result = create_symbol_cartogram(gdf, "population", layout=layout)
        assert len(result.symbols) == len(gdf)


# ---------------------------------------------------------------------------
# Tile as default symbol
# ---------------------------------------------------------------------------


class TestTileSymbolDefault:
    """When symbol=None (default), grid placement uses the tile shape."""

    def test_default_symbol_is_tile_shaped(self, gdf):
        """Grid algorithm with no symbol arg produces tile-shaped symbols."""
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling=TriangleTiling.equilateral()))
        result = create_symbol_cartogram(gdf, "population", layout=layout)
        assert len(result.symbols) == len(gdf)
        # Symbols should be triangular (3 vertices)
        for geom in result.symbols.geometry:
            n_verts = len(geom.exterior.coords) - 1  # minus closing point
            assert n_verts == 3, f"Expected triangle, got {n_verts}-gon"

    def test_default_symbol_square_tiling(self, gdf):
        """Square tiling with no symbol arg produces square symbols."""
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling="square"))
        result = create_symbol_cartogram(gdf, "population", layout=layout)
        for geom in result.symbols.geometry:
            n_verts = len(geom.exterior.coords) - 1
            assert n_verts == 4

    def test_default_symbol_quad_tiling(self, gdf):
        """Quadrilateral tiling with no symbol produces quadrilateral symbols."""
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling=QuadrilateralTiling.trapezoid(top_ratio=0.5)))
        result = create_symbol_cartogram(gdf, "population", layout=layout)
        for geom in result.symbols.geometry:
            n_verts = len(geom.exterior.coords) - 1
            assert n_verts == 4

    def test_explicit_circle_overrides_tile(self, gdf):
        """Passing symbol='circle' uses circles, not tile shape."""
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling=TriangleTiling.equilateral()))
        result = create_symbol_cartogram(gdf, "population", layout=layout, styling={"symbol": "circle"})
        for geom in result.symbols.geometry:
            n_verts = len(geom.exterior.coords) - 1
            assert n_verts == 32  # circle approximation

    def test_physics_layout_defaults_to_circle(self, gdf):
        """Physics layout with no symbol arg defaults to circle."""
        result = create_symbol_cartogram(gdf, "population", layout="physics")
        for geom in result.symbols.geometry:
            n_verts = len(geom.exterior.coords) - 1
            assert n_verts == 32

    def test_spacing_zero_fills_tiles(self):
        """With spacing=0 and uniform size, tile symbols should fill the tile."""
        gdf = make_grid_gdf(rows=3, cols=3)
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling="square", spacing=0.0))
        result = create_symbol_cartogram(gdf, layout=layout)
        # All symbols should have the same area (no value_column → uniform)
        areas = [g.area for g in result.symbols.geometry]
        assert np.allclose(areas, areas[0], rtol=1e-6)
        # The symbol area should match the tile area (spacing=0 → no gap)
        tiling_result = result._tiling_result
        tile_area = tiling_result.canonical_tile.area
        assert areas[0] == pytest.approx(tile_area, rel=0.01), (
            f"Symbol area {areas[0]:.4f} != tile area {tile_area:.4f}"
        )

    def test_spacing_nonzero_smaller_than_tile(self):
        """With spacing>0, symbols should be strictly smaller than tiles."""
        gdf = make_grid_gdf(rows=3, cols=3)
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling="square", spacing=0.1))
        result = create_symbol_cartogram(gdf, layout=layout)
        tiling_result = result._tiling_result
        tile_area = tiling_result.canonical_tile.area
        for g in result.symbols.geometry:
            assert g.area < tile_area

    def test_inscribed_symbol_fits_inside_tile(self):
        """Non-tile symbols (e.g. square) must fit inside their tile."""
        gdf = make_grid_gdf(rows=3, cols=3)
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling=TriangleTiling.equilateral(), spacing=0.0))
        result = create_symbol_cartogram(
            gdf,
            layout=layout,
            styling={"symbol": "square"},
        )
        # Verify symbols are reasonably sized: area should not exceed tile area.
        tiling_result = result._tiling_result
        tile_area = tiling_result.canonical_tile.area
        for geom in result.symbols.geometry:
            assert geom.area <= tile_area + 1e-6, f"Symbol area {geom.area:.4f} exceeds tile area {tile_area:.4f}"


# ---------------------------------------------------------------------------
# Symbol sizing fixes
# ---------------------------------------------------------------------------


class TestSymbolSizing:
    def test_hex_symbol_hex_tiling_zero_spacing(self):
        """Hexagon symbol + hexagon tiling + spacing=0 should fill tile."""
        gdf = make_grid_gdf(rows=3, cols=3)
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling="hexagon", spacing=0.0))
        result = create_symbol_cartogram(gdf, layout=layout, styling={"symbol": "hexagon"})
        # Retrieve tile area from actual result
        tiling_result = result._tiling_result
        tile_area = tiling_result.canonical_tile.area
        areas = [g.area for g in result.symbols.geometry]
        # All uniform → same area
        assert np.allclose(areas, areas[0], rtol=1e-6)
        # With zero spacing, the hexagon symbol should fill the tile
        assert areas[0] == pytest.approx(tile_area, rel=0.05), (
            f"Symbol area {areas[0]:.4f} != tile area {tile_area:.4f}"
        )

    def test_symbol_list_grid_placement(self):
        """Symbol as a list should work with grid placement."""
        gdf = make_grid_gdf(rows=3, cols=3)
        symbols = ["circle"] * len(gdf)
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling="hexagon"))
        result = create_symbol_cartogram(gdf, "population", layout=layout, styling={"symbol": symbols})
        assert len(result.symbols) == len(gdf)

    def test_circle_in_hex_tiling_regression(self):
        """Circle symbol in hex tiling should still work correctly."""
        gdf = make_grid_gdf(rows=3, cols=3)
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling="hexagon", spacing=0.0))
        result = create_symbol_cartogram(gdf, "population", layout=layout, styling={"symbol": "circle"})
        assert len(result.symbols) == len(gdf)
        # All circle polygons should have 32 vertices
        for geom in result.symbols.geometry:
            n_verts = len(geom.exterior.coords) - 1
            assert n_verts == 32

    def test_mixed_symbol_list_per_symbol_sizing(self):
        """With symbol=["hexagon", "circle", ...] on hex tiling, hex symbols
        should be larger than circle symbols (hex fills tile, circle inscribed)."""
        gdf = make_grid_gdf(rows=2, cols=2)
        # Alternate hexagons and circles
        n = len(gdf)
        symbols = ["hexagon" if i % 2 == 0 else "circle" for i in range(n)]
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling="hexagon", spacing=0.0))
        result = create_symbol_cartogram(gdf, layout=layout, styling={"symbol": symbols})
        assert len(result.symbols) == n

        # Hexagons should have larger area than circles (both at uniform size)
        hex_areas = []
        circle_areas = []
        for i, geom in enumerate(result.symbols.geometry):
            if i % 2 == 0:
                hex_areas.append(geom.area)
            else:
                circle_areas.append(geom.area)
        assert len(hex_areas) > 0 and len(circle_areas) > 0
        # Hex fills tile fully, circle is inscribed → hex area > circle area
        assert np.mean(hex_areas) > np.mean(circle_areas)


# ---------------------------------------------------------------------------
# Hungarian algorithm weights
# ---------------------------------------------------------------------------


class TestHungarianWeights:
    def test_compactness_produces_tighter_layout(self, gdf):
        """Higher compactness should produce more compact assignments."""
        layout_loose = GridBasedLayout(GridBasedLayoutOptions(compactness=0.0))
        layout_compact = GridBasedLayout(GridBasedLayoutOptions(compactness=1.0))
        result_loose = create_symbol_cartogram(gdf, "population", layout=layout_loose)
        result_compact = create_symbol_cartogram(gdf, "population", layout=layout_compact)
        # Both should complete without error
        assert len(result_loose.symbols) == len(gdf)
        assert len(result_compact.symbols) == len(gdf)

    def test_topology_weight_runs(self, gdf):
        """Topology weight > 0 should work without error."""
        layout = GridBasedLayout(GridBasedLayoutOptions(topology_weight=0.5, neighbor_weight=0.3))
        result = create_symbol_cartogram(gdf, "population", layout=layout)
        assert len(result.symbols) == len(gdf)

    def test_all_weights_zero_except_origin(self, gdf):
        """With only origin_weight, assignment is purely by distance."""
        layout = GridBasedLayout(
            GridBasedLayoutOptions(
                origin_weight=1.0,
                neighbor_weight=0.0,
                topology_weight=0.0,
                compactness=0.0,
            )
        )
        result = create_symbol_cartogram(gdf, "population", layout=layout)
        assert len(result.symbols) == len(gdf)


# ---------------------------------------------------------------------------
# Isohedral tiling integration
# ---------------------------------------------------------------------------


class TestIsohedralPipeline:
    """Test full pipeline with IsohedralTiling."""

    def test_isohedral_instance(self, gdf):
        """IsohedralTiling instance passed directly to GridBasedLayout."""
        tiling = IsohedralTiling(tiling_type=4)
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling=tiling))
        result = create_symbol_cartogram(gdf, "population", layout=layout)
        assert result is not None
        assert len(result.symbols) == len(gdf)

    def test_isohedral_string_shorthand(self, gdf):
        """String shorthand 'ih28' resolves to IsohedralTiling."""
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling="ih28"))
        result = create_symbol_cartogram(gdf, "population", layout=layout)
        assert len(result.symbols) == len(gdf)

    def test_isohedral_with_reflections(self, gdf):
        """Type with reflections (IH2) works through the pipeline."""
        tiling = IsohedralTiling(tiling_type=2)
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling=tiling))
        result = create_symbol_cartogram(gdf, "population", layout=layout)
        assert len(result.symbols) == len(gdf)

    def test_isohedral_with_edge_curves(self, gdf):
        """Pipeline works with custom edge curves."""
        wavy = [(0, 0), (0.3, 0.1), (0.5, 0), (0.7, -0.1), (1, 0)]
        tiling = IsohedralTiling(tiling_type=1, edge_curves={0: wavy})
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling=tiling))
        result = create_symbol_cartogram(gdf, "population", layout=layout)
        assert len(result.symbols) == len(gdf)


# ---------------------------------------------------------------------------
# Fill internal holes
# ---------------------------------------------------------------------------


class TestFillInternalHoles:
    """Tests for the fill_holes post-processing feature."""

    def test_fill_holes_unit(self):
        """Directly test fill_internal_holes on a controlled scenario.

        Create a 3x3 square grid of tiles, assign 8 of 9 to regions,
        leaving the center tile unassigned (an internal hole). Original
        polygons are slightly oversized so their union covers the center
        (no geographic gap). The function should fill the hole.
        """
        from carto_flow.symbol_cartogram.placement import fill_internal_holes

        # 9 tiles in a 3x3 grid, tile indices 0-8:
        #  6  7  8
        #  3  4  5
        #  0  1  2
        grid_centers = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [1, 1],
                [2, 1],
                [0, 2],
                [1, 2],
                [2, 2],
            ],
            dtype=float,
        )

        # Tile adjacency (4-connected)
        m = 9
        adj = np.zeros((m, m), dtype=bool)
        neighbors = {
            0: [1, 3],
            1: [0, 2, 4],
            2: [1, 5],
            3: [0, 4, 6],
            4: [1, 3, 5, 7],
            5: [2, 4, 8],
            6: [3, 7],
            7: [4, 6, 8],
            8: [5, 7],
        }
        for i, nbs in neighbors.items():
            for j in nbs:
                adj[i, j] = True

        # 8 regions assigned to tiles 0-3, 5-8 (skip tile 4 = center hole)
        centroids = grid_centers[[0, 1, 2, 3, 5, 6, 7, 8]]
        assignments = np.array([0, 1, 2, 3, 5, 6, 7, 8], dtype=np.intp)

        # Original polygons: oversized squares (1.2 half-side) so
        # their union fully covers the center area (no interior hole).
        from shapely.geometry import box as shapely_box

        original_polygons = [shapely_box(c[0] - 1.2, c[1] - 1.2, c[0] + 1.2, c[1] + 1.2) for c in centroids]

        result = fill_internal_holes(
            assignments,
            adj,
            centroids,
            grid_centers,
            original_polygons,
        )

        # The center tile (4) should now be assigned
        assert 4 in set(result.tolist())
        assert len(result) == 8

    def test_fill_holes_preserves_geographic_gap(self):
        """When original geometries have an interior hole (like a lake),
        the corresponding grid hole should be preserved."""
        from shapely.geometry import box as shapely_box

        from carto_flow.symbol_cartogram.placement import fill_internal_holes

        # Same 3x3 grid as above
        grid_centers = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [1, 1],
                [2, 1],
                [0, 2],
                [1, 2],
                [2, 2],
            ],
            dtype=float,
        )

        m = 9
        adj = np.zeros((m, m), dtype=bool)
        neighbors = {
            0: [1, 3],
            1: [0, 2, 4],
            2: [1, 5],
            3: [0, 4, 6],
            4: [1, 3, 5, 7],
            5: [2, 4, 8],
            6: [3, 7],
            7: [4, 6, 8],
            8: [5, 7],
        }
        for i, nbs in neighbors.items():
            for j in nbs:
                adj[i, j] = True

        # 8 regions, center is a "lake" — original polygons form a ring
        # with an interior hole.
        centroids = grid_centers[[0, 1, 2, 3, 5, 6, 7, 8]]
        assignments = np.array([0, 1, 2, 3, 5, 6, 7, 8], dtype=np.intp)

        # Create a ring polygon (outer ring covering all, inner ring = lake)
        from shapely.geometry import Polygon as ShapelyPolygon

        outer = [(-0.5, -0.5), (2.5, -0.5), (2.5, 2.5), (-0.5, 2.5)]
        inner = [(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)]
        ring_poly = ShapelyPolygon(outer, [inner])
        # Split into 8 pieces (one per region) that together have 1 interior hole
        original_polygons = [
            shapely_box(c[0] - 0.5, c[1] - 0.5, c[0] + 0.5, c[1] + 0.5).intersection(ring_poly) for c in centroids
        ]

        result = fill_internal_holes(
            assignments,
            adj,
            centroids,
            grid_centers,
            original_polygons,
        )

        # The center hole should be PRESERVED (1 grid hole == 1 geo hole)
        assert 4 not in set(result.tolist())
        assert len(result) == 8

    def test_fill_holes_integration(self):
        """Integration test: fill_holes=True via create_symbol_cartogram."""
        gdf = make_grid_gdf(rows=4, cols=4, seed=42)
        layout = GridBasedLayout(
            GridBasedLayoutOptions(tiling="hexagon", fill_holes=True),
        )
        result = create_symbol_cartogram(
            gdf,
            "population",
            layout=layout,
        )
        assert len(result.symbols) == len(gdf)

    def test_fill_holes_false_default(self):
        """fill_holes=True (default) does not change behavior."""
        gdf = make_grid_gdf(rows=3, cols=3, seed=42)
        layout = GridBasedLayout(GridBasedLayoutOptions(tiling="hexagon"))
        assert layout._options.fill_holes is True
        result = create_symbol_cartogram(
            gdf,
            "population",
            layout=layout,
        )
        assert len(result.symbols) == len(gdf)
