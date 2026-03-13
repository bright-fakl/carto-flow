"""Tests to improve symbol cartogram test coverage."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import box


def make_test_gdf(n: int = 5, seed: int = 42) -> gpd.GeoDataFrame:
    """Create a simple test GeoDataFrame with adjacent squares."""
    rng = np.random.default_rng(seed)
    geoms = []
    for i in range(n):
        x = i * 2.0
        y = 0.0
        geoms.append(box(x, y, x + 2.0, y + 2.0))
    return gpd.GeoDataFrame(
        {"population": rng.integers(100, 10000, size=n).astype(float)},
        geometry=geoms,
    )


def make_grid_gdf(rows: int = 3, cols: int = 3, seed: int = 42) -> gpd.GeoDataFrame:
    """Create a grid of adjacent squares."""
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


class TestSymbolCartogramCoverage:
    """Tests to improve coverage for symbol cartogram module."""

    def test_result_properties(self):
        """Test various properties of SymbolCartogramResult."""
        from carto_flow.symbol_cartogram import CirclePhysicsLayout, create_symbol_cartogram
        from carto_flow.symbol_cartogram.status import SymbolCartogramStatus

        gdf = make_test_gdf()
        layout = CirclePhysicsLayout(max_iterations=50)
        result = create_symbol_cartogram(
            gdf,
            "population",
            layout=layout,
            show_progress=False,
        )

        # Test status property
        assert result.status is not None
        assert result.status == SymbolCartogramStatus.COMPLETED

        # Test metrics property
        assert result.metrics is not None
        assert "displacement_mean" in result.metrics
        assert "displacement_max" in result.metrics
        assert "displacement_std" in result.metrics

        # Test simulation_history property (when not saved)
        assert result.simulation_history is None

    def test_compute_adjacency_binary_mode(self):
        """Test compute_adjacency with binary mode."""
        from carto_flow.symbol_cartogram.adjacency import compute_adjacency
        from carto_flow.symbol_cartogram.options import AdjacencyMode

        gdf = make_grid_gdf(rows=2, cols=2)
        adj = compute_adjacency(gdf, mode=AdjacencyMode.BINARY)

        assert adj.shape == (4, 4)
        assert np.allclose(adj, adj.T)  # Binary mode is symmetric
        assert np.all(np.diag(adj) == 0)  # No self-adjacency

    def test_styling_symbol_options(self):
        """Test Styling class with different symbol options."""
        from carto_flow.symbol_cartogram import GridBasedLayout, Styling, create_symbol_cartogram

        gdf = make_grid_gdf()
        layout = GridBasedLayout(tiling="hexagon")
        styling = Styling(symbol="circle")
        result = create_symbol_cartogram(
            gdf,
            "population",
            layout=layout,
            styling=styling,
            show_progress=False,
        )

        assert len(result.symbols) == len(gdf)

    def test_layout_options(self):
        """Test layout options for grid and physics layouts."""
        from carto_flow.symbol_cartogram import GridBasedLayout, create_symbol_cartogram

        gdf = make_grid_gdf()
        layout = GridBasedLayout(tiling="square", fill_holes=True)
        result = create_symbol_cartogram(
            gdf,
            "population",
            layout=layout,
            show_progress=False,
        )

        assert len(result.symbols) == len(gdf)

    def test_tiling_creation(self):
        """Test creating tilings directly."""
        from carto_flow.symbol_cartogram.tiling import (
            HexagonTiling,
            SquareTiling,
        )

        square = SquareTiling()
        hexagon = HexagonTiling()

        assert square is not None
        assert hexagon is not None

    def test_tiling_properties(self):
        """Test tiling properties."""
        from carto_flow.symbol_cartogram.tiling import SquareTiling

        tiling = SquareTiling()
        result = tiling.generate((0, 0, 5, 5))

        assert result is not None
        assert result.polygons is not None
        assert len(result.polygons) > 0
        assert result.centers is not None
        assert len(result.centers) == len(result.polygons)
