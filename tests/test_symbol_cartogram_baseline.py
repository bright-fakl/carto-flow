"""Baseline tests for symbol_cartogram behavior.

These tests capture the current API contract so we can verify that the
code produces expected results.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
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


class TestCurrentAPIContract:
    """Tests capturing the current API that must be preserved."""

    def test_basic_proportional_cartogram(self):
        """Basic proportional cartogram returns valid result."""
        from carto_flow.symbol_cartogram import CirclePhysicsLayout, create_symbol_cartogram

        gdf = make_test_gdf()
        layout = CirclePhysicsLayout(max_iterations=50)
        result = create_symbol_cartogram(
            gdf,
            "population",
            layout=layout,
            show_progress=False,
        )

        assert result.symbols is not None
        assert len(result.symbols) == len(gdf)
        assert "_symbol_x" in result.symbols.columns
        assert "_symbol_y" in result.symbols.columns
        assert "_symbol_size" in result.symbols.columns
        assert "_displacement" in result.symbols.columns
        assert result.status is not None
        assert result.metrics is not None
        assert "displacement_mean" in result.metrics
        assert "displacement_max" in result.metrics

    def test_uniform_size_mode(self):
        """Uniform sizing (no value_column) creates equal-sized symbols."""
        from carto_flow.symbol_cartogram import CirclePhysicsLayout, create_symbol_cartogram

        gdf = make_test_gdf()
        layout = CirclePhysicsLayout(max_iterations=50)
        result = create_symbol_cartogram(
            gdf,
            layout=layout,
            show_progress=False,
        )

        sizes = result.symbols["_symbol_size"].values
        assert np.allclose(sizes, sizes[0])

    def test_grid_placement(self):
        """Grid placement works with default options."""
        from carto_flow.symbol_cartogram import (
            GridBasedLayout,
            Styling,
            create_symbol_cartogram,
        )

        gdf = make_grid_gdf()
        layout = GridBasedLayout(tiling="square")
        styling = Styling(symbol="square")
        result = create_symbol_cartogram(
            gdf,
            "population",
            layout=layout,
            styling=styling,
        )

        assert len(result.symbols) == len(gdf)

    def test_topology_preserving_simulator(self):
        """Topology-preserving simulator runs successfully."""
        from carto_flow.symbol_cartogram import (
            CirclePackingLayout,
            CirclePackingLayoutOptions,
            create_symbol_cartogram,
        )

        gdf = make_grid_gdf(rows=2, cols=3)
        layout = CirclePackingLayout(CirclePackingLayoutOptions(max_iterations=30))
        result = create_symbol_cartogram(
            gdf,
            "population",
            layout=layout,
            show_progress=False,
        )

        assert len(result.symbols) == len(gdf)
        assert result.metrics is not None

    def test_single_geometry(self):
        """Single geometry returns immediately without simulation."""
        from carto_flow.symbol_cartogram import (
            SymbolCartogramStatus,
            create_symbol_cartogram,
        )

        gdf = make_test_gdf(n=1)
        result = create_symbol_cartogram(gdf, "population", show_progress=False)

        assert len(result.symbols) == 1
        assert result.status == SymbolCartogramStatus.COMPLETED
        assert result.metrics["displacement_mean"] == 0.0

    def test_null_values_skipped(self):
        """Null values are skipped with warning."""
        from carto_flow.symbol_cartogram import CirclePhysicsLayout, create_symbol_cartogram

        gdf = make_test_gdf(n=5)
        gdf.loc[gdf.index[0], "population"] = np.nan
        layout = CirclePhysicsLayout(max_iterations=50)
        with pytest.warns(UserWarning, match="Skipping 1 rows"):
            result = create_symbol_cartogram(
                gdf,
                "population",
                layout=layout,
                show_progress=False,
            )

        assert len(result.symbols) == 4
        assert result.metrics["n_skipped"] == 1

    def test_empty_gdf_raises(self):
        """Empty GeoDataFrame raises ValueError."""
        from carto_flow.symbol_cartogram import create_symbol_cartogram

        gdf = gpd.GeoDataFrame({"population": []}, geometry=[])
        with pytest.raises(ValueError, match="empty"):
            create_symbol_cartogram(gdf, "population")

    def test_missing_column_raises(self):
        """Missing value column raises ValueError."""
        from carto_flow.symbol_cartogram import create_symbol_cartogram

        gdf = make_test_gdf()
        with pytest.raises(ValueError, match="not found"):
            create_symbol_cartogram(gdf, "nonexistent")

    def test_save_history(self):
        """History is saved when requested."""
        from carto_flow.symbol_cartogram import CirclePhysicsLayout, create_symbol_cartogram

        gdf = make_grid_gdf(rows=2, cols=2)
        layout = CirclePhysicsLayout(max_iterations=20)
        result = create_symbol_cartogram(
            gdf,
            "population",
            layout=layout,
            show_progress=False,
            save_history=True,
        )

        assert result.simulation_history is not None
        assert result.simulation_history.positions is not None
        assert len(result.simulation_history.positions) > 0

    def test_presets_return_dicts(self):
        """All presets are callable and produce valid kwarg dicts."""
        from carto_flow.symbol_cartogram import (
            preset_demers,
            preset_dorling,
            preset_fast,
            preset_quality,
            preset_tile_map,
            preset_topology_preserving,
        )

        presets = [
            preset_dorling,
            preset_topology_preserving,
            preset_demers,
            preset_tile_map,
            preset_fast,
            preset_quality,
        ]
        for preset_fn in presets:
            kwargs = preset_fn()
            assert isinstance(kwargs, dict)
            assert "layout" in kwargs

    def test_result_to_geodataframe(self):
        """to_geodataframe preserves original columns."""
        from carto_flow.symbol_cartogram import CirclePhysicsLayout, create_symbol_cartogram

        gdf = make_test_gdf()
        gdf["region_name"] = [f"Region_{i}" for i in range(len(gdf))]
        layout = CirclePhysicsLayout(max_iterations=50)
        result = create_symbol_cartogram(
            gdf,
            "population",
            layout=layout,
            show_progress=False,
        )

        exported = result.to_geodataframe()
        assert "region_name" in exported.columns
        assert "population" in exported.columns


class TestDirectSimulatorUsage:
    """Tests for direct simulator usage that must remain unchanged."""

    def test_circle_physics_simulator(self):
        """CirclePhysicsSimulator runs with kwargs."""
        from carto_flow.symbol_cartogram.placement import CirclePhysicsSimulator

        n = 5
        positions = np.random.default_rng(42).uniform(0, 10, (n, 2))
        radii = np.full(n, 1.0)

        sim = CirclePhysicsSimulator(
            positions=positions.copy(),
            radii=radii,
            original_positions=positions,
            adjacency=None,
            spacing=0.05,
            compactness=0.5,
            topology_weight=0.0,
        )
        final_positions, info, _history = sim.run(
            max_iterations=20,
            show_progress=False,
            save_history=False,
        )
        assert final_positions.shape == (n, 2)
        assert "converged" in info
        assert "iterations" in info

    def test_topology_preserving_simulator(self):
        """TopologyPreservingSimulator runs with kwargs."""
        from carto_flow.symbol_cartogram.placement import TopologyPreservingSimulator

        n = 4
        positions = np.array([[0, 0], [2, 0], [0, 2], [2, 2]], dtype=float)
        radii = np.full(n, 0.8)
        adjacency = np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 1, 0],
            ],
            dtype=float,
        )

        sim = TopologyPreservingSimulator(
            positions=positions.copy(),
            radii=radii,
            original_positions=positions,
            adjacency=adjacency,
            spacing=0.05,
            compactness=0.5,
            topology_weight=0.3,
        )
        final_positions, info, _history = sim.run(
            max_iterations=20,
            show_progress=False,
            save_history=False,
        )
        assert final_positions.shape == (n, 2)
        assert "converged" in info

    def test_compute_adjacency(self):
        """compute_adjacency produces correct adjacency matrix."""
        from carto_flow.symbol_cartogram.adjacency import compute_adjacency
        from carto_flow.symbol_cartogram.options import AdjacencyMode

        gdf = make_grid_gdf(rows=2, cols=2)
        adj = compute_adjacency(gdf, mode=AdjacencyMode.BINARY)

        assert adj.shape == (4, 4)
        assert np.allclose(adj, adj.T)  # Binary mode is symmetric
        assert np.all(np.diag(adj) == 0)  # No self-adjacency
        # Each corner square touches 2 neighbors in a 2x2 grid
        assert np.sum(adj > 0) > 0


class TestVisualization:
    """Smoke tests for visualization functions."""

    def test_plot_adjacency_runs(self):
        """plot_adjacency returns AdjacencyPlotResult without error."""
        import matplotlib

        matplotlib.use("Agg")
        from carto_flow.symbol_cartogram import (
            AdjacencyPlotResult,
            CirclePhysicsLayout,
            create_symbol_cartogram,
            plot_adjacency,
        )

        gdf = make_grid_gdf(rows=2, cols=2)
        layout = CirclePhysicsLayout(max_iterations=10)
        result = create_symbol_cartogram(
            gdf,
            "population",
            layout=layout,
            show_progress=False,
        )
        plot_result = plot_adjacency(result)
        assert isinstance(plot_result, AdjacencyPlotResult)
        assert plot_result.ax is not None

    def test_plot_adjacency_with_edge_color(self):
        """plot_adjacency accepts edge_color parameter."""
        import matplotlib

        matplotlib.use("Agg")
        from carto_flow.symbol_cartogram import CirclePhysicsLayout, create_symbol_cartogram, plot_adjacency

        gdf = make_grid_gdf(rows=2, cols=2)
        layout = CirclePhysicsLayout(max_iterations=10)
        result = create_symbol_cartogram(
            gdf,
            "population",
            layout=layout,
            show_progress=False,
        )
        plot_result = plot_adjacency(result, edge_color="red", show_symbols=False)
        assert plot_result.ax is not None

    def test_plot_tiling_runs(self):
        """plot_tiling returns TilingPlotResult for grid placement result."""
        import matplotlib

        matplotlib.use("Agg")
        from carto_flow.symbol_cartogram import (
            GridBasedLayout,
            Styling,
            TilingPlotResult,
            create_symbol_cartogram,
            plot_tiling,
        )

        gdf = make_grid_gdf(rows=2, cols=2)
        layout = GridBasedLayout(tiling="square")
        styling = Styling(symbol="square")
        result = create_symbol_cartogram(
            gdf,
            "population",
            layout=layout,
            styling=styling,
        )
        plot_result = plot_tiling(result)
        assert isinstance(plot_result, TilingPlotResult)
        assert plot_result.ax is not None

    def test_plot_tiling_raises_for_physics(self):
        """plot_tiling raises ValueError for non-grid placement result."""
        import matplotlib

        matplotlib.use("Agg")
        from carto_flow.symbol_cartogram import CirclePhysicsLayout, create_symbol_cartogram, plot_tiling

        gdf = make_grid_gdf(rows=2, cols=2)
        layout = CirclePhysicsLayout(max_iterations=10)
        result = create_symbol_cartogram(
            gdf,
            "population",
            layout=layout,
            show_progress=False,
        )
        with pytest.raises(ValueError, match="grid-based"):
            plot_tiling(result)
