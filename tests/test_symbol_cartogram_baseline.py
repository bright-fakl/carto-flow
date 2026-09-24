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
        from carto_flow.symbol_cartogram import CirclePackingLayout, create_symbol_cartogram

        gdf = make_test_gdf()
        layout = CirclePackingLayout(max_iterations=50)
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
        assert result.placement_metrics is not None
        assert "displacement_mean" in result.placement_metrics
        assert "displacement_max" in result.placement_metrics

    def test_uniform_size_mode(self):
        """Uniform sizing (no value_column) creates equal-sized symbols."""
        from carto_flow.symbol_cartogram import CirclePackingLayout, create_symbol_cartogram

        gdf = make_test_gdf()
        layout = CirclePackingLayout(max_iterations=50)
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
        assert result.placement_metrics is not None

    def test_single_geometry(self):
        """Single geometry returns immediately without simulation."""
        from carto_flow.symbol_cartogram import (
            SymbolCartogramStatus,
            create_symbol_cartogram,
        )

        gdf = make_test_gdf(n=1)
        result = create_symbol_cartogram(gdf, "population", show_progress=False)

        assert len(result.symbols) == 1
        assert result.status == SymbolCartogramStatus.CONVERGED
        assert result.placement_metrics["displacement_mean"] == 0.0

    def test_no_adjacent_pairs(self):
        """Input whose regions share no boundary lays out without error."""
        from carto_flow.symbol_cartogram import create_symbol_cartogram

        gdf = gpd.GeoDataFrame(
            {"population": [100.0, 200.0, 300.0]},
            geometry=[box(0, 0, 1, 1), box(10, 0, 11, 1), box(20, 0, 21, 1)],
        )
        result = create_symbol_cartogram(gdf, "population", show_progress=False)

        assert len(result.symbols) == 3

    def test_null_values_skipped(self):
        """Null values are skipped with warning."""
        from carto_flow.symbol_cartogram import CirclePackingLayout, create_symbol_cartogram

        gdf = make_test_gdf(n=5)
        gdf.loc[gdf.index[0], "population"] = np.nan
        layout = CirclePackingLayout(max_iterations=50)
        with pytest.warns(UserWarning, match="Skipping 1 rows"):
            result = create_symbol_cartogram(
                gdf,
                "population",
                layout=layout,
                show_progress=False,
            )

        assert len(result.symbols) == 4
        assert result.placement_metrics["n_skipped"] == 1

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
        from carto_flow.symbol_cartogram import CirclePackingLayout, create_symbol_cartogram

        gdf = make_grid_gdf(rows=2, cols=2)
        layout = CirclePackingLayout(max_iterations=20)
        result = create_symbol_cartogram(
            gdf,
            "population",
            layout=layout,
            show_progress=False,
            save_history=True,
        )

        assert result.layout_result.history is not None
        assert result.layout_result.history.positions is not None
        assert len(result.layout_result.history.positions) > 0

    def test_preset_classmethods(self):
        """CirclePackingLayout preset classmethods return configured instances."""
        from carto_flow.symbol_cartogram import CirclePackingLayout

        for name in ("centroid", "dorling", "geographic", "dorling_grouped", "geographic_grouped"):
            layout = getattr(CirclePackingLayout, name)()
            assert isinstance(layout, CirclePackingLayout)

    def test_preset_functions_importable(self):
        """Named cartogram functions are importable from the top-level package."""
        from carto_flow.symbol_cartogram import (
            centroid_cartogram,
            demers_cartogram,
            dorling_cartogram,
            dorling_grouped_cartogram,
            geographic_cartogram,
            geographic_grouped_cartogram,
            tile_map_cartogram,
        )

        for fn in (
            centroid_cartogram,
            demers_cartogram,
            dorling_cartogram,
            dorling_grouped_cartogram,
            geographic_cartogram,
            geographic_grouped_cartogram,
            tile_map_cartogram,
        ):
            assert callable(fn)

    def test_result_to_geodataframe(self):
        """to_geodataframe preserves original columns."""
        from carto_flow.symbol_cartogram import CirclePackingLayout, create_symbol_cartogram

        gdf = make_test_gdf()
        gdf["region_name"] = [f"Region_{i}" for i in range(len(gdf))]
        layout = CirclePackingLayout(max_iterations=50)
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

    def test_topology_preserving_simulator(self):
        """TopologyPreservingSimulator runs with kwargs."""
        from carto_flow.symbol_cartogram.layouts.packing._simulator import TopologyPreservingSimulator

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
            CirclePackingLayout,
            create_symbol_cartogram,
            plot_adjacency,
        )

        gdf = make_grid_gdf(rows=2, cols=2)
        layout = CirclePackingLayout(max_iterations=10)
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
        from carto_flow.symbol_cartogram import CirclePackingLayout, create_symbol_cartogram, plot_adjacency

        gdf = make_grid_gdf(rows=2, cols=2)
        layout = CirclePackingLayout(max_iterations=10)
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

    def test_plot_tiling_raises_for_packing(self):
        """plot_tiling raises ValueError for non-grid placement result."""
        import matplotlib

        matplotlib.use("Agg")
        from carto_flow.symbol_cartogram import CirclePackingLayout, create_symbol_cartogram, plot_tiling

        gdf = make_grid_gdf(rows=2, cols=2)
        layout = CirclePackingLayout(max_iterations=10)
        result = create_symbol_cartogram(
            gdf,
            "population",
            layout=layout,
            show_progress=False,
        )
        with pytest.raises(TypeError, match="grid or mosaic"):
            plot_tiling(result)


class TestSizeNormalizationDefault:
    """Which ``size_normalization`` the pipeline applies when none is given."""

    def test_prepare_layout_data_defaults_to_total(self):
        """Total symbol area matches total geometry area by default."""
        from carto_flow.symbol_cartogram.layouts import prepare_layout_data

        gdf = make_grid_gdf()
        data = prepare_layout_data(gdf, "population")

        total_symbol_area = float(np.pi * np.sum(data.sizes**2))
        assert total_symbol_area == pytest.approx(float(gdf.geometry.area.sum()))

    def test_radius_layouts_default_to_total(self):
        """Layouts that place symbols by radius inherit the ``"total"`` default."""
        from carto_flow.symbol_cartogram import create_layout

        gdf = make_grid_gdf()
        for layout in ("packing", "topology", "centroid"):
            data_sizes = create_layout(gdf, "population", layout=layout, show_progress=False).sizes
            total_symbol_area = float(np.pi * np.sum(data_sizes**2))
            assert total_symbol_area == pytest.approx(float(gdf.geometry.area.sum())), layout

    def test_grid_layout_defaults_to_max(self):
        """The grid layout keeps ``"max"``: the largest symbol is one unit cell.

        Its tile lattice is calibrated from the largest symbol, so ``"total"``
        would rescale lattice and symbols together and move the whole grid off
        the scale of the input geometries.
        """
        from carto_flow.symbol_cartogram import create_layout

        gdf = make_grid_gdf()
        result = create_layout(gdf, "population", layout="grid", show_progress=False)

        mean_area = float(gdf.geometry.area.mean())
        largest_symbol_area = float(np.pi * np.max(result.sizes) ** 2)
        assert largest_symbol_area == pytest.approx(mean_area)

    def test_explicit_value_overrides_the_layout_default(self):
        """A caller-supplied value wins over the layout's own default."""
        from carto_flow.symbol_cartogram import create_layout

        gdf = make_grid_gdf()
        result = create_layout(gdf, "population", layout="grid", size_normalization="total", show_progress=False)

        total_symbol_area = float(np.pi * np.sum(result.sizes**2))
        assert total_symbol_area == pytest.approx(float(gdf.geometry.area.sum()))

    def test_mosaic_layout_is_unaffected(self):
        """Mosaic symbol scale comes from the tile lattice, not the normalisation."""
        from carto_flow.symbol_cartogram import create_symbol_cartogram

        gdf = make_grid_gdf()
        as_max = create_symbol_cartogram(
            gdf, "population", layout="mosaic", size_normalization="max", show_progress=False
        )
        as_total = create_symbol_cartogram(
            gdf, "population", layout="mosaic", size_normalization="total", show_progress=False
        )

        np.testing.assert_allclose(
            as_max.symbols.geometry.area.to_numpy(),
            as_total.symbols.geometry.area.to_numpy(),
        )


class TestSizeNormalizationValidation:
    """An unrecognised ``size_normalization`` is rejected, not silently ignored."""

    @pytest.mark.parametrize("bad", ["Total", "totals", "mean", "", None, 1])
    def test_invalid_value_raises(self, bad):
        """Anything outside the two options raises rather than falling back."""
        from carto_flow.symbol_cartogram.layouts import prepare_layout_data

        gdf = make_grid_gdf()
        with pytest.raises(ValueError, match="Unknown size_normalization"):
            prepare_layout_data(gdf, "population", size_normalization=bad)

    def test_message_names_the_value_and_the_valid_options(self):
        """The message is actionable: what was passed, and what is accepted."""
        from carto_flow.symbol_cartogram.layouts import prepare_layout_data

        gdf = make_grid_gdf()
        with pytest.raises(ValueError) as excinfo:
            prepare_layout_data(gdf, "population", size_normalization="Total")

        message = str(excinfo.value)
        assert "'Total'" in message
        assert '"total"' in message
        assert '"max"' in message

    @pytest.mark.parametrize("valid", ["max", "total"])
    def test_valid_values_still_accepted(self, valid):
        """Both documented options keep working."""
        from carto_flow.symbol_cartogram.layouts import prepare_layout_data

        gdf = make_grid_gdf()
        data = prepare_layout_data(gdf, "population", size_normalization=valid)

        assert len(data.sizes) == len(gdf)

    def test_entry_points_reject_the_value_too(self):
        """The guard sits where every entry point funnels through."""
        from carto_flow.symbol_cartogram import create_layout, create_symbol_cartogram

        gdf = make_grid_gdf()
        for call in (create_layout, create_symbol_cartogram):
            with pytest.raises(ValueError, match="Unknown size_normalization"):
                call(gdf, "population", size_normalization="mean", show_progress=False)

    def test_layout_defaults_still_resolve(self):
        """Omitting the argument still reaches the layout's own default."""
        from carto_flow.symbol_cartogram import create_layout

        gdf = make_grid_gdf()
        packed = create_layout(gdf, "population", layout="topology", show_progress=False)
        gridded = create_layout(gdf, "population", layout="grid", show_progress=False)

        assert float(np.pi * np.sum(packed.sizes**2)) == pytest.approx(float(gdf.geometry.area.sum()))
        assert float(np.pi * np.max(gridded.sizes) ** 2) == pytest.approx(float(gdf.geometry.area.mean()))


# Registry key -> the ``layout_type`` the layout records on its result.
EXPECTED_LAYOUT_TYPES = {
    "centroid": "centroid",
    "flow_density": "flow_density",
    "grid": "grid",
    "mosaic": "mosaic",
    "packing": "packing",
    "topology": "packing",  # registered alias of the packing layout
}


class TestLayoutTypeProvenance:
    """``layout_type`` names the layout that produced the result.

    Its one functional job is selecting the result class when a serialized
    result is read back, so both halves are pinned here.
    """

    def test_every_registered_layout_is_covered(self):
        """The table above lists every registry key, so none can drift."""
        from carto_flow.symbol_cartogram.layouts.base import _LAYOUT_REGISTRY

        assert set(_LAYOUT_REGISTRY) == set(EXPECTED_LAYOUT_TYPES)

    @pytest.mark.parametrize("name", sorted(EXPECTED_LAYOUT_TYPES))
    def test_layout_type_is_the_registry_key(self, name):
        from carto_flow.symbol_cartogram import create_layout

        gdf = make_grid_gdf(rows=3, cols=3)
        result = create_layout(gdf, "population", layout=name, show_progress=False)

        assert result.layout_type == EXPECTED_LAYOUT_TYPES[name]

    @pytest.mark.parametrize("name", sorted(EXPECTED_LAYOUT_TYPES))
    def test_serialize_round_trip_restores_the_result_class(self, name):
        from carto_flow.symbol_cartogram import create_layout
        from carto_flow.symbol_cartogram.layouts.layout_result import (
            GridLayoutResult,
            LayoutResult,
            MosaicLayoutResult,
        )

        gdf = make_grid_gdf(rows=3, cols=3)
        result = create_layout(gdf, "population", layout=name, show_progress=False)

        restored = LayoutResult.from_serialized(result.serialize())

        expected_cls = {"grid": GridLayoutResult, "mosaic": MosaicLayoutResult}.get(
            EXPECTED_LAYOUT_TYPES[name], LayoutResult
        )
        assert type(restored) is expected_cls
        assert restored.layout_type == EXPECTED_LAYOUT_TYPES[name]
        np.testing.assert_allclose(restored.positions, result.positions)
