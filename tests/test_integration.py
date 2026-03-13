"""Integration tests for complete cartogram workflows."""

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


class TestIntegration:
    """Integration tests for complete cartogram workflows."""

    def test_flow_cartogram_workflow(self):
        """Test complete flow cartogram workflow."""
        from carto_flow.flow_cartogram import MorphOptions, morph_gdf

        gdf = make_grid_gdf()
        result = morph_gdf(
            gdf,
            "population",
            options=MorphOptions.preset_fast(),
        )

        assert result is not None
        assert result.status is not None

    def test_proportional_cartogram_workflow(self):
        """Test complete proportional cartogram workflow."""
        from carto_flow.proportional_cartogram import partition_geometries

        gdf = make_test_gdf()
        result = partition_geometries(
            gdf,
            columns=["population"],
            method="split",
            normalization="sum",
        )

        assert result is not None

    def test_symbol_cartogram_workflow(self):
        """Test complete symbol cartogram workflow."""
        from carto_flow.symbol_cartogram import CirclePhysicsLayout, create_symbol_cartogram

        gdf = make_test_gdf()
        layout = CirclePhysicsLayout(max_iterations=50)
        result = create_symbol_cartogram(
            gdf,
            "population",
            layout=layout,
            show_progress=False,
        )

        assert result is not None
        assert len(result.symbols) == len(gdf)

    def test_tiling_cartogram_workflow(self):
        """Test complete tiling cartogram workflow."""
        from carto_flow.symbol_cartogram import (
            GridBasedLayout,
            Styling,
            create_symbol_cartogram,
        )

        gdf = make_grid_gdf()
        layout = GridBasedLayout(tiling="hexagon")
        styling = Styling(symbol="hexagon")
        result = create_symbol_cartogram(
            gdf,
            "population",
            layout=layout,
            styling=styling,
            show_progress=False,
        )

        assert result is not None
        assert len(result.symbols) == len(gdf)
