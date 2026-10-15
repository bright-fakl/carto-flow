"""Tests for the parameter vocabulary of the symbol cartogram plot functions."""

import warnings

import geopandas as gpd
import matplotlib
import numpy as np
import pytest
from shapely.geometry import box

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from carto_flow.symbol_cartogram import (
    CirclePackingLayout,
    GridBasedLayout,
    IsohedralTiling,
    SquareTiling,
    Styling,
    create_symbol_cartogram,
    plot_adjacency,
    plot_adjacency_heatmap,
    plot_displacement,
    plot_tiling,
)
from carto_flow.symbol_cartogram.visualization import plot_comparison


def _gdf(rows=2, cols=2):
    rng = np.random.default_rng(0)
    geoms = [box(c, r, c + 1, r + 1) for r in range(rows) for c in range(cols)]
    n = rows * cols
    return gpd.GeoDataFrame(
        {"population": rng.integers(100, 10000, size=n).astype(float), "name": list("abcd")[:n]}, geometry=geoms
    )


@pytest.fixture(scope="module")
def gdf():
    return _gdf()


@pytest.fixture(scope="module")
def packed(gdf):
    return create_symbol_cartogram(
        gdf, "population", layout=CirclePackingLayout(max_iterations=10), show_progress=False
    )


@pytest.fixture(scope="module")
def tiled(gdf):
    return create_symbol_cartogram(
        gdf,
        "population",
        layout=GridBasedLayout(tiling="square"),
        styling=Styling(symbol="square"),
        show_progress=False,
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _colors(result):
    return np.concatenate([np.asarray(c.get_facecolor()) for c in result.collections])


class TestSymbolPlot:
    def test_facecolor_binds_data_and_old_column_is_an_alias(self, packed):
        new = packed.plot(facecolor="population", legend=False)
        with pytest.warns(DeprecationWarning, match="use 'facecolor'"):
            old = packed.plot(column="population", legend=False)
        np.testing.assert_allclose(_colors(new), _colors(old))

    def test_old_column_and_new_facecolor_together_keep_the_new_one(self, packed):
        with pytest.warns(DeprecationWarning):
            both = packed.plot(column="population", facecolor="#4e79a7", legend=False)
        np.testing.assert_allclose(_colors(both), _colors(packed.plot(facecolor="#4e79a7", legend=False)))

    def test_legend_kwargs_and_labels(self, packed):
        packed.plot(facecolor="population", legend_kwargs={"shrink": 0.5}, labels="name")
        with pytest.warns(DeprecationWarning, match="'legend_kwds' is deprecated; use 'legend_kwargs'"):
            packed.plot(facecolor="population", legend_kwds={"shrink": 0.5})
        with pytest.warns(DeprecationWarning, match="'label' is deprecated; use 'labels'"):
            packed.plot(label="name")

    def test_other_legend_kwargs_names(self, packed):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            packed.plot(
                facecolor="population",
                edgecolor="name",
                edge_legend_kwargs={"loc": "upper left"},
                hatch="name",
                hatch_legend_kwargs={"loc": "upper left"},
                linewidth="population",
                linewidth_legend_kwargs={"loc": "upper left"},
                alpha="population",
                alpha_legend_kwargs={"shrink": 0.5},
                labels="name",
                label_color="population",
                label_legend_kwargs={"shrink": 0.5},
            )
        with pytest.warns(DeprecationWarning, match="'hatch_legend_kwds' is deprecated"):
            packed.plot(hatch="name", hatch_legend_kwds={"loc": "upper left"})

    def test_plot_comparison(self, gdf, packed):
        plot_comparison(gdf, packed, facecolor="population")
        plot_comparison(gdf, packed, facecolor="#4e79a7")
        with pytest.warns(DeprecationWarning, match="use 'facecolor'"):
            plot_comparison(gdf, packed, column="population")


class TestOtherPlots:
    def test_adjacency_link_names(self, packed):
        plot_adjacency(packed, link_color="red", link_alpha=0.5, link_linewidth=2, node_markersize=10)
        plot_adjacency(packed, link_cmap="cool", link_legend=True, link_legend_kwargs={"shrink": 0.5})
        old_calls = {
            "edge_color": ("link_color", "red"),
            "edge_cmap": ("link_cmap", "cool"),
            "edge_alpha": ("link_alpha", 0.5),
            "edge_width": ("link_linewidth", 2),
            "node_size": ("node_markersize", 10),
            "colorbar": ("link_legend", False),
            "colorbar_kwds": ("link_legend_kwargs", {"shrink": 0.5}),
        }
        for old, (new, value) in old_calls.items():
            with pytest.warns(DeprecationWarning, match=f"'{old}' is deprecated; use '{new}'"):
                plot_adjacency(packed, **{old: value})

    def test_adjacency_heatmap_names(self, packed):
        plot_adjacency_heatmap(packed, legend=True, legend_kwargs={"shrink": 0.5}, tick_labelsize=8)
        with pytest.warns(DeprecationWarning, match="'colorbar' is deprecated; use 'legend'"):
            plot_adjacency_heatmap(packed, colorbar=False)
        with pytest.warns(DeprecationWarning, match="'tick_fontsize' is deprecated; use 'tick_labelsize'"):
            plot_adjacency_heatmap(packed, tick_fontsize=8)

    def test_displacement_arrows_kwargs(self, packed):
        result = plot_displacement(packed, arrows_kwargs={"color": "black"})
        assert result.arrows is not None
        with pytest.warns(DeprecationWarning, match="arrow_color"):
            old = plot_displacement(packed, arrow_color="black", arrow_alpha=0.3)
        assert old.arrows.get_alpha() == 0.3

    def test_tiling_facecolors(self, tiled):
        plot_tiling(tiled, assigned_facecolor="#e5e5e5", unassigned_facecolor="0.95")
        with pytest.warns(DeprecationWarning, match="'assigned_color' is deprecated; use 'assigned_facecolor'"):
            plot_tiling(tiled, assigned_color="#e5e5e5")
        with pytest.warns(DeprecationWarning, match="'unassigned_color' is deprecated"):
            tiled.plot_tiling(unassigned_color="0.95")

    def test_tiling_object_plots(self):
        tiling = SquareTiling().generate()
        tiling.plot(facecolor="orange", edgecolor="black", cmap="hot", color_by="index")
        with pytest.warns(DeprecationWarning, match="'face_color' is deprecated; use 'facecolor'"):
            tiling.plot(face_color="orange")
        with pytest.warns(DeprecationWarning, match="'colormap' is deprecated; use 'cmap'"):
            tiling.plot(colormap="hot", color_by="index")
        prototile = IsohedralTiling(1).generate()
        prototile.plot_tile(facecolor="orange", edgecolor="black")
        with pytest.warns(DeprecationWarning, match="'edge_color' is deprecated; use 'edgecolor'"):
            prototile.plot_tile(edge_color="black")
