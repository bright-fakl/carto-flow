"""Tests for the style of the symbol plots."""

import geopandas as gpd
import matplotlib
import numpy as np
import pytest
from shapely.geometry import box

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgba

from carto_flow import Channel, Palettes, PlotStyle, set_style
from carto_flow.symbol_cartogram import (
    CirclePackingLayout,
    GridBasedLayout,
    create_symbol_cartogram,
    plot_adjacency,
    plot_adjacency_heatmap,
    plot_comparison,
    plot_displacement,
    plot_tiling,
)


@pytest.fixture(autouse=True)
def reset():
    yield
    set_style(None)
    plt.close("all")


@pytest.fixture(scope="module")
def gdf():
    rng = np.random.default_rng(0)
    geoms = [box(c, r, c + 1, r + 1) for r in range(3) for c in range(3)]
    return gpd.GeoDataFrame(
        {
            "population": rng.integers(100, 10000, size=9).astype(float),
            "change": np.linspace(-5, 15, 9),
            "name": list("abcdefghi"),
            "group": list("aaabbbccc"),
        },
        geometry=geoms,
    )


@pytest.fixture(scope="module")
def result(gdf):
    return create_symbol_cartogram(
        gdf, "population", layout=CirclePackingLayout(max_iterations=20), show_progress=False
    )


@pytest.fixture(scope="module")
def grid_result(gdf):
    return create_symbol_cartogram(gdf, "population", layout=GridBasedLayout(), show_progress=False)


def _fill(plot_result):
    return np.concatenate([np.asarray(c.get_facecolor()) for c in plot_result.collections])


def _colorbar_axes(fig):
    return [a for a in fig.axes if a.get_label() == "<colorbar>"]


class TestSymbols:
    def test_default_style_draws_the_surface_and_palette(self, result):
        plain = result.plot(style="classic", facecolor="population")
        styled_ = result.plot(facecolor="population")
        assert to_hex(styled_.ax.get_facecolor()) == "#eef1f4"
        assert not np.allclose(_fill(plain), _fill(styled_))

    def test_categorical_uses_the_categorical_palette(self, result):
        palette = ["#ff0000", "#00ff00", "#0000ff"]
        style = PlotStyle(palettes=Palettes(categorical=palette))
        plot = result.plot(facecolor="group", style=style)
        fill = {to_hex(c) for c in _fill(plot)}
        assert fill == set(palette)

    def test_cmap_dict_overrides_the_style(self, result):
        style = PlotStyle(palettes=Palettes(categorical=["#00ff00", "#0000ff", "#ffff00"]))
        plot = result.plot(facecolor="group", cmap={"a": "#ff0000"}, style=style)
        assert "#ff0000" in {to_hex(c) for c in _fill(plot)}

    def test_sequential_palette_of_the_style(self, result):
        style = PlotStyle(palettes=Palettes(sequential="Reds"))
        plot = result.plot(facecolor="population", style=style, legend=False)
        explicit = result.plot(facecolor="population", cmap="Reds", style="classic", legend=False)
        assert np.allclose(_fill(plot), _fill(explicit))

    def test_cmap_of_the_call_wins(self, result):
        style = PlotStyle(palettes=Palettes(sequential="Reds"))
        plot = result.plot(facecolor="population", cmap="Blues", style=style, legend=False)
        explicit = result.plot(facecolor="population", cmap="Blues", style="classic", legend=False)
        assert np.allclose(_fill(plot), _fill(explicit))

    def test_diverging_role_by_center(self, result):
        style = PlotStyle(palettes=Palettes(diverging="RdBu"))
        plot = result.plot(facecolor="change", center=0, style=style, legend=False)
        explicit = result.plot(facecolor="change", cmap="RdBu", style="classic", legend=False)
        assert not np.allclose(_fill(plot), _fill(explicit))  # symmetric about the center

    def test_categorical_flag_makes_numbers_categories(self, result):
        plot = result.plot(facecolor="population", categorical=True)
        assert plot.legend is not None
        assert not _colorbar_axes(plot.ax.figure)

    def test_edge_and_linewidth_channels(self, result):
        style = PlotStyle(channels={"edge": Channel(value="#123456"), "linewidth": Channel(value=2.0)})
        plot = result.plot(style=style)
        collection = plot.collections[0]
        assert to_hex(collection.get_edgecolor()[0]) == "#123456"
        assert collection.get_linewidth()[0] == 2.0

    def test_call_edge_wins_over_the_style(self, result):
        style = PlotStyle(channels={"edge": Channel(value="#123456")})
        plot = result.plot(edgecolor="#654321", style=style)
        assert to_hex(plot.collections[0].get_edgecolor()[0]) == "#654321"

    def test_fill_channel_value(self, result):
        style = PlotStyle(channels={"fill": Channel(value="#abcdef")})
        plot = result.plot(style=style, alpha=1.0)
        assert to_hex(plot.collections[0].get_facecolor()[0]) == "#abcdef"

    def test_alpha_range_channel(self, result):
        style = PlotStyle(channels={"alpha": Channel(range=(0.5, 0.6))})
        plot = result.plot(alpha="population", style=style, legend=False)
        alphas = np.asarray(plot.collections[0].get_facecolor())[:, 3]
        assert alphas.min() >= 0.5 - 1e-9
        assert alphas.max() <= 0.6 + 1e-9

    def test_hatch_cycle(self, result):
        style = PlotStyle(hatch_cycle=("x", "o"))
        plot = result.plot(hatch="group", edgecolor="black", style=style)
        hatches = {h for c in plot.collections for h in [c.get_hatch()] if h}
        assert hatches <= {"x", "o", None}
        assert hatches

    def test_label_color_follows_the_text_color(self, result):
        plot = result.plot(labels="name", style="dark")
        assert to_hex(plot.labels[0].get_color()) == "#e6e9ec"

    def test_annotation_size(self, result):
        plot = result.plot(labels="name", style=PlotStyle(annotation_size=13))
        assert plot.labels[0].get_fontsize() == 13

    def test_legend_label_is_respected(self, result):
        plot = result.plot(facecolor="population", legend_kwargs={"label": "People"})
        assert plot.colorbar.ax.get_ylabel() == "People"

    def test_log_scale_colorbar(self, result):
        style = PlotStyle(color_scale="log")
        plot = result.plot(facecolor="population", style=style)
        assert plot.colorbar.norm.__class__.__name__ == "LogNorm"

    def test_session_style_applies(self, result):
        set_style("dark")
        plot = result.plot()
        assert to_hex(plot.ax.get_facecolor()) == "#1b1f24"

    def test_style_context_is_restored(self, result):
        result.plot(style="dark")
        assert to_hex(result.plot().ax.get_facecolor()) == "#eef1f4"


class TestOtherPlots:
    def test_comparison_original_panel_uses_the_palette(self, gdf, result):
        style = PlotStyle(palettes=Palettes(categorical=["#ff0000", "#00ff00", "#0000ff"]))
        plot = plot_comparison(gdf, result, facecolor="group", style=("default", style))
        original = {to_hex(c) for col in plot.original_collections for c in col.get_facecolor()}
        assert original == {"#ff0000", "#00ff00", "#0000ff"}
        assert to_hex(plot.ax_original.get_facecolor()) == "#eef1f4"

    def test_comparison_classic_matches_geopandas(self, gdf, result):
        plot = plot_comparison(gdf, result, facecolor="population", cmap="plasma", style="classic")
        ref = gdf.plot(column="population", cmap="plasma").collections[0]
        assert np.allclose(plot.original_collections[0].get_facecolor(), ref.get_facecolor())

    def test_adjacency_uses_the_sequential_palette(self, result):
        style = PlotStyle(palettes=Palettes(sequential="Reds"))
        plot = plot_adjacency(result, style=style, link_legend=False)
        explicit = plot_adjacency(result, link_cmap="Reds", style="classic", link_legend=False)
        assert np.allclose(plot.edges.get_colors(), explicit.edges.get_colors())

    def test_adjacency_link_cmap_wins(self, result):
        style = PlotStyle(palettes=Palettes(sequential="Reds"))
        plot = plot_adjacency(result, link_cmap="Blues", style=style, link_legend=False)
        explicit = plot_adjacency(result, link_cmap="Blues", style="classic", link_legend=False)
        assert np.allclose(plot.edges.get_colors(), explicit.edges.get_colors())

    def test_adjacency_legend_label_in_kwargs(self, result):
        plot = plot_adjacency(result, link_legend_kwargs={"label": "w"})
        assert plot.colorbar.ax.get_ylabel() == "w"

    def test_heatmap_uses_the_sequential_palette(self, result):
        style = PlotStyle(palettes=Palettes(sequential="Reds"))
        plot = plot_adjacency_heatmap(result, style=style)
        assert plot.image.get_cmap().name == "Reds"

    def test_heatmap_default_without_a_palette(self, result):
        plot = plot_adjacency_heatmap(result, style="classic")
        assert plot.image.get_cmap().name == "YlOrRd"

    def test_tiling_uses_the_reference_layer(self, grid_result):
        style = PlotStyle(
            reference_facecolor="#102030", reference_edgecolor="#405060", reference_linewidth=2.0, reference_alpha=0.9
        )
        plot = plot_tiling(grid_result, show_symbols=False, style=style)
        tiles = plot.unassigned_tiles or plot.assigned_tiles
        assert tiles is not None
        assert to_hex(tiles.get_edgecolor()[0]) == "#405060"
        assert tiles.get_linewidth()[0] == 2.0

    def test_tiling_call_wins(self, grid_result):
        style = PlotStyle(reference_edgecolor="#405060")
        plot = plot_tiling(grid_result, show_symbols=False, tile_edgecolor="#010203", style=style)
        tiles = plot.unassigned_tiles or plot.assigned_tiles
        assert to_hex(tiles.get_edgecolor()[0]) == "#010203"

    def test_tiling_classic_defaults(self, grid_result):
        plot = plot_tiling(grid_result, show_symbols=False, style="classic")
        tiles = plot.assigned_tiles
        assert to_hex(tiles.get_edgecolor()[0]) == "#999999"
        assert to_rgba("#d4e6f1")[:3] == tuple(tiles.get_facecolor()[0][:3])

    def test_tiling_method_accepts_style(self, grid_result):
        plot = grid_result.plot_tiling(show_symbols=False, style="dark")
        assert plot.ax is not None

    def test_displacement_surface(self, result):
        plot = plot_displacement(result, style="dark")
        assert to_hex(plot.ax.figure.get_facecolor()) == "#1b1f24"

    def test_adjacency_colorbar_axes_in_legend_kwargs(self, result):
        _, axes = plt.subplots(1, 2)
        plot = plot_adjacency(result, ax=axes[1], link_legend_kwargs={"ax": axes})
        assert plot.colorbar is not None
