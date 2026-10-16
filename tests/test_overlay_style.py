"""Tests for the overlay colors of a style."""

import geopandas as gpd
import matplotlib
import numpy as np
import pytest
from shapely.geometry import box

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

from carto_flow import PlotStyle, set_style
from carto_flow.symbol_cartogram import CirclePackingLayout, create_symbol_cartogram, plot_displacement
from carto_flow.voronoi_cartogram import RasterBackend, VoronoiOptions, create_voronoi_cartogram
from carto_flow.voronoi_cartogram.visualization import plot_convergence
from carto_flow.voronoi_cartogram.visualization import plot_displacement as voronoi_displacement


@pytest.fixture(autouse=True)
def reset():
    yield
    set_style(None)
    plt.close("all")


@pytest.fixture(scope="module")
def gdf():
    rng = np.random.default_rng(0)
    return gpd.GeoDataFrame(
        {"population": rng.integers(100, 10000, size=9).astype(float)},
        geometry=[box(c, r, c + 1, r + 1) for r in range(3) for c in range(3)],
    )


@pytest.fixture(scope="module")
def voronoi(gdf):
    return create_voronoi_cartogram(gdf, backend=RasterBackend(resolution=50), options=VoronoiOptions(n_iter=5))


@pytest.fixture(scope="module")
def symbols(gdf):
    return create_symbol_cartogram(
        gdf, "population", layout=CirclePackingLayout(max_iterations=20), show_progress=False
    )


def test_pick():
    style = PlotStyle(accent_color="#123456")
    assert style.pick("accent_color", "red") == "#123456"
    assert style.pick("positive_color", "green") == "green"


def test_convergence_colors(voronoi):
    style = PlotStyle(series_color="#111111", accent_color="#222222")
    plot = plot_convergence(voronoi, style=style)
    lines = plot.ax.get_lines()
    assert to_hex(lines[0].get_color()) == "#111111"
    assert to_hex(lines[1].get_color()) == "#222222"


def test_convergence_classic_keeps_the_colors(voronoi):
    lines = plot_convergence(voronoi, style="classic").ax.get_lines()
    assert to_hex(lines[0].get_color()) == "#2c7bb6"
    assert to_hex(lines[1].get_color()) == "#e74c3c"


def test_voronoi_displacement_markers(voronoi):
    plot = voronoi_displacement(voronoi, show_positions=True, style=PlotStyle(accent_color="#222222"))
    colors = {to_hex(c) for col in plot.ax.collections for c in col.get_facecolor()}
    assert "#222222" in colors


def test_symbol_displacement_arrows(symbols):
    plot = plot_displacement(symbols, style=PlotStyle(accent_color="#222222"))
    assert to_hex(plot.arrows.get_facecolor()[0]) == "#222222"


def test_symbol_displacement_classic(symbols):
    plot = plot_displacement(symbols, style="classic")
    assert to_hex(plot.arrows.get_facecolor()[0]) == "#ff0000"


def test_preset_overlay_colors_differ_by_polarity():
    assert PlotStyle.default().accent_color != PlotStyle.dark().accent_color
    assert PlotStyle.classic().accent_color is None


class TestFlowAndLegend:
    @pytest.fixture(scope="class")
    def cartogram(self, gdf):
        import carto_flow.flow_cartogram as flow

        return flow.create_flow_cartogram(
            gdf, "population", options=flow.MorphOptions(grid=32, n_iter=3, save_internals=True, show_progress=False)
        )

    def test_velocity_color_follows_the_accent(self, cartogram):
        import carto_flow.flow_cartogram as flow
        from carto_flow.flow_cartogram.visualization import _with_arrow_color

        assert _with_arrow_color(flow.VelocityPlotOptions(), PlotStyle(accent_color="#222222")).color == "#222222"
        assert _with_arrow_color(flow.VelocityPlotOptions(), "classic").color == "black"
        assert _with_arrow_color(flow.VelocityPlotOptions(), "classic", "white").color == "white"
        assert _with_arrow_color(flow.VelocityPlotOptions(color="red"), "dark").color == "red"

    def test_velocity_plot_runs_in_every_style(self, cartogram):
        from carto_flow.flow_cartogram.visualization import plot_velocity_field

        for style in ("classic", "default", "dark"):
            plot_velocity_field(cartogram, style=style)

    def test_linewidth_legend_key_uses_the_axes_edge_color(self, symbols):
        plot = symbols.plot(linewidth="population", style="dark")
        handles = plot.linewidth_legend.legend_handles
        assert to_hex(handles[0].get_color()) == "#6b747c"
