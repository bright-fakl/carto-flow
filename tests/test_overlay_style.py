"""Tests for the overlay colors of a style."""

import geopandas as gpd
import matplotlib
import numpy as np
import pytest
from shapely.geometry import box

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

from carto_flow import Channel, PlotStyle, set_style
from carto_flow.symbol_cartogram import PackingLayout, create_symbol_cartogram, plot_displacement
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
    return create_symbol_cartogram(gdf, "population", layout=PackingLayout(max_iterations=20), show_progress=False)


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


class TestDataDrivenEdges:
    def test_width_edge_defaults_to_the_stroke_color(self, symbols):
        for style, expected in (("classic", "#000000"), ("default", "#3b4650"), ("dark", "#c9d1d9")):
            plot = symbols.plot(linewidth="population", style=style)
            assert to_hex(plot.collections[0].get_edgecolor()[0]) == expected

    def test_hatch_edge_defaults_to_the_stroke_color(self, symbols):
        plot = symbols.plot(hatch="population", style=PlotStyle(channels={"stroke": Channel(value="#123456")}))
        assert to_hex(plot.collections[0].get_edgecolor()[0]) == "#123456"

    def test_hatch_without_edge_does_not_warn_but_explicit_none_does(self, symbols):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            symbols.plot(hatch="/", style="classic")
        with pytest.warns(UserWarning, match="hatch"):
            symbols.plot(hatch="/", edgecolor="none", style="classic")

    def test_scalar_width_keeps_the_style_edge(self, symbols):
        plot = symbols.plot(linewidth=1.0, style="default")
        assert to_hex(plot.collections[0].get_edgecolor()[0]) == "#dde4ea"

    def test_explicit_edge_wins(self, symbols):
        plot = symbols.plot(linewidth="population", edgecolor="#123456", style="dark")
        assert to_hex(plot.collections[0].get_edgecolor()[0]) == "#123456"

    def test_stroke_is_a_channel(self):
        assert PlotStyle.classic().channel_value("stroke") is None
        assert PlotStyle.default().channel_value("stroke") != PlotStyle.dark().channel_value("stroke")


class TestScales:
    @pytest.mark.parametrize("scale", ["linear", "sqrt", "log", "auto", "rank"])
    def test_widths_span_the_range(self, symbols, scale):
        plot = symbols.plot(linewidth="population", linewidth_norm=scale, linewidth_range=(1.0, 5.0), style="classic")
        widths = plot.collections[0].get_linewidth()
        assert min(widths) == pytest.approx(1.0)
        assert max(widths) == pytest.approx(5.0)

    def test_scales_order_the_middle_values(self):
        values = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
        style = PlotStyle()
        t = {name: style.unit_scale(values, name)(values) for name in ("linear", "sqrt", "log", "rank")}
        assert t["log"][2] == pytest.approx(0.5)
        assert t["rank"][2] == pytest.approx(0.5)
        assert t["linear"][2] < t["sqrt"][2] < t["log"][2]

    def test_inverse(self):
        values = np.array([1.0, 3.0, 10.0, 40.0, 200.0])
        for name in ("linear", "sqrt", "log", "rank"):
            scale = PlotStyle().unit_scale(values, name)
            assert scale.inverse(scale(values)) == pytest.approx(values)

    def test_ties_share_a_rank(self):
        values = np.array([1.0, 1.0, 1.0, 9.0])
        t = PlotStyle().unit_scale(values, "rank")(values)
        assert t[0] == t[1] == t[2] < t[3]

    def test_auto_follows_the_span(self):
        narrow, wide = np.array([1.0, 5.0]), np.array([1.0, 500.0])
        assert PlotStyle().unit_scale(narrow, "auto").name == "linear"
        assert PlotStyle().unit_scale(wide, "auto").name == "log"
        assert PlotStyle(log_span=3).unit_scale(wide, "auto").name == "linear"

    def test_log_needs_positive_values(self):
        with pytest.raises(ValueError, match="positive"):
            PlotStyle().unit_scale(np.array([0.0, 1.0]), "log")

    def test_normalize_instance(self):
        from matplotlib.colors import Normalize

        scale = PlotStyle().unit_scale(np.array([0.0, 10.0]), Normalize(0, 20))
        assert scale(np.array([10.0]))[0] == pytest.approx(0.5)

    def test_channel_scale_of_the_style_and_the_call_wins(self, symbols):
        style = PlotStyle(channels={"linewidth": Channel(scale="rank", range=(1.0, 5.0))})
        by_style = symbols.plot(linewidth="population", style=style).collections[0].get_linewidth()
        by_call = symbols.plot(linewidth="population", linewidth_norm="linear", style=style).collections[0]
        assert not np.allclose(by_style, by_call.get_linewidth())

    def test_unknown_scale(self):
        with pytest.raises(ValueError):
            Channel(scale="cubic")

    def test_channel_scale_merges(self):
        merged = PlotStyle(channels={"linewidth": Channel(scale="rank")}) | PlotStyle(
            channels={"linewidth": Channel(range=(1.0, 2.0))}
        )
        assert merged.channels["linewidth"].scale == "rank"
        assert merged.channels["linewidth"].range == (1.0, 2.0)

    def test_alpha_and_label_scales(self, symbols):
        plot = symbols.plot(
            alpha="population",
            alpha_norm="rank",
            labels="population",
            label_fontsize="population",
            label_fontsize_norm="sqrt",
            legend=False,
        )
        assert plot.labels

    def test_legend_values_follow_the_scale(self, symbols):
        plot = symbols.plot(linewidth="population", linewidth_norm="log", style="classic")
        labels = [t.get_text() for t in plot.linewidth_legend.get_texts()]
        values = [float(v) for v in labels]
        assert values == sorted(values)
        ratios = [values[i + 1] / values[i] for i in range(len(values) - 1)]
        assert ratios == pytest.approx([ratios[0]] * len(ratios), rel=0.02)


def test_adjacency_nodes_use_the_stroke_color(symbols):
    from carto_flow.symbol_cartogram import plot_adjacency

    for style, expected in (("classic", "#000000"), ("default", "#3b4650"), ("dark", "#c9d1d9")):
        plot = plot_adjacency(symbols, style=style, link_legend=False)
        assert to_hex(plot.nodes.get_facecolor()[0]) == expected
