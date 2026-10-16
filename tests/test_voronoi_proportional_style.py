"""Tests for the styles of the Voronoi and proportional plots."""

import geopandas as gpd
import matplotlib
import numpy as np
import pytest
from shapely.geometry import box

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, to_hex

from carto_flow import Palettes, PlotStyle, set_style
from carto_flow.proportional_cartogram import partition_geometries, plot_dot_density, plot_partitions
from carto_flow.style import resolve_style, styled
from carto_flow.voronoi_cartogram import RasterBackend, VoronoiOptions, create_voronoi_cartogram
from carto_flow.voronoi_cartogram.animation import animate_voronoi_history, save_animation
from carto_flow.voronoi_cartogram.visualization import plot_compactness, plot_comparison


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
            "name": list("abcdefghi"),
            "group": list("aaabbbccc"),
            "pop_a": rng.integers(100, 10000, size=9).astype(float),
            "pop_b": rng.integers(100, 10000, size=9).astype(float),
        },
        geometry=geoms,
    )


@pytest.fixture(scope="module")
def result(gdf):
    return create_voronoi_cartogram(
        gdf, backend=RasterBackend(resolution=50), options=VoronoiOptions(n_iter=5, record_history=True)
    )


@pytest.fixture(scope="module")
def partitioned(gdf):
    return partition_geometries(gdf, columns=["pop_a", "pop_b"], method="split", normalization="sum")


def _fill(plot_result):
    collections = getattr(plot_result, "partition_collections", None) or plot_result.collections
    return np.concatenate([np.asarray(c.get_facecolor()) for c in collections])


def _colorbar_axes(fig):
    return [a for a in fig.axes if a.get_label() == "<colorbar>"]


class TestStyledDecorator:
    def test_the_style_of_the_call_is_the_style_of_the_block(self):
        seen = []

        @styled
        def plot():
            seen.append(resolve_style())

        plot(style="dark")
        plot()
        assert seen[0] == PlotStyle.dark() and seen[1] == PlotStyle.default()


class TestStyleFields:
    def test_reference_and_annotation_fields_merge_like_the_others(self):
        merged = PlotStyle.default() | PlotStyle(reference_facecolor="#123456", annotation_size=14)
        assert merged.reference_facecolor == "#123456" and merged.annotation_size == 14
        assert merged.reference_edgecolor == PlotStyle.default().reference_edgecolor
        assert PlotStyle.classic().reference_facecolor is None and PlotStyle.classic().annotation_size is None


class TestVoronoi:
    def test_surface_edges_and_categorical_palette(self, result):
        palette = ["#ff0000", "#00ff00", "#0000ff"]
        style = PlotStyle.default().copy_with(palettes=Palettes(categorical=palette))
        plot = result.plot(style=style)
        assert to_hex(plot.ax.figure.get_facecolor()) == PlotStyle.default().rc["figure.facecolor"]
        assert {to_hex(c) for c in _fill(plot)} == set(palette)
        edge = to_hex(plot.collections[0].get_edgecolor()[0])
        assert edge == PlotStyle.default().channel_value("edge")

    def test_classic_keeps_the_former_defaults(self, result):
        plot = result.plot(style="classic", facecolor="population")
        assert to_hex(plot.ax.figure.get_facecolor()) == "#ffffff"
        assert to_hex(plot.collections[0].get_edgecolor()[0]) == "#000000"

    def test_errors_are_diverging_and_symmetric(self, result):
        plot = result.plot(facecolor="area_error_pct", legend=True, style="default")
        low, high = _colorbar_axes(plot.ax.figure)[0].get_ylim()
        assert low == pytest.approx(-high)

    def test_norm_center_and_the_colorbar_of_the_style(self, result):
        plot = result.plot(facecolor="population", legend=True, norm="log", style="default")
        assert isinstance(plot.collections[0].norm, LogNorm)
        assert len(_colorbar_axes(plot.ax.figure)) == 1
        plot = result.plot(facecolor="population", legend=True, center=5000, style="default")
        low, high = _colorbar_axes(plot.ax.figure)[0].get_ylim()
        assert (low + high) / 2 == pytest.approx(5000)
        plot = result.plot(facecolor="population", legend=True, legend_label="People", style="default")
        assert _colorbar_axes(plot.ax.figure)[0].get_ylabel() == "People"

    def test_vmin_and_vmax_fix_a_linear_normalization(self, result):
        plot = result.plot(facecolor="population", vmin=0, vmax=20000, style="default")
        norm = plot.collections[0].norm
        assert type(norm) is Normalize and (norm.vmin, norm.vmax) == (0, 20000)

    def test_categorical_column_and_flag(self, result):
        plot = result.plot(facecolor="name", style="default")
        assert plot.collections
        assert result.plot(facecolor="population", categorical=True, style="classic").collections

    def test_palette_of_the_call_beats_the_style(self, result):
        green = PlotStyle(palettes=Palettes(sequential="Greens"))
        by_style = result.plot(facecolor="population", style=green)
        by_call = result.plot(facecolor="population", cmap="Greens", style="classic")
        np.testing.assert_allclose(_fill(by_style), _fill(by_call))

    def test_labels_take_the_annotation_size_and_text_color_of_the_style(self, result):
        style = PlotStyle.dark().copy_with(annotation_size=13)
        plot = result.plot(labels=True, style=style)
        text = plot.ax.texts[0]
        assert text.get_fontsize() == 13
        assert to_hex(text.get_color()) == PlotStyle.dark().rc["text.color"]
        assert (
            result.plot(labels=True, label_fontsize=5, label_color="red", style=style).ax.texts[0].get_fontsize() == 5
        )

    def test_comparison_and_compactness(self, gdf, result):
        comparison = plot_comparison(gdf, result, facecolor="population", style="dark")
        assert to_hex(comparison.ax[0].figure.get_facecolor()) == PlotStyle.dark().rc["figure.facecolor"]
        compact = plot_compactness(result, "group", style="default", norm="linear")
        assert to_hex(compact.ax.figure.get_facecolor()) == PlotStyle.default().rc["figure.facecolor"]

    def test_displacement_and_the_animation_keep_the_style(self, result, tmp_path):
        displacement = result.plot_displacement(style="dark")
        assert to_hex(displacement.ax.figure.get_facecolor()) == PlotStyle.dark().rc["figure.facecolor"]
        anim = animate_voronoi_history(result, style="dark")
        anim._init_draw()
        anim._func(1)
        assert to_hex(anim._fig.axes[0].figure.get_facecolor()) == PlotStyle.dark().rc["figure.facecolor"]
        save_animation(anim, tmp_path / "out.gif", fps=2, dpi=30)
        from PIL import Image

        corner = Image.open(tmp_path / "out.gif").convert("RGB").getpixel((1, 1))
        assert corner == tuple(int(PlotStyle.dark().rc["figure.facecolor"][i : i + 2], 16) for i in (1, 3, 5))

    def test_rcparams_are_restored(self, result):
        before = dict(matplotlib.rcParams)
        result.plot(style="dark")
        assert all(matplotlib.rcParams[k] == before[k] for k in PlotStyle.dark().rc)


class TestProportional:
    def test_partitions_take_the_palettes_and_edges_of_the_style(self, partitioned):
        palette = ["#ff0000", "#0000ff"]
        style = PlotStyle.default().copy_with(palettes=Palettes(categorical=palette))
        plot = plot_partitions(partitioned, facecolor="category", style=style)
        colors = {to_hex(c) for c in _fill(plot)[: len(plot.partition_collections[0].get_facecolor())]}
        assert colors <= set(palette)
        assert to_hex(plot.ax.figure.get_facecolor()) == PlotStyle.default().rc["figure.facecolor"]
        assert to_hex(plot.partition_collections[0].get_edgecolor()[0]) == PlotStyle.default().channel_value("edge")

    def test_classic_keeps_the_former_defaults(self, partitioned):
        plot = plot_partitions(partitioned, style="classic")
        assert to_hex(plot.partition_collections[0].get_edgecolor()[0]) == "#ffffff"
        assert to_hex(plot.ax.figure.get_facecolor()) == "#ffffff"

    def test_continuous_data_uses_the_sequential_palette_and_norm(self, partitioned):
        green = PlotStyle(palettes=Palettes(sequential="Greens"))
        by_style = plot_partitions(partitioned, style=green)
        by_call = plot_partitions(partitioned, cmap="Greens", style="classic")
        np.testing.assert_allclose(_fill(by_style), _fill(by_call))
        wide = plot_partitions(partitioned, norm=Normalize(0, 1e9), style="classic")
        assert not np.allclose(_fill(wide), _fill(by_call))
        with_label = plot_partitions(partitioned, legend_label="Area", style="default")
        assert _colorbar_axes(with_label.ax.figure)[0].get_ylabel() == "Area"

    def test_cmap_dict_beats_the_style_and_a_constant_color_has_no_guide(self, partitioned):
        plot = plot_partitions(partitioned, facecolor="category", cmap={"pop_a": "#ff0000"}, style="default")
        assert "#ff0000" in {to_hex(c) for c in _fill(plot)}
        constant = plot_partitions(partitioned, facecolor="#d4e6f1", style="default")
        assert constant.colorbar is None and constant.legend is None

    def test_complement_takes_the_reference_fill_of_the_style(self, partitioned):
        plot = plot_partitions(partitioned, style="default")
        assert plot.complement_collections
        assert to_hex(plot.complement_collections[0].get_facecolor()[0]) == PlotStyle.default().reference_facecolor
        classic = plot_partitions(partitioned, style="classic")
        assert to_hex(classic.complement_collections[0].get_facecolor()[0]) == "#e0e0e0"

    def test_dot_density(self, gdf):
        kwargs = {"columns": ["pop_a", "pop_b"], "n_dots": 10, "normalization": "sum", "seed": 0}
        palette = ["#ff0000", "#0000ff"]
        style = PlotStyle.default().copy_with(palettes=Palettes(categorical=palette))
        plot = plot_dot_density(gdf, style=style, **kwargs)
        assert {to_hex(c.get_facecolor()[0]) for c in plot.dot_collections} == set(palette)
        assert to_hex(plot.base_collection.get_facecolor()[0]) == PlotStyle.default().reference_facecolor
        classic = plot_dot_density(gdf, style="classic", **kwargs)
        assert to_hex(classic.base_collection.get_facecolor()[0]) == "#e0e0e0"
        explicit = plot_dot_density(
            gdf, style="default", reference_facecolor="#123456", cmap={"pop_a": "#ff0000"}, **kwargs
        )
        assert to_hex(explicit.base_collection.get_facecolor()[0]) == "#123456"
        assert "#ff0000" in {to_hex(c.get_facecolor()[0]) for c in explicit.dot_collections}


class TestLegendKeywordAlias:
    def test_old_geopandas_name_is_an_alias(self, result):
        with pytest.warns(DeprecationWarning, match="'legend_kwds' is deprecated; use 'legend_kwargs'"):
            plot = result.plot(facecolor="population", legend=True, legend_kwds={"label": "People"}, style="default")
        assert _colorbar_axes(plot.ax.figure)[0].get_ylabel() == "People"
