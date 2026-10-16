"""Tests for the scales of the magnitudes of the velocity field and the Voronoi displacement, and the style palettes of
the field plots."""

import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, PowerNorm

import carto_flow.data as data
import carto_flow.flow_cartogram as flow
import carto_flow.voronoi_cartogram as vor
from carto_flow import Channel, Palettes, PlotStyle, set_style
from carto_flow.flow_cartogram.animation import animate_fields
from carto_flow.flow_cartogram.visualization import (
    DensityPlotOptions,
    VelocityPlotOptions,
    _prepare_velocity_colors,
    plot_convergence,
    plot_density_field,
    plot_velocity_field,
)


@pytest.fixture(autouse=True)
def reset():
    yield
    set_style(None)
    plt.close("all")


@pytest.fixture(scope="module")
def states():
    return data.load_us_census(population=True).reset_index(drop=True)


@pytest.fixture(scope="module")
def cartogram(states):
    options = flow.MorphOptions(grid=48, n_iter=10, save_internals=True, snapshot_every=5, show_progress=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return flow.create_flow_cartogram(states, "Population", options=options)


@pytest.fixture(scope="module")
def voronoi(states):
    return vor.create_voronoi_cartogram(
        states, weights="Population", backend=vor.RasterBackend(resolution=64), options=vor.VoronoiOptions(n_iter=5)
    )


def _colorbar_norm(fig):
    cbar_axes = [a for a in fig.axes if a.get_label() == "<colorbar>"]
    return cbar_axes[0]


class TestStyleNorm:
    def test_sqrt_is_a_color_scale(self):
        assert isinstance(PlotStyle().norm(np.array([1.0, 100.0]), scale="sqrt"), PowerNorm)
        assert PlotStyle(color_scale="sqrt").color_scale == "sqrt"

    def test_unknown_scale(self):
        with pytest.raises(ValueError):
            PlotStyle(color_scale="cubic")


class TestVelocityNorm:
    def _colors(self, norm, style=None):
        rng = np.random.default_rng(1)
        vx, vy = rng.lognormal(size=(6, 6)), rng.lognormal(size=(6, 6))
        return _prepare_velocity_colors(vx, vy, color_by="magnitude", norm=norm, style=style)

    def test_default_is_linear_without_a_color_scale(self):
        _, _, _, norm = self._colors(None, PlotStyle())
        assert type(norm) is Normalize

    def test_scales(self):
        assert isinstance(self._colors("log")[3], LogNorm)
        assert isinstance(self._colors("sqrt")[3], PowerNorm)
        assert isinstance(self._colors(Normalize(0, 5))[3], Normalize)

    def test_log_ignores_zero_magnitudes(self):
        vx = np.array([[0.0, 1.0], [10.0, 100.0]])
        colors, _, _, _ = _prepare_velocity_colors(vx, np.zeros_like(vx), color_by="magnitude", norm="log")
        assert np.isfinite(colors).all()

    def test_scale_changes_the_colors(self):
        assert not np.allclose(self._colors("linear")[0], self._colors("log")[0])

    def test_alpha_follows_the_scale(self):
        rng = np.random.default_rng(1)
        vx, vy = rng.lognormal(size=(6, 6)), np.zeros((6, 6))
        linear = _prepare_velocity_colors(vx, vy, alpha_by_magnitude=True, norm="linear")[0][:, 3]
        log = _prepare_velocity_colors(vx, vy, alpha_by_magnitude=True, norm="log")[0][:, 3]
        assert not np.allclose(linear, log)

    def test_sequential_palette_of_the_style(self):
        style = PlotStyle(palettes=Palettes(sequential="Reds"))
        colors = self._colors("linear", style)[0]
        explicit = _prepare_velocity_colors(
            *(lambda r: (r.lognormal(size=(6, 6)), r.lognormal(size=(6, 6))))(np.random.default_rng(1)),
            color_by="magnitude",
            cmap="Reds",
            norm="linear",
        )[0]
        assert np.allclose(colors, explicit)

    def test_plot_and_colorbar(self, cartogram):
        plot = plot_velocity_field(cartogram, velocity=VelocityPlotOptions(color_by="magnitude", norm="sqrt"))
        assert isinstance(plot.colorbar.norm, PowerNorm)

    def test_animation_runs(self, cartogram):
        anim = animate_fields(
            cartogram,
            velocity=VelocityPlotOptions(color_by="magnitude", norm="sqrt"),
            density=DensityPlotOptions(),
            fps=2,
            duration=1,
            style="default",
        )
        assert anim is not None


class TestFieldPalettes:
    def test_density_uses_the_sequential_palette(self, cartogram):
        style = PlotStyle(palettes=Palettes(sequential="Reds"))
        plot = plot_density_field(cartogram, style=style)
        assert plot.image.get_cmap().name.startswith("Reds")

    def test_density_difference_uses_the_diverging_palette(self, cartogram):
        style = PlotStyle(palettes=Palettes(diverging="PuOr"))
        plot = plot_density_field(cartogram, density=DensityPlotOptions(normalize="difference"), style=style)
        assert plot.image.get_cmap().name == "PuOr"

    def test_density_cmap_of_the_call_wins(self, cartogram):
        style = PlotStyle(palettes=Palettes(sequential="Reds"))
        plot = plot_density_field(cartogram, density=DensityPlotOptions(cmap="Blues"), style=style)
        assert plot.image.get_cmap().name == "Blues"

    def test_classic_keeps_the_defaults(self, cartogram):
        assert plot_density_field(cartogram, style="classic").image.get_cmap().name == "viridis"
        ratio = plot_density_field(cartogram, density=DensityPlotOptions(normalize="ratio"), style="classic")
        assert ratio.image.get_cmap().name == "RdBu_r"

    def test_workflow_convergence_series_colors(self, cartogram):
        style = PlotStyle(series_color="#112233", highlight_color="#445566")
        plot = plot_convergence(cartogram, style=style)
        colors = {matplotlib.colors.to_hex(line.get_color()) for line in plot.ax.get_lines()}
        assert "#112233" in colors


class TestVoronoiDisplacement:
    def test_classic_colorbar_is_linear_from_zero(self, voronoi):
        plot = voronoi.plot_displacement(style="classic")
        assert _colorbar_norm(plot.ax.figure).get_ylim()[0] == 0.0

    @pytest.mark.parametrize("norm", ["linear", "sqrt", "log", "auto", Normalize(0, 1e6)])
    def test_scales(self, voronoi, norm):
        plot = voronoi.plot_displacement(norm=norm)
        assert plot.ax is not None

    def test_style_palette(self, voronoi):
        style = PlotStyle(palettes=Palettes(sequential="Reds"))
        plot = voronoi.plot_displacement(style=style)
        assert plot.ax is not None

    def test_log_scale_colorbar(self, voronoi):
        plot = voronoi.plot_displacement(norm="log")
        bars = [a for a in plot.ax.figure.axes if a.get_label() == "<colorbar>"]
        assert bars and bars[0].get_yscale() == "log"


class TestCyclicAndColorbars:
    def test_cyclic_is_a_palette_role(self):
        assert PlotStyle(palettes=Palettes(cyclic="hsv")).colormap("cyclic").name == "hsv"
        assert PlotStyle().colormap("cyclic") is None
        assert PlotStyle.default().colormap("cyclic").name == "hsv"
        with pytest.raises(ValueError):
            PlotStyle().colormap("circular")

    def test_cyclic_merges(self):
        merged = PlotStyle(palettes=Palettes(cyclic="hsv")) | PlotStyle(palettes=Palettes(sequential="Reds"))
        assert merged.palettes.cyclic == "hsv"
        assert merged.palettes.sequential == "Reds"

    def test_direction_uses_the_cyclic_palette(self, cartogram):
        style = PlotStyle(palettes=Palettes(cyclic="hsv"))
        plot = plot_velocity_field(cartogram, velocity=VelocityPlotOptions(color_by="direction"), style=style)
        assert plot.colorbar.mappable.cmap.name == "hsv"

    def test_direction_default_and_classic(self, cartogram):
        classic = plot_velocity_field(cartogram, velocity=VelocityPlotOptions(color_by="direction"), style="classic")
        assert classic.colorbar.mappable.cmap.name == "twilight"

    def test_density_colorbar_is_drawn_by_the_style(self, cartogram):
        style = PlotStyle(colorbar_outline_color="#123456", colorbar_outline_width=2.0)
        plot = plot_density_field(cartogram, style=style)
        outline = plot.colorbar.outline
        assert matplotlib.colors.to_hex(outline.get_edgecolor()) == "#123456"

    def test_velocity_colorbar_is_drawn_by_the_style(self, cartogram):
        style = PlotStyle(colorbar_outline_color="#123456", colorbar_outline_width=2.0)
        plot = plot_velocity_field(cartogram, velocity=VelocityPlotOptions(color_by="magnitude"), style=style)
        assert matplotlib.colors.to_hex(plot.colorbar.outline.get_edgecolor()) == "#123456"

    def test_label_in_legend_kwargs_wins(self, cartogram):
        plot = plot_density_field(cartogram, density=DensityPlotOptions(legend_kwargs={"label": "rho"}))
        assert plot.colorbar.ax.get_ylabel() == "rho"

    def test_ratio_ticks_stay_ratios(self, cartogram):
        plot = plot_density_field(cartogram, density=DensityPlotOptions(normalize="ratio"), style="default")
        labels = [t.get_text() for t in plot.colorbar.ax.get_yticklabels()]
        assert any(label in ("1", "2", "0.5", "0.50") for label in labels)

    def test_animation_colorbars(self, cartogram):
        anim = animate_fields(
            cartogram,
            show_velocity=True,
            density=DensityPlotOptions(),
            velocity=VelocityPlotOptions(color_by="direction"),
            fps=2,
            duration=1,
            style="dark",
        )
        anim._init_draw()
        anim._func(1)
        bars = [a for a in anim._fig.axes if a.get_label() == "<colorbar>"]
        assert len(bars) == 2


class TestPreviewsAndTilings:
    @pytest.fixture(scope="class")
    def gdf(self):
        return data.load_us_census(population=True).reset_index(drop=True)

    def test_density_preview_palette_and_colorbar(self, gdf):
        from carto_flow.flow_cartogram import DensityBorderExtension
        from carto_flow.flow_cartogram.density import preview_modulator

        mod = DensityBorderExtension(extension_width=50_000, transition_width=30_000)
        style = PlotStyle(
            palettes=Palettes(diverging="PuOr"), colorbar_outline_color="#123456", colorbar_outline_width=2.0
        )
        plot = preview_modulator(mod, gdf, column="Population", grid_size=24, style=style)
        assert plot.image.get_cmap().name == "PuOr"
        assert matplotlib.colors.to_hex(plot.colorbar.outline.get_edgecolor()) == "#123456"
        classic = preview_modulator(mod, gdf, column="Population", grid_size=24, style="classic")
        assert classic.image.get_cmap().name == "RdBu_r"

    def test_density_preview_cmap_of_the_call_wins(self, gdf):
        from carto_flow.flow_cartogram import DensityBorderExtension
        from carto_flow.flow_cartogram.density import preview_modulator

        mod = DensityBorderExtension(extension_width=50_000, transition_width=30_000)
        style = PlotStyle(palettes=Palettes(diverging="PuOr"))
        plot = preview_modulator(mod, gdf, grid_size=24, cmap="coolwarm", style=style)
        assert plot.image.get_cmap().name == "coolwarm"

    @pytest.mark.parametrize(
        ("heatmap_type", "role"),
        [("magnitude", "sequential"), ("angle", "cyclic"), ("magnitude_diff", "diverging"), ("angle_diff", "cyclic")],
    )
    def test_anisotropy_preview_palettes(self, gdf, heatmap_type, role):
        from carto_flow.flow_cartogram import DirectionalTensor
        from carto_flow.flow_cartogram.anisotropy import preview_modulator

        style = PlotStyle(palettes=Palettes(sequential="Reds", diverging="PuOr", cyclic="hsv"))
        mod = DirectionalTensor.radial(center=(0, 0), Dpar=3.0)
        heatmap = "output" if heatmap_type in ("magnitude", "angle") else "diff"
        plot = preview_modulator(mod, gdf, grid_size=20, heatmap=heatmap, heatmap_type=heatmap_type, style=style)
        expected = {"sequential": "Reds", "diverging": "PuOr", "cyclic": "hsv"}[role]
        assert plot.heatmap_image.get_cmap().name == expected

    def test_tiling_plot_modes(self):
        from carto_flow.symbol_cartogram import SquareTiling

        tiling = SquareTiling().generate(n_tiles=16)
        style = PlotStyle(
            palettes=Palettes(sequential="Reds", cyclic="hsv"), channels={"stroke": Channel(value="#123456")}
        )
        plot = tiling.plot(color_by="x", style=style)
        assert matplotlib.colors.to_hex(plot.tiles.get_edgecolor()[0]) == "#123456"
        reds = tiling.plot(color_by="x", cmap="Reds", style="classic")
        assert np.allclose(plot.tiles.get_facecolor(), reds.tiles.get_facecolor())
        aspect = tiling.plot(color_by="aspect", style=style)
        hsv = tiling.plot(color_by="aspect", cmap="hsv", style="classic")
        assert np.allclose(aspect.tiles.get_facecolor(), hsv.tiles.get_facecolor())

    def test_tile_outline_and_prototile(self):
        from carto_flow.symbol_cartogram import SquareTiling

        style = PlotStyle(channels={"stroke": Channel(value="#123456")})
        plot = SquareTiling().plot_tile(style=style)
        assert matplotlib.colors.to_hex(plot.tile.get_edgecolor()) == "#123456"
        assert matplotlib.colors.to_hex(SquareTiling().plot_tile(style="classic").tile.get_edgecolor()) == "#1971c2"
