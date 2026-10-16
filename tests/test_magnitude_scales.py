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
from carto_flow import Palettes, PlotStyle, set_style
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
