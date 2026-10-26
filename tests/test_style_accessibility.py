"""Accessibility of the built-in palettes: color vision deficiency and contrast of the overlay colors.

Color vision deficiency is simulated with the matrices of Machado, Oliveira and Fernandes (2009) at full severity, in
linear RGB; distances are CIE76 color differences. The thresholds are the properties the palettes have. The case that
fails is marked ``xfail(strict=True)`` so that a change of the palette that fixes one shows up as a failed test.
"""

import itertools

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib.colors import to_rgb

from carto_flow import PlotStyle

MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
}
VISIONS = (None, *MACHADO)
OVERLAYS = ("series_color", "accent_color", "positive_color", "highlight_color")
STYLES = {"default": PlotStyle.default, "dark": PlotStyle.dark}


def _linear(rgb):
    rgb = np.asarray(rgb, float)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _srgb(lin):
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)


def simulate(rgb, vision):
    return np.asarray(rgb, float) if vision is None else _srgb(np.asarray(MACHADO[vision]) @ _linear(rgb))


def lab(rgb):
    xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ _linear(rgb)
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def distance(a, b, vision):
    return float(np.linalg.norm(lab(simulate(a, vision)) - lab(simulate(b, vision))))


def contrast(a, b):
    def luminance(rgb):
        return float(np.array([0.2126, 0.7152, 0.0722]) @ _linear(rgb))

    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def categorical_colors(style, n=10):
    categorical = style.palettes.categorical
    colors = mpl.colormaps[categorical].colors if isinstance(categorical, str) else categorical
    return [to_rgb(c) for c in colors][:n]


def min_pair_distance(colors, vision):
    return min(distance(a, b, vision) for a, b in itertools.combinations(colors, 2))


@pytest.mark.parametrize("name", STYLES)
class TestBuiltinPalettes:
    def test_categories_differ_with_normal_vision(self, name):
        assert min_pair_distance(categorical_colors(STYLES[name]()), None) >= 20

    def test_categories_are_distinct_under_tritanopia(self, name):
        assert min_pair_distance(categorical_colors(STYLES[name]()), "tritan") >= 10

    @pytest.mark.xfail(strict=True, reason="ten qualitative colors cannot all differ for red-green color blindness")
    @pytest.mark.parametrize("vision", ["protan", "deutan"])
    def test_ten_categories_are_distinct_under_red_green_deficiency(self, name, vision):
        assert min_pair_distance(categorical_colors(STYLES[name]()), vision) >= 10

    @pytest.mark.parametrize("vision", ["protan", "deutan"])
    def test_five_categories_are_distinct_under_red_green_deficiency(self, name, vision):
        # a sample of five of the ten colors that is distinct exists: the first five are not guaranteed, the best five are
        colors = categorical_colors(STYLES[name]())
        best = max(min_pair_distance(list(five), vision) for five in itertools.combinations(colors, 5))
        assert best >= 10

    @pytest.mark.parametrize("vision", VISIONS)
    def test_the_ends_of_the_diverging_palette_differ(self, name, vision):
        diverging = mpl.colormaps[STYLES[name]().palettes.diverging]
        assert distance(diverging(0.0)[:3], diverging(1.0)[:3], vision) >= 50

    @pytest.mark.parametrize("overlay", OVERLAYS)
    def test_overlay_colors_contrast_with_the_surface(self, name, overlay):
        style = STYLES[name]()
        assert contrast(to_rgb(getattr(style, overlay)), to_rgb(style.rc["axes.facecolor"])) >= 2.5

    @pytest.mark.parametrize("overlay", OVERLAYS)
    def test_overlay_colors_contrast_with_the_reference_layer(self, name, overlay):
        style = STYLES[name]()
        assert contrast(to_rgb(getattr(style, overlay)), to_rgb(style.reference_facecolor)) >= 2.0


@pytest.mark.parametrize("name", STYLES)
@pytest.mark.parametrize("vision", ["protan", "deutan", "tritan"])
def test_overlay_colors_differ_under_color_vision_deficiency(name, vision):
    style = STYLES[name]()
    assert min_pair_distance([to_rgb(getattr(style, o)) for o in OVERLAYS], vision) >= 12
