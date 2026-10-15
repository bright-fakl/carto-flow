"""Tests for the parameter vocabulary of the flow plot, animate and preview functions."""

import warnings

import geopandas as gpd
import matplotlib
import numpy as np
import pytest
from shapely.geometry import box

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import carto_flow.data as data
import carto_flow.flow_cartogram as flow
from carto_flow.flow_cartogram.animation import (
    animate_geometry_keyframes,
    animate_morph_history,
    animate_workflow,
)
from carto_flow.flow_cartogram.density import Smooth
from carto_flow.flow_cartogram.visualization import plot_comparison


@pytest.fixture(scope="module")
def states():
    return data.load_us_census(population=True).reset_index(drop=True)


@pytest.fixture(scope="module")
def cartogram(states):
    options = flow.MorphOptions(grid=64, n_iter=20, snapshot_every=10, show_progress=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return flow.create_flow_cartogram(states, "Population", options=options)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _fill_colors(result):
    return np.asarray(result.collections[0].get_facecolor())


class TestPlotFill:
    def test_data_literal_and_array(self, cartogram, states):
        by_column = cartogram.plot(facecolor="Population", legend=False)
        by_array = cartogram.plot(facecolor=states["Population"].to_numpy(), legend=False)
        np.testing.assert_allclose(_fill_colors(by_column), _fill_colors(by_array))
        literal = cartogram.plot(facecolor="#d4e6f1", legend=False)
        assert np.allclose(_fill_colors(literal), matplotlib.colors.to_rgba("#d4e6f1"))

    def test_builtin_quantities_match_the_internal_columns(self, cartogram):
        np.testing.assert_allclose(
            _fill_colors(cartogram.plot(facecolor="errors_pct", legend=False)),
            _fill_colors(cartogram.plot(facecolor="_morph_error_pct", legend=False)),
        )
        np.testing.assert_allclose(
            _fill_colors(cartogram.plot(facecolor="density", legend=False)),
            _fill_colors(cartogram.plot(facecolor="_morph_density", legend=False)),
        )

    def test_default_fill_is_the_errors(self, cartogram):
        np.testing.assert_allclose(
            _fill_colors(cartogram.plot(legend=False)),
            _fill_colors(cartogram.plot(facecolor="errors_pct", legend=False)),
        )

    def test_unknown_string_raises(self, cartogram):
        with pytest.raises(ValueError, match="neither a column"):
            cartogram.plot(facecolor="not-a-column")

    def test_old_names_are_aliases_with_a_warning(self, cartogram):
        with pytest.warns(DeprecationWarning, match="use 'facecolor'"):
            old = cartogram.plot(column="Population", legend=False)
        new = cartogram.plot(facecolor="Population", legend=False)
        np.testing.assert_allclose(_fill_colors(old), _fill_colors(new))

    def test_legend_kwargs_reach_the_colorbar(self, cartogram):
        result = cartogram.plot(facecolor="Population", legend=True, legend_kwargs={"label": "People"})
        labels = [a.get_ylabel() for a in result.ax.figure.axes]
        assert "People" in labels

    def test_comparison_uses_the_fill_in_both_panels(self, cartogram, states):
        result = plot_comparison(states, cartogram, facecolor="Population")
        assert len(result.left_collections) and len(result.right_collections)
        with pytest.warns(DeprecationWarning):
            plot_comparison(states, cartogram, column="Population")


class TestAnimations:
    def test_morph_history_fill_and_legend(self, cartogram):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            animate_morph_history(cartogram, duration=0.4, fps=5, facecolor="errors_pct", legend=True)

    def test_morph_history_literal_fill(self, cartogram):
        anim = animate_morph_history(cartogram, duration=0.4, fps=5, facecolor="#d4e6f1")
        assert anim is not None

    def test_morph_history_old_names(self, cartogram):
        with pytest.warns(DeprecationWarning) as record:
            animate_morph_history(
                cartogram, duration=0.4, fps=5, color_by="density", colorbar=True, colorbar_kwargs={"shrink": 0.5}
            )
        text = " ".join(str(r.message) for r in record)
        assert "'color_by' is deprecated; use 'facecolor'" in text
        assert "'colorbar' is deprecated; use 'legend'" in text
        assert "'colorbar_kwargs' is deprecated; use 'legend_kwargs'" in text

    def test_column_named_like_a_builtin_is_read_as_the_column(self, states):
        options = flow.MorphOptions(grid=64, n_iter=20, snapshot_every=10, show_progress=False)
        renamed = states.rename(columns={"Population": "density"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cart = flow.create_flow_cartogram(renamed, "density", options=options)
        with pytest.warns(UserWarning, match="column and a builtin"):
            animate_morph_history(cart, duration=0.4, fps=5, facecolor="density")

    def test_geometry_keyframes(self, states):
        frames = [states, states]
        animate_geometry_keyframes(frames, duration=0.4, fps=5, facecolor="Population", legend=True)
        animate_geometry_keyframes(frames, duration=0.4, fps=5, facecolor=[np.arange(len(states))] * 2)
        animate_geometry_keyframes(frames, duration=0.4, fps=5, facecolor="red")
        with pytest.warns(DeprecationWarning, match="use 'facecolor'"):
            animate_geometry_keyframes(frames, duration=0.4, fps=5, column="Population")
        with pytest.warns(DeprecationWarning, match="use 'facecolor'"):
            animate_geometry_keyframes(frames, duration=0.4, fps=5, color_values=[np.arange(len(states))] * 2)

    def test_workflow(self, states):
        workflow = flow.CartogramWorkflow(states, "Population")
        options = flow.MorphOptions(grid=64, n_iter=20, show_progress=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            workflow.morph(options=options)
        animate_workflow(workflow, duration=0.4, fps=5, facecolor="density", legend=False)
        animate_workflow(workflow, duration=0.4, fps=5, facecolor="Population")
        with pytest.warns(DeprecationWarning, match="'colorbar' is deprecated; use 'legend'"):
            animate_workflow(workflow, duration=0.4, fps=5, colorbar=False)


class TestOptionsObjects:
    def test_velocity_and_density_plot_options(self):
        options = flow.VelocityPlotOptions(legend=False, legend_kwargs={"shrink": 0.5})
        assert options.legend is False and options.legend_kwargs == {"shrink": 0.5}
        with pytest.warns(DeprecationWarning, match="colorbar"):
            old = flow.VelocityPlotOptions(colorbar=False)
        assert old.legend is False
        with pytest.warns(DeprecationWarning):
            assert old.colorbar is False
        with pytest.warns(DeprecationWarning, match="colorbar_kwargs"):
            flow.DensityPlotOptions(colorbar_kwargs={"shrink": 0.5})


class TestPreviewModulator:
    def test_new_and_old_names(self):
        gdf = gpd.GeoDataFrame({"v": [1.0]}, geometry=[box(0, 0, 1, 1)])
        modulator = Smooth(sigma=1.0)
        flow.density.preview_modulator(modulator, gdf, legend=True, legend_kwargs={"shrink": 0.5})
        with pytest.warns(DeprecationWarning, match="'show_colorbar' is deprecated; use 'legend'"):
            flow.density.preview_modulator(modulator, gdf, show_colorbar=False)
