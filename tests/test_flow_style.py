"""Tests for the styles of the flow plots and animations."""

import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import carto_flow.data as data
import carto_flow.flow_cartogram as flow
from carto_flow import Palettes, PlotStyle, style_context
from carto_flow.flow_cartogram.animation import (
    animate_geometry_keyframes,
    animate_morph_history,
    animate_workflow,
    linear_over,
    morph_timeline,
    save_animation,
)
from carto_flow.flow_cartogram.visualization import plot_comparison, plot_convergence


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


def _fill(result):
    return np.asarray(result.collections[0].get_facecolor())


def _colorbar_axes(fig):
    return [a for a in fig.axes if a.get_label() == "<colorbar>"]


class TestClassic:
    def test_reproduces_the_unstyled_plot(self, cartogram):
        gdf = cartogram.to_geodataframe(include_errors=True, include_density=True)
        _, ax = plt.subplots()
        gdf.plot(ax=ax, column="Population", cmap="RdYlGn_r")
        expected = np.asarray(ax.collections[0].get_facecolor())
        result = cartogram.plot(facecolor="Population", legend=False, style="classic")
        np.testing.assert_allclose(_fill(result), expected)

    def test_classic_leaves_the_colorbar_to_geopandas(self, cartogram):
        result = cartogram.plot(facecolor="Population", style="classic")
        assert len(_colorbar_axes(result.ax.figure)) == 1
        assert matplotlib.colors.to_hex(result.ax.figure.get_facecolor()) == "#ffffff"


class TestDefaultAndDark:
    def test_surface_and_boundary(self, cartogram):
        for name in ("default", "dark"):
            style = PlotStyle.default() if name == "default" else PlotStyle.dark()
            result = cartogram.plot(facecolor="Population", style=name)
            assert matplotlib.colors.to_hex(result.ax.figure.get_facecolor()) == style.rc["figure.facecolor"]
            assert matplotlib.colors.to_hex(result.ax.get_facecolor()) == style.rc["axes.facecolor"]
            edge = matplotlib.colors.to_hex(result.collections[0].get_edgecolor()[0])
            assert edge == style.channel_value("edge")

    def test_default_is_the_style_in_effect(self, cartogram):
        a = cartogram.plot(facecolor="Population")
        b = cartogram.plot(facecolor="Population", style="default")
        np.testing.assert_allclose(_fill(a), _fill(b))
        with style_context("dark"):
            c = cartogram.plot(facecolor="Population")
        assert matplotlib.colors.to_hex(c.ax.figure.get_facecolor()) == PlotStyle.dark().rc["figure.facecolor"]

    def test_log_colorbar_with_real_number_ticks(self, cartogram):
        result = cartogram.plot(facecolor="Population", style="default")
        fig = result.ax.figure
        fig.canvas.draw()
        labels = [t.get_text() for t in _colorbar_axes(fig)[0].get_yticklabels() if t.get_text()]
        assert labels and all(label[-1] in "kMB" or label.isdigit() for label in labels)
        assert _colorbar_axes(fig)[0].get_ylabel() == "Population"

    def test_errors_are_diverging_and_symmetric(self, cartogram):
        result = cartogram.plot(style="default")
        low, high = _colorbar_axes(result.ax.figure)[0].get_ylim()
        assert low == pytest.approx(-high)

    def test_rcparams_are_restored(self, cartogram):
        before = dict(matplotlib.rcParams)
        cartogram.plot(facecolor="Population", style="dark")
        assert all(matplotlib.rcParams[k] == before[k] for k in PlotStyle.dark().rc)
        with pytest.raises(ValueError, match="neither a column"):
            cartogram.plot(facecolor="nope", style="dark")
        assert all(matplotlib.rcParams[k] == before[k] for k in PlotStyle.dark().rc)


class TestPrecedence:
    def test_call_beats_style_beats_default(self, cartogram):
        green = PlotStyle(palettes=Palettes(sequential="Greens"))
        by_style = cartogram.plot(facecolor="Population", legend=False, style=green)
        by_call = cartogram.plot(facecolor="Population", legend=False, cmap="Greens", style="classic")
        np.testing.assert_allclose(_fill(by_style), _fill(by_call))
        call_over_style = cartogram.plot(facecolor="Population", legend=False, cmap="Reds", style=green)
        reds = cartogram.plot(facecolor="Population", legend=False, cmap="Reds", style="classic")
        np.testing.assert_allclose(_fill(call_over_style), _fill(reds))

    def test_norm_of_the_call(self, cartogram):
        log = cartogram.plot(facecolor="Population", legend=False, norm="log", style="classic")
        linear = cartogram.plot(facecolor="Population", legend=False, style="classic")
        assert not np.allclose(_fill(log), _fill(linear))
        instance = cartogram.plot(
            facecolor="Population", legend=False, norm=matplotlib.colors.Normalize(0, 1e8), style="classic"
        )
        assert not np.allclose(_fill(instance), _fill(linear))

    def test_legend_label_and_kwargs(self, cartogram):
        result = cartogram.plot(facecolor="Population", legend_label="People", style="default")
        assert _colorbar_axes(result.ax.figure)[0].get_ylabel() == "People"
        result = cartogram.plot(facecolor="Population", legend_kwargs={"label": "Residents"}, style="default")
        assert _colorbar_axes(result.ax.figure)[0].get_ylabel() == "Residents"


class TestRoles:
    def test_categorical_column_uses_the_style_palette(self, cartogram):
        style = PlotStyle(palettes=Palettes(categorical=["#ff0000", "#00ff00", "#0000ff", "#ffff00"]))
        result = cartogram.plot(facecolor="Region", style=style, legend=False)
        colors = {matplotlib.colors.to_hex(c) for c in _fill(result)}
        assert colors == {"#ff0000", "#00ff00", "#0000ff", "#ffff00"}

    def test_center_makes_numeric_data_diverging(self, cartogram):
        result = cartogram.plot(facecolor="Population", center=1e7, style="default")
        low, high = _colorbar_axes(result.ax.figure)[0].get_ylim()
        assert (low + high) / 2 == pytest.approx(1e7)

    def test_categorical_flag(self, cartogram):
        result = cartogram.plot(facecolor="STATE", categorical=True, style="classic")
        assert result.collections


class TestComparison:
    def test_panels_share_the_normalization(self, states, cartogram):
        result = plot_comparison(states, cartogram, facecolor="Population", style="default")
        limits = [a.get_ylim() for a in _colorbar_axes(result.ax_left.figure)]
        assert len(limits) == 2 and limits[0] == pytest.approx(limits[1])

    def test_style_surface(self, states, cartogram):
        result = plot_comparison(states, cartogram, style="dark")
        assert (
            matplotlib.colors.to_hex(result.ax_left.figure.get_facecolor()) == PlotStyle.dark().rc["figure.facecolor"]
        )


class TestOtherPlots:
    def test_convergence_plot_takes_the_style(self, cartogram):
        result = plot_convergence(cartogram, style="dark")
        assert matplotlib.colors.to_hex(result.ax.get_facecolor()) == PlotStyle.dark().rc["axes.facecolor"]


class TestAnimations:
    def test_frames_keep_the_style_after_the_axes_is_cleared(self, cartogram):
        anim = animate_morph_history(cartogram, duration=1.0, fps=5, facecolor="errors_pct", style="dark")
        anim._init_draw()
        for frame in (0, 2, 4):
            anim._func(frame)
            assert matplotlib.colors.to_hex(anim._fig.axes[0].get_facecolor()) == PlotStyle.dark().rc["axes.facecolor"]
        assert matplotlib.colors.to_hex(anim._fig.get_facecolor()) == PlotStyle.dark().rc["figure.facecolor"]

    def test_norm_and_center_and_cmap(self, states, cartogram):
        frames = [states, cartogram.to_geodataframe()]
        animate_geometry_keyframes(frames, duration=1.0, fps=5, facecolor="Population", legend=True, norm="log")
        animate_geometry_keyframes(frames, duration=1.0, fps=5, facecolor="Population", legend=True, center=1e7)
        animate_geometry_keyframes(frames, duration=1.0, fps=5, facecolor="Population", cmap="Reds", style="classic")
        animate_geometry_keyframes(
            frames, duration=1.0, fps=5, facecolor="Population", norm=matplotlib.colors.LogNorm(1e5, 4e7)
        )
        with pytest.raises(ValueError, match="positive"):
            animate_geometry_keyframes(
                frames, duration=1.0, fps=5, facecolor=np.zeros(len(states)), norm="log", legend=True
            )

    def test_default_colorbar_is_drawn_by_the_style(self, states, cartogram):
        frames = [states, cartogram.to_geodataframe()]
        anim = animate_geometry_keyframes(frames, duration=1.0, fps=5, facecolor="Population", legend=True)
        anim._init_draw()
        anim._func(2)
        anim._fig.canvas.draw()
        axes = _colorbar_axes(anim._fig)
        assert len(axes) == 1
        assert any(t.get_text().endswith("M") for t in axes[0].get_yticklabels())

    def test_workflow_and_save(self, states, tmp_path):
        workflow = flow.CartogramWorkflow(states, "Population")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            workflow.morph(options=flow.MorphOptions(grid=64, n_iter=20, show_progress=False))
        anim = animate_workflow(workflow, duration=1.0, fps=5, style="dark")
        save_animation(anim, tmp_path / "out.gif", fps=5, dpi=40)
        from PIL import Image

        frame = Image.open(tmp_path / "out.gif").convert("RGB")
        corner = frame.getpixel((1, 1))
        assert corner == tuple(int(PlotStyle.dark().rc["figure.facecolor"][i : i + 2], 16) for i in (1, 3, 5))


class TestMorphTimeline:
    def test_hold_morph_hold_return(self):
        frames = list("abcde")
        keyframes, times = morph_timeline(frames, hold_start=1.0, morph=4.0, hold_end=2.0, back=3.0)
        assert keyframes == list("a" + "abcde" + "e" + "dcba")
        assert times[0] == 0 and times[1] == 1.0  # the hold on the first state
        assert times[5] == pytest.approx(5.0)  # the end of the morph
        assert times[6] == pytest.approx(7.0)  # the end of the hold on the last state
        assert times[-1] == pytest.approx(10.0) and np.all(np.diff(times) >= 0)

    def test_cosine_easing_is_slow_at_both_ends(self):
        _, cosine = morph_timeline(list(range(11)), hold_start=0, morph=10.0, back=0)
        _, linear = morph_timeline(list(range(11)), hold_start=0, morph=10.0, back=0, easing="linear")
        assert cosine[2] - cosine[1] > linear[2] - linear[1]  # the first step takes longer than at constant speed
        assert np.diff(linear[1:12]).std() < 1e-9

    def test_no_return_and_errors(self):
        keyframes, times = morph_timeline(list("abc"), back=0)
        assert keyframes[-1] == "c" and times[-1] == pytest.approx(1.5 + 7.0 + 2.5)
        with pytest.raises(ValueError, match="at least 2"):
            morph_timeline(["a"])
        with pytest.raises(ValueError, match="easing"):
            morph_timeline(["a", "b"], easing="bounce")  # type: ignore[arg-type]

    def test_from_a_cartogram_with_animate_geometry_keyframes(self, cartogram):
        keyframes, times = morph_timeline(cartogram, hold_start=0.5, morph=1.0, hold_end=0.5, back=1.0)
        anim = animate_geometry_keyframes(
            keyframes,
            keyframe_times=times,
            position_mapper=linear_over("time"),
            duration=float(times[-1]),
            fps=4,
            facecolor="Population",
        )
        anim._init_draw()
        anim._func(0)


class TestLimitsOfTheCall:
    def test_vmin_and_vmax_fix_a_linear_normalization(self, cartogram):
        result = cartogram.plot(facecolor="Population", vmin=0, vmax=3e7, style="default")
        norm = result.collections[0].norm
        assert type(norm) is matplotlib.colors.Normalize and (norm.vmin, norm.vmax) == (0, 3e7)
        low, high = _colorbar_axes(result.ax.figure)[0].get_ylim()
        assert (low, high) == (0, 3e7)

    def test_only_one_limit(self, cartogram):
        result = cartogram.plot(facecolor="Population", vmin=1e6, style="default")
        norm = result.collections[0].norm
        assert norm.vmin == 1e6 and type(norm) is matplotlib.colors.Normalize

    def test_a_named_scale_uses_the_limits(self, cartogram):
        result = cartogram.plot(facecolor="Population", vmin=1e5, vmax=5e7, norm="log", style="default")
        norm = result.collections[0].norm
        assert isinstance(norm, matplotlib.colors.LogNorm) and (norm.vmin, norm.vmax) == (1e5, 5e7)

    def test_a_norm_instance_and_limits_raise(self, cartogram):
        with pytest.raises(ValueError, match="norm or vmin"):
            cartogram.plot(facecolor="Population", vmin=0, norm=matplotlib.colors.Normalize(), style="default")

    def test_classic_passes_the_limits_to_geopandas(self, cartogram):
        result = cartogram.plot(facecolor="Population", vmin=0, vmax=3e7, style="classic")
        norm = result.collections[0].norm
        assert (norm.vmin, norm.vmax) == (0, 3e7)
        assert len(_colorbar_axes(result.ax.figure)) == 1


class TestStyleSequences:
    def test_a_call_takes_a_sequence_of_styles(self, cartogram):
        talk = PlotStyle(rc={"xtick.labelsize": 20})
        result = cartogram.plot(facecolor="Population", style=("dark", talk))
        assert matplotlib.colors.to_hex(result.ax.figure.get_facecolor()) == PlotStyle.dark().rc["figure.facecolor"]
        result.ax.figure.canvas.draw()
        assert result.ax.xaxis.get_ticklabels()[0].get_fontsize() == 20
