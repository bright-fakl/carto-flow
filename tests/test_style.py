"""Tests for PlotStyle, the style registry and the selection of a style."""

import inspect
import threading

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

import carto_flow
from carto_flow.style import (
    Channel,
    Palettes,
    PlotStyle,
    get_style,
    register_style,
    resolve_style,
    set_style,
    style_context,
    styled,
    unregister_style,
)


@pytest.fixture(autouse=True)
def reset_style():
    yield
    set_style(None)
    plt.close("all")


class TestPlotStyle:
    def test_classic_sets_nothing(self):
        style = PlotStyle()
        assert style == PlotStyle.classic()
        assert style.rc == {} and style.colormap("sequential") is None
        assert style.categorical_colors(["a", "b"]) is None
        assert style.channel_value("edge") is None and style.channel_value("edge", "k") == "k"
        assert not style.draws_guide

    def test_presets(self):
        light, dark = PlotStyle.default(), PlotStyle.dark()
        assert light.rc["figure.facecolor"] != dark.rc["figure.facecolor"]
        assert light.draws_guide and dark.draws_guide
        assert light.channel_value("edge") and dark.channel_value("linewidth")

    def test_validation(self):
        with pytest.raises(ValueError, match="color_scale"):
            PlotStyle(color_scale="cubic")
        with pytest.raises(ValueError, match="tick_format"):
            PlotStyle(tick_format="eng")
        with pytest.raises(ValueError, match="colorbar_orientation"):
            PlotStyle(colorbar_orientation="diagonal")
        with pytest.raises(ValueError, match="log_span"):
            PlotStyle(log_span=0)
        with pytest.raises(ValueError, match="unknown channels"):
            PlotStyle(channels={"fil": Channel()})
        with pytest.raises(ValueError, match="not a matplotlib rcParam"):
            PlotStyle(rc={"axes.facecolour": "red"})
        with pytest.raises(ValueError, match="sequential_range"):
            Palettes(sequential_range=(0.8, 0.2))

    def test_copy_with_and_rc_update(self):
        base = PlotStyle.default()
        changed = base.copy_with(rc_update={"font.size": 14}, color_scale="linear")
        assert changed.rc["font.size"] == 14 and changed.rc["figure.facecolor"] == base.rc["figure.facecolor"]
        assert base.rc.get("font.size") is None and changed.color_scale == "linear"
        assert base.copy_with(rc={"font.size": 9}).rc == {"font.size": 9}

    def test_from_mplstyle(self):
        style = PlotStyle.from_mplstyle("ggplot", color_scale="log")
        assert style.rc["axes.facecolor"] == matplotlib.style.library["ggplot"]["axes.facecolor"]
        assert style.color_scale == "log"

    def test_rc_context_applies_and_restores(self):
        style = PlotStyle.default()
        before = matplotlib.rcParams["axes.facecolor"]
        with style.rc_context():
            assert matplotlib.rcParams["axes.facecolor"] == style.rc["axes.facecolor"]
            _, ax = plt.subplots()
            assert ax.get_facecolor()[:3] != (1.0, 1.0, 1.0)
        assert matplotlib.rcParams["axes.facecolor"] == before
        assert style.rc_params() == dict(style.rc)

    def test_apply_sets_surface_on_existing_figure(self):
        fig, ax = plt.subplots()
        PlotStyle.dark().apply(fig)
        assert matplotlib.colors.to_hex(fig.get_facecolor()) == PlotStyle.dark().rc["figure.facecolor"]
        assert matplotlib.colors.to_hex(ax.get_facecolor()) == PlotStyle.dark().rc["axes.facecolor"]


class TestPalettes:
    def test_sequential_is_truncated(self):
        cmap = PlotStyle(palettes=Palettes(sequential="YlGnBu", sequential_range=(0.5, 1.0))).colormap("sequential")
        full = matplotlib.colormaps["YlGnBu"]
        np.testing.assert_allclose(cmap(0.0), full(0.5), atol=0.01)
        np.testing.assert_allclose(cmap(1.0), full(1.0), atol=0.01)

    def test_channel_palettes_override_the_style_palettes(self):
        style = PlotStyle(
            palettes=Palettes(sequential="Blues", diverging="RdBu"),
            channels={"edge": Channel(palettes=Palettes(sequential="Oranges"))},
        )
        assert style.colormap("sequential").name == "Blues"
        assert style.colormap("sequential", "edge").name == "Oranges"
        assert style.colormap("diverging", "edge").name == "RdBu"  # falls back
        with pytest.raises(ValueError, match="role"):
            style.colormap("categorical")

    def test_categorical_forms(self):
        cats = ["a", "b", "c"]
        by_list = PlotStyle(palettes=Palettes(categorical=["#ff0000", "#00ff00"])).categorical_colors(cats)
        assert by_list == ["#ff0000", "#00ff00", "#ff0000"]
        by_name = PlotStyle(palettes=Palettes(categorical="tab10")).categorical_colors(cats)
        assert by_name[0] == "#1f77b4"
        by_dict = PlotStyle(palettes=Palettes(categorical={"b": "#123456"})).categorical_colors(cats)
        assert by_dict[1] == "#123456" and by_dict[0] != "#123456"


class TestNorm:
    def test_scales(self):
        style = PlotStyle(color_scale="auto")
        assert isinstance(style.norm([1, 1000]), LogNorm)  # three decades
        assert type(style.norm([5, 20])) is Normalize  # one decade or less
        assert type(style.norm([0, 1000])) is Normalize  # not positive: linear
        assert isinstance(PlotStyle().norm([1, 1000], scale="log"), LogNorm)
        assert type(PlotStyle().norm([1, 1000])) is Normalize
        with pytest.raises(ValueError, match="positive"):
            PlotStyle().norm([0, 5], scale="log")
        with pytest.raises(ValueError, match="scale"):
            PlotStyle().norm([1, 5], scale="cubic")

    def test_diverging_is_symmetric(self):
        norm = PlotStyle().norm([-1, 3], "diverging")
        assert (norm.vmin, norm.vmax) == (-3, 3)
        norm = PlotStyle().norm([8, 14], "diverging", center=10)
        assert (norm.vmin, norm.vmax) == (6, 14)

    def test_log_span_sets_the_auto_threshold(self):
        assert type(PlotStyle(color_scale="auto", log_span=2).norm([1, 50])) is Normalize
        assert isinstance(PlotStyle(color_scale="auto", log_span=2).norm([1, 500]), LogNorm)


class TestColorbar:
    def test_log_norm_gets_ticks_at_1_2_5_with_si_labels(self):
        style = PlotStyle(tick_format="si", colorbar_outline_color="red", colorbar_outline_width=2.0)
        fig, ax = plt.subplots()
        mappable = plt.cm.ScalarMappable(norm=LogNorm(1e6, 4e7), cmap="viridis")
        mappable.set_array([])
        cbar = style.colorbar(fig, ax, mappable, label="Population")
        fig.canvas.draw()
        labels = [t.get_text() for t in cbar.ax.get_yticklabels()]
        assert labels == ["1M", "2M", "5M", "10M", "20M"]
        assert cbar.ax.get_ylabel() == "Population"
        assert all(s.get_linewidth() == 2.0 for s in cbar.ax.spines.values())

    def test_placement_fields_and_call_arguments(self):
        style = PlotStyle(colorbar_shrink=0.5, colorbar_orientation="horizontal")
        fig, ax = plt.subplots()
        mappable = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap="viridis")
        mappable.set_array([])
        cbar = style.colorbar(fig, ax, mappable, shrink=0.9)
        assert cbar.orientation == "horizontal"

    def test_si_labels_on_a_linear_colorbar(self):
        style = PlotStyle(tick_format="si")
        fig, ax = plt.subplots()
        mappable = plt.cm.ScalarMappable(norm=Normalize(0, 4e6), cmap="viridis")
        mappable.set_array([])
        cbar = style.colorbar(fig, ax, mappable)
        fig.canvas.draw()
        assert any(t.get_text().endswith("M") for t in cbar.ax.get_yticklabels())


class TestSelection:
    def test_default_is_the_style_in_effect(self):
        assert resolve_style() == PlotStyle.default()
        assert get_style() == PlotStyle.default()

    def test_names_objects_and_errors(self):
        assert resolve_style("dark") == PlotStyle.dark()
        mine = PlotStyle(color_scale="log")
        assert resolve_style(mine) is mine
        with pytest.raises(ValueError, match="no style named"):
            resolve_style("nope")
        with pytest.raises(TypeError, match="PlotStyle"):
            resolve_style(3)  # type: ignore[arg-type]

    def test_precedence_call_context_session_default(self):
        set_style("dark")
        assert resolve_style() == PlotStyle.dark()
        with style_context("classic"):
            assert resolve_style() == PlotStyle.classic()
            assert resolve_style("default") == PlotStyle.default()  # the call wins
            with style_context(PlotStyle.default()):
                assert resolve_style() == PlotStyle.default()
            assert resolve_style() == PlotStyle.classic()
        assert resolve_style() == PlotStyle.dark()
        set_style(None)
        assert resolve_style() == PlotStyle.default()

    def test_context_is_reset_after_an_exception(self):
        with pytest.raises(RuntimeError), style_context("dark"):
            raise RuntimeError
        assert resolve_style() == PlotStyle.default()

    def test_context_is_not_seen_by_another_thread(self):
        seen = []
        with style_context("dark"):
            thread = threading.Thread(target=lambda: seen.append(resolve_style()))
            thread.start()
            thread.join()
        assert seen == [PlotStyle.default()]

    def test_register(self):
        mine = PlotStyle(color_scale="log")
        register_style("my-test-style", mine)
        assert resolve_style("my-test-style") is mine
        with pytest.raises(ValueError, match="registered"):
            register_style("my-test-style", PlotStyle())
        register_style("my-test-style", PlotStyle(), overwrite=True)
        assert resolve_style("my-test-style") == PlotStyle()
        for builtin in ("default", "dark", "classic"):
            with pytest.raises(ValueError, match="built-in"):
                register_style(builtin, mine, overwrite=True)
        with pytest.raises(TypeError):
            register_style("x", "not a style")  # type: ignore[arg-type]

    def test_unregister(self):
        register_style("my-removable-style", PlotStyle(color_scale="log"))
        assert resolve_style("my-removable-style").color_scale == "log"
        unregister_style("my-removable-style")
        with pytest.raises(ValueError, match="no style named"):
            resolve_style("my-removable-style")
        with pytest.raises(ValueError, match="no registered style"):
            unregister_style("my-removable-style")
        for builtin in ("default", "dark", "classic"):
            with pytest.raises(ValueError, match="built-in"):
                unregister_style(builtin)
        register_style("my-removable-style", PlotStyle())  # the name is free again
        unregister_style("my-removable-style")

    def test_package_exports(self):
        for name in (
            "PlotStyle",
            "Palettes",
            "Channel",
            "register_style",
            "unregister_style",
            "get_style",
            "set_style",
            "style_context",
        ):
            assert hasattr(carto_flow, name)


class TestStyledDecorator:
    def test_runs_with_the_style_and_restores_rcparams(self):
        seen = {}

        @styled
        def plot(ax=None):
            seen["rc"] = matplotlib.rcParams["axes.facecolor"]
            return ax

        before = matplotlib.rcParams["axes.facecolor"]
        plot(style="dark")
        assert seen["rc"] == PlotStyle.dark().rc["axes.facecolor"]
        assert matplotlib.rcParams["axes.facecolor"] == before
        assert "style" in inspect.signature(plot).parameters

    def test_a_passed_axes_gets_the_surface(self):
        @styled
        def plot(ax=None):
            return ax

        _, ax = plt.subplots()
        plot(ax=ax, style="dark")
        assert matplotlib.colors.to_hex(ax.get_facecolor()) == PlotStyle.dark().rc["axes.facecolor"]

    def test_rcparams_are_restored_after_an_exception(self):
        @styled
        def plot():
            raise RuntimeError

        before = matplotlib.rcParams["axes.facecolor"]
        with pytest.raises(RuntimeError):
            plot(style="dark")
        assert matplotlib.rcParams["axes.facecolor"] == before


class TestMerge:
    def test_what_the_other_style_sets_wins(self):
        base = PlotStyle.default()
        fonts = PlotStyle(rc={"xtick.labelsize": 20}, tick_format="plain")
        merged = base | fonts
        assert merged.rc["xtick.labelsize"] == 20 and merged.rc["figure.facecolor"] == base.rc["figure.facecolor"]
        assert merged.tick_format == "plain" and merged.color_scale == base.color_scale  # None does not override
        assert base.tick_format == "si" and "xtick.labelsize" not in base.rc  # the operands are unchanged

    def test_palettes_and_channels_merge_field_by_field(self):
        base = PlotStyle(
            palettes=Palettes(sequential="Blues", sequential_range=(0.2, 1.0), diverging="PuOr"),
            channels={"edge": Channel(value="#111111", range=(1, 2)), "linewidth": Channel(value=0.5)},
        )
        other = PlotStyle(
            palettes=Palettes(sequential="Greens"),
            channels={"edge": Channel(palettes=Palettes(sequential="Oranges")), "fill": Channel(value="#abcdef")},
        )
        merged = base.merge(other)
        assert merged.palettes.sequential == "Greens" and merged.palettes.sequential_range == (0.2, 1.0)
        assert merged.palettes.diverging == "PuOr"
        edge = merged.channels["edge"]
        assert edge.value == "#111111" and edge.range == (1, 2) and edge.palettes.sequential == "Oranges"
        assert merged.channels["linewidth"].value == 0.5 and merged.channels["fill"].value == "#abcdef"

    def test_names_and_reflected_operator(self):
        assert (PlotStyle() | "dark") == PlotStyle.dark()
        assert ("dark" | PlotStyle(color_scale="log")).color_scale == "log"
        assert ("dark" | PlotStyle(color_scale="log")).rc == PlotStyle.dark().rc
        with pytest.raises(TypeError):
            PlotStyle() | 3  # type: ignore[operator]

    def test_a_sequence_is_merged_in_order(self):
        talk = PlotStyle(rc={"xtick.labelsize": 20})
        assert resolve_style(("dark", talk)) == PlotStyle.dark() | talk
        assert resolve_style(["classic"]) == PlotStyle.classic()
        with pytest.raises(ValueError, match="empty"):
            resolve_style(())
        with style_context(("dark", talk)):
            assert resolve_style().rc["xtick.labelsize"] == 20
        set_style(("dark", talk))
        assert resolve_style().rc["figure.facecolor"] == PlotStyle.dark().rc["figure.facecolor"]

    def test_log_span_merges_like_any_field(self):
        assert (PlotStyle.default() | PlotStyle(log_span=3)).log_span == 3
        assert PlotStyle.default().log_span is None


def test_sqrt_scale():
    from matplotlib.colors import PowerNorm

    assert isinstance(PlotStyle().norm([1, 5], scale="sqrt"), PowerNorm)
    assert PlotStyle(color_scale="sqrt").color_scale == "sqrt"


class TestPersistence:
    @pytest.mark.parametrize("name", ["default", "dark", "classic"])
    def test_builtin_styles_round_trip(self, name):
        import json

        style = {"default": PlotStyle.default, "dark": PlotStyle.dark, "classic": PlotStyle.classic}[name]()
        assert PlotStyle.from_dict(json.loads(json.dumps(style.to_dict()))) == style

    def test_a_file_round_trip_keeps_channels_ranges_and_palettes(self, tmp_path):
        style = PlotStyle.default().copy_with(
            hatch_cycle=("//", "xx"),
            palettes=Palettes(sequential="magma", sequential_range=(0.1, 0.9), categorical={"a": "#ff0000"}),
            channels={"alpha": Channel(value=0.5, range=(0.2, 1.0), scale="sqrt", palettes=Palettes(diverging="PuOr"))},
        )
        style.save(tmp_path / "s.json")
        assert PlotStyle.load(tmp_path / "s.json") == style

    def test_a_colormap_is_stored_by_name_or_by_colors(self):
        from matplotlib.colors import LinearSegmentedColormap

        named = PlotStyle(palettes=Palettes(sequential=matplotlib.colormaps["viridis"]))
        assert named.to_dict()["palettes"]["sequential"] == "viridis"
        custom = LinearSegmentedColormap.from_list("mine", ["#000000", "#ff0000"])
        restored = PlotStyle.from_dict(PlotStyle(palettes=Palettes(sequential=custom)).to_dict())
        np.testing.assert_allclose(
            restored.palettes.sequential(np.linspace(0, 1, 9)), custom(np.linspace(0, 1, 9)), atol=2e-3
        )

    def test_a_style_sheet_with_a_prop_cycle_round_trips(self):
        style = PlotStyle.from_mplstyle("ggplot")
        restored = PlotStyle.from_dict(style.to_dict())
        assert restored.rc["axes.prop_cycle"].by_key() == style.rc["axes.prop_cycle"].by_key()

    def test_only_set_fields_are_written(self):
        assert PlotStyle().to_dict() == {"version": 1}
        assert PlotStyle(tick_format="si").to_dict() == {"version": 1, "tick_format": "si"}

    def test_errors(self):
        with pytest.raises(ValueError, match="unknown style keys"):
            PlotStyle.from_dict({"version": 1, "colour_scale": "log"})
        with pytest.raises(ValueError, match="version"):
            PlotStyle.from_dict({"version": 2})
        with pytest.raises(ValueError, match="unknown palette keys"):
            PlotStyle.from_dict({"palettes": {"sequentail": "viridis"}})
        with pytest.raises(ValueError, match="color_scale"):
            PlotStyle.from_dict({"color_scale": "cubic"})
        with pytest.raises(TypeError, match="rc"):
            PlotStyle(rc={"axes.formatter.limits": object()}).to_dict()

    def test_a_loaded_style_can_be_registered_and_selected(self, tmp_path):
        PlotStyle.dark().save(tmp_path / "d.json")
        register_style("from-file", PlotStyle.load(tmp_path / "d.json"))
        try:
            assert resolve_style("from-file") == PlotStyle.dark()
        finally:
            unregister_style("from-file")
