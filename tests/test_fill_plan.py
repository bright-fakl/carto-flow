"""Tests for the shared resolution of ``facecolor`` into a fill."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from carto_flow._fill import fill_plan


@pytest.fixture
def gdf():
    return gpd.GeoDataFrame(
        {
            "number": [1.0, 2.0, 3.0],
            "text": ["a", "b", "c"],
            "flag": [True, False, True],
            "error": [-1.0, 0.0, 2.0],
            "red": [1.0, 2.0, 3.0],
        },
        geometry=[box(i, 0, i + 1, 1) for i in range(3)],
    )


def test_none(gdf):
    out, fill = fill_plan(gdf, None, None, None)
    assert out is gdf
    assert fill.kwargs == {} and fill.role is None
    _, with_default = fill_plan(gdf, None, None, None, none_kwargs={"facecolor": "steelblue"})
    assert with_default.kwargs == {"facecolor": "steelblue"}


def test_color_and_color_array(gdf):
    assert fill_plan(gdf, "#ff0000", None, None)[1].kwargs == {"facecolor": "#ff0000"}
    colors = ["#ff0000", "#00ff00", "#0000ff"]
    assert fill_plan(gdf, colors, None, None)[1].kwargs == {"facecolor": colors}


@pytest.mark.parametrize(
    ("column", "role"), [("number", "sequential"), ("text", "categorical"), ("flag", "categorical")]
)
def test_roles_by_dtype(gdf, column, role):
    _, fill = fill_plan(gdf, column, None, None)
    assert (fill.role, fill.column, fill.label) == (role, column, column)


def test_categorical_flag_and_center(gdf):
    assert fill_plan(gdf, "number", True, None)[1].role == "categorical"
    assert fill_plan(gdf, "text", False, None)[1].role == "sequential"  # False only prevents the categorical role
    assert fill_plan(gdf, "number", None, 2.0)[1].role == "diverging"


def test_diverging_columns_and_labels(gdf):
    _, fill = fill_plan(gdf, "error", None, None, diverging=("error",), labels={"error": "Error %"})
    assert (fill.role, fill.label) == ("diverging", "Error %")


def test_builtin(gdf):
    _, fill = fill_plan(gdf, "errors", None, None, builtins={"errors": "error"}, labels={"error": "Error %"})
    assert (fill.column, fill.label) == ("error", "Error %")
    with pytest.raises(ValueError, match="not available"):
        fill_plan(gdf, "density", None, None, builtins={"density": "_missing"})


def test_array_of_values_becomes_a_column(gdf):
    out, fill = fill_plan(gdf, np.array([3.0, 2.0, 1.0]), None, None)
    assert fill.column == "_facecolor" and fill.label is None and fill.role == "sequential"
    assert list(out["_facecolor"]) == [3.0, 2.0, 1.0]
    assert "_facecolor" not in gdf.columns


def test_column_that_is_also_a_color_warns_and_wins(gdf):
    with pytest.warns(UserWarning, match="red"):
        _, fill = fill_plan(gdf, "red", None, None)
    assert fill.column == "red"


def test_unknown_string_is_an_error(gdf):
    with pytest.raises(ValueError):
        fill_plan(gdf, "not-a-column-or-color", None, None)


class TestCategoricalColorDict:
    @pytest.fixture
    def regions(self):
        return gpd.GeoDataFrame(
            {"group": ["a", "b", "a", "c"]}, geometry=[box(i, 0, i + 1, 1) for i in range(4)], crs="EPSG:3857"
        )

    def test_dict_colors_the_categories(self, regions):
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_hex

        from carto_flow import PlotStyle
        from carto_flow._fill import Fill, plot_fill

        _, ax = plt.subplots()
        plot_fill(
            ax,
            regions,
            Fill({"column": "group"}, "categorical", "group", "group"),
            PlotStyle(),
            cmap={"a": "#ff0000", "b": "#0000ff", "c": "#00ff00"},
            default_cmap=None,
            norm=None,
            center=None,
            legend=False,
            legend_kwargs=None,
            legend_label=None,
            categorical=None,
            kwargs={},
        )
        colors = [to_hex(c) for c in ax.collections[0].get_facecolor()]
        assert colors == ["#ff0000", "#0000ff", "#ff0000", "#00ff00"]
        plt.close("all")

    def test_dict_in_the_plots(self, regions):
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_hex

        from carto_flow.noncontiguous_cartogram import create_noncontiguous_cartogram

        regions = regions.assign(value=[1.0, 2.0, 3.0, 4.0])
        plot = create_noncontiguous_cartogram(regions, "value").plot(
            facecolor="group", cmap={"a": "#ff0000", "b": "#0000ff"}, legend=False, show_original=False
        )
        colors = {to_hex(c) for c in plot.collections[0].get_facecolor()}
        assert {"#ff0000", "#0000ff"} <= colors
        plt.close("all")
