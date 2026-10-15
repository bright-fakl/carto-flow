"""Tests for the parameter vocabulary of the proportional cartogram plot functions."""

import warnings

import geopandas as gpd
import matplotlib
import numpy as np
import pytest
from shapely.geometry import box

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from carto_flow.proportional_cartogram import partition_geometries, plot_dot_density, plot_partitions


@pytest.fixture(scope="module")
def gdf():
    rng = np.random.default_rng(0)
    geoms = [box(c, r, c + 1, r + 1) for r in range(3) for c in range(3)]
    return gpd.GeoDataFrame(
        {
            "pop_a": rng.integers(100, 10000, size=9).astype(float),
            "pop_b": rng.integers(100, 10000, size=9).astype(float),
            "remainder_value": rng.random(9),
        },
        geometry=geoms,
    )


@pytest.fixture(scope="module")
def partitioned(gdf):
    return partition_geometries(gdf, columns=["pop_a", "pop_b"], method="split", normalization="sum")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _colors(result):
    return np.concatenate([np.asarray(c.get_facecolor()) for c in result.partition_collections])


class TestPartitions:
    def test_default_and_builtin_fill_agree(self, partitioned):
        np.testing.assert_allclose(
            _colors(plot_partitions(partitioned)), _colors(plot_partitions(partitioned, facecolor="area"))
        )

    def test_category_columns_and_color(self, partitioned):
        plot_partitions(partitioned, facecolor="category")
        plot_partitions(partitioned, facecolor=["pop_a", "pop_b"])
        by_list = plot_partitions(partitioned, facecolor=["remainder_value"])
        by_name = plot_partitions(partitioned, facecolor="remainder_value")
        np.testing.assert_allclose(_colors(by_list), _colors(by_name))
        constant = plot_partitions(partitioned, facecolor="#d4e6f1")
        assert np.allclose(_colors(constant), matplotlib.colors.to_rgba("#d4e6f1"))
        assert constant.colorbar is None and constant.legend is None

    def test_cmap_dict_is_the_category_palette(self, partitioned):
        palette = {"pop_a": "#ff0000", "pop_b": "#0000ff"}
        result = plot_partitions(partitioned, facecolor="category", cmap=palette)
        colors = {tuple(np.round(c, 3)) for c in _colors(result)}
        assert colors == {tuple(np.round(matplotlib.colors.to_rgba(c), 3)) for c in palette.values()}

    def test_background_and_complement_names(self, partitioned):
        plot_partitions(partitioned, highlight="pop_a", background_facecolor="#cccccc")
        plot_partitions(partitioned, complement_facecolor="#eeeeee")

    def test_unknown_string_raises(self, partitioned):
        with pytest.raises(ValueError, match="neither a column"):
            plot_partitions(partitioned, facecolor="not-a-column")

    def test_old_names(self, partitioned):
        with pytest.warns(DeprecationWarning, match="'color_by' is deprecated; use 'facecolor'"):
            old = plot_partitions(partitioned, color_by="category")
        np.testing.assert_allclose(_colors(old), _colors(plot_partitions(partitioned, facecolor="category")))
        with pytest.warns(DeprecationWarning, match="'palette' is deprecated; pass the dict as 'cmap'"):
            plot_partitions(partitioned, facecolor="category", palette={"pop_a": "red"})
        with pytest.warns(DeprecationWarning, match="'background_color' is deprecated; use 'background_facecolor'"):
            plot_partitions(partitioned, highlight="pop_a", background_color="#cccccc")
        with pytest.warns(DeprecationWarning, match="'complement_color' is deprecated; use 'complement_facecolor'"):
            plot_partitions(partitioned, complement_color="#eeeeee")
        with pytest.warns(DeprecationWarning, match="'complement_color_by' is deprecated"):
            plot_partitions(partitioned, complement_color_by="remainder_value")


class TestDotDensity:
    def test_new_names(self, gdf):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plot_dot_density(
                gdf,
                columns=["pop_a", "pop_b"],
                n_dots=20,
                normalization="sum",
                seed=0,
                cmap={"pop_a": "red", "pop_b": "blue"},
                markersize=6,
                reference_facecolor="#eeeeee",
                reference_edgecolor="k",
                reference_linewidth=1,
                reference_alpha=0.5,
            )

    def test_old_names(self, gdf):
        old = {
            "size": ("markersize", 6),
            "base_color": ("reference_facecolor", "#eeeeee"),
            "base_edgecolor": ("reference_edgecolor", "k"),
            "base_linewidth": ("reference_linewidth", 1.0),
            "base_alpha": ("reference_alpha", 0.5),
        }
        for old_name, (new_name, value) in old.items():
            with pytest.warns(DeprecationWarning, match=f"'{old_name}' is deprecated; use '{new_name}'"):
                plot_dot_density(gdf, columns="pop_a", n_dots=10, normalization="sum", seed=0, **{old_name: value})
        with pytest.warns(DeprecationWarning, match="'palette' is deprecated"):
            plot_dot_density(
                gdf, columns=["pop_a", "pop_b"], n_dots=10, normalization="sum", seed=0, palette={"pop_a": "red"}
            )
