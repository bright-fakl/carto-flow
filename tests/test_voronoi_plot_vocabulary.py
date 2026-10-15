"""Tests for the parameter vocabulary of the Voronoi cartogram plot and animate functions."""

import warnings

import geopandas as gpd
import matplotlib
import numpy as np
import pytest
from shapely.geometry import box

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from carto_flow.voronoi_cartogram import RasterBackend, VoronoiOptions, create_voronoi_cartogram
from carto_flow.voronoi_cartogram.animation import animate_voronoi_history
from carto_flow.voronoi_cartogram.visualization import (
    plot_cartogram,
    plot_comparison,
    plot_displacement,
)


@pytest.fixture(scope="module")
def gdf():
    rng = np.random.default_rng(0)
    geoms = [box(c, r, c + 1, r + 1) for r in range(3) for c in range(3)]
    return gpd.GeoDataFrame(
        {"population": rng.integers(100, 10000, size=9).astype(float), "name": list("abcdefghi")}, geometry=geoms
    )


@pytest.fixture(scope="module")
def result(gdf):
    return create_voronoi_cartogram(
        gdf,
        backend=RasterBackend(resolution=50),
        options=VoronoiOptions(n_iter=5, record_history=True),
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _colors(plot_result):
    return np.concatenate([np.asarray(c.get_facecolor()) for c in plot_result.collections])


class TestFill:
    def test_column_builtin_literal_and_array(self, result, gdf):
        by_column = result.plot(facecolor="population")
        by_array = result.plot(facecolor=gdf["population"].to_numpy())
        np.testing.assert_allclose(_colors(by_column), _colors(by_array))
        assert np.allclose(_colors(result.plot(facecolor="#d4e6f1")), matplotlib.colors.to_rgba("#d4e6f1"))
        assert len(_colors(result.plot(facecolor="area_error_pct"))) == 9
        assert len(_colors(result.plot())) == 9  # categorical by index

    def test_unknown_string_raises(self, result):
        with pytest.raises(ValueError, match="neither a column"):
            result.plot(facecolor="not-a-column")

    def test_column_that_is_also_a_color_is_the_column(self, gdf):
        renamed = gdf.rename(columns={"population": "wheat"})
        res = create_voronoi_cartogram(renamed, backend=RasterBackend(resolution=50), options=VoronoiOptions(n_iter=5))
        with pytest.warns(UserWarning, match="column and a color"):
            res.plot(facecolor="wheat")

    def test_old_column_name(self, result):
        with pytest.warns(DeprecationWarning, match="'column' is deprecated; use 'facecolor'"):
            old = result.plot(column="population")
        np.testing.assert_allclose(_colors(old), _colors(result.plot(facecolor="population")))
        with pytest.warns(DeprecationWarning):
            plot_cartogram(result, column="population")

    def test_comparison(self, result, gdf):
        plot_comparison(gdf, result, facecolor="population")
        plot_comparison(gdf, result, facecolor="area_error_pct")
        plot_comparison(gdf, result, facecolor="#d4e6f1")
        with pytest.warns(DeprecationWarning, match="use 'facecolor'"):
            plot_comparison(gdf, result, column="population")


class TestKwargsNames:
    def test_displacement_element_dicts(self, result):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plot_displacement(result, geometry_kwargs={"alpha": 0.5}, displacement_kwargs={"color": "k"})
            result.plot_displacement(adjacency_kwargs={"alpha": 0.5})
        old = {
            "geometry_style": "geometry_kwargs",
            "adjacency_style": "adjacency_kwargs",
            "centroid_style": "centroid_kwargs",
            "position_style": "position_kwargs",
            "displacement_style": "displacement_kwargs",
        }
        for old_name, new_name in old.items():
            with pytest.warns(DeprecationWarning, match=f"'{old_name}' is deprecated; use '{new_name}'"):
                plot_displacement(result, **{old_name: {"alpha": 0.5}})
        with pytest.warns(DeprecationWarning, match="'displacement_style'"):
            result.plot_displacement(displacement_style={"color": "k"})

    def test_animate_element_dicts(self, result):
        animate_voronoi_history(result, cell_kwargs={"alpha": 0.5}, point_kwargs={"s": 5})
        with pytest.warns(DeprecationWarning, match="'cell_style' is deprecated; use 'cell_kwargs'"):
            animate_voronoi_history(result, cell_style={"alpha": 0.5})
        with pytest.warns(DeprecationWarning, match="'point_style' is deprecated; use 'point_kwargs'"):
            animate_voronoi_history(result, point_style={"s": 5})
