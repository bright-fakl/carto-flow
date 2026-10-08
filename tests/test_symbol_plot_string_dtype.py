"""Column-driven styling in SymbolCartogram.plot with pandas string dtypes."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import pytest
from shapely.geometry import box

from carto_flow.symbol_cartogram import CirclePackingLayout, create_symbol_cartogram


def make_test_gdf(n: int = 5) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"population": [100.0, 400.0, 900.0, 1600.0, 2500.0][:n]},
        geometry=[box(2.0 * i, 0.0, 2.0 * i + 2.0, 2.0) for i in range(n)],
    )


def _cartogram(dtype):
    gdf = make_test_gdf()
    gdf["cat"] = ["a", "b", "a", "c", "b"]
    gdf["cat"] = gdf["cat"].astype(dtype)
    result = create_symbol_cartogram(
        gdf, "population", layout=CirclePackingLayout(max_iterations=50), show_progress=False
    )
    return result, gdf


@pytest.mark.parametrize("dtype", [object, "string", "category"])
@pytest.mark.parametrize("param", ["edgecolor", "facecolor", "hatch"])
def test_plot_categorical_column_styling(dtype, param):
    result, gdf = _cartogram(dtype)
    kwargs = {param: "cat"}
    if param == "hatch":
        kwargs["edgecolor"] = "black"
    fig, ax = plt.subplots()
    try:
        result.plot(ax=ax, source_gdf=gdf, **kwargs)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("dtype", [float, "Int64", "Float64"])
def test_plot_numeric_column_styling(dtype):
    result, gdf = _cartogram(object)
    gdf["num"] = [1, 2, 3, 4, 5]
    gdf["num"] = gdf["num"].astype(dtype)
    fig, ax = plt.subplots()
    try:
        result.plot(ax=ax, source_gdf=gdf, facecolor="num", edgecolor="num")
    finally:
        plt.close(fig)
