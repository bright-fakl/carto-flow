"""Tests for square collision in the centroid layout."""

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import box

from carto_flow.symbol_cartogram import Styling, create_symbol_cartogram
from carto_flow.symbol_cartogram.layouts.centroid import CentroidLayout
from carto_flow.symbol_cartogram.layouts.centroid._placement import resolve_overlaps
from carto_flow.symbol_cartogram.symbols import CircleSymbol, SquareSymbol


def cluster(n=30, seed=1):
    rng = np.random.default_rng(seed)
    geoms = [box(x, y, x + 1, y + 1) for x, y in rng.uniform(0, 4, (n, 2))]
    return gpd.GeoDataFrame({"v": rng.uniform(1, 20, n)}, geometry=geoms)


def overlap_area(symbols):
    geoms = np.asarray(symbols.geometry.values, dtype=object)
    first, second = shapely.STRtree(geoms).query(geoms, predicate="intersects")
    keep = first < second
    return float(
        sum(shapely.intersection(geoms[i], geoms[j]).area for i, j in zip(first[keep], second[keep], strict=False))
    )


def test_resolve_squares_flush():
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 3, (40, 2))
    half = rng.uniform(0.2, 0.5, 40)
    out, info = resolve_overlaps(positions, half, shape="square", spacing=0.0, max_iterations=100)
    dx = np.abs(out[:, None, 0] - out[None, :, 0])
    dy = np.abs(out[:, None, 1] - out[None, :, 1])
    separation = np.maximum(dx, dy) - (half[:, None] + half[None, :])
    np.fill_diagonal(separation, 1.0)
    assert separation.min() > -1e-3 * half.mean()
    assert info["final_max_overlap"] < 1e-3


def test_resolve_squares_separates_coincident_squares_along_an_axis():
    out, _ = resolve_overlaps(np.zeros((2, 2)), np.array([0.5, 0.5]), shape="square", spacing=0.0, max_iterations=50)
    offset = np.abs(out[0] - out[1])
    assert offset.max() >= 1.0 - 1e-3
    assert offset.min() < 1e-9


def test_resolve_circles_unchanged_by_default():
    positions = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    radii = np.full(3, 0.4)
    default, _ = resolve_overlaps(positions, radii, spacing=0.1)
    explicit, _ = resolve_overlaps(positions, radii, spacing=0.1, shape="circle")
    assert np.array_equal(default, explicit)


def test_square_layout_has_square_canonical_symbol():
    result = create_symbol_cartogram(cluster(), "v", layout=CentroidLayout(shape="square"), show_progress=False)
    assert isinstance(result.layout_result.canonical_symbol, SquareSymbol)
    circle = create_symbol_cartogram(cluster(), "v", layout=CentroidLayout(), show_progress=False)
    assert isinstance(circle.layout_result.canonical_symbol, CircleSymbol)


@pytest.mark.parametrize("spacing", [0.0, 0.05])
def test_squares_do_not_overlap(spacing):
    gdf = cluster()
    result = create_symbol_cartogram(
        gdf,
        "v",
        layout=CentroidLayout(shape="square", spacing=spacing),
        styling=Styling(symbol="square"),
        show_progress=False,
    )
    area = shapely.area(np.asarray(result.symbols.geometry.values, dtype=object))
    assert overlap_area(result.symbols) < 1e-3 * area.mean()


def test_square_area_matches_circle_area_and_is_proportional_to_value():
    gdf = cluster()
    squares = create_symbol_cartogram(
        gdf, "v", layout=CentroidLayout(shape="square"), styling=Styling(symbol="square"), show_progress=False
    )
    circles = create_symbol_cartogram(
        gdf, "v", layout=CentroidLayout(), styling=Styling(symbol="circle"), show_progress=False
    )
    square_area = shapely.area(np.asarray(squares.symbols.geometry.values, dtype=object))
    circle_area = shapely.area(np.asarray(circles.symbols.geometry.values, dtype=object))
    ratio = square_area / gdf["v"].to_numpy()
    assert ratio.max() / ratio.min() == pytest.approx(1.0, abs=1e-6)
    # A polygon circle has slightly less area than the true circle
    assert square_area.sum() == pytest.approx(circle_area.sum(), rel=0.01)


def test_square_layout_with_groups():
    gdf = cluster()
    gdf["g"] = ["a", "b", "c"] * 10
    result = create_symbol_cartogram(
        gdf,
        "v",
        group_by="g",
        layout=CentroidLayout(shape="square"),
        styling=Styling(symbol="square"),
        show_progress=False,
    )
    area = shapely.area(np.asarray(result.symbols.geometry.values, dtype=object))
    assert overlap_area(result.symbols) < 1e-3 * area.mean()


def test_invalid_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        CentroidLayout(shape="hexagon")
