"""Tests for square symbols in the circle packing layout."""

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import box

from carto_flow.symbol_cartogram import Styling, create_symbol_cartogram
from carto_flow.symbol_cartogram.layouts.packing import CirclePackingLayout
from carto_flow.symbol_cartogram.layouts.packing._simulator import TopologyPreservingSimulator
from carto_flow.symbol_cartogram.symbols import CircleSymbol, SquareSymbol


def cluster(n=40, seed=3):
    rng = np.random.default_rng(seed)
    geoms = [box(x, y, x + 1, y + 1) for x, y in rng.uniform(0, 5, (n, 2))]
    return gpd.GeoDataFrame({"v": rng.uniform(1, 20, n)}, geometry=geoms)


def square_gap(layout_result):
    pos = np.array([t.position for t in layout_result.transforms])
    half = layout_result.sizes
    distance = np.maximum(np.abs(pos[:, None, 0] - pos[None, :, 0]), np.abs(pos[:, None, 1] - pos[None, :, 1]))
    np.fill_diagonal(distance, np.inf)
    return (distance - half[:, None] - half[None, :]).min(), half.mean()


def run(shape, **kwargs):
    return create_symbol_cartogram(
        cluster(),
        "v",
        layout=CirclePackingLayout(shape=shape, **kwargs),
        styling=Styling(symbol=shape),
        show_progress=False,
    )


def test_square_layout_has_square_canonical_symbol():
    assert isinstance(run("square").layout_result.canonical_symbol, SquareSymbol)
    assert isinstance(run("circle").layout_result.canonical_symbol, CircleSymbol)


@pytest.mark.parametrize("spacing", [0.0, 0.05])
def test_squares_keep_the_spacing(spacing):
    gap, half = square_gap(run("square", spacing=spacing).layout_result)
    assert gap >= (spacing - 0.01) * half


def test_squares_do_not_overlap_as_polygons():
    symbols = run("square", spacing=0.0).symbols
    geoms = np.asarray(symbols.geometry.values, dtype=object)
    first, second = shapely.STRtree(geoms).query(geoms, predicate="intersects")
    keep = first < second
    overlap = sum(
        shapely.intersection(geoms[i], geoms[j]).area for i, j in zip(first[keep], second[keep], strict=False)
    )
    assert overlap < 1e-3 * shapely.area(geoms).mean()


def test_square_area_is_proportional_to_value():
    gdf = cluster()
    result = run("square")
    area = shapely.area(np.asarray(result.symbols.geometry.values, dtype=object))
    ratio = area / gdf["v"].to_numpy()
    assert ratio.max() / ratio.min() == pytest.approx(1.0, abs=1e-6)


def test_square_packing_with_groups_and_attraction():
    gdf = cluster()
    gdf["g"] = ["a", "b", "c", "d"] * 10
    result = create_symbol_cartogram(
        gdf,
        "v",
        group_by="g",
        layout=CirclePackingLayout(shape="square", spacing=0.0, group_weight=0.5, compactness=0.5),
        show_progress=False,
    )
    gap, half = square_gap(result.layout_result)
    assert gap > -0.01 * half


def test_default_shape_is_circle():
    default = create_symbol_cartogram(cluster(), "v", layout=CirclePackingLayout(), show_progress=False)
    explicit = create_symbol_cartogram(cluster(), "v", layout=CirclePackingLayout(shape="circle"), show_progress=False)
    assert np.array_equal(
        [t.position for t in default.layout_result.transforms], [t.position for t in explicit.layout_result.transforms]
    )


def test_invalid_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        CirclePackingLayout(shape="hexagon")


def make_simulator(positions, half, shape):
    return TopologyPreservingSimulator(
        positions=np.asarray(positions, float), radii=np.asarray(half, float), spacing=0.0, shape=shape
    )


def test_square_pair_distance_and_directions_follow_the_larger_axis():
    sim = make_simulator([[0, 0], [1, 0]], [0.5, 0.5], "square")
    diff = np.array([[3.0, 1.0], [-1.0, -4.0]])
    euclid = np.linalg.norm(diff, axis=1)
    assert np.array_equal(sim._gap_distance(diff, euclid), [3.0, 4.0])
    assert np.array_equal(sim._unit_directions(diff, euclid), [[1.0, 0.0], [0.0, -1.0]])


def test_circle_pair_distance_and_directions_are_euclidean():
    sim = make_simulator([[0, 0], [1, 0]], [0.5, 0.5], "circle")
    diff = np.array([[3.0, 4.0]])
    euclid = np.linalg.norm(diff, axis=1)
    assert np.array_equal(sim._gap_distance(diff, euclid), euclid)
    assert np.allclose(sim._unit_directions(diff, euclid), [[0.6, 0.8]])


def test_overlapping_squares_separate_along_one_axis():
    sim = make_simulator([[0.0, 0.0], [0.5, 0.2]], [0.5, 0.5], "square")
    sim.expansion = 0.0  # the Gauss-Seidel passes only; the global expansion would scale both axes
    y_before = abs(sim.positions[1, 1] - sim.positions[0, 1])
    sim.run_overlap_resolution()
    offset = np.abs(sim.positions[1] - sim.positions[0])
    assert offset.max() >= sim.radii.sum() - 1e-3 * sim.avg_radius
    assert offset.min() == pytest.approx(y_before)
