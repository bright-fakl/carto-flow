"""Zero-valued sizing variables in symbol cartogram layouts.

A zero sizing value is valid input: it yields a zero-radius symbol, which
every layout keeps as a row in the result.
"""

from __future__ import annotations

import warnings

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import MultiPoint, box
from shapely.ops import voronoi_diagram

from carto_flow.symbol_cartogram.layouts import get_layout, prepare_layout_data
from carto_flow.symbol_cartogram.layouts.packing._simulator import _distance_in_radii

# Layouts that place symbols from a sizing variable. Mosaic is exercised with
# and without its pre-morph step.
LAYOUT_CASES = [
    ("centroid", {}),
    ("flow_density", {"max_iterations": 50}),
    ("grid", {}),
    ("mosaic", {"morph": True}),
    ("mosaic", {"morph": False}),
    ("packing", {"max_iterations": 200}),
    ("physics", {"max_iterations": 100}),
]


def grid_gdf(zeros=(0, 5)) -> gpd.GeoDataFrame:
    """A 4x4 grid of unit squares, with some values set to zero."""
    rng = np.random.default_rng(1)
    geoms = [box(c, r, c + 1, r + 1) for r in range(4) for c in range(4)]
    values = rng.integers(100, 10000, size=len(geoms)).astype(float)
    for z in zeros:
        values[z] = 0.0
    return gpd.GeoDataFrame({"population": values}, geometry=geoms)


def voronoi_gdf(zeros=(2,)) -> gpd.GeoDataFrame:
    """Irregular Voronoi cells, so the input is not a regular lattice."""
    rng = np.random.default_rng(3)
    points = MultiPoint([tuple(p) for p in rng.uniform(0, 10, size=(12, 2))])
    envelope = box(0, 0, 10, 10)
    geoms = [g.intersection(envelope) for g in voronoi_diagram(points, envelope=envelope).geoms]
    values = rng.integers(100, 10000, size=len(geoms)).astype(float)
    for z in zeros:
        values[z] = 0.0
    return gpd.GeoDataFrame({"population": values}, geometry=geoms)


INPUTS = {"grid": grid_gdf, "voronoi": voronoi_gdf}


def run_layout(gdf, name, options):
    data = prepare_layout_data(gdf, "population")
    result = type(get_layout(name))(**options).compute(data, show_progress=False)
    positions = np.array([t.position for t in result.transforms], dtype=float)
    return data, result, positions


@pytest.mark.parametrize("input_name", sorted(INPUTS))
@pytest.mark.parametrize(("layout_name", "options"), LAYOUT_CASES)
class TestZeroSizeSymbols:
    """Every layout accepts a zero-valued sizing variable."""

    def test_no_runtime_warning(self, input_name, layout_name, options):
        """No divide-by-zero or invalid-value warning reaches the caller."""
        gdf = INPUTS[input_name]()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            np.seterr(all="warn")
            run_layout(gdf, layout_name, options)
        runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert not runtime, [str(w.message) for w in runtime]

    def test_row_count_and_finite_positions(self, input_name, layout_name, options):
        """The zero-size symbol keeps its row and gets a finite position."""
        gdf = INPUTS[input_name]()
        data, result, positions = run_layout(gdf, layout_name, options)
        assert len(result.transforms) == len(gdf)
        assert np.isfinite(positions).all()
        assert (data.sizes == 0).any()
        assert (data.sizes >= 0).all()


@pytest.mark.parametrize("input_name", sorted(INPUTS))
def test_packing_converges_with_zero_size_symbol(input_name):
    """A zero-size symbol must not defeat the packing convergence test.

    Drift and jitter are displacements measured in symbol radii, so a
    zero-radius symbol made the averages non-finite and the run could never
    report convergence.
    """
    gdf = INPUTS[input_name]()
    _, result, _ = run_layout(gdf, "packing", {"max_iterations": 300})
    assert result.converged
    assert np.isfinite(result.metrics.algorithm.final_drift)
    assert np.isfinite(result.metrics.algorithm.final_jitter)


class TestDistanceInRadii:
    """Distances expressed in symbol radii."""

    def test_positive_radius_is_plain_ratio(self):
        ratio = _distance_in_radii(np.array([1.0, 3.0]), np.array([2.0, 1.5]))
        assert ratio == pytest.approx([0.5, 2.0])

    def test_zero_radius_is_infinite(self):
        """A zero-radius symbol is infinitely many radii from anywhere."""
        ratio = _distance_in_radii(np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        assert np.isinf(ratio).all()
