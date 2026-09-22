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
from carto_flow.symbol_cartogram.tiling import resolve_tiling

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


def all_zero_grid_gdf() -> gpd.GeoDataFrame:
    """A 3x3 grid of unit squares whose sizing values are all zero."""
    geoms = [box(c, r, c + 1, r + 1) for r in range(3) for c in range(3)]
    return gpd.GeoDataFrame({"population": np.zeros(len(geoms))}, geometry=geoms)


def all_zero_voronoi_gdf() -> gpd.GeoDataFrame:
    """Irregular Voronoi cells whose sizing values are all zero."""
    gdf = voronoi_gdf(zeros=())
    gdf["population"] = 0.0
    return gdf


ALL_ZERO_INPUTS = {"grid": all_zero_grid_gdf, "voronoi": all_zero_voronoi_gdf}


@pytest.mark.parametrize("input_name", sorted(ALL_ZERO_INPUTS))
@pytest.mark.parametrize(("layout_name", "options"), LAYOUT_CASES)
class TestAllZeroSizingColumn:
    """Every sizing value being zero is still valid input."""

    def test_no_exception_and_no_runtime_warning(self, input_name, layout_name, options):
        gdf = ALL_ZERO_INPUTS[input_name]()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            np.seterr(all="warn")
            run_layout(gdf, layout_name, options)
        runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert not runtime, [str(w.message) for w in runtime]

    def test_row_count_and_finite_positions(self, input_name, layout_name, options):
        gdf = ALL_ZERO_INPUTS[input_name]()
        data, result, positions = run_layout(gdf, layout_name, options)
        assert len(result.transforms) == len(gdf)
        assert np.isfinite(positions).all()
        assert (data.sizes == 0).all()


@pytest.mark.parametrize("input_name", sorted(ALL_ZERO_INPUTS))
def test_grid_layout_terminates_with_all_zero_sizes(input_name):
    """The grid lattice must have a positive cell size.

    A zero cell size gives a lattice step of zero, so tile generation walks
    the bounds forever.
    """
    gdf = ALL_ZERO_INPUTS[input_name]()
    _, result, _ = run_layout(gdf, "grid", {})
    assert result.metrics.algorithm.tile_size > 0
    assert 0 < result.metrics.algorithm.n_tiles < 10_000


@pytest.mark.parametrize("tiling_name", ["square", "hexagon", "triangle"])
def test_tiling_rejects_non_positive_tile_size(tiling_name):
    """Tile generation refuses a size it could never step across the bounds with."""
    tiling = resolve_tiling(tiling_name)
    with pytest.raises(ValueError, match="tile_size must be positive"):
        tiling.generate(bounds=(0.0, 0.0, 10.0, 10.0), tile_size=0.0)


@pytest.mark.parametrize("input_name", sorted(ALL_ZERO_INPUTS))
def test_force_layouts_leave_zero_size_symbols_at_their_centroids(input_name):
    """With nothing to pack, the force-based layouts are a no-op."""
    gdf = ALL_ZERO_INPUTS[input_name]()
    for layout_name in ("centroid", "flow_density", "packing", "physics"):
        data, _, positions = run_layout(gdf, layout_name, {})
        assert positions == pytest.approx(data.positions, abs=1e-9), layout_name
