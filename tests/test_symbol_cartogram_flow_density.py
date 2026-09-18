"""Tests for FlowDensityLayout (symbol_cartogram.layouts.flow).

All tests use small synthetic grids of squares and a coarse density grid
(``grid_size=64``) so the whole module runs in about a second.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from carto_flow.symbol_cartogram import (
    FlowDensityLayout,
    FlowDensityLayoutOptions,
    create_symbol_cartogram,
)
from carto_flow.symbol_cartogram.layouts.flow import FlowDensityHistory


def make_grid_gdf(rows: int = 3, cols: int = 3, seed: int = 1) -> gpd.GeoDataFrame:
    """Grid of unit squares with varied values."""
    rng = np.random.default_rng(seed)
    geoms = [box(c, r, c + 1, r + 1) for r in range(rows) for c in range(cols)]
    return gpd.GeoDataFrame(
        {"value": rng.uniform(1.0, 10.0, rows * cols)},
        geometry=geoms,
    )


def make_two_clusters(separation: float = 6.0, seed: int = 0) -> gpd.GeoDataFrame:
    """Two 2x2 blocks of squares, *separation* units apart in x, labelled by group."""
    rng = np.random.default_rng(seed)
    geoms, groups, values = [], [], []
    for k, x0 in enumerate((0.0, separation)):
        for r in range(2):
            for c in range(2):
                geoms.append(box(x0 + c, r, x0 + c + 1, r + 1))
                groups.append(f"g{k}")
                values.append(float(rng.uniform(4.0, 10.0)))
    return gpd.GeoDataFrame({"value": values, "group": groups}, geometry=geoms)


def positions_and_radii(result) -> tuple[np.ndarray, np.ndarray]:
    """Final symbol centres and radii from a cartogram result."""
    xy = result.symbols[["_symbol_x", "_symbol_y"]].to_numpy(dtype=float)
    radii = result.symbols["_symbol_size"].to_numpy(dtype=float)
    return xy, radii


def total_overlap(xy: np.ndarray, radii: np.ndarray) -> float:
    """Sum of pairwise penetration depths (0 when no pair overlaps)."""
    total = 0.0
    for i in range(len(radii)):
        for j in range(i + 1, len(radii)):
            distance = float(np.hypot(*(xy[i] - xy[j])))
            total += max(0.0, radii[i] + radii[j] - distance)
    return total


def max_penetration(xy: np.ndarray, radii: np.ndarray) -> float:
    """Largest pairwise penetration depth."""
    return max(
        radii[i] + radii[j] - float(np.hypot(*(xy[i] - xy[j])))
        for i in range(len(radii))
        for j in range(i + 1, len(radii))
    )


class TestFlowDensityLayout:
    """Basic behaviour of the flow-density layout."""

    @pytest.mark.parametrize("layout", ["flow_density", "instance"])
    def test_runs_via_key_and_instance(self, layout):
        """Both the registry key and a layout instance produce a valid result."""
        gdf = make_grid_gdf()
        spec = FlowDensityLayout(max_iterations=50, grid_size=64) if layout == "instance" else "flow_density"
        result = create_symbol_cartogram(gdf, "value", layout=spec, show_progress=False)

        assert len(result.symbols) == len(gdf)
        xy, radii = positions_and_radii(result)
        assert np.isfinite(xy).all()
        assert np.isfinite(radii).all()
        assert (radii > 0).all()
        assert result.layout_result is not None

    def test_residual_overlap_is_small(self):
        """At convergence no pair overlaps by more than a small fraction of a radius.

        The layout converges on the *mean* relative nearest-neighbour spacing
        error, so a few pairs may still touch slightly; it does not guarantee
        exactly zero overlap. The bound here (15% of the mean radius) is well
        below what a failing layout would produce.
        """
        gdf = make_grid_gdf()
        result = create_symbol_cartogram(
            gdf,
            "value",
            layout=FlowDensityLayout(max_iterations=200, grid_size=64, convergence_tolerance=0.02),
            size_normalization="total",
            show_progress=False,
        )
        xy, radii = positions_and_radii(result)
        assert max_penetration(xy, radii) <= 0.15 * float(radii.mean())

    def test_overlapping_input_is_separated(self):
        """Circles that start heavily overlapping end up far less overlapped."""
        gdf = make_grid_gdf()
        result = create_symbol_cartogram(
            gdf,
            "value",
            layout=FlowDensityLayout(max_iterations=200, grid_size=64),
            size_normalization="total",
            show_progress=False,
        )
        layout_result = result.layout_result
        before = total_overlap(layout_result.positions, layout_result.sizes)
        after = total_overlap(*positions_and_radii(result))

        assert before > 0.0
        assert after < 0.5 * before

    def test_group_by_cross_group_pull(self):
        """``cross_group_pull_scale=0.0`` removes attraction between groups.

        Asserted: with two well-separated groups, disabling cross-group pull
        leaves the distance between the group centroids essentially unchanged,
        while the default (pull enabled) contracts it. Total isolation is not
        asserted: the velocity field is global, so a distant group is still
        advected slightly by the far field.
        """
        gdf = make_two_clusters(separation=6.0)
        results = {}
        for scale in (1.0, 0.0):
            results[scale] = create_symbol_cartogram(
                gdf,
                "value",
                group_by="group",
                layout=FlowDensityLayout(max_iterations=200, grid_size=64, cross_group_pull_scale=scale),
                size_normalization="total",
                show_progress=False,
            )

        def group_distance(result):
            xy, _ = positions_and_radii(result)
            return float(np.hypot(*(xy[:4].mean(axis=0) - xy[4:].mean(axis=0))))

        start = 6.0
        assert group_distance(results[0.0]) >= 0.98 * start
        assert group_distance(results[1.0]) < 0.98 * start

        # Groups stay coherent: each symbol is nearer its own group centroid.
        xy, _ = positions_and_radii(results[0.0])
        centroid_a, centroid_b = xy[:4].mean(axis=0), xy[4:].mean(axis=0)
        for i in range(4):
            assert np.linalg.norm(xy[i] - centroid_a) < np.linalg.norm(xy[i] - centroid_b)
        for i in range(4, 8):
            assert np.linalg.norm(xy[i] - centroid_b) < np.linalg.norm(xy[i] - centroid_a)

    def test_metrics_populated(self):
        """Layout metrics carry the FlowDensityMetrics fields."""
        gdf = make_grid_gdf()
        result = create_symbol_cartogram(
            gdf,
            "value",
            layout=FlowDensityLayout(max_iterations=50, grid_size=64),
            show_progress=False,
        )
        metrics = result.layout_result.metrics

        assert metrics is not None
        assert metrics.iterations > 0
        assert isinstance(metrics.converged, bool)
        assert metrics.final_overlaps >= 0
        algorithm = metrics.algorithm
        assert np.isfinite(algorithm.final_error)
        assert np.isfinite(algorithm.final_max_error)
        assert len(algorithm.final_signed_errors) == len(gdf)

    def test_save_history(self):
        """``save_history=True`` records position snapshots and error histories."""
        gdf = make_grid_gdf()
        result = create_symbol_cartogram(
            gdf,
            "value",
            layout=FlowDensityLayout(max_iterations=20, grid_size=64),
            show_progress=False,
            save_history=True,
        )
        history = result.layout_result.history

        assert history is not None
        assert history.positions is not None
        assert len(history.positions) >= 1
        assert isinstance(history.algorithm, FlowDensityHistory)
        assert len(history.algorithm.errors) >= 1
        assert len(history.algorithm.max_errors) == len(history.algorithm.errors)


class TestFlowDensityOptions:
    """Option validation."""

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"spacing": -0.1}, "spacing must be >= 0"),
            ({"dt_factor": 0.0}, "dt_factor must be positive"),
            ({"grid_size": 16}, "grid_size must be >= 32"),
            ({"force_balance": "nope"}, "force_balance must be"),
            ({"cross_group_pull_scale": 1.5}, "cross_group_pull_scale must be in"),
        ],
    )
    def test_invalid_options_raise(self, kwargs, message):
        """Invalid option values raise ValueError with a descriptive message."""
        with pytest.raises(ValueError, match=message):
            FlowDensityLayoutOptions(**kwargs).validate()

    def test_layout_validates_on_construction(self):
        """FlowDensityLayout validates its options in __init__."""
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            FlowDensityLayout(max_iterations=0)
