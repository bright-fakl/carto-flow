"""Tests for voronoi_cartogram module."""

from typing import ClassVar

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from carto_flow.voronoi_cartogram import (
    ExactBackend,
    RasterBackend,
    VoronoiOptions,
    create_voronoi_cartogram,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_grid_gdf(rows: int = 3, cols: int = 3, seed: int = 42) -> gpd.GeoDataFrame:
    """Create a grid of adjacent unit squares with a population column."""
    rng = np.random.default_rng(seed)
    geoms = [box(c, r, c + 1, r + 1) for r in range(rows) for c in range(cols)]
    n = rows * cols
    return gpd.GeoDataFrame(
        {"population": rng.integers(100, 10_000, size=n).astype(float)},
        geometry=geoms,
    )


def make_disconnected_gdf() -> gpd.GeoDataFrame:
    """Two non-adjacent blobs — useful for geodesic-labeling tests."""
    geoms = [
        box(0, 0, 3, 3),  # left cluster
        box(0, 0, 1, 1),
        box(1, 0, 2, 1),
        box(10, 0, 13, 3),  # right cluster (gap > 4 units)
        box(10, 0, 11, 1),
        box(11, 0, 12, 1),
    ]
    return gpd.GeoDataFrame(
        {"population": np.ones(len(geoms), dtype=float)},
        geometry=geoms,
    )


@pytest.fixture
def gdf():
    return make_grid_gdf()


_FAST_OPTIONS = VoronoiOptions(n_iter=5)
_FAST_RASTER = RasterBackend(resolution=50)
_FAST_EXACT = ExactBackend()


# ---------------------------------------------------------------------------
# 1. Smoke tests
# ---------------------------------------------------------------------------


class TestSmoke:
    """Basic smoke tests: does it run without error?"""

    def test_raster_backend(self, gdf):
        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        assert result is not None
        assert len(result.cells) == len(gdf)
        assert len(result.positions) == len(gdf)

    def test_exact_backend(self, gdf):
        result = create_voronoi_cartogram(gdf, backend=_FAST_EXACT, options=_FAST_OPTIONS)
        assert result is not None
        assert len(result.cells) == len(gdf)
        assert len(result.positions) == len(gdf)

    def test_default_backend(self, gdf):
        result = create_voronoi_cartogram(gdf, options=_FAST_OPTIONS)
        assert result is not None

    def test_weighted_raster(self, gdf):
        result = create_voronoi_cartogram(gdf, weights="population", backend=_FAST_RASTER, options=_FAST_OPTIONS)
        assert len(result.cells) == len(gdf)

    def test_2x2_grid(self):
        result = create_voronoi_cartogram(make_grid_gdf(2, 2), backend=_FAST_RASTER, options=_FAST_OPTIONS)
        assert len(result.cells) == 4

    def test_single_geometry(self):
        gdf_one = gpd.GeoDataFrame({"pop": [1.0]}, geometry=[box(0, 0, 1, 1)])
        result = create_voronoi_cartogram(gdf_one, backend=_FAST_RASTER, options=VoronoiOptions(n_iter=2))
        assert len(result.cells) == 1


# ---------------------------------------------------------------------------
# 2. Convergence
# ---------------------------------------------------------------------------


class TestConvergence:
    """CV(area) should decrease (or at minimum not increase) with more iterations."""

    def test_cv_recorded(self, gdf):
        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        assert len(result.convergence_history) == _FAST_OPTIONS.n_iter

    def test_cv_non_negative(self, gdf):
        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        assert all(cv >= 0.0 for cv in result.convergence_history)

    def test_cv_generally_decreases(self, gdf):
        """Final CV should be lower than initial CV for a reasonable run."""
        options = VoronoiOptions(n_iter=20)
        result = create_voronoi_cartogram(gdf, backend=RasterBackend(resolution=100), options=options)
        assert result.convergence_history[-1] <= result.convergence_history[0]

    def test_metrics_keys(self, gdf):
        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        for key in ("n_iterations", "converged", "initial_area_cv", "final_area_cv"):
            assert key in result.metrics

    def test_early_stopping(self):
        options = VoronoiOptions(n_iter=50, area_cv_tol=1e6)  # impossible tight tol → converges
        result = create_voronoi_cartogram(
            make_grid_gdf(2, 2),
            backend=RasterBackend(resolution=50),
            options=options,
        )
        # Should stop before 50 iterations since tol is very lenient
        assert result.metrics["n_iterations"] <= 50


# ---------------------------------------------------------------------------
# 3. Geodesic mode (raster)
# ---------------------------------------------------------------------------


class TestGeodesicMode:
    """Geodesic labeling should assign pixels to the correct disconnected cluster."""

    def test_geodesic_runs(self):
        gdf = make_disconnected_gdf()
        result = create_voronoi_cartogram(
            gdf,
            backend=RasterBackend(resolution=50, distance_mode="geodesic"),
            options=VoronoiOptions(n_iter=3),
        )
        assert len(result.cells) == len(gdf)

    def test_geodesic_cells_non_empty(self):
        gdf = make_disconnected_gdf()
        result = create_voronoi_cartogram(
            gdf,
            backend=RasterBackend(resolution=50, distance_mode="geodesic"),
            options=VoronoiOptions(n_iter=3),
        )
        import shapely

        areas = shapely.area(result.cells)
        assert (areas > 0).all(), "Some cells have zero area in geodesic mode"


# ---------------------------------------------------------------------------
# 4. Topology analysis
# ---------------------------------------------------------------------------


class TestTopologyAnalysis:
    """analyze_topology() should return a well-formed TopologyAnalysis."""

    def test_analyze_runs(self, gdf):
        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        analysis = result.analyze_topology()
        assert analysis is not None

    def test_adjacency_checked(self, gdf):
        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        analysis = result.analyze_topology(adjacency=True)
        assert analysis.n_adjacency_pairs > 0

    def test_adjacency_fraction_in_range(self, gdf):
        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        analysis = result.analyze_topology(adjacency=True)
        frac = analysis.adjacency_fraction
        assert frac is not None
        assert 0.0 <= frac <= 1.0

    def test_analyze_topology_returns_repr(self, gdf):
        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        analysis = result.analyze_topology()
        assert "TopologyAnalysis" in repr(analysis)


# ---------------------------------------------------------------------------
# 5. Topology repair
# ---------------------------------------------------------------------------


class TestTopologyRepair:
    """repair_topology() should produce a valid report with a repaired cartogram."""

    def test_repair_runs(self, gdf):
        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        report = result.repair_topology(adjacency=True)
        assert report is not None

    def test_repaired_cartogram_length(self, gdf):
        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        report = result.repair_topology(adjacency=True)
        assert len(report.cartogram.cells) == len(gdf)

    def test_adjacency_not_worse(self, gdf):
        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        report = result.repair_topology(adjacency=True)
        assert report.after.n_violated_adjacency <= report.before.n_violated_adjacency

    def test_stages_run_recorded(self, gdf):
        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        report = result.repair_topology(adjacency=True, orientation=False)
        assert "adjacency" in report.stages_run


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    def test_uniform_weights(self, gdf):
        """Uniform weights should behave identically to no weights."""
        uniform = np.ones(len(gdf))
        result = create_voronoi_cartogram(gdf, weights=uniform, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        assert len(result.cells) == len(gdf)

    def test_extreme_weight_ratio(self):
        """Very unequal weights should still converge without error."""
        gdf = make_grid_gdf(2, 2)
        weights = np.array([1.0, 1.0, 1.0, 1000.0])
        result = create_voronoi_cartogram(gdf, weights=weights, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        assert len(result.cells) == 4

    def test_weight_column_str(self, gdf):
        result = create_voronoi_cartogram(gdf, weights="population", backend=_FAST_RASTER, options=_FAST_OPTIONS)
        assert len(result.cells) == len(gdf)

    def test_positions_inside_boundary(self, gdf):
        """Final generator positions should lie inside (or on) the boundary."""
        import shapely

        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        boundary = gdf.union_all()
        pts = shapely.points(result.positions)
        inside = shapely.within(pts, boundary.buffer(1e-6))
        assert inside.all(), "Some generator positions fell outside the boundary"


# ---------------------------------------------------------------------------
# 7. Result API
# ---------------------------------------------------------------------------


class TestResultAPI:
    """VoronoiCartogram result object methods."""

    def test_to_geodataframe(self, gdf):
        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        gdf_out = result.to_geodataframe()
        assert isinstance(gdf_out, gpd.GeoDataFrame)
        assert len(gdf_out) == len(gdf)
        # area_error_pct is always appended by to_geodataframe()
        expected_cols = [*list(gdf.columns), "area_error_pct"]
        assert list(gdf_out.columns) == expected_cols

    def test_to_geodataframe_preserves_index(self, gdf):
        gdf_idx = gdf.copy()
        gdf_idx.index = list("abcdefghi")
        result = create_voronoi_cartogram(gdf_idx, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        gdf_out = result.to_geodataframe()
        assert list(gdf_out.index) == list(gdf_idx.index)

    def test_plot_smoke(self, gdf):
        import matplotlib

        matplotlib.use("Agg")
        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        pr = result.plot()
        assert pr.ax is not None
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plot_with_column(self, gdf):
        import matplotlib

        matplotlib.use("Agg")
        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        pr = result.plot(column="population")
        assert pr.ax is not None
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_repr(self, gdf):
        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        r = repr(result)
        assert "VoronoiCartogram" in r


# ---------------------------------------------------------------------------
# 8. Visualization module
# ---------------------------------------------------------------------------


class TestVisualization:
    """Standalone visualization functions."""

    def setup_method(self):
        import matplotlib

        matplotlib.use("Agg")

    def teardown_method(self):
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plot_cartogram(self, gdf):
        from carto_flow.voronoi_cartogram.visualization import plot_cartogram

        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        pr = plot_cartogram(result)
        assert pr.ax is not None

    def test_plot_comparison(self, gdf):
        from carto_flow.voronoi_cartogram.visualization import plot_comparison

        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        pr = plot_comparison(gdf, result)
        assert pr.ax is not None

    def test_plot_convergence(self, gdf):
        from carto_flow.voronoi_cartogram.visualization import plot_convergence

        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        pr = plot_convergence(result)
        assert pr.ax is not None

    def test_plot_displacement(self, gdf):
        from carto_flow.voronoi_cartogram.visualization import plot_displacement

        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        pr = plot_displacement(result)
        assert pr.ax is not None

    def test_plot_topology(self, gdf):
        from carto_flow.voronoi_cartogram.visualization import plot_topology

        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        analysis = result.analyze_topology()
        pr = plot_topology(analysis, result)
        assert pr.ax is not None

    def test_plot_topology_repair(self, gdf):
        from carto_flow.voronoi_cartogram.visualization import plot_topology_repair

        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        report = result.repair_topology(adjacency=True)
        pr = plot_topology_repair(report)
        assert pr.ax is not None


# ---------------------------------------------------------------------------
# 9. History recording
# ---------------------------------------------------------------------------


class TestHistory:
    """VoronoiOptions(record_history=True) should populate result.history."""

    def test_history_recorded(self, gdf):
        options = VoronoiOptions(n_iter=5, record_history=True)
        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=options)
        assert result.history is not None
        assert len(list(result.history)) == 5

    def test_history_snapshot_fields(self, gdf):
        from carto_flow.voronoi_cartogram import VoronoiSnapshot

        options = VoronoiOptions(n_iter=3, record_history=True)
        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=options)
        for snap in result.history:
            assert isinstance(snap, VoronoiSnapshot)
            assert snap.positions.shape == (len(gdf), 2)
            assert snap.area_cv >= 0.0

    def test_history_interval(self, gdf):
        options = VoronoiOptions(n_iter=10, record_history=2)
        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=options)
        assert result.history is not None
        assert len(list(result.history)) == 5  # every 2 of 10

    def test_record_cells(self, gdf):
        options = VoronoiOptions(n_iter=3, record_history=True, record_cells=True)
        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=options)
        for snap in result.history:
            assert snap.cells is not None
            assert len(snap.cells) == len(gdf)

    def test_no_history_by_default(self, gdf):
        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        assert result.history is None


# ---------------------------------------------------------------------------
# 10. Animation
# ---------------------------------------------------------------------------


class TestAnimation:
    """animate_voronoi_history() smoke tests."""

    def test_animate_requires_history(self, gdf):
        from carto_flow.voronoi_cartogram.animation import animate_voronoi_history

        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        with pytest.raises(RuntimeError, match="record_history"):
            animate_voronoi_history(result)

    def test_animate_positions_only(self, gdf):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.animation import FuncAnimation

        from carto_flow.voronoi_cartogram.animation import animate_voronoi_history

        options = VoronoiOptions(n_iter=4, record_history=True)
        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=options)
        anim = animate_voronoi_history(result, show_cells=False)
        assert isinstance(anim, FuncAnimation)
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_animate_with_cells(self, gdf):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.animation import FuncAnimation

        from carto_flow.voronoi_cartogram.animation import animate_voronoi_history

        options = VoronoiOptions(n_iter=3, record_history=True, record_cells=True)
        result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=options)
        anim = animate_voronoi_history(result, show_cells=True)
        assert isinstance(anim, FuncAnimation)
        import matplotlib.pyplot as plt

        plt.close("all")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# 12. Degenerate cells
# ---------------------------------------------------------------------------


class TestDegenerateCells:
    """Cells that collapse to a Point or a line must not leak downstream."""

    def test_line_clip_result_becomes_a_point_placeholder(self):
        """A label region that clips to lines only must not survive as a line.

        A line-only cell used to be kept as the cell geometry and then fed to
        ``shapely.coverage_simplify``, which raised and silently disabled
        boundary smoothing for every cell of that run.
        """
        import warnings

        from carto_flow.voronoi_cartogram.fields._raster import RasterField

        points = np.array([[0.5, 2.0], [2.5, 2.0]])
        field = RasterField(points, box(0, 0, 4, 4), resolution=4)
        # Boundary with a notch exactly over the pixel column labelled 1, so
        # that column's clip result is a MultiLineString (the notch walls).
        notched = box(0, 0, 4, 4).difference(box(1, 0, 2, 4))
        label_2d = np.zeros((4, 4), dtype=np.int32)
        label_2d[:, 1] = 1
        coords = np.array([0.5, 1.5, 2.5, 3.5])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cells = field._label_2d_to_cell_polys(
                label_2d,
                nx=4,
                ny=4,
                dx=1.0,
                dy=1.0,
                x_coords=coords,
                y_coords=coords,
                boundary=notched,
            )

        assert [c.geom_type for c in cells] == ["MultiPolygon", "Point"]
        assert not any("coverage_simplify" in str(w.message) for w in caught)

    def test_power_offset_floor_keeps_every_cell_non_empty(self):
        """A far-below-neighbour power offset must be raised, not left empty."""
        from carto_flow.voronoi_cartogram.fields._raster import RasterField

        points = np.array([[1.0, 2.0], [3.0, 2.0]])
        field = RasterField(points, box(0, 0, 4, 4), resolution=8, area_eq_weight=0.1)
        # Seed 0 sits 2 units from seed 1, so an offset gap larger than d^2 = 4
        # makes seed 0's power cell empty.
        field._power_offsets[:] = [-10.0, 0.0]
        raised = field._nonempty_offsets()
        assert raised[0] >= raised[1] - 4.0
        assert raised[1] == 0.0
        # Own position now wins: |p0 - p0|^2 - lam0 <= |p0 - p1|^2 - lam1
        assert -raised[0] <= 4.0 - raised[1]

    def test_degenerate_cells_are_reported(self):
        """The API surfaces degenerate cells instead of silently dropping them."""
        import warnings

        from shapely.geometry import Point

        from carto_flow.voronoi_cartogram.result import VoronoiCartogram

        gdf = make_grid_gdf(2, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = create_voronoi_cartogram(gdf, backend=_FAST_EXACT, options=_FAST_OPTIONS)
        assert result.degenerate_cells == []

        cells = result.cells.copy()
        cells[1] = Point(result.positions[1])
        degenerate = VoronoiCartogram(
            positions=result.positions,
            cells=cells,
            metrics=result.metrics,
            options=result.options,
            _source_gdf=gdf,
        )
        assert degenerate.degenerate_cells == [gdf.index[1]]
        analysis = degenerate.analyze_topology()
        assert analysis.degenerate_cells == [gdf.index[1]]
        assert "degenerate cells" in repr(analysis)

    def test_create_warns_about_degenerate_cells(self, monkeypatch):
        """`create_voronoi_cartogram` warns when a cell has collapsed."""
        import warnings

        from shapely.geometry import Point

        import carto_flow.voronoi_cartogram.api as api

        gdf = make_grid_gdf(2, 2)
        real = api.RasterBackend.build_field

        def patched(self, *a, **kw):
            fld = real(self, *a, **kw)
            get_cells = fld.get_cells

            def degenerate_cells():
                cells = get_cells()
                cells[0] = Point(fld.get_points()[0])
                return cells

            fld.get_cells = degenerate_cells
            return fld

        monkeypatch.setattr(api.RasterBackend, "build_field", patched)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = create_voronoi_cartogram(gdf, backend=_FAST_RASTER, options=_FAST_OPTIONS)
        assert result.degenerate_cells == [gdf.index[0]]
        assert any("collapsed to a point" in str(w.message) for w in caught)

    def test_power_offset_guard_reduces_degenerate_cells_on_us_districts(self):
        """Regression test for the reproduction case of the investigation.

        US congressional districts simplified at 5000 m, grouped by state, at a
        coarse raster resolution: several cells used to collapse to a Point.
        """
        import warnings

        import carto_flow.data as examples
        import carto_flow.voronoi_cartogram.fields._raster as raster
        from carto_flow.geo_utils.simplification import simplify_coverage

        districts = examples.load_us_census(population=True, level="congressional_district")
        districts = simplify_coverage(districts, tolerance=5000, min_island_size=50000)

        def run():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return create_voronoi_cartogram(
                    districts,
                    backend=RasterBackend(resolution=64),
                    options=VoronoiOptions(n_iter=30, area_cv_tol=0.1),
                    group_by="State Name",
                )

        guarded = run()
        try:
            raster.ENSURE_NONEMPTY_POWER_CELLS = False
            unguarded = run()
        finally:
            raster.ENSURE_NONEMPTY_POWER_CELLS = True

        assert len(guarded.degenerate_cells) < len(unguarded.degenerate_cells)
        # No cell may be a line geometry, at any resolution.
        assert all(c.geom_type in ("Polygon", "MultiPolygon", "Point") for c in guarded.cells)
        assert guarded.metrics["mean_area_error_pct"] <= unguarded.metrics["mean_area_error_pct"]


class TestSliverHoleBoundary:
    """A boundary carrying degenerate rings must not eat whole cells.

    Unioning a polygonal coverage leaves near-collinear "spike" rings with an
    area at the noise floor (1e-11 .. 1e-6 m^2 for metre coordinates).  GEOS
    overlay collapses an ordinary cell that contains such a ring to a zero-area
    LineString, which used to surface as a Point cell (US congressional
    districts, resolution 64: Michigan CD-12, Florida CD-19) or as a zero-area
    Polygon (Missouri CD-7).
    """

    # A real spike ring taken from the union of the simplified US districts.
    SPIKE: ClassVar[list[tuple[float, float]]] = [
        (717583.32787564, 142029.80746155),
        (716681.98648644, 151562.87415980),
        (717132.65718104, 146796.34081068),
    ]

    def _grid(self):
        """A 4x4 label grid with a staircase-shaped cell 1 around the spike."""
        dx = dy = 17000.0
        x_coords = np.array([700000.0 + dx * (i + 0.5) for i in range(4)])
        y_coords = np.array([120000.0 + dy * (j + 0.5) for j in range(4)])
        label_2d = np.zeros((4, 4), dtype=np.int32)
        label_2d[1:3, 1:3] = 1
        label_2d[3, 2] = 1
        points = np.array([[x_coords[0], y_coords[0]], [x_coords[2], y_coords[2]]])
        return label_2d, points, x_coords, y_coords, dx, dy

    def test_spike_ring_does_not_collapse_a_cell(self):
        from shapely.geometry import Polygon

        from carto_flow.voronoi_cartogram.fields._raster import RasterField

        label_2d, points, x_coords, y_coords, dx, dy = self._grid()
        outer = box(700000.0, 120000.0, 700000.0 + 4 * dx, 120000.0 + 4 * dy)
        boundary = Polygon(outer.exterior, [Polygon(self.SPIKE).exterior])
        assert boundary.is_valid

        field = RasterField(points, boundary, resolution=4)
        cells = field._label_2d_to_cell_polys(
            label_2d,
            nx=4,
            ny=4,
            dx=dx,
            dy=dy,
            x_coords=x_coords,
            y_coords=y_coords,
            boundary=boundary,
        )
        # Staircase smoothing (coverage_simplify) moves area between the two
        # cells, so compare generously per cell but exactly over the coverage.
        for i in (0, 1):
            n_px = int((label_2d == i).sum())
            assert cells[i].geom_type in ("Polygon", "MultiPolygon"), cells[i].geom_type
            assert cells[i].area > 0.25 * n_px * dx * dy
        assert sum(c.area for c in cells) == pytest.approx(boundary.area, rel=1e-9)

    def test_field_strips_sliver_rings_from_the_boundary(self):
        from shapely.geometry import Polygon

        from carto_flow.voronoi_cartogram.fields._base import drop_sliver_holes
        from carto_flow.voronoi_cartogram.fields._raster import RasterField

        outer = box(700000.0, 120000.0, 768000.0, 188000.0)
        real_hole = box(740000.0, 140000.0, 745000.0, 145000.0)
        boundary = Polygon(outer.exterior, [Polygon(self.SPIKE).exterior, real_hole.exterior])

        cleaned = drop_sliver_holes(boundary)
        assert len(cleaned.interiors) == 1
        assert cleaned.area == pytest.approx(boundary.area, rel=1e-12)

        field = RasterField(np.array([[710000.0, 130000.0]]), boundary, resolution=4)
        assert len(field.boundary.interiors) == 1
        assert len(field._current_boundary.interiors) == 1

    def test_failed_clip_falls_back_to_the_pixel_union(self, monkeypatch):
        """A collapsing clip must warn and keep the cell's area, not return a Point."""
        import warnings

        import shapely as sh

        from carto_flow.voronoi_cartogram.fields._raster import RasterField

        label_2d, points, x_coords, y_coords, dx, dy = self._grid()
        boundary = box(700000.0, 120000.0, 700000.0 + 4 * dx, 120000.0 + 4 * dy)
        field = RasterField(points, boundary, resolution=4)

        real_intersection = sh.intersection

        def collapsing_intersection(a, b, *args, **kwargs):
            result = real_intersection(a, b, *args, **kwargs)
            # Collapse only the staircase cell (the smaller of the two).
            if getattr(result, "geom_type", "") == "Polygon" and result.area < 6 * dx * dy:
                return sh.LineString([(700000.0, 120000.0), (710000.0, 130000.0)])
            return result

        monkeypatch.setattr(sh, "intersection", collapsing_intersection)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cells = field._label_2d_to_cell_polys(
                label_2d,
                nx=4,
                ny=4,
                dx=dx,
                dy=dy,
                x_coords=x_coords,
                y_coords=y_coords,
                boundary=boundary,
            )

        assert cells[1].geom_type in ("Polygon", "MultiPolygon")
        assert cells[1].area > 0.25 * 5 * dx * dy
        messages = [str(w.message) for w in caught]
        assert any("extraction failed for seed 1" in m for m in messages), messages

    def test_near_zero_area_polygon_counts_as_degenerate(self):
        """A polygon with a sliver area is as degenerate as a Point."""
        from shapely.geometry import Polygon

        from carto_flow.voronoi_cartogram.result import VoronoiCartogram

        gdf = make_grid_gdf(2, 2)
        cells = np.array(
            [box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2), Polygon([(0, 0), (1, 0), (1e-12, 1e-12)])],
            dtype=object,
        )
        result = VoronoiCartogram(
            positions=np.zeros((4, 2)),
            cells=cells,
            metrics={},
            options=VoronoiOptions(),
            _source_gdf=gdf,
        )
        assert result.degenerate_cells == [gdf.index[3]]
