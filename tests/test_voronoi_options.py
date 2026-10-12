"""Option behavior of the Voronoi cartogram: internal settings are deprecated, regular options have their documented effect."""

from __future__ import annotations

import warnings

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from carto_flow import data
from carto_flow.voronoi_cartogram import (
    AdhesiveBoundary,
    ElasticBoundary,
    ExactBackend,
    RasterBackend,
    VoronoiOptions,
    create_voronoi_cartogram,
)


@pytest.fixture(scope="module")
def states():
    return data.load_us_census(population=True).reset_index(drop=True)


def _grid():
    geoms = [box(c, r, c + 1, r + 1) for r in range(3) for c in range(3)]
    return gpd.GeoDataFrame({"population": np.arange(1.0, 10.0)}, geometry=geoms)


class TestInternalSettingsAreDeprecated:
    @pytest.mark.parametrize(
        "factory",
        [
            lambda: RasterBackend(relaxation="lloyd"),
            lambda: RasterBackend(area_equalizer_rate=0.2),
            lambda: RasterBackend(weight_ramp_iters=5),
            lambda: RasterBackend(output_resolution=200),
            lambda: RasterBackend(cell_smoothing_px=1.0),
            lambda: ExactBackend(relaxation=1.0),
            lambda: ElasticBoundary(step_scale=0.5),
            lambda: ElasticBoundary(density_smooth=1.0),
            lambda: ElasticBoundary(min_boundary_points=100),
            lambda: VoronoiOptions(adj_min_shared_length=0.1),
        ],
        ids=[
            "relaxation",
            "area_equalizer_rate",
            "weight_ramp_iters",
            "output_resolution",
            "cell_smoothing_px",
            "exact relaxation",
            "step_scale",
            "density_smooth",
            "min_boundary_points",
            "adj_min_shared_length",
        ],
    )
    def test_a_non_default_value_warns(self, factory):
        with pytest.warns(DeprecationWarning, match="internal tuning setting"):
            factory()

    def test_the_defaults_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            RasterBackend()
            ExactBackend()
            ElasticBoundary()
            VoronoiOptions()
            RasterBackend(relaxation="overrelax", area_equalizer_rate=0.1, weight_ramp_iters=10, cell_smoothing_px=3.0)

    def test_adhesive_boundary_is_deprecated(self):
        with pytest.warns(DeprecationWarning, match="AdhesiveBoundary is deprecated"):
            AdhesiveBoundary(strength=0.3)

    def test_adhesion_strength_is_deprecated(self):
        with pytest.warns(DeprecationWarning, match="adhesion_strength is deprecated"):
            ElasticBoundary(adhesion_strength=0.2)

    def test_a_deprecated_setting_still_works(self):
        with pytest.warns(DeprecationWarning):
            backend = RasterBackend(resolution=40, relaxation="lloyd")
        result = create_voronoi_cartogram(_grid(), backend=backend, options=VoronoiOptions(n_iter=3))
        assert len(result.to_geodataframe()) == 9


class TestFixTopologyWithWeights:
    def test_warns_with_weights(self):
        with pytest.warns(UserWarning, match="suits unweighted runs"):
            create_voronoi_cartogram(
                _grid(),
                weights="population",
                backend=RasterBackend(resolution=40),
                options=VoronoiOptions(n_iter=3, fix_topology=2),
            )

    def test_does_not_warn_without_weights(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            create_voronoi_cartogram(
                _grid(), backend=RasterBackend(resolution=40), options=VoronoiOptions(n_iter=3, fix_topology=2)
            )


def _adjacency_fraction(result):
    analysis = result.analyze_topology()
    return 1.0 - analysis.n_violated_adjacency / analysis.n_adjacency_pairs


class TestRegularOptions:
    def test_adjacency_spring_keeps_adjacent_regions_adjacent(self, states):
        options = VoronoiOptions(n_iter=60)
        plain = create_voronoi_cartogram(states, backend=RasterBackend(resolution=150), options=options)
        spring = create_voronoi_cartogram(
            states, backend=RasterBackend(resolution=150, adjacency_spring=0.6), options=options
        )
        assert _adjacency_fraction(spring) > _adjacency_fraction(plain) + 0.05

    def test_intra_group_spring_keeps_groups_whole(self, states):
        options = VoronoiOptions(n_iter=60)
        counts = []
        for spring in (None, 0.5):
            backend = RasterBackend(resolution=100, adjacency_spring=0.1, intra_group_spring=spring)
            result = create_voronoi_cartogram(states, group_by="Division", backend=backend, options=options)
            counts.append(result.analyze_topology(group_by="Division").n_discontiguous_groups)
        assert counts[1] < counts[0]

    def test_elastic_boundary_changes_the_outline(self, states):
        options = VoronoiOptions(n_iter=40)
        rigid = create_voronoi_cartogram(states, backend=RasterBackend(resolution=120), options=options)
        elastic = create_voronoi_cartogram(
            states, backend=RasterBackend(resolution=120, boundary=ElasticBoundary(strength=0.05)), options=options
        )
        area = states.geometry.union_all().area
        assert rigid.to_geodataframe().geometry.union_all().area == pytest.approx(area, rel=5e-3)
        assert abs(elastic.to_geodataframe().geometry.union_all().area / area - 1) > 0.01

    @pytest.mark.parametrize(("spec", "roundness"), [("bbox", 0.7), ("convex_hull", 0.8), ("circle", 0.95)])
    def test_boundary_spec_sets_the_outline(self, states, spec, roundness):
        result = create_voronoi_cartogram(
            states, boundary=spec, backend=RasterBackend(resolution=100), options=VoronoiOptions(n_iter=20)
        )
        union = states.geometry.union_all()
        cells = result.to_geodataframe().geometry.union_all()
        # the outline is a bbox, hull or circle around the input: it covers the input and is much rounder than it
        assert cells.intersection(union).area > 0.97 * union.area
        assert 4 * np.pi * cells.area / cells.length**2 > roundness
        assert cells.area > 1.1 * union.area

    def test_simplify_tol_reduces_the_outline_vertices(self, states):
        def vertices(tol):
            result = create_voronoi_cartogram(
                states, backend=RasterBackend(resolution=120), options=VoronoiOptions(n_iter=10, simplify_tol=tol)
            )
            union = result.to_geodataframe().geometry.union_all()
            return sum(len(p.exterior.coords) for p in (union.geoms if hasattr(union, "geoms") else [union]))

        assert vertices(100_000.0) < 0.5 * vertices(None)
