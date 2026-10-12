"""Invalid (self-intersecting) output regions of the flow cartogram and of partition_geometries."""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon

from carto_flow import data
from carto_flow.flow_cartogram import MorphOptions, create_flow_cartogram
from carto_flow.geo_utils.validity import invalid_indices, repair_polygons
from carto_flow.proportional_cartogram import partition_geometries

BOWTIE = Polygon([(0, 0), (4, 4), (4, 0), (0, 4)])
SQUARE = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
# a square with a pinched lobe of zero area at the top edge
PINCHED = Polygon([(0, 0), (0, 10), (4, 10), (4, 12), (4, 10), (10, 10), (10, 0)])


class TestRepairPolygons:
    def test_valid_geometries_are_returned_unchanged(self):
        geometries = [SQUARE, MultiPolygon([SQUARE, Polygon([(5, 5), (6, 5), (6, 6)])])]
        assert len(invalid_indices(geometries)) == 0
        repaired = repair_polygons(geometries)
        assert all(a is b for a, b in zip(geometries, repaired, strict=True))

    def test_a_bowtie_becomes_two_valid_triangles(self):
        assert list(invalid_indices([SQUARE, BOWTIE])) == [1]
        repaired = repair_polygons([SQUARE, BOWTIE])
        assert repaired[0] is SQUARE
        assert shapely.is_valid(repaired[1])
        assert isinstance(repaired[1], MultiPolygon)
        assert repaired[1].area == pytest.approx(8.0)

    def test_a_pinched_lobe_of_zero_area_is_dropped(self):
        assert not PINCHED.is_valid
        (repaired,) = repair_polygons([PINCHED])
        assert shapely.is_valid(repaired)
        assert isinstance(repaired, Polygon)
        assert repaired.area == pytest.approx(100.0)

    def test_only_the_given_indices_are_repaired(self):
        repaired = repair_polygons([BOWTIE, BOWTIE], indices=[1])
        assert not shapely.is_valid(repaired[0])
        assert shapely.is_valid(repaired[1])


@pytest.fixture(scope="module")
def minorities():
    """The map of issue 85: Maryland ends invalid when the minority population is mapped at grid 512."""
    gdf = data.load_us_census(population=True, race=True, simplify=1000)
    columns = ["Black or African American", "Asian", "Hispanic or Latino"]
    gdf["Minorities"] = gdf[columns].sum(axis=1)
    return gdf, columns


def _vertex_counts(geometries):
    return [len(shapely.get_coordinates(g)) for g in geometries]


class TestMorphOutput:
    @pytest.fixture(scope="class")
    def result(self, minorities):
        gdf, _ = minorities
        options = MorphOptions(grid_size=512, show_progress=False, snapshot_every=1)
        with pytest.warns(UserWarning, match="not valid polygons"):
            return create_flow_cartogram(gdf, "Minorities", options=options)

    def test_the_morph_warns_and_lists_the_invalid_regions(self, result):
        assert len(result.invalid_regions) >= 1
        assert tuple(invalid_indices(list(result.get_geometry()))) == result.invalid_regions

    def test_the_default_keeps_the_vertices_of_the_input(self, minorities, result):
        """Keyframe animations interpolate vertex by vertex, so the geometries keep the vertices of the input."""
        gdf, _ = minorities
        counts = _vertex_counts(gdf.geometry)
        assert _vertex_counts(result.get_geometry()) == counts
        assert _vertex_counts(result.snapshots[0].geometry) == counts
        assert _vertex_counts(result.to_geodataframe().geometry) == counts

    def test_repair_makes_the_geometries_valid(self, result):
        assert shapely.is_valid(result.get_geometry(repair=True)).all()
        assert shapely.is_valid(result.to_geodataframe(repair=True).geometry.values).all()
        # the stored geometries are not changed
        assert len(invalid_indices(list(result.get_geometry()))) == len(result.invalid_regions)

    def test_the_repair_changes_the_areas_little(self, result):
        a = np.array([g.area for g in result.get_geometry()])
        b = np.array([g.area for g in result.get_geometry(repair=True)])
        assert np.max(np.abs(b / a - 1)) < 3e-3

    def test_valid_runs_have_no_invalid_regions_and_do_not_warn(self, recwarn):
        states = data.load_us_census(population=True).reset_index(drop=True)
        result = create_flow_cartogram(states, "Population", options=MorphOptions(grid_size=128, show_progress=False))
        assert result.invalid_regions == ()
        assert not [w for w in recwarn if "not valid polygons" in str(w.message)]
        assert result.get_geometry(repair=True) is not None


class TestPartition:
    def test_an_invalid_polygon_is_repaired_with_a_warning(self):
        import geopandas as gpd

        gdf = gpd.GeoDataFrame(
            {"a": [0.3, 0.5], "b": [0.7, 0.5]},
            geometry=[PINCHED, SQUARE],
        )
        with pytest.warns(UserWarning, match="not valid polygons"):
            parts = partition_geometries(gdf, ["a", "b"], method="split", normalization="row")
        shares = [parts["geometry_a"].iloc[0].area / 100.0, parts["geometry_b"].iloc[0].area / 100.0]
        assert shares == pytest.approx([0.3, 0.7], abs=0.02)
