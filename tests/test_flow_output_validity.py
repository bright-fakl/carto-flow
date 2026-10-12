"""Invalid (self-intersecting) output regions of the flow cartogram and of partition_geometries."""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon

from carto_flow import data
from carto_flow.flow_cartogram import MorphOptions, morph_gdf
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


class TestMorphOutput:
    def test_the_output_is_valid_by_default(self, minorities):
        gdf, _ = minorities
        result = morph_gdf(gdf, "Minorities", options=MorphOptions(grid_size=512, show_progress=False))
        assert len(invalid_indices(list(result.get_geometry()))) == 0
        assert len(result.invalid_regions) >= 1
        assert shapely.is_valid(result.to_geodataframe().geometry.values).all()

    def test_without_repair_the_output_is_returned_as_computed_with_a_warning(self, minorities):
        gdf, _ = minorities
        options = MorphOptions(grid_size=512, show_progress=False, repair_output=False)
        with pytest.warns(UserWarning, match="not valid polygons"):
            result = morph_gdf(gdf, "Minorities", options=options)
        invalid = invalid_indices(list(result.get_geometry()))
        assert len(invalid) >= 1
        assert tuple(invalid) == result.invalid_regions

    def test_the_repair_changes_the_areas_little(self, minorities):
        gdf, _ = minorities
        raw = morph_gdf(
            gdf, "Minorities", options=MorphOptions(grid_size=512, show_progress=False, repair_output=False)
        )
        repaired = morph_gdf(gdf, "Minorities", options=MorphOptions(grid_size=512, show_progress=False))
        a = np.array([g.area for g in raw.get_geometry()])
        b = np.array([g.area for g in repaired.get_geometry()])
        assert np.max(np.abs(b / a - 1)) < 3e-3

    def test_valid_runs_have_no_invalid_regions(self):
        states = data.load_us_census(population=True).reset_index(drop=True)
        result = morph_gdf(states, "Population", options=MorphOptions(grid_size=128, show_progress=False))
        assert result.invalid_regions == ()

    def test_the_option_is_validated(self):
        with pytest.raises(ValueError, match="repair_output"):
            MorphOptions(repair_output="yes")


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
