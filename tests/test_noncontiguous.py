"""Tests for the non-contiguous (Olson) cartogram."""

import geopandas as gpd
import matplotlib
import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

matplotlib.use("Agg")

from carto_flow.noncontiguous_cartogram import NoncontiguousCartogram, create_noncontiguous_cartogram


def noncontiguous_cartogram(gdf, column, **kwargs):
    return create_noncontiguous_cartogram(gdf, column, **kwargs).geometries


@pytest.fixture
def squares():
    return gpd.GeoDataFrame(
        {"value": [4.0, 1.0, 9.0], "name": list("abc")},
        geometry=[box(0, 0, 2, 2), box(3, 0, 5, 2), box(6, 0, 8, 2)],
        crs="EPSG:3857",
    )


def _area_ratio(result, original):
    return result.geometry.area.to_numpy() / original.geometry.area.to_numpy()


class TestScaling:
    def test_areas_are_proportional_to_the_values(self, squares):
        result = noncontiguous_cartogram(squares, "value")
        assert result.geometry.area.to_numpy() / squares["value"].to_numpy() == pytest.approx(
            [result.geometry.area.iloc[0] / 4.0] * 3
        )

    def test_densest_region_keeps_its_size_and_nothing_grows(self, squares):
        result = noncontiguous_cartogram(squares, "value")
        ratio = _area_ratio(result, squares)
        assert ratio.max() == pytest.approx(1.0)
        assert (ratio <= 1 + 1e-9).all()
        assert ratio[2] == pytest.approx(1.0)  # 9 on area 4 is the densest

    def test_regions_scale_about_their_centroids(self, squares):
        result = noncontiguous_cartogram(squares, "value")
        for before, after in zip(squares.geometry, result.geometry, strict=True):
            assert after.centroid.x == pytest.approx(before.centroid.x)
            assert after.centroid.y == pytest.approx(before.centroid.y)

    def test_shapes_are_kept(self, squares):
        result = noncontiguous_cartogram(squares, "value")
        for after in result.geometry:
            minx, miny, maxx, maxy = after.bounds
            assert maxx - minx == pytest.approx(maxy - miny)

    def test_scale_column_and_other_columns(self, squares):
        result = noncontiguous_cartogram(squares, "value")
        assert result["_ncont_scale"].to_numpy() == pytest.approx(np.sqrt(_area_ratio(result, squares)))
        assert list(result["name"]) == ["a", "b", "c"]
        assert result.crs == squares.crs
        assert "_ncont_scale" not in squares.columns  # the input is not changed

    def test_float_reference_density(self, squares):
        result = noncontiguous_cartogram(squares, "value", reference=1.0, max_scale=None)
        assert result.geometry.area.to_numpy() == pytest.approx(squares["value"].to_numpy())

    def test_smaller_density_makes_regions_larger(self, squares):
        small = noncontiguous_cartogram(squares, "value", reference=4.0, max_scale=None)
        large = noncontiguous_cartogram(squares, "value", reference=1.0, max_scale=None)
        assert (large.geometry.area.to_numpy() > small.geometry.area.to_numpy()).all()


class TestGrowth:
    def test_largest_reference_grows_denser_regions_without_a_cap(self, squares):
        result = noncontiguous_cartogram(squares, "value", reference="largest", max_scale=None)
        ratio = _area_ratio(result, squares)
        # the densities are 1, 0.25 and 2.25: the largest value (9) is also the densest, so nothing grows
        assert ratio[2] == pytest.approx(1.0)
        assert ratio[0] == pytest.approx(1 / 2.25)
        assert (ratio <= 1 + 1e-9).all()

    @pytest.fixture
    def dense_small(self):
        # the largest value (100) is spread over 100 units of area; the second region has half of it on 4 units
        return gpd.GeoDataFrame(
            {"value": [100.0, 50.0]},
            geometry=[box(0, 0, 10, 10), box(20, 0, 22, 2)],
            crs="EPSG:3857",
        )

    def test_cap_warns_and_keeps_regions_at_the_cap(self, dense_small):
        with pytest.warns(UserWarning, match="max_scale"):
            cartogram = create_noncontiguous_cartogram(dense_small, "value", reference="largest")
        assert list(cartogram.capped) == [False, True]
        assert cartogram.scale.max() == pytest.approx(1.0)
        assert cartogram.geometries.geometry.iloc[1].area == pytest.approx(4.0)
        assert cartogram.area_error_pct[1] < 0  # smaller than proportional
        assert cartogram.area_error_pct[0] == pytest.approx(0.0)

    def test_growth_allowed(self, dense_small):
        cartogram = create_noncontiguous_cartogram(dense_small, "value", reference="largest", max_scale=None)
        assert cartogram.geometries.geometry.iloc[1].area == pytest.approx(50.0)  # 12.5 times its area
        assert not cartogram.capped.any()
        assert cartogram.area_error_pct == pytest.approx([0.0, 0.0])

    def test_a_larger_cap(self, dense_small):
        with pytest.warns(UserWarning, match="max_scale"):
            cartogram = create_noncontiguous_cartogram(dense_small, "value", reference="largest", max_scale=2.0)
        assert cartogram.scale[1] == pytest.approx(2.0)


class TestMultipart:
    @pytest.fixture
    def island(self):
        main = box(0, 0, 4, 4)
        far = box(20, 0, 22, 2)
        return gpd.GeoDataFrame(
            {"value": [1.0, 4.0]},
            geometry=[MultiPolygon([main, far]), box(40, 0, 44, 4)],
            crs="EPSG:3857",
        )

    def test_group_scales_the_region_as_one_shape(self, island):
        result = noncontiguous_cartogram(island, "value", multipart="group")
        before, after = island.geometry.iloc[0], result.geometry.iloc[0]
        factor = result["_ncont_scale"].iloc[0]
        assert after.centroid.x == pytest.approx(before.centroid.x)
        assert after.area == pytest.approx(before.area * factor**2)
        # the distance between the parts shrinks with the region
        d_before = before.geoms[0].centroid.distance(before.geoms[1].centroid)
        d_after = after.geoms[0].centroid.distance(after.geoms[1].centroid)
        assert d_after == pytest.approx(d_before * factor)

    def test_parts_scale_about_their_own_centroids(self, island):
        result = noncontiguous_cartogram(island, "value", multipart="parts")
        before, after = island.geometry.iloc[0], result.geometry.iloc[0]
        for old, new in zip(before.geoms, after.geoms, strict=True):
            assert new.centroid.x == pytest.approx(old.centroid.x)
            assert new.centroid.y == pytest.approx(old.centroid.y)
        assert after.area == pytest.approx(before.area * result["_ncont_scale"].iloc[0] ** 2)

    def test_polygons_are_the_same_either_way(self, squares):
        group = noncontiguous_cartogram(squares, "value", multipart="group")
        parts = noncontiguous_cartogram(squares, "value", multipart="parts")
        assert all(a.equals_exact(b, 1e-9) for a, b in zip(group.geometry, parts.geometry, strict=True))

    def test_holes_survive(self):
        ring = Polygon([(0, 0), (6, 0), (6, 6), (0, 6)], holes=[[(2, 2), (4, 2), (4, 4), (2, 4)]])
        gdf = gpd.GeoDataFrame({"value": [1.0, 4.0]}, geometry=[ring, box(10, 0, 14, 4)], crs="EPSG:3857")
        result = noncontiguous_cartogram(gdf, "value")
        assert len(result.geometry.iloc[0].interiors) == 1


class TestValues:
    def test_zero_value_gives_an_empty_geometry(self, squares):
        squares = squares.copy()
        squares.loc[1, "value"] = 0.0
        result = noncontiguous_cartogram(squares, "value")
        assert result.geometry.iloc[1].is_empty
        assert result["_ncont_scale"].iloc[1] == 0.0
        assert not result.geometry.iloc[0].is_empty

    @pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
    def test_invalid_values(self, squares, bad):
        squares = squares.copy()
        squares.loc[0, "value"] = bad
        with pytest.raises(ValueError, match="non-negative"):
            noncontiguous_cartogram(squares, "value")

    def test_all_zero(self, squares):
        squares = squares.copy()
        squares["value"] = 0.0
        with pytest.raises(ValueError, match="positive value"):
            noncontiguous_cartogram(squares, "value")

    def test_text_column(self, squares):
        with pytest.raises(ValueError, match="numbers"):
            noncontiguous_cartogram(squares, "name")

    def test_missing_column(self, squares):
        with pytest.raises(ValueError, match="not in"):
            noncontiguous_cartogram(squares, "nope")


class TestArguments:
    def test_invalid_arguments(self, squares):
        with pytest.raises(ValueError, match="multipart"):
            noncontiguous_cartogram(squares, "value", multipart="all")
        with pytest.raises(ValueError, match="reference"):
            noncontiguous_cartogram(squares, "value", reference="smallest")
        with pytest.raises(ValueError, match="positive"):
            noncontiguous_cartogram(squares, "value", reference=0.0)
        with pytest.raises(ValueError, match="max_scale"):
            noncontiguous_cartogram(squares, "value", max_scale=0.0)

    def test_geographic_crs_warns(self, squares):
        geographic = squares.set_crs("EPSG:4326", allow_override=True)
        with pytest.warns(UserWarning, match="geographic"):
            noncontiguous_cartogram(geographic, "value")

    def test_no_crs_does_not_warn(self, squares):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            noncontiguous_cartogram(squares.set_crs(None, allow_override=True), "value")

    def test_us_states(self):
        import carto_flow.data as data

        states = data.load_us_census(population=True)
        result = noncontiguous_cartogram(states, "Population")
        ratio = _area_ratio(result, states)
        assert ratio.max() == pytest.approx(1.0)
        with pytest.warns(UserWarning, match="max_scale"):
            capped = noncontiguous_cartogram(states, "Population", reference="largest")
        assert capped["_ncont_scale"].max() == pytest.approx(1.0)
        grown = noncontiguous_cartogram(states, "Population", reference="largest", max_scale=None)
        assert (_area_ratio(grown, states) > 1).sum() == 11


class TestResult:
    def test_result_fields(self, squares):
        cartogram = create_noncontiguous_cartogram(squares, "value")
        assert isinstance(cartogram, NoncontiguousCartogram)
        assert cartogram.column == "value"
        assert cartogram.reference_density == pytest.approx(9.0 / 4.0)
        assert cartogram.multipart == "group"
        assert cartogram.source is squares
        assert cartogram.to_geodataframe().equals(cartogram.geometries)
        assert cartogram.to_geodataframe() is not cartogram.geometries

    def test_area_error_is_zero_without_a_cap(self, squares):
        cartogram = create_noncontiguous_cartogram(squares, "value")
        assert cartogram.area_error_pct == pytest.approx([0.0, 0.0, 0.0])

    def test_zero_value_has_no_area_error(self, squares):
        squares = squares.copy()
        squares.loc[1, "value"] = 0.0
        cartogram = create_noncontiguous_cartogram(squares, "value")
        assert np.isnan(cartogram.area_error_pct[1])


class TestPlot:
    @pytest.fixture(autouse=True)
    def close(self):
        yield
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plot_draws_originals_and_regions(self, squares):
        plot = create_noncontiguous_cartogram(squares, "value").plot()
        assert len(plot.original_collections) == 1
        assert len(plot.collections) == 1

    def test_show_original_false(self, squares):
        plot = create_noncontiguous_cartogram(squares, "value").plot(show_original=False)
        assert plot.original_collections == []

    def test_data_and_colors(self, squares):
        cartogram = create_noncontiguous_cartogram(squares, "value")
        assert cartogram.plot(facecolor="value").collections
        categories = cartogram.plot(facecolor="name")
        assert categories.collections
        literal = cartogram.plot(facecolor="#ff0000", style="classic")
        assert literal.collections[0].get_facecolor()[0][:3] == pytest.approx((1.0, 0.0, 0.0))
        assert cartogram.plot(facecolor=np.array([1.0, 2.0, 3.0])).collections
        assert cartogram.plot(facecolor="_ncont_scale", legend=False).collections

    def test_style_fill_value_and_palette(self, squares):
        from carto_flow import Channel, Palettes, PlotStyle

        cartogram = create_noncontiguous_cartogram(squares, "value")
        style = PlotStyle(channels={"fill": Channel(value="#123456")}, palettes=Palettes(sequential="Reds"))
        plain = cartogram.plot(style=style)
        assert matplotlib.colors.to_hex(plain.collections[0].get_facecolor()[0]) == "#123456"
        reds = cartogram.plot(facecolor="value", style=style, legend=False)
        explicit = cartogram.plot(facecolor="value", cmap="Reds", style="classic", legend=False)
        assert np.allclose(reds.collections[0].get_facecolor(), explicit.collections[0].get_facecolor())

    def test_zero_value_region_is_not_drawn(self, squares):
        squares = squares.copy()
        squares.loc[1, "value"] = 0.0
        cartogram = create_noncontiguous_cartogram(squares, "value")
        assert cartogram.plot(facecolor="value").collections
        assert cartogram.plot(facecolor="name").collections

    def test_plot_on_axes(self, squares):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        assert create_noncontiguous_cartogram(squares, "value").plot(ax=ax).ax is ax


class TestReferenceRegion:
    @pytest.fixture
    def three(self):
        # densities 9, 3 and 1: the first region is an outlier
        return gpd.GeoDataFrame(
            {"value": [9.0, 12.0, 4.0]},
            geometry=[box(0, 0, 1, 1), box(3, 0, 5, 2), box(7, 0, 9, 2)],
            crs="EPSG:3857",
            index=["a", "b", "c"],
        )

    def test_region_keeps_its_size(self, three):
        with pytest.warns(UserWarning, match="max_scale"):
            cartogram = create_noncontiguous_cartogram(three, "value", reference_region="b")
        assert cartogram.reference_density == pytest.approx(3.0)
        assert cartogram.scale[1] == pytest.approx(1.0)
        assert cartogram.scale[2] == pytest.approx(np.sqrt(1.0 / 3.0))
        assert list(cartogram.capped) == [True, False, False]
        assert cartogram.scale[0] == pytest.approx(1.0)

    def test_outlier_grows_without_a_cap(self, three):
        cartogram = create_noncontiguous_cartogram(three, "value", reference_region="b", max_scale=None)
        assert cartogram.scale[0] == pytest.approx(np.sqrt(3.0))
        assert not cartogram.capped.any()
        assert cartogram.area_error_pct == pytest.approx([0.0, 0.0, 0.0])

    def test_equals_the_density_as_a_number(self, three):
        by_region = create_noncontiguous_cartogram(three, "value", reference_region="b", max_scale=None)
        by_number = create_noncontiguous_cartogram(three, "value", reference=3.0, max_scale=None)
        assert by_region.scale == pytest.approx(by_number.scale)

    def test_densest_region_as_reference_is_the_default(self, three):
        by_region = create_noncontiguous_cartogram(three, "value", reference_region="a")
        default = create_noncontiguous_cartogram(three, "value")
        assert by_region.scale == pytest.approx(default.scale)

    def test_errors(self, three):
        with pytest.raises(ValueError, match="not both"):
            create_noncontiguous_cartogram(three, "value", reference="largest", reference_region="a")
        with pytest.raises(ValueError, match="not in the index"):
            create_noncontiguous_cartogram(three, "value", reference_region="z")
        zero = three.copy()
        zero.loc["b", "value"] = 0.0
        with pytest.raises(ValueError, match="no positive value"):
            create_noncontiguous_cartogram(zero, "value", reference_region="b")
        duplicated = three.copy()
        duplicated.index = ["a", "b", "b"]
        with pytest.raises(ValueError, match="unique"):
            create_noncontiguous_cartogram(duplicated, "value", reference_region="b")

    def test_integer_labels(self, squares):
        cartogram = create_noncontiguous_cartogram(squares, "value", reference_region=0)
        assert cartogram.reference_density == pytest.approx(1.0)


class TestDrawingOrder:
    def test_larger_regions_are_drawn_first(self, squares):
        import matplotlib.pyplot as plt

        cartogram = create_noncontiguous_cartogram(squares, "value")
        plot = cartogram.plot(facecolor="value", show_original=False, legend=False, style="classic")
        order_drawn = list(plot.collections[0].get_paths())
        areas = [Polygon(path.vertices).area for path in order_drawn]
        assert areas == sorted(areas, reverse=True)
        plt.close("all")

    def test_colors_follow_their_regions(self, squares):
        import matplotlib.pyplot as plt

        cartogram = create_noncontiguous_cartogram(squares, "value")
        colors = np.array(["#ff0000", "#00ff00", "#0000ff"])
        plot = cartogram.plot(facecolor=colors, show_original=False, style="classic")
        collection = plot.collections[0]
        drawn = [Polygon(path.vertices) for path in collection.get_paths()]
        for polygon, face in zip(drawn, collection.get_facecolor(), strict=True):
            index = next(
                i for i, g in enumerate(cartogram.geometries.geometry) if polygon.centroid.distance(g.centroid) < 1e-6
            )
            assert matplotlib.colors.to_hex(face) == colors[index]
        plt.close("all")

    def test_data_values_follow_their_regions(self, squares):
        import matplotlib.pyplot as plt

        cartogram = create_noncontiguous_cartogram(squares, "value")
        plot = cartogram.plot(facecolor="value", show_original=False, legend=False, style="classic")
        collection = plot.collections[0]
        values = np.asarray(collection.get_array())
        drawn = [Polygon(path.vertices) for path in collection.get_paths()]
        for polygon, value in zip(drawn, values, strict=True):
            index = next(
                i for i, g in enumerate(cartogram.geometries.geometry) if polygon.centroid.distance(g.centroid) < 1e-6
            )
            assert value == squares["value"].iloc[index]
        plt.close("all")
