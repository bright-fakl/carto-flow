"""Tests for the input check of the cartogram functions."""

import warnings

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from carto_flow import create_flow_cartogram
from carto_flow.flow_cartogram import MorphOptions
from carto_flow.geo_utils import validate_geometries
from carto_flow.noncontiguous_cartogram import create_noncontiguous_cartogram
from carto_flow.voronoi_cartogram import create_voronoi_cartogram

BOWTIE = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])


def grid(n=3, **columns):
    geoms = [box(i, j, i + 1, j + 1) for j in range(n) for i in range(n)]
    gdf = gpd.GeoDataFrame({"v": np.arange(1.0, n * n + 1)}, geometry=geoms, index=[f"r{k}" for k in range(n * n)])
    for name, values in columns.items():
        gdf[name] = values
    return gdf


def kinds(report):
    return {i.kind for i in report.issues}


def test_clean_input_has_no_issue():
    report = validate_geometries(grid(), "v", coverage=True)
    assert report.ok
    assert report.issues == ()


def test_invalid_geometry_names_row_and_reason():
    gdf = grid()
    gdf.loc["r4", "geometry"] = BOWTIE
    report = validate_geometries(gdf, "v")
    assert not report.ok
    (issue,) = report.errors
    assert issue.kind == "invalid"
    assert issue.rows == ("r4",)
    assert "Self-intersection" in issue.message
    assert "make_valid" in issue.message


def test_null_and_empty_geometries_are_errors():
    gdf = grid()
    gdf.loc["r0", "geometry"] = None
    gdf.loc["r1", "geometry"] = Polygon()
    (issue,) = validate_geometries(gdf).errors
    assert issue.kind == "empty"
    assert set(issue.rows) == {"r0", "r1"}


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0, 0.0])
def test_bad_values_are_errors(bad):
    gdf = grid()
    gdf.loc["r2", "v"] = bad
    (issue,) = validate_geometries(gdf, "v").errors
    assert issue.kind == "values"
    assert issue.rows == ("r2",)


def test_zero_value_can_be_allowed():
    gdf = grid()
    gdf.loc["r2", "v"] = 0.0
    assert validate_geometries(gdf, "v", zero="ok").ok


def test_missing_column_is_an_error():
    with pytest.raises(ValueError, match="not found"):
        validate_geometries(grid(), "nope").raise_for_errors()


def test_non_numeric_values_are_an_error():
    gdf = grid(w=list("abcdefghi"))
    assert kinds(validate_geometries(gdf, "w")) == {"values"}


def test_repeated_vertices_are_information_only():
    gdf = grid()
    gdf.loc["r3", "geometry"] = Polygon([(0, 1), (0, 1), (1, 1), (1, 2), (0, 2)])
    report = validate_geometries(gdf)
    assert report.ok
    assert report.warnings == ()
    (issue,) = report.info
    assert issue.kind == "repeated_vertices"
    assert issue.rows == ("r3",)


def test_overlap_is_a_warning_only_with_coverage():
    gdf = grid()
    gdf.loc["r0", "geometry"] = box(0, 0, 1.5, 1)
    assert validate_geometries(gdf).issues == ()
    report = validate_geometries(gdf, coverage=True)
    assert report.ok
    assert kinds(report) == {"coverage"}


def test_message_counts_rows_beyond_the_listed_ones():
    gdf = grid(n=4)
    gdf["v"] = -1.0
    (issue,) = validate_geometries(gdf, "v").errors
    assert "and 11 more" in issue.message


def test_raise_for_errors_and_warn():
    gdf = grid()
    gdf.loc["r4", "geometry"] = BOWTIE
    with pytest.raises(ValueError, match="r4"):
        validate_geometries(gdf).raise_for_errors()
    gdf = grid()
    gdf.loc["r0", "geometry"] = box(0, 0, 1.5, 1)
    with pytest.warns(UserWarning, match="overlap"):
        validate_geometries(gdf, coverage=True).warn()


FUNCTIONS = {
    "flow": lambda g, **kw: create_flow_cartogram(
        g, "v", options=MorphOptions(grid=32, n_iter=2, show_progress=False), **kw
    ),
    "voronoi": lambda g, **kw: create_voronoi_cartogram(g, weights="v", **kw),
    "noncontiguous": lambda g, **kw: create_noncontiguous_cartogram(g, "v", **kw),
}


@pytest.mark.parametrize("name", FUNCTIONS)
def test_every_function_names_the_invalid_row(name):
    gdf = grid()
    gdf.loc["r4", "geometry"] = BOWTIE
    with pytest.raises(ValueError, match="r4"):
        FUNCTIONS[name](gdf)


@pytest.mark.parametrize("name", ["flow", "voronoi"])
def test_every_function_rejects_nan_values(name):
    gdf = grid()
    gdf.loc["r1", "v"] = np.nan
    with pytest.raises(ValueError):
        FUNCTIONS[name](gdf)


def test_zero_value_is_accepted():
    gdf = grid()
    gdf.loc["r1", "v"] = 0.0
    for name in ("flow", "noncontiguous"):
        FUNCTIONS[name](gdf)


def test_validate_false_skips_the_check():
    gdf = grid()
    gdf.loc["r4", "geometry"] = BOWTIE
    with pytest.raises(ValueError):
        FUNCTIONS["noncontiguous"](gdf)
    FUNCTIONS["noncontiguous"](gdf, validate=False)


def test_bundled_states_give_no_warning():
    from carto_flow.data import load_us_states

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        create_noncontiguous_cartogram(load_us_states().to_crs(5070), "scalerank")


def test_clean_input_gives_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        FUNCTIONS["noncontiguous"](grid())
