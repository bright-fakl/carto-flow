"""
Test the data module with optional dependencies.

This test file verifies that the data module:
1. Can be imported without optional dependencies
2. Loads the bundled US Census snapshot without network access or censusdis
3. Works correctly when optional dependencies are installed
"""

import sys

import pytest


def test_data_module_import():
    """Test that the data module can be imported."""
    import carto_flow.data

    assert hasattr(carto_flow.data, "load_world")
    assert hasattr(carto_flow.data, "load_us_states")
    assert hasattr(carto_flow.data, "load_sample_cities")
    assert hasattr(carto_flow.data, "load_us_census")


def test_load_us_census_without_censusdis(monkeypatch):
    """load_us_census reads the bundled parquet snapshot; it must not need
    censusdis, a network connection, or an API key at runtime."""
    import carto_flow.data

    # Simulate censusdis not being installed / importable at all.
    monkeypatch.setitem(sys.modules, "censusdis", None)
    monkeypatch.setitem(sys.modules, "censusdis.data", None)

    for level in ("state", "congressional_district", "county"):
        for population, race, poverty in (
            (True, False, False),
            (False, True, False),
            (False, False, True),
            (True, True, True),
        ):
            for contiguous_only in (True, False):
                gdf = carto_flow.data.load_us_census(
                    population=population,
                    race=race,
                    poverty=poverty,
                    level=level,
                    contiguous_only=contiguous_only,
                )
                assert len(gdf) > 0
                assert "geometry" in gdf.columns
                if population:
                    assert "Population" in gdf.columns
                if race:
                    assert "White %" in gdf.columns
                if poverty:
                    assert "Below Poverty Level %" in gdf.columns


def test_load_us_census_simplify_above_bundled_tolerance_works():
    import carto_flow.data

    gdf = carto_flow.data.load_us_census(simplify=5000)
    assert len(gdf) > 0


def test_load_us_census_live_fallback_without_censusdis_raises_import_error(monkeypatch):
    """A request outside the bundled snapshot (vintage != 2020) tries a live
    download; if censusdis isn't importable, that must fail with a clear
    ImportError pointing at the 'data' extra, not silently or with some
    other error."""
    import carto_flow.data

    monkeypatch.setitem(sys.modules, "censusdis", None)
    monkeypatch.setitem(sys.modules, "censusdis.data", None)

    with pytest.raises(ImportError, match=r"carto-flow\[data\]"):
        carto_flow.data.load_us_census(vintage=2019)


def test_load_us_census_live_fallback_without_key_raises_before_network(monkeypatch, tmp_path):
    """With censusdis importable but no API key discoverable, load_us_census
    must raise with key instructions and must not attempt a network call."""
    pytest.importorskip("censusdis")
    import carto_flow.data

    monkeypatch.delenv("US_CENSUS_API_KEY", raising=False)
    monkeypatch.setattr(carto_flow.data.Path, "home", lambda: tmp_path)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("censusdis.data.download should not be called without an API key")

    import censusdis.data as ced

    monkeypatch.setattr(ced, "download", _fail_if_called)

    with pytest.raises(RuntimeError, match="API key"):
        carto_flow.data.load_us_census(vintage=2019)


def test_load_world_with_optional_deps():
    """Test that load_world works when dependencies are installed."""
    import carto_flow.data

    try:
        gdf = carto_flow.data.load_world()
        assert gdf is not None
        assert hasattr(gdf, "shape")
        assert gdf.shape[0] > 0
        assert "geometry" in gdf.columns
        assert "pop_est" in gdf.columns
    except ImportError as e:
        pytest.skip(f"Optional dependencies not installed: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")


def test_load_us_states():
    """Test that load_us_states works (with optional dependencies)."""
    import carto_flow.data

    try:
        gdf = carto_flow.data.load_us_states()
        assert gdf is not None
        assert gdf.shape[0] > 0
        assert "geometry" in gdf.columns
    except ImportError as e:
        pytest.skip(f"Optional dependencies not installed: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")


def test_load_sample_cities():
    """Test that load_sample_cities works (with optional dependencies)."""
    import carto_flow.data

    try:
        gdf = carto_flow.data.load_sample_cities()
        assert gdf is not None
        assert gdf.shape[0] > 0
        assert "geometry" in gdf.columns
    except ImportError as e:
        pytest.skip(f"Optional dependencies not installed: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")


@pytest.mark.parametrize("level", ["state", "congressional_district", "county"])
def test_load_us_census_race_groups_do_not_exceed_total(level):
    """The four race/ethnicity groups are disjoint, so they sum to at most Total Race."""
    import carto_flow.data

    gdf = carto_flow.data.load_us_census(race=True, level=level, contiguous_only=False)
    groups = gdf[["White", "Black or African American", "Asian", "Hispanic or Latino"]].sum(axis=1)
    assert (groups <= gdf["Total Race"]).all()


def test_load_us_census_hispanic_or_latino_is_total():
    """Hispanic or Latino holds ACS B03002_012E (all races), not the White-alone subset."""
    import carto_flow.data

    gdf = carto_flow.data.load_us_census(race=True, contiguous_only=False).set_index("State Name")
    # ACS 2020 5-year, B03002_012E
    assert gdf.loc["California", "Hispanic or Latino"] == pytest.approx(15_380_929, rel=0.005)
    assert gdf.loc["Texas", "Hispanic or Latino"] == pytest.approx(11_294_257, rel=0.005)

    states = gdf.drop(index="Puerto Rico")
    share = states["Hispanic or Latino"].sum() / states["Total Race"].sum()
    assert share == pytest.approx(0.18, abs=0.01)


def test_load_us_census_county_level():
    import carto_flow.data

    gdf = carto_flow.data.load_us_census(race=True, poverty=True, level="county")
    assert len(gdf) == 3108
    assert gdf["COUNTY"].notna().all()
    assert {"County Name", "State Name", "State Abbreviation", "Region", "Division", "Population"} <= set(gdf.columns)
    assert (gdf["Population"] > 0).all()
    assert not gdf["State Abbreviation"].isin(["AK", "HI", "PR"]).any()
    assert "DC" in set(gdf["State Abbreviation"])
    assert gdf.crs.to_string() == "ESRI:102008"
    assert gdf.loc[gdf["County Name"].str.startswith("Maricopa"), "Population"].iloc[0] > 4_000_000
    assert len(carto_flow.data.load_us_census(level="county", contiguous_only=False)) == 3221


def _longest_segment(gdf):
    import numpy as np
    import shapely

    return max(
        np.hypot(*np.diff(np.asarray(ring.coords), axis=0).T).max()
        for geom in gdf.geometry
        for part in shapely.get_parts(geom)
        for ring in (part.exterior, *part.interiors)
    )


@pytest.mark.parametrize("level", ["state", "congressional_district", "county"])
def test_load_us_census_max_segment_length_densifies(level):
    import shapely

    import carto_flow.data as data

    base = data.load_us_census(level=level)
    fine = data.load_us_census(level=level, max_segment_length=1000)
    assert _longest_segment(base) <= 5000
    assert _longest_segment(fine) <= 1000
    assert len(fine) == len(base)
    assert list(fine.columns) == list(base.columns)
    assert fine.crs == base.crs
    assert sum(shapely.get_num_coordinates(g) for g in fine.geometry) > sum(
        shapely.get_num_coordinates(g) for g in base.geometry
    )
    # the two sides of a shared boundary stay identical, so the coverage has no gaps or overlaps
    assert fine.geometry.is_valid.all()
    assert fine.geometry.union_all().area == pytest.approx(base.geometry.union_all().area, rel=1e-9)


def test_load_us_census_max_segment_length_at_least_the_source_changes_nothing():
    import carto_flow.data as data

    base = data.load_us_census(level="county")
    same = data.load_us_census(level="county", max_segment_length=5000)
    assert same.geometry.geom_equals_exact(base.geometry, 0).all()


@pytest.mark.parametrize("value", [0, -5, True, "x"])
def test_load_us_census_rejects_a_bad_max_segment_length(value):
    import carto_flow.data as data

    with pytest.raises(ValueError, match="max_segment_length"):
        data.load_us_census(max_segment_length=value)


@pytest.mark.parametrize("loader_name", ["load_world", "load_us_states"])
def test_geographic_loaders_densify_in_degrees(loader_name):
    import numpy as np

    import carto_flow.data as data

    loader = getattr(data, loader_name)
    base = loader()
    fine = loader(max_segment_length=0.1)
    assert base.crs.to_string() == "EPSG:4326"
    assert _longest_segment(base) > 0.5
    assert _longest_segment(fine) <= 0.1 + 1e-9
    assert len(fine) == len(base) and list(fine.columns) == list(base.columns)
    assert fine.geometry.is_valid.all()
    assert np.isclose(fine.geometry.union_all().area, base.geometry.union_all().area, rtol=1e-9)
    assert loader(max_segment_length=1000).geometry.geom_equals_exact(base.geometry, 0).all()


@pytest.mark.parametrize("loader_name", ["load_world", "load_us_states"])
@pytest.mark.parametrize("value", [0, -1, True, "x"])
def test_geographic_loaders_reject_a_bad_max_segment_length(loader_name, value):
    import carto_flow.data as data

    with pytest.raises(ValueError, match="max_segment_length"):
        getattr(data, loader_name)(max_segment_length=value)
