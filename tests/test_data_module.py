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

    for level in ("state", "congressional_district"):
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


def test_load_us_census_bad_vintage_raises():
    import carto_flow.data

    with pytest.raises(ValueError, match="vintage"):
        carto_flow.data.load_us_census(vintage=2019)


def test_load_us_census_simplify_below_bundled_tolerance_raises():
    import carto_flow.data

    with pytest.raises(ValueError, match="bundled"):
        carto_flow.data.load_us_census(simplify=500)


def test_load_us_census_simplify_above_bundled_tolerance_works():
    import carto_flow.data

    gdf = carto_flow.data.load_us_census(simplify=5000)
    assert len(gdf) > 0


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
