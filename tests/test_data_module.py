"""
Test the data module with optional dependencies.

This test file verifies that the data module:
1. Can be imported without optional dependencies
2. Raises appropriate errors when optional dependencies are missing
3. Works correctly when optional dependencies are installed
"""

import pytest


def test_data_module_import():
    """Test that the data module can be imported."""
    import carto_flow.data

    assert hasattr(carto_flow.data, "load_world")
    assert hasattr(carto_flow.data, "load_us_states")
    assert hasattr(carto_flow.data, "load_sample_cities")


def test_load_us_census_without_optional_deps(monkeypatch):
    """Test that load_us_census raises ImportError when censusdis is missing."""
    import carto_flow.data

    # Simulate censusdis not being installed
    def mock_check(*args):
        raise ImportError(
            "The 'censusdis' package is required for loading US Census data, but it is not installed.\n"
            "You can install it with:\n"
            "pip install carto-flow[data]  # to install all optional data dependencies\n"
            "or\n"
            "pip install censusdis  # to install just this package"
        )

    monkeypatch.setattr(carto_flow.data, "_check_optional_dependency", mock_check)

    with pytest.raises(ImportError) as excinfo:
        carto_flow.data.load_us_census()
    assert "pip install carto-flow[data]" in str(excinfo.value)


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
