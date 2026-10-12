import pytest

from carto_flow.data import load_us_census
from carto_flow.flow_cartogram import MorphOptions, create_flow_cartogram


@pytest.fixture(scope="session")
def us_states_gdf():
    """48 contiguous US states with population data (Albers equal-area, metres)."""
    return load_us_census(population=True, contiguous_only=True)


@pytest.fixture(scope="session")
def us_congressional_districts_gdf():
    """~430 contiguous US congressional districts with population data.

    Geometries are simplified at 1 km tolerance to keep benchmark runtimes
    reasonable while still exercising the density/displacement paths with a
    realistic high-geometry-count dataset.
    """
    return load_us_census(level="congressional_district", population=True, contiguous_only=True, simplify=1000)


@pytest.fixture(scope="session", autouse=True)
def warmup_numba(us_states_gdf):
    """Trigger Numba JIT compilation before any benchmark runs.

    Numba compiles kernels on first call; without this warm-up the first
    benchmark case would absorb compilation time and skew the comparison.
    A single iteration on a tiny grid is enough to compile all hot paths.
    """
    create_flow_cartogram(
        us_states_gdf,
        "Population",
        options=MorphOptions(grid=64, n_iter=2, show_progress=False, recompute_every=1),
    )
    # Also warm up the serial displacement JIT variant (compiled separately from parallel)
    create_flow_cartogram(
        us_states_gdf,
        "Population",
        options=MorphOptions(grid=64, n_iter=2, show_progress=False, recompute_every=1),
    )
