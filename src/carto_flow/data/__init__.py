"""
Data loading utilities for example datasets used in documentation and examples.

This module provides convenience functions to load example datasets without
requiring users to manually download and manage data files. The datasets are
either included with the package or downloaded on-demand from reliable sources.

Optional Dependencies
---------------------
This module has optional dependencies that are not required for the core functionality
of carto-flow. These dependencies include:
- geodatasets: For accessing example datasets from the geodatasets library
- censusdis: For accessing US census data (optional, for demographic examples)

If these dependencies are not installed, functions that require them will raise
a clear ImportError with instructions on how to install the missing packages.
"""

import importlib.metadata
from importlib.resources import files
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import geopandas


def _check_optional_dependency(package_name: str, purpose: str) -> None:
    """Check if an optional dependency is installed."""
    try:
        importlib.metadata.distribution(package_name)
    except importlib.metadata.PackageNotFoundError:
        raise ImportError(
            f"The '{package_name}' package is required for {purpose}, but it is not installed.\n"
            "You can install it with:\n"
            f"pip install carto-flow[data]  # to install all optional data dependencies\n"
            f"or\n"
            f"pip install {package_name}  # to install just this package"
        ) from None


def _import_optional_module(module_name: str, package_name: str, purpose: str):
    """Import an optional module with proper error handling."""
    _check_optional_dependency(package_name, purpose)
    try:
        return __import__(module_name, fromlist=[""])
    except ImportError as e:
        raise ImportError(f"Failed to import '{module_name}' module for {purpose}: {e}") from e


def load_world() -> "geopandas.GeoDataFrame":
    """
    Load a world countries dataset with population estimates.

    This dataset includes country boundaries and population estimates from
    Natural Earth (public domain data).

    Returns
    -------
    geopandas.GeoDataFrame
        World countries dataset with geometry and population information.
        Columns:
        - name: Country name
        - continent: Continent name
        - pop_est: Population estimate
        - gdp_md_est: GDP estimate (millions USD)
        - geometry: Country boundaries

    Examples
    --------
    >>> from carto_flow.data import load_world
    >>> gdf = load_world()
    >>> print(gdf.shape)
    (177, 5)
    """
    import geopandas as gpd

    path = files("carto_flow.data").joinpath("world.geojson")
    return gpd.read_file(path)


def load_us_states() -> "geopandas.GeoDataFrame":
    """
    Load a US states dataset with population and area information.

    This dataset includes state boundaries, population estimates, and area
    information from the US Census Bureau.

    Returns
    -------
    geopandas.GeoDataFrame
        US states dataset with geometry and demographic information.

    Examples
    --------
    >>> from carto_flow.data import load_us_states
    >>> gdf = load_us_states()
    >>> print(gdf.shape)
    (52, ...)
    """
    try:
        gpd = _import_optional_module("geopandas", "geopandas", "loading geographic data")
        geodatasets = _import_optional_module("geodatasets", "geodatasets", "loading US states dataset")
        return gpd.read_file(geodatasets.get_path("naturalearth_us_states"))
    except Exception as e:
        raise ImportError(
            "Failed to load US states dataset. Please ensure you have the required dependencies installed.\n"
            "You can install them with:\n"
            "pip install carto-flow[data]"
        ) from e


def load_sample_cities() -> "geopandas.GeoDataFrame":
    """
    Load a sample dataset of cities with population information.

    This dataset includes major cities around the world with population estimates.

    Returns
    -------
    geopandas.GeoDataFrame
        Cities dataset with geometry and population information.

    Examples
    --------
    >>> from carto_flow.data import load_sample_cities
    >>> gdf = load_sample_cities()
    >>> print(gdf.shape)
    (200, ...)
    """
    try:
        gpd = _import_optional_module("geopandas", "geopandas", "loading geographic data")
        geodatasets = _import_optional_module("geodatasets", "geodatasets", "loading cities dataset")
        return gpd.read_file(geodatasets.get_path("naturalearth_cities"))
    except Exception as e:
        raise ImportError(
            "Failed to load cities dataset. Please ensure you have the required dependencies installed.\n"
            "You can install them with:\n"
            "pip install carto-flow[data]"
        ) from e


def load_us_census(
    population: bool = True,
    race: bool = False,
    poverty: bool = False,
    simplify: float | None = None,
    vintage: int = 2020,
    contiguous_only: bool = True,
) -> "geopandas.GeoDataFrame":
    """
    Load US state boundaries with ACS demographic data from the US Census Bureau.

    Downloads American Community Survey (ACS) 5-year estimates for US states,
    projected to the Albers equal-area projection (ESRI:102008). Alaska, Hawaii,
    and Puerto Rico are excluded.

    Parameters
    ----------
    population : bool, default True
        Include total population. Adds columns:
        - ``Population``, ``Population (Millions)``, ``Population Density``
    race : bool, default False
        Include race/ethnicity breakdown. Adds columns:
        - ``Total Race``, ``White``, ``Black or African American``, ``Asian``,
          ``Hispanic or Latino``, and corresponding ``<group> %`` columns.
    poverty : bool, default False
        Include poverty status. Adds columns:
        - ``Total Poverty``, ``Below Poverty Level``, ``Above Poverty Level``,
          ``Below Poverty Level %``, ``Above Poverty Level %``
    simplify : float or None, default None
        If given, simplify geometries using this tolerance in meters (units of
        ESRI:102008). A value of 1000 gives a good balance of detail vs. speed.
        ``None`` keeps full-resolution geometries.
    vintage : int, default 2020
        ACS 5-year vintage year.

    Returns
    -------
    geopandas.GeoDataFrame
        US states dataset in ESRI:102008 (Albers equal-area, meters).

    Examples
    --------
    >>> from carto_flow.data import load_us_census
    >>> gdf = load_us_census(population=True, race=True, simplify=1000)
    >>> print(gdf["State Name"].head())
    """
    _check_optional_dependency("censusdis", "loading US Census data")
    import censusdis
    import censusdis.data as ced

    variables: dict[str, str] = {"NAME": "State Name"}
    if population:
        variables |= {"B01003_001E": "Population"}
    if poverty:
        variables |= {
            "B16009_001E": "Total Poverty",
            "B16009_002E": "Below Poverty Level",
            "B16009_015E": "Above Poverty Level",
        }
    if race:
        variables |= {
            "B03002_001E": "Total Race",
            "B03002_003E": "White",
            "B03002_004E": "Black or African American",
            "B03002_006E": "Asian",
            "B03002_013E": "Hispanic or Latino",
        }

    gdf = ced.download("acs/acs5", vintage, list(variables.keys()), state="*", with_geometry=True)
    gdf.rename(columns=variables, inplace=True)
    gdf = gdf.to_crs("ESRI:102008")

    if contiguous_only:
        gdf = gdf[~gdf["State Name"].isin(["Alaska", "Hawaii", "Puerto Rico"])]

    gdf["State Abbreviation"] = gdf["STATE"].map(censusdis.states.ABBREVIATIONS_FROM_IDS)

    if simplify is not None:
        gdf.geometry = gdf.geometry.simplify(simplify, preserve_topology=True)

    if population:
        gdf["Population (Millions)"] = gdf["Population"] / 1e6
        gdf["Population Density"] = gdf["Population"] / (gdf.area / 1e6)
    if poverty:
        gdf["Below Poverty Level %"] = gdf["Below Poverty Level"] / gdf["Total Poverty"]
        gdf["Above Poverty Level %"] = gdf["Above Poverty Level"] / gdf["Total Poverty"]
    if race:
        for key in ("White", "Black or African American", "Asian", "Hispanic or Latino"):
            gdf[f"{key} %"] = gdf[key] / gdf["Total Race"]

    return gdf


__all__ = [
    "load_sample_cities",
    "load_us_census",
    "load_us_states",
    "load_world",
]
