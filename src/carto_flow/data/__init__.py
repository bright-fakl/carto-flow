"""
Data loading utilities for example datasets used in documentation and examples.

This module provides convenience functions to load example datasets without
requiring users to manually download and manage data files. The datasets are
either included with the package or downloaded on-demand from reliable sources.

Census Data
-----------
``load_us_census`` reads a bundled snapshot (ACS 2020, GeoParquet) shipped
inside the package for its default resolution, so that path needs no network
access, API key, or optional dependency at runtime. Requests outside the
bundled snapshot (a vintage other than 2020, or a ``simplify`` tolerance
finer than 1000 m) fall back to a live download via ``censusdis`` (extra
``data``), which needs a Census API key. See ``load_us_census`` for details
and ``scripts/build_census_snapshot.py`` to regenerate the bundled snapshot.

State-Region and State-Division Mappings
---------------------------------------
The following dictionaries provide mappings between US states (identified by their
2-digit FIPS codes) and their respective census regions and divisions, as defined
by the US Census Bureau:

- `STATE_REGIONS`: Maps state FIPS codes to region codes (1-4)
- `STATE_DIVISIONS`: Maps state FIPS codes to division codes (1-9)
- `REGION_NAMES`: Maps region codes to human-readable region names
- `DIVISION_NAMES`: Maps division codes to human-readable division names
- `REGION_DIVISIONS`: Maps region codes to lists of division codes within that region
"""

import importlib.metadata
import os
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import geopandas
    import pandas

# US Census Bureau region definitions
REGION_NAMES = {"1": "Northeast", "2": "Midwest", "3": "South", "4": "West"}

DIVISION_NAMES = {
    "1": "New England",
    "2": "Middle Atlantic",
    "3": "East North Central",
    "4": "West North Central",
    "5": "South Atlantic",
    "6": "East South Central",
    "7": "West South Central",
    "8": "Mountain",
    "9": "Pacific",
}

REGION_DIVISIONS = {
    "1": ["1", "2"],  # Northeast: New England, Middle Atlantic
    "2": ["3", "4"],  # Midwest: East North Central, West North Central
    "3": ["5", "6", "7"],  # South: South Atlantic, East South Central, West South Central
    "4": ["8", "9"],  # West: Mountain, Pacific
}

# State to Region mapping (FIPS code to region code)
STATE_REGIONS = {
    "01": "3",  # Alabama: South
    "02": "4",  # Alaska: West
    "04": "4",  # Arizona: West
    "05": "3",  # Arkansas: South
    "06": "4",  # California: West
    "08": "4",  # Colorado: West
    "09": "1",  # Connecticut: Northeast
    "10": "3",  # Delaware: South
    "11": "3",  # District of Columbia: South
    "12": "3",  # Florida: South
    "13": "3",  # Georgia: South
    "15": "4",  # Hawaii: West
    "16": "4",  # Idaho: West
    "17": "2",  # Illinois: Midwest
    "18": "2",  # Indiana: Midwest
    "19": "2",  # Iowa: Midwest
    "20": "2",  # Kansas: Midwest
    "21": "3",  # Kentucky: South
    "22": "3",  # Louisiana: South
    "23": "1",  # Maine: Northeast
    "24": "3",  # Maryland: South
    "25": "1",  # Massachusetts: Northeast
    "26": "2",  # Michigan: Midwest
    "27": "2",  # Minnesota: Midwest
    "28": "3",  # Mississippi: South
    "29": "2",  # Missouri: Midwest
    "30": "4",  # Montana: West
    "31": "2",  # Nebraska: Midwest
    "32": "4",  # Nevada: West
    "33": "1",  # New Hampshire: Northeast
    "34": "1",  # New Jersey: Northeast
    "35": "4",  # New Mexico: West
    "36": "1",  # New York: Northeast
    "37": "3",  # North Carolina: South
    "38": "2",  # North Dakota: Midwest
    "39": "2",  # Ohio: Midwest
    "40": "3",  # Oklahoma: South
    "41": "4",  # Oregon: West
    "42": "1",  # Pennsylvania: Northeast
    "44": "1",  # Rhode Island: Northeast
    "45": "3",  # South Carolina: South
    "46": "2",  # South Dakota: Midwest
    "47": "3",  # Tennessee: South
    "48": "3",  # Texas: South
    "49": "4",  # Utah: West
    "50": "1",  # Vermont: Northeast
    "51": "3",  # Virginia: South
    "53": "4",  # Washington: West
    "54": "3",  # West Virginia: South
    "55": "2",  # Wisconsin: Midwest
    "56": "4",  # Wyoming: West
    "72": None,  # Puerto Rico: Not part of any census region
}

# State to Division mapping (FIPS code to division code)
STATE_DIVISIONS = {
    "01": "6",  # Alabama: East South Central
    "02": "9",  # Alaska: Pacific
    "04": "8",  # Arizona: Mountain
    "05": "6",  # Arkansas: East South Central
    "06": "9",  # California: Pacific
    "08": "8",  # Colorado: Mountain
    "09": "1",  # Connecticut: New England
    "10": "5",  # Delaware: South Atlantic
    "11": "5",  # District of Columbia: South Atlantic
    "12": "5",  # Florida: South Atlantic
    "13": "5",  # Georgia: South Atlantic
    "15": "9",  # Hawaii: Pacific
    "16": "8",  # Idaho: Mountain
    "17": "3",  # Illinois: East North Central
    "18": "3",  # Indiana: East North Central
    "19": "4",  # Iowa: West North Central
    "20": "4",  # Kansas: West North Central
    "21": "6",  # Kentucky: East South Central
    "22": "7",  # Louisiana: West South Central
    "23": "1",  # Maine: New England
    "24": "5",  # Maryland: South Atlantic
    "25": "1",  # Massachusetts: New England
    "26": "3",  # Michigan: East North Central
    "27": "4",  # Minnesota: West North Central
    "28": "6",  # Mississippi: East South Central
    "29": "4",  # Missouri: West North Central
    "30": "8",  # Montana: Mountain
    "31": "4",  # Nebraska: West North Central
    "32": "8",  # Nevada: Mountain
    "33": "1",  # New Hampshire: New England
    "34": "2",  # New Jersey: Middle Atlantic
    "35": "8",  # New Mexico: Mountain
    "36": "2",  # New York: Middle Atlantic
    "37": "5",  # North Carolina: South Atlantic
    "38": "4",  # North Dakota: West North Central
    "39": "3",  # Ohio: East North Central
    "40": "7",  # Oklahoma: West South Central
    "41": "9",  # Oregon: Pacific
    "42": "2",  # Pennsylvania: Middle Atlantic
    "44": "1",  # Rhode Island: New England
    "45": "5",  # South Carolina: South Atlantic
    "46": "4",  # South Dakota: West North Central
    "47": "6",  # Tennessee: East South Central
    "48": "7",  # Texas: West South Central
    "49": "8",  # Utah: Mountain
    "50": "1",  # Vermont: New England
    "51": "5",  # Virginia: South Atlantic
    "53": "9",  # Washington: Pacific
    "54": "5",  # West Virginia: South Atlantic
    "55": "3",  # Wisconsin: East North Central
    "56": "8",  # Wyoming: Mountain
    "72": None,  # Puerto Rico: Not part of any census division
}


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

    path = files("carto_flow.data").joinpath("world.parquet")
    return gpd.read_parquet(path)


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
    (51, ...)
    """
    import geopandas as gpd

    path = files("carto_flow.data").joinpath("us_states.parquet")
    return gpd.read_parquet(path)


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
    import geopandas as gpd

    path = files("carto_flow.data").joinpath("cities.parquet")
    return gpd.read_parquet(path)


_BUNDLED_VINTAGE = 2020
_BUNDLED_SIMPLIFY_TOLERANCE = 1000.0
_BUNDLED_DENSIFY_MAX_SEGMENT_LENGTH = 5000.0

_CENSUS_KEY_INSTRUCTIONS = (
    "Get a free key at https://api.census.gov/data/key_signup.html and store it in "
    "~/.censusdis/api_key.txt or set the US_CENSUS_API_KEY environment variable. "
    "Vintage 2020 at the bundled resolution (simplify=None or simplify >= 1000) needs no key."
)


def _census_variables(level: str, population: bool, poverty: bool, race: bool) -> dict[str, str]:
    """Map of raw ACS variable code -> friendly column name for the requested groups."""
    name_col = "State Name" if level == "state" else "District Name"
    variables: dict[str, str] = {"NAME": name_col}
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
    return variables


def _has_census_api_key() -> bool:
    if os.environ.get("US_CENSUS_API_KEY"):
        return True
    key_path = Path.home() / ".censusdis" / "api_key.txt"
    return key_path.exists() and bool(key_path.read_text().strip())


def _load_us_census_bundled(
    level: str, population: bool, poverty: bool, race: bool, simplify: float | None
) -> "geopandas.GeoDataFrame":
    """Read the bundled ACS 2020 GeoParquet snapshot (no network, no key)."""
    import geopandas as gpd

    path = files("carto_flow.data").joinpath(f"us_census_{_BUNDLED_VINTAGE}_{level}.parquet")
    gdf = gpd.read_parquet(path)

    # For state level, the bundled "State Name" column already carries the
    # human-readable name (it equals the raw NAME); renaming NAME too would
    # create a duplicate "State Name" column. For district level, NAME is
    # the district's own name and is distinct from "State Name".
    variables = _census_variables(level, population, poverty, race)
    rename = {k: v for k, v in variables.items() if level != "state" or k != "NAME"}
    gdf = gdf.rename(columns=rename)

    base_cols = ["STATE", "State Abbreviation", "State Name", "geometry"]
    if level == "congressional_district":
        base_cols.insert(1, "CONGRESSIONAL_DISTRICT")
    gdf = gdf[[*rename.values(), *base_cols]].copy()

    if simplify is not None:
        from carto_flow.geo_utils.simplification import simplify_coverage

        gdf = simplify_coverage(gdf, tolerance=simplify, max_segment_length=_BUNDLED_DENSIFY_MAX_SEGMENT_LENGTH)

    return gdf


def _load_us_census_live(
    level: str,
    vintage: int,
    population: bool,
    poverty: bool,
    race: bool,
    simplify: float | None,
) -> "geopandas.GeoDataFrame":
    """Download ACS data live via censusdis (network + API key required)."""
    _check_optional_dependency("censusdis", "downloading Census data for vintage != 2020 or simplify < 1000 m")
    try:
        import censusdis
        import censusdis.data as ced
    except ImportError as e:
        raise ImportError(
            "The 'censusdis' package is required for downloading Census data for vintage != 2020 "
            "or simplify < 1000 m, but it could not be imported.\n"
            "You can install it with:\n"
            "pip install carto-flow[data]  # to install all optional data dependencies\n"
            "or\n"
            "pip install censusdis  # to install just this package"
        ) from e

    if not _has_census_api_key():
        raise RuntimeError(f"No US Census API key found. {_CENSUS_KEY_INSTRUCTIONS}")

    variables = _census_variables(level, population, poverty, race)
    try:
        if level == "state":
            gdf = ced.download("acs/acs5", vintage, list(variables.keys()), state="*", with_geometry=True)
        else:
            gdf = ced.download(
                "acs/acs5",
                vintage,
                list(variables.keys()),
                state="*",
                congressional_district="*",
                with_geometry=True,
            )
    except Exception as e:
        raise RuntimeError(f"Census data download failed: {e}. {_CENSUS_KEY_INSTRUCTIONS}") from e

    gdf = gdf.rename(columns=variables)
    gdf = gdf.to_crs("ESRI:102008")

    gdf["State Abbreviation"] = gdf["STATE"].map(censusdis.states.ABBREVIATIONS_FROM_IDS)
    base_cols = ["STATE", "State Abbreviation", "geometry"]
    if level == "congressional_district":
        gdf["State Name"] = gdf["STATE"].map(censusdis.states.NAMES_FROM_IDS)
        base_cols += ["CONGRESSIONAL_DISTRICT", "State Name"]
    gdf = gdf[[*variables.values(), *base_cols]].copy()

    if simplify is not None:
        from carto_flow.geo_utils.simplification import simplify_coverage

        gdf = simplify_coverage(gdf, tolerance=simplify, max_segment_length=_BUNDLED_DENSIFY_MAX_SEGMENT_LENGTH)

    return gdf


def _finalize_census_gdf(
    gdf: "geopandas.GeoDataFrame",
    level: str,
    population: bool,
    poverty: bool,
    race: bool,
    contiguous_only: bool,
) -> "geopandas.GeoDataFrame":
    """Shared post-processing so the bundled and live paths return identical columns."""
    if level == "state":
        if contiguous_only:
            gdf = gdf[~gdf["State Name"].isin(["Alaska", "Hawaii", "Puerto Rico"])]
    else:
        # Remove non-geographic placeholder districts
        gdf = gdf[gdf["CONGRESSIONAL_DISTRICT"] != "ZZ"]

        if contiguous_only:
            gdf = gdf[~gdf["State Abbreviation"].isin(["AK", "HI", "PR", "DC"])]

    gdf = gdf.copy()

    gdf["Region"] = gdf["STATE"].map(lambda x: REGION_NAMES.get(STATE_REGIONS.get(x or "", ""), None))  # type: ignore[arg-type]
    gdf["Division"] = gdf["STATE"].map(lambda x: DIVISION_NAMES.get(STATE_DIVISIONS.get(x or "", ""), None))  # type: ignore[arg-type]

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


def load_us_census(
    population: bool = True,
    race: bool = False,
    poverty: bool = False,
    level: str = "state",
    simplify: float | None = None,
    vintage: int = 2020,
    contiguous_only: bool = True,
) -> "geopandas.GeoDataFrame":
    """
    Load US boundaries with ACS demographic data from the US Census Bureau.

    For the default vintage (2020) at the bundled resolution (``simplify``
    ``None`` or ``>= 1000``), this reads a bundled snapshot of the American
    Community Survey (ACS) 5-year estimates shipped inside the package as
    GeoParquet: no network access or Census API key is required. Geometries
    are bundled at 500k cartographic-boundary resolution, coverage-simplified
    at 1000 m, then densified so no straight segment exceeds 5 km (see
    ``carto_flow.geo_utils.simplify_coverage`` for why: algorithms that move
    vertices, like the flow cartogram, cannot bend a segment that has no
    interior vertices).

    For any other request — a vintage other than 2020, or a ``simplify``
    tolerance finer than the bundled 1000 m — this falls back to a live
    download via ``censusdis`` (install with ``pip install carto-flow[data]``),
    which needs a Census API key (see
    https://api.census.gov/data/key_signup.html), read from the
    ``US_CENSUS_API_KEY`` environment variable or ``~/.censusdis/api_key.txt``.

    In both cases the result is projected to the Albers equal-area
    projection (ESRI:102008) and has the same columns.

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
    level : str, default "state"
        Geographic level of detail. One of:
        - ``"state"`` — one row per US state. Adds column ``State Name``.
        - ``"congressional_district"`` — one row per congressional district.
          Adds columns ``District Name`` and ``State Name``.
    simplify : float or None, default None
        Simplification tolerance in meters (units of ESRI:102008).
        ``None`` (default) keeps the source resolution: the bundled 1000 m
        (already densified to 5 km segments) for vintage 2020, or full
        resolution for a live download. A value ``>= 1000`` with vintage
        2020 re-simplifies the bundled geometries locally; any other value
        (including ``< 1000``) triggers a live download, simplified to that
        tolerance and densified to 5 km segments.
    vintage : int, default 2020
        ACS 5-year vintage year. 2020 uses the bundled snapshot (at the
        bundled resolution); any other vintage triggers a live download.
    contiguous_only : bool, default True
        Exclude Alaska, Hawaii, Puerto Rico, and DC.

    Returns
    -------
    geopandas.GeoDataFrame
        Dataset in ESRI:102008 (Albers equal-area, meters). Both levels include
        ``State Abbreviation``, ``Region``, and ``Division`` columns.

    Raises
    ------
    ImportError
        If a live download is needed and ``censusdis`` is not installed.
    RuntimeError
        If a live download is needed and no Census API key is found, or the
        download itself fails.

    Examples
    --------
    >>> from carto_flow.data import load_us_census
    >>> gdf = load_us_census(population=True, race=True)
    >>> print(gdf["State Name"].head())

    >>> gdf_dist = load_us_census(level="congressional_district", population=True)
    >>> print(gdf_dist["District Name"].head())
    """
    if level not in ("state", "congressional_district"):
        raise ValueError(f"level must be 'state' or 'congressional_district', got {level!r}")

    use_bundled = vintage == _BUNDLED_VINTAGE and (simplify is None or simplify >= _BUNDLED_SIMPLIFY_TOLERANCE)

    if use_bundled:
        gdf = _load_us_census_bundled(level, population, poverty, race, simplify)
    else:
        gdf = _load_us_census_live(level, vintage, population, poverty, race, simplify)

    return _finalize_census_gdf(gdf, level, population, poverty, race, contiguous_only)


def load_us_state_population() -> "pandas.DataFrame":
    """Load historical US state population estimates (1900-2024).

    Annual population estimates for all 50 states, DC, and Puerto Rico
    (Puerto Rico from 2010 onward only). Pre-1970 values are rounded
    intercensal estimates; 1970-2024 are from the US Census Bureau.

    The data was compiled from:

    - 1900-1969: JoshData/historical-state-population-csv (GitHub)
    - 1970-2024: US Census Bureau population estimate programs

    To update the data, run ``scripts/download_state_population.py``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``year``, ``state_fips``, ``state_name``, ``state_abbr``,
        ``population``.

    Examples
    --------
    >>> from carto_flow.data import load_us_state_population
    >>> df = load_us_state_population()
    >>> df.head(2)
       year state_fips state_name state_abbr  population
    0  1900         01    Alabama         AL     1830000
    1  1900         02     Alaska         AK      ...
    """
    import pandas as pd

    path = files("carto_flow.data").joinpath("us_state_population.csv")
    return pd.read_csv(path)


def load_us_covid_weekly() -> "pandas.DataFrame":
    """Load weekly US COVID-19 confirmed cases per state (2020-01 to 2023-03).

    Values are weekly *new* confirmed cases -- counts, not rates -- for all 50
    states, DC, and five territories, summed from the county-level cumulative
    series and aggregated into weeks ending Sunday. Divide by a population
    column to get a rate.

    The data comes from the Johns Hopkins University Center for Systems Science
    and Engineering (JHU CSSE) COVID-19 Data Repository, licensed CC BY 4.0;
    attribution to JHU is required wherever the data is shown. Collection ended
    on 2023-03-09 and the repository was archived, so the series is final.

    ``state_name`` matches the ``State Name`` column of ``load_us_census``
    for the 50 states and DC.

    To update the data, run ``scripts/download_covid_cases.py``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``week_ending`` (datetime64), ``state_name``, ``new_cases``.

    Examples
    --------
    >>> from carto_flow.data import load_us_covid_weekly
    >>> df = load_us_covid_weekly()
    >>> print(df.shape)
    (9184, 3)
    >>> weekly = df.pivot(index="week_ending", columns="state_name", values="new_cases")
    >>> print(weekly.shape)
    (164, 56)
    """
    import pandas as pd

    path = files("carto_flow.data").joinpath("us_covid_weekly.parquet")
    return pd.read_parquet(path)


__all__ = [
    "DIVISION_NAMES",
    "REGION_DIVISIONS",
    "REGION_NAMES",
    "STATE_DIVISIONS",
    "STATE_REGIONS",
    "load_sample_cities",
    "load_us_census",
    "load_us_covid_weekly",
    "load_us_state_population",
    "load_us_states",
    "load_world",
]
