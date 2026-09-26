#!/usr/bin/env python3
"""Build the bundled weekly US COVID-19 case series used by ``load_us_covid_weekly``.

Data source
-----------
JHU CSSE COVID-19 Data Repository, US confirmed cases time series:

    https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/
    csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv

The source file is county-level *cumulative* confirmed cases, one column per
day. JHU stopped collecting on 2023-03-09 and archived the repository, so the
series is final and will not change. Fetched 2026-09-26.

License
-------
CC BY 4.0, Johns Hopkins University. Attribution is required wherever the data
is shown; ``load_us_covid_weekly`` repeats it in its docstring.

Derivation
----------
County rows are summed to states and territories, differenced to daily new
cases, negative differences (retrospective corrections in the source) clipped
to zero, and summed into weeks ending Sunday. The bundled file holds weekly
new case *counts*, not rates: an example that wants a rate divides by
population itself, which keeps that step visible to the reader.

Output columns
--------------
week_ending, state_name, new_cases

Usage
-----
    uv run python scripts/download_covid_cases.py
    uv run python scripts/download_covid_cases.py --output path/to/output.parquet
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
import requests

SOURCE_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
)

# Non-state rows: cruise ships and the repatriation aggregate carry no geography.
EXCLUDED_STATES = ("Diamond Princess", "Grand Princess")

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "src" / "carto_flow" / "data" / "us_covid_weekly.parquet"


def _fetch(url: str) -> str:
    print(f"  GET {url}")
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    return response.text


def build(text: str) -> pd.DataFrame:
    """Turn the raw county-level cumulative CSV into weekly new cases per state."""
    raw = pd.read_csv(io.StringIO(text))
    print(f"  {len(raw):,} county rows, {len(raw.columns):,} columns")

    date_columns = [c for c in raw.columns if _is_date_column(c)]
    if not date_columns:
        raise ValueError("No date columns found; the source layout may have changed.")

    cumulative = raw.groupby("Province_State")[date_columns].sum().transpose()
    cumulative.index = pd.to_datetime(cumulative.index, format="%m/%d/%y")
    cumulative = cumulative.drop(columns=[c for c in EXCLUDED_STATES if c in cumulative.columns])

    # The first day has no predecessor, so it yields no new-case value.
    daily = cumulative.diff().clip(lower=0).iloc[1:]
    weekly = daily.resample("W").sum()

    df = (
        weekly.stack()
        .rename("new_cases")
        .reset_index()
        .rename(columns={"level_0": "week_ending", "Province_State": "state_name"})
    )
    df["week_ending"] = df["week_ending"].dt.date.astype("datetime64[ns]")
    df["new_cases"] = df["new_cases"].round().astype("int64")
    return df.sort_values(["week_ending", "state_name"]).reset_index(drop=True)


def _is_date_column(name: str) -> bool:
    try:
        pd.to_datetime(name, format="%m/%d/%y")
    except ValueError:
        return False
    return True


def main(output: Path) -> None:
    print("Downloading JHU CSSE US confirmed cases ...")
    df = build(_fetch(SOURCE_URL))

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False, compression="zstd")

    print(f"\nWrote {len(df):,} rows -> {output}")
    print(f"Weeks:              {df['week_ending'].min().date()} to {df['week_ending'].max().date()}")
    print(f"States/territories: {df['state_name'].nunique()}")
    print(f"File size:          {output.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output parquet path (default: {DEFAULT_OUTPUT})",
    )
    main(parser.parse_args().output)
