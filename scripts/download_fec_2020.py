#!/usr/bin/env python3
"""Build the bundled 2020 US presidential results by state used by ``load_us_president_2020``.

Data source
-----------
Federal Election Commission, "Federal Elections 2020", sheet "9. 2020 Pres General Results" of the workbook

    https://www.fec.gov/resources/cms-content/documents/federalelections2020.xlsx

The FEC compiles the official, certified results as reported by the state election offices. The results are final.

License
-------
The workbook is a publication of a United States federal agency, produced at taxpayer expense; the FEC page that
publishes it states no restriction on its reuse, and works of the United States government are not subject to
copyright (17 U.S.C. 105). The only reuse restriction the FEC states concerns information about individual
contributors (52 U.S.C. 30111(a)(4)), which this file does not contain. Attribution to the FEC is given wherever the
data is shown; ``load_us_president_2020`` repeats it in its docstring.

Derivation
----------
The votes of the candidates Biden and Trump are summed over the lines of the sheet that belong to the same state
(fusion-party lines such as New York D + WF for Biden are summed, as the FEC does in its combined totals). The national
totals are checked against the official ones.

Output columns
--------------
state_name, state_abbr, votes_dem, votes_gop

Usage
-----
    uv run --with openpyxl --with requests python scripts/download_fec_2020.py
    uv run --with openpyxl --with requests python scripts/download_fec_2020.py --output path/to/output.csv
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
import requests

SOURCE_URL = "https://www.fec.gov/resources/cms-content/documents/federalelections2020.xlsx"
SHEET = "9. 2020 Pres General Results"
# The official national totals of the two candidates.
NATIONAL_TOTALS = (81_283_501, 74_223_975)

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "src" / "carto_flow" / "data" / "us_president_2020.csv"


def _fetch(url: str) -> bytes:
    print(f"  GET {url}")
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    return response.content


def build(content: bytes) -> pd.DataFrame:
    """Turn the FEC workbook into the votes of the two candidates per state."""
    sheet = pd.read_excel(io.BytesIO(content), sheet_name=SHEET, header=0)
    sheet.columns = [str(c).strip() for c in sheet.columns]
    sheet = sheet[sheet["LAST NAME"].isin(["Biden", "Trump"]) & sheet["GENERAL RESULTS"].notna()]
    table = sheet.pivot_table(
        index=["STATE", "STATE ABBREVIATION"], columns="LAST NAME", values="GENERAL RESULTS", aggfunc="sum"
    ).reset_index()
    table.columns = ["state_name", "state_abbr", "votes_dem", "votes_gop"]
    table[["votes_dem", "votes_gop"]] = table[["votes_dem", "votes_gop"]].astype(int)
    totals = (int(table["votes_dem"].sum()), int(table["votes_gop"].sum()))
    if totals != NATIONAL_TOTALS:
        raise ValueError(f"the totals {totals} differ from the official national totals {NATIONAL_TOTALS}")
    return table.sort_values("state_name").reset_index(drop=True)


def main(output: Path) -> None:
    print("Downloading the FEC 2020 workbook ...")
    table = build(_fetch(SOURCE_URL))
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"\nWrote {len(table)} areas -> {output}")
    print(f"Biden {table['votes_dem'].sum():,}  Trump {table['votes_gop'].sum():,}")
    print(f"File size: {output.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path.")
    args = parser.parse_args()
    main(args.output)
