#!/usr/bin/env python3
"""Draw ``cartogram-example.png`` of the README: a flow, a symbol, a proportional and a dot density cartogram of the
US states in the default plot style.

Usage
-----
    uv run python scripts/make_readme_figure.py
    uv run python scripts/make_readme_figure.py --output path/to/figure.png
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import carto_flow.data as examples
import carto_flow.flow_cartogram as flow
import carto_flow.proportional_cartogram as pc
import carto_flow.symbol_cartogram as smb
from carto_flow import get_style

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "cartogram-example.png"


def main(output: Path) -> None:
    warnings.filterwarnings("ignore")
    states = examples.load_us_census(population=True, race=True, poverty=True, simplify=1000)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    get_style().apply(fig)
    (ax_flow, ax_symbol), (ax_split, ax_dots) = axes

    result = flow.create_flow_cartogram(states, "Population", options=flow.MorphOptions(show_progress=False))
    result.plot("Population (Millions)", ax=ax_flow, legend_kwargs={"shrink": 0.7})
    ax_flow.set_title("Flow cartogram")

    symbols = smb.create_symbol_cartogram(
        states, layout=smb.GridBasedLayout(tiling=smb.HexagonTiling()), show_progress=False
    )
    symbols.plot(
        ax=ax_symbol,
        facecolor="Population (Millions)",
        legend_kwargs={"shrink": 0.7},
        labels=states["State Abbreviation"],
        label_color="w",
        label_kwargs={"fontweight": "bold"},
        label_fontsize=5,
    )
    ax_symbol.set_title("Symbol cartogram")

    partition = pc.partition_geometries(
        states,
        columns=["Black or African American", "Asian", "Hispanic or Latino"],
        method="split",
        direction="horizontal",
        alternate=True,
        normalization="row",
    )
    pc.plot_partitions(partition, facecolor="category", ax=ax_split)
    ax_split.set_title("Proportional cartogram")

    pc.plot_dot_density(
        states,
        columns=["Above Poverty Level", "Below Poverty Level"],
        n_dots=1000,
        normalization="maximum",
        alpha=[0.1, 1.0],
        ax=ax_dots,
    )
    ax_dots.set_title("Dot density")

    for ax in (ax_split, ax_dots):  # a small legend in the lower left corner, away from the map
        legend = ax.get_legend()
        legend.set_loc("lower left")
        legend.set_frame_on(False)
        for text in [*legend.get_texts(), legend.get_title()]:
            text.set_fontsize(8)
    for ax in axes.ravel():
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output, dpi=110)
    print(f"Wrote {output} ({output.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path.")
    main(parser.parse_args().output)
