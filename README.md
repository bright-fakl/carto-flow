# carto-flow

[![Release](https://img.shields.io/github/v/release/bright-fakl/carto-flow)](https://github.com/bright-fakl/carto-flow/releases)
[![Build Status](https://img.shields.io/github/actions/workflow/status/bright-fakl/carto-flow/main.yml?branch=main&label=build)](https://github.com/bright-fakl/carto-flow/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/bright-fakl/carto-flow/branch/main/graph/badge.svg)](https://codecov.io/gh/bright-fakl/carto-flow)
[![Commit Activity](https://img.shields.io/github/commit-activity/m/bright-fakl/carto-flow)](https://github.com/bright-fakl/carto-flow/graphs/commit-activity)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/bright-fakl/carto-flow/blob/main/LICENSE)

A Python library for creating cartograms from geographic data.

## Overview

carto-flow transforms geographic vector data into cartograms — maps where region sizes or symbols are scaled to represent a data variable such as population, GDP, or election results. It supports five kinds of cartograms:

- **Flow cartograms** — regions are continuously deformed so their areas are proportional to a variable, while preserving shape and topology as much as possible (diffusion-based algorithm).
- **Symbol cartograms** — regions are replaced by proportional symbols (circles, squares, hexagons, or custom isohedral tiles) arranged using force-based or grid-based layout.
- **Voronoi cartograms** — each region is a cell of a Voronoi tessellation, relaxed so that the cell areas are proportional to a variable (Lloyd relaxation), optionally starting from a flow cartogram.
- **Non-contiguous cartograms** — each region keeps its shape and is scaled about its own center to an area proportional to a variable.
- **Proportional cartograms** — region polygons are split or shrunk to show sub-group proportions within each geographic unit, and dot density maps show counts as dots.

![Cartogram Gallery](https://github.com/bright-fakl/carto-flow/raw/main/cartogram-example.png)

## Features

- Diffusion-based flow cartogram morphing with FFT-accelerated Poisson solver
- Multi-resolution processing for large datasets
- Force-based symbol placement with overlap resolution and topology preservation
- Isohedral tile support for custom symbol shapes, and tile-count maps (mosaic layout)
- Voronoi cartograms with exact and raster backends and topology repair
- Non-contiguous (Olson) cartograms with a choice of reference density
- Split and shrink operations for proportional cartograms, and dot density maps
- Plot styles (`default`, `dark`, `classic`, or your own) that set the look of every plot and animation in one place
- Animations of the morph, the density and velocity fields
- Bundled example data: US states and districts with census data, world countries, cities, 2020 presidential results
- Rich visualization utilities built on matplotlib and geopandas

## Installation

```bash
pip install carto-flow
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add carto-flow
```

Requires Python 3.10+.

## Quick example

```python
import carto_flow.data as examples
import carto_flow.flow_cartogram as flow
import carto_flow.symbol_cartogram as smb
import carto_flow.voronoi_cartogram as vor
from carto_flow.noncontiguous_cartogram import create_noncontiguous_cartogram

states = examples.load_us_census(population=True)

# Flow cartogram: the area of each state is proportional to its population
flow.create_flow_cartogram(states, "Population").plot("Population (Millions)")

# Symbol cartogram: one hexagon per state
smb.create_symbol_cartogram(states, layout=smb.GridBasedLayout(tiling=smb.HexagonTiling())).plot(
    facecolor="Population (Millions)"
)

# Voronoi cartogram: one cell per state
vor.create_voronoi_cartogram(states, weights="Population").plot("Population (Millions)")

# Non-contiguous cartogram: every state scaled about its own center
create_noncontiguous_cartogram(states, "Population").plot(facecolor="Population (Millions)")
```

Every plot follows a plot style; pass `style="dark"` (or your own `PlotStyle`) to change the look.

## Documentation

Full documentation is available at **<https://bright-fakl.github.io/carto-flow/>**, including:

- [Tutorials](https://bright-fakl.github.io/carto-flow/tutorials/) — step-by-step guides for each cartogram type
- [Gallery](https://bright-fakl.github.io/carto-flow/generated/gallery/) — examples with their code and animations
- [How-to guides](https://bright-fakl.github.io/carto-flow/how-to/) — task-focused recipes
- [Reference](https://bright-fakl.github.io/carto-flow/reference/) — full API reference
- [Explanations](https://bright-fakl.github.io/carto-flow/explanations/) — background on algorithms and design

## Changelog

[CHANGELOG.md](CHANGELOG.md) lists the changes in each release. Upgrading from
1.x requires code changes in the symbol cartogram API; the
[migration guide](https://bright-fakl.github.io/carto-flow/migrating-to-2.0/)
gives the old and new form of every affected call.

## License

See [LICENSE](LICENSE).
