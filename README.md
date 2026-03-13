# carto-flow

[![Release](https://img.shields.io/github/v/release/bright-fakl/carto-flow)](https://github.com/bright-fakl/carto-flow/releases)
[![Build Status](https://img.shields.io/github/actions/workflow/status/bright-fakl/carto-flow/main.yml?branch=main&label=build)](https://github.com/bright-fakl/carto-flow/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/bright-fakl/carto-flow/branch/main/graph/badge.svg)](https://codecov.io/gh/bright-fakl/carto-flow)
[![Commit Activity](https://img.shields.io/github/commit-activity/m/bright-fakl/carto-flow)](https://github.com/bright-fakl/carto-flow/graphs/commit-activity)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/bright-fakl/carto-flow/blob/main/LICENSE)

A Python library for creating cartograms from geographic data.

## Overview

carto-flow transforms geographic vector data into cartograms — maps where region sizes or symbols are scaled to represent a data variable such as population, GDP, or election results. It supports three cartogram styles:

- **Flow cartograms** — regions are continuously deformed so their areas are proportional to a variable, while preserving shape and topology as much as possible (diffusion-based algorithm).
- **Symbol cartograms** — regions are replaced by proportional symbols (circles, squares, hexagons, or custom isohedral tiles) arranged using physics-based or grid-based layout.
- **Proportional cartograms** — region polygons are split or shrunk to show sub-group proportions within each geographic unit.

![Example Flow Cartogram showing population distribution](cartogram-example.png)

## Features

- Diffusion-based flow cartogram morphing with FFT-accelerated Poisson solver
- Multi-resolution processing for large datasets
- Physics-based symbol placement with overlap resolution and topology preservation
- Isohedral tile support for custom symbol shapes
- Split and shrink operations for proportional cartograms
- Dot density cartogram support
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
import geopandas as gpd
from carto_flow.flow_cartogram import morph_gdf

gdf = gpd.read_file("countries.gpkg")
result = morph_gdf(gdf, values=gdf["population"])
result.cartogram_gdf.plot()
```

```python
from carto_flow.symbol_cartogram import create_symbol_cartogram

layout = create_symbol_cartogram(gdf, values=gdf["population"])
layout.plot()
```

## Documentation

Full documentation is available at **<https://bright-fakl.github.io/carto-flow/>**, including:

- [Tutorials](https://bright-fakl.github.io/carto-flow/tutorials/) — step-by-step guides for each cartogram type
- [How-to guides](https://bright-fakl.github.io/carto-flow/how-to/) — task-focused recipes
- [Reference](https://bright-fakl.github.io/carto-flow/reference/) — full API reference
- [Explanations](https://bright-fakl.github.io/carto-flow/explanations/) — background on algorithms and design

## License

See [LICENSE](LICENSE).
