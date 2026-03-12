# carto-flow

[![Release](https://img.shields.io/github/v/release/fkloosterman/carto-flow)](https://img.shields.io/github/v/release/fkloosterman/carto-flow)
[![Build status](https://img.shields.io/github/actions/workflow/status/fkloosterman/carto-flow/main.yml?branch=main)](https://github.com/fkloosterman/carto-flow/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/fkloosterman/carto-flow/branch/main/graph/badge.svg)](https://codecov.io/gh/fkloosterman/carto-flow)
[![Commit activity](https://img.shields.io/github/commit-activity/m/fkloosterman/carto-flow)](https://img.shields.io/github/commit-activity/m/fkloosterman/carto-flow)
[![License](https://img.shields.io/github/license/fkloosterman/carto-flow)](https://img.shields.io/github/license/fkloosterman/carto-flow)

A Python library for creating cartograms from geographic data.

## Overview

carto-flow transforms geographic vector data into cartograms — maps where region sizes or symbols are scaled to represent a data variable such as population, GDP, or election results. It supports three cartogram styles:

- **Flow cartograms** — regions are continuously deformed so their areas are proportional to a variable, while preserving shape and topology as much as possible (diffusion-based algorithm).
- **Symbol cartograms** — regions are replaced by proportional symbols (circles, squares, hexagons, or custom isohedral tiles) arranged using physics-based or grid-based layout.
- **Proportional cartograms** — region polygons are split or shrunk to show sub-group proportions within each geographic unit.

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

Full documentation is available at **<https://fkloosterman.github.io/carto-flow/>**, including:

- [Tutorials](https://fkloosterman.github.io/carto-flow/tutorials/) — step-by-step guides for each cartogram type
- [How-to guides](https://fkloosterman.github.io/carto-flow/how-to/) — task-focused recipes
- [Reference](https://fkloosterman.github.io/carto-flow/reference/) — full API reference
- [Explanations](https://fkloosterman.github.io/carto-flow/explanations/) — background on algorithms and design

## License

See [LICENSE](LICENSE).
