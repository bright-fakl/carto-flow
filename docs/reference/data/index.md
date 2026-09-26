# Data Module

Data loading utilities for example datasets used in documentation and examples. This is an **optional subpackage** that requires additional dependencies.

## Overview

The `data` module provides convenience functions to load example datasets without requiring users to manually download and manage data files. The datasets are either included with the package or downloaded on-demand from reliable sources.

## Installation

This module has **optional dependencies** that are not required for the core functionality of carto-flow. To install the data module dependencies:

```bash
pip install carto-flow[data]
```

This will install:
- `geodatasets` - For accessing example datasets from the geodatasets library
- `censusdis` - Only needed to regenerate the bundled census snapshot with
  `scripts/build_census_snapshot.py`; not required at runtime.

If these dependencies are not installed, functions that require them will raise a clear `ImportError` with instructions on how to install the missing packages.

## Main Interface

| Function | Description |
|----------|-------------|
| **[`load_world()`](api.md)** | World countries with population estimates |
| **[`load_us_states()`](api.md)** | US states with population and area information |
| **[`load_sample_cities()`](api.md)** | Sample cities with population information |
| **[`load_us_census()`](api.md)** | US states/districts with ACS 2020 demographic data (bundled, no network or key needed) |
| **[`load_us_state_population()`](api.md)** | Annual US state population estimates, 1900-2024 |
| **[`load_us_covid_weekly()`](api.md)** | Weekly new confirmed COVID-19 cases per US state, 2020-2023 (JHU CSSE, CC BY 4.0) |

## See Also

- [Flow Cartogram](../flow_cartogram/index.md) — Cartogram generation for area-proportional visualizations
- [Symbol Cartogram](../symbol_cartogram/index.md) — Symbol-based cartograms
- [Proportional Cartogram](../proportional_cartogram/index.md) — Shape splitting and shrinking algorithms
