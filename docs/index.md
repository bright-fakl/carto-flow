# carto-flow

[![Release](https://img.shields.io/github/v/release/bright-fakl/carto-flow)](https://img.shields.io/github/v/release/bright-fakl/carto-flow)
[![Build status](https://img.shields.io/github/actions/workflow/status/bright-fakl/carto-flow/main.yml?branch=main)](https://github.com/bright-fakl/carto-flow/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/bright-fakl/carto-flow)](https://img.shields.io/github/commit-activity/m/bright-fakl/carto-flow)
[![License](https://img.shields.io/github/license/bright-fakl/carto-flow)](https://img.shields.io/github/license/bright-fakl/carto-flow)

**A Python library for creating diverse types of cartograms and proportional spatial visualizations.**

Carto-flow provides tools to transform geographic data into meaningful cartographic representations where visual properties correspond to data values. The library supports multiple cartogram types:

- **Flow-based cartograms**: Smoothly morph geometries while maintaining spatial contiguity and recognizable shapes, where region sizes are proportional to a data variable (e.g., population, GDP)
- **Symbol cartograms**: Represent regions as proportional symbols (circles, squares, hexagons) or custom tile maps
- **Proportional cartograms**: Split and shrink shapes to show proportions within regions
- **Dot density cartograms**: Visualize data through randomly distributed points

The library uses advanced diffusion-based algorithms for smooth morphing, physics-based simulation for optimal symbol placement, and provides tools for multi-resolution processing and batch operations.

![Example Flow Cartogram showing population distribution](./generated/gallery/images/mkd_glr_plot_flow_cartogram_001.png)

This example shows a flow-based cartogram where region sizes are proportional to population, demonstrating how carto-flow preserves spatial relationships while transforming geometries to reflect data values.

## Installation

```bash
pip install carto-flow
```

Requires Python 3.9+ and NumPy 2.0.2+.

## Quick Start

Create a population flow cartogram from a GeoDataFrame:

```python
import carto_flow.flow_cartogram as flow
import carto_flow.data as examples

us_states = examples.load_us_census(population=True)

# compute flow cartogram
result = flow.morph_gdf(
    us_states, 'Population'
)

result.plot('Population (Millions)')
```

## Documentation

Choose the documentation that best fits your needs:

### [Tutorials](tutorials/index.md)
Step-by-step guides to learn carto-flow through hands-on examples and real-world scenarios.

### [How-to Guides](how-to/index.md)
Practical guides showing how to accomplish specific tasks and solve common problems.

### [Reference](reference/index.md)
Complete API reference and technical documentation for all modules and functions.

### [Explanations](explanations/index.md)
Conceptual guides that explain the underlying principles and design decisions behind carto-flow.

## Contributing

We welcome contributions! See our [Contributing Guide](https://github.com/bright-fakl/carto-flow/blob/main/CONTRIBUTING.md) for details on how to get started.

## License

This project is licensed under the terms specified in the [LICENSE](https://github.com/bright-fakl/carto-flow/blob/main/LICENSE) file.
