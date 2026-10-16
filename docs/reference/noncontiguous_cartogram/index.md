# Non-contiguous Cartogram

Olson cartograms: every region keeps its shape and is scaled about its own center to an area proportional to its value.

## Overview

The `noncontiguous_cartogram` module scales each region by the square root of the ratio of the area that its value asks for to its area. The regions do not touch, no shape is distorted and no solver runs. One reference density, the value per unit of area at which a region keeps its size, is the same for all regions.

## Main Interface

| Sub-module | Description |
|------------|-------------|
| **[API](api.md)** | `create_noncontiguous_cartogram` — reference density, growth limit, treatment of multipolygons |
| **[Result](result.md)** | `NoncontiguousCartogram`, `NoncontiguousPlotResult` — scaled regions, scale factors, area errors |
| **[Visualization](visualization.md)** | `plot_cartogram` — scaled regions over the original outlines |

## Quick Start

```python
from carto_flow.noncontiguous_cartogram import create_noncontiguous_cartogram

cartogram = create_noncontiguous_cartogram(states, "Population")
cartogram.plot(facecolor="Region")
scaled = cartogram.to_geodataframe()
```

See [How a non-contiguous cartogram is scaled](../../explanations/noncontiguous-cartogram.md) for the choice of the reference density and the treatment of multipolygons.
