"""
Shape splitting and manipulation algorithms for polygon geometries.

This module provides tools for dividing and shrinking polygon geometries based
on area fractions. This enables visualization of proportional data within
regions, creation of treemap-style layouts, and batch processing of GeoDataFrames.

Functions
---------
split
    Divide a geometry into multiple parts with specified area fractions.
    Supports sequential and treemap splitting strategies.
shrink
    Create concentric shells by shrinking a geometry inward. Returns parts
    from innermost core to outermost shell. Supports area-based and
    thickness-based modes.
partition_geometries
    Batch process geometries in a GeoDataFrame using split or shrink.
    Supports parallel processing and multiple normalization modes.
plot_partitions
    Visualize partitioned geometries as a choropleth map with
    flexible coloring and styling options.

Notes
-----
**Splitting Strategies**

- ``sequential`` (default) - Carves parts one-by-one from edges. Good for
  strip-like divisions.
- ``treemap`` - Recursive binary partitioning. Creates grid-like balanced
  layouts.

**Shrinking Modes**

- ``area`` (default) - Fractions represent direct area ratios.
- ``shell`` - Fractions represent shell thickness (squared for area).

**Normalization Options**

When using ``partition_geometries``, the ``normalization`` parameter controls
how column values are converted to fractions:

- ``sum`` - Divide by sum of all values across all rows.
- ``maximum`` - Divide by maximum row sum.
- ``row`` - Each row sums to 1.0 (no remainder geometry).
- ``None`` - Use values directly (must be 0-1).

Examples
--------
Split a polygon into parts:

>>> from carto_flow.proportional_cartogram import split
>>> from shapely.geometry import Polygon
>>> polygon = Polygon([(0, 0), (10, 0), (10, 5), (0, 5)])
>>> parts = split(polygon, fractions=0.6, direction="vertical")
>>> len(parts)
2

Create concentric shells:

>>> from carto_flow.proportional_cartogram import shrink
>>> shells = shrink(polygon, fractions=0.3)
>>> len(shells)
2

Batch process a GeoDataFrame:

>>> from carto_flow.proportional_cartogram import partition_geometries
>>> result = partition_geometries(
...     gdf,
...     columns=["category_a", "category_b"],
...     method="shrink",
...     normalization="row",
... )

Different splitting strategies:

>>> parts = split(polygon, fractions=[0.3, 0.3, 0.4], strategy="sequential")
>>> parts = split(polygon, fractions=[0.25, 0.25, 0.25, 0.25], strategy="treemap")

Different shrinking modes:

>>> shells = shrink(polygon, fractions=0.3, mode="area")
>>> shells = shrink(polygon, fractions=0.3, mode="shell")

Combining with cartograms for multi-dimensional visualizations:

>>> from carto_flow.flow_cartogram import morph_gdf, MorphOptions
>>> from carto_flow.proportional_cartogram import partition_geometries, plot_partitions
>>> result = morph_gdf(gdf, "total_gdp", options=MorphOptions.preset_balanced())
>>> cartogram = result.geometries
>>> partitioned = partition_geometries(
...     cartogram,
...     columns=["agriculture", "industry", "services"],
...     method="split",
...     normalization="row",
...     strategy="treemap",
... )
>>> plot_partitions(partitioned, color_by="category", legend=True)

See Also
--------
[carto_flow.flow_cartogram][] : Cartogram generation algorithms.
"""

# Shape splitting functions
from .dot_density import generate_dot_density, plot_dot_density
from .partition import partition_geometries
from .plot_results import DotDensityPlotResult, PartitionsPlotResult
from .shrinking import shrink
from .splitting import split
from .visualization import plot_partitions

# Define public API for explicit control over what is exported
__all__ = [
    "DotDensityPlotResult",
    "PartitionsPlotResult",
    "generate_dot_density",
    "partition_geometries",
    "plot_dot_density",
    "plot_partitions",
    "shrink",
    "split",
]
