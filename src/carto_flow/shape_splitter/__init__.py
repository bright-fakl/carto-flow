"""
Shape splitting and manipulation algorithms.

This module provides utilities for splitting and manipulating polygon shapes,
including optimal splitting algorithms that preserve area relationships and
shape shrinking while maintaining general form.

Main Components
---------------
Shape Splitting
    split: Split a shape into N parts with specified area fractions
    partition_geometries: Split multiple geometries with area-based optimization

Shape Manipulation
    shrink: Shrink a shape to create concentric shells with specified area fractions

Examples
--------
Shape splitting:

    >>> from carto_flow.shape_splitter import split
    >>>
    >>> # Binary split: divide into 2 parts with 60/40 ratio
    >>> parts = split(input_shape, 0.6)
    >>> print(f"Part1 area: {parts[0].area:.2f}, Part2 area: {parts[1].area:.2f}")
    >>>
    >>> # N-way split: divide into 3 parts with 30%, 20%, 50% of area
    >>> parts = split(input_shape, [0.3, 0.2, 0.5])
    >>> print(f"Areas: {[p.area for p in parts]}")

Batch processing with GeoDataFrame:

    >>> import geopandas as gpd
    >>> from carto_flow.shape_splitter import partition_geometries
    >>>
    >>> # Process geometries based on column values
    >>> result = partition_geometries(gdf, 'population', method='shrink', normalization='sum')
    >>> print(f"Processed: {result.geometry.area.values}")

Shape shrinking:

    >>> from carto_flow.shape_splitter import shrink
    >>>
    >>> # Binary shrink: shrink to 80% of original area
    >>> parts = shrink(original_shape, 0.8)
    >>> print(f"Shell area: {parts[0].area:.2f}, Core area: {parts[1].area:.2f}")
    >>>
    >>> # N-way shrink: create 3 shells (25% each) and a core (25%)
    >>> parts = shrink(original_shape, [0.25, 0.25, 0.25, 0.25])
    >>> print(f"Areas: {[round(p.area, 2) for p in parts]}")
"""

# Shape splitting functions
from .partition import partition_geometries
from .shrink import shrink
from .split import split

# Define public API for explicit control over what is exported
__all__ = [
    "partition_geometries",
    "shrink",
    "split",
]
