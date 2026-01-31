"""
Shape splitting and manipulation algorithms.

This module provides utilities for splitting and manipulating polygon shapes,
including optimal splitting algorithms that preserve area relationships and
shape shrinking while maintaining general form.

Main Components
---------------
Shape Splitting
    split_shape: Split a shape into two parts along an optimal line
    split_geometries: Split multiple geometries with area-based optimization

Shape Manipulation
    shrink_shape: Shrink a shape while maintaining its general form

Examples
--------
Shape splitting:

    >>> from carto_flow.shape_splitter import split_shape
    >>>
    >>> # Split a shape into two parts with specific area ratio
    >>> part1, part2 = split_shape(input_shape, fraction=0.6)
    >>> print(f"Part1 area: {part1.area:.2f}, Part2 area: {part2.area:.2f}")

Batch processing with GeoDataFrame:

    >>> import geopandas as gpd
    >>> from carto_flow.shape_splitter import split_geometries
    >>>
    >>> # Process geometries based on column values
    >>> result = split_geometries(gdf, 'population', method='shrink', normalization='sum')
    >>> print(f"Processed: {result.geometry.area.values}")

Shape shrinking:

    >>> from carto_flow.shape_splitter import shrink_shape
    >>>
    >>> # Shrink a shape to 80% of its original area
    >>> shrunken, shell = shrink_shape(original_shape, fraction=0.8)
    >>> print(f"Original area: {original_shape.area:.2f}")
    >>> print(f"Shrunken area: {shrunken.area:.2f}")
"""

# Shape splitting functions
from .shape_splitter import (
    shrink_shape,
    split_geometries,
    split_shape,
)

# Define public API for explicit control over what is exported
__all__ = [
    "shrink_shape",
    "split_geometries",
    "split_shape",
]
