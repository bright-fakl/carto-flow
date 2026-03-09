"""
High-performance geometry processing utilities.

This module provides Numba-optimized functions for polygon area computation,
geometry unpacking/reconstruction, and coordinate manipulation. It enables
efficient batch processing of geometries by separating coordinate storage
from geometry objects.

Classes
-------
GeometryCoordinateInfo
    Container for flattened coordinates with reconstruction metadata.

Functions
---------
unpack_geometries
    Convert list of geometries to flattened coordinate array.
unpack_geometry
    Convert single geometry to coordinates and metadata.
reconstruct_geometries
    Rebuild geometries from GeometryCoordinateInfo.
reconstruct_geometry
    Rebuild single geometry from coordinates and metadata.
compute_polygon_area_numba
    Fast shoelace formula for single polygon ring.
compute_complex_polygon_areas_numba
    Parallel area computation for polygons with holes.
find_adjacent_pairs
    Find touching geometry pairs using buffered intersection.
simplify_coverage
    Simplify polygon geometries while preserving shared boundaries.

Notes
-----
**Performance Benefits**

- Numba JIT compilation for shoelace area computation
- Parallel processing of polygon rings via ``prange``
- Lazy computation and caching of ring info
- Direct array operations without geometry reconstruction

**Typical Workflow**

1. Unpack geometries to coordinate array
2. Transform coordinates in-place (displacement, scaling, etc.)
3. Compute areas directly from coordinates (no reconstruction)
4. Reconstruct geometries only when needed for output

Examples
--------
>>> from carto_flow.geo_utils import unpack_geometries, reconstruct_geometries
>>> from shapely.geometry import Polygon
>>>
>>> # Process multiple polygons efficiently
>>> polygons = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
>>> coord_info = unpack_geometries(polygons, precompute_ring_info=True)
>>>
>>> # Transform coordinates in-place
>>> coord_info.coords += displacement_vector
>>> coord_info.invalidate_cache()
>>>
>>> # Compute areas efficiently without reconstruction
>>> areas = coord_info.compute_areas(use_parallel=True)
>>>
>>> # Reconstruct geometries only when needed
>>> final_polygons = reconstruct_geometries(coord_info)

See Also
--------
[carto_flow.flow_cartogram.displacement][] : Coordinate displacement utilities.
"""

# Import and re-export main functions and classes from geometry module
from .adjacency import find_adjacent_pairs
from .geometry import (
    # Classes
    GeometryCoordinateInfo,
    compute_complex_polygon_areas_numba,
    # Area computation functions
    compute_polygon_area_numba,
    # Reconstruction functions
    reconstruct_geometries,
    reconstruct_geometry,
    # Unpacking functions
    unpack_geometries,
    unpack_geometry,
)
from .simplification import simplify_coverage

# Define public API for explicit control over what is exported
__all__ = [
    "GeometryCoordinateInfo",
    "compute_complex_polygon_areas_numba",
    "compute_polygon_area_numba",
    "find_adjacent_pairs",
    "reconstruct_geometries",
    "reconstruct_geometry",
    "simplify_coverage",
    "unpack_geometries",
    "unpack_geometry",
]
