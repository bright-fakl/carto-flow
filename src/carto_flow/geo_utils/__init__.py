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
densify_coverage
    Insert vertices so no straight segment exceeds a given length.
explode_geodataframe
    Force-based separation of overlapping/touching polygons in a GeoDataFrame.
repair_adjacency
    Permute slots so that input-adjacent geometries stay output-adjacent.
repair_compactness
    Boundary swaps between adjacent groups to reduce inertia.
repair_contiguity
    Permute slots so that each group's cells form a contiguous region.
repair_group_assignment
    Reassign group membership to satisfy a contiguity constraint.
components_from_adjacency
    Derive connected components from a dense adjacency matrix.
compute_connected_components
    Detect connected components among geometries (Union-Find over adjacency).
prescale_connected_components
    Uniformly scale each connected component to its target total area.

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
from .contiguity import (
    repair_adjacency,
    repair_compactness,
    repair_contiguity,
    repair_group_assignment,
)
from .explode import explode_geodataframe
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
from .prescale import (
    components_from_adjacency,
    compute_connected_components,
    prescale_connected_components,
)
from .simplification import densify_coverage, simplify_coverage

# Define public API for explicit control over what is exported
__all__ = [
    "GeometryCoordinateInfo",
    "components_from_adjacency",
    "compute_complex_polygon_areas_numba",
    "compute_connected_components",
    "compute_polygon_area_numba",
    "densify_coverage",
    "explode_geodataframe",
    "find_adjacent_pairs",
    "prescale_connected_components",
    "reconstruct_geometries",
    "reconstruct_geometry",
    "repair_adjacency",
    "repair_compactness",
    "repair_contiguity",
    "repair_group_assignment",
    "simplify_coverage",
    "unpack_geometries",
    "unpack_geometry",
]
