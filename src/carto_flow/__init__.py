"""
CartoFlow: High-performance cartogram generation and shape processing library.

This package provides comprehensive tools for creating flow-based cartograms
and processing geospatial shapes with optimized algorithms for performance
and accuracy.

Main Components
---------------

Shape Morphing (Cartogram Generation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Core cartogram algorithms for deforming polygons based on data values:

- morph_gdf: GeoDataFrame-based cartogram generation interface
- morph_geometries: Low-level interface working with shapely geometries
- multiresolution_morph: Multi-resolution morphing with progressive refinement
- MorphComputer: Object-oriented interface with refinement support
- MorphOptions: Configuration options for morphing algorithms
- MorphResult: Complete results container with metadata

Geometry Optimization
~~~~~~~~~~~~~~~~~~~~~
High-performance geometry processing utilities:

- unpack_geometries: Convert shapely geometries to efficient coordinate arrays
- reconstruct_geometries: Rebuild shapely geometries from coordinate arrays
- GeometryCoordinateInfo: Efficient coordinate storage and processing
- compute_polygon_area_numba: Fast area computation using Numba optimization

Examples
--------

Basic cartogram creation:

    >>> import geopandas as gpd
    >>> from carto_flow import morph_gdf, MorphOptions
    >>> from carto_flow.grid import Grid
    >>>
    >>> # Load geographic data
    >>> gdf = gpd.read_file('regions.geojson')
    >>> grid = Grid.from_bounds(gdf.total_bounds, size=100)
    >>>
    >>> # Create population-based cartogram
    >>> result = morph_gdf(gdf, 'population', options=MorphOptions(grid=grid))
    >>> cartogram = result.geometries

Multi-resolution cartogram with refinement:

    >>> from carto_flow import MorphComputer, multiresolution_morph
    >>>
    >>> # Multi-resolution morphing (returns final MorphResult)
    >>> result = multiresolution_morph(gdf, 'population', resolution=512, levels=3)
    >>> cartogram = result.geometries
    >>> grid = result.grid
    >>>
    >>> # Object-oriented approach with refinement
    >>> computer = MorphComputer(gdf, 'population', options=MorphOptions(grid=grid))
    >>> result = computer.morph()
    >>> cartogram, history = result.geometries, result.history
    >>>
    >>> # Refine with different parameters
    >>> computer.set_computation(mean_tol=0.02, n_iter=200)
    >>> refined_result = computer.morph()
"""

# ============================================================================
# Shape Morphing (Cartogram Generation)
# ============================================================================

# Core morphing functions
# ============================================================================
# Geometry Optimization
# ============================================================================
from .optimizations import (
    GeometryCoordinateInfo,
    compute_complex_polygon_areas_numba,
    compute_polygon_area_numba,
    reconstruct_geometries,
    reconstruct_geometry,
    unpack_geometries,
    unpack_geometry,
)

# Object-oriented interface
# Sub-modules from shape_morpher
from .shape_morpher import (
    MorphComputer,
    MorphOptions,
    MorphResult,
    MorphStatus,
    anisotropy,
    density,
    displacement,
    grid,
    history,
    morph_gdf,
    morph_geometries,
    multiresolution_morph,
    velocity,
)

# ============================================================================
# Public API Definition
# ============================================================================

__all__ = [
    "GeometryCoordinateInfo",
    # Object-oriented interface
    "MorphComputer",
    "MorphOptions",
    "MorphResult",
    "MorphStatus",
    # Shape morpher sub-modules
    "anisotropy",
    "compute_complex_polygon_areas_numba",
    # Geometry optimization
    "compute_polygon_area_numba",
    "density",
    "displacement",
    "grid",
    "history",
    # Core morphing functions
    "morph_gdf",
    "morph_geometries",
    "multiresolution_morph",
    "reconstruct_geometries",
    "reconstruct_geometry",
    "unpack_geometries",
    "unpack_geometry",
    "velocity",
]
