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

- create_flow_cartogram: GeoDataFrame-based flow cartogram generation interface (morph_gdf is deprecated)
- morph_geometries: Low-level interface working with shapely geometries
- multiresolution_morph: Deprecated; use create_flow_cartogram with a LadderSpec
- Cartogram: Result class with morphed geometries and rich interface
- CartogramWorkflow: Workflow class for iterative refinement
- MorphOptions: Configuration options for morphing algorithms

Geometry Optimization
~~~~~~~~~~~~~~~~~~~~~
High-performance geometry processing utilities:

- unpack_geometries: Convert shapely geometries to efficient coordinate arrays
- reconstruct_geometries: Rebuild shapely geometries from coordinate arrays
- GeometryCoordinateInfo: Efficient coordinate storage and processing
- compute_polygon_area_numba: Fast area computation using Numba optimization

Symbol Cartogram
~~~~~~~~~~~~~~~~
Create cartograms where each region is represented by a single symbol:

- create_symbol_cartogram: Create symbol cartograms from GeoDataFrames
- SymbolCartogram: Result class with symbol geometries and metrics

Examples
--------

Basic cartogram creation:

    >>> import geopandas as gpd
    >>> from carto_flow import create_flow_cartogram, MorphOptions
    >>>
    >>> # Load geographic data
    >>> gdf = gpd.read_file('regions.geojson')
    >>>
    >>> # Create population-based cartogram
    >>> cartogram = create_flow_cartogram(gdf, 'population', options=MorphOptions.preset_balanced())
    >>> cartogram.plot()
    >>> gdf_result = cartogram.to_geodataframe()

Multi-resolution cartogram with refinement:

    >>> from carto_flow import CartogramWorkflow, LadderSpec, MorphOptions, create_flow_cartogram
    >>>
    >>> # Multi-resolution morphing (returns final Cartogram)
    >>> options = MorphOptions(grid=LadderSpec(min_resolution=128, levels=3))
    >>> cartogram = create_flow_cartogram(gdf, 'population', options=options)
    >>> print(f"Status: {cartogram.status}, Error: {cartogram.get_errors().mean_error_pct:.1f}%")
    >>>
    >>> # Workflow approach with iterative refinement
    >>> workflow = CartogramWorkflow(gdf, 'population')
    >>> workflow.morph()                    # Initial morphing
    >>> workflow.morph(mean_tol=0.01)       # Refine with tighter tolerance
    >>> gdf_result = workflow.to_geodataframe()
"""

# ============================================================================
# Flow-Based Cartogram Generation
# ============================================================================

# Core morphing functions
# ============================================================================
# Geometry Optimization
# ============================================================================
# Object-oriented interface
# Sub-modules from flow_cartogram
from .flow_cartogram import (
    Cartogram,
    CartogramWorkflow,
    GridSpec,
    LadderSpec,
    MorphOptions,
    MorphStatus,
    anisotropy,
    create_flow_cartogram,
    density,
    displacement,
    grid,
    history,
    morph_gdf,
    morph_geometries,
    multiresolution_morph,
    velocity,
)
from .geo_utils import (
    GeometryCoordinateInfo,
    compute_complex_polygon_areas_numba,
    compute_polygon_area_numba,
    reconstruct_geometries,
    reconstruct_geometry,
    simplify_coverage,
    unpack_geometries,
    unpack_geometry,
)

# ============================================================================
# Proportional Cartogram Partitioning
# ============================================================================
from .proportional_cartogram import (
    partition_geometries,
    shrink,
    split,
)

# ============================================================================
# Plot styles
# ============================================================================
from .style import Channel, Palettes, PlotStyle, get_style, register_style, set_style, style_context, unregister_style

# ============================================================================
# Symbol Cartogram
# ============================================================================
from .symbol_cartogram import (
    AdjacencyMode,
    SymbolCartogram,
    SymbolCartogramStatus,
    SymbolShape,
    create_symbol_cartogram,
)

# ============================================================================
# Public API Definition
# ============================================================================

__version__ = "2.0.0"

__all__ = [
    # Symbol cartogram
    "AdjacencyMode",
    # Cartogram classes
    "Cartogram",
    "CartogramWorkflow",
    "Channel",
    # Geometry optimization
    "GeometryCoordinateInfo",
    "GridSpec",
    "LadderSpec",
    "MorphOptions",
    "MorphStatus",
    "Palettes",
    "PlotStyle",
    "SymbolCartogram",
    "SymbolCartogramStatus",
    "SymbolShape",
    # Cartogram sub-modules
    "anisotropy",
    "compute_complex_polygon_areas_numba",
    "compute_polygon_area_numba",
    "create_flow_cartogram",
    "create_symbol_cartogram",
    "density",
    "displacement",
    "get_style",
    "grid",
    "history",
    # Cartogram functions
    "morph_gdf",
    "morph_geometries",
    "multiresolution_morph",
    # Shape splitting
    "partition_geometries",
    "reconstruct_geometries",
    "reconstruct_geometry",
    "register_style",
    "set_style",
    "shrink",
    "simplify_coverage",
    "split",
    "style_context",
    "unpack_geometries",
    "unpack_geometry",
    "unregister_style",
    "velocity",
]

# Optional data module (requires optional dependencies)
try:
    from . import data as data

    __all__.append("data")
except Exception:  # noqa: S110
    pass
