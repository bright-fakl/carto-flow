"""
Shape morphing algorithms for flow-based cartogram generation.

This module provides comprehensive cartogram generation algorithms that create
flow-based cartograms by iteratively deforming polygons based on density-driven
velocity fields. These are the primary entry points for creating cartograms
from geospatial data.

Main Components
---------------
Core Functions

    multiresolution_morph: Multi-resolution morphing with progressive refinement
    morph_gdf: GeoDataFrame-based interface for cartogram generation
    morph_geometries: Low-level interface working directly with shapely geometries

Object-oriented interface

    MorphComputer: Stateful cartogram generation with refinement support
    MorphOptions: Configuration options for morphing algorithms
    MorphResult: Complete results container with metadata

Examples
--------
Basic GeoDataFrame interface:

    >>> from carto_flow.shape_morpher import morph_gdf, MorphComputer, MorphOptions
    >>> from carto_flow.grid import Grid
    >>>
    >>> # Set up computation
    >>> grid = Grid.from_bounds((0, 0, 100, 80), size=100)
    >>>
    >>> # Create cartogram with GeoDataFrame interface
    >>> result = morph_gdf(gdf, 'population', options=MorphOptions(grid=grid))
    >>> cartogram = result.geometries
    >>> history = result.history

Object-oriented approach with refinement:

    >>> computer = MorphComputer(gdf, 'population', options=MorphOptions(grid=grid))
    >>> result = computer.morph()
    >>> cartogram, history = result.geometries, result.history

Multi-resolution cartogram for better convergence:

    >>> # Default: returns final MorphResult
    >>> result = multiresolution_morph(gdf, 'population', resolution=512, levels=3)
    >>> cartogram = result.geometries
    >>> grid = result.grid
    >>>
    >>> # Get all level results
    >>> results = multiresolution_morph(gdf, 'population', return_all_levels=True)
    >>>
    >>> # Get MorphComputer for further refinement
    >>> computer = multiresolution_morph(gdf, 'population', return_computer=True)
"""

# Import sub-modules
from . import (
    animation,
    anisotropy,
    comparison,
    density,
    displacement,
    grid,
    history,
    metrics,
    serialization,
    velocity,
    visualization,
)

# Core morphing functions and classes (from split modules)
from .algorithm import morph_geometries
from .api import morph_gdf, multiresolution_morph
from .computer import MorphComputer, RefinementRun
from .options import (
    MorphOptions,
    MorphOptionsConsistencyError,
    MorphOptionsError,
    MorphOptionsValidationError,
    MorphStatus,
)
from .result import MorphResult

# Define public API for explicit control over what is exported
__all__ = [
    "MorphComputer",
    "MorphOptions",
    "MorphOptionsConsistencyError",
    "MorphOptionsError",
    "MorphOptionsValidationError",
    "MorphResult",
    "MorphStatus",
    "RefinementRun",
    "animation",
    "anisotropy",
    "comparison",
    "density",
    "displacement",
    "grid",
    "history",
    "metrics",
    "morph_gdf",
    "morph_geometries",
    "multiresolution_morph",
    "serialization",
    "velocity",
    "visualization",
]
