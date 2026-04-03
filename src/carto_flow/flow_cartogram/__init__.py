"""
Core cartogram generation algorithms for flow-based shape deformation.

This module provides algorithms for creating contiguous cartograms where region
areas are proportional to a data variable. It uses a diffusion-based flow
algorithm that smoothly deforms geometries while preserving topology and
spatial relationships.

Functions
---------
morph_gdf
    Primary interface for cartogram generation from GeoDataFrames.
morph_geometries
    Low-level interface working directly with Shapely geometries.
multiresolution_morph
    Progressive multi-resolution morphing for better convergence.

Classes
-------
Cartogram
    Result class with morphed geometries, methods for plotting, export, and analysis.
CartogramWorkflow
    Workflow class for iterative refinement with state management.
MorphOptions
    Configuration dataclass with validation and quality presets.
MorphStatus
    Enum indicating morphing outcome (CONVERGED, STALLED, COMPLETED, ORIGINAL).

Notes
-----
**Quality Presets**

MorphOptions provides convenience presets for common use cases:

- ``MorphOptions.preset_fast()`` - Fewer iterations, faster execution
- ``MorphOptions.preset_balanced()`` - Good quality/speed trade-off
- ``MorphOptions.preset_high_quality()`` - More iterations, tighter tolerance

**Cartogram Attributes**

The Cartogram result object contains:

- ``snapshots`` - History of CartogramSnapshot objects with algorithm state
- ``status`` - MorphStatus enum (CONVERGED, STALLED, COMPLETED, RUNNING, ORIGINAL)
- ``niterations`` - Number of iterations completed
- ``duration`` - Computation time in seconds
- ``options`` - MorphOptions used for computation
- ``internals`` - History of internal state (if save_internals=True)
- ``grid`` - Grid used for computation
- ``target_density`` - Target equilibrium density

Access final results via ``cartogram.latest`` or convenience methods:

- ``cartogram.get_geometry()`` - Morphed geometries
- ``cartogram.get_landmarks()`` - Morphed landmarks (if provided)
- ``cartogram.get_coords()`` - Displaced coordinates (if provided)
- ``cartogram.get_errors()`` - MorphErrors with log and percentage error metrics
- ``cartogram.get_density()`` - Current density values
- ``cartogram.to_geodataframe()`` - Export as GeoDataFrame
- ``cartogram.plot()`` - Visualize the cartogram
- ``cartogram.save()`` - Save to file

**Sub-modules**

- ``velocity`` - Velocity field computation using FFTW
- ``density`` - Density field computation from geometries
- ``displacement`` - Coordinate displacement with numba acceleration
- ``grid`` - Grid utilities for spatial discretization
- ``history`` - Iteration history and snapshot management
- ``anisotropy`` - Anisotropy tensor utilities for velocity modulation
- ``animation`` - Animation generation utilities
- ``visualization`` - Visualization tools
- ``serialization`` - Result serialization
- ``metrics`` - Computation metrics
- ``comparison`` - Result comparison utilities

Examples
--------
Create a population cartogram:

>>> import geopandas as gpd
>>> from carto_flow.flow_cartogram import morph_gdf, MorphOptions
>>> gdf = gpd.read_file("regions.geojson")
>>> cartogram = morph_gdf(gdf, "population", options=MorphOptions.preset_balanced())
>>> cartogram.plot()
>>> print(f"Status: {cartogram.status}")
>>> print(f"Error: {cartogram.get_errors().mean_error_pct:.1f}%")

Multi-resolution morphing for better convergence:

>>> from carto_flow.flow_cartogram import multiresolution_morph
>>> cartogram = multiresolution_morph(gdf, "population", levels=3, resolution=512)

Custom configuration:

>>> options = MorphOptions(
...     grid_size=512,
...     n_iter=100,
...     dt=0.5,
...     mean_tol=0.01,
...     max_tol=0.1,
...     save_history=True,
...     show_progress=True,
... )

Using CartogramWorkflow for iterative refinement:

>>> from carto_flow.flow_cartogram import CartogramWorkflow
>>> workflow = CartogramWorkflow(gdf, column="population")
>>> workflow.morph()                    # Initial morphing
>>> workflow.morph(mean_tol=0.01)       # Refine with tighter tolerance
>>> workflow.pop()                      # Undo last refinement
>>> workflow.reset()                    # Reset to original geometries

Accessing sub-modules:

>>> from carto_flow.flow_cartogram import velocity, density, grid
>>> from carto_flow.flow_cartogram.velocity import VelocityComputerFFTW

See Also
--------
[carto_flow.proportional_cartogram][] : Shape splitting and manipulation algorithms.
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

# Core morphing functions and classes
from .algorithm import morph_geometries
from .anisotropy import (
    BoundaryDecay,
    BoundaryNormalDecay,
    DirectionalTensor,
    LocalizedTensor,
    Multiplicative,
    Pipeline,
    Tensor,
    VelocityModulator,
)
from .anisotropy import Smooth as VelocitySmooth
from .anisotropy import preview_modulator as preview_velocity_modulator
from .api import morph_gdf, multiresolution_morph
from .cartogram import Cartogram
from .density import DensityBorderExtension, DensityModulator, DensityPipeline
from .density import Smooth as DensitySmooth
from .density import preview_modulator as preview_density_modulator
from .errors import MorphErrors, compute_error_metrics
from .history import ConvergenceHistory, ErrorRecord
from .options import (
    MorphOptions,
    MorphOptionsConsistencyError,
    MorphOptionsError,
    MorphOptionsValidationError,
    MorphStatus,
)
from .plot_results import (
    CartogramComparisonResult,
    CartogramPlotResult,
    ConvergencePlotResult,
    DensityFieldResult,
    DensityModulatorPreviewResult,
    ModulatorPreviewResult,
    VelocityFieldResult,
    WorkflowConvergencePlotResult,
)
from .timings import Benchmark
from .visualization import DensityPlotOptions, VelocityPlotOptions
from .workflow import CartogramWorkflow

# Define public API for explicit control over what is exported
__all__ = [
    "Benchmark",
    "BoundaryDecay",
    "BoundaryNormalDecay",
    "Cartogram",
    "CartogramComparisonResult",
    "CartogramPlotResult",
    "CartogramWorkflow",
    "ConvergenceHistory",
    "ConvergencePlotResult",
    "DensityBorderExtension",
    "DensityFieldResult",
    "DensityModulator",
    "DensityModulatorPreviewResult",
    "DensityPipeline",
    "DensityPlotOptions",
    "DensitySmooth",
    "DirectionalTensor",
    "ErrorRecord",
    "LocalizedTensor",
    "ModulatorPreviewResult",
    "MorphErrors",
    "MorphOptions",
    "MorphOptionsConsistencyError",
    "MorphOptionsError",
    "MorphOptionsValidationError",
    "MorphStatus",
    "Multiplicative",
    "Pipeline",
    "Tensor",
    "VelocityFieldResult",
    "VelocityModulator",
    "VelocityPlotOptions",
    "VelocitySmooth",
    "WorkflowConvergencePlotResult",
    "animation",
    "anisotropy",
    "comparison",
    "compute_error_metrics",
    "density",
    "displacement",
    "grid",
    "history",
    "metrics",
    "morph_gdf",
    "morph_geometries",
    "multiresolution_morph",
    "preview_density_modulator",
    "preview_velocity_modulator",
    "serialization",
    "velocity",
    "visualization",
]
