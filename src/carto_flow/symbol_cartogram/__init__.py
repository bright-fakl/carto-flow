"""Symbol Cartogram Module
=======================

Create cartograms where each geographic region is represented by a single symbol.
Symbol size can be proportional to a data value or uniform, and placement can be
free (with overlap resolution) or grid-constrained.

Main Function
-------------
create_symbol_cartogram
    Create a symbol cartogram from a GeoDataFrame.

Layout-Styling Separation
-------------------------
Layout
    Abstract base for layout algorithms.
PhysicsBasedLayout
    Layout from physics-based simulation.
GridBasedLayout
    Layout from grid-based assignment.
CentroidLayout
    Layout that places symbols at geometry centroids.
LayoutResult
    Immutable output from layout algorithms.
LayoutData
    Preprocessed data for layout algorithms.
Transform
    Transformation applied to a symbol.
prepare_layout_data
    Prepare data for layout algorithms.

Configuration
-------------
GridPlacementOptions
    Options for grid-based placement.
PhysicsSimulatorOptions
    Options for CirclePhysicsSimulator.
TopologySimulatorOptions
    Options for TopologyPreservingSimulator.
CentroidLayoutOptions
    Options for centroid-based placement.
SymbolShape
    Shape of the symbols (CIRCLE, SQUARE, HEXAGON).
AdjacencyMode
    How adjacency is computed (BINARY or WEIGHTED).

Presets
-------
preset_dorling, preset_topology_preserving, preset_demers, preset_tile_map,
preset_fast, preset_quality
    Factory functions returning kwargs dicts for ``create_symbol_cartogram``.

Result
------
SymbolCartogram
    Result container with symbol geometries and metrics.
SimulationHistory
    Per-iteration diagnostics and optional position snapshots.
SymbolCartogramStatus
    Computation status (CONVERGED, COMPLETED, ORIGINAL).

Visualization
-------------
plot_adjacency
    Visualize the adjacency graph overlaid on the cartogram.
plot_adjacency_heatmap
    Render the adjacency matrix as a heatmap.
plot_comparison
    Side-by-side comparison of original geometries and symbols.
plot_displacement
    Plot displacement arrows from original centroids to symbol centers.

Plot Results
------------
SymbolsPlotResult
    Artists returned by ``plot_symbols()`` / ``SymbolCartogram.plot()``.
AdjacencyPlotResult
    Artists returned by ``plot_adjacency()``.
DisplacementPlotResult
    Artists returned by ``plot_displacement()``.
TilingPlotResult
    Artists returned by ``plot_tiling()``.
TilingGridPlotResult
    Artists returned by ``TilingResult.plot()``.
PrototilePlotResult
    Artists returned by ``TilingResult.plot_tile()`` / ``Tiling.plot_tile()``.
AdjacencyHeatmapResult
    Artists returned by ``plot_adjacency_heatmap()``.
ComparisonPlotResult
    Artists returned by ``plot_comparison()``.

Utilities
---------
compute_symbol_sizes
    Compute symbol sizes from data values.
generate_grid
    Generate a regular grid of cells.
compute_adjacency
    Compute adjacency matrix from polygon geometries.
create_circle, create_square, create_hexagon
    Create symbol polygons.

Examples
--------
Classic Dorling cartogram (proportional circles, free placement):

>>> from carto_flow.symbol_cartogram import create_symbol_cartogram
>>> result = create_symbol_cartogram(gdf, "population")
>>> result.plot(column="population", cmap="Reds")

Using a preset:

>>> from carto_flow.symbol_cartogram.presets import preset_tile_map
>>> result = create_symbol_cartogram(gdf, **preset_tile_map())
>>> result.plot(column="category", categorical=True)

"""

from .adjacency import compute_adjacency
from .api import create_layout, create_symbol_cartogram
from .data_prep import (
    LayoutData,
    compute_symbol_sizes,
    prepare_layout_data,
)
from .grid import compute_grid_symbol_size, generate_grid
from .layout import (
    CentroidLayout,
    CirclePackingLayout,
    CirclePhysicsLayout,
    GridBasedLayout,
    Layout,
    get_layout,
    register_layout,
)
from .layout_result import LayoutResult, Transform
from .options import (
    AdjacencyMode,
    CentroidLayoutOptions,
    CirclePackingLayoutOptions,
    CirclePhysicsLayoutOptions,
    ForceMode,
    GridBasedLayoutOptions,
    SymbolOrientation,
    SymbolShape,
)
from .plot_results import (
    AdjacencyHeatmapResult,
    AdjacencyPlotResult,
    ComparisonPlotResult,
    DisplacementPlotResult,
    PrototilePlotResult,
    SymbolsPlotResult,
    TilingGridPlotResult,
    TilingPlotResult,
)
from .presets import (
    preset_demers,
    preset_dorling,
    preset_fast,
    preset_quality,
    preset_tile_map,
    preset_topology_preserving,
)
from .result import SimulationHistory, SymbolCartogram
from .status import SymbolCartogramStatus
from .styling import FitMode, Styling
from .symbols import (
    CircleSymbol,
    HexagonSymbol,
    IsohedralTileSymbol,
    SquareSymbol,
    Symbol,
    SymbolParam,
    SymbolSpec,
    TileSymbol,
    TransformedSymbol,
    create_circle,
    create_hexagon,
    create_square,
    create_symbols,
    resolve_symbol,
)
from .tiling import (
    HexagonTiling,
    IsohedralTiling,
    QuadrilateralTiling,
    SquareTiling,
    TileAdjacencyType,
    TileTransform,
    Tiling,
    TilingResult,
    TriangleTiling,
    resolve_tiling,
)
from .visualization import plot_adjacency, plot_adjacency_heatmap, plot_comparison, plot_displacement, plot_tiling

__all__ = [
    "AdjacencyHeatmapResult",
    "AdjacencyMode",
    "AdjacencyPlotResult",
    "CentroidLayout",
    "CentroidLayoutOptions",
    "CirclePackingLayout",
    "CirclePackingLayoutOptions",
    "CirclePhysicsLayout",
    "CirclePhysicsLayoutOptions",
    "CircleSymbol",
    "ComparisonPlotResult",
    "DisplacementPlotResult",
    "FitMode",
    "ForceMode",
    "GridBasedLayout",
    "GridBasedLayoutOptions",
    "HexagonSymbol",
    "HexagonTiling",
    "IsohedralTileSymbol",
    "IsohedralTiling",
    "Layout",
    "LayoutData",
    "LayoutResult",
    "PrototilePlotResult",
    "QuadrilateralTiling",
    "SimulationHistory",
    "SquareSymbol",
    "SquareTiling",
    "Styling",
    "Symbol",
    "SymbolCartogram",
    "SymbolCartogramStatus",
    "SymbolOrientation",
    "SymbolParam",
    "SymbolShape",
    "SymbolSpec",
    "SymbolsPlotResult",
    "TileAdjacencyType",
    "TileSymbol",
    "TileTransform",
    "Tiling",
    "TilingGridPlotResult",
    "TilingPlotResult",
    "TilingResult",
    "Transform",
    "TransformedSymbol",
    "TriangleTiling",
    "compute_adjacency",
    "compute_grid_symbol_size",
    "compute_symbol_sizes",
    "create_circle",
    "create_hexagon",
    "create_layout",
    "create_square",
    "create_symbol_cartogram",
    "create_symbols",
    "generate_grid",
    "get_layout",
    "plot_adjacency",
    "plot_adjacency_heatmap",
    "plot_comparison",
    "plot_displacement",
    "plot_tiling",
    "prepare_layout_data",
    "preset_demers",
    "preset_dorling",
    "preset_fast",
    "preset_quality",
    "preset_tile_map",
    "preset_topology_preserving",
    "register_layout",
    "resolve_symbol",
    "resolve_tiling",
]
