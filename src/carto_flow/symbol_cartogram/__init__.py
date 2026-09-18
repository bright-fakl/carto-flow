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

Named cartogram functions
------------------------
centroid_cartogram, dorling_cartogram, geographic_cartogram,
dorling_grouped_cartogram, geographic_grouped_cartogram,
demers_cartogram, tile_map_cartogram
    Convenience entry points that wrap ``create_symbol_cartogram`` with
    preset algorithm and styling choices. The first five form a 2x2 grid
    (grouped/not x origin-pull/centroid-pull) plus the Stage-1-only centroid
    variant. Each exposes only the parameters that vary meaningfully for its style.

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

>>> from carto_flow.symbol_cartogram import tile_map_cartogram
>>> result = tile_map_cartogram(gdf)
>>> result.plot(column="category", categorical=True)

"""

from .adjacency import compute_adjacency
from .api import create_layout, create_symbol_cartogram
from .grid import compute_grid_symbol_size, generate_grid
from .layouts import (
    AlgorithmMetrics,
    CentroidLayout,
    CentroidLayoutOptions,
    CentroidMetrics,
    CirclePackingAdvancedOptions,
    CirclePackingLayout,
    CirclePackingLayoutOptions,
    CirclePhysicsLayout,
    CirclePhysicsLayoutOptions,
    FlowDensityLayout,
    FlowDensityLayoutOptions,
    GridBasedLayout,
    GridBasedLayoutOptions,
    GridMetrics,
    Layout,
    LayoutData,
    LayoutResult,
    SimulationHistory,
    Transform,
    compute_symbol_sizes,
    get_layout,
    prepare_layout_data,
    register_layout,
)
from .options import (
    AdjacencyMode,
    ForceMode,
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
    centroid_cartogram,
    demers_cartogram,
    dorling_cartogram,
    dorling_grouped_cartogram,
    geographic_cartogram,
    geographic_grouped_cartogram,
    tile_map_cartogram,
)
from .result import SymbolCartogram
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
    "AlgorithmMetrics",
    "CentroidLayout",
    "CentroidLayoutOptions",
    "CentroidMetrics",
    "CirclePackingAdvancedOptions",
    "CirclePackingLayout",
    "CirclePackingLayoutOptions",
    "CirclePhysicsLayout",
    "CirclePhysicsLayoutOptions",
    "CircleSymbol",
    "ComparisonPlotResult",
    "DisplacementPlotResult",
    "FitMode",
    "FlowDensityHistory",
    "FlowDensityLayout",
    "FlowDensityLayoutOptions",
    "FlowDensityMetrics",
    "ForceMode",
    "GridBasedLayout",
    "GridBasedLayoutOptions",
    "GridMetrics",
    "HexagonSymbol",
    "HexagonTiling",
    "IsohedralTileSymbol",
    "IsohedralTiling",
    "Layout",
    "LayoutData",
    "LayoutResult",
    "PackingHistory",
    "PackingMetrics",
    "PhysicsHistory",
    "PhysicsMetrics",
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
    "centroid_cartogram",
    "compute_adjacency",
    "compute_grid_symbol_size",
    "compute_symbol_sizes",
    "create_circle",
    "create_hexagon",
    "create_layout",
    "create_square",
    "create_symbol_cartogram",
    "create_symbols",
    "demers_cartogram",
    "dorling_cartogram",
    "dorling_grouped_cartogram",
    "generate_grid",
    "geographic_cartogram",
    "geographic_grouped_cartogram",
    "get_layout",
    "plot_adjacency",
    "plot_adjacency_heatmap",
    "plot_comparison",
    "plot_displacement",
    "plot_tiling",
    "prepare_layout_data",
    "register_layout",
    "resolve_symbol",
    "resolve_tiling",
    "tile_map_cartogram",
]
