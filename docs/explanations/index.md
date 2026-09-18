# Explanations

Conceptual guides that explain the underlying principles and design decisions behind carto-flow.

## Flow Cartogram

Conceptual explanations specific to flow-based contiguous cartograms.

### [Algorithm](flow-cartogram-algorithm.md)
The iterative density-equalizing algorithm: mathematical foundation, FFT-based velocity computation, displacement, convergence, multi-resolution morphing, and limitations.

### [History and Snapshot System](flow-cartogram-history.md)
Dual-mode iteration tracking: lightweight `ConvergenceHistory` for scalar metrics at every iteration, and `History` snapshots for full geometry state at selected iterations. Covers `CartogramSnapshot`, `CartogramInternalsSnapshot`, and the `BaseSnapshot` interface.

### [Workflow System](flow-cartogram-workflow.md)
`CartogramWorkflow` and `Cartogram`: the high-level API for GeoDataFrame-based morphing, multi-run state management, the `Cartogram` result object, and serialization.

## Proportional Cartogram

Conceptual explanations specific to proportional (area-based) cartograms.

### [Splitting and Shrinking](proportional-cartogram.md)
Two algorithms for dividing polygon areas by numeric fractions: splitting (disjoint planar cuts, sequential and treemap strategies) and shrinking (concentric shells via negative buffering). Covers the root-finding core mechanism, multi-fraction peeling, normalization modes for batch processing, and a comparison with simple geometric scaling.

## Symbol Cartogram

Conceptual explanations specific to symbol-based cartograms (Dorling-style, tile maps, etc.).

### [Pipeline Overview](symbol-cartogram-pipeline.md)
The four-stage pipeline from GeoDataFrame to `SymbolCartogram`: data preprocessing (centroids, proportional symbol sizes, adjacency matrix), the `Layout` / `LayoutResult` system, the `Styling` system (symbol shapes, per-geometry overrides, fit modes), and the `SymbolCartogram` result object.

### [Grid Layout Algorithm](symbol-cartogram-grid-layout.md)
Hungarian assignment of regions to tiles in a regular grid: the cost matrix (origin distance, compactness, BFS neighbor hops, angular topology), iterative refinement, and post-processing for hole filling and island correction.

### [Circle Packing Layout Algorithm](symbol-cartogram-circle-packing.md)
Two-stage position-based dynamics: Stage 1 resolves overlaps via global expansion and local separation; Stage 2 packs circles using four attraction forces, explicit contact reaction constraints (tangential sliding), and EMA-smoothed step integration.

### [Flow Density Layout Algorithm](symbol-cartogram-flow-density-layout.md)
Divergence field built from anisotropic Gaussian blobs at predicted contact points, solved with an FFT Poisson solve; circle centroids are advected through the resulting velocity field until spacing converges.

### [Tiling System](tiling-system.md)
Regular grids of congruent polygon tiles used as assignment targets for the grid layout: square, hexagonal, triangle, quadrilateral, and the 81 Grünbaum–Shephard isohedral types with customizable edge curves and an interactive design UI.
