# How-to Guides

Practical notebooks showing how to accomplish specific tasks with carto-flow. Each guide answers one focused question.

## Flow Cartogram

### [Diagnose and Improve Convergence](diagnose-convergence.ipynb)
Understand what `CONVERGED`, `COMPLETED`, and `STALLED` status mean, read error metrics, and apply the right fix — from iterative refinement with `CartogramWorkflow` to multi-resolution morphing.

### [Save and Export Results](save-and-export-results.ipynb)
Save a cartogram to disk and reload it, or export the morphed geometries to a GeoDataFrame / GeoPackage for use in GIS tools.

### [Reduce Boundary Distortion](reduce-boundary-distortion.ipynb)
Use `prescale_components`, `DensityBorderExtension`, anisotropy fields, and velocity modifiers like `BoundaryDecay` to control distortion at the edges of your cartogram.

### [Animate the Morphing Process](create-animation.ipynb)
Capture intermediate frames with `snapshot_every`, build an animation with `animate_morph_history()`, and save to GIF or MP4.

### [Co-Morph Landmarks and Coordinates](co-morph-landmarks-and-coordinates.ipynb)
Displace cities, annotations, or point coordinates consistently with the cartogram using `landmarks` (GeoDataFrames) or `displacement_coords` (NumPy arrays).

## Proportional Cartogram

### [Apply Partitions to a Flow Cartogram](partitions-on-flow-cartogram.ipynb)
Combine the two modules: morph geometries with the flow cartogram, then apply proportional splits or shrinks to show a second variable within each region.

## Symbol Cartogram

### [Style Symbols by Category](style-symbols-by-category.ipynb)
Separate layout computation from styling: map data columns to colors, shapes, opacity, hatching, and labels using `plot()` keyword arguments.

### [Create a Tile-Count Map](tile-count-maps.ipynb)
Use `tile_count` to assign each region an integer number of tiles (e.g. parliamentary seats), export group-level union geometries with `to_geodataframe(level="group")`, and apply per-group symbol shapes with `Styling.set_group_symbol()`.

## Voronoi Cartogram

### [Choose the Right Backend](voronoi_cartogram/choose-backend.ipynb)
Compare `RasterBackend` (fast, pixel-based) and `ExactBackend` (exact Voronoi geometry, no weights support) with a timing and accuracy comparison and a decision table.

### [Coastal and Peninsula Regions](voronoi_cartogram/coastal-regions.ipynb)
Prevent pixel labels from crossing bays and straits by switching to `distance_mode="geodesic"` — a BFS that propagates only through active (land) pixels.

### [Inspect and Improve Convergence](voronoi_cartogram/inspect-convergence.ipynb)
Read `metrics`, plot convergence curves and area errors, visualise displacement, and tune `n_iter`, `area_cv_tol`, `prescale_components`, and the relaxation schedule.

### [Analyze and Fix Topology](voronoi_cartogram/topology.ipynb)
Detect satellite cells in grouped datasets with `plot_topology()`, and repair them post-hoc with `repair_topology()` or automatically via `TopologyRepair` in `VoronoiOptions`.

### [Customize the Boundary](voronoi_cartogram/boundaries.ipynb)
Change the outer clipping shape (`"union"`, `"bbox"`, `"circle"`, custom geometry) and choose between fixed, `AdhesiveBoundary`, and `ElasticBoundary` generator behaviour.
