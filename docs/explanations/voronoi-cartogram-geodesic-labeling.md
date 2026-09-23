# Geodesic Labeling in the Voronoi Cartogram

## Overview

The `RasterBackend` labels each pixel in the grid by assigning it to a
generator (geometry centroid). By default this uses **Euclidean
nearest-neighbor**: each pixel goes to the generator with the smallest
straight-line distance. For many datasets this is sufficient, but it fails
for geographies where land regions are separated by water.

Setting `distance_mode="geodesic"` switches to **geodesic labeling**: labels
propagate only through active (land) pixels via a multi-source BFS. Pixels
can never "jump" across inactive (water) regions, so cross-bay or
cross-strait assignments are eliminated.

The implementation is in
[`geodesic.py`](https://github.com/bright-fakl/carto-flow/blob/main/src/carto_flow/voronoi_cartogram/geodesic.py).

---

## Motivation: Why Euclidean Labeling Fails

Consider a coastal dataset where two states share a bay. Euclidean
nearest-neighbor may assign bay pixels to the geographically closer
generator even if reaching that generator requires crossing open water.
The result is a cell that "bleeds" across the bay boundary — incorrect
area attribution and misleading topology.

Example: in a US dataset that includes coastal geometries, Florida's
Atlantic and Gulf pixels might be split by the Euclidean nearest-neighbor
boundary in ways that cross the Florida peninsula — avoided by geodesic
labeling.

---

## Algorithm

### Pixel Grid

The `RasterBackend` rasterises the outer boundary union onto a `resolution × resolution`
grid. Each pixel is classified as:

- **Active** — its center falls inside the outer boundary (land)
- **Inactive** — outside (water / no data)

Only active pixels participate in labeling.

### Seed Snapping

Each generator $\mathbf{p}_i$ must seed the BFS from an active pixel.
The procedure:

1. Project all generators onto the grid to get candidate pixel coordinates.
2. For any generator that falls on an inactive pixel, find the nearest active
   pixel via a `cKDTree` query over all active pixel coordinates.
3. Assign seeds greedily: if two generators map to the same pixel, the
   second is assigned to its next-nearest unoccupied active pixel.

### Multi-Source BFS

A Numba-compiled multi-source BFS (`_bfs` in `geodesic.py`) simultaneously
expands all seed labels through the active pixel graph (4-connected
neighborhood). Each pixel is visited at most once and inherits the label of
the first wave that reaches it.

Because the BFS cannot cross inactive pixels, regions separated by water
get correctly separated labels.

### Unseeded Component Fallback

If the outer boundary is a `MultiPolygon` (e.g. a mainland + islands), some
connected components of the active pixel graph may not contain any generator
seed. Without intervention those pixels would remain unlabeled.

The algorithm detects unseeded components after the initial BFS:

1. Label connected components of the active pixel graph (`skimage.measure.label` or equivalent).
2. For each component that has no seed, compute its center of mass and add an
   extra seed there, inheriting the label of the geographically nearest generator.
3. Re-run a local BFS to fill the unseeded component.

This ensures full coverage even for archipelago datasets.

---

## Decision Guide

| Scenario | Recommended mode |
|---|---|
| Simple convex or compact geometries (e.g. administrative regions with no water bodies) | `distance_mode="euclidean"` (default) |
| Coastal or island datasets where regions are separated by water | `distance_mode="geodesic"` |

```python
import carto_flow.voronoi_cartogram as vor

result = vor.create_voronoi_cartogram(
    gdf, weights="population",
    backend=vor.RasterBackend(distance_mode="geodesic"),
)
```

!!! note
    `area_equalizer_rate` applies only to `distance_mode="euclidean"`.
    The geodesic BFS assigns pixels by wavefront distance, not by a
    power-diagram metric, so the offset term has no effect and is ignored.

---

## Implementation Notes

- The BFS is compiled with Numba (`@njit`) for near-C performance. The
  first call incurs a JIT compilation overhead; subsequent calls are fast.
- For very high `resolution` values (> 800), the BFS dominates runtime.
  For typical values (300–512), the overhead is negligible compared to the
  centroid computation.
- The `distance_mode` parameter is exclusive to `RasterBackend`; `ExactBackend`
  always uses exact shapely geometry and does not need a labeling mode.

---

## Source Reference

- `geodesic_label_active()` — top-level entry point; cKDTree snapping + BFS
- `_bfs()` — Numba-compiled multi-source BFS kernel
- Both in [`src/carto_flow/voronoi_cartogram/geodesic.py`](https://github.com/bright-fakl/carto-flow/blob/main/src/carto_flow/voronoi_cartogram/geodesic.py)
