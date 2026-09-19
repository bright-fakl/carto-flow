# Mosaic Layout Algorithm

## Overview

The mosaic layout assigns each input region an **exact integer number of tiles** from a regular
tiling that covers the study area. Unlike the grid layout, which assigns exactly one tile per
region and encodes data values through symbol size, the mosaic layout encodes values through
*tile count*: a region with twice the value gets twice as many tiles.

The primary use case is proportional tile maps — colloquially "tilegrams" — where each tile
carries equal weight and the total tile count for a region is the meaningful quantity, such as
parliamentary seat maps.

Source: [layouts/mosaic/\_\_init\_\_.py](https://github.com/bright-fakl/carto-flow/blob/main/src/carto_flow/symbol_cartogram/layouts/mosaic/__init__.py),
[layouts/mosaic/\_assignment.py](https://github.com/bright-fakl/carto-flow/blob/main/src/carto_flow/symbol_cartogram/layouts/mosaic/_assignment.py),
[layouts/mosaic/\_calibration.py](https://github.com/bright-fakl/carto-flow/blob/main/src/carto_flow/symbol_cartogram/layouts/mosaic/_calibration.py)

**Comparison with `GridBasedLayout`**:

| | `GridBasedLayout` | `MosaicLayout` |
|---|---|---|
| Tiles per region | 1 | `tile_count[g]` (integer) |
| Data encoding | symbol size | tile count |
| Typical use | Dorling / tile maps | Tilegrams / seat maps |
| Assignment | per-region (1:1) | slot-expanded (N:M) |

---

## Algorithm Stages

```
Input: GeoDataFrame + tile_count column
   ↓
[Optional] Flow morphing pre-step
   ↓
Component detection (Union-Find)
   ↓
Tile grid calibration (√ gradient descent)
   ↓
Slot-expanded Hungarian assignment
   ↓  ↑  (iterative connectivity repair)
Output: LayoutResult
```

---

## Flow Morphing Pre-step

When `morph=True` (default), the input geometries are first morphed with the flow cartogram
algorithm using `values=counts`:

```python
flow_result = morph_geometries(geometries, values=counts, options=morph_opts)
working_geometries = list(flow_result.latest.geometry)
```

This pre-distorts each region's shape so its area becomes proportional to its target tile count
*before* the tile assignment runs. The result is a better initial geometry-to-tile correspondence,
reducing the number of connectivity repair iterations needed.

The morphing depth is controlled by `morph_options` (default: `MorphOptions(n_iter=100)`).
Setting `morph=False` skips the pre-step and assigns tiles to the original geometries directly.

---

## Component Detection

`prepare_layout_data` derives connected components from the geometry adjacency matrix with
union-find (`geo_utils.prescale.components_from_adjacency`) and stores them on `LayoutData`,
so the layout does not recompute them:

```python
component_labels, components = components_from_adjacency(adjacency_G)
```

Each component is assigned tiles from its own spatial region of the grid, preventing tiles from
one disconnected landmass being assigned to a geographically separate one. True island groups
— regions that share no boundary with any other region — are processed as singleton components.

---

## Tile Grid Calibration

Given the total target count $N = \sum_g \text{counts}[g]$, the calibration step finds the
tile size at which exactly $N$ tiles are *core* tiles - tiles whose intersection with the
study union covers at least `min_overlap_frac` of the tile area (default 0.1, low enough to
catch tiles over narrow peninsulas).

**Analytical estimate**: starting from the relationship between tile area and study area:

$$
\text{tile\_size}_0 = \text{ref\_size} \cdot \sqrt{\frac{\text{study\_area}}{\text{unit\_area} \cdot N}}
$$

where $\text{ref\_size}$ and $\text{unit\_area}$ are properties of the tiling at unit scale.

**Iterative refinement**: because $n_{\text{core}} \propto 1/\text{tile\_size}^2$, the correction
is a simple multiplicative step:

$$
\text{tile\_size}_{k+1} = \text{tile\_size}_k \cdot \sqrt{\frac{n_{\text{core},k}}{N}}
$$

This converges in a few iterations. Because tile counts are integers and oscillate at this
scale, refinement stops once $|n_{\text{core}} - N| \leq 1$, or after 20 passes. During
refinement the study union is simplified to a quarter of the tile size to keep the repeated
intersections cheap; the final tiling is always built against the original geometry.

The tile pool is then split per component and grown outward:

- **Core tiles**: overlap the study union by at least `min_overlap_frac` (about $N$ of them)
- **Ring tiles**: `extra_tile_rings` rings of tile-adjacent neighbours around the core
  (default 1 ring). These carry a high outside penalty, so they act as reserve - selected only
  when cost pressure exhausts the interior. After the solve, any assigned ring tile that has an
  unassigned core neighbour of the same region is swapped back inward, which removes holes.

The Hungarian assignment selects $N$ tiles from core + rings; unused tiles remain unassigned.
`core_tile_indices` and `pool_tile_indices` on the result expose both sets, and
`plot_tiling(show_pool=True)` colours their borders.

---

## Cost Function

The Hungarian assignment minimises a cost $C(g, j)$ for placing geometry $g$ at tile $j$:

$$
C(g, j) = w_d \cdot d_{\text{norm}}(g,j)
         + w_o \cdot \text{outside}(j)
         - w_i \cdot \text{connectivity}(j)
         + w_n \cdot \text{neighbor}(g,j)
$$

with $w_d$ = `distance_weight`, $w_o$ = `outside_penalty`, $w_i$ = `interior_bonus` and
$w_n$ = `neighbor_weight`. Every term is normalised to $[0, 1]$, so the weights are
comparable across datasets.

| Term | Formula | Role |
|------|---------|------|
| $d_{\text{norm}}(g,j)$ | $\|\mathbf{p}_g - \mathbf{t}_j\|^2 \,/\, \max_{g',j'}\|\mathbf{p}_{g'} - \mathbf{t}_{j'}\|^2$ | Pull tiles toward their region's centroid |
| $\text{outside}(j)$ | $1 - \text{area}(j \cap \text{union}) \,/\, \text{area}(j)$ | Penalty for tiles that extend outside the study area |
| $\text{connectivity}(j)$ | fraction of tile $j$'s neighbours that are also pool tiles | Bonus for interior tiles; pushes surplus tiles to the periphery |
| $\text{neighbor}(g,j)$ | distance (or BFS hops with `neighbor_bfs=True`) from $j$ to the tile pools of $g$'s geographic neighbours | Keep neighbouring regions next to each other |

Distances are squared, so the distance term grows quadratically with displacement.
The neighbour term is recomputed from the current assignment after every solve (warm-started
from geometry centroids on the first pass) and rescaled to `neighbor_weight`; setting
`neighbor_weight=0` disables it.

---

## Slot-Expanded Assignment

Each geometry $g$ is replicated into $\text{counts}[g]$ **slots** in the cost matrix.
All slots of the same geometry share the same cost row. The rectangular cost matrix has
shape $(N_c, n_{\text{pool},c})$ per component $c$, where $n_{\text{pool},c} \geq N_c$;
each component is solved independently against its own tile pool.

`scipy.optimize.linear_sum_assignment` solves the resulting linear assignment problem in
$O(N^3)$ time, assigning exactly one tile to each slot and therefore exactly $\text{counts}[g]$
tiles to each geometry.

---

## Connectivity Repair

A single-pass Hungarian solve does not guarantee that tiles assigned to the same geometry
form a contiguous block. The iterative repair loop fixes two types of problems:

**Intra-region disconnection**: tiles belonging to geometry $g$ that are not reachable from
the main tile cluster of $g$ via tile-adjacency edges.

**Inter-region gaps**: pairs of geographically adjacent geometries $(g_1, g_2)$ whose tile
sets share no tile-adjacency edge — a topological neighbourhood relationship is broken.

Iterations are ranked lexicographically by

$$
\bigl(n_{\text{split}},\ n_{\text{disconnected}} \cdot w_{\text{disc}} + n_{\text{gaps}}\bigr)
$$

where $n_{\text{split}}$ is the number of **regions** (with `group_by`: groups) whose tiles do
not form a single connected block, and $w_{\text{disc}} = $ `disconnected_score_weight`
(default 100). The first term decides; the second only separates iterations that split the same
number of regions, prioritising intra-region contiguity over inter-region adjacency.

Ranking by split regions rather than by disconnected tiles matters: an iteration can cut the
number of stray tiles while scattering them over more regions. On US states that is exactly what
used to happen — the tile score preferred an assignment with 9 split states over one with 2.

If the score is non-zero, the cost matrix is modified before the next re-solve:

- **Disconnected tiles**: cost raised by $C_{\max} \cdot$ `disconnected_penalty_mult` (default 10×)
- **Bridge-candidate tiles** at gaps: cost reduced by $C_{\max} \cdot$ `gap_bridge_mult` (default 5×)

The best-scoring assignment across all iterations is returned, and the loop stops early when an
iteration fails to improve the score — in particular an iteration that splits more regions than
the incumbent is never kept. Convergence (score = 0) is declared as soon as all regions are
contiguous and all geographic neighbours share a tile edge.

`MosaicMetrics` reports the outcome: `n_noncontiguous_regions`, `n_split_groups` and
`repair_passes` (solves actually run, not the `max_connectivity_iters` cap).
`AlgorithmMetrics.converged` requires exact counts **and** zero split regions and groups, so it
is `False` on a result that has every tile count right but a region in two pieces.

With `swap_repair_passes > 0`, a final swap-based repair stage
(`geo_utils.contiguity.repair_group_assignment`, shared with the voronoi cartogram) exchanges
tiles between regions while preserving tile counts. It is off by default.

### What is guaranteed, and what is not

- **Guaranteed**: each region receives exactly `tile_count[g]` tiles. The slot expansion makes
  this structural, so `MosaicMetrics.regions_correct == regions_total` for any input that fits
  in the calibrated pool.
- **Best effort**: intra-region contiguity and inter-region adjacency. The repair loop keeps the
  best assignment it finds; on hard inputs some regions can still come out split. Check
  `MosaicMetrics` and the tiling plot before trusting a specific figure.

---

## Groups, islands and pre-scaling

`group_by` assigns one tile per input geometry but enforces contiguity at the group level: all
tiles of one group must form a connected block. Groups are **split at component boundaries**, so
a group that spans a mainland and an island becomes two independent parts, each with the tile
count of its own geometries. Without this split the solver would try to connect tiles across
open water.

`pre_scale=True` (a `create_layout` / `create_symbol_cartogram` argument, not a layout option)
uniformly scales each connected component so its area matches its share of the data before
anything else runs. It only matters for multi-component inputs, where a small island with a
large tile count would otherwise have to borrow tiles from its neighbours' space.

---

## Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tiling` | `"hexagon"` | Tile shape. Any `Tiling` instance or string shorthand. |
| `morph` | `True` | Run flow morphing pre-step to pre-distort geometries. |
| `morph_options` | `None` | `MorphOptions` for the pre-step (default: `n_iter=100`). |
| `tile_size` | `None` | Explicit tile size; bypasses calibration if provided. |
| `spacing` | `0.0` | Gap between symbols as a fraction of tile size (0-1). |
| `extra_tile_rings` | `1` | Rings of adjacent tiles added to each component's pool as reserve. |
| `min_overlap_frac` | `0.1` | Minimum tile/union overlap for a tile to count as core. |
| `hungarian_options` | `None` | `HungarianOptions` for cost weights; defaults shown below. |

**`HungarianOptions` defaults**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `distance_weight` | 1.0 | Centroid distance weight. Fix at 1.0 and tune the others relative to it. |
| `outside_penalty` | 1.0 | Weight on the outside-fraction penalty. |
| `interior_bonus` | 0.5 | Connectivity bonus weight (interior-tile preference). |
| `neighbor_weight` | 0.3 | Weight on the neighbour-proximity term; 0 disables it. |
| `neighbor_bfs` | `False` | Measure neighbour proximity in tile-graph hops instead of distance. |
| `max_connectivity_iters` | 15 | Maximum connectivity repair iterations. |
| `disconnected_penalty_mult` | 10.0 | Cost raise for disconnected tiles (x current max cost). |
| `gap_bridge_mult` | 5.0 | Cost reduction for bridge-candidate tiles (x current max cost). |
| `disconnected_score_weight` | 100 | Tie-break weight per disconnected tile vs. per gap. |
| `swap_repair_passes` | 0 | Passes of the final swap-based repair stage; 0 disables it. |

The cost weights were renamed from `alpha` / `beta` / `delta` to `distance_weight` /
`outside_penalty` / `interior_bonus` before the layout became public; there are no aliases.
