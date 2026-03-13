# Grid Layout Algorithm

## Overview

The grid-based layout assigns each input region to a tile in a regular grid — square, hexagonal, or any isohedral tiling — such that the total assignment cost is minimized. Unlike the physics-based layouts, it involves no iterative simulation; positions are determined by solving a combinatorial assignment problem using the Hungarian algorithm.

Source: [layout.py](https://github.com/bright-fakl/carto-flow/blob/main/src/carto_flow/symbol_cartogram/layout.py) (`GridBasedLayout`), [placement.py](https://github.com/bright-fakl/carto-flow/blob/main/src/carto_flow/symbol_cartogram/placement.py) (`assign_to_grid_hungarian`, `fill_internal_holes`, `_fix_island_assignments`)

---

## Assignment as an Optimization Problem

Given $n$ regions with centroids $\mathbf{p}_i$ and $m \geq n$ tiles with centers $\mathbf{t}_j$, the algorithm finds a bijection $\sigma : \{1 \ldots n\} \to \{1 \ldots m\}$ minimizing total cost:

$$
\min_\sigma \sum_{i=1}^{n} C\!\left(i,\, \sigma(i)\right)
$$

The cost matrix $C(i, j)$ is a weighted sum of four independent terms, each normalized to $[0, 1]$ before combination:

| Term | Formula | Default weight |
|------|---------|---------------|
| **Origin** | $\|\mathbf{p}_i - \mathbf{t}_j\|^2$ | `origin_weight = 0.5` |
| **Compactness** | $\|\mathbf{t}_j - \bar{\mathbf{t}}\|^2$ | `compactness = 0.5` |
| **Neighbor** | $\max(\text{bfs\_hops}(j,\, t_k) - 1,\; 0)^2$ | `neighbor_weight = 0.3` |
| **Topology** | $\min(\|\theta_{ik} - \theta_{jl}\|,\; 2\pi - \|\theta_{ik} - \theta_{jl}\|) / \pi$ | `topology_weight = 0.2` |

where $\bar{\mathbf{t}}$ is the grid centroid, $\text{bfs\_hops}(j, t_k)$ is the tile-adjacency hop distance from candidate tile $j$ to the tile $t_k$ currently assigned to neighboring region $k$, and $\theta_{ik}$ and $\theta_{jl}$ are the geographic and grid bearings for an adjacent pair $(i, k)$.

The assignment is solved with `scipy.optimize.linear_sum_assignment`, which implements the Hungarian algorithm in $O(n^3)$ time.

---

## Origin Cost

The squared Euclidean distance from region centroid $\mathbf{p}_i$ to tile center $\mathbf{t}_j$:

$$
C^{\text{origin}}(i, j) = \frac{\|\mathbf{p}_i - \mathbf{t}_j\|^2}{\max_{i,j}\|\mathbf{p}_i - \mathbf{t}_j\|^2}
$$

This is the only cost term that is non-zero even before any regions are assigned, so it is also used to compute the initial assignment from which the iterative refinement starts.

## Compactness Cost

The squared distance from tile $j$ to the grid centroid $\bar{\mathbf{t}}$:

$$
C^{\text{compact}}(j) = \frac{\|\mathbf{t}_j - \bar{\mathbf{t}}\|^2}{\max_j \|\mathbf{t}_j - \bar{\mathbf{t}}\|^2}
$$

This term is the same for all regions, pulling assignments toward the center of the grid.

---

## Neighbor Cost (Iterative Refinement)

The neighbor and topology costs depend on the current assignment, so the full cost cannot be evaluated before any assignment exists. The algorithm uses iterative refinement:

1. Solve the initial assignment using only the origin cost.
2. Compute the neighbor and topology costs based on the current assignment.
3. Solve the full combined cost.
4. Repeat steps 2–3 until the assignment stops changing, or after 10 iterations.

### BFS Hop Distance

For each adjacent region pair $(i, k)$ and candidate tile $j$, the neighbor cost penalizes placing region $i$ far (in tile-adjacency hops) from the tile currently assigned to neighbor $k$:

$$
C^{\text{neigh}}(i, j) \mathrel{+}= w_{ik} \cdot \max\!\left(\text{hops}(j,\, \sigma(k)) - 1,\; 0\right)^2
$$

where $w_{ik}$ is the adjacency weight for pair $(i, k)$. Edge-adjacent tiles (1 hop) incur zero cost; vertex-only adjacent tiles receive a fixed penalty of 0.2; non-adjacent tiles incur quadratically growing cost.

### Reverse Neighbor Penalty

An additional penalty discourages placing region $i$ in a tile whose existing tile-neighbors are occupied by regions that are not geographic neighbors of $i$:

$$
C^{\text{rev}}(i, j) = \sum_{l \in \text{tile-neighbors}(j)} \mathbf{1}\!\left[\sigma^{-1}(l) \not\sim i\right]
$$

where $\sigma^{-1}(l)$ is the region currently in tile $l$ and $\not\sim$ means "not geographically adjacent."

---

## Topology (Angular Orientation) Cost

For each adjacent pair $(i, k)$, the original geographic bearing from $\mathbf{p}_i$ to $\mathbf{p}_k$ is compared to the grid bearing from candidate tile $j$ to the tile of $k$. The angular error, mapped to $[0, 1]$, penalizes reversals of neighbor directions:

$$
C^{\text{topo}}(i, j) \mathrel{+}= w_{ik} \cdot \frac{\min\!\left(|\theta_{ik} - \theta_{jl}|,\; 2\pi - |\theta_{ik} - \theta_{jl}|\right)}{\pi}
$$

---

## Combined Cost

The four terms are combined as:

$$
C(i, j) = w^{\text{origin}} \cdot C^{\text{origin}} + w^{\text{neigh}} \cdot \left(C^{\text{neigh}} + C^{\text{rev}}\right) + w^{\text{topo}} \cdot C^{\text{topo}} + w^{\text{compact}} \cdot C^{\text{compact}}
$$

---

## Post-Processing

After assignment, two correction passes address mismatches between the grid topology and the geographic topology.

### Hole Filling

The union of all input polygons may have interior holes (lakes, enclaves). Grid cells that fall inside such holes should remain unassigned. However, if the number of unassigned interior tiles exceeds the number of geographic holes, `fill_internal_holes()` fills the excess by shifting assignment chains.

The algorithm detects interior tiles (not reachable from the grid boundary via unassigned tiles), groups them into connected components, and fills the smallest excess components by BFS: it finds a path from the hole through assigned tiles to a boundary tile, then shifts each region along the path one step, vacating the boundary tile.

### Island Correction

`_fix_island_assignments()` ensures consistency between geographic island/connectivity structure and tile connectivity:

- **Connect non-islands**: if geographically connected regions are assigned to tiles that form multiple disconnected tile-components, the smaller components are reassigned to tiles adjacent to the main cluster (preferring tiles close to geographic neighbors).
- **Disconnect true islands**: if a geographically isolated region (true island) is assigned to a tile edge-adjacent to the main tile cluster, it is reassigned to an unassigned tile that is not adjacent to the main cluster.

---

## Output

Each assigned tile's rotation and reflection are inherited from the tiling geometry and stored in the region's `Transform`. The symbol scale is set proportionally:

$$
\text{scale}_i = \frac{\text{effective\_native\_size}_i}{\text{base\_native\_size}}
$$

where the native sizes are computed from the area-equivalent radii divided by the canonical symbol's area factor.

---

## Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tiling` | `"hexagon"` | Tiling type (string, Tiling instance, or preset name) |
| `origin_weight` | 0.5 | Weight for origin distance cost |
| `compactness` | 0.5 | Weight for grid centrality cost |
| `neighbor_weight` | 0.3 | Weight for neighbor hop and reverse-neighbor costs |
| `topology_weight` | 0.2 | Weight for angular orientation cost |
| `spacing` | 0.0 | Gap between symbols as fraction of tile size |
| `rotation` | 0.0 | Grid rotation in degrees |
| `fill_holes` | `True` | Enable interior hole filling |
| `fix_islands` | `True` | Enable island/connectivity correction |
