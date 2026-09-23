# Contiguity Repair in the Voronoi Cartogram

## Overview

When a dataset contains **grouped sub-regions** — for example, congressional
districts grouped by state — the Voronoi relaxation may produce **satellite
cells**: Voronoi cells that belong to a group but are geometrically
disconnected from the rest of that group's cells. A satellite arises when a
generator for district A drifts close enough to district B's territory that
an intervening district C "clips" district A's cell in two.

Contiguity repair detects these satellites and corrects them by permuting
generator positions so that each group's cells form a contiguous region.

The implementation is in
[`contiguity.py`](https://github.com/bright-fakl/carto-flow/blob/main/src/carto_flow/voronoi_cartogram/contiguity.py).

---

## Problem Statement

Given:
- A set of Voronoi cells (one per district) partitioned into groups (states)
- An adjacency graph on cells derived from shared edges

A group has a **satellite** if its cells form more than one connected
component in the adjacency graph. The satellite components are all
components except the largest (the "main body").

Naive approaches (e.g. re-initializing dislocated generators) tend to
oscillate. The slot-permutation approach avoids this by operating directly
on the assignment of generators to slots, preserving area balance.

---

## Algorithm: Slot Permutation

The repair works on a **slot** abstraction: slot $s$ holds a generator with
a fixed position, and the group assignment of slot $s$ can be changed by
swapping it with another slot.

### Step 1 — Build the Adjacency Graph

`find_adjacent_pairs(cells)` (from `geo_utils/adjacency.py`) identifies
pairs of cells that share an edge (within a small tolerance). This gives
a cell-level adjacency graph $G$.

### Step 2 — Find Satellites

For each group, run a BFS on $G$ restricted to that group's slots. Groups
with more than one connected component have satellites.

### Step 3 — Route a Swap Path

For each satellite component, find a path in the full adjacency graph $G$
from any cell in the satellite to any cell in the group's main body. The
path traverses cells of **other** groups.

Along this path, slots are swapped one by one, moving the satellite's label
toward the main body. After each swap the group now occupies a different
slot; the swap cascade terminates when the satellite merges with the main body.

### Step 4 — Articulation-Point Guard

Before performing a swap, the algorithm checks whether the swap would
**disconnect** another group (i.e. whether the cell being vacated is an
articulation point in the other group's subgraph). If so, the swap is
skipped and an alternative path is tried.

This guard prevents "solving" one satellite at the cost of creating a new
one in a different group.

### Step 4.5 — Compactness Enhancement

After contiguity is restored, a second pass improves the spatial compactness of
each group via boundary swaps between adjacent groups. Each candidate swap is
accepted only when it reduces a combined inertia-plus-shared-edge-length metric
**and** does not disconnect either involved group (articulation-point check
applied again). This stage runs automatically whenever `group_by` is supplied;
it has no separate flag in `TopologyRepair`.

### Step 5 — Generator Relocation

After permuting slots (which moves group assignments), the generator
**positions** are also moved. For a 2-swap between slots $d$ and $d'$,
the new position is computed via **intersection-midpoint mirroring**:

1. Cast rays from $\mathbf{p}_d$ and $\mathbf{p}_{d'}$ toward each other's
   cell boundary.
2. Find the intersection points $f_a$, $f_b$ of the rays with those boundaries.
3. Place the new generator at $2 \cdot \text{midpoint}(f_a, f_b) - \mathbf{p}_d$.

This mirrors the generator across the shared boundary, reducing overshoot for
large ↔ small cell swaps. For degenerate cases (rays miss the cell) it
falls back to a simple position swap.

### Iteration

Passes repeat until no satellites remain or `max_passes` is reached.
A "lock" set prevents re-swapping the same slot in the same pass, avoiding
oscillation.

---

## API

### Automated Repair During Iteration

Pass a `TopologyRepair` object to `VoronoiOptions.fix_topology`:

```python
import carto_flow.voronoi_cartogram as vor

options = vor.VoronoiOptions(
    n_iter=100,
    fix_topology=vor.TopologyRepair(
        every=5,          # run repair every 5 iterations
        max_passes=3,     # up to 3 swap passes per repair step
        group_contiguity=True,  # fix satellite cells
        adjacency=True,         # fix violated adjacency
        orientation=True,       # fix misaligned generator orientation
    ),
)

result = vor.create_voronoi_cartogram(
    districts_gdf,
    weights="population",
    group_by="state",
    options=options,
)
```

When `fix_topology` is set, repair runs automatically at iteration multiples
of `every`. Running repair periodically during iteration (rather than only at
the end) allows the relaxation to recover from the permutation and reach a
better final state.

### Post-hoc Repair

Call `repair_topology()` on the result after the main run:

```python
result = vor.create_voronoi_cartogram(
    districts_gdf,
    weights="population",
    group_by="state",
)

report = result.repair_topology("state")
result = report.cartogram

# Inspect what was fixed
print(f"Discontiguous groups remaining: {report.after.n_discontiguous_groups}")
print(f"Violated adjacency pairs: {report.after.n_violated_adjacency}")
```

`TopologyRepairReport` contains:
- `.cartogram` — the repaired `VoronoiCartogram`
- `.before` — `TopologyAnalysis` snapshot taken before repair
- `.after` — `TopologyAnalysis` snapshot taken after repair

### Standalone Function

`make_groups_contiguous()` operates directly on a GeoDataFrame of Voronoi
cells without going through the cartogram pipeline:

```python
from carto_flow.voronoi_cartogram import make_groups_contiguous

fixed_gdf, remaining_satellites = make_groups_contiguous(
    voronoi_cells_gdf,
    group_by="state",
    max_passes=20,
    show_progress=True,
)
print(f"Unresolved satellites: {len(remaining_satellites)}")
```

`remaining_satellites` is a list of `(group_id, row_indices)` tuples for
any components that could not be merged (e.g. topological dead-ends).

---

## Topology Analysis

`TopologyAnalysis` (from `result.py`) reports three types of topology issues:

| Issue | Attribute | Description |
|---|---|---|
| Discontiguous groups | `discontiguous_groups` | Groups with satellite cells |
| Violated adjacency | `violated_adjacency` | Pairs that should be adjacent but share no edge |
| Misaligned orientation | `misaligned_orientation` | Pairs where relative generator orientation does not match expected adjacency direction |

All three can be repaired by `TopologyRepair` with the corresponding flags
(`group_contiguity`, `adjacency`, `orientation`).

---

## Source Reference

- `_compose_topology_permutation()` — assembles the full permutation from satellite detection + path routing
- `make_groups_contiguous()` — public standalone entry point
- Both in [`src/carto_flow/voronoi_cartogram/contiguity.py`](https://github.com/bright-fakl/carto-flow/blob/main/src/carto_flow/voronoi_cartogram/contiguity.py)
- `TopologyRepair` — [`src/carto_flow/voronoi_cartogram/options.py`](https://github.com/bright-fakl/carto-flow/blob/main/src/carto_flow/voronoi_cartogram/options.py)
- `TopologyAnalysis`, `TopologyRepairReport` — [`src/carto_flow/voronoi_cartogram/result.py`](https://github.com/bright-fakl/carto-flow/blob/main/src/carto_flow/voronoi_cartogram/result.py)
