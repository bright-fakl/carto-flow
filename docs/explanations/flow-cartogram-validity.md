# Output Validity

The flow cartogram moves the vertices of the input polygons and nothing else. This page explains why that can produce polygons that cross themselves, and how the two repair methods of `MorphOptions.validity` work. The options are listed in [MorphOptions](flow-cartogram-algorithm.md#morphoptions).

## The problem

Four words are used on this page.

- A **segment** is the straight line between two neighboring vertices of a ring.
- The **boundary after the flow** is the curve that the points of a segment form after the flow has moved them.
- The **chord** is the straight line between the new positions of the two vertices of a segment after they moved.
- The **sagitta** is the largest distance between the chord and the boundary after the flow.

Before the morph a boundary segment is a straight line between two vertices. The flow moves every point of that line, not only the two vertices, and the points end up on a curve. The morph moves only the vertices and draws a straight line between their new positions (the chord), so the chord can differ from the curve that the boundary would follow, and where the boundary of a region comes close to another boundary the chord can cross it.

For a boundary after the flow with curvature $\kappa$ and a chord of length $\ell$, the sagitta is about

$$
s \approx \frac{\kappa\,\ell^{2}}{8}.
$$

A boundary that comes closer to the curve than the sagitta, on the side of the chord, can cross the chord. Two cases are typical. In a region that is squeezed into a strip narrower than the sagitta of its boundary segments, the other boundary of the strip lies between the chord and the curve. A region that is enlarged stretches its segments, and the sagitta grows with the square of the length $\ell$.

![A squeezed region between two regions before and after the flow: a long segment on its upper boundary becomes a curve after the flow, and its chord crosses the finer lower boundary in two places](../generated/diagrams/validity-chord-and-path.svg)

*Figure 1. The upper boundary of the squeezed region is one long segment, the lower boundary has shorter segments. After the flow the region is valid (the green curves do not touch), but the chord of the long segment lies below the curve and crosses the chords of the lower boundary.*

A flow path is the curve that a single point follows while the velocity field moves it. The boundary after the flow is made of the end points of the flow paths of all points of the segment. The repair described below uses flow paths: a vertex that is inserted into a segment is moved along its own flow path and so lands on the boundary after the flow.

## Choosing a value

`MorphOptions.validity` takes one of the following values. They are mutually exclusive.

| Value | What it does | Output | Use it when |
|---|---|---|---|
| `"check"` (default) | Tests the returned best iterate once, at the end of the run. The invalid regions are listed in `Cartogram.invalid_regions` and the run warns. The geometry is not changed. | The vertices of the input. | The input has vertices dense enough for the flow (no warning about long segments), or the vertex set of the output has to equal that of the input. |
| `"repair-crossings"` | Stores the velocity fields of a span of windows, tests all segments for crossings at the end of the span, splits the crossing segments and moves the new vertices along their flow paths ([below](#repair-crossings)). | The vertices of the input and the inserted vertices. | `"check"` reports invalid regions. |
| `"repair-quadtree"` | Builds a mesh of triangles from the density, inserts vertices so that every segment lies inside one triangle, and moves the vertices with the mesh: only the nodes of the mesh are advected, and a vertex keeps its position inside its triangle ([below](#repair-quadtree)). No crossing test, no stored fields. | The vertices of the input and the inserted vertices, more of them than with `"repair-crossings"`. Inserted vertices that are not needed any more are removed. | The input has segments much longer than the grid cells. |

`validity_repair_every` (windows per span) and `validity_memory_mb` (memory budget for the stored fields) apply to `"repair-crossings"` only. With `"check"` and `"repair-quadtree"` they are validated and unused.

Densify coarse input first (`carto_flow.geo_utils.densify_coverage`, or `max_segment_length` of the geometry loaders): short segments have a small sagitta, so fewer crossings arise and fewer vertices have to be inserted. The repair works on the crossings that remain. `"repair-quadtree"` is the choice for coarse input that is not densified: it does not depend on finding crossings, because the mesh gives every segment that leaves a triangle a vertex at the triangle edge ([below](#repair-quadtree)).

## `"repair-crossings"`

### Windows and spans

The velocity field is recomputed from the density every few iterations. A **window** is the span of iterations between two recomputes of the field. Inside a window the field is fixed, so every vertex moves independently of all others, and the path of a vertex through the window is known. A **span** is `validity_repair_every` windows (default 10). The crossing test and the repair run at the end of a span.

To repair a span afterwards, the morph has to be able to repeat the movement of a new vertex. It therefore keeps, for the current span, the coordinates at its start, the field of each window (as float32) and the step size of each iteration.

![A timeline of eleven windows in three spans with the stored fields below, the memory budget that ends a span early, and a run in which the cap on inserted vertices is reached](../generated/diagrams/validity-spans-and-memory.svg)

*Figure 2. Top: windows grouped in spans; a check runs between two spans and at the end of the run. Middle: the fields stored at the end of each window; they are released at a check. Here `validity_repair_every` is 5 and the memory budget is reached with four stored fields, so the spans end after four windows. Bottom: the cap on the number of inserted vertices.*

### The check

At the end of a span:

1. **The crossing test.** A uniform grid is built over all segments of all regions together (cell size six times the mean segment length), and the segments that share a cell are tested against each other. A region that overlaps a neighbor is found as well as a self-intersection. Two segments cross when the end points of each lie strictly on opposite sides of the other. Segments that share an end point, touch, are collinear, or are identical (the two sides of a shared boundary) do not count.
2. **The split.** Every crossing segment is split at the midpoint of its chord at the start of the span. Segments that are identical to a crossing segment (the other side of a shared boundary) are split too, and the end points of the chord are put in a fixed order before the midpoint is taken, so the two regions of a shared boundary get bit-identical new vertices. The split does not move or remove the original vertices. A segment whose chord at the start is shorter than $2 \cdot 10^{-7}$ grid cells is not split.
3. **The replay.** The new vertex is moved through the recorded steps of the span with the stored fields, in the same order and with the same step sizes as the other vertices. It ends on its flow path, and so on the boundary after the flow, and not on the chord.
4. **The rounds.** The new segments are tested again, together with the segments near the crossings, and the crossing ones are split in the same way. A check runs at most 14 rounds. It stops earlier when no crossing is left, when no segment can be split, or when the number of crossing segments has grown by a factor of 1.4 in each of the last three rounds, which is what a fold of the flow itself looks like.

![A crossing of two chords at the end of a span, and the same boundaries after the midpoints of the chords at the start of the span were inserted and moved along their flow paths](../generated/diagrams/validity-split-and-replay.svg)

*Figure 3. Left: the chords at the start of the span with the midpoints of the segments that will cross. Middle: at the end of the span the chords cross. Right: the inserted vertices (orange) lie on the green curves, not on the chords (the hollow marker is the midpoint of the chord at the end, for comparison). The new chords do not cross.*

The run is also checked when it ends (converged, stalled or at `n_iter`), whatever its position in the span. The returned best iterate is checked as well. If it lies in the span, its new vertices are replayed to the iteration of the best iterate. If it lies in an earlier span, a new vertex is placed at the midpoint of its two neighbors, so the best iterate keeps its shape. Every insertion round is recorded in `Cartogram.vertex_lineage` ([lineage](#lineage-and-animations)).

### Memory budget

A stored field has two components of $n_x n_y$ cells in float32:

$$
2 \cdot 4 \cdot n_x n_y \ \text{bytes}.
$$

`validity_memory_mb` (default 128, in units of $2^{20}$ bytes) is the budget for the fields of a span. When the stored fields have reached the budget at the start of a window, the span ends early: the crossing test and the repair run, the stored fields are released and a new span starts. The number of windows between two checks is the smaller of `validity_repair_every` and the number of fields that reach the budget, so it is shorter than `validity_repair_every` on a fine grid or with a small budget.

A window's field is always stored. The memory of the check is therefore bounded by the budget plus one field plus one copy of the coordinates, whatever `recompute_every` and `validity_repair_every` are.

Short spans cost time, because every check tests all segments. A check after one or two windows also finds crossings that the next window would remove, which costs accuracy and time. A budget that holds only one or two fields of the finest grid is not useful: for `validity_repair_every` windows per span the budget has to be at least `validity_repair_every` times the field size, or `validity_repair_every` has to be lowered to what the budget holds.

### Cap on inserted vertices

The number of vertices that a run inserts is capped at `max(5000, n // 100)`, with `n` the number of input vertices; this is independent of the memory budget. A crossing that the flow itself creates (a fold) is found again by every check and splitting does not remove it, so without the cap the insertions would not end. When a check would exceed the cap, it stops before that round and the run warns. The run then continues as with `"check"`: no further fields are stored, no further check or repair runs, and the polygons that are invalid at the end are listed in `Cartogram.invalid_regions`.

### What remains invalid

The test detects crossings of segments only. A region that remains invalid is listed in `Cartogram.invalid_regions`: a fold that the flow itself creates (splitting does not remove it), zero-width features of the input such as a cusp, or a hole that leaves its shell without crossing it. The cap can also leave regions invalid.

## `"repair-quadtree"`

`"repair-crossings"` finds crossings after they have happened and repairs them. `"repair-quadtree"` does not test for crossings. It changes how the vertices move: they are tied to a mesh of triangles, and only the mesh is advected. A segment inside one triangle stays a straight segment inside the triangle after the map, and segments in different triangles cannot cross as long as the triangles do not overlap.

`"repair-quadtree"` is based on the line densification of Miaji et al. \[3\]: a graded quadtree is built from the density and triangulated, vertices are inserted where the boundaries cross the edges of the triangulation, and the vertices move with the piecewise-affine map of the triangles. This implementation differs from the published method in four ways. The mesh is built from the density at a recompute of the velocity field and kept for several windows. The leaves are triangulated as fans from a center point, not by a constrained Delaunay triangulation, and a fan that inverts is repaired by moving its center into the kernel of the leaf. The number of vertices is kept down by removing inserted vertices that are not needed, checked against crossings, not by a topology-preserving simplification of the whole geometry, so every input vertex stays. And the map is applied to the nodes of the time-stepped flow of this library, not to a density-equalizing map computed from the whole density in one step.

### The mesh

1. **The quadtree.** The grid is padded to a square with a power of two cells per side. The square is a leaf. The leaf whose grid cells have the largest range of the density (maximum minus minimum) is split into four leaves, again and again, until the number of leaves reaches a budget of 1/64 of the number of grid cells (at least four leaves) or the density is constant in every leaf. Leaves are therefore small where the density has a large range, typically at the boundaries between regions of different density. The tree is then graded: neighboring leaves differ by at most one level.
2. **The triangulation.** Every leaf is a fan of triangles: one triangle from the center of the leaf to each pair of neighboring boundary nodes. The boundary nodes are the four corners and, on a side whose neighbor is one level deeper, the midpoint of that side. This node is shared with the two smaller leaves and is called a hanging node. Because the larger leaf has the hanging node on its boundary, the triangles of neighboring leaves fit along their shared side. A leaf has at most eight triangles.
3. **The insertion.** A vertex is inserted where a segment crosses a side of a leaf or an edge of a fan, so that afterwards every segment lies inside one triangle. The end points of a segment are put in a fixed order before the crossing is computed, so the two regions of a shared boundary get bit-identical vertices. Input vertices are kept.

![A quadtree on a toy density with a high-density region, and the same leaves with their fan triangles, hanging nodes and the vertices inserted where the boundary of the region crosses a mesh edge](../generated/diagrams/validity-quadtree-mesh.svg)

*Figure 4. Left: the quadtree of a density with one high-density region; the leaves are small along the boundary of the region. Right: every leaf is a fan of triangles (thin lines); the small squares are hanging nodes. Orange vertices are inserted where the boundary crosses a leaf side or a fan edge. The figure uses a larger leaf budget than the default so that the leaves are visible.*

### The map

A vertex $p$ inside a triangle with nodes $q_1, q_2, q_3$ has barycentric coordinates $(\lambda_1, \lambda_2, \lambda_3)$ with $\lambda_i \ge 0$ and $\lambda_1 + \lambda_2 + \lambda_3 = 1$:

$$
p = \lambda_1 q_1 + \lambda_2 q_2 + \lambda_3 q_3 .
$$

The coordinates are computed when the mesh is built and then kept. In every iteration only the nodes are advected through the velocity field, to $q_i'$, and every vertex is placed at

$$
p' = \lambda_1 q_1' + \lambda_2 q_2' + \lambda_3 q_3' .
$$

The center of a fan is not advected: it is put at the mean of the four corners of its leaf. This map is affine inside each triangle and continuous across the sides of triangles. An affine map keeps straight lines straight, so the segments inside one triangle remain segments. As long as no triangle is inverted (no triangle has a non-positive area), the triangles after the map do not overlap, so no two points end up at the same place and boundaries cannot cross. Landmarks, probes and displacement coordinates are not part of the mesh and move with the field as before.

A vertex does not follow its own flow path. It follows the nodes of its triangle.

![One triangle with a boundary segment, and the same triangle after its nodes moved: the segment between the inserted vertices and the vertices inside stays straight, and each vertex keeps its barycentric coordinates](../generated/diagrams/validity-affine-triangle.svg)

*Figure 5. A triangle before and after its nodes are advected. The boundary enters and leaves through two edges, where vertices were inserted (orange). After the map the vertices have the same weights on the corners, and the segments are straight.*

### Reuse and rebuild

The mesh is built at the start of a window only when it is needed. The mesh of the last window is kept while no fan has inverted since it was built and every triangle has an area of at least 0.1 % of its area at construction. The mesh is built again when either condition fails, and at the start of every level of a ladder (each level is a separate morph). A rebuild builds the quadtree from the current density, prunes inserted vertices ([below](#pruning)), inserts vertices for the new mesh and locates every vertex in its new triangle.

### Inverted fans

A leaf that is squeezed into a sliver can become a polygon with a reflex corner. The center of its fan, the mean of the four corners, can then lie outside the **kernel** of the polygon, the set of points from which the whole boundary can be seen. From such a center some triangle of the fan has a non-positive area: the fan is inverted. The check runs after every iteration:

1. The center of an inverted leaf is moved to the centroid of its kernel. The kernel is computed by clipping the bounding box of the leaf with the half plane on the inner side of each edge of the polygon in turn. From a point of the kernel all triangles of the fan have a positive area. The mesh is then rebuilt at the next window.
2. A leaf whose kernel is empty, or whose fan is still inverted from the kernel centroid, is limited: its nodes move a half, then a quarter, and so on, and finally none of the step, until no fan is inverted (at most six rounds).
3. An iteration that still ends with an inverted triangle is counted, and the run warns when it ends that the output may contain invalid polygons. The guarantee does not hold for the window of such an iteration.

![A square leaf with its fan, the same leaf squeezed into a polygon with a reflex corner whose fan from the mean of the corners has two inverted triangles, and the fan from the centroid of the kernel](../generated/diagrams/validity-inverted-fan.svg)

*Figure 6. Left: a leaf at construction; the center is the mean of the corners. Middle: after the corners moved, the leaf has a reflex corner and the mean lies outside the leaf; two triangles (red) are inverted. Right: the kernel (purple) is the set of points that see the whole boundary; the fan from its centroid has no inverted triangle.*

### Pruning

Each rebuild inserts vertices for the new mesh, so inserted vertices accumulate where an earlier mesh needed them. At each rebuild, before the new insertion, inserted vertices that are not needed any more are removed. Only inserted vertices can be removed; input vertices are kept, and the vertices of a ring stay in order.

A vertex is a candidate if it is an inserted vertex, it lies within 0.02 cells of the chord of its two neighbors, and the hash of its coordinates is larger than the hash of both neighbors. The hash makes sure that no two adjacent vertices are removed together, and that the two rings of a shared boundary, which have identical vertices, decide alike. A candidate is removed unless one of two conditions holds:

- the new chord (between its neighbors) crosses a segment or the chord of another candidate;
- a vertex of another ring lies within 0.02 cells of the chord. This finds a small ring (an island or a hole) inside the sliver between the old path and the chord, which no crossing test sees.

The second test looks only at rings of at most 64 vertices. This limit is a heuristic: a ring with more vertices that lies entirely inside a sliver narrower than the tolerance is not found.

### Cost

No velocity field is stored: the state of a window is the mesh and 36 bytes per vertex (the triangle and the barycentric coordinates). The output has more vertices than with `"repair-crossings"`, and the method takes more time. Because the vertices follow the nodes of a mesh and not their own flow paths, a vertex can end at a different position than with `"repair-crossings"`.

## Lineage and animations

The output of both repair methods has more vertices than the input. `Cartogram.vertex_lineage` records every change of the vertex set as a round. An insertion round records, for each new vertex, the segment it splits and the fraction along it: 0.5 with `"repair-crossings"` (the midpoint of the chord at the start of the span), and the position of the crossing with the mesh edge with `"repair-quadtree"`. The end points of the segment can be vertices that an earlier round inserted. With `"repair-quadtree"` there are also removal rounds, which record the vertices that remain. Input vertices are never removed and the order of the vertices of a ring is kept. `vertex_lineage` is `None` with `"check"`.

`Cartogram.lift_coordinates(coords)` takes the coordinates of a state with fewer vertices, either the input or a snapshot recorded earlier in the run, and returns them at the vertices of the returned geometries. A vertex that was inserted later is placed at the same fraction between its two parents in the given state, so the lifted polyline is the same shape as the given one, with more vertices on its segments. A vertex that was removed later is dropped. The number of rows has to be that of a state of the run, otherwise a `ValueError` is raised. After removals two states of the run can have the same number of vertices; then the number of rounds that the state has seen is passed as `state` (0 for the input), and without it a `ValueError` is raised. `Cartogram.lift_geometries` does for geometries what `lift_coordinates` does for coordinates and takes the same `state`.

The lineage of a `"repair-quadtree"` run ends at the returned best iterate: the vertex set and the lineage are reset to those of the best iterate when the run returns it. A snapshot recorded after the best iterate is not a state of the lineage and cannot be lifted.

`Cartogram.save` stores the lineage and `Cartogram.load` restores it, so a loaded cartogram lifts the input (the source geometries it also stores) to its vertex set and can be aligned with other cartograms. A file saved without a lineage loads with `vertex_lineage = None`. A file stores only the best iterate, so a loaded cartogram has no other snapshots to lift.

![A polyline with six vertices, the same polyline lifted to nine vertices, and the final state with nine vertices and the paths that a linear interpolation takes](../generated/diagrams/validity-lift.svg)

*Figure 7. Lifting a state to the final vertex set. Left: a state with six vertices. Middle: the vertices that later rounds inserted are added between their parents (here at fraction 0.5) (the second round splits a segment that the first round created); the polyline is unchanged. Right: the final state has the same vertices; the lines connect each vertex of the lifted state with its final position.*

### Animating cartograms with inserted vertices

Linear interpolation between two states moves vertex $k$ of the first state to vertex $k$ of the second one, so it needs equal vertex sets. Snapshots keep the vertices they had when they were recorded, so snapshots of one run can have different numbers of vertices. Each level of a ladder is a separate morph that starts from the output of the level before, with its own lineage. Landmarks, probes and displacement coordinates are separate point sets and are not affected.

- `animate_morph_history(cartogram, interpolation="linear")` lifts every snapshot to the vertices of the returned geometries when the snapshots have different numbers of vertices, so it works as it does without `validity`. A snapshot that cannot be lifted raises a `ValueError` that names it; `interpolation="nearest"` shows the snapshots as recorded and needs no lifting. With `"repair-quadtree"` the animation ends at the best iterate (`through="best"`, the default); with `through="last"` it raises a `ValueError` that names the first snapshot after it. A lifted snapshot drops the vertices that were removed later, so it differs from the recorded snapshot by the removal tolerance.
- `animate_geometry_keyframes(..., interpolation="linear")` takes arbitrary keyframes and does not lift. The precomputed animation of colored polygons raises a `ValueError` for keyframes with different numbers of vertices; the other animations resample the rings by index, which misaligns the vertices. Give the keyframes equal vertex sets first.

`Cartogram.lift_geometries(geometries)` lifts the geometries of the input, or of a snapshot, to the vertices and ring structure of the returned geometries, for example to animate between the input and the result with the lifted input as a keyframe:

```python
start = cartogram.lift_geometries(gdf.geometry)
keyframes = [gdf.assign(geometry=start), cartogram.to_geodataframe()]
anim = animate_geometry_keyframes(keyframes, interpolation="linear", facecolor="population")
```

`align_vertices(sources)` gives the outputs of several morphs of the same input (for example the same map with the values of different years) one vertex set. Every vertex of an output has a position along the rings of the input, which the vertex lineage gives: an input vertex at its index, an inserted vertex between its parents. The result has, in every ring, the vertices at the union of the positions of all sources. Each output is re-sampled at these positions along its own segments, so its shape, area and validity are unchanged and only vertices on its own segments are added. A source is a `Cartogram` (a morph of the input itself) or a `CartogramWorkflow` (a ladder or a chain of refinements, which start from the output of the one before and are followed through all levels). Sources without a lineage (`validity="check"`) can be mixed with others, and so can morphs with different values of `validity`. The sources must have the same number of geometries, parts, rings and input vertices per ring.

![Two outputs of the same input with different inserted vertices, and the same outputs after align_vertices, which gives both the union of their vertices](../generated/diagrams/validity-align.svg)

*Figure 8. Output a has inserted vertices between input vertices 1 and 2 and between 3 and 4; output b between 2 and 3 and between 3 and 4. After `align_vertices` both have the three positions, and the vertex that an output did not have (hollow) lies on its own segment.*

```python
from carto_flow.flow_cartogram import align_vertices

options = MorphOptions(validity="repair-crossings")
a = create_flow_cartogram(gdf, "pop_2010", options=options)
b = create_flow_cartogram(gdf, "pop_2020", options=options)
keyframes = [gdf.assign(geometry=geometry) for geometry in align_vertices([a, b])]
anim = animate_geometry_keyframes(keyframes, interpolation="linear", facecolor="population")
```

## How the methods differ

`"repair-crossings"` lets every vertex move on its own flow path and repairs afterwards: it finds the crossings at the end of a span and inserts vertices there. It inserts only where a crossing was found, stores the velocity fields of a span, and a crossing that appears and disappears between two checks is never seen. `"repair-quadtree"` constrains the movement: while no triangle is inverted, segments cannot cross, wherever they are and whether or not a test would find them. It inserts a vertex at every crossing of a segment with a mesh edge, removes the ones that are not needed later, stores no field, and its vertices follow a mesh and not their own flow paths.

## Limits

- The repairs act on crossings of segments. A fold that the flow itself creates, a zero-width feature of the input, and a hole that leaves its shell without crossing it are not repaired; these regions are listed in `Cartogram.invalid_regions`.
- `"repair-crossings"` starts from the crossings that the test finds at the end of a span. On coarse input, where segments are many grid cells long, it can leave invalid polygons; densify the input or use `"repair-quadtree"`.
- `"repair-quadtree"` guarantees that segments do not cross only while no triangle is inverted. An inversion that the repair of the fan cannot remove is reported with a warning at the end of the run. The pruning test for foreign vertices looks at rings of at most 64 vertices.
- The output has more vertices than the input. `"repair-crossings"` caps the number of vertices that a run inserts; `"repair-quadtree"` has no cap.
- The figures of this page are drawn when the documentation is built by [make_validity_figures.py](https://github.com/bright-fakl/carto-flow/blob/main/docs/diagrams/make_validity_figures.py), which uses only NumPy and matplotlib. They apply the steps to a small analytic velocity field and do not call the library.

## References

\[2\] M. T. Gastner, V. Seguy, and P. More, "Fast flow-based algorithm for creating density-equalizing map projections," *Proc. Natl. Acad. Sci. U.S.A.*, vol. 115, no. 10, pp. E2156–E2164, 2018. DOI: [10.1073/pnas.1712674115](https://doi.org/10.1073/pnas.1712674115)

\[3\] N. Z. Miaji, A. Singhania, M. E. Goh, C. Le, A. Tharatipyakul, and M. T. Gastner, "Topology-preserving line densification for creating contiguous cartograms," arXiv preprint, 2025. arXiv: [2511.08121](https://arxiv.org/abs/2511.08121)

The numbering follows the reference list of the [algorithm page](flow-cartogram-algorithm.md#references).
