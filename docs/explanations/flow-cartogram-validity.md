# Output Validity

The flow cartogram moves the vertices of the input polygons and nothing else. This page explains why that can produce polygons that cross themselves, and what the values of `MorphOptions.validity` do about it. The options are listed in [MorphOptions](flow-cartogram-algorithm.md#morphoptions).

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

`validity_repair_every` (windows per span) and `validity_memory_mb` (memory budget for the stored fields) apply to `"repair-crossings"` only. With `"check"` they are validated and unused.

Densify coarse input first (`carto_flow.geo_utils.densify_coverage`, or `max_segment_length` of the geometry loaders): short segments have a small sagitta, so fewer crossings arise and fewer vertices have to be inserted. The repair works on the crossings that remain.

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

## Lineage and animations

The output of `"repair-crossings"` has more vertices than the input. `Cartogram.vertex_lineage` records every insertion round: a new vertex lies at fraction 0.5 between the two end points of the segment that it splits, and these can themselves be vertices that an earlier round inserted. No input vertex moves or is removed, so the vertices of a state are always a subset of the vertices of a later state. `vertex_lineage` is `None` with `"check"`.

`Cartogram.lift_coordinates(coords)` takes the coordinates of a state with fewer vertices, either the input or a snapshot recorded earlier in the run, and returns them at the vertices of the returned geometries. A vertex that was inserted later is placed at the same fraction between its two parents in the given state, so the lifted polyline is the same shape as the given one, with more vertices on its segments. The number of rows has to be that of a state of the run, otherwise a `ValueError` is raised. `Cartogram.lift_geometries` does for geometries what `lift_coordinates` does for coordinates.

![A polyline with six vertices, the same polyline lifted to nine vertices, and the final state with nine vertices and the paths that a linear interpolation takes](../generated/diagrams/validity-lift.svg)

*Figure 4. Lifting a state to the final vertex set. Left: a state with six vertices. Middle: the vertices that later rounds inserted are added at fraction 0.5 between their parents (the second round splits a segment that the first round created); the polyline is unchanged. Right: the final state has the same vertices; the lines connect each vertex of the lifted state with its final position.*

### Animating cartograms with inserted vertices

Linear interpolation between two states moves vertex $k$ of the first state to vertex $k$ of the second one, so it needs equal vertex sets. Snapshots keep the vertices they had when they were recorded, so snapshots of one run can have different numbers of vertices. Each level of a ladder is a separate morph that starts from the output of the level before, with its own lineage. Landmarks, probes and displacement coordinates are separate point sets and are not affected.

- `animate_morph_history(cartogram, interpolation="linear")` lifts every snapshot to the vertices of the returned geometries when the snapshots have different numbers of vertices, so it works as it does without `validity`. A snapshot that cannot be lifted raises a `ValueError` that names it; `interpolation="nearest"` shows the snapshots as recorded and needs no lifting.
- `animate_geometry_keyframes(..., interpolation="linear")` takes arbitrary keyframes and does not lift. The precomputed animation of colored polygons raises a `ValueError` for keyframes with different numbers of vertices; the other animations resample the rings by index, which misaligns the vertices. Give the keyframes equal vertex sets first.

`Cartogram.lift_geometries(geometries)` lifts the geometries of the input, or of a snapshot, to the vertices and ring structure of the returned geometries, for example to animate between the input and the result with the lifted input as a keyframe:

```python
start = cartogram.lift_geometries(gdf.geometry)
keyframes = [gdf.assign(geometry=start), cartogram.to_geodataframe()]
anim = animate_geometry_keyframes(keyframes, interpolation="linear", facecolor="population")
```

`align_vertices(sources)` gives the outputs of several morphs of the same input (for example the same map with the values of different years) one vertex set. Every vertex of an output has a position along the rings of the input, which the vertex lineage gives: an input vertex at its index, an inserted vertex between its parents. The result has, in every ring, the vertices at the union of the positions of all sources. Each output is re-sampled at these positions along its own segments, so its shape, area and validity are unchanged and only vertices on its own segments are added. A source is a `Cartogram` (a morph of the input itself) or a `CartogramWorkflow` (a ladder or a chain of refinements, which start from the output of the one before and are followed through all levels). Sources without a lineage (`validity="check"`) can be mixed with others. The sources must have the same number of geometries, parts, rings and input vertices per ring.

![Two outputs of the same input with different inserted vertices, and the same outputs after align_vertices, which gives both the union of their vertices](../generated/diagrams/validity-align.svg)

*Figure 5. Output a has inserted vertices between input vertices 1 and 2 and between 3 and 4; output b between 2 and 3 and between 3 and 4. After `align_vertices` both have the three positions, and the vertex that an output did not have (hollow) lies on its own segment.*

```python
from carto_flow.flow_cartogram import align_vertices

options = MorphOptions(validity="repair-crossings")
a = create_flow_cartogram(gdf, "pop_2010", options=options)
b = create_flow_cartogram(gdf, "pop_2020", options=options)
keyframes = [gdf.assign(geometry=geometry) for geometry in align_vertices([a, b])]
anim = animate_geometry_keyframes(keyframes, interpolation="linear", facecolor="population")
```

## Limits

- The repair removes crossings of segments. It does not remove a fold that the flow itself creates, a zero-width feature of the input, or a hole that leaves its shell without crossing it; these regions are listed in `Cartogram.invalid_regions`.
- The repair starts from the crossings that the test finds at the end of a span. On coarse input, where segments are many grid cells long, it can leave invalid polygons; densify the input first.
- The output has more vertices than the input, and the cap limits how many a run inserts.
- The figures of this page are drawn when the documentation is built by [make_validity_figures.py](https://github.com/bright-fakl/carto-flow/blob/main/docs/diagrams/make_validity_figures.py), which uses only NumPy and matplotlib. They apply the steps to a small analytic velocity field and do not call the library.
