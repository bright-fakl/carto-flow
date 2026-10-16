# Non-contiguous Cartogram

## Overview

A non-contiguous (Olson) cartogram keeps the shape of every region and scales it about its own center so that its area is proportional to its value. The regions do not touch, so there is no distortion of the shapes, no solver and no convergence to check. The cost is that the map no longer tiles the plane, and that the regions of low density become small.

`create_noncontiguous_cartogram(gdf, column)` takes the area of a region as

$$
A_i' = \frac{v_i}{\rho}
$$

for one reference density $\rho$ (value per unit of area) that is the same for all regions, and scales the region by the linear factor $s_i = \sqrt{A_i' / A_i} = \sqrt{(v_i / A_i) / \rho}$.

## The reference density

The only choice that changes the map is $\rho$. The factor of a region is 1 when its own density $v_i / A_i$ equals $\rho$, larger when the region is denser.

- **`reference="densest"`** (default) sets $\rho$ to the highest density of any region. That region keeps its size, every other region is smaller, and no region grows. Every area is proportional to its value. The price is that regions that are much less dense than the densest one are small: with US states and population, DC is the densest region and 43 of 49 states fall below 5 % of their area.
- **`reference="largest"`** sets $\rho$ to the density of the region with the largest value, as the Olson cartogram of the R package `cartogram` does by default. The largest region keeps its size, but every region that is denser would have to grow: with population, California is the reference and 11 of 49 states are denser, New Jersey would grow to 4.6 times its area and DC to 41.6 times.
- **A number** sets $\rho$ directly, in value per unit of area of the coordinate system. A smaller $\rho$ makes all regions larger.

`max_scale` limits the linear factor. With the default of 1 nothing grows, and a region that would have to grow is kept at its size; it is then smaller than proportional, a `UserWarning` counts these regions, `NoncontiguousCartogram.capped` marks them and `area_error_pct` gives the error. A larger `max_scale` lets regions grow (and overlap their neighbors), and `None` sets no limit.

## Multipolygons

A region with several parts (islands, exclaves) can be scaled in two ways, chosen with `multipart`.

- **`"group"`** (default) scales the region as one shape about its centroid. The region stays a smaller copy of itself: the parts keep their sizes relative to each other and the distances between them shrink with the region. This treats a region as one unit, as the Olson cartogram does. A part that is far from the others moves toward the centroid of the region, which can lie in the sea or outside every part.
- **`"parts"`** scales each part about its own centroid. Every part stays at its place, on its own land, and the gaps between the parts grow in proportion to the parts, so the region looks more fragmented.

For a region with one polygon the two are the same.

## Scope

Values must be finite and non-negative. A region with the value 0 becomes an empty geometry (the row is kept, its scale factor is 0). A geographic coordinate system gives areas in square degrees, so `create_noncontiguous_cartogram` warns; project the data first.
