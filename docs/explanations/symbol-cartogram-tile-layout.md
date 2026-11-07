# Tile Layout Algorithm

## Overview

The tile layout (`layout="tiles"`, class `TileLayout`) places every region on tiles of a regular
lattice, one tile per region or an exact number with `tile_count`, and keeps the tiles of a
`group_by` group together. One algorithm serves both cases. It works in two stages that can be
inspected separately: **allocation** decides which lattice tiles may be used, **assignment** gives
those tiles to the regions.

Source: [layouts/tiles/\_\_init\_\_.py](https://github.com/bright-fakl/carto-flow/blob/main/src/carto_flow/symbol_cartogram/layouts/tiles/__init__.py),
[layouts/tiles/\_allocation.py](https://github.com/bright-fakl/carto-flow/blob/main/src/carto_flow/symbol_cartogram/layouts/tiles/_allocation.py),
[layouts/tiles/\_assignment.py](https://github.com/bright-fakl/carto-flow/blob/main/src/carto_flow/symbol_cartogram/layouts/tiles/_assignment.py),
[layouts/tiles/\_descent.py](https://github.com/bright-fakl/carto-flow/blob/main/src/carto_flow/symbol_cartogram/layouts/tiles/_descent.py)

```python
from carto_flow.symbol_cartogram import create_symbol_cartogram, TileLayout

result = create_symbol_cartogram(states, layout=TileLayout(compactness=0.75))
result = create_symbol_cartogram(districts, group_by="state", layout="tiles")
```

## Lattice

A tile has the mean area of a needed tile: total input area divided by the number of tiles. The
lattice covers the bounds of the input, enlarged where needed to hold at least twice the needed
tiles. `rotation` and `offset` turn and shift it as for the grid and mosaic layouts.

## Allocation

The input geometries are split into **land parts**: connected components of their union, where
components joined by one multi-part region form one part.

The number of usable tiles follows the `compactness` option on a geometric scale,

`n_allowed = round(n * (m / n) ** (1 - compactness))`,

with `n` tiles needed and `m` lattice tiles in the window. Compactness 1 uses exactly `n` tiles,
compactness 0 uses all `m`. Each land part receives tiles in proportion to the tiles it needs. The
tiles are taken in the order of the smallest scaled copy of the part's outline that contains
their centers, so the allowed set follows the silhouette. Each part is then made one connected
piece, and tiles that leave a pocket (an unused tile with most neighbors used) are filled by moving
tiles of the largest part.

`components` sets how land parts relate:

- `"separate"`: each part sits at its own position.
- `"connected"`: smaller parts are placed against the block of the largest part, on the side of
  their real position.
- `"spaced"`: as connected, with one unused tile between each part and the block.

A region may only use tiles of its own land part, in every mode. Land parts always come from the
input geometries, also with `pre_morph`.

A region with parts more than one tile away from its main part receives a share of its tile count
for each remote part, in proportion to area, when the share rounds to at least one tile. With one
tile the tile goes to the main part.

## Assignment

### One tile per region

Tiles are assigned by an exact optimal assignment (Hungarian) of regions to the allowed tiles. The
cost of a tile is the squared distance, in tile steps, from its center to the region geometry, plus
0.1 times the squared distance to its centroid. The surplus tiles go to a reserve whose cost is
the distance of a tile to the boundary of the convex hull of the allowed tiles, so unused tiles
end up on the edge of the allowed set and not as holes inside it.

A contact descent then swaps regions that are within three hops and moves regions to free tiles
within two hops, while the energy drops. The energy is 0.5 times the displacement plus 0.5 times
the squared hop gap of neighboring regions that do not touch (capped), plus 0.5 for each pair of
touching regions that are not neighbors, plus one per pocket. Neighbors are regions whose
shared border is longer than 0.05 tile; see
[`find_bordering_pairs`][carto_flow.geo_utils.adjacency.find_bordering_pairs]. With
`protect_contacts=True` a move that lowers the number of touching neighbor pairs among the moved
regions is rejected.

### Tile counts and groups

With `tile_count` or `group_by`, the group (the region itself with `tile_count`) has a capacity
equal to its tile count, and the same transport problem is solved exactly with replicated rows.
The reserve cost gains one per used neighbor of an unused tile, from the previous solution, in
three passes. The result is repaired in two steps:

1. A swap search exchanges tiles between neighboring groups when that lowers the number of extra
   components of the two groups, or keeps it and lowers the cost.
2. Chain moves shift a stray tile along up to three groups.

Within a group, tiles go to the group's geometries by matching their centroids, after fitting the
centroids to the tiles by translation and uniform scaling.

Contiguity is not guaranteed at `compactness=1`, where no tile is spare: a group can remain in
two pieces (Florida in a districts-by-state map, for instance). Below 1 the repair finds a
contiguous solution for typical inputs.

## Pre-morph

`pre_morph=True` morphs the geometries with the flow cartogram to equal area per tile before the
assignment; the assignment then draws tiles to the morphed geometries. It helps most when
region sizes differ widely (US states, districts). The land parts stay those of the input.

## Cost and limits

- The assignment is dense in the number of usable tiles: time and memory grow with its square.
- The result is deterministic.
- The layout does not reproduce every shape feature; a region can sit a hop away from some of
  its neighbors, and at low compactness a few pockets can remain.

## Choosing a layout

| Need | Layout |
|---|---|
| One tile per region, symbol size encodes a value | `grid` or `tiles` |
| Exact tile counts per region | `mosaic` or `tiles` |
| Groups of tiles kept together | `mosaic` or `tiles` |
| Control of how far tiles may spread from the outline | `tiles` (`compactness`) |
