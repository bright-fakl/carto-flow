# Migrating to 2.0

carto-flow 2.0 reorganizes the symbol cartogram package, renames part of its
public API, and changes several defaults. The flow, proportional and Voronoi
cartogram APIs are unchanged.

Each section below gives the 1.x call, the 2.0 call, and what happens if the 1.x
call is left in place: an exception, a warning, or silently different output.
Sections that raise are the ones to fix first. The full list of changes is in the
[changelog](changelog.md).

## Import paths inside `symbol_cartogram`

The layout classes, their option dataclasses, `LayoutData`, `LayoutResult`,
`Transform` and `SimulationHistory` moved into
`carto_flow.symbol_cartogram.layouts`. The modules `layout.py`,
`layout_result.py`, `data_prep.py` and `placement.py` no longer exist, and
`symbol_cartogram.options` keeps only the enums (`AdjacencyMode`, `ForceMode`,
`SymbolShape`, `SymbolOrientation`).

Every public name is still re-exported from the package root, so importing from
`carto_flow.symbol_cartogram` needs no change:

```python
# Works in 1.x and 2.0
from carto_flow.symbol_cartogram import CirclePackingLayout, CirclePackingLayoutOptions
```

Submodule imports must be updated:

```python
# 1.x
from carto_flow.symbol_cartogram.layout import CirclePackingLayout, Layout, get_layout
from carto_flow.symbol_cartogram.layout_result import LayoutResult, Transform
from carto_flow.symbol_cartogram.data_prep import LayoutData, prepare_layout_data
from carto_flow.symbol_cartogram.options import CirclePackingLayoutOptions, GridBasedLayoutOptions

# 2.0
from carto_flow.symbol_cartogram.layouts import (
    CirclePackingLayout,
    CirclePackingLayoutOptions,
    GridBasedLayoutOptions,
    Layout,
    LayoutData,
    LayoutResult,
    Transform,
    get_layout,
    prepare_layout_data,
)
```

The internals of `placement.py` are now private modules under each layout
(`layouts/grid/_placement.py`, `layouts/packing/_simulator.py`,
`layouts/flow/_simulator.py`) and are not part of the public API.

**If you change nothing:** `ModuleNotFoundError` for the removed modules,
`ImportError` for an option dataclass imported from `symbol_cartogram.options`.

## `value_column` is now `size`

```python
# 1.x
result = create_symbol_cartogram(gdf, value_column="population")

# 2.0
result = create_symbol_cartogram(gdf, size="population")
```

The parameter is positional in both `create_symbol_cartogram` and
`create_layout`, so `create_symbol_cartogram(gdf, "population")` works in both
releases.

**If you change nothing:** `TypeError: got an unexpected keyword argument
'value_column'`.

## `layout="physics"` is now `layout="packing"`

The `physics` layout was removed. Circle packing supersedes it and, unlike
physics, honors `source_indices` (so a region's `tile_count` tiles stay together)
and `group_by`.

```python
# 1.x
result = create_symbol_cartogram(gdf, "population", layout="physics")

# 2.0
result = create_symbol_cartogram(gdf, "population", layout="packing")
```

With explicit options:

```python
# 1.x
from carto_flow.symbol_cartogram import CirclePhysicsLayout, CirclePhysicsLayoutOptions

layout = CirclePhysicsLayout(CirclePhysicsLayoutOptions(topology_weight=0.5, max_iterations=500))

# 2.0
from carto_flow.symbol_cartogram import CirclePackingLayout, CirclePackingLayoutOptions

layout = CirclePackingLayout(CirclePackingLayoutOptions(topology_weight=0.5, max_iterations=500))
```

`CirclePhysicsLayoutOptions` and `CirclePackingLayoutOptions` do not share every
field, so check the options reference for any field that fails. `PhysicsHistory`
and `PhysicsMetrics` were removed with the layout; the packing equivalents are
`PackingHistory` and `PackingMetrics`.

**If you change nothing:** `ValueError: Unknown layout 'physics'` from the string
form, `ImportError` from the class form.

## The default layout changed

A call that passes no `layout` used the physics layout in 1.x and uses circle
packing in 2.0.

```python
# 1.x: physics
# 2.0: packing
result = create_symbol_cartogram(gdf, "population")
```

To keep symbol positions comparable across the two releases there is no
substitute — the physics layout is gone and packing is a different algorithm.
Pin `layout="packing"` explicitly so the call does not depend on the default.

**If you change nothing:** no error and different symbol positions.

## `layout_type` values

The packing and flow-density layouts recorded `layout_type="physics"` on their
results. They now record their own registry keys.

```python
# 1.x
if result.layout_result.layout_type == "physics":
    ...

# 2.0
if result.layout_result.layout_type in ("packing", "flow_density"):
    ...
```

The registry keys are `packing`, `flow_density`, `grid`, `mosaic` and `centroid`.
`topology` remains a registered alias for `packing`.

**If you change nothing:** no error; a branch keyed on `"physics"` never runs.

## `group_by` on the grid layout

The grid layout passed `group_by` through to the result for labeling but placed
symbols as if it were absent. It now raises. The layouts that honor `group_by`
are centroid, flow density, mosaic and packing.

```python
# 1.x: grouping ignored during placement
result = create_symbol_cartogram(gdf, "population", layout="grid", group_by="State Name")

# 2.0: use a layout that honors the grouping
result = create_symbol_cartogram(gdf, "population", layout="mosaic", group_by="State Name")
```

If the grouping was only ever used for styling or export, drop `group_by` and
keep the grid layout, or switch to the mosaic layout, which assigns each group a
connected block of tiles.

**If you change nothing:** `ValueError` naming the layouts that honor
`group_by`.

## `group_by` with an inert grouping force

Circle packing and flow density honor `group_by` through one option each, and
that option defaults to a no-op. Both now warn when `group_by` is given and the
option is still at its default.

```python
# 1.x and 2.0: warns, grouping does not affect placement
result = create_symbol_cartogram(gdf, "population", layout="packing", group_by="State Name")

# 2.0: set the force that acts on the grouping
from carto_flow.symbol_cartogram import CirclePackingLayout

result = create_symbol_cartogram(
    gdf,
    "population",
    layout=CirclePackingLayout(group_weight=0.5),
    group_by="State Name",
    collapse_group=1.0,
)
```

For the flow-density layout the option is `cross_group_pull_scale`, which must
drop below 1.0 to separate groups:

```python
from carto_flow.symbol_cartogram import FlowDensityLayout

result = create_symbol_cartogram(
    gdf,
    "population",
    layout=FlowDensityLayout(cross_group_pull_scale=0.0),
    group_by="State Name",
)
```

`dorling_grouped_cartogram` and `geographic_grouped_cartogram` set
`group_weight` themselves and do not warn.

**If you change nothing:** a `UserWarning`, and placement that ignores the
grouping exactly as in 1.x.

## `size_normalization` default

In 1.x `size_normalization` defaulted to `"max"` for every layout. In 2.0 each
layout supplies its own default: `"total"` for the layouts that place symbols by
radius (circle packing, centroid, flow density) and `"max"` for the grid layout,
whose lattice is calibrated from the largest symbol. The mosaic layout takes its
symbol scale from the tile lattice, so `size_normalization` has no effect there.

With `"total"` the total symbol area equals the total geometry area. With
`"max"` the largest symbol has the mean geometry area, so under the default
`sqrt` scale the covered fraction of the map is `mean(value) / max(value)` and
depends on how skewed the sizing column is. For most inputs `"total"` produces
visibly larger symbols.

To keep 1.x symbol scale, pass it explicitly:

```python
# 1.x default
result = create_symbol_cartogram(gdf, "population", layout="packing")

# 2.0, same symbol scale as 1.x
result = create_symbol_cartogram(gdf, "population", layout="packing", size_normalization="max")
```

An unrecognized value used to fall through to the `"max"` branch; it now raises.

```python
# 1.x: silently behaved as "max"
# 2.0: ValueError
create_symbol_cartogram(gdf, "population", size_normalization="sum")
```

**If you change nothing:** no error and larger symbols from the radius-based
layouts; `ValueError` for a value other than `"max"` or `"total"`.

## Presets became named cartogram functions

The `preset_*` functions returned kwargs dicts to splat into
`create_symbol_cartogram`. They were replaced by functions that take the
GeoDataFrame and return a `SymbolCartogram`.

```python
# 1.x
from carto_flow.symbol_cartogram.presets import preset_dorling

result = create_symbol_cartogram(gdf, "population", **preset_dorling())

# 2.0
from carto_flow.symbol_cartogram import dorling_cartogram

result = dorling_cartogram(gdf, "population")
```

| 1.x preset | 2.0 function |
| --- | --- |
| `preset_dorling()` | `dorling_cartogram(gdf, size)` |
| `preset_topology_preserving()` | `geographic_cartogram(gdf, size)` |
| `preset_demers()` | `demers_cartogram(gdf, size)` |
| `preset_tile_map()` | `tile_map_cartogram(gdf)` |
| `preset_fast()` | no equivalent; pass `CirclePackingLayout(max_iterations=100, convergence_tolerance=1e-3)` |
| `preset_quality()` | no equivalent; pass `CirclePackingLayout(max_iterations=1000, convergence_tolerance=1e-5, topology_weight=0.5)` |

Symbol positions differ from the 1.x presets in every case. `preset_dorling`
used the physics layout, which is gone; `preset_topology_preserving` used the
packing layout with `topology_weight=0.5` and otherwise default weights, while
`geographic_cartogram` also sets `origin_weight=0.5` and `neighbor_weight=0.5`.
`dorling_cartogram` packs toward the global centroid, `geographic_cartogram`
holds symbols near their geographic positions. Two grouped variants,
`dorling_grouped_cartogram` and `geographic_grouped_cartogram`, and a
`centroid_cartogram` that only removes overlap, have no 1.x preset.

**If you change nothing:** `ImportError` for every `preset_*` name.

## Result attributes: metrics and history

```python
# 1.x
result.metrics["converged"]
result.simulation_history.drift
result.simulation_history.overlaps

# 2.0
result.placement_metrics["converged"]
result.layout_result.history.algorithm.drift
result.layout_result.history.overlaps
```

`SimulationHistory` keeps only what every iterative layout records (`positions`,
`overlaps`); the algorithm-specific arrays moved to its `algorithm` subobject,
which is a `PackingHistory` or a `FlowDensityHistory` depending on the layout.
The scalar summaries (`converged`, `iterations`, `final_overlaps`, plus an
algorithm subobject) are on `AlgorithmMetrics` at
`result.layout_result.metrics`. The convenience properties `converged`,
`iterations` and `overlaps` on `LayoutResult` are unchanged, so
`result.layout_result.converged` reads the same in both releases.

**If you change nothing:** `AttributeError` for both `metrics` and
`simulation_history`.

## `HungarianOptions` fields

`MosaicLayout` is new in 2.0, so this section applies only to code written
against an unreleased `main`.

```python
# pre-release main
HungarianOptions(
    gap_bridge_mult=2.0,
    disconnected_score_weight=0.5,
    max_connectivity_iters=15,
    disconnected_penalty_mult=3.0,
)

# 2.0
HungarianOptions(max_connectivity_iters=15, penalize_disconnected=True)
```

`gap_bridge_mult` and `disconnected_score_weight` were removed after measuring
no effect on any tested input. `disconnected_penalty_mult` became the boolean
`penalize_disconnected`: the multiplier itself is fixed internally, and only
switching the penalty on or off changed results. `max_connectivity_iters` now
defaults to 5 instead of 15; iterations past the fifth did not improve
connectivity on any tested input. Pass 15 explicitly to reproduce the old
behavior.

**If you change nothing:** `TypeError` for the removed and renamed fields; with
none of them passed, slightly different tile assignments from the lower
iteration cap.

## `MosaicLayoutOptions.spacing`

Also pre-release only. The default changed from 0.0 to 0.05, matching the grid
layout, so mosaic tiles no longer touch.

```python
# 2.0, seamless tiles as before
layout = MosaicLayout(spacing=0.0)
```

**If you change nothing:** no error; a visible gap between tiles. Tile
assignment is unaffected.

## Bundled datasets

`world.geojson`, `us_states.geojson` and `cities.geojson` were replaced by
GeoParquet files, and the `world` dataset was repaired so it survives
reprojection. `load_world`, `load_us_states` and `load_sample_cities` are
unchanged. Code that opened the packaged files by name must not:

```python
# 1.x
from importlib.resources import files
import geopandas as gpd

gdf = gpd.read_file(files("carto_flow.data").joinpath("world.geojson"))

# 2.0
from carto_flow.data import load_world

gdf = load_world()
```

`load_us_census` reads a bundled ACS 2020 snapshot for its default resolution,
so it no longer needs a Census API key or the `data` extra. A vintage other than
2020, or a `simplify` tolerance finer than 1000 m, still falls back to a live
`censusdis` download and still needs a key.

**If you change nothing:** `FileNotFoundError` for a direct path to a removed
`.geojson`; world cartograms differ slightly from 1.x because of the repaired
geometry.
