# Changelog

All notable changes to carto-flow are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and carto-flow uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

Releases before 2.0.0 have no changelog entry; see the
[git history](https://github.com/bright-fakl/carto-flow/commits/main) for those.

## [2.0.0] - 2026-09-24

Step-by-step edits for every break below are in the
[migration guide](https://bright-fakl.github.io/carto-flow/migrating-to-2.0/),
including what a caller sees if they change nothing.

### Breaking changes

- **`symbol_cartogram` reorganized into a `layouts/` subpackage.** The layout
  classes, their option dataclasses, `LayoutData`, `LayoutResult`, `Transform`
  and `SimulationHistory` now live in `carto_flow.symbol_cartogram.layouts`;
  `symbol_cartogram.options` keeps only the enums. The modules `layout.py`,
  `layout_result.py`, `data_prep.py` and `placement.py` are gone. Every name is
  still re-exported from `carto_flow.symbol_cartogram`, so package-level imports
  are unaffected; submodule imports are not.
  ([#19](https://github.com/bright-fakl/carto-flow/pull/19))
- **`create_symbol_cartogram` and `create_layout` renamed `value_column` to
  `size`.** The parameter is positional in both, so only keyword callers are
  affected. ([#19](https://github.com/bright-fakl/carto-flow/pull/19))
- **The `preset_*` functions were replaced by named cartogram functions.**
  `preset_dorling`, `preset_topology_preserving`, `preset_demers`,
  `preset_tile_map`, `preset_fast` and `preset_quality` returned kwargs dicts;
  `dorling_cartogram`, `geographic_cartogram`, `demers_cartogram`,
  `tile_map_cartogram`, `centroid_cartogram`, `dorling_grouped_cartogram` and
  `geographic_grouped_cartogram` take a GeoDataFrame and return a
  `SymbolCartogram`. ([#19](https://github.com/bright-fakl/carto-flow/pull/19))
- **`SymbolCartogram.metrics` renamed `placement_metrics`, and
  `SymbolCartogram.simulation_history` removed** in favor of
  `SymbolCartogram.layout_result.history`. Algorithm-specific per-iteration
  arrays (`drift`, `jitter`, `drift_rate`, `velocity`) moved off
  `SimulationHistory` onto its `algorithm` subobject (`PackingHistory` or
  `FlowDensityHistory`), and the scalar summaries onto `AlgorithmMetrics`.
  ([#19](https://github.com/bright-fakl/carto-flow/pull/19))
- **The `physics` layout was removed and the default layout changed from
  `"physics"` to `"packing"`.** `CirclePhysicsLayout`,
  `CirclePhysicsLayoutOptions`, `PhysicsHistory` and `PhysicsMetrics` are gone.
  `CirclePackingLayout` supersedes them: it honors `source_indices` and
  `group_by`, which physics did not.
  ([#54](https://github.com/bright-fakl/carto-flow/pull/54))
- **`LayoutResult.layout_type` records the layout's registry key.** Packing and
  flow-density results reported `layout_type="physics"`; they now report
  `"packing"` and `"flow_density"`.
  ([#54](https://github.com/bright-fakl/carto-flow/pull/54))
- **`group_by` raises for layouts that ignore the grouping.** The grid layout
  passed `group_by` through for labeling only and placed symbols as if it were
  absent; it now raises `ValueError`. The circle packing and flow-density
  layouts warn when the option that acts on the grouping is at its inert default
  (`group_weight=0.0` and `cross_group_pull_scale=1.0`).
  ([#39](https://github.com/bright-fakl/carto-flow/pull/39))
- **`size_normalization` defaults per layout instead of to `"max"`.** The
  layouts that place symbols by radius — circle packing, centroid, flow
  density — default to `"total"`, so total symbol area equals total geometry
  area. The grid layout keeps `"max"`, since its lattice is calibrated from the
  largest symbol; the mosaic layout is unaffected, because its symbol scale
  comes from the tile lattice. An unrecognized value raises `ValueError` instead
  of silently behaving as `"max"`.
  ([#42](https://github.com/bright-fakl/carto-flow/pull/42))
- **Bundled datasets are GeoParquet rather than GeoJSON.** `world.geojson`,
  `us_states.geojson` and `cities.geojson` were replaced by `.parquet` files and
  `pyarrow` became a required dependency. `load_world`, `load_us_states` and
  `load_sample_cities` are unchanged; code that opened the packaged files
  directly is not.
  ([#17](https://github.com/bright-fakl/carto-flow/pull/17))

`MosaicLayout` is new in this release, so its option changes below break only
code written against an unreleased `main`:

- `HungarianOptions.gap_bridge_mult` and `HungarianOptions.disconnected_score_weight`
  were removed; both were measurably inert.
  ([#43](https://github.com/bright-fakl/carto-flow/pull/43))
- `HungarianOptions.max_connectivity_iters` default 15 -> 5, and
  `disconnected_penalty_mult` (float) was replaced by `penalize_disconnected`
  (bool). ([#48](https://github.com/bright-fakl/carto-flow/pull/48))
- `MosaicLayoutOptions.spacing` default 0.0 -> 0.05, matching the grid layout.
  Rendering only. ([#56](https://github.com/bright-fakl/carto-flow/pull/56))

### Added

- `MosaicLayout`, a contiguous tilegram layout for the symbol cartogram: each
  region receives a connected block of lattice tiles, assigned by a Hungarian
  matching over a morphed target field and repaired by chain swaps. Accepts both
  `tile_count` and `group_by`.
  ([#21](https://github.com/bright-fakl/carto-flow/pull/21))
- `FlowDensityLayout`, which advects symbols along the gradient of a smoothed
  density field derived from their target sizes, with `cross_group_pull_scale`
  to separate groups.
  ([#19](https://github.com/bright-fakl/carto-flow/pull/19))
- `voronoi_cartogram`, a new submodule producing area-equalized Voronoi
  cartograms by Lloyd relaxation, with raster and geometric backends, geodesic
  labeling, contiguity analysis and repair, visualization and animation.
  ([#15](https://github.com/bright-fakl/carto-flow/pull/15))
- A bundled ACS 2020 census snapshot. `load_us_census` reads it for its default
  resolution, so the common path needs no Census API key, no network access and
  no optional dependency. Requests outside the snapshot still fall back to
  `censusdis`. Congressional districts were added as a resolution, alongside
  `load_us_state_population` for a yearly population series.
  ([#13](https://github.com/bright-fakl/carto-flow/pull/13),
  [#16](https://github.com/bright-fakl/carto-flow/pull/16),
  [#17](https://github.com/bright-fakl/carto-flow/pull/17))
- `MosaicLayoutOptions.min_one_tile_per_region`, which places a region left
  without any tile on the nearest free cell. A warning names the regions that
  would otherwise disappear.
  ([#47](https://github.com/bright-fakl/carto-flow/pull/47))
- Grouped symbol cartograms: `group_by`, `tile_count`, `collapse_group`,
  `tile_size_expansion` and `pre_scale` on `create_symbol_cartogram` and
  `create_layout`, per-group styling through `Styling.set_group_symbol`,
  `Styling.group_transform` and `Styling.set_group_params`, and the
  `dorling_grouped_cartogram` / `geographic_grouped_cartogram` entry points.
  ([#19](https://github.com/bright-fakl/carto-flow/pull/19))
- Flow cartogram performance options: `MorphOptions.parallel_fft` and
  `parallel_density` for threaded FFT and density rasterization, a bounding-box
  filter, and `MorphOptions.benchmark` with a `Benchmark` object recording
  per-stage timings.
  ([#7](https://github.com/bright-fakl/carto-flow/pull/7),
  [#11](https://github.com/bright-fakl/carto-flow/pull/11),
  [#14](https://github.com/bright-fakl/carto-flow/pull/14))
- Shared geometry utilities in `carto_flow.geo_utils`: `explode_geodataframe`,
  `compute_connected_components`, `components_from_adjacency`,
  `prescale_connected_components`, `densify_coverage`, `repair_contiguity`,
  `repair_adjacency`, `repair_compactness` and `repair_group_assignment`.
  ([#18](https://github.com/bright-fakl/carto-flow/pull/18))

### Changed

- `SymbolCartogram.to_geodataframe(level="group", ...)` read the group index as
  a row index into the source GeoDataFrame, so group attributes were taken from
  an unrelated region whenever a group's first member was not at that row. It
  now reads the group's first source row. Output that relied on the old behavior
  was wrong. ([#57](https://github.com/bright-fakl/carto-flow/pull/57))
- The bundled `world.parquet` was repaired: invalid rings were fixed and
  boundaries densified so the dataset survives reprojection. World cartograms
  differ slightly from 1.x.
  ([#45](https://github.com/bright-fakl/carto-flow/pull/45))
- `MosaicLayoutOptions.interior_bonus` default 0.5 -> 2.0, which reduces split
  regions. ([#40](https://github.com/bright-fakl/carto-flow/pull/40))
- Mosaic component tile pools are disjoint, so a tile can no longer be claimed
  by two geographic components.
  ([#46](https://github.com/bright-fakl/carto-flow/pull/46))

### Fixed

- Flow cartogram: a zero sizing value made `mean_log_error` and `max_log_error`
  infinite for the whole morph, defeating the convergence check and emitting
  divide-by-zero warnings. The convergence loop now floors the area ratio at one
  grid cell area. ([#35](https://github.com/bright-fakl/carto-flow/pull/35))
- `prescale_connected_components` used signed values, so a component whose values
  summed to zero was skipped as carrying no data; it now takes absolute values,
  collapses a zero-target component to exactly zero area, and rejects an all-zero
  dataset. ([#38](https://github.com/bright-fakl/carto-flow/pull/38))
- Symbol cartogram: `plot_adjacency` drew edges from raw tile positions for
  grouped results instead of aggregating tiles back to geometry level; packing
  convergence metrics went non-finite for zero-size symbols; a `tile_count` of
  zero crashed the mosaic layout; an input with no adjacent pairs, or a single
  symbol, crashed or never converged under the packing layout.
  ([#19](https://github.com/bright-fakl/carto-flow/pull/19),
  [#38](https://github.com/bright-fakl/carto-flow/pull/38),
  [#41](https://github.com/bright-fakl/carto-flow/pull/41),
  [#54](https://github.com/bright-fakl/carto-flow/pull/54))
- Mosaic connectivity repair: repair was scored by disconnected tiles rather
  than split regions, stranded ring tiles were never pulled back into empty core
  tiles, enclosed holes survived when their core status differed, and the
  swap-back reach was too short to close longer splits.
  ([#29](https://github.com/bright-fakl/carto-flow/pull/29),
  [#30](https://github.com/bright-fakl/carto-flow/pull/30),
  [#31](https://github.com/bright-fakl/carto-flow/pull/31),
  [#32](https://github.com/bright-fakl/carto-flow/pull/32),
  [#34](https://github.com/bright-fakl/carto-flow/pull/34))
- `MosaicLayout` read the per-symbol group id array as if it were per-geometry,
  corrupting connectivity repair and per-region tile counts.
  ([#23](https://github.com/bright-fakl/carto-flow/pull/23))
- Voronoi cartogram: degenerate cells broke adjacency and line clipping and made
  the area equalizer run away, raster cell extraction lost whole cells, the
  `coverage_simplify` tolerance was interpreted in the wrong units, and
  `simplify_coverage` produced sliver interior rings.
  ([#24](https://github.com/bright-fakl/carto-flow/pull/24),
  [#25](https://github.com/bright-fakl/carto-flow/pull/25),
  [#27](https://github.com/bright-fakl/carto-flow/pull/27),
  [#28](https://github.com/bright-fakl/carto-flow/pull/28))
- `carto_flow.flow_cartogram.__all__` listed `Harmonic`, a name the package
  never defined, which made `from carto_flow.flow_cartogram import *` fail.
  ([#14](https://github.com/bright-fakl/carto-flow/pull/14))

### Performance

- `repair_contiguity` skips path enumeration for pairs it cannot connect, and
  mosaic calibration skips adjacency computation and vectorizes lattice
  adjacency. ([#36](https://github.com/bright-fakl/carto-flow/pull/36),
  [#44](https://github.com/bright-fakl/carto-flow/pull/44))

[2.0.0]: https://github.com/bright-fakl/carto-flow/releases/tag/2.0.0
