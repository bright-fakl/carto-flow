# Changelog

All notable changes to carto-flow are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and carto-flow uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

Releases before 2.0.0 have no changelog entry; see the
[git history](https://github.com/bright-fakl/carto-flow/commits/main) for those.

## [Unreleased]

### Added

- **`multiresolution_morph` and `CartogramWorkflow.morph_multiresolution` warn when the
  grid cannot resolve the smallest regions, and accept `min_resolution="auto"` and
  `levels="auto"`.** The warning appears when the smallest target region covers fewer
  than 2.5 cells at the coarsest grid or more than 1 % of the regions are smaller than
  one cell at the finest grid. `"auto"` takes the smallest power of two from 64 to 512
  at which the smallest region covers 2.5 cells, and doubles up to 2048.
- **`create_voronoi_cartogram` warns when `fix_topology` is used with `weights`:**
  the topology repair makes a weighted run much slower and can move generators away
  from their weighted positions. ([#126](https://github.com/bright-fakl/carto-flow/pull/126))
- **A flow cartogram warns when regions of the result are invalid, and
  `Cartogram.invalid_regions` lists them.** A region that is much narrower than a grid
  cell somewhere could be pinched during the morph until its boundary crossed itself,
  and nothing in the result said so. **`Cartogram.get_geometry(repair=True)`** and
  **`Cartogram.to_geodataframe(repair=True)`** make those regions valid
  (`shapely.make_valid`, zero-area parts dropped; a region can become a MultiPolygon
  and its vertices no longer correspond to the input), so the default is `False`:
  animations between cartograms need the vertices of the input in every geometry.
  ([#125](https://github.com/bright-fakl/carto-flow/pull/125))
- **`create_voronoi_cartogram(..., premorph=True | MorphOptions)`** first morphs
  the regions with the flow cartogram by the same weights and runs the Voronoi
  relaxation on the morphed regions; the flow result is kept in
  `result.premorph`. Without weights the morph equalizes the region areas.
  ([#112](https://github.com/bright-fakl/carto-flow/pull/112))
- **`RasterBackend(generator_anchor=a)`** pulls each generator toward its
  starting position (strength in `[0, 1]`). The default is `0.25` with `premorph`
  and `0` otherwise, so runs without `premorph` are unchanged.
  ([#112](https://github.com/bright-fakl/carto-flow/pull/112))
- **`VoronoiOptions.area_error_tol`** (default `0.01`): the tolerance on weighted
  cell areas relative to their share of the boundary area.
  ([#111](https://github.com/bright-fakl/carto-flow/pull/111))

### Changed

- **`MorphOptions.adaptive_dt`** (new, default `True`, so flow results change) shrinks the step of the flow
  cartogram when the area errors alternate in sign. Near the target a constant step
  overshoots, the areas alternate around their targets with an amplitude proportional
  to `dt`, and a tolerance below that amplitude is met only by chance. When more than
  30 % of the regions changed the sign of their error in the last 3 iterations, the
  step is multiplied by `1 - 0.6 * share` (down to `dt / 8`) and never grows back;
  the sign of an error counts only above a dead band of `0.1 * log2(1 + max_tol)`,
  and the rule does not depend on `recompute_on_rise`. The first
  steps of a run that starts near the target are limited by the score, so a refinement
  of a converged cartogram continues from it. Of 32 tight-tolerance runs on the
  states and districts 32 converge against 18, the highest true error in the late
  phase of the states at grid 256 and `dt` 0.2 drops from 2.0 % to 0.3 %, and a
  second run on a converged states cartogram takes 1 iteration instead of 4. Runs at
  the default tolerances take the same iterations as before in the 12 cases measured
  at `dt` 0.2 to 0.6 (at most 13 % more at `dt` 0.8); a run that stops after a shrunk
  step stops within the tolerance, not necessarily as close to the target as a
  constant step that lands low.
  `adaptive_dt=False` restores the constant step.
  ([#123](https://github.com/bright-fakl/carto-flow/pull/123))
- **The density is rasterized by the area each region covers in a cell by
  default** (`MorphOptions.density_rasterization="exact"`) instead of by the
  region at the cell center; `"center"` restores the previous behavior. Regions
  smaller than a cell take part in the flow, the coarse level of a multiresolution
  morph no longer plateaus on them (states from grid 128: 63 iterations against
  134; districts: 123 against 231), and rasterization is faster from grid 256 up.
  Results change slightly: the default states run converges in 112 iterations
  instead of 113, and its maximum area error at grid 256 is 0.78 % instead of
  4.36 %.
  ([#119](https://github.com/bright-fakl/carto-flow/pull/119))
- **Flow-cartogram error metrics floor areas relative to the mean target area,
  not at one grid cell.** `MorphOptions.error_floor` (default `1e-3`) replaces the
  floor of `grid.dx * grid.dy`, so the reported errors (`mean_error_pct`,
  `max_error_pct`, the convergence history and the snapshot errors) no longer
  depend on the grid size and are comparable between the levels of a
  multiresolution morph; the initial error of the 432 districts is 513 % on every
  grid (it was 176 % at grid 128 and 509 % at 1024). Runs on the bundled states and
  districts are unchanged. On the counties the reported mean error at grid 256
  is now 58.6 % (it was 4.5 % while the true mean error was 58.5 %). `0` disables
  the floor.
  ([#121](https://github.com/bright-fakl/carto-flow/pull/121))
- **Weighted Voronoi cells follow their weights.** With `weights` and euclidean
  labeling, `RasterBackend` builds power diagrams (straight, convex-before-clipping
  cells) and solves the final areas exactly, so every cell is within
  `area_error_tol` of its weight share; on the 49 states the mean area error is
  0.04 % against 77.6 % before. Every weighted `RasterBackend` result changes;
  unweighted results are unchanged. A weighted run that misses the tolerance is
  reported as `converged=False` with a `RuntimeWarning`, which includes
  `ExactBackend` and `distance_mode="geodesic"`, since both still ignore weights.
  ([#111](https://github.com/bright-fakl/carto-flow/pull/111))
- **The weighted `ElasticBoundary`** is driven by the input density carried along by
  the boundary flow, keeps the boundary area at its initial value, and its step is
  about twice as fast.
  ([#111](https://github.com/bright-fakl/carto-flow/pull/111))
- **A flow morph that does not converge returns its best iterate** (lowest
  combined error score), and stops after `stall_patience` consecutive cycles
  without progress. `History.best_iteration` points at the returned snapshot,
  the snapshots stay in iteration order, and `animate_morph_history`,
  `animate_fields` and `animate_workflow_fields` stop at it unless called with
  `through="last"`. New option `MorphOptions.recompute_on_rise` recomputes the
  velocity field early when the score rises; `recompute_every` stays the
  maximum interval.
  ([#110](https://github.com/bright-fakl/carto-flow/pull/110))

### Deprecated

- **Voronoi tuning settings become internal.** A non-default value of
  `RasterBackend.relaxation`, `area_equalizer_rate`, `weight_ramp_iters`,
  `output_resolution` and `cell_smoothing_px`, `ExactBackend.relaxation`,
  `ElasticBoundary.step_scale`, `density_smooth` and `min_boundary_points`, and
  `VoronoiOptions.adj_min_shared_length` warns and will no longer be accepted in the
  next major release; the defaults are the tested values.
  ([#126](https://github.com/bright-fakl/carto-flow/pull/126))
- **`AdhesiveBoundary` and `ElasticBoundary.adhesion_strength`** are deprecated and
  will be removed in the next major release; use `ElasticBoundary` (strength 0.02 to
  0.05), or `generator_anchor` with `premorph`.
  ([#126](https://github.com/bright-fakl/carto-flow/pull/126))

- **`Cartogram.latest`, `CartogramWorkflow.latest` and `History.latest()`.**
  `Cartogram.best` is the snapshot of the returned state and `Cartogram.last`
  the last stored one; `CartogramWorkflow.last` is the most recent result and
  `History.last()` the last snapshot. `Cartogram.latest` returns `best`. On a
  run that does not converge, `best` is the earlier iterate that the run now
  returns, not the final iterate.
  ([#110](https://github.com/bright-fakl/carto-flow/pull/110))

### Fixed

- **The `MorphOptions` presets stopped before converging.** Their iteration limits
  (30, 100 and 300) date from before the stall rule, the exact rasterization and the
  adaptive step: on the bundled states the three presets ended with a mean area
  error of 42 %, 13 % and 49 %. The limits are now 150 (`preset_fast`), 500
  (`preset_balanced`, the same values as `MorphOptions()`) and 1000
  (`preset_high_quality`), and all three converge on the states and congressional
  districts. ([#127](https://github.com/bright-fakl/carto-flow/pull/127))
- **`partition_geometries` returned empty or wrong parts for invalid input
  polygons without an error:** invalid geometries are now repaired with
  `shapely.make_valid` and a warning names the rows.
  ([#125](https://github.com/bright-fakl/carto-flow/pull/125))
- **A region whose ring has a small self-intersection could be missing from the
  density field of the exact rasterizer:** the ring was judged counter-clockwise
  although its signed area was negative, so the region got negative coverage and
  was clipped to zero. Rings are now oriented by their signed area.
  ([#124](https://github.com/bright-fakl/carto-flow/pull/124))
- **Long `ElasticBoundary` runs could drop the mainland and collapse the
  boundary:** when a rebuilt ring was self-intersecting, `make_valid` returned a
  geometry collection and only items of type `Polygon` were kept.
  ([#111](https://github.com/bright-fakl/carto-flow/pull/111))

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

- The flow cartogram morph is substantially faster than 1.1.2. Measured on
  contiguous US states with identical geometry and options, serial in both
  versions: 3.5x at a 128-cell grid, 5.3x at the default 256, 10.0x at 512 and
  11.6x at 1024. Iteration counts are identical at every grid size, so the
  convergence path is unchanged — the work per iteration got cheaper. The gain
  is in the density rasterization and FFT solve rather than in vertex
  advection, so it grows with grid size and shrinks as vertex count rises.
  ([#11](https://github.com/bright-fakl/carto-flow/pull/11))
- `repair_contiguity` skips path enumeration for pairs it cannot connect, and
  mosaic calibration skips adjacency computation and vectorizes lattice
  adjacency. ([#36](https://github.com/bright-fakl/carto-flow/pull/36),
  [#44](https://github.com/bright-fakl/carto-flow/pull/44))

[2.0.0]: https://github.com/bright-fakl/carto-flow/releases/tag/2.0.0
