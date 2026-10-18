# Changelog

All notable changes to carto-flow are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and carto-flow uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

Releases before 2.0.0 have no changelog entry; see the
[git history](https://github.com/bright-fakl/carto-flow/commits/main) for those.

## [Unreleased]

### Added

- **`max_segment_length` densifies the boundaries of the geometry loaders at load time.** `load_us_census`, `load_world` and
  `load_us_states` take `max_segment_length`: no straight segment of the result exceeds it, and the two sides of a shared
  boundary get identical vertices. The unit is the one of the CRS: meters for the census data, degrees for the world and
  US states datasets, which are in EPSG:4326. Densifying in degrees before projecting follows the graticule, so a border
  along a parallel curves correctly (3.5 km off the chord on 6 degrees of the 49th parallel in EPSG:5070). The bundled
  census geometries have segments of at most 5 km, which is too sparse for flow cartograms of districts and counties; 1000
  is a good value for counties (194,000 vertices at 5 km, 769,000 at 1 km).
- **A warning when the input segments are too long for the expected stretch.** A segment grows with the linear enlargement
  of its region, so on input with strongly enlarged regions the straight edges between sparse vertices cross where polygons
  are narrow, which leaves invalid output polygons and raises the error of the coarse levels of a ladder. `create_flow_cartogram`
  and `CartogramWorkflow` now estimate the enlargement of every region from the ratio of its target to its current area and warn
  when more than 0.5 % of the segments are expected to exceed 4 cells of the finest grid, with the `max_segment_length` to
  pass to `densify_coverage`. On the 3,108 counties a ladder from 256 cells reaches 1.5 % mean error instead of 4.2 % with
  segments of 1 km instead of 5 km, and the invalid polygons fall from 321 to 144; the bundled states do not warn.
- **`load_us_census(level="county")` loads the 3,108 contiguous counties** (3,221 with `contiguous_only=False`) from a bundled ACS 2020
  snapshot, with the same population, race and poverty columns as the other levels plus `County Name`. The snapshot is
  coverage-simplified at 1000 m and densified to 5 km segments like the state and district levels.
- **`PackingLayout(shape="square")` packs squares.** The packing simulator treats the symbols as axis-aligned squares with
  the area of the circles: the distance of a pair is the larger axis offset, and overlaps and contacts are resolved
  along that axis, so squares touch along flat edges and slide along each other. The topology, neighbor and compaction forces
  are those of the circle layout. On the census snapshot the squares keep 0.62 of the neighboring pairs close (circles: 0.63).
  The default `shape="circle"` gives the same positions as before.
- **`CentroidLayout(shape="square")` places squares that sit flush.** The layout separates axis-aligned squares with the
  area of the circles (along the axis with the smaller penetration), and its canonical symbol is a square, so squares drawn
  on it leave no gaps and no overlaps. With `spacing=0` the edges touch. The default is `shape="circle"`; a gallery example
  compares the two.
- **`validate_geometries` checks the input of a cartogram.** `carto_flow.geo_utils.validate_geometries(gdf, values, coverage=...)`
  returns a `GeometryReport` with errors (null, empty and invalid geometries, with the reason from
  `shapely.explain_validity`; values that are not finite or are negative), warnings (coordinates in longitude and latitude; polygons that overlap or
  whose shared edges do not match) and information (repeated consecutive vertices). Every issue names the rows by index
  label. `create_flow_cartogram`, `CartogramWorkflow`, `create_voronoi_cartogram` and `create_noncontiguous_cartogram`
  run it first and take `validate=False` to skip it.
- **`PlotStyle` sets the look of the flow plots and animations in one place.** The surface
  and text colors, the boundary, one palette per data role (sequential, diverging,
  categorical), the color scale and the colorbar are fields of a frozen `PlotStyle`, with
  a matplotlib `rc` dict for the generic settings (`PlotStyle.from_mplstyle` reads a
  matplotlib style sheet). The built-in styles are `"default"`, `"dark"` and `"classic"`
  (the look before styles). `style=` selects a style for a call, `style_context(...)` for a
  block, `set_style(...)` for the session, `register_style(name, style)` adds one and
  `unregister_style(name)` removes one (registered styles live in the memory of the process; the
  how-to shows how to register them at startup);
  an argument of the call wins over the style. `Cartogram.plot`, `plot_cartogram`,
  `plot_comparison`, the field and convergence plots and all the `animate_*` functions take
  `style=`. The role of the data is decided by the data: `"errors_pct"` and any data with
  `center=` are diverging (symmetric around the center), text, categorical and boolean
  columns (or `categorical=True`) are categorical, the rest is sequential. `norm=` (a
  `Normalize`, or `"linear"`, `"log"`, `"auto"`) is accepted by `Cartogram.plot`,
  `plot_cartogram`, `plot_comparison` and the animations of geometry, with `center=` and
  `categorical=`; a logarithmic colorbar has ticks at 1, 2 and 5 per decade in real
  numbers (`1M`, `2M`, `5M`). `save_animation` saves the frames with the surface of the
  figure. Styles combine: `a | b` (or `a.merge(b)`) is a style in which what `b` sets wins
  (the `rc` dicts, palettes and channels merge field by field), and `style=`,
  `style_context` and `set_style` take a sequence of styles or names, merged in order. A
  `vmin` or `vmax` in the call fixes a linear normalization in the plots (with `norm="log"`
  they are the limits of the logarithmic scale), and `norm="linear"`,
  `color_scale="linear"` or `log_span` keep a linear scale for data that spans more than one
  decade.
- **The Voronoi and proportional plots take `style=` too.** `VoronoiCartogram.plot`,
  `plot_cartogram`, `plot_comparison`, `plot_compactness`, `plot_displacement`, the topology and
  convergence plots and `animate_voronoi_history`, and `plot_partitions` and `plot_dot_density`,
  use the surface, the palettes by data role (the categorical colors for the cells of
  `facecolor=None` and for categories, `RdYlBu_r` symmetric around 0 for `"area_error_pct"`,
  the sequential palette for numeric data), the boundary and the colorbar of the style, and
  take `norm=` (the Voronoi plots also `center=` and `categorical=`), `legend_kwargs=` and
  `legend_label=`. The edge color and width default to the style, then to the former values
  (black and 0.4 for the Voronoi cells, white and 0.5 for the partitions). A `PlotStyle` has the new
  fields `reference_facecolor`, `reference_edgecolor`, `reference_linewidth` and `reference_alpha` (the
  context layer: the base of a dot density map, the complement of `plot_partitions`) and
  `annotation_size` (the labels of the Voronoi cells). `save_animation` of the Voronoi module saves
  the frames with the surface of the figure.
- **The symbol plots take `style=` too.** `SymbolCartogram.plot`, `plot_symbols`, `plot_comparison`,
  `plot_adjacency`, `plot_adjacency_heatmap`, `plot_tiling` and `plot_displacement` use the
  surface, the palettes by data role (categorical colors for categories, the sequential palette for
  numbers, the diverging palette with `center=`), the colorbar and the boundary of the style. The
  fill, edge and label colors take their palette from the style unless `cmap` is given; the
  channels `fill`, `alpha`, `edge`, `linewidth` and `label` of a style set the default value or, with
  `range=`, the output range of the data-driven alpha, line width and label size. `hatch_cycle` of a
  style sets the hatch patterns of the categories, and the reference fields set the tiles of
  `plot_tiling`. `plot_symbols` takes `center=`
  and `categorical=`; the original panel of `plot_comparison` takes the palettes of the style and `cmap`, `center`,
  `categorical` and `norm` of the call. The colorbars of the adjacency plots follow the colorbar of the style.
- **A `PlotStyle` sets the colors of the diagnostic overlays.** The new fields `series_color` (the
  convergence curve, the cells and centroids of the Voronoi animation), `accent_color` (displacement
  targets, violated adjacency, the final value of the convergence plot, the displacement arrows of the
  symbol plot, the extra-ring tiles of `plot_tiling`), `positive_color` (the convergence mark, reassigned
  cells) and `highlight_color` (satellite cells, the core tiles of `plot_tiling`) are set by the
  `"default"` and `"dark"` styles and unset in `"classic"`, which keeps the former colors. The base
  layers of the Voronoi displacement, topology and repair plots use the reference fill and the axes edge
  color of the style, and the nodes of `plot_adjacency` the stroke color. `PlotStyle.pick(name, default)`
  returns a field or the default. `VelocityPlotOptions.color` is `None` by default and takes the
  accent color of the style (in `"classic"` black on the bare axes of `plot_velocity_field`, white over the density of the animations), the gray reference lines of the workflow convergence
  plot and the key of the line-width legend of the symbol plot take the axes edge color.
- **A how-to compares the symbol and Voronoi layouts on raw and on flow-pre-morphed geometries** ("Pre-Morph Before Symbol and Voronoi Layouts"): the iterations of each layout, the figures, and when to skip the pre-morph.
- **`alpha_norm`, `linewidth_norm` and `label_fontsize_norm` set how a column is spread over the range of a scalar channel of the symbol plots.** They take `"linear"`, `"sqrt"`, `"log"` (positive data), `"auto"` (logarithmic when the data spans more than `log_span` decades), `"rank"` (equal steps for equal numbers of symbols) or a `Normalize`. `Channel(scale=...)` sets the default of a style (the `"default"` and `"dark"` styles take `"auto"` for the line width, in 1 to 4.5 points). The line-width legend and the alpha colorbar follow the scale.
- **The magnitudes of the velocity field and of the Voronoi displacement take a scale, and the field plots use the palettes of the style.** `VelocityPlotOptions(norm=...)` and the `norm=` of `plot_displacement` of the Voronoi cartogram take a `Normalize` or `"linear"`, `"sqrt"`, `"log"` or `"auto"`; the scale sets the color of the arrows (`color_by="magnitude"`), the opacity (`alpha_by_magnitude`) and the colorbar, and defaults to the color scale of the style, else linear (a logarithmic scale leaves out zero magnitudes). `"sqrt"` is a new color scale for every `norm=` and for `PlotStyle(color_scale=...)`. The density and velocity plots and animations, and the Voronoi displacement arrows, use the sequential palette of the style (the diverging palette for the density relative to the mean) where the call gives no `cmap`, the colorbars of the density and velocity plots and animations are drawn by the style, a style has a new `cyclic` palette role (`Palettes(cyclic=...)`: `hsv` in `"default"` and `"dark"`, because the two ends of `twilight` are as light or as dark as the surface) that colors the direction of the velocity, and the convergence plots use the series and highlight colors of the style, and the remaining plot functions take `style=` too: the density and velocity modulator previews (`preview_modulator`: diverging, sequential and cyclic palettes by what is shown, colorbars of the style) and the tiling plots (`TilingResult.plot`, `plot_tile`, `plot_prototile`: the stroke color for the outlines, the cyclic palette for the aspect, the sequential palette for the rest, the categorical palette for the edge shapes); `"classic"` keeps `viridis`, `RdBu_r`, `plasma`, `C0` and `C1`.
- **`create_noncontiguous_cartogram(gdf, column)` makes a non-contiguous (Olson) cartogram.** The new `noncontiguous_cartogram` module scales every region about its own centroid, keeping its shape, so that its area is proportional to its value. `reference` sets the density at which a region keeps its size: `"densest"` (default; no region grows), `"largest"` (the region with the largest value, as in the R package `cartogram`) or a number; `reference_region` names a region (by index label) that keeps its size, to leave out an outlier such as DC; `max_scale` limits the growth of a region (default 1: nothing grows, with a warning that counts the capped regions); `multipart` scales a multipolygon as one shape about its centroid (`"group"`, default) or each part about its own centroid (`"parts"`). The `NoncontiguousCartogram` result has the scaled `geometries`, the scale factors, the `capped` regions and `area_error_pct`, and `plot(facecolor=..., style=...)` draws the regions over the original outlines, largest first.
- **The algorithm explanation has three figures:** the diffusion idea (two regions before and after the flow), the electrostatic reading of the velocity solve (the potential and the field of two opposite charges) and a one-dimensional example of the loop (density, target density and velocity of three regions at three iterations). `docs/diagrams/make_flow_algorithm_figures.py` draws them with the colors of the default plot style when the documentation is built (a mkdocs hook; the images are not committed).
- **A how-to shows how to steer the flow with a `DirectionalTensor`** ("Steer the Flow with a Directional Tensor"): a horizontal, vertical, radial and tangential preferred direction compared on the same data, with the convergence check and the choice of the ratio.
- **"Style Plots and Animations" is a notebook that shows the plots and the animations.** The how-to was a Markdown page with code blocks and no output; it is now a notebook that the documentation build executes: the built-in styles on the same cartogram, `style_context` and `set_style`, the palettes of the data types, your own style, combined styles and two animations.
- **`morph_timeline(snapshots)` builds the keyframes and times of a morph that holds, eases
  through the morph, holds and returns**, for `animate_geometry_keyframes` with
  `position_mapper=linear_over("time")`.
- **A gallery animation morphs a dot density map into a flow cartogram** (`plot_flow_dot_morph.py`): dots generated on the true-area map are carried as landmarks and drawn at every saved iteration with `get_landmarks(iteration, uniform=True)`.
- **`Cartogram.landmark_area_scaling` and `Cartogram.get_landmarks(iteration, uniform=True)`:**
  the morph sums the divergence of the velocity field along the path of every landmark
  vertex (one bilinear sample per vertex and iteration), which gives the area scaling of
  the flow there; every snapshot stores it (`CartogramSnapshot.landmark_area_scaling`).
  The flow scales the area less near the border of a region than inside, so landmarks that
  are uniform in a region before the morph crowd the border of an expanding region;
  `uniform=True` remaps the distance of point landmarks to the border of their region so
  that they are uniform again at any saved iteration, with moves of a few percent of the
  width of the region and a correction that changes smoothly between iterations, so it
  can be used for the frames of an animation. `MorphOptions(landmark_probes=50)` carries
  probe points, uniform in every region, to measure the crowding independently of the
  landmarks, so that lines, polygons, other point sets and, with
  `Cartogram.get_coords(uniform=True)`, displacement coordinates can be corrected too.
- **`carto_flow.data.load_us_president_2020()` loads the 2020 US presidential election results by state** (51 areas: the votes for Biden and Trump, the official results compiled by the Federal Election Commission, bundled; `scripts/download_fec_2020.py` rebuilds the file and checks the national totals). A gallery example, "Area Bias in an Election Map", draws the results on a choropleth, a non-contiguous cartogram and a flow cartogram and reads the area-weighted Democratic share and the blue share of the area off each map.
- **`create_flow_cartogram(gdf, column, *, landmarks, options, displacement_coords, density_per)`
  is the entry point of the flow cartogram, and `MorphOptions.grid` selects the
  resolution.** An int or `(nx, ny)` is one grid; `GridSpec(size, margin, square)` is one
  grid with its margin and squareness; `LadderSpec(min_resolution, levels, margin, square)`
  and `"auto"` run a ladder of resolutions from coarse to fine. The ladder is run by
  `create_flow_cartogram` and `CartogramWorkflow.morph`, so callers that take a
  `MorphOptions` (the Voronoi `premorph=`, the mosaic layout's `morph_options`) accept it
  without special code. A prebuilt `Grid` is also a `grid` value.
- **`morph_gdf`, `multiresolution_morph` and the workflow morph methods warn when the
  grid cannot resolve the smallest regions; the multiresolution methods accept `min_resolution="auto"` and
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

- **`demers_cartogram` packs squares that touch their neighbors.** The preset used the grid layout with a square tiling,
  which put each square in a lattice cell: with the default `size_normalization="max"` the squares covered 15 % of the
  geometry area and none of the neighboring squares touched. It now uses `PackingLayout(shape="square")` with
  `size_normalization="total"` (new parameter), so the total square area equals the geometry area and 76 % of the
  neighboring pairs on the census snapshot touch or nearly touch. The grid version is still available as
  `create_symbol_cartogram(gdf, size, layout="grid", styling=Styling(symbol="square"))`.
- **`CentroidLayout` keeps the requested `spacing`.** The cleanup passes after the attraction cycles are capped, so overlaps
  from the last attraction step could remain: on the census snapshot with `spacing=0.05` 34 of 49 circles were closer than
  the spacing, some touching. A final pass now runs until no overlap is left. Symbols move 3 to 10 % farther from their
  centroids on average (more at larger spacing).
- **Invalid input raises a `ValueError` that names the rows.** A self-intersecting polygon, a null or empty geometry or a
  NaN value in the data column used to fail late, for example with a `GEOSException` from inside the morph; the flow,
  Voronoi and non-contiguous cartograms now stop before the run. Repair invalid polygons with `shapely.make_valid` or
  `carto_flow.geo_utils.repair_polygons`, or pass `validate=False`.
- **The flow plots and animations have a new default look.** A tinted surface, a boundary
  slightly darker than the surface, a truncated `YlGnBu` for continuous data, `RdYlBu_r`
  for errors, a logarithmic colorbar for positive data that spans more than one decade,
  and a colorbar with an outline. `style="classic"` (or `set_style("classic")`) gives the
  look of earlier releases. The `cmap` of `Cartogram.plot`, `plot_cartogram` and the
  three animations of geometry is `None` in the signature and resolves to the palette of
  the style for the data role, else to the former default (`"RdYlGn_r"` or `"viridis"`).
- **`facecolor` is the fill of the flow plot and animate functions.** It takes a column
  of the data, a built-in quantity (`"errors_pct"`, `"density"`, and in the animations
  `"original"`), a numeric array with one value per geometry, a list of arrays for the
  animations of keyframes, or a color (`"red"`, `"#d4e6f1"`), which gives a constant
  fill. A string that is a column and also a built-in quantity or a color is read as
  the column, then as the built-in quantity, and a `UserWarning` is issued. A string
  that is none of these raises `ValueError`: `Cartogram.plot(column="typo")` used to draw
  an uncolored map without a message. The built-in quantities have the same names in the
  static and animated plots (`"errors_pct"` is the column `_morph_error_pct`).
- **`VoronoiOptions.prescale_components` also pre-scales the `premorph`.** With
  `premorph=True` the flow-cartogram morph pre-scales its connected components when the
  Voronoi option is set (on the 52 US regions including Alaska, Hawaii and Puerto Rico
  the morph converges in 58 instead of 168 iterations); a `MorphOptions` given as
  `premorph` is used as given when it sets `prescale_components`. To tell an explicit
  setting from the default, `MorphOptions.prescale_components` is now `None` by default,
  which behaves as `False`.
- **`n_iter` is the total number of iterations of a ladder of resolutions.** Each level
  may use what the previous levels left, and the ladder stops when the total is used up;
  before, every level could use `n_iter`. A list of `MorphOptions` in
  `CartogramWorkflow.morph_multiresolution` keeps one `n_iter` per level.
- **`recompute_every` and `snapshot_every` may be larger than `n_iter`.** The check is removed:
  an interval longer than the run gives one field computation (one snapshot), and the
  remaining iterations of a ladder can fall below the interval.

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

- **`CirclePackingLayout`, `CirclePackingLayoutOptions` and `CirclePackingAdvancedOptions`** are now `PackingLayout`,
  `PackingLayoutOptions` and `PackingAdvancedOptions`, because the layout packs squares as well as circles. The old names
  still work, emit a `DeprecationWarning` and will be removed in the next major release. The layout keys `"packing"` and
  `"topology"` are unchanged.

- **Old parameter names of the plot, animate and preview functions of the four cartogram types.** Each
  concept now has one name, shared by all plot functions (the old names of every module are listed). The old names still work and
  warn, and will be removed in the next major release:

  | function | old name | new name |
  |---|---|---|
  | `Cartogram.plot`, `plot_cartogram`, `plot_comparison` | `column` | `facecolor` |
  | `animate_morph_history`, `animate_workflow` | `color_by` | `facecolor` |
  | `animate_geometry_keyframes` | `column`, `color_values` | `facecolor` |
  | `animate_morph_history`, `animate_geometry_keyframes`, `animate_workflow` | `colorbar`, `colorbar_label`, `colorbar_kwargs` | `legend`, `legend_label`, `legend_kwargs` |
  | `Cartogram.plot`, `plot_cartogram` | `legend_kwds` | `legend_kwargs` |
  | both `preview_modulator` functions | `show_colorbar`, `colorbar_kwargs`, `diff_colorbar_kwargs`, `heatmap_colorbar_kwargs` | `legend`, `legend_kwargs`, `diff_legend_kwargs`, `heatmap_legend_kwargs` |
  | `DensityPlotOptions`, `VelocityPlotOptions` | `colorbar`, `colorbar_kwargs` | `legend`, `legend_kwargs` |
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
  | `SymbolCartogram.plot`, `plot_symbols` | `column`, `label` | `facecolor`, `labels` |
  | `SymbolCartogram.plot`, `plot_symbols` | `legend_kwds`, `edge_legend_kwds`, `hatch_legend_kwds`, `linewidth_legend_kwds`, `alpha_legend_kwds`, `label_legend_kwds` | the same with `_kwargs` |
  | `plot_comparison` (symbol) | `column` | `facecolor` |
  | `plot_adjacency` | `edge_color`, `edge_cmap`, `edge_alpha`, `edge_width`, `node_size`, `colorbar`, `colorbar_kwds` | `link_color`, `link_cmap`, `link_alpha`, `link_linewidth`, `node_markersize`, `link_legend`, `link_legend_kwargs` |
  | `plot_adjacency_heatmap` | `colorbar`, `colorbar_kwds`, `tick_fontsize` | `legend`, `legend_kwargs`, `tick_labelsize` |
  | `plot_displacement` | `arrow_color`, `arrow_alpha` | keys of `arrows_kwargs` |
  | the `plot_tiling` functions | `assigned_color`, `unassigned_color` | `assigned_facecolor`, `unassigned_facecolor` |
  | the plot methods of a tiling and of a tile | `face_color`, `edge_color`, `colormap` | `facecolor`, `edgecolor`, `cmap` |
  | `VoronoiCartogram.plot`, Voronoi `plot_cartogram`, `plot_comparison` | `column` | `facecolor` |
  | `plot_displacement` (Voronoi) | `geometry_style`, `adjacency_style`, `centroid_style`, `position_style`, `displacement_style` | `geometry_kwargs`, `adjacency_kwargs`, `centroid_kwargs`, `position_kwargs`, `displacement_kwargs` |
  | `animate_voronoi_history` | `cell_style`, `point_style` | `cell_kwargs`, `point_kwargs` |
  | `plot_partitions` | `color_by` | `facecolor` (`"area"`, `"category"`, a column name, a list of columns or a color; `None` is `"area"`) |
  | `plot_partitions` | `background_color`, `complement_color`, `complement_color_by` | `background_facecolor`, `complement_facecolor` (a column or a color) |
  | `plot_partitions`, `plot_dot_density` | `palette` | `cmap` accepts the `{category: color}` dict (when `palette` and a colormap name were both given, the dict replaces the name and the other categories take the `tab10` colors) |
  | `plot_dot_density` | `size`, `base_color`, `base_edgecolor`, `base_linewidth`, `base_alpha` | `markersize`, `reference_facecolor`, `reference_edgecolor`, `reference_linewidth`, `reference_alpha` |

- **`morph_gdf` and `multiresolution_morph`** are deprecated and will be removed in the
  next major release; use `create_flow_cartogram`, with `MorphOptions(grid=LadderSpec(...))`
  for a ladder of resolutions. Per-level option lists remain available through
  `CartogramWorkflow.morph_multiresolution`.
- **`MorphOptions.grid_size`, `grid_margin` and `grid_square`** are deprecated:
  `grid_size` is an alias of `grid` (which now also takes an int, a tuple, `GridSpec`,
  `LadderSpec`, `"auto"` or a `Grid`), and the margin and squareness belong in a `GridSpec`
  or `LadderSpec`. `grid_margin` and `grid_square` default to `None` (0.5 and `False` for an
  int or tuple `grid`).

- **`Cartogram.latest`, `CartogramWorkflow.latest` and `History.latest()`.**
  `Cartogram.best` is the snapshot of the returned state and `Cartogram.last`
  the last stored one; `CartogramWorkflow.last` is the most recent result and
  `History.last()` the last snapshot. `Cartogram.latest` returns `best`. On a
  run that does not converge, `best` is the earlier iterate that the run now
  returns, not the final iterate.
  ([#110](https://github.com/bright-fakl/carto-flow/pull/110))

### Fixed

- **Square flow grids have a cell size that matches their coordinates.** With `grid_square=True` (the default of
  `LadderSpec`, so every multiresolution run), the cell size was set to the average of the two directions while the
  domain kept its own extent in the longer one, so the nodes of the grid were spaced slightly differently from the cell size
  used to locate a vertex (0.4 % on the county example below, a third of a cell at the far edge). A vertex could be
  placed in the wrong cell and interpolate with a weight outside 0 to 1, which made the displacement jump at cell boundaries where the velocity changes fast and folded polygons. The domain now grows,
  centered, to a whole number of cells of the larger spacing. On the 3,108 counties on a square grid of 128 the number of
  invalid output polygons after 200 iterations falls from 1,350 to 70; states, districts and counties at other grids have
  fewer invalid polygons in most cases and converge as before. The grid of a square run is up to one cell larger than before.
- **The stall rule requires an improvement that follows from the step, so it no longer ends runs on fine grids that
  converge.** A step moves the fastest point `dt` cells, so on a finer grid the error falls more slowly per iteration,
  and the fixed required improvement per window (2 %) stopped runs that were making steady progress: Arizona's 15
  counties stalled at 50 iterations at grids 512 and 1024 although they converge in 384 and 758 iterations, the 3,108
  counties stalled at their start error (929 % mean) at 1024, and the 49 states stalled at 2048 at 168 % mean. A window
  now has to lower the error by `stall_min_efficiency` (default 0.05) times the distance the boundaries moved in the
  window divided by the area-weighted size of the regions, so the requirement falls with the cell size, `dt` and the
  adaptive step, and rises for small regions. The counties at 1024 now reach a 4.2 % mean error (median 0.1 %) in 700
  iterations. The option `stall_min_improvement` of the unreleased stall rule is replaced by `stall_min_efficiency`.
- **`cmap={category: color}` works in the flow, Voronoi and non-contiguous plots.** The dictionary was passed to matplotlib as a colormap and raised `TypeError: unhashable type: 'dict'`; the categories that are not in it take the `tab10` colors.
- **Data-driven edges of the symbol plots are visible without an `edgecolor`, and the scalar channels take a scale.** The edge color defaulted to the boundary color of the style, a faint color made for separating symbols, or to `"none"` in `"classic"`, so a data-driven `linewidth` or a `hatch` could not be seen. A style has a new `"stroke"` channel (the color of an edge that carries data; `"#3b4650"` in `"default"`, `"#c9d1d9"` in `"dark"`, black in `"classic"`) which `plot_symbols` uses when `linewidth` is a column or `hatch` is set and no `edgecolor` is given; the hatch warning is raised only for an explicit `edgecolor="none"`.
- **`legend_kwargs={"label": ...}` of the symbol plots sets the label of the colorbar** (it was ignored in favor of the column name).
- **`plot_partitions` drew the complement black when the partitions were colored by a continuous value**
  (`facecolor="area"` or a numeric column): the missing color value of the complement became NaN and
  was colored with the "bad" color of the colormap instead of `complement_facecolor`.
- **`prescale_components` moves the landmarks with their component.** The pre-scaling
  scaled the geometries but left the landmarks where they were, so for a region that was
  scaled down they started outside it (only 39 % of the dots of a shrunken region of a
  two-component test case were inside it at the end). Landmarks are scaled with the
  component that contains them; points outside every region are not moved. The
  displacement coordinates are not: they are not tied to the geometries (a grid carried
  along to show the deformation of the plane may be wanted without the pre-scaling); the
  new `MorphOptions.prescale_coords` scales them too.
- **Landmarks of different geometry types (points with lines or polygons) raised an
  error.** Landmarks may now mix any geometry types. Their coordinates are read and
  written with `shapely.get_coordinates` and `shapely.set_coordinates`, which also
  makes rebuilding the landmark geometries for each snapshot about 30 times faster.
  Z values of landmarks are dropped.
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
