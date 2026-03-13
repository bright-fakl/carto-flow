# Workflow System

## Overview

The flow cartogram module provides two API levels:

- **Low-level**: `morph_geometries(geometries, values, options)` operates on raw Shapely geometry sequences and returns a `Cartogram` result object. It has no dependency on geopandas.
- **High-level**: `CartogramWorkflow` wraps `morph_geometries()` with GeoDataFrame input/output, automatic area-unit handling, and multi-run state management. This is the primary entry point for most use cases.

The two are related as follows: each call to `CartogramWorkflow.morph()` invokes `morph_geometries()` internally and wraps the result in a `Cartogram` stored on the workflow.

Relevant source files: [workflow.py](https://github.com/bright-fakl/carto-flow/blob/main/src/carto_flow/flow_cartogram/workflow.py), [cartogram.py](https://github.com/bright-fakl/carto-flow/blob/main/src/carto_flow/flow_cartogram/cartogram.py), [serialization.py](https://github.com/bright-fakl/carto-flow/blob/main/src/carto_flow/flow_cartogram/serialization.py).

---

## CartogramWorkflow

### Purpose

`CartogramWorkflow` manages a sequence of morphing runs on a fixed input GeoDataFrame. The original (unmorphed) state is always preserved at index 0; each subsequent morph appends a new `Cartogram` to the result list. The target density is always computed from the original data, so it remains invariant across runs and refinements.

### Construction

```python
CartogramWorkflow(
    gdf,           # GeoDataFrame with polygon geometries
    column,        # name of the value column
    landmarks=None,  # optional GeoDataFrame of landmark geometries
    coords=None,   # optional coordinate array/grid/mesh to displace
    options=None,  # MorphOptions (default options used if None)
    density_per=None,  # unit string: 'm2', 'km2', 'ha', 'acre', 'sqft', 'sqmi'
)
```

`density_per` is a convenience parameter. If supplied, the workflow inspects the GeoDataFrame's CRS to determine the native area unit and computes the `area_scale` for `MorphOptions` automatically. For example, `density_per='km2'` when the CRS is in metres sets `area_scale=1e-6` so that densities are expressed per km². Alternatively, `area_scale` can be set directly on `MorphOptions`.

On construction the workflow creates the index-0 `Cartogram` representing the unmodified input (`status = MorphStatus.ORIGINAL`).

### Internal State

| Attribute | Description |
|---|---|
| `_original_gdf` | Copy of the input GeoDataFrame. Never modified. |
| `_column` | Value column name. |
| `_results` | `list[Cartogram]`. Index 0 = original; index 1+ = morph results. |
| `_original_landmarks_gdf` | Copy of landmark GeoDataFrame, or `None`. |

### Morphing

**`morph(options=None, **overrides) → Cartogram`**

Runs one complete morphing pass. The input geometry for the pass is taken from the latest `Cartogram` in `_results`; the target density is recomputed from the original data. Options can be fully replaced or individual fields overridden by passing keyword arguments.

**`morph_multiresolution(min_resolution, levels, margin, square, options=None) → Cartogram`**

Builds a dyadic grid hierarchy with `levels` grids and runs one morphing pass per grid level, from coarsest to finest. Each pass's output geometry is the input to the next. Each pass appends a separate `Cartogram` to `_results`. Execution stops early if the coarser-level result has already converged.

`prescale_components` is applied only at the first (coarsest) level; subsequent levels disable it automatically to avoid rescaling already-morphed geometry.

### Access and State Management

```python
workflow[0]         # original Cartogram
workflow[-1]        # latest Cartogram (same as workflow.latest)
workflow.original   # same as workflow[0]
workflow.latest     # same as workflow[-1]
workflow.is_morphed  # True if any morph runs exist

workflow.pop(n=1)   # remove and return last n Cartograms (cannot remove original)
workflow.reset()    # discard all morph results, keeping only original
```

### Export

```python
workflow.to_geodataframe(
    run_id=None,         # index into _results; defaults to latest
    iteration=None,      # iteration within that run; defaults to latest
    include_errors=True,
    include_density=True,
)
```

Delegates to `Cartogram.to_geodataframe()`. Returns a GeoDataFrame with the morphed geometry and optional error/density columns.

---

## Cartogram

`Cartogram` is a dataclass that holds the complete result of a single `morph_geometries()` call. It is created by the algorithm and stored on `CartogramWorkflow._results`, but it can also be used independently via the low-level API.

### Data Held

| Field | Type | Description |
|---|---|---|
| `snapshots` | `History[CartogramSnapshot]` | Snapshots of the morphed state at saved iterations |
| `convergence` | `ConvergenceHistory` | Scalar error metrics recorded at every iteration |
| `status` | `MorphStatus` | Termination reason |
| `niterations` | `int` | Number of iterations completed |
| `duration` | `float` | Wall-clock time in seconds |
| `options` | `MorphOptions` or `None` | Options used for this run |
| `grid` | `Grid` or `None` | Grid used for this run |
| `target_density` | `float` or `None` | Equilibrium density $\rho^*$ |
| `internals` | `History[CartogramInternalsSnapshot]` or `None` | Grid-level debug snapshots, if `save_internals=True` |

Source references (`_source_gdf`, `_value_column`) are stored as private attributes and enable `to_geodataframe()` and `plot()` to reconstruct a GeoDataFrame without requiring the caller to supply the original data again.

### MorphStatus

| Value | Meaning |
|---|---|
| `ORIGINAL` | Unmorphed placeholder; no algorithm was run |
| `CONVERGED` | Both `mean_tol` and `max_tol` thresholds were met |
| `STALLED` | Max error increased for `stall_patience` consecutive iterations |
| `COMPLETED` | `n_iter` iterations reached without converging |
| `FAILED` | An exception occurred during morphing |

### Accessing Results

All accessor methods accept an optional `iteration` parameter. When omitted, the value defaults to the latest available snapshot.

```python
cartogram.latest                      # most recent CartogramSnapshot
cartogram.get_errors(iteration=None)  # MorphErrors with per-geometry arrays
cartogram.get_geometry(iteration=None)
cartogram.get_density(iteration=None)
cartogram.get_landmarks(iteration=None)
cartogram.get_coords(iteration=None)
```

For convergence history (scalar metrics, available for every iteration):

```python
cartogram.get_convergence_errors(iteration=None)  # returns ErrorRecord
cartogram.convergence.to_dict()                   # dict of arrays for plotting
```

### Export

```python
cartogram.to_geodataframe(
    iteration=None,
    include_errors=True,
    include_density=True,
)
```

Returns a GeoDataFrame with the morphed geometry replacing the original, plus optional columns `mean_error_pct`, `max_error_pct`, and per-geometry `error_pct` and `density`.

```python
cartogram.plot(column=None, iteration=None, cmap='RdYlGn_r', legend=True, ax=None)
```

Delegates to `visualization.plot_cartogram()` and returns a `CartogramPlotResult` containing the matplotlib axes and named artist references.

---

## Serialization

**JSON** (`cartogram.save(path)` / `Cartogram.load(path)`)

Saves the latest snapshot's geometries as GeoJSON plus the convergence history, options (numeric fields only), grid parameters, and status. A loaded `Cartogram` contains only one snapshot but supports `to_geodataframe()` and `plot()` immediately. JSON format is suitable for archiving a final result or sharing with non-Python tools.

**Pickle** (`save_workflow(workflow, path)` / `load_workflow(path)`)

Serializes the entire `CartogramWorkflow` including all runs and full Python objects. A loaded workflow can continue morphing from where it left off. Pickle format is intended for checkpointing long-running sessions.

**Convergence history export** (`export_history(cartogram, path)`)

Writes the `ConvergenceHistory` arrays to CSV or JSON. Useful for producing convergence plots outside Python.
