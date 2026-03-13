# History and Snapshot System

## Overview

The flow cartogram algorithm runs for up to hundreds of iterations. The history system provides two complementary mechanisms for tracking what happened:

1. **`ConvergenceHistory`** records four scalar error metrics at *every* iteration. It is cheap to maintain and supports full convergence analysis without large memory overhead.
2. **`History`** holds `CartogramSnapshot` objects at *selected* iterations. Each snapshot includes the full set of morphed geometries and per-geometry error arrays. Snapshots are expensive relative to scalar records, so they are saved selectively.

These two mechanisms are independent and serve different purposes: `ConvergenceHistory` is for plotting and detecting convergence trends; `History` snapshots are for examining or exporting the deformed geometries at specific points in the run.

Both are stored on the `Cartogram` result object. A third variant, `CartogramInternalsSnapshot`, captures raw grid-level fields for debugging and is stored separately in `Cartogram.internals`.

Relevant source files: [history.py](https://github.com/bright-fakl/carto-flow/blob/main/src/carto_flow/flow_cartogram/history.py), [cartogram.py](https://github.com/bright-fakl/carto-flow/blob/main/src/carto_flow/flow_cartogram/cartogram.py).

---

## Snapshot Class Hierarchy

```
BaseSnapshot (abstract)
├── CartogramSnapshot      — user-facing morphed state
└── CartogramInternalsSnapshot  — grid-level debug state
```

### BaseSnapshot

`BaseSnapshot` is the abstract interface required by the `History` container. Any snapshot class that implements this interface can be stored in a `History` object.

Required interface:
- `iteration: int` — mandatory field identifying which iteration the snapshot belongs to
- `has_variable(name: str) -> bool` — returns `True` if the named attribute exists and is not `None`
- `get_variable(name: str) -> Any` — retrieves the attribute value
- `get_all_variables() -> dict` — returns all non-`None` fields as a dictionary

### CartogramSnapshot (External Snapshots)

Captures the user-facing algorithm state at a saved iteration. Fields:

| Field | Type | Description |
|---|---|---|
| `iteration` | `int` | Iteration number |
| `geometry` | sequence of geometries | Deformed polygon geometries at this iteration |
| `landmarks` | sequence of geometries or `None` | Deformed landmark geometries, if landmarks were provided |
| `coords` | array-like or `None` | Deformed coordinates, in the same format as the input |
| `errors` | `MorphErrors` or `None` | Full error metrics including per-geometry log₂ arrays and scalar aggregates |
| `density` | `np.ndarray` or `None` | Per-geometry density values at this iteration |

`MorphErrors` holds both per-geometry arrays (`log_errors`, `errors_pct`) and scalars (`mean_log_error`, `max_log_error`, `mean_error_pct`, `max_error_pct`).

### CartogramInternalsSnapshot (Internal Snapshots)

Captures the raw grid-level state used by the FFT solver. Created only when `MorphOptions.save_internals = True`. Fields:

| Field | Type | Description |
|---|---|---|
| `iteration` | `int` | Iteration number |
| `rho` | `np.ndarray` | Density field on the grid, shape `(ny, nx)` |
| `vx`, `vy` | `np.ndarray` | Effective velocity components after any modulation, shape `(ny, nx)` |
| `geometry_mask` | `np.ndarray` | Cell-to-geometry assignment: value $k$ means the cell is inside geometry $k$; $-1$ means exterior, shape `(ny, nx)` |

Internal snapshots are stored in `Cartogram.internals` (a separate `History` instance), not in `Cartogram.snapshots`. This keeps external and internal histories independent.

---

## History Container

`History` is a generic ordered container for any `BaseSnapshot` subclass. It supports several access patterns:

**Index access:**
```python
history[0]      # first snapshot
history[-1]     # most recent snapshot
history[1:3]    # slice returning a list
```

**Iteration-based lookup:**
```python
history.get_snapshot(iteration_num)         # returns snapshot or None
history.get_variable_at_iteration('errors', 50)  # variable value at specific iteration
```

**Extracting a variable across all snapshots:**
```python
history.get_variable_history('errors')  # list of MorphErrors objects
```

**Convenience:**
```python
history.latest()          # most recent snapshot
history.get_iterations()  # sorted list of all iteration numbers
```

---

## ConvergenceHistory

`ConvergenceHistory` stores scalar error metrics for every iteration using pre-allocated NumPy arrays. Memory is approximately 40 bytes per iteration, independent of the number of geometries.

Stored arrays:

| Array | Description |
|---|---|
| `iterations` | Iteration numbers |
| `mean_log_errors` | Mean log₂ area error across all geometries |
| `max_log_errors` | Maximum log₂ area error |
| `mean_errors_pct` | Mean percentage area error |
| `max_errors_pct` | Maximum percentage area error |

Access patterns:

```python
convergence[i]                  # returns ErrorRecord for index i
convergence.get_by_iteration(50)  # lookup by iteration number
convergence.to_dict()           # returns dict of arrays (for plotting)
```

`ErrorRecord` is a lightweight dataclass with five fields: `iteration`, `mean_log_error`, `max_log_error`, `mean_error_pct`, `max_error_pct`.

`ConvergenceHistory` calls `finalize()` internally at the end of the algorithm run to trim the pre-allocated arrays to the actual number of iterations completed.

---

## When Snapshots Are Taken

The algorithm loop in `morph_geometries()` controls when snapshots are created:

- **`ConvergenceHistory`**: updated at *every* iteration, unconditionally.
- **`CartogramSnapshot`**: created at:
  - every `snapshot_every` iterations (if the option is set; default is `None`, meaning periodic saving is disabled)
  - the final iteration, regardless of termination reason (convergence, stall, or iteration limit)
- **`CartogramInternalsSnapshot`**: created at the same moments as `CartogramSnapshot`, but only if `save_internals=True`.

This design means that a `Cartogram` object always contains at minimum one `CartogramSnapshot` (the final state) and a complete `ConvergenceHistory`.
