"""
Serialization utilities for cartogram results.

Export, save, and load utilities for Cartogram and CartogramWorkflow.

Functions
---------
save_cartogram
    Save a Cartogram to JSON format.
load_cartogram
    Load a Cartogram from JSON format.
save_workflow
    Serialize CartogramWorkflow state to file.
load_workflow
    Load CartogramWorkflow state from file.
export_history
    Export convergence history to CSV or JSON.

Examples
--------
>>> from carto_flow.flow_cartogram import morph_gdf, MorphOptions
>>> from carto_flow.flow_cartogram.serialization import save_cartogram
>>>
>>> cartogram = morph_gdf(gdf, 'population', options=MorphOptions.preset_fast())
>>> save_cartogram(cartogram, 'cartogram.json')
>>> cartogram.save('cartogram.gpkg')  # Or use the Cartogram.save() method
"""

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from .cartogram import Cartogram
    from .history import ConvergenceHistory, History
    from .workflow import CartogramWorkflow

__all__ = [
    "export_history",
    "load_cartogram",
    "load_workflow",
    "save_cartogram",
    "save_workflow",
]


def save_cartogram(
    cartogram: "Cartogram",
    path: str | Path,
) -> None:
    """Save a Cartogram to JSON format.

    Saves geometries, source data, error metrics, and convergence history
    so the Cartogram can be fully restored with load_cartogram() /
    Cartogram.load().

    Parameters
    ----------
    cartogram : Cartogram
        Cartogram instance to save
    path : str or Path
        Output file path (typically .json extension)

    Examples
    --------
    >>> save_cartogram(cartogram, 'result.json')
    >>> loaded = load_cartogram('result.json')
    """
    path = Path(path)

    data: dict = {
        "status": cartogram.status.value if hasattr(cartogram.status, "value") else str(cartogram.status),
        "niterations": cartogram.niterations,
        "duration": cartogram.duration,
        "target_density": cartogram.target_density,
    }

    # Grid with full bounds for reconstruction
    if cartogram.grid:
        data["grid"] = {
            "bounds": list(cartogram.grid.bounds),
            "sx": cartogram.grid.sx,
            "sy": cartogram.grid.sy,
        }

    # Options (numeric fields only — callables are not JSON-serializable)
    if cartogram.options:
        data["options"] = {
            "n_iter": cartogram.options.n_iter,
            "dt": cartogram.options.dt,
            "mean_tol": cartogram.options.mean_tol,
            "max_tol": cartogram.options.max_tol,
        }

    # Latest morphed geometries
    latest = cartogram.snapshots.latest() if cartogram.snapshots else None
    if latest is not None and latest.geometry is not None:
        data["latest_geometries"] = [geom.__geo_interface__ for geom in latest.geometry]
        if latest.density is not None:
            data["density"] = latest.density.tolist()

    # Error metrics — scalars + per-geometry arrays
    errors = cartogram.get_errors()
    if errors is not None:
        data["errors"] = {
            "mean_log_error": errors.mean_log_error,
            "max_log_error": errors.max_log_error,
            "mean_error_pct": errors.mean_error_pct,
            "max_error_pct": errors.max_error_pct,
            "log_errors": errors.log_errors.tolist(),
            "errors_pct": errors.errors_pct.tolist(),
        }

    # Source GeoDataFrame — original geometries + attribute columns
    if cartogram._source_gdf is not None:
        src = cartogram._source_gdf
        if src.crs is not None:
            data["crs"] = src.crs.to_string()
        data["index"] = list(src.index)
        non_geom_cols = [c for c in src.columns if c != src.geometry.name]
        data["source_columns"] = non_geom_cols
        # Convert column values to Python scalars for JSON serialisation
        records = src[non_geom_cols].to_dict(orient="records")
        data["source_records"] = [
            {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()} for row in records
        ]
        data["source_geometries"] = [geom.__geo_interface__ for geom in src.geometry]

    if cartogram._value_column is not None:
        data["value_column"] = cartogram._value_column

    # Convergence history (all iterations)
    if cartogram.convergence is not None and len(cartogram.convergence) > 0:
        data["convergence"] = {
            "iterations": cartogram.convergence.iterations.tolist(),
            "mean_log_errors": cartogram.convergence.mean_log_errors.tolist(),
            "max_log_errors": cartogram.convergence.max_log_errors.tolist(),
            "mean_errors_pct": cartogram.convergence.mean_errors_pct.tolist(),
            "max_errors_pct": cartogram.convergence.max_errors_pct.tolist(),
        }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_cartogram(path: str | Path) -> "Cartogram":
    """Load a Cartogram from JSON format.

    Restores a Cartogram saved by save_cartogram() / Cartogram.save(),
    including morphed geometries, source GeoDataFrame, error metrics,
    and convergence history.

    Parameters
    ----------
    path : str or Path
        Path to saved JSON file

    Returns
    -------
    Cartogram
        Fully restored Cartogram with one snapshot (the final result).
        Supports to_geodataframe() and plot() immediately after loading.

    Raises
    ------
    ValueError
        If the file was saved without geometries (old metadata-only format).
    """
    import numpy as np
    from shapely.geometry import shape

    from .cartogram import Cartogram
    from .errors import MorphErrors
    from .grid import Grid
    from .history import CartogramSnapshot, ConvergenceHistory, History
    from .options import MorphStatus

    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    if "latest_geometries" not in data:
        raise ValueError(
            "This JSON contains metadata only and cannot be restored. "
            "Re-save the cartogram using Cartogram.save() with the current version."
        )

    # Morphed geometries
    morphed_geoms = [shape(g) for g in data["latest_geometries"]]

    # Error metrics
    errors = None
    if "errors" in data:
        e = data["errors"]
        errors = MorphErrors(
            log_errors=np.array(e["log_errors"]),
            mean_log_error=e["mean_log_error"],
            max_log_error=e["max_log_error"],
            errors_pct=np.array(e["errors_pct"]),
            mean_error_pct=e["mean_error_pct"],
            max_error_pct=e["max_error_pct"],
        )

    density = np.array(data["density"]) if "density" in data else None

    niterations = data.get("niterations", 0)
    snapshot = CartogramSnapshot(
        iteration=niterations,
        geometry=morphed_geoms,
        errors=errors,
        density=density,
    )
    history: History[CartogramSnapshot] = History()
    history.add_snapshot(snapshot)

    # Source GeoDataFrame
    source_gdf = None
    if "source_geometries" in data:
        import geopandas as gpd

        src_geoms = [shape(g) for g in data["source_geometries"]]
        records = data.get("source_records", [{} for _ in src_geoms])
        index = data.get("index", list(range(len(src_geoms))))
        source_gdf = gpd.GeoDataFrame(records, geometry=src_geoms, index=index)
        if "crs" in data:
            source_gdf = source_gdf.set_crs(data["crs"])

    # Convergence history
    conv = None
    if "convergence" in data:
        c = data["convergence"]
        conv = ConvergenceHistory()
        conv._iterations = np.array(c["iterations"], dtype=np.int64)
        conv._mean_log_errors = np.array(c["mean_log_errors"])
        conv._max_log_errors = np.array(c["max_log_errors"])
        conv._mean_errors_pct = np.array(c["mean_errors_pct"])
        conv._max_errors_pct = np.array(c["max_errors_pct"])
        conv._size = len(c["iterations"])

    # Grid
    grid = None
    if "grid" in data and "bounds" in data["grid"]:
        g = data["grid"]
        grid = Grid(bounds=tuple(g["bounds"]), size=(g["sx"], g["sy"]))

    # Status
    try:
        status = MorphStatus(data.get("status", "completed"))
    except ValueError:
        status = MorphStatus.COMPLETED

    return Cartogram(
        snapshots=history,
        convergence=conv,
        status=status,
        niterations=niterations,
        duration=data.get("duration", 0.0),
        target_density=data.get("target_density"),
        grid=grid,
        _source_gdf=source_gdf,
        _value_column=data.get("value_column"),
    )


def save_workflow(
    workflow: "CartogramWorkflow",
    path: str | Path,
) -> None:
    """Serialize CartogramWorkflow state to a pickle file.

    Saves the complete state including original data, all results,
    and options for later resumption.

    Parameters
    ----------
    workflow : CartogramWorkflow
        CartogramWorkflow instance to save
    path : str or Path
        Output file path (typically .pkl extension)

    Examples
    --------
    >>> workflow = CartogramWorkflow(gdf, 'population')
    >>> workflow.morph()
    >>> save_workflow(workflow, 'checkpoint.pkl')
    """
    path = Path(path)
    with open(path, "wb") as f:
        pickle.dump(workflow, f)


def load_workflow(path: str | Path) -> "CartogramWorkflow":
    """Load CartogramWorkflow state from a pickle file.

    Parameters
    ----------
    path : str or Path
        Path to saved state file

    Returns
    -------
    CartogramWorkflow
        Restored CartogramWorkflow instance with full state

    Examples
    --------
    >>> workflow = load_workflow('checkpoint.pkl')
    >>> workflow.morph()  # Continue refinement
    """
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


def export_history(
    history: Union["History", "ConvergenceHistory"],
    path: str | Path,
    output_format: str = "csv",
) -> None:
    """Export convergence history to CSV or JSON format.

    Parameters
    ----------
    history : History or ConvergenceHistory
        History object containing iteration snapshots, OR
        ConvergenceHistory with scalar error metrics for all iterations.
    path : str or Path
        Output file path
    output_format : str, default='csv'
        Output format: 'csv' or 'json'

    Examples
    --------
    >>> export_history(cartogram.convergence, 'convergence.csv')
    >>> export_history(cartogram.snapshots, 'convergence.json', output_format='json')
    """
    from .history import ConvergenceHistory

    path = Path(path)

    # Extract data from history
    data = []

    # Handle ConvergenceHistory (preferred - has all iterations)
    if isinstance(history, ConvergenceHistory):
        for i in range(len(history)):
            data.append({
                "iteration": int(history.iterations[i]),
                "mean_log_error": float(history.mean_log_errors[i]),
                "max_log_error": float(history.max_log_errors[i]),
                "mean_error_pct": float(history.mean_errors_pct[i]),
                "max_error_pct": float(history.max_errors_pct[i]),
            })
    else:
        # Fall back to extracting from snapshots
        for snapshot in history.snapshots:
            row: dict[str, Any] = {"iteration": snapshot.iteration}

            # Extract error metrics from MorphErrors object
            if hasattr(snapshot, "errors") and snapshot.errors is not None:
                row["mean_log_error"] = snapshot.errors.mean_log_error
                row["max_log_error"] = snapshot.errors.max_log_error
                row["mean_error_pct"] = snapshot.errors.mean_error_pct
                row["max_error_pct"] = snapshot.errors.max_error_pct

            data.append(row)

    if output_format.lower() == "json":
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    elif output_format.lower() == "csv":
        if not data:
            return
        # Write CSV manually to avoid pandas dependency
        with open(path, "w") as f:
            headers = list(data[0].keys())
            f.write(",".join(headers) + "\n")
            for row in data:
                values = [str(row.get(h, "")) for h in headers]
                f.write(",".join(values) + "\n")
    else:
        raise ValueError(f"Unsupported format: {output_format}. Use 'csv' or 'json'.")
