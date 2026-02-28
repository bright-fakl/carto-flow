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
>>> from carto_flow.shape_morpher import morph_gdf, MorphOptions
>>> from carto_flow.shape_morpher.serialization import save_cartogram
>>>
>>> cartogram = morph_gdf(gdf, 'population', options=MorphOptions.preset_fast())
>>> save_cartogram(cartogram, 'cartogram.json')
>>> cartogram.save('cartogram.gpkg')  # Or use the Cartogram.save() method
"""

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Union

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
    path: Union[str, Path],
) -> None:
    """Save a Cartogram to JSON format.

    Saves the cartogram metadata, status, and error metrics.
    For full geometry preservation, use cartogram.save() with a
    geospatial format like .gpkg.

    Parameters
    ----------
    cartogram : Cartogram
        Cartogram instance to save
    path : str or Path
        Output file path (typically .json extension)

    Examples
    --------
    >>> save_cartogram(cartogram, 'result.json')
    """
    path = Path(path)

    # Extract serializable data
    data = {
        "status": cartogram.status.value if hasattr(cartogram.status, "value") else str(cartogram.status),
        "niterations": cartogram.niterations,
        "duration": cartogram.duration,
        "target_density": cartogram.target_density,
    }

    # Add error metrics if available
    errors = cartogram.get_errors()
    if errors:
        data["errors"] = {
            "mean_log_error": errors.mean_log_error,
            "max_log_error": errors.max_log_error,
            "mean_error_pct": errors.mean_error_pct,
            "max_error_pct": errors.max_error_pct,
        }

    # Add grid info if available
    if cartogram.grid:
        data["grid"] = {
            "sx": cartogram.grid.sx,
            "sy": cartogram.grid.sy,
            "dx": cartogram.grid.dx,
            "dy": cartogram.grid.dy,
        }

    # Add options info if available
    if cartogram.options:
        data["options"] = {
            "n_iter": cartogram.options.n_iter,
            "dt": cartogram.options.dt,
            "mean_tol": cartogram.options.mean_tol,
            "max_tol": cartogram.options.max_tol,
        }

    # Add snapshot history summary
    if cartogram.snapshots:
        data["snapshots"] = {
            "count": len(cartogram.snapshots.snapshots),
            "iterations": cartogram.snapshots.get_iterations(),
        }

    # Add convergence history (all iterations)
    if cartogram.convergence is not None and len(cartogram.convergence) > 0:
        data["convergence"] = {
            "count": len(cartogram.convergence),
            "iterations": cartogram.convergence.iterations.tolist(),
            "mean_log_errors": cartogram.convergence.mean_log_errors.tolist(),
            "max_log_errors": cartogram.convergence.max_log_errors.tolist(),
            "mean_errors_pct": cartogram.convergence.mean_errors_pct.tolist(),
            "max_errors_pct": cartogram.convergence.max_errors_pct.tolist(),
        }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_cartogram(path: Union[str, Path]) -> "Cartogram":
    """Load a Cartogram from JSON format.

    Note: This loads metadata only. For full geometry restoration,
    the original GeoDataFrame and morphing must be re-run, or use
    pickle-based save_workflow/load_workflow.

    Parameters
    ----------
    path : str or Path
        Path to saved JSON file

    Returns
    -------
    Cartogram
        Partial Cartogram with metadata (no geometries)

    Raises
    ------
    NotImplementedError
        Full Cartogram restoration from JSON is not yet implemented.
        Use save_workflow/load_workflow for complete state preservation.
    """
    raise NotImplementedError(
        "Full Cartogram restoration from JSON is not yet implemented. "
        "Use save_workflow/load_workflow for complete state preservation, "
        "or re-run the morphing with the original data."
    )


def save_workflow(
    workflow: "CartogramWorkflow",
    path: Union[str, Path],
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


def load_workflow(path: Union[str, Path]) -> "CartogramWorkflow":
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
    path: Union[str, Path],
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
            row = {"iteration": snapshot.iteration}

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
