"""
===========================================================================
Serialization utilities for cartogram results
===========================================================================

Export, save, and load utilities for morphing results and state.

Main Components
---------------
- export_result: Export morphed geometries to GeoJSON, GeoPackage, or Shapefile
- save_state: Serialize MorphComputer state to file
- load_state: Load MorphComputer state from file
- export_history: Export convergence history to CSV or JSON

Example
-------
>>> from carto_flow.shape_morpher import morph_gdf, MorphOptions
>>> from carto_flow.shape_morpher.serialization import export_result, export_history
>>>
>>> result = morph_gdf(gdf, 'population', options=MorphOptions.preset_fast())
>>> export_result(result, 'cartogram.geojson')
>>> export_history(result.history, 'convergence.csv')
"""

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from .computer import MorphComputer
    from .history import History
    from .result import MorphResult

__all__ = [
    "export_history",
    "export_result",
    "load_state",
    "save_state",
]


def export_result(
    result: "MorphResult",
    path: Union[str, Path],
    driver: Optional[str] = None,
) -> None:
    """Export morphed geometries to a geospatial file format.

    Parameters
    ----------
    result : MorphResult
        Morphing result containing geometries to export
    path : str or Path
        Output file path. Format is inferred from extension if driver not specified.
    driver : str, optional
        Output format driver (e.g., 'GeoJSON', 'GPKG', 'ESRI Shapefile').
        If None, inferred from file extension.

    Examples
    --------
    >>> export_result(result, 'output.geojson')
    >>> export_result(result, 'output.gpkg', driver='GPKG')
    """
    path = Path(path)

    # Get the GeoDataFrame from result
    gdf = result.geometries
    if hasattr(gdf, "to_file"):
        if driver:
            gdf.to_file(path, driver=driver)
        else:
            gdf.to_file(path)
    else:
        raise TypeError("result.geometries must be a GeoDataFrame with to_file method")


def save_state(
    computer: "MorphComputer",
    path: Union[str, Path],
) -> None:
    """Serialize MorphComputer state to a pickle file.

    Saves the complete state including original data, current results,
    refinement history, and options for later resumption.

    Parameters
    ----------
    computer : MorphComputer
        MorphComputer instance to save
    path : str or Path
        Output file path (typically .pkl extension)

    Examples
    --------
    >>> computer = MorphComputer(gdf, 'population')
    >>> computer.morph()
    >>> save_state(computer, 'checkpoint.pkl')
    """
    path = Path(path)
    with open(path, "wb") as f:
        pickle.dump(computer, f)


def load_state(path: Union[str, Path]) -> "MorphComputer":
    """Load MorphComputer state from a pickle file.

    Parameters
    ----------
    path : str or Path
        Path to saved state file

    Returns
    -------
    MorphComputer
        Restored MorphComputer instance with full state

    Examples
    --------
    >>> computer = load_state('checkpoint.pkl')
    >>> computer.morph()  # Continue refinement
    """
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


def export_history(
    history: "History",
    path: Union[str, Path],
    output_format: str = "csv",
) -> None:
    """Export convergence history to CSV or JSON format.

    Parameters
    ----------
    history : History
        History object containing iteration snapshots
    path : str or Path
        Output file path
    output_format : str, default='csv'
        Output format: 'csv' or 'json'

    Examples
    --------
    >>> export_history(result.history, 'convergence.csv')
    >>> export_history(result.history, 'convergence.json', output_format='json')
    """
    path = Path(path)

    # Extract data from history
    data = []
    for snapshot in history.snapshots:
        row = {"iteration": snapshot.iteration}
        if hasattr(snapshot, "mean_error") and snapshot.mean_error is not None:
            row["mean_error"] = snapshot.mean_error
        if hasattr(snapshot, "max_error") and snapshot.max_error is not None:
            row["max_error"] = snapshot.max_error
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
