"""Data preparation for symbol cartogram layouts.

This module provides utilities to prepare GeoDataFrame data for layout
algorithms, including computing symbol sizes based on data values and
extracting geographic information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .adjacency import compute_adjacency
from .options import AdjacencyMode

if TYPE_CHECKING:
    import geopandas as gpd


@dataclass
class LayoutData:
    """Preprocessed data for layout algorithms.

    Created by prepare_layout_data(). Advanced users can inspect/modify
    before passing to Layout.compute().

    Attributes
    ----------
    positions : NDArray[np.floating]
        Initial positions (centroids), shape (n, 2).
    sizes : NDArray[np.floating]
        Area-equivalent symbol sizes (circle radii), shape (n,).
        Symbol area = pi x size^2. Use area_factor to convert to native
        half-extent for rendering.
    adjacency : NDArray[np.floating]
        Adjacency matrix, shape (n, n).
    bounds : tuple[float, float, float, float]
        Geographic bounds (xmin, ymin, xmax, ymax).
    mean_area : float
        Mean area of input geometries (unit cell area).
    source_gdf : gpd.GeoDataFrame
        Original input geometries.
    valid_mask : NDArray[np.bool_] | None
        Boolean mask indicating which rows had valid (non-null) values.
        None if all values were valid.

    """

    positions: NDArray[np.floating]
    sizes: NDArray[np.floating]
    adjacency: NDArray[np.floating]
    bounds: tuple[float, float, float, float]
    mean_area: float
    source_gdf: gpd.GeoDataFrame
    valid_mask: NDArray[np.bool_] | None = None


def compute_symbol_sizes(
    values: ArrayLike,
    scale: Literal["sqrt", "linear", "log"] = "sqrt",
    target_max_size: float | None = None,
    size_max_value: float | None = None,
    size_clip: bool = True,
) -> NDArray[np.floating]:
    """Compute symbol sizes from data values.

    Uses abs(values) for size computation. Sign can be used for styling.

    The returned sizes are **area-equivalent radii**, meaning the symbol
    area equals pi x size^2. This ensures symbols of different shapes with
    the same size have the same visual area.

    Parameters
    ----------
    values : array-like
        Data values to scale.
    scale : str
        "sqrt": Area ∝ value (perceptually accurate, default)
        "linear": Size ∝ value
        "log": Logarithmic scaling via log(1 + x)
    target_max_size : float, optional
        Maximum symbol size (area-equivalent radius). If None, normalized
        to 1.0.
    size_max_value : float, optional
        Reference max value for consistent scaling across cartograms.
        When set, sizes are scaled relative to this value instead of
        the data maximum.
    size_clip : bool
        Whether to clip values exceeding size_max_value. Default True.

    Returns
    -------
    np.ndarray
        Symbol sizes (area-equivalent radii), always non-negative.

    Examples
    --------
    >>> sizes = compute_symbol_sizes([1, 4, 9], scale="sqrt", target_max_size=1.0)
    >>> # sqrt scaling: sqrt([1,4,9]) = [1,2,3], normalized to [0.33, 0.67, 1.0]

    """
    values_arr = np.abs(np.asarray(values, dtype=float))

    # Apply scaling
    if scale == "sqrt":
        scaled = np.sqrt(values_arr)
        ref_max = np.sqrt(size_max_value) if size_max_value is not None else None
    elif scale == "linear":
        scaled = values_arr.copy()
        ref_max = size_max_value
    elif scale == "log":
        scaled = np.log1p(values_arr)
        ref_max = np.log1p(size_max_value) if size_max_value is not None else None
    else:
        raise ValueError(f"Unknown scale: {scale}")

    # Normalize to [0, 1] range
    if ref_max is not None:
        if ref_max > 0:
            normalized = scaled / ref_max
            if size_clip:
                normalized = np.clip(normalized, 0, 1)
        else:
            normalized = np.zeros_like(scaled)
    else:
        max_val = scaled.max()
        normalized = scaled / max_val if max_val > 0 else np.zeros_like(scaled)

    # Scale to target max size
    if target_max_size is not None:
        return normalized * target_max_size
    return normalized


def prepare_layout_data(
    gdf: gpd.GeoDataFrame,
    value_column: str | None = None,
    *,
    size_scale: Literal["sqrt", "linear", "log"] = "sqrt",
    size_max_value: float | None = None,
    size_clip: bool = True,
    adjacency_mode: AdjacencyMode | Literal["binary", "weighted", "area_weighted"] = "binary",
    distance_tolerance: float | None = None,
    size_normalization: Literal["max", "total"] = "max",
) -> LayoutData:
    """Prepare data for layout algorithms."""
    if len(gdf) == 0:
        raise ValueError("GeoDataFrame is empty")

    if value_column is not None and value_column not in gdf.columns:
        raise ValueError(f"Column '{value_column}' not found in GeoDataFrame")
    """Prepare data for layout algorithms.

    This function:

    1. Computes unit cell area as mean geometry area
    2. Extracts centroids as initial positions
    3. Computes symbol sizes (proportional or uniform)
    4. Computes adjacency matrix

    The unit cell is a circle with area equal to the mean geometry area.
    Symbol sizes are area-equivalent radii relative to this unit cell.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with polygon geometries.
    value_column : str, optional
        Column for proportional sizing. If None, uses uniform sizing
        (all symbols same size as unit cell).
    size_scale : str
        Scaling method: "sqrt" (area ∝ value, default), "linear", or "log".
    size_max_value : float, optional
        Reference max value for consistent scaling across cartograms.
    size_clip : bool
        Whether to clip values exceeding size_max_value.
    adjacency_mode : str
        "binary" (adjacent or not), "weighted" (shared border fraction),
        or "area_weighted" (neighbor area fraction).
    distance_tolerance : float, optional
        Buffer for adjacency detection.
    size_normalization : str
        How to normalise symbol sizes relative to original geometry areas:

        - ``"max"`` *(default)*: the largest symbol has area equal to the mean
          geometry area (``π x unit_cell_radius²``).  Relative proportions
          mirror the data values.
        - ``"total"``: all sizes are scaled by a single global factor so that
          ``Σ(π x size²) = Σ(geometry_area)``.  Relative proportions are
          preserved; total symbol area equals total original area.  For
          uniform sizing (no *value_column*) the two modes are identical.

    Returns
    -------
    LayoutData
        Preprocessed data ready for Layout.compute().

    Examples
    --------
    >>> # Prepare data for layout
    >>> data = prepare_layout_data(gdf, "population")
    >>> # Inspect or modify data
    >>> data.sizes *= 0.9  # Scale down all sizes
    >>> # Pass to layout
    >>> result = layout.compute(data)

    """
    n = len(gdf)

    # 1. Compute unit cell area from mean geometry area
    geometry_areas = np.array([g.area for g in gdf.geometry])
    mean_area = float(np.mean(geometry_areas))
    unit_cell_radius = np.sqrt(mean_area / np.pi)

    # 2. Extract centroids
    positions = np.array([[g.centroid.x, g.centroid.y] for g in gdf.geometry])

    # 3. Compute symbol sizes
    valid_mask = np.ones(len(gdf), dtype=bool)
    if value_column is not None:
        # Proportional sizing
        values = gdf[value_column].values

        # Handle null values
        null_mask = pd.isnull(values)
        if np.any(null_mask):
            import warnings

            n_null = np.sum(null_mask)
            warnings.warn(f"Skipping {n_null} rows with null values", UserWarning, stacklevel=2)
            valid_mask = ~null_mask
            # Keep only non-null values
            values = values[valid_mask]
            positions = positions[valid_mask]
            geometry_areas = geometry_areas[valid_mask]
            gdf = gdf[valid_mask]

        sizes = compute_symbol_sizes(
            values,
            scale=size_scale,
            target_max_size=unit_cell_radius,
            size_max_value=size_max_value,
            size_clip=size_clip,
        )
    else:
        # Uniform sizing: all symbols same as unit cell
        sizes = np.full(n, unit_cell_radius)

    # 3b. Apply total-area normalization if requested
    if size_normalization == "total":
        current_total = float(np.pi * np.sum(sizes**2))
        target_total = float(np.sum(geometry_areas))
        if current_total > 0:
            sizes = sizes * float(np.sqrt(target_total / current_total))

    # 4. Compute adjacency
    adjacency = compute_adjacency(
        gdf,
        mode=AdjacencyMode(adjacency_mode),
        distance_tolerance=distance_tolerance,
    )

    return LayoutData(
        positions=positions,
        sizes=sizes,
        adjacency=adjacency,
        bounds=tuple(gdf.total_bounds),
        mean_area=mean_area,
        source_gdf=gdf,
        valid_mask=valid_mask if np.any(~valid_mask) else None,
    )
