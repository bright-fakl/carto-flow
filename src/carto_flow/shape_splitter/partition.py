"""
Batch geometry partitioning for GeoDataFrames.

This module provides the partition_geometries function for batch processing
of geometries in a GeoDataFrame using either shrinking or splitting methods.
Supports parallelization and progress bars.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import geopandas as gpd
import numpy as np
import pandas as pd

from .shrink import shrink
from .split import split

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

__all__ = ["partition_geometries"]


def _process_single_geometry(
    geom: BaseGeometry,
    full_fractions: list[float],
    method: Literal["shrink", "split"],
    n_columns: int,
    has_remainder: bool,
    # shrink params
    simplify: float | None,
    mode: Literal["area", "shell"],
    tol: float,
    # split params
    direction: Literal["vertical", "horizontal"],
    alternate: bool,
    strategy: Literal["sequential", "treemap"],
) -> tuple[list[BaseGeometry], BaseGeometry | None]:
    """
    Process a single geometry with shrink or split method.

    Returns tuple of (column_parts, complement_part).
    complement_part is None if has_remainder is False.
    """

    def _make_empty():
        return geom.buffer(1e-10).buffer(-1e10)

    try:
        if method == "shrink":
            parts = shrink(
                geom,
                full_fractions,
                simplify=simplify,
                mode=mode,
                tol=tol,
            )
        else:  # method == "split"
            parts = split(
                geom,
                full_fractions,
                direction=direction,
                alternate=alternate,
                strategy=strategy,
                tol=tol,
            )
    except Exception:
        # Fallback: first column gets original, others get empty
        column_parts = [geom, *[_make_empty() for _ in range(n_columns - 1)]]
        complement_part = _make_empty() if has_remainder else None
        return column_parts, complement_part
    else:
        column_parts = parts[:n_columns]
        complement_part = parts[-1] if has_remainder else None
        return column_parts, complement_part


def partition_geometries(
    gdf: pd.DataFrame | gpd.GeoDataFrame,
    columns: str | Sequence[str],
    method: Literal["shrink", "split"] = "shrink",
    normalization: Literal["sum", "maximum", "row", None] = None,
    simplify: float | None = None,
    mode: Literal["area", "shell"] = "area",
    tol: float = 0.05,
    direction: Literal["vertical", "horizontal"] = "vertical",
    alternate: bool = True,
    strategy: Literal["sequential", "treemap"] = "sequential",
    invert: bool = False,
    copy: bool = True,
    n_jobs: int = 1,
    progress: bool = False,
) -> gpd.GeoDataFrame:
    """
    Process geometries in a GeoDataFrame using either shrinking or splitting methods.

    This function applies geometric operations to all geometries in a GeoDataFrame
    based on values in one or more specified columns. Supports both single-fraction
    operations (one column) and multi-fraction operations (multiple columns for
    N-way splits/shrinks).

    Parameters
    ----------
    gdf : Union[pd.DataFrame, gpd.GeoDataFrame]
        Input DataFrame containing geometries.
    columns : str or Sequence[str]
        Column name(s) containing values to use for area scaling/splitting.

        - **Single string**: One fraction per geometry. Output includes the
          processed geometry and its complement.
        - **List of strings**: Multiple fractions per geometry (N-way split/shrink).
          Each column provides one fraction. Output includes N geometry columns
          (one per input column) plus a complement if fractions sum < 1.0.
    method : {'shrink', 'split'}, default='shrink'
        Processing method to apply:

        - **'shrink'**: Creates concentric shells from outside to inside.
          For N fractions, produces N parts (N-1 shells + 1 core).
        - **'split'**: Divides geometries into parts with specified area ratios.
          For N fractions, produces N parts.
    normalization : {'sum', 'maximum', 'row', None}, default=None
        Normalization method for computing fractions:

        - **'sum'**: Normalize by sum of all row sums. Each value becomes
          value / (sum of all values across all columns and rows).
          All geometries will have remainders.
        - **'maximum'**: Normalize by maximum row sum. The geometry with the
          largest total gets no remainder; others are scaled proportionally.
        - **'row'**: Normalize each row independently to sum to 1.0. No remainders.
          Only valid for multiple columns.
        - **None**: Use values directly as fractions. Values should be in [0, 1]
          range and row sums must not exceed 1.0.
    simplify : float, optional
        Simplification tolerance (Douglas-Peucker). Only used with 'shrink' method.
    mode : {'area', 'shell'}, default='area'
        Interpretation mode for fractions (only used with 'shrink' method):

        - **'area'**: Fractions represent direct area ratios
        - **'shell'**: Fractions represent shell thickness ratios (squared for area)
    tol : float, default=0.05
        Tolerance for root finding (shrink) or area matching (split).
    direction : {'vertical', 'horizontal'}, default='vertical'
        Initial direction for splitting (only used with 'split' method).
    alternate : bool, default=True
        Whether to alternate direction for sequential splits (only used with
        'split' method and strategy='sequential').
    strategy : {'sequential', 'treemap'}, default='sequential'
        Splitting strategy for N-way splits (only used with 'split' method):

        - **'sequential'**: Parts are carved off one-by-one from edges.
        - **'treemap'**: Recursive binary partitioning for grid-like patterns.
    invert : bool, default=False
        Whether to invert computed fractions (1 - fraction).
    copy : bool, default=True
        Whether to return a copy of the input DataFrame.
    n_jobs : int, default=1
        Number of parallel jobs for processing geometries.

        - **1**: Sequential processing (no parallelization)
        - **-1**: Use all available CPU cores
        - **n > 1**: Use n parallel workers

        Parallelization is beneficial for large datasets (>100 geometries).
        Requires joblib package for n_jobs != 1.
    progress : bool, default=False
        Whether to display a progress bar during processing.
        Requires tqdm package when enabled.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with processed geometries. Output columns depend on input:

        - **Single column 'col'**: ``geometry_col``, ``geometry_complement``
        - **Multiple columns ['a', 'b', 'c']**: ``geometry_a``, ``geometry_b``,
          ``geometry_c``, and ``geometry_complement`` if any row has remainder.

        For shrink method with multiple columns, parts are ordered from outermost
        shell to innermost core. For split method, parts correspond to each fraction.

    Raises
    ------
    ValueError
        If columns not found, contain invalid values, or row sums exceed 1.0
        when normalization is None.
    TypeError
        If input is not a pandas DataFrame or lacks geometry column.

    Notes
    -----
    **Normalization methods for multiple columns**:

    - **'sum'**: value / (sum of all row sums)
      - Preserves relative proportions both within and across geometries
      - Geometries with larger totals get more filled area
      - All geometries have remainders

    - **'maximum'**: value / (max row sum)
      - Row with largest total fills completely (no remainder)
      - Other rows scaled proportionally with remainders

    - **'row'**: value / (that row's sum)
      - Each geometry's fractions sum to 1.0, no remainders
      - Only considers relative proportions within each geometry

    Examples
    --------
    Single column shrinking:

    >>> import geopandas as gpd
    >>> from carto_flow.shape_splitter import partition_geometries
    >>>
    >>> result = partition_geometries(gdf, 'population', normalization='sum')
    >>> # Output columns: geometry_population, geometry_complement

    Multi-column N-way splitting:

    >>> # Split each geometry into 3 parts based on sector values
    >>> result = partition_geometries(
    ...     gdf,
    ...     columns=['agriculture', 'industry', 'services'],
    ...     method='split',
    ...     normalization='row'
    ... )
    >>> # Output columns: geometry_agriculture, geometry_industry, geometry_services

    Multi-column shrinking with remainder:

    >>> # Shrink based on category values, keeping complement
    >>> result = partition_geometries(
    ...     gdf,
    ...     columns=['cat_a', 'cat_b'],
    ...     method='shrink',
    ...     normalization='maximum'
    ... )
    >>> # Output: geometry_cat_a (outer shell), geometry_cat_b (core),
    >>> #         geometry_complement (if any row sum < 1.0)

    Parallel processing with progress bar:

    >>> result = partition_geometries(
    ...     gdf,
    ...     columns='population',
    ...     n_jobs=-1,      # Use all CPU cores
    ...     progress=True   # Show progress bar
    ... )
    """
    # --- Input validation ---
    if method not in ["shrink", "split"]:
        raise ValueError(f"Invalid method '{method}'. Must be 'shrink' or 'split'.")

    if not isinstance(gdf, (pd.DataFrame, gpd.GeoDataFrame)):
        raise TypeError(f"Input must be a pandas DataFrame, got {type(gdf)}")

    # Ensure we have a GeoDataFrame
    if not isinstance(gdf, gpd.GeoDataFrame):
        if "geometry" not in gdf.columns:
            raise ValueError("DataFrame must contain a 'geometry' column with Shapely geometries")
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry")

    # Normalize columns parameter to list
    column_list = [columns] if isinstance(columns, str) else list(columns)

    if not column_list:
        raise ValueError("columns cannot be empty")

    is_multi_column = len(column_list) > 1

    # Validate columns exist and contain numeric data
    for col in column_list:
        if col not in gdf.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame columns: {list(gdf.columns)}")
        if not pd.api.types.is_numeric_dtype(gdf[col]):
            raise ValueError(f"Column '{col}' must contain numeric values")
        if (gdf[col] < 0).any():
            raise ValueError(f"Column '{col}' contains negative values. All values must be non-negative.")

    # Validate normalization parameter
    valid_normalizations = {"sum", "maximum", "row", None}
    if normalization not in valid_normalizations:
        raise ValueError(f"Invalid normalization '{normalization}'. Must be one of: {valid_normalizations}")

    if normalization == "row" and not is_multi_column:
        raise ValueError(
            "normalization='row' is only valid for multiple columns. For single column, use 'sum', 'maximum', or None."
        )

    # Create copy if requested
    result_gdf = gdf.copy() if copy else gdf

    # --- Extract values and compute fractions ---
    # Get values as 2D array: shape (n_geometries, n_columns)
    values = gdf[column_list].values.astype(float)

    # Compute row sums for normalization
    row_sums = values.sum(axis=1)

    # Check for all-zero rows
    zero_rows = row_sums == 0
    if zero_rows.all():
        raise ValueError("All rows have zero values across specified columns. Cannot compute meaningful fractions.")

    # Compute fractions based on normalization mode
    if normalization == "sum":
        # Normalize by sum of all row sums
        total_sum = row_sums.sum()
        if total_sum == 0:
            raise ValueError("Sum of all values is zero. Cannot normalize.")
        fractions = values / total_sum

    elif normalization == "maximum":
        # Normalize by maximum row sum
        max_row_sum = row_sums.max()
        if max_row_sum == 0:
            raise ValueError("Maximum row sum is zero. Cannot normalize.")
        fractions = values / max_row_sum

    elif normalization == "row":
        # Normalize each row to sum to 1.0
        # Handle zero rows by setting fractions to 0 (will produce empty geometries)
        fractions = np.zeros_like(values)
        nonzero_mask = row_sums > 0
        fractions[nonzero_mask] = values[nonzero_mask] / row_sums[nonzero_mask, np.newaxis]

    else:  # normalization is None
        # Use values directly as fractions
        fractions = values.copy()

    # Apply inversion if requested
    if invert:
        fractions = 1.0 - fractions
        fractions = np.clip(fractions, 0.0, 1.0)

    # --- Validate fractions ---
    # Check row sums don't exceed 1.0
    frac_row_sums = fractions.sum(axis=1)
    exceeds_one = frac_row_sums > 1.0 + 1e-9  # Small tolerance for floating point

    if exceeds_one.any():
        bad_indices = np.where(exceeds_one)[0]
        bad_sums = frac_row_sums[exceeds_one]
        raise ValueError(
            f"Row fractions exceed 1.0 at indices {bad_indices.tolist()} "
            f"(sums: {[round(s, 4) for s in bad_sums.tolist()]}). "
            f"Use normalization='row' to auto-normalize, "
            f"or ensure input values sum to <= 1.0 per row."
        )

    # Determine if we need a complement column
    # (any row has fractions summing to less than 1.0 - tolerance)
    has_remainder = (frac_row_sums < 1.0 - 1e-9).any()

    # Compute remainder fractions
    remainder_fractions = 1.0 - frac_row_sums
    remainder_fractions = np.clip(remainder_fractions, 0.0, 1.0)

    # --- Process geometries ---
    n_geometries = len(result_gdf)
    n_columns = len(column_list)

    # Prepare inputs for processing
    geometries = list(result_gdf.geometry)
    full_fractions_list = []
    for idx in range(n_geometries):
        row_fractions = fractions[idx]
        row_remainder = remainder_fractions[idx]
        if has_remainder:
            full_fractions_list.append([*row_fractions, row_remainder])
        else:
            full_fractions_list.append(list(row_fractions))

    # Define processing function for a single geometry
    def process_one(geom, full_fracs):
        return _process_single_geometry(
            geom=geom,
            full_fractions=full_fracs,
            method=method,
            n_columns=n_columns,
            has_remainder=has_remainder,
            simplify=simplify,
            mode=mode,
            tol=tol,
            direction=direction,
            alternate=alternate,
            strategy=strategy,
        )

    # Process geometries (parallel or sequential)
    if n_jobs == 1:
        # Sequential processing
        if progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(
                    zip(geometries, full_fractions_list),
                    total=n_geometries,
                    desc="Partitioning geometries",
                )
            except ImportError:
                warnings.warn(
                    "tqdm not installed. Install with 'pip install tqdm' for progress bars.",
                    UserWarning,
                    stacklevel=2,
                )
                iterator = zip(geometries, full_fractions_list)
        else:
            iterator = zip(geometries, full_fractions_list)

        results = [process_one(geom, fracs) for geom, fracs in iterator]
    else:
        # Parallel processing with joblib
        try:
            from joblib import Parallel, delayed
        except ImportError as e:
            raise ImportError("joblib is required for parallel processing. Install with 'pip install joblib'.") from e

        if progress:
            try:
                from tqdm.auto import tqdm

                results = Parallel(n_jobs=n_jobs)(
                    delayed(process_one)(geom, fracs)
                    for geom, fracs in tqdm(
                        zip(geometries, full_fractions_list),
                        total=n_geometries,
                        desc="Partitioning geometries",
                    )
                )
            except ImportError:
                warnings.warn(
                    "tqdm not installed. Install with 'pip install tqdm' for progress bars.",
                    UserWarning,
                    stacklevel=2,
                )
                results = Parallel(n_jobs=n_jobs)(
                    delayed(process_one)(geom, fracs) for geom, fracs in zip(geometries, full_fractions_list)
                )
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_one)(geom, fracs) for geom, fracs in zip(geometries, full_fractions_list)
            )

    # Unpack results into output structure
    output_geoms: dict[str, list] = {col: [] for col in column_list}
    complement_geoms: list = []

    for column_parts, complement_part in results:
        for i, col in enumerate(column_list):
            output_geoms[col].append(column_parts[i])
        if has_remainder:
            complement_geoms.append(complement_part)

    # --- Build output DataFrame ---
    # First geometry column becomes the active geometry
    first_col = column_list[0]
    result_gdf = result_gdf.set_geometry(gpd.GeoSeries(output_geoms[first_col], crs=result_gdf.crs))
    result_gdf = result_gdf.rename_geometry(f"geometry_{first_col}")

    # Add remaining geometry columns
    for col in column_list[1:]:
        result_gdf[f"geometry_{col}"] = gpd.GeoSeries(output_geoms[col], crs=result_gdf.crs)

    # Add complement column if needed
    if has_remainder:
        result_gdf["geometry_complement"] = gpd.GeoSeries(complement_geoms, crs=result_gdf.crs)

    return result_gdf
