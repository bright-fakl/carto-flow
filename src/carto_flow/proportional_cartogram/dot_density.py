"""
Dot density map generation for GeoDataFrames.

Provides functions to generate randomly positioned markers inside geometries
proportional to column values, and to visualize them as dot density maps.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import Collection, PathCollection
    from matplotlib.legend import Legend
    from shapely.geometry.base import BaseGeometry

    from .plot_results import DotDensityPlotResult

__all__ = ["generate_dot_density", "plot_dot_density"]


def _sample_points_in_geometry(
    geom: BaseGeometry,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample n random points uniformly inside a geometry using rejection sampling.

    Parameters
    ----------
    geom : BaseGeometry
        Shapely geometry to sample from.
    n : int
        Number of points to generate.
    rng : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n, 2) with (x, y) coordinates. Returns empty array
        if n == 0 or geometry is empty.
    """
    if n <= 0 or geom is None or geom.is_empty:
        return np.empty((0, 2))

    minx, miny, maxx, maxy = geom.bounds
    w, h = maxx - minx, maxy - miny
    if w == 0 or h == 0:
        return np.empty((0, 2))

    # Use shapely.contains_xy for vectorized point-in-polygon (shapely >= 2.0)
    try:
        from shapely import contains_xy as _contains_xy

        def _inside(xs, ys):
            return _contains_xy(geom, xs, ys)

    except (ImportError, AttributeError):
        from shapely.geometry import Point

        def _inside(xs, ys):
            return np.array([geom.contains(Point(x, y)) for x, y in zip(xs, ys)])

    collected: list[np.ndarray] = []
    collected_count = 0
    batch_size = max(n * 4, 100)

    while collected_count < n:
        xs = rng.uniform(minx, maxx, batch_size)
        ys = rng.uniform(miny, maxy, batch_size)
        mask = _inside(xs, ys)
        pts = np.column_stack([xs[mask], ys[mask]])
        if len(pts) > 0:
            collected.append(pts)
            collected_count += len(pts)

    return np.vstack(collected)[:n]


def generate_dot_density(
    gdf: gpd.GeoDataFrame,
    columns: str | Sequence[str],
    n_dots: int = 100,
    normalization: Literal["sum", "maximum", "row", None] = None,
    invert: bool = False,
    seed: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Generate randomly positioned dots inside geometries to represent variable magnitudes.

    For each geometry, dots are sampled once (one rejection-sampling pass) and then
    assigned to categories by shuffling and slicing. This means dots from different
    categories do not overlap and the full sampling cost is paid once per geometry.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with polygon geometries.
    columns : str or Sequence[str]
        Column name(s) containing values to represent as dots.

        - **Single string**: Single-variable density plot. The geometry with the
          largest value gets ``n_dots`` markers; others are scaled proportionally.
          Use ``normalization='maximum'`` (or leave ``None``, which applies
          ``'maximum'`` automatically for a single column).
        - **List of strings**: Multi-fraction dot density map. Each column
          contributes dots of a distinct category, colored separately.
    n_dots : int, default=100
        Number of dots for a fully-filled geometry (fraction = 1.0). Each
        geometry gets ``round(fraction * n_dots)`` dots per category. For a
        single column with ``normalization='maximum'``, the geometry with the
        maximum value receives ``n_dots`` dots.
    normalization : {'sum', 'maximum', 'row', None}, default=None
        How to convert column values to fractions:

        - **'sum'**: Divide each value by the sum of all values across all columns
          and rows.
        - **'maximum'**: Divide by the maximum row sum; the geometry with the
          largest total gets ``n_dots`` dots in total.
        - **'row'**: Normalise each row to sum to 1.0 (only valid for multiple
          columns). Every geometry gets exactly ``n_dots`` dots in total.
        - **None**: Use values directly as fractions. Must be in [0, 1] and row
          sums must not exceed 1.0.

        For a single column, ``None`` behaves the same as ``'maximum'``.
    invert : bool, default=False
        Whether to invert computed fractions (1 - fraction).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame of Point geometries with columns:

        - ``geometry`` : Point
        - ``category`` : str — the column name this dot represents
        - ``original_index`` : the index value of the source row in ``gdf``
        - ``value`` : float — the raw value from the source column

    Raises
    ------
    ValueError
        If columns are not found, contain invalid values, or row sums exceed 1.0
        when ``normalization`` is None.
    TypeError
        If ``gdf`` is not a GeoDataFrame.

    Examples
    --------
    Single-variable density:

    >>> dots = generate_dot_density(gdf, "population", n_dots=100,
    ...                             normalization="maximum", seed=0)
    >>> dots.plot(markersize=3)

    Multi-fraction dot density:

    >>> dots = generate_dot_density(
    ...     gdf,
    ...     columns=["agriculture", "industry", "services"],
    ...     n_dots=200,
    ...     normalization="row",
    ...     seed=42,
    ... )
    >>> dots.groupby("category").size()
    """
    # --- Input validation ---
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(f"gdf must be a GeoDataFrame, got {type(gdf)}")

    column_list = [columns] if isinstance(columns, str) else list(columns)
    if not column_list:
        raise ValueError("columns cannot be empty")

    is_multi_column = len(column_list) > 1

    for col in column_list:
        if col not in gdf.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame columns: {list(gdf.columns)}")
        if not pd.api.types.is_numeric_dtype(gdf[col]):
            raise ValueError(f"Column '{col}' must contain numeric values")
        if (gdf[col] < 0).any():
            raise ValueError(f"Column '{col}' contains negative values. All values must be non-negative.")

    valid_normalizations = {"sum", "maximum", "row", None}
    if normalization not in valid_normalizations:
        raise ValueError(f"Invalid normalization '{normalization}'. Must be one of: {valid_normalizations}")

    if normalization == "row" and not is_multi_column:
        raise ValueError(
            "normalization='row' is only valid for multiple columns. "
            "For a single column, use 'sum', 'maximum', or None."
        )

    # --- Compute fractions ---
    values = gdf[column_list].values.astype(float)
    row_sums = values.sum(axis=1)

    if row_sums.sum() == 0:
        raise ValueError("All values are zero. Cannot compute meaningful dot counts.")

    # For a single column with normalization=None, treat as 'maximum'
    effective_normalization = normalization
    if not is_multi_column and normalization is None:
        effective_normalization = "maximum"

    if effective_normalization == "sum":
        total_sum = row_sums.sum()
        fractions = values / total_sum

    elif effective_normalization == "maximum":
        max_row_sum = row_sums.max()
        if max_row_sum == 0:
            raise ValueError("Maximum row sum is zero. Cannot normalise.")
        fractions = values / max_row_sum

    elif effective_normalization == "row":
        fractions = np.zeros_like(values)
        nonzero_mask = row_sums > 0
        fractions[nonzero_mask] = values[nonzero_mask] / row_sums[nonzero_mask, np.newaxis]

    else:  # normalization is None (multi-column: use values directly)
        fractions = values.copy()

    if invert:
        fractions = np.clip(1.0 - fractions, 0.0, 1.0)

    # Validate fractions
    frac_row_sums = fractions.sum(axis=1)
    exceeds_one = frac_row_sums > 1.0 + 1e-9
    if exceeds_one.any():
        bad = np.where(exceeds_one)[0]
        raise ValueError(
            f"Row fractions exceed 1.0 at indices {bad.tolist()}. "
            f"Use normalization='row' to auto-normalise, or ensure values sum to <= 1.0 per row."
        )

    # --- Sample dots ---
    rng = np.random.default_rng(seed)

    all_records: list[dict] = []
    geometries_list = list(gdf.geometry)
    indices = list(gdf.index)

    for i, (geom, orig_idx) in enumerate(zip(geometries_list, indices)):
        row_fracs = fractions[i]
        row_vals = values[i]

        # Integer dot counts per category
        counts = np.array([round(float(f) * n_dots) for f in row_fracs])
        total = int(counts.sum())

        if total == 0 or geom is None or geom.is_empty:
            continue

        # Single rejection-sampling pass for all dots in this geometry
        pts = _sample_points_in_geometry(geom, total, rng)
        if len(pts) == 0:
            continue

        # Shuffle and assign to categories by slicing
        idx_perm = rng.permutation(len(pts))
        pts = pts[idx_perm]

        start = 0
        for j, (col_name, count) in enumerate(zip(column_list, counts)):
            count = int(count)
            if count == 0:
                start += count
                continue
            end = start + count
            chunk = pts[start:end]
            start = end
            for x, y in chunk:
                from shapely.geometry import Point

                all_records.append({
                    "geometry": Point(x, y),
                    "category": col_name,
                    "original_index": orig_idx,
                    "value": float(row_vals[j]),
                })

    if not all_records:
        return gpd.GeoDataFrame(
            columns=["geometry", "category", "original_index", "value"],
            crs=gdf.crs,
        )

    return gpd.GeoDataFrame(all_records, crs=gdf.crs)


def plot_dot_density(
    gdf: gpd.GeoDataFrame,
    dots_gdf: gpd.GeoDataFrame | None = None,
    columns: str | Sequence[str] | None = None,
    # Dot generation params (only used when dots_gdf is None)
    n_dots: int = 100,
    normalization: Literal["sum", "maximum", "row", None] = None,
    invert: bool = False,
    seed: int | None = None,
    # Dot styling
    palette: dict[str, str] | None = None,
    cmap: str | None = None,
    size: float = 4,
    alpha: float | Sequence[float] = 0.7,
    marker: str | Sequence[str] = "o",
    # Base geometry styling
    base_color: str = "#e0e0e0",
    base_edgecolor: str = "white",
    base_linewidth: float = 0.5,
    base_alpha: float = 1.0,
    # General
    ax: Axes | None = None,
    legend: bool = True,
    **kwargs: Any,
) -> DotDensityPlotResult:
    """
    Plot a dot density map with base geometries as background.

    Dots are placed randomly inside each geometry, with the count per category
    proportional to the corresponding column value. Each category is rendered
    in a distinct color.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with polygon geometries. Used for the background
        and, when ``dots_gdf`` is None, for generating dots.
    dots_gdf : gpd.GeoDataFrame, optional
        Pre-generated dots from :func:`generate_dot_density`. If provided,
        ``columns``, ``n_dots``, ``normalization``, ``invert``, and ``seed``
        are ignored.
    columns : str or Sequence[str], optional
        Column name(s) for dot generation. Required when ``dots_gdf`` is None.
    n_dots : int, default=100
        Passed to :func:`generate_dot_density` when ``dots_gdf`` is None.
    normalization : {'sum', 'maximum', 'row', None}, default=None
        Passed to :func:`generate_dot_density` when ``dots_gdf`` is None.
    invert : bool, default=False
        Passed to :func:`generate_dot_density` when ``dots_gdf`` is None.
    seed : int, optional
        Passed to :func:`generate_dot_density` when ``dots_gdf`` is None.
    palette : dict[str, str], optional
        Mapping from category names to colors. If not provided, colors are
        drawn from ``cmap``.
    cmap : str, optional
        Colormap name for categorical colors. Defaults to ``'tab10'``.
    size : float, default=4
        Marker size in points.
    alpha : float or Sequence[float], default=0.7
        Marker transparency. A single float applies to all categories. A
        sequence of floats sets transparency per category, in the order
        categories appear in ``dots_gdf["category"].unique()``. Length must
        match the number of categories.
    marker : str or Sequence[str], default='o'
        Matplotlib marker style. A single string applies to all categories.
        A sequence sets the marker per category (same ordering and length
        rules as ``alpha``). Any marker accepted by ``ax.scatter()`` works,
        e.g. ``'o'``, ``'s'``, ``'^'``, ``'*'``.
    base_color : str, default='#e0e0e0'
        Fill color for the background geometries.
    base_edgecolor : str, default='white'
        Edge color for background geometries.
    base_linewidth : float, default=0.5
        Edge line width for background geometries.
    base_alpha : float, default=1.0
        Transparency for background geometries.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    legend : bool, default=True
        Whether to add a legend for dot categories.
    **kwargs
        Additional keyword arguments passed to ``ax.scatter()``.

    Returns
    -------
    DotDensityPlotResult
        Result with axes, scatter collections, base polygon collection, and legend.

    Examples
    --------
    Quick single-variable density plot:

    >>> result = plot_dot_density(gdf, columns="population",
    ...                           n_dots=100, normalization="maximum")

    Multi-fraction with custom colors:

    >>> result = plot_dot_density(
    ...     gdf,
    ...     columns=["agriculture", "industry", "services"],
    ...     normalization="row",
    ...     palette={"agriculture": "#4daf4a", "industry": "#377eb8", "services": "#e41a1c"},
    ...     n_dots=150,
    ... )

    Reuse pre-generated dots:

    >>> dots = generate_dot_density(gdf, "population", n_dots=100, seed=0)
    >>> result = plot_dot_density(gdf, dots_gdf=dots)
    """
    from .plot_results import DotDensityPlotResult

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Generate dots if not provided
    if dots_gdf is None:
        if columns is None:
            raise ValueError("Either dots_gdf or columns must be provided.")
        dots_gdf = generate_dot_density(
            gdf,
            columns=columns,
            n_dots=n_dots,
            normalization=normalization,
            invert=invert,
            seed=seed,
        )

    # Plot base geometries
    base_collection: Collection | None = None
    before = len(ax.collections)
    gdf.plot(
        ax=ax,
        color=base_color,
        edgecolor=base_edgecolor,
        linewidth=base_linewidth,
        alpha=base_alpha,
    )
    new_colls = list(ax.collections[before:])
    if new_colls:
        base_collection = new_colls[0]

    # Determine categories and colors
    if len(dots_gdf) == 0:
        ax.set_aspect("equal")
        return DotDensityPlotResult(
            ax=ax,
            dot_collections=[],
            base_collection=base_collection,
            legend=None,
        )

    categories = list(dots_gdf["category"].unique())
    if cmap is None:
        cmap = "tab10"
    colormap = plt.get_cmap(cmap)
    category_colors: dict[str, str] = {}
    for i, cat in enumerate(categories):
        if palette and cat in palette:
            category_colors[cat] = palette[cat]
        else:
            category_colors[cat] = mcolors.to_hex(
                colormap(i % colormap.N if hasattr(colormap, "N") else i / max(len(categories) - 1, 1))
            )

    # Resolve alpha and marker per category
    if isinstance(alpha, (int, float)):
        alpha_vals = [float(alpha)] * len(categories)
    else:
        alpha_vals = list(alpha)
        if len(alpha_vals) != len(categories):
            raise ValueError(
                f"alpha has {len(alpha_vals)} values but there are {len(categories)} "
                f"categories: {categories}. Provide a single float or one value per category."
            )

    if isinstance(marker, str):
        marker_vals = [marker] * len(categories)
    else:
        marker_vals = list(marker)
        if len(marker_vals) != len(categories):
            raise ValueError(
                f"marker has {len(marker_vals)} values but there are {len(categories)} "
                f"categories: {categories}. Provide a single string or one value per category."
            )

    # Plot dots per category
    dot_collections: list[PathCollection] = []
    for i, cat in enumerate(categories):
        subset = dots_gdf[dots_gdf["category"] == cat]
        xs = subset.geometry.x.values
        ys = subset.geometry.y.values
        sc = ax.scatter(
            xs,
            ys,
            c=category_colors[cat],
            s=size,
            alpha=alpha_vals[i],
            marker=marker_vals[i],
            label=cat,
            **kwargs,
        )
        dot_collections.append(sc)

    ax.set_aspect("equal")

    # Legend
    legend_art: Legend | None = None
    if legend and categories:
        legend_art = ax.legend(loc="best", title="Category", markerscale=2)

    return DotDensityPlotResult(
        ax=ax,
        dot_collections=dot_collections,
        base_collection=base_collection,
        legend=legend_art,
    )
