"""
Visualization utilities for shape_splitter results.

Plotting utilities for visualizing partitioned geometries from shrink, split,
and partition_geometries operations.

Main Components
---------------
- plot_partitions: Plot partitioned geometries as a choropleth map

Example
-------
>>> from carto_flow.shape_splitter import partition_geometries
>>> from carto_flow.shape_splitter.visualization import plot_partitions
>>>
>>> result = partition_geometries(gdf, ['agriculture', 'industry'], method='split')
>>> plot_partitions(result)
"""

from __future__ import annotations

import colorsys
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = ["plot_partitions"]


def _adjust_color(
    color: str | tuple,
    saturation: float | None = None,
    lightness: float | None = None,
) -> str:
    """
    Adjust saturation and/or lightness of a color.

    Parameters
    ----------
    color : str or tuple
        Color in any matplotlib-accepted format.
    saturation : float, optional
        Saturation multiplier (0-1). 0 = grayscale, 1 = original.
    lightness : float, optional
        Lightness multiplier (0-2). 0 = black, 1 = original, 2 = white.

    Returns
    -------
    str
        Hex color string.
    """
    # Convert to RGB
    rgb = mcolors.to_rgb(color)

    # Convert to HLS (hue, lightness, saturation)
    hue, lum, sat = colorsys.rgb_to_hls(*rgb)

    # Apply adjustments
    if saturation is not None:
        sat = max(0, min(1, sat * saturation))

    if lightness is not None:
        # lightness=0 → black, lightness=1 → original, lightness=2 → white
        lum = lum * lightness if lightness <= 1 else lum + (1 - lum) * (lightness - 1)
        lum = max(0, min(1, lum))

    # Convert back to RGB
    r, g, b = colorsys.hls_to_rgb(hue, lum, sat)
    return mcolors.to_hex((r, g, b))


def _get_colors_from_cmap(
    values: np.ndarray,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> list[str]:
    """Get hex colors for values using a colormap."""
    colormap = plt.get_cmap(cmap)
    if vmin is None:
        vmin = np.nanmin(values)
    if vmax is None:
        vmax = np.nanmax(values)

    # Normalize values
    normalized = (values - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(values)

    # Get colors
    colors = [mcolors.to_hex(colormap(v)) for v in normalized]
    return colors


def plot_partitions(
    gdf: gpd.GeoDataFrame,
    columns: list[str] | None = None,
    # Coloring
    color_by: Literal["area", "category"] | list[str] = "area",
    cmap: str | None = None,
    palette: dict[str, str] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    # Standard styling (for highlighted partitions)
    edgecolor: str = "white",
    linewidth: float = 0.5,
    # Highlighting
    highlight: str | list[str] | None = None,
    background_alpha: float | None = None,
    background_color: str | None = None,
    background_saturation: float | None = None,
    background_lightness: float | None = None,
    background_edgecolor: str | None = None,
    background_linewidth: float | None = None,
    # Complement styling
    include_complement: bool = True,
    complement_color_by: str | None = None,
    complement_color: str = "#e0e0e0",
    complement_alpha: float = 1.0,
    complement_edgecolor: str | None = None,
    complement_linewidth: float | None = None,
    # General
    ax: Axes | None = None,
    legend: bool = True,
    **kwargs: Any,
) -> Axes:
    """
    Plot partitioned geometries as a choropleth map.

    Takes the output of partition_geometries() and plots all partition
    geometries together, with coloring based on area, category, or data values.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Output from partition_geometries() containing geometry columns
        named ``geometry_{column}`` for each input column.
    columns : list[str], optional
        Original column names used in partition_geometries(). If None,
        auto-detects from geometry column names (geometry_* pattern).

    color_by : {'area', 'category'} or list[str], default='area'
        How to color the partitions:

        - **'area'**: Color by partition area (continuous colormap)
        - **'category'**: Color by partition category/column name (categorical)
        - **list[str]**: Data column names for each partition. Length must
          match number of partitions. Single-element list colors all partitions
          by that column.
    cmap : str, optional
        Colormap name for continuous data. Defaults to 'viridis'.
    palette : dict[str, str], optional
        Color mapping for categorical data. Keys are category values,
        values are colors. If not provided, uses ``cmap`` or 'tab10'.
    vmin, vmax : float, optional
        Value range for continuous colormap normalization.

    edgecolor : str, default='white'
        Edge color for partition boundaries.
    linewidth : float, default=0.5
        Line width for partition boundaries.

    highlight : str or list[str], optional
        Partition category/categories to highlight. Other partitions are
        styled according to background_* parameters.
    background_alpha : float, optional
        Alpha (transparency) for non-highlighted partitions.
    background_color : str, optional
        Fixed color for non-highlighted partitions. Overrides data coloring.
    background_saturation : float, optional
        Saturation multiplier (0-1) for non-highlighted partitions.
        0 = grayscale, 1 = original saturation.
    background_lightness : float, optional
        Lightness multiplier for non-highlighted partitions.
        0 = black, 1 = original, 2 = white.
    background_edgecolor : str, optional
        Edge color for non-highlighted partitions. Defaults to edgecolor.
    background_linewidth : float, optional
        Line width for non-highlighted partitions. Defaults to linewidth.

    include_complement : bool, default=True
        Whether to include the complement geometry (remainder) in the plot.
    complement_color_by : str, optional
        Data column to use for complement coloring. If None, uses
        ``complement_color``.
    complement_color : str, default='#e0e0e0'
        Fixed color for complement when ``complement_color_by`` is None.
    complement_alpha : float, default=1.0
        Alpha for complement partitions.
    complement_edgecolor : str, optional
        Edge color for complement. Defaults to ``edgecolor``.
    complement_linewidth : float, optional
        Line width for complement. Defaults to ``linewidth``.

    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    legend : bool, default=True
        Whether to show a legend or colorbar.
    **kwargs
        Additional arguments passed to GeoDataFrame.plot().

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.

    Raises
    ------
    ValueError
        If color_by list length doesn't match partition count, or if
        specified columns don't exist.

    Examples
    --------
    Plot partitions colored by area:

    >>> result = partition_geometries(gdf, ['pop_a', 'pop_b'], method='split')
    >>> plot_partitions(result)

    Plot partitions colored by category:

    >>> plot_partitions(result, color_by='category')

    Color each partition by its corresponding data column:

    >>> plot_partitions(result, color_by=['pop_a', 'pop_b'])

    Highlight specific partition with others faded:

    >>> plot_partitions(result, highlight='pop_a', background_alpha=0.3)

    Highlight with desaturated background:

    >>> plot_partitions(result, highlight='pop_a', background_saturation=0.2)

    Custom complement styling:

    >>> plot_partitions(result, complement_color_by='remainder_value')
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Auto-detect partition columns if not specified
    if columns is None:
        geometry_cols = [col for col in gdf.columns if col.startswith("geometry_") and col != "geometry_complement"]
        columns = [col.replace("geometry_", "") for col in geometry_cols]

    if not columns:
        raise ValueError(
            "No partition geometry columns found. Expected columns named "
            "'geometry_{name}' from partition_geometries() output."
        )

    # Build list of geometry columns (excluding complement)
    geom_columns = [f"geometry_{col}" for col in columns]

    # Validate geometry columns exist
    missing = [col for col in geom_columns if col not in gdf.columns]
    if missing:
        raise ValueError(f"Geometry columns not found: {missing}")

    # Handle color_by as list - validate length
    color_by_columns: list[str] | None = None
    if isinstance(color_by, list):
        if len(color_by) == 1:
            # Single column colors all partitions
            color_by_columns = color_by * len(columns)
        elif len(color_by) == len(columns):
            color_by_columns = color_by
        else:
            raise ValueError(
                f"color_by list length ({len(color_by)}) must match "
                f"partition count ({len(columns)}) or be 1. "
                f"Partitions: {columns}"
            )
        # Validate columns exist
        for col in color_by_columns:
            if col not in gdf.columns:
                raise ValueError(f"Color column '{col}' not found in GeoDataFrame")

    # Normalize highlight to a set
    highlight_set: set[str] | None = None
    if highlight is not None:
        highlight_set = {highlight} if isinstance(highlight, str) else set(highlight)

    # Determine if coloring is categorical
    is_categorical = False
    if color_by == "category":
        is_categorical = True
    elif color_by_columns is not None:
        # Check dtype of color columns
        dtypes = [gdf[col].dtype for col in set(color_by_columns)]
        is_categorical = all(not pd.api.types.is_numeric_dtype(dt) for dt in dtypes)

    # Set default cmap
    if cmap is None:
        cmap = "tab10" if is_categorical else "viridis"

    # Collect all values for consistent color scaling (continuous only)
    all_values: list[float] = []
    if color_by == "area":
        for geom_col in geom_columns:
            for geom in gdf[geom_col]:
                if geom is not None and not geom.is_empty:
                    all_values.append(geom.area)
    elif color_by_columns is not None and not is_categorical:
        for col in set(color_by_columns):
            all_values.extend(gdf[col].dropna().tolist())
        if complement_color_by is not None and complement_color_by in gdf.columns:
            all_values.extend(gdf[complement_color_by].dropna().tolist())

    # Compute vmin/vmax from all values if not specified
    if all_values and not is_categorical:
        if vmin is None:
            vmin = min(all_values)
        if vmax is None:
            vmax = max(all_values)

    # Build records for plotting
    records = []
    for idx, row in gdf.iterrows():
        for i, (col_name, geom_col) in enumerate(zip(columns, geom_columns)):
            geom = row[geom_col]
            if geom is None or geom.is_empty:
                continue

            # Determine color value
            if color_by == "area":
                color_value = geom.area
            elif color_by == "category":
                color_value = col_name
            elif color_by_columns is not None:
                color_value = row[color_by_columns[i]]
            else:
                color_value = None

            records.append({
                "geometry": geom,
                "category": col_name,
                "color_value": color_value,
                "original_index": idx,
                "is_complement": False,
            })

    # Add complement geometries if requested
    if include_complement and "geometry_complement" in gdf.columns:
        for idx, row in gdf.iterrows():
            geom = row["geometry_complement"]
            if geom is None or geom.is_empty:
                continue

            # Determine complement color value
            if complement_color_by is not None and complement_color_by in gdf.columns:
                color_value = row[complement_color_by]
            else:
                color_value = None  # Will use fixed color

            records.append({
                "geometry": geom,
                "category": "complement",
                "color_value": color_value,
                "original_index": idx,
                "is_complement": True,
            })

    if not records:
        ax.set_title("No geometries to plot")
        ax.set_aspect("equal")
        return ax

    # Create unified GeoDataFrame
    plot_gdf = gpd.GeoDataFrame(records, crs=gdf.crs)

    # Assign colors to each geometry
    colors = []
    for _, record in plot_gdf.iterrows():
        if record["is_complement"]:
            # Complement coloring
            if record["color_value"] is not None:
                # Data-driven complement color
                if is_categorical:
                    if palette and record["color_value"] in palette:
                        colors.append(palette[record["color_value"]])
                    else:
                        # Use colormap for categorical
                        cat_values = plot_gdf[~plot_gdf["is_complement"]]["color_value"].unique()
                        cat_cmap = plt.get_cmap(cmap)
                        cat_colors = {
                            v: mcolors.to_hex(cat_cmap(i / max(len(cat_values) - 1, 1)))
                            for i, v in enumerate(cat_values)
                        }
                        colors.append(cat_colors.get(record["color_value"], complement_color))
                else:
                    # Continuous
                    c = _get_colors_from_cmap(np.array([record["color_value"]]), cmap, vmin, vmax)[0]
                    colors.append(c)
            else:
                colors.append(complement_color)
        else:
            # Regular partition coloring
            if is_categorical:
                if palette and record["color_value"] in palette:
                    colors.append(palette[record["color_value"]])
                else:
                    cat_values = plot_gdf[~plot_gdf["is_complement"]]["color_value"].unique()
                    cat_cmap = plt.get_cmap(cmap)
                    cat_colors = {
                        v: mcolors.to_hex(cat_cmap(i / max(len(cat_values) - 1, 1))) for i, v in enumerate(cat_values)
                    }
                    colors.append(cat_colors.get(record["color_value"], "#888888"))
            else:
                c = _get_colors_from_cmap(np.array([record["color_value"]]), cmap, vmin, vmax)[0]
                colors.append(c)

    plot_gdf["_color"] = colors

    # Apply highlight styling
    alphas = []
    edge_colors = []
    line_widths = []

    for _, record in plot_gdf.iterrows():
        cat = record["category"]
        is_comp = record["is_complement"]
        base_color = record["_color"]

        if is_comp:
            # Complement styling
            alphas.append(complement_alpha)
            edge_colors.append(complement_edgecolor or edgecolor)
            line_widths.append(complement_linewidth or linewidth)
        elif highlight_set is not None and cat not in highlight_set:
            # Background (non-highlighted) styling
            # Apply color adjustments
            if background_color is not None:
                plot_gdf.at[_, "_color"] = background_color
            elif background_saturation is not None or background_lightness is not None:
                adjusted = _adjust_color(
                    base_color,
                    saturation=background_saturation,
                    lightness=background_lightness,
                )
                plot_gdf.at[_, "_color"] = adjusted

            alphas.append(background_alpha if background_alpha is not None else 1.0)
            edge_colors.append(background_edgecolor or edgecolor)
            line_widths.append(background_linewidth or linewidth)
        else:
            # Highlighted or no highlight specified
            alphas.append(1.0)
            edge_colors.append(edgecolor)
            line_widths.append(linewidth)

    plot_gdf["_alpha"] = alphas
    plot_gdf["_edgecolor"] = edge_colors
    plot_gdf["_linewidth"] = line_widths

    # Plot each geometry individually to support per-geometry styling
    for _, record in plot_gdf.iterrows():
        gpd.GeoDataFrame([{"geometry": record["geometry"]}], crs=plot_gdf.crs).plot(
            ax=ax,
            color=record["_color"],
            alpha=record["_alpha"],
            edgecolor=record["_edgecolor"],
            linewidth=record["_linewidth"],
            **kwargs,
        )

    ax.set_aspect("equal")

    # Add legend/colorbar
    if legend:
        if is_categorical:
            # Categorical legend
            cat_values = plot_gdf[~plot_gdf["is_complement"]]["color_value"].unique()
            cat_cmap = plt.get_cmap(cmap)
            handles = []
            for i, val in enumerate(cat_values):
                if palette and val in palette:
                    c = palette[val]
                else:
                    c = mcolors.to_hex(cat_cmap(i / max(len(cat_values) - 1, 1)))
                handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor=edgecolor, label=str(val)))
            if handles:
                ax.legend(handles=handles, loc="best")
        else:
            # Continuous colorbar
            if vmin is not None and vmax is not None:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
                sm.set_array([])
                plt.colorbar(sm, ax=ax)

    # Set title
    if highlight_set is not None:
        highlight_str = ", ".join(sorted(highlight_set))
        ax.set_title(f"Highlighted: {highlight_str}")
    elif color_by == "area":
        ax.set_title("Partitions by Area")
    elif color_by == "category":
        ax.set_title("Partitions by Category")
    elif color_by_columns is not None:
        ax.set_title(f"Partitions by {', '.join(set(color_by_columns))}")

    return ax
