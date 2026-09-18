"""Symbol cartogram creation function and helpers.

This module contains the ``create_symbol_cartogram`` entry point and its
private helpers. The ``__init__`` module re-exports the public function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import geopandas as gpd
import numpy as np

from .options import AdjacencyMode
from .result import SymbolCartogram

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .layouts import Layout, LayoutResult
    from .styling import Styling


def create_symbol_cartogram(
    gdf: gpd.GeoDataFrame,
    size: str | None = None,
    *,
    tile_count: str | None = None,
    group_by: str | None = None,
    # Layout algorithm
    layout: Layout | str = "physics",
    # Preprocessing options (passed to create_layout)
    size_scale: Literal["sqrt", "linear", "log"] = "sqrt",
    size_max_value: float | None = None,
    size_clip: bool = True,
    size_normalization: Literal["max", "total"] = "max",
    tile_size_expansion: Literal["shared", "copied"] = "copied",
    collapse_group: float = 0.0,
    adjacency_mode: AdjacencyMode = AdjacencyMode.BINARY,
    adjacency: NDArray | None = None,
    distance_tolerance: float | None = None,
    pre_scale: bool = False,
    # Runtime options (passed to create_layout)
    show_progress: bool = True,
    save_history: bool = False,
    # Styling
    styling: Styling | None = None,
) -> SymbolCartogram:
    """Create a symbol cartogram from a GeoDataFrame.

    Each region is represented by a single symbol (circle, square, or hexagon).
    Symbol size is proportional to a data value when ``size`` is
    provided, or uniform when it is omitted.

    This is a convenience function that combines layout computation and styling
    in a single call. For more control, use ``create_layout`` and then apply
    styling via ``LayoutResult.style()`` or ``Styling.apply()``.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with polygon geometries.
    size : str, optional
        Column for proportional sizing. When provided, symbols are sized
        proportionally to values. When absent, all symbols have uniform size.
    tile_count : str, optional
        Column with integer tile counts. When provided, each region is
        represented by that many symbols. Cannot be combined with group_by.
    group_by : str, optional
        Column for grouping symbols. When provided, group_index is stored
        on the result for group-level styling and export. Cannot be combined
        with tile_count.
    layout : Layout or str
        Layout instance or string shorthand ("physics", "topology", "grid").
        Pass a Layout instance for full control over algorithm options.
    size_scale : str
        Scaling method for proportional sizing: "sqrt", "linear", or "log".
    size_max_value : float, optional
        Reference max value for consistent scaling across cartograms.
    size_clip : bool
        Whether to clip values exceeding size_max_value.
    size_normalization : str
        How to normalise symbol sizes relative to original geometry areas:

        - ``"max"`` *(default)*: the largest symbol has area equal to the mean
          geometry area.
        - ``"total"``: all sizes are scaled so that the total symbol area equals
          the total original geometry area.  Useful when you want circle area-sum
          to match the geographic area-sum (standard for Dorling cartograms).
    collapse_group : float
        Pre-collapse starting positions toward each group's centroid (0-1).
        At 1.0 all items in a group start coincident at the group centroid,
        matching ``tile_count`` behavior. Only has effect when ``group_by``
        or ``tile_count`` is set. Default: 0.0.
    adjacency_mode : AdjacencyMode
        How to compute adjacency: BINARY, WEIGHTED, or AREA_WEIGHTED.
    adjacency : np.ndarray, optional
        Pre-computed adjacency matrix. If provided, adjacency_mode is ignored.
    distance_tolerance : float, optional
        Buffer distance for adjacency detection.
    pre_scale : bool
        Uniformly scale each geographically connected component of the input
        so its area matches its share of the data before layout. Useful for
        multi-component inputs (e.g. mainland plus islands). Default False.
    show_progress : bool
        Display progress feedback during placement.
    save_history : bool
        Record position snapshots per iteration.
    styling : Styling or None
        Styling configuration. If None, uses defaults (circle symbol, scale=1.0).

    Returns
    -------
    SymbolCartogram
        Result object containing symbol geometries and metrics.

    Examples
    --------
    Proportional sizing (Dorling-style):

    >>> result = create_symbol_cartogram(gdf, "population")

    With layout options:

    >>> from carto_flow.symbol_cartogram import PhysicsBasedLayout, TopologySimulatorOptions
    >>> layout = PhysicsBasedLayout(TopologySimulatorOptions(spacing=0.1, max_iterations=1000))
    >>> result = create_symbol_cartogram(gdf, "population", layout=layout)

    With styling:

    >>> from carto_flow.symbol_cartogram import Styling
    >>> styling = Styling(symbol="hexagon", scale=0.9)
    >>> result = create_symbol_cartogram(gdf, "population", styling=styling)

    Grid layout with preprocessing options:

    >>> result = create_symbol_cartogram(
    ...     gdf, "population",
    ...     layout="grid",
    ...     size_scale="linear",
    ...     adjacency_mode=AdjacencyMode.WEIGHTED,
    ... )

    """
    from .styling import Styling as StylingClass

    # Create layout
    layout_result = create_layout(
        gdf,
        size,
        tile_count=tile_count,
        group_by=group_by,
        layout=layout,
        size_scale=size_scale,
        size_max_value=size_max_value,
        size_clip=size_clip,
        size_normalization=size_normalization,
        tile_size_expansion=tile_size_expansion,
        collapse_group=collapse_group,
        adjacency_mode=adjacency_mode,
        adjacency=adjacency,
        distance_tolerance=distance_tolerance,
        pre_scale=pre_scale,
        show_progress=show_progress,
        save_history=save_history,
    )

    # Apply styling
    if styling is None:
        styling = StylingClass()
    elif isinstance(styling, dict):
        styling = StylingClass(**styling)

    cartogram = styling.apply(layout_result)
    # Store source_gdf reference for attribute merging
    cartogram._source_gdf = gdf
    return cartogram


# ---------------------------------------------------------------------------
# New Layout-Styling Separation API (Phase 3)
# ---------------------------------------------------------------------------


def create_layout(
    gdf: gpd.GeoDataFrame,
    size: str | None = None,
    *,
    tile_count: str | None = None,
    group_by: str | None = None,
    layout: Layout | str = "physics",
    # Preprocessing options
    size_scale: Literal["sqrt", "linear", "log"] = "sqrt",
    size_max_value: float | None = None,
    size_clip: bool = True,
    size_normalization: Literal["max", "total"] = "max",
    tile_size_expansion: Literal["shared", "copied"] = "copied",
    collapse_group: float = 0.0,
    adjacency_mode: AdjacencyMode = AdjacencyMode.BINARY,
    adjacency: NDArray | None = None,
    distance_tolerance: float | None = None,
    pre_scale: bool = False,
    # Runtime options
    show_progress: bool = True,
    save_history: bool = False,
) -> LayoutResult:
    """Create a layout with preprocessing and algorithm execution.

    This is the first step in the layout-styling separation pattern.
    The returned LayoutResult can be styled multiple times without
    re-running the expensive layout computation.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with polygon geometries.
    size : str, optional
        Column for proportional sizing.
    tile_count : str, optional
        Column with integer tile counts. Cannot be combined with group_by.
    group_by : str, optional
        Column for grouping symbols. Cannot be combined with tile_count.
    layout : Layout or str
        Layout instance or string shorthand ("physics", "topology", "grid").
    size_scale : str
        Scaling method for proportional sizing: "sqrt", "linear", or "log".
    size_max_value : float, optional
        Reference max value for consistent scaling across cartograms.
    size_clip : bool
        Whether to clip values exceeding size_max_value.
    size_normalization : str
        How to normalise symbol sizes relative to original geometry areas:

        - ``"max"`` *(default)*: the largest symbol has area equal to the mean
          geometry area.
        - ``"total"``: all sizes are scaled so that the total symbol area equals
          the total original geometry area.
    collapse_group : float
        Pre-collapse starting positions toward each group's centroid (0-1).
        At 1.0 all items in a group start coincident at the group centroid,
        matching ``tile_count`` behavior. Only has effect when ``group_by``
        or ``tile_count`` is set. Default: 0.0.
    adjacency_mode : AdjacencyMode
        How to compute adjacency: BINARY, WEIGHTED, or AREA_WEIGHTED.
    adjacency : np.ndarray, optional
        Pre-computed adjacency matrix. If provided, adjacency_mode is ignored.
    distance_tolerance : float, optional
        Buffer distance for adjacency detection.
    pre_scale : bool
        Uniformly scale each geographically connected component of the input
        so its area matches its share of the data before layout. Useful for
        multi-component inputs (e.g. mainland plus islands). Default False.
    show_progress : bool
        Display progress feedback during placement.
    save_history : bool
        Record position snapshots per iteration.

    Returns
    -------
    LayoutResult
        Immutable layout result ready for styling.

    Examples
    --------
    >>> # Simple usage - default physics layout
    >>> result = create_layout(gdf, "population")

    >>> # String lookup with preprocessing options
    >>> result = create_layout(gdf, "population", layout="grid", size_scale="linear")

    >>> # Explicit Layout instance
    >>> from carto_flow.symbol_cartogram import PhysicsBasedLayout, TopologySimulatorOptions
    >>> layout = PhysicsBasedLayout(TopologySimulatorOptions(spacing=0.1))
    >>> result = create_layout(gdf, "population", layout=layout)

    >>> # Then style the result
    >>> cartogram = result.style(symbol="hexagon", scale=0.9)

    """
    from .layouts import get_layout, prepare_layout_data

    # Resolve layout
    if isinstance(layout, str):
        layout = get_layout(layout)

    # Preprocess
    data = prepare_layout_data(
        gdf,
        size,
        tile_count=tile_count,
        group_by=group_by,
        size_scale=size_scale,
        size_max_value=size_max_value,
        size_clip=size_clip,
        size_normalization=size_normalization,
        tile_size_expansion=tile_size_expansion,
        collapse_group=collapse_group,
        adjacency_mode=adjacency_mode,
        distance_tolerance=distance_tolerance,
        pre_scale=pre_scale,
    )

    # Override adjacency if provided
    if adjacency is not None:
        n = len(data.positions)
        adj_matrix = np.asarray(adjacency, dtype=float)
        if adj_matrix.shape != (n, n):
            raise ValueError(f"adjacency must have shape ({n}, {n}), got {adj_matrix.shape}")
        import dataclasses

        data = dataclasses.replace(data, adjacency=adj_matrix)

    # Compute
    return layout.compute(data, show_progress=show_progress, save_history=save_history)
