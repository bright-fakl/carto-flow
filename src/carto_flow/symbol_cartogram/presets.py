"""Named cartogram entry points.

Each function wraps ``create_symbol_cartogram`` with a specific algorithmic
style and exposes only the parameters that meaningfully vary for that style.
For full control use ``create_symbol_cartogram`` or ``create_layout`` directly.

2x2 structure
-------------
The four main cartogram styles differ on two axes:

- **Grouped vs not grouped**: whether symbols belong to labeled groups
  (``group_by`` / ``tile_count``).
- **Pull toward origin vs toward centroid**: whether symbols stay close to
  geographic positions or pack toward a common center.

+------------------+-----------------------------+-----------------------------+
|                  | Not grouped                 | Grouped                     |
+==================+=============================+=============================+
| **Origin pull**  | ``geographic_cartogram``    | ``geographic_grouped_cart…`` |
+------------------+-----------------------------+-----------------------------+
| **Centroid pull**| ``dorling_cartogram``       | ``dorling_grouped_cartogram``|
+------------------+-----------------------------+-----------------------------+

``centroid_cartogram`` is a special case: Stage 1 only (no compaction), no
pull in either direction.

Grid-based
----------
demers_cartogram
    Proportional squares on a grid (Demers style).
tile_map_cartogram
    Uniform hexagons on a grid (tile map).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import geopandas as gpd

from .api import create_symbol_cartogram
from .layouts import CirclePackingLayout, GridBasedLayout, GridBasedLayoutOptions
from .result import SymbolCartogram
from .styling import Styling

if TYPE_CHECKING:
    pass


def centroid_cartogram(
    gdf: gpd.GeoDataFrame,
    size: str | None = None,
    *,
    spacing: float = 0.05,
    expansion: float = 0.0,
    size_normalization: Literal["max", "total"] = "total",
    show_progress: bool = True,
) -> SymbolCartogram:
    """Place symbols at geographic centroids with overlap removal only.

    Stage 1 only — no Stage 2 compaction. Fast and geographically accurate.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input geometries.
    size : str, optional
        Column for proportional sizing. Uniform size when omitted.
    spacing : float
        Minimum gap between symbols as fraction of average radius. Default: 0.05
    expansion : float
        Stage 1 expansion fraction (0-1). 1 = full exact expansion (default);
        0 = Gauss-Seidel only (more topology-preserving). Default: 1.0
    size_normalization : {"max", "total"}
        ``"total"`` (total symbol area equals total geometry area) or
        ``"max"`` (largest symbol has the mean geometry area, so coverage is
        ``mean(value) / max(value)``). Default: ``"total"``
    show_progress : bool
        Display progress bar. Default: True

    """
    return create_symbol_cartogram(
        gdf,
        size,
        layout=CirclePackingLayout.centroid(spacing=spacing, expansion=expansion),
        size_normalization=size_normalization,
        show_progress=show_progress,
    )


def dorling_cartogram(
    gdf: gpd.GeoDataFrame,
    size: str | None = None,
    *,
    spacing: float = 0.05,
    compactness: float = 0.8,
    topology_weight: float = 0.0,
    neighbor_weight: float = 0.0,
    size_normalization: Literal["max", "total"] = "total",
    show_progress: bool = True,
) -> SymbolCartogram:
    """Classic Dorling cartogram: compact proportional circles.

    Circles pack toward the global centroid; no origin attraction.
    ``size_normalization`` is ``"total"`` so total circle area equals total
    geographic area (classic Dorling convention).

    Parameters
    ----------
    gdf : GeoDataFrame
        Input geometries.
    size : str, optional
        Column for proportional sizing. Uniform size when omitted.
    spacing : float
        Minimum gap between symbols as fraction of average radius. Default: 0.05
    compactness : float
        Global centroid attraction strength (0-1). Default: 0.8
    topology_weight : float
        Topology preservation strength (0-1). Default: 0.0
    neighbor_weight : float
        Neighbor tangency force strength. Default: 0.0
    size_normalization : {"max", "total"}
        ``"total"`` (total symbol area equals total geometry area) or
        ``"max"`` (largest symbol has the mean geometry area, so coverage is
        ``mean(value) / max(value)``). Default: ``"total"``
    show_progress : bool
        Display progress bar. Default: True

    """
    return create_symbol_cartogram(
        gdf,
        size,
        layout=CirclePackingLayout.dorling(
            spacing=spacing,
            compactness=compactness,
            topology_weight=topology_weight,
            neighbor_weight=neighbor_weight,
        ),
        size_normalization=size_normalization,
        show_progress=show_progress,
    )


def geographic_cartogram(
    gdf: gpd.GeoDataFrame,
    size: str | None = None,
    *,
    spacing: float = 0.05,
    origin_weight: float = 0.5,
    topology_weight: float = 0.5,
    neighbor_weight: float = 0.5,
    size_normalization: Literal["max", "total"] = "total",
    show_progress: bool = True,
) -> SymbolCartogram:
    """Geography-preserving cartogram.

    Symbols stay close to their geographic positions via origin attraction.
    No global compaction.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input geometries.
    size : str, optional
        Column for proportional sizing. Uniform size when omitted.
    spacing : float
        Minimum gap between symbols as fraction of average radius. Default: 0.05
    origin_weight : float
        Origin attraction strength (0-1). Default: 0.5
    topology_weight : float
        Topology preservation strength (0-1). Default: 0.5
    neighbor_weight : float
        Neighbor tangency force strength. Default: 0.5
    size_normalization : {"max", "total"}
        ``"total"`` (total symbol area equals total geometry area) or
        ``"max"`` (largest symbol has the mean geometry area, so coverage is
        ``mean(value) / max(value)``). Default: ``"total"``
    show_progress : bool
        Display progress bar. Default: True

    """
    return create_symbol_cartogram(
        gdf,
        size,
        layout=CirclePackingLayout.geographic(
            spacing=spacing,
            origin_weight=origin_weight,
            topology_weight=topology_weight,
            neighbor_weight=neighbor_weight,
        ),
        size_normalization=size_normalization,
        show_progress=show_progress,
    )


def dorling_grouped_cartogram(
    gdf: gpd.GeoDataFrame,
    size: str | None = None,
    *,
    spacing: float = 0.05,
    group_by: str | None = None,
    tile_count: str | None = None,
    compactness: float = 0.1,
    group_weight: float = 0.5,
    topology_weight: float = 0.5,
    neighbor_weight: float = 0.5,
    collapse_group: float = 1.0,
    size_normalization: Literal["max", "total"] = "total",
    show_progress: bool = True,
) -> SymbolCartogram:
    """Grouped Dorling cartogram: circles pack toward group centroids.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input geometries.
    size : str, optional
        Column for proportional sizing. Uniform size when omitted.
    spacing : float
        Minimum gap between symbols as fraction of average radius. Default: 0.05
    group_by : str, optional
        Column defining symbol groups.
    tile_count : str, optional
        Column with integer tile counts per geometry.
    compactness : float
        Global centroid attraction strength (0-1). Default: 0.1
    group_weight : float
        Group centroid attraction strength. Default: 0.5
    topology_weight : float
        Topology preservation strength (0-1). Default: 0.5
    neighbor_weight : float
        Neighbor tangency force strength. Default: 0.5
    collapse_group : float
        Fraction to collapse group members toward their group centroid before
        Stage 1 (0-1). 1.0 = fully coincident (recommended to prevent
        intermingling). Default: 1.0
    size_normalization : {"max", "total"}
        ``"total"`` (total symbol area equals total geometry area) or
        ``"max"`` (largest symbol has the mean geometry area, so coverage is
        ``mean(value) / max(value)``). Default: ``"total"``
    show_progress : bool
        Display progress bar. Default: True

    """
    return create_symbol_cartogram(
        gdf,
        size,
        layout=CirclePackingLayout.dorling_grouped(
            spacing=spacing,
            compactness=compactness,
            group_weight=group_weight,
            topology_weight=topology_weight,
            neighbor_weight=neighbor_weight,
        ),
        group_by=group_by,
        tile_count=tile_count,
        collapse_group=collapse_group,
        size_normalization=size_normalization,
        show_progress=show_progress,
    )


def geographic_grouped_cartogram(
    gdf: gpd.GeoDataFrame,
    size: str | None = None,
    *,
    spacing: float = 0.05,
    group_by: str | None = None,
    tile_count: str | None = None,
    origin_weight: float = 0.5,
    group_weight: float = 0.3,
    topology_weight: float = 0.5,
    neighbor_weight: float = 0.5,
    collapse_group: float = 1.0,
    size_normalization: Literal["max", "total"] = "total",
    show_progress: bool = True,
) -> SymbolCartogram:
    """Grouped geography-preserving cartogram.

    Symbols stay near geographic positions while maintaining group cohesion.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input geometries.
    size : str, optional
        Column for proportional sizing. Uniform size when omitted.
    spacing : float
        Minimum gap between symbols as fraction of average radius. Default: 0.05
    group_by : str, optional
        Column defining symbol groups.
    tile_count : str, optional
        Column with integer tile counts per geometry.
    origin_weight : float
        Origin attraction strength (0-1). Default: 0.5
    group_weight : float
        Group centroid attraction strength. Default: 0.3
    topology_weight : float
        Topology preservation strength (0-1). Default: 0.5
    neighbor_weight : float
        Neighbor tangency force strength. Default: 0.5
    collapse_group : float
        Fraction to collapse group members toward their group centroid before
        Stage 1 (0-1). 1.0 = fully coincident (recommended to prevent
        intermingling). Default: 1.0
    size_normalization : {"max", "total"}
        ``"total"`` (total symbol area equals total geometry area) or
        ``"max"`` (largest symbol has the mean geometry area, so coverage is
        ``mean(value) / max(value)``). Default: ``"total"``
    show_progress : bool
        Display progress bar. Default: True

    """
    return create_symbol_cartogram(
        gdf,
        size,
        layout=CirclePackingLayout.geographic_grouped(
            spacing=spacing,
            origin_weight=origin_weight,
            group_weight=group_weight,
            topology_weight=topology_weight,
            neighbor_weight=neighbor_weight,
        ),
        group_by=group_by,
        tile_count=tile_count,
        collapse_group=collapse_group,
        size_normalization=size_normalization,
        show_progress=show_progress,
    )


def demers_cartogram(
    gdf: gpd.GeoDataFrame,
    size: str | None = None,
    *,
    spacing: float = 0.05,
    show_progress: bool = True,
) -> SymbolCartogram:
    """Demers-style cartogram: proportional squares on a grid.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input geometries.
    size : str, optional
        Column for proportional sizing. Uniform size when omitted.
    spacing : float
        Gap between squares as fraction of cell size. Default: 0.05
    show_progress : bool
        Display progress bar. Default: True

    """
    return create_symbol_cartogram(
        gdf,
        size,
        layout=GridBasedLayout(GridBasedLayoutOptions(tiling="square", spacing=spacing)),
        styling=Styling(symbol="square"),
        show_progress=show_progress,
    )


def tile_map_cartogram(
    gdf: gpd.GeoDataFrame,
    *,
    spacing: float = 0.05,
    rotation: float = 0.0,
    show_progress: bool = True,
) -> SymbolCartogram:
    """Tile map: uniform hexagons on a grid.

    Symbols are always uniform size (no ``size`` parameter) — this is a
    tile map, not a proportional cartogram.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input geometries.
    spacing : float
        Gap between hexagons as fraction of cell size. Default: 0.05
    rotation : float
        Grid rotation in degrees (counter-clockwise). Default: 0.0
    show_progress : bool
        Display progress bar. Default: True

    """
    return create_symbol_cartogram(
        gdf,
        layout=GridBasedLayout(
            GridBasedLayoutOptions(
                tiling="hexagon",
                spacing=spacing,
                rotation=rotation,
            )
        ),
        styling=Styling(symbol="hexagon"),
        show_progress=show_progress,
    )
