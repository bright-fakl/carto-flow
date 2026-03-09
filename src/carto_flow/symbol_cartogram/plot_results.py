"""Result dataclasses returned by visualization functions.

Each plot function returns one of these dataclasses, providing named access to
the underlying matplotlib artist objects alongside the axes, so users can
further style or query the rendered elements after plotting.

Examples
--------
>>> pr = result.plot(facecolor="population", cmap="Reds")
>>> pr.collections[0].set_alpha(0.5)
>>> pr.colorbar.set_label("Population (restyled)")
>>> pr.ax.set_title("Custom title")

>>> ar = plot_adjacency(result, original_gdf=gdf, show_original=True)
>>> k = next(k for k, (i, j) in enumerate(ar.edge_pairs) if {i, j} == {0, 1})
>>> lws = ar.edges.get_linewidths()
>>> lws[k] = 5.0
>>> ar.edges.set_linewidths(lws)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.collections import Collection, LineCollection, PatchCollection, PathCollection, PolyCollection
    from matplotlib.colorbar import Colorbar
    from matplotlib.image import AxesImage
    from matplotlib.legend import Legend
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.quiver import Quiver
    from matplotlib.text import Text


@dataclass
class SymbolsPlotResult:
    """Artists produced by ``plot_symbols()`` and ``SymbolCartogram.plot()``.

    Attributes
    ----------
    ax : plt.Axes
        The axes containing all artists.
    collections : list[PatchCollection]
        Symbol patch collections.  Contains one ``PatchCollection`` when no
        hatching is used; multiple ``PatchCollection``\\s (one per unique hatch
        pattern) when *hatch* is active.  Mutate these to restyle fills,
        edges, or transparency after plotting.
    labels : list[Text]
        Per-symbol ``Text`` artists in symbol order (matching
        ``result.symbols``).  Empty when *label* was not requested.
    colorbar : Colorbar or None
        Colorbar for the facecolor channel.  Present when *facecolor* maps a
        numeric column.
    edge_colorbar : Colorbar or None
        Colorbar for the edgecolor channel.  Present when *edgecolor* maps a
        numeric column **different** from *facecolor*.
    legend : Legend or None
        Patch legend for the facecolor channel.  Present when *facecolor*
        maps a categorical column.
    hatch_legend : Legend or None
        Patch legend showing hatch-pattern ↔ category mappings.  Present
        when *hatch* is a data-driven column name.
    edge_legend : Legend or None
        Patch legend for the edgecolor channel.  Present when *edgecolor*
        maps a categorical column **different** from *facecolor*.
    """

    ax: plt.Axes
    collections: list[PatchCollection]
    labels: list[Text] = field(default_factory=list)
    colorbar: Colorbar | None = None
    edge_colorbar: Colorbar | None = None
    legend: Legend | None = None
    hatch_legend: Legend | None = None
    edge_legend: Legend | None = None


@dataclass
class AdjacencyPlotResult:
    """Artists produced by ``plot_adjacency()``.

    Attributes
    ----------
    ax : plt.Axes
        The axes containing all artists.
    edges : LineCollection or None
        ``LineCollection`` for adjacency edges.  ``None`` when no edges exist
        (empty or all-zero adjacency matrix).
    edge_pairs : list[tuple[int, int]]
        Maps edge index *k* to the pair of symbol indices ``(i, j)`` that
        edge *k* connects.  Parallel to the segments in *edges*.  Empty when
        *edges* is ``None``.
    nodes : PathCollection or None
        ``PathCollection`` (from ``ax.scatter``) for node markers.  ``None``
        when *node_size* = 0.  Positional index *i* corresponds to symbol *i*
        in ``result.symbols``.
    colorbar : Colorbar or None
        Colorbar for edge weights.  Present when *edge_color* is ``None``
        (edges colored by weight via *edge_cmap*) and *colorbar* = ``True``.
    original_collections : list[Collection]
        Collections added by the ``original_gdf.plot()`` underlay when
        *show_original* = ``True``.  Empty otherwise.  Geopandas may create
        multiple collections for mixed geometry types; individual row mapping
        is not supported.
    symbols : SymbolsPlotResult or None
        Result from the symbol underlay when *show_symbols* = ``True``.
        ``None`` otherwise.
    """

    ax: plt.Axes
    edges: LineCollection | None
    edge_pairs: list[tuple[int, int]]
    nodes: PathCollection | None
    colorbar: Colorbar | None = None
    original_collections: list[Collection] = field(default_factory=list)
    symbols: SymbolsPlotResult | None = None


@dataclass
class DisplacementPlotResult:
    """Artists produced by ``plot_displacement()``.

    Attributes
    ----------
    ax : plt.Axes
        The axes containing all artists.
    arrows : Quiver
        The ``Quiver`` artist for displacement arrows.  Arrow *i* corresponds
        to symbol *i* in ``result.symbols``.
    symbols : SymbolsPlotResult or None
        Result from the symbol underlay when *show_symbols* = ``True``.
        ``None`` otherwise.
    """

    ax: plt.Axes
    arrows: Quiver
    symbols: SymbolsPlotResult | None = None


@dataclass
class TilingPlotResult:
    """Artists produced by ``plot_tiling()``.

    Attributes
    ----------
    ax : plt.Axes
        The axes containing all artists.
    assigned_tiles : PatchCollection or None
        ``PatchCollection`` for tiles that have a region assigned.  ``None``
        when *show_assigned* = ``False`` or no assigned tiles exist.
    unassigned_tiles : PatchCollection or None
        ``PatchCollection`` for empty tiles.  ``None`` when
        *show_unassigned* = ``False`` or all tiles are assigned.
    symbols : SymbolsPlotResult or None
        Result from the symbol overlay when *show_symbols* = ``True``.
        ``None`` otherwise.
    """

    ax: plt.Axes
    assigned_tiles: PatchCollection | None
    unassigned_tiles: PatchCollection | None
    symbols: SymbolsPlotResult | None = None


@dataclass
class AdjacencyHeatmapResult:
    """Artists produced by ``plot_adjacency_heatmap()``.

    Attributes
    ----------
    ax : plt.Axes
        The axes containing all artists.
    image : AxesImage
        The ``AxesImage`` from ``ax.imshow()``.  Use ``image.set_cmap()``,
        ``image.set_norm()``, or ``image.set_data()`` to restyle the heatmap.
    annotations : list[Text]
        ``Text`` artists for cell value annotations.  Empty when
        *show_values* = ``False``.
    colorbar : Colorbar or None
        The colorbar attached to the heatmap.  ``None`` when
        *colorbar* = ``False``.
    """

    ax: plt.Axes
    image: AxesImage
    annotations: list[Text] = field(default_factory=list)
    colorbar: Colorbar | None = None


@dataclass
class TilingGridPlotResult:
    """Artists produced by :meth:`TilingResult.plot`.

    Attributes
    ----------
    ax : plt.Axes
        The axes containing all artists.
    fig : plt.Figure
        The figure containing *ax*.
    tiles : PolyCollection
        ``PolyCollection`` of all rendered tile polygons.  Restyle via
        ``tiles.set_facecolors()``, ``tiles.set_edgecolors()``, etc.
    """

    ax: plt.Axes
    fig: plt.Figure
    tiles: PolyCollection


@dataclass
class PrototilePlotResult:
    """Artists produced by :meth:`TilingResult.plot_tile` and :meth:`Tiling.plot_tile`.

    Attributes
    ----------
    ax : plt.Axes
        The axes containing all artists.
    fig : plt.Figure
        The figure containing *ax*.
    tile : MplPolygon
        ``matplotlib.patches.Polygon`` for the prototile outline.  Restyle
        via ``tile.set_facecolor()``, ``tile.set_edgecolor()``, etc.
    vertices : PathCollection or None
        Scatter ``PathCollection`` for vertex dots.  ``None`` when
        *show_vertices* = ``False``.
    """

    ax: plt.Axes
    fig: plt.Figure
    tile: MplPolygon
    vertices: PathCollection | None = None


@dataclass
class ComparisonPlotResult:
    """Artists produced by ``plot_comparison()``.

    Attributes
    ----------
    fig : plt.Figure
        The ``Figure`` containing both subplots.
    ax_original : plt.Axes
        The left axes with the original ``GeoDataFrame`` geometry.
    ax_cartogram : plt.Axes
        The right axes with the symbol cartogram.
    symbols : SymbolsPlotResult
        Full result from ``result.plot()`` on the right panel.
    original_collections : list[Collection]
        Collections added by the ``original_gdf.plot()`` call on the left
        panel.  Geopandas may create multiple collections for mixed geometry
        types; individual row mapping is not supported.
    """

    fig: plt.Figure
    ax_original: plt.Axes
    ax_cartogram: plt.Axes
    symbols: SymbolsPlotResult
    original_collections: list[Collection] = field(default_factory=list)
