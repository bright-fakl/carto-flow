"""Result dataclasses returned by visualization functions.

Each plot function returns one of these dataclasses, providing named access to
the underlying matplotlib artist objects alongside the axes, so users can
further style or query the rendered elements after plotting.

Examples
--------
>>> pr = plot_partitions(result, color_by="category")
>>> pr.partition_collections[0].set_alpha(0.5)
>>> pr.legend.set_title("Custom legend title")

>>> pr = plot_partitions(result, color_by="area")
>>> pr.colorbar.set_label("Area (km²)")
>>> pr.complement_collections[0].set_edgecolor("red")

>>> dr = plot_dot_density(gdf, columns="population", n_dots=100)
>>> dr.dot_collections[0].set_sizes([6])
>>> dr.legend.set_title("Variable")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.collections import Collection, PathCollection
    from matplotlib.colorbar import Colorbar
    from matplotlib.legend import Legend


@dataclass
class PartitionsPlotResult:
    """Artists produced by ``plot_partitions()``.

    Attributes
    ----------
    ax : plt.Axes
        The axes containing all artists.
    partition_collections : list[Collection]
        Geometry collections for non-complement partitions, in the order
        they were plotted (one collection per styling group — geometries
        sharing the same alpha, edgecolor, and linewidth are batched together).
    complement_collections : list[Collection]
        Geometry collections for complement (remainder) geometries, in the
        order they were plotted.  Empty when *include_complement* = ``False``
        or when no complement geometries exist.
    colorbar : Colorbar or None
        Colorbar for continuous coloring.  Present when *color_by* is not
        ``'category'`` and *legend* = ``True``.
    legend : Legend or None
        Patch legend for categorical coloring.  Present when *color_by* =
        ``'category'`` and *legend* = ``True``.
    """

    ax: plt.Axes
    partition_collections: list[Collection]
    complement_collections: list[Collection] = field(default_factory=list)
    colorbar: Colorbar | None = None
    legend: Legend | None = None


@dataclass
class DotDensityPlotResult:
    """Artists produced by ``plot_dot_density()``.

    Attributes
    ----------
    ax : plt.Axes
        The axes containing all artists.
    dot_collections : list[PathCollection]
        Scatter collections for dots, one per category, in the order categories
        were encountered in the ``dots_gdf``.
    base_collection : Collection or None
        The polygon patch collection for the background geometries.
        ``None`` when no base geometries were plotted.
    legend : Legend or None
        Categorical legend for dot colors. Present when *legend* = ``True``
        and at least one category exists.
    """

    ax: plt.Axes
    dot_collections: list[PathCollection]
    base_collection: Collection | None = None
    legend: Legend | None = None
