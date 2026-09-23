# ruff: noqa: RUF002
"""Result dataclasses returned by visualization functions.

Each plot function returns one of these dataclasses, providing named access to
the underlying matplotlib artist objects alongside the axes, so users can
further style or query the rendered elements after plotting.

Examples
--------
>>> cr = cartogram.plot(column="population")
>>> cr.collections[0].set_edgecolor("black")

>>> conv = plot_convergence(cartogram)
>>> if conv.max_line:
...     conv.max_line.set_linewidth(3)
>>> conv.tolerance_lines[0].set_linestyle("--")

>>> df_result = plot_density_field(cartogram)
>>> df_result.colorbar.set_label("Density (km⁻²)")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.collections import Collection
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage
    from matplotlib.lines import Line2D
    from matplotlib.quiver import Quiver


@dataclass
class CartogramPlotResult:
    """Artists produced by ``plot_cartogram()`` and ``Cartogram.plot()``.

    Attributes
    ----------
    ax : plt.Axes
        The axes containing all artists.
    collections : list[Collection]
        Geometry collections added by ``gdf.plot()``.  Geopandas may create
        multiple collections for mixed geometry types; individual row mapping
        is not supported.
    """

    ax: plt.Axes
    collections: list[Collection]


@dataclass
class CartogramComparisonResult:
    """Artists produced by ``plot_comparison()``.

    Attributes
    ----------
    fig : plt.Figure
        The ``Figure`` containing both subplots.
    ax_left : plt.Axes
        The left axes.
    ax_right : plt.Axes
        The right axes (always the ``Cartogram``).
    left_collections : list[Collection]
        Collections added by the left panel's ``gdf.plot()`` call.
    right_collections : list[Collection]
        Collections added by the right panel's ``gdf.plot()`` call.
    """

    fig: Figure
    ax_left: plt.Axes
    ax_right: plt.Axes
    left_collections: list[Collection]
    right_collections: list[Collection]


@dataclass
class ConvergencePlotResult:
    """Artists produced by ``plot_convergence()``.

    Attributes
    ----------
    ax : plt.Axes
        Primary axes.  Holds the max error line when *show_both* = ``True``,
        or the mean error line when *show_both* = ``False``.
    max_line : Line2D or None
        The max error ``Line2D`` on the primary axis.  ``None`` when
        *show_both* = ``False`` or when no max-error data is available.
    mean_line : Line2D or None
        The mean error ``Line2D``.  On *ax_mean* when *show_both* = ``True``,
        otherwise on *ax*.  ``None`` when no mean-error data is available.
    ax_mean : Axes or None
        Secondary twin y-axis for mean error.  Present only when
        *show_both* = ``True`` and mean-error data exists.
    tolerance_lines : list[Line2D]
        Horizontal lines (``ax.axhline``) for error tolerances.  Order:
        mean tolerance first (on *ax_mean* or *ax*), then max tolerance
        (on *ax*) — only lines that were actually drawn are included.
    recompute_lines : list[Line2D]
        Vertical lines (``ax.axvline``) marking density recomputation
        iterations.  Empty when *show_recompute* = ``False``.
    """

    ax: plt.Axes
    max_line: Line2D | None
    mean_line: Line2D | None
    ax_mean: Axes | None
    tolerance_lines: list[Line2D] = field(default_factory=list)
    recompute_lines: list[Line2D] = field(default_factory=list)


@dataclass
class DensityFieldResult:
    """Artists produced by ``plot_density_field()``.

    Attributes
    ----------
    ax : plt.Axes
        The axes containing the density heatmap.
    image : AxesImage
        The ``AxesImage`` from ``ax.imshow()``.  Use ``image.set_cmap()`` or
        ``image.set_norm()`` to restyle the colormap after plotting.
    colorbar : Colorbar
        The colorbar attached to the heatmap.
    """

    ax: plt.Axes
    image: AxesImage
    colorbar: Colorbar


@dataclass
class VelocityFieldResult:
    """Artists produced by ``plot_velocity_field()``.

    Attributes
    ----------
    ax : plt.Axes
        The axes containing the quiver plot.
    arrows : Quiver
        The ``Quiver`` artist for velocity arrows.
    colorbar : Colorbar or None
        Colorbar for arrow coloring.  Present when *colorbar* = ``True`` and
        *color_by* is not ``None``.
    """

    ax: plt.Axes
    arrows: Quiver
    colorbar: Colorbar | None = None


@dataclass
class WorkflowConvergencePlotResult:
    """Artists produced by ``plot_workflow_convergence()``.

    Attributes
    ----------
    ax : plt.Axes
        Primary axes.  Holds max error lines when *metric* is ``'max'`` or
        ``'both'``, otherwise mean error lines.
    ax_mean : Axes or None
        Secondary twin y-axis for mean error.  Present only when
        *metric* = ``'both'``.
    max_lines : list[Line2D]
        One ``Line2D`` per run for max error (on *ax*).  Empty when
        *metric* = ``'mean'``.
    mean_lines : list[Line2D]
        One ``Line2D`` per run for mean error.  On *ax_mean* when
        *metric* = ``'both'``, otherwise on *ax*.  Empty when
        *metric* = ``'max'``.
    boundary_lines : list[Line2D]
        Vertical lines (``ax.axvline``) at run boundaries.  Empty when
        *show_run_boundaries* = ``False`` or there is only one run.
    tolerance_lines : list[Line2D]
        Horizontal lines (``ax.axhline``) for error tolerances.
    """

    ax: plt.Axes
    ax_mean: Axes | None
    max_lines: list[Line2D] = field(default_factory=list)
    mean_lines: list[Line2D] = field(default_factory=list)
    boundary_lines: list[Line2D] = field(default_factory=list)
    tolerance_lines: list[Line2D] = field(default_factory=list)


@dataclass
class ModulatorPreviewResult:
    """Artists produced by :func:`preview_modulator`.

    Attributes
    ----------
    ax : plt.Axes
        The axes containing all artists.
    arrows : Quiver
        The ``Quiver`` artist for the modulated velocity arrows, colored by
        amplification factor.
    colorbar : Colorbar
        Colorbar for the arrow amplification color scale.
    geometry_collections : list[Collection]
        Collections added by ``gdf.plot()`` for the background geometry layer.
        Empty when no GeoDataFrame was provided.
    input_arrows : Quiver or None
        The ``Quiver`` artist for the pre-modulation velocity arrows.
        ``None`` when ``'input'`` is not in *show_vectors*.
    diff_arrows : Quiver or None
        The ``Quiver`` artist for the difference field ``(vx_out − vx_in,
        vy_out − vy_in)``.  ``None`` when ``'diff'`` is not in *show_vectors*.
    diff_colorbar : Colorbar or None
        Colorbar for the difference quiver.  ``None`` when not shown.
    heatmap_image : AxesImage or None
        The ``AxesImage`` from ``ax.imshow()`` for the heatmap overlay
        (weight, magnitude, or angle field).  ``None`` when *heatmap* is
        ``None``.
    heatmap_colorbar : Colorbar or None
        Colorbar for the heatmap.  ``None`` when *heatmap* is ``None`` or
        *show_colorbar* is ``False``.
    """

    ax: plt.Axes
    arrows: Quiver
    colorbar: Colorbar
    geometry_collections: list[Collection] = field(default_factory=list)
    input_arrows: Quiver | None = None
    diff_arrows: Quiver | None = None
    diff_colorbar: Colorbar | None = None
    heatmap_image: AxesImage | None = None
    heatmap_colorbar: Colorbar | None = None


@dataclass
class DensityModulatorPreviewResult:
    """Artists produced by :func:`preview_density_modulator`.

    Attributes
    ----------
    ax : plt.Axes
        The axes containing all artists.
    image : AxesImage
        The main density heatmap — output density, input density, or ratio
        field, depending on the ``show`` parameter.
    colorbar : Colorbar or None
        Colorbar for the heatmap.  ``None`` when *show_colorbar* is ``False``.
    geometry_collections : list[Collection]
        Collections added by ``gdf.plot()`` for the background geometry layer.
        Empty when *show_geometry* is ``False``.
    """

    ax: plt.Axes
    image: AxesImage
    colorbar: Colorbar | None
    geometry_collections: list[Collection] = field(default_factory=list)
