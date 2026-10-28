# ruff: noqa: RUF002
"""Tiling abstractions for tile-based symbol cartograms.

This module defines the ``Tiling`` ABC and concrete tiling classes that generate
congruent tilings of the plane. Each tiling produces a ``TilingResult`` containing
tile polygons, per-tile transforms, and precomputed adjacency matrices. The tilings
are `isotile <https://github.com/bright-fakl/isotile>`_ tilings with a size convention
(``tile_size``) and the output format of the layouts.

Tiling Classes
--------------
SquareTiling
    Regular square grid.
HexagonTiling
    Regular hexagonal grid (pointy-top or flat-top).
TriangleTiling
    Tiling by congruent triangles (any triangle tiles the plane).
QuadrilateralTiling
    Tiling by congruent quadrilaterals (any simple quadrilateral tiles the plane).
IsohedralTiling
    Tiling of one of the 81 isohedral types, optionally with curved edges.

Tiles cover the region (bounds with margin): a tile is returned if it overlaps the region
with positive area. The lattice is anchored at the center of the region, so a region that
is expanded by the same amount on all sides keeps its tiles. The order of the tiles is not
part of the interface.
"""

from __future__ import annotations

import copy
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import isotile as _isotile
import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.sparse import csr_array
from shapely.affinity import rotate, scale, translate
from shapely.geometry import Point, Polygon
from typing_extensions import Self

from .._plot_params import deprecated_params
from ..style import resolve_style, styled

if TYPE_CHECKING:
    from isotile import AdjacencyKind, TileSet

    from .symbols import Symbol


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Default bounding box used by :meth:`Tiling.generate` when *bounds* is omitted.
DEFAULT_BOUNDS: tuple[float, float, float, float] = (-1.0, -1.0, 1.0, 1.0)

#: Default tile count used by :meth:`Tiling.generate` when neither *n_tiles* nor
#: *tile_size* is specified.
DEFAULT_N_TILES: int = 64


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TileTransform:
    """Transform for a single tile: center position, rotation, and flip.

    Attributes
    ----------
    center : tuple[float, float]
        Tile center in world coordinates.
    rotation : float
        Rotation angle in degrees (counter-clockwise).
    flipped : bool
        Whether the tile is reflected about the vertical axis (before rotation).

    """

    center: tuple[float, float]
    rotation: float = 0.0
    flipped: bool = False


@dataclass
class TilingResult:
    """Output of tiling generation.

    Attributes
    ----------
    polygons : list[Polygon]
        Tile polygons in world coordinates.
    transforms : list[TileTransform]
        Per-tile transforms (center, rotation, flip).
    adjacency : scipy.sparse.csr_array
        (m, m) sparse boolean adjacency matrix between tiles, of the kind
        given to ``generate`` (shared edge by default). Symmetric, with sorted
        column indices in every row: the neighbors of tile ``t`` are
        ``adjacency.indices[adjacency.indptr[t]:adjacency.indptr[t + 1]]``.
    vertex_adjacency : scipy.sparse.csr_array | None
        (m, m) sparse boolean vertex-only adjacency matrix (tiles sharing
        exactly one vertex but no edge), in the same layout as ``adjacency``.
        *None* for a result that was loaded.
    tile_size : float
        Size parameter used for generation.
    inscribed_radius : float
        Distance from the center of the canonical tile to its boundary, at
        the generated size.
    canonical_tile : Polygon
        The unit tile shape (before transforms), at the generated size.
    n_base_vertices : int or None
        Topological vertex count of the prototile.  For most tilings this
        equals ``len(canonical_tile.exterior.coords) - 1``, but for
        isohedral tilings with custom edge curves the ``canonical_tile``
        polygon has many more points due to spline interpolation.  Used by
        :meth:`plot_tile` to select the correct auto-color.
    tile_set : isotile.TileSet or None
        The isotile tiles (cell and aspect of each tile, the tiling) in the
        order of ``polygons``.  *None* for a result that was loaded or
        rotated.

    """

    polygons: list[Polygon]
    transforms: list[TileTransform]
    adjacency: csr_array
    vertex_adjacency: csr_array | None
    tile_size: float
    inscribed_radius: float
    canonical_tile: Polygon
    n_base_vertices: int | None = None
    tile_set: TileSet | None = None

    @property
    def centers(self) -> NDArray[np.floating]:
        """(m, 2) array of tile centers."""
        return np.array([(t.center[0], t.center[1]) for t in self.transforms], dtype=float)

    @property
    def n_tiles(self) -> int:
        """Number of tiles."""
        return len(self.polygons)

    def rotate(self, angle: float) -> TilingResult:
        """Rotate all tile positions and orientations.

        Returns a new TilingResult with all tile centers rotated about the
        origin (0, 0) and all tile rotations incremented by the given angle.
        The tiles are not regenerated, so a region that the tiles covered is
        not covered afterwards, and a result far from the origin moves far.
        To rotate the lattice about the center of a region and cover the region
        exactly, use :meth:`Tiling.rotated` before :meth:`Tiling.generate`.

        Parameters
        ----------
        angle : float
            Rotation angle in degrees (counter-clockwise).

        Returns
        -------
        TilingResult
            New TilingResult with rotated tiles.

        Notes
        -----
        The adjacency matrices are preserved since rotation is a rigid
        transformation that doesn't change neighbor relationships.
        ``tile_set`` is not carried over.

        """
        cos_a = np.cos(np.radians(angle))
        sin_a = np.sin(np.radians(angle))

        # Rotate each tile's center and add to its rotation
        new_transforms = []
        new_polygons = []
        for t, poly in zip(self.transforms, self.polygons, strict=False):
            # Rotate center position
            cx, cy = t.center
            new_cx = cx * cos_a - cy * sin_a
            new_cy = cx * sin_a + cy * cos_a

            # Add rotation angle
            new_rotation = t.rotation + angle

            new_transforms.append(
                TileTransform(
                    center=(new_cx, new_cy),
                    rotation=new_rotation,
                    flipped=t.flipped,
                ),
            )

            # Rotate the polygon geometry
            new_polygons.append(rotate(poly, angle, origin=(0, 0), use_radians=False))

        # Rotate the canonical tile
        rotated_canonical = rotate(self.canonical_tile, angle, origin=(0, 0), use_radians=False)

        return TilingResult(
            polygons=new_polygons,
            transforms=new_transforms,
            adjacency=self.adjacency,  # Adjacency unchanged by rotation
            vertex_adjacency=self.vertex_adjacency,
            tile_size=self.tile_size,
            inscribed_radius=self.inscribed_radius,  # Unchanged for rotation about origin
            canonical_tile=rotated_canonical,
            n_base_vertices=self.n_base_vertices,
        )

    @deprecated_params(facecolor="face_color", edgecolor="edge_color", cmap="colormap")
    @styled
    def plot(
        self,
        ax=None,
        *,
        color_by: str = "uniform",
        facecolor: str | None = None,
        edgecolor: str | None = None,
        cmap: str | None = None,
        alpha: float = 0.85,
        linewidth: float = 0.5,
    ):
        """Plot the tiling using matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created if *ax* is ``None``.
        color_by : {"uniform", "aspect", "rotation", "index", "x", "y", "random"}
            Coloring scheme:

            * ``"uniform"`` — all tiles use *facecolor*.
            * ``"aspect"`` — color by the aspect of the tile, its index within the
              cell of the tiling (``aspect / n_aspects`` → cmap). Needs
              ``tile_set``; raises ``ValueError`` if it is ``None``. A tiling
              with one aspect has one color.
            * ``"rotation"`` — color by tile rotation angle (0–360 → cmap).
            * ``"index"`` — color by tile index (0…n−1 → cmap).
            * ``"x"`` — color by tile centroid x coordinate (normalized).
            * ``"y"`` — color by tile centroid y coordinate (normalized).
            * ``"random"`` — random color per tile drawn from *cmap*.
        facecolor : str, optional
            Fill color used in ``"uniform"`` mode. Default: the fill value of the style, else ``"#4e9af1"``.
        edgecolor : str, optional
            Stroke color for tile outlines (the outline is the subject of the plot). Default: the stroke color of the
            style, else ``"#333333"``.
        cmap : str, optional
            Matplotlib cmap name used in the other modes. Default: the cyclic colormap of the style for ``"aspect"``
            and ``"rotation"`` and the sequential colormap for the rest, else ``"tab10"``.
        alpha : float
            Tile fill transparency.
        linewidth : float
            Tile outline width in points.

        Returns
        -------
        TilingGridPlotResult
            Named result with ``ax``, ``fig``, and ``tiles`` (the
            ``PolyCollection``).

        Examples
        --------
        >>> from carto_flow.symbol_cartogram import SquareTiling
        >>> pr = SquareTiling().generate().plot(color_by="index")
        >>> pr.tiles.set_alpha(0.5)
        """
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection

        from .plot_results import TilingGridPlotResult

        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
        else:
            fig = ax.get_figure()

        _CMAP_MODES = {"aspect", "rotation", "index", "x", "y", "random"}
        if color_by not in _CMAP_MODES | {"uniform"}:
            raise ValueError(
                f"Unknown color_by={color_by!r}. "
                "Choose from 'uniform', 'aspect', 'rotation', 'index', 'x', 'y', 'random'."
            )
        if color_by == "aspect" and self.tile_set is None:
            raise ValueError(
                "color_by='aspect' needs the tile set of the tiling, which this TilingResult does not have "
                "(it was loaded from a file or rotated). Use color_by='rotation' instead."
            )

        st = resolve_style()
        if facecolor is None:
            facecolor = st.channel_value("fill", "#4e9af1")
        if edgecolor is None:
            edgecolor = st.channel_value("stroke", "#333333")
        n = len(self.polygons)
        colormap = (
            plt.get_cmap(cmap)
            if cmap is not None
            else st.colormap("cyclic" if color_by in ("aspect", "rotation") else "sequential") or plt.get_cmap("tab10")
        )

        coords = [list(zip(*p.exterior.xy, strict=False)) for p in self.polygons]
        if color_by == "uniform":
            colors = [facecolor] * n
        elif color_by == "aspect":
            tile_set = cast("TileSet", self.tile_set)  # checked above
            n_aspects = tile_set.tiling.type.n_aspects
            colors = [mcolors.to_hex(colormap(int(a) / n_aspects)) for a in tile_set.aspect]
        elif color_by == "rotation":
            colors = [mcolors.to_hex(colormap((tr.rotation % 360) / 360.0)) for tr in self.transforms]
        elif color_by == "index":
            colors = [mcolors.to_hex(colormap(i / max(n - 1, 1))) for i in range(n)]
        elif color_by == "x":
            xs = self.centers[:, 0]
            lo, hi = xs.min(), xs.max()
            values = (xs - lo) / max(hi - lo, 1e-9)
            colors = [mcolors.to_hex(colormap(v)) for v in values]
        elif color_by == "y":
            ys = self.centers[:, 1]
            lo, hi = ys.min(), ys.max()
            values = (ys - lo) / max(hi - lo, 1e-9)
            colors = [mcolors.to_hex(colormap(v)) for v in values]
        else:  # random
            colors = [mcolors.to_hex(colormap(v)) for v in np.random.default_rng().random(n)]

        col = PolyCollection(
            coords,
            facecolors=colors,
            edgecolors=edgecolor,
            linewidths=linewidth,
            alpha=alpha,
        )
        ax.add_collection(col)
        ax.autoscale_view()
        ax.axis("off")
        return TilingGridPlotResult(ax=ax, fig=fig, tiles=col)

    @deprecated_params(facecolor="face_color", edgecolor="edge_color")
    @styled
    def plot_tile(
        self,
        ax=None,
        *,
        facecolor: str | None = None,
        edgecolor: str | None = None,
        alpha: float = 0.7,
        show_vertices: bool = True,
        show_vertex_labels: bool = False,
        linewidth: float = 1.5,
    ):
        """Plot the canonical (proto)tile shape using matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created if *ax* is ``None``.
        facecolor : str, optional
            Fill color.  When ``None`` the color is chosen automatically by
            vertex count to match the panel UI palette
            (3 → pink, 4 → blue, 5 → yellow, 6 → green).
        edgecolor : str, optional
            Outline color. Default: the stroke color of the style, else ``"#1971c2"``.
        alpha : float
            Fill transparency.
        show_vertices : bool
            Draw a dot at each tile vertex.
        show_vertex_labels : bool
            Annotate each vertex with its 0-based index.
        linewidth : float
            Outline width in points.

        Returns
        -------
        PrototilePlotResult
            Named result with ``ax``, ``fig``, ``tile`` (the patch), and
            ``vertices`` (scatter artist, or ``None``).

        Examples
        --------
        >>> from carto_flow.symbol_cartogram import IsohedralTiling
        >>> result = IsohedralTiling(1).generate()
        >>> pr = result.plot_tile(show_vertex_labels=True)
        >>> pr.tile.set_facecolor("coral")
        """
        return _plot_polygon_tile(
            self.canonical_tile,
            ax,
            facecolor,
            edgecolor,
            alpha,
            show_vertices,
            show_vertex_labels,
            linewidth,
            n_vertices=self.n_base_vertices,
        )


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class Tiling(ABC):
    """A congruent tiling of the plane.

    Subclasses implement specific tiling patterns. The ``generate`` method
    produces tiles covering a bounding box.
    """

    @abstractmethod
    def generate(
        self,
        bounds: tuple[float, float, float, float] = DEFAULT_BOUNDS,
        n_tiles: int | None = None,
        tile_size: float | None = None,
        margin: float = 0.1,
        adjacency_type: AdjacencyKind = "edge",
    ) -> TilingResult:
        """Generate the tiles that cover the bounding box.

        A tile is returned if it overlaps the box expanded by *margin* with
        positive area. The lattice is anchored at the center of the expanded
        box.

        Parameters
        ----------
        bounds : tuple, optional
            Bounding box as (minx, miny, maxx, maxy).  Defaults to the unit
            box ``(-1, -1, 1, 1)`` (see :data:`DEFAULT_BOUNDS`).
        n_tiles : int, optional
            Approximate number of tiles: the tile size is set so that this
            many tiles fit into the expanded box, and the tiles along its
            boundary come on top. Mutually exclusive with *tile_size*.
            When neither is given, defaults to :data:`DEFAULT_N_TILES`.
        tile_size : float, optional
            Size of each tile. Mutually exclusive with *n_tiles*.
        margin : float
            Fractional margin around bounding box.
        adjacency_type : {"edge", "vertex", "vertex-only"}
            Kind of ``adjacency``: tiles sharing an edge, tiles sharing an
            edge or a vertex, or tiles sharing only a vertex.

        Returns
        -------
        TilingResult

        """
        ...

    def rotated(self, angle: float) -> Tiling:
        """A copy of this tiling whose lattice is rotated counter-clockwise by *angle* degrees.

        The rotation is about the center of the region that :meth:`generate`
        covers (the center of ``bounds`` expanded by ``margin``), not about
        the origin, and the tiles cover the region exactly whatever the angle.
        Tile transforms include the rotation. Rotations add up:
        ``tiling.rotated(a).rotated(b)`` is ``tiling.rotated(a + b)``.

        Parameters
        ----------
        angle : float
            Angle in degrees, counter-clockwise.

        Returns
        -------
        Tiling

        """
        raise NotImplementedError(f"{type(self).__name__} does not support rotated()")

    def with_offset(self, offset: tuple[float, float]) -> Tiling:
        """A copy of this tiling whose lattice is shifted by a fraction of its two translation vectors.

        A lattice is invariant under shifts by whole multiples of its two
        translation vectors, ``translation_u`` and ``translation_v`` of the
        placed tiling (isotile). The tiling that :meth:`generate` places is
        shifted by ``u * translation_u + v * translation_v`` for
        ``offset=(u, v)``. For the square tiling these are the two sides of a
        tile along x and y. For the hexagon tiling they are two of the vectors
        between neighboring hexagon centers, 60 degrees apart, and for the
        other tilings they are the translations of the tiling type. They turn
        with :meth:`rotated`. Without an offset the lattice is anchored at the
        center of the region.

        The offset is periodic: ``(u + 1, v)`` and ``(u, v + 1)`` give the
        same tiles as ``(u, v)``. The default ``(0, 0)`` leaves the tiling
        unchanged.

        Parameters
        ----------
        offset : tuple of float
            ``(u, v)``, finite numbers.

        Returns
        -------
        Tiling

        """
        raise NotImplementedError(f"{type(self).__name__} does not support with_offset()")

    @property
    def _n_base_vertices(self) -> int | None:
        """Topological prototile vertex count, or *None* to auto-detect."""
        return None

    def _resolve_generate_args(
        self,
        bounds: tuple[float, float, float, float] | None,
        n_tiles: int | None,
        tile_size: float | None,
    ) -> tuple[tuple[float, float, float, float], int | None, float | None]:
        """Apply defaults for *bounds* and *n_tiles*/*tile_size*."""
        if bounds is None:
            bounds = DEFAULT_BOUNDS
        if tile_size is not None and tile_size <= 0:
            # A non-positive tile size gives a lattice step of zero, which no
            # amount of stepping can walk across the bounds.
            raise ValueError(f"tile_size must be positive, got {tile_size}")
        if n_tiles is None and tile_size is None:
            n_tiles = DEFAULT_N_TILES
        return bounds, n_tiles, tile_size

    @deprecated_params(facecolor="face_color", edgecolor="edge_color")
    @styled
    def plot_tile(
        self,
        ax=None,
        *,
        facecolor: str | None = None,
        edgecolor: str | None = None,
        alpha: float = 0.7,
        show_vertices: bool = True,
        show_vertex_labels: bool = False,
        linewidth: float = 1.5,
    ):
        """Plot the canonical (proto)tile shape using matplotlib.

        No call to :meth:`generate` is needed — the tile is derived directly
        from the tiling's :attr:`canonical_tile` property.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created if *ax* is ``None``.
        facecolor : str, optional
            Fill color.  When ``None`` the color is chosen automatically by
            vertex count to match the panel UI palette
            (3 → pink, 4 → blue, 5 → yellow, 6 → green).
        edgecolor : str, optional
            Outline color. Default: the stroke color of the style, else ``"#1971c2"``.
        alpha : float
            Fill transparency.
        show_vertices : bool
            Draw a dot at each tile vertex.
        show_vertex_labels : bool
            Annotate each vertex with its 0-based index.
        linewidth : float
            Outline width in points.

        Returns
        -------
        PrototilePlotResult
            Named result with ``ax``, ``fig``, ``tile`` (the patch), and
            ``vertices`` (scatter artist, or ``None``).
        """
        return _plot_polygon_tile(
            self.canonical_tile,
            ax,
            facecolor,
            edgecolor,
            alpha,
            show_vertices,
            show_vertex_labels,
            linewidth,
            n_vertices=self._n_base_vertices,
        )

    @property
    @abstractmethod
    def canonical_tile(self) -> Polygon:
        """The tile shape at unit size, centered at origin."""
        ...

    @abstractmethod
    def canonical_symbol(self) -> Symbol:
        """Return the Symbol matching this tiling's tile shape.

        Returns
        -------
        Symbol
            A symbol instance that can render tiles from this tiling.

        """
        ...

    def tile_size_for_symbol_size(self, symbol_size: float, spacing: float = 0.0) -> float:
        """Compute tile_size so that a symbol of given size fills the tile.

        The symbol is defined in the unit square [-0.5, 0.5]² and scaled by
        ``2 * symbol_size``. This method computes the appropriate tile_size
        parameter for the tiling's generate() method so that the tile matches
        the scaled symbol.

        Parameters
        ----------
        symbol_size : float
            The native half-extent of the symbol (after area_factor conversion).
        spacing : float
            Additional spacing as a fraction of symbol size (default 0.0).

        Returns
        -------
        float
            The tile_size parameter to pass to generate().

        Notes
        -----
        Different tilings interpret tile_size differently:
        - SquareTiling: tile_size = side length
        - HexagonTiling: tile_size = circumradius (center to vertex)
        - TriangleTiling: tile_size = scale factor for unit-area tile

        """
        # Default implementation: assume tile_size = 2 * symbol_size
        # (correct for SquareTiling, wrong for others)
        return 2 * symbol_size * (1 + spacing)

    def symbol_size_for_tile_size(self, tile_size: float) -> float:
        """Return the native symbol size whose symbol exactly fills a tile.

        Inverse of :meth:`tile_size_for_symbol_size` with ``spacing=0``.

        Parameters
        ----------
        tile_size : float
            The ``tile_size`` of a generated tiling.

        Returns
        -------
        float
            The symbol size (as used for ``LayoutResult.base_size``) at which
            the canonical symbol coincides with a tile of this size.

        """
        return tile_size / self.tile_size_for_symbol_size(1.0)

    @classmethod
    @abstractmethod
    def from_polygon(cls, polygon: Polygon) -> Self:
        """Create a tiling from a user-supplied polygon."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plot_polygon_tile(
    polygon: Polygon,
    ax,
    face_color: str | None,
    edge_color: str | None,
    alpha: float,
    show_vertices: bool,
    show_vertex_labels: bool,
    linewidth: float,
    n_vertices: int | None = None,
):
    """Shared matplotlib renderer for a single tile polygon.

    Used by both :meth:`Tiling.plot_tile` and :meth:`TilingResult.plot_tile`.
    All matplotlib imports are lazy so the dependency remains optional.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    from .plot_results import PrototilePlotResult

    if edge_color is None:
        edge_color = resolve_style().channel_value("stroke", "#1971c2")

    _VERTEX_COLORS = {3: "#f5b7b1", 4: "#aed6f1", 5: "#f9e79f", 6: "#abebc6"}

    coords = list(polygon.exterior.coords[:-1])  # drop closing duplicate

    if face_color is None:
        # Use n_vertices when supplied (curved tiles have many interpolated
        # points in exterior.coords that don't reflect the topological count).
        n_verts = n_vertices if n_vertices is not None else len(coords)
        face_color = _VERTEX_COLORS.get(n_verts, "#d5d8dc")

    if ax is None:
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
    else:
        fig = ax.get_figure()

    patch = MplPolygon(
        coords,
        closed=True,
        facecolor=face_color,
        edgecolor=edge_color,
        linewidth=linewidth,
        alpha=alpha,
    )
    ax.add_patch(patch)

    vertices_artist = None
    if show_vertices or show_vertex_labels:
        xs, ys = zip(*coords, strict=False)
        if show_vertices:
            vertices_artist = ax.scatter(xs, ys, color=edge_color, s=30, zorder=5)
        if show_vertex_labels:
            for i, (x, y) in enumerate(coords):
                ax.annotate(
                    str(i),
                    (x, y),
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    xytext=(0, 6),
                    textcoords="offset points",
                )

    ax.autoscale_view()
    ax.axis("off")
    return PrototilePlotResult(ax=ax, fig=fig, tile=patch, vertices=vertices_artist)


# ---------------------------------------------------------------------------
# Tilings over isotile
# ---------------------------------------------------------------------------

#: Area of the regular hexagon with circumradius 1.
_HEXAGON_AREA = 3.0 * math.sqrt(3.0) / 2.0


def _orientation(transforms: NDArray[np.floating]) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """Rotation (degrees) and flip of (n, 2, 3) similarity maps, as stored in :class:`TileTransform`.

    A reflected map (negative determinant) is ``R(theta) * diag(-1, 1)``, so both arguments of ``atan2`` are negated
    to recover ``theta``.
    """
    m = transforms[:, :, :2]
    flipped = m[:, 0, 0] * m[:, 1, 1] - m[:, 0, 1] * m[:, 1, 0] < 0
    sign = np.where(flipped, -1.0, 1.0)
    return np.degrees(np.arctan2(sign * m[:, 1, 0], sign * m[:, 0, 0])), flipped


def _unit_tile(tiling: _isotile.IsohedralTiling, side_per_size: float) -> Polygon:
    """Prototile of ``tiling`` with its area centroid at the origin, scaled to ``tile_size`` 1."""
    outline = tiling.prototile(normalized=True)
    centroid = Polygon(outline).centroid
    return Polygon((outline - np.array([centroid.x, centroid.y])) * side_per_size)


def _tile_size_for_bounding_radius(unit_tile: Polygon, symbol_size: float, spacing: float) -> float:
    """``tile_size`` at which the farthest vertex of the tile is at distance ``symbol_size`` from its center."""
    radius = float(np.max(np.linalg.norm(np.array(unit_tile.exterior.coords), axis=1)))
    return (symbol_size / radius) * (1 + spacing)


def _finite_float(value: object, name: str) -> float:
    """``float(value)`` if it is a finite real number, else ``ValueError``."""
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number, got {value!r}") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return result


def _validated_offset(offset: object) -> tuple[float, float]:
    """``(u, v)`` as floats; ``ValueError`` unless it is a pair of finite numbers."""
    try:
        u, v = offset  # type: ignore[misc]
    except (TypeError, ValueError):
        raise ValueError(f"offset must be a pair (u, v) of finite numbers, got {offset!r}") from None
    return _finite_float(u, "offset[0]"), _finite_float(v, "offset[1]")


class _IsotileTiling(Tiling):
    """A :class:`Tiling` that is an ``isotile.IsohedralTiling`` with a size convention.

    The frame of the isotile tiling gives the orientation of the tiles; :meth:`generate` sets the scale and the
    position (``IsohedralTiling.fill_region``). ``tile_size`` is ``sqrt(tile area) / side_per_size``.
    """

    #: Lattice offset (``with_offset``), in multiples of the two translation vectors of the placed tiling.
    _offset: tuple[float, float] = (0.0, 0.0)
    #: Rotation in degrees added by ``rotated``. ``canonical_tile`` stays the unrotated tile, so the rotation is part of
    #: the tile transforms.
    _rotation: float = 0.0

    def __init__(
        self,
        tiling: _isotile.IsohedralTiling,
        side_per_size: float = 1.0,
        unit_tile: Polygon | None = None,
    ) -> None:
        self._iso = tiling
        self._side_per_size = side_per_size
        self._unit_tile = unit_tile if unit_tile is not None else _unit_tile(tiling, side_per_size)

    @property
    def canonical_tile(self) -> Polygon:
        """The tile at ``tile_size`` 1, centered at its area centroid."""
        return self._unit_tile

    def rotated(self, angle: float) -> Self:
        angle = _finite_float(angle, "angle")
        clone = copy.copy(self)
        clone._iso = self._iso.rotated(angle)
        clone._rotation = self._rotation + angle
        return clone

    def with_offset(self, offset: tuple[float, float]) -> Self:
        clone = copy.copy(self)
        clone._offset = _validated_offset(offset)
        return clone

    def generate(
        self,
        bounds: tuple[float, float, float, float] = DEFAULT_BOUNDS,
        n_tiles: int | None = None,
        tile_size: float | None = None,
        margin: float = 0.1,
        adjacency_type: AdjacencyKind = "edge",
    ) -> TilingResult:
        bounds, n_tiles, tile_size = self._resolve_generate_args(bounds, n_tiles, tile_size)
        if tile_size is None:
            tiles = self._iso.fill_region(bounds, n_tiles=n_tiles, margin=margin)
        else:
            tiles = self._iso.fill_region(bounds, tile_size=tile_size * self._side_per_size, margin=margin)
        if self._offset != (0.0, 0.0):
            # fill_region anchors the lattice at the center of the region; move it by the offset and cover the
            # same region again.
            placed = tiles.tiling
            frame = np.array(placed.frame, dtype=float)
            frame[:, 2] += self._offset[0] * np.asarray(placed.translation_u) + self._offset[1] * np.asarray(
                placed.translation_v
            )
            x0, y0, x1, y1 = (float(b) for b in bounds)
            dx, dy = (x1 - x0) * margin, (y1 - y0) * margin
            tiles = placed.with_frame(frame).cover((x0 - dx, y0 - dy, x1 + dx, y1 + dy))

        polygons = tiles.polygons()
        centers = shapely.get_coordinates(shapely.centroid(np.asarray(polygons, dtype=object)))
        orientation = tiles.transforms
        if self._rotation != 0.0:
            a = math.radians(self._rotation)
            turn = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            orientation = np.einsum("ij,njk->nik", turn, orientation)
        rotation, flipped = _orientation(orientation)
        transforms = [
            TileTransform(center=(float(c[0]), float(c[1])), rotation=float(r), flipped=bool(f))
            for c, r, f in zip(centers, rotation, flipped, strict=True)
        ]
        adjacency = tiles.adjacency(adjacency_type)
        vertex_adjacency = tiles.adjacency("vertex-only")
        adjacency.sort_indices()
        vertex_adjacency.sort_indices()

        # The scale of the placed tiling relative to this one gives the tile size; the area of the placed tiling would
        # lose precision far from the origin.
        linear = tiles.tiling.frame[:, :2]
        relative_scale = math.sqrt(abs(np.linalg.det(linear) / np.linalg.det(self._iso.frame[:, :2])))
        tile_size = relative_scale * math.sqrt(self._iso.area) / self._side_per_size
        canonical = scale(self._unit_tile, xfact=tile_size, yfact=tile_size, origin=(0, 0))
        return TilingResult(
            polygons=polygons,
            transforms=transforms,
            adjacency=adjacency,
            vertex_adjacency=vertex_adjacency,
            tile_size=tile_size,
            inscribed_radius=float(canonical.exterior.distance(Point(0.0, 0.0))),
            canonical_tile=canonical,
            n_base_vertices=self._n_base_vertices,
            tile_set=tiles,
        )


# ---------------------------------------------------------------------------
# SquareTiling
# ---------------------------------------------------------------------------


class SquareTiling(_IsotileTiling):
    """Regular square grid tiling.

    All tiles are axis-aligned squares of equal size. ``tile_size`` is the side length.
    """

    def __init__(self) -> None:
        super().__init__(_isotile.IsohedralTiling.square())

    def canonical_symbol(self) -> Symbol:
        """Return a SquareSymbol for this tiling."""
        from .symbols import SquareSymbol

        return SquareSymbol()

    @classmethod
    def from_polygon(cls, polygon: Polygon) -> SquareTiling:
        """Create from polygon (must be a square; ignored for SquareTiling)."""
        return cls()


# ---------------------------------------------------------------------------
# HexagonTiling
# ---------------------------------------------------------------------------


class HexagonTiling(_IsotileTiling):
    """Regular hexagonal grid tiling.

    ``tile_size`` is the circumradius (center to vertex).

    Parameters
    ----------
    pointy_top : bool
        If True (default), hexagons have a vertex at the top; otherwise they have a flat top (a vertex at angle 0).

    """

    def __init__(self, pointy_top: bool = True) -> None:
        self.pointy_top = pointy_top
        super().__init__(_isotile.IsohedralTiling.hexagon(pointy_top), side_per_size=math.sqrt(_HEXAGON_AREA))

    def canonical_symbol(self) -> Symbol:
        """Return a HexagonSymbol matching this tiling's orientation."""
        from .symbols import HexagonSymbol

        return HexagonSymbol(pointy_top=self.pointy_top)

    def tile_size_for_symbol_size(self, symbol_size: float, spacing: float = 0.0) -> float:
        """Compute tile_size for hexagon tiling.

        For HexagonTiling, tile_size is the circumradius (center to vertex).
        HexagonSymbol has bounding_radius = 0.5, so after scaling by 2*size,
        the circumradius equals symbol_size.

        Therefore: tile_size = symbol_size * (1 + spacing)
        """
        return symbol_size * (1 + spacing)

    @classmethod
    def from_polygon(cls, polygon: Polygon) -> HexagonTiling:
        """Create from polygon (must be a hexagon; ignored for HexagonTiling)."""
        return cls()


# ---------------------------------------------------------------------------
# TriangleTiling
# ---------------------------------------------------------------------------


def _normalized_polygon(tile: Polygon, n_vertices: int, name: str) -> Polygon:
    """``tile`` with its centroid at the origin, scaled to unit area."""
    if len(tile.exterior.coords) != n_vertices + 1:  # the closing point repeats the first
        raise ValueError(f"{name} must have {n_vertices} vertices, got {len(tile.exterior.coords) - 1}")
    centroid = tile.centroid
    shifted = translate(tile, -centroid.x, -centroid.y)
    area = abs(shifted.area)
    if area < 1e-12:
        raise ValueError(f"{name} has zero area")
    factor = 1.0 / np.sqrt(area)
    return scale(shifted, xfact=factor, yfact=factor, origin=(0, 0))


class TriangleTiling(_IsotileTiling):
    """Tiling by congruent triangles.

    Any triangle tiles the Euclidean plane. The tiling is constructed by
    pairing triangles via 180° rotation about an edge midpoint to form
    parallelograms, which then tile by translation. ``tile_size`` is the
    square root of the tile area.

    Parameters
    ----------
    tile : Polygon
        A triangle (3-vertex polygon). Will be normalized to unit area
        internally. The shape (angles/proportions) is preserved.

    """

    def __init__(self, tile: Polygon) -> None:
        unit_tile = _normalized_polygon(tile, 3, "Triangle")
        super().__init__(
            _isotile.IsohedralTiling.triangle(np.array(unit_tile.exterior.coords[:-1])), unit_tile=unit_tile
        )

    def canonical_symbol(self) -> Symbol:
        """Return a TileSymbol wrapping this triangle's canonical tile."""
        from .symbols import TileSymbol

        return TileSymbol(self.canonical_tile)

    def tile_size_for_symbol_size(self, symbol_size: float, spacing: float = 0.0) -> float:
        """Compute tile_size for triangle tiling.

        TriangleTiling uses unit-area tiles. TileSymbol normalizes to
        bounding_radius = 0.5, so we need to compute the original tile's
        bounding radius to find the correct scale factor.
        """
        return _tile_size_for_bounding_radius(self._unit_tile, symbol_size, spacing)

    @classmethod
    def from_polygon(cls, polygon: Polygon) -> TriangleTiling:
        """Create a tiling from a user-supplied triangle."""
        return cls(polygon)

    # -- Convenience constructors --

    @classmethod
    def equilateral(cls) -> TriangleTiling:
        """Equilateral triangle tiling."""
        s = 1.0
        h = s * np.sqrt(3) / 2
        return cls(Polygon([(0, 0), (s, 0), (s / 2, h)]))

    @classmethod
    def right_isosceles(cls) -> TriangleTiling:
        """Right isosceles triangle tiling (45-45-90)."""
        return cls(Polygon([(0, 0), (1, 0), (0, 1)]))

    @classmethod
    def right(cls, aspect_ratio: float = 1.0) -> TriangleTiling:
        """Right triangle with given aspect ratio (height/base).

        Parameters
        ----------
        aspect_ratio : float
            Ratio of height to base. Default 1.0 gives a right isosceles.

        """
        return cls(Polygon([(0, 0), (1, 0), (0, aspect_ratio)]))

    @classmethod
    def isosceles(cls, apex_angle: float = 60.0) -> TriangleTiling:
        """Isosceles triangle with given apex angle (degrees).

        Parameters
        ----------
        apex_angle : float
            Angle at the apex in degrees. Default 60.0 gives equilateral.

        """
        half_angle = np.radians(apex_angle / 2)
        base = 2 * np.sin(half_angle)
        height = np.cos(half_angle)
        return cls(Polygon([(-base / 2, 0), (base / 2, 0), (0, height)]))


# ---------------------------------------------------------------------------
# QuadrilateralTiling
# ---------------------------------------------------------------------------


class QuadrilateralTiling(_IsotileTiling):
    """Tiling by congruent quadrilaterals.

    Any simple quadrilateral tiles the Euclidean plane via 180° rotation
    about edge midpoints. ``tile_size`` is the square root of the tile area.

    Parameters
    ----------
    tile : Polygon
        A simple quadrilateral (4-vertex polygon), convex or not. Will be
        normalized to unit area internally. The shape (angles/proportions)
        is preserved.

    Raises
    ------
    ValueError
        If the polygon does not have 4 vertices, has zero area or is
        self-intersecting.

    """

    def __init__(self, tile: Polygon) -> None:
        unit_tile = _normalized_polygon(tile, 4, "Quadrilateral")
        super().__init__(
            _isotile.IsohedralTiling.quadrilateral(np.array(unit_tile.exterior.coords[:-1])), unit_tile=unit_tile
        )

    def canonical_symbol(self) -> Symbol:
        """Return a TileSymbol wrapping this quadrilateral's canonical tile."""
        from .symbols import TileSymbol

        return TileSymbol(self.canonical_tile)

    def tile_size_for_symbol_size(self, symbol_size: float, spacing: float = 0.0) -> float:
        """Compute tile_size for quadrilateral tiling.

        QuadrilateralTiling uses unit-area tiles. TileSymbol normalizes to
        bounding_radius = 0.5, so we need to compute the original tile's
        bounding radius to find the correct scale factor.
        """
        return _tile_size_for_bounding_radius(self._unit_tile, symbol_size, spacing)

    @classmethod
    def from_polygon(cls, polygon: Polygon) -> QuadrilateralTiling:
        """Create a tiling from a user-supplied quadrilateral."""
        return cls(polygon)

    # -- Convenience constructors --

    @classmethod
    def parallelogram(cls, angle: float = 60.0, aspect_ratio: float = 1.0) -> QuadrilateralTiling:
        """Parallelogram with given angle and aspect ratio.

        Parameters
        ----------
        angle : float
            Interior angle in degrees (0 < angle < 180).
        aspect_ratio : float
            Ratio of side b to side a.

        """
        a = 1.0
        b = aspect_ratio
        rad = np.radians(angle)
        return cls(
            Polygon([
                (0, 0),
                (a, 0),
                (a + b * np.cos(rad), b * np.sin(rad)),
                (b * np.cos(rad), b * np.sin(rad)),
            ]),
        )

    @classmethod
    def rectangle(cls, aspect_ratio: float = 1.0) -> QuadrilateralTiling:
        """Rectangle with given aspect ratio (height/width).

        Parameters
        ----------
        aspect_ratio : float
            Ratio of height to width. Default 1.0 gives a square.

        """
        return cls(Polygon([(0, 0), (1, 0), (1, aspect_ratio), (0, aspect_ratio)]))

    @classmethod
    def rhombus(cls, angle: float = 60.0) -> QuadrilateralTiling:
        """Rhombus (all sides equal) with given angle.

        Parameters
        ----------
        angle : float
            Interior angle in degrees.

        """
        return cls.parallelogram(angle=angle, aspect_ratio=1.0)

    @classmethod
    def trapezoid(cls, top_ratio: float = 0.5, height: float = 1.0) -> QuadrilateralTiling:
        """Isosceles trapezoid.

        Parameters
        ----------
        top_ratio : float
            Ratio of top edge to bottom edge. Must be < 1.
        height : float
            Height of the trapezoid.

        """
        bottom = 1.0
        top = bottom * top_ratio
        offset = (bottom - top) / 2
        return cls(Polygon([(0, 0), (bottom, 0), (bottom - offset, height), (offset, height)]))


# ---------------------------------------------------------------------------
# IsohedralTiling
# ---------------------------------------------------------------------------


def _type_number(self_or_type: object, tiling_type: int | None, method: str) -> int:
    """The type number of ``method`` called on an instance or on the class."""
    if isinstance(self_or_type, IsohedralTiling):
        return int(self_or_type._iso.type.number)
    if isinstance(self_or_type, int):
        return self_or_type
    if tiling_type is None:
        raise TypeError(
            f"{method}() requires a tiling type number when called on the class, e.g. IsohedralTiling.{method}(83)",
        )
    return tiling_type


class IsohedralTiling(_IsotileTiling):
    """Isohedral tiling of one of the 81 types of the Grünbaum-Shephard classification.

    An :class:`isotile.IsohedralTiling` as a :class:`Tiling`; ``tile_size`` is the square root of the tile area. Each
    type has 0-6 free parameters controlling tile vertex positions, and supports optional custom edge curves for
    Escher-style tiles. Type metadata, presets and serialization are those of isotile; the methods of this class
    that provide them call isotile.

    Parameters
    ----------
    tiling_type : int
        Isohedral type number (IH1-IH93, with some gaps). Use
        ``IsohedralTiling.available_types()`` for the full list.
    parameters : list[float], optional
        Free parameters for the tiling type. If ``None``, uses defaults.
    edge_curves : dict[int, list[tuple[float, float]]], optional
        Custom edge curves keyed by edge shape ID. Each curve is a list of
        ``(x, y)`` points in canonical space from ``(0, 0)`` to ``(1, 0)``.
        Symmetry constraints (S, U) are enforced automatically. Edge shape
        IDs and types can be inspected via ``type_info()``. If ``None``, all
        edges are straight lines.
    edge_curve_modes : dict[int, str], optional
        ``"spline"`` (default) or ``"linear"`` per shape ID in *edge_curves*.

    Raises
    ------
    ValueError
        If the type does not exist, the parameters are invalid, a curve is given for a straight (``I``) edge or is
        malformed, or the prototile with its curves is not a simple polygon.

    """

    def __init__(
        self,
        tiling_type: int,
        parameters: list[float] | None = None,
        edge_curves: dict[int, list[tuple[float, float]]] | None = None,
        edge_curve_modes: dict[int, str] | None = None,
    ) -> None:
        super().__init__(
            _isotile.IsohedralTiling(
                tiling_type, parameters, edge_curves=edge_curves, edge_curve_modes=edge_curve_modes
            )
        )

    @classmethod
    def _wrap(cls, tiling: _isotile.IsohedralTiling) -> IsohedralTiling:
        new = cls.__new__(cls)
        _IsotileTiling.__init__(new, tiling)
        return new

    @property
    def isotile(self) -> _isotile.IsohedralTiling:
        """The underlying ``isotile.IsohedralTiling``."""
        return self._iso

    @property
    def _n_base_vertices(self) -> int:
        """Topological vertex count of the prototile (ignores curve interpolation points)."""
        return int(self._iso.type.n_vertices)

    def canonical_symbol(self) -> Symbol:
        """Return an IsohedralTileSymbol with this tiling's parameters and curves."""
        from .symbols import IsohedralTileSymbol

        curves = {k: [(float(x), float(y)) for x, y in v] for k, v in self._iso.edge_curves.items()}
        return IsohedralTileSymbol(
            tiling_type=int(self._iso.type.number),
            prototile_params=[float(p) for p in self._iso.parameters],
            edge_curves=curves or None,
            edge_curve_modes=dict(self._iso.edge_curve_modes) or None,
        )

    def tile_size_for_symbol_size(self, symbol_size: float, spacing: float = 0.0) -> float:
        """Compute tile_size for isohedral tiling.

        IsohedralTiling uses unit-area tiles. TileSymbol normalizes to
        bounding_radius = 0.5, so we need to compute the original tile's
        bounding radius to find the correct scale factor.
        """
        return _tile_size_for_bounding_radius(self._unit_tile, symbol_size, spacing)

    @classmethod
    def from_polygon(cls, polygon: Polygon) -> IsohedralTiling:
        """Not supported — use ``IsohedralTiling(tiling_type, parameters)``."""
        raise NotImplementedError(
            "IsohedralTiling is defined by type number and parameters, "
            "not by polygon shape. Use IsohedralTiling(tiling_type=...) instead.",
        )

    # -- Type metadata, presets, serialization: those of isotile -----------------------------------------------

    @staticmethod
    def available_types() -> list[int]:
        """Return list of valid isohedral tiling type numbers (``isotile.tiling_types``)."""
        return list(_isotile.tiling_types())

    @staticmethod
    def find_types(
        *,
        num_vertices: int | None = None,
        num_parameters: int | None = None,
        max_parameters: int | None = None,
        num_aspects: int | None = None,
        has_reflections: bool | None = None,
        edge_types: set[str] | None = None,
        all_edges: str | None = None,
        has_customizable_edges: bool | None = None,
    ) -> list[int]:
        """Find isohedral tiling types matching the given criteria (``isotile.find_types``).

        All arguments are optional filters. Only types satisfying every
        specified filter are returned.

        Returns
        -------
        list[int]
            Matching isohedral type numbers, sorted.

        Examples
        --------
        >>> IsohedralTiling.find_types(num_vertices=4, all_edges="J")
        [31, 33, 34, 36, 41, 43, 44, 52, 55, 59, 61, 68, 71]
        >>> IsohedralTiling.find_types(num_vertices=6, max_parameters=0)
        [10, 11, 18, 20]

        """
        return _isotile.find_types(
            num_vertices=num_vertices,
            num_parameters=num_parameters,
            max_parameters=max_parameters,
            num_aspects=num_aspects,
            has_reflections=has_reflections,
            edge_types=edge_types,
            all_edges=all_edges,
            has_customizable_edges=has_customizable_edges,
        )

    @classmethod
    def from_preset(cls, name: str, **kwargs) -> IsohedralTiling:
        """Create an IsohedralTiling from a named preset (``isotile.IsohedralTiling.from_preset``).

        Parameters
        ----------
        name : str
            Preset name.  Use ``IsohedralTiling.list_presets()`` for options.
        **kwargs
            Overrides of the preset: ``parameters``, ``edge_curves``, ``edge_curve_modes``.

        Raises
        ------
        ValueError
            If the preset does not exist.

        """
        return cls._wrap(_isotile.IsohedralTiling.from_preset(name, **kwargs))

    @classmethod
    def list_presets(cls) -> dict[str, str]:
        """Return a dict mapping preset names to their descriptions (``isotile.list_presets``)."""
        return _isotile.list_presets()

    def to_dict(self) -> dict:
        """Serialize the tiling to a JSON-compatible dict (``isotile.IsohedralTiling.to_dict``)."""
        return self._iso.to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> IsohedralTiling:
        """Load a tiling from a dict as produced by :meth:`to_dict` (``isotile.IsohedralTiling.from_dict``)."""
        return cls._wrap(_isotile.IsohedralTiling.from_dict(data))

    def save(self, path: str | Path) -> None:
        """Save the tiling to a JSON file (``isotile.IsohedralTiling.save``)."""
        self._iso.save(path)

    @classmethod
    def load(cls, path: str | Path) -> IsohedralTiling:
        """Load a tiling from a JSON file written by :meth:`save` (``isotile.IsohedralTiling.load``)."""
        return cls._wrap(_isotile.IsohedralTiling.load(path))

    def type_info(self_or_type: IsohedralTiling | int | None = None, tiling_type: int | None = None) -> dict:
        """Return metadata for an isohedral tiling type (``isotile.TilingType.info``).

        Can be called on an instance (``t.type_info()``) or on the class
        (``IsohedralTiling.type_info(83)``).

        Parameters
        ----------
        tiling_type : int, optional
            Isohedral type number.  Not needed when called on an instance.

        """
        return _isotile.get_type(_type_number(self_or_type, tiling_type, "type_info")).info()

    def describe(self_or_type: IsohedralTiling | int | None = None, tiling_type: int | None = None) -> str:
        """Return a human-readable summary of an isohedral tiling type (``isotile.TilingType.describe``).

        Can be called on an instance (``t.describe()``) or on the class
        (``IsohedralTiling.describe(83)``).

        Parameters
        ----------
        tiling_type : int, optional
            Isohedral type number.  Not needed when called on an instance.

        """
        return _isotile.get_type(_type_number(self_or_type, tiling_type, "describe")).describe()

    @styled
    def plot_prototile(self, ax=None, show_labels: bool = True):
        """Plot the prototile with labeled edges (``isotile.plot.plot_prototile``).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.  If ``None``, a new figure is created.
        show_labels : bool
            Whether to annotate edges with their ID and type.

        Returns
        -------
        matplotlib.axes.Axes

        """
        from isotile.plot import plot_prototile

        return plot_prototile(self._iso, ax=ax, show_labels=show_labels)


# ---------------------------------------------------------------------------
# Registry: string → Tiling
# ---------------------------------------------------------------------------

_TILING_REGISTRY: dict[str, type[Tiling] | Tiling] = {
    "square": SquareTiling,
    "hexagon": HexagonTiling,
    "triangle": TriangleTiling,
    "quadrilateral": QuadrilateralTiling,
    "isohedral": IsohedralTiling,
}


def resolve_tiling(spec: str | Tiling) -> Tiling:
    """Resolve a tiling specification to a Tiling instance.

    Parameters
    ----------
    spec : str or Tiling
        A string shorthand (``"square"``, ``"hexagon"``, ``"triangle"``,
        ``"quadrilateral"``, ``"isohedral"``, ``"ih<N>"``), or a ``Tiling``
        instance.

    Returns
    -------
    Tiling

    """
    if isinstance(spec, Tiling):
        return spec
    if isinstance(spec, str):
        # Handle "ih<N>" shorthand (e.g. "ih4", "ih28")
        ih_match = re.fullmatch(r"ih(\d+)", spec, re.IGNORECASE)
        if ih_match:
            return IsohedralTiling(tiling_type=int(ih_match.group(1)))

        # Check built-in tiling registry first (square, hexagon, etc.)
        entry = _TILING_REGISTRY.get(spec)
        if entry is None:
            # Check isohedral preset names (e.g. "scalloped_hexagon")
            if spec in IsohedralTiling.list_presets():
                return IsohedralTiling.from_preset(spec)
            valid = ", ".join(sorted(_TILING_REGISTRY.keys()))
            raise ValueError(f"Unknown tiling: {spec!r}. Valid tilings: {valid} (or 'ih<N>' for isohedral type N)")
        if isinstance(entry, type):
            # For triangle/quad, use equilateral/square defaults
            if entry is TriangleTiling:
                return TriangleTiling.equilateral()
            if entry is QuadrilateralTiling:
                return QuadrilateralTiling.parallelogram()
            if entry is IsohedralTiling:
                return IsohedralTiling(tiling_type=1)
            return entry()
        return entry  # Already an instance
    raise TypeError(f"tiling must be a str or Tiling instance, got {type(spec).__name__}")
