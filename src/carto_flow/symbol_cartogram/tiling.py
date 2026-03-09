# ruff: noqa: RUF002
"""Tiling abstractions for tile-based symbol cartograms.

This module defines the ``Tiling`` ABC and concrete tiling classes that generate
congruent tilings of the plane. Each tiling produces a ``TilingResult`` containing
tile polygons, per-tile transforms, and a precomputed adjacency matrix.

Tiling Classes
--------------
SquareTiling
    Regular square grid.
HexagonTiling
    Regular hexagonal grid (pointy-top).
TriangleTiling
    Tiling by congruent triangles (any triangle tiles the plane).
QuadrilateralTiling
    Tiling by congruent quadrilaterals (any simple quadrilateral tiles the plane).
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from shapely.affinity import rotate, scale, translate
from shapely.geometry import Point, Polygon, box

if TYPE_CHECKING:
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
# Enums
# ---------------------------------------------------------------------------


class TileAdjacencyType(str, Enum):
    """How tile adjacency is determined."""

    EDGE = "edge"  # Tiles sharing a full edge
    VERTEX = "vertex"  # Tiles sharing an edge or a vertex


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
    adjacency : NDArray[np.bool_]
        (m, m) boolean edge adjacency matrix between tiles.
    vertex_adjacency : NDArray[np.bool_] | None
        (m, m) boolean vertex-only adjacency matrix (tiles sharing exactly
        one vertex but no edge).  *None* for tilings that don't compute it.
    tile_size : float
        Size parameter used for generation.
    inscribed_radius : float
        Radius of the largest circle fitting inside the canonical tile
        at the generated size.
    canonical_tile : Polygon
        The unit tile shape (before transforms), at the generated size.
    n_base_vertices : int or None
        Topological vertex count of the prototile.  For most tilings this
        equals ``len(canonical_tile.exterior.coords) - 1``, but for
        isohedral tilings with custom edge curves the ``canonical_tile``
        polygon has many more points due to spline interpolation.  Used by
        :meth:`plot_tile` to select the correct auto-colour.

    """

    polygons: list[Polygon]
    transforms: list[TileTransform]
    adjacency: NDArray[np.bool_]
    vertex_adjacency: NDArray[np.bool_] | None
    tile_size: float
    inscribed_radius: float
    canonical_tile: Polygon
    n_base_vertices: int | None = None

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
        origin and all tile rotations incremented by the given angle.

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

        """
        cos_a = np.cos(np.radians(angle))
        sin_a = np.sin(np.radians(angle))

        # Rotate each tile's center and add to its rotation
        new_transforms = []
        new_polygons = []
        for t, poly in zip(self.transforms, self.polygons):
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

    def plot(
        self,
        ax=None,
        *,
        color_by: str = "uniform",
        face_color: str = "#4e9af1",
        edge_color: str = "#333333",
        colormap: str = "tab10",
        alpha: float = 0.85,
        linewidth: float = 0.5,
    ):
        """Plot the tiling using matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created if *ax* is ``None``.
        color_by : {"uniform", "aspect", "index", "x", "y", "random"}
            Colouring scheme:

            * ``"uniform"`` — all tiles use *face_color*.
            * ``"aspect"`` — colour by tile rotation angle (0–360 → colormap).
            * ``"index"`` — colour by tile index (0…n−1 → colormap).
            * ``"x"`` — colour by tile centroid x coordinate (normalised).
            * ``"y"`` — colour by tile centroid y coordinate (normalised).
            * ``"random"`` — random colour per tile drawn from *colormap*.
        face_color : str
            Hex fill colour used in ``"uniform"`` mode.
        edge_color : str
            Stroke colour for tile outlines.
        colormap : str
            Matplotlib colormap name used in ``"aspect"`` and ``"index"`` modes.
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

        _CMAP_MODES = {"aspect", "index", "x", "y", "random"}
        if color_by not in _CMAP_MODES | {"uniform"}:
            raise ValueError(
                f"Unknown color_by={color_by!r}. Choose from 'uniform', 'aspect', 'index', 'x', 'y', 'random'."
            )

        n = len(self.polygons)
        cmap = plt.get_cmap(colormap) if color_by in _CMAP_MODES else None

        coords = [list(zip(*p.exterior.xy)) for p in self.polygons]
        if color_by == "uniform":
            colors = [face_color] * n
        elif color_by == "aspect":
            colors = [mcolors.to_hex(cmap((tr.rotation % 360) / 360.0)) for tr in self.transforms]
        elif color_by == "index":
            colors = [mcolors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]
        elif color_by == "x":
            xs = self.centers[:, 0]
            lo, hi = xs.min(), xs.max()
            values = (xs - lo) / max(hi - lo, 1e-9)
            colors = [mcolors.to_hex(cmap(v)) for v in values]
        elif color_by == "y":
            ys = self.centers[:, 1]
            lo, hi = ys.min(), ys.max()
            values = (ys - lo) / max(hi - lo, 1e-9)
            colors = [mcolors.to_hex(cmap(v)) for v in values]
        else:  # random
            colors = [mcolors.to_hex(cmap(v)) for v in np.random.default_rng().random(n)]

        col = PolyCollection(
            coords,
            facecolors=colors,
            edgecolors=edge_color,
            linewidths=linewidth,
            alpha=alpha,
        )
        ax.add_collection(col)
        ax.autoscale_view()
        ax.axis("off")
        return TilingGridPlotResult(ax=ax, fig=fig, tiles=col)

    def plot_tile(
        self,
        ax=None,
        *,
        face_color: str | None = None,
        edge_color: str = "#1971c2",
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
        face_color : str, optional
            Fill colour.  When ``None`` the colour is chosen automatically by
            vertex count to match the panel UI palette
            (3 → pink, 4 → blue, 5 → yellow, 6 → green).
        edge_color : str
            Outline colour.
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
            face_color,
            edge_color,
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
        adjacency_type: TileAdjacencyType = TileAdjacencyType.EDGE,
    ) -> TilingResult:
        """Generate tiles covering the bounding box.

        Parameters
        ----------
        bounds : tuple, optional
            Bounding box as (minx, miny, maxx, maxy).  Defaults to the unit
            box ``(-1, -1, 1, 1)`` (see :data:`DEFAULT_BOUNDS`).
        n_tiles : int, optional
            Approximate number of tiles. Mutually exclusive with *tile_size*.
            When neither is given, defaults to :data:`DEFAULT_N_TILES`.
        tile_size : float, optional
            Size of each tile. Mutually exclusive with *n_tiles*.
        margin : float
            Fractional margin around bounding box.
        adjacency_type : TileAdjacencyType
            Whether adjacency requires shared edges or just shared vertices.

        Returns
        -------
        TilingResult

        """
        ...

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
        if n_tiles is None and tile_size is None:
            n_tiles = DEFAULT_N_TILES
        return bounds, n_tiles, tile_size

    def plot_tile(
        self,
        ax=None,
        *,
        face_color: str | None = None,
        edge_color: str = "#1971c2",
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
        face_color : str, optional
            Fill colour.  When ``None`` the colour is chosen automatically by
            vertex count to match the panel UI palette
            (3 → pink, 4 → blue, 5 → yellow, 6 → green).
        edge_color : str
            Outline colour.
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
            face_color,
            edge_color,
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
    edge_color: str,
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
        xs, ys = zip(*coords)
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


def _apply_margin(bounds: tuple[float, float, float, float], margin: float) -> tuple[float, float, float, float]:
    """Expand bounds by a fractional margin."""
    minx, miny, maxx, maxy = bounds
    w = maxx - minx
    h = maxy - miny
    return (minx - w * margin, miny - h * margin, maxx + w * margin, maxy + h * margin)


def _compute_adjacency_from_lattice(
    centers: NDArray[np.floating],
    neighbor_offsets: list[tuple[float, float]],
    tol: float,
) -> NDArray[np.bool_]:
    """Compute adjacency matrix from lattice offsets.

    For each tile center, check which other centers match the expected
    neighbor offsets (within tolerance).

    Parameters
    ----------
    centers : (m, 2) array
        Tile center positions.
    neighbor_offsets : list of (dx, dy)
        Expected offsets to neighbors from any given tile.
    tol : float
        Distance tolerance for matching.

    Returns
    -------
    (m, m) boolean adjacency matrix.

    """
    m = len(centers)
    adj = np.zeros((m, m), dtype=bool)
    for dx, dy in neighbor_offsets:
        # For each tile, find tiles at offset (dx, dy)
        shifted = centers + np.array([dx, dy])
        # Compute distances from shifted positions to all centers
        for i in range(m):
            dists = np.sum((centers - shifted[i]) ** 2, axis=1)
            matches = np.where(dists < tol**2)[0]
            for j in matches:
                if i != j:
                    adj[i, j] = True
                    adj[j, i] = True
    return adj


def _compute_adjacency_matrices(
    polygons: list[Polygon],
    tol: float = 1e-6,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Compute edge and vertex-only adjacency by counting shared vertices.

    Two tiles share an edge iff they have **>= 2 vertices** within *tol*.
    Two tiles share only a vertex iff they have **exactly 1** shared vertex.

    Parameters
    ----------
    polygons : list of Polygon
        Tile polygons.
    tol : float
        Distance tolerance for vertex matching.

    Returns
    -------
    edge_adj : (m, m) boolean adjacency matrix (shared edge).
    vertex_adj : (m, m) boolean adjacency matrix (shared vertex only, no edge).

    """
    from shapely import STRtree

    m = len(polygons)
    edge_adj = np.zeros((m, m), dtype=bool)
    vertex_adj = np.zeros((m, m), dtype=bool)

    # Extract vertex arrays (excluding closing duplicate) for each polygon
    coords = [np.array(p.exterior.coords[:-1]) for p in polygons]

    # Use spatial index for candidate filtering
    tree = STRtree(polygons)
    tol_sq = tol * tol

    for i, poly_i in enumerate(polygons):
        candidates = tree.query(poly_i.buffer(tol))
        vi = coords[i]
        for j in candidates:
            if j <= i:
                continue
            vj = coords[j]
            diff = vi[:, np.newaxis, :] - vj[np.newaxis, :, :]
            sq_dists = (diff * diff).sum(axis=2)
            n_shared = int((sq_dists.min(axis=1) <= tol_sq).sum())
            if n_shared >= 2:
                edge_adj[i, j] = True
                edge_adj[j, i] = True
            elif n_shared == 1:
                vertex_adj[i, j] = True
                vertex_adj[j, i] = True

    return edge_adj, vertex_adj


def _compute_adjacency_geometric(
    polygons: list[Polygon],
    adjacency_type: TileAdjacencyType,
    tol: float = 1e-6,
) -> NDArray[np.bool_]:
    """Compute adjacency from geometric relationships (slower, but general).

    Parameters
    ----------
    polygons : list of Polygon
        Tile polygons.
    adjacency_type : TileAdjacencyType
        EDGE (shared edge) or VERTEX (shared edge or vertex).
    tol : float
        Buffer tolerance for intersection detection.

    Returns
    -------
    (m, m) boolean adjacency matrix.

    """
    edge_adj, vertex_adj = _compute_adjacency_matrices(polygons, tol=tol)
    if adjacency_type == TileAdjacencyType.EDGE:
        return edge_adj
    # VERTEX: edge OR vertex-only
    return edge_adj | vertex_adj


# ---------------------------------------------------------------------------
# SquareTiling
# ---------------------------------------------------------------------------


class SquareTiling(Tiling):
    """Regular square grid tiling.

    All tiles are axis-aligned squares of equal size.
    """

    @property
    def canonical_tile(self) -> Polygon:
        """Unit square centered at origin."""
        return box(-0.5, -0.5, 0.5, 0.5)

    def canonical_symbol(self) -> Symbol:
        """Return a SquareSymbol for this tiling."""
        from .symbols import SquareSymbol

        return SquareSymbol()

    @classmethod
    def from_polygon(cls, polygon: Polygon) -> SquareTiling:
        """Create from polygon (must be a square; ignored for SquareTiling)."""
        return cls()

    def generate(
        self,
        bounds: tuple[float, float, float, float] = DEFAULT_BOUNDS,
        n_tiles: int | None = None,
        tile_size: float | None = None,
        margin: float = 0.1,
        adjacency_type: TileAdjacencyType = TileAdjacencyType.EDGE,
    ) -> TilingResult:
        bounds, n_tiles, tile_size = self._resolve_generate_args(bounds, n_tiles, tile_size)

        minx, miny, maxx, maxy = _apply_margin(bounds, margin)
        width = maxx - minx
        height = maxy - miny

        if tile_size is None:
            tile_size = np.sqrt(width * height / n_tiles)

        s = tile_size

        # Generate tile centers on a regular grid
        xs = np.arange(minx + s / 2, maxx + s / 2, s)
        ys = np.arange(miny + s / 2, maxy + s / 2, s)
        grid_x, grid_y = np.meshgrid(xs, ys)
        centers = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        # Create polygons and transforms
        base = box(-s / 2, -s / 2, s / 2, s / 2)
        polygons = []
        transforms = []
        for cx, cy in centers:
            polygons.append(translate(base, cx, cy))
            transforms.append(TileTransform(center=(cx, cy)))

        # Compute adjacency from lattice offsets
        edge_offsets = [(s, 0), (-s, 0), (0, s), (0, -s)]
        diag_offsets = [(s, s), (s, -s), (-s, s), (-s, -s)]
        if adjacency_type == TileAdjacencyType.EDGE:
            adj = _compute_adjacency_from_lattice(centers, edge_offsets, tol=s * 0.1)
        else:
            adj = _compute_adjacency_from_lattice(centers, edge_offsets + diag_offsets, tol=s * 0.1)
        vert_adj = _compute_adjacency_from_lattice(centers, diag_offsets, tol=s * 0.1)

        inscribed_r = s / 2  # half-side
        canonical = box(-s / 2, -s / 2, s / 2, s / 2)

        return TilingResult(
            polygons=polygons,
            transforms=transforms,
            adjacency=adj,
            vertex_adjacency=vert_adj,
            tile_size=s,
            inscribed_radius=inscribed_r,
            canonical_tile=canonical,
        )


# ---------------------------------------------------------------------------
# HexagonTiling
# ---------------------------------------------------------------------------


def _create_hexagon(cx: float, cy: float, size: float) -> Polygon:
    """Create a pointy-top hexagon centered at (cx, cy) with given size."""
    base_angle = np.pi / 6  # 30 deg for pointy-top
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + base_angle
    x = cx + size * np.cos(angles)
    y = cy + size * np.sin(angles)
    return Polygon(zip(x, y, strict=True))


class HexagonTiling(Tiling):
    """Regular hexagonal grid tiling (pointy-top orientation).

    Parameters
    ----------
    pointy_top : bool
        If True (default), hexagons have a vertex at top.

    """

    def __init__(self, pointy_top: bool = True) -> None:
        self.pointy_top = pointy_top

    @property
    def canonical_tile(self) -> Polygon:
        """Unit hexagon centered at origin."""
        return _create_hexagon(0, 0, 1.0)

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

    def generate(
        self,
        bounds: tuple[float, float, float, float] = DEFAULT_BOUNDS,
        n_tiles: int | None = None,
        tile_size: float | None = None,
        margin: float = 0.1,
        adjacency_type: TileAdjacencyType = TileAdjacencyType.EDGE,
    ) -> TilingResult:
        bounds, n_tiles, tile_size = self._resolve_generate_args(bounds, n_tiles, tile_size)

        minx, miny, maxx, maxy = _apply_margin(bounds, margin)
        width = maxx - minx
        height = maxy - miny

        if tile_size is None:
            # Hexagon area = (3*sqrt(3)/2) * s^2 ≈ 2.598 * s^2
            tile_size = np.sqrt(width * height / n_tiles / 2.598)

        size = tile_size

        # Hexagon dimensions (pointy-top)
        hex_width = size * np.sqrt(3)  # edge to edge (horizontal)
        hex_height = size * 2  # vertex to vertex (vertical)

        # Spacing between centers
        dx = hex_width
        dy = hex_height * 0.75

        # Generate centers
        centers_list = []
        row = 0
        y = miny
        while y < maxy + dy:
            x_offset = hex_width / 2 if row % 2 == 1 else 0
            x = minx + x_offset
            while x < maxx + dx:
                centers_list.append((x, y))
                x += dx
            y += dy
            row += 1

        centers = np.array(centers_list, dtype=float)

        # Create polygons and transforms
        polygons = []
        transforms = []
        for cx, cy in centers:
            polygons.append(_create_hexagon(cx, cy, size))
            transforms.append(TileTransform(center=(cx, cy)))

        # Compute adjacency from lattice offsets
        # Pointy-top hex: 6 neighbors at specific offsets
        edge_offsets = [
            (dx, 0),
            (-dx, 0),  # horizontal neighbors
            (dx / 2, dy),
            (-dx / 2, dy),  # upper neighbors
            (dx / 2, -dy),
            (-dx / 2, -dy),  # lower neighbors
        ]
        # For hex grids, edge and vertex neighbors are the same (6 neighbors)
        adj = _compute_adjacency_from_lattice(centers, edge_offsets, tol=size * 0.1)

        inscribed_r = size * np.sqrt(3) / 2  # center to edge
        canonical = _create_hexagon(0, 0, size)

        # Hex grids have no vertex-only neighbors — all 6 are edge neighbors
        m = len(polygons)
        vert_adj = np.zeros((m, m), dtype=bool)

        return TilingResult(
            polygons=polygons,
            transforms=transforms,
            adjacency=adj,
            vertex_adjacency=vert_adj,
            tile_size=size,
            inscribed_radius=inscribed_r,
            canonical_tile=canonical,
        )


# ---------------------------------------------------------------------------
# TriangleTiling
# ---------------------------------------------------------------------------


class TriangleTiling(Tiling):
    """Tiling by congruent triangles.

    Any triangle tiles the Euclidean plane. The tiling is constructed by
    pairing triangles via 180° rotation about an edge midpoint to form
    parallelograms, which then tile by translation.

    Parameters
    ----------
    tile : Polygon
        A triangle (3-vertex polygon). Will be normalized to unit area
        internally. The shape (angles/proportions) is preserved.

    """

    def __init__(self, tile: Polygon) -> None:
        coords = list(tile.exterior.coords)
        if len(coords) != 4:  # 3 vertices + closing point
            raise ValueError(f"Triangle must have 3 vertices, got {len(coords) - 1}")
        # Normalize: translate centroid to origin, scale to unit area
        centroid = tile.centroid
        shifted = translate(tile, -centroid.x, -centroid.y)
        area = abs(shifted.area)
        if area < 1e-12:
            raise ValueError("Triangle has zero area")
        factor = 1.0 / np.sqrt(area)
        self._unit_tile = scale(shifted, xfact=factor, yfact=factor, origin=(0, 0))

    @property
    def canonical_tile(self) -> Polygon:
        """Unit-area triangle centered at origin."""
        return self._unit_tile

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
        # Compute bounding radius of the unit-area canonical tile
        coords = np.array(self._unit_tile.exterior.coords)
        r_orig = float(np.max(np.linalg.norm(coords, axis=1)))
        # tile_size * r_orig = symbol_size
        return (symbol_size / r_orig) * (1 + spacing)

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

    def generate(
        self,
        bounds: tuple[float, float, float, float] = DEFAULT_BOUNDS,
        n_tiles: int | None = None,
        tile_size: float | None = None,
        margin: float = 0.1,
        adjacency_type: TileAdjacencyType = TileAdjacencyType.EDGE,
    ) -> TilingResult:
        bounds, n_tiles, tile_size = self._resolve_generate_args(bounds, n_tiles, tile_size)

        minx, miny, maxx, maxy = _apply_margin(bounds, margin)
        width = maxx - minx
        height = maxy - miny

        if tile_size is None:
            # Each tile has area = tile_size^2 (since unit tile has area 1)
            tile_size = np.sqrt(width * height / n_tiles)

        # Scale unit tile to desired size
        s = tile_size
        tile_a = scale(self._unit_tile, xfact=s, yfact=s, origin=(0, 0))
        verts_a = np.array(tile_a.exterior.coords[:-1])  # 3 vertices

        # Pick the longest edge for pairing
        edges = [(0, 1), (1, 2), (2, 0)]
        edge_lengths = [np.linalg.norm(verts_a[e[1]] - verts_a[e[0]]) for e in edges]
        pair_edge_idx = int(np.argmax(edge_lengths))
        i0, i1 = edges[pair_edge_idx]
        midpoint = (verts_a[i0] + verts_a[i1]) / 2

        # Rotate 180° about the midpoint to get the paired triangle
        tile_b = rotate(tile_a, 180, origin=tuple(midpoint))
        np.array(tile_b.exterior.coords[:-1])

        # The two triangles form a parallelogram. Find translation vectors.
        # The parallelogram has 4 vertices from the union of triangle vertices.
        # Translation vectors are the two independent edge vectors of the
        # parallelogram (edges of triangle A that are NOT the pairing edge).
        other_edges = [e for idx, e in enumerate(edges) if idx != pair_edge_idx]
        vec1 = verts_a[other_edges[0][1]] - verts_a[other_edges[0][0]]
        # The second translation vector goes from a vertex of triangle A
        # to the corresponding vertex of triangle B (across the pairing edge)
        # This is equivalent to 2 * (midpoint - opposite_vertex_of_A) rotated,
        # but simpler: it's the other parallelogram edge.
        # For the parallelogram formed by A and B, the translation vectors are:
        # vec1 = edge vector along one non-paired edge of A
        # vec2 = edge vector along the other non-paired edge of A
        vec2 = verts_a[other_edges[1][1]] - verts_a[other_edges[1][0]]

        # Centroid of tile A (relative offset from origin)
        centroid_a = np.array([tile_a.centroid.x, tile_a.centroid.y])
        # Centroid of tile B
        centroid_b = np.array([tile_b.centroid.x, tile_b.centroid.y])
        # Offset from A's centroid to B's centroid
        offset_b = centroid_b - centroid_a

        # Generate lattice points
        # We need enough lattice points to cover the bounding box
        # Use a generous range of integer coefficients
        max_extent = max(width, height)
        vec_norms = [np.linalg.norm(v) for v in [vec1, vec2]]
        n1 = int(np.ceil(max_extent / max(vec_norms[0], 1e-10))) + 2
        n2 = int(np.ceil(max_extent / max(vec_norms[1], 1e-10))) + 2

        polygons = []
        transforms = []
        centers_list = []

        for i in range(-n1, n1 + 1):
            for j in range(-n2, n2 + 1):
                offset = i * vec1 + j * vec2

                # Place triangle A
                center_a = centroid_a + offset
                if minx - s <= center_a[0] <= maxx + s and miny - s <= center_a[1] <= maxy + s:
                    poly_a = translate(tile_a, offset[0], offset[1])
                    polygons.append(poly_a)
                    transforms.append(
                        TileTransform(
                            center=(center_a[0], center_a[1]),
                            rotation=0.0,
                            flipped=False,
                        ),
                    )
                    centers_list.append(center_a)

                # Place triangle B (rotated 180°)
                center_b = centroid_a + offset + offset_b
                if minx - s <= center_b[0] <= maxx + s and miny - s <= center_b[1] <= maxy + s:
                    poly_b = translate(tile_b, offset[0], offset[1])
                    polygons.append(poly_b)
                    transforms.append(
                        TileTransform(
                            center=(center_b[0], center_b[1]),
                            rotation=180.0,
                            flipped=False,
                        ),
                    )
                    centers_list.append(center_b)

        np.array(centers_list, dtype=float) if centers_list else np.empty((0, 2))

        # Compute adjacency
        edge_adj, vert_adj = _compute_adjacency_matrices(polygons, tol=s * 0.01)
        adj = edge_adj | vert_adj if adjacency_type == TileAdjacencyType.VERTEX else edge_adj

        # Inscribed radius of the scaled tile
        inscribed_r = float(tile_a.exterior.distance(Point(tile_a.centroid)))

        return TilingResult(
            polygons=polygons,
            transforms=transforms,
            adjacency=adj,
            vertex_adjacency=vert_adj,
            tile_size=s,
            inscribed_radius=inscribed_r,
            canonical_tile=tile_a,
        )


# ---------------------------------------------------------------------------
# QuadrilateralTiling
# ---------------------------------------------------------------------------


class QuadrilateralTiling(Tiling):
    """Tiling by congruent quadrilaterals.

    Any simple quadrilateral tiles the Euclidean plane via 180° rotation
    about edge midpoints.

    Parameters
    ----------
    tile : Polygon
        A quadrilateral (4-vertex polygon). Will be normalized to unit area
        internally. The shape (angles/proportions) is preserved.

    """

    def __init__(self, tile: Polygon) -> None:
        coords = list(tile.exterior.coords)
        if len(coords) != 5:  # 4 vertices + closing point
            raise ValueError(f"Quadrilateral must have 4 vertices, got {len(coords) - 1}")
        # Normalize: translate centroid to origin, scale to unit area
        centroid = tile.centroid
        shifted = translate(tile, -centroid.x, -centroid.y)
        area = abs(shifted.area)
        if area < 1e-12:
            raise ValueError("Quadrilateral has zero area")
        factor = 1.0 / np.sqrt(area)
        self._unit_tile = scale(shifted, xfact=factor, yfact=factor, origin=(0, 0))

    @property
    def canonical_tile(self) -> Polygon:
        """Unit-area quadrilateral centered at origin."""
        return self._unit_tile

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
        # Compute bounding radius of the unit-area canonical tile
        coords = np.array(self._unit_tile.exterior.coords)
        r_orig = float(np.max(np.linalg.norm(coords, axis=1)))
        # tile_size * r_orig = symbol_size
        return (symbol_size / r_orig) * (1 + spacing)

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

    def generate(
        self,
        bounds: tuple[float, float, float, float] = DEFAULT_BOUNDS,
        n_tiles: int | None = None,
        tile_size: float | None = None,
        margin: float = 0.1,
        adjacency_type: TileAdjacencyType = TileAdjacencyType.EDGE,
    ) -> TilingResult:
        bounds, n_tiles, tile_size = self._resolve_generate_args(bounds, n_tiles, tile_size)

        minx, miny, maxx, maxy = _apply_margin(bounds, margin)
        width = maxx - minx
        height = maxy - miny

        if tile_size is None:
            tile_size = np.sqrt(width * height / n_tiles)

        # Scale unit tile to desired size
        s = tile_size
        tile_base = scale(self._unit_tile, xfact=s, yfact=s, origin=(0, 0))
        verts = np.array(tile_base.exterior.coords[:-1])  # 4 vertices
        c_base = np.array([tile_base.centroid.x, tile_base.centroid.y])

        # For any quadrilateral, the tiling alternates between type-0 (original
        # orientation) and type-1 (rotated 180°) tiles. The fundamental domain
        # contains 2 tiles. We find the translation vectors by a 2-hop search:
        # type-0 → type-1 (edge neighbor) → type-0 (next edge neighbor).

        # Step 1: Create all 4 type-1 neighbors (rotate about each edge midpoint)
        type1_tiles = []
        for ei in range(4):
            ej = (ei + 1) % 4
            mid = (verts[ei] + verts[ej]) / 2
            t1_tile = rotate(tile_base, 180, origin=tuple(mid))
            type1_tiles.append(t1_tile)

        # Step 2: For each type-1 neighbor, rotate about its edge midpoints to
        # find type-0 tiles (2-hop). Collect all translation vectors.
        all_vecs = []
        for t1_tile in type1_tiles:
            t1_verts = np.array(t1_tile.exterior.coords[:-1])
            for ei in range(4):
                ej = (ei + 1) % 4
                mid = (t1_verts[ei] + t1_verts[ej]) / 2
                type0_tile = rotate(t1_tile, 180, origin=tuple(mid))
                c_type0 = np.array([type0_tile.centroid.x, type0_tile.centroid.y])
                vec = c_type0 - c_base
                if np.linalg.norm(vec) > s * 0.01:  # skip self
                    all_vecs.append(vec)

        # Step 3: Find 2 shortest independent translation vectors
        all_vecs.sort(key=lambda v: np.linalg.norm(v))
        t1 = all_vecs[0]
        t2 = None
        for v in all_vecs[1:]:
            cross = abs(t1[0] * v[1] - t1[1] * v[0])
            if cross > s * 0.01:  # independent
                t2 = v
                break
        if t2 is None:
            raise ValueError("Could not find independent translation vectors")

        # Step 4: The type-1 tile offset (pick the first edge neighbor)
        tile_b = type1_tiles[0]
        c_b = np.array([tile_b.centroid.x, tile_b.centroid.y])
        offset_b = c_b - c_base

        # Generate lattice with 2 tiles per domain point
        max_extent = max(width, height)
        norms = [np.linalg.norm(t1), np.linalg.norm(t2)]
        n1 = int(np.ceil(max_extent / max(norms[0], 1e-10))) + 2
        n2 = int(np.ceil(max_extent / max(norms[1], 1e-10))) + 2

        polygons = []
        transforms_list = []
        centers_list = []

        for i in range(-n1, n1 + 1):
            for j in range(-n2, n2 + 1):
                lattice_offset = i * t1 + j * t2

                # Place type-0 tile
                center_a = c_base + lattice_offset
                if minx - s <= center_a[0] <= maxx + s and miny - s <= center_a[1] <= maxy + s:
                    poly_a = translate(tile_base, lattice_offset[0], lattice_offset[1])
                    polygons.append(poly_a)
                    transforms_list.append(
                        TileTransform(
                            center=(center_a[0], center_a[1]),
                            rotation=0.0,
                            flipped=False,
                        ),
                    )
                    centers_list.append(center_a)

                # Place type-1 tile
                center_b = c_base + lattice_offset + offset_b
                if minx - s <= center_b[0] <= maxx + s and miny - s <= center_b[1] <= maxy + s:
                    poly_b = translate(tile_b, lattice_offset[0], lattice_offset[1])
                    polygons.append(poly_b)
                    transforms_list.append(
                        TileTransform(
                            center=(center_b[0], center_b[1]),
                            rotation=180.0,
                            flipped=False,
                        ),
                    )
                    centers_list.append(center_b)

        np.array(centers_list, dtype=float) if centers_list else np.empty((0, 2))

        # Compute adjacency geometrically (quad tilings have complex
        # adjacency patterns that are hard to precompute analytically)
        edge_adj, vert_adj = _compute_adjacency_matrices(polygons, tol=s * 0.01)
        adj = edge_adj | vert_adj if adjacency_type == TileAdjacencyType.VERTEX else edge_adj

        inscribed_r = float(tile_base.exterior.distance(Point(tile_base.centroid)))

        return TilingResult(
            polygons=polygons,
            transforms=transforms_list,
            adjacency=adj,
            vertex_adjacency=vert_adj,
            tile_size=s,
            inscribed_radius=inscribed_r,
            canonical_tile=tile_base,
        )


# ---------------------------------------------------------------------------
# IsohedralTiling
# ---------------------------------------------------------------------------


class IsohedralTiling(Tiling):
    """Isohedral tiling based on the Grünbaum-Shephard classification.

    Wraps the ``tactile`` library to generate any of 81 isohedral tiling types.
    Each type has 0-6 free parameters controlling tile vertex positions, and
    supports optional custom edge curves for Escher-style tiles.

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

    """

    def __init__(
        self,
        tiling_type: int,
        parameters: list[float] | None = None,
        edge_curves: dict[int, list[tuple[float, float]]] | None = None,
        edge_curve_modes: dict[int, str] | None = None,
    ) -> None:
        from tactile import IsohedralTiling as _TactileIH
        from tactile import tiling_types

        if tiling_type not in tiling_types:
            raise ValueError(
                f"Invalid isohedral tiling type: {tiling_type}. Use IsohedralTiling.available_types() for valid types.",
            )
        self._type = tiling_type
        self._ih = _TactileIH(tiling_type)
        if parameters is not None:
            self._ih.parameters = list(parameters)
        self._edge_curves = {k: list(v) for k, v in edge_curves.items()} if edge_curves else {}
        self._edge_curve_modes = dict(edge_curve_modes) if edge_curve_modes else {}

        # Build prototile and normalize to unit area.
        # Use vertex-only area for normalization (curves don't change
        # tessellation area) and warn if custom curves cause self-intersection.
        vertex_poly = Polygon([(v.x, v.y) for v in self._ih.vertices])
        area = abs(vertex_poly.area)
        if area < 1e-12:
            raise ValueError(f"Tiling type {tiling_type} produces a degenerate tile")

        proto = self._build_prototile(self._edge_curve_modes)
        if self._edge_curves and not proto.is_valid:
            import warnings

            warnings.warn(
                f"IH{tiling_type}: the supplied edge_curves produce a "
                f"self-intersecting prototile. Tiles will be repaired "
                f"automatically via make_valid(), but visual artefacts may "
                f"occur. Consider using a less extreme curve.",
                stacklevel=2,
            )

        cx, cy = proto.centroid.x, proto.centroid.y
        factor = 1.0 / math.sqrt(area)
        coords = [((x - cx) * factor, (y - cy) * factor) for x, y in proto.exterior.coords[:-1]]
        self._unit_tile = Polygon(coords)

    def _enforce_edge_symmetry(
        self,
        curve: list[tuple[float, float]],
        edge_shape,
    ) -> list[tuple[float, float]]:
        """Apply symmetry constraints to an edge curve.

        Parameters
        ----------
        curve : list of (x, y)
            Raw curve points from (0, 0) to (1, 0).
        edge_shape : EdgeShape
            The edge's symmetry type (I, J, S, U).

        Returns
        -------
        list of (x, y)
            Curve with symmetry enforced.

        """
        from tactile import EdgeShape

        if edge_shape == EdgeShape.I:
            return [(0.0, 0.0), (1.0, 0.0)]
        if edge_shape == EdgeShape.J:
            return list(curve)
        # Take first-half points (x <= 0.5)
        first_half = [(x, y) for x, y in curve if x <= 0.5 + 1e-9]
        if not first_half:
            first_half = [(0.0, 0.0)]
        if edge_shape == EdgeShape.S:
            # 180° rotational symmetry about (0.5, 0)
            second_half = [(1.0 - x, -y) for x, y in reversed(first_half)]
        else:  # EdgeShape.U
            # Mirror about x = 0.5
            second_half = [(1.0 - x, y) for x, y in reversed(first_half)]
        # Remove duplicate midpoint
        if (
            len(first_half) > 0
            and len(second_half) > 0
            and abs(first_half[-1][0] - second_half[0][0]) < 1e-9
            and abs(first_half[-1][1] - second_half[0][1]) < 1e-9
        ):
            second_half = second_half[1:]
        return first_half + second_half

    def _build_prototile(self, curve_modes: dict[int, str] | None = None) -> Polygon:
        """Build the prototile polygon with optional custom edge curves.

        Parameters
        ----------
        curve_modes : dict[int, str], optional
            Dictionary mapping shape IDs to curve modes ("linear" or "spline").
            If not provided, all curves are interpolated with splines.

        """
        from tactile import Point as _TPoint
        from tactile import mul as _mul

        points: list[tuple[float, float]] = []
        for shape_info in self._ih.shapes:
            raw_curve = self._edge_curves.get(shape_info.id)
            if raw_curve is None:
                # Straight line in canonical edge space
                curve = [(0.0, 0.0), (1.0, 0.0)]
            else:
                curve = self._enforce_edge_symmetry(raw_curve, shape_info.shape)

            # Interpolate curve only if mode is spline
            if curve_modes and shape_info.id in curve_modes and curve_modes[shape_info.id] == "linear":
                interpolated = curve
            else:
                interpolated = self._interpolate_curve(curve)

            # Transform curve through the edge's affine matrix
            transformed = [_mul(shape_info.T, _TPoint(x, y)) for x, y in interpolated]
            if shape_info.rev:
                transformed = transformed[::-1]

            # Append all points except the last (shared with next edge)
            for p in transformed[:-1]:
                points.append((p.x, p.y))

        return Polygon(points)

    def _interpolate_curve(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Interpolate a list of points with a parametric spline for smooth curves.

        Uses parametric interpolation (x(t), y(t)) to handle curves with vertical
        segments where x may not be strictly increasing.
        """
        import numpy as np
        from scipy.interpolate import CubicSpline

        # If too few points, return as is
        if len(points) <= 2:
            return points

        # Compute cumulative arc length parameter t
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])

        # Arc length parameter (cumulative distance along curve)
        dx = np.diff(x)
        dy = np.diff(y)
        dists = np.sqrt(dx**2 + dy**2)
        t = np.concatenate([[0], np.cumsum(dists)])

        # Normalize t to [0, 1]
        if t[-1] > 0:
            t = t / t[-1]

        # Parametric spline: x(t) and y(t)
        try:
            # Create dense interpolation (100 points)
            t_new = np.linspace(0, 1, 100)

            # Handle edge case where all x or y values are the same
            if len(np.unique(x)) < 2:
                xs_new = np.full_like(t_new, x[0])
            else:
                cs_x = CubicSpline(t, x, bc_type="natural")
                xs_new = cs_x(t_new)

            if len(np.unique(y)) < 2:
                ys_new = np.full_like(t_new, y[0])
            else:
                cs_y = CubicSpline(t, y, bc_type="natural")
                ys_new = cs_y(t_new)

            return list(zip(xs_new, ys_new))
        except Exception:
            # Fallback to original points if interpolation fails
            return points

    @property
    def canonical_tile(self) -> Polygon:
        """Unit-area prototile centered at origin."""
        return self._unit_tile

    @property
    def _n_base_vertices(self) -> int:
        """Topological vertex count of the prototile (ignores curve interpolation points)."""
        return self._ih.num_vertices

    def canonical_symbol(self) -> Symbol:
        """Return an IsohedralTileSymbol with this tiling's parameters and curves."""
        from .symbols import IsohedralTileSymbol

        return IsohedralTileSymbol(
            tiling_type=self._type,
            prototile_params=list(self._ih.parameters),
            edge_curves=self._edge_curves.copy() if self._edge_curves else None,
            edge_curve_modes=self._edge_curve_modes.copy() if self._edge_curve_modes else None,
        )

    def tile_size_for_symbol_size(self, symbol_size: float, spacing: float = 0.0) -> float:
        """Compute tile_size for isohedral tiling.

        IsohedralTiling uses unit-area tiles. TileSymbol normalizes to
        bounding_radius = 0.5, so we need to compute the original tile's
        bounding radius to find the correct scale factor.
        """
        # Compute bounding radius of the unit-area canonical tile
        coords = np.array(self._unit_tile.exterior.coords)
        r_orig = float(np.max(np.linalg.norm(coords, axis=1)))
        # tile_size * r_orig = symbol_size
        return (symbol_size / r_orig) * (1 + spacing)

    @classmethod
    def from_polygon(cls, polygon: Polygon) -> IsohedralTiling:
        """Not supported — use ``IsohedralTiling(tiling_type, parameters)``."""
        raise NotImplementedError(
            "IsohedralTiling is defined by type number and parameters, "
            "not by polygon shape. Use IsohedralTiling(tiling_type=...) instead.",
        )

    @staticmethod
    def available_types() -> list[int]:
        """Return list of valid isohedral tiling type numbers."""
        from tactile import tiling_types

        return list(tiling_types)

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
        """Find isohedral tiling types matching the given criteria.

        All arguments are optional filters.  Only types satisfying every
        specified filter are returned.

        Parameters
        ----------
        num_vertices : int, optional
            Exact number of prototile vertices (3, 4, 5, or 6).
        num_parameters : int, optional
            Exact number of free parameters.
        max_parameters : int, optional
            Maximum number of free parameters.
        num_aspects : int, optional
            Exact number of aspects (copies per fundamental domain).
        has_reflections : bool, optional
            If ``True``, only types with reflected aspects.
            If ``False``, only types without reflections.
        edge_types : set of str, optional
            Required edge types, e.g. ``{"J", "S"}``.
            Only types whose edges include *all* listed types are returned.
        all_edges : str, optional
            If set, only types where *every* edge is this type (e.g. ``"J"``).
        has_customizable_edges : bool, optional
            If ``True``, at least one edge is J, S, or U.
            If ``False``, all edges are I (straight).

        Returns
        -------
        list[int]
            Matching isohedral type numbers, sorted.

        Examples
        --------
        >>> IsohedralTiling.find_types(num_vertices=3, all_edges="J")
        []  # no triangular type has all-J edges
        >>> IsohedralTiling.find_types(num_vertices=4, all_edges="J")
        [41, 43, 44, 52, 55, 59, 61, 68, 71]
        >>> IsohedralTiling.find_types(num_vertices=6, max_parameters=0)
        [10, 31, 33, 34, 36]

        """
        from tactile import IsohedralTiling as _TactileIH
        from tactile import tiling_types

        matches = []
        for tp in tiling_types:
            ih = _TactileIH(tp)

            if num_vertices is not None and ih.num_vertices != num_vertices:
                continue
            if num_parameters is not None and ih.num_parameters != num_parameters:
                continue
            if max_parameters is not None and ih.num_parameters > max_parameters:
                continue
            if num_aspects is not None and ih.num_aspects != num_aspects:
                continue

            edge_names = {s.shape.name for s in ih.shapes}

            if edge_types is not None and not edge_types.issubset(edge_names):
                continue
            if all_edges is not None and edge_names != {all_edges}:
                continue
            if has_customizable_edges is not None:
                has_custom = bool(edge_names - {"I"})
                if has_custom != has_customizable_edges:
                    continue
            if has_reflections is not None:
                refl = any(asp[0] * asp[4] - asp[1] * asp[3] < 0 for asp in ih.aspects)
                if refl != has_reflections:
                    continue

            matches.append(tp)

        return matches

    # ------------------------------------------------------------------
    # Presets: curated tiling configurations
    # ------------------------------------------------------------------

    _PRESETS: ClassVar[dict[str, dict]] = {
        # =============================================================
        # Standard shapes — straight edges, default parameters
        # =============================================================
        "regular_hexagon": {
            "tiling_type": 10,
            "description": (
                "IH10. Regular hexagon with straight edges. 6-fold rotational symmetry (p6), no free parameters."
            ),
        },
        "square": {
            "tiling_type": 41,
            "description": ("IH41. Square with straight edges. Translational symmetry (p1), 2 free parameters."),
        },
        "equilateral_triangle": {
            "tiling_type": 93,
            "description": (
                "IH93. Equilateral triangle with straight edges. Full triangular symmetry (p6m), no free parameters."
            ),
        },
        "rhombus": {
            "tiling_type": 68,
            "description": (
                "IH68. Rhombus (diamond) with straight edges. "
                "Translational symmetry (p1), 1 free parameter "
                "controlling the acute angle."
            ),
        },
        "cairo": {
            "tiling_type": 28,
            "description": (
                "IH28. Cairo tiling — irregular pentagon with straight "
                "edges and 4-fold rotational symmetry (p4). "
                "2 free parameters."
            ),
        },
        # =============================================================
        # Distinctive straight-edge tilings with non-default parameters
        # =============================================================
        "wide_parallelogram": {
            "tiling_type": 43,
            "parameters": [0.4, 0.3],
            "description": ("IH43. Wide parallelogram with 180° rotational symmetry (p2). 2 aspects."),
        },
        # =============================================================
        # Curved-edge tilings — custom edge curves
        # =============================================================
        "scalloped_hexagon": {
            "tiling_type": 10,
            "edge_curves": {
                0: [(0, 0), (0.2, 0.08), (0.5, 0.12), (0.8, 0.08), (1, 0)],
            },
            "edge_curve_modes": {
                0: "spline",
            },
            "description": (
                "IH10. Regular hexagon with scalloped (outward-bulging) "
                "edges. 6-fold symmetry, single shared curve on all "
                "6 edges."
            ),
        },
        "wavy_square": {
            "tiling_type": 41,
            "edge_curves": {
                0: [(0, 0), (0.25, 0.12), (0.5, 0), (0.75, -0.12), (1, 0)],
                1: [(0, 0), (0.25, -0.12), (0.5, 0), (0.75, 0.12), (1, 0)],
            },
            "edge_curve_modes": {
                0: "spline",
                1: "spline",
            },
            "description": (
                "IH41. Square with zigzag edges. Opposite edge pairs use mirrored zigzag curves that interlock."
            ),
        },
        "puzzle_piece": {
            "tiling_type": 41,
            "edge_curves": {
                0: [
                    (0, 0),
                    (0.35, 0),
                    (0.35, 0.06),
                    (0.38, 0.12),
                    (0.44, 0.16),
                    (0.5, 0.18),
                    (0.56, 0.16),
                    (0.62, 0.12),
                    (0.65, 0.06),
                    (0.65, 0),
                    (1, 0),
                ],
                1: [
                    (0, 0),
                    (0.35, 0),
                    (0.35, 0.06),
                    (0.38, 0.12),
                    (0.44, 0.16),
                    (0.5, 0.18),
                    (0.56, 0.16),
                    (0.62, 0.12),
                    (0.65, 0.06),
                    (0.65, 0),
                    (1, 0),
                ],
            },
            "edge_curve_modes": {
                0: "spline",
                1: "spline",
            },
            "description": (
                "IH41. Square with jigsaw puzzle-piece edges — each "
                "edge has a protruding knob that fits into the "
                "neighbouring tile's socket."
            ),
        },
        "arrow_quad": {
            "tiling_type": 43,
            "edge_curves": {
                0: [(0, 0), (0.3, 0), (0.5, 0.15), (0.7, 0), (1, 0)],
                1: [(0, 0), (0.3, 0), (0.5, -0.15), (0.7, 0), (1, 0)],
            },
            "edge_curve_modes": {
                0: "linear",
                1: "linear",
            },
            "description": (
                "IH43. Quadrilateral with arrow-pointed edges — each "
                "edge has a triangular bump or dent at its center. "
                "180° rotational symmetry (p2)."
            ),
        },
        "wavy_triangle": {
            "tiling_type": 84,
            "edge_curves": {
                0: [(0, 0), (0.25, 0.15), (0.5, 0), (0.75, -0.15), (1, 0)],
                1: [(0, 0), (0.25, 0.15), (0.5, 0), (0.75, -0.15), (1, 0)],
                2: [(0, 0), (0.25, 0.15), (0.5, 0), (0.75, -0.15), (1, 0)],
            },
            "edge_curve_modes": {
                0: "spline",
                1: "spline",
                2: "spline",
            },
            "description": (
                "IH84. Triangle with bulging (outward-curved) S-type edges. 2 aspects with 180° rotational symmetry."
            ),
        },
        "interlocking_hexagon": {
            "tiling_type": 10,
            "edge_curves": {
                0: [(0, 0), (-0.12, 0.2), (0.5, 0.6), (1.12, 0.2), (1, 0)],
            },
            "edge_curve_modes": {
                0: "spline",
            },
            "description": (
                "IH10. Regular hexagon with interlocking protrusions. "
                "The single shared curve creates a strongly "
                "convex, lens-like profile on all 6 edges. "
                "6-fold rotational symmetry (p6)."
            ),
        },
        "flowers": {
            "tiling_type": 21,
            "parameters": [0.104512294489, 0.65],
            "edge_curves": {
                0: [(0, 0), (0.25, 0.08), (0.5, 0.0), (0.75, -0.08), (1.0, 0)],
                1: [(0, 0), (0.25, 0.08), (0.5, 0.12), (0.75, 0.08), (1, 0)],
                2: [(0, 0), (0.25, 0.08), (0.5, 0.12), (0.75, 0.08), (1, 0)],
            },
            "edge_curve_modes": {
                0: "spline",
                1: "spline",
                2: "spline",
            },
            "description": (
                "IH21. Pentagon with mixed curved edges: one S-shaped "
                "edge (sinuous, 180° symmetric) and two pairs of "
                "gently scalloped J-edges. 6 aspects."
            ),
        },
        "scalloped_quad": {
            "tiling_type": 33,
            "edge_curves": {
                0: [(0, 0), (0.25, 0.08), (0.5, 0.12), (0.75, 0.08), (1, 0)],
                1: [(0, 0), (0.25, 0.08), (0.5, 0.12), (0.75, 0.08), (1, 0)],
            },
            "edge_curve_modes": {
                0: "spline",
                1: "spline",
            },
            "description": (
                "IH33. Quadrilateral with uniformly scalloped edges — "
                "all 4 edges share the same gentle outward bulge. "
                "3 aspects, no free parameters."
            ),
        },
        "tree": {
            "tiling_type": 4,
            "parameters": [
                0.027230055765024366,
                0.5,
                0.315470053838,
                0.5,
                0.315470053838,
                0.5,
            ],
            "edge_curves": {
                0: [(0, 0), (0.17, -0.16), (0.5, 0.0), (0.83, 0.16), (1.0, 0)],
                1: [(0, 0), (0.5, 0.3), (1, 0)],
                2: [(0, 0), (0.17, 0.16), (0.5, 0.0), (0.83, -0.16), (1.0, 0)],
                3: [(0, 0), (0.2, 0.1), (0.4, -0.1), (0.5, 0.0), (0.6, 0.1), (0.8, -0.1), (1.0, 0)],
                4: [(0, 0), (0.2, -0.1), (0.4, 0.1), (0.5, 0.0), (0.6, -0.1), (0.8, 0.1), (1.0, 0)],
            },
            "edge_curve_modes": {
                0: "spline",
                1: "spline",
                2: "spline",
                3: "spline",
            },
            "description": (
                "IH4. Hexagonal tile with 5 distinct edge shapes and "
                "6 parameters. Interlocking S-curves and wavy edges "
                "produce a tree-like silhouette. 2 aspects."
            ),
        },
    }

    @classmethod
    def from_preset(cls, name: str, **kwargs) -> IsohedralTiling:
        """Create an IsohedralTiling from a named preset.

        Parameters
        ----------
        name : str
            Preset name.  Use ``IsohedralTiling.list_presets()`` for options.
        **kwargs
            Overrides passed to ``IsohedralTiling()``, e.g.
            ``edge_curves``, ``parameters``.

        Returns
        -------
        IsohedralTiling

        """
        if name not in cls._PRESETS:
            valid = ", ".join(sorted(cls._PRESETS))
            raise ValueError(f"Unknown preset: {name!r}. Available presets: {valid}")
        preset = cls._PRESETS[name]
        init_kwargs: dict = {"tiling_type": preset["tiling_type"]}
        if "parameters" in preset:
            init_kwargs["parameters"] = preset["parameters"]
        if "edge_curves" in preset:
            init_kwargs["edge_curves"] = preset["edge_curves"]
        if "edge_curve_modes" in preset:
            init_kwargs["edge_curve_modes"] = preset["edge_curve_modes"]
        init_kwargs.update(kwargs)
        return cls(**init_kwargs)

    @classmethod
    def list_presets(cls) -> dict[str, str]:
        """Return a dict mapping preset names to their descriptions.

        Returns
        -------
        dict[str, str]

        """
        return {name: p["description"] for name, p in cls._PRESETS.items()}

    def to_dict(self) -> dict:
        """Serialize the tiling configuration to a dictionary.

        Returns
        -------
        dict
            Dictionary with keys ``tiling_type``, ``parameters``, and
            ``edge_curves`` suitable for passing to :meth:`from_dict` or
            saving as JSON.

        """
        return {
            "version": "1.0",
            "tiling_type": self._type,
            "parameters": list(self._ih.parameters),
            "edge_curves": {str(k): [list(pt) for pt in v] for k, v in self._edge_curves.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> IsohedralTiling:
        """Load a tiling from a serialized dictionary.

        Parameters
        ----------
        data : dict
            Dictionary as produced by :meth:`to_dict`.

        Returns
        -------
        IsohedralTiling

        """
        tiling_type = data["tiling_type"]
        parameters = data.get("parameters")
        edge_curves_raw = data.get("edge_curves", {})
        edge_curves: dict[int, list[tuple[float, float]]] = {
            int(k): [tuple(pt) for pt in v]  # type: ignore[misc]
            for k, v in edge_curves_raw.items()
        }
        return cls(
            tiling_type=tiling_type,
            parameters=parameters,
            edge_curves=edge_curves if edge_curves else None,
        )

    def save(self, path: str | Path) -> None:
        """Save the tiling configuration to a JSON file.

        Parameters
        ----------
        path : str or Path
            Destination file path.  The file is created or overwritten.

        """
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> IsohedralTiling:
        """Load a tiling configuration from a JSON file.

        Parameters
        ----------
        path : str or Path
            Path to a JSON file previously written by :meth:`save`.

        Returns
        -------
        IsohedralTiling

        """
        import json

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    _EDGE_DESCRIPTIONS: ClassVar[dict[str, str]] = {
        "J": "Free curve",
        "S": "180° rotationally symmetric",
        "U": "Mirror symmetric",
        "I": "Straight (fixed)",
    }

    def type_info(self_or_type=None, tiling_type: int | None = None) -> dict:
        """Return metadata for an isohedral tiling type.

        Can be called on an instance (``t.type_info()``) or on the class
        (``IsohedralTiling.type_info(83)``).

        Parameters
        ----------
        tiling_type : int, optional
            Isohedral type number.  Not needed when called on an instance.

        Returns
        -------
        dict
            Keys include ``num_parameters``, ``num_vertices``, ``num_aspects``,
            ``num_edge_shapes``, ``edge_shapes``, ``default_parameters``,
            ``edges``, ``has_reflections``, ``customizable_shape_ids``,
            ``parameters``.

            The ``edges`` list has one entry per prototile edge with keys:
            ``edge_index``, ``shape_id``, ``shape_type``, ``description``,
            ``customizable``.

        """
        from tactile import EdgeShape, tiling_types
        from tactile import IsohedralTiling as _TactileIH

        # Resolve tiling_type from instance or argument
        if isinstance(self_or_type, IsohedralTiling):
            tiling_type = self_or_type._type
        elif isinstance(self_or_type, int):
            tiling_type = self_or_type
        elif tiling_type is None:
            raise TypeError(
                "type_info() requires a tiling type number when called on the class, e.g. IsohedralTiling.type_info(83)",
            )

        if tiling_type not in tiling_types:
            raise ValueError(f"Invalid isohedral tiling type: {tiling_type}")
        ih = _TactileIH(tiling_type)

        # Build per-edge info
        edges = []
        for idx, shape_info in enumerate(ih.shapes):
            stype = shape_info.shape.name
            edges.append({
                "edge_index": idx,
                "shape_id": shape_info.id,
                "shape_type": stype,
                "description": IsohedralTiling._EDGE_DESCRIPTIONS.get(stype, stype),
                "customizable": shape_info.shape != EdgeShape.I,
            })

        # Detect reflections by checking aspect determinants
        has_reflections = False
        for asp in ih.aspects:
            det = asp[0] * asp[4] - asp[1] * asp[3]
            if det < 0:
                has_reflections = True
                break

        customizable_ids = sorted({e["shape_id"] for e in edges if e["customizable"]})

        # Compute per-parameter descriptions by finite differences on
        # vertex positions: which vertices move, in what direction.
        param_descs: list[dict] = []
        base_verts = [(v.x, v.y) for v in ih.vertices]
        base_params = list(ih.parameters)
        delta = 1e-4
        for pi in range(ih.num_parameters):
            perturbed = list(base_params)
            perturbed[pi] += delta
            ih.parameters = perturbed
            new_verts = [(v.x, v.y) for v in ih.vertices]
            ih.parameters = list(base_params)  # restore

            affected: list[dict] = []
            for vi in range(ih.num_vertices):
                dx = (new_verts[vi][0] - base_verts[vi][0]) / delta
                dy = (new_verts[vi][1] - base_verts[vi][1]) / delta
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    affected.append({
                        "vertex": vi,
                        "dx_dp": round(dx, 4),
                        "dy_dp": round(dy, 4),
                    })

            # Build a human-readable summary
            if not affected:
                summary = "no visible effect on vertices"
            else:
                parts = []
                for a in affected:
                    dx, dy = a["dx_dp"], a["dy_dp"]
                    if abs(dy) < 1e-6:
                        direction = "horizontally"
                    elif abs(dx) < 1e-6:
                        direction = "vertically"
                    else:
                        direction = "diagonally"
                    parts.append(f"V{a['vertex']} {direction}")
                summary = "moves " + ", ".join(parts)

            param_descs.append({
                "index": pi,
                "default": base_params[pi],
                "affected_vertices": affected,
                "description": summary,
            })

        return {
            "num_parameters": ih.num_parameters,
            "num_vertices": ih.num_vertices,
            "num_aspects": ih.num_aspects,
            "num_edge_shapes": ih.num_edge_shapes,
            "edge_shapes": [s.shape.name for s in ih.shapes],
            "default_parameters": list(base_params),
            "edges": edges,
            "has_reflections": has_reflections,
            "customizable_shape_ids": customizable_ids,
            "parameters": param_descs,
        }

    def describe(self_or_type=None, tiling_type: int | None = None) -> str:
        """Return a human-readable summary of an isohedral tiling type.

        Can be called on an instance (``t.describe()``) or on the class
        (``IsohedralTiling.describe(83)``).

        Parameters
        ----------
        tiling_type : int, optional
            Isohedral type number.  Not needed when called on an instance.

        Returns
        -------
        str
            Multi-line description of the tiling type.

        """
        # Resolve tiling_type from instance or argument
        if isinstance(self_or_type, IsohedralTiling):
            tiling_type = self_or_type._type
        elif isinstance(self_or_type, int):
            tiling_type = self_or_type
        elif tiling_type is None:
            raise TypeError(
                "describe() requires a tiling type number when called on the class, e.g. IsohedralTiling.describe(83)",
            )

        info = IsohedralTiling.type_info(tiling_type)
        lines = [f"IH{tiling_type} — Isohedral Tiling Type {tiling_type}"]

        # Parameters
        n_params = info["num_parameters"]
        if n_params == 0:
            lines.append("  Parameters: none (fixed shape)")
        else:
            lines.append(f"  Parameters: {n_params} free")
            for pd in info["parameters"]:
                lines.append(f"    [{pd['index']}] default={pd['default']:.4g}  — {pd['description']}")

        # Prototile shape
        lines.append(f"  Prototile: {info['num_vertices']} vertices, {info['num_vertices']} edges")
        refl = "with reflections" if info["has_reflections"] else "no reflections"
        lines.append(f"  Aspects: {info['num_aspects']} ({refl})")
        lines.append("  Edges:")

        # Track how many edges share each shape_id
        id_counts: dict[int, int] = {}
        for e in info["edges"]:
            id_counts[e["shape_id"]] = id_counts.get(e["shape_id"], 0) + 1

        for e in info["edges"]:
            sid = e["shape_id"]
            tag = e["shape_type"]
            desc = e["description"]
            shared = f", same shape as id {sid}" if id_counts[sid] > 1 else ""
            lines.append(f"    Edge {e['edge_index']} (shape_id={sid}): {tag} — {desc}{shared}")

        cids = info["customizable_shape_ids"]
        if cids:
            lines.append(f"  Customizable shape IDs: {{{', '.join(map(str, cids))}}}")
        else:
            lines.append("  No customizable edges (all straight)")

        return "\n".join(lines)

    def plot_prototile(self, ax=None, show_labels: bool = True):
        """Plot the prototile with labeled, color-coded edges.

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
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
        from tactile import EdgeShape
        from tactile import Point as _TPoint
        from tactile import mul as _mul

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 6))

        # Qualitative palette for shape_ids
        palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999"]

        # Collect per-edge info and polylines
        edge_data: list[dict] = []
        for shape_info in self._ih.shapes:
            raw_curve = self._edge_curves.get(shape_info.id)
            if raw_curve is None:
                curve = [(0.0, 0.0), (1.0, 0.0)]
            else:
                curve = self._enforce_edge_symmetry(raw_curve, shape_info.shape)

            transformed = [_mul(shape_info.T, _TPoint(x, y)) for x, y in curve]
            if shape_info.rev:
                transformed = transformed[::-1]

            pts = [(p.x, p.y) for p in transformed]
            edge_data.append({
                "pts": pts,
                "shape_id": shape_info.id,
                "shape_type": shape_info.shape.name,
                "customizable": shape_info.shape != EdgeShape.I,
            })

        # Track which shape_ids appear multiple times
        id_counts: dict[int, int] = {}
        for ed in edge_data:
            id_counts[ed["shape_id"]] = id_counts.get(ed["shape_id"], 0) + 1

        # Draw filled polygon
        all_pts = []
        for ed in edge_data:
            all_pts.extend(ed["pts"][:-1])
        poly_patch = plt.Polygon(all_pts, closed=True, facecolor="#f0f0f0", edgecolor="none")
        ax.add_patch(poly_patch)

        # Draw each edge and build legend handles (one per unique shape_id)
        from matplotlib.lines import Line2D

        legend_entries: dict[int, tuple[str, str, str]] = {}  # id → (color, ls, type)

        for idx, ed in enumerate(edge_data):
            xs = [p[0] for p in ed["pts"]]
            ys = [p[1] for p in ed["pts"]]
            color = palette[ed["shape_id"] % len(palette)]
            lw = 2.5 if ed["customizable"] else 1.5
            ls = "-" if ed["customizable"] else "--"
            ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, solid_capstyle="round")

            if ed["shape_id"] not in legend_entries:
                desc = self._EDGE_DESCRIPTIONS.get(ed["shape_type"], ed["shape_type"])
                legend_entries[ed["shape_id"]] = (color, ls, desc)

            # Compute the geometric midpoint along the polyline
            pts = ed["pts"]
            if len(pts) < 2:
                mx, my = pts[0]
                ax0, ay0, ax1, ay1 = mx, my, mx, my
            else:
                # Cumulative arc lengths
                seg_lens = [
                    math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)
                ]
                total = sum(seg_lens)
                half = total / 2.0
                # Walk along the polyline to find the midpoint
                acc = 0.0
                mx, my = pts[0]
                seg_idx = 0
                for si, sl in enumerate(seg_lens):
                    if acc + sl >= half:
                        seg_idx = si
                        t = (half - acc) / sl if sl > 0 else 0.5
                        mx = pts[si][0] + t * (pts[si + 1][0] - pts[si][0])
                        my = pts[si][1] + t * (pts[si + 1][1] - pts[si][1])
                        break
                    acc += sl
                # Arrow segment: two points straddling the midpoint
                eps_t = min(0.15, 0.5)  # fraction of the segment
                t0 = max(0.0, t - eps_t)
                t1 = min(1.0, t + eps_t)
                si = seg_idx
                ax0 = pts[si][0] + t0 * (pts[si + 1][0] - pts[si][0])
                ay0 = pts[si][1] + t0 * (pts[si + 1][1] - pts[si][1])
                ax1 = pts[si][0] + t1 * (pts[si + 1][0] - pts[si][0])
                ay1 = pts[si][1] + t1 * (pts[si + 1][1] - pts[si][1])

            if show_labels:
                tag = ed["shape_type"]
                shared = f", ={ed['shape_id']}" if id_counts[ed["shape_id"]] > 1 else ""
                label = f"edge {idx} ({tag}{shared})"
                ax.annotate(
                    label,
                    (mx, my),
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": color, "alpha": 0.85},
                )

            # Arrow at midpoint showing direction
            arrow = FancyArrowPatch(
                (ax0, ay0),
                (ax1, ay1),
                arrowstyle="-|>",
                mutation_scale=12,
                color=color,
                linewidth=0,
            )
            ax.add_patch(arrow)

        # Vertex dots with index labels
        vertices = list(self._ih.vertices)
        vx = [v.x for v in vertices]
        vy = [v.y for v in vertices]
        ax.scatter(vx, vy, color="black", s=40, zorder=5)
        if show_labels:
            # Compute centroid to offset labels outward
            cx = sum(vx) / len(vx)
            cy = sum(vy) / len(vy)
            for i, (x, y) in enumerate(zip(vx, vy)):
                # Nudge label away from centroid
                dx, dy = x - cx, y - cy
                dist = math.sqrt(dx * dx + dy * dy) or 1.0
                offset = 0.08 * max(
                    max(vx) - min(vx),
                    max(vy) - min(vy),
                )
                nx, ny = x + dx / dist * offset, y + dy / dist * offset
                ax.annotate(
                    f"V{i}",
                    (nx, ny),
                    fontsize=9,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox={"boxstyle": "circle,pad=0.15", "fc": "white", "ec": "black", "linewidth": 0.8},
                    zorder=6,
                )

        # Legend
        handles = []
        for sid in sorted(legend_entries):
            color, ls, desc = legend_entries[sid]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linewidth=2.5,
                    linestyle=ls,
                    label=f"id {sid}: {desc}",
                ),
            )
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

        ax.set_aspect("equal")
        ax.set_title(f"IH{self._type} Prototile")
        ax.autoscale_view()
        margin = 0.1 * max(
            ax.get_xlim()[1] - ax.get_xlim()[0],
            ax.get_ylim()[1] - ax.get_ylim()[0],
        )
        ax.set_xlim(ax.get_xlim()[0] - margin, ax.get_xlim()[1] + margin)
        ax.set_ylim(ax.get_ylim()[0] - margin, ax.get_ylim()[1] + margin)
        ax.set_axis_off()

        return ax

    def generate(
        self,
        bounds: tuple[float, float, float, float] = DEFAULT_BOUNDS,
        n_tiles: int | None = None,
        tile_size: float | None = None,
        margin: float = 0.1,
        adjacency_type: TileAdjacencyType = TileAdjacencyType.EDGE,
        compute_adjacency: bool = True,
    ) -> TilingResult:
        bounds, n_tiles, tile_size = self._resolve_generate_args(bounds, n_tiles, tile_size)

        minx, miny, maxx, maxy = _apply_margin(bounds, margin)
        width = maxx - minx
        height = maxy - miny

        if tile_size is None:
            tile_size = math.sqrt(width * height / n_tiles)

        # Compute scale factor from the vertex-only (straight-edge) polygon.
        # Custom edge curves don't change the tiling lattice or the area each
        # tile occupies — they only reshape visual boundaries.  Using the
        # curved area would give a wrong scale when curves are extreme.
        from tactile import Point as _TPoint
        from tactile import mul as _mul

        vertex_poly = Polygon([(v.x, v.y) for v in self._ih.vertices])
        native_area = abs(vertex_poly.area)
        scale_factor = tile_size / math.sqrt(native_area)

        # Build curved prototile for tile rendering
        native_proto = self._build_prototile(self._edge_curve_modes)
        native_verts = list(native_proto.exterior.coords[:-1])

        # Center of the target region
        cx = (minx + maxx) / 2.0
        cy = (miny + maxy) / 2.0

        # Convert world bounds to native coordinates (centered on origin).
        # Expand by half a lattice vector in each direction to ensure
        # tactile's scanline fill covers the full region.  Also add a
        # small asymmetric perturbation to avoid degenerate cases
        # (division-by-zero when lattice-space points align exactly).
        half_w = (width / 2.0) / scale_factor
        half_h = (height / 2.0) / scale_factor
        t1_vec = self._ih.t1
        t2_vec = self._ih.t2
        lat_max = max(abs(t1_vec.x), abs(t1_vec.y), abs(t2_vec.x), abs(t2_vec.y), 1e-6)
        pad = 0.5 * lat_max
        eps = 0.01 * lat_max
        native_bounds = (
            -half_w - pad - eps,
            -half_h - pad - 0.7 * eps,
            half_w + pad + eps,
            half_h + pad + 0.7 * eps,
        )

        polygons = []
        transforms = []

        for tile_info in self._ih.fill_region_bounds(*native_bounds):
            T = tile_info.T

            # Transform prototile vertices by the tile's affine matrix
            world_verts = []
            for vx, vy in native_verts:
                p = _mul(T, _TPoint(vx, vy))
                wx = p.x * scale_factor + cx
                wy = p.y * scale_factor + cy
                world_verts.append((wx, wy))

            poly = Polygon(world_verts)
            if not poly.is_valid:
                from shapely.validation import make_valid

                poly = make_valid(poly)
                # make_valid may return a GeometryCollection; keep only polygons
                if poly.geom_type == "MultiPolygon":
                    poly = max(poly.geoms, key=lambda g: g.area)
                elif poly.geom_type != "Polygon":
                    continue
            if poly.area < 1e-12:
                continue

            # Extract rotation and flip from the 2x3 affine matrix.
            # For reflected transforms (det < 0), the 2x2 part is
            # R(θ)·diag(-1,1) = [[-cosθ, -sinθ],[-sinθ, cosθ]],
            # so we negate both args to atan2 to recover the correct θ.
            det = T[0] * T[4] - T[1] * T[3]
            flipped = det < 0
            rotation = math.degrees(math.atan2(-T[3], -T[0])) if flipped else math.degrees(math.atan2(T[3], T[0]))

            center = (poly.centroid.x, poly.centroid.y)
            polygons.append(poly)
            transforms.append(
                TileTransform(
                    center=center,
                    rotation=rotation,
                    flipped=flipped,
                ),
            )

        if not polygons:
            raise ValueError(
                f"Tiling type {self._type} generated 0 tiles in the given "
                f"bounds. Try increasing n_tiles or adjusting parameters.",
            )

        # Deduplicate overlapping tiles — tactile's fill_region_bounds
        # can produce duplicate tiles at the boundary of its scanline fill.
        if len(polygons) > 1:
            centers_arr = np.array([(p.centroid.x, p.centroid.y) for p in polygons])
            keep = np.ones(len(polygons), dtype=bool)
            dedup_tol = tile_size * 0.01
            for i in range(len(polygons)):
                if not keep[i]:
                    continue
                for j in range(i + 1, len(polygons)):
                    if keep[j] and np.linalg.norm(centers_arr[i] - centers_arr[j]) < dedup_tol:
                        keep[j] = False
            if not keep.all():
                polygons = [p for p, k in zip(polygons, keep) if k]
                transforms = [t for t, k in zip(transforms, keep) if k]

        # Adjacency
        if compute_adjacency:
            edge_adj, vert_adj = _compute_adjacency_matrices(polygons, tol=tile_size * 0.01)
            adj = edge_adj | vert_adj if adjacency_type == TileAdjacencyType.VERTEX else edge_adj
        else:
            _n = len(polygons)
            adj = np.zeros((_n, _n), dtype=bool)
            vert_adj = adj

        # Inscribed radius of the scaled canonical tile
        scaled_proto = scale(
            self._unit_tile,
            xfact=tile_size,
            yfact=tile_size,
            origin=(0, 0),
        )
        inscribed_r = float(scaled_proto.exterior.distance(Point(scaled_proto.centroid)))

        return TilingResult(
            polygons=polygons,
            transforms=transforms,
            adjacency=adj,
            vertex_adjacency=vert_adj,
            tile_size=tile_size,
            inscribed_radius=inscribed_r,
            canonical_tile=scaled_proto,
            n_base_vertices=self._ih.num_vertices,
        )


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
            if spec in IsohedralTiling._PRESETS:
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
