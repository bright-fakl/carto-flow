"""Symbol creation utilities for symbol cartograms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .layout_result import Transform

import numpy as np
from numpy.typing import NDArray
from shapely.affinity import rotate, scale, translate
from shapely.geometry import Point, Polygon

# ---------------------------------------------------------------------------
# Low-level creation functions (kept for backward compat and internal use)
# ---------------------------------------------------------------------------


def create_circle(
    center: tuple[float, float],
    radius: float,
    n_points: int = 32,
) -> Polygon:
    """Create a circular polygon.

    Parameters
    ----------
    center : tuple[float, float]
        Center coordinates (x, y).
    radius : float
        Circle radius.
    n_points : int, optional
        Number of points to approximate the circle. Default is 32.

    Returns
    -------
    Polygon
        Circular polygon.

    """
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return Polygon(zip(x, y, strict=True))


def create_square(
    center: tuple[float, float],
    half_side: float,
) -> Polygon:
    """Create an axis-aligned square polygon.

    Parameters
    ----------
    center : tuple[float, float]
        Center coordinates (x, y).
    half_side : float
        Half the side length (distance from center to edge).

    Returns
    -------
    Polygon
        Square polygon.

    """
    cx, cy = center
    return Polygon([
        (cx - half_side, cy - half_side),
        (cx + half_side, cy - half_side),
        (cx + half_side, cy + half_side),
        (cx - half_side, cy + half_side),
    ])


def create_hexagon(
    center: tuple[float, float],
    size: float,
    pointy_top: bool = True,
) -> Polygon:
    """Create an axis-aligned hexagonal polygon.

    Parameters
    ----------
    center : tuple[float, float]
        Center coordinates (x, y).
    size : float
        Distance from center to vertex.
    pointy_top : bool, optional
        If True, hexagon has a vertex at top (pointy-top orientation).
        If False, hexagon has a flat edge at top (flat-top orientation).
        Default is True.

    Returns
    -------
    Polygon
        Hexagonal polygon.

    """
    base_angle = np.pi / 6 if pointy_top else 0  # 30 deg offset for pointy-top
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + base_angle
    x = center[0] + size * np.cos(angles)
    y = center[1] + size * np.sin(angles)
    return Polygon(zip(x, y, strict=True))


# ---------------------------------------------------------------------------
# Symbol class hierarchy
# ---------------------------------------------------------------------------


class Symbol(ABC):
    """A symbol defined in the unit square [-0.5, 0.5]².

    Subclass this to create custom symbols. The symbol geometry is defined
    once in unit coordinates via :meth:`unit_polygon`, then scaled and
    positioned by :meth:`create`.

    Properties :attr:`bounding_radius` and :attr:`inscribed_radius` are
    used by placement algorithms to convert between symbol size and the
    effective collision radius.
    """

    @abstractmethod
    def unit_polygon(self) -> Polygon:
        """Return the symbol as a polygon in [-0.5, 0.5]²."""
        ...

    def create(
        self,
        center: tuple[float, float],
        size: float | tuple[float, float],
        rotation: float = 0.0,
        flipped: bool = False,
    ) -> Polygon:
        """Create a scaled, optionally transformed polygon at *center*.

        Parameters
        ----------
        center : tuple[float, float]
            Center coordinates (x, y).
        size : float or tuple[float, float]
            Scalar size (isotropic) or ``(sx, sy)`` for anisotropic
            scaling.  The unit polygon is scaled by ``2 * size`` in each
            direction so that *size* corresponds to the half-extent.
        rotation : float
            Rotation angle in degrees (counter-clockwise). Applied after
            scaling and flipping, before translation.
        flipped : bool
            Whether to reflect the symbol about the vertical axis before
            rotation. Used for tiling patterns that involve reflection.

        """
        p = self.unit_polygon()
        if isinstance(size, tuple | list | np.ndarray):
            sx, sy = float(size[0]) * 2, float(size[1]) * 2
        else:
            sx = sy = float(size) * 2
        p = scale(p, xfact=sx, yfact=sy, origin=(0, 0))
        if flipped:
            p = scale(p, xfact=-1, yfact=1, origin=(0, 0))
        if rotation != 0.0:
            p = rotate(p, rotation, origin=(0, 0))
        return translate(p, *center)

    @property
    def bounding_radius(self) -> float:
        """Radius of the bounding circle of the unit polygon.

        Used by circle-based simulators as the collision radius.
        """
        coords = np.array(self.unit_polygon().exterior.coords)
        return float(np.max(np.linalg.norm(coords, axis=1)))

    @property
    def inscribed_radius(self) -> float:
        """Radius of the largest inscribed circle centred at the origin.

        Equal to the distance from the origin to the nearest polygon edge.
        """
        return float(self.unit_polygon().exterior.distance(Point(0, 0)))

    @property
    def area_factor(self) -> float:
        """Factor to convert area-equivalent size to native half-extent.

        For a symbol with area-equivalent size S (where the symbol area
        should equal π * S²), the native half-extent used for rendering
        is S * area_factor.

        This ensures that symbols of different shapes with the same
        area-equivalent size have the same visual area.

        Returns
        -------
        float
            The area correction factor.

        """
        # Default: compute from unit polygon area
        # After scaling by 2*size, the polygon area is:
        #   scaled_area = unit_area * (2*size)² = unit_area * 4 * size²
        # We want: scaled_area = π * size²
        # So: unit_area * 4 * (size * factor)² = π * size²
        # factor = sqrt(π / (4 * unit_area))
        unit_area = self.unit_polygon().area
        return float(np.sqrt(np.pi / (4 * unit_area)))

    def get_geometry(
        self,
        position: tuple[float, float],
        size: float,
        transform: Transform | None = None,
        offset: tuple[float, float] = (0.0, 0.0),
    ) -> Polygon:
        """Return shapely geometry for this symbol.

        Default implementation uses unit_polygon() + transform application.
        Subclasses may override for efficiency.

        Parameters
        ----------
        position : tuple[float, float]
            Center position (x, y).
        size : float
            Symbol size (radius or half-width).
        transform : Transform or None
            Additional rotation (radians), scale, reflection.
            If None, uses identity transform.
        offset : tuple[float, float]
            Offset in unit coordinates, applied after scaling but before
            rotation/reflection. Used for FILL mode fitting where the symbol
            center is shifted from the tile center.

        Returns
        -------
        Polygon
            The transformed symbol geometry.

        """
        if transform is None:
            transform = Transform()

        # Start with unit polygon
        p = self.unit_polygon()

        # Apply scale
        effective_size = size * transform.scale
        p = scale(p, xfact=effective_size * 2, yfact=effective_size * 2, origin=(0, 0))

        # Apply offset (in scaled coordinates)
        if offset != (0.0, 0.0):
            p = translate(p, offset[0] * effective_size, offset[1] * effective_size)

        # Apply reflection
        if transform.reflection:
            p = scale(p, xfact=-1, yfact=1, origin=(0, 0))

        # Apply rotation (convert radians to degrees)
        if transform.rotation != 0.0:
            p = rotate(p, np.degrees(transform.rotation), origin=(0, 0))

        # Translate to position
        return translate(p, position[0], position[1])

    def modify(self, **params) -> Symbol:
        """Return new symbol with modified parameters.

        Default implementation returns self (immutable).
        Subclasses with parameters should override.

        Parameters
        ----------
        **params
            Parameters to modify (subclass-specific).

        Returns
        -------
        Symbol
            Modified symbol (may be same instance if no changes).

        """
        return self


class CircleSymbol(Symbol):
    """Circle inscribed in the unit square (radius = 0.5)."""

    def unit_polygon(self) -> Polygon:
        return create_circle((0, 0), 0.5, n_points=32)

    @property
    def bounding_radius(self) -> float:
        return 0.5

    @property
    def inscribed_radius(self) -> float:
        return 0.5

    @property
    def area_factor(self) -> float:
        """Circle is the reference: area = π * size²."""
        return 1.0

    def modify(self, **params) -> Symbol:
        """Return self (circle has no modifiable parameters)."""
        return self


class SquareSymbol(Symbol):
    """Square filling the unit square (half-side = 0.5)."""

    def unit_polygon(self) -> Polygon:
        return create_square((0, 0), 0.5)

    @property
    def bounding_radius(self) -> float:
        return np.sqrt(2) / 2  # ≈ 0.707

    @property
    def inscribed_radius(self) -> float:
        return 0.5

    def modify(self, **params) -> Symbol:
        """Return self (square has no modifiable parameters)."""
        return self


class HexagonSymbol(Symbol):
    """Hexagon inscribed in the unit square.

    Parameters
    ----------
    pointy_top : bool
        If True (default), hexagon has a vertex at top.

    """

    def __init__(self, pointy_top: bool = True) -> None:
        self.pointy_top = pointy_top

    def unit_polygon(self) -> Polygon:
        return create_hexagon((0, 0), 0.5, pointy_top=self.pointy_top)

    @property
    def bounding_radius(self) -> float:
        return 0.5  # center-to-vertex

    @property
    def inscribed_radius(self) -> float:
        return 0.5 * np.sqrt(3) / 2  # center-to-edge ≈ 0.433

    def modify(self, pointy_top: bool | None = None, **params) -> HexagonSymbol:
        """Return new symbol with modified parameters.

        Parameters
        ----------
        pointy_top : bool or None
            If provided, set the orientation.

        Returns
        -------
        HexagonSymbol
            New symbol with modified parameters.

        """
        if pointy_top is None or pointy_top == self.pointy_top:
            return self
        return HexagonSymbol(pointy_top=pointy_top)


class TriangleSymbol(Symbol):
    """Equilateral triangle inscribed in unit circle (radius = 0.5).

    Parameters
    ----------
    pointing_up : bool
        If True (default), triangle points upward. If False, points downward.

    """

    def __init__(self, pointing_up: bool = True) -> None:
        self.pointing_up = pointing_up

    def unit_polygon(self) -> Polygon:
        offset = np.pi / 2 if self.pointing_up else -np.pi / 2
        angles = np.linspace(0, 2 * np.pi, 4)[:-1] + offset
        x = 0.5 * np.cos(angles)
        y = 0.5 * np.sin(angles)
        return Polygon(zip(x, y, strict=True))

    @property
    def bounding_radius(self) -> float:
        return 0.5

    def modify(self, pointing_up: bool | None = None, **params) -> TriangleSymbol:
        """Return new symbol with modified orientation.

        Parameters
        ----------
        pointing_up : bool or None
            If provided, set the orientation.

        """
        if pointing_up is None or pointing_up == self.pointing_up:
            return self
        return TriangleSymbol(pointing_up=pointing_up)


class DiamondSymbol(Symbol):
    """Diamond (square rotated 45°) inscribed in unit circle (radius = 0.5)."""

    def unit_polygon(self) -> Polygon:
        return Polygon([(0, 0.5), (0.5, 0), (0, -0.5), (-0.5, 0)])

    @property
    def bounding_radius(self) -> float:
        return 0.5

    @property
    def inscribed_radius(self) -> float:
        return float(np.sqrt(2) / 4)  # ≈ 0.354, distance from center to edge

    def modify(self, **params) -> DiamondSymbol:
        """Return self (diamond has no modifiable parameters)."""
        return self


class PentagonSymbol(Symbol):
    """Regular pentagon inscribed in unit circle (radius = 0.5).

    Parameters
    ----------
    pointy_top : bool
        If True (default), pentagon has a vertex at top.
        If False, has a flat edge at top.

    """

    def __init__(self, pointy_top: bool = True) -> None:
        self.pointy_top = pointy_top

    def unit_polygon(self) -> Polygon:
        offset = np.pi / 2 if self.pointy_top else 0
        angles = np.linspace(0, 2 * np.pi, 6)[:-1] + offset
        x = 0.5 * np.cos(angles)
        y = 0.5 * np.sin(angles)
        return Polygon(zip(x, y, strict=True))

    @property
    def bounding_radius(self) -> float:
        return 0.5

    def modify(self, pointy_top: bool | None = None, **params) -> PentagonSymbol:
        """Return new symbol with modified orientation.

        Parameters
        ----------
        pointy_top : bool or None
            If provided, set the orientation.

        """
        if pointy_top is None or pointy_top == self.pointy_top:
            return self
        return PentagonSymbol(pointy_top=pointy_top)


class StarSymbol(Symbol):
    """N-pointed star inscribed in unit circle (outer radius = 0.5).

    Parameters
    ----------
    n_points : int
        Number of star points. Default is 5.
    inner_radius_ratio : float
        Ratio of inner (valley) radius to outer radius. Default is 0.4.

    """

    def __init__(self, n_points: int = 5, inner_radius_ratio: float = 0.4) -> None:
        self.n_points = n_points
        self.inner_radius_ratio = inner_radius_ratio

    def unit_polygon(self) -> Polygon:
        n = self.n_points
        outer = 0.5
        inner = 0.5 * self.inner_radius_ratio
        angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 2 * n, endpoint=False)
        radii = np.where(np.arange(2 * n) % 2 == 0, outer, inner)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return Polygon(zip(x, y, strict=True))

    @property
    def bounding_radius(self) -> float:
        return 0.5

    def modify(
        self,
        n_points: int | None = None,
        inner_radius_ratio: float | None = None,
        **params,
    ) -> StarSymbol:
        """Return new symbol with modified parameters.

        Parameters
        ----------
        n_points : int or None
            New number of points. If None, keeps current.
        inner_radius_ratio : float or None
            New inner radius ratio. If None, keeps current.

        """
        new_n = n_points if n_points is not None else self.n_points
        new_ir = inner_radius_ratio if inner_radius_ratio is not None else self.inner_radius_ratio
        if new_n == self.n_points and new_ir == self.inner_radius_ratio:
            return self
        return StarSymbol(n_points=new_n, inner_radius_ratio=new_ir)


class TileSymbol(Symbol):
    """Symbol whose shape matches a tiling's canonical tile.

    Normalizes an arbitrary polygon to fit in the unit square with
    bounding radius = 0.5, centered at the origin.

    Parameters
    ----------
    polygon : Polygon
        The tile polygon (at any scale). Centroid is translated to the
        origin and the polygon is scaled so that the bounding radius
        (max distance from center to vertex) equals 0.5.

    """

    def __init__(self, polygon: Polygon) -> None:
        cx, cy = polygon.centroid.x, polygon.centroid.y
        coords = np.array(polygon.exterior.coords)
        centered = coords - np.array([cx, cy])
        max_r = float(np.max(np.linalg.norm(centered, axis=1)))
        if max_r < 1e-12:
            raise ValueError("Polygon has zero extent")
        factor = 0.5 / max_r
        self._unit = Polygon(centered * factor)
        self._bounding_r = 0.5
        self._inscribed_r = float(self._unit.exterior.distance(Point(0, 0)))

    def unit_polygon(self) -> Polygon:
        return self._unit

    @property
    def bounding_radius(self) -> float:
        return self._bounding_r

    @property
    def inscribed_radius(self) -> float:
        return self._inscribed_r

    def modify(self, **params) -> Symbol:
        """Return self (TileSymbol does not support modification)."""
        return self


class IsohedralTileSymbol(Symbol):
    """Isohedral tiling tile as symbol with modifiable parameters/curves.

    Unlike TileSymbol which just wraps a polygon, this class wraps
    IsohedralTiling and supports:
    - Prototile parameter modification
    - Edge curve customization
    - Immutable modify() pattern for styling

    The symbol is normalized to have bounding_radius = 0.5, matching
    the convention of other symbols (TileSymbol, HexagonSymbol, etc.).

    Parameters
    ----------
    tiling_type : int
        Isohedral type number (IH1-IH93).
    prototile_params : list[float], optional
        Free parameters for the tiling type.
    edge_curves : dict[int, list[tuple]], optional
        Custom edge curves keyed by shape ID.

    """

    def __init__(
        self,
        tiling_type: int,
        prototile_params: list[float] | None = None,
        edge_curves: dict[int, list[tuple[float, float]]] | None = None,
        edge_curve_modes: dict[int, str] | None = None,
    ) -> None:
        self.tiling_type = tiling_type
        self.prototile_params = prototile_params
        self.edge_curves = edge_curves or {}
        self.edge_curve_modes = edge_curve_modes or {}

        # Import here to avoid circular import
        from .tiling import IsohedralTiling

        # Delegate to IsohedralTiling for geometry
        self._tiling = IsohedralTiling(
            tiling_type,
            parameters=prototile_params,
            edge_curves=edge_curves,
            edge_curve_modes=edge_curve_modes,
        )

        # Normalize to bounding_radius = 0.5 (like TileSymbol)
        unit_tile = self._tiling.canonical_tile
        coords = np.array(unit_tile.exterior.coords)
        max_r = float(np.max(np.linalg.norm(coords, axis=1)))
        if max_r < 1e-12:
            raise ValueError("Isohedral tile has zero extent")
        self._scale_factor = 0.5 / max_r
        self._bounding_r = 0.5
        self._inscribed_r = float(unit_tile.exterior.distance(Point(0, 0))) * self._scale_factor

    def unit_polygon(self) -> Polygon:
        """Return normalized prototile with bounding_radius = 0.5."""
        # Scale the canonical tile to have bounding_radius = 0.5
        return scale(self._tiling.canonical_tile, xfact=self._scale_factor, yfact=self._scale_factor, origin=(0, 0))

    @property
    def bounding_radius(self) -> float:
        """Bounding radius is always 0.5 after normalization."""
        return self._bounding_r

    @property
    def inscribed_radius(self) -> float:
        """Inscribed radius after normalization."""
        return self._inscribed_r

    def modify(
        self,
        prototile_params: list[float] | None = None,
        edge_curves: dict[int, list[tuple[float, float]]] | None = None,
        edge_curve_modes: dict[int, str] | None = None,
        **params,
    ) -> IsohedralTileSymbol:
        """Return new symbol with modified parameters.

        Merges new parameters/curves with existing ones.

        Parameters
        ----------
        prototile_params : list[float] or None
            New prototile parameters (replaces existing).
        edge_curves : dict or None
            New edge curves (merged with existing).
        edge_curve_modes : dict or None
            New edge curve modes (merged with existing).

        Returns
        -------
        IsohedralTileSymbol
            New symbol with modified parameters.

        """
        new_params = prototile_params if prototile_params is not None else self.prototile_params
        new_curves = {**self.edge_curves, **(edge_curves or {})}
        new_modes = {**self.edge_curve_modes, **(edge_curve_modes or {})}
        return IsohedralTileSymbol(self.tiling_type, new_params, new_curves, new_modes)


class TransformedSymbol(Symbol):
    """Wrapper that applies rotation/reflection to a base symbol.

    The transformation is applied BEFORE fitting, so the fit is computed
    on the transformed shape. This is useful for FILL mode where a rotated
    symbol (e.g., a diamond) needs different fit characteristics than the
    base symbol (e.g., a square).

    Parameters
    ----------
    base : Symbol
        The base symbol to transform.
    rotation : float
        Rotation angle in degrees (counter-clockwise). Applied before
        reflection.
    reflection : bool
        Whether to reflect the symbol about the vertical axis. Applied
        after rotation.

    Examples
    --------
    Create a diamond (45° rotated square):

    >>> diamond = TransformedSymbol(SquareSymbol(), rotation=45)
    >>> styling = Styling(symbol=diamond, fit_mode='fill')

    Create a mirrored hexagon:

    >>> mirrored = TransformedSymbol(HexagonSymbol(), reflection=True)

    """

    def __init__(
        self,
        base: Symbol,
        rotation: float = 0.0,
        reflection: bool = False,
    ) -> None:
        self.base = base
        self.rotation = rotation  # degrees
        self.reflection = reflection
        self._cache_unit_polygon()

    def _cache_unit_polygon(self) -> None:
        """Compute and cache the transformed unit polygon and radii."""
        p = self.base.unit_polygon()
        if self.reflection:
            p = scale(p, xfact=-1, yfact=1, origin=(0, 0))
        if self.rotation != 0.0:
            p = rotate(p, self.rotation, origin=(0, 0))  # rotate() uses degrees
        self._unit = p

        # Compute bounding and inscribed radii from transformed polygon
        coords = np.array(p.exterior.coords)
        self._bounding_r = float(np.max(np.linalg.norm(coords, axis=1)))
        self._inscribed_r = float(p.exterior.distance(Point(0, 0)))

    def unit_polygon(self) -> Polygon:
        """Return the transformed unit polygon."""
        return self._unit

    @property
    def bounding_radius(self) -> float:
        """Bounding radius of transformed symbol."""
        return self._bounding_r

    @property
    def inscribed_radius(self) -> float:
        """Inscribed radius of transformed symbol."""
        return self._inscribed_r

    def modify(
        self,
        rotation: float | None = None,
        reflection: bool | None = None,
        **params,
    ) -> TransformedSymbol:
        """Return new symbol with modified transformation.

        Parameters
        ----------
        rotation : float or None
            New rotation angle in degrees. If None, keeps current.
        reflection : bool or None
            New reflection flag. If None, keeps current.

        Returns
        -------
        TransformedSymbol
            New symbol with modified transformation.

        """
        new_rotation = rotation if rotation is not None else self.rotation
        new_reflection = reflection if reflection is not None else self.reflection
        if new_rotation == self.rotation and new_reflection == self.reflection:
            return self
        return TransformedSymbol(self.base, new_rotation, new_reflection)


# ---------------------------------------------------------------------------
# Registry and resolution
# ---------------------------------------------------------------------------

BUILTIN_SYMBOLS: dict[str, Symbol] = {
    "circle": CircleSymbol(),
    "square": SquareSymbol(),
    "hexagon": HexagonSymbol(),
    "triangle": TriangleSymbol(),
    "diamond": DiamondSymbol(),
    "pentagon": PentagonSymbol(),
    "star": StarSymbol(),
}

# Type aliases
SymbolSpec = str | Symbol
SymbolParam = SymbolSpec | Sequence[SymbolSpec]


def resolve_symbol(spec: SymbolSpec) -> Symbol:
    """Resolve a symbol specification to a :class:`Symbol` instance.

    Parameters
    ----------
    spec : str or Symbol
        A built-in name (``"circle"``, ``"square"``, ``"hexagon"``)
        or a :class:`Symbol` instance.

    Returns
    -------
    Symbol

    Raises
    ------
    ValueError
        If *spec* is a string that doesn't match a built-in symbol.
    TypeError
        If *spec* is not a string or :class:`Symbol`.

    """
    if isinstance(spec, Symbol):
        return spec
    if isinstance(spec, str):
        if spec not in BUILTIN_SYMBOLS:
            raise ValueError(f"Unknown symbol: {spec!r}. Valid built-in symbols: {list(BUILTIN_SYMBOLS)}")
        return BUILTIN_SYMBOLS[spec]
    raise TypeError(f"symbol must be a str or Symbol instance, got {type(spec).__name__}")


# ---------------------------------------------------------------------------
# Batch creation
# ---------------------------------------------------------------------------


def _get_size(sizes: NDArray[np.floating], i: int) -> float | tuple[float, float]:
    """Extract size for symbol *i*: scalar from (n,) or tuple from (n,2)."""
    if sizes.ndim == 2:
        return (float(sizes[i, 0]), float(sizes[i, 1]))
    return float(sizes[i])


def create_symbols(
    positions: NDArray[np.floating],
    sizes: NDArray[np.floating],
    symbol: SymbolParam,
    rotations: NDArray[np.floating] | None = None,
    flipped: NDArray[np.bool_] | None = None,
) -> list[Polygon]:
    """Create symbol polygons, supporting per-geometry specification.

    Parameters
    ----------
    positions : np.ndarray of shape (n, 2)
        Symbol center positions.
    sizes : np.ndarray of shape (n,) or (n, 2)
        Symbol sizes (isotropic or anisotropic).
    symbol : str, Symbol, or sequence thereof
        A single symbol specification applied to all geometries, or a
        sequence of length *n* for per-geometry symbols.
    rotations : np.ndarray of shape (n,), optional
        Per-symbol rotation angles in degrees. If None, no rotation.
    flipped : np.ndarray of shape (n,), optional
        Per-symbol flip flags. If None, no flipping.

    Returns
    -------
    list[Polygon]

    """
    n = len(positions)

    def _get_rot(i: int) -> float:
        return float(rotations[i]) if rotations is not None else 0.0

    def _get_flip(i: int) -> bool:
        return bool(flipped[i]) if flipped is not None else False

    if isinstance(symbol, Sequence) and not isinstance(symbol, str):
        if len(symbol) != n:
            raise ValueError(f"symbol sequence length ({len(symbol)}) != number of positions ({n})")
        resolved = [resolve_symbol(s) for s in symbol]
        return [
            resolved[i].create(
                tuple(positions[i]),
                _get_size(sizes, i),
                rotation=_get_rot(i),
                flipped=_get_flip(i),
            )
            for i in range(n)
        ]
    sym = resolve_symbol(symbol)
    return [
        sym.create(
            tuple(positions[i]),
            _get_size(sizes, i),
            rotation=_get_rot(i),
            flipped=_get_flip(i),
        )
        for i in range(n)
    ]
