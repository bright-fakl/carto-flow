"""Layout result classes for symbol cartograms.

This module defines the immutable LayoutResult that captures the output
of layout algorithms, and the Transform dataclass for per-geometry transforms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .symbols import Symbol


@dataclass(frozen=True)
class Transform:
    """Transformation applied to a symbol.

    Note: This is an internal class that stores rotation in radians.
    User-facing APIs (Styling, TransformedSymbol) use degrees for convenience.

    Attributes
    ----------
    position : tuple[float, float]
        Center position (x, y) for the symbol.
    rotation : float
        Rotation angle in radians (counter-clockwise). Default: 0.0
        User-facing APIs accept degrees and convert internally.
    scale : float
        Scale multiplier. Default: 1.0
    reflection : bool
        Whether to reflect the symbol about the vertical axis. Default: False

    """

    position: tuple[float, float] = (0.0, 0.0)
    rotation: float = 0.0
    scale: float = 1.0
    reflection: bool = False

    def compose(self, other: Transform) -> Transform:
        """Compose this transform with another.

        The resulting transform applies this transform first, then the other.

        Parameters
        ----------
        other : Transform
            Transform to apply after this one.

        Returns
        -------
        Transform
            Composed transform.

        """
        # Compose positions (add translation)
        new_x = self.position[0] + other.position[0]
        new_y = self.position[1] + other.position[1]

        # Compose rotations (add angles)
        new_rotation = self.rotation + other.rotation

        # Compose scales (multiply)
        new_scale = self.scale * other.scale

        # Compose reflections (XOR)
        new_reflection = self.reflection != other.reflection

        return Transform(
            position=(new_x, new_y),
            rotation=new_rotation,
            scale=new_scale,
            reflection=new_reflection,
        )


@dataclass
class LayoutResult:
    """Immutable output from layout algorithm.

    Contains canonical symbol and per-geometry transforms, plus the
    serializable preprocessing data (positions, sizes, adjacency, bounds, crs).
    The original GeoDataFrame is NOT stored here (too large for serialization).

    Attributes
    ----------
    canonical_symbol : Symbol
        The base symbol shape for this layout.
    transforms : list[Transform]
        One transform per input geometry.
    base_size : float
        Reference size for scaling.
    positions : NDArray[np.floating]
        Original centroid positions, shape (n, 2).
    sizes : NDArray[np.floating]
        Symbol sizes, shape (n,).
    adjacency : NDArray[np.floating]
        Adjacency matrix from original geometries, shape (n, n).
    bounds : tuple[float, float, float, float]
        Geographic bounds (xmin, ymin, xmax, ymax).
    crs : str | None
        CRS information as WKT string from source GeoDataFrame.
    algorithm_info : dict
        Algorithm-specific metadata.
    simulation_history : SimulationHistory | None
        Per-iteration diagnostics and optional position snapshots.

    """

    canonical_symbol: Symbol
    transforms: list[Transform]
    base_size: float
    positions: NDArray[np.floating]
    sizes: NDArray[np.floating]
    adjacency: NDArray[np.floating]
    bounds: tuple[float, float, float, float]
    crs: str | None = None  # WKT string to preserve projection info
    algorithm_info: dict[str, Any] = field(default_factory=dict)
    simulation_history: Any = None
    valid_mask: NDArray[np.bool_] | None = None

    def style(
        self,
        styling: Styling | None = None,
        **kwargs,
    ) -> SymbolCartogram:
        """Apply styling to create symbol cartogram.

        Parameters
        ----------
        styling : Styling or None
            Pre-configured Styling object.
        **kwargs
            Convenience kwargs for simple cases (symbol, scale, etc.)
            Creates a temporary Styling object internally.

        Returns
        -------
        SymbolCartogram
            Rendered cartogram with styled symbols.

        """
        # Import here to avoid circular import
        from .styling import Styling

        if styling is None:
            styling = Styling(**kwargs)
        return styling.apply(self)

    def serialize(self) -> dict[str, Any]:
        """Serialize for persistence.

        Returns
        -------
        dict
            Serialized representation of the layout result.

        Notes
        -----
        All data is serialized including positions, sizes, adjacency, bounds, crs.
        The original GeoDataFrame is NOT stored - it must be provided separately
        when attribute merging is needed via SymbolCartogram.to_geodataframe().

        Examples
        --------
        >>> import json
        >>> result = create_layout(gdf, "population")
        >>> serialized = result.serialize()
        >>> json.dump(serialized, open("layout.json", "w"))

        """
        # Serialize transforms
        transforms_data = [
            {
                "position": list(t.position),
                "rotation": t.rotation,
                "scale": t.scale,
                "reflection": t.reflection,
            }
            for t in self.transforms
        ]

        # Serialize canonical symbol (by class name and params)
        symbol_class = type(self.canonical_symbol).__name__
        symbol_params: dict[str, Any] = {}

        # Handle different symbol types
        if hasattr(self.canonical_symbol, "pointy_top"):
            symbol_params["pointy_top"] = self.canonical_symbol.pointy_top
        if hasattr(self.canonical_symbol, "tiling_type"):
            # IsohedralTileSymbol
            symbol_params["tiling_type"] = self.canonical_symbol.tiling_type
            if (
                hasattr(self.canonical_symbol, "prototile_params")
                and self.canonical_symbol.prototile_params is not None
            ):
                symbol_params["prototile_params"] = self.canonical_symbol.prototile_params
            if hasattr(self.canonical_symbol, "edge_curves") and self.canonical_symbol.edge_curves:
                symbol_params["edge_curves"] = self.canonical_symbol.edge_curves

        # Filter algorithm_info to only JSON-safe scalar values.
        # Non-serializable objects (TilingResult, numpy arrays) are stripped here
        # and handled separately at the SymbolCartogram level.
        safe_info = {
            k: v for k, v in self.algorithm_info.items() if isinstance(v, str | int | float | bool | type(None))
        }

        return {
            "canonical_symbol": {
                "class": symbol_class,
                "params": symbol_params,
            },
            "transforms": transforms_data,
            "base_size": self.base_size,
            "positions": self.positions.tolist(),
            "sizes": self.sizes.tolist(),
            "adjacency": self.adjacency.tolist(),
            "bounds": list(self.bounds),
            "crs": self.crs,
            "algorithm_info": safe_info,
        }

    @classmethod
    def from_serialized(cls, data: dict[str, Any]) -> LayoutResult:
        """Reconstruct from serialized data.

        Parameters
        ----------
        data : dict
            Serialized layout result from serialize().

        Returns
        -------
        LayoutResult
            Reconstructed layout result.

        Examples
        --------
        >>> import json
        >>> serialized = json.load(open("layout.json"))
        >>> result = LayoutResult.from_serialized(serialized)
        >>> cartogram = result.style(symbol="circle")

        """
        from .symbols import (
            CircleSymbol,
            HexagonSymbol,
            IsohedralTileSymbol,
            SquareSymbol,
        )

        # Reconstruct canonical symbol
        symbol_data = data["canonical_symbol"]
        symbol_class_name = symbol_data["class"]
        symbol_params = symbol_data.get("params", {})

        symbol_classes = {
            "CircleSymbol": CircleSymbol,
            "SquareSymbol": SquareSymbol,
            "HexagonSymbol": HexagonSymbol,
            "IsohedralTileSymbol": IsohedralTileSymbol,
        }

        symbol_cls = symbol_classes.get(symbol_class_name)
        if symbol_cls is None:
            raise ValueError(f"Unknown symbol class: {symbol_class_name}")

        # Handle IsohedralTileSymbol specially
        if symbol_class_name == "IsohedralTileSymbol":
            tiling_type = symbol_params.get("tiling_type", 1)
            prototile_params = symbol_params.get("prototile_params")
            edge_curves = symbol_params.get("edge_curves")
            canonical_symbol = IsohedralTileSymbol(
                tiling_type=tiling_type,
                prototile_params=prototile_params,
                edge_curves=edge_curves,
            )
        else:
            canonical_symbol = symbol_cls(**symbol_params)

        # Reconstruct transforms
        transforms = [
            Transform(
                position=tuple(t["position"]),
                rotation=t["rotation"],
                scale=t["scale"],
                reflection=t["reflection"],
            )
            for t in data["transforms"]
        ]

        # Reconstruct arrays
        positions = np.array(data["positions"], dtype=float)
        sizes = np.array(data["sizes"], dtype=float)
        adjacency = np.array(data["adjacency"], dtype=float)
        bounds = tuple(data["bounds"])
        crs = data.get("crs")

        return cls(
            canonical_symbol=canonical_symbol,
            transforms=transforms,
            base_size=data["base_size"],
            positions=positions,
            sizes=sizes,
            adjacency=adjacency,
            bounds=bounds,
            crs=crs,
            algorithm_info=data.get("algorithm_info", {}),
        )


# Import at end to avoid circular import
if TYPE_CHECKING:
    from .result import SymbolCartogram
    from .styling import Styling
