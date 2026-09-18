"""Configuration options for symbol cartogram generation."""

from __future__ import annotations

from enum import Enum


class SymbolShape(str, Enum):
    """Shape of the symbols."""

    CIRCLE = "circle"
    SQUARE = "square"
    HEXAGON = "hexagon"


class AdjacencyMode(str, Enum):
    """How adjacency is computed."""

    BINARY = "binary"  # Adjacent or not (0/1)
    WEIGHTED = "weighted"  # Fraction of perimeter shared (asymmetric)
    AREA_WEIGHTED = "area_weighted"  # Neighbor area fraction (rows sum to 1)


class SymbolOrientation(str, Enum):
    """How symbols are oriented relative to their tile."""

    UPRIGHT = "upright"  # Symbol stays axis-aligned regardless of tile
    WITH_TILE = "with_tile"  # Symbol rotates/flips with its tile


class ForceMode(str, Enum):
    """How attraction force magnitude is computed.

    Applies to both origin attraction force and global centroid attraction force.
    """

    DIRECTION = "direction"  # Constant magnitude with drop-off near target (default)
    LINEAR = "linear"  # Force proportional to distance (spring)
    NORMALIZED = "normalized"  # Force proportional to distance / radius
