"""
Geometry shrinking utilities.

This module provides functions for shrinking geometries to create concentric
shells with specified area fractions. Uses numerical root finding to achieve
precise area targets.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from scipy.optimize import root_scalar

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

__all__ = ["shrink"]


def _shrunken_area(buffer: float, geom: BaseGeometry, target_area: float) -> float:
    """Compute difference between buffered area and target area."""
    return geom.buffer(buffer).area - target_area


def _shrink_single(
    geom: BaseGeometry,
    fraction: float,
    simplify: float | None = None,
    mode: Literal["area", "shell"] = "area",
    tol: float = 0.05,
) -> tuple[BaseGeometry, BaseGeometry]:
    """
    Internal: Shrink a geometry to a specified area fraction.

    Uses negative buffer with root finding optimization. Called by shrink() for
    single-fraction operations.

    Parameters
    ----------
    geom : BaseGeometry
        Input geometry to shrink.
    fraction : float
        Target area fraction in [0, 1].
    simplify : float, optional
        Simplification tolerance (Douglas-Peucker).
    mode : {'area', 'shell'}
        'area' for direct fraction, 'shell' squares the fraction.
    tol : float
        Root finding tolerance.

    Returns
    -------
    tuple[BaseGeometry, BaseGeometry]
        (shrunken_geometry, shell_geometry)
    """
    # Input validation for fraction
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"fraction must be in range [0, 1], got {fraction}")

    # Handle edge cases
    if fraction == 0.0:
        # Return empty geometry of the same type
        # Strategy: First expand slightly, then shrink massively to guarantee empty result
        # Direct large negative buffer on some geometries may not fully eliminate them
        # due to floating point precision or complex boundary conditions
        shrunken_geom = geom.buffer(1e-10).buffer(-1e10)  # Create empty geometry
        shell_geom = geom  # Shell is the entire original geometry
        return shrunken_geom, shell_geom
    elif fraction == 1.0:
        shrunken_geom = geom  # Return original geometry unchanged
        shell_geom = geom.buffer(1e-10).buffer(-1e10)  # Create empty geometry
        return shrunken_geom, shell_geom

    if mode == "shell":
        fraction = fraction**2

    # Input validation for simplify
    if simplify is not None:
        if not isinstance(simplify, (int, float)) or simplify <= 0:
            raise ValueError(f"simplify must be a positive number, got {simplify}")
        # Check if simplify is reasonable compared to geometry size
        xmin, ymin, xmax, ymax = geom.bounds
        shortest_edge = min(xmax - xmin, ymax - ymin)
        if simplify > shortest_edge * 0.5:  # More than 50% of shortest edge
            warnings.warn(
                f"simplify tolerance ({simplify}) is large compared to geometry size "
                f"({shortest_edge}). Consider using a smaller value.",
                UserWarning,
                stacklevel=2,
            )

    # Apply simplification if requested
    working_geom = geom.simplify(simplify, preserve_topology=True) if simplify else geom

    # Compute target area
    target_area = fraction * working_geom.area

    # Improved buffer range and starting point
    xmin, ymin, xmax, ymax = working_geom.bounds
    width = xmax - xmin
    height = ymax - ymin
    shortest_edge = min(width, height)

    # Conservative bracket: from 0 to -shortest_edge/2
    # This ensures we don't shrink more than half the shortest dimension
    bracket_left = -shortest_edge / 2.0
    bracket_right = 0.0

    # Better starting point: use a fraction of the expected buffer distance
    # For area reduction, buffer distance is typically negative and proportional to sqrt(area_ratio)
    expected_buffer_magnitude = shortest_edge * (1.0 - (fraction**0.5)) * 0.5
    x0 = -expected_buffer_magnitude  # Start with negative buffer

    try:
        result = root_scalar(
            _shrunken_area,
            args=(working_geom, target_area),
            x0=x0,
            bracket=(bracket_left, bracket_right),
            rtol=tol,
        )
    except ValueError as e:
        # Handle root finding failures
        if "bracket" in str(e).lower():
            warnings.warn(
                f"Could not find valid bracket for root finding. Using fallback buffer distance. Error: {e}",
                UserWarning,
                stacklevel=2,
            )
            # Fallback: use estimated buffer distance
            fallback_buffer = -shortest_edge * (1.0 - fraction**0.5) * 0.3
            shrunken_geom = working_geom.buffer(fallback_buffer)
            shell_geom = working_geom.difference(shrunken_geom)
            return shrunken_geom, shell_geom
        else:
            raise
    else:
        # Validate result
        if not result.converged:
            warnings.warn(
                f"Root finding did not converge. Final area may not match target exactly. "
                f"Convergence flag: {result.flag}",
                UserWarning,
                stacklevel=2,
            )

        shrunken_geom = working_geom.buffer(result.root)
        shell_geom = working_geom.difference(shrunken_geom)
        return shrunken_geom, shell_geom


def shrink(
    geom: BaseGeometry,
    fractions: float | Sequence[float],
    simplify: float | None = None,
    mode: Literal["area", "shell"] = "area",
    tol: float = 0.05,
) -> list[BaseGeometry]:
    """
    Shrink a geometry to create concentric shells with specified area fractions.

    This function reduces a geometry's area by creating one or more concentric
    shells. For N fractions, it creates N parts: N-1 shells from outside to
    inside, plus the innermost core.

    Parameters
    ----------
    geom : shapely.geometry.base.BaseGeometry
        Input geometry to shrink. Can be Polygon, MultiPolygon, or any
        geometry that supports buffer operations.
    fractions : float or Sequence[float]
        Target area fractions for each part.

        - **Single float**: Shrink to one target fraction, returning [shell, core]
          where shell has area (1-fraction) and core has area fraction.
          Must be in range [0, 1].
        - **Sequence of floats**: Create N parts. Values represent the area
          fraction of each part from outside to inside.
          Values should be non-negative and sum to approximately 1.0.
          Example: [0.25, 0.25, 0.25, 0.25] creates 3 shells + 1 core, each 25%.
    simplify : float, optional
        Simplification tolerance for the Douglas-Peucker algorithm.
        Applied before shrinking to reduce complexity.
    mode : {'area', 'shell'}, default='area'
        Interpretation mode for fractions:

        - **'area'**: Fractions represent direct area ratios
        - **'shell'**: Fractions represent shell thickness ratios (squared for area)
    tol : float, default=0.05
        Relative tolerance for the root finding algorithm.

    Returns
    -------
    list[BaseGeometry]
        List of geometry parts from outermost to innermost.

        - For single float f: returns [shell, core] where shell has area
          (1-f)*original and core has area f*original.
        - For sequence of N values: returns N geometries [part_0, ..., part_{N-1}]
          where part_0 is the outermost shell and part_{N-1} is the core.
        - Zero fractions produce empty geometries in the corresponding position.

    Raises
    ------
    ValueError
        If fractions are negative or don't sum to ~1.
    TypeError
        If geom is not a valid Shapely geometry

    Examples
    --------
    Single shrink (binary):

    >>> from shapely.geometry import Polygon
    >>> from carto_flow.shape_splitter import shrink
    >>>
    >>> square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    >>> parts = shrink(square, 0.5)
    >>> len(parts)  # [shell, core]
    2
    >>> print(f"Shell area: {parts[0].area:.1f}")  # 50.0
    >>> print(f"Core area: {parts[1].area:.1f}")  # 50.0

    Create concentric shells with equal fractions:

    >>> # 4 equal parts: 3 shells + core (25% each)
    >>> parts = shrink(square, [0.25, 0.25, 0.25, 0.25])
    >>> len(parts)  # 4 parts
    4
    >>> print(f"Areas: {[round(p.area, 1) for p in parts]}")  # [25.0, 25.0, 25.0, 25.0]

    Unequal shells:

    >>> # Outer shell 50%, middle shell 30%, core 20%
    >>> parts = shrink(square, [0.5, 0.3, 0.2])
    >>> print(f"Core area: {parts[-1].area:.1f}")  # 20.0

    Shell mode for thickness-based shrinking:

    >>> parts = shrink(square, [0.5, 0.5], mode='shell')
    >>> # Areas will be based on squared fractions
    """

    def _make_empty() -> BaseGeometry:
        """Create an empty geometry."""
        return geom.buffer(1e-10).buffer(-1e10)

    # Handle single fraction (binary shrink)
    if isinstance(fractions, (int, float)):
        fraction = float(fractions)
        # _shrink_single returns (shrunken_core, shell)
        core, shell = _shrink_single(geom, fraction, simplify=simplify, mode=mode, tol=tol)
        # Return [shell, core] - shell has area (1-fraction), core has area fraction
        return [shell, core]

    # Convert to list for easier manipulation
    frac_list = list(fractions)

    # Validate fractions
    if not frac_list:
        raise ValueError("fractions sequence cannot be empty")

    if len(frac_list) == 1:
        # Single fraction in sequence - same as scalar
        return shrink(geom, frac_list[0], simplify=simplify, mode=mode, tol=tol)

    for i, f in enumerate(frac_list):
        if f < 0.0:
            raise ValueError(f"All fractions must be non-negative, got {f} at index {i}")

    # Check sum is close to 1
    total = sum(frac_list)
    if total > 0 and not (0.99 <= total <= 1.01):
        warnings.warn(
            f"Fractions sum to {total:.3f}, not 1.0. Parts will be normalized to sum to total area.",
            stacklevel=2,
        )

    # Handle all-zero case
    if total == 0:
        return [_make_empty() for _ in frac_list]

    # Normalize fractions to sum to 1
    frac_list = [f / total for f in frac_list]

    # Create shells progressively from outside to inside
    # Each fraction (except the last) becomes a shell
    # The last fraction becomes the core
    parts: list[BaseGeometry] = []
    current_geom = geom
    remaining_fraction = 1.0

    for target_frac in frac_list[:-1]:  # All but last (which is core)
        # Handle near-zero fractions - produce empty shell
        if target_frac < 1e-9:
            parts.append(_make_empty())
            continue

        # Compute the target fraction to shrink TO (what remains after removing this shell)
        target_remaining = remaining_fraction - target_frac

        # Shrink to that fraction relative to current geometry
        # e.g., if we have 1.0 and want to remove 0.25, we shrink to 0.75/1.0 = 0.75
        shrink_to_frac = target_remaining / remaining_fraction

        # Clamp to valid range
        shrink_to_frac = max(0.001, min(0.999, shrink_to_frac))

        try:
            shrunken, shell = _shrink_single(current_geom, shrink_to_frac, simplify=simplify, mode=mode, tol=tol)
            parts.append(shell)
            current_geom = shrunken
            remaining_fraction = target_remaining
        except (ValueError, TypeError) as e:
            warnings.warn(
                f"Shrink operation failed at fraction {target_frac}: {e}",
                stacklevel=2,
            )
            # Add empty geometry for this shell and continue
            parts.append(_make_empty())

    # Add the core (final remaining geometry)
    parts.append(current_geom)

    return parts
