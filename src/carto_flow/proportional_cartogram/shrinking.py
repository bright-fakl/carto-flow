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

import numpy as np
import shapely
from scipy.optimize import brentq
from shapely.geometry.polygon import orient

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

__all__ = ["shrink"]


def _isotropic_frame(geom: BaseGeometry) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Centroid and the linear map ``T`` (with inverse) that equalizes the spread of ``geom``.

    ``T`` is the inverse square root of the covariance matrix of the area
    distribution, so the mapped geometry has the same extent in every
    direction. Returns ``None`` when the covariance is degenerate.
    """
    area = cx = cy = ixx = iyy = ixy = 0.0
    for part in shapely.get_parts(geom):
        if part.geom_type != "Polygon" or part.is_empty:
            continue
        part = orient(part, 1.0)
        for ring in (part.exterior, *part.interiors):
            x0, y0 = np.asarray(ring.coords)[:-1].T
            x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
            cross = x0 * y1 - x1 * y0
            area += cross.sum() / 2
            cx += ((x0 + x1) * cross).sum() / 6
            cy += ((y0 + y1) * cross).sum() / 6
            ixx += ((x0**2 + x0 * x1 + x1**2) * cross).sum() / 12
            iyy += ((y0**2 + y0 * y1 + y1**2) * cross).sum() / 12
            ixy += ((x0 * y1 + 2 * x0 * y0 + 2 * x1 * y1 + x1 * y0) * cross).sum() / 24
    if not area > 0:
        return None
    center = np.array([cx / area, cy / area])
    cov = np.array([[ixx / area, ixy / area], [ixy / area, iyy / area]]) - np.outer(center, center)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if not eigenvalues[0] > 1e-12 * eigenvalues[1]:
        return None
    forward = eigenvectors @ np.diag(eigenvalues**-0.5) @ eigenvectors.T
    inverse = eigenvectors @ np.diag(eigenvalues**0.5) @ eigenvectors.T
    return center, forward, inverse


def _shrink_single(
    geom: BaseGeometry,
    fraction: float,
    simplify: float | None = None,
    mode: Literal["area", "shell"] = "area",
    tol: float = 0.01,
    isotropic: bool = False,
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
        Simplification tolerance (Visvalingam-Whyatt via coverage_simplify).
    mode : {'area', 'shell'}
        'area' for direct fraction, 'shell' squares the fraction.
    tol : float
        Relative tolerance on the area of the shrunken geometry.
    isotropic : bool
        Erode in a frame where the geometry has equal spread in all directions.

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
    working_geom = shapely.coverage_simplify(geom, simplify) if simplify else geom

    # Optionally erode in a frame where the geometry is isotropic. A linear map
    # scales all areas by the same factor, so the area fraction is unchanged.
    frame = _isotropic_frame(working_geom) if isotropic else None
    solve_geom = working_geom
    if frame is not None:
        center, forward, _ = frame
        solve_geom = shapely.transform(working_geom, lambda points: (points - center) @ forward.T)

    # Compute target area
    target_area = fraction * solve_geom.area

    # Erosion by half the shortest bounding-box side removes the whole geometry
    xmin, ymin, xmax, ymax = solve_geom.bounds
    shortest_edge = min(xmax - xmin, ymax - ymin)

    # Bracketed solve on the relative area residual, which is -1 for a collapsed
    # geometry and 1/fraction - 1 > 0 at zero buffer. The residual is reported
    # as 0 once it is below tol, which ends the solve at that buffer distance.
    best_residual = float("inf")
    best_geom = solve_geom

    def residual(buffer: float) -> float:
        nonlocal best_residual, best_geom
        candidate = solve_geom.buffer(buffer)
        value = candidate.area / target_area - 1.0
        if abs(value) < abs(best_residual):
            best_residual, best_geom = value, candidate
        return 0.0 if abs(value) < tol else value

    brentq(residual, -shortest_edge / 2.0, 0.0, xtol=shortest_edge * 1e-12)
    if abs(best_residual) >= tol:
        warnings.warn(
            f"shrink reached area error {best_residual:+.2e} for fraction {fraction}, above tol={tol}.",
            UserWarning,
            stacklevel=3,
        )

    shrunken_geom = best_geom
    if frame is not None:
        center, _, inverse = frame
        shrunken_geom = shapely.transform(best_geom, lambda points: points @ inverse.T + center)
    shell_geom = working_geom.difference(shrunken_geom)
    return shrunken_geom, shell_geom


def shrink(
    geom: BaseGeometry,
    fractions: float | Sequence[float],
    simplify: float | None = None,
    mode: Literal["area", "shell"] = "area",
    tol: float = 0.01,
    isotropic: bool = False,
) -> list[BaseGeometry]:
    """
    Shrink a geometry to create concentric shells with specified area fractions.

    This function reduces a geometry's area by creating one or more concentric
    shells. For N fractions, it creates N parts: 1 innermost core plus N-1
    shells expanding outward.

    Parameters
    ----------
    geom : shapely.geometry.base.BaseGeometry
        Input geometry to shrink. Can be Polygon, MultiPolygon, or any
        geometry that supports buffer operations.
    fractions : float or Sequence[float]
        Target area fractions for each part.

        - **Single float**: Shrink to one target fraction, returning [core, shell]
          where core has area fraction and shell has area (1-fraction).
          Must be in range [0, 1].
        - **Sequence of floats**: Create N parts. Values represent the area
          fraction of each part from inside to outside (core first).
          Values should be non-negative and sum to approximately 1.0.
          Example: [0.25, 0.25, 0.25, 0.25] creates 1 core + 3 shells, each 25%.
    simplify : float, optional
        Simplification tolerance (Visvalingam-Whyatt via ``shapely.coverage_simplify``).
        Applied before shrinking to reduce numerical artifacts from highly detailed boundaries.
    mode : {'area', 'shell'}, default='area'
        Interpretation mode for fractions:

        - **'area'**: Fractions represent direct area ratios
        - **'shell'**: Fractions represent shell thickness ratios (squared for area)
    tol : float, default=0.01
        Relative tolerance on the area of each shrunken part: the buffer
        distance is refined until ``abs(area / target_area - 1) < tol``.
        A warning is issued if the tolerance cannot be reached.
    isotropic : bool, default=False
        Erode in a coordinate frame where the geometry has equal spread in all
        directions (a linear map by the inverse square root of its area
        covariance, undone afterwards). An elongated geometry then shrinks to
        a part with the proportions of the original instead of a thin strip
        along its long axis. Parts are still intersections of the original
        geometry, so its outer boundary is unchanged. Shapes with strongly
        curved or concave boundaries can still produce thin parts.

    Returns
    -------
    list[BaseGeometry]
        List of geometry parts from innermost to outermost. Each part
        corresponds positionally to its input fraction (``fractions[i]``
        maps to ``parts[i]``), consistent with :func:`split`.

        - For single float f: returns [core, shell] where core has area
          f*original and shell has area (1-f)*original.
        - For sequence of N values: returns N geometries [part_0, ..., part_{N-1}]
          where part_0 is the innermost core and part_{N-1} is the outermost shell.
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
    >>> from carto_flow.proportional_cartogram import shrink
    >>>
    >>> square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    >>> parts = shrink(square, 0.5)
    >>> len(parts)  # [core, shell]
    2
    >>> print(f"Core area: {parts[0].area:.1f}")  # 50.0
    >>> print(f"Shell area: {parts[1].area:.1f}")  # 50.0

    Create concentric shells with equal fractions:

    >>> # 4 equal parts: core + 3 shells (25% each)
    >>> parts = shrink(square, [0.25, 0.25, 0.25, 0.25])
    >>> len(parts)  # 4 parts
    4
    >>> print(f"Areas: {[round(p.area, 1) for p in parts]}")  # [25.0, 25.0, 25.0, 25.0]

    Unequal shells (core first):

    >>> # Core 20%, middle shell 30%, outer shell 50%
    >>> parts = shrink(square, [0.2, 0.3, 0.5])
    >>> print(f"Core area: {parts[0].area:.1f}")  # 20.0

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
        core, shell = _shrink_single(geom, fraction, simplify=simplify, mode=mode, tol=tol, isotropic=isotropic)
        # Return [core, shell] - core has area fraction, shell has area (1-fraction)
        return [core, shell]

    # Convert to list for easier manipulation
    frac_list = list(fractions)

    # Validate fractions
    if not frac_list:
        raise ValueError("fractions sequence cannot be empty")

    if len(frac_list) == 1:
        # Single fraction in sequence - same as scalar
        return shrink(geom, frac_list[0], simplify=simplify, mode=mode, tol=tol, isotropic=isotropic)

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

    # Fractions are ordered core-first (innermost to outermost).
    # The internal algorithm peels shells from outside in, so reverse the
    # input for processing and reverse the output to restore core-first order.
    frac_list = list(reversed(frac_list))

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
            shrunken, shell = _shrink_single(
                current_geom, shrink_to_frac, simplify=simplify, mode=mode, tol=tol, isotropic=isotropic
            )
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

    # Reverse to return [core, inner_shell, ..., outer_shell]
    return list(reversed(parts))
