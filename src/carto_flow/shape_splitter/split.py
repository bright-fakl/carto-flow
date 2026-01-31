"""
Geometry splitting utilities.

This module provides functions for splitting geometries into parts with
specified area fractions. Supports both sequential and treemap splitting
strategies.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from shapely.geometry import box

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

__all__ = ["split"]


def _split_binary(
    geom: BaseGeometry,
    fraction: float,
    direction: Literal["vertical", "horizontal"] = "vertical",
    tol: float = 0.01,
) -> tuple[BaseGeometry, BaseGeometry]:
    """
    Internal: Split a geometry into two parts along a line.

    Uses root finding to optimize cut position for target area fraction.
    Called by split() for binary splitting operations.

    Parameters
    ----------
    geom : BaseGeometry
        Input geometry to split.
    fraction : float
        Target area fraction for first part, in (0, 1).
    direction : {'vertical', 'horizontal'}
        Direction of the splitting line.
    tol : float
        Absolute tolerance for area matching.

    Returns
    -------
    tuple[BaseGeometry, BaseGeometry]
        (part1, part2) where part1 has approximately fraction * original_area.
    """
    # Input validation
    if not (0.0 < fraction < 1.0):
        raise ValueError(f"fraction must be in range (0, 1), got {fraction}")

    if direction not in ["vertical", "horizontal"]:
        raise ValueError(f"direction must be 'vertical' or 'horizontal', got {direction}")

    # Get geometry bounds
    try:
        min_x, min_y, max_x, max_y = geom.bounds
    except AttributeError as err:
        raise TypeError("geom must be a Shapely geometry with bounds attribute") from err

    # Define objective function for optimization
    def area_difference(split_pos: float) -> float:
        """Compute difference between target area and actual area of one part."""
        try:
            # Create bounding box and use intersection for both Polygon and MultiPolygon
            if direction == "vertical":
                if split_pos <= min_x:
                    # Left of geometry: left area is 0
                    return 0 - fraction * geom.area
                elif split_pos >= max_x:
                    # Right of geometry: left area equals total area
                    return geom.area - fraction * geom.area

                # Create left and right parts using bounding box intersection
                left_bbox = box(min_x, min_y, split_pos, max_y)
                right_bbox = box(split_pos, min_y, max_x, max_y)

                left_part = geom.intersection(left_bbox)
                right_part = geom.intersection(right_bbox)

                if left_part.is_empty and right_part.is_empty:
                    return geom.area - fraction * geom.area

                # Use the non-empty part or combine if both exist
                if not left_part.is_empty and not right_part.is_empty:
                    # Both parts exist, return signed difference for left part
                    left_area = left_part.area
                    target_area = fraction * geom.area
                    return left_area - target_area
                elif not left_part.is_empty:
                    left_area = left_part.area
                    return left_area - fraction * geom.area
                else:
                    right_area = right_part.area
                    return right_area - (1 - fraction) * geom.area

            else:  # horizontal split
                if split_pos <= min_y:
                    # Below geometry: bottom area is 0
                    return 0 - fraction * geom.area
                elif split_pos >= max_y:
                    # Above geometry: bottom area equals total area
                    return geom.area - fraction * geom.area

                # Create bottom and top parts using bounding box intersection
                bottom_bbox = box(min_x, min_y, max_x, split_pos)
                top_bbox = box(min_x, split_pos, max_x, max_y)

                bottom_part = geom.intersection(bottom_bbox)
                top_part = geom.intersection(top_bbox)

                if bottom_part.is_empty and top_part.is_empty:
                    return geom.area - fraction * geom.area

                # Use the non-empty part or combine if both exist
                if not bottom_part.is_empty and not top_part.is_empty:
                    # Both parts exist, return signed difference for bottom part
                    bottom_area = bottom_part.area
                    target_area = fraction * geom.area
                    return bottom_area - target_area
                elif not bottom_part.is_empty:
                    bottom_area = bottom_part.area
                    return bottom_area - fraction * geom.area
                else:
                    top_area = top_part.area
                    return top_area - (1 - fraction) * geom.area

        except Exception:
            # If splitting fails, return large difference
            return 1e10

    # Use root finding to find optimal split position
    try:
        from scipy.optimize import root_scalar

        # For vertical splits, search in x-direction
        if direction == "vertical":
            # Search the full x-range of the geometry
            bracket_left = min_x
            bracket_right = max_x

            result = root_scalar(area_difference, bracket=(bracket_left, bracket_right), xtol=tol)
        else:  # horizontal
            # Search the full y-range of the geometry
            bracket_left = min_y
            bracket_right = max_y

            result = root_scalar(area_difference, bracket=(bracket_left, bracket_right), xtol=tol)

        if not result.converged:
            raise ValueError(f"Root finding did not converge. Final flag: {result.flag}")

        optimal_pos = result.root

        # Create the final split using bounding box intersection
        if direction == "vertical":
            left_bbox = box(min_x, min_y, optimal_pos, max_y)
            right_bbox = box(optimal_pos, min_y, max_x, max_y)

            left_part = geom.intersection(left_bbox)
            right_part = geom.intersection(right_bbox)

            # Return non-empty parts
            if not left_part.is_empty and not right_part.is_empty:
                # Both parts exist, determine which is "first" based on target fraction
                if abs(left_part.area - fraction * geom.area) < abs(right_part.area - fraction * geom.area):
                    return left_part, right_part
                else:
                    return right_part, left_part
            elif not left_part.is_empty:
                return left_part, right_bbox.difference(left_part).intersection(geom)
            else:
                return right_part, left_bbox.difference(right_part).intersection(geom)

        else:  # horizontal
            bottom_bbox = box(min_x, min_y, max_x, optimal_pos)
            top_bbox = box(min_x, optimal_pos, max_x, max_y)

            bottom_part = geom.intersection(bottom_bbox)
            top_part = geom.intersection(top_bbox)

            # Return non-empty parts
            if not bottom_part.is_empty and not top_part.is_empty:
                # Both parts exist, determine which is "first" based on target fraction
                if abs(bottom_part.area - fraction * geom.area) < abs(top_part.area - fraction * geom.area):
                    return bottom_part, top_part
                else:
                    return top_part, bottom_part
            elif not bottom_part.is_empty:
                return bottom_part, top_bbox.difference(bottom_part).intersection(geom)
            else:
                return top_part, bottom_bbox.difference(top_part).intersection(geom)

    except ImportError as err:
        raise ImportError("scipy is required for shape splitting. Install with: pip install scipy") from err
    except Exception as e:
        raise ValueError(f"Failed to split geometry: {e}") from e


def _split_treemap(
    geom: BaseGeometry,
    fractions: list[float],
    direction: Literal["vertical", "horizontal"],
    tol: float,
) -> list[BaseGeometry]:
    """
    Internal: Split geometry using treemap-style recursive binary partitioning.

    Recursively divides the geometry by finding a balanced split point in the
    fractions list, splitting the geometry proportionally, then recursing on
    each half with alternating directions.

    Parameters
    ----------
    geom : BaseGeometry
        Geometry to split.
    fractions : list[float]
        Normalized fractions (must sum to 1.0). Zero fractions produce empty geometries.
    direction : {'vertical', 'horizontal'}
        Current split direction.
    tol : float
        Tolerance for binary splits.

    Returns
    -------
    list[BaseGeometry]
        Parts in the same order as input fractions.
    """

    def _make_empty() -> BaseGeometry:
        """Create an empty geometry."""
        return geom.buffer(1e-10).buffer(-1e10)

    # Base case: single fraction means return the geometry as-is (or empty if ~0)
    if len(fractions) == 1:
        if fractions[0] < 1e-9:
            return [_make_empty()]
        return [geom]

    # Base case: two fractions means single binary split
    if len(fractions) == 2:
        # Handle zero fractions
        if fractions[0] < 1e-9:
            return [_make_empty(), geom]
        if fractions[1] < 1e-9:
            return [geom, _make_empty()]

        total = fractions[0] + fractions[1]
        split_frac = fractions[0] / total
        split_frac = max(0.001, min(0.999, split_frac))
        try:
            part1, part2 = _split_binary(geom, split_frac, direction=direction, tol=tol)
        except (ValueError, TypeError):
            # If split fails, return geometry for first, empty for second
            return [geom, _make_empty()]
        else:
            return [part1, part2]

    # Find split point: partition fractions into two groups with balanced total weight
    # Compute cumulative sums
    cumsum = []
    running = 0.0
    for f in fractions:
        running += f
        cumsum.append(running)

    total = cumsum[-1]
    half = total / 2.0

    # Find index where cumulative sum is closest to half
    # We split AFTER this index (so indices 0..split_idx go left, split_idx+1..end go right)
    best_idx = 0
    best_diff = abs(cumsum[0] - half)
    for i in range(1, len(cumsum) - 1):  # Don't consider last index (would put everything left)
        diff = abs(cumsum[i] - half)
        if diff < best_diff:
            best_diff = diff
            best_idx = i

    # Split fractions into left and right groups
    left_fractions = fractions[: best_idx + 1]
    right_fractions = fractions[best_idx + 1 :]

    left_sum = sum(left_fractions)
    right_sum = sum(right_fractions)

    # Split geometry at left_sum / total
    split_frac = left_sum / total
    split_frac = max(0.001, min(0.999, split_frac))

    try:
        left_geom, right_geom = _split_binary(geom, split_frac, direction=direction, tol=tol)
    except (ValueError, TypeError):
        # If split fails, assign all fractions to original geometry
        return [geom] * len(fractions)

    # Alternate direction for next level
    next_dir: Literal["vertical", "horizontal"] = "horizontal" if direction == "vertical" else "vertical"

    # Recursively split each part
    # Normalize fractions within each group
    left_normalized = [f / left_sum for f in left_fractions]
    right_normalized = [f / right_sum for f in right_fractions]

    left_parts = _split_treemap(left_geom, left_normalized, next_dir, tol)
    right_parts = _split_treemap(right_geom, right_normalized, next_dir, tol)

    return left_parts + right_parts


def split(
    geom: BaseGeometry,
    fractions: float | Sequence[float],
    direction: Literal["vertical", "horizontal"] = "vertical",
    alternate: bool = True,
    strategy: Literal["sequential", "treemap"] = "sequential",
    tol: float = 0.01,
) -> list[BaseGeometry]:
    """
    Split a geometry into multiple parts with specified area fractions.

    This function divides a geometry into N parts. Each fraction specifies the
    relative area of one resulting part. Two splitting strategies are available:

    - **sequential**: Carves off parts one-by-one from edges, optionally alternating
      direction. Creates strip-like or staircase patterns.
    - **treemap**: Recursively partitions using balanced binary splits with automatic
      direction alternation. Creates grid-like patterns with compact regions.

    Parameters
    ----------
    geom : shapely.geometry.base.BaseGeometry
        Input geometry to split. Should be a Polygon or MultiPolygon.
    fractions : float or Sequence[float]
        Target area fractions for the resulting parts.

        - **Single float**: Split into 2 parts with areas (fraction, 1-fraction).
          Must be in range (0, 1).
        - **Sequence of floats**: Split into N parts where N = len(fractions).
          Each value represents the target area fraction for that part.
          Values should be non-negative and sum to approximately 1.0.
          Zero fractions produce empty geometries in the corresponding position.
          Example: [0.3, 0.2, 0.5] creates 3 parts with 30%, 20%, 50% of total area.
    direction : {'vertical', 'horizontal'}, default='vertical'
        Initial direction of the splitting line:

        - **'vertical'**: Split along vertical lines (constant x-coordinate)
        - **'horizontal'**: Split along horizontal lines (constant y-coordinate)
    alternate : bool, default=True
        For sequential strategy only. Whether to alternate between vertical and
        horizontal directions for each subsequent split. If False, all splits
        use the same direction. Ignored for treemap strategy (which always alternates).
    strategy : {'sequential', 'treemap'}, default='sequential'
        Splitting strategy for N-way splits:

        - **'sequential'**: Parts are carved off one-by-one from the remaining
          geometry. Creates asymmetric strip/staircase patterns. Good when order
          matters or for ranked data.
        - **'treemap'**: Recursive binary partitioning that balances total weight
          on each side at every split. Creates grid-like patterns with compact,
          similarly-shaped regions. Good for categorical data where each part
          should have similar visual prominence.
    tol : float, default=0.01
        Absolute tolerance for area matching in each split operation.

    Returns
    -------
    list[BaseGeometry]
        List of geometry parts. For single float input, returns 2 parts.
        For sequence input with N values, returns N parts.
        Parts are ordered corresponding to input fractions.

    Raises
    ------
    ValueError
        If fractions are not valid (out of range, don't sum to ~1, etc.)
    TypeError
        If geom is not a valid Shapely geometry

    Examples
    --------
    Binary split (single fraction):

    >>> from shapely.geometry import Polygon
    >>> from carto_flow.shape_splitter import split
    >>>
    >>> rect = Polygon([(0, 0), (10, 0), (10, 5), (0, 5)])
    >>> parts = split(rect, 0.3, direction='vertical')
    >>> len(parts)  # 2 parts
    2
    >>> print(f"Areas: {[round(p.area, 1) for p in parts]}")  # [15.0, 35.0]

    Sequential strategy (default):

    >>> # Split into 4 strips, alternating direction
    >>> parts = split(rect, [0.25, 0.25, 0.25, 0.25], strategy='sequential')
    >>> len(parts)  # 4 parts
    4

    Treemap strategy for grid-like layout:

    >>> # Split into 4 compact regions (2x2 grid pattern)
    >>> parts = split(rect, [0.25, 0.25, 0.25, 0.25], strategy='treemap')
    >>> len(parts)  # 4 parts
    4

    Sequential without alternation (vertical strips):

    >>> parts = split(rect, [0.25, 0.25, 0.25, 0.25], strategy='sequential', alternate=False)
    """

    def _make_empty() -> BaseGeometry:
        """Create an empty geometry."""
        return geom.buffer(1e-10).buffer(-1e10)

    # Handle single fraction (binary split)
    if isinstance(fractions, (int, float)):
        part1, part2 = _split_binary(geom, float(fractions), direction=direction, tol=tol)
        return [part1, part2]

    # Convert to list for easier manipulation
    frac_list = list(fractions)

    # Validate strategy
    if strategy not in ("sequential", "treemap"):
        raise ValueError(f"strategy must be 'sequential' or 'treemap', got '{strategy}'")

    # Validate fractions
    if not frac_list:
        raise ValueError("fractions sequence cannot be empty")

    if len(frac_list) == 1:
        # Single fraction in sequence means binary split
        return split(geom, frac_list[0], direction=direction, alternate=alternate, strategy=strategy, tol=tol)

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

    # Dispatch to appropriate strategy
    if strategy == "treemap":
        return _split_treemap(geom, frac_list, direction, tol)

    # Sequential N-way splitting (default)
    parts: list[BaseGeometry] = []
    remaining = geom
    remaining_fraction = 1.0
    current_dir: Literal["vertical", "horizontal"] = direction

    for target_frac in frac_list[:-1]:  # Process all but last fraction
        # Handle near-zero fractions - produce empty part
        if target_frac < 1e-9:
            parts.append(_make_empty())
            continue

        # Compute relative fraction for this split
        # e.g., if we want 0.3 of total and have 1.0 remaining, split at 0.3/1.0 = 0.3
        # if we want 0.2 of total and have 0.7 remaining, split at 0.2/0.7 ≈ 0.286
        relative_frac = target_frac / remaining_fraction

        # Clamp to valid range
        relative_frac = max(0.001, min(0.999, relative_frac))

        try:
            part, remaining = _split_binary(remaining, relative_frac, direction=current_dir, tol=tol)
            parts.append(part)
            remaining_fraction -= target_frac
        except (ValueError, TypeError) as e:
            warnings.warn(f"Split operation failed at fraction {target_frac}: {e}", stacklevel=2)
            # Add empty geometry for this part and continue
            parts.append(_make_empty())

        # Alternate direction if requested
        if alternate:
            current_dir = "horizontal" if current_dir == "vertical" else "vertical"

    # Add the last remaining part
    parts.append(remaining)

    return parts
