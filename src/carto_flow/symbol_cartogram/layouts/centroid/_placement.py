"""Circle overlap resolution for centroid-based layout."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def compact_initial_positions(
    centroid: NDArray[np.floating],
    radius: float,
    count: int,
    spacing: float = 0.0,
) -> NDArray[np.floating]:
    """Arrange `count` circles in a Fibonacci spiral around centroid.

    Produces a compact, non-overlapping arrangement with no coincident starts.

    Parameters
    ----------
    centroid : (2,) array
        Center point for the arrangement.
    radius : float
        Circle radius.
    count : int
        Number of circles to arrange.
    spacing : float
        Gap between circles as fraction of radius. Default: 0.0

    Returns
    -------
    positions : (count, 2) array
        Circle center positions.

    Examples
    --------
    >>> centroid = np.array([0.0, 0.0])
    >>> positions = compact_initial_positions(centroid, 1.0, 7)
    >>> positions.shape
    (7, 2)

    """
    if count == 1:
        return centroid[None, :].copy()
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # ~137.5 degrees
    step = 2.0 * radius * (1.0 + spacing)
    positions = np.empty((count, 2))
    for k in range(count):
        r = np.sqrt(k) * step
        theta = k * golden_angle
        positions[k] = centroid + r * np.array([np.cos(theta), np.sin(theta)])
    return positions


def resolve_circle_overlaps(
    positions: NDArray[np.floating],
    radii: NDArray[np.floating],
    *,
    spacing: float = 0.05,
    max_iterations: int = 20,
    overlap_tolerance: float = 1e-4,
    global_step_fraction: float = 0.5,
    local_step_fraction: float = 0.5,
    max_expansion_factor: float = 2.0,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.floating], dict[str, Any]]:
    """Resolve circle overlaps via global expansion + local separation.

    Alternates partial global expansion from the weighted centroid with
    partial local pairwise separation until all overlaps are resolved.

    This is a standalone function extracted from TopologyPreservingSimulator
    for reuse in simple layout algorithms that only need overlap resolution.

    Parameters
    ----------
    positions : (n, 2) array
        Circle center positions.
    radii : (n,) array
        Circle radii.
    spacing : float
        Minimum gap as fraction of average radius. Default: 0.05
    max_iterations : int
        Maximum outer iterations. Default: 20
    overlap_tolerance : float
        Convergence tolerance as fraction of average radius. Default: 1e-4
    global_step_fraction : float
        Fraction of global expansion to apply per iteration (0-1]. Default: 0.5
    local_step_fraction : float
        Fraction of local separation to apply per iteration (0-1]. Default: 0.5
    max_expansion_factor : float
        Maximum expansion factor clamp per iteration. Must be > 1.0. Default: 2.0
    rng : np.random.Generator, optional
        Random number generator for coincident circle handling.
        If None, a default generator with seed 42 is used.

    Returns
    -------
    positions : (n, 2) array
        Resolved positions with no overlaps (or minimal remaining overlap).
    info : dict
        Statistics with keys:
        - "iterations": Number of iterations performed
        - "final_max_overlap": Maximum remaining overlap as fraction of avg radius

    Examples
    --------
    >>> positions = np.array([[0, 0], [0.5, 0], [1, 0]])
    >>> radii = np.array([0.4, 0.4, 0.4])
    >>> resolved, info = resolve_circle_overlaps(positions, radii, spacing=0.1)
    >>> info["iterations"]
    3

    """
    n = len(positions)
    if n == 0:
        return positions.copy(), {"iterations": 0, "final_max_overlap": 0.0}

    # Use provided RNG or create default
    if rng is None:
        rng = np.random.default_rng(42)

    # Work on a copy
    positions = positions.astype(float).copy()
    radii = radii.astype(float).copy()

    # Compute scale factor to normalize to unit-ish coordinates
    all_mins = positions - radii[:, None]
    all_maxs = positions + radii[:, None]
    min_coords = all_mins.min(axis=0)
    max_coords = all_maxs.max(axis=0)
    center = (min_coords + max_coords) / 2
    extent = np.max(max_coords - min_coords)
    scale = extent if extent > 0 else 1.0

    # Normalize positions and radii
    positions_norm = (positions - center) / scale
    radii_norm = radii / scale

    # Compute spacing in normalized coordinates
    avg_radius = float(np.mean(radii_norm))
    spacing_norm = spacing * avg_radius
    tol_overlap = overlap_tolerance * avg_radius

    # Precompute upper triangular indices for vectorized operations
    i_idx, j_idx = np.triu_indices(n, k=1)
    all_radii_sum = radii_norm[i_idx] + radii_norm[j_idx]

    def weighted_centroid() -> NDArray[np.floating]:
        """Compute area-weighted centroid."""
        weights = radii_norm**2
        return (positions_norm * weights[:, None]).sum(axis=0) / weights.sum()

    def global_expansion_factor() -> float:
        """Compute the exact global expansion factor to resolve all overlaps."""
        diff = positions_norm[j_idx] - positions_norm[i_idx]
        d = np.linalg.norm(diff, axis=1)
        d_min = all_radii_sum + spacing_norm

        valid = (d > 1e-10) & (d < d_min)
        if not np.any(valid):
            return 1.0

        s_pairs = d_min[valid] / d[valid]
        return float(np.max(s_pairs))

    def separate_overlapping_pairs_partial(step_fraction: float = 0.5) -> float:
        """Single pass pushing overlapping circles apart with partial steps."""
        diff = positions_norm[j_idx] - positions_norm[i_idx]
        d = np.linalg.norm(diff, axis=1)
        d_min = all_radii_sum + spacing_norm
        overlap = d_min - d

        overlap_mask = overlap > 0
        if not np.any(overlap_mask):
            return 0.0

        max_overlap = float(np.max(overlap[overlap_mask]))

        idx = np.where(overlap_mask)[0]
        diff_overlap = diff[idx]
        d_overlap = d[idx]
        overlap_vals = overlap[idx]

        valid_dist = d_overlap > 1e-10
        directions = np.zeros((len(idx), 2))

        if np.any(valid_dist):
            directions[valid_dist] = diff_overlap[valid_dist] / d_overlap[valid_dist, None]

        if np.any(~valid_dist):
            n_coincident = int(np.sum(~valid_dist))
            random_dirs = rng.standard_normal((n_coincident, 2))
            random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
            directions[~valid_dist] = random_dirs

        shifts = step_fraction * 0.5 * overlap_vals[:, None] * directions

        np.add.at(positions_norm, i_idx[idx], -shifts)
        np.add.at(positions_norm, j_idx[idx], shifts)

        return max_overlap

    # Main iteration loop
    C = weighted_centroid()
    max_overlap = 0.0
    iterations_done = 0

    for _ in range(max_iterations):
        iterations_done += 1

        # 1. Compute exact global expansion factor
        s_global = global_expansion_factor()

        # 2. Apply partial global expansion (clamped)
        s_global = min(s_global, max_expansion_factor)
        if s_global > 1.0:
            s_applied = 1.0 + global_step_fraction * (s_global - 1.0)
            positions_norm = C + s_applied * (positions_norm - C)

        # 3. Single-pass partial local separation
        max_overlap = separate_overlapping_pairs_partial(step_fraction=local_step_fraction)

        # 4. Check convergence
        if max_overlap < tol_overlap:
            break

    # Convert back to original coordinate system
    final_positions = positions_norm * scale + center

    return final_positions, {
        "iterations": iterations_done,
        "final_max_overlap": max_overlap,
    }
