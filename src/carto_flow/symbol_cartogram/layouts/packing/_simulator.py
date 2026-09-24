"""Two-stage circle packing simulator."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

_MAX_EXPANSION_FACTOR = 2.0


def _distance_in_radii(distance: NDArray[np.floating], radii: NDArray[np.floating]) -> NDArray[np.float64]:
    """Express a distance as a multiple of each symbol's radius.

    A symbol whose sizing value is zero has no radius, so any positive
    distance from it is infinitely many radii away; the ratio is ``inf``
    for those symbols.
    """
    return np.divide(
        distance,
        radii,
        out=np.full(np.shape(distance), np.inf, dtype=float),
        where=radii > 0,
    )


def _area_weighted_centroid(positions: NDArray[np.floating], radii: NDArray[np.floating]) -> NDArray[np.float64]:
    """Centroid of *positions* weighted by symbol area.

    Falls back to the unweighted mean when every symbol is zero-sized and
    the weights carry no information.
    """
    weights = radii**2
    total = float(weights.sum())
    if total <= 0:
        return np.asarray(positions, dtype=float).mean(axis=0)
    return (positions * weights[:, None]).sum(axis=0) / total


class ExponentialMovingStats:
    """EMA tracker for mean and std of vector-valued observations.

    Maintains per-element exponential moving averages of mean and variance
    for vector-valued time series. Used for steady-state detection by
    tracking displacement vectors over iterations.

    Parameters
    ----------
    n : int
        Number of elements (e.g., circles).
    dim : int
        Vector dimension (e.g., 2 for 2D displacement).
    n_eff : int
        Effective window size. Alpha = 2 / (n_eff + 1).
    adaptive : bool
        If True, alpha starts at 1/k and decays to the fixed value,
        providing faster initial convergence.

    """

    def __init__(self, n: int, dim: int, n_eff: int = 20, *, adaptive: bool = True):
        self.alpha = 2.0 / (n_eff + 1)
        self.mean = np.zeros((n, dim))
        self.var = np.zeros((n, dim))
        self._initialized = False
        self._adaptive = adaptive
        self._k = 0

    def update(self, x: NDArray[np.floating]) -> None:
        """Update with new observation x of shape (n, dim)."""
        if not self._initialized:
            self.mean[:] = x
            self._initialized = True
            self._k = 1
            return
        self._k += 1
        a = max(self.alpha, 1.0 / self._k) if self._adaptive else self.alpha
        delta = x - self.mean
        self.mean += a * delta
        self.var[:] = (1 - a) * (self.var + a * delta**2)

    @property
    def mean_magnitude(self) -> NDArray[np.floating]:
        """Per-element magnitude of mean vector: ||mu_i||."""
        return np.linalg.norm(self.mean, axis=1)

    @property
    def std_magnitude(self) -> NDArray[np.floating]:
        """Per-element std magnitude: sqrt(sum of component variances)."""
        return np.sqrt(np.sum(self.var, axis=1))


class ScalarEMA:
    """EMA tracker for a single scalar value.

    Parameters
    ----------
    n_eff : int
        Effective window size. Alpha = 2 / (n_eff + 1).
    adaptive : bool
        If True, alpha starts at 1/k and decays to the fixed value.
    initial_value : float
        Value before the first update. Default: 0.0

    """

    def __init__(
        self,
        n_eff: int = 20,
        *,
        adaptive: bool = True,
        initial_value: float = 0.0,
    ):
        self.alpha = 2.0 / (n_eff + 1)
        self.value: float = initial_value
        self._initialized = False
        self._adaptive = adaptive
        self._k = 0

    def update(self, x: float) -> None:
        """Update with new scalar observation."""
        if not self._initialized:
            self.value = x
            self._initialized = True
            self._k = 1
            return
        self._k += 1
        a = max(self.alpha, 1.0 / self._k) if self._adaptive else self.alpha
        self.value += a * (x - self.value)


class TopologyPreservingSimulator:
    """Two-phase force-based simulator with topology preservation.

    This simulator uses:

    - **Overlap Resolution Phase**: Global expansion + overlap projection
      to reach a non-overlapping state
    - **Packing Phase**: Force-based refinement with:
      - Distance-gated angular topology force (preserves neighbor directions)
      - Neighbor tangency spring (pulls separated neighbors together)
      - Global centroid attraction force (pulls toward original centroid)
      - Origin attraction force (pulls each circle toward its original position)
      - Iterative contact reaction (handles compressive forces at contacts)

    The contact reaction constraint allows circles to slide along each
    other without penetrating, enabling tighter packing while maintaining
    topology.

    Parameters
    ----------
    positions : NDArray[np.floating]
        Initial symbol positions, shape (n, 2).
    radii : NDArray[np.floating]
        Symbol radii, shape (n,).
    original_positions : NDArray[np.floating] | None
        Original centroid positions for topology reference.
    adjacency : NDArray[np.floating] | None
        Adjacency matrix of shape (n, n). Required for topology forces.
    spacing : float
        Minimum gap as fraction of average radius. Default: 0.05
    compactness : float
        Global compaction strength (0-1). Default: 0.5
    topology_weight : float
        Topology preservation strength (0-1). Default: 0.3
    overlap_tolerance : float
        Overlap tolerance for overlap resolution convergence, as fraction of
        average radius. Default: 1e-4
    expansion_max_iterations : int
        Maximum Gauss-Seidel iterations for overlap resolution. Default: 20
    expansion : float
        Stage 1 expansion fraction (0-1). 0 = GS-only; 1 = full exact
        expansion (default); intermediate = partial expansion with GS
        handling residuals. Default: 1.0
    topology_gate_distance : float
        Topology force gate distance (in sum of radii). Default: 2.5
    neighbor_weight : float
        Neighbor tangency force coefficient. Default: 0.5
    origin_weight : float
        Origin attraction force strength. Pulls each circle toward its
        original position. Set to 0 to disable. Default: 0.0
        - 0: No origin attraction (current behavior)
        - 0.1-0.5: Gentle pull toward original positions
        - > 1.0: Strong pull, may interfere with topology preservation
    force_mode : str
        How attraction force magnitude is computed. Applies to both the
        global centroid attraction force and the origin attraction force.
        Default: "direction"
        - "direction": Constant magnitude with drop-off near target
        - "linear": Force proportional to distance (spring)
        - "normalized": Force proportional to distance / radius
    contact_tolerance : float
        Contact detection tolerance (fraction of sum of radii). Default: 0.02
    contact_iterations : int
        Number of contact reaction passes per packing step. Default: 3
    max_step : float
        Maximum step size (fraction of avg radius). Default: 0.3
    contact_transfer_ratio : float
        Balance between cancel (0) and transfer (1) of compressive forces
        at contact points. Default: 0.5
    contact_elasticity : float
        Controls net compression vs bounce behavior (-1 to 1). Default: 0.0
    size_sensitivity : float
        Controls how step size scales with circle radius. Default: 0.0
    overlap_projection_iters : int
        Overlap projection iterations per packing step. Default: 5
    step_smoothing_window : int
        EMA window for step smoothing. Default: 20
    convergence_window : int
        EMA window for displacement convergence tracking. Default: 50
    adaptive_ema : bool
        Whether EMA uses adaptive warmup (alpha starts at 1/k). Default: True

    Notes
    -----
    The topology force uses distance gating: it only acts when circles
    are within ``topology_gate_distance * (r_i + r_j)`` distance. This prevents distant
    circles from exerting topology forces, focusing preservation on
    local relationships.

    """

    def __init__(
        self,
        positions: NDArray[np.floating],
        radii: NDArray[np.floating],
        original_positions: NDArray[np.floating] | None = None,
        adjacency: NDArray[np.floating] | None = None,
        spacing: float = 0.05,
        compactness: float = 0.5,
        topology_weight: float = 0.3,
        overlap_tolerance: float = 1e-4,
        expansion_max_iterations: int = 20,
        expansion: float = 1.0,
        topology_gate_distance: float = 2.5,
        neighbor_weight: float = 0.5,
        origin_weight: float = 0.0,
        group_ids: NDArray[np.intp] | None = None,
        group_weight: float = 0.0,
        force_mode: str = "direction",
        contact_tolerance: float = 0.02,
        contact_iterations: int = 3,
        max_step: float = 0.3,
        contact_transfer_ratio: float = 0.5,
        contact_elasticity: float = 0.0,
        size_sensitivity: float = 0.0,
        local_step_fraction: float = 0.5,
        overlap_projection_iters: int = 5,
        step_smoothing_window: int = 20,
        convergence_window: int = 50,
        adaptive_ema: bool = True,
    ):
        self.n = len(positions)

        # Handle None original_positions
        if original_positions is None:
            original_positions = positions.copy()

        # Compute scale factor to normalize to unit-ish coordinates
        # Use bounding box that includes full circle extent (position ± radius)
        all_mins = np.vstack([positions - radii[:, None], original_positions - radii[:, None]])
        all_maxs = np.vstack([positions + radii[:, None], original_positions + radii[:, None]])
        min_coords = all_mins.min(axis=0)
        max_coords = all_maxs.max(axis=0)
        self.center = (min_coords + max_coords) / 2
        extent = np.max(max_coords - min_coords)
        self.scale = extent if extent > 0 else 1.0

        # Store normalized positions
        self.positions = (positions.astype(float) - self.center) / self.scale
        self.original_positions = (original_positions.astype(float) - self.center) / self.scale
        self.radii = radii.astype(float) / self.scale

        # Compute spacing in normalized coordinates
        self.avg_radius = float(np.mean(self.radii))
        self.spacing = spacing * self.avg_radius

        # Store adjacency and extract pairs with per-pair weights
        self.adjacency = adjacency
        self.adjacency_pairs: list[tuple[int, int]] = []
        self._adj_weight_list: list[float] = []
        if adjacency is not None:
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    w_ij = adjacency[i, j]
                    w_ji = adjacency[j, i]
                    if w_ij > 0 or w_ji > 0:
                        self.adjacency_pairs.append((i, j))
                        # Symmetric max for asymmetric matrices
                        self._adj_weight_list.append(max(w_ij, w_ji))

        # Store parameters
        self.compactness = compactness
        self.topology_weight = topology_weight
        self.overlap_tolerance = overlap_tolerance
        self.expansion_max_iterations = expansion_max_iterations
        self.expansion = expansion
        self.topology_gate_distance = topology_gate_distance
        self.neighbor_weight = neighbor_weight
        self.origin_weight = origin_weight
        self.group_weight = group_weight
        self.force_mode = force_mode

        # Precompute per-group index masks for group centroid attraction
        if group_ids is not None and group_weight > 0:
            unique_groups = np.unique(group_ids)
            self._group_masks = [group_ids == g for g in unique_groups]
        else:
            self._group_masks = []
        self.contact_tolerance = contact_tolerance
        self.contact_iterations = contact_iterations
        self.max_step = max_step
        self.contact_transfer_ratio = np.clip(contact_transfer_ratio, 0, 1)
        self.contact_elasticity = np.clip(contact_elasticity, -1, 1)
        self.size_sensitivity = size_sensitivity
        self.local_step_fraction = local_step_fraction
        self.overlap_projection_iters = overlap_projection_iters
        self.step_smoothing_window = step_smoothing_window
        self.convergence_window = convergence_window
        self.adaptive_ema = adaptive_ema

        # EMA accumulators (reset at start of each run)
        self._reset_ema_state()
        # Use local RNG for reproducibility without global state
        self._rng = np.random.default_rng(42)

        # Precomputed data for vectorized force computation
        # reshape keeps the (m, 2) shape when there are no adjacent pairs at all
        self.adj_pairs = np.array(self.adjacency_pairs, dtype=np.intp).reshape(-1, 2)  # (m, 2)
        self.adj_weights = np.array(self._adj_weight_list, dtype=float)  # (m,)
        self.radii_sum = self.radii[self.adj_pairs[:, 0]] + self.radii[self.adj_pairs[:, 1]]  # (m,)

        # Precompute static topology data (original_positions never changes)
        if self.topology_weight > 0 and len(self.adj_pairs) > 0:
            v0 = self.original_positions[self.adj_pairs[:, 1]] - self.original_positions[self.adj_pairs[:, 0]]  # (m, 2)
            d0 = np.linalg.norm(v0, axis=1)  # (m,)
            self.topo_valid = d0 > 1e-10  # (m,) boolean mask for non-incident circles
            self.u0 = np.empty((len(self.adj_pairs), 2))  # (m, 2)
            self.u0[self.topo_valid] = v0[self.topo_valid] / d0[self.topo_valid, None]  # original unit vectors
        else:
            self.topo_valid = np.zeros(len(self.adj_pairs), dtype=bool)
            self.u0 = np.empty((len(self.adj_pairs), 2))

        # Precompute original centroid for stable global compaction
        self.original_centroid = _area_weighted_centroid(self.original_positions, self.radii)

        # Precompute upper triangular indices for vectorized overlap projection and contact reaction
        i_idx, j_idx = np.triu_indices(self.n, k=1)
        self.all_pairs_i = i_idx  # (m,)
        self.all_pairs_j = j_idx  # (m,)
        self.all_radii_sum = self.radii[i_idx] + self.radii[j_idx]  # (m,)

    def _reset_ema_state(self) -> None:
        """Reset EMA accumulators for a fresh run."""
        self._step_smooth = ExponentialMovingStats(
            self.n,
            2,
            n_eff=self.step_smoothing_window,
            adaptive=self.adaptive_ema,
        )
        self._disp_stats: ExponentialMovingStats | None = None

    def _weighted_centroid(self) -> NDArray[np.floating]:
        """Compute area-weighted centroid."""
        return _area_weighted_centroid(self.positions, self.radii)

    def _count_overlaps(self) -> int:
        """Count number of overlapping circle pairs."""
        diff = self.positions[self.all_pairs_j] - self.positions[self.all_pairs_i]
        dist = np.linalg.norm(diff, axis=1)
        min_dist = self.all_radii_sum + self.spacing
        return int(np.sum(dist < min_dist))

    def _separate_overlapping_pairs(self, max_iter: int = 30) -> float:
        """Iteratively push overlapping circles apart.

        Returns the maximum remaining overlap after projection.
        """
        for _ in range(max_iter):
            # Compute pairwise differences using precomputed indices
            diff = self.positions[self.all_pairs_j] - self.positions[self.all_pairs_i]  # (m, 2)
            dist = np.linalg.norm(diff, axis=1)  # (m,)

            # Compute min distances and overlaps using precomputed radii_sum
            min_dist = self.all_radii_sum + self.spacing
            overlap = min_dist - dist  # (m,)

            # Find overlapping pairs
            overlap_mask = overlap > 0
            if not np.any(overlap_mask):
                return 0.0

            # Get max overlap
            max_overlap = float(np.max(overlap[overlap_mask]))

            # Early exit if remaining overlap is tiny
            tol = self.overlap_tolerance * self.avg_radius
            if max_overlap < tol:
                return max_overlap

            # Get indices of overlapping pairs
            idx = np.where(overlap_mask)[0]
            diff_overlap = diff[idx]  # (k, 2)
            dist_overlap = dist[idx]  # (k,)
            overlap_vals = overlap[idx]  # (k,)

            # Compute directions for overlapping pairs
            valid_dist = dist_overlap > 1e-10
            directions = np.zeros((len(idx), 2))

            # For pairs with valid distance, use computed direction
            if np.any(valid_dist):
                directions[valid_dist] = diff_overlap[valid_dist] / dist_overlap[valid_dist, None]

            # For coincident pairs, use random directions
            if np.any(~valid_dist):
                n_coincident = np.sum(~valid_dist)
                random_dirs = self._rng.standard_normal((n_coincident, 2))
                random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
                directions[~valid_dist] = random_dirs

            # Compute shifts
            shifts = 0.5 * overlap_vals[:, None] * directions  # (k, 2)

            # Apply shifts using np.add.at
            np.add.at(self.positions, self.all_pairs_i[idx], -shifts)
            np.add.at(self.positions, self.all_pairs_j[idx], shifts)

        return max_overlap

    def _global_expansion_factor(self) -> float:
        """Compute the exact global expansion factor to resolve all overlaps.

        For each overlapping pair, the expansion factor needed is d_min / d.
        The global factor is the maximum over all pairs: the minimum uniform
        scaling from the centroid that would resolve every overlap.

        Returns 1.0 if no overlaps exist.
        """
        diff = self.positions[self.all_pairs_j] - self.positions[self.all_pairs_i]
        d = np.linalg.norm(diff, axis=1)
        d_min = self.all_radii_sum + self.spacing

        valid = (d > 1e-10) & (d < d_min)
        if not np.any(valid):
            return 1.0

        s_pairs = d_min[valid] / d[valid]
        return float(np.max(s_pairs))

    def _separate_overlapping_pairs_partial(self, step_fraction: float = 0.5) -> float:
        """Single pass pushing overlapping circles apart with partial steps.

        Called once per outer iteration to maintain balance with global
        expansion. The ratio between global_step_fraction and step_fraction
        directly controls relative correction weight.

        Parameters
        ----------
        step_fraction : float
            Fraction of full separation to apply (0-1].

        Returns
        -------
        max_overlap : float
            Maximum remaining overlap after projection.

        """
        diff = self.positions[self.all_pairs_j] - self.positions[self.all_pairs_i]
        d = np.linalg.norm(diff, axis=1)
        d_min = self.all_radii_sum + self.spacing
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
            random_dirs = self._rng.standard_normal((n_coincident, 2))
            random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
            directions[~valid_dist] = random_dirs

        shifts = step_fraction * 0.5 * overlap_vals[:, None] * directions

        np.add.at(self.positions, self.all_pairs_i[idx], -shifts)
        np.add.at(self.positions, self.all_pairs_j[idx], shifts)

        return max_overlap

    def _jitter_coincident(self) -> None:
        """Apply tiny random perturbation to coincident circle pairs.

        Coincident circles (d < epsilon) cannot be separated by global expansion
        (no direction to expand toward). A sub-radius perturbation breaks symmetry
        so the expansion and Gauss-Seidel phases can act on them.
        """
        diff = self.positions[self.all_pairs_j] - self.positions[self.all_pairs_i]
        d = np.linalg.norm(diff, axis=1)
        coincident = d < 1e-10
        if not np.any(coincident):
            return
        idx = np.where(coincident)[0]
        dirs = self._rng.standard_normal((len(idx), 2))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        epsilon = 1e-3 * self.avg_radius
        np.add.at(self.positions, self.all_pairs_i[idx], -epsilon * dirs)
        np.add.at(self.positions, self.all_pairs_j[idx], epsilon * dirs)

    def _separate_overlapping_pairs_sequential(self) -> float:
        """One Gauss-Seidel pass: resolve overlapping pairs sequentially.

        Pairs are processed in descending overlap order. After each pair is
        resolved the updated positions are used for subsequent pairs in the
        same pass — this prevents circles from being pushed back through
        neighbors that were just separated (topology-preserving property that
        Jacobi simultaneous updates cannot guarantee).

        Returns
        -------
        max_overlap : float
            Maximum overlap seen at the start of this pass (before corrections).
        """
        diff = self.positions[self.all_pairs_j] - self.positions[self.all_pairs_i]
        d = np.linalg.norm(diff, axis=1)
        d_min = self.all_radii_sum + self.spacing
        overlap = d_min - d

        # Process in descending overlap order; stop when overlap becomes non-positive
        order = np.argsort(-overlap)
        max_overlap = float(overlap[order[0]]) if len(order) > 0 else 0.0
        if max_overlap <= 0:
            return 0.0

        for k in order:
            if overlap[k] <= 0:
                break
            i, j = int(self.all_pairs_i[k]), int(self.all_pairs_j[k])
            # Recompute live distance — positions may have changed earlier in this pass
            delta = self.positions[j] - self.positions[i]
            dist = float(np.linalg.norm(delta))
            d_min_k = float(self.radii[i] + self.radii[j] + self.spacing)
            ov = d_min_k - dist
            if ov <= 0:
                continue
            if dist > 1e-10:
                direction = delta / dist
            else:
                direction = self._rng.standard_normal(2)
                direction /= np.linalg.norm(direction)
            shift = self.local_step_fraction * ov * direction
            self.positions[i] -= shift
            self.positions[j] += shift

        return max_overlap

    def run_overlap_resolution(self) -> dict[str, Any]:
        """Overlap Resolution Phase: topology-preserving separation in three sub-phases.

        A) Jitter any coincident circles with a sub-radius perturbation so global
           expansion has a direction to act on them.
        B) Single global expansion scaled by ``expansion`` (0-1), capped at
           ``_MAX_EXPANSION_FACTOR`` — pure radial scale, fully topology-preserving.
           ``expansion=0`` skips this step (GS-only).
        C) Gauss-Seidel sequential local separation to full convergence, handling
           residual overlaps left by the capped expansion.

        Can be called independently or as part of :meth:`run`.

        Returns
        -------
        info : dict
            ``{"iterations": int, "final_max_overlap": float}``

        """
        # A: break coincident circle symmetry
        self._jitter_coincident()

        # B: single topology-preserving global expansion
        if self.expansion > 0:
            C = self._weighted_centroid()
            s = self._global_expansion_factor()
            if s > 1.0:
                s = min(s, _MAX_EXPANSION_FACTOR)
                s_applied = 1.0 + self.expansion * (s - 1.0)
                self.positions = C + s_applied * (self.positions - C)

        # C: Gauss-Seidel sequential separation to convergence
        tol = self.overlap_tolerance * self.avg_radius
        max_overlap = 0.0
        gs_iters = 0
        for _gs_iters in range(self.expansion_max_iterations):
            max_overlap = self._separate_overlapping_pairs_sequential()
            if max_overlap < tol:
                break

        return {"iterations": 1 + gs_iters, "final_max_overlap": max_overlap}

    def _compute_forces(self) -> NDArray[np.floating]:
        """Compute packing forces: topology, neighbor tangency, global centroid attraction, origin attraction.

        Returns the total force array of shape (n, 2).
        """
        F = np.zeros_like(self.positions)

        # --- Distance-gated angular topology force ---
        if self.topology_weight > 0 and np.any(self.topo_valid):
            # Compute current vectors for all pairs
            v = self.positions[self.adj_pairs[:, 1]] - self.positions[self.adj_pairs[:, 0]]  # (m, 2)
            d = np.linalg.norm(v, axis=1)  # (m,)

            # Valid current distances mask (combine with precomputed topo_valid)
            valid = self.topo_valid & (d > 1e-10)
            if np.any(valid):
                # Compute gap (same definition as neighbor force: distance - sum of radii - spacing)
                gap = d[valid] - self.radii_sum[valid] - self.spacing

                # Smooth decay weight: w=1 for gap<=0, w=0 for gap>topology_gate_distance, smooth decay in between
                w = np.clip(1 - gap / self.topology_gate_distance, 0, 1) ** 2  # (k,)

                # Get indices of pairs with non-zero weight
                idx = np.where(valid)[0][w > 0]
                if len(idx) > 0:
                    # Current unit vectors for pairs with non-zero weight
                    u = v[idx] / d[idx, None]  # (p, 2)

                    # Topology force (u0 precomputed), weighted by adjacency
                    Ft = (
                        self.topology_weight * w[w > 0, None] * self.adj_weights[idx, None] * (self.u0[idx] - u)
                    )  # (p, 2)

                    # Accumulate forces using np.add.at for unbuffered accumulation
                    np.add.at(F, self.adj_pairs[idx, 0], -Ft)
                    np.add.at(F, self.adj_pairs[idx, 1], Ft)

        # --- Neighbor tangency force ---
        if len(self.adj_pairs) > 0:
            # Compute all pairwise differences
            dx = self.positions[self.adj_pairs[:, 1]] - self.positions[self.adj_pairs[:, 0]]  # (m, 2)
            d = np.linalg.norm(dx, axis=1)  # (m,)

            # Compute target distance and gap
            target = self.radii_sum + self.spacing
            gap = d - target

            # Positive gap mask (pull together) - also handles d <= 1e-10 since gap would be negative
            gap_mask = gap > 0
            if np.any(gap_mask):
                # Get indices of pairs with positive gap
                idx = np.where(gap_mask)[0]

                # Unit direction
                n_ij = dx[idx] / d[idx, None]  # (p, 2)

                # Strength
                strength = np.minimum(_distance_in_radii(gap[idx], target[idx]), 1.0)  # (p,)

                # Neighbor force, weighted by adjacency
                Fn = self.neighbor_weight * self.adj_weights[idx, None] * strength[:, None] * n_ij  # (p, 2)

                # Accumulate
                np.add.at(F, self.adj_pairs[idx, 0], Fn)
                np.add.at(F, self.adj_pairs[idx, 1], -Fn)

        # --- Global centroid attraction force ---
        if self.compactness > 0:
            # Use precomputed original centroid for stability
            d_vec = self.original_centroid - self.positions  # (n, 2)
            dn = np.linalg.norm(d_vec, axis=1)  # (n,)

            # Valid distances mask (needed to avoid division by zero)
            valid = dn > 1e-10
            if np.any(valid):
                # Normalize directions
                d_vec_valid = d_vec[valid]
                dn_valid = dn[valid]
                directions = d_vec_valid / dn_valid[:, None]  # (k, 2)

                # Compute force magnitude based on mode
                if self.force_mode == "direction":
                    # Default: constant magnitude with drop-off near centroid
                    w_compact = np.clip(_distance_in_radii(dn_valid, self.radii[valid]), 0, 1)  # (k,)
                    force_mag = self.compactness * w_compact  # (k,)

                elif self.force_mode == "linear":
                    # Linear spring: force proportional to distance
                    force_mag = self.compactness * dn_valid  # (k,)

                elif self.force_mode == "normalized":
                    # Normalized: force proportional to distance / radius
                    force_mag = self.compactness * _distance_in_radii(dn_valid, self.radii[valid])  # (k,)
                else:
                    # Fallback to direction mode
                    w_compact = np.clip(_distance_in_radii(dn_valid, self.radii[valid]), 0, 1)
                    force_mag = self.compactness * w_compact

                # Apply to valid indices
                F[valid] += force_mag[:, None] * directions

        # --- Origin attraction force ---
        if self.origin_weight > 0:
            # Vector from current position to original position
            displacement = self.original_positions - self.positions  # (n, 2)

            # Compute distance
            dist = np.linalg.norm(displacement, axis=1)  # (n,)

            # Valid mask for non-zero distances
            valid = dist > 1e-10
            if np.any(valid):
                # Direction toward original position
                direction = displacement[valid] / dist[valid, None]  # (k, 2)

                # Compute force magnitude based on mode
                if self.force_mode == "direction":
                    # Default: constant magnitude with drop-off near origin
                    w_origin = np.clip(_distance_in_radii(dist[valid], self.radii[valid]), 0, 1)  # (k,)
                    force_mag = self.origin_weight * w_origin  # (k,)

                elif self.force_mode == "linear":
                    # Linear spring: force proportional to distance
                    force_mag = self.origin_weight * dist[valid]  # (k,)

                elif self.force_mode == "normalized":
                    # Normalized: force proportional to distance / radius
                    force_mag = self.origin_weight * _distance_in_radii(dist[valid], self.radii[valid])  # (k,)
                else:
                    # Fallback to direction mode
                    w_origin = np.clip(_distance_in_radii(dist[valid], self.radii[valid]), 0, 1)
                    force_mag = self.origin_weight * w_origin

                # Apply force
                F[valid] += force_mag[:, None] * direction

        # --- Group centroid attraction ---
        # Pulls each circle toward the current centroid of its group.
        # Uses the current (not original) group centroid so the group drifts as a unit.
        if self.group_weight > 0 and self._group_masks:
            for mask in self._group_masks:
                group_center = self.positions[mask].mean(axis=0)
                displacement = group_center - self.positions[mask]
                dist = np.linalg.norm(displacement, axis=1)
                valid = dist > 1e-10
                if np.any(valid):
                    direction = displacement[valid] / dist[valid, None]
                    r_mask = self.radii[mask]
                    if self.force_mode == "direction":
                        w = np.clip(_distance_in_radii(dist[valid], r_mask[valid]), 0, 1)
                        force_mag = self.group_weight * w
                    elif self.force_mode == "linear":
                        force_mag = self.group_weight * dist[valid]
                    elif self.force_mode == "normalized":
                        force_mag = self.group_weight * _distance_in_radii(dist[valid], r_mask[valid])
                    else:
                        w = np.clip(_distance_in_radii(dist[valid], r_mask[valid]), 0, 1)
                        force_mag = self.group_weight * w
                    idx = np.where(mask)[0][valid]
                    F[idx] += force_mag[:, None] * direction

        return F

    def _resolve_contacts(self, F: NDArray[np.floating]) -> None:
        """Apply contact reaction with configurable transfer and elasticity.

        This allows circles to slide along each other without penetrating.
        Modifies F in place.

        The behavior is controlled by contact_transfer_ratio and contact_elasticity:
        - contact_transfer_ratio: Balance between cancel (0) and transfer (1)
        - contact_elasticity: Controls net compression vs bounce (-1 to 1)
        """
        # Precompute a and b from parameters
        s = (1 - self.contact_elasticity) / (1 + self.contact_elasticity + 1e-10)
        a = (1 - self.contact_transfer_ratio) ** s
        b = self.contact_transfer_ratio**s

        for _ in range(self.contact_iterations):
            # Compute pairwise differences using precomputed indices
            dx = self.positions[self.all_pairs_j] - self.positions[self.all_pairs_i]  # (m, 2)
            d = np.linalg.norm(dx, axis=1)  # (m,)

            # Valid distances mask
            valid = d > 1e-10
            if not np.any(valid):
                continue

            # Compute radii sums and contact tolerance using precomputed radii_sum
            r_sum = self.all_radii_sum + self.spacing
            tol = self.contact_tolerance * r_sum

            # Contact mask
            contact_mask = valid & (d < r_sum + tol)
            if not np.any(contact_mask):
                continue

            # Get indices of contacting pairs
            idx = np.where(contact_mask)[0]
            dx_contact = dx[idx]  # (k, 2)
            d_contact = d[idx]  # (k,)

            # Unit directions
            n_ij = dx_contact / d_contact[:, None]  # (k, 2)
            n_ji = -n_ij  # Direction from j to i

            # Compute compressive components
            F_i = F[self.all_pairs_i[idx]]
            F_j = F[self.all_pairs_j[idx]]
            comp_i = np.sum(F_i * n_ij, axis=1)
            comp_j = np.sum(F_j * n_ji, axis=1)

            # Apply reaction (remove compressive components)
            comp_i_mask = comp_i > 0
            comp_j_mask = comp_j > 0

            if np.any(comp_i_mask):
                idx_i = idx[comp_i_mask]
                compressive_i = comp_i[comp_i_mask, None] * n_ij[comp_i_mask]
                np.add.at(F, self.all_pairs_i[idx_i], -a * compressive_i)

            if np.any(comp_j_mask):
                idx_j = idx[comp_j_mask]
                compressive_j = comp_j[comp_j_mask, None] * n_ji[comp_j_mask]
                np.add.at(F, self.all_pairs_j[idx_j], -a * compressive_j)

            # Transfer compressive forces
            if np.any(comp_i_mask):
                idx_i = idx[comp_i_mask]
                transfer_i = b * comp_i[comp_i_mask, None] * n_ij[comp_i_mask]
                np.add.at(F, self.all_pairs_j[idx_i], transfer_i)

            if np.any(comp_j_mask):
                idx_j = idx[comp_j_mask]
                transfer_j = b * comp_j[comp_j_mask, None] * n_ji[comp_j_mask]
                np.add.at(F, self.all_pairs_i[idx_j], transfer_j)

    def packing_step(self) -> tuple[float, float]:
        """Perform one packing refinement iteration.

        Returns
        -------
        drift : float
            Mean relative magnitude of smoothed displacement vectors.
            Trends to zero at steady state (symmetric jitter cancels).
        jitter : float
            Mean relative std of displacement vectors. Measures
            oscillation amplitude around equilibrium.

        """
        # Compute forces and resolve contacts
        F = self._compute_forces()
        self._resolve_contacts(F)

        # Integrate with fixed step clamping
        norms = np.linalg.norm(F, axis=1)
        scale = np.minimum(1.0, self.max_step / (norms + 1e-8))
        if self.avg_radius > 0:
            effective_radius = self.avg_radius * (self.radii / self.avg_radius) ** self.size_sensitivity
        else:
            # Every symbol is zero-sized: there is no step scale and nothing to pack.
            effective_radius = np.zeros(self.n)
        step = F * scale[:, None] * effective_radius[:, None]

        # Smooth step via EMA
        self._step_smooth.update(step)

        old_positions = self.positions.copy()
        self.positions += self._step_smooth.mean

        # Hard non-overlap projection
        self._separate_overlapping_pairs(max_iter=self.overlap_projection_iters)

        # Actual displacement after projection (true net movement)
        displacement = self.positions - old_positions

        # Update displacement statistics (vector EMA for convergence)
        if self._disp_stats is None:
            self._disp_stats = ExponentialMovingStats(
                self.n,
                2,
                n_eff=self.convergence_window,
                adaptive=self.adaptive_ema,
            )
        self._disp_stats.update(displacement)

        # Displacement is measured relative to symbol size. Symbols with a zero
        # sizing value have no radius to measure against and are left out of
        # the averages; when every symbol is zero-sized there is nothing to
        # pack and both statistics are zero.
        sized = self.radii > 0
        if np.any(sized):
            drift = float(np.mean(self._disp_stats.mean_magnitude[sized] / self.radii[sized]))
            jitter = float(np.mean(self._disp_stats.std_magnitude[sized] / self.radii[sized]))
        else:
            drift = jitter = 0.0

        return drift, jitter

    def run_circle_packing(
        self,
        max_iterations: int = 500,
        tolerance: float = 0.025,
        show_progress: bool = True,
        save_history: bool = False,
    ) -> tuple[NDArray[np.floating], dict[str, Any], list[NDArray[np.floating]] | None]:
        """Run circle packing (Stage 2) until convergence or max iterations.

        Resets EMA state so this method can be called multiple times or
        after :meth:`run_overlap_resolution`. Uses :meth:`packing_step`
        internally.

        Parameters
        ----------
        max_iterations : int
            Maximum number of packing iterations. Default: 500
        tolerance : float
            Convergence threshold for mean step size. Default: 1e-4
        show_progress : bool
            Whether to display progress bar. Default: True
        save_history : bool
            Whether to save position history. Default: False

        Returns
        -------
        positions : NDArray[np.floating]
            Final symbol positions in original coordinate system.
        info : dict
            Simulation statistics including:
            - "converged": Whether convergence criteria met
            - "final_overlaps": Remaining overlap count
            - "stage2_iterations": Packing iteration count
            - "final_drift": Final smoothed drift metric
            - "final_jitter": Final jitter metric
            - "drift_history": Per-iteration drift values
            - "jitter_history": Per-iteration jitter values
        history : list[NDArray[np.floating]] | None
            Position history if save_history=True, else None.

        """
        self._reset_ema_state()
        history: list[NDArray[np.floating]] | None = [] if save_history else None

        iterator: Any = range(max_iterations)
        if show_progress:
            iterator = tqdm(iterator, desc="Refining topology", leave=True)

        converged = False
        stage2_iters = 0
        drift_history: list[float] = []
        jitter_history: list[float] = []
        drift_rate_history: list[float] = []
        drift_rate_neg_frac_history: list[float] = []
        overlap_history: list[int] = []
        prev_drift: float | None = None
        drift_rate_ema = ScalarEMA(n_eff=20, adaptive=self.adaptive_ema)
        neg_frac_ema = ScalarEMA(n_eff=20, adaptive=self.adaptive_ema, initial_value=0.5)

        for _ in iterator:
            stage2_iters += 1

            if history is not None:
                history.append(self.positions * self.scale + self.center)

            drift, jitter = self.packing_step()
            drift_history.append(drift)
            jitter_history.append(jitter)

            if prev_drift is None:
                drift_rate_history.append(float("nan"))
                drift_rate_neg_frac_history.append(float("nan"))
            else:
                drift_rate = drift - prev_drift
                drift_rate_ema.update(drift_rate)
                neg_frac_ema.update(1.0 if drift_rate <= 0 else 0.0)

                # drift_rate_history.append(drift_rate_ema.value)
                drift_rate_history.append(drift_rate)
                drift_rate_neg_frac_history.append(neg_frac_ema.value)
            prev_drift = drift

            overlaps = self._count_overlaps()
            overlap_history.append(overlaps)

            if show_progress:
                iterator.set_postfix({"drift": f"{drift:.2e}", "jitter": f"{jitter:.2e}", "overlaps": overlaps})

            # Steady state: smoothed net displacement below tolerance
            if neg_frac_ema.value < 0.5 and drift < tolerance:
                converged = True
                if show_progress:
                    iterator.update()
                    iterator.close()
                break

        final_overlaps = self._count_overlaps()

        info = {
            "converged": converged,
            "final_overlaps": final_overlaps,
            "stage2_iterations": stage2_iters,
            "final_drift": drift_history[-1] if drift_history else 0.0,
            "final_jitter": jitter_history[-1] if jitter_history else 0.0,
            "drift_history": drift_history,
            "jitter_history": jitter_history,
            "drift_rate_history": drift_rate_history,
            "drift_rate_neg_frac_history": drift_rate_neg_frac_history,
            "overlap_history": overlap_history,
        }

        final_positions = self.positions * self.scale + self.center
        return final_positions, info, history

    def run(
        self,
        max_iterations: int = 500,
        tolerance: float = 0.025,
        show_progress: bool = True,
        save_history: bool = False,
    ) -> tuple[NDArray[np.floating], dict[str, Any], list[NDArray[np.floating]] | None]:
        """Run both stages: overlap resolution then circle packing.

        Equivalent to calling :meth:`run_overlap_resolution` followed by
        :meth:`run_circle_packing`. Can be called multiple times; EMA
        state is reset at the start of each packing phase.

        Parameters
        ----------
        max_iterations : int
            Maximum number of packing iterations. Default: 500
        tolerance : float
            Convergence threshold for mean step size. Default: 1e-4
        show_progress : bool
            Whether to display progress bar. Default: True
        save_history : bool
            Whether to save position history. Default: False

        Returns
        -------
        positions : NDArray[np.floating]
            Final symbol positions in original coordinate system.
        info : dict
            Simulation statistics including:
            - "iterations": Total iterations (stage1 + stage2)
            - "converged": Whether convergence criteria met
            - "final_overlaps": Remaining overlap count
            - "stage1_iterations": Overlap resolution iteration count
            - "stage2_iterations": Packing iteration count
            - "final_drift": Final smoothed drift metric
            - "final_jitter": Final jitter metric
            - "drift_history": Per-iteration drift values
            - "jitter_history": Per-iteration jitter values
        history : list[NDArray[np.floating]] | None
            Position history if save_history=True, else None.

        """
        stage1_info = self.run_overlap_resolution()

        # A single symbol cannot overlap or drift, and a zero average radius
        # means every symbol is zero-sized: there is nothing to pack in either
        # case, so skip straight to the no-op result.
        if max_iterations == 0 or self.avg_radius <= 0 or self.n < 2:
            final_positions = self.positions * self.scale + self.center
            info = {
                "iterations": stage1_info["iterations"],
                "stage1_iterations": stage1_info["iterations"],
                "stage2_iterations": 0,
                "cleanup_iterations": 0,
                "converged": True,
                "final_overlaps": self._count_overlaps(),
                "final_drift": 0.0,
                "final_jitter": 0.0,
                "drift_history": [],
                "jitter_history": [],
                "drift_rate_history": [],
                "drift_rate_neg_frac_history": [],
                "overlap_history": [],
            }
            return final_positions, info, None

        _positions, stage2_info, history = self.run_circle_packing(
            max_iterations=max_iterations,
            tolerance=tolerance,
            show_progress=show_progress,
            save_history=save_history,
        )

        # Final GS cleanup: resolve residual overlaps left by Stage 2's Jacobi
        # projection without rerunning global expansion. Forces during Stage 2
        # push circles to the constraint boundary; Jacobi may leave small
        # residuals at convergence. GS here enforces the hard non-overlap
        # constraint cleanly and costs one pass rather than slowing every step.
        tol = self.overlap_tolerance * self.avg_radius
        cleanup_iters = 0
        for _cleanup_iters in range(self.expansion_max_iterations):
            max_overlap = self._separate_overlapping_pairs_sequential()
            if max_overlap < tol:
                break

        final_positions = self.positions * self.scale + self.center
        info = {
            "iterations": stage1_info["iterations"] + stage2_info["stage2_iterations"],
            "stage1_iterations": stage1_info["iterations"],
            "cleanup_iterations": cleanup_iters + 1,
            **stage2_info,
        }

        return final_positions, info, history
