"""Physics-based circle simulator."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


class CirclePhysicsSimulator:
    """Physics-based simulator for resolving circle overlaps.

    Uses a two-phase approach:
    1. Separation phase: Strong repulsion only (no attraction) until overlaps resolved
    2. Settling phase: Gentle attraction while maintaining separation

    Internally normalizes to unit scale for numerical stability.
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
        damping: float = 0.85,
        dt: float = 0.15,
        max_velocity: float = 3.0,
        k_repel: float = 15.0,
        k_attract: float = 2.0,
    ):
        """Initialize simulator.

        Parameters
        ----------
        compactness : float
            Balance between centroid-attraction and neighbor-attraction (0-1).
            0 = symbols attracted only to original centroids
            1 = symbols attracted only to neighbors (tight cluster)
        damping : float
            Velocity damping factor (0-1). Default: 0.85
        dt : float
            Integration timestep. Default: 0.15
        max_velocity : float
            Maximum velocity magnitude. Default: 3.0
        k_repel : float
            Repulsion force coefficient. Default: 15.0
        k_attract : float
            Base attraction force coefficient. Default: 2.0

        """
        self.n = len(positions)
        self.adjacency = adjacency
        self.damping = damping
        self.compactness = compactness
        self.topology_weight = topology_weight

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
        self.velocities = np.zeros_like(self.positions)

        # Compute spacing in normalized coordinates
        self.avg_radius = float(np.mean(self.radii))
        self.spacing = spacing * self.avg_radius

        # Physics parameters
        self.dt = dt
        self.max_velocity = max_velocity
        self.k_repel = k_repel
        self.k_attract = k_attract

    def _count_overlaps(self) -> int:
        """Count current number of overlapping pairs."""
        count = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                diff = self.positions[j] - self.positions[i]
                dist = float(np.linalg.norm(diff))
                min_dist = self.radii[i] + self.radii[j] + self.spacing
                if dist < min_dist:
                    count += 1
        return count

    def step(self, apply_attraction: bool = False) -> tuple[int, float, float]:
        """Perform one simulation step.

        Parameters
        ----------
        apply_attraction : bool
            Whether to apply attraction toward original positions.

        Returns
        -------
        n_overlaps : int
            Number of overlapping pairs.
        max_velocity : float
            Maximum velocity magnitude.
        max_gap_ratio : float
            Maximum gap ratio (how far neighbors are from touching).

        """
        forces = np.zeros_like(self.positions)

        # 1. Repulsive forces from overlaps (always applied)
        n_overlaps = self._compute_repulsion(forces)

        # 2. Attractive forces (only when requested and no overlaps)
        # Always compute gap_ratio for tracking, but only apply forces when no overlaps
        gap_ratio = self._compute_gap_ratio() if self.adjacency is not None else 0.0

        if apply_attraction and n_overlaps == 0:
            self._compute_origin_attraction(forces)
            if self.adjacency is not None and self.compactness > 0:
                self._compute_neighbor_attraction(forces, gap_ratio)

        # 3. Clamp forces
        force_magnitudes = np.linalg.norm(forces, axis=1, keepdims=True)
        max_force = 30.0  # High limit to allow fast overlap resolution
        scale_factors = np.where(force_magnitudes > max_force, max_force / (force_magnitudes + 1e-10), 1.0)
        forces *= scale_factors

        # 4. Integration with velocity clamping
        self.velocities += forces * self.dt

        # Adaptive damping based on phase
        if n_overlaps > 0:
            # Separation phase: low damping for fast overlap resolution
            effective_damping = 0.5
        elif gap_ratio >= 0:
            if gap_ratio < 0.02:
                # Very close to target - higher damping to allow final approach
                effective_damping = 0.7
            elif gap_ratio < 0.1:
                # Close - moderate damping to prevent oscillation
                effective_damping = 0.5
            else:
                # Far from target - low damping for fast movement
                effective_damping = 0.5
        else:
            effective_damping = self.damping

        self.velocities *= effective_damping

        vel_magnitudes = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        scale_factors = np.where(vel_magnitudes > self.max_velocity, self.max_velocity / (vel_magnitudes + 1e-10), 1.0)
        self.velocities *= scale_factors

        self.positions += self.velocities * self.dt

        # Check for NaN/Inf
        if not np.all(np.isfinite(self.positions)):
            self.positions = self.original_positions.copy()
            self.velocities = np.zeros_like(self.positions)
            return n_overlaps, 0.0, gap_ratio

        max_velocity = float(np.max(np.linalg.norm(self.velocities, axis=1)))
        return n_overlaps, max_velocity, gap_ratio

    def _compute_repulsion(self, forces: NDArray[np.floating]) -> int:
        """Compute repulsive forces between overlapping circles."""
        n_overlaps = 0

        for i in range(self.n):
            for j in range(i + 1, self.n):
                diff = self.positions[j] - self.positions[i]
                dist = float(np.linalg.norm(diff))
                min_dist = self.radii[i] + self.radii[j] + self.spacing

                if dist < min_dist:
                    if dist > 1e-10:
                        n_overlaps += 1
                        overlap = min_dist - dist

                        # Non-linear force for deep overlaps
                        overlap_ratio = overlap / min_dist
                        force_multiplier = 1.0 + overlap_ratio * 3.0

                        direction = diff / dist
                        # Ensure minimum force even for tiny overlaps
                        min_repel_force = 0.5 * self.k_repel * self.avg_radius
                        force_mag = max(self.k_repel * overlap * force_multiplier, min_repel_force)
                        force = force_mag * direction
                        forces[i] -= force
                        forces[j] += force
                    else:
                        # Exactly overlapping - strong random push
                        random_dir = np.random.randn(2)
                        norm = float(np.linalg.norm(random_dir))
                        if norm > 1e-10:
                            random_dir /= norm
                        else:
                            random_dir = np.array([1.0, 0.0])
                        force = self.k_repel * min_dist * 2.0 * random_dir  # Stronger push
                        forces[i] -= force
                        forces[j] += force
                        n_overlaps += 1

        return n_overlaps

    def _compute_origin_attraction(self, forces: NDArray[np.floating]) -> None:
        """Attractive force toward original positions (weighted by 1 - compactness)."""
        displacement = self.original_positions - self.positions
        weight = (1 - self.compactness) * self.k_attract
        forces += weight * displacement

    def _compute_gap_ratio(self) -> float:
        """Compute max gap ratio between adjacent symbols (without applying forces)."""
        if self.adjacency is None:
            return 0.0

        max_gap_ratio = 0.0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.adjacency[i, j] > 0:
                    diff = self.positions[j] - self.positions[i]
                    dist = float(np.linalg.norm(diff))
                    target_dist = self.radii[i] + self.radii[j] + self.spacing
                    if dist > 1e-10 and target_dist > 0:
                        gap_ratio = (dist - target_dist) / target_dist
                        max_gap_ratio = max(max_gap_ratio, gap_ratio)
        return max_gap_ratio

    def _compute_neighbor_attraction(self, forces: NDArray[np.floating], current_gap_ratio: float) -> None:
        """Attractive forces between original neighbors (weighted by compactness)."""
        if self.adjacency is None:
            return

        for i in range(self.n):
            for j in range(i + 1, self.n):
                adj_weight = self.adjacency[i, j]
                if adj_weight > 0:
                    diff = self.positions[j] - self.positions[i]
                    dist = float(np.linalg.norm(diff))

                    if dist > 1e-10:
                        direction = diff / dist
                        target_dist = self.radii[i] + self.radii[j] + self.spacing

                        # Attract if farther than touching distance. A zero
                        # touching distance means both symbols are zero-sized
                        # and there is no gap to measure against.
                        if dist > target_dist > 0:
                            gap = dist - target_dist
                            local_gap_ratio = gap / target_dist

                            # Scale force based on gap: stronger when far, min force when close
                            if local_gap_ratio > 0.5:
                                # Far apart: use strong constant force to pull together quickly
                                force_mag = self.compactness * self.k_attract * adj_weight * 0.5
                            elif local_gap_ratio < 0.1:
                                # Close: use minimum force to prevent stalling
                                min_force = 0.1 * self.k_attract
                                force_mag = max(
                                    self.compactness * self.k_attract * adj_weight * gap,
                                    min_force * self.compactness * adj_weight,
                                )
                            else:
                                # Medium range: proportional force
                                force_mag = self.compactness * self.k_attract * adj_weight * gap
                            force = force_mag * direction
                            forces[i] += force
                            forces[j] -= force

    def run(
        self,
        max_iterations: int = 500,
        tolerance: float = 1e-4,
        show_progress: bool = True,
        save_history: bool = False,
    ) -> tuple[NDArray[np.floating], dict[str, Any], list[NDArray[np.floating]] | None]:
        """Run simulation until convergence or max iterations.

        Returns
        -------
        positions : np.ndarray
            Final symbol positions (in original coordinate system).
        info : dict
            Simulation statistics.
        history : list or None
            Position history if save_history=True (in original coordinates).

        """
        history: list[NDArray[np.floating]] | None = [] if save_history else None
        velocity_history: list[float] = []
        overlap_history: list[int] = []

        iterator: Any = range(max_iterations)
        if show_progress:
            iterator = tqdm(iterator, desc="Resolving overlaps", leave=False)

        converged = False
        final_overlaps = 0
        final_velocity = 0.0
        iterations_run = 0
        overlap_free_streak = 0

        for _ in iterator:
            iterations_run += 1
            if history is not None:
                history.append(self.positions * self.scale + self.center)

            # Always pass apply_attraction=True - step() internally only applies
            # attraction when n_overlaps == 0
            n_overlaps, max_velocity, gap_ratio = self.step(apply_attraction=True)
            velocity_history.append(max_velocity)
            overlap_history.append(n_overlaps)

            if overlap_free_streak >= 10:
                pass
            final_overlaps = n_overlaps
            final_velocity = max_velocity

            # Track overlap-free iterations
            if n_overlaps == 0:
                overlap_free_streak += 1
            else:
                overlap_free_streak = 0

            if show_progress:
                # Phase is "settling" when no overlaps (attraction was applied)
                phase = "settling" if n_overlaps == 0 else "separating"
                postfix = {"phase": phase, "overlaps": n_overlaps}
                if n_overlaps == 0:
                    postfix["gap"] = f"{gap_ratio:.1%}"
                postfix["vel"] = f"{max_velocity:.2e}"
                iterator.set_postfix(postfix)

            # Check convergence
            if n_overlaps == 0:
                if self.adjacency is not None and self.compactness > 0:
                    # With neighbor attraction: converge when neighbors are close to touching
                    # gap_ratio < 0.02 means within 2% of target distance
                    if gap_ratio < 0.02:
                        converged = True
                        break
                # No neighbor attraction - converge on velocity
                elif max_velocity < tolerance:
                    converged = True
                    break

        info = {
            "iterations": iterations_run,
            "converged": converged,
            "final_overlaps": final_overlaps,
            "final_max_velocity": final_velocity,
            "velocity_history": velocity_history,
            "overlap_history": overlap_history,
        }

        final_positions = self.positions * self.scale + self.center
        return final_positions, info, history
