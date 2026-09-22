"""Physics-based circle layout: CirclePhysicsLayout and CirclePhysicsLayoutOptions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..base import Layout, _apply_kwargs_to_options, _build_physics_layout_result
from ..data_prep import LayoutData
from ..layout_result import LayoutResult

__all__ = ["CirclePhysicsLayout", "CirclePhysicsLayoutOptions", "PhysicsHistory", "PhysicsMetrics"]


@dataclass
class PhysicsHistory:
    """Per-iteration diagnostics for CirclePhysicsLayout."""

    velocity: NDArray[np.floating]


@dataclass
class PhysicsMetrics:
    """Final scalar metrics for CirclePhysicsLayout."""

    final_max_velocity: float


@dataclass
class CirclePhysicsLayoutOptions:
    """Options for CirclePhysicsLayout (velocity-based physics).

    This simulator uses a two-phase approach:
    1. Separation phase: Strong repulsion until overlaps resolved
    2. Settling phase: Gentle attraction while maintaining separation

    Parameters
    ----------
    max_iterations : int
        Maximum iterations for simulation. Default: 500
    convergence_tolerance : float
        Threshold for declaring convergence. Default: 1e-4
    damping : float
        Velocity damping factor (0-1, exclusive). Higher values cause
        faster energy dissipation. Default: 0.85
    dt : float
        Integration timestep. Larger values give faster but less stable
        simulation. Default: 0.15
    max_velocity : float
        Maximum velocity magnitude (clamped). Default: 3.0
    k_repel : float
        Repulsion force coefficient. Higher values push overlapping
        circles apart more aggressively. Default: 15.0
    k_attract : float
        Base attraction force coefficient. Default: 2.0

    """

    # Simulation parameters
    max_iterations: int = 500
    convergence_tolerance: float = 1e-4

    # High-level placement parameters (passed to simulator constructor)
    spacing: float = 0.05  # Gap as fraction of avg symbol size
    compactness: float = 0.5  # 0 = centroid only, 1 = neighbor only
    topology_weight: float = 0.3  # 0 = ignore topology, 1 = strong

    # Physics-specific parameters
    damping: float = 0.85
    dt: float = 0.15
    max_velocity: float = 3.0
    k_repel: float = 15.0
    k_attract: float = 2.0

    def validate(self) -> None:
        """Validate options."""
        errors = []
        if self.max_iterations < 1:
            errors.append("max_iterations must be positive")
        if self.convergence_tolerance <= 0:
            errors.append("convergence_tolerance must be positive")
        if not 0 <= self.spacing <= 1:
            errors.append("spacing must be between 0 and 1")
        if not 0 <= self.compactness <= 1:
            errors.append("compactness must be between 0 and 1")
        if not 0 <= self.topology_weight <= 1:
            errors.append("topology_weight must be between 0 and 1")
        if not 0 < self.damping < 1:
            errors.append("damping must be between 0 and 1 (exclusive)")
        if self.dt <= 0:
            errors.append("dt must be positive")
        if self.max_velocity <= 0:
            errors.append("max_velocity must be positive")
        if self.k_repel < 0:
            errors.append("k_repel must be non-negative")
        if self.k_attract < 0:
            errors.append("k_attract must be non-negative")
        if errors:
            raise ValueError("; ".join(errors))


class CirclePhysicsLayout(Layout):
    """Layout using velocity-based physics simulation.

    Two-phase approach:
    1. Separation phase: Strong repulsion until overlaps resolved
    2. Settling phase: Gentle attraction while maintaining separation

    Parameters
    ----------
    options : CirclePhysicsLayoutOptions, optional
        Full options object. Defaults to CirclePhysicsLayoutOptions().
    **kwargs
        Individual option overrides.

    """

    def __init__(self, options: CirclePhysicsLayoutOptions | None = None, /, **kwargs) -> None:
        if options is None:
            options = CirclePhysicsLayoutOptions()
        self._options = _apply_kwargs_to_options(options, kwargs)
        self._options.validate()

    def _compute(self, data: LayoutData, show_progress: bool = True, save_history: bool = False) -> LayoutResult:
        """Run physics simulation and return result.

        Parameters
        ----------
        data : LayoutData
            Preprocessed layout data.
        show_progress : bool
            Display progress feedback during placement.
        save_history : bool
            Record position snapshots per iteration.

        Returns
        -------
        LayoutResult
            Immutable layout result with CircleSymbol as canonical.

        """
        from ._simulator import CirclePhysicsSimulator

        sim = CirclePhysicsSimulator(
            positions=data.positions,
            radii=data.sizes,
            adjacency=data.adjacency,
            spacing=self._options.spacing,
            compactness=self._options.compactness,
            topology_weight=self._options.topology_weight,
            damping=self._options.damping,
            dt=self._options.dt,
            max_velocity=self._options.max_velocity,
            k_repel=self._options.k_repel,
            k_attract=self._options.k_attract,
        )
        positions, info, history = sim.run(
            max_iterations=self._options.max_iterations,
            tolerance=self._options.convergence_tolerance,
            show_progress=show_progress,
            save_history=save_history,
        )

        from ..layout_result import AlgorithmMetrics, SimulationHistory

        sim_history = SimulationHistory(
            positions=[np.array(h) for h in history] if history is not None else None,
            overlaps=np.array(info.pop("overlap_history"), dtype=np.intp),
            algorithm=PhysicsHistory(
                velocity=np.array(info.pop("velocity_history")),
            ),
        )
        metrics = AlgorithmMetrics(
            converged=info.pop("converged"),
            iterations=info.pop("iterations"),
            final_overlaps=info.pop("final_overlaps"),
            algorithm=PhysicsMetrics(
                final_max_velocity=info.pop("final_max_velocity"),
            ),
        )
        return _build_physics_layout_result(positions, info, history, data, metrics=metrics, sim_history=sim_history)
