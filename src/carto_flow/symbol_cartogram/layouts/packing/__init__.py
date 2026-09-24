"""Two-stage circle packing layout: CirclePackingLayout and CirclePackingLayoutOptions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ...options import ForceMode
from ..base import Layout, _apply_kwargs_to_options, _build_force_layout_result
from ..data_prep import LayoutData
from ..layout_result import LayoutResult

__all__ = [
    "CirclePackingAdvancedOptions",
    "CirclePackingLayout",
    "CirclePackingLayoutOptions",
    "PackingHistory",
    "PackingMetrics",
]


@dataclass
class PackingHistory:
    """Per-iteration diagnostics for CirclePackingLayout."""

    drift: NDArray[np.floating]
    jitter: NDArray[np.floating]
    drift_rate: NDArray[np.floating]
    drift_rate_neg_frac: NDArray[np.floating]


@dataclass
class PackingMetrics:
    """Final scalar metrics for CirclePackingLayout."""

    final_drift: float
    final_jitter: float


@dataclass
class CirclePackingAdvancedOptions:
    """Low-level algorithm tuning for CirclePackingLayout.

    These parameters rarely need changing. The defaults are calibrated for
    typical cartogram inputs. Override only when you have a specific reason.

    Parameters
    ----------
    local_step_fraction : float
        Fraction of overlap to correct per Gauss-Seidel step in Stage 1 (0-1].
        Lower values (e.g. 0.2) reduce intermingling in dense configurations
        at the cost of more passes. Default: 0.5
    expansion_max_iterations : int
        Maximum Gauss-Seidel passes for Stage 1 overlap resolution. Exits
        early when max overlap falls below ``overlap_tolerance``. Default: 500
    topology_gate_distance : float
        Topology forces only act when circles are within
        ``topology_gate_distance * (r_i + r_j)`` of each other. Default: 2.5
    contact_tolerance : float
        Contact detection threshold as fraction of radii sum. Default: 0.02
    contact_iterations : int
        Contact reaction passes per Stage 2 packing step. Default: 3
    contact_elasticity : float
        Net compression vs bounce behavior at contact points (-1 to 1).
        0 = neutral; < 0 = compression; > 0 = bounce. Only has effect when
        ``contact_transfer_ratio`` (core option) is between 0 and 1.
        Default: 0.0
    size_sensitivity : float
        How step size scales with circle radius in Stage 2 (-1 to 1).
        0 = all circles use avg_radius; 1 = proportional to radius;
        -1 = inverse. Default: 0.0
    overlap_projection_iters : int
        Jacobi overlap projection passes per Stage 2 packing step. Default: 5
    step_smoothing_window : int
        EMA effective window size for step smoothing. Default: 20
    convergence_window : int
        EMA effective window size for displacement convergence tracking.
        Default: 50
    adaptive_ema : bool
        Whether EMA uses adaptive warmup (alpha starts at 1/k). Default: True

    """

    # Stage 1: overlap resolution
    local_step_fraction: float = 0.5
    expansion_max_iterations: int = 500

    # Stage 2: force gating
    topology_gate_distance: float = 2.5

    # Stage 2: contact reaction
    contact_tolerance: float = 0.02
    contact_iterations: int = 3
    contact_elasticity: float = 0.0

    # Stage 2: step scaling
    size_sensitivity: float = 0.0
    overlap_projection_iters: int = 5

    # EMA configuration
    step_smoothing_window: int = 20
    convergence_window: int = 50
    adaptive_ema: bool = True

    def validate(self) -> None:
        """Validate advanced options."""
        errors = []
        if not 0 < self.local_step_fraction <= 1:
            errors.append("local_step_fraction must be > 0 and <= 1")
        if self.expansion_max_iterations < 1:
            errors.append("expansion_max_iterations must be >= 1")
        if self.topology_gate_distance <= 0:
            errors.append("topology_gate_distance must be positive")
        if not 0 < self.contact_tolerance < 1:
            errors.append("contact_tolerance must be > 0 and < 1")
        if self.contact_iterations < 1:
            errors.append("contact_iterations must be >= 1")
        if not -1 <= self.contact_elasticity <= 1:
            errors.append("contact_elasticity must be between -1 and 1")
        if self.overlap_projection_iters < 1:
            errors.append("overlap_projection_iters must be >= 1")
        if self.step_smoothing_window < 1:
            errors.append("step_smoothing_window must be >= 1")
        if self.convergence_window < 1:
            errors.append("convergence_window must be >= 1")
        if errors:
            raise ValueError("; ".join(errors))


@dataclass
class CirclePackingLayoutOptions:
    """Options for CirclePackingLayout (two-stage with contact reaction).

    Two-stage approach:

    - **Stage 1** (Feasibility): Single global expansion + Gauss-Seidel
      overlap resolution to reach a non-overlapping configuration.
    - **Stage 2** (Refinement): Force-based packing with contact reaction,
      topology and neighbor forces, and optional origin/group attraction.

    Low-level algorithm tuning is available via the ``advanced`` field.

    Parameters
    ----------
    max_iterations : int
        Maximum Stage 2 packing iterations. 0 = Stage 1 only (no compaction).
        Default: 500
    expansion : float
        Stage 1 expansion fraction (0-1). 0 = GS-only (most topology-
        preserving); 1 = full exact expansion (fastest, default); intermediate
        = partial expansion with GS handling residuals. Default: 1.0
    convergence_tolerance : float
        Convergence threshold for mean relative displacement. Default: 0.025
    spacing : float
        Minimum gap between symbols as fraction of average radius. Default: 0.05
    compactness : float
        Global centroid attraction strength (0-1). Pulls all circles toward
        the weighted centroid of the layout. Default: 0.1
    topology_weight : float
        Topology preservation strength (0-1). Pulls each adjacent pair toward
        their original relative angle. Default: 1.0
    overlap_tolerance : float
        Stage 1 convergence: max overlap as fraction of average radius.
        Default: 1e-4
    neighbor_weight : float
        Neighbor tangency force strength. Pulls separated adjacent circles
        together. Default: 1.0
    origin_weight : float
        Origin attraction strength. Pulls each circle toward its starting
        position (geographic centroid, or group centroid when
        ``collapse_group`` is used). Default: 0.1
    group_weight : float
        Group centroid attraction strength during Stage 2. Active when
        ``group_by`` or ``tile_count`` is set. Use ``collapse_group`` in
        ``create_layout`` to prevent intermingling before Stage 2. Default: 0.0
    force_mode : ForceMode
        How attraction force magnitude scales with distance. Applies to
        compactness, origin, and group forces. Default: ForceMode.DIRECTION

        - DIRECTION: constant magnitude with drop-off near target
        - LINEAR: proportional to distance (spring-like)
        - NORMALIZED: proportional to distance / radius
    max_step : float
        Maximum step size per iteration as fraction of average radius.
        Default: 0.3
    contact_transfer_ratio : float
        Balance between dissipating (0) and transferring (1) compressive
        forces at contact points. Default: 0.5
    advanced : CirclePackingAdvancedOptions
        Low-level algorithm tuning. See :class:`CirclePackingAdvancedOptions`.

    """

    # Convergence / Stage 1
    max_iterations: int = 500
    expansion: float = 1.0
    convergence_tolerance: float = 0.025

    # Geometry
    spacing: float = 0.05

    # Stage 2: force weights
    compactness: float = 0.1
    topology_weight: float = 1.0
    neighbor_weight: float = 1.0
    origin_weight: float = 0.1
    group_weight: float = 0.0
    force_mode: ForceMode = ForceMode.DIRECTION

    # Stage 2: step control
    max_step: float = 0.3
    contact_transfer_ratio: float = 0.5

    # Stage 1: convergence tolerance
    overlap_tolerance: float = 1e-4

    # Advanced tuning
    advanced: CirclePackingAdvancedOptions = field(default_factory=CirclePackingAdvancedOptions)

    def validate(self) -> None:
        """Validate options."""
        errors = []
        if self.max_iterations < 0:
            errors.append("max_iterations must be non-negative (0 = Stage 1 only)")
        if not 0 <= self.expansion <= 1:
            errors.append("expansion must be between 0 and 1")
        if self.convergence_tolerance <= 0:
            errors.append("convergence_tolerance must be positive")
        if not 0 <= self.spacing <= 1:
            errors.append("spacing must be between 0 and 1")
        if not 0 <= self.compactness <= 1:
            errors.append("compactness must be between 0 and 1")
        if not 0 <= self.topology_weight <= 1:
            errors.append("topology_weight must be between 0 and 1")
        if self.overlap_tolerance <= 0:
            errors.append("overlap_tolerance must be positive")
        if self.neighbor_weight < 0:
            errors.append("neighbor_weight must be non-negative")
        if self.origin_weight < 0:
            errors.append("origin_weight must be non-negative")
        if self.group_weight < 0:
            errors.append("group_weight must be non-negative")
        if self.max_step <= 0:
            errors.append("max_step must be positive")
        if not 0 <= self.contact_transfer_ratio <= 1:
            errors.append("contact_transfer_ratio must be between 0 and 1")
        if errors:
            raise ValueError("; ".join(errors))
        self.advanced.validate()


class CirclePackingLayout(Layout):
    """Layout using two-stage circle packing simulation.

    Two-stage approach:
    1. Overlap Resolution: Global expansion + Gauss-Seidel separation
    2. Circle Packing: Force-based refinement with contact reaction

    Parameters
    ----------
    options : CirclePackingLayoutOptions, optional
        Full options object. Defaults to CirclePackingLayoutOptions().
    **kwargs
        Individual option overrides (core options only; use
        ``advanced=CirclePackingAdvancedOptions(...)`` for advanced tuning).

    """

    #: ``group_by`` drives the Stage 2 group attraction force (``group_weight``).
    supports_group_by = True

    # ------------------------------------------------------------------
    # Named presets
    # ------------------------------------------------------------------

    @classmethod
    def centroid(cls, *, spacing: float = 0.05, expansion: float = 0.0) -> CirclePackingLayout:
        """Stage 1 only: place at geographic centroids then separate overlaps.

        No Stage 2 compaction (``max_iterations=0``). ``expansion=0`` (default)
        uses Gauss-Seidel only (more topology-preserving); ``expansion=1`` adds
        a single global expansion step (faster for large overlaps).
        """
        return cls(
            spacing=spacing,
            expansion=expansion,
            topology_weight=0,
            neighbor_weight=0,
            compactness=0,
            origin_weight=0,
            max_iterations=0,
        )

    @classmethod
    def dorling(
        cls,
        *,
        spacing: float = 0.05,
        compactness: float = 0.8,
        topology_weight: float = 0.0,
        neighbor_weight: float = 0.0,
    ) -> CirclePackingLayout:
        """Compact circles pulled toward the global centroid — classic Dorling style.

        No origin attraction. Enable ``topology_weight`` / ``neighbor_weight``
        to add spatial structure while keeping the compact character.
        """
        return cls(
            spacing=spacing,
            topology_weight=topology_weight,
            neighbor_weight=neighbor_weight,
            compactness=compactness,
            origin_weight=0,
        )

    @classmethod
    def geographic(
        cls,
        *,
        spacing: float = 0.05,
        origin_weight: float = 0.5,
        topology_weight: float = 0.5,
        neighbor_weight: float = 0.5,
    ) -> CirclePackingLayout:
        """Geography-preserving — symbols stay close to original positions.

        No global compaction. Reduce ``topology_weight`` / ``neighbor_weight``
        for a looser layout that still tracks geography.
        """
        return cls(
            spacing=spacing,
            topology_weight=topology_weight,
            neighbor_weight=neighbor_weight,
            compactness=0,
            origin_weight=origin_weight,
        )

    @classmethod
    def dorling_grouped(
        cls,
        *,
        spacing: float = 0.05,
        compactness: float = 0.1,
        group_weight: float = 0.5,
        topology_weight: float = 0.5,
        neighbor_weight: float = 0.5,
    ) -> CirclePackingLayout:
        """Grouped Dorling — circles pack toward group centroids.

        Intended with ``collapse_group=1.0`` so symbols start coincident within
        each group. ``group_weight`` provides the main cohesion force;
        ``compactness`` adds mild global pull. ``group_weight`` is inert when no
        groups are defined.
        """
        return cls(
            spacing=spacing,
            topology_weight=topology_weight,
            neighbor_weight=neighbor_weight,
            compactness=compactness,
            origin_weight=0,
            group_weight=group_weight,
        )

    @classmethod
    def geographic_grouped(
        cls,
        *,
        spacing: float = 0.05,
        origin_weight: float = 0.5,
        group_weight: float = 0.3,
        topology_weight: float = 0.5,
        neighbor_weight: float = 0.5,
    ) -> CirclePackingLayout:
        """Grouped geography-preserving — symbols stay near geographic positions.

        Origin tracks the group starting centroid (geographic or collapsed)
        when used with ``collapse_group``. ``group_weight`` adds soft intra-group
        cohesion on top of origin attraction.
        """
        return cls(
            spacing=spacing,
            topology_weight=topology_weight,
            neighbor_weight=neighbor_weight,
            compactness=0,
            origin_weight=origin_weight,
            group_weight=group_weight,
        )

    # ------------------------------------------------------------------

    def __init__(self, options: CirclePackingLayoutOptions | None = None, /, **kwargs) -> None:
        if options is None:
            options = CirclePackingLayoutOptions()
        self._options = _apply_kwargs_to_options(options, kwargs)
        self._options.validate()

    def _inert_group_by_warning(self) -> str | None:
        """Warn when group_by was given but the group force is switched off."""
        if self._options.group_weight > 0:
            return None
        return (
            "group_by was given to CirclePackingLayout but group_weight is 0.0, so the grouping "
            "does not affect placement. Set group_weight (e.g. CirclePackingLayout(group_weight=0.5), "
            "optionally with collapse_group in create_layout) to group symbols."
        )

    def _compute(self, data: LayoutData, show_progress: bool = True, save_history: bool = False) -> LayoutResult:
        """Run circle packing simulation and return result.

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
        from ._simulator import TopologyPreservingSimulator

        adv = self._options.advanced

        # original_positions: use the simulation starting positions (data.positions),
        # which reflect any collapse_group pre-processing applied in data_prep.
        # This keeps the bounding box tight when positions are collapsed, matching
        # tile_count behavior. origin_weight then attracts toward the starting
        # configuration (group centroid when collapsed, geography otherwise).
        original_positions = data.positions

        sim = TopologyPreservingSimulator(
            positions=data.positions,
            original_positions=original_positions,
            radii=data.sizes,
            adjacency=data.adjacency,
            spacing=self._options.spacing,
            compactness=self._options.compactness,
            topology_weight=self._options.topology_weight,
            overlap_tolerance=self._options.overlap_tolerance,
            expansion_max_iterations=adv.expansion_max_iterations,
            expansion=self._options.expansion,
            topology_gate_distance=adv.topology_gate_distance,
            neighbor_weight=self._options.neighbor_weight,
            origin_weight=self._options.origin_weight,
            group_ids=data.group_ids,
            group_weight=self._options.group_weight,
            force_mode=self._options.force_mode.value,
            contact_tolerance=adv.contact_tolerance,
            contact_iterations=adv.contact_iterations,
            max_step=self._options.max_step,
            contact_transfer_ratio=self._options.contact_transfer_ratio,
            contact_elasticity=adv.contact_elasticity,
            size_sensitivity=adv.size_sensitivity,
            local_step_fraction=adv.local_step_fraction,
            overlap_projection_iters=adv.overlap_projection_iters,
            step_smoothing_window=adv.step_smoothing_window,
            convergence_window=adv.convergence_window,
            adaptive_ema=adv.adaptive_ema,
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
            algorithm=PackingHistory(
                drift=np.array(info.pop("drift_history")),
                jitter=np.array(info.pop("jitter_history")),
                drift_rate=np.array(info.pop("drift_rate_history")),
                drift_rate_neg_frac=np.array(info.pop("drift_rate_neg_frac_history")),
            ),
        )
        metrics = AlgorithmMetrics(
            converged=info.pop("converged"),
            iterations=info.pop("iterations"),
            final_overlaps=info.pop("final_overlaps"),
            algorithm=PackingMetrics(
                final_drift=info.pop("final_drift"),
                final_jitter=info.pop("final_jitter"),
            ),
        )
        return _build_force_layout_result(
            positions, info, history, data, "packing", metrics=metrics, sim_history=sim_history
        )
