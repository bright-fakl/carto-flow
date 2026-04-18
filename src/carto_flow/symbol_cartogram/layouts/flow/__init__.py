"""Flow-density layout: FlowDensityLayout and FlowDensityLayoutOptions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..base import Layout, _apply_kwargs_to_options, _build_physics_layout_result
from ..data_prep import LayoutData
from ..layout_result import LayoutResult

__all__ = [
    "FlowDensityFieldSnapshots",
    "FlowDensityHistory",
    "FlowDensityLayout",
    "FlowDensityLayoutOptions",
    "FlowDensityMetrics",
]


@dataclass
class FlowDensityFieldSnapshots:
    """Density grid metadata and per-recompute snapshots.

    One instance per ``FlowDensityHistory`` when ``save_density_fields=True``.
    Snapshots are taken once per density recompute interval (every N iterations),
    not every iteration.

    Attributes
    ----------
    fields : list[dict]
        Each element is a dict with keys ``iteration``, ``rho``, ``vx``, ``vy``.
    bounds : tuple
        ``(xmin, ymin, xmax, ymax)`` of the density grid.
    shape : tuple
        ``(rows, cols)`` of the density grid.
    x_coords, y_coords : NDArray
        Cell-centre coordinates along each axis.
    """

    fields: list[dict]
    bounds: tuple
    shape: tuple
    x_coords: NDArray[np.floating]
    y_coords: NDArray[np.floating]


@dataclass
class FlowDensityHistory:
    """Simulation diagnostics for FlowDensityLayout.

    Attributes
    ----------
    errors, max_errors : NDArray
        Per-iteration mean and max relative NN spacing errors.
    density_snapshots : FlowDensityFieldSnapshots | None
        Density and velocity field snapshots, one per recompute interval.
        Only populated when ``save_density_fields=True``.
    """

    errors: NDArray[np.floating]
    max_errors: NDArray[np.floating]
    density_snapshots: FlowDensityFieldSnapshots | None = None


@dataclass
class FlowDensityMetrics:
    """Final scalar metrics for FlowDensityLayout."""

    final_error: float
    final_max_error: float
    final_signed_errors: NDArray[np.floating]


@dataclass
class FlowDensityLayoutOptions:
    """Options for FlowDensityLayout (Gaussian density-field advection).

    Positions circles by constructing a divergence field from per-pair
    Gaussian blobs placed at predicted contact points, then advecting
    centroids through the resulting velocity field.

    Parameters
    ----------
    spacing : float
        Minimum gap between circle boundaries as a fraction of the mean
        circle radius. Default: 0.05
    sigma_perp_factor : float
        Controls the perpendicular width of each contact-point Gaussian:
        sigma_perp = factor * min(r_i, r_j). Default: 1.0
    smooth : float
        Gaussian filter sigma for the density field in real-world coordinate
        units (same units as centroid positions). Converted internally to grid
        cells via grid.dx / grid.dy after the grid is constructed.
        0 = no smoothing. Default: 0.0
    damp : bool
        Apply exponential dampening to amplitude when circles are far from
        their target distance, preventing extreme distortion. Default: True
    use_gabriel : bool
        Use the Gabriel graph (subset of Delaunay) for adjacency pairs.
        False = full Delaunay triangulation. Default: True
    max_iterations : int
        Maximum number of advection steps. Default: 500
    recompute_every : int
        Rebuild density and velocity fields every N steps. Default: 5
    dt_factor : float
        Timestep = factor * min(dx, dy) / max_velocity. Default: 0.3
    convergence_tolerance : float
        Stop when mean relative NN spacing error < tolerance. Default: 0.05
    grid_size : int
        Grid resolution (square). Larger values improve accuracy at the
        cost of runtime. Default: 256
    save_density_fields : bool
        Record the density field (rho) and velocity fields (vx, vy) at every
        density recompute step. Results are accessible via
        ``layout_result.history.algorithm.density_snapshots``
        (a ``FlowDensityFieldSnapshots`` instance). Default: False
    force_balance : float or {"count", "rms", "repulse"}, default 1.0
        Relative weighting of push (overlapping) vs pull (too-far) pair
        contributions to the density field. Only the ratio push/pull affects
        the velocity field shape; the adaptive timestep absorbs any global scale.
        ``float > 0``: push_scale = force_balance, pull_scale = 1.0. Values
          > 1 amplify push relative to pull, helping resolve remaining overlaps.
        ``"count"``: push_scale = 1/N_push, pull_scale = 1/N_pull — each pair
          contributes equally regardless of push/pull count imbalance.
        ``"rms"``: scales so RMS(push field) = RMS(pull field) — equalises
          field energy per type.
        ``"repulse"``: pull_scale = 0 — only overlapping pairs contribute,
          so circles are separated without being packed together. Default: 1.0
    cross_group_pull_scale : float, default 1.0
        Pull-force multiplier for cross-group pairs (pairs where the two circles
        belong to different groups, as determined by ``group_by`` or
        ``tile_count``). 0.0 = repulse-only across groups (circles from
        different groups separate but do not attract); 1.0 = same as
        within-group (no distinction). Only has effect when the layout is run
        with ``group_by`` or ``tile_count``. Default: 1.0

    """

    spacing: float = 0.05
    sigma_perp_factor: float = 1.0
    smooth: float = 0.0
    damp: bool = True
    use_gabriel: bool = True
    max_iterations: int = 500
    recompute_every: int = 5
    dt_factor: float = 0.3
    convergence_tolerance: float = 0.05
    grid_size: int = 256
    save_density_fields: bool = False
    force_balance: float | str = 1.0
    cross_group_pull_scale: float = 1.0

    def validate(self) -> None:
        """Validate options."""
        errors = []
        if self.spacing < 0:
            errors.append("spacing must be >= 0")
        if self.sigma_perp_factor <= 0:
            errors.append("sigma_perp_factor must be positive")
        if self.smooth < 0:
            errors.append("smooth must be >= 0")
        if self.max_iterations < 1:
            errors.append("max_iterations must be >= 1")
        if self.recompute_every < 1:
            errors.append("recompute_every must be >= 1")
        if self.dt_factor <= 0:
            errors.append("dt_factor must be positive")
        if self.convergence_tolerance <= 0:
            errors.append("convergence_tolerance must be positive")
        if self.grid_size < 32:
            errors.append("grid_size must be >= 32")
        if isinstance(self.force_balance, str):
            if self.force_balance not in ("count", "rms", "repulse"):
                errors.append('force_balance must be a float > 0 or "count", "rms", or "repulse"')
        elif self.force_balance <= 0:
            errors.append("force_balance must be > 0")
        if not (0.0 <= self.cross_group_pull_scale <= 1.0):
            errors.append("cross_group_pull_scale must be in [0.0, 1.0]")
        if errors:
            raise ValueError("; ".join(errors))


class FlowDensityLayout(Layout):
    """Layout using Gaussian density-field flow advection.

    Positions circles by building a divergence field from per-pair Gaussian
    blobs placed at predicted contact points and advecting centroids through
    the resulting velocity field. Unlike physics-based layouts, the field
    covers the full domain so there is no background sink pulling circles
    into empty space.

    Returns LayoutResult with CircleSymbol as canonical symbol.

    Parameters
    ----------
    options : FlowDensityLayoutOptions, optional
        Full options object. Defaults to FlowDensityLayoutOptions().
    **kwargs
        Individual option overrides applied on top of *options*.

    Examples
    --------
    >>> layout = FlowDensityLayout()
    >>> layout = FlowDensityLayout(max_iterations=200, spacing=0.1)

    """

    def __init__(self, options: FlowDensityLayoutOptions | None = None, /, **kwargs) -> None:
        if options is None:
            options = FlowDensityLayoutOptions()
        self._options = _apply_kwargs_to_options(options, kwargs)
        self._options.validate()

    def compute(self, data: LayoutData, show_progress: bool = True, save_history: bool = False) -> LayoutResult:
        """Run flow-density advection and return result.

        Parameters
        ----------
        data : LayoutData
            Preprocessed layout data.
        show_progress : bool
            Print progress every 20 steps.
        save_history : bool
            Record position snapshots per step.

        Returns
        -------
        LayoutResult
            Immutable layout result with CircleSymbol as canonical.

        """
        from ._simulator import run_flow_density

        opts = self._options
        mean_radius = float(np.mean(data.sizes))

        spacing_abs = opts.spacing * mean_radius

        if data.source_indices is not None:
            group_ids = data.source_indices.astype(np.int32)
        elif data.group_ids is not None:
            group_ids = data.group_ids.astype(np.int32)
        else:
            group_ids = None

        positions, info, history = run_flow_density(
            centroids=data.positions,
            radii=data.sizes,
            spacing_abs=spacing_abs,
            grid_size=opts.grid_size,
            smooth=opts.smooth,
            sigma_perp_factor=opts.sigma_perp_factor,
            damp=opts.damp,
            use_gabriel=opts.use_gabriel,
            max_iterations=opts.max_iterations,
            recompute_every=opts.recompute_every,
            dt_factor=opts.dt_factor,
            convergence_tolerance=opts.convergence_tolerance,
            show_progress=show_progress,
            save_history=save_history,
            save_density_fields=opts.save_density_fields,
            force_balance=opts.force_balance,
            group_ids=group_ids,
            cross_group_pull_scale=opts.cross_group_pull_scale,
        )

        from ..layout_result import AlgorithmMetrics, SimulationHistory

        density_snapshots = None
        if "density_fields" in info:
            density_snapshots = FlowDensityFieldSnapshots(
                fields=info.pop("density_fields"),
                bounds=info.pop("grid_bounds"),
                shape=info.pop("grid_shape"),
                x_coords=info.pop("grid_x_coords"),
                y_coords=info.pop("grid_y_coords"),
            )

        history_arrays = [np.array(h) for h in history] if history is not None else None
        sim_history = SimulationHistory(
            positions=history_arrays,
            overlaps=np.array(info.pop("n_overlaps_history"), dtype=np.intp),
            algorithm=FlowDensityHistory(
                errors=np.array(info.pop("errors")),
                max_errors=np.array(info.pop("max_errors")),
                density_snapshots=density_snapshots,
            ),
        )
        metrics = AlgorithmMetrics(
            converged=info.pop("converged"),
            iterations=info.pop("iterations"),
            final_overlaps=info.pop("n_overlaps"),
            algorithm=FlowDensityMetrics(
                final_error=info.pop("final_error"),
                final_max_error=info.pop("final_max_error"),
                final_signed_errors=np.array(info.pop("final_signed_errors")),
            ),
        )
        return _build_physics_layout_result(
            positions, info, history_arrays, data, metrics=metrics, sim_history=sim_history
        )
