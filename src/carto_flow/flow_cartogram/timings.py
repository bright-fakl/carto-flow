"""Per-phase wall-clock timing for the flow cartogram algorithm.

Populated when ``MorphOptions.benchmark=True``.

Classes
-------
Benchmark
    Accumulated wall-clock time (seconds) per algorithm phase.
BenchmarkRuns
    Collects multiple :class:`Benchmark` instances across rounds and
    exports them as a dict-of-lists for ``benchmark.extra_info`` or
    ``pd.DataFrame()``.
"""

import dataclasses

from carto_flow._benchmark import BenchmarkBase, BenchmarkRuns

__all__ = ["Benchmark", "BenchmarkRuns"]


@dataclasses.dataclass
class Benchmark(BenchmarkBase):
    """Accumulated wall-clock time (seconds) per algorithm phase.

    Populated when ``MorphOptions.benchmark=True``.
    All times are measured with ``time.perf_counter()``.

    Inherits common fields from :class:`~carto_flow._benchmark.BenchmarkBase`:
    ``setup_s``, ``other_s``, ``total_s``, ``niterations``, ``status``,
    ``mean_error_pct``, ``max_error_pct``.

    ``total_s`` is the full wall time from function entry to result
    construction, identical to ``Cartogram.duration``.  ``setup_s`` is
    the pre-loop portion; the loop portion is ``total_s - setup_s``.

    ``other_s`` captures loop time not attributed to the five named loop
    phases (progress-bar updates, snapshot saving, convergence bookkeeping,
    etc.) and is computed as
    ``total_s - setup_s - sum(named loop phases)``.

    Attributes
    ----------
    density_s : float
        Time spent in ``compute_density_field_from_geometries``.
        Only charged on iterations where the density is recomputed
        (every ``recompute_every`` iterations).
    velocity_s : float
        Time spent in ``velocity_computer.compute(rho)`` (FFT Poisson solve).
        Only charged on the same iterations as ``density_s``.
    displacement_s : float
        Time spent displacing geometry coordinates, landmarks, and any
        additional displacement-field coordinates via ``displace_coords_numba``.
        Charged every iteration.
    areas_s : float
        Time spent in ``flat_geoms.compute_areas()``.
        Charged every iteration.
    errors_s : float
        Time spent in ``compute_error_metrics()``.
        Charged every iteration.
    snapshot_s : float
        Time spent reconstructing geometries and building ``CartogramSnapshot``
        objects (charged only on iterations where a snapshot is saved).
    density_calls : int
        Number of iterations on which the density field was recomputed.
    """

    # Flow-specific timing fields (end in _s → auto-prefixed with t_ by to_dict)
    density_s: float = 0.0
    velocity_s: float = 0.0
    displacement_s: float = 0.0
    areas_s: float = 0.0
    errors_s: float = 0.0
    snapshot_s: float = 0.0

    # Flow-specific metadata
    density_calls: int = 0
    fft_threads: int = 0
    density_threads: int = 0
    n_geometries: int = 0
    n_vertices: int = 0
    n_landmark_vertices: int = 0
    n_coord_points: int = 0
