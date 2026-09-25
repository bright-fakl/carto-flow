"""
Benchmarks for flow cartogram performance.

Two groups:

Group A — parallelism & bounding-box filter (fixed iterations)
    All cases run exactly n_iter=100 iterations so wall-time comparisons are clean.

Group B — morph_gdf vs multiresolution_morph (convergence)
    Both run to convergence.  niterations and status are stored in extra_info
    to explain any timing difference.

Run with:
    uv run pytest benchmarks/ --benchmark-enable --benchmark-sort=name -v
"""

import gc

import pytest

from carto_flow.flow_cartogram import MorphOptions, morph_gdf
from carto_flow.flow_cartogram.timings import BenchmarkRuns


def _gc_collect():
    """Run gc.collect() and return None.

    pytest-benchmark's pedantic() interprets a truthy return value from setup
    as (args, kwargs) to unpack.  gc.collect() returns the number of objects
    freed, which can be non-zero, causing a TypeError.  Wrapping it here
    ensures setup always returns None.
    """
    gc.collect()


# ---------------------------------------------------------------------------
# Group A: fixed-iteration cases
# ---------------------------------------------------------------------------

_FIXED = {
    "n_iter": 100,
    "show_progress": False,
    # Tight tolerances ensure the algorithm does not converge within n_iter iterations,
    # so every Group A case performs the same amount of work.
    "mean_tol": 0.02,
    "max_tol": 0.05,
    "stall_patience": None,
    "benchmark": True,
}

_NPARALLEL = 8

_PARALLEL_CASES = [
    pytest.param({"parallel_fft": False, "parallel_density": False}, id="serial"),
    pytest.param({"parallel_fft": _NPARALLEL, "parallel_density": _NPARALLEL}, id="parallel"),
    pytest.param({"parallel_fft": _NPARALLEL, "parallel_density": False}, id="parallel (velocity)"),
    pytest.param({"parallel_fft": False, "parallel_density": _NPARALLEL}, id="parallel (density)"),
]

_GRID_SIZES = [256, 512, 1024, 2048]

_WARMUP_ROUNDS = 1
_ROUNDS = 10


@pytest.mark.parametrize("grid_size", _GRID_SIZES)
@pytest.mark.parametrize("overrides", _PARALLEL_CASES)
def test_parallel_options(benchmark, us_states_gdf, overrides, grid_size):
    options = MorphOptions(**_FIXED, grid_size=grid_size, **overrides)
    runs = BenchmarkRuns()

    def run():
        r = morph_gdf(us_states_gdf, "Population", options=options)
        runs.add(r.benchmark)
        return r

    # Fewer rounds for large grids: peak ~3.9 GB at 4096 means back-to-back rounds
    # can OOM if GC hasn't freed previous allocations. setup=gc.collect runs before
    # each round (untimed) to force reclamation. Fewer rounds are statistically fine
    # at large sizes because longer runs have smaller relative variance.
    benchmark.pedantic(run, setup=_gc_collect, rounds=_ROUNDS, warmup_rounds=_WARMUP_ROUNDS, iterations=1)
    benchmark.extra_info.update(runs[_WARMUP_ROUNDS:].to_dict())


# ---------------------------------------------------------------------------
# Group B: convergence comparison
# ---------------------------------------------------------------------------

# No benchmark=True — Group B is about total wall time, not phase breakdown.
_CONVERGENCE_OPTS = MorphOptions(n_iter=500, grid_size=512, show_progress=False, mean_tol=0.005, max_tol=0.01)


def test_morph_gdf_convergence(benchmark, us_states_gdf):
    def run():
        return morph_gdf(us_states_gdf, "Population", options=_CONVERGENCE_OPTS)

    result = benchmark.pedantic(run, rounds=_ROUNDS, warmup_rounds=_WARMUP_ROUNDS, iterations=1)
    errors = result.get_errors()
    benchmark.extra_info.update(
        niterations=result.niterations,
        status=str(result.status),
        mean_error_pct=round(errors.mean_error_pct, 2) if errors else None,
        max_error_pct=round(errors.max_error_pct, 2) if errors else None,
    )


def test_multiresolution_convergence(benchmark, us_states_gdf):
    from carto_flow.flow_cartogram.workflow import CartogramWorkflow

    def run():
        workflow = CartogramWorkflow(
            us_states_gdf,
            "Population",
            options=MorphOptions(n_iter=500, show_progress=False, mean_tol=0.005, max_tol=0.01),
        )
        workflow.morph_multiresolution(min_resolution=64, levels=4)
        return workflow

    workflow = benchmark.pedantic(run, rounds=_ROUNDS, warmup_rounds=_WARMUP_ROUNDS, iterations=1)
    final = workflow.latest
    errors = final.get_errors()
    # Sum iterations across all morphed levels (results[0] is the original, niterations=0)
    total_iters = sum(c.niterations for c in workflow.results[1:])
    benchmark.extra_info.update(
        niterations=total_iters,
        status=str(final.status),
        mean_error_pct=round(errors.mean_error_pct, 2) if errors else None,
        max_error_pct=round(errors.max_error_pct, 2) if errors else None,
    )


# ---------------------------------------------------------------------------
# Group C: congressional districts (high geometry count)
# ---------------------------------------------------------------------------

_CONGRESSIONAL_GRID_SIZES = [256, 512, 1024, 2048]

_CONGRESSIONAL_FIXED = {
    "n_iter": 100,
    "show_progress": False,
    "mean_tol": 0.02,
    "max_tol": 0.05,
    "stall_patience": None,
    "benchmark": True,
}


@pytest.mark.parametrize("grid_size", _CONGRESSIONAL_GRID_SIZES)
@pytest.mark.parametrize("overrides", _PARALLEL_CASES)
def test_congressional_districts(benchmark, us_congressional_districts_gdf, overrides, grid_size):
    options = MorphOptions(**_CONGRESSIONAL_FIXED, grid_size=grid_size, **overrides)

    runs = BenchmarkRuns()

    def run():
        r = morph_gdf(us_congressional_districts_gdf, "Population", options=options)
        runs.add(r.benchmark)
        return r

    benchmark.pedantic(run, rounds=_ROUNDS, warmup_rounds=_WARMUP_ROUNDS, iterations=1)
    benchmark.extra_info.update(runs[_WARMUP_ROUNDS:].to_dict())


# ---------------------------------------------------------------------------
# Group D: vertex scaling (US states, fixed grid, varying simplification)
# ---------------------------------------------------------------------------

# Vertex-count cases, coarse to fine. The bundled census snapshot is
# coverage-simplified at 1000 m, so `simplify` below that re-fetches finer
# geometry from the Census API and needs a key, which CI does not have. Detail
# above the bundled resolution is produced by densifying instead: segmentizing
# splits existing edges, so it raises the vertex count the morph has to advect
# without needing source geometry the package does not ship.
#
# ("simplify", tol) drops vertices; ("segmentize", max_len) adds them; None is
# the bundled geometry. Spans roughly 5k to 240k vertices on contiguous states.
_VERTEX_CASES = [
    ("simplify", 10_000),
    ("simplify", 5_000),
    None,
    ("segmentize", 2_000),
    ("segmentize", 1_000),
    ("segmentize", 500),
]


def _vertex_case_id(case):
    return "bundled" if case is None else f"{case[0]}-{case[1]}"


def _load_states_at(case):
    """Contiguous US states at the requested vertex detail."""
    import shapely

    from carto_flow.data import load_us_census

    if case is not None and case[0] == "simplify":
        return load_us_census(population=True, contiguous_only=True, simplify=case[1])
    gdf = load_us_census(population=True, contiguous_only=True)
    if case is not None:
        gdf = gdf.copy()
        gdf["geometry"] = shapely.segmentize(gdf.geometry.values, case[1])
    return gdf


_VERTEX_FIXED = {
    "n_iter": 100,
    "show_progress": False,
    "grid_size": 512,
    "mean_tol": 0.02,
    "max_tol": 0.05,
    "stall_patience": None,
    "benchmark": True,
}


@pytest.mark.parametrize("overrides", _PARALLEL_CASES)
@pytest.mark.parametrize("vertex_case", _VERTEX_CASES, ids=_vertex_case_id)
def test_vertex_scaling(benchmark, vertex_case, overrides):
    gdf = _load_states_at(vertex_case)
    options = MorphOptions(**_VERTEX_FIXED, **overrides)
    runs = BenchmarkRuns()

    def run():
        r = morph_gdf(gdf, "Population", options=options)
        runs.add(r.benchmark)
        return r

    benchmark.pedantic(run, setup=_gc_collect, rounds=5, warmup_rounds=_WARMUP_ROUNDS, iterations=1)
    benchmark.extra_info.update(runs[_WARMUP_ROUNDS:].to_dict())
