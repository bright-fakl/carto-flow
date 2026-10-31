"""
Benchmarks for Voronoi cartogram performance.

Runtime against the number of regions (US states, congressional districts)
with the default backend, weighted by population.

Run with:
    uv run pytest benchmarks/test_voronoi_cartogram.py --benchmark-enable -v
"""

import gc

import pytest

from carto_flow.voronoi_cartogram import create_voronoi_cartogram

_ROUNDS = 5
_WARMUP_ROUNDS = 1


def _gc_collect():
    gc.collect()


@pytest.mark.parametrize("dataset", ["us_states_gdf", "us_congressional_districts_gdf"])
def test_voronoi_runtime(benchmark, request, dataset):
    gdf = request.getfixturevalue(dataset)

    def run():
        return create_voronoi_cartogram(gdf, weights="Population")

    result = benchmark.pedantic(run, setup=_gc_collect, rounds=_ROUNDS, warmup_rounds=_WARMUP_ROUNDS, iterations=1)
    benchmark.extra_info.update(n_regions=len(gdf), status=str(getattr(result, "status", None)))
