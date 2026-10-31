"""
Benchmarks for proportional cartogram (shrink / split partition) performance.

Runtime against the number of regions (US states, congressional districts).
Each region is partitioned by a single fraction column.

Run with:
    uv run pytest benchmarks/test_proportional_cartogram.py --benchmark-enable -v
"""

import gc

import pytest

from carto_flow.proportional_cartogram import partition_geometries

_ROUNDS = 5
_WARMUP_ROUNDS = 1


def _gc_collect():
    gc.collect()


@pytest.mark.parametrize("method", ["shrink", "split"])
@pytest.mark.parametrize("dataset", ["us_states_gdf", "us_congressional_districts_gdf"])
def test_partition_runtime(benchmark, request, dataset, method):
    gdf = request.getfixturevalue(dataset).copy()
    # Fraction of each region's area assigned to the first part: normalized population.
    gdf["fraction"] = gdf["Population"] / gdf["Population"].max()

    def run():
        return partition_geometries(gdf, "fraction", method=method)

    result = benchmark.pedantic(run, setup=_gc_collect, rounds=_ROUNDS, warmup_rounds=_WARMUP_ROUNDS, iterations=1)
    benchmark.extra_info.update(n_regions=len(gdf), n_parts=len(result))
