"""Tests for the path enumeration and the satellite search inside ``repair_contiguity``.

``_enumerate_paths`` is a closure inside ``repair_contiguity`` and cannot be
imported, so it is lifted out of the function source.  Its distance-pruned
search must yield exactly the chains, in the same order, as a plain search
over all simple paths of up to ``max_len`` nodes.
"""

from __future__ import annotations

import heapq
import inspect
import random
import textwrap
from collections import deque

import pytest

from carto_flow.geo_utils.contiguity import repair_contiguity

DEFAULT_MAX_LEN = 10


def _pruned(adj, max_k=20):
    lines = inspect.getsource(repair_contiguity).splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("    def _enumerate_paths(")]
    if len(starts) != 1:
        pytest.fail("could not locate _enumerate_paths in repair_contiguity source")
    end = next(i for i in range(starts[0] + 1, len(lines)) if lines[i].startswith("    def "))
    text = textwrap.dedent("\n".join(lines[starts[0] : end]))
    namespace: dict = {"heapq": heapq, "deque": deque, "adj": adj, "max_candidate_paths": max_k}
    exec(text, namespace)  # noqa: S102 - the source is our own module's
    return namespace["_enumerate_paths"]


def _plain(adj, src, dst, forbidden, max_len, max_k=20):
    """Reference: shortest-first search over all simple paths, no pruning."""
    heap = [(1, (s,)) for s in src if s not in forbidden]
    heapq.heapify(heap)
    found = []
    while heap and len(found) < max_k:
        length, path = heapq.heappop(heap)
        if path[-1] in dst:
            found.append(list(path))
            continue
        if length >= max_len:
            continue
        for nb in adj[path[-1]]:
            if nb not in path and nb not in forbidden:
                heapq.heappush(heap, (length + 1, (*path, nb)))
    return found


def _from_edges(n, edges):
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _chain(n):
    return _from_edges(n, [(i, i + 1) for i in range(n - 1)])


CASES = [
    ("chain at the bound", _chain(4), {0}, {3}, set(), 4),
    ("chain one node short", _chain(4), {0}, {3}, set(), 3),
    ("cut by forbidden", _chain(5), {0}, {4}, {2}, DEFAULT_MAX_LEN),
    ("detour within bound", _from_edges(6, [(0, 1), (1, 3), (0, 4), (4, 5), (5, 3)]), {0}, {3}, {1}, 4),
    ("detour beyond bound", _from_edges(6, [(0, 1), (1, 3), (0, 4), (4, 5), (5, 3)]), {0}, {3}, {1}, 3),
    ("source forbidden", _chain(3), {0}, {2}, {0}, DEFAULT_MAX_LEN),
    ("two sources, one forbidden", _from_edges(4, [(0, 2), (1, 2), (2, 3)]), {0, 1}, {3}, {0}, 3),
    ("isolated", _from_edges(3, []), {0}, {2}, set(), DEFAULT_MAX_LEN),
    ("cycle", _from_edges(6, [(i, (i + 1) % 6) for i in range(6)]), {0}, {3}, set(), 4),
    ("eleven nodes", _chain(11), {0}, {10}, set(), DEFAULT_MAX_LEN),
    ("ten nodes", _chain(11), {0}, {9}, set(), DEFAULT_MAX_LEN),
]


@pytest.mark.parametrize(("name", "adj", "src", "dst", "forbidden", "max_len"), CASES, ids=[c[0] for c in CASES])
def test_pruned_enumeration_matches_plain(name, adj, src, dst, forbidden, max_len):
    assert list(_pruned(adj)(src, dst, forbidden, max_len)) == _plain(adj, src, dst, forbidden, max_len)


def test_pruned_enumeration_matches_plain_on_random_graphs():
    rng = random.Random(0)  # noqa: S311 - seeded test data
    for _ in range(60):
        n = rng.randint(6, 16)
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.25]
        adj = _from_edges(n, edges)
        nodes = list(range(n))
        rng.shuffle(nodes)
        src, dst, forbidden = set(nodes[:2]), set(nodes[2:4]), set(nodes[4:6])
        max_len = rng.randint(2, 8)
        for max_k in (3, 20):
            assert list(_pruned(adj, max_k)(src, dst, forbidden, max_len)) == _plain(
                adj, src, dst, forbidden, max_len, max_k
            )


def test_pruned_enumeration_is_linear_when_unreachable():
    """A destination beyond the depth bound costs neighbor lookups, not path expansions."""

    class Counting(list):
        lookups = 0

        def __getitem__(self, index):
            self.lookups += 1
            return super().__getitem__(index)

    size = 6
    edges = [(r * size + c, r * size + c + 1) for r in range(size) for c in range(size - 1)]
    edges += [(r * size + c, (r + 1) * size + c) for r in range(size - 1) for c in range(size)]
    base = _from_edges(size * size, edges)
    adj = Counting(base)
    assert list(_pruned(adj)({0}, {size * size - 1}, set(), 8)) == []
    assert adj.lookups <= 2 * size * size


def test_satellite_other_than_second_largest_is_connected():
    """The satellite being repaired neither forbids its own tiles nor is the largest one spared.

    Satellites are processed nearest first, so the one being repaired is often
    not the second largest component.  Group A has the main body {0, 1, 2}, a
    larger satellite {6, 7} and a nearer, smaller satellite {4}; the chain for
    {4} must find slot 3 even though {6, 7} is kept free of it.
    """
    groups = list("AAABABAAB")
    adj = _from_edges(len(groups), [(i, i + 1) for i in range(len(groups) - 1)])
    slot_of, _ = repair_contiguity(None, groups, adjacency=adj, max_passes=1)
    slots_a = {int(slot_of[i]) for i, g in enumerate(groups) if g == "A"}
    assert {0, 1, 2, 3, 4} <= slots_a


def test_us_states_squares_do_not_leave_a_split_region():
    """Two split regions no longer take turns at repairing each other.

    On the 49 states with 350 tiles and squares, repairing a satellite of one
    region used to split a neighboring region that was already split, and the
    next pass undid it, until the passes ran out with a region in two blocks.
    """
    import numpy as np

    from carto_flow.data import load_us_census
    from carto_flow.symbol_cartogram import MosaicLayout, create_layout

    gdf = load_us_census(level="state", population=True, contiguous_only=True).reset_index(drop=True)
    population = gdf["Population"].to_numpy(dtype=float)
    gdf["tiles"] = np.maximum(1, np.round(population / population.sum() * 350)).astype(int)
    result = create_layout(
        gdf, tile_count="tiles", layout=MosaicLayout(tiling="square", morph=False), show_progress=False
    )
    assert result.metrics.algorithm.n_noncontiguous_regions == 0
