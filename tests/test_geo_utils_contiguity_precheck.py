"""Tests for the reachability pre-check inside ``repair_contiguity``.

``repair_contiguity`` guards its path enumeration with a node-level BFS that
returns early when no destination slot is reachable.  The pre-check duplicates
four things the enumeration also does -- the source filter, the
``forbidden_nodes`` handling, the depth bound, and the convention that
``max_len`` counts *nodes* rather than edges -- so a change to one and not the
other would make the repair silently stop repairing.  These tests pin that the
two agree.

Both are closures inside ``repair_contiguity`` and cannot be imported, so the
enumerator is lifted out of the function source (``_enumerators``) in two
forms: as written, and with the pre-check block deleted.  Any disagreement
between them is a bug in the pre-check.
"""

from __future__ import annotations

import heapq
import inspect
import textwrap
from collections import deque

import numpy as np
import pytest

from carto_flow.geo_utils.contiguity import repair_contiguity

# ``_enumerate_paths`` is called with the default depth bound everywhere in
# ``repair_contiguity``, so this is the bound that end-to-end repairs see.
DEFAULT_MAX_LEN = 10


# --------------------------------------------------------------------------
# lifting the closure out of the function source
# --------------------------------------------------------------------------


def _enumerator_source() -> list[str]:
    lines = inspect.getsource(repair_contiguity).splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("    def _enumerate_paths(")]
    if len(starts) != 1:
        pytest.fail("could not locate _enumerate_paths in repair_contiguity source")
    start = starts[0]
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("    def "))
    return textwrap.dedent("\n".join(lines[start:end])).splitlines()


def _without_precheck(source: list[str]) -> list[str]:
    """Delete the pre-check block, leaving the bare enumeration."""
    first = [i for i, line in enumerate(source) if line.strip().startswith("# Reachability pre-check")]
    last = [i for i, line in enumerate(source) if line.strip() == "if not reachable:"]
    if len(first) != 1 or len(last) != 1 or source[last[0] + 1].strip() != "return":
        pytest.fail(
            "the reachability pre-check in repair_contiguity no longer matches the "
            "markers this test strips it by; update the test or remove the pre-check"
        )
    return source[: first[0]] + source[last[0] + 2 :]


def _precheck_only(source: list[str]) -> list[str]:
    """Keep the signature and the pre-check, returning its verdict as a bool."""
    first = [i for i, line in enumerate(source) if line.strip().startswith("# Reachability pre-check")]
    last = [i for i, line in enumerate(source) if line.strip() == "if not reachable:"]
    if len(first) != 1 or len(last) != 1 or source[last[0] + 1].strip() != "return":
        pytest.fail(
            "the reachability pre-check in repair_contiguity no longer matches the "
            "markers this test lifts it by; update the test or remove the pre-check"
        )
    signature = [line.replace("def _enumerate_paths(", "def _precheck(") for line in source[: first[0]]]
    return signature + source[first[0] : last[0]] + ["    return reachable"]


def _enumerators(adj: list[set[int]], max_k: int = 20):
    """Return ``(with_precheck, without_precheck)`` enumerators over *adj*."""
    source = _enumerator_source()
    built = []
    for text in ("\n".join(source), "\n".join(_without_precheck(source))):
        built.append(_compile(text, "_enumerate_paths", adj, max_k))
    return built[0], built[1]


def _precheck(adj: list[set[int]], max_k: int = 20):
    """Return the pre-check alone, as a function returning its verdict."""
    return _compile("\n".join(_precheck_only(_enumerator_source())), "_precheck", adj, max_k)


def _compile(text: str, name: str, adj: list[set[int]], max_k: int):
    namespace: dict = {"heapq": heapq, "deque": deque, "adj": adj, "max_candidate_paths": max_k}
    exec(text, namespace)  # noqa: S102 - the source is our own module's
    return namespace[name]


class _CountingAdjacency(list):
    """Adjacency list that counts neighbour lookups."""

    def __init__(self, rows):
        super().__init__(rows)
        self.lookups = 0

    def __getitem__(self, index):
        self.lookups += 1
        return super().__getitem__(index)


# --------------------------------------------------------------------------
# small hand-built graphs
# --------------------------------------------------------------------------


def _chain(n: int) -> list[set[int]]:
    """Path graph ``0 - 1 - ... - n-1``."""
    adj: list[set[int]] = [set() for _ in range(n)]
    for i in range(n - 1):
        adj[i].add(i + 1)
        adj[i + 1].add(i)
    return adj


def _from_edges(n: int, edges) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    return adj


# (name, adj, src, dst, forbidden, max_len, expect_a_path)
AGREEMENT_CASES = [
    # exact boundary: 4 nodes, reached with max_len 4 and not with 3
    ("chain at the bound", _chain(4), {0}, {3}, set(), 4, True),
    ("chain one node short", _chain(4), {0}, {3}, set(), 3, False),
    # forbidden node is what makes the destination unreachable
    ("cut by forbidden", _chain(5), {0}, {4}, {2}, DEFAULT_MAX_LEN, False),
    # forbidden node forces the long way round, still inside max_len
    (
        "detour within bound",
        _from_edges(6, [(0, 1), (1, 3), (0, 4), (4, 5), (5, 3)]),
        {0},
        {3},
        {1},
        4,
        True,
    ),
    # same detour, one node too long
    (
        "detour beyond bound",
        _from_edges(6, [(0, 1), (1, 3), (0, 4), (4, 5), (5, 3)]),
        {0},
        {3},
        {1},
        3,
        False,
    ),
    # the only source is itself forbidden
    ("source forbidden", _chain(3), {0}, {2}, {0}, DEFAULT_MAX_LEN, False),
    # one of two sources is forbidden; the other still reaches
    ("one source left", _from_edges(4, [(0, 2), (1, 2), (2, 3)]), {0, 1}, {3}, {0}, 3, True),
    # destination is also the source
    ("source is destination", _chain(3), {0}, {0, 2}, set(), 1, True),
    # no edges at all
    ("isolated", _from_edges(3, []), {0}, {2}, set(), DEFAULT_MAX_LEN, False),
    # cycle: two routes of different length
    ("cycle", _from_edges(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]), {0}, {3}, set(), 4, True),
    # star centre forbidden disconnects the leaves
    ("star centre forbidden", _from_edges(4, [(0, 1), (1, 2), (1, 3)]), {0}, {2, 3}, {1}, DEFAULT_MAX_LEN, False),
]


@pytest.mark.parametrize(
    ("name", "adj", "src", "dst", "forbidden", "max_len", "expect_a_path"),
    AGREEMENT_CASES,
    ids=[case[0] for case in AGREEMENT_CASES],
)
def test_precheck_never_suppresses_a_path(name, adj, src, dst, forbidden, max_len, expect_a_path):
    """The guarded enumeration yields exactly what the bare one yields."""
    guarded, bare = _enumerators(adj)
    bare_paths = list(bare(src, dst, forbidden, max_len))
    guarded_paths = list(guarded(src, dst, forbidden, max_len))

    assert bool(bare_paths) is expect_a_path, f"{name}: hand-computed expectation is wrong"
    assert guarded_paths == bare_paths
    # The verdict itself, not just its effect: it must skip exactly when the
    # enumeration would have yielded nothing.
    assert _precheck(adj)(src, dst, forbidden, max_len) is expect_a_path


def test_boundary_is_counted_in_nodes_not_edges():
    """A 10-node route is inside the default bound; an 11-node route is not."""
    guarded, bare = _enumerators(_chain(11))

    assert list(guarded({0}, {9}, set(), DEFAULT_MAX_LEN)) == [list(range(10))]
    assert list(guarded({0}, {10}, set(), DEFAULT_MAX_LEN)) == []
    assert list(bare({0}, {10}, set(), DEFAULT_MAX_LEN)) == []


def test_precheck_short_circuits_the_unreachable_case():
    """When it skips, the pre-check costs one lookup per reachable node."""
    # A 4x4 grid plus one destination beyond the default depth bound: the bare
    # enumeration must walk every simple path before the heap empties.
    edges = []
    for r in range(4):
        for c in range(4):
            node = r * 4 + c
            if c < 3:
                edges.append((node, node + 1))
            if r < 3:
                edges.append((node, node + 4))
    adj = _CountingAdjacency(_from_edges(17, edges))
    guarded, bare = _enumerators(adj)

    adj.lookups = 0
    assert list(guarded({0}, {16}, set(), DEFAULT_MAX_LEN)) == []
    guarded_lookups = adj.lookups

    adj.lookups = 0
    assert list(bare({0}, {16}, set(), DEFAULT_MAX_LEN)) == []
    bare_lookups = adj.lookups

    assert guarded_lookups <= len(adj)
    assert bare_lookups > 10 * guarded_lookups


def test_precheck_agrees_on_random_small_graphs():
    """Fuzz the two against each other on small random graphs."""
    rng = np.random.default_rng(20260923)

    def _pick(count: int, k: int) -> set[int]:
        return {int(v) for v in rng.choice(count, size=k, replace=False)}

    for _ in range(300):
        n = int(rng.integers(2, 10))
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.3]
        adj = _from_edges(n, edges)
        src = _pick(n, int(rng.integers(1, min(2, n) + 1)))
        dst = _pick(n, int(rng.integers(1, min(3, n) + 1)))
        forbidden = _pick(n, int(rng.integers(0, min(3, n) + 1)))
        max_len = int(rng.integers(1, 7))
        guarded, bare = _enumerators(adj)
        bare_paths = list(bare(src, dst, forbidden, max_len))
        assert list(guarded(src, dst, forbidden, max_len)) == bare_paths
        assert _precheck(adj)(src, dst, forbidden, max_len) is bool(bare_paths)


# --------------------------------------------------------------------------
# end to end through repair_contiguity
# --------------------------------------------------------------------------


def _chain_with_satellite(n_slots: int):
    """Chain of slots; group ``"A"`` holds both ends, every other slot is a singleton."""
    adj = _chain(n_slots)
    groups = [f"s{i}" for i in range(n_slots)]
    groups[0] = "A"
    groups[n_slots - 1] = "A"
    return adj, groups


def test_repair_moves_a_reachable_satellite():
    """A satellite within the depth bound is walked back to the main body."""
    adj, groups = _chain_with_satellite(5)
    slot_of, discontiguous = repair_contiguity(None, groups, adjacency=adj)

    assert discontiguous == []
    # District 4 (the satellite) ends up next to district 0 (the main body).
    assert slot_of[0] == 0
    assert slot_of[4] == 1
    assert sorted(slot_of.tolist()) == list(range(5))
    assert {slot_of[0], slot_of[4]} == {0, 1}


def test_repair_at_the_depth_bound_and_one_past_it():
    """10 nodes from satellite to main body repairs; 11 nodes does not."""
    adj, groups = _chain_with_satellite(DEFAULT_MAX_LEN)
    slot_of, discontiguous = repair_contiguity(None, groups, adjacency=adj)
    assert discontiguous == []
    assert {slot_of[0], slot_of[DEFAULT_MAX_LEN - 1]} == {0, 1}

    adj, groups = _chain_with_satellite(DEFAULT_MAX_LEN + 1)
    slot_of, discontiguous = repair_contiguity(None, groups, adjacency=adj)
    assert discontiguous == [("A", [DEFAULT_MAX_LEN])]
    assert np.array_equal(slot_of, np.arange(DEFAULT_MAX_LEN + 1))


def test_repair_with_three_components_still_converges():
    """Each satellite is routed home while the other one is forbidden."""
    # 0 - 1 - 2 - 3 - 4 with group A at slots 0, 2 and 4: while one satellite
    # searches for a route, the other satellite is in ``forbidden_nodes``.
    adj = _chain(5)
    groups = ["A", "s1", "A", "s3", "A"]
    slot_of, discontiguous = repair_contiguity(None, groups, adjacency=adj)

    assert discontiguous == []
    assert {slot_of[0], slot_of[2], slot_of[4]} == {0, 1, 2}
    assert sorted(slot_of.tolist()) == list(range(5))
