"""Tests for scripts/build_visual_checks_index.py.

The script has no import path of its own, so it is loaded from source. These
pin the two behaviors that have silently regressed before: the order pages are
listed in, and which of them count as still needing review.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_visual_checks_index.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("build_visual_checks_index", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(root: Path, name: str, **meta) -> Path:
    directory = root / name
    directory.mkdir()
    lines = "\n".join(f"{k}: {v}" for k, v in meta.items())
    (directory / "summary.md").write_text(f"---\n{lines}\n---\n\nbody\n", encoding="utf-8")
    return directory


class TestOrdering:
    def test_newest_first_regardless_of_pr_number(self, mod, tmp_path):
        """A dated page with no PR sorts among the PRs of the same date, not last."""
        _write(tmp_path, "old-pr", pr=99, title="old", description="d", date="2026-01-01")
        _write(tmp_path, "investigation", title="inv", description="d", date="2026-06-01")
        _write(tmp_path, "new-pr", pr=1, title="new", description="d", date="2026-06-01")

        dirs = [mod.scan_pr_dir(p) for p in tmp_path.iterdir() if p.is_dir()]
        dirs.sort(key=mod._sort_key)

        assert [d.title for d in dirs] == ["new", "inv", "old"]

    def test_status_edit_does_not_reorder(self, mod, tmp_path):
        """--set-status rewrites summary.md; that must not move the page."""
        a = _write(tmp_path, "a", pr=10, title="ten", description="d", date="2026-06-01")
        b = _write(tmp_path, "b", pr=20, title="twenty", description="d", date="2026-06-01")
        before = sorted((mod.scan_pr_dir(p) for p in (a, b)), key=mod._sort_key)

        mod.set_status(a, "reviewed")
        after = sorted((mod.scan_pr_dir(p) for p in (a, b)), key=mod._sort_key)

        assert [d.title for d in before] == [d.title for d in after]

    def test_pr_number_only_breaks_ties_within_a_date(self, mod, tmp_path):
        import os

        a = _write(tmp_path, "a", pr=10, title="ten", description="d", date="2026-06-01")
        b = _write(tmp_path, "b", pr=20, title="twenty", description="d", date="2026-06-01")
        for d in (a, b):  # identical mtimes, so only the PR number separates them
            os.utime(d / "summary.md", (1_000_000, 1_000_000))

        dirs = [mod.scan_pr_dir(p) for p in tmp_path.iterdir() if p.is_dir()]
        dirs.sort(key=mod._sort_key)

        assert [d.title for d in dirs] == ["twenty", "ten"]

    def test_mtime_ignores_the_generated_index(self, mod, tmp_path):
        """The builder writes index.html into every directory; counting it would
        reset each mtime on every rebuild and destroy the fallback ordering."""
        directory = _write(tmp_path, "undated", title="u", description="d")
        before = mod._dir_mtime(directory)
        index = directory / "index.html"
        index.write_text("<html></html>", encoding="utf-8")
        import os

        os.utime(index, (before + 10_000, before + 10_000))

        assert mod._dir_mtime(directory) == before


class TestStatus:
    def test_declared_status_wins_over_pr_state(self, mod, tmp_path):
        directory = _write(tmp_path, "d", pr=7, title="t", description="d", status="needs review")
        pr_dir = mod.scan_pr_dir(directory)

        assert mod.resolve_status(pr_dir, {7: "MERGED"}) == "needs review"

    @pytest.mark.parametrize(
        ("state", "expected"),
        [("MERGED", "merged"), ("CLOSED", "closed"), ("OPEN", "needs review")],
    )
    def test_falls_back_to_pr_state(self, mod, tmp_path, state, expected):
        directory = _write(tmp_path, f"d-{state}", pr=7, title="t", description="d")
        pr_dir = mod.scan_pr_dir(directory)

        assert mod.resolve_status(pr_dir, {7: state}) == expected

    def test_directory_without_pr_needs_review(self, mod, tmp_path):
        """An investigation is outstanding until someone says otherwise."""
        pr_dir = mod.scan_pr_dir(_write(tmp_path, "inv", title="t", description="d"))

        assert mod.resolve_status(pr_dir, {}) == "needs review"

    def test_unknown_when_gh_gave_nothing(self, mod, tmp_path):
        pr_dir = mod.scan_pr_dir(_write(tmp_path, "d", pr=7, title="t", description="d"))

        assert mod.resolve_status(pr_dir, {}) == "unknown"

    def test_merged_counts_as_done(self, mod):
        assert "merged" in mod.DONE_STATUSES
        assert "needs review" not in mod.DONE_STATUSES


class TestMetadata:
    def test_pr_and_url_are_optional(self, mod, tmp_path):
        """Investigation workspaces belong to no PR; requiring them produced a
        warning per directory on every build."""
        pr_dir = mod.scan_pr_dir(_write(tmp_path, "inv", title="t", description="d"))

        assert pr_dir.warnings == []
        assert pr_dir.pr is None

    def test_missing_title_warns(self, mod, tmp_path):
        pr_dir = mod.scan_pr_dir(_write(tmp_path, "inv", description="d"))

        assert any("title" in w for w in pr_dir.warnings)


class TestIndexPage:
    def test_unreviewed_rows_are_marked(self, mod, tmp_path):
        done = _write(tmp_path, "done", pr=1, title="done", description="d", date="2026-06-01")
        todo = _write(tmp_path, "todo", title="todo", description="d", date="2026-06-02")
        dirs = [mod.scan_pr_dir(p) for p in (todo, done)]
        statuses = {todo: "needs review", done: "merged"}

        page = mod.build_index_page(dirs, statuses)

        assert "1 of 2 awaiting review" in page
        assert page.count('<tr class="todo"') == 1
        assert page.count("<tr ") == 2


class TestVocabulary:
    def test_the_three_states_are_disjoint(self, mod):
        assert not (mod.OUTSTANDING_STATUSES & mod.PARKED_STATUSES)
        assert not (mod.PARKED_STATUSES & mod.DONE_STATUSES)
        assert not (mod.OUTSTANDING_STATUSES & mod.DONE_STATUSES)

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            ("needs review", "todo"),
            ("deferred", "parked"),
            ("reviewed", "done"),
            ("merged", "done"),
            ("closed", "done"),
            ("superseded", "done"),
            ("nonsense", "todo"),
        ],
    )
    def test_status_class(self, mod, status, expected):
        assert mod.status_class(status) == expected

    def test_unknown_declared_status_warns(self, mod, tmp_path):
        """A typo must not silently read as outstanding forever."""
        pr_dir = mod.scan_pr_dir(_write(tmp_path, "d", title="t", description="d", status="reviewd"))
        mod.resolve_status(pr_dir, {})

        assert any("unknown status" in w for w in pr_dir.warnings)


class TestSetStatus:
    def test_adds_then_replaces(self, mod, tmp_path):
        directory = _write(tmp_path, "d", title="t", description="d")

        mod.set_status(directory, "deferred")
        assert "status: deferred" in (directory / "summary.md").read_text()

        mod.set_status(directory, "reviewed")
        text = (directory / "summary.md").read_text()
        assert "status: reviewed" in text
        assert "deferred" not in text

    def test_rejects_an_unknown_status(self, mod, tmp_path):
        directory = _write(tmp_path, "d", title="t", description="d")

        with pytest.raises(SystemExit, match="Unknown status"):
            mod.set_status(directory, "reviewd")

    def test_leaves_the_body_alone(self, mod, tmp_path):
        directory = _write(tmp_path, "d", title="t", description="d")

        mod.set_status(directory, "reviewed")

        assert (directory / "summary.md").read_text().endswith("body\n")


class TestStatusSortOrder:
    def test_status_rank_matches_the_python_vocabulary(self, mod):
        """The client-side rank table and the Python vocabulary must not drift."""
        import re

        ranks = dict(re.findall(r'"([a-z ]+)": (\d+)', mod.SORT_SCRIPT))

        assert set(ranks) == mod.KNOWN_STATUSES
        assert ranks["needs review"] == "0", "outstanding items must sort first"
        assert int(ranks["deferred"]) < min(int(ranks[s]) for s in mod.DONE_STATUSES)

    def test_list_status_puts_outstanding_first(self, mod, tmp_path, capsys):
        done = _write(tmp_path, "a-done", pr=1, title="done", description="d", date="2026-06-03")
        parked = _write(tmp_path, "b-parked", title="parked", description="d", date="2026-06-02")
        todo = _write(tmp_path, "c-todo", title="todo", description="d", date="2026-06-01")
        dirs = [mod.scan_pr_dir(p) for p in (done, parked, todo)]
        statuses = {done: "merged", parked: "deferred", todo: "needs review"}

        mod.list_status(dirs, statuses)

        printed = [ln.split()[-1] for ln in capsys.readouterr().out.strip().split("\n")]
        assert printed == ["c-todo", "b-parked", "a-done"]


class TestClickSortDirection:
    def test_first_click_is_useful_per_column(self, mod):
        """Clicking Date once must show newest first, not oldest."""
        import re

        match = re.search(r"FIRST_ASC = \[([^\]]+)\]", mod.SORT_SCRIPT)
        assert match
        first_asc = [v.strip() == "true" for v in match.group(1).split(",")]

        assert len(first_asc) == 6
        assert first_asc[0] is True, "Status: outstanding first"
        assert first_asc[1] is False, "PR: highest first"
        assert first_asc[4] is False, "Date: newest first"
        assert first_asc[5] is False, "Figures: most first"


class TestTieBreaking:
    def test_date_accepts_an_optional_time(self, mod, tmp_path):
        """A page written later the same day can say so and lead its day."""
        morning = _write(tmp_path, "am", title="am", description="d", date="2026-06-01")
        evening = _write(tmp_path, "pm", title="pm", description="d", date="2026-06-01 18:00")

        dirs = sorted((mod.scan_pr_dir(p) for p in (morning, evening)), key=mod._sort_key)

        assert [d.title for d in dirs] == ["pm", "am"]

    def test_rows_carry_their_generated_order(self, mod, tmp_path):
        """The client-side sort breaks ties on this, so a column where many rows
        share a value reproduces the default order instead of drifting."""
        a = _write(tmp_path, "a", pr=2, title="a", description="d", date="2026-06-01")
        b = _write(tmp_path, "b", pr=1, title="b", description="d", date="2026-06-01")
        dirs = sorted((mod.scan_pr_dir(p) for p in (a, b)), key=mod._sort_key)

        page = mod.build_index_page(dirs, {a: "merged", b: "merged"})

        assert 'data-ord="0"' in page
        assert 'data-ord="1"' in page
        assert "a.dataset.ord" in mod.SORT_SCRIPT
