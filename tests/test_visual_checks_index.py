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

    def test_exploration_is_open(self, mod, tmp_path):
        """An exploration is outstanding until someone says otherwise."""
        pr_dir = mod.scan_pr_dir(_write(tmp_path, "inv", title="t", description="d"))

        assert mod.resolve_status(pr_dir, {}) == "open"

    def test_pr_page_before_the_pr_exists_needs_review(self, mod, tmp_path):
        pr_dir = mod.scan_pr_dir(_write(tmp_path, "draft", kind="pr", title="t", description="d"))

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

        ranks = dict(re.findall(r'"([a-z -]+)": (\d+)', mod.SORT_SCRIPT))

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
        """Clicking Created or Closed once must show newest first, not oldest."""
        import re

        match = re.search(r"FIRST_ASC = \[([^\]]+)\]", mod.SORT_SCRIPT)
        assert match
        first_asc = [v.strip() == "true" for v in match.group(1).split(",")]

        assert len(first_asc) == 9
        assert first_asc[0] is True, "Status: outstanding first"
        assert first_asc[3] is False, "PR: highest first"
        assert first_asc[6] is False, "Created: newest first"
        assert first_asc[7] is False, "Closed: newest first"
        assert first_asc[8] is False, "Figures: most first"


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


class TestDefaultRoot:
    def test_environment_variable_wins(self, mod, monkeypatch, tmp_path):
        monkeypatch.setenv("VISUAL_CHECKS_ROOT", str(tmp_path))
        assert mod.default_root() == tmp_path

    def test_env_file_is_read(self, mod, monkeypatch, tmp_path):
        monkeypatch.delenv("VISUAL_CHECKS_ROOT", raising=False)
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        (tmp_path / ".env").write_text('OTHER=1\nVISUAL_CHECKS_ROOT="/data/checks"\n', encoding="utf-8")
        assert mod.default_root() == Path("/data/checks")

    def test_falls_back_to_sibling_checkout(self, mod, monkeypatch, tmp_path):
        monkeypatch.delenv("VISUAL_CHECKS_ROOT", raising=False)
        repo = tmp_path / "carto-flow"
        repo.mkdir()
        monkeypatch.setattr(mod, "REPO_ROOT", repo)
        assert mod.default_root() == tmp_path / "carto-flow-dev"

    def test_hidden_directories_are_not_pages(self, mod, monkeypatch, tmp_path):
        _write(tmp_path, "pr1-a", pr=1, title="a", description="d")
        (tmp_path / ".git").mkdir()
        monkeypatch.setattr(sys, "argv", ["prog", "--root", str(tmp_path), "--offline"])
        mod.main()
        assert not (tmp_path / ".git" / "index.html").exists()
        assert ".git" not in (tmp_path / "index.html").read_text(encoding="utf-8")


class TestIssueField:
    def test_issue_numbers_accept_lists_and_hashes(self, mod):
        assert mod._issue_numbers("74") == [74]
        assert mod._issue_numbers("#74, 75") == [74, 75]
        assert mod._issue_numbers("none") == []

    def test_page_links_the_issue(self, mod, tmp_path):
        d = _write(tmp_path, "a", pr=1, title="a", description="d", issue="74")
        page = mod.build_pr_page(mod.scan_pr_dir(d), "merged")
        assert "https://github.com/bright-fakl/carto-flow/issues/74" in page


class TestClosedAndUpdatedDates:
    def test_closed_time_prefers_merge_time(self, mod):
        rows = [
            {"number": 1, "state": "MERGED", "mergedAt": "2026-10-08T15:05:00Z", "closedAt": "2026-10-08T15:06:00Z"},
            {"number": 2, "state": "CLOSED", "mergedAt": None, "closedAt": "2026-10-09T08:00:00Z"},
            {"number": 3, "state": "OPEN", "mergedAt": None, "closedAt": None},
        ]
        closed = mod.pr_closed_from_rows(rows)
        assert set(closed) == {1, 2}
        assert len(closed[1]) == len("2026-10-08 15:05")
        assert closed[1] < closed[1][:11] + "99:99"  # parseable, local time
        assert mod.pr_states_from_rows(rows) == {1: "MERGED", 2: "CLOSED", 3: "OPEN"}

    def test_page_shows_created_updated_and_closed(self, mod, tmp_path):
        d = _write(tmp_path, "a", pr=1, title="a", description="d", date="2026-10-01 09:00", updated="2026-10-02 10:00")
        page = mod.build_pr_page(mod.scan_pr_dir(d), "merged", "2026-10-03 11:00")
        for label, value in (
            ("Created", "2026-10-01 09:00"),
            ("Updated", "2026-10-02 10:00"),
            ("Closed", "2026-10-03 11:00"),
        ):
            assert f"<dt>{label}</dt><dd>{value}</dd>" in page

    def test_index_has_closed_column(self, mod, tmp_path):
        d = _write(tmp_path, "a", pr=1, title="a", description="d", date="2026-10-01 09:00")
        pr = mod.scan_pr_dir(d)
        page = mod.build_index_page([pr], {pr.path: "merged"}, {1: "2026-10-03 11:00"})
        assert '<th data-col="7" title="Sort by Closed">Closed</th>' in page
        assert "<td>2026-10-03 11:00</td>" in page


class TestDisplayTimezone:
    def test_closed_time_follows_the_configured_zone(self, mod, monkeypatch):
        rows = [{"number": 1, "state": "MERGED", "mergedAt": "2026-10-08T19:26:00Z", "closedAt": None}]
        monkeypatch.setenv("VISUAL_CHECKS_TZ", "America/New_York")
        assert mod.pr_closed_from_rows(rows) == {1: "2026-10-08 15:26"}
        monkeypatch.setenv("VISUAL_CHECKS_TZ", "UTC")
        assert mod.pr_closed_from_rows(rows) == {1: "2026-10-08 19:26"}

    def test_unknown_zone_falls_back(self, mod, monkeypatch, capsys):
        monkeypatch.setenv("VISUAL_CHECKS_TZ", "Not/AZone")
        assert mod.display_timezone() is None
        assert "unknown time zone" in capsys.readouterr().err


class TestStableOrder:
    def test_pages_with_equal_dates_are_ordered_by_name(self, mod, monkeypatch, tmp_path):
        import re

        for name in ("zeta", "alpha", "mid"):
            _write(tmp_path, name, title=name, description="d", date="2026-06-01")
        monkeypatch.setattr(sys, "argv", ["prog", "--root", str(tmp_path), "--offline"])
        mod.main()
        page = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert re.findall(r'href="(\w+)/index.html"', page) == ["alpha", "mid", "zeta"]


class TestKindTopicAndStatus:
    def test_kind_defaults_follow_the_pr_number(self, mod, tmp_path):
        assert mod.scan_pr_dir(_write(tmp_path, "a", pr=1, title="t", description="d")).kind == "pr"
        assert mod.scan_pr_dir(_write(tmp_path, "b", title="t", description="d")).kind == "exploration"

    def test_unknown_kind_and_topic_warn(self, mod, tmp_path):
        page = mod.scan_pr_dir(_write(tmp_path, "a", kind="idea", topic="flow, graphs", title="t", description="d"))

        assert any("unknown kind" in w for w in page.warnings)
        assert any("unknown topic 'graphs'" in w for w in page.warnings)
        assert page.topics == ["flow", "graphs"]

    def test_exploration_with_a_pr_number_warns(self, mod, tmp_path):
        page = mod.scan_pr_dir(_write(tmp_path, "a", kind="proposal", pr=3, title="t", description="d"))

        assert any("--convert" in w for w in page.warnings)

    def test_status_must_fit_the_kind(self, mod, tmp_path):
        exploration = mod.scan_pr_dir(_write(tmp_path, "a", title="t", description="d", status="merged"))
        pr = mod.scan_pr_dir(_write(tmp_path, "b", pr=2, title="t", description="d", status="led-to"))
        mod.resolve_status(exploration, {})
        mod.resolve_status(pr, {})

        assert any("does not apply" in w for w in exploration.warnings)
        assert any("does not apply" in w for w in pr.warnings)

    def test_no_action_needs_an_outcome(self, mod, tmp_path):
        bare = mod.scan_pr_dir(_write(tmp_path, "a", title="t", description="d", status="no-action"))
        explained = mod.scan_pr_dir(
            _write(tmp_path, "b", title="t", description="d", status="no-action", outcome="negative result")
        )
        mod.resolve_status(bare, {})
        mod.resolve_status(explained, {})

        assert any("outcome" in w for w in bare.warnings)
        assert explained.warnings == []

    @pytest.mark.parametrize(
        ("status", "expected"),
        [("open", "todo"), ("on-hold", "parked"), ("led-to", "done"), ("superseded", "done"), ("no-action", "done")],
    )
    def test_exploration_status_classes(self, mod, status, expected):
        assert mod.status_class(status) == expected

    def test_vocabularies_cover_every_known_status(self, mod):
        assert mod.PR_STATUSES | mod.EXPLORATION_STATUSES == mod.KNOWN_STATUSES


class TestRelatedLinks:
    def test_reverse_link_is_derived(self, mod, tmp_path):
        a = mod.scan_pr_dir(_write(tmp_path, "study", title="study", description="d", status="led-to"))
        b = mod.scan_pr_dir(_write(tmp_path, "change", pr=5, title="change", description="d", related="study"))
        incoming = mod.link_graph([a, b])
        pages = {"study": a, "change": b}

        assert incoming == {"study": ["change"], "change": []}
        assert "Referenced by" in mod.build_pr_page(a, "led-to", None, pages, incoming)
        assert "Related" in mod.build_pr_page(b, "needs review", None, pages, incoming)

    def test_unknown_target_warns(self, mod, tmp_path):
        a = mod.scan_pr_dir(_write(tmp_path, "a", title="t", description="d", related="ghost"))
        mod.link_graph([a])

        assert any("'ghost' names no page" in w for w in a.warnings)

    def test_closed_page_without_a_successor_warns(self, mod, tmp_path):
        alone = mod.scan_pr_dir(_write(tmp_path, "alone", title="t", description="d", status="led-to"))
        mod.link_graph([alone])

        assert any("no 'related' link" in w for w in alone.warnings)


class TestConvert:
    def test_proposal_becomes_a_pr_page_in_place(self, mod, tmp_path):
        directory = _write(
            tmp_path, "idea", kind="proposal", title="t", description="d", status="open", date="2026-06-01"
        )

        mod.convert_to_pr(directory, 42)
        page = mod.scan_pr_dir(directory)

        assert directory.is_dir() and page.kind == "pr" and page.pr == 42
        assert page.url == "https://github.com/bright-fakl/carto-flow/pull/42"
        assert "status" not in page.meta and page.meta["date"] == "2026-06-01"
        assert page.warnings == []

    def test_converting_a_pr_page_is_refused(self, mod, tmp_path):
        directory = _write(tmp_path, "done", kind="pr", pr=1, title="t", description="d")

        with pytest.raises(SystemExit, match="already a PR page"):
            mod.convert_to_pr(directory, 2)


class TestIndexFilters:
    def test_rows_and_controls_carry_kind_topic_and_status(self, mod, tmp_path):
        d = _write(tmp_path, "a", kind="proposal", topic="flow, voronoi", title="a", description="d")
        page = mod.scan_pr_dir(d)

        html = mod.build_index_page([page], {d: "open"})

        assert 'data-kind="proposal"' in html and 'data-topics=" flow voronoi "' in html
        assert 'id="filter-kind"' in html and 'id="filter-superseded"' in html
        assert '<option value="voronoi">' in html

    def test_filters_survive_leaving_and_returning(self, mod):
        """Back link and back button both reload the page; the selection is stored and re-applied."""
        assert "sessionStorage.setItem" in mod.SORT_SCRIPT
        assert 'addEventListener("pageshow"' in mod.SORT_SCRIPT
