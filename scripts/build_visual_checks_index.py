#!/usr/bin/env python3
"""Regenerate the visual_checks/ table-of-contents and per-PR pages.

Each subdirectory of the root holds a set of PNGs and a ``summary.md``
written by hand for a PR's before/after visual review. ``summary.md`` must
start with a fenced metadata block, parsed as simple ``key: value`` lines
(no YAML library, no nesting):

    ---
    pr: 27
    title: Fix coverage_simplify tolerance units in raster Voronoi cells
    description: One-line what changed and what to look for in the figures.
    url: https://github.com/bright-fakl/carto-flow/pull/27
    branch: fix/voronoi-smoothing-tolerance
    base: fix/voronoi-cell-extraction
    date: 2026-09-18
    before: origin/fix/voronoi-cell-extraction
    after: fix/voronoi-smoothing-tolerance @ <sha>
    inputs: districts (bundled, simplify 5000 m, min_island 50000), states (bundled)
    ---

``title`` and ``description`` are required - a missing one prints a warning
naming the directory and falls back to the directory name. ``pr``, ``url``,
``status``, ``branch``, ``base``, ``date``, ``before``, ``after`` and
``inputs`` are optional and, when present, are rendered in the page's
definition list.  ``pr`` and ``url`` are optional because several directories
are investigation workspaces that belong to no single PR.

Review status comes from the PR's GitHub state via one ``gh`` call - merged or
closed is done, open needs review - and an explicit ``status:`` always wins, so
a merged PR can still be flagged for follow-up.  Directories with no PR and no
``status:`` need review: an investigation is outstanding until someone says it
is not.  The recognised values are ``needs review`` (outstanding), ``deferred``
(a deliberate park), and ``reviewed`` / ``merged`` / ``closed`` / ``superseded``
(done); anything else warns.  The generated pages are static, so there is no
control to click - use ``--set-status DIR STATUS`` to change one and
``--list-status`` to print them all.

Below the metadata block, ``summary.md`` may contain any number of caption
lines anywhere in the file:

    figure: <filename> — <caption>

Each PNG with a matching ``figure:`` line uses that caption; otherwise the
filename itself is used as the caption.

This script builds:

- ``<root>/index.html`` - a table of contents: a table of PR (linked to the
  GitHub PR), Title (linked to the page), Description, Date, Figures -
  sorted newest first, by the ``date`` metadata field where present and by
  directory mtime otherwise.  PR number only breaks ties.
- ``<root>/<subdir>/index.html`` - a standalone page per subdirectory:
  ``<h1>PR #N: title</h1>``, a definition list (URL, branch, base, date,
  before/after, inputs), the description, the rest of ``summary.md``
  rendered as HTML, and every PNG in the directory as an ``<img>`` with its
  caption (relative link, not embedded). A "back to index" link appears at
  top and bottom.
- ``<root>/README.md`` - a short note on how to add a check, the header
  format, and the command to rebuild.

Only stdlib is used - no markdown library, no third-party dependencies.

Usage:
    uv run python scripts/build_visual_checks_index.py
    uv run python scripts/build_visual_checks_index.py --root visual_checks
    uv run python scripts/build_visual_checks_index.py --root /path/to/visual_checks
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

STYLE = """
  :root { color-scheme: light dark; }
  body {
    margin: 0;
    padding: 24px;
    padding-top: calc(24px + env(safe-area-inset-top, 0px));
    padding-bottom: calc(24px + env(safe-area-inset-bottom, 0px));
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: #fafafa;
    color: #1a1a1a;
    line-height: 1.5;
  }
  h1 { font-size: 1.6rem; }
  h2 { font-size: 1.3rem; margin-top: 2rem; border-bottom: 2px solid #ddd; padding-bottom: 0.3rem; }
  h3, h4 { font-size: 1.05rem; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.85rem; }
  th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }
  th[data-col] { cursor: pointer; user-select: none; }
  th[data-col]:hover { background: #eef; }
  tr.todo { background: #fffbe6; }
  tr.todo td:first-child { border-left: 3px solid #d98c00; }
  tr.parked { background: #f2f2f5; }
  tr.parked td:first-child { border-left: 3px solid #8a8a99; }
  .status { font-size: 0.8rem; white-space: nowrap; }
  .status.todo { color: #8a5a00; font-weight: 600; }
  .status.parked { color: #5a5a6a; font-style: italic; }
  .status.done { color: #4a7a4a; }
  th { background: rgba(0,0,0,0.05); }
  code { background: rgba(0,0,0,0.06); padding: 0.1em 0.35em; border-radius: 4px; font-size: 0.9em; }
  ul { padding-left: 1.4rem; }
  dl.meta-list { margin: 1rem 0; }
  dl.meta-list dt { font-weight: 600; float: left; clear: left; width: 8rem; color: #555; }
  dl.meta-list dd { margin-left: 8rem; margin-bottom: 0.35rem; }
  .figure { margin: 1.5rem 0; padding: 12px; background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; }
  .caption { font-size: 0.85rem; color: #555; margin: 0.5rem 0 0; }
  img { display: block; margin: 0 auto; max-width: 100%; }
  .toc-item { margin: 1rem 0; padding: 12px; background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; }
  .toc-item h2 { margin-top: 0; border-bottom: none; padding-bottom: 0; }
  .meta { font-size: 0.8rem; color: #777; }
  .desc { color: #333; }
  .back { display: inline-block; margin: 1rem 0; font-size: 0.9rem; }
  a { color: #0645ad; }
  table { overflow-x: auto; display: block; }
"""

DARK_STYLE = """
  @media (prefers-color-scheme: dark) {
    body { background: #1a1a1a; color: #eee; }
    table, th, td { border-color: #444 !important; }
    th[data-col]:hover { background: #26304a !important; }
    tr.todo { background: #2a2618 !important; }
    tr.parked { background: #222 !important; }
    .status.parked { color: #9a9aaa !important; }
    .status.todo { color: #e0b050 !important; }
    .status.done { color: #8fbf8f !important; }
    th { background: rgba(255,255,255,0.08); }
    code { background: rgba(255,255,255,0.1); }
    .figure, .toc-item { background: #242424; border-color: #333; }
    .meta { color: #999; }
    .desc { color: #ddd; }
    dl.meta-list dt { color: #aaa; }
    a { color: #7fb3ff; }
  }
"""

README_TEXT = """# Visual checks

Before/after visual review pages for PRs, built by
`scripts/build_visual_checks_index.py`. This directory is gitignored - it is
a local review workspace, not part of the repo.

## Adding a check

1. Create a subdirectory (any name, e.g. `pr27-voronoi-smoothing-tolerance`).
2. Drop PNGs and a `summary.md` in it. `summary.md` must start with a fenced
   metadata header:

   ```
   ---
   pr: 27
   title: Fix coverage_simplify tolerance units in raster Voronoi cells
   description: One-line what changed and what to look for in the figures.
   url: https://github.com/bright-fakl/carto-flow/pull/27
   branch: fix/voronoi-smoothing-tolerance
   base: fix/voronoi-cell-extraction
   date: 2026-09-18
   before: origin/fix/voronoi-cell-extraction
   after: fix/voronoi-smoothing-tolerance @ <sha>
   inputs: districts (bundled, simplify 5000 m, min_island 50000), states (bundled)
   ---
   ```

   `title` and `description` are required; `pr`, `url`, `status`, `branch`,
   `base`, `date`, `before`, `after`, and `inputs` are optional and rendered
   in a definition list on the page.  Pages are listed newest first, by `date`
   where given and by directory mtime otherwise.

   Review status is taken from the PR's GitHub state unless `status:` says
   otherwise. Values: `needs review`, `deferred`, `reviewed`, `merged`,
   `closed`, `superseded`. Change one with
   `--set-status <dir> <status>`; list them all with `--list-status`.

3. Optionally caption a figure by adding a line anywhere below the header:

   ```
   figure: <filename> — <caption>
   ```

   One line per PNG that needs a caption; PNGs without one use their
   filename.

4. Write the rest of `summary.md` freely (headings, tables, bullet lists,
   paragraphs, inline code) - it is rendered below the description.

## Rebuilding

```
uv run python scripts/build_visual_checks_index.py --root /path/to/visual_checks
```

Regenerates `index.html` and every `<subdir>/index.html`, plus this file.
"""


SORT_SCRIPT = """
<script>
(function () {
  var table = document.getElementById("toc");
  if (!table) return;
  var body = table.tBodies[0];
  var state = {};
  // First click sorts the way that column is actually useful: newest dates,
  // highest PR, most figures, but outstanding statuses first.
  var FIRST_ASC = [true, false, true, true, false, false];
  // Status sorts by how much attention an item still needs, not alphabetically:
  // "needs review" before "deferred" before anything finished.
  var STATUS_RANK = {
    "needs review": 0,
    "deferred": 1,
    "reviewed": 2,
    "merged": 3,
    "closed": 4,
    "superseded": 5
  };
  function cellValue(row, i) {
    var text = (row.cells[i].innerText || "").trim();
    if (i === 0) {                                                  // Status
      var rank = STATUS_RANK[text.toLowerCase()];
      return rank === undefined ? -1 : rank;                        // unknown first
    }
    if (i === 1) return parseInt(text.replace("#", ""), 10) || -1;  // PR
    if (i === 5) return parseInt(text, 10) || 0;                    // Figures
    return text.toLowerCase();
  }
  table.querySelectorAll("th[data-col]").forEach(function (th) {
    th.addEventListener("click", function () {
      var i = +th.dataset.col;
      var asc = state[i] = (i in state) ? !state[i] : FIRST_ASC[i];
      var rows = Array.prototype.slice.call(body.rows);
      rows.sort(function (a, b) {
        var x = cellValue(a, i), y = cellValue(b, i);
        if (x < y) return asc ? -1 : 1;
        if (x > y) return asc ? 1 : -1;
        return 0;
      });
      rows.forEach(function (r) { body.appendChild(r); });
    });
  });
})();
</script>
"""


def _page_shell(title: str, body: str) -> str:
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{html.escape(title)}</title>
<style>
{STYLE}
{DARK_STYLE}
</style>
</head>
<body>
{body}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Tiny Markdown -> HTML converter (headings, paragraphs, pipe tables,
# bullet lists, inline code). Deliberately minimal - not a general parser.
# ---------------------------------------------------------------------------


def _render_inline(text: str) -> str:
    text = html.escape(text)
    # Inline code: `...`
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


def _render_table(lines: list[str]) -> str:
    rows = [line.strip().strip("|").split("|") for line in lines]
    rows = [[cell.strip() for cell in row] for row in rows]
    header, *rest = rows
    # Second row is the separator (e.g. "---|---"); drop it if present.
    if rest and all(re.fullmatch(r":?-{2,}:?", cell) for cell in rest[0]):
        rest = rest[1:]
    out = ["<table>", "<tr>" + "".join(f"<th>{_render_inline(c)}</th>" for c in header) + "</tr>"]
    for row in rest:
        out.append("<tr>" + "".join(f"<td>{_render_inline(c)}</td>" for c in row) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def render_markdown(text: str) -> str:
    lines = text.splitlines()
    html_parts: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Skip `figure: ...` caption directives - they are metadata, not prose.
        if re.match(r"^figure:\s*\S", stripped):
            i += 1
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            html_parts.append(f"<h{level}>{_render_inline(heading_match.group(2))}</h{level}>")
            i += 1
            continue

        if stripped.startswith("|"):
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            html_parts.append(_render_table(table_lines))
            continue

        if re.match(r"^[-*]\s+", stripped):
            items = []
            while i < n and re.match(r"^[-*]\s+", lines[i].strip()):
                items.append(re.sub(r"^[-*]\s+", "", lines[i].strip()))
                i += 1
            html_parts.append("<ul>" + "".join(f"<li>{_render_inline(item)}</li>" for item in items) + "</ul>")
            continue

        # Paragraph: gather consecutive non-blank, non-special lines.
        para_lines = [stripped]
        i += 1
        while i < n and lines[i].strip() and not re.match(r"^(#{1,6}\s|\||[-*]\s|figure:\s*\S)", lines[i].strip()):
            para_lines.append(lines[i].strip())
            i += 1
        html_parts.append(f"<p>{_render_inline(' '.join(para_lines))}</p>")

    return "\n".join(html_parts)


# ---------------------------------------------------------------------------
# summary.md metadata header parsing
# ---------------------------------------------------------------------------

REQUIRED_META_KEYS = ("title", "description")
OPTIONAL_META_KEYS = ("pr", "url", "status", "branch", "base", "date", "before", "after", "inputs")
META_LABELS = {
    "url": "URL",
    "branch": "Branch",
    "base": "Base",
    "date": "Date",
    "status": "Status",
    "before": "Before",
    "after": "After",
    "inputs": "Inputs",
}


def _split_metadata_block(summary_text: str) -> tuple[dict[str, str], str]:
    """Parse a leading ``---`` fenced ``key: value`` block, if present.

    Returns (metadata dict, remaining text after the block). If there is no
    metadata block, returns ({}, summary_text) unchanged.
    """
    lines = summary_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, summary_text

    meta: dict[str, str] = {}
    end_index = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = i
            break
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", line)
        if match:
            meta[match.group(1).strip().lower()] = match.group(2).strip()

    if end_index is None:
        # Unterminated block - treat as no metadata rather than swallow the file.
        return {}, summary_text

    remainder = "\n".join(lines[end_index + 1 :])
    return meta, remainder


def _parse_figure_captions(text: str) -> dict[str, str]:
    """Parse ``figure: <filename> — <caption>`` lines anywhere in text."""
    captions: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^\s*figure:\s*(\S+)\s*[-\u2013\u2014]\s*(.+?)\s*$", line)
        if match:
            captions[match.group(1)] = match.group(2)
    return captions


def _dir_mtime(directory: Path) -> float:
    """Newest mtime among files directly in the directory.

    ``index.html`` is skipped: this script writes one into every subdirectory,
    so counting it would reset each directory's mtime on every rebuild and make
    it useless for ordering.
    """
    mtimes = [p.stat().st_mtime for p in directory.iterdir() if p.is_file() and p.name != "index.html"]
    return max(mtimes) if mtimes else directory.stat().st_mtime


def _entry_time(pr: PrDir) -> float:
    """Sort timestamp: the ``date`` metadata field when usable, else mtime."""
    raw = pr.meta.get("date")
    if raw:
        try:
            return datetime.strptime(str(raw).strip(), "%Y-%m-%d").timestamp()
        except ValueError:
            pass
    return pr.mtime


# Recognised review states.  `needs review` is outstanding, `deferred` is a
# deliberate park (neither finished nor awaiting attention), the rest are done.
OUTSTANDING_STATUSES = {"needs review"}
PARKED_STATUSES = {"deferred"}
DONE_STATUSES = {"reviewed", "merged", "closed", "superseded"}
KNOWN_STATUSES = OUTSTANDING_STATUSES | PARKED_STATUSES | DONE_STATUSES


def status_class(status: str) -> str:
    """CSS class for a status: one of ``todo``, ``parked`` or ``done``."""
    if status in PARKED_STATUSES:
        return "parked"
    if status in DONE_STATUSES:
        return "done"
    return "todo"


# PR state -> status label, used when `summary.md` does not set one.
_PR_STATE_STATUS = {"MERGED": "merged", "CLOSED": "closed", "OPEN": "needs review"}


def fetch_pr_states(timeout: float = 20.0) -> dict[int, str]:
    """Map PR number -> GitHub state, via one `gh` call.

    Returns an empty map when `gh` is missing, unauthenticated or offline; the
    caller then falls back to whatever `summary.md` declares.  One batched call
    rather than one per directory.
    """
    try:
        proc = subprocess.run(
            ["gh", "pr", "list", "--state", "all", "--limit", "200", "--json", "number,state"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if proc.returncode != 0:
        return {}
    try:
        return {int(row["number"]): str(row["state"]) for row in json.loads(proc.stdout or "[]")}
    except (ValueError, KeyError, TypeError):
        return {}


def resolve_status(pr: PrDir, pr_states: dict[int, str]) -> str:
    """Review status for one directory.

    An explicit ``status:`` in ``summary.md`` always wins, so a merged PR can
    still be flagged for follow-up.  Otherwise the PR's GitHub state decides.
    Directories with no PR and no ``status:`` are "needs review": an
    investigation is outstanding until someone says it is not.
    """
    declared = (pr.meta.get("status") or "").strip().lower()
    if declared:
        if declared not in KNOWN_STATUSES:
            pr.warnings.append(
                f"visual_checks/{pr.path.name}: unknown status {declared!r} in summary.md; "
                f"expected one of {', '.join(sorted(KNOWN_STATUSES))}"
            )
        return declared
    if pr.pr is not None and pr.pr in pr_states:
        return _PR_STATE_STATUS.get(pr_states[pr.pr], "needs review")
    if pr.pr is not None:
        return "unknown"
    return "needs review"


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------


@dataclass
class PrDir:
    path: Path
    pr: int | None
    title: str
    description: str
    url: str | None
    meta: dict[str, str]
    body_text: str
    figure_count: int
    mtime: float
    warnings: list[str] = field(default_factory=list)


def scan_pr_dir(directory: Path) -> PrDir:
    summary_path = directory / "summary.md"
    summary_text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    meta, body_text = _split_metadata_block(summary_text)

    warnings: list[str] = []
    for key in REQUIRED_META_KEYS:
        if not meta.get(key):
            warnings.append(f"visual_checks/{directory.name}: missing required metadata key '{key}' in summary.md")

    pr_raw = meta.get("pr")
    pr_number: int | None = None
    if pr_raw:
        try:
            pr_number = int(pr_raw)
        except ValueError:
            warnings.append(f"visual_checks/{directory.name}: metadata 'pr' is not an integer ({pr_raw!r})")

    title = meta.get("title") or directory.name
    description = meta.get("description") or ""
    url = meta.get("url") or None

    figure_count = len(sorted(directory.glob("*.png")))

    for msg in warnings:
        print(f"WARNING: {msg}", file=sys.stderr)

    return PrDir(
        path=directory,
        pr=pr_number,
        title=title,
        description=description,
        url=url,
        meta=meta,
        body_text=body_text,
        figure_count=figure_count,
        mtime=_dir_mtime(directory),
        warnings=warnings,
    )


def build_pr_page(pr: PrDir, status: str = "needs review") -> str:
    captions = _parse_figure_captions(pr.body_text)

    dl_items: list[tuple[str, str]] = []
    dl_items.append(("Status", f'<span class="status {status_class(status)}">{html.escape(status)}</span>'))
    if pr.url:
        dl_items.append(("URL", f'<a href="{html.escape(pr.url)}">{html.escape(pr.url)}</a>'))
    else:
        dl_items.append(("URL", "(missing)"))
    for key in ("branch", "base", "date", "before", "after", "inputs"):
        value = pr.meta.get(key)
        if value:
            dl_items.append((META_LABELS[key], _render_inline(value)))
    dl_html = (
        '<dl class="meta-list">\n'
        + "\n".join(f"<dt>{html.escape(label)}</dt><dd>{value}</dd>" for label, value in dl_items)
        + "\n</dl>"
    )

    description_html = f'<p class="desc">{_render_inline(pr.description)}</p>' if pr.description else ""

    body_html = render_markdown(pr.body_text)

    figures = []
    for png in sorted(pr.path.glob("*.png")):
        caption = captions.get(png.name, png.name)
        figures.append(
            f'<div class="figure">\n'
            f'<img src="{html.escape(png.name)}" alt="{html.escape(caption)}">\n'
            f'<p class="caption">{_render_inline(caption)}</p>\n'
            f"</div>"
        )

    heading = f"PR #{pr.pr}: {pr.title}" if pr.pr is not None else pr.title
    back_link = '<a class="back" href="../index.html">&larr; back to index</a>'

    body = (
        f"{back_link}\n"
        f"<h1>{html.escape(heading)}</h1>\n"
        f"{dl_html}\n"
        f"{description_html}\n"
        f"{body_html}\n"
        f"{''.join(figures)}\n"
        f"{back_link}\n"
    )
    return _page_shell(heading, body)


def build_index_page(pr_dirs: list[PrDir], statuses: dict[Path, str] | None = None) -> str:
    statuses = statuses or {}
    rows = []
    for pr in pr_dirs:
        status = statuses.get(pr.path, "needs review")
        cls = status_class(status)
        row_class = "" if cls == "done" else f' class="{cls}"'
        status_cell = f'<span class="status {cls}">{html.escape(status)}</span>'
        pr_cell = (
            f'<a href="{html.escape(pr.url)}">#{pr.pr}</a>'
            if pr.url and pr.pr is not None
            else (str(pr.pr) if pr.pr is not None else "-")
        )
        title_cell = f'<a href="{html.escape(pr.path.name)}/index.html">{html.escape(pr.title)}</a>'
        desc_cell = html.escape(pr.description)
        date_cell = html.escape(pr.meta.get("date", ""))
        rows.append(
            f"<tr{row_class}>"
            f"<td>{status_cell}</td>"
            f"<td>{pr_cell}</td>"
            f"<td>{title_cell}</td>"
            f"<td>{desc_cell}</td>"
            f"<td>{date_cell}</td>"
            f"<td>{pr.figure_count}</td>"
            "</tr>"
        )

    headers = ("Status", "PR", "Title", "Description", "Date", "Figures")
    header_row = "".join(f'<th data-col="{i}" title="Sort by {h}">{h}</th>' for i, h in enumerate(headers))
    table = (
        f'<table id="toc">\n<thead><tr>{header_row}</tr></thead>\n<tbody>\n' + "\n".join(rows) + "\n</tbody>\n</table>"
    )

    counts = Counter(status_class(statuses.get(pr.path, "needs review")) for pr in pr_dirs)
    parts = [f"{counts['todo']} of {len(pr_dirs)} awaiting review"]
    if counts["parked"]:
        parts.append(f"{counts['parked']} deferred")
    lead = f"<p>{', '.join(parts)}. Newest first; click a column heading to re-sort.</p>"
    body = "<h1>Visual checks</h1>\n" + lead + "\n" + table + SORT_SCRIPT
    return _page_shell("Visual checks", body)


def _sort_key(pr: PrDir) -> tuple[float, int]:
    # Newest first, by the `date` metadata field when present and the directory
    # mtime otherwise.  PR number breaks ties within a date.  mtime is
    # deliberately NOT a tie-break here: --set-status rewrites summary.md, so
    # using it would let changing a status silently reorder the whole index.
    return (-_entry_time(pr), -(pr.pr or 0))


def set_status(directory: Path, status: str) -> None:
    """Write ``status:`` into a directory's summary.md, replacing any existing one.

    The generated pages are static files, so there is no control in the browser
    to click; this is how a status is changed by hand or by a script.
    """
    status = status.strip().lower()
    if status not in KNOWN_STATUSES:
        raise SystemExit(f"Unknown status {status!r}; expected one of {', '.join(sorted(KNOWN_STATUSES))}")
    summary = directory / "summary.md"
    if not summary.is_file():
        raise SystemExit(f"No summary.md in {directory}")

    text = summary.read_text(encoding="utf-8")
    match = re.match(r"(---\n)(.*?)(\n---\n)", text, re.S)
    if not match:
        raise SystemExit(f"{summary} has no metadata header to write into")

    header = match.group(2)
    if re.search(r"^status:.*$", header, re.M):
        header = re.sub(r"^status:.*$", f"status: {status}", header, count=1, flags=re.M)
    else:
        header = f"{header}\nstatus: {status}"
    summary.write_text(text[: match.start(2)] + header + text[match.end(2) :], encoding="utf-8")
    print(f"{directory.name}: status set to {status}")


def list_status(pr_dirs: list[PrDir], statuses: dict[Path, str]) -> None:
    """Print each directory and its status, outstanding first."""
    order = {"todo": 0, "parked": 1, "done": 2}
    for pr in sorted(pr_dirs, key=lambda d: (order[status_class(statuses[d.path])], -_entry_time(d))):
        pr_label = f"#{pr.pr}" if pr.pr is not None else "-"
        print(f"{statuses[pr.path]:<13} {pr_label:>5}  {pr.path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate visual_checks/index.html and per-PR pages.")
    parser.add_argument(
        "--root",
        default="visual_checks",
        help="Root directory containing per-PR subdirectories (default: visual_checks, relative to cwd).",
    )
    parser.add_argument(
        "--set-status",
        nargs=2,
        metavar=("DIR", "STATUS"),
        help=f"Set a directory's review status and rebuild. One of: {', '.join(sorted(KNOWN_STATUSES))}.",
    )
    parser.add_argument(
        "--list-status",
        action="store_true",
        help="Print each page's status, outstanding first, and exit without rebuilding.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip the gh lookup for PR state; use the status declared in summary.md only.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Root directory not found: {root}")

    if args.set_status:
        name, status = args.set_status
        set_status(root / name, status)

    pr_dirs = [scan_pr_dir(p) for p in root.iterdir() if p.is_dir()]

    pr_dirs.sort(key=_sort_key)

    pr_states = {} if args.offline else fetch_pr_states()
    statuses = {pr.path: resolve_status(pr, pr_states) for pr in pr_dirs}
    if not pr_states and not args.offline and any(pr.pr is not None for pr in pr_dirs):
        print("note: could not read PR state from gh; using declared status only", file=sys.stderr)

    if args.list_status:
        list_status(pr_dirs, statuses)
        return

    for pr in pr_dirs:
        (pr.path / "index.html").write_text(build_pr_page(pr, statuses[pr.path]), encoding="utf-8")

    (root / "index.html").write_text(build_index_page(pr_dirs, statuses), encoding="utf-8")
    (root / "README.md").write_text(README_TEXT, encoding="utf-8")

    total_warnings = sum(len(pr.warnings) for pr in pr_dirs)
    print(f"Wrote {root / 'index.html'}, {root / 'README.md'}, and {len(pr_dirs)} PR page(s).")
    if total_warnings:
        print(f"{total_warnings} warning(s) - see above.", file=sys.stderr)


if __name__ == "__main__":
    main()
