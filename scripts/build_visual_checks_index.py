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
    issue: 26
    url: https://github.com/bright-fakl/carto-flow/pull/27
    branch: fix/voronoi-smoothing-tolerance
    base: fix/voronoi-cell-extraction
    date: 2026-09-18 14:30
    before: origin/fix/voronoi-cell-extraction
    after: fix/voronoi-smoothing-tolerance @ <sha>
    inputs: districts (bundled, simplify 5000 m, min_island 50000), states (bundled)
    ---

``title`` and ``description`` are required - a missing one prints a warning
naming the directory and falls back to the directory name. All other keys are
optional and, when present, are rendered in the page's definition list: ``pr``,
``issue``, ``url``, ``kind``, ``topic``, ``status``, ``outcome``, ``related``,
``branch``, ``base``, ``date``, ``updated``, ``before``, ``after`` and ``inputs``.

``kind`` is ``pr`` (a change under review; the default when ``pr`` is set),
``exploration`` (experiments and investigations; the default otherwise) or
``proposal`` (a change not yet in a PR).  ``topic`` is a comma-separated list
from ``flow``, ``voronoi``, ``symbol``, ``proportional``, ``data``, ``infra``.

Status of a ``pr`` page comes from the PR's GitHub state via one ``gh`` call -
merged or closed is done, open needs review - and an explicit ``status:``
always wins, so a merged PR can still be flagged for follow-up.  Its values are
``needs review`` (outstanding), ``deferred`` (a deliberate park), and
``reviewed`` / ``merged`` / ``closed`` / ``superseded`` (done).  An exploration
or proposal has no GitHub state; its ``status:`` is ``open`` (outstanding,
the default), ``on-hold`` (parked), or one of three closed outcomes:
``led-to`` (followed by a proposal or PR), ``superseded`` (replaced by a newer
exploration or synthesis page) and ``no-action`` (concluded, nothing follows;
``outcome:`` must say why).  A closed ``led-to`` or ``superseded`` page needs
a ``related`` link to its successor.  The generated pages are static, so there
is no control to click - use ``--set-status DIR STATUS`` to change one and
``--list-status`` to print them all.

``related`` lists the directory names of other pages, comma-separated; the reverse link
("Referenced by") is derived, so each link is written once.

A page's directory name is its permanent identifier: other pages and external
links point at it, so a page is never renamed.  Converting a proposal into a PR
edits the metadata only (``--convert DIR PR``).  Name new pages for their
subject, without a ``pr<N>-`` or ``proposal-`` prefix, since the kind and the PR
number are metadata and change over a page's life.

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

The root directory is the first of: ``--root``, ``VISUAL_CHECKS_ROOT`` in the
environment, ``VISUAL_CHECKS_ROOT`` in a ``.env`` file at the repository root,
and the sibling checkout ``../carto-flow-dev``.  Close times from GitHub are
shown in the zone named by ``VISUAL_CHECKS_TZ`` (an IANA name such as
``America/New_York``), else the system zone; set it wherever pages are rebuilt
on a machine in another zone.  Directories whose names start
with ``.`` are ignored.  PR states are read from ``--repo`` (default
``bright-fakl/carto-flow``).

Usage:
    uv run python scripts/build_visual_checks_index.py
    uv run python scripts/build_visual_checks_index.py --root /path/to/carto-flow-dev
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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

Review pages for PRs, explorations and proposals, built by
`scripts/build_visual_checks_index.py` in the carto-flow repository. This
repository holds the pages and is published with GitHub Pages at
https://bright-fakl.github.io/carto-flow-dev/. Pages are plain static files,
so a local clone can be opened directly in a browser without a server.

## Kinds, topics and status

Every page has a `kind`:

- `pr`: a change under review. This is the default when `pr` is set.
- `exploration`: an experiment or investigation. This is the default otherwise.
- `proposal`: a change worked out in a page but not yet in a PR.

`topic` is a comma-separated list from `flow`, `voronoi`, `symbol`, `proportional`,
`data` and `infra`. The index filters and sorts on kind, topic and status.

The status of a `pr` page comes from the PR on GitHub (see below). An exploration or
proposal has no GitHub state, so its `status` is set by hand:

| status | meaning |
|---|---|
| `open` | in progress or waiting for a decision (the default) |
| `on-hold` | paused on purpose |
| `led-to` | closed; followed by a proposal or PR |
| `superseded` | closed; replaced by a newer exploration or a synthesis page |
| `no-action` | closed; concluded and nothing follows. Requires `outcome:` saying why, for example a negative result, reference data only, or a rejected proposal |

Link pages with `related: <directory>, <directory>`. Write each link once; the page
at the other end shows it as "Referenced by". A `led-to` or `superseded` page needs
a link to its successor in either direction. Superseded pages are hidden in the
index until "show superseded" is ticked.

A directory name is the permanent identifier of a page. Links from other pages,
PR descriptions and issues use it, so never rename a directory. Name pages for their
subject, without a `pr<N>-` or `proposal-` prefix; the kind and the PR number are
metadata and change during a page's life.

- An exploration is evidence (figures and tables from a specific commit). Leave it
  as it is, set it to `led-to`, and write the proposal or PR as a new page that
  lists it under `related`.
- A proposal that becomes one PR is converted in place:
  `build_visual_checks_index.py --convert <directory> <PR number>` sets `kind: pr`,
  `pr` and `url` and removes `status`. Add a short "as proposed / as built" section.
  A proposal that splits into several PRs stays a proposal, set to `led-to`, with
  one `related` entry per PR.

## Every PR needs a page

Each PR gets its own page, `<slug>/summary.md`, whether or not the
change produces figures. Review happens from the generated index, not from
the PR description, so evidence kept only in the description or inside an
investigation workspace is easy to miss.

- For a PR page, set `pr`, `title`, `description` and `url` in the header
  (the builder itself only requires `title` and `description`), and `issue`
  when the PR addresses a GitHub issue.
- Before the PR exists, leave out `pr` and `url` (do not write `TBD`; it
  warns) and set `branch`. Add both once the PR is open; the directory keeps its name.
- A change that can alter cartogram output (algorithm, solver, repair,
  option defaults, data resolution) needs side-by-side before/after figures
  on the standard inputs (US states, congressional districts), so the
  reviewer can judge the result.
- A change with nothing to show (bit-identical output, housekeeping, error
  messages) still gets a page. It says so and gives the written evidence,
  for example the test results or the checked outputs that are unchanged.
- A page with figures also contains the script that produced them,
  `make_figures.py`, so the figures can be regenerated and checked against
  later code. It runs from a carto-flow checkout with
  `uv run python make_figures.py`, uses bundled data only, writes the PNGs next
  to itself and starts with a comment saying what it produces. Name the
  commit it ran against in `after` (`branch @ <sha>`). Do not commit caches or
  downloaded data; the page says how to get them instead.
- Explorations may hold the underlying figures. The PR page then points at the
  panels that justify it and lists the exploration under `related`.
- Do not write `status: needs review` in a new `pr` page. That is already the
  default, and an explicit `status:` overrides the PR's GitHub state, so the
  page would stay "needs review" after the PR merges. Set `status:` on a `pr`
  page only to override, for example `deferred` for parked work.

## Adding a check

1. Create a subdirectory named for its subject, e.g. `voronoi-smoothing-tolerance`.
2. Drop PNGs and a `summary.md` in it. `summary.md` must start with a fenced
   metadata header:

   ```
   ---
   pr: 27
   title: Fix coverage_simplify tolerance units in raster Voronoi cells
   description: One-line what changed and what to look for in the figures.
   issue: 26
   url: https://github.com/bright-fakl/carto-flow/pull/27
   branch: fix/voronoi-smoothing-tolerance
   base: fix/voronoi-cell-extraction
   date: 2026-09-18 14:30
   before: origin/fix/voronoi-cell-extraction
   after: fix/voronoi-smoothing-tolerance @ <sha>
   inputs: districts (bundled, simplify 5000 m, min_island 50000), states (bundled)
   ---
   ```

   `title` and `description` are required; `pr`, `issue` (the GitHub issue the
   change addresses, e.g. `74` or `74, 75`), `url`, `kind`, `topic`, `status`, `outcome`,
   `related`, `branch`, `base`, `date`, `updated`, `before`, `after`, and `inputs` are optional and rendered
   in a definition list on the page.  Pages are listed newest first, by `date`
   where given and by directory mtime otherwise.  Write `date`
   (the creation time) and `updated` (the last time the figures were
   regenerated) as `YYYY-MM-DD HH:MM`, local time: without the time, pages from
   the same day sort by PR number. The closed time is not written by hand; it
   is the merge or close time of the PR on GitHub.

   The status of a `pr` page is taken from the PR's GitHub state unless `status:` says
   otherwise. Values for a `pr` page: `needs review`, `deferred`, `reviewed`, `merged`,
   `closed`, `superseded`; for other kinds see the table above. Change one with
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
uv run python scripts/build_visual_checks_index.py --root /path/to/carto-flow-dev
```

Regenerates `index.html` and every `<subdir>/index.html`, plus this file.

Close times from GitHub are shown in the zone named by `VISUAL_CHECKS_TZ` (an IANA name, for
example `America/New_York`), else the system zone. Set it wherever pages are rebuilt on a machine
in another zone, so the generated files do not differ between machines.

## Publishing

Commit the page directory together with the regenerated `index.html` files
and push to `main`. GitHub Pages serves the repository as it is; there is no
build step. Create a page before the PR exists, then fill in `pr` and `url` once
the PR is open.
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
  var FIRST_ASC = [true, true, true, false, true, true, false, false, false];
  // Status sorts by how much attention an item still needs, not alphabetically:
  // "needs review" before "deferred" before anything finished.
  var STATUS_RANK = {
    "needs review": 0,
    "open": 0,
    "deferred": 1,
    "on-hold": 1,
    "reviewed": 2,
    "merged": 3,
    "closed": 4,
    "led-to": 5,
    "no-action": 6,
    "superseded": 7
  };
  function cellValue(row, i) {
    var text = (row.cells[i].innerText || "").trim();
    if (i === 0) {                                                  // Status
      var rank = STATUS_RANK[text.toLowerCase()];
      return rank === undefined ? -1 : rank;                        // unknown first
    }
    if (i === 3) return parseInt(text.replace("#", ""), 10) || -1;  // PR
    if (i === 8) return parseInt(text, 10) || 0;                    // Figures
    return text.toLowerCase();
  }
  // Filters: a row shows when it matches every selection.  Superseded pages
  // are hidden until asked for.
  var CLASS_NAMES = {awaiting: "todo", parked: "parked", done: "done"};
  function applyFilters() {
    var kind = document.getElementById("filter-kind").value;
    var topic = document.getElementById("filter-topic").value;
    var cls = CLASS_NAMES[document.getElementById("filter-class").value] || "";
    var superseded = document.getElementById("filter-superseded").checked;
    var shown = 0;
    Array.prototype.forEach.call(body.rows, function (row) {
      var d = row.dataset;
      var show = (!kind || d.kind === kind) &&
                 (!topic || d.topics.indexOf(" " + topic + " ") >= 0) &&
                 (!cls || d.class === cls) &&
                 (superseded || d.status !== "superseded");
      row.hidden = !show;
      if (show) shown += 1;
    });
    document.getElementById("filter-count").textContent = shown + " of " + body.rows.length + " shown";
  }
  document.querySelectorAll(".filters select, .filters input").forEach(function (el) {
    el.addEventListener("change", applyFilters);
  });
  applyFilters();
  table.querySelectorAll("th[data-col]").forEach(function (th) {
    th.addEventListener("click", function () {
      var i = +th.dataset.col;
      var asc = state[i] = (i in state) ? !state[i] : FIRST_ASC[i];
      var rows = Array.prototype.slice.call(body.rows);
      rows.sort(function (a, b) {
        var x = cellValue(a, i), y = cellValue(b, i);
        if (x < y) return asc ? -1 : 1;
        if (x > y) return asc ? 1 : -1;
        // Ties fall back to the order the page was generated in, so sorting a
        // column where many rows share a value (every page from the same day,
        // say) reproduces the default order rather than whatever the previous
        // click happened to leave behind.
        return (+a.dataset.ord) - (+b.dataset.ord);
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
OPTIONAL_META_KEYS = (
    "pr",
    "issue",
    "url",
    "kind",
    "topic",
    "status",
    "outcome",
    "related",
    "branch",
    "base",
    "date",
    "updated",
    "before",
    "after",
    "inputs",
)
META_LABELS = {
    "url": "URL",
    "branch": "Branch",
    "base": "Base",
    "date": "Created",
    "updated": "Updated",
    "closed": "Closed",
    "status": "Status",
    "kind": "Kind",
    "topic": "Topic",
    "outcome": "Outcome",
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
    """Sort timestamp from the ``date`` field, else the directory mtime.

    ``date`` may carry an optional ``HH:MM``; without one it resolves to
    midnight, so pages from the same day are separated by PR number instead.
    """
    raw = str(pr.meta.get("date") or "").strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).timestamp()
        except ValueError:
            continue
    return pr.mtime


# Recognized statuses.  A PR page uses the first group, an exploration or
# proposal the second; the sets below are their union.  Outstanding items need
# attention, parked ones are a deliberate pause, the rest are finished.
PR_STATUSES = {"needs review", "deferred", "reviewed", "merged", "closed", "superseded"}
EXPLORATION_STATUSES = {"open", "on-hold", "led-to", "superseded", "no-action"}
OUTSTANDING_STATUSES = {"needs review", "open"}
PARKED_STATUSES = {"deferred", "on-hold"}
DONE_STATUSES = {"reviewed", "merged", "closed", "superseded", "led-to", "no-action"}
KNOWN_STATUSES = OUTSTANDING_STATUSES | PARKED_STATUSES | DONE_STATUSES

KINDS = ("pr", "exploration", "proposal")
TOPICS = ("flow", "voronoi", "symbol", "proportional", "data", "infra")


def status_class(status: str) -> str:
    """CSS class for a status: one of ``todo``, ``parked`` or ``done``."""
    if status in PARKED_STATUSES:
        return "parked"
    if status in DONE_STATUSES:
        return "done"
    return "todo"


DEFAULT_REPO = "bright-fakl/carto-flow"
ROOT_VARIABLE = "VISUAL_CHECKS_ROOT"
TIMEZONE_VARIABLE = "VISUAL_CHECKS_TZ"
REPO_ROOT = Path(__file__).resolve().parent.parent


def display_timezone() -> tzinfo | None:
    """Time zone for dates read from GitHub: ``VISUAL_CHECKS_TZ`` (an IANA name), else the system zone.

    Pages are rebuilt on machines in different zones (a laptop, a CI runner in
    UTC); a fixed zone keeps the generated files identical between them.
    """
    name = os.environ.get(TIMEZONE_VARIABLE, "").strip()
    if not name:
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        print(f"WARNING: unknown time zone {name!r} in {TIMEZONE_VARIABLE}; using the system zone", file=sys.stderr)
        return None


def default_root() -> Path:
    """Root directory when ``--root`` is not given.

    ``VISUAL_CHECKS_ROOT`` from the environment, else from ``.env`` at the
    repository root (``KEY=value`` lines), else the sibling ``../carto-flow-dev``.
    """
    value = os.environ.get(ROOT_VARIABLE, "")
    env_file = REPO_ROOT / ".env"
    if not value and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition("=")
            if key.strip() == ROOT_VARIABLE:
                value = rest.strip().strip("\"'")
    return Path(value).expanduser() if value else REPO_ROOT.parent / "carto-flow-dev"


# PR state -> status label, used when `summary.md` does not set one.
_PR_STATE_STATUS = {"MERGED": "merged", "CLOSED": "closed", "OPEN": "needs review"}


def fetch_pr_rows(repo: str = DEFAULT_REPO, timeout: float = 20.0) -> list[dict]:
    """PRs of ``repo`` (number, state, closedAt, mergedAt), via one `gh` call.

    Returns an empty list when `gh` is missing, unauthenticated or offline; the
    caller then falls back to whatever `summary.md` declares.  One batched call
    rather than one per directory.
    """
    try:
        proc = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "gh",
                "pr",
                "list",
                "--repo",
                repo,
                "--state",
                "all",
                "--limit",
                "500",
                "--json",
                "number,state,closedAt,mergedAt",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    try:
        rows = json.loads(proc.stdout or "[]")
    except ValueError:
        return []
    return [row for row in rows if isinstance(row, dict) and "number" in row]


def pr_states_from_rows(rows: list[dict]) -> dict[int, str]:
    """Map PR number -> GitHub state."""
    return {int(row["number"]): str(row.get("state", "")) for row in rows}


def pr_closed_from_rows(rows: list[dict]) -> dict[int, str]:
    """Map PR number -> merge or close time as local ``YYYY-MM-DD HH:MM``, for closed PRs."""
    closed = {}
    for row in rows:
        raw = row.get("mergedAt") or row.get("closedAt")
        if not raw:
            continue
        try:
            moment = datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(display_timezone())
        except ValueError:
            continue
        closed[int(row["number"])] = moment.strftime("%Y-%m-%d %H:%M")
    return closed


def fetch_pr_states(repo: str = DEFAULT_REPO, timeout: float = 20.0) -> dict[int, str]:
    """Map PR number -> GitHub state of ``repo``."""
    return pr_states_from_rows(fetch_pr_rows(repo, timeout))


def resolve_status(pr: PrDir, pr_states: dict[int, str]) -> str:
    """Status for one directory.

    An explicit ``status:`` in ``summary.md`` always wins, so a merged PR can
    still be flagged for follow-up.  Otherwise a PR page takes the PR's GitHub
    state, and an exploration or proposal is ``open``: it is outstanding until
    someone says it is not.
    """
    declared = (pr.meta.get("status") or "").strip().lower()
    name = pr.path.name
    if declared:
        if declared not in KNOWN_STATUSES:
            pr.warnings.append(
                f"{name}: unknown status {declared!r} in summary.md; expected one of {', '.join(sorted(KNOWN_STATUSES))}"
            )
        else:
            allowed = PR_STATUSES if pr.kind == "pr" else EXPLORATION_STATUSES
            if declared not in allowed:
                pr.warnings.append(
                    f"{name}: status {declared!r} does not apply to kind {pr.kind!r}; expected one of "
                    f"{', '.join(sorted(allowed))}"
                )
        if declared == "no-action" and not pr.meta.get("outcome"):
            pr.warnings.append(f"{name}: status 'no-action' needs an 'outcome:' line saying why")
        return declared
    if pr.kind != "pr":
        return "open"
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
    kind: str = "exploration"
    topics: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def scan_pr_dir(directory: Path) -> PrDir:
    summary_path = directory / "summary.md"
    summary_text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    meta, body_text = _split_metadata_block(summary_text)

    warnings: list[str] = []
    for key in REQUIRED_META_KEYS:
        if not meta.get(key):
            warnings.append(f"{directory.name}: missing required metadata key '{key}' in summary.md")

    pr_raw = meta.get("pr")
    pr_number: int | None = None
    if pr_raw:
        try:
            pr_number = int(pr_raw)
        except ValueError:
            warnings.append(f"{directory.name}: metadata 'pr' is not an integer ({pr_raw!r})")

    issue_raw = (meta.get("issue") or "").strip()
    if issue_raw and not _issue_numbers(issue_raw):
        warnings.append(f"{directory.name}: metadata 'issue' has no issue number ({issue_raw!r})")

    kind = (meta.get("kind") or "").strip().lower() or ("pr" if pr_number is not None else "exploration")
    if kind not in KINDS:
        warnings.append(f"{directory.name}: unknown kind {kind!r}; expected one of {', '.join(KINDS)}")
    elif kind != "pr" and pr_number is not None:
        warnings.append(f"{directory.name}: kind {kind!r} with a 'pr' number; convert it with --convert")

    topics = _split_list(meta.get("topic", ""))
    for topic in topics:
        if topic not in TOPICS:
            warnings.append(f"{directory.name}: unknown topic {topic!r}; expected {', '.join(TOPICS)}")

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
        kind=kind,
        topics=topics,
        related=_split_list(meta.get("related", ""), lower=False),
        warnings=warnings,
    )


def _split_list(raw: str, lower: bool = True) -> list[str]:
    """Comma- or space-separated values of a metadata line."""
    parts = [part.strip() for part in raw.replace(",", " ").split() if part.strip()]
    return [part.lower() for part in parts] if lower else parts


def _issue_numbers(raw: str) -> list[int]:
    """Issue numbers from a comma- or space-separated ``issue`` value; others are ignored."""
    return [int(part.lstrip("#")) for part in raw.replace(",", " ").split() if part.lstrip("#").isdigit()]


def link_graph(pr_dirs: list[PrDir]) -> dict[str, list[str]]:
    """Map directory name -> names of the pages that list it in ``related``.

    Warns about ``related`` entries naming no page and about a closed
    ``led-to`` or ``superseded`` page with no link to a successor in either
    direction.
    """
    names = {pr.path.name for pr in pr_dirs}
    incoming: dict[str, list[str]] = {name: [] for name in names}
    for pr in pr_dirs:
        for target in pr.related:
            if target not in names:
                pr.warnings.append(f"{pr.path.name}: related {target!r} names no page")
                print(f"WARNING: {pr.warnings[-1]}", file=sys.stderr)
            else:
                incoming[target].append(pr.path.name)
    for pr in pr_dirs:
        closed_with_successor = (pr.meta.get("status") or "").strip().lower() in {"led-to", "superseded"}
        if closed_with_successor and pr.kind != "pr" and not pr.related and not incoming[pr.path.name]:
            pr.warnings.append(f"{pr.path.name}: closed with a successor but no 'related' link either way")
            print(f"WARNING: {pr.warnings[-1]}", file=sys.stderr)
    return incoming


def build_pr_page(
    pr: PrDir,
    status: str = "needs review",
    closed: str | None = None,
    pages: dict[str, PrDir] | None = None,
    incoming: dict[str, list[str]] | None = None,
) -> str:
    captions = _parse_figure_captions(pr.body_text)
    pages = pages or {}
    incoming = incoming or {}

    dl_items: list[tuple[str, str]] = []
    dl_items.append(("Kind", html.escape(pr.kind)))
    if pr.topics:
        dl_items.append(("Topic", html.escape(", ".join(pr.topics))))
    dl_items.append(("Status", f'<span class="status {status_class(status)}">{html.escape(status)}</span>'))
    if pr.meta.get("outcome"):
        dl_items.append((META_LABELS["outcome"], _render_inline(pr.meta["outcome"])))
    for label, ids in (
        ("Related", pr.related),
        ("Referenced by", [i for i in incoming.get(pr.path.name, []) if i not in pr.related]),
    ):
        links = [
            f'<a href="../{html.escape(pages[i].path.name)}/index.html">{html.escape(pages[i].title)}</a> '
            f'<span class="meta">({html.escape(pages[i].kind)})</span>'
            for i in ids
            if i in pages
        ]
        if links:
            dl_items.append((label, "<br>".join(links)))
    if pr.url:
        dl_items.append(("URL", f'<a href="{html.escape(pr.url)}">{html.escape(pr.url)}</a>'))
    elif pr.kind == "pr":
        dl_items.append(("URL", "(missing)"))
    issues = _issue_numbers(pr.meta.get("issue", ""))
    if issues:
        links = ", ".join(f'<a href="https://github.com/{DEFAULT_REPO}/issues/{n}">#{n}</a>' for n in issues)
        dl_items.append(("Issue", links))
    for key in ("branch", "base", "date", "updated", "closed", "before", "after", "inputs"):
        value = closed if key == "closed" else pr.meta.get(key)
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

    heading = f"PR #{pr.pr}: {pr.title}" if pr.pr is not None and pr.kind == "pr" else pr.title
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


def build_index_page(
    pr_dirs: list[PrDir], statuses: dict[Path, str] | None = None, closed: dict[int, str] | None = None
) -> str:
    statuses = statuses or {}
    closed = closed or {}
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
        closed_cell = html.escape(closed.get(pr.pr, "") if pr.pr is not None else "")
        rows.append(
            f'<tr{row_class} data-ord="{len(rows)}" data-kind="{pr.kind}" data-topics=" {" ".join(pr.topics)} " '
            f'data-status="{html.escape(status)}" data-class="{cls}">'
            f"<td>{status_cell}</td>"
            f"<td>{html.escape(pr.kind)}</td>"
            f"<td>{html.escape(', '.join(pr.topics))}</td>"
            f"<td>{pr_cell}</td>"
            f"<td>{title_cell}</td>"
            f"<td>{desc_cell}</td>"
            f"<td>{date_cell}</td>"
            f"<td>{closed_cell}</td>"
            f"<td>{pr.figure_count}</td>"
            "</tr>"
        )

    headers = ("Status", "Kind", "Topic", "PR", "Title", "Description", "Created", "Closed", "Figures")
    header_row = "".join(f'<th data-col="{i}" title="Sort by {h}">{h}</th>' for i, h in enumerate(headers))
    table = (
        f'<table id="toc">\n<thead><tr>{header_row}</tr></thead>\n<tbody>\n' + "\n".join(rows) + "\n</tbody>\n</table>"
    )

    counts = Counter(status_class(statuses.get(pr.path, "needs review")) for pr in pr_dirs)
    parts = [f"{counts['todo']} of {len(pr_dirs)} awaiting review"]
    if counts["parked"]:
        parts.append(f"{counts['parked']} deferred")
    lead = f"<p>{', '.join(parts)}. Newest first; click a column heading to re-sort.</p>"
    body = "<h1>Visual checks</h1>\n" + lead + "\n" + _filter_controls(pr_dirs) + table + SORT_SCRIPT
    return _page_shell("Visual checks", body)


def _filter_controls(pr_dirs: list[PrDir]) -> str:
    """Filter row above the table; the script in ``SORT_SCRIPT`` applies it."""

    def select(name: str, label: str, values: list[str]) -> str:
        options = "".join(f'<option value="{html.escape(v)}">{html.escape(v)}</option>' for v in values)
        return f'<label>{label} <select id="filter-{name}"><option value="">all</option>{options}</select></label> '

    kinds = [k for k in KINDS if any(pr.kind == k for pr in pr_dirs)]
    topics = [t for t in TOPICS if any(t in pr.topics for pr in pr_dirs)]
    states = ["awaiting", "parked", "done"]
    return (
        '<p class="filters">'
        + select("kind", "Kind", kinds)
        + select("topic", "Topic", topics)
        + select("class", "State", states)
        + '<label><input type="checkbox" id="filter-superseded"> show superseded</label> '
        + '<span id="filter-count" class="meta"></span></p>\n'
    )


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


def convert_to_pr(directory: Path, number: int, repo: str = DEFAULT_REPO) -> None:
    """Turn a proposal's page into a PR page: set ``kind``, ``pr`` and ``url``, drop ``status``.

    The directory keeps its name, so links to the page keep working; the status
    is then taken from the PR's GitHub state.
    """
    summary = directory / "summary.md"
    if not summary.is_file():
        raise SystemExit(f"No summary.md in {directory}")
    text = summary.read_text(encoding="utf-8")
    match = re.match(r"(---\n)(.*?)(\n---\n)", text, re.S)
    if not match:
        raise SystemExit(f"{summary} has no metadata header to write into")
    header = match.group(2)
    if re.search(r"^kind:\s*pr\s*$", header, re.M):
        raise SystemExit(f"{directory.name} is already a PR page")
    header = re.sub(r"^(kind|pr|url|status):.*\n?", "", header, flags=re.M).rstrip("\n")
    header += f"\nkind: pr\npr: {number}\nurl: https://github.com/{repo}/pull/{number}"
    summary.write_text(text[: match.start(2)] + header + text[match.end(2) :], encoding="utf-8")
    print(f"{directory.name}: converted to PR #{number}")


def list_status(pr_dirs: list[PrDir], statuses: dict[Path, str]) -> None:
    """Print each directory and its status, outstanding first."""
    order = {"todo": 0, "parked": 1, "done": 2}
    for pr in sorted(pr_dirs, key=lambda d: (order[status_class(statuses[d.path])], -_entry_time(d))):
        pr_label = f"#{pr.pr}" if pr.pr is not None else "-"
        print(f"{statuses[pr.path]:<13} {pr_label:>5}  {pr.path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the visual checks index and per-PR pages.")
    parser.add_argument(
        "--root",
        default=None,
        help=f"Root directory containing per-PR subdirectories (default: ${ROOT_VARIABLE}, else ../carto-flow-dev).",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"GitHub repository whose PR states set the review status (default: {DEFAULT_REPO}).",
    )
    parser.add_argument(
        "--set-status",
        nargs=2,
        metavar=("DIR", "STATUS"),
        help=f"Set a directory's status and rebuild. One of: {', '.join(sorted(KNOWN_STATUSES))}.",
    )
    parser.add_argument(
        "--convert",
        nargs=2,
        metavar=("DIR", "PR"),
        help="Turn a proposal page into the page of PR number PR (metadata only; the directory keeps its name).",
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

    root = Path(args.root) if args.root else default_root()
    if not root.is_dir():
        raise SystemExit(f"Root directory not found: {root}")

    if args.set_status:
        name, status = args.set_status
        set_status(root / name, status)

    if args.convert:
        name, number = args.convert
        convert_to_pr(root / name, int(number), args.repo)

    pr_dirs = [scan_pr_dir(p) for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith(".")]

    pr_dirs.sort(key=_sort_key)

    pr_rows = [] if args.offline else fetch_pr_rows(args.repo)
    pr_states = pr_states_from_rows(pr_rows)
    pr_closed = pr_closed_from_rows(pr_rows)
    statuses = {pr.path: resolve_status(pr, pr_states) for pr in pr_dirs}
    if not pr_states and not args.offline and any(pr.pr is not None for pr in pr_dirs):
        print("note: could not read PR state from gh; using declared status only", file=sys.stderr)

    if args.list_status:
        list_status(pr_dirs, statuses)
        return

    incoming = link_graph(pr_dirs)
    pages = {pr.path.name: pr for pr in pr_dirs}
    for pr in pr_dirs:
        closed = pr_closed.get(pr.pr) if pr.pr is not None else None
        (pr.path / "index.html").write_text(
            build_pr_page(pr, statuses[pr.path], closed, pages, incoming), encoding="utf-8"
        )

    (root / "index.html").write_text(build_index_page(pr_dirs, statuses, pr_closed), encoding="utf-8")
    (root / "README.md").write_text(README_TEXT, encoding="utf-8")

    total_warnings = sum(len(pr.warnings) for pr in pr_dirs)
    print(f"Wrote {root / 'index.html'}, {root / 'README.md'}, and {len(pr_dirs)} PR page(s).")
    if total_warnings:
        print(f"{total_warnings} warning(s) - see above.", file=sys.stderr)


if __name__ == "__main__":
    main()
