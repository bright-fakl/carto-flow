#!/usr/bin/env python3
"""Regenerate the visual_checks/ table-of-contents and per-PR pages.

Each subdirectory of the root holds a set of PNGs and a ``summary.md``
written by hand for a PR's before/after visual review. This script builds:

- ``<root>/index.html`` - a table of contents listing every subdirectory
  (sorted, newest mtime first) with title, one-line description, figure
  count, and a link to its page.
- ``<root>/<subdir>/index.html`` - a standalone page per subdirectory with
  the subdirectory's title, its ``summary.md`` rendered as HTML, and every
  PNG in the directory as an ``<img>`` (relative link, not embedded).

Only stdlib is used - no markdown library, no third-party dependencies.

Usage:
    uv run python scripts/build_visual_checks_index.py
    uv run python scripts/build_visual_checks_index.py --root visual_checks
    uv run python scripts/build_visual_checks_index.py --root /path/to/visual_checks
"""

from __future__ import annotations

import argparse
import html
import re
from dataclasses import dataclass
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
  th { background: rgba(0,0,0,0.05); }
  code { background: rgba(0,0,0,0.06); padding: 0.1em 0.35em; border-radius: 4px; font-size: 0.9em; }
  ul { padding-left: 1.4rem; }
  .figure { margin: 1.5rem 0; padding: 12px; background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; }
  .caption { font-size: 0.85rem; color: #555; margin: 0.5rem 0 0; }
  img { display: block; margin: 0 auto; max-width: 100%; }
  .toc-item { margin: 1rem 0; padding: 12px; background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; }
  .toc-item h2 { margin-top: 0; border-bottom: none; padding-bottom: 0; }
  .meta { font-size: 0.8rem; color: #777; }
  .desc { color: #333; }
  .back { display: inline-block; margin-top: 2rem; font-size: 0.9rem; }
  a { color: #0645ad; }
  table { overflow-x: auto; display: block; }
"""

DARK_STYLE = """
  @media (prefers-color-scheme: dark) {
    body { background: #1a1a1a; color: #eee; }
    table, th, td { border-color: #444 !important; }
    th { background: rgba(255,255,255,0.08); }
    code { background: rgba(255,255,255,0.1); }
    .figure, .toc-item { background: #242424; border-color: #333; }
    .meta { color: #999; }
    .desc { color: #ddd; }
    a { color: #7fb3ff; }
  }
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
        while i < n and lines[i].strip() and not re.match(r"^(#{1,6}\s|\||[-*]\s)", lines[i].strip()):
            para_lines.append(lines[i].strip())
            i += 1
        html_parts.append(f"<p>{_render_inline(' '.join(para_lines))}</p>")

    return "\n".join(html_parts)


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------


@dataclass
class PrDir:
    path: Path
    title: str
    description: str
    figure_count: int
    mtime: float


def _title_from_summary(summary_text: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.*)$", summary_text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return fallback


def _description_from_summary(summary_text: str) -> str:
    """First plain paragraph line after the title heading, if any."""
    lines = summary_text.splitlines()
    seen_title = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            seen_title = True
            continue
        if not seen_title:
            continue
        if stripped.startswith("|") or re.match(r"^[-*]\s+", stripped):
            continue
        return stripped
    return ""


def _dir_mtime(directory: Path) -> float:
    """Newest mtime among files directly in the directory."""
    mtimes = [p.stat().st_mtime for p in directory.iterdir() if p.is_file()]
    return max(mtimes) if mtimes else directory.stat().st_mtime


def scan_pr_dir(directory: Path) -> PrDir:
    summary_path = directory / "summary.md"
    summary_text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    title = _title_from_summary(summary_text, directory.name)
    description = _description_from_summary(summary_text)
    figure_count = len(sorted(directory.glob("*.png")))
    return PrDir(
        path=directory,
        title=title,
        description=description,
        figure_count=figure_count,
        mtime=_dir_mtime(directory),
    )


def _strip_title_heading(summary_text: str) -> str:
    """Drop the first top-level heading - it's already shown as the page <h1>."""
    return re.sub(r"^#\s+.*\n?", "", summary_text, count=1)


def build_pr_page(pr: PrDir) -> str:
    summary_path = pr.path / "summary.md"
    summary_html = (
        render_markdown(_strip_title_heading(summary_path.read_text(encoding="utf-8"))) if summary_path.exists() else ""
    )

    figures = []
    for png in sorted(pr.path.glob("*.png")):
        figures.append(
            f'<div class="figure">\n'
            f'<img src="{html.escape(png.name)}" alt="{html.escape(png.name)}">\n'
            f'<p class="caption">{html.escape(png.name)}</p>\n'
            f"</div>"
        )

    body = (
        f"<h1>{html.escape(pr.title)}</h1>\n"
        f"{summary_html}\n"
        f"{''.join(figures)}\n"
        f'<a class="back" href="../index.html">&larr; back to index</a>\n'
    )
    return _page_shell(pr.title, body)


def build_index_page(pr_dirs: list[PrDir]) -> str:
    items = []
    for pr in pr_dirs:
        desc_html = f'<p class="desc">{html.escape(pr.description)}</p>' if pr.description else ""
        items.append(
            f'<div class="toc-item">\n'
            f'<h2><a href="{html.escape(pr.path.name)}/index.html">{html.escape(pr.title)}</a></h2>\n'
            f"{desc_html}\n"
            f'<p class="meta">{pr.figure_count} figure(s)</p>\n'
            f"</div>"
        )

    body = "<h1>Visual checks</h1>\n<p>Before/after visual review pages for open PRs.</p>\n" + "".join(items)
    return _page_shell("Visual checks", body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate visual_checks/index.html and per-PR pages.")
    parser.add_argument(
        "--root",
        default="visual_checks",
        help="Root directory containing per-PR subdirectories (default: visual_checks, relative to cwd).",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Root directory not found: {root}")

    pr_dirs = sorted(
        (scan_pr_dir(p) for p in root.iterdir() if p.is_dir()),
        key=lambda pr: pr.mtime,
        reverse=True,
    )

    for pr in pr_dirs:
        (pr.path / "index.html").write_text(build_pr_page(pr), encoding="utf-8")

    (root / "index.html").write_text(build_index_page(pr_dirs), encoding="utf-8")

    print(f"Wrote {root / 'index.html'} and {len(pr_dirs)} PR page(s).")


if __name__ == "__main__":
    main()
