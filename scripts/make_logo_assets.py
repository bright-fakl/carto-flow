"""Generate the carto-flow logo assets in ``docs/assets/logo/``.

The mark is an irregular, country-like outline cut into four regions by two wavy gaps that cross at
72 degrees. The two right-hand regions are amber, the two left-hand regions blue. The wordmark text is
converted to outlined paths from DejaVu Sans Bold, so the SVG files contain no live text and need no font.

Run from the repository root (numpy, shapely and matplotlib come from the project environment; the
other packages are only needed here and are not project dependencies)::

    uv run --with cairosvg --with pillow --with fonttools python scripts/make_logo_assets.py

The output directory can be changed with ``--out``. The script prints the WCAG contrast ratios of the
logo colors against the docs surfaces.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from shapely.affinity import rotate
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

# Colors. Blue and the dark amber are the PlotStyle colors; the light amber is darker than the
# PlotStyle highlight (#d98200, 2.9:1 on white) so that it reaches 3:1 on the light surfaces.
LIGHT = {"blue": "#2c6e9e", "amber": "#c27400", "text": "#1b1f24"}
DARK = {"blue": "#85b7e0", "amber": "#facc38", "text": "#e8edf2"}
# Docs surfaces: Material default scheme (white) and slate scheme (hsl(232, 15%, 14%)), headers white and black.
SURFACES = {"light page": "#ffffff", "light header": "#ffffff", "dark page": "#1e2029", "dark header": "#000000"}
DARK_BG = "#1b1f24"  # surface of the social preview and of the preview figures

GAP_NORMAL, GAP_FAVICON, GAP_HERO = 8.0, 10.0, 6.0
TAGLINE = "Cartograms from geographic data in Python"

# Outline: radial harmonics (order, amplitude, phase), x/y scale and radius in the 96 x 96 viewBox.
HARMONICS = [(2, 0.09, 2.2), (3, 0.12, 0.4), (5, 0.05, 1.7)]
SX, SY, RADIUS = 0.85, 1.0, 40.0
TILT_V, TILT_H = 10.0, -8.0  # rotation of the vertical and horizontal gap in degrees (72 degrees between them)
WAVE_AMP, WAVE_FV, WAVE_FH = 5.0, 16.0, 20.0


def blob(cx: float = 48.0, cy: float = 48.0, m: int = 240) -> Polygon:
    t = np.linspace(0, 2 * math.pi, m, endpoint=False)
    rad = 1 + sum(a * np.cos(k * t + ph) for k, a, ph in HARMONICS)
    return Polygon(list(zip(cx + SX * RADIUS * rad * np.cos(t), cy + SY * RADIUS * rad * np.sin(t), strict=True)))


def split(gap: float, cut_x: float, cut_y: float) -> list[Polygon]:
    """The outline minus a wavy vertical and a wavy horizontal gap of width ``gap``, sorted top to bottom."""
    ys = np.linspace(0, 96, 120)
    v = LineString([(cut_x + WAVE_AMP * math.sin(y / WAVE_FV + 0.4), y) for y in ys])
    h = LineString([(x, cut_y + WAVE_AMP * math.sin(x / WAVE_FH + 1.2)) for x in ys])
    v = rotate(v, TILT_V, origin=(48, 48))
    h = rotate(h, TILT_H, origin=(48, 48))
    cut = unary_union([v, h]).buffer(gap / 2, cap_style="flat")
    shape = blob()
    pieces = shape.difference(cut)
    geoms = list(getattr(pieces, "geoms", [pieces]))
    if len(geoms) != 4 or min(g.area for g in geoms) < 0.08 * shape.area:
        raise ValueError("cut does not give four regions")
    return sorted(geoms, key=lambda g: (-g.centroid.y, g.centroid.x))


def best_cut() -> tuple[int, int]:
    """Cut position with four regions, the smallest as large as possible and largest/smallest >= 1.4."""
    best = None
    for cx in range(30, 64, 2):
        for cy in range(34, 66, 2):
            try:
                areas = sorted(g.area for g in split(GAP_NORMAL, cx, cy))
            except ValueError:
                continue
            if areas[-1] / areas[0] >= 1.4 and (best is None or areas[0] > best[0]):
                best = (areas[0], cx, cy)
    if best is None:
        raise ValueError("no cut with four regions")
    return best[1], best[2]


def regions(gap: float) -> tuple[list[list[tuple[float, float]]], list[bool]]:
    """Region outlines and a flag per region (True = amber: the two with the largest centroid x)."""
    cx, cy = best_cut()
    pieces = split(gap, cx, cy)
    amber = set(sorted(range(4), key=lambda i: -pieces[i].centroid.x)[:2])
    return [list(g.simplify(0.03).exterior.coords) for g in pieces], [i in amber for i in range(4)]


def d(points: list[tuple[float, float]]) -> str:
    return "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in points) + " Z"


def mark_bounds(gap: float = GAP_NORMAL) -> tuple[float, float, float, float]:
    polys, _ = regions(gap)
    xs = [p[0] for poly in polys for p in poly]
    ys = [p[1] for poly in polys for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def mark_body(gap: float, blue: str, amber: str, classes: bool = False) -> str:
    polys, is_amber = regions(gap)
    out = ""
    for poly, am in zip(polys, is_amber, strict=True):
        fill = f'class="{"a" if am else "b"}"' if classes else f'fill="{amber if am else blue}"'
        out += f'<path {fill} d="{d(poly)}"/>'
    return out


def square_viewbox(gap: float, margin: float = 2.0) -> tuple[float, float, float]:
    """Square viewBox (x, y, size) centered on the mark's bounding box."""
    x0, y0, x1, y1 = mark_bounds(gap)
    size = max(x1 - x0, y1 - y0) + 2 * margin
    return (x0 + x1) / 2 - size / 2, (y0 + y1) / 2 - size / 2, size


def svg_open(vb: tuple[float, float, float, float], title: str) -> str:
    x, y, w, h = vb
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x:.2f} {y:.2f} {w:.2f} {h:.2f}" '
        f'width="{w:.0f}" height="{h:.0f}" role="img" aria-label="{title}"><title>{title}</title>'
    )


def mark_svg(pal: dict[str, str], gap: float = GAP_NORMAL, mono: bool = False) -> str:
    x, y, s = square_viewbox(gap)
    if mono:
        body = "".join(f'<path d="{d(p)}"/>' for p in regions(gap)[0])
        return svg_open((x, y, s, s), "carto-flow") + f'<g fill="currentColor">{body}</g></svg>\n'
    return svg_open((x, y, s, s), "carto-flow") + mark_body(gap, pal["blue"], pal["amber"]) + "</svg>\n"


def favicon_svg() -> str:
    x, y, s = square_viewbox(GAP_FAVICON, margin=1.0)
    style = (
        f"<style>.b{{fill:{LIGHT['blue']}}}.a{{fill:{LIGHT['amber']}}}"
        f"@media (prefers-color-scheme:dark){{.b{{fill:{DARK['blue']}}}.a{{fill:{DARK['amber']}}}}}</style>"
    )
    return svg_open((x, y, s, s), "carto-flow") + style + mark_body(GAP_FAVICON, "", "", classes=True) + "</svg>\n"


# ---- Text outlines ----------------------------------------------------------------------------------


def find_font(name: str) -> str:
    import matplotlib

    return str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name)


def text_paths(
    text: str, font_file: str, size: float, x: float, baseline: float, colors: dict[int, str], default: str
) -> tuple[str, float]:
    """Outlined text: one <path> per glyph. Returns the markup and the end x. ``colors`` maps char index to fill."""
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen
    from fontTools.ttLib import TTFont

    font = TTFont(font_file)
    gs = font.getGlyphSet()
    cmap = font.getBestCmap()
    scale = size / font["head"].unitsPerEm
    out = ""
    for i, ch in enumerate(text):
        name = cmap[ord(ch)]
        if ch != " ":
            pen = SVGPathPen(gs, ntos=lambda v: f"{v:.2f}")
            gs[name].draw(TransformPen(pen, (scale, 0, 0, -scale, x, baseline)))
            out += f'<path fill="{colors.get(i, default)}" d="{pen.getCommands()}"/>'
        x += font["hmtx"][name][0] * scale
    return out, x


def wordmark_svg(pal: dict[str, str]) -> str:
    x0, y0, x1, y1 = mark_bounds(GAP_NORMAL)
    mh = y1 - y0
    size = 58.0
    text = "carto-flow"
    # Hyphen blue, the second "o" amber.
    glyphs, end = text_paths(
        text, find_font("DejaVuSans-Bold.ttf"), size, (x1 - x0) + 16, y0 + mh / 2 + 0.28 * size,
        {5: pal["blue"], 8: pal["amber"]}, pal["text"],
    )  # fmt: skip
    # Mark shifted to the origin.
    mark = f'<g transform="translate({-x0:.2f},{-y0:.2f})">{mark_body(GAP_NORMAL, pal["blue"], pal["amber"])}</g>'
    # Trailing advance of the last glyph is kept as right padding of 0; the viewBox ends at the text.
    return svg_open((0, 0, end, mh), "carto-flow") + mark + glyphs + "</svg>\n"


# ---- Contrast ---------------------------------------------------------------------------------------


def luminance(hex_color: str) -> float:
    c = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def contrast_report() -> str:
    rows = []
    for variant, pal, surfaces in (
        ("light", LIGHT, ("light page", "light header")),
        ("dark", DARK, ("dark page", "dark header")),
    ):
        for s in surfaces:
            for role in ("blue", "amber", "text"):
                rows.append(
                    f"{variant:5s} {role:5s} {pal[role]} on {s:12s} {SURFACES[s]}: {contrast(pal[role], SURFACES[s]):.2f}"
                )
    rows.append(f"light amber on {'dark page'} (wrong variant): {contrast(LIGHT['amber'], SURFACES['dark page']):.2f}")
    return "\n".join(rows)


# ---- Rasterizing ------------------------------------------------------------------------------------


def to_png(svg: str, path: Path, width: int, height: int | None = None, background: str | None = None) -> None:
    import cairosvg

    cairosvg.svg2png(
        bytestring=svg.encode(),
        write_to=str(path),
        output_width=width,
        output_height=height,
        background_color=background,
    )


def centered_png(svg_text: str, path: Path, canvas: int, mark_px: int, background: str) -> None:
    """Mark of ``mark_px`` pixels centered on an opaque ``canvas`` x ``canvas`` background."""
    import io

    import cairosvg
    from PIL import Image

    buf = io.BytesIO()
    cairosvg.svg2png(bytestring=svg_text.encode(), write_to=buf, output_width=mark_px, output_height=mark_px)
    im = Image.open(buf).convert("RGBA")
    base = Image.new("RGBA", (canvas, canvas), background)
    off = (canvas - mark_px) // 2
    base.alpha_composite(im, (off, off))
    base.convert("RGB").save(path)


def social_svg(pal: dict[str, str], bg: str) -> str:
    """1280 x 640 card: wordmark scaled up, tagline below, on a plain surface."""
    wm = wordmark_svg(pal)
    inner = wm[wm.index("</title>") + len("</title>") : -len("</svg>\n")]
    vb = [float(v) for v in wm.split('viewBox="')[1].split('"')[0].split()]
    scale = 780 / vb[2]
    wm_h = vb[3] * scale
    tag, tag_end = text_paths(TAGLINE, find_font("DejaVuSans.ttf"), 34, 0, 0, {}, pal["text"])
    top = (640 - (wm_h + 70 + 34)) / 2
    left = (1280 - 780) / 2
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 640" width="1280" height="640">'
        f'<rect width="1280" height="640" fill="{bg}"/>'
        f'<g transform="translate({left:.1f},{top:.1f}) scale({scale:.4f})">{inner}</g>'
        f'<g transform="translate({(1280 - tag_end) / 2:.1f},{top + wm_h + 70:.1f})" opacity="0.85">{tag}</g></svg>\n'
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("docs/assets/logo"))
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)

    for name, pal in (("light", LIGHT), ("dark", DARK)):
        (out / f"mark-{name}.svg").write_text(mark_svg(pal))
        (out / f"wordmark-{name}.svg").write_text(wordmark_svg(pal))
    (out / "mark-mono.svg").write_text(mark_svg(LIGHT, mono=True))
    (out / "mark-hero-light.svg").write_text(mark_svg(LIGHT, GAP_HERO))
    (out / "mark-hero-dark.svg").write_text(mark_svg(DARK, GAP_HERO))
    (out / "favicon.svg").write_text(favicon_svg())

    to_png(mark_svg(LIGHT), out / "favicon-32.png", 32, 32)
    centered_png(mark_svg(LIGHT), out / "apple-touch-icon-180.png", 180, 136, "#ffffff")
    social = social_svg(DARK, DARK_BG)
    to_png(social, out / "social-preview-1280x640.png", 1280, 640)
    print(contrast_report())


if __name__ == "__main__":
    main()
