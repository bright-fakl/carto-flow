"""Figures of the explanation "Output Validity".

Draws the figures of ``docs/explanations/flow-cartogram-validity.md`` with the colors of the default plot style. The
documentation build runs it through the hook ``docs/hooks/generate_diagrams.py`` and writes the files to
``docs/generated/diagrams``, which is not committed. To draw them by hand:

    python docs/diagrams/make_validity_figures.py [OUTPUT_DIRECTORY]

The figures are drawn with NumPy and matplotlib alone. They apply the steps of the algorithms to a small analytic
velocity field and small examples; they illustrate the algorithms and do not call the library.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle

from carto_flow import PlotStyle

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "generated" / "diagrams"  # replaced by the argument of generate()
STYLE = PlotStyle.default()
SURFACE = STYLE.rc["figure.facecolor"]
TEXT = STYLE.rc["text.color"]
SPINE = STYLE.rc["axes.edgecolor"]
CATEGORICAL = STYLE.palettes.categorical

# One color for one thing in all figures of the page.
INPUT = CATEGORICAL[0]  # vertices of the input
INSERTED = CATEGORICAL[1]  # inserted vertices
PATH = STYLE.positive_color  # the curve a boundary follows under the flow
CHORD = "#3b4650"  # the straight segment between two vertices
CROSSING = STYLE.accent_color  # crossings
REGION = STYLE.reference_facecolor  # regions that are not the subject of the figure
STRIP = STYLE.colormap("sequential")(0.12)  # the region that is squeezed

LABEL = 10
SMALL = 9


def _save(fig, name, **kwargs):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, facecolor=SURFACE, **kwargs)
    plt.close(fig)


def _axes(ax, xlim, ylim, title=None, aspect=1.0):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(aspect)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, color=TEXT, loc="left", pad=4)


def _legend(fig, entries, ncol, y=0.0):
    handles = []
    for kind, color, label in entries:
        if kind == "dot":
            handles.append(
                Line2D([], [], marker="o", ls="", color=color, markeredgecolor=SURFACE, markersize=8, label=label)
            )
        elif kind == "hollow":
            handles.append(
                Line2D(
                    [], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=color, markersize=8, label=label
                )
            )
        elif kind == "cross":
            handles.append(
                Line2D([], [], marker="X", ls="", color=color, markeredgecolor=SURFACE, markersize=9, label=label)
            )
        elif kind == "dashed":
            handles.append(Line2D([], [], color=color, lw=2, ls=(0, (4, 2)), label=label))
        elif kind == "patch":
            handles.append(Patch(facecolor=color, edgecolor="none", label=label))
        else:
            handles.append(Line2D([], [], color=color, lw=2.5, label=label))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=ncol,
        frameon=False,
        fontsize=LABEL,
        labelcolor=TEXT,
        bbox_to_anchor=(0.5, y),
        columnspacing=1.6,
        handlelength=1.8,
    )


# ---------------------------------------------------------------------------------------------------------------
# A small velocity field and the segments of a squeezed strip
# ---------------------------------------------------------------------------------------------------------------

STRIP_HALF = 0.5  # the strip lies between y = -0.5 and y = 0.5
BUMP_WIDTH = 1.6
STRETCH = 1.7  # the strip figures stretch the vertical axis, to make the strip visible


def _bump(x):
    return np.exp(-((x / BUMP_WIDTH) ** 2))


def velocity(p):
    """A smooth compression of the strip toward its upper boundary, with a small shear, near x = 0."""
    x, y = p[..., 0], p[..., 1]
    vy = 2.6 * _bump(x) * (1 - 0.7 * (y + STRIP_HALF))
    vx = 0.35 * _bump(x) * (y + STRIP_HALF)
    return np.stack([vx, vy], axis=-1)


def advect(points, duration=1.0, steps=40):
    """Move points through the fixed field (fourth-order Runge-Kutta)."""
    p = np.array(points, float)
    h = duration / steps
    for _ in range(steps):
        k1 = velocity(p)
        k2 = velocity(p + 0.5 * h * k1)
        k3 = velocity(p + 0.5 * h * k2)
        k4 = velocity(p + h * k3)
        p = p + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return p


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_cross(a, b, p, q):
    """True if the segments ab and pq cross (strictly opposite orientations on both sides)."""
    return _orient(a, b, p) * _orient(a, b, q) < 0 and _orient(p, q, a) * _orient(p, q, b) < 0


def crossing_point(a, b, p, q):
    d1, d2 = b - a, q - p
    t = ((p[0] - a[0]) * d2[1] - (p[1] - a[1]) * d2[0]) / (d1[0] * d2[1] - d1[1] * d2[0])
    return a + t * d1


def crossing_segments(rings):
    """(ring, segment) of every segment that crosses a segment of any ring."""
    flagged = []
    for i, ring in enumerate(rings):
        for k in range(len(ring) - 1):
            if any(
                segments_cross(ring[k], ring[k + 1], other[j], other[j + 1])
                for other in rings
                for j in range(len(other) - 1)
            ):
                flagged.append((i, k))
    return flagged


def crossing_points(rings):
    points = []
    for i, ring in enumerate(rings):
        for k in range(len(ring) - 1):
            for m in range(i, len(rings)):
                other = rings[m]
                for j in range(k + 1 if m == i else 0, len(other) - 1):
                    if segments_cross(ring[k], ring[k + 1], other[j], other[j + 1]):
                        points.append(crossing_point(ring[k], ring[k + 1], other[j], other[j + 1]))
    return np.array(points)


def split_flagged(rings, flagged):
    """Insert the midpoint of the chord of every flagged segment; also returns the inserted points per ring."""
    new_rings, inserted = [], []
    for i, ring in enumerate(rings):
        points, added = [ring[0]], []
        for k in range(len(ring) - 1):
            if (i, k) in flagged:
                midpoint = 0.5 * (ring[k] + ring[k + 1])
                points.append(midpoint)
                added.append(len(points) - 1)
            points.append(ring[k + 1])
        new_rings.append(np.array(points))
        inserted.append(added)
    return new_rings, inserted


def strip_scene():
    """Start and end of the boundaries of the strip: a coarse upper boundary and a finer lower boundary."""
    upper = np.array([[-3.0, STRIP_HALF], [3.0, STRIP_HALF]])
    lower = np.column_stack([np.arange(-3.0, 3.1), np.full(7, -STRIP_HALF)])
    return [upper, lower]


def true_boundaries(rings, n=60):
    """The curves of the boundaries after the flow: the points of each segment moved through the field."""
    paths = []
    for ring in rings:
        pieces = [
            advect(ring[k] + np.linspace(0, 1, n)[:, None] * (ring[k + 1] - ring[k])) for k in range(len(ring) - 1)
        ]
        paths.append(np.vstack(pieces))
    return paths


def _fill_regions(ax, upper_path, lower_path, xlim, ylim):
    """Region above the upper boundary, the strip, and region below the lower boundary, as polygons."""
    upper_path = np.vstack([[xlim[0], upper_path[0, 1]], upper_path, [xlim[1], upper_path[-1, 1]]])
    lower_path = np.vstack([[xlim[0], lower_path[0, 1]], lower_path, [xlim[1], lower_path[-1, 1]]])
    top = [(xlim[1], ylim[1]), (xlim[0], ylim[1])]
    bottom = [(xlim[1], ylim[0]), (xlim[0], ylim[0])]
    ax.add_patch(Polygon(np.vstack([upper_path, top]), fc=REGION, ec="none", zorder=0))
    ax.add_patch(Polygon(np.vstack([lower_path, bottom]), fc=REGION, ec="none", zorder=0))
    ax.add_patch(Polygon(np.vstack([upper_path, lower_path[::-1]]), fc=STRIP, ec="none", zorder=0))


def _polyline(ax, ring, **kwargs):
    ax.plot(ring[:, 0], ring[:, 1], **kwargs)


# ---------------------------------------------------------------------------------------------------------------
# Figure 1: the chord differs from the curve of the boundary
# ---------------------------------------------------------------------------------------------------------------


def chord_and_path():
    xlim, ylim = (-4.0, 4.0), (-0.9, 1.6)
    start = strip_scene()
    end = [advect(r) for r in start]
    paths = true_boundaries(start)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))

    ax = axes[0]
    _axes(ax, xlim, ylim, "Before the flow", STRETCH)
    _fill_regions(
        ax, np.array([[xlim[0], 0.5], [xlim[1], 0.5]]), np.array([[xlim[0], -0.5], [xlim[1], -0.5]]), xlim, ylim
    )
    for ring in start:
        _polyline(ax, ring, color=CHORD, lw=1.8, zorder=2)
        ax.scatter(ring[:, 0], ring[:, 1], s=46, color=INPUT, edgecolor=SURFACE, zorder=4)
    ax.text(-3.9, 1.3, "region A", fontsize=LABEL, color=TEXT)
    ax.text(-3.9, -0.12, "squeezed region", fontsize=LABEL, color=TEXT)
    ax.text(-3.9, -0.8, "region B", fontsize=LABEL, color=TEXT)
    ax.annotate(
        "one long segment",
        (0.0, 0.5),
        (0.0, 1.2),
        ha="center",
        fontsize=LABEL,
        color=TEXT,
        arrowprops={"arrowstyle": "-", "color": SPINE, "lw": 1},
    )

    ax = axes[1]
    _axes(ax, xlim, ylim, "After the flow", STRETCH)
    _fill_regions(ax, paths[0], paths[1], xlim, ylim)
    ax.plot(paths[0][:, 0], paths[0][:, 1], color=PATH, lw=2.4, zorder=3)
    ax.plot(paths[1][:, 0], paths[1][:, 1], color=PATH, lw=1.6, zorder=3)
    _polyline(ax, end[0], color=CHORD, lw=1.8, ls=(0, (4, 2)), zorder=3)
    _polyline(ax, end[1], color=CHORD, lw=1.8, zorder=3)
    for ring in end:
        ax.scatter(ring[:, 0], ring[:, 1], s=46, color=INPUT, edgecolor=SURFACE, zorder=5)
    crossings = crossing_points(end)
    ax.scatter(crossings[:, 0], crossings[:, 1], s=110, marker="X", color=CROSSING, edgecolor=SURFACE, zorder=6)
    # the sagitta of the upper segment at its middle
    k = len(paths[0]) // 2
    x_mid = paths[0][k, 0]
    y_chord = np.interp(x_mid, end[0][:, 0], end[0][:, 1])
    ax.annotate(
        "",
        (x_mid, paths[0][k, 1]),
        (x_mid, y_chord),
        arrowprops={"arrowstyle": "<->", "color": TEXT, "lw": 1.2},
        zorder=6,
    )
    ax.annotate(
        "sagitta\n" + r"$\approx \kappa\,\ell^{2}/8$",
        (x_mid, paths[0][k, 1]),
        (x_mid + 0.7, 1.2),
        fontsize=LABEL,
        color=TEXT,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": SPINE, "lw": 1},
    )
    ax.annotate(
        "crossing",
        crossings[0],
        (-3.3, 1.2),
        fontsize=LABEL,
        color=TEXT,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": SPINE, "lw": 1},
        zorder=7,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.2, wspace=0.04)
    _legend(
        fig,
        [
            ("dot", INPUT, "vertex"),
            ("line", CHORD, "chord between the moved vertices"),
            ("line", PATH, "boundary after the flow (curve)"),
            ("cross", CROSSING, "crossing"),
        ],
        4,
    )
    _save(fig, "validity-chord-and-path.svg")


# ---------------------------------------------------------------------------------------------------------------
# Figure 2: split the crossing segments and replay the new vertices
# ---------------------------------------------------------------------------------------------------------------


def split_and_replay():
    xlim, ylim = (-4.0, 4.0), (-1.1, 1.8)
    start = strip_scene()
    end = [advect(r) for r in start]
    flagged = crossing_segments(end)
    start_split, inserted = split_flagged(start, flagged)
    end_split = [advect(r) for r in start_split]
    if crossing_segments(end_split):
        raise AssertionError("one round must remove the crossings of the example")
    paths = true_boundaries(start)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    titles = ["1  Start of the span", "2  End of the span", "3  After the split"]
    for ax, title in zip(axes, titles, strict=True):
        _axes(ax, xlim, ylim, title, STRETCH)

    # panel 1: the chords at the start and the midpoints that will be inserted
    ax = axes[0]
    _fill_regions(
        ax, np.array([[xlim[0], 0.5], [xlim[1], 0.5]]), np.array([[xlim[0], -0.5], [xlim[1], -0.5]]), xlim, ylim
    )
    for ring in start:
        _polyline(ax, ring, color=CHORD, lw=1.8, zorder=2)
        ax.scatter(ring[:, 0], ring[:, 1], s=40, color=INPUT, edgecolor=SURFACE, zorder=4)
    for ring_split, added in zip(start_split, inserted, strict=True):
        ax.scatter(ring_split[added, 0], ring_split[added, 1], s=60, color=INSERTED, edgecolor=SURFACE, zorder=5)

    # panel 2: the end of the span, with crossings
    ax = axes[1]
    _fill_regions(ax, paths[0], paths[1], xlim, ylim)
    for path in paths:
        ax.plot(path[:, 0], path[:, 1], color=PATH, lw=1.8, zorder=3)
    for ring in end:
        _polyline(ax, ring, color=CHORD, lw=1.6, zorder=2)
        ax.scatter(ring[:, 0], ring[:, 1], s=40, color=INPUT, edgecolor=SURFACE, zorder=4)
    crossings = crossing_points(end)
    ax.scatter(crossings[:, 0], crossings[:, 1], s=100, marker="X", color=CROSSING, edgecolor=SURFACE, zorder=6)

    # panel 3: the new vertices on the paths
    ax = axes[2]
    _fill_regions(ax, paths[0], paths[1], xlim, ylim)
    for path in paths:
        ax.plot(path[:, 0], path[:, 1], color=PATH, lw=1.8, zorder=3)
    for ring, added in zip(end_split, inserted, strict=True):
        _polyline(ax, ring, color=CHORD, lw=1.6, zorder=2)
        is_new = np.zeros(len(ring), bool)
        is_new[added] = True
        ax.scatter(ring[~is_new, 0], ring[~is_new, 1], s=40, color=INPUT, edgecolor=SURFACE, zorder=4)
        ax.scatter(ring[is_new, 0], ring[is_new, 1], s=60, color=INSERTED, edgecolor=SURFACE, zorder=5)
    # the midpoint of the chord at the end of the span, for comparison with the replayed vertex
    upper_added = inserted[0][0]
    midpoint_end = 0.5 * (end[0][0] + end[0][1])
    ax.scatter(*midpoint_end, s=60, facecolor=SURFACE, edgecolor=CHORD, zorder=5)
    ax.annotate(
        "midpoint of the chord\nat the end",
        midpoint_end,
        (-3.9, -0.95),
        fontsize=SMALL,
        color=TEXT,
        arrowprops={"arrowstyle": "-", "color": SPINE, "lw": 1},
    )
    ax.annotate(
        "replayed vertex",
        end_split[0][upper_added],
        (0.9, 1.5),
        fontsize=SMALL,
        color=TEXT,
        arrowprops={"arrowstyle": "-", "color": SPINE, "lw": 1},
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.22, wspace=0.05)
    _legend(
        fig,
        [
            ("dot", INPUT, "input vertex"),
            ("dot", INSERTED, "inserted vertex"),
            ("hollow", CHORD, "midpoint of the chord at the end"),
            ("line", CHORD, "chord"),
            ("line", PATH, "boundary after the flow (curve)"),
            ("cross", CROSSING, "crossing"),
        ],
        6,
    )
    _save(fig, "validity-split-and-replay.svg")


# ---------------------------------------------------------------------------------------------------------------
# Figure 3: windows, spans, stored fields, memory budget and cap
# ---------------------------------------------------------------------------------------------------------------


def spans_and_memory():
    every, budget, n_windows = 5, 3.5, 11  # windows per span, budget in fields, windows of the run
    stored, spans, checks = 0, [[]], []
    heights = []
    for w in range(1, n_windows + 1):
        if stored >= budget or len(spans[-1]) >= every:  # the rule of the start of a window
            checks.append(w - 0.5)
            spans.append([])
            stored = 0
        spans[-1].append(w)
        stored += 1
        heights.append(stored)
    checks.append(n_windows + 0.5)  # the check at the end of the run

    fig = plt.figure(figsize=(11, 7.4))
    grid = fig.add_gridspec(
        3, 1, height_ratios=[1.0, 1.5, 1.5], hspace=0.55, left=0.08, right=0.97, top=0.96, bottom=0.07
    )
    shade = [STYLE.colormap("sequential")(0.12), REGION]

    # (a) windows grouped in spans, with the checks
    ax = fig.add_subplot(grid[0])
    ax.set_xlim(0.4, n_windows + 0.6)
    ax.set_ylim(0, 3.0)
    ax.axis("off")
    for s, span in enumerate(spans):
        ax.add_patch(Rectangle((span[0] - 0.5, 0.2), len(span), 1.0, fc=shade[s % 2], ec=SPINE, lw=1))
        for w in span:
            ax.add_patch(Rectangle((w - 0.42, 0.3), 0.84, 0.5, fc=SURFACE, ec=SPINE, lw=0.8))
            ax.text(w, 0.55, f"{w}", ha="center", va="center", fontsize=SMALL, color=TEXT)
            ax.text(w, 1.0, "field", ha="center", va="center", fontsize=7, color=TEXT)
        ax.text(span[0] - 0.4, 1.45, f"span {s + 1}", fontsize=LABEL, color=TEXT, va="bottom")
    for c in checks:
        ax.plot([c, c], [0.05, 1.4], color=CROSSING, lw=2.5)
    ax.text(
        checks[0],
        2.35,
        "crossing test and repair;\nthe stored fields are released",
        ha="center",
        fontsize=SMALL,
        color=TEXT,
        va="center",
    )
    ax.annotate("", (checks[0], 1.5), (checks[0], 2.05), arrowprops={"arrowstyle": "-", "color": SPINE, "lw": 1})
    ax.annotate("", (checks[-1], 1.5), (checks[-1], 2.05), arrowprops={"arrowstyle": "-", "color": SPINE, "lw": 1})
    ax.set_xlim(0.4, n_windows + 0.6)
    ax.text(
        checks[-1] + 0.45,
        2.35,
        "end of the run: the current\nstate and the best iterate",
        ha="right",
        fontsize=SMALL,
        color=TEXT,
        va="center",
    )
    ax.set_title("Windows (velocity field recomputed at each) grouped in spans", fontsize=11, color=TEXT, loc="left")

    # (b) stored fields
    ax = fig.add_subplot(grid[1], sharex=None)
    ax.set_xlim(0.4, n_windows + 0.6)
    for w, height in enumerate(heights, start=1):
        for level in range(height):
            ax.add_patch(Rectangle((w - 0.4, level + 0.05), 0.8, 0.9, fc=INPUT, ec=SURFACE, lw=0.8, alpha=0.85))
    ax.axhline(budget, color=CROSSING, lw=1.6, ls=(0, (4, 2)))
    ax.axhline(every, color=SPINE, lw=1.6, ls=(0, (1, 2)))
    ax.text(n_windows + 0.55, budget - 0.12, "validity_memory_mb", ha="right", va="top", fontsize=SMALL, color=CROSSING)
    ax.text(
        n_windows + 0.55,
        every + 0.1,
        "validity_repair_every windows",
        ha="right",
        va="bottom",
        fontsize=SMALL,
        color=TEXT,
    )
    for c in checks[:-1]:
        ax.axvline(c, color=CROSSING, lw=1.0, alpha=0.5)
    ax.set_ylim(0, every + 1.1)
    ax.set_yticks([])
    ax.set_xticks(range(1, n_windows + 1))
    ax.tick_params(colors=TEXT, labelsize=SMALL)
    ax.set_xlabel("window", fontsize=LABEL, color=TEXT)
    ax.set_ylabel("stored fields", fontsize=LABEL, color=TEXT)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_edgecolor(SPINE)
    ax.set_title("Stored fields at the end of each window", fontsize=11, color=TEXT, loc="left")
    ax.text(
        0.5 * (checks[0] + checks[1]) if len(checks) > 2 else checks[0] + 2,
        budget + 0.45,
        "the stored fields have reached the budget:\nthe span ends before the fifth window",
        ha="center",
        fontsize=SMALL,
        color=TEXT,
        va="bottom",
    )

    # (c) the cap on inserted vertices
    ax = fig.add_subplot(grid[2])
    added = [14, 17, 21, 24]  # toy numbers of vertices inserted by the checks
    totals = np.cumsum(added)
    cap = 100.0
    attempt = totals[-1] + 24
    xs = np.arange(1, len(added) + 1)
    ax.bar(xs, totals, width=0.6, color=INSERTED, alpha=0.9)
    ax.bar([len(added) + 1], [attempt], width=0.6, fc="none", ec=INSERTED, ls=(0, (3, 2)), lw=1.6)
    ax.axhline(cap, color=CROSSING, lw=1.6, ls=(0, (4, 2)))
    ax.text(0.45, cap + 3, "cap on inserted vertices: max(5000, n // 100)", fontsize=SMALL, color=CROSSING, va="bottom")
    ax.text(
        len(added) + 1.45,
        attempt * 0.5,
        "this check would exceed the cap:\nit stops before the round and warns.\nNo further field is stored and no\nfurther check runs; regions that are\ninvalid at the end are listed in\nCartogram.invalid_regions.",
        fontsize=SMALL,
        color=TEXT,
        va="center",
    )
    ax.set_xlim(0.4, len(added) + 5.2)
    ax.set_ylim(0, cap * 1.3)
    ax.set_xticks(range(1, len(added) + 2))
    ax.set_yticks([])
    ax.tick_params(colors=TEXT, labelsize=SMALL)
    ax.set_xlabel("check", fontsize=LABEL, color=TEXT)
    ax.set_ylabel("inserted vertices\nin the run", fontsize=LABEL, color=TEXT)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_edgecolor(SPINE)
    ax.set_title("A run in which the cap is reached (schematic)", fontsize=11, color=TEXT, loc="left")
    _save(fig, "validity-spans-and-memory.svg")


# ---------------------------------------------------------------------------------------------------------------
# Figure 7: lifting a state to the final vertex set, and aligning two outputs
# ---------------------------------------------------------------------------------------------------------------


def _lift(values, rounds):
    """The lift of ``VertexLineage.lift``: rounds are lists of (left index, fraction) in the vertex set before."""
    for round_ in rounds:
        n_new = len(round_)
        left = np.array([r[0] for r in round_])
        fraction = np.array([r[1] for r in round_])
        out = np.empty((len(values) + n_new, values.shape[1]))
        is_new = np.zeros(len(out), bool)
        # the k-th inserted vertex follows the vertex ``left[k]`` of the set before the round
        new_index = left + 1 + np.arange(n_new)
        is_new[new_index] = True
        out[~is_new] = values
        out[new_index] = (1 - fraction)[:, None] * values[left] + fraction[:, None] * values[left + 1]
        values = out
    return values


def _field_map(points):
    """A smooth map of the plane that stands for the flow between the input and the final state."""
    x, y = points[:, 0], points[:, 1]
    return np.column_stack([
        x + 0.25 * np.sin(1.1 * y + 0.6 * x),
        0.55 * y + 0.5 * np.exp(-((x - 2.6) ** 2) / 1.3) + 0.04 * x**2,
    ])


def lift_state():
    ring = np.array([[0.0, 0.0], [1.0, 0.7], [2.0, 0.2], [3.0, 1.3], [4.0, 0.8], [5.0, 1.8]])
    rounds = [[(1, 0.5), (3, 0.5)], [(2, 0.5)]]  # second round splits the segment between an old and a new vertex
    # vertex sets: input (6) -> 8 -> 9; the fraction positions along the input ring follow from the lineage
    lifted = _lift(ring, rounds)
    is_input = np.zeros(len(lifted), bool)
    positions = _lift(np.arange(len(ring), dtype=float)[:, None], rounds)[:, 0]
    is_input[np.isclose(positions, np.round(positions))] = True
    final = _field_map(lifted)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 2.7))
    xlim, ylim = (-0.5, 5.7), (-0.9, 2.3)
    for ax in axes:
        _axes(ax, xlim, ylim)
    axes[0].set_title("State to lift: 6 vertices", fontsize=11, color=TEXT, loc="left", pad=4)
    axes[1].set_title("Lifted: 9 vertices, same polyline", fontsize=11, color=TEXT, loc="left", pad=4)
    axes[2].set_title("Final state: 9 vertices", fontsize=11, color=TEXT, loc="left", pad=4)
    axes[0].plot(ring[:, 0], ring[:, 1], color=CHORD, lw=1.8, zorder=2)
    axes[0].scatter(ring[:, 0], ring[:, 1], s=50, color=INPUT, edgecolor=SURFACE, zorder=4)
    axes[1].plot(lifted[:, 0], lifted[:, 1], color=CHORD, lw=1.8, zorder=2)
    axes[1].scatter(lifted[is_input, 0], lifted[is_input, 1], s=50, color=INPUT, edgecolor=SURFACE, zorder=4)
    axes[1].scatter(lifted[~is_input, 0], lifted[~is_input, 1], s=65, color=INSERTED, edgecolor=SURFACE, zorder=5)
    axes[2].plot(final[:, 0], final[:, 1], color=CHORD, lw=1.8, zorder=2)
    axes[2].scatter(final[is_input, 0], final[is_input, 1], s=50, color=INPUT, edgecolor=SURFACE, zorder=4)
    axes[2].scatter(final[~is_input, 0], final[~is_input, 1], s=65, color=INSERTED, edgecolor=SURFACE, zorder=5)
    # the lifted state and the final state have equal vertex sets: connect the pairs
    for p, q in zip(lifted, final, strict=True):
        axes[2].plot([p[0], q[0]], [p[1], q[1]], color=SPINE, lw=0.8, ls=":", zorder=1)
    axes[2].plot(lifted[:, 0], lifted[:, 1], color=SPINE, lw=0.8, zorder=1)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.2, wspace=0.04)
    _legend(
        fig,
        [
            ("dot", INPUT, "vertex of the state"),
            ("dot", INSERTED, "inserted vertex, at fraction 0.5 between its parents"),
            ("line", SPINE, "linear interpolation path (right)"),
        ],
        3,
    )
    _save(fig, "validity-lift.svg")


def align():
    ring = np.array([[0.0, 0.0], [1.0, 0.8], [2.0, 0.3], [3.0, 1.3], [4.0, 0.9], [5.0, 1.7]])

    def output(positions, shift):
        """An output: the input ring moved, with vertices inserted at the given positions along the input."""
        index = np.arange(len(ring), dtype=float)
        all_positions = np.sort(np.concatenate([index, positions]))
        points = np.column_stack([
            np.interp(all_positions, index, ring[:, 0]),
            np.interp(all_positions, index, ring[:, 1]),
        ])
        moved = _field_map(points + shift)
        return all_positions, moved

    pos_a, out_a = output([1.5, 3.5], np.array([0.0, 0.0]))
    pos_b, out_b = output([2.5, 3.5], np.array([0.2, -0.3]))
    union = np.unique(np.concatenate([pos_a, pos_b]))
    aligned = []
    for pos, out in ((pos_a, out_a), (pos_b, out_b)):
        aligned.append(np.column_stack([np.interp(union, pos, out[:, 0]), np.interp(union, pos, out[:, 1])]))

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.4))
    xlim, ylim = (-0.6, 5.9), (-0.9, 2.3)
    for ax in axes.ravel():
        _axes(ax, xlim, ylim)
    names = ["output a", "output b"]
    for column, (pos, out, al, name) in enumerate(zip((pos_a, pos_b), (out_a, out_b), aligned, names, strict=True)):
        ax = axes[0, column]
        ax.set_title(f"{name}: own vertices", fontsize=11, color=TEXT, loc="left", pad=4)
        is_input = np.isclose(pos, np.round(pos))
        ax.plot(out[:, 0], out[:, 1], color=CHORD, lw=1.8, zorder=2)
        ax.scatter(out[is_input, 0], out[is_input, 1], s=50, color=INPUT, edgecolor=SURFACE, zorder=4)
        ax.scatter(out[~is_input, 0], out[~is_input, 1], s=65, color=INSERTED, edgecolor=SURFACE, zorder=5)
        ax = axes[1, column]
        ax.set_title(f"{name}: after align_vertices", fontsize=11, color=TEXT, loc="left", pad=4)
        ax.plot(al[:, 0], al[:, 1], color=CHORD, lw=1.8, zorder=2)
        ax.scatter(
            al[np.isclose(union, np.round(union)), 0],
            al[np.isclose(union, np.round(union)), 1],
            s=50,
            color=INPUT,
            edgecolor=SURFACE,
            zorder=4,
        )
        own = np.isin(union, pos) & ~np.isclose(union, np.round(union))
        added = ~np.isin(union, pos)
        ax.scatter(al[own, 0], al[own, 1], s=65, color=INSERTED, edgecolor=SURFACE, zorder=5)
        ax.scatter(al[added, 0], al[added, 1], s=65, facecolor=SURFACE, edgecolor=INSERTED, linewidth=1.8, zorder=5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.12, wspace=0.04, hspace=0.3)
    _legend(
        fig,
        [
            ("dot", INPUT, "input vertex"),
            ("dot", INSERTED, "inserted vertex"),
            ("hollow", INSERTED, "vertex of the other output, added on a segment"),
        ],
        3,
    )
    _save(fig, "validity-align.svg")


def generate(out_dir=None):
    """Draw the figures into ``out_dir`` (default ``docs/generated/diagrams``)."""
    global OUT
    if out_dir is not None:
        OUT = Path(out_dir)
    chord_and_path()
    split_and_replay()
    spans_and_memory()
    lift_state()
    align()


if __name__ == "__main__":
    import sys

    generate(sys.argv[1] if len(sys.argv) > 1 else None)
