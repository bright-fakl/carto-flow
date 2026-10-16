# ruff: noqa: RUF001
"""Figures of the explanation "Flow Cartogram Algorithm".

Draws ``flow-diffusion-idea.svg``, ``flow-electrostatic-analogy.png`` and ``flow-1d-mechanics.png`` with the colors of
the default plot style. The documentation build runs it through the hook ``docs/hooks/generate_diagrams.py`` and writes
the files to ``docs/generated/diagrams``, which is not committed. To draw them by hand:

    python docs/diagrams/make_flow_algorithm_figures.py [OUTPUT_DIRECTORY]

The figures are drawn with NumPy and matplotlib alone; they illustrate the algorithm and do not call the solver.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Rectangle

from carto_flow import PlotStyle

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "generated" / "diagrams"  # replaced by the argument of generate()
STYLE = PlotStyle.default()
SURFACE = STYLE.rc["figure.facecolor"]
TEXT = STYLE.rc["text.color"]
SPINE = STYLE.rc["axes.edgecolor"]
EDGE = STYLE.reference_edgecolor
SEQUENTIAL = STYLE.colormap("sequential")
DIVERGING = STYLE.colormap("diverging")
DARK = SEQUENTIAL(1.0)
LIGHT_TEXT = "white"


def _save(fig, name, **kwargs):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, facecolor=SURFACE, **kwargs)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# Diffusion idea: two regions before and after the flow
# ---------------------------------------------------------------------------------------------------------------


def diffusion_idea():
    def density_color(d):
        return SEQUENTIAL((np.log10(d) - np.log10(0.4)) / (np.log10(6) - np.log10(0.4)))

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    height, y0, unit = 3.2, 2.0, 2.0

    def pair(x0, wa, wb, da, db, label_a, label_b, tag_a, tag_b, note):
        ax.add_patch(Rectangle((x0, y0), wa, height, fc=density_color(da), ec=EDGE, lw=2))
        ax.add_patch(Rectangle((x0 + wa, y0), wb, height, fc=density_color(db), ec=EDGE, lw=2))
        ax.text(
            x0 + wa / 2,
            y0 + height / 2,
            label_a,
            ha="center",
            va="center",
            fontsize=11,
            color=LIGHT_TEXT,
            fontweight="bold",
        )
        ax.text(
            x0 + wa + wb / 2,
            y0 + height / 2,
            label_b,
            ha="center",
            va="center",
            fontsize=11,
            color=TEXT,
            fontweight="bold",
        )
        ax.text(x0 + wa / 2, y0 + height + 0.25, tag_a, ha="center", fontsize=11, color=TEXT)
        ax.text(x0 + wa + wb / 2, y0 + height + 0.25, tag_b, ha="center", fontsize=11, color=TEXT)
        ax.text(x0 + (wa + wb) / 2, y0 - 0.4, note, ha="center", va="top", fontsize=11, color=TEXT)

    # A: 4 people on 1 km2 (dense), B: 2 people on 3 km2 (sparse); at equal density 1.5 they have 2.67 and 1.33 km2
    pair(
        0.3,
        1.0 * unit,
        3.0 * unit,
        4.0,
        2 / 3,
        "A\n4 people\n1 km²",
        "B\n2 people\n3 km²",
        "dense",
        "sparse",
        "Before: density = people ÷ area",
    )
    ax.add_patch(
        FancyArrowPatch(
            (8.7, y0 + height / 2), (10.0, y0 + height / 2), arrowstyle="-|>", mutation_scale=22, lw=2.5, color=TEXT
        )
    )
    ax.text(9.35, y0 + height / 2 + 0.45, "flow", ha="center", fontsize=11, color=TEXT, style="italic")
    pair(
        10.7,
        2.667 * unit,
        1.333 * unit,
        1.5,
        1.5,
        "A\n4 people\n2.7 km²",
        "B\n2 people\n1.3 km²",
        "grows",
        "shrinks",
        "After: the same density everywhere",
    )
    _save(fig, "flow-diffusion-idea.svg")


# ---------------------------------------------------------------------------------------------------------------
# Electrostatic analogy: two opposite charges, the potential and the field
# ---------------------------------------------------------------------------------------------------------------


def electrostatic_analogy():
    half_w, half_h = 4.8, 2.7
    plus, minus = (-1.7, 0.0), (1.7, 0.0)
    soften = 0.12  # a finite-size blob of density instead of a point

    def potential(x, y):
        return sum(q / np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + soften) for q, (cx, cy) in ((1.0, plus), (-1.0, minus)))

    def field(x, y):
        ex = ey = 0.0
        for q, (cx, cy) in ((1.0, plus), (-1.0, minus)):
            d3 = ((x - cx) ** 2 + (y - cy) ** 2 + soften) ** 1.5
            ex = ex + q * (x - cx) / d3
            ey = ey + q * (y - cy) / d3
        return ex, ey

    nx = 480
    xs, ys = np.linspace(-half_w, half_w, nx), np.linspace(-half_h, half_h, int(nx * half_h / half_w))
    x, y = np.meshgrid(xs, ys)
    vmax = 1.6
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.axis("off")
    image = ax.imshow(
        np.clip(potential(x, y), -vmax, vmax),
        extent=[-half_w, half_w, -half_h, half_h],
        origin="lower",
        cmap=DIVERGING,
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
        interpolation="bilinear",
    )
    gx, gy = np.meshgrid(np.arange(-half_w + 0.3, half_w, 0.48), np.arange(-half_h + 0.26, half_h, 0.48))
    ex, ey = field(gx, gy)
    magnitude = np.hypot(ex, ey)
    length = 0.46 * (magnitude / magnitude.max()) ** 0.22
    keep = (np.hypot(gx - plus[0], gy - plus[1]) > 0.45) & (np.hypot(gx - minus[0], gy - minus[1]) > 0.45)
    ax.quiver(
        gx[keep],
        gy[keep],
        (ex / magnitude * length)[keep],
        (ey / magnitude * length)[keep],
        color=TEXT,
        edgecolor="white",
        linewidth=0.6,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        pivot="middle",
        width=0.0026,
        headwidth=4.2,
        headlength=4.4,
        headaxislength=3.8,
        alpha=0.85,
    )
    for (cx, cy), symbol in ((plus, "+"), (minus, "−")):
        ax.scatter([cx], [cy], s=1500, facecolor="none", edgecolor="white", linewidth=3, zorder=5)
        ax.text(
            cx, cy - 0.04, symbol, color="white", ha="center", va="center", fontsize=30, fontweight="bold", zorder=6
        )
    ax.set_xlim(-half_w, half_w)
    ax.set_ylim(-half_h, half_h)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, ticks=[-vmax, 0, vmax])
    colorbar.ax.set_yticklabels(["−", "0", "+"])
    colorbar.outline.set_edgecolor(SPINE)
    colorbar.set_label("potential", color=TEXT)
    colorbar.ax.tick_params(colors=TEXT)
    _save(fig, "flow-electrostatic-analogy.png", dpi=110, bbox_inches="tight")


# ---------------------------------------------------------------------------------------------------------------
# One-dimensional mechanics: the loop of the algorithm on three regions
# ---------------------------------------------------------------------------------------------------------------


def mechanics_1d():
    length, n_regions, pad, cells = 10.0, 3, 3.0, 640
    dt, tol = 0.2, 0.005
    x = np.array([0.0, 4.0, 7.0, 10.0])  # region boundaries
    mass = np.array([2.0, 6.0, 1.0]) * np.diff(x)  # value = density x length
    rho_bar = mass.sum() / length
    edges = np.linspace(-pad, length + pad, cells + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    dx = centers[1] - centers[0]
    k = 2 * np.pi * np.fft.fftfreq(cells, dx)

    def state(xn):
        """Density per region, and the solver quantities on the grid, for the boundaries xn."""
        rho_regions = mass / np.diff(xn)
        cumulative = np.concatenate([[0.0], np.cumsum(mass)])
        c = np.interp(edges, xn, cumulative)  # exact area-weighted rasterization, target density outside the map
        c = np.where(edges < xn[0], -rho_bar * (xn[0] - edges), c)
        c = np.where(edges > xn[-1], cumulative[-1] + rho_bar * (edges - xn[-1]), c)
        rho_grid = np.diff(c) / dx
        rho_hat = np.fft.fft(rho_grid - rho_grid.mean())
        k2 = k**2
        k2[0] = 1.0
        phi_hat = -rho_hat / k2
        phi_hat[0] = 0.0
        return rho_regions, rho_grid, np.real(np.fft.ifft(1j * k * phi_hat))  # v = phi'

    v_first = np.abs(state(x)[2]).max()
    frames, xn = [], x.copy()
    for iteration in range(6000):
        rho_regions, rho_grid, v = state(xn)
        frames.append((iteration, xn.copy(), rho_regions, rho_grid, v))
        if np.abs(np.log2(rho_regions / rho_bar)).max() < np.log2(1 + tol):
            break
        xn = xn + np.interp(xn, centers, v) * dt * dx / v_first
    else:
        raise AssertionError("the one-dimensional example did not converge")
    chosen = [frames[0], frames[len(frames) // 8], frames[-1]]  # start, a snapshot on the way, the end

    rho_max = max(f[2].max() for f in frames)
    rho_min = min(f[2].min() for f in frames)
    v_max = max(np.abs(f[4]).max() for f in frames)
    norm = LogNorm(vmin=rho_min * 0.9, vmax=rho_max)
    fig, axes = plt.subplots(
        4,
        len(chosen),
        figsize=(11, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [0.45, 0.45, 1.6, 1.2], "hspace": 0.9},
    )
    rows = ["Original", "Current", "Density ρ = value / length", "Velocity v"]
    for column, (iteration, xn, rho_regions, _, v) in enumerate(chosen):
        for row, ax in enumerate(axes[:, column]):
            ax.set_facecolor(SURFACE)
            for spine in ax.spines.values():
                spine.set_edgecolor(SPINE)
            ax.set_xlim(-pad, length + pad)
            ax.set_yticks([])
            ax.tick_params(colors=TEXT, labelsize=9)
            if row < 3:
                ax.tick_params(labelbottom=False)
            if column == 0:
                ax.set_title(rows[row], loc="left", fontsize=10, color=TEXT, pad=3)
        axes[0, column].set_ylim(0, 1)
        axes[1, column].set_ylim(0, 1)
        axes[2, column].set_ylim(0, rho_max * 1.12)
        axes[3, column].set_ylim(-1.15, 1.15)
        original_density = mass / np.diff(x)
        for j in range(n_regions):
            axes[0, column].axvspan(x[j], x[j + 1], color=SEQUENTIAL(norm(original_density[j])), lw=0)
            axes[1, column].axvspan(xn[j], xn[j + 1], color=SEQUENTIAL(norm(rho_regions[j])), lw=0)
        axes[0, column].vlines(x, 0, 1, color=TEXT, lw=1.5)
        axes[1, column].vlines(xn, 0, 1, color=TEXT, lw=1.5)
        for j in range(n_regions + 1):
            axes[1, column].figure.add_artist(
                ConnectionPatch(
                    (x[j], 0),
                    (xn[j], 1),
                    "data",
                    "data",
                    axesA=axes[0, column],
                    axesB=axes[1, column],
                    color=SPINE,
                    lw=1.0,
                    ls="--",
                )
            )
        ax = axes[2, column]
        ax.axhline(rho_bar, color=TEXT, ls="--", lw=1)
        ax.stairs(rho_regions, xn, color=DARK, lw=2.5)
        ax.stairs(rho_regions, xn, fill=True, color=DARK, alpha=0.18)
        ax.hlines(rho_bar, -pad, xn[0], color=DARK, lw=2.5)
        ax.hlines(rho_bar, xn[-1], length + pad, color=DARK, lw=2.5)
        if column == 0:
            ax.text(-pad + 0.1, rho_bar, "target density ρ̄", va="bottom", fontsize=9, color=TEXT)
        else:
            ax.stairs(mass / np.diff(x), x, color=SPINE, lw=1.2, ls=":")
        ax = axes[3, column]
        ax.axhline(0, color=SPINE, lw=0.8)
        ax.plot(centers, v / v_max, color=DIVERGING(0.12), lw=2)
        ax.set_xlabel(f"iteration {iteration}", fontsize=11, color=TEXT)
    _save(fig, "flow-1d-mechanics.png", dpi=100, bbox_inches="tight")


def generate(out_dir=None):
    """Draw the three figures into ``out_dir`` (default ``docs/generated/diagrams``)."""
    global OUT
    if out_dir is not None:
        OUT = Path(out_dir)
    diffusion_idea()
    electrostatic_analogy()
    mechanics_1d()


if __name__ == "__main__":
    import sys

    generate(sys.argv[1] if len(sys.argv) > 1 else None)
