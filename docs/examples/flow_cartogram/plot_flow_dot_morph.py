"""
Dot Density Map into a Flow Cartogram
=====================================

A dot density map of poverty morphs into a flow cartogram.

The dots are drawn once on the map with the true land area and carried along with the flow as landmarks.
In the cartogram the area of a state is proportional to its population, so the density of the dots
is the poverty rate.
"""

# %%
# Generate the dots
# -----------------
# Every dot stands for the same number of people below the poverty line. The state with the most people has about
# 1,500 dots, and a state has as many dots as its count asks for, spread uniformly over its area.

import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import carto_flow.data as examples
import carto_flow.flow_cartogram as flow
import carto_flow.proportional_cartogram as prop

us_states = examples.load_us_census(population=True, poverty=True, simplify=1000)
column = "Below Poverty Level"

n_dots = round(us_states[column].max() / 25_000)
dots = prop.generate_dot_density(us_states, columns=[column], n_dots=n_dots, normalization="maximum", seed=7)
people_per_dot = us_states[column].max() / n_dots
print(f"{len(dots)} dots, one dot is {people_per_dot:,.0f} people")

# %%
# Morph with the dots as landmarks
# --------------------------------
# The dots are passed as ``landmarks``, so every snapshot of the morph stores where they are.
# ``snapshot_every`` saves a snapshot every few iterations.

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cartogram = flow.create_flow_cartogram(
        us_states,
        "Population",
        landmarks=dots.geometry,
        options=flow.MorphOptions(area_scale=1e-6, snapshot_every=4, show_progress=False),
    )
print(cartogram.status.name)

# %%
# Frames
# ------
# Carried dots pile up a little near the borders of their state, because the flow squeezes the area differently there.
# ``get_landmarks(iteration, uniform=True)`` remaps their distance to the border of the state so that dots that were
# uniform before the morph are uniform again, at any saved iteration.

iterations = cartogram.snapshots.get_iterations()
frames = [
    (
        cartogram.get_geometry(iteration),
        np.array([[p.x, p.y] for p in cartogram.get_landmarks(iteration, uniform=True)]),
    )
    for iteration in iterations
]
print(f"{len(frames)} frames")

# %%
# Animate
# -------
# The animation holds the true-area map, eases through the snapshots (slowly at the start and the end, where the flow
# is slow) and holds the cartogram.

hold, steps = 6, 36
timeline = (
    [0] * hold
    + [round((len(frames) - 1) * (0.5 - 0.5 * np.cos(np.pi * k / (steps - 1)))) for k in range(steps)]
    + [len(frames) - 1] * hold
)

fig, ax = plt.subplots(figsize=(8, 5))
fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.04)


def draw(frame):
    ax.clear()
    geometry, points = frames[timeline[frame]]
    gpd.GeoSeries(list(geometry)).plot(ax=ax, facecolor="0.94", edgecolor="0.55", linewidth=0.6)
    ax.scatter(points[:, 0], points[:, 1], s=3, color="#081d58", linewidths=0, alpha=0.85)
    ax.set_axis_off()
    area_proportional = timeline[frame] >= (len(frames) - 1) / 2
    ax.set_title(
        "Area proportional to population: dot density is the poverty rate" if area_proportional else "True land area",
        fontsize=10,
    )
    ax.text(
        0.5,
        -0.02,
        f"1 dot = {people_per_dot:,.0f} residents below the poverty line",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="0.4",
    )


anim = FuncAnimation(fig, draw, frames=len(timeline), interval=1000 / 15)
