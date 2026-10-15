"""
Proportional Cartogram
======================

A proportional cartogram in which each US State is split into
partitions that represent the distribution of three major minority
groups across US states.
Uses `carto_flow.proportional_cartogram.create_proportional_cartogram`
"""

import matplotlib
from matplotlib import patheffects

matplotlib.use("Agg")

import carto_flow.data as examples
import carto_flow.proportional_cartogram as pc

# %%
# Load dataset and create cartogram.

us_states = examples.load_us_census(population=True, race=True, simplify=1000)

partition = pc.partition_geometries(
    us_states,
    columns=["Black or African American", "Asian", "Hispanic or Latino"],
    method="split",
    direction="horizontal",
    alternate=True,
    normalization="row",
)

# %%
# Plot the cartogram.
result = pc.plot_partitions(
    partition,
    facecolor="category",
    edgecolor="white",
    cmap={"Black or African American": "indigo", "Asian": "darkorchid", "Hispanic or Latino": "plum"},
)

result.ax.axis("off")
result.ax.set_title("Distribution of Minority Groups")

for _, row in us_states.iterrows():
    h = result.ax.text(
        row.geometry.centroid.x,
        row.geometry.centroid.y,
        row["State Abbreviation"],
        va="center",
        ha="center",
        color="white",
        fontweight="bold",
    )

    h.set(path_effects=[patheffects.withStroke(linewidth=1, foreground="k")])
