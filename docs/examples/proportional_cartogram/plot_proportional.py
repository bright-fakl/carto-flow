"""
Shells and Cores
================

A proportional cartogram of poverty in the United States.
"""

# %%
# Create cartogram by shrinking geometries to create concentric shells.

import carto_flow.data as examples
from carto_flow.proportional_cartogram import partition_geometries
from carto_flow.proportional_cartogram.visualization import plot_partitions

us_states = examples.load_us_census(population=True, race=True, poverty=True, simplify=1000)

partition = partition_geometries(us_states, columns=["Below Poverty Level %", "Above Poverty Level %"], method="shrink")

plot = plot_partitions(
    partition,
    facecolor="category",
    edgecolor="white",
    cmap={"Below Poverty Level %": "darkgreen", "Above Poverty Level %": "#cceecc"},
)

_ = plot.ax.axis("off")
_ = plot.ax.set_title("Poverty in the US")
