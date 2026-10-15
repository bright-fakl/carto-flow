"""
Dot Density Cartogram
=====================

A dot density cartogram of poverty levels across US states.
Uses `carto_flow.proportional_cartogram.plot_dot_density`.
"""

# %%
# Load dataset and plot dot density cartogram.
import matplotlib

matplotlib.use("Agg")

import carto_flow.data as examples
import carto_flow.proportional_cartogram as pc

us_states = examples.load_us_census(population=True, poverty=True, simplify=1000)

result = pc.plot_dot_density(
    us_states,
    columns=["Above Poverty Level", "Below Poverty Level"],
    n_dots=1000,
    normalization="maximum",
    alpha=[0.1, 1.0],
    cmap={"Below Poverty Level": "#ff0000", "Above Poverty Level": "#000000"},
)

result.ax.axis("off")
