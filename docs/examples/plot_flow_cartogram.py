"""
Flow Cartogram
==============

A flow-based cartogram where US State areas are proportional to population.
Uses `carto_flow.flow_cartogram.create_flow_cartogram`.
"""

# %%
# Load dataset and run the morphing algorithm.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import carto_flow.data as examples
import carto_flow.flow_cartogram as flow

us_states = examples.load_us_census(population=True)

# compute flow cartogram
result = flow.create_flow_cartogram(us_states, "Population", options=flow.MorphOptions(show_progress=False))

# %%
# Plot the cartogram.
_, ax = plt.subplots(1, 1, figsize=(10, 6))

p = result.plot(
    "Population (Millions)",
    ax=ax,
    vmin=0,
    vmax=40,
    legend=True,
    legend_kwds={"shrink": 0.6, "label": "Population (Millions)"},
    cmap="RdYlGn_r",
    linewidth=0.5,
)

p.ax.set(title="Cartogram of US States by Population")
p.ax.axis("off")

plt.tight_layout()
