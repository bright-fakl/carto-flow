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
from carto_flow import get_style

us_states = examples.load_us_census(population=True)

# compute flow cartogram
result = flow.create_flow_cartogram(us_states, "Population", options=flow.MorphOptions(show_progress=False))

# %%
# Plot the cartogram.
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
get_style().apply(fig)

p = result.plot("Population (Millions)", ax=ax)

p.ax.set(title="Cartogram of US States by Population")
p.ax.axis("off")

plt.tight_layout()
