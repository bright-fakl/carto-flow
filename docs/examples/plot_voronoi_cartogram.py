"""
Voronoi Cartogram
=================

A Voronoi tesselation cartogram where US States are represented
by Voronoi cells whose areas are proportional to population.
Uses `carto_flow.voronoi_cartogram.create_voronoi_cartogram`.
"""

# %%
# Load dataset and run the morphing algorithm.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import carto_flow.data as examples
import carto_flow.voronoi_cartogram as vor
from carto_flow import get_style

us_states = examples.load_us_census(population=True)

# compute Voronoi cartogram
result = vor.create_voronoi_cartogram(
    us_states,
    weights="Population",
    backend=vor.RasterBackend(resolution=256, boundary=vor.ElasticBoundary(strength=0.02), adjacency_spring=0.2),
    options=vor.VoronoiOptions(n_iter=100, show_progress=False),
)

# %%
# Plot the cartogram.
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
get_style().apply(fig)

p = result.plot("Population (Millions)", ax=ax)

p.ax.set(title="Cartogram of US States by Population")
p.ax.axis("off")

plt.tight_layout()
