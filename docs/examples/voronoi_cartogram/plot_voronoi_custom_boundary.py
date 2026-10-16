"""
Circular Voronoi Cartogram
==========================

Population cartogram of US States fitted inside a custom circular boundary.
"""

# %%
# Build a circular boundary and run the cartogram inside it.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import carto_flow.data as examples
import carto_flow.voronoi_cartogram as vor
from carto_flow import get_style

us_states = examples.load_us_census(population=True)

result = vor.create_voronoi_cartogram(
    us_states,
    weights="Population (Millions)",
    boundary="circle",
    backend=vor.RasterBackend(resolution=256),
    options=vor.VoronoiOptions(n_iter=300, area_cv_tol=0.05),
)

# %%
# Plot the cartogram.
fig, ax = plt.subplots(figsize=(8, 7))
get_style().apply(fig)
result.plot("Population (Millions)", ax=ax)
ax.set(title="Cartogram of US States by Population — circular boundary")
ax.axis("off")
plt.tight_layout()
