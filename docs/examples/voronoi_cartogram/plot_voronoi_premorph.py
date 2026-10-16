"""
Voronoi Cartogram from a Flow Cartogram
=======================================

Population cartogram of US States: the flow cartogram (left) and the Voronoi
cartogram built from it with ``premorph=True`` (right). The Voronoi cells
start from the morphed regions, keep the flow cartogram's outline, and have
areas proportional to population.
"""

# %%
# Run the Voronoi cartogram on the flow-morphed states.
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
    premorph=True,
    backend=vor.RasterBackend(resolution=256),
    options=vor.VoronoiOptions(n_iter=300, area_cv_tol=0.05),
)

# %%
# Plot the flow cartogram and the Voronoi cells side by side.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
get_style().apply(fig)
result.premorph.plot("Population (Millions)", ax=axes[0], legend=False)
result.plot("Population (Millions)", ax=axes[1], legend=False)
for ax, title in zip(axes, ["Flow cartogram", "Voronoi cartogram, premorph=True"], strict=True):
    ax.set(title=title)
    ax.axis("off")
plt.tight_layout()
