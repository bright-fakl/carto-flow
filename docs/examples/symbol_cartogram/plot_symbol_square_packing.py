"""
Square Packing Layout
=====================

Pack squares with the circle packing layout, so that they keep their neighbors and touch along flat edges.
"""

# %%
# ``shape="square"`` makes the packing layout treat the symbols as axis-aligned squares with the area of the circles.
# The topology, neighbor and compaction forces are those of the circle layout; squares touch along flat edges and slide
# along each other.

import matplotlib.pyplot as plt

import carto_flow.data as examples
import carto_flow.symbol_cartogram as smb
from carto_flow import get_style

us_states = examples.load_us_census(population=True)

on_circles = smb.create_symbol_cartogram(
    us_states,
    "Population",
    layout=smb.CirclePackingLayout(spacing=0.0),
    styling=smb.Styling(symbol="square"),
    show_progress=False,
)
packed = smb.create_symbol_cartogram(
    us_states,
    "Population",
    layout=smb.CirclePackingLayout(shape="square", spacing=0.0),
    styling=smb.Styling(symbol="square"),
    show_progress=False,
)

# %%
# Plot the two layouts side by side.

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
get_style().apply(fig)

for ax, carto, title in [
    (axes[0], on_circles, "Squares on a circle layout"),
    (axes[1], packed, "Square packing"),
]:
    carto.plot(ax=ax, facecolor="Population (Millions)", legend=False)
    ax.set_title(title)

plt.tight_layout()
