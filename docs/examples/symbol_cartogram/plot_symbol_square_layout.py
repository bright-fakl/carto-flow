"""
Square Layout
=============

Place squares at the centroids of the original geometries so that they sit flush against each other.
"""

# %%
# A square drawn on a layout of circles sits inside its circle, so gaps remain between the squares. With
# ``shape="square"`` the layout separates axis-aligned squares of the same area, and with ``spacing=0`` their
# edges touch.

import matplotlib.pyplot as plt

import carto_flow.data as examples
import carto_flow.symbol_cartogram as smb
from carto_flow import get_style

us_states = examples.load_us_census(population=True)

on_circles = smb.create_symbol_cartogram(
    us_states,
    "Population",
    layout=smb.CentroidLayout(spacing=0.0),
    styling=smb.Styling(symbol="square"),
    show_progress=False,
)
flush = smb.create_symbol_cartogram(
    us_states,
    "Population",
    layout=smb.CentroidLayout(shape="square", spacing=0.0),
    styling=smb.Styling(symbol="square"),
    show_progress=False,
)

# %%
# Plot the two layouts side by side.

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
get_style().apply(fig)

for ax, carto, title in [
    (axes[0], on_circles, "Squares on a circle layout"),
    (axes[1], flush, 'Square layout (shape="square")'),
]:
    carto.plot(ax=ax, facecolor="Population (Millions)", legend=False)
    ax.set_title(title)

plt.tight_layout()
