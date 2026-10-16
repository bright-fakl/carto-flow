"""
Isohedral Tiling Presets
========================

Plot isohedral tiling presets.
"""

# %%
# Load data and create the symbol cartogram.

import matplotlib.pyplot as plt

import carto_flow.symbol_cartogram as smb
from carto_flow import get_style

# %%
# Plot the presets

fig, axes = plt.subplots(3, 5, sharex=True, sharey=True)
get_style().apply(fig)

for key, ax in zip(smb.IsohedralTiling.list_presets().keys(), axes.ravel(), strict=False):
    t = smb.IsohedralTiling.from_preset(key)
    t.plot_tile(ax=ax, show_vertices=False)
    ax.axis("square")
    ax.set(xlim=(-1, 1), ylim=(-1, 1))
    ax.set_title(key, fontsize=7)
