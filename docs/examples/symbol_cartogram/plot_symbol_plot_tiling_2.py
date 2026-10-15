"""
Visualize Tiling
================

Visualize a layout's tiling grid.
"""

# %%
# Plot the symbol cartogram

import matplotlib.pyplot as plt

import carto_flow.symbol_cartogram as smb

tiling = smb.IsohedralTiling.from_preset("wavy_triangle")
grid = tiling.generate()

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

_ = tiling.plot_tile(ax=axes[0], show_vertices=False, facecolor="orange", edgecolor="black")

_ = grid.plot(
    ax=axes[1],
    color_by="aspect",
    cmap="hot",
)

_ = axes[0].axis("square")
_ = axes[1].axis("square")
