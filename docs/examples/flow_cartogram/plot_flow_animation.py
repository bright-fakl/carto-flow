"""
Flow Cartogram Animation
========================

Animates the diffusion algorithm snapshots using `animate_morph_history`.
"""

# %%
# Build the cartogram with snapshot

import carto_flow.data as examples
import carto_flow.flow_cartogram as flow
from carto_flow.flow_cartogram.animation import animate_morph_history

gdf = examples.load_us_census(population=True)

cartogram = flow.create_flow_cartogram(
    gdf,
    "Population",
    options=flow.MorphOptions.preset_balanced().copy_with(
        n_iter=250, snapshot_every=5, area_scale=1e-6, show_progress=False
    ),
)

# %%
# Animate the morphing snapshots.

anim = animate_morph_history(
    cartogram,
    duration=2.0,  # total animation length in seconds
    fps=15,
    facecolor="errors_pct",
    cmap="PiYG",
    figsize=(8, 5),
    show_axes=False,
    legend=True,
    vmin=-1000,
    vmax=1000,
    linewidth=0.5,
    edgecolor="black",
    title="iteration {iteration:.1f} | error = {mean_error_pct:.1f}%",
)
