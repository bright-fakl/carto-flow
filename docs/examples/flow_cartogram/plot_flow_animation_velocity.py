"""
Velocity Field Animation
========================

Animates the internal velocity field using `animate_velocity_field`.
"""

# %%
# Build the cartogram with snapshot

import carto_flow.data as examples
import carto_flow.flow_cartogram as flow
from carto_flow.flow_cartogram.animation import animate_velocity_field

gdf = examples.load_us_census(population=True)

cartogram = flow.create_flow_cartogram(
    gdf,
    "Population",
    options=flow.MorphOptions.preset_balanced().copy_with(
        n_iter=250,
        snapshot_every=5,
        area_scale=1e-6,
        show_progress=False,
        save_internals=True,
    ),
)

# %%
# Animate the morphing snapshots.

anim = animate_velocity_field(
    cartogram,
    duration=2.0,
    fps=15,
    show_axes=False,
    figsize=(8, 5),
    velocity=flow.VelocityPlotOptions(legend_kwargs={"shrink": 0.7}, color_by="direction", velocity_scale=3),
)
