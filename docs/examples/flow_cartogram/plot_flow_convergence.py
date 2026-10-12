"""
Flow Cartogram Convergence
==========================

Plots the convergence of the diffusion algorithm using `plot_convergence`.
"""

# %%
# Build the cartogram

import matplotlib.pyplot as plt

import carto_flow.data as examples
import carto_flow.flow_cartogram as flow

gdf = examples.load_us_census(population=True)

cartogram = flow.create_flow_cartogram(
    gdf,
    "Population",
    options=flow.MorphOptions.preset_balanced().copy_with(n_iter=250, area_scale=1e-6, show_progress=False),
)

# %%
# Plot convergence of mean and max errors.

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

plot = flow.visualization.plot_convergence(cartogram, ax=ax, show_recompute=True)
