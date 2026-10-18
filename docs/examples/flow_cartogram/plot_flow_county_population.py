"""
US County Population
====================

A flow cartogram of the 3,108 contiguous US counties, each sized by its population.

Counties are a demanding input: a county in the Mountain West has a target area
a thousandth of the one it starts with, and the smallest targets are far below
one cell of a grid fine enough for the largest ones. A resolution ladder
(``grid="auto"``) starts coarse, where the large-scale rearrangement is cheap,
and refines until the small counties are resolved.
"""

# %%
# Load data
# ---------
# The bundled county snapshot has the ACS 2020 population and poverty counts.

import warnings

import matplotlib.pyplot as plt
import numpy as np

import carto_flow.data as examples
import carto_flow.flow_cartogram as flow
from carto_flow import get_style

counties = examples.load_us_census(level="county", population=True, poverty=True)
print(
    f"{len(counties)} counties, population from {counties['Population'].min():,.0f} to {counties['Population'].max():,.0f}"
)

# %%
# Compute the cartogram
# ---------------------
# ``grid="auto"`` picks the coarsest grid on which the smallest county covers
# at least 2.5 cells and doubles it up to 2048. ``n_iter`` is the total
# number of iterations of the ladder; counties need more than the default.

# The run warns that the smallest counties are below one cell of every grid, and
# that some output polygons are not valid; both are expected here and handled
# below, so the warnings are not repeated.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Grid resolution")
    warnings.filterwarnings("ignore", message=r"\d+ output regions are not valid")
    warnings.filterwarnings("ignore", message="Input geometries contain a straight segment")
    cartogram = flow.create_flow_cartogram(
        counties,
        "Population",
        options=flow.MorphOptions(grid="auto", n_iter=1000, area_scale=1e-6, show_progress=False),
    )
print(cartogram.status.name, cartogram.niterations)

# %%
# Plot
# ----
# Color shows the share of the population below the poverty level and heavy
# lines are state boundaries. The area of a county is proportional to its
# population, so the large metropolitan counties dominate the map. A few output
# polygons cross themselves; ``repair=True`` makes them valid for plotting.

result = cartogram.to_geodataframe(repair=True)
states = result.dissolve("State Abbreviation")

fig, ax = plt.subplots(figsize=(14, 8))
get_style().apply(fig)
result.plot(
    column="Below Poverty Level %",
    cmap="YlOrRd",
    vmin=0.05,
    vmax=0.35,
    linewidth=0.1,
    edgecolor="white",
    legend=True,
    legend_kwds={"shrink": 0.5, "label": "Share below the poverty level"},
    ax=ax,
)
states.boundary.plot(ax=ax, color="0.2", linewidth=0.6)
ax.set_title("US counties, area proportional to population")
ax.axis("off")
plt.tight_layout()

# %%
# Accuracy
# --------
# The area error of a county is its area against the share of the total area
# its population entitles it to. Almost all counties are within a few percent;
# the few large errors belong to counties whose target is below one cell of the
# finest grid.

areas = result.geometry.area.to_numpy()
target = counties["Population"].to_numpy() / counties["Population"].sum() * areas.sum()
error = np.abs(areas - target) / target * 100
print(
    f"median error {np.median(error):.2f}%, 95th percentile {np.percentile(error, 95):.1f}%, mean {error.mean():.1f}%"
)
