"""
Non-contiguous Cartogram
========================

US states scaled about their centers to an area proportional to their population.
"""

# %%
# Every state keeps its shape and is scaled to ``population / density``. With the default reference, the density of
# the densest region (DC), no state grows and every area is proportional to its population. DC is nine times as
# dense as the next densest state, so most states become very small.

import warnings

import carto_flow.data as examples
from carto_flow.noncontiguous_cartogram import create_noncontiguous_cartogram

states = examples.load_us_census(population=True, simplify=1000)

cartogram = create_noncontiguous_cartogram(states, "Population")
plot = cartogram.plot(facecolor="Population")
_ = plot.ax.set_title("Reference: the densest region (DC)")

# %%
# ``reference_region`` names a region that keeps its size. With New Jersey, the next densest state, as the reference,
# only DC is denser than the reference. Growth is capped at the original size by default, so DC keeps its size and
# is smaller than proportional; ``capped`` marks it.

new_jersey = states.index[states["State Abbreviation"] == "NJ"][0]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cartogram = create_noncontiguous_cartogram(states, "Population", reference_region=new_jersey)
plot = cartogram.plot(facecolor="Population")
_ = plot.ax.set_title("Reference: New Jersey (DC capped)")

# %%
# With ``max_scale=None`` DC grows to about nine times its area and overlaps its neighbors. The regions are drawn in
# order of decreasing area, so the smaller states stay visible on top of it.

cartogram = create_noncontiguous_cartogram(states, "Population", reference_region=new_jersey, max_scale=None)
plot = cartogram.plot(facecolor="Population")
_ = plot.ax.set_title("Reference: New Jersey (DC grows)")

# %%
# The density of the state with the largest population (California) is the reference of the Olson cartogram of the R
# package ``cartogram``. Eleven states are denser and would have to grow; capped, they keep their original size.

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    capped = create_noncontiguous_cartogram(states, "Population", reference="largest")
plot = capped.plot(facecolor="_ncont_scale", legend_label="scale factor")
_ = plot.ax.set_title(f"Reference: the largest state ({capped.capped.sum()} regions capped)")
