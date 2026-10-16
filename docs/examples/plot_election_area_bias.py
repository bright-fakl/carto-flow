"""
Area Bias in an Election Map
============================

The 2020 US presidential election on a choropleth, a non-contiguous cartogram and a flow cartogram.

A map colors areas, and the eye adds up the area of each color. Land area is a poor measure of votes: the large
states of the interior are mostly Republican and hold few people, so a choropleth looks redder than the election
was. A cartogram whose areas are proportional to the population corrects this.

Data: Federal Election Commission, official 2020 presidential results by state.
"""

# %%
# Join the results to the states
# ------------------------------
# The two-party Democratic share is the votes for Biden over the votes for Biden and Trump. The population of a state
# sizes its area in the cartograms.

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize

import carto_flow.data as examples
import carto_flow.flow_cartogram as flow
from carto_flow import get_style
from carto_flow.noncontiguous_cartogram import create_noncontiguous_cartogram

states = examples.load_us_census(population=True, simplify=1000)
results = examples.load_us_president_2020()
states = states.merge(results, left_on="State Abbreviation", right_on="state_abbr")
states["dem_share"] = states["votes_dem"] / (states["votes_dem"] + states["votes_gop"])
states["winner"] = np.where(states["votes_dem"] > states["votes_gop"], "Biden", "Trump")

true_share = states["votes_dem"].sum() / (states["votes_dem"].sum() + states["votes_gop"].sum())
print(f"Democratic share of the two-party vote in these states: {true_share:.1%}")

# %%
# Three maps with the same data
# -----------------------------
# The non-contiguous cartogram scales every state about its own center. New Jersey, the next densest state after DC,
# keeps its size, and DC, which is nine times as dense, is kept at its size (smaller than proportional). The flow
# cartogram moves the borders so that the areas are proportional to the population and the states keep touching.

new_jersey = states.index[states["State Abbreviation"] == "NJ"][0]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    noncontiguous = create_noncontiguous_cartogram(states, "Population", reference_region=new_jersey)
    contiguous = flow.create_flow_cartogram(
        states, "Population", options=flow.MorphOptions(area_scale=1e-6, show_progress=False)
    )

maps = {
    "Choropleth\narea = land": states,
    "Non-contiguous cartogram\narea ∝ population": noncontiguous.geometries,
    "Flow cartogram\narea ∝ population": contiguous.to_geodataframe(),
}


def area_weighted(gdf, column="dem_share"):
    """What the eye reads: the value of each state weighted by the area that it is drawn with."""
    areas = gdf.geometry.area.to_numpy()
    return float(np.sum(areas * gdf[column].to_numpy()) / areas.sum())


norm = Normalize(0.3, 0.7)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
get_style().apply(fig)
for ax, (title, gdf) in zip(axes, maps.items(), strict=True):
    gdf.plot(column="dem_share", cmap="RdBu", norm=norm, edgecolor="white", linewidth=0.4, ax=ax)
    ax.set_title(f"{title}\narea-weighted Democratic share {area_weighted(gdf):.1%}", fontsize=10)
    ax.set_axis_off()
fig.colorbar(ScalarMappable(norm, "RdBu"), ax=axes, shrink=0.6, label="Democratic share of the two-party vote")

# %%
# The area-weighted share is what a reader takes from the map. It is about 5 percentage points too low on the
# choropleth and within 0.2 points of the actual share on the two cartograms.

for title, gdf in maps.items():
    print(f"{title.splitlines()[0]:<26} {area_weighted(gdf):.1%}  (actual {true_share:.1%})")

# %%
# Who won each state
# ------------------
# The winner of a state is a category, and ``cmap`` takes a color for each category. On the choropleth about 43 % of
# the area is blue; in the cartograms the share of the area that is blue is close to the share of the voters who
# live in the states that Biden won.

colors = {"Biden": "#3b7fb8", "Trump": "#c4423a"}
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
get_style().apply(fig)
for ax, (title, gdf) in zip(axes, maps.items(), strict=True):
    if gdf is states:
        gdf.plot(column="winner", cmap=ListedColormap(list(colors.values())), edgecolor="white", linewidth=0.4, ax=ax)
    else:
        owner = noncontiguous if gdf is noncontiguous.geometries else contiguous
        owner.plot(facecolor="winner", cmap=colors, legend=False, ax=ax)
    blue = gdf.geometry.area[gdf["winner"] == "Biden"].sum() / gdf.geometry.area.sum()
    ax.set_title(f"{title}\n{blue:.0%} of the area is blue", fontsize=10)
    ax.set_axis_off()

voters = states.loc[states["winner"] == "Biden", "Population"].sum() / states["Population"].sum()
print(f"Share of the population in the states that Biden won: {voters:.0%}")
