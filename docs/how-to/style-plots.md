# Style Plots and Animations

A `PlotStyle` sets the look of the plots and animations of the flow, Voronoi, proportional and symbol cartograms in one place: the surface and text colors, the boundary, the palettes by data type, the color scale and the colorbar. A call decides which data a plot shows, the style decides how it looks by default, and an argument of the call always wins over the style.

## Choose a style

Three styles are built in: `"default"` (the look of the library), `"dark"` and `"classic"` (the look before styles: matplotlib and library defaults).

```python
cartogram.plot(facecolor="Population")                  # the default style
cartogram.plot(facecolor="Population", style="dark")    # one plot
```

For a block of code or a whole session:

```python
import carto_flow as cf

with cf.style_context("dark"):
    cartogram.plot(facecolor="Population")
    anim = animate_morph_history(cartogram, facecolor="errors_pct", legend=True)

cf.set_style("dark")      # the session, like matplotlib.pyplot.style.use
cf.set_style(None)        # back to the default
```

The style of a call is the `style=` argument, else the block, else the session, else `"default"`.

Plots that take `style=`: `Cartogram.plot`, `plot_cartogram`, `plot_comparison`, the field and convergence plots and the `animate_*` functions of the flow cartogram; `VoronoiCartogram.plot`, `plot_displacement`, `plot_comparison`, `plot_compactness`, the topology plots, the convergence plot and `animate_voronoi_history`; `plot_partitions` and `plot_dot_density` of the proportional cartogram; `SymbolCartogram.plot`, `plot_symbols`, `plot_comparison`, `plot_adjacency`, `plot_adjacency_heatmap`, `plot_tiling` and `plot_displacement` of the symbol cartogram.

## What the style sets for each type of data

The plot function decides the role of the data from the data itself, and the style supplies the palette of that role:

| data | role | palette |
|---|---|---|
| a numeric column or array | sequential | `palettes.sequential` (the default style truncates `YlGnBu` so that no fill is close to the surface) |
| `facecolor="errors_pct"`, or any data with `center=` | diverging, symmetric around the center | `palettes.diverging` |
| a text, categorical or boolean column, or `categorical=True` | categorical | `palettes.categorical` |

```python
cartogram.plot(facecolor="Region")                      # categories
cartogram.plot(facecolor="Population", center=1e7)      # diverging around 10 million
cartogram.plot(facecolor="Population", norm="log")      # a logarithmic scale
cartogram.plot(facecolor="Population", cmap="Reds")     # a palette of the call
```

The default style takes a logarithmic scale for positive data that spans more than one decade (`color_scale="auto"`) and writes the tick labels in real numbers (`1M`, `2M`, `5M`). `norm=` accepts a matplotlib `Normalize` or one of `"linear"`, `"log"`, `"auto"`.

To keep a linear scale for data that spans more than one decade, pass `norm="linear"` to the call, set `color_scale="linear"` in a style (`PlotStyle.default().copy_with(color_scale="linear")`), or raise the threshold with `log_span` (the number of decades that triggers `"auto"`, default 1). A `vmin` or `vmax` in the call also fixes a linear scale; combined with `norm="log"` they are the limits of the logarithmic scale.

## Write your own style

Start from a built-in style and change what you need:

```python
from carto_flow import Channel, Palettes, PlotStyle, register_style

mine = PlotStyle.default().copy_with(
    palettes=Palettes(sequential="Blues", sequential_range=(0.2, 1.0), diverging="PuOr"),
    channels={"edge": Channel(value="#444444"), "linewidth": Channel(value=0.5)},
    color_scale="linear",
    rc_update={"font.size": 12},     # merges into the matplotlib rcParams of the style
)
register_style("mine", mine)
cartogram.plot(facecolor="Population", style="mine")
```

- `rc` holds matplotlib rcParams (surface, text and tick colors, font sizes, grid, legend). `copy_with(rc=...)` replaces them, `copy_with(rc_update=...)` merges.
- `PlotStyle.from_mplstyle("ggplot")` or `PlotStyle.from_mplstyle("my.mplstyle")` builds a style from a matplotlib style sheet.
- `palettes` holds the colormap of each role. `channels["edge"].value` and `channels["linewidth"].value` are the boundary color and width of the regions.
- `channels` also sets the fixed fill (`"fill"`), opacity (`"alpha"`) and the boundary of the symbols; with `range=(low, high)` a channel sets the output range of the data-driven `alpha`, `linewidth` and `label` (font size) of `plot_symbols`. `hatch_cycle` is the sequence of hatch patterns assigned to the categories of `hatch=`.
- `reference_facecolor`, `reference_edgecolor`, `reference_linewidth` and `reference_alpha` are the layer of context geometries (the base of a dot density map, the complement of `plot_partitions`, the tiles of `plot_tiling`); `annotation_size` is the font size of the labels placed on a map (`label_fontsize` of the Voronoi plots).
- `series_color`, `accent_color`, `positive_color` and `highlight_color` are the colors of the diagnostic overlays of the Voronoi plots and animation: the convergence curve and the cells and centroids of an animation, the marks of a deviation (displacement targets, violated adjacency), of a good outcome (convergence, reassigned cells) and of a case to look at (satellite cells). The base layers of these plots use the reference fill and the axes edge color.
- `color_scale` (`"linear"`, `"log"`, `"auto"`), `tick_format` (`"plain"`, `"si"`) and the `colorbar_*` fields set the colorbar.
- A field left `None` leaves matplotlib or the library default alone, so `PlotStyle()` is the classic look.
- `style.rc_params()` returns the rcParams, to use the look in your own matplotlib code with `plt.rc_context(style.rc_params())`. A figure that you created yourself gets the surface with `style.apply(fig)`.

`"default"`, `"dark"` and `"classic"` cannot be replaced; `register_style(name, style, overwrite=True)` replaces another registered name, and `unregister_style(name)` removes one.

### Keep your styles between sessions

Registered styles live in the memory of the running Python process; nothing is saved to disk, so they are gone when the session ends. To have them in every session, register them in code that runs at startup:

- Put the `register_style(...)` calls in a module of your own, for example `my_styles.py` next to your project or on your Python path, and import it (`import my_styles`) at the top of a script or notebook.
- Or put the calls in a startup script that runs automatically: an IPython startup file (`~/.ipython/profile_default/startup/10-carto-flow-styles.py`) for interactive sessions and notebooks, or a file named by the `PYTHONSTARTUP` environment variable for the plain Python prompt.

```python
# my_styles.py
from carto_flow import PlotStyle, register_style, set_style

register_style("house", PlotStyle.default().copy_with(rc_update={"font.size": 12}))
set_style("house")      # optional: make it the style of the session
```

A matplotlib style sheet (`PlotStyle.from_mplstyle("my.mplstyle")`) can hold the `rc` settings in a file, but not the palettes, channels, color scale or colorbar fields.

## Combine styles

A field that a style leaves `None` is left alone, so a style can set only part of the look and be layered over another. `a | b` (or `a.merge(b)`) is a style in which whatever `b` sets wins: the `rc` dicts merge, the palettes and the channels merge field by field, and the other fields of `b` that are not `None` replace those of `a`.

```python
talk_fonts = PlotStyle(rc={"xtick.labelsize": 20, "ytick.labelsize": 20, "axes.labelsize": 22})
big_dark = PlotStyle.dark() | talk_fonts
```

`style=`, `style_context` and `set_style` also take a sequence of styles or names, merged in order:

```python
cartogram.plot(facecolor="Population", style=("dark", talk_fonts))
```

## Figure size, large type and animations

The size of a figure is not part of a style: pass `figsize=` (and `dpi=` when saving). The large type of a slide is a style with larger fonts:

```python
talk = PlotStyle(rc={"xtick.labelsize": 20, "ytick.labelsize": 20, "axes.labelsize": 22})
register_style("talk", talk)
anim = animate_geometry_keyframes(keyframes, style=("default", "talk"), figsize=(16, 9), legend=True)
save_animation(anim, "map.mp4", fps=30, dpi=120)   # the saved frames have the surface of the style
```

`morph_timeline` turns the snapshots of a cartogram into keyframes that hold on the first state, ease through the morph, hold on the last state and return:

```python
from carto_flow.flow_cartogram.animation import animate_geometry_keyframes, linear_over, morph_timeline

keyframes, times = morph_timeline(cartogram, hold_start=1.5, morph=7.0, hold_end=2.5, back=5.0)
anim = animate_geometry_keyframes(
    keyframes,
    keyframe_times=times,
    position_mapper=linear_over("time"),
    duration=times[-1],
    fps=30,
    facecolor="Population",
    legend=True,
)
```

`back=0` leaves out the return; `easing="linear"` plays the morph at constant speed.
