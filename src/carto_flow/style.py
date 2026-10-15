"""Plot styles for the plots and animations of the cartograms.

A :class:`PlotStyle` sets the look of a plot in one place: the surface, text and boundary colors, the palettes by data
role (sequential, diverging, categorical), the color scale and the colorbar. A call decides which data a plot shows; the
style decides how it looks by default, and an explicit argument of the call wins over the style.

Functions
---------
register_style, get_style
    Register a style under a name; the style in effect or a registered style.
set_style, style_context
    Set the style of the session or of a block of code.

Examples
--------
>>> from carto_flow import PlotStyle, style_context
>>> cartogram.plot(facecolor="Population", style="dark")
>>> with style_context(PlotStyle.default().copy_with(rc_update={"xtick.labelsize": 20})):
...     cartogram.plot(facecolor="Population")
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
import inspect
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import matplotlib as mpl
import matplotlib.style
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap, LogNorm, Normalize, to_hex, to_rgb

__all__ = [
    "Channel",
    "Palettes",
    "PlotStyle",
    "get_style",
    "rc_wrapped",
    "register_style",
    "resolve_style",
    "set_style",
    "style_context",
    "styled",
]

ROLES = ("sequential", "diverging", "categorical")
CHANNELS = ("fill", "edge", "hatch", "alpha", "linewidth", "label")
_SCALES = ("linear", "log", "auto")
_TICK_FORMATS = ("plain", "si")
_ORIENTATIONS = ("vertical", "horizontal")


@dataclass(frozen=True)
class Palettes:
    """Colormaps of a style by data role.

    Attributes
    ----------
    sequential : str or Colormap, optional
        Colormap of continuous data.
    sequential_range : tuple of float, optional
        ``(lo, hi)`` part of ``sequential`` that is used, in 0..1 (the pale end is cut so that no fill is close to a
        tinted surface).
    diverging : str or Colormap, optional
        Colormap of data with a meaningful center (errors, differences).
    categorical : str, Colormap, sequence of colors or dict, optional
        Colors of categories: a qualitative colormap name, a list of colors, or a ``{category: color}`` dict (categories
        that are not in the dict take the colors of ``tab10``).
    """

    sequential: str | Colormap | None = None
    sequential_range: tuple[float, float] | None = None
    diverging: str | Colormap | None = None
    categorical: str | Colormap | Sequence[str] | Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if self.sequential_range is not None:
            lo, hi = self.sequential_range
            if not 0.0 <= lo < hi <= 1.0:
                raise ValueError(f"sequential_range must satisfy 0 <= lo < hi <= 1, got {self.sequential_range!r}")


@dataclass(frozen=True)
class Channel:
    """Appearance of a visual channel (``fill``, ``edge``, ``hatch``, ``alpha``, ``linewidth`` or ``label``).

    Attributes
    ----------
    value : Any, optional
        What the channel looks like when it is not data-driven: a color, a width, an alpha or a hatch pattern.
    palettes : Palettes, optional
        Palettes of this channel when it is data-driven; a field left ``None`` falls back to the palettes of the style.
    range : tuple of float, optional
        Output range of a scalar channel (``alpha``, ``linewidth``) when it is data-driven.
    """

    value: Any = None
    palettes: Palettes | None = None
    range: tuple[float, float] | None = None


@dataclass(frozen=True)
class PlotStyle:
    """The look of a plot.

    A field set to ``None`` leaves the matplotlib or library default alone, so ``PlotStyle()`` is the classic look and
    :meth:`default` sets the fields of the look of the library.

    Attributes
    ----------
    rc : dict
        Matplotlib rcParams (surface and text colors, font sizes, grid, legend) applied while a plot is drawn.
    palettes : Palettes
        Colormaps by data role, used for every color channel.
    channels : dict
        :class:`Channel` by channel name; ``channels["edge"].value`` is the boundary color and
        ``channels["linewidth"].value`` its width.
    color_scale : {"linear", "log", "auto"}, optional
        Scale of continuous data; ``"auto"`` takes a log scale when all values are positive and span more than
        ``log_span`` decades.
    log_span : float
        Decades that trigger ``color_scale="auto"``.
    tick_format : {"plain", "si"}, optional
        Format of the tick labels of a colorbar; ``"si"`` writes ``1M`` and ``2.5k``.
    colorbar_orientation, colorbar_shrink, colorbar_pad, colorbar_aspect : optional
        Placement of a colorbar.
    colorbar_outline_color, colorbar_outline_width : optional
        Outline of a colorbar.
    """

    rc: Mapping[str, Any] = field(default_factory=dict)
    palettes: Palettes = field(default_factory=Palettes)
    channels: Mapping[str, Channel] = field(default_factory=dict)
    color_scale: str | None = None
    log_span: float = 1.0
    tick_format: str | None = None
    colorbar_orientation: str | None = None
    colorbar_shrink: float | None = None
    colorbar_pad: float | None = None
    colorbar_aspect: float | None = None
    colorbar_outline_color: str | None = None
    colorbar_outline_width: float | None = None

    def __post_init__(self) -> None:
        if self.color_scale is not None and self.color_scale not in _SCALES:
            raise ValueError(f"color_scale must be one of {_SCALES} or None, got {self.color_scale!r}")
        if self.tick_format is not None and self.tick_format not in _TICK_FORMATS:
            raise ValueError(f"tick_format must be one of {_TICK_FORMATS} or None, got {self.tick_format!r}")
        if self.colorbar_orientation is not None and self.colorbar_orientation not in _ORIENTATIONS:
            raise ValueError(
                f"colorbar_orientation must be one of {_ORIENTATIONS} or None, got {self.colorbar_orientation!r}"
            )
        if not self.log_span > 0:
            raise ValueError(f"log_span must be positive, got {self.log_span!r}")
        unknown = set(self.channels) - set(CHANNELS)
        if unknown:
            raise ValueError(f"unknown channels {sorted(unknown)}; the channels are {CHANNELS}")
        for key in self.rc:
            if key not in mpl.rcParams:
                raise ValueError(f"{key!r} is not a matplotlib rcParam")

    # ------------------------------------------------------------------ construction

    def copy_with(self, *, rc_update: Mapping[str, Any] | None = None, **changes: Any) -> PlotStyle:
        """Copy of the style with fields replaced; ``rc_update`` merges into ``rc`` instead of replacing it."""
        if rc_update:
            changes["rc"] = {**changes.get("rc", self.rc), **rc_update}
        return dataclasses.replace(self, **changes)

    @classmethod
    def from_mplstyle(cls, name_or_path: str, **fields: Any) -> PlotStyle:
        """Style whose ``rc`` is a matplotlib style sheet: a name of ``matplotlib.style.available`` or a path."""
        if name_or_path in matplotlib.style.library:
            rc = dict(matplotlib.style.library[name_or_path])
        else:
            rc = dict(mpl.rc_params_from_file(name_or_path, use_default_template=False))
        return cls(rc=rc, **fields)

    @classmethod
    def classic(cls) -> PlotStyle:
        """The look before styles: nothing is set, matplotlib and the library defaults apply."""
        return cls()

    @classmethod
    def default(cls) -> PlotStyle:
        """The look of the library: tinted surface, boundary slightly darker than the surface, truncated ``YlGnBu``."""
        return _light()

    @classmethod
    def dark(cls) -> PlotStyle:
        """The look of the library on a dark surface."""
        return _dark()

    # ------------------------------------------------------------------ settings

    def rc_params(self) -> dict[str, Any]:
        """The rcParams of the style, for ``matplotlib.pyplot.rc_context`` in the user's own code."""
        return dict(self.rc)

    def rc_context(self):
        """Context manager that applies the rcParams of the style."""
        return mpl.rc_context(rc=dict(self.rc))

    def channel_value(self, channel: str, default: Any = None) -> Any:
        """The value of a channel when it is not data-driven, or ``default``."""
        found = self.channels.get(channel)
        return default if found is None or found.value is None else found.value

    def _palettes(self, channel: str) -> tuple[Palettes, Palettes | None]:
        found = self.channels.get(channel)
        return self.palettes, (found.palettes if found is not None else None)

    def colormap(self, role: str, channel: str = "fill") -> Colormap | None:
        """The colormap of a continuous role (``"sequential"`` or ``"diverging"``), or ``None`` if the style has none.

        The sequential colormap is truncated to ``sequential_range``.
        """
        if role not in ("sequential", "diverging"):
            raise ValueError(f"role must be 'sequential' or 'diverging', got {role!r}")
        base, override = self._palettes(channel)
        spec = getattr(override, role, None) if override is not None else None
        spec = spec if spec is not None else getattr(base, role)
        if spec is None:
            return None
        cmap = mpl.colormaps[spec] if isinstance(spec, str) else spec
        rng = (override.sequential_range if override is not None and override.sequential_range else None) or (
            base.sequential_range
        )
        if role == "sequential" and rng is not None:
            cmap = LinearSegmentedColormap.from_list(f"{cmap.name}_t", cmap(np.linspace(rng[0], rng[1], 256)))
        return cmap

    def categorical_colors(self, categories: Sequence[Any], channel: str = "fill") -> list[str] | None:
        """One color per category, or ``None`` if the style has no categorical palette."""
        base, override = self._palettes(channel)
        spec = override.categorical if override is not None and override.categorical is not None else None
        spec = spec if spec is not None else base.categorical
        if spec is None:
            return None
        n = len(categories)
        if isinstance(spec, Mapping):
            fallback = _cycle(mpl.colormaps["tab10"], n)
            return [to_hex(spec[c]) if c in spec else fallback[i] for i, c in enumerate(categories)]
        if isinstance(spec, str):
            return _cycle(mpl.colormaps[spec], n)
        if isinstance(spec, Colormap):
            return _cycle(spec, n)
        colors = [to_hex(c) for c in spec]
        return [colors[i % len(colors)] for i in range(n)]

    def norm(self, values: Any, role: str = "sequential", *, center: float | None = None, scale: str | None = None):
        """The normalization of the values of a continuous role.

        ``"diverging"`` is symmetric around ``center`` (default 0). ``"sequential"`` is linear or logarithmic by
        ``scale`` (default: ``color_scale`` of the style, else linear).
        """
        data = np.asarray(values, dtype=float)
        data = data[np.isfinite(data)]
        if data.size == 0:
            return Normalize()
        if role == "diverging":
            c = 0.0 if center is None else float(center)
            span = float(np.max(np.abs(data - c))) or 1.0
            return Normalize(vmin=c - span, vmax=c + span)
        scale = scale or self.color_scale or "linear"
        if scale not in _SCALES:
            raise ValueError(f"scale must be one of {_SCALES}, got {scale!r}")
        vmin, vmax = float(data.min()), float(data.max())
        positive = vmin > 0
        if scale == "auto":
            scale = "log" if positive and vmax / vmin > 10**self.log_span else "linear"
        if scale == "log":
            if not positive:
                raise ValueError("a logarithmic color scale needs positive values")
            return LogNorm(vmin=vmin, vmax=vmax)
        return Normalize(vmin=vmin, vmax=vmax)

    # ------------------------------------------------------------------ drawing

    @property
    def draws_guide(self) -> bool:
        """Whether the style changes the colorbar (otherwise the plot function leaves it to matplotlib)."""
        return any(
            getattr(self, name) is not None
            for name in (
                "tick_format",
                "colorbar_orientation",
                "colorbar_shrink",
                "colorbar_pad",
                "colorbar_aspect",
                "colorbar_outline_color",
                "colorbar_outline_width",
            )
        ) or self.color_scale in ("log", "auto")

    def colorbar(self, fig: Any, ax: Any, mappable: Any, label: str | None = None, **kwargs: Any):
        """Draw a colorbar for ``mappable`` with the placement, outline and tick labels of the style.

        Keyword arguments of the call win over the style. A logarithmic norm gets ticks at 1, 2 and 5 per decade.
        """
        kw: dict[str, Any] = {}
        for name, key in (
            ("colorbar_orientation", "orientation"),
            ("colorbar_shrink", "shrink"),
            ("colorbar_pad", "pad"),
            ("colorbar_aspect", "aspect"),
        ):
            if getattr(self, name) is not None:
                kw[key] = getattr(self, name)
        kw.update(kwargs)
        cbar = fig.colorbar(mappable, ax=ax, **kw)
        if label:
            cbar.set_label(label)
        for spine in cbar.ax.spines.values():
            if self.colorbar_outline_color is not None:
                spine.set_edgecolor(self.colorbar_outline_color)
            if self.colorbar_outline_width is not None:
                spine.set_linewidth(self.colorbar_outline_width)
        norm = getattr(mappable, "norm", None)
        if isinstance(norm, LogNorm) and norm.vmin and norm.vmax:
            ticks = _log_ticks(norm.vmin, norm.vmax)
            cbar.set_ticks(ticks)
            cbar.minorticks_off()
            cbar.set_ticklabels([_si(t) if self.tick_format == "si" else f"{t:g}" for t in ticks])
        elif self.tick_format == "si":
            cbar.formatter = mpl.ticker.FuncFormatter(lambda x, _: _si(x))
            cbar.update_ticks()
        return cbar

    def apply_axes(self, ax: Any) -> None:
        """Apply the surface of the style to an axes that was not created while the style was in effect."""
        color = self.rc.get("axes.facecolor")
        if color is not None:
            ax.set_facecolor(color)

    def apply(self, fig: Any) -> None:
        """Apply the surface of the style to a figure and its axes."""
        color = self.rc.get("figure.facecolor")
        if color is not None:
            fig.patch.set_facecolor(color)
        for ax in fig.axes:
            self.apply_axes(ax)


# ---------------------------------------------------------------------- helpers


def _cycle(cmap: Colormap, n: int) -> list[str]:
    """``n`` colors of a qualitative colormap, repeated if it has fewer, or spread over a continuous one."""
    colors = getattr(cmap, "colors", None)
    if colors is not None:
        return [to_hex(colors[i % len(colors)]) for i in range(n)]
    return [to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]


def _log_ticks(vmin: float, vmax: float) -> list[float]:
    lo, hi = math.floor(math.log10(vmin)), math.ceil(math.log10(vmax))
    ticks = [m * 10.0**e for e in range(lo, hi + 1) for m in (1, 2, 5)]
    inside = [t for t in ticks if vmin * (1 - 1e-9) <= t <= vmax * (1 + 1e-9)]
    return inside or [vmin, vmax]


def _si(x: float) -> str:
    """Number with a suffix: ``1M``, ``2.5k``, ``850``."""
    for factor, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(x) >= factor:
            return f"{x / factor:.3g}{suffix}"
    return f"{x:.3g}"


def _mix(color: str, other: str, share: float) -> str:
    a, b = np.array(to_rgb(color)), np.array(to_rgb(other))
    r, g, b_ = (float(v) for v in a * (1 - share) + b * share)
    return to_hex((r, g, b_))


def _light() -> PlotStyle:
    surface, edge, text, spine = "#eef1f4", "#dde4ea", "#222222", "#7a848c"
    tab10 = [_mix(c, surface, 0.15) for c in mpl.colormaps["tab10"].colors]  # type: ignore[attr-defined]
    return PlotStyle(
        rc={
            "figure.facecolor": surface,
            "axes.facecolor": surface,
            "savefig.facecolor": surface,
            "text.color": text,
            "axes.labelcolor": text,
            "axes.edgecolor": spine,
            "xtick.color": text,
            "ytick.color": text,
            "grid.color": "#cfd6dc",
            "legend.facecolor": surface,
            "legend.edgecolor": spine,
        },
        palettes=Palettes(sequential="YlGnBu", sequential_range=(0.22, 1.0), diverging="RdYlBu_r", categorical=tab10),
        channels={"edge": Channel(value=edge), "linewidth": Channel(value=0.8)},
        color_scale="auto",
        tick_format="si",
        colorbar_shrink=0.8,
        colorbar_outline_color=spine,
        colorbar_outline_width=1.0,
    )


def _dark() -> PlotStyle:
    surface, edge, text, spine = "#1b1f24", "#2f363d", "#e6e9ec", "#6b747c"
    return PlotStyle(
        rc={
            "figure.facecolor": surface,
            "axes.facecolor": surface,
            "savefig.facecolor": surface,
            "text.color": text,
            "axes.labelcolor": text,
            "axes.edgecolor": spine,
            "xtick.color": text,
            "ytick.color": text,
            "grid.color": "#39414a",
            "legend.facecolor": surface,
            "legend.edgecolor": spine,
        },
        palettes=Palettes(
            sequential="viridis", sequential_range=(0.25, 1.0), diverging="RdYlBu_r", categorical="tab10"
        ),
        channels={"edge": Channel(value=edge), "linewidth": Channel(value=0.8)},
        color_scale="auto",
        tick_format="si",
        colorbar_shrink=0.8,
        colorbar_outline_color=spine,
        colorbar_outline_width=1.0,
    )


# ---------------------------------------------------------------------- registry and selection

_BUILTIN = {"default": PlotStyle.default, "dark": PlotStyle.dark, "classic": PlotStyle.classic}
_registry: dict[str, PlotStyle] = {}
_session: PlotStyle | None = None
_context: contextvars.ContextVar[PlotStyle | None] = contextvars.ContextVar("carto_flow_plot_style", default=None)


def register_style(name: str, style: PlotStyle, *, overwrite: bool = False) -> None:
    """Register ``style`` under ``name``, so that ``style="name"`` selects it.

    ``"default"``, ``"dark"`` and ``"classic"`` cannot be replaced; another name that exists raises unless
    ``overwrite`` is true.
    """
    if not isinstance(style, PlotStyle):
        raise TypeError(f"style must be a PlotStyle, got {type(style).__name__}")
    if name in _BUILTIN:
        raise ValueError(f"{name!r} is a built-in style and cannot be replaced")
    if name in _registry and not overwrite:
        raise ValueError(f"a style named {name!r} is registered; pass overwrite=True to replace it")
    _registry[name] = style


def _lookup(style: PlotStyle | str) -> PlotStyle:
    if isinstance(style, PlotStyle):
        return style
    if isinstance(style, str):
        if style in _BUILTIN:
            return _BUILTIN[style]()
        if style in _registry:
            return _registry[style]
        raise ValueError(f"no style named {style!r}; available: {sorted([*_BUILTIN, *_registry])}")
    raise TypeError(f"style must be a PlotStyle, a registered name or None, got {type(style).__name__}")


def get_style(name: str | None = None) -> PlotStyle:
    """The registered style ``name``, or the style in effect (block, session, then ``default``) if ``None``."""
    return resolve_style(name)


def set_style(style: PlotStyle | str | None) -> None:
    """Set the style of the session (process-wide, like ``matplotlib.pyplot.style.use``); ``None`` resets it."""
    global _session
    _session = None if style is None else _lookup(style)


@contextlib.contextmanager
def style_context(style: PlotStyle | str) -> Iterator[PlotStyle]:
    """Context manager that sets the style for a block of code (thread and async safe)."""
    resolved = _lookup(style)
    token = _context.set(resolved)
    try:
        yield resolved
    finally:
        _context.reset(token)


def resolve_style(style: PlotStyle | str | None = None) -> PlotStyle:
    """The style of a call: the argument, else the block, else the session, else ``default``."""
    if style is not None:
        return _lookup(style)
    return _context.get() or _session or PlotStyle.default()


def styled(func):
    """Decorator that adds a ``style`` keyword to a plot function.

    The function runs with the rcParams of the style, so the figure and axes it creates, and the text it draws, have the
    look of the style; an axes passed in as ``ax`` gets the surface of the style.
    """
    signature = inspect.signature(func)
    has_ax = "ax" in signature.parameters

    @functools.wraps(func)
    def wrapper(*args, style: PlotStyle | str | None = None, **kwargs):
        resolved = resolve_style(style)
        with resolved.rc_context():
            if has_ax:
                ax = signature.bind_partial(*args, **kwargs).arguments.get("ax")
                if ax is not None:
                    resolved.apply_axes(ax)
            return func(*args, **kwargs)

    params = [p for p in signature.parameters.values() if p.kind is not inspect.Parameter.VAR_KEYWORD]
    var_kw = [p for p in signature.parameters.values() if p.kind is inspect.Parameter.VAR_KEYWORD]
    params.append(
        inspect.Parameter("style", inspect.Parameter.KEYWORD_ONLY, default=None, annotation="PlotStyle | str | None")
    )
    wrapper.__signature__ = signature.replace(parameters=[*params, *var_kw])  # type: ignore[attr-defined]
    return wrapper


def rc_wrapped(style: PlotStyle, func):
    """``func`` run with the rcParams of ``style``, for the callbacks of an animation."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with style.rc_context():
            return func(*args, **kwargs)

    return wrapper
