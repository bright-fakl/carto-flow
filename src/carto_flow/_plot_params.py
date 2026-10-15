"""Shared helpers for the parameter vocabulary of the plot functions.

``deprecated_params`` keeps old parameter names working as aliases. ``classify_string`` decides whether a string
parameter of a channel is a column, a built-in quantity or a color.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Collection, Mapping
from typing import Any


def deprecated_params(**renames: str | tuple[str, ...]):
    """Accept the old names ``renames`` (new name -> old name or names) of a function as deprecated aliases.

    An old name given in the call is passed on under the new name with a ``DeprecationWarning``. Giving an old name
    and its new name together raises ``TypeError``.
    """
    mapping = {new: (old,) if isinstance(old, str) else tuple(old) for new, old in renames.items()}

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for new, olds in mapping.items():
                for old in olds:
                    if old not in kwargs:
                        continue
                    if new in kwargs:
                        raise TypeError(f"{func.__name__}() got both '{old}' (deprecated) and '{new}'")
                    warnings.warn(
                        f"{func.__name__}(): '{old}' is deprecated; use '{new}'",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecated_init_params(**renames: str | tuple[str, ...]):
    """Class decorator for dataclasses: accept the old field names of ``renames`` (new name -> old name or names).

    Old names are accepted by ``__init__`` and as attributes, with a ``DeprecationWarning``.
    """
    mapping = {new: (old,) if isinstance(old, str) else tuple(old) for new, old in renames.items()}

    def decorator(cls):
        original = cls.__init__

        @functools.wraps(original)
        def __init__(self, *args, **kwargs):
            for new, olds in mapping.items():
                for old in olds:
                    if old not in kwargs:
                        continue
                    if new in kwargs:
                        raise TypeError(f"{cls.__name__}() got both '{old}' (deprecated) and '{new}'")
                    warnings.warn(
                        f"{cls.__name__}: '{old}' is deprecated; use '{new}'", DeprecationWarning, stacklevel=2
                    )
                    kwargs[new] = kwargs.pop(old)
            original(self, *args, **kwargs)

        cls.__init__ = __init__

        def alias(old: str, new: str) -> property:
            def get(self):
                warnings.warn(f"{cls.__name__}.{old} is deprecated; use {new}", DeprecationWarning, stacklevel=2)
                return getattr(self, new)

            def set_(self, value):
                warnings.warn(f"{cls.__name__}.{old} is deprecated; use {new}", DeprecationWarning, stacklevel=2)
                setattr(self, new, value)

            return property(get, set_)

        for new, olds in mapping.items():
            for old in olds:
                setattr(cls, old, alias(old, new))
        return cls

    return decorator


def _is_color(value: str) -> bool:
    import matplotlib.colors as mcolors

    try:
        return bool(mcolors.is_color_like(value))
    except Exception:
        return False


def classify_string(
    value: str,
    columns: Collection[str],
    builtins: Mapping[str, Any] | Collection[str] = (),
    *,
    stacklevel: int = 3,
) -> str:
    """Resolve a string parameter of a channel: ``"column"``, ``"builtin"`` or ``"color"``.

    A column of the data wins over a built-in quantity, which wins over a color. A ``UserWarning`` is issued when the
    string matches more than one kind. Raises ``ValueError`` when it matches none.
    """
    kinds = []
    if value in columns:
        kinds.append("column")
    if value in builtins:
        kinds.append("builtin")
    if _is_color(value):
        kinds.append("color")
    if not kinds:
        available = sorted({*map(str, columns), *map(str, builtins)})
        raise ValueError(f"'{value}' is neither a column, a built-in quantity nor a color. Available: {available}")
    if len(kinds) > 1:
        warnings.warn(
            f"'{value}' is a {' and a '.join(kinds)}; using the {kinds[0]}. Pass the values (for example "
            "gdf['name']) to bind data, or a color tuple (matplotlib.colors.to_rgb('name')) for a color.",
            UserWarning,
            stacklevel=stacklevel,
        )
    return kinds[0]
