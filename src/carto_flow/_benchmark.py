"""Shared benchmarking base classes for carto_flow submodules."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

__all__ = ["BenchmarkBase", "BenchmarkRuns"]


@dataclasses.dataclass
class BenchmarkBase:
    """Self-contained per-run benchmark record.

    Subclass this in each submodule, adding phase-timing fields as float
    attributes whose names end in ``_s``.  ``to_dict()`` automatically
    prefixes those with ``t_`` and rounds them; all other fields are kept
    as-is.

    Common fields (shared by all submodule benchmarks):

    Timing (names end in ``_s`` — serialised as ``t_<name>``):
        setup_s   — pre-loop initialisation time
        other_s   — unattributed loop time
        total_s   — full wall-clock time

    Metadata:
        niterations    — iterations completed
        status         — algorithm status string
        mean_error_pct — mean area error as a percentage
        max_error_pct  — maximum area error as a percentage
    """

    # Timing fields (end in _s → serialised as t_<name>)
    setup_s: float = 0.0
    other_s: float = 0.0
    total_s: float = 0.0

    # Algorithm metadata
    niterations: int = 0
    status: str | None = None
    mean_error_pct: float | None = None
    max_error_pct: float | None = None

    def to_dict(self, decimals: int = 3) -> dict:
        """Return a flat dict representation.

        Float fields whose names end in ``_s`` are serialised as
        ``t_<name>`` and rounded to *decimals* places.  All other fields
        are kept as-is.
        """
        result = {}
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            if f.name.endswith("_s") and val is not None:
                result[f"t_{f.name}"] = round(val, decimals)
            else:
                result[f.name] = val
        return result


class BenchmarkRuns:
    """Collects :class:`BenchmarkBase` instances across benchmark rounds.

    Implements the sequence interface (``__len__``, ``__iter__``,
    ``__getitem__``).  Works with any ``BenchmarkBase`` subclass so it is
    reusable across submodules.

    Example usage in a pytest-benchmark test::

        runs = BenchmarkRuns()

        def run():
            result = morph_gdf(gdf, "col", options=options)
            runs.add(result.benchmark)
            return result

        benchmark.pedantic(run, ...)
        benchmark.extra_info.update(runs.to_dict())

    The resulting ``extra_info`` is a dict of lists, directly usable as
    ``pd.DataFrame(benchmark.extra_info)``.
    """

    def __init__(self) -> None:
        self._runs: list[BenchmarkBase] = []

    def add(self, b: BenchmarkBase | None) -> None:
        """Append *b* to the collection; silently ignores ``None``."""
        if b is not None:
            self._runs.append(b)

    def __bool__(self) -> bool:
        return bool(self._runs)

    def __len__(self) -> int:
        return len(self._runs)

    def __iter__(self) -> Iterator[BenchmarkBase]:
        return iter(self._runs)

    def __getitem__(self, index: int | slice) -> BenchmarkBase | BenchmarkRuns:
        if isinstance(index, slice):
            result = BenchmarkRuns()
            result._runs = self._runs[index]
            return result
        return self._runs[index]

    def to_dict(self) -> dict:
        """Return a dict-of-lists suitable for ``benchmark.extra_info`` or ``pd.DataFrame()``.

        Each key maps to a list with one value per collected run, in
        insertion order.  Returns an empty dict when no runs have been
        added.
        """
        if not self._runs:
            return {}
        keys = list(self._runs[0].to_dict())
        result = {k: [b.to_dict()[k] for b in self._runs] for k in keys}
        result["round"] = list(range(len(self._runs)))
        return result
