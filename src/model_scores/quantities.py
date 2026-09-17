"""Named statistics, and the panel of them that is scored together."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = ["Quantity", "Panel", "MEAN", "MEDIAN", "Q90", "Q95", "Q99", "STDEV",
           "CATALOG", "DEFAULT_PANEL", "panel_from_names"]


@dataclass(frozen=True)
class Quantity:
    """A named statistic of a sample.

    ``fn`` must accept ``(array, axis)`` and reduce along ``axis``, so that we can
    evaluate it on a whole resample matrix in one call. An ``axis`` of None means
    reduce the flat array, which is the convention numpy uses.
    """

    name: str
    fn: Callable[[np.ndarray, int | None], np.ndarray]
    units: str = ""

    def __call__(self, a: np.ndarray, axis: int | None = None):
        return self.fn(a, axis)


def _quantile(p: float) -> Callable[[np.ndarray, int | None], np.ndarray]:
    return lambda a, axis: np.quantile(a, p, axis=axis)


MEAN = Quantity("mean", lambda a, axis: np.mean(a, axis=axis))
MEDIAN = Quantity("median", lambda a, axis: np.median(a, axis=axis))
Q90 = Quantity("q90", _quantile(0.90))
Q95 = Quantity("q95", _quantile(0.95))
Q99 = Quantity("q99", _quantile(0.99))
STDEV = Quantity("stdev", lambda a, axis: np.std(a, axis=axis, ddof=1))

CATALOG: dict[str, Quantity] = {q.name: q for q in (MEAN, MEDIAN, Q90, Q95, Q99, STDEV)}


@dataclass(frozen=True)
class Panel:
    """The quantities scored together, all of which must pass.

    We take the mean, the median and the 90th percentile by default. The mean
    moves with the bulk, the median resists a single wild run, and the upper
    percentile is where a model that gets the body right and the tail wrong shows
    itself. Add ``Q99`` when the tail is the point of the comparison, and ``STDEV``
    when dispersion is part of what the model claims to reproduce.
    """

    quantities: tuple[Quantity, ...] = (MEAN, MEDIAN, Q90)

    def __iter__(self):
        return iter(self.quantities)

    def __len__(self) -> int:
        return len(self.quantities)

    def __getitem__(self, i) -> Quantity:
        return self.quantities[i]

    @property
    def first(self) -> Quantity:
        return self.quantities[0]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(q.name for q in self.quantities)


#: The panel every entry point falls back to. The class is frozen, so sharing one
#: instance as a default argument is safe (and saves constructing a new one on
#: every call).
DEFAULT_PANEL = Panel()


def panel_from_names(spec: str) -> Panel:
    """Build a panel from a comma separated list such as ``"mean,median,q90"``."""
    try:
        return Panel(tuple(CATALOG[n.strip()] for n in spec.split(",") if n.strip()))
    except KeyError as e:
        raise ValueError(f"unknown quantity {e}; choose from {', '.join(CATALOG)}") from None
