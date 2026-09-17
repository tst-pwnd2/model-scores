"""Resampling machinery.

Everything stochastic in the package lives here. As such, pinning a seed at this
boundary pins the whole procedure, which is what makes a reported number
reproducible by whoever reads it.
"""

from __future__ import annotations

import numpy as np

from .quantities import Quantity

__all__ = ["DEFAULT_BOOTSTRAP", "DEFAULT_ALPHA", "DEFAULT_SEED",
           "rng_from", "clean", "resample", "standard_error"]

DEFAULT_BOOTSTRAP = 2000
DEFAULT_ALPHA = 0.05
DEFAULT_SEED = 20260917


def rng_from(seed) -> np.random.Generator:
    """Accept a generator or a seed, so callers can share a stream or pin one."""
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def clean(a) -> np.ndarray:
    """Flatten to float and drop non-finite values."""
    a = np.asarray(a, dtype=float).ravel()
    return a[np.isfinite(a)]


def resample(a: np.ndarray, quantity: Quantity, reps: int,
             rng: np.random.Generator, size: int | None = None) -> np.ndarray:
    """The bootstrap distribution of ``quantity`` over samples of ``size`` drawn from ``a``.

    ``size`` defaults to the size of ``a``. Passing a different one is how the
    resolution calculation asks what a submission of that size would look like were
    it drawn from the reference, which is, by definition, what a correct model is.
    """
    n = a.size if size is None else size
    idx = rng.integers(0, a.size, (reps, n))
    return np.asarray(quantity(a[idx], 1), dtype=float)


def standard_error(a: np.ndarray, quantity: Quantity, size: int, reps: int,
                   rng: np.random.Generator) -> float:
    """Bootstrap standard error of ``quantity`` at sample size ``size``."""
    return float(np.std(resample(a, quantity, reps, rng, size=size), ddof=1))
