"""The divergence diagnostic, computed the way hcs_detect computes it.

We report it against the floor a correct model of the same submission size
attains, rather than against zero. At a finite submission size that floor is a
large fraction of what any real model scores, so the raw number on its own is
somewhat misleading.

The diagnostic ranks disagreements. Note that it does not gate them. It inherits
both the reference dispersion and the submission size, and as such cannot carry a
margin that means one fixed thing across pairs, which is the same objection the
critique raises against scoring on a single standardised distance.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .bootstrap import DEFAULT_SEED, clean, rng_from

__all__ = ["DEFAULT_KL_BINS", "kl_divergence", "kl_bits", "kl_floor"]

DEFAULT_KL_BINS = 20


def kl_divergence(p_counts: Sequence[float], q_counts: Sequence[float],
                  eps: float = 1e-3) -> float:
    """KL(P || Q) in bits from two count histograms, with additive smoothing.

    Signature and smoothing follow hcs_detect.detectors.divergence, so a number
    here means what it means there.
    """
    p = np.asarray(p_counts, dtype=float) + eps
    q = np.asarray(q_counts, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log2(p / q)))


def kl_bits(model, reference, bins: int = DEFAULT_KL_BINS,
            log_bins: bool = False) -> float:
    """KL(model || reference) over shared bins spanning both samples.

    We take linear edges by default, since per-run rates and fractions are bounded.
    Set ``log_bins`` for the positive heavy-tailed quantities hcs_detect log-bins,
    e.g., inter-arrival times.
    """
    model, reference = clean(model), clean(reference)
    if model.size == 0 or reference.size == 0:
        return float("nan")
    lo = min(reference.min(), model.min())
    hi = max(reference.max(), model.max())
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return float("nan")
    if log_bins:
        lo = max(lo, 1e-9)
        edges = np.logspace(math.log10(lo), math.log10(hi * 1.000001), bins + 1)
    else:
        edges = np.linspace(lo, hi + (hi - lo) * 1e-6, bins + 1)
    return kl_divergence(np.histogram(model, bins=edges)[0],
                         np.histogram(reference, bins=edges)[0])


def kl_floor(reference, n_model: int, seed=DEFAULT_SEED, reps: int = 60,
             bins: int = DEFAULT_KL_BINS, log_bins: bool = False) -> float:
    """The KL a correct model of ``n_model`` runs would score against this reference.

    We draw the model from the reference's own empirical distribution, so the only
    thing left in the divergence is finite-sample noise. The floor is therefore a
    property of the reference and the submission size alone.
    """
    rng = rng_from(seed)
    reference = clean(reference)
    if reference.size == 0 or n_model <= 0:
        return float("nan")
    vals = [kl_bits(rng.choice(reference, n_model, replace=True), reference, bins, log_bins)
            for _ in range(reps)]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")
