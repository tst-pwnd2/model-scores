"""R1 and R2.

We score the panel as an intersection-union test, i.e., every member must clear
its own margin for the pair to pass. Note that such a test holds level alpha
overall, so we owe no multiplicity correction for the size of the panel.

Every quantity returns an interval, and its verdict is a reading of that interval
rather than a substitute for it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .bootstrap import DEFAULT_ALPHA, DEFAULT_BOOTSTRAP, DEFAULT_SEED, clean, resample, rng_from
from .divergence import DEFAULT_KL_BINS, kl_bits, kl_floor
from .quantities import DEFAULT_PANEL, Panel, Quantity
from .results import Interval, PairResult, QuantityResult
from .verdicts import FAIL, NO_MARGIN, NOT_CERTIFIABLE, ORDER, PASS, worst

__all__ = ["difference_interval", "classify", "score", "score_all", "tally", "summary",
           "has_relative_margin", "absolute_for", "scope_matches", "MIN_REFERENCE",
           "MIN_MODEL", "DEFAULT_ZERO_GUARD"]

#: How close to zero the reference statistic may sit, in standard errors of the
#: reference mean, before a relative margin stops meaning anything.
DEFAULT_ZERO_GUARD = 5.0

#: A pair thinner than this on either side resolves nothing, so we do not score
#: it: the result would describe the sample size rather than the model.
MIN_REFERENCE = 30
MIN_MODEL = 8


def has_relative_margin(reference: np.ndarray, zero_guard: float = DEFAULT_ZERO_GUARD) -> bool:
    """Whether a relative margin is meaningful for this reference sample.

    Lives here, and is used both by :func:`score` and by the callers that want
    the answer without a model, so that the two cannot drift apart.
    """
    reference = clean(reference)
    if reference.size < 2:
        return False
    se_mean = reference.std(ddof=1) / math.sqrt(reference.size)
    return abs(reference.mean()) >= zero_guard * se_mean


def scope_matches(scope: str, key: Sequence) -> bool:
    """Whether a dotted scope selects this key.

    A scope is matched segment by segment against the key, which is
    ``(metric, series, mode, window)``. ``*`` matches one segment and a scope
    shorter than the key matches on its prefix, so ``integrity`` takes every
    integrity pair and ``integrity.*.independent`` takes only the independent
    ones. A bare ``*`` takes everything.
    """
    if scope == "*":
        return True
    parts = scope.split(".")
    if len(parts) > len(key):
        return False
    # strict=False on purpose: a short scope is a prefix, which is the point.
    return all(p == "*" or p == str(k) for p, k in zip(parts, key, strict=False))


def specificity(scope: str) -> tuple[int, int]:
    """How particular a scope is: named segments first, then length."""
    if scope == "*":
        return (0, 0)
    parts = scope.split(".")
    return (sum(1 for p in parts if p != "*"), len(parts))


def absolute_for(key: Sequence, table: dict[str, dict[str, float]] | None) -> dict[str, float]:
    """The absolute margins that apply to one key.

    Every matching scope contributes, least particular first, so a general
    statement can be narrowed by a particular one. Scoping matters more than it
    looks: the same metric name covers a cumulative series sitting near one and
    an independent series sitting at zero, and a tolerance in units that suits
    the second is a far stricter test on the first.
    """
    if not table:
        return {}
    merged: dict[str, float] = {}
    for scope in sorted(table, key=specificity):
        if scope_matches(scope, key):
            merged.update(table[scope])
    return merged


def difference_interval(reference: np.ndarray, model: np.ndarray, quantity: Quantity,
                        reps: int, alpha: float, rng) -> Interval:
    """Two-sample percentile bootstrap interval for model minus reference.

    We resample each side at its own size, so the interval widens correctly when
    either corpus is small. We take the plug-in difference as the point estimate
    (resp., the bootstrap percentiles as the endpoints), which keeps the estimate
    free of resampling bias.
    """
    boot = (resample(model, quantity, reps, rng)
            - resample(reference, quantity, reps, rng))
    lo, hi = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])
    point = float(quantity(model, None)) - float(quantity(reference, None))
    return Interval(point, float(lo), float(hi), 1 - 2 * alpha)


def classify(interval: Interval, margin: float) -> str:
    """Read a verdict off an interval. Containment is strict either way."""
    if interval.lo > -margin and interval.hi < margin:
        return PASS
    if interval.lo > margin or interval.hi < -margin:
        return FAIL
    return NOT_CERTIFIABLE


def score(reference, model, panel: Panel = DEFAULT_PANEL, margin: float = 0.10,
          absolute_margins: dict[str, float] | None = None,
          key: tuple = (), alpha: float = DEFAULT_ALPHA,
          reps: int = DEFAULT_BOOTSTRAP, seed=DEFAULT_SEED,
          zero_guard: float = DEFAULT_ZERO_GUARD, kl_bins: int = DEFAULT_KL_BINS,
          log_bins: bool = False, diagnostics: bool = True) -> PairResult:
    """Score one model sample against one reference sample on a panel.

    ``margin`` is relative, namely a fraction of the reference statistic. Supply
    ``absolute_margins`` keyed by quantity name to override it with a margin in the
    metric's own units, which is the right form whenever the program has stated a
    tolerance that does not scale with the reference.

    ``zero_guard`` sets how close to zero the reference statistic may sit, in
    standard errors of the reference mean, before we refuse a relative margin.
    Inventing an absolute margin from the reference's own spread at that point
    would reintroduce exactly the circularity this procedure exists to avoid. As
    such, we return the pair as NO MARGIN and ask the program for a number.

    We clean both samples of non-finite values first, and raise ValueError if
    either ends up empty.
    """
    rng = rng_from(seed)
    reference, model = clean(reference), clean(model)
    if reference.size == 0 or model.size == 0:
        raise ValueError(f"{key or 'pair'}: empty sample after removing non-finite values")

    absolute_margins = absolute_margins or {}
    near_zero = not has_relative_margin(reference, zero_guard)

    result = PairResult(key=key, n_reference=int(reference.size),
                        n_model=int(model.size), near_zero=bool(near_zero))

    for quantity in panel:
        base = float(quantity(reference, None))
        interval = difference_interval(reference, model, quantity, reps, alpha, rng)
        delta = absolute_margins.get(quantity.name)
        if delta is None:
            if near_zero or abs(base) <= 1e-300:
                result.quantities[quantity.name] = QuantityResult(
                    quantity.name, base, float(quantity(model, None)), interval,
                    None, None, NO_MARGIN)
                continue
            delta = margin * abs(base)
        # A near-zero base makes a relative reading meaningless, which is the
        # whole reason the pair needed an absolute margin. Do not offer one.
        relative = (interval.scaled(abs(base))
                    if abs(base) > 1e-300 and not near_zero else None)
        result.quantities[quantity.name] = QuantityResult(
            quantity.name, base, float(quantity(model, None)), interval, relative,
            float(delta), classify(interval, float(delta)))

    result.verdict = worst(q.verdict for q in result.quantities.values())

    if diagnostics:
        result.kl = kl_bits(model, reference, kl_bins, log_bins)
        result.kl_floor = kl_floor(reference, int(model.size), rng, bins=kl_bins,
                                   log_bins=log_bins)
        from .resolution import variance_share
        result.variance_share_model = variance_share(reference, int(model.size),
                                                     panel.first, reps, rng)
    return result


def score_all(reference: dict[tuple, np.ndarray], model: dict[tuple, np.ndarray],
              panel: Panel = DEFAULT_PANEL, margin: float = 0.10,
              min_reference: int = MIN_REFERENCE, min_model: int = MIN_MODEL,
              absolute: dict[str, dict[str, float]] | None = None,
              **kw) -> list[PairResult]:
    """Score every key both mappings carry.

    We skip a key that is too thin on either side, since a handful of runs resolves
    nothing and would only add pairs that are, by construction, not certifiable.

    ``absolute`` carries margins stated in a metric's own units, keyed by a
    dotted scope over the key and then by quantity. Those are what a pair whose
    reference statistic sits at zero needs, and they have to come from the
    program rather than from the reference's own spread.
    """
    out = []
    for key in sorted(set(reference) & set(model)):
        r, m = clean(reference[key]), clean(model[key])
        if r.size < min_reference or m.size < min_model:
            continue
        out.append(score(r, m, panel, margin, key=key,
                         absolute_margins=absolute_for(key, absolute), **kw))
    return out


def tally(results: Sequence[PairResult]) -> dict[str, int]:
    counts = {k: 0 for k in ORDER}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    return counts


def summary(results: Sequence[PairResult]) -> str:
    """The whole-campaign read, namely the tally, the divergence ranking and the
    margins the campaign was able to certify.
    """
    if not results:
        return "no comparable pairs"
    n = len(results)
    names = ", ".join(results[0].quantities)
    lines = [f"{n} pairs, panel of {names}, all required to pass", ""]
    for verdict, count in tally(results).items():
        lines.append(f"  {verdict:>16s}  {count:4d}  {count / n * 100:5.1f}%")

    scored = [r for r in results if math.isfinite(r.kl_excess)]
    if scored:
        excess = np.array([r.kl_excess for r in scored])
        lines += ["", f"KL above floor, bits, over {len(scored)} pairs: "
                      f"median {np.median(excess):+.3f}, "
                      f"90th {np.quantile(excess, 0.9):+.3f}, max {excess.max():+.3f}",
                  "", "  worst by excess:"]
        for r in sorted(scored, key=lambda x: -x.kl_excess)[:10]:
            lines.append(f"    {r.kl_excess:+7.3f}  {r.label:<58s}  {r.verdict}")

    # A relative summary of a pair whose reference sits at zero would be the
    # nonsense this procedure declines to produce, absolute margin or not.
    have = [r for r in results if r.has_margin and not r.near_zero]
    if have:
        first = results[0].quantities and next(iter(results[0].quantities))
        margins = np.array([r[first].relative_certifiable_margin for r in have
                            if math.isfinite(r[first].relative_certifiable_margin)])
        if margins.size:
            lines += ["", f"certifiable margin on the {first}, relative: "
                          f"median {np.median(margins) * 100:.1f}%, "
                          f"90th {np.quantile(margins, 0.9) * 100:.1f}%"]
        shares = np.array([r.variance_share_model for r in have
                           if math.isfinite(r.variance_share_model)])
        if shares.size:
            lines.append("model share of the variance of the difference: "
                         f"median {np.median(shares) * 100:.1f}%")
    return "\n".join(lines)
