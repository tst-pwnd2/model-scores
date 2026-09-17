"""R3 and R4. We compute both from the reference alone, with no model involved.

That is the whole point of them. A correct model has the reference's distribution
by definition, so we can take both standard errors from the reference: one at its
own size, one at the submission size. The resolution can therefore be published
before anyone submits anything. That is to say, the margin can be agreed in
advance, and a margin finer than the resolution asks a question the data cannot
answer.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable

from .bootstrap import (
    DEFAULT_ALPHA,
    DEFAULT_BOOTSTRAP,
    DEFAULT_SEED,
    clean,
    rng_from,
    standard_error,
)
from .quantities import DEFAULT_PANEL, MEDIAN, Panel, Quantity

__all__ = ["resolution", "panel_resolution", "resolution_curve", "variance_share"]


def _z(alpha: float, power: float) -> float:
    n = statistics.NormalDist()
    return n.inv_cdf(1 - alpha) + n.inv_cdf(power)


def resolution(reference, n_model: int, quantity: Quantity = MEDIAN,
               alpha: float = DEFAULT_ALPHA, power: float = 0.95,
               reps: int = DEFAULT_BOOTSTRAP, seed=DEFAULT_SEED,
               relative: bool = True) -> float:
    """The finest margin a correct model of ``n_model`` runs could certify.

    The margin must exceed ``(z_{1-alpha} + z_{power}) * SE`` for a correct model to
    clear it at the stated confidence and power. At the defaults (95 percent
    confidence, 95 percent power) that constant is 3.605.

    Returns a fraction of the reference statistic when ``relative``, otherwise
    the margin in the metric's own units. Returns nan when the reference
    statistic is too close to zero for a relative margin to mean anything.
    """
    rng = rng_from(seed)
    reference = clean(reference)
    se_ref = standard_error(reference, quantity, reference.size, reps, rng)
    se_mod = standard_error(reference, quantity, n_model, reps, rng)
    delta = _z(alpha, power) * math.hypot(se_ref, se_mod)
    if not relative:
        return delta
    base = abs(float(quantity(reference, None)))
    return delta / base if base > 1e-300 else float("nan")


def panel_resolution(reference, n_model: int, panel: Panel = DEFAULT_PANEL,
                     **kw) -> dict[str, float]:
    """``resolution`` for every quantity in the panel, plus the binding one.

    An intersection-union test is only as fine as its coarsest member, so the entry
    under ``"panel"`` is the margin to publish. In particular, upper percentiles
    resolve more coarsely than the mean, which is the price of looking at the tail.
    A margin
    chosen from the mean alone therefore leaves the tail permanently not
    certifiable, a point to settle before agreeing to a number.
    """
    out = {q.name: resolution(reference, n_model, q, **kw) for q in panel}
    finite = [v for v in out.values() if math.isfinite(v)]
    out["panel"] = max(finite) if finite else float("nan")
    return out


def resolution_curve(reference, sizes: Iterable[int], quantity: Quantity = MEDIAN,
                     **kw) -> list[tuple[int, float]]:
    """``resolution`` across a range of submission sizes. This is the R4 exhibit.

    The curve flattens once the model side stops dominating the combined standard
    error. Where it flattens is the point past which more model runs buy nothing,
    and past which only more trials would help.
    """
    return [(int(n), resolution(reference, int(n), quantity, **kw)) for n in sizes]


def variance_share(reference, n_model: int, quantity: Quantity = MEDIAN,
                   reps: int = DEFAULT_BOOTSTRAP, seed=DEFAULT_SEED) -> float:
    """The model side's share of the variance of the difference.

    Above one half, we are the reason the comparison is blunt, and submitting more
    runs sharpens it without asking anything of the evaluator. Below one half, the
    reference binds, i.e., more model runs will not move the verdict.
    """
    rng = rng_from(seed)
    reference = clean(reference)
    v_ref = standard_error(reference, quantity, reference.size, reps, rng) ** 2
    v_mod = standard_error(reference, quantity, n_model, reps, rng) ** 2
    total = v_ref + v_mod
    return v_mod / total if total > 0 else float("nan")
