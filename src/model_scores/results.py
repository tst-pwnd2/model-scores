"""What the procedure returns.

The interval is the product. The verdict, the certifiable margin and the printed
line are all readings of it. This is the point of R2: a reader who disagrees with
our choice of margin can still use everything here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .verdicts import NO_MARGIN, PASS

__all__ = ["Interval", "QuantityResult", "PairResult"]


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval, in the metric's own units."""

    point: float
    lo: float
    hi: float
    confidence: float

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def scaled(self, by: float) -> Interval:
        return Interval(self.point / by, self.lo / by, self.hi / by, self.confidence)

    def __str__(self) -> str:
        return f"{self.point:+.4g} [{self.lo:+.4g}, {self.hi:+.4g}]"


@dataclass(frozen=True)
class QuantityResult:
    """One quantity's outcome."""

    quantity: str
    reference: float
    model: float
    interval: Interval                      # model minus reference, absolute
    relative: Interval | None            # the same, as a fraction of |reference|
    margin: float | None                 # absolute, in the metric's units
    verdict: str

    @property
    def certifiable_margin(self) -> float:
        """The tightest margin at which this pair would have passed.

        The interval must fit strictly inside the margin, so this is the larger of
        the two endpoint magnitudes. Set against the margin that was applied, it
        says how much room was left (resp., how much was wanted).
        """
        return max(abs(self.interval.lo), abs(self.interval.hi))

    @property
    def relative_certifiable_margin(self) -> float:
        base = abs(self.reference)
        return self.certifiable_margin / base if base > 1e-300 else float("nan")

    def line(self) -> str:
        if self.relative is not None:
            v, mul, unit = self.relative, 100.0, "%"
            m = ("" if self.margin is None else
                 f"   margin {self.margin / max(abs(self.reference), 1e-300) * 100:.1f}%")
        else:
            v, mul, unit = self.interval, 1.0, ""
            m = "" if self.margin is None else f"   margin {self.margin:.4g}"
        return (f"{self.quantity:>8s}  {v.point * mul:+8.2f}{unit} "
                f"[{v.lo * mul:+8.2f}, {v.hi * mul:+8.2f}]{unit}{m}   {self.verdict}")


@dataclass
class PairResult:
    """The panel outcome for one comparison, plus its diagnostics."""

    key: tuple = ()
    n_reference: int = 0
    n_model: int = 0
    quantities: dict[str, QuantityResult] = field(default_factory=dict)
    verdict: str = PASS
    kl: float = float("nan")
    kl_floor: float = float("nan")
    variance_share_model: float = float("nan")
    near_zero: bool = False

    def __getitem__(self, name: str) -> QuantityResult:
        return self.quantities[name]

    def __iter__(self):
        return iter(self.quantities.values())

    @property
    def label(self) -> str:
        return ".".join(str(k) for k in self.key) or "pair"

    @property
    def kl_excess(self) -> float:
        """KL above what a correct model of this submission size would score."""
        return self.kl - self.kl_floor

    @property
    def has_margin(self) -> bool:
        return self.verdict != NO_MARGIN

    def report(self) -> str:
        head = (f"{self.label}   n_ref={self.n_reference} "
                f"n_model={self.n_model}   {self.verdict}")
        body = ["    " + q.line() for q in self.quantities.values()]
        tail = (f"     KL {self.kl:.3f} bits, floor {self.kl_floor:.3f}, "
                f"excess {self.kl_excess:+.3f}   "
                f"model share of variance {self.variance_share_model * 100:.1f}%")
        return "\n".join([head] + body + [tail])

    def to_dict(self) -> dict:
        def iv(i):
            return None if i is None else dict(point=i.point, lo=i.lo, hi=i.hi,
                                               confidence=i.confidence)
        return dict(
            key=list(self.key), n_reference=self.n_reference, n_model=self.n_model,
            verdict=self.verdict, kl=self.kl, kl_floor=self.kl_floor,
            kl_excess=self.kl_excess, near_zero=self.near_zero,
            variance_share_model=self.variance_share_model,
            quantities={
                n: dict(reference=q.reference, model=q.model, absolute=iv(q.interval),
                        relative=iv(q.relative), margin=q.margin, verdict=q.verdict,
                        certifiable_margin=q.certifiable_margin)
                for n, q in self.quantities.items()},
        )
