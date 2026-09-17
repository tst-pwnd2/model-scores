"""The four outcomes a comparison can reach, and the rule by which we combine them.

The outcomes sit on two axes rather than one. Three of them report what the data
say about the model, and the fourth reports a question we declined to pose. The
precedence below keeps that distinction visible.
"""

from __future__ import annotations

__all__ = ["PASS", "FAIL", "NOT_CERTIFIABLE", "NO_MARGIN", "SEVERITY", "worst"]

#: The whole interval lies inside the margin.
PASS = "PASS"
#: The whole interval lies outside it, so the difference exceeds the margin.
FAIL = "FAIL"
#: The interval straddles the boundary, i.e., the data do not decide the question
#: at this margin.
NOT_CERTIFIABLE = "NOT CERTIFIABLE"
#: A relative margin requires a reference statistic we can distinguish from zero.
#: Where it is indistinguishable, and where the program has supplied no absolute
#: margin in its place, we report the pair here rather than invent one.
NO_MARGIN = "NO MARGIN"

#: Precedence when a panel disagrees. NO MARGIN outranks a substantive verdict,
#: since the question was never posed, and reporting it as a failure would credit
#: the procedure with an answer it did not produce.
SEVERITY: dict[str, int] = {PASS: 0, NOT_CERTIFIABLE: 1, FAIL: 2, NO_MARGIN: 3}

ORDER = (PASS, FAIL, NOT_CERTIFIABLE, NO_MARGIN)


def worst(verdicts, default: str = PASS) -> str:
    """The intersection-union rule: every member must pass for the panel to pass."""
    return max(verdicts, key=lambda v: SEVERITY[v], default=default)
