"""Panel scoring for model-versus-trial comparison.

This package implements the procedure recommended in docs/tost-critique.pdf. We
replace a single standardised distance and a pass bit with four things:

  R1  A panel of named quantities, each scored against a margin in its own
      units, all of them required to pass. That is an intersection-union test,
      so it holds level alpha overall and owes no multiplicity correction.
      See ``quantities`` and ``scoring``.

  R2  An interval for every quantity, reported alongside the verdict. The
      interval is the product, and the verdict is a reading of it. See
      ``results``.

  R3  The achievable resolution, computed from the reference alone, with no
      model involved. A margin finer than the resolution asks a question the
      data cannot answer, and its failures therefore carry no information.
      See ``resolution``.

  R4  The variance share, which says which side of the comparison supplies the
      noise, and a projection of how the resolution tightens as the submission
      grows. Also in ``resolution``.

A divergence diagnostic sits beside the verdicts, in bits, computed as hcs_detect
computes it and scored against the floor a correct model of the same submission
size attains. It ranks disagreements. It does not gate them. See ``divergence``.

Library use:

    from model_scores import Panel, score, panel_resolution

    result = score(reference, model, Panel(), margin=0.10)
    print(result.verdict, result["median"].interval)
    print(panel_resolution(reference, n_model=40)["panel"])

Command line use:

    python3 -m model_scores REFERENCE.json MODEL_DIR --margin 0.10 --resolution 40,507

Layout:

    verdicts.py    the four outcomes and the intersection-union rule
    quantities.py  Quantity, Panel, and the catalog of statistics
    bootstrap.py   resampling, the only stochastic code in the package
    results.py     Interval, QuantityResult, PairResult
    divergence.py  the KL diagnostic and its floor
    resolution.py  R3 and R4, both model-free
    scoring.py     R1 and R2
    corpus.py      the CP3 file format, and the only part that knows it
    plots.py       the three figures, behind the optional plots extra
    cli.py         the command line

Requires numpy. The figures additionally want matplotlib, which the ``plots``
extra supplies. Nothing else outside the standard library. See README.md for the
uv workflow and the verdict table.
"""

from __future__ import annotations

from .bootstrap import DEFAULT_ALPHA, DEFAULT_BOOTSTRAP, DEFAULT_SEED
from .corpus import load_model, load_reference
from .divergence import DEFAULT_KL_BINS, kl_bits, kl_divergence, kl_floor
from .quantities import (
                         CATALOG,
                         MEAN,
                         MEDIAN,
                         Q90,
                         Q95,
                         Q99,
                         STDEV,
                         Panel,
                         Quantity,
                         panel_from_names,
)
from .resolution import panel_resolution, resolution, resolution_curve, variance_share
from .results import Interval, PairResult, QuantityResult
from .scoring import classify, difference_interval, score, score_all, summary, tally
from .verdicts import FAIL, NO_MARGIN, NOT_CERTIFIABLE, PASS, SEVERITY, worst

__version__ = "1.0.0"

__all__ = [
    "PASS", "FAIL", "NOT_CERTIFIABLE", "NO_MARGIN", "SEVERITY", "worst",
    "Quantity", "Panel", "MEAN", "MEDIAN", "Q90", "Q95", "Q99", "STDEV",
    "CATALOG", "DEFAULT_PANEL", "panel_from_names",
    "Interval", "QuantityResult", "PairResult",
    "score", "score_all", "classify", "difference_interval", "summary", "tally",
    "resolution", "panel_resolution", "resolution_curve", "variance_share",
    "kl_divergence", "kl_bits", "kl_floor",
    "load_reference", "load_model",
    "DEFAULT_ALPHA", "DEFAULT_BOOTSTRAP", "DEFAULT_SEED", "DEFAULT_KL_BINS",
    "__version__",
]
