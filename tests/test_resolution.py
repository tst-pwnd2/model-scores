"""R3 and R4.

Recall that neither takes a model. That is what makes the resolution publishable
before a submission exists, i.e., what makes a margin negotiable in advance.
"""

import math

from model_scores import panel_resolution, resolution, resolution_curve, variance_share

from .conftest import MOD_N, REF_N


def test_more_model_runs_resolve_a_finer_margin(reference):
    coarse = resolution(reference, 10)
    middling = resolution(reference, MOD_N)
    fine = resolution(reference, REF_N)
    assert fine < middling < coarse


def test_the_panel_resolves_no_finer_than_its_coarsest_quantity(reference):
    r = panel_resolution(reference, MOD_N)
    assert r["panel"] == max(r[n] for n in ("mean", "median", "q90"))


def test_an_upper_percentile_is_the_binding_member(reference):
    """This is the price of looking at the tail, and the reason to set margins per
    quantity rather than once for the panel.
    """
    r = panel_resolution(reference, MOD_N)
    assert r["q90"] > r["mean"]


def test_the_model_dominates_the_variance_at_the_submission_size(reference):
    assert variance_share(reference, MOD_N) > 0.9


def test_the_two_sides_contribute_equally_at_equal_size(reference):
    assert math.isclose(variance_share(reference, REF_N), 0.5, abs_tol=0.05)


def test_the_curve_is_monotone(reference):
    curve = resolution_curve(reference, [10, 20, 40, 80, 160, 320])
    margins = [m for _, m in curve]
    assert margins == sorted(margins, reverse=True)


def test_an_absolute_resolution_carries_the_metric_units(reference):
    absolute = resolution(reference, MOD_N, relative=False)
    relative = resolution(reference, MOD_N)
    assert math.isclose(absolute / abs(reference.mean()), relative, rel_tol=0.05)
