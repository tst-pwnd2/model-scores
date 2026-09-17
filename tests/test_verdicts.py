"""The four outcomes mean what the critique says they mean.

Each case below builds a model whose relationship to the reference we know, and
asserts the verdict the procedure ought to reach.
"""

import pytest

from model_scores import FAIL, NO_MARGIN, NOT_CERTIFIABLE, PASS, panel_resolution, score

from .conftest import MOD_N, MU, SIGMA


def test_correct_model_passes_at_a_margin_it_can_resolve(reference, correct_model):
    assert score(reference, correct_model, margin=0.25).verdict == PASS


def test_a_model_well_off_the_reference_fails(reference, rng):
    model = rng.normal(MU * 1.25, SIGMA, MOD_N)
    assert score(reference, model, margin=0.10).verdict == FAIL


def test_a_margin_finer_than_the_resolution_is_not_certifiable(reference, correct_model):
    assert score(reference, correct_model, margin=0.01).verdict == NOT_CERTIFIABLE


def test_a_reference_centred_on_zero_admits_no_relative_margin(zero_centred, rng):
    model = rng.normal(0.0, SIGMA, MOD_N)
    assert score(zero_centred, model, margin=0.10).verdict == NO_MARGIN


def test_a_correct_model_clears_the_published_resolution(reference, correct_model):
    margin = panel_resolution(reference, MOD_N)["panel"]
    assert score(reference, correct_model, margin=margin * 1.2).verdict == PASS


@pytest.mark.parametrize("margin", [0.02, 0.05, 0.10, 0.25])
def test_a_wrong_model_is_never_certified(reference, rng, margin):
    """A model at twice the reference fails at every margin we would plausibly set."""
    model = rng.normal(MU * 2, SIGMA, MOD_N)
    assert score(reference, model, margin=margin).verdict == FAIL
