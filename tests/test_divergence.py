"""The KL diagnostic, and why we read it against a floor rather than against zero.

At a finite submission size a correct model still scores a positive divergence.
The floor is that score, and the excess over it is the interpretable quantity.
"""

import math

from model_scores import kl_bits, kl_divergence, kl_floor

from .conftest import MOD_N, MU, SIGMA


def test_divergence_rises_with_disagreement(reference, rng):
    near = kl_bits(rng.normal(MU, SIGMA, MOD_N), reference)
    far = kl_bits(rng.normal(MU * 1.5, SIGMA, MOD_N), reference)
    assert far > near


def test_the_floor_is_positive_at_a_finite_submission_size(reference):
    assert kl_floor(reference, MOD_N) > 0


def test_a_correct_model_sits_near_the_floor_not_near_zero(reference, correct_model):
    floor = kl_floor(reference, MOD_N)
    assert abs(kl_bits(correct_model, reference) - floor) < 0.5 * floor + 0.2


def test_the_floor_falls_as_the_submission_grows(reference):
    assert kl_floor(reference, 400) < kl_floor(reference, 20)


def test_identical_histograms_diverge_by_nothing():
    assert math.isclose(kl_divergence([4, 2, 1], [4, 2, 1]), 0.0, abs_tol=1e-12)


def test_smoothing_keeps_an_empty_bin_finite():
    assert math.isfinite(kl_divergence([1, 0, 3], [0, 2, 3]))
