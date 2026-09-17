"""The interval is the product. The verdict, the certifiable margin and the
printed line are readings of it.
"""

import math

import pytest

from model_scores import score

from .conftest import MOD_N, MU, SIGMA


@pytest.fixture
def biased(reference, rng):
    return score(reference, rng.normal(MU * 1.12, SIGMA, MOD_N), margin=0.10)


def test_the_interval_brackets_the_point_estimate(biased):
    q = biased["mean"]
    assert q.interval.lo < q.interval.point < q.interval.hi


def test_the_point_estimate_recovers_the_true_offset(biased):
    assert biased["mean"].interval.lo < 0.12 * MU < biased["mean"].interval.hi


def test_the_certifiable_margin_is_the_tightest_claim_supported(biased):
    q = biased["mean"]
    assert q.certifiable_margin >= abs(q.interval.point)
    assert q.certifiable_margin == max(abs(q.interval.lo), abs(q.interval.hi))


def test_relative_and_absolute_agree_up_to_the_reference_scale(biased):
    q = biased["mean"]
    assert math.isclose(q.relative.point * abs(q.reference), q.interval.point, rel_tol=1e-9)


def test_the_interval_widens_as_the_submission_shrinks(reference, rng):
    wide = score(reference, rng.normal(MU, SIGMA, 10), margin=0.10)["median"].interval.width
    tight = score(reference, rng.normal(MU, SIGMA, 200), margin=0.10)["median"].interval.width
    assert tight < wide


def test_results_round_trip_through_json(biased):
    import json
    assert json.loads(json.dumps(biased.to_dict()))["verdict"] == biased.verdict
