"""A margin stated in the metric's own units, which is what a zero-centred
reference needs.

The procedure refuses to invent one from the reference's own spread, so these
pairs come back as NO MARGIN until the program supplies a number. These cases
check that supplying one decides the pair, and that the decision is the right
way round.
"""

import numpy as np

from model_scores import FAIL, NO_MARGIN, PASS, score
from model_scores.cli import parse_absolute
from model_scores.scoring import absolute_for, has_relative_margin, score_all

from .conftest import MOD_N, MU, SIGMA


def test_an_absolute_margin_decides_a_zero_centred_pair(zero_centred, rng):
    model = rng.normal(0.0, SIGMA, MOD_N)
    assert score(zero_centred, model, margin=0.10).verdict == NO_MARGIN
    # The two samples share a distribution, so a margin of two sigma is one the
    # difference sits well inside.
    decided = score(zero_centred, model, margin=0.10,
                    absolute_margins={"mean": 2 * SIGMA, "median": 2 * SIGMA,
                                      "q90": 2 * SIGMA})
    assert decided.verdict == PASS


def test_an_absolute_margin_still_fails_a_model_that_is_off(zero_centred, rng):
    model = rng.normal(4 * SIGMA, SIGMA, MOD_N)
    decided = score(zero_centred, model, margin=0.10,
                    absolute_margins={"mean": SIGMA, "median": SIGMA, "q90": SIGMA})
    assert decided.verdict == FAIL


def test_a_near_zero_pair_reports_no_relative_reading(zero_centred, rng):
    """A relative reading of a number indistinguishable from zero is the
    nonsense the procedure exists to avoid, so it is not offered even as
    decoration."""
    model = rng.normal(0.0, SIGMA, MOD_N)
    decided = score(zero_centred, model, margin=0.10,
                    absolute_margins={"mean": 2 * SIGMA})
    assert decided["mean"].relative is None
    assert decided["mean"].margin == 2 * SIGMA


def test_the_zero_guard_is_one_rule_used_in_both_places(reference, zero_centred):
    assert has_relative_margin(reference)
    assert not has_relative_margin(zero_centred)


def test_a_particular_scope_narrows_a_general_one():
    table = parse_absolute("q90=1.0,integrity:q90=0.02,integrity:mean=0.01")
    assert absolute_for(("goodput", "goodput", "cumulative", "t0_t900"), table) == {"q90": 1.0}
    assert absolute_for(("integrity", "alice_1", "independent", "t0_t900"), table) == {
        "q90": 0.02, "mean": 0.01}
    assert absolute_for((), None) == {}


def test_a_scope_can_select_one_mode_and_leave_the_other_alone():
    """The reason scoping has to reach past the metric name: integrity
    cumulative sits near one and integrity independent sits at zero, so a
    tolerance in units that suits the second is a much stricter test on the
    first."""
    table = parse_absolute("integrity.*.independent:mean=0.01")
    assert absolute_for(("integrity", "alice_1", "independent", "t0_t900"), table) == {
        "mean": 0.01}
    assert absolute_for(("integrity", "alice_1", "cumulative", "t0_t900"), table) == {}


def test_score_all_applies_the_table_by_metric(zero_centred, rng):
    key = ("integrity", "alice_1", "independent", "t0_t900")
    other = ("goodput", "goodput", "cumulative", "t0_t900")
    model = rng.normal(0.0, SIGMA, MOD_N)
    reference = {key: zero_centred, other: rng.normal(MU, SIGMA, 507)}
    submission = {key: model, other: rng.normal(MU, SIGMA, MOD_N)}
    table = parse_absolute(f"integrity.*.independent:mean={2 * SIGMA},"
                           f"integrity.*.independent:median={2 * SIGMA},"
                           f"integrity.*.independent:q90={2 * SIGMA}")
    got = {r.key: r.verdict for r in
           score_all(reference, submission, margin=0.25, absolute=table, reps=400)}
    assert got[key] == PASS      # the absolute margin reached it
    assert got[other] == PASS    # and the relative margin still governs the other


def test_parse_absolute_rejects_what_it_cannot_use():
    import pytest
    for bad in ("q90", "nonesuch=1.0", "q90=banana"):
        with pytest.raises(SystemExit):
            parse_absolute(bad)


def test_an_absolute_margin_does_not_enter_the_relative_summary(zero_centred, rng):
    """A pair rescued by an absolute margin must not contribute a relative
    certifiable margin to the campaign summary, where the base is zero."""
    from model_scores.scoring import summary
    key = ("integrity", "alice_1", "independent", "t0_t900")
    model = rng.normal(0.0, SIGMA, MOD_N)
    results = score_all({key: zero_centred}, {key: model}, margin=0.10,
                        absolute=parse_absolute(f"mean={2 * SIGMA}"), reps=400)
    assert np.isfinite(results[0].n_model)
    assert "certifiable margin" not in summary(results)
