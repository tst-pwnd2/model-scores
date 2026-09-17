"""The panel is an intersection-union test.

The tail is the reason it has more than one member. A model can match the body of
the reference and miss its tail, and the median alone will pass it.
"""

import pytest

from model_scores import PASS, Q99, STDEV, Panel, panel_from_names, score


def test_the_upper_percentile_catches_a_tail_the_median_misses(reference, fat_tailed_model):
    result = score(reference, fat_tailed_model, margin=0.10)
    assert result["median"].verdict == PASS
    assert result["q90"].verdict != PASS


def test_the_panel_takes_the_worst_verdict_present(reference, fat_tailed_model):
    result = score(reference, fat_tailed_model, margin=0.10)
    assert result.verdict != PASS
    assert result.verdict == max(
        (q.verdict for q in result), key=lambda v: {"PASS": 0, "NOT CERTIFIABLE": 1,
                                                    "FAIL": 2, "NO MARGIN": 3}[v])


def test_an_absolute_margin_overrides_the_relative_one_per_quantity(reference, correct_model):
    result = score(reference, correct_model, margin=0.25, absolute_margins={"mean": 0.5})
    assert result["mean"].margin == 0.5
    assert result["median"].margin != 0.5


def test_the_panel_is_configurable(reference, correct_model):
    result = score(reference, correct_model, Panel((Q99, STDEV)), margin=0.25)
    assert set(result.quantities) == {"q99", "stdev"}


def test_panel_from_names_rejects_what_it_does_not_have():
    assert panel_from_names("mean,q99").names == ("mean", "q99")
    with pytest.raises(ValueError):
        panel_from_names("mean,mode")
