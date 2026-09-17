"""The figures. We assert that they draw and what they encode, not how they look.

Every check here is a claim the critique makes elsewhere in prose: that a verdict
never rides on colour alone, that the status colours are the reserved ones, and
that the interval figure shows a spread of verdicts rather than four copies of
the worst one.
"""

import pytest

from model_scores import FAIL, NOT_CERTIFIABLE, PASS, Panel, score, score_all
from model_scores.verdicts import ORDER

from .conftest import MOD_N, MU, SIGMA

plots = pytest.importorskip("model_scores.plots",
                            reason="figures need the plots extra")
pytest.importorskip("matplotlib", reason="figures need the plots extra")


@pytest.fixture
def corpus(reference, rng):
    """One reference against several models, spanning three verdicts."""
    models = {
        ("goodput", "goodput", "cumulative", "t0_t900"): rng.normal(MU, SIGMA, MOD_N),
        ("goodput", "goodput", "cumulative", "t0_t1800"): rng.normal(MU * 1.6, SIGMA, MOD_N),
        ("goodput", "goodput", "independent", "t900_t1800"): rng.normal(MU * 1.1, SIGMA, MOD_N),
    }
    refs = dict.fromkeys(models)
    for k in refs:
        refs[k] = reference
    return refs, models


def test_every_verdict_has_a_glyph():
    """Colour never carries a verdict alone, since the status reds and greens do
    not separate under deuteranopia.
    """
    assert set(plots.GLYPH) == set(ORDER)
    assert len(set(plots.GLYPH.values())) == len(ORDER)


def test_the_status_palette_is_the_reserved_one():
    assert plots.STATUS[PASS] == "#0ca30c"
    assert plots.STATUS[FAIL] == "#d03b3b"
    assert plots.STATUS[NOT_CERTIFIABLE] == "#fab219"


def test_the_curve_series_stay_within_the_validated_slots():
    """Only the first three categorical slots clear the all-pairs floors."""
    assert len(plots.SERIES) == 3


def test_representative_spreads_across_verdicts(corpus):
    refs, models = corpus
    results = score_all(refs, models, margin=0.10)
    chosen = plots.representative(results, limit=4)
    assert len({r.verdict for r in chosen}) == len(chosen)


def test_interval_plot_writes_a_file(reference, correct_model, tmp_path):
    result = score(reference, correct_model, margin=0.10, key=("m", "s", "mode", "t0_t900"))
    out = plots.interval_plot([result], str(tmp_path / "iv.png"))
    assert (tmp_path / "iv.png").stat().st_size > 0
    assert out.endswith("iv.png")


def test_resolution_plot_writes_a_file(reference, tmp_path):
    plots.resolution_plot(reference, str(tmp_path / "res.png"), sizes=(10, 40, 160),
                          reps=200, proposed_margin=0.10, current=MOD_N)
    assert (tmp_path / "res.png").stat().st_size > 0


def test_verdict_map_writes_a_file(corpus, tmp_path):
    refs, models = corpus
    results = score_all(refs, models, margin=0.10)
    plots.verdict_map(results, str(tmp_path / "map.png"))
    assert (tmp_path / "map.png").stat().st_size > 0


def test_write_all_produces_the_three_figures(corpus, reference, tmp_path):
    refs, models = corpus
    results = score_all(refs, models, margin=0.10)
    written = plots.write_all(results, str(tmp_path / "figs"), reference=reference,
                              panel=Panel(), margin=0.10, reps=200)
    assert len(written) == 3
    assert all(p.endswith(".png") for p in written)


def test_an_empty_result_set_is_an_error_not_a_blank_figure(tmp_path):
    with pytest.raises(ValueError):
        plots.verdict_map([], str(tmp_path / "x.png"))
    with pytest.raises(ValueError):
        plots.interval_plot([], str(tmp_path / "y.png"))
