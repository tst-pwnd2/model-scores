"""Latency, which arrives as a distribution per run rather than a scalar.

Every other metric gives one number per run. Latency gives thousands, one per
message, and the adapter's job is to get from there to the across-run array the
rest of the package takes. These cases fix what that reduction means and check
the trap it exists to avoid.
"""

import json

import numpy as np

from model_scores.bootstrap import rng_from
from model_scores.corpus import (
    LATENCY_STATS,
    load_model_latency,
    load_reference_latency,
    merge,
)
from model_scores.quantities import MEDIAN
from model_scores.scoring import difference_interval


def write_reference(directory, runs):
    """One file per trial, each {mode: {window: [per-message latency]}}."""
    directory.mkdir(exist_ok=True)
    for i, messages in enumerate(runs):
        (directory / f"latencies_run{i:03d}.json").write_text(json.dumps(
            {"cumulative": {"t0_t900": list(messages)},
             "independent": {"t0_t900": list(messages)}}))
    return directory


def write_model(directory, runs):
    """One performance metrics file per run, with latency nested inside."""
    directory.mkdir(exist_ok=True)
    for i, messages in enumerate(runs):
        (directory / f"2026-09-02_run{i:03d}_performance_metrics.json").write_text(
            json.dumps({"latency": {"cumulative": {"t0_t900": list(messages)},
                                    "independent": {"t0_t900": list(messages)}}}))
    return directory


def test_a_run_becomes_one_number_per_key(tmp_path, rng):
    runs = [rng.lognormal(1.0, 0.5, 300) for _ in range(12)]
    ref = load_reference_latency(str(write_reference(tmp_path / "lat", runs)))
    key = ("latency", "p50", "cumulative", "t0_t900")
    assert key in ref
    assert ref[key].shape == (12,)
    # The entry for a run is that run's own median, not anything pooled.
    assert np.isclose(sorted(ref[key])[0], min(np.median(r) for r in runs))


def test_the_series_slot_names_the_within_run_summary(tmp_path, rng):
    runs = [rng.lognormal(1.0, 0.5, 300) for _ in range(6)]
    ref = load_reference_latency(str(write_reference(tmp_path / "lat", runs)),
                                 stats=("p50", "p90", "mean"))
    stats = {key[1] for key in ref}
    assert stats == {"p50", "p90", "mean"}
    p50 = ref[("latency", "p50", "cumulative", "t0_t900")]
    p90 = ref[("latency", "p90", "cumulative", "t0_t900")]
    assert np.all(p90 > p50)


def test_the_model_side_needs_no_separate_tree(tmp_path, rng):
    runs = [rng.lognormal(1.0, 0.5, 200) for _ in range(5)]
    mod = load_model_latency(str(write_model(tmp_path / "model", runs)))
    assert set(mod) == {("latency", s, m, "t0_t900")
                        for s in LATENCY_STATS
                        for m in ("cumulative", "independent")}


def test_an_empty_window_contributes_nothing(tmp_path, rng):
    """A window in which a run sent no messages is absent from that run, so two
    keys can rest on different numbers of runs."""
    directory = tmp_path / "lat"
    directory.mkdir()
    (directory / "a.json").write_text(json.dumps(
        {"cumulative": {"t0_t900": [1.0, 2.0], "t0_t1800": []}}))
    (directory / "b.json").write_text(json.dumps(
        {"cumulative": {"t0_t900": [3.0, 4.0], "t0_t1800": [5.0]}}))
    ref = load_reference_latency(str(directory))
    assert ref[("latency", "p50", "cumulative", "t0_t900")].shape == (2,)
    assert ref[("latency", "p50", "cumulative", "t0_t1800")].shape == (1,)


def test_pooling_messages_would_understate_the_interval(tmp_path, rng):
    """Why the reduction is per run.

    Messages inside a run share its conditions. Pooling every message from every
    run and resampling those treats correlated observations as independent, and
    the interval it returns is far tighter than the campaign earns. Here each
    run has its own level, so the between-run spread is the real uncertainty.
    """
    levels = rng.normal(10.0, 3.0, 30)
    runs = [rng.normal(level, 0.2, 500) for level in levels]
    ref = load_reference_latency(str(write_reference(tmp_path / "lat", runs)))
    per_run = ref[("latency", "p50", "cumulative", "t0_t900")]
    pooled = np.concatenate(runs)

    model_levels = rng.normal(10.0, 3.0, 12)
    model_runs = [rng.normal(level, 0.2, 500) for level in model_levels]
    model_per_run = np.array([np.median(r) for r in model_runs])

    honest = difference_interval(per_run, model_per_run, MEDIAN, 600, 0.05, rng_from(3))
    optimistic = difference_interval(pooled, np.concatenate(model_runs), MEDIAN,
                                     600, 0.05, rng_from(3))
    assert honest.width > 5 * optimistic.width


def test_merge_puts_the_two_adapters_in_one_comparison(tmp_path, rng):
    runs = [rng.lognormal(1.0, 0.5, 100) for _ in range(4)]
    ref = load_reference_latency(str(write_reference(tmp_path / "lat", runs)))
    other = {("goodput", "goodput", "cumulative", "t0_t900"): np.arange(4.0)}
    both = merge(other, ref)
    assert len(both) == len(ref) + 1
    assert ("goodput", "goodput", "cumulative", "t0_t900") in both
