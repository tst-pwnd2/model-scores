"""The corpus adapter, exercised on a miniature of the real file layout.

The miniature carries one series of each kind, which is enough to fix the key
shape and the across-run assembly.
"""

import json

import pytest

from model_scores import load_model, load_reference

REFERENCE = {"compiled_performance_metrics": {
    "integrity": {"alice_1_integrity": {"independent": {"t0_t900": [0.1, 0.2, 0.3]}}},
    "goodput": {"cumulative": {"t0_t900": [0.7, 0.8]}},
    "availability": {"cumulative": {"t0_t900": [0.9]}},
}}

RUN = {
    "integrity": {"alice_1_integrity": {"independent": {"t0_t900": 0.15}}},
    "goodput": {"cumulative": {"t0_t900": 0.75}},
    "availability": {"cumulative": {"t0_t900": 0.95}},
}


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "compiled.json").write_text(json.dumps(REFERENCE))
    runs = tmp_path / "runs"
    runs.mkdir()
    for i in range(3):
        (runs / f"2026-09-04_run{i}_performance_metrics.json").write_text(json.dumps(RUN))
    return tmp_path


def test_keys_have_one_shape_across_metrics(corpus):
    reference = load_reference(corpus / "compiled.json")
    assert ("integrity", "alice_1_integrity", "independent", "t0_t900") in reference
    assert ("goodput", "goodput", "cumulative", "t0_t900") in reference
    assert all(len(k) == 4 for k in reference)


def test_the_reference_keeps_its_across_trial_lists(corpus):
    reference = load_reference(corpus / "compiled.json")
    assert list(reference[("integrity", "alice_1_integrity", "independent", "t0_t900")]) \
        == [0.1, 0.2, 0.3]


def test_the_model_is_assembled_across_run_files(corpus):
    model, n_runs = load_model(corpus / "runs")
    assert n_runs == 3
    assert model[("goodput", "goodput", "cumulative", "t0_t900")].size == 3


def test_reference_and_model_keys_line_up(corpus):
    reference = load_reference(corpus / "compiled.json")
    model, _ = load_model(corpus / "runs")
    assert set(reference) == set(model)


def test_an_empty_directory_is_an_error_not_an_empty_result(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path)
