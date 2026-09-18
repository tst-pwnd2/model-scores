"""The resolution, asked for before a submission exists.

R3 and R4 involve no model, so the command must not require one. This is the
step that decides whether a proposed margin is a question the campaign can
answer, and it has to be available while the margin is still being argued
about.
"""

import json

import numpy as np
import pytest

from model_scores.cli import main

from .conftest import MU, REF_N, SIGMA


def write_reference(path, rng, keys=(("goodput", "goodput", "cumulative", "t0_t900"),)):
    document = {}
    for metric, series, mode, window in keys:
        values = list(rng.normal(MU, SIGMA, REF_N))
        if metric == "integrity":
            document.setdefault("integrity", {}).setdefault(series, {}) \
                    .setdefault(mode, {})[window] = values
        else:
            document.setdefault(metric, {}).setdefault(mode, {})[window] = values
    path.write_text(json.dumps({"compiled_performance_metrics": document}))
    return path


def test_the_resolution_needs_no_model(tmp_path, capsys):
    rng = np.random.default_rng(11)
    reference = write_reference(tmp_path / "compiled.json", rng)
    assert main([str(reference), "--resolution", "40,507"]) == 0
    out = capsys.readouterr().out
    assert "no model involved" in out
    assert "40 model runs" in out and "507 model runs" in out
    # The larger submission resolves the finer margin.
    margins = [float(line.split("%")[0].split()[-1])
               for line in out.splitlines() if "model runs" in line]
    assert margins[0] > margins[1]


def test_without_a_model_it_insists_on_sizes(tmp_path):
    rng = np.random.default_rng(12)
    reference = write_reference(tmp_path / "compiled.json", rng)
    with pytest.raises(SystemExit) as caught:
        main([str(reference)])
    assert "--resolution" in str(caught.value)


def test_a_reference_with_no_relative_margin_says_so(tmp_path, capsys):
    rng = np.random.default_rng(13)
    document = {"integrity": {"alice_1": {"independent": {
        "t0_t900": list(rng.normal(0.0, SIGMA, REF_N))}}}}
    path = tmp_path / "compiled.json"
    path.write_text(json.dumps({"compiled_performance_metrics": document}))
    assert main([str(path), "--resolution", "40"]) == 0
    out = capsys.readouterr().out
    assert "0 with a meaningful relative margin" in out
    assert "no pair carries a relative margin" in out
