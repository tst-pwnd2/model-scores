"""The CP3 corpus adapter, namely the one module here that knows the data layout.

Everything above it takes plain arrays. As such, scoring a different campaign is a
matter of replacing this file, and nothing else.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

__all__ = ["METRICS", "load_reference", "load_model", "LATENCY_STATS",
           "load_reference_latency", "load_model_latency", "merge"]

#: The scalar metrics carried at the top level, beside the per-series integrity block.
METRICS = ("goodput", "availability")

#: A key is (metric, series, mode, window). The scalar metrics repeat the metric
#: name in the series slot, so that every key has the same shape.
Key = tuple[str, str, str, str]


def _walk(document):
    """Yield (key, value) for every leaf in one performance metrics document."""
    for series, modes in document.get("integrity", {}).items():
        for mode, windows in modes.items():
            for window, value in windows.items():
                yield ("integrity", series, mode, window), value
    for metric in METRICS:
        for mode, windows in document.get(metric, {}).items():
            for window, value in windows.items():
                yield (metric, metric, mode, window), value


#: The within-run summaries latency is reduced to. The body and the tail: a
#: model can reproduce one and miss the other, and that is the difference the
#: panel exists to catch.
LATENCY_STATS = ("p50", "p90")

#: Every within-run summary this adapter knows how to take.
LATENCY_REDUCERS = {
    "mean": lambda a: float(np.mean(a)),
    "p50": lambda a: float(np.quantile(a, 0.50)),
    "p90": lambda a: float(np.quantile(a, 0.90)),
    "p95": lambda a: float(np.quantile(a, 0.95)),
    "p99": lambda a: float(np.quantile(a, 0.99)),
}


def _reduce_latency(block, stats) -> dict[Key, float]:
    """One run's latency block, reduced to one number per key.

    The block is ``{mode: {window: [one latency per message]}}``. Each window
    holds a distribution, not a scalar, which is what makes latency different
    from every other metric here and is the whole reason this function exists.

    __Why reduce per run at all.__ The run is the unit of randomness. Messages
    inside a run share its conditions, so pooling every message from every run
    and resampling those would treat correlated observations as independent and
    return intervals far tighter than the campaign earns. Reducing each run
    first keeps the bootstrap resampling runs, which is what the rest of this
    package does with every other metric.
    """
    out: dict[Key, float] = {}
    for mode, windows in (block or {}).items():
        for window, values in windows.items():
            if not values:
                continue
            a = np.asarray(values, dtype=float)
            a = a[np.isfinite(a)]
            if a.size == 0:
                continue
            for stat in stats:
                out[("latency", stat, mode, window)] = LATENCY_REDUCERS[stat](a)
    return out


def load_reference_latency(directory: str, stats=LATENCY_STATS) -> dict[Key, np.ndarray]:
    """Read the evaluator's separate latency tree, one file per trial.

    The compiled performance metrics carry no latency values: their ``latency``
    block is a list of filenames pointing here. Each file is
    ``{mode: {window: [per-message latency]}}`` for one trial, and the array we
    return is across trials, one entry per trial per key.

    Reading all 507 takes a few seconds and about a gigabyte of JSON, so this
    is not done unless asked for.
    """
    files = sorted(glob.glob(os.path.join(directory, "*.json")))
    if not files:
        raise FileNotFoundError(f"no *.json under {directory}")
    return _stack(files, lambda document: _reduce_latency(document, stats))


def load_model_latency(directory: str, stats=LATENCY_STATS) -> dict[Key, np.ndarray]:
    """Read the latency block carried inside each per-run model file.

    The model side needs no separate tree: a run's performance metrics file
    already holds the same ``{mode: {window: [...]}}`` shape under ``latency``.
    """
    files = sorted(glob.glob(os.path.join(directory, "*_performance_metrics.json")))
    if not files:
        raise FileNotFoundError(f"no *_performance_metrics.json under {directory}")
    return _stack(files, lambda document: _reduce_latency(document.get("latency"), stats))


def _stack(files, reduce_one) -> dict[Key, np.ndarray]:
    """Apply a per-file reduction and stack the results by key.

    A run that carries nothing for a key simply does not contribute to it, so
    two keys can rest on different numbers of runs. The scored result reports
    ``n_reference`` and ``n_model`` per pair for that reason.
    """
    store: dict[Key, list[float]] = {}
    for path in files:
        with open(path) as fh:
            document = json.load(fh)
        for key, value in reduce_one(document).items():
            store.setdefault(key, []).append(value)
    return {k: np.asarray(v, dtype=float) for k, v in store.items()}


def merge(*mappings: dict[Key, np.ndarray]) -> dict[Key, np.ndarray]:
    """Combine keyed arrays from several adapters into one comparison."""
    out: dict[Key, np.ndarray] = {}
    for mapping in mappings:
        out.update(mapping)
    return out


def load_reference(path: str) -> dict[Key, np.ndarray]:
    """Read the evaluator's compiled_performance_metrics.json into keyed arrays.

    Each value there is already the across-trial list, so we aggregate nothing.
    """
    with open(path) as fh:
        document = json.load(fh)["compiled_performance_metrics"]
    return {key: np.asarray(values, dtype=float) for key, values in _walk(document)}


def load_model(directory: str) -> tuple[dict[Key, np.ndarray], int]:
    """Read one {date}_run{i}_performance_metrics.json per run into keyed arrays.

    Each file carries one scalar per key, so we assemble the across-run array here.
    Returns the arrays and the number of runs found.
    """
    files = sorted(glob.glob(os.path.join(directory, "*_performance_metrics.json")))
    if not files:
        raise FileNotFoundError(f"no *_performance_metrics.json under {directory}")
    store: dict[Key, list[float]] = {}
    for path in files:
        with open(path) as fh:
            document = json.load(fh)
        for key, value in _walk(document):
            store.setdefault(key, []).append(value)
    return {k: np.asarray(v, dtype=float) for k, v in store.items()}, len(files)
