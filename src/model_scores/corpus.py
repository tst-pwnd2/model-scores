"""The CP3 corpus adapter, namely the one module here that knows the data layout.

Everything above it takes plain arrays. As such, scoring a different campaign is a
matter of replacing this file, and nothing else.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

__all__ = ["METRICS", "load_reference", "load_model"]

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
