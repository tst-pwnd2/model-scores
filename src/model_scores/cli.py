"""Command line entry point.

For the sake of being self-contained, the report it prints carries the tally, the
divergence ranking and, on request, the achievable resolution, so that a reader
has everything the recommendations rest on without opening the JSON.

    python3 -m model_scores REFERENCE.json MODEL_DIR
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence

import numpy as np

from .bootstrap import DEFAULT_ALPHA, DEFAULT_BOOTSTRAP, DEFAULT_SEED
from .corpus import (
    LATENCY_REDUCERS,
    LATENCY_STATS,
    load_model,
    load_model_latency,
    load_reference,
    load_reference_latency,
    merge,
)
from .quantities import CATALOG, panel_from_names
from .resolution import panel_resolution, variance_share
from .scoring import MIN_REFERENCE, has_relative_margin, score_all, summary
from .verdicts import SEVERITY


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="model-scores", description="Panel scoring for model submissions.")
    p.add_argument("reference", help="compiled_performance_metrics.json")
    p.add_argument("model_dir", nargs="?",
                   help="directory of per-run performance metrics. Leave it out "
                        "to report the achievable resolution from the reference "
                        "alone, which needs no model and can be done before a "
                        "submission exists")
    p.add_argument("--margin", type=float, default=0.10,
                   help="relative margin, as a fraction (default 0.10)")
    p.add_argument("--quantities", default="mean,median,q90",
                   help=f"panel members from: {', '.join(CATALOG)}")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--reps", type=int, default=DEFAULT_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--resolution", metavar="N,N,...",
                   help="report the achievable margin at these submission sizes")
    p.add_argument("--latency", metavar="DIR",
                   help="the evaluator's latency tree, one file per trial. The "
                        "compiled metrics carry no latency values, only "
                        "filenames pointing here. The model side is read from "
                        "MODEL_DIR, which already holds the same block")
    p.add_argument("--latency-stats", default=",".join(LATENCY_STATS),
                   help=f"within-run summaries to score latency on, from: "
                        f"{', '.join(LATENCY_REDUCERS)} "
                        f"(default {','.join(LATENCY_STATS)})")
    p.add_argument("--absolute", metavar="[SCOPE:]QUANTITY=VALUE,...", default="",
                   help="margins in the metric's own units, which is what a pair "
                        "whose reference statistic sits at zero needs. SCOPE is a "
                        "dotted prefix of metric.series.mode.window, with * for any "
                        "one segment, so integrity.*.independent takes the "
                        "independent integrity pairs and nothing else. A more "
                        "particular scope narrows a more general one")
    p.add_argument("--plots", metavar="DIR",
                   help="write the three figures into DIR (needs the plots extra)")
    p.add_argument("--plot-format", default="png", choices=("png", "pdf", "svg"),
                   help="figure format (default png; pdf for a LaTeX document)")
    p.add_argument("--detail", type=int, default=0,
                   help="print the full panel for the N worst pairs")
    p.add_argument("--out", help="write the full result set as JSON")
    return p


def parse_absolute(spec: str) -> dict[str, dict[str, float]]:
    """Read ``integrity.*.independent:q90=0.02,mean=1.5`` into a scope table.

    An entry with no scope lands under ``"*"`` and applies to every pair.
    """
    table: dict[str, dict[str, float]] = {}
    for entry in (e.strip() for e in spec.split(",") if e.strip()):
        if "=" not in entry:
            raise SystemExit(f"--absolute wants [SCOPE:]QUANTITY=VALUE, got {entry!r}")
        name, _, value = entry.partition("=")
        scope, _, quantity = name.rpartition(":")
        quantity = quantity.strip()
        if quantity not in CATALOG:
            raise SystemExit(f"--absolute: unknown quantity {quantity!r}; "
                             f"choose from {', '.join(CATALOG)}")
        try:
            table.setdefault(scope.strip() or "*", {})[quantity] = float(value)
        except ValueError:
            raise SystemExit(f"--absolute: {value!r} is not a number") from None
    return table


def report_resolution(reference, keys, panel, sizes, reps, seed) -> None:
    """R3 and R4, which involve no model at all."""
    print("achievable resolution, from the reference alone, no model involved.")
    print("median over pairs of the panel margin a correct submission could certify:")
    for n in sizes:
        margins = [panel_resolution(reference[k], n, panel, reps=reps, seed=seed)["panel"]
                   for k in keys]
        margins = [m for m in margins if math.isfinite(m)]
        if not margins:
            print(f"   {n:5d} model runs   no pair carries a relative margin")
            continue
        share = np.median([variance_share(reference[k], n, panel.first, reps, seed)
                           for k in keys])
        print(f"   {n:5d} model runs   {np.median(margins) * 100:6.2f}%   "
              f"model share of variance {share * 100:5.1f}%")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        panel = panel_from_names(args.quantities)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    absolute = parse_absolute(args.absolute)
    sizes = [int(x) for x in args.resolution.split(",")] if args.resolution else []
    stats = [s.strip() for s in args.latency_stats.split(",") if s.strip()]
    unknown = [s for s in stats if s not in LATENCY_REDUCERS]
    if unknown:
        raise SystemExit(f"--latency-stats: unknown {', '.join(unknown)}; "
                         f"choose from {', '.join(LATENCY_REDUCERS)}")

    reference = load_reference(args.reference)
    if args.latency:
        reference = merge(reference, load_reference_latency(args.latency, stats))

    if args.model_dir is None:
        # The resolution needs no model, so neither does this path. It is how a
        # margin gets agreed before a submission exists.
        if not sizes:
            raise SystemExit("without MODEL_DIR, pass --resolution N,N to say "
                             "which submission sizes to report")
        keys = [k for k, v in sorted(reference.items())
                if len(v) >= MIN_REFERENCE and has_relative_margin(v)]
        print(f"reference pairs {len(reference)}, "
              f"{len(keys)} with a meaningful relative margin\n")
        report_resolution(reference, keys, panel, sizes,
                          max(400, args.reps // 4), args.seed)
        return 0

    model, n_runs = load_model(args.model_dir)
    if args.latency:
        model = merge(model, load_model_latency(args.model_dir, stats))
    print(f"reference pairs {len(reference)}, model pairs {len(model)}, "
          f"shared {len(set(reference) & set(model))}")
    if args.latency:
        print(f"latency reduced per run to {', '.join(stats)}, "
              f"since the run is the unit of randomness")
    print(f"model runs {n_runs}, margin {args.margin * 100:.1f}% relative")
    for scope, entries in sorted(absolute.items()):
        stated = ", ".join(f"{q} {v:g}" for q, v in sorted(entries.items()))
        where = "every pair" if scope == "*" else scope
        print(f"absolute margin on {where}: {stated}")
    print()

    results = score_all(reference, model, panel, args.margin, absolute=absolute,
                        alpha=args.alpha, reps=args.reps, seed=args.seed)
    print(summary(results))

    if sizes:
        print()
        report_resolution(reference, [r.key for r in results if not r.near_zero],
                          panel, sizes, max(400, args.reps // 4), args.seed)

    if args.plots:
        from .plots import write_all
        sample = next((reference[r.key] for r in results if r.key in reference), None)
        try:
            written = write_all(results, args.plots, reference=sample, panel=panel,
                                margin=args.margin, fmt=args.plot_format,
                                reps=max(400, args.reps // 4), seed=args.seed)
        except ImportError as e:
            raise SystemExit(str(e)) from None
        print("\nfigures:")
        for p in written:
            print(f"   {p}")

    if args.detail:
        print("\nworst pairs in full:")
        worst_first = sorted(results, key=lambda r: (-SEVERITY[r.verdict], -r.kl_excess))
        for r in worst_first[:args.detail]:
            print()
            print(r.report())

    if args.out:
        with open(args.out, "w") as fh:
            json.dump([r.to_dict() for r in results], fh, indent=1)
        print(f"\nwrote {args.out}")
    return 0
