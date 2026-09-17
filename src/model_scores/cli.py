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
from .corpus import load_model, load_reference
from .quantities import CATALOG, panel_from_names
from .resolution import panel_resolution, variance_share
from .scoring import score_all, summary
from .verdicts import SEVERITY


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="model-scores", description="Panel scoring for model submissions.")
    p.add_argument("reference", help="compiled_performance_metrics.json")
    p.add_argument("model_dir", help="directory of per-run performance metrics")
    p.add_argument("--margin", type=float, default=0.10,
                   help="relative margin, as a fraction (default 0.10)")
    p.add_argument("--quantities", default="mean,median,q90",
                   help=f"panel members from: {', '.join(CATALOG)}")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--reps", type=int, default=DEFAULT_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--resolution", metavar="N,N,...",
                   help="report the achievable margin at these submission sizes")
    p.add_argument("--plots", metavar="DIR",
                   help="write the three figures into DIR (needs the plots extra)")
    p.add_argument("--plot-format", default="png", choices=("png", "pdf", "svg"),
                   help="figure format (default png; pdf for a LaTeX document)")
    p.add_argument("--detail", type=int, default=0,
                   help="print the full panel for the N worst pairs")
    p.add_argument("--out", help="write the full result set as JSON")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        panel = panel_from_names(args.quantities)
    except ValueError as e:
        raise SystemExit(str(e)) from None

    reference = load_reference(args.reference)
    model, n_runs = load_model(args.model_dir)
    print(f"reference pairs {len(reference)}, model pairs {len(model)}, "
          f"shared {len(set(reference) & set(model))}")
    print(f"model runs {n_runs}, margin {args.margin * 100:.1f}% relative\n")

    results = score_all(reference, model, panel, args.margin,
                        alpha=args.alpha, reps=args.reps, seed=args.seed)
    print(summary(results))

    if args.resolution:
        reps = max(400, args.reps // 4)
        keys = [r.key for r in results if r.has_margin]
        print("\nachievable resolution, from the reference alone, no model involved.")
        print("median over pairs of the panel margin a correct submission could certify:")
        for n in (int(x) for x in args.resolution.split(",")):
            margins = [panel_resolution(reference[k], n, panel, reps=reps,
                                        seed=args.seed)["panel"] for k in keys]
            margins = [m for m in margins if math.isfinite(m)]
            share = np.median([variance_share(reference[k], n, panel.first, reps, args.seed)
                               for k in keys])
            print(f"   {n:5d} model runs   {np.median(margins) * 100:6.2f}%   "
                  f"model share of variance {share * 100:5.1f}%")

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
