"""Figures for the three things a table conveys badly.

We draw exactly three. The interval plot shows why a pair reached its verdict,
which is R2. The resolution curve shows what more runs would buy, which is R4.
The verdict map shows where the failures sit, which a tally over hundreds of
pairs hides entirely.

Colour follows the job it does. The four verdicts are a status encoding, so they
take the reserved status palette rather than a series palette. Recall that the
status greens and reds do not separate under deuteranopia (CVD Delta E 4.1
measured on this pair), and as such every verdict cell and marker carries a
glyph, and the legend carries the glyph beside the swatch. Colour never carries
the verdict alone. The curves are a genuine series encoding and take the first
three categorical slots, which are the three that clear the all-pairs floors.

These are static figures for a document, so they ship no hover layer and no dark
mode. Ask for those if the figures end up in something interactive.

matplotlib is an optional dependency. Install it with ``uv sync --extra plots``.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Iterable, Sequence

import numpy as np

from .quantities import DEFAULT_PANEL, Panel
from .resolution import resolution
from .results import PairResult
from .verdicts import FAIL, NO_MARGIN, NOT_CERTIFIABLE, ORDER, PASS

__all__ = ["STATUS", "GLYPH", "SERIES", "interval_plot", "resolution_plot",
           "verdict_map", "representative", "write_all"]

#: The reserved status palette. PASS, NOT CERTIFIABLE and FAIL take good,
#: warning and critical. NO MARGIN takes the muted ink instead of a fourth
#: status step, since it reports an absent question rather than a degree of
#: badness, and should not read as a severity.
STATUS = {
    PASS: "#0ca30c",
    NOT_CERTIFIABLE: "#fab219",
    FAIL: "#d03b3b",
    NO_MARGIN: "#898781",
}

#: The secondary encoding that makes the status colours legible under CVD, in
#: grayscale print and in forced-colors mode.
GLYPH = {PASS: "P", NOT_CERTIFIABLE: "?", FAIL: "F", NO_MARGIN: "-"}

#: Categorical slots 1 to 3. These three clear the all-pairs separation floors,
#: which is why the curve figure carries three quantities and not more.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

_RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": SECONDARY,
    "ytick.labelcolor": SECONDARY,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
}


def _plt():
    """Import matplotlib on demand, with a message that says what to install."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "plots need matplotlib, which is an optional dependency. "
            "Install it with: uv sync --extra plots"
        ) from e
    return plt


def _window_order(window: str) -> tuple:
    """Sort windows by the times in their names, e.g., t900_t1800 before t9900_t10800."""
    nums = [int(n) for n in re.findall(r"\d+", window)]
    return (nums[0] if nums else 0, nums[1] if len(nums) > 1 else 0)


def _recede(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)


# --------------------------------------------------------------------------
# R2. Why a pair reached its verdict.
# --------------------------------------------------------------------------

def interval_plot(results: Sequence[PairResult], path: str, title: str | None = None):
    """A forest plot of the per-quantity intervals, with the margin drawn in.

    This is the figure that separates the two ways a pair can miss. A model nine
    percent off sits well outside the margin with a tight interval. A model that
    is merely unresolved sits near zero with an interval too wide to certify. The
    old pass bit rendered both as the same character.

    We give each pair its own panel and its own x range. A single shared axis
    would let one gross failure at a hundred percent squash every other interval
    against zero, which loses the comparison the figure exists to make. Recall
    that the comparison is between an interval and its margin, and the margin is
    drawn in each panel, so faceting preserves it.

    Values are relative, i.e., a percentage of the reference statistic. A pair
    with no relative margin is drawn on its absolute scale, and its panel says so.
    """
    plt = _plt()
    results = list(results)
    if not results or not any(r.quantities for r in results):
        raise ValueError("no quantities to plot")

    per = max(len(r.quantities) for r in results)
    height = len(results) * (0.30 * per + 0.62) + 1.1
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(len(results), 1, figsize=(7.2, height),
                                 squeeze=False)
        axes = axes.ravel()

        for ax, pair in zip(axes, results, strict=True):
            rows = list(pair.quantities.items())
            n = len(rows)
            margin_pct, lo_all, hi_all = None, [], []

            for i, (_, q) in enumerate(rows):
                y = n - 1 - i
                iv = q.relative if q.relative is not None else q.interval
                scale = 100.0 if q.relative is not None else 1.0
                colour = STATUS[q.verdict]
                lo, hi, point = iv.lo * scale, iv.hi * scale, iv.point * scale
                lo_all.append(lo)
                hi_all.append(hi)
                ax.plot([lo, hi], [y, y], color=colour, linewidth=2.0,
                        solid_capstyle="round", zorder=3)
                ax.plot([point], [y], marker="o", markersize=8, color=colour,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4,
                        linestyle="none")
                ax.annotate(GLYPH[q.verdict], (point, y), color=SURFACE,
                            fontsize=5.5, fontweight="bold", ha="center",
                            va="center", zorder=5)
                if q.margin is not None and q.reference:
                    margin_pct = q.margin / abs(q.reference) * 100.0

            if margin_pct is not None:
                ax.axvspan(-margin_pct, margin_pct, color=GRID, alpha=0.55, zorder=0)
                for x in (-margin_pct, margin_pct):
                    ax.axvline(x, color=AXIS, linewidth=1.0,
                               linestyle=(0, (4, 3)), zorder=1)
                lo_all.append(-margin_pct * 1.6)
                hi_all.append(margin_pct * 1.6)
            ax.axvline(0.0, color=AXIS, linewidth=1.0, zorder=1)

            span = max(hi_all) - min(lo_all)
            ax.set_xlim(min(lo_all) - 0.08 * span, max(hi_all) + 0.08 * span)
            ax.set_yticks(range(n))
            ax.set_yticklabels([name for name, _ in reversed(rows)])
            ax.set_ylim(-0.6, n - 0.4)
            ax.grid(axis="y", visible=False)
            unit = "percent of the reference" if pair.has_margin else "absolute"
            head = f"{_pair_label(pair)}"
            if margin_pct is not None:
                head += f"      margin {margin_pct:.0f}%"
            else:
                head += "      no margin, absolute scale"
            ax.set_title(head, loc="left", fontsize=8.5)
            ax.set_xlabel(f"model minus reference, {unit}", fontsize=8)
            _recede(ax)

        fig.suptitle(title or "Where each quantity sits against the margin",
                     x=0.012, ha="left", fontsize=11, fontweight="bold", color=INK)
        fig.tight_layout(rect=(0, 0.02, 1, 0.985))
        _verdict_legend(fig, {q.verdict for r in results for q in r}, plt)
        fig.savefig(path)
        plt.close(fig)
    return path


def _pair_label(pair: PairResult) -> str:
    """metric.series, mode, window, with the redundant series name dropped."""
    metric, series, mode, window = (list(pair.key) + [""] * 4)[:4]
    name = series if series != metric else metric
    return f"{name}   {mode}   {window.replace('_', ' ')}"


def _verdict_legend(fig, verdicts, plt, pad: float = 0.03):
    """Swatch, glyph and word together, placed clear of the tick labels.

    The glyph is what carries the verdict where the colours do not separate, so
    it never leaves the legend. We measure the drawn extent of the axes rather
    than reserving a guessed margin, since rotated window labels are tall and
    their height depends on the campaign.
    """
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", markersize=8, linestyle="none",
                      color=STATUS[v], label=f"{GLYPH[v]}   {v}")
               for v in ORDER if v in verdicts]
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    inverse = fig.transFigure.inverted()
    bottom = min(ax.get_tightbbox(renderer).transformed(inverse).y0
                 for ax in fig.axes)
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, bottom - pad),
               ncol=min(4, len(handles)), handletextpad=0.4, columnspacing=1.8,
               labelcolor=SECONDARY, fontsize=8, frameon=False)


# --------------------------------------------------------------------------
# R4. What more runs would buy.
# --------------------------------------------------------------------------

def resolution_plot(reference, path: str, sizes: Iterable[int] = (10, 20, 40, 80, 160,
                                                                  320, 507),
                    panel: Panel = DEFAULT_PANEL, current: int | None = None,
                    proposed_margin: float | None = None,
                    title: str | None = None, **kw):
    """The achievable margin against submission size, one curve per quantity.

    The curve flattens where the model side stops dominating the combined
    standard error. Recall that no model enters this calculation, so the figure
    can be drawn and agreed before a submission exists.

    ``proposed_margin`` draws the margin under discussion as a horizontal rule.
    Where a curve crosses it is the submission size that margin requires.
    """
    plt = _plt()
    sizes = [int(n) for n in sizes]

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        for slot, quantity in enumerate(panel):
            ys = [resolution(reference, n, quantity, **kw) * 100 for n in sizes]
            colour = SERIES[slot % len(SERIES)]
            ax.plot(sizes, ys, color=colour, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3,
                    label=quantity.name)
            ax.annotate(f" {quantity.name}", (sizes[-1], ys[-1]), color=colour,
                        fontsize=9, fontweight="bold", va="center", ha="left")

        if proposed_margin is not None:
            y = proposed_margin * 100
            ax.axhline(y, color=SECONDARY, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
            ax.annotate(f"proposed margin {y:.0f}%", (sizes[0], y), color=SECONDARY,
                        fontsize=8, va="bottom", ha="left",
                        xytext=(0, 4), textcoords="offset points")
        if current is not None:
            ax.axvline(current, color=AXIS, linewidth=1.0, zorder=1)
            ax.annotate(f"we submit {current}", (current, ax.get_ylim()[1]),
                        color=MUTED, fontsize=8, va="top", ha="left",
                        xytext=(4, -4), textcoords="offset points")

        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(n) for n in sizes])
        ax.set_xlim(sizes[0] * 0.9, sizes[-1] * 1.25)
        ax.set_xlabel("model runs submitted")
        ax.set_ylabel("finest margin a correct model could certify, percent")
        ax.set_title(title or "What more runs would buy", loc="left")
        ax.legend(loc="upper right", labelcolor=SECONDARY, fontsize=8, ncol=len(panel))
        _recede(ax)
        fig.savefig(path)
        plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Where the failures sit.
# --------------------------------------------------------------------------

def verdict_map(results: Sequence[PairResult], path: str, title: str | None = None):
    """A verdict per cell, over series and window, faceted by mode.

    A tally over hundreds of pairs says how many failed. It does not say whether
    the failures are scattered or concentrated, and that difference is the whole
    diagnostic value of the campaign: a row that fails end to end points at one
    mechanism, whereas the same count sprinkled across the grid does not.

    Every cell carries a glyph as well as a colour, and the facets share one row
    order, so a row can be read straight across.
    """
    plt = _plt()
    results = list(results)
    modes = sorted({r.key[2] for r in results})
    if not modes:
        raise ValueError("no results to plot")

    rows = sorted({f"{r.key[0]}.{r.key[1]}" for r in results})
    facets = []
    for mode in modes:
        sel = [r for r in results if r.key[2] == mode]
        cols = sorted({r.key[3] for r in sel}, key=_window_order)
        facets.append((mode, sel, cols))

    widths = [max(1, len(c)) for _, _, c in facets]
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            1, len(facets),
            figsize=(1.9 + 0.29 * sum(widths), 1.7 + 0.27 * len(rows)),
            gridspec_kw={"width_ratios": widths, "wspace": 0.06})
        axes = np.atleast_1d(axes)

        for k, (ax, (mode, sel, cols)) in enumerate(zip(axes, facets, strict=True)):
            index = {(f"{r.key[0]}.{r.key[1]}", r.key[3]): r.verdict for r in sel}
            for x, window in enumerate(cols):
                for y, row in enumerate(rows):
                    verdict = index.get((row, window))
                    if verdict is None:
                        continue
                    ax.add_patch(plt.Rectangle((x + 0.05, y + 0.05), 0.90, 0.90,
                                               facecolor=STATUS[verdict],
                                               edgecolor=SURFACE, linewidth=0.8))
                    ax.annotate(GLYPH[verdict], (x + 0.5, y + 0.5), color=SURFACE,
                                fontsize=6, fontweight="bold", ha="center",
                                va="center")
            ax.set_xlim(0, len(cols))
            ax.set_ylim(0, len(rows))
            ax.set_xticks([i + 0.5 for i in range(len(cols))])
            ax.set_xticklabels([c.replace("_", " ") for c in cols],
                               rotation=90, fontsize=6)
            if k == 0:
                ax.set_yticks([i + 0.5 for i in range(len(rows))])
                ax.set_yticklabels([r.split(".")[-1][:22] for r in rows], fontsize=6.5)
            else:
                ax.set_yticks([])
            ax.set_title(mode, loc="left", fontsize=10)
            ax.grid(visible=False)
            for side in ("top", "right", "left", "bottom"):
                ax.spines[side].set_visible(False)
            ax.tick_params(length=0)

        fig.suptitle(title or "Where the verdicts fall", x=0.012, ha="left",
                     fontsize=11, fontweight="bold", color=INK)
        # The cell axes carry fixed limits and manual patches, which tight_layout
        # cannot reason about. We let savefig's tight bounding box do the trimming
        # instead, and the legend measures the drawn extent for itself.
        fig.subplots_adjust(top=0.90, bottom=0.16, left=0.13, right=0.99)
        _verdict_legend(fig, {r.verdict for r in results}, plt)
        fig.savefig(path)
        plt.close(fig)
    return path


# --------------------------------------------------------------------------

def representative(results: Sequence[PairResult], limit: int = 4) -> list:
    """One pair per verdict present, worst first, up to ``limit``.

    We take a spread rather than the worst few. Four failures in a row all say
    the same thing, whereas a failure beside an uncertifiable pair shows the
    distinction the interval exists to make. Within a verdict we take the pair
    the divergence ranks worst, since that is the one worth looking at.
    """
    out = []
    for verdict in (FAIL, NOT_CERTIFIABLE, PASS, NO_MARGIN):
        pool = [r for r in results if r.verdict == verdict]
        if not pool:
            continue
        out.append(max(pool, key=lambda r: r.kl_excess
                       if math.isfinite(r.kl_excess) else -math.inf))
        if len(out) >= limit:
            break
    return out


def write_all(results: Sequence[PairResult], directory: str, reference=None,
              panel: Panel = DEFAULT_PANEL, margin: float | None = None,
              worst: int = 4, fmt: str = "png", **kw) -> list:
    """Write the three figures into ``directory`` and return the paths written.

    ``reference`` is one reference sample, which the resolution curve needs. We
    take the pair with the widest interval when none is named, since that is the
    pair whose resolution binds the campaign.
    """
    os.makedirs(directory, exist_ok=True)
    results = list(results)
    written = []

    chosen = representative(results, worst)
    if chosen:
        written.append(interval_plot(
            chosen, os.path.join(directory, f"fig-intervals.{fmt}"),
            title="Why these pairs reached their verdicts"))

    scored = [r for r in results if r.has_margin]
    if scored:
        written.append(verdict_map(
            results, os.path.join(directory, f"fig-verdicts.{fmt}")))

    if reference is not None:
        current = results[0].n_model if results else None
        written.append(resolution_plot(
            reference, os.path.join(directory, f"fig-resolution.{fmt}"),
            panel=panel, current=current, proposed_margin=margin, **kw))
    return written
