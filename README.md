# model-scores

Panel scoring for model-versus-trial comparison. This package implements the
procedure recommended in `../../docs/tost-critique.pdf`. We replace a single
standardised distance and a pass bit with four things.

**R1, the panel.** A set of named quantities, each scored against a margin in
its own units, all of them required to pass. That is an intersection-union
test, so it holds level alpha overall and owes no multiplicity correction.

**R2, the interval.** Every quantity returns an interval, and the verdict is a
reading of it. A reader who disagrees with our choice of margin can still use the
interval.

**R3, the resolution.** The finest margin a correct model of a given size could
certify, computed from the reference alone, with no model involved. Note that a
margin finer than the resolution asks a question the data cannot answer, and its
failures therefore carry no information.

**R4, the variance share.** Which side of the comparison supplies the noise,
and how the resolution tightens as the submission grows.

A divergence diagnostic sits beside the verdicts, in bits, computed the way
`hcs_detect` computes it and scored against the floor a correct model of the same
submission size attains. It ranks disagreements. It does not gate them.

## Verdicts

| Verdict | Meaning |
| --- | --- |
| `PASS` | The whole interval lies inside the margin. |
| `FAIL` | The whole interval lies outside it. |
| `NOT CERTIFIABLE` | The interval straddles the boundary. The data do not decide the question at this margin. |
| `NO MARGIN` | The reference statistic is indistinguishable from zero, so a relative margin is not meaningful and no absolute one was supplied. |

`NO MARGIN` outranks the others when a panel disagrees. The question was never
posed in those cases, and reporting it as a failure would credit the procedure
with an answer it did not produce.

## Install

```sh
cd ~/model-scores
uv sync                  # scoring only, numpy
uv sync --extra plots    # adds matplotlib for the figures
```

## Run

```sh
uv run model-scores REFERENCE.json MODEL_DIR --margin 0.10 --resolution 40,507
```

From elsewhere in the tree, with paths relative to where you are:

```sh
uv run --project ~/model-scores model-scores \
    .build/cp3/compiled/compiled_performance_metrics.json \
    .build/cp3/model/results/single11 --margin 0.10
```

## Figures

```sh
uv run model-scores REFERENCE.json MODEL_DIR --plots figs --plot-format pdf
```

Three figures, which are the three things a table conveys badly.

**`fig-intervals`** puts each quantity's interval against its margin, one panel
per pair. This is the figure that separates the two ways a pair can miss: a
model nine percent off sits outside the margin with a tight interval, whereas a
model that is merely unresolved sits near zero with an interval too wide to
certify. The old pass bit rendered both as the same character. We pick one pair
per verdict rather than the worst few, since four failures in a row all say the
same thing.

**`fig-resolution`** plots the achievable margin against submission size, one
curve per quantity, with the proposed margin as a horizontal rule. Where a curve
crosses that rule is the number of runs the margin requires. Recall that no model
enters the calculation, so the figure can be drawn and agreed before a submission
exists. This is the R4 argument in one picture.

**`fig-verdicts`** maps the verdict of every pair over series and window, faceted
by mode. A tally says how many failed. It does not say whether the failures are
scattered or concentrated, and that difference is the whole diagnostic value of
the campaign: a row that fails end to end points at one mechanism, whereas the
same count sprinkled across the grid does not.

On color: the four verdicts take the reserved status palette, and the three
curves take the first three categorical slots, which are the three that clear the
all-pairs separation floors. The status greens and reds do not separate under
deuteranopia, so every cell and marker carries a glyph (`P`, `F`, `?`, `-`) and
the legend carries the glyph beside the swatch. Color never carries a verdict
alone, and the figures therefore survive grayscale printing.

These are static figures for a document, so they ship no hover layer and no dark
mode.

## Library use

```python
from model_scores import Panel, Q99, score, panel_resolution, variance_share

result = score(reference, model, Panel(), margin=0.10)
print(result.verdict)
print(result["median"].interval)          # the product
print(result["median"].certifiable_margin)  # the tightest claim supported

panel_resolution(reference, n_model=40)["panel"]  # the margin to publish
variance_share(reference, n_model=40)             # who is supplying the noise
```

Add `Q99` to the panel when the tail is the point of the comparison, and `STDEV`
when dispersion is part of what the model claims to reproduce.

## Tests

```sh
uv run pytest
```

The suite is behavioral. Each case builds a model whose relationship to the
reference we know, and asserts the verdict the procedure ought to reach. The
point is that the four outcomes mean what the critique says they mean, rather
than that every line is covered.

## Layout

| Module | Holds |
| --- | --- |
| `verdicts` | The four outcomes and the intersection-union rule. |
| `quantities` | `Quantity`, `Panel`, the statistic catalog. |
| `bootstrap` | Resampling. The only stochastic code in the package. |
| `results` | `Interval`, `QuantityResult`, `PairResult`. |
| `divergence` | The KL diagnostic and its floor. |
| `resolution` | R3 and R4, both model-free. |
| `scoring` | R1 and R2. |
| `corpus` | The CP3 file format, and the only module that knows it. |
| `cli` | Argument parsing and reporting. |
| `plots` | The three figures. Needs the `plots` extra. |

Everything above `corpus.py` takes plain arrays. As such, scoring a different
campaign is a matter of replacing that one file, and nothing else.

## What remains open

- The margins are ours, not the program's. R3 says what a margin *could* be; it
  does not say what it *should* be. Those numbers want agreeing with the
  evaluator before the next campaign.
- Latency is not covered. The per-run performance metrics carry integrity,
  goodput and availability. The latency tree is a separate corpus and would need
  its own adapter.
- The divergence diagnostic is calibrated for ranking, not for gating. We have
  not found a margin on it that means one fixed thing across pairs, and we are
  somewhat doubtful that one exists.
- The figures are built for print. An interactive version would want a hover
  layer on the interval panels and a dark mode stepped from the same ramps,
  neither of which is drawn here.
