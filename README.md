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
certify, computed from the reference alone, with no model involved. Below the
resolution a correct model cannot PASS, so a margin set there turns passes into
NOT CERTIFIABLE. A FAIL still means what it says, since a correct model's
interval is centred on zero and cannot lie wholly outside a margin. What a
too-fine margin costs is the ability to certify, not the ability to detect.

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
cd tools/model-scores
uv sync                  # scoring only, numpy
uv sync --extra plots    # adds matplotlib for the figures
```

## Using it properly

The procedure has an order, and taking the steps out of order is the way to get
a number that looks like an answer and is not one.

### 1. Ask what the data can resolve, before choosing a margin

R3 needs no model. It takes the reference alone and says the finest margin a
*correct* submission of a given size could certify. Leave the model directory
out and that is all it does, so the margin can be argued about before anyone
submits anything:

```sh
uv run model-scores REFERENCE.json --resolution 40,120,507
```

```
reference pairs 336, 259 with a meaningful relative margin

achievable resolution, from the reference alone, no model involved.
median over pairs of the panel margin a correct submission could certify:
      40 model runs     7.02%   model share of variance  92.6%
     120 model runs     4.58%   model share of variance  80.7%
     507 model runs     2.91%   model share of variance  49.9%
```

Read that as three statements. At 40 runs, a 10 percent margin is a question
the data can answer and a 5 percent margin is not: a correct model would fail
the second one about as often as a wrong one, so its failures would carry no
information. Getting to 5 percent takes about 120 runs. And the model side is
supplying 93 percent of the variance of the difference at 40 runs, so runs are
what buys resolution here; more trials would buy almost nothing.

Add `MODEL_DIR` back and the same table prints after the scoring, which is how
to check after the fact that the margin a campaign used was one it could
resolve.

### 2. Score, with the margin that survived step one

```sh
uv run model-scores REFERENCE.json MODEL_DIR --margin 0.10
```

```
reference pairs 336, model pairs 336, shared 336
model runs 40, margin 10.0% relative

336 pairs, panel of mean, median, q90, all required to pass

              PASS   136   40.5%
              FAIL    89   26.5%
   NOT CERTIFIABLE    33    9.8%
         NO MARGIN    78   23.2%
```

A pair is one (metric, series, mode, window). It is scored only when both sides
carry it, the reference has at least 30 values and the model at least 8: a
handful of runs resolves nothing and would land in NOT CERTIFIABLE by
construction, which tells a reader about the sample size rather than about the
model.

### 3. Read the four counts as four different situations

They are not one scale from good to bad.

- **PASS and FAIL** are answers. The data decided.
- **NOT CERTIFIABLE** is a question the campaign could not settle at this
  margin. It is a statement about resolution, not about the model, and the fix
  is runs or a coarser margin.
- **NO MARGIN** is a question nobody posed. The reference statistic is
  indistinguishable from zero, so a relative margin has nothing to be relative
  to. In the run above that is 78 pairs, most of them integrity series that sit
  at zero across the board. The procedure asks for an absolute tolerance in the
  metric's own units rather than inventing one from the reference's own spread,
  which is the circularity the whole construction exists to avoid. Supply one
  with `--absolute`, covered next.

If a summary of the campaign has to be a single number, it is not the pass
rate. The pass rate mixes all four of these together.

### 4. Where a relative margin cannot reach, state one in units

```sh
uv run model-scores REFERENCE.json MODEL_DIR --margin 0.10 \
    --absolute 'integrity.*.independent:mean=0.01,integrity.*.independent:median=0.01,integrity.*.independent:q90=0.02'
```

```
absolute margin on integrity.*.independent: mean 0.01, median 0.01, q90 0.02

              PASS   229   68.2%
              FAIL    66   19.6%
   NOT CERTIFIABLE    40   11.9%
         NO MARGIN     1    0.3%
```

A scope is a dotted prefix of `metric.series.mode.window`, with `*` for any one
segment. Scoping has to reach past the metric name, and the run above is why:
integrity cumulative sits near one while integrity independent sits at zero, so
a tolerance of 0.01 that is generous for the second is a one percent test on
the first. Stating `integrity:` for both turns 17 passing pairs into failures
and tells you nothing about the model.

Two things to be clear about before quoting a number produced this way. Within
its scope the absolute margin **replaces** the relative one for every pair, not
only for the pairs that had no relative margin: that is why FAIL also falls
here, since the independent pairs that did have a base were being held to ten
percent of a number near zero. And the values have to come from the program.
The ones above are illustrative, chosen as roughly one to two standard
deviations of the independent series, and they are not anybody's stated
tolerance.

### 5. Look at the intervals, not only the verdicts

```sh
uv run model-scores REFERENCE.json MODEL_DIR --margin 0.10 --detail 5
```

```
integrity.alice_6_integrity.independent.t11700_t12600   n_ref=507 n_model=40   NO MARGIN
        mean     -0.00 [   -0.00,    -0.00]   NO MARGIN
      median     +0.00 [   -0.00,    +0.00]   NO MARGIN
         q90     -0.00 [   -0.00,    -0.00]   NO MARGIN
     KL 1.521 bits, floor 0.269, excess +1.252   model share of variance 93.0%
```

The verdict is a reading of the interval, and a reader who disagrees with the
margin can still use the interval. Two pairs that both say FAIL can be quite
different: one whose interval sits well outside the margin is a model that is
wrong, and one whose interval is merely wide is a campaign that did not look
long enough. `certifiable_margin` on each quantity says the tightest claim the
pair would have supported.

### 6. For latency, reduce each run before comparing

Latency is the one metric that arrives as a distribution per run rather than a
scalar, and the evaluator keeps it in its own tree: the compiled file's
`latency` block holds 507 filenames, not values.

```sh
uv run model-scores REFERENCE.json MODEL_DIR --margin 0.10 \
    --latency ~/latencies
```

```
reference pairs 392, model pairs 390, shared 390
latency reduced per run to p50, p90, since the run is the unit of randomness
```

The model side needs no second directory: each per-run file already carries the
same block. Both sides are reduced the same way, one number per run per window,
and the `series` slot of the key names which summary it is, so
`latency.p50.cumulative.t0_t9900` is the across-run array of per-run median
latencies. `--latency-stats` chooses them, from `mean`, `p50`, `p90`, `p95` and
`p99`.

**Why not pool the messages.** Because messages inside a run share its
conditions. Resampling a pooled bag of every message from every run treats
correlated observations as independent, and returns an interval far tighter
than the campaign earns. Reducing per run keeps the bootstrap resampling runs,
which is what happens with every other metric here. The test for this is in
`tests/test_latency.py`, and on a corpus with real between-run variation the
pooled interval comes out more than five times too narrow.

**Read latency per member, not per panel.** The panel verdict is the worst
member and the panel resolution is also the worst member, and on latency they
are not the same member:

```
latency.p50.cumulative.t0_t9900   n_ref=507 n_model=40   FAIL
        mean    -42.64% [  -52.41,   -32.70]%   margin 10.0%   FAIL
      median    -38.06% [  -60.55,    -6.24]%   margin 10.0%   NOT CERTIFIABLE
         q90    -56.55% [  -62.16,   -50.22]%   margin 10.0%   FAIL
   achievable at 40 runs, per member: mean 46%, median 150%, q90 29%
```

The across-run median is the unresolvable member here, at 150 percent, while
the mean and the upper decile resolve to 46 and 29 percent and both fail
decisively. A single panel-level resolution of 150 percent would have hidden
that.

### 7. Treat the divergence as a ranking, not a gate

The KL figure is in bits, computed the way `hcs_detect` computes it, and scored
against the floor a correct model of the same size attains. Excess above that
floor orders the disagreements, which is what it is for. No margin on it means
one fixed thing across pairs, so it decides nothing.

## Running it from elsewhere in the tree

Paths are relative to where you are:

```sh
uv run --project tools/model-scores model-scores \
    .build/cp3/compiled/compiled_performance_metrics.json \
    .build/cp3/model/results/single11 --margin 0.10
```

## What it expects as input

**`REFERENCE.json`** is the evaluator's `compiled_performance_metrics.json`,
whose top-level key is `compiled_performance_metrics`. Each leaf there is
already the across-trial list, so nothing is aggregated on the way in.

It is a delivered artifact, not something this repo builds.
It carries its own provenance beside the metrics:
`source_count` is 507 and `source_keys` lists all 507 of them, each of the form
`pwnd_cp3_scenario_1_final_20260804143611/performance_analysis/performance_metrics.json`.
So the reference is 507 trial runs of CP3 scenario 1 from the 2026-08-04
campaign, with one per-run file per trial compiled into across-trial lists. Each
of those per-run files has the same shape as the ones in `MODEL_DIR`: the
evaluator's side and ours differ in where the runs came from, not in what a run
records.

Two things follow from that. The reference cannot be regenerated here, so a
question about how a number was produced is a question for the evaluator.
And `scripts/aggregate_performance_metrics.py` in this repo, which does the
same job for our own runs, is not what produced it: that one writes
`combined_results.json` with no wrapper key and no source list.

The compiled file also carries a `latency` block, and it holds no metric
values: it is a list of 507 filenames pointing into a separate `latencies/`
tree. Pass that tree as `--latency DIR` and the pairs join the comparison. Each
file there is one trial, shaped `{mode: {window: [one latency per message]}}`,
and the model's own per-run files carry the same block under `latency`.

**`MODEL_DIR`** holds one `*_performance_metrics.json` per run, named like
`2026-09-02_run10_performance_metrics.json`. Each file carries one scalar per
key, and the across-run array is assembled from the set of files. The run count
in the header is the number of files found, so a short count there is the first
thing to check when a result looks unresolved.

Both are walked into keys of the shape `(metric, series, mode, window)`.
`integrity` is keyed by series; `goodput` and `availability` repeat the metric
name in the series slot so that every key has the same shape.

`corpus.py` is the only module that knows any of this. Everything above it
takes plain arrays, so a different campaign means replacing that one file.

## The flags that change an answer

| Flag | Default | What it decides |
| --- | --- | --- |
| `--margin` | `0.10` | The relative margin, as a fraction of the reference statistic. Step one above is how to choose it. |
| `--quantities` | `mean,median,q90` | Panel members, all of which must pass. Add `q99` when the tail is the point, `stdev` when dispersion is part of the claim. A longer panel is a stricter test, and needs no multiplicity correction. |
| `--alpha` | `0.05` | Each interval is a two-sided 90 percent interval at this alpha. |
| `--reps` | `2000` | Bootstrap resamples. Lower is faster and noisier; the verdict near a boundary is the part that moves. |
| `--seed` | `20260917` | Resampling is the only stochastic thing here, so the same seed gives the same report. Changing it is a way to check that a borderline verdict is stable. |
| `--resolution` | off | Sizes to report the achievable margin at. With no `MODEL_DIR`, the only thing reported. |
| `--absolute` | off | `[SCOPE:]QUANTITY=VALUE`, comma separated. Margins in the metric's own units. A more particular scope narrows a more general one. |
| `--latency DIR` | off | The evaluator's latency tree, one file per trial. Adds 56 pairs and takes about 15 seconds to read. |
| `--latency-stats` | `p50,p90` | Within-run summaries latency is reduced to, from `mean`, `p50`, `p90`, `p95`, `p99`. |
| `--detail N` | `0` | Full panel for the N worst pairs, worst by verdict then by divergence excess. |
| `--out` | off | The whole result set as JSON, including every interval. |
| `--plots DIR` | off | The three figures. Needs the `plots` extra. |

## Five ways to get it wrong

- **A margin finer than the resolution.** Passing becomes unreachable, so the
  campaign reports NOT CERTIFIABLE where it should have reported PASS. A FAIL
  at such a margin is still a real difference. Run `--resolution` first, and
  per quantity when the panel members disagree about what is resolvable.
- **Reading NO MARGIN as a failure.** It credits the procedure with an answer
  it did not produce. Those pairs need an absolute tolerance from the program.
- **Scoping an absolute margin too broadly.** A tolerance in units means one
  thing on a series near zero and quite another on a series near one, even when
  both carry the same metric name.
- **Reading NOT CERTIFIABLE as a pass.** It is not a near miss in either
  direction. It says the campaign was too short to tell.
- **Quoting the pass rate alone.** It buries the distinction the four verdicts
  exist to draw, which is the distinction the old pass bit could not make.

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

`load_reference_latency` and `load_model_latency` return the same keyed arrays
from the latency trees, and `merge` puts several adapters into one comparison.

`score` takes `absolute_margins` keyed by quantity for one pair, and `score_all`
takes `absolute` keyed by scope and then by quantity for a whole campaign, which
is what the command builds from `--absolute`. `has_relative_margin(sample)` is
the zero guard on its own, and answers whether a pair can carry a relative
margin at all without scoring anything.

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
| `corpus` | The CP3 file format, latency included, and the only module that knows it. |
| `cli` | Argument parsing and reporting. |
| `plots` | The three figures. Needs the `plots` extra. |

Everything above `corpus.py` takes plain arrays. As such, scoring a different
campaign is a matter of replacing that one file, and nothing else.

## Open Questions

- The margins are ours, not the program's. R3 says what a margin *could* be; it
  does not say what it *should* be. Those numbers want agreeing with the
  evaluator before the next campaign.
- Latency covers the aggregate stream only. The model's per-run files also
  carry `latency_by_client` and `latency_to_client`, which would localise a
  latency gap to a client the way the integrity series does. The evaluator's
  tree carries no such breakdown, so there is nothing to compare them against
  today.
- The divergence diagnostic is calibrated for ranking, not for gating. We have
  not found a margin on it that means one fixed thing across pairs, and we are
  somewhat doubtful that one exists.
