# Examples

Runnable companions to `docs/factor-partition-examples.md`. The models are
defined once, in `models.py`, and consumed by everything here — the model the
docs show, the model the demos run and the model the validation experiment
replicates are the same object.

| script | what it does |
|---|---|
| `three_routes.py` | Example 1: `d = exp(Ax)[By + exp(Cz)](1 + Fw)`. Prints the derived partition (`gcr` / `log-gcr` / `nuts` from one model), samples it, reports posteriors against truth and the prediction-space residual. |
| `hierarchy.py` | Example 2: the field variant `d = exp(Ax)[B w1 + exp(Cz)](1 + F w2)`, `w1 ~ p(. \| y)`. Prints the partition for BOTH parameterisations of `y` (which must be identical — the ancestry ejection is structural), then samples the linear one. |
| `validate_sampling.py` | The validation experiment: is the sampling **correct** and **smooth**? Replications at fresh noise (and, for the hierarchy, a fresh field realisation from its own prior), with the acceptance criteria registered in the script's docstring before any run. |

Every script takes `--quick` (demos) or `--smoke` (validation) for a
CI-sized run, and `--seed`.

## What `validate_sampling.py` measures, and what counts as a pass

Registered up front — the criteria live in the docstring, so moving them to
fit a result would be visible in git:

Two arms per replication: the factor sweep under test, and a **pure-NUTS
control** on the same graph at the same budget — same joint density by
construction, so a control that passes while the sweep struggles says "the
sweep mixes slowly here", not "the posterior is wrong".

**Correct** (judged on the control) — truths drawn from the priors each
replication, so a correct sampler's coverage is exactly nominal and its
pulls unit normal; pooled coverage within two binomial sigmas of nominal
**at the replication count** (`± 2√(p(1−p)/R)` — the unit is the
replication, since one run's scalars share its noise and its chain); and
normalised errors with mean in `±0.5`, rms in `[0.5, 1.7]` — the sharp
criterion at small `R`.

**Smooth** (judged on the factor arm) — the derived partition identical
across every replication (a verdict that flipped with the noise would be
probe luck, not a rule); zero exceptions in either arm; and on the first
three replications a second factor chain, started with the NUTS remainder
half a prior width away, whose agreement with the first is reported — the
cheap detector for valley non-mixing, which Example 1's geometry makes a
live concern.

**Efficiency** (measured, never thresholded) — the factor arm's pull-rms
against the control's at equal budget. The first registered run measured
the sweep 2–4× worse on both of these SMALL models, the finding was kept,
and the docs page carries it with the mechanism and the regime where the
sweep earns its keep (an exact block too large for NUTS to step at all).

The `--smoke` mode runs the same machinery at chain lengths too short to
mix, and does not score the verdicts — usefully, the two-chain detector
*fires* there (shifts of 3–12), which is the detector demonstrating it has
teeth rather than the experiment failing.

`tests/test_examples.py` keeps the demos running as part of the suite;
the full validation run is minutes-long by design (replications are the
price of a coverage statement) and is invoked by hand.
