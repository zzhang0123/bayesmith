# Amortized calibration: the reference NPE's number, and the candidate protocol

> **文档状态：`record`** · 已落地批次/审计/测量的历史记录，写作当天为真，非当前权威。索引见 docs/README.md。

**Date:** 2026-09-04 · **Task:** R3 Task 9 of
[the R3 plan](../plans/2026-09-02-r3-model-checking.md) · **Design basis:**
[top-level design](2026-08-30-bayesmith-top-level-design.md) §1.5 (the six
replacement thresholds), §2.3 ("现有轻量 Gaussian-mixture NPE 先作为
reference/compatibility implementation 保留，只有当 BayesFlow、sbiJAX 或其他候选
在真实 workload 上通过 §1.5 的替换门槛后才退役"), §8 R3's sixth gate ("若没有候选
通过，记录结论而不伪造唯一 production winner").

**Probe:** `docs/probes/probe_29_amortized_candidates.py`, committed and
runnable. **Pins:** three cells in `tests/evaluation/test_sbc.py`, full
layer — `test_the_exact_posterior_is_calibrated_through_the_sampler_arm`,
`test_the_reference_npe_goes_through_the_sampler_arm_over_every_replicate`
(both unconditional) and `test_the_reference_npe_reproduces_its_recorded_calibration`
(recorded, behind a trajectory witness). Why they are split that way is
[below](#the-pins-and-what-they-can-still-catch).

---

## Conclusion

**No candidate passed. The local reference NPE is retained.**

That is one of the two conclusions the plan's Task 9.3 permits, and the
reasons are below rather than summarised.

**BayesFlow 2.0.14** fails §1.5 rows 2 and 3. Row 3: it is **not calibrated**
on this problem at the declared budget — SBC FAIL, KS D = 0.1183, p = 0.0004,
with all 300 replicates usable, so this is a calibration failure and not a run
failure. Row 2: it **exits 1** under `JAX_ENABLE_X64=1`, and x64 is not
optional here — the R3 plan's §0.10 records that this package's
identifiability and prior-sensitivity diagnostics refuse a float32 ambient
environment outright.

**sbiJAX 0.4.0** passes rows 2 and 5 cleanly and is the candidate to
re-examine: native JAX, seeded by `jax.random.key`, reproducible to every
printed digit across two runs, x64-clean in the sense row 2 asks for (it
trains, samples and returns float64), and calibrated here **in float32** (SBC
PASS, p = 0.1958) — under `JAX_ENABLE_X64=1` that same verdict flips to FAIL,
which row 3 records. It does not yet clear rows 1 and 3. Row 1 is **unmeasured**: one
1-D linear-Gaussian problem is not this project's problem family. Row 3 is
partial in two ways — its 90% interval coverage is **0.843**, outside the
`[0.848, 0.952]` band the reference (0.890) and the exact posterior (0.910)
both sit inside, and it spent 13.2 s training plus 2.1 s sampling against the
reference's 1.8 s, on a problem where a closed form exists.

Nothing here says the reference NPE is *good*. It says that **this trained
instance of it**, at these seeds, is calibrated on the problem it was written
for, at a cost nothing measured here beat, and that no candidate has yet
earned the replacement §1.5 requires. The qualifier is load-bearing and was
measured after the fact: re-running the identical recipe at 24 other
init/train seed pairs fails the same KS test in 6 of them
([below](#the-pinned-subject-was-a-trajectory-not-a-property)). That is a
statement about the estimator at this budget, not about the harness.

---

## What was measured, and by what

Every arm below goes through **the same harness**: Task 6's
`bayesmith.evaluation.sbc.sbc_ranks` sampler arm, which §0.11 of the plan
promised would accept any `sampler(datum, key, n) -> draws`. This is where
that promise is spent. The rank definition, the KS test, the Bonferroni level
and the verdict all come from `sbc_report`; no arm scores itself.

```
PROBE29_BAYESFLOW_PYTHON=<bf venv>/bin/python \
PROBE29_SBIJAX_PYTHON=<sj venv>/bin/python \
PYTHONPATH=. .venv/bin/python docs/probes/probe_29_amortized_candidates.py
```
```
PROBE29_EXIT=0
```

**Shared conditions.** Bank `draw_bank(key(0), 2048)` — the joint
`theta ~ N(0.5, 2)`, `x = theta * A + N(0, 0.4)` on eight points, from
`tests/test_amortize.py`. Budget 1500 Adam steps at batch 256, lr 1e-3, 10%
held out. Calibration `key(11)`, N = 300 replicates, 200 posterior draws each,
`ALPHA = 0.05`, K = 1 coordinate so the level is 0.05, D106 floor 100.

| arm | verdict | KS D | KS p | 90% coverage | usable | train s | sample s |
|---|---|---|---|---|---|---|---|
| exact posterior (control) | **PASS** | 0.0517 | 0.3867 | 0.910 | 300/300 | — | — |
| reference `NeuralPosterior` | **PASS** | 0.0683 | 0.1159 | 0.890 | 300/300 | 1.8 | in-harness |
| BayesFlow 2.0.14 (FlowMatching) | **FAIL** | 0.1183 | 0.0004 | 0.807 | 300/300 | 12.0 | 52.8 |
| sbiJAX 0.4.0 (MDN, 1 component) | **PASS** | 0.0617 | 0.1958 | 0.843 | 300/300 | 13.2 | 2.1 |

The reference's width against the closed form, at the three observations
`tests/test_amortize.py` scores its own estimator on: **0.9660, 1.0287,
0.9539** of the exact posterior width, with means **+0.2605, −0.0213,
+0.0054** exact sds away.

### The control is not decoration

probe_29 §1 puts the amortize problem's **exact** posterior — closed form, in
numpy, `tests/test_amortize.py::exact_posterior` — through the same harness on
the graph probe_29 builds. Everything else rests on that graph's forward law
being the same joint `draw_bank` samples; if it were not, the exact posterior
of one problem would be scored against replicates of another and would not be
calibrated. It is: PASS at p = 0.3867. That is the boundary check for the
whole page, and it is why "the same bank" is a checked claim here rather than
a comment.

**And it is now checked by a test rather than only by the probe.** The first
version of this page left §1 to the probe, which no test ran; it is now
`tests/evaluation/test_sbc.py::test_the_exact_posterior_is_calibrated_through_the_sampler_arm`,
unconditional, full layer. Measured: a graph/bank divergence of 40% in the
observation noise — `dist.Normal(m, b.NOISE)` → `dist.Normal(m, 1.4 * b.NOISE)`
in `probe_29.amortize_graph` — turns that cell red at KS D = 0.1267,
p = 1.179e-4, `PYTEST_EXIT=1`. A divergence of that size can no longer pass
unnoticed.

The candidate arms add a second such check. Their draws are computed in
another process, in call order, and only that order pairs replicate *i*'s
draws with replicate *i*'s truth. probe_29's `Replay` compares the datum the
harness hands it against the taped one **bit for bit** on every call and
raises rather than ranking, because a silent misalignment would still produce
ranks, a KS test and a verdict — all about nothing.

### The p-value is a draw, so the seed was swept

Holding the trained reference fixed and moving only the harness seed over
`key(11)` .. `key(20)`, N = 300:

| arm | worst p | median p | best p | max D | coverage range |
|---|---|---|---|---|---|
| reference `NeuralPosterior` | 0.1159 | 0.3180 | 0.8815 | 0.0683 | 0.870 – 0.910 |
| exact posterior | 0.1008 | 0.4071 | 0.9795 | 0.0700 | 0.877 – 0.917 |

The two spreads are the same spread. The reference is not *marginally*
calibrated at N = 300; it is drawing from the null the way a correct posterior
does. `key(11)` — probe_29's own seed, not one chosen from this sweep — is an
ordinary draw from it.

**What this sweep does not bear on.** It holds the TRAINED WEIGHTS FIXED and
moves only the replicate draws, so it bounds Monte-Carlo noise in the harness
and nothing else. It is not evidence about a platform change, which moves the
weights. The first version of this page used it that way; the correction, and
the two experiments that do bear on it, are under
[The pins](#the-pins-and-what-they-can-still-catch).

### Where this differs from probe_28 §6, and why the number moved

probe_28 §6 measured the same estimator with **its own** rank loop and its own
replicate draws (`key(5)` truths, `key(6)` observations, `np.mean(s < t)`,
`np.quantile` coverage) and reported KS p = 0.8815, coverage 0.900. Re-run on
this checkout it still reports exactly that:

```
PYTHONPATH=. .venv/bin/python docs/probes/probe_28_model_checking_seams.py 100 6
```
```
train: 2048 bank, 1500 steps, 1.7s, best_step=322
SBC ranks over 300 prior draws: KS D=0.0333 p=0.8815; 90% interval coverage=0.900
```

probe_29 gets p = 0.1159 for the same estimator because it is a **different
replicate set** — the harness draws its own through `SimulationTask(PRIOR)`
from `key(11)` — not because anything regressed. Both numbers are ordinary
draws from the spread in the table above, whose worst is 0.1159 and whose best
is 0.8815: at `key(13)` the harness happens to produce the same rounded
statistic, `D = 0.0333`, and therefore the same `p = 0.8815`. That is a
coincidence of two ten-in-three-hundred rank excursions, not a reproduction of
probe_28's replicate set. Task 9 pins what it measured **through the
harness**, which is the arrangement §0.11 asked for; probe_28 §6's loop stays
the record it was.

---

## §1.5, row by row

Six conditions, all of which an upstream must meet *simultaneously* before
bayesmith gives up a same-layer implementation. A row that was not measured
says **unmeasured**; it does not say "probably fine".

### BayesFlow 2.0.14 (keras 3.15.1, jax backend)

| # | §1.5 threshold | verdict | measured |
|---|---|---|---|
| 1 | general enough for the target problem family, not just a demo | **unmeasured** | Only the 1-D linear-Gaussian problem of `tests/test_amortize.py` was run. Nothing here exercised plates, multi-latent graphs or astronomy-shaped models. |
| 2 | JAX / JIT / PyTree / **x64** / device / RNG compatible, or a measurable and acceptable boundary cost | **FAIL** | `JAX_ENABLE_X64=1` → exit 1, `TypeError: cond branches must have equal output types … true_fun has type float32[] but … false_fun has type float64[]`, raised from `bayesflow/utils/integrate.py:417` inside the FlowMatching ODE integrator. Also: `pip install bayesflow` installs **no** JAX backend (`ModuleNotFoundError: No module named 'jax'` on import until `jax` is installed by hand), and `KERAS_BACKEND` must be set in the environment. |
| 3 | passes a representative benchmark on correctness, compile time, runtime, memory | **FAIL on correctness** | SBC **FAIL**: KS D = 0.1183, p = 0.0004 at 1504 gradient steps — 300 of 300 replicates usable, so this is a calibration failure and not a run failure. 90% interval coverage 0.807 against a nominal 0.90: the posterior is too narrow. Runtime: 12.0 s to train and **52.8 s** to draw 300 × 200 samples, against 1.8 s and in-harness for the reference. Memory unmeasured. |
| 4 | maintenance active, versions and failure behaviour traceable, termination and diagnostics exposed | **partial** | Version and dependency pins are declared (`keras>=3.15.0`) and it is actively released. But its **default** inference network does not build on the pairing it declares: `bf.BasicWorkflow(inference_network=bf.networks.CouplingFlow(), …).fit_offline(...)` raises `RuntimeError: Unable to automatically build the model …`, with and without `standardize`, affine or spline transform, on bayesflow 2.0.14 + keras 3.15.1 + jax. `FlowMatching` and `DiffusionModel` build; the measurement above uses `FlowMatching`. |
| 5 | the adapter is thin — no copy of an upstream state machine, no backend object in the core API | **partial** | `probe_29.run_bayesflow` is 89 lines including its comments and the
`CouplingFlow` probe, and reimplements nothing. But it must set a **process-global** seed: neither `BasicWorkflow` nor `fit_offline` takes one, and unseeded, two runs of the identical command gave final losses 0.452678 / 0.456457 and sample sds 1.9166 / 1.6419. With `keras.utils.set_random_seed(2)` two runs agreed to every printed digit. A process-global RNG sits badly with §9.3's requirement that a random acceptance test fix and declare its seed, and with a harness that derives one key per replicate by `fold_in`. |
| 6 | absent or upgraded optional dependency → an explicit Refusal, with a contract test and an independent oracle | **not reached** | Not a dependency of this package, so there is nothing yet to refuse. The independent oracle exists and was used (`exact_posterior`). The contract test would be the work of adoption; adoption is not recommended. |

### sbiJAX 0.4.0

| # | §1.5 threshold | verdict | measured |
|---|---|---|---|
| 1 | general enough for the target problem family | **unmeasured** | Same one problem. sbiJAX exposes `npe`, `nle`, `nre`, `fmpe`, `npse`, ABC/SMC variants and its own `sbc`, so the surface is broad; nothing here measured it on a graph this project would actually fit. |
| 2 | JAX / JIT / PyTree / x64 / device / RNG compatible | **PASS** | Native JAX (jax 0.11.1, the same version this repository pins), haiku + optax. Under `JAX_ENABLE_X64=1`, fed float64, it trains and samples: exit 0, draws come back `float64`. Seeded by `jax.random.key`, and reproducible: two full probe_29 runs agreed to every printed digit. In the float32 default it returns float32 draws. |
| 3 | representative benchmark on correctness, compile time, runtime, memory | **partial** | Correctness on this problem: SBC **PASS**, KS D = 0.0617, p = 0.1958. But 90% interval coverage **0.843** against the reference's 0.890 and the exact posterior's 0.910 — outside the three-sigma binomial band `0.90 ± 3·sqrt(0.9·0.1/300) = [0.848, 0.952]` that both the reference and the control sit inside. Runtime 13.2 s to train + 2.1 s to sample, against the reference's 1.8 s. Memory unmeasured. **That PASS is a float32-only statement**: re-run on 2026-09-04 under `JAX_ENABLE_X64=1`, same command and same seeds, the verdict FLIPS to **FAIL** — KS D = 0.0817, p = 0.0345, coverage 0.917, with early stopping firing at 43 of the 188 epochs (344 gradient steps) instead of 64 (512). Row 2's "exit 0, draws come back float64" is unaffected and remains true; it is the *calibration* verdict that is not arithmetic-stable. This is the second measured instance of the fragility recorded under [The pins](#the-pins-and-what-they-can-still-catch) below, and it is on a different package. (The float32 re-run reproduced p = 0.1958, coverage 0.843 and 64 epochs exactly; only wall-clock moved, 13.0 s train and 2.4 s sample.) |
| 4 | maintenance, traceable versions and failure behaviour | **partial** | Released, versioned, and `train` returns an `Info` carrying the per-epoch losses. Its early stopping is on by default and fired here: 64 epochs of the 188 requested, so it spent **512** of the 1500 gradient steps the budget allowed. That is legitimate behaviour, and it is also why the runtime comparison above is generous to sbiJAX rather than harsh. |
| 5 | thin adapter | **PASS** | `probe_29.run_sbijax` is 65 lines including comments. One real friction: `sbijax.sample` conditions on ONE observable — handed a (300, 8) stack it returns 300 × 200 draws for a single condition — so the replicate loop is required, not a missed optimisation. |
| 6 | absent optional dependency → explicit Refusal, contract test, independent oracle | **not reached** | As above. |

### Reference `NeuralPosterior` (this package)

Recorded for comparison; it is not a candidate for its own replacement.

| # | threshold | measured |
|---|---|---|
| 1 | generality | Deliberately narrow: a Gaussian-mixture NPE, kept as reference/compatibility only (top-level design §2.3). |
| 2 | JAX / x64 / RNG | Native JAX and Equinox; `sample(datum, key, n)` takes a key per call, which is why it drops into the harness's sampler arm with a two-line wrapper. |
| 3 | correctness / runtime | SBC PASS, p = 0.1159; width 0.954 – 1.029 of the exact posterior and mean +0.2605 / −0.0213 / +0.0054 exact sds from it; 1.8 s to train at this budget. **At these seeds**: 6 of 24 re-runs of the identical recipe fail the same KS test, and `|bias|` reaches 0.5468 — see [The pins](#the-pins-and-what-they-can-still-catch). |
| 4 | failure behaviour | `train_posterior` refuses a non-finite estimator (D43) and reports `best_step`; here 322 of 1500. |
| 5 | adapter | None — it is first-party. |
| 6 | optional-dependency refusal | Not optional. |

---

## The pins, and what they can still catch

Three cells in `tests/evaluation/test_sbc.py`, all full layer, **7.84 s
together** — they share one training run and two harness runs through
`functools.cache`. All three import probe_29 rather than restating the
fixture, so the graph, the bank, the budget and the seeds have one home.

| cell | half | what it asserts |
|---|---|---|
| `test_the_exact_posterior_is_calibrated_through_the_sampler_arm` | **unconditional** | probe_29 §1's control: APPLICABLE × PASS, the whole replicate census, and the draw budget read off the sampler OBJECT |
| `test_the_reference_npe_goes_through_the_sampler_arm_over_every_replicate` | **unconditional** | the same contract for the trained arm, plus a gross-error width floor `0.60 < ratio < 1.60` |
| `test_the_reference_npe_reproduces_its_recorded_calibration` | **recorded / conditional** | behind a trajectory witness: the PASS, the width ratios pinned at ±0.05 and the mean biases at ±0.07 |

The first version of this page was ONE cell, and it pinned the trained
network's PASS unconditionally. Two things were wrong with it, both found by
adversarial review and both confirmed here by re-measuring rather than by
argument; both are recorded below, because the repair is only legible next to
them.

1. **A broken implementation passed it.** A constant `+0.01` added to
   `means[component]` in `NeuralPosterior.sample` — 0.147 exact sds of mean
   bias — left the width ratios untouched, *improved* the KS statistic, and
   went green. The instrument that sees it was already being computed and
   thrown away: `probe.width_against_exact()` returns `(ratio, bias)` and the
   cell wrote `...[0]`. That is [the bias pin](#the-width-instruments-and-which-half-of-each-is-derived),
   and M5 in the table.
2. **The pinned subject was a trajectory, not a property**, and the evidence
   the page gave for its stability did not bear on the question. That is the
   next subsection.

### The pinned subject was a trajectory, not a property

That version said the pin's risk was the declared ALPHA and cited the harness
seed sweep for it. **The sweep does not bear on the question**: it holds the
TRAINED WEIGHTS FIXED and moves only the replicate draws, so it bounds
Monte-Carlo noise in the harness — and the weights are what a platform change
moves. Two experiments that do bear on it, measured on this machine on
2026-09-04:

* **Re-run the identical recipe at 24 other init/train seed pairs**
  (`key(s)` / `key(s+100)`, s = 0..23) — same bank, same harness key `key(11)`,
  same budget, same machine, nothing but the initialisation and the shuffling
  changed. **6 of the 24 fail this arm's own KS test**: p = 0.0025, 0.0207,
  0.0016, 0.0037, 0.0046, 0.0037. `best_step` runs 292 to 1190. The 72 width
  ratios span `[0.8036, 1.1932]`, and `|bias|` reaches **0.5468** exact sds.
  ALPHA is 0.05; a quarter of the re-runs of its own recipe fail.
* **`JAX_ENABLE_X64=1`.** This is *not* an arithmetic-only change — it replaces
  the RNG stream as well — so on its own it over-states a BLAS swap, and this
  page does not claim otherwise. But two of its effects cannot be the stream:
  `best_step` moves **322 → 580**, and one width ratio moves 0.9539 → 1.1103
  and one bias +0.2605 → +0.0753, which are **9× and 8×** the 0.0170 / 0.0232
  Monte-Carlo spread the same statistics show over twelve draw keys at FIXED
  weights. That is the weights diverging, which is what a BLAS swap can also
  do.

**Linux remains unmeasured for this arm.** The honest statement after those
two experiments is not "the risk is the declared 5% and not a hidden
fragility" — it is that the subject is a 1500-step float32 Adam trajectory,
and a trajectory is not a property. A second, independent instance of the same
thing is in §1.5 above: sbiJAX's KS verdict flips PASS → FAIL under
`JAX_ENABLE_X64=1`.

### Where the fragility is NOT

Not in the harness, and not in the rank statistic. A rank is a **count**, and
counts do not move at rounding scale. Nudging every posterior draw by a
relative factor, on both arms:

| nudge | exact arm, ranks changed | reference arm | KS D / p |
|---|---|---|---|
| `1e-7` (≈ float32 eps) | 0 / 300 | 0 / 300 | unchanged |
| `1e-6` | 0 / 300 | 0 / 300 | unchanged |
| `1e-5` | 1 / 300 | 3 / 300 | unchanged |
| `1e-4` | 15 / 300 | 17 / 300 | unchanged |

`D` and `p` are identical to four digits in every row — 0.0517 / 0.3867 for
the exact arm, 0.0683 / 0.1159 for the reference. So the arm that can carry an
**unconditional** PASS is the one with no optimisation in it, and that is why
probe_29 §1 is now a test of its own.

### The ladder, and the rung this took

CLAUDE.md's order, tried in order. **(a) construct deterministically** — not
available; the trajectory is the thing being measured. **(b) assert the
property with a derived band** — not available for the mean bias: over the 24
retrains `|bias|` reaches 0.5468, *above* the 0.4079 that a 0.01 shift in
standardized latent space produces, so no band both survives the recipe and
catches the shift. **(c) make the platform-dependent PREMISE conditional and
recorded while keeping the CONTRACT assertions unconditional and ahead of
it** — this is the rung taken, with **(d) skip loudly** as its else-branch.

**The witness.** The recorded cell asserts nothing until `best_step == 322`
and the validation minimum matches `-0.5286270976` to `rel=1e-4`; otherwise it
skips with `THIS IS NOT A PASS`, naming both numbers it measured.
`best_step` is a 1500-way discrete fingerprint of the optimisation — **24
distinct values over the 24-seed sweep**, and 322 → 580 under x64 — so it
detects a diverged trajectory rather than a diverged bit. The loss is checked
beside it because a coincident argmin index is not a coincident set of
weights. `1e-4` sits between two measured scales: float32 eps on this loss is
6e-8 (900× below) and the gap to the second-best validation value is 1.49e-3
relative (15× above), so no other step of this run can satisfy it.

That branch is reachable and was exercised: mutant M6 below changes training,
and the cell skips printing `best_step=1454 (recorded 322), validation
minimum=2.132643461227417`.

### The width instruments, and which half of each is derived

**Unconditional floor, `0.60 < ratio < 1.60`.** FORM derived — the target is a
closed form in numpy, so 1.0 is where a correct estimator sits and no
measurement of this machine chose it. CONSTANTS fitted, against the 24-retrain
sweep rather than one run: all 72 ratios fall in `[0.8036, 1.1932]`, so the
binding margin is 0.204, twelve times the 0.0170 Monte-Carlo spread.

**This is looser than the `(0.80, 1.20)` it replaces, and that is a
weakening, stated as one.** The reason is that `(0.80, 1.20)` was not an
unconditional claim: the same sweep puts ratios at 0.8036 and 1.1932 — inside,
by 0.004 and 0.007 out of a half-width of 0.20, i.e. 2–3% of it — and x64
moves one ratio by 78% of that half-width. A band with 2% of margin under a
re-run of its own recipe records a trajectory. What it caught is not lost: it
moves into the recorded cell as a **±0.05 pin, four times tighter than the old
band**, behind the witness. M2 below is the mutant that measures the
difference: under `(0.80, 1.20)` it failed on one of three ratios; under the
pin it fails on all three, at 4× the tolerance.

It is deliberately **not** `tests/test_amortize.py::test_the_posterior_width_matches`'
`0.75 < ratio < 1.35`, which owns this criterion in that file at a LARGER
budget (bank 8192, 2000 steps). At Task 9's budget the retrain cloud reaches
0.8036 — 0.054 above that floor, three times the Monte-Carlo spread — too thin
to carry an unconditional guard here.

**Recorded pins, `±0.05` on the ratio and `±0.07` on the bias.** FORM derived —
behind the witness these statistics are reproducible to ten digits in-process,
so the pin's unit is the instrument's own noise floor. CONSTANT fitted: three
times the Monte-Carlo spread the same statistics show over twelve draw keys at
fixed weights (0.0170 and 0.0232), rounded DOWN to two digits.

**The bias pin is new, and it is the repair for a broken implementation that
walked past the first version.** `probe.width_against_exact()` returns
`(ratio, bias)` and the first version wrote `...[0]`, discarding the other
half of the closed-form comparison on the same line. A constant `+0.01` added
to `means[component]` in `NeuralPosterior.sample` — 0.147 exact sds — leaves
the width ratios untouched, *improves* the KS statistic to D = 0.0533,
p = 0.3486, and passed. It is M5 below.

### What the census assertion is, now that a rename cannot walk past it

The census reports the arm as the string `"sampler"`, and a guard that reads a
spelling can be walked past by a rename. Both unconditional cells now check
that string beside an object-level fact: the callable the harness drove is a
`_CountedSampler`, it was driven exactly 300 times, and every call asked for
200 draws. M8 below is the mutant that separates the two — it halves the draw
budget, leaves the census tuple and the KS verdict untouched, and dies only on
the call record.

### Mutation table

Each mutation applied to `84c6611`, the three cells run, `__pycache__` removed,
the tree restored with `git checkout -- src/` (or `docs/probes/` for M7) —
narrowed to the directory the mutants are in, per CLAUDE.md. `RESTORE_EXIT=0`
and `git status --short` empty after every row. Only exit **1** is a test
failure.

Command, for every row:

```
PYTHONPATH=<worktree>/src:<worktree> .venv/bin/python -m pytest \
  -p no:cacheprovider -m full \
  tests/evaluation/test_sbc.py::test_the_exact_posterior_is_calibrated_through_the_sampler_arm \
  tests/evaluation/test_sbc.py::test_the_reference_npe_goes_through_the_sampler_arm_over_every_replicate \
  tests/evaluation/test_sbc.py::test_the_reference_npe_reproduces_its_recorded_calibration
```

| # | file | mutation | which cell died, and on what | exit |
|---|---|---|---|---|
| M1 | `amortize.py` | `sample`: `scales[component]` → `2.0 * scales[component]` | **both** — the unconditional width floor at ratios `[1.9321, 2.0575, 1.9079]`, and the recorded PASS at `KS D=0.1967 p=1.214e-10` | `1` |
| M2 | `amortize.py` | same, `1.2 * scales[component]` | the **width pin** — `[1.1592, 1.2345, 1.1447]` against `0.9660/1.0287/0.9539 ± 0.05`, all three mismatched, max difference 0.2058. The PASS survives it, which is why a width instrument exists | `1` |
| M5 | `amortize.py` | `sample`: `means[component]` → `means[component] + 0.01` | the **bias pin** — `[0.4079, 0.1261, 0.1528]` against `0.2605/−0.0213/0.0054 ± 0.07`, all three mismatched, max difference **0.14742478**, the constant shift. The census, the width and the KS verdict all survive it (D = 0.0533, p = 0.3486, *better* than the pinned 0.1159) | `1` |
| M3 | `evaluation/sbc.py` | `_accumulate`: `truth[index]` → `truth[0]` | the **exact-posterior control**, unconditionally, at `KS D=0.8717 p=1.33e-267`; and the recorded PASS at `D=0.8733` | `1` |
| M4 | `evaluation/sbc.py` | `sbc_ranks`: `route = "sampler"` → `route = "npe"` | the **census**, in both unconditional cells — `(300, 300, 0, 0, 0, 'npe') == (300, 300, 0, 0, 0, 'sampler')` at index 5 | `1` |
| M6 | `amortize.py` | `_mixture`: `features` → `0.0 * features`, an NPE that IGNORES the datum | the **unconditional width floor** at ratios `[15.1805, 15.1805, 15.1805]`. This is `sbc.py`'s own documented blind spot: it is uniform in rank by construction and the SBC verdict PASSES it. The recorded cell SKIPS — it changes training, so the witness fires: `best_step=1454 (recorded 322), validation minimum=2.132643461227417` | `1` |
| M7 | `docs/probes/probe_29…py` | `amortize_graph`: `dist.Normal(m, b.NOISE)` → `dist.Normal(m, 1.4 * b.NOISE)`, the graph drifting from the bank | the **exact-posterior control** at `KS D=0.1267 p=1.179e-4`, and the recorded PASS on the same statistic | `1` |
| M8 | `evaluation/sbc.py` | `sbc_ranks`: `sampler_draws` → `sampler_draws // 2` at the call site | the **draw-budget record on the sampler object**, in both unconditional cells — `[100, 100, …] == [200, 200, …]`. The census tuple, the KS verdict, the width and the bias all survive it; this is the row that says the object-level half is not decoration | `1` |

**And the same broken implementation at the scope where it used to survive.**
The review that found M5 showed it passing
`pytest tests/test_amortize.py tests/evaluation/test_sbc.py -n 4` at
`80 passed`, `PYTEST_EXIT=0`. Re-run here: clean, `82 passed`,
`PYTEST_EXIT=0`; with M5 applied, `1 failed, 81 passed`, `PYTEST_EXIT=1`, the
failure being `test_the_reference_npe_reproduces_its_recorded_calibration`.

### What is still uncaught, said plainly

* A **training-side** mutation that changes the trajectory without making the
  estimator grossly wrong reaches the witness, not an assertion: the recorded
  cell skips and the unconditional floor `0.60 < ratio < 1.60` is wide. M6 is
  the case where the floor catches it anyway; a subtler one would show up as a
  loud skip, which is rung (d)'s behaviour and not a pass, but is also not a
  red test.
* A mean shift **smaller than about 0.15 exact sds** is below what any
  unconditional instrument here can separate from retraining, per the 0.5468
  measurement above. It is caught only where the witness holds.
* §1.5 row 1 for every arm, and Linux for all of them.


## What this page does not claim

* **That the reference NPE is good.** SBC asks whether a route's stated
  uncertainty is consistent with its stated prior. `sbc.py`'s own module
  docstring records the blind spot this leaves: a "posterior" that discards
  the observation and returns prior draws is uniform in rank BY CONSTRUCTION
  and scores PASS. The width- and bias-against-closed-form numbers above are
  the half of this page that says the estimator read the data, and mutant M6
  is the measurement: an NPE whose `_mixture` discards the datum PASSES the
  SBC conclusion here and is caught only by the width floor, at ratios
  15.1805.
* **That BayesFlow or sbiJAX is bad.** Both were run on ONE 1-D
  linear-Gaussian problem, at ONE budget, through ONE adapter written by
  someone who had not used either package before. §1.5 row 1 is `unmeasured`
  for both and that is the honest state.
* **That the candidate comparison is a benchmark.** The budgets are matched as
  far as three different training APIs allow: 1500 gradient steps at batch 256
  became 188 epochs × 8 steps = 1504 for BayesFlow, and 64 of 188 requested
  epochs = 512 steps for sbiJAX, whose default early stopping fired. The
  wall-clock column should be read with that in mind.

## What would change the conclusion

* **BayesFlow:** an x64 path through `FlowMatching` (or any inference network
  that builds), plus a calibrated result at this budget. Both are single
  measurements away and neither is a matter of opinion — re-run probe_29.
* **sbiJAX:** a measurement on a problem this project actually fits, and an
  explanation of the coverage gap (0.843 against a nominal 0.90 while its KS
  test passes). That gap is the sharper of the two signals here and is the
  thing to look at next.
* **Either:** §1.5 row 6 — an adapter with a contract test and an explicit
  Refusal when the optional dependency is missing, on the pattern
  `bayesmith/evaluation/loo.py` already sets for ArviZ.

## Reproducing

Two throwaway venvs, neither a dependency of this package and neither the
repository's own `.venv`:

```
python -m venv bf_venv && ./bf_venv/bin/python -m pip install bayesflow jax==0.11.1
python -m venv sj_venv && ./sj_venv/bin/python -m pip install sbijax
```

Both installs exited `0` on 2026-09-04 (bayesflow 2.0.14 pulling keras 3.15.1;
sbijax 0.4.0 pulling jax 0.11.1, optax, dm-haiku, surjectors, blackjax).
`pip install bayesflow` alone leaves no Keras backend installed, which is why
`jax` is named explicitly above.

