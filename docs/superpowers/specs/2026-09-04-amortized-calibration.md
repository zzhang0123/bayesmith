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
runnable. **Pin:**
`tests/evaluation/test_sbc.py::test_the_reference_npe_is_calibrated_through_the_sampler_arm`
(full layer).

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
printed digit across two runs, x64-clean, and calibrated here (SBC PASS,
p = 0.1958). It does not yet clear rows 1 and 3. Row 1 is **unmeasured**: one
1-D linear-Gaussian problem is not this project's problem family. Row 3 is
partial in two ways — its 90% interval coverage is **0.843**, outside the
`[0.848, 0.952]` band the reference (0.890) and the exact posterior (0.910)
both sit inside, and it spent 13.2 s training plus 2.1 s sampling against the
reference's 1.8 s, on a problem where a closed form exists.

Nothing here says the reference NPE is *good*. It says it is calibrated on the
problem it was written for, at a cost nothing measured here beat, and that no
candidate has yet earned the replacement §1.5 requires.

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
| 3 | representative benchmark on correctness, compile time, runtime, memory | **partial** | Correctness on this problem: SBC **PASS**, KS D = 0.0617, p = 0.1958. But 90% interval coverage **0.843** against the reference's 0.890 and the exact posterior's 0.910 — outside the three-sigma binomial band `0.90 ± 3·sqrt(0.9·0.1/300) = [0.848, 0.952]` that both the reference and the control sit inside. Runtime 13.2 s to train + 2.1 s to sample, against the reference's 1.8 s. Memory unmeasured. |
| 4 | maintenance, traceable versions and failure behaviour | **partial** | Released, versioned, and `train` returns an `Info` carrying the per-epoch losses. Its early stopping is on by default and fired here: 64 epochs of the 188 requested, so it spent **512** of the 1500 gradient steps the budget allowed. That is legitimate behaviour, and it is also why the runtime comparison above is generous to sbiJAX rather than harsh. |
| 5 | thin adapter | **PASS** | `probe_29.run_sbijax` is 65 lines including comments. One real friction: `sbijax.sample` conditions on ONE observable — handed a (300, 8) stack it returns 300 × 200 draws for a single condition — so the replicate loop is required, not a missed optimisation. |
| 6 | absent optional dependency → explicit Refusal, contract test, independent oracle | **not reached** | As above. |

### Reference `NeuralPosterior` (this package)

Recorded for comparison; it is not a candidate for its own replacement.

| # | threshold | measured |
|---|---|---|
| 1 | generality | Deliberately narrow: a Gaussian-mixture NPE, kept as reference/compatibility only (top-level design §2.3). |
| 2 | JAX / x64 / RNG | Native JAX and Equinox; `sample(datum, key, n)` takes a key per call, which is why it drops into the harness's sampler arm with a two-line wrapper. |
| 3 | correctness / runtime | SBC PASS, p = 0.1159; width 0.954 – 1.029 of the exact posterior; 1.8 s to train at this budget. |
| 4 | failure behaviour | `train_posterior` refuses a non-finite estimator (D43) and reports `best_step`; here 322 of 1500. |
| 5 | adapter | None — it is first-party. |
| 6 | optional-dependency refusal | Not optional. |

---

## The pin, and what it can still catch

`tests/evaluation/test_sbc.py::test_the_reference_npe_is_calibrated_through_the_sampler_arm`,
full layer, ~8 s. It imports probe_29 rather than restating the fixture, so
the graph, the bank, the budget and the seeds have one home.

**The load-bearing assertion is the property** — APPLICABLE × PASS at
`ALPHA / K = 0.05`, over all 300 replicates with none refused, unconverged or
undrawn. **Expected false positives: 0.05**, by construction, because that is
what ALPHA declares.

**There is deliberately no assertion on the KS digits.** At N = 300 a PASS
already means `D <= 0.077832` (`scipy.stats.kstwo.ppf(0.95, 300)`), so any
band on D loose enough to survive a
platform change is implied by the PASS and can never fail on its own. The same
turned out to be true of coverage: sweeping a width distortion at this seed
— applied in the SAMPLER, by rescaling each replicate's draws about their own
mean, so that one training run serves the whole row — the three-sigma binomial
band `0.90 ± 3·sqrt(0.9·0.1/300) = [0.848, 0.952]` is crossed only where the
KS verdict has already failed —

| width factor | verdict | KS p | 90% coverage |
|---|---|---|---|
| 0.5 | FAIL | 0.000000 | 0.577 |
| 0.8 | FAIL | 0.000820 | 0.817 |
| 0.9 | FAIL | 0.017384 | 0.857 |
| **1.0** | **PASS** | **0.115886** | **0.890** |
| 1.1 | PASS | 0.249229 | 0.910 |
| 1.2 | PASS | 0.055759 | 0.940 |
| 1.3 | FAIL | 0.004596 | 0.957 |
| 2.0 | FAIL | 0.000000 | 0.997 |

— so the drift detector reads a quantity the verdict cannot see: the
estimator's width against the **closed form**. Its FORM is derived (the target
is exact, so 1.0 is where a correct estimator sits); its CONSTANTS 0.80 and
1.20 are **fitted**, at more than four times the largest deviation measured
(the binding side is 0.9539, 0.0461 from 1.0, against a half-width of 0.20), and
shaped like `tests/test_amortize.py::test_the_posterior_width_matches`' own
`0.75 < ratio < 1.35` at a larger budget.

The 1.2 row is why that detector exists: a 20% width error passes the KS test
at this N and fails the width band. That row was then re-measured as a SOURCE
mutation (M2 below) rather than a sampler-side rescaling, and the two agree:
`conclusion: PASS`, `theta: KS D=0.0767 p=0.05576 over 300 ranks, against
alpha/K=0.05`, coverage `0.94` — inside the band — and ratios
`[1.1592, 1.2345, 1.1447]`.

### Mutation table

Each mutation applied to the source at `b1fcb46`, the pin run, the tree
restored with `git checkout -- src/`. The restore is narrowed to `src/`
because that is where the mutants are and this batch's own work is entirely
outside it — CLAUDE.md's rule, applied. Only exit **1** is a test failure.

| # | file | mutation | what died | exit |
|---|---|---|---|---|
| M1 | `amortize.py` | `NeuralPosterior.sample`: `scales[component]` → `2.0 * scales[component]` | the **PASS** — `Conclusion.FAIL`, `theta: KS D=0.1967 p=1.214e-10 over 300 ranks, against alpha/K=0.05` | `1` |
| M2 | `amortize.py` | same, `1.2 * scales[component]` | the **width band** — ratios `[1.1592, 1.2345, 1.1447]`. The PASS **survives**, measured: `conclusion: PASS`, `KS D=0.0767 p=0.05576`, coverage `0.94`. This mutant is the whole argument for the band | `1` |
| M3 | `evaluation/sbc.py` | `_accumulate`: `truth[index]` → `truth[0]`, ranking every replicate against the first one's truth | the **PASS** — `theta: KS D=0.8733 p=2.478e-269` | `1` |
| M4 | `evaluation/sbc.py` | `sbc_ranks`: `route = "sampler"` → `route = "npe"` | the **census** — `(300, 300, 0, 0, 0, 'npe') == (300, 300, 0, 0, 0, 'sampler')` fails at index 5 | `1` |

Restore verified after each: `git status --short src/` empty, and the pin
green again at `1 passed in 7.90s`, exit `0`. That last check is not
ceremony — CLAUDE.md records a session where two repairs were silently
reverted by a later run's own opening restore, and "reverted" looked exactly
like "did not work".

M1 was then re-run against the committed tree rather than the pre-commit one,
because the whole point of "HEAD has to be what you want back" is that the
table describes the code that shipped: same mutation, same failure
(`theta: KS D=0.1967 p=1.214e-10`), `PYTEST_EXIT=1`, `RESTORE_EXIT=0`,
`git status --short` empty, pin green again at `1 passed in 7.61s`, exit `0`.

**What M4 says about the census.** `sbc_report` abstains *before* it passes
whenever a replicate is missing, so given a PASS the numeric part of the
census — 300 requested, 300 usable, nothing refused or unconverged or undrawn
— is already implied. It is asserted anyway because §8's gate G8 requires the
verdict be recomputable from the findings alone, and because the route label
is the one element a PASS does not imply. M4 is the kill for that element.

---

## What this page does not claim

* **That the reference NPE is good.** SBC asks whether a route's stated
  uncertainty is consistent with its stated prior. `sbc.py`'s own module
  docstring records the blind spot this leaves: a "posterior" that discards
  the observation and returns prior draws is uniform in rank BY CONSTRUCTION
  and scores PASS. The width-against-closed-form numbers above are the half of
  this page that says the estimator read the data.
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

