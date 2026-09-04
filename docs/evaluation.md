# Model checking: eight report kinds, two axes, and what a PASS does not mean

> **文档状态：`module-spec`** · 已发布模块/能力的当前设计文档，从属于顶层设计；改动对应代码须同步本页。索引见 docs/README.md。

`bayesmith.evaluation` is the layer R3 opened between the execution adapters
and a workflow. Everything in it reads a finished `PosteriorResult`,
`PredictiveResult` or `SimulationResult` and produces an `EvaluationReport`
whose `subject_ref` points back at what it read. It writes no Result, modifies
none, and chooses no algorithm.

Three things it deliberately does not do, from the top-level design §2.4:

* It does not **modify** a result. A check that could reweight a posterior
  would be an inference step wearing a verdict's schema.
* It does not **choose** an algorithm. Deciding a check failed and re-running
  with a longer chain is a workflow's decision, taken from a report.
* It does not **re-judge** a verdict that already has a home. `identifiability`
  and `prior_sensitivity` decide against thresholds owned by
  `bayesmith.diagnose`; this layer reads those verdicts and files them.

Dependencies run one way — `evaluation` imports `dispatch`, `graph`,
`artifacts` and `bridge.arviz`, and none of them imports it.
`tests/test_layering.py` holds that direction; the shortcut it forbids
(`dispatch` importing a check so that a run could grade itself) is exactly the
arrangement §2.4 describes as the thing to avoid. **ArviZ stays optional**: it
is imported inside the one function that needs it, never at module scope and
never through the package `__init__`, so a clone without it gets an
`UNVERIFIABLE` report rather than an `ImportError` — and `test_layering.py`
checks that in a subprocess rather than trusting the sentence.

The governing spec is the
[top-level design](superpowers/specs/2026-08-30-bayesmith-top-level-design.md)
§2.4, §3.3 and §8 R3; the execution plan is
[the R3 plan](superpowers/plans/2026-09-02-r3-model-checking.md), whose §0
holds the rulings this page describes. The artifact protocol the reports live
in is [`docs/artifacts.md`](artifacts.md).

---

## The one entry point

```python
from bayesmith.evaluation import check_posterior

gate = check_posterior(
    graph, posterior, key=..., budget=..., model_ref=..., carried=(),
)
```

`check_posterior` runs the checks that apply to one posterior, files each
outcome in its own slot, and hands the slots to
`bayesmith.artifacts.gates.aggregate_gate`. It reaches no verdict of its own
and compares no number against a constant — every threshold it depends on is
owned by the module that decided it. Two of its eight slots are *carried*
rather than run (below).

Each check is also callable on its own: `posterior_predictive_check`,
`prior_predictive_check`, `held_out_report`, `loo_report`,
`simulation_based_calibration` / `sbc_report`, `identifiability_report`,
`prior_sensitivity_report`.

---

## Eight report kinds

`report_kind` is a **code string**, frozen by R1 as a code rather than an enum,
so a new kind of report costs no schema migration. The codes below are the
constants the modules define, not a list retyped here — five are importable
(`heldout.REPORT_KIND`, `loo.REPORT_KIND`, `sbc.REPORT_KIND`,
`diagnostics.IDENTIFIABILITY`, `diagnostics.PRIOR_SENSITIVITY`) and three are
spelled at their call sites.

| `report_kind` | Subject | The question | Written by |
|---|---|---|---|
| `posterior_predictive_check` | `PredictiveResult` | Can the fitted model reproduce the data it was fitted to? | `evaluation.checks` |
| `prior_predictive_check` | `SimulationResult` (PRIOR) | Does the model's own prior generate data like this? | `evaluation.checks` |
| `held_out_prediction` | `PredictiveResult` + its source posterior | How well does it predict the points its mask withheld? | `evaluation.heldout` |
| `loo_psis` | `PosteriorResult` / `PredictiveResult` with a pointwise log density | What is the PSIS-LOO elpd, and is that estimate reliable? | `evaluation.loo` (via `arviz.loo`) |
| `sbc` | `PosteriorResult` of the first usable replicate | Does the route return the posterior it claims? | `evaluation.sbc` |
| `identifiability` | `PosteriorResult` | Were the parameters determined by the data at all? | `evaluation.diagnostics` |
| `prior_sensitivity` | `PosteriorResult` | Would a planned prior perturbation have moved the answer? | `evaluation.diagnostics` |
| `chain_diagnostics` | `PosteriorResult` | Did the sampler converge? | `dispatch.task`, at the time the posterior is made |

`chain_diagnostics` is filed by the execution layer while it makes the
posterior; re-deciding convergence here would put a second owner on it. The
gate accepts it, and `sbc`, from the caller.

---

## The two axes

A report answers on two axes, and the pair is refused at construction if it is
not one of five legal combinations:

* **Applicability** — `APPLICABLE`, `INAPPLICABLE`, `UNVERIFIABLE`.
  `INAPPLICABLE` says the check does not apply to this kind of subject;
  `UNVERIFIABLE` says it applies but the inputs it needs are missing. Only one
  of those is worth chasing, which is why they are not one status.
* **Conclusion** — `PASS`, `FAIL`, `ABSTAIN`.

| Applicability | Conclusion | Legal |
|---|---|:---:|
| `APPLICABLE` | `PASS` / `FAIL` / `ABSTAIN` | yes |
| `INAPPLICABLE` | `ABSTAIN` | yes |
| `UNVERIFIABLE` | `ABSTAIN` | yes |
| `INAPPLICABLE` / `UNVERIFIABLE` | `PASS` / `FAIL` | **refused** |

Only an `APPLICABLE` check may `PASS` or `FAIL`. A check that did not apply, or
could not run, has concluded nothing — and a `PASS` from one would be counted
by `aggregate_gate` as a check that was made. That is the top-level design
§2.4's requirement that a run failure must not be dressed as a `FAIL`, enforced
where the report is built rather than trusted at every call site.

---

## The verdict table

Every row is a branch in the module named beside it, and every non-`APPLICABLE`
route carries a finding code a consumer can read without parsing prose.

### `posterior_predictive_check` and `prior_predictive_check` — `evaluation.checks`

| Situation | Applicability | Conclusion | Finding code |
|---|---|---|---|
| every discrepancy cell inside the band | `APPLICABLE` | `PASS` | — |
| any cell outside the band | `APPLICABLE` | `FAIL` | `discrepancy_outside_band` |
| draws below the D105 floor | `APPLICABLE` | `ABSTAIN` | `draws_below_resolution` |
| the subject holds no replicated observations | `INAPPLICABLE` | `ABSTAIN` | `no_replicated_draws` |
| the simulation did not draw from the prior (prior check only) | `INAPPLICABLE` | `ABSTAIN` | `parameter_source_not_prior` |
| a latent's draws are missing, so the mean cannot be recomputed | `UNVERIFIABLE` | `ABSTAIN` | `discrepancy_needs_latent_draws` |
| the observed node is correlated or non-Gaussian | `UNVERIFIABLE` | `ABSTAIN` | `predictive_noise_unsupported` |
| a discrepancy returned a non-finite value | `UNVERIFIABLE` | `ABSTAIN` | `discrepancy_not_finite` |

The default discrepancy set is five callables — `mean`, `sd`, `smallest`,
`largest` (reading `y` only) and `residual_sd` (reading `y - loc`). A user
callable is recorded by its **importable identity** (`module.qualname`) and
never as an object; a lambda or a REPL function is refused, because a callable
has no canonical form and an artifact may not carry one.

### `held_out_prediction` — `evaluation.heldout`

| Situation | Applicability | Conclusion | Finding code |
|---|---|---|---|
| every held-out PIT inside the Bonferroni-corrected band | `APPLICABLE` | `PASS` | — |
| any held-out PIT in a corrected tail | `APPLICABLE` | `FAIL` | `held_out_point` |
| the graph withholds nothing (mask all `True`) | `INAPPLICABLE` | `ABSTAIN` | `no_held_out_points` |
| latent draws incomplete | `UNVERIFIABLE` | `ABSTAIN` | `latent_draws_incomplete` |
| the source posterior carries no draws | `UNVERIFIABLE` | `ABSTAIN` | `source_posterior_holds_no_draws` |
| correlated or non-Gaussian observation | `UNVERIFIABLE` | `ABSTAIN` | `predictive_noise_unsupported` |

A held-out point is a position where an observed node's `observed_mask` is
`False` — the graph's own statement of what was conditioned on, so there is one
answer to "which points did the posterior see?" rather than two. Conditioning
uses only the `True` positions; prediction reaches every position.
`elpd_heldout` is **reported and never thresholded**: there is no scale on
which one elpd is good without a second model to compare against, and model
comparison is not in this release. The verdict comes from the PIT alone.

### `loo_psis` — `evaluation.loo`

| Situation | Applicability | Conclusion | Finding code |
|---|---|---|---|
| `arviz.loo` returns no warning | `APPLICABLE` | `PASS` | `psis_reliability` (`observed=False`) |
| `arviz.loo` warns (its own `good_k` rule) | `APPLICABLE` | `ABSTAIN` | `psis_reliability` (`observed=True`) |
| the result carries no pointwise log density | `INAPPLICABLE` | `ABSTAIN` | `no_pointwise_log_likelihood` |
| the sample is importance-weighted | `UNVERIFIABLE` | `ABSTAIN` | `weighted_sample` |
| ArviZ is not installed | `UNVERIFIABLE` | `ABSTAIN` | `arviz_unavailable` |

The first two rows are one finding read two ways, not two codes: whenever
`arviz.loo` ran, the report carries all three of `psis_reliability`,
`loo_psis_estimate` (elpd, se, p_loo, `n_data_points`, `n_samples`,
`max_pareto_k`, `good_k`, scale) and `arviz_version`, and it is
`psis_reliability`'s `observed` that decided the conclusion. So the verdict is
recomputable from the findings rather than only reported beside them.

**`loo_psis` has no `FAIL` arm, and that is deliberate rather than an
omission.** A high Pareto k says the importance-sampling *estimate* is
unreliable; it does not say the model is wrong. So a warning maps to `ABSTAIN`,
and the only thing an elpd could fail against — another model — is not
something this release compares. The rule that decides reliability is arviz's,
not a second cutoff of this package's.

A weighted sample is declined rather than averaged over: R2's export carries no
weights, so `arviz.loo` would cross-validate the proposal rather than the
posterior and would return the same numbers whether the weights were
near-uniform or collapsed onto one draw. That branch sits *before* the arviz
import, so the answer is a property of the artifact rather than of the
environment.

### `sbc` — `evaluation.sbc`

| Situation | Applicability | Conclusion | Finding code |
|---|---|---|---|
| every coordinate's rank-uniformity KS p at or above `ALPHA / K` | `APPLICABLE` | `PASS` | `sbc_rank_uniformity` |
| any coordinate below `ALPHA / K` | `APPLICABLE` | `FAIL` | `sbc_rank_uniformity` |
| any replicate produced no rank | `APPLICABLE` | `ABSTAIN` | `replicates_not_completed` |
| usable replicates below the D106 floor | `APPLICABLE` | `ABSTAIN` | `replicates_below_floor` |

Every SBC report is `APPLICABLE`: the harness generates its own subject, so
there is no result it could fail to apply to. When the prior simulation that
feeds the harness is itself refused, `simulation_based_calibration` returns
that `Refusal` rather than a report. As with `loo_psis`, the per-coordinate
`sbc_rank_uniformity` findings and the `sbc_replicate_accounting` finding are
present in **every** report, abstaining ones included — the two ABSTAIN codes
are what *decided*, not all that was recorded, so a reader of an abstained
report can still see how close the ranks were.

The rank is continuous and weighted — `r = Σ_i w_i · 1[θ_i < θ_true]` in
`[0, 1]` — because the classical integer rank would need a resample to accept
weights, and a resample would make the answer depend on a second RNG nothing
declared. **A replicate that did not finish abstains the whole report, with a
count**, never a silently dropped replicate: the three failure buckets
(`refused`, `unconverged`, `undrawn`) all reach the findings. An SBC that
discards its failures and reports uniformity over the survivors is checking an
easier question under this one's name.

The harness takes either arm — a real `PosteriorTask` per replicate (so exact
and sampled routes are covered by one harness), or any
`sampler(datum, key, n) -> draws`, which is how an amortized estimator is
scored. Both reach `sbc_report` with the same ranks.

### `identifiability` and `prior_sensitivity` — `evaluation.diagnostics`

| Kind | Situation | Applicability | Conclusion |
|---|---|---|---|
| `identifiability` | `nullity == 0` | `APPLICABLE` | `PASS` |
| `identifiability` | `nullity > 0` (findings carry the participation) | `APPLICABLE` | `FAIL` |
| `prior_sensitivity` | refit converged, every coordinate verified, worst \|shift\| < `CRITERION_SHIFT` | `APPLICABLE` | `PASS` |
| `prior_sensitivity` | verified, worst \|shift\| ≥ `CRITERION_SHIFT` | `APPLICABLE` | `FAIL` |
| `prior_sensitivity` | refit did not converge, or any coordinate unverified | `APPLICABLE` | `ABSTAIN` |
| either | the diagnostic refused (float32 ambient precision) | `UNVERIFIABLE` | `ABSTAIN` |

These two **project** rather than judge: they read `nullity`, `participation`,
`worst`, `verified`, `refit_converged` and `criterion_std` off the
`bayesmith.diagnose` reports and file them. `diagnostics.py` contains no float
literal at all, and `tests/evaluation/test_diagnostics.py` asserts that — the
same rule stated as something that can fail.

The unverified branch is checked **before** the shift is compared, and the
order is load-bearing: the measured fixture in the tests has a worst shift of
+2.2 σ with a `False` in `verified`, so a projection that looked at the shift
first would file a confident `FAIL` taken from a number nothing cross-checked.

Both diagnostics refuse a float32 ambient precision, and they refuse a graph
whose constants were traced outside the block as well — the graph must be built
inside `jax.enable_x64(True)`, not only the call. Outside one they return an
`UNVERIFIABLE` report carrying the refusal text verbatim, rather than raising.

---

## The declared rate and the two floors

Three registered numbers, and no fourth. Everything else in this layer — the
band, the Bonferroni factors, the tail masses — is arithmetic on them.

| ID | Name | Value | Provenance |
|---|---|---|---|
| D104 | `evaluation.ALPHA` | `0.05` | **borrowed** |
| D105 | `evaluation.DRAW_FLOOR` | `40` | **derived** |
| D106 | `evaluation.REPLICATE_FLOOR` | `100` | **derived** |

**D104 is borrowed**, and saying so is the point: 0.05 is statistics'
conventional default, not a number derived from anything measured in this
repository. A derived threshold and a conventional one answer differently to
"why that value?", and a reader who cannot tell which kind they are looking at
will treat both as negotiable. One constant rather than one per check, so the
close-out can multiply cells by a single rate to state an expected
false-positive count for the suite.

Derived from it, and carrying no number of their own:

* the predictive band is `[ALPHA/2, 1 - ALPHA/2]` per discrepancy;
* held-out prediction over `m` points tests each tail at `ALPHA/(2m)`;
* SBC over `K` latent coordinates tests each at `ALPHA/K`.

**D105 is `ceil(1 / (ALPHA / 2)) = 40`.** A p-value from `N` equally weighted
draws is a multiple of `1/N`. If `1/N` exceeds the `ALPHA/2` tail width, no
attainable p-value lies strictly inside a tail without also being 0 or 1, and
the check has stopped being a measurement. The formula is the registered thing,
so a change to `ALPHA` moves the floor with it rather than leaving a stale 40.

**D106 is 100, and it is not the 50 the R3 plan proposed.** The plan's 50 came
from a sweep of one seed. Task 6.1 required the sweep be repeated over ten
seeds (`jax.random.key(23 + k)`) with the *worst* p-value deciding.
`docs/probes/probe_30_sbc_replicate_floor.py` is that sweep; re-run here (188 s,
exit 0) it reproduces, for a 2×-too-wide and a 2×-too-narrow posterior on the
straight-line fixture:

| N | worst 2× wide | worst 2× narrow | against `ALPHA/3` |
|---|---|---|---|
| 20 | 0.4813 | 0.7045 | undetectable |
| 50 | 0.1384 | 0.1384 | crosses, and crosses `ALPHA` too |
| 100 | 0.0054 | 0.0054 | 3× inside the margin |

At N = 50 the worst seed puts a doubled posterior width at p = 0.1384 — past
`ALPHA = 0.05` itself, so one seed in ten would have missed a 2× error
outright. The plan's own escape clause was taken and the floor raised; the
probe prints that verdict itself rather than leaving it to a reader
(`CROSSES -- raise the floor to 100`). **The floor is a power measurement
against a 2× *width* error, not a promise about every error** — and the same
sweep's control column shows why a single seed could not have settled it: a
CORRECT posterior at N = 100 scored anywhere from 0.0105 to 0.9532 across the
ten.

### The report-level rate is not α, and the arithmetic is stated rather than hidden

The band is `[α/2, 1 - α/2]` **per discrepancy**, and Bonferroni is reserved
for held-out points and SBC coordinates. So a predictive report over `K` cells
fails a correct model with probability up to `1 - (1 - α)**K` — **0.23 at the
default five**. That is the frozen ruling's consequence, not a defect corrected
by inventing a factor the ruling did not grant. What it obliges is that the
number be countable, which is why every cell gets its own finding.

---

## The `model_checking@1` gate

`GateDefinition(name="model_checking", version=1)` declares eight
requirements — two required, six optional:

| Requirement | Required | Run by `check_posterior`? |
|---|:---:|---|
| `posterior_predictive_check` | **yes** | run |
| `identifiability` | **yes** | run |
| `prior_predictive_check` | no | run |
| `held_out_prediction` | no | run |
| `loo_psis` | no | run |
| `prior_sensitivity` | no | run |
| `sbc` | no | **carried** |
| `chain_diagnostics` | no | **carried** |

**The split is not about cost.** `posterior_predictive_check` asks whether the
fitted model can reproduce its data; `identifiability` asks whether the
parameters were determined by that data at all. A posterior that fails either
has not been checked in any useful sense, so their absence must leave the gate
undecided. The other six each answer a question that is real but conditional —
there may be no held-out point, no ArviZ, no calibration campaign.

Two are **carried** rather than run for the same reason no number lives in
`gate.py`: `sbc` costs at least `REPLICATE_FLOOR` posterior fits, so running it
there would mean the gate choosing a replicate count; `chain_diagnostics` is
decided by the execution layer while it makes the posterior. A caller with
neither gets two `NOT_ATTEMPTED` slots, which is the honest record.

`blocked_actions` is empty **on purpose**: the field carries action codes, and
this release declares no action registry. Inventing codes so the field looked
used would create a vocabulary this release cannot honour — the same species of
mistake as inventing a threshold.

`prerequisites_ready` and `inputs_current` are **read, not decided**: the first
is the posterior's own `predictive_ready`, the second its lifecycle status.
A check that raises becomes an `ErrorRecord` in its slot, never an exception out
of the runner — otherwise a gate that could not run a check would produce no
record instead of a record saying so.

The gate truth table itself (`aggregate_gate`'s fixed priority) is documented
in [`docs/artifacts.md`](artifacts.md#gate-truth-table); this layer does not
extend or second-guess it.

---

## What a PASS does not mean

A page that listed what this layer decides without saying what a `PASS` does
**not** mean would be the more dangerous half of the truth. Every item below is
a measurement rather than a caution, and each says where its numbers come
from: the transcripts marked **re-measured here** were produced while this page
was written, and the rest are quoted from the pinning test or record page named
beside them.

### A predictive check has limited power, and the limit is measurable

`curved_line` is a straight-line model fitted to data with a real quadratic
term. At curvature 0.6 the check catches it loudly. At 0.15 — the same model,
the same missing term, a quarter of the size — it **passes all five default
discrepancies** (re-measured here, through the shipped default set):

```
curved_line(0.6)   applicable   fail  mean=0.8815  sd=0.0000  smallest=0.8275  largest=0.0000  residual_sd=0.0000
curved_line(0.15)  applicable   pass  mean=0.4640  sd=0.5495  smallest=0.1615  largest=0.1045  residual_sd=0.3410
```

The honest reading of a pass is *"these statistics of these replicated datasets
do not separate the model from the data"*, and nothing wider. Every predictive
check writes that into its own report's `meta.summary` — the tail of it reads
`a pass bounds these statistics and nothing wider` — because a summary travels
with the artifact and a page does not. (`sbc` gets no such sentence in its
summary; its blind spot is held by the module docstring and the pinning test
named below instead.) The 0.15 case is pinned by a **green test** —
`tests/evaluation/test_checks.py::test_a_real_misspecification_this_check_does_not_catch` —
which is the strongest place to keep a caveat: a reader who takes a `PASS` for
"the model is correct" is contradicted by a passing test in the same file as
the pins.

### An SBC pass does not say the route read the data

A "posterior" that ignores `y` and hands back prior draws is uniform in rank
**by construction** — the position of a prior draw among prior draws is uniform
— so it scores `APPLICABLE × PASS`. Quoted from the pinning cell's own
record, with a sampler that discards its datum and returns
`2.0 * normal(key, (n,))`, N = 100 replicates on the straight-line fixture: KS p = 0.9532, 0.7265, 0.6004 and
0.1842 at seeds 0, 1, 3 and 4 — a pass at every one — with seed 2's 0.0474 the
false positive `ALPHA` declares in advance. The cell that pins it is
`test_a_posterior_that_ignores_the_data_is_still_calibrated`.

SBC asks whether a route's stated uncertainty is consistent with its stated
prior. **A route that is trivially self-consistent answers yes.** Something
that reads the observation — a predictive check, a held-out score — has to be
reported alongside it before anyone says a route works.

A second limit of the same shape, and a property of the *fixture* rather than
of the harness: where the likelihood dominates the prior, a posterior computed
under the **wrong prior** still passes. Quoted from `sbc.py`'s module docstring: at N = 100, a conjugate sampler told the
prior is `N(0, 0.5)` while the model declares `N(0, 2)` gives p = 0.1166 /
0.4411 / 0.4049 at seeds 0 / 1 / 2, and is caught only once the sampler's prior
is wrong by a further factor of five. What SBC has power against is set by the
fixture as much as by the replicate count.

### The amortized battery has a hole, and it is recorded rather than patched

[The amortized calibration record](superpowers/specs/2026-09-04-amortized-calibration.md)
prices what the local reference NPE's pins can and cannot catch. Two results
belong on this page:

* A **datum-path defect** — dropping the mean subtraction from `_mixture`'s
  standardisation — makes that arm fail its own SBC verdict, and the three
  cells that now stand **survive it** (`2 passed, 1 skipped`, exit 0). The
  single cell they replaced caught it, at a margin of 0.0023 from its own
  threshold, while failing 6 of 24 re-runs of its own recipe on correct code. A
  test with a 25% false-positive rate catching a true defect by two parts in a
  thousand is a coin that landed the right way up, and the record says which
  kill was given up and why rather than leaving a reader to find out.
* An NPE whose `_mixture` **discards the datum entirely** passes the SBC
  conclusion and is caught only by a width floor, at width ratios of 15.1805 —
  the same blind spot as the paragraph above, reached by a different route.

### A clean-looking gate PASS can sit beside a check that could not answer

`aggregate_gate` was re-probed here -- one required `PASS` slot beside one
optional slot in six states, reproducing what `gate.py`'s own docstring records:

```
optional slot ABSENT from slots        status=evaluated verdict=pass   n_refs=1 findings=[]
optional slot NOT_ATTEMPTED            status=evaluated verdict=pass   n_refs=1 findings=[]
optional slot UNVERIFIABLE x ABSTAIN   status=evaluated verdict=pass   n_refs=2 findings=[]
optional slot APPLICABLE x ABSTAIN     status=evaluated verdict=pass   n_refs=2 findings=[]
optional slot APPLICABLE x PASS        status=evaluated verdict=pass   n_refs=2 findings=[]
optional slot APPLICABLE x FAIL        status=evaluated verdict=pass   n_refs=2 findings=['optional_report_failed']
```

On the optional axis the status, the verdict and the findings are identical
whether a check was never attempted or was attempted and could not answer.
**`report_refs` is the one field that separates them.** (On the required axis
they are separate findings — `required_report_missing` against
`required_report_abstained`.)

So `check_posterior` **never drops a report**: a check that ran files its
report whatever the report says, and only a check nobody ran gets
`NOT_ATTEMPTED`. A runner that skipped abstaining optional reports because
"they change nothing" would emit a byte-identical clean `PASS` for a run where a
check was attempted and failed to answer. **Reading a `model_checking@1` verdict
without reading its slots is therefore not enough** — the verdict is honest
about what the required checks said, and the slots are where "and six other
things were asked" lives.

### And the coverage that is simply absent

* Correlated and non-Gaussian observed nodes have **no** predictive, held-out
  or predictive-check coverage. They reach `predictive_noise_unsupported` —
  a typed refusal or an `UNVERIFIABLE` report, never an approximation.
  **A prior simulation that ran is not evidence that a check will apply to
  it.** The prior arm samples each node from the distribution the node itself
  declares, so it is defined where the predictive seam is not: re-measured here on a
  graph whose observed node is a `CirculantNormal`, `SimulationTask(PRIOR)`
  returns a `SimulationResult` and `prior_predictive_check` on that very
  result returns `UNVERIFIABLE × ABSTAIN` with `predictive_noise_unsupported`.
  The two have different domains, and the report is where the difference is
  reported.
* **WAIC is not provided.** `hasattr(arviz, "waic")` is `False` in 1.3.0, and
  WAIC is one of the statistics this package reuses rather than reimplements,
  so R3's "LOO/WAIC" ships as LOO-PSIS alone. `tests/evaluation/test_loo.py`
  holds the absence, so a later arviz growing one is news rather than a silent
  divergence.
* No model **comparison**. `loo_psis` reports one model's elpd and its
  reliability; comparing two is a later release's Action registry.

---

## Where these numbers come from

`docs/probes/probe_28_model_checking_seams.py` is committed and runnable, and
is the measurement the R3 plan's §0 quotes:

```bash
PYTHONPATH=. .venv/bin/python docs/probes/probe_28_model_checking_seams.py 100
```

Nine sections; pass section numbers after the replicate count to run a subset.
`docs/probes/probe_30_sbc_replicate_floor.py` is the ten-seed sweep that set
D106, and `docs/probes/probe_29_amortized_candidates.py` is the amortized
comparison. The registered thresholds are in `tests/numerical_gates/registry.py`
with their provenance; the acceptance cells are under `tests/evaluation/`, split
between the fast layer and the `full` layer as `CLAUDE.md` describes.

**And what checks this page.** `tests/test_document_status.py` holds its status
line and its index row in both directions, and nothing else does — so the two
tables above are, today, prose that agrees with the code because it was read off
the code, not prose a test compares against it. `docs/artifacts.md` has a
lightweight content guard for exactly this reason
(`test_the_artifact_docs_pin_the_five_results_and_grounds`, which pins the five
Result names and the `grounds` field without freezing the prose around them).
The equivalent here would pin the eight `report_kind` codes against
`MODEL_CHECKING.requirements` rather than against a list retyped in a test, so a
ninth kind, or a renamed one, would red this page instead of quietly outdating
it. It is not written yet, and this paragraph is the record of that rather than
a plan hidden in a docstring.

---

## Ownership

`bayesmith.evaluation` is first-party core: what it owns is observation
grouping, applicability judgement and the gate semantics. Mature general
statistics stay upstream — ArviZ owns LOO, PSIS and Pareto k̂, and this layer
adds no second cutoff of its own. See [`docs/ownership.md`](ownership.md).
