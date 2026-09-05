# R5 execution plan: nested residual evidence

> **文档状态：`plan-active`** · 尚未执行完的计划，仍指导后续工作。索引见 docs/README.md。

**Date:** 2026-09-04 · **Baseline:** `2d873f1` (R4 closed, unreleased; §0.13 — HEAD moved five times while this plan was written, and two suite runs had to be discarded because of it)

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task by task. Steps use
> checkbox (`- [ ]`) syntax. **This document is a plan and nothing more. No task starts until
> the owner has read it; each task then runs in its own execution session.**

**Goal:** answer `Z = p(d | M)` for the models R4 refuses — the ones that leave a residual
integral after exact elimination. The residual integral is run by an **upstream** nested
sampler chosen by a measured bake-off, never by a sampler written here. What bayesmith owns is
the compilation that produces the integral, the eligibility that decides it may be run, the
oracle that grades the answer, and the gate that decides whether the answer may be reported.

**Architecture:** three modules, no new top-level subpackage. `dispatch/evidence.py` gains the
residual-problem compiler (`CompiledEvidenceProblem`) and its eligibility; `bridge/` gains one
adapter module per surviving backend, beside `bridge/numpyro_bridge.py` which is the precedent;
`evaluation/evidence.py` gains the termination, stability and transform reports and their gate
rows. Dependency direction is unchanged.

**Tech Stack:** Python 3.11 frozen/slots dataclasses, JAX (x64), deterministic trapezoid
quadrature as the residual oracle, pytest with declared seeds and budgets, ruff;
`tests/numerical_gates/registry.py` for any new threshold.

**Spec:** [top-level design](../specs/2026-08-30-bayesmith-top-level-design.md) §1.4, §1.5 (the
six upstream-replacement conditions — R5's central instrument), §2.3 (EvidenceResult contents,
RunRecord contents), §3.5 (the nested-sampling decision), §4.3 (gate status/verdict), §4.4
(Refusal vs exception), §6.4, §7.3 (the nine backend questions), §8 R5, §9.1 (oracle ladder),
§9.3 (seeds and declared false-positive rates), §9.4, §10.2, §11.4, §11.9;
[R4 close-out](../specs/2026-09-04-r4-close-out.md); [`docs/evidence.md`](../../evidence.md).
---

## 0. Frozen rulings and execution boundary

These are R5's implementation inputs. They are not re-invented during execution. Every ruling
says what was measured and where. Measurements marked **[planning]** were taken in this
checkout at `895e181` while writing this plan; the owning task re-measures each as its own red
assertion, because a measurement performed once is a measurement that goes stale.

### 0.1 R5 adds no field to any artifact, because R1 already built the seam

**Ruling.** `EvidenceResult`, `EvidenceComponent`, `RunRecord`, `EvidenceTask` and
`InferencePlanRecord` are used exactly as they stand. R4's red line 1 — *no change to R1/R2/R3
frozen schema* — carries into R5 unweakened.

**Why.** **[planning]** Every concept §2.3 requires of a nested-sampling evidence already has a
field, and R4 left them at their defaults:

| §2.3 requirement | field that already exists |
|---|---|
| log evidence | `EvidenceResult.log_evidence` |
| estimated uncertainty | `EvidenceResult.standard_error`, and `EvidenceComponent.standard_error` per term |
| weighted samples (**required**, not the disjunction) | `EvidenceResult.posterior_representation`, with `WeightedDrawsPosterior(draws, log_weights, ess, khat, unreliable, method)` |
| termination information | `RunRecord.termination: TerminationRecord(reason, iterations, message)` |
| repeated-run consistency | `EvidenceResult.repeat_result_refs`, `consistency_report_ref` |
| normalization audit reference | `EvidenceResult.normalization_audit_refs` |
| exact + residual composition | `EvidenceResult.exact_components` and `residual_component`, and `InferencePlanRecord.exact_elimination` / `residual_parameters`, which already refuse a name in both |
| backend identity | `RunRecord.backend: BackendRef`, which **raises** on the name `"auto"` |
| budget, seed, dtype, jax config | `RunRecord.budget / seed / dtype / jax_config` |
| requested backend and its knobs | `EvidenceTask.backend`, `.backend_options`, `.repeat_count`, `.reconstruct_posterior` |

**One row is narrower than §2.3 alone would suggest.** §2.3 permits a disjunction —
*"weighted samples 或 posterior reconstruction capability"* — but §3.5 requires of a
nested-sampling backend that *"backend 的 termination、logZ uncertainty、weighted samples 和
diagnostics 全部进入 EvidenceResult"*. For R5's route the disjunction is closed: weighted samples
are required, so `posterior_representation` is populated or the result is incomplete.

**And one validator R5 must plan around, in two ways.** `log_evidence` and
`EvidenceComponent.log_value` both pass through `_finite`.

* It refuses `inf` and `nan`, so **a residual `log Z` of `-inf` is unrepresentable** — a model
  whose evidence underflows to zero cannot be reported as a number and must become a `Refusal` or
  an ABSTAIN. Task 6 asserts which.
* It also requires `type(value) in (int, float)`, so **`np.float64`, `jax` scalars and 0-d arrays
  all raise `TypeError`.** Every number the adapter files must be passed through `float()` first.
  A backend returns none of these as Python floats.

`EvidenceComponent`'s own docstring states the design R5 executes: *"a number that is
closed-form and a number that was sampled carry different warrants, and a single total would
hide which part the error bar belongs to"* (`artifacts/results.py:399`). R5 fills that in; it
does not redesign it.

**Two consequences that are not free.**

* **`ComputeBudget` has no live-point field.** Its knobs are `draws, warmup, chains,
  max_iterations, max_wall_clock_seconds` (`artifacts/base.py:417`). A nested sampler's central
  knob is the live-point count, and it has no home there. It goes in
  `EvidenceTask.backend_options`, which is what §7.3 question 5 asks for — minimum leakage of
  backend-specific objects into the public surface. **`ComputeBudget` gains no field.**
* **`TerminationReason` is a closed enum**: `COMPLETED, CONVERGED, BUDGET_EXHAUSTED,
  TOLERANCE_UNMET, DIVERGED, INTERRUPTED`. Each backend's termination signal is mapped to
  exactly one member by a **declared table with a test that the table is total**. An unmapped
  backend signal is an error, not a default — a nested sampler that stopped for an unrecognised
  reason must not be recorded as `COMPLETED`. **If a backend's real termination signal has no
  honest member, stop and rule; do not add an enum member silently, because that IS a frozen
  schema change.**

### 0.2 The one capability that does not exist is the prior/likelihood split, and it is first-party

**Ruling.** `CompiledEvidenceProblem` carries `log_prior(θ)` and `log_likelihood(θ)`
separately, plus the parameter layout, the prior transform or prior sampler, and the exact
components already eliminated. Building it is a **compiler pass owned by bayesmith** under
§1.5's first clause ("Graph 语义、结构发现、前提验证和 task-aware compilation"). No backend is
allowed to re-read the Graph.

**Why.** Nested sampling requires the separation; nothing in bayesmith produces it.
`graph/evaluate.py:193` sums every `Probabilistic` node's `log_prob` into one total and then
adds `joint_prior` and every `evidence_terms` entry in one loop. Latent and observed are not
distinguished at any point.

**The partition, and the census that grades it [planning].** Prior side = latent
`Probabilistic` nodes + `graph.joint_prior`. Likelihood side = observed `Probabilistic` nodes
(honouring `observed_mask`) + `graph.evidence_terms`. Over every fixture
`tests/exact/models.py` ships, evaluated at a draw from each graph's own prior:

| outcome | count | reading |
|---|---:|---|
| `log_prior + log_likelihood == log_joint` **bitwise** | 46 | at ONE draw — see below |
| both sides `nan`, `log_joint` also `nan` | 2 | faithful; and the split **localises** the `nan` to the likelihood, which `log_joint` alone cannot |
| prior cannot be sampled at all (`NotImplementedError`) | 1 | `improper_outside_prior` — the case R4 already refuses as `evidence_prior_proper` |
| requires constructor arguments, not run | 5 | census gap, closed by the owning task |

**Denominator: 54 = 49 graphs + 5 fixture functions never instantiated.** §0.3's and §0.15's
censuses use **49 graphs** and carry no uninstantiated rows. Both are right and they are not the
same denominator — §0.15 states that hazard, and this table is where the plan first walked into
it.

Zero cases where a constant moved sides.

**But bitwise equality is NOT a property of the partition, and asserting it would be red line 9.**
`log_joint` accumulates `total = total + jnp.sum(term)` over every `Probabilistic` node in **one**
running sum, in graph order (`graph/evaluate.py:210-217`). Splitting into two accumulators
reorders that sum, and float addition is not associative. Swept over 200 prior draws per fixture:

| fixture | bitwise | worst gap |
|---|---:|---:|
| `straight_line` | 200/200 | 0 |
| `diamond_ancestor` | 200/200 | 0 |
| `three_latent_chain` | 200/200 | 0 |
| **`two_observations`** | **186/200** | 1.8e-12 |
| **`observation_reused_downstream`** | **182/200** | 2.8e-14 |

It fails 7–9% of draws on the two-observation fixtures, and the 46/46 above was **one lucky
draw**. A red assertion written from it would go red or green depending on which draw the
implementer picked — and again differently on Linux, where the `log_prob` kernels contract
differently. Its stated diagnosis ("a constant moved sides") would send the reader hunting for a
term that was never lost.

**So Task 1 asserts two things instead, and neither pins a summation order:**

1. **Per node term, bitwise**: every `Probabilistic` node's contribution appears **exactly once**,
   on the side the partition says. This is what actually catches a moved constant, and it is
   exact by construction rather than by luck.
2. **The recomposition, to a derived band** — `n · eps · max(|log_joint|, 1)`, the shape D107
   already uses — over a **declared multi-draw seed set**, not one draw.

> **Note the asymmetry the naming invites.** `Graph.evidence_terms` holds graph-level
> *likelihood* factors, not evidence — the **R4 plan** records this and declines to rename it
> (`2026-09-04-r4-evidence.md:650`). 〔Execution write-back, red line 11 (Wave A review): this
> line said "the R4 close-out", and the close-out contains **zero** occurrences of
> `evidence_terms`. A citation to the wrong document is the same defect as a false docstring, and
> it propagated from here into `dispatch/evidence.py`'s docstring before a reviewer checked it.〕
> R5 puts them on the likelihood side of the split. **The assignment is asserted by
> consequence** (the bitwise identity above, plus `∫ exp(log_prior) = 1` through R4's
> `audit_prior`), never by the field's name.
### 0.3 The admitted class widens by exactly two rows, and a third is refused by its own name

**Ruling.** `_evidence_structure_refusal` (`dispatch/task.py:1139`) today admits
`not sampled and method == "gcr"` and refuses everything else as
`evidence_residual_integral_required`. R5 admits two more rows and **no others**:

| row | condition | route |
|---|---|---|
| (a) whole-graph exact | `sampled == () and method == "gcr"` | R4's exact assembly, unchanged |
| **(b) exact + residual** | `exact != () and method == "gcr" and sampled != ()` | exact collapse, then nested sampling of the residual |
| **(c) all-residual** | `exact == () and sampled != ()` | nested sampling of the whole thing |
| (d) `method` is `gcr+snis` or `gcr+mh` | any | **refused, under a new premise of its own** |
| **(e) no latents at all** | `exact == () and sampled == ()` | **refused — under a premise whose message is TRUE of it** |

**Row (e) is reachable and the four-row taxonomy missed it. [planning]** A graph of only
`const` / `det` / `observe` traces fine, `plan.exact` and `plan.sampled` are both `None`, and it
is refused today under `evidence_residual_integral_required` with the message *"What is left over
here needs a numerical integral over the residual problem"* — **which is false about it.** Its
evidence is `p(d)` with no integral at all: the likelihood's own normalising constant, exactly
computable. After R5's widening this is the *only* graph still reaching that premise, so Task 7.6's
instruction to delete the untrue part of its message is about row (e) or it is about nothing.
Task 7.4's red **enumerates the reachable `(exact.method, sampled)` product** rather than the
rows, so a sixth case cannot hide the same way. 〔`log-gcr` is not a fifth `method`: `InferencePlan`
builds blocks only from `Classification`, whose method is one of `gcr` / `gcr+snis` / `gcr+mh` /
`nuts`; `log-gcr` comes from `factor_partition`, which does not feed `InferencePlan`.〕

**Why (d) is refused rather than admitted.** `gcr+snis` means the covariance moves with the
block, so the GLS fixed point is only a proposal and the exact answer needs a self-normalised
importance reweighting (`dispatch/classify.py:571`). Its residual factor is the SNIS
*normaliser* — a genuine residual evidence, and **not a nested-sampling problem**. It has a
different estimator, a different failure mode (heavy-tailed weights, `khat`) and no oracle in
this repository. Admitting it under R5's gate would put a number behind a gate that never
tested it, which is §11.4's risk exactly. R5 refuses it by name and records what the route
would be, the treatment R4 gave the chain and campaign families.

**This is not a small class [planning].** Compiling every shipped fixture:

〔**Execution write-back, red line 11 (Task 1).** This table is a census over the **49 graphs
whose fixtures take no constructor arguments.** Task 1 declared arguments for the other five and
re-ran it over all **54**: four of them (`cancelling_sum`, `many_observations`, `roundoff_stress`,
`wide_plate`) are class (a) and one (`sigma_functional_block`) is class (d), so the complete
census is **19 / 7 / 13 / 10 / 5**. Classes (b) and (c) — R5's whole subject — are unchanged. The
table below is kept at its original denominator with the correction stated, because the plan's
own §0.15 note says a count whose denominator is unstated is one the next reader re-derives
differently, and silently restating it here would leave two numbers in the batch with no
explanation of the gap. `docs/probes/probe_34_residual_seams.py` §2 prints both and checks
`49 + 5 == 54`.〕

| class | count (of 49) | fixtures |
|---|---:|---|
| (a) whole-graph exact | 15 (**19** of 54) | R4 answers these |
| **(b) exact + residual, `gcr`** | **6** | `diamond_ancestor`, `indirect_ancestor`, `shared_ancestor`, `three_latent_chain`, `overflowing_outside_latent`, `improper_outside_prior` |
| (b′) exact + residual, `gcr+mh` | 1 | `mixed_radiometer` — **refused** by (d) |

> **One label per condition, because this document confused itself once already.** "class (b)"
> means **the six `gcr` graphs** everywhere below. Where all seven exact+residual graphs are meant,
> the plan writes **"(b) ∪ (b′)"** — which is what §0.15's premise table counts.
| **(c) all-residual** | **13** | incl. `student_t_likelihood`, `non_gaussian_observed_node`, `cubic_tail`, `bilinear_pair` |
| (d) `gcr+snis`, no sampled block | 9 (**10** of 54) | **refused** |
| compile refuses today | 5 | four deliberately malformed; `plated_student_t_latent` is **well formed** and simply outside the exact path (`NotGaussian`) |

**And the price of (d)'s refusal is measured [planning].** Calling `marginal_log_density`
directly on `mixed_radiometer` — a `gcr+mh` graph — returns a log evidence **1.4e5 nats** from
the uncollapsed truth. It is not a defect: `execute.py:775` already refuses this case, and its
message predicts the number ("would freeze that sigma at the prior centre and return a
plausible but wrong marginal"). It is the size of what the guard is worth, and R5 carries the
same condition into the collapse-then-sample path rather than re-deriving it.

### 0.4 Every residual integral this package ships is one- or two-dimensional, so the top-tier oracle is reachable for all of them

**Ruling.** The residual evidence is graded against **deterministic trapezoid quadrature of
`log_joint` over ALL latents — the UNCOLLAPSED integral** — on every admitted fixture. A backend
number that agrees only with another backend number grades nothing.

**Which side is the oracle has to be said, because getting it backwards is how R4's square-root
route fooled its own test.** Quadrature of the *collapsed* density is **not** an independent
oracle for the collapse: it calls `collapse_graph` and `marginal_log_density`, which are the
production path, and §9.1 forbids proving one entry point with another entry into the same
kernel. R4's close-out prices that mistake exactly — scaling `dense_operator`'s return by 1.03
left the cross-route test *completely blind*. The uncollapsed quadrature shares only `log_joint`
and `evaluate`, which are the model's own definition rather than the elimination algorithm, and
`docs/evidence.md` already names it *"the genuinely third route"*.

**Two oracles, named, and the plan uses the names everywhere rather than the word "quadrature":**

| name | what it integrates | §9.1 tier | may grade |
|---|---|---|---|
| **`oracle_joint`** | `log_joint` over **all** latents (residual + exact) | **1** — independent | the elimination, and end-to-end |
| **`oracle_collapsed`** | `p(τ)·exp(marginal_log_density(τ))` over the residual only | **5** — self-consistency w.r.t. the elimination | the **sampler only**, never the elimination |

**So three gradings, kept apart rather than conflated into one number:**

| what is graded | graded against | catches |
|---|---|---|
| `collapse_graph` | `oracle_joint` | a wrong exact elimination |
| the backend, on the collapsed problem | `oracle_collapsed` | a wrong residual integral |
| **end to end** | **`oracle_joint`** | either, and their composition |

G2 and Task 5.2's correctness column require **`oracle_joint`**. The first two gradings exist so
that a failure of the third says **which half** moved.

**And the dimension census above counts the wrong dimension for `oracle_joint`.** 1–2 is the
**residual** dimension; `oracle_joint` integrates residual **plus** exact. Today's worst is
`three_latent_chain` — residual `('tau','x')` plus exact `('y',)` — so **3**. Task 3(iii) asks for
a residual of dimension ≥ 4, and with any exact block `oracle_joint` for it is **≥ 5-D**, where
trapezoid is not viable. **Task 3.2's stop-rule measures the bound on `oracle_joint`'s total
latent dimension, not on the residual's**, or it writes an overstated gradeable domain into the
module spec.

**Why [planning].** Measured over all **20** shipped graphs with a sampled block — class (b) ∪ (b′)
is 7 and class (c) is 13 — counting scalar parameters from the **declared shape** rather than from
a prior draw:

| residual dimension | count |
|---:|---:|
| 1 | **15** |
| 2 | **5** |

〔An earlier count here said 19/14/5. It sized each latent from a **prior draw**, so
`improper_outside_prior` — whose prior cannot be sampled — fell out of the histogram without
appearing anywhere as a gap. Sizing from the declared shape gives 20/15/5. Fourth instance in this
plan of a check that could not distinguish "absent" from "did not run", and the second in a census
this document wrote itself.〕

Maximum 2. Trapezoid quadrature over one or two dimensions is cheap and convergent, and the
repository already uses the idiom (`tests/dispatch/test_acceptance.py:150` nests two
`np.trapezoid` calls). So **every admitted fixture has an analytic-grade oracle before any
backend is chosen.**

**The honest consequence, stated rather than hidden: on this fixture set a nested sampler is
strictly worse than the oracle grading it.** The fixtures are the *instrument*, not the use
case. The use case is the dimension and the geometry the fixture set does not contain, which is
why Task 6 builds the multimodal fixture §8 R5 requires, with a constructed closed-form `log Z`
rather than a sampled one.

**And the central completion gate is already reachable [planning].** §8 R5 requires that
"exact collapse 与未 collapse 的小问题对照一致". Measured on all four `gcr` class-(b) fixtures with
one sampled latent, collapsed against a full uncollapsed quadrature over both latents:

| fixture | gap `oracle_collapsed` − `oracle_joint` | span used |
|---|---:|---|
| `diamond_ancestor` | `0.0` | τ ∈ (−4, 8), x ∈ (−6, 6) |
| `indirect_ancestor` | `-8.9e-16` | τ ∈ (−4, 8), x ∈ (−6, 6) |
| `shared_ancestor` | `-1.8e-15` | τ ∈ **(0.15, 4.5)**, x ∈ (−1.5, 3.5) — **excludes τ ≤ 0, 3.167e-05 of the prior mass** |
| `overflowing_outside_latent` | `0.0` | τ ∈ (−60, 60), x ∈ (−8, 8) |

**The GAP is the quantity; the values are not.** A twelve-digit trapezoid sum is a
platform-dependent reduction and red line 9 forbids pinning one. 〔The values measured on
macOS/Accelerate were `-5.965611563400`, `-6.511601603549`, `-8.964352931180`,
`-17.470672900784`; they are recorded here as a comment for whoever re-derives the spans, not as
expected values.〕

One or two ULP, and the two sides converge in lockstep under refinement — the increments agree
to every printed digit, which is what a correct elimination looks like and what a near-miss
does not. So R5's central gate has a working oracle before any backend is chosen.

**A known hazard region for `oracle_joint`, recorded before it is walked into.** The exact
linear-Gaussian path carries **two unbounded error laws with different origins**, and neither
dominates: measured over a 7x7 sweep against an exact `Fraction` oracle, the shipped route wins
15 cells, a recentred one wins 21, and 13 tie — at `m = s = 2^60` the shipped route is 1.4e-14
and the recentred one **3.7e+05**, while at `m = s = 2^-60` the shipped route is **5.3 nats**
wrong and the recentred one 8.9e-15. That is a **dispatcher** problem in
`boundary-validation.md`'s sense, not a replacement, and R5 does not solve it.

**`oracle_collapsed` RECORDS the exact block's `|m| / s` beside its value, the way it already
records the excluded prior mass. It gates nothing, and it does not abstain.** A number never
travels without its domain; that is the whole of the remedy.

〔**Execution write-back, red line 11 (Wave B). The remedy this paragraph first gave was wrong
three ways, and its root error is the one the same commit was written to correct.**

It said: *"Task 2's oracle declares the `|m|/s` range it evaluates over and stays inside the
covered region, or it abstains."*

1. **`|m|/s` cannot separate the cells this section names.** Both corners quoted above are
   `m = s` — so `|m|/s = 1.0` in each, and one is 1.4e-14 while the other is 5.3 nats. A ceiling
   on that ratio admits both or refuses both. The prescription could not refuse the very cells it
   was written about.
2. **It would suppress a detection.** `oracle_joint` does not share the exact route's error law,
   it *detects* it: `log_joint` evaluates each node's own `log_prob` and never reaches
   `nuisance_prior`, so there is no `m/s` entry to cancel in a QR. Measured on `w ~ N(m, s)`,
   `mu = w X`, `d ~ N(mu, 0.5)`, n=4, against the closed form — at `|m|/s = 4e15` the shipped
   route is **0.28 nats** out and `oracle_joint` matches to the last bit. Abstaining there would
   have hidden exactly that.
3. **A ceiling is a third number**, and red line 8 pre-authorises only D111 and D112. Writing it
   as prescribed would itself have had to stop the work.

**The root error is the one `b44e073` was correcting, made inside the correction.** That commit
replaced an invented `s**2` mechanism with a measured law, `eps · |m| / s` — measured with `m`
fixed and `s` swept. This section then used that law's *parameter* as a *region coordinate*, in a
region where a different law operates. An error law's parameter is not a coordinate for the
region it was measured in, and the slice it was measured on is part of the measurement.

Measured over every Wave B fixture: the exact block's prior mean is exactly **0** in all four
class-(b) cases, and 0.5625 for the mixture fixture's `b`. So the hazard region is entered by
nothing, and the tier-1 comparison grades the route at the ratios actually used.

One further limit, narrowed after this line first overstated it: **the Cholesky-based
`gaussian_log_evidence` helper in `tests/exact/residual_models.py` raises
`LinAlgError: Matrix is not positive definite` at `s = 2^60`**, because float64 `slogdet` reports
`sign = 0` on that covariance. That is a fact about **one test helper**, not about float64. An
earlier version of this sentence said "that corner has no usable float64 reference at all", which
generalises one implementation to a class of them — the same over-wide move this section has now
made twice. Whether another float64 formulation survives there is **untested**; what is known is
that this one does not, and that an exact `Fraction` oracle does.〕

**The oracle carries its own convergence certificate, and this is a stop-rule.** A gap between
two quadratures where one has not converged is indistinguishable from a defect in the other,
and this plan walked into it: **the first grid run on `shared_ancestor` reported a gap of
6.55 nats.** The collapsed side was stable to twelve digits; the uncollapsed side had moved
1.4 nats between refinements and was simply not converged, because the fixture declares
`x ~ N(0, |tau|)` and the span included `tau ≈ 0`, where the conditional width goes to zero.
Widening the span off the singularity and refining took the same comparison to `-1.8e-15`.
A reader given only the first table would have opened a defect against `collapse_graph`.
**Every quadrature oracle in R5 refines until its value stops moving, prints the refinement
history beside the value, and ABSTAINS rather than reporting a gap it cannot certify.** R4's
`audit_prior` already carries this idea as D109; R5's oracle reuses the concept and derives its
own tolerance.
### 0.5 §1.5's six conditions are R5's decision instrument, and each is a countable gate

**Ruling.** §1.5 states six conditions that must hold **together** before bayesmith gives up an
in-house implementation to an upstream one. The backend bake-off is scored against these six
and nothing else. Each is restated below as something a run can answer, with the artefact that
answers it. **A candidate that fails any one of the six does not become the production adapter,
whatever it scores on the other five.**

| # | §1.5 (原文) | countable form | artefact that answers it |
|---|---|---|---|
| 1 | 对目标 problem family 足够通用，而不是只覆盖 demo | runs every admitted fixture in classes (b) and (c) — 20 today — plus the multimodal and the heavy-tailed fixture, without per-fixture special-casing | the bake-off table: one row per fixture, PASS/FAIL/CRASH |
| 2 | 与 JAX、JIT、PyTree、x64、目标 device 和随机数语义兼容，或有可测且可接受的边界成本 | runs under `jax_enable_x64(True)`; accepts a JAX callable; takes a `jax.random.key`; the same key gives the same `log Z` bitwise on a repeat | a bitwise-reproducibility assertion per backend |
| 3 | 在 correctness、compile time、runtime 和 memory 上通过代表性 benchmark | `\|log Z − quadrature\|` inside a **declared, derived** band on every fixture; compile seconds, wall seconds and peak memory recorded per fixture | the bake-off table + `TimingRecord.compile_seconds` |
| 4 | 维护活跃、版本和失败行为可追踪，能暴露 termination 与必要 diagnostics | last release date, release cadence and open-issue count recorded with the date read; every termination signal maps to a `TerminationReason` member; `log Z` uncertainty is a real estimate, not a constant | Task 2's maintenance record + the totality test of §0.1 |
| 5 | adapter 足够薄，不需要复制一套上游内部状态机，也不把 backend 对象变成核心 API | adapter is one module; no backend type appears in any artifact field; measured as a line count and an import-direction assertion | `tests/test_layering.py` + a public-API assertion |
| 6 | optional dependency 缺失或升级时能明确 Refuse，并有 contract test 与独立 oracle | with the package uninstalled, an `EvidenceTask` returns a capability `Refusal` and the core suite is green; the independent oracle is §0.4's quadrature | Task 3's built-and-run absence test |

**§7.3's nine questions are answered in the same table**, not separately: 1–2 and 7 by
`CompiledEvidenceProblem`'s contract, 3 by gate 6, 4 by gate 3's oracle, 5 by gate 5, 6 by a
pinned-version drift test, 8 by §0.6's second-backend rule, 9 by the six above.

**No winner is presumed.** BlackJAX nested sampling and JAXNS enter on identical terms. If
neither passes all six, **the plan's answer is "no candidate passed", recorded with the table
that says so**, and R5 ships the compiler, the eligibility, the oracle and the refusal without
a production adapter. That outcome is a completed R5, not a failed one: §8 R5's first
completion gate asks for *"backend 决策有可复现 benchmark、oracle 结果和明确适用域，而不是预设
偏好"* — a reproducible decision, not a backend. **Task 5 does not start until Task 4 has
written the verdict, in either direction.**

### 0.6 A second backend is kept only against a written rule, and the default is to drop it

**Ruling.** After the bake-off, the second-placed candidate is **removed** unless it satisfies
one of exactly two conditions, each of which must be shown with a run:

1. **A different applicable domain** — there is at least one admitted fixture the winner
   refuses or gets wrong and it does not; or
2. **Independent cross-check value** — the two disagree in a way that detects a fault the
   oracle does not, demonstrated by a mutation that one backend catches and the other does not.

"Both work" is not a reason. §11.9 names backend bloat as a risk and §7.3 ends
*"不以 backend 数量作为成熟度指标"*. If the second is kept, the reason goes in
`docs/ownership.md`, which is the `decision-home` for this class of decision.

### 0.7 "Same budget" means equal likelihood evaluations, and the obvious way to count them returns 1

**Ruling.** The shared currency is the **number of log-likelihood evaluations**. It is obtained
by **arithmetic from the sampler's declared parameters** — which bayesmith controls for a backend
whose loop it writes, and reads for a backend that reports it — and that arithmetic is **audited
once against a ground-truth count** on a small problem before it is trusted at scale. Each
backend's own knobs are tuned to hit the same evaluation count on a calibration fixture and then
**frozen across the whole table**.

**A wrapping Python counter does not work, and it fails silently. [planning]** The obvious
implementation — wrap `log_likelihood` in a closure that increments a dict — counts **traces**,
not evaluations. Measured:

| what was run | true evaluations | Python counter reports |
|---|---:|---:|
| 1000 separate `jax.jit` calls | 1000 | **1** |
| one `jit(vmap(f))` over 1000 points | 1000 | **1** |
| one `jit` of a `lax.scan` over 1000 points | 1000 | **1** |

It returns a small, plausible, wrong number with no error — the failure family this repository is
built around, and it would have made the bake-off's central quantity fiction. **The audit
therefore uses `jax.experimental.io_callback` or an unjitted run on a deliberately tiny problem,
and its only job is to prove the arithmetic formula right**; the formula is then what runs at
scale.

Recorded beside the count, never substituted for it: wall-clock seconds, compile seconds, peak
device memory, iterations, and each backend's native budget setting. A comparison at equal
wall-clock would compare two JIT compilers; a comparison at equal live points would compare
nothing, because the two backends do not mean the same thing by a live point.

**Two stop-rules, both run at the top of Task 5.**

1. **If a backend's evaluation count cannot be established either by bayesmith owning the loop or
   by an audited formula — if the only source is a number the backend reports and the audit does
   not confirm — stop and rule.** A bake-off whose two sides spent unverified and possibly
   different amounts is not a bake-off, and its number would be read as a correctness comparison
   for as long as the page survives.
2. **If a backend cannot be driven to within ±10% of the target evaluation count** on the
   calibration fixture, stop and rule before running the table.

### 0.8 Repeated-run stability is the gate's business, and it is declared before it is measured

**Ruling.** `EvidenceTask.repeat_count` — read and refused by R4 — is honoured. `n ≥ 3`
independent runs differing **only** in seed; the seed set is part of the fixture (§9.3). The
consistency report compares the spread of the `n` `log Z` values against the backends' own
reported `standard_error`, and the gate FAILs when the two disagree by more than a declared
factor.

**The statistic, the tolerance and the acceptable false-positive rate are declared in the plan
before the first run** (§9.3), and the number becomes a registered gate (§0.10). The direction
that matters is the asymmetric one: **a spread much LARGER than the reported error bar is the
failure §11.4 exists to catch** — an estimator confidently reporting a precision it does not
have. A spread much smaller is recorded and does not FAIL, because an over-conservative error
bar misleads no one about the evidence.

**Stop-rule.** If a backend's reported `standard_error` turns out to be a placeholder rather
than an estimate — a constant, or a value that does not move with the budget — **stop and rule
before the bake-off scores it**. Feeding a placeholder into a stability gate produces a PASS
that means nothing, and §1.5 condition 4 requires the diagnostic be exposed for real.
### 0.9 The optional dependency is absent by default, and the absence path is built and run

**Ruling.** The backend is an **optional extra**. `pip install bayesmith` does not pull it. With
it absent, an `EvidenceTask` over an admitted class-(b)/(c) graph returns a capability
`Refusal` — §4.4's fourth example, *"optional backend 未安装"* — and every other task on the
same graph is unaffected, asserted in the same test. The core suite is green with the extra
absent, and that is the **default** state of CI's fast layer.

**Why this needs saying [planning].** Neither candidate is installed in this checkout today
(`blackjax` ABSENT, `jaxns` ABSENT against `jax 0.11.1`), so "absent" is not a hypothetical
state to be simulated — it is the state the repository is in, and the state most consumers will
be in. A refusal path that is only ever exercised by monkeypatching is a refusal path that has
never run.

**And there is no precedent to copy: `pyproject.toml` has no `[project.optional-dependencies]`
table at all.** R5's would be the first extra this package has ever declared, which means three
things have no existing shape — the extra itself, a CI job that runs the suite **without** it,
and a wheel test that installs it. `publish.yml` builds and tests the wheel; **a wheel that is
only ever tested with the extra present has never tested the refusal**, and a wheel only ever
tested without it has never tested the adapter. **Task 10 builds both.**

〔Execution write-back, red line 11 (Wave D). This said *"Task 4 builds both"*, which over-scoped
Task 4 into CI work its own file list does not mention and which it did not do — no workflow file
was touched. Moved to Task 10, where the built-wheel gate already lives.

**The cost of not having had it is measured, not hypothetical.** Wave D's review found that
installing `blackjax` reddens the suite, and a job that installs the extra and runs the suite is
exactly what would have caught that before a tag rather than in a review. §0.9's own argument —
a wheel only ever tested one way has never tested the other — turns out to apply to the *suite*
as well as the wheel, and the failure it predicted arrived by the route it named.〕

〔**Measured by Wave D, and recorded here because it lives nowhere else** (red line 18). The wheel
was built and its metadata read: it declares `Provides-Extra: blackjax` and `Provides-Extra:
jaxns` with the matching `Requires-Dist` markers, and the four hard dependencies are unchanged.
So `pip install "bayesmith[blackjax]"` — **the exact string the capability refusal hands a
caller** — is satisfiable by the built artefact and not merely by the source tree.

**Nothing asserts that today.** A refusal whose remedy is an install command is only as good as
the command, and the command is a claim about a wheel that no test in this repository builds.
Wave D declined to add a wheel build to the fast layer, which is right — it belongs in
`publish.yml`, and that is Task 10's. **Task 10 owes the assertion, not a repeat of the
measurement.**〕

> **Note the ordering trap.** `numpyro` is a hard dependency here, not an extra — the pyproject
> comment says it is "the last row of the dispatch table, not an optional extra". So the two
> existing `pytest.importorskip` sites (both under `tests/crosscheck/`, guarding the sibling
> checkout) are the only degradation precedent, and their convention is the **opposite** of what
> R5 needs: they SKIP. R5's absence path must RUN and assert a `Refusal`.

**The absence test is built as a bypass, per red line 1:** the reviewer writes a case that
reaches the sampler with the dependency missing and shows what happens, rather than asserting
that a guard exists.

### 0.10 A stochastic threshold has no precedent in the registry, and the way to build one is already in the repository

**Ruling.** Every threshold R5 introduces is registered in `tests/numerical_gates/registry.py`
with a boundary grid and one fast-layer cell. **The stochastic quantity is never sampled inside
the gate.** It is computed upstream and passed in, so the gate itself stays a pure scalar
comparison that can be mutated without re-running a hundred seeds.

**Why, and why this is a ruling rather than a preference. [planning]** There is no seeded gate in
the tree: across all of `tests/numerical_gates/`, `default_rng` / `np.random` / `PRNGKey` occurs
**exactly once**, in a docstring describing a probe run offline. All 113 entries grade
deterministic linear algebra, and `ThresholdProvenance` has five members — `derived`, `borrowed`,
`magic`, `exact_or_domain`, `api_contract` — none of which is a statistical category. (Measured
distribution: derived 46, exact_or_domain 30, api_contract 21, borrowed 16, **magic 0**.)

**The precedent exists one layer up, in SBC, and R5 copies it.** `evaluation/sbc.py:209`:

```python
def ranks_are_uniform(p_value: float, level: float) -> bool:
```

with a docstring saying it was given its own name *"so that it can be exercised and mutated
directly rather than only through"* a full replicate run. Seven registry entries already grade a
stochastic-flavoured scalar this way — a replicate count, a p-value, a timing-noise fraction —
and none of them samples. So R5's stability threshold is registered as
`spread_within_reported_error(spread, reported, factor)`, a scalar comparison with a
**`derived`** provenance whose form comes from the estimator's variance argument, and the
seed-driven machinery lives in the test that calls it.

**Stop-rule, run at the top of Task 9.** If a threshold R5 needs cannot be expressed as a pure
scalar comparison over an upstream-computed quantity — if the gate itself would have to sample —
**stop and rule**, choosing between a new `ThresholdProvenance` member (a registry schema change)
and demoting it to §9.3's *"定期 calibration job"*. Do not invent a sixth provenance in passing.

**And registering a gate is not one edit — it is seven, plus a count pin. [planning]** A
`GateEntry` has 22 fields and none is authored directly: entries are built by `_entry(seed)` from
a `_Seed` plus six side tables keyed by `gate_id`. Then the count pins fire. **There are eight,
not the four the R4 close-out names, and that page's description is itself imprecise — it says
`registry.py` pins the two-sided count twice, but `registry.py` contains no 97 at all; its two
literals are 113, the TOTAL:**

| # | pin | value | layer |
|---|---|---|---|
| 1 | `registry.py:7219` — **raises at IMPORT**, so it is a collection error everywhere | 113 | fast |
| 2 | `test_registry.py:1820-1822` — distinct admitted / refused / oracle strings | 113 ×3 | fast |
| 3 | `test_boundary_provider_contract.py:50` | 97 | fast |
| 4 | `test_boundary_cases.py:69` | 97 | **full only** |
| 5 | `test_boundary_cases.py:331` — witness pin, written `2 * 97` so it moves with #4 | 2×97 | **full only** |
| 6 | `test_boundary_cases.py:78` — atom cases | 212 | **full only** |

**Pins 4-6 are not "mostly nightly" — they are structurally invisible to the fast layer.**
Measured collection under `-m "not full"`: `test_boundary_cases.py` **0 tests**,
`test_boundary_provider_contract.py` 108, `test_registry.py` 126, `test_boundary_layering.py` 99.
The whole module carries `pytestmark = pytest.mark.full`, so no amount of fast-layer green says
anything about three of the eight pins.
| 7-8 | `test_registry.py:91-92` — raw AST census | RAISE 195, COMPARE 380 | fast |

**Pins 7 and 8 are the landmine, and they are not about gates at all.** They move as soon as R5
adds a gate-bearing module to `SOURCE_PATHS` — **or merely adds a comparison or a `raise` to an
already-scanned module.** Every task that touches `src/` can move them. And pins 4-6 are
full-layer only, so a green fast layer says nothing about them: R4 shipped a batch that left the
nightly red for exactly this reason. **Every task's green step re-runs
`pytest -m "not full" tests/numerical_gates/` (390 today, exit 0), and Task 10 runs the full
layer before the close-out is written.**

**Red line 8 applies unchanged and is the strictest line in this plan**: a threshold appearing
during execution means stop, register, then write code. R5 pre-authorises **D111 and D112** only,
and reserves **D113-D115 with expected consumption 0**. Note that **the registry has no
decision-id field** — D-numbers appear only inside `threshold=` prose — so the highest registered
D cannot be grepped from the registry (its highest prose mention is D106). D110 is the highest
anywhere in the repository; D111 is the next free id.

### 0.11 x64 is a premise, and the scale check is a separate one

**Ruling.** `evidence_requires_x64` — R4's premise, raised where the assembly is built — extends
to the residual route unchanged, and is asserted **before** the backend is called rather than
after it returns. The consequence is asserted, not the spelling: the same problem is run at
float32 and float64 and the gap in `log Z` is measured, as R4 did (3.4e-07 against 8.9e-16).

The **scale** check is separate and new: a nested sampler's likelihood is exponentiated
internally, so a `log_likelihood` whose dynamic range exceeds what the backend's own
accumulator carries is a silent failure rather than an overflow. **[planning]** The shipped
fixtures already contain the stress: **at prior-draw seed 0**, `element_contrast_sigma_plate`
has `log L = −1.009e10` and `high_snr_curvature` `= −8.785e12`. 〔The seed is load-bearing and
an earlier draft omitted it: at seed 1 `element_contrast_sigma_plate` gives **−22.76**, nine
orders of magnitude away. A dynamic-range figure without its draw is not a measurement, so the
check sweeps a declared seed set rather than quoting one.〕 The check reports the
range and refuses where it exceeds the declared domain; it does not rescale the problem, because
a rescaled likelihood is a different `Z`.

### 0.12 What R5 inherits from R4 as a premise, not as a re-derivation

R4's evidence premises apply to the residual route unchanged and are evaluated **before** any
backend runs. **The order below is the code's, read out of `_evidence_precompile_refusal`
(`dispatch/task.py:1023-1137`) — an earlier draft of this section invented a different one, and
Task 7 was told to assert it:**

1. **`evidence_requires_x64`** — first, not third. Decided by outcome (`jnp.result_type(float)`),
   which is why §0.16's jaxns hazard reaches it.
2. **`task_options_recognised`** — `repeat_count` / `reconstruct_posterior`. Absent from the
   earlier draft entirely, and §0.8 and Tasks 6/9 all depend on lifting it (see Task 7.6).
3. **One loop over `audit_graph_priors(graph)` in `graph.latents` order**, out of which
   `evidence_prior_normalised`, `evidence_prior_proper` and `evidence_prior_undeclared` emerge
   **interleaved by latent, not ordered by premise.** There is no order in which one of the three
   precedes another; whichever latent comes first answers first.
4. Then, and only then, the structure gate.

〔**Execution write-back, red line 11 (Task 7.5/7.6): Task 7 MOVED row 2, and the new order is
what is asserted.** The chain is now

1. `evidence_requires_x64` — unchanged, still first, still decided by outcome;
2. the `audit_graph_priors` loop — unchanged, still interleaved by latent;
3. the **structure** gate — `evidence_residual_method_unsupported` for the two prediction-dependent
   methods, `evidence_residual_integral_required` for a graph with no latent at all;
4. `task_options_recognised` — **moved here from position 2**;
5. `evidence_conditional_prior_proper` — new, residual route only;
6. `capability_unavailable_r1` — new here, the missing residual backend.

**Why row 2 had to move.** Task 7.6 requires the option arm to refuse both options on an *exact*
evidence and admit them on a *residual* one. Which route a task takes is a property of the PLAN,
and at position 2 there is no plan — `_evidence_precompile_refusal` runs before `compile_plan`.
Narrowing it in place is not possible, so it moved to just after the gate that decides the route.

**What that changes, stated rather than left to be discovered:** a graph with BOTH an improper
prior and an unread option now answers `evidence_prior_proper` where it used to answer
`task_options_recognised`. No shipped fixture and no test was in that intersection — both option
tests use a class-(a) graph — but the ordering is observable and it is now asserted directly in
`TestThePremiseChainsOrder` rather than inferred.

**Rows 5 and 6 are in that order deliberately**: a model fault is named before a release limit. A
caller told "come back when the extra is installed" would install it and get the same wrong
integral. It also keeps row 5 reachable — behind the capability refusal it would be dead code on
every input the package can build, which is red line 13's fault (a).〕

`evidence_comparability` is **not** a compile premise at all — it is a report kind in
`evaluation/evidence.py`. R5 leaves it unchanged and asserts separately that the fingerprint rule
does not silently begin admitting a comparison between a sampled number and an exact one
**without** the sampled one's error bar being carried into the Bayes factor.

**[planning]** A nested sampler needs a prior it can *sample*, which is strictly stronger than one
that integrates to one: `improper_outside_prior` raises `NotImplementedError` at prior sampling.
That is a fact about the sampler's requirement, not about the premise order.
### 0.13 The baseline is `2d873f1`, it moved five times while this plan was being written, and it will move again

**Ruling.** R5 begins at `2d873f1`, where the fast layer is **3264 passed, 0 failures, 0 errors,
0 skipped**, 306.0s, exit 0 — with `git rev-parse HEAD` **and** `git status --porcelain` recorded
before and after the run and both unchanged. The first executing session re-measures rather than
quoting this, and records the same four artefacts beside anything it does quote.

**Why.** This is not procedural caution; it happened here. A concurrent session pushed
`9af68c3` (20:01) and `895e181` (20:21) while this plan was being drafted, and **a fast-layer run
started for this plan at 20:19 and finished at 20:24 straddled the second one.** It reported
`3257 passed, exit 0`, and that number was about a tree that changed underneath it. It has been
discarded and re-measured; the baseline count below is the re-measurement.

**The re-measurement returned the same number, and that is the part worth keeping.** Measured at
`895e181` with HEAD recorded before and after and diffed: **3257 passed**, 294.6s, exit 0. The
void run also said 3257. `CLAUDE.md` already has the sentence for
this — *"The number happened to be right, which is the bad outcome, because nothing in the run
could have said otherwise"* — written there about a commit message asserting a count from a log
that had none. **A number quoted from a run whose HEAD was not pinned on both sides is not a
measurement, however right it turns out to be**, and the same discipline that makes
`git checkout -- src/` the correct restore in mutation testing — HEAD must be the thing you want
back — applies to reading a count off a suite.

That session reports one more adversarial review still running in its own worktree, which may
produce another commit; it pushes, the reviewer does not.

**It happened a third time before this plan was finished, and what caught it is worth copying.**
A "final confirmation" run of the fast layer was started while that session had **uncommitted**
test additions sitting in the shared tree (`tests/marginal/test_campaign_per_epoch_prior.py`,
+252 lines). The only signal was `test_readme_count.py` going red —
*"README.md says 5895 tests; a collection finds 5902"* — a test with **nothing to do with the
change being verified**. Had that assertion been written as a lower bound rather than as an
equality, the run would have come back green and its number would have gone into this document.

**An assertion pinned by equality is the reason this was visible at all**, and it is the
counter-example to the rule that count pins are a maintenance burden: the burden is what makes
them able to speak. §0.10 lists eight of them; this is why they are worth their cost.

**And the same run leaked a number in the other direction, which is the sharper half.** That
voided run also reported a fast-layer total, and it was read — in a message to the other session —
as "the count did not move". It had moved: the run's collection happened before the other
session's test file was written, and the true fast layer at `2d873f1` is **3264**, not 3257 — seven tests the other session added,
all of them in the fast layer. The
asymmetry is the lesson: **`test_readme_count.py` spoke because something pins it by equality;
the fast-layer total is pinned by nothing, so it did not speak — it simply propagated.** A number
that no assertion holds is a number that travels from a voided run into a document without
resistance, and the only defence is to refuse to quote a run that was not HEAD-pinned on both
sides, however plausible its number looks. **This plan quotes only runs with `sha`, `sha_after`, a `tree_before`/`tree_after` pair, and a
`git merge-base --is-ancestor` check that the recorded SHA is STILL ON THE BRANCH.**

〔Execution write-back, Wave A. The fifth artefact was added after a run reported `exit 0` with
its SHA identical before and after — fully compliant on its face — pointing at a commit that a
later `--amend` had taken off `main`. The four-artefact set proves *nothing moved during the
run*; it cannot prove *the thing measured still exists*. Amend BEFORE the run, never after.

And the run's own tree is not passive: **do not edit the repository while a verification run is
in flight.** Three runs were voided this way in Wave A alone, each time for an edit that could
not have changed the outcome — which is not the point, because a run whose tree moved cannot say
that.〕

### 0.14 One item closed while this plan was written, with its authorisation outside the repository

**`marginal/campaign.py`'s per-epoch prior is repaired at `895e181`**, not open. `epoch_terms`
read one entry of each per-epoch latent's declared prior and broadcast it, so a campaign
declaring `tau = [0.5, 1, 2, 4]` integrated three of four epochs against a prior the model does
not declare — measured at `-3.086436` against a hand-written dense truth of `-5.838560`, 2.75
nats, finite and plausible and unwarned. The repair makes the prior a mapped argument rather than
a closed-over scalar, and is graded against an oracle sharing no linear algebra with it.

> **Owner: confirm this one, because its authorisation is not in the repository.**
> `_handoff-next.md` — gitignored — assigned this decision to you explicitly (*"修它要先决定
> 「异质的 per-epoch prior 对向量化 fold 意味着什么」… 归 owner 裁决"*), and two hours later the
> same session landed it, its commit message calling the earlier assessment "too conservative".
> That session reports it put three options to you and that you chose to land it. **This plan
> takes no position on that**: a conversation is not something the repository can be asked about.
> What the repository shows is a reserved ruling reversed, with the authorisation living nowhere
> a check can reach it — the same shape as the six stale copies `CLAUDE.md` opens with. The code
> and its independent oracle read as sound and R5 does not depend on the outcome either way. One
> sentence from you closes it.

**And it was then adversarially reviewed, which changed what this section can say about it.** The
review found **no numerical defect** — 38 cells against a dense oracle, worst 1.07e-14; the
homogeneous case bitwise unchanged over 362 hex-pinned values — and then put half the bug back and
ran the whole fast layer **green**:

| mutant | change | `tests/marginal` | verdict |
|---|---|---:|---|
| M8 | the declared **mean** collapses again (`loc.ravel()[0]`) | 549 passed, exit 0 | **SURVIVED** |
| M2 | `target = (size,)`, dropping the per-component width axis | 549 passed, exit 0 | **SURVIVED** |

M8 sits **4.221 nats** from the dense truth with the fast layer green. Both are now killed by
runs rather than by argument (M2: 553 passed / 3 failed; M8: 549 passed / 7 failed, against a
556-passing baseline), and the repair is stronger than when this section first described it.

> **The reason it happened is the sentence R5 needs, and it is not about campaigns.** A per-epoch
> declaration has three dimensions — mean, width, per-component width. The repair fixed the width
> and **its tests varied the width**. The other two are constant in every campaign fixture this
> package ships, so neither mutant had anywhere to die.
>
> **A repair's tests inherit the repairer's blind spot, because both come out of one mental model
> of the bug.** This is the same shape R4's review found ("every fixture had one latent, one
> observation, one scalar sigma"), and it is aimed directly at R5's **Task 7**, where the same
> hand writes the propriety restatement and the tests that grade it. Task 7's review prompt must
> name the dimensions the repair did **not** vary and require a mutant in each.

**A second open item, disclosed rather than repaired.** A per-epoch prior degrades silently:
relative error 1.7e-09 at width 1e-8, 6.9e-06 at 1e-12, 5.7e-02 at 1e-16, and **`-inf` with no
refusal at 1e-300**. The failure **pre-dates** the repair — a homogeneous all-`1e-300` campaign
was `-inf` on both commits — and what the repair changed is that it is now reachable from any
epoch slot rather than only the first.

〔**Execution write-back, red line 11: the mechanism this paragraph gave was invented, and the
correction changes what a remedy would have to be.** It said the far end is non-finite because
`s**2` underflows at 1e-300, "IEEE, not a BLAS choice". Measured by the session that wrote it:
**nothing on that path squares `s`** — `nuisance_prior` builds `1/s` and `m/s`, and the only
`**2` is on the noise sigma. And an underflow story cannot be non-monotonic in `s`, while this
is: `s = 1e-200` returns a finite `-3.840157`, `1e-307` is finite again, and only `1e-300` is
`-inf`.

The real control is the prior **MEAN**, with error law `eps · |m| / s`:

| width | `m = 0.4` | `m = 0` |
|---|---|---|
| 1e-16 | 5.7e-02 | 8.4e-16 |
| 1e-300 | `-inf` | **8.9e-15** |

At `m = 0` every width down to `1e-307` is exact to one ulp. **So a floor on `s` — the obvious
remedy, and the one this paragraph implied — cannot separate the good cells from the bad ones at
all.** The observation was real and the explanation was made up; citing the two as one thing is
how the wrong remedy got written down. Same lesson as this plan's `evidence_terms` miscitation,
one level deeper: an explanation is not a measurement.〕

**Setting a floor to refuse it would change released behaviour — owner's call, and R5 does not
touch it, and the floor it would have set is now known to be the wrong shape.**

**Still open, and R5 does not touch it:** the `gcr+snis` family, refused by §0.3(d). Nine shipped
fixtures sit there. Its residual factor is a self-normalised importance normaliser with no oracle
in this repository, and admitting it is a feature with its own gate, not a row in R5's table.
### 0.15 R5's headline case is blocked by the PRIOR AUDIT, not by the structure gate, and widening the structure gate alone would admit one fixture out of seven

**Ruling.** R5 widens **two** predicates, not one. `_evidence_structure_refusal` is the obvious
half. The half that actually gates class (b) is `audit_prior`'s propriety rule, and R5 restates
what propriety **means** for a residual evidence rather than relaxing the rule.

**Why. [planning]** Compiling an `EvidenceTask` over every shipped fixture and reading the
premise that actually fires:

| class | graphs | `evidence_prior_proper` | `evidence_residual_integral_required` |
|---|---:|---:|---:|
| (a) whole-exact | 15 | — | — (all **admitted**, 15/15) |
| **(b) exact + residual** | **7** | **6** | **1** |
| (c) all-residual | 13 | 1 | 12 |
| (d) `gcr+snis` | 9 | — | 9 |

> **Counting convention, because two independent censuses of this differed.** These are
> **graphs**, not fixture functions: `flagged_line` returns three objects, two of which are
> graphs, so it contributes 2 to class (a). Counting fixture functions gives 14; counting them
> while letting a tuple return fall into an `except: continue` gives 13. All three numbers
> describe the same set. **Task 7's census states its convention in the assertion**, because a
> count whose denominator is unstated is a count the next reader re-derives differently.

**Class (a) is 15/15 ADMITTED, and that framing matters.** R4's prior audit refuses **nothing**
R4 admits. This is an R5 widening problem, not an R4 defect — the version of the audit that
misjudged stock priors existed and was repaired at `1c92e70`, before R4 closed.

Named, because the pattern is the point:

| fixture | class | premise | finding |
|---|---|---|---|
| `diamond_ancestor` | (b) | `evidence_prior_proper` | `('x', 'unverifiable')` |
| `indirect_ancestor` | (b) | `evidence_prior_proper` | `('x', 'unverifiable')` |
| `shared_ancestor` | (b) | `evidence_prior_proper` | `('x', 'unverifiable')` |
| `three_latent_chain` | (b) | `evidence_prior_proper` | `('x', 'unverifiable')` |
| `mixed_radiometer` | (b) | `evidence_prior_proper` | `('w', 'unverifiable')` |
| `improper_outside_prior` | (b) | `evidence_prior_proper` | `('z', 'improper')` — **correct** |
| `overflowing_outside_latent` | (b) | `evidence_residual_integral_required` | `(('z',), 'gcr')` |
| `orphaned_child_latent` | (c) | `evidence_prior_proper` | `('v', 'unverifiable')` |

**The mechanism, and why it is not a bug.** `docs/evidence.md` states the rule: `audit_prior`
returns UNVERIFIABLE when *"the latent has parents — a hierarchical prior is not a fixed density;
`p(w)` exists only once `s` is integrated out"*. And **a latent with a latent parent is precisely
what makes a graph class (b)**: the ancestor rule ejects the parent to the sampled block, which is
what leaves an exact block and a residual beside it. Class (b) and "has a hierarchical prior" are
very nearly the same set. The audit runs before the structure gate, so it answers first.

The rule is **right for R4's class** — a whole-graph-exact evidence needs each declared prior to
be a fixed normalised density — and **wrong for R5's**, for a reason that is about the
mathematics rather than about the ordering:

> What a nested sampler integrates against is the **residual block's** prior. The exact block's
> prior is a *conditional* density, and R4's assembly already integrates it in closed form at
> each value of the residual parameters — that is what `marginal_log_density` is. Requiring
> `p(x)` to be a proper marginal in isolation asks the wrong question of `x`: nothing ever
> integrates against `p(x)`; things integrate against `p(x | τ)`.

**And `p(x | τ)` is NOT proper for every τ — an earlier draft of this section said it was, and
the fixture that refutes it is one this plan pins a number for.** `shared_ancestor` declares
`tau ~ N(2.0, 0.5)` and `x ~ N(0, |tau|)`. Measured:

| τ | scale | `log p(x=0 | τ)` | `log p(x=1 | τ)` |
|---|---:|---:|---:|
| 0.1 | 1.0e-01 | +1.3836 | −48.62 |
| 1e-3 | 1.0e-03 | +5.9888 | −5.0e+05 |
| 1e-8 | 1.0e-08 | +17.5017 | −5.0e+15 |
| **0.0** | 0 | **nan** | **nan** |

and `P(τ ≤ 0)` under the declared prior is **3.167e-05** — τ = 0 is interior to the support, not
outside it. The conditional is a Dirac there, and `marginal_log_density` is not evaluable.

**The true statement, and the ruling that follows from it.** `p(x | τ)` is proper for **almost
every** τ; its degeneracy set has Lebesgue measure zero, so `Z` is finite while the integrand is
undefined on a null set. That is enough for the mathematics and **not** enough for a quadrature,
which evaluates at points.

**So the declared range is a DOMAIN OF THE ORACLE, not a propriety requirement**, and it is
recorded as such:

* the residual block's prior is audited for propriety over its **full declared support** — this
  is what admits `shared_ancestor` and refuses `improper_outside_prior`;
* the **oracle** integrates over a recorded finite span, and **every pinned number carries its
  span beside it**;
* a quadrature whose span excludes part of the support reports **how much prior mass it excluded**
  — 3.167e-05 for `shared_ancestor` — and the gate ABSTAINS rather than PASSes when that mass
  exceeds the agreement band it is being asked to certify.

**This is not bookkeeping.** §3.4's eligibility list names *"是否存在会让 Bayes factor 无意义的
data-dependent prior 或隐式截断"* — an implicit truncation is a thing that makes a Bayes factor
meaningless. §0.4's earlier draft performed exactly that truncation and did not record it, which
is how a pinned twelve-digit number can be an integral over a support nobody declared.

〔**Execution write-back, red line 11 (Wave C). This ruling named the wrong block, and the
working half of its own reasoning is the half the conclusion dropped.** Measured over its four
named targets — which latent the audit actually refuses, and which block holds it:

| fixture | refused latent | block |
|---|---|---|
| `diamond_ancestor` | `x` | **EXACT** |
| `indirect_ancestor` | `x` | **EXACT** |
| `shared_ancestor` | `x` | **EXACT** |
| `three_latent_chain` | `x` | residual |

**Three of the four carry the offending latent in the EXACT block**, so a rule about "the residual
block's joint prior" addresses the wrong half for the majority of the cases it was written to
admit. The paragraph above gets it right — *"the exact block's prior is a conditional density, and
R4's assembly already integrates it in closed form at each value of the residual parameters"* —
and then the conclusion drops exactly that clause. The conclusion was right about the outcome and
wrong about the reason, which is the harder error to notice because the fixtures still pass.

**Two further corrections from the same wave.** The restatement is **unimplementable where the
audit runs**: the audit is pre-compile and has no partition to factorise along, and moving it to
where one exists turns `lying_block_member`'s Refusal into a raised `StructureError`. What
shipped is the **graph's** factorisation, not the residual block's. And **class (b) has two
shapes**: `overflowing_outside_latent` and `mixture_prior_residual` are root-only, so the
restatement has nothing to do with their admission at all.〕

**So the restatement, corrected:** propriety for the residual route is a property of the
**graph's** factorisation — each latent's conditional given its parents, whichever block holds it
— and the clause that does the work is that **an exact-block latent's prior is a conditional
density integrated in closed form at each residual value** — each residual latent's conditional density given its
residual-block parents, with the block's roots audited by R4's existing one-dimensional rule. A
latent whose parents lie inside the residual block is admitted through its conditional; a latent
whose prior is genuinely improper (`improper_outside_prior`'s `z`) is refused exactly as today.
`audit_prior`'s per-latent marginal rule is kept unchanged for class (a).

〔**Execution write-back, red line 11 (Task 7.2). "The residual block's" is the wrong half, and
it is also unavailable where the audit runs. What shipped is THE GRAPH's joint prior**, factorised
along the graph: each latent's conditional given its own parents, roots through R4's unchanged
one-dimensional rule, non-roots at the prior centre. Three measurements forced it.

1. **There is no block partition at that point.** The prior audit is a PRE-compile refusal
   (`_evidence_precompile_refusal`, called before `compile_plan`), so "the residual block" does not
   exist yet. And it cannot simply be moved: `lying_block_member`'s prior is unnormalised AND its
   graph does not compile, so after the move its `evidence_prior_normalised` **Refusal becomes a
   raised `StructureError`** — measured.
2. **The residual block is the wrong half.** Of the four fixtures §0.15 names as its target,
   **three carry the offending latent in the EXACT block** — `diamond_ancestor` and
   `indirect_ancestor` reach `x` through a deterministic node, `shared_ancestor` through `tau`.
   Only `three_latent_chain`'s `x` is residual. A residual-only audit leaves three of the four
   unexamined rather than admitted for a reason.
3. **The graph's own factorisation is strictly stronger and costs nothing**: `p(θ) = Π p(θᵢ |
   parentsᵢ)` needs no partition, covers both blocks, and subsumes §0.20. Measured over every
   shipped graph, the only verdicts it moves are the **seven** latents that have parents,
   spread over **six** graphs -- ``three_latent_chain`` carries two, and an earlier write-back
   here said "six latents" for both numbers
   (`diamond_ancestor`, `indirect_ancestor`, `mixed_radiometer`, `orphaned_child_latent`,
   `shared_ancestor`, `three_latent_chain`); the whole-graph-exact class has none, so R4's census
   is untouched, and that is asserted rather than argued.〕

**Stop-rule, run at the top of Task 7.** Re-measure this table first. **If widening the structure
gate alone admits fewer than all seven class-(b) fixtures — it admits exactly one today — stop
and rule on propriety before writing any adapter wiring.** A plan that widened only the structure
refusal would pass its own tests on class (c), ship, and silently never admit class (b) at all —
and class (b) *is* §8 R5's headline work item ("exact collapse 后再调用 nested sampling").

〔**Execution write-back, red line 11 (Task 7.1): the stop-rule FIRED, and the table re-measured
exactly.** Six of the seven (b) ∪ (b′) graphs answered `evidence_prior_proper` — five
`unverifiable`, one `improper` — and only `overflowing_outside_latent` reached the structure
premise. §0.15's ruling was therefore executed rather than re-decided.

**But the class has TWO shapes and this section's framing sees only one.** §0.15 says class (b)
and "has a hierarchical prior" are very nearly the same set. Measured after Wave B landed a fifth
class-(b) fixture, they are not: `overflowing_outside_latent` **and** `mixture_prior_residual`
carry only ROOT latents, so R4's rule answers for them unchanged and the restatement has nothing
to do with their admission — the structure gate alone is what was blocking them.
`mixture_prior_residual` is also the first **multimodal** graph the headline class admits. So
Task 7.2's target is four fixtures for the propriety half and the two above for the structure
half, and asserting only the four would leave "is the offending latent hierarchical" as a
dimension the family holds constant.〕

**The conditional-propriety argument has a caveat, and it changes what Task 7 must assert.**
The identity the restatement rests on is

```
Z = ∫ p(τ) [ ∫ p(x|τ) p(d|x,τ) dx ] dτ
```

— the inner integral is `marginal_log_density`, and it needs `p(x|τ)` proper **at every τ**, not
on average. What guards that today is not a typed refusal: `dispatch/collapse.py:224` wraps the
divergence check in `eqx.error_if(offset, ~pivots_constrain_block(pivots, column), ...)`, which
inside a traced sampler loop is a **runtime abort**. Three consequences:

* it is **per-evaluation**, which is the right semantics and is what makes the restatement sound;
* whether a prior that diverges only in some corner of τ-space is caught **depends on whether the
  sampler walked there** — a guard whose firing is sampler-dependent is not a boundary;
* it **aborts rather than returning a `Refusal`**, so R5's gate gets an exception where §4.4
  requires a statistical boundary to be a filable refusal.

〔**Execution write-back, red line 11 (Task 7.3). The third bullet is wrong, and the correction
makes the case for the check STRONGER rather than weaker: it does not abort at all.** Built as a
bypass and run — `x`'s conditional width infinite below a cut in τ, the cut placed inside
`compile`'s own probe grid — `marginal_log_density` returns **`-inf`**, silently, at every point
in the corner. The `eqx.error_if` never fires. The reason is in `pivots_constrain_block`: its
floor is **relative**, `sqrt(eps) · max(pivot)`, over the joint prior-and-data information, so a
block the DATA constrains passes the guard however improper its prior is. The guard is about an
unconstrained posterior direction, which is a different thing from an improper prior.

And `-inf` is the value §0.1 records as unrepresentable in `EvidenceResult.log_evidence`, so left
alone this surfaces as a validator `TypeError` far from its cause, or not at all. So the second
bullet is also understated: it is not that the firing is sampler-dependent, it is that there is
no firing.

**Scope correction, measured.** The conditional premise covers **residual** latents. An
exact-block latent whose conditional scale degenerates at a probe point is already refused by
`check_gaussian` during `compile_plan` — at the same points, by the same criterion — and
**symmetrically: the posterior task raises too**. That makes it a graph-level structural refusal
rather than an evidence boundary, so Task 7 does not restate it as one. Which side a fault lands
on is decided by whether the latent is collapsible: linear in the prediction it joins the exact
block and `check_gaussian` answers; quadratic it stays residual and the evidence premise
answers.〕

〔**Execution write-back, red line 11 (Wave C): no coordinate on τ separates a proper conditional
from a degenerate one.** At `τ = 0`, `shared_ancestor` is degenerate and `three_latent_chain` is
ordinary — the same τ, opposite verdicts. A declared *range* therefore cannot be the check, for
the same reason the `|m|/s` ceiling could not be: **a parameter that indexes one fixture's failure
is not a coordinate for the region.** That is now the second time this plan has made that move,
and the second time a session running it caught it. What shipped reads the **density** instead,
and what it cannot see is **recorded rather than gated** — the same resolution Wave B reached.〕

**So Task 7's stop-rule, corrected twice over: conditional propriety is verified by READING
THE DENSITY at the probe points, not by declaring a range — no coordinate on τ separates the
two verdicts — and what the check cannot see travels with the result rather than gating it**, and
a violation inside that range produces a `Refusal`, not an `eqx.error_if` abort. R4 never met
this because class (a) has no τ to vary.

〔**Execution write-back, red line 11 (Task 7.3): what the declared range can be indexed BY.**
The plan does not say, and the obvious answer does not work. **No coordinate on τ separates a
proper conditional from a degenerate one.** Measured at τ = 0: `shared_ancestor`'s `p(x|τ)` is
degenerate there while `three_latent_chain`'s and `mixed_radiometer`'s are ordinary, because each
declares a different function of τ as its width. Any ceiling on τ admits all three or refuses all
three — the same shape of error as the `|m|/s` ceiling §0.4 retracted, and for the same reason: a
parameter of one fixture's error law is not a coordinate on the region.

So the check reads the **density**, not a coordinate: it pins the parent at each point of the
package's existing probe grid and hands the realised conditional to R4's unchanged
one-dimensional rule. No new number, and the verdict carries the points it was taken at. What it
cannot see — a degeneracy between the grid's points or outside its ends — is **recorded rather
than gated**, because closing it needs a claim about the MEASURE of the degenerate set (`Z` is
finite when that set is null and undefined when it is not), and that is a threshold R5
pre-authorises none for.〕

**And this is where the ordering has to be asserted rather than assumed.** The premise a graph
refuses under does not identify its structure: `mixed_radiometer` is `gcr+mh` and refuses under
`evidence_prior_proper`, not under anything structural. Task 7 asserts the **order** of the
premise chain explicitly (§0.12), because two premises both holding and the earlier one answering
is not a bug, but reading structure off the answer is.
### 0.16 What was measured about the two candidates while planning, and what was deliberately NOT decided

**Ruling.** The facts below were measured in throwaway environments on 2026-09-04 (macOS arm64,
Accelerate). They are **inputs to Task 5's bake-off, not its verdict.** Task 5 re-measures every
one of them and adds the measurements that would actually separate the two, which have not been
made. Nothing here pre-selects a backend, and §0.5's "no candidate passed" branch stays live.

〔**Execution write-back, red line 11 (Wave D). Four of this section's measurements did not
reproduce, and the reason is instructive in three of the four cases: they are properties of the
ENVIRONMENT the measurement was taken in, recorded as properties of the package.**

* **Package counts are 3 and 16, not 4 and 19.** How many packages an install *adds* depends on
  what is already present, so the number belongs to the base environment and not to the
  candidate. Re-measured in throwaway copies of this repo's environment.
* **jaxns declares `jax>=0.6.0`.** §0.16 recorded no bound for it. A bound that exists and a
  bound that was not looked for are different findings, and only the second was written down.
* **`tfp_nightly` resolved to `dev20260905` — it moved overnight.** Which is not a contradiction
  but a confirmation: §0.16's own argument against jaxns is that an unbounded nightly is not
  pinnable, and the number changing between two consecutive days is that argument measured
  rather than asserted.
* **Three of Wave B's five fixtures are class (b), not two.** That count came from THIS session
  and it was wrong: the census filter treated `**overrides` as a required argument, because a
  `VAR_KEYWORD` parameter has no default, and so dropped `mixture_prior_residual` — which is
  class (b) with `exact=('b',)`, `method="gcr"`, `sampled=('w',)`. Third time in this batch that
  a census of mine dropped a row; the first two dropped them silently, this one named the drop
  and the reason it gave was wrong. **A named drop is not a correct drop.**〕

**Both install beside this repo's installed `jax`, and both coexist.** `jax` stays at `0.11.1` in
every case. 〔"Pin" is the wrong word and this plan used it: `pyproject.toml` declares
`jax>=0.5`, an open lower bound. 0.11.1 is what is **installed here**, not what is pinned — so
§1.5 condition 2 is a question about the installed stack, and a future `jax` is a separate
question this measurement does not answer.〕

| | blackjax 1.6.2 | jaxns 2.6.9 |
|---|---|---|
| packages added | **3** (4 in the planning environment) | **16** (19 in the planning environment) |
| declared `jax` bound | `>=0.9.0`, no upper cap | **`>=0.6.0`** — §0.16 first recorded none |
| heavy dependency | none | **`tfp_nightly`, unbounded** |
| entry point | `blackjax.nss` (nested slice sampling) | its own `Prior`/`Model` DSL over a unit hypercube |
| termination condition | **none — the caller writes the loop** | 11-field `TerminationCondition`, 12-bit `termination_reason` |
| `log Z` uncertainty | **none — the caller derives it** | `log_Z_uncert` returned |
| x64 | **runs at float32 and returns a plausible number** | **hard `OverflowError` without x64** |
| on the analytic 2-D Gaussian, 4 seeds | all within 0.46σ | all within 1.10σ, mean error −0.0003 |

**Three of these are load-bearing and each cuts a different way.**

**(1) jaxns writes `jax.config` at import, process-globally, and a FAILED import writes it too.**
`jaxns/internals/mixed_precision.py:13-15` runs `jax.config.update('jax_enable_x64', True)` at
module scope behind a `UserWarning`. This repository's stated rule is the opposite —
*"this package's rule is that `src/` never touches `jax.config`, so the caller opens
`with jax.enable_x64(True):`"* (`diagnose/identifiability.py:46-48`). **Measured consequence:**
with jaxns preloaded, `tests/dispatch/test_task_execution.py:339` —
`assert dict(run.jax_config)["jax_enable_x64"] is False` — fails with `assert True is False`.
**The dangerous case is narrower than an earlier draft of this section claimed, and the narrowing
matters.** In *this* checkout jaxns is not installed at all, so `import jaxns` raises
`ModuleNotFoundError` before any jaxns code runs, and the flag is `False` before and after —
measured. The contaminating case is the one in between: **jaxns installed, its `tfp_nightly`
dependency broken.** There `mixed_precision.py` runs and writes the flag, and *then* the import
fails — with an `AttributeError`, which `except ImportError` would not catch either. So the
hazard is not "a failed import always poisons the process"; it is **"a partially-installed jaxns
poisons the process, and the exception it finally raises is not the one a capability probe
catches."** Still the failure family `CLAUDE.md` is built around, and still invisible to a probe
that only asks whether the import succeeded. **Any capability probe R5 writes must be shown not
to do this in all three states — absent, broken, present — and the demonstration is a run, not a
code read.**

**Measured here, and it is worse than a red test. [planning]** Under a process-global x64 flip
(`JAX_ENABLE_X64=1`, which is what a jaxns import produces), **two** tests fail where both pass
without it:

```
FAILED tests/dispatch/test_evidence_task.py::test_a_float32_environment_is_refused_by_name
FAILED tests/dispatch/test_task_execution.py::test_the_run_record_says_what_actually_ran
```

The first is the one that matters. `dispatch/task.py:1045` decides `evidence_requires_x64` by
**outcome** — `ambient = jnp.result_type(float)` — deliberately, so that a caller who used the
context manager and a caller who threw the process-global switch get the same answer. That
design is correct and the gate is behaving correctly: the flag really was flipped. **The
consequence is that a third-party import chooses the caller's precision, and R4's precision gate
stops firing.** Nothing computes a wrong number — the assembly at float64 is right. What breaks
is the *semantics*: the gate exists so the caller declares the precision explicitly.

> **Owner decision, if Task 5 selects jaxns.** Either that test runs in a subprocess (the shape
> `test_public_api`'s subprocess assertions already use), or R4's gate needs a criterion a
> third-party import cannot move. **Both are changes to R4's shipped behaviour and neither
> belongs inside an R5 task.** Task 5's verdict must name this as a cost of choosing jaxns, and
> Task 6 does not start until it is ruled on.

**(2) The x64 contract differs qualitatively, and it is a selection criterion rather than a
detail.** jaxns REFUSES without x64 — which agrees with R4's `evidence_requires_x64`. blackjax
RUNS at float32 and returns a finite, plausible number. **If blackjax is chosen, bayesmith
enforces the x64 gate itself, because blackjax will not.** §3.5 already says x64 and scale checks
belong to the evidence gate; this measurement says which backend makes that load-bearing.

**Do NOT write that float32 biases blackjax's evidence.** Four seeds gave mixed signs
(+0.108, +0.055, −0.031, −0.033) with mean error +0.024 against a reported σ of ≈0.058 — that
does not establish a bias. What the four seeds DO support: float32 scatter (0.069) is about **4×**
the float64 scatter (0.016) **while the reported σ barely moves** (0.058 vs 0.062) — the error
bar degrades as a description of the spread, which is exactly what §0.8's stability gate is for.
Establishing or excluding a bias needs ~100 seeds and is a Task 5 measurement, not an assumption.

**(3) `tfp_nightly` is unbounded and resolves to a same-day build.** 2974 nightly releases exist;
today's resolution was `0.26.0.dev20260904`. Stable `tensorflow-probability 0.25.0` does **not**
substitute — `import jaxns` then dies with
`AttributeError: module 'jax.interpreters.xla' has no attribute 'pytype_aval_mappings'`. **An
optional dependency whose resolution changes daily cannot be pinned in a wheel that `publish.yml`
tests**, and this project has already spent four release tags on non-reproducible numerical
environments. Task 5 scores this against §1.5 condition 4 with the dates read.

**Two traps recorded so the adapter does not step in them.**

* **blackjax's live particles are not posterior samples.** Its own docstring: they *"collapse to
  the highest-likelihood mode"* at termination; correctly-weighted draws need `finalise()` then
  `sample()`. Wiring `state.particles` into `posterior_representation` passes a smoke test and is
  wrong.
* **`num_inner_steps >= max(5, 2 * dim)`**, documented, below which the evidence is biased
  **upward** for `dim > 10`. That is an empirical dispatch threshold in an upstream library, and
  `CLAUDE.md`'s boundary-validation rule applies to it: the adapter derives it from the residual
  dimension and tests **at** the boundary, including the extreme values.

**What has NOT been measured, and therefore what Task 5 must not skip:**

1. **Neither backend on a multimodal or non-Gaussian fixture.** The analytic Gaussian does not
   separate them — both are correct on it. blackjax's docstring warns its covariance proposal
   *"bridges between modes only up to moderate separation"*; jaxns advertises no equivalent. **This
   is the measurement most likely to decide, and it does not exist yet.** It needs Task 3's
   fixtures first, which is why Task 3 precedes Task 5.
2. **Neither backend on Linux/OpenBLAS.** Everything above is macOS/Accelerate. Given the four
   burned tags, a backend decision recorded as reproducible without the ubuntu job is not
   reproducible.
3. **Whether blackjax's cost is intrinsic.** End-to-end favours jaxns by two orders of magnitude
   (≈0.20 s vs ≈90 s), **but the cause is blackjax's post-processing, not its sampler**: the
   sampling loop is 1.4–2.6 s while `finalise` scales superlinearly (1.73 s at 595 steps,
   10.71 s at 2381). Treat the 90 s as the cost of naive documented usage until someone tries a
   pre-allocated buffer or a `lax.scan`. **Scoring §1.5 condition 3 on the naive number would be
   scoring an implementation choice as a library property.**
4. **What a blackjax adapter would have to OWN.** blackjax supplies neither a termination
   condition nor a `log Z` uncertainty, so both become bayesmith code with their own tests and
   mutation coverage. That is real scope, it lands on §1.5 condition 5 (adapter thinness) in the
   opposite direction from the dependency count, and Task 5 scores both rather than one.
### 0.17 `CompiledProblem` does not exist, so `CompiledEvidenceProblem` cannot be its variant, and that is said out loud

**Ruling.** R5 builds `CompiledEvidenceProblem` **standalone**, and records that the parent §8 R5
tells it to be a variant of has never been written. R5 does **not** build a general
`CompiledProblem` — that is compiler-breadth work, R6's subject, and none of §8 R5's seven work
items asks for it.

**Why. [planning]** `grep -rn --include='*.py' "CompiledProblem" src/ tests/` returns **nothing**.
The name appears four times in the normative design — as a Layer-2 artifact (line 190), as what
first-party compiler passes must output (207), as what first-party and upstream backends coexist
through (231), and in §7.3's question 7 (834) — and twice more as the thing
`CompiledEvidenceProblem` is to be a variant *of* (482, 998). It is an unbuilt design name. What
plays its role today is `InferencePlan` (`dispatch/plan.py:524`), which is an `eqx.Module` holding
the **Graph itself** — precisely what line 192 forbids a `CompiledProblem` from making a backend
re-interpret.

**So the ruling has a consequence that is the actual point.** `CompiledEvidenceProblem` is
designed to line 192's contract even though its parent is absent: *"residual log density,
transforms, constant terms, reconstruction map and what the backend adapter needs — but it may not
re-interpret the Graph."* Concretely, **it holds no `Graph`**, asserted by type in Task 1, so that
when `CompiledProblem` is eventually built the variant relation is a refactor rather than a
redesign. §7.3 question 7 — *"能否由 CompiledProblem 无损提供，而不让 backend 重新解释 Graph"* —
is answerable about `CompiledEvidenceProblem` alone, and Task 5 answers it that way.

**Write it into the close-out rather than leaving it for the next reader to discover.** A plan
that silently shipped a "variant" of a fiction would leave §8 R6 to find out.

### 0.18 §1.5 condition 3 expands to §3.5's eight axes, not to four

**Ruling.** §1.5 condition 3 says *"在 correctness、compile time、runtime 和 memory 上通过代表性
benchmark"* — four axes. It is **not** the operative list for this bake-off. §3.5 states the
comparison R5 is actually asked for:

> 使用相同的 analytic oracles、代表性的 astronomy-shaped residual problems 和运行预算比较
> **correctness、bias、uncertainty、termination、multimodal behavior、JIT/compile cost、memory
> 和 API stability**

Eight axes, and four of them — bias, uncertainty, termination, multimodal behaviour — are absent
from §1.5's four. **Task 5's table has eight columns, not four**, and the four extra ones are
where the two candidates are most likely to differ: §0.16 measured that the analytic Gaussian
separates them on none of §1.5's four.

**Two operationalisations, flagged as such rather than quoted as the document's words.**

* **"Minimal leakage", not zero.** §7.3 clause 5 asks *"backend-specific object 泄漏到公共 API 的
  范围是否最小"* — minimal, not none. R5 operationalises it as: **no backend type in any artifact
  field**, backend knobs confined to `EvidenceTask.backend_options`, and the adapter the only
  module importing the backend. That is a choice this plan makes, stronger than the document
  requires, and it is recorded as a choice.
* **"Generality" is not enumerated in §1.5.** Condition 1 says only *"对目标 problem family 足够
  通用，而不是只覆盖 demo"*. The dimension range, the non-Gaussian and the multimodal requirements
  come from §8 R5's completion gates and §3.5, not from that line. §0.5's row 1 reads them
  together; the join is this plan's, not the document's.
### 0.19 `evidence@2` is a SECOND gate definition, because "required here, optional there" has no field

**Ruling.** R5 ships `evidence@2` as a **second `GateDefinition` object beside a retained
`evidence@1`**, selected by `EvidenceTask.quality_gate`. It does not bump `EVIDENCE.version`.

**Why.** An earlier draft said `evidence@2` would "require the first two for a residual evidence
and not for an exact one". **There is no field that can say that.** `ReportRequirement`
(`artifacts/gates.py:103-125`) carries exactly `name`, `required: bool`,
`optional_error_blocks: bool` — no applicability-by-subject. And `GateDefinition` is a single
module constant `EVIDENCE` (`evaluation/evidence.py:422`) with one `version`; there is no registry
keyed by identity. Expressing applicability-by-subject would need a new field on
`ReportRequirement`, which **red line 4 forbids**.

**And bumping the version reddens a test outside the owning task's file list.**
`tests/evaluation/test_evidence_report.py:286` asserts `EVIDENCE.identity == "evidence@1"`. Task 8
lists only `evaluation/evidence.py` and `evaluation/gate.py`, so under red line 10 ("stage only
the files that task lists") the task would be uncommittable. **Task 8's file list is corrected**,
and `evaluation/gate.py` is **removed** from it — that module is `model_checking@1`'s runner; the
evidence gate lives entirely in `evaluation/evidence.py`.

### 0.20 The restatement removes the exact block from the audit, and one line has to put it back

**Ruling — WITHDRAWN, and it was false of the fixtures it is about.**
〔Execution write-back, red line 11 (Wave C). Measured: R4's rule short-circuits at
`if tuple(node.parents): ... continue` (`dispatch/evidence.py:956`) and **never reaches the
density this section's reasoning describes**. So "still passes R4's one-dimensional rule" is not
a statement about the latents in question — the rule does not evaluate them. The hole this
section worried about is closed by `check_gaussian`, which answers earlier and **symmetrically**:
the posterior task raises on the same graph, which is the asymmetry test §2.2 asks for, in the
one direction that matters. The original ruling read as follows and is kept for the record.〕

**Superseded ruling.** Task 7.2 asserts that **every exact-block latent with no residual-block
parent still passes R4's one-dimensional rule.** §0.15's restatement audits the residual block's joint prior
and says nothing about the exact block's own root priors, which R4 audited.

**Why this is a ruling and not a bug report.** The hole is **currently unreachable, and by
accident.** `_is_gaussian` gates exact-block membership, so an exact-block latent's own prior is a
diagonal Gaussian — proper and normalised by construction — and a `joint_prior`-covered latent
carries an `ImproperUniform` node density that fails `_is_gaussian` and is ejected to the residual
block, where `evidence_prior_undeclared` still sees it. So no live counterexample exists today.

**But the restatement's soundness would then rest on `_is_gaussian`, a predicate in a different
module written for a different purpose, and nothing would say so.** R6 widens the exact class;
that is precisely what opens this. One assertion costs nothing now and is unwritable later, once
the reason it passes has been forgotten.

〔**Execution write-back, red line 11 (Task 7.2). The ruling above is FALSE of the fixtures it is
about, and it was written from the density family rather than from the rule.** Measured:
`diamond_ancestor` and `indirect_ancestor` each carry an exact-block `x` whose only parent is a
**deterministic** node, so `x` has no residual-block parent and this ruling predicts PROPER.
R4's rule answered **UNVERIFIABLE** for both — because it short-circuits on `node.parents` being
non-empty and **never reaches the density at all**. The "Why" paragraph describes what
`_is_gaussian` guarantees about the declaration; the rule under discussion does not look at the
declaration. A prediction about a check has to be run against the check.

**The hole is real and it is closed by something else.** What protects an exact-block latent is
`check_gaussian`, which every block member passes through and which refuses a scale that is not
strictly positive and finite — at the same probe points a conditional check would use, and by the
same criterion, since a Gaussian with finite positive scale is proper and normalised. It answers
earlier and it answers for every task kind alike, which is why Task 7 does not restate it as an
evidence premise. §0.20's worry that the soundness would rest on a predicate in another module
with nothing saying so is **correct and now said**: it rests on `check_gaussian`, and the
assertion naming it is `test_an_exact_block_conditional_is_already_refused_structurally`, which
also pins the symmetry — the posterior task raises too.

**And the ruling's premise is wrong one level up.** It says §0.15's restatement "audits the
residual block's joint prior and says nothing about the exact block's own root priors". What
shipped audits **the graph's** joint prior, factorised along the graph, so every latent in both
blocks is audited through its own conditional. There is no exact-block gap to put back.〕

---

## Tasks

Waves, because agents must not edit the same file in parallel. **Wave A:** Task 1.
**Wave B:** Tasks 2, 3 (disjoint files). **Wave C:** Task 7 alone — the gates widen.
**Wave D:** Task 4 alone. **Wave E:** Task 5 alone — the decision. **Wave F:** Task 6 alone.
**Wave G:** Tasks 8, 9. **Wave H:** Task 10 alone. Shared files (`dispatch/task.py`,
`dispatch/evidence.py`, `pyproject.toml`, `README.md`,
`tests/numerical_gates/source_manifest.py`) are merged by the human integrator, never by an
agent.

**Execution order, since it is no longer task order:**
`1 → (2, 3) → 7 → 4 → 5 → 6 → (8, 9) → 10`.

> **The ordering is not the obvious one, and two measurements forced it.**
>
> **Task 7 moved ahead of Task 4** because Task 4's capability refusal is **unreachable until the
> gates widen**. Measured: today every class-(b) and class-(c) graph is refused earlier — six
> under `evidence_prior_proper`, thirteen under `evidence_residual_integral_required` — so a
> "backend absent" refusal can fire on none of them, and putting the capability check *ahead* of
> the structure gate would break class (a)'s 15/15 admission that R4 closed on.
>
> **Tasks 6 and 7 were in one wave and both modify `dispatch/evidence.py`**, which is the exact
> thing the wave rule exists to prevent. They are now separate waves.
>
> **Consequence for the stop-rules:** if Task 4 finds neither backend installable, Task 5 becomes
> a written "no candidate passed" and **Task 6 does not run — but Task 7 has already widened the
> gates.** Task 7.4 therefore requires classes (b)/(c) to route to the **capability refusal** when
> no adapter exists, so the widened classes are never admitted with nothing behind them.

**Every wave ends with an adversarial review before its commits are pushed** — see red line 1
for the contract, which is not optional and not a code read.

---

### Task 1: the prior/likelihood split, and `CompiledEvidenceProblem`

**Files:** Modify `src/bayesmith/dispatch/evidence.py`; Create
`tests/dispatch/test_compiled_evidence_problem.py`, `docs/probes/probe_34_residual_seams.py`.

- [ ] **1.1 Red.** Over **every** fixture in `tests/exact/models.py`, including the five that
      take constructor arguments, assert the two properties of §0.2 — (a) **per node term,
      bitwise**: each `Probabilistic` node's contribution appears exactly once and on the side the
      partition says; (b) **the recomposition to a derived band**
      `n · eps · max(|log_joint|, 1)` over a **declared seed set**, not one draw. **Do not assert
      the total bitwise**: measured, it fails 7–9% of draws on the two-observation fixtures
      because splitting one accumulator into two reorders a float sum, and pinning the order is
      red line 9. Assert the two `nan` fixtures put the
      `nan` on the likelihood side and leave `log_prior` finite. Assert
      `improper_outside_prior` raises at prior sampling rather than returning a number. Assert
      `CompiledEvidenceProblem` refuses construction when `exact_elimination` and
      `residual_parameters` share a name. Expect FAIL (module absent).
- [ ] **1.2 Implement.** The partition of §0.2, the parameter layout, and the prior transform
      or prior sampler. `CompiledEvidenceProblem` is a frozen dataclass; it holds **callables
      and arrays, no Graph and no backend object**.
- [ ] **1.3 Stop-rule, run here.** Record how many fixtures fail the bitwise identity. **If any
      fixture that is not already refused by an R4 premise fails it, stop and rule before
      widening.** A partition that is merely close has lost a constant, which is the entire
      failure mode the evidence layer exists to prevent. 〔Planning measurement: 46 bitwise,
      2 faithful-`nan`, 1 refused at prior sampling, 5 not run. The five are this step's job.〕
- [ ] **1.4 Green + lint.** `ruff check --no-cache` on the touched paths.
- [ ] **1.5 Commit.** `feat: compile a residual evidence problem with its prior and likelihood separated`

---

### Task 2: the quadrature oracle, with a convergence certificate it can fail

**Files:** Create `tests/dispatch/test_residual_oracle.py`; Modify
`docs/probes/probe_34_residual_seams.py`.

- [ ] **2.1 Red.** A 1-D and 2-D trapezoid oracle that **refines until its value stops moving**
      and returns ABSTAIN rather than a number when it does not. Assert it abstains on a span
      deliberately placed off the mass. Assert collapsed-vs-uncollapsed agreement on all four
      `gcr` class-(b) fixtures within a **derived** band. **The uncollapsed side is the oracle**
      (§0.4); the collapsed side is what is being graded, and the test says so in its own
      docstring so the next reader cannot invert them. Expect FAIL.
- [ ] **2.1b Demonstrate the oracle is not blind, per red line 1.** Scale
      `dense_operator`'s return by 1.03 and show the collapsed-vs-uncollapsed comparison goes RED.
      R4's close-out records that the square-root cross-route test survived exactly this mutation
      — *"5 passed, exit 0"* — because both of its sides shared `dense_operator`. **An oracle
      that cannot show what it kills is not an oracle**, and this is the specific mutation that
      proves this one is not the same mistake wearing a new name.
- [ ] **2.2 Implement.** **The GAP is the expected value; the twelve-digit sums are not**
      (§0.4, red line 9). 〔Planning gaps, each with the span that produced it recorded beside it:
      `diamond_ancestor` `0.0`, `indirect_ancestor` `±8.9e-16`, `shared_ancestor` `±1.8e-15`,
      `overflowing_outside_latent` `0.0`. An independent re-measurement with its own spans
      reproduced **all four gaps** and **differed on two of the values** — `shared_ancestor` by
      2.7e-6 because the planning span integrated only `τ > 0`, and `overflowing_outside_latent`
      by 1.3e-3 because its Cauchy tail needs `|z| ≤ 100` where the planning span used ~half that.
      Both differences are the oracle's own convergence, which is exactly why 2.1 requires the
      certificate and 2.3 pins the near-miss.〕
- [ ] **2.3 The near-miss is a regression test, not an anecdote.** Pin the unconverged
      `shared_ancestor` span that produced a **6.55-nat** gap during planning, and assert the
      oracle ABSTAINS on it rather than reporting the gap. Without this cell the certificate is
      a comment.
- [ ] **2.4 Class census as a standing test.** Assert the class table of §0.3 over every shipped
      fixture — 15 (a), 6 (b), 1 (b′), 13 (c), 9 (d), 5 compile-refused. A census run once is a
      census that goes stale; R4 learned this on the prior audit.
- [ ] **2.5 Green + lint. 2.6 Commit.** `test: an independent quadrature oracle for the residual integral, with a certificate it can fail`

---

### Task 3: the fixtures R5's completion gates require and this package does not have

**Files:** Create `tests/exact/residual_models.py`, `tests/dispatch/test_residual_fixtures.py`,
`docs/probes/probe_35_oracle_dimension.py`. **`tests/exact/models.py` is NOT modified**, and that
is a ruling rather than a convenience: it is a census DENOMINATOR in three places — Task 1 walks
it for the prior/likelihood split, Task 2's standing census pins 15/6/1/13/9/5 over its 49
no-argument graphs and 19/6/1/13/10/5 over all 54, and Task 7 walks it again for the propriety
audit. A fixture added there moves numbers in three places that have nothing to do with the
fixture.

> **The consequence, stated because it reads as a bug later.** `residual_models.py` is **not** in
> the census denominator, so its five fixtures — including the two class-(b) ones — do not appear
> in those counts and never will. Intended. A reader who finds `mixture_prior_residual` absent
> from Task 7's class table has found the design, not a gap.

- [ ] **3.1 Red.** Three fixtures, each with a **constructed closed-form** `log Z`, not a
      sampled one: (i) a genuinely **multimodal** residual posterior — well-separated mixture
      components whose evidence is a sum of closed-form terms; (ii) a **heavy-tailed** residual
      prior extending `overflowing_outside_latent`'s Cauchy; (iii) a residual of **dimension
      ≥ 4**, above anything shipped today. Assert each against its closed form and against the
      Task 2 oracle where the oracle applies. Expect FAIL.
- [ ] **3.2 Stop-rule, run here — IT FIRED, and the answer is below.** Record the dimension at
      which **`oracle_joint`** — total latent dimension, residual **plus** exact (§0.4) — stops
      converging within the declared budget. That dimension is the boundary of R5's gradeable
      domain: above it a backend answer has no independent oracle, and §9.1 does not admit a
      number whose only check is the route that produced it.

      〔**Measured at `MAX_POINTS = 5e7`. The boundary is FOUR.**

      | d | certified | n | evaluations | value − closed form |
      |---:|---|---:|---:|---:|
      | 1 | yes | 33 | 33 | −1.235e-10 |
      | 2 | yes | 65 | 4,225 | −1.743e-10 |
      | 3 | yes | 65 | 274,625 | −2.609e-10 |
      | 4 | yes | 65 | 17,850,625 | −3.646e-10 |
      | 5 | **no** | 33 | 39,135,393 | −6.094e-10 |
      | 6 | **no** | 17 | 24,137,569 | **+6.595e-03** |

      **The last column is the part that matters.** At `d = 5` the uncertified value is right to
      ten decimals; at `d = 6` it is wrong in the third — and **nothing about either number says
      which is which.** That is the certificate's whole reason for existing, and it is why an
      uncertified value is not a value.

      With the budget lifted the boundary is five (1.16e9 points, ~19 GB, 53 s); six needs
      `65**6 = 7.5e10`. So "the grid stopped it" and "the arithmetic stopped it" are different
      facts, and `probe_35_oracle_dimension.py` keeps them apart.

      **The domain statement is stricter than it looks.** `oracle_joint` integrates residual plus
      exact, so a residual of four beside ANY exact block is five axes and is already outside the
      gradeable domain at a test-affordable budget. **Task 5's correctness column depends on
      this.** Task 10 carries it into `docs/residual-evidence.md`; until that page exists this
      line is its home.〕
- [ ] **3.3 Green + lint. 3.4 Commit.** `test: a multimodal, a heavy-tailed and a four-dimensional residual fixture with closed-form evidence`

---

### Task 4: the optional extra, and the refusal that is the default state

**Files:** Modify `pyproject.toml`, `src/bayesmith/dispatch/task.py`, **`README.md`**; Create
`tests/dispatch/test_backend_absent.py`, `docs/probes/probe_36_backend_survey.py`.

〔Execution write-back, red line 11 (Wave D). This list named
`src/bayesmith/artifacts/refusal.py`, which the task correctly did **not** need to modify, and
omitted `README.md`, which it must — `tests/test_readme_count.py` pins the count by equality and
any new test moves it. A file list that names a file the work does not touch and omits one red
line 10 forces is worse than no list: it makes the executor choose between two rules.〕

- [ ] **4.1 Red.** With no backend installed — **the state this checkout is in today** — an
      `EvidenceTask` over a class-(b)/(c) graph returns a capability `Refusal` naming the
      missing extra and the install command, and a `PosteriorTask` over the **same graph** is
      unaffected. Assert both halves; the second is what separates a boundary from a
      regression. **This red is only reachable because Task 7 (wave C) has already widened both
      gates** — before that every one of these graphs is refused earlier. Expect FAIL.
- [ ] **4.2 Implement.** The extras table, the premise, its `_REMEDIES` row, and the probe.
      〔Execution write-back: this said "the import guard", which **contradicts 4.4's ruling that
      the probe must never import the candidate.** There is no import to guard; the probe reads
      `importlib.metadata` and the guarding is that it does not import. Wave D's review built the
      bypass this wording invites — falling back to `__import__` when the metadata lookup raises
      — and it survives the suite.〕 No `pytest.importorskip` on the absence path — the absence path must RUN.
- [ ] **4.3 The survey, recorded not assumed.** §0.16 carries a planning-time measurement of most
      of this; **re-measure rather than copy it**, and add what §0.16 says is missing. For each
      candidate: installed version, its
      declared `jax` requirement, whether it resolves against this repo's pin **without moving
      it**, whether the two coexist in one environment, the nested-sampling entry point's real
      signature, last release date and open-issue count with the date read, and the smallest
      end-to-end run against a closed-form 1-D Gaussian with its error and reported uncertainty.
- [ ] **4.4 Build the capability probe as a bypass, because one candidate poisons it.** jaxns
      writes `jax.config` at import, and §0.16 measured that **a FAILED import writes it too** —
      so `try: import jaxns except ImportError: pass` flips `jax_enable_x64` process-globally
      while returning a clean-looking negative. Write the probe, run it, and **show** the flag is
      unchanged afterwards. A probe asserted rather than demonstrated is the failure family this
      repository is built around.
- [ ] **4.5 Stop-rule, run here.** **If a candidate cannot be installed against this repo's
      pinned `jax` without moving the pin, it is out** — §1.5 condition 2 — and Task 5 scores it
      as failed rather than pretending the comparison is open. **If neither installs, stop:
      Task 5 becomes a written "no candidate passed" and Task 6 does not run.**
- [ ] **4.6 Stop-rule — IT FIRED. See the ruling below.** Install each candidate into a scratch
      copy of the venv and run
      `tests/dispatch/test_task_execution.py::test_the_run_record_says_what_actually_ran`.
      §0.16 measured that a global x64 flip turns **two** tests red — that one, and
      `tests/dispatch/test_evidence_task.py::test_a_float32_environment_is_refused_by_name`,
      which is R4's own `evidence_requires_x64` gate no longer firing. **If merely making a
      candidate importable reddens the suite, stop and rule** before the bake-off scores it — a
      backend that cannot be installed beside the tests is not a backend this package can adopt,
      whatever it scores on correctness, and repairing R4's shipped gate is not an R5 task.

      〔**Execution write-back, red line 11 (Wave D): the rule FIRED, and was reported as not
      firing.** Measured on a `git archive` snapshot in isolated venvs — control without
      blackjax, `1 failed, 3548 passed`; with blackjax 1.6.2 and nothing else changed,
      `2 failed, 3547 passed`. **Installing the thing the extra exists to install reddens the
      suite.** The executing session reported "neither fired" because the check it ran was two
      NAMED TESTS rather than the suite — a check narrower than the claim it was asked to
      support, which is this batch's founding failure in the instrument that was supposed to
      catch it.

      **The ruling, given by the plan's owner: repair, do not stop.** This rule exists to catch
      *a backend this package cannot adopt*, and that is not what fired. The cause is Task 4's own
      new test deriving its "absent" baseline from the checkout's package set — a test defect. The
      packaging is confirmed correct independently: the wheel carries both `Provides-Extra` rows,
      the probe reads the extra back out of an installed wheel, and the two §0.16 tests pass with
      the candidates installed while the `JAX_ENABLE_X64=1` control reddens them.

      **But it is recorded as FIRED.** "Did not fire" and "fired for a reason that turned out to
      be ours" are different facts and only the second is true. This also corrects the write-back
      that argued the rule *could not fire on any input*: it could and did — what could not fire
      was the narrower reading that was checked.〕
- [ ] **4.7 Green + lint. 4.8 Commit.** `feat: refuse a residual evidence by name when the optional sampler is absent`

---

### Task 5: the bake-off — §1.5's six conditions, both candidates, one verdict

**Files:** Create `docs/superpowers/specs/2026-09-XX-r5-backend-evaluation.md`,
`docs/probes/probe_37_backend_bakeoff.py`.

**This task produces a DECISION, not code.** No adapter is written here.

- [ ] **5.1** Calibrate the shared budget: drive each candidate to the same log-likelihood
      evaluation count on one fixture. **Do not wrap the likelihood in a Python counter** — §0.7
      measured that it reports 1 after 1000 jitted calls. Derive the count arithmetically and
      **audit the formula once** with `io_callback` or an unjitted run on a tiny problem.
      **Stop-rules, run here (§0.7):** a count that cannot be established or audited stops the
      work; so does a candidate that cannot be held within ±10% of the target.
- [ ] **5.2** Run the table: every admitted fixture from classes (b) and (c), plus Task 3's
      three, for each candidate, at the calibrated budget, at x64, with a declared seed set.
      **Eight columns, per §0.18 — correctness, bias, uncertainty, termination, multimodal
      behaviour, JIT/compile cost, memory, API stability — not §1.5 condition 3's four.** §0.16
      measured that the analytic Gaussian separates the candidates on none of the four; the
      deciding evidence is in the other four and does not exist yet.
- [ ] **5.2a The gradeable dimension bites earlier than the residual dimension suggests.**
      `oracle_joint` integrates residual **plus** exact, and Task 3.2 measured the boundary at
      **four total axes** at the declared budget. So a residual of dimension 4 beside any exact
      block is five axes and has **no independent oracle at a test-affordable budget** — the
      correctness column cannot be filled for it, and §9.1 does not admit a number whose only
      check is the route that produced it. Decide per fixture whether the column is *filled* or
      *withheld*, and say which; a blank that reads as a pass is the failure §11.4 names.
      〔`docs/probes/probe_35_oracle_dimension.py --lift-the-budget` separates "the grid stopped
      it" from "the arithmetic stopped it" — behind a flag and not run by CI, because at five
      axes it is 1.16e9 points and ~19 GB. Use it when the bake-off needs to know which of the
      two stopped a cell.〕
- [ ] **5.2b Re-run under Linux.** Everything in §0.16 is macOS/Accelerate. **A backend decision
      recorded as reproducible without the ubuntu job is not reproducible** — that claim cost this
      project four release tags. Record which CPU the job drew.
- [ ] **5.3** Score §1.5's six conditions one at a time, in the table of §0.5, each with the
      run that answers it. A condition answered by argument rather than by a run is scored
      **not met**.
- [ ] **5.4 Stop-rule, run here.** If a candidate's reported `standard_error` does not move with
      the budget, it is a placeholder: **stop and rule** before scoring §1.5 condition 4 (§0.8).
- [ ] **5.4b Name the costs that are not correctness.** For each candidate, the verdict records:
      the dependency footprint (§0.16), whether adopting it reddens any existing test and what
      repairing that would change about **R4's shipped behaviour** (§0.16's owner decision), and
      what the adapter would have to OWN that the backend does not supply — a termination policy
      and an uncertainty estimator are bayesmith code with their own mutation coverage, and they
      land on §1.5 condition 5 in the opposite direction from a small dependency count.
- [ ] **5.5 Write the verdict, including the branch nobody wants.** One of: a named winner with
      its applicable domain; a winner plus a retained second **with the run that justifies
      retention** under §0.6; or **"no candidate passed"**, with the table that says so and the
      list of which condition each failed. **The third branch closes R5 legitimately** — §8 R5
      asks for a reproducible decision, not for a backend.
- [ ] **5.6 Commit.** `docs: the R5 backend evaluation, six conditions scored on measured runs`

> **Owner gate.** **Task 6** does not start until the owner has read 5.5. A backend adapter
> written before the decision is a decision made by whoever wrote it first. 〔Execution write-back,
> red line 11: this line said "Tasks 6 **and 7**" until the wave reorder moved Task 7 ahead of
> Task 5 — Task 4's refusal is unreachable until the gates widen (§0.3, Task 4.1). Task 7 gates
> nothing on the backend decision and runs in wave C; only Task 6 waits.〕

---

### Task 6: the production adapter — only if Task 5 named a winner

**Files:** Create `src/bayesmith/bridge/<backend>_bridge.py`,
`tests/bridge/test_<backend>_bridge.py`; Modify `src/bayesmith/dispatch/evidence.py`,
`tests/test_layering.py`.

- [ ] **6.1 Red.** The adapter consumes a `CompiledEvidenceProblem` and **nothing else** — no
      `Graph`, asserted by type. It returns a populated `EvidenceResult`: `log_evidence`,
      `standard_error`, `residual_component` with its own `standard_error` and a non-empty
      `method`, `posterior_representation` as a `WeightedDrawsPosterior`, and a `RunRecord`
      whose `backend` is the real name (never `"auto"`) and whose `termination` maps through
      §0.1's declared total table. Assert the exact components R4 already computes survive
      unchanged beside the residual one. Expect FAIL.
- [ ] **6.2 Stop-rule, run BEFORE implementing: where does `CompiledEvidenceProblem` live?**
      `dispatch/execute.py:32` already carries a module-scope
      `from bayesmith.bridge.numpyro_bridge import nuts`, so the declared direction is
      **`dispatch → bridge`**; `bridge/numpyro_bridge.py` reaches back only *inside functions*.
      6.1 requires the adapter to consume a `CompiledEvidenceProblem` and assert it **by type** —
      and that class lives in `dispatch/evidence.py`, so a module-scope import of it from
      `bridge/` closes a cycle that
      `tests/test_layering.py::test_the_module_scope_import_graph_is_acyclic` raises on. **Rule
      first**: move the class to a leaf module (or to `artifacts/`) so `bridge` may import it at
      module scope, or declare the adapter's import function-scope and widen the
      one-function-scope-borrow rule. The plan's architecture paragraph claims "dependency
      direction is unchanged", and only one of those two keeps that true.
- [ ] **6.3 Implement.** One module. No backend type appears in any artifact field, asserted.
      Every number filed is passed through `float()` first — `_finite` requires
      `type(value) in (int, float)`, so a `np.float64` or a jax scalar raises `TypeError`
      (§0.1).
- [ ] **6.4 Stop-rule, run here.** If any backend termination signal has no honest
      `TerminationReason` member, **stop and rule** (§0.1). Do not add an enum member — that is
      a frozen-schema change and red line 4 forbids it.
- [ ] **6.5 Green + lint. 6.6 Commit.** `feat: a residual evidence adapter that consumes only a compiled problem`

---

### Task 7: BOTH gates widen — propriety first, then structure — and `gcr+snis` is refused by its own name

**Files:** Modify `src/bayesmith/dispatch/evidence.py` (the propriety predicate),
`src/bayesmith/dispatch/task.py`, `src/bayesmith/artifacts/refusal.py`; Modify
`tests/dispatch/test_evidence_task.py`, `tests/dispatch/test_evidence_audit.py`,
`tests/test_layering.py`.

- [ ] **7.1 Stop-rule, run FIRST, before any code.** Re-measure §0.15's table: for every shipped
      fixture, its structural class and the premise its `EvidenceTask` actually refuses under.
      **If class (b) is gated by `evidence_prior_proper` rather than by the structure premise —
      6 of 7 today — stop and rule on §0.15's propriety restatement before writing anything
      else.** Widening the structure gate alone admits one class-(b) fixture out of seven, and
      class (b) is §8 R5's headline work item.
- [ ] **7.2 Red, propriety.** The residual block's joint prior is audited **factorised along the
      graph**: each residual latent's conditional given its residual-block parents, with the
      block's roots through R4's existing one-dimensional rule. Assert all four hierarchical
      class-(b) fixtures pass; assert `improper_outside_prior` still refuses on `z`; assert class
      (a)'s per-latent marginal rule is **unchanged**, by re-running R4's shipped-fixture census
      and getting its recorded verdicts back. The census assertion **states its counting
      convention** (§0.15). **The target is FIVE fixtures by name.** Four from R4's family —
      `diamond_ancestor`,
      `indirect_ancestor`, `shared_ancestor`, `three_latent_chain` — because of the seven
      (b) ∪ (b′) graphs, `mixed_radiometer` stays refused by row (d), `improper_outside_prior`
      stays refused on `z`, and `overflowing_outside_latent` is unblocked by the structure
      widening alone. **The fifth is `mixture_prior_residual`** (`tests/exact/residual_models.py`,
      landed by Wave B at `9ba54c9`): verified class (b) with `method == "gcr"` — exact `('b',)`,
      sampled `('w',)` — and its residual posterior is **multimodal**, two well-separated modes
      carrying 0.43 and 0.57 of the mass, closed-form evidence `-13.332624149718965`.
      **Every class-(b) fixture this task had before it was unimodal**, so modality is a
      dimension the family held constant and the first multimodal graph R5's headline class
      admits would otherwise be one nothing here exercised — red line 1's enumeration, applied
      before the reviewer has to find it. It costs nothing: the fixture exists and its truth is
      closed-form. Also assert §0.20: every exact-block latent with no residual-block parent
      still passes R4's one-dimensional rule. Expect FAIL.
- [ ] **7.3 Red, conditional propriety over a declared range.** `p(x | τ)` must be proper across a
      **declared range of τ**, checked before the sampler runs, and a violation inside that range
      is a `Refusal`. Build the bypass: a prior that diverges only in a corner of τ-space, and
      show what today actually does. 〔**Execution write-back: today's behaviour is worse than
      this plan recorded.** The guard was described as an `eqx.error_if` abort, objectionable
      only for being an exception where §4.4 requires a Refusal. Built as a bypass and run: **it
      does not abort.** `marginal_log_density` returns `-inf` **silently**, because
      `pivots_constrain_block`'s floor is *relative* over the joint prior-and-data information —
      the data still constrains the block however improper the prior is. And `-inf` is the value
      §0.1 records as **unrepresentable** in `log_evidence`. So the failure is not "an exception
      where a refusal belongs"; it is a finite-looking pipeline producing a number the schema
      cannot carry.〕 §4.4 requires a statistical
      boundary to be a Refusal, not an exception. Expect FAIL.
- [ ] **7.4 Red, structure.** Rows (b) and (c) of §0.3 reach the adapter; row (d) refuses under a
      **new premise of its own** rather than `evidence_residual_integral_required`, whose message
      names R5 and would now be false. Expect FAIL.
- [ ] **7.5 Red, ordering.** Assert the premise chain's ORDER explicitly (§0.12) and that each
      refused graph still compiles a posterior task unchanged. **Do not infer structure from the
      premise a graph refuses under** — `mixed_radiometer` is `gcr+mh` and answers
      `evidence_prior_proper`; a test that read structure off the answer would pass today and
      mean nothing.
- [ ] **7.6 Lift the option refusal, which lives in a file no later task lists.**
      `task.py:1067-1095` refuses any `EvidenceTask` naming `repeat_count` or
      `reconstruct_posterior` under `task_options_recognised`. §0.8 honours `repeat_count`, §0.1
      requires weighted samples, Task 6.1 populates `posterior_representation` and Task 9.2 needs
      `repeat_count = n ≥ 3` — **and neither Task 6 nor Task 9 lists `task.py`.** Replace the arm
      here with one that still refuses both options on an **exact** evidence and admits them on a
      residual one. (`repeat_count = 0` stays refused at construction by
      `EvidenceTask.__post_init__`; that is `artifacts/tasks.py` and does not move.)
- [ ] **7.7 Implement the rest.** The new premise, its `_REMEDIES` row, the vocabulary
      enumeration, and the deletion of the part of `evidence_residual_integral_required`'s message
      that is no longer true — which after R5 is about **row (e)** and nothing else (§0.3). R4
      plan §0.14: each widening deletes its own refusal.
- [ ] **7.8 The reviewer enumerates the fixture family's CONSTANT dimensions and mutates each.**
      〔Wave B measured this over the same class-(b) family Task 7 grades, and hands over the
      list: **observed nodes per graph — exactly one, 5 of 5; the observation is a descendant of
      the exact block — yes, 5 of 5; the eliminated block's prior MEAN — exactly 0.0, 4 of 5;
      does `|m|/s` vary across the span — no, 5 of 5.** Every one of its four surviving mutants
      lived in one of those rows. Task 7 grades the same family, so it inherits the same holes
      unless it varies them. Wave B added `outside_observation_pair` (an observation the exact
      block does not reach) and `shifted_block_prior` (`x ~ N(tau, 0.3)`, so `|m|/s` sweeps
      1.3 → 13.3 across the span), both class (b) — use them rather than rebuilding them.〕
      Not "the axes this task did not vary" — red line 1 says why that weaker form fails.
      Mechanically: over the class-(b)/(c) fixture set, list every dimension of a prior
      declaration that takes the **same value in all of them** — the number of residual latents,
      hierarchy depth, whether a residual root is also an exact-block parent, the declared
      support's shape, whether any declaration is vector-valued — and produce one mutant per
      constant dimension.
- [ ] **7.9 Green + lint. 7.10 Commit.** `feat: audit the residual block's prior factorised, and admit the residual evidence classes`

---

### Task 8: the reports and the gate, at a new gate version

**Files:** Modify `src/bayesmith/evaluation/evidence.py`, `tests/evaluation/test_evidence_report.py`.
**Not `evaluation/gate.py`** — that module is `model_checking@1`'s runner; the evidence gate lives
entirely in `evaluation/evidence.py`. `test_evidence_report.py` is in the list because it asserts
`EVIDENCE.identity == "evidence@1"`, and red line 10 stages only the files a task lists.

- [ ] **8.1 Red.** Four report kinds — termination, repeated-run stability, residual geometry
      and dimensionality, prior-transform validity — each reaching PASS, FAIL and ABSTAIN.
      **`evidence@2` is a SECOND `GateDefinition` beside a retained `evidence@1`** (§0.19),
      selected by `EvidenceTask.quality_gate`; `EVIDENCE.version` is **not** bumped, because
      "required here, optional there" has no field on `ReportRequirement` and adding one is red
      line 4. Assert `evidence@1`'s identity is unchanged. Assert §4.3's
      operational status ahead of verdict: BLOCKED, INVALIDATED, ERROR, EVALUATED, and that
      report ORDER does not change the verdict. Expect FAIL.
- [ ] **8.2 Build the bypass, per red line 1.** The current guard requires that a filed slot's
      report **BE** the object the owning check returned. Write a bypass that files a
      PASS-shaped object the check did not return, run it, and show it dies. `CLAUDE.md`
      records that this exact guard was defeated once by `float("0.30")` and `operator.lt`; a
      guard whose bypass was never built is a guard nobody has tested.
- [ ] **8.3 Green + lint. 8.4 Commit.** `feat: termination, stability, geometry and transform reports behind evidence@2`

---

### Task 9: repeated runs, and the thresholds they need

**Files:** Modify `src/bayesmith/dispatch/evidence.py`,
`tests/numerical_gates/registry.py`, `tests/numerical_gates/boundary_residual.py`.

- [ ] **9.1 Write the threshold as a pure scalar comparison FIRST, per §0.10.**
      `spread_within_reported_error(spread, reported, factor)`, on SBC's
      `ranks_are_uniform(p_value, level)` pattern — the gate never samples; the seeded machinery
      lives in the test that calls it. Provenance `derived`, the form from the estimator's own
      variance argument. **Stop-rule: if the threshold cannot be written that way — if the gate
      itself would have to sample — stop and rule** between a new `ThresholdProvenance` member (a
      registry schema change) and §9.3's calibration-job demotion. Do not invent a sixth
      provenance in passing. Red line 8 is not satisfied by registering the number afterwards.
- [ ] **9.2 Red.** `repeat_count = n ≥ 3` produces `n` runs differing only in seed, populates
      `repeat_result_refs` and `consistency_report_ref`, and FAILs the gate when the spread
      exceeds the reported `standard_error` by more than the declared factor. Assert the
      asymmetry of §0.8: a spread much smaller does not FAIL. Assert `repeat_count = 0` is still
      refused at the Task boundary. Expect FAIL.
- [ ] **9.3 Implement**, with the seed set as part of the fixture and the acceptable
      false-positive rate declared in the test's own docstring (§9.3).
- [ ] **9.4 Boundary grid + fast-layer cell** for every registered threshold, plus **the seven
      side tables and the count pins** §0.10 enumerates — a `GateEntry` is built, not authored,
      and eight pins move. `test_boundary_layering.py` must stay green, and so must the three
      **full-layer-only** pins that a green fast layer cannot see.
- [ ] **9.4b Add the per-provider mutation-spec module and its own literal count**, the shape
      `test_boundary_sbc_provider.py` and its three siblings already use. R5's provider without
      one repeats R4's omission.
- [ ] **9.5 Green + lint. 9.6 Commit.** `feat: repeated-run stability, and the thresholds that decide it`

---

### Task 10: registry, docs, close-out — one session, no parallelism

**Files:** Modify `tests/numerical_gates/source_manifest.py`, `docs/README.md`, `README.md`,
`CLAUDE.md` **and** `AGENTS.md`; Create `docs/residual-evidence.md`,
`docs/superpowers/specs/2026-09-XX-r5-close-out.md`.

- [ ] **10.1** Regenerate the source manifest and the doc index; move this plan to `record` with
      every divergence written back on its own line, per red line 11.
- [ ] **10.2** `docs/residual-evidence.md` as a `module-spec`, in `docs/evidence.md`'s shape,
      including its own **"what a PASS does not mean"** section. At minimum: a stability PASS
      does not mean the estimate is unbiased, only that repeats agreed; a termination
      `CONVERGED` is the backend's opinion of its own stopping rule, not a statement about `Z`;
      an oracle agreement is a statement about dimension ≤ the boundary Task 3.2 measured.
- [ ] **10.3** Update `docs/ownership.md` — a `decision-home` — with the backend decision and,
      if a second was retained, the run that justified it.
- [ ] **10.4** Edit `CLAUDE.md` and `AGENTS.md` **in the same commit**, or
      `tests/test_agent_notes_are_one_file.py` reds. Re-measure the fast-layer count; do not
      quote it. 〔The accepted baseline is at `2d873f1`: **3264 passed, 0 failures, 0 errors,
      0 skipped**, 306.0s, exit 0, HEAD **and tree** pinned before and after. An earlier draft of this line
      quoted a **different** run — 306.1s, attributed to `9af68c3` — which is the very run §0.13
      voids for having no `sha` files at all. The plan broke its own rule in its own body, in the
      task whose job is to re-measure rather than quote. `CLAUDE.md` says 3249, true at R4's
      close.〕
- [ ] **10.5** Write the close-out on the real merged SHA with this run's JUnit counts. Lead
      with what the adversarial reviews found, as R4's does. Nothing is borrowed from R4's green.
---

**A note on how this plan was reviewed, because it is the shape every wave must copy.** Before
this document was handed over, two adversarial reviews ran in isolated worktrees under red line 1.
They **blocked it**: fifteen confirmed defects, including two that changed the mathematics —
`p(x | τ)` is not proper for every τ on a fixture this plan pins a number for, and the
prior/likelihood identity is float-associativity rather than a property of the partition, failing
7–9% of draws where the planning census got 46/46 on one lucky draw. A third was found by running
this plan's own rule rather than trusting it: the likelihood-evaluation counter §0.7 originally
mandated reports **1** after 1000 jitted calls. All fifteen are repaired above, each on the line
that asserted the wrong thing.

**Every one of them was invisible to reading**, and one reviewer walked into the same family
mid-audit — a background job piped through `| tail -40`, so the exit code it read was `tail`'s and
22 bytes of nothing looked like a pass. That is seven instances in this one batch of the failure
`CLAUDE.md` opens with, and none was caught by inspection.

## Red lines (self-check before each commit; a violation rolls that task back)

**1. Adversarial review runs BEFORE the wave's commits are pushed, in its own worktree.**
〔Execution write-back, red line 11 (Wave A): "before the commits are **pushed**" is load-bearing
and this line nearly read "before they are made". It cannot: `CLAUDE.md`'s mutation protocol
restores with `git checkout -- src/`, which restores to **HEAD**, so mutating an uncommitted tree
is a silent full revert of the work. Wave A did exactly that and lost its implementation — the
untracked test and probe survived, the tracked `src/` change did not, which is the failure
`CLAUDE.md` describes verbatim. The order is therefore: **commit locally → mutate → review →
follow-up commits → push.** A mutation script must additionally refuse to start on a dirty tree;
Wave A's does, and it fired on the next run because a timeout had left a mutant behind.〕
`Agent(..., isolation: "worktree")`. The prompt says **"build the bypass and run it"**, not "read
the code". The reviewer must return: what code it wrote, whether the suite is still green with
that code present, and a mutation table **naming every survivor individually** — a count is not a
table. R4 ran three such reviews and **two of them BLOCKED already-committed work**: one found 12
survivors out of 28 where self-review had found 9 kills out of 10, because every fixture had one
latent, one observation and one scalar sigma; another found a stock `HalfCauchy` judged IMPROPER
by a rule whose verdict depended on the units. **A shared worktree voids the result** — R4's first
review edited `src/` while the main session ran tests in the same checkout, and that batch of
mutation results was about the reviewer's mutants.

**The reviewer's mutation set is enumerated from the FIXTURES, not from the change.** Measured
twice. R4's review found seven survivors because *"every fixture had one latent, one observation,
one scalar sigma"*; §0.14's found two more — one 4.221 nats wrong with the whole fast layer green
— because a per-epoch declaration has three dimensions and every shipped fixture holds two of them
constant. **Both were stated as a property of the fixture family, not of the diff.**

The weaker form — *asking the author which axes they did not vary* — does not work, and the reason
is exact: the author's decomposition of the problem is what produced the blind spot, so the list
they write is isomorphic to the change they made and omits the same axes again. The session that
wrote §0.14's repair said it plainly — its mental model of the bug had no "mean" axis in it, and
the line above the one being fixed was edited without registering that it was a different thing.
**"Which dimensions take the same value in every fixture" requires no understanding of the change
and is answerable by grep.** That is the form every wave's prompt uses.

**2. A stop-rule is executed at the point it names, in the session that reaches it.** Not
deferred to the end, not "noted for later", not satisfied by writing it down again. R4's plan
§4.3 carried one, the executing session skipped it, and what it would have caught was six
crashes rather than the refusals the rule anticipated. Every `Stop-rule, run here` box in this
plan is a place where the work **stops** and the owner rules if the condition fires.

**3. The backend decision is measured, and "no candidate passed" is a legal answer.** Neither
BlackJAX nor JAXNS is presumed. §1.5's six conditions are scored one at a time, each against a
run rather than an argument, and a candidate failing any one does not ship. If none passes, that
is written down as the verdict with the table behind it, and R5 closes without a production
adapter. §8 R5's gate asks for a reproducible decision, not for a backend.

4. **No change to R1/R2/R3 frozen schema.** No new field, and **no new enum member** —
`TerminationReason` gaining a value to fit a backend is a schema change wearing a mapping's name.
§0.1 establishes that R5 needs none.

5. **No number without an independent oracle.** The residual `log Z` is graded against quadrature
of the graph's own density (§9.1 tier 1), never against another backend. Two backends agreeing is
a cross-check, which is tier 3, and it does not substitute for tier 1.

6. **No tolerance widened to reach green, and no oracle reported without its convergence
certificate.** A quadrature that has not converged and a defect in the thing it grades are
indistinguishable — measured during planning at 6.55 nats on a fixture whose true gap is 1.8e-15.

7. **Assert the consequence, not the spelling.** No guard that reads how the source is written.
`CLAUDE.md` records two guards defeated in one day by writing the forbidden thing a second way —
`ProducerRef as _PR`, and a threshold spelled `float("0.30")` with `operator.lt`. Ask first what
the forbidden thing looks like written differently.

8. **A new threshold stops the work.** Only **D111** and **D112** are pre-authorised.
**D113–D115 reserved, expected consumption 0.** A third number appearing during execution means
stop, register with a boundary grid and a fast-layer cell, then write code. And the registry has
no seeded gate at all, so the FIRST statistical threshold additionally needs §0.10's ruling.

9. **No fixture pins one machine's arithmetic.** Construct the stress deterministically, or
assert a property whose form is derived, saying which half is derived and which is measured. Four
release tags were spent on this.

10. **One task, one commit, red → implement → green → lint.** Stage only the files that task
lists; never `git add -A`. Run `git commit` as its own command. `ruff check --no-cache` — the
cache has held a lie in this checkout before.

11. **A decision's resolution goes on the line that asked the question** — in this document.
〔And **verify it landed.** Four repairs to first-review defects were written and silently lost:
the patch script applied edits in a batch and wrote once at the end, so a single failed anchor
discarded all of them, and only the one that printed an error was noticed and re-applied. Task 8
then carried a file list the review had already rejected, and the plan contradicted its own
§0.19, for two waves. Apply edits one at a time and write after each; a batch that can lose four
repairs while reporting one failure is the same disease as everything else here — a result that
cannot distinguish "applied" from "silently discarded".〕

12. **`CLAUDE.md` and `AGENTS.md` move in the same commit**, and every count is re-measured
rather than quoted.

13. **Put the old bug back and re-run. If the suite is still green, the repair is not
demonstrated.** Executable, not a maxim.

14. **A check that can decline to run must RECORD that it declined, in a value distinct from
"ran and found nothing".** Otherwise its silence is indistinguishable from its pass, and every
consumer downstream reads the second when the first is true.

〔Added during Wave C, whose review found the rule's own subject in the guard written to enforce
§0.15. `conditional_prior_range_report` reports `degenerate == ()` in two different situations:
it swept the grid and found no degenerate cell, and **`_probe_values` returned `None` so there
was no grid at all** — which it does for a StudentT, Laplace, Uniform, LogNormal or Cauchy
hyperprior. An identical `inf`-width conditional is refused under a `Normal` parent and
**admitted** under a StudentT one, where the corner carries **0.058** of the prior mass. The
refusal message then asserts "its conditionals are densities across the range the integral
covers", which is not merely optimistic but false: there was no range.

**The repair is not more probe points.** More points close the holes where the grid missed
something and leave this class untouched, because the failure is that the grid did not exist.
The two states have to be different values.

This is `CLAUDE.md`'s founding disease — a result that cannot distinguish the thing it names from
a thing resembling it — reaching the guard layer, and it is the eleventh instance in this batch.
The nine before it were checks that could not tell "absent" from "did not run"; the tenth was a
repair that could not tell itself from the bug. This one is a check that **reports the same value
whether or not it ran**, which is the same disease with the evidence deliberately discarded
rather than merely unavailable.〕

**It catches two different faults and cannot tell them apart, which is why the check is stated
before the diagnosis.** A green suite with the bug restored means either (a) **no input exists**
on which the fix and the bug differ — the fix is a rewrite, dead on everything the package can
build; or (b) inputs exist but **the fixture set has none** — the fix is real and ungraded. The
remedies differ: (a) needs a different fix, (b) needs a fixture. The check does not need to know
which, and an earlier wording of this line said only "what changed was the spelling", which
describes (a) and would have let (b) walk past.

〔Added during Wave A, which produced the first instance. `shapes` was always `()` because the
code read `graph.shape` behind a `hasattr` guard and `Graph` has no such attribute. The repair
replaced it with `batch_shape + event_shape` broadcast against the plate — the correct
expression — and **0 of 54 shipped latents have `batch_shape + event_shape` non-empty**, so on
every input this package contains the repair is identically the bug. Three mutants lived there,
inside the fix.

**Red line 13 does NOT subsume red line 1, and Wave B measured the gap.** That wave ran red
line 13 on **eight** repairs — each graded by restoring the old code — and **all eight went red**,
so the rule passed every time. Its adversarial review then found **four surviving mutants**, none
of which those eight checks could have caught, and the one mutant the review scored KILLED was
the one whose grading test had already been written.

The reason is structural: **every red line 13 mutant is enumerated from the repair's own diff.**
The rule asks "does this change behave differently from what it replaced", which is a question
about the change. It cannot ask "which dimension does the fixture family hold constant", which is
a question about the *fixtures* and is the one red line 1 exists for. **Passing 13 says nothing
about 1.** Wave B's session read red line 1 before starting and it did not stop them, which is
the honest form of this warning: knowing the rule is not the same as having a procedure that
executes it.

**A comment recording a removal is not a guard against re-adding it.** Wave B, making a field
nullable, re-introduced a condition **three lines under the comment explaining why it had been
removed** — the exact conjunct its own review had scored SURVIVED as decorative. The refactor made
the removal look like an oversight, and the comment said only that it *was* removed. The repair is
that the comment now says both why the thing is absent **and** why re-adding it would be
unreachable code guarding an unreachable state, because the next person making that change will
reach for it too.

**Both faults occurred in this batch, one per session.** (a) is Wave A's `_latent_shape`, above.
(b) is the campaign repair at `895e181`: `loc.ravel()[0]` was a genuine behaviour change on a
heterogeneous per-epoch mean, and **no shipped campaign fixture varies that mean**, so restoring
the old line left the suite green — the mutant M8 that a review later found surviving, 4.221 nats
from the dense truth. Same check, same verdict, different repair.

This is the tenth instance of `CLAUDE.md`'s family in this batch and the first that is not a
CHECK. The nine before were checks that could not distinguish the thing they named from a thing
resembling it. This is a **repair** that cannot distinguish itself from what it replaced, and its
diagnostic signal is weaker than a defeated guard's: a bypassed guard is still there to read,
whereas a repair equivalent to the old bug on all available inputs **reads as correct**, passes,
and carries a sincere commit message. What separated them was two fixtures the package does not
have.〕

15. **A test on a report is not a test on the gate.** Assert at the layer the consumer reads,
not at the layer that is convenient to call.

〔Added during Wave C. Two of its own tests called `conditional_prior_range_report` **directly**,
so two mutants that changed the **route** — whether the gate consults the report at all — left
them green. The report was correct in both cases; nothing was asking the gate. Red line 7 says
assert the consequence rather than the spelling; this is the same rule one layer out, and it is
the layer where a guard actually protects something. A report is an intermediate value, and a
test on an intermediate value grades the calculation, not the decision.〕

16. **An enumeration is a claim with a denominator, and this batch has never once got the
denominator right on the first attempt.** State the denominator wherever a count is asserted, and
treat any enumeration as provisional until something has tried to falsify it.

**The deny-list is the special case**: a deny-list is an enumeration of every bad value, so
**a new case must default to REFUSED, not admitted** — prefer an allow-list wherever an
unconsidered value can arrive.

〔**Every wave was wrong about an enumeration, and Wave B was wrong twice.** Wave A: the
graph-level terms, absent from 0 of 54 fixtures, so four mutants had nowhere to die. Wave B's
first review: five constant dimensions, correct and **incomplete** — its second found eleven
more, and **three of those were introduced by the first review's own repair fixtures**, so the
repair for a bad enumeration widened the thing it was repairing. Wave C: one parent per
conditioned latent, scalar, and every swept parent a root `Normal`, constant across all 60 graphs.

That is the pattern the deny-list case sits inside. Wave C shipped
`_RESIDUAL_METHODS_REFUSED`, a deny-list, with a comment claiming a new method
would be "a KeyError-shaped omission rather than a silent admission". A frozenset membership test
cannot raise `KeyError`: measured with `method="gcr+newthing"`, the structure refusal returned
`None` and the graph reached the capability refusal — **silently admitted** the moment a backend
exists. Replaced with `_RESIDUAL_METHODS_ADMITTED = {"gcr"}`, which also repaired a second false
message for free. The general form: a deny-list is a claim to have enumerated every bad case, and
this project has now been wrong about an enumeration in every wave.〕

17. **A hand-off separates what was MEASURED from what is PREDICTED, in the message itself.**

〔Added after this session sent one message carrying "3602 passed, exit 0" (measured, true) and
"your census test is now red" (predicted from the plan, **false** — the executing wave had already
updated it). Nothing in the message separated them, so the unmeasured half read with the authority
of the measured half, and the recipient was about to write a fix for a test that did not need one.
The failure was not predicting instead of measuring; it was **shipping both in one breath**.

**Corollary, from the same message:** it also said "I have not touched it", which was true of this
session and **not of the wave it was merging** — Task 7's executor had. *"I have not touched it"*
and *"nothing in the branch I am merging has touched it"* are different claims, and after a merge
only the second one is useful.〕

18. **A hand-off that depends on a particular session still being alive is not a hand-off.**
Put it in the plan, in a file, or in a failure message — anywhere a stranger will meet it without
knowing to ask.

〔Stated by Wave B's executor as it neared the end of its context, and it is the operating
principle this whole batch has been running on without writing down. Three sessions executed R5
in one checkout; each will end. What survived is what landed somewhere a reader trips over:
`test_where_each_new_fixture_sits_in_the_structural_taxonomy` is safe because its **failure
message names its own remedy** — Task 7's executor updated it without asking anyone, and produced
a better version than its author's, because it also recorded that the taxonomy columns had not
moved. What did not survive is whatever stayed in a session's head.

The corollary is a duty rather than a caution: **if you can think of something you would want
from a session that is ending, ask now.** Waiting until the moment you need it is waiting until
it is gone.〕


---

## Completion gates: §8 R5's seven, made countable

| # | §8 R5 gate | countable |
|---|---|---|
| G1 | backend 决策有可复现 benchmark、oracle 结果和明确适用域，而不是预设偏好 | §1.5's six conditions scored per candidate per fixture, each row citing the run; the applicable domain stated as a structure class and a dimension bound; **or a written "no candidate passed" with the same table** |
| G2 | 在 analytic oracle 上 log evidence 与声明误差一致 | every admitted fixture within the declared band of **`oracle_joint`** (§0.4) — never `oracle_collapsed`, which is tier 5 with respect to the elimination, the oracle carrying its convergence certificate and its `dense_operator` kill demonstration; the band's FORM derived |
| G3 | 至少覆盖一个非高斯或多模态 fixture | Task 3's multimodal fixture with a **constructed** closed-form `log Z`, plus the heavy-tailed one; `student_t_likelihood` and `non_gaussian_observed_node` already ship and are included |
| G4 | exact collapse 与未 collapse 的小问题对照一致 | all four `gcr` class-(b) fixtures, `oracle_collapsed` against `oracle_joint`, within the derived band, **each with its integration span recorded** 〔planning: gaps ≤3.6e-15, and two of the four spans were found to be unconverged on re-measurement — the gaps held, the values did not〕; and the three gradings of §0.4 kept separate, so a failure names which half moved |
| G5 | optional dependency 缺失时清晰拒绝，不破坏核心安装 | the extra absent — CI's default — an `EvidenceTask` refuses by name and a `PosteriorTask` on the same graph is unaffected; built-and-run, not mocked |
| G6 | 不稳定的重复估计不能通过 evidence gate | `n ≥ 3` seeds; a run whose spread exceeds its reported `standard_error` by the declared factor FAILs `evidence@2`; the asymmetry of §0.8 asserted in both directions |
| G7 | EvidenceResult 可用于 posterior reconstruction 和比较报告 | `WeightedDrawsPosterior` populated with `log_weights`, `ess`, `khat` — from `finalise()`+`sample()`, **never from live particles**, which are the highest-likelihood mode (§0.16); a Bayes factor between a residual evidence and an exact one carries the sampled side's error bar |

**A repository defect found by Wave C, for Task 10 to place.** `tests/test_readme_count.py`
shells out with `sys.executable` and **no `PYTHONPATH`**, so **inside a git worktree a bare
`python` resolves `bayesmith` to the shared checkout at `/Users/zzhang/projects/bayesmith/src`,
not to the worktree.** It therefore collects the worktree's tests against the *other* checkout's
source, and errors on any symbol the branch adds. Same family as the ruff cache: a result that
cannot distinguish "the count is wrong" from "the check measured the wrong tree". It matters now
because every adversarial review in this batch runs in a worktree. `CLAUDE.md` is where it
belongs, and `CLAUDE.md` cannot move without its `AGENTS.md` twin under red line 12, so Task 10
carries it.


Plus the four shared gates R1–R4 used: source full suite, `ruff check --no-cache`, built-wheel
suite, rheplicant consumer gate. **R4 skipped the last two because it was not a release; R5
should not inherit that** — R5 adds an optional extra and a `pyproject.toml` change, which is
precisely what a source-tree run cannot check. See §0.9.

---

## D-numbering and probe numbering

- **Probes:** `probe_34_residual_seams.py` (Task 1), **`probe_35_oracle_dimension.py` (Task 2)**,
  `probe_36_backend_survey.py` (Task 4), `probe_37_backend_bakeoff.py` (Task 5). Highest existing
  is `probe_33`; **`probe_32` was reserved and never written** — the manifest-generator spike
  shipped as `tools/sync_source_manifest.py` instead — and it stays unused rather than being
  recycled.

  〔**Execution write-back, red line 11 (Wave B).** This plan gave `probe_35` to Task 4 and
  `probe_36` to Task 5, and did not foresee Task 2 needing a probe at all. Task 2's executing
  session took `probe_35` for the oracle-dimension sweep, which is the right number for the
  order the work actually ran in; Tasks 4 and 5 shift up by one. Recorded here rather than
  renamed there, because the file exists and the plan is what was wrong.〕

- **D-numbers:** highest registered was **D110**, confirmed against
  `tests/numerical_gates/registry.py` rather than against the docs. R5 declares **D111** and
  **D112**; **D113–D115 reserved, expected consumption 0.** The likelihood-evaluation budget, the
  seed set and the five component names are fixtures and derivations, not thresholds, and are not
  numbered. Magic-number target: 0.

  - **D111 — CONSUMED (Task 2).** The residual-oracle agreement band, `1e-9` relative.
    〔The plan pre-authorised D111, so red line 8 has not fired. **What still has to be on the
    registry line is which half is derived and which is measured**: `ThresholdProvenance` has
    exactly five members — `derived`, `borrowed`, `magic`, `exact_or_domain`, `api_contract` —
    and none of them is a statistical category, so a level that was measured must say so in the
    prose beside the enum, the way D107's "derived form, measured coefficient" does. §0.10's
    stop-rule applies to the NEXT one, not this one.〕
  - **D112 — free.** Reserved for the repeated-run stability factor, form per §0.10's ruling.

  〔**Execution write-back, red line 11 (Wave B close-out): D111 needs no registry entry, and
  the definition of done's item 6 is over-broad as written.** It says D111 "is registered with
  boundary grids and named fast-layer cells". Measured against the precedent rather than
  reasoned: `GATE_REGISTRY` holds **113** gates, and the two the evidence layer added are
  `EVIDENCE:increments_converge:convergent-increment-ratio` (D109) and
  `EVIDENCE:mass_is_normalised:normalisation-tolerance` (D110). **D107 is not among them.** It
  appears only in `docs/evidence.md`'s number table and in test prose, and no gate id in the
  registry contains a D-number at all.

  The line the precedent draws is not "which D-numbers matter" but WHERE THE THRESHOLD LIVES.
  D109 and D110 are predicates in `src/bayesmith/dispatch/evidence.py`, which is in
  `SOURCE_PATHS`; the registry exists so that a threshold in shipped source cannot exist without
  a boundary grid and a mutation. D107 governs a COMPARISON IN A TEST and is registered nowhere.
  D111 is D107's kind — `AGREEMENT_FLOOR` is in `tests/dispatch/residual_oracle.py`, which the
  source scan does not walk — so registering it would mean inventing a `SourceAnchor` for a
  module the census has no opinion about, and moving the 97/212 pins for a gate that guards no
  shipped code.

  What item 6 is actually asking for, and what Wave B delivered: the band's FORM is derived and
  its LEVEL is measured, both halves said out loud, with the measurement pinned in both
  directions by tests rather than by prose — it is four orders above the routes'
  float-level disagreement (worst 1.95e-14 on `indirect_ancestor`) and seven below the smallest
  defect it must catch (2e-2 from scaling `dense_operator` by 1.03). `ThresholdProvenance` has
  no member for a measured statistical level, which is exactly why that sits in prose here and
  beside D107 in `docs/evidence.md`.

  **If Task 5 consumes D112 for the repeated-run stability factor, this ruling does not
  transfer.** §0.10 requires that gate to be a real one — a stochastic threshold with no
  precedent in the registry — and if it lands in `src/` it is D109's kind and needs the grid,
  the fast-layer cell and the pin updates.〕
---

## What R5 explicitly does not do

- **Write a nested sampler.** §1.5 places general nested sampling outside bayesmith's ownership
  in as many words. If no upstream passes the six conditions, R5 ships the compiler and the
  refusal, not a first-party sampler.
- **Admit `gcr+snis` or `gcr+mh`.** §0.3(d). Refused by name, with the route recorded.
- **Chain evidence.** Still refused; three of `chain.py`'s constants remain unconvicted by
  deletion, which R4 recorded and R5 does not repair.
- **A discrete evidence route.** R4 recorded that `exact/discrete.py`'s
  `marginal_log_likelihood` is `log p(θ, y)` and not `log p(y)`. That statement stands and R5
  does not act on it.
- **Campaign evidence.** The marginal-layer defect is repaired (§0.14); the **evidence** layer's
  refusal of the campaign family is unchanged, because an evidence over a campaign needs its own
  oracle and R5's is built for the residual integral.
- **Correlated observation noise**, on the same terms R4 set: admitted only where a dense oracle
  exists, refused rather than shipped ungated where none does.
- **Rescale a likelihood to fit a backend's accumulator.** §0.11: a rescaled likelihood is a
  different `Z`. R5 measures the range and refuses outside the declared domain.
- **Add a `ComputeBudget` field for live points**, or a `TerminationReason` member. §0.1.
- **Amortized or SBI backends.** §7.3 routes those to a separate evaluation; BlackJAX being
  adopted here would not make it a candidate there.

---

## Definition of done

"R5 closed" may be written only when all of the following hold:

1. G1–G7 each have this-run measured evidence, and the shared gates are green on one commit —
   **including the built-wheel suite and the consumer gate**, because R5 changes packaging;
2. `CompiledEvidenceProblem` exists, carries prior and likelihood separately, holds no `Graph`
   and no backend object, and §0.2's TWO assertions hold over every shipped fixture — per-node
   bitwise, and the recomposition inside a derived band over a declared seed set;
3. the backend verdict is written with §1.5's six conditions scored per candidate against runs —
   **or "no candidate passed" is written with the same table**, which closes R5 legitimately;
4. classes (b) and (c) are admitted **through both widened gates** — propriety and structure,
   §0.15 — with the four named fixtures passing; `gcr+snis`/`gcr+mh` refused under its own
   premise; row (e), the latent-free graph, refused under a premise whose message is true of it;
   and every R4 premise still fires ahead of the backend **in the code's real order** (§0.12),
   with the posterior asymmetry asserted;
5. `oracle_joint` and `oracle_collapsed` are named separately, only the first grades the
   elimination, every oracle carries a convergence certificate it can fail with its span recorded
   beside every number, the 6.55-nat near-miss is pinned as a regression, and the
   `dense_operator × 1.03` kill is demonstrated;
6. every consumed D-number that is a threshold **in `SOURCE_PATHS` source** is registered with a
   boundary grid and a named fast-layer cell; every one that governs a **comparison in a test**
   carries its derived form and its measured level in prose, with the level pinned in both
   directions by tests — D107's precedent, measured in §D-numbering. D111 is the second kind.
   §0.10's ruling on statistical provenance is written on its own line;
7. every wave had an adversarial review in its own worktree that built and ran a bypass, and each
   review's survivors are named individually in the close-out;
8. R1–R4 schema is unmoved — **no field and no enum member** — `evidence@1` still answers
   `"evidence@1"`, the exact route's numbers are unmoved, and the module-scope import graph is
   still acyclic (§0.19, Task 6.2);
9. the close-out uses this batch's real SHA and JUnit counts, borrows nothing from R4, and leads
   with what the reviews found.
