# R5 backend evaluation — BlackJAX NSS and JAXNS, scored on measured runs

> **文档状态：`record`** · 已落地批次/审计/测量的历史记录，非当前权威。索引见 docs/README.md。

**Date:** 2026-09-06 · **Task:** R5 Task 5 · **Probe:** `docs/probes/probe_37_backend_bakeoff.py`

This page holds the verdict R5's first completion gate asks for: *"backend 决策有可复现
benchmark、oracle 结果和明确适用域，而不是预设偏好"*. It scores §1.5's six conditions,
one at a time, each against a run. A condition answered by argument rather than by a run
is scored **not met** (plan 5.3).

**Nothing here is an adapter.** Task 5 produces a decision; Task 6 does not start until the
owner has read the verdict in §9.

---

## 1. What was run

`docs/probes/probe_37_backend_bakeoff.py`, eleven sections, in two environments. Every cell
of sections 4–7 runs in its own **subprocess**, for three reasons that were measured rather
than assumed: importing `jaxns` writes `jax_enable_x64` process-globally, so a shared process
lets one candidate set the other's arithmetic; a compile time is only a compile time in a
process that has not already compiled the same graph; and peak RSS belongs to a process.

**The fixture set is 21 rows, and the census is the dispatcher's, not a list.** Plan §0.5 row 1
says "20 today". Measured: of the 57 graphs the two fixture modules ship, **18** reach
`capability_unavailable_r1` — the premise a graph arrives at when every statement about the
MODEL has passed and the one missing thing is a sampler. Those are 5 class-(b) and 13
class-(c). Task 3's three join them, for 21. The two exact+residual graphs that do **not**
join are refused about themselves rather than about the backend: `improper_outside_prior`
under `evidence_prior_proper`, and `mixed_radiometer` under
`evidence_residual_method_unsupported` (it is `gcr+mh`, plan §0.3 row (d)).

**Class (b) is run as R5's route (b)**: `collapse_graph` eliminates the exact block and the
sampler sees the reduced graph. Both routes produce the same total `log Z`, and `oracle_joint`
grades the *uncollapsed* model, so it shares nothing with either.

---

## 2. What each backend needs that a `CompiledEvidenceProblem` does not carry

`CompiledEvidenceProblem` carries `log_prior`, `log_likelihood`, `prior_sample`,
`exact_elimination`, `residual_parameters`, `shapes`, `prior_terms`, `likelihood_terms`.

**BlackJAX consumes it and nothing else.** `blackjax.nss` requires
`(logprior_fn, loglikelihood_fn, num_inner_steps)`; the initial live set comes from
`prior_sample`. Nothing beyond the compiled problem is needed — and nothing beyond the
samples is supplied: **no termination rule, no `log Z`, and no uncertainty**. All three are
written by the caller — in this probe, three blocks of `run_blackjax`, and in production three
bayesmith modules with their own mutation coverage.

**JAXNS cannot consume it.** `jaxns.Prior` takes a tfp distribution or, through
`BaseAbstractPrior`, the pair `_forward(U) -> X` / `_inverse(X) -> U`: a **quantile map**
between the unit hypercube and the parameter. None of `log_prior`, `log_likelihood`,
`prior_sample` yields one — a density is not invertible by inspection and a sampler is not a
transform. The generic route, and the one this bake-off uses, is a bounded box `B` with

```
jaxns prior       = Uniform(B)
jaxns likelihood  = log pi(x) + log L(x) + log|B|
```

so jaxns integrates `∫_B pi L dx`. Three consequences, and all three are costs:

1. **`B` is a bayesmith object jaxns needs and blackjax does not.** Here it is the same
   `Span` tuple the quadrature oracle already requires, which is why it exists at all; a
   production adapter would have to derive one per graph.
2. **The answer is the evidence truncated to `B`.** The mass outside is an error the oracle's
   own truncation term bounds, and it is an error blackjax does not have.
3. **The nested sampler now contracts against `pi L` rather than `L`.** That is a different
   problem from the one nested sampling's efficiency argument is about.

---

## 3. §5.1 — the shared budget, and the audit that makes the count usable

The shared currency is **log-likelihood evaluations**, and neither backend's count is taken
on trust. The instrument is checked first: a plain Python closure reports **1** for a jitted
`lax.scan` over 1000 points and **1** for an unjitted `vmap` over 1000, re-measured in this
checkout, so both audits use `jax.experimental.io_callback`.

**BlackJAX — derived, then confirmed exactly.**

```
evaluations = num_live + Σ_steps ( Σ num_expansions + Σ num_shrink
                                   + 2 · num_delete · num_inner_steps )
```

The `+2` per slice step is the pair of terminating `in_slice` calls `stepping_out`'s two
`lax.while_loop` conditions make and do not count: `in_slice(left) & (n > 0)` is not
short-circuiting, so the condition evaluates the likelihood on the iteration that ends the
loop as well, once on each side. `num_expansions` counts body executions; `_shrink`'s
condition reads no likelihood, so `num_shrink` is exact. Audited: **derived 457, truth 457**.

Taking that audit needed one more step, and it is the section's own instance of §0.7's
problem. `io_callback` is refused inside a `lax.while_loop` with a batched predicate, which
`blackjax.ns.from_mcmc`'s `jax.vmap(mcmc_kernel)` makes it; and a Python counter under
`jax.disable_jit()` does not help, because `vmap` traces regardless — an audit taken that way
reported **exactly 6 calls per slice step at three different step counts**, a trace count
wearing an evaluation count's clothes. So the `vmap` is replaced by an index-and-restore for
`num_delete == 1`, and the probe asserts that shim is inert by running the same seed both
ways and comparing expansion counts, shrink counts and every dead particle's log-likelihood.

**JAXNS — its own report, admissible because the audit confirms it.** The audit runs at two
sample caps, because the answer is not the same at both:

| `max_samples` | binds? | reported | `io_callback` truth | exact |
|---:|---|---:|---:|---|
| 1000 | no | 45 639 | 45 639 | **yes** |
| 300 | yes | 12 792 | 13 993 | **no** |

At a cap the run does not reach, jaxns's report is exact. At a cap that binds it is short by
the work already spent in the shell the cap interrupted. So the count is usable only when
`reached-max-samples` is absent from the termination bits, and every cell records the bits and
carries a `count_audited` flag. No row of §5.2 or §5.3 hit the cap.

**Calibration.** Target **200 000** evaluations, tolerance **±10 %**, on
`student_t_likelihood` — class (c), one residual axis, certified oracle, non-Gaussian, so the
budget is not calibrated on the one shape §0.16 already measured both candidates identical on.
Both settled in two trials and the knobs were then frozen for the whole table:

| candidate | knob | value | evaluations |
|---|---|---:|---:|
| blackjax | `num_live` (`num_delete=1`) | 816 | 202 667 |
| jaxns | `num_live_points` | 304 | 198 217 |

blackjax's `num_inner_steps` is **derived per fixture** as `max(5, 2·dim)`, the bound its own
docstring gives and the one §0.16 flags as an empirical dispatch threshold; it is not a free
knob and is not tuned.

**Neither §0.7 stop-rule fires.** The count is established for both — one by an audited
formula, one by an audited report — and both were driven inside ±10 % of the target.

---

## 4. §5.4's stop-rule — is either reported uncertainty a placeholder?

Run before either candidate is scored on §1.5 condition 4, over a budget ladder of
×0.25 / ×1 / ×4 on the calibration fixture, whose oracle is **−9.933854850401152**.

| candidate | evaluations | `log Z` | reported σ |
|---|---:|---:|---:|
| blackjax | 50 316 | −9.92266 | 0.1117 |
| blackjax | 202 667 | −10.0158 | 0.05472 |
| blackjax | 803 522 | −9.90939 | 0.02923 |
| jaxns | 49 933 | −9.69493 | 0.2673 |
| jaxns | 198 217 | −9.92533 | 0.1421 |
| jaxns | 787 380 | −9.95291 | 0.07163 |

**Neither is a placeholder.** Both error bars move with the budget and shrink monotonically,
and both HALVE as the budget quadruples — the `1/√N` scaling nested sampling predicts, which is
a stronger statement than "the number changed". **The stop-rule does not fire for either
candidate**, and §1.5 condition 4's uncertainty clause may be scored.

**One asymmetry the stop-rule as written does not cover, and it had to be decided here.**
blackjax reports no `log Z` at all, so there is no "backend's reported `standard_error`" to
test: what is tested above is the error bar `run_blackjax` computes from
`blackjax.ns.utils.log_weights`' 100-path volume ensemble. That it scales as `1/√N` is evidence
about bayesmith's estimator on blackjax's samples. §1.5 condition 5 counts the writing of it
separately, in §8.3.

Worth recording beside it, because it is the shape the seed sweep in §6 tests: at the same
budget jaxns reports an error bar about 2.6× wider than blackjax's, and its single-seed error
is the smaller of the two. A tighter bar is not a better one.

---

## 5. §5.2 — the table

### 5.0 The correctness column, and where it is WITHHELD

The reference is `oracle_joint`: deterministic trapezoid quadrature of the model's own
`log_joint` over every latent, carrying the convergence certificate Task 2 built. It reaches
the model's `log_prob` and shares nothing with the elimination or with either backend, so it
may grade both — design §9.1 tier 1. Red line 5: a backend is never graded against the other
backend, which is tier 3 and does not substitute.

**Spans are placed by a rule declared before any span existed**, because a hand-placed span per
fixture is how a correctness table gets tuned to the answer it wanted. Where this repository
already declares one — `test_residual_oracle.py::CLASS_B`, `CAUCHY_SPAN`, `_quartet_spans` —
the bake-off uses that, with its committed reasons. Otherwise: **K = 9** prior standard
deviations either side of the prior mean, read off the latent's own distribution at the prior
centre. If the oracle then abstains **because the mass is still growing at the span edge** — its
`rho >= 1` test, the one a wider span can answer — the rule escalates once to K = 25 and records
which rung was used. It does not escalate on a refinement-budget abstain, which a wider span
makes strictly worse. The refinement count is arithmetic: the largest ladder of `n -> 2n - 1`
doublings whose last grid still fits the oracle's own `MAX_POINTS`.

#### The certificate is not sufficient, and this bake-off found out the expensive way

`high_snr_curvature` was certified at **−2 376 535.508689067**, with a bound of `9.54e-09`.
Both candidates answered **+131.2** and **+131.5**. A table built on that certificate would
have reported both backends 2.4 million nats wrong on a shipped fixture.

The oracle is the one that was wrong, and nothing in its certificate could say so. The fixture
has `sigma = 2e-6`, so the integrand falls 150 nats within `1e-5` of its maximum at `w = 1.0`,
while the finest grid the point budget affords on `w ∈ (−9, 9)` is spaced `7.0e-4` — **three
orders of magnitude too coarse**. The trapezoid never sampled the peak, so refining the grid
changed almost nothing, so the increment-ratio test and the geometric-tail test both passed on
the integral of everything except the mass. **A quadrature that misses a peak converges
beautifully.**

Established without reference to either backend, from the model's own `log_joint`:

| w | `log_joint` |
|---|---:|
| 1.0 | +145.022 |
| 1.0 ± 1e−6 | +143.501 |
| 1.0 ± 3e−6 | +131.332 |
| 1.0 ± 1e−5 | −7.085 |
| 1.0 ± 1e−3 | −1 520 930 |

Laplace from the peak's own height and curvature — `145.02 + log(√(2π)·5.7e−7)` — gives
**+131.6**. The −2.4e6 the oracle reported is `log_joint` at the grid point nearest the peak.

**The repair, and it is a declared rule rather than a special case.** A quadrature is admitted
only if its finest grid puts at least one point within one curvature width of the integrand's
peak — the weakest test that catches this, with no tunable factor in it. The peak and its
widths come from multi-start gradient ascent on `log_joint` (the prior centre plus eight prior
draws, so a multimodal integrand does not have one mode's width reported as its own) with the
Hessian read at the best point. When the declared span's grid fails that test, the span ladder
gains a last rung: **the same K = 9 half-widths, placed on the peak**. On `high_snr_curvature`
that certifies **+131.56930589624307 ± 9.3e−13**, which is the Laplace number and both
candidates' answers.

> **This is a finding about R5's own Task 2 oracle, not about a backend**, and it is the one
> thing in this document that would have inverted a verdict. It belongs to Task 10's carry-over
> list: `tests/dispatch/residual_oracle.py`'s certificate has three conditions and needs a
> fourth. Until it has one, `oracle_joint` may certify a value that missed the mass, and the
> two fixtures where that is most likely are the ones with a small `sigma`.

**The span the oracle grades on is NOT the domain handed to jaxns.** A peak-placed span tells
the reader where the mass is; handing it to a backend tells the backend. blackjax draws from
the prior and is told nothing, so jaxns's box is always the prior-shaped rung — the declared or
K-rule span — never the peak one. Measured on `high_snr_curvature`: with the peak span as its
box jaxns integrates a domain `1e-5` wide around the mass and spends 131 k evaluations; with
the prior-shaped span it must find a `6e-7` peak inside a span 18 wide, spends **613 k**, and
lands at `+131.613 ± 0.272`. The second is the comparison; the first would have been a gift.

**17 of 21 correctness cells are fillable. 4 are WITHHELD, and they say why.**

| fixture | axes | why |
|---|---:|---|
| `nan_at_negative_probes` | 1 | the integrand is `nan` at 19 of 201 grid points — a property of the model's domain, and one a backend meets too |
| `bright_and_faint_pair` | 2 | the next refinement needs `12801² = 1.64e8` points, above the `5e7` budget, on both the rule span and the peak span |
| `three_latent_chain` | 3 | the next refinement needs `513³ = 1.35e8` points |
| `undeclared_quartet` | 4 | the next refinement needs `129⁴ = 2.77e8` points |

Plan §11.4 and §5.2a: **these cells are WITHHELD and never blank**, because a blank reads as a
pass. Nothing in this document scores either candidate on a row whose oracle abstained — and
**three of the four are rows where the two candidates disagree by more than either claims**,
which is why that matters rather than being a formality. `bright_and_faint_pair`: −33.51 ± 0.09
against −34.29 ± 0.25, a gap of 0.79 nats. `nan_at_negative_probes`: −1.515 ± 0.020 against
−0.817 ± 0.108, a gap of 0.70. `three_latent_chain`: −5.110 ± 0.016 against −4.862 ± 0.154. On
the fourth, `undeclared_quartet`, they agree to 0.0006. With no oracle, nothing here can say
which is right on any of them.

> **Write-back to plan §3.2 and §5.2a (red line 11).** §3.2 records "the boundary is FOUR",
> measured on the oracle-dimension sweep's own model. Over the fixtures the bake-off actually
> runs, that is not where the boundary is: **`three_latent_chain` at three axes is already
> ungradeable at the declared budget, and `undeclared_quartet` at four is too — with the
> suite's own posterior-placed spans.** The boundary is a property of the integrand and the
> dimension jointly, not of the dimension. §5.2a's instruction survives unchanged (decide per
> fixture, filled or withheld, and say which); its stated threshold does not.

### 5.1 The band the gap is read against

A row's gap is `log Z − oracle`, and it is read against

```
band = 3 · σ_reported  +  oracle bound
```

The **form** is derived: each side contributes the error it states about itself, which is the
same shape D107 and D111 take. The **level** — the factor 3 — is *borrowed*: three standard
deviations is statistics' conventional two-sided ≈0.27 % level, declared here in advance and
not derived from anything measured in this repository. That is D104's provenance, one column
over.

**The verdict does not turn on the factor, and that is checkable rather than asserted.** The
smallest factor at which every gradeable row of both candidates still passes is **1.918** —
jaxns on `indirect_ancestor`. So anything from 1.92 upward gives the same condition-3 outcome,
and the choice of 3 buys margin rather than a result. Below 1.92 rows begin to fall out for
both candidates, jaxns first.

**It is a comparison in a document, not a threshold in `src/`.** On the precedent the R5
plan's own D-numbering write-back sets — D107 governs a comparison in a test and is registered
nowhere, D111 likewise — it takes no registry entry and no D-number, and **Task 5 consumes
neither D112 nor anything else.** If Task 9's repeated-run stability factor lands in shipped
source, that one is D109's kind and needs D112 with a boundary grid and a fast-layer cell;
this ruling does not transfer to it.

### 5.1a What the correctness column is actually grading, per candidate

The two columns are **not the same measurement**, and saying so is not a caveat but the
finding §1.5 condition 5 exists to surface.

* For **jaxns**, `log Z` and `log_Z_uncert` come out of `NestedSamplerResults`. The column
  grades jaxns.
* For **blackjax**, the sampler returns dead particles and nothing else. The evidence, its
  error bar, and the rule that decided when to stop are all written in `run_blackjax` — the
  Skilling volume estimate, an ensemble over 100 simulated volume paths from
  `blackjax.ns.utils.log_weights`, and a `dlogZ < 1e-3` remaining-evidence test. **The column
  grades bayesmith's own estimator running on blackjax's samples.**

So a blackjax row that lands inside the band is evidence about a pairing, not about a library,
and every one of those three pieces would be bayesmith code with its own mutation coverage.
That lands on §1.5 condition 5 in the direction opposite to blackjax's small dependency count,
and §5.4b requires both to be scored rather than one.

### 5.2 The rows

One seed (0), the frozen knobs, x64, 21 fixtures. `—` in a `z` column means the oracle is
WITHHELD for that row and nothing is scored there.

| fixture | oracle | blackjax `log Z` ± σ | z | jaxns `log Z` ± σ | z |
|---|---:|---:|---:|---:|---:|
| `affine_only_at_zero` | −5.6337 | −5.6276 ± 0.0679 | +0.09 | −5.9211 ± 0.1740 | −1.65 |
| `bilinear_pair` | −4.4035 | −4.3305 ± 0.0772 | +0.95 | −4.2690 ± 0.2120 | +0.63 |
| `bright_and_faint_channels` | −33.2885 | −33.2993 ± 0.0932 | −0.12 | −33.1762 ± 0.1961 | +0.57 |
| `bright_and_faint_observations` | −253.0350 | −253.0650 ± 0.0840 | −0.35 | −252.9990 ± 0.1983 | +0.18 |
| `bright_and_faint_pair` | **WITHHELD** | −33.5112 ± 0.0869 | — | −34.2930 ± 0.2470 | — |
| `cauchy_residual_pair` | −2.2218 | −2.2066 ± 0.0434 | +0.35 | −2.1268 ± 0.1748 | +0.54 |
| `cubic_tail` | −5.9755 | −5.9736 ± 0.0507 | +0.04 | −5.7822 ± 0.1258 | +1.54 |
| `diamond_ancestor` | −5.9656 | −5.9641 ± 0.0067 | +0.23 | −5.8048 ± 0.0874 | +1.84 |
| `faint_alone` | +14.9767 | +14.9698 ± 0.0825 | −0.08 | +14.9250 ± 0.1992 | −0.26 |
| `high_snr_curvature` | +131.5690 | +131.6320 ± 0.1370 | +0.46 | +131.6870 ± 0.2803 | +0.42 |
| `indirect_ancestor` | −6.5116 | −6.5144 ± 0.0053 | −0.53 | −6.3426 ± 0.0881 | +1.92 |
| `mixture_prior_residual` | −13.3326 | −13.3675 ± 0.0551 | −0.63 | −13.2633 ± 0.0840 | +0.83 |
| `nan_at_negative_probes` | **WITHHELD** | −1.5152 ± 0.0205 | — | −0.8174 ± 0.1079 | — |
| `non_gaussian_observed_node` | −4.7653 | −4.8015 ± 0.0546 | −0.66 | −4.5709 ± 0.1436 | +1.35 |
| `orphaned_child_latent` | −5.8227 | −5.8646 ± 0.0574 | −0.73 | −5.7685 ± 0.1722 | +0.31 |
| `overflowing_outside_latent` | −17.4694 | **DEGENERATE** | — | −17.3913 ± 0.0601 | +1.30 |
| `quadratic_claim` | −6.4667 | −6.4993 ± 0.0620 | −0.53 | −6.4388 ± 0.1513 | +0.18 |
| `shared_ancestor` | −8.9643 | −8.9651 ± 0.0055 | −0.14 | −8.8870 ± 0.0581 | +1.33 |
| `student_t_likelihood` | −9.9338 | −10.0158 ± 0.0547 | −1.50 | −9.9253 ± 0.1421 | +0.06 |
| `three_latent_chain` | **WITHHELD** | −5.1096 ± 0.0162 | — | −4.8624 ± 0.1535 | — |
| `undeclared_quartet` | **WITHHELD** | −5.9800 ± 0.0621 | — | −5.9794 ± 0.1809 | — |

**Every gradeable row of both candidates is inside the band** — 16 of 16 for blackjax and 17 of
17 for jaxns. Max |z| is 1.50 and 1.92 against a band of 3σ plus the oracle's own bound, so
§1.5 condition 3's correctness clause is met by both on every row it can be asked about.

Per-candidate totals over the run: blackjax spends **146 267 – 856 895** evaluations (median
253 051), 2.49 s median wall (max 4.50), 732 MB median peak RSS (max 902). jaxns spends
**113 699 – 740 820** (median 264 172), 1.93 s median wall (max 2.21), 578 MB median RSS (max
609).

#### Three things the row-by-row view hides

**1. `overflowing_outside_latent` — blackjax does not answer it, and jaxns does.** Its Cauchy
prior draws reach `|z| ~ 1e6`, where the collapsed evidence density overflows: **800 of 816
initial live particles have a `nan` log-likelihood**, the remaining 8 sit at `−8.8e+125`, and
blackjax reports none of it. The consequence is not a wrong number but a run that cannot end —
every termination rule that reads `max(loglikelihood)` reads `nan`, so the comparison is False
forever. Measured before the probe learned to refuse it: the loop ran to its 100 000-iteration
cap and `finalise` then sat in `CompileCpuExecutableInternal` for seven minutes without
returning. jaxns meets none of this, because its box is `z ∈ (−100, 100)` and the overflow is
outside it. **The bounded domain that costs jaxns everywhere else is what carries it here**,
and this is §0.6's first retention condition met by measurement: an admitted fixture the other
candidate cannot answer.

**2. At seed 0 the sign of the gap is not symmetric between them — and §5.3 does not confirm
it.** Over the graded rows blackjax's z is positive on **6 of 16** (mean −0.198, median |z|
0.41) and jaxns's on **15 of 17** (mean +0.653, median |z| 0.63). A two-sided sign test on
those counts gives p ≈ 0.80 and p ≈ 0.0023.

**That number is reported here and it is not the finding, because the seed sweep contradicts
it.** Every row above is one seed, so each z carries the run's scatter as well as any bias, and
§5.3 — five seeds on four fixtures — puts jaxns's mean within **0.05 standard errors** of the
oracle on all three of its gradeable fixtures. `diamond_ancestor` is the clearest case: seed 0
gave jaxns z = +1.84, the largest positive gap in the table, and the five-seed mean is
−5.96503 against an oracle of −5.96561, a gap of +0.0006. Seed 0 was an unlucky draw.

So the honest statement is the weaker one: **at one seed per fixture, jaxns's gaps run
predominantly positive; over five seeds on the four fixtures where that was tested, neither
candidate shows a detectable bias.** Settling which is right needs the seed sweep over all 17
gradeable rows, which this task did not run and §11 records as not covered. §0.18 named bias as
a deciding axis; on this evidence it does not decide.

**3. Where the oracle is withheld, the two do not agree.** `bright_and_faint_pair`: −33.51 ±
0.09 against −34.29 ± 0.25, a gap of 0.78, three times jaxns's error and nine times blackjax's.
`nan_at_negative_probes`: −1.515 ± 0.020 against −0.817 ± 0.108, a gap of 0.70. `three_latent_
chain`: −5.110 ± 0.016 against −4.862 ± 0.154. On `undeclared_quartet` alone they agree to
0.0006. Red line 5 applies in both directions: agreement is tier 3 and does not certify, and
disagreement names no culprit.

### 5.3 Repeated runs — five seeds on four declared rows

Seeds `(0, 1, 2, 3, 4)`, declared before the runs. The four rows are the multimodal fixture,
the heavy-tailed one, the four-axis one and a Gaussian control — chosen for what they are, not
for what they scored.

| fixture | candidate | mean | spread | reported σ | spread / σ | mean − oracle |
|---|---|---:|---:|---:|---:|---:|
| `diamond_ancestor` | blackjax | −5.96984 | 0.00736 | 0.00646 | 1.14 | −0.0042 |
| `diamond_ancestor` | jaxns | −5.96503 | 0.1076 | 0.0918 | 1.17 | **+0.0006** |
| `cauchy_residual_pair` | blackjax | −2.21295 | 0.01713 | 0.03762 | 0.46 | +0.0088 |
| `cauchy_residual_pair` | jaxns | −2.22773 | 0.2873 | 0.1756 | 1.64 | −0.0059 |
| `mixture_prior_residual` | blackjax | −13.3399 | 0.02372 | 0.05059 | 0.47 | −0.0073 |
| `mixture_prior_residual` | jaxns | −13.3370 | 0.04802 | 0.08601 | 0.56 | −0.0044 |
| `undeclared_quartet` | blackjax | −5.96407 | 0.03376 | 0.06246 | 0.54 | oracle WITHHELD |
| `undeclared_quartet` | jaxns | −5.97139 | 0.1719 | 0.1799 | 0.96 | oracle WITHHELD |

**Bias: neither candidate shows one.** Every mean is within 1.3 standard errors of the oracle,
and jaxns's are within 0.05. The multimodal row matters most and both find it: blackjax is
0.0073 low and jaxns 0.0044 low against a constructed closed form, so neither has collapsed to
one component — which is what §0.16 named as the measurement most likely to decide and it does
not decide.

**Stability: no ratio here is distinguishable from 1.** The probe prints "spread EXCEEDS the
reported bar" for three cells, and that is a raw comparison rather than a test. With `n = 5`
the sample standard deviation has four degrees of freedom, so its 95 % interval spans roughly
0.6 to 2.9 times the true value — **1.14, 1.17 and even 1.64 all sit inside it.** §0.8's
asymmetry is the right instrument and this `n` cannot drive it. **That is what §0.10 and D112
are for and it belongs to Task 9**: a stability gate needs a declared factor, a declared `n`
and a declared false-positive rate before a ratio means anything.

Recorded and not scored: at the same budget jaxns's spread is 4–17× blackjax's on three of the
four rows.

---

## 6. §1.5's six conditions, scored one at a time

Each row is scored against **§0.5's countable form**, not against §1.5's prose, because that is
the instrument the plan froze. §5.3's rule applies: a condition answered by argument rather
than by a run is scored **not met**.

| # | condition | blackjax | jaxns |
|---|---|---|---|
| 1 | generality over the admitted family | **NOT MET** | MET |
| 2 | JAX / JIT / PyTree / x64 / device / RNG | MET | MET |
| 3 | correctness, compile, runtime, memory (+ §3.5's four more) | MET | MET |
| 4 | maintenance, versions, failure behaviour, termination, diagnostics | MET | MET |
| 5 | adapter thinness | **NOT MET** | **NOT MET** |
| 6 | refusal when absent, contract test, independent oracle | MET | MET |

### Condition 1 — generality

*Countable form: runs every admitted fixture in classes (b) and (c) plus the multimodal and the
heavy-tailed one, without per-fixture special-casing; artefact = one row per fixture,
PASS/FAIL/CRASH.*

**jaxns answers 21 of 21. blackjax answers 20** — `overflowing_outside_latent` is DEGENERATE
(§5.2). Neither needed per-fixture special-casing: blackjax's `num_inner_steps` is derived from
the residual dimension by its own documented `max(5, 2·dim)`, and jaxns's box comes from the
same span rule everywhere.

### Condition 2 — JAX compatibility

*Countable form: runs under `jax_enable_x64(True)`; accepts a JAX callable; takes a
`jax.random.key`; the same key gives the same `log Z` bitwise on a repeat.*

Both met, on runs. Both accept **`jax.random.key` and `jax.random.PRNGKey`**, and both repeat
bitwise on the same key — blackjax `−10.0158` twice, jaxns `−9.92533` twice.

Two boundary costs recorded beside it, one each: **jaxns writes `jax_enable_x64` process-
globally at import**, against this package's rule that `src/` never touches `jax.config`; and
**blackjax runs at float32 and returns a finite, plausible number**, so bayesmith enforces the
x64 gate itself because blackjax will not.

### Condition 3 — correctness, and §3.5's eight axes rather than §1.5's four

Plan §0.18 rules that condition 3 expands to §3.5's eight axes, not §1.5's four, because the
four extra ones are where the two were most likely to differ. Measured, they are not where the
two differ: the deciding evidence turned out to be conditions 1 and 4.

| axis | blackjax | jaxns |
|---|---|---|
| correctness | every gradeable row in band; max \|z\| **1.50** | every gradeable row in band; max \|z\| **1.92** |
| bias | none detectable over 5 seeds × 3 gradeable rows | none detectable; means within 0.05 se |
| uncertainty | halves as the budget quadruples; spread/σ 0.46–1.14 | halves as the budget quadruples; spread/σ 0.56–1.64 |
| termination | **none of its own** — the rule is bayesmith's | its own, an 11-field condition, 12-bit reason |
| multimodal | mean 0.0073 low on the constructed closed form | 0.0044 low |
| JIT / compile | first call 0.47 s = compile + one NS step | first call ≈1.0 s = compile + the whole run |
| memory | peak RSS median **732 MB**, max 902 | median **578 MB**, max 609 |
| API stability | stable across this work | `TerminationCondition()` disables every rule; its docstring's `dlogZ` default is not the code's |

Both MET. The compile column is **not the same quantity** for the two — jaxns compiles the
entire run, blackjax one step — and is reported as two different things rather than as one
comparison.

### Condition 4 — maintenance, read on 2026-09-06

*Countable form: last release date, release cadence and open-issue count recorded with the date
read; every termination signal maps to a `TerminationReason` member; `log Z` uncertainty is a
real estimate, not a constant.*

| | last release | releases in 365 days | open issues | last repo push |
|---|---|---:|---:|---|
| blackjax 1.6.2 | 2026-07-16 | **6** | 41 | 2026-08-31 |
| jaxns 2.6.9 | **2025-08-03** | **0** | 9 | 2026-09-04 |
| `tfp-nightly` (jaxns's) | 2026-09-06 | **354** | — | — |

Both MET **as the countable form is written** — the form asks for a record, not for a bar. The
facts it records are not symmetric and the asymmetry is the largest in this document:
**jaxns's released package has not moved in thirteen months while its own hard dependency
ships a new build every day.** Its repository is active — pushed two days ago — so this is not
abandonment; it is that the artefact a wheel depends on is a year old and pinned to something
unpinnable. §11.7 names adapter drift and this project has spent four release tags on
non-reproducible numerical environments, which is what makes that fact load-bearing rather
than trivia. **Applying a bar to it is an owner decision and §9 states it.**

Both `log Z` uncertainties are real estimates (§4). Termination signals map, with one
exception: blackjax's degenerate state has **no honest `TerminationReason` member**, and red
line 4 forbids adding one. The right reading is that it is not a termination at all — a
collapsed density that is `nan` over the prior is a statement about the *problem*, so it
belongs in a `Refusal` premise rather than a `TerminationRecord`. Task 6 or Task 7 rules.

### Condition 5 — adapter thinness

*Countable form: adapter is one module; no backend type appears in any artifact field; measured
as a line count and an import-direction assertion. Artefact: `tests/test_layering.py` + a
public-API assertion.*

**NOT MET for both, and for a reason that is about neither of them**: the artefact requires an
adapter, Task 6 writes the adapter, and Task 6 is gated on this verdict. §14 states the
circularity. What was measured instead, by running it: the blackjax driver is 164 lines (125
non-comment) and owns the NS loop, the termination rule, `log Z` and its uncertainty; the jaxns
driver is 123 lines (89) and owns the bounded box and the prior-into-likelihood
reparametrisation. Design §7.3 question 7 — *can the compiled problem supply the target
representation losslessly* — is answered **yes** for blackjax and **no** for jaxns (§2).

### Condition 6 — refusal when absent, and an independent oracle

*Countable form: with the package uninstalled, an `EvidenceTask` returns a capability `Refusal`
and the core suite is green; the independent oracle is §0.4's quadrature.*

Both MET, and neither by anything this task built: Task 4 shipped the refusal built-and-run
(`tests/dispatch/test_backend_absent.py`), and absence is this repository's default state — the
probe reports ABSENT for both candidates when run in `.venv`. Task 2 shipped the oracle. The
core suite is green with both absent, measured as this commit's own gate: **3692 tests, 0
failures, 0 errors, 4 skipped, exit 0** (§10).

Recorded against it: **the oracle reaches 17 of the 21 admitted rows** (§5.0), so the
independent-oracle half of this condition holds over a domain that is smaller than the admitted
class, and §5.0's certificate defect had to be repaired before even that was true.

---

## 7. §7.3's nine questions, answered in this table rather than separately

Design §7.3 asks nine questions before a backend is adopted. §0.5 rules that they are answered
in the same table as §1.5's six, not in a second pass. Where they are answered is recorded
here so that a reader can find each one:

| # | §7.3 question | where |
|---|---|---|
| 1 | 它补足了什么 bayesmith 自己不应重写的能力？ | §2 — the nested-sampling loop; and for blackjax, *only* that |
| 2 | 能否服从现有 Task/Plan/Result/Refusal 协议？ | §5's `CompiledEvidenceProblem` contract; §9's termination map |
| 3 | optional dependency 缺失时是否优雅退化？ | §6 condition 6 — Task 4's built-and-run absence path |
| 4 | 是否有独立 oracle 或 cross-check？ | §5.0 — `oracle_joint`, tier 1, and the 4 rows it does not reach |
| 5 | backend-specific object 泄漏到公共 API 的范围是否最小？ | §6 condition 5 |
| 6 | 升级 backend 时如何检测语义漂移？ | §8 — the version and release record, and `tfp_nightly`'s daily move |
| 7 | 目标表示能否由 CompiledProblem 无损提供，而不让 backend 重新解释 Graph？ | §2 — **and this is where the two candidates part** |
| 8 | 第二个 adapter 是否带来独立验证或新的适用域？ | §9 — §0.6's two conditions |
| 9 | 是否通过 §1.5 的六道门槛？ | §6 |

**Question 7 is answered by a run and the answer is not the same for the two candidates.**
BlackJAX's target representation is `(log_prior, log_likelihood)` plus a prior draw, which is
three of `CompiledEvidenceProblem`'s own fields. JAXNS's is a quantile map, which the compiled
problem cannot supply losslessly — so consuming it means either a new field on the compiled
problem or the box reparametrisation of §2, and the second is bayesmith re-expressing the
prior for the backend's convenience.

---

## 8. §5.4b — the costs that are not correctness

§1.5's six conditions are not all about the number. Three costs are recorded here because a
verdict that names only accuracy is a verdict a later reader will have to re-derive.

### 8.1 Dependency footprint

Measured on 2026-09-06 in throwaway environments, against a base of exactly this package's
declared runtime dependencies — `jax`, `equinox`, `numpy`, `numpyro` — which resolves to **13
packages**. `jax` stays at `0.11.1` in every case.

| | packages added | what |
|---|---:|---|
| blackjax 1.6.2 | **3** | `blackjax`, `absl-py`, `optax` |
| jaxns 2.6.9 | **19** | `jaxns`, `tfp-nightly`, `dm-tree`, `matplotlib`, `pillow`, `contourpy`, `kiwisolver`, `fonttools`, `cycler`, `pyparsing`, `python-dateutil`, `six`, `attrs`, `cloudpickle`, `decorator`, `gast`, `wrapt`, `packaging`, `absl-py` |

§0.16's Wave D write-back recorded 3 and 16 against a fuller base, and the difference is the
write-back's own point restated: **how many packages an install adds is a property of the base
environment, not of the candidate.** Both numbers are right about their base; neither is a
property of jaxns. What survives a change of base is the composition — jaxns pulls a plotting
stack and an unbounded nightly build into a wheel this project publishes, and blackjax pulls
an optimiser.

`tfp-nightly` moved during this session. The environment every run above used was built at
09:35 on 2026-09-06 and resolved **`0.26.0.dev20260905`**; the index queried at 11:00 the same
morning offered **`0.26.0.dev20260906`**. §0.16 says it "resolves to a same-day build";
measured across 2026-09-04, -05 and -06 it resolves to a build that moves daily, and the build
for the day this evaluation was written appeared while the evaluation was being written.
**2976 releases exist, 354 of them in the last 365 days.** Stable
`tensorflow-probability 0.25.0` does not substitute — §0.16 measured the `AttributeError`
`import jaxns` then raises.

### 8.2 What adopting each one would cost R4's shipped behaviour

`jaxns/internals/mixed_precision.py` runs `jax.config.update('jax_enable_x64', True)` at module
scope, behind a `UserWarning`. This repository's rule is the opposite — `src/` never touches
`jax.config`, and the caller opens `with jax.enable_x64(True):`. Wave D's write-back to §0.16
narrowed what that costs: the redness follows the **flip**, not the **install**, because
nothing in this package imports either candidate and Task 4's capability probe reads installed
metadata rather than a module body. So declaring the extra costs R4's gate nothing.

**An adapter is a different question, and it is the one Task 6 inherits.** An adapter that
imports `jaxns` at module scope flips `jax_enable_x64` for the whole process the first time
anything imports the adapter, and `dispatch/task.py:1045` decides `evidence_requires_x64` by
outcome (`jnp.result_type(float)`) precisely so that a caller who threw the process-global
switch and one who used the context manager get the same answer. The gate would then stop
firing — not because a number is wrong, but because the caller no longer declared the
precision. §0.16's owner decision stands and is restated in §9.

BlackJAX cuts the other way and it is a cost too: it **runs at float32 and returns a finite,
plausible number**. Nothing refuses. If blackjax is chosen, bayesmith enforces the x64 gate
itself, because blackjax will not.

### 8.3 What the adapter would have to OWN

| | blackjax | jaxns |
|---|---|---|
| the nested-sampling loop | **bayesmith writes it** | jaxns |
| a termination rule | **bayesmith writes it** | jaxns (11-field `TerminationCondition`) |
| `log Z` | **bayesmith writes it** | jaxns |
| `log Z` uncertainty | **bayesmith writes it**, from the volume ensemble `blackjax.ns.utils.log_weights` does supply | jaxns |
| the prior representation | the compiled problem's own two callables | **bayesmith must supply a bounded box and fold the prior into the likelihood** |
| weighted posterior draws | `finalise()` + `sample()` — never the live particles, which collapse to the highest-likelihood mode | jaxns |
| a usable `finalise` | **bayesmith writes it** — the shipped one compiles one XLA operand per NS step and did not return on a 100 000-step run | jaxns |
| refusing a `nan` likelihood | **bayesmith writes it** — blackjax accepts one silently and no termination rule can then fire | not reached: the box keeps it out of the overflow |

Each row in the left column is bayesmith code with its own mutation coverage, and §1.5
condition 5 counts them against blackjax in the direction opposite to its dependency count.
Each row in the right column is a bayesmith object jaxns needs that the compiled problem does
not carry, and design §7.3 question 7 counts that against jaxns.

---

## 9. §5.5 — the verdict

### No candidate passed.

That is §0.5's third branch and it closes R5 legitimately: §8 R5's first completion gate asks
for *"backend 决策有可复现 benchmark、oracle 结果和明确适用域，而不是预设偏好"* — a reproducible
decision, not a backend. **R5 ships the compiler, the eligibility, the oracle and the refusal
without a production adapter unless the owner rules otherwise below.**

The two candidates did not fail the same way, and the difference is the useful part:

* **BlackJAX fails condition 1 and condition 5.** It does not answer `overflowing_outside_latent`
  — 800 of 816 live particles initialise at `nan` and blackjax does not say so — and condition
  5's artefact does not exist yet.
* **JAXNS fails condition 5 only.** It answers all 21 rows, is in band on all 17 gradeable
  ones, shows no detectable bias over the seeded subset, and reports a `log Z` uncertainty that
  scales correctly.

**Condition 5 is the reason both fail, and it is a defect in the instrument rather than in
either candidate** (§14). Read that off before reading anything else into "no candidate
passed": strike it and the sentence becomes *blackjax fails condition 1 and jaxns fails
nothing*, which is a materially different page. It is left standing because §0.5 froze the six
conditions and §5.3 says an argued condition is not met, and relaxing an instrument to reach a
result is what this evaluation exists not to do.

### Two owner decisions, and what each implies

**Decision A — is condition 5 scorable before an adapter exists?** §14 shows it is not: its
artefact is `tests/test_layering.py` plus a public-API assertion, and Task 6 writes the adapter
Task 6 is gated on. If the owner rules condition 5 **deferred to Task 6** rather than failed:

* jaxns passes all five scorable conditions.
* blackjax still fails condition 1.

**Decision B — is `overflowing_outside_latent`'s degeneracy a backend failure or a fixture
pathology?** The fixture's Cauchy prior reaches `|z| ~ 1e6`, where the *collapsed* density
overflows; blackjax meets it because it samples the declared prior, and jaxns does not because
its domain is bounded and it therefore answers the truncated integral — which is also the one
the oracle grades. If the owner rules it a **fixture pathology** and excludes the row:

* blackjax passes conditions 1–4 and 6.
* Both then stand or fall together on Decision A.

**Neither decision makes both candidates pass, and neither makes both fail.** They are stated
as decisions rather than resolved here because §0.5 froze the six conditions as the instrument
and neither question is a measurement.

**The four corners, so the owner can read the consequence off rather than derive it:**

| | B: backend failure | B: fixture pathology |
|---|---|---|
| **A: condition 5 fails** | neither passes | neither passes |
| **A: condition 5 deferred** | **jaxns passes; blackjax fails 1** | **both pass** |

The bottom-right corner is the only one that admits blackjax, and it is also the only one that
would put §0.6's second-backend rule into play — where §9's last subsection has already found
retention condition 1 met in jaxns's favour and condition 2 not shown.

### The applicable domain, if an adapter is ever written

G1 requires this stated as a structure class and a dimension bound, and the runs support:

* **Structure**: plan §0.3's classes (b) and (c) — exact-plus-residual under `gcr`, and
  all-residual. `gcr+snis` and `gcr+mh` are refused by name and were not evaluated.
* **Gradeable dimension**: **one and two total latent axes**, plus the three- and four-axis
  cases only where a peak-placed span certifies. Measured: `three_latent_chain` (3 axes) and
  `undeclared_quartet` (4) have no independent oracle at the declared budget, and
  `bright_and_faint_pair` has none at two. So R5's gradeable domain is **17 of the 21 admitted
  rows**, and outside it a residual evidence would be reported without a tier-1 check.
* **Priors**: any the compiled problem can draw from, for blackjax. For jaxns, any for which a
  bounded box can be declared — which excludes nothing measured here, and costs a truncation
  everywhere.
* **Not covered**: five or more total axes, `nan` regions inside the prior's support, and any
  fixture where the two candidates disagree by more than either claims (three of the four
  WITHHELD rows do).

### §0.6 — the second-backend rule, scored

§0.6 removes the second-placed candidate unless one of exactly two conditions is shown with a
run. **Condition 1 is met, in jaxns's favour**: `overflowing_outside_latent` is an admitted
fixture blackjax does not answer and jaxns does, at z = +1.30 against a certified oracle. That
is the "different applicable domain" clause, measured rather than argued.

Condition 2 — independent cross-check value — is **not** shown. The two disagree on three
WITHHELD rows by more than either claims, which is a detection with no oracle behind it, and
red line 5 says two backends agreeing is tier 3 and does not substitute for tier 1. Nothing
here demonstrates a fault one catches and the other does not.


## 10. Provenance

Every number in this document comes from one run of one file. The file is
`docs/probes/probe_37_backend_bakeoff.py` at blob `1f97305bbfecf83849c91a625c80278df1ea34a7`, and the run is
`runs/probe37-20260906T111626/` — `log` for the printed sections, `bakeoff.json` for every cell's full record.
`runs/` is gitignored, so the JSON does not travel with the commit; the probe does, and it
reproduces the run.

| item | value |
|---|---|
| `sha` (before) | `4b690f79ab37bdc8bc78a88939fa2b56b7616667` |
| `sha_after` | `4b690f79ab37bdc8bc78a88939fa2b56b7616667` |
| `tree_before` | `46558a4c179893912c3950391b9bd9355afbdd78` |
| `tree_after` | `46558a4c179893912c3950391b9bd9355afbdd78` |
| `git merge-base --is-ancestor` | `yes` |
| suite | exit `0`; `tests=3692 failures=0 errors=0 skipped=4` |
| probe blob | `1f97305bbfecf83849c91a625c80278df1ea34a7` |
| run directory | `runs/probe37-20260906T111626` |

**The gate ran on the staged tree, and one thing changed after it**: this provenance table,
which cannot record a run that has not happened. `tree_before` and `tree_after` are `46558a4c`
and the committed tree is one paragraph later; `git diff` between them is this section. The
earlier discipline note in §11 applies to the earlier attempt, where the trees were HEAD's
because nothing was staged; here they are the work's own.

**Two runs were discarded before this one and the reasons are on the record**, because a
verdict that quietly re-ran until it liked the answer is the thing this page exists not to be.
The first passed `jaxns.TerminationCondition()`, which disables every stopping rule, so all 21
rows terminated on structural exhaustion — a property of the call site read as one of the
library. The second was measured with the wrong `ruff`; the project's own binary found a
late-binding closure in the probe, and fixing it changed the file the numbers came from.
Neither discarded run's numbers appear here.

### Environments

| | macOS | Linux |
|---|---|---|
| CPU | Apple M3 Ultra (arm64) | `QEMU TCG CPU version 2.5+`, flags `fma sse4_2 avx avx2`, SIMD `X86_V3`; no AVX-512 |
| BLAS | Accelerate | scipy-openblas 0.3.34, `DYNAMIC_ARCH`, `OPENBLAS_CORETYPE=ZEN` |
| jax / jaxlib | 0.11.1 | 0.11.1 |
| blackjax / jaxns | 1.6.2 / 2.6.9 | 1.6.2 / 2.6.9 |

The Linux side is a `linux/amd64` container under colima. Its flag set matches the AMD EPYC
7763 that `ubuntu-latest` served in one of three measured runs — and **there is no such thing
as "the runner"**: the same pool has also served an EPYC 9V74 and a Xeon 8370C, both with
AVX-512. So this reproduces one machine, not the pool, and it is quoted for **correctness and
termination only**. Under emulation a wall-clock number is a statement about the emulator.

### §5.2b — the Linux re-run, on a declared subset

The whole probe does not finish under QEMU: `census()` compiles all 57 shipped graphs through
`bayesmith.compile`, which is JAX tracing, and fifteen minutes had not got past it. So the
Linux half covers **five fixtures named in advance for what each settles** — the calibration
fixture, the peak-resolution repair, the multimodal one, the heavy-tailed one, and the row
blackjax cannot answer — driven through the probe's `--cell` path, which builds one graph
instead of censusing them all.

**The oracle is bitwise identical on all five.** This is the measurement §5.2b exists for:
macOS numpy uses Accelerate, the Linux wheel uses scipy-openblas, and that difference is what
burned four release tags.

| fixture | span | macOS and Linux |
|---|---|---:|
| `student_t_likelihood` | `rule@9` | −9.933854850401152 ± 7.83e−13 |
| `high_snr_curvature` | **`peak`** | **131.56930589624307 ± 9.32e−13** |
| `mixture_prior_residual` | `suite` | −13.332624149718963 ± 6.10e−14 |
| `cauchy_residual_pair` | `suite` | −2.22179383874746 ± 4.41e−10 |
| `overflowing_outside_latent` | `suite` | −17.4694048411445 ± 4.74e−09 |

The second row is the one that had to hold: **§5.0's peak-resolution repair is not
platform-local.** On OpenBLAS the oracle also rejects the declared span's grid, also falls to
the peak rung, and certifies the same value to the last digit. Had it not, the correctness
column would have been an artefact of one BLAS.

**The sampler cells reproduce too, and "bitwise" is the wrong word for it.** Ten cells ran on
Linux against the same seeds and the same frozen knobs — the nine that return a number, and the
degenerate one.

| | identical | worst relative difference |
|---|---:|---|
| evaluation count | **9 of 9** | — |
| `log Z` | 7 of 9 | **4.1e−16** (`overflowing_outside_latent`, jaxns) |
| reported σ | 5 of 9 | **9.8e−13** (same cell) |

**Every evaluation count is identical**, so both backends took the same path through the same
number of likelihood calls on both platforms. Every `log Z` is identical or within two ULP. The
error bars differ on four cells, the largest by `9.8e−13` relative — in the `std` over the
volume ensemble and in jaxns's own `log_Z_uncert`, both reductions, which is exactly where
Accelerate and OpenBLAS have differed in this repository before.

The largest difference anywhere is **nine orders of magnitude below the band it feeds** (§5.1),
so no row changes side and no condition changes score. It is written out rather than rounded to
"identical" because four cells are not identical and the next reader should not have to
rediscover that.

**The condition-1 failure reproduces exactly.** blackjax on `overflowing_outside_latent` is
DEGENERATE on Linux too, with the same **800 of 816** non-finite live particles and the same 816
evaluations, and jaxns answers it at −17.39127624503766 ± 0.0601 against the same certified
oracle. The one row that decides condition 1 is not a macOS artefact.


## 11. What this evaluation does not cover

Recorded rather than left to be discovered, because §11.4's failure is a number read as
stronger than the run behind it.

* **Four of 21 correctness cells are WITHHELD** (§5.0). Nothing here says either candidate is
  right or wrong on `nan_at_negative_probes`, `bright_and_faint_pair`, `three_latent_chain` or
  `undeclared_quartet` — and on three of the four the two disagree by more than either claims.
* **The budget is equal on the calibration fixture and nowhere else.** §0.7's protocol tunes
  each candidate's knobs to hit 200 000 evaluations on `student_t_likelihood` and then freezes
  them, so every other row spends what its own termination rule requires. The per-row spends
  are in the table: blackjax spends 146 k to 857 k and jaxns 114 k to 741 k, so the "equal
  budget" is equal at one row and within a factor of six elsewhere.
* **jaxns is measured through the box reparametrisation and no other route**, because no other
  generic one exists (§2). A finding about that route is a finding about the only way bayesmith
  could drive jaxns from a compiled problem today; it is not a finding about jaxns given a
  prior it can represent natively.
* **blackjax is measured through an evidence estimator, a termination rule and an error bar
  that bayesmith wrote** (§5.1a). Those three are not blackjax's, and a different three would
  give different rows.
* **Timing is macOS/Accelerate on one machine.** The Linux re-run (§10) grades correctness and
  termination; under QEMU emulation a wall-clock number is a statement about the emulator, and
  it is not quoted as one about a backend.
* **The suite gate's `tree_before` / `tree_after` are blind here.** `git write-tree` reports
  the INDEX, and none of this task's work was staged while the suite ran, so both hashes are
  HEAD's tree whatever the working tree held. The handoff records that pair as blind to a
  commit made mid-run; this is the stronger case, blind to everything unstaged. What carries
  the claim is `sha` equal to `sha_after`, the ancestry check, and the discipline of not
  editing during the run.
* **One seed per row in §5.2, five seeds on four declared rows in §5.3.** A per-row bias
  statement over one seed is a per-row scatter statement; only the seeded subset supports the
  word, and it covers four rows of 21.
* **`n = 5` cannot drive §0.8's stability gate.** The spread-to-bar ratios in §5.3 range 0.46
  to 1.64 and none is distinguishable from 1 at four degrees of freedom. Task 9 declares the
  factor, the `n` and the false-positive rate; this task measured the quantity and did not
  gate on it.
* **One unexercised branch in the probe, declared rather than discovered.** If every live
  particle's log-likelihood were `-inf` — allowed, since `-inf` is not degenerate — the
  termination test would compare `-inf` against the running evidence and stop at the first
  iteration with a meaningless `log Z`. No row reached it: the smallest evaluation count in the
  table is 113 699 against a `num_live` of 816 and 304, so every run took thousands of
  iterations. It is a hole in the probe and it is named here because a hole the author found
  and left is worth more than one a reader finds first.

---

## 12. Reading the probe's output yourself

```bash
# this repository's own environment: both candidates ABSENT, and the probe says so
.venv/bin/python docs/probes/probe_37_backend_bakeoff.py

# an environment carrying both, which is what this page scores
uv venv /tmp/env --python 3.12
uv pip install --python /tmp/env/bin/python "jax>=0.5" "equinox>=0.13" numpy \
  "numpyro>=0.15" "blackjax>=1.6" "jaxns>=2.6" pytest hypothesis
uv pip install --python /tmp/env/bin/python --no-deps -e .
PYTHONPATH=$PWD /tmp/env/bin/python docs/probes/probe_37_backend_bakeoff.py \
  --out /tmp/bakeoff.json
```

`--sections` takes a comma-separated list; `0` is the oracle table, which every other section
depends on and which therefore always runs.

**Under emulation, drive cells rather than sections.** `census()` compiles all 57 shipped
graphs through `bayesmith.compile`, which is JAX tracing, and fifteen minutes of QEMU had not
got past it. `--cell` takes the classification in the request and builds one graph, so the
Linux subset of §10 runs that way:

```bash
docker run --rm --platform linux/amd64 -e OPENBLAS_CORETYPE=ZEN \
  -v "$PWD":/src:ro -w /tmp <image> bash -lc \
  'pip install --quiet --no-deps /src && PYTHONPATH=/src python /src/docs/probes/probe_37_backend_bakeoff.py --cell "$REQUEST"'
```

The `-v` source must be a path colima shares — under `$HOME`, not `/private/tmp`, which is
how the first attempt failed with `can't open file`.

---

## 13. Carry-over for other tasks

Each of these was found by Task 5 and belongs to someone else. Named here so the finder is not
also the fixer, which is how a bake-off turns into a refactor.

| finding | owner |
|---|---|
| `oracle_joint`'s certificate can pass on a value whose grid missed the peak (§5.0). It has three conditions and needs a fourth. | Task 10, and `tests/dispatch/residual_oracle.py` |
| R5's route (b) does not compile: `collapse_graph` returns a `ReducedGraph` whose `.latents` raises, and `compile_evidence_problem` reads it on its first line. `as_graph` is the documented route for an evidence-aware consumer; the guard's message names three safe routes and R5 makes that list incomplete. | Task 6 (plan §6.2a) |
| plan §3.2's "the boundary is FOUR" is a statement about one sweep model; three axes is already ungradeable on a shipped fixture. | Task 10, and `docs/residual-evidence.md` when it exists |
| §0.16's owner decision on jaxns's process-global x64 write attaches to the ADAPTER, not to the extra. | Task 6, and only if jaxns is chosen |
| `jaxns.TerminationCondition()` disables every stopping rule; the real defaults live in `jaxns/public.py:161` and apply only when `term_cond is None`. Its own docstring names a `dlogZ` default the code does not use. | recorded here; a jaxns adapter must pass `None`, never the constructor |
| blackjax accepts a `nan` log-likelihood without a word, and every termination rule that reads `max(loglikelihood)` then reads `nan` and never fires. A blackjax adapter must check the initial live set. | Task 6, and only if blackjax is chosen |
| `blackjax.ns.utils.finalise` concatenates with one XLA operand per NS step. On a 100 000-step run the compile did not return in seven minutes; concatenating on the host costs milliseconds and gives the same arrays. | Task 6 |
| **`CLAUDE.md`'s lint recipe names `ruff`, and a bare `ruff` on this machine is not the project's.** `/opt/homebrew/Caskroom/miniconda/base/bin/ruff` is **0.15.12** and reports **39 errors** on `src/ tests/` at a clean HEAD; `.venv/bin/ruff` is **0.16.4** and reports none. Same family as the ruff cache one file over: a result that cannot distinguish "the code is dirty" from "a different checker ran". The recipe should say `.venv/bin/ruff`. | Task 10, and `CLAUDE.md` + `AGENTS.md` in one commit (red line 12) |

---

## 14. A circularity in §1.5 condition 5, named rather than worked around

§0.5's countable form of condition 5 is *"adapter is one module; no backend type appears in any
artifact field; measured as a line count and an import-direction assertion"*, and its artefact
is **`tests/test_layering.py` + a public-API assertion**. Both require an adapter to exist.
Task 6 writes the adapter, and Task 6 does not start until the owner has read this verdict.

**So condition 5's stated artefact cannot be produced by the task that is required to score
it**, and §5.3's rule — a condition answered by argument rather than by a run is scored **not
met** — applies to both candidates equally and for a reason that is about neither of them.

What this evaluation does instead, and what it does not claim: §8.3 measures, by running them,
what a driver for each backend had to contain and what each backend does not supply. That is
evidence about condition 5's subject. It is not the assertion §0.5 names, and it is not
reported as one. **The owner has a decision here**, and it is stated in §9 with the rest.

