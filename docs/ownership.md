# Implementation ownership

> **文档状态：`decision-home`** · 某类决定的唯一登记处，仍在更新；决定的答案写回提出它的那一行。索引见 docs/README.md。

This page records who should own bayesmith's current implementation surface.
It is an R0 inventory, not a claim that every current implementation keeps its
present shape forever. The governing rule is the
[top-level design](design.md):
bayesmith owns graph-aware statistical semantics and strategically simple exact
routes; mature upstream libraries should own generally applicable numerical
algorithms when they meet the package's correctness, JAX, performance,
maintenance and observability requirements.

## Ownership classes

| Class | Meaning | Long-term action |
|---|---|---|
| **First-party core** | The behavior is part of bayesmith's differentiated statistical contract. | Maintain, optimize and test here; an upstream implementation may be an additional backend but does not silently replace the contract. |
| **Thin adapter** | bayesmith owns translation into its Graph/Task semantics while an upstream package owns the numerical engine. | Keep the adapter small, optional where possible, and guarded by contract tests and version provenance. |
| **Reference / upstream candidate** | A local implementation currently supplies a route or oracle, but the generic algorithm is not strategic ownership. | Preserve compatibility while evaluating sufficiently general and efficient upstream replacements; do not expand into an algorithm zoo. |
| **Compatibility** | The code exists so previously published imports or schemas fail safely or continue with a warning. | Do not add new behavior; retire only at the documented boundary. |

“Upstream candidate” does not mean “scheduled for deletion”. Replacement needs
a problem-family benchmark and an eligible adapter. Conversely, historical
origin in rheplicant does not make rheplicant the current owner after the
one-implementation migration has moved the behavior and its oracles here.

## Exhaustive current module inventory

The patterns below cover every Python module shipped under `src/bayesmith`. A
new module must either fit an existing row or update this inventory in the same
change -- the table was born at the R0 close-out, and two modules have reached
it late: R2's `bridge.arviz`, and R3's `bayesmith.evaluation`, which 0.7.2
opened in the same release that added the `bridge.arviz` row and declared the
table exhaustive at 69. The check below exists to catch the third.

The inventory is checked against the actual package files by
`tests/test_ownership.py`. Package wildcards cover their implementation modules;
explicit rows name standalone contracts and adapters. There is no hand-maintained
module count. Historical census measurements remain in the Git history.

| Module surface | Class | What bayesmith owns | Boundary or intended evolution |
|---|---|---|---|
| `bayesmith.artifacts.*` | First-party core | The serialisable Task/Result/Report/Refusal/Gate protocol, canonical codec, fingerprints, invalidation matrix and deterministic gate aggregation -- the semantic core R1 publishes. | Pure data: no JAX, NumPyro, Equinox or Graph imports. It is the stable protocol the dispatch layer adapts into, not a runtime. |
| `bayesmith.graph.*` | First-party core | Graph, nodes, tracing, evaluation, structural reduction and their invariants. | Upstream distributions may be carried by nodes, but no backend may reinterpret Graph semantics. |
| `bayesmith.dispatch.*` | First-party core | Classification, partitioning, task-independent planning, exact-first execution policy, fallback reasons and user-visible route explanations. | A sampler backend executes an approved residual problem; it does not choose or redescribe the partition. |
| `bayesmith.dispatch.task` | First-party core | The Graph↔artifact orchestration adapter: manifests, task-aware compile, and the mechanical projection of existing posterior/point-estimate execution into typed Results. | The orchestration facade and its private identity projection connect runtime Graph/InferencePlan values to artifacts; manifests and fingerprints live in `dispatch._task_identity`, while public imports remain compatible. Numerical decisions remain with their owning routines. |
| `bayesmith.exact.*` | First-party core | Checked Gaussian/linear/log-linear structure, conditioning certificates, first-party Wiener/GCR/GLS sampling and solving, exact discrete enumeration, corrections, Fisher and reduced-basis graph semantics. | The linear sampler remains first-party. Generic low-level kernels may be reused when they preserve the same certificates and oracles. |
| `bayesmith.marginal.*` | First-party core | Square-root information terms, exact folding/marginalisation, campaign and chain semantics, graph-derived factorization, diagnostics and the premise-checked logdet ladder. | These are normalized marginal-likelihood components consumed by graph-level evidence assembly. External residual integration must consume their normalized output rather than replace it. |
| `bayesmith.diagnose.*` | First-party core | Graph-native identifiability, coupling, local structure, prior sensitivity, MAP interpretation and prior diagnostics. | Mature libraries may supply low-level statistics, while observation grouping, applicability and graph interpretation remain here. |
| `bayesmith.evaluation.*` | First-party core | The model-checking layer R3 opened: reports ABOUT finished Results, never Results. It reads `PosteriorResult`, `PredictiveResult` and `SimulationResult` and produces `EvaluationReport`s whose `subject_ref` points back at what was read; the one declared false-positive rate (`ALPHA`) every R3 check shares lives here. | Depends only downward (`dispatch`, `graph`, `artifacts`, `bridge.arviz`); `artifacts` and `dispatch` never import it, and `tests/test_layering.py` holds that in a subprocess. It does not modify a result, choose an algorithm, or re-judge a verdict that has a home in `bayesmith.diagnose`. Mature statistics (LOO, PSIS) stay upstream; what is owned is observation grouping, applicability and the gate semantics. |
| `bayesmith.distributions` | First-party core | Distribution declarations required by Graph semantics, including the real-coordinate convention for complex latents. | Prefer upstream distributions when their support and transform semantics are identical; compatibility of stored Graphs remains a bayesmith obligation. |
| `bayesmith.cumulants.*` | First-party core | Manual Edgeworth field likelihoods and the cumulant-contraction interface; dense reference and low-rank contractions, with optional periodic Gaussian/Fourier construction utilities. | Reuses NumPyro distributions and JAX differentiation/FFT; coordinate representations remain below the graph/workflow boundary. Global validity, automatic order selection and non-Gaussian predictive sampling are deferred. |
| `bayesmith.compiled` | First-party core | Backend-neutral compiled evidence problem, term partition and parameter-layout contract. | A leaf shared by dispatch and backend adapters; importing it does not import Graph, artifacts or a sampler. The residual execution route remains unavailable. |
| `bayesmith.errors` | First-party core | Typed refusal and invariant vocabulary exposed by current APIs. | R1 may adapt these errors into typed `Refusal.grounds`; exceptions remain for implementation or environment failures. |
| `bayesmith.__init__`, `bayesmith.exact.__init__`, `bayesmith.dispatch.__init__`, `bayesmith.diagnose.__init__`, `bayesmith.marginal.__init__`, `bayesmith.bridge.__init__`, `bayesmith.artifacts.__init__` | First-party core | Public facade, lazy-loading behavior and stable re-export decisions. | Backend-native objects receive weaker compatibility guarantees than bayesmith artifacts. |
| `bayesmith.bridge.numpyro_bridge` | Thin adapter | Lossless translation between Graph and NumPyro plus conditioning/predictive semantics at the seam. | NumPyro owns NUTS and `Predictive`; the adapter must not fork their algorithms or leak backend objects into future common Result schemas. |
| `bayesmith.bridge.arviz` | Thin adapter | The export projection only: which of a Result's `NamedArray`s become arviz's `posterior`, `posterior_predictive`, `log_likelihood` and `observed_data` groups, and the observation-unit and chain axis names that must survive the trip. | Optional dependency, export-only (R2 §0.8). `import arviz` happens inside `to_inference_data`, so the module imports where arviz is absent; no number is recomputed on the way out. ArviZ owns LOO/WAIC and the plotting ecosystem; bayesmith keeps observation grouping and applicability, and adding a criterion computation here would take ownership R3 has not granted. |
| `bayesmith.bridge.jaxns_bridge` | Thin adapter | Translation from a compiled problem to JAXNS and back to numerical results, including termination facts. | Experimental low-level adapter. JAXNS was selected; public residual EvidenceTask integration, box/truncation semantics and stability reporting remain incomplete. |
| `bayesmith.reweight` | First-party core | Fixed-reference posterior recycling, full graph density ratios and marginal hyperparameter MAP semantics. | Reuses existing GCR sampling, weight normalization and `optimize.minimize`; raw Monte Carlo estimates and Kish ESS do not certify evidence or convergence. |
| Graph-facing parts of `bayesmith.optimize.__init__` | First-party core | Graph objective construction, full-density versus block semantics, loss sense and result interpretation. | These semantics stay stable if the optimizer engine changes. `optimize` became a package in 0.10.0 so the certificate could live beside them; the module's public names did not move. |
| Generic optimizer in `bayesmith.optimize.__init__` | Reference / upstream candidate | Current working implementation and regression reference. | Evaluate mature JAX optimizers before adding local algorithms; replacement is allowed only with equivalent failure reporting and measured performance. Since 0.10.0 a replacement is easier to judge, not harder: `certify=` measures the ANSWER from the objective's own derivatives, so a candidate engine is compared on cost and on how often it reaches a certifiable point, rather than on trust. |
| `bayesmith.optimize.certify` | First-party core | The convergence CERTIFICATE: the Newton decrement, its upper bound from the recomputed residual and a curvature floor, which floors count as a proof, and the Newton polish that makes a first-order iterate certifiable. | Lifted whole from `rheplicant.inference.certify` at `e1acb6f` (T-004 half B), which is the one-implementation migration moving a behaviour and its oracles here -- the paragraph above this table is the rule it follows. It takes plain callables and pytrees: no Graph, no artifacts, no parameter names, and nothing from `bayesmith` at all, which is why it could move byte for byte and why `tests/test_certify.py` came with it unchanged. Generic linear algebra (CG, Lanczos) may be replaced by an upstream implementation; what is NOT negotiable is which floors certify. A probe may refuse and must never approve. |
| Graph-facing contract and result representation in `bayesmith.amortize` | First-party core | Simulation-bank meaning, conditioning interface, validation provenance, and the mapping into the heuristic/amortized posterior representation -- landed in R2 as `bayesmith.dispatch.amortized`, which encodes a trained `NeuralPosterior` as a `FittedConditionalPosterior` plus an `ArtifactKind.ESTIMATOR` artifact. | Training and calibration gates remain visible even when an upstream estimator supplies the network. R3 closed the calibration half and left the execution half exactly where R2 put it: the local `NeuralPosterior` has been through `evaluation.sbc`'s sampler arm and scored PASS (KS D 0.0683, p 0.1159, 90% coverage 0.890, 300/300 replicates usable), and **no execution route returns an amortized `PosteriorResult` still** -- `SUPPORTED_TASK_KINDS` has no amortized member and `execute_task` constructs no `FittedConditionalPosterior`. A calibration measurement is not an execution route, and this row must not be read as claiming the second because the first happened. |
| Local neural estimator and training loop in `bayesmith.amortize` | Reference / upstream candidate | Current compatibility route and a small independent reference. | BayesFlow, sbiJAX or another eligible SBI backend may become the production engine; do not grow local NPE architecture families without measured need. |
| `bayesmith.evidence`, `bayesmith.evidence.__init__` | Compatibility | Deprecated deep-import aliases and an unambiguous migration warning to `bayesmith.marginal`. | Retires at 1.0 into removal or a tombstone. It must never host the future EvidenceTask implementation. |

## Current execution routes

| Route | Numerical owner | Semantic owner | Current status |
|---|---|---|---|
| Linear-Gaussian posterior mean and draws | bayesmith | bayesmith | First-party production route. |
| Log-linear/log-Gaussian graph transform | bayesmith; rheplicant consumes shared arithmetic | bayesmith | First-party graph route with consumer-specific adaptation outside this repository. |
| Exact discrete enumeration | bayesmith | bayesmith | Exact oracle and direct route; dispatcher selection remains a documented gap. |
| General NUTS posterior | NumPyro | bayesmith Graph translation and dispatch | Production fallback through a thin adapter. |
| Posterior predictive: observed-data replay and replicated draws | bayesmith | bayesmith | First-party production route since R2, route-independent (it consumes any draws posterior rather than branching on the method that produced it). Diagonal-Gaussian observed nodes only; anything else is a typed `predictive_noise_unsupported` Refusal, never an approximation. |
| Forward simulation from prior, fixed or posterior parameters | bayesmith | bayesmith | First-party production route since R3. Not a second simulator: the posterior-sourced arm is bitwise identical to a predictive run at the same key, and the fixed arm is the same call with the setting repeated along the draw axis, so both inherit the row above's diagonal-Gaussian domain and refuse outside it with `predictive_noise_unsupported`. **The prior arm does not**: it samples each node from the distribution the node itself declares, so it is defined on graphs the predictive seam refuses -- measured, on a graph whose observed node is a `CirculantNormal`, `prior_draws` returns draws where `replicated_draws` raises `NotGaussian`. That asymmetry is a property of the two generation laws, not an oversight, and it is why a prior predictive CHECK can still be unverifiable on a graph whose prior simulation ran. |
| Gradient MAP | Current local optimizer, upstream replacement eligible | bayesmith | Production compatibility route; generic kernel ownership is provisional. |
| Amortized posterior | Current local estimator, upstream replacement eligible | bayesmith contract; heuristic approximation must remain visible | Reference/compatibility route, and still not an executable one. R2 landed the encoding (`FittedConditionalPosterior` + ESTIMATOR artifact); R3 measured the calibration and ran the upstream evaluation, and **no candidate passed**, so the local reference is retained. BayesFlow 2.0.14 fails §1.5 rows 2 and 3 (SBC FAIL at p = 0.0004; exits 1 under `JAX_ENABLE_X64=1`); sbiJAX 0.4.0 passes rows 2 and 5 and is the one to re-examine, but row 1 is unmeasured and its x64 verdict flips to FAIL. See the record. What is still open after R3 is the execution route, not the calibration number. |
| Streamed marginal-likelihood terms | bayesmith | bayesmith | First-party production route. |
| Graph-level Bayesian evidence | bayesmith analytic assembly; experimental JAXNS adapter | bayesmith eligibility, normalization and Result semantics | Whole-graph linear-Gaussian evidence executes with proper normalized priors and x64. Residual EvidenceTask execution remains unavailable even when an extra is installed. |

## Why the near-side copies are being retired rather than kept in step

T-004 retires the rows in `tests/crosscheck/test_provenance.py` that required
rheplicant to keep its own `SqrtInfo`, `marginalise`, `check_linearity` and
`linear_operator`. The alternative policy -- keep both copies and hold them in
step with cross-checks -- was already being run, and this is the measurement
that ended it.

`_worse` is a six-line reduction used by the affinity check on both sides:
`max` that propagates NaN, over values that may carry an `Unresolved` marker
meaning "this probe declined to judge". bayesmith's copy
(`src/bayesmith/exact/linearity.py:155`) branches on the marker and returns
`Unresolved(worst)` whenever either argument carries it, with a comment saying
why: "Without this the flag would be dropped whenever the plain float happened
to be the larger of the two, which is a coin flip." rheplicant's copy
(`src/rheplicant/inference/linear.py:477`) has no such branch; its body is
`return current if current >= value else value`.

Measured 2026-09-20 in this checkout, against rheplicant at `8ecc708`, calling
both functions on the same four inputs:

| inputs | bayesmith | rheplicant |
|---|---|---|
| `Unresolved(0.1), 0.5` | `Unresolved(0.5)` | `0.5` |
| `Unresolved(0.5), 0.1` | `Unresolved(0.5)` | `Unresolved(0.5)` |
| `0.5, Unresolved(0.1)` | `Unresolved(0.5)` | `0.5` |
| `0.1, Unresolved(0.5)` | `Unresolved(0.5)` | `Unresolved(0.5)` |

Two of the four disagree, and they are the two where the unresolved probe is
not also the larger number -- the coin flip, arriving as predicted. Both sides
share the `Unresolved` TYPE (it is in `bayesmith.exact.__all__`, imported
there rather than respelled), so this is a difference in the reduction and not
in the vocabulary.

Three things about this make it the argument rather than an anecdote:

- **Neither docstring says the two differ.** rheplicant's reads "``max`` that
  PROPAGATES NaN, and keeps an `Unresolved` that wins" -- which describes
  bayesmith's rule and not the code beneath it, since the marker survives only
  by winning on magnitude. Documentation stayed in step while the code did
  not, so reading either file gives the same wrong impression of the other.
- **Nothing caught it.** The function is private on both sides, so the
  symbol-level provenance guard did not name it, and the surviving
  `test_linear.py` comparisons read the affinity VERDICT rather than the
  marker on the reported number. A copy below the granularity of every guard
  is a copy nobody is holding in step.
- **The drift is silent in the direction that matters.** A lost `Unresolved`
  turns "this probe could not be judged at this precision" into a clean
  number, which is the reading that lets a caller proceed.

This is the same failure this repository has spent the most time on, in a
different medium: six copies of one measurement going stale on a day none of
them was edited. The conclusion drawn there applies here -- one
implementation, or a test; not two implementations and a hope. The copies are
therefore retired to one owner rather than kept in step, and the ruling that
the Bayesian numerics belong to bayesmith is what decides which owner.

What remains true while the retirement is in progress: rheplicant still
defines all four symbols today, so `tests/crosscheck/test_sqrtinfo_agrees.py`
and `tests/crosscheck/test_linear.py` still compare two implementations, and
a mutation in this package's own arithmetic still turns them red (measured;
the table is in `test_provenance.py`'s `DELEGATION_PERMITTED` comment). The
guard that replaced the retired rows asserts the one thing the retirement must
not cost: when a definition leaves the far side, its comparison here leaves in
the same change rather than passing against itself.

## Review rule

An ownership change is a product decision. It must update this page and the
top-level design together, name the independent oracle and compatibility path,
and state whether old code is deleted, retained as a reference, or kept only as
an adapter. Merely adding an optional dependency does not transfer ownership.

### T-004 consumer integration acceptance (2026-09-20)

Against rheplicant `716d38c`, `SamplingPlan.estimate` now calls the shared
`bayesmith.optimize.certify` machinery. Its provenance row therefore moves
from `OWN` to `SHARED_KERNEL`; the dispatch comparisons test route assembly,
iterates and moments, not independent implementations of that certificate.
The radiometer frozen-noise fixed point is still compared under an explicitly
uncertified fixed budget. A separate test requires the normal certified route
to refuse it: unchanged iterates do not imply a stationary full joint density.
The positive certificate oracle remains in `tests/test_certify.py`.


### Cross-checks moved downstream (2026-10-02)

`tests/crosscheck/`, including the provenance table the two sections around
this one describe, was removed from this repository together with the
`crosscheck.yml` and `seam.yml` workflows. rheplicant is the downstream; a
suite here that imports it cannot pass on its own, and its CI could not go
green until rheplicant's main moved. The comparisons and the delegation guard
are for rheplicant's suite to carry. Their last revision here is commit
`d861220`. The two sections that follow and precede are kept as the record of
what those tests established.

### Local rheplicant module splits (2026-09-21)

The local rheplicant 0.9 wheel built at `624b396` moves `Block` to
`plan_results`, four linear solver/conditioning entry points to `linear_solve`,
and the estimate body to `plan_estimate.run_estimate`. Public imports remain
facades. The provenance table now names those definitions and uses an explicit
legacy map only when the corresponding split module is absent. A missing
symbol or lost delegation in an existing split module remains a failure.
This supports the pre-split and local split layouts without changing CI's
rheplicant installation source or claiming the split has been published.

`run_estimate` directly reads certificate size utilities; that alone does not
prove that it uses the shared certificate arithmetic. A separate provenance
row follows `plan_settings._certify` (formerly `plan._certify`), and a runtime
sentinel at `bayesmith.optimize.certify.decrement` must be reached from the
public `SamplingPlan.estimate` exit. Export-identity and forwarding checks keep
the guarded definitions connected to the public calls.

The downstream stability table's `_zeta_joint` citation to
`tests/crosscheck/test_provenance.py` is incorrect: that file does not compare
joint covariances. `tests/marginal/test_chain_joint_covariance.py` validates
bayesmith's assembly against its own independent mathematical oracle.
rheplicant has its own full-covariance oracle tests in
`tests/evidence/test_chain_smoother.py`, including
`test_the_whole_covariance_matches_including_the_cross_epoch_blocks` and
`test_a_wide_chains_whole_covariance_matches_too`. These are separate oracle
checks, not a cross-package equality test; the downstream table should say so.
The downstream repository was inspected but not modified here.
