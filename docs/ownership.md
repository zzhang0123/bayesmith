# Implementation ownership

> **文档状态：`decision-home`** · 某类决定的唯一登记处，仍在更新；决定的答案写回提出它的那一行。索引见 docs/README.md。

This page records who should own bayesmith's current implementation surface.
It is an R0 inventory, not a claim that every current implementation keeps its
present shape forever. The governing rule is the
[top-level design](superpowers/specs/2026-08-30-bayesmith-top-level-design.md):
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
change -- the table was born at the R0 close-out and R2's `bridge.arviz` is the
one module that reached it late, which is what the check below exists to catch
next time.

Re-measure rather than trust this sentence. Read the `Module surface` column,
expand each `x.*` pattern over the shipped modules, and list what nothing
matches. Note that the obvious pathspec is not the whole package: `git ls-files
'src/bayesmith/**/*.py'` returned 64 files here and misses the five top-level
modules (`__init__`, `amortize`, `distributions`, `errors`, `optimize`), so
pass `'src/bayesmith/*.py' 'src/bayesmith/**/*.py'` and expect 69. Measured
2026-09-03, after the `bridge.arviz` row below: 69 modules, 0 uncovered; the
same check before that row was added reported 1.

| Module surface | Class | What bayesmith owns | Boundary or intended evolution |
|---|---|---|---|
| `bayesmith.artifacts.*` | First-party core | The serialisable Task/Result/Report/Refusal/Gate protocol, canonical codec, fingerprints, invalidation matrix and deterministic gate aggregation -- the semantic core R1 publishes. | Pure data: no JAX, NumPyro, Equinox or Graph imports. It is the stable protocol the dispatch layer adapts into, not a runtime. |
| `bayesmith.graph.*` | First-party core | Graph, nodes, tracing, evaluation, structural reduction and their invariants. | Upstream distributions may be carried by nodes, but no backend may reinterpret Graph semantics. |
| `bayesmith.dispatch.*` | First-party core | Classification, partitioning, task-independent planning, exact-first execution policy, fallback reasons and user-visible route explanations. | A sampler backend executes an approved residual problem; it does not choose or redescribe the partition. |
| `bayesmith.dispatch.task` | First-party core | The Graph↔artifact orchestration adapter: manifests, task-aware compile, and the mechanical projection of existing posterior/point-estimate execution into typed Results. | The one module where a runtime Graph/InferencePlan meets the artifact protocol; it projects existing numerical results and never decides a number. |
| `bayesmith.exact.*` | First-party core | Checked Gaussian/linear/log-linear structure, conditioning certificates, first-party Wiener/GCR/GLS sampling and solving, exact discrete enumeration, corrections, Fisher and reduced-basis graph semantics. | The linear sampler remains first-party. Generic low-level kernels may be reused when they preserve the same certificates and oracles. |
| `bayesmith.marginal.*` | First-party core | Square-root information terms, exact folding/marginalisation, campaign and chain semantics, graph-derived factorization, diagnostics and the premise-checked logdet ladder. | These are marginal-likelihood components, not the future graph-level EvidenceTask. External residual integration must consume their normalized output rather than replace it. |
| `bayesmith.diagnose.*` | First-party core | Graph-native identifiability, coupling, local structure, prior sensitivity, MAP interpretation and prior diagnostics. | Mature libraries may supply low-level statistics, while observation grouping, applicability and graph interpretation remain here. |
| `bayesmith.distributions` | First-party core | Distribution declarations required by Graph semantics, including the real-coordinate convention for complex latents. | Prefer upstream distributions when their support and transform semantics are identical; compatibility of stored Graphs remains a bayesmith obligation. |
| `bayesmith.errors` | First-party core | Typed refusal and invariant vocabulary exposed by current APIs. | R1 may adapt these errors into typed `Refusal.grounds`; exceptions remain for implementation or environment failures. |
| `bayesmith.__init__`, `bayesmith.exact.__init__`, `bayesmith.dispatch.__init__`, `bayesmith.diagnose.__init__`, `bayesmith.marginal.__init__`, `bayesmith.bridge.__init__`, `bayesmith.artifacts.__init__` | First-party core | Public facade, lazy-loading behavior and stable re-export decisions. | Backend-native objects receive weaker compatibility guarantees than bayesmith artifacts. |
| `bayesmith.bridge.numpyro_bridge` | Thin adapter | Lossless translation between Graph and NumPyro plus conditioning/predictive semantics at the seam. | NumPyro owns NUTS and `Predictive`; the adapter must not fork their algorithms or leak backend objects into future common Result schemas. |
| `bayesmith.bridge.arviz` | Thin adapter | The export projection only: which of a Result's `NamedArray`s become arviz's `posterior`, `posterior_predictive`, `log_likelihood` and `observed_data` groups, and the observation-unit and chain axis names that must survive the trip. | Optional dependency, export-only (R2 §0.8). `import arviz` happens inside `to_inference_data`, so the module imports where arviz is absent; no number is recomputed on the way out. ArviZ owns LOO/WAIC and the plotting ecosystem; bayesmith keeps observation grouping and applicability, and adding a criterion computation here would take ownership R3 has not granted. |
| Graph-facing parts of `bayesmith.optimize` | First-party core | Graph objective construction, full-density versus block semantics, loss sense and result interpretation. | These semantics stay stable if the optimizer engine changes. |
| Generic optimizer in `bayesmith.optimize` | Reference / upstream candidate | Current working implementation and regression reference. | Evaluate mature JAX optimizers before adding local algorithms; replacement is allowed only with equivalent failure reporting and measured performance. |
| Graph-facing contract and result representation in `bayesmith.amortize` | First-party core | Simulation-bank meaning, conditioning interface, validation provenance, and the mapping into the heuristic/amortized posterior representation -- landed in R2 as `bayesmith.dispatch.amortized`, which encodes a trained `NeuralPosterior` as a `FittedConditionalPosterior` plus an `ArtifactKind.ESTIMATOR` artifact. | Training and calibration gates remain visible even when an upstream estimator supplies the network. What is still open is calibration, not encoding: no execution route returns an amortized `PosteriorResult` yet. |
| Local neural estimator and training loop in `bayesmith.amortize` | Reference / upstream candidate | Current compatibility route and a small independent reference. | BayesFlow, sbiJAX or another eligible SBI backend may become the production engine; do not grow local NPE architecture families without measured need. |
| `bayesmith.evidence` | Compatibility | Deprecated deep-import aliases and an unambiguous migration warning to `bayesmith.marginal`. | Retires at 1.0 into removal or a tombstone. It must never host the future EvidenceTask implementation. |

## Current execution routes

| Route | Numerical owner | Semantic owner | Current status |
|---|---|---|---|
| Linear-Gaussian posterior mean and draws | bayesmith | bayesmith | First-party production route. |
| Log-linear/log-Gaussian graph transform | bayesmith; rheplicant consumes shared arithmetic | bayesmith | First-party graph route with consumer-specific adaptation outside this repository. |
| Exact discrete enumeration | bayesmith | bayesmith | Exact oracle and direct route; dispatcher selection remains a documented gap. |
| General NUTS posterior | NumPyro | bayesmith Graph translation and dispatch | Production fallback through a thin adapter. |
| Posterior predictive: observed-data replay and replicated draws | bayesmith | bayesmith | First-party production route since R2, route-independent (it consumes any draws posterior rather than branching on the method that produced it). Diagonal-Gaussian observed nodes only; anything else is a typed `predictive_noise_unsupported` Refusal, never an approximation. |
| Gradient MAP | Current local optimizer, upstream replacement eligible | bayesmith | Production compatibility route; generic kernel ownership is provisional. |
| Amortized posterior | Current local estimator, upstream replacement eligible | bayesmith contract; heuristic approximation must remain visible | Reference/compatibility route. R2 landed the encoding (`FittedConditionalPosterior` + ESTIMATOR artifact); pending R3 calibration and upstream evaluation. |
| Streamed marginal-likelihood terms | bayesmith | bayesmith | First-party production route. |
| Graph-level Bayesian evidence and Bayes factors | None yet | bayesmith will own eligibility and Result semantics | Not implemented in R0; planned for R4–R5. |

## Review rule

An ownership change is a product decision. It must update this page and the
top-level design together, name the independent oracle and compatibility path,
and state whether old code is deleted, retained as a reference, or kept only as
an adapter. Merely adding an optional dependency does not transfer ownership.
