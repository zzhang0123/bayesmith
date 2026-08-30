# Implementation ownership

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

The patterns below cover every Python module shipped under `src/bayesmith` at
this R0 close-out. A new module must either fit an existing row or update this
inventory in the same change.

| Module surface | Class | What bayesmith owns | Boundary or intended evolution |
|---|---|---|---|
| `bayesmith.graph.*` | First-party core | Graph, nodes, tracing, evaluation, structural reduction and their invariants. | Upstream distributions may be carried by nodes, but no backend may reinterpret Graph semantics. |
| `bayesmith.dispatch.*` | First-party core | Classification, partitioning, task-independent planning, exact-first execution policy, fallback reasons and user-visible route explanations. | A sampler backend executes an approved residual problem; it does not choose or redescribe the partition. |
| `bayesmith.exact.*` | First-party core | Checked Gaussian/linear/log-linear structure, conditioning certificates, first-party Wiener/GCR/GLS sampling and solving, exact discrete enumeration, corrections, Fisher and reduced-basis graph semantics. | The linear sampler remains first-party. Generic low-level kernels may be reused when they preserve the same certificates and oracles. |
| `bayesmith.marginal.*` | First-party core | Square-root information terms, exact folding/marginalisation, campaign and chain semantics, graph-derived factorization, diagnostics and the premise-checked logdet ladder. | These are marginal-likelihood components, not the future graph-level EvidenceTask. External residual integration must consume their normalized output rather than replace it. |
| `bayesmith.diagnose.*` | First-party core | Graph-native identifiability, coupling, local structure, prior sensitivity, MAP interpretation and prior diagnostics. | Mature libraries may supply low-level statistics, while observation grouping, applicability and graph interpretation remain here. |
| `bayesmith.distributions` | First-party core | Distribution declarations required by Graph semantics, including the real-coordinate convention for complex latents. | Prefer upstream distributions when their support and transform semantics are identical; compatibility of stored Graphs remains a bayesmith obligation. |
| `bayesmith.errors` | First-party core | Typed refusal and invariant vocabulary exposed by current APIs. | R1 may adapt these errors into typed `Refusal.grounds`; exceptions remain for implementation or environment failures. |
| `bayesmith.__init__`, `bayesmith.exact.__init__`, `bayesmith.dispatch.__init__`, `bayesmith.diagnose.__init__`, `bayesmith.marginal.__init__`, `bayesmith.bridge.__init__` | First-party core | Public facade, lazy-loading behavior and stable re-export decisions. | Backend-native objects receive weaker compatibility guarantees than bayesmith artifacts. |
| `bayesmith.bridge.numpyro_bridge` | Thin adapter | Lossless translation between Graph and NumPyro plus conditioning/predictive semantics at the seam. | NumPyro owns NUTS and `Predictive`; the adapter must not fork their algorithms or leak backend objects into future common Result schemas. |
| Graph-facing parts of `bayesmith.optimize` | First-party core | Graph objective construction, full-density versus block semantics, loss sense and result interpretation. | These semantics stay stable if the optimizer engine changes. |
| Generic optimizer in `bayesmith.optimize` | Reference / upstream candidate | Current working implementation and regression reference. | Evaluate mature JAX optimizers before adding local algorithms; replacement is allowed only with equivalent failure reporting and measured performance. |
| Graph-facing contract and result representation in `bayesmith.amortize` | First-party core | Simulation-bank meaning, conditioning interface, validation provenance and eventual mapping into heuristic/amortized PosteriorResult. | Training and calibration gates remain visible even when an upstream estimator supplies the network. |
| Local neural estimator and training loop in `bayesmith.amortize` | Reference / upstream candidate | Current compatibility route and a small independent reference. | BayesFlow, sbiJAX or another eligible SBI backend may become the production engine; do not grow local NPE architecture families without measured need. |
| `bayesmith.evidence` | Compatibility | Deprecated deep-import aliases and an unambiguous migration warning to `bayesmith.marginal`. | Retires at 1.0 into removal or a tombstone. It must never host the future EvidenceTask implementation. |

## Current execution routes

| Route | Numerical owner | Semantic owner | Current status |
|---|---|---|---|
| Linear-Gaussian posterior mean and draws | bayesmith | bayesmith | First-party production route. |
| Log-linear/log-Gaussian graph transform | bayesmith; rheplicant consumes shared arithmetic | bayesmith | First-party graph route with consumer-specific adaptation outside this repository. |
| Exact discrete enumeration | bayesmith | bayesmith | Exact oracle and direct route; dispatcher selection remains a documented gap. |
| General NUTS posterior | NumPyro | bayesmith Graph translation and dispatch | Production fallback through a thin adapter. |
| Gradient MAP | Current local optimizer, upstream replacement eligible | bayesmith | Production compatibility route; generic kernel ownership is provisional. |
| Amortized posterior | Current local estimator, upstream replacement eligible | bayesmith contract; heuristic approximation must remain visible | Reference/compatibility route pending R2/R3 calibration and upstream evaluation. |
| Streamed marginal-likelihood terms | bayesmith | bayesmith | First-party production route. |
| Graph-level Bayesian evidence and Bayes factors | None yet | bayesmith will own eligibility and Result semantics | Not implemented in R0; planned for R4–R5. |

## Review rule

An ownership change is a product decision. It must update this page and the
top-level design together, name the independent oracle and compatibility path,
and state whether old code is deleted, retained as a reference, or kept only as
an adapter. Merely adding an optional dependency does not transfer ownership.
