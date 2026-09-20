# The 0.9 stable baseline

> **文档状态：`module-spec`** · 0.9 的公共契约、实验边界与兼容政策；从属于顶层设计。

## What is frozen

The stable baseline freezes the responsibility boundaries and the supported
behavior of existing public entry points. It does not certify every model, make
an approximate density exact, or declare the R8 requirements for 1.0 complete.
The version is 0.9.0; publication and release readiness require their own evidence.

The maintained surface includes graph construction/evaluation, supported
structural dispatch and exact solvers, the NumPyro posterior bridge, typed Tasks,
Results, Refusals and serialization, applicable prediction/model checks, and
whole-graph linear-Gaussian evidence. Each operation retains its explicit shape,
precision, prior, covariance and applicability requirements. A typed refusal is
part of the contract, not a missing value to replace with an approximation.

Existing root exports and documented module APIs remain available. The
`dispatch.classify.prior_environment` and `dispatch.prior_environment` imports
remain aliases of the one implementation in `exact._environment`. Public
identity helpers remain at `dispatch.task` while their private implementation
lives in `dispatch._task_identity`. These moves do not create new public APIs.

## Three distinct capability levels

| Level | Surface | What users may conclude |
|---|---|---|
| Maintained | Graph, compiler, supported exact/NUTS routes, `marginal.*`, `diagnose.*`, the graph-facing half of `optimize`, artifacts, applicable evaluation, analytic evidence | The documented contract is covered by regression checks within its declared domain. |
| Experimental | `reweight`, `cumulants`, amortized reference implementation, the generic descent engine inside `optimize`, low-level `jaxns_bridge` and compiled residual problem | Useful primitives with explicit limits; no general overlap, density-validity, calibration or convergence certification. |
| Unavailable public route | Residual `EvidenceTask`, executable amortized task route, conditional-flow likelihood, R7 workflow engine | Installing an optional backend or constructing a schema does not enable an unimplemented route. |

### Every module the sibling package imports, and its level

The table above is by surface. A consumer reads by MODULE, and the downstream
package (rheplicant) was left without a level for three of the families it
depends on, `diagnose.*`, `optimize` and `marginal.*`, while depending on all
three. Measured 2026-09-20 in this checkout, by reading rheplicant's own
`from bayesmith... import` lines: twenty-one distinct module paths, in eight
families.

| Module family | Level | Regression evidence measured here | Note |
|---|---|---|---|
| `errors`, `distributions` | Maintained | part of the typed-refusal and Graph-semantics contract above | No change. |
| `exact.*` (imported: `solve`, `fisher`, `linearity`, `gaussian`, `gls`, `block`, `loglinear`, `precision`, `reduced_basis`) | Maintained | `tests/exact/` collects 595 | Already covered as "supported exact routes". `exact.precision.diagonal_from` is the one name that subpackage exports; the protocol and its implementations stay internal. |
| `dispatch.factor` | Maintained | covered as "compiler" above | No change. |
| `marginal.*` (imported: `sqrtinfo`, `chain`, `compress`, `diagnostics`; level covers the family) | **Maintained** | `tests/marginal/` collects 564, plus the `EAGER:`/`PLAN:` logdet gates in `tests/numerical_gates/` | Square-root information terms, exact folding and marginalisation, the chain recursion and the premise-checked logdet ladder. A typed refusal is the answer where a premise fails; it is not an approximation to work around. |
| `diagnose.*` (imported: `identifiability`, `sensitivity`, `local`; level covers the family) | **Maintained** | `tests/diagnose/` collects 170, plus `tests/numerical_gates/boundary_diagnose_graph.py` | Graph-native identifiability, coupling, local structure and prior sensitivity. These report ABOUT a model; they do not certify that a sampler converged. |
| `optimize` — `fit`, `check_loss_sense`, `sense_of`, `Fit` | **Maintained** | `tests/test_optimize.py` collects 46 | The graph-facing half: which density is descended (the FULL joint, D7), block versus whole-graph semantics, the loss-sense guard, and what a `Fit` means. These survive a change of engine. |
| `optimize` — the descent engine inside `minimize` | **Experimental (reference implementation)** | same 46 | Two methods, `"adam"` and `"gradient"`, hand-written over `jax.lax.scan`, `steps` taken exactly with no early stop. Treat it as a working reference and a regression baseline, not as a maintained optimiser. The ENGINE being a reference is now separable from the ANSWER being trustworthy: see the row below. |
| `optimize.certify`, and `certify=` on `minimize`/`fit` | **Maintained** | `tests/test_certify.py` 50, `tests/test_optimize_certificate.py` 20 | Added in 0.10.0. `Fit.converged` is a proof that the Newton decrement at the returned point is within a stated distance, in units of the curvature's own standard deviation. Only a proven curvature floor certifies — a Lanczos probe may refuse and never approve. Where the arithmetic cannot support the claim the answer is `Fit.refusal` with its reason, not a certificate. |
| `amortize` | Experimental | already listed as "amortized reference implementation" | The execution route remains unavailable; see `docs/ownership.md`. |

`optimize`'s split is the one that matters to a consumer, so state it without
the table: **the semantics are maintained, the engine is a reference
implementation, and since 0.10.0 the ANSWER can be certified independently of
the engine that found it.** What `fit` optimises, which latents it holds, how
the loss sense is checked and what the returned `Fit` reports are contract.

How far the descent got used to be outside the contract entirely: 0.9.0 ran a
fixed number of steps and reported the objective it reached, and a caller who
needed to know whether that point was the optimum had to compare it against
something this package did not supply. 0.10.0 supplies it. `certify=<limit>`
makes `Fit.converged` a proof that the point is within `limit` of the minimum
in units of the curvature's own standard deviation — measured from the
objective's own gradient and Hessian-vector products, so it does not care
which engine produced the point or how many steps it took. A reference
optimiser with a certified answer is a different thing from a reference
optimiser you have to trust.

The certified quantity is the local Newton decrement. Its interpretation as
an actual distance to a minimum is exact for a quadratic objective; for a
nonlinear objective it concerns the local quadratic model and does not prove
a global MAP. An upper bound exceeding the requested limit is insufficient
to certify, not proof that the actual decrement exceeds that limit.

Two further limits, stated because they are what a reader will assume otherwise. The
verdict is about the objective it was handed: with `names=`, it certifies that
BLOCK's conditional minimum, not the graph's joint MAP. And without
`certify=`, `Fit.converged` is `False` — because nothing was measured, not
because something failed; `Fit.refusal` is what distinguishes the two.

TRIS and Campbell campaigns are application/research assets. Their measurements
retain their date, model, sample budget and validation status; they do not enlarge
the core package's guarantee. In particular, failed TRIS recovery remains failed.

## Compatibility policy

- A patch release repairs correctness or packaging within an existing supported
  domain. It may refuse an input that previously produced a mathematically wrong
  answer; the release notes must name the trigger and changed behavior.
- A minor pre-1.0 release may add capabilities or deliberately adjust an API, but
  must state the migration and test existing supported imports and consumers.
- Underscore-prefixed implementation modules are internal. Their ownership and
  dependency direction are stable design decisions; their filenames are not a
  consumer extension protocol.
- Artifact codec/schema changes require explicit versioning and old/new-version
  refusal tests. A refactor must not silently change field meanings or provenance.
- The deprecated `bayesmith.evidence` aliases still refer to `marginal`. They are
  not reused for graph-level evidence and are not removed before the stated 1.0
  boundary.
- Generic backend objects and algorithms stay owned upstream. Optional extras do
  not become required dependencies, and no import silently selects an unfinished
  evidence route.

## Release acceptance

Acceptance is ordered: current core/consumer regression evidence; independently
reviewed numerical and structural changes; wheel content/install checks; rebuild
from sdist and execution of its portable tests; Python 3.11 compatibility smoke;
documentation build, tutorials, links and browser checks; finally version,
changelog and artifact hashes. Repository-only governance, website and research
tests remain in source CI rather than being shipped without their inputs.

Exact test counts belong to a particular collection or JUnit artifact. A green
subset does not imply a green full suite, and locally defined CI workflows are
not evidence that remote Linux jobs have run. Remaining external checks must be
named before a candidate is described as ready to publish.

## Runtime compatibility boundary

The 0.9 baseline caps NumPyro below 0.22. Clean-install validation with 0.22
exposed changed argument-validation and Gibbs dynamic-support behavior that breaks
existing refusal and sampling contracts. This baseline is verified with 0.21;
0.22 support requires a separate adapter/contract migration. The cap does not
claim that every older dependency combination has been tested.
