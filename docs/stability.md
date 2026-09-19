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
| Maintained | Graph, compiler, supported exact/NUTS routes, artifacts, applicable evaluation, analytic evidence | The documented contract is covered by regression checks within its declared domain. |
| Experimental | `reweight`, `cumulants`, amortized reference implementation, low-level `jaxns_bridge` and compiled residual problem | Useful primitives with explicit limits; no general overlap, density-validity, calibration or convergence certification. |
| Unavailable public route | Residual `EvidenceTask`, executable amortized task route, conditional-flow likelihood, R7 workflow engine | Installing an optional backend or constructing a schema does not enable an unimplemented route. |

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
