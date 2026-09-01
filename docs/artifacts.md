# The artifact protocol: Tasks, Results, provenance and gates

> **文档状态：`module-spec`** · 已发布模块/能力的当前设计文档，从属于顶层设计；改动对应代码须同步本页。索引见 docs/README.md。

`bayesmith.artifacts` is the R1 leaf layer: serialisable, invalidatable,
evaluable protocol objects for Bayesian tasks. It is pure data -- nothing in
it imports the Graph layer, the dispatch layer, JAX, Equinox or NumPyro at
module scope, and nothing holds a runtime object. A Graph, a callable, a
compiled executable or a backend handle is represented by a reference, a
fingerprint or a runtime attachment, never pickled into the artifact itself.
The public inventory lives in `bayesmith.artifacts/__init__.py`; the runtime
bridge is `bayesmith.dispatch.task`, reached from the package root through
`compile_task` / `execute_task`.

The governing spec is the
[top-level design](superpowers/specs/2026-08-30-bayesmith-top-level-design.md)
§2, §4 and §8 R1; the execution plan is
[the R1 plan](superpowers/plans/2026-08-30-r1-task-artifact-provenance.md).

---

## Five in, five out

Five Tasks, five Results, one mapping, exhaustive and one-to-one (§0 ruling 1).
A task that cannot be compiled or executed returns a `Refusal`, never a Report
standing in for the main Result.

| Task | Result |
|---|---|
| `PosteriorTask` | `PosteriorResult` |
| `EvidenceTask` | `EvidenceResult` |
| `PredictiveTask` | `PredictiveResult` |
| `PointEstimateTask` | `PointEstimateResult` |
| `SimulationTask` | `SimulationResult` |

`PredictiveResult` is its own Result, not `SimulationResult` plus a Report: a
predictive task conditions on data, names the posterior it came from, and
distinguishes replicated sites from carried-forward latents; a simulation has
none of those. R1 answers `PosteriorTask` and `PointEstimateTask`; the other
three tasks and their Results are frozen schema, and the runtime bridge
returns a typed `Refusal` with the code `capability_unavailable_r1`.

### Posterior representations

A posterior Result carries exactly one of four tagged representations:
`DrawsPosterior`, `WeightedDrawsPosterior`, `AnalyticPosterior` or
`FittedConditionalPosterior`. R1 adapts the existing exact-draw and NUTS routes
into the first two; the analytic and amortized shapes are reserved.

---

## The envelope and the run record

Every artifact carries an `ArtifactMeta` (kind, schema version, UUID4 id,
revision, UTC creation time, producer, parent refs, fingerprint bundle,
lifecycle, warnings, summary) and every Result carries a `RunRecord` (seed,
dtype, devices, JAX config, backend, budget, termination, timing,
approximation). Identity and content are different axes: `(artifact_id,
revision)` says *which* artifact, the fingerprint bundle says *what it was
made of*, and only the latter is read by caching and invalidation.

---

## Fingerprint boundaries

A `Fingerprint` is `(kind, algorithm="sha256-v1", digest)` over the canonical
payload bytes. The bundle has seven fixed slots, each with a stated boundary:

| Slot | Contains | Explicitly excludes |
|---|---|---|
| `model_source` | `ModelRef.identifier`, source digest, distribution package/version, build arguments | memory addresses, bare `repr(callable)` |
| `graph_structure` | node order, node type/name/parents/plate, support, `linear_in`, `depends_on_prediction`, shape/dtype metadata, joint-prior/evidence-term type and `over`, plate name/size, callable module+qualname | `Const.value`, observed and mask values |
| `data` | Const/observed/mask name, dtype, shape and bytes, explicit caller data payload | display options, runtime cache |
| `task` | task kind, statistical semantics, backend policy, budget, solver/optimizer options, gate identity | progress bar, print width |
| `compilation` | block partition, exact elimination, residual variables, method, tol, fallback policy, compiler version | wall-clock timing |
| `evaluation` | report kind, threshold, grouping, repeats, applicability policy, gate definition/version | the Result arrays themselves |
| `environment` | Python/bayesmith/backend/JAX versions, x64/dtype/device platform | host path, scratch directories |

`ModelRef.from_callable()` only derives a source digest when `inspect.getsource()`
finds stable source; otherwise the caller must supply the digest -- never a
`repr()` fallback.

---

## Invalidation matrix

`InvalidationPolicy.default()` encodes, per artifact category, which changed
slots invalidate the artifact. "Invalidate" produces an immutable, revision
`n + 1` copy marked INVALIDATED with the changed inputs and time recorded; it
never rewrites the old revision. ENVIRONMENT is in no row: a backend patch
leaves stored artifacts readable and gives the next run new provenance.

| Change | Plan | Result | EvaluationReport/Gate |
|---|:---:|:---:|:---:|
| model source / graph structure | invalidate | invalidate | invalidate |
| data / task | invalidate | invalidate | invalidate |
| compilation | — | invalidate | invalidate |
| evaluation threshold/grouping | — | reusable | invalidate |
| display option | reusable | reusable | reusable |
| backend patch/environment | readable; next run re-provenances | readable; next run re-provenances | re-evaluated by new Result identity |

---

## Refusal: `grounds`, not `evidence`

A `Refusal` is what is returned when a task cannot be answered as asked. Its
fields are fixed by §0 ruling 3: a stable `failed_premise` code, a non-empty
tuple of `Finding` under the field name `grounds`, a `ScopeRef`, and a
non-empty tuple of `Remedy`. The field is called `grounds` and never
`evidence`, because this package computes marginal likelihoods and "evidence"
is already the name of a number here. The adapters read structured fields of
the existing verdicts (`NotGaussian.reason`, `NotLogLinear.reason`, the MAP
verdict class variable) -- they never parse an exception string.

---

## Gate truth table

Status and verdict are two axes (§0 ruling 7). `BLOCKED`, `INVALIDATED` and
`ERROR` carry no verdict; only `EVALUATED` carries `PASS`, `FAIL` or
`ABSTAIN`. `aggregate_gate` applies §0.6's fixed priority, independent of the
order the slots arrive in:

| Situation | status | verdict |
|---|---|---|
| prerequisite missing | `BLOCKED` | `None` |
| input/report stale | `INVALIDATED` | `None` |
| required attempted error | `ERROR` | `None` |
| blocking optional error | `ERROR` | `None` |
| non-blocking optional error + required pass | `EVALUATED` | `PASS` |
| required applicable fail | `EVALUATED` | `FAIL` |
| required never produced | `EVALUATED` | `ABSTAIN` |
| required unverifiable/abstain | `EVALUATED` | `ABSTAIN` |
| required inapplicable | `EVALUATED` | `ABSTAIN` |
| optional inapplicable + all required pass | `EVALUATED` | `PASS` |
| all required applicable pass | `EVALUATED` | `PASS` |

`FAIL` outranks `ABSTAIN`; both outrank `PASS`. A slot carrying both a report
and an error, a duplicate requirement name, or a slot the schema never
declared are refused at the boundary.

---

## Persistence

`dump_artifact(artifact, path)` and `load_artifact(path, *, expected=None)` are
the two public persistence entry points. The disk format is NOT a bare pickle:
it is a canonical JSON transport envelope `ArtifactFile(format=
"bayesmith-artifact", codec_version=1, payload_sha256, payload_base64)` where
`payload_base64` is `canonical_dumps(artifact)` and `payload_sha256` is its
SHA-256. Writes go through a same-directory temporary file and `os.replace`
(no half-written file); reads verify format, codec version (a newer version
raises `UnsupportedSchemaVersion`, never a guessed migration), digest and
expected type before returning anything.

---

## Ownership

`bayesmith.artifacts` is first-party semantic core; `bayesmith.dispatch.task`
is the first-party orchestration adapter that projects existing runtime
results into these types. Generic backend algorithms remain upstream or
reference ownership -- see [docs/ownership.md](ownership.md).
