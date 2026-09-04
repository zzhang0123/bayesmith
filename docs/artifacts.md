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
R2 moved exactly two things on this page: it made `PredictiveTask` executable,
and it added one member to `ArtifactKind`. Its plan is
[the R2 plan](superpowers/plans/2026-08-31-r2-predictive-seam.md), and what was
accepted rather than merely planned is
[the R2 close-out](superpowers/specs/2026-08-31-r2-close-out.md).

R3 moved two more, and neither is a schema change: it made `SimulationTask`
executable, and it began writing `EvaluationReport`s under seven new
`report_kind` codes -- which cost no migration, because R1 froze that field as
a code string rather than as an enum. The layer that writes them is
[`docs/evaluation.md`](evaluation.md); its plan is
[the R3 plan](superpowers/plans/2026-09-02-r3-model-checking.md).

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
none of those.

Four of the five are executed and one is not. `SUPPORTED_TASK_KINDS` in
`bayesmith.dispatch.task` is `{POSTERIOR, POINT_ESTIMATE, PREDICTIVE,
SIMULATION}` -- R1 answered the first two, R2 added the third and R3 the
fourth. `EvidenceTask` alone remains frozen schema and nothing else: the
runtime bridge refuses it in `_refuse_before_compiling`, before a plan is paid
for, with the code `capability_unavailable_r1` and a `Finding` whose `expected`
field is that same supported set. A caller therefore reads which questions are
answered off the refusal it just received, not off this page, which is the only
version of that list that cannot go stale.

R3's `SimulationTask` runs from all three `ParameterSource` kinds, and all
three reach the observations through primitives R2 already built rather than
through a simulator written beside them. `PRIOR` walks the graph with
`prior_draws`, observed nodes included, which is exactly the prior predictive;
`FIXED` pushes one setting through `forward_draws`; `POSTERIOR_RESULT` pushes a
source posterior's draws through the same `replicated_draws` call, with the
same key, that a predictive run makes -- so the two tasks answer with the same
bits, and
`test_a_posterior_source_simulation_is_the_predictive_replication_bit_for_bit`
says so at `rtol=0`. One forward model, reached by three routes.

### Posterior representations

A posterior Result carries exactly one of four tagged representations:
`DrawsPosterior`, `WeightedDrawsPosterior`, `AnalyticPosterior` or
`FittedConditionalPosterior`. R1 adapts the existing exact-draw and NUTS routes
into the first two.

`FittedConditionalPosterior` is no longer reserved.
`bayesmith.dispatch.amortized` encodes a trained
`bayesmith.amortize.NeuralPosterior` into one:
`fitted_conditional_posterior()` returns the representation together with the
`EstimatorArtifact` its `estimator_ref` points at, so a caller can persist the
estimator and keep the reference honest. What crosses into the artifact layer is
a reference, an opaque `bytes` blob (the parameter leaves that
`equinox.tree_serialise_leaves` writes) and a canonical manifest -- the embed
callable's module and qualname, the mixture shape, the MLP's structure, the
training bank's standardization arrays. Never the `eqx.Module` and never the
callable, because a callable has no canonical form (§0 ruling 4). That split is
what keeps this page's opening claim true: JAX and equinox are imported on the
dispatch side of the bridge, and the artifacts layer still sees no array of
theirs.

Two things that does not yet mean. **No execution route emits an amortized
posterior** -- `execute_task` never constructs a `FittedConditionalPosterior`,
and R3 did not change that. What R3 did add is the calibration number, which is
a different thing: the local `NeuralPosterior` has now been through the SBC
harness's sampler arm and scored `PASS` at KS D = 0.0683, p = 0.1159, with 90%
interval coverage 0.890 over 300 replicates
([the record](superpowers/specs/2026-09-04-amortized-calibration.md)). A
measurement is not a route. And `AnalyticPosterior` is still reserved in the
original sense: nothing under `src/` constructs one.

---

## The predictive seam: replay, replication, and what is refused

`execute_task(planned, key=..., source_posterior=...)` runs a predictive task.
Both keyword arguments are required there and a missing one raises rather than
being invented: minting a key here would produce a run that is irreproducible
while looking exactly like a seeded one, and there is no artifact store in this
release to load the source posterior from.

The source posterior is checked twice before any number is generated. Its
`(artifact_id, revision)` must be the pair `task.source_posterior_ref` names --
a different posterior is a caller error and raises. Then its `data`,
`graph_structure` and `model_source` fingerprints must equal this task's, and a
disagreement is not an exception but a typed `Refusal` with the premise
`posterior_data_mismatch`, whose `Finding.observed` lists exactly which of the
three slots moved.

Replay and replication are two verbs over ONE forward model. Both
`bayesmith.dispatch.predictive` primitives read the same `loc`/`scale` from
`bayesmith.exact.gaussian.observation_parts`; `replicated_draws` calls
`Normal(loc, scale).sample`, one draw per source draw with the draw axis
one-to-one and no resampling, and `pointwise_log_likelihood` calls
`Normal(loc, scale).log_prob(observed)`, zeroing masked positions. There is no
second, hand-written simulator that could drift from the density.
`observation_parts` is a diagonal walk, so a correlated or non-Gaussian
observed node raises `NotGaussian` out of the primitive, and the bridge adapts
it into the `predictive_noise_unsupported` Refusal rather than approximating
quietly.

### The observation unit

The pointwise array is indexed by draw first. After that leading axis:

* one observed node keeps its own axis names -- its declared plate names, then
  `{name}_dim{i}` for axes the graph has no name for;
* several observed nodes are flattened and concatenated into a single
  `observation` axis, in declaration order.

`PredictiveResult` records what those units were: `observation_unit` is the
observed node names joined by `", "`, and `grouping` is the plate the observed
nodes share when they share exactly one and `None` otherwise. Both are `None`
when the graph observes nothing.

Mind the two field names, which are not the same word: a `PosteriorResult`
carries `pointwise_log_likelihood`, a `PredictiveResult` carries
`pointwise_log_density`. The array inside both is the one
`pointwise_log_likelihood()` produced and is named `log_likelihood` either way.

### `predictive_ready`, and abstaining instead of fabricating

A posterior run computes the pointwise log-likelihood whenever the graph has an
observed node, and swallows `NotGaussian` into `None` when the observation is
not diagonal Gaussian. Three fields then agree by construction:
`log_density_availability` is `POINTWISE` exactly when
`pointwise_log_likelihood` is present -- `PosteriorResult.__post_init__`
asserts that biconditional in both directions -- and `predictive_ready` is that
same fact as a flag a caller can branch on before compiling a predictive task.
A correlated or non-Gaussian observation leaves all three at
`NONE` / `None` / `False`: an ABSTAIN, which is a verdict, rather than a number
that was never earned.

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

`ArtifactKind` has four members -- `PLAN`, `RESULT`, `EVALUATION_REPORT` and
`ESTIMATOR` -- and the last is R2's single addition to R1's frozen schema. It is
the invalidation taxonomy and not a catalogue of artifacts: which of the five
Results a reference points at is `ResultKind`'s business, and asking this enum
to carry that as well would put five identical rows in the matrix below.
`ESTIMATOR` earns a row of its own because fitted weights are a separate thing
to go stale -- they are trained once, referenced by a
`FittedConditionalPosterior`, and outlive the run that made them.

`InvalidationPolicy.default()` encodes, per artifact category, which changed
slots invalidate the artifact. "Invalidate" produces an immutable, revision
`n + 1` copy marked INVALIDATED with the changed inputs and time recorded; it
never rewrites the old revision. ENVIRONMENT is in no row: a backend patch
leaves stored artifacts readable and gives the next run new provenance.

| Change | Plan | Result | Estimator | EvaluationReport/Gate |
|---|:---:|:---:|:---:|:---:|
| model source / graph structure | invalidate | invalidate | invalidate | invalidate |
| data / task | invalidate | invalidate | invalidate | invalidate |
| compilation | — | invalidate | invalidate | invalidate |
| evaluation threshold/grouping | — | reusable | reusable | invalidate |
| display option | reusable | reusable | reusable | reusable |
| backend patch/environment | readable; next run re-provenances | readable; next run re-provenances | readable; next run re-provenances | re-evaluated by new Result identity |

The `Estimator` column is `Result`'s, slot for slot: `_MODEL_AND_INPUTS` plus
`COMPILATION`. The narrower four-slot reading is the more accurate one -- an
estimator's weights do not depend on a block partition, a tolerance or a
fallback policy -- and the R2 close-out chose "same as RESULT" anyway, on the
ground that it errs toward retraining rather than toward silently reusing
weights across a changed compilation. That is a decision, not an oversight,
and `test_amortized_encoding.py`'s
`test_the_estimator_row_invalidates_on_model_graph_data_and_task` pins it: five
slots affect, `EVALUATION` and `ENVIRONMENT` do not.

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

## `report_kind`: the codes in circulation

An `EvaluationReport` names what kind of report it is with a **code string**,
not an enum. That is an R1 ruling, and it is what let R3 add seven kinds
without a schema version: a release that starts asking a new question costs no
migration and no stored artifact becomes unreadable.

What a code does *not* buy is silent tolerance. A gate declares the kinds it
aggregates, and a report of any other kind is refused by name rather than
filed -- see below.

Eight codes are written in this release. One is R2's, filed by the execution
layer beside a sampled posterior; the other seven are R3's, written by
`bayesmith.evaluation`:

| `report_kind` | Written by | Release |
|---|---|---|
| `chain_diagnostics` | `dispatch.task`, while it makes the posterior | R2 |
| `posterior_predictive_check` | `evaluation.checks` | R3 |
| `prior_predictive_check` | `evaluation.checks` | R3 |
| `held_out_prediction` | `evaluation.heldout` | R3 |
| `loo_psis` | `evaluation.loo` | R3 |
| `sbc` | `evaluation.sbc` | R3 |
| `identifiability` | `evaluation.diagnostics` | R3 |
| `prior_sensitivity` | `evaluation.diagnostics` | R3 |

Five of the eight are importable constants; three are spelled at their call
sites, and nothing routes on a spelling -- the gate looks a report's slot up by
the report's OWN `report_kind`, and a code it does not declare raises by name
rather than being dropped into a slot that would then read as unattempted.

What each code decides, on which of the two axes, and what each `PASS` does
**not** mean, is [`docs/evaluation.md`](evaluation.md). The `evaluation`
fingerprint slot above is what retires these reports: it covers report kind,
threshold, grouping, repeats, applicability policy and gate identity, so a
changed threshold invalidates the report without touching the Result it judged.

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
