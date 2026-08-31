"""The artifact envelope: what every artifact carries, and how it is retired.

Plan §0.2 fixes the envelope; the invariants tested here are the ones that
make it worth having rather than the ones that make it compile.

**An artifact is not a handle on the thing it describes.** A caller who hands
in an array and then edits that array must not have edited the artifact, so
:class:``NamedArray`` copies on the way in and hands back a read-only view. The
copy and the read-only flag are two separate defects if either is missing: the
first lets a caller change history from outside, the second from inside.

**Identity and content are different axes.** ``artifact_id`` is a UUID4 minted
once; the fingerprint bundle says what the artifact was made of. So a version
is ``(artifact_id, revision)``, and invalidation APPENDS revision ``n + 1`` with
the changed inputs recorded rather than editing revision ``n`` -- history that
can be rewritten is not provenance. The tests below check the original object
after the invalidation, because an in-place mutation would leave every other
assertion here passing.

**Two enums that are both spelled "exact".** ``ApproximationClass.EXACT`` and
``TargetFidelity.EXACT`` have the same string value and answer different
questions -- §0.2 splits them exactly because iid exact-linear draws are
``MONTE_CARLO`` in representation and ``EXACT`` in target. A field validated with
``isinstance(x, str)`` would accept them swapped and lose the distinction the
split was made for, so the swap is tested directly.

**Time has one spelling.** All times are UTC RFC 3339. A naive timestamp is a
different instant on every machine, and ``+02:00`` is a reader's timezone
rather than the artifact's, so both are refused; ``+00:00`` is the same instant
as ``Z`` and is normalised to it rather than kept as a second spelling.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import uuid

import numpy as np
import pytest

from bayesmith.artifacts._codec import (
    ArtifactCodecError,
    canonical_dumps,
    canonical_loads,
)
from bayesmith.artifacts.base import (
    SCHEMA_VERSION,
    ApproximationClass,
    ApproximationRecord,
    ArtifactMeta,
    ArtifactRef,
    ArtifactStatus,
    BackendRef,
    ComputeBudget,
    DeviceRecord,
    ErrorRecord,
    LifecycleRecord,
    NamedArray,
    ProducerRef,
    RunRecord,
    RunWarning,
    SeedRecord,
    TargetFidelity,
    TerminationReason,
    TerminationRecord,
    TimingRecord,
    canonical_scalar_options,
    canonical_value_options,
    invalidate_meta,
    new_artifact_meta,
    utc_timestamp,
)
from bayesmith.artifacts.identity import (
    ArtifactKind,
    FingerprintBundle,
    FingerprintKind,
    InvalidationPolicy,
    fingerprint,
)

K = FingerprintKind

START = "2026-08-30T12:00:00Z"
FINISH = "2026-08-30T12:00:09.500000Z"


def bundle(**overrides: object) -> FingerprintBundle:
    payloads: dict[str, object] = {
        "model_source": "pilot",
        "graph_structure": "theta -> y",
        "data": "y=[1,2,3]",
        "task": "posterior",
        "compilation": "gls",
        "evaluation": "rank",
        "environment": "py3.11",
    }
    payloads.update(overrides)
    return FingerprintBundle(
        **{name: fingerprint(K(name), value) for name, value in payloads.items()}
    )


def producer() -> ProducerRef:
    return ProducerRef(package="bayesmith", version="0.6.2")


def meta(**overrides: object) -> ArtifactMeta:
    fields: dict[str, object] = {
        "artifact_type": ArtifactKind.RESULT,
        "fingerprints": bundle(),
        "producer": producer(),
        "summary": "a posterior over theta",
    }
    fields.update(overrides)
    return new_artifact_meta(**fields)


def plan_ref() -> ArtifactRef:
    return ArtifactRef(
        artifact_id=str(uuid.uuid4()),
        revision=0,
        artifact_type=ArtifactKind.PLAN,
    )


def run_record(**overrides: object) -> RunRecord:
    fields: dict[str, object] = {
        "run_id": str(uuid.uuid4()),
        "plan_ref": plan_ref(),
        "fingerprints": bundle(),
        "seed": SeedRecord(seed=0, key_algorithm="threefry2x32"),
        "dtype": "float64",
        "devices": (
            DeviceRecord(platform="cpu", device_kind="CPU", device_id=0),
        ),
        "jax_config": (("jax_enable_x64", True),),
        "backend": BackendRef(name="numpyro", version="0.15.0"),
        "budget": ComputeBudget(draws=1000, warmup=500, chains=4),
        "termination": TerminationRecord(reason=TerminationReason.COMPLETED),
        "timing": TimingRecord(
            started_at=START, finished_at=FINISH, wall_clock_seconds=9.5
        ),
        "approximation": ApproximationRecord(
            representation_class=ApproximationClass.MONTE_CARLO,
            target_fidelity=TargetFidelity.EXACT,
        ),
        "warnings": (),
    }
    fields.update(overrides)
    return RunRecord(**fields)


# -------------------------------------------------------------- NamedArray


def test_a_named_array_copies_its_input_and_hands_back_a_read_only_view():
    source = np.array([1.0, 2.0, 3.0])
    named = NamedArray(name="theta", value=source, dims=("draw",))

    source[0] = 99.0
    assert named.value[0] == 1.0

    with pytest.raises(ValueError, match="read-only"):
        named.value[0] = 7.0


def test_dims_must_count_the_axes():
    value = np.zeros((2, 3))
    assert NamedArray(name="theta", value=value, dims=("chain", "draw")).dims == (
        "chain",
        "draw",
    )
    with pytest.raises(ValueError, match="dims"):
        NamedArray(name="theta", value=value, dims=("draw",))
    with pytest.raises(ValueError, match="dims"):
        NamedArray(name="theta", value=value, dims=("chain", "draw", "extra"))
    with pytest.raises(ValueError, match="dims"):
        NamedArray(name="theta", value=value, dims=("chain", ""))


def test_a_dim_name_may_repeat_because_a_covariance_has_two_of_one_axis():
    covariance = NamedArray(
        name="covariance", value=np.eye(3), dims=("parameter", "parameter")
    )
    assert covariance.dims == ("parameter", "parameter")


def test_a_named_array_normalises_layout_and_survives_the_wire():
    fortran = np.asfortranarray(np.arange(6, dtype=np.float64).reshape(2, 3))
    named = NamedArray(name="x", value=fortran, dims=("row", "column"))
    assert named.value.flags["C_CONTIGUOUS"]

    restored = canonical_loads(canonical_dumps(named), expected=NamedArray)
    assert restored == named
    assert restored.value.dtype == named.value.dtype
    assert not restored.value.flags["WRITEABLE"]


def test_named_arrays_compare_by_their_bytes_including_nan():
    left = NamedArray(name="x", value=np.array([np.nan, 1.0]), dims=("draw",))
    right = NamedArray(name="x", value=np.array([np.nan, 1.0]), dims=("draw",))
    assert left == right
    assert hash(left) == hash(right)

    assert left != NamedArray(name="y", value=np.array([np.nan, 1.0]), dims=("draw",))
    assert left != NamedArray(name="x", value=np.array([np.nan, 2.0]), dims=("draw",))
    assert left != NamedArray(
        name="x", value=np.array([np.nan, 1.0], dtype=np.float32), dims=("draw",)
    )
    assert left != "x"


def test_an_object_array_and_an_ndarray_subclass_are_refused():
    class Sub(np.ndarray):
        pass

    # Refused by the CODEC's dtype whitelist, which is the only one: an
    # object array is arbitrary Python wearing a shape.
    with pytest.raises(ArtifactCodecError, match="not canonical"):
        NamedArray(name="x", value=np.array([object()], dtype=object), dims=("i",))
    with pytest.raises(TypeError, match="subclass"):
        NamedArray(name="x", value=np.zeros(3).view(Sub), dims=("i",))


def test_a_named_array_needs_a_name():
    with pytest.raises(ValueError, match="name"):
        NamedArray(name="", value=np.zeros(3), dims=("i",))


# ------------------------------------------------------------------- time


def test_utc_timestamp_is_rfc3339_in_the_z_spelling():
    stamp = utc_timestamp()
    assert stamp.endswith("Z")
    assert dt.datetime.fromisoformat(stamp).tzinfo is not None

    fixed = dt.datetime(2026, 8, 30, 12, 0, tzinfo=dt.UTC)
    assert utc_timestamp(fixed) == "2026-08-30T12:00:00Z"


def test_a_timestamp_must_be_utc_and_has_one_spelling():
    with pytest.raises(ValueError, match="UTC"):
        meta(created_at="2026-08-30T12:00:00")
    with pytest.raises(ValueError, match="UTC"):
        meta(created_at="2026-08-30T12:00:00+02:00")
    with pytest.raises(ValueError, match="RFC 3339"):
        meta(created_at="last tuesday")

    normalised = meta(created_at="2026-08-30T12:00:00+00:00")
    assert normalised.created_at == "2026-08-30T12:00:00Z"


# ---------------------------------------------------------------- options


def test_scalar_options_are_sorted_key_unique_and_scalar():
    assert canonical_scalar_options(
        "jax_config", (("b", 1), ("a", True))
    ) == (("a", True), ("b", 1))

    with pytest.raises(ValueError, match="twice"):
        canonical_scalar_options("jax_config", (("a", 1), ("a", 2)))
    with pytest.raises(TypeError, match="scalar"):
        canonical_scalar_options("jax_config", (("a", (1, 2)),))
    with pytest.raises(TypeError, match="scalar"):
        canonical_scalar_options("jax_config", (("a", np.zeros(2)),))


def test_value_options_admit_the_nested_tuples_scalar_options_refuse():
    assert canonical_value_options(
        "backend_options", (("shape", (2, 3)), ("names", ("a", ("b",))))
    ) == (("names", ("a", ("b",))), ("shape", (2, 3)))

    with pytest.raises(TypeError, match="canonical"):
        canonical_value_options("backend_options", (("a", np.zeros(2)),))
    with pytest.raises(TypeError, match="pairs"):
        canonical_value_options("backend_options", ("a", "b"))


# --------------------------------------------------------- the small records


def test_a_budget_counts_no_negatives_and_no_booleans():
    empty = ComputeBudget()
    assert empty.draws is None and empty.max_wall_clock_seconds is None
    assert ComputeBudget(draws=0).draws == 0

    with pytest.raises(ValueError, match="draws"):
        ComputeBudget(draws=-1)
    with pytest.raises(TypeError, match="chains"):
        ComputeBudget(chains=True)
    with pytest.raises(ValueError, match="max_wall_clock_seconds"):
        ComputeBudget(max_wall_clock_seconds=-0.5)
    with pytest.raises(ValueError, match="max_wall_clock_seconds"):
        ComputeBudget(max_wall_clock_seconds=float("inf"))


def test_a_duration_is_finite_and_non_negative_and_time_moves_forward():
    with pytest.raises(ValueError, match="wall_clock_seconds"):
        TimingRecord(started_at=START, finished_at=FINISH, wall_clock_seconds=-1.0)
    with pytest.raises(ValueError, match="wall_clock_seconds"):
        TimingRecord(
            started_at=START, finished_at=FINISH, wall_clock_seconds=float("nan")
        )
    with pytest.raises(ValueError, match="compile_seconds"):
        TimingRecord(
            started_at=START,
            finished_at=FINISH,
            wall_clock_seconds=1.0,
            compile_seconds=-1.0,
        )
    with pytest.raises(ValueError, match="before it started"):
        TimingRecord(started_at=FINISH, finished_at=START, wall_clock_seconds=1.0)


def test_a_backend_ref_records_what_ran_not_what_was_asked_for():
    assert BackendRef(name="numpyro").version is None
    with pytest.raises(ValueError, match="auto"):
        BackendRef(name="auto")
    with pytest.raises(ValueError, match="name"):
        BackendRef(name="")


def test_a_seed_and_a_device_carry_non_negative_identity():
    assert SeedRecord(seed=0, key_algorithm="threefry2x32").seed == 0
    with pytest.raises(ValueError, match="seed"):
        SeedRecord(seed=-1, key_algorithm="threefry2x32")
    with pytest.raises(ValueError, match="key_algorithm"):
        SeedRecord(seed=1, key_algorithm="")
    with pytest.raises(ValueError, match="device_id"):
        DeviceRecord(platform="cpu", device_kind="CPU", device_id=-1)
    with pytest.raises(ValueError, match="platform"):
        DeviceRecord(platform="", device_kind="CPU", device_id=0)


def test_a_termination_reason_must_be_the_enum_and_iterations_non_negative():
    record = TerminationRecord(reason=TerminationReason.CONVERGED, iterations=12)
    assert record.reason is TerminationReason.CONVERGED

    with pytest.raises(TypeError, match="reason"):
        TerminationRecord(reason="converged")
    with pytest.raises(ValueError, match="iterations"):
        TerminationRecord(reason=TerminationReason.CONVERGED, iterations=-1)


def test_the_two_approximation_axes_are_independent():
    # §0.2: iid exact-linear draws. Monte Carlo in representation, exact in
    # target -- the whole reason these are two enums and not one.
    iid = ApproximationRecord(
        representation_class=ApproximationClass.MONTE_CARLO,
        target_fidelity=TargetFidelity.EXACT,
    )
    assert iid.representation_class is ApproximationClass.MONTE_CARLO
    assert iid.target_fidelity is TargetFidelity.EXACT

    # Both members are the string "exact", so a str check accepts the swap.
    assert ApproximationClass.EXACT == TargetFidelity.EXACT
    with pytest.raises(TypeError, match="representation_class"):
        ApproximationRecord(
            representation_class=TargetFidelity.EXACT,
            target_fidelity=TargetFidelity.EXACT,
        )
    with pytest.raises(TypeError, match="target_fidelity"):
        ApproximationRecord(
            representation_class=ApproximationClass.EXACT,
            target_fidelity=ApproximationClass.EXACT,
        )
    with pytest.raises(TypeError, match="representation_class"):
        ApproximationRecord(
            representation_class="monte_carlo", target_fidelity=TargetFidelity.EXACT
        )


def test_approximation_details_are_canonical_scalars():
    record = ApproximationRecord(
        representation_class=ApproximationClass.HEURISTIC,
        target_fidelity=TargetFidelity.APPROXIMATE,
        details=(("tol", 1e-6), ("family", "normal")),
    )
    assert record.details == (("family", "normal"), ("tol", 1e-6))
    with pytest.raises(TypeError, match="scalar"):
        ApproximationRecord(
            representation_class=ApproximationClass.HEURISTIC,
            target_fidelity=TargetFidelity.APPROXIMATE,
            details=(("draws", np.zeros(2)),),
        )


def test_a_warning_and_an_error_record_need_a_code():
    warned = RunWarning(code="ess_below_floor", message="ESS 42 < 400")
    assert warned.scope is None
    with pytest.raises(ValueError, match="code"):
        RunWarning(code="", message="ESS 42 < 400")

    failure = ErrorRecord(
        code="report_failed", message="rank statistic raised", exception_type="ValueError"
    )
    assert failure.traceback_ref is None
    with pytest.raises(ValueError, match="exception_type"):
        ErrorRecord(code="report_failed", message="raised", exception_type="")


# ------------------------------------------------------------ refs and meta


def test_an_artifact_ref_carries_a_uuid4_a_revision_and_a_kind():
    ref = plan_ref()
    assert ref.revision == 0
    assert ref.artifact_type is ArtifactKind.PLAN

    with pytest.raises(ValueError, match="artifact_id"):
        ArtifactRef(artifact_id="plan-1", revision=0, artifact_type=ArtifactKind.PLAN)
    with pytest.raises(ValueError, match="artifact_id"):
        ArtifactRef(
            artifact_id=str(uuid.uuid1()), revision=0, artifact_type=ArtifactKind.PLAN
        )
    with pytest.raises(ValueError, match="revision"):
        ArtifactRef(
            artifact_id=str(uuid.uuid4()), revision=-1, artifact_type=ArtifactKind.PLAN
        )
    with pytest.raises(TypeError, match="artifact"):
        ArtifactRef(artifact_id=str(uuid.uuid4()), revision=0, artifact_type="plan")


def test_a_current_lifecycle_carries_no_invalidation():
    current = LifecycleRecord(status=ArtifactStatus.CURRENT)
    assert current.invalidated_at is None
    assert current.changed_inputs == ()

    with pytest.raises(ValueError, match="current"):
        LifecycleRecord(status=ArtifactStatus.CURRENT, invalidated_at=START)
    with pytest.raises(ValueError, match="current"):
        LifecycleRecord(status=ArtifactStatus.CURRENT, changed_inputs=(K.DATA,))


def test_an_invalidated_lifecycle_says_when_and_why():
    with pytest.raises(ValueError, match="invalidated_at"):
        LifecycleRecord(status=ArtifactStatus.INVALIDATED, changed_inputs=(K.DATA,))
    with pytest.raises(ValueError, match="changed_inputs"):
        LifecycleRecord(status=ArtifactStatus.INVALIDATED, invalidated_at=START)

    record = LifecycleRecord(
        status=ArtifactStatus.INVALIDATED,
        invalidated_at=START,
        changed_inputs=(K.TASK, K.DATA),
    )
    assert record.changed_inputs == (K.DATA, K.TASK)

    with pytest.raises(ValueError, match="twice"):
        LifecycleRecord(
            status=ArtifactStatus.INVALIDATED,
            invalidated_at=START,
            changed_inputs=(K.DATA, K.DATA),
        )


def test_new_artifact_meta_mints_a_uuid4_at_revision_zero():
    first = meta()
    second = meta()
    assert first.artifact_id != second.artifact_id
    assert uuid.UUID(first.artifact_id).version == 4
    assert first.revision == 0
    assert first.schema_version == SCHEMA_VERSION
    assert first.lifecycle.status is ArtifactStatus.CURRENT
    assert first.created_at.endswith("Z")


def test_a_meta_round_trips_its_uuid_and_created_at():
    original = meta()
    restored = canonical_loads(canonical_dumps(original), expected=ArtifactMeta)
    assert restored == original
    assert restored.artifact_id == original.artifact_id
    assert restored.created_at == original.created_at


def test_a_schema_version_must_be_a_positive_int():
    with pytest.raises(ValueError, match="schema_version"):
        meta(schema_version=0)
    with pytest.raises(ValueError, match="schema_version"):
        meta(schema_version=-1)
    # A bool IS an int, and version True is version 1 by accident.
    with pytest.raises(TypeError, match="schema_version"):
        meta(schema_version=True)
    with pytest.raises(TypeError, match="schema_version"):
        meta(schema_version="1")


def test_a_parent_cannot_be_listed_twice():
    parent = plan_ref()
    assert meta(parent_refs=(parent,)).parent_refs == (parent,)

    with pytest.raises(ValueError, match="parent"):
        meta(parent_refs=(parent, parent))

    # The same artifact at two revisions is the same ambiguity wearing a hat:
    # which version was this made from?
    with pytest.raises(ValueError, match="parent"):
        meta(parent_refs=(parent, dataclasses.replace(parent, revision=1)))


def test_an_invalidated_meta_cannot_be_revision_zero():
    with pytest.raises(ValueError, match="revision"):
        ArtifactMeta(
            artifact_type=ArtifactKind.RESULT,
            schema_version=SCHEMA_VERSION,
            artifact_id=str(uuid.uuid4()),
            revision=0,
            created_at=START,
            producer=producer(),
            parent_refs=(),
            fingerprints=bundle(),
            lifecycle=LifecycleRecord(
                status=ArtifactStatus.INVALIDATED,
                invalidated_at=START,
                changed_inputs=(K.DATA,),
            ),
            warnings=(),
            summary="",
        )


# ------------------------------------------------------------- invalidation


def test_invalidate_meta_appends_a_revision_and_leaves_the_original_alone():
    before = bundle()
    original = meta(fingerprints=before)
    after = bundle(data="y=[1,2,4]")

    retired = invalidate_meta(
        original,
        before=before,
        after=after,
        policy=InvalidationPolicy.default(),
        at=START,
    )

    assert retired.artifact_id == original.artifact_id
    assert retired.revision == original.revision + 1
    assert retired.lifecycle.status is ArtifactStatus.INVALIDATED
    assert retired.lifecycle.invalidated_at == START
    assert retired.lifecycle.changed_inputs == (K.DATA,)
    # The retired copy records what it was MADE from, not what replaced it.
    assert retired.fingerprints == before

    # Everything else is the same artifact: the retired revision keeps the
    # creation time and the lineage it was made with, and records the
    # invalidation time separately.
    assert retired.created_at == original.created_at
    assert retired.parent_refs == original.parent_refs
    assert retired.producer == original.producer

    # Append-only: revision 0 is untouched.
    assert original.revision == 0
    assert original.lifecycle.status is ArtifactStatus.CURRENT
    assert original.lifecycle.invalidated_at is None


def test_invalidate_meta_stamps_the_time_itself_when_not_given_one():
    before = bundle()
    retired = invalidate_meta(
        meta(fingerprints=before),
        before=before,
        after=bundle(data="moved"),
        policy=InvalidationPolicy.default(),
    )
    assert retired.lifecycle.invalidated_at.endswith("Z")


def test_invalidate_meta_refuses_a_change_the_policy_does_not_propagate():
    before = bundle()
    result = meta(artifact_type=ArtifactKind.RESULT, fingerprints=before)
    after = bundle(evaluation="rank at 0.01")

    with pytest.raises(ValueError, match="evaluation"):
        invalidate_meta(
            result, before=before, after=after, policy=InvalidationPolicy.default()
        )

    report = meta(artifact_type=ArtifactKind.EVALUATION_REPORT, fingerprints=before)
    retired = invalidate_meta(
        report, before=before, after=after, policy=InvalidationPolicy.default()
    )
    assert retired.lifecycle.changed_inputs == (K.EVALUATION,)


def test_invalidate_meta_refuses_a_before_that_is_not_the_artifacts_own():
    before = bundle()
    original = meta(fingerprints=before)
    with pytest.raises(ValueError, match="own"):
        invalidate_meta(
            original,
            before=bundle(data="somebody else's data"),
            after=bundle(data="moved"),
            policy=InvalidationPolicy.default(),
        )


def test_invalidate_meta_refuses_when_nothing_changed():
    before = bundle()
    with pytest.raises(ValueError, match="nothing"):
        invalidate_meta(
            meta(fingerprints=before),
            before=before,
            after=bundle(),
            policy=InvalidationPolicy.default(),
        )


def test_an_artifact_is_invalidated_once():
    before = bundle()
    retired = invalidate_meta(
        meta(fingerprints=before),
        before=before,
        after=bundle(data="moved"),
        policy=InvalidationPolicy.default(),
    )
    with pytest.raises(ValueError, match="already"):
        invalidate_meta(
            retired,
            before=before,
            after=bundle(data="moved again"),
            policy=InvalidationPolicy.default(),
        )


def test_the_invalidated_copy_round_trips():
    before = bundle()
    retired = invalidate_meta(
        meta(fingerprints=before),
        before=before,
        after=bundle(task="a different task"),
        policy=InvalidationPolicy.default(),
    )
    assert canonical_loads(canonical_dumps(retired), expected=ArtifactMeta) == retired


# ---------------------------------------------------------------- RunRecord


def test_a_run_record_round_trips_and_pins_the_plan_it_ran():
    record = run_record()
    assert canonical_loads(canonical_dumps(record), expected=RunRecord) == record

    with pytest.raises(ValueError, match="plan"):
        run_record(
            plan_ref=ArtifactRef(
                artifact_id=str(uuid.uuid4()),
                revision=0,
                artifact_type=ArtifactKind.RESULT,
            )
        )


def test_a_run_record_refuses_a_repeated_device_and_a_nested_config_value():
    device = DeviceRecord(platform="cpu", device_kind="CPU", device_id=0)
    with pytest.raises(ValueError, match="devices"):
        run_record(devices=(device, device))
    with pytest.raises(TypeError, match="scalar"):
        run_record(jax_config=(("shape", (2, 3)),))
    with pytest.raises(ValueError, match="run_id"):
        run_record(run_id="run-1")


def test_a_run_record_may_have_no_seed_and_no_device():
    record = run_record(seed=None, devices=())
    assert record.seed is None
    assert record.devices == ()
