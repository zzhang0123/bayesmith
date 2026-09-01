"""The seven fingerprint slots, and what a change to each one invalidates.

A fingerprint is not a hash of an object; it is a hash of a *decision about
what counts*. Plan §0.3 fixes seven slots and, for each, a boundary between
what belongs inside and what must never reach the digest -- so the tests that
matter here are the ones standing on that boundary rather than the ones
showing that hashing works:

* **one byte of an array is inside the data slot.** Not "approximately the
  same numbers": the digest is taken over bytes, so two arrays that plot alike
  are two data sets;
* **a progress label is outside the task slot**, and so is the task's own
  ``task_id``. A digest that moved when a progress bar was renamed would
  invalidate every downstream artifact for a cosmetic edit;
* **insertion order is outside every slot**, because a Python mapping
  remembers the order its keys arrived in and nothing about a model does;
* **``repr(callable)`` is outside the model-source slot** -- it carries a memory
  address, so a digest built from it would call one model two models in one
  process, and a rerun in a fresh one a third.

The invalidation matrix is the other half, and it is table-driven on purpose
(§0.3): "an evaluation threshold moved, so re-run the report and keep the
Result" is a row in a table, and an ``if`` chain that grows a branch per artifact
type is how a table acquires a case nobody wrote down.

**What the ``TaskLike`` stand-in does and does not prove.** ``fingerprint()``
hashes the payload it is handed, so *excluding* a display field is the
caller's decision rather than the digest's. Task 3 owns the real
``task_fingerprint``; until it lands, the stand-in below states the §0.3 task
boundary in the only place it currently exists, and when Task 3 arrives its
payload builder is what these mutants should be pointed at.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect

import numpy as np
import pytest

from bayesmith.artifacts._codec import (
    ArtifactCodecError,
    canonical_dumps,
    canonical_loads,
)
from bayesmith.artifacts.identity import (
    FINGERPRINT_ALGORITHM,
    ArtifactKind,
    Fingerprint,
    FingerprintBundle,
    FingerprintKind,
    InvalidationPolicy,
    ModelRef,
    changed_fingerprints,
    fingerprint,
    model_ref_from_callable,
)

K = FingerprintKind

#: A well-formed digest, for the cases that are about something else.
ZERO_DIGEST = "0" * 64


def model_builder(scale: float = 1.0) -> str:
    """A module-level callable, so ``inspect.getsource`` can find its text."""
    return f"scale={scale}"


def make_builder():
    """Return a fresh function object with the same source text every time.

    Two calls give two objects at two addresses and one source. That is the
    whole anti-``repr`` argument in one fixture: a source digest has to agree
    across the pair, and an address-derived one cannot.
    """

    def built(scale: float = 1.0) -> str:
        return f"scale={scale}"

    return built


def exec_defined():
    """A callable whose source ``inspect`` cannot recover -- no file holds it."""
    namespace: dict[str, object] = {}
    # The S102 waiver below is the fixture's whole point: a function that
    # exists without a file is exactly the case model_ref_from_callable must
    # refuse, and the code being executed is the literal on this line.
    exec(compile("def built(): return 1", "<generated>", "exec"), namespace)  # noqa: S102
    return namespace["built"]


@dataclasses.dataclass(frozen=True, slots=True)
class TaskLike:
    """A stand-in for the Task 3 tasks, with fields on both sides of §0.3.

    ``chains``/``draws``/``tolerance`` change what is computed; ``task_id`` is
    identity rather than semantics, and ``progress_label`` is display.
    """

    task_id: str
    progress_label: str
    chains: int
    draws: int
    tolerance: float

    def fingerprint_payload(self) -> dict[str, object]:
        # §0.3: the task slot carries statistical semantics and budget. Not
        # the progress bar, and not the identity the artifact was stamped with.
        return {
            "chains": self.chains,
            "draws": self.draws,
            "tolerance": self.tolerance,
        }


TASK = TaskLike(
    task_id="a2f0b9a1-0b2f-4c3d-8e4f-5a6b7c8d9e0f",
    progress_label="sampling",
    chains=4,
    draws=1000,
    tolerance=1e-6,
)


def model_ref() -> ModelRef:
    return model_ref_from_callable(
        model_builder,
        identifier="pilot",
        package="bayesmith",
        package_version="0.6.2",
        build_arguments=(("scale", 2.0),),
    )


def payloads() -> dict[str, object]:
    """One payload per slot, each shaped like what §0.3 says the slot holds."""
    return {
        "model_source": model_ref(),
        "graph_structure": (
            ("theta", "normal", (), "prior"),
            ("y", "normal", ("theta",), "likelihood"),
        ),
        "data": {"y": np.arange(6, dtype=np.float64).reshape(2, 3)},
        "task": TASK.fingerprint_payload(),
        "compilation": {"method": "gls", "tol": 1e-10, "blocks": ("theta",)},
        "evaluation": {"report": "rank", "threshold": 0.05},
        "environment": {"python": "3.11.9", "jax": "0.5.0", "x64": True},
    }


def make_bundle(**overrides: object) -> FingerprintBundle:
    """A full seven-slot bundle, with any slot's PAYLOAD replaced."""
    slots = payloads() | overrides
    return FingerprintBundle(
        **{name: fingerprint(K(name), value) for name, value in slots.items()}
    )


# ----------------------------------------------------------------- the digest


def test_a_fingerprint_is_sha256_over_the_canonical_bytes():
    payload = {"y": np.arange(4, dtype=np.int64)}
    taken = fingerprint(K.DATA, payload)
    assert taken.kind is K.DATA
    assert taken.algorithm == FINGERPRINT_ALGORITHM
    assert taken.digest == hashlib.sha256(canonical_dumps(payload)).hexdigest()


def test_one_flipped_byte_of_an_array_changes_the_data_digest():
    original = np.linspace(0.0, 1.0, 8)
    raw = bytearray(original.tobytes())
    raw[0] ^= 0x01
    mutant = np.frombuffer(bytes(raw), dtype=original.dtype).reshape(original.shape)

    # Close enough that every tolerance-based comparison calls them one array.
    assert np.allclose(original, mutant)
    assert not np.array_equal(original, mutant)
    assert fingerprint(K.DATA, original).digest != fingerprint(K.DATA, mutant).digest


def test_memory_layout_and_mapping_order_stay_out_of_a_digest():
    c_order = np.arange(6, dtype=np.float64).reshape(2, 3)
    f_order = np.asfortranarray(c_order)
    assert fingerprint(K.DATA, c_order).digest == fingerprint(K.DATA, f_order).digest

    first = {"alpha": 1, "beta": 2}
    second = {"beta": 2, "alpha": 1}
    assert list(first) != list(second)
    assert fingerprint(K.TASK, first).digest == fingerprint(K.TASK, second).digest


@pytest.mark.parametrize(
    ("field", "value", "moves"),
    [
        ("chains", 8, True),
        ("draws", 2000, True),
        ("tolerance", 1e-8, True),
        ("progress_label", "warming up", False),
        ("task_id", "ffffffff-0b2f-4c3d-8e4f-5a6b7c8d9e0f", False),
    ],
)
def test_the_task_digest_follows_semantics_and_ignores_display(field, value, moves):
    mutant = dataclasses.replace(TASK, **{field: value})
    assert getattr(mutant, field) != getattr(TASK, field)

    before = fingerprint(K.TASK, TASK.fingerprint_payload())
    after = fingerprint(K.TASK, mutant.fingerprint_payload())
    assert (before.digest != after.digest) is moves


@pytest.mark.parametrize(
    "digest",
    ["", "nothex" * 10, "AB" * 32, "ab" * 31, "ab" * 33, "ab" * 32 + " "],
)
def test_a_digest_must_be_lowercase_sha256_hex(digest):
    with pytest.raises(ValueError, match="digest"):
        Fingerprint(kind=K.DATA, algorithm=FINGERPRINT_ALGORITHM, digest=digest)


def test_a_fingerprint_names_its_algorithm_and_refuses_another():
    with pytest.raises(ValueError, match="algorithm"):
        Fingerprint(kind=K.DATA, algorithm="md5", digest=ZERO_DIGEST)


def test_a_bare_string_is_not_a_fingerprint_kind():
    # A StrEnum member compares equal to its value, so an isinstance check is
    # the only thing standing between the enum and any string at all.
    assert K.DATA == "data"
    with pytest.raises(TypeError, match="kind"):
        Fingerprint(kind="data", algorithm=FINGERPRINT_ALGORITHM, digest=ZERO_DIGEST)


# ----------------------------------------------------------------- the bundle


def test_the_bundle_has_exactly_the_seven_slots_the_kinds_name():
    slots = tuple(field.name for field in dataclasses.fields(FingerprintBundle))
    assert slots == (
        "model_source",
        "graph_structure",
        "data",
        "task",
        "compilation",
        "evaluation",
        "environment",
    )
    assert set(slots) == {kind.value for kind in FingerprintKind}


def test_the_last_three_slots_are_optional_and_the_first_four_are_not():
    slots = payloads()
    partial = FingerprintBundle(
        model_source=fingerprint(K.MODEL_SOURCE, slots["model_source"]),
        graph_structure=fingerprint(K.GRAPH_STRUCTURE, slots["graph_structure"]),
        data=fingerprint(K.DATA, slots["data"]),
        task=fingerprint(K.TASK, slots["task"]),
    )
    assert partial.compilation is None
    assert partial.evaluation is None
    assert partial.environment is None

    with pytest.raises(TypeError, match="required"):
        FingerprintBundle(
            model_source=fingerprint(K.MODEL_SOURCE, slots["model_source"]),
        )


def test_a_slot_must_hold_a_fingerprint_of_its_own_kind():
    slots = payloads()
    with pytest.raises(ValueError, match="data"):
        FingerprintBundle(
            model_source=fingerprint(K.MODEL_SOURCE, slots["model_source"]),
            graph_structure=fingerprint(K.GRAPH_STRUCTURE, slots["graph_structure"]),
            data=fingerprint(K.TASK, slots["data"]),
            task=fingerprint(K.TASK, slots["task"]),
        )


@pytest.mark.parametrize("slot", [kind.value for kind in FingerprintKind])
def test_changed_fingerprints_names_exactly_the_slot_that_moved(slot):
    before = make_bundle()
    after = make_bundle(**{slot: {"moved": slot}})
    assert changed_fingerprints(before, after) == frozenset({K(slot)})
    assert changed_fingerprints(after, before) == frozenset({K(slot)})


def test_an_optional_slot_appearing_or_vanishing_is_a_change():
    full = make_bundle()
    without = dataclasses.replace(full, compilation=None)
    assert changed_fingerprints(full, without) == frozenset({K.COMPILATION})
    assert changed_fingerprints(without, full) == frozenset({K.COMPILATION})
    assert changed_fingerprints(without, without) == frozenset()


def test_an_unchanged_bundle_changes_nothing():
    assert changed_fingerprints(make_bundle(), make_bundle()) == frozenset()


def test_changed_fingerprints_refuses_something_that_is_not_a_bundle():
    with pytest.raises(TypeError, match="bundle"):
        changed_fingerprints(make_bundle(), object())


# ------------------------------------------------------------ the model source


def test_model_ref_from_callable_digests_the_source():
    ref = model_ref_from_callable(model_builder, identifier="pilot")
    source = inspect.getsource(model_builder)
    assert ref.identifier == "pilot"
    assert ref.source_digest == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert ref.package == "tests"


def test_two_callables_with_one_source_have_one_source_digest():
    first = make_builder()
    second = make_builder()
    assert first is not second
    assert repr(first) != repr(second)  # what a repr-derived digest would see

    left = model_ref_from_callable(first, identifier="pilot")
    right = model_ref_from_callable(second, identifier="pilot")
    assert left.source_digest == right.source_digest
    assert fingerprint(K.MODEL_SOURCE, left) == fingerprint(K.MODEL_SOURCE, right)


def test_a_callable_without_recoverable_source_is_refused_not_repr_hashed():
    built = exec_defined()
    with pytest.raises(ValueError, match="source_digest"):
        model_ref_from_callable(built, identifier="generated")

    supplied = model_ref_from_callable(
        built, identifier="generated", source_digest="c0ffee"
    )
    assert supplied.source_digest == "c0ffee"


def test_a_builtin_is_refused_for_the_same_reason():
    with pytest.raises(ValueError, match="source_digest"):
        model_ref_from_callable(len, identifier="len")


def test_a_non_callable_is_not_a_model():
    with pytest.raises(TypeError, match="callable"):
        model_ref_from_callable("bayesmith.model", identifier="pilot")


def test_a_callable_has_no_canonical_payload_at_all():
    assert "0x" in repr(model_builder)
    with pytest.raises(ArtifactCodecError, match="canonical"):
        fingerprint(K.MODEL_SOURCE, model_builder)
    with pytest.raises(ArtifactCodecError, match="canonical"):
        ModelRef(
            identifier="pilot",
            source_digest=ZERO_DIGEST,
            build_arguments=(("prior", model_builder),),
        )


def test_a_model_ref_must_pin_its_source_somehow():
    with pytest.raises(ValueError, match="source_digest"):
        ModelRef(identifier="pilot")
    assert ModelRef(identifier="pilot", source_digest=ZERO_DIGEST).package is None
    pinned_by_release = ModelRef(
        identifier="pilot", package="bayesmith", package_version="0.6.2"
    )
    assert pinned_by_release.source_digest is None


def test_build_arguments_are_key_unique_order_independent_and_canonical():
    left = ModelRef(
        identifier="pilot",
        source_digest=ZERO_DIGEST,
        build_arguments=(("scale", 2.0), ("knots", 3)),
    )
    right = ModelRef(
        identifier="pilot",
        source_digest=ZERO_DIGEST,
        build_arguments=(("knots", 3), ("scale", 2.0)),
    )
    assert left == right
    assert left.build_arguments == (("knots", 3), ("scale", 2.0))

    with pytest.raises(ValueError, match="build_arguments"):
        ModelRef(
            identifier="pilot",
            source_digest=ZERO_DIGEST,
            build_arguments=(("scale", 2.0), ("scale", 3.0)),
        )


def test_a_model_ref_needs_an_identifier():
    with pytest.raises(ValueError, match="identifier"):
        ModelRef(identifier="", source_digest=ZERO_DIGEST)


# ------------------------------------------------------ the invalidation matrix

MATRIX = (
    (frozenset({K.MODEL_SOURCE}), True, True, True),
    (frozenset({K.GRAPH_STRUCTURE}), True, True, True),
    (frozenset({K.DATA}), True, True, True),
    (frozenset({K.TASK}), True, True, True),
    (frozenset({K.COMPILATION}), False, True, True),
    (frozenset({K.EVALUATION}), False, False, True),
    (frozenset({K.ENVIRONMENT}), False, False, False),
    (frozenset(), False, False, False),
    (frozenset({K.EVALUATION, K.COMPILATION}), False, True, True),
    (frozenset({K.ENVIRONMENT, K.DATA}), True, True, True),
)

MATRIX_IDS = tuple("+".join(sorted(row[0])) or "display-option" for row in MATRIX)


@pytest.mark.parametrize(
    ("changed", "plan", "result", "report"), MATRIX, ids=MATRIX_IDS
)
def test_default_invalidation_matrix(changed, plan, result, report):
    policy = InvalidationPolicy.default()
    assert policy.affected(ArtifactKind.PLAN, changed) is plan
    assert policy.affected(ArtifactKind.RESULT, changed) is result
    assert policy.affected(ArtifactKind.EVALUATION_REPORT, changed) is report


def test_the_matrix_table_covers_every_kind_on_both_axes():
    singles = {next(iter(row[0])) for row in MATRIX if len(row[0]) == 1}
    assert singles == set(FingerprintKind)
    assert {kind.value for kind in ArtifactKind} == {
        "plan",
        "result",
        "evaluation_report",
        "estimator",
    }


def test_only_the_report_is_invalidated_when_an_evaluation_threshold_moves():
    before = make_bundle()
    after = make_bundle(evaluation={"report": "rank", "threshold": 0.01})
    changed = changed_fingerprints(before, after)
    assert changed == frozenset({K.EVALUATION})

    policy = InvalidationPolicy.default()
    assert policy.affected(ArtifactKind.PLAN, changed) is False
    assert policy.affected(ArtifactKind.RESULT, changed) is False
    assert policy.affected(ArtifactKind.EVALUATION_REPORT, changed) is True


def test_a_display_only_change_moves_no_slot_and_invalidates_nothing():
    before = make_bundle()
    renamed = dataclasses.replace(TASK, progress_label="warming up")
    after = make_bundle(task=renamed.fingerprint_payload())

    changed = changed_fingerprints(before, after)
    assert changed == frozenset()
    policy = InvalidationPolicy.default()
    assert not any(policy.affected(kind, changed) for kind in ArtifactKind)


def test_a_compilation_change_spares_the_plan():
    before = make_bundle()
    after = make_bundle(compilation={"method": "nuts", "tol": None, "blocks": ()})
    changed = changed_fingerprints(before, after)
    assert changed == frozenset({K.COMPILATION})

    policy = InvalidationPolicy.default()
    assert policy.affected(ArtifactKind.PLAN, changed) is False
    assert policy.affected(ArtifactKind.RESULT, changed) is True


def test_the_policy_refuses_bare_strings_where_enums_are_meant():
    # Either of these would otherwise answer False rather than raise: an Enum
    # hashes by NAME, so neither the dict lookup nor the set intersection an
    # implementation performs would ever find the string spelling.
    policy = InvalidationPolicy.default()
    with pytest.raises(TypeError, match="artifact"):
        policy.affected("result", frozenset({K.DATA}))
    with pytest.raises(TypeError, match="fingerprint"):
        policy.affected(ArtifactKind.RESULT, frozenset({"data"}))


def test_the_policy_takes_any_iterable_of_kinds_and_ignores_its_order():
    policy = InvalidationPolicy.default()
    assert policy.affected(ArtifactKind.RESULT, {K.EVALUATION, K.COMPILATION}) is True
    assert policy.affected(ArtifactKind.RESULT, [K.COMPILATION, K.EVALUATION]) is True
    # An ITERATOR, read once: an implementation that materialised `changed`
    # twice would ask its second question of an exhausted one and answer
    # False -- which is indistinguishable from a correct 'unaffected'.
    assert policy.affected(ArtifactKind.RESULT, iter([K.COMPILATION])) is True
    assert policy.affected(ArtifactKind.RESULT, iter([K.EVALUATION])) is False


def test_a_policy_missing_a_row_is_refused():
    with pytest.raises(ValueError, match="every artifact kind"):
        InvalidationPolicy(sensitivities=((ArtifactKind.PLAN, (K.DATA,)),))


def test_a_policy_row_is_normalised_and_key_unique():
    policy = InvalidationPolicy.default()
    assert policy == InvalidationPolicy.default()
    rows = dict(policy.sensitivities)
    assert rows[ArtifactKind.PLAN] == tuple(
        sorted((K.DATA, K.GRAPH_STRUCTURE, K.MODEL_SOURCE, K.TASK))
    )
    assert K.ENVIRONMENT not in rows[ArtifactKind.EVALUATION_REPORT]

    with pytest.raises(ValueError, match="twice"):
        InvalidationPolicy(
            sensitivities=(
                (ArtifactKind.PLAN, (K.DATA,)),
                (ArtifactKind.PLAN, (K.TASK,)),
                (ArtifactKind.RESULT, (K.DATA,)),
                (ArtifactKind.EVALUATION_REPORT, (K.DATA,)),
            )
        )


# ------------------------------------------------------------------- the wire


def test_the_identity_types_round_trip_through_the_codec():
    ref = model_ref()
    assert canonical_loads(canonical_dumps(ref), expected=ModelRef) == ref

    taken = fingerprint(K.DATA, {"y": np.arange(3, dtype=np.int64)})
    assert canonical_loads(canonical_dumps(taken), expected=Fingerprint) == taken

    bundle = make_bundle()
    restored = canonical_loads(canonical_dumps(bundle), expected=FingerprintBundle)
    assert restored == bundle
    assert changed_fingerprints(restored, bundle) == frozenset()


def test_a_bundle_encodes_the_same_bytes_twice():
    assert canonical_dumps(make_bundle()) == canonical_dumps(make_bundle())
