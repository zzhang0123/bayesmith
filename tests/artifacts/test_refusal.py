"""A refusal is a structured verdict, not an apology (§0.6).

Plan §0 ruling 3 is the whole of this module's design. A method that does not
apply produces a ``Refusal`` carrying machine-readable ``grounds`` and machine-
readable ``remedies``; it never asks a caller to parse a sentence, and it never
calls its evidence ``evidence`` -- that field name is frozen OUT, because the
adapter in the sibling repository and this package must agree on one name for
the thing a consumer branches on.

**Both lists are non-empty on purpose.** A refusal with no grounds says "no"
and gives a caller nothing to act on; a refusal with no remedies says "no" and
gives them nowhere to go. Each is refused at construction, where the caller
who could still add them is standing.

**A refusal is not an exception in a costume.** A malformed Graph, a
programming error and an out-of-memory are exceptions and stay exceptions
(§0.6). What arrives here is a statistical judgement: the premise this route
needs is false, here is what was measured, here is what to try instead.

**The message is for a human and changes nothing.** ``failed_premise`` is a
stable code, a :class:``Finding`` carries the numbers, and rewriting either
message leaves both untouched -- the tests below pin that, because prose that
a consumer has started to parse becomes an interface nobody meant to publish.
"""

from __future__ import annotations

import dataclasses
import uuid

import pytest

from bayesmith.artifacts._codec import (
    ArtifactCodecError,
    canonical_dumps,
    canonical_loads,
)
from bayesmith.artifacts.base import (
    ArtifactRef,
    ComputeBudget,
    ProducerRef,
    new_artifact_meta,
)
from bayesmith.artifacts.identity import (
    ArtifactKind,
    FingerprintBundle,
    FingerprintKind,
    fingerprint,
)
from bayesmith.artifacts.refusal import (
    Conservatism,
    FallbackOption,
    Finding,
    Refusal,
    Remedy,
    ScopeKind,
    ScopeRef,
)
from bayesmith.artifacts.results import result_kind
from bayesmith.artifacts.tasks import (
    PosteriorTask,
    TaskKind,
    new_task_meta,
)

K = FingerprintKind


def bundle() -> FingerprintBundle:
    return FingerprintBundle(
        **{
            name: fingerprint(K(name), value)
            for name, value in (
                ("model_source", "pilot"),
                ("graph_structure", "theta -> y"),
                ("data", "y=[1,2,3]"),
                ("task", "posterior"),
            )
        }
    )


def ref(kind: ArtifactKind = ArtifactKind.RESULT) -> ArtifactRef:
    return ArtifactRef(artifact_id=str(uuid.uuid4()), revision=0, artifact_type=kind)


def meta(**overrides: object):
    fields: dict[str, object] = {
        "artifact_type": ArtifactKind.RESULT,
        "fingerprints": bundle(),
        "producer": ProducerRef(package="bayesmith", version="0.6.2"),
        "summary": "the exact route does not apply",
    }
    fields.update(overrides)
    return new_artifact_meta(**fields)


def task() -> PosteriorTask:
    return PosteriorTask(
        meta=new_task_meta(label="fit theta"),
        backend="auto",
        budget=ComputeBudget(draws=4),
    )


def finding(**overrides: object) -> Finding:
    fields: dict[str, object] = {
        "code": "not_gaussian",
        "message": "the likelihood of y is Student-t, so the exact route needs a "
        "Gaussian it does not have",
        "observed": "student_t",
        "expected": "gaussian",
        "artifact_refs": (),
    }
    fields.update(overrides)
    return Finding(**fields)


def remedy(**overrides: object) -> Remedy:
    fields: dict[str, object] = {
        "action": "use_nuts",
        "message": "run the same task on the NUTS backend",
        "parameters": (("backend", "numpyro"), ("draws", 1000)),
    }
    fields.update(overrides)
    return Remedy(**fields)


def scope(**overrides: object) -> ScopeRef:
    fields: dict[str, object] = {"kind": ScopeKind.NODE, "name": "y"}
    fields.update(overrides)
    return ScopeRef(**fields)


def fallback(**overrides: object) -> FallbackOption:
    fields: dict[str, object] = {
        "task_kind": TaskKind.POSTERIOR,
        "backend": "numpyro",
        "conservatism": Conservatism.MORE_CONSERVATIVE,
        "automatic": False,
    }
    fields.update(overrides)
    return FallbackOption(**fields)


def refusal(**overrides: object) -> Refusal:
    fields: dict[str, object] = {
        "meta": meta(),
        "task": task(),
        "failed_premise": "gaussian_likelihood",
        "grounds": (finding(),),
        "scope": scope(),
        "remedies": (remedy(),),
        "fallback": fallback(),
    }
    fields.update(overrides)
    return Refusal(**fields)


# ------------------------------------------------------------- the frozen shape


def test_a_refusal_has_grounds_and_has_no_evidence():
    """§0 ruling 3, as the two assertions the ruling actually makes."""
    names = {field.name for field in dataclasses.fields(Refusal)}
    assert "grounds" in names
    assert "evidence" not in names


def test_the_refusal_shape_is_the_one_the_plan_froze():
    """Every field named, so that a new one has to be decided on rather than
    added. A refusal is what two repositories agree about; a field that
    appeared here without a ruling would appear in neither's translation."""
    assert [field.name for field in dataclasses.fields(Refusal)] == [
        "meta",
        "task",
        "failed_premise",
        "grounds",
        "scope",
        "remedies",
        "fallback",
    ]


def test_grounds_may_not_be_empty():
    with pytest.raises(ValueError, match="grounds"):
        refusal(grounds=())


def test_remedies_may_not_be_empty():
    with pytest.raises(ValueError, match="remedies"):
        refusal(remedies=())


def test_a_refusal_is_not_a_result():
    """§0 ruling 1: a refusal stands in FOR a result and is never mistaken for
    one, so the result taxonomy refuses it rather than defaulting."""
    with pytest.raises(TypeError):
        result_kind(refusal())


# ---------------------------------------------------------------- round trips


def test_a_refusal_round_trips_whole():
    original = refusal()
    restored = canonical_loads(canonical_dumps(original), expected=Refusal)
    assert restored == original
    for field in dataclasses.fields(original):
        assert getattr(restored, field.name) == getattr(original, field.name)


@pytest.mark.parametrize(
    "builder", [scope, remedy, finding, fallback], ids=lambda f: f.__name__
)
def test_each_refusal_part_round_trips_on_its_own(builder):
    original = builder()
    assert canonical_loads(canonical_dumps(original), expected=type(original)) == (
        original
    )


def test_a_scope_names_a_kind_and_a_thing():
    assert scope().kind is ScopeKind.NODE
    assert scope().name == "y"
    with pytest.raises(TypeError):
        ScopeRef(kind="node", name="y")


def test_a_fallback_names_the_task_it_would_run_and_how_conservative_it_is():
    """A fallback that is LESS conservative has to say so: 'we fell back' reads
    as 'we were careful', and sometimes the cheaper route is the one being
    offered."""
    assert fallback().task_kind is TaskKind.POSTERIOR
    assert set(Conservatism) == {
        Conservatism.EQUIVALENT,
        Conservatism.MORE_CONSERVATIVE,
        Conservatism.LESS_CONSERVATIVE,
    }
    with pytest.raises(TypeError):
        fallback(conservatism="more_conservative")


def test_a_refusal_may_offer_no_fallback():
    """Not every refusal has somewhere to send the caller; ``None`` says so
    rather than a fallback nobody would take."""
    assert refusal(fallback=None).fallback is None


# ------------------------------------------------- prose changes nothing that matters


def test_rewriting_the_messages_leaves_the_codes_and_the_scope_alone():
    """The property a consumer relies on: it branches on codes, so an editor
    improving the English must not be able to change what it does."""
    original = refusal()
    rewritten = dataclasses.replace(
        original,
        grounds=tuple(
            dataclasses.replace(item, message="Reworded for the release notes.")
            for item in original.grounds
        ),
        remedies=tuple(
            dataclasses.replace(item, message="Try this instead, please.")
            for item in original.remedies
        ),
    )
    assert rewritten.failed_premise == original.failed_premise
    assert rewritten.scope == original.scope
    assert [item.code for item in rewritten.grounds] == [
        item.code for item in original.grounds
    ]
    assert [item.observed for item in rewritten.grounds] == [
        item.observed for item in original.grounds
    ]
    assert [item.action for item in rewritten.remedies] == [
        item.action for item in original.remedies
    ]
    assert rewritten != original


def test_a_code_is_a_code_and_not_a_sentence():
    """A code with a space in it is prose that a consumer will end up
    comparing against a string literal it wrote by eye."""
    with pytest.raises(ValueError, match="code"):
        finding(code="not a gaussian")
    with pytest.raises(ValueError, match="action"):
        remedy(action="use NUTS")
    with pytest.raises(ValueError, match="failed_premise"):
        refusal(failed_premise="gaussian likelihood")


def test_a_finding_carries_its_numbers_rather_than_rendering_them():
    """The G11 lesson, in the class this package refuses through: the failing
    path must return what the passing path measured, not a sentence about it."""
    measured = finding(code="condition_number", observed=1.5e12, expected=1e8)
    assert measured.observed == 1.5e12
    assert measured.expected == 1e8
    restored = canonical_loads(canonical_dumps(measured), expected=Finding)
    assert restored.observed == 1.5e12


# -------------------------------------------------------- malformed is refused


def test_a_message_is_required_where_a_human_has_to_read_one():
    with pytest.raises(ValueError):
        finding(message="")
    with pytest.raises(ValueError):
        remedy(message="")


def test_a_finding_holds_canonical_values_and_never_a_runtime_object():
    with pytest.raises(TypeError):
        finding(observed=lambda: None)
    with pytest.raises(TypeError):
        remedy(parameters=(("callback", lambda: None),))


def test_the_task_a_refusal_carries_is_one_of_the_five():
    with pytest.raises(TypeError):
        refusal(task="a posterior, please")


def test_a_refusal_stands_in_for_a_plan_or_a_result_and_not_for_a_verdict():
    """§0 ruling 1: a Report does not replace the main Result, and a refusal is
    not a report -- it is the plan or the result that did not happen."""
    assert refusal(meta=meta(artifact_type=ArtifactKind.PLAN)).meta.artifact_type is (
        ArtifactKind.PLAN
    )
    with pytest.raises(ValueError, match="evaluation_report"):
        refusal(meta=meta(artifact_type=ArtifactKind.EVALUATION_REPORT))


def test_a_refusal_payload_cannot_be_forged_into_another_class():
    payload = canonical_dumps(refusal())
    forged = payload.replace(
        b"bayesmith.artifacts.refusal.Refusal",
        b"bayesmith.artifacts.refusal.Rejection",
    )
    with pytest.raises(ArtifactCodecError, match="not a registered artifact type"):
        canonical_loads(forged)
