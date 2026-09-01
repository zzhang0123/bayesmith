"""Two axes, one truth table, and an answer that cannot depend on order.

§0 ruling 7 and §0.6. The two things this module pins are the two that a gate
gets wrong quietly.

**A status and a verdict are different questions.** ``BLOCKED``,
``INVALIDATED`` and ``ERROR`` say the gate did not get to judge, and carry no
verdict at all; only ``EVALUATED`` has one. A single enum with a ``FAIL`` that
doubles as "we could not check" is the flattening this split exists to prevent
-- a run that never happened would be indistinguishable from one that failed,
and the remedy for those two is not the same.

**Aggregation must not read the order it was handed.** The tests below permute
every multi-slot case and assert the WHOLE ``GateResult`` is identical, because
the bug this replaces is the classic one: a loop that keeps the last verdict it
saw, which is right whenever the inputs happen to arrive in a helpful order.
FAIL beating ABSTAIN is asserted in both orders for the same reason.

**Every applicability/conclusion pair is tried.** Nine combinations, five legal,
and the four that are not are refused where the report is built rather than
where it is read -- an INAPPLICABLE check that reports PASS has passed nothing,
and a gate that received one would aggregate it faithfully.
"""

from __future__ import annotations

import itertools
import uuid

import pytest

from bayesmith.artifacts._codec import canonical_dumps, canonical_loads
from bayesmith.artifacts.base import (
    ArtifactRef,
    ErrorRecord,
    ProducerRef,
    new_artifact_meta,
)
from bayesmith.artifacts.gates import (
    AttemptStatus,
    GateDefinition,
    GateResult,
    OperationalStatus,
    ReportRequirement,
    ReportSlot,
    aggregate_gate,
)
from bayesmith.artifacts.identity import (
    ArtifactKind,
    FingerprintBundle,
    FingerprintKind,
    fingerprint,
)
from bayesmith.artifacts.refusal import Remedy
from bayesmith.artifacts.reports import Applicability, Conclusion, EvaluationReport

K = FingerprintKind
A = Applicability
C = Conclusion
S = OperationalStatus

REQUIRED_NAME = "rank_uniformity"
SECOND_NAME = "posterior_contraction"
OPTIONAL_NAME = "energy_diagnostic"

#: §0.6's legal pairs, spelled here rather than imported: a test that reads the
#: implementation's own table agrees with whatever that table says, including
#: when it is wrong.
LEGAL_PAIRS = {
    (A.APPLICABLE, C.PASS),
    (A.APPLICABLE, C.FAIL),
    (A.APPLICABLE, C.ABSTAIN),
    (A.INAPPLICABLE, C.ABSTAIN),
    (A.UNVERIFIABLE, C.ABSTAIN),
}


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


def meta(kind: ArtifactKind = ArtifactKind.EVALUATION_REPORT):
    return new_artifact_meta(
        artifact_type=kind,
        fingerprints=bundle(),
        producer=ProducerRef(package="bayesmith", version="0.6.2"),
        summary="the posterior gate",
    )


def remedy() -> Remedy:
    return Remedy(
        action="raise_draws",
        message="run again with four times the draws",
        parameters=(("draws", 4000),),
    )


def report(
    applicability: Applicability = A.APPLICABLE,
    conclusion: Conclusion = C.PASS,
    kind: str = "rank_uniformity@1",
) -> EvaluationReport:
    return EvaluationReport(
        meta=meta(),
        subject_ref=ref(),
        report_kind=kind,
        applicability=applicability,
        conclusion=conclusion,
        findings=(),
    )


def requirement(
    name: str = REQUIRED_NAME,
    *,
    required: bool = True,
    optional_error_blocks: bool = False,
) -> ReportRequirement:
    return ReportRequirement(
        name=name, required=required, optional_error_blocks=optional_error_blocks
    )


def gate(*requirements: ReportRequirement) -> GateDefinition:
    return GateDefinition(
        name="posterior-gate",
        version=1,
        requirements=requirements or (requirement(),),
        blocked_actions=("publish", "compare_models"),
        remedies=(remedy(),),
    )


def produced(
    req: ReportRequirement,
    applicability: Applicability = A.APPLICABLE,
    conclusion: Conclusion = C.PASS,
    *,
    invalidated: bool = False,
) -> ReportSlot:
    return ReportSlot(
        requirement=req,
        report=report(applicability, conclusion),
        attempt_status=AttemptStatus.ATTEMPTED,
        invalidated=invalidated,
        error=None,
    )


def errored(req: ReportRequirement) -> ReportSlot:
    return ReportSlot(
        requirement=req,
        report=None,
        attempt_status=AttemptStatus.ATTEMPTED,
        invalidated=False,
        error=ErrorRecord(
            code="backend_unavailable",
            message="the diagnostic backend was not reachable",
            exception_type="RuntimeError",
        ),
    )


def unattempted(req: ReportRequirement) -> ReportSlot:
    return ReportSlot(
        requirement=req,
        report=None,
        attempt_status=AttemptStatus.NOT_ATTEMPTED,
        invalidated=False,
        error=None,
    )


def gate_result(**overrides: object) -> GateResult:
    fields: dict[str, object] = {
        "meta": meta(),
        "definition": gate(),
        "status": S.EVALUATED,
        "verdict": C.PASS,
        "report_refs": (),
        "findings": (),
        "blocked_actions": ("publish",),
        "remedies": (remedy(),),
    }
    fields.update(overrides)
    return GateResult(**fields)


# ----------------------------------------------- 5.1 the two axes, exhaustively


@pytest.mark.parametrize("conclusion", sorted(Conclusion, key=lambda item: item.value))
@pytest.mark.parametrize(
    "applicability", sorted(Applicability, key=lambda item: item.value)
)
def test_every_applicability_and_conclusion_pair_is_legal_or_refused(
    applicability, conclusion
):
    if (applicability, conclusion) in LEGAL_PAIRS:
        built = report(applicability, conclusion)
        assert built.applicability is applicability
        assert built.conclusion is conclusion
    else:
        with pytest.raises(ValueError, match="applicab"):
            report(applicability, conclusion)


def test_exactly_five_of_the_nine_pairs_are_legal():
    """The count, so that widening the table is a decision rather than a
    side effect of loosening one branch."""
    assert len(list(itertools.product(Applicability, Conclusion))) == 9
    assert len(LEGAL_PAIRS) == 5


@pytest.mark.parametrize("status", [S.BLOCKED, S.INVALIDATED, S.ERROR])
def test_a_gate_that_did_not_evaluate_carries_no_verdict(status):
    """§0 ruling 7. A verdict beside BLOCKED would be a judgement of a run that
    never happened."""
    assert gate_result(status=status, verdict=None).verdict is None
    with pytest.raises(ValueError, match="verdict"):
        gate_result(status=status, verdict=C.PASS)


def test_an_evaluated_gate_carries_a_verdict():
    with pytest.raises(ValueError, match="verdict"):
        gate_result(status=S.EVALUATED, verdict=None)


def test_a_gate_result_round_trips():
    original = gate_result()
    assert canonical_loads(canonical_dumps(original), expected=GateResult) == original


# ------------------------------------------------------ 5.2 the truth table


def row_prerequisite_missing():
    definition = gate()
    return definition, False, True, (produced(definition.requirements[0]),)


def row_inputs_stale():
    definition = gate()
    return definition, True, False, (produced(definition.requirements[0]),)


def row_report_stale():
    definition = gate()
    return (
        definition,
        True,
        True,
        (produced(definition.requirements[0], invalidated=True),),
    )


def row_required_attempted_error():
    definition = gate()
    return definition, True, True, (errored(definition.requirements[0]),)


def row_blocking_optional_error():
    definition = gate(
        requirement(),
        requirement(OPTIONAL_NAME, required=False, optional_error_blocks=True),
    )
    required, optional = definition.requirements
    return definition, True, True, (produced(required), errored(optional))


def row_non_blocking_optional_error():
    definition = gate(
        requirement(),
        requirement(OPTIONAL_NAME, required=False, optional_error_blocks=False),
    )
    required, optional = definition.requirements
    return definition, True, True, (produced(required), errored(optional))


def row_required_applicable_fail():
    definition = gate()
    return (
        definition,
        True,
        True,
        (produced(definition.requirements[0], A.APPLICABLE, C.FAIL),),
    )


def row_required_never_produced():
    definition = gate()
    return definition, True, True, (unattempted(definition.requirements[0]),)


def row_required_unverifiable():
    definition = gate()
    return (
        definition,
        True,
        True,
        (produced(definition.requirements[0], A.UNVERIFIABLE, C.ABSTAIN),),
    )


def row_required_abstains():
    definition = gate()
    return (
        definition,
        True,
        True,
        (produced(definition.requirements[0], A.APPLICABLE, C.ABSTAIN),),
    )


def row_required_inapplicable():
    definition = gate()
    return (
        definition,
        True,
        True,
        (produced(definition.requirements[0], A.INAPPLICABLE, C.ABSTAIN),),
    )


def row_optional_inapplicable_and_required_pass():
    definition = gate(requirement(), requirement(OPTIONAL_NAME, required=False))
    required, optional = definition.requirements
    return (
        definition,
        True,
        True,
        (produced(required), produced(optional, A.INAPPLICABLE, C.ABSTAIN)),
    )


def row_all_required_pass():
    definition = gate(requirement(), requirement(SECOND_NAME))
    first, second = definition.requirements
    return definition, True, True, (produced(first), produced(second))


TRUTH_TABLE = [
    ("prerequisite missing", row_prerequisite_missing, S.BLOCKED, None),
    ("input stale", row_inputs_stale, S.INVALIDATED, None),
    ("report stale", row_report_stale, S.INVALIDATED, None),
    ("required attempted error", row_required_attempted_error, S.ERROR, None),
    ("blocking optional error", row_blocking_optional_error, S.ERROR, None),
    (
        "non-blocking optional error and required pass",
        row_non_blocking_optional_error,
        S.EVALUATED,
        C.PASS,
    ),
    ("required applicable fail", row_required_applicable_fail, S.EVALUATED, C.FAIL),
    ("required never produced", row_required_never_produced, S.EVALUATED, C.ABSTAIN),
    ("required unverifiable", row_required_unverifiable, S.EVALUATED, C.ABSTAIN),
    ("required abstains", row_required_abstains, S.EVALUATED, C.ABSTAIN),
    ("required inapplicable", row_required_inapplicable, S.EVALUATED, C.ABSTAIN),
    (
        "optional inapplicable and all required pass",
        row_optional_inapplicable_and_required_pass,
        S.EVALUATED,
        C.PASS,
    ),
    ("all required applicable pass", row_all_required_pass, S.EVALUATED, C.PASS),
]


@pytest.mark.parametrize(
    ("builder", "status", "verdict"),
    [row[1:] for row in TRUTH_TABLE],
    ids=[row[0] for row in TRUTH_TABLE],
)
def test_the_gate_truth_table(builder, status, verdict):
    definition, prerequisites_ready, inputs_current, slots = builder()
    result = aggregate_gate(
        definition,
        meta=meta(),
        prerequisites_ready=prerequisites_ready,
        inputs_current=inputs_current,
        slots=slots,
    )
    assert result.status is status
    assert result.verdict is verdict


@pytest.mark.parametrize(
    "builder",
    [row[1] for row in TRUTH_TABLE],
    ids=[row[0] for row in TRUTH_TABLE],
)
def test_the_answer_does_not_depend_on_the_order_the_slots_arrived_in(builder):
    """Every permutation of the same slots, against the same meta, must give
    the same whole result -- findings and report refs included, not just the
    verdict."""
    definition, prerequisites_ready, inputs_current, slots = builder()
    envelope = meta()
    results = [
        aggregate_gate(
            definition,
            meta=envelope,
            prerequisites_ready=prerequisites_ready,
            inputs_current=inputs_current,
            slots=permutation,
        )
        for permutation in itertools.permutations(slots)
    ]
    assert results
    for result in results[1:]:
        assert result == results[0]


def test_the_permutation_sweep_has_something_to_permute():
    """The sibling of the sweep above, and the reason it is worth its run.

    With one slot in every row, permuting is the identity and the guard would
    go green against an aggregator that read the order it was handed. This
    fails if the table ever loses its multi-slot rows.
    """
    widths = [len(builder()[3]) for _, builder, _, _ in TRUTH_TABLE]
    assert max(widths) >= 2, widths
    assert sum(1 for width in widths if width >= 2) >= 3, widths


def test_fail_beats_abstain_whichever_order_they_arrive_in():
    """A frozen precedence, and the one a first/last-wins loop gets right half
    the time."""
    definition = gate(requirement(), requirement(SECOND_NAME))
    first, second = definition.requirements
    abstaining = produced(first, A.APPLICABLE, C.ABSTAIN)
    failing = produced(second, A.APPLICABLE, C.FAIL)
    for slots in ((abstaining, failing), (failing, abstaining)):
        result = aggregate_gate(
            definition,
            meta=meta(),
            prerequisites_ready=True,
            inputs_current=True,
            slots=slots,
        )
        assert result.status is S.EVALUATED
        assert result.verdict is C.FAIL


def test_a_blocked_gate_says_why_in_a_finding():
    definition, ready, current, slots = row_prerequisite_missing()
    result = aggregate_gate(
        definition,
        meta=meta(),
        prerequisites_ready=ready,
        inputs_current=current,
        slots=slots,
    )
    assert [finding.code for finding in result.findings] == ["prerequisite_missing"]


def test_an_errored_required_report_names_the_error_code_it_saw():
    definition, ready, current, slots = row_required_attempted_error()
    result = aggregate_gate(
        definition,
        meta=meta(),
        prerequisites_ready=ready,
        inputs_current=current,
        slots=slots,
    )
    assert [finding.code for finding in result.findings] == ["required_report_errored"]
    assert result.findings[0].observed == "backend_unavailable"


def test_a_passing_gate_states_no_findings():
    definition, ready, current, slots = row_all_required_pass()
    result = aggregate_gate(
        definition,
        meta=meta(),
        prerequisites_ready=ready,
        inputs_current=current,
        slots=slots,
    )
    assert result.findings == ()
    assert len(result.report_refs) == 2


def test_the_blocked_actions_and_remedies_come_from_the_definition_unedited():
    """§0.6: the aggregator does not invent what a gate governs. What a
    consumer must read to know whether they are blocked is the status."""
    definition, ready, current, slots = row_all_required_pass()
    result = aggregate_gate(
        definition,
        meta=meta(),
        prerequisites_ready=ready,
        inputs_current=current,
        slots=slots,
    )
    assert result.blocked_actions == definition.blocked_actions
    assert result.remedies == definition.remedies


def test_an_optional_failure_is_recorded_and_changes_no_verdict():
    """The frozen rule reads 'any REQUIRED applicable FAIL'. An optional check
    is advisory by construction -- a caller who wants it to bite marks it
    required -- so it is reported and does not decide."""
    definition = gate(requirement(), requirement(OPTIONAL_NAME, required=False))
    required, optional = definition.requirements
    result = aggregate_gate(
        definition,
        meta=meta(),
        prerequisites_ready=True,
        inputs_current=True,
        slots=(produced(required), produced(optional, A.APPLICABLE, C.FAIL)),
    )
    assert result.verdict is C.PASS
    assert [finding.code for finding in result.findings] == ["optional_report_failed"]


# ------------------------------------------------- the boundary refuses ambiguity


def test_a_slot_cannot_hold_both_a_report_and_an_error():
    with pytest.raises(ValueError, match="error"):
        ReportSlot(
            requirement=requirement(),
            report=report(),
            attempt_status=AttemptStatus.ATTEMPTED,
            error=ErrorRecord(
                code="crashed", message="it crashed", exception_type="RuntimeError"
            ),
        )


def test_a_slot_that_was_never_attempted_holds_neither():
    with pytest.raises(ValueError, match="attempt"):
        ReportSlot(
            requirement=requirement(),
            report=report(),
            attempt_status=AttemptStatus.NOT_ATTEMPTED,
        )


def test_a_slot_with_no_report_cannot_be_stale():
    with pytest.raises(ValueError, match="invalidated"):
        ReportSlot(
            requirement=requirement(),
            report=None,
            attempt_status=AttemptStatus.NOT_ATTEMPTED,
            invalidated=True,
        )


def test_two_slots_for_one_requirement_are_refused():
    definition = gate()
    required = definition.requirements[0]
    with pytest.raises(ValueError, match=REQUIRED_NAME):
        aggregate_gate(
            definition,
            meta=meta(),
            prerequisites_ready=True,
            inputs_current=True,
            slots=(produced(required), produced(required, A.APPLICABLE, C.FAIL)),
        )


def test_a_slot_the_schema_never_declared_is_refused():
    with pytest.raises(ValueError, match="undeclared_check"):
        aggregate_gate(
            gate(),
            meta=meta(),
            prerequisites_ready=True,
            inputs_current=True,
            slots=(produced(requirement("undeclared_check")),),
        )


def test_a_slot_that_disagrees_with_the_schema_is_refused():
    """Same name, different requirement: a slot claiming a required check is
    optional would downgrade it, and the gate would pass on a report nobody
    was allowed to skip."""
    definition = gate()
    with pytest.raises(ValueError, match=REQUIRED_NAME):
        aggregate_gate(
            definition,
            meta=meta(),
            prerequisites_ready=True,
            inputs_current=True,
            slots=(produced(requirement(REQUIRED_NAME, required=False)),),
        )


def test_a_gate_definition_names_each_requirement_once():
    with pytest.raises(ValueError, match=REQUIRED_NAME):
        gate(requirement(), requirement())


def test_a_gate_definition_names_at_least_one_requirement():
    with pytest.raises(ValueError):
        GateDefinition(
            name="empty-gate", version=1, requirements=(), blocked_actions=(), remedies=()
        )


def test_a_required_check_does_not_also_declare_an_optional_error_rule():
    """``optional_error_blocks`` answers a question only an optional
    requirement has; a required one's error always blocks. Two answers to one
    question is the shape this package refuses everywhere else."""
    with pytest.raises(ValueError, match="optional_error_blocks"):
        requirement(required=True, optional_error_blocks=True)


def test_a_gate_identity_is_spelled_the_way_a_task_asks_for_it():
    """``PosteriorTask.quality_gate`` is a versioned gate identity string; this
    is the definition it names, so the two spellings have to be one."""
    assert gate().identity == "posterior-gate@1"
