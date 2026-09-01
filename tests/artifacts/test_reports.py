"""What compilation decided, and what a judgement about a result looks like.

Three records, one theme: a plan is auditable or it is a black box that
happened to run.

**An analysis report is an interpretation of a graph, not a copy of one.** It
holds the model reference, the graph's structure fingerprint and structured
findings -- never the Graph, never a node object, never a callable (§0 ruling
4). Two runs of the same model on the same structure produce the same
fingerprint here, on any machine.

**A plan record says what it will do, in the order it will do it.** Blocks keep
their declared order because that order IS the schedule, and a name may appear
in only one block: two blocks claiming ``theta`` is a schedule with no answer
to which one draws it, and whichever ran second would silently win.

**An evaluation report is a judgement with two axes.** Task 4 freezes its
shape; §0 ruling 7 splits applicability from conclusion, and the legal
combinations of the two are pinned in ``tests/artifacts/test_gates.py`` together
with the aggregation that reads them -- one home for one decision.
"""

from __future__ import annotations

import dataclasses
import hashlib
import uuid

import pytest

from bayesmith.artifacts._codec import canonical_dumps, canonical_loads
from bayesmith.artifacts.base import (
    ApproximationClass,
    ApproximationRecord,
    ArtifactRef,
    ComputeBudget,
    ProducerRef,
    TargetFidelity,
    new_artifact_meta,
)
from bayesmith.artifacts.identity import (
    ArtifactKind,
    FingerprintBundle,
    FingerprintKind,
    ModelRef,
    fingerprint,
)
from bayesmith.artifacts.refusal import Finding, ScopeKind, ScopeRef
from bayesmith.artifacts.reports import (
    AnalysisFinding,
    AnalysisReport,
    Applicability,
    Conclusion,
    EvaluationReport,
    InferencePlanRecord,
    PlanBlockRecord,
)

K = FingerprintKind

DIGEST = hashlib.sha256(b"def pilot(): ...").hexdigest()


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


def ref(kind: ArtifactKind = ArtifactKind.PLAN) -> ArtifactRef:
    return ArtifactRef(artifact_id=str(uuid.uuid4()), revision=0, artifact_type=kind)


def meta(**overrides: object):
    fields: dict[str, object] = {
        "artifact_type": ArtifactKind.PLAN,
        "fingerprints": bundle(),
        "producer": ProducerRef(package="bayesmith", version="0.6.2"),
        "summary": "one Gaussian block",
    }
    fields.update(overrides)
    return new_artifact_meta(**fields)


def model_ref() -> ModelRef:
    return ModelRef(identifier="pilot", source_digest=DIGEST)


def analysis_finding(**overrides: object) -> AnalysisFinding:
    fields: dict[str, object] = {
        "code": "linearity_probe",
        "conclusion": "affine_in_theta",
        "scope": ScopeRef(kind=ScopeKind.NODE, name="y"),
        "measurements": (("max_absolute_error", 4.4e-16), ("probes", 3)),
        "grounds": ("three_point_probe", "exact_arithmetic"),
    }
    fields.update(overrides)
    return AnalysisFinding(**fields)


def analysis_report(**overrides: object) -> AnalysisReport:
    fields: dict[str, object] = {
        "meta": meta(),
        "model_ref": model_ref(),
        "graph_fingerprint": fingerprint(K.GRAPH_STRUCTURE, "theta -> y"),
        "findings": (analysis_finding(),),
        "candidate_routes": ("exact_gls", "gibbs", "nuts"),
    }
    fields.update(overrides)
    return AnalysisReport(**fields)


def block(**overrides: object) -> PlanBlockRecord:
    fields: dict[str, object] = {
        "names": ("theta",),
        "method": "exact_gls",
        "reason_codes": ("affine_in_theta", "gaussian_likelihood"),
        "kappa": 12.5,
        "tolerance": 1e-10,
        "approximation": ApproximationRecord(
            representation_class=ApproximationClass.EXACT,
            target_fidelity=TargetFidelity.EXACT,
        ),
    }
    fields.update(overrides)
    return PlanBlockRecord(**fields)


def plan(**overrides: object) -> InferencePlanRecord:
    analysis = ref(ArtifactKind.PLAN)
    fields: dict[str, object] = {
        "meta": meta(parent_refs=(analysis,)),
        "task_id": str(uuid.uuid4()),
        "model_ref": model_ref(),
        "analysis_report_ref": analysis,
        "blocks": (block(), block(names=("sigma",), method="nuts")),
        "exact_elimination": ("nuisance",),
        "residual_parameters": ("theta", "sigma"),
        "backend": "auto",
        "premises": ("gaussian_likelihood", "affine_in_theta"),
        "budget": ComputeBudget(draws=1000, warmup=500, chains=4),
        "quality_gate": "posterior-gate@1",
        "fallback_policy": "nuts_on_collapse",
    }
    fields.update(overrides)
    return InferencePlanRecord(**fields)


def evaluation_report(**overrides: object) -> EvaluationReport:
    fields: dict[str, object] = {
        "meta": meta(artifact_type=ArtifactKind.EVALUATION_REPORT),
        "subject_ref": ref(ArtifactKind.RESULT),
        "report_kind": "rank_uniformity@1",
        "applicability": Applicability.APPLICABLE,
        "conclusion": Conclusion.PASS,
        "findings": (),
    }
    fields.update(overrides)
    return EvaluationReport(**fields)


# ------------------------------------------------------------- analysis report


def test_an_analysis_report_round_trips():
    original = analysis_report()
    restored = canonical_loads(canonical_dumps(original), expected=AnalysisReport)
    assert restored == original
    assert restored.findings[0].measurements == (
        ("max_absolute_error", 4.4e-16),
        ("probes", 3),
    )


def test_an_analysis_report_pins_the_structure_it_read():
    """The graph fingerprint is in the graph_structure slot, not any other: a
    digest filed under the wrong slot answers for an input it never saw."""
    assert analysis_report().graph_fingerprint.kind is K.GRAPH_STRUCTURE
    with pytest.raises(ValueError, match="graph_structure"):
        analysis_report(graph_fingerprint=fingerprint(K.DATA, "theta -> y"))


def test_candidate_routes_keep_their_order_and_admit_no_duplicate():
    """The order is the preference order, so sorting it would silently rerank
    the routes; a repeat would make a route look twice as available."""
    assert analysis_report().candidate_routes == ("exact_gls", "gibbs", "nuts")
    with pytest.raises(ValueError, match="nuts"):
        analysis_report(candidate_routes=("nuts", "nuts"))


def test_an_analysis_finding_carries_measurements_and_reason_codes():
    assert analysis_finding().grounds == ("three_point_probe", "exact_arithmetic")
    with pytest.raises(ValueError):
        analysis_finding(grounds=())
    with pytest.raises(ValueError, match="conclusion"):
        analysis_finding(conclusion="affine in theta")


def test_a_runtime_object_cannot_enter_an_analysis_report():
    """§0 ruling 4, at the boundary where the Graph is nearest: this record is
    made BY reading a graph, so it is the one most tempting to let hold one."""
    with pytest.raises(TypeError):
        analysis_report(model_ref=lambda: None)
    with pytest.raises(TypeError):
        analysis_finding(measurements=(("probe", lambda: None),))


# ------------------------------------------------------------------ plan record


def test_a_plan_record_round_trips():
    original = plan()
    restored = canonical_loads(canonical_dumps(original), expected=InferencePlanRecord)
    assert restored == original
    for field in dataclasses.fields(original):
        assert getattr(restored, field.name) == getattr(original, field.name)


def test_the_plan_shape_is_the_one_the_plan_froze():
    assert [field.name for field in dataclasses.fields(InferencePlanRecord)] == [
        "meta",
        "task_id",
        "model_ref",
        "analysis_report_ref",
        "blocks",
        "exact_elimination",
        "residual_parameters",
        "backend",
        "premises",
        "budget",
        "quality_gate",
        "fallback_policy",
    ]


def test_block_order_is_the_schedule_and_survives_a_round_trip():
    original = plan()
    assert [item.names for item in original.blocks] == [("theta",), ("sigma",)]
    restored = canonical_loads(canonical_dumps(original), expected=InferencePlanRecord)
    assert [item.names for item in restored.blocks] == [("theta",), ("sigma",)]


def test_one_name_cannot_be_claimed_by_two_blocks():
    with pytest.raises(ValueError, match="theta"):
        plan(blocks=(block(), block(method="nuts")))


def test_a_block_names_at_least_one_parameter_and_says_why_its_method_was_chosen():
    with pytest.raises(ValueError):
        block(names=())
    with pytest.raises(ValueError, match="reason_codes"):
        block(reason_codes=())


def test_a_name_cannot_be_both_eliminated_and_left_in_the_residual():
    with pytest.raises(ValueError, match="theta"):
        plan(exact_elimination=("theta",))


def test_a_plan_records_the_analysis_it_was_made_from_as_a_parent():
    """Lineage is not decoration: the §0.3 matrix retires a plan through its
    parents, so a plan whose analysis report is not among them would survive a
    change to the reading it was built on."""
    original = plan()
    assert original.analysis_report_ref in original.meta.parent_refs
    with pytest.raises(ValueError, match="parent"):
        plan(meta=meta())


def test_a_plan_may_have_been_made_without_an_analysis_report():
    assert plan(analysis_report_ref=None, meta=meta()).analysis_report_ref is None


def test_a_plan_record_holds_no_callable():
    with pytest.raises(TypeError):
        plan(model_ref=lambda: None)
    with pytest.raises(TypeError):
        block(approximation=lambda: None)


def test_a_plan_is_an_artifact_of_the_plan_kind():
    with pytest.raises(ValueError, match="plan"):
        plan(meta=meta(artifact_type=ArtifactKind.RESULT, parent_refs=()))


def test_a_block_condition_number_and_tolerance_are_positive_or_absent():
    assert block(kappa=None, tolerance=None).kappa is None
    with pytest.raises(ValueError):
        block(kappa=0.0)
    with pytest.raises(ValueError, match="finite"):
        block(tolerance=float("nan"))


# ------------------------------------------------------------ evaluation report


def test_an_evaluation_report_round_trips():
    original = evaluation_report()
    restored = canonical_loads(canonical_dumps(original), expected=EvaluationReport)
    assert restored == original


def test_the_evaluation_report_shape_is_the_one_the_plan_froze():
    assert [field.name for field in dataclasses.fields(EvaluationReport)] == [
        "meta",
        "subject_ref",
        "report_kind",
        "applicability",
        "conclusion",
        "findings",
    ]


def test_the_two_axes_are_two_enums():
    """§0 ruling 7: whether the check applied and what it concluded are
    different questions, so they are different types and cannot be swapped."""
    assert set(Applicability) == {
        Applicability.APPLICABLE,
        Applicability.INAPPLICABLE,
        Applicability.UNVERIFIABLE,
    }
    assert set(Conclusion) == {Conclusion.PASS, Conclusion.FAIL, Conclusion.ABSTAIN}
    with pytest.raises(TypeError):
        evaluation_report(applicability="applicable")
    with pytest.raises(TypeError):
        evaluation_report(conclusion=Applicability.APPLICABLE)


def test_an_evaluation_report_judges_a_result_and_is_filed_as_a_report():
    with pytest.raises(ValueError, match="result"):
        evaluation_report(subject_ref=ref(ArtifactKind.PLAN))
    with pytest.raises(ValueError, match="evaluation_report"):
        evaluation_report(meta=meta())


def test_an_evaluation_report_carries_findings_of_the_refusal_family():
    """One Finding class, so a gate's reason and a refusal's reason are the
    same shape and a consumer needs one branch, not two."""
    reported = evaluation_report(
        applicability=Applicability.APPLICABLE,
        conclusion=Conclusion.FAIL,
        findings=(
            Finding(
                code="rank_statistics_not_uniform",
                message="the rank histogram is U-shaped",
                observed=0.004,
                expected=0.05,
                artifact_refs=(),
            ),
        ),
    )
    assert reported.findings[0].code == "rank_statistics_not_uniform"
    restored = canonical_loads(canonical_dumps(reported), expected=EvaluationReport)
    assert restored.findings == reported.findings
