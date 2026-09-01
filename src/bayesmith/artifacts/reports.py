"""What compilation decided, and what a judgement about a result says.

Three records with one purpose: a plan that cannot be read back is a black box
that happened to run, and a verdict that cannot be read back is an opinion.

**An analysis report is an interpretation, never a copy of the graph.** It
holds a :class:``bayesmith.artifacts.identity.ModelRef``, the graph's structure
fingerprint and structured findings. §0 ruling 4 is nearest to being broken
here, because this record is MADE by reading a Graph -- so it is the one that
must hold none of it.

**A plan record is a schedule.** Blocks keep their declared order, because that
order is what will run; a parameter belongs to exactly one block, because two
blocks claiming it is a schedule with no answer to which draws it. The
``premises`` it lists are the codes a :class:``bayesmith.artifacts.refusal.Refusal``
names in ``failed_premise`` when one of them turns out to be false -- one
vocabulary, both directions.

**An evaluation report has two axes.** §0 ruling 7: whether the check applied
and what it concluded are different questions, so they are different enums.
Which combinations of the two are legal, and how several reports aggregate
into one gate verdict, are pinned in :mod:``bayesmith.artifacts.gates`` -- one
home for one decision.

Layering: ``base``, ``identity`` and ``refusal`` (§0.1). The code validators are
refusal's, since that module is where "a code, not prose" is decided.
"""

from __future__ import annotations

import dataclasses
import math
from enum import StrEnum

from ._codec import register_artifact_type
from .base import (
    ApproximationRecord,
    ArtifactMeta,
    ArtifactRef,
    ComputeBudget,
    _instance,
    _member,
    _optional_text,
    _tuple_of,
    _uuid4,
    canonical_value_options,
)
from .identity import ArtifactKind, Fingerprint, FingerprintKind, ModelRef
from .refusal import Finding, ScopeRef, _code, _code_tuple

__all__ = [
    "AnalysisFinding",
    "AnalysisReport",
    "PlanBlockRecord",
    "InferencePlanRecord",
    "Applicability",
    "Conclusion",
    "EvaluationReport",
]


#: "No budget was stated". One shared object rather than a default_factory
#: because ComputeBudget is frozen: there is nothing to mutate, so two plans
#: built a second apart hold one value and compare equal.
_NO_BUDGET = ComputeBudget()


def _positive_or_none(label: str, value: object) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float):
        raise TypeError(f"{label} is a number; got {value!r}")
    if not math.isfinite(value):
        raise ValueError(
            f"{label} must be finite; got {value!r}. An unavailable number is "
            "None -- NaN compares unequal to itself and reads as a measurement"
        )
    if value <= 0.0:
        raise ValueError(f"{label} is a positive number; got {value!r}")
    return float(value)


def _envelope(label: str, meta: object, kind: ArtifactKind) -> None:
    _instance(label, meta, ArtifactMeta)
    if meta.artifact_type is not kind:
        raise ValueError(
            f"{label} is of artifact kind {kind.value}; got "
            f"{meta.artifact_type.value!r}. The kind is the row the §0.3 "
            "invalidation matrix reads, so an artifact filed under another "
            "would be kept when it should have been retired"
        )


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class AnalysisFinding:
    """One thing compilation concluded about the graph, and what it measured.

    ``grounds`` is a tuple of reason CODES rather than of nested findings: at
    compile time the support for a conclusion is a set of checks that passed,
    and naming them keeps the record flat enough to read. It is the same idea
    as a refusal's ``grounds`` -- what stands behind the claim, structured --
    which is why it carries the same name.
    """

    code: str
    conclusion: str
    scope: ScopeRef
    measurements: tuple[tuple[str, object], ...] = ()
    grounds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _code("an analysis finding's code", self.code)
        _code("an analysis finding's conclusion", self.conclusion)
        _instance("an analysis finding's scope", self.scope, ScopeRef)
        object.__setattr__(
            self,
            "measurements",
            canonical_value_options(
                "an analysis finding's measurements", self.measurements
            ),
        )
        _code_tuple("an analysis finding's grounds", self.grounds)
        if not self.grounds:
            raise ValueError(
                "an analysis finding names the checks it rests on; a conclusion "
                "with no grounds is the compiler's opinion"
            )


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class AnalysisReport:
    """Compile-time reading of a graph: what it is, and what could run on it.

    ``candidate_routes`` is a PREFERENCE order, so it is neither sorted nor
    deduplicated into a set -- reordering it would rerank the routes, and a
    repeated route would make one option look like two.
    """

    meta: ArtifactMeta
    model_ref: ModelRef
    graph_fingerprint: Fingerprint
    findings: tuple[AnalysisFinding, ...] = ()
    candidate_routes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _envelope("an analysis report's meta", self.meta, ArtifactKind.PLAN)
        _instance("an analysis report's model_ref", self.model_ref, ModelRef)
        _instance(
            "an analysis report's graph_fingerprint", self.graph_fingerprint, Fingerprint
        )
        if self.graph_fingerprint.kind is not FingerprintKind.GRAPH_STRUCTURE:
            raise ValueError(
                "an analysis report's graph_fingerprint is of kind "
                f"graph_structure; got {self.graph_fingerprint.kind.value!r}"
            )
        _tuple_of("an analysis report's findings", self.findings, AnalysisFinding)
        _code_tuple("candidate_routes", self.candidate_routes)


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class PlanBlockRecord:
    """One block of the schedule: which parameters, by which method, and why.

    ``kappa`` and ``tolerance`` are the numbers that justify the method where it
    has them, and ``None`` where the method has no such knob -- an exact
    elimination has no tolerance to report, and reporting 0 would say it had
    one and met it perfectly.
    """

    names: tuple[str, ...]
    method: str
    reason_codes: tuple[str, ...] = ()
    kappa: float | None = None
    tolerance: float | None = None
    approximation: ApproximationRecord | None = None

    def __post_init__(self) -> None:
        _code_tuple("a block's names", self.names)
        if not self.names:
            raise ValueError("a block names at least one parameter")
        _code("a block's method", self.method)
        _code_tuple("a block's reason_codes", self.reason_codes)
        if not self.reason_codes:
            raise ValueError(
                "a block's reason_codes say why this method was chosen for "
                "these parameters; a method chosen for no recorded reason is "
                "exactly the plan this record exists to make auditable"
            )
        object.__setattr__(self, "kappa", _positive_or_none("a block's kappa", self.kappa))
        object.__setattr__(
            self, "tolerance", _positive_or_none("a block's tolerance", self.tolerance)
        )
        if self.approximation is not None:
            _instance("a block's approximation", self.approximation, ApproximationRecord)


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class InferencePlanRecord:
    """The compiled plan, as a record: what will run, on what, under what.

    The analysis report it was built from is a PARENT, not merely a field.
    §0.3 retires a plan through the inputs it was made from, so a plan whose
    reading of the graph is not in its lineage would outlive a change to that
    reading.
    """

    meta: ArtifactMeta
    task_id: str
    model_ref: ModelRef
    analysis_report_ref: ArtifactRef | None = None
    blocks: tuple[PlanBlockRecord, ...] = ()
    exact_elimination: tuple[str, ...] = ()
    residual_parameters: tuple[str, ...] = ()
    backend: str = "auto"
    premises: tuple[str, ...] = ()
    budget: ComputeBudget = _NO_BUDGET
    quality_gate: str | None = None
    fallback_policy: str | None = None

    def __post_init__(self) -> None:
        _envelope("a plan record's meta", self.meta, ArtifactKind.PLAN)
        _uuid4("a plan record's task_id", self.task_id)
        _instance("a plan record's model_ref", self.model_ref, ModelRef)
        if self.analysis_report_ref is not None:
            _instance(
                "analysis_report_ref", self.analysis_report_ref, ArtifactRef
            )
            if self.analysis_report_ref.artifact_type is not ArtifactKind.PLAN:
                raise ValueError(
                    "analysis_report_ref points at a "
                    f"{self.analysis_report_ref.artifact_type.value}; an "
                    "analysis report is a planning artifact"
                )
            if self.analysis_report_ref not in self.meta.parent_refs:
                raise ValueError(
                    "the analysis report a plan was made from is one of its "
                    "parent_refs; lineage that omits an input cannot retire the "
                    "plan when that input changes"
                )
        _tuple_of("a plan record's blocks", self.blocks, PlanBlockRecord)
        claimed: dict[str, int] = {}
        for position, item in enumerate(self.blocks):
            for name in item.names:
                if name in claimed:
                    raise ValueError(
                        f"blocks {claimed[name]} and {position} both claim "
                        f"{name!r}; a schedule with one parameter in two blocks "
                        "has no answer to which of them draws it"
                    )
                claimed[name] = position
        _code_tuple("exact_elimination", self.exact_elimination)
        _code_tuple("residual_parameters", self.residual_parameters)
        both = sorted(set(self.exact_elimination) & set(self.residual_parameters))
        if both:
            raise ValueError(
                f"{both} are named as both eliminated and residual; an "
                "eliminated parameter is precisely one the plan does not carry"
            )
        _code("a plan record's backend", self.backend)
        _code_tuple("premises", self.premises)
        _instance("a plan record's budget", self.budget, ComputeBudget)
        _optional_text("quality_gate", self.quality_gate)
        _optional_text("fallback_policy", self.fallback_policy)


@register_artifact_type
class Applicability(StrEnum):
    """Whether the check could say anything about this subject (§0 ruling 7).

    ``INAPPLICABLE`` and ``UNVERIFIABLE`` are different answers: the first says
    the check does not apply to this kind of result, the second says it applies
    but the inputs it needs are missing. Collapsing them would make "we cannot
    check this" and "there is nothing to check" one status, and only one of
    those is worth chasing.
    """

    APPLICABLE = "applicable"
    INAPPLICABLE = "inapplicable"
    UNVERIFIABLE = "unverifiable"


@register_artifact_type
class Conclusion(StrEnum):
    """What an applicable check concluded, and what a gate aggregates to.

    One enum for both levels on purpose. A report's conclusion and a gate's
    verdict are the same three answers about different scopes, and two enums
    with identical members would be two types nothing stops a caller swapping
    -- the mistake ``ApproximationClass`` and ``TargetFidelity`` are kept apart
    to avoid is the OPPOSITE case: there, one word means two things.
    """

    PASS = "pass"
    FAIL = "fail"
    ABSTAIN = "abstain"


#: §0.6's legal pairs, and the whole of the two-axis ruling in one table.
#: Only an APPLICABLE check may PASS or FAIL: a check that did not apply, or
#: could not be run, has concluded nothing about the subject, and a PASS from
#: one would be counted by :func:``bayesmith.artifacts.gates.aggregate_gate`` as
#: a check that was made. The other direction is the same statement: ABSTAIN is
#: available to all three, because "applicable but undecided" is a real
#: outcome.
_LEGAL_PAIRS = frozenset(
    {
        (Applicability.APPLICABLE, Conclusion.PASS),
        (Applicability.APPLICABLE, Conclusion.FAIL),
        (Applicability.APPLICABLE, Conclusion.ABSTAIN),
        (Applicability.INAPPLICABLE, Conclusion.ABSTAIN),
        (Applicability.UNVERIFIABLE, Conclusion.ABSTAIN),
    }
)


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class EvaluationReport:
    """One check's judgement of one result, on the two axes of §0 ruling 7.

    The legal combinations are refused HERE, where the report is built, rather
    than where a gate reads it: an inapplicable check reporting PASS has passed
    nothing, and an aggregator handed one would count it faithfully.
    """

    meta: ArtifactMeta
    subject_ref: ArtifactRef
    report_kind: str
    applicability: Applicability
    conclusion: Conclusion
    findings: tuple[Finding, ...] = ()

    def __post_init__(self) -> None:
        _envelope(
            "an evaluation report's meta", self.meta, ArtifactKind.EVALUATION_REPORT
        )
        _instance("an evaluation report's subject_ref", self.subject_ref, ArtifactRef)
        if self.subject_ref.artifact_type is not ArtifactKind.RESULT:
            raise ValueError(
                "an evaluation report judges a result; its subject_ref points "
                f"at a {self.subject_ref.artifact_type.value}"
            )
        _code("an evaluation report's report_kind", self.report_kind)
        _member("applicability", self.applicability, Applicability)
        _member("conclusion", self.conclusion, Conclusion)
        if (self.applicability, self.conclusion) not in _LEGAL_PAIRS:
            raise ValueError(
                f"an {self.applicability.value} check cannot conclude "
                f"{self.conclusion.value}; only an applicable check reaches a "
                "pass or a fail, and the other two abstain"
            )
        _tuple_of("an evaluation report's findings", self.findings, Finding)
