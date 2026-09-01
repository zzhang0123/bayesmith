"""Quality gates: several reports in, one status and one verdict out (§0.6).

Two rulings do all the work here.

**§0 ruling 7: status and verdict are different axes.** ``BLOCKED``,
``INVALIDATED`` and ``ERROR`` mean the gate never got to judge and carry no
verdict at all; only ``EVALUATED`` carries ``PASS``, ``FAIL`` or ``ABSTAIN``.
Flattening the two would make "we could not check" indistinguishable from "we
checked and it failed", and those have different remedies.

**The answer may not depend on the order the slots arrived in.** So
:func:``aggregate_gate`` builds a mapping from requirement name to slot,
verifies it against the schema, and then walks the DEFINITION's requirements --
never the caller's list. The bug this shape exists to prevent is the loop that
keeps the last verdict it saw, which is correct exactly when the inputs happen
to arrive in a helpful order and silently wrong otherwise. Findings and report
references come out in definition order for the same reason: a result that
compares unequal to itself under a permutation is not a record.

The priority is §0.6's, in order: a missing prerequisite blocks; a stale input
or a stale report invalidates; an attempted required error, or an optional
error the schema says blocks, is an error; and only then is a verdict computed
-- FAIL if any required applicable report failed, ABSTAIN if any required
report is missing, inapplicable, unverifiable or undecided, and PASS only when
every required applicable report passed.

Layering: ``base``, ``refusal`` and ``reports`` (§0.1).
"""

from __future__ import annotations

import dataclasses
from enum import StrEnum

from ._codec import register_artifact_type
from .base import (
    ArtifactMeta,
    ArtifactRef,
    ErrorRecord,
    _count,
    _instance,
    _member,
    _tuple_of,
)
from .identity import ArtifactKind
from .refusal import Finding, Remedy, _code, _code_tuple
from .reports import Applicability, Conclusion, EvaluationReport, _envelope

__all__ = [
    "OperationalStatus",
    "AttemptStatus",
    "ReportRequirement",
    "ReportSlot",
    "GateDefinition",
    "GateResult",
    "aggregate_gate",
]


def _flag(label: str, value: object) -> bool:
    """A bool, and not a truthy int.

    The same rule as :mod:``bayesmith.artifacts.tasks``'s, restated rather than
    imported because §0.1 puts ``tasks`` outside this module's dependencies and
    a layering that is edged around for one validator is not a layering.
    """
    if type(value) is not bool:
        raise TypeError(f"{label} is a bool; got {value!r}")
    return value


@register_artifact_type
class OperationalStatus(StrEnum):
    """Whether the gate got to judge at all (§0 ruling 7).

    Three of the four say it did not, and each names a different thing to fix:
    a missing prerequisite is upstream work, an invalidated input is a re-run,
    an error is a failure to investigate. Only ``EVALUATED`` carries a verdict.
    """

    BLOCKED = "blocked"
    INVALIDATED = "invalidated"
    ERROR = "error"
    EVALUATED = "evaluated"


@register_artifact_type
class AttemptStatus(StrEnum):
    """Whether anybody tried to produce this report.

    The distinction the truth table turns on: a required report that was never
    attempted leaves the gate undecided (ABSTAIN), while one that was attempted
    and errored is an ERROR. Without this field the two look identical -- no
    report either way -- and the milder answer would win.
    """

    NOT_ATTEMPTED = "not_attempted"
    ATTEMPTED = "attempted"


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class ReportRequirement:
    """One check a gate asks for, and how much it matters.

    ``optional_error_blocks`` answers a question only an OPTIONAL requirement
    has: a required check that errored always blocks. Declaring it on a
    required requirement is refused rather than ignored -- a field that is
    silently ignored somewhere is a field somebody will set and rely on.
    """

    name: str
    required: bool = True
    optional_error_blocks: bool = False

    def __post_init__(self) -> None:
        _code("a requirement's name", self.name)
        _flag("a requirement's required", self.required)
        _flag("optional_error_blocks", self.optional_error_blocks)
        if self.required and self.optional_error_blocks:
            raise ValueError(
                f"{self.name!r} is required, so optional_error_blocks says "
                "nothing: an attempted required report that errored blocks the "
                "gate whatever this field holds"
            )


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class ReportSlot:
    """What became of one requirement on one subject.

    Exactly one outcome at a time: not attempted, a report, or an error. §0.6
    refuses a slot holding a report AND an error at the boundary, because an
    aggregator reading whichever it checks first would produce a defensible
    answer from an impossible input.
    """

    requirement: ReportRequirement
    report: EvaluationReport | None = None
    attempt_status: AttemptStatus = AttemptStatus.NOT_ATTEMPTED
    invalidated: bool = False
    error: ErrorRecord | None = None

    def __post_init__(self) -> None:
        _instance("a slot's requirement", self.requirement, ReportRequirement)
        if self.report is not None:
            _instance("a slot's report", self.report, EvaluationReport)
        _member("a slot's attempt_status", self.attempt_status, AttemptStatus)
        if self.error is not None:
            _instance("a slot's error", self.error, ErrorRecord)
        if self.report is not None and self.error is not None:
            raise ValueError(
                "a slot holds the report it produced or the error it hit, "
                "never both; which one an aggregator believed would then "
                "depend on the order it checked them in"
            )
        if self.attempt_status is AttemptStatus.NOT_ATTEMPTED:
            if self.report is not None or self.error is not None:
                raise ValueError(
                    "a slot whose attempt_status is not_attempted holds no "
                    "report and no error; something produced this one"
                )
        elif self.report is None and self.error is None:
            raise ValueError(
                "an attempted slot holds the report it produced or the error "
                "it hit; neither is an attempt nobody recorded the outcome of"
            )
        _flag("a slot's invalidated", self.invalidated)
        if self.invalidated and self.report is None:
            raise ValueError(
                "a slot with no report cannot be invalidated; there is nothing "
                "there to have gone stale"
            )


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class GateDefinition:
    """The schema of a gate: what it asks for, and what it governs.

    ``blocked_actions`` and ``remedies`` belong to the DEFINITION, not to any
    one run of it, which is why :func:``aggregate_gate`` copies them into the
    result unedited.
    """

    name: str
    version: int
    requirements: tuple[ReportRequirement, ...]
    blocked_actions: tuple[str, ...] = ()
    remedies: tuple[Remedy, ...] = ()

    def __post_init__(self) -> None:
        _code("a gate's name", self.name)
        _count("a gate's version", self.version)
        if self.version < 1:
            raise ValueError(f"a gate's version starts at 1; got {self.version!r}")
        _tuple_of("a gate's requirements", self.requirements, ReportRequirement)
        if not self.requirements:
            raise ValueError(
                "a gate asks for at least one report; a gate with no "
                "requirements passes everything, silently"
            )
        seen: set[str] = set()
        for item in self.requirements:
            if item.name in seen:
                raise ValueError(
                    f"a gate names the requirement {item.name!r} twice; one of "
                    "the two declarations would decide, and nothing says which"
                )
            seen.add(item.name)
        _code_tuple("a gate's blocked_actions", self.blocked_actions)
        _tuple_of("a gate's remedies", self.remedies, Remedy)

    @property
    def identity(self) -> str:
        """``name@version``: the spelling a task's ``quality_gate`` carries.

        One spelling, because a task asks for a gate by string and a result
        must be checkable against the gate that was asked for.
        """
        return f"{self.name}@{self.version}"

    def requirement(self, name: str) -> ReportRequirement:
        """The declared requirement called ``name``, or a refusal naming it."""
        for item in self.requirements:
            if item.name == name:
                return item
        raise ValueError(
            f"{name!r} is not a requirement of gate {self.identity}; a slot the "
            f"schema never declared would be aggregated as if it had been asked "
            f"for. Declared: {sorted(item.name for item in self.requirements)}"
        )


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class GateResult:
    """One gate's answer: a status always, a verdict only when it judged."""

    meta: ArtifactMeta
    definition: GateDefinition
    status: OperationalStatus
    verdict: Conclusion | None = None
    report_refs: tuple[ArtifactRef, ...] = ()
    findings: tuple[Finding, ...] = ()
    blocked_actions: tuple[str, ...] = ()
    remedies: tuple[Remedy, ...] = ()

    def __post_init__(self) -> None:
        _envelope("a gate result's meta", self.meta, ArtifactKind.EVALUATION_REPORT)
        _instance("a gate result's definition", self.definition, GateDefinition)
        _member("a gate result's status", self.status, OperationalStatus)
        if self.status is OperationalStatus.EVALUATED:
            if self.verdict is None:
                raise ValueError(
                    "an evaluated gate records the verdict it reached; "
                    "evaluated with no verdict says it judged and then forgot "
                    "the answer"
                )
            _member("an evaluated gate's verdict", self.verdict, Conclusion)
        elif self.verdict is not None:
            raise ValueError(
                f"a {self.status.value} gate carries no verdict; it did not "
                f"reach one, and {self.verdict.value!r} beside that status "
                "would be a judgement of a run that never happened"
            )
        _tuple_of("a gate result's report_refs", self.report_refs, ArtifactRef)
        seen: set[str] = set()
        for ref in self.report_refs:
            if ref.artifact_type is not ArtifactKind.EVALUATION_REPORT:
                raise ValueError(
                    f"a gate result's report_refs point at evaluation reports; "
                    f"got a {ref.artifact_type.value}"
                )
            if ref.artifact_id in seen:
                raise ValueError(f"report_refs names {ref.artifact_id} twice")
            seen.add(ref.artifact_id)
        _tuple_of("a gate result's findings", self.findings, Finding)
        _code_tuple("a gate result's blocked_actions", self.blocked_actions)
        _tuple_of("a gate result's remedies", self.remedies, Remedy)


def _ref_to(report: EvaluationReport) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=report.meta.artifact_id,
        revision=report.meta.revision,
        artifact_type=report.meta.artifact_type,
    )


def aggregate_gate(
    definition: GateDefinition,
    *,
    meta: ArtifactMeta,
    prerequisites_ready: bool,
    inputs_current: bool,
    slots: tuple[ReportSlot, ...],
) -> GateResult:
    """§0.6's priority, applied to a schema-checked mapping of slots.

    The slots are indexed by requirement name and then read in the order the
    DEFINITION declares, so the caller's ordering reaches nothing: not the
    status, not the verdict, not the findings, not the report references.
    """
    _instance("aggregate_gate's definition", definition, GateDefinition)
    _instance("aggregate_gate's meta", meta, ArtifactMeta)
    _flag("prerequisites_ready", prerequisites_ready)
    _flag("inputs_current", inputs_current)
    if not isinstance(slots, tuple):
        raise TypeError(f"slots is a tuple of ReportSlot; got {slots!r}")

    by_name: dict[str, ReportSlot] = {}
    for slot in slots:
        _instance("a gate slot", slot, ReportSlot)
        name = slot.requirement.name
        if name in by_name:
            raise ValueError(
                f"two slots claim the requirement {name!r}; the second would "
                "overwrite the first, which is the silent bug §0.6 names"
            )
        declared = definition.requirement(name)
        if declared != slot.requirement:
            raise ValueError(
                f"the slot for {name!r} disagrees with gate {definition.identity}: "
                f"the schema declares {declared!r} and the slot carries "
                f"{slot.requirement!r}. A slot that downgrades a required check "
                "to an optional one would let the gate pass on a report nobody "
                "was allowed to skip"
            )
        by_name[name] = slot

    ordered = tuple(
        (item, by_name.get(item.name)) for item in definition.requirements
    )
    report_refs = tuple(
        _ref_to(slot.report)
        for _, slot in ordered
        if slot is not None and slot.report is not None
    )

    def answer(
        status: OperationalStatus,
        verdict: Conclusion | None,
        findings: list[Finding],
    ) -> GateResult:
        return GateResult(
            meta=meta,
            definition=definition,
            status=status,
            verdict=verdict,
            report_refs=report_refs,
            findings=tuple(findings),
            # Verbatim from the definition (§0.6): the aggregator does not
            # decide what a gate governs. Whether they are actually blocked is
            # read off the status and the verdict.
            blocked_actions=definition.blocked_actions,
            remedies=definition.remedies,
        )

    if not prerequisites_ready:
        return answer(
            OperationalStatus.BLOCKED,
            None,
            [
                Finding(
                    code="prerequisite_missing",
                    message=(
                        f"gate {definition.identity} needs an artifact that does "
                        "not exist yet, so no report could be produced"
                    ),
                    observed="missing",
                    expected="present",
                )
            ],
        )

    stale = [item.name for item, slot in ordered if slot is not None and slot.invalidated]
    if not inputs_current or stale:
        findings = []
        if not inputs_current:
            findings.append(
                Finding(
                    code="inputs_invalidated",
                    message=(
                        f"an input of gate {definition.identity} has been "
                        "invalidated, so its reports judge a subject that has "
                        "moved"
                    ),
                    observed="invalidated",
                    expected="current",
                )
            )
        findings.extend(
            Finding(
                code="report_invalidated",
                message=f"the report for {name!r} has been invalidated",
                observed=name,
                expected="current",
            )
            for name in stale
        )
        return answer(OperationalStatus.INVALIDATED, None, findings)

    errors = [
        Finding(
            code=(
                "required_report_errored"
                if item.required
                else "blocking_optional_report_errored"
            ),
            message=(
                f"the attempt to produce {item.name!r} failed with "
                f"{slot.error.code!r}: {slot.error.message}"
            ),
            observed=slot.error.code,
            expected="a report",
        )
        for item, slot in ordered
        if slot is not None
        and slot.error is not None
        and (item.required or item.optional_error_blocks)
    ]
    if errors:
        return answer(OperationalStatus.ERROR, None, errors)

    findings = []
    failed = False
    abstained = False
    for item, slot in ordered:
        report = slot.report if slot is not None else None
        if not item.required:
            if (
                report is not None
                and report.applicability is Applicability.APPLICABLE
                and report.conclusion is Conclusion.FAIL
            ):
                findings.append(
                    Finding(
                        code="optional_report_failed",
                        message=(
                            f"the optional check {item.name!r} failed; §0.6 lets "
                            "only required checks decide, so this is reported "
                            "rather than counted"
                        ),
                        observed=report.conclusion.value,
                        expected=Conclusion.PASS.value,
                    )
                )
            continue

        if report is None:
            abstained = True
            findings.append(
                Finding(
                    code="required_report_missing",
                    message=f"the required check {item.name!r} produced no report",
                    observed="missing",
                    expected="a report",
                )
            )
            continue
        if report.applicability is Applicability.INAPPLICABLE:
            abstained = True
            findings.append(
                Finding(
                    code="required_report_inapplicable",
                    message=(
                        f"the required check {item.name!r} does not apply to this "
                        "subject, so nothing about it has been verified"
                    ),
                    observed=report.applicability.value,
                    expected=Applicability.APPLICABLE.value,
                )
            )
            continue
        if report.applicability is Applicability.UNVERIFIABLE:
            abstained = True
            findings.append(
                Finding(
                    code="required_report_unverifiable",
                    message=(
                        f"the required check {item.name!r} applies but could not "
                        "be run on what was available"
                    ),
                    observed=report.applicability.value,
                    expected=Applicability.APPLICABLE.value,
                )
            )
            continue
        if report.conclusion is Conclusion.FAIL:
            failed = True
            findings.append(
                Finding(
                    code="required_report_failed",
                    message=f"the required check {item.name!r} failed",
                    observed=report.conclusion.value,
                    expected=Conclusion.PASS.value,
                )
            )
        elif report.conclusion is Conclusion.ABSTAIN:
            abstained = True
            findings.append(
                Finding(
                    code="required_report_abstained",
                    message=(
                        f"the required check {item.name!r} applied and reached no "
                        "conclusion"
                    ),
                    observed=report.conclusion.value,
                    expected=Conclusion.PASS.value,
                )
            )

    # Read off two flags, never off the last slot seen: FAIL outranks ABSTAIN
    # (§0.6), and both outrank PASS.
    if failed:
        verdict = Conclusion.FAIL
    elif abstained:
        verdict = Conclusion.ABSTAIN
    else:
        verdict = Conclusion.PASS
    return answer(OperationalStatus.EVALUATED, verdict, findings)
