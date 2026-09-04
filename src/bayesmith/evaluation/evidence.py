"""What an ``EvidenceResult`` can say about itself, and about another one.

Three reports and a gate. They are the evaluation half of R4: the assembly
lives in ``dispatch.evidence`` and decides nothing here, and nothing here
changes a number -- §2.4's rule that evaluation judges a Result and does not
modify it or choose an algorithm.

``evidence_normalization_audit``
    Every latent's prior, weighed. An evidence is defined only if each prior
    has finite mass and it is THE evidence only if each mass is one, so this
    report is where those two verdicts are filed for a Result that already
    exists. The compile-time gate refuses before assembling; this says the same
    thing about a Result someone hands you later.

``evidence_prior_sensitivity``
    ``d log Z / d log s``, in nats per e-fold of prior width. R4 REPORTS it and
    does not threshold it, and that is deliberate: no derivation for a
    criterion exists yet, and ``CRITERION_SHIFT`` -- the only prior-sensitivity
    number this package has -- is derived from a chain's MCSE of a posterior
    mean, which has nothing to do with nats of log evidence.

    **It is a different report kind from R3's ``prior_sensitivity`` and the
    difference is not bookkeeping.** R3's measures a posterior MODE
    displacement in posterior sigmas. Centre a prior on the mode and the
    displacement is zero, so R3's report PASSes -- while changing that same
    prior's WIDTH moves ``log Z`` by exactly ``-sum(log s)``. The two disagree
    on the perturbation ``Z`` is most sensitive to, and §1.4 invariant 3
    forbids two statistical objects sharing a name.

``evidence_comparability``
    Whether two ``EvidenceResult``s may be divided. Delivered as a report and
    not as a Refusal, because a ``Refusal`` cannot express it: its
    ``__post_init__`` calls ``task_kind(self.task)``, which raises for anything
    that is not one of the five Tasks, so there is no Refusal whose subject is
    a pair of Results.

Dependency direction: this module reads ``artifacts`` and ``dispatch`` and is
read by nothing in ``src``. ``tests/test_layering.py`` holds that.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

from bayesmith.artifacts.base import ArtifactRef, new_artifact_meta
from bayesmith.artifacts.gates import GateDefinition, ReportRequirement
from bayesmith.artifacts.identity import (
    ArtifactKind,
    FingerprintKind,
    changed_fingerprints,
)
from bayesmith.artifacts.refusal import Finding, Remedy
from bayesmith.artifacts.reports import (
    Applicability,
    Conclusion,
    EvaluationReport,
)
from bayesmith.artifacts.results import EvidenceResult
from bayesmith.dispatch.evidence import PriorVerdict, assemble_exact, audit_graph_priors
from bayesmith.dispatch.task import PRODUCER
from bayesmith.graph.graph import Graph

__all__ = [
    "EVIDENCE",
    "EVIDENCE_COMPARABILITY",
    "EVIDENCE_NORMALIZATION_AUDIT",
    "EVIDENCE_PRIOR_SENSITIVITY",
    "comparability_report",
    "normalization_audit_report",
    "prior_sensitivity_report",
]

#: The three report kinds R4 adds. Codes, not schema: ``EvaluationReport``'s
#: ``report_kind`` has been a free code string since R1 and a new kind is not
#: a schema change.
EVIDENCE_NORMALIZATION_AUDIT = "evidence_normalization_audit"
EVIDENCE_PRIOR_SENSITIVITY = "evidence_prior_sensitivity"
EVIDENCE_COMPARABILITY = "evidence_comparability"

#: Prior widths, as multiples of the declared one, that the sensitivity report
#: perturbs across. Not a threshold and not registered as one: nothing is
#: compared against them. They are the abscissa of a reported derivative, and
#: the span is one e-fold either side because that is the unit the derivative
#: is quoted in.
_WIDTH_FACTORS: tuple[float, ...] = (1.0 / math.e, 1.0, math.e)


def _ref_to(result: EvidenceResult) -> ArtifactRef:
    """A reference that carries the revision, like every other one here."""
    return ArtifactRef(
        artifact_id=result.meta.artifact_id,
        revision=result.meta.revision,
        artifact_type=ArtifactKind.RESULT,
    )


def _report(
    subject: EvidenceResult,
    *,
    kind: str,
    applicability: Applicability,
    conclusion: Conclusion,
    findings: tuple[Finding, ...],
    summary: str,
    extra_parents: tuple[ArtifactRef, ...] = (),
) -> EvaluationReport:
    """The envelope every report here writes, assembled once.

    The fingerprints are the SUBJECT's own, exactly as R3's projections do it:
    a report is retired through the inputs of the result it judged, so one
    carrying a bundle of its own would survive a change to the model it was
    taken on.
    """
    ref = _ref_to(subject)
    return EvaluationReport(
        meta=new_artifact_meta(
            artifact_type=ArtifactKind.EVALUATION_REPORT,
            fingerprints=subject.run.fingerprints,
            producer=PRODUCER,
            parent_refs=(ref, *extra_parents),
            summary=summary,
        ),
        subject_ref=ref,
        report_kind=kind,
        applicability=applicability,
        conclusion=conclusion,
        findings=findings,
    )


def normalization_audit_report(
    subject: EvidenceResult, graph: Graph
) -> EvaluationReport:
    """Every latent's prior, weighed, filed against an existing Result.

    PASS when every prior is proper and normalised; FAIL when one is not;
    ABSTAIN when the audit could not decide one. UNVERIFIABLE is reserved for
    the case where nothing could be weighed at all -- a graph with no latents
    is not a model whose priors passed.
    """
    audits = audit_graph_priors(graph)
    if not audits:
        return _report(
            subject,
            kind=EVIDENCE_NORMALIZATION_AUDIT,
            applicability=Applicability.UNVERIFIABLE,
            conclusion=Conclusion.ABSTAIN,
            findings=(
                Finding(
                    code="no_latent_to_audit",
                    message="this graph declares no latent, so there is no "
                    "prior for an evidence to be an integral against",
                    observed=0,
                    expected="at_least_one",
                ),
            ),
            summary="no prior to weigh",
        )

    findings = tuple(
        Finding(
            code=f"prior.{audit.verdict.value}",
            message=audit.reason,
            observed=(audit.latent, audit.verdict.value)
            if audit.mass is None
            else (audit.latent, audit.verdict.value, float(audit.mass)),
            expected=("proper", 1.0),
        )
        for audit in audits
    )
    unusable = [
        a
        for a in audits
        if a.verdict is not PriorVerdict.PROPER or not a.normalised
    ]
    undecided = [a for a in unusable if a.verdict is PriorVerdict.UNVERIFIABLE]
    if undecided:
        conclusion = Conclusion.ABSTAIN
        summary = f"{len(undecided)} prior(s) the mass check could not decide"
    elif unusable:
        conclusion = Conclusion.FAIL
        summary = f"{len(unusable)} prior(s) do not define a p(d)"
    else:
        conclusion = Conclusion.PASS
        summary = f"{len(audits)} prior(s) proper and normalised"
    return _report(
        subject,
        kind=EVIDENCE_NORMALIZATION_AUDIT,
        applicability=Applicability.APPLICABLE,
        conclusion=conclusion,
        findings=findings,
        summary=summary,
    )


def _rescaled(graph: Graph, latent: str, factor: float) -> Graph | None:
    """``graph`` with one latent's declared prior width multiplied.

    ``None`` where the prior has no width to move -- the perturbation is
    defined on a declared scale, and a prior that does not declare one is not
    a prior this report can differentiate against.
    """
    node = graph.node(latent)
    from bayesmith.graph.evaluate import apply_probabilistic

    try:
        declared = apply_probabilistic(graph, node, {})
        loc = float(jnp.asarray(declared.loc))
        scale = float(jnp.asarray(declared.scale))
    except Exception:  # noqa: BLE001 - anything unreadable is the same answer
        return None
    if not (math.isfinite(loc) and math.isfinite(scale) and scale > 0.0):
        return None

    widened = scale * factor
    rebuilt = tuple(
        (
            type(node)(
                **{
                    **{
                        field: getattr(node, field)
                        for field in node.__dataclass_fields__
                        if field != "dist_fn"
                    },
                    "dist_fn": _normal_at(loc, widened),
                }
            )
            if item.name == latent
            else item
        )
        for item in graph.nodes
        for node in (item,)
    )
    return Graph(
        nodes=rebuilt,
        plates=graph.plates,
        joint_prior=graph.joint_prior,
        evidence_terms=graph.evidence_terms,
    )


def _normal_at(loc: float, scale: float) -> Any:
    """A zero-argument ``dist_fn`` for a Normal with these parameters."""
    import numpyro.distributions as dist

    def built() -> Any:
        return dist.Normal(loc, scale)

    return built


def prior_sensitivity_report(
    subject: EvidenceResult, graph: Graph, block: tuple[str, ...]
) -> EvaluationReport:
    """``d log Z / d log s`` per latent, reported and not judged.

    The derivative is taken by central difference across one e-fold either
    side of each declared width, which is the unit it is quoted in, so no
    step size is chosen and nothing is compared against a threshold.

    APPLICABLE x PASS whenever a derivative could be taken at all. R4 does not
    threshold this: a criterion would need a derivation nobody has, and the
    number's job here is to be READ. Filing it as a PASS with the measurement
    in the findings is the honest shape -- the alternative is a threshold
    invented to make the report look decisive.
    """
    measured: list[Finding] = []
    for latent in block:
        values: list[float] = []
        for factor in _WIDTH_FACTORS:
            perturbed = _rescaled(graph, latent, factor)
            if perturbed is None:
                values = []
                break
            try:
                values.append(float(assemble_exact(perturbed, block, {}).log_evidence))
            except Exception:  # noqa: BLE001 - an unusable perturbation is no slope
                values = []
                break
        if len(values) != len(_WIDTH_FACTORS):
            measured.append(
                Finding(
                    code="prior_width_not_perturbable",
                    message=f"{latent}'s declared prior has no width this "
                    f"report can move, so log Z has no slope against one",
                    observed=latent,
                    expected="a_declared_scale",
                )
            )
            continue
        slope = (values[-1] - values[0]) / 2.0
        measured.append(
            Finding(
                code="d_log_evidence_d_log_width",
                message=f"log Z moves {slope:+.6f} nats per e-fold of "
                f"{latent}'s declared prior width",
                observed=(latent, slope, values[0], values[1], values[2]),
                expected="reported_not_thresholded",
            )
        )
    if not measured:
        return _report(
            subject,
            kind=EVIDENCE_PRIOR_SENSITIVITY,
            applicability=Applicability.INAPPLICABLE,
            conclusion=Conclusion.ABSTAIN,
            findings=(
                Finding(
                    code="no_block_to_perturb",
                    message="this evidence names no exactly integrated latent, "
                    "so there is no declared width to differentiate against",
                    observed=0,
                    expected="at_least_one",
                ),
            ),
            summary="nothing to perturb",
        )
    usable = [f for f in measured if f.code == "d_log_evidence_d_log_width"]
    return _report(
        subject,
        kind=EVIDENCE_PRIOR_SENSITIVITY,
        applicability=(
            Applicability.APPLICABLE if usable else Applicability.INAPPLICABLE
        ),
        conclusion=Conclusion.PASS if usable else Conclusion.ABSTAIN,
        findings=tuple(measured),
        summary=f"d log Z / d log s reported for {len(usable)} latent(s)",
    )


def comparability_report(
    first: EvidenceResult, second: EvidenceResult
) -> EvaluationReport:
    """Whether these two evidences may be divided.

    Comparable iff the DATA fingerprint is unchanged between them.
    ``MODEL_SOURCE`` and ``GRAPH_STRUCTURE`` differ by construction in any
    model comparison -- that is what a model comparison IS -- so ``DATA`` is
    the only slot that can be required.

    Both sides are first checked against themselves. Nothing in ``_envelope``
    requires a Result's ``meta.fingerprints`` to equal its ``run.fingerprints``,
    and a rule reading ``meta`` on one artifact and ``run`` on the other would
    compare two different bundles while looking symmetric.
    """
    for label, result in (("first", first), ("second", second)):
        if result.meta.fingerprints != result.run.fingerprints:
            return _report(
                first,
                kind=EVIDENCE_COMPARABILITY,
                applicability=Applicability.UNVERIFIABLE,
                conclusion=Conclusion.ABSTAIN,
                findings=(
                    Finding(
                        code="result_disagrees_with_itself",
                        message=f"the {label} result's meta fingerprints are "
                        f"not its run's, so there is no single bundle to "
                        f"compare the other against",
                        observed=label,
                        expected="meta_equals_run",
                    ),
                ),
                summary="a result carries two fingerprint bundles",
                extra_parents=(_ref_to(second),),
            )

    changed = changed_fingerprints(first.run.fingerprints, second.run.fingerprints)
    other = _ref_to(second)
    if FingerprintKind.DATA in changed:
        return _report(
            first,
            kind=EVIDENCE_COMPARABILITY,
            applicability=Applicability.INAPPLICABLE,
            conclusion=Conclusion.ABSTAIN,
            findings=(
                Finding(
                    code="different_data_semantics",
                    message="these two evidences were computed against "
                    "different data, so their ratio is not a Bayes factor -- "
                    "it is p(d1|M1)/p(d2|M2), which compares nothing",
                    observed=tuple(sorted(item.value for item in changed)),
                    expected="data_unchanged",
                ),
            ),
            summary="not comparable: the data differs",
            extra_parents=(other,),
        )
    return _report(
        first,
        kind=EVIDENCE_COMPARABILITY,
        applicability=Applicability.APPLICABLE,
        conclusion=Conclusion.PASS,
        findings=(
            Finding(
                code="same_data_semantics",
                message="both evidences were computed against the same data, "
                "so their difference is a log Bayes factor",
                observed=(
                    float(first.log_evidence),
                    float(second.log_evidence),
                    float(first.log_evidence - second.log_evidence),
                ),
                expected="data_unchanged",
            ),
        ),
        summary="comparable: log Bayes factor "
        f"{float(first.log_evidence - second.log_evidence):+.6f}",
        extra_parents=(other,),
    )


#: R4's gate, separate from ``model_checking@1`` on purpose.
#:
#: Folding these into ``model_checking`` would move its ``== 8`` requirement
#: pin and the 40320-permutation sweep behind it, for reports about a different
#: artifact kind: ``model_checking`` judges a posterior and this judges an
#: evidence. One gate, one subject.
EVIDENCE = GateDefinition(
    name="evidence",
    version=1,
    requirements=(
        ReportRequirement(name=EVIDENCE_NORMALIZATION_AUDIT, required=True),
        ReportRequirement(name=EVIDENCE_PRIOR_SENSITIVITY, required=False),
        ReportRequirement(name=EVIDENCE_COMPARABILITY, required=False),
    ),
    blocked_actions=(),
    remedies=(
        Remedy(
            action="declare_a_proper_normalised_prior_on_every_latent",
            message=(
                "p(d) is the integral of the likelihood against the prior, so "
                "a prior with infinite mass leaves it undefined and one whose "
                "mass is not one leaves it scaled. Neither affects the "
                "posterior, which is why an evidence has its own gate."
            ),
        ),
        Remedy(
            action="read_the_prior_sensitivity_rather_than_pass_it",
            message=(
                "d log Z / d log s is REPORTED and not thresholded in this "
                "release: no derivation for a criterion exists, and the "
                "posterior-mode criterion this package already has measures a "
                "different perturbation. A PASS here means the slope was "
                "measured, not that it is small."
            ),
        ),
    ),
)


# `jax` is imported for the x64 contexts every caller runs inside.
_ = jax
