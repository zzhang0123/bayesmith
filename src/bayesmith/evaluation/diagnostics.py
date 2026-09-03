"""Two design-time diagnostics, filed on §0 ruling 7's two axes.

:func:`~bayesmith.diagnose.identifiability.identifiability` and
:func:`~bayesmith.diagnose.sensitivity.prior_sensitivity` already reach a
verdict, against thresholds that already have an owner:
:data:`~bayesmith.diagnose.identifiability.DEFAULT_RANK_RTOL` and
:data:`~bayesmith.diagnose.sensitivity.CRITERION_SHIFT`, each justified where
it is declared.  What is missing is a way for a gate to READ those verdicts --
an :class:`~bayesmith.artifacts.reports.EvaluationReport` whose ``subject_ref``
names the result being judged and whose findings carry the numbers the verdict
was taken on.

So this module PROJECTS.  It reads ``nullity``, ``participation``, ``worst``,
``verified``, ``refit_converged`` and ``criterion_std`` off the reports and
files them.  It compares nothing against a number of its own -- the plan's §0.2
puts it plainly: a second copy of a threshold is how two answers to one
question come to be written down, and they agree until the day one of them is
retuned.  ``tests/evaluation/test_diagnostics.py`` asserts that this file
contains no float literal at all, which is the same rule stated as something
that can fail.

**Three verdict shapes, and the order between them is load-bearing.**

* APPLICABLE × PASS / FAIL -- the diagnostic ran and decided.
* APPLICABLE × ABSTAIN -- ``prior_sensitivity`` ran, and its two routes did
  not verify each other (the refit did not converge, or some element of
  ``verified`` is False).  A shift no second route confirmed is not evidence
  either way, so this is checked BEFORE the shift is compared: the measured
  fixture in the tests has a worst shift of +2.2 sigma with a False in
  ``verified``, and a projection that looked at the shift first would file a
  confident FAIL taken from a number nothing cross-checked.
* UNVERIFIABLE × ABSTAIN -- the diagnostic refused.  §0.10's headline case is
  a float32 ambient precision, which both functions refuse by name and which
  probe_28 §9 hit on its own first run; the refusal text goes into the finding
  verbatim, because half of it is the remedy (the GRAPH has to be built inside
  ``jax.enable_x64(True)``, not only the call).  A caller assembling a gate
  gets a report rather than an exception, which is §7.3's "degrade
  gracefully" and §2.4's "a run failure must not be dressed as a FAIL".

**Layering.** ``evaluation`` reads ``diagnose``, ``artifacts`` and ``graph``;
none of them reads back (``tests/test_layering.py``).  Nothing here imports
arviz, at module scope or otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import numpy as np

from bayesmith.artifacts.base import (
    ArtifactRef,
    new_artifact_meta,
)
from bayesmith.artifacts.identity import ArtifactKind
from bayesmith.artifacts.refusal import Finding
from bayesmith.artifacts.reports import (
    Applicability,
    Conclusion,
    EvaluationReport,
)
from bayesmith.artifacts.results import PosteriorResult
from bayesmith.diagnose.identifiability import (
    DEFAULT_RANK_RTOL,
    IdentifiabilityReport,
    identifiability,
)
from bayesmith.diagnose.sensitivity import (
    CRITERION_SHIFT,
    PriorSensitivityReport,
    prior_sensitivity,
)
from bayesmith.dispatch.task import PRODUCER
from bayesmith.errors import BayesmithError
from bayesmith.graph.graph import Graph

__all__ = [
    "IDENTIFIABILITY",
    "PRIOR_SENSITIVITY",
    "identifiability_report",
    "prior_sensitivity_report",
    "DEFAULT_RANK_RTOL",
    "CRITERION_SHIFT",
]

#: The two ``report_kind`` codes of §0.2's table, spelled once. A report kind
#: is a code a consumer branches on (R1 freezes the field as a code, not as an
#: enum), so it is a name here rather than a string literal at three call
#: sites.
IDENTIFIABILITY = "identifiability"
PRIOR_SENSITIVITY = "prior_sensitivity"


def _subject_ref(subject: PosteriorResult) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=subject.meta.artifact_id,
        revision=subject.meta.revision,
        artifact_type=ArtifactKind.RESULT,
    )


def _report(
    subject: PosteriorResult,
    *,
    kind: str,
    applicability: Applicability,
    conclusion: Conclusion,
    findings: tuple[Finding, ...],
    summary: str,
) -> EvaluationReport:
    """The envelope every projection here writes, assembled in one place.

    The fingerprints are the SUBJECT's own, exactly as
    ``dispatch.task._chain_diagnostics_report`` does it: §0.3 retires a report
    through the inputs of the result it judged, so a report carrying a bundle
    of its own would survive a change to the model it was taken on.
    """
    ref = _subject_ref(subject)
    return EvaluationReport(
        meta=new_artifact_meta(
            artifact_type=ArtifactKind.EVALUATION_REPORT,
            fingerprints=subject.run.fingerprints,
            producer=PRODUCER,
            parent_refs=(ref,),
            summary=summary,
        ),
        subject_ref=ref,
        report_kind=kind,
        applicability=applicability,
        conclusion=conclusion,
        findings=findings,
    )


def _refusal_finding(error: BayesmithError) -> Finding:
    """The diagnostic's own refusal, carried verbatim.

    ``observed`` is the exception's TYPE name and ``expected`` the empty
    string: a consumer branching on "was this refused, and by what" needs a
    code, and the prose belongs in ``message`` where nothing branches on it.
    Verbatim rather than summarised because the remedy is inside the text --
    the float32 refusal's last sentence is the one that says the graph has to
    be built inside the block, and it is the half a paraphrase drops.
    """
    return Finding(
        code="diagnostic_refused",
        message=str(error),
        observed=type(error).__name__,
        expected="",
    )


def identifiability_report(
    graph: Graph,
    subject: PosteriorResult,
    *,
    names: Sequence[str] | str | None = None,
    rtol: float = DEFAULT_RANK_RTOL,
) -> EvaluationReport:
    """File :func:`~bayesmith.diagnose.identifiability.identifiability`'s rank
    verdict as a report about ``subject``.

    PASS at nullity 0, FAIL above it; §0.2's row, and the whole comparison.
    The rank cut is ``rtol``'s, decided in ``diagnose`` and passed through so
    that a caller who has read
    :attr:`~bayesmith.diagnose.identifiability.IdentifiabilityReport.weakest_identified`
    can tighten it without this layer acquiring an opinion about the value.

    Args:
        graph: the model ``subject``'s posterior was taken over. Build it
            inside ``with jax.enable_x64(True):`` -- see the module docstring.
        subject: the result this report judges. Only its identity and its
            fingerprints are read; no draw is touched.
        names: which latents to analyse, as
            :func:`~bayesmith.diagnose.identifiability.identifiability` takes
            them. ``None`` is the joint question over all of them, which is
            the one a per-block guard cannot ask.
        rtol: the relative rank tolerance.

    Returns:
        APPLICABLE × PASS when nothing is degenerate, APPLICABLE × FAIL with a
        ``null_participation`` finding naming the latents the first blind
        direction mixes, or UNVERIFIABLE × ABSTAIN carrying the diagnostic's
        refusal.
    """
    try:
        report = identifiability(graph, names=names, rtol=rtol)
    except BayesmithError as error:
        return _report(
            subject,
            kind=IDENTIFIABILITY,
            applicability=Applicability.UNVERIFIABLE,
            conclusion=Conclusion.ABSTAIN,
            findings=(_refusal_finding(error),),
            summary="identifiability could not be evaluated",
        )
    return _report(
        subject,
        kind=IDENTIFIABILITY,
        applicability=Applicability.APPLICABLE,
        conclusion=Conclusion.PASS if report.nullity == 0 else Conclusion.FAIL,
        findings=_identifiability_findings(report),
        summary=f"joint Jacobian rank {report.rank} of {report.n_par}",
    )


def _identifiability_findings(report: IdentifiabilityReport) -> tuple[Finding, ...]:
    """``nullity``, the spectrum it was read off, and who the blindness mixes.

    ``nullity`` comes first and alone decides the verdict -- ``observed ==
    expected`` reproduces it with no arithmetic, which is R3's G8 ("a verdict
    is recomputable from the report's own fields") in its simplest form. The
    spectrum finding is the audit trail underneath it, and
    ``null_participation`` exists only where there is a direction to describe:
    a fully identified model has none, and a finding whose payload would be
    empty says nothing.
    """
    findings = [
        Finding(
            code="nullity",
            message=(
                f"{report.nullity} of {report.n_par} parameter directions "
                "leave the prediction unchanged to first order"
            ),
            observed=report.nullity,
            expected=0,
        ),
        Finding(
            code="rank_spectrum",
            message=(
                f"rank {report.rank} of {report.n_par} over {report.n_data} "
                f"data points; weakest identified direction at "
                f"{report.weakest_identified:.6g} of the largest"
            ),
            observed=(
                report.n_par,
                report.n_data,
                report.rank,
                report.rtol,
                report.threshold,
                report.weakest_identified,
            ),
            expected=report.n_par,
        ),
    ]
    for index in range(report.nullity):
        shares = tuple(
            (name, float(share))
            for name, share in sorted(report.participation(index).items())
        )
        spelled = ", ".join(f"{name} {share:.4f}" for name, share in shares)
        findings.append(
            Finding(
                code="null_participation",
                message=f"blind direction {index} is carried by {spelled}",
                observed=(index, shares),
                expected=None,
            )
        )
    return tuple(findings)


def prior_sensitivity_report(
    graph: Graph,
    subject: PosteriorResult,
    *,
    names: Sequence[str] | str | None = None,
    at: dict[str, jax.Array] | None = None,
) -> EvaluationReport:
    """File :func:`~bayesmith.diagnose.sensitivity.prior_sensitivity`'s worst
    mode displacement as a report about ``subject``.

    §0.2's row: PASS while the worst ``|shift_sigma|`` is below
    :data:`~bayesmith.diagnose.sensitivity.CRITERION_SHIFT`, FAIL at or above
    it, ABSTAIN when the closed form and the refit did not verify each other,
    UNVERIFIABLE when the diagnostic refused. The verification is read FIRST;
    the module docstring says why.

    Args:
        graph: the model, built inside ``with jax.enable_x64(True):``.
        subject: the result this report judges.
        names: which latents' priors to analyse; ``None`` means all of them.
        at: where to expand, as ``prior_sensitivity`` takes it.

    Returns:
        An :class:`~bayesmith.artifacts.reports.EvaluationReport` whose
        ``worst_shift`` finding carries ``(latent, index, shift)`` against
        ``CRITERION_SHIFT``, and whose ``prior_width_margin`` finding carries
        the declared width against the width at which that latent's shift
        would reach it.
    """
    try:
        report = prior_sensitivity(graph, names=names, at=at)
    except BayesmithError as error:
        return _report(
            subject,
            kind=PRIOR_SENSITIVITY,
            applicability=Applicability.UNVERIFIABLE,
            conclusion=Conclusion.ABSTAIN,
            findings=(_refusal_finding(error),),
            summary="prior sensitivity could not be evaluated",
        )
    name, index, shift = report.worst
    verified = int(np.count_nonzero(report.verified))
    unverified = not report.refit_converged or verified < report.n_par
    if unverified:
        conclusion = Conclusion.ABSTAIN
    elif abs(shift) < CRITERION_SHIFT:
        conclusion = Conclusion.PASS
    else:
        conclusion = Conclusion.FAIL
    return _report(
        subject,
        kind=PRIOR_SENSITIVITY,
        applicability=Applicability.APPLICABLE,
        conclusion=conclusion,
        findings=_sensitivity_findings(report, verified=verified),
        summary=f"worst prior-induced shift {shift:+.4g} sigma at {name}[{index}]",
    )


def _sensitivity_findings(
    report: PriorSensitivityReport, *, verified: int
) -> tuple[Finding, ...]:
    """The displacement, the width that would have caused it, and the check.

    Three findings and each is read by a different reader. ``worst_shift`` is
    the verdict's own pair -- ``|observed[2]| < expected`` reproduces PASS
    against FAIL. ``refit_verification`` is the ABSTAIN's, and its ``expected``
    spells the whole premise (the refit converged, and every element agreed)
    rather than a boolean whose two ways of being False are indistinguishable.
    ``prior_width_margin`` decides nothing and is the number a reader CHOOSING
    a prior wants: the declared width, against the width at which this latent
    would move by ``CRITERION_SHIFT``.
    """
    name, index, shift = report.worst
    column = report.for_latent(name)
    prior_std = float(column["prior_std"].ravel()[index])
    criterion_std = float(column["criterion_std"].ravel()[index])
    return (
        Finding(
            code="worst_shift",
            message=(
                f"the declared priors moved {name}[{index}] by {shift:+.6g} "
                "posterior sigma"
            ),
            observed=(name, index, shift),
            expected=CRITERION_SHIFT,
        ),
        Finding(
            code="prior_width_margin",
            message=(
                f"{name}[{index}] would shift by the criterion at a prior "
                f"width of {criterion_std:.6g}; the declared width is "
                f"{prior_std:.6g}"
            ),
            observed=(name, index, prior_std),
            expected=criterion_std,
        ),
        Finding(
            code="refit_verification",
            message=(
                f"{verified} of {report.n_par} elements agreed between the "
                "closed form and the refit"
                + ("" if report.refit_converged else "; the refit did not converge")
            ),
            observed=(report.refit_converged, verified, report.n_par),
            expected=(True, report.n_par, report.n_par),
        ),
    )
