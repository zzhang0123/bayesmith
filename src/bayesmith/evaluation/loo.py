"""PSIS-LOO for a result that carries a pointwise log density (§0.9).

Leave-one-out cross-validation by Pareto-smoothed importance sampling is a
mature, general statistic, so §1.5 puts it upstream: ``arviz.loo`` computes it
and this module owns only the three things arviz cannot know -- which result is
being judged, whether LOO applies to it at all, and how the answer is filed.
Nothing here recomputes an elpd, a Pareto k or a reliability rule.

**The chain axis, and why it is supplied here rather than in the export.**
``az.loo`` reads arviz's ``chain``/``draw`` sample dims, and probe_28 §2
measured what happens without them: ``AttributeError: 'DataArray' object has
no attribute 'chain'``. R2's export is nevertheless right to leave an iid
result on one flat ``draw`` axis --
:class:`~bayesmith.artifacts.results.DrawsPosterior`'s ``chain_shape is None``
means "this run had no chain structure to diagnose", and manufacturing one
there would hand a chain diagnostic something to split that never ran. So the
``(1, n)`` shape is a CONSUMER's adaptation to one upstream's axis convention,
it lives in the consumer, and R2's export semantics are untouched. A result
that DID run chains is read at the shape it recorded.

**The verdict is arviz's own.** A high Pareto k says the importance-sampling
ESTIMATE is unreliable; it does not say the model is wrong. So a warning maps
to ABSTAIN, never FAIL (§0.2), and the rule that decides it is arviz's
``good_k`` -- this module adds no second cutoff of its own. That is §0.8's
"one decision, one home" applied to a threshold: two Pareto-k rules would
agree right up until somebody tuned one of them.

**An importance-weighted sample is declined, not averaged over.** R2's export
projects draws, replicated draws, pointwise log density and observed data; it
carries no weights, and ``arviz.loo`` has none to honour. Handed a
:class:`~bayesmith.artifacts.results.WeightedDrawsPosterior` it would
therefore cross-validate the PROPOSAL rather than the posterior, and it would
return the same numbers whether the weights were near-uniform or collapsed
onto one draw -- measured on ``radiometer``: an identical elpd, se, p_loo and
max Pareto k for a sample whose weight ESS is 1997.3 of 2000 and for the same
sample with a weight ESS of 1.000. §0.2's contract is that a check which
cannot apply says so, so this is UNVERIFIABLE / ABSTAIN with a finding naming
the loss, and the branch sits BEFORE the arviz import so the answer is a
property of the artifact rather than of the environment.

**WAIC is not here, and its absence is recorded rather than patched.**
``hasattr(arviz, "waic")`` is False in 1.3.0 (probe_28 §2). §1.5 lists WAIC
among the statistics this package reuses instead of reimplementing, so R3's
"LOO/WAIC" ships as LOO-PSIS and ``tests/evaluation/test_loo.py`` holds the
absence, so that a later arviz growing one is news rather than a silent
divergence.

**ArviZ stays optional.** It is imported inside :func:`loo_report`, never at
module scope, so a clone without the dev extra gets an UNVERIFIABLE report
(§7.3) rather than an ImportError at import time.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bayesmith import __version__
from bayesmith.artifacts.base import (
    ArtifactRef,
    ProducerRef,
    new_artifact_meta,
)
from bayesmith.artifacts.identity import ArtifactKind
from bayesmith.artifacts.refusal import Finding
from bayesmith.artifacts.reports import Applicability, Conclusion, EvaluationReport
from bayesmith.artifacts.results import (
    PosteriorResult,
    PredictiveResult,
    WeightedDrawsPosterior,
)
from bayesmith.bridge.arviz import to_inference_data
from bayesmith.graph.graph import Graph

__all__ = ["REPORT_KIND", "loo_report"]

#: §0.2's row this module fills. A code, not prose: the gate in
#: ``evaluation/gate.py`` looks reports up by it.
REPORT_KIND = "loo_psis"

PRODUCER = ProducerRef(package="bayesmith", version=__version__)


def _pointwise(result: Any) -> Any:
    """The per-observation log density a result carries, whatever it calls it.

    A posterior names it ``pointwise_log_likelihood`` and a predictive names it
    ``pointwise_log_density``; they are the same array in the same layout, and
    LOO wants exactly one of them.
    """
    if isinstance(result, PosteriorResult):
        return result.pointwise_log_likelihood
    if isinstance(result, PredictiveResult):
        return result.pointwise_log_density
    raise TypeError(
        "loo_report judges a posterior or a predictive result; got "
        f"{type(result).__name__}"
    )


def _chain_shape(
    result: Any, override: tuple[int, int] | None, draws: int
) -> tuple[int, int]:
    """``(chains, draws)`` for the export, defaulting to the iid ``(1, n)``.

    Three sources, in falling order of authority: what the caller supplied,
    what the result itself recorded, and -- only when the result recorded
    nothing because it ran no chains -- the single-chain shape ``az.loo``
    requires. The last one is a RESHAPE, not a claim: 2000 iid draws laid out
    as one chain of 2000 are the same 2000 draws.
    """
    if override is not None:
        return override
    recorded = getattr(getattr(result, "representation", None), "chain_shape", None)
    if recorded is not None:
        return recorded
    return (1, draws)


def _weighted(result: Any, source_posterior: Any) -> WeightedDrawsPosterior | None:
    """The importance-weighted sample in play, if the caller handed one over.

    Two places to look, because a result does not always carry the sample it
    was made from. A :class:`~bayesmith.artifacts.results.PosteriorResult`
    holds its own representation; a
    :class:`~bayesmith.artifacts.results.PredictiveResult` holds only a
    ``source_posterior_ref``, so a caller who still has the source hands it in
    the way it hands in ``chain_shape``.

    The four posterior representations are frozen by R1, so naming the one
    that carries weights ENUMERATES rather than guesses -- a ``hasattr`` on
    ``log_weights`` would silently answer "no" for anything spelled
    differently, and answering "no" here is the failure mode this function
    exists to remove.
    """
    for candidate in (result, source_posterior):
        representation = getattr(candidate, "representation", None)
        if isinstance(representation, WeightedDrawsPosterior):
            return representation
    return None


def _finite(value: Any) -> float | None:
    """A float, or None where the number is not one.

    A NaN Pareto k means arviz could not fit the tail, and recording NaN would
    put a value that compares unequal to itself where a consumer expects a
    measurement -- :class:`~bayesmith.artifacts.refusal.Finding` treats an
    unavailable number as None for that reason.
    """
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _report(
    result: Any,
    *,
    applicability: Applicability,
    conclusion: Conclusion,
    findings: tuple[Finding, ...],
    summary: str,
) -> EvaluationReport:
    subject_ref = ArtifactRef(
        artifact_id=result.meta.artifact_id,
        revision=result.meta.revision,
        artifact_type=ArtifactKind.RESULT,
    )
    return EvaluationReport(
        meta=new_artifact_meta(
            artifact_type=ArtifactKind.EVALUATION_REPORT,
            fingerprints=result.run.fingerprints,
            producer=PRODUCER,
            parent_refs=(subject_ref,),
            summary=summary,
        ),
        subject_ref=subject_ref,
        report_kind=REPORT_KIND,
        applicability=applicability,
        conclusion=conclusion,
        findings=findings,
    )


def loo_report(
    result: Any,
    *,
    graph: Graph,
    chain_shape: tuple[int, int] | None = None,
    source_posterior: Any = None,
) -> EvaluationReport:
    """Judge one result's PSIS-LOO, via ``arviz.loo`` over R2's export.

    Args:
        result: a :class:`~bayesmith.artifacts.results.PosteriorResult` or
            :class:`~bayesmith.artifacts.results.PredictiveResult`.
        graph: the model the result came from, handed on to the export for the
            observed data. It is READ, never modified -- this layer evaluates
            results and does not touch the model (§2.4).
        chain_shape: ``(chains, draws)`` to override what the result recorded.
            A predictive result carries no chain shape of its own, so a caller
            holding the source posterior's may supply it here; left out, an
            unchained result is exported as one chain.
        source_posterior: the :class:`~bayesmith.artifacts.results.PosteriorResult`
            a predictive result came from, read ONLY for its representation --
            it is what says whether the sample is importance-weighted, and a
            ``source_posterior_ref`` is a pointer, not a representation. It
            deliberately does not supply ``chain_shape``: that override already
            has a home above, and one argument quietly feeding two decisions is
            how the second one stops being visible.

    Returns:
        An :class:`~bayesmith.artifacts.reports.EvaluationReport` of kind
        ``loo_psis``, filed against ``result``:

        * APPLICABLE / PASS -- arviz reported no reliability warning.
        * APPLICABLE / ABSTAIN -- arviz warned. The ESTIMATE is unreliable,
          which is not a claim about the model (§0.2).
        * INAPPLICABLE / ABSTAIN -- the result carries no pointwise log
          density, so there is no LOO to compute.
        * UNVERIFIABLE / ABSTAIN -- the sample is importance-weighted, and
          R2's export carries no weights for arviz to honour.
        * UNVERIFIABLE / ABSTAIN -- arviz is not installed (§7.3).
    """
    if not isinstance(graph, Graph):
        raise TypeError(f"loo_report's graph is a Graph; got {graph!r}")

    pointwise = _pointwise(result)
    if pointwise is None:
        return _report(
            result,
            applicability=Applicability.INAPPLICABLE,
            conclusion=Conclusion.ABSTAIN,
            findings=(
                Finding(
                    code="no_pointwise_log_likelihood",
                    message=(
                        "leave-one-out is computed from the per-observation log "
                        "density, and this result carries none"
                    ),
                    observed=False,
                    expected=True,
                ),
            ),
            summary="no pointwise log density to cross-validate",
        )

    weighted = _weighted(result, source_posterior)
    if weighted is not None:
        draws = int(weighted.log_weights.value.shape[0])
        return _report(
            result,
            applicability=Applicability.UNVERIFIABLE,
            conclusion=Conclusion.ABSTAIN,
            findings=(
                Finding(
                    code="weights_not_carried_by_export",
                    message=(
                        "this sample is importance-weighted and the ArviZ "
                        "export carries no weights, so arviz.loo would "
                        "cross-validate the proposal rather than the posterior"
                    ),
                    observed=False,
                    expected=True,
                ),
                Finding(
                    code="weighted_sample",
                    message="the weighted representation this report declined to read",
                    observed=(
                        ("representation", type(weighted).__name__),
                        ("method", weighted.method),
                        ("draws", draws),
                        ("recorded_ess", _finite(weighted.ess)),
                        ("recorded_khat", _finite(weighted.khat)),
                        ("recorded_unreliable", bool(weighted.unreliable)),
                    ),
                ),
            ),
            summary="the sample is weighted and the export is not; LOO was not computed",
        )

    try:
        import arviz as az
    except ImportError as exc:
        return _report(
            result,
            applicability=Applicability.UNVERIFIABLE,
            conclusion=Conclusion.ABSTAIN,
            findings=(
                Finding(
                    code="arviz_unavailable",
                    message=(
                        f"PSIS-LOO is arviz's to compute and it is not "
                        f"importable here: {exc}"
                    ),
                    observed=False,
                    expected=True,
                ),
            ),
            summary="arviz is not installed; LOO was not computed",
        )

    draws = int(pointwise.value.shape[0])
    idata = to_inference_data(
        result, graph=graph, chain_shape=_chain_shape(result, chain_shape, draws)
    )
    elpd = az.loo(idata)

    warned = bool(elpd.warning)
    pareto_k = getattr(elpd, "pareto_k", None)
    max_pareto_k = (
        None if pareto_k is None else _finite(np.nanmax(np.asarray(pareto_k)))
    )
    findings = (
        Finding(
            code="psis_reliability",
            message=(
                "arviz reports the importance-sampling estimate unreliable; the "
                "model is not what is in question"
                if warned
                else "arviz reports no Pareto-k warning"
            ),
            observed=warned,
            expected=False,
        ),
        Finding(
            code="loo_psis_estimate",
            message="elpd_loo and the diagnostics arviz returned beside it",
            observed=(
                ("elpd", _finite(elpd.elpd)),
                ("se", _finite(elpd.se)),
                ("p_loo", _finite(elpd.p)),
                ("n_data_points", int(elpd.n_data_points)),
                ("n_samples", int(elpd.n_samples)),
                ("max_pareto_k", max_pareto_k),
                ("good_k", _finite(getattr(elpd, "good_k", None))),
                ("scale", str(elpd.scale)),
            ),
        ),
        Finding(
            code="arviz_version",
            message="the upstream that computed the estimate above",
            observed=az.__version__,
        ),
    )
    return _report(
        result,
        applicability=Applicability.APPLICABLE,
        conclusion=Conclusion.ABSTAIN if warned else Conclusion.PASS,
        findings=findings,
        summary="psis-loo via arviz",
    )
