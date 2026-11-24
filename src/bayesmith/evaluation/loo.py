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

**The version a report records is the ESTIMATOR's, and ``arviz`` is not it.**
``az.loo`` is a re-export: ``az.loo.__module__`` is ``arviz_stats.loo.loo``,
so the elpd, ``se``, ``p_loo`` and Pareto k above are ``arviz-stats``'s
numbers and ``az.__version__`` is a fact about the umbrella. Measured
2026-09-06: arviz-stats 1.3.1 -> 1.3.2 changed ``se`` by ``sqrt(n / (n - 1))``
-- a ddof 0->1 change that broke two pinned constants in
``tests/evaluation/test_loo.py`` -- while ``az.__version__`` read ``1.3.0`` on
both sides. A report stored before that release and one stored after were
therefore indistinguishable by their own provenance, on exactly the number
that moved. So ``loo_estimator_version`` names the distribution that provides
the callable that ran, and ``arviz_version`` stays beside it recording what it
actually is.

**That provenance is READ off the callable, not spelled.** A constant
``"arviz-stats"`` here would be a claim about today's arviz, and this finding
exists because the last such claim went stale; a later arviz that computes
``loo`` in-house again would walk straight past it. :func:`_estimator_origin`
asks ``az.loo`` where it lives instead, so the finding is true by
construction rather than by maintenance.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
from typing import Any

import numpy as np

from bayesmith.artifacts.base import (
    ArtifactRef,
    new_artifact_meta,
)
from bayesmith.artifacts.identity import (
    ArtifactKind,
    FingerprintKind,
    ModelRef,
    fingerprint,
)
from bayesmith.artifacts.refusal import Finding
from bayesmith.artifacts.reports import Applicability, Conclusion, EvaluationReport
from bayesmith.artifacts.results import (
    DrawsPosterior,
    PosteriorResult,
    PredictiveResult,
    WeightedDrawsPosterior,
)
from bayesmith.bridge.arviz import to_inference_data
from bayesmith.dispatch._task_identity import (
    data_manifest,
    graph_manifest_for_identifier,
)
from bayesmith.dispatch.task import PRODUCER
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Probabilistic

__all__ = ["REPORT_KIND", "loo_report"]

#: §0.2's row this module fills. A code, not prose: the gate in
#: ``evaluation/gate.py`` looks reports up by it.
REPORT_KIND = "loo_psis"



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


def _source_weighting(
    result: PosteriorResult | PredictiveResult,
    source_posterior: PosteriorResult | None,
) -> tuple[str | None, WeightedDrawsPosterior | None]:
    """Read the source representation or the weighting recorded at execution."""
    recorded = None
    if isinstance(result, PosteriorResult):
        posterior = result
    else:
        recorded = dict(result.run.approximation.details).get("source_weighting")
        if source_posterior is None:
            return recorded, None
        ref = result.source_posterior_ref
        if not isinstance(source_posterior, PosteriorResult) or (
            source_posterior.meta.artifact_id != ref.artifact_id
            or source_posterior.meta.revision != ref.revision
        ):
            raise TypeError(
                "source_posterior must match the predictive source_posterior_ref"
            )
        posterior = source_posterior
    representation = posterior.representation
    if isinstance(representation, WeightedDrawsPosterior):
        weighting, weighted = "importance", representation
    elif isinstance(representation, DrawsPosterior):
        weighting, weighted = "equal", None
    else:
        return None, None
    if recorded is not None and recorded != weighting:
        raise TypeError("source posterior weighting differs from the predictive record")
    return weighting, weighted


def _graph_identity_finding(result, graph, model_ref, extra_data) -> Finding | None:
    """Verify the graph inputs that select the observation columns and their factors."""
    recorded = result.run.fingerprints
    identifier = dict(result.run.approximation.details).get("model_identifier")
    if model_ref is not None:
        if not isinstance(model_ref, ModelRef):
            raise TypeError("model_ref must be the ModelRef used to compile the result")
        if fingerprint(FingerprintKind.MODEL_SOURCE, model_ref) != recorded.model_source:
            return Finding(code="loo_graph_mismatch", message="The model source fingerprint differs.")
        identifier = model_ref.identifier
    if identifier is None:
        return Finding(
            code="loo_model_ref_required",
            message="Supply the original model_ref to verify this result's graph structure.",
        )
    structure = graph_manifest_for_identifier(graph, identifier)
    if fingerprint(FingerprintKind.GRAPH_STRUCTURE, structure) != recorded.graph_structure:
        return Finding(code="loo_graph_mismatch", message="The graph structure fingerprint differs.")
    if fingerprint(FingerprintKind.DATA, data_manifest(graph, extra_data)) != recorded.data:
        return Finding(
            code="loo_graph_mismatch",
            message="The graph data fingerprint differs. Supply the original data, masks and extra_data.",
        )
    return None


def _dependent_factor(graph: Graph, retained: set[str]) -> tuple[str, str] | None:
    """Find a retained observation used to parameterise another fitted factor.

    A deterministic node carries all of its parents' inputs forward. At a
    probabilistic node the value in the fitted joint is an observed datum or
    a latent draw. A fully masked observation contributes no factor, and its
    recorded value stays fixed when another observation is left out.
    """
    origins: dict[str, str] = {}
    for node in graph.nodes:
        parent = next((origins[name] for name in node.parents if name in origins), None)
        if isinstance(node, Probabilistic):
            if parent is not None and (node.is_latent or node.name in retained):
                return parent, node.name
            if node.name in retained:
                origins[node.name] = node.name
        elif parent is not None:
            origins[node.name] = parent
    return None


def _retain_observed_units(idata: Any, graph: Graph) -> None:
    """Select the same site order and row-major units as pointwise_log_likelihood."""
    if all(graph.node(name).observed_mask is None for name in graph.observed):
        return
    masks = []
    for name in graph.observed:
        mask = graph.node(name).observed_mask
        shape = idata["observed_data"][name].shape
        masks.append(np.broadcast_to(True if mask is None else mask, shape).ravel())
    retained = np.concatenate(masks)
    if not retained.all():
        from xarray import Dataset

        likelihood = idata["log_likelihood"]["log_likelihood"].values
        chains, draws = likelihood.shape[:2]
        values = likelihood.reshape(chains, draws, -1)[:, :, retained]
        idata["log_likelihood"] = Dataset(
            {"log_likelihood": (("chain", "draw", "observation"), values)}
        )


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


#: The module maps to exactly one installed distribution, and it has a version.
ORIGIN_FOUND = "found"
#: More than one distribution provides the module, so no single version is its
#: version. Recording either one would be a coin toss written down as a fact.
ORIGIN_AMBIGUOUS = "ambiguous"
#: Nothing installed claims the module, or the lookup did not complete. The
#: module IMPORTED -- the estimate above is its output -- so this is never
#: "the package is absent"; it is "this run could not say which package it is",
#: and ``dispatch/task.py``'s ``EXTRA_UNKNOWN`` is the same ruling one layer
#: down: a probe that reports "I could not look" as "I looked and found
#: nothing" has made its silence indistinguishable from its answer.
ORIGIN_UNMAPPED = "unmapped"


def _estimator_origin(estimator: Any) -> tuple[tuple[str, object], ...]:
    """Which distribution, at which version, provides the callable that ran.

    ``__module__`` says where the function that produced the estimate lives and
    :func:`importlib.metadata.packages_distributions` maps that top-level
    package back to the distribution that installed it, so the answer is a
    property of the object rather than of this line. ``az.loo`` is patched or
    wrapped in some environments; ``__module__`` then names the wrapper, which
    is still the honest answer to "what computed this".

    Three outcomes, not two, for the reason :data:`ORIGIN_UNMAPPED` gives.
    ``distribution`` and ``version`` are None on both of the other two, so a
    consumer reads ``lookup`` before it reads a version rather than mistaking
    an absent one for a measurement.
    """
    module = getattr(estimator, "__module__", None) or ""
    qualname = getattr(estimator, "__qualname__", None) or repr(estimator)
    called = f"{module}.{qualname}" if module else qualname
    top = module.split(".", 1)[0]

    def origin(
        distribution: str | None, version: str | None, lookup: str
    ) -> tuple[tuple[str, object], ...]:
        return (
            ("callable", called),
            ("distribution", distribution),
            ("version", version),
            ("lookup", lookup),
        )

    try:
        providers = sorted(importlib.metadata.packages_distributions().get(top, ()))
        installed = (
            importlib.metadata.version(providers[0]) if len(providers) == 1 else None
        )
    except Exception:  # noqa: BLE001 -- any failure is "could not look"
        return origin(None, None, ORIGIN_UNMAPPED)
    if len(providers) == 1:
        return origin(providers[0], installed, ORIGIN_FOUND)
    if providers:
        return origin(", ".join(providers), None, ORIGIN_AMBIGUOUS)
    return origin(None, None, ORIGIN_UNMAPPED)


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
    model_ref: ModelRef | None = None,
    extra_data: tuple[tuple[str, Any], ...] = (),
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
            a predictive result came from. Its ID and revision must match.
            It supplies weighting provenance for older predictives and any latent
            draws omitted from the predictive that the ArviZ export needs. It
            deliberately does not supply ``chain_shape``: that override already
            has a home above, and one argument quietly feeding two decisions is
            how the second one stops being visible.

        model_ref: original model reference for older results which do not record
            their model identifier. When supplied, its source fingerprint is checked.
        extra_data: additional conditioning data supplied to compile_task, if any.

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

    weighting, weighted = _source_weighting(result, source_posterior)
    if weighting not in ("importance", "equal"):
        return _report(
            result,
            applicability=Applicability.UNVERIFIABLE,
            conclusion=Conclusion.ABSTAIN,
            findings=(
                Finding(
                    code="source_weighting_unknown",
                    message=(
                        "A predictive result with no recorded weighting needs its source posterior."
                    ),
                ),
            ),
            summary="source posterior weighting is unknown",
        )
    if weighting == "importance":
        description = ()
        if weighted is not None:
            description = (
                Finding(
                    code="weighted_sample",
                    message="the weighted representation supplied to this check",
                    observed=(
                        ("representation", type(weighted).__name__),
                        ("method", weighted.method),
                        ("draws", int(weighted.log_weights.value.shape[0])),
                        ("recorded_ess", _finite(weighted.ess)),
                        ("recorded_khat", _finite(weighted.khat)),
                        ("recorded_unreliable", bool(weighted.unreliable)),
                    ),
                ),
            )
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
                *description,
            ),
            summary="the sample is weighted and the export is not; LOO was not computed",
        )

    identity = _graph_identity_finding(result, graph, model_ref, extra_data)
    if identity is not None:
        return _report(
            result,
            applicability=Applicability.UNVERIFIABLE,
            conclusion=Conclusion.ABSTAIN,
            findings=(identity,),
            summary="the supplied graph could not be matched to the fitted result",
        )

    retained = {
        name for name in graph.observed
        if graph.node(name).observed_mask is None or np.any(graph.node(name).observed_mask)
    }
    if not retained:
        return _report(
            result,
            applicability=Applicability.INAPPLICABLE,
            conclusion=Conclusion.ABSTAIN,
            findings=(
                Finding(
                    code="no_observed_units", message="Every observation was held out."
                ),
            ),
            summary="no fitted observations to cross-validate",
        )

    dependency = _dependent_factor(graph, retained)
    if dependency is not None:
        parent, child = dependency
        return _report(
            result,
            applicability=Applicability.INAPPLICABLE,
            conclusion=Conclusion.ABSTAIN,
            findings=(
                Finding(
                    code="dependent_observation_factors",
                    message=(
                        f"Leaving out {parent!r} also changes the factor for {child!r}. "
                        "Ordinary pointwise PSIS-LOO requires a single removable likelihood factor."
                    ),
                    observed=dependency,
                ),
            ),
            summary="dependent observations require conditional leave-one-out densities",
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
    export_result = result
    if isinstance(result, PredictiveResult) and not set(graph.latents).issubset(
        array.name for array in result.latent_draws
    ):
        if source_posterior is None:
            return _report(
                result,
                applicability=Applicability.UNVERIFIABLE,
                conclusion=Conclusion.ABSTAIN,
                findings=(
                    Finding(
                        code="source_draws_required",
                        message=(
                            "The predictive omitted latent draws needed by the ArviZ projection. "
                            "Supply its source posterior."
                        ),
                    ),
                ),
                summary="source draws are required for the ArviZ projection",
            )
        export_result = dataclasses.replace(
            result, latent_draws=source_posterior.representation.draws
        )
    idata = to_inference_data(
        export_result, graph=graph, chain_shape=_chain_shape(result, chain_shape, draws)
    )
    _retain_observed_units(idata, graph)
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
            code="loo_estimator_version",
            message="the distribution whose loo computed the estimate above",
            observed=_estimator_origin(az.loo),
        ),
        Finding(
            code="arviz_version",
            message=(
                "the arviz umbrella this module imported; it re-exports loo "
                "rather than computing it"
            ),
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
