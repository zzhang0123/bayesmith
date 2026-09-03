"""Prior and posterior predictive checks: does the model generate data like this?

One question, asked twice from different parameter sources, and answered as an
:class:`~bayesmith.artifacts.reports.EvaluationReport` on §0 ruling 7's two
axes.  A *posterior* predictive check reads a
:class:`~bayesmith.artifacts.results.PredictiveResult` -- replicated datasets
drawn at the posterior's own draws -- and asks whether the observed data look
like one of them.  A *prior* predictive check reads a
:class:`~bayesmith.artifacts.results.SimulationResult` whose parameter source
is the PRIOR and asks the same question of the prior predictive distribution,
which is a question about the priors rather than about the fit.

**A PASS here does not mean the model is correct, and this module is required
to say so where it cannot be skipped over.**  Measured, on this checkout, by
``docs/probes/probe_28_model_checking_seams.py`` §1 and re-measured through the
typed path while writing this file: a straight line fitted to data with a real
quadratic term of curvature 0.6 scores ``p = 0.0000`` on ``sd`` and on
``residual_sd`` -- caught, loudly.  The SAME model with curvature 0.15 scores
``0.5495`` and ``0.3410``: a misspecification of exactly the kind this check
exists to find, sitting comfortably inside the band.  So the honest reading of
a pass is "these statistics of these replicated datasets do not separate the
model from the data", and nothing wider.  It is written into every report's
``meta.summary`` as well as here, because a summary travels with the artifact
and a module docstring does not.

**The discrepancy contract (§0.3).**  A discrepancy is a callable
``(y, loc) -> one scalar per draw``.  Both arguments arrive already flattened
to ``(draws, units)``: ``y`` is the replicated (or the observed, broadcast)
data and ``loc`` is the observation's mean at the same draw, recomputed from
the latent draws through :func:`~bayesmith.exact.gaussian.observation_parts` --
the same loc/scale walk R2's predictive seam uses, never a second forward
model.  Five defaults are provided; four read only ``y`` and the fifth reads
``y - loc``.  A user callable is recorded by its IMPORTABLE IDENTITY and never
as an object: see :func:`discrepancy_identity`.

**Two registered thresholds and no third.**  :data:`ALPHA
<bayesmith.evaluation.ALPHA>` (D104) is the declared two-sided false-positive
rate; :data:`DRAW_FLOOR` (D105) is derived from it.  Everything else here --
the band, the tail mass, the weighting -- is arithmetic on those two.

**The report-level rate is not α, and the reason is arithmetic.**  §0.4 fixes
the band at ``[α/2, 1 - α/2]`` PER discrepancy and reserves Bonferroni for
held-out points and SBC coordinates, so a report over ``K`` cells fails a
correct model with probability up to ``1 - (1 - α)**K`` -- 0.23 at the default
five.  That is the frozen ruling's consequence, not a defect to be corrected
here by inventing a factor §0.4 did not grant; what it obliges is that the
number be COUNTABLE, which is why every cell gets its own finding and why the
close-out multiplies cells by α rather than guessing.  This package's own
acceptance tests pin measured p-values at fixed seeds, so none of them is a
random trial in the first place.

Layering: ``artifacts``, ``exact.gaussian``, ``graph`` and this package's own
``ALPHA``.  Deliberately NOT ``dispatch``: ``tests/test_layering.py``'s
``test_graph_is_the_foundation_and_dispatch_is_the_top`` asserts that nothing
depends on ``dispatch`` at module scope, and this module needs nothing from it
that ``exact.gaussian`` does not already provide.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Sequence
from typing import Any

import jax
import numpy as np

from bayesmith import __version__
from bayesmith.artifacts.base import (
    ArtifactRef,
    NamedArray,
    ProducerRef,
    new_artifact_meta,
)
from bayesmith.artifacts.identity import ArtifactKind
from bayesmith.artifacts.refusal import Finding
from bayesmith.artifacts.reports import (
    Applicability,
    Conclusion,
    EvaluationReport,
)
from bayesmith.artifacts.results import (
    PosteriorResult,
    PredictiveResult,
    SimulationResult,
    WeightedDrawsPosterior,
)
from bayesmith.artifacts.tasks import ParameterSourceKind
from bayesmith.errors import NotGaussian
from bayesmith.evaluation import ALPHA
from bayesmith.exact.correct import self_normalise
from bayesmith.exact.gaussian import observation_parts
from bayesmith.graph.evaluate import evaluate
from bayesmith.graph.graph import Graph

__all__ = [
    "DRAW_FLOOR",
    "DEFAULT_DISCREPANCIES",
    "PRODUCER",
    "mean",
    "sd",
    "smallest",
    "largest",
    "residual_sd",
    "discrepancy_identity",
    "tail_mass_within_rate",
    "draws_resolve_the_band",
    "posterior_predictive_check",
    "prior_predictive_check",
]

#: The package and version that wrote a report from this layer.
#:
#: The same value as :data:`bayesmith.dispatch.task.PRODUCER`, spelled again
#: rather than imported, because importing it would put an ``evaluation ->
#: dispatch`` edge at module scope and ``tests/test_layering.py`` asserts that
#: the in-degree of ``dispatch`` is zero.  A second spelling of one fact is the
#: defect this repository has spent the most time repairing, so it is not left
#: to a hope: ``tests/evaluation/test_checks.py`` compares the two objects, and
#: editing either alone turns the suite red.
PRODUCER = ProducerRef(package="bayesmith", version=__version__)

#: D105.  The smallest number of draws at which a p-value can LAND in the tail
#: the declared rate reserves, derived from D104 and carrying no number of its
#: own: ``ceil(1 / (ALPHA / 2)) = 40``.
#:
#: A p-value computed from ``N`` equally weighted draws is a multiple of
#: ``1/N``, so its resolution is ``1/N``.  The band's tail is ``ALPHA / 2``
#: wide on each side; if ``1/N`` exceeds that width, no attainable p-value lies
#: strictly inside a tail without also being 0 or 1, and the check has stopped
#: being a measurement and become a coin.  The existing suite's
#: ``ComputeBudget(draws=8)`` is exactly that case -- p can only be one of
#: 0, 0.125, ..., 1.0 -- and a check run on it is reported as an ABSTAIN
#: rather than as the PASS it would otherwise always produce.
#:
#: Provenance is **derived**: the formula is the registered thing, so a change
#: to ALPHA moves this with it instead of leaving a stale 40 behind.
DRAW_FLOOR: int = math.ceil(1.0 / (ALPHA / 2.0))


def draws_resolve_the_band(draws: int) -> bool:
    """D105: ``draws >= DRAW_FLOOR`` -- this many draws can resolve a tail.

    Closed at the boundary: at exactly ``DRAW_FLOOR`` draws the resolution
    ``1/N`` equals the tail width ``ALPHA/2``, so a p-value can reach the
    tail's edge, which is the least this check needs to be able to say
    anything.  Below it the tail is narrower than one draw.
    """
    return draws >= DRAW_FLOOR


def tail_mass_within_rate(tail_mass: float) -> bool:
    """D104: ``tail_mass >= ALPHA / 2`` -- the p-value is inside the band.

    ``tail_mass`` is ``min(p, 1 - p)``, so this ONE comparison is the whole
    two-sided band ``[ALPHA/2, 1 - ALPHA/2]``: a p below the lower edge and a
    p above the upper edge both give a tail mass below ``ALPHA/2``.  Written
    as one number against one threshold rather than as two comparisons because
    the report records what was compared, and ``1 - p >= ALPHA/2`` and
    ``p <= 1 - ALPHA/2`` are not the same floating-point question -- a reader
    recomputing the verdict from the finding would occasionally get the other
    answer at the last ULP.

    Closed at the boundary: the declared rate is the mass a correct model is
    ALLOWED to put in the tails, so a p-value sitting exactly on the edge has
    not exceeded it.
    """
    return tail_mass >= ALPHA / 2.0


# ------------------------------------------------------------- discrepancies


def mean(y: np.ndarray, loc: np.ndarray) -> np.ndarray:
    """The mean of the observations, per draw.  Reads ``y`` only."""
    del loc
    return np.mean(y, axis=-1)


def sd(y: np.ndarray, loc: np.ndarray) -> np.ndarray:
    """The spread of the observations, per draw.  Reads ``y`` only."""
    del loc
    return np.std(y, axis=-1)


def smallest(y: np.ndarray, loc: np.ndarray) -> np.ndarray:
    """The minimum observation, per draw.  §0.3's ``min``.  Reads ``y`` only.

    Spelled ``smallest`` rather than ``min`` because a module-level ``min``
    would shadow the builtin this module calls a few lines away, and the
    recorded identity of a discrepancy is its module and qualname -- so the
    name is not a private detail that can be renamed later without notice.
    """
    del loc
    return np.min(y, axis=-1)


def largest(y: np.ndarray, loc: np.ndarray) -> np.ndarray:
    """The maximum observation, per draw.  §0.3's ``max``.  Reads ``y`` only."""
    del loc
    return np.max(y, axis=-1)


def residual_sd(y: np.ndarray, loc: np.ndarray) -> np.ndarray:
    """The spread of ``y - loc``, per draw.  The one default that reads ``loc``.

    The sharpest of the five on this package's own fixtures, and the reason
    the default set is not "four functions of the data": a quadratic term the
    linear model cannot express leaves the data's residuals wider than any
    replicated dataset's, which is what scores 0.0000 on ``curved_line(0.6)``.
    """
    return np.std(y - loc, axis=-1)


#: §0.3's five: stable, covariate-free, and named so a report can record which
#: was used.  Order is the order §0.3 lists them in, and it is preserved into
#: the report's findings so two runs of one check compare equal.
DEFAULT_DISCREPANCIES: tuple[Callable[..., Any], ...] = (
    mean,
    sd,
    smallest,
    largest,
    residual_sd,
)


def discrepancy_identity(discrepancy: object) -> str:
    """``module.qualname``, and only when importing it back returns THIS object.

    §0.3 says a user discrepancy is recorded by its importable identity, on
    :func:`bayesmith.dispatch.amortized._embed_identity`'s rule -- and adds
    that a lambda or a REPL function is refused.  Those two clauses do not
    agree as written, which was measured rather than assumed:
    ``_embed_identity`` reads ``__module__`` and ``__qualname__`` and refuses
    only when one of them is empty, and a lambda has ``__qualname__ ==
    '<lambda>'``, which is not empty.  Handed one, it returns
    ``'__main__.<lambda>'`` -- an address that will never resolve again.

    So the rule here is the same identity with the missing half supplied: the
    identity must RESOLVE, by importing the module and walking the qualname,
    and the object it resolves to must be the one that was passed.  That
    refuses a lambda (``getattr(module, '<lambda>')`` does not exist), a
    closure (``'outer.<locals>.inner'``), a REPL function (``__main__`` has no
    file to re-import from in another process), and the subtler case neither
    of the first two covers: a module-level name that has since been rebound
    to something else, where the address survives and its meaning does not.

    Raises:
        ValueError: when there is no identity, or it does not resolve back.
    """
    module_name = getattr(discrepancy, "__module__", None)
    qualname = getattr(discrepancy, "__qualname__", None)
    if not isinstance(module_name, str) or not module_name:
        raise ValueError(
            f"the discrepancy {discrepancy!r} has no __module__; a report "
            "records where a statistic was defined, not the object itself"
        )
    if not isinstance(qualname, str) or not qualname:
        raise ValueError(
            f"the discrepancy {discrepancy!r} has no __qualname__; a report "
            "records where a statistic was defined, not the object itself"
        )
    identity = f"{module_name}.{qualname}"
    try:
        resolved: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            resolved = getattr(resolved, part)
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            f"the discrepancy {identity!r} does not import back: {exc}. A "
            "lambda, a function defined inside another function and a REPL "
            "definition all have an address and none of them has a home, so "
            "recording one would put a name in the artifact that no later "
            "reader can turn into the statistic that was actually computed"
        ) from exc
    if resolved is not discrepancy:
        raise ValueError(
            f"{identity!r} resolves to {resolved!r}, which is not the "
            f"discrepancy that was passed ({discrepancy!r}); the identity a "
            "report would record names a different statistic"
        )
    return identity


# ------------------------------------------------------------------ plumbing


def _result_ref(result: PosteriorResult | PredictiveResult | SimulationResult):
    """The version-identified reference an evaluation report points at."""
    return ArtifactRef(
        artifact_id=result.meta.artifact_id,
        revision=result.meta.revision,
        artifact_type=ArtifactKind.RESULT,
    )


def _report(
    *,
    subject: PredictiveResult | SimulationResult,
    report_kind: str,
    applicability: Applicability,
    conclusion: Conclusion,
    findings: tuple[Finding, ...],
    summary: str,
) -> EvaluationReport:
    """One report, with the subject as its parent and the caveat in its summary."""
    subject_ref = _result_ref(subject)
    return EvaluationReport(
        meta=new_artifact_meta(
            artifact_type=ArtifactKind.EVALUATION_REPORT,
            fingerprints=subject.meta.fingerprints,
            producer=PRODUCER,
            parent_refs=(subject_ref,),
            summary=summary,
        ),
        subject_ref=subject_ref,
        report_kind=report_kind,
        applicability=applicability,
        conclusion=conclusion,
        findings=findings,
    )


def _flat(arrays: Sequence[NamedArray]) -> dict[str, np.ndarray]:
    """``{name: (draws, units)}`` -- every axis after the draw axis flattened.

    A discrepancy is defined on observation UNITS, and a plated node arrives
    with the plate as a second axis; flattening here means the contract is one
    shape rather than one shape per node topology.
    """
    return {
        array.name: np.asarray(array.value).reshape(int(array.value.shape[0]), -1)
        for array in arrays
    }


def _conditioned_units(graph: Graph, name: str, units: int) -> np.ndarray:
    """The flat positions of ``name`` that were conditioned on (§0.8).

    A masked position was NOT taken, so it contributes no term to the joint
    and no term to a posterior predictive comparison either: the check asks
    whether the data the model was FITTED to look like data it generates.  The
    held-out positions are ``held_out_prediction``'s subject, not this one's.
    """
    mask = graph.node(name).observed_mask
    if mask is None:
        return np.ones(units, dtype=bool)
    return np.broadcast_to(np.asarray(mask).reshape(-1), (units,))


def _observation_loc(
    graph: Graph, latents: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """``{obs: (draws, units)}`` -- the observation mean at every latent draw.

    Through :func:`~bayesmith.exact.gaussian.observation_parts`, which is the
    same diagonal walk the predictive seam generated the replicated draws
    with, so ``loc`` is the mean of the very distribution ``y_rep`` came from
    rather than a second opinion about it.
    """
    _data, loc, _scale = jax.vmap(
        lambda values: observation_parts(graph, evaluate(graph, values))
    )(latents)
    return {
        name: np.asarray(value).reshape(int(np.shape(value)[0]), -1)
        for name, value in loc.items()
    }


def _p_value(
    weights: np.ndarray, t_replicated: np.ndarray, t_observed: np.ndarray
) -> float:
    """``sum_i w_i * 1[T(rep_i) >= T(obs_i)] / sum_i w_i`` -- §0.3's weighted p.

    The draw axis is one-to-one: draw ``i`` of the replicated data is compared
    against the observed statistic evaluated at draw ``i``'s own ``loc``, and
    weighted by draw ``i``'s own posterior weight.  Nothing is resampled and
    no weight is dropped (R2 §0.5).

    **Divided by the weight total even though the weights are normalised.**
    Measured while writing this file: 4000 weights of ``1/4000``, every
    indicator true, summed to ``1.0000000000000004`` -- so the "p-value" left
    the unit interval and ``min(p, 1 - p)`` came out at ``-4.4e-16``, a
    negative tail mass that the band would then refuse for the right verdict
    and an impossible reason.  Dividing by the same sum in the same order
    makes the saturated cases exactly 1.0 and exactly 0.0, which is what a
    proportion of a finite sample is.
    """
    total = float(np.sum(weights))
    return float(np.sum(weights * (t_replicated >= t_observed))) / total


def _weights(posterior: PosteriorResult) -> np.ndarray:
    """The source posterior's normalised draw weights, uniform where iid.

    ``softmax`` through :func:`~bayesmith.exact.correct.self_normalise`, which
    is where this package's weight normalisation lives -- importance weights
    carry an arbitrary additive constant and a naive ``exp`` overflows on it.
    """
    representation = posterior.representation
    count = int(representation.draws[0].value.shape[0])
    if not isinstance(representation, WeightedDrawsPosterior):
        return np.full(count, 1.0 / count)
    weights, _ess = self_normalise(np.asarray(representation.log_weights.value))
    return np.asarray(weights)


def _cell_finding(
    *,
    node: str,
    identity: str,
    p_value: float,
    tail_mass: float,
    within: bool,
) -> Finding:
    """One (observed node, discrepancy) cell, with everything to recompute it.

    ``observed`` is ``(node, identity, p, tail_mass)`` and ``expected`` is the
    declared tail rate, so a consumer re-derives the verdict as
    ``tail_mass >= expected`` -- the SAME expression
    :func:`tail_mass_within_rate` evaluates, rather than a restatement of it
    that agrees to within a rounding.  §8 R3's gate 8 asks for exactly this:
    a verdict recomputable from the report, with no sampler log in the loop.
    """
    edge = ALPHA / 2.0
    return Finding(
        code="discrepancy_within_band" if within else "discrepancy_outside_band",
        message=(
            f"{identity} on {node!r}: p={p_value:.6f}, tail mass "
            f"{tail_mass:.6f} against the declared {edge:.6f} "
            f"(band [{edge:.6f}, {1.0 - edge:.6f}])"
        ),
        observed=(node, identity, p_value, tail_mass),
        expected=edge,
    )


def _predictive_check(
    *,
    graph: Graph,
    subject: PredictiveResult | SimulationResult,
    report_kind: str,
    latents: dict[str, Any],
    replicated: tuple[NamedArray, ...],
    weights: np.ndarray,
    discrepancies: Sequence[Callable[..., Any]],
    subject_label: str,
) -> EvaluationReport:
    """The body both checks share; they differ only in where the draws came from.

    Written once because a prior predictive check and a posterior predictive
    check ARE one procedure over two parameter sources -- §0.7's whole reason
    for making prior sampling a third primitive of the predictive seam rather
    than a second simulator.  Two copies here would be the same defect one
    layer up.
    """
    identities = tuple(discrepancy_identity(item) for item in discrepancies)
    if not identities:
        raise TypeError(
            "a predictive check needs at least one discrepancy; with none "
            "there is no statistic to compare and the report would be a PASS "
            "nobody measured"
        )

    if not replicated:
        return _report(
            subject=subject,
            report_kind=report_kind,
            applicability=Applicability.INAPPLICABLE,
            conclusion=Conclusion.ABSTAIN,
            findings=(
                Finding(
                    code="no_replicated_draws",
                    message=f"the {subject_label} holds no replicated "
                    "observations, so there is nothing to compare the data "
                    "against",
                    observed=0,
                    expected="at_least_one_observed_site",
                ),
            ),
            summary=f"{report_kind}: inapplicable, no replicated observations",
        )

    draws = int(weights.shape[0])
    if not draws_resolve_the_band(draws):
        return _report(
            subject=subject,
            report_kind=report_kind,
            applicability=Applicability.APPLICABLE,
            conclusion=Conclusion.ABSTAIN,
            findings=(
                Finding(
                    code="draws_below_resolution",
                    message=f"{draws} draws resolve a p-value only to "
                    f"{1.0 / draws:.6f}, which is coarser than the "
                    f"{ALPHA / 2.0:.6f} tail the declared rate reserves; the "
                    "check abstains rather than reporting a pass it could not "
                    "have failed",
                    observed=draws,
                    expected=DRAW_FLOOR,
                ),
            ),
            summary=f"{report_kind}: abstained, {draws} draws below the floor",
        )

    absent = tuple(name for name in graph.latents if name not in latents)
    if absent:
        return _report(
            subject=subject,
            report_kind=report_kind,
            applicability=Applicability.UNVERIFIABLE,
            conclusion=Conclusion.ABSTAIN,
            findings=(
                Finding(
                    code="discrepancy_needs_latent_draws",
                    message=f"the observation mean cannot be recomputed: the "
                    f"{subject_label} carries no draws of {list(absent)}, and "
                    "a discrepancy's second argument is that mean. Whether a "
                    "particular callable reads it is not knowable without "
                    "running it, so the check reports unverifiable rather "
                    "than guessing which of them would have been fine",
                    observed=tuple(absent),
                    expected=graph.latents,
                ),
            ),
            summary=f"{report_kind}: unverifiable, latent draws missing",
        )

    try:
        loc = _observation_loc(graph, latents)
    except NotGaussian as exc:
        return _report(
            subject=subject,
            report_kind=report_kind,
            applicability=Applicability.UNVERIFIABLE,
            conclusion=Conclusion.ABSTAIN,
            findings=(
                Finding(
                    code="predictive_noise_unsupported",
                    message=str(exc),
                    observed=tuple(graph.observed),
                    expected="diagonal_normal",
                ),
            ),
            summary=f"{report_kind}: unverifiable, non-diagonal observation",
        )

    replicated_flat = _flat(replicated)
    findings: list[Finding] = []
    for node, y_replicated in replicated_flat.items():
        units = int(y_replicated.shape[1])
        taken = _conditioned_units(graph, node, units)
        observed = np.broadcast_to(
            np.asarray(graph.node(node).observed).reshape(-1), (units,)
        )
        y_observed = np.broadcast_to(observed, y_replicated.shape)[:, taken]
        node_loc = loc[node][:, taken]
        for identity, discrepancy in zip(identities, discrepancies, strict=True):
            t_replicated = np.asarray(discrepancy(y_replicated[:, taken], node_loc))
            t_observed = np.asarray(discrepancy(y_observed, node_loc))
            if not np.all(np.isfinite(t_replicated)) or not np.all(
                np.isfinite(t_observed)
            ):
                return _report(
                    subject=subject,
                    report_kind=report_kind,
                    applicability=Applicability.UNVERIFIABLE,
                    conclusion=Conclusion.ABSTAIN,
                    findings=(
                        Finding(
                            code="discrepancy_not_finite",
                            message=f"{identity} on {node!r} returned a "
                            "non-finite value; a p-value counted over one is a "
                            "number with no meaning rather than a small one",
                            observed=(node, identity),
                            expected="finite",
                        ),
                    ),
                    summary=f"{report_kind}: unverifiable, {identity} not finite",
                )
            p_value = _p_value(weights, t_replicated, t_observed)
            tail_mass = min(p_value, 1.0 - p_value)
            findings.append(
                _cell_finding(
                    node=node,
                    identity=identity,
                    p_value=p_value,
                    tail_mass=tail_mass,
                    within=tail_mass_within_rate(tail_mass),
                )
            )

    outside = sum(
        1 for item in findings if item.code == "discrepancy_outside_band"
    )
    passed = outside == 0
    return _report(
        subject=subject,
        report_kind=report_kind,
        applicability=Applicability.APPLICABLE,
        conclusion=Conclusion.PASS if passed else Conclusion.FAIL,
        findings=tuple(findings),
        summary=(
            f"{report_kind}: {len(findings)} discrepancy cells at {draws} "
            f"draws, {outside} outside the declared band -- a pass bounds "
            "these statistics and nothing wider"
        ),
    )


def posterior_predictive_check(
    graph: Graph,
    predictive: PredictiveResult,
    *,
    source_posterior: PosteriorResult,
    discrepancies: Sequence[Callable[..., Any]] = DEFAULT_DISCREPANCIES,
) -> EvaluationReport:
    """Do the observed data look like the model's replicated datasets? (§0.2)

    Args:
        graph: the model the predictive result was produced from.  It carries
            the observed data and the loc/scale walk; the result carries the
            draws.
        predictive: the subject.  Its ``replicated_draws`` are the replicated
            datasets and its ``meta`` is what the report's ``subject_ref``
            points at.
        source_posterior: the posterior ``predictive.source_posterior_ref``
            names.  Required, and checked by id and revision, for the same
            reason :func:`bayesmith.dispatch.task.execute_task` requires it:
            there is no artifact store to load one from, and §0.3 computes the
            p-value with the SOURCE posterior's weights -- a weighted
            posterior scored as though it were iid is a different number that
            looks exactly like this one.
        discrepancies: callables on §0.3's ``(y, loc) -> per-draw scalar``
            contract, defaulting to :data:`DEFAULT_DISCREPANCIES`.  Each is
            recorded by :func:`discrepancy_identity` and never as an object.

    Returns:
        An :class:`~bayesmith.artifacts.reports.EvaluationReport` of kind
        ``posterior_predictive_check``: APPLICABLE and PASS when every cell's
        p-value is inside the declared band, APPLICABLE and FAIL when one is
        outside, APPLICABLE and ABSTAIN below :data:`DRAW_FLOOR` draws, and
        UNVERIFIABLE where the discrepancy has no definition on this subject.

    Raises:
        TypeError: for the wrong kind of argument, for a source posterior that
            is not the one the result names, or for an empty discrepancy set.
        ValueError: for a discrepancy with no importable identity.
    """
    if not isinstance(graph, Graph):
        raise TypeError(f"posterior_predictive_check's graph is a Graph; got {graph!r}")
    if not isinstance(predictive, PredictiveResult):
        raise TypeError(
            "posterior_predictive_check judges a PredictiveResult; got "
            f"{type(predictive).__name__}"
        )
    if not isinstance(source_posterior, PosteriorResult):
        raise TypeError(
            "source_posterior is the PosteriorResult the predictive result "
            f"names; got {type(source_posterior).__name__}"
        )
    reference = predictive.source_posterior_ref
    if (
        source_posterior.meta.artifact_id != reference.artifact_id
        or source_posterior.meta.revision != reference.revision
    ):
        raise TypeError(
            "the supplied source posterior is not the version this predictive "
            "result's source_posterior_ref names; its weights would be some "
            "other run's"
        )
    weights = _weights(source_posterior)
    latents = {
        array.name: np.asarray(array.value) for array in source_posterior.representation.draws
    }
    for array in predictive.replicated_draws:
        if int(array.value.shape[0]) != int(weights.shape[0]):
            raise TypeError(
                f"the predictive result's {array.name!r} has "
                f"{array.value.shape[0]} draws and the source posterior has "
                f"{weights.shape[0]}; the p-value pairs draw i of one with "
                "draw i of the other, so a mismatch is not a rescaling"
            )
    return _predictive_check(
        graph=graph,
        subject=predictive,
        report_kind="posterior_predictive_check",
        latents=latents,
        replicated=predictive.replicated_draws,
        weights=weights,
        discrepancies=discrepancies,
        subject_label="predictive result",
    )


def prior_predictive_check(
    graph: Graph,
    simulation: SimulationResult,
    *,
    discrepancies: Sequence[Callable[..., Any]] = DEFAULT_DISCREPANCIES,
) -> EvaluationReport:
    """Could the PRIORS have generated data like this? (§0.2)

    The same procedure as :func:`posterior_predictive_check` over a different
    parameter source, and a different question: this one is answerable before
    any fit, and what it catches is a prior so wide (or so narrow) that the
    data it implies are nothing like the data in hand.  Measured on this
    checkout: ``straight_line``'s own ``w ~ N(0, 2)`` puts all five default
    cells inside the band, and the same model under ``w ~ N(0, 1e6)`` saturates
    ``sd`` at 1.0000 and ``residual_sd`` at 0.0000 -- while ``mean``,
    ``smallest`` and ``largest`` stay comfortably inside it, which is the
    limited-power sentence at the top of this module, measured again.

    Args:
        graph: the model, carrying the observed data the prior draws are
            compared against.
        simulation: the subject.  Its ``parameter_source`` must be the PRIOR:
            a simulation from a posterior or from fixed values answers a
            different question, and is reported INAPPLICABLE rather than
            silently scored as though it were this one.
        discrepancies: as in :func:`posterior_predictive_check`.

    Returns:
        An :class:`~bayesmith.artifacts.reports.EvaluationReport` of kind
        ``prior_predictive_check``.  Prior draws are iid, so every draw
        carries weight ``1/N``: there is no source posterior to read weights
        from and none is invented.

    Raises:
        TypeError: for the wrong kind of argument, or an empty discrepancy set.
        ValueError: for a discrepancy with no importable identity.
    """
    if not isinstance(graph, Graph):
        raise TypeError(f"prior_predictive_check's graph is a Graph; got {graph!r}")
    if not isinstance(simulation, SimulationResult):
        raise TypeError(
            "prior_predictive_check judges a SimulationResult; got "
            f"{type(simulation).__name__}"
        )
    source = simulation.parameter_source.kind
    if source is not ParameterSourceKind.PRIOR:
        return _report(
            subject=simulation,
            report_kind="prior_predictive_check",
            applicability=Applicability.INAPPLICABLE,
            conclusion=Conclusion.ABSTAIN,
            findings=(
                Finding(
                    code="parameter_source_not_prior",
                    message=f"this simulation drew from the {source.value} "
                    "source; a prior predictive check is a statement about "
                    "the priors, and scoring draws from anywhere else would "
                    "answer a question nobody asked under this report's name",
                    observed=source.value,
                    expected=ParameterSourceKind.PRIOR.value,
                ),
            ),
            summary="prior_predictive_check: inapplicable, source is not the prior",
        )
    latents = {
        array.name: np.asarray(array.value) for array in simulation.latent_draws
    }
    count = (
        int(simulation.observation_draws[0].value.shape[0])
        if simulation.observation_draws
        else 0
    )
    weights = np.full(count, 1.0 / count) if count else np.zeros(0)
    return _predictive_check(
        graph=graph,
        subject=simulation,
        report_kind="prior_predictive_check",
        latents=latents,
        replicated=simulation.observation_draws,
        weights=weights,
        discrepancies=discrepancies,
        subject_label="simulation result",
    )
