"""Held-out prediction: score the observations the graph declared unseen.

R3 §0.8's ruling in one function.  A held-out point is not a new task
parameter and not a second list of indices kept beside the data -- it is a
position where an observed node's ``observed_mask`` is False.  The mask is
already the graph's own statement of what was conditioned on
(``log_joint`` and
:func:`~bayesmith.dispatch.predictive.pointwise_log_likelihood` both read it,
and probe_28 §7 measured the masked positions' pointwise log-likelihood as
exactly 0), so reading it here means there is ONE answer to "which points did
the posterior see?" rather than two that agree until someone edits one.

**Conditioning and prediction cover different sets, on purpose.**  Only the
True positions condition.  Prediction reaches every position, because the
loc/scale :func:`~bayesmith.exact.gaussian.observation_parts` reads exist at a
masked position exactly as they do at a conditioned one -- nothing about the
forward model knows a point was withheld.  That asymmetry is the whole content
of a held-out check: the predictive at a point the posterior never saw is an
honest out-of-sample prediction, and comparing it to the value sitting there
is the only comparison in this package that is not, in part, a comparison with
itself.

**Two numbers per held-out point, and they answer different questions.**

* The PIT, ``F_pred(y_j) = Σ_i w_i Φ((y_j - loc_ij) / scale_ij)`` -- where the
  observed value falls in its own predictive distribution.  Under a calibrated
  model it is Uniform(0, 1), which is what makes a TAIL a testable event.
* ``elpd_heldout = Σ_j log Σ_i w_i p(y_j | θ_i)`` -- the log predictive density
  actually achieved.  It is REPORTED, never thresholded: there is no scale on
  which "-1.02 is good and -24.05 is bad" without a second model to compare
  against, and model comparison is R7's (§"R3 明确不做的事").  The verdict
  comes from the PIT alone; the elpd is what a later comparison will read.

**The band is derived, not chosen.**  §0.4 declares ONE false-positive rate for
the whole layer, :data:`bayesmith.evaluation.ALPHA` (D104), and everything here
is arithmetic on it: a two-sided test of one point's PIT rejects outside
``[α/2, 1 - α/2]``; m points tested together with a family-wise error of at
most α is Bonferroni's ``α/m`` per point, hence ``α/(2m)`` per TAIL.  No number
is introduced here -- ``_tail`` is the derivation written down once so that a
reader can check it and a mutation can break it.

**Weights come from the source posterior, because that is where they live.**  A
:class:`~bayesmith.artifacts.results.PredictiveResult` carries draws and no
weights; an importance-weighted posterior (the ``snis`` route) whose draws were
averaged uniformly would produce a confident, wrong PIT and nothing in the
report would look unusual.  So the source posterior is a required argument, and
it is checked against the reference the predictive result already carries
rather than trusted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import numpy as np
import numpyro.distributions as dist

from bayesmith.artifacts.base import ArtifactRef, new_artifact_meta
from bayesmith.artifacts.identity import ArtifactKind
from bayesmith.artifacts.refusal import Finding
from bayesmith.artifacts.reports import Applicability, Conclusion, EvaluationReport
from bayesmith.artifacts.results import (
    DrawsPosterior,
    PosteriorResult,
    PredictiveResult,
    WeightedDrawsPosterior,
)
from bayesmith.dispatch.task import PRODUCER
from bayesmith.errors import NotGaussian
from bayesmith.evaluation import ALPHA
from bayesmith.exact.gaussian import observation_parts
from bayesmith.graph.evaluate import evaluate
from bayesmith.graph.graph import Graph

__all__ = ["REPORT_KIND", "held_out_report"]

#: §0.2's row for this check.  A ``report_kind`` is a code, not a schema
#: (R1 ruling), which is why a new kind of report costs no migration.
REPORT_KIND = "held_out_prediction"


def _tail(points: int) -> float:
    """The per-tail rejection probability for ``points`` held-out points.

    DERIVED from :data:`~bayesmith.evaluation.ALPHA` and nothing else.  Half of
    α because the test is two-sided -- a PIT of 0.999 is as much evidence
    against the predictive as one of 0.001 -- and divided by the number of
    points because m simultaneous tests each at α would reject a calibrated
    model about m times as often as declared, which would make the ONE number
    §0.4 declares a function of how many points the fixture happened to
    withhold.
    """
    if points < 1:
        raise ValueError(f"a Bonferroni factor needs at least one point; got {points}")
    return ALPHA / (2.0 * points)


def _held_out(graph: Graph) -> dict[str, np.ndarray]:
    """``{observed node: flat indices whose mask is False}``, empty entries dropped.

    Flat indices rather than per-axis ones so that a plated or matrix-shaped
    observed node is scored by the same code path as a vector; the report names
    the node and the flat position, which is what
    :func:`~bayesmith.dispatch.predictive.pointwise_log_likelihood` also
    indexes by.
    """
    positions: dict[str, np.ndarray] = {}
    for name in graph.observed:
        mask = graph.node(name).observed_mask
        if mask is None:
            continue
        flat = np.asarray(mask).reshape(-1)
        held = np.flatnonzero(~flat)
        if held.size:
            positions[name] = held
    return positions


def _weights(representation: object) -> np.ndarray | None:
    """Normalised draw weights, or ``None`` for a posterior that holds no draws.

    ``None`` is a real answer rather than a fallback to uniform: an analytic or
    a fitted-conditional posterior has no draw axis to weight, and inventing
    one would let the check report a number computed from draws that do not
    exist.
    """
    if isinstance(representation, WeightedDrawsPosterior):
        raw = np.asarray(representation.log_weights.value, dtype=np.float64)
        shifted = np.exp(raw - raw.max())
        return shifted / shifted.sum()
    if isinstance(representation, DrawsPosterior):
        count = int(representation.draws[0].value.shape[0])
        return np.full(count, 1.0 / count)
    return None


def _parts(
    graph: Graph, latent_draws: Mapping[str, Any], names: tuple[str, ...]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{node: (log_prob, cdf)}`` at the observed data, one row per draw.

    The loc/scale are RECOMPUTED on the carried latent draws through the same
    :func:`~bayesmith.exact.gaussian.observation_parts` the R2 predictive seam
    uses, and no new draw is taken.  Re-sampling would answer a different
    question -- "how far is this point from a fresh replicate?" carries the
    observation noise twice -- and would also make the report depend on a key
    the subject result does not carry.
    """

    def per_draw(values: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
        data, loc, scale = observation_parts(graph, evaluate(graph, values))
        return {
            name: (
                dist.Normal(loc[name], scale[name]).log_prob(data[name]),
                dist.Normal(loc[name], scale[name]).cdf(data[name]),
            )
            for name in names
        }

    stacked = jax.vmap(per_draw)(dict(latent_draws))
    return {
        name: (
            np.asarray(stacked[name][0], dtype=np.float64).reshape(
                int(np.shape(stacked[name][0])[0]), -1
            ),
            np.asarray(stacked[name][1], dtype=np.float64).reshape(
                int(np.shape(stacked[name][1])[0]), -1
            ),
        )
        for name in names
    }


def _log_mean_exp(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``log Σ_i w_i exp(values[i])`` down the draw axis, shift-stabilised."""
    ceiling = values.max(axis=0)
    return ceiling + np.log((weights[:, None] * np.exp(values - ceiling)).sum(axis=0))


def _report(
    predictive: PredictiveResult,
    *,
    applicability: Applicability,
    conclusion: Conclusion,
    findings: tuple[Finding, ...],
    summary: str,
) -> EvaluationReport:
    """The envelope, filled the one way this module fills it.

    ``subject_ref`` and the single parent are the same reference: §0.1 says a
    report points at what it read, and lineage that omitted the subject could
    not be retired when the subject was.
    """
    subject = ArtifactRef(
        artifact_id=predictive.meta.artifact_id,
        revision=predictive.meta.revision,
        artifact_type=ArtifactKind.RESULT,
    )
    return EvaluationReport(
        meta=new_artifact_meta(
            artifact_type=ArtifactKind.EVALUATION_REPORT,
            fingerprints=predictive.meta.fingerprints,
            producer=PRODUCER,
            parent_refs=(subject,),
            summary=summary,
        ),
        subject_ref=subject,
        report_kind=REPORT_KIND,
        applicability=applicability,
        conclusion=conclusion,
        findings=findings,
    )


def held_out_report(
    graph: Graph,
    predictive: PredictiveResult,
    *,
    source_posterior: PosteriorResult,
) -> EvaluationReport:
    """Score the positions ``graph`` masked out, against ``predictive``'s draws.

    Args:
        graph: the model the predictive result was produced against.  Its
            observed nodes' ``observed_mask`` is the definition of held out.
        predictive: the subject.  Its carried ``latent_draws`` are replayed
            through the forward model; nothing is re-sampled.
        source_posterior: the posterior ``predictive.source_posterior_ref``
            names.  Required for its WEIGHTS, which a predictive result does
            not carry -- see the module docstring.

    Returns:
        An :class:`~bayesmith.artifacts.reports.EvaluationReport` of kind
        ``held_out_prediction``, on §0.2's row:

        * APPLICABLE / PASS -- every held-out point's PIT is inside
          ``[α/2m, 1 - α/2m]``.
        * APPLICABLE / FAIL -- at least one is outside it.
        * INAPPLICABLE / ABSTAIN -- the graph masks nothing out, so there is
          no out-of-sample point to score.  Not a PASS: a check with no
          subject has verified nothing, and a gate counting it as a check that
          was made would be counting an empty set.
        * UNVERIFIABLE / ABSTAIN -- the predictive cannot be evaluated: a
          correlated or non-Gaussian observed node (§0.4's coverage domain),
          a source posterior that holds no draws, or a subject whose carried
          latents do not cover the graph.

    Raises:
        TypeError: if the arguments are of the wrong type, or if
            ``source_posterior`` is not the artifact and revision
            ``predictive.source_posterior_ref`` names.  That is a caller
            error, not a verdict about a model: a report saying "unverifiable"
            would file the mistake as a property of the subject.
    """
    if not isinstance(graph, Graph):
        raise TypeError(f"held_out_report's graph is a Graph; got {graph!r}")
    if not isinstance(predictive, PredictiveResult):
        raise TypeError(
            f"held_out_report judges a PredictiveResult; got {predictive!r}"
        )
    if not isinstance(source_posterior, PosteriorResult):
        raise TypeError(
            "held_out_report's source_posterior is a PosteriorResult; got "
            f"{source_posterior!r}"
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

    positions = _held_out(graph)
    if not positions:
        return _report(
            predictive,
            applicability=Applicability.INAPPLICABLE,
            conclusion=Conclusion.ABSTAIN,
            findings=(
                Finding(
                    code="no_held_out_points",
                    message=(
                        "every observed position is conditioned on: the graph's "
                        "masks are all True (or absent), so there is no "
                        "out-of-sample point for this check to score"
                    ),
                    observed=0,
                    expected=1,
                ),
            ),
            summary="no held-out points are declared by this graph",
        )

    carried = {array.name: array.value for array in predictive.latent_draws}
    missing = tuple(name for name in graph.latents if name not in carried)
    if missing:
        return _report(
            predictive,
            applicability=Applicability.UNVERIFIABLE,
            conclusion=Conclusion.ABSTAIN,
            findings=(
                Finding(
                    code="latent_draws_incomplete",
                    message=(
                        "the predictive result carries no draws for "
                        f"{list(missing)}, so its forward model cannot be "
                        "replayed; the predictive task's latent_sites decides "
                        "what is carried"
                    ),
                    observed=tuple(sorted(carried)),
                    expected=tuple(graph.latents),
                ),
            ),
            summary="the subject does not carry the latents the graph needs",
        )

    weights = _weights(source_posterior.representation)
    if weights is None:
        return _report(
            predictive,
            applicability=Applicability.UNVERIFIABLE,
            conclusion=Conclusion.ABSTAIN,
            findings=(
                Finding(
                    code="source_posterior_holds_no_draws",
                    message=(
                        "a held-out score averages over posterior draws, and "
                        "this source posterior is a "
                        f"{type(source_posterior.representation).__name__} -- "
                        "there is no draw axis to weight"
                    ),
                    observed=type(source_posterior.representation).__name__,
                    expected=("DrawsPosterior", "WeightedDrawsPosterior"),
                ),
            ),
            summary="the source posterior holds no draws to average over",
        )

    names = tuple(positions)
    try:
        parts = _parts(graph, carried, names)
    except NotGaussian as exc:
        return _report(
            predictive,
            applicability=Applicability.UNVERIFIABLE,
            conclusion=Conclusion.ABSTAIN,
            findings=(
                Finding(
                    code="predictive_noise_unsupported",
                    message=str(exc),
                    observed=(exc.reason, exc.found or "", exc.node or ""),
                    expected="diagonal_normal",
                ),
            ),
            summary="held-out scoring needs a diagonal-Gaussian observation",
        )

    draws = int(next(iter(parts.values()))[0].shape[0])
    if draws != weights.shape[0]:
        raise TypeError(
            f"the subject holds {draws} draws and its source posterior "
            f"{weights.shape[0]}; a weight per draw is what makes the average "
            "a posterior average"
        )

    count = int(sum(int(held.size) for held in positions.values()))
    tail = _tail(count)
    findings: list[Finding] = []
    elpd = 0.0
    failed = 0
    for name in names:
        log_prob, cdf = parts[name]
        held = positions[name]
        lpd = _log_mean_exp(log_prob[:, held], weights)
        pit = (weights[:, None] * cdf[:, held]).sum(axis=0)
        elpd += float(lpd.sum())
        for position, point, density in zip(held, pit, lpd, strict=True):
            inside = tail <= float(point) <= 1.0 - tail
            failed += 0 if inside else 1
            findings.append(
                Finding(
                    code="held_out_point",
                    message=(
                        f"{name}[{int(position)}] was withheld from "
                        f"conditioning; its PIT is "
                        f"{'inside' if inside else 'outside'} the band"
                    ),
                    observed=(name, int(position), float(point), float(density)),
                    expected=(tail, 1.0 - tail),
                )
            )

    findings.append(
        Finding(
            code="held_out_elpd",
            message=(
                "log predictive density summed over the held-out points, "
                "averaged over the source posterior's weighted draws; reported "
                "rather than thresholded -- a single model's elpd has no scale"
            ),
            observed=(elpd, count, draws),
            expected=None,
        )
    )

    return _report(
        predictive,
        applicability=Applicability.APPLICABLE,
        conclusion=Conclusion.FAIL if failed else Conclusion.PASS,
        findings=tuple(findings),
        summary=(
            f"{failed} of {count} held-out points fall outside the "
            f"[{tail:.6g}, {1.0 - tail:.6g}] PIT band"
        ),
    )
