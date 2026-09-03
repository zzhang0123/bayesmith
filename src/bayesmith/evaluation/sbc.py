"""Simulation-based calibration: does the route return the posterior it claims?

The one check in this layer that does not read a finished result and stop.
SBC generates its own subject: draw ``theta_true`` from the prior, draw ``y``
from the model at that ``theta_true``, infer a posterior from ``y`` alone, and
ask where ``theta_true`` fell inside it.  Over many replicates those positions
must be uniform, because a correct posterior places the truth uniformly by
construction.  A route that is systematically too wide puts the truth in the
middle too often; one that is too narrow pushes it into the tails.

**What a PASS here does NOT say: that the route read the data.**  A "posterior"
that ignores ``y`` and hands back prior draws is uniform in rank BY
CONSTRUCTION -- the position of a prior draw among prior draws is uniform -- so
it scores APPLICABLE x PASS from this harness.  Measured on this checkout with
a sampler that discards its ``datum`` and returns ``2.0 * normal(key, (n,))``,
N = 100 replicates on the straight-line fixture: KS p = 0.9532, 0.7265, 0.6004
and 0.1842 at seeds 0, 1, 3 and 4 -- PASS at every one -- with seed 2's 0.0474
the false positive ``ALPHA = 0.05`` declares in advance.  SBC asks whether a
route's stated uncertainty is consistent with its stated prior; a route that is
trivially self-consistent answers yes.  Something that reads the observation --
a posterior predictive check, a held-out score -- has to be reported alongside
it before anyone says a route works.  This is the discipline the R3 plan's
§0.3 already imposes on PPC ("do not tell a PASS as 'the model is correct'"),
and it is owed here for the same reason.  The cell that pins it is
``test_a_posterior_that_ignores_the_data_is_still_calibrated``.

A second limit of the same shape, and a property of the FIXTURE rather than of
this harness: where the likelihood dominates the prior, a posterior computed
under the WRONG prior still passes.  Measured on the same fixture at N = 100, a
conjugate sampler told the prior is ``N(0, 0.5)`` while the model declares
``N(0, 2)`` gives p = 0.1166 / 0.4411 / 0.4049 at seeds 0 / 1 / 2 -- PASS at
all three -- and is only caught once the sampler's prior is wrong by a further
factor of five (``N(0, 0.1)``: p = 2.5e-15 / 2.6e-16 / 6.8e-24).  What SBC has
power against is set by the fixture as much as by the replicate count, and
D106 is a floor measured against a 2x WIDTH error, not a promise about every
error.

**Three rulings from the R3 plan's §0.6 are implemented here literally, and
none of them is re-decided in this module.**

*The rank is continuous and weighted.*  ``r = sum_i w_i * 1[theta_i <
theta_true]`` in ``[0, 1]``, with ``w_i = 1/n`` for an equally weighted sample
and the posterior's own normalised weights for an importance-weighted one.
The classical integer rank would need a resample to accept weights, and R2
§0.5 already ruled that a weighted result is not resampled on the way into a
judgement -- a resample would make the check's answer depend on a second RNG
that nothing declared.

*Uniformity is tested with :func:`scipy.stats.kstest`,* one test per scalar
latent coordinate, against a Bonferroni-corrected level ``alpha / K`` for the
``K`` coordinates.  ``alpha`` is D104 and lives in this package's ``__init__``;
the Bonferroni factor is a derivation from it, not a second threshold.

*A replicate that did not finish makes the whole report ABSTAIN, with a
count.*  Never a silently dropped replicate.  SBC checks the algorithm AND its
implementation, so the replicates that refused, diverged or came back without
draws are part of what is being measured: an SBC that discards its failures
and reports uniformity over the survivors is checking a different, easier
question and calling the answer by this one's name.  The three failure buckets
are counted separately (``refused``, ``unconverged``, ``undrawn``) and all
three appear in the report's findings.

**The replicate floor is D106 = 100**, and the arithmetic that fixes it is
:func:`replicates_meet_floor`'s only job.  See its docstring for the
measurement -- including why the R3 plan's provisional 50 did not survive
being measured over ten seeds instead of one.

**Two arms, one rank definition (§0.11).**  The route arm compiles and
executes a real :class:`~bayesmith.artifacts.tasks.PosteriorTask` per
replicate, so exact and sampled routes are covered by the same harness.  The
sampler arm takes any ``sampler(datum, key, n) -> draws`` and is what an
amortized estimator is scored through.  Both reach
:func:`sbc_report` with the same :class:`SbcRanks`, which is the point: a
comparison between a NUTS posterior and a neural one is only a comparison if
the two were judged by identical arithmetic.

Layering: this module reads ``dispatch``, ``graph`` and ``artifacts`` and is
read by none of them (§0.1).  It creates no Result and modifies none: the
posteriors it ranks are produced by ``dispatch.task`` exactly as any other
caller would produce them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import numpy as np
from scipy import stats

from ..artifacts.base import (
    ArtifactRef,
    ComputeBudget,
    FingerprintBundle,
    ProducerRef,
    TerminationReason,
    new_artifact_meta,
)
from ..artifacts.identity import ArtifactKind, ModelRef
from ..artifacts.refusal import Finding, Refusal
from ..artifacts.reports import Applicability, Conclusion, EvaluationReport
from ..artifacts.results import (
    DrawsPosterior,
    PosteriorResult,
    SimulationResult,
    WeightedDrawsPosterior,
)
from ..artifacts.tasks import (
    ParameterSource,
    ParameterSourceKind,
    PosteriorTask,
    SimulationTask,
    new_task_meta,
)
from ..dispatch.task import PRODUCER, compile_task, execute_task
from ..graph.graph import Graph
from . import ALPHA

__all__ = [
    "REPLICATE_FLOOR",
    "REPORT_KIND",
    "SbcRanks",
    "continuous_rank",
    "ranks_are_uniform",
    "replicates_meet_floor",
    "sbc_ranks",
    "sbc_report",
    "simulation_based_calibration",
]

#: Borrowed from ``dispatch.task`` rather than rebuilt, so a report and the
#: result it judges name the same producer at the same version.
_PRODUCER: ProducerRef = PRODUCER

#: The ``report_kind`` code §0.2 gives this check. A code, not prose: R1 froze
#: ``report_kind`` as a code string, so a new kind of report is a new code
#: rather than a schema change.
REPORT_KIND = "sbc"

#: D106. The number of USABLE replicates below which this check abstains.
#:
#: Provenance is **derived**, and the derivation is a power measurement rather
#: than a formula: it is the replicate count at which a 2x width error in the
#: posterior is detected by the rank-uniformity test at every one of ten
#: seeds.  ``docs/probes/probe_30_sbc_replicate_floor.py`` is the measurement,
#: and :func:`replicates_meet_floor` carries the numbers it produced.
REPLICATE_FLOOR = 100

#: A replicate finished if it stopped for one of these two reasons. Every other
#: :class:`~bayesmith.artifacts.base.TerminationReason` -- a budget that ran
#: out, a tolerance that was never met, a diverged chain, an interrupted run --
#: is a replicate whose posterior is not the posterior the route claims to
#: produce, so ranking it would fold a known execution failure into a
#: statistical verdict.
_FINISHED = (TerminationReason.COMPLETED, TerminationReason.CONVERGED)

#: ``sampler(datum, key, n) -> {latent name: draws}``. The draw axis leads, and
#: each array's remaining shape is the latent's own, so a sampler and a route
#: hand :func:`continuous_rank` arrays of the same shape (§0.11).
Sampler = Callable[
    [Mapping[str, np.ndarray], jax.Array, int], Mapping[str, Any]
]

#: ``build(data) -> Graph``: the model callable re-traced on ONE replicate's
#: simulated data. §0.6 requires the same model callable and a REBUILT graph --
#: each replicate conditions on different data, so its Graph is a different
#: object with the same ``model_ref`` and the same structure fingerprint.
GraphBuilder = Callable[[Mapping[str, np.ndarray]], Graph]


def replicates_meet_floor(usable: int, floor: int = REPLICATE_FLOOR) -> bool:
    """D106: ``usable >= floor`` -- enough replicates to detect a 2x width error.

    Closed on purpose, and at 100 rather than at the 50 the R3 plan proposed.

    The plan's 50 came from probe_28 §4, which swept ONE seed per distortion.
    Task 6.1 required the sweep be repeated across ten seeds
    (``jax.random.key(23 + k)``, k in 0..9) and the WORST -- least detectable --
    p-value be the number the floor answers to.  Measured by
    ``docs/probes/probe_30_sbc_replicate_floor.py`` on this checkout, over the
    straight-line fixture, KS p for a 2x-too-wide and a 2x-too-narrow
    posterior:

    ======  ==============  ================  =========================
    N       worst 2x wide   worst 2x narrow   verdict against alpha / 3
    ======  ==============  ================  =========================
    20      0.4813          0.7045            undetectable
    50      0.1384          0.1384            crosses, and crosses alpha too
    100     0.0054          0.0054            3x inside the margin
    ======  ==============  ================  =========================

    At N = 50 the worst seed puts a doubled posterior width at p = 0.1384 --
    not merely past the plan's alpha/3 = 0.0167 tripwire but past alpha = 0.05
    itself, so one seed in ten would have MISSED a 2x error outright.  The
    plan's own escape clause ("if the worst p at N = 50 crosses alpha/3, raise
    the floor to 100 and record that you did") is therefore taken here, and
    this docstring is the record.

    ``>=`` rather than ``>``: the floor is the smallest count that was measured
    to work, so a run that reaches it exactly has what the measurement asked
    for.  An open boundary would refuse the very count the number was chosen
    to name.
    """
    return usable >= floor


def ranks_are_uniform(p_value: float, level: float) -> bool:
    """``p_value >= level`` -- this coordinate's ranks are uniform enough.

    The comparison that turns the KS test into a verdict, given its own name
    so that it can be exercised and mutated directly rather than only through
    a hundred-replicate calibration run.

    ``level`` is NOT a threshold of this module's: it is ``ALPHA / K``, the
    Bonferroni correction §0.4 derives from D104 for the ``K`` scalar latent
    coordinates tested together. D104 is declared once, in this package's
    ``__init__``; the factor K is arithmetic. Two checks in one layer each
    choosing their own false-positive rate is what having one ALPHA prevents,
    and it is why this function takes the level rather than reaching for it.

    ``>=`` rather than ``>``: alpha is the size of the test -- the probability
    of rejecting a CORRECT posterior -- so the rejection region is ``p <
    alpha``, and a p-value landing exactly on the level is inside the region
    the test was declared to accept.
    """
    return p_value >= level


def continuous_rank(
    draws: np.ndarray, weights: np.ndarray, truth: float
) -> float:
    """§0.6's rank: ``sum_i w_i * 1[theta_i < truth]``, in ``[0, 1]``.

    ``draws`` is one scalar coordinate's draws and ``weights`` the posterior's
    normalised weights over the same draw axis.  Strict ``<`` matches the
    classical rank statistic's convention; for a continuous posterior the tie
    set has probability zero, and for a discrete one the choice would have to
    be made by whoever owns that discreteness rather than here.
    """
    below = np.asarray(draws) < truth
    return float(np.sum(np.asarray(weights) * below))


@dataclasses.dataclass(frozen=True, slots=True)
class SbcRanks:
    """What the replicate loop measured, with no verdict attached.

    Separated from :func:`sbc_report` so that the two arms of §0.11 -- a
    compiled route and an arbitrary sampler -- reach one judgement through one
    object, and so that a caller who wants the ranks themselves (a histogram,
    a plot) does not have to reach inside a report to get them.

    ``ranks`` is one tuple per coordinate, each holding one rank per USABLE
    replicate; the failure counts say how many replicates are missing from
    those tuples and why.
    """

    coordinates: tuple[str, ...]
    ranks: tuple[tuple[float, ...], ...]
    requested: int
    usable: int
    refused: int
    unconverged: int
    undrawn: int
    route: str
    subject_ref: ArtifactRef
    fingerprints: FingerprintBundle

    @property
    def unusable(self) -> int:
        """Replicates that produced no rank, whatever the reason."""
        return self.refused + self.unconverged + self.undrawn


def _flat_coordinates(name: str, shape: tuple[int, ...]) -> tuple[str, ...]:
    """Scalar coordinate names for one latent: ``w`` or ``z[0]``, ``z[1]``...

    A vector latent is K separate calibration questions, not one: a route can
    be calibrated in a scale parameter and badly wrong in a location, and a
    single pooled test would average the second away.
    """
    if not shape:
        return (name,)
    size = int(np.prod(shape, dtype=int))
    return tuple(f"{name}[{index}]" for index in range(size))


def _posterior_draws(
    result: PosteriorResult,
) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
    """``({latent: draws}, weights)`` for a posterior that holds draws.

    ``None`` for the two representations that hold none -- an analytic
    posterior and a fitted conditional one.  Those are not failures of the
    run; they are results this rank definition has nothing to compute on, and
    §0.6's accounting counts them rather than dropping them.

    The seam a test mutates to build a miscalibrated route out of a correct
    one: stretching the draws returned here is exactly the 2x-width error the
    floor was measured against.
    """
    representation = result.representation
    if isinstance(representation, WeightedDrawsPosterior):
        log_weights = np.asarray(representation.log_weights.value, dtype=np.float64)
        shifted = np.exp(log_weights - np.max(log_weights))
        weights = shifted / np.sum(shifted)
    elif isinstance(representation, DrawsPosterior):
        count = int(representation.draws[0].value.shape[0])
        weights = np.full(count, 1.0 / count, dtype=np.float64)
    else:
        return None
    draws = {
        array.name: np.asarray(array.value) for array in representation.draws
    }
    return draws, weights


def _result_ref(result: PosteriorResult | SimulationResult) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=result.meta.artifact_id,
        revision=result.meta.revision,
        artifact_type=ArtifactKind.RESULT,
    )


def _prior_simulation(
    graph: Graph,
    *,
    key: jax.Array,
    replicates: int,
    model_ref: ModelRef,
) -> SimulationResult | Refusal:
    """The ``(theta_true, y)`` pairs, from ONE ``SimulationTask(PRIOR)``.

    One task with ``budget.draws = replicates`` rather than one task per
    replicate: ``prior_draws`` is vmapped over the draw axis, so the whole set
    of truths and datasets is a single execution, and the replicates share one
    provenance record instead of N nearly identical ones.

    §0.7 already made this the model's own forward law -- the observed nodes
    are drawn from their own distributions -- so SBC does not carry a second
    simulator, which is what invariant 1 forbids.
    """
    task = SimulationTask(
        meta=new_task_meta(label="sbc prior replicates"),
        parameter_source=ParameterSource(kind=ParameterSourceKind.PRIOR),
        latent_sites=graph.latents,
        observed_sites=graph.observed,
        budget=ComputeBudget(draws=replicates),
    )
    planned = compile_task(graph, task, model_ref=model_ref)
    if isinstance(planned, Refusal):
        return planned
    return execute_task(planned, key=key)


def _route_replicate(
    build: GraphBuilder,
    datum: Mapping[str, np.ndarray],
    key: jax.Array,
    *,
    model_ref: ModelRef,
    budget: ComputeBudget,
    nuts_on_collapse: bool,
) -> PosteriorResult | Refusal:
    """One replicate through the real dispatch: rebuild, compile, execute."""
    task = PosteriorTask(
        meta=new_task_meta(label="sbc replicate"),
        budget=budget,
        nuts_on_collapse=nuts_on_collapse,
    )
    planned = compile_task(build(datum), task, model_ref=model_ref)
    if isinstance(planned, Refusal):
        return planned
    return execute_task(planned, key=key)


def _covers(
    draws: Mapping[str, np.ndarray], truths: Mapping[str, np.ndarray]
) -> bool:
    """Whether these draws can rank every latent the simulation gave a truth.

    They cannot when the route ELIMINATED a latent: such a posterior carries
    the name in ``eliminated_latents`` and draws for the others, which is a
    correct result and an unrankable replicate. Ranking it anyway would leave
    the eliminated coordinate with fewer ranks than its neighbours -- one KS
    test quietly run at a different N from the one beside it -- so the
    replicate is counted as undrawn instead.
    """
    return all(name in draws for name in truths)


def _sampler_draws(
    sampler: Sampler,
    datum: Mapping[str, np.ndarray],
    key: jax.Array,
    count: int,
    truths: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
    """Adapt §0.11's plain sampler contract to the route arm's shape."""
    drawn = sampler(datum, key, count)
    draws = {name: np.asarray(value) for name, value in drawn.items()}
    if not _covers(draws, truths):
        return None
    n = int(draws[next(iter(truths))].shape[0])
    return draws, np.full(n, 1.0 / n, dtype=np.float64)


def sbc_ranks(
    graph: Graph,
    *,
    key: jax.Array,
    replicates: int,
    model_ref: ModelRef,
    build: GraphBuilder | None = None,
    budget: ComputeBudget | None = None,
    nuts_on_collapse: bool = False,
    sampler: Sampler | None = None,
    sampler_draws: int = 0,
    subject_ref: ArtifactRef | None = None,
) -> SbcRanks | Refusal:
    """Run the replicate loop and return its ranks, unjudged.

    Args:
        graph: the model, used for the prior simulation. Its own observed data
            is not read -- ``SimulationTask(PRIOR)`` draws the observations
            from their own distributions.
        key: PRNG key. Split once into a simulation key and a replicate key,
            then ``fold_in`` per replicate, so replicate i is reproducible on
            its own and adding replicates does not disturb the earlier ones.
        replicates: how many ``(theta_true, y)`` pairs to draw. Never
            defaulted: a replicate count is a budget, and D106 is a floor on
            it rather than a value for it.
        model_ref: what the model callable is, for every compiled task.
        build: ``build(data) -> Graph``, the route arm. Required unless
            ``sampler`` is given.
        budget: the per-replicate posterior budget for the route arm.
        nuts_on_collapse: passed to each replicate's ``PosteriorTask``. The
            default is False so that a route that collapses is COUNTED as an
            unusable replicate rather than silently answered by a different
            route -- SBC is a statement about one route.
        sampler: the §0.11 arm. ``sampler(datum, key, n) -> {name: draws}``.
        sampler_draws: ``n`` for the sampler arm.
        subject_ref: required with ``sampler``, which produces no Result to
            point at; ignored by the route arm, which uses the first usable
            replicate's posterior (§0.2's "routing representative").

    Returns:
        :class:`SbcRanks`, or the ``Refusal`` that stopped the prior
        simulation. A REPLICATE's refusal is never returned: §0.6 says a
        refused replicate is COUNTED and the report abstains, so propagating
        one would replace a verdict about a calibration run with a verdict
        about its first bad draw.
    """
    if not isinstance(graph, Graph):
        raise TypeError(f"sbc_ranks' graph is a Graph; got {graph!r}")
    count = int(replicates)
    if count < 1:
        raise ValueError(f"sbc_ranks needs at least one replicate; got {replicates!r}")
    if sampler is None and build is None:
        raise TypeError(
            "sbc_ranks needs either build=... (compile and execute a "
            "PosteriorTask per replicate) or sampler=... (§0.11's "
            "sampler(datum, key, n) arm); it was given neither"
        )
    if sampler is not None and subject_ref is None:
        raise TypeError(
            "the sampler arm produces no PosteriorResult, so the report has "
            "nothing to point at unless the caller says what the sampler "
            "represents: pass subject_ref=..."
        )
    if sampler is not None and sampler_draws < 1:
        raise ValueError(
            "the sampler arm needs the draw count to ask its sampler for: "
            f"sampler_draws={sampler_draws!r}. A rank over zero draws is 0.0 "
            "for every replicate, which is a perfectly non-uniform sample and "
            "would read as a catastrophically miscalibrated route"
        )

    simulation_key, replicate_key = jax.random.split(key)
    simulation = _prior_simulation(
        graph, key=simulation_key, replicates=count, model_ref=model_ref
    )
    if isinstance(simulation, Refusal):
        return simulation

    truths = {array.name: np.asarray(array.value) for array in simulation.latent_draws}
    data = {
        array.name: np.asarray(array.value) for array in simulation.observation_draws
    }

    coordinates: list[str] = []
    for array in simulation.latent_draws:
        coordinates.extend(_flat_coordinates(array.name, array.value.shape[1:]))
    collected: dict[str, list[float]] = {name: [] for name in coordinates}

    refused = 0
    unconverged = 0
    undrawn = 0
    # The route arm learns its representative from the replicates; the sampler
    # arm cannot, which is why that arm requires the caller to say.
    representative = subject_ref if sampler is not None else None
    route = "sampler" if sampler is not None else ""

    for index in range(count):
        datum = {name: value[index] for name, value in data.items()}
        replicate_seed = jax.random.fold_in(replicate_key, index)
        if sampler is not None:
            drawn = _sampler_draws(
                sampler, datum, replicate_seed, sampler_draws, truths
            )
        else:
            result = _route_replicate(
                build,
                datum,
                replicate_seed,
                model_ref=model_ref,
                budget=budget if budget is not None else ComputeBudget(),
                nuts_on_collapse=nuts_on_collapse,
            )
            if isinstance(result, Refusal):
                refused += 1
                continue
            if result.run.termination.reason not in _FINISHED:
                unconverged += 1
                continue
            drawn = _posterior_draws(result)
            if drawn is not None and not _covers(drawn[0], truths):
                drawn = None
            if drawn is not None:
                if representative is None:
                    representative = _result_ref(result)
                if not route:
                    route = result.representation.method
        if drawn is None:
            undrawn += 1
            continue
        _accumulate(collected, drawn, truths, index)

    if representative is None:
        # No replicate produced a posterior, so §0.2's "routing representative"
        # does not exist. The SimulationResult does, and it is the result this
        # calibration run actually produced -- so the report that abstains
        # points at the replicate set rather than at nothing.
        representative = _result_ref(simulation)

    return SbcRanks(
        coordinates=tuple(coordinates),
        ranks=tuple(tuple(collected[name]) for name in coordinates),
        requested=count,
        usable=count - refused - unconverged - undrawn,
        refused=refused,
        unconverged=unconverged,
        undrawn=undrawn,
        route=route or "unnamed",
        subject_ref=representative,
        fingerprints=simulation.meta.fingerprints,
    )


def _accumulate(
    collected: dict[str, list[float]],
    drawn: tuple[Mapping[str, np.ndarray], np.ndarray],
    truths: Mapping[str, np.ndarray],
    index: int,
) -> None:
    """Append one replicate's rank for every scalar coordinate it covers."""
    draws, weights = drawn
    for name, truth in truths.items():
        flat_truth = np.asarray(truth[index]).reshape(-1)
        sample = np.asarray(draws[name])
        flat = sample.reshape(sample.shape[0], -1)
        for position, coordinate in enumerate(
            _flat_coordinates(name, truth.shape[1:])
        ):
            collected[coordinate].append(
                continuous_rank(flat[:, position], weights, float(flat_truth[position]))
            )


def _accounting(ranks: SbcRanks, floor: int) -> Finding:
    """The replicate census, in the one finding a reader checks the count on."""
    return Finding(
        code="sbc_replicate_accounting",
        message=(
            f"{ranks.usable} of {ranks.requested} replicates produced a rank "
            f"on the {ranks.route} route "
            f"(refused {ranks.refused}, unconverged {ranks.unconverged}, "
            f"undrawn {ranks.undrawn})"
        ),
        observed=(
            ranks.requested,
            ranks.usable,
            ranks.refused,
            ranks.unconverged,
            ranks.undrawn,
            ranks.route,
        ),
        expected=floor,
    )


def _uniformity_findings(
    ranks: SbcRanks, level: float
) -> tuple[tuple[Finding, ...], bool]:
    """One KS finding per coordinate, and whether every one of them passed."""
    findings: list[Finding] = []
    # True with no evidence only when there are no ranks at all, which is
    # reachable only on a path that has already decided to ABSTAIN; a verdict
    # is never read off an empty sample.
    calibrated = True
    for coordinate, values in zip(ranks.coordinates, ranks.ranks, strict=True):
        if not values:
            continue
        test = stats.kstest(np.asarray(values), "uniform")
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
        if not ranks_are_uniform(p_value, level):
            calibrated = False
        findings.append(
            Finding(
                code="sbc_rank_uniformity",
                message=(
                    f"{coordinate}: KS D={statistic:.4f} p={p_value:.4g} over "
                    f"{len(values)} ranks, against alpha/K={level:.4g}"
                ),
                observed=(coordinate, statistic, p_value, len(values)),
                expected=level,
            )
        )
    return tuple(findings), calibrated


def sbc_report(
    ranks: SbcRanks,
    *,
    alpha: float = ALPHA,
    floor: int = REPLICATE_FLOOR,
) -> EvaluationReport:
    """Judge one :class:`SbcRanks`, on §0.2's two axes.

    * Any replicate that produced no rank -> APPLICABLE x ABSTAIN, finding
      ``replicates_not_completed``, with all three counts. Checked BEFORE the
      floor, because "45 of 100 replicates diverged" and "you only asked for
      45" are different facts and the first is the one worth reporting.
    * Fewer usable replicates than D106 -> APPLICABLE x ABSTAIN, finding
      ``replicates_below_floor``.
    * Otherwise every coordinate's KS p against ``alpha / K`` decides:
      all at or above -> PASS, any below -> FAIL.

    ``alpha / K`` is the Bonferroni correction §0.4 derives for K simultaneous
    coordinates; it is a derivation from D104 and carries no threshold of its
    own. The per-coordinate findings record the level they were compared
    against, so the verdict can be recomputed from the report -- which is
    §8's gate G8, and the reason the level is in ``expected`` rather than only
    in the message.
    """
    level = alpha / len(ranks.coordinates)
    findings, calibrated = _uniformity_findings(ranks, level)
    accounting = _accounting(ranks, floor)

    if ranks.unusable > 0:
        return _report(
            ranks,
            Conclusion.ABSTAIN,
            (
                Finding(
                    code="replicates_not_completed",
                    message=(
                        f"{ranks.unusable} of {ranks.requested} replicates "
                        "produced no rank; an SBC that drops its failures is "
                        "measuring the replicates that happened to work"
                    ),
                    observed=(ranks.refused, ranks.unconverged, ranks.undrawn),
                    expected=0,
                ),
                accounting,
                *findings,
            ),
            "replicates did not all complete",
        )
    if not replicates_meet_floor(ranks.usable, floor):
        return _report(
            ranks,
            Conclusion.ABSTAIN,
            (
                Finding(
                    code="replicates_below_floor",
                    message=(
                        f"{ranks.usable} usable replicates is below the D106 "
                        f"floor of {floor}, at which a 2x posterior width "
                        "error was detectable at every measured seed"
                    ),
                    observed=ranks.usable,
                    expected=floor,
                ),
                accounting,
                *findings,
            ),
            "below the replicate floor",
        )
    return _report(
        ranks,
        Conclusion.PASS if calibrated else Conclusion.FAIL,
        (accounting, *findings),
        "rank uniformity over every latent coordinate",
    )


def _report(
    ranks: SbcRanks,
    conclusion: Conclusion,
    findings: Sequence[Finding],
    summary: str,
) -> EvaluationReport:
    return EvaluationReport(
        meta=new_artifact_meta(
            artifact_type=ArtifactKind.EVALUATION_REPORT,
            fingerprints=ranks.fingerprints,
            producer=_PRODUCER,
            parent_refs=(ranks.subject_ref,),
            summary=f"simulation-based calibration: {summary}",
        ),
        subject_ref=ranks.subject_ref,
        report_kind=REPORT_KIND,
        applicability=Applicability.APPLICABLE,
        conclusion=conclusion,
        findings=tuple(findings),
    )


def simulation_based_calibration(
    graph: Graph,
    *,
    key: jax.Array,
    replicates: int,
    model_ref: ModelRef,
    build: GraphBuilder | None = None,
    budget: ComputeBudget | None = None,
    nuts_on_collapse: bool = False,
    sampler: Sampler | None = None,
    sampler_draws: int = 0,
    subject_ref: ArtifactRef | None = None,
    alpha: float = ALPHA,
    floor: int = REPLICATE_FLOOR,
) -> EvaluationReport | Refusal:
    """Run the replicates and judge them: :func:`sbc_ranks` then :func:`sbc_report`.

    The convenience the gate runner calls. It decides nothing the two halves
    do not: everything about which arm runs is in the first, everything about
    the verdict is in the second, and this function only refuses to judge what
    could not be measured.
    """
    ranks = sbc_ranks(
        graph,
        key=key,
        replicates=replicates,
        model_ref=model_ref,
        build=build,
        budget=budget,
        nuts_on_collapse=nuts_on_collapse,
        sampler=sampler,
        sampler_draws=sampler_draws,
        subject_ref=subject_ref,
    )
    if isinstance(ranks, Refusal):
        return ranks
    return sbc_report(ranks, alpha=alpha, floor=floor)
