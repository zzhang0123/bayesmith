"""R3 Task 4: what `held_out_report` decides, and what would have to break first.

Every number below was measured in this checkout before it was written down.
The two elpd anchors are probe_28 §7's own
(``PYTHONPATH=. .venv/bin/python docs/probes/probe_28_model_checking_seams.py
100 7`` prints ``-1.019`` and ``-24.047``), reproduced here through the TYPED
task path rather than the probe's ``compile(graph).sample`` shortcut -- which
is itself worth one line of the record: the two paths agree to every digit
printed, so the seam the plan measured and the seam this module reads are the
same seam.

**The fixtures are constructed, not sampled into place.**  The Bonferroni pin
below needs a held-out point whose PIT lands strictly between ``α/4`` and
``α/2``; searching for a seed until one did would be a fixture that pins one
run of a random experiment.  Instead the held-out VALUE is a chosen constant
(``_TIGHT_VALUE``) and the conditioning data is untouched, so the property it
rests on -- ``0.0125 < PIT < 0.025`` -- has margin on both sides that a sweep
in this file measures rather than assumes.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import const, det, observe, sample, trace
from bayesmith.artifacts.base import (
    ArtifactRef,
    ComputeBudget,
    NamedArray,
    new_artifact_meta,
)
from bayesmith.artifacts.identity import ArtifactKind
from bayesmith.artifacts.refusal import Refusal
from bayesmith.artifacts.reports import Applicability, Conclusion, EvaluationReport
from bayesmith.artifacts.results import (
    AnalyticPosterior,
    DrawsPosterior,
    LogDensityAvailability,
    PredictiveResult,
    WeightedDrawsPosterior,
)
from bayesmith.artifacts.tasks import PosteriorTask, PredictiveTask, new_task_meta
from bayesmith.dispatch.predictive import pointwise_log_likelihood
from bayesmith.dispatch.task import PRODUCER, compile_task, execute_task
from bayesmith.evaluation import ALPHA, held_out_report
from bayesmith.evaluation.heldout import REPORT_KIND, _tail
from bayesmith.exact.gaussian import observation_parts
from bayesmith.graph.evaluate import evaluate
from tests.dispatch.test_task_protocol import model_ref

# ------------------------------------------------------------------ fixtures

X = jnp.linspace(1.0, 4.0, 8)
SIGMA = 0.5
PRIOR_STD = 2.0
NOISE = SIGMA * jax.random.normal(jax.random.key(0), X.shape)
STRAIGHT = 2.5 * X + NOISE
CURVED = 2.5 * X + 0.6 * X**2 + NOISE

#: A held-out value chosen (not searched for) so that its PIT falls between
#: ``ALPHA / 4`` and ``ALPHA / 2`` -- inside a two-point band, outside a
#: one-point one.  ``test_the_bonferroni_factor_is_the_held_out_count`` is the
#: only thing that depends on where it lands, and it measures the margin.
_TIGHT_VALUE = 7.72

#: The SECOND observed node's grid, deliberately (2, 3) rather than a vector.
#: Two things ride on its shape: `m` has to be summed ACROSS nodes rather than
#: read off the first one, and the module's claim that a matrix-shaped node
#: takes the vector's code path because positions are FLAT has to be something
#: a test holds rather than something the docstring asserts.
GRID_E = jnp.reshape(jnp.linspace(0.5, 3.0, 6), (2, 3))
NOISE_E = SIGMA * jax.random.normal(jax.random.key(1), GRID_E.shape)
MATRIX = 2.5 * GRID_E + NOISE_E

#: Its mask, withholding the (0, 2) and (1, 1) entries -- flat positions 2 and
#: 4 in C order, and 3 and 4 in Fortran order, so the two orders are
#: distinguishable by the positions the report names.
MASK_E = [[True, True, False], [True, False, True]]

BUDGET = ComputeBudget(draws=2000, warmup=1000, chains=1)


def line(xs, data, mask=None):
    """`straight_line`'s model at caller-supplied grid, data and mask."""

    def model():
        grid = const("X", jnp.asarray(xs))
        w = sample("w", lambda: dist.Normal(0.0, PRIOR_STD))
        mu = det("mu", lambda w_, x_: w_ * x_, w, grid, linear_in=("w",))
        observe(
            "d",
            lambda m: dist.Normal(m, SIGMA),
            mu,
            obs=jnp.asarray(data),
            mask=None if mask is None else jnp.asarray(mask),
        )

    return trace(model)


def line_and_a_matrix_shaped_node():
    """One latent, TWO observed nodes, each withholding some of its own points.

    Every other fixture in this file masks exactly one observed node, which
    leaves two things in `held_out_report` unpinned: whether `_held_out` walks
    all the observed nodes or stops at the first one, and whether the
    Bonferroni `m` and the elpd are summed across nodes or taken from one.
    Both are invisible while there is only ever one node to get right.

    The second node is matrix-shaped for the second reason above; `w` is shared
    so the graph stays a single-latent conjugate problem on the exact route.
    """

    def model():
        grid = const("X", X)
        w = sample("w", lambda: dist.Normal(0.0, PRIOR_STD))
        mu = det("mu", lambda w_, x_: w_ * x_, w, grid, linear_in=("w",))
        observe(
            "d",
            lambda m: dist.Normal(m, SIGMA),
            mu,
            obs=STRAIGHT,
            mask=jnp.asarray([True] * 6 + [False] * 2),
        )
        grid_e = const("Xe", GRID_E)
        nu = det("nu", lambda w_, x_: w_ * x_, w, grid_e, linear_in=("w",))
        observe(
            "e",
            lambda m: dist.Normal(m, SIGMA),
            nu,
            obs=MATRIX,
            mask=jnp.asarray(MASK_E),
        )

    return trace(model)


def line_beside_a_correlated_node(data, mask):
    """The §0.4 coverage boundary, in the only shape that can carry a mask.

    Measured first, and it changed the fixture: ``observe`` REFUSES a mask on a
    correlated node outright ("a CirculantPrecision has no per-sample sigma, so
    there is nothing for a mask to select"), so "a masked correlated
    observation" is not a graph that exists.  What does exist, and is what a
    real model looks like when it goes out of the diagonal domain, is a graph
    with two observed nodes -- one masked and diagonal, one correlated.
    ``observation_parts`` walks BOTH, so the correlated one takes the whole
    graph out of the coverage domain, which is the honest answer: the premise
    "this graph's observations are diagonal Gaussian" is false, and a partial
    score would be a number resting on it anyway.
    """
    lag = np.minimum(np.arange(len(data)), len(data) - np.arange(len(data)))
    kernel = jnp.asarray(1.0 * 0.4**lag + 0.5)

    def model():
        grid = const("X", X)
        w = sample("w", lambda: dist.Normal(0.0, PRIOR_STD))
        mu = det("mu", lambda w_, x_: w_ * x_, w, grid, linear_in=("w",))
        observe(
            "d",
            lambda m: dist.Normal(m, SIGMA),
            mu,
            obs=jnp.asarray(data),
            mask=jnp.asarray(mask),
        )
        observe(
            "c",
            lambda m: dist.CirculantNormal(m, kernel),
            mu,
            depends_on_prediction=False,
            obs=jnp.asarray(data),
        )

    return trace(model)


def posterior_of(graph, seed=3):
    task = PosteriorTask(
        meta=new_task_meta(label="heldout"), budget=BUDGET, nuts_on_collapse=False
    )
    planned = compile_task(graph, task, model_ref=model_ref())
    assert not isinstance(planned, Refusal), planned
    result = execute_task(planned, key=jax.random.key(seed))
    assert not isinstance(result, Refusal), result
    return result


def source_ref(posterior):
    return ArtifactRef(
        artifact_id=posterior.meta.artifact_id,
        revision=posterior.meta.revision,
        artifact_type=ArtifactKind.RESULT,
    )


def predictive_of(graph, posterior, *, latent_sites=("w",), seed=4):
    """The subject, over every observed node the GRAPH declares.

    Reading the sites off the graph rather than writing ``("d",)`` here is what
    lets the two-node fixture below exist at all; for every single-node fixture
    it is the same tuple it always was, so nothing that was measured moves.
    """
    task = PredictiveTask(
        meta=new_task_meta(label="heldout"),
        source_posterior_ref=source_ref(posterior),
        conditioned_sites=graph.observed,
        replicated_sites=graph.observed,
        latent_sites=latent_sites,
    )
    planned = compile_task(graph, task, model_ref=model_ref())
    assert not isinstance(planned, Refusal), planned
    result = execute_task(planned, key=jax.random.key(seed), source_posterior=posterior)
    assert not isinstance(result, Refusal), result
    return result


def hand_built_predictive(posterior):
    """A latent-only `PredictiveResult` over a posterior's own draws.

    Needed where `execute_task` will not produce one -- a correlated
    observation is refused at the predictive seam, so the only way to ask what
    this module does with such a graph is to build the subject directly.  The
    protocol allows it: a predictive result holds latent draws, replicated
    draws, or both.
    """
    return PredictiveResult(
        meta=new_artifact_meta(
            artifact_type=ArtifactKind.RESULT,
            fingerprints=posterior.meta.fingerprints,
            producer=PRODUCER,
            summary="latent draws, carried without replication",
        ),
        run=posterior.run,
        source_posterior_ref=source_ref(posterior),
        latent_draws=tuple(posterior.representation.draws),
    )


@functools.cache
def scored(kind):
    """`(report, graph, predictive, posterior)` for one named fixture, once."""
    graphs = {
        "straight": lambda: line(X, STRAIGHT, [True] * 6 + [False] * 2),
        "curved": lambda: line(X, CURVED, [True] * 6 + [False] * 2),
        "unmasked": lambda: line(X, STRAIGHT),
        "all_true": lambda: line(X, STRAIGHT, [True] * 8),
        "tight_one": lambda: line(
            X[:7],
            np.concatenate([np.asarray(STRAIGHT[:6]), [_TIGHT_VALUE]]),
            [True] * 6 + [False],
        ),
        "two_nodes": line_and_a_matrix_shaped_node,
        "tight_two": lambda: line(
            X,
            np.concatenate(
                [np.asarray(STRAIGHT[:6]), [_TIGHT_VALUE], [float(STRAIGHT[7])]]
            ),
            [True] * 6 + [False, False],
        ),
    }
    graph = graphs[kind]()
    posterior = posterior_of(graph)
    predictive = predictive_of(graph, posterior)
    report = held_out_report(graph, predictive, source_posterior=posterior)
    return report, graph, predictive, posterior


def points(report):
    """`{(node, index): (pit, lpd)}` read back out of the findings."""
    return {
        (finding.observed[0], finding.observed[1]): (
            finding.observed[2],
            finding.observed[3],
        )
        for finding in report.findings
        if finding.code == "held_out_point"
    }


def elpd(report):
    for finding in report.findings:
        if finding.code == "held_out_elpd":
            return finding.observed[0]
    raise AssertionError("no held_out_elpd finding")


def codes(report):
    return {finding.code for finding in report.findings}


# --------------------------------------------------------- the derived band


def test_the_band_is_alpha_halved_and_bonferroni_corrected():
    """§0.4's derivation, written where a change to it is visible.

    Two halvings, and they are different halvings: `/2` because the test is
    two-sided, `/m` because m points are tested at once.  A guard that only
    checked `_tail(1)` would pass for an implementation that ignored the point
    count entirely, which is the mutation
    `test_the_bonferroni_factor_is_the_held_out_count` kills at the fixture
    level.
    """
    assert _tail(1) == ALPHA / 2
    assert _tail(2) == ALPHA / 4
    assert _tail(5) == ALPHA / 10
    with pytest.raises(ValueError, match="at least one point"):
        _tail(0)


def test_the_reported_band_is_the_one_the_verdict_used():
    """Every point's `expected` pair is `(tail, 1 - tail)` for THIS report's m."""
    report, *_ = scored("straight")
    tail = ALPHA / (2 * 2)
    for finding in report.findings:
        if finding.code == "held_out_point":
            assert finding.expected == (tail, 1.0 - tail)


# ------------------------------------------------------- the §0.8 premise


def test_masked_positions_contribute_nothing_to_the_conditioning():
    """probe_28 §7's other measurement, and the premise §0.8 rests on.

    If a masked position DID condition, then "held out" would be a label with
    no consequence and every number below would be an in-sample score wearing
    an out-of-sample name.  Pinned here rather than trusted because this
    module reads the same mask to decide what to score.
    """
    _report, graph, _predictive, posterior = scored("straight")
    draws = {array.name: array.value for array in posterior.representation.draws}
    pointwise = pointwise_log_likelihood(graph, draws)
    mask = np.asarray(graph.node("d").observed_mask)
    assert np.all(np.asarray(pointwise.value)[:, ~mask] == 0.0)
    assert np.all(np.asarray(pointwise.value)[:, mask] != 0.0)


# ---------------------------------------------------------------- the anchors


def test_a_masked_straight_line_passes_and_reports_probe_28s_elpd():
    report, *_ = scored("straight")

    assert isinstance(report, EvaluationReport)
    assert report.report_kind == REPORT_KIND
    assert report.applicability is Applicability.APPLICABLE
    assert report.conclusion is Conclusion.PASS
    assert len(points(report)) == 2
    assert elpd(report) == pytest.approx(-1.019270, abs=1e-2)
    measured = points(report)
    assert measured[("d", 6)][0] == pytest.approx(0.3161115, abs=1e-3)
    assert measured[("d", 7)][0] == pytest.approx(0.6282724, abs=1e-3)


def test_a_curved_line_fails_on_the_points_it_never_saw():
    """The G2 row: a model wrong in a way the CONDITIONED points barely show.

    The same six points condition both fixtures and the linear fit absorbs the
    curvature over them; it is the two withheld points, at the end of the grid
    where the quadratic term is largest, that the predictive cannot reach.
    """
    report, *_ = scored("curved")

    assert report.applicability is Applicability.APPLICABLE
    assert report.conclusion is Conclusion.FAIL
    assert elpd(report) == pytest.approx(-24.046592, abs=1e-2)
    assert all(pit > 0.99 for pit, _lpd in points(report).values())


def test_the_held_out_points_are_scored_out_of_sample():
    """The curved fixture's own control: its CONDITIONED points look fine.

    Without this line the previous test is also passed by an implementation
    that scores every position, since a badly misspecified model fails on all
    of them.  Here the six conditioned points' PITs are measured directly --
    through `observation_parts` rather than through this module -- and every
    one of them is less extreme than either withheld point.
    """
    _report, graph, _predictive, posterior = scored("curved")
    draws = {a.name: jnp.asarray(a.value) for a in posterior.representation.draws}

    def parts(values):
        data, loc, scale = observation_parts(graph, evaluate(graph, values))
        return dist.Normal(loc["d"], scale["d"]).cdf(data["d"])

    cdf = np.asarray(jax.vmap(parts)(draws), dtype=np.float64)
    pit = cdf.mean(axis=0)
    mask = np.asarray(graph.node("d").observed_mask)
    assert np.all(pit[mask] < 0.99)
    assert np.all(pit[~mask] > 0.99)
    assert pit[mask].max() < pit[~mask].min()


# ------------------------------------------------------------- the Bonferroni


def test_the_bonferroni_factor_is_the_held_out_count():
    """One PIT, two verdicts, and the only difference is m.

    The two fixtures condition on the SAME six points at the same six grid
    positions and withhold the same seventh value, so the shared held-out
    point's PIT is one number measured twice.  The seven-point fixture holds
    out only it (m=1, band `[0.025, 0.975]`); the eight-point fixture also
    holds out a value sitting on the line (m=2, band `[0.0125, 0.9875]`).
    The shared PIT lies between the two lower edges, so the verdict flips and
    nothing else could have flipped it.
    """
    one, *_ = scored("tight_one")
    two, *_ = scored("tight_two")

    shared_one = points(one)[("d", 6)][0]
    shared_two = points(two)[("d", 6)][0]
    assert shared_one == pytest.approx(shared_two, abs=1e-4)
    assert ALPHA / 4 < shared_one < ALPHA / 2, shared_one

    assert one.conclusion is Conclusion.FAIL
    assert two.conclusion is Conclusion.PASS
    assert len(points(one)) == 1 and len(points(two)) == 2


def test_the_tight_fixtures_margin_is_measured_not_assumed():
    """How far `_TIGHT_VALUE`'s PIT sits from each edge it must not touch.

    A pin that lands 1e-6 inside a boundary is a pin that a BLAS difference
    retires.  This records the distance in both directions so a future reader
    can see whether the previous test has room -- and turns red if a change to
    the seam moves the PIT toward either edge rather than only when it crosses.

    **The measurement, so that `2e-3` is not read as derived.**  It is not: the
    FORM of this test is derived (the previous test needs the PIT strictly
    inside `[ALPHA/4, ALPHA/2]`, so the quantity to record is its distance to
    each edge), and the CONSTANT is chosen, at roughly a third of what was
    measured.  Measured here, macOS/Accelerate and linux-amd64/OpenBLAS-ZEN
    printing the same digits::

        PIT             0.018669340414803656
        - ALPHA / 4     0.0061693404148036556
        ALPHA / 2 -     0.0063306595851963451

    So the guard has about 3x headroom on both sides.  It is written as a floor
    on the margin rather than as a two-sided band around the measured PIT
    because a band would ADMIT the region `<=` refuses -- this repository has
    twice shipped a widened band that silently released a mutant its
    predecessor killed.
    """
    one, *_ = scored("tight_one")
    pit = points(one)[("d", 6)][0]
    assert pit - ALPHA / 4 > 2e-3, pit
    assert ALPHA / 2 - pit > 2e-3, pit


# ----------------------------------------------------- more than one node


def test_a_second_masked_node_is_scored_and_counted_with_the_first():
    """Two observed nodes withhold points; `m` is 4, not either node's 2.

    Every other fixture here masks exactly ONE observed node, and while that is
    true an implementation that walked only the first observed node with a
    non-empty mask would be indistinguishable from the right one -- the point
    findings, the Bonferroni `m` and the summed elpd would all agree with
    themselves.  This fixture is what separates them: `d` withholds 2 of 8 and
    `e` withholds 2 of 6, so a first-node-only walk reports 2 points, a band of
    `ALPHA / 4` and an elpd of -0.925 instead of 4, `ALPHA / 8` and -2.126.
    """
    report, *_ = scored("two_nodes")

    measured = points(report)
    assert sorted(measured) == [("d", 6), ("d", 7), ("e", 2), ("e", 4)]

    tail = ALPHA / (2 * 4)
    for finding in report.findings:
        if finding.code == "held_out_point":
            assert finding.expected == (tail, 1.0 - tail)

    assert elpd(report) == pytest.approx(
        sum(lpd for _pit, lpd in measured.values()), rel=1e-12
    )
    assert elpd(report) == pytest.approx(-2.126227, abs=1e-2)
    from_d_alone = sum(
        lpd for (node, _index), (_pit, lpd) in measured.items() if node == "d"
    )
    assert from_d_alone == pytest.approx(-0.924644, abs=1e-2)

    for finding in report.findings:
        if finding.code == "held_out_elpd":
            assert finding.observed[1] == 4  # the summed point count


def test_a_matrix_shaped_node_is_scored_at_the_flat_positions_of_its_mask():
    """The module docstring's claim about flat indices, held by a test.

    `_held_out` flattens the mask, so a (2, 3) observed node is scored by the
    same code that scores a vector and the report names positions 2 and 4.
    Both halves are checked: that those ARE the C-order positions of the two
    False entries (in Fortran order they would be 3 and 4), and that the PIT
    filed at flat 2 is the one belonging to element (0, 2) -- computed here
    through `observation_parts` directly rather than through this module, and
    distinct enough from its neighbour that the mapping is not vacuous.
    """
    report, graph, _predictive, posterior = scored("two_nodes")
    mask = np.asarray(graph.node("e").observed_mask)
    assert mask.shape == (2, 3)
    assert list(np.flatnonzero(~mask.reshape(-1))) == [2, 4]

    draws = {a.name: jnp.asarray(a.value) for a in posterior.representation.draws}

    def cdf(values):
        data, loc, scale = observation_parts(graph, evaluate(graph, values))
        return dist.Normal(loc["e"], scale["e"]).cdf(data["e"])

    stacked = np.asarray(jax.vmap(cdf)(draws), dtype=np.float64)
    assert stacked.shape[1:] == (2, 3)

    measured = points(report)
    assert measured[("e", 2)][0] == pytest.approx(stacked[:, 0, 2].mean(), abs=1e-12)
    assert measured[("e", 4)][0] == pytest.approx(stacked[:, 1, 1].mean(), abs=1e-12)
    assert abs(stacked[:, 0, 2].mean() - stacked[:, 1, 1].mean()) > 0.1


# ------------------------------------------------------------------- weights


def test_a_weighted_posterior_is_scored_with_its_weights():
    """Drop the weights and this PASSES; keep them and it FAILS.

    The tilt is synthetic on purpose.  A real ``gcr+snis`` run on the
    radiometer fixture was measured first and its weighted and uniform answers
    differ in the fourth decimal (elpd -1.604429 against -1.610707), which
    makes it useless as a guard: an implementation that silently averaged
    uniformly would pass it.  So the source posterior's weights are replaced
    by an exponential tilt of its own draws, ``60 * (w - mean(w))``, which
    concentrates the posterior on its upper tail and moves the held-out PIT
    from 0.316 to 0.0029 -- across the band edge, so the VERDICT carries the
    difference rather than a tolerance.
    """
    _report, graph, predictive, posterior = scored("straight")
    draws = np.asarray(posterior.representation.draws[0].value, dtype=np.float64)
    tilted = dataclasses.replace(
        posterior,
        representation=WeightedDrawsPosterior(
            draws=posterior.representation.draws,
            log_weights=NamedArray(
                name="log_weights",
                value=60.0 * (draws - draws.mean()),
                dims=("draw",),
            ),
            method="tilted",
        ),
    )
    report = held_out_report(graph, predictive, source_posterior=tilted)

    assert report.conclusion is Conclusion.FAIL
    measured = points(report)
    assert measured[("d", 6)][0] < 0.005  # outside the band; carries the FAIL
    assert measured[("d", 7)][0] < 0.05  # 0.628 unweighted -- moved, but inside
    assert elpd(report) == pytest.approx(-6.443619, abs=1e-2)


# ----------------------------------------------------- the two abstain rows


@pytest.mark.parametrize("kind", ["unmasked", "all_true"])
def test_a_graph_that_withholds_nothing_is_inapplicable(kind):
    """§0.2: not a PASS.  A check with no subject has verified nothing."""
    report, *_ = scored(kind)

    assert report.applicability is Applicability.INAPPLICABLE
    assert report.conclusion is Conclusion.ABSTAIN
    assert codes(report) == {"no_held_out_points"}


def test_a_correlated_observation_is_unverifiable():
    """§0.4's coverage domain, carried through as UNVERIFIABLE rather than FAIL.

    A correlated observation is not evidence against the model -- it is a
    model this package's diagonal walk cannot score.  Filing it as a FAIL
    would put a modelling verdict on a coverage gap.
    """
    graph = line_beside_a_correlated_node(STRAIGHT, [True] * 6 + [False] * 2)
    posterior = posterior_of(graph)
    report = held_out_report(
        graph, hand_built_predictive(posterior), source_posterior=posterior
    )

    assert report.applicability is Applicability.UNVERIFIABLE
    assert report.conclusion is Conclusion.ABSTAIN
    assert codes(report) == {"predictive_noise_unsupported"}


def test_a_subject_that_carries_no_latents_is_unverifiable():
    """`latent_sites=()` produces a result whose forward model cannot be replayed."""
    graph = line(X, STRAIGHT, [True] * 6 + [False] * 2)
    posterior = posterior_of(graph)
    predictive = predictive_of(graph, posterior, latent_sites=())
    report = held_out_report(graph, predictive, source_posterior=posterior)

    assert report.applicability is Applicability.UNVERIFIABLE
    assert report.conclusion is Conclusion.ABSTAIN
    assert codes(report) == {"latent_draws_incomplete"}
    finding = report.findings[0]
    assert finding.observed == () and finding.expected == ("w",)


def test_a_posterior_with_no_draw_axis_is_unverifiable():
    """An analytic posterior has nothing to average over, and says so."""
    _report, graph, predictive, posterior = scored("straight")
    analytic = dataclasses.replace(
        posterior,
        representation=AnalyticPosterior(
            family="gaussian",
            parameters=(NamedArray(name="w_mean", value=np.array([2.5]), dims=("w",)),),
        ),
        pointwise_log_likelihood=None,
        log_density_availability=LogDensityAvailability.NONE,
    )
    report = held_out_report(graph, predictive, source_posterior=analytic)

    assert report.applicability is Applicability.UNVERIFIABLE
    assert codes(report) == {"source_posterior_holds_no_draws"}


# ----------------------------------------------------- caller errors, lineage


def test_another_runs_posterior_is_refused_as_a_caller_error():
    """Not a report.  The weights would be some other run's, and a verdict
    saying "unverifiable" would file the caller's mistake as a property of the
    model."""
    _report, graph, predictive, _posterior = scored("straight")
    other = posterior_of(line(X, STRAIGHT, [True] * 6 + [False] * 2), seed=11)
    with pytest.raises(TypeError, match="source_posterior_ref names"):
        held_out_report(graph, predictive, source_posterior=other)


def test_a_source_posterior_of_a_different_length_is_refused_not_broadcast():
    """The `weights.shape[0] != draws` guard, and what it stops.

    Not a hypothetical.  The failure it catches is a source posterior holding
    ONE draw against a subject holding 2000: NumPy broadcasts `(1, 1)` against
    `(2000, k)` without complaint, so with the guard removed this call returns
    a report -- a PIT and an elpd computed by weighting 2000 draws with one
    weight, which is a number no reader could tell from a posterior average.
    The guard is the difference between a refusal and a plausible wrong answer,
    which is why it is checked rather than assumed to be unreachable.
    """
    _report, graph, predictive, posterior = scored("straight")
    assert isinstance(posterior.representation, DrawsPosterior)
    truncated = dataclasses.replace(
        posterior,
        representation=DrawsPosterior(
            draws=tuple(
                dataclasses.replace(array, value=np.asarray(array.value)[:1])
                for array in posterior.representation.draws
            ),
            chain_shape=None,
            method=posterior.representation.method,
        ),
        pointwise_log_likelihood=dataclasses.replace(
            posterior.pointwise_log_likelihood,
            value=np.asarray(posterior.pointwise_log_likelihood.value)[:1],
        ),
    )

    with pytest.raises(TypeError, match="a weight per draw"):
        held_out_report(graph, predictive, source_posterior=truncated)


def test_the_report_points_at_what_it_read():
    report, _graph, predictive, _posterior = scored("straight")

    assert report.subject_ref.artifact_id == predictive.meta.artifact_id
    assert report.subject_ref.revision == predictive.meta.revision
    assert report.subject_ref.artifact_type is ArtifactKind.RESULT
    assert report.meta.parent_refs == (report.subject_ref,)


def test_the_verdict_can_be_recomputed_from_the_findings():
    """G8: no log reading.  Every input the verdict used is in the findings."""
    for kind, expected in (("straight", Conclusion.PASS), ("curved", Conclusion.FAIL)):
        report, *_ = scored(kind)
        recomputed = Conclusion.PASS
        for finding in report.findings:
            if finding.code != "held_out_point":
                continue
            low, high = finding.expected
            if not low <= finding.observed[2] <= high:
                recomputed = Conclusion.FAIL
        assert recomputed is expected is report.conclusion


def test_the_report_round_trips_through_the_codec():
    from bayesmith.artifacts._codec import canonical_dumps, canonical_loads

    report, *_ = scored("curved")
    restored = canonical_loads(canonical_dumps(report))

    assert restored == report
    assert elpd(restored) == elpd(report)
