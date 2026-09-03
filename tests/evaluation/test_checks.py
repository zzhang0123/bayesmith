"""The predictive checks: every verdict, and the numbers each one rests on.

Every p-value below was MEASURED through this module's own API on this
checkout and is pinned as a number, not as an inequality.  That is affordable
because none of these is a random trial: the seed, the draw count and the
discrepancy set are all fixed, so a p-value here is a deterministic function of
the fixture.  §9.3's "declare the seed, the budget and the tolerated
false-positive rate in advance" is therefore satisfied at its strongest -- the
tolerated false-positive count of this file is ZERO, because nothing in it is
re-drawn.

**Where a pin is a count rather than a float.**  An iid posterior of 2000
draws gives every draw weight 1/2000, so a p-value is a multiple of 0.0005 and
``pytest.approx(..., abs=1e-9)`` is pinning the exact count of draws that
crossed.  The weighted fixture (``radiometer``, gcr+snis) is different: its
weights come out of a float32 JAX computation, so its p-values are pinned at
``rel=1e-6``, which is looser than the counted ones and tighter than any
change to the seam would be.

**Measured on Linux as well as on the development laptop.**  Sixteen fixtures
in this repository once wrote down what one machine's arithmetic produced and
called it a property, and four release tags were spent on it.  Every pin below
was re-run under ``linux/amd64`` with ``OPENBLAS_CORETYPE=ZEN`` -- Accelerate
against scipy-openblas, which is the difference that burned the four tags --
and all 47 tests in this file passed unchanged on both.  A pin that had moved
would have been rewritten as a property before it was committed, not widened.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import const, det, observe, sample, trace
from bayesmith.artifacts._codec import (
    ArtifactCodecError,
    canonical_dumps,
    canonical_loads,
    canonical_payload,
)
from bayesmith.artifacts.base import (
    ArtifactKind,
    ArtifactRef,
    ComputeBudget,
    new_artifact_meta,
)
from bayesmith.artifacts.gates import (
    AttemptStatus,
    GateDefinition,
    OperationalStatus,
    ReportRequirement,
    ReportSlot,
    aggregate_gate,
)
from bayesmith.artifacts.refusal import Finding, Refusal
from bayesmith.artifacts.reports import Applicability, Conclusion, EvaluationReport
from bayesmith.artifacts.results import PredictiveResult, WeightedDrawsPosterior
from bayesmith.artifacts.tasks import (
    ParameterSource,
    PosteriorTask,
    PredictiveTask,
    SimulationTask,
    new_task_meta,
)
from bayesmith.dispatch.task import PRODUCER as DISPATCH_PRODUCER
from bayesmith.dispatch.task import compile_task, execute_task
from bayesmith.evaluation import ALPHA
from bayesmith.evaluation import checks as checks_module
from bayesmith.evaluation.checks import (
    DEFAULT_DISCREPANCIES,
    DRAW_FLOOR,
    discrepancy_identity,
    draws_resolve_the_band,
    posterior_predictive_check,
    prior_predictive_check,
    tail_mass_within_rate,
)
from tests.dispatch.test_task_protocol import model_ref
from tests.exact.models import radiometer, straight_line

_PROBE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "docs"
    / "probes"
    / "probe_28_model_checking_seams.py"
)


def probe():
    """probe_28 as a module -- the same loader ``test_probe28_pins.py`` uses.

    ``curved_line`` lives there and nowhere else, and §0.3's whole
    limited-power measurement is a comparison between its 0.6 and its 0.15, so
    the fixture is imported rather than restated: a second copy of it here
    would be a second model that agreed with the plan's numbers by luck.
    """
    spec = importlib.util.spec_from_file_location(_PROBE.stem, _PROBE)
    assert spec is not None and spec.loader is not None, _PROBE
    module = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = [str(_PROBE)]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


# ------------------------------------------------------------------ fixtures


def posterior_of(graph, key, *, draws=2000):
    task = PosteriorTask(
        meta=new_task_meta(label="r3-checks"),
        budget=ComputeBudget(draws=draws, warmup=min(1000, draws), chains=1),
        nuts_on_collapse=False,
    )
    planned = compile_task(graph, task, model_ref=model_ref())
    assert not isinstance(planned, Refusal), planned
    result = execute_task(planned, key=key)
    assert not isinstance(result, Refusal), result
    return result


def predictive_of(graph, posterior, key, *, latent_sites=("w",)):
    task = PredictiveTask(
        meta=new_task_meta(label="r3-checks-ppc"),
        source_posterior_ref=ArtifactRef(
            artifact_id=posterior.meta.artifact_id,
            revision=posterior.meta.revision,
            artifact_type=ArtifactKind.RESULT,
        ),
        conditioned_sites=("d",),
        replicated_sites=("d",),
        latent_sites=latent_sites,
    )
    planned = compile_task(graph, task, model_ref=model_ref())
    assert not isinstance(planned, Refusal), planned
    return execute_task(planned, key=key, source_posterior=posterior)


def prior_simulation(graph, key, draws, *, latent_sites=("w",)):
    task = SimulationTask(
        meta=new_task_meta(label="r3-checks-prior"),
        parameter_source=ParameterSource.prior(),
        latent_sites=latent_sites,
        observed_sites=("d",),
        budget=ComputeBudget(draws=draws),
    )
    planned = compile_task(graph, task, model_ref=model_ref())
    assert not isinstance(planned, Refusal), planned
    return execute_task(planned, key=key)


def checked(graph, key=1, *, draws=2000):
    """``(posterior, predictive)`` at one seed -- the pair every PPC pin uses."""
    root = jax.random.key(key)
    posterior = posterior_of(graph, root, draws=draws)
    return posterior, predictive_of(graph, posterior, jax.random.fold_in(root, 7))


def line_with_prior(prior_std):
    """``straight_line``'s data under a prior of the caller's choosing."""
    module = probe()
    observed = jnp.asarray(straight_line().node("d").observed)

    def model():
        xs = const("X", module.X)
        w = sample("w", lambda: dist.Normal(0.0, prior_std))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, module.SIGMA), mu, obs=observed)

    return trace(model)


def masked_line(mask):
    """Four points on a line; ``mask`` says which of them were taken."""
    x = jnp.linspace(1.0, 4.0, 4)
    data = jnp.array([2.4, 5.1, 7.6, 10.2])

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 2.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, 0.5), mu, obs=data, mask=mask)

    return trace(model)


def correlated_line(size=8, weight=2.0, decay=0.4):
    """An observed node whose noise is CirculantNormal -- outside R2 §0.4."""
    lag = np.minimum(np.arange(size), size - np.arange(size))
    kernel = jnp.asarray(1.0 * decay**lag + 0.5)
    x = jnp.linspace(1.0, 4.0, size)

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 5.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe(
            "d",
            lambda m: dist.CirculantNormal(m, kernel),
            mu,
            depends_on_prediction=False,
            obs=weight * x,
        )

    return trace(model)


def excess_kurtosis(y, loc):
    """A user discrepancy with a home: defined at module scope, so importable.

    Not in the default set (§0.3 keeps that to five stable statistics) and not
    a lambda, which is the whole point of its presence here.
    """
    del loc
    centred = y - np.mean(y, axis=-1, keepdims=True)
    return np.mean(centred**4, axis=-1) / np.mean(centred**2, axis=-1) ** 2 - 3.0


def p_values(report):
    """``{short discrepancy name: p}`` off a report's own findings."""
    return {
        finding.observed[1].rsplit(".", 1)[1]: finding.observed[2]
        for finding in report.findings
    }


# ------------------------------------------------------- D104 and D105 first


def test_the_draw_floor_is_derived_from_the_declared_rate_not_written_beside_it():
    """D105 (§0.5).  The FORMULA is the registered thing, so the value follows.

    Recomputed from ALPHA rather than compared against a literal 40 -- if
    someone edits D104, this test must move with it instead of pinning a stale
    integer that used to be its consequence.  The 40 appears once, as what the
    formula gives at the α the layer currently declares.
    """
    assert DRAW_FLOOR == math.ceil(1.0 / (ALPHA / 2.0))
    assert DRAW_FLOOR == 40
    # ... and the derivation is a resolution argument, so it has to say the
    # thing it claims: at the floor, one draw is no coarser than one tail.
    assert 1.0 / DRAW_FLOOR <= ALPHA / 2.0
    assert 1.0 / (DRAW_FLOOR - 1) > ALPHA / 2.0


def test_both_thresholds_are_written_as_derivations_of_alpha_not_as_literals():
    """The registry files D105 as *derived* and D104's band as arithmetic on
    ALPHA; this is the test that makes those words checkable.

    Measured: replacing ``DRAW_FLOOR``'s formula with the literal 40 survives
    every other test in this file, because at the α the layer currently
    declares the formula gives 40 -- so an equality against the formula agrees
    with the literal and says nothing.  What the provenance actually claims is
    about the SHAPE of the definition: the floor follows α, and the band's
    edge is ``ALPHA / 2`` rather than a 0.025 that would sit unchanged beside
    a changed α.  Read off the module's own source, which is where that claim
    can be seen.
    """
    import ast

    source = pathlib.Path(checks_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    floor = next(
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "DRAW_FLOOR"
    )
    assert floor.value is not None
    names = {n.id for n in ast.walk(floor.value) if isinstance(n, ast.Name)}
    assert "ALPHA" in names, ast.unparse(floor.value)
    assert not isinstance(floor.value, ast.Constant), ast.unparse(floor.value)

    band = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "tail_mass_within_rate"
    )
    comparison = next(
        node for node in ast.walk(band) if isinstance(node, ast.Compare)
    )
    edge = comparison.comparators[0]
    assert "ALPHA" in {
        n.id for n in ast.walk(edge) if isinstance(n, ast.Name)
    }, ast.unparse(edge)
    assert not isinstance(edge, ast.Constant), ast.unparse(edge)


@pytest.mark.parametrize(
    ("draws", "resolves"),
    [(1, False), (8, False), (39, False), (40, True), (41, True), (2000, True)],
)
def test_the_draw_floor_is_closed_at_its_boundary(draws, resolves):
    """8 is the existing suite's own budget and 39/40 are the two faces."""
    assert draws_resolve_the_band(draws) is resolves


@pytest.mark.parametrize(
    ("p_value", "inside"),
    [
        (0.0, False),
        (0.02, False),
        (0.025, True),
        (0.5, True),
        (0.975, True),
        (0.98, False),
        (1.0, False),
    ],
)
def test_the_tail_mass_predicate_is_the_two_sided_band(p_value, inside):
    """D104 (§0.4): one comparison, both tails.

    ``min(p, 1 - p) >= ALPHA / 2`` has to agree with the band §0.4 writes,
    ``[ALPHA/2, 1 - ALPHA/2]``, at both edges and outside both of them -- and
    it is the comparison, not the band, that the report records, so a reader
    recomputing a verdict gets the same answer at the last ULP.
    """
    assert tail_mass_within_rate(min(p_value, 1.0 - p_value)) is inside
    assert (ALPHA / 2.0 <= p_value <= 1.0 - ALPHA / 2.0) is inside


def test_a_tie_counts_towards_the_p_value_and_the_weights_are_renormalised():
    """§0.3 writes ``1[T(rep_i) >= T(obs_i)]``, and the ``>=`` is the whole
    content of this test.

    No fixture in this file can tell ``>=`` from ``>``: a tie between two
    continuous statistics has probability zero, so the two spellings agree on
    every measured cell and a mutation from one to the other survives them
    all.  Called directly with a tie, they differ by the whole answer -- 1.0
    against 0.0 -- which is what makes the convention checkable rather than
    merely written down.

    The second half is the renormalisation, measured during this task: 4000
    weights of 1/4000 with every indicator true summed to 1.0000000000000004,
    so ``min(p, 1 - p)`` came out at -4.4e-16 -- a negative probability.  A
    saturated cell must be exactly 1.0.
    """
    tied = np.array([2.0, 2.0, 2.0, 2.0])
    weights = np.full(4, 0.25)
    assert checks_module._p_value(weights, tied, tied) == 1.0

    many = np.full(4000, 1.0 / 4000)
    saturated = np.ones(4000)
    assert checks_module._p_value(many, saturated, np.zeros(4000)) == 1.0
    assert checks_module._p_value(many, np.zeros(4000), saturated) == 0.0
    assert min(1.0, 1.0 - checks_module._p_value(many, saturated, np.zeros(4000))) == 0.0


def test_the_evaluation_layer_writes_reports_as_the_same_producer_dispatch_does():
    """``checks.PRODUCER`` is spelled twice, so a test holds the two together.

    It cannot be imported from ``dispatch.task``: ``tests/test_layering.py``
    asserts the in-degree of ``dispatch`` is zero, and a module-scope import
    here would make it one.  A second spelling of one fact is the defect this
    repository has repaired most often, so it is enforced rather than trusted.
    """
    assert checks_module.PRODUCER == DISPATCH_PRODUCER


# ---------------------------------------------- §0.3: identity, never object


def test_a_lambda_has_an_address_and_no_home_so_it_is_refused():
    """§0.3's "lambda / REPL function refused", which is NOT what the rule it
    cites does on its own.

    Measured: ``dispatch.amortized._embed_identity`` accepts a lambda and
    returns ``'<module>.<lambda>'``, because its only test is that
    ``__module__`` and ``__qualname__`` are non-empty and ``'<lambda>'`` is
    non-empty.  The identity a report records has to be one a later reader can
    turn back into the statistic, so this one additionally resolves it.
    """
    from bayesmith.dispatch.amortized import _embed_identity

    def anonymous(y, loc):  # a closure: 'test_....<locals>.anonymous'
        return y

    assert _embed_identity(anonymous).endswith("<locals>.anonymous")
    with pytest.raises(ValueError, match="does not import back"):
        discrepancy_identity(anonymous)
    with pytest.raises(ValueError, match="does not import back"):
        discrepancy_identity(lambda y, loc: y)


def test_a_rebound_module_level_name_is_refused_too(monkeypatch):
    """The case neither "lambda" nor "closure" covers: the address survives and
    stops meaning what it meant.

    ``original`` still calls itself ``bayesmith.evaluation.checks.sd``, and
    that name now resolves to a different function.  Reading ``__qualname__``
    accepts it and records an identity that would give a later reader the
    wrong statistic; resolving it and comparing objects refuses it.
    """
    original = checks_module.sd
    monkeypatch.setattr(checks_module, "sd", checks_module.mean)
    assert original.__qualname__ == "sd"
    with pytest.raises(ValueError, match="is not the discrepancy that was passed"):
        discrepancy_identity(original)


def test_the_five_default_discrepancies_all_have_importable_identities():
    """§0.3's default set, and the order it is listed in, which the report
    preserves so two runs of one check compare equal."""
    assert [item.__name__ for item in DEFAULT_DISCREPANCIES] == [
        "mean",
        "sd",
        "smallest",
        "largest",
        "residual_sd",
    ]
    for item in DEFAULT_DISCREPANCIES:
        identity = discrepancy_identity(item)
        assert identity == f"bayesmith.evaluation.checks.{item.__name__}"


def test_the_codec_is_what_refuses_a_callable_in_an_artifact():
    """§0 ruling 4, at the two places a callable could get in.

    The report records an identity STRING, and the reason that is not merely a
    convention is that the layer below refuses the alternative: a ``Finding``
    will not hold a function, and the codec will not encode one.  Both are
    checked, because a caller building a finding by hand meets the first and a
    caller round-tripping an artifact meets the second.
    """
    with pytest.raises(TypeError, match="canonical value"):
        Finding(code="c", message="m", observed=checks_module.mean)
    with pytest.raises(ArtifactCodecError, match="no canonical encoding"):
        canonical_payload({"discrepancy": checks_module.mean})


def test_a_user_discrepancy_reaches_the_report_by_identity(measured):
    """A statistic this module has never heard of, recorded by where it lives."""
    posterior, predictive = measured["straight_line"]
    report = posterior_predictive_check(
        straight_line(),
        predictive,
        source_posterior=posterior,
        discrepancies=(excess_kurtosis,),
    )
    (finding,) = report.findings
    assert finding.observed[1] == f"{excess_kurtosis.__module__}.excess_kurtosis"
    assert finding.observed[2] == pytest.approx(0.19550, abs=1e-9)
    assert report.conclusion is Conclusion.PASS


# ------------------------------------------------ the measured PPC fixtures


@pytest.fixture(scope="module")
def measured():
    """Every (posterior, predictive) pair the pins below read, drawn once.

    Module-scoped because each pair is 1000 warmup + 2000 draws through the
    typed task path and four of them are needed; the seeds are fixed, so
    sharing them across tests changes no number.
    """
    module = probe()
    return {
        "straight_line": checked(straight_line()),
        "radiometer": checked(radiometer()),
        "curved_06": checked(module.curved_line(curvature=0.6)),
        "curved_015": checked(module.curved_line(curvature=0.15)),
    }


def test_a_calibrated_fixture_passes_and_its_five_p_values_are_pinned(measured):
    """G1's first row: ``straight_line`` at 2000 gcr draws, all five in band.

    Counted, not approximated: an iid posterior weights every draw 1/2000, so
    0.3005 is 601 draws of 2000 and 0.0450 is 90.  ``residual_sd``'s 0.3270
    is the number probe_28 §1 printed as ``p_scale``, reproduced here through
    the typed ``PredictiveResult`` -- the untyped probe and the artifact path
    agree to every digit, which is what makes the plan's §0.12 table a
    statement about this code rather than about a script.
    """
    posterior, predictive = measured["straight_line"]
    report = posterior_predictive_check(
        straight_line(), predictive, source_posterior=posterior
    )
    assert report.report_kind == "posterior_predictive_check"
    assert report.applicability is Applicability.APPLICABLE
    assert report.conclusion is Conclusion.PASS
    assert p_values(report) == {
        "mean": pytest.approx(0.3005, abs=1e-9),
        "sd": pytest.approx(0.9465, abs=1e-9),
        "smallest": pytest.approx(0.0450, abs=1e-9),
        "largest": pytest.approx(0.3605, abs=1e-9),
        "residual_sd": pytest.approx(0.3270, abs=1e-9),
    }
    assert all(
        finding.code == "discrepancy_within_band" for finding in report.findings
    )


def test_a_weighted_source_is_scored_with_its_weights(measured):
    """G1's second row, and the reason ``source_posterior`` is not optional.

    ``radiometer`` routes to gcr+snis, so its posterior is a
    ``WeightedDrawsPosterior`` whose heaviest draw carries 1.6x the uniform
    weight.  The pinned 0.746309 is probe_28 §1's weighted ``p_scale``
    (0.7463) recomputed here.  The second half of the test is the guard that
    matters: recomputing the same cell with the weights thrown away moves the
    answer, so a version of this check that dropped them would be caught by a
    number rather than by a reading of the code.
    """
    posterior, predictive = measured["radiometer"]
    graph = radiometer()
    report = posterior_predictive_check(graph, predictive, source_posterior=posterior)
    assert isinstance(posterior.representation, WeightedDrawsPosterior)
    assert report.conclusion is Conclusion.PASS
    assert p_values(report)["residual_sd"] == pytest.approx(0.746309, rel=1e-6)

    weights = checks_module._weights(posterior)
    assert float(weights.max()) > 1.5 / len(weights)
    uniform = np.full(len(weights), 1.0 / len(weights))
    monkeyed = dataclasses.replace(
        posterior,
        representation=posterior.representation.__class__(
            draws=posterior.representation.draws,
            log_weights=dataclasses.replace(
                posterior.representation.log_weights,
                value=np.zeros_like(weights),
            ),
            ess=posterior.representation.ess,
            khat=posterior.representation.khat,
            unreliable=posterior.representation.unreliable,
            method=posterior.representation.method,
        ),
    )
    flat = posterior_predictive_check(
        graph,
        dataclasses.replace(
            predictive,
            source_posterior_ref=ArtifactRef(
                artifact_id=monkeyed.meta.artifact_id,
                revision=monkeyed.meta.revision,
                artifact_type=ArtifactKind.RESULT,
            ),
        ),
        source_posterior=monkeyed,
    )
    assert np.allclose(checks_module._weights(monkeyed), uniform)
    assert p_values(flat)["residual_sd"] != p_values(report)["residual_sd"]


def test_a_misspecified_fixture_fails_on_three_of_the_five(measured):
    """G2: ``curved_line(0.6)`` -- a quadratic term the model cannot express.

    Saturated, and that is why it is pinned as an equality rather than as a
    bound: not one of 2000 replicated datasets reaches the observed spread, so
    ``sd``, ``largest`` and ``residual_sd`` are exactly 0.0.  ``mean`` and
    ``smallest`` sit comfortably inside the band on the SAME data, which is
    the limited-power sentence with a subject.
    """
    posterior, predictive = measured["curved_06"]
    report = posterior_predictive_check(
        probe().curved_line(curvature=0.6), predictive, source_posterior=posterior
    )
    assert report.applicability is Applicability.APPLICABLE
    assert report.conclusion is Conclusion.FAIL
    assert p_values(report) == {
        "mean": pytest.approx(0.8815, abs=1e-9),
        "sd": 0.0,
        "smallest": pytest.approx(0.8275, abs=1e-9),
        "largest": 0.0,
        "residual_sd": 0.0,
    }
    outside = [
        finding.observed[1].rsplit(".", 1)[1]
        for finding in report.findings
        if finding.code == "discrepancy_outside_band"
    ]
    assert outside == ["sd", "largest", "residual_sd"]


def test_a_real_misspecification_this_check_does_not_catch(measured):
    """§0.3's power statement, as a test rather than as a sentence.

    ``curved_line(0.15)`` is wrong in exactly the way ``curved_line(0.6)`` is
    wrong -- the same model, the same missing quadratic term, a quarter of the
    size -- and this check PASSES it on all five statistics.  A reader who
    takes a PASS for "the model is correct" is contradicted by a green test in
    the same file as the pins, which is the strongest place to put the caveat.
    """
    posterior, predictive = measured["curved_015"]
    report = posterior_predictive_check(
        probe().curved_line(curvature=0.15), predictive, source_posterior=posterior
    )
    assert report.conclusion is Conclusion.PASS
    assert p_values(report) == {
        "mean": pytest.approx(0.4640, abs=1e-9),
        "sd": pytest.approx(0.5495, abs=1e-9),
        "smallest": pytest.approx(0.1615, abs=1e-9),
        "largest": pytest.approx(0.1045, abs=1e-9),
        "residual_sd": pytest.approx(0.3410, abs=1e-9),
    }
    assert "a pass bounds these statistics and nothing wider" in report.meta.summary


def test_eight_draws_abstain_rather_than_pass():
    """G3: D105's whole purpose.

    Eight draws resolve a p-value to 0.125, six times the 0.025 tail the
    declared rate reserves -- so no p-value it can produce lies inside a tail
    without being 0 or 1.  The report is APPLICABLE (the check does apply to
    this subject) and ABSTAIN (it could not decide), which is §0 ruling 7's
    two axes doing the work one axis could not.
    """
    graph = straight_line()
    posterior, predictive = checked(graph, draws=8)
    report = posterior_predictive_check(graph, predictive, source_posterior=posterior)
    assert report.applicability is Applicability.APPLICABLE
    assert report.conclusion is Conclusion.ABSTAIN
    (finding,) = report.findings
    assert finding.code == "draws_below_resolution"
    assert finding.observed == 8
    assert finding.expected == DRAW_FLOOR


# ------------------------------------------------------- prior predictive


def test_the_models_own_prior_generates_data_like_the_data():
    """G1: ``straight_line``'s ``w ~ N(0, 2)``, 4000 prior draws at seed 9.

    Reproducible to the draw and with no linear solve anywhere in it: a prior
    draw is threefry bits, a scalar-vector product and an elementwise Normal
    sample.  0.11375 is 455 draws of 4000.
    """
    graph = straight_line()
    simulation = prior_simulation(graph, jax.random.key(9), 4000)
    report = prior_predictive_check(graph, simulation)
    assert report.report_kind == "prior_predictive_check"
    assert report.applicability is Applicability.APPLICABLE
    assert report.conclusion is Conclusion.PASS
    assert p_values(report) == {
        "mean": pytest.approx(0.11375, abs=1e-9),
        "sd": pytest.approx(0.29175, abs=1e-9),
        "smallest": pytest.approx(0.05750, abs=1e-9),
        "largest": pytest.approx(0.11250, abs=1e-9),
        "residual_sd": pytest.approx(0.05150, abs=1e-9),
    }


def test_a_prior_a_million_times_too_wide_fails_on_both_sides():
    """G2: the same data under ``w ~ N(0, 1e6)``, saturated in BOTH directions.

    ``sd`` is exactly 1.0 -- every one of 4000 prior datasets is wider than
    the data -- and ``residual_sd`` is exactly 0.0, because the observed
    residual against a prior loc of order 1e6 dwarfs any replicated one.  The
    two saturate opposite ends of the band, which is what makes this a
    two-sided check rather than a one-sided one.  ``mean``, ``smallest`` and
    ``largest`` all pass on the same draws.
    """
    graph = line_with_prior(1e6)
    simulation = prior_simulation(graph, jax.random.key(9), 4000)
    report = prior_predictive_check(graph, simulation)
    assert report.conclusion is Conclusion.FAIL
    scores = p_values(report)
    assert scores["sd"] == 1.0
    assert scores["residual_sd"] == 0.0
    assert scores["mean"] == pytest.approx(0.5120, abs=1e-9)


def test_a_simulation_from_a_posterior_is_inapplicable_not_failed():
    """G3: the source is a different question, so the check says so.

    INAPPLICABLE rather than UNVERIFIABLE: the inputs are all present and
    perfectly good, they are just not the ones a statement about the priors is
    made from.  §0 ruling 7 keeps those apart because only one of them is
    worth chasing.
    """
    graph = straight_line()
    posterior = posterior_of(graph, jax.random.key(1), draws=64)
    task = SimulationTask(
        meta=new_task_meta(label="r3-checks-from-posterior"),
        parameter_source=ParameterSource.from_posterior_result(
            ArtifactRef(
                artifact_id=posterior.meta.artifact_id,
                revision=posterior.meta.revision,
                artifact_type=ArtifactKind.RESULT,
            )
        ),
        latent_sites=("w",),
        observed_sites=("d",),
    )
    planned = compile_task(graph, task, model_ref=model_ref())
    simulation = execute_task(
        planned, key=jax.random.key(3), source_posterior=posterior
    )
    report = prior_predictive_check(graph, simulation)
    assert report.applicability is Applicability.INAPPLICABLE
    assert report.conclusion is Conclusion.ABSTAIN
    (finding,) = report.findings
    assert finding.code == "parameter_source_not_prior"
    assert (finding.observed, finding.expected) == ("posterior_result", "prior")


def test_a_simulation_carrying_no_latent_draws_is_unverifiable():
    """G3: the second argument of every discrepancy cannot be built.

    A simulation that named no ``latent_sites`` has observations and no
    parameters, so ``loc`` cannot be recomputed.  Whether a particular
    callable reads ``loc`` is not knowable without running it -- so the check
    reports UNVERIFIABLE for the whole set rather than guessing which members
    would have been fine.
    """
    graph = straight_line()
    simulation = prior_simulation(graph, jax.random.key(9), 4000, latent_sites=())
    report = prior_predictive_check(graph, simulation)
    assert report.applicability is Applicability.UNVERIFIABLE
    assert report.conclusion is Conclusion.ABSTAIN
    (finding,) = report.findings
    assert finding.code == "discrepancy_needs_latent_draws"
    assert (finding.observed, finding.expected) == (("w",), ("w",))


def test_a_correlated_observation_is_unverifiable_not_an_exception():
    """R2 §0.4's coverage domain, reached through the one path that reaches it.

    A predictive TASK on this graph is refused by ``execute_task`` before any
    report exists, so the branch would be unreachable there.  A PRIOR
    simulation is not: ``prior_draws`` samples each node's own distribution,
    which a CirculantNormal answers perfectly well -- and then the check's
    ``observation_parts`` walk raises ``NotGaussian``.  The report abstains
    with the premise named; §7.3's "degrade, do not raise".
    """
    graph = correlated_line()
    assert isinstance(
        predictive_of(graph, posterior_of(graph, jax.random.key(1), draws=64),
                      jax.random.key(2)),
        Refusal,
    )
    simulation = prior_simulation(graph, jax.random.key(9), 100)
    report = prior_predictive_check(graph, simulation)
    assert report.applicability is Applicability.UNVERIFIABLE
    assert report.conclusion is Conclusion.ABSTAIN
    (finding,) = report.findings
    assert finding.code == "predictive_noise_unsupported"
    assert finding.expected == "diagonal_normal"


# --------------------------------------------------------------- the mask


def test_only_the_conditioned_points_are_compared():
    """§0.8: a masked position was not taken, so it is not what was fitted.

    The oracle is computed here from the replicated draws and the graph's own
    data, restricted by hand to the three conditioned units -- so this is not
    "masked differs from unmasked" (which a wrong restriction would also
    satisfy) but "the answer equals the one the conditioned units give".  The
    unmasked comparison is kept as the second half: on the same draws the
    numbers really do move, so the restriction is doing something.
    """
    graph = masked_line(jnp.array([True, True, True, False]))
    posterior, predictive = checked(graph)
    report = posterior_predictive_check(
        graph, predictive, source_posterior=posterior,
        discrepancies=(checks_module.largest,),
    )
    replicated = np.asarray(predictive.replicated_draws[0].value)
    observed = np.asarray(graph.node("d").observed)
    taken = np.asarray(graph.node("d").observed_mask)
    expected = float(
        np.mean(
            np.max(replicated[:, taken], axis=-1) >= np.max(observed[taken])
        )
    )
    assert p_values(report)["largest"] == pytest.approx(expected, abs=1e-12)
    assert p_values(report)["largest"] == pytest.approx(0.4670, abs=1e-9)

    unmasked = posterior_predictive_check(
        masked_line(None), predictive, source_posterior=posterior,
        discrepancies=(checks_module.largest,),
    )
    assert p_values(unmasked)["largest"] == pytest.approx(0.4375, abs=1e-9)


# --------------------------------------------- the report as an artifact


def test_the_report_points_at_the_result_it_judged_and_survives_the_codec(measured):
    """§0.2's ``subject_ref``, §0.3's lineage, and the round trip R1 requires.

    The subject is a PARENT as well as the subject: §0.3 retires an artifact
    through the inputs it was made from, so a report whose result is not in
    its lineage would outlive a change to that result.
    """
    posterior, predictive = measured["straight_line"]
    report = posterior_predictive_check(
        straight_line(), predictive, source_posterior=posterior
    )
    assert report.subject_ref.artifact_id == predictive.meta.artifact_id
    assert report.subject_ref.revision == predictive.meta.revision
    assert report.subject_ref.artifact_type is ArtifactKind.RESULT
    assert report.meta.parent_refs == (report.subject_ref,)
    assert report.meta.artifact_type is ArtifactKind.EVALUATION_REPORT
    assert report.meta.fingerprints == predictive.meta.fingerprints

    restored = canonical_loads(
        canonical_dumps(report), expected=EvaluationReport
    )
    assert restored == report
    assert len(restored.findings) == 5


def test_every_verdict_is_recomputable_from_the_findings_alone(measured):
    """§8 R3's gate 8: no sampler log in the loop.

    Each cell's finding carries ``(node, identity, p, tail_mass)`` and the
    declared tail as ``expected``, so the whole report's conclusion is
    ``all(tail_mass >= expected)`` -- the same expression the production
    predicate evaluates, not a restatement of it that agrees to a rounding.
    """
    for name, conclusion in (
        ("straight_line", Conclusion.PASS),
        ("curved_06", Conclusion.FAIL),
    ):
        posterior, predictive = measured[name]
        graph = (
            straight_line()
            if name == "straight_line"
            else probe().curved_line(curvature=0.6)
        )
        report = posterior_predictive_check(
            graph, predictive, source_posterior=posterior
        )
        recomputed = all(
            finding.observed[3] >= finding.expected for finding in report.findings
        )
        assert (Conclusion.PASS if recomputed else Conclusion.FAIL) is conclusion
        assert report.conclusion is conclusion
        for finding in report.findings:
            node, _identity, p_value, tail_mass = finding.observed
            assert node == "d"
            assert tail_mass == min(p_value, 1.0 - p_value)


@pytest.mark.parametrize(
    ("name", "applicability", "conclusion"),
    [
        ("straight_line", Applicability.APPLICABLE, Conclusion.PASS),
        ("curved_06", Applicability.APPLICABLE, Conclusion.FAIL),
    ],
)
def test_a_report_this_module_makes_is_one_a_gate_can_aggregate(
    measured, name, applicability, conclusion
):
    """The R1-frozen legal-pair table, exercised through the aggregator.

    Task 3 is the first task that produces an ``EvaluationReport``, so the
    pairs it emits have to be ones ``aggregate_gate`` counts rather than
    refuses.  Feeding them through a one-requirement gate is the check that
    the two axes were filled in correctly -- an INAPPLICABLE report claiming
    PASS would have been refused at construction, and an APPLICABLE one
    claiming the wrong verdict would show up as the wrong gate verdict here.
    """
    posterior, predictive = measured[name]
    graph = (
        straight_line() if name == "straight_line" else probe().curved_line(curvature=0.6)
    )
    report = posterior_predictive_check(graph, predictive, source_posterior=posterior)
    assert (report.applicability, report.conclusion) == (applicability, conclusion)

    requirement = ReportRequirement(name="posterior_predictive_check")
    definition = GateDefinition(
        name="model_checking", version=1, requirements=(requirement,)
    )
    result = aggregate_gate(
        definition,
        meta=new_artifact_meta(
            artifact_type=ArtifactKind.EVALUATION_REPORT,
            fingerprints=predictive.meta.fingerprints,
            producer=checks_module.PRODUCER,
            summary="one-requirement probe of the legal pairs",
        ),
        prerequisites_ready=True,
        inputs_current=True,
        slots=(
            ReportSlot(
                requirement=requirement,
                report=report,
                attempt_status=AttemptStatus.ATTEMPTED,
            ),
        ),
    )
    assert result.status is OperationalStatus.EVALUATED
    assert result.verdict is conclusion


def test_an_abstained_report_leaves_a_required_gate_undecided():
    """The third arm of the same table: ABSTAIN is not a quiet PASS."""
    graph = straight_line()
    posterior, predictive = checked(graph, draws=8)
    report = posterior_predictive_check(graph, predictive, source_posterior=posterior)
    requirement = ReportRequirement(name="posterior_predictive_check")
    definition = GateDefinition(
        name="model_checking", version=1, requirements=(requirement,)
    )
    result = aggregate_gate(
        definition,
        meta=new_artifact_meta(
            artifact_type=ArtifactKind.EVALUATION_REPORT,
            fingerprints=predictive.meta.fingerprints,
            producer=checks_module.PRODUCER,
            summary="one-requirement probe of the abstain arm",
        ),
        prerequisites_ready=True,
        inputs_current=True,
        slots=(
            ReportSlot(
                requirement=requirement,
                report=report,
                attempt_status=AttemptStatus.ATTEMPTED,
            ),
        ),
    )
    assert result.status is OperationalStatus.EVALUATED
    assert result.verdict is Conclusion.ABSTAIN


# ------------------------------------------------------- caller-side errors


def test_a_predictive_result_with_no_replicated_draws_is_inapplicable(measured):
    """Nothing to compare the data against; not a failure of the model."""
    posterior, predictive = measured["straight_line"]
    stripped = dataclasses.replace(predictive, replicated_draws=())
    assert isinstance(stripped, PredictiveResult)
    report = posterior_predictive_check(
        straight_line(), stripped, source_posterior=posterior
    )
    assert report.applicability is Applicability.INAPPLICABLE
    assert report.conclusion is Conclusion.ABSTAIN
    assert report.findings[0].code == "no_replicated_draws"


def test_the_wrong_source_posterior_is_refused_before_any_number(measured):
    """§0.3 weights the p-value with the SOURCE posterior's weights, so the
    result's own ``source_posterior_ref`` decides which posterior that is."""
    _posterior, predictive = measured["straight_line"]
    other = measured["radiometer"][0]
    with pytest.raises(TypeError, match="not the version this predictive"):
        posterior_predictive_check(
            straight_line(), predictive, source_posterior=other
        )


def test_a_draw_count_mismatch_is_refused_rather_than_broadcast(measured):
    """The draw axis is one-to-one (R2 §0.5): draw i against draw i."""
    posterior, predictive = measured["straight_line"]
    short = posterior_of(straight_line(), jax.random.key(1), draws=64)
    impersonating = dataclasses.replace(
        short,
        meta=dataclasses.replace(
            short.meta,
            artifact_id=posterior.meta.artifact_id,
            revision=posterior.meta.revision,
        ),
    )
    with pytest.raises(TypeError, match="draws and the source posterior has"):
        posterior_predictive_check(
            straight_line(), predictive, source_posterior=impersonating
        )


def test_a_check_with_no_discrepancy_is_a_caller_error(measured):
    """A PASS nobody measured is worse than a refusal."""
    posterior, predictive = measured["straight_line"]
    with pytest.raises(TypeError, match="at least one discrepancy"):
        posterior_predictive_check(
            straight_line(), predictive, source_posterior=posterior, discrepancies=()
        )


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda p, r: posterior_predictive_check(object(), r, source_posterior=p),
         "graph is a Graph"),
        (lambda p, r: posterior_predictive_check(straight_line(), p, source_posterior=p),
         "judges a PredictiveResult"),
        (lambda p, r: posterior_predictive_check(straight_line(), r, source_posterior=r),
         "is the PosteriorResult"),
        (lambda p, r: prior_predictive_check(object(), r),
         "graph is a Graph"),
        (lambda p, r: prior_predictive_check(straight_line(), r),
         "judges a SimulationResult"),
    ],
)
def test_the_wrong_kind_of_argument_is_a_type_error(measured, call, match):
    posterior, predictive = measured["straight_line"]
    with pytest.raises(TypeError, match=match):
        call(posterior, predictive)
