"""Running a compiled task: the same numbers, with provenance around them.

The whole of Task 7 is that the artifact path must be a WRAPPER. Every test in
the first half runs the old entry point and the new one on one graph, one key
and one budget, and compares them element by element -- ``rtol=0``, so a
resample, an extra ``jax.random.split`` or a differently ordered call would show
up as a difference rather than as a tolerance. The old objects are untouched:
:class:`~bayesmith.dispatch.execute.Posterior` and
:class:`~bayesmith.dispatch.execute.Estimate` come back exactly as they did.

The second half is what the wrapper adds. A Result names the plan it came from
by id AND revision; a run record says which backend actually ran, at what
dtype, on which device, under which budget, and how it stopped. Two things are
deliberately NOT invented: a graph drawn iid has no chain, so it carries no
``chain_shape`` and no r-hat, and a weighted sample keeps its weights rather
than being handed back as an unweighted one that happens to be wrong.
"""

from __future__ import annotations

import dataclasses
import math
import uuid

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from bayesmith import compile as compile_graph
from bayesmith import const, det, observe, optimize, sample, trace
from bayesmith.artifacts.base import (
    ApproximationClass,
    ArtifactKind,
    ArtifactRef,
    ComputeBudget,
    NamedArray,
    TargetFidelity,
    TerminationReason,
)
from bayesmith.artifacts.refusal import CAPABILITY_UNAVAILABLE_R1, Refusal
from bayesmith.artifacts.results import (
    DrawsPosterior,
    LogDensityAvailability,
    PointEstimateResult,
    PosteriorResult,
    PredictiveResult,
    SimulationResult,
    WeightedDrawsPosterior,
)
from bayesmith.artifacts.tasks import (
    Estimand,
    EvidenceTask,
    ParameterSource,
    PointEstimateTask,
    PosteriorTask,
    PredictiveTask,
    SimulationTask,
    new_task_meta,
)
from bayesmith.diagnose.map import map_estimate
from bayesmith.dispatch.task import (
    MAP_METHODS,
    compile_task,
    execute_task,
    plan_ref,
)
from bayesmith.graph.evaluate import evaluate
from bayesmith.optimize import fit
from tests.dispatch.test_task_protocol import model_ref
from tests.exact.models import (
    bilinear_pair,
    mixed_radiometer,
    non_gaussian_observed_node,
    radiometer,
    straight_line,
)

BUDGET = ComputeBudget(draws=8, warmup=8, chains=1)

#: Enough draws for PSIS to have a tail to fit. See the khat tests below.
WEIGHTED = ComputeBudget(draws=32, warmup=8, chains=1)


def posterior_task(**overrides) -> PosteriorTask:
    """A task whose fallback policy matches the runtime default.

    ``PosteriorTask.nuts_on_collapse`` defaults to True and
    :meth:`~bayesmith.dispatch.plan.InferencePlan.sample`'s keyword defaults to
    False -- the protocol's default is the caller's consent, the runtime's is
    the measured preference for annotating rather than substituting. The
    parity tests below pin the numbers, not that disagreement, so they ask for
    the runtime's default explicitly on both sides.
    """
    fields: dict = {
        "meta": new_task_meta(label="run"),
        "budget": BUDGET,
        "nuts_on_collapse": False,
    }
    fields.update(overrides)
    return PosteriorTask(**fields)


def point_task(**overrides) -> PointEstimateTask:
    fields: dict = {"meta": new_task_meta(label="point"), "estimand": Estimand.MAP}
    fields.update(overrides)
    return PointEstimateTask(**fields)


def planned_for(graph, task):
    planned = compile_task(graph, task, model_ref=model_ref())
    assert not isinstance(planned, Refusal), planned
    return planned


def drawn(result: PosteriorResult) -> dict:
    return {array.name: array.value for array in result.representation.draws}


def same(old, new) -> None:
    """Bitwise, through ``assert_allclose`` at zero tolerance.

    A tolerance here would hide exactly the failure this file exists to catch:
    an adapter that re-ran the sampler, split the key once more, or reordered
    the calls produces numbers that are close and not the same.
    """
    np.testing.assert_allclose(np.asarray(old), np.asarray(new), rtol=0.0, atol=0.0)


def details(result) -> dict:
    return dict(result.run.approximation.details)


# --------------------------------------------------------- 7.1 numerical parity


def test_pure_exact_draws_are_the_old_draws():
    """``straight_line`` is whole-graph exact with a constant sigma: iid GCR
    draws, no chain, and ESS is the draw count exactly."""
    graph = straight_line()
    key = jax.random.key(11)
    old = compile_graph(graph).sample(
        key, num_samples=8, num_warmup=8, num_chains=1, nuts_on_collapse=False
    )
    result = execute_task(planned_for(graph, posterior_task()), key=key)

    assert isinstance(result, PosteriorResult)
    assert isinstance(result.representation, DrawsPosterior)
    assert result.representation.method == old.method == "gcr"
    assert set(drawn(result)) == set(old.samples)
    for name, draws in old.samples.items():
        same(draws, drawn(result)[name])
    assert result.representation.chain_shape is None
    assert details(result)["effective_sample_size"] == old.ess


def test_a_weighted_sample_keeps_its_weights_and_its_diagnostics():
    """``radiometer``'s sigma tracks its own prediction, so the exact draws are
    a proposal corrected by self-normalised weights. Dropping the weights
    would leave a WRONG unweighted sample rather than a lossy one, which is
    why §0.5 gives the weighted case its own arm.

    Thirty-two draws rather than eight, measured: PSIS has no tail to fit at
    eight and answers ``inf``, which the next test is about -- so the khat
    compared here is a number both paths actually took.
    """
    graph = radiometer()
    key = jax.random.key(5)
    old = compile_graph(graph).sample(
        key, num_samples=32, num_warmup=8, num_chains=1, nuts_on_collapse=False
    )
    result = execute_task(planned_for(graph, posterior_task(budget=WEIGHTED)), key=key)

    assert isinstance(result.representation, WeightedDrawsPosterior)
    assert old.log_weights is not None
    assert math.isfinite(old.khat)
    for name, draws in old.samples.items():
        same(draws, drawn(result)[name])
    same(old.log_weights, result.representation.log_weights.value)
    assert result.representation.ess == old.ess
    assert result.representation.khat == old.khat
    assert result.representation.unreliable == old.unreliable
    assert result.representation.method == old.method == "gcr+snis"


def test_a_diagnostic_that_could_not_be_fitted_is_absent_rather_than_infinite():
    """Eight draws leave PSIS no tail, and the old path reports ``inf`` for
    khat. §0.5 keeps "not computed" as ``None`` and refuses a non-finite
    number as a measurement, so the projection drops it -- and drops nothing
    else: ``unreliable`` is a stored verdict taken at the run's own threshold
    and comes through exactly as it was."""
    graph = radiometer()
    key = jax.random.key(5)
    old = compile_graph(graph).sample(
        key, num_samples=8, num_warmup=8, num_chains=1, nuts_on_collapse=False
    )
    result = execute_task(planned_for(graph, posterior_task()), key=key)

    assert not math.isfinite(old.khat)
    assert result.representation.khat is None
    assert result.representation.unreliable is old.unreliable is True
    assert result.representation.ess == old.ess


def test_the_two_places_the_effective_sample_size_appears_cannot_drift():
    """§0.5 gives the weighted arm an ``ess`` field and the unweighted arms
    none, so the run record carries the number for every path and the weighted
    representation carries it as well. Two homes for one measurement is the
    defect this repository has spent the most time repairing; the protocol
    fixes both fields, so what holds them together is this test."""
    graph = radiometer()
    result = execute_task(
        planned_for(graph, posterior_task(budget=WEIGHTED)), key=jax.random.key(5)
    )
    assert details(result)["effective_sample_size"] == result.representation.ess


def test_a_sampled_graph_is_the_old_chain():
    """``bilinear_pair`` declares an affinity that is false, so the classifier
    routes the whole graph to NUTS -- the third of §6.4's shapes."""
    graph = bilinear_pair()
    key = jax.random.key(3)
    old = compile_graph(graph).sample(
        key, num_samples=8, num_warmup=8, num_chains=1, nuts_on_collapse=False
    )
    result = execute_task(planned_for(graph, posterior_task()), key=key)

    assert result.representation.method == old.method == "nuts"
    for name, draws in old.samples.items():
        same(draws, drawn(result)[name])
    assert result.representation.chain_shape == (1, 8)
    assert details(result)["effective_sample_size"] == old.ess


def test_a_posterior_mean_is_the_old_estimate():
    graph = straight_line()
    old = compile_graph(graph).estimate()
    task = point_task(estimand=Estimand.POSTERIOR_MEAN)
    result = execute_task(planned_for(graph, task))

    assert isinstance(result, PointEstimateResult)
    assert result.estimand is Estimand.POSTERIOR_MEAN
    values = {array.name: array.value for array in result.values}
    for name, value in old.values.items():
        same(value, values[name])
    same(old.residual, result.residual)
    assert result.iterations == int(old.iterations)
    assert result.objective is None
    assert not result.local_only


def test_a_newton_map_is_the_old_map_estimate():
    """The graph is built inside the x64 block, not merely the call: ``const``
    and ``observe`` capture their arrays at trace time, and
    :func:`~bayesmith.diagnose.map.map_estimate` refuses a float32 starting
    point rather than pretending the wider call widened the model."""
    with jax.enable_x64(True):
        graph = straight_line()
        old = map_estimate(graph)
        result = execute_task(planned_for(graph, point_task()))

    values = {array.name: array.value for array in result.values}
    for name, value in old.point.items():
        same(value, values[name])
    assert result.objective == old.objective
    assert result.gradient_norm == old.gradient_norm
    assert result.iterations == old.steps
    assert result.local_only


def test_a_descent_map_is_the_old_fit():
    graph = straight_line()
    task = point_task(
        optimizer_options=(("method", "adam"),),
        budget=ComputeBudget(max_iterations=25),
    )
    old = fit(graph, method="adam", steps=25)
    result = execute_task(planned_for(graph, task))

    values = {array.name: array.value for array in result.values}
    for name, value in old.values.items():
        same(value, values[name])
    same(old.objective, result.objective)
    assert result.iterations == 25
    assert result.local_only


def test_the_map_methods_are_the_optimisers_own_plus_the_newton_seam():
    """``MAP_METHODS`` names three routes and two of them belong to
    :mod:`bayesmith.optimize`. A second copy of a vocabulary is what goes
    stale, so the copy is held to its source here rather than trusted."""
    assert MAP_METHODS == ("newton", *optimize._METHODS)


def test_an_unknown_optimiser_is_refused_before_anything_runs():
    refusal = compile_task(
        straight_line(),
        point_task(optimizer_options=(("method", "bfgs"),)),
        model_ref=model_ref(),
    )
    assert isinstance(refusal, Refusal)
    assert refusal.failed_premise == "task_options_recognised"
    assert refusal.grounds[0].observed == "bfgs"
    assert refusal.grounds[0].expected == MAP_METHODS


# ------------------------------------------------------------- 7.2 provenance


def test_a_result_names_the_plan_it_came_from_by_id_and_revision():
    """A reference carrying only an id would be satisfied by a later,
    invalidated revision of the same plan -- the impersonation §0.2 rules
    out."""
    planned = planned_for(straight_line(), posterior_task())
    result = execute_task(planned, key=jax.random.key(2))
    reference = plan_ref(planned.record)

    assert result.run.plan_ref == reference
    assert result.meta.parent_refs == (reference,)
    assert reference.artifact_id == planned.record.meta.artifact_id
    assert reference.revision == planned.record.meta.revision
    assert result.meta.artifact_type is ArtifactKind.RESULT


def test_the_run_record_says_what_actually_ran():
    planned = planned_for(straight_line(), posterior_task())
    key = jax.random.key(7)
    result = execute_task(planned, key=key)
    run = result.run

    assert run.backend.name == "bayesmith"
    assert run.seed is not None
    assert run.seed.seed == 7
    assert run.seed.key_algorithm == str(jax.random.key_impl(key))
    assert run.dtype == str(np.asarray(next(iter(drawn(result).values()))).dtype)
    assert run.devices
    assert dict(run.jax_config)["jax_enable_x64"] is False
    assert run.budget == ComputeBudget(draws=8, warmup=8, chains=1)
    assert run.timing.wall_clock_seconds >= 0.0
    assert run.approximation.representation_class is ApproximationClass.MONTE_CARLO
    assert run.approximation.target_fidelity is TargetFidelity.EXACT
    assert run.fingerprints.environment is not None
    assert run.fingerprints.compilation == planned.record.meta.fingerprints.compilation


def test_the_backend_that_ran_is_recorded_apart_from_the_one_requested():
    """A task asks for ``auto`` and a run record may not answer ``auto``: the
    graph that goes to NUTS is run by numpyro, and the record is where that
    is written down."""
    planned = planned_for(bilinear_pair(), posterior_task())
    result = execute_task(planned, key=jax.random.key(4))

    assert planned.task.backend == "auto"
    assert planned.record.backend == "auto"
    assert result.run.backend.name == "numpyro"
    assert result.run.backend.version == numpyro.__version__
    assert details(result)["requested_backend"] == "auto"
    assert details(result)["method"] == "nuts"
    assert details(result)["planned_method"] == "nuts"


def test_a_run_that_took_no_key_records_no_seed():
    """A point estimate splits nothing, so there is no entropy to record and
    ``None`` is the honest answer rather than a zero that reads as a seed."""
    task = point_task(estimand=Estimand.POSTERIOR_MEAN)
    result = execute_task(planned_for(straight_line(), task))
    assert result.run.seed is None
    assert result.run.termination.reason is TerminationReason.CONVERGED
    assert result.run.termination.iterations == result.iterations


def test_iid_draws_carry_no_chain_and_no_invented_r_hat():
    """``gcr`` draws independently, so split r-hat has no referent at all.
    Reporting a number for it would invent one; the field stays empty and the
    effective sample size, which is real, is what a caller reads."""
    result = execute_task(
        planned_for(straight_line(), posterior_task()), key=jax.random.key(2)
    )
    assert result.representation.chain_shape is None
    codes = {warning.code for warning in result.run.warnings}
    assert not {code for code in codes if code.startswith("chain_")}


def test_a_short_chain_is_reported_as_uncertified_rather_than_as_converged():
    """Eight draws cannot certify convergence -- ``CHAIN_ESS_FLOOR`` is 100 --
    and the task asked for convergence by default. Both halves are recorded:
    the sites that could not be certified, and that the task's requirement was
    not met."""
    result = execute_task(
        planned_for(bilinear_pair(), posterior_task()), key=jax.random.key(4)
    )
    codes = {warning.code for warning in result.run.warnings}
    assert "chain_not_converged" in codes
    assert "convergence_not_certified" in codes
    assert result.run.termination.reason is TerminationReason.COMPLETED
    assert all(warning.message for warning in result.run.warnings)


def test_the_fallback_policy_the_task_asked_for_is_the_one_recorded():
    planned = planned_for(radiometer(), posterior_task(nuts_on_collapse=True))
    assert planned.record.fallback_policy == "nuts_on_collapse"
    other = planned_for(radiometer(), posterior_task(nuts_on_collapse=False))
    assert other.record.fallback_policy == "annotate_on_collapse"


def test_a_diagonal_exact_posterior_records_pointwise_density():
    """The exact diagonal route can replay the observations, so it reports
    POINTWISE rather than NONE (§4.1)."""
    result = execute_task(
        planned_for(straight_line(), posterior_task()), key=jax.random.key(2)
    )
    assert result.pointwise_log_likelihood is not None
    assert result.log_density_availability is LogDensityAvailability.POINTWISE
    assert result.predictive_ready
    assert result.pointwise_log_likelihood.dims[0] == "draw"
    assert result.pointwise_log_likelihood.value.shape[0] == 8
    assert result.eliminated_latents == ()


def test_a_nuts_posterior_records_pointwise_density():
    """A Gaussian observation sampled by NUTS still has a pointwise likelihood
    to replay -- pointwise does not require the exact route (§4.1)."""
    result = execute_task(
        planned_for(bilinear_pair(), posterior_task()), key=jax.random.key(4)
    )
    assert result.representation.method == "nuts"
    assert result.pointwise_log_likelihood is not None
    assert result.log_density_availability is LogDensityAvailability.POINTWISE
    assert result.predictive_ready


def test_a_correlated_posterior_abstains_from_pointwise_density():
    """A correlated (CirculantNormal) observation has no diagonal loc/scale to
    replay, so the result ABSTAINs rather than fabricating one (§4.1)."""
    result = execute_task(
        planned_for(_correlated_graph(), posterior_task()), key=jax.random.key(2)
    )
    assert result.pointwise_log_likelihood is None
    assert result.log_density_availability is LogDensityAvailability.NONE
    assert not result.predictive_ready


def test_a_non_gaussian_posterior_abstains_from_pointwise_density():
    """A Student-t observation routed to NUTS has no sigma to replay, so the
    result ABSTAINs rather than fabricating one (§4.1)."""
    result = execute_task(
        planned_for(non_gaussian_observed_node(), posterior_task()),
        key=jax.random.key(2),
    )
    assert result.representation.method == "nuts"
    assert result.pointwise_log_likelihood is None
    assert result.log_density_availability is LogDensityAvailability.NONE
    assert not result.predictive_ready


# ------------------------------------------------------------ 7.4 the refusals


def test_a_posterior_run_without_a_key_is_a_programming_error():
    """Not a Refusal: a missing key is a caller mistake, and §0.6 keeps those
    as exceptions rather than dressing them as verdicts about a method."""
    planned = planned_for(straight_line(), posterior_task())
    with pytest.raises(TypeError, match="key"):
        execute_task(planned, key=None)


def test_a_hand_built_task_for_a_capability_r1_lacks_is_refused_defensively():
    """``compile_task`` never produces one of these; a caller who assembles a
    PlannedTask by hand gets the same verdict rather than an execution that
    half-works."""
    planned = planned_for(straight_line(), posterior_task())
    forged = dataclasses.replace(planned, task=EvidenceTask(meta=new_task_meta()))
    refusal = execute_task(forged)

    assert isinstance(refusal, Refusal)
    assert refusal.failed_premise == CAPABILITY_UNAVAILABLE_R1
    assert refusal.meta.artifact_type is ArtifactKind.RESULT
    assert refusal.remedies


def test_a_convergence_failure_is_raised_rather_than_refused(monkeypatch):
    """``ConvergenceError`` is a real execution failure -- the reweighting did
    not reach a fixed point -- and a workflow marks it ERROR. Turning it into
    a Refusal would file a broken run under "this method does not apply".

    The failure is injected at the seam rather than provoked by a fixture,
    because what is under test is which of the two kinds of bad news the
    adapter produces, not the arithmetic that would produce it.
    """
    from bayesmith.errors import ConvergenceError

    planned = planned_for(
        radiometer(), point_task(estimand=Estimand.POSTERIOR_MEAN)
    )

    def refuse(*args, **kwargs):
        raise ConvergenceError("the GLS reweighting did not reach a fixed point")

    monkeypatch.setattr(type(planned.runtime_plan), "estimate", refuse)
    with pytest.raises(ConvergenceError):
        execute_task(planned)


def test_execute_task_refuses_something_that_is_not_a_planned_task():
    with pytest.raises(TypeError, match="PlannedTask"):
        execute_task(object())


# ------------------------------------------- 7.5 the old API, entirely unmoved


def test_the_old_entry_points_return_the_old_types():
    """The red line of this session, as an assertion rather than as a habit:
    nothing above changed what ``sample`` and ``estimate`` hand back."""
    graph = straight_line()
    plan = compile_graph(graph)
    posterior = plan.sample(jax.random.key(1), num_samples=8, num_warmup=8)
    estimate = plan.estimate()

    assert type(posterior).__name__ == "Posterior"
    assert type(estimate).__name__ == "Estimate"
    assert posterior.log_weights is None
    assert math.isfinite(float(estimate.residual))
    assert isinstance(dict(posterior.samples), dict)
    assert uuid.UUID(planned_for(graph, posterior_task()).record.task_id).version == 4


def test_a_swept_graph_reports_a_chain_even_though_its_method_names_a_solve():
    """The trap ``_ran_a_chain`` exists for.

    The mixed path reports the EXACT BLOCK's method, so a plan whose exact
    block is solved by ``gcr`` comes back labelled with that even though an
    HMCGibbs chain is what produced the draws -- while a whole-graph ``gcr``
    plan carries the same label and draws iid. Reading the label would file a
    chain's draws as independent ones and lose the structure every chain
    diagnostic is about, so the plan is what is read instead.
    """
    graph = mixed_radiometer()
    key = jax.random.key(6)
    old = compile_graph(graph).sample(
        key, num_samples=8, num_warmup=8, num_chains=1, nuts_on_collapse=False
    )
    result = execute_task(planned_for(graph, posterior_task()), key=key)

    assert compile_graph(graph).sampled is not None, "this fixture must be mixed"
    for name, draws in old.samples.items():
        same(draws, drawn(result)[name])
    assert result.representation.method == old.method
    assert result.representation.chain_shape == (1, 8)
    assert result.run.backend.name == "numpyro"
    assert details(result)["planned_method"] == old.method


def test_a_mixed_graph_still_sweeps_through_the_old_path():
    """The mixed shape is the one this session must not disturb, so it is run
    through the OLD entry point here and compared against the new one's
    refusal to take a posterior mean of it (which is a compile-time verdict,
    not a change to the sweep)."""
    graph = mixed_radiometer()
    posterior = compile_graph(graph).sample(
        jax.random.key(6), num_samples=8, num_warmup=8, num_chains=1
    )
    assert set(posterior.samples) == set(graph.latents)
    refusal = compile_task(
        graph, point_task(estimand=Estimand.POSTERIOR_MEAN), model_ref=model_ref()
    )
    assert isinstance(refusal, Refusal)


# ------------------------------------------------------------- predictive seam


def _correlated_graph(size=8, weight=2.0, decay=0.4):
    """A graph whose observed node declares correlated (CirculantNormal) noise."""
    lag = np.minimum(np.arange(size), size - np.arange(size))
    kernel = jnp.asarray(1.0 * decay**lag + 0.5)
    x = jnp.linspace(1.0, 4.0, size)
    data = weight * x

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 5.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.CirculantNormal(m, kernel), mu,
                depends_on_prediction=False, obs=data)

    return trace(model)


def _source_ref(source):
    return ArtifactRef(
        artifact_id=source.meta.artifact_id,
        revision=source.meta.revision,
        artifact_type=ArtifactKind.RESULT,
    )


def _predictive_task(source, **overrides):
    fields = {
        "meta": new_task_meta(label="ppc"),
        "source_posterior_ref": _source_ref(source),
        "conditioned_sites": ("d",),
        "replicated_sites": ("d",),
        "latent_sites": ("w",),
    }
    fields.update(overrides)
    return PredictiveTask(**fields)


def test_a_predictive_task_executes_into_a_predictive_result():
    graph = straight_line()
    posterior = execute_task(planned_for(graph, posterior_task()), key=jax.random.key(2))
    task = _predictive_task(posterior)
    planned = planned_for(graph, task)
    result = execute_task(planned, key=jax.random.key(3), source_posterior=posterior)

    assert isinstance(result, PredictiveResult)
    assert result.source_posterior_ref == task.source_posterior_ref
    latent = {array.name: array for array in result.latent_draws}
    replicated = {array.name: array for array in result.replicated_draws}
    assert set(latent) == {"w"}
    assert set(replicated) == {"d"}
    assert latent["w"].value.shape[0] == replicated["d"].value.shape[0] == 8
    observed = np.asarray(graph.node("d").observed)
    assert not bool(np.any(replicated["d"].value == observed))
    assert result.pointwise_log_density is not None
    assert result.pointwise_log_density.value.shape[0] == 8


def test_a_predictive_task_refuses_a_mismatched_source_posterior():
    posterior = execute_task(
        planned_for(straight_line(), posterior_task()), key=jax.random.key(2)
    )
    other = straight_line(weight=3.0)  # same structure, different data
    planned = planned_for(other, _predictive_task(posterior))
    result = execute_task(planned, key=jax.random.key(3), source_posterior=posterior)

    assert isinstance(result, Refusal)
    assert result.failed_premise == "posterior_data_mismatch"
    assert result.grounds and result.remedies


def test_a_predictive_task_refuses_a_correlated_observation():
    graph = _correlated_graph()
    posterior = execute_task(planned_for(graph, posterior_task()), key=jax.random.key(2))
    planned = planned_for(graph, _predictive_task(posterior))
    result = execute_task(planned, key=jax.random.key(3), source_posterior=posterior)

    assert isinstance(result, Refusal)
    assert result.failed_premise == "predictive_noise_unsupported"
    assert result.grounds and result.remedies



# ------------------------------------------------------------ simulation seam


def _simulation_task(source, **overrides):
    fields = {
        "meta": new_task_meta(label="forward"),
        "parameter_source": source,
        "latent_sites": ("w",),
        "observed_sites": ("d",),
        "budget": ComputeBudget(draws=64),
    }
    fields.update(overrides)
    return SimulationTask(**fields)


def _drawn(result) -> dict:
    return {array.name: array for array in result.latent_draws} | {
        array.name: array for array in result.observation_draws
    }


def test_a_prior_simulation_task_executes_into_a_simulation_result():
    """R3 §0.7: the fifth task kind is answered, and the projection is
    mechanical.

    ``latent_draws`` and ``observation_draws`` share the draw axis -- the
    artifact refuses them otherwise -- so a consumer that pairs the i-th
    parameter with the i-th dataset is reading a guarantee rather than a
    convention. That pairing is the whole of what SBC needs from this result.
    """
    graph = straight_line()
    task = _simulation_task(ParameterSource.prior())
    planned = planned_for(graph, task)
    result = execute_task(planned, key=jax.random.key(5))

    assert isinstance(result, SimulationResult)
    assert result.parameter_source == ParameterSource.prior()
    arrays = _drawn(result)
    assert set(arrays) == {"w", "d"}
    assert arrays["w"].value.shape == (64,)
    assert arrays["d"].value.shape == (64, 8)
    assert arrays["d"].dims == ("draw", "d_dim0")
    assert result.run.budget.draws == 64
    assert result.run.approximation.representation_class is (
        ApproximationClass.MONTE_CARLO
    )
    assert result.run.approximation.target_fidelity is TargetFidelity.EXACT
    assert result.run.termination.reason is TerminationReason.COMPLETED
    assert result.run.seed is not None
    assert result.meta.parent_refs == (plan_ref(planned.record),)
    assert result.run.plan_ref == plan_ref(planned.record)
    assert dict(result.run.approximation.details)["parameter_source"] == "prior"
    assert result.parameters == (), "the prior fixes nothing to record"
    assert result.run.backend.name == "bayesmith", "no numpyro kernel ran"



def test_a_prior_simulation_draws_the_observations_it_did_not_condition_on():
    """The observed node is DRAWN, not carried: a prior simulation that handed
    back the conditioning data as its ``observation_draws`` would be a
    convincing-looking bank with no information in it, and SBC built on it
    would report perfect calibration for any sampler at all."""
    graph = straight_line()
    result = execute_task(
        planned_for(graph, _simulation_task(ParameterSource.prior())),
        key=jax.random.key(5),
    )
    observed = np.asarray(graph.node("d").observed)
    assert not bool(np.any(_drawn(result)["d"].value == observed))


def test_a_plated_prior_simulation_carries_the_plate_axis():
    """§0.7's plate expansion, seen through the task rather than the primitive:
    the ``obs`` plate names the second axis and there are six of it, not one
    value repeated."""
    from tests.exact.models import plated_latent

    graph = plated_latent()
    task = _simulation_task(
        ParameterSource.prior(),
        latent_sites=("z",),
        observed_sites=("d",),
        budget=ComputeBudget(draws=16),
    )
    result = execute_task(planned_for(graph, task), key=jax.random.key(7))

    arrays = _drawn(result)
    assert arrays["z"].value.shape == (16, 6)
    assert arrays["z"].dims == ("draw", "obs")
    assert len(np.unique(arrays["z"].value[0])) == 6


def test_a_fixed_simulation_centres_on_the_locs_evaluate_computes():
    """The FIXED arm carries the parameter setting it was given in
    ``parameters`` and draws observations around it.

    The band is the estimator's (``sigma / sqrt(n)``, five of them), not a
    measured tolerance; sigma is ``straight_line``'s own 0.5. Measured
    occupancy at this seed: worst 1.684 sigma of the five.
    """
    graph = straight_line()
    fixed = ParameterSource.fixed(
        (NamedArray(name="w", value=np.asarray(2.5), dims=()),)
    )
    task = _simulation_task(fixed, budget=ComputeBudget(draws=2000))
    result = execute_task(planned_for(graph, task), key=jax.random.key(8))

    assert isinstance(result, SimulationResult)
    assert [array.name for array in result.parameters] == ["w"]
    arrays = _drawn(result)
    assert set(arrays) == {"d"}, "a fixed parameter is not a drawn latent"
    loc = np.asarray(evaluate(graph, {"w": jnp.asarray(2.5)})["mu"])
    mean = arrays["d"].value.mean(axis=0)
    assert np.all(np.abs(mean - loc) <= 5.0 * 0.5 / np.sqrt(2000))


def test_a_posterior_source_simulation_is_the_predictive_replication_bit_for_bit():
    """§0.7's sharpest requirement, and the reason it is worth a test of its
    own.

    A ``SimulationTask`` reading ``POSTERIOR_RESULT`` and a ``PredictiveTask``
    reading the same posterior are asking the same question through two
    schemas. If they answered with two different sets of numbers -- because
    one split the key once more, or reached for its own forward model -- then
    "the posterior predictive" would mean two things in one package, and no
    test comparing either to a reference could tell you which one was wrong.

    ``rtol=0``: close is not the claim.
    """
    graph = straight_line()
    posterior = execute_task(
        planned_for(graph, posterior_task()), key=jax.random.key(2)
    )
    source_ref = _source_ref(posterior)
    key = jax.random.key(3)

    predictive = execute_task(
        planned_for(graph, _predictive_task(posterior)),
        key=key,
        source_posterior=posterior,
    )
    simulation = execute_task(
        planned_for(
            graph,
            _simulation_task(ParameterSource.from_posterior_result(source_ref)),
        ),
        key=key,
        source_posterior=posterior,
    )

    assert isinstance(simulation, SimulationResult)
    replicated = {array.name: array.value for array in predictive.replicated_draws}
    same(replicated["d"], _drawn(simulation)["d"].value)
    # ... and the latents it was pushed forward from are the same draws too.
    same(drawn(posterior)["w"], _drawn(simulation)["w"].value)
    # n came from the SOURCE, not from the task's budget (which says 64 here).
    # §0.7 says budget.draws records n, and for this arm n is however many
    # draws the posterior has -- a run record states what was spent, and
    # honouring the task's number instead would state a count nothing produced.
    assert simulation.run.budget.draws == BUDGET.draws == 8
    assert simulation.parameter_source.posterior_ref == source_ref
    assert simulation.run.backend.name == predictive.run.backend.name


def test_a_posterior_source_simulation_refuses_a_posterior_of_other_data():
    """The same premise the predictive arm checks, checked here too: a
    posterior drawn from different data pushed forward against this graph is a
    typed refusal, not a bank of plausible-looking numbers."""
    posterior = execute_task(
        planned_for(straight_line(), posterior_task()), key=jax.random.key(2)
    )
    other = straight_line(weight=3.0)  # same structure, different data
    task = _simulation_task(
        ParameterSource.from_posterior_result(_source_ref(posterior))
    )
    result = execute_task(
        planned_for(other, task), key=jax.random.key(3), source_posterior=posterior
    )

    assert isinstance(result, Refusal)
    assert result.failed_premise == "posterior_data_mismatch"
    assert result.grounds and result.remedies


def test_a_simulation_task_with_no_draw_count_is_a_type_error():
    """``budget.draws`` is where n lives (§0.7), and this module's docstring
    says nothing here decides a number.

    Inventing a default draw count would be a numerical decision taken inside
    an adapter that is meant to take none -- and unlike a sampler's draw
    count, there is no runtime seam here whose signature owns the default to
    be read off. So the field is required and the error names it.
    """
    task = _simulation_task(ParameterSource.prior(), budget=ComputeBudget())
    with pytest.raises(TypeError, match="budget"):
        execute_task(planned_for(straight_line(), task), key=jax.random.key(5))


def test_a_simulation_task_refuses_a_correlated_observation():
    """§0.4's coverage domain, reaching the fifth task: the fixed and
    posterior arms share ``replicated_draws``' diagonal walk, so they refuse
    where it refuses."""
    graph = _correlated_graph()
    fixed = ParameterSource.fixed(
        (NamedArray(name="w", value=np.asarray(2.0), dims=()),)
    )
    task = _simulation_task(fixed, budget=ComputeBudget(draws=4))
    result = execute_task(planned_for(graph, task), key=jax.random.key(3))

    assert isinstance(result, Refusal)
    assert result.failed_premise == "predictive_noise_unsupported"
    assert result.grounds and result.remedies


def test_a_fixed_simulation_that_names_only_latents_says_what_is_empty():
    """A fixed source draws no latents, so a task naming only ``latent_sites``
    is asking for draws of the values it just fixed.

    Worth its own message because the artifact's validation would speak second
    and say something true but unhelpful -- the run record's dtype is computed
    from the produced arrays and there are none, so the failure would arrive as
    a numpy ``result_type`` complaint about an empty argument list.
    """
    fixed = ParameterSource.fixed(
        (NamedArray(name="w", value=np.asarray(2.5), dims=()),)
    )
    task = _simulation_task(fixed, observed_sites=(), budget=ComputeBudget(draws=4))
    with pytest.raises(TypeError, match="no draws"):
        execute_task(planned_for(straight_line(), task), key=jax.random.key(8))


def test_a_simulation_result_this_module_produced_round_trips():
    """``tests/artifacts`` round-trips SimulationResults it built by hand;
    this round-trips one the EXECUTOR built.

    The two are not the same claim. A hand-built fixture chooses its own
    dims, its own details tuple and its own dtypes, so it can round-trip
    happily while the real projection puts something the codec refuses --
    a jax array where numpy was expected, a detail value that is not
    canonical -- into the same schema. An SBC harness stores these banks, so
    the one that matters is the one this module writes.
    """
    from bayesmith.artifacts._codec import canonical_dumps, canonical_loads

    result = execute_task(
        planned_for(
            straight_line(),
            _simulation_task(ParameterSource.prior(), budget=ComputeBudget(draws=8)),
        ),
        key=jax.random.key(5),
    )
    assert canonical_loads(canonical_dumps(result), expected=SimulationResult) == result
