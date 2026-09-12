"""Policies are executable contracts, tested against real graph densities."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import observe, plate, sample, trace
from bayesmith.artifacts import (
    DiagnosticPolicy,
    InitializationPolicy,
    NamedArray,
    PosteriorTask,
    StoppingPolicy,
    new_task_meta,
    task_fingerprint,
)
from bayesmith.dispatch.initialization import complete_initial_values


def positive_vector():
    def model():
        p = sample("p", lambda: dist.LogNormal(jnp.zeros(3), 0.8))
        observe("y", lambda x: dist.Normal(x, 0.3), p, obs=jnp.array([1.0, 2.0, 3.0]))

    return trace(model)


def test_policy_is_part_of_task_identity():
    task = PosteriorTask(meta=new_task_meta())
    assert task.diagnostics == DiagnosticPolicy()
    assert task.initialization == InitializationPolicy()
    changed = dataclasses.replace(task, stopping=StoppingPolicy(mode="checkpoints"))
    assert task_fingerprint(task) != task_fingerprint(changed)


@pytest.mark.parametrize(
    "kwargs",
    [{"batch_size": 0}, {"min_draws": 3}, {"consecutive": 0}, {"rhat_max": 0.9}],
)
def test_invalid_stopping_policy_refuses(kwargs):
    with pytest.raises((TypeError, ValueError)):
        StoppingPolicy(**kwargs)


def test_partial_initial_values_keep_supplied_entries_per_chain():
    graph = positive_vector()
    policy = InitializationPolicy(
        values=(
            NamedArray(
                "p",
                np.array([[1.7, 0.0, 0.0], [2.3, 0.0, 0.0]]),
                ("chain", "parameter"),
            ),
        ),
        masks=(
            NamedArray(
                "p", np.array([[True, False, False]] * 2), ("chain", "parameter")
            ),
        ),
    )
    values, attempts = complete_initial_values(graph, jax.random.key(61), policy, 2)
    again, _ = complete_initial_values(graph, jax.random.key(61), policy, 2)
    assert np.array_equal(np.asarray(values[0]["p"]), np.asarray(again[0]["p"]))
    assert float(values[0]["p"][0]) == pytest.approx(1.7)
    assert float(values[1]["p"][0]) == pytest.approx(2.3)
    assert all(np.all(np.asarray(v["p"]) > 0) for v in values)
    assert not np.array_equal(
        np.asarray(values[0]["p"])[1:], np.asarray(values[1]["p"])[1:]
    )
    assert len(attempts) == 2


def test_invalid_supplied_value_is_not_clipped_or_replaced():
    policy = InitializationPolicy(
        values=(NamedArray("p", np.array([-1.0, 2.0, 3.0]), ("p",)),)
    )
    with pytest.raises(ValueError, match="p.*support"):
        complete_initial_values(positive_vector(), jax.random.key(1), policy, 1)


def test_mask_is_explicit_and_same_shape():
    with pytest.raises(ValueError, match="mask"):
        InitializationPolicy(
            values=(NamedArray("p", np.zeros(3), ("p",)),),
            masks=(NamedArray("p", np.ones(2, dtype=bool), ("p",)),),
        )


@pytest.mark.parametrize("event_size", [2, 3])
def test_initialization_preserves_event_axes_inside_a_plate(event_size):
    def model():
        group = plate("group", 3)
        sample(
            "x",
            lambda: dist.MultivariateNormal(jnp.zeros(event_size), jnp.eye(event_size)),
            plate=group,
        )

    graph = trace(model)
    points, attempts = complete_initial_values(
        graph, jax.random.key(62), InitializationPolicy(), 2
    )
    assert attempts == (1, 1)
    assert all(point["x"].shape == (3, event_size) for point in points)
    assert not np.array_equal(points[0]["x"][0], points[0]["x"][1])


def test_checkpoint_batches_are_one_chain_in_chain_major_order():
    from numpyro.infer import MCMC, NUTS

    from bayesmith.bridge.numpyro_bridge import to_numpyro
    from bayesmith.dispatch.sampling import SamplingControl, run_mcmc

    graph = positive_vector()
    key = jax.random.key(91)

    def sampler():
        return MCMC(
            NUTS(to_numpyro(graph)),
            num_warmup=20,
            num_samples=24,
            num_chains=2,
            chain_method="sequential",
            progress_bar=False,
        )

    whole = sampler()
    whole.run(key)
    chunked = sampler()
    control = SamplingControl(
        stopping=StoppingPolicy(
            mode="checkpoints", min_draws=8, batch_size=8, ess_min=1e6
        )
    )
    actual = run_mcmc(chunked, key, graph, control)
    # State/RNG continuity gives identical draws, not merely matching moments.
    np.testing.assert_array_equal(actual["p"], whole.get_samples()["p"])
    assert control.details["stop_reason"] == "draw_cap"
    assert control.details["draws_per_chain"] == 24
    assert len(control.checkpoints) == 3
    assert int(np.asarray(chunked.last_state.i)[0]) == 44


def test_explicit_initialization_reaches_gibbs_state():
    from bayesmith import compile
    from bayesmith.dispatch.sampling import SamplingControl, prepare_initial_params
    from bayesmith.exact.gibbs import assemble
    from tests.exact.models import mixed_radiometer

    graph = mixed_radiometer()
    plan = compile(graph)
    values = {name: np.asarray(0.375 + i) for i, name in enumerate(graph.latents)}
    policy = InitializationPolicy(
        values=tuple(NamedArray(n, v, ()) for n, v in values.items())
    )
    control = SamplingControl(initialization=policy)
    mcmc = assemble(
        graph, plan.exact.latents, tol=plan.exact.tol, num_warmup=8, num_samples=8
    )
    params = prepare_initial_params(
        mcmc, graph, jax.random.key(4), control, plan.exact.latents
    )
    state = mcmc.sampler.init(jax.random.key(5), 8, dict(params), (), {})
    for name in plan.exact.latents:
        np.testing.assert_array_equal(state.z[name], values[name])


def test_preflight_checks_user_priors_and_records_budget_skips():
    from bayesmith.dispatch.task import compile_task
    from tests.dispatch.test_task_protocol import model_ref
    from tests.exact.models import straight_line

    graph = straight_line()
    planned = compile_task(
        graph,
        PosteriorTask(
            meta=new_task_meta(), diagnostics=DiagnosticPolicy(max_parameters=0)
        ),
        model_ref=model_ref(),
    )
    findings = {f.code: f for f in planned.analysis.findings}
    assert findings["initial_point"].conclusion == "passed"
    assert findings["joint_geometry"].conclusion == "skipped_budget"
    assert findings["block_0_jeffreys"].conclusion == "skipped_budget"
    assert (
        dict(findings["block_0_prior"].measurements)["policy"]
        == "user_specified_unchanged"
    )


def test_unmet_required_diagnostic_and_unsupported_order_refuse_before_sampling():
    from bayesmith.artifacts import Refusal
    from bayesmith.dispatch.task import compile_task
    from tests.dispatch.test_task_protocol import model_ref
    from tests.exact.models import straight_line

    for options in (
        {
            "diagnostics": DiagnosticPolicy(
                max_parameters=0, required=("joint_geometry",)
            )
        },
        {"block_order": (("unknown",),)},
    ):
        result = compile_task(
            straight_line(),
            PosteriorTask(meta=new_task_meta(), **options),
            model_ref=model_ref(),
        )
        assert isinstance(result, Refusal)


def test_variance_information_prevents_false_nonidentification():
    from bayesmith.dispatch.preflight import _geometry, likelihood_information

    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe(
            "y",
            lambda v: dist.Normal(jnp.zeros(3), jnp.exp(v)),
            x,
            obs=jnp.array([0.2, -0.7, 1.3]),
        )

    with jax.enable_x64():
        graph = trace(model)
        matrix = likelihood_information(graph, ("x",), {"x": jnp.array(0.3)})
        # Fisher for 3 independent Normal(0, exp(x)) observations is 2*3.
        np.testing.assert_allclose(matrix, [[6.0]], rtol=1e-12)
        assert _geometry(matrix)[0] == 1


def test_task_checkpoint_records_actual_count_and_cap_not_convergence():
    from bayesmith.artifacts import ComputeBudget, TerminationReason
    from bayesmith.dispatch.task import compile_task, execute_task
    from tests.dispatch.test_task_protocol import model_ref

    task = PosteriorTask(
        meta=new_task_meta(),
        budget=ComputeBudget(draws=16, warmup=16, chains=2),
        stopping=StoppingPolicy(
            mode="checkpoints", min_draws=8, batch_size=8, ess_min=1e6
        ),
    )
    result = execute_task(
        compile_task(positive_vector(), task, model_ref=model_ref()),
        key=jax.random.key(7),
    )
    assert result.representation.chain_shape == (2, 16)
    assert result.run.termination.reason is TerminationReason.BUDGET_EXHAUSTED
    assert result.run.initial_values[0].value.shape == (2, 3)
    details = dict(result.run.sampling_details)
    assert details["stop_reason"] == "draw_cap"
    assert len(details["checkpoints"]) == 2


def test_improper_positive_prior_uses_support_interior():
    from numpyro.distributions import constraints

    def model():
        x = sample("x", lambda: dist.ImproperUniform(constraints.positive, (), ()))
        observe("y", lambda v: dist.Normal(v, 0.2), x, obs=jnp.array(1.4))

    values, _ = complete_initial_values(
        trace(model), jax.random.key(3), InitializationPolicy(), 1
    )
    assert float(values[0]["x"]) > 0


def test_checkpoint_convergence_requires_consecutive_checks_and_recovers_gaussian():
    from numpyro.infer import MCMC, NUTS

    from bayesmith.bridge.numpyro_bridge import to_numpyro
    from bayesmith.dispatch.sampling import SamplingControl, run_mcmc

    def model():
        x = sample("x", lambda: dist.Normal(0.0, 2.0))
        observe("y", lambda v: dist.Normal(v, 0.5), x, obs=jnp.array(1.7))

    graph = trace(model)
    runner = MCMC(
        NUTS(to_numpyro(graph)),
        num_warmup=200,
        num_samples=1200,
        num_chains=2,
        chain_method="sequential",
        progress_bar=False,
    )
    control = SamplingControl(
        stopping=StoppingPolicy(
            mode="checkpoints", min_draws=300, batch_size=300, mcse_mean=0.1
        )
    )
    samples = run_mcmc(runner, jax.random.key(442), graph, control)
    details = control.details
    assert details["stop_reason"] == "diagnostics_converged"
    assert details["draws_per_chain"] < 1200
    assert len(control.checkpoints) >= 2
    assert all(dict(c)["passed"] for c in control.checkpoints[-2:])
    # Independent conjugate oracle, generous Monte Carlo bound on this run.
    expected_mean = (1.7 / 0.5**2) / (1 / 2.0**2 + 1 / 0.5**2)
    mcse = dict(control.checkpoints[-1])["mcse_mean_max"]
    assert abs(float(samples["x"].mean()) - expected_mean) < 5 * mcse


def test_task_policy_and_run_records_roundtrip(tmp_path):
    from bayesmith.artifacts import dump_artifact, load_artifact

    task = PosteriorTask(
        meta=new_task_meta(),
        initialization=InitializationPolicy(
            values=(NamedArray("p", np.array([0.25, 0.0, 0.0]), ("parameter",)),),
            masks=(NamedArray("p", np.array([True, False, False]), ("parameter",)),),
        ),
    )
    path = tmp_path / "task.json"
    dump_artifact(task, path)
    restored = load_artifact(path)
    assert restored == task
    assert task_fingerprint(restored) == task_fingerprint(task)


def test_fixed_covariance_linear_block_reports_conditional_flat():
    from bayesmith.dispatch.task import compile_task
    from tests.dispatch.test_task_protocol import model_ref
    from tests.exact.models import straight_line

    with jax.enable_x64():
        planned = compile_task(
            straight_line(), PosteriorTask(meta=new_task_meta()), model_ref=model_ref()
        )
        findings = {f.code: f for f in planned.analysis.findings}
        assert findings["block_0_jeffreys"].conclusion == "flat"


def test_valid_supplied_point_survives_backend_retracing():
    from numpyro.infer import MCMC, NUTS

    from bayesmith.bridge.numpyro_bridge import to_numpyro
    from bayesmith.dispatch.sampling import SamplingControl, run_mcmc

    def model():
        p = sample("p", lambda: dist.LogNormal(jnp.log(101.0), 0.001))
        observe("y", lambda v: dist.Normal(v, v - 100), p, obs=jnp.array(101.2))

    graph = trace(model)
    mcmc = MCMC(
        NUTS(to_numpyro(graph)), num_warmup=10, num_samples=16, progress_bar=False
    )
    control = SamplingControl(
        initialization=InitializationPolicy(
            values=(NamedArray("p", np.array(101.0), ()),)
        ),
        stopping=StoppingPolicy(
            mode="checkpoints", min_draws=8, batch_size=8, ess_min=1e6
        ),
    )
    result = run_mcmc(mcmc, jax.random.key(7), graph, control)
    assert result["p"].shape == (16,)
    assert np.all(np.isfinite(result["p"]))


def test_strong_prior_sensitivity_does_not_pass_required_check():
    from bayesmith.artifacts import Refusal
    from bayesmith.dispatch.task import compile_task
    from tests.dispatch.test_task_protocol import model_ref
    from tests.exact.models import straight_line

    with jax.enable_x64():
        result = compile_task(
            straight_line(prior_mean=9.0, prior_std=0.1),
            PosteriorTask(
                meta=new_task_meta(),
                diagnostics=DiagnosticPolicy(required=("block_0_prior_sensitivity",)),
            ),
            model_ref=model_ref(),
        )
    assert isinstance(result, Refusal)


def test_independent_routes_do_not_hide_invalid_user_options():
    from bayesmith.artifacts import Refusal
    from bayesmith.dispatch.task import compile_task
    from tests.dispatch.test_task_protocol import model_ref
    from tests.exact.models import straight_line

    for overrides in (
        {
            "initialization": InitializationPolicy(
                values=(NamedArray("unknown", np.array(1.0), ()),)
            )
        },
        {"stopping": StoppingPolicy(mode="checkpoints")},
        {"stopping": StoppingPolicy(mcse_mean=1e-9)},
    ):
        result = compile_task(
            straight_line(),
            PosteriorTask(
                meta=new_task_meta(),
                diagnostics=DiagnosticPolicy(enabled=False),
                **overrides,
            ),
            model_ref=model_ref(),
        )
        assert isinstance(result, Refusal)


def test_gibbs_checkpoint_continuation_keeps_adaptation_and_rngs():
    from bayesmith import compile
    from bayesmith.dispatch.sampling import SamplingControl, run_mcmc
    from bayesmith.exact.gibbs import assemble
    from tests.exact.models import mixed_radiometer

    graph = mixed_radiometer()
    plan = compile(graph)

    def make():
        return assemble(
            graph,
            plan.exact.latents,
            tol=plan.exact.tol,
            method=plan.exact.method,
            sigma_rebuild=plan.sigma_needs_rebuild,
            num_warmup=20,
            num_samples=24,
            num_chains=2,
        )

    policy = InitializationPolicy(
        values=(NamedArray("tau", np.array([4.0, 5.0]), ("chain",)),)
    )
    whole, batches = make(), make()
    one = run_mcmc(
        whole,
        jax.random.key(80),
        graph,
        SamplingControl(initialization=policy),
        plan.exact.latents,
    )
    control = SamplingControl(
        initialization=policy,
        stopping=StoppingPolicy(
            mode="checkpoints", min_draws=8, batch_size=8, ess_min=1e6
        ),
    )
    many = run_mcmc(batches, jax.random.key(80), graph, control, plan.exact.latents)
    for name in graph.latents:
        np.testing.assert_array_equal(one[name], many[name])
    np.testing.assert_array_equal(
        whole.last_state.hmc_state.adapt_state.step_size,
        batches.last_state.hmc_state.adapt_state.step_size,
    )


def test_soft_time_cap_records_actual_batch_count():
    from numpyro.infer import MCMC, NUTS

    from bayesmith.bridge.numpyro_bridge import to_numpyro
    from bayesmith.dispatch.sampling import SamplingControl, run_mcmc

    graph = positive_vector()
    mcmc = MCMC(
        NUTS(to_numpyro(graph)), num_warmup=8, num_samples=16, progress_bar=False
    )
    control = SamplingControl(stopping=StoppingPolicy(batch_size=4), max_seconds=1e-12)
    draws = run_mcmc(mcmc, jax.random.key(5), graph, control)
    assert draws["p"].shape[0] == 4
    assert control.details["stop_reason"] == "time_cap"


def test_ill_conditioned_flat_jeffreys_is_not_reported_nonflat_from_roundoff():
    from bayesmith import det
    from bayesmith.dispatch.task import compile_task
    from tests.dispatch.test_task_protocol import model_ref

    with jax.enable_x64():
        u = jnp.array([[1.0, 1.0], [1.0, 1.001], [1.0, 0.999], [1.0, 1.0005]])

        def model():
            p = sample("p", lambda: dist.Normal(jnp.zeros(2), 1.0))
            mu = det("mu", lambda v: jnp.exp(u @ v), p)
            observe("y", lambda m: dist.Normal(m, 0.01 * m), mu, obs=jnp.ones(4))

        task = PosteriorTask(
            meta=new_task_meta(),
            initialization=InitializationPolicy(
                values=(NamedArray("p", np.zeros(2), ("parameter",)),)
            ),
        )
        planned = compile_task(trace(model), task, model_ref=model_ref())
        finding = next(
            f for f in planned.analysis.findings if f.code == "block_0_jeffreys"
        )
        # Analytic Fisher = (10000+2) U.T U, independent of p.
        assert finding.conclusion != "nonflat"
        assert dict(finding.measurements).get("rank") == 2


def test_required_jeffreys_check_accepts_a_resolved_flatness_assessment():
    from bayesmith.artifacts import Refusal
    from bayesmith.dispatch.task import compile_task
    from tests.dispatch.test_task_protocol import model_ref
    from tests.exact.models import straight_line

    with jax.enable_x64():
        result = compile_task(
            straight_line(),
            PosteriorTask(
                meta=new_task_meta(),
                diagnostics=DiagnosticPolicy(required=("block_0_jeffreys",)),
            ),
            model_ref=model_ref(),
        )
    assert not isinstance(result, Refusal)


def test_collapsed_time_cap_reconstructs_only_actual_draws_and_active_order():
    from bayesmith.artifacts import ComputeBudget, Refusal, TerminationReason
    from bayesmith.dispatch.task import compile_task, execute_task
    from tests.dispatch.test_collapse import _collapse_graph
    from tests.dispatch.test_task_protocol import model_ref

    with jax.enable_x64():
        graph = _collapse_graph()
        task = PosteriorTask(
            meta=new_task_meta(),
            budget=ComputeBudget(
                draws=16, warmup=8, chains=1, max_wall_clock_seconds=1e-12
            ),
            backend_options=(("collapse", True),),
            block_order=(("th",),),
            stopping=StoppingPolicy(batch_size=4),
        )
        planned = compile_task(graph, task, model_ref=model_ref())
        assert not isinstance(planned, Refusal)
        order = next(f for f in planned.analysis.findings if f.code == "sampling_order")
        assert dict(order.measurements)["order"] == (("th",),)
        result = execute_task(planned, key=jax.random.key(22))
        assert result.representation.chain_shape == (1, 4)
        assert all(a.value.shape[0] == 4 for a in result.representation.draws)
        assert result.run.termination.reason is TerminationReason.BUDGET_EXHAUSTED


def test_unsupported_family_preflight_is_unresolved_and_does_not_break_compilation():
    from bayesmith.artifacts import Refusal
    from bayesmith.dispatch.task import compile_task
    from tests.dispatch.test_task_protocol import model_ref

    with jax.enable_x64():

        def model():
            x = sample("x", lambda: dist.Normal(0.0, 1.0))
            observe(
                "y", lambda v: dist.Poisson(rate=jnp.exp(v)), x, obs=jnp.array([0.0, 1.0])
            )

        planned = compile_task(
            trace(model), PosteriorTask(meta=new_task_meta()), model_ref=model_ref()
        )
    assert not isinstance(planned, Refusal)
    finding = next(f for f in planned.analysis.findings if f.code == "block_0_jeffreys")
    assert finding.conclusion == "unresolved"
