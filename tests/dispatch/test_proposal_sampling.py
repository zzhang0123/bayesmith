"""Whole-proposal and mixed chains retain real states across checkpoints."""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import observe, sample, trace
from bayesmith.artifacts import InitializationPolicy, NamedArray, StoppingPolicy
from bayesmith.dispatch.sampling import SamplingControl
from tests.exact.test_proposals import api, gaussian_graph, policy


def sampler_api():
    assert (
        importlib.util.find_spec("bayesmith.dispatch.proposal_sampling") is not None
    ), "the whole-proposal and mixed MCMC adapters are missing"
    from bayesmith.dispatch import proposal_sampling

    return proposal_sampling


def mixed_graph():
    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        z = sample("z", lambda: dist.Normal(0.0, 1.0))
        observe("y", lambda a, b: dist.Normal(a + b, 0.5), x, z, obs=jnp.asarray(0.7))

    return trace(model)


def run(graph, compiled, control, chain_method="sequential", **kwargs):
    return sampler_api().run_proposal_mcmc(
        graph,
        compiled,
        jax.random.key(17),
        num_warmup=12,
        num_samples=24,
        num_chains=2,
        chain_method=chain_method,
        nuts_options=None,
        progress_bar=False,
        control=control,
        **kwargs,
    )


@pytest.mark.parametrize("mixed", [False, True])
def test_checkpoint_batches_match_continuous_run_bitwise(mixed):
    graph = mixed_graph() if mixed else gaussian_graph()
    values = {name: jnp.asarray(0.1) for name in graph.latents}
    compiled = api().compile_proposals(graph, (policy(steps=2),), values)
    init = InitializationPolicy(
        values=tuple(NamedArray(n, np.asarray(v), ()) for n, v in values.items())
    )
    whole_control = SamplingControl(initialization=init)
    chunked_control = SamplingControl(
        initialization=init,
        stopping=StoppingPolicy(
            mode="checkpoints", min_draws=8, batch_size=8, ess_min=1e6
        ),
    )
    whole = run(graph, compiled, whole_control)
    chunked = run(graph, compiled, chunked_control)
    for name in graph.latents:
        np.testing.assert_array_equal(chunked.samples[name], whole.samples[name])
    assert chunked.method == "proposal_mh"
    assert chunked.log_weights is None
    assert set(chunked.diagnostics) == set(graph.latents)
    assert chunked_control.details["draws_per_chain"] == 24
    assert chunked_control.details["stop_reason"] == "draw_cap"
    assert len(chunked_control.checkpoints) == 3
    record = dict(chunked_control.details["proposal_diagnostics"][0])
    assert record["attempted"] == 96
    assert record["accepted"] > 30
    assert record["numerical_failure"] == 0
    assert dict(whole_control.details["proposal_diagnostics"][0]) == record


@pytest.mark.parametrize("mixed", [False, True])
def test_vectorized_chains_and_single_chain_kernel_jit(mixed):
    graph = mixed_graph() if mixed else gaussian_graph()
    compiled = api().compile_proposals(
        graph, (policy(),), {n: jnp.asarray(0.1) for n in graph.latents}
    )
    control = SamplingControl(initialization=InitializationPolicy())
    posterior = run(graph, compiled, control, chain_method="vectorized")
    assert all(v.shape == (48,) for v in posterior.samples.values())
    assert all(bool(jnp.all(jnp.isfinite(v))) for v in posterior.samples.values())
    assert dict(control.details["proposal_diagnostics"][0])["attempted"] == 48


def test_all_proposal_kernel_has_only_real_model_sites_and_jittable_state():
    p = sampler_api()
    graph = gaussian_graph()
    compiled = api().compile_proposals(graph, (policy(),), {"x": jnp.asarray(0.1)})
    kernel = p.ProposalKernel(compiled)
    state = kernel.init(jax.random.key(4), 0, {"x": jnp.asarray(0.1)}, (), {})
    assert set(state.z) == {"x"}
    transition = jax.jit(lambda s: kernel.sample(s, (), {}))
    with jax.transfer_guard("disallow"):
        next_state = transition(state)
    assert bool(jnp.isfinite(next_state.z["x"]))
    assert int(next_state.proposal_diagnostics.attempted[0]) == 1


def test_mixed_recomputes_inner_potential_and_gradient_after_proposal():
    p = sampler_api()
    from numpyro.infer import NUTS

    from bayesmith.bridge.numpyro_bridge import to_numpyro

    graph = mixed_graph()
    compiled = api().compile_proposals(
        graph, (policy(),), {"x": jnp.asarray(0.1), "z": jnp.asarray(0.1)}
    )
    kernel = p.ProposalGibbs(NUTS(to_numpyro(graph)), compiled)
    state = kernel.init(jax.random.key(2), 5, None, (), {})
    advanced = jax.jit(lambda s: kernel.sample(s, (), {}))(state)
    potential = kernel.inner_kernel._potential_fn_gen(
        _gibbs_sites={"x": advanced.z["x"]}
    )
    value, gradient = jax.value_and_grad(potential)(advanced.hmc_state.z)
    np.testing.assert_allclose(advanced.hmc_state.potential_energy, value, rtol=2e-6)
    np.testing.assert_allclose(advanced.hmc_state.z_grad["z"], gradient["z"], rtol=2e-6)


def test_mixed_dynamic_support_matches_independent_stationary_law():
    from scipy.stats import truncnorm

    def model():
        x = sample("x", lambda: dist.Uniform(0.2, 2.0))
        sample("child", lambda v: dist.Uniform(0.0, v), x)
        observe("y", lambda v: dist.Normal(v, 0.35), x, obs=jnp.asarray(0.9))

    graph = trace(model)
    compiled = api().compile_proposals(
        graph, (policy(),), {"x": jnp.asarray(1.0), "child": jnp.asarray(0.5)}
    )
    control = SamplingControl(initialization=InitializationPolicy())
    posterior = sampler_api().run_proposal_mcmc(
        graph,
        compiled,
        jax.random.key(71),
        num_warmup=150,
        num_samples=1800,
        num_chains=2,
        chain_method="vectorized",
        nuts_options=None,
        progress_bar=False,
        control=control,
    )
    x, child = (
        np.asarray(posterior.samples["x"]),
        np.asarray(posterior.samples["child"]),
    )
    mean, variance = truncnorm.stats(
        (0.2 - 0.9) / 0.35, (2.0 - 0.9) / 0.35, loc=0.9, scale=0.35, moments="mv"
    )
    assert np.all((0 < child) & (child < x))
    assert abs(x.mean() - mean) < 0.04
    assert abs(x.var() - variance) < 0.025
    assert abs((child / x).mean() - 0.5) < 0.03
    assert abs((child / x).var() - 1 / 12) < 0.012
    assert abs(child.mean() - 0.5 * mean) < 0.035


def test_each_actual_initial_chain_requires_a_finite_proposal_before_warmup():
    def model():
        x = sample("x", lambda: dist.Uniform(-3.0, 3.0))
        observe("y", lambda v: dist.Normal(v * v, 0.5), x, obs=jnp.asarray(1.0))

    graph = trace(model)
    compiled = api().compile_proposals(
        graph, (policy(damping=0.0),), {"x": jnp.asarray(1.0)}
    )
    control = SamplingControl(
        initialization=InitializationPolicy(
            values=(NamedArray("x", np.array([1.0, 0.0]), ("chain",)),)
        )
    )
    with pytest.raises(ValueError, match="chain 1.*proposal|proposal.*chain 1"):
        run(graph, compiled, control)


@pytest.mark.parametrize("mixed", [False, True])
def test_x64_mcmc_counter_state_has_consistent_dtype(mixed):
    with jax.enable_x64():
        graph = mixed_graph() if mixed else gaussian_graph()
        compiled = api().compile_proposals(
            graph, (policy(),), {n: jnp.asarray(0.1) for n in graph.latents}
        )
        control = SamplingControl(initialization=InitializationPolicy())
        posterior = run(graph, compiled, control)
        assert all(v.dtype == jnp.float64 for v in posterior.samples.values())
        assert dict(control.details["proposal_diagnostics"][0])["attempted"] == 48


@pytest.mark.parametrize("nuts_remainder", [False, True])
def test_mixed_correction_schedule_records_real_mh_counts_and_resumes_in_fp64(
    nuts_remainder,
):
    with jax.enable_x64():

        def model():
            x = sample("x", lambda: dist.Normal(0.0, 1.0))
            z = sample("z", lambda: dist.Normal(0.0, 1.0))
            if nuts_remainder:
                w = sample("w", lambda: dist.Normal(0.0, 1.0))
                observe(
                    "y",
                    lambda a, b, c: dist.Normal(a + b + c, 0.5),
                    x,
                    z,
                    w,
                    obs=jnp.asarray(0.7),
                )
            else:
                observe(
                    "y",
                    lambda a, b: dist.Normal(a + b, 0.5),
                    x,
                    z,
                    obs=jnp.asarray(0.7),
                )

        graph = trace(model)
        initial = {n: jnp.asarray(0.1) for n in graph.latents}
        compiled = api().compile_proposals(
            graph,
            (policy(mh_correction=False), policy(names=("z",), mh_correction=True)),
            initial,
        )
        init = InitializationPolicy(
            values=tuple(NamedArray(n, np.asarray(v), ()) for n, v in initial.items())
        )
        whole_control = SamplingControl(initialization=init)
        chunked_control = SamplingControl(
            initialization=init,
            stopping=StoppingPolicy(
                mode="checkpoints", min_draws=8, batch_size=8, ess_min=1e6
            ),
        )
        whole = run(graph, compiled, whole_control)
        chunked = run(graph, compiled, chunked_control)
        assert whole.method == chunked.method == "proposal_approximate"
        assert "approximate" in whole.reason.lower()
        for name in graph.latents:
            np.testing.assert_array_equal(whole.samples[name], chunked.samples[name])
        records = [dict(row) for row in chunked_control.details["proposal_diagnostics"]]
        assert records[0]["mh_correction"] is False
        assert records[0]["mh_attempted"] == records[0]["mh_accepted"] == 0
        assert records[0]["mh_acceptance_rate"] is None
        assert records[0]["accepted"] == records[0]["attempted"] == 48
        assert records[1]["mh_correction"] is True
        assert records[1]["mh_attempted"] == records[1]["attempted"] == 48
        assert records[1]["mh_accepted"] == records[1]["accepted"] > 0
        assert chunked_control.details["target_fidelity"] == "approximate"
