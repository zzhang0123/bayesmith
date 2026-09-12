"""Automatic discovery must certify functions, not annotations or anchor values."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import compile, det, observe, sample, trace
from bayesmith.diagnose import structure
from bayesmith.dispatch.classify import prior_environment
from bayesmith.dispatch.factor import factor_partition


@pytest.fixture(autouse=True)
def double_precision():
    with jax.enable_x64():
        yield


def graph_for(mean, *, prior="normal", scale=lambda a, b: 0.5, declared=()):
    def model():
        a = sample("a", lambda: dist.Normal(0.0, 1.0) if prior == "normal"
                   else dist.Uniform(-2.0, 2.0))
        b = sample("b", lambda: dist.Normal(0.0, 1.0))
        mu = det("mu", mean, a, b, linear_in=declared)
        observe("y", lambda m, x, z: dist.Normal(m, scale(x, z)),
                mu, a, b, obs=0.7)
    return trace(model)


def certificate(graph, names):
    function = getattr(structure, "conditional_affinity_certificate", None)
    assert callable(function), "symbolic-complement affinity certificate is missing"
    return function(graph, names, prior_environment(graph))


def test_public_compile_finds_joint_affinity_without_declarations():
    graph = graph_for(lambda a, b: 2 * a + 3 * b + 0.4)
    plan = compile(graph)
    assert [(b.latents, b.method) for b in plan.blocks] == [(('a', 'b'), 'gcr')]
    assert plan.exact.structure["certified"]
    assert "structurally certified" in str(plan)
    estimate = plan.estimate()
    # Independent posterior precision: I + h h^T / sigma^2.
    h = np.array([2.0, 3.0])
    expected = np.linalg.solve(np.eye(2) + np.outer(h, h) / 0.25,
                               h * (0.7 - 0.4) / 0.25)
    np.testing.assert_allclose([estimate.values[n] for n in ("a", "b")],
                               expected, rtol=1e-8, atol=1e-10)


def test_factor_partition_splits_undeclared_product():
    plan = factor_partition(graph_for(lambda a, b: a * b))
    assert [(b.latents, b.method) for b in plan.blocks] == [(("a",), "gcr"), (("b",), "gcr")]
    assert all(b.structure["certified"] for b in plan.blocks)


def test_default_compile_executes_every_conditional_block():
    plan = compile(graph_for(lambda a, b: a * b))
    assert [(b.latents, b.method) for b in plan.blocks] == [(("a",), "gcr"), (("b",), "gcr")]
    assert "Gibbs" in str(plan)
    assert "iid" not in str(plan)
    draws = plan.sample(jax.random.key(9), num_warmup=12, num_samples=25)
    assert draws.method == "factor_gibbs"
    assert set(draws.samples) == {"a", "b"}
    for values in draws.samples.values():
        assert values.shape == (25,)
        assert np.isfinite(values).all()
        assert np.std(values) > 0
    with pytest.raises(NotImplementedError, match="joint posterior mean"):
        plan.estimate()


@pytest.mark.parametrize("prior", ["normal", "uniform"])
def test_structure_does_not_depend_on_gaussian_prior(prior):
    graph = graph_for(lambda a, b: a + b, prior=prior)
    proof = certificate(graph, ("a", "b"))
    assert proof["certified"]
    assert proof["gaussian_priors"] is (prior == "normal")
    plan = compile(graph)
    if prior == "uniform":
        assert "a" in plan.sampled.latents


def test_external_nonlinear_coefficient_is_conditionally_affine():
    graph = graph_for(lambda a, b: a * jnp.sin(b))
    assert certificate(graph, ("a",))["certified"]
    assert not certificate(graph, ("b",))["certified"]
    assert compile(graph).exact.latents == ("a",)


@pytest.mark.parametrize("mean", [
    lambda a, b: a + b * a**2,
    lambda a, b: jnp.where(b >= 0, a, a**2),
    lambda a, b: jax.lax.cond(b >= 0, lambda: a, lambda: a**2),
])
def test_zero_or_selected_branch_cannot_hide_outside_dependence(mean):
    graph = graph_for(mean, declared=("a",))
    assert not certificate(graph, ("a",))["certified"]
    assert not any("a" in block.latents and block.method == "gcr"
                   for block in compile(graph).blocks)


def test_covariance_dependence_is_separate_from_affinity():
    graph = graph_for(lambda a, b: a + b, scale=lambda a, b: jnp.exp(a))
    proof = certificate(graph, ("a", "b"))
    assert proof["certified"]
    assert not proof["covariance_independent"]
    assert compile(graph).exact.method != "gcr"


def test_small_nonlinearity_that_passes_probes_does_not_gain_a_proof():
    graph = graph_for(lambda a, b: a + 1e-30 * a**3 + b, declared=("a", "b"))
    assert not certificate(graph, ("a",))["certified"]
    assert not any("a" in block.latents and block.method == "gcr"
                   for block in compile(graph).blocks)


def test_custom_derivative_does_not_certify_the_primal():
    @jax.custom_jvp
    def deceptive(x):
        return x**2

    @deceptive.defjvp
    def derivative(primals, tangents):
        return deceptive(primals[0]), tangents[0]

    graph = graph_for(lambda a, b: deceptive(a) + b, declared=("a", "b"))
    assert not certificate(graph, ("a",))["certified"]


def test_final_group_is_certified_with_all_other_latents_symbolic():
    def model():
        a = sample("a", lambda: dist.Normal(0., 1.))
        b = sample("b", lambda: dist.Normal(0., 1.))
        c = sample("c", lambda: dist.Normal(0., 1.))
        mu = det("mu", lambda x, y, z: x*y*z, a, b, c)
        observe("y", lambda m: dist.Normal(m, 0.5), mu, obs=0.2)
    plan = factor_partition(trace(model))
    assert [b.latents for b in plan.blocks] == [("a",), ("b",), ("c",)]


def test_joint_prior_is_not_omitted_by_automatic_gaussian_update():
    from types import SimpleNamespace

    graph = graph_for(lambda a, b: a + b)
    prior = SimpleNamespace(over=("a",),
                            log_density=lambda graph, values: -10 * values["a"]**2)
    graph = eqx.tree_at(lambda g: g.joint_prior, graph, prior, is_leaf=lambda x: x is None)
    assert certificate(graph, ("a",))["certified"]
    assert not certificate(graph, ("a",))["gaussian_priors"]
    assert all(b.method == "nuts" for b in compile(graph).blocks)


@pytest.mark.parametrize("scale", [
    lambda a, b: jnp.exp(0.0 * a + 0.0 * b),
    lambda a, b: jnp.exp(jnp.zeros(2) @ jnp.stack((a, b))),
])
def test_literal_and_closed_constant_zeros_prove_covariance_independence(scale):
    graph = graph_for(lambda a, b: a + b, scale=scale)
    assert certificate(graph, ("a", "b"))["covariance_independent"]
    assert compile(graph).exact.method == "gcr"


def test_task_reports_multiple_blocks_as_a_chain_and_round_trips_evidence():
    from bayesmith.artifacts import ComputeBudget, PosteriorTask, new_task_meta
    from bayesmith.dispatch.task import compile_task, execute_task
    from tests.dispatch.test_task_protocol import model_ref

    task = PosteriorTask(meta=new_task_meta(), budget=ComputeBudget(draws=12, warmup=5, chains=1))
    planned = compile_task(graph_for(lambda a, b: a*b), task, model_ref=model_ref())
    assert planned.runtime_plan.method == "factor_gibbs"
    result = execute_task(planned, key=jax.random.key(19))
    assert result.representation.chain_shape == (1, 12)


@pytest.mark.parametrize("mixed", [False, True])
def test_complex_factor_chain_preserves_model_coordinates(mixed):
    from bayesmith.distributions import ComplexNormal

    def model():
        a = sample("a", lambda: ComplexNormal(jnp.zeros(2, dtype=complex), 1.))
        b = sample("b", lambda: dist.Normal(0., 1.))
        if mixed:
            c = sample("c", lambda: dist.Uniform(-1., 1.))
            observe("z", lambda c: dist.Normal(c, 1.), c, obs=0.1)
        mu = det("mu", lambda a, b: jnp.real(a)*b, a, b)
        observe("y", lambda m: dist.Normal(m, 1.), mu, obs=jnp.array([0.2, 0.4]))
    plan = compile(trace(model))
    assert plan.method == "factor_gibbs"
    result = plan.sample(jax.random.key(4), num_warmup=3, num_samples=8)
    assert result.samples["a"].shape == (8, 2)
    assert jnp.iscomplexobj(result.samples["a"])
    assert np.isfinite(result.samples["a"]).all()
    assert result.ess > 0


def test_mixed_complex_gibbs_updates_the_encoded_real_sites():
    """A complex exact block must update NumPyro's ``__re/__im`` sites."""
    from bayesmith.distributions import ComplexNormal

    def model():
        a = sample("a", lambda: ComplexNormal(jnp.asarray(0.0 + 0.0j), 1.0))
        c = sample("c", lambda: dist.Normal(0.0, 1.0))
        mu = det("mu", lambda z: jnp.real(z), a)
        observe("y", lambda m: dist.Normal(m, 0.1), mu, obs=2.0)
        observe("z", lambda x: dist.Normal(jnp.sin(x), 1.0), c, obs=0.0)

    plan = compile(trace(model))
    assert plan.sampled.latents == ("c",)
    result = plan.sample(jax.random.key(23), num_warmup=8, num_samples=24)
    assert float(np.mean(np.real(result.samples["a"]))) > 1.0


@pytest.mark.parametrize("prior", [lambda: dist.Uniform(-1., 1.),
                                  lambda: dist.Gamma(2., 1.)])
def test_single_exact_block_samples_with_non_location_family_remainder(prior):
    def model():
        a = sample("a", lambda: dist.Normal(0., 1.))
        b = sample("b", prior)
        observe("y", lambda a, b: dist.Normal(a + b, 1.), a, b, obs=0.2)

    plan = compile(trace(model))
    assert plan.exact.latents == ("a",)
    result = plan.sample(jax.random.key(35), num_warmup=3, num_samples=8)
    assert set(result.samples) == {"a", "b"}
    assert all(np.isfinite(value).all() for value in result.samples.values())


def test_mixed_factor_chain_refuses_unsupported_vectorized_hmcgibbs():
    def model():
        a = sample("a", lambda: dist.Normal(0., 1.))
        b = sample("b", lambda: dist.Normal(0., 1.))
        c = sample("c", lambda: dist.Uniform(-1., 1.))
        observe("y", lambda a, b, c: dist.Normal(a*b + c, 1.),
                a, b, c, obs=0.2)

    plan = compile(trace(model))
    assert plan.method == "factor_gibbs"
    with pytest.raises(NotImplementedError, match="vectorized"):
        plan.sample(jax.random.key(35), num_chains=2, chain_method="vectorized")
