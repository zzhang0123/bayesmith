"""Analytic demo priors must target the expected Fisher in the stated scope."""

import importlib

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import trace
from bayesmith.graph.evaluate import log_joint


@pytest.fixture(autouse=True)
def double_precision():
    with jax.enable_x64(True):
        yield


def _helper():
    return importlib.import_module("examples.inference.jeffreys_priors")


def _fixture(case):
    module = importlib.import_module(f"examples.inference.{case}")
    if case == "exponential_decay":
        time = jnp.tile(jnp.linspace(0.0, 5.0, 120), 2)
        channel = jnp.repeat(jnp.arange(2), 120)
        values = {"amplitude": jnp.array([1.3, 2.1]),
                  "rate": jnp.array([0.4, 1.6]), "offset": jnp.array([0.2, -0.3])}
        graph = trace(module.model, time, channel, jnp.zeros_like(time))
    elif case == "bernoulli":
        axis = jnp.linspace(-2.0, 2.0, 20)
        grid1, grid2 = jnp.meshgrid(axis, axis, indexing="xy")
        X = module.design_matrix(grid1.ravel(), grid2.ravel())
        values = {"beta": jnp.array([-0.8, 1.6, -0.4, 0.3])}
        graph = trace(module.model, X, jnp.zeros(X.shape[0]))
    elif case == "multiplicative_noise":
        U, A = module.design_matrices(jnp.linspace(0.0, 1.0, 128))
        values = {"p_g": jnp.array([0.1, -0.2]), "p_n": jnp.array([1.3, 0.8])}
        graph = trace(module.model, U, A, jnp.zeros(U.shape[0]))
    else:
        index = jnp.repeat(jnp.arange(8), 40)
        values = {"population": jnp.array(0.8), "groups": jnp.linspace(-0.1, 1.4, 8)}
        graph = trace(module.model, index, jnp.zeros(index.size))
    return module, graph, values


@pytest.mark.parametrize("case", ["exponential_decay", "multiplicative_noise"])
def test_gaussian_prior_matches_exact_expected_score_curvature(case):
    """Dropping covariance curvature or cross-block terms changes this oracle."""
    module, graph, values = _fixture(case)
    prior_graph = _helper().with_jeffreys_prior(case, graph)
    if case == "exponential_decay":
        at = jnp.concatenate([values[n] for n in ("amplitude", "rate", "offset")])

        def law(theta):
            mean = module.decay(graph.node("time").value, theta[:2], theta[2:4],
                                theta[4:], graph.node("channel").value)
            return module.gaussian_observation(mean)
    else:
        at = jnp.concatenate([values["p_g"], values["p_n"]])

        def law(theta):
            mean = module.mean_signal(graph.node("U").value, graph.node("A").value,
                                      theta[:2], theta[2:])
            return module.relative_gaussian(mean)
    distribution = law(at)
    sigma = jnp.sqrt(distribution.variance)
    upper, lower = distribution.mean + sigma, distribution.mean - sigma

    def expected_loss(theta):
        trial = law(theta)
        return -0.5 * (trial.log_prob(upper).sum() + trial.log_prob(lower).sum())

    # Normal log-likelihood Hessians are quadratic in standardized residuals;
    # the two-point rule integrates their expectation exactly, without MC.
    matrix = np.asarray(jax.hessian(expected_loss)(at))
    sign, determinant = np.linalg.slogdet(matrix)
    assert sign == 1
    actual = prior_graph.joint_prior.log_density(prior_graph, values)
    np.testing.assert_allclose(actual, 0.5 * determinant, rtol=2e-11, atol=2e-11)
    if case == "multiplicative_noise":
        separate = sum(0.5 * np.linalg.slogdet(matrix[s, s])[1]
                       for s in (slice(0, 2), slice(2, 4)))
        assert abs(float(actual) - separate) > 1e-3


def test_logistic_prior_matches_exact_bernoulli_expected_curvature():
    _, graph, values = _fixture("bernoulli")
    transformed = _helper().with_jeffreys_prior("bernoulli", graph)
    X = graph.node("X").value
    # In canonical logits, every binary outcome has this same Hessian.
    matrix = jax.hessian(lambda b: jax.nn.softplus(X @ b).sum())(values["beta"])
    expected = 0.5 * np.linalg.slogdet(np.asarray(matrix))[1]
    actual = transformed.joint_prior.log_density(transformed, values)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_hierarchical_prior_integrates_groups_and_preserves_their_conditional_law():
    module, graph, values = _fixture("hierarchical")
    transformed = _helper().with_jeffreys_prior("hierarchical", graph)
    covariance = module.GROUP_SD**2 * np.ones((40, 40)) + 0.25**2 * np.eye(40)
    information = 8 * np.ones(40) @ np.linalg.solve(covariance, np.ones(40))
    expected = 0.5 * np.log(information)
    for population in (-2.0, 0.4, 2.7):
        at = {**values, "population": jnp.asarray(population),
              "groups": values["groups"] + population}
        np.testing.assert_allclose(transformed.joint_prior.log_density(transformed, at),
                                   expected, rtol=1e-12)
        np.testing.assert_allclose(log_joint(transformed, at) - log_joint(graph, at),
                                   expected, rtol=1e-11)
    assert transformed.node("groups") is graph.node("groups")


@pytest.mark.parametrize("case", ["exponential_decay", "bernoulli", "multiplicative_noise", "hierarchical"])
def test_prior_factor_preserves_bounded_support_and_is_jittable(case):
    _, graph, values = _fixture(case)
    transformed = _helper().with_jeffreys_prior(case, graph)
    prior = transformed.joint_prior
    actual = jax.jit(lambda v: prior.log_density(transformed, v))(values)
    np.testing.assert_allclose(log_joint(transformed, values) - log_joint(graph, values),
                               actual, atol=1e-8, rtol=1e-9)
    gradients = jax.grad(lambda v: prior.log_density(transformed, v))(values)
    assert all(np.all(np.isfinite(g)) for g in gradients.values())
    outside = {**values, prior.over[0]: jnp.full_like(values[prior.over[0]], 100.0)}
    assert np.isneginf(prior.log_density(transformed, outside))
    assert all(transformed.node(n) is graph.node(n) for n in graph.latents)


def test_decay_zero_amplitude_has_zero_density_without_a_ridge():
    _, graph, values = _fixture("exponential_decay")
    transformed = _helper().with_jeffreys_prior("exponential_decay", graph)
    actual = transformed.joint_prior.log_density(
        transformed, {**values, "amplitude": jnp.array([0.0, 2.1])})
    assert np.isneginf(actual)


def test_multiplicative_jeffreys_is_gain_independent_and_scales_as_inverse_square():
    _, graph, values = _fixture("multiplicative_noise")
    transformed = _helper().with_jeffreys_prior("multiplicative_noise", graph)
    log_prior = lambda v: transformed.joint_prior.log_density(transformed, v)
    base = log_prior(values)
    np.testing.assert_allclose(log_prior({**values, "p_g": jnp.array([-0.3, 0.4])}), base)
    np.testing.assert_allclose(log_prior({**values, "p_n": values["p_n"] * 2}) - base,
                               -2 * np.log(2), atol=1e-12)


@pytest.mark.parametrize("alteration", ["density", "proper_prior", "wrapper_density", "already_declared"])
def test_registered_prior_refuses_a_different_model_or_a_second_prior(alteration):
    _, graph, _ = _fixture("bernoulli")
    if alteration == "density":
        graph = eqx.tree_at(lambda g: g.node("obs").dist_fn, graph,
                            lambda logits: dist.Bernoulli(logits=2 * logits))
    elif alteration == "proper_prior":
        graph = eqx.tree_at(lambda g: g.node("beta").dist_fn, graph,
                            lambda: dist.Normal(jnp.zeros(4), 1.0).to_event(1))
    elif alteration == "wrapper_density":
        def wrong_prior():
            law = dist.Uniform(jnp.full(4, -4.0), jnp.full(4, 4.0)).to_event(1)
            law.log_prob = lambda value: -jnp.sum(value**2)
            return law
        graph = eqx.tree_at(lambda g: g.node("beta").dist_fn, graph, wrong_prior)
    else:
        graph = _helper().with_jeffreys_prior("bernoulli", graph)
    with pytest.raises(ValueError):
        _helper().with_jeffreys_prior("bernoulli", graph)


def test_composed_process_has_no_fabricated_jeffreys_density():
    with pytest.raises(ValueError, match="marginal"):
        _helper().with_jeffreys_prior("composed_process", None)


def test_explicit_unavailable_counterpart_does_not_report_success(tmp_path, monkeypatch, capsys):
    import json

    from examples.inference.jeffreys_demo import main

    monkeypatch.setattr("sys.argv", ["jeffreys_demo", "--case", "composed_process",
                                    "--output", str(tmp_path)])
    assert main() == 2
    assert "no posterior samples were generated" in capsys.readouterr().err
    status = json.loads((tmp_path / "composed_process/status.json").read_text())
    assert status["status"] == "not_constructed"
    assert not (tmp_path / "composed_process/result.json").exists()
