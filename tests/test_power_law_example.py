"""Independent Fisher/posterior oracles and paired finite-sample bias checks."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from scipy.integrate import dblquad

from bayesmith import observe, plate, sample, trace
from bayesmith.dispatch.preflight import likelihood_information
from bayesmith.errors import GraphError
from bayesmith.graph.evaluate import log_joint
from examples.inference import power_law as demo
from examples.inference.jeffreys_priors import with_jeffreys_prior


@pytest.fixture(autouse=True)
def double_precision():
    with jax.enable_x64(True):
        yield


def fixture():
    x, channel = demo.design()
    return trace(demo.model, x, channel, jnp.zeros_like(x)), {
        "amplitude": jnp.array([1.4, 0.9]),
        "alpha": jnp.array([-1.1, 0.8]),
    }


def test_jeffreys_factor_and_automatic_information_match_score_expectation():
    graph, values = fixture()
    transformed = with_jeffreys_prior("power_law", graph)
    information = likelihood_information(graph, ("amplitude", "alpha"), values)
    x, channel = demo.design()
    at = jnp.concatenate([values["amplitude"], values["alpha"]])

    def law(theta):
        return demo.gaussian_observation(
            demo.mean_signal(x, theta[:2], theta[2:], channel)
        )

    original = law(at)
    # Symmetric Normal outcomes integrate the expected Hessian exactly.
    expected = jax.hessian(
        lambda t: (
            -0.5
            * sum(
                law(t).log_prob(original.mean + sign * original.scale).sum()
                for sign in (-1, 1)
            )
        )
    )(at)
    np.testing.assert_allclose(information, expected, rtol=2e-12)
    factor = transformed.joint_prior
    actual = jax.jit(lambda v: factor.log_density(transformed, v))(values)
    np.testing.assert_allclose(actual, 0.5 * np.linalg.slogdet(expected)[1], atol=2e-12)
    np.testing.assert_allclose(
        log_joint(transformed, values) - log_joint(graph, values), actual
    )
    gradient = jax.jit(jax.grad(lambda v: factor.log_density(transformed, v)))(values)
    np.testing.assert_allclose(gradient["amplitude"], 1 / values["amplitude"])
    jacobian = jax.jacfwd(lambda t: law(t).mean)(at)
    direct = lambda t: (
        0.5
        * jnp.linalg.slogdet(
            jax.jacfwd(lambda v: law(v).mean)(t).T
            @ jax.jacfwd(lambda v: law(v).mean)(t)
            / demo.NOISE_SD**2
        )[1]
    )
    np.testing.assert_allclose(gradient["alpha"], jax.grad(direct)(at)[2:], atol=2e-11)
    assert jacobian.shape == (2 * demo.OBSERVATIONS, 4)
    assert abs(float(actual) - 0.5 * np.log(np.diag(expected)).sum()) > 0.01
    assert np.isneginf(
        factor.log_density(transformed, {**values, "amplitude": jnp.array([-1.0, 1.0])})
    )


@pytest.mark.parametrize("jeffreys", [False, True])
def test_quadrature_oracle_matches_integrated_original_gaussian_density(jeffreys):
    from examples.inference.power_law_reference import posterior_summaries

    x = np.array([0.5, 0.8, 1.4, 2.0])
    data = np.array([[2.0, 1.5, 0.9, 0.7], [0.7, 0.9, 1.2, 1.8]])
    actual = posterior_summaries(x, data, jeffreys=jeffreys)
    assert actual["quadrature"]["passed"]
    for j in range(2):

        def density(alpha, amplitude, j=j):
            u = x**alpha
            prior = 1.0
            if jeffreys:
                J = np.column_stack([u, amplitude * u * np.log(x)]) / demo.NOISE_SD
                prior = np.sqrt(np.linalg.det(J.T @ J))
            return (
                np.exp(-0.5 * np.sum(((data[j] - amplitude * u) / demo.NOISE_SD) ** 2))
                * prior
            )

        def integral(moment):
            return dblquad(
                lambda a, A: moment(A, a) * density(a, A),
                0.2,
                3.0,
                lambda A: -2.0,
                lambda A: 2.0,
                epsabs=2e-10,
                epsrel=2e-9,
            )[0]

        norm = integral(lambda A, a: 1.0)
        for k in range(2):
            mean = integral(lambda A, a, k=k: (A, a)[k]) / norm
            variance = (
                integral(lambda A, a, k=k, mean=mean: ((A, a)[k] - mean) ** 2) / norm
            )
            np.testing.assert_allclose(actual["mean"][2 * j + k], mean, rtol=2e-8)
            np.testing.assert_allclose(
                actual["sd"][2 * j + k], np.sqrt(variance), rtol=2e-8
            )


def test_repeated_bias_uses_declared_gaussian_forward_model_and_paired_data():
    from examples.inference.power_law_reference import (
        bias_experiment,
        posterior_summaries,
    )

    x, channel = demo.design()
    truths = {"amplitude": jnp.array([1.4, 0.9]), "alpha": jnp.array([-1.1, 0.8])}
    result = bias_experiment(truths, repeats=4, seed=912)
    assert result["quadrature"]["passed"]
    data = (
        demo.gaussian_observation(demo.mean_signal(x, **truths, channel=channel))
        .sample(jax.random.key(912), (4,))
        .reshape(4, 2, -1)
    )
    truth = np.array([1.4, -1.1, 0.9, 0.8])
    for row in result["rows"]:
        values = np.array(
            [
                posterior_summaries(
                    np.asarray(x[: demo.OBSERVATIONS]),
                    d,
                    jeffreys=row["prior"] == "jeffreys",
                )[row["estimator"]]
                for d in data
            ]
        )
        np.testing.assert_allclose(
            row["bias"], (values - truth).mean(axis=0), atol=2e-9
        )
        np.testing.assert_allclose(
            row["mcse"], (values - truth).std(axis=0, ddof=1) / 2, atol=2e-9
        )


def test_profile_map_resolves_interior_modes_near_both_boundaries():
    from examples.inference.power_law_reference import _map

    x = np.array([0.5, 2.0])
    for alpha in (-1.999, 1.999):
        actual = _map(x, 1.4 * x**alpha, 0, False, demo.NOISE_SD)
        np.testing.assert_allclose(actual, [1.4, alpha], atol=2e-7, rtol=0)


def test_reference_cannot_hide_nonfinite_later_channel(monkeypatch):
    from examples.inference import power_law_reference as reference

    integrate = reference._integrate

    def broken(x, data, channel, *args):
        result = integrate(x, data, channel, *args)
        if channel == 1:
            result["mean"][0, 0] = np.nan
        return result

    monkeypatch.setattr(reference, "_integrate", broken)
    x = np.array([0.5, 0.8, 1.4, 2.0])
    result = reference.posterior_summaries(x, np.array([x, x]))
    assert not result["quadrature"]["passed"]


@pytest.mark.parametrize(
    "mode",
    [
        "moving_cutoff",
        "custom_density",
        "changed_transform",
        "transform_density",
        "stopped_shape",
    ],
)
def test_nonregular_or_changed_pareto_is_not_certified(mode):
    def law(value):
        scale = 1 + value**2 if mode == "moving_cutoff" else 1.0
        shape = (
            jax.lax.stop_gradient(value) + value if mode == "stopped_shape" else value
        )
        result = dist.Pareto(scale=scale, alpha=shape)
        if mode == "custom_density":
            result.log_prob = lambda x: -(x**2)
        elif mode == "changed_transform":
            result.transforms = result.transforms[:1]
        elif mode == "transform_density":
            # Same transform type, but density now has Pareto shape value**2.
            result.transforms[0].log_abs_det_jacobian = (
                lambda x, y, intermediates=None: (
                    x + (value**2 - value) * x - jnp.log(value)
                )
            )
        return result

    def model():
        value = sample("value", lambda: dist.Uniform(1, 4))
        observe("obs", law, value, obs=jnp.ones(4) * 10)

    graph = trace(model)
    with pytest.raises(GraphError):
        likelihood_information(graph, ("value",), {"value": jnp.array(2.0)})


@pytest.mark.parametrize("layout", ["plate", "expanded", "event", "masked"])
def test_pareto_information_preserves_replication_and_mask_semantics(layout):
    def model():
        rows = plate("rows", 3) if layout == "plate" else None
        alpha = sample(
            "alpha", lambda: dist.Uniform(1.0, 4.0), **({"plate": rows} if rows else {})
        )

        def law(a):
            result = dist.Pareto(1.0, a)
            if layout in ("expanded", "event"):
                result = result.expand((3,))
            return result.to_event(1) if layout == "event" else result

        observe(
            "obs",
            law,
            alpha,
            obs=jnp.array([2.0, 3.0, 4.0]),
            mask=jnp.array([True, False, True]) if layout == "masked" else None,
            **({"plate": rows} if rows else {}),
        )

    values = {"alpha": jnp.full(3, 2.0) if layout == "plate" else jnp.array(2.0)}
    information = likelihood_information(trace(model), ("alpha",), values)
    expected = (
        np.eye(3) / 4
        if layout == "plate"
        else np.array([[0.5 if layout == "masked" else 0.75]])
    )
    np.testing.assert_allclose(information, expected, atol=1e-14)
