"""Independent density, cumulant and coordinate checks for the sky experiment."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from scipy.integrate import quad
from scipy.special import gammaln
from scipy.stats import norm

from bayesmith.cumulants import FieldEdgeworth, LowRankCumulants
from examples.inference.campbell_sky import (
    SkyConfig,
    beam_eigenvalues,
    beam_matrix,
    coefficient_logpdf,
    connected_cumulant,
    deconvolve,
    edgeworth_field_correction,
    oracle_log_evidence,
    oracle_posterior_mean,
    poisson_gaussian_prior,
    simulate,
)


def test_beam_is_invertible_and_creates_connected_spatial_structure():
    config = SkyConfig(shape=(4, 5))
    data = simulate(config, 12)
    np.testing.assert_allclose(
        deconvolve(data["observed_sky"], config),
        data["observed_coefficients"],
        atol=2e-15,
    )
    matrix = beam_matrix(config)
    np.testing.assert_allclose(
        matrix @ data["coefficients"].ravel(), data["sky"].ravel(), atol=1e-15
    )
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-15)
    assert np.min(beam_eigenvalues(config)) > 0
    assert connected_cumulant(config, 2.0, (0, 0, 1)) > 0
    assert connected_cumulant(config, 2.0, (0, 0, 0, 1)) > 0
    for rate in (0.5, 2.0, 8.0):
        assert connected_cumulant(config, rate, (0,)) == 0.0
        assert connected_cumulant(config, rate, (0, 0)) == pytest.approx(0.4)
        assert connected_cumulant(config, rate, (0, 1)) == pytest.approx(0.12)


@pytest.mark.parametrize("rate", [0.5, 2.0, 8.0])
def test_mixture_moments_obey_campbell_with_mean_and_variance_fixed(rate):
    config = SkyConfig(shape=(3, 3))
    count = np.arange(97)
    weights = np.exp(count * np.log(rate) - rate - gammaln(count + 1))
    mu = np.sqrt(config.shot_variance / rate) * (count - rate)
    variance = config.diffuse_variance
    # Explicit Gaussian component moments, independent of the field formula.
    mean = weights @ mu
    m2 = weights @ (mu**2 + variance)
    m3 = weights @ (mu**3 + 3 * mu * variance)
    m4 = weights @ (mu**4 + 6 * mu**2 * variance + 3 * variance**2)
    assert mean == pytest.approx(0.0, abs=1e-13)
    assert m2 == pytest.approx(1.0, abs=1e-13)
    assert m3 == pytest.approx(config.shot_variance**1.5 / np.sqrt(rate), abs=1e-12)
    assert m4 - 3 * m2 * m2 == pytest.approx(config.shot_variance**2 / rate, abs=1e-12)


def test_campbell_cross_cumulants_match_cumulant_generating_function_derivatives():
    config = SkyConfig(shape=(3, 3))
    rows = jnp.asarray(beam_matrix(config)[[0, 1]])
    amplitude = np.sqrt(config.shot_variance / config.rate)

    def cgf(t):
        argument = t @ rows
        return (
            config.rate
            * jnp.sum(jnp.expm1(amplitude * argument) - amplitude * argument)
            + config.diffuse_variance * jnp.sum(argument**2) / 2
        )

    derivative = cgf
    for order in range(1, 5):
        derivative = jax.jacfwd(derivative)
        tensor = derivative(jnp.zeros(2))
        for indices in [(0,) * order, (0,) * (order - 1) + (1,)]:
            assert float(tensor[indices]) == pytest.approx(
                connected_cumulant(config, config.rate, indices), rel=2e-6, abs=1e-8
            )


def test_numpyro_target_density_matches_independent_scipy_sum():
    config = SkyConfig(shape=(3, 3))
    values = np.linspace(-2, 3, 9).reshape(3, 3)
    for rate in (0.5, 2.0, 8.0):
        actual = poisson_gaussian_prior(jnp.array(rate), config).log_prob(
            jnp.asarray(values)
        )
        expected = coefficient_logpdf(values, rate, config)
        np.testing.assert_allclose(actual, expected, atol=4e-6)


def test_analytic_evidence_and_mean_match_direct_latent_quadrature():
    config = SkyConfig(shape=(3, 3))
    y, rate = 0.7, 1.3

    def joint(x):
        return np.exp(coefficient_logpdf(x, rate, config)) * norm.pdf(
            y, loc=x, scale=config.noise_std
        )

    integral = quad(joint, -8, 10, epsabs=1e-11, epsrel=1e-10)[0]
    mean = (
        quad(lambda x: x * joint(x), -8, 10, epsabs=1e-11, epsrel=1e-10)[0] / integral
    )
    assert oracle_log_evidence(y, rate, config) == pytest.approx(
        np.log(integral), abs=1e-10
    )
    assert float(oracle_posterior_mean(y, rate, config)) == pytest.approx(
        mean, abs=1e-10
    )


def test_count_cutoff_converges_without_renormalization_drift():
    config = SkyConfig()
    y = simulate(config, 20260915)["observed_coefficients"]
    for rate in (0.5, 2.0, 8.0):
        assert oracle_log_evidence(y, rate, config, count_max=48) == pytest.approx(
            oracle_log_evidence(y, rate, config, count_max=96), abs=1e-11
        )


def test_joint_edgeworth_polynomial_matches_existing_field_operator():
    config = SkyConfig(shape=(3, 3))
    k3 = config.shot_variance**1.5 / np.sqrt(config.rate)
    k4 = config.shot_variance**2 / config.rate
    directions = jnp.eye(9).reshape(9, 3, 3)
    field = FieldEdgeworth(
        dist.Normal(jnp.zeros((3, 3)), 1.0).to_event(2),
        LowRankCumulants(directions, {3: jnp.full(9, k3), 4: jnp.full(9, k4)}),
        order=4,
    )
    values = jnp.asarray(np.random.default_rng(20).normal(size=(3, 3)))
    np.testing.assert_allclose(
        field.correction(values),
        edgeworth_field_correction(values, config.rate, config),
        atol=3e-6,
    )


def test_joint_edgeworth_has_an_explicit_negative_density_witness():
    config = SkyConfig()
    # At zero all H3 vanish; H4=3, H6=-15. For this 144-cell map
    # the order-4 correction is 1 - 3.84/rate, hence -0.92 at rate=2.
    assert float(
        edgeworth_field_correction(np.zeros(config.shape), config.rate, config)
    ) == pytest.approx(-0.92)


@pytest.mark.parametrize(
    "changes",
    [
        {"beam_neighbor": 0.125},
        {"count_max": 4},
        {"diffuse_variance": 0.0},
        {"rate": 0.0},
    ],
)
def test_invalid_experiment_models_are_refused(changes):
    with pytest.raises(ValueError):
        dataclasses.replace(SkyConfig(), **changes)
