"""Pointwise coupling between two latent blocks.

Every numerical truth here is dense NumPy algebra on a matrix written in this
file.  In particular the covariance oracle deliberately forms ``inv(F)``;
the implementation is required to take the other route, through precision
Cholesky factors, so agreement is evidence rather than shared machinery.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import det, observe, sample, trace
from bayesmith.diagnose.coupling import (
    Measured,
    Refused,
    _classify_correlation,
    _condition_number,
    block_coupling,
)


@pytest.fixture(autouse=True)
def _double_precision():
    """Construct graphs and take every numerical verdict in float64."""
    with jax.enable_x64(True):
        yield


def _linear_gaussian_graph(
    precision: np.ndarray,
    *,
    split: int,
    prior_std: np.ndarray,
):
    """A graph whose posterior precision is the caller's dense matrix.

    ``A.T @ A = precision - prior_precision`` is built only here, with
    ``np.linalg``.  The non-unit widths are part of every fixture: unit widths
    make the prior log determinant vanish and have hidden an upstream defect.
    """
    prior_precision = np.diag(1.0 / prior_std**2)
    likelihood_precision = precision - prior_precision
    design = np.linalg.cholesky(likelihood_precision).T
    n_second = precision.shape[0] - split

    def model():
        first = sample(
            "first",
            lambda: dist.Normal(
                jnp.zeros(split), jnp.asarray(prior_std[:split])
            ).to_event(1),
        )
        second = sample(
            "second",
            lambda: dist.Normal(
                jnp.zeros(n_second), jnp.asarray(prior_std[split:])
            ).to_event(1),
        )
        prediction = det(
            "prediction",
            lambda x, theta: jnp.asarray(design) @ jnp.concatenate((x, theta)),
            first,
            second,
            linear_in=("first", "second"),
        )
        observe(
            "data",
            lambda mu: dist.Normal(mu, 1.0).to_event(1),
            prediction,
            obs=jnp.zeros(precision.shape[0]),
        )

    return trace(model)


def _covariance_cca(precision: np.ndarray, split: int) -> np.ndarray:
    """Canonical correlations through the forbidden covariance route."""
    covariance = np.linalg.inv(precision)
    c_xx = covariance[:split, :split]
    c_xt = covariance[:split, split:]
    c_tt = covariance[split:, split:]
    root_x = np.linalg.cholesky(c_xx)
    root_t = np.linalg.cholesky(c_tt)
    whitened = np.linalg.solve(root_x, c_xt)
    whitened = np.linalg.solve(root_t, whitened.T).T
    return np.linalg.svd(whitened, compute_uv=False)


def _m1_additive_fixture():
    """M1: ``mu = x 1_20 + exp(theta)`` with the measured noisy data."""
    data = 4.0 + 0.5 * np.random.default_rng(0).normal(size=20)

    def model():
        x = sample("x", lambda: dist.Normal(0.0, 10.0))
        theta = sample("theta", lambda: dist.Normal(0.0, 1.0))
        prediction = det(
            "prediction",
            lambda a, t: a * jnp.ones(20) + jnp.exp(t),
            x,
            theta,
        )
        observe(
            "data",
            lambda mu: dist.Normal(mu, 0.5).to_event(1),
            prediction,
            obs=jnp.asarray(data),
        )

    # Independent Newton solve on the two handwritten derivatives.  This is
    # the exact observed Hessian, not Fisher information: their small, real
    # difference is the M1/M2 regression pin.
    mean = float(np.mean(data))
    point = np.array([0.0, 0.0])
    for _ in range(20):
        x, theta = point
        exponential = np.exp(theta)
        residual = x + exponential - mean
        gradient = np.array(
            [80.0 * residual + x / 100.0, 80.0 * residual * exponential + theta]
        )
        hessian = np.array(
            [
                [80.01, 80.0 * exponential],
                [
                    80.0 * exponential,
                    80.0 * (exponential**2 + residual * exponential) + 1.0,
                ],
            ]
        )
        step = np.linalg.solve(hessian, gradient)
        point -= step
        if np.max(np.abs(step)) < 1e-14:
            break
    return trace(model), data, point, hessian


def test_m1_additivity_has_no_cheap_certificate_and_the_large_coupling_survives():
    graph, _, point, observed_hessian = _m1_additive_fixture()

    mixed = jax.jacfwd(jax.jacrev(lambda z: z[0] + jnp.exp(z[1])))(jnp.asarray(point))[
        0, 1
    ]
    assert float(mixed) == 0.0

    covariance = np.linalg.inv(observed_hessian)
    true_correlation = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
    assert true_correlation == pytest.approx(-0.9942708160481082, abs=1e-14)

    report = block_coupling(
        graph,
        ("x",),
        ("theta",),
        at={"x": jnp.asarray(point[0]), "theta": jnp.asarray(point[1])},
    )
    assert isinstance(report.correlation, Measured)
    assert report.correlation.value == pytest.approx(0.9940992046901365, abs=2e-13)
    assert report.blind_to == ("gaussian-only",)


@pytest.mark.parametrize(
    ("theta_precision", "expected"),
    [
        (np.diag([1.0, 100.0]), 100.0 / (1.0 - 0.99**2)),
        (np.diag([100.0, 1.0]), 100.0 * (1.0 - 0.99**2)),
    ],
)
def test_m5_marginal_condition_is_measured_not_inferred_from_c(
    theta_precision, expected
):
    root = np.linalg.cholesky(theta_precision)
    cross = np.array([[0.99, 0.0]]) @ root.T
    precision = np.block([[np.ones((1, 1)), cross], [cross.T, theta_precision]])
    prior_std = np.array([10.0, 11.0, 12.0])
    graph = _linear_gaussian_graph(precision, split=1, prior_std=prior_std)

    report = block_coupling(
        graph,
        ("first",),
        ("second",),
        at={"first": jnp.zeros(1), "second": jnp.zeros(2)},
    )
    assert isinstance(report.correlation, Measured)
    assert report.correlation.value == pytest.approx(0.99, abs=2e-14)
    assert report.kappa_marg == pytest.approx(expected, rel=2e-13)


def test_m6_precision_cholesky_matches_an_independent_covariance_oracle():
    rng = np.random.default_rng(619)
    design = rng.normal(size=(9, 6))
    prior_std = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    precision = design.T @ design + np.diag(1.0 / prior_std**2)
    graph = _linear_gaussian_graph(precision, split=3, prior_std=prior_std)

    report = block_coupling(
        graph,
        ("first",),
        ("second",),
        at={"first": jnp.zeros(3), "second": jnp.zeros(3)},
    )
    expected = _covariance_cca(precision, split=3)
    assert report.canonical_correlations == pytest.approx(expected, abs=1e-15)
    assert np.max(np.abs(report.canonical_correlations - expected)) <= 1e-15
    assert isinstance(report.correlation, Measured)
    assert report.correlation.n_correlations == 3
    expected_floor = (
        np.sqrt(
            np.linalg.cond(precision[:3, :3])
            * np.linalg.cond(precision[3:, 3:])
        )
        * np.finfo(np.float64).eps
    )
    assert report.correlation.floor == pytest.approx(
        expected_floor, rel=2e-15, abs=0.0
    )
    assert report.kappa_cond == pytest.approx(np.linalg.cond(precision[3:, 3:]))
    assert report.kappa_joint == pytest.approx(np.linalg.cond(precision))


def test_at_is_required_because_this_module_does_not_choose_a_mode():
    precision = np.array([[1.0, 0.2], [0.2, 2.0]])
    graph = _linear_gaussian_graph(precision, split=1, prior_std=np.array([3.0, 4.0]))
    with pytest.raises(TypeError, match="at"):
        block_coupling(graph, ("first",), ("second",))


def test_a_value_equal_to_the_noise_floor_is_refused_not_reported_low():
    verdict = _classify_correlation(0.25, floor=0.25, n_correlations=1)
    assert isinstance(verdict, Refused)
    assert verdict.verdict == "refused"
    assert verdict.verdict != "low"


def test_a_correlation_below_the_floor_is_refused_in_the_public_report():
    precision = np.diag([2.0, 5.0])
    graph = _linear_gaussian_graph(precision, split=1, prior_std=np.array([2.0, 3.0]))
    report = block_coupling(
        graph,
        ("first",),
        ("second",),
        at={"first": jnp.zeros(1), "second": jnp.zeros(1)},
    )
    assert isinstance(report.correlation, Refused)
    assert report.correlation.verdict == "refused"
    assert "noise floor" in report.correlation.reason


def test_d74_floor_refuses_a_value_between_bare_eps_and_whitening_noise():
    within = np.diag([1.0, 1e12])
    cross = np.zeros((2, 2))
    cross[0, 0] = 1e-13
    precision = np.block([[within, cross], [cross.T, within]])
    graph = _linear_gaussian_graph(
        precision,
        split=2,
        prior_std=np.array([3.0, 4.0, 5.0, 6.0]),
    )

    report = block_coupling(
        graph,
        ("first",),
        ("second",),
        at={"first": jnp.zeros(2), "second": jnp.zeros(2)},
    )

    assert np.finfo(np.float64).eps < report.canonical_correlations[0]
    assert isinstance(report.correlation, Refused)
    assert "noise floor" in report.correlation.reason


def test_condition_number_uses_distinct_sentinels_for_singular_and_nonfinite():
    singular = _condition_number(np.diag([0.0, 1.0]))
    nonfinite = _condition_number(np.diag([1.0, np.inf]))
    assert np.isinf(singular)
    assert np.isnan(nonfinite)


def test_roundoff_perfect_correlation_is_refused_when_conditioning_is_infinite():
    def model():
        first = sample("first", lambda: dist.Normal(0.0, 1e7))
        second = sample("second", lambda: dist.Normal(0.0, 1e7))
        prediction = det("prediction", lambda x, y: 10.0 * (x + y), first, second)
        observe("data", lambda mu: dist.Normal(mu, 1.0), prediction, obs=0.0)

    report = block_coupling(
        trace(model),
        ("first",),
        ("second",),
        at={"first": jnp.asarray(0.0), "second": jnp.asarray(0.0)},
    )

    assert np.isinf(report.kappa_marg)
    assert isinstance(report.correlation, Refused)
    assert "one" in report.correlation.reason


def test_map_and_coupling_share_one_refused_verdict_type():
    from bayesmith.diagnose.map import Refused as MapRefused

    assert MapRefused is Refused
