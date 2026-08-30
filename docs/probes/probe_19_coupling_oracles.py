"""P1: rerun the dense oracles behind ``diagnose.coupling``.

The reference path is deliberately NumPy and deliberately forms
``inv(precision)`` for M6.  The production path uses block Cholesky factors
of precision and never forms covariance.  M1 additionally distinguishes the
observed Hessian from Fisher information; the two close numbers are not two
roundings of one calculation.

Measured 2026-08-29::

    M1 mixed second derivative                    0.0000000000000000e+00
    M1 observed-Hessian posterior correlation   -0.9942708160481132
    M1 Fisher canonical correlation              0.9940992046901392
    M5 diag(1,100), c=.99: kappa marginal     5025.125628140696
    M5 diag(100,1), c=.99: kappa marginal       1.990000000000
    M6 max |precision route - covariance route|  3.3306690738754696e-16
    M6 whitening floor                            9.2838294835945328e-16
    D74 c between eps and whitening floor         refused

Exit code 0 means the probe completed, never that a scientific verdict is
automatically accepted.

Run from the repository root:

    .venv/bin/python docs/probes/probe_19_coupling_oracles.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from bayesmith import det, observe, sample, trace
from bayesmith.diagnose.coupling import Measured, Refused, block_coupling


def graph_from_precision(precision, split, prior_std):
    prior_precision = np.diag(1.0 / prior_std**2)
    design = np.linalg.cholesky(precision - prior_precision).T

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
                jnp.zeros(precision.shape[0] - split),
                jnp.asarray(prior_std[split:]),
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


def covariance_cca(precision, split):
    covariance = np.linalg.inv(precision)
    root_x = np.linalg.cholesky(covariance[:split, :split])
    root_t = np.linalg.cholesky(covariance[split:, split:])
    whitened = np.linalg.solve(root_x, covariance[:split, split:])
    whitened = np.linalg.solve(root_t, whitened.T).T
    return np.linalg.svd(whitened, compute_uv=False)


def m1():
    data = 4.0 + 0.5 * np.random.default_rng(0).normal(size=20)
    mean = float(np.mean(data))
    point = np.zeros(2)
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

    mixed = jax.jacfwd(jax.jacrev(lambda z: z[0] + jnp.exp(z[1])))(jnp.asarray(point))[
        0, 1
    ]
    covariance = np.linalg.inv(hessian)
    correlation = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
    report = block_coupling(
        trace(model),
        ("x",),
        ("theta",),
        at={"x": jnp.asarray(point[0]), "theta": jnp.asarray(point[1])},
    )
    assert isinstance(report.correlation, Measured)
    print(f"M1 mixed second derivative                    {float(mixed):.16e}")
    print(f"M1 observed-Hessian posterior correlation   {correlation:.16f}")
    print(
        f"M1 Fisher canonical correlation              {report.correlation.value:.16f}"
    )


def m5():
    prior_std = np.array([10.0, 11.0, 12.0])
    for diagonal in ([1.0, 100.0], [100.0, 1.0]):
        theta = np.diag(diagonal)
        cross = np.array([[0.99, 0.0]]) @ np.linalg.cholesky(theta).T
        precision = np.block([[np.ones((1, 1)), cross], [cross.T, theta]])
        report = block_coupling(
            graph_from_precision(precision, 1, prior_std),
            ("first",),
            ("second",),
            at={"first": jnp.zeros(1), "second": jnp.zeros(2)},
        )
        print(
            f"M5 diag({diagonal[0]:g},{diagonal[1]:g}), c=.99: "
            f"kappa marginal {report.kappa_marg:.12f}"
        )


def m6():
    rng = np.random.default_rng(619)
    design = rng.normal(size=(9, 6))
    prior_std = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    precision = design.T @ design + np.diag(1.0 / prior_std**2)
    report = block_coupling(
        graph_from_precision(precision, 3, prior_std),
        ("first",),
        ("second",),
        at={"first": jnp.zeros(3), "second": jnp.zeros(3)},
    )
    difference = np.max(
        np.abs(report.canonical_correlations - covariance_cca(precision, 3))
    )
    expected_floor = (
        np.sqrt(
            np.linalg.cond(precision[:3, :3])
            * np.linalg.cond(precision[3:, 3:])
        )
        * np.finfo(np.float64).eps
    )
    assert isinstance(report.correlation, Measured)
    assert report.correlation.n_correlations == 3
    assert np.isclose(report.correlation.floor, expected_floor, rtol=2e-15)
    print(f"M6 max |precision route - covariance route|  {difference:.16e}")
    print(f"M6 whitening floor                            {expected_floor:.16e}")


def d74():
    within = np.diag([1.0, 1e12])
    cross = np.zeros((2, 2))
    cross[0, 0] = 1e-13
    precision = np.block([[within, cross], [cross.T, within]])
    report = block_coupling(
        graph_from_precision(
            precision,
            2,
            np.array([3.0, 4.0, 5.0, 6.0]),
        ),
        ("first",),
        ("second",),
        at={"first": jnp.zeros(2), "second": jnp.zeros(2)},
    )
    correlation = report.canonical_correlations[0]
    assert np.finfo(np.float64).eps < correlation
    assert isinstance(report.correlation, Refused)
    assert "noise floor" in report.correlation.reason
    print("D74 c between eps and whitening floor         refused")


if __name__ == "__main__":
    with jax.enable_x64(True):
        m1()
        m5()
        m6()
        d74()
