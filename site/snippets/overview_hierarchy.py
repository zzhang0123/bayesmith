"""Run with JAX_ENABLE_X64=1 python overview_hierarchy.py."""

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

import bayesmith as bs

A = jnp.array([[1.0, 0.3], [0.2, 1.0], [0.5, -0.4], [1.0, 1.0]])
DATA = jnp.array([0.4, 1.1, -0.2, 1.5])


def model(data):
    # A hyperparameter is an ordinary latent sample node.
    theta = bs.sample("theta", lambda: dist.Normal(0.0, 0.5))
    scale = bs.det("scale", jnp.exp, theta)

    # The distribution is constructed from the parent's current value.
    x = bs.sample("x", lambda sd: dist.Normal(jnp.zeros(2), sd).to_event(1), scale)
    design = bs.const("A", A)
    signal = bs.det("signal", lambda matrix, value: matrix @ value, design, x)

    # An observed distribution can have uncertain parameters too.
    sigma = bs.sample("sigma", lambda: dist.LogNormal(-1.0, 0.3))
    bs.observe("d", lambda mean, sd: dist.Normal(mean, sd), signal, sigma, obs=data)


def normal_log_density(value, mean, sd):
    """Independent Gaussian formula, including its scale normalization."""
    return np.sum(
        -0.5 * ((value - mean) / sd) ** 2 - np.log(sd) - 0.5 * np.log(2 * np.pi)
    )


def verify_density():
    """Check the displayed factorization at 27 points, not one fitted mode."""
    graph = bs.trace(model, DATA)
    errors = []
    for theta in (-0.7, 0.0, 0.5):
        for sigma in (0.2, 0.5, 1.1):
            for x in (np.zeros(2), np.array([-0.4, 0.7]), np.array([1.0, -0.3])):
                expected = (
                    normal_log_density(theta, 0.0, 0.5)
                    + normal_log_density(x, 0.0, np.exp(theta))
                    + normal_log_density(np.log(sigma), -1.0, 0.3)
                    - np.log(sigma)
                    + normal_log_density(np.asarray(DATA), np.asarray(A) @ x, sigma)
                )
                actual = float(
                    bs.log_joint(
                        graph,
                        {
                            "theta": jnp.asarray(theta),
                            "x": jnp.asarray(x),
                            "sigma": jnp.asarray(sigma),
                        },
                    )
                )
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                errors.append(abs(actual - expected))
    return max(errors)


if __name__ == "__main__":
    print(bs.compile(bs.trace(model, DATA)))
    print("27 density checks passed; maximum absolute error:", verify_density())
