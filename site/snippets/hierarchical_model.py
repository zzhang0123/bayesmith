"""A flowchart whose dependencies define a hierarchical probability model."""

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

import bayesmith as bs


def model(data):
    population_mean = bs.sample("population_mean", lambda: dist.Normal(0.0, 2.0))
    units = bs.plate("units", len(data))
    level = bs.sample(
        "level", lambda mean: dist.Normal(mean, 0.5), population_mean, plate=units
    )
    signal = bs.det("signal", lambda value: 2.0 * value, level, plate=units)
    bs.observe(
        "data", lambda mean: dist.Normal(mean, 0.2), signal, obs=data, plate=units
    )


def normal_log_density(value, mean, scale):
    """Independent scalar Gaussian formula, summed over repeated units."""
    return np.sum(
        -0.5 * np.log(2.0 * np.pi * scale**2) - 0.5 * ((value - mean) / scale) ** 2
    )


def main():
    data = jnp.array([0.5, 1.7, -0.3])
    graph = bs.trace(model, data)
    point = {"population_mean": jnp.asarray(0.3), "level": jnp.array([0.2, 0.8, -0.2])}
    actual = bs.log_joint(graph, point)
    mean, levels = (
        float(point["population_mean"]),
        np.asarray(point["level"], dtype=float),
    )
    expected = (
        normal_log_density(mean, 0.0, 2.0)
        + normal_log_density(levels, mean, 0.5)
        + normal_log_density(np.asarray(data, dtype=float), 2.0 * levels, 0.2)
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    print("Graph log joint:", float(actual))
    print("Independent hierarchical factorization:", expected)


if __name__ == "__main__":
    main()
