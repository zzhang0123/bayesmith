"""A joint map likelihood with manually specified connected cumulants.

Run: .venv/bin/python examples/inference/cumulant_field.py
Add --sample to exercise parameter inference. The observation is an explicit
toy array; this example does not pretend to simulate an Edgeworth field.
"""

import argparse

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from bayesmith import compile, log_joint, observe, sample, trace
from bayesmith.cumulants import FieldEdgeworth, LowRankCumulants, PeriodicNormal


def make_graph():
    coordinates = jnp.linspace(-1.0, 1.0, 8)
    x, y = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    directions = jnp.stack((jnp.ones_like(x) / 8, (x + y) / 8))
    observation = 0.1 + 0.15 * x + 0.08 * y
    fx, fy = jnp.meshgrid(jnp.fft.fftfreq(8), jnp.fft.fftfreq(8), indexing="ij")
    power = 0.8 + fx**2 + fy**2

    def likelihood(theta):
        reference = PeriodicNormal(jnp.ones_like(observation) * theta, power)
        cumulants = LowRankCumulants(
            directions,
            {
                3: jnp.stack((0.04 + theta / 20, jnp.array(0.02))),
                4: jnp.array([0.03, 0.01]),
            },
        )
        return FieldEdgeworth(reference, cumulants, order=4)

    def model():
        theta = sample("theta", lambda: dist.Uniform(-0.2, 0.2))
        observe("map", likelihood, theta, obs=observation)

    return trace(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    graph = make_graph()
    density = lambda theta: log_joint(graph, {"theta": theta})
    theta = jnp.array(0.05)
    print("joint log density:", float(jax.jit(density)(theta)))
    print("d log density / d theta:", float(jax.jit(jax.grad(density))(theta)))
    if args.sample:
        posterior = compile(graph).sample(
            jax.random.key(5), num_warmup=100, num_samples=200
        )
        print("theta posterior mean:", float(jnp.mean(posterior.samples["theta"])))


if __name__ == "__main__":
    main()
