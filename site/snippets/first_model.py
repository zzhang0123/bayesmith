"""Run with: python first_model.py"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

import bayesmith as bs

DATA = jnp.array([0.8, 1.0, 1.2])


def model(data):
    level = bs.sample("level", lambda: dist.Normal(0.0, 2.0))
    bs.observe("signal", lambda x: dist.Normal(x, 0.5), level, obs=data)


def main():
    graph = bs.trace(model, DATA)
    plan = bs.compile(graph)
    print(plan)
    posterior = plan.sample(jax.random.key(7), num_samples=256)
    print("Posterior sample shape:", posterior.samples["level"].shape)
    print("Posterior mean estimate:", posterior.samples["level"].mean())


if __name__ == "__main__":
    main()
