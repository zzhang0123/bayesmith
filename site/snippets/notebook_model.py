"""The hierarchical notebook's model. Run with: JAX_ENABLE_X64=1 python notebook_model.py"""

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

import bayesmith as bs

INDICES = np.repeat(np.arange(8), 40)
# Placeholder observations: the routes in the plan do not depend on their values.
DATA = jnp.asarray(np.random.default_rng(0).normal(size=INDICES.size))


def model(indices, data):
    """As recorded in the notebook run."""
    index = bs.const("group_index", indices)
    population = bs.sample("population", lambda: dist.Uniform(-3.0, 3.0))
    groups = bs.sample(
        "groups",
        lambda mu: dist.Normal(mu, 0.6).expand((8,)).to_event(1),
        population,
    )
    location = bs.det(
        "location", lambda u, i: u[i], groups, index, linear_in=("groups",)
    )
    bs.observe("obs", lambda loc: dist.Normal(loc, 0.25), location, obs=data)


if __name__ == "__main__":
    print(bs.compile(bs.trace(model, INDICES, DATA)))
