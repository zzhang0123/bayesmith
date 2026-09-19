"""Declare and score a small field likelihood; this is not a sampling claim."""

import jax.numpy as jnp
import numpyro.distributions as dist

from bayesmith import log_joint, observe, sample, trace
from bayesmith.cumulants import FieldEdgeworth, LowRankCumulants


def likelihood(theta):
    base = dist.Normal(jnp.ones((2, 2)) * theta, 1.0).to_event(2)
    cumulants = LowRankCumulants(
        jnp.array([[[0.5, 0.2], [0.1, 0.7]]]),
        {3: jnp.reshape(theta / 10, (1,)), 4: jnp.array([0.02])},
    )
    return FieldEdgeworth(base, cumulants, order=4)


def model():
    theta = sample("theta", lambda: dist.Uniform(-0.2, 0.2))
    observe("map", likelihood, theta, obs=jnp.array([[0.2, 0.1], [0.4, -0.1]]))


graph = trace(model)
print(log_joint(graph, {"theta": jnp.array(0.0)}))
