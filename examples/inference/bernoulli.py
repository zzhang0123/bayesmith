"""Two covariates and their interaction -> four logits coefficients -> binary data."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from bayesmith import const, det, observe, sample, trace

from .common import Demo, infer, samples
from .priors import uniform_prior


def coefficient_prior():
    return uniform_prior("bernoulli", "beta")


def design_matrix(x1, x2):
    return jnp.column_stack([jnp.ones_like(x1), x1, x2, x1 * x2])


def linear_logits(X, beta):
    return X @ beta


def binary_observation(logits):
    return dist.Bernoulli(logits=logits)


def model(design, data):
    X = const("X", design)
    beta = sample("beta", coefficient_prior)
    logits = det("logits", linear_logits, X, beta, linear_in=("beta",))
    observe("obs", binary_observation, logits, obs=data)


def run(seed=0, draws=2000, warmup=1000, *, graph_transform=None):
    sim_key, infer_key = jax.random.split(jax.random.key(seed))
    axis = jnp.linspace(-2.0, 2.0, 20)
    grid1, grid2 = jnp.meshgrid(axis, axis, indexing="xy")
    x1, x2 = grid1.ravel(), grid2.ravel()
    X = design_matrix(x1, x2)
    truths = {"beta": jnp.array([-0.3, 1.1, -0.7, 0.35])}
    # The unified SimulationTask does not yet support Bernoulli; sample the
    # same declared observation operator, using only the forward key and truth.
    law = binary_observation(linear_logits(X, truths["beta"]))
    data = law.sample(sim_key)
    graph = trace(model, X, data)
    if graph_transform is not None:
        graph = graph_transform(graph)
    plan, posterior = infer(graph, model, infer_key, draws, warmup)
    probability = jax.nn.sigmoid(samples(posterior)["beta"] @ X.T)
    return Demo(
        "Two-feature Bernoulli regression",
        graph,
        plan,
        posterior,
        None,
        truths,
        {"beta": jnp.sqrt(coefficient_prior().variance)},
        x1,
        data,
        law.mean,
        probability,
        simulation_method="Declared Bernoulli operator.sample(key)",
        note="NUTS samples all four coefficients. The heatmap is a probability surface over two inputs; the observation operator generates binary outcomes.",
        view={
            "kind": "surface",
            "x2": x2.tolist(),
            "shape": [20, 20],
            "x_label": "Feature 1",
            "y_label": "Feature 2",
        },
    )
