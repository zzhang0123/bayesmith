"""A hierarchy: population mean -> group means -> indexed repeated observations."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from bayesmith import const, det, observe, sample, trace

from .common import Demo, infer, samples, simulate_fixed
from .priors import uniform_prior

GROUPS = 8
GROUP_SD = 0.6


# 1. A probabilistic operator can itself depend on a random parent.
def population_prior():
    return uniform_prior("hierarchical", "population")


def group_prior(population):
    return dist.Normal(population, GROUP_SD).expand((GROUPS,)).to_event(1)


def gather(group_means, group_index):
    return group_means[group_index]


def gaussian_observation(location):
    return dist.Normal(location, 0.25)


# 2. A vector latent and an indexing operator, with a genuine stochastic edge.
def model(indices, data):
    index = const("group_index", indices)
    population = sample("population", population_prior)
    groups = sample("groups", group_prior, population)
    location = det("location", gather, groups, index, linear_in=("groups",))
    observe("obs", gaussian_observation, location, obs=data)


def run(seed=0, draws=2000, warmup=1000, *, graph_transform=None):
    group_key, sim_key, infer_key = jax.random.split(jax.random.key(seed), 3)
    index = jnp.repeat(jnp.arange(GROUPS), 40)
    population = jnp.array(0.8)
    # 3. Realise the random effects from their own conditional prior, then
    # generate observations. Do not silently set every effect to its mean.
    groups = group_prior(population).sample(group_key)
    truths = {"population": population, "groups": groups}
    template = trace(model, index, jnp.zeros(index.shape))
    data, simulation = simulate_fixed(template, model, truths, sim_key)
    graph = trace(model, index, data)
    if graph_transform is not None:
        graph = graph_transform(graph)
    # 4. Recover the realised effects AND the population hyperparameter.
    plan, posterior = infer(graph, model, infer_key, draws, warmup)
    s = samples(posterior)
    return Demo(
        "Hierarchical group measurements",
        graph,
        plan,
        posterior,
        simulation,
        truths,
        {
            "population": jnp.sqrt(population_prior().variance),
            "groups": jnp.sqrt(population_prior().variance + GROUP_SD**2),
        },
        index,
        data,
        gather(groups, index),
        s["groups"][:, index],
        note="Group measurements do not determine the population mean exactly; only eight groups inform it.",
    )
