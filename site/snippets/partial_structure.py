"""A hierarchy with exact group means. Run with: JAX_ENABLE_X64=1 python partial_structure.py"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

import bayesmith as bs

GROUPS, PER_GROUP = 6, 5
INDEX = np.repeat(np.arange(GROUPS), PER_GROUP)
PRIOR_SD, GROUP_SD, NOISE_SD = 2.0, 0.6, 0.25


def hierarchy(index, data):
    index = bs.const("group_index", index)
    population = bs.sample("population", lambda: dist.Normal(0.0, PRIOR_SD))
    groups = bs.sample(
        "groups",
        lambda mu: dist.Normal(mu * jnp.ones(GROUPS), GROUP_SD).to_event(1),
        population,
    )
    location = bs.det("location", lambda u, i: u[i], groups, index)
    bs.observe("obs", lambda loc: dist.Normal(loc, NOISE_SD), location, obs=data)


def simulate(seed=0, population=0.8):
    """Set a truth, draw group means from their prior, then draw data."""
    rng = np.random.default_rng(seed)
    groups = population + GROUP_SD * rng.standard_normal(GROUPS)
    data = groups[INDEX] + NOISE_SD * rng.standard_normal(INDEX.size)
    return {"population": population, "groups": groups}, data


def exact_posterior(data):
    """Closed-form posterior of (population, groups): the model is jointly Gaussian."""
    prior = np.full((GROUPS + 1, GROUPS + 1), PRIOR_SD**2)
    prior[1:, 1:] += GROUP_SD**2 * np.eye(GROUPS)
    design = np.zeros((INDEX.size, GROUPS + 1))
    design[np.arange(INDEX.size), 1 + INDEX] = 1.0
    precision = np.linalg.inv(prior) + design.T @ design / NOISE_SD**2
    covariance = np.linalg.inv(precision)
    mean = covariance @ design.T @ data / NOISE_SD**2
    return mean, np.sqrt(np.diag(covariance))


def compare(seed=1, num_warmup=1000, num_samples=4000):
    """Sample through the compiled plan and set it beside the exact posterior."""
    truth, data = simulate()
    plan = bs.compile(bs.trace(hierarchy, INDEX, jnp.asarray(data)))
    draws = plan.sample(
        jax.random.key(seed), num_warmup=num_warmup, num_samples=num_samples
    ).samples
    sampled = np.column_stack([draws["population"], draws["groups"]])
    exact_mean, exact_sd = exact_posterior(data)
    names = ["population"] + [f"groups[{g}]" for g in range(GROUPS)]
    truths = [truth["population"], *truth["groups"]]
    return [
        {
            "name": name,
            "truth": float(truths[i]),
            "exact_mean": float(exact_mean[i]),
            "sampled_mean": float(sampled[:, i].mean()),
            "exact_sd": float(exact_sd[i]),
            "sampled_sd": float(sampled[:, i].std()),
        }
        for i, name in enumerate(names)
    ]


if __name__ == "__main__":
    _, data = simulate()
    print(bs.compile(bs.trace(hierarchy, INDEX, jnp.asarray(data))))
    for row in compare():
        print(row)
