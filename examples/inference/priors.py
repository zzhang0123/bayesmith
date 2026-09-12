"""Declared finite flat priors for demo root parameters, in model coordinates.

These bounds are part of each model, never fitted to a dataset. Descendant
random effects retain their conditional generative distributions.
"""

TOP_LEVEL_BOUNDS = {
    "power_law": {
        "amplitude": ([0.2] * 2, [3.0] * 2),
        "alpha": ([-2.0] * 2, [2.0] * 2),
    },
    "linear_gaussian": {"beta": ([-4.0] * 4, [4.0] * 4)},
    "exponential_decay": {
        "amplitude": ([0.0] * 2, [4.0] * 2),
        "rate": ([0.1] * 2, [2.5] * 2),
        "offset": ([-1.0] * 2, [1.0] * 2),
    },
    "hierarchical": {"population": (-3.0, 3.0)},
    "bernoulli": {"beta": ([-4.0] * 4, [4.0] * 4)},
    "multiplicative_noise": {
        "p_g": ([-0.5] * 2, [0.5] * 2),
        "p_n": ([0.1] * 2, [3.0] * 2),
    },
    "composed_process": {
        "power_amplitude": (0.1, 1.2),
        "nonlinear": ([0.25, 0.05], [0.75, 0.20]),
        "background": ([2.0, -1.0], [4.0, 1.0]),
        "gain": ([-0.25] * 2, [0.25] * 2),
        "sigma_w": (0.003, 0.06),
    },
}


def uniform_prior(case, name):
    import jax.numpy as jnp
    import numpyro.distributions as dist

    low, high = (jnp.asarray(v) for v in TOP_LEVEL_BOUNDS[case][name])
    return dist.Uniform(low, high).to_event(low.ndim)
