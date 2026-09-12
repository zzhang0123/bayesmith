"""Four basis coefficients -> matrix product -> Gaussian observations."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from bayesmith import const, det, observe, sample, trace

from .common import Demo, infer, samples, simulate_fixed
from .priors import uniform_prior


def coefficient_prior():
    return uniform_prior("linear_gaussian", "beta")


def basis_design(x):
    return jnp.column_stack(
        [jnp.ones_like(x), x, jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x)]
    )


def linear_mean(X, beta):
    return X @ beta


def gaussian_observation(location):
    return dist.Normal(location, 0.25)


def model(design, data):
    X = const("X", design)
    beta = sample("beta", coefficient_prior)
    location = det("location", linear_mean, X, beta, linear_in=("beta",))
    observe("obs", gaussian_observation, location, obs=data)


def analytic_check(x, data, posterior_samples, chain_shape=None):
    """Independent Gaussian reference with negligible, explicitly bounded truncation.

    Rebuild the design from raw x, not basis_design() or the traced graph.
    The oracle therefore catches a changed operator shared by both directions.
    """
    x = np.asarray(x)
    design = np.column_stack([np.ones_like(x), x, np.sin(np.pi * x), np.cos(np.pi * x)])
    covariance = np.linalg.inv(design.T @ design / 0.25**2)
    mean = covariance @ design.T @ np.asarray(data) / 0.25**2
    reference_sd = np.sqrt(np.diag(covariance))
    values = np.asarray(posterior_samples["beta"])
    from scipy.special import ndtr

    from .priors import TOP_LEVEL_BOUNDS

    low, high = np.asarray(TOP_LEVEL_BOUNDS["linear_gaussian"]["beta"])
    outside_mass_bound = float(
        np.sum(ndtr((low - mean) / reference_sd) + ndtr((mean - high) / reference_sd))
    )
    n = len(values)
    n_variance = n
    if chain_shape is not None:
        from bayesmith.dispatch.execute import chain_diagnostics

        n = min(
            n,
            chain_diagnostics(posterior_samples, num_chains=chain_shape[0])["beta"].ess,
        )
        n_variance = min(
            len(values),
            chain_diagnostics(
                {"squared_deviation": (values - mean) ** 2}, num_chains=chain_shape[0]
            )["squared_deviation"].ess,
        )
    mean_z = np.abs(values.mean(axis=0) - mean) / (reference_sd / np.sqrt(n))
    # Variance has a different autocorrelation time from the mean. Estimate
    # its MCSE from squared deviations, then apply the square-root delta method.
    variance_mcse = ((values - mean) ** 2).std(axis=0, ddof=1) / np.sqrt(n_variance)
    sd_mcse = variance_mcse / (2 * reference_sd)
    sd_z = np.abs(values.std(axis=0, ddof=1) - reference_sd) / sd_mcse
    return {
        "passed": bool(
            outside_mass_bound < 1e-12 and np.all(mean_z < 6) and np.all(sd_z < 6)
        ),
        "reference": "unbounded_flat_Gaussian_with_negligible_truncation_bound",
        "outside_prior_box_mass_upper_bound": outside_mass_bound,
        "effective_draws_mean": float(n),
        "effective_draws_variance": float(n_variance),
        "sd_mcse_delta_method": sd_mcse.tolist(),
        "mean_standard_errors": mean_z.tolist(),
        "sd_standard_errors": sd_z.tolist(),
        "reference_mean": mean.tolist(),
        "reference_covariance": covariance.tolist(),
        "standard_error_limit": 6,
    }


def run(seed=0, draws=2000, warmup=1000, *, proposals=(), initialization=None):
    sim_key, infer_key = jax.random.split(jax.random.key(seed))
    x = jnp.linspace(-1.0, 1.0, 96)
    X = basis_design(x)
    truths = {"beta": jnp.array([-0.4, 1.6, 0.55, -0.35])}
    template = trace(model, X, jnp.zeros_like(x))
    data, simulation = simulate_fixed(template, model, truths, sim_key)
    graph = trace(model, X, data)
    plan, posterior = infer(graph, model, infer_key, draws, warmup,
                            proposals=proposals, initialization=initialization)
    signal = samples(posterior)["beta"] @ X.T
    return Demo(
        "Four-dimensional linear Gaussian regression",
        graph,
        plan,
        posterior,
        simulation,
        truths,
        {"beta": jnp.sqrt(coefficient_prior().variance)},
        x,
        data,
        linear_mean(X, truths["beta"]),
        signal,
        note="The likelihood is linear Gaussian. The finite flat prior truncates the Gaussian posterior, so the current compiler uses NUTS. The independent Gaussian oracle also checks that truncation mass is negligible for this dataset.",
        view={"kind": "curve", "x_label": "Input x", "basis": np.asarray(X).tolist()},
    )
