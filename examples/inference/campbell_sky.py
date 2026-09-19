"""A beam-smoothed Poisson sky with analytic connected cumulants.

Point sources lie on a periodic Cartesian grid, with independent Poisson
counts per cell and identical positive amplitudes. A Gaussian diffuse field
makes the latent prior continuous, allowing Gaussian importance proposals.
The shot amplitude sqrt(v_shot / rate) and mean subtraction hold all first
and second cumulants fixed while the rate controls non-Gaussianity.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, logsumexp
from scipy.stats import poisson

from bayesmith import observe, sample, trace


@dataclass(frozen=True)
class SkyConfig:
    shape: tuple[int, int] = (12, 12)
    rate: float = 2.0
    shot_variance: float = 0.8
    diffuse_variance: float = 0.2
    noise_std: float = 0.1
    beam_neighbor: float = 0.1
    rate_bounds: tuple[float, float] = (0.5, 8.0)
    count_max: int = 48

    def __post_init__(self):
        if len(self.shape) != 2 or min(self.shape) < 3:
            raise ValueError("shape must be a 2D grid with both sides >= 3")
        if not 0 < self.beam_neighbor < 0.125:
            raise ValueError("beam_neighbor must be in (0, 1/8) for invertibility")
        if (
            not self.shot_variance > 0
            or not self.diffuse_variance > 0
            or not self.noise_std > 0
        ):
            raise ValueError("shot/diffuse variances and noise_std must be positive")
        lo, hi = self.rate_bounds
        if not 0 < lo < hi or not lo <= self.rate <= hi:
            raise ValueError("positive rate bounds must contain the injected rate")
        if not isinstance(self.count_max, int) or self.count_max < 1:
            raise ValueError("count_max must be a positive integer")
        if poisson.sf(self.count_max, hi) > 1e-12:
            raise ValueError("count cutoff leaves more than 1e-12 Poisson tail mass")

    @property
    def variance(self):
        return self.shot_variance + self.diffuse_variance


def beam(values, neighbor=0.1):
    """Positive, unit-sum, periodic 5-point beam; acts on the last two axes."""
    return (1 - 4 * neighbor) * values + neighbor * sum(
        np.roll(values, shift, axis=axis) for axis in (-2, -1) for shift in (-1, 1)
    )


def beam_eigenvalues(config):
    fy = 2 * np.pi * np.fft.fftfreq(config.shape[0])
    fx = 2 * np.pi * np.fft.fftfreq(config.shape[1])
    return (
        1
        - 4 * config.beam_neighbor
        + 2 * config.beam_neighbor * (np.cos(fy[:, None]) + np.cos(fx[None, :]))
    )


def deconvolve(values, config):
    return np.fft.ifft2(np.fft.fft2(values) / beam_eigenvalues(config)).real


def beam_matrix(config):
    n = int(np.prod(config.shape))
    return (
        beam(np.eye(n).reshape((n,) + config.shape), config.beam_neighbor)
        .reshape(n, n)
        .T
    )


def beam_rows(config, pixels):
    """Selected symmetric beam rows, without allocating an n-by-n matrix."""
    indices = np.asarray(pixels, dtype=int)
    impulses = np.zeros((len(indices), int(np.prod(config.shape))))
    impulses[np.arange(len(indices)), indices] = 1
    return beam(
        impulses.reshape((len(indices),) + config.shape), config.beam_neighbor
    ).reshape(impulses.shape)


def connected_cumulant(config, rate, pixels):
    """Campbell: K_r(i1,...,ir) = lambda*a^r*sum_j prod_l B[il,j].

    Gaussian diffuse covariance contributes at r=2 only; r=1 vanishes after
    the specified rate-dependent centering. Pixel indices are flattened.
    """
    order = len(pixels)
    if order < 1:
        raise ValueError("at least one pixel is required")
    if order == 1:
        return 0.0
    multiplier = (
        config.variance
        if order == 2
        else config.shot_variance ** (order / 2) * rate ** (1 - order / 2)
    )
    rows = beam_rows(config, pixels)
    return float(multiplier * np.sum(np.prod(rows, axis=0)))


def simulate(config, seed):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(config.rate, size=config.shape)
    coefficients = np.sqrt(config.shot_variance / config.rate) * (counts - config.rate)
    coefficients += np.sqrt(config.diffuse_variance) * rng.normal(size=config.shape)
    observed_coefficients = coefficients + config.noise_std * rng.normal(
        size=config.shape
    )
    return {
        "counts": counts,
        "coefficients": coefficients,
        "sky": beam(coefficients, config.beam_neighbor),
        "observed_sky": beam(observed_coefficients, config.beam_neighbor),
        "observed_coefficients": observed_coefficients,
    }


def poisson_gaussian_prior(rate, config):
    """Normalized finite mixture; omitted full-Poisson tail is bounded above.

    The generator uses untruncated Poisson counts. Numerics condition the
    count on k<=count_max; cutoff convergence is checked independently.
    """
    count = jnp.arange(config.count_max + 1, dtype=jnp.result_type(rate, float))
    logits = dist.Poisson(rate).log_prob(count)
    means = jnp.sqrt(config.shot_variance / rate) * (count - rate)
    return dist.MixtureSameFamily(
        dist.Categorical(logits=logits),
        dist.Normal(means, jnp.sqrt(config.diffuse_variance)),
    ).expand(config.shape)


def make_graph(observed_coefficients, config, rate=None):
    """Reference and target share the same linear coefficient observation.

    In map coordinates the noise covariance is sigma_n^2 B B^T. Inverting
    the known beam makes it diagonal. The fixed map-coordinate Jacobian
    cancels between the target and reference; no modes are discarded.
    """

    def model():
        prior = (
            dist.Normal(jnp.zeros(config.shape), jnp.sqrt(config.variance))
            if rate is None
            else poisson_gaussian_prior(rate, config)
        )
        coefficients = sample("coefficients", lambda: prior)
        observe(
            "data",
            lambda x: dist.Normal(x, config.noise_std),
            coefficients,
            obs=observed_coefficients,
        )

    return trace(model)


def coefficient_logpdf(values, rate, config, *, include_noise=False, count_max=None):
    """Independent NumPy/SciPy Gaussian-Poisson sum, no NumPyro/JAX calls."""
    count = np.arange((config.count_max if count_max is None else count_max) + 1)
    logits = count * np.log(rate) - rate - gammaln(count + 1)
    logits -= logsumexp(logits)
    means = np.sqrt(config.shot_variance / rate) * (count - rate)
    variance = config.diffuse_variance + (config.noise_std**2 if include_noise else 0)
    terms = (
        logits
        - 0.5 * (np.asarray(values)[..., None] - means) ** 2 / variance
        - 0.5 * np.log(2 * np.pi * variance)
    )
    return logsumexp(terms, axis=-1)


def oracle_log_evidence(observed_coefficients, rate, config, *, count_max=None):
    # Analytically integrate the Gaussian coefficient conditional on each
    # Poisson count. Independence leaves a product of 1D convolutions.
    return float(
        coefficient_logpdf(
            observed_coefficients, rate, config, include_noise=True, count_max=count_max
        ).sum()
    )


def oracle_posterior_mean(observed_coefficients, rate, config):
    count = np.arange(config.count_max + 1)
    logits = count * np.log(rate) - rate - gammaln(count + 1)
    means = np.sqrt(config.shot_variance / rate) * (count - rate)
    y = np.asarray(observed_coefficients)[..., None]
    noise_variance = config.noise_std**2
    logits = logits - 0.5 * (y - means) ** 2 / (
        config.diffuse_variance + noise_variance
    )
    weights = np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))
    component_means = (config.diffuse_variance * y + noise_variance * means) / (
        config.diffuse_variance + noise_variance
    )
    return np.sum(weights * component_means, axis=-1)


def oracle_profile(observed_coefficients, config, size=401):
    grid = np.linspace(*config.rate_bounds, size)
    scores = np.array(
        [oracle_log_evidence(observed_coefficients, r, config) for r in grid]
    )
    # Locate the global grid maximum before bounded scalar refinement; the
    # parameter family can have several stationary points.
    index = int(np.argmax(scores))
    left, right = grid[max(0, index - 1)], grid[min(size - 1, index + 1)]
    fit = minimize_scalar(
        lambda r: -oracle_log_evidence(observed_coefficients, r, config),
        bounds=(left, right),
        method="bounded",
    )
    candidates = [float(fit.x), float(grid[0]), float(grid[-1])]
    mode = max(
        candidates, key=lambda r: oracle_log_evidence(observed_coefficients, r, config)
    )
    density = np.exp(scores - scores.max())
    cdf = cumulative_trapezoid(density, grid, initial=0)
    cdf /= cdf[-1]
    return {
        "grid": grid,
        "log_evidence": scores,
        "mode": mode,
        "interval95": np.interp([0.025, 0.975], cdf, grid),
    }


def edgeworth_field_correction(coefficients, rate, config):
    """Exact order-4 *joint* Edgeworth polynomial in this independent basis.

    This includes the i!=j K3_i*K3_j terms, not a product of per-pixel
    truncated factors. A small-map test compares it with FieldEdgeworth.
    """
    z = np.asarray(coefficients) / np.sqrt(config.variance)
    h3 = z**3 - 3 * z
    h4 = z**4 - 6 * z**2 + 3
    h6 = z**6 - 15 * z**4 + 45 * z**2 - 15
    s3 = h3.sum(axis=(-2, -1))
    k3 = config.shot_variance**1.5 / np.sqrt(rate) / config.variance**1.5
    k4 = config.shot_variance**2 / rate / config.variance**2
    return (
        1
        + k3 * s3 / 6
        + k4 * h4.sum(axis=(-2, -1)) / 24
        + k3**2 * (h6.sum(axis=(-2, -1)) + s3**2 - (h3**2).sum(axis=(-2, -1))) / 72
    )


def cumulant_monte_carlo(config, seed=902, batches=48, per_batch=2048):
    """Independent local sky realizations and unbiased joint cumulants.

    Only source cells contributing to either of two neighboring map pixels
    are needed. Batch-to-batch scatter supplies Monte Carlo standard errors.
    """
    rng = np.random.default_rng(seed)
    matrix = beam_rows(config, [0, 1])
    matrix = matrix[:, np.any(matrix != 0, axis=0)]
    shape = (batches, per_batch, matrix.shape[1])
    coefficient = np.sqrt(config.shot_variance / config.rate) * (
        rng.poisson(config.rate, size=shape) - config.rate
    )
    coefficient += np.sqrt(config.diffuse_variance) * rng.normal(size=shape)
    maps = coefficient @ matrix.T
    u, v = maps[..., 0], maps[..., 1]
    u = u - u.mean(axis=1, keepdims=True)
    v = v - v.mean(axis=1, keepdims=True)
    n = per_batch
    m20, m11 = (u * u).mean(axis=1), (u * v).mean(axis=1)
    third = n * n / ((n - 1) * (n - 2))
    fourth = n * n / ((n - 1) * (n - 2) * (n - 3))
    estimates = np.stack(
        [
            n * m20 / (n - 1),
            third * (u**3).mean(axis=1),
            fourth * ((n + 1) * (u**4).mean(axis=1) - 3 * (n - 1) * m20**2),
            third * (u * u * v).mean(axis=1),
            fourth * ((n + 1) * (u**3 * v).mean(axis=1) - 3 * (n - 1) * m20 * m11),
        ],
        axis=1,
    )
    combinations = [(0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 1), (0, 0, 0, 1)]
    theory = np.array(
        [connected_cumulant(config, config.rate, p) for p in combinations]
    )
    mean = estimates.mean(axis=0)
    se = estimates.std(axis=0, ddof=1) / np.sqrt(batches)
    return {
        "names": ["K00", "K000", "K0000", "K001", "K0001"],
        "theory": theory,
        "estimate": mean,
        "standard_error": se,
        "z_score": (mean - theory) / se,
        "realizations": batches * per_batch,
    }
