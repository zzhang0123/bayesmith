"""Bounded numerical references, evaluated on the host outside inference.

Two-dimensional Gauss-Legendre rules integrate each curve's (A, alpha)
posterior. Refinement must agree within 0.001 posterior SD. These are numerical
references, not closed-form posteriors or a guarantee of bias reduction.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from .priors import TOP_LEVEL_BOUNDS

ORDERS = (128, 256)
REFINEMENT_TOLERANCE = 0.001
COORDINATES = ("A₁", "α₁", "A₂", "α₂")


def _bounds(channel):
    return [
        (
            TOP_LEVEL_BOUNDS["power_law"][name][0][channel],
            TOP_LEVEL_BOUNDS["power_law"][name][1][channel],
        )
        for name in ("amplitude", "alpha")
    ]


def _log_shape(alpha, logx, sigma):
    exponent = 2 * np.asarray(alpha)[..., None] * logx
    total = logsumexp(exponent, axis=-1)
    weights = np.exp(exponent - total[..., None])
    center = weights @ logx
    variance = np.sum(weights * (logx - center[..., None]) ** 2, axis=-1)
    return total + 0.5 * np.log(variance) - 2 * np.log(sigma)


def _map(x, data, channel, jeffreys, sigma):
    """Profile the conditional amplitude mode; refine all grid maxima and edges."""
    (alow, ahigh), (plow, phigh) = _bounds(channel)
    logx = np.log(x)

    def profile(alpha):
        basis = np.exp(np.asarray(alpha)[..., None] * logx)
        precision = np.sum(basis**2, axis=-1) / sigma**2
        score = basis @ data / sigma**2
        if jeffreys:
            hypotenuse = np.hypot(score, 2 * np.sqrt(precision))
            amplitude = np.where(
                score >= 0,
                (score + hypotenuse) / (2 * precision),
                2 / (hypotenuse - score),
            )
        else:
            amplitude = score / precision
        amplitude = np.clip(amplitude, alow, ahigh)
        residual = data - amplitude[..., None] * basis
        value = -0.5 * np.sum(residual**2, axis=-1) / sigma**2
        if jeffreys:
            value += np.log(amplitude) + _log_shape(alpha, logx, sigma)
        return value, amplitude

    grid = np.linspace(plow, phigh, 129)
    values, _ = profile(grid)
    peaks = (
        np.flatnonzero((values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:])) + 1
    )
    candidates = [plow, phigh, grid[int(np.argmax(values))]]
    brackets = [(grid[0], grid[1]), (grid[-2], grid[-1])]
    brackets.extend((grid[i - 1], grid[i + 1]) for i in peaks)
    for bracket in brackets:
        fit = minimize_scalar(
            lambda p: -profile(p)[0],
            bounds=bracket,
            method="bounded",
            options={"xatol": 1e-11},
        )
        if not fit.success:
            raise RuntimeError("Power-law profile MAP refinement failed.")
        candidates.append(fit.x)
    values, amplitudes = profile(np.asarray(candidates))
    best = int(np.argmax(values))
    return [float(amplitudes[best]), float(candidates[best])]


def _integrate(x, datasets, channel, jeffreys, order, sigma):
    nodes, weights = leggauss(order)
    axes = [
        (lo + (nodes + 1) * (hi - lo) / 2, weights * (hi - lo) / 2)
        for lo, hi in _bounds(channel)
    ]
    (amplitude, wa), (alpha, wp) = axes
    logx = np.log(x)
    basis = np.exp(alpha[:, None] * logx)
    precision = np.sum(basis**2, axis=-1) / sigma**2
    log_weight = np.log(wp[:, None]) + np.log(wa[None, :])
    if jeffreys:
        log_weight = (
            log_weight
            + _log_shape(alpha, logx, sigma)[:, None]
            + np.log(amplitude)[None, :]
        )
    means, deviations, normalizers = [], [], []
    # Bound temporary storage independently of the number of datasets.
    for start in range(0, len(datasets), 16):
        data = datasets[start : start + 16]
        ahat = (data @ basis.T / sigma**2) / precision
        residual = data[:, None, :] - ahat[:, :, None] * basis[None, :, :]
        loss = np.sum(residual**2, axis=-1) / sigma**2
        log_density = (
            -0.5
            * (
                loss[:, :, None]
                + precision[None, :, None]
                * (amplitude[None, None, :] - ahat[:, :, None]) ** 2
            )
            + log_weight
        )
        log_norm = logsumexp(log_density, axis=(1, 2))
        probability = np.exp(log_density - log_norm[:, None, None])
        marginal_a, marginal_p = probability.sum(axis=1), probability.sum(axis=2)
        mean_a, mean_p = marginal_a @ amplitude, marginal_p @ alpha
        sd_a = np.sqrt(np.sum(marginal_a * (amplitude - mean_a[:, None]) ** 2, axis=1))
        sd_p = np.sqrt(np.sum(marginal_p * (alpha - mean_p[:, None]) ** 2, axis=1))
        means.extend(np.column_stack([mean_a, mean_p]))
        deviations.extend(np.column_stack([sd_a, sd_p]))
        normalizers.extend(log_norm)
    return {
        "mean": np.asarray(means),
        "sd": np.asarray(deviations),
        "log_normalizer": np.asarray(normalizers),
    }


def _summaries(x, datasets, *, jeffreys):
    from .power_law import NOISE_SD

    x, datasets = np.asarray(x, dtype=float), np.asarray(datasets, dtype=float)
    if (
        x.ndim != 1
        or datasets.ndim != 3
        or datasets.shape[1:] != (2, x.size)
        or datasets.shape[0] == 0
        or not np.all(np.isfinite(datasets))
        or not np.all(np.isfinite(x))
        or np.any(x <= 0)
        or np.ptp(x) == 0
    ):
        raise ValueError(
            "Use finite datasets of shape (replicates,2,len(x)) and distinct positive x."
        )
    output = {name: [] for name in ("mean", "sd", "map")}
    mean_error, sd_error, log_norm_error = [], [], []
    finite_outputs = True
    for channel in range(2):
        data = datasets[:, channel]
        coarse, fine = [
            _integrate(x, data, channel, jeffreys, order, NOISE_SD) for order in ORDERS
        ]
        finite_outputs = finite_outputs and all(
            all(np.all(np.isfinite(value)) for value in rule.values())
            and np.all(rule["sd"] > 0)
            for rule in (coarse, fine)
        )
        for field in ("mean", "sd"):
            output[field].append(fine[field])
        output["map"].append(
            np.asarray([_map(x, d, channel, jeffreys, NOISE_SD) for d in data])
        )
        finite_outputs = finite_outputs and np.all(np.isfinite(output["map"][-1]))
        mean_error.extend((np.abs(fine["mean"] - coarse["mean"]) / fine["sd"]).ravel())
        sd_error.extend((np.abs(fine["sd"] - coarse["sd"]) / fine["sd"]).ravel())
        log_norm_error.extend(np.abs(fine["log_normalizer"] - coarse["log_normalizer"]))
    errors = np.array([np.max(mean_error), np.max(sd_error), np.max(log_norm_error)])
    return {
        **{k: np.concatenate(v, axis=1) for k, v in output.items()},
        "quadrature": {
            "orders": list(ORDERS),
            "finite_outputs": bool(finite_outputs),
            "passed": bool(
                finite_outputs
                and np.all(np.isfinite(errors))
                and np.all(errors < REFINEMENT_TOLERANCE)
            ),
            "max_mean_difference_sd": float(errors[0]),
            "max_relative_sd_difference": float(errors[1]),
            "max_log_normalizer_difference": float(errors[2]),
            "tolerance": REFINEMENT_TOLERANCE,
        },
    }


def posterior_summaries(x, data, *, jeffreys=False):
    result = _summaries(x, np.asarray(data)[None], jeffreys=jeffreys)
    return {
        "coordinates": list(COORDINATES),
        **{k: result[k][0].tolist() for k in ("map", "mean", "sd")},
        "quadrature": result["quadrature"],
        "method": "Refined 2D Gauss-Legendre integration; joint MAP by profiled optimization",
    }


def sampling_check(posterior, chain_shape, reference):
    from numpyro.diagnostics import effective_sample_size

    draws = np.stack(
        [
            posterior["amplitude"][:, 0],
            posterior["alpha"][:, 0],
            posterior["amplitude"][:, 1],
            posterior["alpha"][:, 1],
        ],
        axis=-1,
    )
    ess = np.minimum(
        draws.shape[0], effective_sample_size(draws.reshape(*chain_shape, 4))
    )
    mcse = np.asarray(reference["sd"]) / np.sqrt(ess)
    errors = np.abs(draws.mean(axis=0) - reference["mean"])
    return {
        "passed": bool(
            reference["quadrature"]["passed"]
            and np.all(np.isfinite(mcse))
            and np.all(errors <= 6 * mcse)
        ),
        "coordinates": list(COORDINATES),
        "mean_absolute_error": errors.tolist(),
        "mean_mcse": mcse.tolist(),
        "quadrature": reference["quadrature"],
        "criterion": "Refined quadrature passes; each sampled mean within six Monte Carlo SE of reference",
    }


def bias_experiment(truths, *, repeats=512, seed=912):
    import jax

    from .power_law import (
        NOISE_SD,
        OBSERVATIONS,
        design,
        gaussian_observation,
        mean_signal,
    )

    if repeats < 2:
        raise ValueError(
            "Bias Monte Carlo standard errors require at least two datasets."
        )
    x, channel = design()
    law = gaussian_observation(mean_signal(x, **truths, channel=channel))
    data = np.asarray(law.sample(jax.random.key(seed), (repeats,))).reshape(
        repeats, 2, OBSERVATIONS
    )
    truth = np.array(
        [
            truths["amplitude"][0],
            truths["alpha"][0],
            truths["amplitude"][1],
            truths["alpha"][1],
        ],
        dtype=float,
    )
    results = {
        prior: _summaries(x[:OBSERVATIONS], data, jeffreys=prior == "jeffreys")
        for prior in ("flat", "jeffreys")
    }
    rows = []
    for prior, result in results.items():
        for estimator in ("map", "mean"):
            errors = result[estimator] - truth
            rows.append(
                {
                    "prior": prior,
                    "estimator": estimator,
                    "bias": errors.mean(axis=0).tolist(),
                    "mcse": (errors.std(axis=0, ddof=1) / np.sqrt(repeats)).tolist(),
                    "rmse": np.sqrt(np.mean(errors**2, axis=0)).tolist(),
                }
            )
    differences = {}
    for estimator in ("map", "mean"):
        delta = results["jeffreys"][estimator] - results["flat"][estimator]
        differences[estimator] = {
            "mean": delta.mean(axis=0).tolist(),
            "mcse": (delta.std(axis=0, ddof=1) / np.sqrt(repeats)).tolist(),
        }
    return {
        "kind": "power_law_regression",
        "coordinates": list(COORDINATES),
        "repeats": repeats,
        "seed": seed,
        "observations_per_channel": OBSERVATIONS,
        "noise_sd": NOISE_SD,
        "truth": truth.tolist(),
        "rows": rows,
        "paired_jeffreys_minus_uniform": differences,
        "quadrature": {
            "passed": all(r["quadrature"]["passed"] for r in results.values()),
            "by_prior": {p: r["quadrature"] for p, r in results.items()},
        },
        "method": "Paired Gaussian forward samples; refined quadrature means and profile MAP; no MCMC",
    }
