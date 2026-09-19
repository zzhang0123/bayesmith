"""Large Campbell sky: diagonal Gaussian draws, joint and factorized IS.

Run: python -m examples.inference.campbell_sky_scaling --output runs/campbell-large
No dense pixel covariance or cumulant tensor is materialized. Factorization
is exact only for this deliberately independent source-coordinate model.
"""

import argparse
import dataclasses
import hashlib
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

from .campbell_plot import render_figures
from .campbell_sky import (
    SkyConfig,
    beam,
    cumulant_monte_carlo,
    deconvolve,
    oracle_log_evidence,
    oracle_posterior_mean,
    oracle_profile,
    simulate,
)


@jax.jit
def gaussian_draws(key, observed, standard_normals):
    """Diagonal GCR: m + sqrt(C) z; z's shape specifies the draw bank."""
    variance = 1 / (1 + 1 / 0.1**2)
    mean = observed / (1 + 0.1**2)
    return mean + jnp.sqrt(variance) * jax.random.normal(key, standard_normals.shape)


def draw_bank(key, observed, draws):
    # These constants are the declared scaling experiment's reference model.
    return gaussian_draws(key, jnp.asarray(observed), jnp.empty((draws, observed.size)))


def make_scores(samples, config, chunk=64):
    """Stream cells to bound mixture memory; retain only joint log weights.

    sum_j log mean_s r_sj and log mean_s prod_j r_sj are DISTINCT estimators.
    Exact source-coordinate independence licenses the former here.
    """
    samples = jnp.asarray(samples)
    draws, cells = samples.shape
    if cells % chunk:
        raise ValueError("cell count must be divisible by chunk")
    blocks = samples.reshape(draws, cells // chunk, chunk).transpose(1, 0, 2)
    count = jnp.arange(config.count_max + 1)

    @jax.jit
    def scores(rate):
        logp_count = count * jnp.log(rate) - rate - jax.scipy.special.gammaln(count + 1)
        logp_count -= jax.scipy.special.logsumexp(logp_count)
        centers = jnp.sqrt(config.shot_variance / rate) * (count - rate)

        def step(carry, x):
            local, joint, ess_min, ess_sum = carry
            target = jax.scipy.special.logsumexp(
                logp_count
                - 0.5 * (x[..., None] - centers) ** 2 / config.diffuse_variance
                - 0.5 * jnp.log(2 * jnp.pi * config.diffuse_variance),
                axis=-1,
            )
            reference = -0.5 * x**2 / config.variance - 0.5 * jnp.log(
                2 * jnp.pi * config.variance
            )
            ratio = target - reference
            normalizer = jax.scipy.special.logsumexp(ratio, axis=0)
            ess = jnp.exp(
                2 * normalizer - jax.scipy.special.logsumexp(2 * ratio, axis=0)
            )
            return (
                local + jnp.sum(normalizer - jnp.log(draws)),
                joint + jnp.sum(ratio, axis=1),
                jnp.minimum(ess_min, ess.min()),
                ess_sum + ess.sum(),
            ), None

        initial = (
            jnp.array(0.0),
            jnp.zeros(draws),
            jnp.array(float(draws)),
            jnp.array(0.0),
        )
        (local, joint, ess_min, ess_sum), _ = jax.lax.scan(step, initial, blocks)
        norm = jax.scipy.special.logsumexp(joint)
        joint_ess = jnp.exp(2 * norm - jax.scipy.special.logsumexp(2 * joint))
        return jnp.array(
            [
                local,
                norm - jnp.log(draws),
                joint_ess,
                jnp.exp(joint.max() - norm),
                ess_min,
                ess_sum / cells,
            ]
        )

    return scores


def maximize_grid(function, grid, values):
    index = int(np.argmax(values))
    if index in (0, len(grid) - 1):
        return float(grid[index])
    result = minimize_scalar(
        lambda rate: -function(rate),
        bounds=(grid[index - 1], grid[index + 1]),
        method="bounded",
        options={"xatol": 2e-4},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.x)


def run(output, draws=2048, side=128, grid_size=61):
    if side < 8 or side % 8 or draws < 2 or grid_size < 5:
        raise ValueError(
            "side must be a positive multiple of 8; draws>=2; grid_size>=5"
        )
    output.mkdir(parents=True, exist_ok=True)
    config = SkyConfig(shape=(side, side))
    scaling = []
    for width in (16, 64, 128, 256):
        obs = np.zeros(width * width)
        start = time.perf_counter()
        bank = draw_bank(jax.random.key(80), obs, 1024)
        jax.block_until_ready(bank)
        cold = time.perf_counter() - start
        timings = []
        for seed in (81, 82, 83):
            start = time.perf_counter()
            bank = draw_bank(jax.random.key(seed), obs, 1024)
            jax.block_until_ready(bank)
            timings.append(time.perf_counter() - start)
        scaling.append(
            {
                "side": width,
                "cells": width**2,
                "draws": 1024,
                "cold_seconds": cold,
                "warm_seconds_median": float(np.median(timings)),
                "warm_seconds": timings,
                "scalar_draws": 1024 * width**2,
                "bank_mib": bank.nbytes / 2**20,
            }
        )
        del bank
        print("Gaussian", scaling[-1], flush=True)
    sky = simulate(config, 20260915)
    observed = deconvolve(sky["observed_sky"], config).ravel()
    start = time.perf_counter()
    oracle = oracle_profile(observed, config)
    oracle_seconds = time.perf_counter() - start
    reference_evidence = float(
        np.sum(-(observed**2) / (2 * 1.01) - 0.5 * np.log(2 * np.pi * 1.01))
    )
    grid = np.linspace(*config.rate_bounds, grid_size)
    oracle_grid = np.array([oracle_log_evidence(observed, r, config) for r in grid])
    banks = []
    for seed in (71, 72):
        start = time.perf_counter()
        samples = draw_bank(jax.random.key(seed), observed, draws)
        jax.block_until_ready(samples)
        draw_seconds = time.perf_counter() - start
        score = make_scores(samples, config)
        start = time.perf_counter()
        jax.block_until_ready(score(jnp.asarray(config.rate)))
        compile_seconds = time.perf_counter() - start
        start = time.perf_counter()
        profile = np.array([np.asarray(score(jnp.asarray(rate))) for rate in grid])
        local_mode = maximize_grid(
            lambda r, score=score: float(score(jnp.asarray(r))[0]), grid, profile[:, 0]
        )
        joint_mode = maximize_grid(
            lambda r, score=score: float(score(jnp.asarray(r))[1]), grid, profile[:, 1]
        )
        fitted = np.asarray(score(jnp.asarray(local_mode)))
        truth_score = np.asarray(score(jnp.asarray(config.rate)))
        joint_fitted = np.asarray(score(jnp.asarray(joint_mode)))
        weight_seconds = time.perf_counter() - start
        entry = {
            "seed": seed,
            "draw_seconds": draw_seconds,
            "score_compile_seconds": compile_seconds,
            "profile_and_fit_seconds": weight_seconds,
            "factorized_map": local_mode,
            "joint_map": joint_mode,
            "joint_ess_at_truth": float(truth_score[2]),
            "joint_max_weight_at_truth": float(truth_score[3]),
            "joint_ess_at_joint_map": float(joint_fitted[2]),
            "local_min_ess_at_fit": float(fitted[4]),
            "local_mean_ess_at_fit": float(fitted[5]),
            "factorized_log_evidence_error_at_fit": float(
                float(fitted[0])
                + reference_evidence
                - oracle_log_evidence(observed, local_mode, config)
            ),
            "factorized_profile": (
                profile[:, 0].astype(float) + reference_evidence
            ).tolist(),
            "joint_profile": (
                profile[:, 1].astype(float) + reference_evidence
            ).tolist(),
        }
        banks.append(entry)
        print(
            "Fit",
            {k: v for k, v in entry.items() if not k.endswith("profile")},
            flush=True,
        )
    result = {
        "case": "campbell_sky",
        "kind": "analytic_validation",
        "config": dataclasses.asdict(config),
        "draws": draws,
        "grid": grid.tolist(),
        "oracle_profile": oracle_grid.tolist(),
        "oracle_map": oracle["mode"],
        "oracle_interval95": np.asarray(oracle["interval95"]).tolist(),
        "oracle_seconds": oracle_seconds,
        "scaling": scaling,
        "banks": banks,
        "poisson_tail_union_bound": float(
            observed.size * poisson.sf(config.count_max, config.rate_bounds[1])
        ),
        "cutoff_log_evidence_difference": oracle_log_evidence(
            observed, config.rate, config, count_max=96
        )
        - oracle_log_evidence(observed, config.rate, config),
        "environment": {
            "platform": platform.platform(),
            "jax": jax.__version__,
            "devices": str(jax.devices()),
            "x64": jax.config.jax_enable_x64,
        },
        "reference_sampler": "Closed-form diagonal GCR, specialized to variance=1 and noise_std=0.1; not a large-graph compiler benchmark",
        "timings": "Synchronized JAX CPU wall clock; cold includes compilation, warm median of 3; no NUTS comparison measured",
        "inference": "Uniform rate hyperprior on [0.5,8]; fixed Gaussian banks; joint and independence-factorized marginal MAP",
    }
    result["cumulants"] = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in cumulant_monte_carlo(config).items()
    }
    result["source_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            "campbell_sky.py",
            "campbell_sky_scaling.py",
            "campbell_plot.py",
            "campbell_distributions.py",
        )
    }
    source_directory = output / "source"
    source_directory.mkdir(exist_ok=True)
    for name in result["source_sha256"]:
        (source_directory / name).write_bytes(
            Path(__file__).with_name(name).read_bytes()
        )
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        output / "maps.npz",
        **sky,
        analytic_mean=beam(
            oracle_posterior_mean(observed, oracle["mode"], config).reshape(
                config.shape
            )
        ),
    )
    render_figures(result, output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/campbell-large"))
    parser.add_argument("--draws", type=int, default=2048)
    parser.add_argument("--side", type=int, default=128)
    parser.add_argument("--grid-size", type=int, default=61)
    args = parser.parse_args()
    run(args.output, args.draws, args.side, args.grid_size)


if __name__ == "__main__":
    main()
