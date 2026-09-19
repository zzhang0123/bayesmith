"""Run and plot the Campbell-sky reference-reweighting experiment.

.venv/bin/python -m examples.inference.campbell_sky_validation
Artifacts default to runs/campbell-sky/. No precision configuration is changed.
"""

import argparse
import dataclasses
import importlib.metadata
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial import HermiteE, Polynomial
from scipy.stats import poisson

from bayesmith import compile
from bayesmith.cumulants import FieldEdgeworth, LowRankCumulants
from bayesmith.distributions import EdgeworthExpansion
from bayesmith.reweight import PosteriorReweighting
from examples.inference.campbell_sky import (
    SkyConfig,
    beam,
    coefficient_logpdf,
    cumulant_monte_carlo,
    deconvolve,
    edgeworth_field_correction,
    make_graph,
    oracle_log_evidence,
    oracle_posterior_mean,
    oracle_profile,
    simulate,
)


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def edgeworth_audit(samples, config):
    k3 = config.shot_variance**1.5 / np.sqrt(config.rate)
    k4 = config.shot_variance**2 / config.rate
    polynomial = HermiteE(
        [
            1,
            0,
            0,
            k3 / (6 * config.variance**1.5),
            k4 / (24 * config.variance**2),
            0,
            k3 * k3 / (72 * config.variance**3),
        ]
    ).convert(kind=Polynomial)
    roots = polynomial.deriv().roots()
    stationary = roots.real[np.abs(roots.imag) < 1e-8]
    minimum_at = float(stationary[np.argmin(polynomial(stationary))])
    joint = edgeworth_field_correction(samples, config.rate, config)
    # A concrete witness in the actual field implementation, even when every
    # sampled coefficient happens to lie in a positive part of the expansion.
    import numpyro.distributions as dist

    cells = int(np.prod(config.shape))
    field = FieldEdgeworth(
        dist.Normal(jnp.zeros(config.shape), np.sqrt(config.variance)).to_event(2),
        LowRankCumulants(
            jnp.eye(cells).reshape((cells,) + config.shape),
            {3: jnp.full(cells, k3), 4: jnp.full(cells, k4)},
        ),
        order=4,
    )
    zero_correction = float(field.correction(jnp.zeros(config.shape)))
    return {
        "order": 4,
        "scalar_global_minimum": float(polynomial(minimum_at)),
        "scalar_minimum_at_standardized_coefficient": minimum_at,
        "joint_negative_weight_fraction": float(np.mean(joint < 0)),
        "joint_minimum_on_reference_bank": float(np.min(joint)),
        "joint_correction_at_zero_field_actual_operator": zero_correction,
        "joint_correction_at_zero_field_analytic": float(
            edgeworth_field_correction(np.zeros(config.shape), config.rate, config)
        ),
        "status": "INVALID_DENSITY"
        if polynomial(minimum_at) < 0 or np.any(joint < 0) or zero_correction < 0
        else "NO_NEGATIVE_VALUE_FOUND",
        "policy": "No clipping, dropping negative weights, or absolute values; full Poisson mixture supplies the proper target.",
    }


def plot_result(
    config,
    sky,
    estimate_mean,
    oracle,
    profile_grid,
    profiles,
    cumulants,
    result,
    output,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), layout="constrained")
    limit = float(np.max(np.abs(sky["sky"])))
    for ax, values, title in zip(
        axes[0],
        [sky["sky"], sky["observed_sky"], estimate_mean],
        ["Injected Poisson sky", "Observed sky", "Reweighted posterior mean"],
        strict=True,
    ):
        handle = ax.imshow(
            values, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit
        )
        ax.set(title=title, xlabel="pixel x", ylabel="pixel y")
    fig.colorbar(
        handle, ax=axes[0].tolist(), label="Mean-subtracted sky amplitude", shrink=0.8
    )

    ax = axes[1, 0]
    yshift = max(oracle["log_evidence"])
    ax.plot(
        oracle["grid"],
        oracle["log_evidence"] - yshift,
        color="#192c4d",
        lw=2.3,
        label="Independent analytic integral",
    )
    for i, values in enumerate(profiles):
        ax.plot(
            profile_grid,
            values - yshift,
            color=["#ce5a32", "#268d8b"][i],
            linestyle=["--", ":"][i],
            lw=1.8,
            label=["GCR bank A", "Fresh GCR bank B"][i],
        )
    ax.axvline(
        config.rate,
        color="black",
        linestyle="--",
        lw=1,
        label=f"Injected rate = {config.rate:g}",
    )
    ax.axvspan(*oracle["interval95"], color="#192c4d", alpha=0.07)
    ax.set(
        title="Hyperparameter likelihood: full posterior weights",
        xlabel="Poisson rate per source cell",
        ylabel="Log marginal likelihood relative to oracle peak",
        ylim=(-7, 0.7),
    )
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1, 1]
    labels = [r"$K_{00}$", r"$K_{000}$", r"$K_{0000}$", r"$K_{001}$", r"$K_{0001}$"]
    ax.errorbar(
        np.arange(5),
        cumulants["estimate"] / cumulants["theory"],
        yerr=2 * cumulants["standard_error"] / np.abs(cumulants["theory"]),
        fmt="o",
        color="#268d8b",
        capsize=4,
    )
    ax.axhline(1, color="#192c4d", lw=1)
    ax.set(
        xticks=np.arange(5),
        xticklabels=labels,
        ylabel="Simulation / Campbell prediction",
        title="Spatial connected cumulants",
    )
    ax.text(
        0.03,
        0.04,
        f"{cumulants['realizations']:,} independent realizations\nError bars: 2 Monte Carlo standard errors",
        transform=ax.transAxes,
        fontsize=8,
    )

    ax = axes[1, 2]
    x = np.linspace(-4, 4, 600)
    exact = np.exp(coefficient_logpdf(x, config.rate, config))
    gaussian = np.exp(-x * x / (2 * config.variance)) / np.sqrt(
        2 * np.pi * config.variance
    )
    k3 = config.shot_variance**1.5 / np.sqrt(config.rate)
    k4 = config.shot_variance**2 / config.rate
    approximation = EdgeworthExpansion(
        0.0, np.sqrt(config.variance), cumulants=(k3, k4), order=4
    )
    signed = gaussian * np.asarray(approximation.correction(jnp.asarray(x)))
    ax.plot(x, exact, color="#192c4d", label="Full Poisson + Gaussian")
    ax.plot(
        x,
        gaussian,
        color="#999999",
        linestyle=":",
        label="Same mean and variance Gaussian",
    )
    ax.plot(
        x, signed, color="#ce5a32", linestyle="--", label="Order-4 Edgeworth (signed)"
    )
    ax.fill_between(x, signed, 0, where=signed < 0, color="#ce5a32", alpha=0.3)
    ax.axhline(0, color="#777777", lw=0.6)
    ax.annotate(
        "Negative Edgeworth tail",
        xy=(-3.18, -0.0013),
        xytext=(-3.9, 0.07),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": "#ce5a32"},
    )
    ax.set(
        title="Source-basis density and Edgeworth check",
        xlabel="Coefficient",
        ylabel="Density",
    )
    ax.legend(fontsize=8)
    fig.suptitle(
        "Campbell sky: cheap Gaussian draws, non-Gaussian parameter inference",
        fontsize=17,
        fontweight="bold",
    )
    fig.supxlabel(
        f"{config.shape[0]} x {config.shape[1]} periodic sky | mean and covariance held fixed across rates | GCR MAP {result['fits'][0]['rate_map']:.3f}, analytic MAP {oracle['mode']:.3f} | finite-map uncertainty is shown by the shaded 95% interval",
        fontsize=10,
    )
    fig.savefig(output / "campbell_sky.png", dpi=180)
    fig.savefig(output / "campbell_sky.pdf")
    plt.close(fig)


def run(output, draws=8192, steps=220, ensemble=24):
    output.mkdir(parents=True, exist_ok=True)
    config = SkyConfig()
    seed = 20260915
    started = time.monotonic()
    sky = simulate(config, seed)
    observed = deconvolve(sky["observed_sky"], config)
    oracle = oracle_profile(observed, config)
    print(
        f"Injected rate {config.rate}; analytic marginal MAP {oracle['mode']:.6f}; interval {oracle['interval95']}",
        flush=True,
    )
    reference = make_graph(observed, config)
    plan = compile(reference)
    if (
        plan.exact is None
        or plan.exact.method != "gcr"
        or plan.sampled is not None
        or plan.sigma_needs_rebuild
    ):
        raise RuntimeError("the reference must admit fixed-covariance GCR sampling")
    lo, hi = config.rate_bounds
    to_rate = lambda u: lo + (hi - lo) * jax.nn.sigmoid(u)
    from_rate = lambda r: np.log((r - lo) / (hi - r))
    reference_log_evidence = float(
        np.sum(
            -(observed**2) / (2 * (config.variance + config.noise_std**2))
            - 0.5 * np.log(2 * np.pi * (config.variance + config.noise_std**2))
        )
    )
    profile_grid = np.linspace(lo + 0.001, hi - 0.001, 61)
    profiles, fits, stored_samples = [], [], []
    for key_seed in (71, 72):
        sampled_at = time.monotonic()
        posterior = plan.sample(jax.random.key(key_seed), num_samples=draws)
        problem = PosteriorReweighting.from_graphs(
            reference,
            posterior.samples,
            lambda u: make_graph(observed, config, to_rate(u)),
        )
        fit = problem.fit(jnp.asarray(from_rate(3.0)), steps=steps, learning_rate=0.06)
        fitted_rate = float(to_rate(fit.parameters))
        log_score = eqx.filter_jit(problem.log_objective)
        profile = np.array(
            [
                float(log_score(jnp.asarray(from_rate(r)))) + reference_log_evidence
                for r in profile_grid
            ]
        )
        profiles.append(profile)
        draws_array = np.asarray(posterior.samples["coefficients"])
        coefficient_mean = np.einsum(
            "n,nij->ij", np.asarray(fit.importance.weights), draws_array
        )
        analytic_mean = oracle_posterior_mean(observed, fitted_rate, config)
        fit_record = {
            "key_seed": key_seed,
            "sampling_method": posterior.method,
            "draws": draws,
            "rate_map": fitted_rate,
            "log_evidence_ratio": float(fit.importance.log_mean_weight),
            "oracle_log_evidence_ratio_at_fitted_rate": oracle_log_evidence(
                observed, fitted_rate, config
            )
            - reference_log_evidence,
            "kish_ess": float(fit.importance.ess),
            "ess_fraction": float(fit.importance.ess_fraction),
            "largest_weight": float(fit.importance.max_weight),
            "gradient_norm_in_logit_rate": float(fit.gradient_norm),
            "coefficient_mean_rmse_against_analytic_target": float(
                np.sqrt(np.mean((coefficient_mean - analytic_mean) ** 2))
            ),
            "elapsed_seconds": time.monotonic() - sampled_at,
        }
        fits.append(fit_record)
        stored_samples.append((draws_array, coefficient_mean))
        print("Completed bank:", json.dumps(fit_record), flush=True)
    cumulants = cumulant_monte_carlo(config)
    print("Campbell cumulant z-scores:", cumulants["z_score"], flush=True)
    audit = edgeworth_audit(stored_samples[0][0], config)
    print("Edgeworth audit:", json.dumps(audit), flush=True)
    cutoff_error = max(
        abs(
            oracle_log_evidence(observed, r, config)
            - oracle_log_evidence(observed, r, config, count_max=2 * config.count_max)
        )
        for r in oracle["grid"]
    )
    ensemble_records = []
    for index in range(ensemble):
        alternate = simulate(config, 1000 + index)
        prof = oracle_profile(alternate["observed_coefficients"], config, size=251)
        ensemble_records.append(
            {
                "seed": 1000 + index,
                "oracle_map": prof["mode"],
                "interval95": prof["interval95"],
                "covers_injected_rate": bool(
                    prof["interval95"][0] <= config.rate <= prof["interval95"][1]
                ),
            }
        )
    result = {
        "config": dataclasses.asdict(config),
        "simulation_seed": seed,
        "hyperprior": "Uniform in rate on rate_bounds; optimize through logit without adding its Jacobian",
        "oracle": oracle,
        "fits": fits,
        "campbell_cumulants": cumulants,
        "edgeworth": audit,
        "maximum_poisson_tail_mass_per_cell": float(poisson.sf(config.count_max, hi)),
        "union_bound_omitted_count_probability_whole_map": float(
            np.prod(config.shape) * poisson.sf(config.count_max, hi)
        ),
        "maximum_log_evidence_change_doubling_count_cutoff": cutoff_error,
        "beam_roundtrip_maximum_error": float(
            np.max(np.abs(observed - sky["observed_coefficients"]))
        ),
        "oracle_only_ensemble": ensemble_records,
        "oracle_only_ensemble_coverage": float(
            np.mean([r["covers_injected_rate"] for r in ensemble_records])
        )
        if ensemble_records
        else None,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("jax", "numpy", "numpyro", "scipy")
        },
        "jax_x64_enabled": bool(jax.config.jax_enable_x64),
        "elapsed_seconds": time.monotonic() - started,
        "limitations": [
            "Toy grid and beam-correlated observation noise make an independent source basis and analytic oracle available.",
            "The two fits test Monte Carlo reproducibility for one sky; the oracle-only ensemble measures finite-sky variation, not calibration of the reweighting algorithm.",
            "Kish ESS does not certify missing-tail coverage; no automatic convergence verdict is issued.",
        ],
    }
    result = jsonable(result)
    (output / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(
        output / "maps_and_profiles.npz",
        **sky,
        posterior_mean=beam(stored_samples[0][1], config.beam_neighbor),
        reference_mean=beam(
            observed * config.variance / (config.variance + config.noise_std**2),
            config.beam_neighbor,
        ),
        profile_rates=profile_grid,
        reweighted_log_evidence=np.asarray(profiles),
    )
    plot_result(
        config,
        sky,
        beam(stored_samples[0][1], config.beam_neighbor),
        oracle,
        profile_grid,
        profiles,
        cumulants,
        result,
        output,
    )
    print("Saved", output, "elapsed", result["elapsed_seconds"], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/campbell-sky"))
    parser.add_argument("--draws", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--ensemble", type=int, default=24)
    args = parser.parse_args()
    if args.draws < 2 or args.steps < 1 or args.ensemble < 0:
        parser.error("draws >= 2, steps >= 1 and ensemble >= 0 are required")
    run(args.output, args.draws, args.steps, args.ensemble)


if __name__ == "__main__":
    main()
