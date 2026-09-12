"""Model adequacy and posterior sky products, separate from chain convergence."""

import numpy as np

from .tris_products import posterior_sky_products


def summarize(values):
    values = np.asarray(values)
    return {name: np.asarray(value).tolist() for name, value in {
        "mean": values.mean(axis=0), "sd": values.std(axis=0, ddof=1),
        "lower": np.quantile(values, .025, axis=0), "median": np.median(values, axis=0),
        "upper": np.quantile(values, .975, axis=0),
    }.items()}


def sky_components(bundle, draws):
    """Use paired joint draws; never combine marginal medians into a new sky."""
    count = len(draws["amplitude"])
    rsb_a = np.asarray(draws.get("rsb_amplitude", np.zeros(count)))
    rsb_beta = np.asarray(draws.get("rsb_beta", np.zeros(count)))
    background408 = rsb_a * .408 ** rsb_beta
    galactic = np.asarray(bundle["template_k"])[None, :] + np.asarray(draws["haslam_monopole_K"])[:, None] - background408[:, None]
    region = np.asarray(bundle["region"])
    amplitude = np.asarray(draws["amplitude"])[:, region]
    beta = np.asarray(draws["beta"])[:, region]
    zero = np.asarray(draws["zero_standard"])
    levels = zero * np.column_stack((np.full(count, .066), np.where(zero[:, 1] < 0, .300, .430)))
    return amplitude, beta, galactic, background408, levels


def posterior_checks(bundle, draws, *, seed):
    """Persist ring PPCs, support distances, prior departures, and map equivalence."""
    amp, beta, galactic, background408, levels = sky_components(bundle, draws)
    count = len(amp)
    a = np.asarray(draws.get("rsb_amplitude", np.zeros(count)))
    b = np.asarray(draws.get("rsb_beta", np.zeros(count)))
    products = posterior_sky_products(bundle, draws["amplitude"], draws["beta"])
    # The joint model subtracts B408 BEFORE amplitude recalibration and adds
    # it back afterwards; instrument zero corrections never enter sky maps.
    calibrated = amp * galactic + background408[:, None] + float(bundle["reference_cmb_k"])
    for name, values in (("haslam", calibrated), ("haslam_change", calibrated - products["haslam_reference_k"])):
        for stat, value in summarize(values).items():
            products[f"{name}_{stat}"] = np.asarray(value)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 731]))
    frequencies, map_chi, ring_chi = [], np.zeros(count), np.zeros(count)
    for i, frequency in enumerate(bundle["frequency_mhz"]):
        sky = amp * galactic * (frequency / 408.) ** beta + bundle["cmb_k"][i] + (a * (frequency / 1000.) ** b)[:, None]
        mean = sky @ bundle[f"operator_{i}"].T - levels[:, i, None]
        sigma, observed = bundle[f"sigma_k_{i}"], bundle[f"data_k_{i}"]
        replicated = mean + sigma * rng.normal(size=mean.shape)
        residual = observed - mean.mean(axis=0)
        d_obs = np.sum(((observed - mean) / sigma) ** 2, axis=1)
        d_rep = np.sum(((replicated - mean) / sigma) ** 2, axis=1)
        ppc = float(np.mean(d_rep >= d_obs))
        frequencies.append({
            "frequency_mhz": float(frequency), "ra_deg": bundle[f"ra_deg_{i}"].tolist(),
            "data_k": observed.tolist(), "sigma_k": sigma.tolist(),
            "mean_function": summarize(mean), "predictive": summarize(replicated),
            "residual_k": residual.tolist(), "standardized_residual": (residual / sigma).tolist(),
            "chi_square_at_posterior_mean": float(np.sum((residual / sigma) ** 2)),
            "chi_square_per_observation": float(np.mean((residual / sigma) ** 2)),
            "residual_rms_k": float(np.sqrt(np.mean(residual ** 2))),
            "ppc_tail_probability": ppc, "ppc_replicates": count,
            "ppc_resolution": 1 / count,
            "adequacy": "mismatch" if ppc < .01 or ppc > .99 else "not_flagged",
        })
        white_mean = sky @ bundle[f"response_{i}"].T - levels[:, i, None] * bundle[f"offset_response_{i}"]
        map_chi += np.sum((bundle[f"whitened_data_{i}"] - white_mean) ** 2, axis=1)
        ring_chi += d_obs
        for stat, value in summarize(mean @ bundle[f"weights_{i}"].T + bundle[f"bias_{i}"]).items():
            products[f"map_{stat}_{i}"] = np.asarray(value)
        products[f"sky_mean_{i}"] = sky.mean(axis=0)
        products[f"sky_sd_{i}"] = sky.std(axis=0, ddof=1)
    error = np.max(np.abs((map_chi - map_chi[0]) - (ring_chi - ring_chi[0])))
    scale = max(1., np.max(np.abs(ring_chi - ring_chi[0])))
    diagnostics = {
        "model_adequacy": "mismatch" if any(row["adequacy"] == "mismatch" for row in frequencies) else "not_flagged",
        "frequencies": frequencies,
        "map_likelihood": {"passed": bool(error / scale < 1e-7), "relative_delta_error": float(error / scale), "draws_checked": count},
        "support": {"minimum_galactic_temperature_K": summarize(galactic.min(axis=1)),
                    "fraction_below_0_01_K": float(np.mean(galactic.min(axis=1) < .01))},
        "prior_departures": {"zero_standard": summarize(draws["zero_standard"]),
                             "zero_level_K": summarize(levels),
                             "haslam_monopole_standard": summarize(np.asarray(draws["haslam_monopole_K"]) / 3)},
    }
    return diagnostics, products
