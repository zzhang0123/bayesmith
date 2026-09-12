"""Read-only recovery diagnostics; generating truth is used only after inference."""

import hashlib
import json

import numpy as np
from scipy.stats import chi2, invgamma, truncnorm


def known_mean_reference(data, mean, truth, *, low=0.003, high=0.06):
    """Conditional scale posterior for y / mu - 1 ~ Normal(0, sigma**2).

    A Uniform density in sigma gives an inverse-gamma density in sigma squared,
    with shape (n-1)/2. This diagnostic conditions on the generating mean; it
    is not the joint posterior used in the demo.
    """
    z = np.asarray(data) / np.asarray(mean) - 1
    energy = float(np.sum(z**2))
    law = invgamma((z.size - 1) / 2, scale=energy / 2)
    cdf_low, cdf_high = law.cdf([low**2, high**2])
    quantiles = np.sqrt(law.ppf(cdf_low + np.array([0.005, 0.5, 0.995]) * (cdf_high - cdf_low)))
    return {
        "scope": "Diagnostic only: condition on the known generating mean.",
        "observations": int(z.size), "noise_energy": energy,
        "realized_noise_rms": float(np.sqrt(energy / z.size)),
        "energy_lower_tail_probability": float(chi2.cdf(energy / truth**2, z.size)),
        "uniform_sigma_posterior_quantiles": quantiles.tolist(),
    }


def compare_saved_runs(baseline, mild):
    """Compare fixed artifacts without altering chains, intervals or gate results."""
    paths = (baseline, mild)
    reports = [json.loads((path / "result.json").read_text()) for path in paths]
    first, second = reports
    if first.get("variant") is not None or (second.get("variant") or {}).get("kind") != "mild_noise_prior":
        raise ValueError("Noise-prior comparison requires a Uniform baseline and a mild-noise-prior counterpart")
    paired = {
        key: first[key] == second[key]
        for key in ("seed", "requested_draws_per_chain", "warmup", "chain_shape",
                    "interval_mass", "max_sd_ratio")
    }
    paired.update({
        key: first["signal"][key] == second["signal"][key]
        for key in ("x", "data", "truth")
    })
    paired["parameter_truths"] = (
        {p["name"]: p["truth"] for p in first["parameters"]}
        == {p["name"]: p["truth"] for p in second["parameters"]}
    )
    if not all(paired.values()):
        raise ValueError("Noise-prior comparison requires matched simulation and sampling settings")
    rows, chains = [], []
    for path, report in zip(paths, reports, strict=True):
        parameter = next(p for p in report["parameters"] if p["name"] == "sigma_w")
        with np.load(path / "posterior.npz") as saved:
            chain = saved["sigma_w"].reshape(report["chain_shape"])
        chains.append(chain)
        diagnostics = report["checks"]["chain_diagnostics"]
        rows.append({
            **parameter, "recovery_passed": report["passed"],
            "diagnostics_passed": diagnostics["passed"],
            "divergences": diagnostics["divergences"],
            "chain_upper_995": np.quantile(chain, 0.995, axis=1).tolist(),
            "draws_above_truth": np.sum(chain > parameter["truth"], axis=1).tolist(),
        })
    variant = second["variant"]
    loc, scale = variant["loc"], variant["scale"]
    low, high = variant["support"]
    flat_draws = chains[0].reshape(-1)
    weights = truncnorm.pdf(flat_draws, (low-loc)/scale, (high-loc)/scale, loc=loc, scale=scale)
    weights /= weights.sum()
    return {
        "matched": paired, "rows": rows, "prior": variant,
        "chain_shape": first["chain_shape"], "interval_mass": first["interval_mass"],
        "inputs_sha256": {
            label: {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                    for name in ("result.json", "posterior.npz")}
            for label, path in zip(("baseline", "mild-prior"), paths, strict=True)
        },
        "known_mean": known_mean_reference(first["signal"]["data"], first["signal"]["truth"], rows[0]["truth"]),
        "reweighting_check": {
            "note": "Sensitivity diagnostic only; normalized prior-ratio weights on baseline draws. Weight ESS does not account for chain autocorrelation.",
            "predicted_mild_mean": float(weights @ flat_draws),
            "weight_ess": float(1 / np.sum(weights**2)),
        },
    }
