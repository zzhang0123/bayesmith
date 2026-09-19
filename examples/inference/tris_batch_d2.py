"""Batch D2: foreground-basis extensions and prior/covariance sensitivity.

Two separate questions, kept apart:

* **Extensions.**  Does a more flexible foreground basis absorb the residual?
  Three forward models are fitted by maximum a posteriori on the SAME whitened
  data with the SAME coefficient priors: the production 3-region piecewise
  foreground, a 6-region refinement, and a 5-function smooth basis in Galactic
  latitude.  Only the basis changes, so the comparison isolates flexibility.
* **Sensitivity.**  The audited posterior draws are reweighted to alternate
  priors and an alternate external calibration.  Reweighting is exact
  self-normalised importance sampling when the proposal is the original
  posterior; the effective sample size is reported so an invalid reweighting is
  visible rather than believed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from examples.inference.tris_batch_b import load_draws
from examples.inference.tris_forward_baseline import zero_levels

SURVEYS = ("LWA", "ARCADE")
LOG_2PI = np.log(2.0 * np.pi)


def log_normal(x, sigma):
    x = np.asarray(x, dtype=float)
    return -0.5 * (x / sigma) ** 2 - np.log(sigma) - 0.5 * LOG_2PI


def importance(weights_log_ratio):
    """Self-normalised weights and their effective sample size."""
    ratio = np.asarray(weights_log_ratio, dtype=float)
    if ratio.ndim != 1:
        raise ValueError("importance weights must be one value per draw")
    ratio = ratio - ratio.max()
    weights = np.exp(ratio)
    weights /= weights.sum()
    return weights, float(1.0 / np.sum(weights**2))


def reweighted_summary(weights, values):
    values = np.asarray(values, dtype=float)
    flat = values.reshape(values.shape[0], -1)
    mean = (weights[:, None] * flat).sum(axis=0)
    variance = (weights[:, None] * (flat - mean) ** 2).sum(axis=0)
    unweighted = flat.mean(axis=0)
    return {
        "mean": mean.tolist(),
        "sd": np.sqrt(variance).tolist(),
        "unweighted_mean": unweighted.tolist(),
        "shift_in_sd": (
            (mean - unweighted) / np.sqrt(variance + 1e-300)
        ).tolist(),
    }


# ---------------------------------------------------------------------------
# Sensitivity reweighting
# ---------------------------------------------------------------------------


def prior_width_ratio(values, old_sd, new_sd):
    """Log prior ratio for widening or narrowing a Normal(0, sd) coordinate."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("a prior-width sensitivity needs one value per draw")
    return log_normal(values, new_sd) - log_normal(values, old_sd)


def zero820_symmetric_ratio(zero_standard, symmetric_scale):
    """Old two-piece half-normal versus a symmetric Normal on the 820 K level."""
    z = np.asarray(zero_standard, dtype=float)
    scale = np.where(z > 0.0, 0.430, 0.300)
    level = z * scale
    new = log_normal(level, symmetric_scale) + np.log(scale)
    return new - log_normal(z, 1.0)


def external_tau_ratio(draws, external, surveys, tau_scale):
    """Log likelihood ratio for scaling the shared calibration tau."""
    labels = np.asarray(external["survey"])
    mask = np.isin(labels, list(surveys))
    frequency = np.asarray(external["frequency_mhz"], dtype=float)[mask]
    data = np.asarray(external["temperature_rj_k"], dtype=float)[mask]
    sigma = np.asarray(external["sigma_independent_rj_k"], dtype=float)[mask]
    tau = np.asarray(external["tau_rj_k"], dtype=float)[mask]
    from examples.inference.tris_forward_baseline import cmb_rj_temperature

    def log_likelihood(scale):
        amplitude = np.asarray(draws["rsb_amplitude"], dtype=float)
        beta = np.asarray(draws["rsb_beta"], dtype=float)
        background = amplitude[:, None] * (frequency[None, :] / 1000.0) ** beta[:, None]
        calibration = np.asarray(draws["calibration_standard"], dtype=float)
        code = np.asarray(external["survey_code"])[mask]
        mean = (
            cmb_rj_temperature(frequency)[None, :]
            + background
            + calibration[:, code] * tau[None, :] * scale
        )
        return -0.5 * np.sum(((data[None, :] - mean) / sigma[None, :]) ** 2, axis=1)

    return log_likelihood(tau_scale) - log_likelihood(1.0)


# ---------------------------------------------------------------------------
# Extension models fitted by MAP
# ---------------------------------------------------------------------------


def equal_area_bins(latitude_deg, regions):
    """Region codes from equal-area bins in |sin b|."""
    absolute = np.abs(np.asarray(latitude_deg, dtype=float))
    # The fraction of the sphere in |b| < B is sin(B), so equal-area bands
    # have edges arcsin(k / regions).
    edges = np.rad2deg(np.arcsin(np.arange(1, regions) / regions))
    return np.digitize(absolute, edges)


def raised_cosine_basis(latitude_deg, functions=5, width=None):
    """Smooth non-negative basis functions over |b| in [0, 90] degrees."""
    absolute = np.abs(np.asarray(latitude_deg, dtype=float))
    if width is None:
        width = 90.0 / functions
    centers = (np.arange(functions) + 0.5) * width
    offsets = (absolute[:, None] - centers[None, :]) / width
    basis = 0.5 * (1.0 + np.cos(np.pi * np.clip(offsets, -1.0, 1.0)))
    return basis


def _level(zero_standard):
    return zero_levels(zero_standard)


def region_fields(bundle, coefficients, region):
    amplitude = np.exp(np.asarray(coefficients["log_amplitude"], dtype=float))
    beta = np.asarray(coefficients["beta"], dtype=float)
    return amplitude[region], beta[region]


def smooth_fields(bundle, coefficients, basis):
    amplitude = np.exp(np.asarray(basis, dtype=float) @ np.asarray(coefficients["log_amplitude"], dtype=float))
    beta = np.asarray(basis, dtype=float) @ np.asarray(coefficients["beta"], dtype=float)
    return amplitude, beta


def whitened_mean(bundle, amplitude_pixel, beta_pixel, zero_standard, monopole):
    template = np.asarray(bundle["template_k"], dtype=float)
    levels = _level(zero_standard)
    chunks = []
    for index, frequency in enumerate(np.asarray(bundle["frequency_mhz"], dtype=float)):
        response = np.asarray(bundle[f"response_{index}"], dtype=float)
        offset = np.asarray(bundle[f"offset_response_{index}"], dtype=float)
        scaling = amplitude_pixel * (frequency / 408.0) ** beta_pixel
        sky = (template + monopole) * scaling + float(bundle["cmb_k"][index])
        chunks.append(response @ sky - offset * levels[index])
    return np.concatenate(chunks)


def whitened_data(bundle):
    return np.concatenate(
        [
            np.asarray(bundle[f"whitened_data_{index}"], dtype=float)
            for index in range(len(np.asarray(bundle["frequency_mhz"])))
        ]
    )


def fit_map(bundle, decode, start, *, prior_sd, max_nfev=4000):
    """Least-squares MAP fit of the whitened residual plus prior residuals."""
    from scipy.optimize import least_squares

    data = whitened_data(bundle)
    location = np.zeros(np.size(start))

    def residual(theta):
        fields = decode(theta)
        mean = whitened_mean(bundle, fields[0], fields[1], fields[2], fields[3])
        return np.concatenate(
            [data - mean, (np.asarray(theta) - location) / prior_sd]
        )

    fit = least_squares(residual, start, max_nfev=max_nfev, ftol=1e-12, xtol=1e-12)
    fields = decode(fit.x)
    mean = whitened_mean(bundle, fields[0], fields[1], fields[2], fields[3])
    chi_square = float(np.sum((data - mean) ** 2))
    return {
        "success": bool(fit.success),
        "cost": float(fit.cost),
        "parameters": int(np.size(start)),
        "chi_square": chi_square,
        "chi_square_per_observation": chi_square / data.size,
        "reduced_cost_per_observation": float(fit.cost) / data.size,
        "theta": np.asarray(fit.x).tolist(),
    }


def region_models(bundle, regions, prior_sd=(0.5, 0.5)):
    latitude = np.asarray(bundle["galactic_latitude_deg"], dtype=float)
    region = equal_area_bins(latitude, regions)
    start = np.concatenate(
        [
            np.full(regions, np.log(1.6)),
            np.full(regions, -2.75),
            np.zeros(2),
            np.zeros(1),
        ]
    )
    widths = np.concatenate(
        [
            np.full(regions, prior_sd[0]),
            np.full(regions, prior_sd[1]),
            np.ones(2),
            np.full(1, 3.0),
        ]
    )

    def decode(theta):
        amplitude, beta = region_fields(
            bundle,
            {
                "log_amplitude": theta[:regions],
                "beta": theta[regions : 2 * regions],
            },
            region,
        )
        return amplitude, beta, theta[2 * regions : 2 * regions + 2], float(theta[-1])

    return {
        "label": f"{regions}-region",
        "start": start,
        "prior_sd": widths,
        "decode": decode,
        "regions": regions,
    }


def smooth_model(bundle, functions=5, prior_sd=(0.5, 0.5)):
    latitude = np.asarray(bundle["galactic_latitude_deg"], dtype=float)
    basis = raised_cosine_basis(latitude, functions=functions)
    start = np.concatenate(
        [
            np.full(functions, np.log(1.6)),
            np.full(functions, -2.75),
            np.zeros(2),
            np.zeros(1),
        ]
    )
    widths = np.concatenate(
        [
            np.full(functions, prior_sd[0]),
            np.full(functions, prior_sd[1]),
            np.ones(2),
            np.full(1, 3.0),
        ]
    )

    def decode(theta):
        amplitude, beta = smooth_fields(
            bundle,
            {
                "log_amplitude": theta[:functions],
                "beta": theta[functions : 2 * functions],
            },
            basis,
        )
        return amplitude, beta, theta[2 * functions : 2 * functions + 2], float(theta[-1])

    return {
        "label": f"smooth-{functions}",
        "start": start,
        "prior_sd": widths,
        "decode": decode,
        "functions": functions,
    }


def render(document: dict) -> str:
    lines = [
        "# Batch D2: foreground extensions and sensitivity",
        "",
        (
            f"Run {document['run_id']} at {document['created_utc']}; "
            f"elapsed {document['elapsed_s']} s."
        ),
        f"bayesmith HEAD {document['git']}",
        "",
        "## Extension models at their MAP, same data and same coefficient priors",
        "",
        "| model | parameters | chi2 | chi2 / N | cost / N |",
        "|---|---|---|---|---|",
    ]
    for row in document["extensions"]:
        lines.append(
            "| {label} | {parameters} | {chi_square:.1f} | "
            "{chi_square_per_observation:.4f} | "
            "{reduced_cost_per_observation:.4f} |".format(**row)
        )
    lines += [
        "",
        "The smooth model has the largest N and the smallest chi2/N is the",
        "question; a cost that grows shows the extra basis is prior-penalised.",
        "",
        "## Prior and calibration sensitivity by importance reweighting",
        "",
        "| variant | parameter | posterior mean | posterior SD | shift / SD | ESS | trustworthy |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in document["sensitivity"]:
        for name, summary in entry["summaries"].items():
            means = np.atleast_1d(summary["mean"])
            sds = np.atleast_1d(summary["sd"])
            shifts = np.atleast_1d(summary["shift_in_sd"])
            for index in range(means.size):
                lines.append(
                    "| {label} | {name}[{index}] | {mean:.4f} | {sd:.4f} | "
                    "{shift:+.2f} | {ess:.0f} | {trust} |".format(
                        label=entry["label"],
                        name=name,
                        index=index,
                        mean=means[index],
                        sd=sds[index],
                        shift=shifts[index],
                        ess=entry["ess"],
                        trust=entry["trustworthy"],
                    )
                )
    lines += [
        "",
        "## Limits",
        "",
        "- The extension comparison uses the TRIS whitened data only (no external",
        "  likelihood), so the foreground basis is the only thing changing.",
        "- A lower chi2/N with more parameters is not evidence by itself; the",
        "  uniform prior residuals are reported through cost / N.",
        "- Reweighting is exact only in the limit of the original proposal; the",
        "  reported ESS is the diagnostic.  Values of a few hundred or less are",
        "  not trustworthy and are flagged by the number itself.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audited-run", type=Path, required=True)
    parser.add_argument("--external-input", type=Path, default=Path("runs/tris-rsb-input"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.audited_run / "audited-input" / "maps.npz", allow_pickle=False) as archive:
        bundle = {name: archive[name] for name in archive.files}
    external = {
        name: value
        for name, value in np.load(
            args.external_input / "external.npz", allow_pickle=False
        ).items()
    }

    document = {
        "schema": "bayesmith.tris.batch-d2.v1",
        "run_id": args.output.name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": subprocess.check_output(
            ["git", "-C", ".", "rev-parse", "HEAD"], text=True
        ).strip(),
        "extensions": [],
        "sensitivity": [],
    }

    for model in (
        region_models(bundle, 3),
        region_models(bundle, 6),
        smooth_model(bundle, functions=5),
    ):
        row = fit_map(
            bundle, model["decode"], model["start"], prior_sd=model["prior_sd"]
        )
        row["label"] = model["label"]
        document["extensions"].append(row)

    for variant, include_rsb in (("no_rsb", False), ("rsb", True)):
        result_path = args.audited_run / "audited" / f"tris_haslam_{variant}" / "result.json"
        if not result_path.is_file():
            continue
        chain_shape = json.loads(result_path.read_text())["chain_shape"]
        draws = load_draws(
            args.audited_run / "audited" / f"tris_haslam_{variant}" / "posterior.npz",
            chain_shape,
        )
        flat = {name: np.asarray(value).reshape(-1, *np.asarray(value).shape[2:]) for name, value in draws.items()}
        count = flat["zero_standard"].shape[0]

        variants = [
            (
                "600 zero prior SD 1 -> 3",
                prior_width_ratio(flat["zero_standard"][:, 0], 1.0, 3.0),
                {"zero_standard": flat["zero_standard"]},
            ),
            (
                "820 zero symmetric 0.365 K",
                zero820_symmetric_ratio(flat["zero_standard"][:, 1], 0.365),
                {"zero_standard": flat["zero_standard"]},
            ),
            (
                "Haslam monopole prior SD 3 -> 10",
                prior_width_ratio(flat["haslam_monopole_K"], 3.0, 10.0),
                {"haslam_monopole_K": flat["haslam_monopole_K"]},
            ),
        ]
        if include_rsb:
            for scale in (1.5, 0.5):
                variants.append(
                    (
                        f"external calibration tau x {scale}",
                        external_tau_ratio(flat, external, SURVEYS, scale),
                        {"calibration_standard": flat["calibration_standard"]},
                    )
                )
        for label, ratio, values in variants:
            if ratio.shape[0] != count:
                raise ValueError("reweight ratio does not match the draws")
            weights, ess = importance(ratio)
            entry = {
                "label": label,
                "variant": variant,
                "ess": ess,
                "trustworthy": bool(ess > 100.0),
                "draws": int(count),
            }
            if ess < 10.0:
                # An effective sample size near one means the original posterior
                # does not cover the changed prior at all; its reweighted moments
                # are noise and are deliberately not reported.
                entry["summaries"] = {}
                entry["note"] = (
                    "ESS below 10: the original posterior does not cover this "
                    "prior change, so a fresh fit is required and no reweighted "
                    "moment is reported."
                )
            else:
                entry["summaries"] = {
                    name: reweighted_summary(weights, value)
                    for name, value in values.items()
                }
            document["sensitivity"].append(entry)

    document["elapsed_s"] = round(time.time() - started, 2)
    (args.output / "batch_d2.json").write_text(
        json.dumps(document, indent=2, sort_keys=True, default=float) + "\n"
    )
    (args.output / "batch_d2.md").write_text(render(document))
    print(json.dumps({"output": str(args.output), "elapsed_s": document["elapsed_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
