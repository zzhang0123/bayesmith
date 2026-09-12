"""Cross-survey posterior prediction for the TRIS + Haslam RSB analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .tris_rsb_data import HC_OVER_K_MHZ_K, survey_covariance

CMB_THERMODYNAMIC_K = 2.725


def _cmb_rj_temperature(frequency_mhz):
    frequency = np.asarray(frequency_mhz, dtype=float)
    x = HC_OVER_K_MHZ_K * frequency / CMB_THERMODYNAMIC_K
    return CMB_THERMODYNAMIC_K * x / np.expm1(x)


def _logmeanexp(values):
    values = np.asarray(values, dtype=float)
    maximum = np.max(values)
    return float(maximum + np.log(np.mean(np.exp(values - maximum))))


def _multivariate_normal_logpdf(value, mean, covariance):
    residual = np.asarray(value) - np.asarray(mean)
    factor = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(factor, residual)
    return float(-0.5 * (len(residual) * math.log(2 * math.pi) + 2 * np.log(np.diag(factor)).sum() + whitened @ whitened))


def heldout_log_predictive_density(draws, heldout, *, include_rsb):
    """Integrate the unseen survey's shared calibration analytically."""
    frequency = np.asarray(heldout["frequency_mhz"], dtype=float)
    data = np.asarray(heldout["temperature_rj_k"], dtype=float)
    covariance = survey_covariance(
        heldout["sigma_independent_rj_k"], heldout["survey_code"], heldout["tau_rj_k"]
    )
    if np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError("held-out survey covariance must be positive definite")
    if include_rsb:
        amplitude = np.asarray(draws["rsb_amplitude"], dtype=float)
        beta = np.asarray(draws["rsb_beta"], dtype=float)
        if amplitude.ndim != 1 or beta.shape != amplitude.shape:
            raise ValueError("RSB held-out draws must be aligned one-dimensional arrays")
        background = amplitude[:, None] * (frequency[None, :] / 1000.0) ** beta[:, None]
    else:
        draw_count = len(next(iter(draws.values())))
        background = np.zeros((draw_count, len(frequency)))
    if len(background) < 8:
        raise ValueError("held-out score requires at least eight posterior draws")
    scores = np.asarray([
        _multivariate_normal_logpdf(data, _cmb_rj_temperature(frequency) + component, covariance)
        for component in background
    ])
    batches = np.array_split(scores, 4)
    batch_scores = np.asarray([_logmeanexp(batch) for batch in batches])
    return {
        "log_predictive_density": _logmeanexp(scores),
        "mcse": float(np.std(batch_scores, ddof=1) / math.sqrt(len(batch_scores))),
        "draws": len(scores),
        "score_draws": scores.tolist(),
    }


def assemble_comparison(parent_output: Path) -> dict:
    """Bind M0/M1 only when their full-data inputs are byte-identical."""
    parent = Path(parent_output)
    reports, source_hashes = {}, {}
    for variant in ("no_rsb", "rsb"):
        path = parent / f"tris_haslam_{variant}" / "result.json"
        if not path.is_file():
            raise ValueError(f"missing full-data {variant} result")
        raw = path.read_bytes()
        reports[variant] = json.loads(raw)
        source_hashes[path.relative_to(parent).as_posix()] = hashlib.sha256(raw).hexdigest()
    left = reports["no_rsb"]["data_manifest"].get("common_input_sha256")
    right = reports["rsb"]["data_manifest"].get("common_input_sha256")
    if not left or left != right:
        raise ValueError("full models do not share one common input manifest")
    result = {
        "case": "tris_haslam_rsb_comparison",
        "kind": "real_observations",
        "title": "TRIS + Haslam + RSB: predictive comparison",
        "passed": bool(reports["no_rsb"].get("passed") and reports["rsb"].get("passed")),
        "comparison_status": "available" if reports["no_rsb"].get("passed") and reports["rsb"].get("passed") else "blocked_nonconverged_chains",
        "data_manifest": {"common_input_sha256": left, "source_result_sha256": source_hashes},
        "models": {key: {name: report.get(name) for name in ("case", "title", "passed", "parameters", "checks", "priors", "execution", "note")} for key, report in reports.items()},
        "comparison_policy": "Directional held-out posterior predictive scoring is reserved for converged refits; no Bayes factor and no two-survey PSIS-LOO.",
        "note": "The current full-data chains did not meet convergence criteria, so no held-out score or physical interpretation is reported.",
    }
    output = parent / "tris_haslam_rsb_comparison"
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    assemble_comparison(args.input)


if __name__ == "__main__":
    main()
