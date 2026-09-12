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


def heldout_log_predictive_density(draws, heldout, *, include_rsb, chain_shape=None):
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
    shape = (1, len(scores)) if chain_shape is None else tuple(chain_shape)
    if len(shape) != 2 or np.prod(shape) != len(scores) or shape[1] < 8:
        raise ValueError("held-out chain shape must match the saved posterior draws")
    weights = np.exp(scores - np.max(scores)).reshape(shape)
    batches_per_chain = min(20, max(2, shape[1] // 50))
    batch_means = np.array([part.mean() for chain in weights
                            for part in np.array_split(chain, batches_per_chain)])
    # Delta method on the mean predictive density, not an average of log scores.
    # Taking the larger chain/batch estimate exposes between-chain variation.
    error = float(np.std(batch_means, ddof=1) / math.sqrt(len(batch_means)))
    if shape[0] > 1:
        error = max(error, float(np.std(weights.mean(axis=1), ddof=1) / math.sqrt(shape[0])))
    return {
        "log_predictive_density": _logmeanexp(scores),
        "mcse": error / float(weights.mean()),
        "mcse_method": "delta method, max of within-chain batch means and between-chain means",
        "batches_per_chain": batches_per_chain,
        "chain_shape": list(shape),
        "weight_concentration_ess": float(weights.sum() ** 2 / np.sum(weights ** 2)),
        "draws": len(scores),
        "score_draws": scores.tolist(),
    }


def score_refits(parent, common_hash):
    """Validate both directional refits before exposing any comparative score."""
    paths = {(variant, train): parent / "heldout" / f"{variant}_train_{train}"
             for variant in ("no_rsb", "rsb") for train in ("LWA", "ARCADE")}
    if any(not (path / filename).is_file() for path in paths.values()
           for filename in ("result.json", "posterior.npz")):
        return "missing_heldout_refits", [], {}
    folds = {key: json.loads((path / "result.json").read_text()) for key, path in paths.items()}
    for (variant, train), fold in folds.items():
        if (fold.get("variant") != variant or fold.get("included_surveys") != [train]
                or fold.get("data_manifest", {}).get("common_input_sha256") != common_hash):
            raise ValueError("held-out refit has wrong variant, training survey, or input identity")
    if not all(fold.get("passed") for fold in folds.values()):
        return "blocked_nonconverged_refits", [], {}
    with np.load(parent / "tris_haslam_rsb" / "external.npz", allow_pickle=False) as data:
        external = {name: data[name] for name in data.files}
    rows, hashes = [], {}
    for train, held in (("LWA", "ARCADE"), ("ARCADE", "LWA")):
        mask = external["survey"] == held
        heldout = {name: value[mask] for name, value in external.items()}
        scores = {}
        for variant in ("no_rsb", "rsb"):
            path = paths[variant, train]
            for filename in ("result.json", "posterior.npz"):
                hashes[(path / filename).relative_to(parent).as_posix()] = hashlib.sha256((path / filename).read_bytes()).hexdigest()
            with np.load(path / "posterior.npz", allow_pickle=False) as data:
                draws = {name: data[name] for name in data.files}
            scores[variant] = heldout_log_predictive_density(
                draws, heldout, include_rsb=variant == "rsb", chain_shape=folds[variant, train]["chain_shape"],
            )
            scores[variant].pop("score_draws")
        rows.append({"train": train, "heldout": held, "models": scores,
                     "delta_M1_minus_M0": scores["rsb"]["log_predictive_density"] - scores["no_rsb"]["log_predictive_density"],
                     "delta_mcse": math.hypot(scores["rsb"]["mcse"], scores["no_rsb"]["mcse"])})
    return "available", rows, hashes


def assemble_comparison(parent_output: Path, *, baseline_parent: Path | None = None) -> dict:
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
    passed = bool(reports["no_rsb"].get("passed") and reports["rsb"].get("passed"))
    status, scores, fold_hashes = score_refits(parent, left) if passed else ("blocked_nonconverged_chains", [], {})
    result = {
        "case": "tris_haslam_rsb_comparison",
        "kind": "real_observations",
        "title": "TRIS + Haslam + RSB: predictive comparison",
        "passed": passed,
        "comparison_status": status,
        "heldout_scores": scores,
        "data_manifest": {"common_input_sha256": left, "source_result_sha256": source_hashes, "source_refit_sha256": fold_hashes},
        "models": {key: {name: report.get(name) for name in ("case", "title", "kind", "method", "seed", "warmup", "passed", "parameters", "checks", "priors", "execution", "note", "chain_shape", "model_adequacy", "frequencies", "support", "prior_departures", "blocking", "preflight", "sky_products")} for key, report in reports.items()},
        "comparison_policy": "Conditional prediction of published foreground-subtracted summaries; shared literature inputs prevent independent instrument validation. No Bayes factor or two-survey PSIS-LOO.",
        "note": "Sampling convergence and predictive scores do not validate the rigid three-region foreground or establish the physical origin of an RSB.",
    }
    if baseline_parent is not None:
        baseline = {}
        for variant in ("no_rsb", "rsb"):
            path = Path(baseline_parent) / f"tris_haslam_{variant}" / "result.json"
            original = json.loads(path.read_text())
            if original["data_manifest"].get("common_input_sha256") != left:
                raise ValueError("review baseline differs from the refitted data")
            baseline[variant] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                 "checks": original["checks"], "chain_shape": original["chain_shape"],
                                 "execution": original["execution"]}
            with np.load(path.parent / "posterior.npz", allow_pickle=False) as draws, np.load(path.parent / "maps.npz", allow_pickle=False) as maps:
                background = draws["rsb_amplitude"] * .408 ** draws["rsb_beta"] if variant == "rsb" else 0.
                distance = (maps["template_k"].min() + draws["haslam_monopole_K"] - background).reshape(original["chain_shape"])
                baseline[variant]["median_min_template_K_by_chain"] = np.median(distance, axis=1).tolist()
        result["review_baseline"] = baseline
    output = parent / "tris_haslam_rsb_comparison"
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    assemble_comparison(args.input, baseline_parent=args.baseline)


if __name__ == "__main__":
    main()
