"""Host-side statistics for a registered fixed-root repeated-simulation study."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from scipy.stats import beta, t


def mean_uncertainty(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite independent replicate values are required")
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(values.size))
    half = float(t.ppf(0.975, values.size-1) * se)
    return {"mean": mean, "mcse": se, "interval_95": [mean-half, mean+half]}


def coverage_uncertainty(covered):
    covered = np.asarray(covered, dtype=bool)
    n, k = covered.size, int(covered.sum())
    if n == 0:
        raise ValueError("Coverage requires at least one replicate")
    return {
        "covered": k, "total": n, "rate": k/n,
        "interval_95": [float(beta.ppf(0.025, k, n-k+1)) if k else 0.0,
                        float(beta.ppf(0.975, k+1, n-k)) if k < n else 1.0],
        "interval_method": "Clopper-Pearson binomial; independent datasets",
    }


def summarize_coordinate(rows):
    errors = np.array([row["mean"] - row["truth"] for row in rows])
    error = mean_uncertainty(errors)
    squared = mean_uncertainty(errors**2)
    rmse = float(np.sqrt(squared["mean"]))
    return {
        "replicates": len(rows), "bias": error["mean"], "bias_mcse": error["mcse"],
        "bias_interval_95": error["interval_95"], "rmse": rmse,
        "rmse_mcse_delta": squared["mcse"]/(2*rmse) if rmse else 0.0,
        "mean_posterior_sd": float(np.mean([row["posterior_sd"] for row in rows])),
        "coverage": {mass: coverage_uncertainty([row["intervals"][mass][0] <= row["truth"] <= row["intervals"][mass][1] for row in rows])
                     for mass in rows[0]["intervals"]},
    }


def read_run(study, attempt, config):
    path = study / attempt["result"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != attempt["result_sha256"]:
        raise ValueError("Saved result differs from the completed attempt")
    report = json.loads(path.read_text())
    if not attempt.get("source_matches_registration"):
        raise ValueError("Run did not use the registered inference sources")
    if report["seed"] != attempt["seed"] or report["chain_shape"] != [config["chains"], config["draws"]] or report["warmup"] != config["warmup"]:
        raise ValueError("Run seed or sampling budget differs from registration")
    if (report.get("variant") is not None) != (attempt["prior"] == "mild"):
        raise ValueError("Prior variant differs from registration")
    if attempt["prior"] == "mild":
        variant, declared = report["variant"], config["mild_prior"]
        if variant.get("kind") != "mild_noise_prior" or any(variant.get(k) != declared[k] for k in ("loc", "scale")) or variant.get("support") != [declared["low"], declared["high"]]:
            raise ValueError("Mild prior differs from registration")
    draws_path = path.with_name("posterior.npz")
    columns = {}
    try:
        with np.load(draws_path) as saved:
            for name in saved.files:
                values = saved[name]
                for index in np.ndindex(values.shape[1:]):
                    coordinate = name + (str(list(index)) if index else "")
                    columns[coordinate] = values[(slice(None), *index)]
    except (OSError, EOFError, BadZipFile, ValueError) as exc:
        raise OSError(f"Unreadable posterior archive: {draws_path.name}: {exc}") from exc
    rows = {}
    for row in report["parameters"]:
        values = columns[row["name"]]
        if values.size != config["draws"] * config["chains"] or not np.isfinite(values).all():
            raise ValueError("Invalid saved posterior draws")
        np.testing.assert_allclose(values.mean(), row["mean"], rtol=1e-12, atol=1e-14)
        rows[row["name"]] = {
            "truth": row["truth"], "mean": row["mean"], "posterior_sd": row["posterior_sd"],
            "intervals": {str(mass): np.quantile(values, [(1-mass)/2, (1+mass)/2]).tolist()
                          for mass in config["interval_masses"]},
        }
    for name, value in config["fixed_roots"].items():
        value = np.asarray(value)
        for index in np.ndindex(value.shape):
            coordinate = name + (str(list(index)) if index else "")
            if rows[coordinate]["truth"] != float(value[index]):
                raise ValueError("Generating root parameter differs from registration")
    signal = report["signal"]
    simulation_hash = hashlib.sha256(json.dumps({
        "x": signal["x"], "data": signal["data"], "mean_truth": signal["truth"],
        "truths": {k:v["truth"] for k,v in rows.items()},
    }, sort_keys=True).encode()).hexdigest()
    return {
        **attempt, "rows": rows, "simulation_sha256": simulation_hash,
        "posterior_sha256": hashlib.sha256(draws_path.read_bytes()).hexdigest(),
        "noise_realization": signal["view"].get("noise_realization"),
        "observations": len(signal["data"]),
        "divergences": report["checks"]["chain_diagnostics"]["divergences"],
        "sigma_diagnostics": report["checks"]["chain_diagnostics"]["sites"]["sigma_w"],
        "method": report["method"],
    }


def summarize_study(study):
    config = json.loads((study / "registered.json").read_text())
    completion = json.loads((study / "completion.json").read_text())
    if not completion["source_unchanged"]:
        raise ValueError("Frozen study engine changed")
    attempts = json.loads((study / "progress.json").read_text())
    expected = {(seed, prior) for seed in config["seeds"] for prior in config["priors"]}
    if len(attempts) != len(expected) or {(r["seed"], r["prior"]) for r in attempts} != expected:
        raise ValueError("Study does not account for every registered attempt")
    completed = []
    failed = [a for a in attempts if not a["completed"]]
    for attempt in attempts:
        if attempt["completed"]:
            try:
                completed.append(read_run(study, attempt, config))
            except OSError as exc:
                # Older runners could write a report before failing to export
                # draws. Retain that attempt without dropping other datasets.
                failed.append({**attempt, "completed": False, "artifact_read_error": str(exc)})
    groups = {}
    for prior in config["priors"]:
        runs = [r for r in completed if r["prior"] == prior]
        clean = [r for r in runs if r["diagnostics_passed"]]
        groups[prior] = {
            "attempted": len(config["seeds"]), "completed": len(runs),
            "diagnostics_passed": len(clean),
            "all_coordinates_recovered": sum(r["recovery_passed"] for r in runs),
            "wall_seconds_sum": sum(r["elapsed_seconds"] for r in runs),
            "parameters_all_completed": {name: summarize_coordinate([r["rows"][name] for r in runs])
                                         for name in runs[0]["rows"]} if len(runs) > 1 else {},
            "parameters_diagnostics_passed": {name: summarize_coordinate([r["rows"][name] for r in clean])
                                              for name in clean[0]["rows"]} if len(clean) > 1 else {},
        }
    by_key = {(r["seed"], r["prior"]): r for r in completed}
    pairs = []
    for seed in config["seeds"]:
        if all((seed, prior) in by_key for prior in config["priors"]):
            baseline, mild = (by_key[seed, prior] for prior in ("uniform", "mild"))
            if baseline["simulation_sha256"] != mild["simulation_sha256"]:
                raise ValueError("Paired priors did not use identical simulated data and truths")
            pairs.append((baseline, mild))
    paired = {}
    if len(pairs) > 1:
        for name in pairs[0][0]["rows"]:
            errors = np.array([[r["rows"][name]["mean"] - r["rows"][name]["truth"] for r in pair] for pair in pairs])
            paired[name] = {
                "pairs": len(pairs), "mean_change": mean_uncertainty(errors[:, 1]-errors[:, 0]),
                "absolute_error_change": mean_uncertainty(np.abs(errors[:, 1])-np.abs(errors[:, 0])),
                "squared_error_change": mean_uncertainty(errors[:, 1]**2-errors[:, 0]**2),
            }
    return {
        "kind": config["kind"], "registered": config, "completion": completion,
        "summary_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "groups": groups, "paired_mild_minus_uniform": paired,
        "runs": completed, "failed_attempts": failed,
        "uncertainty": "Bias and paired mean changes use Student-t intervals over independent datasets. Bias MCSE includes finite-chain noise. RMSE SE uses a delta approximation. Coverage intervals are exact binomial intervals. All intervals are pointwise; the primary parameter is sigma_w.",
        "scope": "Fixed root parameters; process instance and observational noise redrawn. Not prior-predictive SBC, not a parameter-grid study, and not a universal unbiasedness claim. Diagnostic failures remain in primary completed-run summaries; passing subsets are separate.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_study(args.study)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for prior, group in summary["groups"].items():
        print(prior, group["completed"], group["diagnostics_passed"],
              group["parameters_all_completed"].get("sigma_w"))


if __name__ == "__main__":
    main()
