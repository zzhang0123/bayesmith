"""Compare saved precision at two observation counts, without reweighting draws."""

import hashlib
import json


def compare_observation_counts(baseline, larger):
    reports = [json.loads((path / "result.json").read_text()) for path in (baseline, larger)]
    first, second = reports
    if first.get("variant") is not None or (second.get("variant") or {}).get("prior") != "bounded_uniform_roots":
        raise ValueError("Observation-count comparison requires the original Uniform priors")
    for key in ("seed", "chain_shape", "warmup", "interval_mass", "max_sd_ratio"):
        if first[key] != second[key]:
            raise ValueError(f"Observation-count comparison changed {key}")
    truths = [{p["name"]: p["truth"] for p in report["parameters"]} for report in reports]
    if truths[0] != truths[1]:
        raise ValueError("Observation-count comparison changed parameter or process-instance truth")
    rows = []
    for report in reports:
        rows.append({
            "observations": len(report["signal"]["data"]),
            "parameters": {p["name"]: p for p in report["parameters"]},
            "recovery_passed": report["checks"]["recovery"],
            "diagnostics_passed": report["checks"]["chain_diagnostics"]["passed"],
            "divergences": report["checks"]["chain_diagnostics"]["divergences"],
            "noise_realization": report["signal"]["view"]["noise_realization"],
            "method": report["method"],
        })
    if rows[0]["observations"] >= rows[1]["observations"]:
        raise ValueError("Second run must contain more observations")
    if second["variant"].get("observations") != rows[1]["observations"]:
        raise ValueError("Observation-count metadata disagrees with saved data")
    return {
        "rows": rows, "seed": first["seed"], "chain_shape": first["chain_shape"],
        "inputs_sha256": {str(path): hashlib.sha256((path / "result.json").read_bytes()).hexdigest()
                          for path in (baseline, larger)},
        "same_process_and_parameter_truths": True,
        "noise_is_nested": False,
        "independent_replicates": False,
        "scope": "One realization at each size, with identical priors and generating process instance. The same random stream is assigned to a denser input grid; this is not a controlled repeated-simulation estimate of sample-size effects.",
    }
