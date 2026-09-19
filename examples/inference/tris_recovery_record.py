"""Assemble D16 evidence from completed runs; preserve rejected experiments."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def artifact(path):
    raw = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "record": json.loads(raw),
    }


def record(root, rho):
    directory = rho / "runs/tris-local-fields-20260915"
    runs = {}
    for name in (
        "simulation-fixed-step-independent",
        "simulation-reference-metric",
        "simulation-collapsed-haslam",
        "simulation-collapsed-fixed-half",
        "simulation-optical-mean",
    ):
        run_dir = directory / name
        entry = artifact(run_dir / "summary.json")
        if entry["record"]["status"] in ("initializing", "pilot_running", "chain_running"):
            raise ValueError(f"run is still active: {name}")
        entry["exit_code"] = int((directory / f"{name}.exit").read_text())
        entry["actual_saved_chains"] = len(list(run_dir.glob("chain*.npz")))
        if entry["actual_saved_chains"] != entry["record"]["completed_chains"]:
            raise ValueError(f"saved chain count differs: {name}")
        entry["pilot"] = artifact(run_dir / "pilot.json")
        for filename in ("failed_chain_audit.json", "leapfrog_audit.json", "slow_mode_audit.json", "ppc.json",
                         "posterior_quadrature.json"):
            if (run_dir / filename).exists():
                entry[filename] = artifact(run_dir / filename)
        runs[name] = entry
    evidence = {
        name: artifact(directory / path)
        for name, path in (
            ("reference_metric", "reference-metric/summary.json"),
            ("collapsed_metric", "reference-metric-collapsed/summary.json"),
            ("optical_mean_metric", "reference-metric-optical-mean-absolute/summary.json"),
            ("gaussian_control", "control-ess-v3/gaussian.json"),
            ("ess_reference_audit", "ess-reference-audit.json"),
            ("divergence_replay", "optical-mean-divergence-replay/summary.json"),
            ("rejected_step", "optical-mean-divergence-replay/rejected_step_audit.json"),
            ("offset_boundary", "optical-mean-divergence-replay/offset_boundary_audit.json"),
            ("fixed_states_q32_q64", "quadrature-failed-states-q32-q64.json"),
            ("fixed_states_q64_q128", "quadrature-failed-states-q64-q128.json"),
        )
    }
    tests = {}
    for name in ("rho-sample-quadrature", "rho-reference-metric", "rho-ess-antithetic",
                 "rho-collapsed-haslam", "rho-d16-regression", "rho-optical-mean",
                 "rho-d16-final"):
        path = directory / "tests" / f"{name}.xml"
        suites = list(ET.parse(path).getroot().iter("testsuite"))
        tests[name] = {
            key: sum(int(s.attrib.get(key, 0)) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        tests[name].update(
            exit_code=int(path.with_suffix(".exit").read_text()),
            junit=str(path.resolve()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    result = {
        "status": "record",
        "date": "2026-09-15",
        "decision": "D16",
        "task": "T-001",
        "scientific_certification": False,
        "interpretation": (
            "Actual stages and tests; rejected chains remain diagnostic only. "
            "A sampling pass is not physical-model or numerical certification."
        ),
        "runs": runs,
        "evidence": evidence,
        "tests": tests,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output = root / "examples/inference/data/tris_recovery_20260915.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot(result, output.with_suffix(".png"))
    plot_slow_mode(directory / "simulation-collapsed-fixed-half/slow_mode_series.npz",
                   output.with_name("tris_recovery_20260915_slow_mode.png"))
    return result


def plot(result, path):
    import matplotlib.pyplot as plt

    labels = ["Original metric\nhalf step", "Reference metric\nexplicit zero",
              "Reference metric\nmarginal zero", "Marginal zero\nhalf step", "Optical mean\ncoordinate"]
    pilots = [value["pilot"]["record"] for value in result["runs"].values()]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    colors = ["#b75743" if not pilot["proceed"] else "#227a72" for pilot in pilots]
    axes[0].bar(range(5), [p["divergences"] for p in pilots], color=colors)
    axes[0].set(title="Pilot divergences", ylabel="Count", xticks=range(5), xticklabels=labels)
    # The explicit reference pilot has a historical invalid estimate. Do not
    # silently replace its recorded outcome with a later diagnostic implementation.
    ess = [p["min_ess_per_draw"] for p in pilots]
    for i, value in enumerate(ess):
        if value > 0:
            axes[1].bar(i, value, color=colors[i])
        else:
            axes[1].text(i, 0.015, "Invalid ESS\n(run stopped)", ha="center", fontsize=8)
    axes[1].axhline(0.05, ls="--", color="#333333", lw=1)
    axes[1].set(title="Pilot minimum ESS / draw", xticks=range(5), xticklabels=labels)
    values = [result["evidence"][k]["record"]["density"]["delta_log_density_sd"]
              for k in ("fixed_states_q32_q64", "fixed_states_q64_q128")]
    axes[2].bar(["32 to 64", "64 to 128"], values, color="#536b98")
    axes[2].set(title="Integration sensitivity: five failed states",
                ylabel="SD of full log-density change [nat]", xlabel="Quadrature NSIDE")
    for axis in axes:
        axis.tick_params(axis="x", labelsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("D16 computational checks — no accepted real-data posterior", fontsize=13)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_slow_mode(source, path):
    import matplotlib.pyplot as plt
    import numpy as np

    with np.load(source, allow_pickle=False) as data:
        eta, normalization, xi = data["eta"], data["normalization"], data["xi"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(eta - eta.mean(), lw=0.8, label="Original optical log-EM monopole", color="#b75743")
    axes[0].plot(xi - xi.mean(), lw=0.8, label="Observed optical log-mean", color="#227a72")
    axes[0].set(xlabel="Saved draw", ylabel="Centered log coordinate", title="Slow drift versus measured combination")
    axes[0].legend(fontsize=8)
    axes[1].scatter(normalization, eta, s=4, alpha=0.35, color="#536b98")
    axes[1].set(xlabel="Shape log-mean-exp normalization", ylabel="Original optical log-EM monopole",
                title=f"Correlation = {np.corrcoef(eta, normalization)[0, 1]:.6f}")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Rejected chain: normalization coupling, not a posterior constraint", fontsize=12)
    fig.savefig(path, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    record(root, root.parent / "rheplicant")
