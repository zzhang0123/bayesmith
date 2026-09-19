"""Rebuild the D13 record from stage states, artifacts and actual JUnit results."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def build(rho_root, bayes_root):
    artifacts = {}

    def read(path):
        data = path.read_bytes()
        artifacts[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    record = {
        "schema": "tris.physical_modes_study.v1",
        "scientific_certification": False,
    }
    record["control"] = read(rho_root / "control-v2/gaussian.json")
    record["runs"] = {}
    for name in ("simulation16", "simulation16-optical", "real16-optical"):
        fit = read(rho_root / name / "summary.json")
        saved = sorted((rho_root / name).glob("chain[0-9]*.npz"))
        reported = fit.get(
            "completed_chains", fit["chains"] if fit["status"] == "sampled" else 0
        )
        if len(saved) != reported:
            raise ValueError(f"saved/reported chain count mismatch: {name}")
        for path in saved:
            artifacts[str(path.resolve())] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        fit["actual_saved_chains"] = len(saved)
        fit["process_exit"] = int((rho_root / f"{name}.exit").read_text())
        fit["pilot"] = read(rho_root / name / "pilot.json")
        record["runs"][name] = fit
    for name, path in (
        ("simulation_ppc", "simulation16-optical/ppc.json"),
        ("real_stopped_diagnostics", "real16-optical/stopped_chain_diagnostics.json"),
        ("real_optimized_point", "real16-optical/map_probe.json"),
        ("quadrature", "quadrature.json"),
        ("d12_ess_reanalysis", "ess_reanalysis.json"),
    ):
        record[name] = read(rho_root / path)
    record["validation"] = {}
    for name, root, filename in (
        ("rheplicant", rho_root, "regression"),
        ("bayesmith", bayes_root, "bayes"),
    ):
        xml = root / "tests" / f"{filename}.xml"
        tests = list(ET.parse(xml).iter("testcase"))
        counts = {
            tag: sum(t.find(tag) is not None for t in tests)
            for tag in ("failure", "error", "skipped")
        }
        counts["passed"] = len(tests) - sum(counts.values())
        counts["exit"] = int((root / "tests" / f"{filename}.exit").read_text())
        counts["junit"] = str(xml.resolve())
        record["validation"][name] = counts
        artifacts[str(xml.resolve())] = hashlib.sha256(xml.read_bytes()).hexdigest()
    record["artifact_sha256"] = artifacts
    return record


def plot(record, rho_root, output):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    for name, label, color in (
        ("simulation16", "EM coordinate: stopped", "#b44a3e"),
        ("simulation16-optical", "Optical coordinate: admitted", "#197c81"),
    ):
        with np.load(rho_root / name / "pilot_samples.npz") as data:
            axes[0].plot(data["te"], color=color, alpha=0.8, linewidth=0.8, label=label)
    axes[0].axhline(7200, color="black", ls=":", label="Injected Te")
    axes[0].set(
        xlabel="Pilot retained draw",
        ylabel="Te [K]",
        title="Diagnose the slow parameter",
    )
    axes[0].legend(fontsize=7)
    values = [
        record["runs"][name]["pilot"]["min_ess_per_draw"]
        for name in ("simulation16", "simulation16-optical")
    ]
    axes[1].bar(["EM", "Optical"], values, color=["#b44a3e", "#197c81"])
    axes[1].axhline(0.05, color="black", ls=":", label="Predeclared pilot gate")
    axes[1].set(
        yscale="log",
        ylabel="Minimum within-chain ESS / draw",
        title="A gate before long sampling",
    )
    axes[1].legend(fontsize=7)
    probe = record["real_optimized_point"]
    values = [
        probe["haslam"]["q_per_observation"],
        probe["ha_data"]["q_per_observation"],
    ]
    axes[2].bar(["Haslam", "H-alpha"], values, color="#bd7938")
    axes[2].axhline(1, color="black", ls=":")
    axes[2].set(
        yscale="log",
        ylabel="Optimized-point residual quadratic / N",
        title="Real-data mismatch; not posterior PPC",
    )
    fig.suptitle("D13: simulation passed; real model remains unaccepted", fontsize=12)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rho-root", "bayes-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    record = build(args.rho_root, args.bayes_root)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    plot(record, args.rho_root, args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
