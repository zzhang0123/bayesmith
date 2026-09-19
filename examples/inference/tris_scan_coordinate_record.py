"""Record D17's actual stages and diagnostics without treating exit0 as acceptance."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from examples.inference.tris_recovery_record import artifact


def record(root, rho):
    directory = rho / "runs/tris-scan-coordinate-20260915"
    fit_dir = directory / "simulation"
    fit = artifact(fit_dir / "summary.json")
    status = fit["record"]["status"]
    if status in ("initializing", "pilot_running", "chain_running"):
        raise ValueError("D17 is still running; a finished snapshot is not available")
    actual = len(list(fit_dir.glob("chain*.npz")))
    if actual != fit["record"]["completed_chains"]:
        raise ValueError("recorded and actual chain counts disagree")
    fit.update(actual_saved_chains=actual, exit_code=int((directory / "simulation.exit").read_text()))
    evidence = {
        "fit": fit,
        "pilot": artifact(fit_dir / "pilot.json"),
        "metric": artifact(directory / "reference-metric/summary.json"),
    }
    for filename in ("ppc.json", "posterior_quadrature.json"):
        if (fit_dir / filename).exists():
            evidence[filename] = artifact(fit_dir / filename)
    programs = {}
    for name, relative, exit_name, program in (
        ("verification", "simulation/verification.json", "verification", "verify_samples.py"),
        ("slow_mode", "simulation/slow_mode_audit.json", "slow-mode", "audit_slow_mode.py"),
        ("conditional_em_shape", "conditional-em-shape-control/summary.json",
         "conditional-em-shape", "conditional_shape_control.py"),
        ("linear_information", "linear-information/summary.json",
         "linear-information", "linear_information.py"),
        ("linear_information_prior", "linear-information-prior/summary.json",
         "linear-information-prior", "linear_information_prior.py"),
        ("metric", "reference-metric/summary.json", "metric", "build_metric.py"),
    ):
        entry = artifact(directory / relative)
        entry["exit_code"] = int((directory / f"{exit_name}.exit").read_text())
        raw = (directory / program).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != entry["record"]["program_sha256"]:
            raise ValueError(f"diagnostic program changed after its run: {program}")
        if entry["exit_code"] != 0 or entry["record"].get("nonfinite_diagnostic_paths"):
            raise ValueError(f"diagnostic did not complete cleanly: {name}")
        for source, sha in entry["record"]["source_sha256"].items():
            if sha != fit["record"]["source_sha256"].get(source):
                raise ValueError(f"diagnostic and simulation source differ: {name}/{source}")
        evidence[name] = entry
        # Keep the run-specific diagnostic programs recoverable outside ignored runs/.
        programs[program] = {
            "path": str((directory / program).resolve()), "sha256": digest, "source": raw.decode(),
        }
    if evidence["conditional_em_shape"]["record"]["status"] != "completed":
        raise ValueError("conditional control is still incomplete")
    tests = {}
    for name in ("rho-scan-coordinate", "rho-regression"):
        path = directory / "tests" / f"{name}.xml"
        suites = list(ET.parse(path).getroot().iter("testsuite"))
        tests[name] = {
            key: sum(int(s.attrib.get(key, 0)) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        tests[name].update(
            exit_code=int(path.with_suffix(".exit").read_text()),
            junit=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    output = root / "examples/inference/data/tris_scan_coordinate_20260915.json"
    result = {
        "status": "record", "date": "2026-09-15", "decision": "D17", "task": "T-001",
        "scientific_certification": False,
        "interpretation": "Same physical target, exact explicit820 joint coordinate; actual diagnostics determine stage acceptance.",
        "evidence": evidence, "tests": tests, "diagnostic_programs": programs,
        "generator_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("tris_recovery_record.py"))
        },
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def plot_information(root, result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    information = result["evidence"]["linear_information_prior"]["record"]["results"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), layout="constrained")
    for ax, field, title in zip(axes, ("em_shape", "beta"), ("EM shape: 191 directions", "Beta: 9 directions")):
        for dataset, label, color in (
            ("halpha_only", "H-alpha only", "#a77b35"),
            ("all_data", "Haslam + H-alpha + TRIS", "#246d89"),
        ):
            ratios = information[dataset][field]["local_gaussian_variance_ratio"]
            ax.plot(range(1, len(ratios) + 1), ratios, label=label, color=color,
                    marker="o" if field == "beta" else None, markersize=4)
        ax.axhline(.5, linestyle=":", color="#666666", linewidth=1)
        ax.set(title=title, xlabel="Direction rank (each case sorted)",
               ylabel="Local Gaussian variance / prior variance", ylim=(-.025, 1.05))
        ax.grid(alpha=.15)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Information at the injected sky: local approximation, not posterior estimates", fontsize=12)
    fig.supxlabel("Other mean parameters use existing Gaussian prior curvature; fixed noise; 820 zero locally flat.", fontsize=9)
    fig.savefig(root / "examples/inference/data/tris_scan_coordinate_20260915_information.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    result = record(root, root.parent / "rheplicant")
    plot_information(root, result)
