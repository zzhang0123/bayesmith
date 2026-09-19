"""Checkpoint D20 inputs, actual chain prefixes, execution state and validation."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from examples.inference.tris_recovery_record import artifact


def record(root, rho, *, allow_active=False):
    directory = rho / "runs/tris-calibration-reference-20260915"
    run = directory / "simulation"
    entry = artifact(run / "summary.json")
    fit = entry["record"]
    exit_path = directory / "simulation.exit"
    entry["exit_code"] = int(exit_path.read_text()) if exit_path.exists() else None
    active = entry["exit_code"] is None
    entry["current_stage"] = (
        f"chain{fit['running_chain'] - 1}" if fit["status"] == "chain_running"
        else fit["status"]
    )
    entry["last_saved_phase"] = fit.get("last_screen", {}).get("phase")
    if active and not allow_active:
        raise ValueError("simulation has not exited; use an explicitly incomplete snapshot")
    entry["actual_saved_complete_chains"] = len(list(run.glob("chain[0-9].npz")))
    if entry["actual_saved_complete_chains"] != fit["completed_chains"]:
        raise ValueError("saved complete chains differ from the summary checkpoint")
    partials = {}
    for path in sorted(run.glob("partial_chain[0-9].npz")):
        with np.load(path, allow_pickle=False) as data:
            count = len(data["te"])
            if any(value.shape[0] != count for value in data.values()):
                raise ValueError("unaligned saved partial chain")
        partials[path.name] = {"draws": count, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    entry["saved_partial_chains"] = partials
    entry["prefixes"] = [artifact(p) for p in sorted(run.glob("chain*_prefix*.json"))]
    for name in ("pilot.json", "geometry.json"):
        if (run / name).exists():
            entry[name] = artifact(run / name)
    for name, digest in fit["source_sha256"].items():
        if hashlib.sha256((run / "source" / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"fit source snapshot differs: {name}")
    evidence = {"simulation": entry}
    programs = {}
    for name, relative, exit_name, program in (
        ("gaussian", "gaussian-control/gaussian.json", "gaussian-control", "gaussian_chunk_control.py"),
        ("previous_prefixes", "previous-prefixes.json", "previous-prefixes", "audit_previous_prefixes.py"),
    ):
        item = artifact(directory / relative)
        item["exit_code"] = int((directory / f"{exit_name}.exit").read_text())
        if item["exit_code"] != 0 or item["record"].get("nonfinite_diagnostic_paths"):
            raise ValueError(f"validation incomplete or invalid: {name}")
        evidence[name] = item
        path = directory / program
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["record"]["program_sha256"]:
            raise ValueError(f"validation program changed: {program}")
        programs[program] = {"source": raw.decode(), "sha256": hashlib.sha256(raw).hexdigest()}
    path = directory / "run_explicit.py"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != fit["program_sha256"]:
        raise ValueError("experiment program changed while running")
    programs[path.name] = {"source": raw.decode(), "sha256": hashlib.sha256(raw).hexdigest()}
    for name, record_name, program_name in (
        ("fine_inputs", "fine-inputs", "prepare_fine.py"),
        ("pilot_quadrature", "pilot-quadrature", "check_pilot_quadrature.py"),
        ("pilot_quadrature256", "pilot-quadrature256", "check_pilot_quadrature256.py"),
        ("prefix_audit", "prefix-audit", "audit_prefix.py"),
        ("prefix_compensation", "prefix-compensation", "probe_prefix_compensation.py"),
        ("joint_move_probe", "joint-move-probe", "probe_joint_moves.py"),
    ):
        path = directory / f"{record_name}.json"
        if not path.exists():
            continue
        item = artifact(path)
        item["exit_code"] = int((directory / f"{record_name}.exit").read_text())
        if item["exit_code"] or item["record"].get("nonfinite_diagnostic_paths"):
            raise ValueError(f"diagnostic incomplete or invalid: {name}")
        raw = (directory / program_name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["record"]["program_sha256"]:
            raise ValueError(f"diagnostic program changed: {program_name}")
        evidence[name] = item
        programs[program_name] = {
            "source": raw.decode(), "sha256": hashlib.sha256(raw).hexdigest(),
        }
    figure_path = directory / "figure.json"
    if figure_path.exists():
        item = artifact(figure_path)
        item["exit_code"] = int((directory / "figure-astronomy.exit").read_text())
        raw = (directory / "plot_diagnostics.py").read_bytes()
        if item["exit_code"] or hashlib.sha256(raw).hexdigest() != item["record"]["program_sha256"]:
            raise ValueError("Figure execution or program differs")
        figure = Path(item["record"]["output"])
        if hashlib.sha256(figure.read_bytes()).hexdigest() != item["record"]["output_sha256"]:
            raise ValueError("Diagnostic figure changed")
        item["environment"] = "base astronomy Python; rheplicant venv lacks matplotlib (exit1)"
        evidence["figure"] = item
        programs["plot_diagnostics.py"] = {
            "source": raw.decode(), "sha256": hashlib.sha256(raw).hexdigest(),
        }
    current = directory / "gaussian-control/gaussian_samples.npz"
    original = rho / "runs/tris-joint-compensation-20260915/gaussian-control/gaussian_samples.npz"
    with np.load(current) as new, np.load(original) as old:
        if set(new.files) != set(old.files):
            raise ValueError("Gaussian control parameter sets differ")
        equality = {name: bool(np.array_equal(new[name], old[name])) for name in new.files}
    evidence["gaussian_stream_comparison"] = {
        "bitwise_equal": equality,
        "input_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in (current, original)},
    }
    tests = {}
    for name, path in (
        ("chunks", rho / "runs/tris-joint-compensation-20260915/tests/chunks.xml"),
        ("regression", directory / "tests/regression-final.xml"),
    ):
        suites = list(ET.parse(path).getroot().iter("testsuite"))
        tests[name] = {
            key: sum(int(s.attrib.get(key, 0)) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        tests[name].update(exit_code=int(path.with_suffix(".exit").read_text()), junit=str(path.resolve()))
        if tests[name]["exit_code"] or tests[name]["errors"] or tests[name]["failures"]:
            raise ValueError(f"test validation failed or incomplete: {name}")
    result = {
        "status": "record", "decision": "D20", "date": "2026-09-15", "task": "T-001",
        "snapshot_at_utc": datetime.now(UTC).isoformat(),
        "all_experiments_completed": not active,
        "active_experiments": ["simulation"] if active else [],
        "scientific_certification": False, "evidence": evidence,
        "diagnostic_programs": programs, "tests": tests,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation": "Explicit original calibration prior; exact marginal physical target. Saved prefixes are not complete chains or posterior certification.",
    }
    (root / "examples/inference/data/tris_calibration_reference_20260915.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-active", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    record(root, root.parent / "rheplicant", allow_active=args.allow_active)
