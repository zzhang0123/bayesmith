"""Archive completed D19 compensation and coordinate experiments, never certify."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from examples.inference.tris_recovery_record import artifact


def record(root, rho, *, allow_active=False):
    directory = rho / "runs/tris-joint-compensation-20260915"
    evidence, programs = {}, {}
    active = []
    specifications = [
        ("compensation", "geometry/summary.json", "geometry", "diagnose_compensation.py"),
        ("simulation", "simulation-map-metric/summary.json", "simulation-map-metric",
         "run_amplitude_reference.py"),
        ("integrator", "integrator-audit/summary.json", "integrator-audit", "audit_integrator.py"),
        ("equivalence", "equivalence.json", "equivalence", "audit_equivalence.py"),
        ("reference_compensation", "reference-compensation.json", "reference-compensation",
         "audit_reference_compensation.py"),
        ("pilot_coordinates", "simulation-quarter-step/pilot_coordinate_audit.json", "pilot-coordinate",
         "audit_pilot_coordinate.py"),
        ("integrated_reference", "integrated-reference-probe.json", "integrated-reference-probe",
         "probe_integrated_reference.py"),
        ("reference_residual", "reference-residual-probe.json", "reference-residual-probe",
         "probe_reference_residual.py"),
        ("calibration_reference", "calibration-reference-probe.json", "calibration-reference-probe",
         "probe_calibration_reference.py"),
        ("explicit_reference_identity", "explicit-reference-equivalence.json", "explicit-reference-equivalence",
         "audit_explicit_reference.py"),
        ("failed_chain", "simulation-quarter-step/failed_chain_audit.json", "failed-chain-audit",
         "audit_failed_chain.py"),
        ("simulation_quarter", "simulation-quarter-step/summary.json", "simulation-quarter-step",
         "run_amplitude_quarter.py"),
    ]
    for name, relative, exit_name, program in specifications:
        entry = artifact(directory / relative)
        exit_path = directory / f"{exit_name}.exit"
        entry["exit_code"] = int(exit_path.read_text()) if exit_path.exists() else None
        running = entry["record"].get("status") in ("initializing", "pilot_running", "chain_running")
        if running and allow_active and entry["exit_code"] is None:
            active.append(name)
        elif running or entry["exit_code"] != 0:
            raise ValueError(f"diagnostic incomplete or active: {name}")
        if entry["record"].get("nonfinite_diagnostic_paths"):
            raise ValueError(f"diagnostic incomplete or invalid: {name}")
        raw = (directory / program).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != entry["record"]["program_sha256"]:
            raise ValueError(f"program changed after diagnostic: {program}")
        programs[program] = {"sha256": digest, "source": raw.decode()}
        evidence[name] = entry
    for name, folder in (("simulation", "simulation-map-metric"),
                         ("simulation_quarter", "simulation-quarter-step")):
        fit = evidence[name]["record"]
        if name not in active and fit["status"] not in (
            "stopped_at_pilot", "stopped_during_chains", "completed"
        ):
            raise ValueError(f"full-target diagnostic still active or invalid: {name}")
        actual = len(list((directory / folder).glob("chain*.npz")))
        if fit["completed_chains"] != actual:
            raise ValueError("recorded and actual formal chains differ")
        evidence[name]["actual_saved_chains"] = actual
        evidence[name]["pilot"] = artifact(directory / folder / "pilot.json")
        evidence[name]["geometry"] = artifact(directory / folder / "geometry.json")
        for filename, expected in fit["source_sha256"].items():
            raw = (directory / folder / "source" / filename).read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f"saved fit source hash differs: {filename}")
    control = artifact(directory / "gaussian-control/gaussian.json")
    control["exit_code"] = int((directory / "gaussian-control.exit").read_text())
    if control["exit_code"] != 0 or not control["record"].get("known_target_pass"):
        raise ValueError("Gaussian target control incomplete")
    evidence["gaussian_control"] = control
    prepared = root / "runs/tris-joint-compensation-20260915/prepared-simulation"
    preparation = artifact(prepared / "reference.json")
    if hashlib.sha256((prepared / "data.npz").read_bytes()).hexdigest() != (
        preparation["record"]["output_sha256"]
    ):
        raise ValueError("prepared reference data changed")
    evidence["preparation"] = preparation
    program = root / "examples/inference/tris_amplitude_reference_prepare.py"
    raw = program.read_bytes()
    if hashlib.sha256(raw).hexdigest() != preparation["record"]["source_sha256"][program.name]:
        raise ValueError("preparation program changed since geometry generation")
    programs[program.name] = {"sha256": hashlib.sha256(raw).hexdigest(), "source": raw.decode()}
    assessment = directory / "assess_amplitude.py"
    raw = assessment.read_bytes()
    programs[assessment.name] = {
        "sha256": hashlib.sha256(raw).hexdigest(), "source": raw.decode(),
        "validation": "Incomplete-run refusal exercised; completed-fit assessment not yet run.",
    }
    tests = {}
    for name in ("amplitude", "regression-final"):
        path = directory / "tests" / f"{name}.xml"
        suites = list(ET.parse(path).getroot().iter("testsuite"))
        tests[name] = {
            key: sum(int(s.attrib.get(key, 0)) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        tests[name].update(
            exit_code=int(path.with_suffix(".exit").read_text()), junit=str(path.resolve()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        if tests[name]["exit_code"] or tests[name]["errors"] or tests[name]["failures"]:
            raise ValueError(f"test run incomplete or failed: {name}")
    result = {
        "status": "record", "date": "2026-09-15", "decision": "D19", "task": "T-001",
        "snapshot_at_utc": datetime.now(UTC).isoformat(),
        "active_experiments": active, "all_experiments_completed": not active,
        "scientific_certification": False, "evidence": evidence, "tests": tests,
        "diagnostic_programs": programs,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation": (
            "Original full physical target retained; local compensation, exact-coordinate "
            "equivalence and integrator stability are distinct from posterior acceptance."
        ),
    }
    output = root / "examples/inference/data/tris_joint_compensation_20260915.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-active", action="store_true",
                        help="Save an explicitly incomplete snapshot while a fit runs.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    record(root, root.parent / "rheplicant", allow_active=args.allow_active)
