"""Record D21's actual prefix state, source bindings and independent controls."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from examples.inference.tris_recovery_record import artifact


def record(root, rho, *, allow_active=False):
    directory = rho / "runs/tris-joint-slice-20260915"
    run = directory / "simulation"
    entry = artifact(run / "summary.json")
    fit = entry["record"]
    exit_path = directory / "simulation.exit"
    entry["exit_code"] = int(exit_path.read_text()) if exit_path.exists() else None
    active = entry["exit_code"] is None
    if active and not allow_active:
        raise ValueError("simulation has not exited; use an explicitly incomplete snapshot")
    entry["actual_saved_complete_chains"] = len(list(run.glob("chain[0-9].npz")))
    if entry["actual_saved_complete_chains"] != fit["completed_chains"]:
        raise ValueError("saved full chains differ from summary checkpoint")
    entry["saved_partial_chains"] = {}
    for path in sorted(run.glob("partial_chain[0-9]_samples.npz")):
        with np.load(path, allow_pickle=False) as data:
            count = len(data["te"])
            if any(value.shape[0] != count for value in data.values()):
                raise ValueError("unaligned saved partial chain")
        entry["saved_partial_chains"][path.name] = {
            "draws": count, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    entry["prefixes"] = [artifact(p) for p in sorted(run.glob("*_prefix*.json"))]
    for name, digest in fit["source_sha256"].items():
        if hashlib.sha256((run / "source" / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"fit source snapshot differs: {name}")
    control = artifact(directory / "gaussian-control/summary.json")
    control["exit_code"] = int((directory / "gaussian-control.exit").read_text())
    if (
        control["exit_code"] or not control["record"]["sampling_pass"]
        or not control["record"]["known_target_pass"]
    ):
        raise ValueError("Gaussian control did not complete successfully")
    current = directory / "gaussian-control/samples.npz"
    old = rho / "runs/tris-em-blocks-20260915/gaussian-control-v4/samples.npz"
    with np.load(current, allow_pickle=False) as new, np.load(old, allow_pickle=False) as prior:
        equality = set(new.files) == set(prior.files) and all(
            np.array_equal(new[k], prior[k]) for k in new.files
        )
    if not equality:
        raise ValueError("Gaussian random stream differs from original control")
    control["independent_array_equality"] = equality
    programs = {}
    for name, item in (("run_joint.py", fit), ("gaussian_control.py", control["record"])):
        raw = (directory / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["program_sha256"]:
            raise ValueError(f"experiment program changed: {name}")
        programs[name] = {"source": raw.decode(), "sha256": hashlib.sha256(raw).hexdigest()}
    assessment = artifact(directory / "assessment-refusal.json")
    raw = (directory / "assess_joint.py").read_bytes()
    if hashlib.sha256(raw).hexdigest() != assessment["record"]["assessment_program_sha256"]:
        raise ValueError("Assessment program changed since its refusal check")
    programs["assess_joint.py"] = {"source": raw.decode(), "sha256": hashlib.sha256(raw).hexdigest()}
    tests = {}
    for name in ("chunks-before", "chunks-before-v2", "chunks-after", "regression"):
        path = directory / "tests" / f"{name}.xml"
        suites = list(ET.parse(path).getroot().iter("testsuite"))
        tests[name] = {
            key: sum(int(s.attrib.get(key, 0)) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        tests[name].update(exit_code=int(path.with_suffix(".exit").read_text()), junit=str(path))
        if name in ("chunks-after", "regression") and (
            tests[name]["exit_code"] or tests[name]["failures"] or tests[name]["errors"]
        ):
            raise ValueError(f"validation failed or incomplete: {name}")
    result = {
        "status": "record", "decision": "D21", "date": "2026-09-15", "task": "T-001",
        "snapshot_at_utc": datetime.now(UTC).isoformat(),
        "all_experiments_completed": not active,
        "active_experiments": ["simulation"] if active else [],
        "scientific_certification": False,
        "evidence": {
            "simulation": entry, "gaussian_control": control, "assessment_refusal": assessment,
        },
        "diagnostic_programs": programs, "tests": tests,
        "test_history": (
            "Initial new Gibbs fixture used key instead of the required rng_key keyword; "
            "after fixture correction, the existing NUTS-only chunk guard caused the two "
            "expected failures. Both cases pass after supporting Gibbs/NUTS continuation."
        ),
        "proposal_evidence": "tris_calibration_reference_20260915.json: joint_move_probe",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation": (
            "Complete original physical target; weak EM conditional slices move physical "
            "A through its reference. Conditional HMC energy is not whole-target HMC energy."
        ),
    }
    target = root / "examples/inference/data/tris_joint_slice_20260915.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-active", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    record(root, root.parent / "rheplicant", allow_active=args.allow_active)
