"""Archive D18's completed block-kernel experiments with source and exit checks."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from examples.inference.tris_recovery_record import artifact


def record(root, rho):
    directory = rho / "runs/tris-em-blocks-20260915"
    evidence, programs = {}, {}
    specifications = [
        ("simulation", "simulation/summary.json", "simulation", "run_blocked.py"),
        ("gaussian_control", "gaussian-control-v2/summary.json", "gaussian-control-v2",
         "gaussian_control_v2.py"),
        ("slow_mode", "simulation/slow_mode_audit.json", "slow-mode", "audit_slow_mode.py"),
        ("slice_probe", "simulation/slice_probe.json", "slice-probe", "probe_slice.py"),
        ("subblocks_control_short", "gaussian-control-v3/summary.json", "gaussian-control-v3",
         "gaussian_control_v3.py"),
        ("subblocks_control", "gaussian-control-v4/summary.json", "gaussian-control-v4",
         "gaussian_control_v4.py"),
        ("subblocks_simulation", "simulation-subblocks4-v2/summary.json", "simulation-subblocks4-v2",
         "run_subblocked_v2.py"),
        ("subblocks_slow_mode", "simulation-subblocks4-v2/slow_mode_audit.json", "subblocks-slow-mode",
         "audit_subblock_slow.py"),
        ("efficiency_audit", "efficiency-audit/after-all.json", "efficiency-audit-all",
         "audit_efficiency_all.py"),
    ]
    for size in (4, 17):
        specifications.append((
            f"slice_probe_blocks{size}", f"simulation/slice_probe_blocks{size}.json",
            f"slice-probe-blocks{size}", f"probe_slice_blocks{size}.py",
        ))
    for name, relative, exit_name, program in specifications:
        entry = artifact(directory / relative)
        entry["exit_code"] = int((directory / f"{exit_name}.exit").read_text())
        if entry["exit_code"] != 0 or entry["record"].get("nonfinite_diagnostic_paths"):
            raise ValueError(f"diagnostic incomplete or invalid: {name}")
        raw = (directory / program).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != entry["record"]["program_sha256"]:
            raise ValueError(f"program changed after diagnostic: {program}")
        programs[program] = {"sha256": digest, "source": raw.decode()}
        evidence[name] = entry
    for name, folder in (("simulation", "simulation"), ("subblocks_simulation", "simulation-subblocks4-v2")):
        fit = evidence[name]["record"]
        if fit["status"] in ("prepared", "pilot_running", "chain_running"):
            raise ValueError("full-target diagnostic still active")
        actual = len(list((directory / folder).glob("chain*.npz")))
        if fit["completed_chains"] != actual:
            raise ValueError("recorded and actual formal chains differ")
        evidence[name]["actual_saved_chains"] = actual
        evidence[f"{name}_pilot"] = artifact(directory / folder / "pilot.json")
    log = (directory / "simulation-subblocks4.log").read_bytes()
    evidence["subblocks_precondition_rejection"] = {
        "exit_code": int((directory / "simulation-subblocks4.exit").read_text()),
        "log": log.decode(), "log_sha256": hashlib.sha256(log).hexdigest(),
        "output_files": sorted(p.name for p in (directory / "simulation-subblocks4").iterdir()),
        "interpretation": "Rejected by the failed short-control prerequisite before initialization or sampling.",
    }
    evidence["efficiency_bug_reproduction"] = artifact(directory / "efficiency-audit/before.json")
    tests = {}
    for name in ("kernel", "regression", "subblocks", "efficiency", "regression-final"):
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
    result = {
        "status": "record", "date": "2026-09-15", "decision": "D18", "task": "T-001",
        "scientific_certification": False, "evidence": evidence, "tests": tests,
        "diagnostic_programs": programs,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation": (
            "Original full physical target retained; actual stage results remain separate. "
            "Known-target controls and fixed-state movement probes do not admit a posterior."
        ),
    }
    output = root / "examples/inference/data/tris_em_blocks_20260915.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    record(root, root.parent / "rheplicant")
