"""Archive the D22 computation policy, controls, tests and actual physical state."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from examples.inference.tris_recovery_record import artifact


def record(root, rho, *, allow_active=False):
    directory = rho / "runs/tris-multichain-budget-20260915"
    run = directory / "simulation"
    fit = artifact(run / "summary.json")
    value = fit["record"]
    exit_path = directory / "simulation.exit"
    active = not exit_path.exists()
    active_names = ["simulation"] if active else []
    if active and not allow_active:
        raise ValueError("physical experiment is still active; explicit incomplete snapshot required")
    fit["exit_code"] = None if active else int(exit_path.read_text())
    fit["actual_saved_complete_chains"] = len(list(run.glob("chain[0-9].npz")))
    if fit["actual_saved_complete_chains"] != value["completed_chains"]:
        raise ValueError("complete-chain files and recorded status differ")
    fit["latest_final_gate_draws"] = max(
        (n for n in value["assessment_draws"] if n <= value["draws"]), default=None,
    )
    fit["diagnostics_current_for_saved_draws"] = (
        fit["latest_final_gate_draws"] == value["draws"]
    )
    draws = 0
    for item in value["chunks"]:
        path = run / item["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("saved chunk hash differs")
        with np.load(path, allow_pickle=False) as data:
            size = data["sample_te"].shape[1]
            if any(v.shape[:2] != (4, size) for v in data.values()):
                raise ValueError("unaligned chain/sample dimensions")
        draws += size
    if draws != value["draws"]:
        raise ValueError("saved chunk draws and summary differ")
    if draws:
        state = run / value["checkpoint"]["file"]
        if hashlib.sha256(state.read_bytes()).hexdigest() != value["checkpoint"]["sha256"]:
            raise ValueError("saved full-state checkpoint differs")
    for name, sha in value["source_sha256"].items():
        if hashlib.sha256((run / "source" / name).read_bytes()).hexdigest() != sha:
            raise ValueError(f"source snapshot differs: {name}")
    evidence = {"simulation": fit}
    for name, filename in (("screen_control", "screen-control.json"),
                           ("gaussian_control", "gaussian-control/summary.json")):
        evidence[name] = artifact(directory / filename)
        evidence[name]["exit_code"] = int((directory / f"{name.replace('_', '-')}.exit").read_text())
    if (directory / "d21-prefix-audit.json").exists():
        evidence["d21_prefix_audit"] = artifact(directory / "d21-prefix-audit.json")
        evidence["d21_prefix_audit"]["exit_code"] = int(
            (directory / "d21-prefix-audit.exit").read_text()
        )
    if (directory / "assessment-refusal.json").exists():
        evidence["assessment_refusal"] = artifact(directory / "assessment-refusal.json")
    if (directory / "science-assessment-refusal.json").exists():
        evidence["science_assessment_refusal"] = artifact(directory / "science-assessment-refusal.json")
    if (directory / "source-convention-probe.json").exists():
        evidence["source_convention_probe"] = artifact(directory / "source-convention-probe.json")
        evidence["source_convention_probe"]["exit_code"] = int(
            (directory / "source-convention-probe.exit").read_text()
        )
    if (directory / "truth-audit.json").exists():
        evidence["truth_audit"] = artifact(directory / "truth-audit.json")
        evidence["truth_audit"]["exit_code"] = int((directory / "truth-audit.exit").read_text())
    evidence["prefix_inspections"] = [artifact(p) for p in sorted(directory.glob("inspection_*.json"))]
    evidence["physical_prefix_probes"] = [
        artifact(p) for p in sorted(directory.glob("physical-probe-*.json"))
    ]
    programs = {}
    for name, expected in (
        ("run_budget.py", value["program_sha256"]),
        ("screen_control.py", evidence["screen_control"]["record"]["program_sha256"]),
        ("gaussian_control.py", evidence["gaussian_control"]["record"]["program_sha256"]),
        ("assess_budget.py", None),
        ("assess_science.py", None),
        ("inspect_prefix.py", None),
    ):
        data = (directory / name).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if expected is not None and expected != sha:
            raise ValueError(f"executed program changed: {name}")
        programs[name] = {"source": data.decode(), "sha256": sha}
    if "d21_prefix_audit" in evidence:
        raw = (directory / "audit_d21_prefix.py").read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if evidence["d21_prefix_audit"]["record"]["program_sha256"] != sha:
            raise ValueError("D21 prefix audit program changed")
        programs["audit_d21_prefix.py"] = {"source": raw.decode(), "sha256": sha}
    if "assessment_refusal" in evidence and (
        evidence["assessment_refusal"]["record"]["assessment_program_sha256"]
        != programs["assess_budget.py"]["sha256"]
    ):
        raise ValueError("Assessment program changed since its refusal test")
    if "science_assessment_refusal" in evidence and (
        evidence["science_assessment_refusal"]["record"]["program_sha256"]
        != programs["assess_science.py"]["sha256"]
    ):
        raise ValueError("Science assessment program changed since its refusal test")
    if "source_convention_probe" in evidence:
        raw = (directory / "source_convention_probe.py").read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if evidence["source_convention_probe"]["record"]["program_sha256"] != sha:
            raise ValueError("Source convention probe program changed")
        programs["source_convention_probe.py"] = {"source": raw.decode(), "sha256": sha}
    if "truth_audit" in evidence:
        raw = (directory / "truth_audit.py").read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if evidence["truth_audit"]["record"]["program_sha256"] != sha:
            raise ValueError("Truth audit program changed")
        programs["truth_audit.py"] = {"source": raw.decode(), "sha256": sha}
    for item in evidence["prefix_inspections"]:
        if item["record"]["program_sha256"] != programs["inspect_prefix.py"]["sha256"]:
            raise ValueError("Prefix inspector source changed")
    for key, filename, program in (
        ("prior_volume_control", "prior-volume-control.json", "prior_volume_control.py"),
        ("hierarchical_prior_control", "hierarchical-prior-control.json",
         "hierarchical_prior_control.py"),
        ("hierarchical_physical_check", "hierarchical-physical-check.json",
         "hierarchical_physical_check.py"),
    ):
        path = directory / filename
        if not path.exists():
            continue
        evidence[key] = artifact(path)
        evidence[key]["exit_code"] = int(path.with_suffix(".exit").read_text())
        data = (directory / program).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if evidence[key]["record"]["program_sha256"] != sha:
            raise ValueError(f"Executed control changed: {program}")
        programs[program] = {"source": data.decode(), "sha256": sha}
    if evidence["physical_prefix_probes"]:
        data = (directory / "physical_prefix_probe.py").read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if any(item["record"]["program_sha256"] != sha
               for item in evidence["physical_prefix_probes"]):
            raise ValueError("Physical prefix probe source changed")
        programs["physical_prefix_probe.py"] = {"source": data.decode(), "sha256": sha}
    tests = {}
    for name in ("checkpoint-tests", "workflow-tests", "regression",
                 "recovery-tests", "recovery-final-tests", "hierarchical-tests",
                 "hierarchical-final-tests", "rheplicant-suite", "scale-coordinate-tests",
                 "scale-coordinate-final-tests"):
        path = directory / f"{name}.xml"
        if not path.exists() or not path.with_suffix(".exit").exists():
            tests[name] = {"status": "not_completed"}
            continue
        suites = list(ET.parse(path).getroot().iter("testsuite"))
        tests[name] = {k: sum(int(s.get(k, 0)) for s in suites)
                       for k in ("tests", "failures", "errors", "skipped")}
        tests[name].update(junit=str(path), exit_code=int(path.with_suffix(".exit").read_text()))
    for name in ("bayesmith-fast", "astronomy-quadrature", "astronomy-quadrature-venv",
                 "astronomy-quadrature-local"):
        path = root / "runs/tris-multichain-budget-20260915" / f"{name}.xml"
        if not path.exists() or not path.with_suffix(".exit").exists():
            tests[name] = {"status": "not_completed"}
            continue
        suites = list(ET.parse(path).getroot().iter("testsuite"))
        tests[name] = {k: sum(int(s.get(k, 0)) for s in suites)
                       for k in ("tests", "failures", "errors", "skipped")}
        tests[name].update(junit=str(path), exit_code=int(path.with_suffix(".exit").read_text()))
    for name in ("run_astronomy_tests.py", "plot_prior_control.py", "plot_variance_control.py"):
        data = (root / "runs/tris-multichain-budget-20260915" / name).read_bytes()
        programs[name] = {"source": data.decode(), "sha256": hashlib.sha256(data).hexdigest()}
    if (run / "assessment/summary.json").exists():
        evidence["independent_assessment"] = artifact(run / "assessment/summary.json")
    if (run / "assessment/ppc.json").exists():
        evidence["conditional_ppc"] = artifact(run / "assessment/ppc.json")
    candidate = rho / "runs/tris-hierarchical-scale-20260915"
    if (candidate / "preparation/summary.json").exists():
        evidence["candidate_preparation"] = artifact(candidate / "preparation/summary.json")
        exit_path = candidate / "preparation.exit"
        evidence["candidate_preparation"]["exit_code"] = (
            int(exit_path.read_text()) if exit_path.exists() else None
        )
        if not exit_path.exists():
            active_names.append("candidate_preparation")
        data = (candidate / "prepare.py").read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if evidence["candidate_preparation"]["record"]["program_sha256"] != sha:
            raise ValueError("Candidate preparation program changed")
        programs["candidate_prepare.py"] = {"source": data.decode(), "sha256": sha}
    if (candidate / "initialization-review.json").exists():
        evidence["candidate_initialization_review"] = artifact(
            candidate / "initialization-review.json"
        )
    if (candidate / "anchor-preparation/summary.json").exists():
        evidence["candidate_anchor_preparation"] = artifact(candidate / "anchor-preparation/summary.json")
        exit_path = candidate / "anchor-preparation.exit"
        evidence["candidate_anchor_preparation"]["exit_code"] = (
            int(exit_path.read_text()) if exit_path.exists() else None
        )
        if not exit_path.exists():
            active_names.append("candidate_anchor_preparation")
        data = (candidate / "prepare_anchors.py").read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if evidence["candidate_anchor_preparation"]["record"]["program_sha256"] != sha:
            raise ValueError("Anchor preparation program changed")
        programs["candidate_prepare_anchors.py"] = {"source": data.decode(), "sha256": sha}
    for label, program in (("nonlinear-control", "nonlinear_control.py"),
                           ("nonlinear-control-fine", "nonlinear_control_fine.py"),
                           ("pilot", "pilot.py")):
        path = candidate / label / "summary.json"
        if not path.exists():
            continue
        key = "candidate_" + label.replace("-", "_")
        evidence[key] = artifact(path)
        exit_path = candidate / f"{label}.exit"
        evidence[key]["exit_code"] = int(exit_path.read_text()) if exit_path.exists() else None
        if not exit_path.exists():
            active_names.append(key)
        data = (candidate / program).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        recorded = evidence[key]["record"]
        expected = recorded.get("program_sha256", recorded.get("binding", {}).get("program_sha256"))
        if expected != sha:
            raise ValueError(f"Candidate control program changed: {program}")
        programs["candidate_" + program] = {"source": data.decode(), "sha256": sha}
    variance = rho / "runs/tris-variance-gibbs-20260915"
    for label, program in (("preparation", "prepare.py"), ("control", "control.py"),
                           ("pilot", "pilot.py"), ("simulation", "run_chains.py"),
                           ("assessment", "assess_chains.py")):
        folder = "simulation/assessment" if label == "assessment" else label
        path = variance / folder / "summary.json"
        if not path.exists():
            continue
        key = "variance_gibbs_" + label
        evidence[key] = artifact(path)
        exit_path = variance / f"{label}.exit"
        evidence[key]["exit_code"] = int(exit_path.read_text()) if exit_path.exists() else None
        if not exit_path.exists():
            active_names.append(key)
        data = (variance / program).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        recorded = evidence[key]["record"]
        expected = recorded.get("program_sha256", recorded.get("binding", {}).get("program_sha256"))
        if label == "assessment":
            expected = recorded["assessment_program_sha256"]
        if expected != sha:
            raise ValueError(f"Variance Gibbs program changed: {program}")
        programs["variance_gibbs_" + program] = {"source": data.decode(), "sha256": sha}
        if label == "simulation":
            saved_draws = 0
            for chunk in recorded["chunks"]:
                chunk_path = variance / label / chunk["file"]
                if hashlib.sha256(chunk_path.read_bytes()).hexdigest() != chunk["sha256"]:
                    raise ValueError("Variance Gibbs chunk changed")
                with np.load(chunk_path, allow_pickle=False) as arrays:
                    size = arrays["sample_te"].shape[1]
                    if any(v.shape[:2] != (4, size) for v in arrays.values()):
                        raise ValueError("Variance Gibbs sample shapes differ")
                    saved_draws += size
            if saved_draws != recorded["draws"]:
                raise ValueError("Variance Gibbs saved draws differ")
            if saved_draws:
                checkpoint = recorded["checkpoint"]
                raw = (variance / label / checkpoint["file"]).read_bytes()
                if hashlib.sha256(raw).hexdigest() != checkpoint["sha256"]:
                    raise ValueError("Variance Gibbs checkpoint changed")
            evidence[key]["latest_final_gate_draws"] = recorded["last_final_gate_draws"]
    for label in ("execution-contract", "entry-refusal", "initial-device-error", "quadrature-refusal",
                  "prefix-recovery-01000", "posterior-quadrature", "prior-sensitivity"):
        if (variance / f"{label}.json").exists():
            evidence["variance_gibbs_" + label.replace("-", "_")] = artifact(
                variance / f"{label}.json"
            )
    for name in ("assess_chains.py", "run_chains_before_device_preflight.py", "check_quadrature.py",
                 "inspect_recovery_prefix.py", "prior_sensitivity.py"):
        path = variance / name
        if path.exists():
            data = path.read_bytes()
            programs["variance_gibbs_" + name] = {
                "source": data.decode(), "sha256": hashlib.sha256(data).hexdigest(),
            }
    for label in ("ppc", "recovery"):
        path = variance / "simulation/assessment" / f"{label}.json"
        if path.exists():
            evidence["variance_gibbs_" + label] = artifact(path)
    refusal = evidence.get("variance_gibbs_entry_refusal", {}).get("record", {})
    for name, expected in refusal.get("program_sha256", {}).items():
        if programs["variance_gibbs_" + name]["sha256"] != expected:
            raise ValueError("Variance Gibbs entry changed since refusal test")
    path = variance / "tests/junit.xml"
    if path.exists() and (variance / "tests/exit").exists():
        suites = list(ET.parse(path).getroot().iter("testsuite"))
        tests["variance-gibbs"] = {k: sum(int(s.get(k, 0)) for s in suites)
                                    for k in ("tests", "failures", "errors", "skipped")}
        tests["variance-gibbs"].update(
            junit=str(path), exit_code=int((variance / "tests/exit").read_text()),
        )
    path = variance / "regression.xml"
    if path.exists() and path.with_suffix(".exit").exists():
        suites = list(ET.parse(path).getroot().iter("testsuite"))
        tests["variance-gibbs-regression"] = {k: sum(int(s.get(k, 0)) for s in suites)
                                               for k in ("tests", "failures", "errors", "skipped")}
        tests["variance-gibbs-regression"].update(
            junit=str(path), exit_code=int(path.with_suffix(".exit").read_text()),
        )
    fine = rho / "runs/tris-variance-fine-20260915"
    warm_execution = (fine / "real-warm-execution-contract.json").is_file()
    real_program = "run_real_warm.py" if warm_execution else "run_real.py"
    real_assessor = "assess_real_warm.py" if warm_execution else "assess_real.py"
    real_quadrature = "check_real_quadrature_warm.py" if warm_execution else "check_real_quadrature.py"
    for label, filename, program in (
        ("preparation", "preparation/summary.json", "prepare.py"),
        ("pilot", "pilot/summary.json", "pilot.py"),
        ("simulation", "simulation/summary.json", "run_chains.py"),
        ("assessment", "simulation/assessment/summary.json", "assess_chains.py"),
        ("real-preparation", "real-prepared/manifest.json", "prepare_real_input.py"),
        ("real-geometry", "real-preparation/summary.json", "prepare_real_geometry.py"),
        ("real-metric-candidate", "real-metric-candidate/summary.json", "prepare_real_metric_candidate.py"),
        ("real-candidate-pilot", "real-candidate-pilot/summary.json", "pilot_real_candidate.py"),
        ("real-reference-scale-preparation", "real-reference-scale-preparation/summary.json", "prepare_reference_scale_geometry.py"),
        ("reference-scale-control", "reference-scale-control/summary.json", "control_reference_scale.py"),
        ("real-warm-reference", "real-warm-reference/summary.json", "prepare_real_warm_reference.py"),
        ("real-warm-metric-pilot", "real-warm-metric-pilot/summary.json", "pilot_real_warm_metric.py"),
        ("real-pilot", "real-pilot/summary.json", "pilot_real.py"),
        ("real-warm-starts", "real-warm-starts/summary.json", "prepare_real_warm_starts.py"),
        ("real", "real/summary.json", real_program),
        ("real-assessment", "real/assessment/summary.json", real_assessor),
        ("real-adaptive-reference", "real-adaptive-reference/summary.json", "prepare_real_adaptive_reference.py"),
        ("real-adaptive", "real-adaptive/summary.json", "run_real_adaptive.py"),
        ("real-adaptive-assessment", "real-adaptive/assessment/summary.json", "assess_real_adaptive.py"),
        ("real-refined-reference", "real-refined-reference/summary.json", "prepare_real_refined_reference.py"),
        ("real-refined-pilot", "real-refined-pilot/summary.json", "pilot_real_refined.py"),
        ("real-refined", "real-refined/summary.json", "run_real_refined.py"),
        ("real-refined-assessment", "real-refined/assessment/summary.json", "assess_real_refined.py"),
    ):
        path = fine / filename
        if not path.exists():
            continue
        key = "variance_fine_" + label.replace("-", "_")
        evidence[key] = artifact(path)
        recorded = evidence[key]["record"]
        exit_path = fine / f"{label}.exit"
        evidence[key]["exit_code"] = int(exit_path.read_text()) if exit_path.exists() else None
        if not exit_path.exists():
            active_names.append(key)
        data = (fine / program).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        expected = recorded.get("program_sha256", recorded.get("binding", {}).get("program_sha256"))
        if label in ("assessment", "real-assessment", "real-adaptive-assessment", "real-refined-assessment"):
            expected = recorded["assessment_program_sha256"]
        if expected != sha:
            raise ValueError(f"Fine-grid program changed: {program}")
        programs["variance_fine_" + program] = {"source": data.decode(), "sha256": sha}
        if label in ("simulation", "real", "real-adaptive", "real-refined"):
            saved_draws = 0
            for chunk in recorded["chunks"]:
                chunk_path = fine / label / chunk["file"]
                if hashlib.sha256(chunk_path.read_bytes()).hexdigest() != chunk["sha256"]:
                    raise ValueError("Fine-grid chunk changed")
                with np.load(chunk_path, allow_pickle=False) as arrays:
                    size = arrays["sample_te"].shape[1]
                    if any(v.shape[:2] != (4, size) for v in arrays.values()):
                        raise ValueError("Fine-grid sample shapes differ")
                    saved_draws += size
            if saved_draws != recorded["draws"]:
                raise ValueError("Fine-grid saved draws differ")
            if saved_draws:
                checkpoint = recorded["checkpoint"]
                data = (fine / label / checkpoint["file"]).read_bytes()
                if hashlib.sha256(data).hexdigest() != checkpoint["sha256"]:
                    raise ValueError("Fine-grid checkpoint changed")
            warmup_count = 0
            for chunk in recorded.get("warmup_chunks", []):
                chunk_path = fine / label / chunk["file"]
                if hashlib.sha256(chunk_path.read_bytes()).hexdigest() != chunk["sha256"]:
                    raise ValueError("Fine-grid warmup chunk changed")
                with np.load(chunk_path, allow_pickle=False) as arrays:
                    size = arrays["sample_te"].shape[1]
                    if any(v.shape[:2] != (4, size) for v in arrays.values()):
                        raise ValueError("Fine-grid warmup shapes differ")
                    warmup_count += size
                if chunk["iteration"] != warmup_count:
                    raise ValueError("Fine-grid warmup iterations differ")
            if warmup_count != recorded.get("warmup_iteration", 0):
                raise ValueError("Fine-grid saved warmup differs")
            if warmup_count:
                checkpoint = recorded["warmup_checkpoint"]
                data = (fine / label / checkpoint["file"]).read_bytes()
                if hashlib.sha256(data).hexdigest() != checkpoint["sha256"]:
                    raise ValueError("Fine-grid warmup checkpoint changed")
            evidence[key]["latest_final_gate_draws"] = recorded["last_final_gate_draws"]
    for label in ("execution-contract", "entry-refusal", "posterior-quadrature",
                  "quadrature-refusal", "prior-sensitivity", "beam-refinement",
                  "real-entry-refusal", "real-initialization-error", "real-execution-contract",
                  "formal-start-probe", "real-diagnostic-refusal", "real-posterior-quadrature",
                  "real-sensitivity", "runtime-probe-default", "runtime-probe-limited",
                  "runtime-comparison", "ra-deletion-verification", "real-ra-deletion",
                  "sensitivity-review-verification", "real-sensitivity-review",
                  "real-initial-cost-probe", "real-metric-drift-probe",
                  "real-metric-candidate-probe", "visible-warmup-verification",
                  "visible-warmup-restart-verification", "active-frequency-verification",
                  "reference-scale-verification", "real-pilot-operator-stop",
                  "warmup-metric-verification", "parallel-warmup-verification",
                  "real-warm-formal-start-probe", "real-warm-execution-contract",
                  "simulation-collapsed-offset-review", "real-collapsed-offset-review",
                  "real-four-chain-curvature", "real-operator-stop", "real-adaptive-execution-contract",
                  "adaptive-window-audit", "adaptive-warm-numerics", "refined-warm-numerics",
                  "real-adaptive-quadrature-stop", "real-refined-execution-contract",
                  "real-refined-posterior-quadrature", "real-refined-sensitivity",
                  "real-refined-ra-deletion", "real-refined-sensitivity-review",
                  "real-refined-collapsed-offset-review",
                  "real-adaptive-posterior-quadrature", "real-adaptive-sensitivity",
                  "real-adaptive-ra-deletion", "real-adaptive-sensitivity-review",
                  "real-adaptive-collapsed-offset-review"):
        path = fine / f"{label}.json"
        if path.exists():
            key = "variance_fine_" + label.replace("-", "_")
            evidence[key] = artifact(path)
            exit_path = path.with_suffix(".exit")
            if label == "adaptive-warm-numerics":
                exit_path = fine / "adaptive-warm-numerics-checked.exit"
            if exit_path.exists():
                evidence[key]["exit_code"] = int(exit_path.read_text())
    refinement = fine.parent / "tris-adaptive-refinement-20260915"
    for label, filename, exit_path in (
        ("haslam512_geometry", "geometry-summary.json",
         root / "runs/tris-adaptive-refinement-20260915/haslam512-geometry.exit"),
        ("haslam_refinement", "haslam-refinement.json", refinement / "haslam-refinement-v2.exit"),
        ("joint512_preparation", "joint512-preparation.json",
         root / "runs/tris-adaptive-refinement-20260915/joint512-preparation.exit"),
    ):
        path = refinement / filename
        if not path.is_file():
            continue
        evidence[label] = artifact(path)
        evidence[label]["exit_code"] = int(exit_path.read_text()) if exit_path.exists() else None
        for name, sha in evidence[label]["record"].get("artifact_sha256", {}).items():
            if hashlib.sha256((refinement / name).read_bytes()).hexdigest() != sha:
                raise ValueError("Refinement artifact changed")
    for path in (
        root / "runs/tris-adaptive-refinement-20260915/prepare_haslam512_geometry.py",
        root / "runs/tris-adaptive-refinement-20260915/prepare_joint512.py",
        refinement / "check_haslam_refinement.py", refinement / "check_haslam_refinement_v2.py",
    ):
        if path.is_file():
            data = path.read_bytes()
            programs[path.name] = {"source": data.decode(), "sha256": hashlib.sha256(data).hexdigest()}
    for name in ("run_chains.py", "assess_chains.py", "check_quadrature.py",
                 "real_admission.py", "pilot_real.py", "run_real.py", "assess_real.py",
                 "prepare_real_geometry_before_latent_filter.py", "probe_formal_starts.py",
                 "check_real_quadrature.py", "real_sensitivity.py", "component_report.py",
                 "probe_runtime_threads.py", "real_ra_deletion.py", "verify_ra_deletion.py",
                 "review_real_sensitivities.py", "verify_sensitivity_review.py",
                 "probe_real_initial_cost.py", "probe_real_metric_drift.py",
                 "probe_real_metric_candidate.py", "visible_warmup.py",
                 "verify_visible_warmup.py", "verify_visible_warmup_restart.py",
                 "active_frequency_model.py", "verify_active_frequency_model.py",
                 "reference_scaled_normal.py", "verify_reference_scaled_normal.py",
                 "warmup_metric.py", "verify_warmup_metric.py",
                 "parallel_warmup.py", "verify_parallel_warmup.py", "probe_real_warm_starts.py",
                 "prepare_real_warm_contract.py", "run_real_warm.py", "assess_real_warm.py",
                 "check_real_quadrature_warm.py", "review_collapsed_offsets.py",
                 "probe_real_four_chain_curvature.py", "prepare_real_adaptive_reference.py",
                 "prepare_real_adaptive_contract.py", "run_real_adaptive.py",
                 "assess_real_adaptive.py", "check_real_quadrature_adaptive.py",
                 "real_adaptive_sensitivity.py", "real_adaptive_ra_deletion.py",
                 "review_real_adaptive_sensitivities.py", "review_adaptive_collapsed_offsets.py",
                 "probe_adaptive_warm_numerics.py", "probe_adaptive_warm_numerics_checked.py",
                 "probe_refined_warm_numerics.py", "prepare_real_refined_reference.py",
                 "pilot_real_refined.py", "run_real_refined.py", "assess_real_refined.py",
                 "check_real_quadrature_refined.py", "real_refined_sensitivity.py",
                 "real_refined_ra_deletion.py", "review_real_refined_sensitivities.py",
                 "review_refined_collapsed_offsets.py"):
        path = fine / name
        if path.exists():
            data = path.read_bytes()
            programs["variance_fine_" + name] = {
                "source": data.decode(), "sha256": hashlib.sha256(data).hexdigest(),
            }
    refusal = evidence.get("variance_fine_entry_refusal", {}).get("record", {})
    for name, expected in refusal.get("program_sha256", {}).items():
        if programs["variance_fine_" + name]["sha256"] != expected:
            raise ValueError("Fine-grid entry changed since refusal test")
    refusal = evidence.get("variance_fine_real_entry_refusal", {}).get("record", {})
    for name, expected in refusal.get("program_sha256", {}).items():
        if programs["variance_fine_" + name]["sha256"] != expected:
            raise ValueError("Real entry changed since refusal test")
    refusal = evidence.get("variance_fine_real_diagnostic_refusal", {}).get("record", {})
    for name, expected in refusal.get("program_sha256", {}).items():
        if programs["variance_fine_" + name]["sha256"] != expected:
            raise ValueError("Real diagnostic changed since refusal test")
    for label, program in (
        ("formal_start_probe", "probe_formal_starts.py"),
        ("posterior_quadrature", "check_quadrature.py"),
        ("real_posterior_quadrature", real_quadrature),
        ("real_warm_formal_start_probe", "probe_real_warm_starts.py"),
        ("real_warm_execution_contract", "prepare_real_warm_contract.py"),
        ("simulation_collapsed_offset_review", "review_collapsed_offsets.py"),
        ("real_collapsed_offset_review", "review_collapsed_offsets.py"),
        ("real_sensitivity", "real_sensitivity.py"),
        ("runtime_probe_default", "probe_runtime_threads.py"),
        ("runtime_probe_limited", "probe_runtime_threads.py"),
        ("real_ra_deletion", "real_ra_deletion.py"),
        ("ra_deletion_verification", "real_ra_deletion.py"),
        ("real_sensitivity_review", "review_real_sensitivities.py"),
        ("sensitivity_review_verification", "review_real_sensitivities.py"),
        ("real_initial_cost_probe", "probe_real_initial_cost.py"),
        ("real_metric_drift_probe", "probe_real_metric_drift.py"),
        ("real_metric_candidate_probe", "probe_real_metric_candidate.py"),
        ("visible_warmup_verification", "verify_visible_warmup.py"),
        ("active_frequency_verification", "verify_active_frequency_model.py"),
        ("reference_scale_verification", "verify_reference_scaled_normal.py"),
    ):
        result = evidence.get("variance_fine_" + label, {}).get("record")
        if result is not None and result["program_sha256"] != programs["variance_fine_" + program]["sha256"]:
            raise ValueError(f"Executed fine-grid diagnostic changed: {program}")
    for label in ("real-metric-drift-probe", "real-metric-candidate-probe",
                  "real-warm-formal-start-probe"):
        probe = evidence.get("variance_fine_" + label.replace("-", "_"), {}).get("record")
        if probe is not None:
            states = fine / f"{label}-states.npz"
            if hashlib.sha256(states.read_bytes()).hexdigest() != probe["states_sha256"]:
                raise ValueError(f"Real metric probe states changed: {label}")
    candidate = evidence.get("variance_fine_real_metric_candidate", {}).get("record")
    if candidate is not None:
        for name, sha in candidate.get("artifact_sha256", {}).items():
            path = fine / "real-metric-candidate" / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise ValueError(f"Real candidate metric artifact changed: {name}")
        for name, sha in candidate["input_sha256"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha:
                raise ValueError(f"Real candidate metric input changed: {name}")
    verification = evidence.get("variance_fine_visible_warmup_verification", {}).get("record")
    if (verification is not None
            and verification["helper_sha256"] != programs["variance_fine_visible_warmup.py"]["sha256"]):
        raise ValueError("Verified warmup helper changed")
    for label in ("visible_warmup_restart_verification", "warmup_metric_verification",
                  "parallel_warmup_verification"):
        restart = evidence.get("variance_fine_" + label, {}).get("record")
        if restart is not None:
            for name, sha in restart["source_sha256"].items():
                if sha != programs["variance_fine_" + name]["sha256"]:
                    raise ValueError(f"Warmup verification source changed: {name}")
    for label, helper in (("active_frequency_verification", "active_frequency_model.py"),
                          ("reference_scale_verification", "reference_scaled_normal.py"),
                          ("real_reference_scale_preparation", "reference_scaled_normal.py")):
        verified = evidence.get("variance_fine_" + label, {}).get("record")
        if verified is not None and verified["helper_sha256"] != programs["variance_fine_" + helper]["sha256"]:
            raise ValueError(f"Verified helper changed: {helper}")
    for label in ("real-reference-scale-preparation", "real-warm-reference", "real-warm-starts"):
        prepared = evidence.get("variance_fine_" + label.replace("-", "_"), {}).get("record")
        if prepared is not None:
            for name, sha in prepared.get("artifact_sha256", {}).items():
                if hashlib.sha256((fine / label / name).read_bytes()).hexdigest() != sha:
                    raise ValueError(f"Prepared artifact changed: {label}/{name}")
            for name, sha in prepared["input_sha256"].items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha:
                    raise ValueError(f"Preparation input changed: {name}")
    stopped = evidence.get("variance_fine_real_pilot_operator_stop", {}).get("record")
    if stopped is not None:
        original = fine / "real-pilot/summary-before-operator-stop.json"
        if hashlib.sha256(original.read_bytes()).hexdigest() != stopped["summary_before_sha256"]:
            raise ValueError("Preserved original pilot summary changed")
    control = evidence.get("variance_fine_reference_scale_control", {}).get("record")
    if control is not None:
        if control["binding"]["helper_sha256"] != programs["variance_fine_reference_scaled_normal.py"]["sha256"]:
            raise ValueError("Reference-scale control helper changed")
        for name in ("samples.npz", "extras.npz"):
            path = fine / "reference-scale-control" / name
            if path.exists():
                evidence["variance_fine_reference_scale_control"][name] = {
                    "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
    for label in ("real-candidate-pilot", "real-warm-metric-pilot", "real-refined-pilot"):
        pilot = evidence.get("variance_fine_" + label.replace("-", "_"), {}).get("record")
        if pilot is not None:
            for chunk in pilot["chunks"]:
                path = fine / label / chunk["file"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != chunk["sha256"]:
                    raise ValueError(f"Pilot chunk changed: {label}")
            if "checkpoint" in pilot:
                checkpoint = pilot["checkpoint"]
                path = fine / label / checkpoint["file"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != checkpoint["sha256"]:
                    raise ValueError(f"Pilot checkpoint changed: {label}")
    profile = fine / "real-pilot-process-sample.txt"
    if profile.exists():
        evidence["variance_fine_real_pilot_process_sample"] = {
            "path": str(profile), "sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
            "exit_code": int(profile.with_suffix(".exit").read_text()),
            "interpretation": "Read-only macOS stack sample; not a numerical or sampling test.",
        }
    for label, folder in (("coarse", variance / "simulation/assessment"),
                          ("fine", fine / "simulation/assessment"),
                          ("real", fine / "real/assessment"),
                          ("real_adaptive", fine / "real-adaptive/assessment"),
                          ("real_refined", fine / "real-refined/assessment")):
        path = folder / "components.json"
        if not path.is_file():
            continue
        evidence[label + "_components"] = artifact(path)
        component = evidence[label + "_components"]["record"]
        if component["program_sha256"] != programs["variance_fine_component_report.py"]["sha256"]:
            raise ValueError("Executed component program changed")
        if component["artifact_sha256"] != hashlib.sha256((folder / "components.npz").read_bytes()).hexdigest():
            raise ValueError("Component artifact changed")
    for label in ("simulation_collapsed_offset_review", "real_collapsed_offset_review",
                  "real_adaptive_collapsed_offset_review", "real_refined_collapsed_offset_review"):
        corrected = evidence.get("variance_fine_" + label, {}).get("record")
        if corrected is not None:
            for name, sha in corrected["input_sha256"].items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha:
                    raise ValueError("Conditional-offset review input changed")
            for name, sha in corrected["source_sha256"].items():
                data = (rho / "examples/TRIS" / name).read_bytes()
                if hashlib.sha256(data).hexdigest() != sha:
                    raise ValueError("Conditional-offset helper changed")
                programs["variance_fine_" + name] = {"source": data.decode(), "sha256": sha}
    for label in ("tests", "tests-final", "readme-tests", "numerics-tests", "readme-numerics-tests",
                  "collapsed-shift-red", "collapsed-shift-verification",
                  "collapsed-shift-final-verification"):
        path = fine / label / "junit.xml"
        exit_path = fine / label / "exit"
        if path.exists() and exit_path.exists():
            suites = list(ET.parse(path).getroot().iter("testsuite"))
            key = "variance-fine-" + label
            tests[key] = {k: sum(int(s.get(k, 0)) for s in suites)
                          for k in ("tests", "failures", "errors", "skipped")}
            tests[key].update(junit=str(path), exit_code=int(exit_path.read_text()))
    if (fine / "beam-preparation/summary.json").exists():
        evidence["variance_fine_beam_preparation"] = artifact(fine / "beam-preparation/summary.json")
        beam_program = root / "runs/tris-multichain-budget-20260915/prepare_beam512.py"
        data = beam_program.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if evidence["variance_fine_beam_preparation"]["record"]["program_sha256"] != sha:
            raise ValueError("Beam preparation program changed")
        programs["prepare_beam512.py"] = {"source": data.decode(), "sha256": sha}
        evidence["variance_fine_beam_preparation"]["exit_code"] = int(
            (beam_program.parent / "beam-preparation.exit").read_text()
        )
    notebook_path = root / "runs/tris-physical-inference-notebook-20260915/execution.json"
    if notebook_path.exists():
        evidence["physical_notebook"] = artifact(notebook_path)
        source = (root / "examples/inference/tris_physical_notebook.py").read_bytes()
        sha = hashlib.sha256(source).hexdigest()
        if evidence["physical_notebook"]["record"]["source_sha256"] != sha:
            raise ValueError("Notebook must be re-executed after its source changes")
        programs["tris_physical_notebook.py"] = {"source": source.decode(), "sha256": sha}
    if (fine / "plot-geometry/summary.json").is_file():
        evidence["plot_geometry"] = artifact(fine / "plot-geometry/summary.json")
        program = root / "runs/tris-multichain-budget-20260915/prepare_plot_geometry.py"
        data = program.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if evidence["plot_geometry"]["record"]["program_sha256"] != sha:
            raise ValueError("Executed plot geometry program changed")
        programs["prepare_plot_geometry.py"] = {"source": data.decode(), "sha256": sha}
        evidence["plot_geometry"]["exit_code"] = int(
            (program.parent / "plot-geometry.exit").read_text()
        )
    moment_check = notebook_path.parent / "log-field-moment-check.json"
    if moment_check.is_file():
        evidence["plot_log_field_moment_check"] = artifact(moment_check)
    browser = notebook_path.parent / "browser-verification.json"
    if browser.is_file():
        evidence["browser_verification"] = artifact(browser)
    for regression_name in ("notebook-regression", "conditional-offset-regression", "adaptive-regression", "refined-regression", "finalizer-verification"):
        regression = notebook_path.parent / regression_name
        if (regression / "junit.xml").is_file() and (regression / "exit").is_file():
            suites = list(ET.parse(regression / "junit.xml").getroot().iter("testsuite"))
            key = "physical-" + regression_name
            tests[key] = {
                k: sum(int(s.get(k, 0)) for s in suites)
                for k in ("tests", "failures", "errors", "skipped")
            }
            tests[key].update(
                junit=str(regression / "junit.xml"), exit_code=int((regression / "exit").read_text()),
            )
    finalizer_binding = notebook_path.parent / "finalizer-verification/binding.json"
    if finalizer_binding.is_file():
        for name, sha in json.loads(finalizer_binding.read_text()).items():
            data = Path(name).read_bytes()
            if hashlib.sha256(data).hexdigest() != sha:
                raise ValueError("Verified artifact finalizer changed")
            programs[Path(name).name] = {"source": data.decode(), "sha256": sha}
    stopped = evidence.get("variance_fine_real_operator_stop", {}).get("record")
    if stopped:
        prior = fine / "real/summary-before-operator-stop.json"
        if hashlib.sha256(prior.read_bytes()).hexdigest() != stopped["prior_summary_sha256"]:
            raise ValueError("Preserved pre-stop real summary changed")
        checkpoint = fine / "real" / stopped["checkpoint"]["file"]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != stopped["checkpoint"]["sha256"]:
            raise ValueError("Preserved pre-stop real checkpoint changed")
    pipeline = fine / "completion-driver/summary.json"
    if pipeline.is_file():
        evidence["completion_driver"] = artifact(pipeline)
        driver = evidence["completion_driver"]["record"]
        source = (fine / "complete_pipeline.py").read_bytes()
        sha = hashlib.sha256(source).hexdigest()
        if driver["driver_sha256"] != sha:
            raise ValueError("Active completion driver changed")
        programs["complete_pipeline.py"] = {"source": source.decode(), "sha256": sha}
        exit_path = fine / "completion-driver.exit"
        evidence["completion_driver"]["exit_code"] = int(exit_path.read_text()) if exit_path.exists() else None
        if not exit_path.exists():
            active_names.append("completion_driver")
        verification = fine / "completion-wait-verification.json"
        evidence["completion_wait_verification"] = artifact(verification)
        if evidence["completion_wait_verification"]["record"]["program_sha256"] != sha:
            raise ValueError("Completion wait changed since its race tests")
        source = (fine / "verify_completion_wait.py").read_bytes()
        programs["verify_completion_wait.py"] = {
            "source": source.decode(), "sha256": hashlib.sha256(source).hexdigest(),
        }
        previous = fine / "completion-driver-before-exit-race-fix/summary.json"
        if previous.is_file():
            evidence["completion_driver_before_exit_race_fix"] = artifact(previous)
            evidence["completion_driver_before_exit_race_fix"].update(
                exit_code=int((fine / "completion-driver-before-exit-race-fix.exit").read_text()),
                stopped_while_waiting_only=True,
            )
            source = (fine / "complete_pipeline_before_exit_race_fix.py").read_bytes()
            programs["complete_pipeline_before_exit_race_fix.py"] = {
                "source": source.decode(), "sha256": hashlib.sha256(source).hexdigest(),
            }
    for driver_label, driver_program in (
        ("warm_completion_driver", "complete_warm_pipeline.py"),
        ("adaptive_completion_driver", "complete_adaptive_pipeline.py"),
        ("refined_completion_driver", "complete_refined_pipeline.py"),
    ):
        folder = driver_label.replace("_", "-")
        pipeline = fine / folder / "summary.json"
        if not pipeline.is_file():
            continue
        evidence[driver_label] = artifact(pipeline)
        driver = evidence[driver_label]["record"]
        source = (fine / driver_program).read_bytes()
        sha = hashlib.sha256(source).hexdigest()
        if driver["program_sha256"] != sha:
            raise ValueError("Completion driver changed")
        programs[driver_program] = {"source": source.decode(), "sha256": sha}
        for name, expected in driver["programs_sha256"].items():
            if programs["variance_fine_" + name]["sha256"] != expected:
                raise ValueError(f"Frozen pipeline stage changed: {name}")
        exit_path = fine / (folder + ".exit")
        evidence[driver_label]["exit_code"] = int(exit_path.read_text()) if exit_path.exists() else None
        if not exit_path.exists():
            active_names.append(driver_label)
    finalizer = notebook_path.parent / "adaptive-finalization/summary.json"
    if finalizer.is_file():
        evidence["adaptive_artifact_finalization"] = artifact(finalizer)
        manifest_path = finalizer.parent / "snapshot/manifest.json"
        if manifest_path.is_file():
            preserved = json.loads(manifest_path.read_text())
            for item in preserved.values():
                path = Path(item["copy"])
                if not path.is_absolute():
                    path = root / path
                if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                    raise ValueError("Preserved finalization snapshot changed")
            evidence["adaptive_artifact_snapshot"] = artifact(manifest_path)
    refined_finalizer = notebook_path.parent / "finalize_refined_artifacts.py"
    if refined_finalizer.is_file():
        data = refined_finalizer.read_bytes()
        programs[refined_finalizer.name] = {"source": data.decode(), "sha256": hashlib.sha256(data).hexdigest()}
    if active_names and not allow_active:
        raise ValueError("an experiment is incomplete; explicit incomplete snapshot required")
    result = {
        "status": "record", "task": "T-001", "decision": "D22", "date": "2026-09-15",
        "snapshot_at_utc": datetime.now(UTC).isoformat(),
        "active_experiments": active_names,
        "all_experiments_completed": not active_names, "scientific_certification": False,
        "evidence": evidence, "diagnostic_programs": programs, "tests": tests,
        "candidate_hierarchical_model": {
            "status": (
                "real_assessed_conditional_results_pending_scientific_review"
                if any(k in evidence for k in ("variance_fine_real_assessment", "variance_fine_real_adaptive_assessment", "variance_fine_real_refined_assessment")) else
                "real_refined_pilot_or_sampling_incomplete"
                if "variance_fine_real_refined_reference" in evidence else
                "real_stopped_for_quadrature_refinement"
                if "variance_fine_real_adaptive_quadrature_stop" in evidence else
                "real_admitted_sampling_incomplete"
                if "variance_fine_real_pilot" in evidence else
                "experimental_sampling_not_yet_admitted_for_real_data"
            ),
            "source_sha256": hashlib.sha256(
                (rho / "examples/TRIS/physical_modes_hierarchical.py").read_bytes()
            ).hexdigest(),
            "scale_coordinate_source_sha256": hashlib.sha256(
                (rho / "examples/TRIS/physical_modes_scale_coordinate.py").read_bytes()
            ).hexdigest(),
            "variance_gibbs_source_sha256": hashlib.sha256(
                (rho / "examples/TRIS/physical_modes_variance_gibbs.py").read_bytes()
            ).hexdigest(),
        },
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation": (
            "Known stationary short-prefix rejection calibrates a computational rule. "
            "It does not retroactively pass D21 or certify the physical posterior. "
            "Final multichain acceptance thresholds remain unchanged."
        ),
    }
    (root / "examples/inference/data/tris_multichain_budget_20260915.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-active", action="store_true")
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    record(root, root.parent / "rheplicant", allow_active=options.allow_active)
