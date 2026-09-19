"""Run common-data TRIS × Haslam analyses with and without an RSB term."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

import jax
import numpy as np

from bayesmith import compile_task, execute_task, trace
from bayesmith.artifacts import (
    ComputeBudget,
    DrawsPosterior,
    PosteriorResult,
    PosteriorTask,
    Refusal,
    StoppingPolicy,
    dump_artifact,
    model_ref_from_callable,
    new_task_meta,
)

from .__main__ import finite_json, positive, provenance
from .block_inspection import inspect_blocks
from .common import diagnostic_checks, graph_rows, mermaid, require_result, samples
from .tris_prepare import sha256
from .tris_rsb_diagnostics import posterior_checks
from .tris_rsb_external import load_external
from .tris_rsb_initialization import data_initialization
from .tris_rsb_sky import model, model_inputs


@dataclasses.dataclass(frozen=True)
class CommonInputs:
    tris_directory: Path
    external_directory: Path
    tris_manifest: dict
    external_manifest: dict
    tris_bundle: dict
    external: dict


def _canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def load_common_inputs(tris_directory: Path, external_directory: Path) -> CommonInputs:
    tris_directory, external_directory = map(Path, (tris_directory, external_directory))
    tris_manifest = json.loads((tris_directory / "manifest.json").read_text())
    external, external_manifest = load_external(external_directory)
    if tris_manifest.get("schema") != "bayesmith.tris.maps.v1":
        raise ValueError("unsupported TRIS map schema")
    for directory, filename, manifest in (
        (tris_directory, "maps.npz", tris_manifest),
        (external_directory, "external.npz", external_manifest),
    ):
        if sha256(directory / filename) != manifest.get("input_sha256"):
            raise ValueError(f"{filename} hash differs from its provenance manifest")
    tris_bundle = _load_npz(tris_directory / "maps.npz")
    return CommonInputs(tris_directory, external_directory, tris_manifest, external_manifest, tris_bundle, external)


def build_common_manifest(common: CommonInputs) -> dict:
    content = {
        "schema": "bayesmith.tris.rsb.common-input.v1",
        "tris_maps_sha256": common.tris_manifest["input_sha256"],
        "external_input_sha256": common.external_manifest["input_sha256"],
        "external_rows": len(common.external["frequency_mhz"]),
        "external_surveys": list(dict.fromkeys(str(name) for name in common.external["survey"])),
        "frequency_mhz": np.asarray(common.external["frequency_mhz"]).tolist(),
        "calibration_tau_rj_k": np.asarray(common.external.get("tau_rj_k", [])).tolist(),
        "external_schema": common.external_manifest["schema"],
        "covariance_model": ("inclusive published covariance; no added calibration latent"
                             if "covariance_rj_k2" in common.external else "legacy survey rank-one approximation"),
        "frozen_reference_case": "tris_haslam",
    }
    return {**content, "common_input_sha256": _canonical_digest(content)}


def summarize(values):
    values = np.asarray(values)
    summary = {
        "mean": values.mean(axis=0), "sd": values.std(axis=0, ddof=1),
        "lower": np.quantile(values, .025, axis=0), "median": np.median(values, axis=0),
        "upper": np.quantile(values, .975, axis=0),
    }
    return {name: float(value) if np.ndim(value) == 0 else value.tolist() for name, value in summary.items()}


def _parameter_rows(draws):
    rows = []
    for name, values in draws.items():
        values = np.asarray(values)
        for index in np.ndindex(values.shape[1:]):
            part = values[(slice(None),) + index]
            rows.append({"name": name + (str(list(index)) if index else ""), "parameter": name, "index": list(index), **summarize(part)})
    return rows


def run_variant(common, *, variant, included_surveys, output, seed, draws, warmup, chains=4, target_accept=.95):
    if not jax.config.jax_enable_x64:
        raise ValueError("TRIS RSB inference requires jax.enable_x64(True)")
    include_rsb = variant == "rsb"
    if variant not in {"no_rsb", "rsb"}:
        raise ValueError("variant must be no_rsb or rsb")
    inputs = model_inputs(common.tris_bundle, common.external, included_surveys, include_rsb)
    graph = trace(model, *inputs)
    initialization, initialization_record = data_initialization(graph, inputs, chains=chains, seed=seed)
    kernel_options = (("dense_mass", True), ("target_accept_prob", target_accept))
    task = PosteriorTask(
        meta=new_task_meta(label=f"TRIS + Haslam {'RSB' if include_rsb else 'no-RSB'} real-data inference"),
        budget=ComputeBudget(draws=draws, warmup=warmup, chains=chains), chain_method="sequential",
        initialization=initialization,
        nuts_on_collapse=False, stopping=StoppingPolicy(rhat_max=1.01, ess_min=400),
        backend_options=(("progress_bar", False), ("nuts_options", kernel_options)),
    )
    plan_key, sample_key = jax.random.split(jax.random.key(seed))
    plan = compile_task(graph, task, model_ref=model_ref_from_callable(model, identifier=model.__module__), key=plan_key)
    if isinstance(plan, Refusal):
        raise RuntimeError(f"TRIS RSB compilation refused: {plan}")  # noqa: TRY004
    posterior = require_result(execute_task(plan, key=sample_key), PosteriorResult)
    if not isinstance(posterior.representation, DrawsPosterior):
        raise TypeError("TRIS RSB case requires unweighted posterior draws")
    posterior_samples = samples(posterior)
    diagnostics = diagnostic_checks(posterior)
    adequacy, predictions = posterior_checks(common.tris_bundle, posterior_samples, seed=seed)
    if not adequacy["map_likelihood"]["passed"]:
        raise ValueError("joint map likelihood does not preserve ring likelihood differences")
    manifest = build_common_manifest(common)
    report = {
        "case": f"tris_haslam_{variant}", "kind": "real_observations", "title": f"TRIS + Haslam {'+ RSB' if include_rsb else 'without RSB'}",
        "variant": variant, "included_surveys": list(included_surveys), "seed": seed, "warmup": warmup,
        "requested_draws_per_chain": draws, "chain_shape": posterior.representation.chain_shape,
        "method": posterior.representation.method, "passed": diagnostics["passed"],
        "latent_names": list(graph.latents), "parameters": _parameter_rows(posterior_samples),
        "checks": {"chain_diagnostics": diagnostics, "map_likelihood": adequacy.pop("map_likelihood")}, "data_manifest": manifest,
        **adequacy,
        "sky_products": {"definition": "a(region)*(H-CMB408+z_H-B408)+CMB408+B408", "uncertainty": "conditional regional extrapolation; does not include model discrepancy"},
        "priors": {"amplitude": "three independent Uniform(0.2, 3)", "beta": "three independent Uniform(-4, -1.5)", "zero_standard": "two independent Normal(0,1); corrections 0.066 z and (0.300 if z<0 else 0.430) z K", **({"rsb_amplitude": "Uniform(0, 5) K at 1 GHz", "rsb_beta": "Uniform(-4, -1.5)"} if include_rsb else {}), "haslam_monopole_K": "Normal(0, 3 K)", "calibration_standard": "Normal(0, 1); tau is survey-wide; assumed covariance of published summaries"},
        "provenance": provenance(), "model_source": Path(__file__).with_name("tris_rsb_sky.py").read_text(),
        "blocking": inspect_blocks(graph, plan.runtime_plan), "dag": graph_rows(graph),
        "preflight": [{"code": finding.code, "conclusion": finding.conclusion, "scope": {"kind": finding.scope.kind.value, "name": finding.scope.name}, "measurements": dict(finding.measurements), "grounds": list(finding.grounds)} for finding in plan.analysis.findings],
        "execution": {"termination": posterior.run.termination.reason.value, "sampling": dict(posterior.run.sampling_details), "initial_values": {item.name: item.value.tolist() for item in posterior.run.initial_values}, "budget": dataclasses.asdict(task.budget), "stopping_policy": dataclasses.asdict(task.stopping), "wall_clock_seconds": posterior.run.timing.wall_clock_seconds},
        "note": "Real observations. Convergence, TRIS residual adequacy, and cross-survey predictive comparison are distinct findings.",
    }
    output = Path(output)
    report["execution"].update(initialization=initialization_record, nuts_options=dict(kernel_options))
    if "covariance_rj_k2" in common.external:
        report["priors"].pop("calibration_standard", None)
        report["external_covariance"] = "published inclusive covariance; calibration marginalized already"
        report["note"] += " External foreground errors shared with Haslam are not jointly propagated; conditional diagnostic only."
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(finite_json(report), indent=2) + "\n")
    np.savez_compressed(output / "posterior.npz", **posterior_samples)
    np.savez_compressed(output / "predictions.npz", **predictions)
    shutil.copy2(common.tris_directory / "maps.npz", output / "maps.npz")
    shutil.copy2(common.external_directory / "external.npz", output / "external.npz")
    shutil.copy2(common.tris_directory / "manifest.json", output / "manifest.json")
    shutil.copy2(common.external_directory / "external_manifest.json", output / "external_manifest.json")
    (output / "shared_input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "dag.mmd").write_text(mermaid(report["dag"]))
    (output / "plan.txt").write_text(str(plan.runtime_plan) + "\n")
    for name, artifact in (("posterior", posterior), ("analysis", plan.analysis), ("task", task)):
        dump_artifact(artifact, output / f"{name}.artifact.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tris-input", type=Path, required=True)
    parser.add_argument("--external-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/inference-demo-verified"))
    parser.add_argument("--variant", choices=("no_rsb", "rsb", "both"), default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draws", type=positive, default=2000)
    parser.add_argument("--warmup", type=positive, default=1500)
    parser.add_argument("--chains", type=positive, default=4)
    parser.add_argument("--target-accept", type=float, default=.95)
    parser.add_argument("--train-survey", choices=("LWA", "ARCADE"), help="run only the specified directional refit")
    parser.add_argument("--heldout", action="store_true", help="also refit each variant with one external survey omitted")
    args = parser.parse_args()
    common = load_common_inputs(args.tris_input, args.external_input)
    surveys = tuple(dict.fromkeys(common.external["survey"].tolist()))
    if args.heldout and len(surveys) < 2:
        parser.error("survey holdout requires at least two actual surveys")
    variants = ("no_rsb", "rsb") if args.variant == "both" else (args.variant,)
    with jax.enable_x64(True):
        for index, variant in enumerate(variants):
            if args.train_survey:
                report = run_variant(common, variant=variant, included_surveys=(args.train_survey,), output=args.output / "heldout" / f"{variant}_train_{args.train_survey}", seed=args.seed + index, draws=args.draws, warmup=args.warmup, chains=args.chains, target_accept=args.target_accept)
                print(variant, args.train_survey, report["checks"]["chain_diagnostics"], flush=True)
                continue
            report = run_variant(common, variant=variant, included_surveys=surveys, output=args.output / f"tris_haslam_{variant}", seed=args.seed + index, draws=args.draws, warmup=args.warmup, chains=args.chains, target_accept=args.target_accept)
            print(variant, report["checks"]["chain_diagnostics"], flush=True)
            if args.heldout:
                for fold, survey in enumerate(surveys):
                    report = run_variant(common, variant=variant, included_surveys=(survey,), output=args.output / "heldout" / f"{variant}_train_{survey}", seed=args.seed + 10 + 2 * index + fold, draws=args.draws, warmup=args.warmup, chains=args.chains, target_accept=args.target_accept)
                    print(variant, survey, report["checks"]["chain_diagnostics"], flush=True)


if __name__ == "__main__":
    main()
