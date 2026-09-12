"""Fit a real TRIS map dataset with bayesmith, without simulation/recovery gates.

python -m examples.inference.tris_case --input runs/tris-input \
    --output runs/inference-demo-verified/tris_haslam
"""

from __future__ import annotations

import argparse
import dataclasses
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
from .tris_products import posterior_sky_products
from .tris_sky import REGIONS, model, model_inputs, sky_draws, zero_levels


def summarize(values):
    values = np.asarray(values)
    return {
        "mean": values.mean(axis=0).tolist(),
        "sd": values.std(axis=0, ddof=1).tolist(),
        "lower": np.quantile(values, 0.025, axis=0).tolist(),
        "median": np.median(values, axis=0).tolist(),
        "upper": np.quantile(values, 0.975, axis=0).tolist(),
    }


def data_checks(bundle, inputs):
    """Measure actual likelihood differences; no ring observations enter sampling."""
    from .tris_sky import map_mean, spectral_scaling

    design, cmb, offset, channel, frequencies, data = inputs
    trials = [
        ([1.0, 1.0, 1.0], [-2.8, -2.8, -2.8], [0.0, 0.0]),
        ([1.2, 0.9, 1.1], [-2.6, -2.9, -3.1], [0.2, -0.3]),
        ([0.8, 1.3, 1.0], [-3.2, -2.5, -2.7], [-0.4, 0.5]),
    ]
    chi_map, chi_ring = [], []
    for amp, beta, z in trials:
        amp, beta, z = map(np.asarray, (amp, beta, z))
        sky = sky_draws(bundle, amp[None], beta[None])[0]
        levels = np.asarray(zero_levels(z))
        mean = map_mean(
            spectral_scaling(amp, beta, frequencies),
            levels,
            design,
            cmb,
            offset,
            channel,
        )
        chi_map.append(float(np.sum((np.asarray(data - mean)) ** 2)))
        chi_ring.append(
            sum(
                float(
                    np.sum(
                        (
                            (
                                bundle[f"data_k_{i}"]
                                - bundle[f"operator_{i}"] @ sky[i]
                                + levels[i]
                            )
                            / bundle[f"sigma_k_{i}"]
                        )
                        ** 2
                    )
                )
                for i in range(2)
            )
        )
    delta_error = np.max(np.abs(np.diff(chi_map) - np.diff(chi_ring)))
    scale = max(1.0, np.max(np.abs(np.diff(chi_ring))))
    return {
        "map_chi_square": chi_map,
        "ring_chi_square": chi_ring,
        "max_delta_error": float(delta_error),
        "relative_delta_error": float(delta_error / scale),
        "passed": bool(delta_error / scale < 1e-7),
    }


def run(input_directory, output, *, seed=0, draws=2000, warmup=1500):
    if not jax.config.jax_enable_x64:
        raise ValueError("TRIS inference requires jax.enable_x64(True)")
    manifest = json.loads((input_directory / "manifest.json").read_text())
    if manifest["schema"] != "bayesmith.tris.maps.v1":
        raise ValueError("unsupported TRIS map schema")
    if sha256(input_directory / "maps.npz") != manifest["input_sha256"]:
        raise ValueError("TRIS map hash differs from its provenance manifest")
    with np.load(input_directory / "maps.npz", allow_pickle=False) as archive:
        bundle = {name: archive[name] for name in archive.files}
    if "reference_cmb_k" not in bundle:
        raise ValueError("Re-run tris_prepare to record reference_cmb_k for the Haslam sky products")
    inputs = model_inputs(bundle)
    equivalence = data_checks(bundle, inputs)
    if not equivalence["passed"]:
        raise ValueError(
            f"map likelihood did not preserve the ring likelihood: {equivalence}"
        )
    graph = trace(model, *inputs)
    task = PosteriorTask(
        meta=new_task_meta(label="TRIS real-data Haslam spectral inference"),
        budget=ComputeBudget(draws=draws, warmup=warmup, chains=2),
        chain_method="sequential",
        nuts_on_collapse=False,
        stopping=StoppingPolicy(rhat_max=1.01, ess_min=400),
        backend_options=(("progress_bar", False),),
    )
    key_plan, key_sample = jax.random.split(jax.random.key(seed))
    plan = compile_task(
        graph,
        task,
        model_ref=model_ref_from_callable(model, identifier=model.__module__),
        key=key_plan,
    )
    if isinstance(plan, Refusal):
        raise RuntimeError(f"TRIS compilation refused: {plan}")  # noqa: TRY004
    print(plan.runtime_plan, flush=True)
    posterior = require_result(execute_task(plan, key=key_sample), PosteriorResult)
    if not isinstance(posterior.representation, DrawsPosterior):
        raise TypeError("TRIS case requires unweighted posterior draws")
    posterior_samples = samples(posterior)
    diagnostics = diagnostic_checks(posterior)
    levels = np.asarray(jax.vmap(zero_levels)(posterior_samples["zero_standard"]))
    skies = sky_draws(bundle, posterior_samples["amplitude"], posterior_samples["beta"])
    parameter_rows = []
    for name in ("amplitude", "beta", "zero_level_K"):
        values = levels if name == "zero_level_K" else posterior_samples[name]
        for j in range(values.shape[1]):
            parameter_rows.append(
                dict(
                    name=f"{name}[{j}]",
                    parameter=name,
                    index=j,
                    region=REGIONS[j]
                    if name != "zero_level_K"
                    else f"{bundle['frequency_mhz'][j]} MHz",
                    **summarize(values[:, j]),
                )
            )
    # Use the same posterior draws for a mean-function band and a noisy
    # replicated-observation band. PPC discrepancy is evaluated at EACH draw.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 731]))
    summaries = []
    predictions = posterior_sky_products(
        bundle, posterior_samples["amplitude"], posterior_samples["beta"]
    )
    for i in range(2):
        a, sigma, observed = (
            bundle[f"{key}_{i}"] for key in ("operator", "sigma_k", "data_k")
        )
        mean = skies[:, i] @ a.T - levels[:, i, None]
        replicated = mean + rng.normal(size=mean.shape) * sigma
        residual = observed - mean.mean(axis=0)
        d_obs = np.sum(((observed - mean) / sigma) ** 2, axis=1)
        d_rep = np.sum(((replicated - mean) / sigma) ** 2, axis=1)
        ppc_tail = float(np.mean(d_rep >= d_obs))
        transferred = mean @ bundle[f"weights_{i}"].T + bundle[f"bias_{i}"]
        map_summary = summarize(transferred)
        summary = {
            "frequency_mhz": float(bundle["frequency_mhz"][i]),
            "ra_deg": bundle[f"ra_deg_{i}"].tolist(),
            "data_k": observed.tolist(),
            "sigma_k": sigma.tolist(),
            "mean_function": summarize(mean),
            "predictive": summarize(replicated),
            "residual_k": residual.tolist(),
            "standardized_residual": (residual / sigma).tolist(),
            "chi_square_at_posterior_mean": float(np.sum((residual / sigma) ** 2)),
            "chi_square_per_observation": float(np.mean((residual / sigma) ** 2)),
            "residual_rms_k": float(np.sqrt(np.mean(residual**2))),
            "ppc_tail_probability": ppc_tail,
            "ppc_replicates": len(mean),
            "adequacy": "mismatch"
            if ppc_tail < 0.01 or ppc_tail > 0.99
            else "not_flagged",
        }
        summaries.append(summary)
        for key, value in map_summary.items():
            predictions[f"map_{key}_{i}"] = value
        predictions[f"sky_mean_{i}"] = skies[:, i].mean(axis=0)
        predictions[f"sky_sd_{i}"] = skies[:, i].std(axis=0, ddof=1)
    dag = graph_rows(graph)
    report = {
        "case": "tris_haslam",
        "kind": "real_observations",
        "title": "TRIS × Haslam: spectral sky inference",
        "seed": seed,
        "warmup": warmup,
        "requested_draws_per_chain": draws,
        "chain_shape": posterior.representation.chain_shape,
        "method": posterior.representation.method,
        "passed": diagnostics["passed"],
        "checks": {"chain_diagnostics": diagnostics, "map_likelihood": equivalence},
        "model_adequacy": "mismatch"
        if any(s["adequacy"] == "mismatch" for s in summaries)
        else "not_flagged",
        "parameters": parameter_rows,
        "interval_mass": 0.95,
        "frequencies": summaries,
        "priors": {
            "amplitude": "independent Uniform(0.2, 3.0), three regions",
            "beta": "independent Uniform(-4.0, -1.5), three regions",
            "zero_standard": "independent Normal(0, 1), two shared offsets",
            "offset_K": "correction to measured sky: 600: 0.066 z; 820: 0.300 z if z<0, otherwise 0.430 z; subtracted in predicted data",
        },
        "provenance": provenance(),
        "model_source": Path(__file__).with_name("tris_sky.py").read_text(),
        "data_manifest": manifest,
        "sky_products": {
            "reference_frequency_mhz": 408.0,
            "reference_cmb_k": float(bundle["reference_cmb_k"]),
            "draws": len(posterior_samples["amplitude"]),
            "grid": "same HEALPix RING equatorial grid as maps.npz",
            "definition": "a(p)=a[region]; beta(p)=beta[region]; recalibrated Haslam=a(p)*template_k+reference_cmb_k",
            "uncertainty": "sample SD and 95% equal-tail intervals, conditional on the fixed template and three-region model; not independent pixel measurements",
        },
        "blocking": inspect_blocks(graph, plan.runtime_plan),
        "dag": dag,
        "preflight": [
            {
                "code": f.code,
                "conclusion": f.conclusion,
                "scope": {"kind": f.scope.kind.value, "name": f.scope.name},
                "measurements": dict(f.measurements),
                "grounds": list(f.grounds),
            }
            for f in plan.analysis.findings
        ],
        "execution": {
            "termination": posterior.run.termination.reason.value,
            "sampling": dict(posterior.run.sampling_details),
            "initial_values": {
                a.name: a.value.tolist() for a in posterior.run.initial_values
            },
            "budget": dataclasses.asdict(task.budget),
            "diagnostic_policy": dataclasses.asdict(task.diagnostics),
            "stopping_policy": dataclasses.asdict(task.stopping),
            "wall_clock_seconds": posterior.run.timing.wall_clock_seconds,
        },
        "note": "Real observations; no truth, recovery criterion or simulated input. Convergence does not imply an adequate sky model.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(finite_json(report), indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(
        output / "posterior.npz", **posterior_samples, zero_level_K=levels
    )
    np.savez_compressed(output / "predictions.npz", **predictions)
    shutil.copy2(input_directory / "maps.npz", output / "maps.npz")
    shutil.copy2(input_directory / "manifest.json", output / "manifest.json")
    shutil.copytree(input_directory / "archive", output / "archive", dirs_exist_ok=True)
    (output / "dag.mmd").write_text(mermaid(dag))
    (output / "plan.txt").write_text(str(plan.runtime_plan) + "\n")
    for name, artifact in [
        ("posterior", posterior),
        ("analysis", plan.analysis),
        ("task", task),
    ]:
        dump_artifact(artifact, output / f"{name}.artifact.json")
    (output / "README.md").write_text(
        "# TRIS × Haslam real-data inference\n\n"
        + f"Sampling diagnostics: {diagnostics['passed']}. Model adequacy: {report['model_adequacy']}.\n\n"
        + "See result.json for the measured findings; manifest.json identifies the real observations.\n"
        + "maps.npz contains the full map covariance and response, not just independent pixel errors.\n"
    )
    print(
        json.dumps(
            {
                "diagnostics": diagnostics,
                "parameters": parameter_rows,
                "model_adequacy": report["model_adequacy"],
                "chi_square_per_observation": [
                    s["chi_square_per_observation"] for s in summaries
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("runs/tris-input"))
    parser.add_argument(
        "--output", type=Path, default=Path("runs/inference-demo-verified/tris_haslam")
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draws", type=positive, default=2000)
    parser.add_argument("--warmup", type=positive, default=1500)
    args = parser.parse_args()
    with jax.enable_x64(True):
        result = run(
            args.input,
            args.output,
            seed=args.seed,
            draws=args.draws,
            warmup=args.warmup,
        )
    # A model mismatch is a scientific finding, not a sampler execution failure.
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
