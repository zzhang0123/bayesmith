"""Run operator -> DAG -> simulation -> inference -> recovery checks.

    python -m examples.inference --case all --output runs/inference-demo

Default seed and budgets are registered demo settings, not selected by retries.
A recovery or diagnostic failure saves its report and exits 1.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import jax
import numpy as np

from bayesmith.artifacts import dump_artifact

from .block_inspection import inspect_blocks
from .common import diagnostic_checks, graph_rows, mermaid, recovery_checks, samples

CASES = (
    "linear_gaussian",
    "power_law",
    "hierarchical",
    "bernoulli",
    "multiplicative_noise",
    "composed_process",
)


def positive(value):
    parsed = int(value)
    if parsed < 4:
        raise argparse.ArgumentTypeError("Use at least four draws/warmup steps.")
    return parsed


def finite_json(value):
    """Unavailable numeric diagnostics stay null, with their failure preserved."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def provenance():
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        # HEAD alone cannot identify uncommitted library fixes used by a run.
        "library_source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root / "src" / "bayesmith").rglob("*.py"))
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("jax", "numpy", "numpyro")
        },
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }


def save_demo(name, demo, output, seed, draws, warmup):
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    posterior_samples = samples(demo.posterior)
    recovery = recovery_checks(posterior_samples, demo.truths, demo.prior_sds)
    diagnostics = diagnostic_checks(demo.posterior)
    checks = {"recovery": recovery["passed"], "chain_diagnostics": diagnostics}
    if name == "linear_gaussian":
        from .linear_gaussian import analytic_check

        checks["analytic_oracle"] = analytic_check(
            demo.x,
            demo.data,
            posterior_samples,
            demo.posterior.representation.chain_shape,
        )
    elif name == "power_law":
        from .power_law_reference import sampling_check

        checks["quadrature_oracle"] = sampling_check(
            posterior_samples, demo.posterior.representation.chain_shape,
            demo.view["posterior_reference"],
        )
        checks["repeated_reference"] = demo.view["bias_experiment"]["quadrature"]
    passed = recovery["passed"] and diagnostics["passed"]
    if "analytic_oracle" in checks:
        passed = passed and checks["analytic_oracle"]["passed"]
    if "quadrature_oracle" in checks:
        passed = passed and checks["quadrature_oracle"]["passed"] and checks["repeated_reference"]["passed"]
    dag = graph_rows(demo.graph)
    signal = np.asarray(demo.signal_draws)
    lo, med, hi = np.quantile(signal, [0.005, 0.5, 0.995], axis=0)
    report = {
        "case": name,
        "title": demo.title,
        "passed": bool(passed),
        "seed": seed,
        "requested_draws_per_chain": draws,
        "warmup": warmup,
        "chain_shape": demo.posterior.representation.chain_shape,
        "method": "+".join(
            dict.fromkeys(block.method for block in demo.plan.runtime_plan.blocks)
        ),
        "posterior_representation_method": demo.posterior.representation.method,
        "backend": demo.posterior.run.backend.name,
        "simulation_method": demo.simulation_method,
        "note": demo.note,
        "variant": demo.view.get("data_variant", demo.view.get("prior_variant", demo.view.get("jeffreys_prior"))),
        "proposal_selection": getattr(demo.plan.runtime_plan, "selection", None),
        "proposal_policies": [dict(p.as_options()) for p in (
            tuple(b.policy for b in demo.plan.runtime_plan.compiled.blocks)
            if hasattr(demo.plan.runtime_plan, "compiled") else demo.plan.task.proposals
        )],
        "interval_mass": recovery["interval_mass"],
        "max_sd_ratio": recovery["max_sd_ratio"],
        "parameters": recovery["parameters"],
        "checks": checks,
        "blocking": inspect_blocks(demo.graph, demo.plan.runtime_plan),
        "blocking_probes": demo.blocking_probes,
        "preflight": [
            {
                "code": f.code,
                "conclusion": f.conclusion,
                "scope": {"kind": f.scope.kind.value, "name": f.scope.name},
                "measurements": dict(f.measurements),
                "grounds": list(f.grounds),
            }
            for f in demo.plan.analysis.findings
        ],
        "execution": {
            "termination": demo.posterior.run.termination.reason.value,
            "message": demo.posterior.run.termination.message,
            "initial_values": {
                a.name: a.value.tolist() for a in demo.posterior.run.initial_values
            },
            "sampling": dict(demo.posterior.run.sampling_details),
            "budget": dataclasses.asdict(demo.plan.task.budget),
            "diagnostic_policy": dataclasses.asdict(demo.plan.task.diagnostics),
            "stopping_policy": dataclasses.asdict(demo.plan.task.stopping),
            "wall_clock_seconds": demo.posterior.run.timing.wall_clock_seconds,
        },
        "dag": dag,
        "provenance": provenance(),
        "signal": {
            "view": demo.view,
            "x": np.asarray(demo.x).tolist(),
            "data": np.asarray(demo.data).tolist(),
            "truth": np.asarray(demo.true_signal).tolist(),
            "lower": lo.tolist(),
            "median": med.tolist(),
            "upper": hi.tolist(),
            "meaning": "99% posterior interval for the conditional mean/probability; excludes new observation noise",
        },
    }
    (directory / "result.json").write_text(
        json.dumps(finite_json(report), indent=2, allow_nan=False) + "\n"
    )
    (directory / "dag.mmd").write_text(mermaid(dag))
    (directory / "plan.txt").write_text(str(demo.plan.runtime_plan) + "\n")
    np.savez_compressed(directory / "posterior.npz", **posterior_samples)
    dump_artifact(demo.posterior, directory / "posterior.artifact.json")
    dump_artifact(demo.plan.analysis, directory / "analysis.artifact.json")
    dump_artifact(demo.plan.task, directory / "task.artifact.json")
    if demo.simulation is not None:
        dump_artifact(demo.simulation, directory / "simulation.artifact.json")
    lines = [
        f"# {demo.title}",
        "",
        f"**{'PASS' if passed else 'FAIL'}** · seed {seed} · {report['method']}",
        "",
        demo.note,
        "",
        "```mermaid",
        mermaid(dag).rstrip(),
        "```",
        "",
        "| Parameter | Generating value | Posterior mean | SD | 99% interval | Covered | SD / prior SD |",
        "|---|---:|---:|---:|---|---|---:|",
    ]
    for row in recovery["parameters"]:
        lines.append(
            f"| {row['name']} | {row['truth']:.4f} | {row['mean']:.4f} | {row['posterior_sd']:.4f} "
            f"| [{row['lower']:.4f}, {row['upper']:.4f}] | {row['covered']} | {row['sd_ratio']:.3f} |"
        )
    lines += [
        "",
        "Checks and run provenance: `result.json`. Full draws: `posterior.npz`.",
        "",
        "This fixed-truth recovery run is not an SBC coverage study.",
    ]
    (directory / "README.md").write_text("\n".join(lines) + "\n")
    print(
        f"\n{demo.title}: {'PASS' if passed else 'FAIL'} ({report['method']})",
        flush=True,
    )
    print(demo.plan.runtime_plan, flush=True)
    for row in recovery["parameters"]:
        print(
            f"  {row['name']:16s} truth={row['truth']: .4f}  mean={row['mean']: .4f}  "
            f"99%=[{row['lower']: .4f}, {row['upper']: .4f}]  "
            f"covered={row['covered']}  informative={row['informative']}",
            flush=True,
        )
    print(f"  Reports: {directory}\n  Checks: {finite_json(checks)}", flush=True)
    return bool(passed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", *CASES, "exponential_decay"), default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draws", type=positive, default=2000)
    parser.add_argument("--warmup", type=positive, default=1000)
    parser.add_argument("--output", type=Path, default=Path("runs/inference-demo"))
    args = parser.parse_args()
    print(
        "Criteria: every generating value inside its marginal 99% interval; "
        "posterior SD < 0.5 prior SD; bayesmith chain diagnostics PASS. "
        "No retries or adaptive tolerances.",
        flush=True,
    )
    selected = CASES if args.case == "all" else (args.case,)
    passed = True
    with jax.enable_x64(True):
        for name in selected:
            print(f"Running {name} ...", flush=True)
            module = importlib.import_module(f"examples.inference.{name}")
            demo = module.run(args.seed, args.draws, args.warmup)
            accepted = save_demo(
                name, demo, args.output, args.seed, args.draws, args.warmup
            )
            passed = passed and accepted
            del demo
            jax.clear_caches()
    print(f"OVERALL: {'PASS' if passed else 'FAIL'}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
