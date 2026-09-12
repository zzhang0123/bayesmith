"""Paired repeated-simulation study of demo 06, with fixed root parameters.

    python -m examples.inference.composed_repeated --output runs/composed-repeated

Seeds are registered before inference. Every seed redraws the random process and
observation noise through composed_process.run. Coverage misses are retained;
initialization or sampling failures are recorded and never replaced by new seeds.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from zipfile import BadZipFile

PRIORS = ("uniform", "mild")


def engine_fingerprint():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "src/bayesmith").rglob("*.py"))
    paths += [root / "examples/inference" / name for name in (
        "composed_process.py", "common.py", "priors.py", "__main__.py",
        "composed_repeated.py",
    )]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".pending")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def worker(output, seed, prior, draws, warmup):
    import jax

    from .__main__ import save_demo
    from .composed_process import run

    with jax.enable_x64(True):
        demo = run(seed, draws, warmup, mild_prior=prior == "mild", observations=384)
        passed = save_demo("composed_process", demo, output, seed, draws, warmup)
    return 0 if passed else 1


def execute_one(output, seed, prior, config):
    folder = output / f"seed-{seed:04d}" / prior
    started = time.perf_counter()
    record = {
        "seed": seed, "prior": prior, "exit_code": None, "completed": False,
        "result": str((folder / "composed_process/result.json").relative_to(output)),
    }
    try:
        folder.mkdir(parents=True, exist_ok=False)
        snapshot = output / "engine"
        environment = {**os.environ, "PYTHONPATH": os.pathsep.join((str(snapshot / "src"), str(snapshot)))}
        with (folder / "run.log").open("w") as log:
            process = subprocess.run([
                sys.executable, "-m", "examples.inference.composed_repeated",
                "--worker-seed", str(seed), "--prior", prior, "--output", str(folder),
                "--draws", str(config["draws"]), "--warmup", str(config["warmup"]),
            ], stdout=log, stderr=subprocess.STDOUT, check=False, cwd=snapshot, env=environment)
        record["exit_code"] = process.returncode
        path = output / record["result"]
        if path.exists() and process.returncode in (0, 1):
            import numpy as np

            report = json.loads(path.read_text())
            with np.load(path.with_name("posterior.npz")) as saved:
                if not saved.files or any(not np.isfinite(saved[name]).all() for name in saved.files):
                    raise ValueError("Missing or nonfinite posterior archive")
            if process.returncode != (0 if report["passed"] else 1):
                raise ValueError("Worker exit differs from the saved recovery/diagnostic result")
            provenance = report["provenance"]
            actual = {**provenance["library_source_sha256"],
                      **{f"examples/inference/{name}": value for name, value in provenance["source_sha256"].items()}}
            record["source_matches_registration"] = all(actual.get(k) == v for k, v in config["inference_source_sha256"].items())
            record["diagnostics_passed"] = report["checks"]["chain_diagnostics"]["passed"]
            record["recovery_passed"] = report["checks"]["recovery"]
            record["result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            record["completed"] = True
    except (OSError, EOFError, BadZipFile, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    record["elapsed_seconds"] = time.perf_counter() - started
    if folder.is_dir():
        try:
            write_json(folder / "attempt.json", record)
        except OSError as exc:
            record["attempt_write_error"] = str(exc)
            record["completed"] = False
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/composed-repeated"))
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--prior", choices=PRIORS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.draws < 4 or args.warmup < 4 or args.repeats < 2 or args.start_seed < 0:
        parser.error("Use at least 4 draws/warmup, 2 repeats and a nonnegative start seed")
    if args.worker_seed is not None:
        if args.prior is None:
            parser.error("Worker requires --prior")
        return worker(args.output, args.worker_seed, args.prior, args.draws, args.warmup)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "registered.json").exists():
        parser.error("Output already contains a registered study; use a new directory")
    fingerprint = engine_fingerprint()
    # Freeze the library and demo sources for the batch. Peer edits in the shared
    # checkout must not change the inference engine half-way through this study.
    root = Path(__file__).resolve().parents[2]
    snapshot = args.output / "engine"
    sources = sorted((root / "src/bayesmith").rglob("*.py"))
    sources += sorted((root / "examples/inference").glob("*.py"))
    for source in sources:
        target = snapshot / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for relative, expected in fingerprint.items():
        if hashlib.sha256((snapshot / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Source changed while creating the frozen study engine")
    config = {
        "kind": "fixed_root_repeated_simulation", "registered_at": datetime.now(UTC).isoformat(),
        "seeds": list(range(args.start_seed, args.start_seed + args.repeats)),
        "priors": list(PRIORS), "draws": args.draws, "warmup": args.warmup,
        "chains": 2, "workers": args.workers, "interval_masses": [0.68, 0.95, 0.99],
        "fixed_roots": {"power_amplitude": 0.5, "nonlinear": [0.45, 0.09],
                        "background": [3.0, 0.2], "gain": [0.08, -0.05], "sigma_w": 0.015},
        "redrawn": ["instance", "observation_noise"],
        "mild_prior": {"family": "TruncatedNormal", "loc": 0.02, "scale": 0.01,
                       "low": 0.003, "high": 0.06},
        "estimator": "posterior mean", "primary_parameter": "sigma_w",
        "policy": "All registered attempts remain visible. No seed replacement, prior tuning, truth-based initialization or stopping on coverage. Primary summaries include every completed run; diagnostic-passing subsets are labeled separately. This is not prior-predictive SBC.",
        "inference_source_sha256": fingerprint,
    }
    write_json(args.output / "registered.json", config)
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute_one, args.output, seed, prior, config)
                   for seed in config["seeds"] for prior in PRIORS]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            write_json(args.output / "progress.json", sorted(records, key=lambda r: (r["seed"], r["prior"])))
            print(f"{len(records)}/{len(futures)} seed={record['seed']} {record['prior']}: "
                  f"completed={record['completed']}, diagnostics={record.get('diagnostics_passed')}, "
                  f"recovery={record.get('recovery_passed')}, {record['elapsed_seconds']:.1f}s", flush=True)
    unchanged = all(hashlib.sha256((snapshot / relative).read_bytes()).hexdigest() == value
                    for relative, value in fingerprint.items())
    matching = all(r.get("source_matches_registration", False) for r in records)
    write_json(args.output / "completion.json", {
        "source_unchanged": unchanged, "attempts": len(records),
        "completed": sum(r["completed"] for r in records),
        "all_runs_match_registered_sources": matching,
        "note": "A completed experiment need not cover truth in every dataset.",
    })
    return 0 if unchanged and matching and all(r["completed"] for r in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
