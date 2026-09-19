"""Sequential, receipted full-data and four buffered-RA continuous-field fits."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1500)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--target-accept", type=float, default=0.995)
    parser.add_argument("--min-amplitude", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--stage", choices=("full", "heldout"), required=True)
    parser.add_argument(
        "--noise", choices=("floor", "correlated"), default="correlated"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1"
    )
    folds = [None] if args.stage == "full" else list(range(4))
    for fold in folds:
        for ell in range(args.min_amplitude, 3):
            label = (
                "full" if fold is None else f"fold{fold}"
            ) + f"-a{ell}-{args.noise}"
            directory = args.output / label
            receipt = args.output / f"{label}.receipt.json"
            # Existing runs are immutable. A failed diagnostic gets a new directory,
            # never silently replaced by a retry or relabelled as a valid result.
            if directory.exists() or receipt.exists():
                raise ValueError(
                    f"existing run {directory}; select a fresh output root"
                )
            command = [
                sys.executable,
                "-m",
                "examples.TRIS.continuous_background",
                "--response",
                str(args.response),
                "--output",
                str(directory),
                "--l-amplitude",
                str(ell),
                "--noise",
                args.noise,
                "--draws",
                str(args.draws),
                "--warmup",
                str(args.warmup),
                "--target-accept",
                str(args.target_accept),
                "--seed",
                str(19140 + ell + 10 * (0 if fold is None else fold + 1)),
            ]
            if args.progress:
                command.append("--progress")
            if fold is not None:
                command.extend(["--fold", str(fold)])
            started = time.time()
            print(json.dumps({"starting": label}), flush=True)
            with (args.output / f"{label}.log").open("w") as log:
                result = subprocess.run(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            record = {
                "command": command,
                "exit_code": result.returncode,
                "seconds": time.time() - started,
            }
            if (directory / "result.json").exists():
                fit = json.loads((directory / "result.json").read_text())
                record.update(
                    sampling_pass=fit["sampling_pass"],
                    failures=fit["diagnostic_failures"],
                )
            receipt.write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps({"finished": label, **record}), flush=True)
            if result.returncode:
                raise RuntimeError(
                    f"{label} failed with exit {result.returncode}; inspect its log"
                )


if __name__ == "__main__":
    main()
