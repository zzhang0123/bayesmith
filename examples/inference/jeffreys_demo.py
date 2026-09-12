"""Explicit research runner for retired Jeffreys comparisons.

Use an explicit --case and an output outside the notebook catalogue.
Data, seeds, truth and recovery thresholds match the baseline demonstrations.
Other implemented comparisons remain available through an explicit --case.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from functools import partial
from pathlib import Path

import jax

from .__main__ import CASES, positive, save_demo
from .jeffreys_priors import (
    SUPPORTED_CASES,
    UNSUPPORTED_VARIANTS,
    metadata_for,
    with_jeffreys_prior,
)


def run_case(case, seed=0, draws=2000, warmup=1000):
    """Run the existing simulation and inference with a bounded prior factor."""
    if case not in SUPPORTED_CASES:
        raise ValueError(f"{case}: exact marginal Fisher prior has not been constructed.")
    module = importlib.import_module(f"examples.inference.{case}")
    demo = module.run(seed, draws, warmup,
                      graph_transform=partial(with_jeffreys_prior, case))
    demo.view["jeffreys_prior"] = metadata_for(case)
    demo.note = (
        "Jeffreys-prior counterpart on the same simulated data and original finite "
        "root bounds. " + demo.view["jeffreys_prior"]["conditional_law_note"]
        + " Recovery uses the same baseline prior-scale reference and fixed 99% intervals."
    )
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/inference-demo/jeffreys"))
    parser.add_argument("--case", choices=("all", *SUPPORTED_CASES, *UNSUPPORTED_VARIANTS),
                        required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draws", type=positive, default=2000)
    parser.add_argument("--warmup", type=positive, default=1000)
    args = parser.parse_args()
    cases = tuple(c for c in CASES if c in SUPPORTED_CASES) if args.case == "all" else (args.case,)
    passed = True
    with jax.enable_x64(True):
        for case in cases:
            if case not in SUPPORTED_CASES:
                continue
            demo = run_case(case, args.seed, args.draws, args.warmup)
            passed = save_demo(case, demo, args.output, args.seed, args.draws, args.warmup) and passed
            del demo
            jax.clear_caches()
    for case in UNSUPPORTED_VARIANTS:
        if args.case not in ("all", case):
            continue
        destination = args.output / case
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "status.json").write_text(
            json.dumps(metadata_for(case), indent=2, ensure_ascii=False) + "\n"
        )
        print(f"{case}: marginal Jeffreys prior is not implemented; no posterior "
              "samples were generated. See status.json.", file=sys.stderr)
    if args.case in UNSUPPORTED_VARIANTS:
        return 2
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
