"""One demo 06 dataset with 3,840 observations and the original Uniform priors."""

import argparse
from pathlib import Path

import jax

from .__main__ import positive, save_demo
from .composed_process import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/inference-demo"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draws", type=positive, default=2000)
    parser.add_argument("--warmup", type=positive, default=1000)
    args = parser.parse_args()
    with jax.enable_x64(True):
        demo = run(args.seed, args.draws, args.warmup, observations=3840)
        passed = save_demo("composed_process", demo, args.output, args.seed, args.draws, args.warmup)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
