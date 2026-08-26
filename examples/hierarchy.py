"""Example 2 of the docs, runnable: two kinds of noise, and a hierarchy.

    python examples/hierarchy.py               # the documented settings
    python examples/hierarchy.py --quick       # a fast smoke (CI-sized)

Prints the partition for BOTH parameterisations of the field's statistics --
linear and nonlinear in the hyperparameter ``y`` -- which must come out
identical, because the ancestry ejection that routes ``y`` is structural.
Then samples the linear variant and reports every level of the hierarchy
against its truth, the field ``w1`` against its own per-run realisation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import hierarchy

from bayesmith.dispatch.factor import factor_partition, sample_factors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small chains, for CI")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    warmup, samples = (150, 250) if args.quick else (1500, 2000)

    plans = {}
    for kind in ("linear", "nonlinear"):
        graph, _ = hierarchy(jax.random.key(args.seed), kind)
        plans[kind] = factor_partition(graph)
        print(f"-- y parameterises the field {kind.upper()}LY --")
        print(plans[kind])
        print()

    shape = lambda plan: [(b.latents, b.method) for b in plan.blocks]
    assert shape(plans["linear"]) == shape(plans["nonlinear"]), (
        "the ejection is structural, so the two parameterisations must "
        "partition identically"
    )
    print("(identical, as the ancestry rule requires)\n")

    graph, truths = hierarchy(jax.random.key(args.seed), "linear")
    draws = sample_factors(
        graph,
        plans["linear"],
        jax.random.key(args.seed + 1),
        num_warmup=warmup,
        num_samples=samples,
    )
    for name, truth in truths.items():
        mean = jnp.mean(draws[name], axis=0)
        std = jnp.std(draws[name], axis=0)
        pull = float(jnp.max(jnp.abs(mean - truth) / std))
        print(
            f"{name}: {jnp.round(mean, 3)} +- {jnp.round(std, 3)}"
            f"   truth {jnp.round(truth, 3)}   |pull| max {pull:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
