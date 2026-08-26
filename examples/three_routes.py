"""Example 1 of the docs, runnable: three factors, three routes.

    python examples/three_routes.py            # the documented settings
    python examples/three_routes.py --quick    # a fast smoke (CI-sized)

Prints the derived partition, then samples it and reports each latent's
posterior against the truth, plus the prediction-space residual in units of
the noise -- the number that separates "the model is fit" from "the
parameters are pinned", which on this model are different claims (see the
valley discussion on the docs page).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import A, B, C, F, three_routes

from bayesmith.dispatch.factor import factor_partition, sample_factors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small chains, for CI")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    warmup, samples = (150, 250) if args.quick else (2000, 3000)

    graph, truths = three_routes(jax.random.key(args.seed))
    plan = factor_partition(graph)
    print(plan)
    print()

    draws = sample_factors(
        graph,
        plan,
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
            f"   truth {truth}   |pull| max {pull:.2f}"
        )

    fitted = jnp.exp(A @ jnp.mean(draws["x"], axis=0)) * (
        B @ jnp.mean(draws["y"], axis=0)
        + jnp.exp(C @ jnp.mean(draws["z"], axis=0))
    )
    truth_mu = jnp.exp(A @ truths["x"]) * (
        B @ truths["y"] + jnp.exp(C @ truths["z"])
    )
    residual = float(
        jnp.sqrt(jnp.mean(((fitted - truth_mu) / (F * truth_mu)) ** 2))
    )
    print(f"prediction-space residual rms / (F mu) = {residual:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
