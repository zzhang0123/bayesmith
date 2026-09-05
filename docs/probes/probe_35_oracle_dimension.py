"""probe_35 -- where a product-grid oracle stops being one, measured twice.

Run:  .venv/bin/python docs/probes/probe_35_oracle_dimension.py
      .venv/bin/python docs/probes/probe_35_oracle_dimension.py --lift-the-budget

R5 Task 3.2 asks for the dimension at which the quadrature oracle stops
converging within the time budget, because that dimension is the boundary of
R5's gradeable domain and goes into the module spec. The answer has two halves
and they are different facts, which is why this probe runs the sweep twice:

* **At the declared budget** (``residual_oracle.MAX_POINTS``, 5e7 points) the
  boundary is **four**. ``tests/dispatch/test_residual_fixtures.py`` asserts
  that, so it cannot go stale.
* **With the budget lifted** the boundary is **five**, and six is out of reach
  of any grid rather than of this one: five certifies at ``n = 65``, which is
  1.16e9 points and about 19 GB, while six would need ``65**6 = 7.5e10``.

Keeping the two apart matters because they have different remedies. A budget
boundary moves if someone buys memory. An arithmetic boundary does not.

**The second sweep allocates about 19 GB and takes about a minute per grid.**
It is behind ``--lift-the-budget`` for that reason, and it is not run by CI.

The family is ``tests/exact/residual_models.py::undeclared_family``: ``d``
latents on a design whose columns do not separate, affine in all of them and not
declared so, so every latent lands in the sampled block while the evidence stays
a closed form. A separable family would be the wrong instrument -- a product
grid gets a separable integral right one axis at a time, for a reason that has
nothing to do with dimension.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

from bayesmith.graph.reduction import as_graph
from tests.dispatch.residual_oracle import (
    AGREEMENT_FLOOR,
    MAX_POINTS,
    Span,
    oracle_joint,
)
from tests.exact import residual_models as rm

LIFTED = 2_000_000_000


def sweep(budget: int, dimensions: tuple[int, ...]) -> None:
    print("=" * 78)
    print(f"budget {budget:.3e} points ({budget * 8 / 2**30:.1f} GB per array)")
    print("=" * 78)
    print(
        f"  {'d':>2} {'cert':>5} {'n':>5} {'n_eval':>16} {'secs':>7}  "
        f"{'value - closed form':>22}  why"
    )
    with jax.enable_x64(True):
        for dimension in dimensions:
            graph = as_graph(rm.undeclared_family(dimension=dimension))
            parts = rm.undeclared_family_parts(graph, dimension=dimension)
            exact = rm.gaussian_log_evidence(
                parts["design"],
                parts["offset"],
                parts["data"],
                parts["prior_mean"],
                parts["prior_std"],
                parts["sigma"],
            )
            mean, spread, _precision = rm.gaussian_posterior(
                parts["design"],
                parts["offset"],
                parts["data"],
                parts["prior_mean"],
                parts["prior_std"],
                parts["sigma"],
            )
            spans = tuple(
                Span(name, float(c - 6.5 * w), float(c + 6.5 * w))
                for name, c, w in zip(parts["latents"], mean, spread, strict=True)
            )
            started = time.perf_counter()
            found = oracle_joint(
                graph,
                spans,
                resolution=AGREEMENT_FLOOR,
                start=9,
                refinements=8,
                max_points=budget,
            )
            elapsed = time.perf_counter() - started
            count = found.certificate.history[-1][0]
            error = found.certificate.history[-1][1] - exact
            why = "" if found.certified else found.certificate.refused.split(";")[0]
            print(
                f"  {dimension:>2} {found.certified!s:>5} {count:>5} "
                f"{count**dimension:>16,} {elapsed:>7.2f}  {error:>+22.6e}  {why[:64]}"
            )
    print("  -> the UNCERTIFIED values are the point: at five the number is right")
    print("     to ten decimals and at six it is wrong in the third, and nothing")
    print("     about either number says which one it is.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lift-the-budget",
        action="store_true",
        help=f"also sweep at {LIFTED:.0e} points -- about 19 GB and a minute per grid",
    )
    options = parser.parse_args()
    sweep(MAX_POINTS, (1, 2, 3, 4, 5, 6))
    if options.lift_the_budget:
        sweep(LIFTED, (4, 5, 6))
