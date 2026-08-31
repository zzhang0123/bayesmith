"""probe_26 -- P7: what the A4 pilot can conclude, and what the ledger records.

Reproduces every number in
`docs/superpowers/specs/2026-08-31-p7-pilot-ledger.md`.  Run from the
repository root with the test package importable:

    PYTHONPATH=. .venv/bin/python docs/probes/probe_26_pilot_and_ledger.py

Three questions, in the order they had to be answered:

1. **Is the ratio the signal?**  Twenty seeds of Neal's funnel and twenty of a
   jointly Gaussian control, at 200 000 draws each.  The funnel's ABSOLUTE
   quadratic reading moves by 2.35x within one feature construction and by
   6.08x against the independently constructed 0.619 the plan recorded; the
   Gaussian control's RATIO never leaves [1.0000001, 1.0000508].
2. **Does the declared multiple separate them?**  7.0 sits between the 6.08x
   estimator spread and the 8.36x worst funnel draw.  20/20 funnel draws veto;
   0/20 Gaussian draws do.
3. **What does the ledger locate a real miss at?**  One mixed fixture, one
   short run, one predicted/measured pair, and the term that dominated the
   prediction.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import solve_triangular

from bayesmith.dispatch.pilot import DECLARED_MULTIPLE, pilot_report

DRAWS = 200_000
SEEDS = 20


def funnel(seed: int, draws: int = DRAWS) -> tuple[np.ndarray, np.ndarray]:
    """v ~ N(0, 3), x ~ N(0, exp(v/2)): Laplace correlation exactly 0.0."""
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, 3.0, size=draws)
    x = rng.normal(0.0, np.exp(v / 2.0))
    return v[:, None], x[:, None]


def gaussian(seed: int, draws: int = DRAWS) -> tuple[np.ndarray, np.ndarray]:
    """A jointly Gaussian pair at rho=0.6: nothing for the squares to find."""
    rng = np.random.default_rng(1000 + seed)
    z = rng.normal(size=(draws, 2))
    a = z[:, 0]
    b = 0.6 * z[:, 0] + math.sqrt(1.0 - 0.36) * z[:, 1]
    return a[:, None], b[:, None]


def cholesky_reference(left: np.ndarray, right: np.ndarray) -> float:
    """The independent oracle: canonical correlation by whitening, not by QR.

    `pilot.canonical_correlation` takes the singular values of `Q_l.T @ Q_r`;
    this takes them of the doubly whitened cross-covariance, the way
    `diagnose/coupling.py` does it.  The two agree to 5.7e-17 on the linear
    reading this script compares, and to 2.3e-15 on the quadratic one -- the
    augmented block's second moment is heavy-tailed by construction, which is
    why the shipped module whitens by QR and not by a Cholesky of it.
    """
    n = left.shape[0]
    centred_left = left - left.mean(axis=0)
    centred_right = right - right.mean(axis=0)
    factor_left = np.linalg.cholesky(centred_left.T @ centred_left / n)
    factor_right = np.linalg.cholesky(centred_right.T @ centred_right / n)
    whitened = solve_triangular(
        factor_left, centred_left.T @ centred_right / n, lower=True
    )
    whitened = solve_triangular(factor_right, whitened.T, lower=True).T
    return float(np.linalg.svd(whitened, compute_uv=False)[0])


def question_one_and_two() -> None:
    print("== funnel, 200 000 draws, 20 seeds ==")
    linear, quadratic, ratios, worst_oracle = [], [], [], 0.0
    vetoes = 0
    for seed in range(SEEDS):
        left, right = funnel(seed)
        report = pilot_report(left, right, n_eff=float(DRAWS))
        worst_oracle = max(
            worst_oracle, abs(report.linear_cc - cholesky_reference(left, right))
        )
        linear.append(report.linear_cc)
        quadratic.append(report.quadratic_cc)
        ratios.append(report.ratio)
        vetoes += int(report.vetoed)
        if seed == 0:
            print("  seed 0:", report.line())
    print(f"  linear    min {min(linear):.6f} max {max(linear):.6f}")
    print(f"  quadratic min {min(quadratic):.6f} max {max(quadratic):.6f} "
          f"(spread {max(quadratic) / min(quadratic):.2f}x)")
    print(f"  ratio     min {min(ratios):.2f} max {max(ratios):.2f}")
    print(f"  vetoes at a declared multiple of {DECLARED_MULTIPLE:g}: {vetoes}/{SEEDS}")
    print(f"  worst disagreement with the cholesky oracle: {worst_oracle:.3e}")
    print("  cross-construction spread against the plan's recorded 0.619: "
          f"{0.619 / min(quadratic):.2f}x")

    print("== jointly Gaussian control, rho=0.6, 20 seeds ==")
    null_ratios = []
    null_vetoes = 0
    for seed in range(SEEDS):
        report = pilot_report(*gaussian(seed), n_eff=float(DRAWS))
        null_ratios.append(report.ratio)
        null_vetoes += int(report.vetoed)
    print(f"  ratio min {min(null_ratios):.7f} max {max(null_ratios):.7f}")
    print(f"  vetoes: {null_vetoes}/{SEEDS}")

    print("== the floor is a real gate ==")
    starved = pilot_report(*funnel(0), n_eff=100.0)
    print(f"  seed 0 declared to carry 100 effective draws: floor "
          f"{starved.floor:.4f} > quadratic {starved.quadratic_cc:.4f}, "
          f"ratio {starved.ratio:.2f} -> vetoed={starved.vetoed}")


def question_three() -> None:
    """One real mixed run, one ledger row. Needs the test fixtures importable."""
    import jax

    from bayesmith import compile as compile_graph
    from tests.exact.models import indirect_ancestor

    print("== the ledger on one real run ==")
    with jax.enable_x64(True):
        graph = indirect_ancestor()
        priced = compile_graph(graph, strategy="cost")
        posterior = priced.sample(jax.random.key(0), num_samples=200, num_warmup=200)
        declared = compile_graph(graph)
        plain = declared.sample(jax.random.key(0), num_samples=200, num_warmup=200)
    print("  ", priced.ladder.line())
    print("  ", posterior.cost.line())
    print("   declared plan carries no ledger:", plain.cost is None)


if __name__ == "__main__":
    question_one_and_two()
    question_three()
