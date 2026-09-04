"""probe_33 -- D107's coefficient, measured rather than chosen.

Run from the repository root:

    PYTHONPATH=. .venv/bin/python docs/probes/probe_33_d107_agreement_band.py

R4 §0.8 fixes D107's FORM by derivation and leaves its COEFFICIENT to
measurement::

    atol = C * n * eps(dtype) * max(kappa_2(C_marginal), |log Z|)

**The plan's first guess at that form was wrong and this probe is why it
changed.** §0.8 wrote ``C * n * eps * kappa_2`` alone, from backward stability
of the QR route. Measured, the worst cell of the whole grid is
``k=2, n=4, prior_std=0.05`` -- where ``kappa_2`` is **1.04**, so the bound is
9.2e-16 while the observed gap is 1.2e-14. Nothing about the conditioning
explains it: the gap there is about thirteen ULP of a log evidence whose
MAGNITUDE is around 10. A sum of logs carries absolute error proportional to
the largest term, and at the well-conditioned end that term is ``log Z`` itself,
not anything ``kappa`` sees.

The form that dominates both regimes was chosen by measurement rather than by
argument. Over the same grid, worst coefficient and spread (max/median):

===================================  ========  ========
form                                 worst C    spread
===================================  ========  ========
``n eps kappa``                        13.462     694x
``n eps |logZ|``                      164.585     442x
``n eps (|logZ| + log kappa)``        106.900     415x
``n eps max(kappa, |logZ|)``            1.134      60x
===================================  ========  ========

The winner is the only one whose coefficient is ``O(1)``, which is what a
backward-stability argument predicts of a correct form -- a form needing a
coefficient of 165 is a form doing the wrong thing with a fudge factor on top.

**Why a fixed absolute tolerance is refused.** Measured in probe_31 §4, the
agreement between the square-root route and a dense analytic value runs from
8.9e-16 at ``prior_std = 0.05`` to 3.9e-13 at ``prior_std = 60`` -- nearly
three orders of magnitude on one fixture family. Any single float is either
slack at the tight end or red at the wide one, and a scale-blind ``atol`` over
fixtures that deliberately sweep the scale is exactly the fixture class that
failed on Linux and cost four release tags.

**The Linux rule.** §0.8 requires C to be re-measured under

    docker run --platform linux/amd64 -e OPENBLAS_CORETYPE=ZEN ...

before registration, and the plan stops for an owner ruling if the two differ
by more than a factor of 4. macOS uses Accelerate and Linux wheels use
scipy-openblas; `ubuntu-latest` is a heterogeneous pool (an EPYC 7763, an EPYC
9V74 and a Xeon 8370C were all measured across three runs), so the band must
not be tuned to a CPU.

Everything runs under ``jax.enable_x64(True)``; without it the same comparison
is only good to about 3e-07, measured in probe_31 §5.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from bayesmith import const, det, observe, sample, trace
from bayesmith.dispatch.collapse import marginal_log_density
from bayesmith.dispatch.evidence import assemble_exact

PRIOR_SCALES = (0.05, 0.25, 1.0, 4.0, 60.0)
DATA_SIZES = (4, 6, 13, 40)
BLOCK_WIDTHS = (1, 2, 3)
SIGMA = 0.5


def _graph(width: int, n: int, prior_std: float):
    """A block of ``width`` exact latents seen through ``n`` observations."""
    key = jax.random.key(3)
    basis = jnp.stack(
        [jnp.linspace(-1.0, 1.0, n) ** power for power in range(width)], axis=1
    )
    truth = jnp.asarray([1.2, -0.4, 0.7][:width])
    data = basis @ truth + SIGMA * jax.random.normal(key, (n,))

    def model():
        block = [
            sample(f"w{i}", lambda: dist.Normal(0.35, prior_std)) for i in range(width)
        ]
        b = const("basis", basis)
        mu = det(
            "mu",
            lambda b_, *w_: b_ @ jnp.stack(w_),
            b,
            *block,
            linear_in=tuple(f"w{i}" for i in range(width)),
        )
        observe("d", lambda m: dist.Normal(m, SIGMA).to_event(1), mu, obs=data)

    names = tuple(f"w{i}" for i in range(width))
    return trace(model), names, np.asarray(basis, float), np.asarray(data, float)


def _dense(basis: np.ndarray, data: np.ndarray, prior_std: float):
    """``(log Z, kappa_2(C))`` from a materialised covariance and nothing else."""
    n, k = basis.shape
    covariance = prior_std**2 * (basis @ basis.T) + SIGMA**2 * np.eye(n)
    residual = data - basis @ np.full(k, 0.35)
    _, logdet = np.linalg.slogdet(covariance)
    value = -0.5 * (
        residual @ np.linalg.solve(covariance, residual)
        + logdet
        + n * np.log(2.0 * np.pi)
    )
    return float(value), float(np.linalg.cond(covariance))


def main() -> None:
    eps = float(np.finfo(np.float64).eps)
    print(f"eps(float64) = {eps:.6e}")
    print(
        f"{'k':>2} {'n':>3} {'prior_std':>10} {'kappa_2(C)':>12} {'|logZ|':>10} "
        f"{'|assembled-dense|':>18} {'ratio C':>10} {'|sqrt-dense|':>14}"
    )
    worst_assembled = 0.0
    worst_sqrt = 0.0
    worst_cell = None
    with jax.enable_x64(True):
        assert jnp.zeros(1).dtype == jnp.float64, "this probe is meaningless in float32"
        for width in BLOCK_WIDTHS:
            for n in DATA_SIZES:
                if n < width + 1:
                    continue
                for prior_std in PRIOR_SCALES:
                    graph, names, basis, data = _graph(width, n, prior_std)
                    assembled = float(assemble_exact(graph, names, {}).log_evidence)
                    sqrt_route = float(marginal_log_density(graph, names, {}))
                    oracle, kappa = _dense(basis, data, prior_std)

                    gap = abs(assembled - oracle)
                    sqrt_gap = abs(sqrt_route - oracle)
                    scale = n * eps * max(kappa, abs(oracle))
                    ratio = gap / scale if scale > 0 else float("nan")
                    print(
                        f"{width:>2} {n:>3} {prior_std:>10} {kappa:>12.4e} "
                        f"{abs(oracle):>10.3f} "
                        f"{gap:>18.4e} {ratio:>10.3f} {sqrt_gap:>14.4e}"
                    )
                    if ratio > worst_assembled:
                        worst_assembled = ratio
                        worst_cell = (width, n, prior_std, kappa, gap)
                    sqrt_ratio = sqrt_gap / scale if scale > 0 else 0.0
                    worst_sqrt = max(worst_sqrt, sqrt_ratio)

    print()
    print(f"worst C over the assembled route : {worst_assembled:.4f}")
    print(f"worst C over the sqrt-info route : {worst_sqrt:.4f}")
    print(f"worst cell (k, n, prior_std, kappa, gap): {worst_cell}")
    print()
    print(
        "D107 is registered as atol = C * n * eps * kappa_2(C_marginal) with C\n"
        "the larger of the two worst ratios above, rounded UP to a round number,\n"
        "and the registry prose says which half is derived and which measured.\n"
        "Re-measure under linux/amd64 + OPENBLAS_CORETYPE=ZEN before registering;\n"
        "a factor of more than 4 between the two stops the plan for a ruling."
    )


if __name__ == "__main__":
    main()
