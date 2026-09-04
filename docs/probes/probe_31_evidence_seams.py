"""probe_31 -- the seams R4 assembles a log evidence out of, measured.

Run from the repository root:

    PYTHONPATH=. .venv/bin/python docs/probes/probe_31_evidence_seams.py

Five measurements, each one a number the R4 plan's frozen rulings rest on.
They are printed rather than asserted; the assertions live in the suite, and
this file exists so that a reader can re-derive them without reading a test.

**Everything here runs under `jax.enable_x64(True)`.** The package does not
turn x64 on -- `grep -rn enable_x64 src/ pyproject.toml` finds only docstrings
telling callers to opt in -- and under the default float32 the same assembly
agrees with a dense analytic log evidence only to about 2e-07, which no
evidence band should accept. Section 5 measures that gap rather than asserting
it, because the dtype is the premise every other number here depends on.

Section 1 is the defect R4 Task 1 repairs, kept here in its pre-repair form as
a description: it is what the arithmetic did at `fb1c21f`, and after the
repair the surplus printed is 0.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from bayesmith import const, det, observe, sample, trace
from bayesmith.dispatch.collapse import (
    collapse_graph,
    marginal_log_density,
    observed_descendants,
)
from bayesmith.exact.block import unchecked_operator
from bayesmith.graph.evaluate import log_joint

N_D, N_E = 6, 3
SIGMA_D, SIGMA_E = 0.5, 0.8
PRIOR_STD, PRIOR_MEAN = 1.7, 0.35
TAU_STD, TAU_MEAN = 1.1, -0.2
TAU_AT = 0.7


def _two_observation_graph(prior_std: float = PRIOR_STD):
    """`d` is downstream of the exact latent `w`; `e` is not."""
    basis = jnp.linspace(-1.0, 1.0, N_D) + 0.3
    d_obs = 1.2 * (basis * 0.9) + SIGMA_D * jax.random.normal(
        jax.random.key(3), (N_D,)
    )
    e_obs = jnp.asarray([0.4, -0.2, 1.1])

    def model():
        w = sample("w", lambda: dist.Normal(PRIOR_MEAN, prior_std))
        tau = sample("tau", lambda: dist.Normal(TAU_MEAN, TAU_STD))
        b = const("basis", basis)
        mu = det("mu", lambda b_, w_: b_ * w_, b, w, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, SIGMA_D).to_event(1), mu, obs=d_obs)
        nu = det("nu", lambda t_: t_ * jnp.ones(N_E), tau)
        observe("e", lambda m: dist.Normal(m, SIGMA_E).to_event(1), nu, obs=e_obs)

    return trace(model), np.asarray(basis, float), np.asarray(d_obs, float), np.asarray(
        e_obs, float
    )


def _dense_log_p_d(
    basis: np.ndarray, data: np.ndarray, prior_std: float = PRIOR_STD
) -> float:
    """log p(d) with w integrated out, by numpy and nothing else.

    Shares no QR, no pivots and no offset arithmetic with the implementation:
    a covariance is materialised, a residual is solved, a slogdet is taken.
    """
    covariance = prior_std**2 * np.outer(basis, basis) + SIGMA_D**2 * np.eye(
        basis.size
    )
    residual = data - basis * PRIOR_MEAN
    _, logdet = np.linalg.slogdet(covariance)
    return float(
        -0.5
        * (
            residual @ np.linalg.solve(covariance, residual)
            + logdet
            + basis.size * np.log(2.0 * np.pi)
        )
    )


def section_1_the_double_count() -> None:
    print("\n== 1. an observation outside the exact block ==")
    graph, basis, d_obs, e_obs = _two_observation_graph()
    print(f"  graph.observed                     : {graph.observed}")
    print(f"  observed_descendants(graph, ('w',)): {observed_descendants(graph, ('w',))}")
    block = unchecked_operator(
        graph, ("w",), at={"tau": jnp.asarray(TAU_AT)}, probe_gaussian=False
    )
    print(f"  block.data keys                    : {sorted(block.data)}")

    at = {"tau": jnp.asarray(TAU_AT)}
    found = float(log_joint(collapse_graph(graph, ("w",), ("tau",)), at))
    log_p_e = float(
        np.sum(
            -0.5 * ((e_obs - TAU_AT) / SIGMA_E) ** 2
            - np.log(SIGMA_E)
            - 0.5 * np.log(2.0 * np.pi)
        )
    )
    log_p_tau = float(
        -0.5 * ((TAU_AT - TAU_MEAN) / TAU_STD) ** 2
        - np.log(TAU_STD)
        - 0.5 * np.log(2.0 * np.pi)
    )
    oracle = _dense_log_p_d(basis, d_obs) + log_p_e + log_p_tau
    print(f"  reduced log_joint                  : {found!r}")
    print(f"  dense numpy oracle                 : {oracle!r}")
    print(f"  surplus                            : {found - oracle!r}")
    print(f"  log p(e | tau)                     : {log_p_e!r}")
    print(
        "  surplus IS log p(e|tau)            : "
        f"{abs((found - oracle) - log_p_e) < 1e-9}   <- True before the R4 repair"
    )
    print(
        f"  surplus is zero                    : {abs(found - oracle) < 1e-9}"
        "   <- True after it"
    )


def section_2_the_determinant_sign() -> None:
    print("\n== 2. the sign the block determinant enters with ==")
    graph, basis, d_obs, _ = _two_observation_graph()
    found = float(marginal_log_density(graph, ("w",), {"tau": jnp.asarray(TAU_AT)}))

    fisher = float(basis @ basis / SIGMA_D**2 + 1.0 / PRIOR_STD**2)
    residual = d_obs - basis * PRIOR_MEAN
    covariance = PRIOR_STD**2 * np.outer(basis, basis) + SIGMA_D**2 * np.eye(
        basis.size
    )
    quadratic = float(residual @ np.linalg.solve(covariance, residual))
    minus_half = -0.5 * (
        quadratic
        + basis.size * np.log(SIGMA_D**2)
        + np.log(PRIOR_STD**2)
        + np.log(fisher)
        + basis.size * np.log(2.0 * np.pi)
    )
    plus_half = minus_half + np.log(fisher)
    print(f"  marginal_log_density               : {found!r}")
    print(f"  determinant lemma, -0.5 logdet F   : {minus_half!r}")
    print(f"  determinant lemma, +0.5 logdet F   : {plus_half!r}")
    print(f"  logdet(F_bb)                       : {np.log(fisher)!r}")
    print(f"  the code uses -0.5                 : {abs(found - minus_half) < 1e-9}")


def section_3_the_prior_scale_canary() -> None:
    print("\n== 3. the prior normaliser is invisible at std = 1 ==")
    print("  -sum(log std) over one latent, by declared prior width:")
    for std in (0.05, 0.25, 1.0, 4.0, 60.0):
        print(f"    std={std:<6} -> -log(std) = {-np.log(std):+.6f}")
    print(
        "  A sweep that does not leave std = 1 cannot see this term go missing,"
        "\n  which is how it shipped missing once already."
    )


def section_4_the_prior_scale_sweep() -> None:
    """The whole point of the sweep: agreement at five declared widths."""
    print("\n== 4. the assembled log evidence across prior scales (x64) ==")
    for std in (0.05, 0.25, 1.0, 4.0, 60.0):
        graph, basis, d_obs, _ = _two_observation_graph(prior_std=std)
        found = float(
            marginal_log_density(graph, ("w",), {"tau": jnp.asarray(TAU_AT)})
        )
        oracle = _dense_log_p_d(basis, d_obs, prior_std=std)
        print(
            f"    std={std:<6} log Z = {found:+.12f}  dense = {oracle:+.12f}  "
            f"diff = {found - oracle:+.3e}"
        )


def section_5_the_dtype_premise() -> None:
    """Run OUTSIDE the x64 context on purpose -- that is the measurement."""
    print("\n== 5. the same assembly without x64 ==")
    print(f"  default dtype here: {jnp.zeros(1).dtype}")
    graph, basis, d_obs, _ = _two_observation_graph()
    found = float(marginal_log_density(graph, ("w",), {"tau": jnp.asarray(TAU_AT)}))
    oracle = _dense_log_p_d(basis, d_obs)
    print(f"  log Z = {found!r}")
    print(f"  dense = {oracle!r}")
    print(f"  diff  = {found - oracle:+.3e}   <- the reason every band is x64")


def main() -> None:
    with jax.enable_x64(True):
        section_1_the_double_count()
        section_2_the_determinant_sign()
        section_3_the_prior_scale_canary()
        section_4_the_prior_scale_sweep()
    section_5_the_dtype_premise()


if __name__ == "__main__":
    main()
