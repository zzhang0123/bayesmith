"""Probe 21: direct logdet methods against independent dense NumPy oracles.

This is an observational probe.  It prints measurements and exits zero when
all rows can be evaluated; its process status is never a correctness verdict.
Run from the repository root with ``.venv/bin/python``.
"""

from __future__ import annotations

import numpy as np

from bayesmith.marginal.logdet import (
    FrozenProbes,
    KroneckerStructure,
    LadderConfig,
    LogDetProblem,
    LowRankFactors,
    check_logdet_premises,
    dense_cholesky_logdet,
    finite_perturbation_logdet,
    frozen_hutchinson_trace_logdet,
    lambda_logdet,
    low_rank_logdet,
    state_space_logdet,
    structured_logdet,
    truncated_trace_logdet,
    whole_trace_log_tail_bound,
)


def _oracle(matrix: np.ndarray) -> float:
    return float(np.linalg.slogdet(matrix)[1])


def _relative(value: float, oracle: float) -> float:
    return abs(value - oracle) / max(abs(value), abs(oracle), 1e-300)


def _print_row(
    level: int,
    method: str,
    value: float,
    matrix: np.ndarray,
    verdict: str,
    *,
    tail_bound: float | None = None,
) -> None:
    oracle = _oracle(matrix)
    bound = "-" if tail_bound is None else f"{tail_bound:.6e}"
    print(
        f"{level:>2}  {method:<31} direct={value: .12e}  "
        f"oracle={oracle: .12e}  rel_err={_relative(value, oracle):.3e}  "
        f"premise={verdict}  tail_bound={bound}"
    )


def main() -> None:
    widths = np.array([1.3, 1.7, 2.2, 2.6])
    lam = np.diag(widths**2)
    factor = np.array([[0.12], [0.07], [0.03], [0.09]])
    perturbation = factor @ factor.T
    sigma = lam + perturbation

    base_problem = LogDetProblem(lam, np.zeros_like(lam))
    base_verdict = check_logdet_premises(base_problem)[0]
    _print_row(
        0,
        "Lambda itself",
        lambda_logdet(lam),
        lam,
        f"{base_verdict.satisfied}: {base_verdict.reason}",
    )

    low_factors = LowRankFactors(factor)
    low_problem = LogDetProblem(lam, perturbation, low_rank_factors=low_factors)
    low_verdict = check_logdet_premises(
        low_problem, config=LadderConfig(low_rank_fraction=0.5)
    )[1]
    _print_row(
        1,
        "low-rank Newton termination",
        low_rank_logdet(lam, perturbation, factors=low_factors),
        sigma,
        f"{low_verdict.satisfied}: {low_verdict.reason}",
    )

    chain = np.array(
        [
            [3.2, 0.25, 0.0, 0.0],
            [0.25, 2.8, -0.18, 0.0],
            [0.0, -0.18, 3.6, 0.21],
            [0.0, 0.0, 0.21, 2.9],
        ]
    )
    chain_problem = LogDetProblem(
        1.4 * np.eye(4), chain - 1.4 * np.eye(4), chain_block_size=1
    )
    chain_verdict = check_logdet_premises(
        chain_problem, config=LadderConfig(low_rank_max=0)
    )[2]
    _print_row(
        2,
        "state-space recursion",
        state_space_logdet(chain, block_size=1),
        chain,
        f"{chain_verdict.satisfied}: {chain_verdict.reason}",
    )

    circulant_first = np.array([3.4, 0.3, 0.1, 0.3])
    circulant = np.vstack([np.roll(circulant_first, index) for index in range(4)])
    structured_problem = LogDetProblem(
        1.6 * np.eye(4),
        circulant - 1.6 * np.eye(4),
        structure_kind="circulant",
    )
    structured_verdict = check_logdet_premises(structured_problem)[3]
    _print_row(
        3,
        "structured exact (circulant)",
        structured_logdet(circulant, kind="circulant"),
        circulant,
        f"{structured_verdict.satisfied}: {structured_verdict.reason}",
    )

    dense_verdict = check_logdet_premises(low_problem)[4]
    _print_row(
        4,
        "dense Cholesky",
        dense_cholesky_logdet(sigma),
        sigma,
        f"{dense_verdict.satisfied}: {dense_verdict.reason}",
    )

    finite_verdict = check_logdet_premises(low_problem)[5]
    _print_row(
        5,
        "finite Newton perturbation",
        finite_perturbation_logdet(lam, perturbation),
        sigma,
        f"{finite_verdict.satisfied}: {finite_verdict.reason}",
    )

    rho = 0.5
    order = 12
    trace_perturbation = rho * lam
    trace_sigma = lam + trace_perturbation
    traces = tuple(4 * rho**power for power in range(1, order + 1))
    trace_problem = LogDetProblem(
        lam,
        trace_perturbation,
        exact_power_traces=traces,
        trace_order=order,
        certified_rho=rho,
    )
    trace_verdict = check_logdet_premises(trace_problem)[6]
    bound = whole_trace_log_tail_bound(rho, order, 4)
    _print_row(
        6,
        "truncated trace-log",
        truncated_trace_logdet(
            lam,
            trace_perturbation,
            exact_power_traces=traces,
            order=order,
            rho=rho,
        ),
        trace_sigma,
        f"{trace_verdict.satisfied}: {trace_verdict.reason}",
        tail_bound=bound,
    )

    probes = FrozenProbes(
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0, 1.0],
            ]
        )
    )
    frozen_problem = LogDetProblem(
        lam, trace_perturbation, frozen_probes=probes, trace_order=order
    )
    frozen_verdict = check_logdet_premises(frozen_problem)[7]
    _print_row(
        7,
        "frozen Hutchinson trace-log",
        frozen_hutchinson_trace_logdet(
            lam, trace_perturbation, probes, order=order, rho=rho
        ),
        trace_sigma,
        f"{frozen_verdict.satisfied}: {frozen_verdict.reason}",
        tail_bound=bound,
    )

    kron_left = np.array([[2.3, 0.2], [0.2, 1.7]])
    kron_right = np.diag([1.4, 2.2])
    kron = np.kron(kron_left, kron_right)
    kron_structure = KroneckerStructure((kron_left, kron_right))
    value = structured_logdet(kron, kind="kronecker", structure=kron_structure)
    _print_row(
        3,
        "structured exact (Kronecker)",
        value,
        kron,
        "True: factors reconstruct Sigma",
    )

    refused = check_logdet_premises(low_problem)[8]
    print(
        f" 8  per-call resampling             direct=REFUSED  oracle=-  "
        f"rel_err=-  premise={refused.satisfied}: {refused.reason}  tail_bound=-"
    )


if __name__ == "__main__":
    main()
