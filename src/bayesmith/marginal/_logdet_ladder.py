"""Eager certification, premise checking, and dispatch for the logdet ladder."""

from __future__ import annotations

import numpy as np

from bayesmith.marginal._logdet_eager import (
    LadderConfig,
    LadderResult,
    LogDetProblem,
    PremiseVerdict,
    ResamplingRefused,
    _algebraic_rank_bound,
    _circulant_eigenvalues,
    _condition_certificate,
    _is_block_chain,
    _is_circulant,
    _is_diagonal,
    _is_positive_definite,
    _is_toeplitz,
    _n,
    _power_traces_match,
    dense_cholesky_logdet,
    finite_perturbation_logdet,
    frozen_hutchinson_trace_logdet,
    lambda_logdet,
    low_rank_logdet,
    spectral_radius,
    state_space_logdet,
    structured_logdet,
    truncated_trace_logdet,
)

__all__ = [
    "check_logdet_premises",
    "dispatch_logdet",
]

_METHODS = (
    "Lambda itself",
    "low-rank determinant lemma / finite e-polynomial",
    "state-space recursion",
    "structured exact",
    "dense Cholesky",
    "finite e-polynomial perturbation",
    "truncated trace-log",
    "frozen Hutchinson trace-log",
    "per-call resampling",
)


def _structure_request(
    problem: LogDetProblem, config: LadderConfig
) -> tuple[str | None, bool, str]:
    sigma = problem.lambda_matrix + problem.perturbation
    if sigma.ndim == 1:
        return "diagonal", bool(np.all(sigma > 0.0)), "compact diagonal entries checked"
    kind = problem.structure_kind
    if kind is None and _is_diagonal(
        sigma, rtol=config.structure_rtol, atol=config.structure_atol
    ):
        kind = "diagonal"
    if kind == "diagonal":
        valid = _is_diagonal(
            sigma, rtol=config.structure_rtol, atol=config.structure_atol
        )
        return (
            kind,
            valid,
            "diagonal entries were checked"
            if valid
            else "off-diagonal entries are nonzero",
        )
    if kind == "circulant":
        valid = _is_circulant(
            sigma, rtol=config.structure_rtol, atol=config.structure_atol
        )
        if valid:
            try:
                _circulant_eigenvalues(sigma)
            except ValueError:
                return kind, False, "circulant payload requires a real positive spectrum"
        return (
            kind,
            valid,
            "cyclic row shifts were checked" if valid else "rows are not cyclic shifts",
        )
    if kind == "toeplitz":
        valid = _is_toeplitz(
            sigma, rtol=config.structure_rtol, atol=config.structure_atol
        )
        return (
            kind,
            valid,
            "constant diagonals were checked"
            if valid
            else "diagonals are not constant",
        )
    if kind == "kronecker":
        if problem.structure is None:
            return kind, False, "no Kronecker factors were supplied for verification"
        factors_spd = all(
            _is_positive_definite(factor) for factor in problem.structure.factors
        )
        if not factors_spd:
            return (
                kind,
                False,
                (
                    "Kronecker factors reconstructing an SPD product are not each "
                    "symmetric positive definite as required by the payload"
                ),
            )
        reconstructed = problem.structure.factors[0]
        for factor in problem.structure.factors[1:]:
            reconstructed = np.kron(reconstructed, factor)
        valid = reconstructed.shape == sigma.shape and np.array_equal(
            reconstructed, sigma
        )
        return (
            kind,
            bool(valid),
            "factors reconstruct Sigma"
            if valid
            else "factors do not reconstruct Sigma",
        )
    return (
        None,
        False,
        "no supported diagonal/circulant/Toeplitz/Kronecker structure was found",
    )


def check_logdet_premises(
    problem: LogDetProblem, *, config: LadderConfig | None = None
) -> tuple[PremiseVerdict, ...]:
    """Judge all nine premises from inputs, without computing any logdet."""
    config = LadderConfig() if config is None else config
    lam = problem.lambda_matrix
    perturb = problem.perturbation
    sigma = lam + perturb
    n = _n(lam)
    sigma_symmetric = sigma.ndim == 1 or bool(np.array_equal(sigma, sigma.T))
    sigma_spd = sigma_symmetric and _is_positive_definite(sigma)
    condition, ceiling, condition_resolved = _condition_certificate(sigma)
    exact_arithmetic_resolved = sigma.ndim == 1 or condition_resolved
    try:
        rank = _algebraic_rank_bound(
            perturb, problem.low_rank_factors, lam
        )
        rank_evidence_valid = True
    except ValueError:
        rank = n
        rank_evidence_valid = False
    actual_rho = spectral_radius(lam, perturb)
    finite_polynomial_stable = actual_rho <= 1.0
    base = bool(np.array_equal(sigma, lam)) and exact_arithmetic_resolved
    has_algebraic_evidence = perturb.ndim == 1 or problem.low_rank_factors is not None
    low_rank = (
        rank_evidence_valid
        and has_algebraic_evidence
        and finite_polynomial_stable
        and sigma_spd
        and exact_arithmetic_resolved
        and rank <= config.low_rank_max
        and rank <= config.low_rank_fraction * n
    )

    if problem.chain_block_size is None or sigma.ndim != 2:
        chain = False
        chain_reason = "no dense matrix and chain block size were supplied"
    else:
        chain_structure = _is_block_chain(
            sigma,
            problem.chain_block_size,
            rtol=config.structure_rtol,
            atol=config.structure_atol,
        )
        chain = chain_structure and sigma_spd and condition_resolved
        if not chain_structure:
            chain_reason = (
                "the supplied matrix is not block tridiagonal at that block size"
            )
        elif not sigma_symmetric:
            chain_reason = "the block-tridiagonal matrix is not symmetric"
        elif not sigma_spd:
            chain_reason = "the symmetric block chain is not positive definite"
        elif not condition_resolved:
            chain_reason = (
                f"block-chain condition {condition:.8g} is not below the strict "
                f"dtype ceiling {ceiling:.8g}"
            )
        else:
            chain_reason = (
                "block-tridiagonal structure, symmetry, and positive definiteness "
                "were verified"
            )
    structure_kind, structured, structure_reason = _structure_request(problem, config)
    if structured and not sigma_spd:
        structured = False
        structure_reason = f"{structure_reason}, but Sigma is " + (
            "not symmetric" if not sigma_symmetric else "not positive definite"
        )
    elif structured and structure_kind != "diagonal" and not condition_resolved:
        structured = False
        structure_reason = (
            f"{structure_reason}, but condition {condition:.8g} is not below "
            f"the strict dtype ceiling {ceiling:.8g}"
        )

    dense = n <= config.dense_max_n and condition < ceiling and sigma_spd
    finite = (
        n <= config.finite_max_n or rank <= config.finite_max_rank
    ) and finite_polynomial_stable and sigma_spd and exact_arithmetic_resolved
    rho = actual_rho if problem.certified_rho is None else problem.certified_rho
    measured_rho_converges = actual_rho < 1.0
    rho_covers_input = actual_rho <= rho
    traces_verified = (
        problem.exact_power_traces is not None
        and problem.trace_order is not None
        and _power_traces_match(
            lam, perturb, problem.exact_power_traces, problem.trace_order
        )
    )
    trace = (
        traces_verified
        and measured_rho_converges
        and rho_covers_input
        and 0.0 <= rho < 1.0
    )
    frozen_width_valid = (
        problem.frozen_probes is not None
        and problem.frozen_probes.values.shape[1] == n
    )
    frozen = (
        frozen_width_valid
        and problem.trace_order is not None
        and measured_rho_converges
        and rho_covers_input
        and 0.0 <= rho < 1.0
    )

    return (
        PremiseVerdict(
            0,
            _METHODS[0],
            base,
            (
                "Sigma equals Lambda exactly"
                if base
                else "Sigma differs from Lambda or is numerically unresolved"
            ),
            {"n": n, "condition": condition, "condition_ceiling": ceiling},
        ),
        PremiseVerdict(
            1,
            _METHODS[1],
            low_rank,
            (
                f"verified rank {rank}; limits are {config.low_rank_max} and "
                f"{config.low_rank_fraction:g}*n; finite-e stability="
                f"{finite_polynomial_stable} "
                f"at rho={actual_rho:.8g}"
            ),
            {
                "n": n,
                "rank": rank,
                "rank_evidence_valid": rank_evidence_valid,
                "finite_polynomial_stable": finite_polynomial_stable,
                "positive_definite": sigma_spd,
                "condition": condition,
                "condition_ceiling": ceiling,
            },
        ),
        PremiseVerdict(
            2,
            _METHODS[2],
            chain,
            chain_reason,
            {"block_size": problem.chain_block_size},
        ),
        PremiseVerdict(
            3, _METHODS[3], structured, structure_reason, {"kind": structure_kind}
        ),
        PremiseVerdict(
            4,
            _METHODS[4],
            dense,
            (
                f"n={n} (limit {config.dense_max_n}); condition={condition:.8g} "
                f"(strict ceiling {ceiling:.8g}); symmetric={sigma_symmetric}; "
                f"positive_definite={sigma_spd}"
            ),
            {
                "n": n,
                "condition": condition,
                "condition_ceiling": ceiling,
                "symmetric": sigma_symmetric,
                "positive_definite": sigma_spd,
            },
        ),
        PremiseVerdict(
            5,
            _METHODS[5],
            finite,
            (
                f"n={n} (limit {config.finite_max_n}); rank={rank} "
                f"(limit {config.finite_max_rank}); finite-e stability="
                f"{finite_polynomial_stable} at rho={actual_rho:.8g}"
            ),
            {
                "n": n,
                "rank": rank,
                "rank_evidence_valid": rank_evidence_valid,
                "finite_polynomial_stable": finite_polynomial_stable,
                "positive_definite": sigma_spd,
                "condition": condition,
                "condition_ceiling": ceiling,
            },
        ),
        PremiseVerdict(
            6,
            _METHODS[6],
            trace,
            "power traces were verified and a conservative strict rho<1 certificate is present"
            if trace
            else (
                "needs verified exact power traces, fixed order, and a conservative "
                f"rho<1 certificate; measured rho={actual_rho:.8g}, certificate={rho:.8g}"
            ),
            {
                "rho": rho,
                "measured_rho": actual_rho,
                "order": problem.trace_order,
                "traces_verified": traces_verified,
            },
        ),
        PremiseVerdict(
            7,
            _METHODS[7],
            frozen,
            "immutable probes, a fixed order, and a strict rho<1 certificate are present"
            if frozen
            else (
                f"immutable FrozenProbes of width n={n}, a fixed order, and a "
                f"conservative rho<1 certificate are required; probe_width_valid="
                f"{frozen_width_valid}, measured_rho={actual_rho:.8g}"
            ),
            {
                "rho": rho,
                "measured_rho": actual_rho,
                "order": problem.trace_order,
                "probe_width_valid": frozen_width_valid,
            },
        ),
        PremiseVerdict(
            8,
            _METHODS[8],
            False,
            "per-call resampling is always refused because noisy logdet breaks HMC reversibility",
        ),
    )


def dispatch_logdet(
    problem: LogDetProblem, *, config: LadderConfig | None = None
) -> LadderResult:
    """Run the first satisfied row, preserving every preceding rejection."""
    config = LadderConfig() if config is None else config
    verdicts = check_logdet_premises(problem, config=config)
    sigma = problem.lambda_matrix + problem.perturbation
    rejected: list[PremiseVerdict] = []
    for verdict in verdicts:
        if not verdict.satisfied:
            rejected.append(verdict)
            continue
        if verdict.level == 0:
            value = lambda_logdet(problem.lambda_matrix)
        elif verdict.level == 1:
            value = low_rank_logdet(
                problem.lambda_matrix,
                problem.perturbation,
                factors=problem.low_rank_factors,
            )
        elif verdict.level == 2:
            value = state_space_logdet(
                sigma,
                block_size=int(problem.chain_block_size),
                rtol=config.structure_rtol,
                atol=config.structure_atol,
            )
        elif verdict.level == 3:
            value = structured_logdet(
                sigma,
                kind=verdict.details["kind"],
                structure=problem.structure,
                rtol=config.structure_rtol,
                atol=config.structure_atol,
            )
        elif verdict.level == 4:
            value = dense_cholesky_logdet(sigma)
        elif verdict.level == 5:
            value = finite_perturbation_logdet(
                problem.lambda_matrix,
                problem.perturbation,
                factors=(
                    problem.low_rank_factors
                    if verdict.details["rank_evidence_valid"]
                    else None
                ),
            )
        elif verdict.level == 6:
            value = truncated_trace_logdet(
                problem.lambda_matrix,
                problem.perturbation,
                exact_power_traces=problem.exact_power_traces,
                order=int(problem.trace_order),
                rho=float(verdict.details["rho"]),
            )
        else:
            value = frozen_hutchinson_trace_logdet(
                problem.lambda_matrix,
                problem.perturbation,
                problem.frozen_probes,
                order=int(problem.trace_order),
                rho=float(verdict.details["rho"]),
            )
        return LadderResult(verdict.level, verdict.method, value, tuple(rejected))
    raise ResamplingRefused(
        "No deterministic log-determinant rung qualified; per-call resampling "
        "is HMC-unsafe and refused.",
        rejected=verdicts,
    )
