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
    _factor_projection_certificate,
    _is_block_chain,
    _is_circulant,
    _is_diagonal,
    _is_positive_definite,
    _is_symmetric,
    _is_toeplitz,
    _n,
    _power_traces_match,
    _symmetric_roundoff_representative,
    _two_sum_error,
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


def _sigma_payload(sigma: np.ndarray, config: LadderConfig) -> np.ndarray:
    """Return the exact dense payload representation used by dispatch."""
    if (
        sigma.ndim == 1
        or np.array_equal(sigma, sigma.T)
        or not _is_symmetric(
            sigma, rtol=config.structure_rtol, atol=config.structure_atol
        )
    ):
        return sigma
    return _symmetric_roundoff_representative(
        sigma, operation="Sigma payload symmetrization"
    )


def _structure_request(
    problem: LogDetProblem, config: LadderConfig, sigma: np.ndarray
) -> tuple[str | None, bool, str]:
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
    n = _n(lam)
    try:
        sigma, _ = _two_sum_error(lam, perturb)
    except ValueError as error:
        sigma_formation_valid = False
        sigma_formation_reason = str(error)
        sigma_exactly_symmetric = False
        sigma_symmetric = False
        sigma_spd = False
        condition = float("inf")
        dtype = lam.dtype if np.issubdtype(lam.dtype, np.inexact) else np.dtype(float)
        ceiling = 1.0 / float(np.finfo(dtype).eps)
        condition_resolved = False
    else:
        sigma_formation_valid = True
        sigma_formation_reason = "Lambda + perturbation addition remained finite"
        sigma_exactly_symmetric = sigma.ndim == 1 or bool(
            np.array_equal(sigma, sigma.T)
        )
        sigma_symmetric = sigma.ndim == 1 or _is_symmetric(
            sigma,
            rtol=config.structure_rtol,
            atol=config.structure_atol,
        )
        symmetric_sigma = _sigma_payload(sigma, config)
        sigma_spd = sigma_symmetric and _is_positive_definite(
            sigma,
            rtol=config.structure_rtol,
            atol=config.structure_atol,
        )
        condition, ceiling, condition_resolved = _condition_certificate(
            symmetric_sigma
        )
    factor_projection_eta: float | None = None
    factor_log_error_bound: float | None = None
    factor_log_error_ceiling: float | None = None
    factor_base_condition: float | None = None
    factor_base_condition_ceiling: float | None = None
    factor_base_log_error_bound: float | None = None
    factor_base_arithmetic_valid = False
    factor_reduced_arithmetic_valid = False
    factor_reduced_eta: float | None = None
    factor_reduced_condition: float | None = None
    factor_reduced_formation_error_norm: float | None = None
    factor_reduced_smallest_singular: float | None = None
    factor_reduced_log_error_bound: float | None = None
    factor_reduced_sign: float | None = None
    factor_total_log_error_bound: float | None = None
    rank_evidence_reason = "no factor payload was supplied"
    if problem.low_rank_factors is None:
        rank = _algebraic_rank_bound(perturb)
        rank_evidence_valid = True
    else:
        try:
            factor_certificate = _factor_projection_certificate(
                perturb, problem.low_rank_factors, lam
            )
        except ValueError as error:
            rank = n
            rank_evidence_valid = False
            rank_evidence_reason = str(error)
        else:
            factor_projection_eta = factor_certificate.eta
            factor_log_error_bound = factor_certificate.log_error_bound
            factor_log_error_ceiling = factor_certificate.ceiling
            factor_base_condition = factor_certificate.base_condition
            factor_base_condition_ceiling = factor_certificate.base_condition_ceiling
            factor_base_log_error_bound = factor_certificate.base_log_error_bound
            factor_base_arithmetic_valid = factor_certificate.base_arithmetic_valid
            factor_reduced_arithmetic_valid = (
                factor_certificate.reduced_arithmetic_valid
            )
            factor_reduced_eta = factor_certificate.reduced_eta
            factor_reduced_condition = factor_certificate.reduced_condition
            factor_reduced_formation_error_norm = (
                factor_certificate.reduced_formation_error_norm
            )
            factor_reduced_smallest_singular = (
                factor_certificate.reduced_smallest_singular
            )
            factor_reduced_log_error_bound = (
                factor_certificate.reduced_log_error_bound
            )
            factor_reduced_sign = factor_certificate.reduced_sign
            factor_total_log_error_bound = factor_certificate.total_log_error_bound
            rank_evidence_valid = factor_certificate.valid
            rank_evidence_reason = factor_certificate.reason
            rank = (
                problem.low_rank_factors.rank_bound
                if rank_evidence_valid
                else n
            )
    try:
        actual_rho = spectral_radius(lam, perturb)
    except ValueError as error:
        actual_rho = float("inf")
        rho_measurement_valid = False
        rho_measurement_reason = str(error)
    else:
        rho_measurement_valid = True
        rho_measurement_reason = "spectral-radius measurement succeeded"
    if not sigma_formation_valid:
        finite_payload_rho = float("inf")
        finite_payload_rho_measurement_valid = False
        finite_payload_rho_measurement_reason = sigma_formation_reason
    else:
        try:
            with np.errstate(over="raise", invalid="raise"):
                finite_payload_perturbation = symmetric_sigma - lam
            finite_payload_rho = spectral_radius(
                lam, finite_payload_perturbation
            )
        except (FloatingPointError, ValueError) as error:
            finite_payload_rho = float("inf")
            finite_payload_rho_measurement_valid = False
            finite_payload_rho_measurement_reason = str(error)
        else:
            finite_payload_rho_measurement_valid = True
            finite_payload_rho_measurement_reason = (
                "finite-payload spectral-radius measurement succeeded"
            )
    finite_polynomial_stable = (
        finite_payload_rho_measurement_valid and finite_payload_rho <= 1.0
    )
    determinant_lemma_payload = bool(
        problem.low_rank_factors is not None
        and rank_evidence_valid
        and sigma_formation_valid
        and sigma_exactly_symmetric
    )
    finite_payload_stable = finite_polynomial_stable or determinant_lemma_payload
    dense_arithmetic_resolved = sigma_formation_valid and (
        sigma.ndim == 1 or condition_resolved
    )
    base = (
        sigma_formation_valid
        and bool(np.array_equal(sigma, lam))
        and dense_arithmetic_resolved
    )
    compact_diagonal_payload = perturb.ndim == 1
    low_rank = (
        rank_evidence_valid
        and (compact_diagonal_payload or determinant_lemma_payload)
        and sigma_spd
        and rank <= config.low_rank_max
        and rank <= config.low_rank_fraction * n
    )

    if not sigma_formation_valid:
        chain = False
        chain_reason = sigma_formation_reason
    elif problem.chain_block_size is None or sigma.ndim != 2:
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
    if sigma_formation_valid:
        structure_kind, structured, structure_reason = _structure_request(
            problem, config, sigma
        )
    else:
        structure_kind = problem.structure_kind
        structured = False
        structure_reason = sigma_formation_reason
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

    dense = n <= config.dense_max_n and condition_resolved and sigma_spd
    finite_size_qualified = n <= config.finite_max_n or (
        (compact_diagonal_payload or determinant_lemma_payload)
        and rank <= config.finite_max_rank
    )
    finite = finite_size_qualified and finite_payload_stable and sigma_spd and (
        determinant_lemma_payload or dense_arithmetic_resolved
    )
    rho = actual_rho if problem.certified_rho is None else problem.certified_rho
    measured_rho_converges = rho_measurement_valid and actual_rho < 1.0
    rho_covers_input = rho_measurement_valid and actual_rho <= rho
    traces_verified = (
        rho_measurement_valid
        and problem.exact_power_traces is not None
        and problem.trace_order is not None
        and _power_traces_match(
            lam, perturb, problem.exact_power_traces, problem.trace_order
        )
    )
    trace = (
        sigma_formation_valid
        and traces_verified
        and measured_rho_converges
        and rho_covers_input
        and 0.0 <= rho < 1.0
    )
    frozen_width_valid = (
        problem.frozen_probes is not None
        and problem.frozen_probes.values.shape[1] == n
    )
    frozen = (
        sigma_formation_valid
        and frozen_width_valid
        and problem.trace_order is not None
        and problem.trace_order >= 0
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
            {
                "n": n,
                "condition": condition,
                "condition_ceiling": ceiling,
                "sigma_formation_valid": sigma_formation_valid,
                "sigma_formation_reason": sigma_formation_reason,
            },
        ),
        PremiseVerdict(
            1,
            _METHODS[1],
            low_rank,
            (
                f"verified rank {rank}; limits are {config.low_rank_max} and "
                f"{config.low_rank_fraction:g}*n; stable factorized "
                f"payload is independent of rho={actual_rho:.8g}"
                if rank_evidence_valid
                else f"factor rank evidence rejected: {rank_evidence_reason}"
            ),
            {
                "n": n,
                "rank": rank,
                "rank_evidence_valid": rank_evidence_valid,
                "finite_polynomial_stable": finite_polynomial_stable,
                "finite_payload_rho": finite_payload_rho,
                "finite_payload_rho_measurement_valid": (
                    finite_payload_rho_measurement_valid
                ),
                "finite_payload_rho_measurement_reason": (
                    finite_payload_rho_measurement_reason
                ),
                "determinant_lemma_payload": determinant_lemma_payload,
                "exactly_symmetric": sigma_exactly_symmetric,
                "positive_definite": sigma_spd,
                "condition": condition,
                "condition_ceiling": ceiling,
                "rank_evidence_reason": rank_evidence_reason,
                "factor_projection_eta": factor_projection_eta,
                "factor_log_error_bound": factor_log_error_bound,
                "factor_log_error_ceiling": factor_log_error_ceiling,
                "factor_base_condition": factor_base_condition,
                "factor_base_condition_ceiling": factor_base_condition_ceiling,
                "factor_base_log_error_bound": factor_base_log_error_bound,
                "factor_base_arithmetic_valid": factor_base_arithmetic_valid,
                "factor_reduced_eta": factor_reduced_eta,
                "factor_reduced_condition": factor_reduced_condition,
                "factor_reduced_formation_error_norm": (
                    factor_reduced_formation_error_norm
                ),
                "factor_reduced_smallest_singular": (
                    factor_reduced_smallest_singular
                ),
                "factor_reduced_log_error_bound": factor_reduced_log_error_bound,
                "factor_reduced_sign": factor_reduced_sign,
                "factor_reduced_arithmetic_valid": factor_reduced_arithmetic_valid,
                "factor_total_log_error_bound": factor_total_log_error_bound,
                "sigma_formation_valid": sigma_formation_valid,
                "sigma_formation_reason": sigma_formation_reason,
                "measured_rho": actual_rho,
                "rho_measurement_valid": rho_measurement_valid,
                "rho_measurement_reason": rho_measurement_reason,
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
                f"{finite_payload_stable} at payload rho="
                f"{finite_payload_rho:.8g}"
            ),
            {
                "n": n,
                "rank": rank,
                "rank_evidence_valid": rank_evidence_valid,
                "finite_polynomial_stable": finite_polynomial_stable,
                "finite_payload_rho": finite_payload_rho,
                "finite_payload_rho_measurement_valid": (
                    finite_payload_rho_measurement_valid
                ),
                "finite_payload_rho_measurement_reason": (
                    finite_payload_rho_measurement_reason
                ),
                "determinant_lemma_payload": determinant_lemma_payload,
                "exactly_symmetric": sigma_exactly_symmetric,
                "positive_definite": sigma_spd,
                "condition": condition,
                "condition_ceiling": ceiling,
                "rank_evidence_reason": rank_evidence_reason,
                "factor_projection_eta": factor_projection_eta,
                "factor_log_error_bound": factor_log_error_bound,
                "factor_log_error_ceiling": factor_log_error_ceiling,
                "factor_base_condition": factor_base_condition,
                "factor_base_condition_ceiling": factor_base_condition_ceiling,
                "factor_base_log_error_bound": factor_base_log_error_bound,
                "factor_base_arithmetic_valid": factor_base_arithmetic_valid,
                "factor_reduced_eta": factor_reduced_eta,
                "factor_reduced_condition": factor_reduced_condition,
                "factor_reduced_formation_error_norm": (
                    factor_reduced_formation_error_norm
                ),
                "factor_reduced_smallest_singular": (
                    factor_reduced_smallest_singular
                ),
                "factor_reduced_log_error_bound": factor_reduced_log_error_bound,
                "factor_reduced_sign": factor_reduced_sign,
                "factor_reduced_arithmetic_valid": factor_reduced_arithmetic_valid,
                "factor_total_log_error_bound": factor_total_log_error_bound,
                "sigma_formation_valid": sigma_formation_valid,
                "sigma_formation_reason": sigma_formation_reason,
                "measured_rho": actual_rho,
                "rho_measurement_valid": rho_measurement_valid,
                "rho_measurement_reason": rho_measurement_reason,
            },
        ),
        PremiseVerdict(
            6,
            _METHODS[6],
            trace,
            (
                "power traces were verified and a conservative strict rho<1 "
                f"certificate covers measured rho={actual_rho:.17g} with "
                f"certificate={rho:.17g}"
            )
            if trace
            else (
                "power-trace rho premise could not be measured: "
                f"{rho_measurement_reason}"
            )
            if not rho_measurement_valid
            else (
                "needs verified exact power traces, fixed order, and a conservative "
                f"rho<1 certificate; measured rho={actual_rho:.17g}, "
                f"certificate={rho:.17g}"
            ),
            {
                "rho": rho,
                "measured_rho": actual_rho,
                "order": problem.trace_order,
                "traces_verified": traces_verified,
                "rho_measurement_valid": rho_measurement_valid,
                "rho_measurement_reason": rho_measurement_reason,
            },
        ),
        PremiseVerdict(
            7,
            _METHODS[7],
            frozen,
            (
                "immutable probes and a fixed order are present; strict "
                f"certificate={rho:.17g} covers measured rho={actual_rho:.17g}"
            )
            if frozen
            else (
                "frozen-probe rho premise could not be measured: "
                f"{rho_measurement_reason}"
            )
            if not rho_measurement_valid
            else (
                f"immutable FrozenProbes of width n={n}, a fixed order, and a "
                f"conservative rho<1 certificate are required; probe_width_valid="
                f"{frozen_width_valid}, measured_rho={actual_rho:.17g}, "
                f"certificate={rho:.17g}"
            ),
            {
                "rho": rho,
                "measured_rho": actual_rho,
                "order": problem.trace_order,
                "probe_width_valid": frozen_width_valid,
                "rho_measurement_valid": rho_measurement_valid,
                "rho_measurement_reason": rho_measurement_reason,
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
    if not any(verdict.satisfied for verdict in verdicts):
        raise ResamplingRefused(
            "No deterministic log-determinant rung qualified; per-call resampling "
            "is HMC-unsafe and refused.",
            rejected=verdicts,
        )
    try:
        sigma, _ = _two_sum_error(
            problem.lambda_matrix, problem.perturbation
        )
    except ValueError as error:
        raise ResamplingRefused(
            "No deterministic log-determinant rung qualified because "
            f"{error}; per-call resampling is HMC-unsafe and refused.",
            rejected=verdicts,
        ) from None
    sigma_payload = _sigma_payload(sigma, config)
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
                sigma_payload,
                block_size=int(problem.chain_block_size),
                rtol=config.structure_rtol,
                atol=config.structure_atol,
            )
        elif verdict.level == 3:
            value = structured_logdet(
                sigma_payload,
                kind=verdict.details["kind"],
                structure=problem.structure,
                rtol=config.structure_rtol,
                atol=config.structure_atol,
            )
        elif verdict.level == 4:
            value = dense_cholesky_logdet(sigma_payload)
        elif verdict.level == 5:
            factors = (
                problem.low_rank_factors
                if verdict.details["determinant_lemma_payload"]
                else None
            )
            value = finite_perturbation_logdet(
                problem.lambda_matrix,
                (
                    problem.perturbation
                    if factors is not None
                    else sigma_payload - problem.lambda_matrix
                ),
                factors=factors,
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
