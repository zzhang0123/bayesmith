"""Adversarial stability tests for determinant-lemma rank evidence."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest

from bayesmith.marginal.logdet import (
    FrozenProbes,
    KroneckerStructure,
    LadderConfig,
    LogDetProblem,
    LowRankFactors,
    ResamplingRefused,
    check_logdet_premises,
    dispatch_logdet,
    finite_perturbation_logdet,
    lambda_logdet,
    low_rank_logdet,
    spectral_radius,
    state_space_logdet,
)


def _decimal_three_by_three_logdet(matrix: np.ndarray) -> float:
    """Evaluate a 3x3 determinant from the exact binary input values."""

    def exact_decimal(value: np.float64) -> Decimal:
        numerator, denominator = float(value).as_integer_ratio()
        return Decimal(numerator) / Decimal(denominator)

    with localcontext() as context:
        context.prec = 100
        rows = [[exact_decimal(value) for value in row] for row in matrix]
        a, b, c = rows[0]
        d, e, f = rows[1]
        g, h, i = rows[2]
        determinant = (
            a * (e * i - f * h)
            - b * (d * i - f * g)
            + c * (d * h - e * g)
        )
        assert determinant > 0
        return float(determinant.ln())


def _decimal_two_by_two_logdet(matrix: np.ndarray) -> float:
    """Evaluate a 2x2 determinant from the exact stored binary values."""

    def exact_decimal(value: np.floating) -> Decimal:
        numerator, denominator = float(value).as_integer_ratio()
        return Decimal(numerator) / Decimal(denominator)

    with localcontext() as context:
        context.prec = 200
        a, b = (exact_decimal(value) for value in matrix[0])
        c, d = (exact_decimal(value) for value in matrix[1])
        determinant = a * d - b * c
        assert determinant > 0
        return float(determinant.ln())


def _factor_only_config(*, finite_max_rank: int = 0) -> LadderConfig:
    return LadderConfig(
        low_rank_max=0,
        low_rank_fraction=1.0,
        dense_max_n=0,
        finite_max_n=0,
        finite_max_rank=finite_max_rank,
    )


def _scaled_near_null_factor_problem(
    rho_gap_in_eps: int,
) -> tuple[LogDetProblem, LowRankFactors, np.ndarray, np.ndarray]:
    eps = np.finfo(np.float64).eps
    scale = 2.0**24
    rho = 1.0 - rho_gap_in_eps * eps
    lam = scale * np.array([[1.0, rho], [rho, 1.0]])
    unit = math.sqrt(scale) * np.array([1.0, 1.0]) / math.sqrt(2.0)
    perturbation = unit[:, None] @ unit[None, :]
    factors = LowRankFactors(unit[:, None])
    return (
        LogDetProblem(lam, perturbation, low_rank_factors=factors),
        factors,
        lam,
        perturbation,
    )


def _non_power_of_two_near_null_factor_problem(
    rho_gap_in_eps: int,
) -> tuple[LogDetProblem, LowRankFactors, np.ndarray, np.ndarray]:
    eps = np.finfo(np.float64).eps
    scale = 1.3 * 2.0**24
    rho = 1.0 - rho_gap_in_eps * eps
    lam = scale * np.array([[1.0, rho], [rho, 1.0]])
    unit = math.sqrt(scale) * np.array([1.0, 1.0]) / math.sqrt(2.0)
    perturbation = unit[:, None] @ unit[None, :]
    factors = LowRankFactors(unit[:, None])
    return (
        LogDetProblem(lam, perturbation, low_rank_factors=factors),
        factors,
        lam,
        perturbation,
    )


def _mixed_factor_dtype_problem(
    rho_gap_in_eps: int,
) -> tuple[LogDetProblem, LowRankFactors, np.ndarray, np.ndarray]:
    dtype = np.float32
    eps = np.finfo(dtype).eps
    rho = dtype(1.0 - rho_gap_in_eps * eps)
    lam = np.eye(8, dtype=dtype) * dtype(4.0)
    lam[:2, :2] = dtype(4.0) * np.array(
        [[1.0, rho], [rho, 1.0]], dtype=dtype
    )
    perturbation = np.zeros((8, 8), dtype=dtype)
    perturbation[:2, :2] = dtype(2.8)
    left = np.zeros((8, 1), dtype=np.float64)
    right = np.zeros((8, 1), dtype=np.float64)
    left[:2, 0] = 1.0
    right[:2, 0] = float(dtype(2.8))
    factors = LowRankFactors(left, right)
    return (
        LogDetProblem(lam, perturbation, low_rank_factors=factors),
        factors,
        lam,
        perturbation,
    )


def test_authoritative_perturbation_rejects_roundoff_outside_factor_spans():
    """A floating product can contain a logdet-relevant off-span direction."""
    rng = np.random.default_rng(192)
    left = 1.0e4 * rng.normal(size=(3, 2))
    perturbation = left @ left.T
    lam = 5.0e-8 * np.eye(3)
    sigma = lam + perturbation
    factors = LowRankFactors(left)
    problem = LogDetProblem(lam, perturbation, low_rank_factors=factors)
    config = _factor_only_config()

    oracle = _decimal_three_by_three_logdet(sigma)
    assert oracle == pytest.approx(23.325967986165086, rel=0.0, abs=2.0e-15)
    assert np.linalg.cond(sigma) > 3.0e15

    verdict = check_logdet_premises(problem, config=config)[1]
    assert verdict.satisfied is False
    assert verdict.details["rank_evidence_valid"] is False
    assert verdict.details["factor_projection_eta"] > 1.0
    assert math.isinf(verdict.details["factor_log_error_bound"])
    assert verdict.details["factor_log_error_ceiling"] == math.sqrt(
        np.finfo(float).eps
    )

    for direct in (low_rank_logdet, finite_perturbation_logdet):
        with pytest.raises(ValueError, match="factor.*projection.*logdet"):
            direct(lam, perturbation, factors=factors)
    with pytest.raises(ResamplingRefused):
        dispatch_logdet(problem, config=config)


def test_maximum_finite_power_of_two_gauge_is_balanced_exactly(monkeypatch):
    """A 2**1023/2**-1023 gauge must not overflow QR or the reduced lemma."""
    from bayesmith.marginal import _logdet_eager as eager

    lam_diagonal = np.array([2.0, 3.0, 4.0, 5.0])
    lam = np.diag(lam_diagonal)
    basis = np.zeros((4, 2))
    basis[0, 0] = 0.5
    basis[1, 1] = 0.5
    perturbation = basis @ basis.T
    gauge = np.array([2.0**1023, 2.0**-1023])
    left = basis * gauge
    right = basis / gauge
    factors = LowRankFactors(left, right)
    problem = LogDetProblem(lam, perturbation, low_rank_factors=factors)
    config = LadderConfig(low_rank_fraction=1.0)
    analytic = math.fsum(
        math.log(float(value))
        for value in lam_diagonal + np.diag(perturbation)
    )

    assert np.all(np.isfinite(left))
    assert np.all(np.isfinite(right))
    assert np.array_equal(left @ right.T, perturbation)

    def refuse_generic_dense_solve(*args, **kwargs):
        del args, kwargs
        raise AssertionError("an exactly diagonal Lambda must use element division")

    def refuse_full_sigma_cholesky(matrix):
        del matrix
        raise AssertionError("the determinant lemma may not factor full Sigma")

    monkeypatch.setattr(eager, "dense_cholesky_logdet", refuse_full_sigma_cholesky)

    verdict = check_logdet_premises(problem, config=config)[1]
    with monkeypatch.context() as solve_guard:
        solve_guard.setattr(np.linalg, "solve", refuse_generic_dense_solve)
        direct = low_rank_logdet(lam, perturbation, factors=factors)
    result = dispatch_logdet(problem, config=config)

    assert verdict.satisfied is True
    assert verdict.details["factor_projection_eta"] < 1.0e-14
    assert verdict.details["factor_log_error_bound"] < verdict.details[
        "factor_log_error_ceiling"
    ]
    assert result.level == 1
    assert direct == pytest.approx(analytic, rel=0.0, abs=2.0e-14)
    assert result.value == direct


def test_rank_deficient_k_greater_than_n_and_one_sided_zero_columns_are_exact():
    """Zero columns need an exact branch, not a numerical-rank threshold."""
    u = np.array([0.03, -0.02, 0.01])
    v = np.array([0.01, 0.04, -0.03])
    zero = np.zeros(3)
    huge = np.array([np.finfo(float).max, 0.0, 0.0])
    left = np.column_stack((u, u, zero, huge, v))
    right = np.column_stack((u, u, huge, zero, v))
    perturbation = left @ right.T
    lam = np.diag([2.0, 3.0, 4.0])
    factors = LowRankFactors(left, right)
    problem = LogDetProblem(lam, perturbation, low_rank_factors=factors)
    config = _factor_only_config(finite_max_rank=5)
    sign, oracle = np.linalg.slogdet(lam + perturbation)

    assert sign == 1.0
    assert factors.rank_bound == 5 > lam.shape[0]
    verdicts = check_logdet_premises(problem, config=config)
    direct = finite_perturbation_logdet(lam, perturbation, factors=factors)
    result = dispatch_logdet(problem, config=config)

    assert verdicts[1].satisfied is False
    assert verdicts[1].details["rank_evidence_valid"] is True
    assert verdicts[5].satisfied is True
    assert result.level == 5
    assert direct == pytest.approx(oracle, rel=0.0, abs=2.0e-14)
    assert result.value == direct


def test_default_rank_one_hundred_projection_certificate_keeps_level_five():
    """The log-error certificate must retain the accepted B5 large-rank case."""
    n, rank, rho = 300, 100, 1.0e5
    widths = np.linspace(1.3, 2.3, n)
    lambda_diagonal = widths**2
    lam = np.diag(lambda_diagonal)
    basis = np.zeros((n, rank))
    basis[np.arange(rank), np.arange(rank)] = math.sqrt(rho) * widths[:rank]
    basis[200, 0] = 0.5 * math.sqrt(rho) * widths[200]
    gauge = np.power(2.0, (np.arange(rank) % 7) - 3)
    left = basis * gauge
    right = basis / gauge
    perturbation = left @ right.T
    factors = LowRankFactors(left, right)
    problem = LogDetProblem(lam, perturbation, low_rank_factors=factors)

    verdicts = check_logdet_premises(problem)
    certificate = verdicts[5].details
    result = dispatch_logdet(problem)
    sign, oracle = np.linalg.slogdet(lam + perturbation)

    assert sign == 1.0
    assert certificate["rank_evidence_valid"] is True
    assert certificate["factor_projection_eta"] == pytest.approx(
        3.680298160417677e-11, rel=2.0e-3
    )
    assert certificate["factor_log_error_bound"] < certificate[
        "factor_log_error_ceiling"
    ]
    assert verdicts[5].satisfied is True
    assert result.level == 5
    assert result.value == pytest.approx(oracle, rel=2.0e-12)


def test_exact_symmetric_kronecker_payload_preserves_minimum_subnormal_entries():
    """Averaging an already symmetric matrix can underflow its off-diagonal."""
    minimum_subnormal = np.nextafter(0.0, 1.0)
    factor = np.array(
        [[2.0, minimum_subnormal], [minimum_subnormal, 3.0]]
    )
    singleton = np.ones((1, 1))
    sigma = np.kron(factor, singleton)
    lam = 0.5 * np.eye(2)
    problem = LogDetProblem(
        lam,
        sigma - lam,
        structure_kind="kronecker",
        structure=KroneckerStructure((factor, singleton)),
    )

    verdict = check_logdet_premises(problem)[3]
    result = dispatch_logdet(problem)

    assert np.array_equal(sigma, sigma.T)
    assert verdict.satisfied is True
    assert result.level == 3
    assert result.value == pytest.approx(math.log(6.0), rel=0.0, abs=2.0e-15)


@pytest.mark.parametrize(
    ("input_dtype", "canonical_dtype"),
    [(np.float16, np.float32), (np.longdouble, np.float64)],
)
def test_noncanonical_float_kronecker_is_promoted_before_payload_linalg(
    input_dtype, canonical_dtype
):
    """Unsupported eigvalsh/cholesky dtypes must not leak TypeError."""
    factor = np.array([[2.0, 0.125], [0.125, 3.0]], dtype=input_dtype)
    singleton = np.ones((1, 1), dtype=input_dtype)
    sigma = np.kron(factor, singleton)
    lam = np.asarray(0.5 * np.eye(2), dtype=input_dtype)
    problem = LogDetProblem(
        lam,
        sigma - lam,
        structure_kind="kronecker",
        structure=KroneckerStructure((factor, singleton)),
    )
    sign, oracle = np.linalg.slogdet(sigma.astype(canonical_dtype))

    assert sign == 1.0
    assert problem.lambda_matrix.dtype.type is canonical_dtype
    assert problem.perturbation.dtype.type is canonical_dtype
    assert all(
        candidate.dtype.type is canonical_dtype
        for candidate in problem.structure.factors
    )
    verdict = check_logdet_premises(problem)[3]
    direct = low_rank_logdet(lam, sigma - lam)
    result = dispatch_logdet(problem)

    assert verdict.satisfied is True
    assert direct == pytest.approx(oracle, rel=0.0, abs=2.0e-6)
    assert result.level == 3
    assert result.value == pytest.approx(oracle, rel=0.0, abs=2.0e-6)


def test_invalid_dense_factor_payload_cannot_use_rank_to_bypass_finite_size(
    monkeypatch,
):
    """A rejected lemma payload must not turn rank into a dense complexity pass."""
    from bayesmith.marginal import _logdet_eager as eager

    n = 257
    left = np.full((n, 1), 0.01)
    right = left.copy()
    right[0, 0] += 5.0e-12
    perturbation = left @ right.T
    lam = np.eye(n)
    problem = LogDetProblem(
        lam,
        perturbation,
        low_rank_factors=LowRankFactors(left, right),
    )
    config = LadderConfig(
        low_rank_max=0,
        low_rank_fraction=1.0,
        dense_max_n=0,
        finite_max_n=0,
        finite_max_rank=1,
    )

    verdict = check_logdet_premises(problem, config=config)[5]
    assert verdict.details["rank_evidence_valid"] is True
    assert verdict.details["determinant_lemma_payload"] is False
    assert verdict.satisfied is False

    def refuse_dense_payload(matrix):
        del matrix
        raise AssertionError("invalid factor evidence may not trigger dense Cholesky")

    monkeypatch.setattr(eager, "dense_cholesky_logdet", refuse_dense_payload)
    with pytest.raises(ResamplingRefused):
        dispatch_logdet(problem, config=config)


@pytest.mark.parametrize(
    ("dtype", "expected_oracle"),
    [
        (np.float64, -34.945041100449046),
        (np.float32, -14.843772903947063),
    ],
)
def test_factor_payload_refuses_sigma_beyond_dtype_condition_resolution(
    dtype, expected_oracle
):
    """A reduced lemma cannot recover a stored near-null Sigma direction."""
    rho = np.nextafter(dtype(1.0), dtype(0.0), dtype=dtype)
    lam = np.array([[1.0, rho], [rho, 1.0]], dtype=dtype)
    unit = np.array([1.0, 1.0], dtype=dtype) / dtype(np.sqrt(dtype(2.0)))
    perturbation = unit[:, None] @ unit[None, :]
    factors = LowRankFactors(unit[:, None])
    sigma = lam + perturbation
    problem = LogDetProblem(lam, perturbation, low_rank_factors=factors)
    config = LadderConfig(
        low_rank_max=1,
        low_rank_fraction=1.0,
        dense_max_n=0,
        finite_max_n=0,
        finite_max_rank=0,
    )
    oracle = _decimal_two_by_two_logdet(sigma)

    assert oracle == pytest.approx(expected_oracle, rel=0.0, abs=2.0e-15)
    verdict = check_logdet_premises(problem, config=config)[1]
    assert verdict.details["condition"] > verdict.details["condition_ceiling"]
    assert verdict.satisfied is False
    assert verdict.details["determinant_lemma_payload"] is False

    for direct in (low_rank_logdet, finite_perturbation_logdet):
        with pytest.raises(ValueError):
            direct(lam, perturbation, factors=factors)
    with pytest.raises(ResamplingRefused):
        dispatch_logdet(problem, config=config)


def test_nonfinite_rho_measurement_is_a_rejected_premise_not_linalg_leak():
    """Overflow in Lambda^-1 P must remain diagnostic at every public boundary."""
    lam = np.diag([1.0e-309, 2.0])
    left = np.array([[1.0], [0.0]])
    perturbation = left @ left.T
    factors = LowRankFactors(left)
    problem = LogDetProblem(
        lam,
        perturbation,
        structure_kind="kronecker",
        low_rank_factors=factors,
    )
    config = LadderConfig(
        low_rank_max=1,
        low_rank_fraction=1.0,
        dense_max_n=0,
        finite_max_n=0,
        finite_max_rank=0,
    )

    with pytest.raises(ValueError, match="spectral-radius.*finite"):
        spectral_radius(lam, perturbation)
    with pytest.raises(ValueError):
        low_rank_logdet(lam, perturbation, factors=factors)

    verdicts = check_logdet_premises(problem, config=config)
    for level in (1, 5, 6, 7):
        details = verdicts[level].details
        assert details["rho_measurement_valid"] is False
        assert "spectral-radius" in details["rho_measurement_reason"]
        assert math.isinf(details["measured_rho"])
    assert verdicts[1].satisfied is False
    assert verdicts[5].satisfied is False
    assert verdicts[6].satisfied is False
    assert verdicts[7].satisfied is False
    with pytest.raises(ResamplingRefused):
        dispatch_logdet(problem, config=config)


def test_factor_certificate_includes_lambda_plus_perturbation_addition_roundoff():
    """A resolved Sigma can still differ materially from exact Lambda + P."""
    problem, factors, lam, perturbation = _scaled_near_null_factor_problem(3)
    config = LadderConfig(
        low_rank_max=1,
        low_rank_fraction=1.0,
        dense_max_n=0,
        finite_max_n=0,
        finite_max_rank=0,
    )
    sigma = lam + perturbation
    oracle = _decimal_two_by_two_logdet(sigma)
    verdict = check_logdet_premises(problem, config=config)[1]

    assert oracle == pytest.approx(-0.28768207245178123, rel=0.0, abs=2.0e-16)
    assert verdict.details["condition"] < verdict.details["condition_ceiling"]
    assert verdict.details["factor_projection_eta"] == pytest.approx(0.25)
    assert verdict.details["factor_log_error_bound"] == pytest.approx(
        0.5753641449035618
    )
    assert verdict.satisfied is False
    with pytest.raises(ValueError, match="projection.*logdet"):
        low_rank_logdet(lam, perturbation, factors=factors)
    with pytest.raises(ResamplingRefused):
        dispatch_logdet(problem, config=config)


@pytest.mark.parametrize(
    ("position", "rho_gap_in_eps", "accepted"),
    [
        ("above", 2**27 - 3, False),
        ("at", 2**27 - 1, False),
        ("below", 2**31, True),
    ],
)
def test_addition_roundoff_certificate_boundary_grid(
    position, rho_gap_in_eps, accepted
):
    """T-delta, T, T+delta pin the stronger log-error certificate boundary."""
    problem, factors, lam, perturbation = _scaled_near_null_factor_problem(
        rho_gap_in_eps
    )
    config = LadderConfig(
        low_rank_max=1,
        low_rank_fraction=1.0,
        dense_max_n=0,
        finite_max_n=0,
        finite_max_rank=0,
    )
    sigma = lam + perturbation
    oracle = _decimal_two_by_two_logdet(sigma)
    verdict = check_logdet_premises(problem, config=config)[1]
    bound = verdict.details["factor_log_error_bound"]
    ceiling = verdict.details["factor_log_error_ceiling"]

    assert verdict.details["condition"] < verdict.details["condition_ceiling"]
    if position == "above":
        assert bound > ceiling
    elif position == "at":
        assert bound == pytest.approx(ceiling, rel=1.0e-15, abs=0.0)
        assert bound > ceiling
    else:
        assert bound < ceiling
    assert verdict.satisfied is accepted

    if accepted:
        assert verdict.details["factor_total_log_error_bound"] <= ceiling
        direct = low_rank_logdet(lam, perturbation, factors=factors)
        result = dispatch_logdet(problem, config=config)
        assert result.level == 1
        assert direct == result.value
        assert abs(direct - oracle) <= ceiling
    else:
        with pytest.raises(ValueError, match="projection.*logdet"):
            low_rank_logdet(lam, perturbation, factors=factors)
        with pytest.raises(ResamplingRefused):
            dispatch_logdet(problem, config=config)


def test_overflowing_sigma_sum_is_rejected_at_all_public_boundaries():
    """Finite Lambda and P must not leak FPE when their stored sum overflows."""
    lam = np.diag([1.0e308, 1.0e308])
    left = np.array([[1.0e154], [0.0]])
    perturbation = left @ left.T
    factors = LowRankFactors(left)
    problem = LogDetProblem(
        lam,
        perturbation,
        structure_kind="kronecker",
        low_rank_factors=factors,
    )
    config = LadderConfig(
        low_rank_max=1,
        low_rank_fraction=1.0,
        dense_max_n=0,
        finite_max_n=0,
        finite_max_rank=0,
    )

    assert np.all(np.isfinite(lam))
    assert np.all(np.isfinite(perturbation))
    for direct in (low_rank_logdet, finite_perturbation_logdet):
        with pytest.raises(ValueError, match=r"Lambda \+ perturbation.*addition"):
            direct(lam, perturbation, factors=factors)

    with np.errstate(over="raise", invalid="raise"):
        verdicts = check_logdet_premises(problem, config=config)
    for level in (1, 5):
        details = verdicts[level].details
        assert details["sigma_formation_valid"] is False
        assert "Lambda + perturbation" in details["sigma_formation_reason"]
        assert details["rank_evidence_valid"] is False
        assert verdicts[level].satisfied is False
    with np.errstate(over="raise", invalid="raise"), pytest.raises(
        ResamplingRefused
    ):
        dispatch_logdet(problem, config=config)

    compact_lam = np.array([1.0e308, 1.0e308])
    compact_perturbation = np.array([1.0e308, 0.0])
    compact_problem = LogDetProblem(compact_lam, compact_perturbation)
    for direct in (low_rank_logdet, finite_perturbation_logdet):
        with pytest.raises(ValueError, match=r"Lambda \+ perturbation.*addition"):
            direct(compact_lam, compact_perturbation)
    with np.errstate(over="raise", invalid="raise"):
        compact_verdicts = check_logdet_premises(compact_problem, config=config)
    assert not any(verdict.satisfied for verdict in compact_verdicts)
    with np.errstate(over="raise", invalid="raise"), pytest.raises(
        ResamplingRefused
    ):
        dispatch_logdet(compact_problem, config=config)


@pytest.mark.parametrize(
    ("rho_gap_in_eps", "expected_oracle", "accepted"),
    [
        (3, -0.025317807984290126, False),
        (9, 1.0732944806838192, False),
        (2**26, 16.872578782662348, False),
        (2**28, 18.258873131746686, False),
        (2**31, 20.3383145336326, True),
    ],
)
def test_dense_lambda_base_arithmetic_has_a_strict_solve_logdet_certificate(
    rho_gap_in_eps, expected_oracle, accepted
):
    """Projection accuracy cannot certify an unresolved Lambda factorization."""
    problem, factors, lam, perturbation = (
        _non_power_of_two_near_null_factor_problem(rho_gap_in_eps)
    )
    config = LadderConfig(
        low_rank_max=1,
        low_rank_fraction=1.0,
        dense_max_n=0,
        finite_max_n=0,
        finite_max_rank=0,
    )
    oracle = _decimal_two_by_two_logdet(lam + perturbation)
    verdict = check_logdet_premises(problem, config=config)[1]
    details = verdict.details

    assert oracle == pytest.approx(expected_oracle, rel=0.0, abs=2.0e-15)
    assert details["condition"] < details["condition_ceiling"]
    assert details["factor_base_condition_ceiling"] == pytest.approx(
        1.0 / math.sqrt(np.finfo(np.float64).eps)
    )
    assert details["factor_base_arithmetic_valid"] is accepted
    assert details["rank_evidence_valid"] is accepted
    assert verdict.satisfied is accepted
    if accepted:
        direct = low_rank_logdet(lam, perturbation, factors=factors)
        result = dispatch_logdet(problem, config=config)
        assert result.level == 1
        assert direct == result.value
        assert abs(direct - oracle) <= details["factor_log_error_ceiling"]
    else:
        assert (
            details["factor_base_condition"]
            >= details["factor_base_condition_ceiling"]
            or details["factor_base_log_error_bound"]
            > details["factor_log_error_ceiling"]
        )
        with pytest.raises(ValueError, match="base.*condition"):
            low_rank_logdet(lam, perturbation, factors=factors)
        with pytest.raises(ResamplingRefused):
            dispatch_logdet(problem, config=config)


@pytest.mark.parametrize(
    ("position", "rho_gap_in_eps", "rank_valid", "accepted", "rejection"),
    [
        ("below", 2, True, True, None),
        ("at", 3, False, False, "projection.*logdet"),
        ("above", 4, True, True, None),
    ],
)
def test_mixed_factor_dtype_certificate_uses_stored_matrix_target_dtype(
    position, rho_gap_in_eps, rank_valid, accepted, rejection
):
    """T-delta/T/T+delta stay tied to float32 Lambda/P, not float64 factors."""
    problem, factors, lam, perturbation = _mixed_factor_dtype_problem(
        rho_gap_in_eps
    )
    config = LadderConfig(
        low_rank_max=1,
        low_rank_fraction=1.0,
        dense_max_n=0,
        finite_max_n=0,
        finite_max_rank=0,
    )
    sigma = lam + perturbation
    oracle = _decimal_two_by_two_logdet(sigma[:2, :2]) + 6.0 * math.log(4.0)
    verdict = check_logdet_premises(problem, config=config)[1]
    details = verdict.details

    assert details["factor_log_error_ceiling"] == pytest.approx(
        math.sqrt(np.finfo(np.float32).eps)
    )
    assert details["factor_base_arithmetic_valid"] is True
    if position == "at":
        assert details["factor_log_error_bound"] > details[
            "factor_log_error_ceiling"
        ]
    else:
        assert details["factor_log_error_bound"] < details[
            "factor_log_error_ceiling"
        ]
    assert details["rank_evidence_valid"] is rank_valid
    assert verdict.satisfied is accepted

    if accepted:
        direct = low_rank_logdet(lam, perturbation, factors=factors)
        result = dispatch_logdet(problem, config=config)
        assert result.level == 1
        assert direct == result.value
        assert abs(direct - oracle) <= details["factor_log_error_ceiling"]
    else:
        with pytest.raises(ValueError, match=rejection):
            low_rank_logdet(lam, perturbation, factors=factors)
        with pytest.raises(ResamplingRefused):
            dispatch_logdet(problem, config=config)


def test_diagonal_lambda_rejects_near_singular_reduced_lemma_arithmetic():
    """A stable diagonal base does not certify cancellation in I + correction."""
    loading = np.array(
        [
            float.fromhex("0x1.90beee1395683p+0"),
            float.fromhex("0x1.91ec5c932f540p+0"),
        ]
    )
    scale = float.fromhex("0x1.a0a50a1d01a62p-3")
    left = loading[:, None]
    right = (-scale * loading)[:, None]
    perturbation = left @ right.T
    lam = np.eye(2)
    sigma = lam + perturbation
    factors = LowRankFactors(left, right)
    problem = LogDetProblem(lam, perturbation, low_rank_factors=factors)
    config = LadderConfig(
        low_rank_max=1,
        low_rank_fraction=1.0,
        dense_max_n=0,
        finite_max_n=0,
        finite_max_rank=0,
    )
    oracle = _decimal_two_by_two_logdet(sigma)
    verdict = check_logdet_premises(problem, config=config)[1]
    details = verdict.details

    assert oracle == pytest.approx(-34.94450169735032, rel=0.0, abs=2.0e-14)
    assert details["condition"] < details["condition_ceiling"]
    assert details["factor_projection_eta"] == 0.0
    assert details["factor_log_error_bound"] == 0.0
    assert details["factor_reduced_log_error_bound"] > details[
        "factor_log_error_ceiling"
    ]
    assert details["factor_total_log_error_bound"] > details[
        "factor_log_error_ceiling"
    ]
    assert details["rank_evidence_valid"] is False
    assert verdict.satisfied is False
    for direct in (low_rank_logdet, finite_perturbation_logdet):
        with pytest.raises(ValueError, match="reduced.*arithmetic"):
            direct(lam, perturbation, factors=factors)
    with pytest.raises(ResamplingRefused):
        dispatch_logdet(problem, config=config)


def test_diagonal_lambda_rejects_nonnormal_near_singular_reduced_matrix():
    """A nonnormal reduced matrix needs its own arithmetic error certificate."""
    first_diagonal = 46313081.06922824
    off_diagonal = 42615587.450995795
    second_diagonal = 39213290.324580334
    lam = np.diag([first_diagonal, second_diagonal])
    perturbation = np.array(
        [[0.0, off_diagonal], [off_diagonal, 0.0]]
    )
    sigma = lam + perturbation
    factors = LowRankFactors(np.eye(2), perturbation.T)
    problem = LogDetProblem(lam, perturbation, low_rank_factors=factors)
    config = LadderConfig(
        low_rank_max=2,
        low_rank_fraction=1.0,
        dense_max_n=0,
        finite_max_n=0,
        finite_max_rank=0,
    )
    oracle = _decimal_two_by_two_logdet(sigma)
    verdict = check_logdet_premises(problem, config=config)[1]
    details = verdict.details

    assert oracle == pytest.approx(0.7115561335016318, rel=0.0, abs=2.0e-15)
    assert details["condition"] < details["condition_ceiling"]
    assert details["factor_projection_eta"] == 0.0
    assert details["factor_reduced_log_error_bound"] > details[
        "factor_log_error_ceiling"
    ]
    assert details["rank_evidence_valid"] is False
    assert verdict.satisfied is False
    for direct in (low_rank_logdet, finite_perturbation_logdet):
        with pytest.raises(ValueError, match="reduced.*arithmetic"):
            direct(lam, perturbation, factors=factors)
    with pytest.raises(ResamplingRefused):
        dispatch_logdet(problem, config=config)


def test_finite_rung_certifies_the_exact_symmetrized_payload_rho():
    """Rung 5 must measure the perturbation that dispatch actually executes."""
    lam = np.eye(2)
    diagonal = np.nextafter(1.0, 0.0)
    perturbation = np.array(
        [[diagonal, 1.0e-14], [0.0, diagonal]]
    )
    sigma = lam + perturbation
    symmetric_sigma = sigma / 2.0 + sigma.T / 2.0
    finite_payload_perturbation = symmetric_sigma - lam
    problem = LogDetProblem(lam, perturbation)
    config = LadderConfig(
        low_rank_max=0,
        dense_max_n=0,
        finite_max_n=2,
        finite_max_rank=0,
    )

    assert spectral_radius(lam, perturbation) < 1.0
    assert spectral_radius(lam, finite_payload_perturbation) > 1.0
    verdict = check_logdet_premises(problem, config=config)[5]

    assert verdict.details["measured_rho"] < 1.0
    assert verdict.details["finite_payload_rho"] > 1.0
    assert verdict.details["finite_payload_rho_measurement_valid"] is True
    assert verdict.details["finite_polynomial_stable"] is False
    assert verdict.satisfied is False
    with pytest.raises(ResamplingRefused):
        dispatch_logdet(problem, config=config)


def test_exact_symmetric_minimum_subnormal_spd_is_not_averaged_to_zero():
    """Exact symmetry preserves a condition-one SPD matrix at the dtype floor."""
    minimum_subnormal = np.nextafter(0.0, 1.0)
    lam = np.diag([minimum_subnormal, minimum_subnormal])
    problem = LogDetProblem(lam, np.zeros_like(lam))
    analytic = 2.0 * math.log(minimum_subnormal)

    direct = lambda_logdet(lam)
    verdict = check_logdet_premises(problem)[0]
    result = dispatch_logdet(problem)

    assert math.isfinite(direct)
    assert direct == pytest.approx(analytic, rel=0.0, abs=3.0e-13)
    assert verdict.details["condition"] == 1.0
    assert verdict.satisfied is True
    assert result.level == 0
    assert result.value == direct


def test_extreme_finite_asymmetry_does_not_overflow_symmetry_predicates():
    """Finite opposite-sign entries are nonsymmetric, not an allclose FPE."""
    lam = 1.7e308 * np.eye(2)
    perturbation = np.array([[0.0, 1.0e308], [-1.0e308, 0.0]])
    sigma = lam + perturbation
    problem = LogDetProblem(
        lam,
        perturbation,
        frozen_probes=FrozenProbes([[1.0, 1.0]]),
        trace_order=0,
        certified_rho=0.6,
    )
    config = LadderConfig(
        low_rank_max=0,
        dense_max_n=0,
        finite_max_n=0,
        finite_max_rank=0,
    )

    assert spectral_radius(lam, perturbation) < 0.6
    with np.errstate(over="raise", invalid="raise"):
        verdicts = check_logdet_premises(problem, config=config)
        result = dispatch_logdet(problem, config=config)
        with pytest.raises(ValueError, match="symmetric positive definite"):
            lambda_logdet(sigma)
        with pytest.raises(ValueError, match="symmetric matrix"):
            state_space_logdet(sigma, block_size=1)

    assert verdicts[7].satisfied is True
    assert result.level == 7
    assert math.isfinite(result.value)
