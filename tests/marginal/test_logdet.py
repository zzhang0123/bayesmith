"""Independent oracles and boundary tests for the log-determinant ladder.

Every exact expected value is NumPy's dense ``slogdet`` on a matrix assembled
in this file.  Approximation bounds are scalar formulas written here rather
than calls back into the implementation.
"""

from __future__ import annotations

import dataclasses
import math
from decimal import Decimal, localcontext
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.marginal.logdet import (
    FrozenProbes,
    KroneckerStructure,
    LadderConfig,
    LogDetProblem,
    LowRankFactors,
    ResamplingRefused,
    RhoCertificate,
    audit_retained_lambda_logdet,
    audit_retained_operator_norm,
    audit_retained_power_traces,
    audit_retained_rho,
    certify_warmup_rho,
    check_logdet_premises,
    choose_trace_order,
    dense_cholesky_logdet,
    dispatch_logdet,
    finite_perturbation_logdet,
    frozen_hutchinson_trace_logdet,
    lambda_logdet,
    low_rank_logdet,
    make_frozen_trace_log_plan,
    make_trace_log_plan,
    resampled_trace_logdet,
    spectral_radius,
    state_space_logdet,
    structured_logdet,
    trace_log_tail_bound,
    truncated_trace_logdet,
    whole_trace_log_tail_bound,
)


def _oracle(matrix: np.ndarray) -> float:
    sign, value = np.linalg.slogdet(matrix)
    assert sign == 1.0
    return float(value)


def _relative(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-300)


def _exact_three_by_three_toeplitz_logdet(link: float) -> float:
    """High-precision log(1 - 2 link**2) from the exact binary64 input."""
    rational_link = Fraction.from_float(link)
    determinant = Fraction(1, 1) - 2 * rational_link**2
    assert determinant > 0
    with localcontext() as context:
        context.prec = 100
        decimal_determinant = Decimal(determinant.numerator) / Decimal(
            determinant.denominator
        )
        return float(decimal_determinant.ln())


def _exact_three_by_three_toeplitz_rho(link: float) -> float:
    """High-precision ``rho((Sigma - Lambda) Lambda^-1)`` at ``Lambda = 2 I``.

    ``Sigma`` is the 3x3 symmetric Toeplitz with unit diagonal and off-diagonal
    ``link`` that the rung-5 fixture below builds; its eigenvalues are ``1`` and
    ``1 +/- sqrt(2) link``.  ``spectral_radius`` measures ``rho(P Lambda^-1)``
    (its ``_x_matrix`` is ``solve(Lambda.T, P.T).T``), so at ``Lambda = 2 I``
    that spectrum is ``{sigma_k / 2 - 1}`` and the largest magnitude in it is
    the one the SMALLEST eigenvalue of ``Sigma`` carries, ``(1 + sqrt(2) link)
    / 2``.  Closed form because no binary64 eigensolve can resolve which side
    of one it is on: it sits 1.106 * 2**-53 below one, inside the eigensolver's
    own backward error.
    """
    rational_link = Fraction.from_float(link)
    with localcontext() as context:
        context.prec = 100
        decimal_link = Decimal(rational_link.numerator) / Decimal(
            rational_link.denominator
        )
        return float((Decimal(1) + Decimal(2).sqrt() * decimal_link) / Decimal(2))


def _exact_two_by_two_correlation_logdet(correlation: float) -> float:
    """High-precision log(1 - correlation**2) from the binary64 input."""
    rational_correlation = Fraction.from_float(correlation)
    determinant = Fraction(1, 1) - rational_correlation**2
    assert determinant > 0
    with localcontext() as context:
        context.prec = 100
        decimal_determinant = Decimal(determinant.numerator) / Decimal(
            determinant.denominator
        )
        return float(decimal_determinant.ln())


def _exact_two_by_two_scaled_spd_logdet(scale: float) -> float:
    """High-precision log(3 * scale**2) from the exact binary64 scale."""
    rational_scale = Fraction.from_float(scale)
    determinant = 3 * rational_scale**2
    assert determinant > 0
    with localcontext() as context:
        context.prec = 100
        decimal_determinant = Decimal(determinant.numerator) / Decimal(
            determinant.denominator
        )
        return float(decimal_determinant.ln())


def _exact_two_by_two_fraction_logdet(matrix: np.ndarray) -> float:
    """High-precision logdet from the exact binary64 entries of a 2x2 matrix."""
    entries = tuple(Fraction.from_float(float(value)) for value in matrix.flat)
    determinant = entries[0] * entries[3] - entries[1] * entries[2]
    assert determinant > 0
    with localcontext() as context:
        context.prec = 100
        decimal_determinant = Decimal(determinant.numerator) / Decimal(
            determinant.denominator
        )
        return float(decimal_determinant.ln())


_UNSCALED_SVD_NOT_FOOLED = (
    "THIS IS NOT A PASS. The masking the calling test is named for did not "
    "happen on this build: an unscaled np.linalg.cond read {reading!r}, at or "
    "above the 1/eps ceiling 4503599627370496.0, so a gate that skipped the "
    "exact power-of-two rescale would have refused this matrix anyway and the "
    "refusal asserted above is no longer evidence about the rescale. The true "
    "condition is 4503599627370495.2500000000000001249..., below the ceiling: "
    "the smaller eigenvalue is 1.9999999999999999 units of 2**-1074 and an SVD "
    "can only answer in whole units. Measured 2026-09-03, numpy 2.5.2 both "
    "sides: Apple Accelerate answers 2 units and reads 4503599627370495.5; "
    "scipy-openblas 0.3.34 answers 1 unit and reads 9007199254740991.0. "
    "Neither reading is resolved -- on the exactly rescaled matrix, which has "
    "the same spectrum, the same two libraries read 1.0503813809299122e+16 and "
    "8492068896701030.0 against that true 4503599627370495.25, so sigma_min "
    "carries O(1) relative error at this condition and Accelerate's agreement "
    "is the subnormal grid pinning the answer, not the solver resolving it. "
    "Rebuilding the fixture cannot rescue this: being fooled needs the "
    "REPORTED sigma_min above sigma_max*eps while the true one is below it, "
    "i.e. an error of the right sign and under half a denormal unit, which no "
    "solver delivers where its own error on sigma_min is O(1)."
)


def _exact_two_by_two_symmetric_spectrum(
    matrix: np.ndarray,
) -> tuple[Decimal, Decimal]:
    """Condition number and smallest eigenvalue of a symmetric 2x2, exactly.

    The two subnormal-scale fixtures below cannot ask LAPACK this question.
    Their smaller eigenvalue is 1.9999999999999999 units of 2**-1074, so any
    SVD must answer 1 unit or 2, and which one it answers is the installed
    library's business: measured 2026-09-03 on one numpy 2.5.2, Apple
    Accelerate answers 2 units -> cond 4503599627370495.5, scipy-openblas
    0.3.34 answers 1 -> cond 9007199254740991.0, and those straddle the 1/eps
    ceiling. Both callers asserted the Accelerate digits as a premise until
    that date and went red on Linux for a matrix nothing about which changed.

    The eigenvalues are ``((a+d) +/- sqrt((a-d)**2 + 4*b**2))/2``, the entries
    are exactly representable, and Fractions plus 100 decimal digits settle
    both bit-identically on either platform (verified on both, to every digit).
    Decimals are returned rather than floats on purpose. The true condition
    here is 4503599627370495.2500000000000001249..., and ``float()`` of it is
    4503599627370495.5 -- one ULP below the ceiling a caller compares it
    against, and byte-for-byte the Accelerate LAPACK reading this helper exists
    to stop trusting. It gets there only on that tail: bare ...495.25 is an
    exact binary64 tie and rounds the other way, to ...495.0. A Decimal
    comparison has neither the knife edge nor the confusing digits.
    """
    upper_left, upper_right, lower_left, lower_right = (
        Fraction.from_float(float(value)) for value in matrix.flat
    )
    assert upper_right == lower_left
    trace = upper_left + lower_right
    discriminant = (upper_left - lower_right) ** 2 + 4 * upper_right**2
    with localcontext() as context:
        context.prec = 100
        trace_exact = Decimal(trace.numerator) / Decimal(trace.denominator)
        gap = (
            Decimal(discriminant.numerator) / Decimal(discriminant.denominator)
        ).sqrt()
        smallest_eigenvalue = (trace_exact - gap) / 2
        largest_eigenvalue = (trace_exact + gap) / 2
        assert smallest_eigenvalue > 0
        return largest_eigenvalue / smallest_eigenvalue, smallest_eigenvalue


def _independent_power_traces(
    lam: np.ndarray, perturbation: np.ndarray, order: int
) -> tuple[float, ...]:
    """Dense NumPy traces, independent of the production trace provider."""
    x_matrix = (
        perturbation / lam
        if lam.ndim == 1
        else np.linalg.solve(lam.T, perturbation.T).T
    )
    power = np.ones_like(x_matrix) if lam.ndim == 1 else np.eye(lam.shape[0])
    traces = []
    for _ in range(order):
        power = power * x_matrix if lam.ndim == 1 else power @ x_matrix
        traces.append(float(np.sum(power) if lam.ndim == 1 else np.trace(power)))
    return tuple(traces)


def _spd_fixture(n: int = 5) -> tuple[np.ndarray, np.ndarray]:
    widths = np.linspace(1.3, 2.1, n)
    lam = np.diag(widths**2)
    grid = np.arange(1, n + 1, dtype=float)
    factor = np.column_stack((0.07 * grid, 0.03 * grid[::-1]))
    return lam, factor @ factor.T


def test_resampling_is_refused_before_it_can_make_hmc_nondeterministic():
    """Deleting the refusal must expose a fresh random estimate to HMC."""
    lam, perturbation = _spd_fixture(3)
    with pytest.raises(ResamplingRefused, match="HMC"):
        resampled_trace_logdet(lam, perturbation, order=5, probes=8)


def test_trace_log_runtime_kernel_is_jittable_and_has_the_finite_series_gradient():
    """Converting traced runtime values to NumPy breaks NUTS before sampling."""
    widths = np.array([1.3, 1.8, 2.4])
    lam = np.diag(widths**2)
    certificate = certify_warmup_rho(
        [0.37],
        margin=0.0,
        tolerance=1e-3,
        multiplicity=3,
        lambda_logdets=[lambda_logdet(lam)],
    )
    perturbation = 0.37 * lam
    traces = _independent_power_traces(lam, perturbation, certificate.order)
    problem = LogDetProblem(
        lam,
        perturbation,
        exact_power_traces=traces,
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )
    with jax.enable_x64(True):
        plan = make_trace_log_plan(problem, certificate)
        base = jnp.sum(jnp.log(jnp.asarray(widths**2)))

        def evaluate(rho):
            dynamic_traces = jnp.stack(
                [3.0 * rho**power for power in range(1, plan.order + 1)]
            )
            return plan(base, dynamic_traces)

        rho = jnp.array(0.37)
        got = jax.jit(evaluate)(rho)
        gradient = jax.jit(jax.grad(evaluate))(rho)
        want = base + 3.0 * sum(
            (-1.0) ** (power + 1) * rho**power / power
            for power in range(1, plan.order + 1)
        )
        want_gradient = 3.0 * sum(
            (-1.0) ** (power + 1) * rho ** (power - 1)
            for power in range(1, plan.order + 1)
        )
    assert got == pytest.approx(float(want), rel=2e-7)
    assert gradient == pytest.approx(float(want_gradient), rel=2e-7)


def test_frozen_runtime_kernel_is_jittable_and_differentiable():
    """The frozen estimator's matrix products must stay in JAX at runtime."""
    probes = FrozenProbes([[1.0, 1.0], [1.0, -1.0]])
    lam = np.array([1.7, 2.6])
    perturbation = 0.2 * lam
    certificate = certify_warmup_rho(
        [0.2],
        margin=0.0,
        tolerance=1e-3,
        multiplicity=2,
        lambda_logdets=[lambda_logdet(lam)],
        x_operator_norms=[0.2],
    )
    problem = LogDetProblem(
        lam,
        perturbation,
        frozen_probes=probes,
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )
    with jax.enable_x64(True):
        plan = make_frozen_trace_log_plan(problem, certificate)
        base = jnp.log(1.7) + jnp.log(2.6)

        def evaluate(rho):
            return plan(base, rho * jnp.eye(2))

        value = jax.jit(evaluate)(jnp.array(0.2))
        gradient = jax.jit(jax.grad(evaluate))(jnp.array(0.2))
    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)


def _assert_frozen_runtime_matches_eager_and_oracle(
    n: int, probes_count: int, *, compact: bool
) -> None:
    """Shared numerical assertion also used by the mutation-kill test."""
    rng = np.random.default_rng(7100 + 10 * n + probes_count + int(compact))
    widths = np.linspace(1.3, 2.3, n)
    lam_diagonal = widths**2
    if compact:
        eigenvalues = np.linspace(0.01, 0.05, n)
        lam = lam_diagonal
        perturbation = eigenvalues * lam_diagonal
        x_matrix = eigenvalues
        probe_values = rng.normal(size=(probes_count, n))
    else:
        basis, _ = np.linalg.qr(rng.normal(size=(n, n)))
        eigenvalues = np.linspace(0.01, 0.05, n)
        square_root = np.diag(widths)
        perturbation = square_root @ basis @ np.diag(eigenvalues) @ basis.T @ square_root
        lam = np.diag(lam_diagonal)
        x_matrix = np.linalg.solve(lam.T, perturbation.T).T
        probe_values = rng.choice((-1.0, 1.0), size=(probes_count, n))
    probes = FrozenProbes(probe_values)
    certificate = certify_warmup_rho(
        [0.05],
        margin=0.0,
        tolerance=1.0e-7,
        multiplicity=n,
        lambda_logdets=[lambda_logdet(lam)],
        x_operator_norms=[
            float(np.max(np.abs(x_matrix)))
            if np.ndim(x_matrix) == 1
            else float(np.linalg.norm(np.abs(x_matrix), ord=2))
        ],
    )
    problem = LogDetProblem(
        lam,
        perturbation,
        frozen_probes=probes,
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )
    with jax.enable_x64(True):
        plan = make_frozen_trace_log_plan(problem, certificate)
        compiled = jax.jit(lambda base, x_value: plan(base, x_value))
        runtime = float(
            compiled(
                jnp.asarray(lambda_logdet(lam)), jnp.asarray(x_matrix)
            )
        )
    eager = frozen_hutchinson_trace_logdet(
        lam,
        perturbation,
        probes,
        order=certificate.order,
        rho=certificate.certified_rho,
    )
    oracle = _oracle(np.diag(lam + perturbation) if compact else lam + perturbation)
    assert runtime == pytest.approx(eager, rel=2e-13, abs=2e-13)
    assert _relative(runtime, oracle) < 5e-2


@pytest.mark.parametrize("n", [2, 10, 100])
def test_frozen_runtime_compact_diagonal_matches_eager_and_slogdet(n):
    """The JAX kernel must use elementwise action for compact diagonal X."""
    _assert_frozen_runtime_matches_eager_and_oracle(
        n, max(1, n // 8), compact=True
    )


@pytest.mark.parametrize("n", [2, 10, 100])
@pytest.mark.parametrize("probe_fraction", ["one", "eighth", "full"])
def test_frozen_runtime_probe_count_grid_matches_eager_and_slogdet(
    n, probe_fraction
):
    """Rung 7 is evaluated away from the p=n Hadamard exact-trace corner."""
    probes_count = {
        "one": 1,
        "eighth": max(1, n // 8),
        "full": n,
    }[probe_fraction]
    _assert_frozen_runtime_matches_eager_and_oracle(
        n, probes_count, compact=False
    )


def test_frozen_runtime_oracle_assertion_kills_a_constant_kernel(monkeypatch):
    """The numerical oracle, unlike an isfinite assertion, kills this mutation."""
    from bayesmith.marginal import _logdet_runtime as runtime

    monkeypatch.setattr(
        runtime,
        "frozen_hutchinson_trace_logdet",
        lambda *args, **kwargs: jnp.asarray(1.0),
    )
    with pytest.raises(AssertionError):
        _assert_frozen_runtime_matches_eager_and_oracle(10, 1, compact=False)


def test_validated_runtime_plans_capture_order_and_frozen_probes():
    """Runtime callers cannot lower order or swap/redraw probes per evaluation."""
    import bayesmith.marginal.logdet as module

    certificate = certify_warmup_rho(
        [0.2],
        margin=0.05,
        tolerance=1e-4,
        multiplicity=2,
        lambda_logdets=[math.log(1.7) + math.log(2.6)],
        x_operator_norms=[0.2],
    )
    lam = np.array([1.7, 2.6])
    perturbation = 0.2 * lam
    traces = _independent_power_traces(
        lam, perturbation, certificate.order
    )
    trace_problem = LogDetProblem(
        lam,
        perturbation,
        exact_power_traces=traces,
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )
    with jax.enable_x64(True):
        trace_plan = make_trace_log_plan(trace_problem, certificate)
        base = jnp.sum(jnp.log(jnp.asarray(lam)))

        def trace_runtime(rho):
            dynamic = jnp.stack(
                [2.0 * rho**power for power in range(1, trace_plan.order + 1)]
            )
            return trace_plan(base, dynamic)

        assert jnp.isfinite(jax.jit(trace_runtime)(jnp.array(0.2)))
        assert jnp.isfinite(jax.jit(jax.grad(trace_runtime))(jnp.array(0.2)))
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            trace_plan._order = 1
        with pytest.raises(TypeError):
            trace_plan(base, jnp.asarray(traces), order=1)
        with pytest.raises(ValueError, match="at least.*certified order"):
            trace_plan(base, jnp.asarray(traces[: trace_plan.order - 1]))
        with pytest.raises(ValueError, match="scalar"):
            trace_plan(jnp.asarray([base, base]), jnp.asarray(traces))
        with pytest.raises(ValueError, match="real floating"):
            trace_plan(base, jnp.asarray(traces, dtype=jnp.complex128))

        source = np.array([[1.0, 1.0], [1.0, -1.0]])
        probes = FrozenProbes(source)
        frozen_problem = LogDetProblem(
            lam,
            perturbation,
            frozen_probes=probes,
            trace_order=certificate.order,
            certified_rho=certificate.certified_rho,
        )
        frozen_plan = make_frozen_trace_log_plan(frozen_problem, certificate)

        def frozen_runtime(rho):
            return frozen_plan(base, rho * jnp.eye(2))

        first = jax.jit(frozen_runtime)(jnp.array(0.2))
        source[0, 0] = -99.0
        public = probes.values
        with pytest.raises(ValueError):
            public.setflags(write=True)
        second = jax.jit(frozen_runtime)(jnp.array(0.2))
        assert float(first) == float(second)
        assert jnp.isfinite(jax.jit(jax.grad(frozen_runtime))(jnp.array(0.2)))
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            frozen_plan._probes = FrozenProbes([[1.0, 1.0], [1.0, -1.0]])
        with pytest.raises(TypeError):
            frozen_plan(base, 0.2 * jnp.eye(2), probes=FrozenProbes([[1.0, 1.0]]))
        with pytest.raises(ValueError, match=r"compact \(n,\).*dense \(n, n\)"):
            frozen_plan(base, 0.2 * jnp.eye(3))
        with pytest.raises(ValueError, match="real floating"):
            frozen_plan(base, 0.2 * jnp.eye(2, dtype=jnp.complex128))

    assert "truncated_trace_logdet_runtime" not in module.__all__
    assert "frozen_hutchinson_trace_logdet_runtime" not in module.__all__


@pytest.mark.parametrize("method", [lambda_logdet, dense_cholesky_logdet])
def test_base_and_dense_exact_methods_match_an_independent_slogdet(method):
    """A missing Lambda term or half-logdet convention changes this result."""
    lam = np.diag(np.array([1.3, 1.8, 2.4]) ** 2)
    assert method(lam) == pytest.approx(_oracle(lam), rel=1e-14, abs=1e-14)


@pytest.mark.parametrize("method", [lambda_logdet, dense_cholesky_logdet])
def test_dense_cholesky_scales_a_well_conditioned_subnormal_spd_matrix(method):
    """Raw Cholesky rounds det=3m^2 to 4m^2 even though cond(Sigma)=3."""
    smallest = float(np.nextafter(0.0, 1.0))
    matrix = smallest * np.array([[2.0, 1.0], [1.0, 2.0]])
    oracle = _exact_two_by_two_scaled_spd_logdet(smallest)

    assert np.linalg.cond(matrix) == pytest.approx(3.0)
    assert method(matrix) == pytest.approx(oracle, rel=0.0, abs=5.0e-13)


def test_level_zero_dispatch_scales_a_well_conditioned_subnormal_lambda():
    """The Lambda-itself rung must inherit the scaled dense Cholesky arithmetic."""
    smallest = float(np.nextafter(0.0, 1.0))
    matrix = smallest * np.array([[2.0, 1.0], [1.0, 2.0]])
    oracle = _exact_two_by_two_scaled_spd_logdet(smallest)
    problem = LogDetProblem(matrix, np.zeros_like(matrix))

    result = dispatch_logdet(problem)

    assert result.level == 0
    assert result.value == pytest.approx(oracle, rel=0.0, abs=5.0e-13)


def test_level_four_dispatch_scales_a_well_conditioned_subnormal_sigma():
    """The dense rung must not silently return log(4m^2) for det(Sigma)=3m^2."""
    smallest = float(np.nextafter(0.0, 1.0))
    matrix = smallest * np.array([[2.0, 1.0], [1.0, 2.0]])
    lam = smallest * np.eye(2)
    oracle = _exact_two_by_two_scaled_spd_logdet(smallest)
    problem = LogDetProblem(lam, matrix - lam)

    result = dispatch_logdet(problem)

    assert result.level == 4
    assert result.value == pytest.approx(oracle, rel=0.0, abs=5.0e-13)


def test_low_rank_and_finite_routes_share_the_same_newton_arithmetic_bitwise():
    """A duplicate determinant-lemma implementation breaks bitwise identity."""
    lam, perturbation = _spd_fixture(7)
    low_rank = low_rank_logdet(lam, perturbation)
    finite = finite_perturbation_logdet(lam, perturbation)
    assert low_rank == finite
    assert low_rank == pytest.approx(_oracle(lam + perturbation), rel=2e-13)


def test_low_rank_exactness_never_uses_scale_dependent_numerical_rank():
    """A tiny P direction can become order-one after multiplication by Lambda^-1."""
    lam = np.array([1.0e-20, 2.3])
    perturbation = np.array([1.0e-20, 0.0])
    got = low_rank_logdet(lam, perturbation)
    want = float(np.sum(np.log(lam + perturbation)))
    assert got == pytest.approx(want, rel=2e-14, abs=2e-14)


def test_dense_low_rank_premise_requires_factors_and_uses_their_algebraic_rank():
    """Matrix-rank tolerance is not proof that Newton identities terminate."""
    lam = np.diag(np.linspace(1.3, 2.4, 8) ** 2)
    left = np.column_stack((np.linspace(0.02, 0.09, 8), np.linspace(0.07, 0.01, 8)))
    right = left
    perturbation = left @ left.T
    assert check_logdet_premises(LogDetProblem(lam, perturbation))[1].satisfied is False

    factors = LowRankFactors(left, right)
    problem = LogDetProblem(lam, perturbation, low_rank_factors=factors)
    verdict = check_logdet_premises(
        problem, config=LadderConfig(low_rank_fraction=0.5)
    )[1]
    assert verdict.satisfied is True
    assert verdict.details["rank"] == 2
    low_rank = low_rank_logdet(lam, perturbation, factors=factors)
    finite = finite_perturbation_logdet(lam, perturbation, factors=factors)
    assert low_rank == finite
    assert low_rank == pytest.approx(_oracle(lam + perturbation), rel=2e-13)


def test_low_rank_factor_proof_refuses_a_tiny_unrepresented_amplified_residual():
    """An allclose residual is not algebraic rank evidence after Lambda^-1."""
    lam = np.diag([0.1, 1.0e-20])
    left = np.array([[0.2], [1.0e-11]])
    factors = LowRankFactors(left)
    represented = left @ left.T
    perturbation = represented + np.diag([0.0, 5.0e-21])
    sigma = lam + perturbation
    order = 30
    traces = _independent_power_traces(lam, perturbation, order)
    x_matrix = np.linalg.solve(lam.T, perturbation.T).T
    rho = float(np.max(np.abs(np.linalg.eigvals(x_matrix))))

    problem = LogDetProblem(
        lam,
        perturbation,
        low_rank_factors=factors,
        exact_power_traces=traces,
        trace_order=order,
        certified_rho=rho,
    )
    config = LadderConfig(
        low_rank_max=1,
        low_rank_fraction=1.0,
        dense_max_n=0,
        finite_max_n=0,
        finite_max_rank=1,
    )
    verdicts = check_logdet_premises(problem, config=config)
    assert verdicts[1].satisfied is False
    assert verdicts[5].satisfied is False
    with pytest.raises(
        ValueError,
        match=r"Lambda\^-1-amplified residual.*amplified residual=0\.5",
    ):
        low_rank_logdet(lam, perturbation, factors=factors)
    with pytest.raises(
        ValueError,
        match=r"Lambda\^-1-amplified residual.*amplified residual=0\.5",
    ):
        finite_perturbation_logdet(lam, perturbation, factors=factors)

    with pytest.raises(ValueError, match="condition"):
        finite_perturbation_logdet(lam, perturbation)
    result = dispatch_logdet(problem, config=config)
    assert result.level == 6
    assert result.value == pytest.approx(_oracle(sigma), rel=2e-10)


@pytest.mark.parametrize(
    ("n", "rank"),
    [
        (20, 2),
        (50, 4),
        (12, 9),
        (37, 6),
        (101, 7),
        (200, 8),
        (300, 5),
        (64, 3),
        (260, 4),
    ],
)
def test_explicit_self_factors_alias_on_all_reproduced_blas_shapes(n, rank):
    """``LowRankFactors(L, L)`` must use one buffer and one BLAS product."""
    rng = np.random.default_rng(4000 + n + rank)
    left = 0.01 * rng.normal(size=(n, rank))
    perturbation = left @ left.T
    factors = LowRankFactors(left, left)
    lam = np.linspace(1.3, 2.3, n) ** 2
    problem = LogDetProblem(lam, np.diag(perturbation), low_rank_factors=None)
    dense_problem = LogDetProblem(
        np.diag(lam), perturbation, low_rank_factors=factors
    )

    assert factors.left is factors.right
    assert np.array_equal(factors.left @ factors.right.T, perturbation)
    assert check_logdet_premises(
        dense_problem,
        config=LadderConfig(low_rank_max=rank, low_rank_fraction=1.0),
    )[1].satisfied
    value = low_rank_logdet(np.diag(lam), perturbation, factors=factors)
    assert _relative(value, _oracle(np.diag(lam) + perturbation)) < 2e-12
    assert check_logdet_premises(problem)[1].details["rank"] == n


def test_transposed_fortran_loadings_keep_exact_rank_evidence_at_level_one():
    """Preserving K-order in the artifact changes BLAS rounding and drops rung 1."""
    n, rank = 32, 2
    left = 0.03 * np.random.default_rng(0).normal(size=(n, rank))
    perturbation = left @ left.T
    stored_loadings = left.T.copy(order="C")
    factors = LowRankFactors(stored_loadings.T)
    widths = np.linspace(1.3, 2.4, n)
    lam = np.diag(widths**2)

    result = dispatch_logdet(
        LogDetProblem(lam, perturbation, low_rank_factors=factors)
    )

    assert result.level == 1
    assert _relative(result.value, _oracle(lam + perturbation)) < 2e-13


@pytest.mark.parametrize("n", [277, 511])
@pytest.mark.parametrize(
    ("origin_order", "supplied_order"), [("C", "F"), ("F", "C")]
)
def test_exact_factor_reconstruction_recognizes_both_blas_layout_origins(
    n, origin_order, supplied_order
):
    """Canonicalizing the inverse layout must not change authoritative P."""
    rank = 11
    values = 0.03 * np.random.default_rng(0).normal(size=(n, rank))
    c_factor = np.array(values, order="C")
    f_factor = np.array(values, order="F")
    products = {"C": c_factor @ c_factor.T, "F": f_factor @ f_factor.T}
    factors_by_order = {"C": c_factor, "F": f_factor}
    perturbation = products[origin_order]
    supplied = factors_by_order[supplied_order]
    factors = LowRankFactors(supplied)
    widths = np.linspace(1.3, 2.3, n)
    lam = np.diag(widths**2)
    problem = LogDetProblem(lam, perturbation, low_rank_factors=factors)
    config = LadderConfig(low_rank_max=rank, low_rank_fraction=1.0)

    if np.array_equal(products["C"], products["F"]):
        pytest.skip(
            "THIS IS NOT A PASS. Whether C- and F-order products differ in "
            "their last bit is a property of the installed BLAS, and this one "
            "returns them bitwise equal, so there are no 'both origins' to "
            "recognise here and the cross-layout branch of "
            "_matching_factor_reconstruction is unreachable. Measured "
            "2026-09-02 with numpy 2.5.2 on both sides: Apple Accelerate "
            "separates this shape by 1 ULP at n=277 and 5 at n=511; "
            "scipy-openblas 0.3.34 separates neither, nor any of thirty "
            "randomly generated shapes. Asserting the separation instead of "
            "measuring it is what took 2091 tests down at fixture setup in a "
            "publish job, on a tag that could not be un-pushed."
        )
    verdict = check_logdet_premises(problem, config=config)[1]
    direct = low_rank_logdet(lam, perturbation, factors=factors)
    result = dispatch_logdet(problem, config=config)
    dense_oracle = _oracle(lam + perturbation)

    assert verdict.details["rank_evidence_valid"] is True
    assert verdict.satisfied is True
    assert result.level == 1
    assert direct == pytest.approx(dense_oracle, rel=2e-12)
    assert result.value == direct


def test_float32_explicit_self_factors_preserve_exact_product_evidence():
    """Equivalent copied float32 factors retain bitwise product evidence."""
    rng = np.random.default_rng(551)
    left = (0.03 * rng.normal(size=(37, 6))).astype(np.float32)
    perturbation = left @ left.T
    lam = np.diag(np.linspace(1.3, 2.2, 37, dtype=np.float32) ** 2)
    factors = LowRankFactors(left, left)
    problem = LogDetProblem(lam, perturbation, low_rank_factors=factors)

    verdict = check_logdet_premises(
        problem, config=LadderConfig(low_rank_max=6, low_rank_fraction=1.0)
    )[1]
    assert verdict.satisfied is True
    assert low_rank_logdet(lam, perturbation, factors=factors) == pytest.approx(
        _oracle(lam.astype(float) + perturbation.astype(float)), rel=3e-6
    )


def test_value_equal_mixed_dtype_factors_keep_the_explicit_right_representation():
    """Value equality must not silently demote an explicit float64 right factor."""
    left = np.array([[0.25, 0.5], [0.75, 1.0]], dtype=np.float32)
    right = left.astype(np.float64)

    factors = LowRankFactors(left, right)

    assert factors.left.dtype == np.dtype(np.float32)
    assert factors.right.dtype == np.dtype(np.float64)
    assert factors.left is not factors.right
    assert np.array_equal(factors.right, right)


def test_factorized_logdet_uses_one_promoted_computation_dtype():
    """A float32 Lambda correction cannot accompany float64 certification."""
    n, rank = 32, 4
    lambda_diagonal = np.geomspace(1.0e-4, 1.0e4, n).astype(np.float32)
    lam = np.diag(lambda_diagonal)
    basis, _ = np.linalg.qr(np.random.default_rng(42).normal(size=(n, rank)))
    eigenvalues = np.geomspace(0.1, 100.0, rank)
    left = (
        np.sqrt(lambda_diagonal.astype(np.float64))[:, None]
        * basis
        * np.sqrt(eigenvalues)
    )
    perturbation = left @ left.T
    factors = LowRankFactors(left)
    problem = LogDetProblem(lam, perturbation, low_rank_factors=factors)
    analytic = math.fsum(math.log(float(value)) for value in lambda_diagonal)
    analytic += math.fsum(math.log1p(float(value)) for value in eigenvalues)
    dense_oracle = _oracle(lam.astype(np.float64) + perturbation)

    direct = low_rank_logdet(lam, perturbation, factors=factors)
    result = dispatch_logdet(problem)

    assert result.level == 1
    assert direct == pytest.approx(analytic, rel=0.0, abs=5.0e-11)
    assert direct == pytest.approx(dense_oracle, rel=0.0, abs=5.0e-11)
    assert result.value == direct


def test_low_rank_factor_check_rejects_cancellation_inflated_roundoff_envelope():
    """Huge cancelling factors cannot certify an unrelated low-rank matrix."""
    lam = np.diag(np.array([1.3, 1.8, 2.4]) ** 2)
    perturbation = 0.5 * lam
    left = np.full((3, 2), 1.0e8)
    right = np.column_stack((np.full(3, 1.0e8), np.full(3, -1.0e8)))
    factors = LowRankFactors(left, right)
    problem = LogDetProblem(lam, perturbation, low_rank_factors=factors)

    verdict = check_logdet_premises(
        problem, config=LadderConfig(low_rank_max=2, low_rank_fraction=1.0)
    )[1]
    assert verdict.satisfied is False
    with pytest.raises(ValueError, match=r"Lambda\^-1-amplified"):
        low_rank_logdet(lam, perturbation, factors=factors)


def test_low_rank_factor_check_rejects_any_omitted_near_boundary_eigenvalue():
    """A rounding-size residual can change logdet by order one near eigenvalue -1."""
    widths = np.array([1.3, 1.8, 2.4])
    lam = np.diag(widths**2)
    basis, _ = np.linalg.qr(np.random.default_rng(123).normal(size=(3, 3)))
    eigenvalues = np.array([-1.0, 0.5])
    left = widths[:, None] * basis[:, :2] * np.sqrt(np.abs(eigenvalues))
    right = widths[:, None] * basis[:, :2] * np.sign(eigenvalues) * np.sqrt(
        np.abs(eigenvalues)
    )
    factors = LowRankFactors(left, right)
    perturbation = left @ right.T + 5.0e-16 * lam

    verdict = check_logdet_premises(
        LogDetProblem(lam, perturbation, low_rank_factors=factors),
        config=LadderConfig(low_rank_max=2, low_rank_fraction=1.0),
    )[1]
    assert verdict.satisfied is False
    with pytest.raises(ValueError, match="do not exactly reconstruct"):
        low_rank_logdet(lam, perturbation, factors=factors)


def test_factorized_exact_rungs_replace_adversarial_newton_arithmetic():
    """The D80 factorization must stay accurate where Newton identities failed."""
    n, rank = 128, 16
    lambda_diagonal = np.linspace(1.3, 2.3, n) ** 2
    lam = np.diag(lambda_diagonal)
    basis, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(n, rank)))
    eigenvalues = np.geomspace(0.1, 100.0, rank)
    left = np.sqrt(lambda_diagonal)[:, None] * basis * np.sqrt(eigenvalues)
    factors = LowRankFactors(left)
    perturbation = left @ left.T
    sigma = lam + perturbation
    problem = LogDetProblem(lam, perturbation, low_rank_factors=factors)

    verdicts = check_logdet_premises(problem)
    assert verdicts[1].satisfied is True
    low = low_rank_logdet(lam, perturbation, factors=factors)
    finite = finite_perturbation_logdet(lam, perturbation, factors=factors)
    assert low == finite
    result = dispatch_logdet(problem)
    oracle = _oracle(sigma)
    assert result.level == 1
    assert result.value == low
    assert _relative(result.value, oracle) < 2e-12


def test_finite_e_polynomial_is_stable_for_mixed_sign_spectrum_at_rho_one():
    """The factored e-polynomial must not use a cancelling Newton recurrence."""
    n = 128
    lam = np.linspace(1.3, 2.3, n) ** 2
    eigenvalues = np.linspace(-0.99, 1.0, n)
    perturbation = eigenvalues * lam
    oracle = _oracle(np.diag(lam + perturbation))

    low = low_rank_logdet(lam, perturbation)
    finite = finite_perturbation_logdet(lam, perturbation)
    assert low == finite
    assert _relative(finite, oracle) < 2e-13


def test_finite_e_polynomial_does_not_overflow_at_configurable_high_degree():
    """A finite logdet remains representable when the unfactored determinant does not."""
    n = 2_000
    lam = np.linspace(1.3, 2.3, n) ** 2
    perturbation = lam.copy()
    problem = LogDetProblem(lam, perturbation)
    config = LadderConfig(
        low_rank_max=0,
        dense_max_n=0,
        finite_max_n=0,
        finite_max_rank=n,
    )
    oracle = math.fsum(math.log(value) for value in lam + perturbation)

    assert check_logdet_premises(problem, config=config)[5].satisfied is True
    direct = finite_perturbation_logdet(lam, perturbation)
    assert _relative(direct, oracle) < 2e-13


def test_compact_finite_factorization_avoids_base_correction_cancellation():
    """A well-conditioned Sigma must not inherit cancellation from its preconditioner."""
    n = 8
    lam = np.ones(n)
    lam[0] = 1.0e15
    perturbation = np.zeros(n)
    perturbation[0] = 1.0 - lam[0]
    problem = LogDetProblem(lam, perturbation)

    assert check_logdet_premises(problem)[1].satisfied is True
    result = dispatch_logdet(problem)
    assert result.level == 1
    assert result.value == _oracle(np.diag(lam + perturbation)) == 0.0


@pytest.mark.parametrize("rank", [6, 7, 8, 9])
@pytest.mark.parametrize("rho", [0.5, 1.0, 100.0, 1.0e4, 1.0e9])
def test_determinant_lemma_has_no_trace_series_rho_boundary(rank, rho):
    """The stable determinant lemma works on both sides of rho=1."""
    n = 2 * rank + 3
    lam = 2.25 * np.eye(n)
    eigenvalues = np.geomspace(0.1, rho, rank)
    left = np.zeros((n, rank))
    left[np.arange(rank), np.arange(rank)] = 1.5 * np.sqrt(eigenvalues)
    perturbation = left @ left.T
    factors = LowRankFactors(left)
    sigma = lam + perturbation
    oracle = _oracle(sigma)

    dense = dense_cholesky_logdet(sigma)
    assert math.isfinite(dense)
    assert _relative(dense, oracle) < 1e-9
    low = low_rank_logdet(lam, perturbation, factors=factors)
    reduced = np.eye(rank) + left.T @ np.linalg.solve(lam, left)
    independent_lemma = _oracle(lam) + _oracle(reduced)
    assert math.isfinite(low)
    assert _relative(low, oracle) < 2e-11
    assert _relative(low, dense) < 2e-11
    assert _relative(low, independent_lemma) < 2e-11


def test_determinant_lemma_admitted_beyond_sigma_condition_resolution():
    """kappa(Sigma) >= 1/eps must not reject the exact determinant lemma.

    The lemma factorizes only the k-by-k reduced matrix, so its precision is
    governed by the factor certificate (eta) and kappa(Lambda), never by the
    n-by-n kappa(Sigma).  Here Lambda is diagonal and ill conditioned while the
    rank-one perturbation keeps Sigma non-diagonal, so a kappa(Sigma) gate would
    reject an otherwise exact logdet of log(1.5).
    """
    lam = np.diag([1.0e8, 1.0e-8, 1.0, 1.0])
    left = np.array([[0.0], [0.0], [0.5], [0.5]])
    perturbation = left @ left.T
    factors = LowRankFactors(left)
    sigma = lam + perturbation

    assert np.linalg.cond(sigma) >= 1.0 / np.finfo(float).eps

    problem = LogDetProblem(lam, perturbation, low_rank_factors=factors)
    verdicts = check_logdet_premises(problem)
    details = verdicts[1].details

    assert details["rank_evidence_valid"] is True
    assert details["condition"] >= details["condition_ceiling"]
    assert details["determinant_lemma_payload"] is True

    result = dispatch_logdet(problem)
    assert result.value == pytest.approx(math.log(1.5), rel=0.0, abs=1e-13)
    assert result.value == pytest.approx(_oracle(sigma), rel=0.0, abs=1e-13)


def test_general_determinant_lemma_handles_a_nonsymmetric_reduced_matrix(
    monkeypatch,
):
    """Falling back to the full Sigma Cholesky bypasses the L != R lemma."""
    from bayesmith.marginal import _logdet_eager as eager

    widths = np.array([1.5, 1.75, 2.0, 2.5])
    lam = np.diag(widths**2)
    basis = np.array(
        [[1.0, 0.0], [0.5, 1.0], [0.0, 0.5], [1.0, -0.5]]
    )
    gauge = np.array([[1.0, 2.0], [0.0, 1.0]])
    left = basis @ gauge
    right = basis @ np.linalg.inv(gauge).T
    perturbation = left @ right.T
    factors = LowRankFactors(left, right)
    reduced = np.eye(2) + right.T @ np.linalg.solve(lam, left)
    symmetric_reduced = np.eye(2) + basis.T @ np.linalg.solve(lam, basis)
    analytic_correction = math.log(
        symmetric_reduced[0, 0] * symmetric_reduced[1, 1]
        - symmetric_reduced[0, 1] * symmetric_reduced[1, 0]
    )
    analytic = math.fsum(np.log(np.diag(lam))) + analytic_correction
    dense_oracle = _oracle(lam + perturbation)

    assert not np.allclose(reduced, reduced.T, rtol=0.0, atol=0.0)
    assert np.array_equal(perturbation, perturbation.T)

    def refuse_full_sigma_cholesky(matrix):
        del matrix
        raise AssertionError("the determinant lemma may not factor full Sigma")

    monkeypatch.setattr(eager, "dense_cholesky_logdet", refuse_full_sigma_cholesky)
    value = low_rank_logdet(lam, perturbation, factors=factors)

    assert value == pytest.approx(analytic, rel=2e-13)
    assert value == pytest.approx(dense_oracle, rel=2e-13)


@pytest.mark.parametrize(
    ("gauge_kind", "expected_level"), [("overflow", 1), ("silent", 5)]
)
def test_general_determinant_lemma_is_invariant_to_extreme_exact_gauges(
    gauge_kind, expected_level, monkeypatch
):
    """Raw L/R reduced products can overflow or lose the determinant by nats."""
    from bayesmith.marginal import _logdet_eager as eager

    n, rank = 20, 2
    lambda_diagonal = np.linspace(2.0, 5.0, n)
    lam = np.diag(lambda_diagonal)
    basis = np.zeros((n, rank))
    basis[0] = [1.0, 0.0]
    basis[1] = [1.0, 1.0]
    perturbation = basis @ basis.T
    if gauge_kind == "overflow":
        gauge = np.diag([2.0**600, 2.0**-600])
        inverse_transpose = np.diag([2.0**-600, 2.0**600])
        config = LadderConfig()
    else:
        scale = 2.0**33
        gauge = np.array([[1.0, scale], [0.0, 1.0]])
        inverse_transpose = np.array([[1.0, 0.0], [-scale, 1.0]])
        config = LadderConfig(
            low_rank_max=0,
            dense_max_n=0,
            finite_max_n=0,
            finite_max_rank=rank,
        )
    left = basis @ gauge
    right = basis @ inverse_transpose
    factors = LowRankFactors(left, right)
    sigma = lam + perturbation
    problem = LogDetProblem(lam, perturbation, low_rank_factors=factors)

    inverse_first = 1.0 / lambda_diagonal[0]
    inverse_second = 1.0 / lambda_diagonal[1]
    reduced_00 = 1.0 + inverse_first + inverse_second
    reduced_01 = inverse_second
    reduced_11 = 1.0 + inverse_second
    analytic = math.fsum(math.log(float(value)) for value in lambda_diagonal)
    analytic += math.log(reduced_00 * reduced_11 - reduced_01**2)
    dense_oracle = _oracle(sigma)

    assert np.array_equal(left @ right.T, perturbation)
    assert np.array_equal(perturbation, perturbation.T)

    def refuse_full_sigma_cholesky(matrix):
        del matrix
        raise AssertionError("the determinant lemma may not factor full Sigma")

    monkeypatch.setattr(eager, "dense_cholesky_logdet", refuse_full_sigma_cholesky)
    if expected_level == 1:
        direct = low_rank_logdet(lam, perturbation, factors=factors)
    else:
        direct = finite_perturbation_logdet(lam, perturbation, factors=factors)
    result = dispatch_logdet(problem, config=config)

    assert result.level == expected_level
    assert direct == pytest.approx(analytic, rel=2e-13)
    assert direct == pytest.approx(dense_oracle, rel=2e-13)
    assert result.value == direct


def test_default_rank_one_hundred_problem_uses_level_five_general_lemma():
    """Making finite-e depend on rho refuses this valid expansive exact factor."""
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
    sigma = lam + perturbation
    problem = LogDetProblem(lam, perturbation, low_rank_factors=factors)

    premises = check_logdet_premises(problem)
    assert premises[1].satisfied is False
    assert premises[4].satisfied is False
    assert premises[5].satisfied is True
    result = dispatch_logdet(problem)

    block = sigma[np.ix_([0, 200], [0, 200])]
    block_determinant = block[0, 0] * block[1, 1] - block[0, 1] * block[1, 0]
    untouched = np.ones(n, dtype=bool)
    untouched[[0, 200]] = False
    analytic = math.log(block_determinant) + math.fsum(
        math.log(float(value)) for value in np.diag(sigma)[untouched]
    )
    dense_oracle = _oracle(sigma)

    assert result.level == 5
    assert result.value == pytest.approx(analytic, rel=2e-12)
    assert result.value == pytest.approx(dense_oracle, rel=2e-12)


def test_rank_eight_expansive_regression_uses_stable_determinant_lemma():
    """The old Newton error must not justify rejecting the replacement payload."""
    n, rank = 128, 8
    widths = np.linspace(1.3, 2.3, n)
    lam = np.diag(widths**2)
    basis, _ = np.linalg.qr(np.random.default_rng(81).normal(size=(n, rank)))
    eigenvalues = np.geomspace(0.1, 1.0e4, rank)
    left = widths[:, None] * basis * np.sqrt(eigenvalues)
    perturbation = left @ left.T
    factors = LowRankFactors(left)
    problem = LogDetProblem(lam, perturbation, low_rank_factors=factors)

    verdict = check_logdet_premises(problem)[1]
    assert verdict.satisfied is True
    result = dispatch_logdet(problem)
    assert result.level == 1
    assert _relative(result.value, _oracle(lam + perturbation)) < 2e-11


def test_d2_large_foreground_low_rank_problem_dispatches_at_level_one():
    """n=1000, k=4, rho=100 is the foreground-dominated D2 regression."""
    n, rank, rho = 1_000, 4, 100.0
    lam = 6.25 * np.eye(n)
    basis, _ = np.linalg.qr(np.random.default_rng(20260830).normal(size=(n, rank)))
    left = 2.5 * math.sqrt(rho) * basis
    perturbation = left @ left.T
    factors = LowRankFactors(left)
    problem = LogDetProblem(lam, perturbation, low_rank_factors=factors)
    config = LadderConfig(dense_max_n=0, finite_max_n=0, finite_max_rank=0)

    result = dispatch_logdet(problem, config=config)
    oracle = _oracle(lam + perturbation)
    reduced = np.eye(rank) + left.T @ np.linalg.solve(lam, left)
    independent_lemma = _oracle(lam) + _oracle(reduced)

    assert result.level == 1
    assert _relative(result.value, oracle) < 1e-12
    assert _relative(result.value, independent_lemma) < 1e-12

    scan_n = 64
    scan_lam = 6.25 * np.eye(scan_n)
    scan_basis, _ = np.linalg.qr(
        np.random.default_rng(641).normal(size=(scan_n, rank))
    )
    for scanned_rho in (0.5, 0.99, 1.0, 1.01, 100.0, 1.0e4):
        scan_left = 2.5 * math.sqrt(scanned_rho) * scan_basis
        scan_perturbation = scan_left @ scan_left.T
        scan_factors = LowRankFactors(scan_left)
        scan_value = low_rank_logdet(
            scan_lam, scan_perturbation, factors=scan_factors
        )
        scan_oracle = _oracle(scan_lam + scan_perturbation)
        scan_reduced = np.eye(rank) + scan_left.T @ np.linalg.solve(
            scan_lam, scan_left
        )
        scan_lemma = _oracle(scan_lam) + _oracle(scan_reduced)
        assert _relative(scan_value, scan_oracle) < 1e-12
        assert _relative(scan_value, scan_lemma) < 1e-12


def test_direct_lambda_and_problem_callers_require_exact_symmetry():
    """A default SPD check must not feed a raw asymmetric triangle to Cholesky."""
    basis, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(8, 8)))
    matrix = basis @ np.diag(np.linspace(6.25, 25.0, 8)) @ basis.T
    asymmetry = float(np.max(np.abs(matrix - matrix.T)))
    # The fixture's premise, as a PROPERTY rather than as one machine's digits.
    # What has to be true is that `basis @ diag @ basis.T` comes back
    # asymmetric by roundoff and only by roundoff: zero would mean the fixture
    # stresses nothing, and anything above roundoff would mean it is testing a
    # real asymmetry instead of the Cholesky-input hazard it is named for.
    # This assertion read `== 8.881784197001252e-16` until 2026-09-02, which is
    # what LAPACK's QR happens to leave on Apple Accelerate; scipy-openblas
    # 0.3.34 leaves 1.7763568394002505e-15, twice that, and the literal took
    # the test down on Linux for a matrix it should have accepted. Measured as
    # a fraction of `eps * max|matrix|`: 0.194 on Accelerate, 0.387 on
    # OpenBLAS, so the band below holds with room while still refusing zero.
    assert 0.0 < asymmetry < 2.0 * np.finfo(float).eps * np.max(np.abs(matrix))
    assert np.linalg.cond(matrix) == pytest.approx(4.0, rel=2e-15)

    with pytest.raises(ValueError, match="symmetric positive definite"):
        lambda_logdet(matrix)
    with pytest.raises(ValueError, match="symmetric positive definite"):
        LogDetProblem(matrix, np.zeros_like(matrix))

    symmetric = matrix / 2.0 + matrix.T / 2.0
    problem = LogDetProblem(symmetric, np.zeros_like(symmetric))
    result = dispatch_logdet(problem)
    assert result.level == 0
    assert _relative(result.value, _oracle(symmetric)) < 2e-14


def test_negative_frozen_trace_order_is_rejected_by_the_premise_checker():
    """A satisfied rung may not defer its order-domain failure to the payload."""
    n = 300
    widths = np.linspace(1.3, 2.4, n)
    lam = np.diag(widths**2)
    basis, _ = np.linalg.qr(np.random.default_rng(303).normal(size=(n, n)))
    eigenvalues = np.linspace(0.03, 0.2, n)
    perturbation = (
        np.diag(widths)
        @ basis
        @ np.diag(eigenvalues)
        @ basis.T
        @ np.diag(widths)
    )
    problem = LogDetProblem(
        lam,
        perturbation,
        frozen_probes=FrozenProbes(np.ones((1, n))),
        trace_order=-1,
        certified_rho=0.2,
    )

    verdict = check_logdet_premises(problem)[7]
    assert verdict.satisfied is False
    with pytest.raises(ResamplingRefused):
        dispatch_logdet(problem)


def test_newton_exact_rungs_require_sigma_positive_definite():
    """An even number of negative eigenvalues must not become a log-abs-det."""
    lam = np.diag(np.array([1.3, 1.7, 2.1, 2.5]) ** 2)
    sigma = np.diag([-1.9, -0.8, 3.1, 5.4])
    perturbation = sigma - lam
    problem = LogDetProblem(lam, perturbation)
    verdicts = check_logdet_premises(problem)

    for level in (1, 5):
        assert verdicts[level].satisfied is False
        assert verdicts[level].details["positive_definite"] is False
    with pytest.raises(ValueError, match="Sigma must be symmetric positive definite"):
        low_rank_logdet(lam, perturbation)
    with pytest.raises(ValueError, match="Sigma must be symmetric positive definite"):
        finite_perturbation_logdet(lam, perturbation)


def test_rung_one_premise_rejects_nonsymmetric_low_rank_sigma_at_rho_below_one():
    """The payload domain must be present in the rung-1 premise itself."""
    n = 10
    widths = np.linspace(1.3, 2.4, n)
    lam = np.diag(widths**2)
    left = np.linspace(0.2, 0.5, n)[:, None]
    right = np.linspace(0.6, 0.1, n)[:, None]
    right *= 0.4 / (right.T @ np.linalg.solve(lam, left)).item()
    perturbation = left @ right.T
    factors = LowRankFactors(left, right)
    problem = LogDetProblem(lam, perturbation, low_rank_factors=factors)
    config = LadderConfig(
        low_rank_fraction=1.0,
        dense_max_n=0,
        finite_max_n=0,
        finite_max_rank=0,
    )

    assert spectral_radius(lam, perturbation) == pytest.approx(0.4)
    verdict = check_logdet_premises(problem, config=config)[1]
    assert verdict.satisfied is False
    assert verdict.details["positive_definite"] is False
    with pytest.raises(ResamplingRefused):
        dispatch_logdet(problem, config=config)


def test_rung_five_premise_rejects_nonsymmetric_sigma_at_rho_below_one():
    """The finite-series fallback cannot defer its SPD failure to execution."""
    widths = np.array([1.3, 1.7, 2.2, 2.6])
    lam = np.diag(widths**2)
    x_matrix = np.array(
        [[0.1, 0.2, 0.0, 0.0], [0.0, 0.15, 0.1, 0.0], [0.0, 0.0, 0.2, 0.1], [0.0, 0.0, 0.0, 0.12]]
    )
    perturbation = x_matrix @ lam
    problem = LogDetProblem(lam, perturbation)
    config = LadderConfig(
        low_rank_max=0,
        dense_max_n=0,
        finite_max_n=4,
        finite_max_rank=0,
    )

    assert spectral_radius(lam, perturbation) == pytest.approx(0.2)
    verdict = check_logdet_premises(problem, config=config)[5]
    assert verdict.satisfied is False
    assert verdict.details["positive_definite"] is False
    with pytest.raises(ResamplingRefused):
        dispatch_logdet(problem, config=config)


def test_problem_constructor_requires_spd_lambda_even_for_level_zero():
    """The level-zero payload cannot discover an invalid Lambda after routing."""
    with pytest.raises(ValueError, match="Lambda must be symmetric positive definite"):
        LogDetProblem(np.array([1.7, -2.6]), np.zeros(2))


def test_state_space_recursion_matches_slogdet_on_a_verified_block_chain():
    """Dropping an LDL Schur update gives the wrong determinant."""
    diagonal = [
        np.array([[3.2, 0.2], [0.2, 2.7]]),
        np.array([[3.6, -0.1], [-0.1, 2.9]]),
        np.array([[3.1, 0.15], [0.15, 3.8]]),
    ]
    links = [
        np.array([[0.25, 0.0], [0.04, -0.18]]),
        np.array([[0.16, 0.03], [0.0, 0.21]]),
    ]
    matrix = np.zeros((6, 6))
    for i, block in enumerate(diagonal):
        matrix[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = block
    for i, link in enumerate(links):
        matrix[2 * (i + 1) : 2 * (i + 2), 2 * i : 2 * i + 2] = link
        matrix[2 * i : 2 * i + 2, 2 * (i + 1) : 2 * (i + 2)] = link.T
    assert state_space_logdet(matrix, block_size=2) == pytest.approx(
        _oracle(matrix), rel=2e-13
    )


def _seed_54_block_chain() -> np.ndarray:
    """Exact symmetric SPD chain whose first Schur update is asymmetric by 1 ULP."""
    rng = np.random.default_rng(54)
    diagonal = []
    for _ in range(4):
        sample = rng.normal(size=(2, 2))
        block = sample @ sample.T + 8.0 * np.eye(2)
        diagonal.append(block / 2.0 + block.T / 2.0)
    links = [0.1 * rng.normal(size=(2, 2)) for _ in range(3)]
    matrix = np.zeros((8, 8))
    for index, block in enumerate(diagonal):
        start = 2 * index
        matrix[start : start + 2, start : start + 2] = block
    for index, link in enumerate(links):
        start = 2 * index
        matrix[start + 2 : start + 4, start : start + 2] = link
        matrix[start : start + 2, start + 2 : start + 4] = link.T
    return matrix


def test_state_space_symmetrizes_each_schur_roundoff_representative():
    """A satisfied chain must not fail when a Schur pivot differs by one ULP."""
    matrix = _seed_54_block_chain()

    assert np.array_equal(matrix, matrix.T)
    assert np.all(np.linalg.eigvalsh(matrix) > 0.0)
    assert np.linalg.cond(matrix) == pytest.approx(2.629776849896354)
    assert state_space_logdet(matrix, block_size=2) == pytest.approx(
        _oracle(matrix), rel=2e-13
    )


def test_dispatch_executes_a_satisfied_state_space_roundoff_fixture():
    """The rung-2 premise and payload share one symmetric Schur interpretation."""
    sigma = _seed_54_block_chain()
    problem = LogDetProblem(
        np.eye(sigma.shape[0]), sigma - np.eye(sigma.shape[0]), chain_block_size=2
    )
    verdict = check_logdet_premises(problem)[2]

    assert verdict.satisfied is True
    result = dispatch_logdet(problem)
    assert result.level == 2
    assert result.value == pytest.approx(_oracle(sigma), rel=2e-13)


def test_state_space_scales_a_resolved_subnormal_schur_solve():
    """LAPACK may return infinity on an unscaled, well-conditioned tiny pivot."""
    smallest = float(np.nextafter(0.0, 1.0))
    matrix = np.array(
        [[2.0 * smallest, smallest], [smallest, 2.0 * smallest]]
    )
    oracle = _exact_two_by_two_scaled_spd_logdet(smallest)

    assert np.linalg.cond(matrix) == pytest.approx(3.0)
    assert state_space_logdet(matrix, block_size=1) == pytest.approx(
        oracle, rel=2e-13
    )


def _tolerance_admitted_subnormal_asymmetry(
    *, transpose: bool = False
) -> tuple[np.ndarray, float]:
    """Return a 1-ULP asymmetric input whose rounded mean is singular."""
    smallest = float(np.nextafter(0.0, 1.0))
    matrix = smallest * np.array([[2.0, 1.0], [2.0, 2.0]])
    return (matrix.T if transpose else matrix), smallest


@pytest.mark.parametrize("transpose", [False, True])
def test_subnormal_symmetric_payload_uses_the_safely_rounded_pair_mean(transpose):
    """Halving each addend first invents an SPD matrix instead of the singular mean."""
    matrix, smallest = _tolerance_admitted_subnormal_asymmetry(
        transpose=transpose
    )
    rounded_mean = (matrix + matrix.T) / 2.0
    problem = LogDetProblem(
        smallest * np.eye(2), matrix - smallest * np.eye(2)
    )
    config = LadderConfig(structure_rtol=0.0, structure_atol=smallest)

    verdict = check_logdet_premises(problem, config=config)[4]

    assert np.array_equal(rounded_mean, np.full((2, 2), 2.0 * smallest))
    assert verdict.details["symmetric"] is True
    assert verdict.details["positive_definite"] is False
    assert verdict.satisfied is False


@pytest.mark.parametrize("transpose", [False, True])
def test_state_space_rejects_a_singular_safely_rounded_subnormal_mean(transpose):
    """The direct chain path must not turn the asymmetric mean into det=3m^2."""
    matrix, smallest = _tolerance_admitted_subnormal_asymmetry(
        transpose=transpose
    )

    with pytest.raises(ValueError, match="positive definite"):
        state_space_logdet(
            matrix, block_size=1, rtol=0.0, atol=smallest
        )


@pytest.mark.parametrize("transpose", [False, True])
def test_dispatch_refuses_a_singular_safely_rounded_subnormal_mean(transpose):
    """Premise and payload must agree that no deterministic rung can use this mean."""
    matrix, smallest = _tolerance_admitted_subnormal_asymmetry(
        transpose=transpose
    )
    problem = LogDetProblem(
        smallest * np.eye(2), matrix - smallest * np.eye(2)
    )
    config = LadderConfig(structure_rtol=0.0, structure_atol=smallest)

    with pytest.raises(ResamplingRefused):
        dispatch_logdet(problem, config=config)


@pytest.mark.parametrize("kind", ["diagonal", "circulant", "toeplitz"])
def test_structured_exact_methods_match_slogdet(kind):
    """Each verified structured evaluator must include every eigenvalue."""
    if kind == "diagonal":
        matrix = np.diag([1.4, 2.2, 3.7, 5.1])
    elif kind == "circulant":
        first = np.array([3.4, 0.3, 0.1, 0.3])
        matrix = np.vstack([np.roll(first, i) for i in range(4)])
    else:
        first = np.array([3.8, 0.4, 0.12, 0.03])
        matrix = np.fromfunction(lambda i, j: first[np.abs(i - j).astype(int)], (4, 4))
    assert structured_logdet(matrix, kind=kind) == pytest.approx(
        _oracle(matrix), rel=2e-13
    )


def test_verified_kronecker_method_matches_dense_slogdet():
    """Wrong factor multiplicities are visible on non-unit factor scales."""
    left = np.array([[2.3, 0.2], [0.2, 1.7]])
    right = np.diag([1.4, 2.2, 3.1])
    matrix = np.kron(left, right)
    structure = KroneckerStructure((left, right))
    assert structured_logdet(
        matrix, kind="kronecker", structure=structure
    ) == pytest.approx(_oracle(matrix), rel=2e-13)


def test_kronecker_premise_rejects_factors_outside_payload_domain():
    """Negative-definite factors can reconstruct SPD Sigma but Cholesky rejects them."""
    positive_left = np.array([[2.3, 0.2], [0.2, 1.7]])
    positive_right = np.diag([1.4, 2.2])
    sigma = np.kron(positive_left, positive_right)
    lam = 0.5 * np.eye(sigma.shape[0])
    negative_structure = KroneckerStructure((-positive_left, -positive_right))
    problem = LogDetProblem(
        lam,
        sigma - lam,
        structure_kind="kronecker",
        structure=negative_structure,
    )

    verdict = check_logdet_premises(problem)[3]
    assert verdict.satisfied is False
    assert "factor" in verdict.reason
    result = dispatch_logdet(problem)
    assert result.level == 4
    assert result.value == pytest.approx(_oracle(sigma), rel=2e-13)


@pytest.mark.parametrize("transpose_left", [False, True])
def test_relaxed_symmetry_cannot_route_inexact_kronecker_factors(transpose_left):
    """Tolerance-admitted factors otherwise leak a payload ValueError at rung three."""
    left = np.array([[2.4, 0.3000001], [0.2999999, 2.1]])
    if transpose_left:
        left = left.T
    right = np.array([[1.7, 0.2], [0.2, 1.4]])
    sigma = np.kron(left, right)
    lam = 0.5 * np.eye(sigma.shape[0])
    problem = LogDetProblem(
        lam,
        sigma - lam,
        structure_kind="kronecker",
        structure=KroneckerStructure((left, right)),
    )
    config = LadderConfig(
        low_rank_max=0,
        low_rank_fraction=0.0,
        dense_max_n=0,
        finite_max_n=0,
        finite_max_rank=0,
        structure_atol=5.0e-7,
    )

    with pytest.raises(ResamplingRefused):
        dispatch_logdet(problem, config=config)
    verdict = check_logdet_premises(problem, config=config)[3]
    assert verdict.satisfied is False
    assert "factor" in verdict.reason


def test_relaxed_symmetry_kronecker_claim_falls_through_with_transpose_parity():
    """Rejecting an inexact factor must let dense use one symmetric representative."""
    left = np.array([[2.4, 0.3000001], [0.2999999, 2.1]])
    right = np.array([[1.7, 0.2], [0.2, 1.4]])
    lam = 0.5 * np.eye(4)
    config = LadderConfig(structure_atol=5.0e-7)
    values = []

    for oriented_left in (left, left.T):
        sigma = np.kron(oriented_left, right)
        symmetric_sigma = sigma / 2.0 + sigma.T / 2.0
        problem = LogDetProblem(
            lam,
            sigma - lam,
            structure_kind="kronecker",
            structure=KroneckerStructure((oriented_left, right)),
        )

        result = dispatch_logdet(problem, config=config)
        verdict = check_logdet_premises(problem, config=config)[3]
        assert verdict.satisfied is False
        assert result.level == 4
        assert result.value == pytest.approx(_oracle(symmetric_sigma), rel=2e-13)
        values.append(result.value)

    assert values[0] == values[1]


def test_kronecker_descriptor_rejects_rectangular_factors_before_premise_check():
    """A square Kronecker product does not make rectangular factors SPD."""
    with pytest.raises(ValueError, match="non-empty square"):
        KroneckerStructure((np.ones((2, 3)), np.ones((3, 2))))


def test_false_structure_claims_are_numerically_refused():
    """A bare structure label must not route an unstructured matrix."""
    matrix = np.array([[2.4, 0.3, 0.05], [0.3, 2.1, 0.2], [0.05, 0.2, 3.2]])
    with pytest.raises(ValueError, match="not circulant"):
        structured_logdet(matrix, kind="circulant")
    with pytest.raises(ValueError, match="not a block chain"):
        state_space_logdet(matrix, block_size=1)


@pytest.mark.parametrize(
    "config",
    [
        LadderConfig(structure_rtol=1.0e-6, structure_atol=0.0),
        LadderConfig(structure_rtol=0.0, structure_atol=3.0e-7),
    ],
)
def test_relaxed_symmetry_config_has_checker_execution_and_orientation_parity(config):
    """Hard-coded SPD tolerances or raw Cholesky triangles break this parity."""
    symmetric = np.array([[2.4, 0.3], [0.3, 2.1]])
    skew = np.array([[0.0, 1.0e-7], [-1.0e-7, 0.0]])
    lam = np.diag(np.array([1.3, 1.4]) ** 2)
    values = []

    for sigma in (symmetric + skew, symmetric - skew):
        problem = LogDetProblem(lam, sigma - lam)
        verdict = check_logdet_premises(problem, config=config)[4]
        result = dispatch_logdet(problem, config=config)

        assert verdict.satisfied is True
        assert result.level == 4
        assert result.value == pytest.approx(_oracle(symmetric), rel=2e-13)
        values.append(result.value)

    assert values[0] == values[1]
    with pytest.raises(ValueError, match="symmetric"):
        dense_cholesky_logdet(symmetric + skew)


def test_state_space_logdet_uses_one_tolerant_symmetric_payload():
    """The direct chain API must use the same tolerated matrix for every gate."""
    matrix = np.array([[2.4, 0.3000001], [0.2999999, 2.1]])
    symmetric = matrix / 2.0 + matrix.T / 2.0
    values = [
        state_space_logdet(oriented, block_size=1, rtol=1.0e-6, atol=0.0)
        for oriented in (matrix, matrix.T)
    ]

    assert values[0] == pytest.approx(_oracle(symmetric), rel=2e-13)
    assert values[1] == values[0]


def test_relaxed_symmetry_does_not_admit_an_unexecutable_level_one_factor():
    """A tolerance-only factor Sigma must fall through without leaking ValueError."""
    n = 10
    widths = np.linspace(1.3, 2.1, n)
    lam = np.diag(widths**2)
    left = np.linspace(0.02, 0.05, n)[:, None]
    right = left.copy()
    right[0, 0] += 1.0e-5
    config = LadderConfig(
        low_rank_fraction=1.0,
        structure_rtol=0.0,
        structure_atol=1.0e-6,
    )
    values = []

    for first, second in ((left, right), (right, left)):
        perturbation = first @ second.T
        factors = LowRankFactors(first, second)
        sigma = lam + perturbation
        symmetric_sigma = sigma / 2.0 + sigma.T / 2.0
        problem = LogDetProblem(lam, perturbation, low_rank_factors=factors)

        verdict = check_logdet_premises(problem, config=config)[1]
        result = dispatch_logdet(problem, config=config)

        assert verdict.details["rank_evidence_valid"] is True
        assert verdict.details["determinant_lemma_payload"] is False
        assert verdict.satisfied is False
        assert result.level == 4
        assert result.value == pytest.approx(_oracle(symmetric_sigma), rel=2e-13)
        values.append(result.value)

    assert values[0] == values[1]


def test_relaxed_symmetry_level_five_uses_the_averaged_generic_payload():
    """A rejected factor payload may fall through to executable finite dense work."""
    n = 4
    widths = np.linspace(1.3, 1.9, n)
    lam = np.diag(widths**2)
    left = np.linspace(0.02, 0.05, n)[:, None]
    right = left.copy()
    right[0, 0] += 1.0e-5
    config = LadderConfig(
        low_rank_max=0,
        dense_max_n=0,
        finite_max_n=n,
        finite_max_rank=0,
        structure_rtol=0.0,
        structure_atol=1.0e-6,
    )
    values = []

    for first, second in ((left, right), (right, left)):
        perturbation = first @ second.T
        factors = LowRankFactors(first, second)
        sigma = lam + perturbation
        symmetric_sigma = sigma / 2.0 + sigma.T / 2.0
        problem = LogDetProblem(lam, perturbation, low_rank_factors=factors)

        verdict = check_logdet_premises(problem, config=config)[5]
        result = dispatch_logdet(problem, config=config)

        assert verdict.details["rank_evidence_valid"] is True
        assert verdict.details["determinant_lemma_payload"] is False
        assert verdict.satisfied is True
        assert result.level == 5
        assert result.value == pytest.approx(_oracle(symmetric_sigma), rel=2e-13)
        values.append(result.value)

    assert values[0] == values[1]


@pytest.mark.parametrize("field", ["structure_rtol", "structure_atol"])
@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
def test_structure_tolerances_must_be_finite_and_nonnegative(field, value):
    """Passing unchecked tolerance values into SPD admission makes the gate undefined."""
    with pytest.raises(ValueError, match="structure_rtol and structure_atol"):
        LadderConfig(**{field: value})


@pytest.mark.parametrize(
    ("scale", "accepted"),
    [
        (np.nextafter(1.0, 0.0), True),
        (1.0, True),
        (np.nextafter(1.0, np.inf), False),
    ],
)
def test_structure_atol_boundary_evaluates_diagonal_and_dense_directly(
    scale, accepted
):
    """Diagnostic atol never admits a nonexact matrix to an exact payload."""
    atol = 1.0e-6
    delta = scale * atol
    sigma = np.array([[2.3, delta], [delta, 3.7]])
    lam = np.diag([1.3, 1.7])
    problem = LogDetProblem(
        lam, sigma - lam, structure_kind="diagonal"
    )
    config = LadderConfig(structure_rtol=0.0, structure_atol=atol)
    verdict = check_logdet_premises(problem, config=config)[3]
    dense = dense_cholesky_logdet(sigma)
    oracle = _oracle(sigma)

    assert bool(np.isclose(delta, 0.0, rtol=0.0, atol=atol)) is accepted
    assert verdict.satisfied is False
    assert _relative(dense, oracle) < 2e-13
    with pytest.raises(ValueError, match="not diagonal"):
        structured_logdet(sigma, kind="diagonal", rtol=0.0, atol=atol)


def test_diagonal_exact_rung_rejects_tolerance_only_near_singular_structure():
    """An ignored atol-sized off-diagonal can dominate a near-singular logdet."""
    diagonal, off_diagonal = 1.0001e-13, 1.0e-13
    sigma = np.array(
        [[diagonal, off_diagonal], [off_diagonal, diagonal]]
    )
    lam = 0.5e-13 * np.eye(2)
    problem = LogDetProblem(
        lam, sigma - lam, structure_kind="diagonal"
    )

    verdict = check_logdet_premises(problem)[3]
    assert verdict.satisfied is False
    with pytest.raises(ValueError, match="not diagonal"):
        structured_logdet(sigma, kind="diagonal")
    assert dense_cholesky_logdet(sigma) == pytest.approx(
        _oracle(sigma), rel=2e-13
    )


@pytest.mark.parametrize(
    ("position", "accepted"),
    [
        ("below", True),
        ("at", True),
        ("above", False),
    ],
)
def test_structure_rtol_boundary_evaluates_circulant_and_dense_directly(
    position, accepted
):
    """Diagnostic rtol never admits a nonexact matrix to an exact payload."""
    rtol = 1.0e-6
    first = np.array([3.4, 0.3, 0.1, 0.3])
    sigma = np.vstack([np.roll(first, index) for index in range(4)])
    edge = first[0] * (1.0 + rtol)
    while not np.isclose(edge, first[0], rtol=rtol, atol=0.0):
        edge = np.nextafter(edge, first[0])
    values = {
        "below": np.nextafter(edge, first[0]),
        "at": edge,
        "above": np.nextafter(edge, np.inf),
    }
    while np.isclose(values["above"], first[0], rtol=rtol, atol=0.0):
        values["above"] = np.nextafter(values["above"], np.inf)
    sigma[1, 1] = values[position]
    lam = 1.7 * np.eye(4)
    problem = LogDetProblem(
        lam, sigma - lam, structure_kind="circulant"
    )
    config = LadderConfig(structure_rtol=rtol, structure_atol=0.0)
    verdict = check_logdet_premises(problem, config=config)[3]
    dense = dense_cholesky_logdet(sigma)
    oracle = _oracle(sigma)

    assert (
        bool(np.isclose(values[position], first[0], rtol=rtol, atol=0.0))
        is accepted
    )
    assert verdict.satisfied is False
    assert _relative(dense, oracle) < 2e-13
    with pytest.raises(ValueError, match="not circulant"):
        structured_logdet(sigma, kind="circulant", rtol=rtol, atol=0.0)


def test_circulant_premise_rejects_tolerance_only_fft_domain():
    """A satisfied circulant row must not fail its real-spectrum payload guard."""
    first = np.array([3.0, 0.2 + 5.0e-14, 0.1, 0.2])
    sigma = np.vstack([np.roll(first, index) for index in range(4)])
    lam = np.diag(np.array([1.3, 1.5, 1.7, 1.9]) ** 2)
    problem = LogDetProblem(
        lam, sigma - lam, structure_kind="circulant"
    )
    config = LadderConfig(
        low_rank_max=0,
        dense_max_n=0,
        finite_max_n=0,
        finite_max_rank=0,
    )

    verdict = check_logdet_premises(problem, config=config)[3]
    assert verdict.satisfied is False
    assert "real positive spectrum" in verdict.reason
    with pytest.raises(ResamplingRefused):
        dispatch_logdet(problem, config=config)


def test_chain_structured_and_dense_premises_include_symmetry_and_spd():
    """Sparsity or size alone must not select a direct method outside its domain."""
    nonsymmetric_chain = np.array([[2.4, 0.3, 0.0], [0.1, 2.1, 0.2], [0.0, 0.2, 3.2]])
    lam_chain = 1.3 * np.eye(3)
    chain_problem = LogDetProblem(
        lam_chain,
        nonsymmetric_chain - lam_chain,
        chain_block_size=1,
    )
    chain_verdicts = check_logdet_premises(chain_problem)
    assert chain_verdicts[2].satisfied is False
    assert "symmetric" in chain_verdicts[2].reason
    assert chain_verdicts[4].satisfied is False

    indefinite_toeplitz = np.array([[1.4, 2.0], [2.0, 1.4]])
    lam_structure = 2.3 * np.eye(2)
    structure_problem = LogDetProblem(
        lam_structure,
        indefinite_toeplitz - lam_structure,
        structure_kind="toeplitz",
    )
    structure_verdict = check_logdet_premises(structure_problem)[3]
    assert structure_verdict.satisfied is False
    assert "positive definite" in structure_verdict.reason


def test_dispatcher_rejects_invalid_chain_and_continues_to_a_valid_exact_row():
    """An invalid chain is rejected and dispatch continues to finite exact."""
    sigma = np.array([[2.4, 0.3, 0.08], [0.3, 2.1, 0.2], [0.08, 0.2, 3.2]])
    lam = 2.0 * np.eye(3)
    problem = LogDetProblem(lam, sigma - lam, chain_block_size=1)
    config = LadderConfig(
        low_rank_max=0,
        dense_max_n=0,
        finite_max_n=3,
        finite_max_rank=0,
    )
    result = dispatch_logdet(problem, config=config)
    assert result.level == 5
    assert [item.level for item in result.rejected] == [0, 1, 2, 3, 4]
    assert result.value == pytest.approx(_oracle(sigma), rel=2e-13)


def test_dispatcher_does_not_project_tolerance_only_state_space_structure():
    """An exact block-chain rung cannot silently drop a tolerated far block."""
    sigma = np.diag([2.4, 2.1, 3.2])
    sigma[0, 2] = sigma[2, 0] = 5.0e-7
    lam = 1.3 * np.eye(3)
    problem = LogDetProblem(lam, sigma - lam, chain_block_size=1)
    config = LadderConfig(structure_rtol=0.0, structure_atol=1.0e-6)

    assert check_logdet_premises(problem, config=config)[2].satisfied is False
    result = dispatch_logdet(problem, config=config)
    assert result.level == 4
    assert result.value == pytest.approx(_oracle(sigma), rel=2e-13)


def test_chain_premise_rejects_a_far_block_needed_for_positive_definiteness():
    """A tolerance-sized far block can make full Sigma positive definite."""
    link = math.sqrt(0.5 + 1.0e-8)
    sigma = np.array(
        [[1.0, link, 5.0e-7], [link, 1.0, link], [5.0e-7, link, 1.0]]
    )
    lam = 0.5 * np.eye(3)
    problem = LogDetProblem(lam, sigma - lam, chain_block_size=1)
    config = LadderConfig(structure_rtol=0.0, structure_atol=1.0e-6)

    verdict = check_logdet_premises(problem, config=config)[2]
    assert verdict.satisfied is False
    assert "not block tridiagonal" in verdict.reason
    with pytest.raises(ValueError, match="not a block chain"):
        state_space_logdet(sigma, block_size=1, rtol=0.0, atol=1.0e-6)
    result = dispatch_logdet(problem, config=config)
    assert result.level == 4
    assert _relative(result.value, _oracle(sigma)) < 3.0e-11


def test_chain_exact_rung_rejects_an_unresolved_near_singular_pivot():
    """Block-LDL cannot bypass the condition gate applied to dense exact work."""
    link = np.nextafter(1.0 / math.sqrt(2.0), 0.0)
    sigma = np.array([[1.0, link, 0.0], [link, 1.0, link], [0.0, link, 1.0]])
    problem = LogDetProblem(0.5 * np.eye(3), sigma - 0.5 * np.eye(3), chain_block_size=1)

    verdict = check_logdet_premises(problem)[2]
    assert verdict.satisfied is False
    assert "condition" in verdict.reason
    with pytest.raises(ValueError, match="condition"):
        state_space_logdet(sigma, block_size=1)


def test_toeplitz_exact_rung_rejects_unresolved_near_singular_input():
    """Letting rung 4 retry this rejected matrix returns a 0.1-nat wrong value."""
    link = np.nextafter(1.0 / math.sqrt(2.0), 0.0)
    sigma = np.array([[1.0, link, 0.0], [link, 1.0, link], [0.0, link, 1.0]])
    problem = LogDetProblem(
        0.5 * np.eye(3),
        sigma - 0.5 * np.eye(3),
        structure_kind="toeplitz",
    )

    verdicts = check_logdet_premises(problem)
    exact = _exact_three_by_three_toeplitz_logdet(float(link))
    unstable = dense_cholesky_logdet(sigma)

    assert verdicts[3].satisfied is False
    assert verdicts[4].satisfied is False
    assert "condition" in verdicts[3].reason
    assert abs(unstable - exact) > 0.05
    with pytest.raises(ValueError, match="condition"):
        structured_logdet(sigma, kind="toeplitz")
    with pytest.raises(ResamplingRefused):
        dispatch_logdet(problem)


def test_dense_level_zero_rejects_an_unresolved_zero_perturbation():
    """Zero P must not bypass the dense 1/eps condition-resolution policy."""
    link = np.nextafter(1.0 / math.sqrt(2.0), 0.0)
    lam = np.array([[1.0, link, 0.0], [link, 1.0, link], [0.0, link, 1.0]])
    exact = _exact_three_by_three_toeplitz_logdet(float(link))
    unstable = lambda_logdet(lam)
    problem = LogDetProblem(lam, np.zeros_like(lam))

    verdicts = check_logdet_premises(problem)

    assert verdicts[0].details["condition"] >= verdicts[0].details["condition_ceiling"]
    assert verdicts[0].satisfied is False
    assert verdicts[5].satisfied is False
    assert abs(unstable - exact) > 0.05
    with pytest.raises(ResamplingRefused):
        dispatch_logdet(problem)


def test_dense_level_zero_refuses_condition_hidden_by_subnormal_scale():
    """An unscaled SVD must not admit an unresolved Lambda just below 1/eps."""
    smallest = float(np.nextafter(0.0, 1.0))
    width = 2**52
    lam = smallest * np.array(
        [[width + 1, width - 2], [width - 2, width - 1]], dtype=float
    )
    problem = LogDetProblem(lam, np.zeros_like(lam))
    ceiling = 1.0 / np.finfo(float).eps
    exact = _exact_two_by_two_fraction_logdet(lam)
    exact_condition, smallest_eigenvalue = _exact_two_by_two_symmetric_spectrum(lam)

    # The premise as exact arithmetic over the fixture's own entries instead of
    # as one machine's digits. The true condition, 4503599627370495.25..., is a
    # hair BELOW the ceiling 4503599627370496.0, so a check that resolved this
    # matrix would admit it; the smaller eigenvalue is subnormal, which is what
    # the test is named for and what nothing here asserted before. Both lines
    # read identically on Accelerate and scipy-openblas 0.3.34 (2026-09-03).
    assert exact_condition < ceiling
    assert smallest_eigenvalue < np.finfo(float).smallest_normal
    assert abs(lambda_logdet(lam) - exact) > 0.1

    verdict = check_logdet_premises(problem)[0]
    assert verdict.details["condition"] >= verdict.details["condition_ceiling"]
    assert verdict.satisfied is False
    with pytest.raises(ResamplingRefused):
        dispatch_logdet(problem)

    # The masking comes last, because it is the only platform-dependent line
    # left and the contract above has to run on every build. An unscaled SVD
    # has to be FOOLED here for that refusal to be evidence about the rescale,
    # and whether it is turns on one denormal ULP of the installed LAPACK, so
    # a build where it is not fooled says so instead of passing. Nothing is
    # lost by not asserting it: a fixture that stopped being this matrix is
    # caught by the two exact lines above, on every platform, which is more
    # than np.linalg.cond can promise here.
    naive_condition = float(np.linalg.cond(lam))
    if not naive_condition < ceiling:
        pytest.skip(_UNSCALED_SVD_NOT_FOOLED.format(reading=naive_condition))


def test_dense_level_four_refuses_condition_hidden_by_subnormal_scale():
    """The dense rung must not return a 0.287682-nat-wrong resolved result."""
    smallest = float(np.nextafter(0.0, 1.0))
    width = 2**52
    sigma = smallest * np.array(
        [[width + 1, width - 2], [width - 2, width - 1]], dtype=float
    )
    lam = np.diag(np.diag(sigma))
    problem = LogDetProblem(lam, sigma - lam)
    ceiling = 1.0 / np.finfo(float).eps
    exact = _exact_two_by_two_fraction_logdet(sigma)
    exact_condition, smallest_eigenvalue = _exact_two_by_two_symmetric_spectrum(sigma)

    # Same matrix, same repair as the level-zero fixture above: the premise is
    # the exact spectrum of these entries, not what one LAPACK reports about
    # them. True condition 4503599627370495.25... < ceiling 4503599627370496.0,
    # smaller eigenvalue subnormal, and the 0.287682-nat error this test is
    # named for is bit-identical on both platforms (measured 2026-09-03).
    assert exact_condition < ceiling
    assert smallest_eigenvalue < np.finfo(float).smallest_normal
    assert abs(dense_cholesky_logdet(sigma) - exact) > 0.1

    verdict = check_logdet_premises(problem)[4]
    assert verdict.details["condition"] >= verdict.details["condition_ceiling"]
    assert verdict.satisfied is False
    with pytest.raises(ResamplingRefused):
        dispatch_logdet(problem)

    # The masking comes last, for the reason recorded in the level-zero fixture
    # above: the contract has to run on every build, and whether an unscaled
    # SVD is fooled here is the installed LAPACK's answer to one denormal ULP.
    naive_condition = float(np.linalg.cond(sigma))
    if not naive_condition < ceiling:
        pytest.skip(_UNSCALED_SVD_NOT_FOOLED.format(reading=naive_condition))


def test_factor_free_dense_level_five_rejects_an_unresolved_matrix():
    """The generic finite payload cannot retry an unresolved dense Cholesky."""
    link = np.nextafter(1.0 / math.sqrt(2.0), 0.0)
    sigma = np.array([[1.0, link, 0.0], [link, 1.0, link], [0.0, link, 1.0]])
    lam = 2.0 * np.eye(3)
    perturbation = sigma - lam
    exact = _exact_three_by_three_toeplitz_logdet(float(link))
    config = LadderConfig(
        low_rank_max=0,
        dense_max_n=0,
        finite_max_n=3,
        finite_max_rank=0,
    )
    problem = LogDetProblem(lam, perturbation)

    verdict = check_logdet_premises(problem, config=config)[5]

    # The rung-5 rho premise, as a PROPERTY rather than as one machine's digits.
    # This read `spectral_radius(lam, perturbation) <= 1.0` until 2026-09-03 and
    # cannot: `rho <= 1` and this rung's condition hazard are in algebraic
    # tension, and binary64 does not hold them apart.  `spectral_radius`
    # measures `rho(P Lambda^-1)`, so at `Lambda = c I` the spectrum of
    # `(Sigma - Lambda) Lambda^-1` is `{sigma_k / c - 1}`: `rho <= 1` forces
    # `sigma_max <= 2 c`, and `condition >= 1/eps` -- this rung's whole point --
    # then forces `sigma_min <= 2 c eps`, i.e. `rho >= 1 - 2 eps` for ANY
    # fixture that carries the hazard with a scalar Lambda.  No choice of Sigma
    # escapes that.  Nor does a non-scalar Lambda: rho below one WITH margin
    # needs Lambda near-singular along Sigma's own null direction, which makes
    # `spectral_radius`'s own Lambda^-1 solve unresolved instead.  Measured
    # 2026-09-03 at `Lambda = Sigma + 2**-53 I`, where the exact rho is
    # 0.311263484575682 and Accelerate returns 0.0.
    #
    # So the premise is true but sub-ulp.  The exact rho is
    # `(1 + sqrt(2) link) / 2` = 0.99999999999999987716979741993..., below one
    # by 1.106 * 2**-53.  Correctly rounded to binary64 that is `1 - 2**-53`,
    # so the deficit assertion below sees exactly 0.5 eps while the exact
    # deficit is 0.553 eps -- two right numbers about two different objects.
    #
    # The 8 eps is a bound, not a fit to the two measurements.  `X = Sigma/2 - I`
    # is symmetric with `||X||_2 = 1`, so its eigenvalues are perfectly
    # conditioned (the eigenvector matrix is orthogonal) and a backward-stable
    # eigensolve errs by at most a small polynomial in n = 3 times eps times
    # `||X||_2`, independently of the BLAS.  Measured 2026-09-03 from a
    # bit-identical X -- IEEE-754 makes `link` 0x1.6a09e667f3bcbp-1 on both, and
    # X's entries 0x1.6a09e667f3bcbp-2 and -0x1.0p-1 -- Apple Accelerate returns
    # 0.9999999999999997, 1.0 eps below the exact value, and scipy-openblas
    # 0.3.34 under OPENBLAS_CORETYPE=ZEN, the CI runner's own kernel, returns
    # 1.0000000000000004, 2.5 eps above it: one number, opposite sides of one.
    #
    # What the band does and does not separate, measured rather than argued.
    # It refuses `min|eig(X)|` (1.2e-16), the middle `|eig(X)|` (0.5) and
    # `max|diag(X)|` (0.5), each more than 2e15 eps outside it and each of which
    # PASSED the `<= 1.0` this replaces.  It does NOT separate rho from
    # `||X||_2` (0.0 eps out on Accelerate, 0.5 on scipy-openblas) nor from
    # `max|eig(Sigma Lambda^-1)|`, the forgotten `- I` (1.0 and 2.5 eps out),
    # because at this fixture `|sigma_min/2 - 1|` and `sigma_max/2` agree to
    # within an ulp.  That blind spot belongs to the fixture, not to the band,
    # and the old assertion had it too.
    exact_rho = _exact_three_by_three_toeplitz_rho(float(link))
    measured_rho = spectral_radius(lam, perturbation)
    assert exact_rho < 1.0
    assert 1.0 - exact_rho < 2.0 * np.finfo(float).eps
    # WHAT THIS BAND COSTS, because it is not a pure tightening. `<= 1.0`
    # rejected everything above one; a two-sided band around `exact_rho`
    # ADMITS `[1.0, exact_rho + band]`, so an eigensolve biased upward by less
    # than the band is no longer caught. Measured on macOS/Accelerate against
    # the old body: a +4 eps bias was KILLED before and SURVIVES a band of
    # eight eps; at four it is KILLED again, and a +6 eps bias with it. The
    # constant is therefore 4.0, not the eight first drafted: the forced floor
    # is the 2.5 eps scipy-openblas actually deviates by, and 4.0 clears it
    # with 1.6x headroom while giving back a mutant that eight released.
    #
    # The FORM is derived -- Bauer-Fike on an exactly symmetric X with
    # ||X||_2 = 1 turns a backward-stable eigensolve's backward error into a
    # forward error of c(n) * eps. The CONSTANT is not: nothing here shows
    # c(3) <= 4 for the QR iteration, and the textbook worst case for n = 3 is
    # larger. It is a measured band with stated headroom, and it is written
    # down that way rather than dressed as a proof.
    assert abs(measured_rho - exact_rho) < 4.0 * np.finfo(float).eps

    assert verdict.details["condition"] >= verdict.details["condition_ceiling"]
    assert verdict.satisfied is False
    assert abs(dense_cholesky_logdet(sigma) - exact) > 0.05
    # The condition ceiling is what this rung's story is about, so pin it where
    # no BLAS can move it.  `low_rank_logdet` reaches the same `_newton_logdet`
    # payload with `require_finite_stability=False`, so on this fixture it walks
    # past the indeterminate rho gate and refuses on the ceiling, unbranched.
    # Measured 2026-09-03: "condition 1.4225308e+16" on Accelerate and
    # "condition 1.2738103e+16" on scipy-openblas; with
    # `_require_resolved_dense_condition` stubbed to a no-op it returns the
    # wrong -35.35050620855721 on BOTH, so this line is what keeps the ceiling
    # killable on the platform where the branch below does not reach it.
    with pytest.raises(ValueError, match="condition"):
        low_rank_logdet(lam, perturbation)
    # WHICH of rung 5's two guards refuses first follows that same unresolvable
    # rho, because `_newton_stability` gates on `rho <= 1.0` before the payload
    # reaches the ceiling.  Bind the expected message to the rho this process
    # measured rather than accepting either everywhere: the ceiling on
    # Accelerate, the stability gate on scipy-openblas, and a swap either way
    # still fails.  The branch reads the same `spectral_radius` the gate reads,
    # so what stops it from being a mirror of the implementation is the band
    # assertion above against an independent Decimal oracle -- that ordering is
    # load-bearing; keep the band ahead of the branch.
    refusal = "condition" if measured_rho <= 1.0 else "expansive spectrum"
    with pytest.raises(ValueError, match=refusal):
        finite_perturbation_logdet(lam, perturbation)
    with pytest.raises(ResamplingRefused):
        dispatch_logdet(problem, config=config)


def test_dense_condition_policy_stays_at_one_over_eps_not_one_over_sqrt_eps():
    """A resolved non-diagonal cell between the two ceilings remains admissible."""
    correlation = 1.0 - 2.0e-10
    sigma = np.array([[1.0, correlation], [correlation, 1.0]])
    lam = 2.0 * np.eye(2)
    problem = LogDetProblem(lam, sigma - lam)
    exact = _exact_two_by_two_correlation_logdet(correlation)

    verdict = check_logdet_premises(problem)[4]
    result = dispatch_logdet(problem)

    assert verdict.details["condition"] > 1.0 / math.sqrt(np.finfo(float).eps)
    assert verdict.details["condition"] < verdict.details["condition_ceiling"]
    assert verdict.satisfied is True
    assert result.level == 4
    assert result.value == pytest.approx(exact, rel=0.0, abs=2.0e-12)


def test_every_satisfied_payload_executes_with_validated_fallbacks():
    """A satisfied row must not discover a payload defect only during execution."""
    lam = np.diag([1.7, 2.6])
    perturbation = np.array([[0.17, 0.03], [0.03, 0.26]])
    wrong_width = FrozenProbes([[1.0, 1.0, 1.0]])
    frozen_problem = LogDetProblem(
        lam,
        perturbation,
        frozen_probes=wrong_width,
        trace_order=3,
        certified_rho=0.2,
    )
    disabled = LadderConfig(
        low_rank_max=0,
        dense_max_n=0,
        finite_max_n=0,
        finite_max_rank=0,
    )
    assert check_logdet_premises(frozen_problem, config=disabled)[7].satisfied is False
    with pytest.raises(ResamplingRefused):
        dispatch_logdet(frozen_problem, config=disabled)

    left = np.array([[0.2], [0.1]])
    invalid_factors = LowRankFactors(left)
    represented = left @ left.T
    tiny_lam = np.diag([0.1, 1.0])
    with_residual = represented + np.diag([0.0, 0.05])
    finite_problem = LogDetProblem(
        tiny_lam, with_residual, low_rank_factors=invalid_factors
    )
    finite_config = LadderConfig(
        low_rank_max=0,
        dense_max_n=0,
        finite_max_n=2,
        finite_max_rank=0,
    )
    verdicts = check_logdet_premises(finite_problem, config=finite_config)
    assert verdicts[5].satisfied is True
    result = dispatch_logdet(finite_problem, config=finite_config)
    assert result.level == 5
    assert result.value == pytest.approx(
        _oracle(tiny_lam + with_residual), rel=2e-13
    )


def test_invalid_expansive_factor_evidence_refuses_before_payload_execution():
    """Treating a present but invalid factor as a lemma payload leaks ValueError."""
    widths = np.array([1.3, 2.1])
    lam = np.diag(widths**2)
    left = np.array([[4.0], [1.5]])
    factors = LowRankFactors(left)
    perturbation = left @ left.T + np.diag([0.0, 0.25])
    problem = LogDetProblem(lam, perturbation, low_rank_factors=factors)
    config = LadderConfig(
        low_rank_max=0,
        dense_max_n=0,
        finite_max_n=2,
        finite_max_rank=0,
    )

    verdict = check_logdet_premises(problem, config=config)[5]
    assert verdict.details["rank_evidence_valid"] is False
    assert verdict.details["determinant_lemma_payload"] is False
    assert verdict.satisfied is False
    with pytest.raises(ResamplingRefused):
        dispatch_logdet(problem, config=config)


@pytest.mark.parametrize("rho", [0.01, 0.5, 0.9, 0.99])
@pytest.mark.parametrize("n", [1, 10, 100, 1000])
def test_trace_log_grid_obeys_the_whole_trace_bound(rho, n):
    """Omitting eigenvalue multiplicity fails on X=rho I_n."""
    widths = np.linspace(1.2, 2.3, n)
    lam_diag = widths**2
    perturbation_diag = rho * lam_diag
    order = 12 if rho <= 0.5 else (120 if rho == 0.9 else 1200)
    traces = _independent_power_traces(lam_diag, perturbation_diag, order + 1)
    want = float(np.sum(np.log(lam_diag + perturbation_diag)))
    for candidate_order in (order, order + 1):
        got = truncated_trace_logdet(
            lam_diag,
            perturbation_diag,
            exact_power_traces=traces,
            order=candidate_order,
        )
        bound = n * rho ** (candidate_order + 1) / ((candidate_order + 1) * (1.0 - rho))
        assert abs(got - want) <= bound * (1.0 + 1e-10) + 2e-12


def test_scalar_and_whole_trace_tail_bounds_are_distinct():
    """Returning the scalar bound for a trace silently understates error."""
    scalar = 0.5**7 / (7 * 0.5)
    assert trace_log_tail_bound(0.5, 6) == pytest.approx(scalar)
    assert whole_trace_log_tail_bound(0.5, 6, 13) == pytest.approx(13 * scalar)


@pytest.mark.parametrize("rho", [0.99, 1.0, 1.01])
def test_strict_rho_boundary_refuses_one_and_above(rho):
    """Rows 5/6/7 are evaluated directly below, at, and above strict rho=1."""
    lam = np.diag([1.3, 2.2])
    perturbation = rho * lam
    oracle = _oracle(lam + perturbation)
    if rho <= 1.0:
        exact = finite_perturbation_logdet(lam, perturbation)
        assert math.isfinite(exact)
        assert _relative(exact, oracle) < 2e-13
    else:
        with pytest.raises(ValueError, match="stability"):
            finite_perturbation_logdet(lam, perturbation)
    probes = FrozenProbes([[1.0, 1.0], [1.0, -1.0]])
    if rho < 1.0:
        order = 1200
        traces = _independent_power_traces(lam, perturbation, order)
        truncated = truncated_trace_logdet(
            lam,
            perturbation,
            exact_power_traces=traces,
            order=order,
            rho=rho,
        )
        frozen = frozen_hutchinson_trace_logdet(
            lam, perturbation, probes, order=order, rho=rho
        )
        bound = 2 * rho ** (order + 1) / ((order + 1) * (1.0 - rho))
        assert math.isfinite(truncated) and math.isfinite(frozen)
        assert abs(truncated - oracle) <= bound * (1 + 1e-8) + 2e-13
        assert abs(frozen - oracle) <= bound * (1 + 1e-8) + 2e-13
    else:
        with pytest.raises(ValueError, match="rho < 1"):
            truncated_trace_logdet(
                lam, perturbation, exact_power_traces=[2 * rho], order=1, rho=rho
            )
        with pytest.raises(ValueError, match="rho < 1"):
            frozen_hutchinson_trace_logdet(lam, perturbation, probes, order=1, rho=rho)


@pytest.mark.parametrize(
    ("actual_rho", "certificate", "accepted"),
    [
        (np.nextafter(1.0, 0.0), np.nextafter(1.0, 0.0), True),
        (1.0, np.nextafter(1.0, 0.0), False),
        (np.nextafter(1.0, np.inf), np.nextafter(1.0, 0.0), False),
        (1.0 + 1.0e-13, 1.0 - 1.0e-13, False),
    ],
)
def test_measured_rho_must_independently_stay_strictly_below_one(
    actual_rho, certificate, accepted
):
    """Certificate-comparison tolerance must never excuse crossing rho=1."""
    lam = np.array([2.0, 4.0])
    perturbation = actual_rho * lam
    traces = _independent_power_traces(lam, perturbation, 1)
    probes = FrozenProbes([[1.0, 1.0], [1.0, -1.0]])
    problem = LogDetProblem(
        lam,
        perturbation,
        exact_power_traces=traces,
        frozen_probes=probes,
        trace_order=1,
        certified_rho=certificate,
    )
    verdicts = check_logdet_premises(problem)
    assert bool(verdicts[6].satisfied) is accepted
    assert bool(verdicts[7].satisfied) is accepted
    if accepted:
        truncated_trace_logdet(
            lam,
            perturbation,
            exact_power_traces=traces,
            order=1,
            rho=certificate,
        )
        frozen_hutchinson_trace_logdet(
            lam, perturbation, probes, order=1, rho=certificate
        )
    else:
        with pytest.raises(ValueError, match="measured rho.*rho < 1"):
            truncated_trace_logdet(
                lam,
                perturbation,
                exact_power_traces=traces,
                order=1,
                rho=certificate,
            )
        with pytest.raises(ValueError, match="measured rho.*rho < 1"):
            frozen_hutchinson_trace_logdet(
                lam, perturbation, probes, order=1, rho=certificate
            )


def test_checker_and_payload_reject_a_few_ulp_rho_certificate_understatement():
    """A comparison tolerance makes a numerically smaller certificate unsound."""
    lam = np.array([1.7, 2.6])
    perturbation = 0.2 * lam
    actual_rho = float(np.max(np.abs(perturbation / lam)))
    certificate = actual_rho
    for _ in range(3):
        certificate = float(np.nextafter(certificate, 0.0))
    traces = _independent_power_traces(lam, perturbation, 1)
    probes = FrozenProbes([[1.0, 1.0], [1.0, -1.0]])
    problem = LogDetProblem(
        lam,
        perturbation,
        exact_power_traces=traces,
        frozen_probes=probes,
        trace_order=1,
        certified_rho=certificate,
    )

    assert actual_rho > certificate
    assert np.isclose(actual_rho, certificate, rtol=1e-12, atol=1e-14)
    verdicts = check_logdet_premises(problem)
    assert verdicts[6].satisfied is False
    assert verdicts[7].satisfied is False
    assert format(certificate, ".17g") in verdicts[7].reason
    with pytest.raises(ValueError, match="understates measured rho"):
        truncated_trace_logdet(
            lam,
            perturbation,
            exact_power_traces=traces,
            order=1,
            rho=certificate,
        )
    with pytest.raises(ValueError, match="understates measured rho"):
        frozen_hutchinson_trace_logdet(
            lam, perturbation, probes, order=1, rho=certificate
        )


def test_zero_certificate_cannot_cover_a_near_zero_extensive_rho():
    """An absolute isclose escape hides an order-one-in-n trace-log tail."""
    n = 100_000
    actual_rho = 5.0e-15
    certificate = 0.0
    lam = np.ones(n)
    perturbation = actual_rho * lam
    probes = FrozenProbes(np.ones((1, n)))
    problem = LogDetProblem(
        lam,
        perturbation,
        exact_power_traces=(),
        frozen_probes=probes,
        trace_order=0,
        certified_rho=certificate,
    )

    omitted_logdet = n * math.log1p(actual_rho)
    assert omitted_logdet > 4.9e-10
    verdicts = check_logdet_premises(problem)
    assert verdicts[6].satisfied is False
    assert verdicts[7].satisfied is False
    with pytest.raises(ValueError, match="understates measured rho"):
        truncated_trace_logdet(
            lam,
            perturbation,
            exact_power_traces=(),
            order=0,
            rho=certificate,
        )
    with pytest.raises(ValueError, match="understates measured rho"):
        frozen_hutchinson_trace_logdet(
            lam, perturbation, probes, order=0, rho=certificate
        )


def test_near_one_rho_certificate_must_still_be_an_exact_upper_bound():
    """Relative closeness cannot replace a conservative trace-tail upper bound."""
    actual_rho = float(np.nextafter(1.0, 0.0))
    certificate = float(np.nextafter(actual_rho, 0.0))
    lam = np.array([1.7, 2.6])
    perturbation = actual_rho * lam
    traces = _independent_power_traces(lam, perturbation, 1)
    probes = FrozenProbes([[1.0, 1.0], [1.0, -1.0]])
    problem = LogDetProblem(
        lam,
        perturbation,
        exact_power_traces=traces,
        frozen_probes=probes,
        trace_order=1,
        certified_rho=certificate,
    )

    assert actual_rho > certificate
    verdicts = check_logdet_premises(problem)
    assert verdicts[6].satisfied is False
    assert verdicts[7].satisfied is False
    with pytest.raises(ValueError, match="understates measured rho"):
        truncated_trace_logdet(
            lam,
            perturbation,
            exact_power_traces=traces,
            order=1,
            rho=certificate,
        )
    with pytest.raises(ValueError, match="understates measured rho"):
        frozen_hutchinson_trace_logdet(
            lam, perturbation, probes, order=1, rho=certificate
        )


def test_level_six_refuses_a_bare_operator_without_exact_power_traces():
    """One generic matvec cannot produce exact traces of matrix powers."""
    lam, perturbation = _spd_fixture(4)
    with pytest.raises(ValueError, match="exact power traces"):
        truncated_trace_logdet(lam, perturbation, order=3)


def test_level_six_verifies_power_traces_and_rho_certificate_numerically():
    """Bare trace numbers and an understated rho certificate are not premises."""
    lam = np.diag([1.7, 2.6])
    perturbation = 0.5 * lam
    wrong_traces = (0.8, 0.5)
    problem = LogDetProblem(
        lam,
        perturbation,
        exact_power_traces=wrong_traces,
        trace_order=2,
        certified_rho=0.4,
    )
    assert check_logdet_premises(problem)[6].satisfied is False
    with pytest.raises(ValueError, match="do not match"):
        truncated_trace_logdet(
            lam,
            perturbation,
            exact_power_traces=wrong_traces,
            order=2,
            rho=0.5,
        )


def test_exact_trace_provider_rejects_error_larger_than_a_tiny_tail_budget():
    """An allclose provider error must not masquerade as analytic tail error."""
    n, rho, order = 2, 0.5, 45
    lam = np.array([1.7, 2.6])
    perturbation = rho * lam
    traces = np.array([n * rho**power for power in range(1, order + 1)])
    traces[0] += 9.0e-12
    problem = LogDetProblem(
        lam,
        perturbation,
        exact_power_traces=traces,
        trace_order=order,
        certified_rho=rho,
    )
    tail = whole_trace_log_tail_bound(rho, order, n)
    assert tail < 2.0e-15
    assert check_logdet_premises(problem)[6].satisfied is False
    with pytest.raises(ValueError, match="exact power traces do not match"):
        truncated_trace_logdet(
            lam,
            perturbation,
            exact_power_traces=traces,
            order=order,
            rho=rho,
        )


def test_runtime_plan_rejects_certificate_multiplicity_below_problem_rank_bound():
    """The n=40 scalar certificate must not promise a whole-logdet tolerance."""
    n, rho, tolerance = 40, 0.5, 1.0e-6
    lam = np.linspace(1.3, 2.7, n) ** 2
    perturbation = rho * lam
    scalar = certify_warmup_rho(
        [rho],
        margin=0.0,
        tolerance=tolerance,
        multiplicity=1,
        lambda_logdets=[lambda_logdet(lam)],
    )
    scalar_traces = _independent_power_traces(lam, perturbation, scalar.order)
    scalar_problem = LogDetProblem(
        lam,
        perturbation,
        exact_power_traces=scalar_traces,
        trace_order=scalar.order,
        certified_rho=scalar.certified_rho,
    )
    with jax.enable_x64(True), pytest.raises(
        ValueError, match="multiplicity.*rank bound 40"
    ):
        make_trace_log_plan(scalar_problem, scalar)

    whole = certify_warmup_rho(
        [rho],
        margin=0.0,
        tolerance=tolerance,
        multiplicity=n,
        lambda_logdets=[lambda_logdet(lam)],
    )
    whole_traces = _independent_power_traces(lam, perturbation, whole.order)
    whole_problem = LogDetProblem(
        lam,
        perturbation,
        exact_power_traces=whole_traces,
        trace_order=whole.order,
        certified_rho=whole.certified_rho,
    )
    with jax.enable_x64(True):
        plan = make_trace_log_plan(whole_problem, whole)
        value = float(
            plan(
                jnp.sum(jnp.log(jnp.asarray(lam))),
                jnp.asarray(whole_traces),
            )
        )
    assert plan.order == choose_trace_order(
        rho, 0.5 * tolerance, multiplicity=n
    )
    assert abs(value - _oracle(np.diag(lam + perturbation))) <= tolerance


def test_retained_trace_audit_rechecks_certificate_multiplicity_rank_bound():
    """A retained rank increase must revoke a warmup multiplicity certificate."""
    import bayesmith.marginal.logdet as module

    n, rho, tolerance = 40, 0.5, 1.0e-6
    certificate = certify_warmup_rho(
        [rho], margin=0.0, tolerance=tolerance, multiplicity=1
    )
    lam = np.linspace(1.3, 2.7, n) ** 2
    perturbation = rho * lam
    retained = LogDetProblem(
        lam,
        perturbation,
        exact_power_traces=_independent_power_traces(
            lam, perturbation, certificate.order
        ),
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    assert audit_retained_rho([rho], certificate).passed is True
    report = module.audit_retained_power_traces([retained], certificate)
    assert report.passed is False
    assert report.violations == (0,)


def _certificate_for_guard_tests(**overrides):
    data = {
        "measured_max": 0.2,
        "margin": 0.0,
        "certified_rho": 0.2,
        "order": choose_trace_order(0.2, 5.0e-4, multiplicity=2),
        "tolerance": 1.0e-3,
        "tail_tolerance": 5.0e-4,
        "multiplicity": 2,
    }
    data.update(overrides)
    return RhoCertificate(**data)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_abs_lambda_logdet", -1.0, "lambda-logdet scale"),
        ("max_abs_lambda_logdet", math.nan, "lambda-logdet scale"),
        ("max_x_operator_norm", -1.0, "operator-norm"),
        ("max_x_operator_norm", math.nan, "operator-norm"),
    ],
)
def test_certificate_rejects_invalid_optional_roundoff_bounds(field, value, message):
    """Optional does not mean an invalid supplied scale can be ignored."""
    with pytest.raises(ValueError, match=message):
        _certificate_for_guard_tests(**{field: value})


@pytest.mark.parametrize("tail_fraction", [0.0, 1.0, math.nan])
def test_warmup_rejects_invalid_tail_budget_fraction(tail_fraction):
    with pytest.raises(ValueError, match="tail_fraction"):
        certify_warmup_rho(
            [0.2],
            margin=0.0,
            tolerance=1.0e-3,
            multiplicity=2,
            tail_fraction=tail_fraction,
        )


@pytest.mark.parametrize("values", [[], [math.nan]])
def test_warmup_rejects_invalid_lambda_logdet_measurements(values):
    with pytest.raises(ValueError, match="lambda logdets"):
        certify_warmup_rho(
            [0.2],
            margin=0.0,
            tolerance=1.0e-3,
            multiplicity=2,
            lambda_logdets=values,
        )


@pytest.mark.parametrize("margin", [-1.0, math.nan])
def test_warmup_rejects_invalid_lambda_logdet_margin(margin):
    with pytest.raises(ValueError, match="lambda-logdet safety margin"):
        certify_warmup_rho(
            [0.2],
            margin=0.0,
            tolerance=1.0e-3,
            multiplicity=2,
            lambda_logdet_margin=margin,
        )


@pytest.mark.parametrize("values", [[], [-1.0], [math.nan]])
def test_warmup_rejects_invalid_absolute_x_norm_measurements(values):
    with pytest.raises(ValueError, match="X operator norms"):
        certify_warmup_rho(
            [0.2],
            margin=0.0,
            tolerance=1.0e-3,
            multiplicity=2,
            x_operator_norms=values,
        )


@pytest.mark.parametrize("margin", [-1.0, math.nan])
def test_warmup_rejects_invalid_absolute_x_norm_margin(margin):
    with pytest.raises(ValueError, match="operator-norm safety margin"):
        certify_warmup_rho(
            [0.2],
            margin=0.0,
            tolerance=1.0e-3,
            multiplicity=2,
            x_operator_norm_margin=margin,
        )


def test_retained_scale_audits_reject_missing_certificates_and_empty_inputs():
    no_scales = _certificate_for_guard_tests()
    with pytest.raises(ValueError, match="no lambda-logdet"):
        audit_retained_lambda_logdet([1.0], no_scales)
    with pytest.raises(ValueError, match=r"no \|X\|"):
        audit_retained_operator_norm([0.2], no_scales)

    scales = _certificate_for_guard_tests(
        max_abs_lambda_logdet=2.0, max_x_operator_norm=0.3
    )
    with pytest.raises(ValueError, match="non-empty"):
        audit_retained_lambda_logdet([], scales)
    with pytest.raises(ValueError, match="non-empty"):
        audit_retained_operator_norm([], scales)


def test_empty_retained_trace_audit_is_never_a_vacuous_pass():
    with pytest.raises(ValueError, match="at least one"):
        audit_retained_power_traces([], _certificate_for_guard_tests())


def test_frozen_plan_requires_absolute_x_norm_roundoff_certificate():
    lam = np.array([1.7, 2.6])
    rho = 0.2
    certificate = certify_warmup_rho(
        [rho],
        margin=0.0,
        tolerance=1.0e-3,
        multiplicity=2,
        lambda_logdets=[lambda_logdet(lam)],
    )
    problem = LogDetProblem(
        lam,
        rho * lam,
        frozen_probes=FrozenProbes([[1.0, 1.0], [1.0, -1.0]]),
        trace_order=certificate.order,
        certified_rho=rho,
    )
    with jax.enable_x64(True), pytest.raises(ValueError, match="max_x_operator_norm"):
        make_frozen_trace_log_plan(problem, certificate)


def test_low_rank_guard_errors_identify_rows_and_missing_lambda_evidence():
    from bayesmith.marginal import _logdet_eager as eager

    bad_rows = LowRankFactors(np.ones((2, 1)), np.ones((2, 1)))
    with pytest.raises(ValueError, match="row counts"):
        low_rank_logdet(np.diag([1.3, 1.7, 2.1]), np.eye(3), factors=bad_rows)

    factors = LowRankFactors(np.ones((3, 1)))
    with pytest.raises(ValueError, match="Lambda is required"):
        eager._algebraic_rank_bound(np.eye(3), factors)


def test_direct_block_chain_guard_rejects_asymmetry_before_factorization():
    matrix = np.array([[2.3, 0.2, 0.0], [0.1, 2.7, 0.3], [0.0, 0.3, 3.1]])
    with pytest.raises(ValueError, match="symmetric"):
        state_space_logdet(matrix, block_size=1)

    indefinite = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 2.0], [0.0, 2.0, 1.0]])
    with pytest.raises(ValueError, match="positive definite matrix"):
        state_space_logdet(indefinite, block_size=1)


def test_problem_refuses_probe_objects_that_can_resample_behind_values():
    """A .values-shaped duck type is not evidence that probes are immutable."""

    class ResamplingProbes:
        @property
        def values(self):
            return np.random.default_rng().choice((-1.0, 1.0), size=(8, 2))

    with pytest.raises(TypeError, match="FrozenProbes"):
        LogDetProblem(
            np.array([1.7, 2.6]),
            np.array([0.17, 0.26]),
            frozen_probes=ResamplingProbes(),
            trace_order=3,
            certified_rho=0.1,
        )


def test_frozen_probe_subclasses_cannot_override_values_with_resampling():
    """Only the exact bytes-backed class can satisfy the immutable-probe premise."""

    class ResamplingFrozenProbes(FrozenProbes):
        @property
        def values(self):
            return np.random.default_rng().normal(size=self.shape)

    probes = ResamplingFrozenProbes([[1.0, 1.0], [1.0, -1.0]])
    with pytest.raises(TypeError, match="exact FrozenProbes"):
        LogDetProblem(
            np.array([1.7, 2.6]),
            np.array([0.17, 0.26]),
            frozen_probes=probes,
        )

    lam = np.array([1.7, 2.6])
    perturbation = np.array([0.17, 0.26])
    with pytest.raises(TypeError, match="exact FrozenProbes"):
        frozen_hutchinson_trace_logdet(
            lam, perturbation, probes, order=3, rho=0.1
        )


def test_direct_frozen_estimator_rejects_values_duck_types():
    """The public eager rung must enforce the same exact-type premise."""

    class ResamplingProbes:
        @property
        def values(self):
            return np.random.default_rng().choice((-1.0, 1.0), size=(2, 2))

    with pytest.raises(TypeError, match="exact FrozenProbes"):
        frozen_hutchinson_trace_logdet(
            np.array([1.7, 2.6]),
            np.array([0.17, 0.26]),
            ResamplingProbes(),
            order=3,
            rho=0.1,
        )


def test_direct_frozen_estimator_reads_the_checked_probe_buffer_once(monkeypatch):
    """Width validation and arithmetic must use the very same probe snapshot."""
    probes = FrozenProbes([[1.0, 1.0], [1.0, -1.0]])
    original = FrozenProbes.values
    reads = 0

    def counted_values(instance):
        nonlocal reads
        reads += 1
        return original.__get__(instance, FrozenProbes)

    monkeypatch.setattr(FrozenProbes, "values", property(counted_values))
    frozen_hutchinson_trace_logdet(
        np.array([1.7, 2.6]),
        np.array([0.17, 0.26]),
        probes,
        order=3,
        rho=0.1,
    )
    assert reads == 1


def test_order_selection_is_minimal_at_its_boundary():
    """An off-by-one can select an order whose bound exceeds tolerance."""
    rho, multiplicity, tolerance = 0.5, 7, 1e-4
    order = choose_trace_order(rho, tolerance, multiplicity=multiplicity)
    at = multiplicity * rho ** (order + 1) / ((order + 1) * (1 - rho))
    before = multiplicity * rho**order / (order * (1 - rho))
    assert at <= tolerance
    assert before > tolerance


def test_frozen_probes_are_copied_read_only_and_reused_bitwise():
    """Aliasing mutable probes would make the HMC target move between calls."""
    source = np.array([[1.0, 1.0], [1.0, -1.0]])
    probes = FrozenProbes(source)
    source[0, 0] = 99.0
    with pytest.raises(ValueError):
        probes.values[0, 0] = 5.0
    lam = np.diag([1.7, 2.6])
    perturbation = np.diag([0.17, 0.52])
    values = [
        frozen_hutchinson_trace_logdet(
            lam, perturbation, probes, order=8, rho=0.2
        )
        for _ in range(5)
    ]
    public = probes.values
    try:
        public.setflags(write=True)
        public[0, 0] = -17.0
    except ValueError:
        pass
    second = frozen_hutchinson_trace_logdet(lam, perturbation, probes, order=8, rho=0.2)
    assert values == [values[0]] * 5
    assert values[0] == second
    assert values[0] == pytest.approx(_oracle(lam + perturbation), rel=3e-6)


@pytest.mark.parametrize("rho", [1.0, 1.01])
def test_frozen_taylor_estimator_refuses_without_a_strict_rho_certificate(rho):
    """Frozen probes remove noise, not Taylor divergence at rho>=1."""
    lam = np.diag([1.7, 2.6])
    perturbation = rho * lam
    probes = FrozenProbes([[1.0, 1.0], [1.0, -1.0]])
    with pytest.raises(ValueError, match="rho < 1"):
        frozen_hutchinson_trace_logdet(lam, perturbation, probes, order=8, rho=rho)


def test_warmup_certificate_and_postrun_audit_are_eager_and_conservative():
    """Forgetting margin or retained-sample rechecks hides a rho violation."""
    warmup = certify_warmup_rho(
        [0.2, 0.35, 0.4], margin=0.08, tolerance=1e-6, multiplicity=5
    )
    assert warmup.measured_max == 0.4
    assert warmup.certified_rho == pytest.approx(0.48)
    assert warmup.order == choose_trace_order(0.48, 0.5e-6, multiplicity=5)
    audit = audit_retained_rho([0.3, 0.49, 0.51], warmup)
    assert audit.passed is False
    assert audit.violations == (1, 2)


def test_retained_exact_trace_provider_is_reaudited_after_sampling():
    """Theta-dependent exact traces need the same warmup/runtime/audit lifecycle as rho."""
    import bayesmith.marginal.logdet as module

    lam = np.array([1.7, 2.6])
    perturbation = 0.2 * lam
    certificate = certify_warmup_rho(
        [0.2], margin=0.0, tolerance=1.0e-6, multiplicity=2
    )
    traces = _independent_power_traces(lam, perturbation, certificate.order)
    good = LogDetProblem(
        lam,
        perturbation,
        exact_power_traces=traces,
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )
    bad_traces = list(traces)
    bad_traces[-1] += 1.0e-8
    bad = LogDetProblem(
        lam,
        perturbation,
        exact_power_traces=bad_traces,
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    report = module.audit_retained_power_traces([good, bad], certificate)
    assert report.passed is False
    assert report.violations == (1,)


def test_runtime_factories_refuse_tolerance_below_float32_ulp_at_output_scale():
    """The ULP guard itself refuses accuracy unavailable to JAX float32."""
    n, rho, tolerance = 40, 0.5, 1.0e-6
    lam = np.linspace(1.3, 2.7, n) ** 2
    perturbation = rho * lam
    certificate = certify_warmup_rho(
        [rho],
        margin=0.0,
        tolerance=tolerance,
        multiplicity=n,
        lambda_logdets=[lambda_logdet(lam)],
        x_operator_norms=[rho],
    )
    traces = _independent_power_traces(lam, perturbation, certificate.order)
    probes = FrozenProbes(np.ones((1, n)))
    trace_problem = LogDetProblem(
        lam,
        perturbation,
        exact_power_traces=traces,
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )
    frozen_problem = LogDetProblem(
        lam,
        perturbation,
        frozen_probes=probes,
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="ULP"):
            make_trace_log_plan(trace_problem, certificate)
        with pytest.raises(ValueError, match="ULP"):
            make_frozen_trace_log_plan(frozen_problem, certificate)
    with jax.enable_x64(True):
        assert make_trace_log_plan(trace_problem, certificate).order == certificate.order
        assert (
            make_frozen_trace_log_plan(frozen_problem, certificate).order
            == certificate.order
        )


def test_trace_plan_cannot_leave_the_precision_context_that_certified_it():
    """A float64-certified plan must refuse an execution silently demoted to float32."""
    n, rho = 40, 0.5
    lam = np.linspace(1.3, 2.7, n) ** 2
    perturbation = rho * lam
    certificate = certify_warmup_rho(
        [rho],
        margin=0.0,
        tolerance=1.0e-6,
        multiplicity=n,
        lambda_logdets=[lambda_logdet(lam)],
    )
    traces = _independent_power_traces(lam, perturbation, certificate.order)
    problem = LogDetProblem(
        lam,
        perturbation,
        exact_power_traces=traces,
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    with jax.enable_x64(True):
        plan = make_trace_log_plan(problem, certificate)
    with jax.enable_x64(False), pytest.raises(
        ValueError, match="real floating precision"
    ):
        plan(lambda_logdet(lam), jnp.asarray(traces))


def test_runtime_precision_guard_refuses_float32_base_series_cancellation():
    """Final-output ULP alone would miss float32 cancellation in the trace series."""
    lam = np.array([100.0, 101.0])
    sigma = np.array([1.0, 1.0001])
    perturbation = sigma - lam
    rho = spectral_radius(lam, perturbation)
    certificate = certify_warmup_rho(
        [rho],
        margin=0.0,
        tolerance=1.0e-8,
        multiplicity=2,
        lambda_logdets=[lambda_logdet(lam)],
    )
    traces = _independent_power_traces(lam, perturbation, certificate.order)
    problem = LogDetProblem(
        lam,
        perturbation,
        exact_power_traces=traces,
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    with jax.enable_x64(False), pytest.raises(ValueError, match="roundoff"):
        make_trace_log_plan(problem, certificate)


@pytest.mark.parametrize(
    ("tolerance", "accepted"),
    [
        (1.0, True),
        (1.0e-1, True),
        (1.0e-2, True),
        (1.0e-3, True),
        (1.0e-4, True),
        (2.0e-5, False),
        (1.0e-5, False),
    ],
)
def test_float32_plan_is_decided_by_ulp_and_gamma_bounds(tolerance, accepted):
    """D7 uses the proved error budget, not an unconditional dtype label."""
    lam = (np.array([1.3, 1.9], dtype=np.float32) ** 2).astype(np.float32)
    rho = 0.5
    perturbation = (rho * lam).astype(np.float32)
    certificate = certify_warmup_rho(
        [rho],
        margin=0.0,
        tolerance=tolerance,
        multiplicity=2,
        lambda_logdets=[lambda_logdet(lam)],
    )
    traces = _independent_power_traces(lam, perturbation, certificate.order)
    problem = LogDetProblem(
        lam,
        perturbation,
        exact_power_traces=traces,
        trace_order=certificate.order,
        certified_rho=rho,
    )

    with jax.enable_x64(False):
        if accepted:
            plan = make_trace_log_plan(problem, certificate)
            assert np.dtype(plan._runtime_dtype) == np.dtype(np.float32)
        else:
            with pytest.raises(ValueError, match="ULP|roundoff"):
                make_trace_log_plan(problem, certificate)


def test_low_rank_multiplicity_certificate_is_tight_not_dimension_wide():
    """A proved rank-two update must not be inflated to ambient n=40."""
    n, rank, rho = 40, 2, 0.2
    widths = np.linspace(1.3, 2.7, n)
    lam = np.diag(widths**2)
    basis, _ = np.linalg.qr(np.random.default_rng(402).normal(size=(n, rank)))
    left = widths[:, None] * basis * math.sqrt(rho)
    perturbation = left @ left.T
    factors = LowRankFactors(left)
    certificate = certify_warmup_rho(
        [rho],
        margin=0.0,
        tolerance=1.0e-5,
        multiplicity=rank,
        lambda_logdets=[lambda_logdet(lam)],
    )
    problem = LogDetProblem(
        lam,
        perturbation,
        low_rank_factors=factors,
        exact_power_traces=_independent_power_traces(
            lam, perturbation, certificate.order
        ),
        trace_order=certificate.order,
        certified_rho=rho,
    )

    with jax.enable_x64(True):
        assert make_trace_log_plan(problem, certificate).order == certificate.order


def test_exact_plan_combines_analytic_tail_and_roundoff_in_one_tolerance():
    """Tail and arithmetic cannot each spend the full absolute-error budget."""
    lam = np.array([math.exp(10.0)])
    rho = math.sqrt(2.0e-12)
    perturbation = -rho * lam
    order_one_tail = rho**2 / (2.0 * (1.0 - rho))
    tolerance = order_one_tail * (1.0 + 1.0e-12)
    certificate = certify_warmup_rho(
        [rho],
        margin=0.0,
        tolerance=tolerance,
        multiplicity=1,
        lambda_logdets=[lambda_logdet(lam)],
    )
    traces = _independent_power_traces(lam, perturbation, certificate.order)
    problem = LogDetProblem(
        lam,
        perturbation,
        exact_power_traces=traces,
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    with jax.enable_x64(True):
        plan = make_trace_log_plan(problem, certificate)
        runtime = float(plan(jnp.log(lam[0]), jnp.asarray(traces)))
    oracle = float(np.log(lam[0] + perturbation[0]))
    assert certificate.order > 1
    assert abs(runtime - oracle) <= tolerance


def test_runtime_plan_requires_and_reaudits_a_lambda_logdet_scale_bound():
    """A theta-dependent base scale is certified like rho, not inferred once."""
    n, rho, tolerance = 10_000, 1.0e-8, 1.0e-10
    lam = np.linspace(1.3, 2.3, n) ** 2
    perturbation = rho * lam
    certificate = certify_warmup_rho(
        [rho],
        margin=0.0,
        tolerance=tolerance,
        multiplicity=n,
        lambda_logdets=[lambda_logdet(lam)],
    )
    traces = _independent_power_traces(lam, perturbation, certificate.order)
    problem = LogDetProblem(
        lam,
        perturbation,
        exact_power_traces=traces,
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    with jax.enable_x64(True):
        plan = make_trace_log_plan(problem, certificate)
    assert plan.order == certificate.order
    audit = audit_retained_lambda_logdet(
        [lambda_logdet(lam), 50.0 * n], certificate
    )
    assert audit.passed is False
    assert audit.violations == (1,)

    no_scale = certify_warmup_rho(
        [rho], margin=0.0, tolerance=tolerance, multiplicity=n
    )
    with jax.enable_x64(True), pytest.raises(ValueError, match="max_abs_lambda"):
        make_trace_log_plan(problem, no_scale)

    wide = certify_warmup_rho(
        [rho],
        margin=0.0,
        tolerance=tolerance,
        multiplicity=n,
        lambda_logdets=[lambda_logdet(lam), 50.0 * n],
    )
    with jax.enable_x64(True), pytest.raises(ValueError, match="roundoff"):
        make_trace_log_plan(problem, wide)


def test_frozen_runtime_requires_x64_for_nonnormal_matmul_cancellation():
    """Float32 loses diagonal terms when huge probe products cancel."""
    lam = np.diag(np.array([1.3, 1.9]) ** 2)
    x_matrix = np.array([[0.1, 1.0e8], [0.0, 0.2]])
    perturbation = x_matrix @ lam
    probes = FrozenProbes([[1.0, 1.0], [1.0, -1.0]])
    certificate = certify_warmup_rho(
        [0.2],
        margin=0.0,
        tolerance=1.0e-3,
        multiplicity=2,
        lambda_logdets=[lambda_logdet(lam)],
        x_operator_norms=[np.linalg.norm(np.abs(x_matrix), ord=2)],
    )
    problem = LogDetProblem(
        lam,
        perturbation,
        frozen_probes=probes,
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    with jax.enable_x64(False), pytest.raises(ValueError, match="roundoff"):
        make_frozen_trace_log_plan(problem, certificate)
    with jax.enable_x64(True), pytest.raises(ValueError, match="roundoff"):
        make_frozen_trace_log_plan(problem, certificate)

    narrow = certify_warmup_rho(
        [0.2],
        margin=0.0,
        tolerance=1.0e-3,
        multiplicity=2,
        lambda_logdets=[0.0],
        x_operator_norms=[0.2],
    )
    with jax.enable_x64(True), pytest.raises(ValueError, match="understates"):
        make_frozen_trace_log_plan(problem, narrow)
    audit = audit_retained_operator_norm(
        [0.2, np.linalg.norm(np.abs(x_matrix), ord=2)], narrow
    )
    assert audit.passed is False
    assert audit.violations == (1,)


def test_frozen_precision_rejects_plain_norm_when_absolute_action_norm_is_larger():
    """Matvec roundoff is controlled by ||abs(X)||, not cancellation in ||X||."""
    n = 16
    hadamard = np.ones((1, 1))
    while hadamard.shape[0] < n:
        hadamard = np.block([[hadamard, hadamard], [hadamard, -hadamard]])
    x_matrix = 0.5 * hadamard / math.sqrt(n)
    ordinary_norm = float(np.linalg.norm(x_matrix, ord=2))
    absolute_action_norm = float(np.linalg.norm(np.abs(x_matrix), ord=2))
    assert ordinary_norm == pytest.approx(0.5)
    assert absolute_action_norm == pytest.approx(2.0)

    probes = FrozenProbes(np.ones((1, n)))
    problem_data = {
        "measured_rhos": [0.5],
        "margin": 0.0,
        "tolerance": 1.0e-3,
        "multiplicity": n,
        "lambda_logdets": [n * math.log(2.3)],
    }
    understated = certify_warmup_rho(
        **problem_data, x_operator_norms=[ordinary_norm]
    )
    problem = LogDetProblem(
        2.3 * np.eye(n),
        2.3 * x_matrix,
        frozen_probes=probes,
        trace_order=understated.order,
        certified_rho=understated.certified_rho,
    )
    with jax.enable_x64(True), pytest.raises(
        ValueError, match=r"\|X\| operator-norm.*understates"
    ):
        make_frozen_trace_log_plan(problem, understated)

    conservative = certify_warmup_rho(
        **problem_data, x_operator_norms=[absolute_action_norm]
    )
    with jax.enable_x64(True):
        assert make_frozen_trace_log_plan(problem, conservative).order == problem.trace_order


def test_frozen_precision_roundoff_bound_includes_probe_reduction_count():
    """An unbounded number of frozen probes cannot get a p-free error proof."""
    certificate = certify_warmup_rho(
        [0.1],
        margin=0.0,
        tolerance=1.0e-12,
        multiplicity=1,
        lambda_logdets=[0.0],
        x_operator_norms=[0.1],
    )

    def problem(probe_count):
        return LogDetProblem(
            np.ones(1),
            np.array([0.1]),
            frozen_probes=FrozenProbes(np.ones((probe_count, 1))),
            trace_order=certificate.order,
            certified_rho=certificate.certified_rho,
        )

    with jax.enable_x64(True):
        assert make_frozen_trace_log_plan(problem(1), certificate).order == certificate.order
        with pytest.raises(ValueError, match="roundoff"):
            make_frozen_trace_log_plan(problem(10_000), certificate)


def test_runtime_precision_guard_keeps_compact_diagonal_inputs_matrix_free(
    monkeypatch,
):
    """Checking one ULP must not materialize a 10,000-square diagonal matrix."""
    from bayesmith.marginal import _logdet_eager as eager

    n, rho = 10_000, 0.01
    lam = np.linspace(1.3, 2.3, n) ** 2
    perturbation = rho * lam
    certificate = certify_warmup_rho(
        [rho],
        margin=0.0,
        tolerance=1.0e-6,
        multiplicity=n,
        lambda_logdets=[lambda_logdet(lam)],
    )
    traces = _independent_power_traces(lam, perturbation, certificate.order)
    problem = LogDetProblem(
        lam,
        perturbation,
        exact_power_traces=traces,
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    def refuse_dense(matrix):
        del matrix
        raise AssertionError("compact precision guard materialized a dense matrix")

    monkeypatch.setattr(eager, "_dense", refuse_dense)

    with jax.enable_x64(True):
        assert make_trace_log_plan(problem, certificate).order == certificate.order


@pytest.mark.parametrize("rho", [0.01, 0.5, 0.9, 0.99, 1.01])
def test_spectral_radius_matches_literal_dense_eigenvalue_construction(rho):
    """The dense oracle is the fixture parameter, never production eigvals."""
    lam = np.array([1.7, 2.6])
    assert spectral_radius(lam, rho * lam) == pytest.approx(rho, rel=2e-15)

    rotation = np.array([[1.0, -1.0], [1.0, 1.0]]) / math.sqrt(2.0)
    x_matrix = rotation @ np.diag([rho, 0.2 * rho]) @ rotation.T
    dense_lam = 2.3 * np.eye(2)
    perturbation = x_matrix @ dense_lam
    assert np.max(np.abs(np.diag(x_matrix))) < rho
    assert spectral_radius(dense_lam, perturbation) == pytest.approx(
        rho, rel=2e-13, abs=2e-13
    )

    order = 4
    problem = LogDetProblem(
        dense_lam,
        perturbation,
        exact_power_traces=_independent_power_traces(
            dense_lam, perturbation, order
        ),
        trace_order=order,
        certified_rho=rho,
    )
    disabled = LadderConfig(
        low_rank_max=0,
        dense_max_n=0,
        finite_max_n=0,
        finite_max_rank=0,
    )
    assert check_logdet_premises(problem, config=disabled)[6].satisfied is (
        rho < 1.0
    )
    if rho >= 1.0:
        with pytest.raises(ResamplingRefused):
            dispatch_logdet(problem, config=disabled)


def test_dispatcher_selects_first_satisfied_row_and_exposes_rejections():
    """Reordering rows or discarding failed premise evidence breaks the ladder."""
    lam = np.diag([1.7, 2.1, 2.8, 3.3])
    perturbation = np.diag([0.11, 0.08, 0.05, 0.02])
    problem = LogDetProblem(lam, perturbation, chain_block_size=1)
    config = LadderConfig(
        low_rank_max=0, dense_max_n=1, finite_max_n=1, finite_max_rank=0
    )
    result = dispatch_logdet(problem, config=config)
    assert result.level == 2
    assert result.method == "state-space recursion"
    assert [item.level for item in result.rejected] == [0, 1]
    assert all(item.satisfied is False and item.reason for item in result.rejected)
    assert result.value == pytest.approx(_oracle(lam + perturbation), rel=2e-13)


def test_dispatcher_executes_level_seven_with_fewer_probes_than_dimension():
    """The dispatcher's distinct frozen-Hutchinson payload must be reachable."""
    n = 16
    rng = np.random.default_rng(711)
    widths = np.linspace(1.3, 2.3, n)
    lam = np.diag(widths**2)
    basis, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigenvalues = np.linspace(0.01, 0.05, n)
    perturbation = (
        np.diag(widths) @ basis @ np.diag(eigenvalues) @ basis.T @ np.diag(widths)
    )
    probes = FrozenProbes(rng.choice((-1.0, 1.0), size=(2, n)))
    certificate = certify_warmup_rho(
        [0.05], margin=0.0, tolerance=1.0e-6, multiplicity=n
    )
    problem = LogDetProblem(
        lam,
        perturbation,
        frozen_probes=probes,
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )
    disabled = LadderConfig(
        low_rank_max=0,
        dense_max_n=0,
        finite_max_n=0,
        finite_max_rank=0,
    )

    result = dispatch_logdet(problem, config=disabled)
    assert result.level == 7
    assert result.method == "frozen Hutchinson trace-log"
    assert _relative(result.value, _oracle(lam + perturbation)) < 2e-2


@pytest.mark.parametrize("probes_count", [1, 2, 16])
def test_rungs_six_and_seven_share_order_and_rho_without_sharing_error_claims(
    probes_count,
):
    """Both approximation payloads are directly exercised at the same boundary."""
    n, rho = 16, 0.05
    rng = np.random.default_rng(600 + probes_count)
    widths = np.linspace(1.3, 2.3, n)
    lam = np.diag(widths**2)
    basis, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigenvalues = np.linspace(0.01, rho, n)
    perturbation = (
        np.diag(widths) @ basis @ np.diag(eigenvalues) @ basis.T @ np.diag(widths)
    )
    certificate = float(
        np.nextafter(spectral_radius(lam, perturbation), np.inf)
    )
    order = choose_trace_order(certificate, 1.0e-7, multiplicity=n)
    traces = _independent_power_traces(lam, perturbation, order)
    probes = FrozenProbes(rng.choice((-1.0, 1.0), size=(probes_count, n)))
    exact_trace = truncated_trace_logdet(
        lam,
        perturbation,
        exact_power_traces=traces,
        order=order,
        rho=certificate,
    )
    frozen = frozen_hutchinson_trace_logdet(
        lam, perturbation, probes, order=order, rho=certificate
    )
    oracle = _oracle(lam + perturbation)
    bound = n * certificate ** (order + 1) / (
        (order + 1) * (1.0 - certificate)
    )

    assert abs(exact_trace - oracle) <= bound * (1.0 + 1e-8) + 2e-13
    assert _relative(frozen, oracle) < 2e-2


def test_public_premise_checker_reports_real_domain_evidence_without_running_a_method():
    """A failed direct-method domain is a verdict, not an arithmetic exception."""
    sigma = np.array([[2.4, 0.3, 0.0], [0.1, 2.1, 0.2], [0.0, 0.2, 3.2]])
    lam = 1.3 * np.eye(3)
    problem = LogDetProblem(lam, sigma - lam, chain_block_size=1)
    verdicts = check_logdet_premises(problem)
    assert len(verdicts) == 9
    assert [verdict.level for verdict in verdicts] == list(range(9))
    assert verdicts[2].satisfied is False
    assert verdicts[4].details["symmetric"] is False
    assert verdicts[-1].satisfied is False
    with pytest.raises(ValueError, match="symmetric"):
        state_space_logdet(sigma, block_size=1)


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (LadderConfig(low_rank_max=2, low_rank_fraction=1.0), (True, True, False)),
        (LadderConfig(low_rank_max=8, low_rank_fraction=0.25), (True, True, False)),
    ],
)
def test_low_rank_max_and_fraction_boundaries_evaluate_both_adjacent_methods(
    rank, config, expected
):
    """Rows 1 and 2 agree below, at, and above both low-rank thresholds."""
    n = 8
    widths = np.linspace(1.3, 2.4, n)
    lam = np.diag(widths**2)
    left = np.zeros((n, rank))
    left[np.arange(rank), np.arange(rank)] = np.linspace(0.04, 0.08, rank)
    factors = LowRankFactors(left)
    perturbation = left @ left.T
    sigma = lam + perturbation
    oracle = _oracle(sigma)
    low = low_rank_logdet(lam, perturbation, factors=factors)
    chain = state_space_logdet(sigma, block_size=1)
    verdict = check_logdet_premises(
        LogDetProblem(lam, perturbation, low_rank_factors=factors), config=config
    )[1]
    assert verdict.satisfied is expected[rank - 1]
    assert math.isfinite(low) and math.isfinite(chain)
    assert _relative(low, oracle) < 2e-13
    assert _relative(chain, oracle) < 2e-13
    assert _relative(low, chain) < 2e-13


@pytest.mark.parametrize("n", [2, 3, 4])
def test_dense_size_boundary_evaluates_dense_and_finite_methods(n):
    """Rows 4 and 5 agree immediately below, at, and above dense_max_n=3."""
    widths = np.linspace(1.4, 2.3, n)
    lam = np.diag(widths**2)
    perturbation = 0.07 * lam
    sigma = lam + perturbation
    oracle = _oracle(sigma)
    dense = dense_cholesky_logdet(sigma)
    finite = finite_perturbation_logdet(lam, perturbation)
    verdict = check_logdet_premises(
        LogDetProblem(lam, perturbation), config=LadderConfig(dense_max_n=3)
    )[4]
    assert verdict.satisfied is (n <= 3)
    assert math.isfinite(dense) and math.isfinite(finite)
    assert _relative(dense, oracle) < 2e-13
    assert _relative(finite, oracle) < 2e-13
    assert _relative(dense, finite) < 2e-13


def test_dense_condition_gate_refuses_a_non_diagonal_unresolved_matrix():
    """Dropping rung 4's 1/eps gate admits this cond >= 1e15 fixture."""
    correlation = np.nextafter(1.0, 0.0)
    sigma = 2.3 * np.array(
        [[1.0, correlation], [correlation, 1.0]]
    )
    lam = 0.5 * np.eye(2)
    problem = LogDetProblem(lam, sigma - lam)
    verdict = check_logdet_premises(problem)[4]

    assert verdict.details["condition"] >= 1.0e15
    assert verdict.details["condition"] >= verdict.details["condition_ceiling"]
    assert verdict.satisfied is False
    with pytest.raises(ResamplingRefused):
        dispatch_logdet(problem)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_finite_trace_boundary_compares_both_direct_methods(n):
    """Exact and truncated routes must agree within the certified bound around n=T."""
    rho, order = 0.01, 5
    widths = np.linspace(1.4, 2.2, n)
    lam = np.diag(widths**2)
    perturbation = rho * lam
    certificate = float(
        np.nextafter(spectral_radius(lam, perturbation), np.inf)
    )
    traces = _independent_power_traces(lam, perturbation, order)
    exact = finite_perturbation_logdet(lam, perturbation)
    approximate = truncated_trace_logdet(
        lam,
        perturbation,
        exact_power_traces=traces,
        order=order,
        rho=certificate,
    )
    bound = n * certificate ** (order + 1) / (
        (order + 1) * (1 - certificate)
    )
    assert math.isfinite(exact) and math.isfinite(approximate)
    assert abs(exact - approximate) <= bound * (1 + 1e-8) + 1e-14
    assert _relative(exact, approximate) < 1e-12
    verdict = check_logdet_premises(
        LogDetProblem(
            lam,
            perturbation,
            exact_power_traces=traces,
            trace_order=order,
            certified_rho=certificate,
        ),
        config=LadderConfig(finite_max_n=3, finite_max_rank=0),
    )[5]
    assert verdict.satisfied is (n <= 3)
    oracle = _oracle(lam + perturbation)
    assert _relative(exact, oracle) < 2e-13
    assert abs(approximate - oracle) <= bound * (1 + 1e-8) + 1e-14


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_finite_rank_boundary_evaluates_finite_and_trace_methods(rank):
    """Rows 5 and 6 agree within the bound around finite_max_rank=2."""
    n, order = 8, 8
    widths = np.linspace(1.3, 2.4, n)
    lam = np.diag(widths**2)
    left = np.zeros((n, rank))
    left[np.arange(rank), np.arange(rank)] = np.linspace(0.03, 0.07, rank)
    factors = LowRankFactors(left)
    perturbation = left @ left.T
    certificate = float(
        np.nextafter(spectral_radius(lam, perturbation), np.inf)
    )
    traces = _independent_power_traces(lam, perturbation, order)
    exact = finite_perturbation_logdet(lam, perturbation, factors=factors)
    approximate = truncated_trace_logdet(
        lam,
        perturbation,
        exact_power_traces=traces,
        order=order,
        rho=certificate,
    )
    oracle = _oracle(lam + perturbation)
    bound = n * certificate ** (order + 1) / (
        (order + 1) * (1.0 - certificate)
    )
    verdict = check_logdet_premises(
        LogDetProblem(
            lam,
            perturbation,
            low_rank_factors=factors,
            exact_power_traces=traces,
            trace_order=order,
            certified_rho=certificate,
        ),
        config=LadderConfig(finite_max_n=0, finite_max_rank=2),
    )[5]
    assert verdict.satisfied is (rank <= 2)
    assert math.isfinite(exact) and math.isfinite(approximate)
    assert _relative(exact, oracle) < 2e-13
    assert abs(approximate - oracle) <= bound * (1 + 1e-8) + 1e-14


def test_trace_order_boundary_directly_evaluates_both_neighboring_orders():
    """The selected order and both neighbors are checked against the oracle."""
    n, rho, tolerance = 5, 0.5, 1e-4
    lam = np.diag(np.linspace(1.3, 2.4, n) ** 2)
    perturbation = rho * lam
    selected = choose_trace_order(rho, tolerance, multiplicity=n)
    traces = _independent_power_traces(lam, perturbation, selected + 1)
    oracle = _oracle(lam + perturbation)
    errors = []
    for order in (selected - 1, selected, selected + 1):
        value = truncated_trace_logdet(
            lam,
            perturbation,
            exact_power_traces=traces,
            order=order,
            rho=rho,
        )
        bound = n * rho ** (order + 1) / ((order + 1) * (1.0 - rho))
        assert math.isfinite(value)
        assert abs(value - oracle) <= bound * (1 + 1e-10) + 2e-14
        errors.append(abs(value - oracle))
    assert errors[2] < errors[1] < errors[0]


def test_compact_diagonal_premises_span_four_orders_of_n_and_rank():
    """Premise arithmetic must remain viable at n/rank extremes."""
    for size in (1, 10, 100, 10_000):
        lam = np.full(size, 1.7)
        perturbation = np.full(size, 0.13)
        verdicts = check_logdet_premises(LogDetProblem(lam, perturbation))
        assert verdicts[0].details["n"] == size
        assert verdicts[1].details["rank"] == size


@pytest.mark.parametrize("size", [1, 10, 100, 10_000])
def test_compact_diagonal_direct_newton_spans_four_orders_of_n_and_rank(size):
    """The four-order claim executes Newton and its adjacent dense payload."""
    lam = np.linspace(1.3, 2.3, size) ** 2
    perturbation = 1.0e-5 * lam
    sigma = lam + perturbation
    newton = finite_perturbation_logdet(lam, perturbation)
    dense = dense_cholesky_logdet(sigma)
    oracle = math.fsum(
        _oracle(np.diag(sigma[start : start + 64]))
        for start in range(0, size, 64)
    )

    assert math.isfinite(newton) and math.isfinite(dense)
    assert _relative(newton, oracle) < 2e-13
    assert _relative(dense, oracle) < 2e-13
    assert _relative(newton, dense) < 2e-13


def test_kronecker_descriptor_is_verified_not_trusted():
    """A factor payload inconsistent with Sigma must be rejected."""
    factor = np.diag([1.3, 2.2])
    matrix = np.kron(factor, factor)
    wrong = KroneckerStructure((factor, np.diag([1.3, 2.3])))
    with pytest.raises(ValueError, match="factors do not reconstruct"):
        structured_logdet(matrix, kind="kronecker", structure=wrong)
