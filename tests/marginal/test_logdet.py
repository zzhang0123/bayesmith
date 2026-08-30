"""Independent oracles and boundary tests for the log-determinant ladder.

Every exact expected value is NumPy's dense ``slogdet`` on a matrix assembled
in this file.  Approximation bounds are scalar formulas written here rather
than calls back into the implementation.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.exact.fisher import condition_ceiling
from bayesmith.marginal.logdet import (
    FrozenProbes,
    KroneckerStructure,
    LadderConfig,
    LogDetProblem,
    LowRankFactors,
    ResamplingRefused,
    audit_retained_lambda_logdet,
    audit_retained_operator_norm,
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
    with pytest.raises(ValueError, match=r"Lambda\^-1-amplified"):
        low_rank_logdet(lam, perturbation, factors=factors)
    with pytest.raises(ValueError, match=r"Lambda\^-1-amplified"):
        finite_perturbation_logdet(lam, perturbation, factors=factors)

    with pytest.raises(ValueError, match="condition"):
        finite_perturbation_logdet(lam, perturbation)
    result = dispatch_logdet(problem, config=config)
    assert result.level == 6
    assert result.value == pytest.approx(_oracle(sigma), rel=2e-10)


@pytest.mark.parametrize(
    ("n", "rank"),
    [(20, 2), (50, 4), (12, 9), (37, 6), (101, 7), (200, 8), (300, 5), (64, 3)],
)
def test_self_factors_reconstruct_on_all_reproduced_blas_shapes(n, rank):
    """Copying the same factor into two buffers must not invent a residual."""
    rng = np.random.default_rng(4000 + n + rank)
    left = 0.01 * rng.normal(size=(n, rank))
    perturbation = left @ left.T
    factors = LowRankFactors(left)
    lam = np.linspace(1.3, 2.3, n) ** 2
    problem = LogDetProblem(lam, np.diag(perturbation), low_rank_factors=None)
    dense_problem = LogDetProblem(
        np.diag(lam), perturbation, low_rank_factors=factors
    )

    assert np.array_equal(factors.left @ factors.right.T, perturbation)
    assert check_logdet_premises(
        dense_problem,
        config=LadderConfig(low_rank_max=rank, low_rank_fraction=1.0),
    )[1].satisfied
    value = low_rank_logdet(np.diag(lam), perturbation, factors=factors)
    assert _relative(value, _oracle(np.diag(lam) + perturbation)) < 2e-12
    assert check_logdet_premises(problem)[1].details["rank"] == n


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


def test_low_rank_factor_check_rejects_cancellation_inflated_roundoff_envelope():
    """Huge cancelling factors cannot certify an unrelated low-rank matrix."""
    lam = np.eye(3)
    perturbation = 0.5 * np.eye(3)
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
    basis, _ = np.linalg.qr(np.random.default_rng(123).normal(size=(3, 3)))
    eigenvalues = np.array([-1.0, 0.5])
    left = basis[:, :2] * np.sqrt(np.abs(eigenvalues))
    right = basis[:, :2] * np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))
    factors = LowRankFactors(left, right)
    perturbation = left @ right.T + 5.0e-16 * np.eye(3)
    lam = np.eye(3)

    verdict = check_logdet_premises(
        LogDetProblem(lam, perturbation, low_rank_factors=factors),
        config=LadderConfig(low_rank_max=2, low_rank_fraction=1.0),
    )[1]
    assert verdict.satisfied is False
    with pytest.raises(ValueError, match="do not exactly reconstruct"):
        low_rank_logdet(lam, perturbation, factors=factors)


def test_newton_exact_rungs_refuse_adversarial_spectral_spread_before_dispatch():
    """Cancellation in degree-16 Newton identities must not win over row 4."""
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
    assert verdicts[1].satisfied is False
    assert "stability" in verdicts[1].reason
    assert verdicts[5].satisfied is False
    with pytest.raises(ValueError, match="stability"):
        low_rank_logdet(lam, perturbation, factors=factors)
    with pytest.raises(ValueError, match="stability"):
        finite_perturbation_logdet(lam, perturbation, factors=factors)

    result = dispatch_logdet(problem)
    oracle = _oracle(sigma)
    assert result.level == 4
    assert result.value == pytest.approx(oracle, rel=2e-13)


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
def test_newton_stability_boundary_evaluates_low_rank_and_dense_directly(
    rank, rho
):
    """Rank <= 8 must never override an expansive-spectrum refusal."""
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
    if rho <= 1.0:
        low = low_rank_logdet(lam, perturbation, factors=factors)
        assert math.isfinite(low)
        assert _relative(low, oracle) < 2e-11
        assert _relative(low, dense) < 2e-11
    else:
        with pytest.raises(ValueError, match="stability"):
            low_rank_logdet(lam, perturbation, factors=factors)


def test_rank_eight_expansive_regression_cannot_dispatch_as_exact():
    """The reproduced C1 matrix was wrong by 22% when rank eight escaped rho."""
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
    assert verdict.satisfied is False
    result = dispatch_logdet(problem)
    assert result.level == 4
    assert _relative(result.value, _oracle(lam + perturbation)) < 2e-11


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
    lam = np.eye(4)
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
    """A structure label cannot bypass the common exact-arithmetic condition gate."""
    link = np.nextafter(1.0 / math.sqrt(2.0), 0.0)
    sigma = np.array([[1.0, link, 0.0], [link, 1.0, link], [0.0, link, 1.0]])
    problem = LogDetProblem(
        0.5 * np.eye(3),
        sigma - 0.5 * np.eye(3),
        structure_kind="toeplitz",
    )

    verdict = check_logdet_premises(problem)[3]
    assert verdict.satisfied is False
    assert "condition" in verdict.reason
    with pytest.raises(ValueError, match="condition"):
        structured_logdet(sigma, kind="toeplitz")


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
    first = frozen_hutchinson_trace_logdet(lam, perturbation, probes, order=8, rho=0.2)
    public = probes.values
    try:
        public.setflags(write=True)
        public[0, 0] = -17.0
    except ValueError:
        pass
    second = frozen_hutchinson_trace_logdet(lam, perturbation, probes, order=8, rho=0.2)
    assert first == second
    assert first == pytest.approx(_oracle(lam + perturbation), rel=3e-6)


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
    """A float64 tail certificate cannot promise accuracy unavailable to JAX float32."""
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
        with pytest.raises(ValueError, match="enable_x64"):
            make_trace_log_plan(trace_problem, certificate)
        with pytest.raises(ValueError, match="enable_x64"):
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

    with jax.enable_x64(False), pytest.raises(ValueError, match="requires float64"):
        make_trace_log_plan(problem, certificate)


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
    n, rho, tolerance = 10_000, 1.0e-8, 1.0e-12
    lam = np.ones(n)
    perturbation = rho * lam
    certificate = certify_warmup_rho(
        [rho],
        margin=0.0,
        tolerance=tolerance,
        multiplicity=n,
        lambda_logdets=[0.0],
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
    audit = audit_retained_lambda_logdet([0.0, 50.0 * n], certificate)
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
        lambda_logdets=[0.0, 50.0 * n],
    )
    with jax.enable_x64(True), pytest.raises(ValueError, match="roundoff"):
        make_trace_log_plan(problem, wide)


def test_frozen_runtime_requires_x64_for_nonnormal_matmul_cancellation():
    """Float32 loses diagonal terms when huge probe products cancel."""
    lam = np.eye(2)
    x_matrix = np.array([[0.1, 1.0e8], [0.0, 0.2]])
    probes = FrozenProbes([[1.0, 1.0], [1.0, -1.0]])
    certificate = certify_warmup_rho(
        [0.2],
        margin=0.0,
        tolerance=1.0e-3,
        multiplicity=2,
        lambda_logdets=[0.0],
        x_operator_norms=[np.linalg.norm(np.abs(x_matrix), ord=2)],
    )
    problem = LogDetProblem(
        lam,
        x_matrix,
        frozen_probes=probes,
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    with jax.enable_x64(False), pytest.raises(ValueError, match="requires float64"):
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
        "lambda_logdets": [0.0],
    }
    understated = certify_warmup_rho(
        **problem_data, x_operator_norms=[ordinary_norm]
    )
    problem = LogDetProblem(
        np.eye(n),
        x_matrix,
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
def test_spectral_radius_directly_matches_diagonal_and_dense_nonnormal_oracles(rho):
    """Strict-rho safety is tested at its measurement function, not only downstream."""
    lam = np.array([1.7, 2.6])
    assert spectral_radius(lam, rho * lam) == pytest.approx(rho, rel=2e-15)

    transform = np.array([[1.0, 1000.0], [0.0, 1.0]])
    x_matrix = transform @ np.diag([rho, 0.2 * rho]) @ np.linalg.inv(transform)
    dense_lam = np.diag([1.7, 2.6])
    perturbation = x_matrix @ dense_lam
    oracle = float(np.max(np.abs(np.linalg.eigvals(x_matrix))))
    assert spectral_radius(dense_lam, perturbation) == pytest.approx(
        oracle, rel=2e-13, abs=2e-13
    )


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
    order = choose_trace_order(rho, 1.0e-7, multiplicity=n)
    traces = _independent_power_traces(lam, perturbation, order)
    probes = FrozenProbes(rng.choice((-1.0, 1.0), size=(probes_count, n)))
    exact_trace = truncated_trace_logdet(
        lam,
        perturbation,
        exact_power_traces=traces,
        order=order,
        rho=rho,
    )
    frozen = frozen_hutchinson_trace_logdet(
        lam, perturbation, probes, order=order, rho=rho
    )
    oracle = _oracle(lam + perturbation)
    bound = n * rho ** (order + 1) / ((order + 1) * (1.0 - rho))

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


def test_dense_condition_boundary_uses_dtype_specific_ceiling():
    """A fixed condition ceiling accepts float32 matrices that lost half their digits."""
    ceiling = condition_ceiling(np.dtype(np.float64))
    for ratio in (1.0 - 1e-6, 1.0, 1.0 + 1e-6):
        sigma = np.diag([2.3, 2.3 * ceiling * ratio])
        lam = 0.8 * sigma
        perturbation = sigma - lam
        problem = LogDetProblem(lam, perturbation)
        verdict = check_logdet_premises(problem)[4]
        assert verdict.satisfied is (ratio < 1.0)
        oracle = _oracle(sigma)
        dense = dense_cholesky_logdet(sigma)
        assert math.isfinite(dense)
        assert _relative(dense, oracle) < 2e-13
        if ratio < 1.0:
            finite = finite_perturbation_logdet(lam, perturbation)
            assert math.isfinite(finite)
            assert _relative(finite, oracle) < 2e-13
            assert _relative(dense, finite) < 2e-13
        else:
            with pytest.raises(ValueError, match="condition"):
                finite_perturbation_logdet(lam, perturbation)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_finite_trace_boundary_compares_both_direct_methods(n):
    """Exact and truncated routes must agree within the certified bound around n=T."""
    rho, order = 0.01, 5
    widths = np.linspace(1.4, 2.2, n)
    lam = np.diag(widths**2)
    perturbation = rho * lam
    traces = _independent_power_traces(lam, perturbation, order)
    exact = finite_perturbation_logdet(lam, perturbation)
    approximate = truncated_trace_logdet(
        lam, perturbation, exact_power_traces=traces, order=order, rho=rho
    )
    bound = n * rho ** (order + 1) / ((order + 1) * (1 - rho))
    assert math.isfinite(exact) and math.isfinite(approximate)
    assert abs(exact - approximate) <= bound * (1 + 1e-8) + 1e-14
    assert _relative(exact, approximate) < 1e-12
    verdict = check_logdet_premises(
        LogDetProblem(
            lam,
            perturbation,
            exact_power_traces=traces,
            trace_order=order,
            certified_rho=rho,
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
    rho = float(np.max(np.diag(perturbation) / np.diag(lam)))
    traces = _independent_power_traces(lam, perturbation, order)
    exact = finite_perturbation_logdet(lam, perturbation, factors=factors)
    approximate = truncated_trace_logdet(
        lam, perturbation, exact_power_traces=traces, order=order, rho=rho
    )
    oracle = _oracle(lam + perturbation)
    bound = n * rho ** (order + 1) / ((order + 1) * (1.0 - rho))
    verdict = check_logdet_premises(
        LogDetProblem(
            lam,
            perturbation,
            low_rank_factors=factors,
            exact_power_traces=traces,
            trace_order=order,
            certified_rho=rho,
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
