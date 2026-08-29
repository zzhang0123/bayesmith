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
        [0.37], margin=0.0, tolerance=1e-3, multiplicity=3
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
        [0.2], margin=0.0, tolerance=1e-3, multiplicity=2
    )
    problem = LogDetProblem(
        lam,
        perturbation,
        frozen_probes=probes,
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )
    plan = make_frozen_trace_log_plan(problem, certificate)
    base = jnp.log(1.7) + jnp.log(2.6)

    def evaluate(rho):
        return plan(base, rho * jnp.eye(2))

    value = jax.jit(evaluate)(jnp.array(0.2))
    gradient = jax.jit(jax.grad(evaluate))(jnp.array(0.2))
    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)


def test_validated_runtime_plans_capture_order_and_frozen_probes():
    """Runtime callers cannot lower order or swap/redraw probes per evaluation."""
    import bayesmith.marginal.logdet as module

    certificate = certify_warmup_rho(
        [0.2], margin=0.05, tolerance=1e-6, multiplicity=2
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
    right = np.column_stack((np.linspace(0.04, 0.08, 8), np.linspace(0.03, 0.06, 8)))
    perturbation = left @ right.T
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
    with pytest.raises(ValueError, match="exactly reconstruct"):
        low_rank_logdet(lam, perturbation, factors=factors)
    with pytest.raises(ValueError, match="exactly reconstruct"):
        finite_perturbation_logdet(lam, perturbation, factors=factors)

    safe_exact = finite_perturbation_logdet(lam, perturbation)
    assert safe_exact == pytest.approx(_oracle(sigma), rel=2e-13)
    result = dispatch_logdet(problem, config=config)
    assert result.level == 6
    assert result.value == pytest.approx(_oracle(sigma), rel=2e-10)


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


def test_false_structure_claims_are_numerically_refused():
    """A bare structure label must not route an unstructured matrix."""
    matrix = np.array([[2.4, 0.3, 0.05], [0.3, 2.1, 0.2], [0.05, 0.2, 3.2]])
    with pytest.raises(ValueError, match="not circulant"):
        structured_logdet(matrix, kind="circulant")
    with pytest.raises(ValueError, match="not a block chain"):
        state_space_logdet(matrix, block_size=1)


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
    sigma = np.array([[2.4, 0.3, 0.0], [0.1, 2.1, 0.2], [0.0, 0.2, 3.2]])
    lam = 1.3 * np.eye(3)
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

    left = np.array([[0.2], [1.0e-11]])
    invalid_factors = LowRankFactors(left)
    represented = left @ left.T
    tiny_lam = np.diag([0.1, 1.0e-20])
    with_residual = represented + np.diag([0.0, 5.0e-21])
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
    exact = finite_perturbation_logdet(lam, perturbation)
    oracle = _oracle(lam + perturbation)
    assert math.isfinite(exact)
    assert _relative(exact, oracle) < 2e-13
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
    assert warmup.order == choose_trace_order(0.48, 1e-6, multiplicity=5)
    audit = audit_retained_rho([0.3, 0.49, 0.51], warmup)
    assert audit.passed is False
    assert audit.violations == (1, 2)


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
        finite = finite_perturbation_logdet(lam, perturbation)
        assert math.isfinite(dense) and math.isfinite(finite)
        assert _relative(dense, oracle) < 2e-13
        assert _relative(finite, oracle) < 2e-13
        assert _relative(dense, finite) < 2e-13


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


def test_kronecker_descriptor_is_verified_not_trusted():
    """A factor payload inconsistent with Sigma must be rejected."""
    factor = np.diag([1.3, 2.2])
    matrix = np.kron(factor, factor)
    wrong = KroneckerStructure((factor, np.diag([1.3, 2.3])))
    with pytest.raises(ValueError, match="factors do not reconstruct"):
        structured_logdet(matrix, kind="kronecker", structure=wrong)
