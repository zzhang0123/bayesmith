"""Independent oracles and boundary tests for the log-determinant ladder.

Every exact expected value is NumPy's dense ``slogdet`` on a matrix assembled
in this file.  Approximation bounds are scalar formulas written here rather
than calls back into the implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bayesmith.exact.fisher import condition_ceiling
from bayesmith.marginal.logdet import (
    FrozenProbes,
    KroneckerStructure,
    LadderConfig,
    LogDetProblem,
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


@pytest.mark.parametrize("rho", [0.01, 0.5, 0.9, 0.99])
@pytest.mark.parametrize("n", [1, 10, 100, 1000])
def test_trace_log_grid_obeys_the_whole_trace_bound(rho, n):
    """Omitting eigenvalue multiplicity fails on X=rho I_n."""
    widths = np.linspace(1.2, 2.3, n)
    lam_diag = widths**2
    perturbation_diag = rho * lam_diag
    order = 12 if rho <= 0.5 else (120 if rho == 0.9 else 1200)
    traces = np.array([n * rho**power for power in range(1, order + 1)])
    got = truncated_trace_logdet(
        lam_diag,
        perturbation_diag,
        exact_power_traces=traces,
        order=order,
    )
    want = float(np.sum(np.log(lam_diag + perturbation_diag)))
    bound = n * rho ** (order + 1) / ((order + 1) * (1.0 - rho))
    assert abs(got - want) <= bound * (1.0 + 1e-10) + 2e-12


def test_scalar_and_whole_trace_tail_bounds_are_distinct():
    """Returning the scalar bound for a trace silently understates error."""
    scalar = 0.5**7 / (7 * 0.5)
    assert trace_log_tail_bound(0.5, 6) == pytest.approx(scalar)
    assert whole_trace_log_tail_bound(0.5, 6, 13) == pytest.approx(13 * scalar)


@pytest.mark.parametrize("rho", [0.99, 1.0, 1.01])
def test_strict_rho_boundary_refuses_one_and_above(rho):
    """Changing rho<1 to rho<=1 admits a divergent series."""
    lam = np.diag([1.3, 2.2])
    perturbation = rho * lam
    if rho < 1.0:
        truncated_trace_logdet(
            lam, perturbation, exact_power_traces=[2 * rho], order=1, rho=rho
        )
    else:
        with pytest.raises(ValueError, match="rho < 1"):
            truncated_trace_logdet(
                lam, perturbation, exact_power_traces=[2 * rho], order=1, rho=rho
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
    first = frozen_hutchinson_trace_logdet(lam, perturbation, probes, order=8)
    second = frozen_hutchinson_trace_logdet(lam, perturbation, probes, order=8)
    assert first == second
    assert first == pytest.approx(_oracle(lam + perturbation), rel=3e-6)


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


def test_public_premise_checker_does_no_logdet_arithmetic(monkeypatch):
    """A premise-only call must not accidentally pay for a determinant."""
    lam, perturbation = _spd_fixture(4)
    monkeypatch.setattr(
        np.linalg,
        "slogdet",
        lambda *_: (_ for _ in ()).throw(AssertionError("computed")),
    )
    monkeypatch.setattr(
        np.linalg,
        "cholesky",
        lambda *_: (_ for _ in ()).throw(AssertionError("computed")),
    )
    verdicts = check_logdet_premises(LogDetProblem(lam, perturbation))
    assert len(verdicts) == 9
    assert [verdict.level for verdict in verdicts] == list(range(9))
    assert verdicts[-1].satisfied is False


def test_dense_condition_boundary_uses_dtype_specific_ceiling():
    """A fixed condition ceiling accepts float32 matrices that lost half their digits."""
    ceiling = condition_ceiling(np.dtype(np.float64))
    for ratio in (1.0 - 1e-6, 1.0, 1.0 + 1e-6):
        matrix = np.diag([2.3, 2.3 * ceiling * ratio])
        problem = LogDetProblem(matrix, np.zeros_like(matrix))
        verdict = check_logdet_premises(problem)[4]
        assert verdict.satisfied is (ratio < 1.0)
        assert math.isfinite(dense_cholesky_logdet(matrix))


@pytest.mark.parametrize("n", [2, 3, 4])
def test_finite_trace_boundary_compares_both_direct_methods(n):
    """Exact and truncated routes must agree within the certified bound around n=T."""
    rho, order = 0.01, 5
    widths = np.linspace(1.4, 2.2, n)
    lam = np.diag(widths**2)
    perturbation = rho * lam
    traces = [n * rho**power for power in range(1, order + 1)]
    exact = finite_perturbation_logdet(lam, perturbation)
    approximate = truncated_trace_logdet(
        lam, perturbation, exact_power_traces=traces, order=order, rho=rho
    )
    bound = n * rho ** (order + 1) / ((order + 1) * (1 - rho))
    assert math.isfinite(exact) and math.isfinite(approximate)
    assert abs(exact - approximate) <= bound * (1 + 1e-8) + 1e-14
    assert _relative(exact, approximate) < 1e-12


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
