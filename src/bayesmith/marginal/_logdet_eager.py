"""Eager data types and direct numerical methods for the logdet ladder.

Labels such as ``"circulant"`` and a chain block size are requests to check
structure, not claims a direct method trusts.  The premise checker and
dispatcher live in :mod:`bayesmith.marginal._logdet_ladder`; pure JAX kernels
live in :mod:`bayesmith.marginal._logdet_runtime`.

Write ``Sigma = Lambda + P`` and ``X = P Lambda^-1``.  ``Lambda`` is an
explicit preconditioner design choice, not necessarily the noise covariance.
Foreground-dominated ``P`` can make the direct trace-log expansion diverge;
choosing a diagonal, block-diagonal, or circulant ``Lambda`` closer to
``Sigma`` can reduce ``rho(X)``.  This module never hides a preconditioner
selector.

The finite exact route evaluates the elementary-symmetric polynomial in a
stable factored form.  When ``rank(P) = k``, all ``e_j(X)`` above ``k`` vanish,
so the low-rank determinant lemma is precisely the sparse representation of
the same finite polynomial rather than an unrelated implementation.

This module contains no traced execution.  Its public direct methods remain
available so boundary tests can compare both sides of every dispatch threshold.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Sequence
from typing import Any, Literal

import numpy as np

from bayesmith.exact.fisher import condition_ceiling

__all__ = [
    "FrozenProbes",
    "KroneckerStructure",
    "LadderConfig",
    "LadderResult",
    "LowRankFactors",
    "LogDetProblem",
    "PremiseVerdict",
    "ResamplingRefused",
    "choose_trace_order",
    "dense_cholesky_logdet",
    "finite_perturbation_logdet",
    "frozen_hutchinson_trace_logdet",
    "lambda_logdet",
    "low_rank_logdet",
    "resampled_trace_logdet",
    "spectral_radius",
    "state_space_logdet",
    "structured_logdet",
    "trace_log_tail_bound",
    "truncated_trace_logdet",
    "whole_trace_log_tail_bound",
]



class ResamplingRefused(RuntimeError):
    """A per-call random logdet was requested for an HMC target."""

    def __init__(self, message: str, *, rejected: Sequence[PremiseVerdict] = ()):
        super().__init__(message)
        self.rejected = tuple(rejected)


@dataclasses.dataclass(frozen=True)
class KroneckerStructure:
    """Factors that must reconstruct the supplied dense matrix numerically."""

    factors: tuple[np.ndarray, ...]

    def __init__(self, factors: Iterable[Any]):
        copied = tuple(_read_only_array(factor, ndim=2) for factor in factors)
        if not copied:
            raise ValueError("KroneckerStructure needs at least one factor")
        if any(factor.shape[0] == 0 or factor.shape[0] != factor.shape[1] for factor in copied):
            raise ValueError("Kronecker factors must be non-empty square matrices")
        object.__setattr__(self, "factors", copied)


@dataclasses.dataclass(frozen=True)
class FrozenProbes:
    """Immutable common-random-number probes reused on every evaluation.

    The backing store is ``bytes``, not a NumPy array with a reversible
    ``writeable=False`` flag. :attr:`values` exposes a fresh read-only view,
    so neither source mutation nor changing flags on a returned array can
    alter later evaluations.
    """

    _payload: bytes = dataclasses.field(repr=False)
    shape: tuple[int, int]
    dtype: str

    def __init__(self, values: Any):
        array = _read_only_array(values, ndim=2)
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("FrozenProbes needs a non-empty (probes, n) array")
        if not np.all(np.isfinite(array)):
            raise ValueError("FrozenProbes values must all be finite")
        contiguous = np.ascontiguousarray(array)
        object.__setattr__(self, "_payload", contiguous.tobytes())
        object.__setattr__(self, "shape", tuple(contiguous.shape))
        object.__setattr__(self, "dtype", contiguous.dtype.str)

    @property
    def values(self) -> np.ndarray:
        """A read-only view whose immutable bytes cannot be made writeable."""
        return np.frombuffer(self._payload, dtype=np.dtype(self.dtype)).reshape(
            self.shape
        )


@dataclasses.dataclass(frozen=True)
class LowRankFactors:
    """Algebraic evidence ``P = left @ right.T`` with a fixed column count.

    A factorisation proves ``rank(P) <= k`` without a scale-dependent SVD
    tolerance. Dependent columns only make the bound conservative: the finite
    e-polynomial still terminates by degree ``k``.
    """

    left: np.ndarray
    right: np.ndarray

    def __init__(self, left: Any, right: Any | None = None):
        left_array = _read_only_array(left, ndim=2)
        right_array = (
            left_array
            if right is None
            else _read_only_array(right, ndim=2)
        )
        if left_array.shape[1] != right_array.shape[1]:
            raise ValueError("low-rank factors must have the same column count")
        object.__setattr__(self, "left", left_array)
        object.__setattr__(self, "right", right_array)

    @property
    def rank_bound(self) -> int:
        """The algebraic termination degree proved by the factor widths."""
        return int(self.left.shape[1])


@dataclasses.dataclass(frozen=True)
class LadderConfig:
    """Numerical thresholds used by the eager premise checker."""

    low_rank_max: int = 64
    low_rank_fraction: float = 0.125
    dense_max_n: int = 256
    finite_max_n: int = 32
    finite_max_rank: int = 128
    structure_rtol: float = 1e-11
    structure_atol: float = 1e-13

    def __post_init__(self) -> None:
        integer_fields = (
            self.low_rank_max,
            self.dense_max_n,
            self.finite_max_n,
            self.finite_max_rank,
        )
        if any(value < 0 for value in integer_fields):
            raise ValueError("ladder size and rank thresholds must be non-negative")
        if not 0.0 <= self.low_rank_fraction <= 1.0:
            raise ValueError("low_rank_fraction must lie in [0, 1]")


@dataclasses.dataclass(frozen=True)
class LogDetProblem:
    """Checkable inputs for ``Sigma = Lambda + perturbation``.

    One-dimensional inputs encode diagonal matrices compactly.  They let
    premise-only checks cover large ``n`` and verified ranks without forming
    a quadratic-size fixture.  ``exact_power_traces[r-1]`` must be the exact
    deterministic ``Tr(X**r)``; a generic matvec cannot provide this and is
    therefore not accepted by rung 6.
    """

    lambda_matrix: np.ndarray
    perturbation: np.ndarray
    chain_block_size: int | None = None
    structure_kind: Literal["diagonal", "circulant", "toeplitz", "kronecker"] | None = (
        None
    )
    structure: KroneckerStructure | None = None
    low_rank_factors: LowRankFactors | None = None
    exact_power_traces: tuple[float, ...] | None = None
    frozen_probes: FrozenProbes | None = None
    trace_order: int | None = None
    certified_rho: float | None = None

    def __init__(
        self,
        lambda_matrix: Any,
        perturbation: Any,
        *,
        chain_block_size: int | None = None,
        structure_kind: Literal["diagonal", "circulant", "toeplitz", "kronecker"]
        | None = None,
        structure: KroneckerStructure | None = None,
        low_rank_factors: LowRankFactors | None = None,
        exact_power_traces: Sequence[float] | None = None,
        frozen_probes: FrozenProbes | None = None,
        trace_order: int | None = None,
        certified_rho: float | None = None,
    ):
        lam, perturb = _matrix_pair(lambda_matrix, perturbation)
        lam = _read_only_array(lam)
        perturb = _read_only_array(perturb)
        if not _is_positive_definite(lam):
            raise ValueError("Lambda must be symmetric positive definite")
        if frozen_probes is not None and type(frozen_probes) is not FrozenProbes:
            raise TypeError(
                "frozen_probes must be an exact FrozenProbes bytes-backed instance"
            )
        object.__setattr__(self, "lambda_matrix", lam)
        object.__setattr__(self, "perturbation", perturb)
        object.__setattr__(self, "chain_block_size", chain_block_size)
        object.__setattr__(self, "structure_kind", structure_kind)
        object.__setattr__(self, "structure", structure)
        object.__setattr__(self, "low_rank_factors", low_rank_factors)
        object.__setattr__(
            self,
            "exact_power_traces",
            None
            if exact_power_traces is None
            else tuple(float(x) for x in exact_power_traces),
        )
        object.__setattr__(self, "frozen_probes", frozen_probes)
        object.__setattr__(self, "trace_order", trace_order)
        object.__setattr__(self, "certified_rho", certified_rho)


@dataclasses.dataclass(frozen=True)
class PremiseVerdict:
    """One ladder row's evidence-bearing premise decision."""

    level: int
    method: str
    satisfied: bool
    reason: str
    details: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class LadderResult:
    """The first accepted method and all higher-priority rejections."""

    level: int
    method: str
    value: float
    rejected: tuple[PremiseVerdict, ...]


def _read_only_array(value: Any, *, ndim: int | None = None) -> np.ndarray:
    array = np.array(value, copy=True)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(
            f"expected a {ndim}-dimensional array, got shape {array.shape}"
        )
    if array.ndim not in (1, 2):
        raise ValueError(
            f"matrix inputs must be one- or two-dimensional, got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(float)
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix inputs must be finite")
    array.setflags(write=False)
    return array


def _matrix_pair(
    lambda_matrix: Any, perturbation: Any
) -> tuple[np.ndarray, np.ndarray]:
    lam = np.asarray(lambda_matrix)
    perturb = np.asarray(perturbation)
    if lam.shape != perturb.shape or lam.ndim not in (1, 2):
        raise ValueError(
            "Lambda and perturbation must have the same diagonal-vector or square-matrix shape"
        )
    if lam.ndim == 2 and (lam.shape[0] != lam.shape[1]):
        raise ValueError("Lambda and perturbation matrices must be square")
    return lam, perturb


def _dense(value: np.ndarray) -> np.ndarray:
    return np.diag(value) if value.ndim == 1 else value


def _n(value: np.ndarray) -> int:
    return int(value.shape[0])


def _algebraic_rank_bound(
    value: np.ndarray,
    factors: LowRankFactors | None = None,
    lambda_matrix: np.ndarray | None = None,
) -> int:
    """A safe finite-polynomial termination degree, never a numerical rank.

    A supplied factorisation is checked in the preconditioned geometry.  Only
    an exact array match is algebraic-rank evidence.  A floating residual
    cannot be admitted by a generic tolerance: after multiplication by
    ``Lambda^-1`` it can become order one, and near an eigenvalue of ``-1`` an
    arbitrarily small omitted direction can change the log determinant by
    order one.  Callers needing an approximate factorisation must use an
    approximate rung with an explicit propagated error budget.
    """
    if factors is not None:
        if (
            factors.left.shape[0] != value.shape[0]
            or factors.right.shape[0] != value.shape[0]
        ):
            raise ValueError("low-rank factor row counts must equal perturbation size")
        if value.ndim != 2:
            raise ValueError("dense low-rank factors require a dense perturbation")
        reconstructed = factors.left @ factors.right.T
        if not np.array_equal(reconstructed, value):
            if lambda_matrix is None:
                raise ValueError(
                    "Lambda is required to check low-rank reconstruction in "
                    "the preconditioned geometry"
                )
            inverse = np.linalg.inv(_dense(lambda_matrix))
            residual_x = (value - reconstructed) @ inverse
            amplified = float(np.max(np.abs(residual_x), initial=0.0))
            raise ValueError(
                "low-rank factors do not exactly reconstruct the perturbation; "
                "a nonzero Lambda^-1-amplified residual is not algebraic-rank "
                f"evidence (amplified residual={amplified:.8g})"
            )
        if reconstructed.shape != value.shape:
            raise ValueError(
                "low-rank factors do not reconstruct the perturbation shape"
            )
        return factors.rank_bound
    if value.ndim == 1:
        return int(np.count_nonzero(value != 0.0))
    return int(value.shape[0])


def _is_positive_definite(value: np.ndarray) -> bool:
    if value.ndim == 1:
        return bool(np.all(value > 0.0))
    if value.shape[0] != value.shape[1] or not np.array_equal(value, value.T):
        return False
    return bool(np.all(np.linalg.eigvalsh(value) > 0.0))


def _condition_certificate(value: np.ndarray) -> tuple[float, float, bool]:
    """Return condition, dtype ceiling, and strict numerical-resolution verdict."""
    if value.ndim == 1:
        magnitudes = np.abs(value)
        smallest = float(np.min(magnitudes, initial=math.inf))
        condition = (
            math.inf
            if smallest == 0.0
            else float(np.max(magnitudes, initial=0.0)) / smallest
        )
    else:
        condition = float(np.linalg.cond(value))
    dtype = value.dtype if np.issubdtype(value.dtype, np.inexact) else np.dtype(float)
    ceiling = condition_ceiling(dtype)
    return condition, ceiling, bool(np.isfinite(condition) and condition < ceiling)


def _require_resolved_dense_condition(value: np.ndarray, method: str) -> None:
    """Refuse dense exact arithmetic beyond the repository precision policy."""
    if value.ndim == 1:
        return
    condition, ceiling, resolved = _condition_certificate(value)
    if not resolved:
        raise ValueError(
            f"{method} condition {condition:.8g} is not below the strict "
            f"dtype ceiling {ceiling:.8g}"
        )


def _x_matrix(lambda_matrix: np.ndarray, perturbation: np.ndarray) -> np.ndarray:
    if lambda_matrix.ndim == 1:
        return perturbation / lambda_matrix
    return np.linalg.solve(lambda_matrix.T, perturbation.T).T


def lambda_logdet(lambda_matrix: Any) -> float:
    """Exact logdet of the explicit preconditioner ``Lambda``."""
    lam = _read_only_array(lambda_matrix)
    if not _is_positive_definite(lam):
        raise ValueError("Lambda must be symmetric positive definite")
    if lam.ndim == 1:
        return float(np.sum(np.log(lam)))
    factor = np.linalg.cholesky(lam)
    return float(2.0 * np.sum(np.log(np.diag(factor))))


def _newton_stability(
    lambda_matrix: Any, perturbation: Any, termination: int
) -> tuple[bool, float]:
    """Conservative rho gate shared by both finite e-polynomial entries.

    No termination degree is an escape from the measured ``rho(X) <= 1``
    requirement. This gate makes no determinant computation; the payload uses
    stable factored evaluation rather than a cancelling Newton recurrence.
    """
    rho = spectral_radius(lambda_matrix, perturbation)
    stable = rho <= 1.0
    return stable, rho


def _newton_logdet(
    lambda_matrix: Any,
    perturbation: Any,
    *,
    termination: int,
    factors: LowRankFactors | None,
) -> float:
    """Evaluate the finite e-polynomial through a stable factored form.

    Direct Newton identities are algebraically exact but numerically unsafe:
    mixed-sign spectra cancel and positive high-degree spectra overflow even
    when the log determinant is finite. Compact diagonal products sum
    ``log(Lambda + P)`` entrywise; a symmetric low-rank factor uses its
    determinant-lemma factor;
    all remaining dense inputs use Cholesky. These are representations of the
    same finite polynomial, selected only by verified sparsity evidence.
    """
    lam, perturb = _matrix_pair(lambda_matrix, perturbation)
    if not _is_positive_definite(lam):
        raise ValueError("Lambda must be symmetric positive definite")
    if not _is_positive_definite(lam + perturb):
        raise ValueError("Sigma must be symmetric positive definite")
    stable, rho = _newton_stability(lam, perturb, termination)
    if not stable:
        raise ValueError(
            "finite e-polynomial stability cannot certify an expansive spectrum at "
            f"degree {termination}: measured rho={rho:.8g}. Fall through to "
            "a stable exact rung."
        )
    x = _x_matrix(lam, perturb)
    if x.ndim == 1:
        return math.fsum(
            math.log(float(value)) for value in lam + perturb
        )
    sigma = lam + perturb
    _require_resolved_dense_condition(sigma, "finite e-polynomial")
    if factors is not None and factors.left is factors.right:
        solved = np.linalg.solve(lam, factors.left)
        reduced = np.eye(termination, dtype=lam.dtype) + factors.left.T @ solved
        factor = np.linalg.cholesky(reduced)
        return lambda_logdet(lam) + float(
            2.0 * np.sum(np.log(np.diag(factor)))
        )
    return dense_cholesky_logdet(sigma)


def low_rank_logdet(
    lambda_matrix: Any,
    perturbation: Any,
    *,
    factors: LowRankFactors | None = None,
) -> float:
    """Rung 1: stable rank-``k`` factorization of the finite e-polynomial."""
    lam, perturb = _matrix_pair(lambda_matrix, perturbation)
    termination = _algebraic_rank_bound(perturb, factors, lam)
    return _newton_logdet(
        lambda_matrix,
        perturb,
        termination=termination,
        factors=factors,
    )


def finite_perturbation_logdet(
    lambda_matrix: Any,
    perturbation: Any,
    *,
    factors: LowRankFactors | None = None,
) -> float:
    """Rung 5: stable factored evaluation of the finite e-polynomial."""
    lam, perturb = _matrix_pair(lambda_matrix, perturbation)
    termination = _algebraic_rank_bound(perturb, factors, lam)
    return _newton_logdet(
        lambda_matrix,
        perturb,
        termination=termination,
        factors=factors,
    )


def _is_block_chain(
    matrix: np.ndarray, block_size: int, *, rtol: float, atol: float
) -> bool:
    del rtol, atol
    n = matrix.shape[0]
    if block_size < 1 or n % block_size:
        return False
    blocks = n // block_size
    for row in range(blocks):
        for column in range(blocks):
            if abs(row - column) <= 1:
                continue
            piece = matrix[
                row * block_size : (row + 1) * block_size,
                column * block_size : (column + 1) * block_size,
            ]
            if np.any(piece != 0.0):
                return False
    return True


def state_space_logdet(
    matrix: Any, *, block_size: int, rtol: float = 1e-11, atol: float = 1e-13
) -> float:
    """Rung 2: block-LDL recursion after numerically verifying a chain."""
    dense = _read_only_array(matrix, ndim=2)
    if not _is_block_chain(dense, block_size, rtol=rtol, atol=atol):
        raise ValueError(f"matrix is not a block chain with block_size={block_size}")
    if not np.array_equal(dense, dense.T):
        raise ValueError("a block chain logdet requires a symmetric matrix")
    if not _is_positive_definite(dense):
        raise ValueError("a block chain logdet requires a positive definite matrix")
    _require_resolved_dense_condition(dense, "block-LDL")
    blocks = dense.shape[0] // block_size
    schur = np.array(dense[:block_size, :block_size], copy=True)
    total = lambda_logdet(schur)
    for index in range(1, blocks):
        start = index * block_size
        previous = start - block_size
        link = dense[start : start + block_size, previous:start]
        diagonal = dense[start : start + block_size, start : start + block_size]
        schur = diagonal - link @ np.linalg.solve(schur, link.T)
        total += lambda_logdet(schur)
    return float(total)


def _is_diagonal(matrix: np.ndarray, *, rtol: float, atol: float) -> bool:
    del rtol, atol
    return bool(np.array_equal(matrix, np.diag(np.diag(matrix))))


def _is_circulant(matrix: np.ndarray, *, rtol: float, atol: float) -> bool:
    del rtol, atol
    first = matrix[0]
    expected = np.vstack([np.roll(first, index) for index in range(matrix.shape[0])])
    return bool(np.array_equal(matrix, expected))


def _circulant_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    """Return the real positive FFT spectrum required by the payload."""
    eigenvalues = np.real_if_close(np.fft.fft(matrix[0]))
    if np.iscomplexobj(eigenvalues) or np.any(eigenvalues <= 0.0):
        raise ValueError("circulant matrix must have a real positive spectrum")
    return eigenvalues


def _is_toeplitz(matrix: np.ndarray, *, rtol: float, atol: float) -> bool:
    del rtol, atol
    for offset in range(-matrix.shape[0] + 1, matrix.shape[0]):
        diagonal = np.diag(matrix, k=offset)
        if not np.all(diagonal == diagonal[0]):
            return False
    return True


def structured_logdet(
    matrix: Any,
    *,
    kind: Literal["diagonal", "circulant", "toeplitz", "kronecker"],
    structure: KroneckerStructure | None = None,
    rtol: float = 1e-11,
    atol: float = 1e-13,
) -> float:
    """Rung 3: an exact evaluator after checking the requested structure."""
    dense = _read_only_array(matrix)
    if dense.ndim == 1:
        if kind != "diagonal":
            raise ValueError(f"a diagonal-vector input is not {kind}")
        return lambda_logdet(dense)
    if kind == "diagonal":
        if not _is_diagonal(dense, rtol=rtol, atol=atol):
            raise ValueError("matrix is not diagonal")
        return lambda_logdet(np.diag(dense))
    if kind == "circulant":
        if not _is_circulant(dense, rtol=rtol, atol=atol):
            raise ValueError("matrix is not circulant")
        eigenvalues = _circulant_eigenvalues(dense)
        _require_resolved_dense_condition(dense, "circulant")
        return float(np.sum(np.log(eigenvalues)))
    if kind == "toeplitz":
        if not _is_toeplitz(dense, rtol=rtol, atol=atol):
            raise ValueError("matrix is not Toeplitz")
        _require_resolved_dense_condition(dense, "Toeplitz")
        return dense_cholesky_logdet(dense)
    if kind == "kronecker":
        if structure is None:
            raise ValueError("kronecker evaluation needs factors to verify")
        reconstructed = structure.factors[0]
        for factor in structure.factors[1:]:
            reconstructed = np.kron(reconstructed, factor)
        if reconstructed.shape != dense.shape or not np.array_equal(
            reconstructed, dense
        ):
            raise ValueError("Kronecker factors do not reconstruct the supplied matrix")
        _require_resolved_dense_condition(dense, "Kronecker")
        total_size = dense.shape[0]
        total = 0.0
        for factor in structure.factors:
            total += (total_size // factor.shape[0]) * dense_cholesky_logdet(factor)
        return float(total)
    raise ValueError(f"unsupported structure kind {kind!r}")


def dense_cholesky_logdet(matrix: Any) -> float:
    """Rung 4: exact dense Cholesky arithmetic (premises checked separately)."""
    return lambda_logdet(matrix)


def spectral_radius(lambda_matrix: Any, perturbation: Any) -> float:
    """Eager measured ``rho(P Lambda^-1)`` for dense or diagonal inputs."""
    lam, perturb = _matrix_pair(lambda_matrix, perturbation)
    x = _x_matrix(lam, perturb)
    if x.ndim == 1:
        return float(np.max(np.abs(x), initial=0.0))
    return float(np.max(np.abs(np.linalg.eigvals(x)), initial=0.0))


def _validate_strict_rho(
    lambda_matrix: np.ndarray,
    perturbation: np.ndarray,
    certified_rho: float | None,
) -> tuple[float, float]:
    """Eagerly require both measured and certified rho to be strictly below one."""
    actual_rho = spectral_radius(lambda_matrix, perturbation)
    certificate = actual_rho if certified_rho is None else float(certified_rho)
    if not actual_rho < 1.0:
        raise ValueError(
            f"measured rho={actual_rho} does not satisfy strict rho < 1"
        )
    if not 0.0 <= certificate < 1.0:
        raise ValueError(
            f"trace-log convergence requires certified rho < 1; got {certificate}"
        )
    if actual_rho > certificate and not np.isclose(
        actual_rho, certificate, rtol=1e-12, atol=1e-14
    ):
        raise ValueError(
            f"rho certificate {certificate} understates measured rho {actual_rho}"
        )
    return actual_rho, certificate


def _computed_power_traces(
    lambda_matrix: np.ndarray, perturbation: np.ndarray, order: int
) -> tuple[float, ...]:
    """Power traces derived from the matrices, for eager premise verification."""
    x = _x_matrix(lambda_matrix, perturbation)
    power = np.ones_like(x) if x.ndim == 1 else np.eye(x.shape[0], dtype=x.dtype)
    traces: list[float] = []
    for _ in range(order):
        power = power * x if x.ndim == 1 else power @ x
        traces.append(float(np.sum(power) if x.ndim == 1 else np.trace(power)))
    return tuple(traces)


def _power_traces_match(
    lambda_matrix: np.ndarray,
    perturbation: np.ndarray,
    traces: Sequence[float],
    order: int,
) -> bool:
    if order < 0 or len(traces) < order:
        return False
    supplied = np.asarray(tuple(traces)[:order], dtype=float)
    derived = np.asarray(
        _computed_power_traces(lambda_matrix, perturbation, order), dtype=float
    )
    return bool(np.all(np.isfinite(supplied)) and np.array_equal(supplied, derived))


def trace_log_tail_bound(rho: float, order: int) -> float:
    """Scalar tail bound ``rho**(m+1) / ((m+1)*(1-rho))``."""
    rho = float(rho)
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"trace-log convergence requires rho < 1; got {rho}")
    if order < 0:
        raise ValueError("trace-log order must be non-negative")
    return rho ** (order + 1) / ((order + 1) * (1.0 - rho))


def whole_trace_log_tail_bound(rho: float, order: int, multiplicity: int) -> float:
    """Whole-trace bound: the scalar bound times explicit eigenvalue count."""
    if multiplicity < 1:
        raise ValueError("multiplicity must be positive")
    return multiplicity * trace_log_tail_bound(rho, order)


def choose_trace_order(rho: float, tolerance: float, *, multiplicity: int) -> int:
    """Smallest fixed order whose whole-trace bound meets ``tolerance``."""
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    order = 0
    while whole_trace_log_tail_bound(rho, order, multiplicity) > tolerance:
        order += 1
    return order


def truncated_trace_logdet(
    lambda_matrix: Any,
    perturbation: Any,
    *,
    exact_power_traces: Sequence[float] | None = None,
    order: int,
    rho: float | None = None,
) -> float:
    """Rung 6: deterministic finite trace-log from exact power traces.

    The convergence decision is eager: warmup measures a conservative maximum
    over probe points plus margin and chooses fixed ``order``. Runtime uses
    that fixed order and performs no guard or traced Python branch; retained
    samples are rechecked by :func:`audit_retained_rho`. The warmup rho
    certificate is, like solver ``tol``, the only number between the user and
    silent error.

    ``Lambda`` is the caller's preconditioner, not necessarily the noise
    covariance. Foreground-dominated ``P`` may make direct expansion diverge;
    a diagonal, block, or circulant ``Lambda`` may reduce rho.
    """
    lam, perturb = _matrix_pair(lambda_matrix, perturbation)
    _validate_strict_rho(lam, perturb, rho)
    if exact_power_traces is None:
        raise ValueError(
            "rung 6 requires deterministic exact power traces; one generic "
            "matvec cannot provide Tr(X**r)"
        )
    traces = tuple(float(value) for value in exact_power_traces)
    if order < 0 or len(traces) < order:
        raise ValueError("exact power traces must contain every requested order")
    if not _power_traces_match(lam, perturb, traces, order):
        raise ValueError(
            "supplied exact power traces do not match traces derived from X"
        )
    correction = math.fsum(
        ((-1.0) ** (power + 1)) * traces[power - 1] / power
        for power in range(1, order + 1)
    )
    return lambda_logdet(lam) + correction


def frozen_hutchinson_trace_logdet(
    lambda_matrix: Any,
    perturbation: Any,
    probes: FrozenProbes,
    *,
    order: int,
    rho: float | None = None,
) -> float:
    """Rung 7: frozen-probe Taylor trace-log after an eager strict-rho check."""
    lam, perturb = _matrix_pair(lambda_matrix, perturbation)
    _validate_strict_rho(lam, perturb, rho)
    x = _x_matrix(lam, perturb)
    if probes.values.shape[1] != _n(lam):
        raise ValueError("frozen probe width must equal the matrix dimension")
    if order < 0:
        raise ValueError("order must be non-negative")
    vectors = probes.values.T
    images = np.array(vectors, copy=True)
    estimates: list[float] = []
    for _ in range(order):
        images = x[:, None] * images if x.ndim == 1 else x @ images
        estimates.append(float(np.mean(np.sum(vectors * images, axis=0))))
    correction = math.fsum(
        ((-1.0) ** (power + 1)) * estimates[power - 1] / power
        for power in range(1, order + 1)
    )
    return lambda_logdet(lam) + correction


def resampled_trace_logdet(
    lambda_matrix: Any,
    perturbation: Any,
    *,
    order: int,
    probes: int,
) -> float:
    """Rung 8: always refuse per-call resampling because it is HMC-unsafe."""
    del lambda_matrix, perturbation, order, probes
    raise ResamplingRefused(
        "Per-call probe resampling makes the log determinant noisy, breaks HMC "
        "reversibility, and is always refused. Supply FrozenProbes instead."
    )
