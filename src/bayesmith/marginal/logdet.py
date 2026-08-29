"""A premise-checked log-determinant ladder, from special to general.

The dispatcher walks an ordered base case plus eight rungs and selects the
first whose premise is verified from numerical inputs.  Labels such as
``"circulant"`` and a chain block size are requests to *check* structure, not
claims the dispatcher trusts.  :func:`check_logdet_premises` performs those
checks without computing a log determinant; direct methods stay public so a
boundary test can compare both sides of every threshold.

Write ``Sigma = Lambda + P`` and ``X = P Lambda^-1``.  ``Lambda`` is an
explicit preconditioner design choice, not necessarily the noise covariance.
Foreground-dominated ``P`` can make the direct trace-log expansion diverge;
choosing a diagonal, block-diagonal, or circulant ``Lambda`` closer to
``Sigma`` can reduce ``rho(X)``.  This module never hides a preconditioner
selector.

The finite exact route obtains elementary symmetric polynomials from power
traces using Newton identities.  When ``rank(P) = k``, all ``e_j(X)`` above
``k`` vanish, so the low-rank rung is precisely the sparse termination of the
same routine rather than a second determinant-lemma implementation.

The file has two deliberately different numerical layers. Eager NumPy
functions verify structure, rank evidence, rho certificates, and trace
providers before sampling. Functions ending in ``_runtime`` are pure JAX
kernels: their order is static, their inputs come from those verified
providers, and they contain no Python convergence guard. This split is what
keeps the deterministic approximation differentiable inside HMC without
pretending a theta-dependent premise can be checked there.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Sequence
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from bayesmith.exact.fisher import condition_ceiling

__all__ = [
    "AuditReport",
    "FrozenProbes",
    "KroneckerStructure",
    "LadderConfig",
    "LadderResult",
    "LowRankFactors",
    "LogDetProblem",
    "PremiseVerdict",
    "ResamplingRefused",
    "RhoCertificate",
    "audit_retained_rho",
    "certify_warmup_rho",
    "check_logdet_premises",
    "choose_trace_order",
    "dense_cholesky_logdet",
    "dispatch_logdet",
    "finite_perturbation_logdet",
    "frozen_hutchinson_trace_logdet",
    "frozen_hutchinson_trace_logdet_runtime",
    "lambda_logdet",
    "low_rank_logdet",
    "resampled_trace_logdet",
    "spectral_radius",
    "state_space_logdet",
    "structured_logdet",
    "trace_log_tail_bound",
    "truncated_trace_logdet",
    "truncated_trace_logdet_runtime",
    "whole_trace_log_tail_bound",
]


_METHODS = (
    "Lambda itself",
    "low-rank Newton termination",
    "state-space recursion",
    "structured exact",
    "dense Cholesky",
    "finite Newton perturbation",
    "truncated trace-log",
    "frozen Hutchinson trace-log",
    "per-call resampling",
)


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
    tolerance. Dependent columns only make the bound conservative: Newton
    identities still terminate by degree ``k``.
    """

    left: np.ndarray
    right: np.ndarray

    def __init__(self, left: Any, right: Any | None = None):
        left_array = _read_only_array(left, ndim=2)
        right_array = (
            _read_only_array(left, ndim=2)
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
        object.__setattr__(self, "lambda_matrix", _read_only_array(lam))
        object.__setattr__(self, "perturbation", _read_only_array(perturb))
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


@dataclasses.dataclass(frozen=True)
class RhoCertificate:
    """An eager warmup measurement that fixes the runtime expansion order.

    ``rho`` depends on theta, so warmup must measure a conservative maximum
    over probe points and add a margin. Runtime executes the fixed order with
    no traced Python guard, and retained samples are audited afterward. The
    warmup rho certificate is, like solver ``tol``, the only number between
    the user and silent error.
    """

    measured_max: float
    margin: float
    certified_rho: float
    order: int
    tolerance: float
    multiplicity: int


@dataclasses.dataclass(frozen=True)
class AuditReport:
    """Post-run comparison of retained rho measurements with a certificate."""

    passed: bool
    measured_max: float
    violations: tuple[int, ...]


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
    value: np.ndarray, factors: LowRankFactors | None = None
) -> int:
    """A safe Newton termination degree, never a numerical-rank estimate."""
    if factors is not None:
        if (
            factors.left.shape[0] != value.shape[0]
            or factors.right.shape[0] != value.shape[0]
        ):
            raise ValueError("low-rank factor row counts must equal perturbation size")
        reconstructed = factors.left @ factors.right.T
        if value.ndim != 2 or not np.allclose(
            reconstructed, value, rtol=2e-11, atol=2e-13
        ):
            raise ValueError("low-rank factors do not reconstruct the perturbation")
        return factors.rank_bound
    if value.ndim == 1:
        return int(np.count_nonzero(value != 0.0))
    return int(value.shape[0])


def _is_positive_definite(value: np.ndarray) -> bool:
    if value.ndim == 1:
        return bool(np.all(value > 0.0))
    if not np.allclose(value, value.T, rtol=1e-12, atol=1e-14):
        return False
    return bool(np.all(np.linalg.eigvalsh(value) > 0.0))


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


def _newton_logdet(lambda_matrix: Any, perturbation: Any, *, termination: int) -> float:
    lam, perturb = _matrix_pair(lambda_matrix, perturbation)
    if not _is_positive_definite(lam):
        raise ValueError("Lambda must be symmetric positive definite")
    x = _x_matrix(lam, perturb)
    elementary = [1.0]
    power = np.ones_like(x) if x.ndim == 1 else np.eye(x.shape[0], dtype=x.dtype)
    traces: list[float] = []
    for order in range(1, termination + 1):
        power = power * x if x.ndim == 1 else power @ x
        traces.append(float(np.sum(power) if x.ndim == 1 else np.trace(power)))
        numerator = math.fsum(
            ((-1.0) ** (j - 1)) * elementary[order - j] * traces[j - 1]
            for j in range(1, order + 1)
        )
        elementary.append(numerator / order)
    determinant_ratio = math.fsum(elementary)
    if not np.isfinite(determinant_ratio) or determinant_ratio <= 0.0:
        raise ValueError("Newton identities did not produce a positive determinant")
    return lambda_logdet(lam) + math.log(determinant_ratio)


def low_rank_logdet(
    lambda_matrix: Any,
    perturbation: Any,
    *,
    factors: LowRankFactors | None = None,
) -> float:
    """Rung 1: the rank-``k`` sparse termination of Newton identities."""
    _, perturb = _matrix_pair(lambda_matrix, perturbation)
    return _newton_logdet(
        lambda_matrix,
        perturb,
        termination=_algebraic_rank_bound(perturb, factors),
    )


def finite_perturbation_logdet(
    lambda_matrix: Any,
    perturbation: Any,
    *,
    factors: LowRankFactors | None = None,
) -> float:
    """Rung 5: finite exact Newton expansion, terminating at verified rank."""
    _, perturb = _matrix_pair(lambda_matrix, perturbation)
    return _newton_logdet(
        lambda_matrix,
        perturb,
        termination=_algebraic_rank_bound(perturb, factors),
    )


def _is_block_chain(
    matrix: np.ndarray, block_size: int, *, rtol: float, atol: float
) -> bool:
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
            if not np.allclose(piece, 0.0, rtol=rtol, atol=atol):
                return False
    return True


def state_space_logdet(
    matrix: Any, *, block_size: int, rtol: float = 1e-11, atol: float = 1e-13
) -> float:
    """Rung 2: block-LDL recursion after numerically verifying a chain."""
    dense = _read_only_array(matrix, ndim=2)
    if not _is_block_chain(dense, block_size, rtol=rtol, atol=atol):
        raise ValueError(f"matrix is not a block chain with block_size={block_size}")
    if not np.allclose(dense, dense.T, rtol=rtol, atol=atol):
        raise ValueError("a block chain logdet requires a symmetric matrix")
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
    return bool(np.allclose(matrix, np.diag(np.diag(matrix)), rtol=rtol, atol=atol))


def _is_circulant(matrix: np.ndarray, *, rtol: float, atol: float) -> bool:
    first = matrix[0]
    expected = np.vstack([np.roll(first, index) for index in range(matrix.shape[0])])
    return bool(np.allclose(matrix, expected, rtol=rtol, atol=atol))


def _is_toeplitz(matrix: np.ndarray, *, rtol: float, atol: float) -> bool:
    for offset in range(-matrix.shape[0] + 1, matrix.shape[0]):
        diagonal = np.diag(matrix, k=offset)
        if not np.allclose(diagonal, diagonal[0], rtol=rtol, atol=atol):
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
        eigenvalues = np.real_if_close(np.fft.fft(dense[0]))
        if np.iscomplexobj(eigenvalues) or np.any(eigenvalues <= 0.0):
            raise ValueError("circulant matrix must be symmetric positive definite")
        return float(np.sum(np.log(eigenvalues)))
    if kind == "toeplitz":
        if not _is_toeplitz(dense, rtol=rtol, atol=atol):
            raise ValueError("matrix is not Toeplitz")
        return dense_cholesky_logdet(dense)
    if kind == "kronecker":
        if structure is None:
            raise ValueError("kronecker evaluation needs factors to verify")
        reconstructed = structure.factors[0]
        for factor in structure.factors[1:]:
            reconstructed = np.kron(reconstructed, factor)
        if reconstructed.shape != dense.shape or not np.allclose(
            reconstructed, dense, rtol=rtol, atol=atol
        ):
            raise ValueError("Kronecker factors do not reconstruct the supplied matrix")
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
    return bool(
        np.all(np.isfinite(supplied))
        and np.allclose(supplied, derived, rtol=2e-11, atol=2e-13)
    )


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
    actual_rho = spectral_radius(lam, perturb)
    certified_rho = actual_rho if rho is None else float(rho)
    if not 0.0 <= certified_rho < 1.0:
        raise ValueError(f"trace-log convergence requires rho < 1; got {certified_rho}")
    if actual_rho > certified_rho * (1.0 + 2e-12) + 2e-14:
        raise ValueError(
            f"rho certificate {certified_rho} understates measured rho {actual_rho}"
        )
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


def truncated_trace_logdet_runtime(
    lambda_logdet_value: Any,
    exact_power_traces: Any,
    *,
    order: int,
) -> jnp.ndarray:
    """Pure JAX runtime kernel for a warmup-certified fixed trace order.

    Warmup must verify the exact trace provider, measure a conservative rho
    maximum plus margin, and choose the static ``order`` before tracing. This
    kernel deliberately performs no rho check: runtime checking would put a
    Python branch inside HMC. Retained samples are rechecked afterward. The
    warmup rho certificate is, like solver ``tol``, the only number between
    the user and silent error.
    """
    traces = jnp.asarray(exact_power_traces)[:order]
    powers = jnp.arange(1, order + 1, dtype=traces.dtype)
    signs = jnp.where(powers % 2 == 1, 1.0, -1.0)
    return jnp.asarray(lambda_logdet_value) + jnp.sum(signs * traces / powers)


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
    actual_rho = spectral_radius(lam, perturb)
    certified_rho = actual_rho if rho is None else float(rho)
    if not 0.0 <= certified_rho < 1.0:
        raise ValueError(
            f"frozen Taylor trace-log convergence requires rho < 1; got {certified_rho}"
        )
    if actual_rho > certified_rho * (1.0 + 2e-12) + 2e-14:
        raise ValueError(
            f"rho certificate {certified_rho} understates measured rho {actual_rho}"
        )
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


def frozen_hutchinson_trace_logdet_runtime(
    lambda_logdet_value: Any,
    x_matrix: Any,
    frozen_probe_values: Any,
    *,
    order: int,
) -> jnp.ndarray:
    """Pure JAX frozen-probe Taylor kernel at a warmup-certified fixed order.

    ``frozen_probe_values`` must come from immutable :class:`FrozenProbes`
    data and the caller must have certified strict ``rho < 1`` eagerly. No
    convergence check is repeated inside the trace.
    """
    x = jnp.asarray(x_matrix)
    vectors = jnp.asarray(frozen_probe_values).T
    images = vectors
    correction = jnp.asarray(0.0, dtype=jnp.result_type(x, vectors))
    for power in range(1, order + 1):
        images = x @ images
        trace_estimate = jnp.mean(jnp.sum(vectors * images, axis=0))
        correction = correction + ((-1.0) ** (power + 1) / power) * trace_estimate
    return jnp.asarray(lambda_logdet_value) + correction


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


def certify_warmup_rho(
    measured_rhos: Iterable[float],
    *,
    margin: float,
    tolerance: float,
    multiplicity: int,
) -> RhoCertificate:
    """Eagerly certify warmup's maximum rho plus margin and choose fixed order.

    The warmup rho certificate is, like solver ``tol``, the only number
    between the user and silent error. Runtime must not replace this eager
    decision with a traced branch.
    """
    values = tuple(float(value) for value in measured_rhos)
    if not values or any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(
            "warmup rho measurements must be non-empty, finite, and non-negative"
        )
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("rho safety margin must be finite and non-negative")
    measured_max = max(values)
    certified = measured_max + margin
    if certified >= 1.0:
        raise ValueError(
            f"warmup maximum {measured_max} plus margin {margin} does not certify rho < 1"
        )
    return RhoCertificate(
        measured_max=measured_max,
        margin=float(margin),
        certified_rho=certified,
        order=choose_trace_order(certified, tolerance, multiplicity=multiplicity),
        tolerance=float(tolerance),
        multiplicity=int(multiplicity),
    )


def audit_retained_rho(
    measured_rhos: Iterable[float], certificate: RhoCertificate
) -> AuditReport:
    """Eagerly recheck retained samples against the warmup rho certificate."""
    values = tuple(float(value) for value in measured_rhos)
    if not values or any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(
            "retained rho measurements must be non-empty, finite, and non-negative"
        )
    violations = tuple(
        index for index, value in enumerate(values) if value > certificate.certified_rho
    )
    return AuditReport(
        passed=not violations,
        measured_max=max(values),
        violations=violations,
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
        reconstructed = problem.structure.factors[0]
        for factor in problem.structure.factors[1:]:
            reconstructed = np.kron(reconstructed, factor)
        valid = reconstructed.shape == sigma.shape and np.allclose(
            reconstructed, sigma, rtol=config.structure_rtol, atol=config.structure_atol
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
    sigma_symmetric = sigma.ndim == 1 or bool(
        np.allclose(
            sigma,
            sigma.T,
            rtol=config.structure_rtol,
            atol=config.structure_atol,
        )
    )
    sigma_spd = sigma_symmetric and _is_positive_definite(sigma)
    try:
        rank = _algebraic_rank_bound(perturb, problem.low_rank_factors)
        rank_evidence_valid = True
    except ValueError:
        rank = n
        rank_evidence_valid = False
    base = bool(np.array_equal(sigma, lam))
    has_algebraic_evidence = perturb.ndim == 1 or problem.low_rank_factors is not None
    low_rank = (
        rank_evidence_valid
        and has_algebraic_evidence
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
        chain = chain_structure and sigma_spd
        if not chain_structure:
            chain_reason = (
                "the supplied matrix is not block tridiagonal at that block size"
            )
        elif not sigma_symmetric:
            chain_reason = "the block-tridiagonal matrix is not symmetric"
        elif not sigma_spd:
            chain_reason = "the symmetric block chain is not positive definite"
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

    condition = (
        float(np.max(sigma) / np.min(sigma))
        if sigma.ndim == 1
        else float(np.linalg.cond(sigma))
    )
    dtype = sigma.dtype if np.issubdtype(sigma.dtype, np.inexact) else np.dtype(float)
    ceiling = condition_ceiling(dtype)
    dense = n <= config.dense_max_n and condition < ceiling and sigma_spd
    finite = n <= config.finite_max_n or rank <= config.finite_max_rank
    actual_rho = spectral_radius(lam, perturb)
    rho = actual_rho if problem.certified_rho is None else problem.certified_rho
    rho_covers_input = actual_rho <= rho * (1.0 + 2e-12) + 2e-14
    traces_verified = (
        problem.exact_power_traces is not None
        and problem.trace_order is not None
        and _power_traces_match(
            lam, perturb, problem.exact_power_traces, problem.trace_order
        )
    )
    trace = traces_verified and rho_covers_input and 0.0 <= rho < 1.0
    frozen = (
        problem.frozen_probes is not None
        and problem.trace_order is not None
        and rho_covers_input
        and 0.0 <= rho < 1.0
    )

    return (
        PremiseVerdict(
            0,
            _METHODS[0],
            base,
            "Sigma equals Lambda exactly" if base else "perturbation is nonzero",
            {"n": n},
        ),
        PremiseVerdict(
            1,
            _METHODS[1],
            low_rank,
            f"verified rank {rank}; limits are {config.low_rank_max} and {config.low_rank_fraction:g}*n",
            {"n": n, "rank": rank},
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
            f"n={n} (limit {config.finite_max_n}); rank={rank} (limit {config.finite_max_rank})",
            {"n": n, "rank": rank},
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
            else "immutable FrozenProbes, a fixed order, and a conservative rho<1 certificate are required",
            {"rho": rho, "measured_rho": actual_rho, "order": problem.trace_order},
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
            value = state_space_logdet(sigma, block_size=int(problem.chain_block_size))
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
                factors=problem.low_rank_factors,
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
