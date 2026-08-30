"""Warmup certificates, retained audits, and fixed-order JAX plans."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from bayesmith.marginal import _logdet_runtime as _runtime
from bayesmith.marginal._logdet_eager import (
    FrozenProbes,
    LogDetProblem,
    _algebraic_rank_bound,
    _n,
    _power_traces_match,
    _validate_strict_rho,
    _x_matrix,
    choose_trace_order,
    whole_trace_log_tail_bound,
)

__all__ = [
    "AuditReport",
    "FrozenTraceLogPlan",
    "RhoCertificate",
    "TraceAuditReport",
    "TraceLogPlan",
    "audit_retained_lambda_logdet",
    "audit_retained_operator_norm",
    "audit_retained_power_traces",
    "audit_retained_rho",
    "certify_warmup_rho",
    "make_frozen_trace_log_plan",
    "make_trace_log_plan",
]

_PLAN_TOKEN = object()

@dataclasses.dataclass(frozen=True)
class RhoCertificate:
    """Eager warmup bounds fixing runtime order and absolute arithmetic scale.

    ``rho`` depends on theta, so warmup must measure a conservative maximum
    over probe points and add a margin. Runtime executes the fixed order with
    no traced Python guard, and retained samples are audited afterward. The
    warmup rho certificate is, like solver ``tol``, the only number between
    the user and a silent convergence error.  A tolerance-bearing runtime plan
    additionally requires ``max_abs_lambda_logdet``. Frozen-probe plans also
    require ``max_x_operator_norm``, whose name means
    ``||abs(X)||_2`` (not ``||X||_2``): componentwise absolute value is what
    controls IEEE matrix-action roundoff. Retained samples must audit every
    bound used by their plan.
    """

    measured_max: float
    margin: float
    certified_rho: float
    order: int
    tolerance: float
    tail_tolerance: float
    multiplicity: int
    max_abs_lambda_logdet: float | None = None
    max_x_operator_norm: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.measured_max < 1.0:
            raise ValueError("a rho certificate needs measured_max < 1")
        if not 0.0 <= self.certified_rho < 1.0:
            raise ValueError("a rho certificate needs certified_rho < 1")
        if self.certified_rho < self.measured_max:
            raise ValueError("certified_rho must cover measured_max")
        if (
            self.margin < 0.0
            or self.tolerance <= 0.0
            or not 0.0 < self.tail_tolerance < self.tolerance
            or self.multiplicity < 1
        ):
            raise ValueError("rho certificate margin/tolerance/multiplicity are invalid")
        if self.max_abs_lambda_logdet is not None and (
            not np.isfinite(self.max_abs_lambda_logdet)
            or self.max_abs_lambda_logdet < 0.0
        ):
            raise ValueError("lambda-logdet scale bound must be finite and non-negative")
        if self.max_x_operator_norm is not None and (
            not np.isfinite(self.max_x_operator_norm)
            or self.max_x_operator_norm < 0.0
        ):
            raise ValueError("X operator-norm bound must be finite and non-negative")
        expected_order = choose_trace_order(
            self.certified_rho,
            self.tail_tolerance,
            multiplicity=self.multiplicity,
        )
        if self.order != expected_order:
            raise ValueError(
                f"rho certificate order must be the bound-selected {expected_order}, "
                f"got {self.order}"
            )


@dataclasses.dataclass(frozen=True)
class AuditReport:
    """Post-run comparison of retained rho measurements with a certificate."""

    passed: bool
    measured_max: float
    violations: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class TraceAuditReport:
    """Post-run verification of theta-dependent exact power-trace providers."""

    passed: bool
    violations: tuple[int, ...]


@dataclasses.dataclass(frozen=True, slots=True, init=False)
class TraceLogPlan:
    """Validated fixed-order JAX execution plan for exact power traces.

    Construct with :func:`make_trace_log_plan`; direct construction is
    refused. Runtime arguments are only the theta-dependent Lambda logdet and
    exact trace values. There is no per-call order argument to weaken.
    """

    _order: int
    _runtime_dtype: str

    def __init__(self, token: object, order: int, runtime_dtype: str):
        if token is not _PLAN_TOKEN:
            raise TypeError("TraceLogPlan must be created by make_trace_log_plan")
        object.__setattr__(self, "_order", int(order))
        object.__setattr__(self, "_runtime_dtype", runtime_dtype)

    @property
    def order(self) -> int:
        """The certificate-selected immutable truncation order."""
        return self._order

    def __call__(self, lambda_logdet_value: Any, exact_power_traces: Any) -> jnp.ndarray:
        _require_runtime_precision(
            self._runtime_dtype, lambda_logdet_value, exact_power_traces
        )
        return _runtime.truncated_trace_logdet(
            lambda_logdet_value, exact_power_traces, order=self._order
        )


@dataclasses.dataclass(frozen=True, slots=True, init=False)
class FrozenTraceLogPlan:
    """Validated fixed-order JAX plan capturing immutable frozen probes."""

    _order: int
    _probes: FrozenProbes
    _runtime_dtype: str

    def __init__(
        self, token: object, order: int, probes: FrozenProbes, runtime_dtype: str
    ):
        if token is not _PLAN_TOKEN:
            raise TypeError(
                "FrozenTraceLogPlan must be created by make_frozen_trace_log_plan"
            )
        object.__setattr__(self, "_order", int(order))
        object.__setattr__(self, "_probes", probes)
        object.__setattr__(self, "_runtime_dtype", runtime_dtype)

    @property
    def order(self) -> int:
        """The certificate-selected immutable truncation order."""
        return self._order

    def __call__(self, lambda_logdet_value: Any, x_matrix: Any) -> jnp.ndarray:
        _require_runtime_precision(self._runtime_dtype, lambda_logdet_value, x_matrix)
        return _runtime.frozen_hutchinson_trace_logdet(
            lambda_logdet_value,
            x_matrix,
            self._probes.values,
            order=self._order,
        )


def certify_warmup_rho(
    measured_rhos: Iterable[float],
    *,
    margin: float,
    tolerance: float,
    multiplicity: int,
    tail_fraction: float = 0.5,
    lambda_logdets: Iterable[float] | None = None,
    lambda_logdet_margin: float = 0.0,
    x_operator_norms: Iterable[float] | None = None,
    x_operator_norm_margin: float = 0.0,
) -> RhoCertificate:
    """Certify warmup rho/order and optionally the Lambda-logdet scale.

    The warmup rho certificate is, like solver ``tol``, the only number
    between the user and silent error. Runtime must not replace this eager
    decision with a traced branch. Runtime plan factories require
    ``lambda_logdets`` because an absolute tolerance also needs an arithmetic
    scale bound. ``x_operator_norms`` must contain conservative measurements
    of ``||abs(X)||_2``. Ordinary ``||X||_2`` is unsound here because sign
    cancellation can make it smaller by up to a dimension-dependent factor.
    """
    values = tuple(float(value) for value in measured_rhos)
    if not values or any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(
            "warmup rho measurements must be non-empty, finite, and non-negative"
        )
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("rho safety margin must be finite and non-negative")
    if not np.isfinite(tail_fraction) or not 0.0 < tail_fraction < 1.0:
        raise ValueError("tail_fraction must lie strictly between zero and one")
    bases = None if lambda_logdets is None else tuple(float(x) for x in lambda_logdets)
    if bases is not None and (
        not bases or any(not np.isfinite(value) for value in bases)
    ):
        raise ValueError("warmup lambda logdets must be non-empty and finite")
    if not np.isfinite(lambda_logdet_margin) or lambda_logdet_margin < 0.0:
        raise ValueError("lambda-logdet safety margin must be finite and non-negative")
    norms = (
        None
        if x_operator_norms is None
        else tuple(float(value) for value in x_operator_norms)
    )
    if norms is not None and (
        not norms or any(not np.isfinite(value) or value < 0.0 for value in norms)
    ):
        raise ValueError("warmup X operator norms must be non-empty and non-negative")
    if not np.isfinite(x_operator_norm_margin) or x_operator_norm_margin < 0.0:
        raise ValueError("X operator-norm safety margin must be non-negative")
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
        order=choose_trace_order(
            certified, tolerance * tail_fraction, multiplicity=multiplicity
        ),
        tolerance=float(tolerance),
        tail_tolerance=float(tolerance * tail_fraction),
        multiplicity=int(multiplicity),
        max_abs_lambda_logdet=(
            None
            if bases is None
            else max(abs(value) for value in bases) + lambda_logdet_margin
        ),
        max_x_operator_norm=(
            None if norms is None else max(norms) + x_operator_norm_margin
        ),
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


def audit_retained_lambda_logdet(
    lambda_logdets: Iterable[float], certificate: RhoCertificate
) -> AuditReport:
    """Recheck the absolute runtime base scale used by the roundoff proof."""
    if certificate.max_abs_lambda_logdet is None:
        raise ValueError("certificate has no lambda-logdet scale bound")
    values = tuple(abs(float(value)) for value in lambda_logdets)
    if not values or any(not np.isfinite(value) for value in values):
        raise ValueError("retained lambda logdets must be non-empty and finite")
    violations = tuple(
        index
        for index, value in enumerate(values)
        if value > certificate.max_abs_lambda_logdet
    )
    return AuditReport(not violations, max(values), violations)


def audit_retained_operator_norm(
    absolute_action_norms: Iterable[float], certificate: RhoCertificate
) -> AuditReport:
    """Recheck retained ``||abs(X)||_2`` values for a frozen-probe plan."""
    if certificate.max_x_operator_norm is None:
        raise ValueError("certificate has no |X| operator-norm bound")
    values = tuple(float(value) for value in absolute_action_norms)
    if not values or any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(
            "retained |X| operator norms must be non-empty and non-negative"
        )
    violations = tuple(
        index
        for index, value in enumerate(values)
        if value > certificate.max_x_operator_norm
    )
    return AuditReport(not violations, max(values), violations)


def audit_retained_power_traces(
    problems: Iterable[LogDetProblem], certificate: RhoCertificate
) -> TraceAuditReport:
    """Recheck theta-dependent exact traces on every retained sample.

    Warmup verifies the provider at probe points and binds its fixed order;
    runtime cannot branch on a traced equality check.  This post-run audit is
    therefore the second half of the provider's validity claim, just as
    :func:`audit_retained_rho` is for the spectral-radius certificate.  The
    retained audit is the only thing between a theta-dependent provider and a
    silent analytic-tail error that omitted provider error entirely.
    """
    retained = tuple(problems)
    if not retained:
        raise ValueError("retained trace audit needs at least one problem")
    violations = tuple(
        index
        for index, problem in enumerate(retained)
        if problem.trace_order != certificate.order
        or _retained_rank_exceeds_certificate(problem, certificate)
        or problem.exact_power_traces is None
        or not _power_traces_match(
            problem.lambda_matrix,
            problem.perturbation,
            problem.exact_power_traces,
            certificate.order,
        )
    )
    return TraceAuditReport(passed=not violations, violations=violations)


def _retained_rank_exceeds_certificate(
    problem: LogDetProblem, certificate: RhoCertificate
) -> bool:
    """Revalidate the warmup multiplicity proof at a retained theta."""
    try:
        rank_bound = _algebraic_rank_bound(
            problem.perturbation,
            problem.low_rank_factors,
            problem.lambda_matrix,
        )
    except ValueError:
        return True
    return rank_bound > certificate.multiplicity

def _validate_plan_certificate(
    problem: LogDetProblem, certificate: RhoCertificate
) -> None:
    if problem.trace_order != certificate.order:
        raise ValueError(
            "runtime plan order must be the warmup certificate-selected order "
            f"{certificate.order}; problem carries {problem.trace_order}"
        )
    required_multiplicity = _algebraic_rank_bound(
        problem.perturbation,
        problem.low_rank_factors,
        problem.lambda_matrix,
    )
    if certificate.multiplicity < required_multiplicity:
        raise ValueError(
            f"certificate multiplicity {certificate.multiplicity} is below "
            f"the problem's algebraic rank bound {required_multiplicity}"
        )
    if certificate.max_abs_lambda_logdet is None:
        raise ValueError(
            "runtime plan requires a warmup max_abs_lambda_logdet scale "
            "certificate; pass lambda_logdets to certify_warmup_rho"
        )
    if certificate.max_x_operator_norm is not None:
        x = _x_matrix(problem.lambda_matrix, problem.perturbation)
        actual_norm = (
            float(np.max(np.abs(x), initial=0.0))
            if x.ndim == 1
            else float(np.linalg.norm(np.abs(x), ord=2))
        )
        if actual_norm > certificate.max_x_operator_norm and not np.isclose(
            actual_norm,
            certificate.max_x_operator_norm,
            rtol=1e-12,
            atol=1e-14,
        ):
            raise ValueError(
                f"|X| operator-norm certificate {certificate.max_x_operator_norm} "
                f"understates measured norm {actual_norm}"
            )
    _validate_strict_rho(
        problem.lambda_matrix,
        problem.perturbation,
        certificate.certified_rho,
    )


def _validate_runtime_precision(
    problem: LogDetProblem, certificate: RhoCertificate, *, frozen: bool
) -> str:
    """Certify that runtime roundoff is smaller than the analytic tail budget.

    A final-output ULP is necessary but not sufficient: ``logdet(Lambda)`` can
    nearly cancel the trace series.  The second check therefore uses the
    absolute scale of every warmup term and a conservative gamma bound for the
    fixed number of runtime operations.
    """
    runtime_dtype = np.dtype(jnp.asarray(problem.lambda_matrix).dtype)
    if runtime_dtype.itemsize < np.dtype(np.float64).itemsize:
        raise ValueError(
            "a certified theta-dependent runtime plan requires float64; construct "
            "and run the plan inside `with jax.enable_x64(True):`"
        )
    sigma = problem.lambda_matrix + problem.perturbation
    if sigma.ndim == 1:
        if np.any(sigma <= 0.0):
            raise ValueError("runtime plan requires symmetric positive definite Sigma")
        expected = float(np.sum(np.log(sigma)))
    else:
        sign, expected = np.linalg.slogdet(sigma)
        if sign <= 0.0:
            raise ValueError(
                "runtime plan requires symmetric positive definite Sigma"
            )
    if not np.isfinite(expected):
        raise ValueError("runtime plan requires symmetric positive definite Sigma")
    rounded = np.asarray(abs(expected), dtype=runtime_dtype)
    ulp = float(abs(np.spacing(rounded)))
    if ulp > certificate.tolerance:
        raise ValueError(
            f"runtime {runtime_dtype} ULP {ulp:.8g} at expected logdet scale "
            f"{abs(expected):.8g} exceeds certificate tolerance "
            f"{certificate.tolerance:.8g}; construct and run the plan inside "
            "`with jax.enable_x64(True):` or relax the tolerance"
        )

    base_scale = certificate.max_abs_lambda_logdet
    if frozen:
        if certificate.max_x_operator_norm is None:
            raise ValueError(
                "frozen runtime plan requires a warmup max_x_operator_norm "
                "certificate for ||abs(X)||_2; pass x_operator_norms to "
                "certify_warmup_rho"
            )
        vectors = problem.frozen_probes.values
        probe_count = vectors.shape[0]
        probe_energy = float(np.mean(np.sum(vectors * vectors, axis=1)))
        series_scale = probe_energy * math.fsum(
            certificate.max_x_operator_norm**power / power
            for power in range(1, certificate.order + 1)
        )
    else:
        series_scale = certificate.multiplicity * math.fsum(
            certificate.certified_rho**power / power
            for power in range(1, certificate.order + 1)
        )

    epsilon = float(np.finfo(runtime_dtype).eps)
    n = _n(problem.lambda_matrix)
    operation_count = max(
        1,
        (4 * n + 4 + 2 * probe_count) * certificate.order + 4
        if frozen
        else 6 * certificate.order + 4,
    )
    product = operation_count * epsilon
    gamma = math.inf if product >= 1.0 else product / (1.0 - product)
    roundoff_bound = gamma * (base_scale + series_scale)
    analytic_tail = (
        0.0
        if frozen
        else whole_trace_log_tail_bound(
            certificate.certified_rho,
            certificate.order,
            certificate.multiplicity,
        )
    )
    if analytic_tail + roundoff_bound > certificate.tolerance:
        raise ValueError(
            f"runtime {runtime_dtype} analytic tail plus conservative roundoff "
            f"{analytic_tail + roundoff_bound:.8g} exceeds certificate tolerance "
            f"{certificate.tolerance:.8g}; construct and run the plan inside "
            "`with jax.enable_x64(True):` or relax the tolerance"
        )
    return runtime_dtype.str


def _require_runtime_precision(
    expected_dtype: str, lambda_logdet_value: Any, *values: Any
) -> None:
    """Keep a plan in a runtime precision at least as strong as warmup's."""
    expected = np.dtype(expected_dtype)
    base = jnp.asarray(lambda_logdet_value)
    if base.ndim != 0:
        raise ValueError("runtime lambda_logdet_value must be a scalar")
    actual = (np.dtype(base.dtype),) + tuple(
        np.dtype(jnp.asarray(value).dtype) for value in values
    )
    if (expected.itemsize > 4 and not jax.config.x64_enabled) or any(
        dtype.kind != "f" or dtype.itemsize < expected.itemsize
        for dtype in actual
    ):
        rendered = ", ".join(str(dtype) for dtype in actual)
        raise ValueError(
            f"runtime values must use real floating precision at least "
            f"the plan's certified {expected}; got {rendered}. Keep construction "
            "and execution inside the same `jax.enable_x64` context"
        )


def make_trace_log_plan(
    problem: LogDetProblem, certificate: RhoCertificate
) -> TraceLogPlan:
    """Validate exact traces and bind the certificate-selected runtime order."""
    _validate_plan_certificate(problem, certificate)
    if problem.exact_power_traces is None or not _power_traces_match(
        problem.lambda_matrix,
        problem.perturbation,
        problem.exact_power_traces,
        certificate.order,
    ):
        raise ValueError(
            "runtime trace plan requires bitwise exact power-trace evidence "
            "through the certificate-selected order"
        )
    runtime_dtype = _validate_runtime_precision(problem, certificate, frozen=False)
    return TraceLogPlan(_PLAN_TOKEN, certificate.order, runtime_dtype)


def make_frozen_trace_log_plan(
    problem: LogDetProblem, certificate: RhoCertificate
) -> FrozenTraceLogPlan:
    """Validate and capture immutable probes and certificate-selected order."""
    _validate_plan_certificate(problem, certificate)
    if problem.frozen_probes is None:
        raise ValueError("frozen runtime plan requires FrozenProbes")
    if problem.frozen_probes.values.shape[1] != _n(problem.lambda_matrix):
        raise ValueError(
            "frozen runtime probe width must equal the matrix dimension"
        )
    runtime_dtype = _validate_runtime_precision(problem, certificate, frozen=True)
    return FrozenTraceLogPlan(
        _PLAN_TOKEN, certificate.order, problem.frozen_probes, runtime_dtype
    )
