"""Warmup certificates, retained audits, and fixed-order JAX plans."""

from __future__ import annotations

import dataclasses
import math
import operator
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
    _two_sum_error,
    _validate_strict_rho,
    _x_matrix,
    choose_trace_order,
    lambda_logdet,
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
_FLOAT64_EPSILON = float(np.finfo(np.float64).eps)
_RHO_MULTIPLICITY_LIMIT = int(1.0 / _FLOAT64_EPSILON)


def _normalize_rho_multiplicity(value: Any) -> int:
    """Return one canonical dimension for both gamma_n and the trace tail."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(
            "rho certificate multiplicity must be an integer index, not bool"
        )
    try:
        multiplicity = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(
            "rho certificate multiplicity must be an integer index"
        ) from error
    if multiplicity < 1:
        raise ValueError("rho certificate multiplicity must be positive")
    if multiplicity >= _RHO_MULTIPLICITY_LIMIT:
        raise ValueError(
            "rho certificate multiplicity must satisfy multiplicity * float64 "
            "eps < 1"
        )
    return int(multiplicity)


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
        multiplicity = _normalize_rho_multiplicity(self.multiplicity)
        object.__setattr__(self, "multiplicity", multiplicity)
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
    The certified rho includes the standard binary64 ``gamma_n`` arithmetic
    envelope, ``n eps / (1 - n eps)``, with ``n=multiplicity``.  It scales
    with ``abs(rho)`` and therefore gives an exactly zero measurement no
    absolute tolerance floor.  This covers ordinary dimension-scaled
    remeasurement roundoff; conditioning-amplified solve drift belongs in the
    caller's explicit margin and is caught by the retained audit otherwise.
    """
    multiplicity = _normalize_rho_multiplicity(multiplicity)
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
    raw_bound = float(measured_max + float(margin))
    roundoff_product = multiplicity * _FLOAT64_EPSILON
    gamma_n = roundoff_product / (1.0 - roundoff_product)
    arithmetic_envelope = abs(raw_bound) * gamma_n
    certified = math.nextafter(raw_bound + arithmetic_envelope, math.inf)
    if not certified < 1.0:
        raise ValueError(
            f"warmup maximum {measured_max} plus margin {margin} and its float64 "
            "arithmetic envelope does not certify rho < 1"
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
        multiplicity=multiplicity,
        max_abs_lambda_logdet=(
            None
            if bases is None
            else math.nextafter(
                max(abs(value) for value in bases) + lambda_logdet_margin,
                math.inf,
            )
        ),
        max_x_operator_norm=(
            None
            if norms is None
            else math.nextafter(
                max(norms) + x_operator_norm_margin,
                math.inf,
            )
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
        or not _checked_power_traces_match(
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
    except (ValueError, ArithmeticError):
        return True
    return rank_bound > certificate.multiplicity


def _checked_power_traces_match(
    lambda_matrix: np.ndarray,
    perturbation: np.ndarray,
    traces: tuple[float, ...],
    order: int,
) -> bool:
    """Return false when exact-trace verification arithmetic is unresolved."""
    try:
        with np.errstate(
            divide="raise", invalid="raise", over="raise", under="ignore"
        ):
            return _power_traces_match(
                lambda_matrix,
                perturbation,
                traces,
                order,
            )
    except (ArithmeticError, np.linalg.LinAlgError):
        return False


def _checked_x_operator_norm(
    lambda_matrix: np.ndarray, perturbation: np.ndarray
) -> float:
    """Measure ``||abs(X)||_2`` or refuse unresolved floating arithmetic."""
    failure = (
        "|X| operator-norm validation requires finite resolved "
        "Lambda^-1 perturbation arithmetic"
    )
    try:
        with np.errstate(
            divide="raise", invalid="raise", over="raise", under="ignore"
        ):
            x = _x_matrix(lambda_matrix, perturbation)
            actual_norm = (
                float(np.max(np.abs(x), initial=0.0))
                if x.ndim == 1
                else float(np.linalg.norm(np.abs(x), ord=2))
            )
    except (ArithmeticError, np.linalg.LinAlgError) as error:
        raise ValueError(failure) from error
    if not np.all(np.isfinite(x)) or not np.isfinite(actual_norm):
        raise ValueError(failure)
    return actual_norm


def _checked_lambda_logdet_scale(lambda_matrix: np.ndarray) -> float:
    """Measure ``abs(logdet(Lambda))`` or refuse unresolved arithmetic."""
    failure = (
        "lambda-logdet scale validation requires finite resolved "
        "lambda-logdet arithmetic"
    )
    try:
        with np.errstate(
            divide="raise", invalid="raise", over="raise", under="ignore"
        ):
            actual_scale = abs(float(lambda_logdet(lambda_matrix)))
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
        raise ValueError(failure) from error
    if not np.isfinite(actual_scale):
        raise ValueError(failure)
    return actual_scale


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
    actual_base_scale = _checked_lambda_logdet_scale(problem.lambda_matrix)
    if actual_base_scale > certificate.max_abs_lambda_logdet:
        raise ValueError(
            f"lambda-logdet scale certificate "
            f"{certificate.max_abs_lambda_logdet} understates measured absolute "
            f"logdet {actual_base_scale}"
        )
    if certificate.max_x_operator_norm is not None:
        actual_norm = _checked_x_operator_norm(
            problem.lambda_matrix, problem.perturbation
        )
        if actual_norm > certificate.max_x_operator_norm:
            raise ValueError(
                f"|X| operator-norm certificate {certificate.max_x_operator_norm} "
                f"understates measured norm {actual_norm}"
            )
    _validate_strict_rho(
        problem.lambda_matrix,
        problem.perturbation,
        certificate.certified_rho,
    )


def _canonical_runtime_dtype(problem: LogDetProblem) -> np.dtype:
    """Return the dtype JAX will use for the problem's dynamic matrix values."""
    return np.dtype(jax.dtypes.canonicalize_dtype(problem.lambda_matrix.dtype))


def _canonical_runtime_probes(
    probes: FrozenProbes, runtime_dtype: np.dtype
) -> FrozenProbes:
    """Capture probes once in the exact dtype used by every runtime context.

    A plan built from float32 matrices is allowed both inside and outside an
    x64-enabled JAX context.  Retaining binary64 probe bytes would therefore
    make the target context-dependent.  Underflow is safe because the
    canonical zero is what the plan captures; overflow is not.
    """
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            values = np.asarray(probes.values, dtype=runtime_dtype)
    except (FloatingPointError, OverflowError, ValueError) as error:
        raise ValueError(
            "captured frozen probes must remain finite after conversion to "
            f"runtime {runtime_dtype.name}"
        ) from error
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "captured frozen probes must remain finite after conversion to "
            f"runtime {runtime_dtype.name}"
        )
    return FrozenProbes(values)


def _outward_nonnegative(value: float) -> float:
    """Round a non-negative binary64 proof quantity toward positive infinity."""
    if value == 0.0 or not np.isfinite(value):
        return value
    return math.nextafter(value, math.inf)


def _outward_product(left: float, right: float) -> float:
    """Return an outward binary64 bound for a non-negative product."""
    if left == 0.0 or right == 0.0:
        return 0.0
    product = left * right
    if product == 0.0:
        return math.nextafter(0.0, math.inf)
    return _outward_nonnegative(product)


def _outward_sum(left: float, right: float) -> float:
    """Return an outward binary64 bound for a non-negative sum."""
    if left == 0.0:
        return right
    if right == 0.0:
        return left
    return _outward_nonnegative(left + right)


def _outward_quotient(numerator: float, denominator: int) -> float:
    """Return an outward binary64 bound for division by a positive integer."""
    if numerator == 0.0:
        return 0.0
    quotient = numerator / denominator
    if quotient == 0.0:
        return math.nextafter(0.0, math.inf)
    return _outward_nonnegative(quotient)


def _outward_power_series(base: float, order: int) -> float:
    """Bound ``sum(base**p / p)`` without inward-rounded recurrence steps."""
    power = 1.0
    terms: list[float] = []
    for exponent in range(1, order + 1):
        power = _outward_product(power, base)
        terms.append(_outward_quotient(power, exponent))
    total = math.fsum(terms)
    return _outward_nonnegative(total)


def _frozen_probe_energy_bounds(
    vectors: np.ndarray, runtime_dtype: np.dtype
) -> tuple[float, float]:
    """Return outward total energy and maximum probe norm within dtype range."""
    maximum = float(np.finfo(runtime_dtype).max)
    square_root_maximum = math.sqrt(maximum)
    energies: list[float] = []
    maximum_energy = 0.0
    try:
        for vector in vectors:
            magnitudes = tuple(abs(float(value)) for value in vector)
            if any(value > square_root_maximum for value in magnitudes):
                raise OverflowError
            energy = math.fsum(
                _outward_product(value, value) for value in magnitudes
            )
            energy = _outward_nonnegative(energy)
            if not np.isfinite(energy) or energy > maximum:
                raise OverflowError
            energies.append(energy)
            maximum_energy = max(maximum_energy, energy)
        total_energy = _outward_nonnegative(math.fsum(energies))
    except (ArithmeticError, OverflowError) as error:
        raise ValueError(
            f"runtime {runtime_dtype.name} range requires finite frozen probe energy"
        ) from error
    if not np.isfinite(total_energy) or total_energy > maximum:
        raise ValueError(
            f"runtime {runtime_dtype.name} range requires finite frozen probe energy"
        )
    maximum_norm = math.sqrt(maximum_energy)
    maximum_norm = _outward_nonnegative(maximum_norm)
    return total_energy, maximum_norm


def _runtime_range_product(
    left: float,
    right: float,
    maximum: float,
    runtime_dtype: np.dtype,
    quantity: str,
) -> float:
    """Multiply non-negative bounds or raise before the runtime can overflow."""
    if left == 0.0 or right == 0.0:
        return 0.0
    if (
        not np.isfinite(left)
        or not np.isfinite(right)
        or left > maximum / right
    ):
        raise ValueError(
            f"runtime {runtime_dtype.name} range cannot certify {quantity}"
        )
    result = _outward_product(left, right)
    if not np.isfinite(result) or result > maximum:
        raise ValueError(
            f"runtime {runtime_dtype.name} range cannot certify {quantity}"
        )
    return result


def _runtime_range_sum(
    left: float,
    right: float,
    maximum: float,
    runtime_dtype: np.dtype,
    quantity: str,
) -> float:
    """Add non-negative bounds or raise before the runtime can overflow."""
    if (
        not np.isfinite(left)
        or not np.isfinite(right)
        or right > maximum
        or left > maximum - right
    ):
        raise ValueError(
            f"runtime {runtime_dtype.name} range cannot certify {quantity}"
        )
    result = _outward_sum(left, right)
    if not np.isfinite(result) or result > maximum:
        raise ValueError(
            f"runtime {runtime_dtype.name} range cannot certify {quantity}"
        )
    return result


def _gamma_for_count(count: int, epsilon: float) -> float:
    """Return the standard gamma bound, or infinity outside its domain."""
    product = count * epsilon
    if product >= 1.0:
        return math.inf
    return _outward_nonnegative(product / (1.0 - product))


def _one_plus_gamma(count: int, epsilon: float) -> float:
    """Return an outward bound for ``1 + gamma_count``."""
    return _outward_sum(1.0, _gamma_for_count(count, epsilon))


def _validate_frozen_runtime_range(
    *,
    runtime_dtype: np.dtype,
    certificate: RhoCertificate,
    total_probe_energy: float,
    maximum_probe_norm: float,
    probe_count: int,
    dimension: int,
) -> float:
    """Bound every frozen-kernel intermediate and its correction accumulation."""
    maximum = float(np.finfo(runtime_dtype).max)
    x_bound = float(certificate.max_x_operator_norm)
    if x_bound > maximum:
        raise ValueError(
            f"runtime {runtime_dtype.name} range cannot certify the X "
            "operator-norm bound"
        )

    epsilon = float(np.finfo(runtime_dtype).eps)
    matvec_factor = _one_plus_gamma(dimension, epsilon)
    dot_factor = matvec_factor
    reduction_factor = _one_plus_gamma(probe_count, epsilon)
    addition_factor = _one_plus_gamma(certificate.order, epsilon)

    image_bound = maximum_probe_norm
    total_dot_scale = total_probe_energy
    trace_term_bounds: list[float] = []
    for power in range(1, certificate.order + 1):
        image_bound = _runtime_range_product(
            image_bound,
            x_bound,
            maximum,
            runtime_dtype,
            f"the frozen image at power {power}",
        )
        image_bound = _runtime_range_product(
            image_bound,
            matvec_factor,
            maximum,
            runtime_dtype,
            f"the rounded frozen image at power {power}",
        )
        total_dot_scale = _runtime_range_product(
            total_dot_scale,
            x_bound,
            maximum,
            runtime_dtype,
            f"the frozen probe-image products at power {power}",
        )
        total_dot_scale = _runtime_range_product(
            total_dot_scale,
            matvec_factor,
            maximum,
            runtime_dtype,
            f"the rounded frozen probe-image products at power {power}",
        )
        reduced_sum_bound = _runtime_range_product(
            total_dot_scale,
            dot_factor,
            maximum,
            runtime_dtype,
            f"the frozen dot products at power {power}",
        )
        reduced_sum_bound = _runtime_range_product(
            reduced_sum_bound,
            reduction_factor,
            maximum,
            runtime_dtype,
            f"the frozen probe reduction at power {power}",
        )
        mean_bound = _outward_quotient(reduced_sum_bound, probe_count)
        trace_term_bounds.append(_outward_quotient(mean_bound, power))

    try:
        correction_bound = _outward_nonnegative(math.fsum(trace_term_bounds))
    except ArithmeticError as error:
        raise ValueError(
            f"runtime {runtime_dtype.name} range cannot certify the frozen "
            "correction accumulation"
        ) from error
    return _runtime_range_product(
        correction_bound,
        addition_factor,
        maximum,
        runtime_dtype,
        "the frozen correction accumulation",
    )


def _validate_runtime_precision(
    problem: LogDetProblem,
    certificate: RhoCertificate,
    *,
    frozen: bool,
    frozen_probes: FrozenProbes | None = None,
) -> str:
    """Certify that runtime roundoff is smaller than the analytic tail budget.

    A final-output ULP is necessary but not sufficient: ``logdet(Lambda)`` can
    nearly cancel the trace series.  The second check therefore uses the
    absolute scale of every warmup term and a conservative gamma bound for the
    fixed number of runtime operations.
    """
    runtime_dtype = _canonical_runtime_dtype(problem)
    try:
        sigma, _ = _two_sum_error(
            problem.lambda_matrix,
            problem.perturbation,
        )
    except ValueError as error:
        raise ValueError(
            "runtime precision validation cannot certify finite "
            "Lambda + perturbation arithmetic"
        ) from error

    try:
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            if sigma.ndim == 1:
                if np.any(sigma <= 0.0):
                    raise ValueError(
                        "runtime plan requires symmetric positive definite Sigma"
                    )
                expected = float(np.sum(np.log(sigma)))
            else:
                sign, expected = np.linalg.slogdet(sigma)
                if sign <= 0.0:
                    raise ValueError(
                        "runtime plan requires symmetric positive definite Sigma"
                    )
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError(
            "runtime precision validation could not compute a finite expected "
            "logdet from Lambda + perturbation"
        ) from error
    if not np.isfinite(expected):
        raise ValueError(
            "runtime precision validation requires a finite expected logdet "
            "from Lambda + perturbation"
        )

    try:
        with np.errstate(invalid="raise", over="raise", under="ignore"):
            rounded = np.asarray(abs(expected), dtype=runtime_dtype)
            ulp = float(abs(np.spacing(rounded)))
    except FloatingPointError as error:
        raise ValueError(
            "runtime precision validation could not represent a finite "
            "expected-logdet ULP"
        ) from error
    if not np.isfinite(rounded) or not np.isfinite(ulp):
        raise ValueError(
            "runtime precision validation requires a finite expected-logdet ULP"
        )
    if ulp > certificate.tolerance:
        raise ValueError(
            f"runtime {runtime_dtype} ULP {ulp:.8g} at expected logdet scale "
            f"{abs(expected):.8g} exceeds certificate tolerance "
            f"{certificate.tolerance:.8g}; use a wider input/runtime dtype or "
            "relax the tolerance"
        )

    base_scale = certificate.max_abs_lambda_logdet
    maximum_runtime_value = float(np.finfo(runtime_dtype).max)
    if base_scale > maximum_runtime_value:
        raise ValueError(
            f"runtime {runtime_dtype.name} range cannot certify the "
            "lambda-logdet scale bound"
        )
    if frozen:
        if certificate.max_x_operator_norm is None:
            raise ValueError(
                "frozen runtime plan requires a warmup max_x_operator_norm "
                "certificate for ||abs(X)||_2; pass x_operator_norms to "
                "certify_warmup_rho"
            )
        if frozen_probes is None:
            raise ValueError(
                "runtime precision validation requires canonical frozen probes"
            )
        vectors = frozen_probes.values
        probe_count = vectors.shape[0]
        total_probe_energy, maximum_probe_norm = _frozen_probe_energy_bounds(
            vectors, runtime_dtype
        )
        probe_energy = _outward_quotient(total_probe_energy, probe_count)
        try:
            series_scale = _outward_product(
                probe_energy,
                _outward_power_series(
                    certificate.max_x_operator_norm, certificate.order
                ),
            )
        except ArithmeticError as error:
            raise ValueError(
                "runtime precision validation requires a finite frozen series scale"
            ) from error
        if not np.isfinite(series_scale):
            raise ValueError(
                "runtime precision validation requires a finite frozen series scale"
            )
    else:
        series_scale = _outward_product(
            float(certificate.multiplicity),
            _outward_power_series(
                certificate.certified_rho, certificate.order
            ),
        )

    epsilon = float(np.finfo(runtime_dtype).eps)
    n = _n(problem.lambda_matrix)
    operation_count = max(
        1,
        (4 * n + 4 + 2 * probe_count) * certificate.order + 4
        if frozen
        else 6 * certificate.order + 4,
    )
    gamma = _gamma_for_count(operation_count, epsilon)
    frozen_correction_bound = None
    if frozen:
        frozen_correction_bound = _validate_frozen_runtime_range(
            runtime_dtype=runtime_dtype,
            certificate=certificate,
            total_probe_energy=total_probe_energy,
            maximum_probe_norm=maximum_probe_norm,
            probe_count=probe_count,
            dimension=n,
        )
    base_and_series = _runtime_range_sum(
        base_scale,
        series_scale,
        maximum_runtime_value,
        runtime_dtype,
        "the lambda-logdet plus trace-series scale",
    )
    roundoff_bound = _outward_product(gamma, base_and_series)
    series_with_roundoff = _outward_sum(series_scale, roundoff_bound)
    correction_range_bound = (
        series_with_roundoff
        if frozen_correction_bound is None
        else max(series_with_roundoff, frozen_correction_bound)
    )
    _runtime_range_sum(
        base_scale,
        correction_range_bound,
        maximum_runtime_value,
        runtime_dtype,
        "the final logdet accumulation",
    )
    analytic_tail = (
        0.0
        if frozen
        else _outward_nonnegative(
            whole_trace_log_tail_bound(
                certificate.certified_rho,
                certificate.order,
                certificate.multiplicity,
            )
        )
    )
    total_error_bound = _outward_sum(analytic_tail, roundoff_bound)
    if total_error_bound > certificate.tolerance:
        raise ValueError(
            f"runtime {runtime_dtype} analytic tail plus conservative roundoff "
            f"{total_error_bound:.8g} exceeds certificate tolerance "
            f"{certificate.tolerance:.8g}; use a wider input/runtime dtype or "
            "relax the tolerance"
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
    if problem.exact_power_traces is None or not _checked_power_traces_match(
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
    runtime_dtype = _canonical_runtime_dtype(problem)
    canonical_probes = _canonical_runtime_probes(
        problem.frozen_probes, runtime_dtype
    )
    runtime_dtype_name = _validate_runtime_precision(
        problem,
        certificate,
        frozen=True,
        frozen_probes=canonical_probes,
    )
    return FrozenTraceLogPlan(
        _PLAN_TOKEN, certificate.order, canonical_probes, runtime_dtype_name
    )
