"""Direct, lineage-preserving boundary suites for dynamic PLAN gates."""

from __future__ import annotations

import ast
import math
from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from bayesmith.marginal import _logdet_eager as eager
from bayesmith.marginal import _logdet_plan as plan
from tests.numerical_gates import oracles
from tests.numerical_gates.boundary_contract import REQUIRED_DIRECT_CALLS
from tests.numerical_gates.boundary_core import (
    AtomBaseline,
    AtomDependencyLogic,
    AtomEvidence,
    AtomPrerequisite,
    AtomReducer,
    AtomRelation,
    AtomRelationKind,
    AxisPosition,
    BoundarySuite,
    BoundaryTopology,
    ExecutionClass,
    FixtureFamily,
    GateSide,
    PointRole,
    RawObservation,
    ThresholdPoint,
    axis_position_for_role,
    capability_grid,
    exact_grid,
    float_grid,
    freeze_suite,
    integer_grid,
    make_atom_case,
    make_case,
    make_grid_cases,
    oracle_check,
    realized_point,
    source_alias_canonical,
    source_ast_prerequisites,
)
from tests.numerical_gates.registry import (
    GATE_REGISTRY,
    FixtureScalePolicy,
    GateEntry,
    MutationMode,
    isolatable_atom_ids,
)
from tests.numerical_gates.source_manifest import EXPECTED_SOURCE_MANIFEST
from tests.numerical_gates.source_scan import index_source_text

_ORIGINAL_NEXTAFTER = math.nextafter
_ENTRIES = {
    entry.gate_id: entry
    for entry in GATE_REGISTRY
    if entry.gate_id.startswith("PLAN:")
    and entry.mutation_mode is MutationMode.TWO_SIDED
}
_SYNTAX = {item.candidate_id: item.syntax for item in EXPECTED_SOURCE_MANIFEST}
_NON_UNIT = (1.7, 2.9)
_ORACLE_RHO_MULTIPLICITY_LIMIT = int(
    Decimal(1) / Decimal.from_float(float(np.finfo(np.float64).eps))
)
_FLOAT64_MIN_SUBNORMAL = float(np.nextafter(0.0, math.inf))
_FLOAT64_LAST_FINITE = float(np.finfo(np.float64).max)
_FINITE_LIMIT_GATES = frozenset(set())
_PLAN_SOURCE_PATH = Path(__file__).parents[2] / "src/bayesmith/marginal/_logdet_plan.py"
_PLAN_SOURCE = _PLAN_SOURCE_PATH.read_text()
_PLAN_SOURCE_INDEX = index_source_text(
    _PLAN_SOURCE, "src/bayesmith/marginal/_logdet_plan.py"
)
_ATOM_PROBE_NAME = "__bayesmith_boundary_atom_probe__"
_ACTIVE_ATOM_RECORDERS: list[dict[str, list[Any]]] = []


def _review_extreme_value(point: ThresholdPoint) -> float | None:
    return {
        "minimum positive float64 subnormal": _FLOAT64_MIN_SUBNORMAL,
        "last finite float64": _FLOAT64_LAST_FINITE,
        "first float64 overflow": math.inf,
        "last finite float32 with non-scalar fixture": float(np.finfo(np.float32).max),
    }.get(point.display_value)


@dataclass(frozen=True, slots=True)
class _AtomSpec:
    atom_id: str
    index: int
    syntax: str


@dataclass(frozen=True, slots=True)
class _AtomValue:
    raw: Any
    truth: bool | None
    reducer: AtomReducer = AtomReducer.SCALAR
    provenance: str = ""


@dataclass(frozen=True, slots=True)
class _Outcome:
    observed_side: GateSide
    oracle_side: GateSide
    returns: Mapping[str, Any]
    actual_atoms: Mapping[str, _AtomValue] = MappingProxyType({})
    oracle_atoms: Mapping[str, _AtomValue] = MappingProxyType({})
    payload_checks: tuple[tuple[str, Any, Any, bool], ...] = ()
    notes: tuple[str, ...] = ()
    direct_calls: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        actual_atoms = self.actual_atoms
        oracle_atoms = self.oracle_atoms
        if actual_atoms or oracle_atoms:
            if actual_atoms is oracle_atoms:
                raise AssertionError("production and oracle atom rows are aliased")
            if set(actual_atoms) != set(oracle_atoms):
                raise AssertionError("production and oracle atom IDs differ")
            for atom_id in actual_atoms:
                actual = actual_atoms[atom_id]
                oracle = oracle_atoms[atom_id]
                if actual is oracle:
                    raise AssertionError(
                        f"{atom_id} production and oracle atom values are aliased"
                    )
                if not actual.provenance.startswith("production:"):
                    raise AssertionError(f"{atom_id} has no production atom provenance")
                if not oracle.provenance.startswith("oracle:"):
                    raise AssertionError(f"{atom_id} has no oracle atom provenance")
                if actual.provenance == oracle.provenance:
                    raise AssertionError(f"{atom_id} atom provenances are identical")
        object.__setattr__(self, "returns", MappingProxyType(dict(self.returns)))
        object.__setattr__(
            self, "actual_atoms", MappingProxyType(dict(self.actual_atoms))
        )
        object.__setattr__(
            self, "oracle_atoms", MappingProxyType(dict(self.oracle_atoms))
        )


@dataclass(frozen=True, slots=True)
class _Fixture:
    inputs: Mapping[str, Any]
    point_key: str
    threshold: Any
    dtype: str | None
    axis_key: str
    oracle_name: str
    invoke: Callable[[Mapping[str, Any]], _Outcome]

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))


@dataclass(frozen=True, slots=True)
class _ParameterAxisCell:
    field: str
    value: float
    threshold: float
    point: ThresholdPoint


def _parameter_axis_cell(
    field: str,
    name: str,
    role: PointRole,
    value: float,
    threshold: float,
    side: GateSide,
) -> _ParameterAxisCell:
    return _ParameterAxisCell(
        field,
        value,
        threshold,
        ThresholdPoint(
            role,
            f"parameter-axis:{field}:{name}",
            "real input-parameter axis",
            side,
        ),
    )


_ERROR_BUDGET_AXIS_CELLS = (
    _parameter_axis_cell(
        "margin", "very-low", PointRole.VERY_LOW, -1.0, 0.0, GateSide.REFUSED
    ),
    _parameter_axis_cell(
        "margin",
        "below-ulp",
        PointRole.BELOW_ULP,
        -_FLOAT64_MIN_SUBNORMAL,
        0.0,
        GateSide.REFUSED,
    ),
    _parameter_axis_cell("margin", "at", PointRole.AT, 0.0, 0.0, GateSide.ADMITTED),
    _parameter_axis_cell(
        "margin",
        "above-ulp",
        PointRole.ABOVE_ULP,
        _FLOAT64_MIN_SUBNORMAL,
        0.0,
        GateSide.ADMITTED,
    ),
    _parameter_axis_cell(
        "margin", "very-high", PointRole.VERY_HIGH, 8.0, 0.0, GateSide.ADMITTED
    ),
    _parameter_axis_cell(
        "margin", "nan", PointRole.EXTREME, math.nan, 0.0, GateSide.ADMITTED
    ),
    _parameter_axis_cell(
        "margin", "+inf", PointRole.EXTREME, math.inf, 0.0, GateSide.ADMITTED
    ),
    _parameter_axis_cell(
        "margin", "-inf", PointRole.EXTREME, -math.inf, 0.0, GateSide.REFUSED
    ),
    _parameter_axis_cell(
        "tolerance",
        "very-low",
        PointRole.VERY_LOW,
        -1.0,
        0.0,
        GateSide.REFUSED,
    ),
    _parameter_axis_cell(
        "tolerance",
        "below-ulp",
        PointRole.BELOW_ULP,
        -_FLOAT64_MIN_SUBNORMAL,
        0.0,
        GateSide.REFUSED,
    ),
    _parameter_axis_cell("tolerance", "at", PointRole.AT, 0.0, 0.0, GateSide.REFUSED),
    _parameter_axis_cell(
        "tolerance",
        "above-ulp",
        PointRole.ABOVE_ULP,
        _FLOAT64_MIN_SUBNORMAL,
        0.0,
        GateSide.REFUSED,
    ),
    _parameter_axis_cell(
        "tolerance",
        "very-high",
        PointRole.VERY_HIGH,
        1.0,
        0.0,
        GateSide.ADMITTED,
    ),
    _parameter_axis_cell(
        "tolerance", "nan", PointRole.EXTREME, math.nan, 0.0, GateSide.REFUSED
    ),
    _parameter_axis_cell(
        "tolerance", "+inf", PointRole.EXTREME, math.inf, 0.0, GateSide.ADMITTED
    ),
    _parameter_axis_cell(
        "tolerance", "-inf", PointRole.EXTREME, -math.inf, 0.0, GateSide.REFUSED
    ),
    _parameter_axis_cell(
        "tail_tolerance",
        "lower-very-low",
        PointRole.VERY_LOW,
        -1.0,
        0.0,
        GateSide.REFUSED,
    ),
    _parameter_axis_cell(
        "tail_tolerance",
        "lower-below-ulp",
        PointRole.BELOW_ULP,
        -_FLOAT64_MIN_SUBNORMAL,
        0.0,
        GateSide.REFUSED,
    ),
    _parameter_axis_cell(
        "tail_tolerance",
        "lower-at",
        PointRole.AT,
        0.0,
        0.0,
        GateSide.REFUSED,
    ),
    _parameter_axis_cell(
        "tail_tolerance",
        "lower-above-ulp",
        PointRole.ABOVE_ULP,
        _FLOAT64_MIN_SUBNORMAL,
        0.0,
        GateSide.ADMITTED,
    ),
    _parameter_axis_cell(
        "tail_tolerance",
        "lower-very-high",
        PointRole.VERY_HIGH,
        0.25,
        0.0,
        GateSide.ADMITTED,
    ),
    _parameter_axis_cell(
        "tail_tolerance",
        "nan",
        PointRole.EXTREME,
        math.nan,
        0.0,
        GateSide.REFUSED,
    ),
    _parameter_axis_cell(
        "tail_tolerance",
        "+inf",
        PointRole.EXTREME,
        math.inf,
        0.0,
        GateSide.REFUSED,
    ),
    _parameter_axis_cell(
        "tail_tolerance",
        "-inf",
        PointRole.EXTREME,
        -math.inf,
        0.0,
        GateSide.REFUSED,
    ),
)
_ERROR_BUDGET_AXIS_CELL_BY_DISPLAY = MappingProxyType(
    {cell.point.display_value: cell for cell in _ERROR_BUDGET_AXIS_CELLS}
)


@dataclass(frozen=True, slots=True)
class _AxisOverrideCell:
    gate_id: str
    input_key: str
    threshold: Any
    point: ThresholdPoint
    overrides: Mapping[str, Any]
    realized_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "overrides", MappingProxyType(dict(self.overrides)))


def _scalar_override_axis(
    *,
    gate_id: str,
    input_key: str,
    threshold: float,
    low: float,
    high: float,
    admitted: Callable[[float], bool],
    overrides: Callable[[float], Mapping[str, Any]],
    realized_key: str | None = None,
) -> tuple[_AxisOverrideCell, ...]:
    values = (
        ("very-low", PointRole.VERY_LOW, low),
        (
            "below-ulp",
            PointRole.BELOW_ULP,
            _ORIGINAL_NEXTAFTER(threshold, -math.inf),
        ),
        ("at", PointRole.AT, threshold),
        (
            "above-ulp",
            PointRole.ABOVE_ULP,
            _ORIGINAL_NEXTAFTER(threshold, math.inf),
        ),
        ("very-high", PointRole.VERY_HIGH, high),
        ("nan", PointRole.EXTREME, math.nan),
        ("+inf", PointRole.EXTREME, math.inf),
        ("-inf", PointRole.EXTREME, -math.inf),
    )
    cells = []
    for name, role, value in values:
        side = GateSide.ADMITTED if admitted(value) else GateSide.REFUSED
        cells.append(
            _AxisOverrideCell(
                gate_id,
                input_key,
                threshold,
                ThresholdPoint(
                    role,
                    f"parameter-axis:{gate_id}:{input_key}:{name}",
                    "real input-parameter axis",
                    side,
                ),
                overrides(value),
                realized_key,
            )
        )
    return tuple(cells)


def _finite(value: float) -> bool:
    return math.isfinite(value)


def _integer_override_axis(
    *,
    gate_id: str,
    input_key: str,
    threshold: int,
    low: int,
    high: int,
    extreme: int,
    admitted: Callable[[int], bool],
    overrides: Callable[[int], Mapping[str, Any]],
    realized_key: str | None = None,
) -> tuple[_AxisOverrideCell, ...]:
    values = (
        ("very-low", PointRole.VERY_LOW, low),
        ("below", PointRole.BELOW_INTEGER, threshold - 1),
        ("at", PointRole.AT, threshold),
        ("above", PointRole.ABOVE_INTEGER, threshold + 1),
        ("very-high", PointRole.VERY_HIGH, high),
        ("extreme", PointRole.EXTREME, extreme),
    )
    return tuple(
        _AxisOverrideCell(
            gate_id,
            input_key,
            threshold,
            ThresholdPoint(
                role,
                f"parameter-axis:{gate_id}:{input_key}:{name}",
                "real integer input-parameter axis",
                GateSide.ADMITTED if admitted(value) else GateSide.REFUSED,
            ),
            overrides(value),
            realized_key,
        )
        for name, role, value in values
    )


def _finite_scalar_override_axis(
    *,
    gate_id: str,
    input_key: str,
    threshold: float,
    low: float,
    high: float,
    extreme: float,
    admitted: Callable[[float], bool],
    overrides: Callable[[float], Mapping[str, Any]],
    realized_key: str | None = None,
) -> tuple[_AxisOverrideCell, ...]:
    values = (
        ("very-low", PointRole.VERY_LOW, low),
        (
            "below-ulp",
            PointRole.BELOW_ULP,
            _ORIGINAL_NEXTAFTER(threshold, -math.inf),
        ),
        ("at", PointRole.AT, threshold),
        (
            "above-ulp",
            PointRole.ABOVE_ULP,
            _ORIGINAL_NEXTAFTER(threshold, math.inf),
        ),
        ("very-high", PointRole.VERY_HIGH, high),
        ("finite-extreme", PointRole.EXTREME, extreme),
    )
    return tuple(
        _AxisOverrideCell(
            gate_id,
            input_key,
            threshold,
            ThresholdPoint(
                role,
                f"parameter-axis:{gate_id}:{input_key}:{name}",
                "finite real input-parameter axis",
                GateSide.ADMITTED if admitted(value) else GateSide.REFUSED,
            ),
            overrides(value),
            realized_key,
        )
        for name, role, value in values
    )


def _capability_override_axis(
    *,
    gate_id: str,
    input_key: str,
    cells: tuple[
        tuple[
            str,
            PointRole,
            str,
            GateSide,
            Mapping[str, Any],
        ],
        ...,
    ],
) -> tuple[_AxisOverrideCell, ...]:
    return tuple(
        _AxisOverrideCell(
            gate_id,
            input_key,
            "capability path",
            ThresholdPoint(
                role,
                f"parameter-axis:{gate_id}:{input_key}:{name}",
                "real categorical production path",
                side,
            ),
            overrides,
        )
        for name, role, _value, side, overrides in cells
    )


def _range_product_admitted(left: float, right: float, maximum: float) -> bool:
    if left == 0.0 or right == 0.0:
        return True
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    if left > maximum / right:
        return False
    rounded = left * right
    result = math.nextafter(rounded, math.inf) if rounded != 0.0 else rounded
    return math.isfinite(result) and not result > maximum


def _range_sum_admitted(left: float, right: float, maximum: float) -> bool:
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    if right > maximum or left > maximum - right:
        return False
    rounded = left + right
    result = (
        rounded if left == 0.0 or right == 0.0 else math.nextafter(rounded, math.inf)
    )
    return math.isfinite(result) and not result > maximum


def _rounded_float32_admitted(value: float) -> bool:
    if not math.isfinite(value):
        return False
    try:
        with np.errstate(over="raise", invalid="raise"):
            rounded = np.asarray(abs(value), dtype=np.float32)
            ulp = float(abs(np.spacing(rounded)))
    except FloatingPointError:
        return False
    return bool(np.isfinite(rounded) and math.isfinite(ulp))


def _literal_outward_nonnegative(value: float) -> float:
    if value == 0.0 or not math.isfinite(value):
        return value
    return math.nextafter(value, math.inf)


def _literal_outward_sum(left: float, right: float) -> float:
    if left == 0.0:
        return right
    if right == 0.0:
        return left
    return _literal_outward_nonnegative(left + right)


def _total_error_value(analytic_tail: float, roundoff_error: float) -> float:
    return _literal_outward_sum(
        _literal_outward_nonnegative(analytic_tail), roundoff_error
    )


def _total_error_admitted(
    analytic_tail: float, roundoff_error: float, tolerance: float
) -> bool:
    return not _total_error_value(analytic_tail, roundoff_error) > tolerance


def _independent_outward_product(left: float, right: float) -> float:
    if left == 0.0 or right == 0.0:
        return 0.0
    exact = Fraction.from_float(left) * Fraction.from_float(right)
    rounded = float(exact)
    if rounded == 0.0:
        return _FLOAT64_MIN_SUBNORMAL
    return _literal_outward_nonnegative(rounded)


def _independent_outward_sum(left: float, right: float) -> float:
    if left == 0.0:
        return right
    if right == 0.0:
        return left
    exact = Fraction.from_float(left) + Fraction.from_float(right)
    return _literal_outward_nonnegative(float(exact))


def _independent_outward_quotient(numerator: float, denominator: int) -> float:
    if numerator == 0.0:
        return 0.0
    exact = Decimal.from_float(numerator) / Decimal(denominator)
    rounded = float(exact)
    if rounded == 0.0:
        return _FLOAT64_MIN_SUBNORMAL
    return _literal_outward_nonnegative(rounded)


def _independent_gamma(count: int, epsilon: float) -> float:
    rounded_product = float(Decimal(count) * Decimal.from_float(epsilon))
    if rounded_product >= 1.0:
        return math.inf
    rounded_denominator = float(Decimal(1) - Decimal.from_float(rounded_product))
    exact_quotient = Decimal.from_float(rounded_product) / Decimal.from_float(
        rounded_denominator
    )
    return _literal_outward_nonnegative(float(exact_quotient))


def _independent_one_plus_gamma(count: int, epsilon: float) -> float:
    exact = Decimal(1) + Decimal.from_float(_independent_gamma(count, epsilon))
    return _literal_outward_nonnegative(float(exact))


def _independent_runtime_product(
    left: float, right: float, maximum: float
) -> float:
    if left == 0.0 or right == 0.0:
        return 0.0
    if not math.isfinite(left) or not math.isfinite(right):
        raise ValueError("nonfinite operand")
    if Fraction.from_float(left) > Fraction.from_float(maximum) / Fraction.from_float(
        right
    ):
        raise ValueError("pre-product range")
    result = _independent_outward_product(left, right)
    if not math.isfinite(result) or Decimal.from_float(result) > Decimal.from_float(
        maximum
    ):
        raise ValueError("post-product range")
    return result


def _independent_probe_bounds(
    probe_component: float, probe_count: int, runtime_dtype: np.dtype
) -> tuple[float, float]:
    maximum = float(np.finfo(runtime_dtype).max)
    component = abs(float(probe_component))
    if not math.isfinite(component) or component > math.sqrt(maximum):
        raise ValueError("probe component range")
    energy = _independent_outward_product(component, component)
    energy = _literal_outward_nonnegative(energy)
    if not math.isfinite(energy) or energy > maximum:
        raise ValueError("probe energy range")
    total_exact = Decimal.from_float(energy) * Decimal(probe_count)
    total_energy = _literal_outward_nonnegative(float(total_exact))
    if not math.isfinite(total_energy) or total_energy > maximum:
        raise ValueError("total probe energy range")
    maximum_probe_norm = _literal_outward_nonnegative(math.sqrt(energy))
    return total_energy, maximum_probe_norm


def _independent_frozen_intermediate(
    *,
    probe_component: float,
    runtime_dtype: np.dtype,
    x_bound: float,
    order: int,
    probe_count: int = 1,
    dimension: int = 1,
) -> tuple[float, float]:
    """Replay the frozen proof from represented probe bytes, without plan helpers."""
    maximum = float(np.finfo(runtime_dtype).max)
    total_energy, maximum_probe_norm = _independent_probe_bounds(
        probe_component, probe_count, runtime_dtype
    )

    epsilon = float(np.finfo(runtime_dtype).eps)
    matvec_factor = _independent_one_plus_gamma(dimension, epsilon)
    reduction_factor = _independent_one_plus_gamma(probe_count, epsilon)
    addition_factor = _independent_one_plus_gamma(order, epsilon)
    image_bound = maximum_probe_norm
    total_dot_scale = total_energy
    terms: list[float] = []
    for power in range(1, order + 1):
        image_bound = _independent_runtime_product(image_bound, x_bound, maximum)
        image_bound = _independent_runtime_product(
            image_bound, matvec_factor, maximum
        )
        total_dot_scale = _independent_runtime_product(
            total_dot_scale, x_bound, maximum
        )
        total_dot_scale = _independent_runtime_product(
            total_dot_scale, matvec_factor, maximum
        )
        reduced = _independent_runtime_product(
            total_dot_scale, matvec_factor, maximum
        )
        reduced = _independent_runtime_product(
            reduced, reduction_factor, maximum
        )
        mean = _independent_outward_quotient(reduced, probe_count)
        terms.append(_independent_outward_quotient(mean, power))
    correction_exact = sum(
        (Decimal.from_float(term) for term in terms), start=Decimal(0)
    )
    correction_bound = _literal_outward_nonnegative(float(correction_exact))
    final_bound = _independent_runtime_product(
        correction_bound, addition_factor, maximum
    )
    return correction_bound, final_bound


def _independent_runtime_sum(left: float, right: float, maximum: float) -> float:
    if not math.isfinite(left) or not math.isfinite(right):
        raise ValueError("nonfinite addend")
    if right > maximum:
        raise ValueError("right addend range")
    if Fraction.from_float(left) > Fraction.from_float(
        maximum
    ) - Fraction.from_float(right):
        raise ValueError("pre-sum range")
    result = _independent_outward_sum(left, right)
    if not math.isfinite(result) or result > maximum:
        raise ValueError("post-sum range")
    return result


def _independent_power_series(base: float, order: int) -> float:
    power = 1.0
    terms: list[float] = []
    for exponent in range(1, order + 1):
        power = _independent_outward_product(power, base)
        terms.append(_independent_outward_quotient(power, exponent))
    exact_total = sum(
        (Decimal.from_float(term) for term in terms), start=Decimal(0)
    )
    return _literal_outward_nonnegative(float(exact_total))


def _independent_trace_tail(rho: float, order: int, multiplicity: int) -> float:
    rounded_power = float(Decimal.from_float(rho) ** (order + 1))
    rounded_gap = float(Decimal(1) - Decimal.from_float(rho))
    rounded_denominator = float(Decimal(order + 1) * Decimal.from_float(rounded_gap))
    rounded_scalar = float(
        Decimal.from_float(rounded_power)
        / Decimal.from_float(rounded_denominator)
    )
    rounded_whole = float(
        Decimal(multiplicity) * Decimal.from_float(rounded_scalar)
    )
    return _literal_outward_nonnegative(rounded_whole)


def _independent_runtime_total_error(
    *,
    certified_rho: float,
    order: int,
    multiplicity: int,
    base_scale: float,
    runtime_dtype: np.dtype,
) -> tuple[float, float, float]:
    series = _independent_outward_product(
        float(multiplicity), _independent_power_series(certified_rho, order)
    )
    maximum = float(np.finfo(runtime_dtype).max)
    base_and_series = _independent_runtime_sum(base_scale, series, maximum)
    gamma = _independent_gamma(6 * order + 4, float(np.finfo(runtime_dtype).eps))
    roundoff = _independent_outward_product(gamma, base_and_series)
    analytic_tail = _independent_trace_tail(certified_rho, order, multiplicity)
    total = _literal_outward_sum(analytic_tail, roundoff)
    return analytic_tail, roundoff, total


def _rho_raw_bound_for_target(target: float, multiplicity: int) -> float:
    epsilon = float(np.finfo(np.float64).eps)
    gamma = multiplicity * epsilon / (1.0 - multiplicity * epsilon)
    candidate = target
    for _ in range(32):
        certified = _ORIGINAL_NEXTAFTER(candidate + abs(candidate) * gamma, math.inf)
        if certified == target:
            return candidate
        candidate = _ORIGINAL_NEXTAFTER(candidate, -math.inf)
    raise AssertionError(f"could not realize certified rho target {target!r}")


def _warmup_roundoff_certified(
    rho_value: float, margin: float, multiplicity: int
) -> float:
    raw_bound = float(rho_value + float(margin))
    epsilon = float(np.finfo(np.float64).eps)
    product = multiplicity * epsilon
    envelope = abs(raw_bound) * product / (1.0 - product)
    return _ORIGINAL_NEXTAFTER(float(raw_bound + envelope), math.inf)


def _warmup_roundoff_admitted(
    rho_value: float, margin: float, multiplicity: int
) -> bool:
    return (
        math.isfinite(rho_value)
        and rho_value >= 0.0
        and math.isfinite(margin)
        and margin >= 0.0
        and _warmup_roundoff_certified(rho_value, margin, multiplicity) < 1.0
    )


def _x_compact_entry_admitted(value: float) -> bool:
    try:
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            result = np.float64(1.0) / np.float64(value)
    except FloatingPointError:
        return False
    return bool(np.isfinite(result))


def _x_dense_entry_admitted(value: float) -> bool:
    try:
        with np.errstate(invalid="raise", over="raise"):
            result = np.float64(2.0) * np.float64(value)
    except FloatingPointError:
        return False
    return bool(np.isfinite(result))


_TRACE_ORDER_THRESHOLD = oracles.smallest_trace_order(0.2, 0.05, 2)
_RANK_THRESHOLD = 2
_FLOAT64_MAXIMUM = float(np.finfo(np.float64).max)
_FLOAT32_MAXIMUM = float(np.finfo(np.float32).max)
_PROBE_COMPONENT_THRESHOLD = math.sqrt(_FLOAT64_MAXIMUM)
_PRODUCT_MAXIMUM_THRESHOLD = oracles.outward_nonnegative_oracle(
    oracles.exact_product(2.0, 2.0)
)
_SUM_MAXIMUM_THRESHOLD = oracles.outward_nonnegative_oracle(oracles.exact_sum(2.0, 2.0))
_INTERMEDIATE_PROBE_COMPONENT_THRESHOLD = 255.8218795864772
_INTERMEDIATE_TOTAL_ENERGY_THRESHOLD = 65444.83407515806
_EXPECTED_ULP = math.ulp(1.0)
_EXPECTED_NONFINITE_RESOURCE_AMBIGUITY = (
    "expected/rounded nonfinite roots are resource-unreachable from the "
    "constructor-valid finite SPD inputs exercised here: overflowing the float32 "
    "cast after summing float64 log-eigenvalues needs approximately 4.8e35 "
    "maximum-magnitude diagonal entries, while overflowing the float64 expected "
    "sum needs approximately 2.5e305 entries; dynamic witnesses isolate only "
    "ulp > tolerance on the real tolerance face."
)
_TOTAL_ERROR_RHO_THRESHOLD = 0.1
_TOTAL_ERROR_BASE_SCALE_THRESHOLD = 3.0
_TOTAL_ERROR_TAIL_TOLERANCE = 0.0007407407407407409
_TOTAL_ERROR_ORDER = oracles.smallest_trace_order(
    _TOTAL_ERROR_RHO_THRESHOLD, _TOTAL_ERROR_TAIL_TOLERANCE, 2
)
_TOTAL_ERROR_TOLERANCE_THRESHOLD = _independent_runtime_total_error(
    certified_rho=_TOTAL_ERROR_RHO_THRESHOLD,
    order=_TOTAL_ERROR_ORDER,
    multiplicity=2,
    base_scale=_TOTAL_ERROR_BASE_SCALE_THRESHOLD,
    runtime_dtype=np.dtype(np.float64),
)[2]


def _total_error_order_for_rho(value: float) -> int:
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        return _TOTAL_ERROR_ORDER
    return oracles.smallest_trace_order(value, _TOTAL_ERROR_TAIL_TOLERANCE, 2)


def _total_error_rho_admitted(value: float) -> bool:
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        return False
    order = _total_error_order_for_rho(value)
    return (
        _independent_runtime_total_error(
            certified_rho=value,
            order=order,
            multiplicity=2,
            base_scale=_TOTAL_ERROR_BASE_SCALE_THRESHOLD,
            runtime_dtype=np.dtype(np.float64),
        )[2]
        <= _TOTAL_ERROR_TOLERANCE_THRESHOLD
    )


def _total_error_base_admitted(value: float) -> bool:
    if not math.isfinite(value) or value < 0.0:
        return False
    try:
        total = _independent_runtime_total_error(
            certified_rho=_TOTAL_ERROR_RHO_THRESHOLD,
            order=_TOTAL_ERROR_ORDER,
            multiplicity=2,
            base_scale=value,
            runtime_dtype=np.dtype(np.float64),
        )[2]
    except ValueError:
        return False
    return total <= _TOTAL_ERROR_TOLERANCE_THRESHOLD


def _total_error_rho_overrides(value: float) -> dict[str, Any]:
    order = _total_error_order_for_rho(value)
    valid_rho = math.isfinite(value) and 0.0 <= value < 1.0
    return {
        "measured_max": 0.0,
        "margin": value if valid_rho else 0.0,
        "certified_rho": value,
        "order": order,
        "tolerance": _TOTAL_ERROR_TOLERANCE_THRESHOLD,
        "tail_tolerance": _TOTAL_ERROR_TAIL_TOLERANCE,
        "multiplicity": 2,
        "max_abs_lambda_logdet": _TOTAL_ERROR_BASE_SCALE_THRESHOLD,
    }


def _total_error_base_overrides(value: float) -> dict[str, Any]:
    return {
        "certified_rho": _TOTAL_ERROR_RHO_THRESHOLD,
        "order": _TOTAL_ERROR_ORDER,
        "tolerance": _TOTAL_ERROR_TOLERANCE_THRESHOLD,
        "tail_tolerance": _TOTAL_ERROR_TAIL_TOLERANCE,
        "max_abs_lambda_logdet": value,
    }


def _total_error_tolerance_overrides(value: float) -> dict[str, Any]:
    return {
        "certified_rho": _TOTAL_ERROR_RHO_THRESHOLD,
        "order": _TOTAL_ERROR_ORDER,
        "tolerance": value,
        "tail_tolerance": _TOTAL_ERROR_TAIL_TOLERANCE,
        "max_abs_lambda_logdet": _TOTAL_ERROR_BASE_SCALE_THRESHOLD,
    }


def _frozen_order_certificate_overrides(order: int) -> dict[str, Any]:
    rho = 0.2
    if order == 0:
        tail = 0.75
    elif 0 < order < 512:
        rho_decimal = Decimal.from_float(rho)

        def bound(candidate_order: int) -> Decimal:
            return (
                Decimal(2)
                * rho_decimal ** (candidate_order + 1)
                / (Decimal(candidate_order + 1) * (Decimal(1) - rho_decimal))
            )

        tail = float((bound(order - 1) + bound(order)) / Decimal(2))
    else:
        tail = _FLOAT64_MIN_SUBNORMAL
    total_energy, _ = _independent_probe_bounds(
        200.0, 1, np.dtype(np.float16)
    )
    return {
        "probe_component": 200.0,
        "total_probe_energy": total_energy,
        "measured_max": 0.1,
        "margin": 0.1,
        "certified_rho": rho,
        "order": order,
        "tolerance": max(1.0, 2.0 * tail),
        "tail_tolerance": tail,
        "multiplicity": 2,
        "max_abs_lambda_logdet": 3.0,
        "max_x_operator_norm": 0.997,
    }
_PLAN_TRACE_ORDER_THRESHOLD = oracles.smallest_trace_order(0.1, 0.125, 2)
_DERIVED_ORDER_RHO_THRESHOLD = 0.2
_DERIVED_ORDER_TAIL_THRESHOLD = 0.05
_DERIVED_ORDER_MULTIPLICITY_THRESHOLD = 2
_DERIVED_ORDER_TARGET = 2
_RHO_ROUNDOFF_MULTIPLICITY = 2
_RHO_ROUNDOFF_INPUT_THRESHOLD = _rho_raw_bound_for_target(
    1.0, _RHO_ROUNDOFF_MULTIPLICITY
)
_X_LAMBDA_ENTRY_THRESHOLD = 1.0 / _FLOAT64_MAXIMUM
_X_PERTURBATION_ENTRY_THRESHOLD = _FLOAT64_MAXIMUM / 2.0
_SIGMA_LAMBDA_THRESHOLD = 1.0
_SIGMA_PERTURBATION_THRESHOLD = -1.0
_SERIES_X_BOUND_THRESHOLD = 1.0
_SERIES_PROBE_COMPONENT_THRESHOLD = 1.0
_SERIES_ORDER_THRESHOLD = 7
_SERIES_ORDER_RHO = MappingProxyType(
    {
        0: 0.1,
        6: 0.7544,
        7: 0.7868,
        8: 0.8117,
        10: 0.8474,
        16: 0.9028,
    }
)


def _compact_rank_problem(rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a compact problem whose exact algebraic rank is ``rank``."""
    dimension = max(1, int(rank))
    lambda_matrix = np.full(dimension, 2.0, dtype=np.float64)
    perturbation = np.zeros(dimension, dtype=np.float64)
    perturbation[:rank] = 0.2
    return lambda_matrix, perturbation


def _audit_rank_overrides(rank: int) -> Mapping[str, Any]:
    lambda_matrix, perturbation = _compact_rank_problem(rank)
    traces = oracles.exact_power_traces(
        lambda_matrix, perturbation, _TRACE_ORDER_THRESHOLD
    )
    # Keep the trace-evidence companion at its baseline value while the rank
    # axis sweeps; the exact-trace payload stays consistent with the varying
    # perturbation so the audit itself remains admissible up to the threshold.
    return {
        "lambda_matrix": lambda_matrix,
        "perturbation": perturbation,
        "exact_power_traces": traces,
    }


def _factory_rank_overrides(rank: int) -> Mapping[str, Any]:
    lambda_matrix, perturbation = _compact_rank_problem(rank)
    return {
        "lambda_matrix": lambda_matrix,
        "perturbation": perturbation,
    }


def _lambda_scale_matrix(scale: float) -> np.ndarray:
    if math.isfinite(scale) and scale >= 0.0:
        return np.array([math.exp(scale)], dtype=np.float64)
    if math.isnan(scale):
        return np.array([math.nan], dtype=np.float64)
    if scale == math.inf:
        return np.array([math.inf], dtype=np.float64)
    return np.array([-1.0], dtype=np.float64)


def _x_norm_perturbation(norm: float) -> np.ndarray:
    return np.array([norm], dtype=np.float64)


def _series_order_overrides(order: int) -> Mapping[str, Any]:
    return {
        "measured_max": 0.0,
        "margin": 0.0,
        "certified_rho": _SERIES_ORDER_RHO[order],
        "tail_tolerance": 0.25,
        "tolerance": 0.5,
        "multiplicity": 2,
        "order": order,
        "max_x_operator_norm": 1.0e50,
        "probe_component": 1.0,
    }


_COMPOUND_DOMAIN_AXIS_CELLS = (
    *_scalar_override_axis(
        gate_id="PLAN:certificate:rho-domain-and-coverage",
        input_key="measured_max",
        threshold=1.0,
        low=0.2,
        high=1.5,
        admitted=lambda value: math.isfinite(value) and 0.0 <= value < 1.0,
        overrides=lambda value: {
            "measured_max": value,
            "certified_rho": _ORIGINAL_NEXTAFTER(1.0, -math.inf),
        },
    ),
    *_scalar_override_axis(
        gate_id="PLAN:certificate:rho-domain-and-coverage",
        input_key="certified_rho",
        threshold=1.0,
        low=0.2,
        high=1.5,
        admitted=lambda value: math.isfinite(value) and 0.1 <= value < 1.0,
        overrides=lambda value: {"measured_max": 0.1, "certified_rho": value},
    ),
    *_scalar_override_axis(
        gate_id="PLAN:certificate:optional-scale-domain",
        input_key="max_abs_lambda_logdet",
        threshold=0.0,
        low=-1.0,
        high=8.0,
        admitted=lambda value: math.isfinite(value) and value >= 0.0,
        overrides=lambda value: {"max_abs_lambda_logdet": value},
    ),
    *_scalar_override_axis(
        gate_id="PLAN:certificate:optional-scale-domain",
        input_key="max_x_operator_norm",
        threshold=0.0,
        low=-1.0,
        high=8.0,
        admitted=lambda value: math.isfinite(value) and value >= 0.0,
        overrides=lambda value: {"max_x_operator_norm": value},
    ),
    *_scalar_override_axis(
        gate_id="PLAN:warmup:rho-inputs-and-margin",
        input_key="rho_value",
        threshold=0.0,
        low=-1.0,
        high=0.5,
        admitted=lambda value: math.isfinite(value) and value >= 0.0,
        overrides=lambda value: {"measured_rhos": (value,), "rho_value": value},
    ),
    *_scalar_override_axis(
        gate_id="PLAN:warmup:rho-inputs-and-margin",
        input_key="margin",
        threshold=0.0,
        low=-1.0,
        high=0.2,
        admitted=lambda value: math.isfinite(value) and value >= 0.0,
        overrides=lambda value: {"margin": value},
    ),
    *_scalar_override_axis(
        gate_id="PLAN:warmup:lambda-scale-inputs",
        input_key="lambda_value",
        threshold=0.0,
        low=-8.0,
        high=8.0,
        admitted=_finite,
        overrides=lambda value: {
            "lambda_logdets": (value,),
            "lambda_value": value,
        },
    ),
    *_scalar_override_axis(
        gate_id="PLAN:warmup:lambda-scale-inputs",
        input_key="lambda_logdet_margin",
        threshold=0.0,
        low=-1.0,
        high=0.2,
        admitted=lambda value: math.isfinite(value) and value >= 0.0,
        overrides=lambda value: {"lambda_logdet_margin": value},
    ),
    *_scalar_override_axis(
        gate_id="PLAN:warmup:x-norm-inputs",
        input_key="x_norm_value",
        threshold=0.0,
        low=-1.0,
        high=0.5,
        admitted=lambda value: math.isfinite(value) and value >= 0.0,
        overrides=lambda value: {
            "x_operator_norms": (value,),
            "x_norm_value": value,
        },
    ),
    *_scalar_override_axis(
        gate_id="PLAN:warmup:x-norm-inputs",
        input_key="x_operator_norm_margin",
        threshold=0.0,
        low=-1.0,
        high=0.2,
        admitted=lambda value: math.isfinite(value) and value >= 0.0,
        overrides=lambda value: {"x_operator_norm_margin": value},
    ),
    *_integer_override_axis(
        gate_id="PLAN:audit:retained-trace-evidence",
        input_key="problem_trace_order",
        threshold=_TRACE_ORDER_THRESHOLD,
        low=0,
        high=8,
        extreme=10_000,
        admitted=lambda value: value == _TRACE_ORDER_THRESHOLD,
        overrides=lambda value: {"problem_trace_order": value},
    ),
    *_integer_override_axis(
        gate_id="PLAN:audit:retained-trace-evidence",
        input_key="perturbation",
        threshold=_RANK_THRESHOLD,
        low=0,
        high=8,
        extreme=10_000,
        admitted=lambda value: value <= _RANK_THRESHOLD,
        overrides=_audit_rank_overrides,
        realized_key="retained_rank",
    ),
    *_integer_override_axis(
        gate_id="PLAN:factory-certificate:order-and-rank",
        input_key="problem_trace_order",
        threshold=_TRACE_ORDER_THRESHOLD,
        low=0,
        high=8,
        extreme=10_000,
        admitted=lambda value: value == _TRACE_ORDER_THRESHOLD,
        overrides=lambda value: {"problem_trace_order": value},
    ),
    *_integer_override_axis(
        gate_id="PLAN:frozen:probe-energy-range",
        input_key="probe_count",
        threshold=4,
        low=1,
        high=8,
        extreme=16,
        admitted=lambda value: value < 4,
        overrides=lambda value: {
            "probe_component": 1.0e19,
            "probe_count": value,
            "runtime_dtype": np.dtype(np.float32).str,
        },
    ),
    *_capability_override_axis(
        gate_id="PLAN:frozen:probe-energy-range",
        input_key="runtime_dtype",
        cells=tuple(
            (
                name,
                role,
                dtype,
                side,
                {
                    "probe_component": 1.0e10,
                    "probe_count": 4,
                    "runtime_dtype": dtype,
                },
            )
            for name, role, dtype, side in (
                (
                    "big-endian-float16",
                    PointRole.CAPABILITY_LOW,
                    ">f2",
                    GateSide.REFUSED,
                ),
                (
                    "native-float16",
                    PointRole.VALID_CAPABILITY,
                    "<f2",
                    GateSide.REFUSED,
                ),
                (
                    "big-endian-float32",
                    PointRole.INVALID_CAPABILITY,
                    ">f4",
                    GateSide.ADMITTED,
                ),
                (
                    "native-float32",
                    PointRole.CAPABILITY_HIGH,
                    "<f4",
                    GateSide.ADMITTED,
                ),
                (
                    "native-float64",
                    PointRole.EXTREME,
                    "<f8",
                    GateSide.ADMITTED,
                ),
            )
        ),
    ),
    *_scalar_override_axis(
        gate_id="PLAN:runtime-range:product",
        input_key="left",
        threshold=5.0,
        low=0.0,
        high=10.0,
        admitted=lambda value: _range_product_admitted(value, 2.0, 10.0),
        overrides=lambda value: {"left": value, "right": 2.0, "maximum": 10.0},
    ),
    *_scalar_override_axis(
        gate_id="PLAN:runtime-range:product",
        input_key="right",
        threshold=5.0,
        low=0.0,
        high=10.0,
        admitted=lambda value: _range_product_admitted(2.0, value, 10.0),
        overrides=lambda value: {"left": 2.0, "right": value, "maximum": 10.0},
    ),
    *_scalar_override_axis(
        gate_id="PLAN:runtime-range:product",
        input_key="maximum",
        threshold=_PRODUCT_MAXIMUM_THRESHOLD,
        low=1.0,
        high=10.0,
        admitted=lambda value: _range_product_admitted(2.0, 2.0, value),
        overrides=lambda value: {"left": 2.0, "right": 2.0, "maximum": value},
    ),
    *_scalar_override_axis(
        gate_id="PLAN:runtime-range:sum",
        input_key="left",
        threshold=9.5,
        low=0.0,
        high=10.0,
        admitted=lambda value: _range_sum_admitted(value, 0.5, 10.0),
        overrides=lambda value: {"left": value, "right": 0.5, "maximum": 10.0},
    ),
    *_scalar_override_axis(
        gate_id="PLAN:runtime-range:sum",
        input_key="right",
        threshold=8.0,
        low=0.0,
        high=10.0,
        admitted=lambda value: _range_sum_admitted(2.0, value, 10.0),
        overrides=lambda value: {"left": 2.0, "right": value, "maximum": 10.0},
    ),
    *_scalar_override_axis(
        gate_id="PLAN:runtime-range:sum",
        input_key="maximum",
        threshold=_SUM_MAXIMUM_THRESHOLD,
        low=1.0,
        high=10.0,
        admitted=lambda value: _range_sum_admitted(2.0, 2.0, value),
        overrides=lambda value: {"left": 2.0, "right": 2.0, "maximum": value},
    ),
    *_integer_override_axis(
        gate_id="PLAN:frozen:intermediate-runtime-range",
        input_key="order",
        threshold=3,
        low=0,
        high=8,
        extreme=16,
        admitted=lambda value: 0 <= value < 3,
        overrides=_frozen_order_certificate_overrides,
    ),
    *_finite_scalar_override_axis(
        gate_id="PLAN:runtime:expected-and-ulp-finite",
        input_key="lambda_entry",
        threshold=1.0,
        low=_FLOAT64_MIN_SUBNORMAL,
        high=_FLOAT64_MAXIMUM,
        extreme=1.0e100,
        admitted=lambda value: value > 0.0,
        overrides=lambda value: {
            "lambda_entry": value,
            "perturbation_entry": 0.0,
            "runtime_dtype": "<f8/x64-on",
            "tolerance": 1_000.0,
            "tail_tolerance": 0.25,
        },
    ),
    *_finite_scalar_override_axis(
        gate_id="PLAN:runtime:expected-and-ulp-finite",
        input_key="perturbation_entry",
        threshold=0.0,
        low=-0.5,
        high=1.0,
        extreme=_FLOAT64_MAXIMUM,
        admitted=lambda value: value > -1.0,
        overrides=lambda value: {
            "lambda_entry": 1.0,
            "perturbation_entry": value,
            "runtime_dtype": "<f8/x64-on",
            "tolerance": 1_000.0,
            "tail_tolerance": 0.25,
        },
    ),
    *_capability_override_axis(
        gate_id="PLAN:runtime:expected-and-ulp-finite",
        input_key="runtime_dtype",
        cells=tuple(
            (
                name,
                role,
                descriptor,
                GateSide.ADMITTED,
                {
                    "lambda_entry": math.e,
                    "perturbation_entry": 0.0,
                    "runtime_dtype": descriptor,
                    "tolerance": 1.0,
                    "tail_tolerance": 0.25,
                },
            )
            for name, role, descriptor in (
                ("float16-x64-off", PointRole.CAPABILITY_LOW, "<f2/x64-off"),
                ("float32-x64-off", PointRole.VALID_CAPABILITY, "<f4/x64-off"),
                ("float64-x64-off", PointRole.INVALID_CAPABILITY, "<f8/x64-off"),
                ("float32-x64-on", PointRole.CAPABILITY_HIGH, "<f4/x64-on"),
                ("float64-x64-on", PointRole.EXTREME, "<f8/x64-on"),
            )
        ),
    ),
    *_finite_scalar_override_axis(
        gate_id="PLAN:runtime:expected-and-ulp-finite",
        input_key="tolerance",
        threshold=_EXPECTED_ULP,
        low=_EXPECTED_ULP / 4.0,
        high=_EXPECTED_ULP * 4.0,
        extreme=1.0,
        admitted=lambda value: value >= _EXPECTED_ULP,
        overrides=lambda value: {
            "lambda_entry": math.e,
            "perturbation_entry": 0.0,
            "runtime_dtype": "<f8/x64-on",
            "tolerance": value,
            "tail_tolerance": value / 2.0,
        },
    ),
    *_finite_scalar_override_axis(
        gate_id="PLAN:runtime:total-error-budget",
        input_key="certified_rho",
        threshold=_TOTAL_ERROR_RHO_THRESHOLD,
        low=0.05,
        high=0.5,
        extreme=0.9,
        admitted=_total_error_rho_admitted,
        overrides=_total_error_rho_overrides,
    ),
    *_finite_scalar_override_axis(
        gate_id="PLAN:runtime:total-error-budget",
        input_key="max_abs_lambda_logdet",
        threshold=_TOTAL_ERROR_BASE_SCALE_THRESHOLD,
        low=0.0,
        high=10.0,
        extreme=1.0e300,
        admitted=_total_error_base_admitted,
        overrides=_total_error_base_overrides,
    ),
    *_finite_scalar_override_axis(
        gate_id="PLAN:runtime:total-error-budget",
        input_key="tolerance",
        threshold=_TOTAL_ERROR_TOLERANCE_THRESHOLD,
        low=_ORIGINAL_NEXTAFTER(_TOTAL_ERROR_TAIL_TOLERANCE, math.inf),
        high=_TOTAL_ERROR_TOLERANCE_THRESHOLD * 4.0,
        extreme=1.0,
        admitted=lambda value: not math.isnan(value)
        and value >= _TOTAL_ERROR_TOLERANCE_THRESHOLD,
        overrides=_total_error_tolerance_overrides,
    ),
    *_integer_override_axis(
        gate_id="PLAN:trace-factory:exact-evidence",
        input_key="problem_trace_order",
        threshold=_PLAN_TRACE_ORDER_THRESHOLD,
        low=-1,
        high=8,
        extreme=10_000,
        admitted=lambda value: value == _PLAN_TRACE_ORDER_THRESHOLD,
        overrides=lambda value: {"problem_trace_order": value},
    ),
    *_integer_override_axis(
        gate_id="PLAN:frozen-factory:probe-presence-width",
        input_key="probe_presence",
        threshold=1,
        low=-2,
        high=8,
        extreme=-1,
        admitted=lambda value: value > 0,
        overrides=lambda value: {
            "probe_presence": value,
            "probe_width": 3,
            "frozen_probe_values": (None if value <= 0 else np.full((value, 3), 0.25)),
        },
    ),
    *_finite_scalar_override_axis(
        gate_id="PLAN:certificate:order-is-derived",
        input_key="certified_rho",
        threshold=_DERIVED_ORDER_RHO_THRESHOLD,
        low=0.1,
        high=0.8,
        extreme=0.99,
        admitted=lambda value: (
            oracles.smallest_trace_order(value, _DERIVED_ORDER_TAIL_THRESHOLD, 2)
            == _DERIVED_ORDER_TARGET
        ),
        overrides=lambda value: {
            "measured_max": 0.0,
            "margin": 0.0,
            "certified_rho": value,
            "tail_tolerance": _DERIVED_ORDER_TAIL_THRESHOLD,
            "tolerance": 0.5,
            "multiplicity": 2,
            "order": _DERIVED_ORDER_TARGET,
        },
    ),
    *_finite_scalar_override_axis(
        gate_id="PLAN:certificate:order-is-derived",
        input_key="tail_tolerance",
        threshold=_DERIVED_ORDER_TAIL_THRESHOLD,
        low=0.01,
        high=0.2,
        extreme=0.49,
        admitted=lambda value: (
            oracles.smallest_trace_order(0.2, value, 2) == _DERIVED_ORDER_TARGET
        ),
        overrides=lambda value: {
            "measured_max": 0.0,
            "margin": 0.0,
            "certified_rho": 0.2,
            "tail_tolerance": value,
            "tolerance": 0.5,
            "multiplicity": 2,
            "order": _DERIVED_ORDER_TARGET,
        },
    ),
    *_integer_override_axis(
        gate_id="PLAN:certificate:order-is-derived",
        input_key="multiplicity",
        threshold=3,
        low=1,
        high=6,
        extreme=100,
        admitted=lambda value: (
            oracles.smallest_trace_order(0.2, 0.0625, value) == _DERIVED_ORDER_TARGET
        ),
        overrides=lambda value: {
            "measured_max": 0.0,
            "margin": 0.0,
            "certified_rho": 0.2,
            "tail_tolerance": 0.0625,
            "tolerance": 0.5,
            "multiplicity": value,
            "order": _DERIVED_ORDER_TARGET,
        },
    ),
    *_integer_override_axis(
        gate_id="PLAN:certificate:order-is-derived",
        input_key="order",
        threshold=_DERIVED_ORDER_TARGET,
        low=-2,
        high=8,
        extreme=-10,
        admitted=lambda value: value == _DERIVED_ORDER_TARGET,
        overrides=lambda value: {
            "measured_max": 0.0,
            "margin": 0.0,
            "certified_rho": _DERIVED_ORDER_RHO_THRESHOLD,
            "tail_tolerance": _DERIVED_ORDER_TAIL_THRESHOLD,
            "tolerance": 0.5,
            "multiplicity": _DERIVED_ORDER_MULTIPLICITY_THRESHOLD,
            "order": value,
        },
    ),
    *_scalar_override_axis(
        gate_id="PLAN:audit:retained-rho",
        input_key="retained_value",
        threshold=0.5,
        low=-0.1,
        high=0.8,
        admitted=lambda value: math.isfinite(value) and 0.0 <= value <= 0.5,
        overrides=lambda value: {"retained_value": value, "retained_values": (value,)},
    ),
    *_finite_scalar_override_axis(
        gate_id="PLAN:audit:retained-rho",
        input_key="certified_rho",
        threshold=0.25,
        low=0.1,
        high=0.8,
        extreme=0.9,
        admitted=lambda value: value >= 0.25,
        overrides=lambda value: {
            "measured_max": 0.0,
            "certified_rho": value,
            "order": oracles.smallest_trace_order(value, 0.25, 2),
            "retained_value": 0.25,
            "retained_values": (0.25,),
        },
    ),
    *_scalar_override_axis(
        gate_id="PLAN:audit:retained-lambda-scale",
        input_key="retained_value",
        threshold=2.0,
        low=-4.0,
        high=4.0,
        admitted=lambda value: math.isfinite(value) and abs(value) <= 2.0,
        overrides=lambda value: {"retained_value": value, "retained_values": (value,)},
    ),
    *_finite_scalar_override_axis(
        gate_id="PLAN:audit:retained-lambda-scale",
        input_key="max_abs_lambda_logdet",
        threshold=1.0,
        low=0.0,
        high=4.0,
        extreme=10.0,
        admitted=lambda value: value >= 1.0,
        overrides=lambda value: {
            "max_abs_lambda_logdet": value,
            "retained_value": 1.0,
            "retained_values": (1.0,),
        },
    ),
    *_scalar_override_axis(
        gate_id="PLAN:audit:retained-x-norm",
        input_key="retained_value",
        threshold=0.5,
        low=-0.1,
        high=0.8,
        admitted=lambda value: math.isfinite(value) and 0.0 <= value <= 0.5,
        overrides=lambda value: {"retained_value": value, "retained_values": (value,)},
    ),
    *_finite_scalar_override_axis(
        gate_id="PLAN:audit:retained-x-norm",
        input_key="max_x_operator_norm",
        threshold=0.25,
        low=0.0,
        high=0.8,
        extreme=10.0,
        admitted=lambda value: value >= 0.25,
        overrides=lambda value: {
            "max_x_operator_norm": value,
            "retained_value": 0.25,
            "retained_values": (0.25,),
        },
    ),
    *_finite_scalar_override_axis(
        gate_id="PLAN:factory-certificate:lambda-scale",
        input_key="lambda_matrix",
        threshold=2.0,
        low=0.0,
        high=4.0,
        extreme=10.0,
        admitted=lambda value: math.isfinite(value) and 0.0 <= value <= 2.0,
        overrides=lambda value: {
            "lambda_matrix": _lambda_scale_matrix(value),
            "perturbation": np.array([0.0]),
        },
        realized_key="actual_base_scale",
    ),
    *_finite_scalar_override_axis(
        gate_id="PLAN:factory-certificate:lambda-scale",
        input_key="max_abs_lambda_logdet",
        threshold=1.0,
        low=0.0,
        high=4.0,
        extreme=10.0,
        admitted=lambda value: value >= 1.0,
        overrides=lambda value: {
            "max_abs_lambda_logdet": value,
            "lambda_matrix": _lambda_scale_matrix(1.0),
            "perturbation": np.array([0.0]),
        },
    ),
    *_finite_scalar_override_axis(
        gate_id="PLAN:factory-certificate:x-norm",
        input_key="perturbation",
        threshold=0.5,
        low=0.0,
        high=0.8,
        extreme=0.95,
        admitted=lambda value: math.isfinite(value) and abs(value) <= 0.5,
        overrides=lambda value: {
            "lambda_matrix": np.array([1.0]),
            "perturbation": _x_norm_perturbation(value),
        },
        realized_key="actual_x_norm",
    ),
    *_finite_scalar_override_axis(
        gate_id="PLAN:factory-certificate:x-norm",
        input_key="max_x_operator_norm",
        threshold=0.25,
        low=0.0,
        high=0.8,
        extreme=10.0,
        admitted=lambda value: value >= 0.25,
        overrides=lambda value: {
            "max_x_operator_norm": value,
            "lambda_matrix": np.array([1.0]),
            "perturbation": _x_norm_perturbation(0.25),
        },
    ),
    *_finite_scalar_override_axis(
        gate_id="PLAN:warmup:rho-roundoff-ceiling",
        input_key="rho_value",
        threshold=_RHO_ROUNDOFF_INPUT_THRESHOLD,
        low=0.25,
        high=1.5,
        extreme=_FLOAT64_MAXIMUM,
        admitted=lambda value: _warmup_roundoff_admitted(
            value, 0.0, _RHO_ROUNDOFF_MULTIPLICITY
        ),
        overrides=lambda value: {
            "rho_value": value,
            "measured_rhos": (value,),
            "margin": 0.0,
            "multiplicity": _RHO_ROUNDOFF_MULTIPLICITY,
        },
    ),
    *_finite_scalar_override_axis(
        gate_id="PLAN:warmup:rho-roundoff-ceiling",
        input_key="margin",
        threshold=_RHO_ROUNDOFF_INPUT_THRESHOLD,
        low=0.25,
        high=1.5,
        extreme=_FLOAT64_MAXIMUM,
        admitted=lambda value: _warmup_roundoff_admitted(
            0.0, value, _RHO_ROUNDOFF_MULTIPLICITY
        ),
        overrides=lambda value: {
            "rho_value": 0.0,
            "measured_rhos": (0.0,),
            "margin": value,
            "multiplicity": _RHO_ROUNDOFF_MULTIPLICITY,
        },
    ),
    *_integer_override_axis(
        gate_id="PLAN:warmup:rho-roundoff-ceiling",
        input_key="multiplicity",
        threshold=3,
        low=1,
        high=8,
        extreme=10_000,
        admitted=lambda value: _warmup_roundoff_admitted(
            0.0, _rho_raw_bound_for_target(1.0, 3), value
        ),
        overrides=lambda value: {
            "rho_value": 0.0,
            "measured_rhos": (0.0,),
            "margin": _rho_raw_bound_for_target(1.0, 3),
            "multiplicity": value,
        },
    ),
    *_finite_scalar_override_axis(
        gate_id="PLAN:gamma:operation-count-domain",
        input_key="epsilon",
        threshold=0.25,
        low=float(np.finfo(np.float64).eps),
        high=0.5,
        extreme=float(np.finfo(np.float32).eps),
        admitted=lambda value: 4 * value < 1.0,
        overrides=lambda value: {"operation_count": 4, "epsilon": value},
    ),
    *_integer_override_axis(
        gate_id="PLAN:gamma:operation-count-domain",
        input_key="operation_count",
        threshold=4,
        low=0,
        high=8,
        extreme=-1,
        admitted=lambda value: value * 0.25 < 1.0,
        overrides=lambda value: {"operation_count": value, "epsilon": 0.25},
    ),
    *_capability_override_axis(
        gate_id="PLAN:canonical-probes:runtime-finite",
        input_key="probe_dtype",
        cells=tuple(
            (
                name,
                role,
                dtype,
                GateSide.ADMITTED,
                {
                    "probe_scalar": 1.7,
                    "probe_dtype": dtype,
                    "runtime_dtype": np.dtype(np.float32).str,
                },
            )
            for name, role, dtype in (
                ("big-endian-float16", PointRole.CAPABILITY_LOW, ">f2"),
                ("native-float16", PointRole.VALID_CAPABILITY, "<f2"),
                ("big-endian-float32", PointRole.INVALID_CAPABILITY, ">f4"),
                ("native-float32", PointRole.CAPABILITY_HIGH, "<f4"),
                ("native-float64", PointRole.EXTREME, "<f8"),
            )
        ),
    ),
    *_capability_override_axis(
        gate_id="PLAN:canonical-probes:runtime-finite",
        input_key="runtime_dtype",
        cells=tuple(
            (
                name,
                role,
                dtype,
                GateSide.ADMITTED,
                {
                    "probe_scalar": 1.7,
                    "probe_dtype": np.dtype(np.float64).str,
                    "runtime_dtype": dtype,
                },
            )
            for name, role, dtype in (
                ("big-endian-float16", PointRole.CAPABILITY_LOW, ">f2"),
                ("native-float16", PointRole.VALID_CAPABILITY, "<f2"),
                ("big-endian-float32", PointRole.INVALID_CAPABILITY, ">f4"),
                ("native-float32", PointRole.CAPABILITY_HIGH, "<f4"),
                ("native-float64", PointRole.EXTREME, "<f8"),
            )
        ),
    ),
    *_capability_override_axis(
        gate_id="PLAN:frozen:x-bound-runtime-range",
        input_key="runtime_dtype",
        cells=tuple(
            (
                name,
                role,
                dtype,
                GateSide.ADMITTED,
                {
                    "max_x_operator_norm": 0.2,
                    "runtime_dtype": dtype,
                },
            )
            for name, role, dtype in (
                ("big-endian-float16", PointRole.CAPABILITY_LOW, ">f2"),
                ("native-float16", PointRole.VALID_CAPABILITY, "<f2"),
                ("big-endian-float32", PointRole.INVALID_CAPABILITY, ">f4"),
                ("native-float32", PointRole.CAPABILITY_HIGH, "<f4"),
                ("native-float64", PointRole.EXTREME, "<f8"),
            )
        ),
    ),
    *_capability_override_axis(
        gate_id="PLAN:runtime:base-scale-range",
        input_key="runtime_dtype",
        cells=tuple(
            (
                name,
                role,
                descriptor,
                GateSide.ADMITTED,
                {
                    "max_abs_lambda_logdet": 1.7,
                    "runtime_dtype": descriptor,
                },
            )
            for name, role, descriptor in (
                ("float16-x64-off", PointRole.CAPABILITY_LOW, "<f2/x64-off"),
                ("float32-x64-off", PointRole.VALID_CAPABILITY, "<f4/x64-off"),
                ("float64-x64-off", PointRole.INVALID_CAPABILITY, "<f8/x64-off"),
                ("float32-x64-on", PointRole.CAPABILITY_HIGH, "<f4/x64-on"),
                ("float64-x64-on", PointRole.EXTREME, "<f8/x64-on"),
            )
        ),
    ),
    *_finite_scalar_override_axis(
        gate_id="PLAN:runtime:sigma-finite-and-positive",
        input_key="lambda_entry",
        threshold=_SIGMA_LAMBDA_THRESHOLD,
        low=0.5,
        high=2.0,
        extreme=_FLOAT64_MAXIMUM,
        admitted=lambda value: value > _SIGMA_LAMBDA_THRESHOLD,
        overrides=lambda value: {
            "lambda_entry": value,
            "perturbation_entry": _SIGMA_PERTURBATION_THRESHOLD,
            "matrix_path": "compact-float64",
        },
    ),
    *_finite_scalar_override_axis(
        gate_id="PLAN:runtime:sigma-finite-and-positive",
        input_key="perturbation_entry",
        threshold=_SIGMA_PERTURBATION_THRESHOLD,
        low=-2.0,
        high=1.0,
        extreme=_FLOAT64_MAXIMUM,
        admitted=lambda value: (
            math.isfinite(value) and value > _SIGMA_PERTURBATION_THRESHOLD
        ),
        overrides=lambda value: {
            "lambda_entry": _SIGMA_LAMBDA_THRESHOLD,
            "perturbation_entry": value,
            "matrix_path": "compact-float64",
        },
    ),
    *_capability_override_axis(
        gate_id="PLAN:runtime:sigma-finite-and-positive",
        input_key="matrix_path",
        cells=(
            (
                "compact-float32",
                PointRole.CAPABILITY_LOW,
                "compact-float32",
                GateSide.ADMITTED,
                {
                    "lambda_entry": 2.0,
                    "perturbation_entry": -0.5,
                    "matrix_path": "compact-float32",
                },
            ),
            (
                "compact-float64",
                PointRole.VALID_CAPABILITY,
                "compact-float64",
                GateSide.ADMITTED,
                {
                    "lambda_entry": 2.0,
                    "perturbation_entry": -0.5,
                    "matrix_path": "compact-float64",
                },
            ),
            (
                "compact-singular",
                PointRole.INVALID_CAPABILITY,
                "compact-singular",
                GateSide.REFUSED,
                {
                    "lambda_entry": 2.0,
                    "perturbation_entry": -0.5,
                    "matrix_path": "compact-singular",
                },
            ),
            (
                "dense-float32",
                PointRole.CAPABILITY_HIGH,
                "dense-float32",
                GateSide.ADMITTED,
                {
                    "lambda_entry": 2.0,
                    "perturbation_entry": -0.5,
                    "matrix_path": "dense-float32",
                },
            ),
            (
                "dense-indefinite-float64",
                PointRole.EXTREME,
                "dense-indefinite-float64",
                GateSide.REFUSED,
                {
                    "lambda_entry": 2.0,
                    "perturbation_entry": -0.5,
                    "matrix_path": "dense-indefinite-float64",
                },
            ),
        ),
    ),
    *_finite_scalar_override_axis(
        gate_id="PLAN:runtime:frozen-prerequisites-and-series",
        input_key="max_x_operator_norm",
        threshold=_SERIES_X_BOUND_THRESHOLD,
        low=0.0,
        high=1.0e100,
        extreme=_FLOAT64_MAXIMUM,
        admitted=lambda value: value < _FLOAT64_MAXIMUM,
        overrides=lambda value: {
            "measured_max": 0.0,
            "margin": 0.0,
            "certified_rho": 0.4,
            "tail_tolerance": 0.25,
            "tolerance": 0.5,
            "multiplicity": 2,
            "order": 2,
            "max_x_operator_norm": value,
            "probe_component": 1.0,
        },
    ),
    _AxisOverrideCell(
        "PLAN:runtime:frozen-prerequisites-and-series",
        "max_x_operator_norm",
        _SERIES_X_BOUND_THRESHOLD,
        ThresholdPoint(
            PointRole.EXTREME,
            "parameter-axis:PLAN:runtime:frozen-prerequisites-and-series:"
            "max_x_operator_norm:minimum-subnormal",
            "constructor-valid minimum positive subnormal",
            GateSide.ADMITTED,
        ),
        {
            "measured_max": 0.0,
            "margin": 0.0,
            "certified_rho": 0.4,
            "tail_tolerance": 0.25,
            "tolerance": 0.5,
            "multiplicity": 2,
            "order": 2,
            "max_x_operator_norm": _FLOAT64_MIN_SUBNORMAL,
            "probe_component": 1.0,
        },
    ),
    *_finite_scalar_override_axis(
        gate_id="PLAN:runtime:frozen-prerequisites-and-series",
        input_key="probe_component",
        threshold=_SERIES_PROBE_COMPONENT_THRESHOLD,
        low=0.0,
        high=1.0e100,
        extreme=1.0e154,
        admitted=lambda value: value < 1.0e154,
        overrides=lambda value: {
            "measured_max": 0.0,
            "margin": 0.0,
            "certified_rho": 0.2,
            "tail_tolerance": 0.25,
            "tolerance": 0.5,
            "multiplicity": 2,
            "order": 1,
            "max_x_operator_norm": 2.0,
            "probe_component": value,
        },
    ),
    _AxisOverrideCell(
        "PLAN:runtime:frozen-prerequisites-and-series",
        "probe_component",
        _SERIES_PROBE_COMPONENT_THRESHOLD,
        ThresholdPoint(
            PointRole.EXTREME,
            "parameter-axis:PLAN:runtime:frozen-prerequisites-and-series:"
            "probe_component:minimum-subnormal",
            "constructor-valid minimum positive subnormal",
            GateSide.ADMITTED,
        ),
        {
            "measured_max": 0.0,
            "margin": 0.0,
            "certified_rho": 0.2,
            "tail_tolerance": 0.25,
            "tolerance": 0.5,
            "multiplicity": 2,
            "order": 1,
            "max_x_operator_norm": 2.0,
            "probe_component": _FLOAT64_MIN_SUBNORMAL,
        },
    ),
    *_integer_override_axis(
        gate_id="PLAN:runtime:frozen-prerequisites-and-series",
        input_key="order",
        threshold=_SERIES_ORDER_THRESHOLD,
        low=0,
        high=10,
        extreme=16,
        admitted=lambda value: value < _SERIES_ORDER_THRESHOLD,
        overrides=_series_order_overrides,
    ),
    *_scalar_override_axis(
        gate_id="PLAN:outward-arithmetic:positive-underflow",
        input_key="proof_value",
        threshold=0.0,
        low=-0.5,
        high=8.0,
        admitted=lambda value: value == 0.0 or not math.isfinite(value),
        overrides=lambda value: {
            "proof_value": value,
            "proof_magnitude": abs(value) if math.isfinite(value) else value,
        },
    ),
    *_scalar_override_axis(
        gate_id="PLAN:measurement:x-norm-finite",
        input_key="lambda_entry",
        threshold=_X_LAMBDA_ENTRY_THRESHOLD,
        low=0.0,
        high=1.0,
        admitted=_x_compact_entry_admitted,
        overrides=lambda value: {
            "lambda_entry": value,
            "perturbation_entry": 1.0,
            "matrix_path": "compact-float64",
        },
    ),
    *_scalar_override_axis(
        gate_id="PLAN:measurement:x-norm-finite",
        input_key="perturbation_entry",
        threshold=_X_PERTURBATION_ENTRY_THRESHOLD,
        low=0.0,
        high=_FLOAT64_MAXIMUM,
        admitted=_x_dense_entry_admitted,
        overrides=lambda value: {
            "lambda_entry": 1.0,
            "perturbation_entry": value,
            "matrix_path": "dense-full-float64",
        },
    ),
    *_capability_override_axis(
        gate_id="PLAN:measurement:x-norm-finite",
        input_key="matrix_path",
        cells=(
            (
                "compact-float32",
                PointRole.CAPABILITY_LOW,
                "compact-float32",
                GateSide.ADMITTED,
                {
                    "lambda_entry": 1.0,
                    "perturbation_entry": 0.25,
                    "matrix_path": "compact-float32",
                },
            ),
            (
                "compact-float64",
                PointRole.VALID_CAPABILITY,
                "compact-float64",
                GateSide.ADMITTED,
                {
                    "lambda_entry": 1.0,
                    "perturbation_entry": 0.25,
                    "matrix_path": "compact-float64",
                },
            ),
            (
                "dense-singular-float64",
                PointRole.INVALID_CAPABILITY,
                "dense-singular-float64",
                GateSide.REFUSED,
                {
                    "lambda_entry": 1.0,
                    "perturbation_entry": 0.25,
                    "matrix_path": "dense-singular-float64",
                },
            ),
            (
                "dense-diagonal-float32",
                PointRole.CAPABILITY_HIGH,
                "dense-diagonal-float32",
                GateSide.ADMITTED,
                {
                    "lambda_entry": 1.0,
                    "perturbation_entry": 0.25,
                    "matrix_path": "dense-diagonal-float32",
                },
            ),
            (
                "dense-full-float64",
                PointRole.EXTREME,
                "dense-full-float64",
                GateSide.ADMITTED,
                {
                    "lambda_entry": 1.0,
                    "perturbation_entry": 0.25,
                    "matrix_path": "dense-full-float64",
                },
            ),
        ),
    ),
    *_scalar_override_axis(
        gate_id="PLAN:measurement:lambda-logdet-finite",
        input_key="lambda_entry",
        threshold=0.0,
        low=-1.0,
        high=_FLOAT64_MAXIMUM,
        admitted=lambda value: math.isfinite(value) and value > 0.0,
        overrides=lambda value: {
            "lambda_entry": value,
            "matrix_path": "compact-float64",
        },
    ),
    *_capability_override_axis(
        gate_id="PLAN:measurement:lambda-logdet-finite",
        input_key="matrix_path",
        cells=(
            (
                "compact-float32",
                PointRole.CAPABILITY_LOW,
                "compact-float32",
                GateSide.ADMITTED,
                {"lambda_entry": 2.0, "matrix_path": "compact-float32"},
            ),
            (
                "compact-float64",
                PointRole.VALID_CAPABILITY,
                "compact-float64",
                GateSide.ADMITTED,
                {"lambda_entry": 2.0, "matrix_path": "compact-float64"},
            ),
            (
                "dense-indefinite-float64",
                PointRole.INVALID_CAPABILITY,
                "dense-indefinite-float64",
                GateSide.REFUSED,
                {
                    "lambda_entry": 2.0,
                    "matrix_path": "dense-indefinite-float64",
                },
            ),
            (
                "dense-float32",
                PointRole.CAPABILITY_HIGH,
                "dense-float32",
                GateSide.ADMITTED,
                {"lambda_entry": 2.0, "matrix_path": "dense-float32"},
            ),
            (
                "dense-subnormal-float64",
                PointRole.EXTREME,
                "dense-subnormal-float64",
                GateSide.ADMITTED,
                {
                    "lambda_entry": 2.0,
                    "matrix_path": "dense-subnormal-float64",
                },
            ),
        ),
    ),
)
_COMPOUND_DOMAIN_AXIS_CELLS_BY_GATE = MappingProxyType(
    {
        gate_id: tuple(
            cell for cell in _COMPOUND_DOMAIN_AXIS_CELLS if cell.gate_id == gate_id
        )
        for gate_id in {cell.gate_id for cell in _COMPOUND_DOMAIN_AXIS_CELLS}
    }
)
_COMPOUND_DOMAIN_AXIS_CELL_BY_DISPLAY = MappingProxyType(
    {cell.point.display_value: cell for cell in _COMPOUND_DOMAIN_AXIS_CELLS}
)


def _qualname_owner(qualname: str) -> tuple[Any, str]:
    parts = qualname.removeprefix("<module>.").split(".")
    owner: Any = plan
    for part in parts[:-1]:
        owner = getattr(owner, part)
    return owner, parts[-1]


def _qualname_from_atom(atom_id: str) -> str:
    return atom_id.split("::", 2)[1]


_ATOM_QUALNAMES = {
    _qualname_from_atom(atom_id)
    for entry in _ENTRIES.values()
    for atom_id in entry.conjunction_atom_ids
}
_BASELINE_CALLABLES = MappingProxyType(
    {qualname: getattr(*_qualname_owner(qualname)) for qualname in _ATOM_QUALNAMES}
)


def _ast_key(node: ast.AST) -> tuple[type[ast.AST], int, int, int, int]:
    return (
        type(node),
        int(getattr(node, "lineno", -1)),
        int(getattr(node, "col_offset", -1)),
        int(getattr(node, "end_lineno", -1)),
        int(getattr(node, "end_col_offset", -1)),
    )


def _find_callable_ast(tree: ast.Module, qualname: str) -> ast.FunctionDef:
    parts = qualname.removeprefix("<module>.").split(".")
    body: list[ast.stmt] = tree.body
    for part in parts[:-1]:
        owner = next(
            node
            for node in body
            if isinstance(node, ast.ClassDef) and node.name == part
        )
        body = owner.body
    result = next(
        node
        for node in body
        if isinstance(node, ast.FunctionDef) and node.name == parts[-1]
    )
    result.decorator_list = []
    return result


class _AtomProbeTransformer(ast.NodeTransformer):
    def __init__(self, atom_by_node: Mapping[tuple[Any, ...], str]) -> None:
        self._atom_by_node = atom_by_node

    def generic_visit(self, node: ast.AST) -> ast.AST:
        transformed = super().generic_visit(node)
        atom_id = self._atom_by_node.get(_ast_key(node))
        if atom_id is None:
            return transformed
        if not isinstance(transformed, ast.expr):
            raise TypeError(f"registered PLAN atom is not an expression: {atom_id}")
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=_ATOM_PROBE_NAME, ctx=ast.Load()),
                args=[ast.Constant(atom_id), transformed],
                keywords=[],
            ),
            node,
        )


@cache
def _instrumented_callable(
    gate_id: str,
) -> tuple[Any, str, Callable[..., Any], Any]:
    entry = _ENTRIES[gate_id]
    canonical_ids = tuple(
        atom_id
        for atom_id in entry.conjunction_atom_ids
        if source_alias_canonical(entry, atom_id) == atom_id
    )
    qualnames = {_qualname_from_atom(atom_id) for atom_id in canonical_ids}
    if len(qualnames) != 1:
        raise AssertionError(f"{gate_id} atoms do not share one owning callable")
    qualname = next(iter(qualnames))
    atom_by_node = {
        _ast_key(_PLAN_SOURCE_INDEX[atom_id][1]): atom_id for atom_id in canonical_ids
    }
    tree = ast.parse(_PLAN_SOURCE, filename=str(_PLAN_SOURCE_PATH))
    function = _find_callable_ast(tree, qualname)
    transformed = _AtomProbeTransformer(atom_by_node).visit(function)
    ast.fix_missing_locations(transformed)
    namespace: dict[str, Any] = {}
    code = compile(
        ast.Module(body=[transformed], type_ignores=[]),
        str(_PLAN_SOURCE_PATH),
        "exec",
    )
    exec(code, plan.__dict__, namespace)  # noqa: S102 - reviewed local source only
    owner, attribute = _qualname_owner(qualname)
    return (
        owner,
        attribute,
        namespace[function.name],
        _BASELINE_CALLABLES[qualname],
    )


def _snapshot_atom_raw(raw: Any) -> Any:
    if isinstance(raw, np.ndarray):
        return raw.copy()
    if hasattr(raw, "shape") and not np.isscalar(raw):
        return np.asarray(raw).copy()
    return raw


def _record_plan_atom(atom_id: str, raw: Any) -> Any:
    if not _ACTIVE_ATOM_RECORDERS:
        raise AssertionError("PLAN atom probe executed without an active recorder")
    _ACTIVE_ATOM_RECORDERS[-1].setdefault(atom_id, []).append(_snapshot_atom_raw(raw))
    return raw


@contextmanager
def _capture_production_atoms(entry: GateEntry, *, enabled: bool) -> Any:
    recorder: dict[str, list[Any]] = {}
    if not enabled or not entry.conjunction_atom_ids:
        yield recorder, False
        return
    _owner, _attribute, instrumented, baseline_callable = _instrumented_callable(
        entry.gate_id
    )
    baseline_code = baseline_callable.__code__
    _ACTIVE_ATOM_RECORDERS.append(recorder)
    try:
        # Mutate the stable production callable object in place.  A transparent
        # wrapper may retain that object in a closure even while replacing the
        # module/class attribute; installing the instrumented code on the
        # object therefore composes with the wrapper instead of bypassing it or
        # switching to hand-authored atom evidence based on callable identity.
        baseline_callable.__code__ = instrumented.__code__
        with (
            patch.object(plan, _ATOM_PROBE_NAME, _record_plan_atom, create=True),
        ):
            yield recorder, True
    finally:
        baseline_callable.__code__ = baseline_code
        popped = _ACTIVE_ATOM_RECORDERS.pop()
        if popped is not recorder:
            raise AssertionError("PLAN atom recorder stack lost LIFO identity")


def _traced_wrapper(
    target: Callable[..., Any], label: str, calls: list[str]
) -> Callable[..., Any]:
    def traced(*args: Any, **kwargs: Any) -> Any:
        calls.append(label)
        return target(*args, **kwargs)

    return traced


@contextmanager
def _trace_required_calls(entry: GateEntry) -> Any:
    calls: list[str] = []
    with ExitStack() as stack:
        for label in sorted(REQUIRED_DIRECT_CALLS[entry.gate_id]):
            namespace, attribute = label.split(".", 1)
            if namespace != "plan" or "." in attribute:
                raise AssertionError(f"unsupported PLAN direct-call label: {label}")
            live = getattr(plan, attribute)
            stack.enter_context(
                patch.object(plan, attribute, _traced_wrapper(live, label, calls))
            )
        yield calls


def _truth_from_raw(raw: Any, reducer: AtomReducer) -> bool:
    values = np.asarray(raw)
    if reducer is AtomReducer.SCALAR:
        if values.shape != ():
            raise AssertionError("scalar PLAN atom produced an array")
        return bool(values.item())
    if reducer is AtomReducer.ALL_ELEMENTS:
        return bool(np.all(values))
    if reducer is AtomReducer.ANY_ELEMENT:
        return bool(np.any(values))
    raise AssertionError("an evaluated PLAN atom cannot use NOT_EVALUATED")


def _captured_atom_row(
    entry: GateEntry,
    outcome: _Outcome,
    recorder: Mapping[str, list[Any]],
) -> Mapping[str, _AtomValue]:
    values: list[_AtomValue] = []
    for atom_id in entry.conjunction_atom_ids:
        canonical = source_alias_canonical(entry, atom_id)
        captured = recorder.get(canonical, ())
        if not captured:
            values.append(
                _production_atom(
                    None,
                    None,
                    AtomReducer.NOT_EVALUATED,
                    source=f"instrumented AST {canonical} not evaluated",
                )
            )
            continue
        raw = captured[0]
        reducer = outcome.actual_atoms[atom_id].reducer
        if reducer is AtomReducer.NOT_EVALUATED:
            raise AssertionError(f"{atom_id} executed despite manual skip metadata")
        values.append(
            _production_atom(
                raw,
                _truth_from_raw(raw, reducer),
                reducer,
                source=f"instrumented live AST {canonical}",
            )
        )
    return _atom_row(entry, tuple(values))


def _atom_specs(entry: GateEntry) -> tuple[_AtomSpec, ...]:
    return tuple(
        _AtomSpec(
            atom_id,
            entry.conjunction_atom_ids.index(atom_id),
            _SYNTAX[atom_id],
        )
        for atom_id in isolatable_atom_ids(entry)
    )


def _atom_row(
    entry: GateEntry,
    values: tuple[_AtomValue, ...],
) -> Mapping[str, _AtomValue]:
    if len(values) != len(entry.conjunction_atom_ids):
        raise AssertionError(
            f"{entry.gate_id} produced {len(values)} atom values for "
            f"{len(entry.conjunction_atom_ids)} registered atoms"
        )
    return MappingProxyType(dict(zip(entry.conjunction_atom_ids, values, strict=True)))


def _production_atom(
    raw: Any,
    truth: bool | None,
    reducer: AtomReducer = AtomReducer.SCALAR,
    *,
    source: str,
) -> _AtomValue:
    return _AtomValue(raw, truth, reducer, f"production:{source}")


def _oracle_atom(
    raw: Any,
    truth: bool | None,
    reducer: AtomReducer = AtomReducer.SCALAR,
    *,
    source: str,
) -> _AtomValue:
    return _AtomValue(raw, truth, reducer, f"oracle:{source}")


def _production_aliases(
    raw: Any,
    truth: bool | None,
    reducer: AtomReducer = AtomReducer.SCALAR,
    *,
    source: str,
) -> tuple[_AtomValue, _AtomValue]:
    return (
        _production_atom(raw, truth, reducer, source=source),
        _production_atom(raw, truth, reducer, source=source),
    )


def _oracle_aliases(
    raw: Any,
    truth: bool | None,
    reducer: AtomReducer = AtomReducer.SCALAR,
    *,
    source: str,
) -> tuple[_AtomValue, _AtomValue]:
    return (
        _oracle_atom(raw, truth, reducer, source=source),
        _oracle_atom(raw, truth, reducer, source=source),
    )


def _float_value(
    point: ThresholdPoint,
    threshold: float,
    *,
    low: float | None = None,
    high: float | None = None,
    extreme: float = math.inf,
) -> float:
    role = point.role
    if role is PointRole.VERY_LOW:
        return threshold / 4.0 if low is None else low
    if role is PointRole.BELOW_RELATIVE_1E6:
        return threshold * (1.0 - 1e-6)
    if role is PointRole.BELOW_RELATIVE_1E12:
        return threshold * (1.0 - 1e-12)
    if role is PointRole.BELOW_ULP:
        return _ORIGINAL_NEXTAFTER(threshold, -math.inf)
    if role in {PointRole.AT, PointRole.EXACT}:
        return threshold
    if role is PointRole.ABOVE_ULP:
        return _ORIGINAL_NEXTAFTER(threshold, math.inf)
    if role is PointRole.ABOVE_RELATIVE_1E12:
        return threshold * (1.0 + 1e-12)
    if role is PointRole.ABOVE_RELATIVE_1E6:
        return threshold * (1.0 + 1e-6)
    if role is PointRole.VERY_HIGH:
        return threshold * 2.0 if high is None else high
    if role is PointRole.EXTREME:
        return extreme
    raise AssertionError(f"{role.value} is not a float-grid role")


def _zero_value(
    point: ThresholdPoint,
    *,
    low: float = -1.0,
    high: float = 8.0,
    extreme: float = math.inf,
) -> float:
    if point.role is PointRole.VERY_LOW:
        return low
    if point.role is PointRole.BELOW_ULP:
        return _ORIGINAL_NEXTAFTER(0.0, -math.inf)
    if point.role is PointRole.AT:
        return 0.0
    if point.role is PointRole.ABOVE_ULP:
        return _ORIGINAL_NEXTAFTER(0.0, math.inf)
    if point.role is PointRole.VERY_HIGH:
        return high
    if point.role is PointRole.EXTREME:
        return extreme
    raise AssertionError(f"relative zero role {point.role.value} was not omitted")


def _integer_value(
    point: ThresholdPoint,
    threshold: int,
    *,
    low: int = 0,
    high: int | None = None,
    extreme: int = -1,
) -> int:
    if point.role is PointRole.VERY_LOW:
        return low
    if point.role is PointRole.BELOW_INTEGER:
        return threshold - 1
    if point.role is PointRole.AT:
        return threshold
    if point.role is PointRole.ABOVE_INTEGER:
        return threshold + 1
    if point.role is PointRole.VERY_HIGH:
        return threshold * 4 + 1 if high is None else high
    if point.role is PointRole.EXTREME:
        return extreme
    raise AssertionError(f"{point.role.value} is not an integer-grid role")


def _exact_value(point: ThresholdPoint, threshold: float) -> float:
    if point.role in {
        PointRole.BELOW_RELATIVE_1E6,
        PointRole.BELOW_RELATIVE_1E12,
        PointRole.BELOW_ULP,
        PointRole.AT,
        PointRole.ABOVE_ULP,
        PointRole.ABOVE_RELATIVE_1E12,
        PointRole.ABOVE_RELATIVE_1E6,
    }:
        return _float_value(
            point,
            threshold,
            low=threshold / 4.0,
            high=threshold * 4.0,
            extreme=math.inf,
        )
    if point.role is PointRole.VERY_LOW:
        return threshold / 4.0
    if point.role is PointRole.EXACT:
        return threshold
    if point.role is PointRole.ULP_MISMATCH:
        return _ORIGINAL_NEXTAFTER(threshold, math.inf)
    if point.role is PointRole.SUBNORMAL_MISMATCH:
        return threshold + _ORIGINAL_NEXTAFTER(0.0, math.inf)
    if point.role is PointRole.MATERIAL_MISMATCH:
        return threshold + max(abs(threshold), 1.0) * 1e-6
    if point.role is PointRole.VERY_HIGH:
        return threshold * 4.0
    if point.role is PointRole.EXTREME:
        return math.inf
    raise AssertionError(f"{point.role.value} is not an exact-grid role")


def _raw_equal(left: Any, right: Any) -> bool:
    first = np.asarray(left)
    second = np.asarray(right)
    equal = (
        np.array_equal(first, second, equal_nan=True)
        if first.dtype.kind in {"f", "c"} and second.dtype.kind in {"f", "c"}
        else np.array_equal(first, second)
    )
    return bool(first.dtype == second.dtype and first.shape == second.shape and equal)


def _certificate_fields(
    *,
    measured: float = 0.1,
    certified: float = 0.2,
    margin: float = 0.1,
    tolerance: float = 0.5,
    tail: float = 0.25,
    multiplicity: int = 2,
    lambda_scale: float | None = 3.0,
    x_scale: float | None = 0.5,
    order: int | None = None,
) -> dict[str, Any]:
    selected = (
        oracles.smallest_trace_order(certified, tail, multiplicity)
        if order is None
        else order
    )
    return {
        "measured_max": measured,
        "margin": margin,
        "certified_rho": certified,
        "order": selected,
        "tolerance": tolerance,
        "tail_tolerance": tail,
        "multiplicity": multiplicity,
        "max_abs_lambda_logdet": lambda_scale,
        "max_x_operator_norm": x_scale,
    }


def _certificate_from(data: Mapping[str, Any]) -> plan.RhoCertificate:
    return plan.RhoCertificate(
        measured_max=data["measured_max"],
        margin=data["margin"],
        certified_rho=data["certified_rho"],
        order=data["order"],
        tolerance=data["tolerance"],
        tail_tolerance=data["tail_tolerance"],
        multiplicity=data["multiplicity"],
        max_abs_lambda_logdet=data["max_abs_lambda_logdet"],
        max_x_operator_norm=data["max_x_operator_norm"],
    )


def _validation_outcome(
    *,
    expected_valid: bool,
    operation: Callable[[], Any],
    refused_exceptions: tuple[type[BaseException], ...],
    return_value: Callable[[Any], Mapping[str, Any]],
    actual_atoms: Mapping[str, _AtomValue] = MappingProxyType({}),
    oracle_atoms: Mapping[str, _AtomValue] = MappingProxyType({}),
    payload_checks: tuple[tuple[str, Any, Any, bool], ...] = (),
    direct_calls: tuple[str, ...],
) -> _Outcome:
    try:
        result = operation()
    except refused_exceptions as error:
        observed = GateSide.REFUSED
        returns: Mapping[str, Any] = {"raised_type": type(error).__name__}
    else:
        observed = GateSide.ADMITTED
        returns = return_value(result)
    return _Outcome(
        observed_side=observed,
        oracle_side=GateSide.ADMITTED if expected_valid else GateSide.REFUSED,
        returns=returns,
        actual_atoms=actual_atoms,
        oracle_atoms=oracle_atoms,
        payload_checks=payload_checks,
        direct_calls=direct_calls,
    )


def _certificate_fixture(
    entry: GateEntry,
    point: ThresholdPoint,
    atom: _AtomSpec | None,
) -> _Fixture:
    gate = entry.gate_id
    if gate == "PLAN:multiplicity:index-and-gamma-domain":
        threshold = _ORACLE_RHO_MULTIPLICITY_LIMIT
        if point.display_value == "extreme bool":
            value: Any = True
        elif point.display_value == "extreme float":
            value = 1.0
        elif point.display_value == "extreme huge integer":
            value = 10**1000
        else:
            value = _integer_value(point, threshold, low=1, extreme=-1)
        if atom is not None:
            value = threshold
        inputs = {"multiplicity": value}
        valid = (
            not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, np.integer))
            and 1 <= int(value) < threshold
        )

        def invoke(data: Mapping[str, Any]) -> _Outcome:
            production_product = bool(
                data["multiplicity"] >= plan._RHO_MULTIPLICITY_LIMIT
            )
            oracle_product = int(data["multiplicity"]) >= int(threshold)
            actual_rows = _atom_row(
                entry,
                (
                    _production_atom(
                        production_product,
                        production_product,
                        source="_normalize_rho_multiplicity input comparison",
                    ),
                ),
            )
            oracle_rows = _atom_row(
                entry,
                (
                    _oracle_atom(
                        oracle_product,
                        oracle_product,
                        source="integer-index gamma-domain comparison",
                    ),
                ),
            )
            return _validation_outcome(
                expected_valid=valid,
                operation=lambda: plan._normalize_rho_multiplicity(
                    data["multiplicity"]
                ),
                refused_exceptions=(TypeError, ValueError),
                return_value=lambda result: {"normalized_multiplicity": result},
                actual_atoms=actual_rows,
                oracle_atoms=oracle_rows,
                direct_calls=("plan._normalize_rho_multiplicity",),
            )

        return _Fixture(
            inputs,
            "multiplicity",
            threshold,
            None,
            "multiplicity",
            "operator.index plus exact float64 gamma domain",
            invoke,
        )

    fields = _certificate_fields()
    point_key = "certified_rho"
    threshold: float | int = 1.0
    if gate == "PLAN:certificate:rho-domain-and-coverage":
        fields["certified_rho"] = _float_value(
            point, 1.0, low=0.2, high=1.5, extreme=math.nan
        )
        if atom is not None:
            if atom.index == 0:
                point_key = "measured_max"
                fields[point_key] = -_ORIGINAL_NEXTAFTER(0.0, math.inf)
            elif atom.index == 1:
                fields["certified_rho"] = 1.0
            else:
                fields["measured_max"] = 0.3
                fields["certified_rho"] = 0.2
    elif gate == "PLAN:certificate:error-budget-domain":
        parameter_cell = _ERROR_BUDGET_AXIS_CELL_BY_DISPLAY.get(point.display_value)
        if atom is None and parameter_cell is not None:
            point_key = parameter_cell.field
            threshold = parameter_cell.threshold
            fields[point_key] = parameter_cell.value
        else:
            point_key = "tail_tolerance"
            threshold = fields["tolerance"]
            fields[point_key] = _float_value(
                point, float(threshold), low=0.1, high=0.75, extreme=-2.0
            )
        if atom is not None:
            if atom.index == 0:
                point_key = "margin"
                threshold = 0.0
                fields[point_key] = -_ORIGINAL_NEXTAFTER(0.0, math.inf)
            elif atom.index == 1:
                point_key = "tolerance"
                threshold = 0.0
                fields[point_key] = 0.0
            else:
                fields["tail_tolerance"] = fields["tolerance"]
    elif gate == "PLAN:certificate:optional-scale-domain":
        point_key = "max_abs_lambda_logdet"
        threshold = 0.0
        fields[point_key] = _zero_value(point)
        if atom is not None:
            fields["max_abs_lambda_logdet"] = 3.0
            fields["max_x_operator_norm"] = 0.5
            point_key = (
                "max_abs_lambda_logdet" if atom.index < 5 else "max_x_operator_norm"
            )
            local = atom.index % 5
            if local == 0:
                fields[point_key] = None
            elif local in {1, 2, 3}:
                fields[point_key] = math.inf
            else:
                fields[point_key] = -_ORIGINAL_NEXTAFTER(0.0, math.inf)
    elif gate == "PLAN:certificate:order-is-derived":
        threshold = _DERIVED_ORDER_RHO_THRESHOLD
        fields.update(
            measured_max=0.0,
            margin=0.0,
            certified_rho=_float_value(
                point,
                threshold,
                low=0.1,
                high=0.8,
                extreme=0.99,
            ),
            tail_tolerance=_DERIVED_ORDER_TAIL_THRESHOLD,
            tolerance=0.5,
            multiplicity=_DERIVED_ORDER_MULTIPLICITY_THRESHOLD,
            order=_DERIVED_ORDER_TARGET,
        )
        point_key = "certified_rho"
        if atom is not None:
            fields["certified_rho"] = _DERIVED_ORDER_RHO_THRESHOLD
            fields["order"] = _DERIVED_ORDER_TARGET + 1
    else:
        raise AssertionError(f"unmapped certificate gate {gate}")

    if gate == "PLAN:certificate:error-budget-domain":
        try:
            fields["order"] = oracles.smallest_trace_order(
                fields["certified_rho"],
                fields["tail_tolerance"],
                fields["multiplicity"],
            )
        except (TypeError, ValueError):
            # Invalid domain cells are refused before the derived-order check.
            pass

    measured = fields["measured_max"]
    certified = fields["certified_rho"]
    margin = fields["margin"]
    tolerance = fields["tolerance"]
    tail = fields["tail_tolerance"]
    lambda_scale = fields["max_abs_lambda_logdet"]
    x_scale = fields["max_x_operator_norm"]
    rho_measured = bool(0.0 <= measured < 1.0)
    rho_certified = bool(0.0 <= certified < 1.0)
    rho_under = bool(certified < measured)
    error_margin = bool(margin < 0.0)
    error_tolerance = bool(tolerance <= 0.0)
    error_tail = bool(0.0 < tail < tolerance)

    valid = (
        rho_measured
        and rho_certified
        and not rho_under
        and not error_margin
        and not error_tolerance
        and error_tail
        and (
            lambda_scale is None or math.isfinite(lambda_scale) and lambda_scale >= 0.0
        )
        and (x_scale is None or math.isfinite(x_scale) and x_scale >= 0.0)
    )
    if valid and gate not in {
        "PLAN:certificate:rho-domain-and-coverage",
        "PLAN:certificate:error-budget-domain",
    }:
        valid = fields["order"] == oracles.smallest_trace_order(
            certified, tail, fields["multiplicity"]
        )

    def invoke(data: Mapping[str, Any]) -> _Outcome:
        actual_values: tuple[_AtomValue, ...]
        oracle_values: tuple[_AtomValue, ...]

        def production_skipped(source: str) -> _AtomValue:
            return _production_atom(
                None, None, AtomReducer.NOT_EVALUATED, source=source
            )

        def oracle_skipped(source: str) -> _AtomValue:
            return _oracle_atom(None, None, AtomReducer.NOT_EVALUATED, source=source)

        if gate == "PLAN:certificate:rho-domain-and-coverage":
            production_measured = bool(0.0 <= data["measured_max"] < 1.0)
            measured_value = float(data["measured_max"])
            oracle_measured = bool(
                not math.isnan(measured_value)
                and Decimal(0) <= Decimal.from_float(measured_value) < Decimal(1)
            )
            if not production_measured:
                production_certified = None
                production_under = None
            else:
                production_certified = bool(0.0 <= data["certified_rho"] < 1.0)
                production_under = (
                    None
                    if not production_certified
                    else bool(data["certified_rho"] < data["measured_max"])
                )
            if not oracle_measured:
                oracle_certified = None
                oracle_under = None
            else:
                certified_value = float(data["certified_rho"])
                certified_decimal = Decimal.from_float(certified_value)
                oracle_certified = bool(
                    not math.isnan(certified_value)
                    and Decimal(0) <= certified_decimal < Decimal(1)
                )
                oracle_under = (
                    None
                    if not oracle_certified
                    else certified_decimal
                    < Decimal.from_float(float(data["measured_max"]))
                )
            actual_values = (
                _production_atom(
                    production_measured,
                    production_measured,
                    source="RhoCertificate measured_max comparison",
                ),
                production_skipped("RhoCertificate stopped after measured_max")
                if production_certified is None
                else _production_atom(
                    production_certified,
                    production_certified,
                    source="RhoCertificate certified_rho comparison",
                ),
                production_skipped("RhoCertificate stopped before coverage")
                if production_under is None
                else _production_atom(
                    production_under,
                    production_under,
                    source="RhoCertificate coverage comparison",
                ),
            )
            oracle_values = (
                _oracle_atom(
                    oracle_measured,
                    oracle_measured,
                    source="Decimal measured_max domain",
                ),
                oracle_skipped("Decimal branch stopped after measured_max")
                if oracle_certified is None
                else _oracle_atom(
                    oracle_certified,
                    oracle_certified,
                    source="Decimal certified_rho domain",
                ),
                oracle_skipped("Decimal branch stopped before coverage")
                if oracle_under is None
                else _oracle_atom(
                    oracle_under,
                    oracle_under,
                    source="Decimal certificate coverage",
                ),
            )
        elif gate == "PLAN:certificate:error-budget-domain":
            production_margin = bool(data["margin"] < 0.0)
            production_tolerance = (
                None if production_margin else bool(data["tolerance"] <= 0.0)
            )
            production_tail = (
                None
                if production_margin or production_tolerance
                else bool(0.0 < data["tail_tolerance"] < data["tolerance"])
            )
            margin_value = float(data["margin"])
            tolerance_value = float(data["tolerance"])
            tail_value = float(data["tail_tolerance"])
            margin_decimal = Decimal.from_float(margin_value)
            tolerance_decimal = Decimal.from_float(tolerance_value)
            tail_decimal = Decimal.from_float(tail_value)
            oracle_margin = bool(
                not math.isnan(margin_value) and margin_decimal < Decimal(0)
            )
            oracle_tolerance = (
                None
                if oracle_margin
                else bool(
                    not math.isnan(tolerance_value) and tolerance_decimal <= Decimal(0)
                )
            )
            oracle_tail = (
                None
                if oracle_margin or oracle_tolerance
                else bool(
                    not (math.isnan(tail_value) or math.isnan(tolerance_value))
                    and Decimal(0) < tail_decimal < tolerance_decimal
                )
            )

            def production_item(value: bool | None, source: str) -> _AtomValue:
                return (
                    production_skipped(source)
                    if value is None
                    else _production_atom(value, value, source=source)
                )

            def oracle_item(value: bool | None, source: str) -> _AtomValue:
                return (
                    oracle_skipped(source)
                    if value is None
                    else _oracle_atom(value, value, source=source)
                )

            actual_values = (
                production_item(production_margin, "RhoCertificate margin comparison"),
                production_item(
                    production_tolerance, "RhoCertificate tolerance comparison"
                ),
                production_item(
                    production_tail, "RhoCertificate tail-tolerance comparison"
                ),
            )
            oracle_values = (
                oracle_item(oracle_margin, "Decimal margin domain"),
                oracle_item(oracle_tolerance, "Decimal tolerance domain"),
                oracle_item(oracle_tail, "Decimal tail-tolerance domain"),
            )
        elif gate == "PLAN:certificate:optional-scale-domain":

            def production_optional(
                value: float | None, *, source: str, enabled: bool
            ) -> tuple[_AtomValue, ...]:
                if not enabled:
                    return tuple(
                        production_skipped(f"{source} skipped after prior refusal")
                        for _ in range(5)
                    )
                supplied = value is not None
                if not supplied:
                    return (
                        _production_atom(False, False, source=f"{source} presence"),
                        production_skipped(f"{source} absent invalid predicate"),
                        production_skipped(f"{source} absent finite predicate"),
                        production_skipped(f"{source} absent finite alias"),
                        production_skipped(f"{source} absent negative predicate"),
                    )
                finite_raw = np.isfinite(value)
                finite = bool(finite_raw)
                negative = None if not finite else bool(value < 0.0)
                invalid = bool(not finite or negative)
                return (
                    _production_atom(True, True, source=f"{source} presence"),
                    _production_atom(
                        invalid, invalid, source=f"{source} invalid disjunction"
                    ),
                    *_production_aliases(
                        finite_raw, finite, source=f"{source} np.isfinite"
                    ),
                    production_skipped(f"{source} finite short-circuit")
                    if negative is None
                    else _production_atom(
                        negative, negative, source=f"{source} negative comparison"
                    ),
                )

            def oracle_optional(
                value: float | None, *, source: str, enabled: bool
            ) -> tuple[_AtomValue, ...]:
                if not enabled:
                    return tuple(
                        oracle_skipped(f"{source} skipped after prior refusal")
                        for _ in range(5)
                    )
                supplied = value is not None
                if not supplied:
                    return (
                        _oracle_atom(False, False, source=f"{source} presence"),
                        oracle_skipped(f"{source} absent invalid predicate"),
                        oracle_skipped(f"{source} absent finite predicate"),
                        oracle_skipped(f"{source} absent finite alias"),
                        oracle_skipped(f"{source} absent negative predicate"),
                    )
                finite = math.isfinite(float(value))
                negative = (
                    None
                    if not finite
                    else Decimal.from_float(float(value)) < Decimal(0)
                )
                invalid = bool(not finite or negative)
                return (
                    _oracle_atom(True, True, source=f"{source} presence"),
                    _oracle_atom(invalid, invalid, source=f"{source} invalid domain"),
                    *_oracle_aliases(finite, finite, source=f"{source} math.isfinite"),
                    oracle_skipped(f"{source} finite short-circuit")
                    if negative is None
                    else _oracle_atom(
                        negative, negative, source=f"{source} Decimal sign"
                    ),
                )

            actual_lambda = production_optional(
                data["max_abs_lambda_logdet"],
                source="lambda scale",
                enabled=True,
            )
            oracle_lambda = oracle_optional(
                data["max_abs_lambda_logdet"],
                source="lambda scale",
                enabled=True,
            )
            lambda_refused = actual_lambda[1].truth is True
            oracle_lambda_refused = oracle_lambda[1].truth is True
            actual_values = (
                *actual_lambda,
                *production_optional(
                    data["max_x_operator_norm"],
                    source="X norm scale",
                    enabled=not lambda_refused,
                ),
            )
            oracle_values = (
                *oracle_lambda,
                *oracle_optional(
                    data["max_x_operator_norm"],
                    source="X norm scale",
                    enabled=not oracle_lambda_refused,
                ),
            )
        else:
            production_expected = plan.choose_trace_order(
                data["certified_rho"],
                data["tail_tolerance"],
                multiplicity=data["multiplicity"],
            )
            oracle_expected = oracles.smallest_trace_order(
                data["certified_rho"],
                data["tail_tolerance"],
                data["multiplicity"],
            )
            production_wrong = data["order"] != production_expected
            oracle_wrong = int(data["order"]) != int(oracle_expected)
            actual_values = (
                _production_atom(
                    production_wrong,
                    production_wrong,
                    source="RhoCertificate selected-order comparison",
                ),
            )
            oracle_values = (
                _oracle_atom(
                    oracle_wrong,
                    oracle_wrong,
                    source="Decimal trace-tail minimum order",
                ),
            )

        actual_rows = _atom_row(entry, actual_values)
        oracle_rows = _atom_row(entry, oracle_values)
        return _validation_outcome(
            expected_valid=valid,
            operation=lambda: _certificate_from(data),
            refused_exceptions=(TypeError, ValueError),
            return_value=lambda result: {
                "returned_certified_rho": result.certified_rho,
                "returned_order": result.order,
            },
            actual_atoms=actual_rows,
            oracle_atoms=oracle_rows,
            direct_calls=("plan.RhoCertificate",),
        )

    return _Fixture(
        fields,
        point_key,
        threshold,
        None if isinstance(threshold, int) else np.dtype(np.float64).str,
        point_key,
        "independent certificate domains and Decimal trace-tail order",
        invoke,
    )


def _rho_margin_for_target(target: float, multiplicity: int) -> float:
    return _rho_raw_bound_for_target(target, multiplicity)


def _warmup_fixture(
    entry: GateEntry,
    point: ThresholdPoint,
    atom: _AtomSpec | None,
) -> _Fixture:
    gate = entry.gate_id
    inputs: dict[str, Any] = {
        "measured_rhos": (0.1,),
        "margin": 0.1,
        "tolerance": 0.5,
        "multiplicity": 2,
        "tail_fraction": 0.5,
        "lambda_logdets": (1.7, -2.9),
        "lambda_logdet_margin": 0.2,
        "x_operator_norms": (0.2, 0.3),
        "x_operator_norm_margin": 0.1,
    }
    threshold = 0.0
    point_key = "margin"
    if gate == "PLAN:warmup:rho-inputs-and-margin":
        inputs[point_key] = _zero_value(point, high=0.2)
        if atom is not None:
            if atom.index in {0, 1, 2}:
                point_key = "rho_value"
                inputs["measured_rhos"] = (math.inf,)
                inputs[point_key] = math.inf
            else:
                point_key = "rho_value"
                value = -_ORIGINAL_NEXTAFTER(0.0, math.inf)
                inputs["measured_rhos"] = (value,)
                inputs[point_key] = value
        else:
            inputs["rho_value"] = inputs["measured_rhos"][0]
    elif gate == "PLAN:warmup:tail-fraction":
        threshold = 1.0
        point_key = "tail_fraction"
        inputs[point_key] = _float_value(
            point, threshold, low=0.25, high=1.5, extreme=math.inf
        )
        if atom is not None:
            inputs[point_key] = math.inf if atom.index < 2 else 0.0
    elif gate == "PLAN:warmup:lambda-scale-inputs":
        point_key = "lambda_logdet_margin"
        inputs[point_key] = _zero_value(point)
        if atom is not None:
            if atom.index < 2:
                point_key = "lambda_value"
                inputs["lambda_logdets"] = (math.inf,)
                inputs[point_key] = math.inf
            elif atom.index < 4:
                inputs["lambda_logdet_margin"] = math.inf
            else:
                inputs["lambda_logdet_margin"] = -_ORIGINAL_NEXTAFTER(0.0, math.inf)
        inputs.setdefault("lambda_value", inputs["lambda_logdets"][0])
    elif gate == "PLAN:warmup:x-norm-inputs":
        point_key = "x_operator_norm_margin"
        inputs[point_key] = _zero_value(point)
        if atom is not None:
            if atom.index < 3:
                point_key = "x_norm_value"
                inputs["x_operator_norms"] = (math.inf,)
                inputs[point_key] = math.inf
            elif atom.index == 3:
                point_key = "x_norm_value"
                negative = -_ORIGINAL_NEXTAFTER(0.0, math.inf)
                inputs["x_operator_norms"] = (negative,)
                inputs[point_key] = negative
            elif atom.index < 6:
                inputs["x_operator_norm_margin"] = math.inf
            else:
                inputs["x_operator_norm_margin"] = -_ORIGINAL_NEXTAFTER(0.0, math.inf)
        inputs.setdefault("x_norm_value", inputs["x_operator_norms"][0])
    elif gate == "PLAN:warmup:rho-roundoff-ceiling":
        threshold = 1.0
        point_key = "certified_stage"
        target = _float_value(point, threshold, low=0.25, high=1.5, extreme=math.inf)
        if atom is not None:
            target = 1.0
        inputs["margin"] = (
            target
            if not math.isfinite(target)
            else _rho_margin_for_target(target, inputs["multiplicity"])
        )
        inputs["measured_rhos"] = (0.0,)
        inputs["rho_value"] = 0.0
        inputs["lambda_logdets"] = None
        inputs["x_operator_norms"] = None
    else:
        raise AssertionError(f"unmapped warmup gate {gate}")

    values = inputs["measured_rhos"]
    margin = inputs["margin"]
    fraction = inputs["tail_fraction"]
    bases = inputs["lambda_logdets"]
    lambda_margin = inputs["lambda_logdet_margin"]
    norms = inputs["x_operator_norms"]
    norm_margin = inputs["x_operator_norm_margin"]
    domain_valid = (
        bool(values)
        and all(math.isfinite(value) and value >= 0.0 for value in values)
        and math.isfinite(margin)
        and margin >= 0.0
        and math.isfinite(fraction)
        and 0.0 < fraction < 1.0
        and (
            bases is None
            or bool(bases)
            and all(math.isfinite(value) for value in bases)
        )
        and math.isfinite(lambda_margin)
        and lambda_margin >= 0.0
        and (
            norms is None
            or bool(norms)
            and all(math.isfinite(value) and value >= 0.0 for value in norms)
        )
        and math.isfinite(norm_margin)
        and norm_margin >= 0.0
    )
    expected_certified = math.inf
    if domain_valid:
        raw = max(values) + margin
        epsilon = float(np.finfo(np.float64).eps)
        product = inputs["multiplicity"] * epsilon
        expected_certified = _ORIGINAL_NEXTAFTER(
            raw + abs(raw) * product / (1.0 - product), math.inf
        )
    expected_valid = domain_valid and expected_certified < 1.0

    def invoke(data: Mapping[str, Any]) -> _Outcome:
        captured: list[float] = []

        def production_skipped(source: str) -> _AtomValue:
            return _production_atom(
                None, None, AtomReducer.NOT_EVALUATED, source=source
            )

        def oracle_skipped(source: str) -> _AtomValue:
            return _oracle_atom(None, None, AtomReducer.NOT_EVALUATED, source=source)

        actual_values: tuple[_AtomValue, ...] = ()
        oracle_values: tuple[_AtomValue, ...] = ()
        if gate == "PLAN:warmup:rho-inputs-and-margin":
            production_value = data["measured_rhos"][0]
            production_finite_raw = np.isfinite(production_value)
            production_finite = bool(production_finite_raw)
            production_negative = (
                None if not production_finite else bool(production_value < 0.0)
            )
            production_invalid = bool(not production_finite or production_negative)
            oracle_value = Decimal.from_float(float(production_value))
            oracle_finite = math.isfinite(float(production_value))
            oracle_negative = None if not oracle_finite else oracle_value < Decimal(0)
            oracle_invalid = bool(not oracle_finite or oracle_negative)
            actual_values = (
                _production_atom(
                    production_invalid,
                    production_invalid,
                    source="warmup rho invalid disjunction",
                ),
                *_production_aliases(
                    production_finite_raw,
                    production_finite,
                    source="warmup rho np.isfinite",
                ),
                production_skipped("warmup rho finite short-circuit")
                if production_negative is None
                else _production_atom(
                    production_negative,
                    production_negative,
                    source="warmup rho negative comparison",
                ),
            )
            oracle_values = (
                _oracle_atom(
                    oracle_invalid,
                    oracle_invalid,
                    source="Decimal warmup rho domain",
                ),
                *_oracle_aliases(
                    oracle_finite,
                    oracle_finite,
                    source="math.isfinite warmup rho",
                ),
                oracle_skipped("oracle rho finite short-circuit")
                if oracle_negative is None
                else _oracle_atom(
                    oracle_negative,
                    oracle_negative,
                    source="Decimal warmup rho sign",
                ),
            )
        elif gate == "PLAN:warmup:tail-fraction":
            production_finite_raw = np.isfinite(data["tail_fraction"])
            production_finite = bool(production_finite_raw)
            production_domain = (
                None
                if not production_finite
                else bool(0.0 < data["tail_fraction"] < 1.0)
            )
            oracle_finite = math.isfinite(float(data["tail_fraction"]))
            oracle_decimal = Decimal.from_float(float(data["tail_fraction"]))
            oracle_domain = (
                None if not oracle_finite else Decimal(0) < oracle_decimal < Decimal(1)
            )
            actual_values = (
                *_production_aliases(
                    production_finite_raw,
                    production_finite,
                    source="tail fraction np.isfinite",
                ),
                production_skipped("tail fraction finite short-circuit")
                if production_domain is None
                else _production_atom(
                    production_domain,
                    production_domain,
                    source="tail fraction strict interval",
                ),
            )
            oracle_values = (
                *_oracle_aliases(
                    oracle_finite,
                    oracle_finite,
                    source="tail fraction math.isfinite",
                ),
                oracle_skipped("oracle tail fraction finite short-circuit")
                if oracle_domain is None
                else _oracle_atom(
                    oracle_domain,
                    oracle_domain,
                    source="Decimal tail fraction strict interval",
                ),
            )
        elif gate == "PLAN:warmup:lambda-scale-inputs":
            production_value = data["lambda_logdets"][0]
            production_value_raw = np.isfinite(production_value)
            production_value_finite = bool(production_value_raw)
            oracle_value_finite = math.isfinite(float(production_value))
            if production_value_finite:
                production_margin_raw = np.isfinite(data["lambda_logdet_margin"])
                production_margin_finite = bool(production_margin_raw)
                production_negative = (
                    None
                    if not production_margin_finite
                    else bool(data["lambda_logdet_margin"] < 0.0)
                )
            else:
                production_margin_raw = None
                production_margin_finite = None
                production_negative = None
            if oracle_value_finite:
                oracle_margin_finite = math.isfinite(
                    float(data["lambda_logdet_margin"])
                )
                oracle_negative = (
                    None
                    if not oracle_margin_finite
                    else Decimal.from_float(float(data["lambda_logdet_margin"]))
                    < Decimal(0)
                )
            else:
                oracle_margin_finite = None
                oracle_negative = None
            actual_values = (
                *_production_aliases(
                    production_value_raw,
                    production_value_finite,
                    source="lambda warmup value np.isfinite",
                ),
                *(
                    (
                        production_skipped("lambda value refusal skipped margin"),
                        production_skipped("lambda value refusal skipped margin alias"),
                    )
                    if production_margin_finite is None
                    else _production_aliases(
                        production_margin_raw,
                        production_margin_finite,
                        source="lambda warmup margin np.isfinite",
                    )
                ),
                production_skipped("lambda margin finite short-circuit")
                if production_negative is None
                else _production_atom(
                    production_negative,
                    production_negative,
                    source="lambda warmup margin sign",
                ),
            )
            oracle_values = (
                *_oracle_aliases(
                    oracle_value_finite,
                    oracle_value_finite,
                    source="lambda value math.isfinite",
                ),
                *(
                    (
                        oracle_skipped("lambda value refusal skipped margin"),
                        oracle_skipped("lambda value refusal skipped margin alias"),
                    )
                    if oracle_margin_finite is None
                    else _oracle_aliases(
                        oracle_margin_finite,
                        oracle_margin_finite,
                        source="lambda margin math.isfinite",
                    )
                ),
                oracle_skipped("oracle lambda margin finite short-circuit")
                if oracle_negative is None
                else _oracle_atom(
                    oracle_negative,
                    oracle_negative,
                    source="Decimal lambda margin sign",
                ),
            )
        elif gate == "PLAN:warmup:x-norm-inputs":
            production_value = data["x_operator_norms"][0]
            production_finite_raw = np.isfinite(production_value)
            production_finite = bool(production_finite_raw)
            production_negative = (
                None if not production_finite else bool(production_value < 0.0)
            )
            production_invalid = bool(not production_finite or production_negative)
            oracle_finite = math.isfinite(float(production_value))
            oracle_negative = (
                None
                if not oracle_finite
                else Decimal.from_float(float(production_value)) < Decimal(0)
            )
            oracle_invalid = bool(not oracle_finite or oracle_negative)
            if not production_invalid:
                production_margin_raw = np.isfinite(data["x_operator_norm_margin"])
                production_margin_finite = bool(production_margin_raw)
                production_margin_negative = (
                    None
                    if not production_margin_finite
                    else bool(data["x_operator_norm_margin"] < 0.0)
                )
            else:
                production_margin_raw = None
                production_margin_finite = None
                production_margin_negative = None
            if not oracle_invalid:
                oracle_margin_finite = math.isfinite(
                    float(data["x_operator_norm_margin"])
                )
                oracle_margin_negative = (
                    None
                    if not oracle_margin_finite
                    else Decimal.from_float(float(data["x_operator_norm_margin"]))
                    < Decimal(0)
                )
            else:
                oracle_margin_finite = None
                oracle_margin_negative = None
            actual_values = (
                _production_atom(
                    production_invalid,
                    production_invalid,
                    source="X norm invalid disjunction",
                ),
                *_production_aliases(
                    production_finite_raw,
                    production_finite,
                    source="X norm value np.isfinite",
                ),
                production_skipped("X norm finite short-circuit")
                if production_negative is None
                else _production_atom(
                    production_negative,
                    production_negative,
                    source="X norm value sign",
                ),
                *(
                    (
                        production_skipped("X norm refusal skipped margin"),
                        production_skipped("X norm refusal skipped margin alias"),
                    )
                    if production_margin_finite is None
                    else _production_aliases(
                        production_margin_raw,
                        production_margin_finite,
                        source="X norm margin np.isfinite",
                    )
                ),
                production_skipped("X norm margin finite short-circuit")
                if production_margin_negative is None
                else _production_atom(
                    production_margin_negative,
                    production_margin_negative,
                    source="X norm margin sign",
                ),
            )
            oracle_values = (
                _oracle_atom(
                    oracle_invalid,
                    oracle_invalid,
                    source="Decimal X norm domain",
                ),
                *_oracle_aliases(
                    oracle_finite,
                    oracle_finite,
                    source="X norm math.isfinite",
                ),
                oracle_skipped("oracle X norm finite short-circuit")
                if oracle_negative is None
                else _oracle_atom(
                    oracle_negative,
                    oracle_negative,
                    source="Decimal X norm sign",
                ),
                *(
                    (
                        oracle_skipped("X norm refusal skipped margin"),
                        oracle_skipped("X norm refusal skipped margin alias"),
                    )
                    if oracle_margin_finite is None
                    else _oracle_aliases(
                        oracle_margin_finite,
                        oracle_margin_finite,
                        source="X norm margin math.isfinite",
                    )
                ),
                oracle_skipped("oracle X margin finite short-circuit")
                if oracle_margin_negative is None
                else _oracle_atom(
                    oracle_margin_negative,
                    oracle_margin_negative,
                    source="Decimal X norm margin sign",
                ),
            )

        actual_rows = (
            _atom_row(entry, actual_values) if actual_values else MappingProxyType({})
        )
        oracle_rows = (
            _atom_row(entry, oracle_values) if oracle_values else MappingProxyType({})
        )

        def capture_nextafter(value: float, direction: float) -> float:
            result = _ORIGINAL_NEXTAFTER(value, direction)
            if direction == math.inf:
                captured.append(result)
            return result

        operation = lambda: plan.certify_warmup_rho(
            data["measured_rhos"],
            margin=data["margin"],
            tolerance=data["tolerance"],
            multiplicity=data["multiplicity"],
            tail_fraction=data["tail_fraction"],
            lambda_logdets=data["lambda_logdets"],
            lambda_logdet_margin=data["lambda_logdet_margin"],
            x_operator_norms=data["x_operator_norms"],
            x_operator_norm_margin=data["x_operator_norm_margin"],
        )
        with (
            patch.object(plan.math, "nextafter", side_effect=capture_nextafter),
        ):
            outcome = _validation_outcome(
                expected_valid=expected_valid,
                operation=operation,
                refused_exceptions=(TypeError, ValueError, OverflowError),
                return_value=lambda result: {
                    "returned_certified_rho": result.certified_rho,
                    "returned_order": result.order,
                },
                actual_atoms=actual_rows,
                oracle_atoms=oracle_rows,
                direct_calls=("plan.certify_warmup_rho",),
            )
        if gate != "PLAN:warmup:rho-roundoff-ceiling":
            return outcome
        stage = captured[0] if captured else expected_certified
        production_comparison = bool(stage < 1.0)
        oracle_comparison = Decimal.from_float(float(expected_certified)) < Decimal(1)
        actual_stage_rows = _atom_row(
            entry,
            (
                _production_atom(
                    production_comparison,
                    production_comparison,
                    source="captured certified-rho stage",
                ),
            ),
        )
        oracle_stage_rows = _atom_row(
            entry,
            (
                _oracle_atom(
                    oracle_comparison,
                    oracle_comparison,
                    source="Decimal gamma-envelope ceiling",
                ),
            ),
        )
        returns = dict(outcome.returns)
        returns["certified_stage"] = stage
        return _Outcome(
            outcome.observed_side,
            outcome.oracle_side,
            returns,
            actual_stage_rows,
            oracle_stage_rows,
            direct_calls=outcome.direct_calls,
        )

    return _Fixture(
        inputs,
        point_key,
        threshold,
        np.dtype(np.float64).str,
        "margin" if gate == "PLAN:warmup:rho-roundoff-ceiling" else point_key,
        "independent warmup domains and exact-rational gamma envelope",
        invoke,
    )


def _audit_fixture(
    entry: GateEntry,
    point: ThresholdPoint,
    atom: _AtomSpec | None,
) -> _Fixture:
    gate = entry.gate_id
    certificate = _certificate_fields(certified=0.5, tolerance=0.5, tail=0.25)
    point_key = "retained_value"
    threshold = 0.5
    inputs: dict[str, Any] = dict(certificate)
    if gate == "PLAN:audit:retained-rho":
        value = _float_value(point, threshold, low=0.1, high=0.8, extreme=math.inf)
        inputs.update(retained_values=(value,), retained_value=value)
        domain = math.isfinite(value) and value >= 0.0
        expected_pass = domain and value <= threshold
        method_name = "audit_retained_rho"
    elif gate == "PLAN:audit:retained-lambda-scale":
        threshold = 2.0
        inputs["max_abs_lambda_logdet"] = threshold
        value = _float_value(point, threshold, low=0.25, high=4.0, extreme=math.inf)
        inputs.update(retained_values=(value,), retained_value=value)
        domain = math.isfinite(value)
        expected_pass = domain and abs(value) <= threshold
        method_name = "audit_retained_lambda_logdet"
    elif gate == "PLAN:audit:retained-x-norm":
        threshold = 0.5
        inputs["max_x_operator_norm"] = threshold
        value = _float_value(point, threshold, low=0.1, high=0.8, extreme=math.inf)
        inputs.update(retained_values=(value,), retained_value=value)
        domain = math.isfinite(value) and value >= 0.0
        expected_pass = domain and value <= threshold
        method_name = "audit_retained_operator_norm"
    elif gate == "PLAN:audit:retained-trace-evidence":
        certificate = _certificate_fields(certified=0.2, tolerance=0.2, tail=0.05)
        inputs = dict(certificate)
        lambda_matrix = np.array([2.0, 3.0])
        perturbation = np.array([0.2, 0.15])
        exact = oracles.exact_power_traces(
            lambda_matrix, perturbation, certificate["order"]
        )
        canonical = exact[0]
        threshold = canonical
        value: Any
        traces: tuple[float, ...] | None
        trace_order = certificate["order"]
        if point.role is PointRole.EXTREME:
            traces = None
            value = math.inf
        else:
            value = _exact_value(point, threshold)
            traces = (value, *exact[1:])
        if atom is not None:
            if atom.index == 0:
                trace_order = certificate["order"] + 1
                value = trace_order
            else:
                traces = None
                value = math.inf
        exact_match = (
            traces is not None
            and trace_order == certificate["order"]
            and traces
            == oracles.exact_power_traces(lambda_matrix, perturbation, trace_order)
        )
        expected_pass = bool(exact_match)
        domain = True
        inputs.update(
            lambda_matrix=lambda_matrix,
            perturbation=perturbation,
            problem_trace_order=trace_order,
            exact_power_traces=traces,
            trace_evidence_value=value,
        )
        point_key = "trace_evidence_value"

        def invoke_trace(data: Mapping[str, Any]) -> _Outcome:
            cert = _certificate_from(data)
            problem = eager.LogDetProblem(
                data["lambda_matrix"],
                data["perturbation"],
                trace_order=data["problem_trace_order"],
                certified_rho=data["certified_rho"],
                exact_power_traces=data["exact_power_traces"],
            )
            report = plan.audit_retained_power_traces((problem,), cert)
            observed = GateSide.ADMITTED if report.passed else GateSide.REFUSED
            production_order_mismatch = problem.trace_order != cert.order
            production_missing = (
                None
                if production_order_mismatch
                else problem.exact_power_traces is None
            )
            oracle_order_mismatch = int(data["problem_trace_order"]) != int(
                data["order"]
            )
            oracle_missing = (
                None if oracle_order_mismatch else data["exact_power_traces"] is None
            )

            def production_item(value: Any, source: str) -> _AtomValue:
                truth = None if value is None else bool(value)
                return _production_atom(
                    value,
                    truth,
                    AtomReducer.NOT_EVALUATED if value is None else AtomReducer.SCALAR,
                    source=source,
                )

            def oracle_item(value: bool | None, source: str) -> _AtomValue:
                return _oracle_atom(
                    value,
                    value,
                    AtomReducer.NOT_EVALUATED if value is None else AtomReducer.SCALAR,
                    source=source,
                )

            actual_rows = _atom_row(
                entry,
                (
                    production_item(
                        production_order_mismatch,
                        "audit problem/certificate retained order comparison",
                    ),
                    production_item(
                        production_missing,
                        "audit retained exact-trace presence branch",
                    ),
                ),
            )
            oracle_rows = _atom_row(
                entry,
                (
                    oracle_item(
                        oracle_order_mismatch,
                        "integer retained order identity",
                    ),
                    oracle_item(
                        oracle_missing,
                        "literal retained trace evidence presence",
                    ),
                ),
            )
            return _Outcome(
                observed_side=observed,
                oracle_side=GateSide.ADMITTED if expected_pass else GateSide.REFUSED,
                returns={
                    "audit_passed": report.passed,
                    "audit_violations": report.violations,
                    "retained_rank": int(
                        np.count_nonzero(np.asarray(data["perturbation"]) != 0.0)
                    ),
                },
                actual_atoms=actual_rows,
                oracle_atoms=oracle_rows,
                direct_calls=("plan.audit_retained_power_traces",),
            )

        return _Fixture(
            inputs,
            point_key,
            threshold,
            np.dtype(np.float64).str,
            "trace_evidence_value",
            "independent retained order and explicit matrix-power traces",
            invoke_trace,
        )
    else:
        raise AssertionError(f"unmapped audit gate {gate}")

    def invoke(data: Mapping[str, Any]) -> _Outcome:
        cert = _certificate_from(data)
        method = getattr(plan, method_name)
        try:
            report = method(data["retained_values"], cert)
        except ValueError as error:
            return _Outcome(
                observed_side=GateSide.REFUSED,
                oracle_side=GateSide.REFUSED if not domain else GateSide.ADMITTED,
                returns={"raised_type": type(error).__name__},
                direct_calls=(f"plan.{method.__name__}",),
            )
        observed = GateSide.ADMITTED if report.passed else GateSide.REFUSED
        return _Outcome(
            observed_side=observed,
            oracle_side=GateSide.ADMITTED if expected_pass else GateSide.REFUSED,
            returns={
                "audit_passed": report.passed,
                "audit_measured_max": report.measured_max,
                "audit_violations": report.violations,
            },
            direct_calls=(f"plan.{method.__name__}",),
        )

    return _Fixture(
        inputs,
        point_key,
        threshold,
        np.dtype(np.float64).str,
        point_key,
        "independent retained-domain and literal violation enumeration",
        invoke,
    )


def _measurement_fixture(
    entry: GateEntry,
    point: ThresholdPoint,
    atom: _AtomSpec | None,
) -> _Fixture:
    gate = entry.gate_id
    reviewed_extreme = _review_extreme_value(point)
    if gate == "PLAN:measurement:x-norm-finite":
        threshold = _X_PERTURBATION_ENTRY_THRESHOLD
        target = (
            _float_value(
                point,
                threshold,
                low=0.25,
                high=_FLOAT64_MAXIMUM,
                extreme=math.inf,
            )
            if reviewed_extreme is None
            else reviewed_extreme
        )
        matrix_path = "dense-full-float64"
        if point.display_value == "last finite float64":
            matrix_path = "compact-float64"
        elif point.display_value == "last finite float32 with non-scalar fixture":
            matrix_path = "dense-diagonal-float32"
        inputs: dict[str, Any] = {
            "lambda_entry": 1.0,
            "perturbation_entry": target,
            "matrix_path": matrix_path,
        }
        if atom is not None:
            inputs.update(
                lambda_entry=1.0,
                perturbation_entry=0.25,
                matrix_path=(
                    "dense-overflow-x-float64"
                    if atom.index < 3
                    else "dense-overflow-norm-float64"
                ),
            )

        def invoke(data: Mapping[str, Any]) -> _Outcome:
            path = str(data["matrix_path"])
            dtype = np.dtype(np.float32 if "float32" in path else np.float64)
            lambda_entry = dtype.type(data["lambda_entry"])
            perturbation_entry = dtype.type(data["perturbation_entry"])
            if path.startswith("compact"):
                lambda_matrix = np.array([lambda_entry], dtype=dtype)
                perturbation = np.array([perturbation_entry], dtype=dtype)
            elif path == "dense-overflow-x-float64":
                lambda_matrix = np.diag(
                    np.array([np.finfo(dtype).tiny, 1.0], dtype=dtype)
                )
                perturbation = np.diag(
                    np.array([np.finfo(dtype).max, 0.2], dtype=dtype)
                )
            elif path == "dense-overflow-norm-float64":
                lambda_matrix = np.eye(2, dtype=dtype)
                perturbation = np.full(
                    (2, 2), dtype.type(0.75) * np.finfo(dtype).max, dtype=dtype
                )
            elif path == "dense-singular-float64":
                lambda_matrix = np.array([[0.0, 0.0], [0.0, lambda_entry]], dtype=dtype)
                perturbation = np.full((2, 2), perturbation_entry, dtype=dtype)
            else:
                lambda_matrix = np.eye(2, dtype=dtype) * lambda_entry
                perturbation = (
                    np.full((2, 2), perturbation_entry, dtype=dtype)
                    if path == "dense-full-float64"
                    else np.diag(
                        np.array(
                            [perturbation_entry, perturbation_entry / dtype.type(2.0)],
                            dtype=dtype,
                        )
                    )
                )

            oracle_x: np.ndarray | None = None
            oracle_norm: float | None = None
            try:
                with np.errstate(
                    divide="raise", invalid="raise", over="raise", under="ignore"
                ):
                    if path == "dense-singular-float64":
                        raise np.linalg.LinAlgError("singular capability cell")
                    if path == "dense-overflow-x-float64":
                        exact_ratio = Decimal.from_float(float(np.finfo(dtype).max)) / Decimal.from_float(
                            float(np.finfo(dtype).tiny)
                        )
                        if exact_ratio <= Decimal.from_float(float(np.finfo(dtype).max)):
                            raise AssertionError("overflow-X fixture did not exceed dtype range")
                        oracle_x = np.array([[math.inf, 0.0], [0.0, 0.2]])
                        oracle_norm = None
                    elif path == "dense-overflow-norm-float64":
                        component = dtype.type(0.75) * np.finfo(dtype).max
                        oracle_x = np.full((2, 2), component, dtype=dtype)
                        exact_norm = Decimal(2) * Decimal.from_float(float(component))
                        oracle_norm = (
                            math.inf
                            if exact_norm > Decimal.from_float(float(np.finfo(dtype).max))
                            else float(exact_norm)
                        )
                    else:
                        ratio = perturbation_entry / lambda_entry
                        if path.startswith("compact"):
                            oracle_x = np.array([ratio], dtype=dtype)
                            oracle_norm = float(abs(ratio))
                        elif path == "dense-full-float64":
                            oracle_x = np.full((2, 2), ratio, dtype=dtype)
                            oracle_norm = float(dtype.type(2.0) * abs(ratio))
                        else:
                            oracle_x = np.diag(
                                np.array(
                                    [
                                        ratio,
                                        perturbation_entry / dtype.type(2.0),
                                    ],
                                    dtype=dtype,
                                )
                            )
                            oracle_norm = float(
                                max(
                                    abs(ratio),
                                    abs(perturbation_entry / dtype.type(2.0)),
                                )
                            )
            except (ArithmeticError, FloatingPointError, np.linalg.LinAlgError):
                oracle_x = None
                oracle_norm = None
            expected_valid = (
                oracle_x is not None
                and bool(np.all(np.isfinite(oracle_x)))
                and oracle_norm is not None
                and math.isfinite(oracle_norm)
            )

            captured_x: list[np.ndarray] = []
            captured_norm: list[float] = []
            original_x = plan._x_matrix
            original_norm = np.linalg.norm

            def x_seam(
                lambda_matrix: np.ndarray, perturbation: np.ndarray
            ) -> np.ndarray:
                result = original_x(lambda_matrix, perturbation)
                captured_x.append(np.asarray(result))
                return result

            def norm_seam(value: Any, *args: Any, **kwargs: Any) -> Any:
                result = original_norm(value, *args, **kwargs)
                captured_norm.append(float(result))
                return result

            with (
                patch.object(plan, "_x_matrix", side_effect=x_seam),
                patch.object(plan.np.linalg, "norm", side_effect=norm_seam),
            ):
                outcome = _validation_outcome(
                    expected_valid=expected_valid,
                    operation=lambda: plan._checked_x_operator_norm(
                        lambda_matrix, perturbation
                    ),
                    refused_exceptions=(ValueError,),
                    return_value=lambda result: {"measured_x_norm": result},
                    direct_calls=("plan._checked_x_operator_norm",),
                )
            x = captured_x[0] if captured_x else None
            production_norm = captured_norm[0] if captured_norm else None
            actual_rows: Mapping[str, _AtomValue] = MappingProxyType({})
            oracle_rows: Mapping[str, _AtomValue] = MappingProxyType({})
            if atom is not None:
                if x is None:
                    raise AssertionError("X atom fixture did not reach the predicate")
                production_x_raw = np.isfinite(x)
                production_all_x = bool(np.all(production_x_raw))
                production_norm_raw = (
                    None if production_norm is None else np.isfinite(production_norm)
                )
                production_norm_finite = (
                    None if production_norm_raw is None else bool(production_norm_raw)
                )
                modeled_x = oracle_x
                if modeled_x is None:
                    raise AssertionError("X atom oracle did not resolve its fixture")
                oracle_x_raw = np.array(
                    [math.isfinite(float(value)) for value in modeled_x.ravel()],
                    dtype=bool,
                ).reshape(modeled_x.shape)
                oracle_all_x = all(bool(value) for value in oracle_x_raw.ravel())
                modeled_norm = None if not oracle_all_x else oracle_norm
                oracle_norm_finite = (
                    None if modeled_norm is None else math.isfinite(modeled_norm)
                )
                actual_rows = _atom_row(
                    entry,
                    (
                        _production_atom(
                            production_all_x,
                            production_all_x,
                            source="checked X np.all finite return",
                        ),
                        *_production_aliases(
                            production_x_raw,
                            production_all_x,
                            AtomReducer.ALL_ELEMENTS,
                            source="checked X np.isfinite array",
                        ),
                        *(
                            (
                                _production_atom(
                                    None,
                                    None,
                                    AtomReducer.NOT_EVALUATED,
                                    source="checked X refusal skipped norm",
                                ),
                                _production_atom(
                                    None,
                                    None,
                                    AtomReducer.NOT_EVALUATED,
                                    source="checked X refusal skipped norm alias",
                                ),
                            )
                            if production_norm_finite is None
                            else _production_aliases(
                                production_norm_raw,
                                production_norm_finite,
                                source="checked X retained norm np.isfinite",
                            )
                        ),
                    ),
                )
                oracle_rows = _atom_row(
                    entry,
                    (
                        _oracle_atom(
                            oracle_all_x,
                            oracle_all_x,
                            source="elementwise math.isfinite conjunction",
                        ),
                        *_oracle_aliases(
                            oracle_x_raw,
                            oracle_all_x,
                            AtomReducer.ALL_ELEMENTS,
                            source="elementwise math.isfinite array",
                        ),
                        *(
                            (
                                _oracle_atom(
                                    None,
                                    None,
                                    AtomReducer.NOT_EVALUATED,
                                    source="oracle X refusal skipped norm",
                                ),
                                _oracle_atom(
                                    None,
                                    None,
                                    AtomReducer.NOT_EVALUATED,
                                    source="oracle X refusal skipped norm alias",
                                ),
                            )
                            if oracle_norm_finite is None
                            else _oracle_aliases(
                                oracle_norm_finite,
                                oracle_norm_finite,
                                source="analytic structured-X norm finiteness",
                            )
                        ),
                    ),
                )
            returns = dict(outcome.returns)
            returns.update(
                resolved_lambda_matrix=lambda_matrix,
                resolved_perturbation=perturbation,
                resolved_x=x,
                resolved_x_norm=production_norm,
            )
            return _Outcome(
                observed_side=outcome.observed_side,
                oracle_side=outcome.oracle_side,
                returns=returns,
                actual_atoms=actual_rows,
                oracle_atoms=oracle_rows,
                direct_calls=outcome.direct_calls,
            )

        return _Fixture(
            inputs,
            "perturbation_entry",
            threshold,
            np.dtype(np.float64).str,
            "perturbation_entry",
            "analytic compact/dense solve and abs(X) norm",
            invoke,
        )

    threshold = 0.0
    target = (
        _zero_value(
            point,
            low=-1.0,
            high=_FLOAT64_MAXIMUM,
            extreme=math.inf,
        )
        if reviewed_extreme is None
        else reviewed_extreme
    )
    matrix_path = "compact-float64"
    if point.display_value == "last finite float32 with non-scalar fixture":
        matrix_path = "dense-float32"
    inputs = {
        "lambda_entry": target,
        "matrix_path": matrix_path,
    }
    if atom is not None:
        inputs.update(lambda_entry=2.0, matrix_path="compact-float64")

    def invoke_lambda(data: Mapping[str, Any]) -> _Outcome:
        path = str(data["matrix_path"])
        dtype = np.dtype(np.float32 if "float32" in path else np.float64)
        lambda_entry = dtype.type(data["lambda_entry"])
        if path.startswith("compact"):
            lambda_matrix = np.array([lambda_entry], dtype=dtype)
        elif path == "dense-indefinite-float64":
            lambda_matrix = np.diag(np.array([lambda_entry, -1.0], dtype=dtype))
        elif path == "dense-subnormal-float64":
            lambda_matrix = np.diag(
                np.array([_FLOAT64_MIN_SUBNORMAL, lambda_entry], dtype=dtype)
            )
        else:
            lambda_matrix = np.diag(np.array([lambda_entry, 1.0], dtype=dtype))
        oracle_valid = (
            path != "dense-indefinite-float64"
            and math.isfinite(float(lambda_entry))
            and float(lambda_entry) > 0.0
        )
        outcome = _validation_outcome(
            expected_valid=oracle_valid,
            operation=lambda: plan._checked_lambda_logdet_scale(lambda_matrix),
            refused_exceptions=(ValueError,),
            return_value=lambda result: {"measured_lambda_scale": result},
            direct_calls=("plan._checked_lambda_logdet_scale",),
        )
        actual_scale = outcome.returns.get("measured_lambda_scale")
        actual_rows: Mapping[str, _AtomValue] = MappingProxyType({})
        oracle_rows: Mapping[str, _AtomValue] = MappingProxyType({})
        if atom is not None:
            if actual_scale is None:
                raise AssertionError(
                    "lambda-scale atom fixture did not reach predicate"
                )
            production_raw = np.isfinite(actual_scale)
            production_finite = bool(production_raw)
            oracle_scale = abs(
                math.fsum(
                    math.log(float(value)) for value in np.ravel(lambda_matrix)
                )
            )
            oracle_finite = math.isfinite(oracle_scale)
            actual_rows = _atom_row(
                entry,
                _production_aliases(
                    production_raw,
                    production_finite,
                    source="retained lambda-logdet scale np.isfinite",
                ),
            )
            oracle_rows = _atom_row(
                entry,
                _oracle_aliases(
                    oracle_finite,
                    oracle_finite,
                    source="analytic diagonal logdet math.isfinite",
                ),
            )
        returns = dict(outcome.returns)
        returns.update(
            resolved_lambda_matrix=lambda_matrix,
            resolved_lambda_scale=actual_scale,
        )
        return _Outcome(
            observed_side=outcome.observed_side,
            oracle_side=outcome.oracle_side,
            returns=returns,
            actual_atoms=actual_rows,
            oracle_atoms=oracle_rows,
            direct_calls=outcome.direct_calls,
        )

    return _Fixture(
        inputs,
        "lambda_entry",
        threshold,
        np.dtype(np.float64).str,
        "lambda_entry",
        "independent analytic diagonal logdet scale",
        invoke_lambda,
    )


def _factory_certificate_fixture(
    entry: GateEntry,
    point: ThresholdPoint,
    atom: _AtomSpec | None,
) -> _Fixture:
    gate = entry.gate_id
    fields = _certificate_fields(certified=0.5, tolerance=0.5, tail=0.25)
    inputs: dict[str, Any] = dict(fields)
    inputs.update(
        lambda_matrix=np.array([2.0, 3.0]),
        perturbation=np.array([0.2, -0.3]),
        problem_trace_order=fields["order"],
    )
    if gate == "PLAN:factory-certificate:order-and-rank":
        threshold: float | int = fields["multiplicity"]
        target = _integer_value(point, int(threshold), low=0, high=5, extreme=100)
        if atom is not None:
            target = int(threshold) + 1
        lambda_matrix, perturbation = _compact_rank_problem(target)
        inputs["lambda_matrix"] = lambda_matrix
        inputs["perturbation"] = perturbation
        point_key = "required_multiplicity"
        expected_valid = target <= threshold
    elif gate == "PLAN:factory-certificate:lambda-scale":
        threshold = 2.0
        target = _float_value(point, threshold, low=0.25, high=4.0, extreme=10.0)
        if atom is not None:
            target = _ORIGINAL_NEXTAFTER(threshold, math.inf)
        inputs["max_abs_lambda_logdet"] = threshold
        inputs["lambda_matrix"] = _lambda_scale_matrix(target)
        inputs["perturbation"] = np.array([0.0])
        point_key = "actual_base_scale"
        expected_valid = math.isfinite(target) and 0.0 <= target <= threshold
    elif gate == "PLAN:factory-certificate:x-norm":
        threshold = 0.5
        target = _float_value(point, threshold, low=0.1, high=0.8, extreme=0.95)
        if atom is not None:
            target = _ORIGINAL_NEXTAFTER(threshold, math.inf)
        inputs["max_x_operator_norm"] = threshold
        inputs["lambda_matrix"] = np.array([1.0])
        inputs["perturbation"] = _x_norm_perturbation(target)
        inputs["certified_rho"] = 0.99
        inputs["order"] = oracles.smallest_trace_order(
            inputs["certified_rho"], inputs["tail_tolerance"], inputs["multiplicity"]
        )
        inputs["problem_trace_order"] = inputs["order"]
        point_key = "actual_x_norm"
        expected_valid = math.isfinite(target) and abs(target) <= threshold
    elif gate == "PLAN:factory-certificate:strict-rho":
        threshold = 0.5
        target = _float_value(point, threshold, low=0.1, high=0.8, extreme=1.0)
        # The strict-rho call is the owned decision.  An X-norm certificate
        # would reject the same perturbation one line earlier and hide it.
        inputs["max_x_operator_norm"] = None
        inputs["lambda_matrix"] = np.eye(2)
        inputs["perturbation"] = np.diag([target, -0.1])
        inputs["rho_target"] = target
        point_key = "rho_target"
        expected_valid = math.isfinite(target) and target < 1.0 and target <= threshold
    else:
        raise AssertionError(f"unmapped factory-certificate gate {gate}")

    def invoke(data: Mapping[str, Any]) -> _Outcome:
        cert = _certificate_from(data)
        try:
            problem = eager.LogDetProblem(
                data["lambda_matrix"],
                data["perturbation"],
                trace_order=data["problem_trace_order"],
                certified_rho=data["certified_rho"],
            )
        except ValueError as error:
            return _Outcome(
                observed_side=GateSide.REFUSED,
                oracle_side=GateSide.REFUSED,
                returns={"raised_type": type(error).__name__},
                direct_calls=("eager.LogDetProblem",),
            )
        captured: dict[str, Any] = {}
        original_rank = plan._algebraic_rank_bound
        original_scale = plan._checked_lambda_logdet_scale
        original_norm = plan._checked_x_operator_norm

        def rank_seam(*args: Any, **kwargs: Any) -> int:
            result = original_rank(*args, **kwargs)
            captured["required_multiplicity"] = result
            return result

        def scale_seam(*args: Any, **kwargs: Any) -> float:
            result = original_scale(*args, **kwargs)
            captured["actual_base_scale"] = result
            return result

        def norm_seam(*args: Any, **kwargs: Any) -> float:
            result = original_norm(*args, **kwargs)
            captured["actual_x_norm"] = result
            return result

        with (
            patch.object(plan, "_algebraic_rank_bound", side_effect=rank_seam),
            patch.object(plan, "_checked_lambda_logdet_scale", side_effect=scale_seam),
            patch.object(plan, "_checked_x_operator_norm", side_effect=norm_seam),
        ):
            outcome = _validation_outcome(
                expected_valid=expected_valid,
                operation=lambda: plan._validate_plan_certificate(problem, cert),
                refused_exceptions=(ValueError,),
                return_value=lambda _: {"validation_return": "accepted"},
                direct_calls=("plan._validate_plan_certificate",),
            )
        returns = dict(outcome.returns)
        returns.update(captured)
        actual_rows: Mapping[str, _AtomValue] = MappingProxyType({})
        oracle_rows: Mapping[str, _AtomValue] = MappingProxyType({})
        if gate == "PLAN:factory-certificate:order-and-rank":
            production_comparison = (
                None
                if "required_multiplicity" not in captured
                else cert.multiplicity < captured["required_multiplicity"]
            )
            oracle_comparison = (
                None
                if int(data["problem_trace_order"]) != int(data["order"])
                else int(data["multiplicity"])
                < int(np.count_nonzero(np.asarray(data["perturbation"]) != 0.0))
            )
            actual_rows = _atom_row(
                entry,
                (
                    _production_atom(
                        production_comparison,
                        production_comparison,
                        AtomReducer.NOT_EVALUATED
                        if production_comparison is None
                        else AtomReducer.SCALAR,
                        source="retained required-multiplicity comparison",
                    ),
                ),
            )
            oracle_rows = _atom_row(
                entry,
                (
                    _oracle_atom(
                        oracle_comparison,
                        oracle_comparison,
                        AtomReducer.NOT_EVALUATED
                        if oracle_comparison is None
                        else AtomReducer.SCALAR,
                        source="integer algebraic-rank bound comparison",
                    ),
                ),
            )
        elif gate == "PLAN:factory-certificate:lambda-scale":
            production_comparison = (
                captured["actual_base_scale"] > cert.max_abs_lambda_logdet
            )
            oracle_scale = abs(
                sum(math.log(float(value)) for value in np.ravel(data["lambda_matrix"]))
            )
            oracle_comparison = Decimal.from_float(oracle_scale) > Decimal.from_float(
                float(data["max_abs_lambda_logdet"])
            )
            actual_rows = _atom_row(
                entry,
                (
                    _production_atom(
                        production_comparison,
                        production_comparison,
                        source="retained base-scale certificate comparison",
                    ),
                ),
            )
            oracle_rows = _atom_row(
                entry,
                (
                    _oracle_atom(
                        oracle_comparison,
                        oracle_comparison,
                        source="analytic diagonal-logdet scale comparison",
                    ),
                ),
            )
        elif gate == "PLAN:factory-certificate:x-norm":
            production_comparison = captured["actual_x_norm"] > cert.max_x_operator_norm
            lambda_values = np.asarray(data["lambda_matrix"])
            perturbation_values = np.asarray(data["perturbation"])
            if lambda_values.ndim != 1 or perturbation_values.ndim != 1:
                raise AssertionError("X-norm fixture must retain its analytic diagonal form")
            oracle_norm = max(
                (
                    abs(float(perturbation_value) / float(lambda_value))
                    for lambda_value, perturbation_value in zip(
                        lambda_values, perturbation_values, strict=True
                    )
                ),
                default=0.0,
            )
            oracle_comparison = Decimal.from_float(oracle_norm) > Decimal.from_float(
                float(data["max_x_operator_norm"])
            )
            actual_rows = _atom_row(
                entry,
                (
                    _production_atom(
                        production_comparison,
                        production_comparison,
                        source="retained abs(X)-norm certificate comparison",
                    ),
                ),
            )
            oracle_rows = _atom_row(
                entry,
                (
                    _oracle_atom(
                        oracle_comparison,
                        oracle_comparison,
                        source="independent solve and singular-value comparison",
                    ),
                ),
            )
        return _Outcome(
            observed_side=outcome.observed_side,
            oracle_side=outcome.oracle_side,
            returns=returns,
            actual_atoms=actual_rows,
            oracle_atoms=oracle_rows,
            payload_checks=outcome.payload_checks,
            direct_calls=outcome.direct_calls,
        )

    return _Fixture(
        inputs,
        point_key,
        threshold,
        None if isinstance(threshold, int) else np.dtype(np.float64).str,
        "perturbation"
        if gate.endswith("order-and-rank")
        else (
            "lambda_matrix"
            if gate.endswith("lambda-scale")
            else ("perturbation" if gate.endswith("x-norm") else "rho_target")
        ),
        "independent rank, logdet, abs(X)-norm, and spectral-radius certificate",
        invoke,
    )


def _proof_fixture(
    entry: GateEntry,
    point: ThresholdPoint,
    atom: _AtomSpec | None,
) -> _Fixture:
    gate = entry.gate_id
    if gate == "PLAN:canonical-probes:runtime-finite":
        runtime_dtype = np.dtype(np.float32)
        threshold = float(np.finfo(runtime_dtype).max)
        value = _float_value(
            point,
            threshold,
            low=1.7,
            high=threshold * 1.5,
            extreme=threshold * 4.0,
        )
        inputs = {
            "probe_scalar": value,
            "probe_dtype": np.dtype(np.float64).str,
            "runtime_dtype": runtime_dtype.str,
        }

        def invoke(data: Mapping[str, Any]) -> _Outcome:
            local_runtime_dtype = np.dtype(data["runtime_dtype"])
            source_values = np.asarray(
                [[data["probe_scalar"]]], dtype=np.dtype(data["probe_dtype"])
            )
            probes = eager.FrozenProbes(source_values)
            with np.errstate(over="ignore", invalid="ignore"):
                oracle_converted = np.asarray(source_values, dtype=local_runtime_dtype)
            expected_valid = all(
                math.isfinite(float(item)) for item in oracle_converted.ravel()
            )
            outcome = _validation_outcome(
                expected_valid=expected_valid,
                operation=lambda: plan._canonical_runtime_probes(
                    probes, np.dtype(data["runtime_dtype"])
                ),
                refused_exceptions=(ValueError,),
                return_value=lambda result: {
                    "canonical_probe_values": result.values
                },
                direct_calls=("plan._canonical_runtime_probes",),
            )
            production_raw = np.isfinite(oracle_converted)
            production_all = bool(np.all(production_raw))
            oracle_raw = np.array(
                [math.isfinite(float(item)) for item in oracle_converted.ravel()],
                dtype=bool,
            ).reshape(oracle_converted.shape)
            oracle_all = all(bool(value) for value in oracle_raw.ravel())
            actual_rows = _atom_row(
                entry,
                (
                    _production_atom(
                        production_all,
                        production_all,
                        source="canonical probe np.all return",
                    ),
                    *_production_aliases(
                        production_raw,
                        production_all,
                        AtomReducer.ALL_ELEMENTS,
                        source="canonical probe converted np.isfinite array",
                    ),
                ),
            )
            oracle_rows = _atom_row(
                entry,
                (
                    _oracle_atom(
                        oracle_all,
                        oracle_all,
                        source="elementwise converted-probe conjunction",
                    ),
                    *_oracle_aliases(
                        oracle_raw,
                        oracle_all,
                        AtomReducer.ALL_ELEMENTS,
                        source="elementwise converted-probe math.isfinite",
                    ),
                ),
            )
            returns = dict(outcome.returns)
            returns["converted_values"] = oracle_converted
            return _Outcome(
                observed_side=outcome.observed_side,
                oracle_side=outcome.oracle_side,
                returns=returns,
                actual_atoms=actual_rows,
                oracle_atoms=oracle_rows,
                direct_calls=outcome.direct_calls,
            )

        return _Fixture(
            inputs,
            "probe_scalar",
            threshold,
            np.dtype(np.float64).str,
            "probe_scalar",
            "literal NumPy runtime-dtype cast and scalar finiteness",
            invoke,
        )

    if gate == "PLAN:outward-arithmetic:positive-underflow":
        threshold = 0.0
        value = _zero_value(point, low=-0.5, high=8.0, extreme=math.inf)
        if atom is not None:
            value = 0.0 if atom.index == 0 else math.inf
        magnitude = abs(value) if math.isfinite(value) else value
        inputs = {
            "proof_value": value,
            "proof_magnitude": magnitude,
            "product_right": 0.5,
            "quotient_denominator": 2,
        }

        def invoke(data: Mapping[str, Any]) -> _Outcome:
            direct = plan._outward_nonnegative(data["proof_value"])
            product = plan._outward_product(
                data["proof_magnitude"], data["product_right"]
            )
            summed = plan._outward_sum(data["proof_magnitude"], data["product_right"])
            quotient = plan._outward_quotient(
                data["proof_magnitude"], data["quotient_denominator"]
            )
            production_zero = data["proof_value"] == 0.0
            production_finite_raw = (
                None if production_zero else np.isfinite(data["proof_value"])
            )
            production_finite = (
                None if production_finite_raw is None else bool(production_finite_raw)
            )
            oracle_zero = Decimal.from_float(float(data["proof_value"])) == Decimal(0)
            oracle_finite = (
                None if oracle_zero else math.isfinite(float(data["proof_value"]))
            )
            direct_oracle = (
                data["proof_value"]
                if oracle_zero or oracle_finite is False
                else _ORIGINAL_NEXTAFTER(data["proof_value"], math.inf)
            )
            product_oracle = oracles.outward_nonnegative_oracle(
                oracles.exact_product(data["proof_magnitude"], data["product_right"])
            )
            if data["proof_magnitude"] == 0.0 or data["product_right"] == 0.0:
                product_oracle = 0.0
            sum_oracle = oracles.outward_nonnegative_oracle(
                oracles.exact_sum(data["proof_magnitude"], data["product_right"])
            )
            if data["proof_magnitude"] == 0.0:
                sum_oracle = data["product_right"]
            elif data["product_right"] == 0.0:
                sum_oracle = data["proof_magnitude"]
            quotient_oracle = oracles.outward_nonnegative_oracle(
                oracles.exact_quotient(
                    data["proof_magnitude"], data["quotient_denominator"]
                )
            )
            if data["proof_magnitude"] == 0.0:
                quotient_oracle = 0.0
            preserved = oracle_zero or oracle_finite is False
            production_preserved = _raw_equal(direct, data["proof_value"])
            actual_rows = _atom_row(
                entry,
                (
                    _production_atom(
                        production_zero,
                        production_zero,
                        source="outward input zero comparison",
                    ),
                    *(
                        (
                            _production_atom(
                                None,
                                None,
                                AtomReducer.NOT_EVALUATED,
                                source="outward zero short-circuit",
                            ),
                            _production_atom(
                                None,
                                None,
                                AtomReducer.NOT_EVALUATED,
                                source="outward zero short-circuit alias",
                            ),
                        )
                        if production_finite is None
                        else _production_aliases(
                            production_finite_raw,
                            production_finite,
                            source="outward input np.isfinite",
                        )
                    ),
                ),
            )
            oracle_rows = _atom_row(
                entry,
                (
                    _oracle_atom(
                        oracle_zero,
                        oracle_zero,
                        source="Decimal outward zero identity",
                    ),
                    *(
                        (
                            _oracle_atom(
                                None,
                                None,
                                AtomReducer.NOT_EVALUATED,
                                source="oracle outward zero short-circuit",
                            ),
                            _oracle_atom(
                                None,
                                None,
                                AtomReducer.NOT_EVALUATED,
                                source="oracle outward zero short-circuit alias",
                            ),
                        )
                        if oracle_finite is None
                        else _oracle_aliases(
                            oracle_finite,
                            oracle_finite,
                            source="outward input math.isfinite",
                        )
                    ),
                ),
            )
            return _Outcome(
                observed_side=(
                    GateSide.ADMITTED if production_preserved else GateSide.REFUSED
                ),
                oracle_side=GateSide.ADMITTED if preserved else GateSide.REFUSED,
                returns={
                    "outward_value": direct,
                    "outward_product": product,
                    "outward_sum": summed,
                    "outward_quotient": quotient,
                },
                actual_atoms=actual_rows,
                oracle_atoms=oracle_rows,
                payload_checks=(
                    (
                        "directed rounding value",
                        direct,
                        direct_oracle,
                        _raw_equal(direct, direct_oracle),
                    ),
                    (
                        "directed rounding product",
                        product,
                        product_oracle,
                        _raw_equal(product, product_oracle),
                    ),
                    (
                        "directed rounding sum",
                        summed,
                        sum_oracle,
                        _raw_equal(summed, sum_oracle),
                    ),
                    (
                        "directed rounding quotient",
                        quotient,
                        quotient_oracle,
                        _raw_equal(quotient, quotient_oracle),
                    ),
                ),
                direct_calls=(
                    "plan._outward_nonnegative",
                    "plan._outward_product",
                    "plan._outward_sum",
                    "plan._outward_quotient",
                ),
            )

        return _Fixture(
            inputs,
            "proof_value",
            threshold,
            np.dtype(np.float64).str,
            "proof_value",
            "Decimal/Fraction directed-rounding proof",
            invoke,
        )

    if gate == "PLAN:frozen:probe-energy-range":
        runtime_dtype = np.dtype(np.float64)
        maximum = float(np.finfo(runtime_dtype).max)
        threshold = math.sqrt(maximum)
        value = _float_value(
            point,
            threshold,
            low=1.7,
            high=threshold * 1.1,
            extreme=math.inf,
        )
        probe_count = 1
        if atom is not None:
            runtime_dtype = np.dtype(np.float32)
            maximum = float(np.finfo(runtime_dtype).max)
            threshold = math.sqrt(maximum)
            value = math.sqrt(0.6 * maximum)
            probe_count = 2
        inputs = {
            "probe_component": value,
            "probe_count": probe_count,
            "runtime_dtype": runtime_dtype.str,
        }

        def invoke(data: Mapping[str, Any]) -> _Outcome:
            original = plan._outward_nonnegative
            component = abs(float(data["probe_component"]))
            count = int(data["probe_count"])
            local_dtype = np.dtype(data["runtime_dtype"])
            local_maximum = float(np.finfo(local_dtype).max)
            vectors = np.full((count, 1), component, dtype=np.float64)
            final_outward_call = 2 * count + 1
            call_count = 0
            captured_total: list[float] = []

            def outward_seam(outward_input: float) -> float:
                nonlocal call_count
                call_count += 1
                result = original(outward_input)
                if call_count == final_outward_call:
                    captured_total.append(result)
                return result

            if math.isfinite(component):
                exact_square = Decimal.from_float(component) ** 2
                oracle_energy = oracles.outward_nonnegative_oracle(exact_square)
                oracle_energy = _literal_outward_nonnegative(oracle_energy)
                if math.isfinite(oracle_energy):
                    exact_total = Decimal.from_float(oracle_energy) * Decimal(count)
                    computed_oracle_total = oracles.outward_nonnegative_oracle(
                        exact_total
                    )
                else:
                    computed_oracle_total = math.inf
            else:
                oracle_energy = math.inf
                computed_oracle_total = math.inf
            oracle_total = computed_oracle_total
            expected_valid = (
                count > 0
                and math.isfinite(component)
                and component <= math.sqrt(local_maximum)
                and math.isfinite(oracle_energy)
                and oracle_energy <= local_maximum
                and math.isfinite(oracle_total)
                and oracle_total <= local_maximum
            )
            with patch.object(plan, "_outward_nonnegative", side_effect=outward_seam):
                outcome = _validation_outcome(
                    expected_valid=expected_valid,
                    operation=lambda: plan._frozen_probe_energy_bounds(
                        vectors, local_dtype
                    ),
                    refused_exceptions=(ValueError,),
                    return_value=lambda result: {
                        "returned_total_energy": result[0],
                        "returned_maximum_norm": result[1],
                    },
                    direct_calls=("plan._frozen_probe_energy_bounds",),
                )
            total = captured_total[0] if captured_total else math.inf
            production_finite_raw = np.isfinite(total)
            production_finite = bool(production_finite_raw)
            production_exceeds = (
                None if not production_finite else bool(total > local_maximum)
            )
            oracle_finite = math.isfinite(oracle_total)
            oracle_exceeds = (
                None
                if not oracle_finite
                else Decimal.from_float(oracle_total)
                > Decimal.from_float(local_maximum)
            )
            actual_rows = _atom_row(
                entry,
                (
                    *_production_aliases(
                        production_finite_raw,
                        production_finite,
                        source="retained total energy np.isfinite",
                    ),
                    _production_atom(
                        production_exceeds,
                        production_exceeds,
                        AtomReducer.NOT_EVALUATED
                        if production_exceeds is None
                        else AtomReducer.SCALAR,
                        source="retained total energy maximum comparison",
                    ),
                ),
            )
            oracle_rows = _atom_row(
                entry,
                (
                    *_oracle_aliases(
                        oracle_finite,
                        oracle_finite,
                        source="Decimal total-energy finiteness",
                    ),
                    _oracle_atom(
                        oracle_exceeds,
                        oracle_exceeds,
                        AtomReducer.NOT_EVALUATED
                        if oracle_exceeds is None
                        else AtomReducer.SCALAR,
                        source="Decimal total-energy maximum comparison",
                    ),
                ),
            )
            returns = dict(outcome.returns)
            returns["resolved_total_energy"] = total
            return _Outcome(
                observed_side=outcome.observed_side,
                oracle_side=outcome.oracle_side,
                returns=returns,
                actual_atoms=actual_rows,
                oracle_atoms=oracle_rows,
                direct_calls=outcome.direct_calls,
            )

        return _Fixture(
            inputs,
            "probe_component",
            threshold,
            np.dtype(np.float64).str,
            "probe_component",
            "Decimal sum of represented probe squares against dtype maximum",
            invoke,
        )

    if gate in {"PLAN:runtime-range:product", "PLAN:runtime-range:sum"}:
        maximum = 10.0
        right = 2.0 if gate.endswith("product") else 0.5
        threshold = maximum / right if gate.endswith("product") else maximum - right
        left = _float_value(
            point,
            threshold,
            low=0.0,
            high=maximum,
            extreme=math.inf,
        )
        if atom is not None:
            left, right = 1.0, 2.0
            if atom.index < 2:
                left = math.inf
            elif atom.index < 4:
                right = math.inf
            elif gate.endswith("product") and atom.index == 4:
                left = _ORIGINAL_NEXTAFTER(maximum / right, math.inf)
            elif gate.endswith("sum") and atom.index == 4:
                right = _ORIGINAL_NEXTAFTER(maximum, math.inf)
            elif gate.endswith("sum") and atom.index == 5:
                left = _ORIGINAL_NEXTAFTER(maximum - right, math.inf)
            elif atom.index in ({5, 6} if gate.endswith("product") else {6, 7}):
                maximum = _FLOAT64_MAXIMUM
                if gate.endswith("product"):
                    left = _FLOAT64_MAXIMUM
                    right = 1.0
                else:
                    left = _ORIGINAL_NEXTAFTER(_FLOAT64_MAXIMUM, 0.0)
                    right = _FLOAT64_MAXIMUM - left
            else:
                left = maximum / right if gate.endswith("product") else maximum - right
        inputs = {
            "left": left,
            "right": right,
            "maximum": maximum,
            "runtime_dtype": np.dtype(np.float64).str,
        }

        def invoke(data: Mapping[str, Any]) -> _Outcome:
            original_product = plan._outward_product
            original_sum = plan._outward_sum
            captured: list[float] = []

            def product_seam(left_value: float, right_value: float) -> float:
                result = original_product(left_value, right_value)
                captured.append(result)
                return result

            def sum_seam(left_value: float, right_value: float) -> float:
                result = original_sum(left_value, right_value)
                captured.append(result)
                return result

            left_value = data["left"]
            right_value = data["right"]
            product_zero_short = gate.endswith("product") and (
                left_value == 0.0 or right_value == 0.0
            )
            production_left_raw = (
                None if product_zero_short else np.isfinite(left_value)
            )
            production_left = (
                None if production_left_raw is None else bool(production_left_raw)
            )
            production_right_raw = (
                None if production_left is not True else np.isfinite(right_value)
            )
            production_right = (
                None if production_right_raw is None else bool(production_right_raw)
            )
            if gate.endswith("product"):
                production_first_compare = (
                    None
                    if production_right is not True
                    else bool(left_value > data["maximum"] / right_value)
                )
                precondition = product_zero_short or (
                    production_left is True
                    and production_right is True
                    and production_first_compare is False
                )
            else:
                production_right_exceeds = (
                    None
                    if production_right is not True
                    else bool(right_value > data["maximum"])
                )
                production_left_exceeds = (
                    None
                    if production_right_exceeds is not False
                    else bool(left_value > data["maximum"] - right_value)
                )
                precondition = (
                    production_left is True
                    and production_right is True
                    and production_right_exceeds is False
                    and production_left_exceeds is False
                )
            expected_result = (
                0.0
                if product_zero_short
                else oracles.outward_nonnegative_oracle(
                    oracles.exact_product(left_value, right_value)
                    if gate.endswith("product")
                    else oracles.exact_sum(left_value, right_value)
                )
                if precondition
                else math.inf
            )
            expected_valid = (
                precondition
                and math.isfinite(expected_result)
                and expected_result <= data["maximum"]
            )
            with (
                patch.object(plan, "_outward_product", side_effect=product_seam),
                patch.object(plan, "_outward_sum", side_effect=sum_seam),
            ):
                if gate.endswith("product"):
                    operation = lambda: plan._runtime_range_product(
                        left_value,
                        right_value,
                        data["maximum"],
                        np.dtype(data["runtime_dtype"]),
                        "direct product",
                    )
                else:
                    operation = lambda: plan._runtime_range_sum(
                        left_value,
                        right_value,
                        data["maximum"],
                        np.dtype(data["runtime_dtype"]),
                        "direct sum",
                    )
                outcome = _validation_outcome(
                    expected_valid=expected_valid,
                    operation=operation,
                    refused_exceptions=(ValueError,),
                    return_value=lambda result: {"range_result": result},
                    direct_calls=(
                        "plan._runtime_range_product"
                        if gate.endswith("product")
                        else "plan._runtime_range_sum",
                    ),
                )
            result_value: float | None = captured[0] if captured else None
            production_result_raw = (
                None if result_value is None else np.isfinite(result_value)
            )
            production_result_finite = (
                None if production_result_raw is None else bool(production_result_raw)
            )
            production_result_exceeds = (
                None
                if production_result_finite is not True
                else bool(result_value > data["maximum"])
            )

            oracle_left = (
                None if product_zero_short else math.isfinite(float(left_value))
            )
            oracle_right = (
                None if oracle_left is not True else math.isfinite(float(right_value))
            )
            maximum_decimal = Decimal.from_float(float(data["maximum"]))
            left_decimal = Decimal.from_float(float(left_value))
            right_decimal = Decimal.from_float(float(right_value))
            maximum_is_nan = math.isnan(float(data["maximum"]))
            with localcontext() as context:
                context.prec = 2500
                if gate.endswith("product"):
                    oracle_first_compare = (
                        None
                        if oracle_right is not True
                        else False
                        if maximum_is_nan
                        else left_decimal > maximum_decimal / right_decimal
                    )
                else:
                    oracle_right_exceeds = (
                        None
                        if oracle_right is not True
                        else False
                        if maximum_is_nan
                        else right_decimal > maximum_decimal
                    )
                    oracle_left_exceeds = (
                        None
                        if oracle_right_exceeds is not False
                        else False
                        if maximum_is_nan
                        else left_decimal > maximum_decimal - right_decimal
                    )
            oracle_result = (
                None
                if not precondition or product_zero_short
                else expected_result
            )
            oracle_result_finite = (
                None if oracle_result is None else math.isfinite(oracle_result)
            )
            oracle_result_exceeds = (
                None
                if oracle_result_finite is not True
                else False
                if maximum_is_nan
                else Decimal.from_float(oracle_result) > maximum_decimal
            )

            def production_item(value: Any, source: str) -> _AtomValue:
                truth = None if value is None else bool(value)
                return _production_atom(
                    value,
                    truth,
                    AtomReducer.NOT_EVALUATED if value is None else AtomReducer.SCALAR,
                    source=source,
                )

            def oracle_item(value: bool | None, source: str) -> _AtomValue:
                return _oracle_atom(
                    value,
                    value,
                    AtomReducer.NOT_EVALUATED if value is None else AtomReducer.SCALAR,
                    source=source,
                )

            if gate.endswith("product"):
                actual_rows = _atom_row(
                    entry,
                    (
                        production_item(
                            production_left_raw,
                            "product left np.isfinite",
                        ),
                        production_item(
                            production_left_raw,
                            "product left np.isfinite alias",
                        ),
                        production_item(
                            production_right_raw,
                            "product right np.isfinite",
                        ),
                        production_item(
                            production_right_raw,
                            "product right np.isfinite alias",
                        ),
                        production_item(
                            production_first_compare,
                            "product pre-overflow comparison",
                        ),
                        production_item(
                            production_result_raw,
                            "product retained result np.isfinite",
                        ),
                        production_item(
                            production_result_raw,
                            "product retained result np.isfinite alias",
                        ),
                        production_item(
                            production_result_exceeds,
                            "product retained result maximum comparison",
                        ),
                    ),
                )
                oracle_rows = _atom_row(
                    entry,
                    (
                        oracle_item(oracle_left, "product left math.isfinite"),
                        oracle_item(oracle_left, "product left math.isfinite alias"),
                        oracle_item(oracle_right, "product right math.isfinite"),
                        oracle_item(oracle_right, "product right math.isfinite alias"),
                        oracle_item(
                            oracle_first_compare,
                            "Decimal product pre-overflow comparison",
                        ),
                        oracle_item(
                            oracle_result_finite,
                            "exact outward product finiteness",
                        ),
                        oracle_item(
                            oracle_result_finite,
                            "exact outward product finiteness alias",
                        ),
                        oracle_item(
                            oracle_result_exceeds,
                            "Decimal outward product maximum comparison",
                        ),
                    ),
                )
            else:
                actual_rows = _atom_row(
                    entry,
                    (
                        production_item(production_left_raw, "sum left np.isfinite"),
                        production_item(
                            production_left_raw, "sum left np.isfinite alias"
                        ),
                        production_item(production_right_raw, "sum right np.isfinite"),
                        production_item(
                            production_right_raw, "sum right np.isfinite alias"
                        ),
                        production_item(
                            production_right_exceeds, "sum right maximum comparison"
                        ),
                        production_item(
                            production_left_exceeds,
                            "sum left remaining-range comparison",
                        ),
                        production_item(
                            production_result_raw,
                            "sum retained result np.isfinite",
                        ),
                        production_item(
                            production_result_raw,
                            "sum retained result np.isfinite alias",
                        ),
                        production_item(
                            production_result_exceeds,
                            "sum retained result maximum comparison",
                        ),
                    ),
                )
                oracle_rows = _atom_row(
                    entry,
                    (
                        oracle_item(oracle_left, "sum left math.isfinite"),
                        oracle_item(oracle_left, "sum left math.isfinite alias"),
                        oracle_item(oracle_right, "sum right math.isfinite"),
                        oracle_item(oracle_right, "sum right math.isfinite alias"),
                        oracle_item(
                            oracle_right_exceeds,
                            "Decimal sum right maximum comparison",
                        ),
                        oracle_item(
                            oracle_left_exceeds,
                            "Decimal sum remaining-range comparison",
                        ),
                        oracle_item(
                            oracle_result_finite, "exact outward sum finiteness"
                        ),
                        oracle_item(
                            oracle_result_finite,
                            "exact outward sum finiteness alias",
                        ),
                        oracle_item(
                            oracle_result_exceeds,
                            "Decimal outward sum maximum comparison",
                        ),
                    ),
                )
            returns = dict(outcome.returns)
            returns["resolved_range_result"] = result_value
            return _Outcome(
                observed_side=outcome.observed_side,
                oracle_side=outcome.oracle_side,
                returns=returns,
                actual_atoms=actual_rows,
                oracle_atoms=oracle_rows,
                direct_calls=outcome.direct_calls,
            )

        return _Fixture(
            inputs,
            "left",
            threshold,
            np.dtype(np.float64).str,
            "left",
            "Decimal exact operand range and outward result",
            invoke,
        )

    if gate == "PLAN:gamma:operation-count-domain":
        count = 4
        threshold = 0.25
        epsilon = _float_value(
            point,
            threshold,
            low=float(np.finfo(np.float64).eps),
            high=0.5,
            extreme=math.inf,
        )
        if atom is not None:
            epsilon = threshold
        inputs = {"operation_count": count, "epsilon": epsilon}

        def invoke(data: Mapping[str, Any]) -> _Outcome:
            arithmetic_error: ArithmeticError | None = None
            try:
                result = plan._gamma_for_count(data["operation_count"], data["epsilon"])
            except ArithmeticError as error:
                # A loosened domain guard can expose the zero denominator.
                # Retain that as failed proof evidence instead of letting a
                # harness exception masquerade as a mutation kill.
                arithmetic_error = error
                result = math.nan
            exact_gamma_product = Decimal(data["operation_count"]) * Decimal.from_float(
                float(data["epsilon"])
            )
            # Reproduce the independently specified binary64 staging: the
            # product, denominator, and quotient each round before the final
            # outward step.  Treating the rational expression as one exact
            # operation is a different algorithm and differs by an ulp away
            # from exactly representable epsilon values.
            rounded_product = float(exact_gamma_product)
            if rounded_product >= 1.0:
                expected = math.inf
            else:
                exact_denominator = Decimal(1) - Decimal.from_float(rounded_product)
                rounded_denominator = float(exact_denominator)
                exact_quotient = Decimal.from_float(
                    rounded_product
                ) / Decimal.from_float(rounded_denominator)
                expected = _literal_outward_nonnegative(float(exact_quotient))
            finite = math.isfinite(result)
            production_product = data["operation_count"] * data["epsilon"]
            production_comparison = production_product >= 1.0
            oracle_comparison = rounded_product >= 1.0
            actual_rows = _atom_row(
                entry,
                (
                    _production_atom(
                        production_comparison,
                        production_comparison,
                        source="gamma retained product domain comparison",
                    ),
                ),
            )
            oracle_rows = _atom_row(
                entry,
                (
                    _oracle_atom(
                        oracle_comparison,
                        oracle_comparison,
                        source="Decimal gamma product domain",
                    ),
                ),
            )
            return _Outcome(
                observed_side=GateSide.ADMITTED if finite else GateSide.REFUSED,
                oracle_side=GateSide.ADMITTED
                if math.isfinite(expected)
                else GateSide.REFUSED,
                returns={
                    "gamma_result": result,
                    "raised_type": (
                        None
                        if arithmetic_error is None
                        else type(arithmetic_error).__name__
                    ),
                },
                actual_atoms=actual_rows,
                oracle_atoms=oracle_rows,
                payload_checks=(
                    (
                        "completed gamma denominator evaluation",
                        (
                            None
                            if arithmetic_error is None
                            else type(arithmetic_error).__name__
                        ),
                        None,
                        arithmetic_error is None,
                    ),
                    (
                        "independent staged-binary64 gamma",
                        result,
                        expected,
                        _raw_equal(result, expected),
                    ),
                ),
                direct_calls=("plan._gamma_for_count",),
            )

        return _Fixture(
            inputs,
            "epsilon",
            threshold,
            np.dtype(np.float64).str,
            "epsilon",
            "exact Fraction gamma denominator domain",
            invoke,
        )

    raise AssertionError(f"unmapped proof gate {gate}")


def _runtime_fixture(
    entry: GateEntry,
    point: ThresholdPoint,
    atom: _AtomSpec | None,
) -> _Fixture:
    gate = entry.gate_id
    if gate == "PLAN:frozen:x-bound-runtime-range":
        runtime_dtype = np.dtype(np.float32)
        threshold = float(np.finfo(runtime_dtype).max)
        target = _float_value(
            point,
            threshold,
            low=0.2,
            high=threshold * 1.5,
            extreme=threshold * 4.0,
        )
        if atom is not None:
            target = _ORIGINAL_NEXTAFTER(threshold, math.inf)
        fields = _certificate_fields(
            measured=0.0,
            certified=0.0,
            margin=0.0,
            tolerance=0.5,
            tail=0.25,
            x_scale=target,
            order=0,
        )
        inputs: dict[str, Any] = dict(fields)
        inputs.update(
            runtime_dtype=runtime_dtype.str,
            total_probe_energy=1.0,
            maximum_probe_norm=1.0,
            probe_count=1,
            dimension=1,
        )

        def invoke_x_bound(data: Mapping[str, Any]) -> _Outcome:
            certificate = _certificate_from(data)
            local_maximum = float(np.finfo(np.dtype(data["runtime_dtype"])).max)
            production_comparison = float(certificate.max_x_operator_norm) > float(
                local_maximum
            )
            oracle_comparison = Decimal.from_float(
                float(data["max_x_operator_norm"])
            ) > Decimal.from_float(local_maximum)
            actual_rows = _atom_row(
                entry,
                (
                    _production_atom(
                        production_comparison,
                        production_comparison,
                        source="frozen runtime retained X-bound comparison",
                    ),
                ),
            )
            oracle_rows = _atom_row(
                entry,
                (
                    _oracle_atom(
                        oracle_comparison,
                        oracle_comparison,
                        source="Decimal runtime dtype maximum comparison",
                    ),
                ),
            )
            return _validation_outcome(
                expected_valid=data["max_x_operator_norm"] <= local_maximum,
                operation=lambda: plan._validate_frozen_runtime_range(
                    runtime_dtype=np.dtype(data["runtime_dtype"]),
                    certificate=certificate,
                    total_probe_energy=data["total_probe_energy"],
                    maximum_probe_norm=data["maximum_probe_norm"],
                    probe_count=data["probe_count"],
                    dimension=data["dimension"],
                ),
                refused_exceptions=(ValueError,),
                return_value=lambda result: {"frozen_range_bound": result},
                actual_atoms=actual_rows,
                oracle_atoms=oracle_rows,
                direct_calls=("plan._validate_frozen_runtime_range",),
            )

        return _Fixture(
            inputs,
            "max_x_operator_norm",
            threshold,
            np.dtype(np.float64).str,
            "max_x_operator_norm",
            "literal runtime dtype maximum for certified abs(X) norm",
            invoke_x_bound,
        )

    if gate == "PLAN:frozen:intermediate-runtime-range":
        runtime_dtype = np.dtype(np.float16)
        certificate_fields = _certificate_fields(
            measured=0.1,
            certified=0.2,
            margin=0.1,
            tolerance=0.5,
            tail=0.1,
            x_scale=0.997,
            order=1,
        )
        component = _float_value(
            point,
            _INTERMEDIATE_PROBE_COMPONENT_THRESHOLD,
            low=200.0,
            high=255.9,
            extreme=255.9374923687422,
        )
        if point.role is PointRole.ABOVE_RELATIVE_1E12:
            component = _ORIGINAL_NEXTAFTER(component, -math.inf)
        try:
            target, _ = _independent_probe_bounds(component, 1, runtime_dtype)
        except (ArithmeticError, ValueError, OverflowError):
            target = math.inf
        threshold = _INTERMEDIATE_PROBE_COMPONENT_THRESHOLD
        inputs = dict(certificate_fields)
        inputs.update(
            probe_component=component,
            total_probe_energy=target,
            runtime_dtype=runtime_dtype.str,
            probe_count=1,
            dimension=1,
        )

        def invoke_intermediate(data: Mapping[str, Any]) -> _Outcome:
            certificate = _certificate_from(data)
            original = plan._runtime_range_product
            captured: dict[str, Any] = {}
            local_dtype = np.dtype(data["runtime_dtype"])
            probe_values = np.full(
                (int(data["probe_count"]), int(data["dimension"])),
                float(data["probe_component"]),
                dtype=np.float64,
            )
            def product_seam(
                left: float,
                right: float,
                local_maximum: float,
                dtype: np.dtype,
                quantity: str,
            ) -> float:
                if quantity == "the frozen correction accumulation":
                    captured["final_correction_operand"] = left
                    captured["final_addition_factor"] = right
                result = original(left, right, local_maximum, dtype, quantity)
                if quantity == "the frozen correction accumulation":
                    captured["final_correction_bound"] = result
                return result

            try:
                oracle_operand, oracle_bound = _independent_frozen_intermediate(
                    probe_component=float(data["probe_component"]),
                    runtime_dtype=local_dtype,
                    x_bound=float(data["max_x_operator_norm"]),
                    order=int(data["order"]),
                    probe_count=int(data["probe_count"]),
                    dimension=int(data["dimension"]),
                )
            except (ArithmeticError, ValueError, OverflowError):
                expected_valid = False
                oracle_operand = None
                oracle_bound = None
            else:
                expected_valid = True

            def validate_real_probe() -> float:
                probes = eager.FrozenProbes(probe_values)
                total_energy, maximum_norm = plan._frozen_probe_energy_bounds(
                    probes.values, local_dtype
                )
                captured["resolved_total_probe_energy"] = total_energy
                captured["resolved_maximum_probe_norm"] = maximum_norm
                return plan._validate_frozen_runtime_range(
                    runtime_dtype=local_dtype,
                    certificate=certificate,
                    total_probe_energy=total_energy,
                    maximum_probe_norm=maximum_norm,
                    probe_count=int(data["probe_count"]),
                    dimension=int(data["dimension"]),
                )

            with patch.object(plan, "_runtime_range_product", side_effect=product_seam):
                outcome = _validation_outcome(
                    expected_valid=expected_valid,
                    operation=validate_real_probe,
                    refused_exceptions=(ValueError,),
                    return_value=lambda result: {"frozen_range_bound": result},
                    direct_calls=(
                        "plan._frozen_probe_energy_bounds",
                        "plan._validate_frozen_runtime_range",
                    ),
                )
            returns = dict(outcome.returns)
            returns.update(captured)
            payload_checks: tuple[tuple[str, Any, Any, bool], ...] = ()
            if "resolved_total_probe_energy" in returns:
                payload_checks = (
                    (
                        "real probe derives the declared total energy",
                        returns["resolved_total_probe_energy"],
                        data["total_probe_energy"],
                        _raw_equal(
                            returns["resolved_total_probe_energy"],
                            data["total_probe_energy"],
                        ),
                    ),
                )
            if expected_valid:
                returned_bound = returns.get("frozen_range_bound", math.inf)
                payload_checks = (
                    *payload_checks,
                    (
                        "independent staged final frozen correction operand",
                        returns.get("final_correction_operand"),
                        oracle_operand,
                        _raw_equal(
                            returns.get("final_correction_operand"), oracle_operand
                        ),
                    ),
                    (
                        "independent staged final frozen correction bound",
                        returned_bound,
                        oracle_bound,
                        _raw_equal(returned_bound, oracle_bound),
                    ),
                )
            return _Outcome(
                observed_side=outcome.observed_side,
                oracle_side=outcome.oracle_side,
                returns=returns,
                payload_checks=payload_checks,
                direct_calls=outcome.direct_calls,
            )

        return _Fixture(
            inputs,
            "probe_component",
            threshold,
            np.dtype(np.float64).str,
            "total_probe_energy",
            "Decimal replay from represented probe energy through final range proof",
            invoke_intermediate,
        )

    if gate == "PLAN:runtime-call:scalar-and-dtype":
        role = point.role
        inputs: dict[str, Any] = {
            "expected_dtype": np.dtype(np.float32).str,
            "base_value": np.array(1.7, dtype=np.float32),
            "dynamic_value": np.array([2.9], dtype=np.float32),
            "x64_enabled": False,
            "capability_value": role.value,
        }
        if role is PointRole.CAPABILITY_LOW:
            inputs.update(
                expected_dtype=np.dtype(np.float16).str,
                base_value=np.array(0.7, dtype=np.float16),
                dynamic_value=np.array([3.1], dtype=np.float16),
            )
        elif role is PointRole.INVALID_CAPABILITY:
            inputs["base_value"] = np.array([1.7], dtype=np.float32)
        elif role is PointRole.CAPABILITY_HIGH:
            inputs["dynamic_value"] = np.array([2], dtype=np.int32)
        elif role is PointRole.EXTREME:
            inputs["expected_dtype"] = np.dtype(np.float64).str
        if atom is not None:
            inputs.update(
                expected_dtype=np.dtype(np.float32).str,
                base_value=np.array(1.7, dtype=np.float32),
                dynamic_value=np.array([2.9], dtype=np.float32),
                x64_enabled=False,
            )
            if atom.index == 0:
                inputs.update(
                    expected_dtype=np.dtype(np.float64).str,
                    base_value=np.array(1.7, dtype=np.float64),
                    dynamic_value=np.array([2.9], dtype=np.float64),
                    x64_enabled=True,
                )
            elif atom.index == 1:
                inputs.update(
                    expected_dtype=np.dtype(np.float64).str,
                    base_value=np.array(1.7, dtype=np.float64),
                    dynamic_value=np.array([2.9], dtype=np.float64),
                    x64_enabled=False,
                )
            elif atom.index in {2, 3}:
                inputs["base_value"] = np.array(1, dtype=np.int32)
                inputs["dynamic_value"] = np.array([2], dtype=np.int32)
            else:
                inputs["base_value"] = np.array(1.7, dtype=np.float16)
                inputs["dynamic_value"] = np.array([2.9], dtype=np.float16)

        def invoke_runtime_call(data: Mapping[str, Any]) -> _Outcome:
            expected = np.dtype(data["expected_dtype"])
            scalar = data["base_value"].ndim == 0
            context = jax.enable_x64(data["x64_enabled"])
            with context:
                actual_base_dtype = np.dtype(jnp.asarray(data["base_value"]).dtype)
                actual_dynamic_dtype = np.dtype(
                    jnp.asarray(data["dynamic_value"]).dtype
                )
                production_expected_wide = expected.itemsize > 4
                production_context_failure = (
                    production_expected_wide and not jax.config.x64_enabled
                )
                if production_context_failure:
                    production_kind_failure = None
                    production_dtype_failure = None
                    production_size_failure = None
                else:
                    production_kind_failure = actual_dynamic_dtype.kind != "f"
                    production_size_failure = (
                        None
                        if production_kind_failure
                        else actual_dynamic_dtype.itemsize < expected.itemsize
                    )
                    production_dtype_failure = bool(
                        production_kind_failure or production_size_failure
                    )
                oracle_expected_wide = int(expected.itemsize) > 4
                oracle_context_failure = oracle_expected_wide and not bool(
                    data["x64_enabled"]
                )
                if oracle_context_failure:
                    oracle_kind_failure = None
                    oracle_dtype_failure = None
                    oracle_size_failure = None
                else:
                    oracle_kind_failure = actual_dynamic_dtype.kind not in {"f"}
                    oracle_size_failure = (
                        None
                        if oracle_kind_failure
                        else int(actual_dynamic_dtype.itemsize) < int(expected.itemsize)
                    )
                    oracle_dtype_failure = bool(
                        oracle_kind_failure or oracle_size_failure
                    )
                oracle_valid = (
                    scalar
                    and not oracle_context_failure
                    and not bool(oracle_dtype_failure)
                )
                outcome = _validation_outcome(
                    expected_valid=oracle_valid,
                    operation=lambda: plan._require_runtime_precision(
                        data["expected_dtype"],
                        jnp.asarray(data["base_value"]),
                        jnp.asarray(data["dynamic_value"]),
                    ),
                    refused_exceptions=(ValueError,),
                    return_value=lambda _: {"runtime_precision_return": "accepted"},
                    direct_calls=("plan._require_runtime_precision",),
                )

            def production_item(value: bool | None, source: str) -> _AtomValue:
                return _production_atom(
                    value,
                    value,
                    AtomReducer.NOT_EVALUATED if value is None else AtomReducer.SCALAR,
                    source=source,
                )

            def oracle_item(value: bool | None, source: str) -> _AtomValue:
                return _oracle_atom(
                    value,
                    value,
                    AtomReducer.NOT_EVALUATED if value is None else AtomReducer.SCALAR,
                    source=source,
                )

            actual_rows = _atom_row(
                entry,
                (
                    production_item(
                        production_expected_wide,
                        "runtime expected dtype itemsize comparison",
                    ),
                    production_item(
                        production_context_failure,
                        "runtime x64-context conjunction",
                    ),
                    production_item(
                        production_kind_failure,
                        "runtime dynamic dtype kind comparison",
                    ),
                    production_item(
                        production_dtype_failure,
                        "runtime dtype invalid disjunction",
                    ),
                    production_item(
                        production_size_failure,
                        "runtime dynamic dtype itemsize comparison",
                    ),
                ),
            )
            oracle_rows = _atom_row(
                entry,
                (
                    oracle_item(
                        oracle_expected_wide,
                        "integer expected dtype width",
                    ),
                    oracle_item(
                        oracle_context_failure,
                        "literal x64 capability model",
                    ),
                    oracle_item(
                        oracle_kind_failure,
                        "dtype metadata floating-kind identity",
                    ),
                    oracle_item(
                        oracle_dtype_failure,
                        "dtype metadata invalidity disjunction",
                    ),
                    oracle_item(
                        oracle_size_failure,
                        "integer dtype byte-width comparison",
                    ),
                ),
            )
            returns = dict(outcome.returns)
            returns.update(
                base_dtype=actual_base_dtype.str,
                dynamic_dtype=actual_dynamic_dtype.str,
            )
            return _Outcome(
                observed_side=outcome.observed_side,
                oracle_side=outcome.oracle_side,
                returns=returns,
                actual_atoms=actual_rows,
                oracle_atoms=oracle_rows,
                direct_calls=outcome.direct_calls,
            )

        return _Fixture(
            inputs,
            "capability_value",
            "runtime precision capability",
            None,
            "capability_value",
            "literal scalar rank, JAX context, dtype kind and itemsize",
            invoke_runtime_call,
        )

    fields = _certificate_fields(
        measured=0.0,
        certified=0.0,
        margin=0.0,
        tolerance=0.5,
        tail=0.25,
        lambda_scale=0.0,
        x_scale=0.2,
        order=0,
    )
    inputs = dict(fields)
    inputs.update(
        lambda_matrix=np.array([1.0]),
        perturbation=np.array([0.0]),
        frozen=False,
        frozen_probe_values=np.array([[1.7]]),
    )
    if gate == "PLAN:runtime:sigma-finite-and-positive":
        threshold = _SIGMA_LAMBDA_THRESHOLD
        lambda_entry = _float_value(
            point,
            threshold,
            low=0.5,
            high=2.0,
            extreme=_FLOAT64_MAXIMUM,
        )
        if atom is not None:
            lambda_entry = threshold
        inputs.pop("lambda_matrix")
        inputs.pop("perturbation")
        inputs.update(
            lambda_entry=lambda_entry,
            perturbation_entry=_SIGMA_PERTURBATION_THRESHOLD,
            matrix_path="compact-float64",
        )
        point_key = "lambda_entry"
        expected_valid = lambda_entry > threshold
    elif gate == "PLAN:runtime:expected-and-ulp-finite":
        inputs.pop("lambda_matrix")
        inputs.pop("perturbation")
        threshold = _EXPECTED_ULP
        target = _float_value(
            point,
            threshold,
            low=threshold / 4.0,
            high=threshold * 4.0,
            extreme=threshold / 8.0,
        )
        if atom is not None:
            target = _ORIGINAL_NEXTAFTER(threshold, -math.inf)
        inputs.update(
            tolerance=target,
            tail_tolerance=target / 2.0,
            lambda_entry=math.e,
            perturbation_entry=0.0,
            runtime_dtype="<f8/x64-on",
        )
        point_key = "tolerance"
        expected_valid = target >= _EXPECTED_ULP
    elif gate == "PLAN:runtime:base-scale-range":
        runtime_dtype = np.dtype(np.float32)
        threshold = float(np.finfo(runtime_dtype).max)
        target = _float_value(
            point,
            threshold,
            low=1.7,
            high=threshold * 1.5,
            extreme=threshold * 4.0,
        )
        if atom is not None:
            target = _ORIGINAL_NEXTAFTER(threshold, math.inf)
        inputs.update(
            max_abs_lambda_logdet=target,
            runtime_dtype=f"{runtime_dtype.str}/x64-off",
            lambda_matrix=np.array([1.0], dtype=np.float32),
            perturbation=np.array([0.0], dtype=np.float32),
            runtime_x64=False,
        )
        point_key = "max_abs_lambda_logdet"
        expected_valid = target <= threshold
    elif gate == "PLAN:runtime:frozen-prerequisites-and-series":
        threshold = _SERIES_X_BOUND_THRESHOLD
        target = _float_value(
            point,
            threshold,
            low=0.0,
            high=1.0e100,
            extreme=_FLOAT64_MAXIMUM,
        )
        if atom is not None:
            target = _FLOAT64_MAXIMUM
        inputs.pop("frozen_probe_values")
        inputs.update(
            certified_rho=0.4,
            order=2,
            max_x_operator_norm=target,
            probe_component=1.0,
            frozen=True,
            runtime_x64=True,
        )
        point_key = "max_x_operator_norm"
        expected_valid = target < _FLOAT64_MAXIMUM
    elif gate == "PLAN:runtime:total-error-budget":
        threshold = _TOTAL_ERROR_TOLERANCE_THRESHOLD
        target = _float_value(
            point,
            threshold,
            low=_ORIGINAL_NEXTAFTER(_TOTAL_ERROR_TAIL_TOLERANCE, math.inf),
            high=threshold * 4.0,
            extreme=math.inf,
        )
        if atom is not None:
            target = _ORIGINAL_NEXTAFTER(threshold, -math.inf)
        inputs.update(
            measured_max=0.0,
            margin=_TOTAL_ERROR_RHO_THRESHOLD,
            certified_rho=_TOTAL_ERROR_RHO_THRESHOLD,
            order=_TOTAL_ERROR_ORDER,
            tolerance=target,
            tail_tolerance=_TOTAL_ERROR_TAIL_TOLERANCE,
            max_abs_lambda_logdet=_TOTAL_ERROR_BASE_SCALE_THRESHOLD,
            runtime_x64=True,
        )
        point_key = "tolerance"
        expected_valid = not math.isnan(target) and target >= threshold
    else:
        raise AssertionError(f"unmapped runtime gate {gate}")

    def invoke_runtime(data: Mapping[str, Any]) -> _Outcome:
        try:
            certificate = _certificate_from(data)
        except (TypeError, ValueError) as error:
            if gate != "PLAN:runtime:total-error-budget":
                raise
            return _Outcome(
                observed_side=GateSide.REFUSED,
                oracle_side=GateSide.REFUSED,
                returns={"raised_type": type(error).__name__},
                direct_calls=("plan.RhoCertificate",),
            )
        runtime_x64_enabled = bool(data.get("runtime_x64", False))
        if gate == "PLAN:runtime:sigma-finite-and-positive":
            path = data["matrix_path"]
            dtype = np.float32 if path.endswith("float32") else np.float64
            lambda_entry = data["lambda_entry"]
            perturbation_entry = data["perturbation_entry"]
            if path == "compact-singular":
                lambda_matrix = np.array([lambda_entry], dtype=dtype)
                perturbation = np.array([-lambda_entry], dtype=dtype)
            elif path.startswith("compact"):
                lambda_matrix = np.array([lambda_entry], dtype=dtype)
                perturbation = np.array([perturbation_entry], dtype=dtype)
            elif path == "dense-indefinite-float64":
                lambda_matrix = np.eye(2, dtype=dtype) * lambda_entry
                perturbation = np.diag(
                    np.array([-2.0 * lambda_entry, 0.0], dtype=dtype)
                )
            else:
                lambda_matrix = np.eye(2, dtype=dtype) * lambda_entry
                perturbation = np.eye(2, dtype=dtype) * perturbation_entry
        elif gate == "PLAN:runtime:base-scale-range":
            storage_dtype, context_name = str(data["runtime_dtype"]).split("/", 1)
            runtime_x64_enabled = context_name == "x64-on"
            lambda_matrix = np.asarray(
                data["lambda_matrix"], dtype=np.dtype(storage_dtype)
            )
            perturbation = np.asarray(
                data["perturbation"], dtype=np.dtype(storage_dtype)
            )
        elif gate == "PLAN:runtime:expected-and-ulp-finite":
            storage_dtype, context_name = str(data["runtime_dtype"]).split("/", 1)
            runtime_x64_enabled = context_name == "x64-on"
            lambda_matrix = np.asarray(
                [data["lambda_entry"]], dtype=np.dtype(storage_dtype)
            )
            perturbation = np.asarray(
                [data["perturbation_entry"]], dtype=np.dtype(storage_dtype)
            )
        else:
            lambda_matrix = data["lambda_matrix"]
            perturbation = data["perturbation"]
        problem = eager.LogDetProblem(lambda_matrix, perturbation)
        frozen_values = (
            np.asarray([[data["probe_component"]]], dtype=np.float64)
            if gate == "PLAN:runtime:frozen-prerequisites-and-series"
            else data.get("frozen_probe_values")
        )
        probes = eager.FrozenProbes(frozen_values) if data["frozen"] else None
        captured: dict[str, Any] = {}
        original_power_series = plan._outward_power_series
        original_product = plan._outward_product
        original_sum = plan._outward_sum
        original_range_sum = plan._runtime_range_sum
        original_two_sum = plan._two_sum_error
        original_log = plan.np.log
        original_spacing = plan.np.spacing
        original_tail = plan.whole_trace_log_tail_bound
        series_ready = False
        tail_ready = False
        roundoff_ready = False

        def power_series_seam(base: float, order: int) -> float:
            nonlocal series_ready
            if gate == "PLAN:runtime:frozen-prerequisites-and-series":
                result = original_power_series(base, order)
                series_ready = True
                captured["resolved_power_series"] = result
                return result
            return original_power_series(base, order)

        def product_seam(left: float, right: float) -> float:
            nonlocal roundoff_ready, series_ready
            if series_ready:
                series_ready = False
                result = original_product(left, right)
                captured["resolved_series_scale"] = result
                return result
            if roundoff_ready:
                roundoff_ready = False
                result = original_product(left, right)
                captured["resolved_roundoff_error"] = result
                return result
            return original_product(left, right)

        def tail_seam(*args: Any, **kwargs: Any) -> float:
            nonlocal tail_ready
            result = original_tail(*args, **kwargs)
            tail_ready = True
            captured["resolved_analytic_tail"] = result
            return result

        def sum_seam(left: float, right: float) -> float:
            nonlocal tail_ready
            if tail_ready:
                tail_ready = False
                result = original_sum(left, right)
                captured["resolved_total_error"] = result
                return result
            return original_sum(left, right)

        def range_sum_seam(*args: Any, **kwargs: Any) -> float:
            nonlocal roundoff_ready
            if gate in {
                "PLAN:runtime:base-scale-range",
                "PLAN:runtime:frozen-prerequisites-and-series",
            }:
                return 0.0
            result = original_range_sum(*args, **kwargs)
            if (
                gate == "PLAN:runtime:total-error-budget"
                and len(args) >= 5
                and args[4] == "the lambda-logdet plus trace-series scale"
            ):
                roundoff_ready = True
            return result

        def two_sum_seam(*args: Any, **kwargs: Any) -> Any:
            try:
                result = original_two_sum(*args, **kwargs)
            except ValueError:
                if gate == "PLAN:runtime:sigma-finite-and-positive":
                    attempted = np.asarray(args[0]) + np.asarray(args[1])
                    captured["resolved_sigma"] = attempted.copy()
                    captured["resolved_sigma_scalar"] = float(np.ravel(attempted)[0])
                raise
            else:
                captured["resolved_sigma"] = np.asarray(result[0]).copy()
                captured["resolved_sigma_scalar"] = float(np.ravel(result[0])[0])
                return result

        def log_seam(value: Any, *args: Any, **kwargs: Any) -> Any:
            values = np.asarray(value)
            if gate == "PLAN:runtime:sigma-finite-and-positive" and np.any(
                values <= 0.0
            ):
                return np.zeros_like(values, dtype=float)
            result = original_log(value, *args, **kwargs)
            if gate == "PLAN:runtime:expected-and-ulp-finite":
                captured["resolved_expected_value"] = float(np.sum(result))
            return result

        def spacing_seam(value: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_spacing(value, *args, **kwargs)
            if gate == "PLAN:runtime:expected-and-ulp-finite":
                captured["resolved_rounded_value"] = float(np.asarray(value))
                captured["resolved_ulp_value"] = float(abs(np.asarray(result)))
            return result

        patches = (
            patch.object(plan, "_outward_power_series", side_effect=power_series_seam),
            patch.object(plan, "_outward_product", side_effect=product_seam),
            patch.object(plan, "_outward_sum", side_effect=sum_seam),
            patch.object(plan, "_runtime_range_sum", side_effect=range_sum_seam),
            patch.object(
                plan,
                "whole_trace_log_tail_bound",
                side_effect=tail_seam,
            ),
            patch.object(plan, "_validate_frozen_runtime_range", return_value=0.0)
            if gate == "PLAN:runtime:frozen-prerequisites-and-series"
            else patch.object(
                plan,
                "_validate_frozen_runtime_range",
                wraps=plan._validate_frozen_runtime_range,
            ),
            patch.object(plan, "_two_sum_error", side_effect=two_sum_seam),
            patch.object(plan.np, "log", side_effect=log_seam),
            patch.object(plan.np, "spacing", side_effect=spacing_seam),
        )
        context = jax.enable_x64(runtime_x64_enabled)
        with (
            context,
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
        ):
            runtime_expected_valid = expected_valid
            if gate == "PLAN:runtime:total-error-budget":
                try:
                    _, _, oracle_total = _independent_runtime_total_error(
                        certified_rho=float(data["certified_rho"]),
                        order=int(data["order"]),
                        multiplicity=int(data["multiplicity"]),
                        base_scale=float(data["max_abs_lambda_logdet"]),
                        runtime_dtype=np.dtype(np.float64),
                    )
                except (ArithmeticError, ValueError, OverflowError):
                    runtime_expected_valid = False
                else:
                    runtime_expected_valid = not (
                        oracle_total > float(data["tolerance"])
                    )
            outcome = _validation_outcome(
                expected_valid=runtime_expected_valid,
                operation=lambda: plan._validate_runtime_precision(
                    problem,
                    certificate,
                    frozen=data["frozen"],
                    frozen_probes=probes,
                ),
                refused_exceptions=(ValueError,),
                return_value=lambda result: {"runtime_dtype_return": result},
                direct_calls=("plan._validate_runtime_precision",),
            )
        returns = dict(outcome.returns)
        returns.update(captured)
        if gate == "PLAN:runtime:sigma-finite-and-positive":
            production_sigma = np.asarray(captured["resolved_sigma"])
            production_raw = production_sigma <= 0.0
            production_any_raw = np.any(production_raw)
            production_any = bool(production_any_raw)
            oracle_raw = np.array(
                [
                    Decimal.from_float(float(value)) <= Decimal(0)
                    for value in np.ravel(production_sigma)
                ],
                dtype=bool,
            ).reshape(production_sigma.shape)
            oracle_any = any(bool(value) for value in oracle_raw.ravel())
            actual_rows = _atom_row(
                entry,
                (
                    _production_atom(
                        production_any_raw,
                        production_any,
                        source="runtime retained Sigma np.any return",
                    ),
                    _production_atom(
                        production_raw,
                        production_any,
                        AtomReducer.ANY_ELEMENT,
                        source="runtime retained Sigma nonpositive array",
                    ),
                ),
            )
            oracle_rows = _atom_row(
                entry,
                (
                    _oracle_atom(
                        oracle_any,
                        oracle_any,
                        source="Decimal Sigma nonpositive disjunction",
                    ),
                    _oracle_atom(
                        oracle_raw,
                        oracle_any,
                        AtomReducer.ANY_ELEMENT,
                        source="elementwise Decimal Sigma sign",
                    ),
                ),
            )
        elif gate == "PLAN:runtime:expected-and-ulp-finite":
            production_ulp = float(captured.get("resolved_ulp_value", math.inf))
            production_comparison = production_ulp > certificate.tolerance
            exact_sigma = Decimal.from_float(
                float(lambda_matrix[0])
            ) + Decimal.from_float(float(perturbation[0]))
            staged_sigma = float(exact_sigma)
            oracle_expected = math.log(staged_sigma)
            storage_dtype = np.dtype(problem.lambda_matrix.dtype)
            canonical_dtype = (
                storage_dtype
                if runtime_x64_enabled or storage_dtype.itemsize <= 4
                else np.dtype(np.float32)
            )
            oracle_rounded = np.asarray(abs(oracle_expected), dtype=canonical_dtype)
            oracle_ulp = float(abs(np.spacing(oracle_rounded)))
            oracle_comparison = oracle_ulp > float(data["tolerance"])
            actual_rows = _atom_row(
                entry,
                (
                    _production_atom(
                        production_comparison,
                        production_comparison,
                        source="runtime retained expected-logdet ULP comparison",
                    ),
                ),
            )
            oracle_rows = _atom_row(
                entry,
                (
                    _oracle_atom(
                        oracle_comparison,
                        oracle_comparison,
                        source="math.ulp analytic logdet comparison",
                    ),
                ),
            )
        elif gate == "PLAN:runtime:base-scale-range":
            storage_dtype = np.dtype(problem.lambda_matrix.dtype)
            canonical_dtype = (
                storage_dtype
                if runtime_x64_enabled or storage_dtype.itemsize <= 4
                else np.dtype(np.float32)
            )
            runtime_maximum = float(np.finfo(canonical_dtype).max)
            production_comparison = certificate.max_abs_lambda_logdet > runtime_maximum
            oracle_comparison = Decimal.from_float(
                float(data["max_abs_lambda_logdet"])
            ) > Decimal.from_float(runtime_maximum)
            actual_rows = _atom_row(
                entry,
                (
                    _production_atom(
                        production_comparison,
                        production_comparison,
                        source="runtime retained base-scale maximum comparison",
                    ),
                ),
            )
            oracle_rows = _atom_row(
                entry,
                (
                    _oracle_atom(
                        oracle_comparison,
                        oracle_comparison,
                        source="Decimal runtime dtype maximum comparison",
                    ),
                ),
            )
        elif gate == "PLAN:runtime:frozen-prerequisites-and-series":
            production_raw = np.isfinite(captured["resolved_series_scale"])
            production_finite = bool(production_raw)
            component = Decimal.from_float(abs(float(data["probe_component"])))
            x_bound = Decimal.from_float(float(data["max_x_operator_norm"]))
            exact_probe_energy = component * component
            exact_power_series = sum(
                (x_bound**power) / Decimal(power)
                for power in range(1, int(data["order"]) + 1)
            )
            exact_series_scale = exact_probe_energy * exact_power_series
            oracle_finite = exact_series_scale <= Decimal.from_float(
                float(np.finfo(np.float64).max)
            )
            actual_rows = _atom_row(
                entry,
                _production_aliases(
                    production_raw,
                    production_finite,
                    source="runtime retained frozen series np.isfinite",
                ),
            )
            oracle_rows = _atom_row(
                entry,
                _oracle_aliases(
                    oracle_finite,
                    oracle_finite,
                    source="exact frozen-series math.isfinite",
                ),
            )
        else:
            production_total = captured.get("resolved_total_error")
            production_comparison = (
                None
                if production_total is None
                else production_total > certificate.tolerance
            )
            try:
                _, _, oracle_total = _independent_runtime_total_error(
                    certified_rho=float(data["certified_rho"]),
                    order=int(data["order"]),
                    multiplicity=int(data["multiplicity"]),
                    base_scale=float(data["max_abs_lambda_logdet"]),
                    runtime_dtype=np.dtype(np.float64),
                )
            except (ArithmeticError, ValueError, OverflowError):
                oracle_comparison = None
            else:
                oracle_comparison = oracle_total > float(data["tolerance"])
            actual_rows = _atom_row(
                entry,
                (
                    _production_atom(
                        production_comparison,
                        production_comparison,
                        AtomReducer.NOT_EVALUATED
                        if production_comparison is None
                        else AtomReducer.SCALAR,
                        source="runtime retained total-error comparison",
                    ),
                ),
            )
            oracle_rows = _atom_row(
                entry,
                (
                    _oracle_atom(
                        oracle_comparison,
                        oracle_comparison,
                        AtomReducer.NOT_EVALUATED
                        if oracle_comparison is None
                        else AtomReducer.SCALAR,
                        source="Decimal total-error budget comparison",
                    ),
                ),
            )
        payload_checks = outcome.payload_checks
        if gate == "PLAN:runtime:total-error-budget" and "resolved_total_error" in captured:
            _, _, oracle_total = _independent_runtime_total_error(
                certified_rho=float(data["certified_rho"]),
                order=int(data["order"]),
                multiplicity=int(data["multiplicity"]),
                base_scale=float(data["max_abs_lambda_logdet"]),
                runtime_dtype=np.dtype(np.float64),
            )
            payload_checks = (
                *payload_checks,
                (
                    "independent staged analytic-tail plus roundoff bound",
                    captured["resolved_total_error"],
                    oracle_total,
                    _raw_equal(captured["resolved_total_error"], oracle_total),
                ),
            )
        return _Outcome(
            observed_side=outcome.observed_side,
            oracle_side=outcome.oracle_side,
            returns=returns,
            actual_atoms=actual_rows,
            oracle_atoms=oracle_rows,
            payload_checks=payload_checks,
            direct_calls=outcome.direct_calls,
        )

    return _Fixture(
        inputs,
        point_key,
        threshold,
        np.dtype(np.float64).str,
        point_key,
        "independent SPD/logdet, dtype range, Fraction gamma and error proof",
        invoke_runtime,
    )


def _plan_factory_fixture(
    entry: GateEntry,
    point: ThresholdPoint,
    atom: _AtomSpec | None,
) -> _Fixture:
    gate = entry.gate_id
    fields = _certificate_fields(
        measured=0.1,
        certified=0.1,
        margin=0.0,
        tolerance=0.25,
        tail=0.125,
        lambda_scale=3.0,
        x_scale=0.5,
    )
    inputs: dict[str, Any] = dict(fields)
    inputs.update(
        lambda_matrix=np.array([2.0, 4.0]),
        perturbation=np.array([0.125, 0.25]),
        problem_trace_order=fields["order"],
        certified_problem_rho=fields["certified_rho"],
    )
    if gate == "PLAN:trace-factory:exact-evidence":
        exact = oracles.exact_power_traces(
            inputs["lambda_matrix"], inputs["perturbation"], inputs["order"]
        )
        threshold = exact[0]
        if point.role is PointRole.SUBNORMAL_MISMATCH:
            inputs["perturbation"] = np.array([0.0, 0.0])
            exact = oracles.exact_power_traces(
                inputs["lambda_matrix"], inputs["perturbation"], inputs["order"]
            )
            threshold = exact[0]
            value = _ORIGINAL_NEXTAFTER(0.0, math.inf)
        else:
            value = _exact_value(point, threshold)
        traces: tuple[float, ...] | None = (value, *exact[1:])
        if point.role is PointRole.VERY_LOW:
            inputs["perturbation"] = np.array([0.0625, 0.0])
            traces = oracles.exact_power_traces(
                inputs["lambda_matrix"], inputs["perturbation"], inputs["order"]
            )
            value = traces[0]
        elif point.role is PointRole.VERY_HIGH:
            inputs["perturbation"] = np.array([0.1875, 0.375])
            traces = oracles.exact_power_traces(
                inputs["lambda_matrix"], inputs["perturbation"], inputs["order"]
            )
            value = traces[0]
        elif point.role is PointRole.EXTREME or atom is not None:
            traces = None
            value = math.inf
        inputs.update(exact_power_traces=traces, trace_evidence_value=value)
        expected_valid = traces is not None and traces == oracles.exact_power_traces(
            inputs["lambda_matrix"], inputs["perturbation"], inputs["order"]
        )

        def invoke_trace_factory(data: Mapping[str, Any]) -> _Outcome:
            certificate = _certificate_from(data)
            problem = eager.LogDetProblem(
                data["lambda_matrix"],
                data["perturbation"],
                trace_order=data["problem_trace_order"],
                certified_rho=data["certified_problem_rho"],
                exact_power_traces=data["exact_power_traces"],
            )
            production_missing = problem.exact_power_traces is None
            oracle_missing = data["exact_power_traces"] is None
            actual_rows = _atom_row(
                entry,
                (
                    _production_atom(
                        production_missing,
                        production_missing,
                        source="trace-plan retained exact-evidence presence",
                    ),
                ),
            )
            oracle_rows = _atom_row(
                entry,
                (
                    _oracle_atom(
                        oracle_missing,
                        oracle_missing,
                        source="literal trace-evidence fixture presence",
                    ),
                ),
            )
            return _validation_outcome(
                expected_valid=expected_valid,
                operation=lambda: plan.make_trace_log_plan(problem, certificate),
                refused_exceptions=(ValueError,),
                return_value=lambda result: {"plan_order": result.order},
                actual_atoms=actual_rows,
                oracle_atoms=oracle_rows,
                direct_calls=("plan.make_trace_log_plan",),
            )

        return _Fixture(
            inputs,
            "trace_evidence_value",
            threshold,
            np.dtype(np.float64).str,
            "trace_evidence_value",
            "explicit dense matrix-power trace identity and runtime proof",
            invoke_trace_factory,
        )

    if gate != "PLAN:frozen-factory:probe-presence-width":
        raise AssertionError(f"unmapped plan factory gate {gate}")
    threshold = 3
    width = _integer_value(point, threshold, low=1, high=6, extreme=10)
    if atom is not None:
        width = threshold + 1
    probes = np.full((2, width), 0.25)
    inputs.update(
        lambda_matrix=np.array([2.0, 3.0, 4.0]),
        perturbation=np.array([0.2, -0.3, 0.0]),
        max_abs_lambda_logdet=4.0,
        probe_presence=0 if probes is None else int(probes.shape[0]),
        probe_width=width,
        frozen_probe_values=probes,
    )

    def invoke_frozen_factory(data: Mapping[str, Any]) -> _Outcome:
        certificate = _certificate_from(data)
        frozen = (
            None
            if data["frozen_probe_values"] is None
            else eager.FrozenProbes(data["frozen_probe_values"])
        )
        problem = eager.LogDetProblem(
            data["lambda_matrix"],
            data["perturbation"],
            trace_order=data["problem_trace_order"],
            certified_rho=data["certified_problem_rho"],
            frozen_probes=frozen,
        )
        production_mismatch = (
            None
            if problem.frozen_probes is None
            else problem.frozen_probes.values.shape[1] != plan._n(problem.lambda_matrix)
        )
        oracle_mismatch = (
            None
            if data["frozen_probe_values"] is None
            else int(data["probe_width"])
            != int(np.asarray(data["lambda_matrix"]).shape[-1])
        )
        actual_rows = _atom_row(
            entry,
            (
                _production_atom(
                    production_mismatch,
                    production_mismatch,
                    AtomReducer.NOT_EVALUATED
                    if production_mismatch is None
                    else AtomReducer.SCALAR,
                    source="frozen-plan retained probe-width comparison",
                ),
            ),
        )
        oracle_rows = _atom_row(
            entry,
            (
                _oracle_atom(
                    oracle_mismatch,
                    oracle_mismatch,
                    AtomReducer.NOT_EVALUATED
                    if oracle_mismatch is None
                    else AtomReducer.SCALAR,
                    source="literal frozen-probe matrix-width identity",
                ),
            ),
        )
        width_predicate_calls: list[int] = []
        original_n = plan._n

        def traced_n(value: Any) -> int:
            result = int(original_n(value))
            width_predicate_calls.append(result)
            return result

        with patch.object(plan, "_n", side_effect=traced_n):
            outcome = _validation_outcome(
                expected_valid=(
                    int(data["probe_presence"]) > 0 and data["probe_width"] == threshold
                ),
                operation=lambda: plan.make_frozen_trace_log_plan(problem, certificate),
                refused_exceptions=(ValueError,),
                return_value=lambda result: {"plan_order": result.order},
                actual_atoms=actual_rows,
                oracle_atoms=oracle_rows,
                direct_calls=("plan.make_frozen_trace_log_plan",),
            )
        if point.role is PointRole.EXTREME:
            return outcome
        return replace(
            outcome,
            payload_checks=(
                (
                    "frozen width predicate reached its production dimension",
                    tuple(width_predicate_calls),
                    f"a dimension read ending in {threshold}",
                    bool(width_predicate_calls)
                    and width_predicate_calls[-1] == threshold,
                ),
            ),
        )

    return _Fixture(
        inputs,
        "probe_width",
        threshold,
        None,
        "probe_width",
        "literal frozen-probe presence and matrix width",
        invoke_frozen_factory,
    )


_CERTIFICATE_GATES = {
    "PLAN:multiplicity:index-and-gamma-domain",
    "PLAN:certificate:error-budget-domain",
    "PLAN:certificate:optional-scale-domain",
    "PLAN:certificate:order-is-derived",
}
_WARMUP_GATES = {
    "PLAN:warmup:lambda-scale-inputs",
    "PLAN:warmup:x-norm-inputs",
}
_AUDIT_GATES = {
    "PLAN:audit:retained-rho",
    "PLAN:audit:retained-lambda-scale",
    "PLAN:audit:retained-x-norm",
    "PLAN:audit:retained-trace-evidence",
}
_MEASUREMENT_GATES = {
    "PLAN:measurement:x-norm-finite",
}
_FACTORY_CERTIFICATE_GATES = {
    "PLAN:factory-certificate:order-and-rank",
    "PLAN:factory-certificate:lambda-scale",
    "PLAN:factory-certificate:x-norm",
}
_PROOF_GATES = {
    "PLAN:outward-arithmetic:positive-underflow",
    "PLAN:frozen:probe-energy-range",
    "PLAN:runtime-range:product",
    "PLAN:runtime-range:sum",
    "PLAN:gamma:operation-count-domain",
}
_RUNTIME_GATES = {
    "PLAN:frozen:x-bound-runtime-range",
    "PLAN:frozen:intermediate-runtime-range",
    "PLAN:runtime:sigma-finite-and-positive",
    "PLAN:runtime:expected-and-ulp-finite",
    "PLAN:runtime:base-scale-range",
    "PLAN:runtime:frozen-prerequisites-and-series",
    "PLAN:runtime:total-error-budget",
    "PLAN:runtime-call:scalar-and-dtype",
}
_PLAN_FACTORY_GATES = {
    "PLAN:trace-factory:exact-evidence",
    "PLAN:frozen-factory:probe-presence-width",
}
assert (
    _CERTIFICATE_GATES
    | _WARMUP_GATES
    | _AUDIT_GATES
    | _MEASUREMENT_GATES
    | _FACTORY_CERTIFICATE_GATES
    | _PROOF_GATES
    | _RUNTIME_GATES
    | _PLAN_FACTORY_GATES
) == set(_ENTRIES)


def _fixture(
    entry: GateEntry, point: ThresholdPoint, atom: _AtomSpec | None
) -> _Fixture:
    gate = entry.gate_id
    builders: tuple[
        tuple[
            set[str], Callable[[GateEntry, ThresholdPoint, _AtomSpec | None], _Fixture]
        ],
        ...,
    ] = (
        (_CERTIFICATE_GATES, _certificate_fixture),
        (_WARMUP_GATES, _warmup_fixture),
        (_AUDIT_GATES, _audit_fixture),
        (_MEASUREMENT_GATES, _measurement_fixture),
        (_FACTORY_CERTIFICATE_GATES, _factory_certificate_fixture),
        (_PROOF_GATES, _proof_fixture),
        (_RUNTIME_GATES, _runtime_fixture),
        (_PLAN_FACTORY_GATES, _plan_factory_fixture),
    )
    matches = [builder for gates, builder in builders if gate in gates]
    if len(matches) != 1:
        raise AssertionError(f"PLAN gate has no unique direct builder: {gate}")
    builder = matches[0]
    axis_cell = _COMPOUND_DOMAIN_AXIS_CELL_BY_DISPLAY.get(point.display_value)
    if atom is None and axis_cell is not None:
        if axis_cell.gate_id != gate:
            raise AssertionError("PLAN parameter-axis cell crossed gate ownership")
        baseline = builder(entry, _baseline_point(gate), None)
        inputs = dict(baseline.inputs)
        inputs.update(axis_cell.overrides)

        def invoke_axis(data: Mapping[str, Any]) -> _Outcome:
            outcome = baseline.invoke(data)
            return replace(
                outcome,
                oracle_side=point.expected_side,
                payload_checks=(),
            )

        return _Fixture(
            inputs,
            axis_cell.realized_key or axis_cell.input_key,
            axis_cell.threshold,
            np.dtype(np.float64).str,
            axis_cell.input_key,
            baseline.oracle_name,
            invoke_axis,
        )
    return builder(entry, point, atom)


_INTEGER_GATES = {
    "PLAN:multiplicity:index-and-gamma-domain",
    "PLAN:factory-certificate:order-and-rank",
    "PLAN:frozen-factory:probe-presence-width",
}
_EXACT_GATES: set[str] = set()
_CAPABILITY_GATES = {"PLAN:runtime-call:scalar-and-dtype"}
_ZERO_THRESHOLD_GATES = {
    "PLAN:certificate:optional-scale-domain",
    "PLAN:warmup:rho-inputs-and-margin",
    "PLAN:warmup:lambda-scale-inputs",
    "PLAN:warmup:x-norm-inputs",
    "PLAN:outward-arithmetic:positive-underflow",
    "PLAN:runtime:sigma-finite-and-positive",
}
_OMITTED_RELATIVE = frozenset(
    {
        PointRole.BELOW_RELATIVE_1E6,
        PointRole.BELOW_RELATIVE_1E12,
        PointRole.ABOVE_RELATIVE_1E12,
        PointRole.ABOVE_RELATIVE_1E6,
    }
)


def _topology(gate: str) -> BoundaryTopology:
    if gate in _INTEGER_GATES:
        return BoundaryTopology.INTEGER
    if gate in _EXACT_GATES:
        return BoundaryTopology.EXACT
    if gate in _CAPABILITY_GATES:
        return BoundaryTopology.CAPABILITY
    return BoundaryTopology.FLOAT


def _points(gate: str) -> tuple[ThresholdPoint, ...]:
    if gate in _CAPABILITY_GATES:
        return capability_grid()
    if gate in _EXACT_GATES:
        return exact_grid()
    if gate in _INTEGER_GATES:
        if gate in {
            "PLAN:multiplicity:index-and-gamma-domain",
            "PLAN:gamma:operation-count-domain",
        }:
            return integer_grid(
                below=GateSide.ADMITTED,
                at=GateSide.REFUSED,
                above=GateSide.REFUSED,
                very_low=GateSide.ADMITTED,
                very_high=GateSide.REFUSED,
                extreme=GateSide.ADMITTED
                if gate.endswith("operation-count-domain")
                else GateSide.REFUSED,
            )
        if gate == "PLAN:factory-certificate:order-and-rank":
            return integer_grid(
                below=GateSide.ADMITTED,
                at=GateSide.ADMITTED,
                above=GateSide.REFUSED,
                very_low=GateSide.ADMITTED,
                very_high=GateSide.REFUSED,
                extreme=GateSide.REFUSED,
            )
        return integer_grid(
            below=GateSide.REFUSED,
            at=GateSide.ADMITTED,
            above=GateSide.REFUSED,
            very_low=GateSide.REFUSED,
            very_high=GateSide.REFUSED,
            extreme=GateSide.REFUSED,
        )
    if gate == "PLAN:certificate:order-is-derived":
        return float_grid(
            below=GateSide.REFUSED,
            at=GateSide.ADMITTED,
            above=GateSide.ADMITTED,
            very_low=GateSide.REFUSED,
            very_high=GateSide.REFUSED,
            extreme=GateSide.REFUSED,
        )
    if gate == "PLAN:gamma:operation-count-domain":
        return float_grid(
            below=GateSide.ADMITTED,
            at=GateSide.REFUSED,
            above=GateSide.REFUSED,
            very_low=GateSide.ADMITTED,
            very_high=GateSide.REFUSED,
            extreme=GateSide.REFUSED,
        )
    if gate == "PLAN:audit:retained-trace-evidence":
        # The trace-evidence axis moves the retained value itself, so every
        # non-exact cell fails the literal trace check; the perturbation axis
        # has its own parameter-axis cells with fixed companions.
        return float_grid(
            below=GateSide.REFUSED,
            at=GateSide.ADMITTED,
            above=GateSide.REFUSED,
            very_low=GateSide.REFUSED,
            very_high=GateSide.REFUSED,
            extreme=GateSide.REFUSED,
            include_relative=False,
        )
    if gate == "PLAN:trace-factory:exact-evidence":
        return float_grid(
            below=GateSide.REFUSED,
            at=GateSide.ADMITTED,
            above=GateSide.REFUSED,
            very_low=GateSide.ADMITTED,
            very_high=GateSide.ADMITTED,
            extreme=GateSide.REFUSED,
            include_relative=False,
        )
    if gate == "PLAN:canonical-probes:runtime-finite":
        base = float_grid(
            below=GateSide.ADMITTED,
            at=GateSide.ADMITTED,
            above=GateSide.REFUSED,
            very_low=GateSide.ADMITTED,
            very_high=GateSide.REFUSED,
            extreme=GateSide.REFUSED,
        )
        return tuple(
            ThresholdPoint(
                item.role,
                item.display_value,
                item.delta,
                GateSide.ADMITTED
                if item.role in {PointRole.ABOVE_ULP, PointRole.ABOVE_RELATIVE_1E12}
                else item.expected_side,
            )
            for item in base
        )
    if gate in _ZERO_THRESHOLD_GATES:
        if gate in {
            "PLAN:certificate:optional-scale-domain",
            "PLAN:warmup:rho-inputs-and-margin",
            "PLAN:warmup:lambda-scale-inputs",
            "PLAN:warmup:x-norm-inputs",
        }:
            return float_grid(
                below=GateSide.REFUSED,
                at=GateSide.ADMITTED,
                above=GateSide.ADMITTED,
                very_low=GateSide.REFUSED,
                very_high=GateSide.ADMITTED,
                extreme=GateSide.REFUSED,
                include_relative=False,
            )
        if gate == "PLAN:outward-arithmetic:positive-underflow":
            return float_grid(
                below=GateSide.REFUSED,
                at=GateSide.ADMITTED,
                above=GateSide.REFUSED,
                very_low=GateSide.REFUSED,
                very_high=GateSide.REFUSED,
                extreme=GateSide.ADMITTED,
                include_relative=False,
            )
        if gate == "PLAN:runtime:sigma-finite-and-positive":
            return float_grid(
                below=GateSide.REFUSED,
                at=GateSide.REFUSED,
                above=GateSide.ADMITTED,
                very_low=GateSide.REFUSED,
                very_high=GateSide.ADMITTED,
                extreme=GateSide.ADMITTED,
                include_relative=False,
            )
        return float_grid(
            below=GateSide.REFUSED,
            at=GateSide.REFUSED,
            above=GateSide.ADMITTED,
            very_low=GateSide.REFUSED,
            very_high=GateSide.ADMITTED,
            extreme=GateSide.REFUSED,
            include_relative=False,
        )
    if gate in {
        "PLAN:certificate:rho-domain-and-coverage",
        "PLAN:certificate:error-budget-domain",
        "PLAN:warmup:tail-fraction",
        "PLAN:warmup:rho-roundoff-ceiling",
        "PLAN:frozen:probe-energy-range",
        "PLAN:runtime-range:product",
        "PLAN:runtime-range:sum",
        "PLAN:frozen:intermediate-runtime-range",
    }:
        at = (
            GateSide.ADMITTED
            if gate == "PLAN:certificate:error-budget-domain"
            else GateSide.REFUSED
        )
        # The error-budget grid varies tail_tolerance at its strict upper end.
        if gate == "PLAN:certificate:error-budget-domain":
            at = GateSide.REFUSED
        return float_grid(
            below=GateSide.ADMITTED,
            at=at,
            above=GateSide.REFUSED,
            very_low=GateSide.ADMITTED,
            very_high=GateSide.REFUSED,
            extreme=GateSide.REFUSED,
        )
    if gate in {
        "PLAN:audit:retained-rho",
        "PLAN:audit:retained-lambda-scale",
        "PLAN:audit:retained-x-norm",
        "PLAN:factory-certificate:lambda-scale",
        "PLAN:factory-certificate:x-norm",
        "PLAN:frozen:x-bound-runtime-range",
        "PLAN:runtime:base-scale-range",
    }:
        return float_grid(
            below=GateSide.ADMITTED,
            at=GateSide.ADMITTED,
            above=GateSide.REFUSED,
            very_low=GateSide.ADMITTED,
            very_high=GateSide.REFUSED,
            extreme=GateSide.REFUSED,
        )
    if gate == "PLAN:runtime:total-error-budget":
        return float_grid(
            below=GateSide.REFUSED,
            at=GateSide.ADMITTED,
            above=GateSide.ADMITTED,
            very_low=GateSide.REFUSED,
            very_high=GateSide.ADMITTED,
            extreme=GateSide.ADMITTED,
            include_relative=False,
        )
    if gate == "PLAN:runtime:expected-and-ulp-finite":
        return float_grid(
            below=GateSide.REFUSED,
            at=GateSide.ADMITTED,
            above=GateSide.ADMITTED,
            very_low=GateSide.REFUSED,
            very_high=GateSide.ADMITTED,
            extreme=GateSide.REFUSED,
        )
    if gate == "PLAN:measurement:x-norm-finite":
        return float_grid(
            below=GateSide.ADMITTED,
            at=GateSide.ADMITTED,
            above=GateSide.REFUSED,
            very_low=GateSide.ADMITTED,
            very_high=GateSide.REFUSED,
            extreme=GateSide.REFUSED,
            include_relative=False,
        )
    if gate == "PLAN:measurement:lambda-logdet-finite":
        return float_grid(
            below=GateSide.REFUSED,
            at=GateSide.REFUSED,
            above=GateSide.ADMITTED,
            very_low=GateSide.REFUSED,
            very_high=GateSide.ADMITTED,
            extreme=GateSide.REFUSED,
            include_relative=False,
        )
    if gate == "PLAN:runtime:frozen-prerequisites-and-series":
        return float_grid(
            below=GateSide.ADMITTED,
            at=GateSide.ADMITTED,
            above=GateSide.ADMITTED,
            very_low=GateSide.ADMITTED,
            very_high=GateSide.ADMITTED,
            extreme=GateSide.REFUSED,
        )
    raise AssertionError(f"PLAN float grid has no semantic declaration: {gate}")


def _review_extra_points(gate: str) -> tuple[ThresholdPoint, ...]:
    parameter_points = tuple(
        cell.point for cell in _COMPOUND_DOMAIN_AXIS_CELLS_BY_GATE.get(gate, ())
    )
    if gate == "PLAN:certificate:error-budget-domain":
        return tuple(cell.point for cell in _ERROR_BUDGET_AXIS_CELLS)
    if gate == "PLAN:multiplicity:index-and-gamma-domain":
        return (
            *parameter_points,
            ThresholdPoint(
                PointRole.EXTREME,
                "extreme bool",
                "registry type extreme",
                GateSide.REFUSED,
            ),
            ThresholdPoint(
                PointRole.EXTREME,
                "extreme float",
                "registry type extreme",
                GateSide.REFUSED,
            ),
            ThresholdPoint(
                PointRole.EXTREME,
                "extreme huge integer",
                "registry magnitude extreme",
                GateSide.REFUSED,
            ),
        )
    if gate in _FINITE_LIMIT_GATES:
        return (
            *parameter_points,
            ThresholdPoint(
                PointRole.EXTREME,
                "minimum positive float64 subnormal",
                "dtype magnitude extreme",
                GateSide.ADMITTED,
            ),
            ThresholdPoint(
                PointRole.EXTREME,
                "last finite float64",
                "last finite",
                GateSide.ADMITTED,
            ),
            ThresholdPoint(
                PointRole.EXTREME,
                "first float64 overflow",
                "nextafter(max, +inf)",
                GateSide.REFUSED,
            ),
            ThresholdPoint(
                PointRole.EXTREME,
                "last finite float32 with non-scalar fixture",
                "dtype/shape extreme",
                GateSide.ADMITTED,
            ),
        )
    if gate == "PLAN:canonical-probes:runtime-finite":
        return (
            *parameter_points,
            ThresholdPoint(
                PointRole.EXTREME,
                "post-cast nonfinite probe seam",
                "runtime-dtype conversion produced infinity",
                GateSide.REFUSED,
            ),
        )
    return parameter_points


def _atom_side(gate: str, atom: _AtomSpec) -> GateSide:
    if gate == "PLAN:certificate:optional-scale-domain" and atom.index in {0, 5}:
        return GateSide.ADMITTED
    if gate == "PLAN:outward-arithmetic:positive-underflow":
        return GateSide.ADMITTED
    if gate == "PLAN:runtime-call:scalar-and-dtype" and atom.index == 0:
        return GateSide.ADMITTED
    return GateSide.REFUSED


def _atom_point(gate: str, atom: _AtomSpec) -> ThresholdPoint:
    return ThresholdPoint(
        PointRole.EXTREME,
        f"isolated source atom {atom.index + 1}",
        "real source outcome",
        _atom_side(gate, atom),
    )


def _baseline_point(gate: str) -> ThresholdPoint:
    points = _points(gate)
    if gate == "PLAN:outward-arithmetic:positive-underflow":
        return next(item for item in points if item.role is PointRole.ABOVE_ULP)
    if gate == "PLAN:runtime-range:product":
        return next(item for item in points if item.role is PointRole.BELOW_ULP)
    if gate == "PLAN:runtime-range:sum":
        return next(
            item for item in points if item.role is PointRole.BELOW_RELATIVE_1E6
        )
    return next(item for item in points if item.expected_side is GateSide.ADMITTED)


_ALL_RELATION_DECLARATIONS = {
        "PLAN:audit:retained-trace-evidence": (
            (0, 1),
            (False, False),
            (
                ("dependent", True, None, ((1, None),), "short-circuit"),
                ("independent", True, None, (), None),
            ),
        ),
        "PLAN:canonical-probes:runtime-finite": (
            (0, 1),
            (True, True),
            (
                ("dependent", False, None, ((1, False),), "equivalent"),
                (
                    "dependent",
                    False,
                    None,
                    ((0, False),),
                    "target-implies-prerequisites",
                ),
                (
                    "alias",
                    False,
                    1,
                    ((0, False),),
                    "target-implies-prerequisites",
                ),
            ),
        ),
        "PLAN:certificate:error-budget-domain": (
            (0, 1, 2),
            (False, False, True),
            (
                (
                    "dependent",
                    True,
                    None,
                    ((1, None), (2, None)),
                    "short-circuit",
                ),
                ("dependent", True, None, ((2, None),), "short-circuit"),
                ("independent", False, None, (), None),
            ),
        ),
        "PLAN:certificate:optional-scale-domain": (
            (0, 1, 2, 4, 5, 6, 7, 9),
            (True, False, True, False, True, False, True, False),
            (
                (
                    "dependent",
                    False,
                    None,
                    ((1, None), (2, None), (4, None)),
                    "short-circuit",
                ),
                (
                    "dependent",
                    True,
                    None,
                    (
                        (2, False),
                        (4, None),
                        (5, None),
                        (6, None),
                        (7, None),
                        (9, None),
                    ),
                    "short-circuit",
                ),
                (
                    "dependent",
                    False,
                    None,
                    (
                        (1, True),
                        (4, None),
                        (5, None),
                        (6, None),
                        (7, None),
                        (9, None),
                    ),
                    "short-circuit",
                ),
                (
                    "alias",
                    False,
                    2,
                    (
                        (1, True),
                        (4, None),
                        (5, None),
                        (6, None),
                        (7, None),
                        (9, None),
                    ),
                    "short-circuit",
                ),
                (
                    "dependent",
                    True,
                    None,
                    (
                        (1, True),
                        (5, None),
                        (6, None),
                        (7, None),
                        (9, None),
                    ),
                    "short-circuit",
                ),
                (
                    "dependent",
                    False,
                    None,
                    ((6, None), (7, None), (9, None)),
                    "short-circuit",
                ),
                (
                    "dependent",
                    True,
                    None,
                    ((7, False), (9, None)),
                    "short-circuit",
                ),
                (
                    "dependent",
                    False,
                    None,
                    ((6, True), (9, None)),
                    "short-circuit",
                ),
                (
                    "alias",
                    False,
                    7,
                    ((6, True), (9, None)),
                    "short-circuit",
                ),
                (
                    "dependent",
                    True,
                    None,
                    ((6, True),),
                    "target-implies-prerequisites",
                ),
            ),
        ),
        "PLAN:certificate:order-is-derived": (
            (0,),
            (False,),
            (("independent", True, None, (), None),),
        ),
        "PLAN:certificate:rho-domain-and-coverage": (
            (0, 1, 2),
            (True, True, False),
            (
                (
                    "dependent",
                    False,
                    None,
                    ((1, None), (2, None)),
                    "short-circuit",
                ),
                ("dependent", False, None, ((2, None),), "short-circuit"),
                ("independent", True, None, (), None),
            ),
        ),
        "PLAN:factory-certificate:lambda-scale": (
            (0,),
            (False,),
            (("independent", True, None, (), None),),
        ),
        "PLAN:factory-certificate:order-and-rank": (
            (0,),
            (False,),
            (("independent", True, None, (), None),),
        ),
        "PLAN:factory-certificate:x-norm": (
            (0,),
            (False,),
            (("independent", True, None, (), None),),
        ),
        "PLAN:frozen-factory:probe-presence-width": (
            (0,),
            (False,),
            (("independent", True, None, (), None),),
        ),
        "PLAN:frozen:probe-energy-range": (
            (0, 2),
            (True, False),
            (
                ("dependent", False, None, ((2, None),), "short-circuit"),
                ("alias", False, 0, ((2, None),), "short-circuit"),
                ("independent", True, None, (), None),
            ),
        ),
        "PLAN:frozen:x-bound-runtime-range": (
            (0,),
            (False,),
            (("independent", True, None, (), None),),
        ),
        "PLAN:gamma:operation-count-domain": (
            (0,),
            (False,),
            (("independent", True, None, (), None),),
        ),
        "PLAN:measurement:x-norm-finite": (
            (0, 1, 3),
            (True, True, True),
            (
                (
                    "dependent",
                    False,
                    None,
                    ((1, False), (3, None)),
                    "short-circuit",
                ),
                (
                    "dependent",
                    False,
                    None,
                    ((0, False), (3, None)),
                    "short-circuit",
                ),
                (
                    "alias",
                    False,
                    1,
                    ((0, False), (3, None)),
                    "short-circuit",
                ),
                ("independent", False, None, (), None),
                ("alias", False, 3, (), None),
            ),
        ),
        "PLAN:multiplicity:index-and-gamma-domain": (
            (0,),
            (False,),
            (("independent", True, None, (), None),),
        ),
        "PLAN:outward-arithmetic:positive-underflow": (
            (0, 1),
            (False, True),
            (
                ("dependent", True, None, ((1, None),), "short-circuit"),
                ("independent", False, None, (), None),
                ("alias", False, 1, (), None),
            ),
        ),
        "PLAN:runtime-call:scalar-and-dtype": (
            (0, 1, 2, 3, 4),
            (False, False, False, False, False),
            (
                ("independent", True, None, (), None),
                (
                    "dependent",
                    True,
                    None,
                    ((0, True), (2, None), (3, None), (4, None)),
                    "short-circuit",
                ),
                (
                    "dependent",
                    True,
                    None,
                    ((3, True), (4, None)),
                    "short-circuit",
                ),
                (
                    "dependent",
                    True,
                    None,
                    ((2, True), (4, None)),
                    "short-circuit",
                ),
                (
                    "dependent",
                    True,
                    None,
                    ((3, True),),
                    "target-implies-prerequisites",
                ),
            ),
        ),
        "PLAN:runtime-range:product": (
            (0, 2, 4, 5, 7),
            (True, True, False, True, False),
            (
                (
                    "dependent",
                    False,
                    None,
                    ((2, None), (4, None), (5, None), (7, None)),
                    "short-circuit",
                ),
                (
                    "alias",
                    False,
                    0,
                    ((2, None), (4, None), (5, None), (7, None)),
                    "short-circuit",
                ),
                (
                    "dependent",
                    False,
                    None,
                    ((4, None), (5, None), (7, None)),
                    "short-circuit",
                ),
                (
                    "alias",
                    False,
                    2,
                    ((4, None), (5, None), (7, None)),
                    "short-circuit",
                ),
                (
                    "dependent",
                    True,
                    None,
                    ((5, None), (7, None)),
                    "short-circuit",
                ),
                ("dependent", False, None, ((7, None),), "short-circuit"),
                ("alias", False, 5, ((7, None),), "short-circuit"),
                ("independent", True, None, (), None),
            ),
        ),
        "PLAN:runtime-range:sum": (
            (0, 2, 4, 5, 6, 8),
            (True, True, False, False, True, False),
            (
                (
                    "dependent",
                    False,
                    None,
                    ((2, None), (4, None), (5, None), (6, None), (8, None)),
                    "short-circuit",
                ),
                (
                    "alias",
                    False,
                    0,
                    ((2, None), (4, None), (5, None), (6, None), (8, None)),
                    "short-circuit",
                ),
                (
                    "dependent",
                    False,
                    None,
                    ((4, None), (5, None), (6, None), (8, None)),
                    "short-circuit",
                ),
                (
                    "alias",
                    False,
                    2,
                    ((4, None), (5, None), (6, None), (8, None)),
                    "short-circuit",
                ),
                (
                    "dependent",
                    True,
                    None,
                    ((5, None), (6, None), (8, None)),
                    "short-circuit",
                ),
                (
                    "dependent",
                    True,
                    None,
                    ((6, None), (8, None)),
                    "short-circuit",
                ),
                ("dependent", False, None, ((8, None),), "short-circuit"),
                ("alias", False, 6, ((8, None),), "short-circuit"),
                ("independent", True, None, (), None),
            ),
        ),
        "PLAN:runtime:base-scale-range": (
            (0,),
            (False,),
            (("independent", True, None, (), None),),
        ),
        "PLAN:runtime:expected-and-ulp-finite": (
            (0,),
            (False,),
            (("independent", True, None, (), None),),
        ),
        "PLAN:runtime:frozen-prerequisites-and-series": (
            (0,),
            (True,),
            (
                ("independent", False, None, (), None),
                ("alias", False, 0, (), None),
            ),
        ),
        "PLAN:runtime:sigma-finite-and-positive": (
            (0, 1),
            (False, False),
            (
                ("dependent", True, None, ((1, True),), "equivalent"),
                (
                    "dependent",
                    True,
                    None,
                    ((0, True),),
                    "target-implies-prerequisites",
                ),
            ),
        ),
        "PLAN:runtime:total-error-budget": (
            (0,),
            (False,),
            (("independent", True, None, (), None),),
        ),
        "PLAN:trace-factory:exact-evidence": (
            (0,),
            (False,),
            (("independent", True, None, (), None),),
        ),
        "PLAN:warmup:lambda-scale-inputs": (
            (0, 2, 4),
            (True, True, False),
            (
                (
                    "dependent",
                    False,
                    None,
                    ((2, None), (4, None)),
                    "short-circuit",
                ),
                (
                    "alias",
                    False,
                    0,
                    ((2, None), (4, None)),
                    "short-circuit",
                ),
                ("dependent", False, None, ((4, None),), "short-circuit"),
                ("alias", False, 2, ((4, None),), "short-circuit"),
                ("independent", True, None, (), None),
            ),
        ),
        "PLAN:warmup:rho-inputs-and-margin": (
            (0, 1, 3),
            (False, True, False),
            (
                (
                    "dependent",
                    True,
                    None,
                    ((1, False), (3, None)),
                    "short-circuit",
                ),
                (
                    "dependent",
                    False,
                    None,
                    ((0, True), (3, None)),
                    "short-circuit",
                ),
                (
                    "alias",
                    False,
                    1,
                    ((0, True), (3, None)),
                    "short-circuit",
                ),
                (
                    "dependent",
                    True,
                    None,
                    ((0, True),),
                    "target-implies-prerequisites",
                ),
            ),
        ),
        "PLAN:warmup:rho-roundoff-ceiling": (
            (0,),
            (True,),
            (("independent", False, None, (), None),),
        ),
        "PLAN:warmup:tail-fraction": (
            (0, 2),
            (True, True),
            (
                ("dependent", False, None, ((2, None),), "short-circuit"),
                ("alias", False, 0, ((2, None),), "short-circuit"),
                ("independent", False, None, (), None),
            ),
        ),
        "PLAN:warmup:x-norm-inputs": (
            (0, 1, 3, 4, 6),
            (False, True, False, True, False),
            (
                (
                    "dependent",
                    True,
                    None,
                    ((1, False), (3, None), (4, None), (6, None)),
                    "short-circuit",
                ),
                (
                    "dependent",
                    False,
                    None,
                    ((0, True), (3, None), (4, None), (6, None)),
                    "short-circuit",
                ),
                (
                    "alias",
                    False,
                    1,
                    ((0, True), (3, None), (4, None), (6, None)),
                    "short-circuit",
                ),
                (
                    "dependent",
                    True,
                    None,
                    ((0, True), (4, None), (6, None)),
                    "short-circuit",
                ),
                ("dependent", False, None, ((6, None),), "short-circuit"),
                ("alias", False, 4, ((6, None),), "short-circuit"),
                ("independent", True, None, (), None),
            ),
        ),
}
_RELATION_DECLARATIONS = MappingProxyType(
    {
        gate: group
        for gate, group in _ALL_RELATION_DECLARATIONS.items()
        if gate in _ENTRIES
    }
)


_RELATION_RATIONALES = {
    AtomDependencyLogic.SHORT_CIRCUIT: (
        "the reviewed source branch stops evaluating the declared later atoms"
    ),
    AtomDependencyLogic.EQUIVALENT: (
        "the reviewed scanner parent and child share one source expression"
    ),
    AtomDependencyLogic.TARGET_IMPLIES_PREREQUISITES: (
        "the reviewed target outcome necessarily changes its declared companions"
    ),
}


def _declared_relation(entry: GateEntry, atom: _AtomSpec) -> AtomRelation:
    try:
        canonical_indices, baseline_outcomes, declarations = _RELATION_DECLARATIONS[
            entry.gate_id
        ]
    except KeyError as error:
        raise AssertionError(
            f"{entry.gate_id} has atoms but no frozen relation group"
        ) from error
    if len(declarations) != len(entry.conjunction_atom_ids):
        raise AssertionError(f"{entry.gate_id} frozen relation count drifted")
    kind_name, target, canonical_index, prerequisite_rows, logic_name = declarations[
        atom.index
    ]
    canonical_ids = tuple(
        entry.conjunction_atom_ids[index] for index in canonical_indices
    )
    observed_canonical_ids = tuple(
        candidate
        for candidate in entry.conjunction_atom_ids
        if source_alias_canonical(entry, candidate) == candidate
    )
    if canonical_ids != observed_canonical_ids:
        raise AssertionError(
            f"{entry.gate_id} frozen canonical atom identities drifted"
        )
    if len(baseline_outcomes) != len(canonical_ids):
        raise AssertionError(f"{entry.gate_id} frozen baseline width drifted")
    canonical_atom_id = (
        None if canonical_index is None else entry.conjunction_atom_ids[canonical_index]
    )
    logic = None if logic_name is None else AtomDependencyLogic(logic_name)
    relation = AtomRelation(
        kind=AtomRelationKind(kind_name),
        baselines=tuple(
            AtomBaseline(atom_id, outcome)
            for atom_id, outcome in zip(canonical_ids, baseline_outcomes, strict=True)
        ),
        target_outcome=target,
        canonical_atom_id=canonical_atom_id,
        prerequisites=tuple(
            AtomPrerequisite(entry.conjunction_atom_ids[index], outcome)
            for index, outcome in prerequisite_rows
        ),
        logic=logic,
        rationale=None if logic is None else _RELATION_RATIONALES[logic],
    )
    ast_children = source_ast_prerequisites(
        entry, source_alias_canonical(entry, atom.atom_id)
    )
    if ast_children and not (
        set(ast_children) & {item.atom_id for item in relation.prerequisites}
    ):
        raise AssertionError(
            f"{atom.atom_id} frozen relation lost its source-AST child"
        )
    return relation


_DECLARED_ATOM_COUNT = sum(len(group[2]) for group in _RELATION_DECLARATIONS.values())
_EXPECTED_RELATION_GATES = {
    gate for gate, entry in _ENTRIES.items() if entry.conjunction_atom_ids
}
if set(_RELATION_DECLARATIONS) != _EXPECTED_RELATION_GATES:
    raise AssertionError("PLAN frozen relation gate partition drifted")
_EXPECTED_ATOM_COUNT = sum(
    len(entry.conjunction_atom_ids) for entry in _ENTRIES.values()
)
if _DECLARED_ATOM_COUNT != _EXPECTED_ATOM_COUNT:
    raise AssertionError(
        "PLAN frozen relation table has "
        f"{_DECLARED_ATOM_COUNT} rows, expected {_EXPECTED_ATOM_COUNT}"
    )


def _evidence_from_outcome(
    *,
    entry: GateEntry,
    outcome: _Outcome,
    lineage_key: str,
    oracle_name: str,
) -> tuple[tuple[AtomEvidence, ...], tuple[Any, ...], tuple[Any, ...]]:
    evidence: list[AtomEvidence] = []
    actual_truth: list[Any] = []
    oracle_truth: list[Any] = []
    for atom_id in entry.conjunction_atom_ids:
        actual_value = outcome.actual_atoms[atom_id]
        oracle_value = outcome.oracle_atoms[atom_id]
        evidence.append(
            AtomEvidence(
                atom_id=atom_id,
                raw_actual=actual_value.raw,
                truth=actual_value.truth,
                reducer=actual_value.reducer,
                realized_keys=(lineage_key,),
                oracle=oracle_name,
            )
        )
        actual_truth.append(actual_value.truth)
        oracle_truth.append(oracle_value.truth)
    return tuple(evidence), tuple(actual_truth), tuple(oracle_truth)


def _active_axis_name(entry: GateEntry, fixture: _Fixture) -> str:
    if len(entry.axes) == 1:
        return entry.axes[0].name
    names = {axis.name for axis in entry.axes}
    if fixture.axis_key not in names:
        raise AssertionError(
            f"{entry.gate_id} fixture axis {fixture.axis_key!r} is not one of "
            f"its real registry axes {sorted(names)!r}"
        )
    return fixture.axis_key


def _compound_axis_bindings(
    *,
    entry: GateEntry,
    fixture: _Fixture,
    point: ThresholdPoint,
    active_axis: str,
) -> Mapping[str, tuple[str, Any, AxisPosition]] | None:
    if len(entry.axes) == 1:
        return None
    bindings: dict[str, tuple[str, Any, AxisPosition]] = {}
    for axis in entry.axes:
        input_key = axis.name
        if input_key not in fixture.inputs:
            raise AssertionError(
                f"{entry.gate_id} compound axis {axis.name!r} has no real "
                "direct-input binding"
            )
        bindings[axis.name] = (
            input_key,
            fixture.inputs[input_key],
            axis_position_for_role(point.role)
            if axis.name == active_axis
            else AxisPosition.INTERIOR,
        )
    return MappingProxyType(bindings)


def _runner(
    entry: GateEntry, atom: _AtomSpec | None = None
) -> Callable[[ThresholdPoint], RawObservation]:
    def run(point: ThresholdPoint) -> RawObservation:
        fixture = _fixture(entry, point, atom)
        active_axis = _active_axis_name(entry, fixture)
        with (
            _capture_production_atoms(entry, enabled=atom is not None) as (
                recorder,
                _instrumented,
            ),
            _trace_required_calls(entry) as traced_calls,
        ):
            outcome = fixture.invoke(fixture.inputs)
        if atom is not None:
            if not _instrumented:
                raise AssertionError(f"{entry.gate_id} atom recorder was not installed")
            outcome = replace(
                outcome,
                actual_atoms=_captured_atom_row(entry, outcome, recorder),
            )
        observed_calls = tuple(dict.fromkeys(traced_calls))
        required_calls = REQUIRED_DIRECT_CALLS[entry.gate_id]
        if set(observed_calls) != required_calls:
            raise AssertionError(
                f"{entry.gate_id} direct-call trace differs from its declaration; "
                f"observed={sorted(observed_calls)!r}, "
                f"required={sorted(required_calls)!r}"
            )
        combined = dict(fixture.inputs)
        overlap = set(combined) & set(outcome.returns)
        if overlap:
            raise AssertionError(f"direct input/return overlap: {sorted(overlap)}")
        combined.update(outcome.returns)
        realized = realized_point(
            entry=entry,
            point=point,
            quantity=entry.quantity,
            input_key=fixture.point_key,
            value=combined[fixture.point_key],
            threshold=fixture.threshold,
            dtype=fixture.dtype,
            axis_input_key=fixture.axis_key,
            axis_value=fixture.inputs[fixture.axis_key],
            active_axis=active_axis,
            axis_bindings=_compound_axis_bindings(
                entry=entry,
                fixture=fixture,
                point=point,
                active_axis=active_axis,
            ),
        )
        actual_side = outcome.observed_side.value
        oracle_side = outcome.oracle_side.value
        checks = [
            oracle_check(
                oracle=fixture.oracle_name,
                actual=actual_side,
                expected=oracle_side,
            )
        ]
        for _label, actual_value, oracle_value, passed in outcome.payload_checks:
            checks.append(
                oracle_check(
                    oracle=fixture.oracle_name,
                    actual=passed,
                    expected=True,
                )
            )
        evidence: tuple[AtomEvidence, ...] = ()
        siblings = ()
        if atom is not None:
            evidence, actual_truth, oracle_truth = _evidence_from_outcome(
                entry=entry,
                outcome=outcome,
                lineage_key=fixture.point_key,
                oracle_name=fixture.oracle_name,
            )
            for atom_id in entry.conjunction_atom_ids:
                actual_atom = outcome.actual_atoms[atom_id]
                oracle_atom = outcome.oracle_atoms[atom_id]
                checks.append(
                    oracle_check(
                        oracle=fixture.oracle_name,
                        actual=actual_atom.raw,
                        expected=oracle_atom.raw,
                    )
                )
            siblings = (
                oracle_check(
                    oracle=fixture.oracle_name,
                    actual=actual_truth,
                    expected=oracle_truth,
                ),
            )
        return RawObservation(
            observed_side=outcome.observed_side,
            realized_point=realized,
            realized_inputs=combined,
            direct_input_keys=tuple(fixture.inputs),
            direct_return_keys=tuple(outcome.returns),
            direct_calls=observed_calls,
            oracle_checks=tuple(checks),
            isolated_atom=None if atom is None else atom.atom_id,
            atom_evidence=evidence,
            sibling_premises=siblings,
            notes=outcome.notes,
        )

    return run


def _build_suite(entry: GateEntry) -> BoundarySuite:
    gate = entry.gate_id
    topology = _topology(gate)
    family = (
        FixtureFamily.RHO_CERTIFICATE_TRACE_ORDER
        if gate
        in (
            _CERTIFICATE_GATES
            | _WARMUP_GATES
            | _AUDIT_GATES
            | _FACTORY_CERTIFICATE_GATES
            | _PLAN_FACTORY_GATES
        )
        else FixtureFamily.PLAN_SCALAR_PROOF_RANGE
    )
    methods = tuple(sorted(REQUIRED_DIRECT_CALLS[gate]))
    sample_fixture = _fixture(entry, _baseline_point(gate), None)
    baseline_active_axis = _active_axis_name(entry, sample_fixture)
    oracles_declared = (sample_fixture.oracle_name,)
    non_unit = (
        _NON_UNIT
        if entry.fixture_scale_policy is FixtureScalePolicy.NON_UNIT_REQUIRED
        else None
    )
    grid_cases = make_grid_cases(
        entry=entry,
        points=_points(gate),
        fixture_family=family,
        execution_class=ExecutionClass.VALIDATION_ONLY,
        topology=topology,
        direct_methods=methods,
        independent_oracles=oracles_declared,
        runner=_runner(entry),
        non_unit_scale=non_unit,
        active_axis=baseline_active_axis,
    )
    extra_grid_cases = tuple(
        make_case(
            entry=entry,
            point=point,
            fixture_family=family,
            execution_class=ExecutionClass.VALIDATION_ONLY,
            topology=topology,
            direct_methods=methods,
            independent_oracles=oracles_declared,
            runner=_runner(entry),
            non_unit_scale=non_unit,
            suffix=f"review-extreme-{index}",
            active_axis=_active_axis_name(entry, _fixture(entry, point, None)),
        )
        for index, point in enumerate(_review_extra_points(gate), start=1)
    )
    atom_cases = []
    atom_case_ids: dict[str, str] = {}
    for atom in _atom_specs(entry):
        atom_point = _atom_point(gate, atom)
        relation = _declared_relation(entry, atom)
        case = make_atom_case(
            entry=entry,
            atom_id=atom.atom_id,
            relation=relation,
            point=atom_point,
            fixture_family=family,
            execution_class=ExecutionClass.VALIDATION_ONLY,
            topology=topology,
            direct_methods=methods,
            independent_oracles=oracles_declared,
            runner=_runner(entry, atom),
            non_unit_scale=non_unit,
            active_axis=_active_axis_name(entry, _fixture(entry, atom_point, atom)),
        )
        atom_cases.append(case)
        atom_case_ids[atom.atom_id] = case.case_id
    near_boundary_roles = (
        PointRole.AT,
        PointRole.EXACT,
        PointRole.BELOW_ULP,
        PointRole.ABOVE_ULP,
        PointRole.BELOW_INTEGER,
        PointRole.ABOVE_INTEGER,
        PointRole.ULP_MISMATCH,
        PointRole.SUBNORMAL_MISMATCH,
        PointRole.CAPABILITY_LOW,
        PointRole.INVALID_CAPABILITY,
        PointRole.VALID_CAPABILITY,
        PointRole.CAPABILITY_HIGH,
        PointRole.BELOW_RELATIVE_1E12,
        PointRole.ABOVE_RELATIVE_1E12,
        PointRole.BELOW_RELATIVE_1E6,
        PointRole.ABOVE_RELATIVE_1E6,
        PointRole.MATERIAL_MISMATCH,
        PointRole.VERY_LOW,
        PointRole.VERY_HIGH,
        PointRole.EXTREME,
    )
    role_rank = {role: index for index, role in enumerate(near_boundary_roles)}
    selectable_cases = (*grid_cases, *extra_grid_cases)
    if gate in _FINITE_LIMIT_GATES:
        admitted = next(
            case
            for case in extra_grid_cases
            if case.threshold_point.display_value == "last finite float64"
        )
        refused = next(
            case
            for case in extra_grid_cases
            if case.threshold_point.display_value == "first float64 overflow"
        )
    else:
        admitted = min(
            (
                case
                for case in selectable_cases
                if case.threshold_point.expected_side is GateSide.ADMITTED
            ),
            key=lambda case: role_rank[case.threshold_point.role],
        )
        if gate in {
            "PLAN:runtime-call:scalar-and-dtype",
        }:
            refused = next(
                case
                for case in grid_cases
                if case.threshold_point.role is PointRole.EXTREME
            )
        elif gate == "PLAN:canonical-probes:runtime-finite":
            refused = next(
                case
                for case in extra_grid_cases
                if case.threshold_point.display_value
                == "post-cast nonfinite probe seam"
            )
        elif gate == "PLAN:frozen:probe-energy-range":
            refused = next(
                case
                for case in selectable_cases
                if case.active_axis == "probe_count"
                and case.threshold_point.role is PointRole.AT
            )
        elif gate == "PLAN:factory-certificate:order-and-rank":
            refused = next(
                case
                for case in selectable_cases
                if case.active_axis == "perturbation"
                and case.threshold_point.role is PointRole.ABOVE_INTEGER
            )
        elif gate == "PLAN:runtime:total-error-budget":
            refused = next(
                case
                for case in selectable_cases
                if case.active_axis == "tolerance"
                and case.threshold_point.role is PointRole.BELOW_ULP
            )
        elif gate == "PLAN:trace-factory:exact-evidence":
            refused = next(
                case
                for case in selectable_cases
                if case.active_axis == "trace_evidence_value"
                and case.threshold_point.role is PointRole.ABOVE_ULP
            )
        elif gate == "PLAN:measurement:x-norm-finite":
            refused = next(
                case
                for case in selectable_cases
                if case.active_axis == "perturbation_entry"
                and case.threshold_point.role is PointRole.ABOVE_ULP
            )
        else:
            refused = min(
                (
                    case
                    for case in selectable_cases
                    if case.threshold_point.expected_side is GateSide.REFUSED
                ),
                key=lambda case: role_rank[case.threshold_point.role],
            )
    return freeze_suite(
        gate_id=gate,
        fixture_family=family,
        execution_class=ExecutionClass.VALIDATION_ONLY,
        topology=topology,
        cases=(*grid_cases, *extra_grid_cases, *atom_cases),
        atom_case_ids=atom_case_ids,
        tighten_case_id=admitted.case_id,
        loosen_case_id=refused.case_id,
        ambiguities=(
            (_EXPECTED_NONFINITE_RESOURCE_AMBIGUITY,)
            if gate == "PLAN:runtime:expected-and-ulp-finite"
            else ()
        ),
        omitted_unrepresentable_roles=(
            _OMITTED_RELATIVE
            if gate
            in _ZERO_THRESHOLD_GATES
            | _MEASUREMENT_GATES
            | {
                "PLAN:audit:retained-trace-evidence",
                "PLAN:trace-factory:exact-evidence",
                "PLAN:runtime:total-error-budget",
            }
            else frozenset()
        ),
    )


PLAN_SUITES: tuple[BoundarySuite, ...] = tuple(
    _build_suite(_ENTRIES[gate]) for gate in sorted(_ENTRIES)
)


__all__ = ["PLAN_SUITES"]
