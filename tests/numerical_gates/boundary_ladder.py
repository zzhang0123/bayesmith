"""Direct-method boundary suites for the 21 reviewed LADDER gates.

Every runner owns a concrete fixture record.  Production call arguments are
read from that record, and measured returns/certificate fields are written back
before the common harness fingerprints them.  Atomic cases are keyed by the
scanner's exact source syntax rather than by an ordinal or a generic fallback.
"""

from __future__ import annotations

import ast
import copy
import inspect
import math
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from bayesmith.marginal import _logdet_eager as eager
from bayesmith.marginal import _logdet_ladder as ladder
from tests.numerical_gates import oracles
from tests.numerical_gates.boundary_contract import ALLOWED_DIRECT_CALLS
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
    RealizedAxis,
    RealizedPoint,
    ThresholdPoint,
    exact_grid,
    float_grid,
    freeze_suite,
    integer_grid,
    make_atom_case,
    make_case,
    make_grid_cases,
    numerical_evaluation,
    oracle_check,
    source_ast_prerequisites,
)
from tests.numerical_gates.registry import (
    GATE_REGISTRY,
    FixtureScalePolicy,
    MutationMode,
)
from tests.numerical_gates.source_scan import index_source_text

_ENTRIES = {
    entry.gate_id: entry
    for entry in GATE_REGISTRY
    if entry.mutation_mode is MutationMode.TWO_SIDED
    and entry.gate_id.startswith("LADDER:")
}
_SOURCE_MODULE = "src/bayesmith/marginal/_logdet_ladder.py"
_SOURCE = (Path(__file__).parents[2] / _SOURCE_MODULE).read_text()
_SOURCE_INDEX = index_source_text(_SOURCE, _SOURCE_MODULE)
_ATOM_SYNTAX = {
    atom_id: _SOURCE_INDEX[atom_id][0].syntax
    for entry in _ENTRIES.values()
    for atom_id in entry.conjunction_atom_ids
}


def _source_node_key(node: ast.AST) -> tuple[int, int, int, int]:
    return (
        int(node.lineno),
        int(node.col_offset),
        int(getattr(node, "end_lineno", node.lineno)),
        int(getattr(node, "end_col_offset", node.col_offset)),
    )


_ATOM_IDS_BY_SOURCE_NODE: dict[tuple[int, int, int, int], tuple[str, ...]] = {}
for _entry in _ENTRIES.values():
    for _atom_id in _entry.conjunction_atom_ids:
        _node_key = _source_node_key(_SOURCE_INDEX[_atom_id][1])
        _ATOM_IDS_BY_SOURCE_NODE[_node_key] = (
            *_ATOM_IDS_BY_SOURCE_NODE.get(_node_key, ()),
            _atom_id,
        )

_ACTIVE_SOURCE_RECORDS: ContextVar[dict[str, tuple[Any, dict[str, Any]]] | None] = (
    ContextVar("ladder_active_source_records", default=None)
)


class _PredicateRecorder(ast.NodeTransformer):
    """Wrap the registered source expressions without changing control flow."""

    def generic_visit(self, node: ast.AST) -> ast.AST:
        key = _source_node_key(node) if isinstance(node, ast.expr) else None
        visited = super().generic_visit(node)
        atom_ids = _ATOM_IDS_BY_SOURCE_NODE.get(key) if key is not None else None
        if not atom_ids:
            return visited
        wrapped = ast.Call(
            func=ast.Name(id="__boundary_record_predicate__", ctx=ast.Load()),
            args=[
                ast.Tuple(
                    elts=[ast.Constant(value=atom_id) for atom_id in atom_ids],
                    ctx=ast.Load(),
                ),
                visited,
                ast.Call(
                    func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]
                ),
            ],
            keywords=[],
        )
        return ast.copy_location(wrapped, node)


def _record_source_predicate(
    atom_ids: tuple[str, ...], raw: Any, local_values: dict[str, Any]
) -> Any:
    records = _ACTIVE_SOURCE_RECORDS.get()
    if records is not None:
        retained = np.array(raw, copy=True) if isinstance(raw, np.ndarray) else raw
        snapshot = dict(local_values)
        for atom_id in atom_ids:
            existing = records.get(atom_id)
            if existing is None:
                records[atom_id] = (retained, snapshot)
                continue
            previous = np.asarray(existing[0])
            current = np.asarray(retained)
            if previous.shape == () and current.shape == ():
                records[atom_id] = (
                    np.asarray([previous.item(), current.item()]),
                    snapshot,
                )
            else:
                records[atom_id] = (
                    np.concatenate((previous.reshape(-1), current.reshape(-1))),
                    snapshot,
                )
    return raw


def _instrumented_code_objects() -> dict[str, object]:
    tree = ast.parse(_SOURCE, filename=ladder.__file__)
    result: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start = int(node.lineno)
        end = int(getattr(node, "end_lineno", start))
        if not any(start <= key[0] <= end for key in _ATOM_IDS_BY_SOURCE_NODE):
            continue
        copied = copy.deepcopy(node)
        copied.decorator_list = []
        transformed = _PredicateRecorder().visit(copied)
        module = ast.fix_missing_locations(
            ast.Module(body=[transformed], type_ignores=[])
        )
        namespace = dict(ladder.__dict__)
        namespace["__boundary_record_predicate__"] = _record_source_predicate
        # The transformed AST is compiled only from the checked-in ladder source.
        exec(compile(module, ladder.__file__, "exec"), namespace)  # noqa: S102
        result[node.name] = namespace[node.name].__code__
    return result


_INSTRUMENTED_CODES = _instrumented_code_objects()
_LIVE_FUNCTIONS_AT_IMPORT = {
    name: getattr(ladder, name) for name in _INSTRUMENTED_CODES
}
_LIVE_CODES_AT_IMPORT = {
    name: function.__code__ for name, function in _LIVE_FUNCTIONS_AT_IMPORT.items()
}
_NON_UNIT = (1.3, 2.7)
_PAYLOAD_ORACLE = "independent dense/Decimal payload oracle"
_EQUALITY_WITNESS = "predicate equality face"
_ADJACENT_WITNESS = "predicate adjacent refusal"
_EXTRA_ORACLES: Mapping[str, tuple[str, ...]] = {
    "LADDER:rank:evidence": ("independent exact algebraic rank selection",),
    "LADDER:rho:measurement": ("independent retained spectral-radius measurement",),
    "LADDER:sigma:finite-two-sum": (
        "independent finite-payload measurement reachability",
        "independent completed premise evaluation",
    ),
}
_RELATIVE_ROLES = frozenset(
    {
        PointRole.BELOW_RELATIVE_1E6,
        PointRole.BELOW_RELATIVE_1E12,
        PointRole.ABOVE_RELATIVE_1E12,
        PointRole.ABOVE_RELATIVE_1E6,
    }
)


@dataclass(frozen=True, slots=True)
class _PremiseValue:
    raw_actual: Any
    raw_oracle: Any
    reducer: AtomReducer
    realized_keys: tuple[str, ...]


def _axis_position(role: PointRole) -> AxisPosition:
    if role in {PointRole.VERY_LOW, PointRole.CAPABILITY_LOW}:
        return AxisPosition.VERY_LOW
    if role in {PointRole.VERY_HIGH, PointRole.CAPABILITY_HIGH}:
        return AxisPosition.VERY_HIGH
    if role is PointRole.EXTREME:
        return AxisPosition.EXTREME
    if role in {
        PointRole.AT,
        PointRole.ABOVE_ULP,
        PointRole.ABOVE_INTEGER,
        PointRole.ABOVE_RELATIVE_1E12,
        PointRole.ABOVE_RELATIVE_1E6,
        PointRole.ULP_MISMATCH,
        PointRole.SUBNORMAL_MISMATCH,
        PointRole.MATERIAL_MISMATCH,
        PointRole.INVALID_CAPABILITY,
    }:
        return AxisPosition.ENDPOINT_HIGH
    return AxisPosition.ENDPOINT_LOW


def _float_value(
    point: ThresholdPoint,
    threshold: float,
    *,
    very_low: float,
    very_high: float,
    extreme: float,
) -> float:
    role = point.role
    if role is PointRole.VERY_LOW:
        return float(very_low)
    if role is PointRole.BELOW_RELATIVE_1E6:
        return float(threshold * (1.0 - 1e-6))
    if role is PointRole.BELOW_RELATIVE_1E12:
        return float(threshold * (1.0 - 1e-12))
    if role is PointRole.BELOW_ULP:
        return math.nextafter(float(threshold), -math.inf)
    if role is PointRole.AT:
        return float(threshold)
    if role is PointRole.ABOVE_ULP:
        return math.nextafter(float(threshold), math.inf)
    if role is PointRole.ABOVE_RELATIVE_1E12:
        return float(threshold * (1.0 + 1e-12))
    if role is PointRole.ABOVE_RELATIVE_1E6:
        return float(threshold * (1.0 + 1e-6))
    if role is PointRole.VERY_HIGH:
        return float(very_high)
    return float(extreme)


def _integer_value(
    point: ThresholdPoint,
    threshold: int,
    *,
    very_low: int,
    very_high: int,
    extreme: int,
) -> int:
    if point.role is PointRole.VERY_LOW:
        return int(very_low)
    if point.role is PointRole.BELOW_INTEGER:
        return threshold - 1
    if point.role is PointRole.AT:
        return threshold
    if point.role is PointRole.ABOVE_INTEGER:
        return threshold + 1
    if point.role is PointRole.VERY_HIGH:
        return int(very_high)
    return int(extreme)


def _exact_mismatch(point: ThresholdPoint) -> tuple[float, float]:
    if point.role is PointRole.EXACT:
        return 0.0, 0.0
    if point.role is PointRole.ULP_MISMATCH:
        return float(np.nextafter(0.0, -math.inf)), 0.0
    if point.role is PointRole.SUBNORMAL_MISMATCH:
        return float(np.nextafter(0.0, 1.0)), 0.0
    if point.role is PointRole.MATERIAL_MISMATCH:
        return 1e-6, 0.0
    if point.role is PointRole.VERY_LOW:
        return 1.3, 2.7
    if point.role is PointRole.VERY_HIGH:
        return 2.7, 1.3
    return float(np.finfo(np.float64).max), 0.0


_AXIS_POSITION_INDEX: Mapping[AxisPosition, int] = {
    AxisPosition.VERY_LOW: 0,
    AxisPosition.ENDPOINT_LOW: 1,
    AxisPosition.ENDPOINT_HIGH: 2,
    AxisPosition.VERY_HIGH: 3,
    AxisPosition.EXTREME: 4,
}


def _axis_choice(record: _FixtureRecord, values: tuple[Any, Any, Any, Any, Any]) -> Any:
    """Select one concrete input leaf for the active five-position region."""

    return values[_AXIS_POSITION_INDEX[_axis_position(record.point.role)]]


def _secondary_axis(record: _FixtureRecord) -> str | None:
    """Return a non-primary grid axis, excluding atoms and mutation witnesses."""

    if record.atom_id is not None or record.active_axis is None:
        return None
    if record.point.display_value in {_EQUALITY_WITNESS, _ADJACENT_WITNESS}:
        return None
    primary = _ENTRIES[record.gate_id].axes[0].name
    return None if record.active_axis == primary else record.active_axis


_SECONDARY_AXIS_SIDES: Mapping[tuple[str, str], tuple[GateSide, ...]] = {
    (
        "LADDER:structure:diagonal-tolerance",
        "off_diagonal",
    ): (
        GateSide.REFUSED,
        GateSide.ADMITTED,
        GateSide.REFUSED,
        GateSide.REFUSED,
        GateSide.REFUSED,
    ),
}


def _secondary_axis_points(gate_id: str, axis_name: str) -> tuple[ThresholdPoint, ...]:
    """Five capability positions whose gate side is supplied by a valid fixture."""

    sides = _SECONDARY_AXIS_SIDES.get(
        (gate_id, axis_name),
        (GateSide.ADMITTED,) * 5,
    )
    return (
        ThresholdPoint(
            PointRole.CAPABILITY_LOW,
            "secondary-axis low capability",
            "axis-low",
            sides[0],
        ),
        ThresholdPoint(
            PointRole.VALID_CAPABILITY,
            "secondary-axis endpoint low",
            "endpoint-low",
            sides[1],
        ),
        ThresholdPoint(
            PointRole.INVALID_CAPABILITY,
            "secondary-axis endpoint high",
            "endpoint-high",
            sides[2],
        ),
        ThresholdPoint(
            PointRole.CAPABILITY_HIGH,
            "secondary-axis high capability",
            "axis-high",
            sides[3],
        ),
        ThresholdPoint(
            PointRole.EXTREME,
            "secondary-axis representable extreme",
            "axis-extreme",
            sides[4],
        ),
    )


def _array_descriptor(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "dtype": array.dtype.str,
        "ndim": int(array.ndim),
        "shape": tuple(int(item) for item in array.shape),
        "strides": tuple(int(item) for item in array.strides),
    }


def _axis_input(
    record: _FixtureRecord,
    key: str,
    value: Any,
    *,
    audit_value: Any | None = None,
) -> str:
    record.input(key, value, audit_value=audit_value)
    return key


def _finish_axis_record(
    record: _FixtureRecord,
    *,
    observed: bool,
    oracle: bool,
    return_key: str,
    return_value: Any,
    axis_keys: Mapping[str, str],
    threshold: Any,
    dtype: str | None = None,
) -> RawObservation:
    record.returned(return_key, return_value)
    record.observed_side = _side(observed)
    record.oracle_side = _side(oracle)
    record.bind_axes(**dict(axis_keys))
    active = record.active_axis
    if active is None:
        if len(axis_keys) != 1:
            raise AssertionError(f"{record.gate_id} did not select an active axis")
        active = next(iter(axis_keys))
    record.realization(
        point_key=return_key,
        value=return_value,
        threshold=threshold,
        axis_key=axis_keys[active],
        dtype=dtype,
    )
    return record.finish()


def _lineage_contains(container: Any, candidate: Any) -> bool:
    """Return whether ``candidate`` is a concrete leaf of a call argument."""

    if container is candidate:
        return True
    if isinstance(container, np.ndarray):
        if isinstance(candidate, Mapping):
            descriptor = {
                "dtype": container.dtype.str,
                "ndim": int(container.ndim),
                "shape": tuple(int(item) for item in container.shape),
                "strides": tuple(int(item) for item in container.strides),
            }
            return all(descriptor.get(str(key)) == value for key, value in candidate.items())
        if isinstance(candidate, str):
            return candidate == container.dtype.str
        if isinstance(candidate, np.ndarray):
            return bool(
                container.dtype == candidate.dtype
                and container.shape == candidate.shape
                and np.array_equal(container, candidate)
            )
        if isinstance(candidate, bool):
            return bool(
                container.dtype == np.dtype(bool) and np.any(container == candidate)
            )
        if isinstance(candidate, (int, np.integer)):
            return bool(
                int(candidate) in container.shape or np.any(container == candidate)
            )
        if isinstance(candidate, (float, np.floating)):
            return bool(np.any(container == candidate))
        return False
    if isinstance(container, eager.LogDetProblem):
        return any(
            _lineage_contains(getattr(container, name), candidate)
            for name in (
                "lambda_matrix",
                "perturbation",
                "chain_block_size",
                "structure_kind",
                "structure",
                "low_rank_factors",
                "exact_power_traces",
                "frozen_probes",
                "trace_order",
                "certified_rho",
            )
        )
    if isinstance(container, eager.LadderConfig):
        return any(
            _lineage_contains(getattr(container, name), candidate)
            for name in (
                "low_rank_max",
                "low_rank_fraction",
                "dense_max_n",
                "finite_max_n",
                "finite_max_rank",
                "structure_rtol",
                "structure_atol",
            )
        )
    if isinstance(container, eager.LowRankFactors):
        if isinstance(candidate, Mapping):
            descriptor = {
                "present": True,
                "rank": int(container.rank_bound),
                "left_shape": tuple(int(item) for item in container.left.shape),
                "right_shape": tuple(int(item) for item in container.right.shape),
            }
            if all(
                descriptor.get(str(key)) == value
                for key, value in candidate.items()
            ):
                return True
        return _lineage_contains(container.left, candidate) or _lineage_contains(
            container.right, candidate
        )
    if isinstance(container, eager.KroneckerStructure):
        if isinstance(candidate, Mapping):
            descriptor = {
                "present": True,
                "factor_count": len(container.factors),
                "factor_shapes": tuple(
                    tuple(int(item) for item in factor.shape)
                    for factor in container.factors
                ),
            }
            if all(
                descriptor.get(str(key)) == value
                for key, value in candidate.items()
            ):
                return True
        return any(_lineage_contains(item, candidate) for item in container.factors)
    if isinstance(container, eager.FrozenProbes):
        return _lineage_contains(container.values, candidate)
    if isinstance(container, Mapping):
        return any(_lineage_contains(value, candidate) for value in container.values())
    if isinstance(container, (tuple, list)):
        if (
            isinstance(candidate, type(container))
            and len(container) == len(candidate)
            and all(
                _lineage_contains(left, right)
                for left, right in zip(container, candidate, strict=True)
            )
        ):
            return True
        return any(_lineage_contains(item, candidate) for item in container)
    if isinstance(container, bool) or isinstance(candidate, bool):
        return type(container) is type(candidate) and container == candidate
    if isinstance(container, (int, np.integer)) and isinstance(
        candidate, (int, np.integer)
    ):
        return int(container) == int(candidate)
    if isinstance(container, (float, np.floating)) and isinstance(
        candidate, (float, np.floating)
    ):
        return float(container).hex() == float(candidate).hex()
    return type(container) is type(candidate) and container == candidate


def _ast_operand_values(
    node: ast.AST, local_values: Mapping[str, Any]
) -> tuple[Any, ...]:
    """Resolve source operands without executing calls or product helpers."""

    parents = {
        child: parent
        for parent in ast.walk(node)
        for child in ast.iter_child_nodes(parent)
    }
    values: list[Any] = []

    def resolve(candidate: ast.AST) -> Any:
        if isinstance(candidate, ast.Name):
            return local_values.get(candidate.id, _MISSING)
        if isinstance(candidate, ast.Attribute):
            owner = resolve(candidate.value)
            if owner is _MISSING:
                return _MISSING
            return getattr(owner, candidate.attr, _MISSING)
        if isinstance(candidate, ast.Constant):
            return candidate.value
        return _MISSING

    for candidate in ast.walk(node):
        if not isinstance(candidate, (ast.Name, ast.Attribute)):
            continue
        parent = parents.get(candidate)
        if isinstance(parent, ast.Attribute) and parent.value is candidate:
            continue
        if isinstance(parent, ast.Call) and parent.func is candidate:
            continue
        value = resolve(candidate)
        if value is not _MISSING:
            values.append(value)
    return tuple(values)


_MISSING = object()


@dataclass(slots=True)
class _FixtureRecord:
    gate_id: str
    point: ThresholdPoint
    atom_id: str | None
    active_axis: str | None = None
    values: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    direct_input_keys: list[str] = field(default_factory=list)
    direct_return_keys: list[str] = field(default_factory=list)
    direct_calls: list[str] = field(default_factory=list)
    checks: list[Any] = field(default_factory=list)
    evaluations: list[Any] = field(default_factory=list)
    premises: dict[str, _PremiseValue] = field(default_factory=dict)
    observed_side: GateSide | None = None
    oracle_side: GateSide | None = None
    point_key: str | None = None
    point_value: Any = None
    threshold: Any = None
    dtype: str | None = None
    axis_key: str | None = None
    axis_keys: dict[str, str] = field(default_factory=dict)
    axis_carrier_key: str | None = None
    axis_companion_key: str | None = None
    notes: list[str] = field(default_factory=list)
    source_records: dict[str, tuple[Any, dict[str, Any]]] = field(default_factory=dict)
    call_arguments: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    instrumentation_intact: bool = True

    @property
    def atom_syntax(self) -> str | None:
        return None if self.atom_id is None else _ATOM_SYNTAX[self.atom_id]

    def input(self, key: str, value: Any, *, audit_value: Any | None = None) -> Any:
        self.values[key] = value
        retained = value if audit_value is None else audit_value
        self.audit[key] = retained
        return value

    def raw(self, key: str, value: Any) -> Any:
        self.values[key] = value
        return value

    def invoke(
        self,
        method: str,
        function: Callable[..., Any],
        *,
        result_key: str,
        args: tuple[str, ...] = (),
        kwargs: Mapping[str, str] | None = None,
    ) -> Any:
        module_name, separator, attribute = method.partition(".")
        owner = {"eager": eager, "ladder": ladder}.get(module_name)
        expected = (
            None if owner is None or not separator else getattr(owner, attribute, None)
        )
        if function is not expected:
            raise AssertionError(
                f"{method} callable identity is not the live production attribute"
            )
        keyword_keys = {} if kwargs is None else dict(kwargs)
        positional_values = tuple(self.values[key] for key in args)
        keyword_values = {name: self.values[key] for name, key in keyword_keys.items()}
        bound = inspect.signature(function).bind(*positional_values, **keyword_values)
        self.call_arguments.append((method, dict(bound.arguments)))
        actual_arguments = (*positional_values, *keyword_values.values())
        for key, candidate in self.audit.items():
            if key in self.direct_return_keys or key in self.direct_input_keys:
                continue
            if any(
                _lineage_contains(argument, candidate) for argument in actual_arguments
            ):
                self.direct_input_keys.append(key)
        if method not in self.direct_calls:
            # Record the attempted production call before invoking it so an
            # expected exception path cannot be labelled by hand afterward.
            self.direct_calls.append(method)
        result = function(
            *positional_values,
            **keyword_values,
        )
        self.values[result_key] = result
        return result

    def returned(self, key: str, value: Any) -> Any:
        self.values[key] = value
        self.audit[key] = value
        if key not in self.direct_return_keys:
            self.direct_return_keys.append(key)
        return value

    def realization(
        self,
        *,
        point_key: str,
        value: Any,
        threshold: Any,
        axis_key: str,
        dtype: str | None = None,
    ) -> None:
        self.point_key = point_key
        self.point_value = value
        self.threshold = threshold
        self.axis_key = axis_key
        self.dtype = dtype

    def bind_axes(self, **axis_keys: str) -> None:
        """Bind every exact registry axis to a consumed direct-input key."""

        self.axis_keys.update(axis_keys)

    def finish(self) -> RawObservation:
        if self.observed_side is None or self.oracle_side is None:
            raise AssertionError(f"{self.gate_id} did not record both gate sides")
        if self.point_key is None or self.axis_key is None:
            raise AssertionError(f"{self.gate_id} did not retain a realized point")
        self.checks.insert(
            0,
            oracle_check(
                oracle="independent gate predicate",
                actual=self.observed_side.value,
                expected=self.oracle_side.value,
            ),
        )
        atom_evidence: list[AtomEvidence] = []
        premise_checks: list[Any] = []
        if self.atom_id is not None:
            for atom_id in _ENTRIES[self.gate_id].conjunction_atom_ids:
                syntax = _ATOM_SYNTAX[atom_id]
                if syntax not in self.premises:
                    raise AssertionError(
                        f"{self.gate_id} has no source-premise result for {syntax!r}"
                    )
                premise = self.premises[syntax]
                oracle_name = f"source premise: {syntax}"
                premise_checks.append(
                    oracle_check(
                        oracle=oracle_name,
                        actual=premise.raw_actual,
                        expected=premise.raw_oracle,
                    )
                )
                atom_evidence.append(
                    AtomEvidence(
                        atom_id=atom_id,
                        raw_actual=premise.raw_actual,
                        truth=_premise_truth(premise.raw_actual, premise.reducer),
                        reducer=premise.reducer,
                        realized_keys=premise.realized_keys,
                        oracle=oracle_name,
                    )
                )
        entry = _ENTRIES[self.gate_id]
        if len(entry.axes) == 1:
            axes = (
                RealizedAxis(
                    entry.axes[0].name,
                    _axis_position(self.point.role),
                    self.axis_key,
                    self.audit[self.axis_key],
                ),
            )
            active_axis = entry.axes[0].name
        else:
            expected_axes = {axis.name for axis in entry.axes}
            if self.active_axis not in expected_axes:
                raise AssertionError(
                    f"{self.gate_id} did not select one exact active axis"
                )
            if set(self.axis_keys) != expected_axes:
                raise AssertionError(
                    f"{self.gate_id} axis bindings differ from registry: "
                    f"expected={sorted(expected_axes)!r}, "
                    f"actual={sorted(self.axis_keys)!r}"
                )
            axis_keys = self.axis_keys
            if (
                self.atom_id is None
                and self.axis_carrier_key is not None
                and self.axis_companion_key is not None
            ):
                axis_keys = {
                    name: (
                        self.axis_carrier_key
                        if name == self.active_axis
                        else self.axis_companion_key
                    )
                    for name in expected_axes
                }
            axes = tuple(
                RealizedAxis(
                    axis.name,
                    (
                        _axis_position(self.point.role)
                        if axis.name == self.active_axis
                        else AxisPosition.INTERIOR
                    ),
                    axis_keys[axis.name],
                    self.audit[axis_keys[axis.name]],
                )
                for axis in entry.axes
            )
            active_axis = self.active_axis
        realized = RealizedPoint(
            quantity=entry.quantity,
            input_key=self.point_key,
            value=self.point_value,
            threshold=self.threshold,
            dtype=self.dtype,
            axes=axes,
            active_axis=active_axis,
        )
        return RawObservation(
            observed_side=self.observed_side,
            realized_point=realized,
            realized_inputs=self.audit,
            direct_input_keys=tuple(self.direct_input_keys),
            direct_return_keys=tuple(self.direct_return_keys),
            direct_calls=tuple(self.direct_calls),
            oracle_checks=tuple(self.checks),
            evaluations=tuple(self.evaluations),
            isolated_atom=self.atom_id,
            atom_evidence=tuple(atom_evidence),
            sibling_premises=tuple(premise_checks),
            notes=tuple(self.notes),
        )


@contextmanager
def _recording_production_predicates(record: _FixtureRecord) -> Any:
    """Record live source operands while preserving an outer mutation seam.

    A source mutant loaded before this provider is compiled into the recorder.
    If an in-process test wraps a callable afterward, instrument the production
    function captured by that wrapper instead of replacing the outer callable.
    """

    active: dict[str, tuple[Any, dict[str, Any]]] = {}
    token = _ACTIVE_SOURCE_RECORDS.set(active)
    missing = object()
    recorder_name = "__boundary_record_predicate__"
    previous_recorder = ladder.__dict__.get(recorder_name, missing)
    originals: list[tuple[Callable[..., Any], object, object]] = []
    try:
        ladder.__dict__[recorder_name] = _record_source_predicate
        for name, instrumented_code in _INSTRUMENTED_CODES.items():
            function = _LIVE_FUNCTIONS_AT_IMPORT[name]
            if (
                not callable(function)
                or not hasattr(function, "__code__")
                or function.__code__ is not _LIVE_CODES_AT_IMPORT[name]
            ):
                record.instrumentation_intact = False
                continue
            originals.append((function, function.__code__, instrumented_code))
            function.__code__ = instrumented_code
        yield
    finally:
        for function, original_code, instrumented_code in reversed(originals):
            if function.__code__ is instrumented_code:
                function.__code__ = original_code
            else:
                record.instrumentation_intact = False
        if previous_recorder is missing:
            ladder.__dict__.pop(recorder_name, None)
        else:
            ladder.__dict__[recorder_name] = previous_recorder
        record.source_records.update(active)
        _ACTIVE_SOURCE_RECORDS.reset(token)


@contextmanager
def _suspend_source_recording() -> Any:
    token = _ACTIVE_SOURCE_RECORDS.set(None)
    try:
        yield
    finally:
        _ACTIVE_SOURCE_RECORDS.reset(token)


@dataclass(frozen=True, slots=True)
class _Facts:
    sigma: np.ndarray
    sigma_formation_valid: bool
    sigma_exactly_symmetric: bool
    sigma_symmetric: bool
    sigma_spd: bool
    condition_resolved: bool
    rank: int
    rank_evidence_valid: bool
    rho_measurement_valid: bool
    actual_rho: float
    finite_payload_rho: float
    finite_payload_rho_measurement_valid: bool
    finite_polynomial_stable: bool
    determinant_lemma_payload: bool
    dense_arithmetic_resolved: bool
    compact_diagonal_payload: bool
    chain_structure: bool
    structured: bool
    finite_size_qualified: bool
    finite_payload_stable: bool
    traces_verified: bool
    measured_rho_converges: bool
    rho_covers_input: bool
    frozen_width_valid: bool


def _independent_facts(
    problem: eager.LogDetProblem, config: eager.LadderConfig
) -> _Facts:
    lam = np.asarray(problem.lambda_matrix)
    perturb = np.asarray(problem.perturbation)
    try:
        with np.errstate(over="raise", invalid="raise"):
            sigma = np.add(lam, perturb)
        sigma_formation_valid = bool(np.all(np.isfinite(sigma)))
    except FloatingPointError:
        sigma = np.full_like(lam, math.inf, dtype=float)
        sigma_formation_valid = False
    sigma_exact = bool(sigma.ndim == 1 or np.array_equal(sigma, sigma.T))
    sigma_symmetric = bool(
        sigma_formation_valid
        and (
            sigma.ndim == 1
            or oracles.tolerant_symmetry(
                sigma,
                relative_tolerance=config.structure_rtol,
                absolute_tolerance=config.structure_atol,
            )
        )
    )
    symmetric_sigma = (
        sigma
        if sigma.ndim == 1 or sigma_exact or not sigma_symmetric
        else 0.5 * (sigma + sigma.T)
    )
    if not sigma_formation_valid or not sigma_symmetric:
        sigma_spd = False
        condition_resolved = False
    elif sigma.ndim == 1:
        sigma_spd = bool(np.all(sigma > 0.0))
        condition = float(np.max(sigma) / np.min(sigma)) if sigma_spd else math.inf
        condition_resolved = bool(
            math.isfinite(condition) and condition < 1.0 / np.finfo(sigma.dtype).eps
        )
    else:
        sigma_spd = oracles.symmetric_is_positive_definite(symmetric_sigma)
        condition = (
            oracles.spectral_condition(symmetric_sigma) if sigma_spd else math.inf
        )
        condition_resolved = bool(
            math.isfinite(condition) and condition < 1.0 / np.finfo(sigma.dtype).eps
        )
    n = int(lam.shape[-1])
    if problem.low_rank_factors is None:
        rank = (
            int(np.count_nonzero(perturb != 0.0))
            if perturb.ndim == 1
            else int(perturb.shape[0])
        )
        rank_valid = True
    else:
        factors = problem.low_rank_factors
        reconstructed = np.asarray(factors.left) @ np.asarray(factors.right).T
        rank_valid = bool(np.array_equal(reconstructed, perturb))
        rank = int(factors.rank_bound if rank_valid else n)
    try:
        with np.errstate(over="ignore"):
            actual_rho = oracles.spectral_radius(lam, perturb)
        rho_valid = math.isfinite(actual_rho)
    except (ValueError, np.linalg.LinAlgError):
        actual_rho = math.inf
        rho_valid = False
    if sigma_formation_valid:
        finite_perturbation = symmetric_sigma - lam
        try:
            with np.errstate(over="ignore"):
                finite_rho = oracles.spectral_radius(lam, finite_perturbation)
            finite_rho_valid = math.isfinite(finite_rho)
        except (ValueError, np.linalg.LinAlgError):
            finite_rho = math.inf
            finite_rho_valid = False
    else:
        finite_rho = math.inf
        finite_rho_valid = False
    finite_poly = finite_rho_valid and finite_rho <= 1.0
    determinant = bool(
        problem.low_rank_factors is not None
        and rank_valid
        and sigma_formation_valid
        and sigma_exact
        and condition_resolved
    )
    dense_resolved = sigma_formation_valid and (sigma.ndim == 1 or condition_resolved)
    compact = perturb.ndim == 1
    chain_structure = bool(
        sigma_formation_valid
        and problem.chain_block_size is not None
        and sigma.ndim == 2
        and oracles.is_block_chain(sigma, int(problem.chain_block_size))
    )
    if not sigma_formation_valid:
        structured = False
    elif sigma.ndim == 1:
        structured = bool(np.all(sigma > 0.0))
    elif problem.structure_kind == "diagonal":
        structured = oracles.is_diagonal(sigma)
    elif problem.structure_kind == "circulant":
        structured = oracles.is_circulant(sigma)
        if structured:
            try:
                eigenvalues = oracles.circulant_eigenvalues(sigma)
            except ValueError:
                structured = False
            else:
                structured = bool(np.all(eigenvalues > 0.0))
    elif problem.structure_kind == "toeplitz":
        structured = oracles.is_toeplitz(sigma)
    elif problem.structure_kind == "kronecker" and problem.structure is not None:
        factors = tuple(np.asarray(value) for value in problem.structure.factors)
        reconstructed = oracles.explicit_kronecker(factors)
        structured = bool(
            all(oracles.symmetric_is_positive_definite(value) for value in factors)
            and reconstructed.shape == sigma.shape
            and np.array_equal(reconstructed, sigma)
        )
    else:
        structured = False
    structured = (
        structured
        and sigma_spd
        and (
            sigma.ndim == 1
            or problem.structure_kind == "diagonal"
            or condition_resolved
        )
    )
    finite_size = n <= config.finite_max_n or (
        (compact or determinant) and rank <= config.finite_max_rank
    )
    finite_stable = finite_poly or determinant
    rho = actual_rho if problem.certified_rho is None else problem.certified_rho
    traces_verified = bool(
        rho_valid
        and problem.exact_power_traces is not None
        and problem.trace_order is not None
        and problem.trace_order >= 0
        and tuple(problem.exact_power_traces)
        == oracles.exact_power_traces(lam, perturb, int(problem.trace_order))
    )
    frozen_width = bool(
        problem.frozen_probes is not None and problem.frozen_probes.values.shape[1] == n
    )
    return _Facts(
        sigma=np.asarray(sigma),
        sigma_formation_valid=sigma_formation_valid,
        sigma_exactly_symmetric=sigma_exact,
        sigma_symmetric=sigma_symmetric,
        sigma_spd=sigma_spd,
        condition_resolved=condition_resolved,
        rank=rank,
        rank_evidence_valid=rank_valid,
        rho_measurement_valid=rho_valid,
        actual_rho=actual_rho,
        finite_payload_rho=finite_rho,
        finite_payload_rho_measurement_valid=finite_rho_valid,
        finite_polynomial_stable=finite_poly,
        determinant_lemma_payload=determinant,
        dense_arithmetic_resolved=dense_resolved,
        compact_diagonal_payload=compact,
        chain_structure=chain_structure,
        structured=structured,
        finite_size_qualified=finite_size,
        finite_payload_stable=finite_stable,
        traces_verified=traces_verified,
        measured_rho_converges=rho_valid and actual_rho < 1.0,
        rho_covers_input=rho_valid and actual_rho <= float(rho),
        frozen_width_valid=frozen_width,
    )


def _problem(
    record: _FixtureRecord,
    lambda_matrix: Any,
    perturbation: Any,
    **kwargs: Any,
) -> eager.LogDetProblem:
    built = eager.LogDetProblem(lambda_matrix, perturbation, **kwargs)
    record.input("lambda_matrix", np.asarray(built.lambda_matrix))
    record.input("perturbation", np.asarray(built.perturbation))
    for key in kwargs:
        retained = getattr(built, key)
        record.values[key] = retained
        if key == "low_rank_factors" and retained is not None:
            record.input("factor_left", np.asarray(retained.left))
            record.input("factor_right", np.asarray(retained.right))
        elif key == "structure" and retained is not None:
            for index, factor in enumerate(retained.factors):
                record.input(f"structure_factor_{index}", np.asarray(factor))
        elif key == "frozen_probes" and retained is not None:
            record.input("frozen_probe_values", np.asarray(retained.values))
        elif key == "exact_power_traces" and retained is not None:
            record.input(
                "exact_power_traces", tuple(float(item) for item in retained)
            )
        else:
            record.input(key, retained)
    return record.raw("problem", built)


def _config(record: _FixtureRecord, **kwargs: Any) -> eager.LadderConfig:
    compound_grid = bool(
        record.atom_id is None
        and record.active_axis is not None
        and len(_ENTRIES[record.gate_id].axes) > 1
    )
    carrier_field: str | None = None
    if compound_grid:
        carrier_value = float(_axis_choice(record, (0.125, 0.25, 0.375, 0.5, 0.625)))
        carrier_field = (
            "low_rank_fraction"
            if "low_rank_fraction" not in kwargs
            else "structure_atol"
        )
        kwargs[carrier_field] = carrier_value
    config = eager.LadderConfig(**kwargs)
    retained = {
        "low_rank_max": config.low_rank_max,
        "low_rank_fraction": config.low_rank_fraction,
        "dense_max_n": config.dense_max_n,
        "finite_max_n": config.finite_max_n,
        "finite_max_rank": config.finite_max_rank,
        "structure_rtol": config.structure_rtol,
        "structure_atol": config.structure_atol,
    }
    for key, value in retained.items():
        record.input(f"config_{key}", value)
    if carrier_field is not None:
        record.axis_carrier_key = f"config_{carrier_field}"
        record.axis_companion_key = "config_structure_rtol"
    return record.raw("config", config)


def _check_premises(record: _FixtureRecord) -> tuple[eager.PremiseVerdict, ...]:
    verdicts = record.invoke(
        "ladder.check_logdet_premises",
        ladder.check_logdet_premises,
        result_key="premise_verdicts",
        args=("problem",),
        kwargs={"config": "config"},
    )
    return tuple(verdicts)


def _side(value: bool) -> GateSide:
    return GateSide.ADMITTED if value else GateSide.REFUSED


def _payload_oracle(problem: eager.LogDetProblem) -> float:
    sigma = np.asarray(problem.lambda_matrix) + np.asarray(problem.perturbation)
    if sigma.ndim == 1:
        return oracles.diagonal_logdet(sigma)
    if sigma.shape[0] <= 3:
        return oracles.decimal_logdet(sigma)
    sign, value = np.linalg.slogdet(sigma)
    if sign <= 0.0 or not math.isfinite(float(value)):
        raise ValueError("the independent dense oracle needs positive determinant")
    return float(value)


def _independent_symmetric_representative(value: np.ndarray) -> np.ndarray:
    """Compute the pairwise representative with exact-binary Decimal means."""

    matrix = np.asarray(value)
    if matrix.ndim == 1 or np.array_equal(matrix, matrix.T):
        return np.array(matrix, copy=True)
    result = np.array(matrix, copy=True)
    with localcontext() as context:
        context.prec = 2500
        two = Decimal(2)
        for row in range(matrix.shape[0]):
            for column in range(row + 1, matrix.shape[1]):
                mean = (
                    Decimal.from_float(float(matrix[row, column]))
                    + Decimal.from_float(float(matrix[column, row]))
                ) / two
                rounded = float(mean)
                result[row, column] = rounded
                result[column, row] = rounded
    return result


def _dense_logdet_value(matrix: np.ndarray) -> float:
    value = np.asarray(matrix)
    if value.ndim == 1:
        return float(np.sum(np.log(value)))
    sign, logdet = np.linalg.slogdet(value)
    if sign <= 0.0 or not math.isfinite(float(logdet)):
        raise ValueError("selected Sigma has no finite positive determinant")
    return float(logdet)


def _independent_matrix_logdet(matrix: np.ndarray) -> float:
    value = np.asarray(matrix)
    if value.ndim == 1:
        return oracles.diagonal_logdet(value)
    if value.shape[0] <= 3:
        return oracles.decimal_logdet(value)
    sign, result = np.linalg.slogdet(value)
    if sign <= 0.0 or not math.isfinite(float(result)):
        raise ValueError("the independent matrix oracle needs positive determinant")
    return float(result)


def _record_payload(
    record: _FixtureRecord,
    *,
    method: str,
    function: Callable[..., float],
    args: tuple[str, ...],
    kwargs: Mapping[str, str] | None = None,
    oracle_value: float | None = None,
) -> float:
    actual = float(
        record.invoke(
            method,
            function,
            result_key=f"payload::{method}",
            args=args,
            kwargs=kwargs,
        )
    )
    record.returned(f"payload_value::{method}", actual)
    expected = (
        _payload_oracle(record.values["problem"])
        if oracle_value is None
        else oracle_value
    )
    record.evaluations.append(
        numerical_evaluation(
            method=method,
            oracle=_PAYLOAD_ORACLE,
            actual=actual,
            oracle_value=expected,
        )
    )
    return actual


def _record_payload_attempt(
    record: _FixtureRecord,
    *,
    method: str,
    function: Callable[..., float],
    args: tuple[str, ...],
    kwargs: Mapping[str, str] | None = None,
    oracle_value: float | None = None,
) -> float | None:
    """Attempt the real candidate payload and retain a product exception."""

    try:
        return _record_payload(
            record,
            method=method,
            function=function,
            args=args,
            kwargs=kwargs,
            oracle_value=oracle_value,
        )
    except (
        FloatingPointError,
        np.linalg.LinAlgError,
        OverflowError,
        TypeError,
        ValueError,
    ) as error:
        record.notes.append(
            f"direct-call exception: {method}: {type(error).__name__}: {error}"
        )
        return None


_DIRECT_PRODUCT_EXCEPTIONS = (
    FloatingPointError,
    np.linalg.LinAlgError,
    OverflowError,
    TypeError,
    ValueError,
)


def _capture_direct_product(
    record: _FixtureRecord,
    *,
    side: str,
    method: str,
    function: Callable[..., float],
    args: tuple[str, ...],
    kwargs: Mapping[str, str] | None = None,
) -> float | None:
    """Run one real side payload and retain its value or typed exception."""

    key = f"direct-product::{side}::{method}"
    try:
        value = float(
            record.invoke(
                method,
                function,
                result_key=f"raw::{key}",
                args=args,
                kwargs=kwargs,
            )
        )
    except _DIRECT_PRODUCT_EXCEPTIONS as error:
        record.returned(
            key,
            {
                "status": "exception",
                "method": method,
                "exception_type": type(error).__name__,
                "message": str(error),
            },
        )
        return None
    record.returned(
        key,
        {
            "status": "returned",
            "method": method,
            "value": value,
        },
    )
    return value


def _record_selected_product(
    record: _FixtureRecord,
    *,
    production_admitted: bool,
    oracle_admitted: bool,
    admitted_method: str,
    admitted_value: float | None,
    refused_method: str,
    refused_value: float | None,
    oracle_value: float,
) -> None:
    """Grade the live side selection without routing through the dispatcher."""

    if production_admitted:
        selected_method = admitted_method
        selected_value = admitted_value
    else:
        selected_method = refused_method
        selected_value = refused_value
    if production_admitted != oracle_admitted:
        # The live predicate selected a contract-invalid method (loosen) or
        # refused the contract-valid product (tighten).  Both real side payload
        # results remain retained; the selected contract outcome has no valid
        # scalar and is therefore graded as non-finite against dense truth.
        selected_value = math.nan
    if selected_value is None:
        return
    record.evaluations.append(
        numerical_evaluation(
            method=f"ladder.selected-product/{selected_method}",
            oracle=_PAYLOAD_ORACLE,
            actual=selected_value,
            oracle_value=oracle_value,
        )
    )


def _premise_truth(raw: Any, reducer: AtomReducer) -> bool | None:
    if reducer is AtomReducer.NOT_EVALUATED:
        if raw is not None:
            raise AssertionError("not-evaluated premise must retain raw None")
        return None
    value = np.asarray(raw)
    if reducer is AtomReducer.SCALAR:
        if value.shape != ():
            raise AssertionError("array-valued source atom needs a reducer")
        return bool(value.item())
    if reducer is AtomReducer.ALL_ELEMENTS:
        return bool(np.all(value))
    if reducer is AtomReducer.ANY_ELEMENT:
        return bool(np.any(value))
    raise AssertionError(f"unknown atom reducer {reducer!r}")


def _captured_atom_lineage(
    record: _FixtureRecord,
    atom_id: str,
    captured_locals: Mapping[str, Any],
) -> tuple[str, ...]:
    """Derive an atom's input keys from its source operands and live call map."""

    source_node = _SOURCE_INDEX[atom_id][1]
    operand_values = _ast_operand_values(source_node, captured_locals)
    keys = [
        key
        for key in record.direct_input_keys
        if any(
            _lineage_contains(record.audit[key], operand)
            or _lineage_contains(operand, record.audit[key])
            for operand in operand_values
        )
    ]
    if keys:
        return tuple(keys)

    # Some registered atoms are booleans derived earlier in the production
    # function (for example ``condition_resolved``).  In that case, use only
    # arguments of the source call that are present in the captured frame;
    # this remains call-derived instead of claiming every fixture field.
    captured_values = tuple(captured_locals.values())
    roots = [
        argument
        for _method, arguments in record.call_arguments
        for argument in arguments.values()
        if any(argument is value for value in captured_values)
    ]
    fallback = [
        key
        for key in record.direct_input_keys
        if any(_lineage_contains(root, record.audit[key]) for root in roots)
    ]
    if not fallback:
        raise AssertionError(
            f"{atom_id} source operands have no lineage to the live call arguments"
        )
    return tuple(fallback)


def _atom_premises(
    record: _FixtureRecord,
    declared_actuals: Mapping[str, Any],
    oracles_by_syntax: Mapping[str, Any],
    *,
    reducers: Mapping[str, AtomReducer] | None = None,
) -> None:
    if record.atom_id is None:
        return
    required = {
        _ATOM_SYNTAX[atom_id]
        for atom_id in _ENTRIES[record.gate_id].conjunction_atom_ids
    }
    if set(declared_actuals) != required or set(oracles_by_syntax) != required:
        raise AssertionError(
            f"{record.gate_id} atom fixture must cover exact source syntax"
        )
    active = _ACTIVE_SOURCE_RECORDS.get()
    if active is None:
        raise AssertionError("source-premise recording was not active")
    reducer_map = {} if reducers is None else dict(reducers)
    premises: dict[str, _PremiseValue] = {}
    for atom_id in _ENTRIES[record.gate_id].conjunction_atom_ids:
        syntax = _ATOM_SYNTAX[atom_id]
        captured = active.get(atom_id)
        raw_actual = None if captured is None else captured[0]
        declared_reducer = reducer_map.get(syntax, AtomReducer.SCALAR)
        reducer = (
            AtomReducer.NOT_EVALUATED
            if captured is None
            else (
                AtomReducer.SCALAR
                if declared_reducer is AtomReducer.NOT_EVALUATED
                else declared_reducer
            )
        )
        raw_oracle = None if captured is None else oracles_by_syntax[syntax]
        if (
            captured is not None
            and np.asarray(raw_actual).shape == ()
            and np.asarray(raw_oracle).size == 1
        ):
            raw_oracle = np.asarray(raw_oracle).reshape(-1)[0].item()
        premises[syntax] = _PremiseValue(
            raw_actual=raw_actual,
            raw_oracle=raw_oracle,
            reducer=reducer,
            realized_keys=_captured_atom_lineage(
                record,
                atom_id,
                (
                    captured[1]
                    if captured is not None
                    else next(iter(active.values()))[1]
                ),
            ),
        )
    record.premises = premises


def _sigma_payload_action(record: _FixtureRecord) -> RawObservation:
    tolerance = 2.0
    config = _config(record, structure_rtol=0.0, structure_atol=tolerance)
    if record.atom_id is not None:
        syntax = record.atom_syntax
        if syntax == "sigma.ndim == 1":
            sigma = np.array([1.3, 2.7])
        elif syntax == "np.array_equal(sigma, sigma.T)":
            sigma = np.diag([1.3, 2.7])
        else:
            sigma = np.array([[1.3, math.nextafter(tolerance, math.inf)], [0.0, 2.7]])
        boundary_value = (
            0.0 if sigma.ndim == 1 else float(abs(sigma[0, 1] - sigma[1, 0]))
        )
    else:
        boundary_value = _float_value(
            record.point,
            tolerance,
            very_low=0.0,
            very_high=4.0,
            extreme=1e300,
        )
        if record.point.role is PointRole.VERY_LOW:
            sigma = np.array([1.3, 2.7])
        else:
            sigma = np.array([[1.3, boundary_value], [0.0, 2.7]])
    record.input("sigma", sigma)
    record.input("asymmetry", boundary_value)
    record.input("computation_dtype", sigma.dtype.str)
    payload = record.invoke(
        "ladder._sigma_payload",
        ladder._sigma_payload,
        result_key="sigma_payload",
        args=("sigma", "config"),
    )
    record.returned("payload_changed", not np.array_equal(payload, sigma))
    exact = bool(np.array_equal(sigma, sigma.T))
    oracle_tolerant = bool(
        sigma.ndim == 1
        or oracles.tolerant_symmetry(
            sigma, relative_tolerance=0.0, absolute_tolerance=tolerance
        )
    )
    original = bool(sigma.ndim == 1 or exact or not oracle_tolerant)
    expected_payload = (
        np.array(sigma, copy=True)
        if original
        else _independent_symmetric_representative(sigma)
    )
    _problem(
        record,
        np.eye(sigma.shape[-1]) * 1.3,
        np.asarray(expected_payload) - np.eye(sigma.shape[-1]) * 1.3,
    )
    record.raw("config", config)
    with _suspend_source_recording():
        verdicts = _check_premises(record)
    record.returned("level_three_satisfied", verdicts[3].satisfied)
    production_tolerant = bool(
        eager._is_symmetric(
            sigma,
            rtol=config.structure_rtol,
            atol=config.structure_atol,
        )
    )
    selected = np.asarray(payload)
    record.returned("sigma_payload_selected", selected)
    record.returned("sigma_payload_expected", expected_payload)
    original_truth = _independent_matrix_logdet(np.asarray(sigma))
    selected_truth = _independent_matrix_logdet(expected_payload)
    record.evaluations.extend(
        (
            numerical_evaluation(
                method="ladder._sigma_payload/original-logdet",
                oracle=_PAYLOAD_ORACLE,
                actual=_dense_logdet_value(np.asarray(sigma)),
                oracle_value=original_truth,
            ),
            numerical_evaluation(
                method="ladder._sigma_payload/selected-logdet",
                oracle=_PAYLOAD_ORACLE,
                actual=_dense_logdet_value(selected),
                oracle_value=selected_truth,
            ),
        )
    )
    record.input("selected_sigma", expected_payload)
    if expected_payload.ndim == 1:
        record.input("selected_structure_kind", "diagonal")
        _record_payload(
            record,
            method="eager.structured_logdet",
            function=eager.structured_logdet,
            args=("selected_sigma",),
            kwargs={"kind": "selected_structure_kind"},
            oracle_value=selected_truth,
        )
    elif np.array_equal(expected_payload, expected_payload.T):
        _record_payload(
            record,
            method="eager.dense_cholesky_logdet",
            function=eager.dense_cholesky_logdet,
            args=("selected_sigma",),
            oracle_value=selected_truth,
        )
    production_original = bool(
        np.shares_memory(payload, sigma) or np.array_equal(payload, sigma)
    )
    record.input("original_sigma", np.asarray(sigma))
    record.input("representative_sigma", np.asarray(expected_payload))
    admitted_value = _capture_direct_product(
        record,
        side="admitted",
        method="eager.dense_cholesky_logdet",
        function=eager.dense_cholesky_logdet,
        args=("original_sigma",),
    )
    refused_value = _capture_direct_product(
        record,
        side="refused",
        method="eager.dense_cholesky_logdet",
        function=eager.dense_cholesky_logdet,
        args=("representative_sigma",),
    )
    _record_selected_product(
        record,
        production_admitted=production_original,
        oracle_admitted=original,
        admitted_method="eager.dense_cholesky_logdet",
        admitted_value=admitted_value,
        refused_method="eager.dense_cholesky_logdet",
        refused_value=refused_value,
        oracle_value=original_truth if original else selected_truth,
    )
    record.observed_side = _side(production_original)
    record.oracle_side = _side(original)
    source_exact: bool | None = exact
    source_tolerant: bool | None = production_tolerant
    reducers: dict[str, AtomReducer] = {}
    if record.atom_syntax == "sigma.ndim == 1" and sigma.ndim == 1:
        source_exact = None
        source_tolerant = None
        reducers["np.array_equal(sigma, sigma.T)"] = AtomReducer.NOT_EVALUATED
        reducers[
            "_is_symmetric(sigma, rtol=config.structure_rtol, atol=config.structure_atol)"
        ] = AtomReducer.NOT_EVALUATED
    elif record.atom_syntax == "np.array_equal(sigma, sigma.T)" and exact:
        source_tolerant = None
        reducers[
            "_is_symmetric(sigma, rtol=config.structure_rtol, atol=config.structure_atol)"
        ] = AtomReducer.NOT_EVALUATED
    oracle_source_exact = (
        None
        if source_exact is None
        else bool(
            sigma.shape == sigma.T.shape
            and all(sigma[index] == sigma.T[index] for index in np.ndindex(sigma.shape))
        )
    )
    oracle_source_tolerant = None if source_tolerant is None else oracle_tolerant
    actuals = {
        "sigma.ndim == 1": sigma.ndim == 1,
        "np.array_equal(sigma, sigma.T)": source_exact,
        "_is_symmetric(sigma, rtol=config.structure_rtol, atol=config.structure_atol)": source_tolerant,
    }
    _atom_premises(
        record,
        actuals,
        {
            "sigma.ndim == 1": len(sigma.shape) == 1,
            "np.array_equal(sigma, sigma.T)": oracle_source_exact,
            "_is_symmetric(sigma, rtol=config.structure_rtol, atol=config.structure_atol)": oracle_source_tolerant,
        },
        reducers=reducers,
    )
    record.realization(
        point_key="asymmetry",
        value=boundary_value,
        threshold=tolerance,
        axis_key="sigma",
        dtype=np.dtype(float).str,
    )
    record.bind_axes(
        sigma_layout="sigma",
        sigma_asymmetry="asymmetry",
        structure_atol="config_structure_atol",
        structure_rtol="config_structure_rtol",
        computation_dtype="computation_dtype",
    )
    return record.finish()


def _sigma_finite_action(record: _FixtureRecord) -> RawObservation:
    maximum = float(np.finfo(np.float64).max)
    threshold = maximum / 2.0
    value = _float_value(
        record.point,
        threshold,
        very_low=0.0,
        very_high=0.75 * maximum,
        extreme=maximum,
    )
    lam_value = threshold
    perturb_value = value
    lam = np.array([lam_value, 2.7])
    perturb = np.array([perturb_value, -0.2])
    _problem(record, lam, perturb)
    _config(record)
    record.input("lambda_entry", float(lam[0]))
    record.input("perturbation_entry", float(perturb[0]))
    record.input("computation_dtype", record.values["lambda_matrix"].dtype.str)
    try:
        sigma, remainder = record.invoke(
            "eager._two_sum_error",
            eager._two_sum_error,
            result_key="two_sum",
            args=("lambda_matrix", "perturbation"),
        )
        record.returned(
            "sigma_finite",
            bool(np.all(np.isfinite(sigma)) and np.all(np.isfinite(remainder))),
        )
        direct_valid = True
    except ValueError as error:
        record.returned("sigma_finite", False)
        record.notes.append(type(error).__name__)
        direct_valid = False
    with np.errstate(over="ignore"):
        exact_sum = np.longdouble(lam_value) + np.longdouble(perturb_value)
    oracle_valid = bool(
        np.isfinite(exact_sum) and abs(exact_sum) <= np.finfo(float).max
    )
    with np.errstate(over="ignore", invalid="ignore"):
        record.input("sigma_input", np.asarray(lam) + np.asarray(perturb))
    admitted_value = _capture_direct_product(
        record,
        side="admitted",
        method="eager.finite_perturbation_logdet",
        function=eager.finite_perturbation_logdet,
        args=("lambda_matrix", "perturbation"),
    )
    refused_value = _capture_direct_product(
        record,
        side="refused",
        method="eager.dense_cholesky_logdet",
        function=eager.dense_cholesky_logdet,
        args=("sigma_input",),
    )
    record.input("perturbation_scalar", value)
    record.realization(
        point_key="perturbation_scalar",
        value=value,
        threshold=threshold,
        axis_key="perturbation",
        dtype=np.dtype(float).str,
    )
    try:
        verdicts = _check_premises(record)
    except UnboundLocalError as error:
        record.returned(
            "premise-direct-exception",
            {
                "status": "exception",
                "exception_type": type(error).__name__,
                "message": str(error),
            },
        )
        record.checks.append(
            oracle_check(
                oracle="independent completed premise evaluation",
                actual=type(error).__name__,
                expected="PremiseVerdict tuple",
            )
        )
        _record_selected_product(
            record,
            production_admitted=True,
            oracle_admitted=oracle_valid,
            admitted_method="eager.finite_perturbation_logdet",
            admitted_value=admitted_value,
            refused_method="eager.dense_cholesky_logdet",
            refused_value=refused_value,
            oracle_value=(
                _payload_oracle(record.values["problem"]) if oracle_valid else 0.0
            ),
        )
        record.observed_side = GateSide.ADMITTED
        record.oracle_side = _side(oracle_valid)
        record.bind_axes(
            lambda_entry="lambda_entry",
            perturbation_entry="perturbation_entry",
            computation_dtype="computation_dtype",
        )
        return record.finish()
    details = verdicts[1].details
    production_formation = bool(details["sigma_formation_valid"])
    finite_payload_measurement = bool(details["finite_payload_rho_measurement_valid"])
    record.returned("premise_sigma_formation_valid", production_formation)
    record.returned(
        "premise_finite_payload_measurement_valid", finite_payload_measurement
    )
    record.checks.append(
        oracle_check(
            oracle="independent finite-payload measurement reachability",
            actual=finite_payload_measurement,
            expected=oracle_valid,
        )
    )
    production_valid = bool(
        production_formation and finite_payload_measurement and direct_valid
    )
    record.observed_side = _side(production_valid)
    record.oracle_side = _side(oracle_valid)
    _record_selected_product(
        record,
        production_admitted=production_valid,
        oracle_admitted=oracle_valid,
        admitted_method="eager.finite_perturbation_logdet",
        admitted_value=admitted_value,
        refused_method="eager.dense_cholesky_logdet",
        refused_value=refused_value,
        oracle_value=(
            _payload_oracle(record.values["problem"]) if oracle_valid else 0.0
        ),
    )
    record.bind_axes(
        lambda_entry="lambda_entry",
        perturbation_entry="perturbation_entry",
        computation_dtype="computation_dtype",
    )
    return record.finish()


def _compact_action(record: _FixtureRecord) -> RawObservation:
    threshold = 0.0
    value = _float_value(
        record.point,
        threshold,
        very_low=-2.7,
        very_high=2.7,
        extreme=-1e100,
    )
    if record.atom_id is not None:
        atom_index = list(_ENTRIES[record.gate_id].conjunction_atom_ids).index(
            record.atom_id
        )
        value = 0.0 if atom_index == 0 else -float(np.nextafter(0.0, 1.0))
        sigma = np.array([1.3 + atom_index, value, 2.7])
    else:
        sigma = np.array([1.3, value])
    record.input("compact_entry", value)
    compact_lambda = np.array([1.3, 2.7, 3.1])[: sigma.size]
    _problem(record, compact_lambda, sigma - compact_lambda)
    _config(record)
    record.input("sigma_input", sigma)
    result = record.invoke(
        "ladder._structure_request",
        ladder._structure_request,
        result_key="structure_request",
        args=("problem", "config", "sigma_input"),
    )
    valid = bool(result[1])
    record.returned("structure_valid", valid)
    oracle_valid = bool(all(float(item) > 0.0 for item in sigma))
    record.input("structured_kind", "diagonal")
    record.input("dense_sigma", np.diag(sigma))
    admitted_value = _capture_direct_product(
        record,
        side="admitted",
        method="eager.structured_logdet",
        function=eager.structured_logdet,
        args=("sigma_input",),
        kwargs={"kind": "structured_kind"},
    )
    refused_value = _capture_direct_product(
        record,
        side="refused",
        method="eager.dense_cholesky_logdet",
        function=eager.dense_cholesky_logdet,
        args=("dense_sigma",),
    )
    _record_selected_product(
        record,
        production_admitted=valid,
        oracle_admitted=oracle_valid,
        admitted_method="eager.structured_logdet",
        admitted_value=admitted_value,
        refused_method="eager.dense_cholesky_logdet",
        refused_value=refused_value,
        oracle_value=(
            oracles.diagonal_logdet(sigma) if oracle_valid else 0.0
        ),
    )
    record.observed_side = _side(valid)
    record.oracle_side = _side(oracle_valid)
    raw_positive = sigma > 0.0
    _atom_premises(
        record,
        {
            "np.all(sigma > 0.0)": valid,
            "sigma > 0.0": raw_positive,
        },
        {
            "np.all(sigma > 0.0)": bool(np.all(raw_positive)),
            "sigma > 0.0": np.greater(sigma, 0.0),
        },
        reducers={"sigma > 0.0": AtomReducer.ALL_ELEMENTS},
    )
    record.realization(
        point_key="compact_entry",
        value=value,
        threshold=threshold,
        axis_key="sigma_input",
        dtype=np.dtype(float).str,
    )
    return record.finish()


def _diagonal_action(record: _FixtureRecord) -> RawObservation:
    delta, threshold = _exact_mismatch(record.point)
    scale = 1.3 if record.point.role is not PointRole.VERY_HIGH else 2.7
    kind: str | None = "diagonal"
    if record.atom_syntax == "kind is None":
        kind = None
        delta = 0.0
    elif record.atom_syntax is not None:
        kind = None
        delta = float(np.nextafter(0.0, 1.0))
    if record.point.role is PointRole.EXTREME and record.atom_id is None:
        delta = 1e100
    sigma = np.array([[scale, delta], [0.0, 2.7 * scale]])
    record.input("off_diagonal", delta)
    problem = _problem(
        record,
        np.eye(2) * scale,
        sigma - np.eye(2) * scale,
        structure_kind=kind,
    )
    _config(record, structure_rtol=1e6, structure_atol=1e6)
    record.input("sigma_input", sigma)
    helper = bool(
        record.invoke(
            "eager._is_diagonal",
            eager._is_diagonal,
            result_key="diagonal_helper",
            args=("sigma_input",),
            kwargs={
                "rtol": "config_structure_rtol",
                "atol": "config_structure_atol",
            },
        )
    )
    result = record.invoke(
        "ladder._structure_request",
        ladder._structure_request,
        result_key="structure_request",
        args=("problem", "config", "sigma_input"),
    )
    valid = bool(result[1])
    record.returned("diagonal_helper_result", helper)
    record.returned("structure_valid", valid)
    oracle_valid = oracles.is_diagonal(sigma)
    record.input("selected_structure_kind", result[0] or "diagonal")
    admitted_value = _capture_direct_product(
        record,
        side="admitted",
        method="eager.structured_logdet",
        function=eager.structured_logdet,
        args=("sigma_input",),
        kwargs={"kind": "selected_structure_kind"},
    )
    refused_value = _capture_direct_product(
        record,
        side="refused",
        method="eager.dense_cholesky_logdet",
        function=eager.dense_cholesky_logdet,
        args=("sigma_input",),
    )
    _record_selected_product(
        record,
        production_admitted=valid,
        oracle_admitted=oracle_valid,
        admitted_method="eager.structured_logdet",
        admitted_value=admitted_value,
        refused_method="eager.dense_cholesky_logdet",
        refused_value=refused_value,
        oracle_value=_independent_matrix_logdet(sigma),
    )
    record.observed_side = _side(valid)
    record.oracle_side = _side(oracle_valid)
    _atom_premises(
        record,
        {
            "kind is None": kind is None,
            "_is_diagonal(sigma, rtol=config.structure_rtol, atol=config.structure_atol)": helper,
        },
        {
            "kind is None": problem.structure_kind is None,
            "_is_diagonal(sigma, rtol=config.structure_rtol, atol=config.structure_atol)": oracle_valid,
        },
    )
    record.realization(
        point_key="off_diagonal",
        value=delta,
        threshold=threshold,
        axis_key="sigma_input",
        dtype=np.dtype(float).str,
    )
    if oracle_valid and problem.structure_kind is not None:
        _record_payload(
            record,
            method="eager.structured_logdet",
            function=eager.structured_logdet,
            args=("sigma_input",),
            kwargs={"kind": "structure_kind"},
        )
    record.bind_axes(
        structure_request="structure_kind",
        off_diagonal="off_diagonal",
    )
    return record.finish()


def _circulant_action(record: _FixtureRecord) -> RawObservation:
    delta, threshold = _exact_mismatch(record.point)
    if record.atom_id is not None:
        delta = float(np.nextafter(0.0, 1.0))
    if record.point.role is PointRole.EXTREME and record.atom_id is None:
        delta = 4.0
    first_row = np.array([2.7, 0.2, 0.0, 0.2])
    sigma = np.vstack([np.roll(first_row, index) for index in range(4)])
    sigma[1, 3] = delta
    record.input("circulant_mismatch", delta)
    structure_kind = record.input("structure_kind", "circulant")
    _problem(
        record,
        np.eye(4) * 1.3,
        sigma - np.eye(4) * 1.3,
        structure_kind=structure_kind,
    )
    _config(record, structure_rtol=1e6, structure_atol=1e6)
    record.input("sigma_input", sigma)
    helper = bool(
        record.invoke(
            "eager._is_circulant",
            eager._is_circulant,
            result_key="circulant_helper",
            args=("sigma_input",),
            kwargs={
                "rtol": "config_structure_rtol",
                "atol": "config_structure_atol",
            },
        )
    )
    try:
        spectrum = record.invoke(
            "eager._circulant_eigenvalues",
            eager._circulant_eigenvalues,
            result_key="circulant_spectrum",
            args=("sigma_input",),
        )
        spectrum_valid = bool(np.all(np.asarray(spectrum) > 0.0))
        record.returned("spectrum_minimum", float(np.min(spectrum)))
    except ValueError as error:
        spectrum_valid = False
        record.notes.append(type(error).__name__)
    result = record.invoke(
        "ladder._structure_request",
        ladder._structure_request,
        result_key="structure_request",
        args=("problem", "config", "sigma_input"),
    )
    valid = bool(result[1])
    oracle_structure = oracles.is_circulant(sigma)
    try:
        oracle_spectrum = oracles.circulant_eigenvalues(sigma[0])
        spectrum_scale = max(1.0, float(np.sum(np.abs(sigma[0]))))
        oracle_valid = oracle_structure and bool(
            np.all(np.real(oracle_spectrum) > 0.0)
            and np.all(
                np.abs(np.imag(oracle_spectrum))
                <= 64.0 * np.finfo(float).eps * spectrum_scale
            )
        )
    except ValueError:
        oracle_valid = False
    record.returned("circulant_helper_result", helper)
    record.returned("structure_valid", valid)
    record.input("spectrum_scale", float(first_row[0]))
    admitted_value = _capture_direct_product(
        record,
        side="admitted",
        method="eager.structured_logdet",
        function=eager.structured_logdet,
        args=("sigma_input",),
        kwargs={"kind": "structure_kind"},
    )
    refused_value = _capture_direct_product(
        record,
        side="refused",
        method="eager.dense_cholesky_logdet",
        function=eager.dense_cholesky_logdet,
        args=("sigma_input",),
    )
    _record_selected_product(
        record,
        production_admitted=valid and spectrum_valid,
        oracle_admitted=oracle_valid,
        admitted_method="eager.structured_logdet",
        admitted_value=admitted_value,
        refused_method="eager.dense_cholesky_logdet",
        refused_value=refused_value,
        oracle_value=(
            _independent_matrix_logdet(sigma) if oracle_valid else 0.0
        ),
    )
    record.observed_side = _side(valid and spectrum_valid)
    record.oracle_side = _side(oracle_valid)
    _atom_premises(
        record,
        {
            "_is_circulant(sigma, rtol=config.structure_rtol, atol=config.structure_atol)": helper
        },
        {
            "_is_circulant(sigma, rtol=config.structure_rtol, atol=config.structure_atol)": oracle_structure
        },
    )
    record.realization(
        point_key="circulant_mismatch",
        value=delta,
        threshold=threshold,
        axis_key="sigma_input",
        dtype=np.dtype(float).str,
    )
    if valid and oracle_valid:
        _record_payload(
            record,
            method="eager.structured_logdet",
            function=eager.structured_logdet,
            args=("sigma_input",),
            kwargs={"kind": "structure_kind"},
        )
    record.bind_axes(
        circulant_layout="sigma_input",
        spectrum_scale="spectrum_scale",
    )
    return record.finish()


def _toeplitz_action(record: _FixtureRecord) -> RawObservation:
    delta, threshold = _exact_mismatch(record.point)
    if record.atom_id is not None:
        delta = float(np.nextafter(0.0, 1.0))
    if record.point.role is PointRole.EXTREME and record.atom_id is None:
        delta = 1e100
    sigma = np.diag([2.7, 2.7, 2.7])
    sigma[0, 1] = delta
    sigma[1, 0] = delta
    record.input("toeplitz_mismatch", delta)
    structure_kind = record.input("structure_kind", "toeplitz")
    _problem(
        record,
        np.eye(3) * 1.3,
        sigma - np.eye(3) * 1.3,
        structure_kind=structure_kind,
    )
    _config(record, structure_rtol=1e6, structure_atol=1e6)
    record.input("sigma_input", sigma)
    helper = bool(
        record.invoke(
            "eager._is_toeplitz",
            eager._is_toeplitz,
            result_key="toeplitz_helper",
            args=("sigma_input",),
            kwargs={
                "rtol": "config_structure_rtol",
                "atol": "config_structure_atol",
            },
        )
    )
    result = record.invoke(
        "ladder._structure_request",
        ladder._structure_request,
        result_key="structure_request",
        args=("problem", "config", "sigma_input"),
    )
    valid = bool(result[1])
    oracle_valid = oracles.is_toeplitz(sigma)
    record.returned("toeplitz_helper_result", helper)
    record.returned("structure_valid", valid)
    admitted_value = _capture_direct_product(
        record,
        side="admitted",
        method="eager.structured_logdet",
        function=eager.structured_logdet,
        args=("sigma_input",),
        kwargs={"kind": "structure_kind"},
    )
    refused_value = _capture_direct_product(
        record,
        side="refused",
        method="eager.dense_cholesky_logdet",
        function=eager.dense_cholesky_logdet,
        args=("sigma_input",),
    )
    _record_selected_product(
        record,
        production_admitted=valid,
        oracle_admitted=oracle_valid,
        admitted_method="eager.structured_logdet",
        admitted_value=admitted_value,
        refused_method="eager.dense_cholesky_logdet",
        refused_value=refused_value,
        oracle_value=(
            _independent_matrix_logdet(sigma)
            if oracles.symmetric_is_positive_definite(sigma)
            else 0.0
        ),
    )
    record.observed_side = _side(valid)
    record.oracle_side = _side(oracle_valid)
    _atom_premises(
        record,
        {
            "_is_toeplitz(sigma, rtol=config.structure_rtol, atol=config.structure_atol)": helper
        },
        {
            "_is_toeplitz(sigma, rtol=config.structure_rtol, atol=config.structure_atol)": oracle_valid
        },
    )
    record.realization(
        point_key="toeplitz_mismatch",
        value=delta,
        threshold=threshold,
        axis_key="sigma_input",
        dtype=np.dtype(float).str,
    )
    if valid and oracle_valid:
        _record_payload(
            record,
            method="eager.structured_logdet",
            function=eager.structured_logdet,
            args=("sigma_input",),
            kwargs={"kind": "structure_kind"},
        )
    return record.finish()


def _kronecker_action(record: _FixtureRecord) -> RawObservation:
    delta, threshold = _exact_mismatch(record.point)
    left = np.diag([1.3, 2.7])
    right = np.diag([2.0, 3.0])
    factors = eager.KroneckerStructure((left, right))
    sigma = oracles.explicit_kronecker((left, right))
    structure = factors
    if record.atom_syntax == "_is_positive_definite(factor)":
        right = np.diag([2.0, -float(np.nextafter(0.0, 1.0))])
        factors = eager.KroneckerStructure((left, right))
        sigma = oracles.explicit_kronecker((left, np.diag([2.0, 3.0])))
        structure = factors
        delta = float(np.nextafter(0.0, 1.0))
    elif record.atom_syntax == "reconstructed.shape == sigma.shape":
        factors = eager.KroneckerStructure((left,))
        structure = factors
        delta = 1.0
    elif record.atom_syntax == "np.array_equal(reconstructed, sigma)":
        sigma = sigma.copy()
        sigma[0, 0] = np.nextafter(sigma[0, 0], math.inf)
        delta = float(sigma[0, 0] - 2.6)
    elif record.point.role not in {
        PointRole.EXACT,
        PointRole.VERY_LOW,
        PointRole.VERY_HIGH,
        PointRole.EXTREME,
    }:
        sigma = sigma.copy()
        if record.point.role is PointRole.ULP_MISMATCH:
            threshold = float(sigma[0, 0])
            sigma[0, 0] = np.nextafter(sigma[0, 0], math.inf)
            delta = float(sigma[0, 0])
        else:
            sigma[0, 1] = delta
    if record.point.role is PointRole.VERY_LOW:
        left = 0.5 * left
        factors = eager.KroneckerStructure((left, right))
        sigma = oracles.explicit_kronecker((left, right))
        structure = factors
    if record.point.role is PointRole.VERY_HIGH:
        sigma = 4.0 * sigma
        factors = eager.KroneckerStructure((2.0 * left, 2.0 * right))
        structure = factors
        delta = float(np.asarray(factors.factors[0])[0, 0])
    if record.point.role is PointRole.EXTREME and record.atom_id is None:
        structure = None
        sigma = sigma.copy()
        sigma[0, 1] = 1e100
        delta = 1e100
    if record.atom_id is not None:
        delta = float(np.asarray(sigma).reshape(-1)[0])
        threshold = 0.0
    record.input("kronecker_mismatch", delta)
    structure_kind = record.input("structure_kind", "kronecker")
    _problem(
        record,
        np.eye(sigma.shape[0]) * 1.3,
        sigma - np.eye(sigma.shape[0]) * 1.3,
        structure_kind=structure_kind,
        structure=structure,
    )
    _config(record)
    record.input("sigma_input", sigma)
    presence_descriptor = (
        None
        if structure is None
        else {
            "present": True,
            "factor_count": len(structure.factors),
            "factor_shapes": tuple(
                tuple(int(item) for item in factor.shape)
                for factor in structure.factors
            ),
        }
    )
    record.input(
        "structure_presence",
        structure,
        audit_value=presence_descriptor,
    )
    first_factor = (
        np.asarray(structure.factors[0])
        if structure is not None
        else np.empty((0, 0))
    )
    record.input(
        "factor_shape",
        first_factor,
        audit_value=(
            _array_descriptor(first_factor) if structure is not None else None
        ),
    )
    record.input(
        "factor_spectrum",
        float(first_factor.reshape(-1)[0]) if first_factor.size else 0.0,
    )
    record.input("reconstruction_value", float(np.asarray(sigma).reshape(-1)[0]))
    factor_results = [
        bool(
            record.invoke(
                "eager._is_positive_definite",
                eager._is_positive_definite,
                result_key=f"factor_spd_{index}",
                args=(f"structure_factor_{index}",),
            )
        )
        for index in range(len(structure.factors) if structure is not None else 0)
    ]
    factor_spd = all(factor_results)
    result = record.invoke(
        "ladder._structure_request",
        ladder._structure_request,
        result_key="structure_request",
        args=("problem", "config", "sigma_input"),
    )
    valid = bool(result[1])
    if structure is None:
        reconstructed = np.empty((0, 0))
        oracle_reconstructed = np.empty((0, 0))
        oracle_factor_results = np.array([], dtype=bool)
    else:
        reconstructed = np.asarray(structure.factors[0])
        for factor in structure.factors[1:]:
            reconstructed = np.kron(reconstructed, np.asarray(factor))
        factor_values = tuple(np.asarray(item) for item in structure.factors)
        oracle_reconstructed = oracles.explicit_kronecker(factor_values)
        oracle_factor_results = np.array(
            [
                oracles.symmetric_is_positive_definite(item)
                for item in factor_values
            ],
            dtype=bool,
        )
    shape_matches = reconstructed.shape == sigma.shape
    exact_matches = shape_matches and np.array_equal(reconstructed, sigma)
    oracle_shape_matches = oracle_reconstructed.shape == sigma.shape
    oracle_exact_matches = oracle_shape_matches and np.array_equal(
        oracle_reconstructed, sigma
    )
    oracle_valid = (
        bool(np.all(oracle_factor_results))
        and oracle_shape_matches
        and oracle_exact_matches
    )
    record.returned("structure_valid", valid)
    record.returned("factor_spd", factor_spd)
    admitted_value = _capture_direct_product(
        record,
        side="admitted",
        method="eager.structured_logdet",
        function=eager.structured_logdet,
        args=("sigma_input",),
        kwargs={"kind": "structure_kind", "structure": "structure"},
    )
    refused_value = _capture_direct_product(
        record,
        side="refused",
        method="eager.dense_cholesky_logdet",
        function=eager.dense_cholesky_logdet,
        args=("sigma_input",),
    )
    sigma_spd = bool(
        np.array_equal(sigma, sigma.T)
        and oracles.symmetric_is_positive_definite(sigma)
    )
    _record_selected_product(
        record,
        production_admitted=valid,
        oracle_admitted=oracle_valid,
        admitted_method="eager.structured_logdet",
        admitted_value=admitted_value,
        refused_method="eager.dense_cholesky_logdet",
        refused_value=refused_value,
        oracle_value=_independent_matrix_logdet(sigma) if sigma_spd else 0.0,
    )
    record.observed_side = _side(valid)
    record.oracle_side = _side(oracle_valid)
    _atom_premises(
        record,
        {
            "_is_positive_definite(factor)": np.asarray(factor_results, dtype=bool),
            "reconstructed.shape == sigma.shape": shape_matches,
            "np.array_equal(reconstructed, sigma)": exact_matches,
        },
        {
            "_is_positive_definite(factor)": oracle_factor_results,
            "reconstructed.shape == sigma.shape": oracle_shape_matches,
            "np.array_equal(reconstructed, sigma)": oracle_exact_matches,
        },
        reducers={"_is_positive_definite(factor)": AtomReducer.ALL_ELEMENTS},
    )
    record.realization(
        point_key="kronecker_mismatch",
        value=delta,
        threshold=threshold,
        axis_key="sigma_input",
        dtype=np.dtype(float).str,
    )
    if valid and oracle_valid:
        _record_payload(
            record,
            method="eager.structured_logdet",
            function=eager.structured_logdet,
            args=("sigma_input",),
            kwargs={"kind": "structure_kind", "structure": "structure"},
        )
    record.bind_axes(
        structure_request="structure_kind",
        structure_presence="structure_presence",
        factor_spectrum="factor_spectrum",
        factor_shape="factor_shape",
        reconstruction_value="reconstruction_value",
    )
    return record.finish()


def _shared_sigma_action(record: _FixtureRecord) -> RawObservation:
    ceiling_threshold = 1.3 * np.finfo(np.float64).eps
    value = _float_value(
        record.point,
        ceiling_threshold,
        very_low=-1.0,
        very_high=2.7,
        extreme=math.nan,
    )
    if record.atom_syntax == "sigma.ndim == 1":
        value = 2.7
        sigma = np.array([1.3, value])
    elif (
        record.atom_syntax
        == "_is_symmetric(sigma, rtol=config.structure_rtol, atol=config.structure_atol)"
    ):
        sigma = np.array([[1.3, 0.1], [0.0, 2.7]])
        value = 0.1
    elif (
        record.atom_syntax
        == "_is_positive_definite(sigma, rtol=config.structure_rtol, atol=config.structure_atol)"
    ):
        sigma = np.diag([1.3, -float(np.nextafter(0.0, 1.0))])
        value = float(sigma[1, 1])
    else:
        sigma = np.diag([1.3, value])
    if record.point.role is PointRole.EXTREME and record.atom_id is None:
        sigma = np.diag([1.3, np.finfo(float).max])
        value = float(np.finfo(float).max)
    positive_diagonal = bool(
        np.all(sigma > 0.0)
        if sigma.ndim == 1
        else (
            np.array_equal(sigma, np.diag(np.diag(sigma)))
            and np.all(np.diag(sigma) > 0.0)
        )
    )
    if positive_diagonal:
        lambda_matrix = 0.5 * sigma
    elif sigma.ndim == 1:
        lambda_matrix = np.array([0.65, 1.35])
    else:
        lambda_matrix = np.eye(2) * 0.65
    record.input("smallest_eigenvalue", value)
    _problem(record, lambda_matrix, sigma - lambda_matrix)
    _config(record)
    record.input("sigma_input", sigma)
    record.input(
        "fact_sigma_layout",
        sigma,
        audit_value=_array_descriptor(sigma),
    )
    sigma_symmetry = (
        0.0
        if sigma.ndim == 1
        else float(sigma[0, 1] - sigma[1, 0])
    )
    record.input("sigma_symmetry", sigma_symmetry)
    condition_leaf = float(np.asarray(sigma).reshape(-1)[-1])
    record.input("condition_scale", condition_leaf)
    record.input("computation_dtype", sigma.dtype.str)
    symmetric = bool(
        record.invoke(
            "eager._is_symmetric",
            eager._is_symmetric,
            result_key="symmetric_result",
            args=("sigma_input",),
            kwargs={
                "rtol": "config_structure_rtol",
                "atol": "config_structure_atol",
            },
        )
    )
    spd = bool(
        record.invoke(
            "eager._is_positive_definite",
            eager._is_positive_definite,
            result_key="spd_result",
            args=("sigma_input",),
            kwargs={
                "rtol": "config_structure_rtol",
                "atol": "config_structure_atol",
            },
        )
    )
    condition, ceiling, resolved = record.invoke(
        "eager._condition_certificate",
        eager._condition_certificate,
        result_key="condition_certificate",
        args=("sigma_input",),
    )
    record.returned("condition", float(condition))
    record.returned("condition_ceiling", float(ceiling))
    record.returned("condition_resolved", bool(resolved))
    verdicts = _check_premises(record)
    shared_details = verdicts[4].details
    production_symmetric = bool(shared_details["symmetric"])
    production_spd = bool(shared_details["positive_definite"])
    production_resolved = bool(
        shared_details["condition"] < shared_details["condition_ceiling"]
    )
    production_valid = bool(
        production_symmetric and production_spd and production_resolved
    )
    oracle_symmetric = bool(
        sigma.ndim == 1
        or oracles.tolerant_symmetry(
            sigma,
            relative_tolerance=record.values["config"].structure_rtol,
            absolute_tolerance=record.values["config"].structure_atol,
        )
    )
    oracle_spd = bool(
        np.all(sigma > 0.0)
        if sigma.ndim == 1
        else oracle_symmetric
        and oracles.symmetric_is_positive_definite(
            sigma,
            relative_tolerance=record.values["config"].structure_rtol,
            absolute_tolerance=record.values["config"].structure_atol,
        )
    )
    oracle_condition = (
        float(np.max(np.abs(sigma)) / np.min(np.abs(sigma)))
        if sigma.ndim == 1
        else oracles.spectral_condition(sigma)
    )
    oracle_resolved = bool(
        math.isfinite(oracle_condition)
        and oracle_condition < 1.0 / np.finfo(sigma.dtype).eps
    )
    oracle_valid = oracle_symmetric and oracle_spd and oracle_resolved
    admitted_value = _capture_direct_product(
        record,
        side="admitted",
        method="eager.dense_cholesky_logdet",
        function=eager.dense_cholesky_logdet,
        args=("sigma_input",),
    )
    refused_value = _capture_direct_product(
        record,
        side="refused",
        method="eager.finite_perturbation_logdet",
        function=eager.finite_perturbation_logdet,
        args=("lambda_matrix", "perturbation"),
    )
    _record_selected_product(
        record,
        production_admitted=production_valid,
        oracle_admitted=oracle_valid,
        admitted_method="eager.dense_cholesky_logdet",
        admitted_value=admitted_value,
        refused_method="eager.finite_perturbation_logdet",
        refused_value=refused_value,
        oracle_value=(
            _payload_oracle(record.values["problem"]) if oracle_valid else 0.0
        ),
    )
    record.observed_side = _side(production_valid)
    record.oracle_side = _side(oracle_valid)
    source_symmetric: bool | None = symmetric
    source_reducers: dict[str, AtomReducer] = {}
    if record.atom_syntax == "sigma.ndim == 1" and sigma.ndim == 1:
        source_symmetric = None
        source_reducers[
            "_is_symmetric(sigma, rtol=config.structure_rtol, atol=config.structure_atol)"
        ] = AtomReducer.NOT_EVALUATED
    _atom_premises(
        record,
        {
            "sigma.ndim == 1": sigma.ndim == 1,
            "_is_symmetric(sigma, rtol=config.structure_rtol, atol=config.structure_atol)": source_symmetric,
            "_is_positive_definite(sigma, rtol=config.structure_rtol, atol=config.structure_atol)": spd,
        },
        {
            "sigma.ndim == 1": np.asarray(sigma).ndim == 1,
            "_is_symmetric(sigma, rtol=config.structure_rtol, atol=config.structure_atol)": (
                None if source_symmetric is None else oracle_symmetric
            ),
            "_is_positive_definite(sigma, rtol=config.structure_rtol, atol=config.structure_atol)": oracle_spd,
        },
        reducers=source_reducers,
    )
    record.returned(
        "shared_fact_verdict",
        bool(
            verdicts[4].details["positive_definite"]
            and verdicts[4].details["condition"]
            < verdicts[4].details["condition_ceiling"]
        ),
    )
    record.realization(
        point_key="smallest_eigenvalue",
        value=value,
        threshold=ceiling_threshold,
        axis_key="sigma_input",
        dtype=np.dtype(float).str,
    )
    record.bind_axes(
        sigma_layout="fact_sigma_layout",
        # Atom cases carry no separate symmetry leaf consumed by a direct
        # call; bind the axis to the matrix the symmetry check itself
        # consumes.  Compound grid cases remap every binding onto their
        # carrier/companion config inputs anyway.
        sigma_symmetry="sigma_input",
        smallest_eigenvalue="smallest_eigenvalue",
        condition_scale="condition_scale",
        structure_rtol="config_structure_rtol",
        structure_atol="config_structure_atol",
        computation_dtype="computation_dtype",
    )
    return record.finish()


def _rank_action(record: _FixtureRecord) -> RawObservation:
    mismatch, threshold = _exact_mismatch(record.point)
    left = np.array([[0.2], [-0.1], [0.15], [0.0]])
    exact_perturbation = left @ left.T
    factors: eager.LowRankFactors | None = eager.LowRankFactors(left)
    perturbation = exact_perturbation.copy()
    if record.point.role not in {
        PointRole.EXACT,
        PointRole.VERY_LOW,
        PointRole.VERY_HIGH,
        PointRole.EXTREME,
    }:
        if record.point.role is PointRole.ULP_MISMATCH:
            bad = left.copy()
            threshold = float(left[0, 0])
            bad[0, 0] = np.nextafter(left[0, 0], math.inf)
            mismatch = float(bad[0, 0])
            factors = eager.LowRankFactors(bad)
            perturbation[3, 0] = np.nextafter(0.0, -math.inf)
            perturbation[0, 3] = perturbation[3, 0]
        elif record.point.role is PointRole.SUBNORMAL_MISMATCH:
            perturbation[3, 0] = mismatch
            perturbation[0, 3] = mismatch
        else:
            bad = left.copy()
            bad[-1, 0] = mismatch
            factors = eager.LowRankFactors(bad)
            perturbation[3, 0] = mismatch
            perturbation[0, 3] = mismatch
    if record.point.role is PointRole.VERY_LOW:
        perturbation = np.zeros((4, 4))
        factors = None
        mismatch = 0.0
    if record.point.role is PointRole.VERY_HIGH:
        left = np.column_stack((left, np.array([0.1, 0.2, -0.1, 0.3])))
        perturbation = left @ left.T
        factors = eager.LowRankFactors(left)
    if record.point.role is PointRole.EXTREME:
        factors = None
        perturbation[0, 1] = 1e100
        mismatch = 1e100
    if record.atom_id is not None:
        mismatch = float(np.asarray(perturbation).reshape(-1)[0])
        threshold = 0.0
    record.input("factor_mismatch", mismatch)
    _problem(
        record,
        np.eye(4) * 2.7,
        perturbation,
        low_rank_factors=factors,
    )
    _config(record)
    factor_descriptor = (
        None
        if factors is None
        else {
            "present": True,
            "rank": int(factors.rank_bound),
            "left_shape": tuple(int(item) for item in factors.left.shape),
            "right_shape": tuple(int(item) for item in factors.right.shape),
        }
    )
    record.input(
        "rank_factor_presence",
        factors,
        audit_value=factor_descriptor,
    )
    factor_array = (
        np.asarray(factors.left) if factors is not None else np.empty((0, 0))
    )
    record.input(
        "factor_layout",
        factor_array,
        audit_value=(
            _array_descriptor(factor_array) if factors is not None else None
        ),
    )
    record.input(
        "factor_gauge",
        float(factor_array.reshape(-1)[0]) if factor_array.size else 0.0,
    )
    rank_leaf = int(factors.rank_bound) if factors is not None else int(perturbation.shape[0])
    record.input("perturbation_rank", rank_leaf)
    record.input("lambda_scale", float(record.values["lambda_matrix"].reshape(-1)[0]))
    record.input("computation_dtype", record.values["lambda_matrix"].dtype.str)
    algebraic = int(
        record.invoke(
            "eager._algebraic_rank_bound",
            eager._algebraic_rank_bound,
            result_key="algebraic_rank",
            args=("perturbation",),
        )
    )
    record.returned("algebraic_rank_value", algebraic)
    certificate_valid = True
    if factors is not None:
        try:
            certificate = record.invoke(
                "eager._factor_projection_certificate",
                eager._factor_projection_certificate,
                result_key="factor_certificate",
                args=("perturbation", "low_rank_factors", "lambda_matrix"),
            )
            certificate_valid = bool(certificate.valid)
            record.returned("factor_certificate_valid", certificate_valid)
        except ValueError as error:
            certificate_valid = False
            record.returned("factor_certificate_valid", False)
            record.notes.append(type(error).__name__)
    verdicts = _check_premises(record)
    production_rank = int(verdicts[1].details["rank"])
    reconstruction_valid = bool(
        factors is None
        or np.array_equal(
            np.asarray(factors.left) @ np.asarray(factors.right).T, perturbation
        )
    )
    oracle_valid = reconstruction_valid
    expected_rank = (
        (
            int(np.count_nonzero(perturbation != 0.0))
            if perturbation.ndim == 1
            else int(perturbation.shape[0])
        )
        if factors is None
        else (
            int(factors.rank_bound)
            if reconstruction_valid
            else int(perturbation.shape[0])
        )
    )
    production_valid = bool(
        factors is None or production_rank == int(factors.rank_bound)
    )
    record.returned("selected_rank", production_rank)
    record.checks.append(
        oracle_check(
            oracle="independent exact algebraic rank selection",
            actual=production_rank,
            expected=expected_rank,
        )
    )
    record.observed_side = _side(production_valid)
    record.oracle_side = _side(oracle_valid)
    record.input(
        "sigma_input",
        np.asarray(record.values["lambda_matrix"])
        + np.asarray(record.values["perturbation"]),
    )
    kwargs = {"factors": "low_rank_factors"} if factors is not None else None
    admitted_value = _capture_direct_product(
        record,
        side="admitted",
        method="eager.low_rank_logdet",
        function=eager.low_rank_logdet,
        args=("lambda_matrix", "perturbation"),
        kwargs=kwargs,
    )
    refused_value = _capture_direct_product(
        record,
        side="refused",
        method="eager.dense_cholesky_logdet",
        function=eager.dense_cholesky_logdet,
        args=("sigma_input",),
    )
    _record_selected_product(
        record,
        production_admitted=production_valid,
        oracle_admitted=oracle_valid,
        admitted_method="eager.low_rank_logdet",
        admitted_value=admitted_value,
        refused_method="eager.dense_cholesky_logdet",
        refused_value=refused_value,
        oracle_value=_payload_oracle(record.values["problem"]),
    )
    record.realization(
        point_key="factor_mismatch",
        value=mismatch,
        threshold=threshold,
        axis_key="perturbation",
        dtype=np.dtype(float).str,
    )
    record.bind_axes(
        factor_presence="rank_factor_presence",
        factor_reconstruction="factor_mismatch",
        perturbation_rank="perturbation_rank",
        lambda_scale="lambda_scale",
        factor_layout="factor_layout",
        factor_gauge="factor_gauge",
        computation_dtype="computation_dtype",
    )
    return record.finish()


def _rho_action(record: _FixtureRecord) -> RawObservation:
    threshold = 0.5
    requested_rho = _float_value(
        record.point,
        threshold,
        very_low=0.0,
        very_high=0.9,
        extreme=math.inf,
    )
    if record.point.role is PointRole.EXTREME:
        lam = np.array([np.nextafter(0.0, 1.0), 2.7])
        perturbation = np.array([np.finfo(float).max, 0.0])
    else:
        lam = np.array([1.3, 2.7])
        perturbation = requested_rho * lam
    _problem(record, lam, perturbation)
    _config(record)
    record.input("lambda_scale", float(record.values["lambda_matrix"].reshape(-1)[0]))
    record.input(
        "perturbation_scale",
        float(record.values["perturbation"].reshape(-1)[0]),
    )
    record.input(
        "matrix_geometry",
        record.values["perturbation"],
        audit_value=_array_descriptor(record.values["perturbation"]),
    )
    record.input("computation_dtype", record.values["lambda_matrix"].dtype.str)
    try:
        measured = float(
            record.invoke(
                "eager.spectral_radius",
                eager.spectral_radius,
                result_key="spectral_radius",
                args=("lambda_matrix", "perturbation"),
            )
        )
        production_valid = math.isfinite(measured)
    except ValueError as error:
        measured = math.inf
        production_valid = False
        record.notes.append(type(error).__name__)
    record.returned("measured_rho", measured)
    verdicts = _check_premises(record)
    premise_measured = float(verdicts[1].details["measured_rho"])
    record.returned("premise_measured_rho", premise_measured)
    production_valid = bool(
        verdicts[1].details["rho_measurement_valid"]
        and math.isfinite(premise_measured)
    )
    try:
        with np.errstate(over="ignore"):
            expected = oracles.spectral_radius(lam, perturbation)
        oracle_valid = math.isfinite(expected)
    except (ValueError, np.linalg.LinAlgError):
        expected = math.inf
        oracle_valid = False
    record.checks.append(
        oracle_check(
            oracle="independent retained spectral-radius measurement",
            actual=premise_measured,
            expected=expected,
        )
    )
    record.observed_side = _side(production_valid)
    record.oracle_side = _side(oracle_valid)
    if production_valid and oracle_valid:
        record.evaluations.append(
            numerical_evaluation(
                method="eager.spectral_radius",
                oracle=_PAYLOAD_ORACLE,
                actual=measured,
                oracle_value=expected,
            )
        )
    payload_rho_value = float(expected) if oracle_valid and expected < 1.0 else 0.5
    if oracle_valid and expected < 1.0:
        # Pick the order a real warmup would certify for this rho, so the
        # admitted trace payload converges into the dense-oracle OK band, and
        # feed the exact same float recurrence the production trace validator
        # recomputes, so ``_power_traces_match`` accepts them bitwise.
        payload_order = oracles.smallest_trace_order(payload_rho_value, 1e-4, 2)
        record.input("payload_trace_order", payload_order)
        record.input(
            "payload_exact_power_traces",
            eager._computed_power_traces(lam, perturbation, payload_order),
        )
    else:
        record.input("payload_trace_order", 2)
        record.input(
            "payload_exact_power_traces",
            oracles.exact_power_traces(lam, perturbation, 2),
        )
    record.input("payload_rho", payload_rho_value)
    record.input(
        "sigma_input",
        np.asarray(record.values["lambda_matrix"])
        + np.asarray(record.values["perturbation"]),
    )
    admitted_value = _capture_direct_product(
        record,
        side="admitted",
        method="eager.truncated_trace_logdet",
        function=eager.truncated_trace_logdet,
        args=("lambda_matrix", "perturbation"),
        kwargs={
            "exact_power_traces": "payload_exact_power_traces",
            "order": "payload_trace_order",
            "rho": "payload_rho",
        },
    )
    refused_value = _capture_direct_product(
        record,
        side="refused",
        method="eager.dense_cholesky_logdet",
        function=eager.dense_cholesky_logdet,
        args=("sigma_input",),
    )
    _record_selected_product(
        record,
        production_admitted=production_valid,
        oracle_admitted=oracle_valid,
        admitted_method="eager.truncated_trace_logdet",
        admitted_value=admitted_value,
        refused_method="eager.dense_cholesky_logdet",
        refused_value=refused_value,
        oracle_value=_payload_oracle(record.values["problem"]),
    )
    record.realization(
        point_key="measured_rho",
        value=measured,
        threshold=threshold,
        axis_key="perturbation",
        dtype=np.dtype(float).str,
    )
    record.bind_axes(
        lambda_scale="lambda_scale",
        perturbation_scale="perturbation_scale",
        matrix_geometry="matrix_geometry",
        computation_dtype="computation_dtype",
    )
    return record.finish()


def _finite_rho_action(record: _FixtureRecord) -> RawObservation:
    threshold = 1.0
    rho = _float_value(
        record.point,
        threshold,
        very_low=0.0,
        very_high=2.0,
        extreme=1e100,
    )
    if record.atom_id is not None:
        rho = math.nextafter(1.0, math.inf)
    if record.atom_id is None and record.point.role in {
        PointRole.AT,
        PointRole.ABOVE_ULP,
    }:
        lam = np.array([1.3, 2.7, 3.1])
        perturbation = np.array([0.5 * lam[0], 0.5 * lam[1], rho * lam[2]])
    else:
        lam = np.array([1.3, 2.7])
        perturbation = rho * lam
    _problem(record, lam, perturbation)
    _config(record)
    record.input("lambda_scale", float(record.values["lambda_matrix"].reshape(-1)[0]))
    record.input(
        "perturbation_scale",
        float(record.values["perturbation"].reshape(-1)[0]),
    )
    record.input(
        "matrix_geometry",
        record.values["perturbation"],
        audit_value=_array_descriptor(record.values["perturbation"]),
    )
    record.input("computation_dtype", record.values["lambda_matrix"].dtype.str)
    record.input("determinant_alternative", None)
    measured = float(
        record.invoke(
            "eager.spectral_radius",
            eager.spectral_radius,
            result_key="finite_payload_rho",
            args=("lambda_matrix", "perturbation"),
        )
    )
    record.returned("measured_rho", measured)
    verdicts = _check_premises(record)
    production_valid = bool(verdicts[5].details["finite_polynomial_stable"])
    oracle_rho = oracles.spectral_radius(lam, perturbation)
    oracle_valid = math.isfinite(oracle_rho) and oracle_rho <= 1.0
    record.observed_side = _side(production_valid)
    record.oracle_side = _side(oracle_valid)
    _atom_premises(
        record,
        {"finite_payload_rho <= 1.0": measured <= 1.0},
        {"finite_payload_rho <= 1.0": oracle_valid},
    )
    if production_valid and oracle_valid:
        _record_payload(
            record,
            method="eager.finite_perturbation_logdet",
            function=eager.finite_perturbation_logdet,
            args=("lambda_matrix", "perturbation"),
        )
    record.input(
        "sigma_input",
        np.asarray(record.values["lambda_matrix"])
        + np.asarray(record.values["perturbation"]),
    )
    admitted_value = _capture_direct_product(
        record,
        side="admitted",
        method="eager.finite_perturbation_logdet",
        function=eager.finite_perturbation_logdet,
        args=("lambda_matrix", "perturbation"),
    )
    refused_value = _capture_direct_product(
        record,
        side="refused",
        method="eager.dense_cholesky_logdet",
        function=eager.dense_cholesky_logdet,
        args=("sigma_input",),
    )
    _record_selected_product(
        record,
        production_admitted=production_valid,
        oracle_admitted=oracle_valid,
        admitted_method="eager.finite_perturbation_logdet",
        admitted_value=admitted_value,
        refused_method="eager.dense_cholesky_logdet",
        refused_value=refused_value,
        oracle_value=_payload_oracle(record.values["problem"]),
    )
    record.realization(
        point_key="measured_rho",
        value=measured,
        threshold=threshold,
        axis_key="perturbation",
        dtype=np.dtype(float).str,
    )
    record.bind_axes(
        lambda_scale="lambda_scale",
        perturbation_scale="perturbation_scale",
        matrix_geometry="matrix_geometry",
        computation_dtype="computation_dtype",
        determinant_alternative="determinant_alternative",
    )
    return record.finish()


def _determinant_lemma_action(record: _FixtureRecord) -> RawObservation:
    mismatch, threshold = _exact_mismatch(record.point)
    lam = np.eye(3) * 2.7
    left = np.array([[0.2], [-0.1], [0.0]])
    right = left.copy()
    perturbation = left @ right.T
    factors: eager.LowRankFactors | None = eager.LowRankFactors(left, right)
    syntax = record.atom_syntax
    if syntax == "problem.low_rank_factors is not None":
        factors = None
    elif syntax == "rank_evidence_valid":
        bad = left.copy()
        bad[2, 0] = 1e-2
        factors = eager.LowRankFactors(bad)
    elif syntax == "sigma_formation_valid":
        maximum = np.finfo(float).max
        lam = np.eye(3) * maximum
        perturbation = np.eye(3) * maximum
        factors = eager.LowRankFactors(np.eye(3) * math.sqrt(maximum))
    elif syntax == "sigma_exactly_symmetric":
        right = np.array([[0.2], [-0.1], [0.1]])
        perturbation = left @ right.T
        factors = eager.LowRankFactors(left, right)
    elif syntax == "condition_resolved":
        lam = np.diag([1.3, 1.3, 1.3 * np.finfo(float).eps])
        left = np.zeros((3, 1))
        perturbation = np.zeros((3, 3))
        factors = eager.LowRankFactors(left)
    elif record.point.role not in {
        PointRole.EXACT,
        PointRole.VERY_LOW,
        PointRole.VERY_HIGH,
        PointRole.EXTREME,
    }:
        if record.point.role is PointRole.ULP_MISMATCH:
            bad = left.copy()
            threshold = float(left[0, 0])
            bad[0, 0] = np.nextafter(left[0, 0], math.inf)
            mismatch = float(bad[0, 0])
            factors = eager.LowRankFactors(bad)
            perturbation[2, 0] = np.nextafter(0.0, -math.inf)
            perturbation[0, 2] = perturbation[2, 0]
        elif record.point.role is PointRole.SUBNORMAL_MISMATCH:
            perturbation[2, 0] = mismatch
            perturbation[0, 2] = mismatch
        else:
            bad = left.copy()
            bad[2, 0] = mismatch
            factors = eager.LowRankFactors(bad)
            perturbation[2, 0] = mismatch
            perturbation[0, 2] = mismatch
    elif record.point.role is PointRole.VERY_HIGH:
        lam = np.eye(3) * 13.0
        left = 2.0 * left
        right = left.copy()
        perturbation = left @ right.T
        factors = eager.LowRankFactors(left, right)
        mismatch = 13.0
    elif record.point.role is PointRole.EXTREME:
        factors = None
        perturbation[2, 0] = 1e100
        mismatch = 1e100
    elif record.point.role is PointRole.VERY_LOW:
        left = 0.5 * left
        right = left.copy()
        perturbation = left @ right.T
        factors = eager.LowRankFactors(left, right)
        mismatch = 0.0
    if record.atom_id is not None:
        mismatch = float(np.asarray(perturbation).reshape(-1)[0])
        threshold = 0.0
    record.input("factor_mismatch", mismatch)
    problem = _problem(
        record,
        lam,
        perturbation,
        low_rank_factors=factors,
    )
    _config(record)
    factor_descriptor = (
        None
        if factors is None
        else {
            "present": True,
            "rank": int(factors.rank_bound),
            "left_shape": tuple(int(item) for item in factors.left.shape),
            "right_shape": tuple(int(item) for item in factors.right.shape),
        }
    )
    record.input(
        "determinant_factor_presence",
        factors,
        audit_value=factor_descriptor,
    )
    record.input("sigma_formation", float(np.asarray(lam).reshape(-1)[0]))
    symmetry_leaf = (
        0.0
        if np.asarray(perturbation).ndim == 1
        else float(perturbation[0, 1] - perturbation[1, 0])
    )
    record.input("sigma_symmetry", symmetry_leaf)
    record.input("condition_scale", float(np.asarray(lam).reshape(-1)[-1]))
    if syntax == "sigma_formation_valid":
        reference_lambda = np.eye(3) * 2.7
        reference_left = np.array([[0.2], [-0.1], [0.0]])
        reference_perturbation = reference_left @ reference_left.T
        reference_factors = eager.LowRankFactors(reference_left)
        production_certificate = ladder._factor_projection_certificate

        def retained_valid_certificate(*_args: Any, **_kwargs: Any) -> Any:
            return production_certificate(
                reference_perturbation,
                reference_factors,
                reference_lambda,
            )

        with patch.object(
            ladder,
            "_factor_projection_certificate",
            side_effect=retained_valid_certificate,
        ):
            verdicts = _check_premises(record)
    else:
        verdicts = _check_premises(record)
    production_valid = bool(verdicts[1].details["determinant_lemma_payload"])
    facts = _independent_facts(problem, record.values["config"])
    oracle_exact = facts.sigma_formation_valid and facts.sigma_exactly_symmetric
    oracle_condition = bool(
        facts.sigma_formation_valid
        and oracles.spectral_condition(facts.sigma)
        < 1.0 / np.finfo(facts.sigma.dtype).eps
    )
    oracle_valid = bool(
        factors is not None
        and facts.rank_evidence_valid
        and facts.sigma_formation_valid
        and oracle_exact
        and oracle_condition
    )
    record.returned("determinant_lemma_payload", production_valid)
    record.observed_side = _side(production_valid)
    record.oracle_side = _side(oracle_valid)
    _atom_premises(
        record,
        {
            "problem.low_rank_factors is not None": problem.low_rank_factors
            is not None,
            "rank_evidence_valid": bool(verdicts[1].details["rank_evidence_valid"]),
            "sigma_formation_valid": bool(verdicts[1].details["sigma_formation_valid"]),
            "sigma_exactly_symmetric": bool(verdicts[1].details["exactly_symmetric"]),
            "condition_resolved": bool(
                verdicts[1].details["condition"]
                < verdicts[1].details["condition_ceiling"]
            ),
        },
        {
            "problem.low_rank_factors is not None": factors is not None,
            "rank_evidence_valid": (
                True if syntax == "sigma_formation_valid" else facts.rank_evidence_valid
            ),
            "sigma_formation_valid": facts.sigma_formation_valid,
            "sigma_exactly_symmetric": oracle_exact,
            "condition_resolved": oracle_condition,
        },
    )
    if production_valid and oracle_valid:
        _record_payload(
            record,
            method="eager.low_rank_logdet",
            function=eager.low_rank_logdet,
            args=("lambda_matrix", "perturbation"),
            kwargs={"factors": "low_rank_factors"},
        )
    with np.errstate(over="ignore", invalid="ignore"):
        record.input(
            "sigma_input",
            np.asarray(record.values["lambda_matrix"])
            + np.asarray(record.values["perturbation"]),
        )
    kwargs = {"factors": "low_rank_factors"} if factors is not None else None
    admitted_value = _capture_direct_product(
        record,
        side="admitted",
        method="eager.low_rank_logdet",
        function=eager.low_rank_logdet,
        args=("lambda_matrix", "perturbation"),
        kwargs=kwargs,
    )
    refused_value = _capture_direct_product(
        record,
        side="refused",
        method="eager.dense_cholesky_logdet",
        function=eager.dense_cholesky_logdet,
        args=("sigma_input",),
    )
    _record_selected_product(
        record,
        production_admitted=production_valid,
        oracle_admitted=oracle_valid,
        admitted_method="eager.low_rank_logdet",
        admitted_value=admitted_value,
        refused_method="eager.dense_cholesky_logdet",
        refused_value=refused_value,
        oracle_value=(
            _payload_oracle(problem) if facts.sigma_formation_valid else 0.0
        ),
    )
    record.realization(
        point_key="factor_mismatch",
        value=mismatch,
        threshold=threshold,
        axis_key="perturbation",
        dtype=np.dtype(float).str,
    )
    record.bind_axes(
        factor_presence="determinant_factor_presence",
        factor_reconstruction="factor_mismatch",
        sigma_formation="sigma_formation",
        sigma_symmetry="sigma_symmetry",
        condition_scale="condition_scale",
    )
    return record.finish()


def _rung0_action(record: _FixtureRecord) -> RawObservation:
    mismatch, threshold = _exact_mismatch(record.point)
    lam = np.diag([1.3, 2.7])
    perturbation = np.zeros((2, 2))
    if record.atom_syntax == "sigma_formation_valid":
        maximum = np.finfo(float).max
        lam = np.diag([maximum, maximum])
        perturbation = np.diag([maximum, maximum])
        mismatch = maximum
    elif record.atom_syntax == "bool(np.array_equal(sigma, lam))":
        perturbation[0, 1] = np.nextafter(0.0, 1.0)
        mismatch = float(perturbation[0, 1])
    elif record.atom_syntax == "dense_arithmetic_resolved":
        lam = np.diag([1.3, 1.3 * np.finfo(float).eps])
        mismatch = 1.0
    elif record.point.role not in {
        PointRole.EXACT,
        PointRole.VERY_LOW,
        PointRole.VERY_HIGH,
        PointRole.EXTREME,
    }:
        perturbation[0, 1] = mismatch
    elif record.point.role is PointRole.VERY_LOW:
        lam = np.diag([1.3, 1.7])
        perturbation[0, 0] = np.nextafter(0.0, math.inf)
    elif record.point.role is PointRole.VERY_HIGH:
        lam = np.diag([2.7, 4.1])
        perturbation[0, 0] = np.nextafter(0.0, -math.inf)
    elif record.point.role is PointRole.EXTREME:
        perturbation[0, 1] = 1e100
        mismatch = 1e100
    if record.atom_id is not None:
        mismatch = float(np.asarray(perturbation).reshape(-1)[0])
        threshold = 0.0
    record.input("zero_perturbation_mismatch", mismatch)
    problem = _problem(record, lam, perturbation)
    _config(record)
    record.input("sigma_formation", float(np.asarray(lam).reshape(-1)[0]))
    record.input(
        "sigma_lambda_equality",
        float(np.asarray(perturbation).reshape(-1)[0]),
    )
    record.input("dense_condition", float(np.asarray(lam).reshape(-1)[-1]))
    verdict = _check_premises(record)[0]
    facts = _independent_facts(problem, record.values["config"])
    exact_equal = facts.sigma_formation_valid and np.array_equal(facts.sigma, lam)
    oracle_valid = (
        facts.sigma_formation_valid and exact_equal and facts.dense_arithmetic_resolved
    )
    record.returned("rung0_satisfied", bool(verdict.satisfied))
    record.observed_side = _side(bool(verdict.satisfied))
    record.oracle_side = _side(oracle_valid)
    production_formation = bool(verdict.details["sigma_formation_valid"])
    production_equal = production_formation and bool(
        np.array_equal(np.asarray(lam) + np.asarray(perturbation), lam)
    )
    production_dense = production_formation and bool(
        np.asarray(lam).ndim == 1
        or verdict.details["condition"] < verdict.details["condition_ceiling"]
    )
    _atom_premises(
        record,
        {
            "sigma_formation_valid": production_formation,
            "bool(np.array_equal(sigma, lam))": production_equal,
            "dense_arithmetic_resolved": production_dense,
        },
        {
            "sigma_formation_valid": facts.sigma_formation_valid,
            "bool(np.array_equal(sigma, lam))": exact_equal,
            "dense_arithmetic_resolved": facts.dense_arithmetic_resolved,
        },
    )
    if record.atom_id is None:
        _record_payload_attempt(
            record,
            method="eager.lambda_logdet",
            function=eager.lambda_logdet,
            args=("lambda_matrix",),
        )
    elif oracle_valid:
        _record_payload(
            record,
            method="eager.lambda_logdet",
            function=eager.lambda_logdet,
            args=("lambda_matrix",),
        )
    with np.errstate(over="ignore", invalid="ignore"):
        record.input(
            "sigma_input",
            np.asarray(record.values["lambda_matrix"])
            + np.asarray(record.values["perturbation"]),
        )
    admitted_value = _capture_direct_product(
        record,
        side="admitted",
        method="eager.lambda_logdet",
        function=eager.lambda_logdet,
        args=("lambda_matrix",),
    )
    refused_value = _capture_direct_product(
        record,
        side="refused",
        method="eager.dense_cholesky_logdet",
        function=eager.dense_cholesky_logdet,
        args=("sigma_input",),
    )
    _record_selected_product(
        record,
        production_admitted=bool(verdict.satisfied),
        oracle_admitted=oracle_valid,
        admitted_method="eager.lambda_logdet",
        admitted_value=admitted_value,
        refused_method="eager.dense_cholesky_logdet",
        refused_value=refused_value,
        oracle_value=(
            _payload_oracle(problem) if facts.sigma_formation_valid else 0.0
        ),
    )
    record.realization(
        point_key="zero_perturbation_mismatch",
        value=mismatch,
        threshold=threshold,
        axis_key="perturbation",
        dtype=np.dtype(float).str,
    )
    record.bind_axes(
        sigma_formation="sigma_formation",
        sigma_lambda_equality="sigma_lambda_equality",
        dense_condition="dense_condition",
    )
    return record.finish()


def _rank_problem(
    record: _FixtureRecord,
    rank: int,
    *,
    dimension: int = 4,
    compact: bool = True,
    bad_factors: bool = False,
    sigma_zero: bool = False,
) -> eager.LogDetProblem:
    if compact:
        lam = np.linspace(1.3, 2.7, dimension)
        perturbation = np.zeros(dimension)
        perturbation[:rank] = 0.2
        if sigma_zero:
            perturbation[0] = -lam[0]
        return _problem(record, lam, perturbation)
    columns = max(rank, 1)
    left = np.zeros((dimension, columns))
    for index in range(min(dimension, columns)):
        left[index, index] = 0.2 + 0.01 * index
    perturbation = left @ left.T
    factors_left = left.copy()
    if bad_factors:
        factors_left[-1, 0] = 1e-2
    return _problem(
        record,
        np.eye(dimension) * 2.7,
        perturbation,
        low_rank_factors=eager.LowRankFactors(factors_left),
    )


def _rung1_action(record: _FixtureRecord) -> RawObservation:
    threshold = 2
    rank = _integer_value(record.point, threshold, very_low=0, very_high=4, extreme=5)
    dimension = 8
    dimension_axis = record.active_axis == "rank_evidence"
    if record.atom_id is None and record.point.role is PointRole.VERY_LOW:
        dimension = 1
    elif (
        record.atom_id is None
        and dimension_axis
        and record.point.role is PointRole.VERY_HIGH
    ):
        dimension = 257
    elif (
        record.atom_id is None
        and dimension_axis
        and record.point.role is PointRole.EXTREME
    ):
        dimension = 10_000
    if record.atom_id is not None:
        dimension = 3
    compact = True
    bad_factors = False
    sigma_zero = False
    fraction = 1.0
    max_rank = threshold
    syntax = record.atom_syntax
    if record.point.display_value == _EQUALITY_WITNESS:
        rank = 2
        dimension = 8
        fraction = 0.25
        max_rank = 3
    elif record.point.display_value == _ADJACENT_WITNESS:
        rank = 3
        dimension = 8
        fraction = math.nextafter(3.0 / dimension, 0.0)
        max_rank = 3
    elif syntax == "rank_evidence_valid":
        compact = False
        rank = 1
        bad_factors = True
    elif syntax == "compact_diagonal_payload" or syntax == "determinant_lemma_payload":
        compact = False
        rank = 1
    elif syntax == "sigma_spd":
        rank = 1
        sigma_zero = True
    elif syntax == "rank <= config.low_rank_max":
        rank = 3
        max_rank = 2
    elif syntax == "rank <= config.low_rank_fraction * n":
        rank = 1
        fraction = math.nextafter(1.0 / 3.0, 0.0)
    problem = _rank_problem(
        record,
        max(rank, 0),
        dimension=dimension,
        compact=compact,
        bad_factors=bad_factors,
        sigma_zero=sigma_zero,
    )
    _config(record, low_rank_max=max_rank, low_rank_fraction=fraction)
    record.input(
        "rank_evidence",
        problem.low_rank_factors,
        audit_value=(
            None
            if problem.low_rank_factors is None
            else {
                "present": True,
                "rank": int(problem.low_rank_factors.rank_bound),
                "left_shape": tuple(
                    int(item) for item in problem.low_rank_factors.left.shape
                ),
                "right_shape": tuple(
                    int(item) for item in problem.low_rank_factors.right.shape
                ),
            }
        ),
    )
    record.input(
        "payload_capability",
        problem.perturbation,
        audit_value=_array_descriptor(problem.perturbation),
    )
    record.input("sigma_spd", float(problem.perturbation.reshape(-1)[0]))
    record.input("rank_input", int(rank))
    record.input("dimension", int(dimension))
    verdict = _check_premises(record)[1]
    facts = _independent_facts(problem, record.values["config"])
    n = int(problem.lambda_matrix.shape[-1])
    determinant = facts.determinant_lemma_payload
    oracle_valid = bool(
        facts.rank_evidence_valid
        and (facts.compact_diagonal_payload or determinant)
        and facts.sigma_spd
        and facts.rank <= max_rank
        and facts.rank <= fraction * n
    )
    record.returned("rung1_satisfied", bool(verdict.satisfied))
    record.returned("certified_rank", int(verdict.details["rank"]))
    record.observed_side = _side(bool(verdict.satisfied))
    record.oracle_side = _side(oracle_valid)
    _atom_premises(
        record,
        {
            "rank_evidence_valid": bool(verdict.details["rank_evidence_valid"]),
            "compact_diagonal_payload": problem.perturbation.ndim == 1,
            "determinant_lemma_payload": bool(
                verdict.details["determinant_lemma_payload"]
            ),
            "sigma_spd": bool(verdict.details["positive_definite"]),
            "rank <= config.low_rank_max": int(verdict.details["rank"])
            <= record.values["config"].low_rank_max,
            "rank <= config.low_rank_fraction * n": int(verdict.details["rank"])
            <= record.values["config"].low_rank_fraction * n,
        },
        {
            "rank_evidence_valid": facts.rank_evidence_valid,
            "compact_diagonal_payload": facts.compact_diagonal_payload,
            "determinant_lemma_payload": determinant,
            "sigma_spd": facts.sigma_spd,
            "rank <= config.low_rank_max": facts.rank <= max_rank,
            "rank <= config.low_rank_fraction * n": facts.rank <= fraction * n,
        },
    )
    if record.atom_id is None:
        _record_payload_attempt(
            record,
            method="eager.low_rank_logdet",
            function=eager.low_rank_logdet,
            args=("lambda_matrix", "perturbation"),
            kwargs={"factors": "low_rank_factors"}
            if problem.low_rank_factors is not None
            else None,
        )
    elif verdict.satisfied and oracle_valid:
        _record_payload(
            record,
            method="eager.low_rank_logdet",
            function=eager.low_rank_logdet,
            args=("lambda_matrix", "perturbation"),
            kwargs={"factors": "low_rank_factors"}
            if problem.low_rank_factors is not None
            else None,
        )
    record.input(
        "sigma_input",
        np.asarray(problem.lambda_matrix) + np.asarray(problem.perturbation),
    )
    kwargs = (
        {"factors": "low_rank_factors"}
        if problem.low_rank_factors is not None
        else None
    )
    admitted_value = _capture_direct_product(
        record,
        side="admitted",
        method="eager.low_rank_logdet",
        function=eager.low_rank_logdet,
        args=("lambda_matrix", "perturbation"),
        kwargs=kwargs,
    )
    refused_value = _capture_direct_product(
        record,
        side="refused",
        method="eager.dense_cholesky_logdet",
        function=eager.dense_cholesky_logdet,
        args=("sigma_input",),
    )
    _record_selected_product(
        record,
        production_admitted=bool(verdict.satisfied),
        oracle_admitted=oracle_valid,
        admitted_method="eager.low_rank_logdet",
        admitted_value=admitted_value,
        refused_method="eager.dense_cholesky_logdet",
        refused_value=refused_value,
        oracle_value=_payload_oracle(problem) if facts.sigma_spd else 0.0,
    )
    record.realization(
        point_key="certified_rank",
        value=int(verdict.details["rank"]),
        threshold=threshold,
        axis_key="perturbation",
    )
    record.bind_axes(
        rank_evidence="rank_evidence",
        payload_capability="payload_capability",
        sigma_spd="sigma_spd",
        rank="perturbation",
        dimension="dimension",
        low_rank_max="config_low_rank_max",
        low_rank_fraction="config_low_rank_fraction",
    )
    return record.finish()


def _chain_matrix(delta: float = 0.0, *, pivot: float = 2.7) -> np.ndarray:
    matrix = np.array(
        [
            [pivot, 0.1, 0.0, delta],
            [0.1, 2.7, 0.1, 0.0],
            [0.0, 0.1, 2.7, 0.1],
            [delta, 0.0, 0.1, 2.7],
        ]
    )
    return matrix


def _rung2_action(record: _FixtureRecord) -> RawObservation:
    mismatch, threshold = _exact_mismatch(record.point)
    pivot = 2.7
    condition_fixture = False
    if record.atom_syntax == "chain_structure":
        mismatch = float(np.nextafter(0.0, 1.0))
    elif record.atom_syntax == "sigma_spd":
        mismatch = 0.0
        pivot = -1.0
    elif record.atom_syntax == "condition_resolved":
        mismatch = 0.0
        condition_fixture = True
    elif record.point.role is PointRole.EXTREME:
        mismatch = 1e100
    sigma = (
        np.diag([1.3, 2.7, 2.7, 2.7 * np.finfo(float).eps])
        if condition_fixture
        else _chain_matrix(mismatch, pivot=pivot)
    )
    lambda_matrix = sigma.copy() if condition_fixture else np.eye(4) * 1.3
    record.input("far_block_mismatch", mismatch)
    problem = _problem(
        record,
        lambda_matrix,
        sigma - lambda_matrix,
        chain_block_size=1,
    )
    _config(record)
    record.input("sigma_input", sigma)
    record.input("chain_layout", mismatch)
    record.input("sigma_formation", float(np.asarray(lambda_matrix).reshape(-1)[0]))
    record.input("sigma_spd", float(np.asarray(sigma - lambda_matrix).reshape(-1)[0]))
    record.input("condition_scale", float(np.asarray(sigma).reshape(-1)[-1]))
    record.input("computation_dtype", problem.lambda_matrix.dtype.str)
    captured_chain: list[bool] = []
    production_chain_helper = ladder._is_block_chain

    def capture_chain(*args: Any, **kwargs: Any) -> bool:
        result = bool(production_chain_helper(*args, **kwargs))
        captured_chain.append(result)
        return result

    with patch.object(ladder, "_is_block_chain", side_effect=capture_chain):
        verdicts = _check_premises(record)
    verdict = verdicts[2]
    if len(captured_chain) != 1:
        raise AssertionError("the direct ladder call did not evaluate chain structure")
    chain_helper = captured_chain[0]
    facts = _independent_facts(problem, record.values["config"])
    raw_condition_resolved = bool(
        oracles.spectral_condition(sigma) < 1.0 / np.finfo(sigma.dtype).eps
    )
    oracle_valid = (
        facts.chain_structure and facts.sigma_spd and facts.condition_resolved
    )
    record.returned("chain_helper", chain_helper)
    record.returned("rung2_satisfied", bool(verdict.satisfied))
    record.observed_side = _side(bool(verdict.satisfied))
    record.oracle_side = _side(oracle_valid)
    _atom_premises(
        record,
        {
            "chain_structure": chain_helper,
            "sigma_spd": bool(verdicts[4].details["positive_definite"]),
            "condition_resolved": bool(
                verdicts[4].details["condition"]
                < verdicts[4].details["condition_ceiling"]
            ),
        },
        {
            "chain_structure": facts.chain_structure,
            "sigma_spd": facts.sigma_spd,
            "condition_resolved": raw_condition_resolved,
        },
    )
    if record.atom_id is None:
        _record_payload_attempt(
            record,
            method="eager.state_space_logdet",
            function=eager.state_space_logdet,
            args=("sigma_input",),
            kwargs={"block_size": "chain_block_size"},
        )
    elif oracle_valid:
        _record_payload(
            record,
            method="eager.state_space_logdet",
            function=eager.state_space_logdet,
            args=("sigma_input",),
            kwargs={"block_size": "chain_block_size"},
        )
    admitted_value = _capture_direct_product(
        record,
        side="admitted",
        method="eager.state_space_logdet",
        function=eager.state_space_logdet,
        args=("sigma_input",),
        kwargs={"block_size": "chain_block_size"},
    )
    refused_value = _capture_direct_product(
        record,
        side="refused",
        method="eager.dense_cholesky_logdet",
        function=eager.dense_cholesky_logdet,
        args=("sigma_input",),
    )
    _record_selected_product(
        record,
        production_admitted=bool(verdict.satisfied),
        oracle_admitted=oracle_valid,
        admitted_method="eager.state_space_logdet",
        admitted_value=admitted_value,
        refused_method="eager.dense_cholesky_logdet",
        refused_value=refused_value,
        oracle_value=_payload_oracle(problem) if facts.sigma_spd else 0.0,
    )
    record.realization(
        point_key="far_block_mismatch",
        value=mismatch,
        threshold=threshold,
        axis_key="perturbation",
        dtype=np.dtype(float).str,
    )
    record.bind_axes(
        chain_block_size="chain_block_size",
        chain_layout="chain_layout",
        sigma_formation="sigma_formation",
        sigma_spd="sigma_spd",
        condition_scale="condition_scale",
        computation_dtype="computation_dtype",
    )
    return record.finish()


def _rung3_action(record: _FixtureRecord) -> RawObservation:
    mismatch, threshold = _exact_mismatch(record.point)
    if record.point.role is PointRole.EXTREME:
        mismatch = 1e100
    sigma = np.array([[1.3, mismatch], [0.0, 2.7]])
    record.input("structure_mismatch", mismatch)
    structure_kind = record.input("structure_kind", "diagonal")
    problem = _problem(
        record,
        np.eye(2),
        sigma - np.eye(2),
        structure_kind=structure_kind,
    )
    _config(record)
    record.input("sigma_input", sigma)
    record.input("structure_evidence", mismatch)
    record.input("sigma_formation", float(problem.lambda_matrix.reshape(-1)[0]))
    record.input("sigma_spd", float(problem.perturbation.reshape(-1)[0]))
    record.input("condition_scale", float(np.asarray(sigma).reshape(-1)[-1]))
    verdict = _check_premises(record)[3]
    facts = _independent_facts(problem, record.values["config"])
    record.returned("rung3_satisfied", bool(verdict.satisfied))
    record.observed_side = _side(bool(verdict.satisfied))
    record.oracle_side = _side(facts.structured)
    if record.atom_id is None:
        _record_payload_attempt(
            record,
            method="eager.structured_logdet",
            function=eager.structured_logdet,
            args=("sigma_input",),
            kwargs={"kind": "structure_kind"},
        )
    elif facts.structured:
        _record_payload(
            record,
            method="eager.structured_logdet",
            function=eager.structured_logdet,
            args=("sigma_input",),
            kwargs={"kind": "structure_kind"},
        )
    admitted_value = _capture_direct_product(
        record,
        side="admitted",
        method="eager.structured_logdet",
        function=eager.structured_logdet,
        args=("sigma_input",),
        kwargs={"kind": "structure_kind"},
    )
    refused_value = _capture_direct_product(
        record,
        side="refused",
        method="eager.dense_cholesky_logdet",
        function=eager.dense_cholesky_logdet,
        args=("sigma_input",),
    )
    _record_selected_product(
        record,
        production_admitted=bool(verdict.satisfied),
        oracle_admitted=facts.structured,
        admitted_method="eager.structured_logdet",
        admitted_value=admitted_value,
        refused_method="eager.dense_cholesky_logdet",
        refused_value=refused_value,
        oracle_value=_payload_oracle(problem) if facts.sigma_spd else 0.0,
    )
    record.realization(
        point_key="structure_mismatch",
        value=mismatch,
        threshold=threshold,
        axis_key="perturbation",
        dtype=np.dtype(float).str,
    )
    record.bind_axes(
        structure_request="structure_kind",
        structure_evidence="structure_evidence",
        sigma_formation="sigma_formation",
        sigma_spd="sigma_spd",
        condition_scale="condition_scale",
    )
    return record.finish()


def _rung4_action(record: _FixtureRecord) -> RawObservation:
    threshold = 2
    n = _integer_value(record.point, threshold, very_low=1, very_high=5, extreme=8)
    maximum_n = threshold
    diagonal = np.linspace(1.3, 2.7, max(n, 1))
    if record.atom_syntax == "n <= config.dense_max_n":
        n = 3
        maximum_n = 2
        diagonal = np.linspace(1.3, 2.7, n)
    elif record.atom_syntax == "condition_resolved":
        n = 2
        diagonal = np.array([1.3, 1.3 * np.finfo(float).eps])
    elif record.atom_syntax == "sigma_spd":
        n = 2
        diagonal = np.array([-1.3, 2.7])
    sigma = np.diag(diagonal)
    record.input("dimension", n)
    lambda_matrix = np.eye(n) * (
        1.1
        if record.atom_id is None and record.point.role is PointRole.VERY_LOW
        else 1.3
    )
    problem = _problem(record, lambda_matrix, sigma - lambda_matrix)
    _config(record, dense_max_n=maximum_n)
    record.input("sigma_input", sigma)
    record.input("condition_scale", float(np.asarray(diagonal).reshape(-1)[-1]))
    record.input("sigma_spd", float(np.asarray(problem.perturbation).reshape(-1)[0]))
    record.input("computation_dtype", problem.lambda_matrix.dtype.str)
    verdict = _check_premises(record)[4]
    facts = _independent_facts(problem, record.values["config"])
    raw_condition_resolved = bool(
        oracles.spectral_condition(sigma) < 1.0 / np.finfo(sigma.dtype).eps
    )
    oracle_valid = n <= maximum_n and facts.condition_resolved and facts.sigma_spd
    record.returned("rung4_satisfied", bool(verdict.satisfied))
    record.observed_side = _side(bool(verdict.satisfied))
    record.oracle_side = _side(oracle_valid)
    _atom_premises(
        record,
        {
            "n <= config.dense_max_n": int(verdict.details["n"])
            <= record.values["config"].dense_max_n,
            "condition_resolved": bool(
                verdict.details["condition"] < verdict.details["condition_ceiling"]
            ),
            "sigma_spd": bool(verdict.details["positive_definite"]),
        },
        {
            "n <= config.dense_max_n": n <= maximum_n,
            "condition_resolved": raw_condition_resolved,
            "sigma_spd": facts.sigma_spd,
        },
    )
    if record.atom_id is None:
        _record_payload_attempt(
            record,
            method="eager.dense_cholesky_logdet",
            function=eager.dense_cholesky_logdet,
            args=("sigma_input",),
        )
    elif oracle_valid:
        _record_payload(
            record,
            method="eager.dense_cholesky_logdet",
            function=eager.dense_cholesky_logdet,
            args=("sigma_input",),
        )
    admitted_value = _capture_direct_product(
        record,
        side="admitted",
        method="eager.dense_cholesky_logdet",
        function=eager.dense_cholesky_logdet,
        args=("sigma_input",),
    )
    refused_value = _capture_direct_product(
        record,
        side="refused",
        method="eager.finite_perturbation_logdet",
        function=eager.finite_perturbation_logdet,
        args=("lambda_matrix", "perturbation"),
    )
    _record_selected_product(
        record,
        production_admitted=bool(verdict.satisfied),
        oracle_admitted=oracle_valid,
        admitted_method="eager.dense_cholesky_logdet",
        admitted_value=admitted_value,
        refused_method="eager.finite_perturbation_logdet",
        refused_value=refused_value,
        oracle_value=_payload_oracle(problem) if facts.sigma_spd else 0.0,
    )
    record.realization(
        point_key="dimension",
        value=n,
        threshold=threshold,
        axis_key="lambda_matrix",
    )
    record.bind_axes(
        dimension="dimension",
        dense_max_n="config_dense_max_n",
        condition_scale="condition_scale",
        sigma_spd="sigma_spd",
        computation_dtype="computation_dtype",
    )
    return record.finish()


def _rung5_size_action(record: _FixtureRecord) -> RawObservation:
    threshold = 2
    n = _integer_value(record.point, threshold, very_low=1, very_high=5, extreme=9)
    finite_max_n = threshold
    finite_max_rank = 0
    compact = False
    rank = 0
    syntax = record.atom_syntax
    if syntax == "n <= config.finite_max_n":
        n = 3
    elif (
        syntax
        == "(compact_diagonal_payload or determinant_lemma_payload) and rank <= config.finite_max_rank"
    ) or syntax == "compact_diagonal_payload or determinant_lemma_payload":
        n = 3
        compact = True
        rank = 1
        finite_max_rank = 1
    elif syntax == "rank <= config.finite_max_rank":
        n = 3
        compact = True
        rank = 2
        finite_max_rank = 1
    lam = (
        np.linspace(1.3, 2.7, n)
        if compact
        else np.eye(n)
        * (
            1.1
            if record.atom_id is None and record.point.role is PointRole.VERY_LOW
            else 1.3
        )
    )
    if compact:
        perturbation = np.zeros(n)
        perturbation[: min(rank, n)] = 0.1
    else:
        perturbation = np.zeros((n, n))
        for index in range(min(rank, n)):
            perturbation[index, index] = 0.1
    record.input("dimension", n)
    problem = _problem(record, lam, perturbation)
    config = _config(
        record,
        finite_max_n=finite_max_n,
        finite_max_rank=finite_max_rank,
    )
    record.input(
        "payload_capability",
        problem.perturbation,
        audit_value=_array_descriptor(problem.perturbation),
    )
    record.input("rank_input", int(rank))
    verdicts = _check_premises(record)
    verdict = verdicts[5]
    facts = _independent_facts(problem, config)
    production_n = int(verdict.details["n"])
    production_rank = int(verdict.details["rank"])
    production_compact = problem.perturbation.ndim == 1
    production_determinant = bool(verdict.details["determinant_lemma_payload"])
    production_or = production_compact or production_determinant
    production_rank_ok = production_rank <= config.finite_max_rank
    production_parent = production_or and production_rank_ok
    oracle_or = facts.compact_diagonal_payload or facts.determinant_lemma_payload
    oracle_rank_ok = facts.rank <= config.finite_max_rank
    oracle_parent = oracle_or and oracle_rank_ok
    oracle_size = n <= config.finite_max_n or oracle_parent
    record.returned("finite_size_qualified", bool(verdict.satisfied))
    record.observed_side = _side(bool(verdict.satisfied))
    record.oracle_side = _side(oracle_size)
    _atom_premises(
        record,
        {
            "n <= config.finite_max_n": production_n <= config.finite_max_n,
            "(compact_diagonal_payload or determinant_lemma_payload) and rank <= config.finite_max_rank": production_parent,
            "compact_diagonal_payload or determinant_lemma_payload": production_or,
            "rank <= config.finite_max_rank": production_rank_ok,
        },
        {
            "n <= config.finite_max_n": n <= config.finite_max_n,
            "(compact_diagonal_payload or determinant_lemma_payload) and rank <= config.finite_max_rank": oracle_parent,
            "compact_diagonal_payload or determinant_lemma_payload": oracle_or,
            "rank <= config.finite_max_rank": oracle_rank_ok,
        },
    )
    if record.atom_id is None:
        _record_payload_attempt(
            record,
            method="eager.finite_perturbation_logdet",
            function=eager.finite_perturbation_logdet,
            args=("lambda_matrix", "perturbation"),
        )
    elif verdict.satisfied and oracle_size:
        _record_payload(
            record,
            method="eager.finite_perturbation_logdet",
            function=eager.finite_perturbation_logdet,
            args=("lambda_matrix", "perturbation"),
        )
    record.input(
        "sigma_input",
        np.asarray(problem.lambda_matrix) + np.asarray(problem.perturbation),
    )
    admitted_value = _capture_direct_product(
        record,
        side="admitted",
        method="eager.finite_perturbation_logdet",
        function=eager.finite_perturbation_logdet,
        args=("lambda_matrix", "perturbation"),
    )
    refused_value = _capture_direct_product(
        record,
        side="refused",
        method="eager.dense_cholesky_logdet",
        function=eager.dense_cholesky_logdet,
        args=("sigma_input",),
    )
    _record_selected_product(
        record,
        production_admitted=bool(verdict.satisfied),
        oracle_admitted=oracle_size,
        admitted_method="eager.finite_perturbation_logdet",
        admitted_value=admitted_value,
        refused_method="eager.dense_cholesky_logdet",
        refused_value=refused_value,
        oracle_value=_payload_oracle(problem),
    )
    record.realization(
        point_key="dimension",
        value=n,
        threshold=threshold,
        axis_key="lambda_matrix",
    )
    record.bind_axes(
        dimension="dimension",
        finite_max_n="config_finite_max_n",
        payload_capability="payload_capability",
        rank="rank_input",
        finite_max_rank="config_finite_max_rank",
    )
    return record.finish()


def _rung5_executable_action(record: _FixtureRecord) -> RawObservation:
    mismatch, threshold = _exact_mismatch(record.point)
    lam: np.ndarray = np.array([1.3, 2.7])
    perturbation: np.ndarray = np.array([0.1, -0.1])
    factors: eager.LowRankFactors | None = None
    config_kwargs: dict[str, Any] = {}
    syntax = record.atom_syntax
    if record.atom_id is None:
        lam = np.array([10.0, 10.0])
        unstable = record.point.role in {
            PointRole.ULP_MISMATCH,
            PointRole.SUBNORMAL_MISMATCH,
            PointRole.MATERIAL_MISMATCH,
        }
        perturbation = np.array([mismatch, 11.0 if unstable else 0.1])
    elif syntax == "finite_size_qualified":
        lam = np.eye(3) * 1.3
        perturbation = np.eye(3) * 0.1
        config_kwargs = {"finite_max_n": 2, "finite_max_rank": 0}
    elif syntax == "finite_payload_stable":
        perturbation = 1.1 * lam
    elif syntax == "sigma_spd":
        perturbation = np.array([-1.3, -0.1])
    elif syntax == "determinant_lemma_payload":
        lam = np.eye(3) * 2.7
        left = np.array([[0.2], [-0.1], [0.05]])
        perturbation = left @ left.T
        factors = eager.LowRankFactors(left)
    elif syntax == "dense_arithmetic_resolved":
        lam = np.diag([1.3, 1.3 * np.finfo(float).eps])
        perturbation = np.zeros((2, 2))
    if record.atom_id is not None:
        mismatch = float(np.asarray(perturbation).reshape(-1)[0])
        threshold = 0.0
    record.input("execution_mismatch", mismatch)
    problem = _problem(
        record,
        lam,
        perturbation,
        low_rank_factors=factors,
    )
    config = _config(record, **config_kwargs)
    dimension = int(problem.lambda_matrix.shape[-1])
    record.input("dimension", dimension)
    record.input(
        "payload_capability",
        problem.perturbation,
        audit_value=_array_descriptor(problem.perturbation),
    )
    rank_input = (
        int(problem.low_rank_factors.rank_bound)
        if problem.low_rank_factors is not None
        else int(np.count_nonzero(problem.perturbation))
    )
    record.input("rank_input", rank_input)
    record.input("lambda_scale", float(problem.lambda_matrix.reshape(-1)[0]))
    record.input("perturbation_scale", float(problem.perturbation.reshape(-1)[0]))
    record.input("sigma_formation", float(problem.lambda_matrix.reshape(-1)[-1]))
    record.input("smallest_eigenvalue", float(problem.perturbation.reshape(-1)[-1]))
    symmetry_leaf = (
        0.0
        if problem.perturbation.ndim == 1
        else float(problem.perturbation[0, 1] - problem.perturbation[1, 0])
    )
    record.input("sigma_symmetry", symmetry_leaf)
    factor_descriptor = (
        None
        if factors is None
        else {
            "present": True,
            "rank": int(factors.rank_bound),
            "left_shape": tuple(int(item) for item in factors.left.shape),
            "right_shape": tuple(int(item) for item in factors.right.shape),
        }
    )
    record.input(
        "determinant_factor_presence",
        factors,
        audit_value=factor_descriptor,
    )
    record.input("factor_reconstruction", mismatch)
    record.input("dense_condition", float(problem.lambda_matrix.reshape(-1)[-1]))
    record.input("computation_dtype", problem.lambda_matrix.dtype.str)
    verdicts = _check_premises(record)
    verdict = verdicts[5]
    facts = _independent_facts(problem, config)
    details = verdict.details
    production_size = int(details["n"]) <= config.finite_max_n or (
        (problem.perturbation.ndim == 1 or bool(details["determinant_lemma_payload"]))
        and int(details["rank"]) <= config.finite_max_rank
    )
    production_stable = bool(details["finite_polynomial_stable"]) or bool(
        details["determinant_lemma_payload"]
    )
    production_spd = bool(details["positive_definite"])
    production_determinant = bool(details["determinant_lemma_payload"])
    production_dense = bool(details["sigma_formation_valid"]) and (
        problem.lambda_matrix.ndim == 1
        or details["condition"] < details["condition_ceiling"]
    )
    oracle_valid = bool(
        facts.finite_size_qualified
        and facts.finite_payload_stable
        and facts.sigma_spd
        and (facts.determinant_lemma_payload or facts.dense_arithmetic_resolved)
    )
    record.returned("rung5_satisfied", bool(verdict.satisfied))
    record.observed_side = _side(bool(verdict.satisfied))
    record.oracle_side = _side(oracle_valid)
    _atom_premises(
        record,
        {
            "finite_size_qualified": production_size,
            "finite_payload_stable": production_stable,
            "sigma_spd": production_spd,
            "determinant_lemma_payload": production_determinant,
            "dense_arithmetic_resolved": production_dense,
        },
        {
            "finite_size_qualified": facts.finite_size_qualified,
            "finite_payload_stable": facts.finite_payload_stable,
            "sigma_spd": facts.sigma_spd,
            "determinant_lemma_payload": facts.determinant_lemma_payload,
            "dense_arithmetic_resolved": facts.dense_arithmetic_resolved,
        },
    )
    kwargs = {"factors": "low_rank_factors"} if factors is not None else None
    if record.atom_id is None:
        _record_payload_attempt(
            record,
            method="eager.finite_perturbation_logdet",
            function=eager.finite_perturbation_logdet,
            args=("lambda_matrix", "perturbation"),
            kwargs=kwargs,
        )
    elif verdict.satisfied and oracle_valid:
        _record_payload(
            record,
            method="eager.finite_perturbation_logdet",
            function=eager.finite_perturbation_logdet,
            args=("lambda_matrix", "perturbation"),
            kwargs=kwargs,
        )
    with np.errstate(over="ignore", invalid="ignore"):
        record.input(
            "sigma_input",
            np.asarray(problem.lambda_matrix) + np.asarray(problem.perturbation),
        )
    admitted_value = _capture_direct_product(
        record,
        side="admitted",
        method="eager.finite_perturbation_logdet",
        function=eager.finite_perturbation_logdet,
        args=("lambda_matrix", "perturbation"),
        kwargs=kwargs,
    )
    refused_value = _capture_direct_product(
        record,
        side="refused",
        method="eager.dense_cholesky_logdet",
        function=eager.dense_cholesky_logdet,
        args=("sigma_input",),
    )
    _record_selected_product(
        record,
        production_admitted=bool(verdict.satisfied),
        oracle_admitted=oracle_valid,
        admitted_method="eager.finite_perturbation_logdet",
        admitted_value=admitted_value,
        refused_method="eager.dense_cholesky_logdet",
        refused_value=refused_value,
        oracle_value=_payload_oracle(problem) if facts.sigma_spd else 0.0,
    )
    record.realization(
        point_key="execution_mismatch",
        value=mismatch,
        threshold=threshold,
        axis_key="perturbation",
        dtype=np.dtype(float).str,
    )
    record.bind_axes(
        dimension="dimension",
        finite_max_n="config_finite_max_n",
        payload_capability="payload_capability",
        rank="rank_input",
        finite_max_rank="config_finite_max_rank",
        lambda_scale="lambda_scale",
        perturbation_scale="perturbation_scale",
        sigma_formation="sigma_formation",
        smallest_eigenvalue="smallest_eigenvalue",
        sigma_symmetry="perturbation",
        factor_presence="determinant_factor_presence",
        factor_reconstruction="factor_reconstruction",
        dense_condition="dense_condition",
        computation_dtype="computation_dtype",
    )
    return record.finish()


def _trace_fixture(
    record: _FixtureRecord,
    *,
    actual_rho: float,
    certificate: float,
    order: int | None,
    traces_valid: bool = True,
    overflow: bool = False,
    frozen_width: int | None = None,
    dimension: int = 2,
    probe_count: int = 2,
) -> eager.LogDetProblem:
    if overflow:
        maximum = np.finfo(float).max
        lam = np.array([maximum, maximum])
        perturbation = np.array([maximum, maximum])
    elif dimension == 2:
        # Rung 6 validates the supplied exact power traces bitwise against its
        # own float recurrence.  Exact powers of two scale without rounding,
        # so the Decimal-exact traces and the float recurrence agree bitwise
        # at the orders this fixture feeds; larger rung-7 fixtures do not
        # validate traces and keep the spread linspace.
        lam = np.array([2.0, 4.0])
        perturbation = actual_rho * lam
    else:
        lam = np.linspace(1.3, 2.7, dimension)
        perturbation = actual_rho * lam
    trace_count = 0 if order is None else max(order, 0)
    traces = list(oracles.exact_power_traces(lam, perturbation, trace_count))
    if not traces_valid:
        if traces:
            traces[0] = math.nextafter(traces[0], math.inf)
        else:
            traces = [1.0]
    constructor_order = None if order is None else max(order, 0)
    kwargs: dict[str, Any] = {
        "trace_order": constructor_order,
        "certified_rho": certificate,
        "exact_power_traces": tuple(traces) if order is not None else None,
    }
    if frozen_width is not None:
        rows = np.arange(probe_count, dtype=np.int64)[:, None]
        columns = np.arange(frozen_width, dtype=np.int64)[None, :]
        probes = np.where((rows + columns) % 2 == 0, 1.0, -1.0)
        kwargs["frozen_probes"] = eager.FrozenProbes(probes)
    problem = _problem(record, lam, perturbation, **kwargs)
    if order is not None and order < 0:
        # The public value object rejects negative orders before the ladder can
        # exercise its own source premise.  Retain a validly constructed object
        # and then realize the invalid fixture field at the direct-call seam.
        object.__setattr__(problem, "trace_order", order)
        record.input("trace_order", order)
    return problem


def _rung6_action(record: _FixtureRecord) -> RawObservation:
    threshold = 1.0
    certificate = _float_value(
        record.point,
        threshold,
        very_low=0.0,
        very_high=2.0,
        extreme=1e100,
    )
    actual_rho = 0.1
    order = 2
    traces_valid = True
    overflow = False
    syntax = record.atom_syntax
    witness_threshold = 2.0**-14
    if record.point.display_value == _EQUALITY_WITNESS:
        actual_rho = witness_threshold
        certificate = witness_threshold
    elif record.point.display_value == _ADJACENT_WITNESS:
        actual_rho = witness_threshold
        certificate = math.nextafter(witness_threshold, -math.inf)
    elif syntax == "sigma_formation_valid":
        overflow = True
        actual_rho = 1.0
        certificate = 0.9
    elif syntax == "traces_verified":
        traces_valid = False
        certificate = 0.5
    elif syntax == "measured_rho_converges":
        actual_rho = 1.0
        certificate = 0.9
    elif syntax == "rho_covers_input":
        certificate = 0.05
    elif syntax == "0.0 <= rho < 1.0":
        certificate = 1.0
    problem = _trace_fixture(
        record,
        actual_rho=actual_rho,
        certificate=certificate,
        order=order,
        traces_valid=traces_valid,
        overflow=overflow,
    )
    _config(record)
    record.input("sigma_formation", float(problem.lambda_matrix.reshape(-1)[0]))
    record.input("actual_rho", float(actual_rho))
    record.input(
        "trace_evidence",
        problem.exact_power_traces,
        audit_value=problem.exact_power_traces,
    )
    verdicts = _check_premises(record)
    verdict = verdicts[6]
    facts = _independent_facts(problem, record.values["config"])
    details = verdict.details
    production_formation = bool(verdicts[1].details["sigma_formation_valid"])
    production_traces = bool(details["traces_verified"])
    production_measured = (
        bool(details["rho_measurement_valid"]) and float(details["measured_rho"]) < 1.0
    )
    production_covers = bool(details["rho_measurement_valid"]) and float(
        details["measured_rho"]
    ) <= float(details["rho"])
    production_domain = 0.0 <= float(details["rho"]) < 1.0
    oracle_valid = bool(
        facts.sigma_formation_valid
        and facts.traces_verified
        and facts.measured_rho_converges
        and facts.rho_covers_input
        and 0.0 <= certificate < 1.0
    )
    record.returned("rung6_satisfied", bool(verdict.satisfied))
    record.returned("certificate_rho", float(details["rho"]))
    record.returned("measured_rho", float(details["measured_rho"]))
    record.observed_side = _side(bool(verdict.satisfied))
    record.oracle_side = _side(oracle_valid)
    _atom_premises(
        record,
        {
            "sigma_formation_valid": production_formation,
            "traces_verified": production_traces,
            "measured_rho_converges": production_measured,
            "rho_covers_input": production_covers,
            "0.0 <= rho < 1.0": production_domain,
        },
        {
            "sigma_formation_valid": facts.sigma_formation_valid,
            "traces_verified": facts.traces_verified,
            "measured_rho_converges": facts.measured_rho_converges,
            "rho_covers_input": facts.rho_covers_input,
            "0.0 <= rho < 1.0": 0.0 <= certificate < 1.0,
        },
    )
    payload_kwargs = {
        "exact_power_traces": "exact_power_traces",
        "order": "trace_order",
        "rho": "certified_rho",
    }
    if record.atom_id is None:
        _record_payload_attempt(
            record,
            method="eager.truncated_trace_logdet",
            function=eager.truncated_trace_logdet,
            args=("lambda_matrix", "perturbation"),
            kwargs=payload_kwargs,
        )
    elif verdict.satisfied and oracle_valid:
        _record_payload(
            record,
            method="eager.truncated_trace_logdet",
            function=eager.truncated_trace_logdet,
            args=("lambda_matrix", "perturbation"),
            kwargs=payload_kwargs,
        )
    with np.errstate(over="ignore", invalid="ignore"):
        record.input(
            "sigma_input",
            np.asarray(problem.lambda_matrix) + np.asarray(problem.perturbation),
        )
    admitted_value = _capture_direct_product(
        record,
        side="admitted",
        method="eager.truncated_trace_logdet",
        function=eager.truncated_trace_logdet,
        args=("lambda_matrix", "perturbation"),
        kwargs=payload_kwargs,
    )
    refused_value = _capture_direct_product(
        record,
        side="refused",
        method="eager.dense_cholesky_logdet",
        function=eager.dense_cholesky_logdet,
        args=("sigma_input",),
    )
    _record_selected_product(
        record,
        production_admitted=bool(verdict.satisfied),
        oracle_admitted=oracle_valid,
        admitted_method="eager.truncated_trace_logdet",
        admitted_value=admitted_value,
        refused_method="eager.dense_cholesky_logdet",
        refused_value=refused_value,
        oracle_value=(
            _payload_oracle(problem) if facts.sigma_formation_valid else 0.0
        ),
    )
    record.realization(
        point_key="certificate_rho",
        value=float(details["rho"]),
        threshold=(
            witness_threshold
            if record.point.display_value in {_EQUALITY_WITNESS, _ADJACENT_WITNESS}
            else threshold
        ),
        axis_key="certified_rho",
        dtype=np.dtype(float).str,
    )
    record.bind_axes(
        sigma_formation="sigma_formation",
        actual_rho="perturbation",
        trace_order="trace_order",
        trace_evidence="trace_evidence",
        certified_rho="certified_rho",
    )
    return record.finish()


def _rung7_action(record: _FixtureRecord) -> RawObservation:
    threshold = 0
    order = _integer_value(record.point, threshold, very_low=-2, very_high=4, extreme=8)
    actual_rho = 1e-4
    certificate = 0.5
    overflow = False
    width = 2
    dimension = 2
    probe_count = 2
    if record.atom_id is None and record.point.role is PointRole.VERY_LOW:
        probe_count = 1
    elif record.atom_id is None and record.point.role in {
        PointRole.AT,
        PointRole.ABOVE_INTEGER,
    }:
        probe_count = 3
    elif (
        record.atom_id is None
        and record.active_axis == "sigma_formation"
        and record.point.role is PointRole.VERY_HIGH
    ):
        dimension = 257
        width = dimension
        probe_count = dimension
    elif (
        record.atom_id is None
        and record.active_axis == "sigma_formation"
        and record.point.role is PointRole.EXTREME
    ):
        dimension = 10_000
        width = dimension
        probe_count = 1
    syntax = record.atom_syntax
    witness_rho = 2.0**-14
    if record.point.display_value == _EQUALITY_WITNESS:
        actual_rho = witness_rho
        certificate = witness_rho
    elif record.point.display_value == _ADJACENT_WITNESS:
        actual_rho = witness_rho
        certificate = math.nextafter(witness_rho, -math.inf)
    elif syntax == "sigma_formation_valid":
        overflow = True
        actual_rho = 1.0
        certificate = 0.9
        order = 1
    elif syntax == "frozen_width_valid":
        width = 3
        order = 1
    elif syntax == "problem.trace_order is not None":
        order = None
    elif syntax == "problem.trace_order >= 0":
        order = -1
    elif syntax == "measured_rho_converges":
        actual_rho = 1.0
        certificate = 0.9
        order = 1
    elif syntax == "rho_covers_input":
        certificate = 5e-5
        order = 1
    elif syntax == "0.0 <= rho < 1.0":
        certificate = 1.0
        order = 1
    problem = _trace_fixture(
        record,
        actual_rho=actual_rho,
        certificate=certificate,
        order=order,
        overflow=overflow,
        frozen_width=width,
        dimension=dimension,
        probe_count=probe_count,
    )
    _config(record)
    record.input("sigma_formation", float(problem.lambda_matrix.reshape(-1)[0]))
    record.input("actual_rho", float(actual_rho))
    probe_descriptor = (
        None
        if problem.frozen_probes is None
        else _array_descriptor(problem.frozen_probes.values)
    )
    record.input(
        "probe_presence",
        problem.frozen_probes,
        audit_value=probe_descriptor,
    )
    record.input("probe_width", int(width))
    verdicts = _check_premises(record)
    verdict = verdicts[7]
    facts = _independent_facts(problem, record.values["config"])
    details = verdict.details
    production_formation = bool(verdicts[1].details["sigma_formation_valid"])
    production_width = bool(details["probe_width_valid"])
    production_order_present = details["order"] is not None
    production_order_nonnegative = (
        None if details["order"] is None else int(details["order"]) >= 0
    )
    production_measured = (
        bool(details["rho_measurement_valid"]) and float(details["measured_rho"]) < 1.0
    )
    production_covers = bool(details["rho_measurement_valid"]) and float(
        details["measured_rho"]
    ) <= float(details["rho"])
    production_domain = 0.0 <= float(details["rho"]) < 1.0
    oracle_valid = bool(
        facts.sigma_formation_valid
        and facts.frozen_width_valid
        and problem.trace_order is not None
        and problem.trace_order >= 0
        and facts.measured_rho_converges
        and facts.rho_covers_input
        and 0.0 <= certificate < 1.0
    )
    record.returned("rung7_satisfied", bool(verdict.satisfied))
    record.returned("certificate_rho", float(details["rho"]))
    record.returned("measured_rho", float(details["measured_rho"]))
    record.returned("trace_order_evidence", details["order"])
    record.observed_side = _side(bool(verdict.satisfied))
    record.oracle_side = _side(oracle_valid)
    order_reducer = (
        AtomReducer.NOT_EVALUATED
        if production_order_nonnegative is None
        else AtomReducer.SCALAR
    )
    _atom_premises(
        record,
        {
            "sigma_formation_valid": production_formation,
            "frozen_width_valid": production_width,
            "problem.trace_order is not None": production_order_present,
            "problem.trace_order >= 0": production_order_nonnegative,
            "measured_rho_converges": production_measured,
            "rho_covers_input": production_covers,
            "0.0 <= rho < 1.0": production_domain,
        },
        {
            "sigma_formation_valid": facts.sigma_formation_valid,
            "frozen_width_valid": facts.frozen_width_valid,
            "problem.trace_order is not None": problem.trace_order is not None,
            "problem.trace_order >= 0": (
                None if problem.trace_order is None else problem.trace_order >= 0
            ),
            "measured_rho_converges": facts.measured_rho_converges,
            "rho_covers_input": facts.rho_covers_input,
            "0.0 <= rho < 1.0": 0.0 <= certificate < 1.0,
        },
        reducers={"problem.trace_order >= 0": order_reducer},
    )
    payload_kwargs = {"order": "trace_order", "rho": "certified_rho"}
    if record.atom_id is None:
        _record_payload_attempt(
            record,
            method="eager.frozen_hutchinson_trace_logdet",
            function=eager.frozen_hutchinson_trace_logdet,
            args=("lambda_matrix", "perturbation", "frozen_probes"),
            kwargs=payload_kwargs,
        )
    elif verdict.satisfied and oracle_valid:
        _record_payload(
            record,
            method="eager.frozen_hutchinson_trace_logdet",
            function=eager.frozen_hutchinson_trace_logdet,
            args=("lambda_matrix", "perturbation", "frozen_probes"),
            kwargs=payload_kwargs,
        )
    with np.errstate(over="ignore", invalid="ignore"):
        record.input(
            "sigma_input",
            np.asarray(problem.lambda_matrix) + np.asarray(problem.perturbation),
        )
    admitted_value = _capture_direct_product(
        record,
        side="admitted",
        method="eager.frozen_hutchinson_trace_logdet",
        function=eager.frozen_hutchinson_trace_logdet,
        args=("lambda_matrix", "perturbation", "frozen_probes"),
        kwargs=payload_kwargs,
    )
    refused_value = _capture_direct_product(
        record,
        side="refused",
        method="eager.dense_cholesky_logdet",
        function=eager.dense_cholesky_logdet,
        args=("sigma_input",),
    )
    _record_selected_product(
        record,
        production_admitted=bool(verdict.satisfied),
        oracle_admitted=oracle_valid,
        admitted_method="eager.frozen_hutchinson_trace_logdet",
        admitted_value=admitted_value,
        refused_method="eager.dense_cholesky_logdet",
        refused_value=refused_value,
        oracle_value=(
            _payload_oracle(problem) if facts.sigma_formation_valid else 0.0
        ),
    )
    record.realization(
        point_key="trace_order_evidence",
        value=details["order"],
        threshold=threshold,
        axis_key="frozen_probe_values",
    )
    record.bind_axes(
        sigma_formation="sigma_formation",
        trace_order="trace_order",
        probe_presence="probe_presence",
        probe_width="probe_width",
        actual_rho="perturbation",
        certified_rho="certified_rho",
    )
    return record.finish()


_ACTIONS: Mapping[str, Callable[[_FixtureRecord], RawObservation]] = {
    "LADDER:sigma:payload-symmetry": _sigma_payload_action,
    "LADDER:sigma:finite-two-sum": _sigma_finite_action,
    "LADDER:structure:compact-diagonal-positive": _compact_action,
    "LADDER:structure:diagonal-tolerance": _diagonal_action,
    "LADDER:structure:circulant-tolerance-spectrum": _circulant_action,
    "LADDER:structure:toeplitz-tolerance": _toeplitz_action,
    "LADDER:structure:kronecker-evidence": _kronecker_action,
    "LADDER:sigma:symmetry-spd-condition": _shared_sigma_action,
    "LADDER:rank:evidence": _rank_action,
    "LADDER:rho:measurement": _rho_action,
    "LADDER:finite:payload-rho": _finite_rho_action,
    "LADDER:determinant-lemma:payload": _determinant_lemma_action,
    "LADDER:rung0:base": _rung0_action,
    "LADDER:rung1:low-rank-size": _rung1_action,
    "LADDER:rung2:chain": _rung2_action,
    "LADDER:rung3:structured": _rung3_action,
    "LADDER:rung4:dense": _rung4_action,
    "LADDER:rung5:finite-size": _rung5_size_action,
    "LADDER:rung5:finite-executable": _rung5_executable_action,
    "LADDER:rung6:trace": _rung6_action,
    "LADDER:rung7:frozen": _rung7_action,
}

_TOPOLOGIES: Mapping[str, BoundaryTopology] = {
    "LADDER:sigma:payload-symmetry": BoundaryTopology.FLOAT,
    "LADDER:sigma:finite-two-sum": BoundaryTopology.FLOAT,
    "LADDER:structure:compact-diagonal-positive": BoundaryTopology.FLOAT,
    "LADDER:structure:diagonal-tolerance": BoundaryTopology.EXACT,
    "LADDER:structure:circulant-tolerance-spectrum": BoundaryTopology.EXACT,
    "LADDER:structure:toeplitz-tolerance": BoundaryTopology.EXACT,
    "LADDER:structure:kronecker-evidence": BoundaryTopology.EXACT,
    "LADDER:sigma:symmetry-spd-condition": BoundaryTopology.FLOAT,
    "LADDER:rank:evidence": BoundaryTopology.EXACT,
    "LADDER:rho:measurement": BoundaryTopology.FLOAT,
    "LADDER:finite:payload-rho": BoundaryTopology.FLOAT,
    "LADDER:determinant-lemma:payload": BoundaryTopology.EXACT,
    "LADDER:rung0:base": BoundaryTopology.EXACT,
    "LADDER:rung1:low-rank-size": BoundaryTopology.INTEGER,
    "LADDER:rung2:chain": BoundaryTopology.EXACT,
    "LADDER:rung3:structured": BoundaryTopology.EXACT,
    "LADDER:rung4:dense": BoundaryTopology.INTEGER,
    "LADDER:rung5:finite-size": BoundaryTopology.INTEGER,
    "LADDER:rung5:finite-executable": BoundaryTopology.EXACT,
    "LADDER:rung6:trace": BoundaryTopology.FLOAT,
    "LADDER:rung7:frozen": BoundaryTopology.INTEGER,
}

_FAMILIES: Mapping[str, FixtureFamily] = {
    "LADDER:sigma:payload-symmetry": FixtureFamily.SPD_SPECTRUM_CONDITION,
    "LADDER:sigma:finite-two-sum": FixtureFamily.SPD_SPECTRUM_CONDITION,
    "LADDER:structure:compact-diagonal-positive": FixtureFamily.EXACT_STRUCTURE_EVIDENCE,
    "LADDER:structure:diagonal-tolerance": FixtureFamily.EXACT_STRUCTURE_EVIDENCE,
    "LADDER:structure:circulant-tolerance-spectrum": FixtureFamily.EXACT_STRUCTURE_EVIDENCE,
    "LADDER:structure:toeplitz-tolerance": FixtureFamily.EXACT_STRUCTURE_EVIDENCE,
    "LADDER:structure:kronecker-evidence": FixtureFamily.EXACT_STRUCTURE_EVIDENCE,
    "LADDER:sigma:symmetry-spd-condition": FixtureFamily.SPD_SPECTRUM_CONDITION,
    "LADDER:rank:evidence": FixtureFamily.FACTOR_CERTIFICATES,
    "LADDER:rho:measurement": FixtureFamily.RHO_CERTIFICATE_TRACE_ORDER,
    "LADDER:finite:payload-rho": FixtureFamily.RHO_CERTIFICATE_TRACE_ORDER,
    "LADDER:determinant-lemma:payload": FixtureFamily.FACTOR_CERTIFICATES,
    "LADDER:rung0:base": FixtureFamily.LADDER_SIZE_RANK_ROUTING,
    "LADDER:rung1:low-rank-size": FixtureFamily.LADDER_SIZE_RANK_ROUTING,
    "LADDER:rung2:chain": FixtureFamily.LADDER_SIZE_RANK_ROUTING,
    "LADDER:rung3:structured": FixtureFamily.LADDER_SIZE_RANK_ROUTING,
    "LADDER:rung4:dense": FixtureFamily.LADDER_SIZE_RANK_ROUTING,
    "LADDER:rung5:finite-size": FixtureFamily.LADDER_SIZE_RANK_ROUTING,
    "LADDER:rung5:finite-executable": FixtureFamily.LADDER_SIZE_RANK_ROUTING,
    "LADDER:rung6:trace": FixtureFamily.RHO_CERTIFICATE_TRACE_ORDER,
    "LADDER:rung7:frozen": FixtureFamily.RHO_CERTIFICATE_TRACE_ORDER,
}

_EXECUTION_CLASSES: Mapping[str, ExecutionClass] = {
    gate_id: (
        ExecutionClass.TWO_PAYLOAD
        if gate_id == "LADDER:sigma:payload-symmetry"
        else ExecutionClass.PAYLOAD_OR_REFUSAL
    )
    for gate_id in _ACTIONS
}


def _standard_points(gate_id: str) -> tuple[ThresholdPoint, ...]:
    if gate_id == "LADDER:sigma:payload-symmetry":
        return float_grid(
            below=GateSide.REFUSED,
            at=GateSide.REFUSED,
            above=GateSide.ADMITTED,
            very_low=GateSide.ADMITTED,
            very_high=GateSide.ADMITTED,
            extreme=GateSide.ADMITTED,
            threshold="symmetry tolerance",
        )
    if gate_id == "LADDER:sigma:finite-two-sum":
        return float_grid(
            below=GateSide.ADMITTED,
            at=GateSide.ADMITTED,
            above=GateSide.REFUSED,
            very_low=GateSide.ADMITTED,
            very_high=GateSide.REFUSED,
            threshold="max_float / 2",
        )
    if gate_id == "LADDER:structure:compact-diagonal-positive":
        return float_grid(
            below=GateSide.REFUSED,
            at=GateSide.REFUSED,
            above=GateSide.ADMITTED,
            very_low=GateSide.REFUSED,
            very_high=GateSide.ADMITTED,
            threshold="zero",
            include_relative=False,
        )
    if gate_id == "LADDER:sigma:symmetry-spd-condition":
        return float_grid(
            below=GateSide.REFUSED,
            at=GateSide.REFUSED,
            above=GateSide.ADMITTED,
            very_low=GateSide.REFUSED,
            very_high=GateSide.ADMITTED,
            threshold="condition ceiling eigenvalue",
        )
    if gate_id == "LADDER:rho:measurement":
        return float_grid(
            below=GateSide.ADMITTED,
            at=GateSide.ADMITTED,
            above=GateSide.ADMITTED,
            very_low=GateSide.ADMITTED,
            very_high=GateSide.ADMITTED,
            extreme=GateSide.REFUSED,
            threshold="finite rho",
        )
    if gate_id == "LADDER:finite:payload-rho":
        return float_grid(
            below=GateSide.ADMITTED,
            at=GateSide.ADMITTED,
            above=GateSide.REFUSED,
            very_low=GateSide.ADMITTED,
            very_high=GateSide.REFUSED,
            threshold="rho=1",
        )
    if gate_id == "LADDER:rung6:trace":
        return float_grid(
            below=GateSide.ADMITTED,
            at=GateSide.REFUSED,
            above=GateSide.REFUSED,
            very_low=GateSide.REFUSED,
            very_high=GateSide.REFUSED,
            threshold="certificate rho=1",
        )
    if gate_id == "LADDER:rung1:low-rank-size":
        return integer_grid(
            below=GateSide.ADMITTED,
            at=GateSide.ADMITTED,
            above=GateSide.REFUSED,
            very_low=GateSide.ADMITTED,
            very_high=GateSide.REFUSED,
            threshold="rank=2",
        )
    if gate_id in {"LADDER:rung4:dense", "LADDER:rung5:finite-size"}:
        return integer_grid(
            below=GateSide.ADMITTED,
            at=GateSide.ADMITTED,
            above=GateSide.REFUSED,
            very_low=GateSide.ADMITTED,
            very_high=GateSide.REFUSED,
            threshold="dimension=2",
        )
    if gate_id == "LADDER:rung7:frozen":
        return integer_grid(
            below=GateSide.REFUSED,
            at=GateSide.ADMITTED,
            above=GateSide.ADMITTED,
            very_low=GateSide.REFUSED,
            very_high=GateSide.ADMITTED,
            extreme=GateSide.ADMITTED,
            threshold="order=0",
        )
    exact_low_high = {
        "LADDER:structure:kronecker-evidence",
        "LADDER:rank:evidence",
        "LADDER:determinant-lemma:payload",
        "LADDER:rung0:base",
        "LADDER:rung5:finite-executable",
    }
    points = exact_grid(
        very_low=(GateSide.ADMITTED if gate_id in exact_low_high else GateSide.REFUSED),
        very_high=(
            GateSide.ADMITTED if gate_id in exact_low_high else GateSide.REFUSED
        ),
    )
    if gate_id == "LADDER:rank:evidence":
        points = tuple(
            ThresholdPoint(
                point.role,
                point.display_value,
                point.delta,
                (
                    GateSide.ADMITTED
                    if point.role is PointRole.EXTREME
                    else point.expected_side
                ),
            )
            for point in points
        )
    return points


def _mutation_witness_points(gate_id: str) -> tuple[ThresholdPoint, ...]:
    """Return source-reaching equality/adjacent cells reserved for Task 4."""

    roles = {
        "LADDER:rung1:low-rank-size": (PointRole.AT, PointRole.ABOVE_INTEGER),
        "LADDER:rung6:trace": (PointRole.AT, PointRole.BELOW_ULP),
        "LADDER:rung7:frozen": (PointRole.AT, PointRole.ABOVE_INTEGER),
    }.get(gate_id)
    if roles is None:
        return ()
    equality_role, adjacent_role = roles
    return (
        ThresholdPoint(
            equality_role,
            _EQUALITY_WITNESS,
            "exact predicate equality",
            GateSide.ADMITTED,
        ),
        ThresholdPoint(
            adjacent_role,
            _ADJACENT_WITNESS,
            "nearest source-reaching refusal",
            GateSide.REFUSED,
        ),
    )


_BASELINE_OVERRIDES: Mapping[str, Mapping[str, bool]] = {
    "LADDER:sigma:payload-symmetry": {
        "sigma.ndim == 1": False,
        "np.array_equal(sigma, sigma.T)": False,
        "_is_symmetric(sigma, rtol=config.structure_rtol, atol=config.structure_atol)": True,
    },
    "LADDER:structure:diagonal-tolerance": {
        "kind is None": False,
        "_is_diagonal(sigma, rtol=config.structure_rtol, atol=config.structure_atol)": True,
    },
    "LADDER:sigma:symmetry-spd-condition": {
        "sigma.ndim == 1": False,
        "_is_symmetric(sigma, rtol=config.structure_rtol, atol=config.structure_atol)": True,
        "_is_positive_definite(sigma, rtol=config.structure_rtol, atol=config.structure_atol)": True,
    },
    "LADDER:rung1:low-rank-size": {
        "rank_evidence_valid": True,
        "compact_diagonal_payload": True,
        "determinant_lemma_payload": False,
        "sigma_spd": True,
        "rank <= config.low_rank_max": True,
        "rank <= config.low_rank_fraction * n": True,
    },
    "LADDER:rung5:finite-size": {
        "n <= config.finite_max_n": True,
        "(compact_diagonal_payload or determinant_lemma_payload) and rank <= config.finite_max_rank": False,
        "compact_diagonal_payload or determinant_lemma_payload": False,
        "rank <= config.finite_max_rank": True,
    },
    "LADDER:rung5:finite-executable": {
        "finite_size_qualified": True,
        "finite_payload_stable": True,
        "sigma_spd": True,
        "determinant_lemma_payload": False,
        "dense_arithmetic_resolved": True,
    },
}

_DEPENDENCIES: Mapping[
    tuple[str, str],
    tuple[tuple[tuple[str, bool | None], ...], AtomDependencyLogic, str],
] = {
    (
        "LADDER:structure:diagonal-tolerance",
        "_is_diagonal(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
    ): (
        (("kind is None", True),),
        AtomDependencyLogic.TARGET_IMPLIES_PREREQUISITES,
        "The registered auto-detection helper is reached only when kind is absent.",
    ),
    (
        "LADDER:determinant-lemma:payload",
        "problem.low_rank_factors is not None",
    ): (
        (
            ("rank_evidence_valid", None),
            ("sigma_formation_valid", None),
            ("sigma_exactly_symmetric", None),
            ("condition_resolved", None),
        ),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "An absent factor payload short-circuits the remaining determinant-lemma conjuncts.",
    ),
    (
        "LADDER:determinant-lemma:payload",
        "rank_evidence_valid",
    ): (
        (
            ("sigma_formation_valid", None),
            ("sigma_exactly_symmetric", None),
            ("condition_resolved", None),
        ),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "Invalid rank evidence short-circuits the later determinant-lemma conjuncts.",
    ),
    (
        "LADDER:determinant-lemma:payload",
        "sigma_exactly_symmetric",
    ): (
        (("condition_resolved", None),),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "A nonexact Sigma short-circuits the final condition conjunct.",
    ),
    (
        "LADDER:sigma:payload-symmetry",
        "sigma.ndim == 1",
    ): (
        (
            ("np.array_equal(sigma, sigma.T)", None),
            (
                "_is_symmetric(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
                None,
            ),
        ),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "A one-dimensional payload short-circuits both later OR operands.",
    ),
    (
        "LADDER:sigma:payload-symmetry",
        "np.array_equal(sigma, sigma.T)",
    ): (
        (
            (
                "_is_symmetric(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
                None,
            ),
        ),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "Exact transpose equality short-circuits the tolerant helper.",
    ),
    (
        "LADDER:structure:compact-diagonal-positive",
        "np.all(sigma > 0.0)",
    ): (
        (("sigma > 0.0", False),),
        AtomDependencyLogic.ALL_ELEMENTS,
        "The registered parent is the all-elements reduction of the raw comparison.",
    ),
    (
        "LADDER:structure:compact-diagonal-positive",
        "sigma > 0.0",
    ): (
        (("np.all(sigma > 0.0)", False),),
        AtomDependencyLogic.EQUIVALENT,
        "The selected negative element makes both the comparison context and its parent false.",
    ),
    (
        "LADDER:structure:kronecker-evidence",
        "_is_positive_definite(factor)",
    ): (
        (
            ("reconstructed.shape == sigma.shape", None),
            ("np.array_equal(reconstructed, sigma)", None),
        ),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "A non-SPD factor returns before reconstruction comparisons are evaluated.",
    ),
    (
        "LADDER:structure:kronecker-evidence",
        "reconstructed.shape == sigma.shape",
    ): (
        (("np.array_equal(reconstructed, sigma)", None),),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "A shape mismatch short-circuits exact reconstruction equality.",
    ),
    (
        "LADDER:sigma:symmetry-spd-condition",
        "sigma.ndim == 1",
    ): (
        (
            (
                "_is_symmetric(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
                None,
            ),
        ),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "Compact Sigma short-circuits the matrix symmetry helper.",
    ),
    (
        "LADDER:sigma:symmetry-spd-condition",
        "_is_symmetric(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
    ): (
        (
            (
                "_is_positive_definite(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
                None,
            ),
        ),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "The failed symmetry helper short-circuits the positive-definite helper.",
    ),
    (
        "LADDER:determinant-lemma:payload",
        "sigma_formation_valid",
    ): (
        (
            ("sigma_exactly_symmetric", None),
            ("condition_resolved", None),
        ),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "The fault-seam rank certificate exposes the formation operand; its false result short-circuits later conjuncts.",
    ),
    (
        "LADDER:rung0:base",
        "sigma_formation_valid",
    ): (
        (
            ("bool(np.array_equal(sigma, lam))", None),
            ("dense_arithmetic_resolved", None),
        ),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "Non-finite Sigma short-circuits both later base-rung conjuncts.",
    ),
    (
        "LADDER:rung0:base",
        "bool(np.array_equal(sigma, lam))",
    ): (
        (("dense_arithmetic_resolved", None),),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "A nonzero perturbation short-circuits the final dense-arithmetic conjunct.",
    ),
    (
        "LADDER:rung1:low-rank-size",
        "rank_evidence_valid",
    ): (
        (
            ("compact_diagonal_payload", None),
            ("determinant_lemma_payload", None),
            ("sigma_spd", None),
            ("rank <= config.low_rank_max", None),
            ("rank <= config.low_rank_fraction * n", None),
        ),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "Invalid rank evidence short-circuits every later low-rank conjunct.",
    ),
    (
        "LADDER:rung1:low-rank-size",
        "compact_diagonal_payload",
    ): (
        (("determinant_lemma_payload", True),),
        AtomDependencyLogic.TARGET_IMPLIES_PREREQUISITES,
        "The dense witness retains exact factors so the alternative payload is executable.",
    ),
    (
        "LADDER:rung1:low-rank-size",
        "determinant_lemma_payload",
    ): (
        (("compact_diagonal_payload", False),),
        AtomDependencyLogic.PREREQUISITES_IMPLY_TARGET,
        "The positive determinant-lemma witness is represented by a dense factored payload.",
    ),
    (
        "LADDER:rung1:low-rank-size",
        "sigma_spd",
    ): (
        (
            ("determinant_lemma_payload", None),
            ("rank <= config.low_rank_max", None),
            ("rank <= config.low_rank_fraction * n", None),
        ),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "The compact branch skips determinant evidence and a non-SPD Sigma skips both size comparisons.",
    ),
    (
        "LADDER:rung1:low-rank-size",
        "rank <= config.low_rank_max",
    ): (
        (
            ("determinant_lemma_payload", None),
            ("rank <= config.low_rank_fraction * n", None),
        ),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "The compact branch skips determinant evidence and a failed maximum-rank comparison skips the fraction comparison.",
    ),
    (
        "LADDER:rung1:low-rank-size",
        "rank <= config.low_rank_fraction * n",
    ): (
        (("determinant_lemma_payload", None),),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "The compact payload makes the determinant alternative not evaluated.",
    ),
    (
        "LADDER:rung2:chain",
        "chain_structure",
    ): (
        (("sigma_spd", None), ("condition_resolved", None)),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "A non-chain matrix short-circuits both later rung-two conjuncts.",
    ),
    (
        "LADDER:rung2:chain",
        "sigma_spd",
    ): (
        (("condition_resolved", None),),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "A non-SPD chain short-circuits the final condition conjunct.",
    ),
    (
        "LADDER:rung4:dense",
        "n <= config.dense_max_n",
    ): (
        (("condition_resolved", None), ("sigma_spd", None)),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "An over-size matrix short-circuits both dense arithmetic predicates.",
    ),
    (
        "LADDER:rung4:dense",
        "condition_resolved",
    ): (
        (("sigma_spd", None),),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "An unresolved condition short-circuits the final SPD predicate.",
    ),
    (
        "LADDER:rung5:finite-size",
        "n <= config.finite_max_n",
    ): (
        (("rank <= config.finite_max_rank", None),),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "The false compact/determinant child short-circuits the nested rank comparison.",
    ),
    (
        "LADDER:rung5:finite-size",
        "(compact_diagonal_payload or determinant_lemma_payload) and rank <= config.finite_max_rank",
    ): (
        (
            ("n <= config.finite_max_n", False),
            ("compact_diagonal_payload or determinant_lemma_payload", True),
        ),
        AtomDependencyLogic.TARGET_IMPLIES_PREREQUISITES,
        "The outer size branch is false and both nested operands make the parent true.",
    ),
    (
        "LADDER:rung5:finite-size",
        "compact_diagonal_payload or determinant_lemma_payload",
    ): (
        (
            ("n <= config.finite_max_n", False),
            (
                "(compact_diagonal_payload or determinant_lemma_payload) and rank <= config.finite_max_rank",
                True,
            ),
        ),
        AtomDependencyLogic.TARGET_IMPLIES_PREREQUISITES,
        "The exposed compact child flips true, and with a valid rank makes its parent true.",
    ),
    (
        "LADDER:rung5:finite-size",
        "rank <= config.finite_max_rank",
    ): (
        (
            ("n <= config.finite_max_n", False),
            ("compact_diagonal_payload or determinant_lemma_payload", True),
        ),
        AtomDependencyLogic.TARGET_IMPLIES_PREREQUISITES,
        "The over-size compact fixture reaches and fails the nested rank comparison.",
    ),
    (
        "LADDER:rung5:finite-executable",
        "finite_size_qualified",
    ): (
        (
            ("finite_payload_stable", None),
            ("sigma_spd", None),
            ("determinant_lemma_payload", None),
            ("dense_arithmetic_resolved", None),
        ),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "A failed size qualification short-circuits every executable-rung operand.",
    ),
    (
        "LADDER:rung5:finite-executable",
        "finite_payload_stable",
    ): (
        (
            ("sigma_spd", None),
            ("determinant_lemma_payload", None),
            ("dense_arithmetic_resolved", None),
        ),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "An unstable finite payload short-circuits the remaining executable-rung operands.",
    ),
    (
        "LADDER:rung5:finite-executable",
        "sigma_spd",
    ): (
        (
            ("determinant_lemma_payload", None),
            ("dense_arithmetic_resolved", None),
        ),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "A non-SPD Sigma short-circuits both final execution alternatives.",
    ),
    (
        "LADDER:rung5:finite-executable",
        "determinant_lemma_payload",
    ): (
        (("dense_arithmetic_resolved", None),),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "A determinant-lemma payload satisfies the final OR without evaluating dense arithmetic.",
    ),
    (
        "LADDER:rung6:trace",
        "sigma_formation_valid",
    ): (
        (
            ("traces_verified", None),
            ("measured_rho_converges", None),
            ("rho_covers_input", None),
            ("0.0 <= rho < 1.0", None),
        ),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "Overflowed formation short-circuits all later trace-rung operands.",
    ),
    (
        "LADDER:rung6:trace",
        "traces_verified",
    ): (
        (
            ("measured_rho_converges", None),
            ("rho_covers_input", None),
            ("0.0 <= rho < 1.0", None),
        ),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "Unverified traces short-circuit every later rho operand.",
    ),
    (
        "LADDER:rung6:trace",
        "measured_rho_converges",
    ): (
        (("rho_covers_input", None), ("0.0 <= rho < 1.0", None)),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "A nonconvergent measured rho short-circuits coverage and domain checks.",
    ),
    (
        "LADDER:rung6:trace",
        "rho_covers_input",
    ): (
        (("0.0 <= rho < 1.0", None),),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "An understated certificate short-circuits its later domain check.",
    ),
    (
        "LADDER:rung7:frozen",
        "sigma_formation_valid",
    ): (
        (
            ("frozen_width_valid", None),
            ("problem.trace_order is not None", None),
            ("problem.trace_order >= 0", None),
            ("measured_rho_converges", None),
            ("rho_covers_input", None),
            ("0.0 <= rho < 1.0", None),
        ),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "Overflowed formation short-circuits all later frozen-rung operands.",
    ),
    (
        "LADDER:rung7:frozen",
        "frozen_width_valid",
    ): (
        (
            ("problem.trace_order is not None", None),
            ("problem.trace_order >= 0", None),
            ("measured_rho_converges", None),
            ("rho_covers_input", None),
            ("0.0 <= rho < 1.0", None),
        ),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "A wrong probe width short-circuits every later frozen-rung operand.",
    ),
    (
        "LADDER:rung7:frozen",
        "problem.trace_order is not None",
    ): (
        (
            ("problem.trace_order >= 0", None),
            ("measured_rho_converges", None),
            ("rho_covers_input", None),
            ("0.0 <= rho < 1.0", None),
        ),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "An absent order short-circuits its comparison and every later rho operand.",
    ),
    (
        "LADDER:rung7:frozen",
        "problem.trace_order >= 0",
    ): (
        (
            ("measured_rho_converges", None),
            ("rho_covers_input", None),
            ("0.0 <= rho < 1.0", None),
        ),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "A negative order short-circuits every later rho operand.",
    ),
    (
        "LADDER:rung7:frozen",
        "measured_rho_converges",
    ): (
        (("rho_covers_input", None), ("0.0 <= rho < 1.0", None)),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "A nonconvergent measured rho short-circuits coverage and domain checks.",
    ),
    (
        "LADDER:rung7:frozen",
        "rho_covers_input",
    ): (
        (("0.0 <= rho < 1.0", None),),
        AtomDependencyLogic.SHORT_CIRCUIT,
        "An understated certificate short-circuits its later domain check.",
    ),
}

_ATOM_GATE_SIDES: Mapping[tuple[str, str], GateSide] = {
    ("LADDER:sigma:payload-symmetry", "sigma.ndim == 1"): GateSide.ADMITTED,
    (
        "LADDER:sigma:payload-symmetry",
        "np.array_equal(sigma, sigma.T)",
    ): GateSide.ADMITTED,
    (
        "LADDER:sigma:payload-symmetry",
        "_is_symmetric(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
    ): GateSide.ADMITTED,
    ("LADDER:structure:diagonal-tolerance", "kind is None"): GateSide.ADMITTED,
    ("LADDER:sigma:symmetry-spd-condition", "sigma.ndim == 1"): GateSide.ADMITTED,
    ("LADDER:rung1:low-rank-size", "compact_diagonal_payload"): GateSide.ADMITTED,
    ("LADDER:rung1:low-rank-size", "determinant_lemma_payload"): GateSide.ADMITTED,
    (
        "LADDER:rung5:finite-size",
        "(compact_diagonal_payload or determinant_lemma_payload) and rank <= config.finite_max_rank",
    ): GateSide.ADMITTED,
    (
        "LADDER:rung5:finite-size",
        "compact_diagonal_payload or determinant_lemma_payload",
    ): GateSide.ADMITTED,
    ("LADDER:rung5:finite-executable", "determinant_lemma_payload"): GateSide.ADMITTED,
}


def _baseline_outcomes(gate_id: str) -> dict[str, bool]:
    return {
        _ATOM_SYNTAX[atom_id]: _BASELINE_OVERRIDES.get(gate_id, {}).get(
            _ATOM_SYNTAX[atom_id], True
        )
        for atom_id in _ENTRIES[gate_id].conjunction_atom_ids
    }


def _atom_id_for_syntax(gate_id: str, syntax: str) -> str:
    matches = [
        atom_id
        for atom_id in _ENTRIES[gate_id].conjunction_atom_ids
        if _ATOM_SYNTAX[atom_id] == syntax
    ]
    if len(matches) != 1:
        raise AssertionError(f"{gate_id} syntax does not resolve once: {syntax!r}")
    return matches[0]


def _relation(gate_id: str, atom_id: str) -> AtomRelation:
    entry = _ENTRIES[gate_id]
    syntax = _ATOM_SYNTAX[atom_id]
    outcomes = _baseline_outcomes(gate_id)
    baselines = tuple(
        AtomBaseline(candidate, outcomes[_ATOM_SYNTAX[candidate]])
        for candidate in entry.conjunction_atom_ids
    )
    target = not outcomes[syntax]
    dependency = _DEPENDENCIES.get((gate_id, syntax))
    if dependency is None:
        if source_ast_prerequisites(entry, atom_id):
            raise AssertionError(f"{atom_id} needs an explicit AST dependency")
        return AtomRelation(
            kind=AtomRelationKind.INDEPENDENT,
            baselines=baselines,
            target_outcome=target,
        )
    declared, logic, rationale = dependency
    prerequisites = tuple(
        AtomPrerequisite(_atom_id_for_syntax(gate_id, candidate), expected)
        for candidate, expected in declared
    )
    return AtomRelation(
        kind=AtomRelationKind.DEPENDENT,
        baselines=baselines,
        target_outcome=target,
        prerequisites=prerequisites,
        logic=logic,
        rationale=rationale,
    )


def _runner(
    gate_id: str,
    action: Callable[[_FixtureRecord], RawObservation],
    atom_id: str | None = None,
    active_axis: str | None = None,
) -> Callable[[ThresholdPoint], RawObservation]:
    def run(point: ThresholdPoint) -> RawObservation:
        record = _FixtureRecord(gate_id, point, atom_id, active_axis)
        with _recording_production_predicates(record):
            return action(record)

    return run


def _five_position_points(
    points: tuple[ThresholdPoint, ...],
) -> tuple[ThresholdPoint, ...]:
    selected: dict[AxisPosition, ThresholdPoint] = {}
    for point in points:
        selected.setdefault(_axis_position(point.role), point)
    order = (
        AxisPosition.VERY_LOW,
        AxisPosition.ENDPOINT_LOW,
        AxisPosition.ENDPOINT_HIGH,
        AxisPosition.VERY_HIGH,
        AxisPosition.EXTREME,
    )
    if set(selected) != set(order):
        raise AssertionError("a LADDER axis grid did not expose all five positions")
    return tuple(selected[position] for position in order)


_MUTATION_AXES: Mapping[str, str] = {
    "LADDER:rung1:low-rank-size": "low_rank_fraction",
    "LADDER:rung6:trace": "certified_rho",
    "LADDER:rung7:frozen": "certified_rho",
}


def _build_suite(gate_id: str) -> BoundarySuite:
    entry = _ENTRIES[gate_id]
    action = _ACTIONS[gate_id]
    family = _FAMILIES[gate_id]
    execution_class = _EXECUTION_CLASSES[gate_id]
    topology = _TOPOLOGIES[gate_id]
    methods = tuple(sorted(ALLOWED_DIRECT_CALLS[gate_id]))
    premise_oracles = tuple(
        f"source premise: {_ATOM_SYNTAX[atom_id]}"
        for atom_id in entry.conjunction_atom_ids
    )
    independent_oracles = (
        "independent gate predicate",
        _PAYLOAD_ORACLE,
        *_EXTRA_ORACLES.get(gate_id, ()),
        *premise_oracles,
    )
    non_unit_scale = (
        _NON_UNIT
        if entry.fixture_scale_policy is FixtureScalePolicy.NON_UNIT_REQUIRED
        else None
    )
    standard_points = _standard_points(gate_id)
    if len(entry.axes) == 1:
        grid_cases = make_grid_cases(
            entry=entry,
            points=standard_points,
            fixture_family=family,
            execution_class=execution_class,
            topology=topology,
            direct_methods=methods,
            independent_oracles=independent_oracles,
            runner=_runner(gate_id, action, active_axis=entry.axes[0].name),
            non_unit_scale=non_unit_scale,
        )
    else:
        grid_cases = tuple(
            make_case(
                entry=entry,
                point=point,
                fixture_family=family,
                execution_class=execution_class,
                topology=topology,
                direct_methods=methods,
                independent_oracles=independent_oracles,
                runner=_runner(gate_id, action, active_axis=axis.name),
                non_unit_scale=non_unit_scale,
                suffix=f"{axis.name}-{point.role.value}",
                active_axis=axis.name,
            )
            for axis_index, axis in enumerate(entry.axes)
            for point in (
                standard_points
                if axis_index == 0
                else _five_position_points(standard_points)
            )
        )
    witness_axis = _MUTATION_AXES.get(gate_id, entry.axes[0].name)
    witness_cases = tuple(
        make_case(
            entry=entry,
            point=point,
            fixture_family=family,
            execution_class=execution_class,
            topology=topology,
            direct_methods=methods,
            independent_oracles=independent_oracles,
            runner=_runner(gate_id, action, active_axis=witness_axis),
            non_unit_scale=non_unit_scale,
            suffix=("equality-witness" if index == 0 else "adjacent-witness"),
            active_axis=(witness_axis if len(entry.axes) > 1 else None),
        )
        for index, point in enumerate(_mutation_witness_points(gate_id))
    )
    atom_cases = tuple(
        make_atom_case(
            entry=entry,
            atom_id=atom_id,
            relation=_relation(gate_id, atom_id),
            point=ThresholdPoint(
                role=PointRole.EXTREME,
                display_value=f"source atom: {_ATOM_SYNTAX[atom_id]}",
                delta="source-backed atomic witness",
                expected_side=_ATOM_GATE_SIDES.get(
                    (gate_id, _ATOM_SYNTAX[atom_id]), GateSide.REFUSED
                ),
            ),
            fixture_family=family,
            execution_class=execution_class,
            topology=topology,
            direct_methods=methods,
            independent_oracles=independent_oracles,
            runner=_runner(
                gate_id,
                action,
                atom_id,
                active_axis=entry.axes[0].name,
            ),
            non_unit_scale=non_unit_scale,
            active_axis=(entry.axes[0].name if len(entry.axes) > 1 else None),
        )
        for atom_id in entry.conjunction_atom_ids
    )
    cases = (*grid_cases, *witness_cases, *atom_cases)
    atom_case_ids = {
        atom_id: case.case_id
        for atom_id, case in zip(entry.conjunction_atom_ids, atom_cases, strict=True)
    }
    reserved_roles = {
        "LADDER:sigma:payload-symmetry": (
            PointRole.ABOVE_ULP,
            PointRole.BELOW_ULP,
        ),
        "LADDER:structure:compact-diagonal-positive": (
            PointRole.ABOVE_ULP,
            PointRole.AT,
        ),
        "LADDER:rank:evidence": (
            PointRole.EXACT,
            PointRole.ULP_MISMATCH,
        ),
        "LADDER:finite:payload-rho": (
            PointRole.AT,
            PointRole.ABOVE_ULP,
        ),
        "LADDER:rung4:dense": (
            PointRole.AT,
            PointRole.ABOVE_INTEGER,
        ),
        "LADDER:rung5:finite-size": (
            PointRole.AT,
            PointRole.ABOVE_INTEGER,
        ),
    }.get(gate_id)
    if witness_cases:
        admitted, refused = witness_cases
    elif reserved_roles is not None:
        admitted_role, refused_role = reserved_roles
        admitted = next(
            case for case in grid_cases if case.threshold_point.role is admitted_role
        )
        refused = next(
            case for case in grid_cases if case.threshold_point.role is refused_role
        )
    else:
        admitted = next(
            case
            for case in cases
            if case.threshold_point.expected_side is GateSide.ADMITTED
        )
        refused = next(
            case
            for case in cases
            if case.threshold_point.expected_side is GateSide.REFUSED
        )
    omitted = (
        _RELATIVE_ROLES
        if gate_id == "LADDER:structure:compact-diagonal-positive"
        else frozenset()
    )
    return freeze_suite(
        gate_id=gate_id,
        fixture_family=family,
        execution_class=execution_class,
        topology=topology,
        cases=cases,
        atom_case_ids=atom_case_ids,
        tighten_case_id=admitted.case_id,
        loosen_case_id=refused.case_id,
        ambiguities=(),
        omitted_unrepresentable_roles=omitted,
    )


LADDER_SUITES = tuple(_build_suite(gate_id) for gate_id in sorted(_ENTRIES))

if len(LADDER_SUITES) != 21:
    raise AssertionError("standalone LADDER provider must export exactly 21 suites")
if sum(len(suite.atom_case_ids) for suite in LADDER_SUITES) != 57:
    raise AssertionError("standalone LADDER provider must export exactly 57 atom cases")


__all__ = ["LADDER_SUITES"]
