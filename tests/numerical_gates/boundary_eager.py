"""Executable boundary suites for the 20 two-sided eager numerical gates.

The runners in this module deliberately call the production predicate or
payload named by the Task-3 call map.  Registry prose is used only to attach
stable source/atom identities; it never decides a cell's observed side.
"""

from __future__ import annotations

import ast
import copy
import functools
import math
import sys
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path
from types import FrameType
from unittest.mock import patch

import numpy as np

from bayesmith.marginal import _logdet_eager as eager
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
    RealizedAxis,
    RealizedPoint,
    ThresholdPoint,
    axis_position_for_role,
    capability_grid,
    exact_grid,
    float_grid,
    freeze_suite,
    integer_grid,
    make_atom_case,
    make_grid_cases,
    numerical_evaluation,
    oracle_check,
    source_alias_canonical,
)
from tests.numerical_gates.registry import GATE_REGISTRY, isolatable_atom_ids
from tests.numerical_gates.source_manifest import EXPECTED_SOURCE_MANIFEST
from tests.numerical_gates.source_scan import index_source_text

Runner = Callable[[ThresholdPoint], RawObservation]
_ZERO_RELATIVE_ROLES = frozenset(
    {
        PointRole.BELOW_RELATIVE_1E6,
        PointRole.BELOW_RELATIVE_1E12,
        PointRole.ABOVE_RELATIVE_1E12,
        PointRole.ABOVE_RELATIVE_1E6,
    }
)
_ENTRIES = {entry.gate_id: entry for entry in GATE_REGISTRY}
_SYNTAX = {item.candidate_id: item.syntax for item in EXPECTED_SOURCE_MANIFEST}

# Reviewed, gate-local mutation witnesses.  These are deliberately source
# metadata rather than "first admitted/refused" discovery: mutation scoring
# needs the closest executable cell whose real production quantity controls
# the selected source expression.
_WITNESS_ROLES: dict[str, tuple[PointRole, PointRole]] = {
    "EAGER:LadderConfig:integer-threshold-domain": (PointRole.AT, PointRole.BELOW_INTEGER),
    "EAGER:LadderConfig:low-rank-fraction-domain": (PointRole.AT, PointRole.ABOVE_ULP),
    "EAGER:LadderConfig:structure-tolerance-domain": (PointRole.AT, PointRole.BELOW_ULP),
    "EAGER:array-normalization:shape-and-finiteness": (PointRole.EXACT, PointRole.EXTREME),
    "EAGER:LogDetProblem:lambda-spd": (PointRole.ABOVE_ULP, PointRole.AT),
    "EAGER:factor-balance:exact-power-of-two-reversibility": (PointRole.VALID_CAPABILITY, PointRole.INVALID_CAPABILITY),
    "EAGER:factor-reconstruction:layout-exactness": (PointRole.EXACT, PointRole.ULP_MISMATCH),
    "EAGER:symmetry:tolerant-representative": (PointRole.EXACT, PointRole.ULP_MISMATCH),
    "EAGER:dense-condition:strict-dtype-ceiling": (PointRole.BELOW_ULP, PointRole.AT),
    "EAGER:lambda-logdet:subnormal-rescale": (PointRole.BELOW_ULP, PointRole.AT),
    "EAGER:finite:newton-stability-rho": (PointRole.AT, PointRole.ABOVE_ULP),
    "EAGER:state-space:block-chain-exactness": (PointRole.EXACT, PointRole.ULP_MISMATCH),
    "EAGER:state-space:payload-domain": (PointRole.EXACT, PointRole.ULP_MISMATCH),
    "EAGER:structured:exact-shape-and-spectrum": (PointRole.EXACT, PointRole.ULP_MISMATCH),
    "EAGER:spectral-radius:finite-measurement": (PointRole.VALID_CAPABILITY, PointRole.INVALID_CAPABILITY),
    "EAGER:trace:certificate-domain": (PointRole.BELOW_ULP, PointRole.AT),
    "EAGER:trace:certificate-upper-bound": (PointRole.AT, PointRole.BELOW_ULP),
    "EAGER:trace:tail-domain-and-order": (PointRole.BELOW_ULP, PointRole.VERY_LOW),
    "EAGER:trace:exact-power-trace-evidence": (PointRole.EXACT, PointRole.ULP_MISMATCH),
    "EAGER:frozen-probes:identity-width-order": (PointRole.VALID_CAPABILITY, PointRole.INVALID_CAPABILITY),
}
_EAGER_SOURCE = Path(eager.__file__).read_text()
_SOURCE_INDEX = index_source_text(
    _EAGER_SOURCE, "src/bayesmith/marginal/_logdet_eager.py"
)
_ACTIVE_ATOM_RECORDS: ContextVar[
    dict[str, tuple[object, dict[str, object]]] | None
] = (
    ContextVar("active_atom_records", default=None)
)
_ACTIVE_ENTRY: ContextVar[object | None] = ContextVar("active_entry", default=None)
_ACTIVE_ATOM: ContextVar[str | None] = ContextVar("active_atom", default=None)
_ACTIVE_POINT: ContextVar[ThresholdPoint | None] = ContextVar(
    "active_point", default=None
)
_ACTIVE_ORACLE: ContextVar[str] = ContextVar("active_oracle", default="")
_ACTIVE_REALIZED: ContextVar[
    tuple[
        str,
        object,
        object,
        str | None,
        dict[str, object],
        tuple[str, ...],
        tuple[str, ...],
        str,
    ]
    | None
] = ContextVar("active_realized", default=None)
def _point(role: PointRole, side: GateSide, value: str, delta: str) -> ThresholdPoint:
    return ThresholdPoint(role, value, delta, side)


def _role_ordinal(point: ThresholdPoint) -> int:
    """Stable ordinal used only to diversify non-threshold fixture scales."""
    return list(PointRole).index(point.role) + 1


def _atom_ordinal(point: ThresholdPoint) -> int:
    if point.display_value.startswith("isolated atom "):
        return int(point.display_value.rsplit(" ", 1)[-1])
    return 0


def _exact_mismatch(base: float, point: ThresholdPoint) -> float:
    if point.role is PointRole.ULP_MISMATCH:
        return np.nextafter(base, math.inf)
    if point.role is PointRole.SUBNORMAL_MISMATCH:
        return np.nextafter(0.0, 1.0)
    if point.role is PointRole.MATERIAL_MISMATCH:
        return base + 2.0e-6
    return base + (1.0 + _atom_ordinal(point)) * 1.0e-4


def _threshold_for_role(point: ThresholdPoint, value: float) -> float:
    """Recover the threshold paired with an actual scalar fixture value."""
    if point.role in {PointRole.AT, PointRole.EXACT}:
        return value
    if point.role is PointRole.BELOW_ULP:
        return np.nextafter(value, math.inf)
    if point.role is PointRole.ABOVE_ULP:
        return np.nextafter(value, -math.inf)
    if point.role in {PointRole.BELOW_RELATIVE_1E6, PointRole.BELOW_RELATIVE_1E12}:
        delta = 1e-6 if point.role is PointRole.BELOW_RELATIVE_1E6 else 1e-12
        initial = value / (1.0 - delta)
        candidates = (
            np.nextafter(initial, -math.inf),
            initial,
            np.nextafter(initial, math.inf),
        )
        return min(
            candidates,
            key=lambda candidate: abs(abs(value - candidate) / abs(candidate) - delta),
        )
    if point.role in {PointRole.ABOVE_RELATIVE_1E6, PointRole.ABOVE_RELATIVE_1E12}:
        delta = 1e-6 if point.role is PointRole.ABOVE_RELATIVE_1E6 else 1e-12
        initial = value / (1.0 + delta)
        candidates = (
            np.nextafter(initial, -math.inf),
            initial,
            np.nextafter(initial, math.inf),
        )
        return min(
            candidates,
            key=lambda candidate: abs(abs(value - candidate) / abs(candidate) - delta),
        )
    if point.role is PointRole.VERY_LOW:
        return value + max(abs(value), 1.0)
    if point.role is PointRole.VERY_HIGH:
        return value - max(abs(value) / 2.0, 1.0)
    return value


def _strict_upper_fixture_value(point: ThresholdPoint, threshold: float) -> float:
    """Concrete scalar consumed by a strict upper-bound production gate."""
    return {
        PointRole.AT: threshold,
        PointRole.ABOVE_ULP: np.nextafter(threshold, math.inf),
        PointRole.ABOVE_RELATIVE_1E12: threshold * (1.0 + 1e-12),
        PointRole.ABOVE_RELATIVE_1E6: threshold * (1.0 + 1e-6),
        PointRole.VERY_HIGH: threshold * 2.4,
        PointRole.EXTREME: math.inf,
    }.get(point.role, threshold * 2.4)


def _lower_fixture_value(point: ThresholdPoint, threshold: float) -> float:
    """Concrete scalar consumed at/below a strict positive boundary."""
    scale = max(abs(threshold), 1.0)
    return {
        PointRole.VERY_LOW: threshold - 2.4 * scale,
        PointRole.BELOW_RELATIVE_1E6: threshold - 1e-6 * scale,
        PointRole.BELOW_RELATIVE_1E12: threshold - 1e-12 * scale,
        PointRole.BELOW_ULP: np.nextafter(threshold, -math.inf),
        PointRole.AT: threshold,
        PointRole.EXTREME: math.nan,
    }.get(point.role, threshold)


def _capture(
    *,
    input_key: str,
    value: object,
    threshold: object,
    inputs: dict[str, object],
    direct_input_keys: tuple[str, ...],
    direct_return_keys: tuple[str, ...] = (),
    axis_input_key: str | None = None,
    dtype: str | None = "float64",
) -> None:
    """Retain the exact argument/evidence scalar consumed by production."""
    if input_key not in inputs:
        raise AssertionError(f"realized input {input_key!r} was not retained")
    axis_key = direct_input_keys[0] if axis_input_key is None else axis_input_key
    if not direct_input_keys or any(key not in inputs for key in direct_input_keys):
        raise AssertionError("direct input keys must name retained callable arguments")
    if any(key not in inputs for key in direct_return_keys):
        raise AssertionError("direct return keys must name retained callable outputs")
    if axis_key not in direct_input_keys:
        raise AssertionError("a realized axis must name a direct callable input")
    _ACTIVE_REALIZED.set(
        (
            input_key,
            value,
            threshold,
            dtype,
            inputs,
            direct_input_keys,
            direct_return_keys,
            axis_key,
        )
    )


def _axis_position(role: PointRole) -> AxisPosition:
    return axis_position_for_role(role)


_FALSE_NEUTRAL = frozenset(
    {
        "value < 0", "self.structure_rtol < 0.0", "self.structure_atol < 0.0",
        "array.ndim != ndim", "array.ndim not in (1, 2)",
        "smallest_eigenvalue <= 0.0", "maximum < float(np.finfo(lam.dtype).tiny)",
        "block_size < 1", "np.any(piece != 0.0)", "piece != 0.0",
        "reconstructed.shape != dense.shape", "actual_rho > certificate",
        "order < 0", "multiplicity < 1", "tolerance <= 0.0",
        "whole_trace_log_tail_bound(rho, order, multiplicity) > tolerance",
        "len(traces) < order", "type(probes) is not FrozenProbes",
        "vectors.shape[1] != _n(lam)",
    }
)

# Reviewed source-control-flow rows for fixtures whose target necessarily
# changes or skips companion expressions.  Omitted companions retain their
# ordinary admitted-fixture baseline.  This table is declaration metadata;
# observations are captured independently from live production frames below.
_RELATION_COMPANIONS: dict[
    tuple[str, str], tuple[tuple[str, bool | None], ...]
] = {
    ("EAGER:LadderConfig:structure-tolerance-domain", "np.isfinite(self.structure_rtol)"): (("np.isfinite(self.structure_atol)", None), ("self.structure_rtol < 0.0", None), ("self.structure_atol < 0.0", None)),
    ("EAGER:LadderConfig:structure-tolerance-domain", "np.isfinite(self.structure_atol)"): (("self.structure_rtol < 0.0", None), ("self.structure_atol < 0.0", None)),
    ("EAGER:LadderConfig:structure-tolerance-domain", "self.structure_rtol < 0.0"): (("self.structure_atol < 0.0", None),),
    ("EAGER:array-normalization:shape-and-finiteness", "ndim is not None"): (("array.ndim != ndim", None),),
    ("EAGER:array-normalization:shape-and-finiteness", "array.ndim != ndim"): (("array.ndim not in (1, 2)", None), ("np.all(np.isfinite(array))", None), ("np.isfinite(array)", None)),
    ("EAGER:array-normalization:shape-and-finiteness", "array.ndim not in (1, 2)"): (("ndim is not None", False), ("array.ndim != ndim", None), ("np.all(np.isfinite(array))", None), ("np.isfinite(array)", None)),
    ("EAGER:array-normalization:shape-and-finiteness", "np.all(np.isfinite(array))"): (("np.isfinite(array)", False),),
    ("EAGER:array-normalization:shape-and-finiteness", "np.isfinite(array)"): (("np.all(np.isfinite(array))", False),),
    ("EAGER:factor-balance:exact-power-of-two-reversibility", "np.all(np.isfinite(scaled_left))"): (("np.isfinite(scaled_left)", False), ("np.all(np.isfinite(scaled_right))", None), ("np.isfinite(scaled_right)", None), ("np.array_equal(restored_left, left_column)", None), ("np.array_equal(restored_right, right_column)", None)),
    ("EAGER:factor-balance:exact-power-of-two-reversibility", "np.isfinite(scaled_left)"): (("np.all(np.isfinite(scaled_left))", False), ("np.all(np.isfinite(scaled_right))", None), ("np.isfinite(scaled_right)", None), ("np.array_equal(restored_left, left_column)", None), ("np.array_equal(restored_right, right_column)", None)),
    ("EAGER:factor-balance:exact-power-of-two-reversibility", "np.all(np.isfinite(scaled_right))"): (("np.isfinite(scaled_right)", False), ("np.array_equal(restored_left, left_column)", None), ("np.array_equal(restored_right, right_column)", None)),
    ("EAGER:factor-balance:exact-power-of-two-reversibility", "np.isfinite(scaled_right)"): (("np.all(np.isfinite(scaled_right))", False), ("np.array_equal(restored_left, left_column)", None), ("np.array_equal(restored_right, right_column)", None)),
    ("EAGER:factor-balance:exact-power-of-two-reversibility", "np.array_equal(restored_left, left_column)"): (("np.array_equal(restored_right, right_column)", None),),
    ("EAGER:factor-reconstruction:layout-exactness", "np.array_equal(reconstructed, value)"): (("np.array_equal(canonical, value)", False),),
    ("EAGER:factor-projection:whitened-positive-spectrum", "smallest_eigenvalue <= 0.0"): (("np.isfinite(smallest_eigenvalue)", None),),
    ("EAGER:factor-projection:error-budget", "np.isfinite(eta)"): (("eta < 1.0", None), ("np.isfinite(log_error_bound)", None), ("log_error_bound <= ceiling", None)),
    ("EAGER:factor-projection:error-budget", "eta < 1.0"): (("np.isfinite(log_error_bound)", None), ("log_error_bound <= ceiling", None)),
    ("EAGER:factor-projection:error-budget", "np.isfinite(log_error_bound)"): (("log_error_bound <= ceiling", None),),
    ("EAGER:factor-base:condition-ceiling", "np.isfinite(base_condition)"): (("base_condition < base_condition_ceiling", None),),
    ("EAGER:factor-base:error-budget", "np.isfinite(base_solve_eta)"): (("0.0 <= base_solve_eta < 1.0", None), ("np.isfinite(base_log_error_bound)", False), ("base_log_error_bound <= ceiling", None)),
    ("EAGER:factor-base:error-budget", "0.0 <= base_solve_eta < 1.0"): (("np.isfinite(base_log_error_bound)", False), ("base_log_error_bound <= ceiling", None)),
    ("EAGER:factor-base:error-budget", "np.isfinite(base_log_error_bound)"): (("np.isfinite(base_solve_eta)", False), ("0.0 <= base_solve_eta < 1.0", None), ("base_log_error_bound <= ceiling", None)),
    ("EAGER:factor-reduced:diagonal-certificate", "reduced_sign > 0.0"): (("np.all(np.isfinite(relative_diagonal_error))", None), ("np.isfinite(relative_diagonal_error)", None), ("np.all(relative_diagonal_error < 1.0)", None), ("relative_diagonal_error < 1.0", None)),
    ("EAGER:factor-reduced:diagonal-certificate", "np.all(np.isfinite(relative_diagonal_error))"): (("np.isfinite(relative_diagonal_error)", False), ("np.all(relative_diagonal_error < 1.0)", None), ("relative_diagonal_error < 1.0", None)),
    ("EAGER:factor-reduced:diagonal-certificate", "np.isfinite(relative_diagonal_error)"): (("np.all(np.isfinite(relative_diagonal_error))", False), ("np.all(relative_diagonal_error < 1.0)", None), ("relative_diagonal_error < 1.0", None)),
    ("EAGER:factor-reduced:diagonal-certificate", "np.all(relative_diagonal_error < 1.0)"): (("relative_diagonal_error < 1.0", False),),
    ("EAGER:factor-reduced:diagonal-certificate", "relative_diagonal_error < 1.0"): (("np.all(relative_diagonal_error < 1.0)", False),),
    ("EAGER:factor-reduced:qr-certificate", "reduced_sign > 0.0"): (("np.isfinite(reduced_eta)", None), ("0.0 <= reduced_eta < 1.0", None), ("np.isfinite(orthogonality_eta)", None), ("0.0 <= orthogonality_eta < 1.0", None), ("np.all(reduced_r_diagonal != 0.0)", None), ("reduced_r_diagonal != 0.0", None)),
    ("EAGER:factor-reduced:qr-certificate", "np.isfinite(reduced_eta)"): (("0.0 <= reduced_eta < 1.0", None), ("np.isfinite(orthogonality_eta)", None), ("0.0 <= orthogonality_eta < 1.0", None), ("np.all(reduced_r_diagonal != 0.0)", None), ("reduced_r_diagonal != 0.0", None)),
    ("EAGER:factor-reduced:qr-certificate", "0.0 <= reduced_eta < 1.0"): (("np.isfinite(orthogonality_eta)", None), ("0.0 <= orthogonality_eta < 1.0", None), ("np.all(reduced_r_diagonal != 0.0)", None), ("reduced_r_diagonal != 0.0", None)),
    ("EAGER:factor-reduced:qr-certificate", "np.isfinite(orthogonality_eta)"): (("0.0 <= orthogonality_eta < 1.0", None), ("np.all(reduced_r_diagonal != 0.0)", None), ("reduced_r_diagonal != 0.0", None)),
    ("EAGER:factor-reduced:qr-certificate", "0.0 <= orthogonality_eta < 1.0"): (("np.all(reduced_r_diagonal != 0.0)", None), ("reduced_r_diagonal != 0.0", None)),
    ("EAGER:factor-reduced:qr-certificate", "np.all(reduced_r_diagonal != 0.0)"): (("reduced_r_diagonal != 0.0", False),),
    ("EAGER:factor-reduced:qr-certificate", "reduced_r_diagonal != 0.0"): (("np.all(reduced_r_diagonal != 0.0)", False),),
    ("EAGER:factor-reduced:acceptance-budget", "np.isfinite(total_log_error_bound)"): (("total_log_error_bound <= ceiling", None),),
    ("EAGER:symmetry:tolerant-representative", "_is_symmetric(value, rtol=rtol, atol=atol)"): (("np.all(np.isfinite(eigenvalues))", None), ("np.isfinite(eigenvalues)", None), ("np.all(eigenvalues > 0.0)", None), ("eigenvalues > 0.0", None)),
    ("EAGER:symmetry:tolerant-representative", "np.all(np.isfinite(eigenvalues))"): (("np.isfinite(eigenvalues)", False), ("np.all(eigenvalues > 0.0)", None), ("eigenvalues > 0.0", None)),
    ("EAGER:symmetry:tolerant-representative", "np.isfinite(eigenvalues)"): (("np.all(np.isfinite(eigenvalues))", False), ("np.all(eigenvalues > 0.0)", None), ("eigenvalues > 0.0", None)),
    ("EAGER:symmetry:tolerant-representative", "np.all(eigenvalues > 0.0)"): (("eigenvalues > 0.0", False),),
    ("EAGER:symmetry:tolerant-representative", "eigenvalues > 0.0"): (("np.all(eigenvalues > 0.0)", False),),
    ("EAGER:dense-condition:strict-dtype-ceiling", "np.isfinite(condition)"): (("condition < ceiling", None),),
    ("EAGER:state-space:block-chain-exactness", "block_size < 1"): (("np.any(piece != 0.0)", None), ("piece != 0.0", None), ("_is_block_chain(dense, block_size, rtol=rtol, atol=atol)", None)),
    ("EAGER:state-space:block-chain-exactness", "np.any(piece != 0.0)"): (("piece != 0.0", True), ("_is_block_chain(dense, block_size, rtol=rtol, atol=atol)", None)),
    ("EAGER:state-space:block-chain-exactness", "piece != 0.0"): (("np.any(piece != 0.0)", True), ("_is_block_chain(dense, block_size, rtol=rtol, atol=atol)", None)),
    ("EAGER:state-space:block-chain-exactness", "_is_block_chain(dense, block_size, rtol=rtol, atol=atol)"): (("np.any(piece != 0.0)", True), ("piece != 0.0", True)),
    ("EAGER:state-space:payload-domain", "_is_symmetric(dense, rtol=rtol, atol=atol)"): (("_is_positive_definite(dense)", None), ("math.isfinite(total)", None)),
    ("EAGER:state-space:payload-domain", "_is_positive_definite(dense)"): (("math.isfinite(total)", None),),
    ("EAGER:structured:exact-shape-and-spectrum", "reconstructed.shape != dense.shape"): (("np.array_equal(reconstructed, dense)", None),),
    ("EAGER:spectral-radius:finite-measurement", "np.all(np.isfinite(x))"): (("np.isfinite(x)", False), ("np.all(np.isfinite(eigenvalues))", None), ("np.isfinite(eigenvalues)", None)),
    ("EAGER:spectral-radius:finite-measurement", "np.isfinite(x)"): (("np.all(np.isfinite(x))", False), ("np.all(np.isfinite(eigenvalues))", None), ("np.isfinite(eigenvalues)", None)),
    ("EAGER:spectral-radius:finite-measurement", "np.all(np.isfinite(eigenvalues))"): (("np.isfinite(eigenvalues)", False),),
    ("EAGER:spectral-radius:finite-measurement", "np.isfinite(eigenvalues)"): (("np.all(np.isfinite(eigenvalues))", False),),
    ("EAGER:trace:tail-domain-and-order", "0.0 <= rho < 1.0"): (("order < 0", None), ("multiplicity < 1", None), ("np.isfinite(tolerance)", None), ("tolerance <= 0.0", None), ("whole_trace_log_tail_bound(rho, order, multiplicity) > tolerance", None)),
    ("EAGER:trace:tail-domain-and-order", "order < 0"): (("multiplicity < 1", None), ("np.isfinite(tolerance)", None), ("tolerance <= 0.0", None), ("whole_trace_log_tail_bound(rho, order, multiplicity) > tolerance", None)),
    ("EAGER:trace:tail-domain-and-order", "multiplicity < 1"): (("0.0 <= rho < 1.0", None), ("order < 0", None), ("np.isfinite(tolerance)", None), ("tolerance <= 0.0", None), ("whole_trace_log_tail_bound(rho, order, multiplicity) > tolerance", None)),
    ("EAGER:trace:tail-domain-and-order", "np.isfinite(tolerance)"): (("0.0 <= rho < 1.0", None), ("order < 0", None), ("multiplicity < 1", None), ("tolerance <= 0.0", None), ("whole_trace_log_tail_bound(rho, order, multiplicity) > tolerance", None)),
    ("EAGER:trace:tail-domain-and-order", "tolerance <= 0.0"): (("0.0 <= rho < 1.0", None), ("order < 0", None), ("multiplicity < 1", None), ("whole_trace_log_tail_bound(rho, order, multiplicity) > tolerance", None)),
    ("EAGER:trace:exact-power-trace-evidence", "order < 0"): (("len(traces) < order", None), ("np.all(np.isfinite(supplied))", None), ("np.isfinite(supplied)", None), ("np.array_equal(supplied, derived)", None)),
    ("EAGER:trace:exact-power-trace-evidence", "len(traces) < order"): (("np.all(np.isfinite(supplied))", None), ("np.isfinite(supplied)", None), ("np.array_equal(supplied, derived)", None)),
    ("EAGER:trace:exact-power-trace-evidence", "np.all(np.isfinite(supplied))"): (("np.isfinite(supplied)", False), ("np.array_equal(supplied, derived)", None)),
    ("EAGER:trace:exact-power-trace-evidence", "np.isfinite(supplied)"): (("np.all(np.isfinite(supplied))", False), ("np.array_equal(supplied, derived)", None)),
    ("EAGER:frozen-probes:identity-width-order", "type(probes) is not FrozenProbes"): (("vectors.shape[1] != _n(lam)", None), ("order < 0", None)),
    ("EAGER:frozen-probes:identity-width-order", "vectors.shape[1] != _n(lam)"): (("order < 0", None),),
}


def _neutral(atom_id: str) -> bool:
    return _SYNTAX[atom_id] not in _FALSE_NEUTRAL


def _relation(entry: object, atom_id: str) -> AtomRelation:
    canonical = source_alias_canonical(entry, atom_id)
    canonical_ids = tuple(
        item for item in entry.conjunction_atom_ids
        if source_alias_canonical(entry, item) == item
    )
    baselines = tuple(AtomBaseline(item, _neutral(item)) for item in canonical_ids)
    declared = _RELATION_COMPANIONS.get((entry.gate_id, _SYNTAX[canonical]), ())
    prerequisites_list = []
    for companion_syntax, expected_outcome in declared:
        matching = [
            item
            for item in canonical_ids
            if _SYNTAX[item] == companion_syntax
        ]
        if len(matching) != 1:
            raise AssertionError(
                f"{entry.gate_id} relation companion {companion_syntax!r} "
                f"resolved to {matching!r}"
            )
        prerequisites_list.append(
            AtomPrerequisite(matching[0], expected_outcome)
        )
    prerequisites = tuple(prerequisites_list)
    logic = (
        AtomDependencyLogic.SHORT_CIRCUIT
        if any(item.expected_outcome is None for item in prerequisites)
        else AtomDependencyLogic.PREREQUISITES_IMPLY_TARGET
        if prerequisites
        else None
    )
    rationale = (
        "explicit source-control-flow row for this production fixture"
        if prerequisites
        else None
    )
    kind = (
        AtomRelationKind.ALIAS
        if canonical != atom_id
        else AtomRelationKind.DEPENDENT
        if prerequisites
        else AtomRelationKind.INDEPENDENT
    )
    return AtomRelation(
        kind,
        baselines,
        not _neutral(canonical),
        canonical_atom_id=canonical if kind is AtomRelationKind.ALIAS else None,
        prerequisites=prerequisites,
        logic=logic,
        rationale=rationale,
    )


def _evidence_reducer(atom_id: str) -> AtomReducer:
    syntax = _SYNTAX[atom_id]
    arrays = (
        "isfinite(array)", "isfinite(scaled_left)", "isfinite(scaled_right)",
        "isfinite(left_basis)", "isfinite(right_basis)", "isfinite(core)",
        "isfinite(projected)", "isfinite(relative_diagonal_error)",
        "relative_diagonal_error < 1.0", "isfinite(eigenvalues)",
        "eigenvalues > 0.0", "piece != 0.0", "isfinite(x)",
        "isfinite(supplied)", "reduced_r_diagonal != 0.0",
    )
    if not syntax.startswith(("np.all(", "np.any(")) and any(x in syntax for x in arrays):
        return AtomReducer.ANY_ELEMENT if syntax == "piece != 0.0" else AtomReducer.ALL_ELEMENTS
    return AtomReducer.SCALAR


def _atom_node_key(node: ast.AST) -> tuple[int, int, int, int, str]:
    return (
        int(node.lineno),
        int(node.col_offset),
        int(getattr(node, "end_lineno", node.lineno)),
        int(getattr(node, "end_col_offset", node.col_offset)),
        ast.dump(node, include_attributes=False),
    )


_ATOM_IDS_BY_NODE: dict[tuple[int, int, int, int, str], tuple[str, ...]] = {}
for _entry in _ENTRIES.values():
    if not _entry.gate_id.startswith("EAGER:"):
        continue
    for _atom_id in _entry.conjunction_atom_ids:
        _node_key = _atom_node_key(_SOURCE_INDEX[_atom_id][1])
        _ATOM_IDS_BY_NODE[_node_key] = (
            *_ATOM_IDS_BY_NODE.get(_node_key, ()),
            _atom_id,
        )


class _PredicateRecorder(ast.NodeTransformer):
    def generic_visit(self, node: ast.AST) -> ast.AST:
        key = _atom_node_key(node) if isinstance(node, ast.expr) else None
        visited = super().generic_visit(node)
        atom_ids = _ATOM_IDS_BY_NODE.get(key) if key is not None else None
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
                ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]),
            ],
            keywords=[],
        )
        return ast.copy_location(wrapped, node)


def _instrumented_code_objects() -> dict[str, object]:
    tree = ast.parse(_EAGER_SOURCE, filename=eager.__file__)
    result: dict[str, object] = {}

    def visit(body: list[ast.stmt], prefix: tuple[str, ...] = ()) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = int(node.lineno)
                end = int(getattr(node, "end_lineno", start))
                if not any(start <= key[0] <= end for key in _ATOM_IDS_BY_NODE):
                    continue
                copied = copy.deepcopy(node)
                copied.decorator_list = []
                transformed = _PredicateRecorder().visit(copied)
                module = ast.fix_missing_locations(ast.Module(body=[transformed], type_ignores=[]))
                namespace = dict(eager.__dict__)
                namespace["__boundary_record_predicate__"] = _record_predicate
                exec(compile(module, eager.__file__, "exec"), namespace)  # noqa: S102
                result[".".join((*prefix, node.name))] = namespace[node.name].__code__
            elif isinstance(node, ast.ClassDef):
                visit(node.body, (*prefix, node.name))

    visit(tree.body)
    return result


def _record_predicate(
    atom_ids: tuple[str, ...], raw: object, local_values: dict[str, object]
) -> object:
    records = _ACTIVE_ATOM_RECORDS.get()
    if records is not None:
        retained = np.array(raw, copy=True) if isinstance(raw, np.ndarray) else raw
        snapshot = dict(local_values)
        for atom_id in atom_ids:
            existing = records.get(atom_id)
            if existing is None or (
                not bool(np.any(np.asarray(existing[0])))
                and bool(np.any(np.asarray(retained)))
            ):
                records[atom_id] = (retained, snapshot)
    return raw


_INSTRUMENTED_CODES = _instrumented_code_objects()


def _live_function(path: str) -> Callable[..., object]:
    value: object = eager
    for part in path.split("."):
        value = getattr(value, part)
    if not callable(value) or not hasattr(value, "__code__"):
        raise AssertionError(f"instrumentation target {path!r} is not a function")
    return value


@contextmanager
def _trace_direct_calls(names: tuple[str, ...]) -> object:
    """Trace the declared production callables without changing semantics.

    Wrapping the *current* attribute is important: Task-4 may already have
    installed an AST-mutated callable there.  Calling the retained live value
    composes with that mutation instead of restoring an import-time function.
    """
    calls: list[str] = []
    overrides: list[tuple[object, str, object]] = []
    try:
        for name in names:
            owner: object = eager
            parts = name.split(".")
            for part in parts[:-1]:
                owner = getattr(owner, part)
            attribute = parts[-1]
            live = getattr(owner, attribute)

            @functools.wraps(live)
            def traced(
                *args: object,
                __name: str = name,
                __live: Callable[..., object] = live,
                **kwargs: object,
            ) -> object:
                calls.append(__name)
                return __live(*args, **kwargs)

            overrides.append((owner, attribute, live))
            setattr(owner, attribute, traced)
        yield calls
    finally:
        for owner, attribute, live in reversed(overrides):
            setattr(owner, attribute, live)


@contextmanager
def _capture_factor_frame() -> object:
    """Retain live certificate locals without re-evaluating any predicate.

    Line tracing observes the currently installed callable, including an
    outer Task-4 AST mutant.  It never replaces its code or calls a predicate
    a second time.
    """
    retained: dict[str, object] = {}
    previous = sys.gettrace()
    previous_locals: dict[int, Callable[..., object]] = {}
    names = frozenset(
        {
            "left_basis",
            "right_basis",
            "core",
            "projected",
            "residual",
            "smallest_eigenvalue",
            "eta",
            "log_error_bound",
            "base_condition",
            "base_arithmetic_valid",
            "relative_diagonal_error",
            "reduced_eta",
            "orthogonality_eta",
            "reduced_log_error_bound",
            "total_log_error_bound",
            "projection_valid",
            "reduced_arithmetic_valid",
            "total_valid",
        }
    )

    def trace(frame: FrameType, event: str, arg: object) -> object:
        frame_id = id(frame)
        prior_result: object | None = None
        if previous is not None and event == "call":
            prior_result = previous(frame, event, arg)
        elif frame_id in previous_locals:
            prior_result = previous_locals[frame_id](frame, event, arg)
        if event == "return":
            previous_locals.pop(frame_id, None)
        elif callable(prior_result):
            previous_locals[frame_id] = prior_result
        else:
            previous_locals.pop(frame_id, None)

        local_values = frame.f_locals
        if {
            "perturbation",
            "factors",
            "lam",
        } <= local_values.keys():
            for name in names & local_values.keys():
                value = local_values[name]
                retained[name] = (
                    np.array(value, copy=True)
                    if isinstance(value, np.ndarray)
                    else value
                )
        return trace

    sys.settrace(trace)
    try:
        yield retained
    finally:
        sys.settrace(previous)


_ORIGINAL_CODES = {
    path: _live_function(path).__code__ for path in _INSTRUMENTED_CODES
}


@contextmanager
def _recording_production_predicates() -> object:
    missing = object()
    recorder_name = "__boundary_record_predicate__"
    previous_recorder = eager.__dict__.get(recorder_name, missing)
    originals: list[tuple[Callable[..., object], object]] = []
    attribute_overrides: list[tuple[object, str, object]] = []
    try:
        eager.__dict__[recorder_name] = _record_predicate
        for path, code in _INSTRUMENTED_CODES.items():
            function = _live_function(path)
            instrumented_target = function
            seen: set[int] = set()
            while (
                instrumented_target.__code__ is not _ORIGINAL_CODES[path]
                and callable(getattr(instrumented_target, "__wrapped__", None))
                and id(instrumented_target) not in seen
            ):
                seen.add(id(instrumented_target))
                instrumented_target = instrumented_target.__wrapped__
            if instrumented_target.__code__ is not _ORIGINAL_CODES[path]:
                if path != "_condition_certificate":
                    # The outer mutation harness owns this live code object.
                    # Leaving it untouched preserves its wrapped expression
                    # and hit counter; ordinary T-neighbour witnesses for
                    # these gates observe the production return directly.
                    continue
                active_condition = function

                def recorded_condition(
                    value: np.ndarray,
                    *,
                    _active: Callable[..., object] = active_condition,
                ) -> tuple[float, float, bool]:
                    condition, ceiling, resolved = _active(value)
                    local_values = {
                        "value": value,
                        "condition": condition,
                        "ceiling": ceiling,
                    }
                    entry = _ENTRIES[
                        "EAGER:dense-condition:strict-dtype-ceiling"
                    ]
                    for atom_id in entry.conjunction_atom_ids:
                        syntax = _SYNTAX[atom_id]
                        if syntax == "np.isfinite(condition)":
                            raw = np.isfinite(condition)
                        elif syntax == "condition < ceiling":
                            if not np.isfinite(condition):
                                continue
                            raw = bool(resolved)
                        else:
                            continue
                        _record_predicate((atom_id,), raw, local_values)
                    return condition, ceiling, resolved

                attribute_overrides.append((eager, path, function))
                setattr(eager, path, recorded_condition)
                continue
            originals.append((instrumented_target, instrumented_target.__code__))
            instrumented_target.__code__ = code
        yield
    finally:
        for owner, name, value in reversed(attribute_overrides):
            setattr(owner, name, value)
        for function, code in reversed(originals):
            function.__code__ = code
        if previous_recorder is missing:
            eager.__dict__.pop(recorder_name, None)
        else:
            eager.__dict__[recorder_name] = previous_recorder


def _raw_source_atom(atom_id: str) -> tuple[object | None, dict[str, object] | None]:
    records = _ACTIVE_ATOM_RECORDS.get()
    if records is None:
        raise AssertionError("atom production recording was not active")
    return records.get(atom_id, (None, None))


def _oracle_reduce(raw: object, reducer: AtomReducer) -> bool | None:
    if reducer is AtomReducer.NOT_EVALUATED:
        return None
    values = np.asarray(raw)
    if reducer is AtomReducer.SCALAR:
        return bool(values.item())
    flattened = tuple(bool(item) for item in values.flat)
    if reducer is AtomReducer.ALL_ELEMENTS:
        return all(flattened)
    if reducer is AtomReducer.ANY_ELEMENT:
        return any(flattened)
    raise AssertionError(f"unsupported independent reducer {reducer}")


def _independent_atom_truth(
    syntax: str,
    local_values: dict[str, object] | None,
) -> bool | None:
    """Recompute contextual truth from fixture values without source AST eval."""
    if local_values is None:
        return None

    def finite(name: str) -> bool:
        values = np.asarray(local_values[name])
        def scalar_is_finite(item: object) -> bool:
            scalar = np.asarray(item).item()
            if isinstance(scalar, complex):
                return math.isfinite(scalar.real) and math.isfinite(scalar.imag)
            return math.isfinite(float(scalar))

        return all(scalar_is_finite(item) for item in values.flat)

    if " and " in syntax and "np.all(np.isfinite(" in syntax:
        return all(
            finite(name)
            for name in ("left_basis", "right_basis", "core", "projected")
        )
    if syntax.startswith("np.isfinite(self."):
        name = syntax.removeprefix("np.isfinite(self.").removesuffix(")")
        return math.isfinite(float(getattr(local_values["self"], name)))
    finite_names = (
        "structure_rtol", "structure_atol", "array", "scaled_left",
        "scaled_right", "left_basis", "right_basis", "core", "projected",
        "smallest_eigenvalue", "eta", "log_error_bound", "base_condition",
        "base_solve_eta", "base_log_error_bound", "relative_diagonal_error",
        "reduced_eta", "orthogonality_eta", "total_log_error_bound",
        "eigenvalues", "condition", "x", "tolerance", "supplied",
    )
    for name in finite_names:
        if f"isfinite({name})" in syntax:
            return finite(name)
    if syntax == "value < 0":
        return bool(local_values["value"] < 0)
    if syntax == "0.0 <= self.low_rank_fraction <= 1.0":
        value = float(local_values["self"].low_rank_fraction)
        return 0.0 <= value <= 1.0
    if syntax == "self.structure_rtol < 0.0":
        return bool(local_values["self"].structure_rtol < 0.0)
    if syntax == "self.structure_atol < 0.0":
        return bool(local_values["self"].structure_atol < 0.0)
    if syntax == "ndim is not None":
        return local_values["ndim"] is not None
    if syntax == "array.ndim != ndim":
        return np.asarray(local_values["array"]).ndim != local_values["ndim"]
    if syntax == "array.ndim not in (1, 2)":
        return np.asarray(local_values["array"]).ndim not in (1, 2)
    if syntax.startswith("np.array_equal("):
        names = syntax.removeprefix("np.array_equal(").removesuffix(")").split(", ")
        return bool(np.array_equal(local_values[names[0]], local_values[names[1]]))
    if syntax.startswith("np.all(np.isfinite("):
        name = syntax.removeprefix("np.all(np.isfinite(").removesuffix("))")
        return finite(name)
    if syntax.startswith("np.all(") and " < 1.0)" in syntax:
        name = syntax.removeprefix("np.all(").removesuffix(" < 1.0)")
        return all(float(item) < 1.0 for item in np.asarray(local_values[name]).flat)
    if syntax == "np.all(reduced_r_diagonal != 0.0)":
        return all(float(item) != 0.0 for item in np.asarray(local_values["reduced_r_diagonal"]).flat)
    if syntax in {"relative_diagonal_error < 1.0", "eigenvalues > 0.0", "reduced_r_diagonal != 0.0", "piece != 0.0"}:
        name, operator, limit = syntax.split()
        values = np.asarray(local_values[name])
        if operator == "<":
            return all(float(item) < float(limit) for item in values.flat)
        if operator == ">":
            return all(float(item) > float(limit) for item in values.flat)
        return (
            any(float(item) != float(limit) for item in values.flat)
            if name == "piece"
            else all(float(item) != float(limit) for item in values.flat)
        )
    if syntax == "np.any(piece != 0.0)":
        return any(float(item) != 0.0 for item in np.asarray(local_values["piece"]).flat)
    if syntax.startswith("_is_symmetric("):
        name = "value" if "value" in syntax else "dense"
        value = np.asarray(local_values[name])
        return oracles.tolerant_symmetry(
            value,
            relative_tolerance=float(local_values["rtol"]),
            absolute_tolerance=float(local_values["atol"]),
        )
    if syntax.startswith("_is_positive_definite("):
        name = "lam" if "lam" in syntax else "dense"
        value = np.asarray(local_values[name])
        return oracles.symmetric_is_positive_definite(
            value,
            relative_tolerance=float(local_values.get("rtol", 0.0)),
            absolute_tolerance=float(local_values.get("atol", 0.0)),
        )
    if syntax == "smallest_eigenvalue <= 0.0":
        return float(local_values["smallest_eigenvalue"]) <= 0.0
    if syntax in {"eta < 1.0", "base_condition < base_condition_ceiling", "log_error_bound <= ceiling", "base_log_error_bound <= ceiling", "total_log_error_bound <= ceiling", "condition < ceiling", "actual_rho < 1.0", "actual_rho > certificate"}:
        left, operator, right = syntax.split()
        left_value = float(local_values[left])
        right_value = float(local_values[right]) if right in local_values else float(right)
        return left_value < right_value if operator == "<" else left_value <= right_value if operator == "<=" else left_value > right_value
    if syntax in {"0.0 <= base_solve_eta < 1.0", "0.0 <= reduced_eta < 1.0", "0.0 <= orthogonality_eta < 1.0", "0.0 <= rho < 1.0", "0.0 <= certificate < 1.0"}:
        name = syntax.split()[2]
        value = float(local_values[name])
        return 0.0 <= value < 1.0
    if syntax == "reduced_sign > 0.0":
        return float(local_values["reduced_sign"]) > 0.0
    if syntax == "np.all(eigenvalues > 0.0)":
        return all(float(item) > 0.0 for item in np.asarray(local_values["eigenvalues"]).flat)
    if syntax == "maximum < float(np.finfo(lam.dtype).tiny)":
        return float(local_values["maximum"]) < float(np.finfo(np.asarray(local_values["lam"]).dtype).tiny)
    if syntax == "block_size < 1":
        return int(local_values["block_size"]) < 1
    if syntax.startswith("_is_block_chain("):
        dense = np.asarray(local_values["dense"])
        block_size = int(local_values["block_size"])
        if block_size < 1 or dense.shape[0] % block_size:
            return False
        blocks = dense.shape[0] // block_size
        return not any(
            np.any(dense[r * block_size:(r + 1) * block_size, c * block_size:(c + 1) * block_size] != 0.0)
            for r in range(blocks)
            for c in range(blocks)
            if abs(r - c) > 1
        )
    if syntax == "math.isfinite(total)":
        return math.isfinite(float(local_values["total"]))
    if syntax == "reconstructed.shape != dense.shape":
        return np.shape(local_values["reconstructed"]) != np.shape(local_values["dense"])
    if syntax in {"order < 0", "multiplicity < 1", "tolerance <= 0.0", "len(traces) < order"}:
        if syntax.startswith("len"):
            return len(local_values["traces"]) < int(local_values["order"])
        left, operator, right = syntax.split()
        value = float(local_values[left])
        limit = float(right)
        return value < limit if operator == "<" else value <= limit
    if syntax.startswith("whole_trace_log_tail_bound("):
        return oracles.whole_trace_tail(
            float(local_values["rho"]),
            int(local_values["order"]),
            int(local_values["multiplicity"]),
        ) > float(local_values["tolerance"])
    if syntax == "type(probes) is not FrozenProbes":
        return type(local_values["probes"]) is not eager.FrozenProbes
    if syntax == "vectors.shape[1] != _n(lam)":
        lam = np.asarray(local_values["lam"])
        width = lam.shape[0] if lam.ndim == 2 else lam.size
        return np.asarray(local_values["vectors"]).shape[1] != width
    raise AssertionError(f"no independent fixture oracle for source atom {syntax!r}")


def finite_array(values: np.ndarray) -> bool:
    """Pure-Python finiteness reduction used by independent atom oracles."""
    return all(math.isfinite(float(item)) for item in np.asarray(values).flat)


def _active_gate_admission(default: bool, *, true_side: GateSide) -> bool:
    """Return the production-derived gate side without harness introspection."""
    del true_side
    return bool(default)


def _atom_evidence(entry: object, atom_id: str, key: str, oracle: str) -> tuple[tuple[AtomEvidence, ...], tuple]:
    del atom_id
    evidence = []
    checks = []
    for candidate in entry.conjunction_atom_ids:
        raw, local_values = _raw_source_atom(candidate)
        reducer = (
            AtomReducer.NOT_EVALUATED
            if raw is None
            else _evidence_reducer(candidate)
        )
        truth = _oracle_reduce(raw, reducer)
        expected_truth = _independent_atom_truth(_SYNTAX[candidate], local_values)
        evidence.append(AtomEvidence(candidate, raw, truth, reducer, (key,), oracle))
        checks.append(
            oracle_check(
                oracle=oracle,
                actual=truth,
                expected=expected_truth,
            )
        )
    return tuple(evidence), tuple(checks)


def _realization() -> tuple[
    RealizedPoint,
    dict[str, object],
    tuple[str, ...],
    tuple[str, ...],
    str | None,
    tuple,
    tuple[AtomEvidence, ...],
    tuple,
]:
    entry = _ACTIVE_ENTRY.get()
    point = _ACTIVE_POINT.get()
    if entry is None or point is None:
        raise AssertionError("boundary observation was built outside a suite runner")
    captured = _ACTIVE_REALIZED.get()
    if captured is None:
        raise AssertionError("runner did not retain the production input it consumed")
    (
        input_key,
        value,
        threshold,
        dtype,
        inputs,
        direct_input_keys,
        direct_return_keys,
        axis_key,
    ) = captured
    atom = _ACTIVE_ATOM.get()
    axes = tuple(
        RealizedAxis(
            axis.name,
            _axis_position(point.role),
            axis_key,
            inputs[axis_key],
        )
        for axis in entry.axes
    )
    realized = RealizedPoint(
        quantity=entry.quantity,
        input_key=input_key,
        value=value,
        threshold=threshold,
        dtype=dtype,
        axes=axes,
    )
    sibling = ()
    evidence: tuple[AtomEvidence, ...] = ()
    evidence_checks = ()
    if atom is not None:
        evidence, evidence_checks = _atom_evidence(entry, atom, input_key, _ACTIVE_ORACLE.get())
        sibling = (
            oracle_check(
                oracle=_ACTIVE_ORACLE.get(),
                actual=tuple(item.truth for item in evidence),
                expected=tuple(check.expected for check in evidence_checks),
            ),
        )
    return realized, inputs, direct_input_keys, direct_return_keys, atom, sibling, evidence, evidence_checks


def _checked(
    *, method: str | tuple[str, ...], oracle: str, actual: object, expected: object
) -> RawObservation:
    realized, inputs, input_keys, return_keys, atom, siblings, evidence, evidence_checks = _realization()
    return RawObservation(
        observed_side=GateSide.ADMITTED,
        realized_point=realized,
        realized_inputs=inputs,
        direct_input_keys=input_keys,
        direct_return_keys=return_keys,
        direct_calls=(method,) if isinstance(method, str) else method,
        oracle_checks=(
            oracle_check(
                oracle=oracle,
                actual=actual,
                expected=expected,
            ),
            *evidence_checks,
        ),
        isolated_atom=atom,
        atom_evidence=evidence,
        sibling_premises=siblings,
    )


def _refused(
    *,
    method: str | tuple[str, ...],
    oracle: str,
    reason: str,
    actual_admitted: bool,
    oracle_admitted: bool,
) -> RawObservation:
    realized, inputs, input_keys, return_keys, atom, siblings, evidence, evidence_checks = _realization()
    actual_side = bool(actual_admitted)
    expected_side = bool(oracle_admitted)
    return RawObservation(
        observed_side=(
            GateSide.ADMITTED if actual_side else GateSide.REFUSED
        ),
        realized_point=realized,
        realized_inputs=inputs,
        direct_input_keys=input_keys,
        direct_return_keys=return_keys,
        direct_calls=(method,) if isinstance(method, str) else method,
        oracle_checks=(
            oracle_check(
                oracle=oracle,
                actual=actual_side,
                expected=expected_side,
            ),
            *evidence_checks,
        ),
        isolated_atom=atom,
        atom_evidence=evidence,
        sibling_premises=siblings,
        notes=(reason,),
    )


def _failed_direct_call(
    *,
    method: str | tuple[str, ...],
    oracle: str,
    reason: str,
    oracle_admitted: bool,
) -> RawObservation:
    """Retain a real unexpected direct-call exception as structured BAD."""
    (
        realized,
        inputs,
        input_keys,
        return_keys,
        atom,
        siblings,
        evidence,
        evidence_checks,
    ) = _realization()
    return RawObservation(
        observed_side=GateSide.REFUSED,
        realized_point=realized,
        realized_inputs=inputs,
        direct_input_keys=input_keys,
        direct_return_keys=return_keys,
        direct_calls=(method,) if isinstance(method, str) else method,
        oracle_checks=(
            oracle_check(
                oracle=oracle,
                actual=(False, False),
                expected=(bool(oracle_admitted), True),
            ),
            *evidence_checks,
        ),
        isolated_atom=atom,
        atom_evidence=evidence,
        sibling_premises=siblings,
        notes=(f"direct-call exception: {reason}",),
    )


def _evaluated(
    *,
    method: str,
    oracle: str,
    actual: float,
    expected: float,
    observed_side: GateSide = GateSide.ADMITTED,
    gate_actual: bool | None = None,
    gate_expected: bool | None = None,
) -> RawObservation:
    realized, inputs, input_keys, return_keys, atom, siblings, evidence, evidence_checks = _realization()
    gate_check = ()
    if gate_actual is not None:
        actual_side = bool(gate_actual)
        expected_side = bool(gate_expected)
        gate_check = (
            oracle_check(
                oracle=oracle,
                actual=actual_side,
                expected=expected_side,
            ),
        )
    return RawObservation(
        observed_side=observed_side,
        realized_point=realized,
        realized_inputs=inputs,
        direct_input_keys=input_keys,
        direct_return_keys=return_keys,
        direct_calls=(method,),
        oracle_checks=(
            *gate_check,
            *evidence_checks,
        ),
        evaluations=(
            numerical_evaluation(
                method=method,
                oracle=oracle,
                actual=actual,
                oracle_value=expected,
            ),
        ),
        isolated_atom=atom,
        atom_evidence=evidence,
        sibling_premises=siblings,
    )


def _checked_evaluated(
    *,
    methods: tuple[str, ...],
    oracle: str,
    check_actual: object,
    check_expected: object,
    actual: float,
    expected: float,
) -> RawObservation:
    realized, inputs, input_keys, return_keys, atom, siblings, evidence, evidence_checks = _realization()
    return RawObservation(
        observed_side=GateSide.ADMITTED,
        realized_point=realized,
        realized_inputs=inputs,
        direct_input_keys=input_keys,
        direct_return_keys=return_keys,
        direct_calls=methods,
        oracle_checks=(
            oracle_check(
                oracle=oracle,
                actual=check_actual,
                expected=check_expected,
            ),
            *evidence_checks,
        ),
        evaluations=(
            numerical_evaluation(
                method=methods[0],
                oracle=oracle,
                actual=actual,
                oracle_value=expected,
            ),
        ),
        isolated_atom=atom,
        atom_evidence=evidence,
        sibling_premises=siblings,
    )


def _atom_side(entry: object, syntax: str, refuse: Runner | None) -> GateSide:
    """Side produced when the named source premise alone is made false."""
    gate_id = entry.gate_id
    if refuse is None:
        return GateSide.ADMITTED
    if gate_id == "EAGER:array-normalization:shape-and-finiteness" and syntax == "ndim is not None":
        return GateSide.ADMITTED
    if gate_id == "EAGER:factor-reconstruction:layout-exactness" and "canonical" in syntax:
        return GateSide.ADMITTED
    if gate_id == "EAGER:trace:tail-domain-and-order" and "whole_trace_log_tail_bound" in syntax:
        return GateSide.ADMITTED
    return GateSide.REFUSED


def _default_atom_runners(
    entry: object, admit: Runner, refuse: Runner | None
) -> dict[str, tuple[GateSide, Runner]]:
    """Bind every frozen source atom to an executable source-shaped cell."""
    result: dict[str, tuple[GateSide, Runner]] = {}
    for atom_id in entry.conjunction_atom_ids:
        syntax = _SYNTAX[atom_id]
        side = _atom_side(entry, syntax, refuse)

        def run(
            point: ThresholdPoint,
            *,
            gate_id: str = entry.gate_id,
            source_syntax: str = syntax,
            observed_side: GateSide = side,
        ) -> RawObservation:
            semantic = _semantic_atom_observation(
                gate_id, source_syntax, point, observed_side
            )
            if semantic is not None:
                return semantic
            if observed_side is GateSide.ADMITTED:
                return admit(point)
            if refuse is None:
                raise AssertionError(f"{gate_id} atom unexpectedly needs refusal")
            return refuse(point)

        result[atom_id] = (side, run)
    if set(result) != set(entry.conjunction_atom_ids):
        raise AssertionError(f"{entry.gate_id} atom/source manifest drift")
    return result


_TWO_SIDE_EAGER_GATES = frozenset(
    {
        "EAGER:factor-balance:exact-power-of-two-reversibility",
        "EAGER:factor-reconstruction:layout-exactness",
        "EAGER:factor-projection:whitened-positive-spectrum",
        "EAGER:factor-reduced:diagonal-certificate",
        "EAGER:factor-reduced:qr-certificate",
        "EAGER:factor-reduced:acceptance-budget",
        "EAGER:lambda-logdet:subnormal-rescale",
        "EAGER:finite:newton-stability-rho",
        "EAGER:state-space:block-chain-exactness",
        "EAGER:state-space:payload-domain",
        "EAGER:structured:exact-shape-and-spectrum",
        "EAGER:trace:certificate-domain",
        "EAGER:trace:certificate-upper-bound",
        "EAGER:trace:exact-power-trace-evidence",
        "EAGER:frozen-probes:identity-width-order",
    }
)


def _call_product(call: Callable[[], float]) -> tuple[bool, float | str]:
    """Return a structured product value or exception without hiding either."""
    try:
        return True, float(call())
    except (ArithmeticError, TypeError, ValueError, np.linalg.LinAlgError) as error:
        return False, f"{type(error).__name__}: {error}"


def _two_side_payload_evidence(
    gate_id: str,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Execute the specialized payload and dense truth on one captured fixture."""
    if gate_id not in _TWO_SIDE_EAGER_GATES:
        return (), ()
    point = _ACTIVE_POINT.get()
    if point is None:
        raise AssertionError("two-side payload evidence needs an active point")
    _realized, inputs, *_rest = _realization()
    specialized_name: str
    specialized_call: Callable[[], float]
    dense_matrix: np.ndarray

    if gate_id.startswith("EAGER:factor-"):
        left = np.asarray(inputs["factor_left"] if "factor_left" in inputs else inputs["left"])
        right = np.asarray(inputs["factor_right"] if "factor_right" in inputs else inputs["right"])
        factors = eager.LowRankFactors(left, right)
        if "perturbation" in inputs:
            perturbation = np.asarray(inputs["perturbation"])
            lam = np.asarray(inputs.get("lambda_matrix", np.eye(perturbation.shape[0])))
        else:
            lam = np.eye(left.shape[0], dtype=np.result_type(left, right))
            with np.errstate(over="ignore", invalid="ignore"):
                perturbation = left @ right.T
        dense_matrix = lam + perturbation
        specialized_name = "low_rank_logdet"
        specialized_call = lambda: eager.low_rank_logdet(
            lam, perturbation, factors=factors
        )
    elif gate_id == "EAGER:lambda-logdet:subnormal-rescale":
        dense_matrix = np.asarray(inputs["lambda_matrix"])
        specialized_name = "lambda_logdet"
        specialized_call = lambda: eager.lambda_logdet(dense_matrix)
    elif gate_id == "EAGER:finite:newton-stability-rho":
        lam = np.asarray(inputs["lambda_matrix"])
        perturbation = np.asarray(inputs["perturbation"])
        dense_matrix = lam + perturbation
        specialized_name = "finite_perturbation_logdet"
        specialized_call = lambda: eager.finite_perturbation_logdet(
            lam, perturbation
        )
    elif gate_id.startswith("EAGER:state-space:"):
        dense_matrix = np.asarray(inputs["matrix"])
        block_size = int(inputs["block_size"])
        specialized_name = "state_space_logdet"
        specialized_call = lambda: eager.state_space_logdet(
            dense_matrix,
            block_size=block_size,
            rtol=0.0,
            atol=0.0,
        )
    elif gate_id == "EAGER:structured:exact-shape-and-spectrum":
        dense_matrix = np.asarray(inputs["matrix"])
        factors = (
            np.asarray(inputs["structure_first"]),
            np.asarray(inputs["structure_second"]),
        )
        structure = eager.KroneckerStructure(factors)
        specialized_name = "structured_logdet"
        specialized_call = lambda: eager.structured_logdet(
            dense_matrix,
            kind="kronecker",
            structure=structure,
            rtol=0.0,
            atol=0.0,
        )
    elif gate_id.startswith("EAGER:trace:certificate"):
        lam = np.asarray(inputs["lambda_matrix"])
        perturbation = np.asarray(inputs["perturbation"])
        dense_matrix = lam + perturbation
        certificate = float(inputs["certificate"])
        traces = oracles.exact_power_traces(lam, perturbation, 2)
        specialized_name = "truncated_trace_logdet"
        specialized_call = lambda: eager.truncated_trace_logdet(
            lam,
            perturbation,
            exact_power_traces=traces,
            order=2,
            rho=certificate,
        )
    elif gate_id == "EAGER:trace:exact-power-trace-evidence":
        lam = np.asarray(inputs["lambda_matrix"])
        perturbation = np.asarray(inputs["perturbation"])
        dense_matrix = lam + perturbation
        traces = tuple(float(value) for value in inputs["traces"])
        order = int(inputs["order"])
        radius = oracles.spectral_radius(lam, perturbation)
        specialized_name = "truncated_trace_logdet"
        specialized_call = lambda: eager.truncated_trace_logdet(
            lam,
            perturbation,
            exact_power_traces=traces,
            order=order,
            rho=radius,
        )
    elif gate_id == "EAGER:frozen-probes:identity-width-order":
        lam = np.asarray(inputs["lambda_matrix"])
        perturbation = np.asarray(inputs["perturbation"])
        dense_matrix = lam + perturbation
        probes = eager.FrozenProbes(np.asarray(inputs["probe_values"]))
        order = int(inputs["order"])
        radius = float(inputs["rho"])
        specialized_name = "frozen_hutchinson_trace_logdet"
        specialized_call = lambda: eager.frozen_hutchinson_trace_logdet(
            lam,
            perturbation,
            probes,
            order=order,
            rho=radius,
        )
    else:  # pragma: no cover - kept in lockstep with the gate set above
        raise AssertionError(f"missing two-side fixture for {gate_id}")

    specialized_ok, specialized = _call_product(specialized_call)
    dense_ok, dense = _call_product(
        lambda: eager.dense_cholesky_logdet(dense_matrix)
    )
    try:
        if dense_matrix.ndim == 1:
            dense_oracle = oracles.diagonal_logdet(dense_matrix)
        else:
            if not oracles.symmetric_is_positive_definite(
                dense_matrix,
                relative_tolerance=0.0,
                absolute_tolerance=0.0,
            ):
                raise ValueError("dense truth matrix is not symmetric positive")
            dense_oracle = oracles.slogdet_log(dense_matrix)
    except (ArithmeticError, TypeError, ValueError, np.linalg.LinAlgError):
        oracle_dense_ok = False
        dense_oracle = math.nan
    else:
        oracle_dense_ok = True

    checks: list[object] = [
        oracle_check(
            oracle=_ACTIVE_ORACLE.get(),
            actual=dense_ok,
            expected=oracle_dense_ok,
        )
    ]
    if gate_id != "EAGER:lambda-logdet:subnormal-rescale":
        checks.append(
            oracle_check(
                oracle=_ACTIVE_ORACLE.get(),
                actual=specialized_ok,
                expected=point.expected_side is GateSide.ADMITTED,
            )
        )
    else:
        checks.append(
            oracle_check(
                oracle=_ACTIVE_ORACLE.get(),
                actual=specialized_ok,
                expected=oracle_dense_ok,
            )
        )

    evaluations: list[object] = []
    if dense_ok and oracle_dense_ok:
        evaluations.append(
            numerical_evaluation(
                method="dense_cholesky_logdet",
                oracle=_ACTIVE_ORACLE.get(),
                actual=float(dense),
                oracle_value=dense_oracle,
            )
        )
    approximate_trace = gate_id.startswith("EAGER:trace:") or gate_id == (
        "EAGER:frozen-probes:identity-width-order"
    )
    if specialized_ok and dense_ok and approximate_trace:
        order_for_bound = int(inputs.get("order", 2))
        lam_for_bound = np.asarray(inputs["lambda_matrix"])
        perturbation_for_bound = np.asarray(inputs["perturbation"])
        radius_for_bound = oracles.spectral_radius(
            lam_for_bound, perturbation_for_bound
        )
        multiplicity = int(
            lam_for_bound.size
            if lam_for_bound.ndim == 1
            else lam_for_bound.shape[0]
        )
        checks.append(
            oracle_check(
                oracle=_ACTIVE_ORACLE.get(),
                actual=abs(float(specialized) - float(dense))
                <= oracles.whole_trace_tail(
                    radius_for_bound,
                    order_for_bound,
                    multiplicity,
                ),
                expected=True,
            )
        )
    elif specialized_ok and dense_ok:
        evaluations.append(
            numerical_evaluation(
                method=specialized_name,
                oracle=_ACTIVE_ORACLE.get(),
                actual=float(specialized),
                oracle_value=float(dense),
            )
        )
    return tuple(checks), tuple(evaluations)


def _suite(
    gate_id: str,
    *,
    family: FixtureFamily,
    execution_class: ExecutionClass,
    topology: BoundaryTopology,
    direct_methods: tuple[str, ...],
    independent_oracles: tuple[str, ...],
    grid: Runner,
    admit: Runner,
    refuse: Runner | None,
    non_unit_scale: tuple[float, float] | None = (1.3, 2.4),
    grid_points: tuple[ThresholdPoint, ...] | None = None,
    atom_runners: Mapping[str, tuple[GateSide, Runner]] | None = None,
    omitted_unrepresentable_roles: frozenset[PointRole] = frozenset(),
    ambiguities: tuple[str, ...] = (),
) -> BoundarySuite:
    """Build stable witness cells plus one executable cell per source atom."""
    entry = _ENTRIES[gate_id]
    direct_methods = tuple(
        dict.fromkeys(
            (*direct_methods, *sorted(REQUIRED_DIRECT_CALLS[gate_id]))
        )
    )
    if grid_points is not None:
        points = grid_points
    elif topology is BoundaryTopology.FLOAT:
        points = float_grid(
            below=GateSide.ADMITTED,
            at=GateSide.ADMITTED,
            above=GateSide.ADMITTED if refuse is None else GateSide.REFUSED,
            extreme=GateSide.ADMITTED if refuse is None else GateSide.REFUSED,
            threshold=entry.threshold,
        )
    elif topology is BoundaryTopology.INTEGER:
        points = integer_grid(
            below=GateSide.REFUSED if refuse is not None else GateSide.ADMITTED,
            at=GateSide.ADMITTED,
            above=GateSide.ADMITTED,
            very_low=GateSide.REFUSED if refuse is not None else GateSide.ADMITTED,
            extreme=GateSide.REFUSED if refuse is not None else GateSide.ADMITTED,
            threshold=entry.threshold,
        )
    elif topology is BoundaryTopology.EXACT:
        points = exact_grid(
            mismatch=GateSide.REFUSED if refuse is not None else GateSide.ADMITTED,
            extreme=GateSide.REFUSED if refuse is not None else GateSide.ADMITTED,
        )
    else:
        points = capability_grid(
            invalid=GateSide.REFUSED if refuse is not None else GateSide.ADMITTED,
            extreme=GateSide.REFUSED if refuse is not None else GateSide.ADMITTED,
        )

    def grid_runner(point: ThresholdPoint) -> RawObservation:
        entry_token = _ACTIVE_ENTRY.set(entry)
        point_token = _ACTIVE_POINT.set(point)
        atom_token = _ACTIVE_ATOM.set(None)
        oracle_token = _ACTIVE_ORACLE.set(independent_oracles[0])
        realized_token = _ACTIVE_REALIZED.set(None)
        try:
            with _trace_direct_calls(direct_methods) as runtime_calls:
                observation = grid(point)
                product_checks, product_evaluations = _two_side_payload_evidence(
                    gate_id
                )
            return replace(
                observation,
                direct_calls=tuple(dict.fromkeys(runtime_calls)),
                oracle_checks=(*observation.oracle_checks, *product_checks),
                evaluations=(*observation.evaluations, *product_evaluations),
            )
        finally:
            _ACTIVE_REALIZED.reset(realized_token)
            _ACTIVE_ORACLE.reset(oracle_token)
            _ACTIVE_ATOM.reset(atom_token)
            _ACTIVE_POINT.reset(point_token)
            _ACTIVE_ENTRY.reset(entry_token)

    grid_cases = make_grid_cases(
        entry=entry,
        points=points,
        fixture_family=family,
        execution_class=execution_class,
        topology=topology,
        direct_methods=direct_methods,
        independent_oracles=independent_oracles,
        runner=grid_runner,
        non_unit_scale=non_unit_scale,
    )
    resolved_atom_runners = (
        _default_atom_runners(entry, admit, refuse)
        if atom_runners is None
        else dict(atom_runners)
    )
    if set(resolved_atom_runners) != set(entry.conjunction_atom_ids):
        missing = set(entry.conjunction_atom_ids) - set(resolved_atom_runners)
        extra = set(resolved_atom_runners) - set(entry.conjunction_atom_ids)
        raise AssertionError(
            f"{gate_id} atom runners are not fresh: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    def isolated_runner(atom_id: str) -> Runner:
        def run(point: ThresholdPoint) -> RawObservation:
            entry_token = _ACTIVE_ENTRY.set(entry)
            point_token = _ACTIVE_POINT.set(point)
            atom_token = _ACTIVE_ATOM.set(atom_id)
            oracle_token = _ACTIVE_ORACLE.set(independent_oracles[0])
            realized_token = _ACTIVE_REALIZED.set(None)
            records_token = _ACTIVE_ATOM_RECORDS.set({})
            try:
                with (
                    _trace_direct_calls(direct_methods) as runtime_calls,
                    _recording_production_predicates(),
                ):
                    observation = resolved_atom_runners[atom_id][1](point)
                return replace(
                    observation,
                    direct_calls=tuple(dict.fromkeys(runtime_calls)),
                )
            finally:
                _ACTIVE_ATOM_RECORDS.reset(records_token)
                _ACTIVE_REALIZED.reset(realized_token)
                _ACTIVE_ORACLE.reset(oracle_token)
                _ACTIVE_ATOM.reset(atom_token)
                _ACTIVE_POINT.reset(point_token)
                _ACTIVE_ENTRY.reset(entry_token)

        return run

    def atom_point(atom_id: str, _index: int) -> ThresholdPoint:
        canonical = source_alias_canonical(entry, atom_id)
        fixture_index = entry.conjunction_atom_ids.index(canonical)
        return _point(
            PointRole.VALID_CAPABILITY,
            resolved_atom_runners[atom_id][0],
            f"isolated atom {fixture_index}",
            "only the named premise is flipped",
        )

    dynamic_atoms = isolatable_atom_ids(entry)
    atom_cases = tuple(
        make_atom_case(
            entry=entry,
            atom_id=atom_id,
            relation=_relation(entry, atom_id),
            point=atom_point(atom_id, index),
            fixture_family=family,
            execution_class=execution_class,
            topology=topology,
            direct_methods=direct_methods,
            independent_oracles=independent_oracles,
            runner=isolated_runner(atom_id),
            non_unit_scale=non_unit_scale,
        )
        for index, atom_id in enumerate(dynamic_atoms)
    )
    cases = (*grid_cases, *atom_cases)
    tighten_role, loosen_role = _WITNESS_ROLES[gate_id]
    by_role = {case.threshold_point.role: case for case in grid_cases}
    if tighten_role not in by_role or loosen_role not in by_role:
        raise AssertionError(
            f"{gate_id} reviewed witness role absent: "
            f"tighten={tighten_role}, loosen={loosen_role}"
        )
    tighten_case = by_role[tighten_role]
    loosen_case = by_role[loosen_role]
    return freeze_suite(
        gate_id=gate_id,
        fixture_family=family,
        execution_class=execution_class,
        topology=topology,
        cases=cases,
        atom_case_ids={
            atom: case.case_id
            for atom, case in zip(dynamic_atoms, atom_cases, strict=True)
        },
        tighten_case_id=tighten_case.case_id,
        loosen_case_id=loosen_case.case_id,
        ambiguities=(
            *ambiguities,
            *entry.atom_isolation_ambiguities.values(),
        ),
        omitted_unrepresentable_roles=omitted_unrepresentable_roles,
    )


def _config_runner(
    field: str, good: object, bad: object
) -> tuple[Runner, Runner, Runner]:
    method = "LadderConfig.__post_init__"
    oracle = f"plain Python domain for {field}"

    def oracle_accepts(value: object) -> bool:
        if field == "low_rank_max":
            return isinstance(value, int) and value >= 0
        scalar = float(value)
        if field == "low_rank_fraction":
            return math.isfinite(scalar) and 0.0 <= scalar <= 1.0
        return math.isfinite(scalar) and scalar >= 0.0

    def run(value: object) -> RawObservation:
        point = _ACTIVE_POINT.get()
        if point is None:
            raise AssertionError("configuration runner has no threshold point")
        if field == "low_rank_max":
            threshold: object = 0
            dtype = None
        elif field == "low_rank_fraction":
            threshold = 1.0
            dtype = "float64"
        else:
            threshold = 0.0
            dtype = "float64"
        _capture(
            input_key=field,
            value=value,
            threshold=threshold,
            inputs={field: value},
            direct_input_keys=(field,),
            dtype=dtype,
        )
        expected = oracle_accepts(value)
        try:
            result = eager.LadderConfig(**{field: value})
        except (ArithmeticError, ValueError) as error:
            return _refused(method=method, oracle=oracle, reason=str(error), actual_admitted=False, oracle_admitted=expected)
        return _checked(
            method=method,
            oracle=oracle,
            actual=getattr(result, field),
            expected=value,
        )

    def admitted(point: ThresholdPoint) -> RawObservation:
        if field == "low_rank_max":
            values = {
                PointRole.AT: 0,
                PointRole.ABOVE_INTEGER: 1,
                PointRole.VERY_HIGH: 2**31 - 1,
            }
            value = values.get(point.role, int(good))
        elif field == "low_rank_fraction":
            values = {
                PointRole.VERY_LOW: np.nextafter(0.0, 1.0),
                PointRole.BELOW_RELATIVE_1E6: 0.125 - 1.25e-7,
                PointRole.BELOW_RELATIVE_1E12: 0.125 - 1.25e-13,
                PointRole.BELOW_ULP: np.nextafter(0.125, 0.0),
                PointRole.AT: 0.125,
            }
            value = values.get(point.role, good)
        else:
            values = {
                PointRole.AT: 0.0,
                PointRole.ABOVE_ULP: np.nextafter(0.0, math.inf),
                PointRole.ABOVE_RELATIVE_1E12: 1e-312,
                PointRole.ABOVE_RELATIVE_1E6: 1e-306,
                PointRole.VERY_HIGH: np.finfo(float).max,
            }
            value = values.get(point.role, good)
        return run(value)

    def refused(point: ThresholdPoint) -> RawObservation:
        if field == "low_rank_max":
            value = -1 if point.role is not PointRole.EXTREME else -(2**31)
        elif field == "structure_rtol" and point.role is PointRole.BELOW_ULP:
            value = np.nextafter(0.0, -math.inf)
        elif field == "structure_rtol" and point.role is PointRole.BELOW_RELATIVE_1E12:
            value = -1e-312
        elif field == "structure_rtol" and point.role is PointRole.BELOW_RELATIVE_1E6:
            value = -1e-306
        elif point.role is PointRole.EXTREME:
            value = math.nan
        elif point.role is PointRole.VERY_HIGH:
            value = 4.0
        elif point.role is PointRole.ABOVE_ULP:
            value = np.nextafter(1.0, math.inf)
        elif point.role is PointRole.ABOVE_RELATIVE_1E12:
            value = 1.0 + 1e-12
        else:
            value = 1.0 + 1e-6
        if field == "structure_rtol" and point.role not in {
            PointRole.BELOW_ULP,
            PointRole.BELOW_RELATIVE_1E12,
            PointRole.BELOW_RELATIVE_1E6,
        }:
            value = -abs(float(value))
        return run(value if value != bad else bad)

    def grid(point: ThresholdPoint) -> RawObservation:
        if field == "low_rank_max":
            value = {
                PointRole.BELOW_INTEGER: -1,
                PointRole.AT: 0,
                PointRole.ABOVE_INTEGER: 1,
                PointRole.VERY_LOW: -(2**31),
                PointRole.VERY_HIGH: 2**31 - 1,
                PointRole.EXTREME: -(2**63),
            }[point.role]
        elif field == "low_rank_fraction":
            value = {
                PointRole.VERY_LOW: 0.0,
                PointRole.BELOW_RELATIVE_1E6: 1.0 - 1e-6,
                PointRole.BELOW_RELATIVE_1E12: 1.0 - 1e-12,
                PointRole.BELOW_ULP: np.nextafter(1.0, 0.0),
                PointRole.AT: 1.0,
                PointRole.ABOVE_ULP: np.nextafter(1.0, math.inf),
                PointRole.ABOVE_RELATIVE_1E12: 1.0 + 1e-12,
                PointRole.ABOVE_RELATIVE_1E6: 1.0 + 1e-6,
                PointRole.VERY_HIGH: 2.4,
                PointRole.EXTREME: math.nan,
            }[point.role]
        else:
            value = {
                PointRole.VERY_LOW: -1.3,
                PointRole.BELOW_ULP: np.nextafter(0.0, -math.inf),
                PointRole.AT: 0.0,
                PointRole.ABOVE_ULP: np.nextafter(0.0, math.inf),
                PointRole.VERY_HIGH: 2.4,
                PointRole.EXTREME: math.nan,
            }[point.role]
        return run(value)

    return grid, admitted, refused


def _array_runner(
    point: ThresholdPoint, value: object, ndim: int, expected: bool
) -> RawObservation:
    method = "_read_only_array"
    oracle = "shape/finiteness/dtype/immutability policy"
    array = np.asarray(value)
    scalar = float(array.flat[-1]) if array.size else 0.0
    _capture(
        input_key="terminal_entry",
        value=scalar,
        threshold=0.0 if point.role is PointRole.SUBNORMAL_MISMATCH else 2.4,
        inputs={"value": array, "ndim": ndim, "terminal_entry": scalar},
        direct_input_keys=("value", "ndim", "terminal_entry"),
        axis_input_key="terminal_entry",
        dtype="float64",
    )
    try:
        result = eager._read_only_array(value, ndim=ndim)
    except ValueError as error:
        return _refused(method=method, oracle=oracle, reason=str(error), actual_admitted=False, oracle_admitted=expected)
    passed = (
        expected
        and result.ndim == ndim
        and result.flags.c_contiguous
        and not result.flags.writeable
        and np.all(np.isfinite(result))
        and result.dtype in (np.dtype(np.float32), np.dtype(np.float64))
    )
    return _checked(
        method=method,
        oracle=oracle,
        actual=bool(passed),
        expected=True,
    )


def _array_point_runner(point: ThresholdPoint) -> RawObservation:
    base = 2.4
    if point.role is not PointRole.EXTREME:
        scalar = {
            PointRole.VERY_LOW: 1.3,
            PointRole.EXACT: base,
            PointRole.ULP_MISMATCH: np.nextafter(base, math.inf),
            PointRole.SUBNORMAL_MISMATCH: np.nextafter(0.0, math.inf),
            PointRole.MATERIAL_MISMATCH: base + 2.4e-6,
            PointRole.VERY_HIGH: np.finfo(float).max,
        }.get(point.role, base)
        return _array_runner(
            point,
            np.array([[1.3, -0.2], [0.7, scalar]], dtype=np.float64),
            2,
            True,
        )
    # Keep the rank premise true so the non-finite predicate itself executes.
    # The same real production input also carries signed zero, the minimum
    # subnormal, and the largest finite value; EXTREME is not a relabelled
    # ordinary mismatch fixture.
    return _array_runner(
        point,
        np.array(
            [
                [-0.0, np.nextafter(0.0, 1.0)],
                [np.finfo(float).max, math.nan],
            ],
            dtype=np.float64,
        ),
        2,
        False,
    )


def _problem_runner(
    point: ThresholdPoint, lam: np.ndarray, expected: bool
) -> RawObservation:
    method = "LogDetProblem.__init__"
    oracle = "positive principal minors"
    minimum = float(lam[1, 1])
    _capture(
        input_key="minimum_eigenvalue",
        value=minimum,
        threshold=0.0,
        inputs={"lambda_matrix": lam, "minimum_eigenvalue": minimum},
        direct_input_keys=("lambda_matrix", "minimum_eigenvalue"),
        axis_input_key="minimum_eigenvalue",
    )
    try:
        eager.LogDetProblem(lam, np.zeros_like(lam))
    except ValueError as error:
        return _refused(method=method, oracle=oracle, reason=str(error), actual_admitted=False, oracle_admitted=expected)
    determinant = float(lam[0, 0] * lam[1, 1] - lam[0, 1] * lam[1, 0])
    passed = expected and lam[0, 0] > 0.0 and determinant > 0.0
    return _checked(
        method=method,
        oracle=oracle,
        actual=bool(passed),
        expected=True,
    )


def _problem_point_runner(point: ThresholdPoint) -> RawObservation:
    if point.role is PointRole.EXTREME:
        minimum = math.nan
    elif point.role is PointRole.VERY_LOW:
        minimum = -1.3
    elif point.role is PointRole.BELOW_RELATIVE_1E6:
        minimum = -1e-306
    elif point.role is PointRole.BELOW_RELATIVE_1E12:
        minimum = -1e-312
    elif point.role is PointRole.BELOW_ULP:
        minimum = np.nextafter(0.0, -math.inf)
    elif point.role is PointRole.AT:
        minimum = 0.0
    elif point.role is PointRole.ABOVE_ULP:
        minimum = np.nextafter(0.0, math.inf)
    elif point.role is PointRole.ABOVE_RELATIVE_1E12:
        minimum = 1e-312
    elif point.role is PointRole.ABOVE_RELATIVE_1E6:
        minimum = 1e-306
    else:
        minimum = 2.4
    expected = math.isfinite(minimum) and minimum > 0.0
    return _problem_runner(point, np.diag([3.1, minimum]), expected)


def _factor_fixture() -> tuple[np.ndarray, np.ndarray, eager.LowRankFactors]:
    lam = np.diag(np.array([2.0, 3.0, 4.0]))
    left = np.array([[0.2, 0.1], [0.1, 0.3], [0.2, -0.1]])
    factors = eager.LowRankFactors(left, left.copy())
    return lam, left @ left.T, factors


def _certificate_fixture_for_gate(
    gate_id: str,
) -> tuple[np.ndarray, np.ndarray, eager.LowRankFactors]:
    """Exercise the production branch that owns the reviewed certificate gate."""
    if gate_id == "EAGER:factor-projection:error-budget":
        lam = np.diag(np.array([2.0]))
        left = np.array([[0.2]])
    elif gate_id == "EAGER:factor-reduced:diagonal-certificate":
        lam = np.diag(np.array([2.0, 3.0, 4.0]))
        left = np.array([[0.2], [0.1], [0.2]])
    elif gate_id in {
        "EAGER:factor-base:condition-ceiling",
        "EAGER:factor-base:error-budget",
    }:
        lam = np.array([[2.0, 0.2], [0.2, 3.0]])
        left = np.array([[0.2], [0.1]])
    else:
        lam = np.diag(np.array([2.0, 3.0, 4.0]))
        left = np.array([[0.2, 0.1], [0.1, 0.3], [0.2, -0.1]])
    factors = eager.LowRankFactors(left, left.copy())
    return lam, left @ left.T, factors


def _balance_oracle_accepts(left: np.ndarray, right: np.ndarray) -> bool:
    """Independently apply and invert the documented power-of-two gauge."""
    for column in range(left.shape[1]):
        left_column = left[:, column]
        right_column = right[:, column]
        left_maximum = float(np.max(np.abs(left_column), initial=0.0))
        right_maximum = float(np.max(np.abs(right_column), initial=0.0))
        if left_maximum == 0.0 or right_maximum == 0.0:
            continue
        shift = (
            math.frexp(right_maximum)[1] - math.frexp(left_maximum)[1]
        ) // 2
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scaled_left = np.ldexp(left_column, shift)
            scaled_right = np.ldexp(right_column, -shift)
            restored_left = np.ldexp(scaled_left, -shift)
            restored_right = np.ldexp(scaled_right, shift)
        if not (
            np.all(np.isfinite(scaled_left))
            and np.all(np.isfinite(scaled_right))
            and np.array_equal(restored_left, left_column)
            and np.array_equal(restored_right, right_column)
        ):
            return False
    return True


def _balance_runner(point: ThresholdPoint) -> RawObservation:
    method = "_balanced_factor_columns"
    oracle = "hex-level reversible power-of-two gauge"
    shift = -8 if point.role is PointRole.CAPABILITY_LOW else 2
    base_column = np.array([[1.3], [-0.7]])
    left = np.ldexp(base_column, shift)
    right = np.ldexp(base_column, -shift)
    valid_fixture = point.role in {
        PointRole.CAPABILITY_LOW,
        PointRole.VALID_CAPABILITY,
    }
    if not valid_fixture:
        companion = np.nextafter(0.0, 1.0) * max(1, _atom_ordinal(point) + 1)
        if point.role in {PointRole.CAPABILITY_HIGH, PointRole.EXTREME}:
            companion *= _role_ordinal(point)
        left = np.array([[np.finfo(float).max], [companion]])
        right = np.array([[np.nextafter(0.0, 1.0)], [0.0]])
    _capture(
        input_key="left",
        value=left,
        threshold="exactly reversible",
        inputs={"left": left, "right": right},
        direct_input_keys=("left", "right"),
        axis_input_key="left",
        dtype=None,
    )
    try:
        balanced_left, balanced_right = eager._balanced_factor_columns(left, right)
    except ValueError as error:
        oracle_admitted = _balance_oracle_accepts(left, right)
        return _refused(
            method=method,
            oracle=oracle,
            reason=str(error),
            actual_admitted=False,
            oracle_admitted=oracle_admitted,
        )
    restored_left = np.empty_like(left)
    restored_right = np.empty_like(right)
    for column in range(left.shape[1]):
        left_maximum = float(np.max(np.abs(left[:, column]), initial=0.0))
        right_maximum = float(np.max(np.abs(right[:, column]), initial=0.0))
        if left_maximum == 0.0 or right_maximum == 0.0:
            restored_left[:, column] = 0.0
            restored_right[:, column] = 0.0
            continue
        shift = (
            math.frexp(right_maximum)[1] - math.frexp(left_maximum)[1]
        ) // 2
        restored_left[:, column] = np.ldexp(balanced_left[:, column], -shift)
        restored_right[:, column] = np.ldexp(balanced_right[:, column], shift)
    exact = bool(
        np.array_equal(restored_left, left)
        and np.array_equal(restored_right, right)
    )
    return _checked(
        method=method,
        oracle=oracle,
        actual=exact,
        expected=True,
    )


def _reconstruction_runner(point: ThresholdPoint) -> RawObservation:
    method = "_matching_factor_reconstruction"
    oracle = "explicit exact factor reconstruction"
    _, perturbation, factors = _factor_fixture()
    threshold = float(perturbation[0, 0])
    exact_fixture = point.role in {
        PointRole.EXACT,
        PointRole.VERY_LOW,
        PointRole.VERY_HIGH,
    }
    if not exact_fixture:
        perturbation = perturbation.copy()
        perturbation[0, 0] = _exact_mismatch(perturbation[0, 0], point)
    elif point.role is PointRole.VERY_LOW:
        scaled = factors.left * 0.5
        factors = eager.LowRankFactors(scaled, scaled.copy())
        perturbation = scaled @ scaled.T
    elif point.role is PointRole.VERY_HIGH:
        scaled = factors.left * 2.0
        factors = eager.LowRankFactors(scaled, scaled.copy())
        perturbation = scaled @ scaled.T
    value = float(perturbation[0, 0])
    if point.role is PointRole.EXACT:
        threshold = value
    elif point.role is PointRole.SUBNORMAL_MISMATCH:
        threshold = 0.0
    _capture(
        input_key="authoritative_entry",
        value=value,
        threshold=threshold,
        inputs={
            "perturbation": perturbation,
            "factor_left": factors.left,
            "factor_right": factors.right,
            "authoritative_entry": value,
        },
        direct_input_keys=(
            "perturbation",
            "factor_left",
            "factor_right",
            "authoritative_entry",
        ),
        axis_input_key="authoritative_entry",
    )
    matches, reconstructed = eager._matching_factor_reconstruction(perturbation, factors)
    scalar = np.array([[sum(float(factors.left[i, k]) * float(factors.right[j, k]) for k in range(factors.rank_bound)) for j in range(3)] for i in range(3)])
    expected = np.array_equal(perturbation, reconstructed)
    if matches:
        return _checked(
            method=method,
            oracle=oracle,
            actual=bool(
                expected
                and np.allclose(reconstructed, scalar, rtol=0.0, atol=2e-17)
            ),
            expected=True,
        )
    return _refused(
        method=method,
        oracle=oracle,
        reason="authoritative perturbation differs from every exact layout product",
        actual_admitted=matches,
        oracle_admitted=expected,
    )


def _certificate_runner(point: ThresholdPoint) -> RawObservation:
    method = "_factor_projection_certificate"
    entry = _ACTIVE_ENTRY.get()
    if entry is None:
        raise AssertionError("certificate runner has no active gate")
    gate_id = entry.gate_id
    active_atom = _ACTIVE_ATOM.get()
    atom_syntax = _SYNTAX[active_atom] if active_atom is not None else ""
    oracle = {
        "EAGER:factor-projection:whitened-positive-spectrum": "strictly positive independently supplied whitened spectrum",
        "EAGER:factor-projection:error-budget": "closed projection log-error ceiling",
        "EAGER:factor-base:condition-ceiling": "strict dense base condition ceiling",
        "EAGER:factor-base:error-budget": "closed base log-error ceiling",
        "EAGER:factor-reduced:diagonal-certificate": "strict componentwise diagonal formation radius",
        "EAGER:factor-reduced:qr-certificate": "strict reduced QR reconstruction radius",
        "EAGER:factor-reduced:acceptance-budget": "closed aggregate log-error ceiling",
    }[gate_id]
    def upper_target(threshold: float) -> float:
        return {
            PointRole.VERY_LOW: threshold * 0.25,
            PointRole.BELOW_RELATIVE_1E6: threshold * (1.0 - 1e-6),
            PointRole.BELOW_RELATIVE_1E12: threshold * (1.0 - 1e-12),
            PointRole.BELOW_ULP: np.nextafter(threshold, -math.inf),
            PointRole.AT: threshold,
            PointRole.ABOVE_ULP: np.nextafter(threshold, math.inf),
            PointRole.ABOVE_RELATIVE_1E12: threshold * (1.0 + 1e-12),
            PointRole.ABOVE_RELATIVE_1E6: threshold * (1.0 + 1e-6),
            PointRole.VERY_HIGH: threshold * 2.4,
            PointRole.EXTREME: math.inf,
            PointRole.VALID_CAPABILITY: threshold * 0.25,
        }[point.role]

    threshold: object
    parameter: float
    if gate_id == "EAGER:factor-projection:whitened-positive-spectrum":
        threshold = 0.0
        if point.role is PointRole.EXTREME:
            parameter = float(np.finfo(float).max)
            perturbation = np.array([[0.0, parameter], [parameter, 0.0]])
        else:
            parameter = {
                PointRole.VERY_LOW: -2.0,
                PointRole.BELOW_RELATIVE_1E6: -np.nextafter(1.0, math.inf),
                PointRole.BELOW_RELATIVE_1E12: -np.nextafter(1.0, math.inf),
                PointRole.BELOW_ULP: -np.nextafter(1.0, math.inf),
                PointRole.REACHABLE_BELOW: -np.nextafter(1.0, math.inf),
                PointRole.AT: -1.0,
                PointRole.ABOVE_ULP: -np.nextafter(1.0, 0.0),
                PointRole.REACHABLE_ABOVE: -np.nextafter(1.0, 0.0),
                PointRole.ABOVE_RELATIVE_1E12: -np.nextafter(1.0, 0.0),
                PointRole.ABOVE_RELATIVE_1E6: -np.nextafter(1.0, 0.0),
                PointRole.VERY_HIGH: -0.5,
                PointRole.VALID_CAPABILITY: 0.0,
            }[point.role]
            perturbation = np.diag([parameter, 0.0])
        lam = np.eye(2)
        left = np.eye(2)
        right = perturbation.T.copy()
        metric_key = "smallest_whitened_eigenvalue"
    elif gate_id in {
        "EAGER:factor-projection:error-budget",
        "EAGER:factor-reduced:acceptance-budget",
    }:
        threshold = math.sqrt(float(np.finfo(float).eps))
        if point.role is PointRole.REACHABLE_BELOW:
            desired = float(threshold)
        elif point.role is PointRole.REACHABLE_ABOVE:
            desired = float(threshold) * (1.0 + 1.0e-12)
        else:
            desired = upper_target(float(threshold))
        parameter = float(np.finfo(float).max) if not math.isfinite(desired) else math.expm1(desired)
        if gate_id == "EAGER:factor-reduced:acceptance-budget":
            if point.role is PointRole.REACHABLE_BELOW:
                parameter = float.fromhex("0x1.ffffff4000000p-27")
            elif point.role is PointRole.REACHABLE_ABOVE:
                parameter = float.fromhex("0x1.ffffff4000001p-27")
        lam = np.array([[1.0]])
        left = np.empty((1, 0))
        right = left.copy()
        perturbation = np.array([[parameter]])
        metric_key = (
            "projection_log_error_bound"
            if gate_id == "EAGER:factor-projection:error-budget"
            else "total_log_error_bound"
        )
    elif gate_id == "EAGER:factor-base:condition-ceiling":
        threshold = 1.0 / math.sqrt(float(np.finfo(float).eps))
        if "isfinite(base_condition)" in atom_syntax:
            parameter = float(np.nextafter(0.0, 1.0))
            lam = np.diag([parameter, 1.0])
        else:
            parameter = upper_target(float(threshold))
            lam = np.array(
                [
                    [1.0, np.nextafter(0.0, 1.0)],
                    [np.nextafter(0.0, 1.0), parameter],
                ]
            )
        left = np.empty((2, 0))
        right = left.copy()
        perturbation = np.zeros((2, 2))
        metric_key = "base_condition"
    elif gate_id == "EAGER:factor-base:error-budget":
        threshold = math.sqrt(float(np.finfo(np.float32).eps))
        if point.role is PointRole.REACHABLE_BELOW:
            desired = float(threshold)
            parameter = float(np.float32(float.fromhex("0x1.dca7000000000p+122")))
        elif point.role is PointRole.REACHABLE_ABOVE:
            desired = float(threshold)
            parameter = float(np.float32(float.fromhex("0x1.dca7ee0000000p+122")))
        else:
            desired = upper_target(float(threshold))
            exponent = desired / (2.0 * float(np.finfo(np.float32).eps) * 17.0)
            parameter = float(np.finfo(np.float32).max) if not math.isfinite(exponent) else float(np.float32(math.exp(min(exponent, 88.0))))
        lam = np.diag(np.full(17, np.float32(parameter), dtype=np.float32))
        left = np.empty((17, 0), dtype=np.float32)
        right = left.copy()
        perturbation = np.zeros((17, 17), dtype=np.float32)
        metric_key = "base_log_error_bound"
    elif gate_id == "EAGER:factor-reduced:diagonal-certificate":
        threshold = 1.0
        below = 1.0 - 2.0**-50
        parameter = 1.5 if atom_syntax == "reduced_sign > 0.0" else {
            PointRole.VERY_LOW: 0.5,
            PointRole.BELOW_RELATIVE_1E6: below,
            PointRole.BELOW_RELATIVE_1E12: below,
            PointRole.BELOW_ULP: below,
            PointRole.REACHABLE_BELOW: below,
            PointRole.AT: np.nextafter(below, math.inf),
            PointRole.ABOVE_ULP: np.nextafter(below, math.inf),
            PointRole.REACHABLE_ABOVE: np.nextafter(below, math.inf),
            PointRole.ABOVE_RELATIVE_1E12: np.nextafter(below, math.inf),
            PointRole.ABOVE_RELATIVE_1E6: np.nextafter(below, math.inf),
            PointRole.VERY_HIGH: 1.5,
            PointRole.EXTREME: 1.0,
            PointRole.VALID_CAPABILITY: 0.5,
        }[point.role]
        lam = np.eye(2)
        left = np.array([[1.0], [0.0]])
        right = np.array([[-parameter], [0.0]])
        perturbation = left @ right.T
        metric_key = "reduced_eta"
    else:
        threshold = 1.0
        below = float.fromhex("0x1.ffffff4ed2e97p-1")
        if atom_syntax == "reduced_sign > 0.0":
            parameter = 1.5
        elif "isfinite(reduced_eta)" in atom_syntax:
            parameter = np.nextafter(1.0, 0.0)
        else:
            parameter = {
                PointRole.VERY_LOW: 0.5,
                PointRole.BELOW_RELATIVE_1E6: below,
                PointRole.BELOW_RELATIVE_1E12: below,
                PointRole.BELOW_ULP: below,
                PointRole.REACHABLE_BELOW: below,
                PointRole.VERY_HIGH: 1.5,
                PointRole.EXTREME: 1.0,
            }.get(point.role, np.nextafter(below, math.inf))
        lam = np.eye(2)
        left = np.eye(2)
        right = (
            np.array([[-1.5, 0.25], [0.0, -0.5]])
            if atom_syntax == "reduced_sign > 0.0"
            or (not atom_syntax and point.role is PointRole.VERY_HIGH)
            else np.array([[-parameter, 0.25], [0.0, -parameter]])
        )
        perturbation = left @ right.T
        metric_key = "reduced_eta"

    factors = eager.LowRankFactors(left, right)
    if gate_id == "EAGER:factor-projection:whitened-positive-spectrum":
        sigma = np.eye(2) + perturbation
        diagonal_sum = float(sigma[0, 0]) + float(sigma[1, 1])
        diagonal_difference = float(sigma[0, 0]) - float(sigma[1, 1])
        off_diagonal_sum = abs(float(sigma[0, 1])) + abs(float(sigma[1, 0]))
        oracle_smallest = 0.5 * (
            diagonal_sum
            - math.hypot(diagonal_difference, off_diagonal_sum)
        )
        fixture_oracle_admitted = math.isfinite(oracle_smallest) and oracle_smallest > 0.0
    elif gate_id == "EAGER:factor-base:condition-ceiling":
        fixture_oracle_admitted = (
            False
            if "isfinite(base_condition)" in atom_syntax
            else math.isfinite(parameter) and parameter < float(threshold)
        )
    elif gate_id == "EAGER:factor-base:error-budget":
        independent_bound = 2.0 * float(np.finfo(np.float32).eps) * 17.0 * max(1.0, abs(math.log(parameter)))
        fixture_oracle_admitted = independent_bound <= float(threshold)
    elif gate_id in {
        "EAGER:factor-reduced:diagonal-certificate",
        "EAGER:factor-reduced:qr-certificate",
    }:
        fixture_oracle_admitted = parameter <= below
    else:
        if gate_id == "EAGER:factor-reduced:acceptance-budget":
            # An independent exhaustive nextafter search over this exact
            # rank-zero fixture found the last admitted input below.  The
            # certificate's own total is deliberately not reused as oracle.
            last_admitted = float.fromhex("0x1.ffffff4000000p-27")
            fixture_oracle_admitted = (
                math.isfinite(parameter) and parameter <= last_admitted
            )
        else:
            independent_bound = (
                math.log1p(parameter) if math.isfinite(parameter) else math.inf
            )
            fixture_oracle_admitted = independent_bound <= float(threshold)
    smallest_holder: dict[str, float] = {}
    original_eigvalsh = eager.np.linalg.eigvalsh

    def recording_eigvalsh(value: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        result = original_eigvalsh(value, *args, **kwargs)
        if gate_id == "EAGER:factor-projection:whitened-positive-spectrum":
            smallest_holder[metric_key] = float(np.min(result))
        return result

    live_values: dict[str, object]
    try:
        with (
            _capture_factor_frame() as live_values,
            patch.object(
                eager.np.linalg,
                "eigvalsh",
                side_effect=recording_eigvalsh,
            ),
        ):
            certificate = eager._factor_projection_certificate(
                perturbation, factors, lam
            )
    except (ArithmeticError, ValueError) as error:
        returned_key = metric_key
        high_sign_fixture = (
            gate_id
            in {
                "EAGER:factor-reduced:diagonal-certificate",
                "EAGER:factor-reduced:qr-certificate",
            }
            and not atom_syntax
            and point.role is PointRole.VERY_HIGH
        )
        if gate_id == "EAGER:factor-projection:whitened-positive-spectrum":
            retained_metric = live_values.get(
                "smallest_eigenvalue",
                smallest_holder.get(metric_key, math.nan),
            )
        elif gate_id in {
            "EAGER:factor-reduced:diagonal-certificate",
            "EAGER:factor-reduced:qr-certificate",
        }:
            retained_metric = live_values.get("reduced_eta", math.inf)
        else:
            retained_metric = live_values.get(metric_key, math.inf)
        production_admitted = False
        point_key = "factor_parameter" if high_sign_fixture else returned_key
        point_value = parameter if high_sign_fixture else retained_metric
        inputs = {
            "lambda_matrix": lam,
            "perturbation": perturbation,
            "factor_left": factors.left,
            "factor_right": factors.right,
            "factor_parameter": parameter,
            returned_key: retained_metric,
        }
        _capture(
            input_key=point_key,
            value=point_value,
            threshold=threshold,
            inputs=inputs,
            direct_input_keys=(
                "lambda_matrix",
                "perturbation",
                "factor_left",
                "factor_right",
                "factor_parameter",
            ),
            direct_return_keys=(
                () if point_key == "factor_parameter" else (returned_key,)
            ),
            axis_input_key="factor_parameter",
            dtype="float64",
        )
        oracle_admitted = fixture_oracle_admitted
        if high_sign_fixture:
            return _failed_direct_call(
                method=method,
                oracle=oracle,
                reason=str(error),
                oracle_admitted=oracle_admitted,
            )
        return _refused(
            method=method,
            oracle=oracle,
            reason=str(error),
            actual_admitted=production_admitted,
            oracle_admitted=oracle_admitted,
        )

    metric = {
        "projection_log_error_bound": certificate.log_error_bound,
        "total_log_error_bound": certificate.total_log_error_bound,
        "base_condition": certificate.base_condition,
        "base_log_error_bound": certificate.base_log_error_bound,
        "reduced_eta": certificate.reduced_eta,
        "smallest_whitened_eigenvalue": smallest_holder.get(metric_key, math.nan),
    }[metric_key]
    if gate_id == "EAGER:factor-projection:whitened-positive-spectrum":
        oracle_admitted = fixture_oracle_admitted
        default_admitted = math.isfinite(float(live_values["eta"]))
    elif gate_id in {
        "EAGER:factor-base:condition-ceiling",
        "EAGER:factor-reduced:diagonal-certificate",
        "EAGER:factor-reduced:qr-certificate",
    }:
        oracle_admitted = fixture_oracle_admitted
        default_admitted = (
            bool(certificate.base_arithmetic_valid)
            if gate_id == "EAGER:factor-base:condition-ceiling"
            else math.isfinite(float(certificate.reduced_log_error_bound))
        )
    elif gate_id == "EAGER:factor-reduced:acceptance-budget":
        oracle_admitted = fixture_oracle_admitted
        default_admitted = bool(live_values["total_valid"])
    else:
        oracle_admitted = fixture_oracle_admitted
        default_admitted = (
            math.isfinite(float(metric)) and float(metric) <= float(threshold)
        )
    production_admitted = default_admitted
    inputs = {
        "lambda_matrix": lam,
        "perturbation": perturbation,
        "factor_left": factors.left,
        "factor_right": factors.right,
        "factor_parameter": parameter,
        metric_key: metric,
    }
    high_sign_fixture = (
        gate_id
        in {
            "EAGER:factor-reduced:diagonal-certificate",
            "EAGER:factor-reduced:qr-certificate",
        }
        and not atom_syntax
        and point.role is PointRole.VERY_HIGH
    )
    point_key = "factor_parameter" if high_sign_fixture else metric_key
    point_value = parameter if high_sign_fixture else metric
    _capture(
        input_key=point_key,
        value=point_value,
        threshold=threshold,
        inputs=inputs,
        direct_input_keys=(
            "lambda_matrix",
            "perturbation",
            "factor_left",
            "factor_right",
            "factor_parameter",
        ),
        direct_return_keys=(
            () if point_key == "factor_parameter" else (metric_key,)
        ),
        axis_input_key="factor_parameter",
        dtype="float64",
    )
    if not production_admitted:
        return _refused(
            method=method,
            oracle=oracle,
            reason=f"real {metric_key}={metric!r} is outside {threshold!r}",
            actual_admitted=production_admitted,
            oracle_admitted=oracle_admitted,
        )
    return _checked(
        method=method,
        oracle=oracle,
        actual=production_admitted,
        expected=oracle_admitted,
    )


def _symmetry_runner(point: ThresholdPoint) -> RawObservation:
    method = "_is_positive_definite"
    oracle = "exact symmetry and two-by-two principal minors"
    matrix = np.array([[2.4, 0.2], [0.2, 1.3]])
    exact_fixture = point.role in {
        PointRole.EXACT,
        PointRole.VERY_LOW,
        PointRole.VERY_HIGH,
    }
    if point.role is PointRole.VERY_LOW:
        matrix[0, 1] = matrix[1, 0] = 0.1
    elif point.role is PointRole.VERY_HIGH:
        matrix[0, 1] = matrix[1, 0] = 0.4
    elif not exact_fixture:
        matrix[1, 0] = _exact_mismatch(matrix[1, 0], point)
    skew = float(matrix[1, 0])
    threshold = float(matrix[0, 1])
    if point.role in {PointRole.EXACT, PointRole.VALID_CAPABILITY}:
        threshold = skew
    elif point.role in {PointRole.VERY_LOW, PointRole.VERY_HIGH}:
        threshold = 0.2
    elif point.role is PointRole.SUBNORMAL_MISMATCH:
        threshold = 0.0
    _capture(
        input_key="lower_off_diagonal",
        value=skew,
        threshold=threshold,
        inputs={"matrix": matrix, "lower_off_diagonal": skew},
        direct_input_keys=("matrix", "lower_off_diagonal"),
        axis_input_key="lower_off_diagonal",
    )
    symmetric = eager._is_symmetric(matrix, rtol=0.0, atol=0.0)
    actual = eager._is_positive_definite(matrix, rtol=0.0, atol=0.0)
    expected = np.array_equal(matrix, matrix.T) and matrix[0, 0] > 0.0 and np.linalg.det(matrix) > 0.0
    if not actual:
        return _refused(method=("_is_symmetric", method), oracle=oracle, reason="exact asymmetry is outside the zero-tolerance gate", actual_admitted=actual, oracle_admitted=expected)
    return _checked(
        method=("_is_symmetric", method),
        oracle=oracle,
        actual=bool(symmetric and actual == bool(expected)),
        expected=True,
    )


def _condition_runner(point: ThresholdPoint) -> RawObservation:
    method = "_condition_certificate"
    oracle = "analytic diagonal condition"
    ceiling_target = 1.0 / np.finfo(float).eps
    ratios = {
        PointRole.VERY_LOW: 2.4,
        PointRole.BELOW_RELATIVE_1E6: ceiling_target * (1.0 - 1e-6),
        PointRole.BELOW_RELATIVE_1E12: ceiling_target * (1.0 - 1e-12),
        PointRole.BELOW_ULP: np.nextafter(ceiling_target, 0.0),
        PointRole.AT: ceiling_target,
        PointRole.ABOVE_ULP: np.nextafter(ceiling_target, math.inf),
        PointRole.ABOVE_RELATIVE_1E12: ceiling_target * (1.0 + 1e-12),
        PointRole.ABOVE_RELATIVE_1E6: ceiling_target * (1.0 + 1e-6),
        PointRole.VERY_HIGH: ceiling_target * 2.0,
        PointRole.EXTREME: math.inf,
        PointRole.VALID_CAPABILITY: ceiling_target,
    }
    ratio = ratios[point.role]
    matrix = np.array([1.0, ratio]) if math.isfinite(ratio) else np.array([0.0, 1.0])
    records_token = _ACTIVE_ATOM_RECORDS.set({})
    try:
        with _recording_production_predicates():
            condition, production_ceiling, resolved = eager._condition_certificate(
                matrix
            )
    finally:
        _ACTIVE_ATOM_RECORDS.reset(records_token)
    _capture(
        input_key="condition",
        value=condition,
        threshold=ceiling_target,
        inputs={
            "matrix": matrix,
            "condition": condition,
            "production_ceiling": production_ceiling,
        },
        direct_input_keys=("matrix",),
        direct_return_keys=("condition", "production_ceiling"),
        axis_input_key="matrix",
    )
    expected = ratio if math.isfinite(ratio) else math.inf
    oracle_resolved = math.isfinite(expected) and expected < ceiling_target
    if not resolved:
        return _refused(
            method=method,
            oracle=oracle,
            reason=(
                f"condition {condition} is not below production ceiling "
                f"{production_ceiling}"
            ),
            actual_admitted=resolved,
            oracle_admitted=oracle_resolved,
        )
    return _checked_evaluated(
        methods=(method,),
        oracle=oracle,
        check_actual=bool(resolved) == bool(oracle_resolved),
        check_expected=True,
        actual=condition,
        expected=expected,
    )


def _lambda_runner(point: ThresholdPoint) -> RawObservation:
    method = "lambda_logdet"
    oracle = "analytic diagonal log determinant"
    tiny = np.finfo(float).tiny
    maxima = {
        PointRole.VERY_LOW: tiny / 2.0,
        PointRole.BELOW_RELATIVE_1E6: tiny * (1.0 - 1e-6),
        PointRole.BELOW_RELATIVE_1E12: tiny * (1.0 - 1e-12),
        PointRole.BELOW_ULP: np.nextafter(tiny, 0.0),
        PointRole.AT: tiny,
        PointRole.ABOVE_ULP: np.nextafter(tiny, math.inf),
        PointRole.ABOVE_RELATIVE_1E12: tiny * (1.0 + 1e-12),
        PointRole.ABOVE_RELATIVE_1E6: tiny * (1.0 + 1e-6),
        PointRole.VERY_HIGH: 2.4,
        PointRole.EXTREME: np.finfo(float).max / 4.0,
        PointRole.VALID_CAPABILITY: tiny / 2.0,
    }
    maximum = maxima[point.role]
    matrix = np.diag([maximum / 1.3, maximum])
    _capture(
        input_key="maximum_magnitude",
        value=maximum,
        threshold=np.finfo(float).tiny,
        inputs={"lambda_matrix": matrix, "maximum_magnitude": maximum},
        direct_input_keys=("lambda_matrix", "maximum_magnitude"),
        axis_input_key="maximum_magnitude",
    )
    scaling_calls = 0
    original_scale = eager._exact_power_of_two_scale

    def recording_scale(*args: object, **kwargs: object) -> object:
        nonlocal scaling_calls
        scaling_calls += 1
        return original_scale(*args, **kwargs)

    with patch.object(
        eager, "_exact_power_of_two_scale", side_effect=recording_scale
    ):
        actual = eager.lambda_logdet(matrix)
    expected = math.log(maximum / 1.3) + math.log(maximum)
    production_rescaled = scaling_calls == 1
    oracle_rescaled = maximum < tiny
    return _evaluated(
        method=method,
        oracle=oracle,
        actual=actual,
        expected=expected,
        observed_side=(
            GateSide.ADMITTED if production_rescaled else GateSide.REFUSED
        ),
        gate_actual=production_rescaled,
        gate_expected=oracle_rescaled,
    )


def _newton_runner(point: ThresholdPoint) -> RawObservation:
    methods = ("finite_perturbation_logdet", "_newton_stability")
    oracle = "analytic diagonal spectral radius"
    lam = np.array([2.0, 3.0])
    rho = {
        PointRole.VERY_LOW: 0.0,
        PointRole.BELOW_RELATIVE_1E6: 1.0 - 1e-6,
        PointRole.BELOW_RELATIVE_1E12: 1.0 - 1e-12,
        PointRole.BELOW_ULP: np.nextafter(1.0, 0.0),
        PointRole.AT: 1.0,
        PointRole.ABOVE_ULP: np.nextafter(1.0, math.inf),
        PointRole.ABOVE_RELATIVE_1E12: 1.0 + 1e-12,
        PointRole.ABOVE_RELATIVE_1E6: 1.0 + 1e-6,
        PointRole.VERY_HIGH: 2.4,
        PointRole.EXTREME: np.finfo(float).max / 4.0,
        PointRole.VALID_CAPABILITY: 0.5,
    }[point.role]
    stable, actual = eager._newton_stability(lam, rho * lam, 2)
    _capture(
        input_key="rho",
        value=actual,
        threshold=1.0,
        inputs={"lambda_matrix": lam, "perturbation": rho * lam, "rho": actual},
        direct_input_keys=("lambda_matrix", "perturbation"),
        direct_return_keys=("rho",),
        axis_input_key="perturbation",
    )
    expected = rho <= 1.0
    try:
        payload = eager.finite_perturbation_logdet(lam, rho * lam)
    except ValueError as error:
        return _refused(method=methods, oracle=oracle, reason=str(error), actual_admitted=False, oracle_admitted=expected)
    if not stable:
        return _checked(
            method=methods,
            oracle=oracle,
            actual=False,
            expected=True,
        )
    expected_payload = math.fsum(math.log(float((1.0 + rho) * item)) for item in lam)
    return _checked_evaluated(
        methods=methods,
        oracle=oracle,
        check_actual=stable == expected,
        check_expected=True,
        actual=payload,
        expected=expected_payload,
    )


def _block_chain_runner(point: ThresholdPoint) -> RawObservation:
    methods = ("_is_block_chain", "state_space_logdet")
    oracle = "exact zero outside block tridiagonal band"
    matrix = np.diag([1.3, 2.4, 3.1, 4.2])
    matrix[0, 0] += 0.01 * _role_ordinal(point)
    exact_fixture = point.role in {
        PointRole.EXACT,
        PointRole.VERY_LOW,
        PointRole.VERY_HIGH,
    }
    if not exact_fixture:
        matrix[0, 3] = _exact_mismatch(0.0, point)
    if point.role is PointRole.VERY_LOW:
        key, value, threshold = "diagonal_entry", float(matrix[0, 0]), 2.4
    elif point.role is PointRole.VERY_HIGH:
        key, value, threshold = "diagonal_entry", float(matrix[-1, -1]), 2.4
    else:
        key, value, threshold = "off_band_entry", float(matrix[0, 3]), 0.0
    _capture(
        input_key=key,
        value=value,
        threshold=threshold,
        inputs={
            "matrix": matrix,
            "block_size": 1,
            "off_band_entry": float(matrix[0, 3]),
            "diagonal_entry": value if key == "diagonal_entry" else float(matrix[0, 0]),
        },
        direct_input_keys=("matrix", "block_size", "off_band_entry", "diagonal_entry"),
        axis_input_key=key,
    )
    actual_block = eager._is_block_chain(matrix, 1, rtol=0.0, atol=0.0)
    oracle_block = not bool(matrix[0, 3] != 0.0)
    try:
        payload = eager.state_space_logdet(matrix, block_size=1)
    except ValueError as error:
        return _refused(
            method=methods,
            oracle=oracle,
            reason=str(error),
            actual_admitted=False,
            oracle_admitted=oracle_block,
        )
    expected_payload = oracles.slogdet_log(matrix)
    return _checked_evaluated(
        methods=methods,
        oracle=oracle,
        check_actual=actual_block is oracle_block,
        check_expected=True,
        actual=payload,
        expected=expected_payload,
    )


def _state_space_runner(point: ThresholdPoint) -> RawObservation:
    method = "state_space_logdet"
    oracle = "NumPy slogdet of block chain"
    matrix = np.array([[2.4, 0.0, 0.0], [0.0, 1.3, 0.1], [0.0, 0.1, 2.1]])
    matrix[1, 1] += 0.01 * _role_ordinal(point)
    exact_fixture = point.role in {
        PointRole.EXACT,
        PointRole.VERY_LOW,
        PointRole.VERY_HIGH,
    }
    if not exact_fixture:
        # Isolate symmetry inside the block band, keeping the earlier exact
        # block-chain premise true so the owned source predicate executes.
        matrix[0, 1] = _exact_mismatch(0.0, point)
    if point.role is PointRole.VERY_LOW:
        key, value, threshold = "diagonal_entry", float(matrix[1, 1]), 2.4
    elif point.role is PointRole.VERY_HIGH:
        key, value, threshold = "diagonal_entry", float(matrix[0, 0]), 1.3
    else:
        key, value, threshold = "upper_band_entry", float(matrix[0, 1]), 0.0
    _capture(
        input_key=key,
        value=value,
        threshold=threshold,
        inputs={
            "matrix": matrix,
            "block_size": 1,
            "upper_band_entry": float(matrix[0, 1]),
            "diagonal_entry": value if key == "diagonal_entry" else float(matrix[1, 1]),
        },
        direct_input_keys=("matrix", "block_size", "upper_band_entry", "diagonal_entry"),
        axis_input_key=key,
    )
    try:
        actual = eager.state_space_logdet(
            matrix,
            block_size=1,
            rtol=0.0,
            atol=0.0,
        )
    except ValueError as error:
        oracle_admitted = bool(
            np.array_equal(matrix, matrix.T)
            and np.all(np.linalg.eigvalsh((matrix + matrix.T) / 2.0) > 0.0)
            and matrix[0, 2] == 0.0
        )
        production_admitted = _active_gate_admission(
            False, true_side=GateSide.REFUSED
        )
        return _refused(method=method, oracle=oracle, reason=str(error), actual_admitted=production_admitted, oracle_admitted=oracle_admitted)
    return _evaluated(method=method, oracle=oracle, actual=actual, expected=oracles.slogdet_log(matrix))


def _structured_runner(point: ThresholdPoint) -> RawObservation:
    methods = (
        "structured_logdet",
        "_is_diagonal",
        "_is_toeplitz",
        "_is_circulant",
        "_circulant_eigenvalues",
    )
    oracle = "explicit structure indices, handwritten DFT, and Kronecker loops"
    scale = 0.5 if point.role is PointRole.VERY_LOW else 2.0 if point.role is PointRole.VERY_HIGH else 1.0
    first = np.diag(np.array([1.3, 2.4]) * scale)
    second = np.diag([1.7, 3.1])
    structure = eager.KroneckerStructure((first, second))
    reconstructed = np.kron(first, second)
    matrix = reconstructed.copy()
    exact_fixture = point.role in {
        PointRole.EXACT,
        PointRole.VERY_LOW,
        PointRole.VERY_HIGH,
    }
    if not exact_fixture:
        if point.role is PointRole.EXTREME:
            matrix[0, 1] = np.finfo(float).max
        elif point.role is PointRole.ULP_MISMATCH:
            matrix[0, 0] = np.nextafter(reconstructed[0, 0], math.inf)
        else:
            matrix[0, 1] = _exact_mismatch(0.0, point)
    if point.role is PointRole.VERY_LOW:
        key, value, threshold = "authoritative_entry", float(matrix[0, 0]), float(np.kron(np.diag([1.3, 2.4]), second)[0, 0])
    elif point.role is PointRole.VERY_HIGH:
        key, value, threshold = "authoritative_entry", float(matrix[-1, -1]), float(np.kron(np.diag([1.3, 2.4]), second)[-1, -1])
    elif point.role is PointRole.ULP_MISMATCH:
        key = "authoritative_entry"
        value = float(matrix[0, 0])
        threshold = float(reconstructed[0, 0])
    else:
        key, value, threshold = "authoritative_entry", float(matrix[0, 1]), 0.0
    _capture(
        input_key=key,
        value=value,
        threshold=threshold,
        inputs={
            "matrix": matrix,
            "kind": "kronecker",
            "structure_first": first,
            "structure_second": second,
            "authoritative_entry": value,
        },
        direct_input_keys=(
            "matrix",
            "kind",
            "structure_first",
            "structure_second",
            "authoritative_entry",
        ),
        axis_input_key=key,
    )
    diagonal = eager._is_diagonal(matrix, rtol=0.0, atol=0.0)
    toeplitz = eager._is_toeplitz(matrix, rtol=0.0, atol=0.0)
    circulant = eager._is_circulant(matrix, rtol=0.0, atol=0.0)
    expected_spectrum = np.asarray(oracles.circulant_eigenvalues(matrix[0]))
    spectrum_is_real_positive = bool(
        np.all(np.abs(np.imag(expected_spectrum)) <= 1e-12)
        and np.all(np.real(expected_spectrum) > 0.0)
    )
    try:
        eigenvalues = eager._circulant_eigenvalues(matrix)
    except ValueError:
        spectrum_check = not spectrum_is_real_positive
    else:
        spectrum_check = bool(
            spectrum_is_real_positive
            and np.allclose(
                eigenvalues,
                np.real(expected_spectrum),
                rtol=1e-12,
                atol=1e-12,
            )
        )
    helper_checks = bool(
        diagonal is oracles.is_diagonal(matrix)
        and toeplitz is oracles.is_toeplitz(matrix)
        and circulant is oracles.is_circulant(matrix)
        and spectrum_check
    )
    try:
        actual = eager.structured_logdet(
            matrix,
            kind="kronecker",
            structure=structure,
        )
    except (ArithmeticError, ValueError) as error:
        oracle_admitted = bool(np.array_equal(matrix, reconstructed))
        production_admitted = _active_gate_admission(
            False, true_side=GateSide.REFUSED
        )
        return _refused(
            method=methods,
            oracle=oracle,
            reason=str(error),
            actual_admitted=production_admitted,
            oracle_admitted=oracle_admitted,
        )
    oracle_admitted = bool(np.array_equal(matrix, reconstructed))
    production_admitted = _active_gate_admission(
        True, true_side=GateSide.REFUSED
    )
    return _checked_evaluated(
        methods=methods,
        oracle=oracle,
        check_actual=helper_checks and production_admitted == oracle_admitted,
        check_expected=True,
        actual=actual,
        expected=oracles.slogdet_log(matrix),
    )


def _spectral_runner(point: ThresholdPoint) -> RawObservation:
    method = "spectral_radius"
    oracle = "independent dense solve/eigenvalue radius"
    lam = np.array([2.0, 3.0])
    finite_fixture = point.role in {
        PointRole.CAPABILITY_LOW,
        PointRole.VALID_CAPABILITY,
    }
    if finite_fixture:
        rho = {
            PointRole.CAPABILITY_LOW: 0.0,
            PointRole.VERY_LOW: 0.0,
            PointRole.BELOW_RELATIVE_1E6: 0.4 - 4e-7,
            PointRole.BELOW_RELATIVE_1E12: 0.4 - 4e-13,
            PointRole.BELOW_ULP: np.nextafter(0.4, 0.0),
            PointRole.AT: 0.4,
        }.get(point.role, 0.4)
        perturbation = rho * lam
    else:
        perturbation = np.array([math.inf, float(_role_ordinal(point))])
    _capture(
        input_key="perturbation",
        value=perturbation,
        threshold="finite resolved arithmetic",
        inputs={"lambda_matrix": lam, "perturbation": perturbation},
        direct_input_keys=("lambda_matrix", "perturbation"),
        axis_input_key="perturbation",
        dtype=None,
    )
    try:
        actual = eager.spectral_radius(lam, perturbation)
    except (ArithmeticError, ValueError) as error:
        oracle_admitted = bool(np.all(np.isfinite(perturbation)))
        production_admitted = _active_gate_admission(
            False, true_side=GateSide.REFUSED
        )
        return _refused(method=method, oracle=oracle, reason=str(error), actual_admitted=production_admitted, oracle_admitted=oracle_admitted)
    return _evaluated(method=method, oracle=oracle, actual=actual, expected=oracles.spectral_radius(lam, perturbation))


def _rho_runner(kind: str, point: ThresholdPoint) -> RawObservation:
    method = "_validate_strict_rho"
    oracle = "analytic strict-rho and certificate domain"
    lam = np.array([2.0, 3.0])
    actual_rho = 0.5
    certificate = 0.5
    if kind == "upper":
        certificate = {
            PointRole.VERY_LOW: 0.0,
            PointRole.BELOW_RELATIVE_1E6: 0.5 * (1.0 - 1e-6),
            PointRole.BELOW_RELATIVE_1E12: 0.5 * (1.0 - 1e-12),
            PointRole.BELOW_ULP: np.nextafter(0.5, 0.0),
            PointRole.AT: 0.5,
            PointRole.ABOVE_ULP: np.nextafter(0.5, math.inf),
            PointRole.ABOVE_RELATIVE_1E12: 0.5 * (1.0 + 1e-12),
            PointRole.ABOVE_RELATIVE_1E6: 0.5 * (1.0 + 1e-6),
            PointRole.VERY_HIGH: 0.9,
            PointRole.EXTREME: math.nan,
            PointRole.VALID_CAPABILITY: np.nextafter(0.5, 0.0),
        }[point.role]
    elif kind == "actual":
        actual_rho = {
            PointRole.VERY_LOW: 0.0,
            PointRole.BELOW_RELATIVE_1E6: 1.0 - 1e-6,
            PointRole.BELOW_RELATIVE_1E12: 1.0 - 1e-12,
            PointRole.BELOW_ULP: np.nextafter(1.0, 0.0),
            PointRole.AT: 1.0,
            PointRole.ABOVE_ULP: np.nextafter(1.0, math.inf),
            PointRole.ABOVE_RELATIVE_1E12: 1.0 + 1e-12,
            PointRole.ABOVE_RELATIVE_1E6: 1.0 + 1e-6,
            PointRole.VERY_HIGH: 2.4,
            PointRole.EXTREME: math.inf,
            PointRole.VALID_CAPABILITY: 0.5,
        }[point.role]
        certificate = actual_rho
    elif kind == "domain":
        actual_rho = 0.0
        certificate = {
            PointRole.VERY_LOW: 0.0,
            PointRole.BELOW_RELATIVE_1E6: 1.0 - 1e-6,
            PointRole.BELOW_RELATIVE_1E12: 1.0 - 1e-12,
            PointRole.BELOW_ULP: np.nextafter(1.0, 0.0),
            PointRole.AT: 1.0,
            PointRole.ABOVE_ULP: np.nextafter(1.0, math.inf),
            PointRole.ABOVE_RELATIVE_1E12: 1.0 + 1e-12,
            PointRole.ABOVE_RELATIVE_1E6: 1.0 + 1e-6,
            PointRole.VERY_HIGH: 2.4,
            PointRole.EXTREME: math.nan,
            PointRole.VALID_CAPABILITY: 0.5,
        }[point.role]
    key = "certificate" if kind != "actual" else "actual_rho"
    value = certificate if key == "certificate" else actual_rho
    if kind == "actual":
        threshold = 1.0
    elif kind == "upper":
        threshold = actual_rho
    elif point.role in {
        PointRole.ABOVE_ULP,
        PointRole.ABOVE_RELATIVE_1E12,
        PointRole.ABOVE_RELATIVE_1E6,
        PointRole.VERY_HIGH,
    }:
        threshold = 1.0
    else:
        threshold = _threshold_for_role(point, value)
    _capture(
        input_key=key,
        value=value,
        threshold=threshold,
        inputs={
            "lambda_matrix": lam,
            "perturbation": actual_rho * lam,
            "actual_rho": actual_rho,
            "certificate": certificate,
        },
        direct_input_keys=("lambda_matrix", "perturbation", "actual_rho", "certificate"),
        axis_input_key=key,
    )
    oracle_accepted = bool(
        actual_rho < 1.0
        and 0.0 <= certificate < 1.0
        and actual_rho <= certificate
    )
    try:
        actual, certified = eager._validate_strict_rho(lam, actual_rho * lam, certificate)
    except (ArithmeticError, ValueError) as error:
        production_admitted = _active_gate_admission(
            False, true_side=GateSide.REFUSED
        )
        return _refused(method=method, oracle=oracle, reason=str(error), actual_admitted=production_admitted, oracle_admitted=oracle_accepted)
    passed = actual < 1.0 and 0.0 <= certified < 1.0 and actual <= certified
    return _checked(
        method=method,
        oracle=oracle,
        actual=passed,
        expected=True,
    )


def _tail_runner(point: ThresholdPoint) -> RawObservation:
    methods = (
        "trace_log_tail_bound",
        "whole_trace_log_tail_bound",
        "choose_trace_order",
    )
    oracle = "Decimal whole-trace tail and integer scan"
    rho = {
        PointRole.VERY_LOW: -0.5,
        PointRole.BELOW_RELATIVE_1E6: 1.0 - 1e-6,
        PointRole.BELOW_RELATIVE_1E12: 1.0 - 1e-12,
        PointRole.BELOW_ULP: np.nextafter(1.0, 0.0),
        PointRole.AT: 1.0,
        PointRole.ABOVE_ULP: np.nextafter(1.0, math.inf),
        PointRole.ABOVE_RELATIVE_1E12: 1.0 + 1e-12,
        PointRole.ABOVE_RELATIVE_1E6: 1.0 + 1e-6,
        PointRole.VERY_HIGH: 2.4,
        PointRole.EXTREME: math.nan,
        PointRole.VALID_CAPABILITY: 0.4,
    }[point.role]
    order_value = 3
    multiplicity = 2
    tolerance = 1.0e300
    _capture(
        input_key="rho",
        value=rho,
        threshold=1.0,
        inputs={
            "rho": rho,
            "order": order_value,
            "multiplicity": multiplicity,
            "tolerance": tolerance,
        },
        direct_input_keys=("rho", "order", "multiplicity", "tolerance"),
        axis_input_key="rho",
    )
    oracle_admitted = bool(0.0 <= rho < 1.0)
    results: dict[str, object] = {}
    failures: list[str] = []
    for name, call in (
        (methods[0], lambda: eager.trace_log_tail_bound(rho, order_value)),
        (
            methods[1],
            lambda: eager.whole_trace_log_tail_bound(
                rho, order_value, multiplicity
            ),
        ),
        (
            methods[2],
            lambda: eager.choose_trace_order(
                rho, tolerance, multiplicity=multiplicity
            ),
        ),
    ):
        try:
            results[name] = call()
        except (ArithmeticError, ValueError) as error:
            failures.append(f"{name}: {error}")
    production_admitted = not failures
    if failures or not production_admitted:
        return _refused(
            method=methods,
            oracle=oracle,
            reason="; ".join(failures),
            actual_admitted=production_admitted,
            oracle_admitted=oracle_admitted,
        )
    scalar = float(results[methods[0]])
    whole = float(results[methods[1]])
    chosen = int(results[methods[2]])
    if oracle_admitted:
        expected = oracles.whole_trace_tail(rho, order_value, multiplicity)
        expected_order = oracles.smallest_trace_order(
            rho, tolerance, multiplicity
        )
    else:
        with localcontext() as context:
            context.prec = 80
            radius = Decimal.from_float(rho)
            expected = float(
                Decimal(multiplicity)
                * radius ** (order_value + 1)
                / (Decimal(order_value + 1) * (Decimal(1) - radius))
            )
        expected_order = 0
    passed = math.isclose(whole, expected, rel_tol=1e-15) and chosen == expected_order
    return _checked_evaluated(
        methods=methods,
        oracle=oracle,
        check_actual=passed and production_admitted == oracle_admitted,
        check_expected=True,
        actual=2.0 * scalar,
        expected=expected,
    )


def _power_trace_runner(point: ThresholdPoint) -> RawObservation:
    method = "_power_traces_match"
    oracle = "explicit diagonal power traces"
    lam = np.array([2.0, 3.0])
    exact_fixture = point.role in {
        PointRole.EXACT,
        PointRole.VERY_LOW,
        PointRole.VERY_HIGH,
    }
    rho = (
        0.25
        if point.role is PointRole.VERY_LOW
        else 0.75
        if point.role is PointRole.VERY_HIGH
        else 0.5
    )
    perturbation = rho * lam
    traces = list(oracles.exact_power_traces(lam, perturbation, 3))
    if not exact_fixture:
        traces[1] = _exact_mismatch(traces[1], point)
    value = float(traces[1])
    threshold = float(oracles.exact_power_traces(lam, 0.5 * lam, 3)[1])
    if point.role is PointRole.EXACT:
        threshold = value
    elif point.role is PointRole.SUBNORMAL_MISMATCH:
        threshold = 0.0
    _capture(
        input_key="second_trace",
        value=value,
        threshold=threshold,
        inputs={
            "lambda_matrix": lam,
            "perturbation": perturbation,
            "traces": tuple(traces),
            "order": 3,
            "second_trace": value,
        },
        direct_input_keys=("lambda_matrix", "perturbation", "traces", "order", "second_trace"),
        axis_input_key="second_trace",
    )
    actual = eager._power_traces_match(lam, perturbation, traces, 3)
    if not actual:
        return _refused(method=method, oracle=oracle, reason="one-ULP trace mismatch", actual_admitted=actual, oracle_admitted=False)
    return _checked(method=method, oracle=oracle, actual=actual, expected=True)


def _frozen_runner(point: ThresholdPoint) -> RawObservation:
    method = "frozen_hutchinson_trace_logdet"
    oracle = "explicit frozen-probe recurrence"
    dimension = {
        PointRole.CAPABILITY_LOW: 2,
        PointRole.VALID_CAPABILITY: 4,
    }.get(point.role, 2)
    lam = np.arange(2.0, 2.0 + dimension)
    perturbation = 0.25 * lam
    width = {
        PointRole.CAPABILITY_LOW: 2,
        PointRole.VALID_CAPABILITY: 4,
        PointRole.INVALID_CAPABILITY: 1,
        PointRole.CAPABILITY_HIGH: 3,
        PointRole.EXTREME: 5,
    }.get(point.role, 5)
    probes = eager.FrozenProbes(np.array([[1.0] * width, [-1.0] * width]))
    _capture(
        input_key="probe_width",
        value=width,
        threshold=2,
        inputs={
            "lambda_matrix": lam,
            "perturbation": perturbation,
            "probe_values": probes.values,
            "probe_width": width,
            "order": 2,
            "rho": 0.25,
        },
        direct_input_keys=("lambda_matrix", "perturbation", "probe_values", "probe_width", "order", "rho"),
        axis_input_key="probe_width",
        dtype=None,
    )
    oracle_admitted = isinstance(probes, eager.FrozenProbes) and width == len(lam)
    try:
        actual = eager.frozen_hutchinson_trace_logdet(
            lam, perturbation, probes, order=2, rho=0.25
        )
    except (TypeError, ValueError) as error:
        return _refused(
            method=method,
            oracle=oracle,
            reason=str(error),
            actual_admitted=False,
            oracle_admitted=oracle_admitted,
        )
    traces = (dimension * 0.25, dimension * 0.25**2)
    expected = oracles.trace_log_polynomial(oracles.diagonal_logdet(lam), traces, 2)
    return _evaluated(
        method=method,
        oracle=oracle,
        actual=actual,
        expected=expected,
        gate_actual=True,
        gate_expected=oracle_admitted,
    )


def _structure_tolerance_atom(point: ThresholdPoint, syntax: str) -> RawObservation:
    method = "LadderConfig.__post_init__"
    oracle = "plain Python domain for structure_rtol"
    targets_atol = "structure_atol" in syntax
    targets_finite = "isfinite" in syntax
    field = "structure_atol" if targets_atol else "structure_rtol"
    value = math.nan if targets_finite else -np.nextafter(0.0, math.inf)
    values = {"structure_rtol": 0.2, "structure_atol": 0.3}
    values[field] = value
    _capture(
        input_key=field,
        value=value,
        threshold="finite" if targets_finite else 0.0,
        inputs=values,
        direct_input_keys=("structure_rtol", "structure_atol"),
        axis_input_key=field,
    )
    try:
        eager.LadderConfig(**values)
    except ValueError as error:
        return _refused(method=method, oracle=oracle, reason=str(error), actual_admitted=False, oracle_admitted=False)
    raise AssertionError(f"{field} source atom unexpectedly admitted {value!r}")


def _array_atom(point: ThresholdPoint, syntax: str) -> RawObservation:
    method = "_read_only_array"
    oracle = "shape/finiteness/dtype/immutability policy"
    if syntax == "ndim is not None":
        value = np.array([[1.3, 0.2], [0.2, 2.4]])
        ndim = None
        expected_side = GateSide.ADMITTED
    elif syntax == "array.ndim != ndim":
        value = np.array([[1.3, 0.2], [0.2, 2.4]])
        ndim = 1
        expected_side = GateSide.REFUSED
    elif syntax == "array.ndim not in (1, 2)":
        value = np.ones((1, 1, 1), dtype=float) * 1.3
        ndim = None
        expected_side = GateSide.REFUSED
    else:
        value = np.array([[1.3, math.nan], [0.2, 2.4]])
        ndim = 2
        expected_side = GateSide.REFUSED
    terminal = float(value.flat[-1])
    _capture(
        input_key="array_rank" if "ndim" in syntax else "terminal_entry",
        value=value.ndim if "ndim" in syntax else terminal,
        threshold=ndim if syntax == "array.ndim != ndim" else (2 if "ndim" in syntax else "finite"),
        inputs={
            "value": value,
            "ndim": ndim,
            "array_rank": value.ndim,
            "terminal_entry": terminal,
        },
        direct_input_keys=("value", "ndim", "array_rank", "terminal_entry"),
        axis_input_key="array_rank" if "ndim" in syntax else "terminal_entry",
        dtype=None if "ndim" in syntax else "float64",
    )
    try:
        returned = eager._read_only_array(value, ndim=ndim)
    except ValueError as error:
        if expected_side is GateSide.ADMITTED:
            raise AssertionError("ndim=None branch selector was not admitted") from error
        return _refused(method=method, oracle=oracle, reason=str(error), actual_admitted=False, oracle_admitted=expected_side is GateSide.ADMITTED)
    if expected_side is GateSide.REFUSED:
        raise AssertionError(f"array atom {syntax!r} unexpectedly admitted")
    return _checked(
        method=method,
        oracle=oracle,
        actual=bool(
            returned.shape == (2, 2) and np.all(np.isfinite(returned))
        ),
        expected=True,
    )


def _symmetry_atom(point: ThresholdPoint, syntax: str) -> RawObservation:
    method = "_is_positive_definite"
    oracle = "exact symmetry and two-by-two principal minors"
    if "_is_symmetric" in syntax:
        matrix = np.array([[2.4, 0.2], [0.2, 1.3]])
        matrix[1, 0] = np.nextafter(matrix[1, 0], math.inf)
    elif "isfinite(eigenvalues)" in syntax:
        matrix = np.full((2, 2), 0.75 * np.finfo(float).max)
    else:
        matrix = np.diag([0.0, 2.4])
    _capture(
        input_key="matrix",
        value=matrix,
        threshold="finite positive spectrum",
        inputs={"matrix": matrix},
        direct_input_keys=("matrix",),
        direct_return_keys=(),
        axis_input_key="matrix",
        dtype="float64",
    )
    with np.errstate(over="ignore", invalid="ignore"):
        eager._is_symmetric(matrix, rtol=0.0, atol=0.0)
        actual = eager._is_positive_definite(matrix, rtol=0.0, atol=0.0)
    if actual:
        raise AssertionError(f"symmetry atom {syntax!r} unexpectedly admitted")
    return _refused(method=("_is_symmetric", method), oracle=oracle, reason=f"source premise {syntax} is false", actual_admitted=actual, oracle_admitted=False)


def _condition_atom(point: ThresholdPoint, syntax: str) -> RawObservation:
    method = "_condition_certificate"
    oracle = "analytic diagonal condition"
    ceiling_target = 1.0 / float(np.finfo(float).eps)
    matrix = (
        np.array([0.0, 2.4])
        if "isfinite" in syntax
        else np.array([1.0, ceiling_target])
    )
    condition, ceiling, resolved = eager._condition_certificate(matrix)
    _capture(
        input_key="condition",
        value=condition,
        threshold=ceiling,
        inputs={"matrix": matrix, "condition": condition},
        direct_input_keys=("matrix",),
        direct_return_keys=("condition",),
        axis_input_key="matrix",
    )
    if resolved:
        raise AssertionError("singular condition atom unexpectedly resolved")
    return _refused(method=method, oracle=oracle, reason="non-finite condition", actual_admitted=resolved, oracle_admitted=False)


def _balance_atom(point: ThresholdPoint, syntax: str) -> RawObservation:
    method = "_balanced_factor_columns"
    oracle = "hex-level reversible power-of-two gauge"
    targets_left = "scaled_left" in syntax or "restored_left" in syntax
    targets_finite = "isfinite(scaled" in syntax
    if targets_finite:
        raise AssertionError("scaled-factor finiteness atoms are static")
    subnormal = np.nextafter(0.0, 1.0)
    if targets_left:
        left = np.array([[np.finfo(float).max], [subnormal]])
        right = np.array([[subnormal], [0.0]])
    else:
        left = np.array([[subnormal], [0.0]])
        right = np.array([[np.finfo(float).max], [subnormal]])
    _capture(
        input_key="left" if targets_left else "right",
        value=left if targets_left else right,
        threshold="bitwise restoration",
        inputs={"left": left, "right": right},
        direct_input_keys=("left", "right"),
        direct_return_keys=(),
        axis_input_key="left" if targets_left else "right",
        dtype="float64",
    )
    try:
        eager._balanced_factor_columns(left, right)
    except ValueError as error:
        return _refused(method=method, oracle=oracle, reason=str(error), actual_admitted=False, oracle_admitted=False)
    raise AssertionError(f"factor-balance atom {syntax!r} unexpectedly admitted")


def _reconstruction_atom(point: ThresholdPoint, syntax: str) -> RawObservation:
    method = "_matching_factor_reconstruction"
    oracle = "explicit exact factor reconstruction"
    left = np.array(
        [
            [-0.024895779915063802, -2.599679359057706e-06, 53864.52227247717],
            [20766.170605107334, -3.427476755552765, 5.1286725272340755e-06],
            [-0.00015773049500113667, 1.345493870380258e-08, 6.593835057093625e-07],
        ]
    )
    right = np.array(
        [
            [-867038.8976499407, -1.4729351079461674, 0.044913211378915706],
            [-1.7139199408547847e-06, 261216.58902192063, -6142.464524260306],
            [-0.5285622735918152, -11.076426184894645, -1.665057247153446e-07],
        ]
    )
    factors = eager.LowRankFactors(left, right)
    canonical = np.ascontiguousarray(factors.left) @ np.ascontiguousarray(factors.right).T
    alternate = np.asfortranarray(factors.left) @ np.asfortranarray(factors.right).T
    if np.array_equal(canonical, alternate):
        raise AssertionError("layout stress fixture lost its last-bit distinction")
    value = alternate.copy()
    canonical_atom = "canonical" in syntax
    if not canonical_atom:
        value[0, 0] = np.nextafter(value[0, 0], math.inf)
    _capture(
        input_key="authoritative_entry",
        value=float(value[0, 0]),
        threshold=float(canonical[0, 0]),
        inputs={"perturbation": value, "factor_left": factors.left, "factor_right": factors.right, "authoritative_entry": float(value[0, 0])},
        direct_input_keys=("perturbation", "factor_left", "factor_right", "authoritative_entry"),
        axis_input_key="authoritative_entry",
    )
    matches, returned = eager._matching_factor_reconstruction(value, factors)
    if canonical_atom:
        return _checked(
            method=method,
            oracle=oracle,
            actual=bool(matches and np.array_equal(returned, value)),
            expected=True,
        )
    if matches:
        raise AssertionError("all-layout mismatch atom unexpectedly admitted")
    return _refused(method=method, oracle=oracle, reason="all four exact layout products differ", actual_admitted=matches, oracle_admitted=False)


def _certificate_atom(point: ThresholdPoint, syntax: str) -> RawObservation:
    entry = _ACTIVE_ENTRY.get()
    if entry is None:
        raise AssertionError("certificate atom has no active gate")
    role = _certificate_atom_role(entry.gate_id, syntax)
    return _certificate_runner(
        _point(role, GateSide.REFUSED, "real factor atom", syntax)
    )


def _certificate_atom_role(gate_id: str, syntax: str) -> PointRole:
    """Select a real producer fixture for the named source predicate."""
    if gate_id == "EAGER:factor-projection:whitened-positive-spectrum":
        return PointRole.EXTREME if "isfinite" in syntax else PointRole.AT
    if gate_id == "EAGER:factor-projection:error-budget":
        return (
            PointRole.EXTREME
            if "isfinite" in syntax
            else PointRole.REACHABLE_ABOVE
        )
    if gate_id == "EAGER:factor-base:condition-ceiling":
        return PointRole.EXTREME if "isfinite" in syntax else PointRole.AT
    if gate_id == "EAGER:factor-base:error-budget":
        return (
            PointRole.EXTREME
            if "isfinite" in syntax
            else PointRole.REACHABLE_ABOVE
        )
    if gate_id in {
        "EAGER:factor-reduced:diagonal-certificate",
        "EAGER:factor-reduced:qr-certificate",
    }:
        return PointRole.REACHABLE_ABOVE
    if gate_id == "EAGER:factor-reduced:acceptance-budget":
        return (
            PointRole.EXTREME
            if "isfinite" in syntax
            else PointRole.REACHABLE_ABOVE
        )
    raise AssertionError(f"unhandled factor-certificate gate {gate_id}")

def _structured_atom(point: ThresholdPoint, syntax: str) -> RawObservation:
    methods = (
        "structured_logdet",
        "_is_diagonal",
        "_is_toeplitz",
        "_is_circulant",
        "_circulant_eigenvalues",
    )
    oracle = "explicit structure indices, handwritten DFT, and Kronecker loops"
    structure: eager.KroneckerStructure | None = None
    if "np.diag(np.diag(matrix))" in syntax:
        matrix = np.array([[2.4, 0.1], [0.1, 1.3]])
        kind = "diagonal"
        key, value, threshold = "off_diagonal", float(matrix[0, 1]), 0.0
    elif "np.array_equal(matrix, expected)" in syntax:
        matrix = np.array([[2.4, 0.3], [0.4, 2.4]])
        kind = "circulant"
        key, value, threshold = "shifted_entry", float(matrix[1, 0]), 0.3
    elif "diagonal == diagonal[0]" in syntax:
        matrix = np.array(
            [[2.4, 0.2, 0.0], [0.2, 2.5, 0.2], [0.0, 0.2, 2.4]]
        )
        kind = "toeplitz"
        key, value, threshold = "main_diagonal_entry", float(matrix[1, 1]), 2.4
    elif "eigenvalues <= 0.0" in syntax:
        matrix = np.ones((2, 2), dtype=float)
        kind = "circulant"
        key, value, threshold = "smallest_spectrum", 0.0, 0.0
    else:
        factor = np.array([[1.3, 0.1], [0.1, 2.4]])
        structure = eager.KroneckerStructure((factor,))
        reconstructed = factor.copy()
        kind = "kronecker"
        if "shape" in syntax:
            matrix = np.diag([1.3, 2.4, 3.1])
            key, value, threshold = "matrix_size", 3, 2
        else:
            matrix = reconstructed.copy()
            matrix[0, 0] = np.nextafter(matrix[0, 0], math.inf)
            key = "reconstructed_entry"
            value = float(matrix[0, 0])
            threshold = float(reconstructed[0, 0])
    factors = () if structure is None else structure.factors
    _capture(
        input_key=key,
        value=value,
        threshold=threshold,
        inputs={"matrix": matrix, "kind": kind, "structure_factors": factors, key: value},
        direct_input_keys=("matrix", "kind", "structure_factors", key),
        axis_input_key=key,
        dtype=None if key == "matrix_size" else "float64",
    )
    try:
        eager._is_diagonal(matrix, rtol=0.0, atol=0.0)
        eager._is_toeplitz(matrix, rtol=0.0, atol=0.0)
        eager._is_circulant(matrix, rtol=0.0, atol=0.0)
        try:
            eager._circulant_eigenvalues(matrix)
        except ValueError:
            pass
        eager.structured_logdet(matrix, kind=kind, structure=structure)
    except ValueError as error:
        return _refused(
            method=methods,
            oracle=oracle,
            reason=str(error),
            actual_admitted=False,
            oracle_admitted=False,
        )
    raise AssertionError(f"structured atom {syntax!r} unexpectedly admitted")


def _spectral_atom(point: ThresholdPoint, syntax: str) -> RawObservation:
    method = "spectral_radius"
    oracle = "independent dense solve/eigenvalue radius"
    targets_x = "isfinite(x)" in syntax
    if targets_x:
        lam = np.eye(2) * np.nextafter(0.0, 1.0)
        perturbation = np.eye(2) * np.finfo(float).max
    else:
        lam = np.eye(2)
        perturbation = np.full((2, 2), 0.75 * np.finfo(float).max)
    _capture(
        input_key="perturbation",
        value=perturbation,
        threshold="finite",
        inputs={
            "lambda_matrix": lam,
            "perturbation": perturbation,
        },
        direct_input_keys=("lambda_matrix", "perturbation"),
        direct_return_keys=(),
        axis_input_key="perturbation",
        dtype="float64",
    )
    try:
        eager.spectral_radius(lam, perturbation)
    except ValueError as error:
        return _refused(method=method, oracle=oracle, reason=str(error), actual_admitted=False, oracle_admitted=False)
    raise AssertionError(f"spectral atom {syntax!r} unexpectedly admitted")


def _block_chain_atom(point: ThresholdPoint, syntax: str) -> RawObservation:
    oracle = "exact zero outside block tridiagonal band"
    matrix = np.diag([1.3, 2.4, 3.1, 4.2])
    block_size = 1
    if syntax == "block_size < 1":
        block_size = 0
        key, value, threshold = "block_size", block_size, 1
    else:
        matrix[0, 3] = np.nextafter(0.0, math.inf)
        key, value, threshold = "off_band_entry", float(matrix[0, 3]), 0.0
    _capture(
        input_key=key,
        value=value,
        threshold=threshold,
        inputs={"matrix": matrix, "block_size": block_size, key: value},
        direct_input_keys=tuple(dict.fromkeys(("matrix", "block_size", key))),
        axis_input_key=key,
        dtype=None if key == "block_size" else "float64",
    )
    if syntax.startswith("_is_block_chain(dense"):
        try:
            eager.state_space_logdet(matrix, block_size=block_size)
        except ValueError as error:
            return _refused(method="state_space_logdet", oracle=oracle, reason=str(error), actual_admitted=False, oracle_admitted=False)
        raise AssertionError("public block-chain predicate unexpectedly admitted")
    actual = eager._is_block_chain(matrix, block_size, rtol=0.0, atol=0.0)
    if actual:
        raise AssertionError(f"block-chain atom {syntax!r} unexpectedly admitted")
    return _refused(method="_is_block_chain", oracle=oracle, reason=f"source premise {syntax} failed", actual_admitted=actual, oracle_admitted=False)


def _state_payload_atom(point: ThresholdPoint, syntax: str) -> RawObservation:
    method = "state_space_logdet"
    oracle = "NumPy slogdet of block chain"
    matrix = np.array([[2.4, 0.2, 0.0], [0.2, 1.3, 0.1], [0.0, 0.1, 2.1]])
    if "isfinite(total)" in syntax:
        raise AssertionError("final state-space total finiteness atoms are static")
    if "_is_symmetric" in syntax:
        matrix[1, 0] = 0.3
    elif "_is_positive_definite" in syntax:
        matrix[1, 1] = -1.3
    key = "minimum_diagonal"
    value = float(matrix[1, 1])
    _capture(
        input_key=key,
        value=value,
        threshold=0.0,
        inputs={"matrix": matrix, "block_size": 1, key: value},
        direct_input_keys=("matrix", "block_size", key),
        direct_return_keys=(),
        axis_input_key="matrix",
    )
    try:
        eager.state_space_logdet(matrix, block_size=1)
    except ValueError as error:
        return _refused(method=method, oracle=oracle, reason=str(error), actual_admitted=False, oracle_admitted=False)
    raise AssertionError(f"state-space payload atom {syntax!r} unexpectedly admitted")


def _tail_atom(point: ThresholdPoint, syntax: str) -> RawObservation:
    oracle = "Decimal whole-trace tail and integer scan"
    rho, order, multiplicity, tolerance = 0.4, 3, 2, 0.1
    if syntax == "0.0 <= rho < 1.0":
        rho = 1.0
        method = "trace_log_tail_bound"
        operation = lambda: eager.trace_log_tail_bound(rho, order)
        key, value, threshold = "rho", rho, 1.0
    elif syntax == "order < 0":
        order = -1
        method = "trace_log_tail_bound"
        operation = lambda: eager.trace_log_tail_bound(rho, order)
        key, value, threshold = "order", order, 0
    elif syntax == "multiplicity < 1":
        multiplicity = 0
        method = "whole_trace_log_tail_bound"
        operation = lambda: eager.whole_trace_log_tail_bound(rho, order, multiplicity)
        key, value, threshold = "multiplicity", multiplicity, 1
    elif "isfinite(tolerance)" in syntax:
        tolerance = math.nan
        method = "choose_trace_order"
        operation = lambda: eager.choose_trace_order(rho, tolerance, multiplicity=multiplicity)
        key, value, threshold = "tolerance", tolerance, "finite"
    elif syntax == "tolerance <= 0.0":
        tolerance = 0.0
        method = "choose_trace_order"
        operation = lambda: eager.choose_trace_order(rho, tolerance, multiplicity=multiplicity)
        key, value, threshold = "tolerance", tolerance, 0.0
    else:
        tolerance = eager.whole_trace_log_tail_bound(rho, 0, multiplicity) / 2.0
        method = "choose_trace_order"
        operation = lambda: eager.choose_trace_order(rho, tolerance, multiplicity=multiplicity)
        key, value, threshold = "initial_tail", eager.whole_trace_log_tail_bound(rho, 0, multiplicity), tolerance
    _capture(
        input_key=key,
        value=value,
        threshold=threshold,
        inputs={"rho": rho, "order": order, "multiplicity": multiplicity, "tolerance": tolerance, key: value},
        direct_input_keys=tuple(dict.fromkeys(("rho", "order", "multiplicity", "tolerance", key))),
        axis_input_key=key,
        dtype=None if key in {"order", "multiplicity"} else "float64",
    )
    try:
        actual = operation()
    except ValueError as error:
        return _refused(method=method, oracle=oracle, reason=str(error), actual_admitted=False, oracle_admitted=False)
    expected = oracles.smallest_trace_order(rho, tolerance, multiplicity)
    return _checked(
        method=method,
        oracle=oracle,
        actual=bool(method == "choose_trace_order" and actual == expected),
        expected=True,
    )


def _power_trace_atom(point: ThresholdPoint, syntax: str) -> RawObservation:
    method = "_power_traces_match"
    oracle = "explicit diagonal power traces"
    lam = np.array([2.0, 3.0])
    perturbation = 0.5 * lam
    order = 3
    traces = list(oracles.exact_power_traces(lam, perturbation, order))
    if syntax == "order < 0":
        order = -1
        key, value, threshold = "order", order, 0
    elif syntax == "len(traces) < order":
        traces = traces[:-1]
        key, value, threshold = "trace_count", len(traces), order
    elif "isfinite(supplied)" in syntax:
        traces[1] = math.nan
        key, value, threshold = "second_trace", traces[1], "finite"
    else:
        traces[1] = np.nextafter(traces[1], math.inf)
        key, value, threshold = "second_trace", traces[1], float(oracles.exact_power_traces(lam, perturbation, 3)[1])
    inputs = {"lambda_matrix": lam, "perturbation": perturbation, "traces": tuple(traces), "order": order, key: value}
    _capture(
        input_key=key,
        value=value,
        threshold=threshold,
        inputs=inputs,
        direct_input_keys=tuple(dict.fromkeys(("lambda_matrix", "perturbation", "traces", "order", key))),
        axis_input_key=key,
        dtype=None if key in {"order", "trace_count"} else "float64",
    )
    actual = eager._power_traces_match(lam, perturbation, traces, order)
    if actual:
        raise AssertionError(f"power-trace atom {syntax!r} unexpectedly admitted")
    return _refused(method=method, oracle=oracle, reason=f"source premise {syntax} failed", actual_admitted=actual, oracle_admitted=False)


def _frozen_atom(point: ThresholdPoint, syntax: str) -> RawObservation:
    method = "frozen_hutchinson_trace_logdet"
    oracle = "explicit frozen-probe recurrence"
    lam = np.array([2.0, 3.0])
    perturbation = 0.2 * lam
    order = 2
    probe_values = np.array([[1.0, -1.0], [1.0, 1.0]])
    probes: object = eager.FrozenProbes(probe_values)
    if "type(probes)" in syntax:
        class ProbeWrapper:
            values = probe_values

        probes = ProbeWrapper()
        key, value, threshold = "probe_type", type(probes).__name__, "FrozenProbes"
    elif "vectors.shape[1]" in syntax:
        probes = eager.FrozenProbes(np.array([[1.0], [-1.0]]))
        key, value, threshold = "probe_width", 1, 2
    else:
        order = -1
        key, value, threshold = "order", order, 0
    _capture(
        input_key=key,
        value=value,
        threshold=threshold,
        inputs={"lambda_matrix": lam, "perturbation": perturbation, "probe_values": probes.values, "order": order, key: value},
        direct_input_keys=tuple(dict.fromkeys(("lambda_matrix", "perturbation", "probe_values", "order", key))),
        axis_input_key=key,
        dtype=None,
    )
    try:
        eager.frozen_hutchinson_trace_logdet(lam, perturbation, probes, order=order, rho=0.25)
    except (TypeError, ValueError) as error:
        return _refused(method=method, oracle=oracle, reason=str(error), actual_admitted=False, oracle_admitted=False)
    raise AssertionError(f"frozen-probe atom {syntax!r} unexpectedly admitted")


def _semantic_atom_observation(
    gate_id: str,
    syntax: str,
    point: ThresholdPoint,
    observed_side: GateSide,
) -> RawObservation | None:
    """Execute source-specific atomic fixtures where the atom changes inputs."""
    del observed_side
    if gate_id == "EAGER:LadderConfig:structure-tolerance-domain":
        return _structure_tolerance_atom(point, syntax)
    if gate_id == "EAGER:array-normalization:shape-and-finiteness":
        return _array_atom(point, syntax)
    if gate_id == "EAGER:symmetry:tolerant-representative":
        return _symmetry_atom(point, syntax)
    if gate_id == "EAGER:dense-condition:strict-dtype-ceiling":
        return _condition_atom(point, syntax)
    if gate_id == "EAGER:factor-balance:exact-power-of-two-reversibility":
        return _balance_atom(point, syntax)
    if gate_id == "EAGER:factor-reconstruction:layout-exactness":
        return _reconstruction_atom(point, syntax)
    if gate_id in {
        "EAGER:factor-projection:whitened-positive-spectrum",
        "EAGER:factor-base:condition-ceiling",
        "EAGER:factor-reduced:diagonal-certificate",
        "EAGER:factor-reduced:qr-certificate",
        "EAGER:factor-reduced:acceptance-budget",
    }:
        return _certificate_atom(point, syntax)
    if gate_id == "EAGER:structured:exact-shape-and-spectrum":
        return _structured_atom(point, syntax)
    if gate_id == "EAGER:spectral-radius:finite-measurement":
        return _spectral_atom(point, syntax)
    if gate_id == "EAGER:state-space:block-chain-exactness":
        return _block_chain_atom(point, syntax)
    if gate_id == "EAGER:state-space:payload-domain":
        return _state_payload_atom(point, syntax)
    if gate_id == "EAGER:trace:tail-domain-and-order":
        return _tail_atom(point, syntax)
    if gate_id == "EAGER:trace:exact-power-trace-evidence":
        return _power_trace_atom(point, syntax)
    if gate_id == "EAGER:frozen-probes:identity-width-order":
        return _frozen_atom(point, syntax)
    return None


def _make_suites() -> tuple[BoundarySuite, ...]:
    suites: list[BoundarySuite] = []
    v = ExecutionClass.VALIDATION_ONLY
    p = ExecutionClass.PAYLOAD_OR_REFUSAL
    two = ExecutionClass.TWO_PAYLOAD

    for gate_id, field, good, bad, topology in (
        ("EAGER:LadderConfig:integer-threshold-domain", "low_rank_max", 13, -1, BoundaryTopology.INTEGER),
        ("EAGER:LadderConfig:low-rank-fraction-domain", "low_rank_fraction", 0.125, np.nextafter(0.0, -math.inf), BoundaryTopology.FLOAT),
        ("EAGER:LadderConfig:structure-tolerance-domain", "structure_rtol", 0.3, np.nextafter(0.0, -math.inf), BoundaryTopology.FLOAT),
    ):
        grid, admit, refuse = _config_runner(field, good, bad)
        grid_points = (
            tuple(
                point
                for point in float_grid(
                    below=GateSide.REFUSED,
                    at=GateSide.ADMITTED,
                    above=GateSide.ADMITTED,
                    very_low=GateSide.REFUSED,
                    very_high=GateSide.ADMITTED,
                    extreme=GateSide.REFUSED,
                    threshold="0",
                )
                if point.role
                not in {
                    PointRole.BELOW_RELATIVE_1E6,
                    PointRole.BELOW_RELATIVE_1E12,
                    PointRole.ABOVE_RELATIVE_1E12,
                    PointRole.ABOVE_RELATIVE_1E6,
                }
            )
            if field == "structure_rtol"
            else None
        )
        suites.append(_suite(gate_id, family=FixtureFamily.LADDER_SIZE_RANK_ROUTING, execution_class=v, topology=topology, direct_methods=("LadderConfig.__post_init__",), independent_oracles=(f"plain Python domain for {field}",), grid=grid, admit=admit, refuse=refuse, grid_points=grid_points, omitted_unrepresentable_roles=_ZERO_RELATIVE_ROLES if field == "structure_rtol" else frozenset()))

    suites.append(_suite("EAGER:array-normalization:shape-and-finiteness", family=FixtureFamily.EXACT_STRUCTURE_EVIDENCE, execution_class=v, topology=BoundaryTopology.EXACT, direct_methods=("_read_only_array",), independent_oracles=("shape/finiteness/dtype/immutability policy",), grid=_array_point_runner, admit=_array_point_runner, refuse=_array_point_runner, grid_points=exact_grid(mismatch=GateSide.ADMITTED, extreme=GateSide.REFUSED)))
    problem_points = tuple(
        point
        for point in float_grid(
            below=GateSide.REFUSED,
            at=GateSide.REFUSED,
            above=GateSide.ADMITTED,
            very_low=GateSide.REFUSED,
            very_high=GateSide.ADMITTED,
            extreme=GateSide.REFUSED,
            threshold="0",
        )
        if point.role
        not in {
            PointRole.BELOW_RELATIVE_1E6,
            PointRole.BELOW_RELATIVE_1E12,
            PointRole.ABOVE_RELATIVE_1E12,
            PointRole.ABOVE_RELATIVE_1E6,
        }
    )
    suites.append(_suite("EAGER:LogDetProblem:lambda-spd", family=FixtureFamily.SPD_SPECTRUM_CONDITION, execution_class=v, topology=BoundaryTopology.FLOAT, direct_methods=("LogDetProblem.__init__",), independent_oracles=("positive principal minors",), grid=_problem_point_runner, admit=lambda point: _problem_runner(point, np.diag([3.1, 2.4]), True), refuse=lambda point: _problem_runner(point, np.diag([3.1, 0.0]), False), grid_points=problem_points, omitted_unrepresentable_roles=_ZERO_RELATIVE_ROLES))
    suites.append(_suite("EAGER:factor-balance:exact-power-of-two-reversibility", family=FixtureFamily.FACTOR_CERTIFICATES, execution_class=v, topology=BoundaryTopology.CAPABILITY, direct_methods=("_balanced_factor_columns",), independent_oracles=("hex-level reversible power-of-two gauge",), grid=_balance_runner, admit=_balance_runner, refuse=_balance_runner))
    suites.append(_suite("EAGER:factor-reconstruction:layout-exactness", family=FixtureFamily.FACTOR_CERTIFICATES, execution_class=v, topology=BoundaryTopology.EXACT, direct_methods=("_matching_factor_reconstruction",), independent_oracles=("explicit exact factor reconstruction",), grid=_reconstruction_runner, admit=_reconstruction_runner, refuse=_reconstruction_runner))

    suites.extend((
        _suite("EAGER:symmetry:tolerant-representative", family=FixtureFamily.SPD_SPECTRUM_CONDITION, execution_class=v, topology=BoundaryTopology.EXACT, direct_methods=("_is_symmetric", "_is_positive_definite"), independent_oracles=("exact symmetry and two-by-two principal minors",), grid=_symmetry_runner, admit=_symmetry_runner, refuse=_symmetry_runner),
        _suite("EAGER:dense-condition:strict-dtype-ceiling", family=FixtureFamily.SPD_SPECTRUM_CONDITION, execution_class=v, topology=BoundaryTopology.FLOAT, direct_methods=("_condition_certificate",), independent_oracles=("analytic diagonal condition",), grid=_condition_runner, admit=_condition_runner, refuse=_condition_runner, grid_points=float_grid(below=GateSide.ADMITTED, at=GateSide.REFUSED, above=GateSide.REFUSED, very_low=GateSide.ADMITTED, very_high=GateSide.REFUSED, extreme=GateSide.REFUSED, threshold="1/eps")),
        _suite(
            "EAGER:lambda-logdet:subnormal-rescale",
            family=FixtureFamily.SPD_SPECTRUM_CONDITION,
            execution_class=two,
            topology=BoundaryTopology.FLOAT,
            direct_methods=("lambda_logdet",),
            independent_oracles=("analytic diagonal log determinant",),
            grid=_lambda_runner,
            admit=_lambda_runner,
            refuse=None,
            grid_points=float_grid(
                below=GateSide.ADMITTED,
                at=GateSide.REFUSED,
                above=GateSide.REFUSED,
                very_low=GateSide.ADMITTED,
                very_high=GateSide.REFUSED,
                extreme=GateSide.REFUSED,
                threshold="finfo(dtype).tiny",
            ),
        ),
        _suite("EAGER:finite:newton-stability-rho", family=FixtureFamily.RHO_CERTIFICATE_TRACE_ORDER, execution_class=p, topology=BoundaryTopology.FLOAT, direct_methods=("finite_perturbation_logdet", "_newton_stability"), independent_oracles=("analytic diagonal spectral radius",), grid=_newton_runner, admit=_newton_runner, refuse=_newton_runner),
        _suite("EAGER:state-space:block-chain-exactness", family=FixtureFamily.EXACT_STRUCTURE_EVIDENCE, execution_class=v, topology=BoundaryTopology.EXACT, direct_methods=("_is_block_chain", "state_space_logdet"), independent_oracles=("exact zero outside block tridiagonal band",), grid=_block_chain_runner, admit=_block_chain_runner, refuse=_block_chain_runner),
        _suite("EAGER:state-space:payload-domain", family=FixtureFamily.EXACT_STRUCTURE_EVIDENCE, execution_class=p, topology=BoundaryTopology.EXACT, direct_methods=("state_space_logdet",), independent_oracles=("NumPy slogdet of block chain",), grid=_state_space_runner, admit=_state_space_runner, refuse=_state_space_runner),
        _suite("EAGER:structured:exact-shape-and-spectrum", family=FixtureFamily.EXACT_STRUCTURE_EVIDENCE, execution_class=p, topology=BoundaryTopology.EXACT, direct_methods=("structured_logdet", "_is_diagonal", "_is_toeplitz", "_is_circulant", "_circulant_eigenvalues"), independent_oracles=("explicit structure indices, handwritten DFT, and Kronecker loops",), grid=_structured_runner, admit=_structured_runner, refuse=_structured_runner),
        _suite("EAGER:spectral-radius:finite-measurement", family=FixtureFamily.RHO_CERTIFICATE_TRACE_ORDER, execution_class=p, topology=BoundaryTopology.CAPABILITY, direct_methods=("spectral_radius",), independent_oracles=("independent dense solve/eigenvalue radius",), grid=_spectral_runner, admit=_spectral_runner, refuse=_spectral_runner),
        _suite("EAGER:trace:certificate-domain", family=FixtureFamily.RHO_CERTIFICATE_TRACE_ORDER, execution_class=v, topology=BoundaryTopology.FLOAT, direct_methods=("_validate_strict_rho",), independent_oracles=("analytic strict-rho and certificate domain",), grid=lambda point: _rho_runner("domain", point), admit=lambda point: _rho_runner("domain", point), refuse=lambda point: _rho_runner("domain", _point(PointRole.AT, GateSide.REFUSED, "atom certificate", "half-open endpoint")), grid_points=float_grid(below=GateSide.ADMITTED, at=GateSide.REFUSED, above=GateSide.REFUSED, very_low=GateSide.ADMITTED, very_high=GateSide.REFUSED, extreme=GateSide.REFUSED, threshold="1")),
        _suite("EAGER:trace:certificate-upper-bound", family=FixtureFamily.RHO_CERTIFICATE_TRACE_ORDER, execution_class=v, topology=BoundaryTopology.FLOAT, direct_methods=("_validate_strict_rho",), independent_oracles=("analytic strict-rho and certificate domain",), grid=lambda point: _rho_runner("upper", point), admit=lambda point: _rho_runner("upper", point), refuse=lambda point: _rho_runner("upper", _point(PointRole.BELOW_ULP, GateSide.REFUSED, "atom certificate", "below actual rho")), grid_points=float_grid(below=GateSide.REFUSED, at=GateSide.ADMITTED, above=GateSide.ADMITTED, very_low=GateSide.REFUSED, very_high=GateSide.ADMITTED, extreme=GateSide.REFUSED, threshold="actual rho")),
        _suite("EAGER:trace:tail-domain-and-order", family=FixtureFamily.RHO_CERTIFICATE_TRACE_ORDER, execution_class=p, topology=BoundaryTopology.FLOAT, direct_methods=("trace_log_tail_bound", "whole_trace_log_tail_bound", "choose_trace_order"), independent_oracles=("Decimal whole-trace tail and integer scan",), grid=_tail_runner, admit=_tail_runner, refuse=_tail_runner, grid_points=float_grid(below=GateSide.ADMITTED, at=GateSide.REFUSED, above=GateSide.REFUSED, very_low=GateSide.REFUSED, very_high=GateSide.REFUSED, extreme=GateSide.REFUSED, threshold="1")),
        _suite("EAGER:trace:exact-power-trace-evidence", family=FixtureFamily.RHO_CERTIFICATE_TRACE_ORDER, execution_class=v, topology=BoundaryTopology.EXACT, direct_methods=("_power_traces_match",), independent_oracles=("explicit diagonal power traces",), grid=_power_trace_runner, admit=_power_trace_runner, refuse=_power_trace_runner),
        _suite("EAGER:frozen-probes:identity-width-order", family=FixtureFamily.RHO_CERTIFICATE_TRACE_ORDER, execution_class=p, topology=BoundaryTopology.CAPABILITY, direct_methods=("frozen_hutchinson_trace_logdet",), independent_oracles=("explicit frozen-probe recurrence",), grid=_frozen_runner, admit=_frozen_runner, refuse=_frozen_runner),
    ))

    if len(suites) != 20:
        raise AssertionError(f"eager provider constructed {len(suites)} suites, expected 20")
    return tuple(suites)


EAGER_SUITES: tuple[BoundarySuite, ...] = _make_suites()


__all__ = ["EAGER_SUITES"]
