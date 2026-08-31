"""Executable direct-boundary suites for the cost-scoreboard gates (P5)."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from bayesmith.dispatch import costs as costs_module
from tests.numerical_gates.boundary_core import (
    BoundarySuite,
    BoundaryTopology,
    ExecutionClass,
    FixtureFamily,
    GateSide,
    PointRole,
    RawObservation,
    ThresholdPoint,
    float_grid,
    freeze_suite,
    make_grid_cases,
    oracle_check,
    realized_point,
)
from tests.numerical_gates.registry import GATE_REGISTRY, GateEntry, MutationMode

Runner = Callable[[ThresholdPoint], RawObservation]
_ENTRIES = {
    entry.gate_id: entry
    for entry in GATE_REGISTRY
    if entry.mutation_mode is MutationMode.TWO_SIDED
    and entry.gate_id.startswith("COSTS:")
}


_WITNESS_ROLES = {
    "COSTS:gap_is_contested:contested-bandwidth": (PointRole.BELOW_ULP, PointRole.AT),
    "COSTS:timing_noise_in_domain:proper-fraction": (PointRole.BELOW_ULP, PointRole.AT),
    "COSTS:cg_tol_positive:strictly-positive": (PointRole.SUBNORMAL_MISMATCH, PointRole.EXACT),
}


def _value_for(point: ThresholdPoint, threshold: float, extreme: float) -> float:
    role = point.role
    if role is PointRole.VERY_LOW:
        return threshold * 0.5 if threshold else -1.0
    if role is PointRole.BELOW_RELATIVE_1E6:
        return threshold * (1.0 - 1e-6)
    if role is PointRole.BELOW_RELATIVE_1E12:
        return threshold * (1.0 - 1e-12)
    if role is PointRole.BELOW_ULP:
        return float(np.nextafter(threshold, -math.inf))
    if role is PointRole.AT:
        return threshold
    if role is PointRole.ABOVE_ULP:
        return float(np.nextafter(threshold, math.inf))
    if role is PointRole.ABOVE_RELATIVE_1E12:
        return threshold * (1.0 + 1e-12)
    if role is PointRole.ABOVE_RELATIVE_1E6:
        return threshold * (1.0 + 1e-6)
    if role is PointRole.VERY_HIGH:
        return threshold * 2.0 if threshold else 1.0
    if role is PointRole.EXACT:
        return 0.0
    if role is PointRole.ULP_MISMATCH:
        return float(np.nextafter(0.0, -math.inf))
    if role is PointRole.SUBNORMAL_MISMATCH:
        return float(np.nextafter(0.0, math.inf))
    if role is PointRole.MATERIAL_MISMATCH:
        return 1e-6
    if role is PointRole.EXTREME:
        return extreme
    raise AssertionError(f"unexpected role {role.value}")


def _predicate_runner(
    entry: GateEntry,
    predicate_name: str,
    threshold: float,
    extreme: float,
    * ,
    axis_name: str,
    complement: Callable[[float], bool],
) -> Runner:
    def run(point: ThresholdPoint) -> RawObservation:
        # Look the production callable up at run time so the in-process
        # mutation harness (which swaps the module attribute) is seen.
        predicate = getattr(costs_module, predicate_name)
        value = _value_for(point, threshold, extreme)
        actual = bool(predicate(value))
        expected = bool(complement(value))
        observed_side = GateSide.ADMITTED if actual else GateSide.REFUSED
        inputs = {axis_name: value}
        return RawObservation(
            observed_side=observed_side,
            realized_point=realized_point(
                entry=entry,
                point=point,
                quantity=entry.quantity,
                input_key=axis_name,
                value=value,
                threshold=threshold,
                dtype="float64",
            ),
            realized_inputs=inputs,
            direct_input_keys=(axis_name,),
            direct_return_keys=(),
            direct_calls=(predicate_name,),
            oracle_checks=(
                oracle_check(
                    oracle="independent complementary float comparison",
                    actual=actual,
                    expected=expected,
                ),
            ),
        )

    return run


def _freeze(entry: GateEntry, cases, topology: BoundaryTopology) -> BoundarySuite:
    tighten_role, loosen_role = _WITNESS_ROLES[entry.gate_id]
    tighten = next(c for c in cases if c.threshold_point.role is tighten_role)
    loosen = next(c for c in cases if c.threshold_point.role is loosen_role)
    return freeze_suite(
        gate_id=entry.gate_id,
        fixture_family=FixtureFamily.PLAN_SCALAR_PROOF_RANGE,
        execution_class=ExecutionClass.VALIDATION_ONLY,
        topology=topology,
        cases=cases,
        atom_case_ids={},
        tighten_case_id=tighten.case_id,
        loosen_case_id=loosen.case_id,
    )


def _float_suite(
    entry: GateEntry,
    predicate_name: str,
    threshold: float,
    * ,
    complement: Callable[[float], bool],
    axis_name: str,
    threshold_label: str,
) -> BoundarySuite:
    runner = _predicate_runner(
        entry, predicate_name, threshold, math.inf, axis_name=axis_name, complement=complement
    )
    cases = make_grid_cases(
        entry=entry,
        points=float_grid(
            below=GateSide.ADMITTED,
            at=GateSide.REFUSED,
            above=GateSide.REFUSED,
            very_low=GateSide.ADMITTED,
            very_high=GateSide.REFUSED,
            extreme=GateSide.REFUSED,
            threshold=threshold_label,
        ),
        fixture_family=FixtureFamily.PLAN_SCALAR_PROOF_RANGE,
        execution_class=ExecutionClass.VALIDATION_ONLY,
        topology=BoundaryTopology.FLOAT,
        direct_methods=(predicate_name,),
        independent_oracles=("independent complementary float comparison",),
        runner=runner,
    )
    return _freeze(entry, cases, BoundaryTopology.FLOAT)


def _positive_points() -> tuple[ThresholdPoint, ...]:
    return (
        ThresholdPoint(PointRole.VERY_LOW, "negative ordinary", "axis-low", GateSide.REFUSED),
        ThresholdPoint(PointRole.EXACT, "zero", "zero", GateSide.REFUSED),
        ThresholdPoint(PointRole.ULP_MISMATCH, "negative min-subnormal", "one ULP", GateSide.REFUSED),
        ThresholdPoint(PointRole.SUBNORMAL_MISMATCH, "positive min-subnormal", "minimum subnormal", GateSide.ADMITTED),
        ThresholdPoint(PointRole.MATERIAL_MISMATCH, "positive material", "1e-6", GateSide.ADMITTED),
        ThresholdPoint(PointRole.VERY_HIGH, "positive ordinary", "axis-high", GateSide.ADMITTED),
        ThresholdPoint(PointRole.EXTREME, "negative maximum", "extreme", GateSide.REFUSED),
    )


def _positive_suite(
    entry: GateEntry, predicate_name: str, complement: Callable[[float], bool]
) -> BoundarySuite:
    runner = _predicate_runner(
        entry, predicate_name, 0.0, -np.finfo(float).max, axis_name="tol", complement=complement
    )
    cases = make_grid_cases(
        entry=entry,
        points=_positive_points(),
        fixture_family=FixtureFamily.PLAN_SCALAR_PROOF_RANGE,
        execution_class=ExecutionClass.VALIDATION_ONLY,
        topology=BoundaryTopology.EXACT,
        direct_methods=(predicate_name,),
        independent_oracles=("independent complementary float comparison",),
        runner=runner,
    )
    return _freeze(entry, cases, BoundaryTopology.EXACT)


def _not_ge(value: float, threshold: float) -> bool:
    return not (value >= threshold)


def _not_le(value: float, threshold: float) -> bool:
    return not (value <= threshold)


COSTS_SUITES: tuple[BoundarySuite, ...] = (
    _float_suite(
        _ENTRIES["COSTS:gap_is_contested:contested-bandwidth"],
        "gap_is_contested",
        costs_module.CONTESTED_BANDWIDTH,
        complement=lambda v: _not_ge(v, costs_module.CONTESTED_BANDWIDTH),
        axis_name="gap",
        threshold_label="0.25",
    ),
    _float_suite(
        _ENTRIES["COSTS:timing_noise_in_domain:proper-fraction"],
        "timing_noise_in_domain",
        1.0,
        complement=lambda v: _not_ge(v, 1.0),
        axis_name="tol",
        threshold_label="1.0",
    ),
    _positive_suite(
        _ENTRIES["COSTS:cg_tol_positive:strictly-positive"],
        "cg_tol_positive",
        complement=lambda v: _not_le(v, 0.0),
    ),
)


__all__ = ["COSTS_SUITES"]