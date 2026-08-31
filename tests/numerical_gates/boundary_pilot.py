"""Executable direct-boundary suites for the A4 pilot gates (P7).

Both gates are open lower boundaries on a float axis, so both grids read
below/at = REFUSED and above = ADMITTED.  The sampling floor's threshold is
COMPUTED rather than declared -- ``sqrt(p_aug / n_eff)`` at the funnel's own
p_aug=4 and N_eff=200 000 -- and the grid moves the reading around that
computed value, which is the boundary production actually meets.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from bayesmith.dispatch import pilot as pilot_module
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
    and entry.gate_id.startswith("PILOT:")
}

#: The funnel's own floor: four augmented features against 200 000 draws.
_FLOOR = pilot_module.sampling_floor(4, 200_000.0)

#: Both gates admit ABOVE, so the nearest admitted face is ABOVE_ULP and the
#: nearest refused face is the threshold itself.
_WITNESS_ROLES = {
    "PILOT:quadratic_cc_crosses_floor:sampling-floor": (PointRole.ABOVE_ULP, PointRole.AT),
    "PILOT:ratio_exceeds_declared_multiple:declared-multiple": (
        PointRole.ABOVE_ULP,
        PointRole.AT,
    ),
}


def _value_for(point: ThresholdPoint, threshold: float, extreme: float) -> float:
    role = point.role
    if role is PointRole.VERY_LOW:
        return threshold * 0.5
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
        return threshold * 2.0
    if role is PointRole.EXTREME:
        return extreme
    raise AssertionError(f"unexpected role {role.value}")


def _floor_runner(entry: GateEntry) -> Runner:
    def run(point: ThresholdPoint) -> RawObservation:
        # Resolved at run time so the in-process mutation harness, which swaps
        # the module attribute, is the callable actually exercised.
        predicate = pilot_module.quadratic_cc_crosses_floor
        value = _value_for(point, _FLOOR, -math.inf)
        actual = bool(predicate(value, _FLOOR))
        expected = not (value <= _FLOOR)
        observed_side = GateSide.ADMITTED if actual else GateSide.REFUSED
        return RawObservation(
            observed_side=observed_side,
            realized_point=realized_point(
                entry=entry,
                point=point,
                quantity=entry.quantity,
                input_key="quadratic_cc",
                value=value,
                threshold=_FLOOR,
                dtype="float64",
            ),
            realized_inputs={"quadratic_cc": value},
            direct_input_keys=("quadratic_cc",),
            direct_return_keys=(),
            direct_calls=("quadratic_cc_crosses_floor",),
            oracle_checks=(
                oracle_check(
                    oracle="independent complementary float comparison against sqrt(p_aug / N_eff)",
                    actual=actual,
                    expected=expected,
                ),
            ),
        )

    return run


def _multiple_runner(entry: GateEntry) -> Runner:
    def run(point: ThresholdPoint) -> RawObservation:
        predicate = pilot_module.ratio_exceeds_declared_multiple
        threshold = pilot_module.DECLARED_MULTIPLE
        value = _value_for(point, threshold, -math.inf)
        actual = bool(predicate(value))
        expected = not (value <= threshold)
        observed_side = GateSide.ADMITTED if actual else GateSide.REFUSED
        return RawObservation(
            observed_side=observed_side,
            realized_point=realized_point(
                entry=entry,
                point=point,
                quantity=entry.quantity,
                input_key="ratio",
                value=value,
                threshold=threshold,
                dtype="float64",
            ),
            realized_inputs={"ratio": value},
            direct_input_keys=("ratio",),
            direct_return_keys=(),
            direct_calls=("ratio_exceeds_declared_multiple",),
            oracle_checks=(
                oracle_check(
                    oracle="independent complementary float comparison against the declared multiple",
                    actual=actual,
                    expected=expected,
                ),
            ),
        )

    return run


def _float_suite(
    entry: GateEntry,
    runner: Runner,
    * ,
    method: str,
    oracle: str,
    threshold_label: str,
) -> BoundarySuite:
    cases = make_grid_cases(
        entry=entry,
        points=float_grid(
            below=GateSide.REFUSED,
            at=GateSide.REFUSED,
            above=GateSide.ADMITTED,
            very_low=GateSide.REFUSED,
            very_high=GateSide.ADMITTED,
            extreme=GateSide.REFUSED,
            threshold=threshold_label,
        ),
        fixture_family=FixtureFamily.PLAN_SCALAR_PROOF_RANGE,
        execution_class=ExecutionClass.VALIDATION_ONLY,
        topology=BoundaryTopology.FLOAT,
        direct_methods=(method,),
        independent_oracles=(oracle,),
        runner=runner,
    )
    tighten_role, loosen_role = _WITNESS_ROLES[entry.gate_id]
    tighten = next(c for c in cases if c.threshold_point.role is tighten_role)
    loosen = next(c for c in cases if c.threshold_point.role is loosen_role)
    return freeze_suite(
        gate_id=entry.gate_id,
        fixture_family=FixtureFamily.PLAN_SCALAR_PROOF_RANGE,
        execution_class=ExecutionClass.VALIDATION_ONLY,
        topology=BoundaryTopology.FLOAT,
        cases=cases,
        atom_case_ids={},
        tighten_case_id=tighten.case_id,
        loosen_case_id=loosen.case_id,
    )


PILOT_SUITES: tuple[BoundarySuite, ...] = (
    _float_suite(
        _ENTRIES["PILOT:quadratic_cc_crosses_floor:sampling-floor"],
        _floor_runner(_ENTRIES["PILOT:quadratic_cc_crosses_floor:sampling-floor"]),
        method="quadratic_cc_crosses_floor",
        oracle="independent complementary float comparison against sqrt(p_aug / N_eff)",
        threshold_label="sqrt(p_aug / N_eff)",
    ),
    _float_suite(
        _ENTRIES["PILOT:ratio_exceeds_declared_multiple:declared-multiple"],
        _multiple_runner(
            _ENTRIES["PILOT:ratio_exceeds_declared_multiple:declared-multiple"]
        ),
        method="ratio_exceeds_declared_multiple",
        oracle="independent complementary float comparison against the declared multiple",
        threshold_label="7.0",
    ),
)


__all__ = ["PILOT_SUITES"]
