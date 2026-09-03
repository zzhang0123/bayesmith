"""Executable direct-boundary suites for the SBC gates (R3 Task 6).

Two boundaries, both CLOSED lower ones, so both grids read below = REFUSED and
at/above = ADMITTED -- the opposite reading from the pilot's two open floors,
and the reason the witness roles below are AT for tighten and the nearest cell
below for loosen.

* ``replicates_meet_floor`` is an INTEGER boundary at D106 = 100. Its
  threshold is a declared constant, so the grid moves the count around the
  constant.
* ``ranks_are_uniform`` is a FLOAT boundary at a threshold the caller COMPUTES
  -- ``ALPHA / K`` for the K latent coordinates being tested together -- so the
  grid moves the p-value around that computed level, which is the boundary
  production actually meets. The second axis cell exercises K = 2 rather than
  K = 1, because a correction applied the wrong way round is a different
  number rather than a different comparison.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from bayesmith.evaluation import ALPHA
from bayesmith.evaluation import sbc as sbc_module
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
    integer_grid,
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
    and entry.gate_id.startswith("SBC:")
}

#: The corrected level a one-coordinate report is judged at: D104 over K = 1.
_LEVEL = ALPHA / 1

#: Both gates admit AT the threshold, so the nearest admitted face is the
#: threshold itself and the nearest refused face is the cell below it.
_WITNESS_ROLES = {
    "SBC:replicates_meet_floor:replicate-floor": (
        PointRole.AT,
        PointRole.BELOW_INTEGER,
    ),
    "SBC:ranks_are_uniform:bonferroni-level": (PointRole.AT, PointRole.BELOW_ULP),
}


def _count_for(point: ThresholdPoint, floor: int) -> int:
    role = point.role
    if role is PointRole.VERY_LOW:
        return 4
    if role is PointRole.BELOW_INTEGER:
        return floor - 1
    if role is PointRole.AT:
        return floor
    if role is PointRole.ABOVE_INTEGER:
        return floor + 1
    if role is PointRole.VERY_HIGH:
        return 10 * floor
    if role is PointRole.EXTREME:
        return 0
    raise AssertionError(f"unexpected role {role.value}")


def _p_value_for(point: ThresholdPoint, level: float) -> float:
    role = point.role
    if role is PointRole.VERY_LOW:
        return 0.0
    if role is PointRole.BELOW_RELATIVE_1E6:
        return level * (1.0 - 1e-6)
    if role is PointRole.BELOW_RELATIVE_1E12:
        return level * (1.0 - 1e-12)
    if role is PointRole.BELOW_ULP:
        return float(np.nextafter(level, -math.inf))
    if role is PointRole.AT:
        return level
    if role is PointRole.ABOVE_ULP:
        return float(np.nextafter(level, math.inf))
    if role is PointRole.ABOVE_RELATIVE_1E12:
        return level * (1.0 + 1e-12)
    if role is PointRole.ABOVE_RELATIVE_1E6:
        return level * (1.0 + 1e-6)
    if role is PointRole.VERY_HIGH:
        return 1.0
    if role is PointRole.EXTREME:
        return math.nan
    raise AssertionError(f"unexpected role {role.value}")


def _floor_runner(entry: GateEntry) -> Runner:
    def run(point: ThresholdPoint) -> RawObservation:
        # Resolved at run time so the in-process mutation harness, which swaps
        # the module attribute, is the callable actually exercised.
        predicate = sbc_module.replicates_meet_floor
        floor = sbc_module.REPLICATE_FLOOR
        value = _count_for(point, floor)
        actual = bool(predicate(value, floor))
        expected = not (value < floor)
        return RawObservation(
            observed_side=GateSide.ADMITTED if actual else GateSide.REFUSED,
            realized_point=realized_point(
                entry=entry,
                point=point,
                quantity=entry.quantity,
                input_key="usable",
                value=value,
                threshold=floor,
                dtype=None,
            ),
            realized_inputs={"usable": value, "floor": floor},
            direct_input_keys=("usable", "floor"),
            direct_return_keys=(),
            direct_calls=("replicates_meet_floor",),
            oracle_checks=(
                oracle_check(
                    oracle="independent complementary integer comparison against the declared replicate floor",
                    actual=actual,
                    expected=expected,
                ),
            ),
        )

    return run


def _level_runner(entry: GateEntry) -> Runner:
    def run(point: ThresholdPoint) -> RawObservation:
        predicate = sbc_module.ranks_are_uniform
        value = _p_value_for(point, _LEVEL)
        actual = bool(predicate(value, _LEVEL))
        expected = not (value < _LEVEL) and not math.isnan(value)
        return RawObservation(
            observed_side=GateSide.ADMITTED if actual else GateSide.REFUSED,
            realized_point=realized_point(
                entry=entry,
                point=point,
                quantity=entry.quantity,
                input_key="p_value",
                value=value,
                threshold=_LEVEL,
                dtype="float64",
            ),
            realized_inputs={"p_value": value, "level": _LEVEL},
            direct_input_keys=("p_value", "level"),
            direct_return_keys=(),
            direct_calls=("ranks_are_uniform",),
            oracle_checks=(
                oracle_check(
                    oracle="independent complementary float comparison against the rate-over-coordinates quotient",
                    actual=actual,
                    expected=expected,
                ),
                # K = 2 rather than K = 1: the level a two-coordinate report is
                # judged at is half the one above, and a correction applied the
                # wrong way round would answer 0.1 here.
                oracle_check(
                    oracle="independent complementary float comparison against the rate-over-coordinates quotient",
                    actual=bool(predicate(value, ALPHA / 2)),
                    expected=not (value < ALPHA / 2) and not math.isnan(value),
                ),
            ),
        )

    return run


def _suite(
    entry: GateEntry,
    runner: Runner,
    *,
    points: tuple[ThresholdPoint, ...],
    topology: BoundaryTopology,
    method: str,
    oracle: str,
) -> BoundarySuite:
    cases = make_grid_cases(
        entry=entry,
        points=points,
        fixture_family=FixtureFamily.PLAN_SCALAR_PROOF_RANGE,
        execution_class=ExecutionClass.VALIDATION_ONLY,
        topology=topology,
        direct_methods=(method,),
        independent_oracles=(oracle,),
        runner=runner,
    )
    tighten_role, loosen_role = _WITNESS_ROLES[entry.gate_id]
    tighten = next(case for case in cases if case.threshold_point.role is tighten_role)
    loosen = next(case for case in cases if case.threshold_point.role is loosen_role)
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


_FLOOR_ID = "SBC:replicates_meet_floor:replicate-floor"
_LEVEL_ID = "SBC:ranks_are_uniform:bonferroni-level"

SBC_SUITES: tuple[BoundarySuite, ...] = (
    _suite(
        _ENTRIES[_FLOOR_ID],
        _floor_runner(_ENTRIES[_FLOOR_ID]),
        points=integer_grid(
            below=GateSide.REFUSED,
            at=GateSide.ADMITTED,
            above=GateSide.ADMITTED,
            very_low=GateSide.REFUSED,
            very_high=GateSide.ADMITTED,
            extreme=GateSide.REFUSED,
            threshold="the D106 replicate floor",
        ),
        topology=BoundaryTopology.INTEGER,
        method="replicates_meet_floor",
        oracle="independent complementary integer comparison against the declared replicate floor",
    ),
    _suite(
        _ENTRIES[_LEVEL_ID],
        _level_runner(_ENTRIES[_LEVEL_ID]),
        points=float_grid(
            below=GateSide.REFUSED,
            at=GateSide.ADMITTED,
            above=GateSide.ADMITTED,
            very_low=GateSide.REFUSED,
            very_high=GateSide.ADMITTED,
            extreme=GateSide.REFUSED,
            threshold="the corrected level",
        ),
        topology=BoundaryTopology.FLOAT,
        method="ranks_are_uniform",
        oracle="independent complementary float comparison against the rate-over-coordinates quotient",
    ),
)


__all__ = ["SBC_SUITES"]
