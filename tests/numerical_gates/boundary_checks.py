"""Executable direct-boundary suites for the predictive-check gates (R3 §0.4/§0.5).

Two gates, and they are deliberately different topologies of the same
decision.  ``tail_mass_within_rate`` is a float boundary on a probability:
``min(p, 1 - p) >= ALPHA / 2``, which is the whole two-sided band written as
one comparison.  ``draws_resolve_the_band`` is an INTEGER boundary on a draw
count: ``draws >= ceil(1 / (ALPHA / 2))``, so its neighbours are 39 and 41 and
there is no ULP to speak of.

Both are CLOSED at the threshold -- ``>=`` rather than ``>`` -- so the nearest
ADMITTED cell is the threshold itself and the nearest REFUSED cell is one step
below it.  Every scalar gate registered before these two is OPEN, so its
witness pair puts the threshold cell on the REFUSED side: ``(BELOW_ULP, AT)``
where the predicate admits below, ``(ABOVE_ULP, AT)`` where it admits above,
``(SUBNORMAL_MISMATCH, EXACT)`` for the strictly-positive one.  These are the
first with ``AT`` on the ADMITTED side, and the witness roles are where that
is legible: a reviewer who disagrees about which side the boundary falls on
disagrees with these two lines rather than with a comment.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from bayesmith.evaluation import checks as checks_module
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
    and entry.gate_id.startswith("CHECKS:")
}

#: Both gates admit AT, so the nearest admitted face IS the threshold and the
#: nearest refused face is the step below it.
_WITNESS_ROLES = {
    "CHECKS:tail_mass_within_rate:declared-false-positive-rate": (
        PointRole.AT,
        PointRole.BELOW_ULP,
    ),
    "CHECKS:draws_resolve_the_band:p-value-draw-floor": (
        PointRole.AT,
        PointRole.BELOW_INTEGER,
    ),
}

#: The declared tail, recomputed here from the layer's own α rather than
#: written as 0.025: the boundary this grid sweeps has to move when D104 moves,
#: or the grid would be testing a number the production code no longer uses.
_TAIL = checks_module.ALPHA / 2.0

#: The oracle each grid names, spelled once so the observation and the suite
#: cannot describe the comparison differently.
_TAIL_ORACLE = (
    "independent complementary float comparison against half the declared "
    "false-positive rate"
)
_DRAWS_ORACLE = (
    "independent complementary integer comparison against the derived p-value "
    "resolution floor"
)

#: A tail mass is ``min(p, 1 - p)`` over a probability, so its domain is
#: ``[0, 0.5]`` and 0.5 -- a p-value of exactly one half -- is the axis high.
_MAX_TAIL_MASS = 0.5

#: The refused domain extreme for a tail mass is a NEGATIVE one, not +/-inf on
#: the admitted side: a negative tail mass is not a probability, and it is
#: exactly what an unnormalised weighted sum produced during this task's
#: measurement (4000 weights of 1/4000 summed to 1.0000000000000004, so
#: ``min(p, 1 - p)`` came out at -4.4e-16).  The gate refuses it; the grid says
#: so.
_TAIL_EXTREME = -math.inf

#: A draw count of zero: a result carrying no draws at all.  The integer axis's
#: refused extreme, and unlike -inf it is a value this package can actually be
#: handed.
_DRAWS_EXTREME = 0

#: The existing suite's own small budget (``ComputeBudget(draws=8)``), which is
#: the case D105 exists to catch, and the fixtures' 2000, which is the case it
#: exists to admit.
_DRAWS_VERY_LOW = 8
_DRAWS_VERY_HIGH = 2000


def _tail_value(point: ThresholdPoint) -> float:
    role = point.role
    if role is PointRole.VERY_LOW:
        return _TAIL * 0.5
    if role is PointRole.BELOW_RELATIVE_1E6:
        return _TAIL * (1.0 - 1e-6)
    if role is PointRole.BELOW_RELATIVE_1E12:
        return _TAIL * (1.0 - 1e-12)
    if role is PointRole.BELOW_ULP:
        return float(np.nextafter(_TAIL, -math.inf))
    if role is PointRole.AT:
        return _TAIL
    if role is PointRole.ABOVE_ULP:
        return float(np.nextafter(_TAIL, math.inf))
    if role is PointRole.ABOVE_RELATIVE_1E12:
        return _TAIL * (1.0 + 1e-12)
    if role is PointRole.ABOVE_RELATIVE_1E6:
        return _TAIL * (1.0 + 1e-6)
    if role is PointRole.VERY_HIGH:
        return _MAX_TAIL_MASS
    if role is PointRole.EXTREME:
        return _TAIL_EXTREME
    raise AssertionError(f"unexpected role {role.value}")


def _draws_value(point: ThresholdPoint) -> int:
    role = point.role
    floor = checks_module.DRAW_FLOOR
    if role is PointRole.VERY_LOW:
        return _DRAWS_VERY_LOW
    if role is PointRole.BELOW_INTEGER:
        return floor - 1
    if role is PointRole.AT:
        return floor
    if role is PointRole.ABOVE_INTEGER:
        return floor + 1
    if role is PointRole.VERY_HIGH:
        return _DRAWS_VERY_HIGH
    if role is PointRole.EXTREME:
        return _DRAWS_EXTREME
    raise AssertionError(f"unexpected role {role.value}")


def _tail_runner(entry: GateEntry) -> Runner:
    def run(point: ThresholdPoint) -> RawObservation:
        # Resolved at call time so the in-process mutation harness, which swaps
        # the module attribute, is the callable this cell actually exercises.
        predicate = checks_module.tail_mass_within_rate
        value = _tail_value(point)
        actual = bool(predicate(value))
        expected = not (value < _TAIL)
        observed_side = GateSide.ADMITTED if actual else GateSide.REFUSED
        return RawObservation(
            observed_side=observed_side,
            realized_point=realized_point(
                entry=entry,
                point=point,
                quantity=entry.quantity,
                input_key="tail_mass",
                value=value,
                threshold=_TAIL,
                dtype="float64",
            ),
            realized_inputs={"tail_mass": value},
            direct_input_keys=("tail_mass",),
            direct_return_keys=(),
            direct_calls=("tail_mass_within_rate",),
            oracle_checks=(
                oracle_check(
                    oracle=_TAIL_ORACLE, actual=actual, expected=expected
                ),
            ),
        )

    return run


def _draws_runner(entry: GateEntry) -> Runner:
    def run(point: ThresholdPoint) -> RawObservation:
        predicate = checks_module.draws_resolve_the_band
        floor = checks_module.DRAW_FLOOR
        value = _draws_value(point)
        actual = bool(predicate(value))
        expected = not (value < floor)
        observed_side = GateSide.ADMITTED if actual else GateSide.REFUSED
        return RawObservation(
            observed_side=observed_side,
            realized_point=realized_point(
                entry=entry,
                point=point,
                quantity=entry.quantity,
                input_key="draws",
                value=value,
                threshold=floor,
                dtype="int64",
            ),
            realized_inputs={"draws": value},
            direct_input_keys=("draws",),
            direct_return_keys=(),
            direct_calls=("draws_resolve_the_band",),
            oracle_checks=(
                oracle_check(
                    oracle=_DRAWS_ORACLE, actual=actual, expected=expected
                ),
            ),
        )

    return run


def _freeze(
    entry: GateEntry, cases, topology: BoundaryTopology
) -> BoundarySuite:
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


def _tail_suite(entry: GateEntry) -> BoundarySuite:
    cases = make_grid_cases(
        entry=entry,
        points=float_grid(
            below=GateSide.REFUSED,
            at=GateSide.ADMITTED,
            above=GateSide.ADMITTED,
            very_low=GateSide.REFUSED,
            very_high=GateSide.ADMITTED,
            extreme=GateSide.REFUSED,
            threshold="ALPHA / 2 = 0.025",
        ),
        fixture_family=FixtureFamily.PLAN_SCALAR_PROOF_RANGE,
        execution_class=ExecutionClass.VALIDATION_ONLY,
        topology=BoundaryTopology.FLOAT,
        direct_methods=("tail_mass_within_rate",),
        independent_oracles=(_TAIL_ORACLE,),
        runner=_tail_runner(entry),
    )
    return _freeze(entry, cases, BoundaryTopology.FLOAT)


def _draws_suite(entry: GateEntry) -> BoundarySuite:
    cases = make_grid_cases(
        entry=entry,
        points=integer_grid(
            below=GateSide.REFUSED,
            at=GateSide.ADMITTED,
            above=GateSide.ADMITTED,
            very_low=GateSide.REFUSED,
            very_high=GateSide.ADMITTED,
            extreme=GateSide.REFUSED,
            threshold="ceil(1 / (ALPHA / 2)) = 40",
        ),
        fixture_family=FixtureFamily.PLAN_SCALAR_PROOF_RANGE,
        execution_class=ExecutionClass.VALIDATION_ONLY,
        topology=BoundaryTopology.INTEGER,
        direct_methods=("draws_resolve_the_band",),
        independent_oracles=(_DRAWS_ORACLE,),
        runner=_draws_runner(entry),
    )
    return _freeze(entry, cases, BoundaryTopology.INTEGER)


CHECKS_SUITES: tuple[BoundarySuite, ...] = (
    _tail_suite(_ENTRIES["CHECKS:tail_mass_within_rate:declared-false-positive-rate"]),
    _draws_suite(_ENTRIES["CHECKS:draws_resolve_the_band:p-value-draw-floor"]),
)


__all__ = ["CHECKS_SUITES"]
