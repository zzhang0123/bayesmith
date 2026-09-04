"""Executable direct-boundary suites for the prior audit's two gates (R4).

Both are FLOAT boundaries and they read in opposite directions, which is the
reason the witness roles below differ between them.

* ``increments_converge`` is an OPEN UPPER boundary: the sequence converges
  while ``ratio < threshold``, so below is ADMITTED and at/above is REFUSED.
  The two families it separates are not measured but derived -- a flat density
  doubles its mass with its window, ratio exactly 2, and a ``1/x`` tail adds a
  constant ``log 2``, ratio exactly 1 -- so the EXTREME cell is 1.0 rather than
  something pathological: it is the log-divergent case that sits between the
  families and must refuse.

* ``mass_is_normalised`` is a CLOSED boundary on a distance: normalised while
  ``|mass - 1| <= tolerance``, so at and below are ADMITTED. Its threshold is
  ``max(_NORMALISED_TOLERANCE, uncertainty)``, which the caller COMPUTES -- the
  second argument is the extrapolation's own error -- so the grid moves the
  mass around the level production actually meets, and a second oracle cell
  exercises a non-zero uncertainty, because a tolerance that ignored it would
  answer the same at zero and differently everywhere else.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from bayesmith.dispatch import evidence as evidence_module
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
    and entry.gate_id.startswith("EVIDENCE:")
}

#: The uncertainty the normalisation cells are judged at. Zero, so the fixed
#: floor is what governs -- the second oracle check below moves it.
_NO_UNCERTAINTY = 0.0


def _ratio_for(point: ThresholdPoint, threshold: float) -> float:
    """A ratio placed relative to the convergence threshold."""
    role = point.role
    if role is PointRole.VERY_LOW:
        return 0.0
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
        # A flat density: its mass doubles with its window, exactly.
        return 2.0
    if role is PointRole.EXTREME:
        # The log-divergent 1/x tail, which adds a constant log 2 each time.
        # It sits BETWEEN the two families and must refuse -- the one cell
        # where the derivation, rather than a measurement, says the answer.
        return 1.0
    raise AssertionError(f"unexpected role {role.value}")


def _total_for(point: ThresholdPoint, boundary: float) -> float:
    """A MASS placed relative to ``1 + tolerance``, the boundary it meets.

    The grid moves the mass and not the distance, and that is not cosmetic.
    Placing a point at ``tolerance * (1 - 1e-12)`` and then handing the
    predicate ``1.0 + distance`` destroys the delta: ``1.0 + 1e-6`` has an
    absolute spacing of 2.2e-16, so a 1e-18 offset is not representable there
    and the cell realises nothing. The boundary the predicate actually meets is
    on the mass, so the grid lives there too.
    """
    role = point.role
    if role is PointRole.VERY_LOW:
        return 1.0
    if role is PointRole.BELOW_RELATIVE_1E6:
        return boundary * (1.0 - 1e-6)
    if role is PointRole.BELOW_RELATIVE_1E12:
        return boundary * (1.0 - 1e-12)
    if role is PointRole.BELOW_ULP:
        return float(np.nextafter(boundary, -math.inf))
    if role is PointRole.AT:
        return boundary
    if role is PointRole.ABOVE_ULP:
        return float(np.nextafter(boundary, math.inf))
    if role is PointRole.ABOVE_RELATIVE_1E12:
        return boundary * (1.0 + 1e-12)
    if role is PointRole.ABOVE_RELATIVE_1E6:
        return boundary * (1.0 + 1e-6)
    if role is PointRole.VERY_HIGH:
        # An ImproperUniform over [-2, 5], whose mass is 7.
        return 7.0
    if role is PointRole.EXTREME:
        return math.nan
    raise AssertionError(f"unexpected role {role.value}")


def _ratio_runner(entry: GateEntry) -> Runner:
    def run(point: ThresholdPoint) -> RawObservation:
        # Resolved at run time so the in-process mutation harness, which swaps
        # the module attribute, is the callable actually exercised.
        predicate = evidence_module.increments_converge
        threshold = evidence_module._CONVERGENT_INCREMENT_RATIO
        value = _ratio_for(point, threshold)
        actual = bool(predicate(value, threshold))
        expected = not (value >= threshold)
        return RawObservation(
            observed_side=GateSide.ADMITTED if actual else GateSide.REFUSED,
            realized_point=realized_point(
                entry=entry,
                point=point,
                quantity=entry.quantity,
                input_key="ratio",
                value=value,
                threshold=threshold,
                dtype="float64",
            ),
            realized_inputs={"ratio": value, "threshold": threshold},
            direct_input_keys=("ratio", "threshold"),
            direct_return_keys=(),
            direct_calls=("increments_converge",),
            oracle_checks=(
                oracle_check(
                    oracle="independent complementary float comparison of a separately computed increment ratio",
                    actual=actual,
                    expected=expected,
                ),
                # The two families, whose ratios are closed forms rather than
                # measurements: a flat density is exactly 2 and a 1/x tail is
                # exactly 1, and both must refuse at any threshold below one.
                oracle_check(
                    oracle="independent complementary float comparison of a separately computed increment ratio",
                    actual=bool(predicate(2.0, threshold)),
                    expected=False,
                ),
                oracle_check(
                    oracle="independent complementary float comparison of a separately computed increment ratio",
                    actual=bool(predicate(1.0, threshold)),
                    expected=False,
                ),
            ),
        )

    return run


def _tolerance_runner(entry: GateEntry) -> Runner:
    def run(point: ThresholdPoint) -> RawObservation:
        predicate = evidence_module.mass_is_normalised
        tolerance = evidence_module._NORMALISED_TOLERANCE
        boundary = 1.0 + tolerance
        total = _total_for(point, boundary)
        actual = bool(predicate(total, _NO_UNCERTAINTY))
        expected = not math.isnan(total) and not (abs(total - 1.0) > tolerance)
        return RawObservation(
            observed_side=GateSide.ADMITTED if actual else GateSide.REFUSED,
            realized_point=realized_point(
                entry=entry,
                point=point,
                quantity=entry.quantity,
                input_key="total",
                value=total,
                threshold=boundary,
                dtype="float64",
            ),
            realized_inputs={
                "total": total,
                "tolerance": tolerance,
                "uncertainty": _NO_UNCERTAINTY,
            },
            direct_input_keys=("total", "tolerance", "uncertainty"),
            direct_return_keys=(),
            direct_calls=("mass_is_normalised",),
            oracle_checks=(
                oracle_check(
                    oracle="independent complementary float comparison of a separately computed |mass - 1|",
                    actual=actual,
                    expected=expected,
                ),
                # A non-zero uncertainty, because the tolerance is a MAX of two
                # terms: a version reading only the fixed floor would answer
                # the same at zero and differently here. 3.7e-04 is
                # InverseGamma(1,1)'s measured extrapolation error, the case
                # the second term exists for.
                oracle_check(
                    oracle="independent complementary float comparison of a separately computed |mass - 1|",
                    actual=bool(predicate(1.0 + 3.0e-04, 3.7e-04)),
                    expected=True,
                ),
                oracle_check(
                    oracle="independent complementary float comparison of a separately computed |mass - 1|",
                    actual=bool(predicate(1.0 + 4.0e-04, 3.7e-04)),
                    expected=False,
                ),
            ),
        )

    return run


#: Both are FLOAT boundaries; the convergence one ADMITS below and the
#: normalisation one ADMITS at, so their witness faces differ.
_WITNESS_ROLES = {
    "EVIDENCE:increments_converge:convergent-increment-ratio": (
        PointRole.BELOW_ULP,
        PointRole.AT,
    ),
    "EVIDENCE:mass_is_normalised:normalisation-tolerance": (
        PointRole.AT,
        PointRole.ABOVE_ULP,
    ),
}


def _suite(
    entry: GateEntry,
    runner: Runner,
    *,
    points: tuple[ThresholdPoint, ...],
    method: str,
    oracle: str,
) -> BoundarySuite:
    cases = make_grid_cases(
        entry=entry,
        points=points,
        fixture_family=FixtureFamily.PLAN_SCALAR_PROOF_RANGE,
        execution_class=ExecutionClass.VALIDATION_ONLY,
        topology=BoundaryTopology.FLOAT,
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
        topology=BoundaryTopology.FLOAT,
        cases=cases,
        atom_case_ids={},
        tighten_case_id=tighten.case_id,
        loosen_case_id=loosen.case_id,
    )


_RATIO_ID = "EVIDENCE:increments_converge:convergent-increment-ratio"
_TOLERANCE_ID = "EVIDENCE:mass_is_normalised:normalisation-tolerance"

EVIDENCE_SUITES: tuple[BoundarySuite, ...] = (
    _suite(
        _ENTRIES[_RATIO_ID],
        _ratio_runner(_ENTRIES[_RATIO_ID]),
        points=float_grid(
            below=GateSide.ADMITTED,
            at=GateSide.REFUSED,
            above=GateSide.REFUSED,
            very_low=GateSide.ADMITTED,
            very_high=GateSide.REFUSED,
            extreme=GateSide.REFUSED,
            threshold="the convergent-increment ratio",
        ),
        method="increments_converge",
        oracle="independent complementary float comparison of a separately computed increment ratio",
    ),
    _suite(
        _ENTRIES[_TOLERANCE_ID],
        _tolerance_runner(_ENTRIES[_TOLERANCE_ID]),
        points=float_grid(
            below=GateSide.ADMITTED,
            at=GateSide.ADMITTED,
            above=GateSide.REFUSED,
            very_low=GateSide.ADMITTED,
            very_high=GateSide.REFUSED,
            extreme=GateSide.REFUSED,
            threshold="the normalisation tolerance",
        ),
        method="mass_is_normalised",
        oracle="independent complementary float comparison of a separately computed |mass - 1|",
    ),
)


__all__ = ["EVIDENCE_SUITES"]
