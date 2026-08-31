"""Executable direct-boundary suites for the collapse-arm gates (P6)."""

from __future__ import annotations

import math
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

from bayesmith.dispatch import collapse as collapse_module
from tests.numerical_gates.boundary_core import (
    BoundarySuite,
    BoundaryTopology,
    ExecutionClass,
    FixtureFamily,
    GateSide,
    PointRole,
    RawObservation,
    ThresholdPoint,
    capability_grid,
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
    and entry.gate_id.startswith("COLLAPSE:")
}

_WITNESS_ROLES = {
    "COLLAPSE:pivots:finite": (PointRole.VALID_CAPABILITY, PointRole.INVALID_CAPABILITY),
    "COLLAPSE:pivots:relative-floor": (PointRole.ABOVE_ULP, PointRole.AT),
}

#: sqrt(eps) in float32 -- the relative pivot floor pivots_constrain_block uses
#: at the suite's ambient dtype (a jnp.asarray of Python floats is float32).
#: The float-grid therefore omits the multiplicative relative roles, which a
#: ~1e-4 relative floor cannot represent to the 1e-4 relative tolerance the
#: standard float grid demands; the ULP and endpoint roles are kept.
_FLOOR32 = np.float32(math.sqrt(np.finfo(np.float32).eps))
_FLOOR = float(_FLOOR32)


def _value_for(point: ThresholdPoint) -> float:
    role = point.role
    t = _FLOOR32
    if role is PointRole.VERY_LOW:
        return float(np.float32(t * np.float32(0.5)))
    if role is PointRole.BELOW_ULP:
        return float(np.nextafter(t, np.float32(-math.inf)))
    if role is PointRole.AT:
        return float(t)
    if role is PointRole.ABOVE_ULP:
        return float(np.nextafter(t, np.float32(math.inf)))
    if role is PointRole.VERY_HIGH:
        return float(np.float32(t * np.float32(2.0)))
    if role is PointRole.EXTREME:
        return 0.0
    raise AssertionError(f"unexpected role {role.value}")


def _finite_runner(entry: GateEntry) -> Runner:
    _PIVOTS = {
        PointRole.CAPABILITY_LOW: [0.5, 1.0, 0.25],
        PointRole.VALID_CAPABILITY: [1.0, 2.0, 0.75],
        PointRole.INVALID_CAPABILITY: [1.0, math.nan, 0.5],
        PointRole.CAPABILITY_HIGH: [1.0, math.inf, 0.5],
        PointRole.EXTREME: [1.0, math.nan, 2.0],
    }

    def run(point: ThresholdPoint) -> RawObservation:
        pivots = np.asarray(_PIVOTS[point.role])
        array = jnp.asarray(pivots)
        actual = bool(collapse_module.pivots_are_finite(array))
        expected = bool(np.all(np.isfinite(pivots)))
        observed_side = GateSide.ADMITTED if actual else GateSide.REFUSED
        return RawObservation(
            observed_side=observed_side,
            realized_point=realized_point(
                entry=entry,
                point=point,
                quantity=entry.quantity,
                input_key="pivots",
                value=pivots.tolist(),
                threshold="all finite",
                dtype="float64",
            ),
            realized_inputs={"pivots": pivots.tolist()},
            direct_input_keys=("pivots",),
            direct_return_keys=(),
            direct_calls=("pivots_are_finite",),
            oracle_checks=(
                oracle_check(
                    oracle="independent numpy finiteness check",
                    actual=actual,
                    expected=expected,
                ),
            ),
        )

    return run


def _floor_runner(entry: GateEntry) -> Runner:
    def run(point: ThresholdPoint) -> RawObservation:
        value = _value_for(point)
        pivots = jnp.asarray([value, 1.0])
        actual = bool(collapse_module.pivots_constrain_block(pivots, 1))
        expected = bool(np.float32(value) > _FLOOR32)
        observed_side = GateSide.ADMITTED if actual else GateSide.REFUSED
        return RawObservation(
            observed_side=observed_side,
            realized_point=realized_point(
                entry=entry,
                point=point,
                quantity=entry.quantity,
                input_key="pivot_ratio",
                value=value,
                threshold=_FLOOR,
                dtype="float32",
            ),
            realized_inputs={"pivot_ratio": value},
            direct_input_keys=("pivot_ratio",),
            direct_return_keys=(),
            direct_calls=("pivots_constrain_block",),
            oracle_checks=(
                oracle_check(
                    oracle="independent high-precision floor comparison",
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


def _capability_suite(entry: GateEntry, runner: Runner) -> BoundarySuite:
    cases = make_grid_cases(
        entry=entry,
        points=capability_grid(valid=GateSide.ADMITTED, invalid=GateSide.REFUSED),
        fixture_family=FixtureFamily.PLAN_SCALAR_PROOF_RANGE,
        execution_class=ExecutionClass.VALIDATION_ONLY,
        topology=BoundaryTopology.CAPABILITY,
        direct_methods=("pivots_are_finite",),
        independent_oracles=("independent numpy finiteness check",),
        runner=runner,
    )
    return _freeze(entry, cases, BoundaryTopology.CAPABILITY)


def _float_suite(entry: GateEntry, runner: Runner) -> BoundarySuite:
    cases = make_grid_cases(
        entry=entry,
        points=float_grid(
            below=GateSide.REFUSED,
            at=GateSide.REFUSED,
            above=GateSide.ADMITTED,
            very_low=GateSide.REFUSED,
            very_high=GateSide.ADMITTED,
            extreme=GateSide.REFUSED,
            threshold="sqrt(eps)",
            include_relative=False,
        ),
        fixture_family=FixtureFamily.PLAN_SCALAR_PROOF_RANGE,
        execution_class=ExecutionClass.VALIDATION_ONLY,
        topology=BoundaryTopology.FLOAT,
        direct_methods=("pivots_constrain_block",),
        independent_oracles=("independent high-precision floor comparison",),
        runner=runner,
    )
    return freeze_suite(
        gate_id=entry.gate_id,
        fixture_family=FixtureFamily.PLAN_SCALAR_PROOF_RANGE,
        execution_class=ExecutionClass.VALIDATION_ONLY,
        topology=BoundaryTopology.FLOAT,
        cases=cases,
        atom_case_ids={},
        tighten_case_id=next(
            c.case_id for c in cases if c.threshold_point.role is PointRole.ABOVE_ULP
        ),
        loosen_case_id=next(
            c.case_id for c in cases if c.threshold_point.role is PointRole.AT
        ),
        omitted_unrepresentable_roles=frozenset(
            {
                PointRole.BELOW_RELATIVE_1E6,
                PointRole.BELOW_RELATIVE_1E12,
                PointRole.ABOVE_RELATIVE_1E12,
                PointRole.ABOVE_RELATIVE_1E6,
            }
        ),
    )


COLLAPSE_SUITES: tuple[BoundarySuite, ...] = (
    _capability_suite(
        _ENTRIES["COLLAPSE:pivots:finite"],
        _finite_runner(_ENTRIES["COLLAPSE:pivots:finite"]),
    ),
    _float_suite(
        _ENTRIES["COLLAPSE:pivots:relative-floor"],
        _floor_runner(_ENTRIES["COLLAPSE:pivots:relative-floor"]),
    ),
)


__all__ = ["COLLAPSE_SUITES"]
