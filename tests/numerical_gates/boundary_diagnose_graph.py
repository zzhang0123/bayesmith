"""Executable direct-boundary suites for diagnose and graph gates.

Every runner calls the owning production helper. Values named as inputs are
actual call arguments or fixture fields consumed by a transparent seam;
values named as returns are captured from the real helper result or seam.
"""

from __future__ import annotations

import builtins
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from bayesmith import det, observe, sample, trace
from bayesmith.diagnose import coupling as coupling_module
from bayesmith.diagnose import map as map_module
from bayesmith.diagnose.coupling import Measured
from bayesmith.diagnose.map import MapEstimate
from bayesmith.errors import GraphError
from bayesmith.graph import reduction as reduction_module
from tests.numerical_gates import oracles
from tests.numerical_gates.boundary_core import (
    AtomBaseline,
    AtomDependencyLogic,
    AtomEvidence,
    AtomPrerequisite,
    AtomReducer,
    AtomRelation,
    AtomRelationKind,
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
    integer_grid,
    make_atom_case,
    make_grid_cases,
    numerical_evaluation,
    oracle_check,
    realized_point,
    source_alias_canonical,
)
from tests.numerical_gates.registry import (
    GATE_REGISTRY,
    FixtureScalePolicy,
    GateEntry,
    MutationMode,
)
from tests.numerical_gates.source_manifest import EXPECTED_SOURCE_MANIFEST

Runner = Callable[[ThresholdPoint], RawObservation]
_ENTRIES = {
    entry.gate_id: entry
    for entry in GATE_REGISTRY
    if entry.mutation_mode is MutationMode.TWO_SIDED
    and entry.gate_id.startswith(("COUPLING:", "MAP:", "GRAPH:"))
}
_ATOM_SYNTAX = {item.candidate_id: item.syntax for item in EXPECTED_SOURCE_MANIFEST}
_NON_UNIT = (1.3, 2.4)


@dataclass(frozen=True, slots=True)
class _SuiteSpec:
    family: FixtureFamily
    execution_class: ExecutionClass
    topology: BoundaryTopology
    direct_methods: tuple[str, ...]
    independent_oracles: tuple[str, ...]
    points: tuple[ThresholdPoint, ...]
    grid_runner: Runner
    atom_runner: Callable[[str], Runner]
    ambiguities: tuple[str, ...] = ()


_MUTATION_WITNESS_ROLES = {
    "COUPLING:_classify_correlation:value-finite": (
        PointRole.VALID_CAPABILITY,
        PointRole.INVALID_CAPABILITY,
    ),
    "COUPLING:_classify_correlation:floor-finite": (
        PointRole.VALID_CAPABILITY,
        PointRole.INVALID_CAPABILITY,
    ),
    "COUPLING:_classify_correlation:lower-noise-floor": (
        PointRole.ABOVE_ULP,
        PointRole.AT,
    ),
    "COUPLING:_classify_correlation:upper-noise-floor": (
        PointRole.BELOW_ULP,
        PointRole.AT,
    ),
    "COUPLING:_condition_number:finite-spectrum": (
        PointRole.VALID_CAPABILITY,
        PointRole.INVALID_CAPABILITY,
    ),
    "COUPLING:_condition_number:positive-spectrum": (
        PointRole.SUBNORMAL_MISMATCH,
        PointRole.EXACT,
    ),
    "COUPLING:block_coupling:f-xx-spd": (
        PointRole.VALID_CAPABILITY,
        PointRole.INVALID_CAPABILITY,
    ),
    "COUPLING:block_coupling:f-tt-spd": (
        PointRole.VALID_CAPABILITY,
        PointRole.INVALID_CAPABILITY,
    ),
    "GRAPH:_names:duplicate-multiplicity": (
        PointRole.AT,
        PointRole.ABOVE_INTEGER,
    ),
    "MAP:map_estimate:finite-derivative-payload": (
        PointRole.VALID_CAPABILITY,
        PointRole.INVALID_CAPABILITY,
    ),
    "MAP:map_estimate:stationarity-floor": (
        PointRole.AT,
        PointRole.ABOVE_ULP,
    ),
    "MAP:map_estimate:curvature-scale-clamp": (
        PointRole.AT,
        PointRole.BELOW_ULP,
    ),
    "MAP:map_estimate:relative-positive-curvature": (
        PointRole.ABOVE_ULP,
        PointRole.AT,
    ),
    "MAP:map_estimate:absolute-curvature": (
        PointRole.ABOVE_ULP,
        PointRole.AT,
    ),
}


def _canonical_atoms(entry: GateEntry) -> tuple[str, ...]:
    return tuple(
        atom_id
        for atom_id in entry.conjunction_atom_ids
        if source_alias_canonical(entry, atom_id) == atom_id
    )


def _atom_baselines(entry: GateEntry) -> tuple[AtomBaseline, ...]:
    refusal_condition = entry.gate_id.endswith(
        (
            "lower-noise-floor",
            "upper-noise-floor",
            "positive-spectrum",
            "stationarity-floor",
        )
    )
    return tuple(
        AtomBaseline(atom_id=atom_id, outcome=not refusal_condition)
        for atom_id in _canonical_atoms(entry)
    )


def _map_finite_relation(
    entry: GateEntry,
    atom_id: str,
    *,
    kind: AtomRelationKind,
    canonical_atom_id: str | None,
) -> AtomRelation:
    canonical = source_alias_canonical(entry, atom_id)
    syntax = _ATOM_SYNTAX[canonical]
    canonical_by_syntax = {
        _ATOM_SYNTAX[candidate]: candidate for candidate in _canonical_atoms(entry)
    }
    outer = canonical_by_syntax[
        "bool(jnp.isfinite(value) & jnp.all(jnp.isfinite(gradient)) & "
        "jnp.all(jnp.isfinite(hessian)))"
    ]
    value = canonical_by_syntax["jnp.isfinite(value)"]
    gradient_all = canonical_by_syntax["jnp.all(jnp.isfinite(gradient))"]
    gradient_elements = canonical_by_syntax["jnp.isfinite(gradient)"]
    hessian_all = canonical_by_syntax["jnp.all(jnp.isfinite(hessian))"]
    hessian_elements = canonical_by_syntax["jnp.isfinite(hessian)"]
    if canonical == outer:
        prerequisites = (AtomPrerequisite(value, False),)
        logic = AtomDependencyLogic.ALL_OF
        rationale = "one false finite child makes the source all-of expression false"
    elif canonical == value:
        prerequisites = (AtomPrerequisite(outer, False),)
        logic = AtomDependencyLogic.TARGET_IMPLIES_PREREQUISITES
        rationale = "a non-finite objective makes the enclosing source boolean false"
    elif canonical == gradient_all:
        prerequisites = (
            AtomPrerequisite(gradient_elements, False),
            AtomPrerequisite(outer, False),
        )
        logic = AtomDependencyLogic.EQUIVALENT
        rationale = (
            "the retained gradient has one non-finite element while the other "
            "finite children stay at baseline"
        )
    elif canonical == gradient_elements:
        prerequisites = (
            AtomPrerequisite(gradient_all, False),
            AtomPrerequisite(outer, False),
        )
        logic = AtomDependencyLogic.EQUIVALENT
        rationale = (
            "the elementwise gradient predicate, its all-reducer, and the outer "
            "boolean share this real failure fixture"
        )
    elif canonical == hessian_all:
        prerequisites = (
            AtomPrerequisite(hessian_elements, False),
            AtomPrerequisite(outer, False),
        )
        logic = AtomDependencyLogic.EQUIVALENT
        rationale = (
            "the retained Hessian has one non-finite element while the other "
            "finite children stay at baseline"
        )
    elif canonical == hessian_elements:
        prerequisites = (
            AtomPrerequisite(hessian_all, False),
            AtomPrerequisite(outer, False),
        )
        logic = AtomDependencyLogic.EQUIVALENT
        rationale = (
            "the elementwise Hessian predicate, its all-reducer, and the outer "
            "boolean share this real failure fixture"
        )
    else:  # pragma: no cover - guarded by the committed source manifest
        raise AssertionError(f"unclassified MAP finite atom {syntax!r}")
    return AtomRelation(
        kind=kind,
        baselines=_atom_baselines(entry),
        target_outcome=False,
        canonical_atom_id=canonical_atom_id,
        prerequisites=prerequisites,
        logic=logic,
        rationale=rationale,
    )


def _atom_relation(entry: GateEntry, atom_id: str) -> AtomRelation:
    canonical = source_alias_canonical(entry, atom_id)
    alias = canonical != atom_id
    kind = AtomRelationKind.ALIAS if alias else AtomRelationKind.INDEPENDENT
    canonical_atom_id = canonical if alias else None
    if entry.gate_id == "MAP:map_estimate:finite-derivative-payload":
        return _map_finite_relation(
            entry,
            atom_id,
            kind=AtomRelationKind.ALIAS if alias else AtomRelationKind.DEPENDENT,
            canonical_atom_id=canonical_atom_id,
        )
    baselines = _atom_baselines(entry)
    baseline = {item.atom_id: item.outcome for item in baselines}[canonical]
    if (
        entry.gate_id == "COUPLING:_condition_number:finite-spectrum"
        and _ATOM_SYNTAX[canonical] == "np.isfinite(smallest)"
    ):
        largest = next(
            candidate
            for candidate in _canonical_atoms(entry)
            if _ATOM_SYNTAX[candidate] == "np.isfinite(largest)"
        )
        return AtomRelation(
            kind=AtomRelationKind.ALIAS if alias else AtomRelationKind.DEPENDENT,
            baselines=baselines,
            target_outcome=False,
            canonical_atom_id=canonical_atom_id,
            prerequisites=(AtomPrerequisite(largest, None),),
            logic=AtomDependencyLogic.SHORT_CIRCUIT,
            rationale=(
                "a non-finite smallest eigenvalue returns before the source "
                "evaluates largest finiteness"
            ),
        )
    return AtomRelation(
        kind=kind,
        baselines=baselines,
        target_outcome=not baseline,
        canonical_atom_id=canonical_atom_id,
    )


def _observation(
    *,
    entry: GateEntry,
    point: ThresholdPoint,
    observed_side: GateSide,
    point_key: str,
    point_value: object,
    threshold: object,
    inputs: dict[str, object],
    direct_input_keys: tuple[str, ...],
    direct_return_keys: tuple[str, ...],
    direct_calls: tuple[str, ...],
    oracle_checks: tuple[object, ...],
    evaluations: tuple[object, ...] = (),
    atom_id: str | None = None,
    atom_evidence: tuple[AtomEvidence, ...] = (),
    sibling_premises: tuple[object, ...] = (),
    axis_key: str | None = None,
    dtype: str | None = "float64",
    defensive: bool = False,
    notes: tuple[str, ...] = (),
) -> RawObservation:
    """Build one observation from values retained by the direct runner."""
    linked_axis_key = point_key if axis_key is None else axis_key
    return RawObservation(
        observed_side=observed_side,
        realized_point=realized_point(
            entry=entry,
            point=point,
            quantity=entry.quantity,
            input_key=point_key,
            value=point_value,
            threshold=threshold,
            dtype=dtype,
            axis_input_key=linked_axis_key,
            axis_value=inputs[linked_axis_key],
        ),
        realized_inputs=inputs,
        direct_input_keys=direct_input_keys,
        direct_return_keys=direct_return_keys,
        direct_calls=direct_calls,
        oracle_checks=oracle_checks,
        evaluations=evaluations,
        isolated_atom=atom_id,
        atom_evidence=atom_evidence,
        sibling_premises=sibling_premises,
        defensive=defensive,
        notes=notes,
    )


def _raw_equal(actual: object, expected: object) -> bool:
    first = np.asarray(actual)
    second = np.asarray(expected)
    return bool(
        first.shape == second.shape
        and first.dtype == second.dtype
        and np.array_equal(first, second)
    )


def _atom_values(
    entry: GateEntry,
    inputs: dict[str, object],
    selected_atom_id: str,
) -> dict[str, tuple[object, object, AtomReducer, tuple[str, ...]]]:
    """Evaluate source predicates and independent equivalents on retained values."""
    gate_id = entry.gate_id
    values: dict[str, tuple[object, object, AtomReducer, tuple[str, ...]]] = {}
    for atom_id in entry.conjunction_atom_ids:
        syntax = _ATOM_SYNTAX[atom_id]
        if gate_id.startswith("COUPLING:_classify_correlation"):
            value = float(inputs["value"])
            floor = float(inputs["floor"])
            if syntax == "np.isfinite(value)":
                actual = np.isfinite(value)
                expected = math.isfinite(value)
                keys = ("value",)
            elif syntax == "np.isfinite(floor)":
                actual = np.isfinite(floor)
                expected = math.isfinite(floor)
                keys = ("floor",)
            elif syntax == "value <= floor":
                actual = value <= floor
                expected = not (value > floor)
                keys = ("value", "floor")
            elif syntax == "value >= 1.0 - floor":
                actual = value >= 1.0 - floor
                expected = not (value < math.fsum((1.0, -floor)))
                keys = ("value", "floor")
            else:  # pragma: no cover - protected by the manifest
                raise AssertionError(f"unknown correlation atom {syntax!r}")
            reducer = AtomReducer.SCALAR
        elif gate_id.startswith("COUPLING:_condition_number"):
            smallest = float(inputs["smallest"])
            largest = float(inputs["largest"])
            selected_syntax = _ATOM_SYNTAX[
                source_alias_canonical(entry, selected_atom_id)
            ]
            if syntax == "np.isfinite(smallest)":
                actual = np.isfinite(smallest)
                expected = math.isfinite(smallest)
                keys = ("smallest",)
            elif syntax == "np.isfinite(largest)":
                if selected_syntax == "np.isfinite(smallest)":
                    actual = None
                    expected = None
                    reducer = AtomReducer.NOT_EVALUATED
                else:
                    actual = np.isfinite(largest)
                    expected = math.isfinite(largest)
                keys = ("largest",)
            elif syntax == "smallest <= 0.0":
                actual = smallest <= 0.0
                expected = not (smallest > 0.0)
                keys = ("smallest",)
            else:  # pragma: no cover - protected by the manifest
                raise AssertionError(f"unknown condition atom {syntax!r}")
            if actual is not None:
                reducer = AtomReducer.SCALAR
        elif gate_id == "MAP:map_estimate:finite-derivative-payload":
            objective = float(inputs["objective"])
            gradient = jnp.asarray(inputs["actual_gradient"])
            hessian = jnp.asarray(inputs["actual_hessian"])
            if syntax.startswith("bool("):
                actual = bool(
                    jnp.isfinite(objective)
                    & jnp.all(jnp.isfinite(gradient))
                    & jnp.all(jnp.isfinite(hessian))
                )
                expected = bool(
                    math.isfinite(objective)
                    and np.all(np.isfinite(np.asarray(gradient)))
                    and np.all(np.isfinite(np.asarray(hessian)))
                )
                keys = ("objective", "actual_gradient", "actual_hessian")
                reducer = AtomReducer.SCALAR
            elif syntax == "jnp.isfinite(value)":
                actual = jnp.isfinite(objective)
                expected = np.isfinite(objective)
                keys = ("objective",)
                reducer = AtomReducer.SCALAR
            elif syntax == "jnp.all(jnp.isfinite(gradient))":
                actual = jnp.all(jnp.isfinite(gradient))
                expected = np.all(np.isfinite(np.asarray(gradient)))
                keys = ("actual_gradient",)
                reducer = AtomReducer.SCALAR
            elif syntax == "jnp.isfinite(gradient)":
                actual = jnp.isfinite(gradient)
                expected = np.isfinite(np.asarray(gradient))
                keys = ("actual_gradient",)
                reducer = AtomReducer.ALL_ELEMENTS
            elif syntax == "jnp.all(jnp.isfinite(hessian))":
                actual = jnp.all(jnp.isfinite(hessian))
                expected = np.all(np.isfinite(np.asarray(hessian)))
                keys = ("actual_hessian",)
                reducer = AtomReducer.SCALAR
            elif syntax == "jnp.isfinite(hessian)":
                actual = jnp.isfinite(hessian)
                expected = np.isfinite(np.asarray(hessian))
                keys = ("actual_hessian",)
                reducer = AtomReducer.ALL_ELEMENTS
            else:  # pragma: no cover - protected by the manifest
                raise AssertionError(f"unknown MAP finite atom {syntax!r}")
        elif gate_id == "MAP:map_estimate:stationarity-floor":
            gradient_norm = float(inputs["gradient_norm"])
            hessian_norm = float(inputs["hessian_norm"])
            dimension = np.asarray(inputs["candidate"]).size
            threshold = math.sqrt(np.finfo(float).eps) * dimension * hessian_norm
            actual = gradient_norm > threshold
            expected = not (gradient_norm <= threshold)
            keys = ("gradient_norm", "hessian_norm", "candidate")
            reducer = AtomReducer.SCALAR
        else:  # pragma: no cover - only atom-bearing gates call this helper
            raise AssertionError(f"no atom evaluator for {gate_id}")
        values[atom_id] = (actual, expected, reducer, keys)
    return values


def _atom_audit(
    entry: GateEntry,
    inputs: dict[str, object],
    oracle_name: str,
    selected_atom_id: str,
) -> tuple[tuple[AtomEvidence, ...], tuple[object, ...]]:
    values = _atom_values(entry, inputs, selected_atom_id)
    evidence: list[AtomEvidence] = []
    checks: list[object] = []
    for atom_id in entry.conjunction_atom_ids:
        actual, expected, reducer, keys = values[atom_id]
        raw = np.asarray(actual)
        if reducer is AtomReducer.NOT_EVALUATED:
            truth = None
        elif reducer is AtomReducer.SCALAR:
            truth = bool(raw.item())
        elif reducer is AtomReducer.ALL_ELEMENTS:
            truth = bool(np.all(raw))
        else:  # pragma: no cover - no diagnose/graph ANY_ELEMENT source atom
            truth = bool(np.any(raw))
        evidence.append(
            AtomEvidence(
                atom_id=atom_id,
                raw_actual=actual,
                truth=truth,
                reducer=reducer,
                realized_keys=keys,
                oracle=oracle_name,
            )
        )
        checks.append(
            oracle_check(
                oracle=oracle_name,
                actual=actual,
                expected=expected,
            )
        )
    return tuple(evidence), tuple(checks)


def _freeze(entry: GateEntry, spec: _SuiteSpec) -> BoundarySuite:
    non_unit = (
        _NON_UNIT
        if entry.fixture_scale_policy is FixtureScalePolicy.NON_UNIT_REQUIRED
        else None
    )
    grid_cases = make_grid_cases(
        entry=entry,
        points=spec.points,
        fixture_family=spec.family,
        execution_class=spec.execution_class,
        topology=spec.topology,
        direct_methods=spec.direct_methods,
        independent_oracles=spec.independent_oracles,
        runner=spec.grid_runner,
        non_unit_scale=non_unit,
    )
    atom_cases = []
    atom_case_ids: dict[str, str] = {}
    for atom_id in entry.conjunction_atom_ids:
        case = make_atom_case(
            entry=entry,
            atom_id=atom_id,
            relation=_atom_relation(entry, atom_id),
            point=ThresholdPoint(
                PointRole.EXTREME,
                f"isolated source atom {_ATOM_SYNTAX[atom_id]}",
                "only this source premise is false",
                GateSide.REFUSED,
            ),
            fixture_family=spec.family,
            execution_class=spec.execution_class,
            topology=spec.topology,
            direct_methods=spec.direct_methods,
            independent_oracles=spec.independent_oracles,
            runner=spec.atom_runner(atom_id),
            non_unit_scale=non_unit,
        )
        atom_cases.append(case)
        atom_case_ids[atom_id] = case.case_id
    tighten_role, loosen_role = _MUTATION_WITNESS_ROLES[entry.gate_id]
    admitted = next(
        case
        for case in grid_cases
        if case.threshold_point.expected_side is GateSide.ADMITTED
        and case.threshold_point.role is tighten_role
    )
    refused = next(
        case
        for case in grid_cases
        if case.threshold_point.expected_side is GateSide.REFUSED
        and case.threshold_point.role is loosen_role
    )
    return freeze_suite(
        gate_id=entry.gate_id,
        fixture_family=spec.family,
        execution_class=spec.execution_class,
        topology=spec.topology,
        cases=(*grid_cases, *atom_cases),
        atom_case_ids=atom_case_ids,
        tighten_case_id=admitted.case_id,
        loosen_case_id=refused.case_id,
        ambiguities=spec.ambiguities,
    )


def _float_neighbour(
    point: ThresholdPoint,
    threshold: float,
    *,
    very_low: float,
    very_high: float,
    extreme: float,
) -> float:
    role = point.role
    if role is PointRole.VERY_LOW:
        return very_low
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
        return very_high
    if role is PointRole.EXTREME:
        return extreme
    raise AssertionError(f"{role.value} is not a float-neighbour role")


def _classification_runner(entry: GateEntry, atom_id: str | None = None) -> Runner:
    gate_id = entry.gate_id
    oracle_name = "literal finite closed-noise-floor classification"

    def run(point: ThresholdPoint) -> RawObservation:
        value = 0.4
        floor = 0.01
        if gate_id.endswith("value-finite"):
            if point.role is PointRole.CAPABILITY_LOW:
                value = 0.2
            elif point.role is PointRole.CAPABILITY_HIGH:
                value = math.inf
            elif atom_id is not None or point.role in {
                PointRole.INVALID_CAPABILITY,
                PointRole.EXTREME,
            }:
                value = math.nan
            if point.role is PointRole.EXTREME:
                value = -math.inf
        elif gate_id.endswith("floor-finite"):
            if point.role is PointRole.CAPABILITY_LOW:
                floor = 0.001
            elif point.role is PointRole.CAPABILITY_HIGH:
                floor = math.inf
            elif atom_id is not None or point.role in {
                PointRole.INVALID_CAPABILITY,
                PointRole.EXTREME,
            }:
                floor = math.nan
            if point.role is PointRole.EXTREME:
                floor = -math.inf
        elif gate_id.endswith("lower-noise-floor"):
            floor = 0.25
            value = (
                floor
                if atom_id is not None
                else _float_neighbour(
                    point,
                    floor,
                    very_low=0.0,
                    very_high=0.7,
                    extreme=-1.3,
                )
            )
        elif gate_id.endswith("upper-noise-floor"):
            floor = 0.25
            upper = 1.0 - floor
            value = (
                upper
                if atom_id is not None
                else _float_neighbour(
                    point,
                    upper,
                    very_low=0.4,
                    very_high=0.9,
                    extreme=1.3,
                )
            )
        else:
            raise AssertionError(f"unknown correlation gate {gate_id}")

        result = coupling_module._classify_correlation(
            value,
            floor=floor,
            n_correlations=2,
        )
        result_kind = "measured" if isinstance(result, Measured) else "refused"
        expected_kind = oracles.classify_correlation_side(value, floor)
        observed_side = (
            GateSide.ADMITTED if result_kind == "measured" else GateSide.REFUSED
        )
        point_key = "floor" if gate_id.endswith("floor-finite") else "value"
        if gate_id.endswith("lower-noise-floor"):
            threshold: object = floor
        elif gate_id.endswith("upper-noise-floor"):
            threshold = 1.0 - floor
        else:
            threshold = "finite-domain"

        siblings: tuple[object, ...] = ()
        inputs: dict[str, object] = {
            "value": value,
            "floor": floor,
            "n_correlations": 2,
            "verdict": result_kind,
        }
        atom_evidence: tuple[AtomEvidence, ...] = ()
        if atom_id is not None:
            atom_evidence, atom_checks = _atom_audit(
                entry, inputs, oracle_name, atom_id
            )
            siblings = (*siblings, *atom_checks)
        return _observation(
            entry=entry,
            point=point,
            observed_side=observed_side,
            point_key=point_key,
            point_value=inputs[point_key],
            threshold=threshold,
            inputs=inputs,
            direct_input_keys=("value", "floor", "n_correlations"),
            direct_return_keys=("verdict",),
            direct_calls=("_classify_correlation",),
            oracle_checks=(
                oracle_check(
                    oracle=oracle_name,
                    actual=result_kind,
                    expected=expected_kind,
                ),
            ),
            atom_id=atom_id,
            atom_evidence=atom_evidence,
            sibling_premises=siblings,
            axis_key=point_key,
            dtype=None if "finite" in gate_id else "float64",
        )

    return run


def _condition_runner(entry: GateEntry, atom_id: str | None = None) -> Runner:
    finite_gate = entry.gate_id.endswith("finite-spectrum")
    oracle_name = "analytic symmetric two-point spectrum"

    def run(point: ThresholdPoint) -> RawObservation:
        matrix = np.diag(np.array([1.3, 2.4]))
        if finite_gate:
            maximum = np.finfo(np.float64).max
            if atom_id is not None:
                syntax = _ATOM_SYNTAX[source_alias_canonical(entry, atom_id)]
                matrix = (
                    np.full((2, 2), maximum)
                    if syntax == "np.isfinite(largest)"
                    else np.array(
                        [[maximum, maximum], [maximum, -maximum]]
                    )
                )
            elif point.role is PointRole.INVALID_CAPABILITY:
                matrix = np.array([[maximum, maximum], [maximum, -maximum]])
            elif point.role is PointRole.EXTREME:
                # Both the invalid-capability and extreme matrices overflow
                # LAPACK's eigenvalue recurrence; flipping the sign realizes
                # a different boundary matrix instead of reusing the same
                # production input across two axis positions.
                matrix = np.array([[-maximum, -maximum], [-maximum, maximum]])
            elif point.role is PointRole.CAPABILITY_HIGH:
                matrix = np.full((2, 2), maximum)
            elif point.role is PointRole.CAPABILITY_LOW:
                matrix = np.diag(np.array([0.75, 1.5]))
            elif point.role is not PointRole.VALID_CAPABILITY:
                raise AssertionError(f"unexpected finite-spectrum role {point.role}")
        else:
            role = point.role
            if atom_id is not None:
                # Isolate the registered comparison on a moderate matrix whose
                # LAPACK eigenvalues are bit-identical to the exact Decimal
                # ones; the extreme-magnitude grid matrix rounds differently.
                smallest = -1.3
            elif role is PointRole.VERY_LOW:
                smallest = -1.3
            elif role is PointRole.EXACT:
                smallest = 0.0
            elif role is PointRole.ULP_MISMATCH:
                smallest = float(np.nextafter(0.0, -math.inf))
            elif role is PointRole.SUBNORMAL_MISMATCH:
                smallest = float(np.nextafter(0.0, math.inf))
            elif role is PointRole.MATERIAL_MISMATCH:
                smallest = 1e-6
            elif role is PointRole.VERY_HIGH:
                smallest = 2.4
            elif role is PointRole.EXTREME:
                smallest = -np.finfo(float).max
            else:
                raise AssertionError(f"unexpected positive-spectrum role {role}")
            largest = (
                2.0 * smallest
                if role is PointRole.SUBNORMAL_MISMATCH
                else 3.1
            )
            if role is PointRole.EXTREME:
                matrix = np.array(
                    [
                        [-np.finfo(float).max, -np.finfo(float).max],
                        [-np.finfo(float).max, np.finfo(float).max],
                    ]
                )
            else:
                matrix = np.diag(np.array([smallest, largest]))

        captured: dict[str, np.ndarray] = {}
        original_eigvalsh = coupling_module.np.linalg.eigvalsh

        def capture_eigvalsh(value: np.ndarray) -> np.ndarray:
            spectrum = np.asarray(original_eigvalsh(value))
            captured["spectrum"] = spectrum
            return spectrum

        with patch.object(
            coupling_module.np.linalg,
            "eigvalsh",
            side_effect=capture_eigvalsh,
        ):
            try:
                condition = coupling_module._condition_number(matrix)
            except ZeroDivisionError:
                # A loosened positive-spectrum gate reaches the exact-zero
                # division.  Retain that production consequence as numerical
                # failure evidence instead of turning it into a harness crash.
                condition = math.nan
        spectrum = captured["spectrum"]
        smallest = float(spectrum[0])
        largest = float(spectrum[-1])
        expected_spectrum = oracles.exact_symmetric_two_by_two_eigenvalues(matrix)
        expected_smallest = float(expected_spectrum[0])
        expected_largest = float(expected_spectrum[-1])
        if not (
            math.isfinite(expected_smallest) and math.isfinite(expected_largest)
        ):
            expected_condition = math.nan
        elif expected_smallest <= 0.0:
            expected_condition = math.inf
        else:
            expected_condition = expected_largest / expected_smallest
        condition_matches = (
            math.isnan(condition)
            if math.isnan(expected_condition)
            else condition == expected_condition
        ) and np.array_equal(spectrum, expected_spectrum, equal_nan=True)
        if finite_gate:
            observed_side = (
                GateSide.REFUSED if math.isnan(condition) else GateSide.ADMITTED
            )
            point_key = "smallest" if not math.isfinite(smallest) else "largest"
            threshold: object = "finite-domain"
            dtype = None
        else:
            positive_predicate = smallest <= 0.0
            observed_side = (
                GateSide.REFUSED if positive_predicate else GateSide.ADMITTED
            )
            point_key = "smallest"
            threshold = 0.0
            dtype = "float64"

        siblings: tuple[object, ...] = ()
        inputs: dict[str, object] = {
            "matrix": matrix,
            "spectrum": spectrum,
            "smallest": smallest,
            "largest": largest,
            "condition": condition,
        }
        atom_evidence: tuple[AtomEvidence, ...] = ()
        if atom_id is not None:
            atom_evidence, atom_checks = _atom_audit(
                entry, inputs, oracle_name, atom_id
            )
            siblings = (*siblings, *atom_checks)
        return _observation(
            entry=entry,
            point=point,
            observed_side=observed_side,
            point_key=point_key,
            point_value=inputs[point_key],
            threshold=threshold,
            inputs=inputs,
            direct_input_keys=("matrix",),
            direct_return_keys=("spectrum", "smallest", "largest", "condition"),
            direct_calls=("_condition_number",),
            oracle_checks=(
                oracle_check(
                    oracle=oracle_name,
                    actual=condition_matches,
                    expected=True,
                ),
            ),
            atom_id=atom_id,
            atom_evidence=atom_evidence,
            sibling_premises=siblings,
            axis_key="matrix",
            dtype=dtype,
        )

    return run


def _coupling_graph() -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    prior_std = np.array([1.3, 2.4])
    precision = np.array([[2.4, 0.3], [0.3, 1.7]])
    likelihood = precision - np.diag(1.0 / prior_std**2)
    design = np.linalg.cholesky(likelihood).T

    def model() -> None:
        first = sample("first", lambda: dist.Normal(0.0, prior_std[0]))
        second = sample("second", lambda: dist.Normal(0.0, prior_std[1]))
        prediction = det(
            "prediction",
            lambda x, y: jnp.asarray(design) @ jnp.stack((x, y)),
            first,
            second,
            linear_in=("first", "second"),
        )
        observe(
            "data",
            lambda mu: dist.Normal(mu, 1.0).to_event(1),
            prediction,
            obs=jnp.asarray(observation),
        )

    observation = np.zeros(2, dtype=np.float64)
    return trace(model), prior_std, design, observation


def _block_coupling_runner(entry: GateEntry) -> Runner:
    oracle_name = "analytic within-block inertia and covariance-route CCA"
    target_first = entry.gate_id.endswith("f-xx-spd")

    def run(point: ThresholdPoint) -> RawObservation:
        target_diagonal = {
            PointRole.CAPABILITY_LOW: 0.75,
            PointRole.VALID_CAPABILITY: float(
                np.nextafter(np.finfo(float).tiny, 0.0)
            ),
            PointRole.INVALID_CAPABILITY: 0.0,
            PointRole.CAPABILITY_HIGH: -1.3,
            PointRole.EXTREME: -np.finfo(float).max,
        }[point.role]
        first_diagonal = target_diagonal if target_first else 2.4
        second_diagonal = 1.7 if target_first else target_diagonal
        cross = 0.0
        precision = np.array(
            [[first_diagonal, cross], [cross, second_diagonal]],
            dtype=np.float64,
        )
        fisher_result = SimpleNamespace(
            values=precision,
            spans=((0, 1), (1, 2)),
        )
        result: object
        with jax.enable_x64(True):
            graph, prior_std, design, observation = _coupling_graph()
            try:
                with patch.object(
                    coupling_module,
                    "fisher_information",
                    return_value=fisher_result,
                ):
                    result = coupling_module.block_coupling(
                        graph,
                        "first",
                        "second",
                        at={"first": jnp.array(0.0), "second": jnp.array(0.0)},
                    )
            except GraphError as error:
                result = error
        should_admit = target_diagonal > 0.0
        admitted = not isinstance(result, GraphError)
        if admitted and should_admit:
            correlations = oracles.covariance_canonical_correlations(precision, 1)
            actual = float(result.canonical_correlations[0])
            expected = float(correlations[0])
            evaluations = (
                numerical_evaluation(
                    method="block_coupling",
                    oracle=oracle_name,
                    actual=actual,
                    oracle_value=expected,
                ),
            )
            result_kind = "report"
        else:
            evaluations = ()
            result_kind = "report" if admitted else "graph-error"
        inputs: dict[str, object] = {
            "graph_latents": ("first", "second"),
            "graph_prior_std": prior_std,
            "graph_design": design,
            "graph_observation": observation,
            "first": ("first",),
            "second": ("second",),
            "at": (0.0, 0.0),
            "controlled_precision": precision,
            "fisher_spans": fisher_result.spans,
            "result_kind": result_kind,
        }
        return _observation(
            entry=entry,
            point=point,
            observed_side=(GateSide.ADMITTED if admitted else GateSide.REFUSED),
            point_key="controlled_precision",
            point_value=precision,
            threshold="strict-SPD-domain",
            inputs=inputs,
            direct_input_keys=(
                "graph_latents",
                "graph_prior_std",
                "graph_design",
                "graph_observation",
                "first",
                "second",
                "at",
                "controlled_precision",
                "fisher_spans",
            ),
            direct_return_keys=("result_kind",),
            direct_calls=("block_coupling",),
            oracle_checks=(
                oracle_check(
                    oracle=oracle_name,
                    actual=admitted,
                    expected=should_admit,
                ),
            ),
            evaluations=evaluations,
            axis_key="controlled_precision",
            dtype=None,
            defensive=True,
            notes=("controlled fisher-information return; Cholesky remains real",),
        )

    return run


def _names_runner(entry: GateEntry) -> Runner:
    oracle_name = "collections.Counter duplicate multiplicity"

    def run(point: ThresholdPoint) -> RawObservation:
        role = point.role
        if role is PointRole.VERY_LOW:
            values: tuple[str, ...] | list[str] = []
        elif role is PointRole.BELOW_INTEGER:
            values = ()
        elif role is PointRole.AT:
            values = ("alpha",)
        elif role is PointRole.ABOVE_INTEGER:
            values = ("alpha", "alpha")
        elif role is PointRole.VERY_HIGH:
            values = ("alpha",) * 17
        elif role is PointRole.EXTREME:
            values = ("alpha", "beta", "alpha", "β", "β")
        else:
            raise AssertionError(f"unexpected multiplicity role {role}")
        production_counts: list[int] = []

        class _TrackedNames(tuple):
            def count(self, value: object) -> int:
                result = super().count(value)
                production_counts.append(result)
                return result

        try:
            with patch.object(
                reduction_module,
                "tuple",
                new=_TrackedNames,
                create=True,
            ):
                returned = reduction_module._names(values, argument="nodes")
        except GraphError:
            returned = None
        maximum = max(production_counts, default=0)
        expected_maximum = max(Counter(values).values(), default=0)
        admitted = returned is not None
        expected_admitted = expected_maximum <= 1
        inputs: dict[str, object] = {
            "values": values,
            "argument": "nodes",
            "maximum_multiplicity": maximum,
            "result_kind": "tuple" if admitted else "graph-error",
        }
        return _observation(
            entry=entry,
            point=point,
            observed_side=(GateSide.ADMITTED if admitted else GateSide.REFUSED),
            point_key="maximum_multiplicity",
            point_value=maximum,
            threshold=1,
            inputs=inputs,
            direct_input_keys=("values", "argument"),
            direct_return_keys=("maximum_multiplicity", "result_kind"),
            direct_calls=("_names",),
            oracle_checks=(
                oracle_check(
                    oracle=oracle_name,
                    actual=admitted,
                    expected=expected_admitted,
                ),
                oracle_check(
                    oracle=oracle_name,
                    actual=maximum,
                    expected=expected_maximum,
                ),
            ),
            axis_key="values",
            dtype=None,
        )

    return run


def _positive_spectrum_points() -> tuple[ThresholdPoint, ...]:
    return (
        ThresholdPoint(
            PointRole.VERY_LOW,
            "negative ordinary",
            "axis-low",
            GateSide.REFUSED,
        ),
        ThresholdPoint(PointRole.EXACT, "zero", "zero", GateSide.REFUSED),
        ThresholdPoint(
            PointRole.ULP_MISMATCH,
            "negative min-subnormal",
            "one ULP",
            GateSide.REFUSED,
        ),
        ThresholdPoint(
            PointRole.SUBNORMAL_MISMATCH,
            "positive min-subnormal",
            "minimum subnormal",
            GateSide.ADMITTED,
        ),
        ThresholdPoint(
            PointRole.MATERIAL_MISMATCH,
            "positive material",
            "1e-6",
            GateSide.ADMITTED,
        ),
        ThresholdPoint(
            PointRole.VERY_HIGH,
            "positive ordinary",
            "axis-high",
            GateSide.ADMITTED,
        ),
        ThresholdPoint(
            PointRole.EXTREME,
            "negative maximum",
            "extreme",
            GateSide.REFUSED,
        ),
    )


def _coupling_specs() -> list[tuple[GateEntry, _SuiteSpec]]:
    specs: list[tuple[GateEntry, _SuiteSpec]] = []
    for gate_id in (
        "COUPLING:_classify_correlation:value-finite",
        "COUPLING:_classify_correlation:floor-finite",
    ):
        entry = _ENTRIES[gate_id]
        specs.append(
            (
                entry,
                _SuiteSpec(
                    FixtureFamily.COUPLING,
                    ExecutionClass.VALIDATION_ONLY,
                    BoundaryTopology.CAPABILITY,
                    ("_classify_correlation",),
                    ("literal finite closed-noise-floor classification",),
                    capability_grid(),
                    _classification_runner(entry),
                    lambda atom_id, entry=entry: _classification_runner(entry, atom_id),
                ),
            )
        )
    lower = _ENTRIES["COUPLING:_classify_correlation:lower-noise-floor"]
    specs.append(
        (
            lower,
            _SuiteSpec(
                FixtureFamily.COUPLING,
                ExecutionClass.VALIDATION_ONLY,
                BoundaryTopology.FLOAT,
                ("_classify_correlation",),
                ("literal finite closed-noise-floor classification",),
                float_grid(
                    below=GateSide.REFUSED,
                    at=GateSide.REFUSED,
                    above=GateSide.ADMITTED,
                    very_low=GateSide.REFUSED,
                    very_high=GateSide.ADMITTED,
                    extreme=GateSide.REFUSED,
                    threshold="floor",
                ),
                _classification_runner(lower),
                lambda atom_id: _classification_runner(lower, atom_id),
            ),
        )
    )
    upper = _ENTRIES["COUPLING:_classify_correlation:upper-noise-floor"]
    specs.append(
        (
            upper,
            _SuiteSpec(
                FixtureFamily.COUPLING,
                ExecutionClass.VALIDATION_ONLY,
                BoundaryTopology.FLOAT,
                ("_classify_correlation",),
                ("literal finite closed-noise-floor classification",),
                float_grid(
                    below=GateSide.ADMITTED,
                    at=GateSide.REFUSED,
                    above=GateSide.REFUSED,
                    very_low=GateSide.ADMITTED,
                    very_high=GateSide.REFUSED,
                    extreme=GateSide.REFUSED,
                    threshold="1-floor",
                ),
                _classification_runner(upper),
                lambda atom_id: _classification_runner(upper, atom_id),
            ),
        )
    )
    finite = _ENTRIES["COUPLING:_condition_number:finite-spectrum"]
    specs.append(
        (
            finite,
            _SuiteSpec(
                FixtureFamily.COUPLING,
                ExecutionClass.VALIDATION_ONLY,
                BoundaryTopology.CAPABILITY,
                ("_condition_number",),
                ("analytic symmetric two-point spectrum",),
                capability_grid(),
                _condition_runner(finite),
                lambda atom_id: _condition_runner(finite, atom_id),
            ),
        )
    )
    positive = _ENTRIES["COUPLING:_condition_number:positive-spectrum"]
    specs.append(
        (
            positive,
            _SuiteSpec(
                FixtureFamily.COUPLING,
                ExecutionClass.VALIDATION_ONLY,
                BoundaryTopology.EXACT,
                ("_condition_number",),
                ("analytic symmetric two-point spectrum",),
                _positive_spectrum_points(),
                _condition_runner(positive),
                lambda atom_id: _condition_runner(positive, atom_id),
            ),
        )
    )
    for gate_id in (
        "COUPLING:block_coupling:f-xx-spd",
        "COUPLING:block_coupling:f-tt-spd",
    ):
        block = _ENTRIES[gate_id]
        specs.append(
            (
                block,
                _SuiteSpec(
                    FixtureFamily.COUPLING,
                    ExecutionClass.PAYLOAD_OR_REFUSAL,
                    BoundaryTopology.CAPABILITY,
                    ("block_coupling",),
                    ("analytic within-block inertia and covariance-route CCA",),
                    capability_grid(),
                    _block_coupling_runner(block),
                    lambda _atom_id, block=block: _block_coupling_runner(block),
                ),
            )
        )
    return specs


def _graph_spec() -> tuple[GateEntry, _SuiteSpec]:
    entry = _ENTRIES["GRAPH:_names:duplicate-multiplicity"]
    return (
        entry,
        _SuiteSpec(
            FixtureFamily.GRAPH_DOMAIN_STRUCTURE,
            ExecutionClass.VALIDATION_ONLY,
            BoundaryTopology.INTEGER,
            ("_names",),
            ("collections.Counter duplicate multiplicity",),
            integer_grid(
                below=GateSide.ADMITTED,
                at=GateSide.ADMITTED,
                above=GateSide.REFUSED,
                very_low=GateSide.ADMITTED,
                very_high=GateSide.REFUSED,
                extreme=GateSide.REFUSED,
                threshold="one occurrence",
            ),
            _names_runner(entry),
            lambda _atom_id: _names_runner(entry),
        ),
    )


def _gaussian_graph(centre: np.ndarray, widths: np.ndarray) -> object:
    def model() -> None:
        sample(
            "position",
            lambda: dist.Normal(
                jnp.asarray(centre),
                jnp.asarray(widths),
            ).to_event(1),
        )

    return trace(model)


def _run_map(
    *,
    centre: np.ndarray,
    widths: np.ndarray,
    candidate: np.ndarray,
    quadratic_offset: float | None = None,
    linear_terms: np.ndarray | None = None,
    squared_coefficients: np.ndarray | None = None,
) -> tuple[object, dict[str, object]]:
    """Call ``map_estimate`` and retain values from one coherent objective."""
    retained: dict[str, object] = {}
    quadratic_controls = (
        quadratic_offset,
        linear_terms,
        squared_coefficients,
    )
    if any(value is not None for value in quadratic_controls) and not all(
        value is not None for value in quadratic_controls
    ):
        raise ValueError("a quadratic fixture requires offset, linear, and square terms")
    original_log_joint = map_module.log_joint
    original_grad = map_module.jax.grad
    original_hessian = map_module.jax.hessian
    original_eigvalsh = map_module.np.linalg.eigvalsh
    original_max = map_module.jnp.max
    original_norm = map_module.np.linalg.norm
    original_builtin_max = builtins.max
    original_builtin_bool = builtins.bool

    def fixed_newton(_objective: object, _x0: object) -> tuple[object, int, bool]:
        return jnp.asarray(candidate), 3, True

    def capture_log_joint(graph: object, values: object) -> object:
        if quadratic_offset is None:
            result = original_log_joint(graph, values)
        else:
            vector = jnp.ravel(jnp.asarray(values["position"], dtype=jnp.float64))
            objective = jnp.asarray(quadratic_offset, dtype=jnp.float64)
            for term in np.asarray(linear_terms):
                objective = objective + jnp.vdot(jnp.asarray(term), vector)
            for index, coefficient in enumerate(
                np.asarray(squared_coefficients)
            ):
                objective = (
                    objective
                    + jnp.asarray(coefficient) * vector[index] * vector[index]
                )
            result = -objective
        try:
            retained["actual_objective"] = float(-np.asarray(result))
        except (TypeError, ValueError):
            pass
        return result

    def capture_grad(function: object) -> object:
        real_gradient = original_grad(function)

        def evaluate(value: object) -> object:
            result = real_gradient(value)
            retained["actual_gradient"] = np.asarray(result)
            return result

        return evaluate

    def capture_hessian(function: object) -> object:
        real_hessian = original_hessian(function)

        def evaluate(value: object) -> object:
            result = real_hessian(value)
            retained["actual_hessian"] = np.asarray(result)
            return result

        return evaluate

    def capture_eigvalsh(value: np.ndarray) -> np.ndarray:
        result = np.asarray(original_eigvalsh(value))
        retained["actual_spectrum"] = result
        return result

    def capture_max(value: object, *args: object, **kwargs: object) -> object:
        result = original_max(value, *args, **kwargs)
        if "actual_gradient" in retained and "gradient_norm" not in retained:
            try:
                expected = np.abs(np.asarray(retained["actual_gradient"]))
                if np.array_equal(np.asarray(value), expected):
                    retained["gradient_norm"] = float(np.asarray(result))
            except (TypeError, ValueError):
                pass
        return result

    def capture_norm(value: object, *args: object, **kwargs: object) -> object:
        result = original_norm(value, *args, **kwargs)
        if "actual_hessian" in retained and "hessian_norm" not in retained:
            try:
                if np.array_equal(
                    np.asarray(value), np.asarray(retained["actual_hessian"])
                ):
                    retained["hessian_norm"] = float(result)
            except (TypeError, ValueError):
                pass
        return result

    def capture_builtin_max(*values: object) -> object:
        result = original_builtin_max(*values)
        if (
            len(values) == 2
            and isinstance(values[0], float)
            and isinstance(values[1], (float, np.floating))
        ):
            retained["curvature_scale"] = float(result)
            retained["curvature_threshold"] = float(values[1])
        return result

    def capture_builtin_bool(value: object) -> bool:
        result = original_builtin_bool(value)
        retained["finite_predicate"] = result
        return result

    with jax.enable_x64(True):
        graph = _gaussian_graph(centre, widths)
        with (
            patch.object(map_module, "_newton", side_effect=fixed_newton),
            patch.object(map_module, "log_joint", side_effect=capture_log_joint),
            patch.object(map_module.jax, "grad", side_effect=capture_grad),
            patch.object(map_module.jax, "hessian", side_effect=capture_hessian),
            patch.object(
                map_module.np.linalg,
                "eigvalsh",
                side_effect=capture_eigvalsh,
            ),
            patch.object(map_module.jnp, "max", side_effect=capture_max),
            patch.object(map_module.np.linalg, "norm", side_effect=capture_norm),
            patch.object(map_module, "max", new=capture_builtin_max, create=True),
            patch.object(map_module, "bool", new=capture_builtin_bool, create=True),
        ):
            result = map_module.map_estimate(
                graph,
                at={"position": jnp.asarray(centre)},
            )
    retained["result_kind"] = (
        "map-estimate" if isinstance(result, MapEstimate) else "refused"
    )
    return result, retained


def _finite_atom_payload(
    entry: GateEntry,
    atom_id: str,
) -> tuple[str, float, dict[str, object]]:
    syntax = _ATOM_SYNTAX[atom_id]
    maximum = np.finfo(np.float64).max
    controls: dict[str, object] = {
        "quadratic_offset": 0.0,
        "linear_terms": np.zeros((1, 2), dtype=np.float64),
        "squared_coefficients": np.array([0.125, 0.03125]),
    }
    if syntax.startswith("bool(") or "value" in syntax:
        controls["quadratic_offset"] = math.inf
        return "objective", math.inf, controls
    if "gradient" in syntax:
        controls["linear_terms"] = np.array(
            [[maximum, 0.0], [maximum, 0.0]], dtype=np.float64
        )
        return "gradient_component", math.inf, controls
    if "hessian" in syntax:
        controls["squared_coefficients"] = np.array([maximum, 0.03125])
        return "hessian_component", math.inf, controls
    raise AssertionError(f"unclassified MAP finite atom {syntax!r}")


def _map_runner(entry: GateEntry, atom_id: str | None = None) -> Runner:
    oracle_name = "handwritten Gaussian objective derivatives and eigenvalues"
    gate_id = entry.gate_id

    def run(point: ThresholdPoint) -> RawObservation:
        centre = np.zeros(2, dtype=np.float64)
        widths = np.array([2.0, 4.0], dtype=np.float64)
        candidate = centre.copy()
        controls: dict[str, object] = {}

        def set_curvature_fixture(eigenvalues: np.ndarray) -> None:
            nonlocal widths, controls
            values = np.asarray(eigenvalues, dtype=np.float64)
            if np.all(np.isfinite(values)) and np.all(values > 0.0):
                proposed_widths = 1.0 / np.sqrt(values)
                mantissas, _exponents = np.frexp(proposed_widths)
                if np.all(mantissas == 0.5) and np.array_equal(
                    1.0 / proposed_widths**2, values
                ):
                    widths = proposed_widths
                    controls = {}
                    return
            controls = {
                "quadratic_offset": 0.0,
                "linear_terms": np.zeros((1, values.size), dtype=np.float64),
                "squared_coefficients": values / 2.0,
            }

        if gate_id.endswith("finite-derivative-payload"):
            if atom_id is not None:
                point_key, point_value, controls = _finite_atom_payload(entry, atom_id)
            elif point.role is PointRole.CAPABILITY_LOW:
                point_key = "objective"
                point_value = -1.3
                controls = {
                    "quadratic_offset": point_value,
                    "linear_terms": np.zeros((1, 2), dtype=np.float64),
                    "squared_coefficients": np.array([0.125, 0.03125]),
                }
            elif point.role is PointRole.VALID_CAPABILITY:
                point_key = "objective"
                point_value = 0.0
            elif point.role is PointRole.INVALID_CAPABILITY:
                point_key = "objective"
                point_value = math.inf
                controls = {
                    "quadratic_offset": math.inf,
                    "linear_terms": np.zeros((1, 2), dtype=np.float64),
                    "squared_coefficients": np.array([0.125, 0.03125]),
                }
            elif point.role is PointRole.CAPABILITY_HIGH:
                maximum = np.finfo(np.float64).max
                point_key = "gradient_component"
                point_value = math.inf
                controls = {
                    "quadratic_offset": 0.0,
                    "linear_terms": np.array(
                        [[maximum, 0.0], [maximum, 0.0]], dtype=np.float64
                    ),
                    "squared_coefficients": np.array([0.125, 0.03125]),
                }
            elif point.role is PointRole.EXTREME:
                point_key = "hessian_component"
                point_value = math.inf
                controls = {
                    "quadratic_offset": 0.0,
                    "linear_terms": np.zeros((1, 2), dtype=np.float64),
                    "squared_coefficients": np.array(
                        [np.finfo(np.float64).max, 0.03125]
                    ),
                }
            else:
                raise AssertionError(
                    f"unexpected finite-derivative role {point.role}"
                )
            threshold: object = "finite-domain"
        elif gate_id.endswith("stationarity-floor"):
            hessian_norm = 0.25
            threshold = math.sqrt(np.finfo(float).eps) * 2 * hessian_norm
            point_value = _float_neighbour(
                point,
                threshold,
                very_low=0.0,
                very_high=threshold * 2.4,
                extreme=1.3,
            )
            candidate[0] = 4.0 * point_value
            point_key = "gradient_norm"
        elif gate_id.endswith("curvature-scale-clamp"):
            threshold = 1.0
            point_value = _float_neighbour(
                point,
                threshold,
                very_low=0.25,
                very_high=2.4,
                extreme=np.finfo(float).max / 4.0,
            )
            baseline_floor = (
                np.finfo(float).eps * max(abs(point_value), 1.0) * 2
            )
            if point.role is PointRole.AT:
                smallest_fixture = 3.0 * np.finfo(float).eps
            elif point.role is PointRole.BELOW_ULP:
                relaxed_floor = np.finfo(float).eps * abs(point_value) * 2
                smallest_fixture = float(np.nextafter(relaxed_floor, math.inf))
            elif point.expected_side is GateSide.ADMITTED:
                smallest_fixture = float(np.nextafter(baseline_floor, math.inf))
            else:
                smallest_fixture = baseline_floor
            set_curvature_fixture(np.array([smallest_fixture, point_value]))
            point_key = "largest_eigenvalue"
        elif gate_id.endswith("relative-positive-curvature"):
            largest = 4.0
            threshold = np.finfo(float).eps * largest * 2
            point_value = _float_neighbour(
                point,
                threshold,
                very_low=-1.3,
                very_high=1e-6,
                extreme=-np.finfo(float).max,
            )
            set_curvature_fixture(np.array([point_value, largest]))
            point_key = "smallest_eigenvalue"
        elif gate_id.endswith("absolute-curvature"):
            centre = np.zeros(1, dtype=np.float64)
            widths = np.array([2.0], dtype=np.float64)
            candidate = centre.copy()
            threshold = math.sqrt(np.finfo(float).eps)
            point_value = _float_neighbour(
                point,
                threshold,
                very_low=0.0,
                very_high=1.3,
                extreme=-1.3,
            )
            set_curvature_fixture(np.array([point_value]))
            point_key = "largest_eigenvalue"
        else:
            raise AssertionError(f"unknown MAP gate {gate_id}")

        result, returned = _run_map(
            centre=centre,
            widths=widths,
            candidate=candidate,
            quadratic_offset=controls.get("quadratic_offset"),
            linear_terms=controls.get("linear_terms"),
            squared_coefficients=controls.get("squared_coefficients"),
        )
        gradient = np.asarray(returned["actual_gradient"])
        hessian = np.asarray(returned["actual_hessian"])
        objective = float(returned["actual_objective"])
        spectrum_was_consumed = "actual_spectrum" in returned
        spectrum = (
            np.asarray(returned["actual_spectrum"])
            if spectrum_was_consumed
            else np.asarray([math.nan])
        )
        gradient_norm_was_consumed = "gradient_norm" in returned
        gradient_norm = (
            float(returned["gradient_norm"])
            if gradient_norm_was_consumed
            else float(np.max(np.abs(gradient)))
        )
        hessian_norm_was_consumed = "hessian_norm" in returned
        hessian_norm = (
            float(returned["hessian_norm"]) if hessian_norm_was_consumed else math.nan
        )
        curvature_scale_was_consumed = "curvature_scale" in returned
        curvature_scale = (
            float(returned["curvature_scale"])
            if curvature_scale_was_consumed
            else math.nan
        )
        smallest = float(spectrum[0])
        largest = float(spectrum[-1])
        result_kind = str(returned["result_kind"])

        if controls:
            independent_objective, independent_gradient, independent_hessian = (
                oracles.diagonal_quadratic_payload(
                    candidate,
                    float(controls["quadratic_offset"]),
                    np.asarray(controls["linear_terms"]),
                    np.asarray(controls["squared_coefficients"]),
                )
            )
        else:
            independent_objective, independent_gradient, independent_hessian = (
                oracles.normal_quadratic_payload(candidate, centre, widths)
            )
        independent_spectrum = np.sort(np.diag(independent_hessian))
        independent_gradient_norm = max(
            (abs(float(value)) for value in independent_gradient), default=0.0
        )
        independent_hessian_norm = max(
            (abs(float(value)) for value in np.diag(independent_hessian)),
            default=0.0,
        )

        if gate_id.endswith("finite-derivative-payload"):
            independent_finite_payload = bool(
                math.isfinite(independent_objective)
                and np.all(np.isfinite(independent_gradient))
                and np.all(np.isfinite(independent_hessian))
            )
            production_finite = result_kind == "map-estimate"
            observed_side = (
                GateSide.ADMITTED if production_finite else GateSide.REFUSED
            )
            if atom_id is None and point.role is PointRole.VALID_CAPABILITY:
                point_value = objective
            elif point_key == "gradient_component":
                point_value = float(gradient[0])
            elif point_key == "hessian_component":
                point_value = float(hessian[0, 0])
            else:
                point_value = objective
            expected_admit = independent_finite_payload
        elif gate_id.endswith("stationarity-floor"):
            threshold = oracles.stationarity_floor(
                independent_hessian,
                candidate.size,
                np.finfo(np.float64).eps,
            )
            point_value = gradient_norm
            expected_admit = independent_gradient_norm <= threshold
            observed_side = (
                GateSide.ADMITTED
                if isinstance(result, MapEstimate)
                else GateSide.REFUSED
            )
        elif gate_id.endswith("curvature-scale-clamp"):
            point_value = largest
            if not curvature_scale_was_consumed:
                raise AssertionError("MAP did not execute its curvature-scale clamp")
            expected_admit = bool(
                independent_spectrum[0]
                > oracles.relative_curvature_floor(
                    independent_spectrum[-1],
                    candidate.size,
                    np.finfo(np.float64).eps,
                )
                and independent_spectrum[-1]
                > math.sqrt(np.finfo(np.float64).eps)
            )
            observed_side = (
                GateSide.ADMITTED if isinstance(result, MapEstimate) else GateSide.REFUSED
            )
        elif gate_id.endswith("relative-positive-curvature"):
            threshold = oracles.relative_curvature_floor(
                independent_spectrum[-1],
                candidate.size,
                np.finfo(np.float64).eps,
            )
            point_value = smallest
            expected_admit = independent_spectrum[0] > threshold
            observed_side = (
                GateSide.ADMITTED
                if isinstance(result, MapEstimate)
                else GateSide.REFUSED
            )
        else:
            threshold = math.sqrt(np.finfo(np.float64).eps)
            point_value = largest
            expected_admit = independent_spectrum[-1] > threshold
            observed_side = (
                GateSide.ADMITTED
                if isinstance(result, MapEstimate)
                else GateSide.REFUSED
            )

        siblings: tuple[object, ...] = ()

        inputs: dict[str, object] = {
            "graph_latents": ("position",),
            "centre": centre,
            "widths": widths,
            "candidate": candidate,
            "objective": objective,
            "actual_gradient": gradient,
            "actual_hessian": hessian,
            "gradient_component": float(gradient.flat[0]),
            "hessian_component": float(hessian.flat[0]),
            "result_kind": result_kind,
        }
        direct_input_keys = ["graph_latents", "centre", "widths", "candidate"]
        for key, value in controls.items():
            inputs[key] = value
            direct_input_keys.append(key)
        return_keys = [
            "objective",
            "actual_gradient",
            "actual_hessian",
            "gradient_component",
            "hessian_component",
            "result_kind",
        ]
        if spectrum_was_consumed:
            inputs["actual_spectrum"] = spectrum
            inputs["smallest_eigenvalue"] = smallest
            inputs["largest_eigenvalue"] = largest
            return_keys.extend(
                ("actual_spectrum", "smallest_eigenvalue", "largest_eigenvalue")
            )
        if gradient_norm_was_consumed:
            inputs["gradient_norm"] = gradient_norm
            return_keys.append("gradient_norm")
        if hessian_norm_was_consumed:
            inputs["hessian_norm"] = hessian_norm
            return_keys.append("hessian_norm")
        if curvature_scale_was_consumed:
            inputs["curvature_scale"] = curvature_scale
            return_keys.append("curvature_scale")
        if point_key in direct_input_keys:
            raise AssertionError("MAP point key must retain a production return")
        checks = (
            oracle_check(
                oracle=oracle_name,
                actual=bool(
                    (math.isnan(objective) and math.isnan(independent_objective))
                    or (math.isinf(objective) and objective == independent_objective)
                    or (
                        math.isfinite(objective)
                        and math.isfinite(independent_objective)
                        and math.isclose(
                            objective,
                            independent_objective,
                            rel_tol=1e-15,
                            abs_tol=0.0,
                        )
                    )
                ),
                expected=True,
            ),
            oracle_check(
                oracle=oracle_name,
                actual=bool(
                    np.allclose(
                        gradient,
                        independent_gradient,
                        rtol=8.0 * np.finfo(np.float64).eps,
                        atol=0.0,
                        equal_nan=True,
                    )
                    and np.allclose(
                        hessian,
                        independent_hessian,
                        rtol=8.0 * np.finfo(np.float64).eps,
                        atol=0.0,
                        equal_nan=True,
                    )
                    and (
                        not spectrum_was_consumed
                        or np.allclose(
                            spectrum,
                            independent_spectrum,
                            rtol=8.0 * np.finfo(np.float64).eps,
                            atol=0.0,
                            equal_nan=True,
                        )
                    )
                    and (
                        not gradient_norm_was_consumed
                        or gradient_norm == independent_gradient_norm
                    )
                    and (
                        not hessian_norm_was_consumed
                        or hessian_norm == independent_hessian_norm
                    )
                ),
                expected=True,
            ),
            oracle_check(
                oracle=oracle_name,
                actual=observed_side
                is (GateSide.ADMITTED if expected_admit else GateSide.REFUSED),
                expected=True,
            ),
        )
        evaluations: tuple[object, ...] = ()
        if isinstance(result, MapEstimate) and math.isfinite(independent_objective):
            evaluations = (
                numerical_evaluation(
                    method="map_estimate",
                    oracle=oracle_name,
                    actual=result.objective,
                    oracle_value=independent_objective,
                ),
            )
        if point_key == "objective" and controls:
            axis_key = "quadratic_offset"
        elif point_key == "gradient_component":
            axis_key = "linear_terms"
        elif point_key in {
            "hessian_component",
            "smallest_eigenvalue",
            "largest_eigenvalue",
        } and controls:
            axis_key = "squared_coefficients"
        elif gate_id.endswith("stationarity-floor"):
            axis_key = "candidate"
        else:
            axis_key = "widths"
        atom_evidence: tuple[AtomEvidence, ...] = ()
        if atom_id is not None:
            atom_evidence, atom_checks = _atom_audit(
                entry, inputs, oracle_name, atom_id
            )
            siblings = (*siblings, *atom_checks)
        return _observation(
            entry=entry,
            point=point,
            observed_side=observed_side,
            point_key=point_key,
            point_value=point_value,
            threshold=threshold,
            inputs=inputs,
            direct_input_keys=tuple(direct_input_keys),
            direct_return_keys=tuple(return_keys),
            direct_calls=("map_estimate",),
            oracle_checks=checks,
            evaluations=evaluations,
            atom_id=atom_id,
            atom_evidence=atom_evidence,
            sibling_premises=siblings,
            axis_key=axis_key,
        )

    return run


def _map_specs() -> list[tuple[GateEntry, _SuiteSpec]]:
    specs: list[tuple[GateEntry, _SuiteSpec]] = []
    finite = _ENTRIES["MAP:map_estimate:finite-derivative-payload"]
    specs.append(
        (
            finite,
            _SuiteSpec(
                FixtureFamily.MAP_DERIVATIVES_CURVATURE,
                ExecutionClass.PAYLOAD_OR_REFUSAL,
                BoundaryTopology.CAPABILITY,
                ("map_estimate",),
                ("handwritten Gaussian objective derivatives and eigenvalues",),
                capability_grid(),
                _map_runner(finite),
                lambda atom_id: _map_runner(finite, atom_id),
            ),
        )
    )
    clamp_points = tuple(
        replace(point, expected_side=GateSide.REFUSED)
        if point.role is PointRole.BELOW_ULP
        else point
        for point in float_grid(
            below=GateSide.ADMITTED,
            at=GateSide.ADMITTED,
            above=GateSide.REFUSED,
            very_low=GateSide.ADMITTED,
            very_high=GateSide.REFUSED,
            extreme=GateSide.REFUSED,
            threshold="unit curvature scale",
        )
    )
    for gate_id, points, execution_class in (
        (
            "MAP:map_estimate:stationarity-floor",
            float_grid(
                below=GateSide.ADMITTED,
                at=GateSide.ADMITTED,
                above=GateSide.REFUSED,
                very_low=GateSide.ADMITTED,
                very_high=GateSide.REFUSED,
                extreme=GateSide.REFUSED,
                threshold="gradient floor",
            ),
            ExecutionClass.PAYLOAD_OR_REFUSAL,
        ),
        (
            "MAP:map_estimate:curvature-scale-clamp",
            clamp_points,
            ExecutionClass.PAYLOAD_OR_REFUSAL,
        ),
        (
            "MAP:map_estimate:relative-positive-curvature",
            float_grid(
                below=GateSide.REFUSED,
                at=GateSide.REFUSED,
                above=GateSide.ADMITTED,
                very_low=GateSide.REFUSED,
                very_high=GateSide.ADMITTED,
                extreme=GateSide.REFUSED,
                threshold="relative curvature floor",
            ),
            ExecutionClass.PAYLOAD_OR_REFUSAL,
        ),
        (
            "MAP:map_estimate:absolute-curvature",
            float_grid(
                below=GateSide.REFUSED,
                at=GateSide.REFUSED,
                above=GateSide.ADMITTED,
                very_low=GateSide.REFUSED,
                very_high=GateSide.ADMITTED,
                extreme=GateSide.REFUSED,
                threshold="absolute curvature floor",
            ),
            ExecutionClass.PAYLOAD_OR_REFUSAL,
        ),
    ):
        entry = _ENTRIES[gate_id]
        specs.append(
            (
                entry,
                _SuiteSpec(
                    FixtureFamily.MAP_DERIVATIVES_CURVATURE,
                    execution_class,
                    BoundaryTopology.FLOAT,
                    ("map_estimate",),
                    ("handwritten Gaussian objective derivatives and eigenvalues",),
                    points,
                    _map_runner(entry),
                    lambda atom_id, entry=entry: _map_runner(entry, atom_id),
                    (
                        (
                            (
                                "Policy ambiguity: the dimensionless clamp 1.0 is "
                                "a unit-scale anchor, not a model-derived optimum. "
                                "The 0.0/2.0 mutations prove downstream MAP "
                                "sensitivity only; they do not prove that 1.0 is "
                                "uniquely correct."
                            ),
                        )
                        if gate_id == "MAP:map_estimate:curvature-scale-clamp"
                        else ()
                    ),
                ),
            )
        )
    return specs


_COUPLING_AND_GRAPH = tuple(
    _freeze(entry, spec) for entry, spec in (*_coupling_specs(), _graph_spec())
)
_MAP_SUITES = tuple(_freeze(entry, spec) for entry, spec in _map_specs())

DIAGNOSE_GRAPH_SUITES: tuple[BoundarySuite, ...] = (
    *_COUPLING_AND_GRAPH,
    *_MAP_SUITES,
)


__all__ = ["DIAGNOSE_GRAPH_SUITES"]
