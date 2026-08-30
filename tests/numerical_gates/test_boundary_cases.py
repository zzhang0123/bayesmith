"""Completeness and execution contract for direct-method boundary grids."""

from __future__ import annotations

import ast
import inspect
from collections import Counter
from pathlib import Path

import pytest

from tests.numerical_gates import oracles
from tests.numerical_gates.boundary_cases import (
    ATOM_CASES,
    BOUNDARY_CASES,
    BOUNDARY_SUITES,
    WITNESS_CASES,
)
from tests.numerical_gates.boundary_contract import REQUIRED_DIRECT_CALLS
from tests.numerical_gates.boundary_core import (
    AtomRelationKind,
    AxisPosition,
    BoundaryExecution,
    BoundaryTopology,
    ExecutionClass,
    FixtureFamily,
    GateSide,
    PointRole,
    source_alias_canonical,
    validate_axis_position_values,
    validate_reachable_gap,
)
from tests.numerical_gates.registry import (
    GATE_REGISTRY,
    FixtureScalePolicy,
    MutationMode,
    dynamic_atom_ids,
    isolatable_atom_ids,
)

_TWO_SIDED = tuple(
    entry for entry in GATE_REGISTRY if entry.mutation_mode is MutationMode.TWO_SIDED
)
_REGISTRY = {entry.gate_id: entry for entry in _TWO_SIDED}


@pytest.fixture(scope="module")
def boundary_executions() -> dict[str, BoundaryExecution]:
    """Execute each immutable baseline once for all structural audits."""
    return {case_id: case() for case_id, case in BOUNDARY_CASES.items()}


def test_every_two_sided_gate_has_exactly_one_executable_boundary_suite() -> None:
    required = {entry.gate_id for entry in _TWO_SIDED}

    assert len(required) == 84
    assert len(BOUNDARY_SUITES) == len(required)
    assert set(BOUNDARY_SUITES) == required


def test_every_isolatable_atomic_premise_has_one_named_concrete_case() -> None:
    required = {
        atom_id for entry in _TWO_SIDED for atom_id in isolatable_atom_ids(entry)
    }

    assert len(required) == 213
    assert len(ATOM_CASES) == len(required)
    assert set(ATOM_CASES) == required
    assert len({reference.case_id for reference in ATOM_CASES.values()}) == len(
        required
    )
    for atom_id, reference in ATOM_CASES.items():
        assert reference.atom_id == atom_id
        assert reference.gate_id in _REGISTRY
        assert atom_id in isolatable_atom_ids(_REGISTRY[reference.gate_id])
        assert reference.case_id in BOUNDARY_CASES
        assert BOUNDARY_CASES[reference.case_id].atom_ids == (atom_id,)
        assert callable(reference)


def test_nonisolated_dynamic_atoms_are_retained_as_quantified_ambiguities() -> None:
    ambiguous = {
        atom_id: (entry.gate_id, reason)
        for entry in _TWO_SIDED
        for atom_id, reason in entry.atom_isolation_ambiguities.items()
    }

    assert not ambiguous
    assert not set(ambiguous) & set(ATOM_CASES)
    for atom_id, (gate_id, reason) in ambiguous.items():
        assert atom_id in dynamic_atom_ids(_REGISTRY[gate_id])
        assert reason in BOUNDARY_SUITES[gate_id].ambiguities


def test_every_case_resolves_to_its_gate_and_declared_fixture_family() -> None:
    assert set(FixtureFamily) == {
        suite.fixture_family for suite in BOUNDARY_SUITES.values()
    }
    assert len(BOUNDARY_CASES) == sum(
        len(suite.cases) for suite in BOUNDARY_SUITES.values()
    )
    for gate_id, suite in BOUNDARY_SUITES.items():
        assert suite.cases
        assert suite.gate_id == gate_id
        assert all(case.gate_id == gate_id for case in suite.cases)
        assert all(case.fixture_family is suite.fixture_family for case in suite.cases)
        assert all(callable(case) for case in suite.cases)
        assert suite.tighten_case_id in BOUNDARY_CASES
        assert suite.loosen_case_id in BOUNDARY_CASES
        assert suite.tighten_case_id != suite.loosen_case_id


_FLOAT_ROLES = {
    PointRole.VERY_LOW,
    PointRole.BELOW_RELATIVE_1E6,
    PointRole.BELOW_RELATIVE_1E12,
    PointRole.BELOW_ULP,
    PointRole.AT,
    PointRole.ABOVE_ULP,
    PointRole.ABOVE_RELATIVE_1E12,
    PointRole.ABOVE_RELATIVE_1E6,
    PointRole.VERY_HIGH,
    PointRole.EXTREME,
}
_QUANTIZED_NEIGHBOURHOOD_ROLES = {
    PointRole.BELOW_ULP,
    PointRole.AT,
    PointRole.ABOVE_ULP,
}
_REACHABLE_STRADDLE_ROLES = {
    PointRole.REACHABLE_BELOW,
    PointRole.REACHABLE_ABOVE,
}
_INTEGER_ROLES = {
    PointRole.VERY_LOW,
    PointRole.BELOW_INTEGER,
    PointRole.AT,
    PointRole.ABOVE_INTEGER,
    PointRole.VERY_HIGH,
    PointRole.EXTREME,
}
_EXACT_ROLES = {
    PointRole.VERY_LOW,
    PointRole.EXACT,
    PointRole.ULP_MISMATCH,
    PointRole.SUBNORMAL_MISMATCH,
    PointRole.MATERIAL_MISMATCH,
    PointRole.VERY_HIGH,
    PointRole.EXTREME,
}
_CAPABILITY_ROLES = {
    PointRole.CAPABILITY_LOW,
    PointRole.VALID_CAPABILITY,
    PointRole.INVALID_CAPABILITY,
    PointRole.CAPABILITY_HIGH,
    PointRole.EXTREME,
}


@pytest.mark.parametrize("gate_id", sorted(BOUNDARY_SUITES))
def test_threshold_grid_has_the_required_real_topology(gate_id: str) -> None:
    suite = BOUNDARY_SUITES[gate_id]
    roles = {case.threshold_point.role for case in suite.cases}
    required = {
        BoundaryTopology.FLOAT: _FLOAT_ROLES,
        BoundaryTopology.INTEGER: _INTEGER_ROLES,
        BoundaryTopology.EXACT: _EXACT_ROLES,
        BoundaryTopology.CAPABILITY: _CAPABILITY_ROLES,
    }[suite.topology]

    assert roles >= required - suite.omitted_unrepresentable_roles
    assert not (roles & suite.omitted_unrepresentable_roles)
    if suite.omitted_unrepresentable_roles:
        assert suite.topology is BoundaryTopology.FLOAT
        if suite.omitted_unrepresentable_roles & _QUANTIZED_NEIGHBOURHOOD_ROLES:
            assert suite.ambiguities
            assert roles >= _REACHABLE_STRADDLE_ROLES
        else:
            assert suite.omitted_unrepresentable_roles == {
                PointRole.BELOW_RELATIVE_1E6,
                PointRole.BELOW_RELATIVE_1E12,
                PointRole.ABOVE_RELATIVE_1E12,
                PointRole.ABOVE_RELATIVE_1E6,
            }
    if roles & _REACHABLE_STRADDLE_ROLES:
        assert suite.topology is BoundaryTopology.FLOAT
        assert suite.ambiguities
        assert suite.omitted_unrepresentable_roles & _QUANTIZED_NEIGHBOURHOOD_ROLES
        validate_reachable_gap(suite)
    assert {case.threshold_point.expected_side for case in suite.cases} == {
        GateSide.ADMITTED,
        GateSide.REFUSED,
    }
    single_axis = len(_REGISTRY[gate_id].axes) == 1
    if single_axis and suite.topology is BoundaryTopology.INTEGER:
        assert not roles & {
            PointRole.BELOW_ULP,
            PointRole.ABOVE_ULP,
            PointRole.BELOW_RELATIVE_1E6,
            PointRole.ABOVE_RELATIVE_1E6,
        }
    if single_axis and suite.topology in {
        BoundaryTopology.EXACT,
        BoundaryTopology.CAPABILITY,
    }:
        assert not roles & {
            PointRole.BELOW_ULP,
            PointRole.ABOVE_ULP,
            PointRole.REACHABLE_BELOW,
            PointRole.REACHABLE_ABOVE,
            PointRole.BELOW_INTEGER,
            PointRole.ABOVE_INTEGER,
        }


def test_internal_selectors_use_their_real_observable_execution_class() -> None:
    required_two_payload = {
        "EAGER:lambda-logdet:subnormal-rescale",
    }

    assert all(
        BOUNDARY_SUITES[gate_id].execution_class is ExecutionClass.TWO_PAYLOAD
        for gate_id in required_two_payload
    )
    assert (
        BOUNDARY_SUITES["MAP:map_estimate:curvature-scale-clamp"].execution_class
        is ExecutionClass.PAYLOAD_OR_REFUSAL
    )


@pytest.mark.parametrize("gate_id", sorted(BOUNDARY_SUITES))
def test_every_declared_axis_has_low_endpoints_high_and_extreme(
    gate_id: str,
    boundary_executions: dict[str, BoundaryExecution],
) -> None:
    entry = _REGISTRY[gate_id]
    suite = BOUNDARY_SUITES[gate_id]
    expected_positions = set(AxisPosition) - {AxisPosition.INTERIOR}
    by_axis: dict[str, set[AxisPosition]] = {}
    for case in suite.cases:
        execution = boundary_executions[case.case_id]
        for realized_axis in execution.realized_point.axes:
            if realized_axis.position is AxisPosition.INTERIOR:
                continue
            by_axis.setdefault(realized_axis.axis_name, set()).add(
                realized_axis.position
            )

    assert set(by_axis) == {axis.name for axis in entry.axes}
    assert all(positions == expected_positions for positions in by_axis.values())
    validate_axis_position_values(suite)


def test_oracle_module_has_no_product_import_or_dynamic_product_lookup() -> None:
    source = inspect.getsource(oracles)
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    dynamic_imports = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "bayesmith" not in imported_roots
    assert "__import__" not in dynamic_imports
    assert "import_module" not in dynamic_imports


def test_provider_modules_never_import_or_call_a_dispatcher() -> None:
    provider_paths = tuple(
        Path(__file__).with_name(name)
        for name in (
            "boundary_eager.py",
            "boundary_ladder.py",
            "boundary_plan.py",
            "boundary_diagnose_graph.py",
        )
    )
    for path in provider_paths:
        tree = ast.parse(path.read_text())
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attributes = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        forbidden = {
            name
            for name in names | attributes
            if name == "dispatch_logdet" or name.startswith("dispatch_")
        }
        assert not forbidden, (path, forbidden)


def test_non_unit_scale_policy_is_realized_on_every_applicable_case() -> None:
    for gate_id, suite in BOUNDARY_SUITES.items():
        entry = _REGISTRY[gate_id]
        scales = {case.non_unit_scale for case in suite.cases}
        if entry.fixture_scale_policy is FixtureScalePolicy.NON_UNIT_REQUIRED:
            assert None not in scales, gate_id
            for scale in scales:
                assert scale is not None
                assert len(scale) == 2
                assert scale[0] != scale[1]
                assert 1.0 not in scale


def test_every_reserved_mutation_witness_resolves_to_a_concrete_callable() -> None:
    required = {
        name
        for entry in _TWO_SIDED
        for name in (entry.tighten_witness, entry.loosen_witness)
    }

    assert len(required) == 168
    assert set(WITNESS_CASES) == required
    for name, reference in WITNESS_CASES.items():
        assert reference.name == name
        assert reference.case_id in BOUNDARY_CASES
        assert callable(reference)


def test_case_ids_are_stable_and_not_reused_across_gates() -> None:
    counts = Counter(
        case.case_id for suite in BOUNDARY_SUITES.values() for case in suite.cases
    )

    assert counts
    assert all(count == 1 for count in counts.values())
    assert all(
        case.case_id.startswith(f"{case.gate_id}::boundary::")
        for case in BOUNDARY_CASES.values()
    )


@pytest.mark.parametrize("case_id", sorted(BOUNDARY_CASES))
def test_every_boundary_cell_executes_and_retains_independent_evidence(
    case_id: str,
    boundary_executions: dict[str, BoundaryExecution],
) -> None:
    case = BOUNDARY_CASES[case_id]
    execution = boundary_executions[case_id]

    assert execution.case_id == case_id
    assert execution.gate_id == case.gate_id
    assert execution.threshold_point is case.threshold_point
    assert execution.realized_point.quantity
    assert execution.realized_point.axes
    declared_axis_names = {axis.axis_name for axis in case.axes}
    assert len(execution.realized_point.axes) == len(declared_axis_names)
    assert len({axis.axis_name for axis in execution.realized_point.axes}) == len(
        declared_axis_names
    )
    assert len(execution.realization_fingerprint) == 64
    assert execution.axes == case.axes
    assert execution.direct_calls
    assert execution.oracle_checks or execution.evaluations
    assert execution.observed_gate_side is execution.expected_gate_side
    assert all(check.passed for check in execution.oracle_checks)
    assert all(
        evaluation.verdict is oracles.NumericalVerdict.OK
        for evaluation in execution.evaluations
    )
    assert execution.verdict is oracles.NumericalVerdict.OK
    if case.atom_ids:
        assert execution.isolated_atom == case.atom_ids[0]
        assert execution.atom_relation is case.atom_relation
        assert execution.atom_relation is not None
        assert {item.atom_id for item in execution.atom_evidence} == set(
            _REGISTRY[case.gate_id].conjunction_atom_ids
        )
        assert all(item.realized_keys for item in execution.atom_evidence)
        assert execution.sibling_premises
        assert all(check.passed for check in execution.sibling_premises)
    else:
        assert execution.isolated_atom is None


@pytest.mark.parametrize("gate_id", sorted(BOUNDARY_SUITES))
def test_every_required_callable_executes_across_each_suite(
    gate_id: str,
    boundary_executions: dict[str, BoundaryExecution],
) -> None:
    """Union coverage: branched production paths may skip a helper per cell.

    A cell whose real production path never reaches a helper (for example the
    factor-projection arm of ``rank:evidence`` when no factors exist) still
    drives the gate predicate and both payloads; every registered callable
    must nevertheless execute in at least one cell of its suite, and
    ``execute_case`` keeps every cell inside its per-case allowlist.
    """
    suite = BOUNDARY_SUITES[gate_id]
    covered = set()
    for case in suite.cases:
        if case.atom_ids:
            continue
        covered |= set(boundary_executions[case.case_id].direct_calls)
    missing = REQUIRED_DIRECT_CALLS[gate_id] - covered
    assert not missing, gate_id


@pytest.mark.parametrize("gate_id", sorted(BOUNDARY_SUITES))
def test_key_boundary_roles_realize_distinct_production_inputs(
    gate_id: str,
    boundary_executions: dict[str, BoundaryExecution],
) -> None:
    suite = BOUNDARY_SUITES[gate_id]
    standard_cases = [case for case in suite.cases if not case.atom_ids]
    by_role = {
        case.threshold_point.role: boundary_executions[case.case_id]
        for case in standard_cases
    }
    critical = {
        BoundaryTopology.FLOAT: {
            PointRole.BELOW_ULP,
            PointRole.AT,
            PointRole.ABOVE_ULP,
            PointRole.BELOW_RELATIVE_1E6,
            PointRole.ABOVE_RELATIVE_1E6,
        },
        BoundaryTopology.INTEGER: {
            PointRole.BELOW_INTEGER,
            PointRole.AT,
            PointRole.ABOVE_INTEGER,
        },
        BoundaryTopology.EXACT: {
            PointRole.EXACT,
            PointRole.ULP_MISMATCH,
            PointRole.SUBNORMAL_MISMATCH,
            PointRole.MATERIAL_MISMATCH,
        },
        BoundaryTopology.CAPABILITY: {
            PointRole.CAPABILITY_LOW,
            PointRole.VALID_CAPABILITY,
            PointRole.INVALID_CAPABILITY,
            PointRole.CAPABILITY_HIGH,
            PointRole.EXTREME,
        },
    }[suite.topology]
    critical -= suite.omitted_unrepresentable_roles
    if suite.omitted_unrepresentable_roles & _QUANTIZED_NEIGHBOURHOOD_ROLES:
        critical |= _REACHABLE_STRADDLE_ROLES
    fingerprints = {
        by_role[role].mutation_input_fingerprint for role in critical
    }

    assert len(fingerprints) == len(critical), gate_id


@pytest.mark.parametrize("gate_id", sorted(BOUNDARY_SUITES))
def test_atomic_cases_preserve_relation_aware_consumed_premises(
    gate_id: str,
    boundary_executions: dict[str, BoundaryExecution],
) -> None:
    suite = BOUNDARY_SUITES[gate_id]
    atomic = [boundary_executions[case_id] for case_id in suite.atom_case_ids.values()]

    assert len({execution.isolated_atom for execution in atomic}) == len(atomic)
    by_atom = {execution.isolated_atom: execution for execution in atomic}
    entry = _REGISTRY[gate_id]
    for atom_id, execution in by_atom.items():
        assert atom_id is not None
        relation = execution.atom_relation
        assert relation is not None
        canonical = source_alias_canonical(entry, atom_id)
        if relation.kind is AtomRelationKind.ALIAS:
            assert relation.canonical_atom_id == canonical
            assert (
                execution.realization_fingerprint
                == by_atom[canonical].realization_fingerprint
            )
        elif relation.kind is AtomRelationKind.INDEPENDENT:
            assert canonical == atom_id
            assert not relation.prerequisites
        else:
            assert relation.kind is AtomRelationKind.DEPENDENT
            assert relation.prerequisites
            assert relation.logic is not None
            assert relation.rationale


def test_relative_error_and_verdict_bands_are_exactly_the_shared_contract() -> None:
    assert oracles.relative_error(1.0, 1.0) == 0.0
    assert oracles.relative_error(0.0, 0.0) == 0.0
    assert oracles.relative_error(1e-310, 0.0) == abs(1e-310 - 0.0) / max(
        abs(1e-310), abs(0.0), 1e-300
    )
    assert oracles.numerical_verdict(1.0, 1.0005) is oracles.NumericalVerdict.OK
    assert oracles.numerical_verdict(1.0, 1.01) is oracles.NumericalVerdict.WARN
    assert oracles.numerical_verdict(1.0, 1.2) is oracles.NumericalVerdict.BAD
