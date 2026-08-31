"""Contract tests for the diagnose/graph direct-boundary provider."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest

from tests.numerical_gates import boundary_diagnose_graph as diagnose_provider
from tests.numerical_gates.boundary_core import (
    AtomRelationKind,
    PointRole,
    source_alias_canonical,
)
from tests.numerical_gates.registry import GATE_REGISTRY, MutationMode

pytestmark = pytest.mark.full


def test_diagnose_graph_scores_mutants_from_product_outcomes_only() -> None:
    """The provider cannot turn a harness probe into its own gate verdict."""
    source = Path(__file__).with_name("boundary_diagnose_graph.py").read_text()

    assert "active_mutation_trace" not in source


def test_diagnose_graph_provider_module_exists() -> None:
    """A missing provider leaves all 14 diagnose/graph gates unpinned."""
    assert (
        importlib.util.find_spec("tests.numerical_gates.boundary_diagnose_graph")
        is not None
    )


def test_diagnose_graph_provider_covers_every_owned_gate_and_atom() -> None:
    """Dropping a gate or conjunction atom must break provider completeness."""
    from tests.numerical_gates.boundary_diagnose_graph import (
        DIAGNOSE_GRAPH_SUITES,
    )

    required = {
        entry.gate_id: set(entry.conjunction_atom_ids)
        for entry in GATE_REGISTRY
        if entry.mutation_mode is MutationMode.TWO_SIDED
        and entry.gate_id.startswith(("COUPLING:", "MAP:", "GRAPH:"))
    }
    actual = {
        suite.gate_id: set(suite.atom_case_ids) for suite in DIAGNOSE_GRAPH_SUITES
    }

    assert len(required) == 13
    assert sum(map(len, required.values())) == 21
    assert actual == required


def test_diagnose_graph_cells_execute_real_calls_with_relation_aware_atom_evidence() -> (
    None
):
    """Aliases share source evidence; named cases and raw outcomes stay explicit."""
    from tests.numerical_gates.boundary_diagnose_graph import (
        DIAGNOSE_GRAPH_SUITES,
    )

    for suite in DIAGNOSE_GRAPH_SUITES:
        cases = {case.case_id: case for case in suite.cases}
        assert cases
        assert suite.tighten_case_id in cases
        assert suite.loosen_case_id in cases
        assert suite.tighten_case_id != suite.loosen_case_id
        assert set(suite.atom_case_ids.values()) <= set(cases)

        atom_executions = {}
        for case in cases.values():
            execution = case()
            assert execution.observed_gate_side is execution.expected_gate_side
            assert execution.direct_calls
            assert execution.oracle_checks or execution.evaluations
            if case.atom_ids:
                atom_id = case.atom_ids[0]
                assert execution.isolated_atom == atom_id
                assert execution.atom_relation is case.atom_relation
                assert execution.atom_relation is not None
                assert {item.atom_id for item in execution.atom_evidence} == set(
                    suite.atom_case_ids
                )
                assert all(item.realized_keys for item in execution.atom_evidence)
                assert execution.sibling_premises
                assert all(check.passed for check in execution.sibling_premises)
                atom_executions[atom_id] = execution

        assert set(atom_executions) == set(suite.atom_case_ids)
        entry = next(item for item in GATE_REGISTRY if item.gate_id == suite.gate_id)
        for atom_id, execution in atom_executions.items():
            relation = execution.atom_relation
            assert relation is not None
            canonical = source_alias_canonical(entry, atom_id)
            if relation.kind is AtomRelationKind.ALIAS:
                assert relation.canonical_atom_id == canonical
                assert (
                    execution.realization_fingerprint
                    == atom_executions[canonical].realization_fingerprint
                )
            elif relation.kind is AtomRelationKind.INDEPENDENT:
                assert canonical == atom_id
                assert not relation.prerequisites
            else:
                assert relation.kind is AtomRelationKind.DEPENDENT
                assert relation.prerequisites
                assert relation.logic is not None
                assert relation.rationale


def test_condition_finite_gate_uses_a_real_finite_matrix_boundary() -> None:
    """The eigensolver result must come from the matrix, not an injected seam."""
    from tests.numerical_gates.boundary_diagnose_graph import (
        DIAGNOSE_GRAPH_SUITES,
    )

    suite = next(
        item
        for item in DIAGNOSE_GRAPH_SUITES
        if item.gate_id == "COUPLING:_condition_number:finite-spectrum"
    )
    invalid = next(
        case
        for case in suite.cases
        if not case.atom_ids
        and case.threshold_point.role is PointRole.INVALID_CAPABILITY
    )()

    assert invalid.direct_input_keys == ("matrix",)
    assert np.all(np.isfinite(invalid.realized_point.axes[0].value))
    assert not np.all(np.isfinite(invalid.direct_return_values["spectrum"]))


def test_map_payloads_are_one_coherent_autodiff_quadratic() -> None:
    """No MAP boundary may inject objective derivatives or an unrelated spectrum."""
    source = inspect.getsource(diagnose_provider)

    assert "forced_objective" not in source
    assert "forced_gradient" not in source
    assert "forced_hessian" not in source
    assert "forced_spectrum" not in source

    for suite in diagnose_provider.DIAGNOSE_GRAPH_SUITES:
        if not suite.gate_id.startswith("MAP:"):
            continue
        for case in suite.cases:
            execution = case()
            assert not any(
                key.startswith("forced_") for key in execution.direct_input_keys
            )
            if "actual_spectrum" not in execution.direct_return_keys:
                continue
            hessian = np.asarray(execution.direct_return_values["actual_hessian"])
            spectrum = np.asarray(execution.direct_return_values["actual_spectrum"])
            assert np.array_equal(hessian, np.diag(np.diag(hessian)), equal_nan=True)
            np.testing.assert_allclose(
                spectrum,
                np.sort(np.diag(hessian)),
                rtol=8.0 * np.finfo(np.float64).eps,
                atol=0.0,
                equal_nan=True,
            )


def test_names_multiplicity_is_a_captured_product_result_not_a_fake_input() -> None:
    """The boundary quantity must come from ``names.count`` inside ``_names``."""
    from tests.numerical_gates.boundary_diagnose_graph import (
        DIAGNOSE_GRAPH_SUITES,
    )

    suite = next(
        item
        for item in DIAGNOSE_GRAPH_SUITES
        if item.gate_id == "GRAPH:_names:duplicate-multiplicity"
    )
    duplicate = next(
        case
        for case in suite.cases
        if not case.atom_ids and case.threshold_point.role is PointRole.ABOVE_INTEGER
    )()

    assert "maximum_multiplicity" not in duplicate.direct_input_keys
    assert "maximum_multiplicity" in duplicate.direct_return_keys


def test_reserved_mutation_witnesses_are_the_nearest_executed_gate_faces() -> None:
    """Far-side/empty cells cannot prove that moving a threshold is detected."""
    from tests.numerical_gates.boundary_diagnose_graph import (
        DIAGNOSE_GRAPH_SUITES,
    )

    capability = (PointRole.VALID_CAPABILITY, PointRole.INVALID_CAPABILITY)
    expected = {
        "COUPLING:_classify_correlation:value-finite": capability,
        "COUPLING:_classify_correlation:floor-finite": capability,
        "COUPLING:_classify_correlation:lower-noise-floor": (
            PointRole.ABOVE_ULP,
            PointRole.AT,
        ),
        "COUPLING:_classify_correlation:upper-noise-floor": (
            PointRole.BELOW_ULP,
            PointRole.AT,
        ),
        "COUPLING:_condition_number:finite-spectrum": capability,
        "COUPLING:_condition_number:positive-spectrum": (
            PointRole.SUBNORMAL_MISMATCH,
            PointRole.EXACT,
        ),
        "COUPLING:block_coupling:f-xx-spd": capability,
        "COUPLING:block_coupling:f-tt-spd": capability,
        "GRAPH:_names:duplicate-multiplicity": (
            PointRole.AT,
            PointRole.ABOVE_INTEGER,
        ),
        "MAP:map_estimate:finite-derivative-payload": capability,
        "MAP:map_estimate:stationarity-floor": (
            PointRole.AT,
            PointRole.ABOVE_ULP,
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
    suites = {suite.gate_id: suite for suite in DIAGNOSE_GRAPH_SUITES}

    assert set(suites) == set(expected)
    for gate_id, (tighten_role, loosen_role) in expected.items():
        suite = suites[gate_id]
        cases = {case.case_id: case for case in suite.cases}
        assert cases[suite.tighten_case_id].threshold_point.role is tighten_role
        assert cases[suite.loosen_case_id].threshold_point.role is loosen_role
