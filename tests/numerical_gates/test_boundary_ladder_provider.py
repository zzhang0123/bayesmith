"""Completeness contract for the standalone LADDER Task-3 provider."""

from __future__ import annotations

import ast
import importlib
import math
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayesmith.marginal import _logdet_ladder as ladder
from tests.numerical_gates import boundary_ladder as ladder_provider
from tests.numerical_gates import oracles
from tests.numerical_gates.boundary_contract import REQUIRED_DIRECT_CALLS
from tests.numerical_gates.boundary_core import (
    AxisPosition,
    BoundaryTopology,
    GateSide,
    PointRole,
    validate_axis_position_values,
)
from tests.numerical_gates.mutation_harness import (
    MutationDirection,
    MutationStrategy,
    run_mutation,
)
from tests.numerical_gates.mutation_specs import LADDER_MUTATION_SPECS
from tests.numerical_gates.registry import GATE_REGISTRY, MutationMode
from tests.numerical_gates.source_manifest import EXPECTED_SOURCE_MANIFEST

pytestmark = pytest.mark.full


_PROVIDER_PATH = Path(__file__).with_name("boundary_ladder.py")
_SOURCE_SYNTAX = {item.candidate_id: item.syntax for item in EXPECTED_SOURCE_MANIFEST}
_RUNG_PAYLOAD_METHODS = {
    "LADDER:rung0:base": "eager.lambda_logdet",
    "LADDER:rung1:low-rank-size": "eager.low_rank_logdet",
    "LADDER:rung2:chain": "eager.state_space_logdet",
    "LADDER:rung3:structured": "eager.structured_logdet",
    "LADDER:rung4:dense": "eager.dense_cholesky_logdet",
    "LADDER:rung5:finite-size": "eager.finite_perturbation_logdet",
    "LADDER:rung5:finite-executable": "eager.finite_perturbation_logdet",
    "LADDER:rung6:trace": "eager.truncated_trace_logdet",
    "LADDER:rung7:frozen": "eager.frozen_hutchinson_trace_logdet",
}


def _assert_payload_attempt(execution: object, method: str) -> None:
    assert method in execution.direct_calls
    evaluated = any(item.method == method for item in execution.evaluations)
    exception = any(
        note.startswith(f"direct-call exception: {method}:")
        for note in execution.notes
    )
    assert evaluated or exception


def _selection_evaluations(execution: object) -> tuple[object, ...]:
    return tuple(
        item
        for item in execution.evaluations
        if item.method.startswith("ladder.selected-product/")
    )


def _direct_product_outcomes(execution: object) -> dict[str, object]:
    return {
        key: value
        for key, value in execution.direct_return_values.items()
        if key.startswith("direct-product::")
    }


def test_ladder_predicate_recorder_restores_an_existing_module_global() -> None:
    marker = object()
    name = "__boundary_record_predicate__"
    previous = ladder.__dict__.get(name, marker)
    ladder.__dict__[name] = marker
    record = SimpleNamespace(instrumentation_intact=True, source_records={})
    try:
        with ladder_provider._recording_production_predicates(record):
            pass
        assert ladder.__dict__.get(name) is marker
    finally:
        if previous is marker:
            ladder.__dict__.pop(name, None)
        else:
            ladder.__dict__[name] = previous


def _provider_suites() -> tuple[object, ...]:
    try:
        module = importlib.import_module("tests.numerical_gates.boundary_ladder")
    except ModuleNotFoundError as error:
        pytest.fail(f"standalone LADDER provider is missing: {error}")
    return module.LADDER_SUITES


def test_standalone_ladder_provider_covers_all_gates_and_atoms_once() -> None:
    suites = _provider_suites()
    required = {
        entry.gate_id: set(entry.conjunction_atom_ids)
        for entry in GATE_REGISTRY
        if entry.mutation_mode is MutationMode.TWO_SIDED
        and entry.gate_id.startswith("LADDER:")
    }

    assert len(required) == 21
    assert sum(len(atoms) for atoms in required.values()) == 56
    assert {suite.gate_id for suite in suites} == set(required)
    assert len(suites) == len(required)
    assert {suite.gate_id: set(suite.atom_case_ids) for suite in suites} == required
    assert len(LADDER_MUTATION_SPECS) == 42
    assert {spec.gate_id for spec in LADDER_MUTATION_SPECS} == set(required)
    assert all(
        spec.strategy is not MutationStrategy.FLIP_BOOLEAN
        for spec in LADDER_MUTATION_SPECS
    )


def test_every_standalone_ladder_cell_executes_with_independent_evidence() -> None:
    suites = _provider_suites()

    for suite in suites:
        for case in suite.cases:
            execution = case()
            assert execution.observed_gate_side is execution.expected_gate_side
            assert execution.direct_calls
            assert all(check.passed for check in execution.oracle_checks)
            assert all(check.passed for check in execution.sibling_premises)
            assert all(
                evaluation.verdict is oracles.NumericalVerdict.OK
                for evaluation in execution.evaluations
            )
            assert execution.verdict is oracles.NumericalVerdict.OK
            if case.atom_ids:
                assert execution.isolated_atom == case.atom_ids[0]
                assert execution.atom_evidence
                assert execution.sibling_premises


def test_atom_actual_and_oracle_mappings_do_not_reuse_one_expression() -> None:
    """A source result and its oracle may consume the same fixture, not one value."""
    tree = ast.parse(_PROVIDER_PATH.read_text())
    offenders: list[tuple[int, str]] = []
    for function in (
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ):
        assignments = {
            target.id: node.value
            for node in function.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        for call in (node for node in ast.walk(function) if isinstance(node, ast.Call)):
            if not isinstance(call.func, ast.Name) or call.func.id != "_atom_premises":
                continue
            if len(call.args) < 3:
                continue
            mappings: list[ast.Dict] = []
            for argument in call.args[1:3]:
                resolved = (
                    assignments.get(argument.id)
                    if isinstance(argument, ast.Name)
                    else argument
                )
                if not isinstance(resolved, ast.Dict):
                    break
                mappings.append(resolved)
            if len(mappings) != 2:
                continue
            actual, oracle = mappings
            actual_by_syntax = {
                key.value: value
                for key, value in zip(actual.keys, actual.values, strict=True)
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            oracle_by_syntax = {
                key.value: value
                for key, value in zip(oracle.keys, oracle.values, strict=True)
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            for syntax in actual_by_syntax.keys() & oracle_by_syntax.keys():
                if ast.dump(
                    actual_by_syntax[syntax], include_attributes=False
                ) == ast.dump(oracle_by_syntax[syntax], include_attributes=False):
                    offenders.append((call.lineno, syntax))

    assert not offenders


def _suite(gate_id: str) -> object:
    return next(suite for suite in _provider_suites() if suite.gate_id == gate_id)


def _case(gate_id: str, role: PointRole) -> object:
    return next(
        case
        for case in _suite(gate_id).cases
        if not case.atom_ids and case.threshold_point.role is role
    )


def test_sigma_payload_grades_original_and_selected_values_against_dense_truth() -> (
    None
):
    suite = _suite("LADDER:sigma:payload-symmetry")
    for case in suite.cases:
        execution = case()
        methods = {evaluation.method for evaluation in execution.evaluations}
        assert "ladder._sigma_payload/original-logdet" in methods
        assert "ladder._sigma_payload/selected-logdet" in methods
        assert all(
            "dense" in evaluation.oracle.lower()
            or "decimal" in evaluation.oracle.lower()
            for evaluation in execution.evaluations
        )


def test_sigma_payload_representative_corruption_is_bad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("tests.numerical_gates.boundary_ladder")
    case = _case("LADDER:sigma:payload-symmetry", PointRole.AT)

    def corrupt(sigma: np.ndarray, *, operation: str) -> np.ndarray:
        del operation
        return np.eye(sigma.shape[0]) * 99.0

    monkeypatch.setattr(module.ladder, "_symmetric_roundoff_representative", corrupt)
    execution = case()

    assert execution.verdict is oracles.NumericalVerdict.BAD
    selected = next(
        evaluation
        for evaluation in execution.evaluations
        if evaluation.method == "ladder._sigma_payload/selected-logdet"
    )
    assert selected.verdict is oracles.NumericalVerdict.BAD


def test_rung7_primary_oracle_is_true_dense_logdet() -> None:
    suite = _suite("LADDER:rung7:frozen")
    admitted = [
        case()
        for case in suite.cases
        if case.threshold_point.expected_side is GateSide.ADMITTED
    ]
    assert admitted
    for execution in admitted:
        evaluation = next(
            item
            for item in execution.evaluations
            if item.method == "eager.frozen_hutchinson_trace_logdet"
        )
        inputs = execution.realized_point
        assert inputs is not None
        # The provider retains the true dense target as a separate return key;
        # the frozen recurrence may only be secondary evidence.
        assert math.isclose(
            evaluation.oracle_value,
            execution.oracle_values[0],
            rel_tol=0.0,
            abs_tol=0.0,
        )
        assert (
            "dense" in evaluation.oracle.lower()
            or "decimal" in evaluation.oracle.lower()
        )
        assert evaluation.verdict is oracles.NumericalVerdict.OK


def test_each_ladder_suite_executes_every_required_callable() -> None:
    for suite in _provider_suites():
        calls = {method for case in suite.cases for method in case().direct_calls}
        assert REQUIRED_DIRECT_CALLS[suite.gate_id] <= calls


def test_invoke_rejects_a_label_not_bound_to_the_exact_runtime_callable() -> None:
    module = importlib.import_module("tests.numerical_gates.boundary_ladder")
    point = _case("LADDER:rung4:dense", PointRole.AT).threshold_point
    record = module._FixtureRecord("LADDER:rung4:dense", point, None)
    record.input("matrix", np.diag([1.3, 2.7]))

    with pytest.raises(AssertionError, match="callable identity"):
        record.invoke(
            "eager.dense_cholesky_logdet",
            module.eager.lambda_logdet,
            result_key="wrong",
            args=("matrix",),
        )


def test_invoke_derives_lineage_and_cannot_retroactively_label_values() -> None:
    """A value added after the call cannot masquerade as a consumed argument."""
    module = importlib.import_module("tests.numerical_gates.boundary_ladder")
    point = _case("LADDER:rung4:dense", PointRole.AT).threshold_point
    record = module._FixtureRecord("LADDER:rung4:dense", point, None)
    record.input("matrix", np.diag([1.3, 2.7]))

    record.invoke(
        "eager.lambda_logdet",
        module.eager.lambda_logdet,
        result_key="logdet",
        args=("matrix",),
    )
    record.input("retroactive_label", 1.3)

    assert record.direct_input_keys == ["matrix"]
    assert "retroactive_label" not in record.direct_input_keys


def test_source_recorder_does_not_overwrite_an_in_process_rung4_mutant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("tests.numerical_gates.boundary_ladder")
    original = module.ladder.check_logdet_premises

    def mutated(problem: object, *, config: object = None) -> tuple[object, ...]:
        verdicts = list(original(problem, config=config))
        verdicts[4] = replace(verdicts[4], satisfied=False)
        return tuple(verdicts)

    monkeypatch.setattr(module.ladder, "check_logdet_premises", mutated)
    execution = _case("LADDER:rung4:dense", PointRole.AT)()

    assert execution.observed_gate_side is GateSide.REFUSED
    assert execution.verdict is oracles.NumericalVerdict.BAD


def test_runtime_callable_replacement_does_not_create_an_identity_only_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("tests.numerical_gates.boundary_ladder")
    original = module.ladder.check_logdet_premises

    def forwarding(problem: object, *, config: object = None) -> tuple[object, ...]:
        return original(problem, config=config)

    monkeypatch.setattr(module.ladder, "check_logdet_premises", forwarding)
    execution = _case("LADDER:rung4:dense", PointRole.AT)()

    assert execution.observed_gate_side is GateSide.ADMITTED
    assert all(check.passed for check in execution.oracle_checks)
    assert execution.verdict is oracles.NumericalVerdict.OK


def test_forwarding_wrapper_preserves_atom_evidence_and_fingerprints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recording follows the production callable reached through a wrapper."""
    module = importlib.import_module("tests.numerical_gates.boundary_ladder")
    suite = _suite("LADDER:rung4:dense")
    entry = next(
        item for item in GATE_REGISTRY if item.gate_id == "LADDER:rung4:dense"
    )
    atom_id = next(
        atom
        for atom in entry.conjunction_atom_ids
        if _SOURCE_SYNTAX[atom] == "n <= config.dense_max_n"
    )
    case = next(item for item in suite.cases if item.atom_ids == (atom_id,))
    baseline = case()
    original = module.ladder.check_logdet_premises

    def forwarding(problem: object, *, config: object = None) -> tuple[object, ...]:
        return original(problem, config=config)

    monkeypatch.setattr(module.ladder, "check_logdet_premises", forwarding)
    forwarded = case()

    assert forwarded.realization_fingerprint == baseline.realization_fingerprint
    assert forwarded.mutation_input_fingerprint == baseline.mutation_input_fingerprint
    assert len(forwarded.atom_evidence) == len(baseline.atom_evidence)
    for actual, expected in zip(
        forwarded.atom_evidence, baseline.atom_evidence, strict=True
    ):
        assert actual.atom_id == expected.atom_id
        np.testing.assert_array_equal(actual.raw_actual, expected.raw_actual)
        assert actual.truth is expected.truth
        assert actual.reducer is expected.reducer
        assert actual.realized_keys == expected.realized_keys
        assert actual.oracle == expected.oracle
    assert len(forwarded.sibling_premises) == len(baseline.sibling_premises)
    for actual, expected in zip(
        forwarded.sibling_premises, baseline.sibling_premises, strict=True
    ):
        assert actual.oracle == expected.oracle
        np.testing.assert_array_equal(actual.actual, expected.actual)
        np.testing.assert_array_equal(actual.expected, expected.expected)
        assert actual.passed is expected.passed


@pytest.mark.parametrize(
    "gate_id",
    (
        "LADDER:structure:diagonal-tolerance",
        "LADDER:structure:circulant-tolerance-spectrum",
        "LADDER:structure:toeplitz-tolerance",
        "LADDER:structure:kronecker-evidence",
        "LADDER:rank:evidence",
        "LADDER:determinant-lemma:payload",
        "LADDER:rung0:base",
        "LADDER:rung2:chain",
        "LADDER:rung3:structured",
        "LADDER:rung5:finite-executable",
    ),
)
def test_exact_ulp_and_subnormal_cells_have_distinct_realizations(gate_id: str) -> None:
    ulp = _case(gate_id, PointRole.ULP_MISMATCH)()
    subnormal = _case(gate_id, PointRole.SUBNORMAL_MISMATCH)()

    assert ulp.realization_fingerprint != subnormal.realization_fingerprint
    assert ulp.realized_point.value != subnormal.realized_point.value


@pytest.mark.parametrize(
    "gate_id",
    (
        "LADDER:structure:diagonal-tolerance",
        "LADDER:structure:circulant-tolerance-spectrum",
        "LADDER:structure:toeplitz-tolerance",
        "LADDER:rung0:base",
        "LADDER:rung2:chain",
        "LADDER:rung3:structured",
    ),
)
def test_generic_exact_ulp_mismatch_is_adjacent_to_exact_zero(gate_id: str) -> None:
    execution = _case(gate_id, PointRole.ULP_MISMATCH)()
    minimum_subnormal = float(np.nextafter(0.0, -math.inf))

    assert execution.realized_point.threshold == 0.0
    assert execution.realized_point.value == minimum_subnormal


def test_rung5_exact_roles_are_distinct_values_consumed_in_the_real_matrix() -> None:
    roles = (
        PointRole.EXACT,
        PointRole.ULP_MISMATCH,
        PointRole.SUBNORMAL_MISMATCH,
        PointRole.MATERIAL_MISMATCH,
        PointRole.EXTREME,
    )
    cases = {role: _case("LADDER:rung5:finite-executable", role) for role in roles}
    observations = {
        role: case.runner(case.threshold_point) for role, case in cases.items()
    }

    assert len(
        {
            float(observation.realized_point.value).hex()
            for observation in observations.values()
        }
    ) == len(roles)
    assert observations[PointRole.EXACT].realized_point.value == 0.0
    assert observations[PointRole.ULP_MISMATCH].realized_point.threshold == 0.0
    assert observations[PointRole.SUBNORMAL_MISMATCH].realized_point.threshold == 0.0
    assert observations[PointRole.ULP_MISMATCH].realized_point.value == float(
        np.nextafter(0.0, -math.inf)
    )
    assert observations[PointRole.SUBNORMAL_MISMATCH].realized_point.value == float(
        np.nextafter(0.0, math.inf)
    )
    for observation in observations.values():
        point = observation.realized_point
        assert point.input_key in observation.direct_input_keys
        assert "perturbation" in observation.direct_input_keys
        perturbation = np.asarray(observation.realized_inputs["perturbation"])
        assert any(
            float(item).hex() == float(point.value).hex()
            for item in perturbation.reshape(-1)
        )


def test_rung4_dimension_atom_lineage_comes_from_its_captured_operands() -> None:
    suite = _suite("LADDER:rung4:dense")
    entry = next(item for item in GATE_REGISTRY if item.gate_id == "LADDER:rung4:dense")
    atom_id = next(
        atom
        for atom in entry.conjunction_atom_ids
        if _SOURCE_SYNTAX[atom] == "n <= config.dense_max_n"
    )
    execution = next(case() for case in suite.cases if case.atom_ids == (atom_id,))
    evidence = next(item for item in execution.atom_evidence if item.atom_id == atom_id)

    assert "dimension" in evidence.realized_keys
    assert "config_dense_max_n" in evidence.realized_keys
    assert "config_structure_rtol" not in evidence.realized_keys
    assert set(evidence.realized_keys) < set(execution.direct_input_keys)


def test_rung7_has_no_product_isomorphic_recurrence_oracle() -> None:
    module = importlib.import_module("tests.numerical_gates.boundary_ladder")

    assert not hasattr(module, "_frozen_oracle")


def test_ladder_extreme_axes_are_consumed_by_real_production_inputs() -> None:
    rung1 = _suite("LADDER:rung1:low-rank-size")
    rung7 = _suite("LADDER:rung7:frozen")

    def real_matrix_shape(gate_id: str, role: PointRole) -> tuple[int, ...]:
        case = _case(gate_id, role)
        return np.asarray(
            case.runner(case.threshold_point).realized_inputs["lambda_matrix"]
        ).shape

    rung1_shapes = {
        role: real_matrix_shape(rung1.gate_id, role)
        for role in (PointRole.VERY_HIGH, PointRole.EXTREME)
    }
    rung7_shapes = {
        role: real_matrix_shape(rung7.gate_id, role)
        for role in (PointRole.VERY_HIGH, PointRole.EXTREME)
    }

    assert rung1_shapes[PointRole.VERY_HIGH][-1] == 257
    assert rung1_shapes[PointRole.EXTREME][-1] == 10_000
    assert rung7_shapes[PointRole.VERY_HIGH][-1] == 257
    assert rung7_shapes[PointRole.EXTREME][-1] == 10_000


def test_ladder_key_roles_are_distinct_and_all_axis_positions_are_executed() -> None:
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
    }
    for suite in _provider_suites():
        executions = [(case, case()) for case in suite.cases if not case.atom_ids]
        by_role = {
            case.threshold_point.role: execution for case, execution in executions
        }
        required_roles = critical[suite.topology] - suite.omitted_unrepresentable_roles
        fingerprints = {
            by_role[role].realization_fingerprint for role in required_roles
        }
        positions = {
            axis.position
            for _case_value, execution in executions
            for axis in execution.realized_point.axes
            if axis.axis_name == execution.realized_point.active_axis
        }

        assert len(fingerprints) == len(required_roles), suite.gate_id
        assert positions == set(AxisPosition) - {AxisPosition.INTERIOR}, suite.gate_id


@pytest.mark.parametrize(
    "gate_id",
    (
        "LADDER:determinant-lemma:payload",
        "LADDER:rank:evidence",
        "LADDER:rung0:base",
        "LADDER:rung4:dense",
        "LADDER:rung5:finite-size",
        "LADDER:rung6:trace",
        "LADDER:rung7:frozen",
        "LADDER:sigma:finite-two-sum",
        "LADDER:structure:kronecker-evidence",
    ),
)
def test_ladder_axis_positions_use_distinct_direct_inputs(gate_id: str) -> None:
    validate_axis_position_values(_suite(gate_id))


def test_every_ladder_axis_sweep_is_distinct_with_fixed_companions() -> None:
    for suite in _provider_suites():
        validate_axis_position_values(suite)


def test_threshold_mutation_witnesses_are_adjacent_and_include_equality_faces() -> None:
    for gate_id in (
        "LADDER:rung1:low-rank-size",
        "LADDER:rung6:trace",
        "LADDER:rung7:frozen",
    ):
        suite = _suite(gate_id)
        cases = {case.case_id: case for case in suite.cases}
        tighten = cases[suite.tighten_case_id]
        loosen = cases[suite.loosen_case_id]
        assert "equality" in tighten.threshold_point.display_value
        assert "adjacent" in loosen.threshold_point.display_value


@pytest.mark.parametrize("gate_id", tuple(_RUNG_PAYLOAD_METHODS))
def test_every_rung_grid_and_witness_attempts_its_candidate_payload(
    gate_id: str,
) -> None:
    method = _RUNG_PAYLOAD_METHODS[gate_id]
    for case in _suite(gate_id).cases:
        if case.atom_ids:
            continue
        _assert_payload_attempt(case(), method)


@pytest.mark.parametrize(
    "spec",
    LADDER_MUTATION_SPECS,
    ids=lambda spec: f"{spec.gate_id}-{spec.direction.value}",
)
def test_every_ladder_mutant_has_a_bad_selected_product(spec: object) -> None:
    suite = _suite(spec.gate_id)
    case_id = (
        suite.tighten_case_id
        if spec.direction is MutationDirection.TIGHTEN
        else suite.loosen_case_id
    )
    case = next(item for item in suite.cases if item.case_id == case_id)

    result = run_mutation(spec, case)

    for execution in (result.baseline, result.mutant):
        outcomes = _direct_product_outcomes(execution)
        assert any("::admitted::" in key for key in outcomes), spec.gate_id
        assert any("::refused::" in key for key in outcomes), spec.gate_id
        assert all(
            isinstance(outcome, dict)
            and outcome.get("status") in {"returned", "exception"}
            for outcome in outcomes.values()
        )

    baseline_evaluations = _selection_evaluations(result.baseline)
    mutant_evaluations = _selection_evaluations(result.mutant)
    assert all(
        item.verdict is oracles.NumericalVerdict.OK
        for item in baseline_evaluations
    )
    assert any(
        item.verdict is oracles.NumericalVerdict.BAD
        for item in mutant_evaluations
    )
    assert result.hit_count > 0
    assert result.same_realization


@pytest.mark.parametrize(
    "spec",
    LADDER_MUTATION_SPECS,
    ids=lambda spec: f"{spec.gate_id}-{spec.direction.value}",
)
def test_each_ladder_mutation_hits_the_live_target_on_the_same_frozen_input(
    spec: object,
) -> None:
    suite = _suite(spec.gate_id)
    case_id = (
        suite.tighten_case_id
        if spec.direction is MutationDirection.TIGHTEN
        else suite.loosen_case_id
    )
    case = next(item for item in suite.cases if item.case_id == case_id)

    result = run_mutation(spec, case)

    assert result.baseline.verdict is oracles.NumericalVerdict.OK
    assert result.hit_count > 0
    assert result.same_realization
    assert result.killed


@pytest.mark.parametrize(
    "gate_id",
    ("LADDER:sigma:finite-two-sum", "LADDER:rho:measurement"),
)
def test_expected_extreme_overflow_is_handled_without_runtime_warning(
    gate_id: str,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _case(gate_id, PointRole.EXTREME)()
