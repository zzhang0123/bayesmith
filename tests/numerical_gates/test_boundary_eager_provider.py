"""Load-bearing two-sided checks owned by the EAGER boundary provider."""

from __future__ import annotations

import inspect
import sys
from functools import wraps
from unittest.mock import patch

import numpy as np
import pytest

from bayesmith.marginal import _logdet_eager as eager
from tests.numerical_gates import boundary_eager as eager_provider
from tests.numerical_gates import oracles
from tests.numerical_gates.boundary_contract import REQUIRED_DIRECT_CALLS
from tests.numerical_gates.boundary_core import (
    AxisPosition,
    GateSide,
    RealizedAxis,
    RealizedPoint,
)
from tests.numerical_gates.boundary_eager import _WITNESS_ROLES, EAGER_SUITES
from tests.numerical_gates.mutation_harness import (
    MutationDirection,
    MutationSpec,
    run_mutation,
)
from tests.numerical_gates.mutation_specs import EAGER_MUTATION_SPECS

_SUITES = {suite.gate_id: suite for suite in EAGER_SUITES}


def test_eager_standard_cells_execute_every_required_side_method() -> None:
    """Route grids must execute the specialized method and its direct peer."""
    for suite in EAGER_SUITES:
        for case in suite.cases:
            if case.atom_ids:
                continue
            execution = case()
            assert REQUIRED_DIRECT_CALLS[suite.gate_id] <= set(
                execution.direct_calls
            ), case.case_id


def test_eager_predicate_recorder_restores_an_existing_module_global() -> None:
    marker = object()
    name = "__boundary_record_predicate__"
    previous = eager.__dict__.get(name, marker)
    eager.__dict__[name] = marker
    try:
        with eager_provider._recording_production_predicates():
            pass
        assert eager.__dict__.get(name) is marker
    finally:
        if previous is marker:
            eager.__dict__.pop(name, None)
        else:
            eager.__dict__[name] = previous


def test_factor_frame_capture_composes_with_a_prior_local_tracer() -> None:
    events: list[str] = []

    def local_trace(frame: object, event: str, arg: object) -> object:
        del frame, arg
        events.append(event)
        return local_trace

    def global_trace(frame: object, event: str, arg: object) -> object:
        del frame, arg
        if event != "call":
            raise RuntimeError(f"global tracer incorrectly received {event}")
        return local_trace

    def traced_target() -> int:
        value = 1
        return value

    previous = sys.gettrace()
    sys.settrace(global_trace)
    try:
        with eager_provider._capture_factor_frame():
            assert traced_target() == 1
    finally:
        sys.settrace(previous)

    assert "line" in events
    assert "return" in events


def test_eager_provider_does_not_read_mutation_probe_values() -> None:
    source = inspect.getsource(eager_provider)
    assert "__bayesmith_numerical_gate_mutation_probe__" not in source


def test_eager_atom_evidence_has_no_injected_return_seams() -> None:
    """Atom refusals must come from real production inputs and arithmetic."""
    source = inspect.getsource(eager_provider)
    for fake_name in (
        "eigensolver_return",
        "x_matrix_return",
        "eigenvalues_return",
        "final_total",
        "injected_ldexp",
    ):
        assert fake_name not in source

    forbidden_return_keys = {
        "eigensolver_return",
        "scaled_left",
        "scaled_right",
        "x_matrix_return",
        "eigenvalues_return",
        "final_total",
    }
    for suite in EAGER_SUITES:
        for case in suite.cases:
            execution = case()
            assert forbidden_return_keys.isdisjoint(execution.direct_return_keys)


def test_no_eager_admitted_execution_carries_refusal_or_exception_notes() -> None:
    for suite in EAGER_SUITES:
        for case in suite.cases:
            execution = case()
            assert not (
                execution.observed_gate_side is GateSide.ADMITTED
                and execution.notes
            ), case.case_id


def test_eager_exact_check_cannot_bypass_a_false_actual() -> None:
    realized = RealizedPoint(
        quantity="regression predicate",
        input_key="predicate",
        value=False,
        threshold=True,
        dtype=None,
        axes=(
            RealizedAxis(
                axis_name="regression axis",
                position=AxisPosition.ENDPOINT_LOW,
                input_key="predicate",
                value=False,
            ),
        ),
    )
    retained = (
        realized,
        {"predicate": False},
        ("predicate",),
        (),
        None,
        (),
        (),
        (),
    )
    with patch.object(eager_provider, "_realization", return_value=retained):
        observation = eager_provider._checked(
            method="regression predicate",
            oracle="independent false-vs-true regression",
            actual=False,
            expected=True,
        )

    assert observation.oracle_checks[0].actual is False
    assert observation.oracle_checks[0].expected is True
    assert not observation.oracle_checks[0].passed


def test_every_eager_gate_has_explicit_reviewed_witness_roles() -> None:
    spec_gates = {spec.gate_id for spec in EAGER_MUTATION_SPECS}
    assert set(_WITNESS_ROLES) == set(_SUITES) == spec_gates
    assert len(EAGER_MUTATION_SPECS) == 40
    for gate_id, suite in _SUITES.items():
        tighten_role, loosen_role = _WITNESS_ROLES[gate_id]
        by_id = {case.case_id: case for case in suite.cases}
        assert by_id[suite.tighten_case_id].threshold_point.role is tighten_role
        assert by_id[suite.loosen_case_id].threshold_point.role is loosen_role


@pytest.mark.parametrize(
    "spec",
    EAGER_MUTATION_SPECS,
    ids=lambda spec: f"{spec.direction.value}-{spec.gate_id}",
)
def test_eager_witness_kills_exact_live_source_mutation(spec: MutationSpec) -> None:
    suite = _SUITES[spec.gate_id]
    case_id = (
        suite.tighten_case_id
        if spec.direction is MutationDirection.TIGHTEN
        else suite.loosen_case_id
    )
    case = next(case for case in suite.cases if case.case_id == case_id)
    result = run_mutation(spec, case)
    assert result.baseline.verdict is oracles.NumericalVerdict.OK
    assert result.hit_count > 0
    assert result.same_realization
    assert not (
        result.mutant.observed_gate_side is GateSide.ADMITTED
        and result.mutant.notes
    ), "a direct-call exception/refusal cannot be scored as admitted"
    assert result.mutant.verdict is oracles.NumericalVerdict.BAD
    assert result.killed


def _execution_signature(execution: object) -> tuple[object, ...]:
    def stable(value: object) -> object:
        if isinstance(value, np.ndarray):
            return (value.dtype.str, value.shape, value.tobytes())
        if isinstance(value, dict):
            return tuple((key, stable(item)) for key, item in sorted(value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(stable(item) for item in value)
        return value

    def check_signature(row: object) -> tuple[object, ...]:
        return (
            row.oracle,
            stable(row.actual),
            stable(row.expected),
            row.passed,
        )

    return (
        execution.expected_gate_side,
        execution.observed_gate_side,
        execution.verdict,
        execution.realization_fingerprint,
        execution.mutation_input_fingerprint,
        execution.realized_point,
        execution.axes,
        execution.direct_input_keys,
        execution.direct_return_keys,
        execution.direct_calls,
        tuple(check_signature(row) for row in execution.oracle_checks),
        tuple(
            (
                row.atom_id,
                stable(row.raw_actual),
                row.truth,
                row.reducer,
                row.realized_keys,
                row.oracle,
            )
            for row in execution.atom_evidence
        ),
        tuple(check_signature(row) for row in execution.sibling_premises),
    )


def test_transparent_wrapper_preserves_grid_and_exact_atom_evidence() -> None:
    suite = _SUITES["EAGER:array-normalization:shape-and-finiteness"]
    cases = {case.case_id: case for case in suite.cases}
    selected = (
        cases[suite.tighten_case_id],
        cases[next(iter(suite.atom_case_ids.values()))],
    )
    baseline = tuple(_execution_signature(case()) for case in selected)
    original = eager._read_only_array

    @wraps(original)
    def transparent(*args: object, **kwargs: object) -> object:
        return original(*args, **kwargs)

    with patch.object(eager, "_read_only_array", transparent):
        wrapped = tuple(_execution_signature(case()) for case in selected)

    assert wrapped == baseline


def test_structured_helpers_are_called_on_the_active_boundary_fixture() -> None:
    """Direct-call coverage cannot be padded with an unrelated easy matrix."""
    source = inspect.getsource(eager_provider._structured_runner)

    assert "helper_matrix" not in source
    suite = _SUITES["EAGER:structured:exact-shape-and-spectrum"]
    for case in suite.cases:
        execution = case()
        assert "helper_matrix" not in execution.direct_input_keys
        assert "matrix" in execution.direct_input_keys


def test_spectral_radius_oracle_does_not_share_the_production_solver() -> None:
    source = inspect.getsource(oracles.spectral_radius)

    assert "np.linalg.solve" not in source
    assert "np.linalg.eigvals" not in source


def test_power_trace_oracle_is_exact_without_the_production_linear_algebra() -> None:
    """Dense trace truth must survive a broken NumPy solve implementation."""
    lambda_matrix = np.diag(np.array([4.0, 8.0]))
    perturbation = np.diag(np.array([1.0, 4.0]))

    with patch.object(
        np.linalg,
        "solve",
        side_effect=AssertionError("shared production solve"),
    ):
        actual = oracles.exact_power_traces(lambda_matrix, perturbation, 3)

    assert actual == (0.75, 0.3125, 0.140625)
    source = inspect.getsource(oracles.exact_power_traces)
    assert "np.linalg.solve" not in source


def test_source_atom_oracle_does_not_repeat_symmetry_or_eigensolver_code() -> None:
    source = inspect.getsource(eager_provider._independent_atom_truth)

    assert "np.allclose" not in source
    assert "symmetric_eigenvalues" not in source
