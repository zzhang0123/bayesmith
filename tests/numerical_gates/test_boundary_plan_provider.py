"""Focused behavioral contract for the direct PLAN boundary provider."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import wraps
from importlib import import_module
from unittest.mock import patch

import numpy as np
import pytest

from tests.numerical_gates import oracles
from tests.numerical_gates.boundary_contract import REQUIRED_DIRECT_CALLS
from tests.numerical_gates.boundary_core import (
    AxisPosition,
    GateSide,
    PointRole,
    source_alias_canonical,
)
from tests.numerical_gates.mutation_harness import (
    MutationDirection,
    MutationSpec,
    run_mutation,
)
from tests.numerical_gates.mutation_specs import PLAN_MUTATION_SPECS
from tests.numerical_gates.registry import (
    GATE_REGISTRY,
    MutationMode,
    isolatable_atom_ids,
)

pytestmark = pytest.mark.full


_PLAN_ENTRIES = {
    entry.gate_id: entry
    for entry in GATE_REGISTRY
    if entry.gate_id.startswith("PLAN:")
    and entry.mutation_mode is MutationMode.TWO_SIDED
}


def _raw_fingerprint(value: object) -> tuple[str, tuple[int, ...], bytes]:
    array = np.asarray(value)
    return array.dtype.str, array.shape, array.tobytes()


def _suite(provider: object, gate_id: str) -> object:
    return next(suite for suite in provider.PLAN_SUITES if suite.gate_id == gate_id)


def _case(suite: object, case_id: str) -> object:
    return next(case for case in suite.cases if case.case_id == case_id)


def _raw_evidence_signature(value: object) -> tuple[object, ...]:
    array = np.asarray(value)
    if array.dtype.hasobject:
        payload: object = repr(value)
    else:
        payload = array.tobytes()
    return array.dtype.str, array.shape, payload


def _execution_signature(execution: object) -> tuple[object, ...]:
    return (
        execution.realization_fingerprint,
        execution.mutation_input_fingerprint,
        execution.expected_gate_side,
        execution.observed_gate_side,
        execution.verdict,
        execution.direct_calls,
        tuple(
            (
                evidence.atom_id,
                _raw_evidence_signature(evidence.raw_actual),
                evidence.truth,
                evidence.reducer,
                evidence.realized_keys,
            )
            for evidence in execution.atom_evidence
        ),
    )


def _source_owner(provider: object, gate_id: str) -> tuple[object, str]:
    target_id = _PLAN_ENTRIES[gate_id].mutation_target_ids[0]
    qualname = target_id.split("::", 2)[1]
    return provider._qualname_owner(qualname)


def test_central_plan_mutation_specs_exactly_cover_two_sided_plan_gates() -> None:
    expected = {
        (gate_id, direction)
        for gate_id in _PLAN_ENTRIES
        for direction in MutationDirection
    }
    actual = {(spec.gate_id, spec.direction) for spec in PLAN_MUTATION_SPECS}

    assert len(PLAN_MUTATION_SPECS) == len(expected) == 58
    assert actual == expected
    assert all(
        spec.target_id in _PLAN_ENTRIES[spec.gate_id].mutation_target_ids
        for spec in PLAN_MUTATION_SPECS
    )


def test_multiplicity_witnesses_use_an_independent_frozen_threshold() -> None:
    provider = import_module("tests.numerical_gates.boundary_plan")
    product = import_module("bayesmith.marginal._logdet_plan")
    suite = _suite(provider, "PLAN:multiplicity:index-and-gamma-domain")
    tighten = _case(suite, suite.tighten_case_id)
    loosen = _case(suite, suite.loosen_case_id)
    original = product._RHO_MULTIPLICITY_LIMIT

    assert tighten().verdict is oracles.NumericalVerdict.OK
    assert loosen().verdict is oracles.NumericalVerdict.OK
    with patch.object(product, "_RHO_MULTIPLICITY_LIMIT", original // 2):
        assert tighten().verdict is oracles.NumericalVerdict.BAD
    with patch.object(product, "_RHO_MULTIPLICITY_LIMIT", original * 2):
        assert loosen().verdict is oracles.NumericalVerdict.BAD


def test_strict_rho_call_is_a_static_dependency_not_a_directional_return_gate() -> None:
    provider = import_module("tests.numerical_gates.boundary_plan")
    product = import_module("bayesmith.marginal._logdet_plan")
    eager = import_module("bayesmith.marginal._logdet_eager")
    gate_id = "PLAN:factory-certificate:strict-rho"
    certificate = product.RhoCertificate(
        measured_max=0.1,
        margin=0.0,
        certified_rho=0.5,
        order=oracles.smallest_trace_order(0.5, 0.25, 2),
        tolerance=0.5,
        tail_tolerance=0.25,
        multiplicity=2,
        max_abs_lambda_logdet=0.0,
        max_x_operator_norm=None,
    )
    problem = eager.LogDetProblem(
        np.eye(2),
        np.diag([0.5, -0.1]),
        trace_order=oracles.smallest_trace_order(0.5, 0.25, 2),
        certified_rho=0.5,
    )

    assert gate_id not in {suite.gate_id for suite in provider.PLAN_SUITES}
    assert gate_id not in {spec.gate_id for spec in PLAN_MUTATION_SPECS}
    with patch.object(
        product, "_validate_strict_rho", return_value=(math.inf, math.inf)
    ):
        assert product._validate_plan_certificate(problem, certificate) is None
    with (
        patch.object(
            product,
            "_validate_strict_rho",
            side_effect=ValueError("strict-rho dependency refused"),
        ),
        pytest.raises(ValueError, match="strict-rho dependency refused"),
    ):
        product._validate_plan_certificate(problem, certificate)


def test_lambda_logdet_finite_guard_is_static_not_a_fake_directional_gate() -> None:
    provider = import_module("tests.numerical_gates.boundary_plan")
    product = import_module("bayesmith.marginal._logdet_plan")
    gate_id = "PLAN:measurement:lambda-logdet-finite"
    entry = next(entry for entry in GATE_REGISTRY if entry.gate_id == gate_id)

    assert entry.mutation_mode is MutationMode.STATIC_ONLY
    assert "3.8e36" in entry.static_reason
    assert "2.5e305" in entry.static_reason
    assert gate_id not in {suite.gate_id for suite in provider.PLAN_SUITES}
    assert gate_id not in {spec.gate_id for spec in PLAN_MUTATION_SPECS}
    assert product._checked_lambda_logdet_scale(np.array([2.0])) == pytest.approx(
        math.log(2.0)
    )
    for invalid in (
        np.array([0.0]),
        np.array([math.inf]),
        np.diag([2.0, -1.0]),
    ):
        with pytest.raises(ValueError, match="finite resolved"):
            product._checked_lambda_logdet_scale(invalid)


def test_error_budget_realizes_every_parameter_axis_and_nonfinite_field() -> None:
    provider = import_module("tests.numerical_gates.boundary_plan")
    suite = _suite(provider, "PLAN:certificate:error-budget-domain")
    executions = [case() for case in suite.cases if not case.atom_ids]
    by_axis: dict[str, list[object]] = {
        "margin": [],
        "tolerance": [],
        "tail_tolerance": [],
    }
    for execution in executions:
        axis = next(
            axis
            for axis in execution.realized_point.axes
            if axis.axis_name == execution.realized_point.active_axis
        )
        if axis.input_key in by_axis:
            by_axis[axis.input_key].append(execution)

    required_positions = set(AxisPosition) - {AxisPosition.INTERIOR}
    for field, rows in by_axis.items():
        assert {
            next(
                axis
                for axis in row.realized_point.axes
                if axis.axis_name == row.realized_point.active_axis
            ).position
            for row in rows
        } == required_positions, field
        values = [
            next(
                axis
                for axis in row.realized_point.axes
                if axis.axis_name == row.realized_point.active_axis
            ).value
            for row in rows
        ]
        assert any(math.isnan(float(value)) for value in values), field
        assert any(float(value) == math.inf for value in values), field
        assert any(float(value) == -math.inf for value in values), field

    nonfinite_sides = {
        (row.realized_point.active_axis, float(row.realized_point.value)): (
            row.observed_gate_side,
            row.verdict,
        )
        for row in executions
        if row.realized_point.active_axis in by_axis
        and math.isinf(float(row.realized_point.value))
    }
    assert nonfinite_sides[("margin", math.inf)] == (
        GateSide.ADMITTED,
        oracles.NumericalVerdict.OK,
    )
    assert nonfinite_sides[("margin", -math.inf)][0] is GateSide.REFUSED
    assert nonfinite_sides[("tolerance", math.inf)] == (
        GateSide.ADMITTED,
        oracles.NumericalVerdict.OK,
    )
    assert nonfinite_sides[("tolerance", -math.inf)][0] is GateSide.REFUSED
    assert nonfinite_sides[("tail_tolerance", math.inf)][0] is GateSide.REFUSED
    assert nonfinite_sides[("tail_tolerance", -math.inf)][0] is GateSide.REFUSED


@pytest.mark.parametrize(
    ("gate_id", "axis_keys"),
    (
        (
            "PLAN:certificate:optional-scale-domain",
            ("max_abs_lambda_logdet", "max_x_operator_norm"),
        ),
        (
            "PLAN:warmup:lambda-scale-inputs",
            ("lambda_value", "lambda_logdet_margin"),
        ),
        (
            "PLAN:warmup:x-norm-inputs",
            ("x_norm_value", "x_operator_norm_margin"),
        ),
        (
            "PLAN:audit:retained-trace-evidence",
            ("problem_trace_order", "perturbation", "trace_evidence_value"),
        ),
        (
            "PLAN:factory-certificate:order-and-rank",
            ("problem_trace_order", "perturbation"),
        ),
        (
            "PLAN:frozen:probe-energy-range",
            ("probe_component", "probe_count", "runtime_dtype"),
        ),
        ("PLAN:runtime-range:product", ("left", "right", "maximum")),
        ("PLAN:runtime-range:sum", ("left", "right", "maximum")),
        (
            "PLAN:frozen:intermediate-runtime-range",
            ("total_probe_energy", "order"),
        ),
        (
            "PLAN:runtime:expected-and-ulp-finite",
            ("lambda_entry", "perturbation_entry", "runtime_dtype", "tolerance"),
        ),
        (
            "PLAN:runtime:total-error-budget",
            ("certified_rho", "max_abs_lambda_logdet", "tolerance"),
        ),
        (
            "PLAN:trace-factory:exact-evidence",
            ("problem_trace_order", "trace_evidence_value"),
        ),
        (
            "PLAN:frozen-factory:probe-presence-width",
            ("probe_presence", "probe_width"),
        ),
        (
            "PLAN:certificate:order-is-derived",
            ("order", "certified_rho", "tail_tolerance", "multiplicity"),
        ),
        ("PLAN:audit:retained-rho", ("retained_value", "certified_rho")),
        (
            "PLAN:audit:retained-lambda-scale",
            ("retained_value", "max_abs_lambda_logdet"),
        ),
        (
            "PLAN:audit:retained-x-norm",
            ("retained_value", "max_x_operator_norm"),
        ),
        (
            "PLAN:factory-certificate:lambda-scale",
            ("lambda_matrix", "max_abs_lambda_logdet"),
        ),
        (
            "PLAN:factory-certificate:x-norm",
            ("perturbation", "max_x_operator_norm"),
        ),
        (
            "PLAN:gamma:operation-count-domain",
            ("operation_count", "epsilon"),
        ),
        (
            "PLAN:outward-arithmetic:positive-underflow",
            ("proof_value",),
        ),
        (
            "PLAN:measurement:x-norm-finite",
            ("lambda_entry", "perturbation_entry", "matrix_path"),
        ),
        (
            "PLAN:frozen:x-bound-runtime-range",
            ("max_x_operator_norm", "runtime_dtype"),
        ),
        (
            "PLAN:runtime:base-scale-range",
            ("max_abs_lambda_logdet", "runtime_dtype"),
        ),
        (
            "PLAN:runtime:sigma-finite-and-positive",
            ("lambda_entry", "perturbation_entry", "matrix_path"),
        ),
        (
            "PLAN:runtime:frozen-prerequisites-and-series",
            ("max_x_operator_norm", "probe_component", "order"),
        ),
    ),
)
def test_compound_gates_realize_five_positions_on_each_true_input_axis(
    gate_id: str, axis_keys: tuple[str, ...]
) -> None:
    provider = import_module("tests.numerical_gates.boundary_plan")
    suite = _suite(provider, gate_id)
    executions = [case() for case in suite.cases if not case.atom_ids]
    required_positions = set(AxisPosition) - {AxisPosition.INTERIOR}

    for axis_key in axis_keys:
        rows = [row for row in executions if row.realized_point.active_axis == axis_key]
        assert {
            next(
                axis
                for axis in row.realized_point.axes
                if axis.axis_name == row.realized_point.active_axis
            ).position
            for row in rows
        } == required_positions, (gate_id, axis_key)


@pytest.mark.parametrize(
    ("gate_id", "axis_keys"),
    (
        (
            "PLAN:certificate:optional-scale-domain",
            ("max_abs_lambda_logdet", "max_x_operator_norm"),
        ),
        (
            "PLAN:warmup:lambda-scale-inputs",
            ("lambda_value", "lambda_logdet_margin"),
        ),
        (
            "PLAN:warmup:x-norm-inputs",
            ("x_norm_value", "x_operator_norm_margin"),
        ),
        ("PLAN:runtime-range:product", ("left", "right", "maximum")),
        ("PLAN:runtime-range:sum", ("left", "right", "maximum")),
        (
            "PLAN:audit:retained-rho",
            ("retained_value",),
        ),
        (
            "PLAN:audit:retained-lambda-scale",
            ("retained_value",),
        ),
        (
            "PLAN:audit:retained-x-norm",
            ("retained_value",),
        ),
        (
            "PLAN:measurement:x-norm-finite",
            ("lambda_entry", "perturbation_entry"),
        ),
    ),
)
def test_declared_nonfinite_compound_fields_execute_nan_and_both_infinities(
    gate_id: str, axis_keys: tuple[str, ...]
) -> None:
    provider = import_module("tests.numerical_gates.boundary_plan")
    suite = _suite(provider, gate_id)
    executions = [case() for case in suite.cases if not case.atom_ids]

    for axis_key in axis_keys:
        rows = [row for row in executions if row.realized_point.active_axis == axis_key]
        values = [row.realized_point.value for row in rows]
        assert any(math.isnan(float(value)) for value in values), (
            gate_id,
            axis_key,
        )
        assert any(float(value) == math.inf for value in values), (
            gate_id,
            axis_key,
        )
        assert any(float(value) == -math.inf for value in values), (
            gate_id,
            axis_key,
        )


def test_runtime_capability_grid_uses_five_distinct_consumed_inputs() -> None:
    provider = import_module("tests.numerical_gates.boundary_plan")
    suite = _suite(provider, "PLAN:runtime-call:scalar-and-dtype")
    roles = {
        PointRole.CAPABILITY_LOW,
        PointRole.VALID_CAPABILITY,
        PointRole.INVALID_CAPABILITY,
        PointRole.CAPABILITY_HIGH,
        PointRole.EXTREME,
    }
    executions = [
        case()
        for case in suite.cases
        if not case.atom_ids and case.threshold_point.role in roles
    ]

    assert {row.threshold_point.role for row in executions} == roles
    assert {row.realized_point.value for row in executions} == {
        role.value for role in roles
    }
    assert len({row.mutation_input_fingerprint for row in executions}) == len(roles)


def test_live_outer_mutant_is_preserved_and_keeps_the_same_realization() -> None:
    provider = import_module("tests.numerical_gates.boundary_plan")
    product = import_module("bayesmith.marginal._logdet_plan")
    suite = _suite(provider, "PLAN:multiplicity:index-and-gamma-domain")
    tighten = _case(suite, suite.tighten_case_id)
    baseline = tighten()
    original = product._normalize_rho_multiplicity
    hit_count = 0

    def outer_mutant(value: object) -> int:
        nonlocal hit_count
        hit_count += 1
        if int(value) >= product._RHO_MULTIPLICITY_LIMIT // 2:
            raise ValueError("outer tightened threshold")
        return original(value)

    with patch.object(product, "_normalize_rho_multiplicity", outer_mutant):
        mutant = tighten()

    assert hit_count > 0
    assert _raw_fingerprint(mutant.realized_point.value) == _raw_fingerprint(
        baseline.realized_point.value
    )
    assert mutant.realized_point.threshold == baseline.realized_point.threshold
    assert mutant.direct_input_keys == baseline.direct_input_keys
    assert mutant.verdict is oracles.NumericalVerdict.BAD


def test_finite_gates_reach_last_finite_first_overflow_and_subnormal() -> None:
    provider = import_module("tests.numerical_gates.boundary_plan")
    suite = _suite(provider, "PLAN:measurement:x-norm-finite")
    observations = [
        case.runner(case.threshold_point) for case in suite.cases if not case.atom_ids
    ]
    direct_values = [
        value
        for observation in observations
        for key in ("lambda_entry", "perturbation_entry")
        if key in observation.realized_inputs
        for value in (observation.realized_inputs[key],)
        if isinstance(value, (float, np.floating))
    ]
    assert np.finfo(np.float64).max in direct_values
    assert any(
        0.0 < abs(float(value)) < np.finfo(np.float64).tiny for value in direct_values
    )
    assert any(math.isinf(float(value)) for value in direct_values)

    suite = _suite(provider, "PLAN:runtime:frozen-prerequisites-and-series")
    observations = [
        case.runner(case.threshold_point) for case in suite.cases if not case.atom_ids
    ]
    assert all("series_target" not in row.direct_input_keys for row in observations)
    assert any(
        row.realized_inputs["max_x_operator_norm"] == np.finfo(np.float64).max
        for row in observations
    )
    assert any(
        row.realized_inputs["max_x_operator_norm"] == np.nextafter(0.0, math.inf)
        for row in observations
    )
    assert any(
        math.isinf(float(row.realized_inputs["resolved_series_scale"]))
        for row in observations
    )


def test_finite_gate_tightening_rejects_each_last_finite_witness() -> None:
    provider = import_module("tests.numerical_gates.boundary_plan")
    product = import_module("bayesmith.marginal._logdet_plan")
    suite = _suite(provider, "PLAN:measurement:x-norm-finite")
    tighten = _case(suite, suite.tighten_case_id)
    original = product._checked_x_operator_norm

    def tightened(*args: object) -> float:
        result = original(*args)
        if result > 10.0:
            raise ValueError("tightened finite ceiling")
        return result

    with patch.object(product, "_checked_x_operator_norm", tightened):
        assert tighten().verdict is oracles.NumericalVerdict.BAD

    suite = _suite(provider, "PLAN:runtime:frozen-prerequisites-and-series")
    tighten = next(
        case
        for case in suite.cases
        if not case.atom_ids
        and case.active_axis == "max_x_operator_norm"
        and case.threshold_point.role is PointRole.VERY_HIGH
    )
    observation = tighten.runner(tighten.threshold_point)
    assert observation.realized_inputs["resolved_series_scale"] > 10.0
    original_runtime = product._validate_runtime_precision
    original_isfinite = product.np.isfinite

    def tightened_isfinite(value: object) -> object:
        result = original_isfinite(value)
        values = np.asarray(value)
        if values.shape == () and float(values) > 10.0:
            return np.bool_(False)
        return result

    def tightened_runtime(*args: object, **kwargs: object) -> object:
        with patch.object(product.np, "isfinite", side_effect=tightened_isfinite):
            return original_runtime(*args, **kwargs)

    with patch.object(product, "_validate_runtime_precision", tightened_runtime):
        assert tighten().verdict is oracles.NumericalVerdict.BAD


def test_expected_nonfinite_roots_are_resource_ambiguities_not_dynamic_witnesses() -> (
    None
):
    provider = import_module("tests.numerical_gates.boundary_plan")
    suite = _suite(provider, "PLAN:runtime:expected-and-ulp-finite")
    rendered = "\n".join(suite.ambiguities)
    tighten = _case(suite, suite.tighten_case_id)
    loosen = _case(suite, suite.loosen_case_id)

    assert "4.8e35" in rendered
    assert "2.5e305" in rendered
    assert "expected/rounded nonfinite roots are resource-unreachable" in rendered
    assert "dynamic witnesses isolate only ulp > tolerance" in rendered
    assert tighten.active_axis == loosen.active_axis == "tolerance"


def test_transparent_live_wrappers_do_not_manufacture_bad_evidence() -> None:
    provider = import_module("tests.numerical_gates.boundary_plan")
    product = import_module("bayesmith.marginal._logdet_plan")
    suite = _suite(provider, "PLAN:multiplicity:index-and-gamma-domain")
    atom_case = next(case for case in suite.cases if case.atom_ids)
    original_normalize = product._normalize_rho_multiplicity

    def transparent_normalize(value: object) -> int:
        return original_normalize(value)

    with patch.object(product, "_normalize_rho_multiplicity", transparent_normalize):
        assert atom_case().verdict is oracles.NumericalVerdict.OK

    suite = _suite(provider, "PLAN:frozen-factory:probe-presence-width")
    cases = {case.case_id: case for case in suite.cases}
    original_factory = product.make_frozen_trace_log_plan

    def transparent_factory(*args: object, **kwargs: object) -> object:
        return original_factory(*args, **kwargs)

    with patch.object(product, "make_frozen_trace_log_plan", transparent_factory):
        assert cases[suite.tighten_case_id]().verdict is oracles.NumericalVerdict.OK
        assert cases[suite.loosen_case_id]().verdict is oracles.NumericalVerdict.OK


def test_transparent_wrappers_preserve_every_grid_atom_and_raw_signature() -> None:
    provider = import_module("tests.numerical_gates.boundary_plan")
    for suite in provider.PLAN_SUITES:
        baseline = tuple(_execution_signature(case()) for case in suite.cases)
        owner, attribute = _source_owner(provider, suite.gate_id)
        original = getattr(owner, attribute)

        @wraps(original)
        def transparent(
            *args: object,
            _original: Callable[..., object] = original,
            **kwargs: object,
        ) -> object:
            return _original(*args, **kwargs)

        with patch.object(owner, attribute, transparent):
            wrapped = tuple(_execution_signature(case()) for case in suite.cases)
        assert wrapped == baseline, suite.gate_id


def test_transparent_wrappers_still_execute_every_source_atom_recorder() -> None:
    provider = import_module("tests.numerical_gates.boundary_plan")
    for suite in provider.PLAN_SUITES:
        atom_cases = [case for case in suite.cases if case.atom_ids]
        if not atom_cases:
            continue
        owner, attribute = _source_owner(provider, suite.gate_id)
        original_callable = getattr(owner, attribute)
        original_recorder = provider._record_plan_atom
        recorded: list[str] = []

        @wraps(original_callable)
        def transparent(
            *args: object,
            _original: Callable[..., object] = original_callable,
            **kwargs: object,
        ) -> object:
            return _original(*args, **kwargs)

        def retain(
            atom_id: str,
            raw: object,
            *,
            _recorded: list[str] = recorded,
            _original_recorder: Callable[[str, object], object] = original_recorder,
        ) -> object:
            _recorded.append(atom_id)
            return _original_recorder(atom_id, raw)

        with (
            patch.object(owner, attribute, transparent),
            patch.object(provider, "_record_plan_atom", side_effect=retain),
        ):
            executions = [case() for case in atom_cases]
        assert recorded, suite.gate_id
        expected_atoms = {
            source_alias_canonical(_PLAN_ENTRIES[suite.gate_id], case.atom_ids[0])
            for case in atom_cases
        }
        assert expected_atoms <= set(recorded), suite.gate_id
        assert all(row.verdict is oracles.NumericalVerdict.OK for row in executions)


def test_frozen_factory_loosen_witness_reaches_its_width_predicate() -> None:
    provider = import_module("tests.numerical_gates.boundary_plan")
    product = import_module("bayesmith.marginal._logdet_plan")
    suite = _suite(provider, "PLAN:frozen-factory:probe-presence-width")
    loosen = _case(suite, suite.loosen_case_id)

    with patch.object(
        product,
        "_validate_plan_certificate",
        side_effect=ValueError("earlier same-type failure"),
    ):
        assert loosen().verdict is oracles.NumericalVerdict.BAD


def test_multiplicity_grid_executes_every_registry_extreme() -> None:
    provider = import_module("tests.numerical_gates.boundary_plan")
    suite = _suite(provider, "PLAN:multiplicity:index-and-gamma-domain")
    extremes = [
        case().realized_point.value
        for case in suite.cases
        if not case.atom_ids and case.threshold_point.role is PointRole.EXTREME
    ]

    assert any(type(value) is bool for value in extremes)
    assert any(type(value) is float for value in extremes)
    assert 10**1000 in extremes


def test_outward_side_is_observed_from_the_production_return() -> None:
    provider = import_module("tests.numerical_gates.boundary_plan")
    product = import_module("bayesmith.marginal._logdet_plan")
    suite = _suite(provider, "PLAN:outward-arithmetic:positive-underflow")
    positive = next(
        case
        for case in suite.cases
        if not case.atom_ids and case.threshold_point.role is PointRole.ABOVE_ULP
    )
    original = product._outward_nonnegative

    def preserve_positive(value: float) -> float:
        if value > 0.0:
            return value
        return original(value)

    with patch.object(product, "_outward_nonnegative", side_effect=preserve_positive):
        execution = positive()
        assert execution.observed_gate_side is GateSide.ADMITTED
        assert execution.verdict is oracles.NumericalVerdict.BAD


@pytest.mark.parametrize(
    "spec",
    PLAN_MUTATION_SPECS,
    ids=lambda spec: f"{spec.gate_id}-{spec.direction.value}",
)
def test_every_plan_gate_mutation_turns_its_reserved_witness_bad(
    spec: MutationSpec,
) -> None:
    provider = import_module("tests.numerical_gates.boundary_plan")
    suite = _suite(provider, spec.gate_id)
    case_id = (
        suite.tighten_case_id
        if spec.direction is MutationDirection.TIGHTEN
        else suite.loosen_case_id
    )
    result = run_mutation(spec, _case(suite, case_id))

    assert result.baseline.verdict is oracles.NumericalVerdict.OK
    assert result.hit_count > 0
    assert result.same_realization
    assert result.mutant.verdict is oracles.NumericalVerdict.BAD
    assert result.killed


def test_plan_provider_builds_the_exact_executable_registry_partition() -> None:
    """Catch an omitted PLAN gate, atom, or non-executable direct fixture."""
    try:
        provider = import_module("tests.numerical_gates.boundary_plan")
    except Exception as error:  # noqa: BLE001  # pragma: no cover - RED state
        pytest.fail(f"PLAN provider does not import: {error!r}")

    entries = {
        entry.gate_id: entry
        for entry in GATE_REGISTRY
        if entry.gate_id.startswith("PLAN:")
        and entry.mutation_mode is MutationMode.TWO_SIDED
    }
    expected = {gate_id: set(isolatable_atom_ids(entry)) for gate_id, entry in entries.items()}
    expected_evidence = {
        gate_id: set(entry.conjunction_atom_ids) for gate_id, entry in entries.items()
    }
    suites = {suite.gate_id: suite for suite in provider.PLAN_SUITES}

    assert len(expected) == 29
    assert sum(map(len, expected.values())) == 74
    assert set(suites) == set(expected)
    assert sum(len(suite.atom_case_ids) for suite in suites.values()) == 74
    assert all(
        set(suites[gate_id].atom_case_ids) == atoms
        for gate_id, atoms in expected.items()
    )

    for gate_id, suite in suites.items():
        cases_by_id = {case.case_id: case for case in suite.cases}
        assert (
            cases_by_id[suite.tighten_case_id].threshold_point.expected_side
            is GateSide.ADMITTED
        )
        assert (
            cases_by_id[suite.loosen_case_id].threshold_point.expected_side
            is GateSide.REFUSED
        )
        for case in suite.cases:
            execution = case()
            assert execution.observed_gate_side is execution.expected_gate_side
            assert execution.verdict is oracles.NumericalVerdict.OK
            assert set(execution.direct_calls) == REQUIRED_DIRECT_CALLS[gate_id]
            assert set(execution.direct_calls) == set(case.direct_methods)
            assert execution.oracle_checks or execution.evaluations
            assert all(check.passed for check in execution.oracle_checks)
            assert all(check.passed for check in execution.sibling_premises)
            assert all(
                evaluation.verdict is oracles.NumericalVerdict.OK
                for evaluation in execution.evaluations
            )

            if not case.atom_ids:
                assert execution.isolated_atom is None
                assert not execution.atom_evidence
                continue

            assert execution.isolated_atom == case.atom_ids[0]
            assert execution.atom_relation == case.atom_relation
            evidence_by_id = {
                evidence.atom_id: evidence for evidence in execution.atom_evidence
            }
            assert set(evidence_by_id) == expected_evidence[gate_id]
            assert all(evidence.realized_keys for evidence in evidence_by_id.values())
            for atom_id, evidence in evidence_by_id.items():
                canonical_id = source_alias_canonical(entries[gate_id], atom_id)
                canonical = evidence_by_id[canonical_id]
                assert _raw_fingerprint(evidence.raw_actual) == _raw_fingerprint(
                    canonical.raw_actual
                )
                assert evidence.truth is canonical.truth
                assert evidence.reducer is canonical.reducer


def test_plan_cases_do_not_publish_harness_control_knobs_as_direct_inputs() -> None:
    """Every realized PLAN input must shape the real production call."""
    provider = import_module("tests.numerical_gates.boundary_plan")
    forbidden_exact = {
        "rank_target",
        "scale_target",
        "x_target",
        "correction_target",
        "addition_factor",
        "analytic_tail",
        "roundoff_error",
    }

    for suite in provider.PLAN_SUITES:
        for case in suite.cases:
            observation = case.runner(case.threshold_point)
            forbidden = {
                key
                for key in observation.direct_input_keys
                if key in forbidden_exact or key.startswith(("fault_", "force_"))
            }
            assert not forbidden, (suite.gate_id, case.case_id, sorted(forbidden))


@pytest.mark.parametrize(
    "gate_id",
    (
        "PLAN:factory-certificate:order-and-rank",
        "PLAN:factory-certificate:lambda-scale",
        "PLAN:factory-certificate:x-norm",
    ),
)
def test_factory_certificate_cases_measure_their_real_problem(
    gate_id: str,
) -> None:
    """A selected witness cannot substitute a scalar for a matrix measurement."""
    provider = import_module("tests.numerical_gates.boundary_plan")
    suite = _suite(provider, gate_id)
    selected = {suite.tighten_case_id, suite.loosen_case_id}

    for case in suite.cases:
        if case.case_id not in selected:
            continue
        observation = case.runner(case.threshold_point)
        lambda_matrix = np.asarray(observation.realized_inputs["lambda_matrix"])
        perturbation = np.asarray(observation.realized_inputs["perturbation"])
        if gate_id.endswith("order-and-rank"):
            expected = int(np.count_nonzero(perturbation))
            assert observation.realized_inputs["required_multiplicity"] == expected
        elif gate_id.endswith("lambda-scale"):
            expected = abs(
                math.fsum(math.log(float(value)) for value in lambda_matrix.ravel())
            )
            assert observation.realized_inputs["actual_base_scale"] == expected
        else:
            lambda_diagonal = (
                lambda_matrix if lambda_matrix.ndim == 1 else np.diag(lambda_matrix)
            )
            perturbation_diagonal = (
                perturbation if perturbation.ndim == 1 else np.diag(perturbation)
            )
            ratios = np.abs(perturbation_diagonal / lambda_diagonal)
            expected = float(np.max(ratios, initial=0.0))
            assert observation.realized_inputs["actual_x_norm"] == expected


def test_error_budget_companion_uses_the_real_order_derivation() -> None:
    """The tail-domain witness must not replace choose_trace_order downstream."""
    provider = import_module("tests.numerical_gates.boundary_plan")
    product = import_module("bayesmith.marginal._logdet_plan")
    suite = _suite(provider, "PLAN:certificate:error-budget-domain")
    loosen = _case(suite, suite.loosen_case_id)
    original = product.choose_trace_order
    calls: list[tuple[float, float, int]] = []

    def recording_order(
        rho: float, tolerance: float, *, multiplicity: int
    ) -> int:
        calls.append((rho, tolerance, multiplicity))
        return original(rho, tolerance, multiplicity=multiplicity)

    with patch.object(product, "choose_trace_order", side_effect=recording_order):
        result = run_mutation(
            next(
                spec
                for spec in PLAN_MUTATION_SPECS
                if spec.gate_id == "PLAN:certificate:error-budget-domain"
                and spec.direction is MutationDirection.LOOSEN
            ),
            loosen,
        )

    assert result.killed
    assert calls
    assert calls[-1] == (0.2, 0.5, 2)
