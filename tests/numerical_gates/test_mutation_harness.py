"""Unit contract for source-preserving in-process gate mutations."""

from __future__ import annotations

import importlib

import pytest

import tests.numerical_gates.mutation_harness as mutation_harness_module
from tests.numerical_gates.boundary_core import GateSide, PointRole
from tests.numerical_gates.boundary_diagnose_graph import DIAGNOSE_GRAPH_SUITES
from tests.numerical_gates.mutation_harness import (
    ComparisonThresholdSide,
    MutationDirection,
    MutationSpec,
    MutationStrategy,
    run_mutation,
)
from tests.numerical_gates.mutation_specs import DIAGNOSE_GRAPH_MUTATION_SPECS
from tests.numerical_gates.registry import GATE_REGISTRY

_GRAPH_GATE = "GRAPH:_names:duplicate-multiplicity"


def test_mutation_harness_exposes_no_provider_readable_active_trace() -> None:
    """Providers may observe production results, never the grader's answer key."""
    assert not hasattr(mutation_harness_module, "active_mutation_trace")
    assert not hasattr(mutation_harness_module, "ActiveMutationTrace")


def _first_mutation_module() -> tuple[MutationSpec, object]:
    spec = DIAGNOSE_GRAPH_MUTATION_SPECS[0]
    module_path = spec.target_id.partition("::")[0]
    module = importlib.import_module(mutation_harness_module._module_name(module_path))
    return spec, module


def test_compile_mutant_restores_an_existing_none_temporary(monkeypatch) -> None:
    spec, module = _first_mutation_module()
    name = "__bayesmith_compiled_gate_mutant__"
    monkeypatch.setitem(module.__dict__, name, None)

    mutation_harness_module._compile_mutant(spec)

    assert name in module.__dict__
    assert module.__dict__[name] is None


def test_compile_mutant_restores_existing_temporary_after_compile_failure(
    monkeypatch,
) -> None:
    spec, module = _first_mutation_module()
    name = "__bayesmith_compiled_gate_mutant__"
    monkeypatch.setitem(module.__dict__, name, None)

    def fail_compile(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("injected compile failure")

    monkeypatch.setattr(
        mutation_harness_module,
        "compile",
        fail_compile,
        raising=False,
    )
    with pytest.raises(RuntimeError, match="injected compile failure"):
        mutation_harness_module._compile_mutant(spec)

    assert name in module.__dict__
    assert module.__dict__[name] is None


def _graph_compare_target() -> str:
    entry = next(item for item in GATE_REGISTRY if item.gate_id == _GRAPH_GATE)
    targets = [
        target for target in entry.source_candidate_ids if "::compare::" in target
    ]
    assert len(targets) == 1
    return targets[0]


def _graph_case(*, side: GateSide, role: PointRole):
    suite = next(item for item in DIAGNOSE_GRAPH_SUITES if item.gate_id == _GRAPH_GATE)
    return next(
        case
        for case in suite.cases
        if not case.atom_ids
        and case.threshold_point.expected_side is side
        and case.threshold_point.role is role
    )


@pytest.mark.parametrize(
    ("direction", "side", "role"),
    (
        (MutationDirection.TIGHTEN, GateSide.ADMITTED, PointRole.AT),
        (MutationDirection.LOOSEN, GateSide.REFUSED, PointRole.ABOVE_INTEGER),
    ),
)
def test_boolean_mutation_hits_the_exact_ast_and_turns_the_same_cell_bad(
    direction: MutationDirection,
    side: GateSide,
    role: PointRole,
) -> None:
    spec = MutationSpec(
        gate_id=_GRAPH_GATE,
        direction=direction,
        target_id=_graph_compare_target(),
        strategy=MutationStrategy.SHIFT_COMPARISON,
        true_side=GateSide.REFUSED,
        threshold_side=ComparisonThresholdSide.RIGHT,
    )

    result = run_mutation(spec, _graph_case(side=side, role=role))

    assert result.baseline.verdict.value == "OK"
    assert result.hit_count > 0
    assert result.same_realization
    assert result.killed


def test_mutation_rejects_a_witness_from_a_different_gate() -> None:
    suite = next(
        item
        for item in DIAGNOSE_GRAPH_SUITES
        if item.gate_id == "COUPLING:_classify_correlation:value-finite"
    )
    case = next(item for item in suite.cases if not item.atom_ids)
    spec = MutationSpec(
        gate_id=_GRAPH_GATE,
        direction=MutationDirection.TIGHTEN,
        target_id=_graph_compare_target(),
        strategy=MutationStrategy.FLIP_BOOLEAN,
    )

    with pytest.raises(ValueError, match="belongs to"):
        run_mutation(spec, case)


def test_directional_mutation_declarations_require_their_gate_semantics() -> None:
    with pytest.raises(ValueError, match="predicate=True"):
        MutationSpec(
            gate_id=_GRAPH_GATE,
            direction=MutationDirection.TIGHTEN,
            target_id=_graph_compare_target(),
            strategy=MutationStrategy.FORCE_GATE_SIDE,
        )
    with pytest.raises(ValueError, match="threshold operand"):
        MutationSpec(
            gate_id=_GRAPH_GATE,
            direction=MutationDirection.TIGHTEN,
            target_id=_graph_compare_target(),
            strategy=MutationStrategy.SHIFT_COMPARISON,
            true_side=GateSide.REFUSED,
        )
    with pytest.raises(ValueError, match="replacement"):
        MutationSpec(
            gate_id=_GRAPH_GATE,
            direction=MutationDirection.TIGHTEN,
            target_id=_graph_compare_target(),
            strategy=MutationStrategy.REPLACE_COMPARISON_THRESHOLD,
            true_side=GateSide.REFUSED,
            threshold_side=ComparisonThresholdSide.RIGHT,
        )


@pytest.mark.parametrize(
    ("direction", "side", "role", "replacement"),
    (
        (MutationDirection.TIGHTEN, GateSide.ADMITTED, PointRole.AT, 0),
        (MutationDirection.LOOSEN, GateSide.REFUSED, PointRole.ABOVE_INTEGER, 2),
    ),
)
def test_explicit_comparison_threshold_crosses_a_quantized_witness(
    direction: MutationDirection,
    side: GateSide,
    role: PointRole,
    replacement: int,
) -> None:
    spec = MutationSpec(
        gate_id=_GRAPH_GATE,
        direction=direction,
        target_id=_graph_compare_target(),
        strategy=MutationStrategy.REPLACE_COMPARISON_THRESHOLD,
        true_side=GateSide.REFUSED,
        threshold_side=ComparisonThresholdSide.RIGHT,
        threshold_replacement=replacement,
    )

    result = run_mutation(spec, _graph_case(side=side, role=role))

    assert result.hit_count > 0
    assert result.same_realization
    assert result.killed


def test_diagnose_graph_uses_two_distinct_directional_mutation_families() -> None:
    assert len(DIAGNOSE_GRAPH_MUTATION_SPECS) == 28
    assert all(
        spec.strategy is not MutationStrategy.FLIP_BOOLEAN
        for spec in DIAGNOSE_GRAPH_MUTATION_SPECS
    )


@pytest.mark.parametrize(
    "spec",
    tuple(
        item
        for item in DIAGNOSE_GRAPH_MUTATION_SPECS
        if item.gate_id == "MAP:map_estimate:curvature-scale-clamp"
    ),
    ids=lambda item: item.direction.value,
)
def test_clamp_mutation_changes_the_real_map_result_kind(spec: MutationSpec) -> None:
    """An internal clamp value alone is not an observable mutation kill."""
    suite = next(
        item for item in DIAGNOSE_GRAPH_SUITES if item.gate_id == spec.gate_id
    )
    case_id = (
        suite.tighten_case_id
        if spec.direction is MutationDirection.TIGHTEN
        else suite.loosen_case_id
    )
    case = next(item for item in suite.cases if item.case_id == case_id)

    result = run_mutation(spec, case)

    assert result.baseline.direct_return_values["result_kind"] != (
        result.mutant.direct_return_values["result_kind"]
    )


@pytest.mark.parametrize(
    "spec",
    DIAGNOSE_GRAPH_MUTATION_SPECS,
    ids=lambda item: f"{item.gate_id}-{item.direction.value}",
)
def test_every_diagnose_graph_gate_mutation_turns_its_frozen_witness_bad(
    spec: MutationSpec,
) -> None:
    suite = next(
        item for item in DIAGNOSE_GRAPH_SUITES if item.gate_id == spec.gate_id
    )
    case_id = (
        suite.tighten_case_id
        if spec.direction is MutationDirection.TIGHTEN
        else suite.loosen_case_id
    )
    case = next(item for item in suite.cases if item.case_id == case_id)

    result = run_mutation(spec, case)

    assert result.baseline.verdict.value == "OK"
    assert result.hit_count > 0
    assert result.same_realization
    assert result.killed
