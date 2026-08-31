"""Independent static and declaration audits for Task-3 providers."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.numerical_gates.boundary_contract import (
    ALLOWED_DIRECT_CALLS,
    REQUIRED_DIRECT_CALLS,
)
from tests.numerical_gates.registry import GATE_REGISTRY, MutationMode

_PROVIDERS = (
    Path(__file__).with_name("boundary_eager.py"),
    Path(__file__).with_name("boundary_ladder.py"),
    Path(__file__).with_name("boundary_plan.py"),
    Path(__file__).with_name("boundary_diagnose_graph.py"),
)
_TWO_SIDED_GATE_IDS = tuple(
    sorted(
        entry.gate_id
        for entry in GATE_REGISTRY
        if entry.mutation_mode is MutationMode.TWO_SIDED
    )
)


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def test_direct_call_contract_covers_exactly_the_two_sided_registry() -> None:
    required_gates = {
        entry.gate_id
        for entry in GATE_REGISTRY
        if entry.mutation_mode is MutationMode.TWO_SIDED
    }

    assert len(required_gates) == 88
    assert set(REQUIRED_DIRECT_CALLS) == required_gates
    assert set(ALLOWED_DIRECT_CALLS) == required_gates


@pytest.mark.parametrize("gate_id", _TWO_SIDED_GATE_IDS)
def test_suite_declares_only_its_reviewed_direct_call_allowlist(gate_id: str) -> None:
    from tests.numerical_gates.boundary_cases import BOUNDARY_SUITES

    suite = BOUNDARY_SUITES[gate_id]
    declarations = {case.direct_methods for case in suite.cases}

    assert len(declarations) == 1, gate_id
    declared = set(next(iter(declarations)))
    assert REQUIRED_DIRECT_CALLS[gate_id] <= declared, (
        gate_id,
        "missing",
        sorted(REQUIRED_DIRECT_CALLS[gate_id] - declared),
    )
    assert declared <= ALLOWED_DIRECT_CALLS[gate_id], (
        gate_id,
        "undeclared by call map",
        sorted(declared - ALLOWED_DIRECT_CALLS[gate_id]),
    )


def test_provider_suite_builders_cannot_receive_an_all_methods_shortcut() -> None:
    offenders: list[tuple[str, int, str]] = []
    for path in _PROVIDERS:
        tree = ast.parse(path.read_text())
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            for keyword in call.keywords:
                if keyword.arg != "direct_methods":
                    continue
                for name in ast.walk(keyword.value):
                    if isinstance(name, ast.Name) and name.id.startswith("_ALL_"):
                        offenders.append((path.name, call.lineno, name.id))

    assert not offenders


def test_raw_observations_are_keyworded_and_never_use_a_refusal_sentinel() -> None:
    positional: list[tuple[str, int]] = []
    sentinels: list[tuple[str, int, object]] = []
    for path in _PROVIDERS:
        tree = ast.parse(path.read_text())
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            name = _call_name(call)
            if name == "RawObservation" and call.args:
                positional.append((path.name, call.lineno))
            if name != "oracle_check":
                continue
            keywords = {keyword.arg: keyword.value for keyword in call.keywords}
            actual = keywords.get("actual")
            expected = keywords.get("expected")
            if not isinstance(actual, ast.Constant) or not isinstance(
                expected, ast.Constant
            ):
                continue
            if actual.value == expected.value and actual.value in {
                "admitted",
                "refused",
            }:
                sentinels.append((path.name, call.lineno, actual.value))

    assert not positional
    assert not sentinels


def test_nonempty_refusal_text_is_not_used_as_a_gate_oracle() -> None:
    """A diagnostic message proves reachability, not numerical correctness."""
    offenders: list[tuple[str, int]] = []
    for path in _PROVIDERS:
        tree = ast.parse(path.read_text())
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            if _call_name(call) != "oracle_check":
                continue
            keywords = {keyword.arg: keyword.value for keyword in call.keywords}
            expected = keywords.get("expected")
            if (
                isinstance(expected, ast.Constant)
                and isinstance(expected.value, str)
                and "non-empty direct refusal reason" in expected.value
            ):
                offenders.append((path.name, call.lineno))

    assert not offenders


def test_refusal_helpers_receive_explicit_production_and_oracle_sides() -> None:
    """A refusal helper cannot default both sides to the expected answer."""
    offenders: list[tuple[str, int, tuple[str, ...]]] = []
    required = {"actual_admitted", "oracle_admitted"}
    for path in _PROVIDERS:
        tree = ast.parse(path.read_text())
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            if _call_name(call) != "_refused":
                continue
            supplied = {keyword.arg for keyword in call.keywords}
            missing = required - supplied
            if missing:
                offenders.append((path.name, call.lineno, tuple(sorted(missing))))

    assert not offenders


def test_atom_evidence_cannot_be_derived_from_its_declared_relation() -> None:
    """Relation metadata describes evidence; it cannot manufacture that evidence."""
    offenders: list[tuple[str, str]] = []
    relation_fields = {"baselines", "target_outcome", "prerequisites"}
    for path in _PROVIDERS:
        tree = ast.parse(path.read_text())
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        tainted = {
            name
            for name, function in functions.items()
            if any(
                isinstance(node, ast.Attribute) and node.attr in relation_fields
                for node in ast.walk(function)
            )
        }
        changed = True
        while changed:
            changed = False
            for name, function in functions.items():
                if name in tainted:
                    continue
                called = {
                    node.func.id
                    for node in ast.walk(function)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                }
                if called & tainted:
                    tainted.add(name)
                    changed = True
        for name, function in functions.items():
            constructs_evidence = any(
                isinstance(node, ast.Call) and _call_name(node) == "AtomEvidence"
                for node in ast.walk(function)
            )
            if constructs_evidence and name in tainted:
                offenders.append((path.name, name))

    assert not offenders


def test_oracle_checks_never_compare_an_expression_with_itself() -> None:
    """A literal self-check can make fabricated atom evidence look independent."""
    offenders: list[tuple[str, int]] = []
    for path in _PROVIDERS:
        tree = ast.parse(path.read_text())
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            if _call_name(call) != "oracle_check":
                continue
            keywords = {keyword.arg: keyword.value for keyword in call.keywords}
            actual = keywords.get("actual")
            expected = keywords.get("expected")
            if actual is None or expected is None:
                continue
            if ast.dump(actual, include_attributes=False) == ast.dump(
                expected, include_attributes=False
            ):
                offenders.append((path.name, call.lineno))

    assert not offenders


def test_provider_instrumentation_state_cannot_be_a_numerical_oracle() -> None:
    """A changed code object is expected under mutation and proves no behavior."""
    offenders: list[tuple[str, int]] = []
    instrumentation_names = {"instrumentation_intact"}
    instrumentation_phrases = (
        "callable changed",
        "code object",
        "instrumented exact source",
        "provider import",
    )
    for path in _PROVIDERS:
        tree = ast.parse(path.read_text())
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            if _call_name(call) != "oracle_check":
                continue
            reads_instrumentation_state = any(
                (
                    isinstance(node, ast.Name)
                    and node.id in instrumentation_names
                )
                or (
                    isinstance(node, ast.Attribute)
                    and node.attr in instrumentation_names
                )
                or (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and any(
                        phrase in node.value.lower()
                        for phrase in instrumentation_phrases
                    )
                )
                for node in ast.walk(call)
            )
            if reads_instrumentation_state:
                offenders.append((path.name, call.lineno))

    assert not offenders


def test_providers_cannot_read_mutation_harness_results() -> None:
    """Only product outputs and independent oracles may score a mutant."""
    offenders: list[tuple[str, int]] = []
    for path in _PROVIDERS:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(
                alias.name == "active_mutation_trace" for alias in node.names
            ):
                offenders.append((path.name, node.lineno))
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "active_mutation_trace"
            ):
                offenders.append((path.name, node.lineno))

    assert not offenders


def test_atom_actual_and_oracle_mappings_are_not_the_same_expression() -> None:
    """A source-result mapping cannot also serve as its independent oracle."""
    offenders: list[tuple[str, int, str]] = []
    for path in _PROVIDERS:
        tree = ast.parse(path.read_text())
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            actual: ast.AST | None = None
            expected: ast.AST | None = None
            label = ""
            keywords = {keyword.arg: keyword.value for keyword in call.keywords}
            if "actual_atoms" in keywords and "oracle_atoms" in keywords:
                actual = keywords["actual_atoms"]
                expected = keywords["oracle_atoms"]
                label = "actual_atoms/oracle_atoms"
            elif _call_name(call) == "_atom_premises" and len(call.args) >= 3:
                actual = call.args[1]
                expected = call.args[2]
                label = "_atom_premises"
            if actual is None or expected is None:
                continue
            if ast.dump(actual, include_attributes=False) == ast.dump(
                expected, include_attributes=False
            ):
                offenders.append((path.name, call.lineno, label))

    assert not offenders


def test_atom_relations_are_not_inferred_from_runtime_evidence() -> None:
    """Relation declarations must remain stable when a fixture or mutant changes."""
    runtime_fields = {
        "actual_atoms",
        "oracle_atoms",
        "observed_side",
        "oracle_side",
        "truth",
    }
    offenders: list[tuple[str, str, int]] = []
    for path in _PROVIDERS:
        tree = ast.parse(path.read_text())
        for function in (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            constructs_relation = any(
                isinstance(node, ast.Call) and _call_name(node) == "AtomRelation"
                for node in ast.walk(function)
            )
            if not constructs_relation:
                continue
            reads_runtime_result = any(
                (isinstance(node, ast.Attribute) and node.attr in runtime_fields)
                or (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "invoke"
                )
                for node in ast.walk(function)
            )
            if reads_runtime_result:
                offenders.append((path.name, function.name, function.lineno))

    assert not offenders
