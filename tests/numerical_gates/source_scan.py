"""Deterministic AST census for the Phase-Two numerical-gate sources.

Candidate identities intentionally exclude line numbers.  They combine a module path,
qualified callable, syntax family, normalized AST fingerprint, and a same-scope
occurrence discriminator.  Current line numbers remain available for diagnostics.
"""

from __future__ import annotations

import ast
import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

SOURCE_PATHS = (
    "src/bayesmith/marginal/_logdet_eager.py",
    "src/bayesmith/marginal/_logdet_ladder.py",
    "src/bayesmith/marginal/_logdet_plan.py",
    "src/bayesmith/diagnose/coupling.py",
    "src/bayesmith/diagnose/map.py",
    "src/bayesmith/graph/reduction.py",
    "src/bayesmith/dispatch/costs.py",
    "src/bayesmith/dispatch/collapse.py",
    "src/bayesmith/dispatch/pilot.py",
    # R3 Task 3.  The evaluation layer's first two registered thresholds
    # (D104, D105) live in this module, so the census has to see it -- an
    # unscanned gate-bearing file is a threshold with no boundary grid and no
    # mutation, which is the arrangement this whole directory exists to make
    # impossible.
    "src/bayesmith/evaluation/checks.py",
)


class CandidateFamily(str, Enum):
    """Exhaustive syntax families emitted by the source scanner."""

    RAISE = "raise"
    COMPARE = "compare"
    DECISION_PREDICATE = "decision_predicate"
    BOOLEAN_ATOM = "boolean_atom"
    GATE_QUALIFIER = "gate_qualifier"
    POLICY_LITERAL = "policy_literal"
    PREDICATE_CALL = "predicate_call_atom"
    NUMERICAL_PREMISE_CALL = "numerical_premise_call"
    FINITE_PREDICATE = "finite_predicate"
    LINALG_PREMISE = "linalg_exception_premise"
    LINALG_ATOM = "linalg_call_atom"
    BITWISE_FINITE_CONJUNCTION = "bitwise_finite_conjunction"
    CLAMP_SELECTOR = "clamp_selector"


class CandidateClassification(str, Enum):
    """Exhaustive review outcomes for raw source candidates."""

    NUMERICAL_GATE = "numerical_gate"
    NUMERICAL_SAFETY = "numerical_safety"
    ORDINARY_VALIDATION = "ordinary_validation"
    STRUCTURAL_CONTROL = "structural_control"
    POLICY_STATIC_REFUSAL = "policy_static_refusal"
    PAYLOAD_TERMINAL_RAISE = "payload_terminal_raise"
    STATIC_SELECTOR = "static_selector"


@dataclass(frozen=True, slots=True)
class SourceCandidate:
    candidate_id: str
    module: str
    qualname: str
    family: CandidateFamily
    fingerprint: str
    occurrence: int
    lineno: int
    col_offset: int
    syntax: str


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    candidate_id: str
    classification: CandidateClassification
    syntax: str


class RegistryFreshnessError(AssertionError):
    """Raised when source candidates and the checked-in manifest differ."""


class _BooleanUseCollector(ast.NodeVisitor):
    """Find assignments that feed truth positions within one lexical scope."""

    def __init__(self) -> None:
        self.truth_names: set[str] = set()
        self.assignments: dict[str, ast.expr] = {}
        self.annotated_boolean_names: set[str] = set()
        self.returned_names: set[str] = set()

    def _consume_truth(self, node: ast.expr) -> None:
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            self.truth_names.add(node.id)
        elif isinstance(node, ast.BoolOp):
            for value in node.values:
                self._consume_truth(value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            self._consume_truth(node.operand)
        elif isinstance(node, ast.NamedExpr):
            self._consume_truth(node.value)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assignments[target.id] = node.value
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name) and _is_bool_annotation(node.annotation):
            self.annotated_boolean_names.add(node.target.id)
        if node.value is not None:
            if isinstance(node.target, ast.Name):
                self.assignments[node.target.id] = node.value
            self.visit(node.value)

    def visit_If(self, node: ast.If) -> None:
        self._consume_truth(node.test)
        for statement in (*node.body, *node.orelse):
            self.visit(statement)

    def visit_While(self, node: ast.While) -> None:
        self._consume_truth(node.test)
        for statement in (*node.body, *node.orelse):
            self.visit(statement)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self._consume_truth(node.test)
        self.visit(node.body)
        self.visit(node.orelse)

    def visit_Return(self, node: ast.Return) -> None:
        if isinstance(node.value, ast.Name):
            self.returned_names.add(node.value.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Nested scopes have their own data-flow pass.
        return

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return


def _boolean_consumers(
    statements: Sequence[ast.stmt], *, annotated_boolean_return: bool = False
) -> set[str]:
    collector = _BooleanUseCollector()
    for statement in statements:
        collector.visit(statement)
    # A branch establishes an unconditional truth position.  A bare returned name
    # only does so when its local producer is recognisably boolean; this avoids
    # turning ordinary returned payloads and integer limits into gate candidates.
    names = set(collector.truth_names) | collector.annotated_boolean_names
    if annotated_boolean_return:
        names.update(collector.returned_names)
    names.update(
        name
        for name in collector.returned_names
        if name in collector.assignments
        and _looks_like_predicate(collector.assignments[name])
    )
    changed = True
    while changed:
        changed = False
        for name, value in collector.assignments.items():
            if isinstance(value, ast.Name) and value.id in names and name not in names:
                names.add(name)
                changed = True
        for name in tuple(names):
            value = collector.assignments.get(name)
            if value is None:
                continue
            dependencies = {
                child.id
                for child in ast.walk(value)
                if isinstance(child, ast.Name)
                and isinstance(child.ctx, ast.Load)
                and child.id in collector.assignments
            }
            missing = dependencies - names
            if missing:
                names.update(missing)
                changed = True
    return names


def _annotated_boolean_producers(statements: Sequence[ast.stmt]) -> set[str]:
    """Return Boolean-annotated locals plus their direct alias chain."""

    collector = _BooleanUseCollector()
    for statement in statements:
        collector.visit(statement)
    names = set(collector.annotated_boolean_names)
    changed = True
    while changed:
        changed = False
        for name, value in collector.assignments.items():
            if isinstance(value, ast.Name) and value.id in names and name not in names:
                names.add(name)
                changed = True
    return names


class _CandidateCollector(ast.NodeVisitor):
    def __init__(self, tree: ast.Module) -> None:
        self.qualnames = ["<module>"]
        self.items: list[tuple[CandidateFamily, ast.AST, str]] = []
        self.recorded: set[tuple[CandidateFamily, int]] = set()
        self.boolean_names = [_boolean_consumers(tree.body)]
        self.annotated_boolean_names = [_annotated_boolean_producers(tree.body)]
        self.boolean_returns = [False]
        self.parents: dict[ast.AST, ast.AST] = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }

    @property
    def qualname(self) -> str:
        return ".".join(self.qualnames)

    def _record(self, family: CandidateFamily, node: ast.AST) -> None:
        key = (family, id(node))
        if key in self.recorded:
            return
        self.recorded.add(key)
        self.items.append((family, node, self.qualname))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.qualnames.append(node.name)
        self.boolean_names.append(_boolean_consumers(node.body))
        self.annotated_boolean_names.append(_annotated_boolean_producers(node.body))
        self.generic_visit(node)
        self.annotated_boolean_names.pop()
        self.boolean_names.pop()
        self.qualnames.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.qualnames.append(node.name)
        annotated_boolean = _is_bool_annotation(node.returns)
        self.boolean_names.append(
            _boolean_consumers(
                node.body,
                annotated_boolean_return=annotated_boolean,
            )
        )
        self.annotated_boolean_names.append(_annotated_boolean_producers(node.body))
        self.boolean_returns.append(annotated_boolean)
        self.generic_visit(node)
        self.boolean_returns.pop()
        self.annotated_boolean_names.pop()
        self.boolean_names.pop()
        self.qualnames.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.qualnames.append(node.name)
        annotated_boolean = _is_bool_annotation(node.returns)
        self.boolean_names.append(
            _boolean_consumers(
                node.body,
                annotated_boolean_return=annotated_boolean,
            )
        )
        self.annotated_boolean_names.append(_annotated_boolean_producers(node.body))
        self.boolean_returns.append(annotated_boolean)
        self.generic_visit(node)
        self.boolean_returns.pop()
        self.annotated_boolean_names.pop()
        self.boolean_names.pop()
        self.qualnames.pop()

    def visit_Raise(self, node: ast.Raise) -> None:
        self._record(CandidateFamily.RAISE, node)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        self._record(CandidateFamily.COMPARE, node)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self._record(CandidateFamily.DECISION_PREDICATE, node.test)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._record(CandidateFamily.DECISION_PREDICATE, node.test)
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self._record(CandidateFamily.DECISION_PREDICATE, node.test)
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self._record(CandidateFamily.DECISION_PREDICATE, node)
        for atom in _boolean_atoms(node):
            self._record(CandidateFamily.BOOLEAN_ATOM, atom)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        target_names = {
            target.id for target in node.targets if isinstance(target, ast.Name)
        }
        if target_names & self.boolean_names[-1]:
            self._record(CandidateFamily.DECISION_PREDICATE, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if (
            node.value is not None
            and isinstance(node.target, ast.Name)
            and node.target.id in self.boolean_names[-1]
        ):
            self._record(CandidateFamily.DECISION_PREDICATE, node.value)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None and (
            self.boolean_returns[-1]
            or _looks_like_predicate(node.value)
            or (
                isinstance(node.value, ast.Name)
                and node.value.id in self.annotated_boolean_names[-1]
            )
        ):
            self._record(CandidateFamily.DECISION_PREDICATE, node.value)
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        if isinstance(node.value, ast.Call):
            name = _call_name(node.value.func).rsplit(".", 1)[-1]
            if name.startswith(("_validate", "_require", "check_")):
                self._record(CandidateFamily.DECISION_PREDICATE, node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name.rsplit(".", 1)[-1] == "PremiseVerdict" and len(node.args) >= 3:
            rung = node.args[0]
            qualifier = node.args[2]
            if isinstance(rung, ast.Constant) and rung.value == 8:
                self._record(CandidateFamily.POLICY_LITERAL, qualifier)
            elif isinstance(rung, ast.Constant) and isinstance(rung.value, int):
                self._record(CandidateFamily.GATE_QUALIFIER, qualifier)
        if name.rsplit(".", 1)[-1] in {
            "_algebraic_rank_bound",
            "_condition_certificate",
            "_factor_projection_certificate",
            "_power_traces_match",
            "_runtime_range_product",
            "_sigma_payload",
            "_two_sum_error",
            "lambda_logdet",
            "index",
            "spectral_radius",
        }:
            self._record(CandidateFamily.NUMERICAL_PREMISE_CALL, node)
        if _looks_like_predicate(node):
            self._record(CandidateFamily.PREDICATE_CALL, node)
        if name.rsplit(".", 1)[-1] == "isfinite":
            self._record(CandidateFamily.FINITE_PREDICATE, node)
        if name.rsplit(".", 1)[-1] == "cholesky":
            self._record(CandidateFamily.LINALG_ATOM, node)
            parent = self.parents.get(node)
            while parent is not None and not isinstance(parent, ast.Try):
                parent = self.parents.get(parent)
            self._record(
                CandidateFamily.LINALG_PREMISE,
                parent if isinstance(parent, ast.Try) else node,
            )
        if isinstance(node.func, ast.Name) and node.func.id in {"min", "max"}:
            self._record(CandidateFamily.CLAMP_SELECTOR, node)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.BitAnd) and any(
            isinstance(child, ast.Call)
            and _call_name(child.func).rsplit(".", 1)[-1] == "isfinite"
            for child in ast.walk(node)
        ):
            self._record(CandidateFamily.BITWISE_FINITE_CONJUNCTION, node)
        self.generic_visit(node)


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _is_bool_annotation(node: ast.expr | None) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "bool"
    if isinstance(node, ast.Attribute):
        return _call_name(node) in {"builtins.bool", "typing.bool"}
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value in {"bool", "builtins.bool"}
    return False


def _looks_like_predicate(node: ast.expr) -> bool:
    if isinstance(node, (ast.BoolOp, ast.Compare)):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return True
    if isinstance(node, ast.Call):
        name = _call_name(node.func)
        leaf = name.rsplit(".", 1)[-1]
        return (
            name == "bool"
            or leaf.startswith("is_")
            or any(
                token in name
                for token in (
                    "isfinite",
                    "array_equal",
                    "allclose",
                    "_is_",
                    ".all",
                    ".any",
                )
            )
        )
    return False


def _boolean_atoms(node: ast.expr) -> tuple[ast.expr, ...]:
    """Return the independently mutable leaves of a compound truth expression."""
    if isinstance(node, ast.BoolOp):
        atoms: list[ast.expr] = []
        for value in node.values:
            atoms.extend(_boolean_atoms(value))
        return tuple(atoms)
    return (node,)


def _normalized_ast(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _enumerate_candidates(
    source: str, module: str
) -> tuple[tuple[SourceCandidate, ast.AST], ...]:
    """Authoritative identity traversal shared by scanning and node location."""
    tree = ast.parse(source, filename=module)
    collector = _CandidateCollector(tree)
    collector.visit(tree)
    family_order = {family: index for index, family in enumerate(CandidateFamily)}
    ordered = sorted(
        collector.items,
        key=lambda item: (
            getattr(item[1], "lineno", 0),
            getattr(item[1], "col_offset", 0),
            family_order[item[0]],
        ),
    )
    occurrences: defaultdict[tuple[str, CandidateFamily, str], int] = defaultdict(int)
    candidates: list[tuple[SourceCandidate, ast.AST]] = []
    for family, node, qualname in ordered:
        normalized = _normalized_ast(node)
        fingerprint = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        occurrence_key = (qualname, family, fingerprint)
        occurrence = occurrences[occurrence_key]
        occurrences[occurrence_key] += 1
        candidate_id = (
            f"{module}::{qualname}::{family.value}::{fingerprint}::{occurrence}"
        )
        candidates.append(
            (
                SourceCandidate(
                    candidate_id=candidate_id,
                    module=module,
                    qualname=qualname,
                    family=family,
                    fingerprint=fingerprint,
                    occurrence=occurrence,
                    lineno=getattr(node, "lineno", 0),
                    col_offset=getattr(node, "col_offset", 0),
                    syntax=ast.unparse(node),
                ),
                node,
            )
        )
    return tuple(candidates)


def scan_source_text(source: str, module: str) -> tuple[SourceCandidate, ...]:
    """Scan one Python source string and return candidates in source order."""
    return tuple(candidate for candidate, _ in _enumerate_candidates(source, module))


def locate_candidate_node(source: str, candidate_id: str) -> ast.AST:
    """Locate one candidate AST node using the scanner's exact occurrence rules."""
    module = candidate_id.split("::", 1)[0]
    for candidate, node in _enumerate_candidates(source, module):
        if candidate.candidate_id == candidate_id:
            return node
    raise KeyError(f"candidate identity does not resolve: {candidate_id}")


def index_source_text(
    source: str, module: str
) -> dict[str, tuple[SourceCandidate, ast.AST]]:
    """Index candidates and exact parsed nodes for occurrence-aware validation."""
    return {
        candidate.candidate_id: (candidate, node)
        for candidate, node in _enumerate_candidates(source, module)
    }


def scan_repository(root: Path) -> tuple[SourceCandidate, ...]:
    """Scan exactly the checked-in gate-bearing source modules in SOURCE_PATHS."""
    candidates: list[SourceCandidate] = []
    for relative_path in SOURCE_PATHS:
        path = root / relative_path
        candidates.extend(scan_source_text(path.read_text(), relative_path))
    return tuple(candidates)


def candidate_family_counts(
    candidates: Iterable[SourceCandidate],
) -> dict[CandidateFamily, int]:
    counts = Counter(candidate.family for candidate in candidates)
    return {family: counts[family] for family in CandidateFamily}


def assert_manifest_fresh(
    actual: Sequence[SourceCandidate], expected: Sequence[ManifestEntry]
) -> None:
    """Require an exact candidate-ID match with focused actionable diagnostics."""
    actual_by_id = {candidate.candidate_id: candidate for candidate in actual}
    expected_ids = {entry.candidate_id for entry in expected}
    actual_ids = set(actual_by_id)
    unexpected = sorted(actual_ids - expected_ids)
    missing = sorted(expected_ids - actual_ids)
    duplicates = len(expected) - len(expected_ids)
    syntax_changes = sorted(
        candidate_id
        for candidate_id in actual_ids & expected_ids
        if actual_by_id[candidate_id].syntax
        != next(
            entry.syntax for entry in expected if entry.candidate_id == candidate_id
        )
    )
    if not unexpected and not missing and not duplicates and not syntax_changes:
        return

    details: list[str] = []
    if unexpected:
        rendered = [
            f"{candidate_id} (line {actual_by_id[candidate_id].lineno}: "
            f"{actual_by_id[candidate_id].syntax})"
            for candidate_id in unexpected
        ]
        details.append("unregistered candidates:\n  " + "\n  ".join(rendered))
    if missing:
        details.append("missing checked-in candidates:\n  " + "\n  ".join(missing))
    if duplicates:
        details.append(f"duplicate manifest classifications: {duplicates}")
    if syntax_changes:
        details.append(
            "normalized syntax differs from the checked-in manifest:\n  "
            + "\n  ".join(syntax_changes)
        )
    details.append("add a registry entry and boundary grid")
    raise RegistryFreshnessError("\n".join(details))
