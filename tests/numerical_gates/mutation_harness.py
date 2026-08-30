"""In-process AST mutations for the two-sided numerical-gate witnesses.

The mutator recompiles only the owning function or method and installs it for
the duration of one context manager.  It never edits a source file.  A probe
at the registered AST node records that the intended production expression
was actually reached; provider-side metadata cannot manufacture that hit.
"""

from __future__ import annotations
import __future__

import ast
import copy
import importlib
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from tests.numerical_gates import oracles
from tests.numerical_gates.boundary_core import (
    BoundaryCase,
    BoundaryExecution,
    GateSide,
)
from tests.numerical_gates.source_scan import index_source_text

_ROOT = Path(__file__).resolve().parents[2]
_PROBE_NAME = "__bayesmith_numerical_gate_mutation_probe__"
_SHIFT_NAME = "__bayesmith_numerical_gate_shift_threshold__"


class MutationDirection(str, Enum):
    """Move a gate past one previously admitted or refused concrete cell."""

    TIGHTEN = "tighten"
    LOOSEN = "loosen"


class MutationStrategy(str, Enum):
    """Reviewed transformations supported by the gate mutation harness."""

    FLIP_BOOLEAN = "flip-boolean"
    FORCE_GATE_SIDE = "force-gate-side"
    SHIFT_COMPARISON = "shift-comparison"
    REPLACE_COMPARISON_THRESHOLD = "replace-comparison-threshold"
    NUMERIC_BOUNDARY = "numeric-boundary"
    LINALG_BOUNDARY = "linalg-boundary"
    CLAMP_THRESHOLD = "clamp-threshold"


class ComparisonThresholdSide(str, Enum):
    """Which operand of a simple comparison is the fixed threshold."""

    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True, slots=True)
class MutationSpec:
    """One exact registered production expression and how to move its gate."""

    gate_id: str
    direction: MutationDirection
    target_id: str
    strategy: MutationStrategy
    true_side: GateSide | None = None
    threshold_side: ComparisonThresholdSide | None = None
    threshold_replacement: float | None = None

    def __post_init__(self) -> None:
        if (
            self.strategy is MutationStrategy.FORCE_GATE_SIDE
            and self.true_side is None
        ):
            raise ValueError("a forced gate side needs predicate=True semantics")
        comparison_strategies = {
            MutationStrategy.SHIFT_COMPARISON,
            MutationStrategy.REPLACE_COMPARISON_THRESHOLD,
        }
        if self.strategy in comparison_strategies:
            if self.true_side is None:
                raise ValueError("a comparison shift needs predicate=True semantics")
            if self.threshold_side is None:
                raise ValueError("a comparison shift needs an explicit threshold operand")
        elif self.threshold_side is not None:
            raise ValueError("only a comparison mutation may select a threshold operand")
        if self.strategy is MutationStrategy.REPLACE_COMPARISON_THRESHOLD:
            if self.threshold_replacement is None:
                raise ValueError("an explicit threshold mutation needs a replacement")
            if isinstance(self.threshold_replacement, bool) or not math.isfinite(
                float(self.threshold_replacement)
            ):
                raise ValueError("a threshold replacement must be finite and numeric")
        elif self.threshold_replacement is not None:
            raise ValueError("only an explicit threshold mutation accepts a replacement")


@dataclass(frozen=True, slots=True)
class MutationResult:
    """The immutable evidence retained from one baseline/mutant pair."""

    spec: MutationSpec
    case_id: str
    baseline: BoundaryExecution
    mutant: BoundaryExecution
    hit_count: int

    @property
    def same_realization(self) -> bool:
        """Whether the mutant consumed the exact frozen baseline fixture."""
        return bool(
            self.mutant.mutation_input_fingerprint
            == self.baseline.mutation_input_fingerprint
        )

    @property
    def killed(self) -> bool:
        """A kill needs a real target hit and a BAD result on the same input."""
        return bool(
            self.hit_count > 0
            and self.same_realization
            and self.mutant.verdict is oracles.NumericalVerdict.BAD
        )


@dataclass(slots=True)
class _Probe:
    spec: MutationSpec
    hit_count: int = 0
    values: list[Any] | None = None

    def __post_init__(self) -> None:
        self.values = []

    def _retain(self, value: Any) -> Any:
        assert self.values is not None
        self.values.append(value)
        return value

    def __call__(self, target_id: str, value: Any) -> Any:
        if target_id != self.spec.target_id:
            raise AssertionError(
                f"mutation probe target changed: {target_id!r} != "
                f"{self.spec.target_id!r}"
            )
        self.hit_count += 1
        strategy = self.spec.strategy
        if strategy is MutationStrategy.FORCE_GATE_SIDE:
            if self.spec.true_side is None:
                raise ValueError("a forced gate side needs predicate=True semantics")
            array = np.asarray(value)
            if array.dtype.kind != "b":
                raise TypeError(
                    f"{target_id} returned non-Boolean {array.dtype}/{array.shape}"
                )
            destination = (
                GateSide.REFUSED
                if self.spec.direction is MutationDirection.TIGHTEN
                else GateSide.ADMITTED
            )
            forced = self.spec.true_side is destination
            result = (
                bool(forced)
                if array.shape == ()
                else np.full(array.shape, forced, dtype=bool)
            )
            return self._retain(result)
        if strategy in {
            MutationStrategy.SHIFT_COMPARISON,
            MutationStrategy.REPLACE_COMPARISON_THRESHOLD,
        }:
            array = np.asarray(value)
            if array.dtype.kind != "b":
                raise TypeError(
                    f"{target_id} returned non-Boolean {array.dtype}/{array.shape}"
                )
            result = bool(array) if array.shape == () else array
            return self._retain(result)
        if strategy is MutationStrategy.FLIP_BOOLEAN:
            array = np.asarray(value)
            if array.dtype.kind != "b":
                raise TypeError(
                    f"{target_id} returned non-Boolean {array.dtype}/{array.shape}"
                )
            inverted = np.logical_not(array)
            result = bool(inverted) if inverted.shape == () else inverted
            return self._retain(result)
        if strategy is MutationStrategy.CLAMP_THRESHOLD:
            return self._retain(value)
        if strategy is MutationStrategy.NUMERIC_BOUNDARY:
            thunk = value
            if self.spec.direction is MutationDirection.TIGHTEN:
                thunk()
                return self._retain(math.inf)
            try:
                thunk()
            except (ArithmeticError, FloatingPointError, ValueError):
                pass
            return self._retain(0.0)
        if strategy is MutationStrategy.LINALG_BOUNDARY:
            thunk, operand = value
            if self.spec.direction is MutationDirection.TIGHTEN:
                raise np.linalg.LinAlgError("in-process tightened SPD boundary")
            try:
                result = thunk()
            except np.linalg.LinAlgError:
                matrix = np.asarray(operand)
                result = np.eye(matrix.shape[-1], dtype=matrix.dtype)
            return self._retain(result)
        raise AssertionError(f"unsupported mutation strategy {strategy!r}")


def _module_name(module_path: str) -> str:
    if not module_path.startswith("src/") or not module_path.endswith(".py"):
        raise ValueError(f"mutation target is not a source module: {module_path!r}")
    return module_path.removeprefix("src/").removesuffix(".py").replace("/", ".")


def _owner_node(tree: ast.Module, qualname: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    owner: ast.AST = tree
    parts = qualname.split(".")
    if not parts or parts[0] != "<module>":
        raise ValueError(f"unexpected scanner qualname {qualname!r}")
    for part in parts[1:]:
        body = getattr(owner, "body", ())
        matches = [
            node
            for node in body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == part
        ]
        if len(matches) != 1:
            raise ValueError(f"cannot resolve owner {qualname!r} at {part!r}")
        owner = matches[0]
    if not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise TypeError(f"mutation target owner is not callable: {qualname!r}")
    return owner


def _same_node(left: ast.AST, right: ast.AST) -> bool:
    return bool(
        getattr(left, "lineno", None) == getattr(right, "lineno", None)
        and getattr(left, "col_offset", None) == getattr(right, "col_offset", None)
        and ast.dump(left, include_attributes=False)
        == ast.dump(right, include_attributes=False)
    )


def _probe_call(target_id: str, value: ast.expr, template: ast.AST) -> ast.Call:
    return ast.copy_location(
        ast.Call(
            func=ast.Name(id=_PROBE_NAME, ctx=ast.Load()),
            args=[ast.Constant(target_id), value],
            keywords=[],
        ),
        template,
    )


def _clamp_mutant(call: ast.Call, direction: MutationDirection) -> ast.Call:
    mutated = copy.deepcopy(call)
    numeric = [
        (index, argument)
        for index, argument in enumerate(mutated.args)
        if isinstance(argument, ast.Constant)
        and not isinstance(argument.value, bool)
        and argument.value == 1.0
    ]
    if len(numeric) != 1:
        raise ValueError("clamp mutation requires one literal 1.0 threshold")
    index, original = numeric[0]
    # The mutation direction is defined by the downstream curvature
    # acceptance, not by which arm of ``max`` is selected.  Raising the clamp
    # raises the curvature floor (tighten); removing it lowers that floor
    # (loosen).
    replacement = 2.0 if direction is MutationDirection.TIGHTEN else 0.0
    mutated.args[index] = ast.copy_location(ast.Constant(replacement), original)
    return mutated


def _desired_comparison_truth(spec: MutationSpec) -> bool:
    if spec.true_side is None:
        raise ValueError("a comparison shift needs the gate side for predicate=True")
    destination = (
        GateSide.REFUSED
        if spec.direction is MutationDirection.TIGHTEN
        else GateSide.ADMITTED
    )
    return spec.true_side is destination


def _normalized_operator(
    operator: ast.cmpop, threshold_side: ComparisonThresholdSide
) -> ast.cmpop:
    if threshold_side is ComparisonThresholdSide.RIGHT:
        return operator
    inverse: dict[type[ast.cmpop], type[ast.cmpop]] = {
        ast.Lt: ast.Gt,
        ast.LtE: ast.GtE,
        ast.Gt: ast.Lt,
        ast.GtE: ast.LtE,
    }
    try:
        return inverse[type(operator)]()
    except KeyError as error:
        raise TypeError(
            "a shifted comparison must use <, <=, >, or >="
        ) from error


def _comparison_shift_step(operator: ast.cmpop, desired_truth: bool) -> int:
    if isinstance(operator, (ast.Lt, ast.LtE)):
        return 1 if desired_truth else -1
    if isinstance(operator, (ast.Gt, ast.GtE)):
        return -1 if desired_truth else 1
    raise TypeError("a shifted comparison must use <, <=, >, or >=")


def _shifted_comparison(target: ast.Compare, spec: MutationSpec) -> ast.Call:
    if len(target.ops) != 1 or len(target.comparators) != 1:
        raise TypeError("a comparison shift requires one binary comparison")
    if spec.threshold_side is None:
        raise ValueError("a comparison shift needs an explicit threshold operand")
    mutated = copy.deepcopy(target)
    normalized = _normalized_operator(mutated.ops[0], spec.threshold_side)
    step = _comparison_shift_step(normalized, _desired_comparison_truth(spec))
    if spec.threshold_side is ComparisonThresholdSide.RIGHT:
        threshold = mutated.comparators[0]
        mutated.comparators[0] = ast.copy_location(
            ast.Call(
                func=ast.Name(id=_SHIFT_NAME, ctx=ast.Load()),
                args=[threshold, ast.Constant(step)],
                keywords=[],
            ),
            threshold,
        )
    else:
        threshold = mutated.left
        mutated.left = ast.copy_location(
            ast.Call(
                func=ast.Name(id=_SHIFT_NAME, ctx=ast.Load()),
                args=[threshold, ast.Constant(step)],
                keywords=[],
            ),
            threshold,
        )
    return _probe_call(spec.target_id, mutated, target)


def _replaced_comparison(target: ast.Compare, spec: MutationSpec) -> ast.Call:
    """Replace one literal threshold with a reviewed value across a real gap."""
    if len(target.ops) != 1 or len(target.comparators) != 1:
        raise TypeError("an explicit threshold replacement needs one comparison")
    if spec.threshold_side is None or spec.threshold_replacement is None:
        raise ValueError("an explicit threshold mutation is incomplete")
    mutated = copy.deepcopy(target)
    normalized = _normalized_operator(mutated.ops[0], spec.threshold_side)
    desired_truth = _desired_comparison_truth(spec)
    step = _comparison_shift_step(normalized, desired_truth)
    threshold = (
        mutated.comparators[0]
        if spec.threshold_side is ComparisonThresholdSide.RIGHT
        else mutated.left
    )
    if not (
        isinstance(threshold, ast.Constant)
        and isinstance(threshold.value, (int, float))
        and not isinstance(threshold.value, bool)
    ):
        raise TypeError("an explicit replacement may only replace a numeric literal")
    delta = float(spec.threshold_replacement) - float(threshold.value)
    if delta * step <= 0.0:
        raise ValueError(
            "the explicit replacement does not move the threshold toward the "
            "requested gate side"
        )
    replacement = ast.copy_location(
        ast.Constant(spec.threshold_replacement),
        threshold,
    )
    if spec.threshold_side is ComparisonThresholdSide.RIGHT:
        mutated.comparators[0] = replacement
    else:
        mutated.left = replacement
    return _probe_call(spec.target_id, mutated, target)


def _shift_threshold(value: Any, step: int) -> Any:
    """Move one representable float or one integer in a fixed direction."""
    if step not in {-1, 1}:
        raise ValueError(f"threshold step must be +/-1, got {step!r}")
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("a Boolean threshold cannot be shifted numerically")
    if isinstance(value, int):
        return value + step
    if isinstance(value, np.integer):
        shifted = int(value) + step
        return np.asarray(shifted, dtype=np.asarray(value).dtype)[()]
    array = np.asarray(value)
    if array.dtype.kind != "f":
        raise TypeError(f"threshold has non-numeric dtype {array.dtype}")
    direction = np.asarray(math.inf if step > 0 else -math.inf, dtype=array.dtype)
    shifted = np.nextafter(array, direction, dtype=array.dtype)
    if isinstance(value, np.ndarray):
        return shifted
    if isinstance(value, np.floating):
        return shifted[()]
    return float(shifted)


def _replacement(target: ast.AST, spec: MutationSpec) -> ast.AST:
    if not isinstance(target, ast.expr):
        raise TypeError(f"{spec.target_id} is not an expression target")
    if spec.strategy is MutationStrategy.SHIFT_COMPARISON:
        if not isinstance(target, ast.Compare):
            raise TypeError("a comparison shift requires a comparison target")
        return _shifted_comparison(target, spec)
    if spec.strategy is MutationStrategy.REPLACE_COMPARISON_THRESHOLD:
        if not isinstance(target, ast.Compare):
            raise TypeError("an explicit threshold replacement needs a comparison")
        return _replaced_comparison(target, spec)
    if spec.strategy in {
        MutationStrategy.FLIP_BOOLEAN,
        MutationStrategy.FORCE_GATE_SIDE,
    }:
        return _probe_call(spec.target_id, target, target)
    if spec.strategy is MutationStrategy.NUMERIC_BOUNDARY:
        if not isinstance(target, ast.Call):
            raise TypeError("a numerical premise mutation requires a call target")
        thunk = ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=target,
        )
        return _probe_call(spec.target_id, thunk, target)
    if spec.strategy is MutationStrategy.LINALG_BOUNDARY:
        if not isinstance(target, ast.Call) or not target.args:
            raise TypeError("a linear-algebra mutation requires a call with an operand")
        thunk = ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=target,
        )
        pair = ast.Tuple(elts=[thunk, copy.deepcopy(target.args[0])], ctx=ast.Load())
        return _probe_call(spec.target_id, pair, target)
    if spec.strategy is MutationStrategy.CLAMP_THRESHOLD:
        if not isinstance(target, ast.Call):
            raise TypeError("a clamp mutation requires a call target")
        return _probe_call(
            spec.target_id,
            _clamp_mutant(target, spec.direction),
            target,
        )
    raise AssertionError(f"unsupported mutation strategy {spec.strategy!r}")


def _compile_mutant(
    spec: MutationSpec,
) -> tuple[ModuleType, object, str, object, object]:
    module_path, _separator, _rest = spec.target_id.partition("::")
    source_path = _ROOT / module_path
    source = source_path.read_text(encoding="utf-8")
    try:
        candidate, indexed_target = index_source_text(source, module_path)[
            spec.target_id
        ]
    except KeyError as error:
        raise ValueError(f"mutation target no longer resolves: {spec.target_id}") from error

    tree = ast.parse(source, filename=module_path)
    owner = _owner_node(tree, candidate.qualname)
    mutated_owner = copy.deepcopy(owner)
    targets = [
        node for node in ast.walk(mutated_owner) if _same_node(node, indexed_target)
    ]
    if len(targets) != 1:
        raise ValueError(
            f"mutation target resolves {len(targets)} times inside "
            f"{candidate.qualname}: {spec.target_id}"
        )
    target = targets[0]
    replacement = _replacement(target, spec)

    class Transformer(ast.NodeTransformer):
        def generic_visit(self, node: ast.AST) -> ast.AST:
            if node is target:
                return replacement
            return super().generic_visit(node)

    transformed = Transformer().visit(mutated_owner)
    assert isinstance(transformed, (ast.FunctionDef, ast.AsyncFunctionDef))
    transformed.decorator_list = []
    temporary_name = "__bayesmith_compiled_gate_mutant__"
    transformed.name = temporary_name
    ast.fix_missing_locations(transformed)

    module = importlib.import_module(_module_name(module_path))
    absent = object()
    previous_temporary = module.__dict__.pop(temporary_name, absent)
    try:
        code = compile(
            ast.Module(body=[transformed], type_ignores=[]),
            module_path,
            "exec",
            flags=__future__.annotations.compiler_flag,
        )
        # The code is a transformed function from one of the six checked-in,
        # manifest-pinned modules; neither its path nor its AST is user input.
        exec(code, module.__dict__)  # noqa: S102
        mutant = module.__dict__.pop(temporary_name)
    finally:
        module.__dict__.pop(temporary_name, None)
        if previous_temporary is not absent:
            module.__dict__[temporary_name] = previous_temporary

    owner_parts = candidate.qualname.split(".")[1:]
    runtime_owner: object = module
    for part in owner_parts[:-1]:
        runtime_owner = getattr(runtime_owner, part)
    attribute = owner_parts[-1]
    original = getattr(runtime_owner, attribute)
    mutant.__name__ = getattr(original, "__name__", attribute)
    mutant.__qualname__ = getattr(original, "__qualname__", candidate.qualname)
    mutant.__module__ = getattr(original, "__module__", module.__name__)
    mutant.__defaults__ = getattr(original, "__defaults__", None)
    mutant.__kwdefaults__ = getattr(original, "__kwdefaults__", None)
    mutant.__annotations__ = dict(getattr(original, "__annotations__", {}))
    return module, runtime_owner, attribute, original, mutant


@contextmanager
def installed_mutation(spec: MutationSpec) -> Iterator[_Probe]:
    """Install one exact in-process mutation and restore it on every exit path."""
    module, owner, attribute, original, mutant = _compile_mutant(spec)
    absent = object()
    previous_probe = module.__dict__.get(_PROBE_NAME, absent)
    previous_shift = module.__dict__.get(_SHIFT_NAME, absent)
    probe = _Probe(spec)
    module.__dict__[_PROBE_NAME] = probe
    module.__dict__[_SHIFT_NAME] = _shift_threshold
    setattr(owner, attribute, mutant)
    try:
        yield probe
    finally:
        setattr(owner, attribute, original)
        if previous_probe is absent:
            module.__dict__.pop(_PROBE_NAME, None)
        else:
            module.__dict__[_PROBE_NAME] = previous_probe
        if previous_shift is absent:
            module.__dict__.pop(_SHIFT_NAME, None)
        else:
            module.__dict__[_SHIFT_NAME] = previous_shift


def run_mutation(spec: MutationSpec, case: BoundaryCase) -> MutationResult:
    """Run the frozen case before and during one in-process production mutation."""
    if case.gate_id != spec.gate_id:
        raise ValueError(
            f"case {case.case_id} belongs to {case.gate_id}, not {spec.gate_id}"
        )
    baseline = case()
    with installed_mutation(spec) as probe:
        mutant = case()
    return MutationResult(
        spec=spec,
        case_id=case.case_id,
        baseline=baseline,
        mutant=mutant,
        hit_count=probe.hit_count,
    )


__all__ = [
    "ComparisonThresholdSide",
    "MutationDirection",
    "MutationResult",
    "MutationSpec",
    "MutationStrategy",
    "installed_mutation",
    "run_mutation",
]
