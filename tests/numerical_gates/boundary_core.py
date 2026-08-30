"""Stable execution contract for numerical-gate boundary providers.

Provider modules own inputs and production calls.  This module owns only the
record format, the exact numerical grading rule, and structural validation.
In particular, it never infers a gate verdict from registry prose.
"""

from __future__ import annotations

import ast
import hashlib
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from tests.numerical_gates import oracles
from tests.numerical_gates.registry import (
    GATE_REGISTRY,
    AxisRange,
    GateEntry,
    isolatable_atom_ids,
)
from tests.numerical_gates.source_scan import index_source_text

_ENTRY_BY_GATE = {entry.gate_id: entry for entry in GATE_REGISTRY}


class FixtureFamily(str, Enum):
    """The nine fixture/oracle families required by the Phase-Two plan."""

    SPD_SPECTRUM_CONDITION = "spd-spectrum-condition-common-scale"
    EXACT_STRUCTURE_EVIDENCE = "exact-structure-evidence"
    FACTOR_CERTIFICATES = "factor-certificates"
    RHO_CERTIFICATE_TRACE_ORDER = "rho-certificate-trace-order"
    LADDER_SIZE_RANK_ROUTING = "ladder-size-rank-routing"
    PLAN_SCALAR_PROOF_RANGE = "plan-scalar-proof-range"
    COUPLING = "coupling"
    MAP_DERIVATIVES_CURVATURE = "map-derivatives-curvature"
    GRAPH_DOMAIN_STRUCTURE = "graph-domain-structure"


class ExecutionClass(str, Enum):
    """Whether a failed side has a meaningful numerical payload."""

    VALIDATION_ONLY = "V"
    PAYLOAD_OR_REFUSAL = "P"
    TWO_PAYLOAD = "2P"


class BoundaryTopology(str, Enum):
    """Neighbour construction appropriate to a gate's actual domain."""

    FLOAT = "float"
    INTEGER = "integer"
    EXACT = "exact"
    CAPABILITY = "capability"


class GateSide(str, Enum):
    """A registered gate's admitted or refused side."""

    ADMITTED = "admitted"
    REFUSED = "refused"


class AtomRelationKind(str, Enum):
    """How one named scanner atom can be evidenced by a real failure case."""

    INDEPENDENT = "independent"
    ALIAS = "alias"
    DEPENDENT = "dependent"


class AtomDependencyLogic(str, Enum):
    """Machine-readable relationship between target and companion atoms."""

    ALL_OF = "all-of"
    ANY_OF = "any-of"
    ALL_ELEMENTS = "all-elements"
    PREREQUISITES_IMPLY_TARGET = "prerequisites-imply-target"
    TARGET_IMPLIES_PREREQUISITES = "target-implies-prerequisites"
    EQUIVALENT = "equivalent"
    SHORT_CIRCUIT = "short-circuit"


class AtomReducer(str, Enum):
    """How a raw source value acquires truth in its actual source context."""

    SCALAR = "scalar"
    ALL_ELEMENTS = "all-elements"
    ANY_ELEMENT = "any-element"
    NOT_EVALUATED = "not-evaluated"


@dataclass(frozen=True, slots=True)
class AtomBaseline:
    """Raw contextual truth of one canonical atom in the neutral fixture."""

    atom_id: str
    outcome: bool


@dataclass(frozen=True, slots=True)
class AtomPrerequisite:
    """One source atom whose real outcome is required by a dependency."""

    atom_id: str
    expected_outcome: bool | None


@dataclass(frozen=True, slots=True)
class AtomRelation:
    """Source-backed classification of one registered atomic witness."""

    kind: AtomRelationKind
    baselines: tuple[AtomBaseline, ...]
    target_outcome: bool
    canonical_atom_id: str | None = None
    prerequisites: tuple[AtomPrerequisite, ...] = ()
    logic: AtomDependencyLogic | None = None
    rationale: str | None = None


@dataclass(frozen=True, slots=True)
class AtomEvidence:
    """Raw source result, independent oracle, context, and direct-call lineage."""

    atom_id: str
    raw_actual: Any
    truth: bool | None
    reducer: AtomReducer
    realized_keys: tuple[str, ...]
    oracle: str


class PointRole(str, Enum):
    """Stable semantic locations in a boundary neighbourhood."""

    VERY_LOW = "very-low"
    BELOW_RELATIVE_1E6 = "T-relative-minus-1e-6"
    BELOW_RELATIVE_1E12 = "T-relative-minus-1e-12"
    BELOW_ULP = "nextafter-T-minus"
    REACHABLE_BELOW = "nearest-reachable-below-T"
    BELOW_INTEGER = "T-minus-1"
    AT = "T"
    ABOVE_ULP = "nextafter-T-plus"
    REACHABLE_ABOVE = "nearest-reachable-above-T"
    ABOVE_INTEGER = "T-plus-1"
    ABOVE_RELATIVE_1E12 = "T-relative-plus-1e-12"
    ABOVE_RELATIVE_1E6 = "T-relative-plus-1e-6"
    EXACT = "exact"
    ULP_MISMATCH = "one-ULP-mismatch"
    SUBNORMAL_MISMATCH = "min-subnormal-mismatch"
    MATERIAL_MISMATCH = "material-mismatch"
    CAPABILITY_LOW = "capability-low"
    VALID_CAPABILITY = "valid-capability"
    INVALID_CAPABILITY = "invalid-capability"
    CAPABILITY_HIGH = "capability-high"
    VERY_HIGH = "very-high"
    EXTREME = "extreme"


class AxisPosition(str, Enum):
    """Required locations on every declared parameter axis."""

    VERY_LOW = "very-low"
    ENDPOINT_LOW = "endpoint-low"
    ENDPOINT_HIGH = "endpoint-high"
    VERY_HIGH = "very-high"
    EXTREME = "extreme"
    INTERIOR = "interior-companion"


_ROLE_AXIS_POSITIONS: Mapping[PointRole, frozenset[AxisPosition]] = {
    PointRole.VERY_LOW: frozenset({AxisPosition.VERY_LOW}),
    PointRole.BELOW_RELATIVE_1E6: frozenset({AxisPosition.ENDPOINT_LOW}),
    PointRole.BELOW_RELATIVE_1E12: frozenset({AxisPosition.ENDPOINT_LOW}),
    PointRole.BELOW_ULP: frozenset({AxisPosition.ENDPOINT_LOW}),
    PointRole.REACHABLE_BELOW: frozenset({AxisPosition.ENDPOINT_LOW}),
    PointRole.BELOW_INTEGER: frozenset({AxisPosition.ENDPOINT_LOW}),
    PointRole.AT: frozenset({AxisPosition.ENDPOINT_LOW, AxisPosition.ENDPOINT_HIGH}),
    PointRole.EXACT: frozenset({AxisPosition.ENDPOINT_LOW}),
    PointRole.ABOVE_ULP: frozenset({AxisPosition.ENDPOINT_HIGH}),
    PointRole.REACHABLE_ABOVE: frozenset({AxisPosition.ENDPOINT_HIGH}),
    PointRole.ABOVE_INTEGER: frozenset({AxisPosition.ENDPOINT_HIGH}),
    PointRole.ABOVE_RELATIVE_1E12: frozenset({AxisPosition.ENDPOINT_HIGH}),
    PointRole.ABOVE_RELATIVE_1E6: frozenset({AxisPosition.ENDPOINT_HIGH}),
    PointRole.ULP_MISMATCH: frozenset({AxisPosition.ENDPOINT_HIGH}),
    PointRole.SUBNORMAL_MISMATCH: frozenset({AxisPosition.ENDPOINT_HIGH}),
    PointRole.MATERIAL_MISMATCH: frozenset({AxisPosition.ENDPOINT_HIGH}),
    PointRole.CAPABILITY_LOW: frozenset({AxisPosition.VERY_LOW}),
    PointRole.VALID_CAPABILITY: frozenset({AxisPosition.ENDPOINT_LOW}),
    PointRole.INVALID_CAPABILITY: frozenset({AxisPosition.ENDPOINT_HIGH}),
    PointRole.CAPABILITY_HIGH: frozenset({AxisPosition.VERY_HIGH}),
    PointRole.VERY_HIGH: frozenset({AxisPosition.VERY_HIGH}),
    PointRole.EXTREME: frozenset({AxisPosition.EXTREME}),
}


@dataclass(frozen=True, slots=True)
class ThresholdPoint:
    """One realized threshold point and its independently expected side."""

    role: PointRole
    display_value: str
    delta: str
    expected_side: GateSide


@dataclass(frozen=True, slots=True)
class AxisSample:
    """A concrete declared-axis location attached to an executable cell."""

    axis_name: str
    position: AxisPosition
    display_value: str


@dataclass(frozen=True, slots=True)
class RealizedAxis:
    """One actual parameter-axis value exercised by a production call."""

    axis_name: str
    position: AxisPosition
    input_key: str
    value: Any


@dataclass(frozen=True, slots=True)
class RealizedPoint:
    """Actual scalar/evidence distance consumed by the production predicate."""

    quantity: str
    input_key: str
    value: Any
    threshold: Any
    dtype: str | None
    axes: tuple[RealizedAxis, ...]
    active_axis: str | None = None


@dataclass(frozen=True, slots=True)
class OracleCheck:
    """Non-numerical or exact-domain result from an independent oracle."""

    oracle: str
    actual: Any
    expected: Any
    passed: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "passed", _raw_exact_equal(self.actual, self.expected))


@dataclass(frozen=True, slots=True)
class DirectEvaluation:
    """A direct production payload measured against a non-product oracle."""

    method: str
    oracle: str
    actual: float
    oracle_value: float
    relative_error: float
    verdict: oracles.NumericalVerdict

    @classmethod
    def compare(
        cls,
        *,
        method: str,
        oracle: str,
        actual: float,
        oracle_value: float,
    ) -> DirectEvaluation:
        actual_value = float(actual)
        expected_value = float(oracle_value)
        return cls(
            method=method,
            oracle=oracle,
            actual=actual_value,
            oracle_value=expected_value,
            relative_error=oracles.relative_error(actual_value, expected_value),
            verdict=oracles.numerical_verdict(actual_value, expected_value),
        )


@dataclass(frozen=True, slots=True)
class RawObservation:
    """Provider output before the common contract is applied.

    ``direct_input_keys`` name actual arguments or fixture fields consumed by
    the recorded direct calls.  ``direct_return_keys`` name actual return or
    certificate fields produced by those calls or their predicate seam.  A
    derived gate quantity may use a return key; an axis must use an input key.
    Providers must record the returned object value itself, not recompute an
    equivalent metadata value solely for this audit.
    """

    observed_side: GateSide
    realized_point: RealizedPoint
    realized_inputs: Mapping[str, Any]
    direct_input_keys: tuple[str, ...]
    direct_return_keys: tuple[str, ...]
    direct_calls: tuple[str, ...]
    oracle_checks: tuple[OracleCheck, ...]
    evaluations: tuple[DirectEvaluation, ...] = ()
    isolated_atom: str | None = None
    atom_evidence: tuple[AtomEvidence, ...] = ()
    sibling_premises: tuple[OracleCheck, ...] = ()
    defensive: bool = False
    notes: tuple[str, ...] = ()


CaseRunner = Callable[[ThresholdPoint], RawObservation]


@dataclass(frozen=True, slots=True)
class BoundaryCase:
    """A stable, callable gate or atomic-premise baseline case."""

    case_id: str
    gate_id: str
    atom_ids: tuple[str, ...]
    fixture_family: FixtureFamily
    execution_class: ExecutionClass
    topology: BoundaryTopology
    threshold_point: ThresholdPoint
    axes: tuple[AxisSample, ...]
    direct_methods: tuple[str, ...]
    independent_oracles: tuple[str, ...]
    non_unit_scale: tuple[float, float] | None
    atom_relation: AtomRelation | None
    runner: CaseRunner
    active_axis: str | None = None

    def __call__(self) -> BoundaryExecution:
        """Execute the production call and normalize its retained evidence."""
        return execute_case(self)


@dataclass(frozen=True, slots=True)
class BoundaryExecution:
    """Complete executable record required by the Task-3 shared harness."""

    case_id: str
    gate_id: str
    atom_ids: tuple[str, ...]
    threshold_point: ThresholdPoint
    realized_point: RealizedPoint
    realization_fingerprint: str
    mutation_input_fingerprint: str
    axes: tuple[AxisSample, ...]
    direct_input_keys: tuple[str, ...]
    direct_return_keys: tuple[str, ...]
    direct_return_values: Mapping[str, Any]
    direct_calls: tuple[str, ...]
    oracle_values: tuple[float, ...]
    relative_errors: tuple[float, ...]
    expected_gate_side: GateSide
    observed_gate_side: GateSide
    verdict: oracles.NumericalVerdict
    evaluations: tuple[DirectEvaluation, ...]
    oracle_checks: tuple[OracleCheck, ...]
    isolated_atom: str | None
    atom_relation: AtomRelation | None
    atom_evidence: tuple[AtomEvidence, ...]
    sibling_premises: tuple[OracleCheck, ...]
    defensive: bool
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BoundarySuite:
    """All boundary and axis cells for one reviewed semantic gate."""

    gate_id: str
    fixture_family: FixtureFamily
    execution_class: ExecutionClass
    topology: BoundaryTopology
    cases: tuple[BoundaryCase, ...]
    atom_case_ids: Mapping[str, str]
    tighten_case_id: str
    loosen_case_id: str
    ambiguities: tuple[str, ...] = ()
    omitted_unrepresentable_roles: frozenset[PointRole] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "atom_case_ids", MappingProxyType(dict(self.atom_case_ids))
        )
        relative_roles = {
            PointRole.BELOW_RELATIVE_1E6,
            PointRole.BELOW_RELATIVE_1E12,
            PointRole.ABOVE_RELATIVE_1E12,
            PointRole.ABOVE_RELATIVE_1E6,
        }
        exact_neighbourhood_roles = {
            PointRole.BELOW_ULP,
            PointRole.AT,
            PointRole.ABOVE_ULP,
        }
        permitted_omissions = relative_roles | exact_neighbourhood_roles
        if self.omitted_unrepresentable_roles and self.topology is not BoundaryTopology.FLOAT:
            raise ValueError("only float roles may be unrepresentable")
        if not self.omitted_unrepresentable_roles <= permitted_omissions:
            raise ValueError(
                "only float threshold-neighbourhood roles may be unrepresentable"
            )
        if self.omitted_unrepresentable_roles & exact_neighbourhood_roles:
            if not self.ambiguities or any(not item.strip() for item in self.ambiguities):
                raise ValueError(
                    "an exact-neighbourhood omission needs a quantified ambiguity"
                )
            roles = {case.threshold_point.role for case in self.cases}
            if not {
                PointRole.REACHABLE_BELOW,
                PointRole.REACHABLE_ABOVE,
            } <= roles:
                raise ValueError(
                    "an exact-neighbourhood omission needs measured reachable straddles"
                )


@dataclass(frozen=True, slots=True)
class CaseReference:
    """Concrete callable alias consumed by Task 4."""

    name: str
    gate_id: str
    atom_id: str | None
    case_id: str
    cases: Mapping[str, BoundaryCase]

    def __call__(self) -> BoundaryExecution:
        return self.cases[self.case_id]()


def numerical_evaluation(
    *, method: str, oracle: str, actual: float, oracle_value: float
) -> DirectEvaluation:
    """Build an evaluation using the one permitted relative-error formula."""
    return DirectEvaluation.compare(
        method=method,
        oracle=oracle,
        actual=actual,
        oracle_value=oracle_value,
    )


def _raw_exact_equal(left: Any, right: Any) -> bool:
    """Compare retained raw values without provider-supplied certification."""
    try:
        first = np.asarray(left)
        second = np.asarray(right)
    except (TypeError, ValueError):
        return type(left) is type(right) and left == right
    return bool(
        first.dtype == second.dtype
        and first.shape == second.shape
        and np.array_equal(first, second)
    )


def oracle_check(*, oracle: str, actual: Any, expected: Any) -> OracleCheck:
    """Build an exact/domain check whose verdict is derived centrally."""
    return OracleCheck(oracle, actual, expected)


def _source_family(atom_id: str) -> str:
    try:
        return atom_id.rsplit("::", 4)[2]
    except IndexError as error:
        raise ValueError(f"malformed source atom ID: {atom_id!r}") from error


@cache
def _source_index(module: str) -> Mapping[str, tuple[Any, ast.AST]]:
    source_path = Path(__file__).parents[2] / module
    return MappingProxyType(index_source_text(source_path.read_text(), module))


def source_alias_canonical(entry: GateEntry, atom_id: str) -> str:
    """Return the deterministic canonical atom for one scanner-family alias set."""
    if atom_id not in entry.conjunction_atom_ids:
        raise ValueError(f"{atom_id!r} is not an atom of {entry.gate_id}")
    source_index = _source_index(entry.module)
    try:
        target_node = source_index[atom_id][1]
    except KeyError as error:
        raise ValueError(
            f"registered atom does not resolve in source: {atom_id}"
        ) from error
    aliases = [
        candidate
        for candidate in entry.conjunction_atom_ids
        if source_index[candidate][1] is target_node
    ]
    family_priority = {
        "predicate_call_atom": 0,
        "finite_predicate": 1,
    }
    return min(
        aliases,
        key=lambda candidate: (
            family_priority.get(_source_family(candidate), 2),
            candidate,
        ),
    )


def source_ast_prerequisites(entry: GateEntry, atom_id: str) -> tuple[str, ...]:
    """Return canonical registered atoms strictly contained by ``atom_id``'s AST."""
    if atom_id not in entry.conjunction_atom_ids:
        raise ValueError(f"{atom_id!r} is not an atom of {entry.gate_id}")
    if source_alias_canonical(entry, atom_id) != atom_id:
        return ()
    source_index = _source_index(entry.module)
    try:
        target = source_index[atom_id][1]
    except KeyError as error:
        raise ValueError(
            f"registered atom does not resolve in source: {atom_id}"
        ) from error
    descendant_ids = {id(node) for node in ast.walk(target)} - {id(target)}
    prerequisites: list[str] = []
    for candidate in entry.conjunction_atom_ids:
        canonical = source_alias_canonical(entry, candidate)
        if candidate != canonical or canonical == atom_id:
            continue
        try:
            candidate_node = source_index[canonical][1]
        except KeyError as error:
            raise ValueError(
                f"registered atom does not resolve in source: {canonical}"
            ) from error
        if id(candidate_node) in descendant_ids:
            prerequisites.append(canonical)
    return tuple(prerequisites)


def validate_atom_relation(
    *, entry: GateEntry, atom_id: str, relation: AtomRelation
) -> None:
    """Validate one relation against canonical scanner identity and source AST."""
    if atom_id not in entry.conjunction_atom_ids:
        raise ValueError(f"{atom_id!r} is not an atom of {entry.gate_id}")
    canonical = source_alias_canonical(entry, atom_id)
    ast_prerequisites = source_ast_prerequisites(entry, canonical)
    canonical_ids = tuple(
        candidate
        for candidate in entry.conjunction_atom_ids
        if source_alias_canonical(entry, candidate) == candidate
    )
    baseline_ids = tuple(item.atom_id for item in relation.baselines)
    if len(baseline_ids) != len(set(baseline_ids)):
        raise ValueError("atom baselines repeat a canonical atom")
    if set(baseline_ids) != set(canonical_ids):
        raise ValueError(
            f"atom baselines must cover canonical source identities exactly; "
            f"expected={canonical_ids!r}, actual={baseline_ids!r}"
        )
    if any(type(item.outcome) is not bool for item in relation.baselines):
        raise ValueError("atom baseline outcomes must be raw booleans")
    if type(relation.target_outcome) is not bool:
        raise ValueError("atom target outcome must be a raw boolean")
    baseline_by_id = {item.atom_id: item.outcome for item in relation.baselines}
    if relation.target_outcome is baseline_by_id[canonical]:
        raise ValueError(f"{atom_id} target outcome does not flip its neutral baseline")
    if relation.kind is AtomRelationKind.INDEPENDENT:
        if canonical != atom_id:
            raise ValueError(
                f"{atom_id} is a scanner alias; canonical source identity is {canonical}"
            )
        if ast_prerequisites:
            raise ValueError(
                f"{atom_id} has AST-containment prerequisites {ast_prerequisites!r}"
            )
        if (
            relation.canonical_atom_id is not None
            or relation.prerequisites
            or relation.logic is not None
            or relation.rationale is not None
        ):
            raise ValueError("an independent atom cannot declare alias/dependency data")
        return
    if relation.kind is AtomRelationKind.ALIAS:
        if canonical == atom_id:
            raise ValueError(f"{atom_id} is already its canonical source identity")
        if relation.canonical_atom_id != canonical:
            raise ValueError(
                f"{atom_id} canonical source identity is {canonical}, not "
                f"{relation.canonical_atom_id}"
            )
        if not relation.prerequisites:
            if ast_prerequisites:
                raise ValueError(
                    f"{atom_id} must mirror canonical AST-containment dependencies "
                    f"{ast_prerequisites!r}"
                )
            if relation.logic is not None or relation.rationale is not None:
                raise ValueError(
                    "an alias without prerequisites cannot declare dependency logic"
                )
            return
    elif relation.kind is AtomRelationKind.DEPENDENT:
        if canonical != atom_id:
            raise ValueError(
                f"{atom_id} is a scanner alias and must bind canonical source "
                f"identity {canonical}"
            )
        if relation.canonical_atom_id is not None:
            raise ValueError("a dependent atom cannot declare an alias canonical atom")
    else:
        raise ValueError(f"unknown atom relation kind: {relation.kind!r}")
    if not relation.prerequisites:
        raise ValueError("an atom dependency needs at least one prerequisite")
    if relation.logic is None:
        raise ValueError("a dependent atom needs machine-readable logic")
    if not (relation.rationale and relation.rationale.strip()):
        raise ValueError("a dependent atom needs a reviewable reason")
    prerequisite_ids = tuple(item.atom_id for item in relation.prerequisites)
    if len(prerequisite_ids) != len(set(prerequisite_ids)):
        raise ValueError("a dependent atom repeats a prerequisite")
    if canonical in prerequisite_ids:
        raise ValueError("a dependent atom cannot be its own prerequisite")
    unknown = set(prerequisite_ids) - set(entry.conjunction_atom_ids)
    if unknown:
        raise ValueError(
            f"dependency prerequisites are not gate atoms: {sorted(unknown)}"
        )
    noncanonical = {
        candidate
        for candidate in prerequisite_ids
        if source_alias_canonical(entry, candidate) != candidate
    }
    if noncanonical:
        raise ValueError(
            "dependency prerequisites must use canonical source identities: "
            f"{sorted(noncanonical)}"
        )
    declared_ast = set(prerequisite_ids) & set(ast_prerequisites)
    if ast_prerequisites and not declared_ast:
        raise ValueError(
            f"{atom_id} has AST-containment prerequisites {ast_prerequisites!r}; "
            "the dependency must select at least one real child"
        )
    for prerequisite in relation.prerequisites:
        if (
            prerequisite.expected_outcome is not None
            and type(prerequisite.expected_outcome) is not bool
        ):
            raise ValueError(
                "prerequisite outcomes must be raw booleans or not-evaluated"
            )
        if prerequisite.expected_outcome is baseline_by_id[prerequisite.atom_id]:
            raise ValueError(
                f"dependency {prerequisite.atom_id} does not differ from its baseline"
            )


def validate_atom_evidence(
    *,
    entry: GateEntry,
    atom_id: str,
    relation: AtomRelation,
    evidence: tuple[AtomEvidence, ...],
    available_realized_keys: frozenset[str],
    independent_oracles: frozenset[str],
    oracle_checks: tuple[OracleCheck, ...],
) -> None:
    """Check raw atom/oracle results against permitted truth and lineage."""
    validate_atom_relation(entry=entry, atom_id=atom_id, relation=relation)
    evidence_ids = tuple(item.atom_id for item in evidence)
    if len(evidence_ids) != len(set(evidence_ids)):
        raise AssertionError(f"{atom_id} repeated atom evidence")
    required_ids = set(entry.conjunction_atom_ids)
    if set(evidence_ids) != required_ids:
        missing = sorted(required_ids - set(evidence_ids))
        extra = sorted(set(evidence_ids) - required_ids)
        raise AssertionError(
            f"{atom_id} atom evidence differs from the registry; "
            f"missing={missing!r}, extra={extra!r}"
        )
    baseline_by_id = {item.atom_id: item.outcome for item in relation.baselines}
    expected = {
        candidate: baseline_by_id[source_alias_canonical(entry, candidate)]
        for candidate in entry.conjunction_atom_ids
    }

    def set_alias_group(candidate: str, outcome: bool | None) -> None:
        canonical = source_alias_canonical(entry, candidate)
        for equivalent in entry.conjunction_atom_ids:
            if source_alias_canonical(entry, equivalent) == canonical:
                expected[equivalent] = outcome

    set_alias_group(atom_id, relation.target_outcome)
    if relation.prerequisites:
        for prerequisite in relation.prerequisites:
            set_alias_group(prerequisite.atom_id, prerequisite.expected_outcome)
    actual: dict[str, bool | None] = {}
    by_id = {item.atom_id: item for item in evidence}
    for item in evidence:
        if not item.realized_keys:
            raise AssertionError(f"{item.atom_id} recorded no direct-call lineage")
        missing_lineage = set(item.realized_keys) - available_realized_keys
        if missing_lineage:
            raise AssertionError(
                f"{item.atom_id} atom lineage is not direct-call evidence: "
                f"{sorted(missing_lineage)!r}"
            )
        if item.oracle not in independent_oracles:
            raise AssertionError(
                f"{item.atom_id} used undeclared atom oracle {item.oracle!r}"
            )
        matching_checks = [
            check
            for check in oracle_checks
            if check.oracle == item.oracle
            and check.passed
            and (
                _raw_atom_equal(check.actual, item.raw_actual)
                or _raw_atom_equal(check.actual, item.truth)
            )
        ]
        if not matching_checks:
            raise AssertionError(
                f"{item.atom_id} has no passed independent check for its raw result"
            )
        reduced = _reduce_atom_truth(
            item.raw_actual, item.reducer, atom_id=item.atom_id
        )
        if item.truth is not None and type(item.truth) is not bool:
            raise AssertionError(
                f"{item.atom_id} contextual truth is not Boolean/not-evaluated"
            )
        if item.truth is not reduced:
            raise AssertionError(
                f"{item.atom_id} contextual truth does not match its raw reducer"
            )
        actual[item.atom_id] = item.truth
    for candidate, item in by_id.items():
        canonical = source_alias_canonical(entry, candidate)
        canonical_item = by_id[canonical]
        if not _raw_atom_equal(item.raw_actual, canonical_item.raw_actual):
            raise AssertionError(
                f"{candidate} scanner alias does not share its canonical raw result"
            )
        if item.reducer is not canonical_item.reducer:
            raise AssertionError(
                f"{candidate} scanner alias changed its source truth reducer"
            )
        if item.truth is not canonical_item.truth:
            raise AssertionError(
                f"{candidate} scanner alias changed its contextual truth"
            )
    undeclared_changes = sorted(
        candidate
        for candidate, expected_value in expected.items()
        if actual[candidate]
        is not baseline_by_id[source_alias_canonical(entry, candidate)]
        and actual[candidate] is not expected_value
    )
    if undeclared_changes:
        raise AssertionError(
            f"{atom_id} has undeclared sibling outcome changes: {undeclared_changes!r}"
        )
    mismatches = sorted(
        candidate
        for candidate, expected_value in expected.items()
        if actual[candidate] is not expected_value
    )
    if mismatches:
        raise AssertionError(
            f"{atom_id} atom evidence violates its declared relation: {mismatches!r}"
        )
    _validate_dependency_truth(
        atom_id=atom_id,
        relation=relation,
        evidence=by_id,
        actual=actual,
    )


def _raw_atom_equal(actual: Any, oracle_value: Any) -> bool:
    """Exact equality for scalar or array-valued predicate results."""
    actual_array = np.asarray(actual)
    oracle_array = np.asarray(oracle_value)
    return bool(
        actual_array.dtype == oracle_array.dtype
        and actual_array.shape == oracle_array.shape
        and np.array_equal(actual_array, oracle_array)
    )


def _reduce_atom_truth(
    raw_value: Any, reducer: AtomReducer, *, atom_id: str
) -> bool | None:
    """Apply only the source-context reducer explicitly retained by a provider."""
    if reducer is AtomReducer.NOT_EVALUATED:
        if raw_value is not None:
            raise AssertionError(
                f"{atom_id} not-evaluated evidence must retain raw None"
            )
        return None
    value = np.asarray(raw_value)
    if value.dtype.kind != "b":
        raise AssertionError(f"{atom_id} raw atom result is not Boolean-valued")
    if reducer is AtomReducer.SCALAR:
        if value.shape != ():
            raise AssertionError(
                f"{atom_id} needs an array reducer for raw shape {value.shape!r}"
            )
        return bool(value.item())
    if reducer is AtomReducer.ALL_ELEMENTS:
        return bool(np.all(value))
    if reducer is AtomReducer.ANY_ELEMENT:
        return bool(np.any(value))
    raise AssertionError(f"{atom_id} has unknown source truth reducer {reducer!r}")


def _validate_dependency_truth(
    *,
    atom_id: str,
    relation: AtomRelation,
    evidence: Mapping[str, AtomEvidence],
    actual: Mapping[str, bool | None],
) -> None:
    """Evaluate the declared logical relation on the real retained truth row."""
    if not relation.prerequisites:
        return
    target_truth = actual[atom_id]
    prerequisite_truth = [
        actual[prerequisite.atom_id] for prerequisite in relation.prerequisites
    ]
    target_event = target_truth is relation.target_outcome
    prerequisite_events = [
        actual[prerequisite.atom_id] is prerequisite.expected_outcome
        for prerequisite in relation.prerequisites
    ]
    logic = relation.logic
    if logic is AtomDependencyLogic.ALL_OF:
        passed = target_truth is all(prerequisite_truth)
    elif logic is AtomDependencyLogic.ANY_OF:
        passed = target_truth is any(prerequisite_truth)
    elif logic is AtomDependencyLogic.ALL_ELEMENTS:
        passed = target_truth is all(
            bool(np.all(np.asarray(evidence[prerequisite.atom_id].raw_actual)))
            for prerequisite in relation.prerequisites
        )
    elif logic is AtomDependencyLogic.PREREQUISITES_IMPLY_TARGET:
        passed = not all(prerequisite_events) or target_event
    elif logic is AtomDependencyLogic.TARGET_IMPLIES_PREREQUISITES:
        passed = not target_event or all(prerequisite_events)
    elif logic is AtomDependencyLogic.EQUIVALENT:
        passed = all(value is target_truth for value in prerequisite_truth)
    elif logic is AtomDependencyLogic.SHORT_CIRCUIT:
        passed = target_event and any(value is None for value in prerequisite_truth)
    else:
        raise AssertionError(f"{atom_id} has no evaluable dependency logic")
    if not passed:
        raise AssertionError(
            f"{atom_id} real atom evidence violates dependency logic {logic.value}"
        )


def axis_position_for_role(role: PointRole) -> AxisPosition:
    """Return the canonical position; AT defaults to the lower endpoint."""
    return min(_ROLE_AXIS_POSITIONS[role], key=lambda item: item.value)


def realized_point(
    *,
    entry: GateEntry,
    point: ThresholdPoint,
    quantity: str,
    input_key: str,
    value: Any,
    threshold: Any,
    dtype: str | None,
    axis_input_key: str | None = None,
    axis_value: Any | None = None,
    axis_position: AxisPosition | None = None,
    active_axis: str | None = None,
    axis_bindings: Mapping[str, tuple[str, Any, AxisPosition]] | None = None,
) -> RealizedPoint:
    """Build a linked realized point for the registry's declared axes."""
    if len(entry.axes) > 1 and (active_axis is None or axis_bindings is None):
        raise ValueError(
            "a compound registry entry needs an explicit active axis and bindings"
        )
    if axis_bindings is not None:
        expected_names = {axis.name for axis in entry.axes}
        if set(axis_bindings) != expected_names:
            raise ValueError(
                "axis bindings must cover the compound registry axes exactly; "
                f"expected={sorted(expected_names)!r}, "
                f"actual={sorted(axis_bindings)!r}"
            )
        if active_axis not in expected_names:
            raise ValueError(f"unknown active axis {active_axis!r}")
        axes = tuple(
            RealizedAxis(
                axis.name,
                axis_bindings[axis.name][2],
                axis_bindings[axis.name][0],
                axis_bindings[axis.name][1],
            )
            for axis in entry.axes
        )
        return RealizedPoint(
            quantity=quantity,
            input_key=input_key,
            value=value,
            threshold=threshold,
            dtype=dtype,
            axes=axes,
            active_axis=active_axis,
        )
    linked_key = input_key if axis_input_key is None else axis_input_key
    linked_value = value if axis_value is None else axis_value
    position = (
        axis_position_for_role(point.role) if axis_position is None else axis_position
    )
    return RealizedPoint(
        quantity=quantity,
        input_key=input_key,
        value=value,
        threshold=threshold,
        dtype=dtype,
        axes=tuple(
            RealizedAxis(axis.name, position, linked_key, linked_value)
            for axis in entry.axes
        ),
        active_axis=(entry.axes[0].name if len(entry.axes) == 1 else active_axis),
    )


def execute_case(case: BoundaryCase) -> BoundaryExecution:
    """Execute and validate one provider-owned boundary cell."""
    observation = case.runner(case.threshold_point)
    _validate_realized_point(
        case.threshold_point,
        observation.realized_point,
        active_axis=case.active_axis,
    )
    if not observation.realized_inputs:
        raise AssertionError(f"{case.case_id} recorded no realized production inputs")
    _validate_realized_inputs(case, observation)
    realization_fingerprint = fingerprint_realization(
        observation.realized_point, observation.realized_inputs
    )
    mutation_input_fingerprint = fingerprint_mutation_inputs(
        observation.realized_point,
        observation.realized_inputs,
        observation.direct_input_keys,
    )
    if case.atom_ids:
        if case.atom_relation is None:
            raise AssertionError(f"{case.case_id} has no atom relation")
        if observation.isolated_atom != case.atom_ids[0]:
            raise AssertionError(
                f"{case.case_id} isolated {observation.isolated_atom!r}, "
                f"expected {case.atom_ids[0]!r}"
            )
        validate_atom_evidence(
            entry=_ENTRY_BY_GATE[case.gate_id],
            atom_id=case.atom_ids[0],
            relation=case.atom_relation,
            evidence=observation.atom_evidence,
            available_realized_keys=frozenset(
                {*observation.direct_input_keys, *observation.direct_return_keys}
            ),
            independent_oracles=frozenset(case.independent_oracles),
            oracle_checks=(observation.oracle_checks + observation.sibling_premises),
        )
    elif observation.isolated_atom is not None:
        raise AssertionError(f"{case.case_id} claimed an atom on a non-atomic baseline")
    elif case.atom_relation is not None or observation.atom_evidence:
        raise AssertionError(f"{case.case_id} recorded atom data on a grid baseline")
    if not observation.direct_calls:
        raise AssertionError(f"{case.case_id} recorded no direct production call")
    undeclared_calls = set(observation.direct_calls) - set(case.direct_methods)
    if undeclared_calls:
        raise AssertionError(
            f"{case.case_id} used undeclared direct calls {sorted(undeclared_calls)}"
        )
    undeclared_oracles = (
        {check.oracle for check in observation.oracle_checks}
        | {check.oracle for check in observation.sibling_premises}
        | {evaluation.oracle for evaluation in observation.evaluations}
    )
    undeclared_oracles -= set(case.independent_oracles)
    if undeclared_oracles:
        raise AssertionError(
            f"{case.case_id} used undeclared oracles {sorted(undeclared_oracles)}"
        )
    if not observation.oracle_checks and not observation.evaluations:
        raise AssertionError(f"{case.case_id} recorded no independent oracle evidence")
    if (
        case.execution_class is ExecutionClass.TWO_PAYLOAD
        and not observation.evaluations
    ):
        raise AssertionError(
            f"{case.case_id} is a two-payload boundary without a numerical evaluation"
        )

    comparison_verdicts = [item.verdict for item in observation.evaluations]
    if any(not item.passed for item in observation.oracle_checks):
        comparison_verdicts.append(oracles.NumericalVerdict.BAD)
    if observation.observed_side is not case.threshold_point.expected_side:
        comparison_verdicts.append(oracles.NumericalVerdict.BAD)
    if observation.observed_side is GateSide.ADMITTED and any(
        not (math.isfinite(item.actual) and math.isfinite(item.oracle_value))
        for item in observation.evaluations
    ):
        comparison_verdicts.append(oracles.NumericalVerdict.BAD)
    verdict = _worst_verdict(comparison_verdicts)
    return BoundaryExecution(
        case_id=case.case_id,
        gate_id=case.gate_id,
        atom_ids=case.atom_ids,
        threshold_point=case.threshold_point,
        realized_point=observation.realized_point,
        realization_fingerprint=realization_fingerprint,
        mutation_input_fingerprint=mutation_input_fingerprint,
        axes=case.axes,
        direct_input_keys=observation.direct_input_keys,
        direct_return_keys=observation.direct_return_keys,
        direct_return_values=MappingProxyType(
            {
                key: observation.realized_inputs[key]
                for key in observation.direct_return_keys
            }
        ),
        direct_calls=observation.direct_calls,
        oracle_values=tuple(item.oracle_value for item in observation.evaluations),
        relative_errors=tuple(item.relative_error for item in observation.evaluations),
        expected_gate_side=case.threshold_point.expected_side,
        observed_gate_side=observation.observed_side,
        verdict=verdict,
        evaluations=observation.evaluations,
        oracle_checks=observation.oracle_checks,
        isolated_atom=observation.isolated_atom,
        atom_relation=case.atom_relation,
        atom_evidence=observation.atom_evidence,
        sibling_premises=observation.sibling_premises,
        defensive=observation.defensive,
        notes=observation.notes,
    )


def fingerprint_realization(
    point: RealizedPoint, realized_inputs: Mapping[str, Any]
) -> str:
    """Hash actual values, shapes, layouts, axes, and seam quantities."""
    digest = hashlib.sha256()
    _update_fingerprint(
        digest,
        {
            "quantity": point.quantity,
            "input_key": point.input_key,
            "value": point.value,
            "threshold": point.threshold,
            "dtype": point.dtype,
            "active_axis": point.active_axis,
            "axes": tuple(
                (
                    axis.axis_name,
                    axis.position.value,
                    axis.input_key,
                    axis.value,
                )
                for axis in point.axes
            ),
            "inputs": realized_inputs,
        },
    )
    return digest.hexdigest()


def fingerprint_mutation_inputs(
    point: RealizedPoint,
    realized_inputs: Mapping[str, Any],
    direct_input_keys: tuple[str, ...],
) -> str:
    """Hash only the frozen call inputs and threshold used by a mutant pair.

    ``fingerprint_realization`` deliberately includes retained production
    returns so distinct boundary roles cannot masquerade as one fixture.  A
    successful mutation is expected to change those returns, however.  This
    companion fingerprint excludes return keys while retaining the literal
    threshold, dtype, axes, and every declared production-call input.
    """
    missing = set(direct_input_keys) - set(realized_inputs)
    if missing:
        raise AssertionError(
            f"mutation inputs are missing retained values: {sorted(missing)!r}"
        )
    digest = hashlib.sha256()
    _update_fingerprint(
        digest,
        {
            "quantity": point.quantity,
            "threshold": point.threshold,
            "dtype": point.dtype,
            "active_axis": point.active_axis,
            "axes": tuple(
                (
                    axis.axis_name,
                    axis.position.value,
                    axis.input_key,
                    axis.value,
                )
                for axis in point.axes
            ),
            "direct_inputs": {
                key: realized_inputs[key] for key in sorted(direct_input_keys)
            },
        },
    )
    return digest.hexdigest()


def _update_fingerprint(digest: Any, value: Any) -> None:
    """Canonicalize only reviewable deterministic input types."""
    if value is None:
        digest.update(b"none;")
    elif isinstance(value, bool):
        digest.update(f"bool:{int(value)};".encode())
    elif isinstance(value, (int, np.integer)):
        digest.update(f"int:{int(value)};".encode())
    elif isinstance(value, (float, np.floating)):
        digest.update(f"float:{float(value).hex()};".encode())
    elif isinstance(value, (complex, np.complexfloating)):
        scalar = complex(value)
        digest.update(f"complex:{scalar.real.hex()}:{scalar.imag.hex()};".encode())
    elif isinstance(value, str):
        digest.update(b"str:")
        digest.update(value.encode())
        digest.update(b";")
    elif isinstance(value, bytes):
        digest.update(f"bytes:{len(value)}:".encode())
        digest.update(value)
        digest.update(b";")
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("realized boundary arrays cannot have object dtype")
        digest.update(
            f"array:{value.dtype.str}:{value.shape}:{value.strides}:".encode()
        )
        digest.update(value.tobytes(order="A"))
        digest.update(b";")
    elif isinstance(value, Mapping):
        digest.update(b"mapping{")
        for key in sorted(value, key=str):
            _update_fingerprint(digest, str(key))
            _update_fingerprint(digest, value[key])
        digest.update(b"};")
    elif isinstance(value, (tuple, list)):
        digest.update(f"sequence:{type(value).__name__}:{len(value)}[".encode())
        for item in value:
            _update_fingerprint(digest, item)
        digest.update(b"];")
    else:
        raise TypeError(
            "realized boundary inputs must be scalars, arrays, sequences, or "
            f"mappings; got {type(value).__name__}"
        )


def _validate_realized_point(
    declared: ThresholdPoint,
    realized: RealizedPoint,
    *,
    active_axis: str | None = None,
) -> None:
    """Prove that the actual consumed quantity matches its declared role."""
    if not realized.quantity:
        raise AssertionError("a realized point needs a named consumed quantity")
    if not realized.input_key:
        raise AssertionError("a realized point needs a consumed input key")
    if not realized.axes:
        raise AssertionError("a realized point needs concrete axis values")
    if any(not axis.axis_name for axis in realized.axes):
        raise AssertionError("every realized axis needs a name")
    if any(not axis.input_key for axis in realized.axes):
        raise AssertionError("every realized axis needs an input key")
    axis_names = [axis.axis_name for axis in realized.axes]
    if len(axis_names) != len(set(axis_names)):
        raise AssertionError("a realized point repeated an axis")
    selected_axis = active_axis or realized.active_axis
    if len(realized.axes) == 1:
        selected_axis = realized.axes[0].axis_name if selected_axis is None else selected_axis
    elif selected_axis is None:
        raise AssertionError("a compound realized point needs one active axis")
    if selected_axis not in axis_names:
        raise AssertionError(f"active axis {selected_axis!r} was not realized")
    if (
        active_axis is not None
        and realized.active_axis is not None
        and active_axis != realized.active_axis
    ):
        raise AssertionError("case and realized point disagree on the active axis")
    role = declared.role
    expected_axis_positions = _ROLE_AXIS_POSITIONS[role]
    active = next(axis for axis in realized.axes if axis.axis_name == selected_axis)
    if active.position not in expected_axis_positions:
        raise AssertionError(
            f"{role.value} did not place active axis {selected_axis!r} at a "
            "corresponding position"
        )
    companions = [axis for axis in realized.axes if axis.axis_name != selected_axis]
    if any(axis.position is not AxisPosition.INTERIOR for axis in companions):
        raise AssertionError(
            f"{role.value} did not keep every companion axis at a valid interior"
        )
    value = realized.value
    threshold = realized.threshold

    if role in {
        PointRole.CAPABILITY_LOW,
        PointRole.VALID_CAPABILITY,
        PointRole.INVALID_CAPABILITY,
        PointRole.CAPABILITY_HIGH,
    }:
        return
    if role is PointRole.EXTREME:
        return
    if role in {PointRole.EXACT, PointRole.AT}:
        if not _exact_scalar_equal(value, threshold):
            raise AssertionError(
                f"{role.value} did not realize exact threshold equality"
            )
        return
    if role in {PointRole.BELOW_INTEGER, PointRole.ABOVE_INTEGER}:
        if isinstance(value, bool) or isinstance(threshold, bool):
            raise AssertionError("integer neighbours cannot be bool")
        if not isinstance(value, (int, np.integer)) or not isinstance(
            threshold, (int, np.integer)
        ):
            raise AssertionError("integer neighbours need integer values")
        expected = int(threshold) + (-1 if role is PointRole.BELOW_INTEGER else 1)
        if int(value) != expected:
            raise AssertionError(f"{role.value} did not realize T±1")
        return

    numeric_value, numeric_threshold = _finite_numeric_pair(value, threshold)
    if role in {PointRole.REACHABLE_BELOW, PointRole.REACHABLE_ABOVE}:
        below = role is PointRole.REACHABLE_BELOW
        ordered = (
            numeric_value < numeric_threshold
            if below
            else numeric_value > numeric_threshold
        )
        if not ordered:
            relation = "below" if below else "above"
            raise AssertionError(
                f"{role.value} is not strictly {relation} its threshold"
            )
        return
    if role in {PointRole.BELOW_ULP, PointRole.ABOVE_ULP}:
        dtype = np.dtype(np.float64 if realized.dtype is None else realized.dtype)
        cast_threshold = np.asarray(numeric_threshold, dtype=dtype)
        direction = -math.inf if role is PointRole.BELOW_ULP else math.inf
        expected = np.nextafter(
            cast_threshold,
            np.asarray(direction, dtype=dtype),
            dtype=dtype,
        ).item()
        if not _exact_scalar_equal(numeric_value, expected):
            raise AssertionError(f"{role.value} did not realize one dtype ULP")
        return
    if role in {
        PointRole.BELOW_RELATIVE_1E6,
        PointRole.BELOW_RELATIVE_1E12,
        PointRole.ABOVE_RELATIVE_1E12,
        PointRole.ABOVE_RELATIVE_1E6,
    }:
        target = (
            1e-6
            if role in {PointRole.BELOW_RELATIVE_1E6, PointRole.ABOVE_RELATIVE_1E6}
            else 1e-12
        )
        scale = abs(numeric_threshold)
        if scale == 0.0:
            raise AssertionError(
                "a zero threshold cannot realize multiplicative relative roles"
            )
        relative = abs(numeric_value - numeric_threshold) / scale
        below = role in {
            PointRole.BELOW_RELATIVE_1E6,
            PointRole.BELOW_RELATIVE_1E12,
        }
        if (numeric_value < numeric_threshold) is not below or not math.isclose(
            relative, target, rel_tol=1e-4, abs_tol=target * 1e-8
        ):
            raise AssertionError(f"{role.value} did not realize its relative delta")
        return
    if role is PointRole.ULP_MISMATCH:
        dtype = np.dtype(np.float64 if realized.dtype is None else realized.dtype)
        cast_threshold = np.asarray(numeric_threshold, dtype=dtype)
        neighbors = {
            np.nextafter(
                cast_threshold,
                np.asarray(-math.inf, dtype=dtype),
                dtype=dtype,
            ).item(),
            np.nextafter(
                cast_threshold,
                np.asarray(math.inf, dtype=dtype),
                dtype=dtype,
            ).item(),
        }
        if numeric_value not in neighbors:
            raise AssertionError("one-ULP evidence mismatch is not adjacent")
        return
    if role is PointRole.SUBNORMAL_MISMATCH:
        dtype = np.dtype(np.float64 if realized.dtype is None else realized.dtype)
        smallest = float(np.nextafter(dtype.type(0), dtype.type(1), dtype=dtype))
        if abs(numeric_value - numeric_threshold) != smallest:
            raise AssertionError("subnormal evidence mismatch is not minimum-sized")
        return
    if role is PointRole.MATERIAL_MISMATCH:
        scale = max(abs(numeric_threshold), 1.0)
        if abs(numeric_value - numeric_threshold) < 1e-6 * scale:
            raise AssertionError("material evidence mismatch is smaller than 1e-6")
        return
    if role is PointRole.VERY_LOW and not numeric_value < numeric_threshold:
        raise AssertionError("very-low point is not below its threshold")
    if role is PointRole.VERY_HIGH and not numeric_value > numeric_threshold:
        raise AssertionError("very-high point is not above its threshold")


def _validate_realized_inputs(case: BoundaryCase, observation: RawObservation) -> None:
    """Link quantities to direct arguments/returns and axes to arguments."""
    inputs = observation.realized_inputs
    point = observation.realized_point
    direct_input_keys = observation.direct_input_keys
    direct_return_keys = observation.direct_return_keys
    if not direct_input_keys:
        raise AssertionError(f"{case.case_id} recorded no direct-call input keys")
    if len(set(direct_input_keys)) != len(direct_input_keys):
        raise AssertionError(f"{case.case_id} repeated a direct-call input key")
    if len(set(direct_return_keys)) != len(direct_return_keys):
        raise AssertionError(f"{case.case_id} repeated a direct-call return key")
    overlap = set(direct_input_keys) & set(direct_return_keys)
    if overlap:
        raise AssertionError(
            f"{case.case_id} used keys as both direct inputs and returns: "
            f"{sorted(overlap)}"
        )
    lineage_keys = set(direct_input_keys) | set(direct_return_keys)
    missing_lineage = lineage_keys - set(inputs)
    if missing_lineage:
        raise AssertionError(
            f"{case.case_id} recorded absent direct input/return keys "
            f"{sorted(missing_lineage)}"
        )
    claimed_keys = (
        lineage_keys | {point.input_key} | {axis.input_key for axis in point.axes}
    )
    if "boundary_quantity" in claimed_keys:
        raise AssertionError(
            f"{case.case_id} used synthetic 'boundary_quantity' metadata"
        )
    if point.input_key not in lineage_keys:
        raise AssertionError(
            f"{case.case_id} consumed quantity key {point.input_key!r} is not "
            "a direct-call input or return"
        )
    if point.input_key not in inputs:
        raise AssertionError(
            f"{case.case_id} consumed quantity key {point.input_key!r} is absent"
        )
    if _value_fingerprint(point.value) != _value_fingerprint(inputs[point.input_key]):
        raise AssertionError(
            f"{case.case_id} consumed quantity is not its realized input value"
        )
    declared_axes = {sample.axis_name for sample in case.axes}
    realized_axes = {axis.axis_name for axis in point.axes}
    if realized_axes != declared_axes or len(point.axes) != len(declared_axes):
        raise AssertionError(
            f"{case.case_id} must realize exactly one value for every declared "
            f"axis; got {[axis.axis_name for axis in point.axes]!r}, expected "
            f"{sorted(declared_axes)!r}"
        )
    if len(declared_axes) > 1:
        if case.active_axis is None or point.active_axis is None:
            raise AssertionError(
                f"{case.case_id} compound case did not declare one active axis"
            )
        if case.active_axis != point.active_axis:
            raise AssertionError(
                f"{case.case_id} case/observation active axes disagree"
            )
    elif case.active_axis is not None and case.active_axis not in declared_axes:
        raise AssertionError(
            f"{case.case_id} active axis {case.active_axis!r} is not declared"
        )
    for axis in point.axes:
        if axis.input_key not in direct_input_keys:
            raise AssertionError(
                f"{case.case_id} axis {axis.axis_name!r} does not link to a "
                "direct-call input"
            )
        if axis.input_key not in inputs:
            raise AssertionError(
                f"{case.case_id} axis key {axis.input_key!r} is absent"
            )
        if _value_fingerprint(axis.value) != _value_fingerprint(inputs[axis.input_key]):
            raise AssertionError(
                f"{case.case_id} axis {axis.axis_name!r} is not linked to its input"
            )


def _value_fingerprint(value: Any) -> str:
    digest = hashlib.sha256()
    _update_fingerprint(digest, value)
    return digest.hexdigest()


def validate_axis_position_values(suite: BoundarySuite) -> None:
    """Require five materially distinct direct-input regions on every axis."""
    expected_positions = set(AxisPosition) - {AxisPosition.INTERIOR}
    by_axis: dict[str, dict[AxisPosition, set[str]]] = {}
    companion_values: dict[tuple[str, str], dict[str, list[str]]] = {}
    for case in suite.cases:
        if case.atom_ids:
            continue
        execution = case()
        active_name = execution.realized_point.active_axis
        if active_name is None:
            if len(execution.realized_point.axes) != 1:
                raise AssertionError(
                    f"{case.case_id} did not identify its active compound axis"
                )
            active_name = execution.realized_point.axes[0].axis_name
        active = next(
            axis
            for axis in execution.realized_point.axes
            if axis.axis_name == active_name
        )
        if active.position is AxisPosition.INTERIOR:
            raise AssertionError(f"{case.case_id} made its active axis interior")
        by_axis.setdefault(active.axis_name, {}).setdefault(
            active.position, set()
        ).add(_value_fingerprint(active.value))
        for companion in execution.realized_point.axes:
            if companion.axis_name == active_name:
                continue
            fingerprint = _value_fingerprint(companion.value)
            companion_values.setdefault(
                (active_name, companion.axis_name), {}
            ).setdefault(fingerprint, []).append(case.case_id)

    for (active_name, companion_name), fingerprints in companion_values.items():
        if len(fingerprints) <= 1:
            continue
        owners = [case_ids for case_ids in fingerprints.values()]
        raise AssertionError(
            f"{suite.gate_id} companion axis {companion_name!r} changed while "
            f"sweeping active axis {active_name!r}: {owners!r}"
        )

    for axis_name, positions in by_axis.items():
        if set(positions) != expected_positions:
            raise AssertionError(
                f"{suite.gate_id} axis {axis_name!r} did not realize all five "
                f"positions: {sorted(item.value for item in positions)!r}"
            )
        owners: dict[str, set[AxisPosition]] = {}
        for position, fingerprints in positions.items():
            for fingerprint in fingerprints:
                owners.setdefault(fingerprint, set()).add(position)
        reused = {
            fingerprint: values
            for fingerprint, values in owners.items()
            if len(values) > 1
        }
        if reused:
            rendered = [
                sorted(position.value for position in positions)
                for positions in reused.values()
            ]
            raise AssertionError(
                f"{suite.gate_id} reused a direct input across axis positions "
                f"for {axis_name!r}: {rendered!r}"
            )


def validate_reachable_gap(suite: BoundarySuite) -> None:
    """Prove a claimed reachable straddle has no representable input between it."""
    exact_roles = {PointRole.BELOW_ULP, PointRole.AT, PointRole.ABOVE_ULP}
    if not suite.omitted_unrepresentable_roles & exact_roles:
        return
    executions = {
        case.threshold_point.role: case()
        for case in suite.cases
        if not case.atom_ids
        and case.threshold_point.role
        in {PointRole.REACHABLE_BELOW, PointRole.AT, PointRole.REACHABLE_ABOVE}
    }
    required = {PointRole.REACHABLE_BELOW, PointRole.REACHABLE_ABOVE}
    if not required <= set(executions):
        raise AssertionError(f"{suite.gate_id} has no executable reachable straddle")

    below = executions[PointRole.REACHABLE_BELOW]
    above = executions[PointRole.REACHABLE_ABOVE]
    threshold = float(below.realized_point.threshold)
    if not _raw_exact_equal(
        below.realized_point.threshold, above.realized_point.threshold
    ):
        raise AssertionError(f"{suite.gate_id} reachable sides changed threshold")
    below_output = float(below.realized_point.value)
    above_output = float(above.realized_point.value)
    if not below_output < threshold < above_output:
        raise AssertionError(
            f"{suite.gate_id} reachable outputs do not strictly straddle threshold"
        )

    def active_value(execution: BoundaryExecution) -> float:
        active_name = execution.realized_point.active_axis
        axes = execution.realized_point.axes
        if active_name is None:
            if len(axes) != 1:
                raise AssertionError(
                    f"{execution.case_id} did not name its reachable-gap axis"
                )
            active_name = axes[0].axis_name
        active = next(axis for axis in axes if axis.axis_name == active_name)
        value = float(active.value)
        if not math.isfinite(value):
            raise AssertionError(
                f"{execution.case_id} reachable-gap input is not finite scalar"
            )
        return value

    below_input = active_value(below)
    above_input = active_value(above)
    if PointRole.AT in executions:
        at_input = active_value(executions[PointRole.AT])
        adjacent = (
            float(np.nextafter(below_input, math.inf)) == at_input
            and float(np.nextafter(at_input, math.inf)) == above_input
        )
    else:
        adjacent = float(np.nextafter(below_input, math.inf)) == above_input
    if not adjacent:
        raise AssertionError(
            f"{suite.gate_id} reachable inputs are not adjacent representable floats"
        )

    rendered = "\n".join(suite.ambiguities)
    missing_hex = [
        value.hex()
        for value in (threshold, below_output, above_output)
        if value.hex() not in rendered
    ]
    if missing_hex:
        raise AssertionError(
            f"{suite.gate_id} ambiguity omitted measured float.hex values "
            f"{missing_hex!r}"
        )


def _exact_scalar_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.generic):
        left = left.item()
    if isinstance(right, np.generic):
        right = right.item()
    return type(left) is type(right) and left == right


def _finite_numeric_pair(left: Any, right: Any) -> tuple[float, float]:
    if isinstance(left, bool) or isinstance(right, bool):
        raise TypeError("a numerical boundary quantity cannot be bool")
    try:
        first = float(left)
        second = float(right)
    except (TypeError, ValueError, OverflowError) as error:
        raise AssertionError(
            "the boundary role needs scalar numerical values"
        ) from error
    if not (math.isfinite(first) and math.isfinite(second)):
        raise AssertionError("ordinary boundary neighbours must be finite")
    return first, second


def _worst_verdict(
    verdicts: Iterable[oracles.NumericalVerdict],
) -> oracles.NumericalVerdict:
    ranking = {
        oracles.NumericalVerdict.OK: 0,
        oracles.NumericalVerdict.WARN: 1,
        oracles.NumericalVerdict.BAD: 2,
    }
    return max(verdicts, key=ranking.__getitem__, default=oracles.NumericalVerdict.OK)


def axis_samples(axes: tuple[AxisRange, ...]) -> tuple[AxisSample, ...]:
    """Materialize all five required locations for every declared axis."""
    samples: list[AxisSample] = []
    for axis in axes:
        samples.extend(
            (
                AxisSample(axis.name, AxisPosition.VERY_LOW, axis.low),
                AxisSample(axis.name, AxisPosition.ENDPOINT_LOW, axis.endpoints[0]),
                AxisSample(axis.name, AxisPosition.ENDPOINT_HIGH, axis.endpoints[1]),
                AxisSample(axis.name, AxisPosition.VERY_HIGH, axis.high),
                AxisSample(axis.name, AxisPosition.EXTREME, axis.extreme),
            )
        )
    return tuple(samples)


def float_grid(
    *,
    below: GateSide,
    at: GateSide,
    above: GateSide,
    very_low: GateSide | None = None,
    very_high: GateSide | None = None,
    extreme: GateSide = GateSide.REFUSED,
    threshold: str = "T",
    include_relative: bool = True,
) -> tuple[ThresholdPoint, ...]:
    """Standard float grid with ULP and both relative delta scales."""
    low_side = below if very_low is None else very_low
    high_side = above if very_high is None else very_high
    lower_relative = (
        (
            ThresholdPoint(
                PointRole.BELOW_RELATIVE_1E6,
                f"{threshold} - relative 1e-6",
                "1e-6",
                below,
            ),
            ThresholdPoint(
                PointRole.BELOW_RELATIVE_1E12,
                f"{threshold} - relative 1e-12",
                "1e-12",
                below,
            ),
        )
        if include_relative
        else ()
    )
    upper_relative = (
        (
            ThresholdPoint(
                PointRole.ABOVE_RELATIVE_1E12,
                f"{threshold} + relative 1e-12",
                "1e-12",
                above,
            ),
            ThresholdPoint(
                PointRole.ABOVE_RELATIVE_1E6,
                f"{threshold} + relative 1e-6",
                "1e-6",
                above,
            ),
        )
        if include_relative
        else ()
    )
    return (
        ThresholdPoint(PointRole.VERY_LOW, "axis low", "axis-low", low_side),
        *lower_relative,
        ThresholdPoint(
            PointRole.BELOW_ULP,
            f"nextafter({threshold}, -inf)",
            "one ULP",
            below,
        ),
        ThresholdPoint(PointRole.AT, threshold, "zero", at),
        ThresholdPoint(
            PointRole.ABOVE_ULP,
            f"nextafter({threshold}, +inf)",
            "one ULP",
            above,
        ),
        *upper_relative,
        ThresholdPoint(PointRole.VERY_HIGH, "axis high", "axis-high", high_side),
        ThresholdPoint(PointRole.EXTREME, "IEEE/domain extremes", "extreme", extreme),
    )


def integer_grid(
    *,
    below: GateSide,
    at: GateSide,
    above: GateSide,
    very_low: GateSide | None = None,
    very_high: GateSide | None = None,
    extreme: GateSide = GateSide.REFUSED,
    threshold: str = "T",
) -> tuple[ThresholdPoint, ...]:
    """Standard integer neighbour grid; it deliberately contains no ULPs."""
    low_side = below if very_low is None else very_low
    high_side = above if very_high is None else very_high
    return (
        ThresholdPoint(PointRole.VERY_LOW, "axis low", "axis-low", low_side),
        ThresholdPoint(
            PointRole.BELOW_INTEGER, f"{threshold} - 1", "one integer", below
        ),
        ThresholdPoint(PointRole.AT, threshold, "zero", at),
        ThresholdPoint(
            PointRole.ABOVE_INTEGER, f"{threshold} + 1", "one integer", above
        ),
        ThresholdPoint(PointRole.VERY_HIGH, "axis high", "axis-high", high_side),
        ThresholdPoint(
            PointRole.EXTREME, "integer/domain extremes", "extreme", extreme
        ),
    )


def exact_grid(
    *,
    exact: GateSide = GateSide.ADMITTED,
    mismatch: GateSide = GateSide.REFUSED,
    very_low: GateSide = GateSide.ADMITTED,
    very_high: GateSide = GateSide.ADMITTED,
    extreme: GateSide = GateSide.REFUSED,
) -> tuple[ThresholdPoint, ...]:
    """Exact evidence grid with one-ULP, subnormal, and material mismatches."""
    return (
        ThresholdPoint(PointRole.VERY_LOW, "small exact fixture", "axis-low", very_low),
        ThresholdPoint(PointRole.EXACT, "exact evidence", "zero", exact),
        ThresholdPoint(
            PointRole.ULP_MISMATCH,
            "one-ULP mismatch",
            "one ULP",
            mismatch,
        ),
        ThresholdPoint(
            PointRole.SUBNORMAL_MISMATCH,
            "minimum-subnormal mismatch",
            "minimum subnormal",
            mismatch,
        ),
        ThresholdPoint(
            PointRole.MATERIAL_MISMATCH,
            "material mismatch",
            "1e-6",
            mismatch,
        ),
        ThresholdPoint(
            PointRole.VERY_HIGH, "large exact fixture", "axis-high", very_high
        ),
        ThresholdPoint(
            PointRole.EXTREME, "layout/dtype/domain extremes", "extreme", extreme
        ),
    )


def capability_grid(
    *,
    low: GateSide | None = None,
    valid: GateSide = GateSide.ADMITTED,
    invalid: GateSide = GateSide.REFUSED,
    high: GateSide | None = None,
    extreme: GateSide = GateSide.REFUSED,
) -> tuple[ThresholdPoint, ...]:
    """Five real capability regions without fabricated floating neighbours."""
    low_side = valid if low is None else low
    high_side = invalid if high is None else high
    return (
        ThresholdPoint(
            PointRole.CAPABILITY_LOW,
            "low valid capability",
            "category-low",
            low_side,
        ),
        ThresholdPoint(PointRole.VALID_CAPABILITY, "valid capability", "exact", valid),
        ThresholdPoint(
            PointRole.INVALID_CAPABILITY, "invalid capability", "exact", invalid
        ),
        ThresholdPoint(
            PointRole.CAPABILITY_HIGH,
            "high invalid capability",
            "category-high",
            high_side,
        ),
        ThresholdPoint(
            PointRole.EXTREME, "capability/domain extremes", "extreme", extreme
        ),
    )


def make_case(
    *,
    entry: GateEntry,
    point: ThresholdPoint,
    fixture_family: FixtureFamily,
    execution_class: ExecutionClass,
    topology: BoundaryTopology,
    direct_methods: tuple[str, ...],
    independent_oracles: tuple[str, ...],
    runner: CaseRunner,
    atom_ids: tuple[str, ...] = (),
    atom_relation: AtomRelation | None = None,
    non_unit_scale: tuple[float, float] | None = None,
    suffix: str | None = None,
    active_axis: str | None = None,
) -> BoundaryCase:
    """Construct one stable case; providers still own all semantics."""
    if len(entry.axes) > 1 and active_axis is None:
        raise ValueError(f"{entry.gate_id} compound case needs an active axis")
    resolved_active_axis = (
        entry.axes[0].name if len(entry.axes) == 1 else active_axis
    )
    point_suffix = point.role.value if suffix is None else suffix
    return BoundaryCase(
        case_id=f"{entry.gate_id}::boundary::{point_suffix}",
        gate_id=entry.gate_id,
        atom_ids=atom_ids,
        fixture_family=fixture_family,
        execution_class=execution_class,
        topology=topology,
        threshold_point=point,
        axes=axis_samples(entry.axes),
        direct_methods=direct_methods,
        independent_oracles=independent_oracles,
        non_unit_scale=non_unit_scale,
        atom_relation=atom_relation,
        runner=runner,
        active_axis=resolved_active_axis,
    )


def make_grid_cases(
    *,
    entry: GateEntry,
    points: Iterable[ThresholdPoint],
    fixture_family: FixtureFamily,
    execution_class: ExecutionClass,
    topology: BoundaryTopology,
    direct_methods: tuple[str, ...],
    independent_oracles: tuple[str, ...],
    runner: CaseRunner,
    atoms_by_role: Mapping[PointRole, tuple[str, ...]] | None = None,
    non_unit_scale: tuple[float, float] | None = None,
    active_axis: str | None = None,
) -> tuple[BoundaryCase, ...]:
    """Build a standard grid while retaining provider-owned point execution."""
    atom_map = {} if atoms_by_role is None else dict(atoms_by_role)
    return tuple(
        make_case(
            entry=entry,
            point=point,
            fixture_family=fixture_family,
            execution_class=execution_class,
            topology=topology,
            direct_methods=direct_methods,
            independent_oracles=independent_oracles,
            runner=runner,
            atom_ids=atom_map.get(point.role, ()),
            non_unit_scale=non_unit_scale,
            active_axis=active_axis,
        )
        for point in points
    )


def make_atom_case(
    *,
    entry: GateEntry,
    atom_id: str,
    relation: AtomRelation,
    point: ThresholdPoint,
    fixture_family: FixtureFamily,
    execution_class: ExecutionClass,
    topology: BoundaryTopology,
    direct_methods: tuple[str, ...],
    independent_oracles: tuple[str, ...],
    runner: CaseRunner,
    non_unit_scale: tuple[float, float] | None = None,
    active_axis: str | None = None,
) -> BoundaryCase:
    """Construct one uniquely named, relation-aware executable atomic cell."""
    if atom_id not in isolatable_atom_ids(entry):
        raise ValueError(
            f"{atom_id!r} is static or outside the dynamic atoms of {entry.gate_id}"
        )
    validate_atom_relation(entry=entry, atom_id=atom_id, relation=relation)
    digest = hashlib.sha256(atom_id.encode()).hexdigest()[:16]
    return make_case(
        entry=entry,
        point=point,
        fixture_family=fixture_family,
        execution_class=execution_class,
        topology=topology,
        direct_methods=direct_methods,
        independent_oracles=independent_oracles,
        runner=runner,
        atom_ids=(atom_id,),
        atom_relation=relation,
        non_unit_scale=non_unit_scale,
        suffix=f"atom-{digest}",
        active_axis=active_axis,
    )


def freeze_suite(
    *,
    gate_id: str,
    fixture_family: FixtureFamily,
    execution_class: ExecutionClass,
    topology: BoundaryTopology,
    cases: Iterable[BoundaryCase],
    atom_case_ids: Mapping[str, str],
    tighten_case_id: str,
    loosen_case_id: str,
    ambiguities: tuple[str, ...] = (),
    omitted_unrepresentable_roles: frozenset[PointRole] = frozenset(),
) -> BoundarySuite:
    """Freeze provider output after checking local identities."""
    materialized = tuple(cases)
    case_ids = [case.case_id for case in materialized]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"{gate_id} has duplicate boundary case IDs")
    if any(case.gate_id != gate_id for case in materialized):
        raise ValueError(f"{gate_id} suite contains a foreign gate case")
    if tighten_case_id not in case_ids or loosen_case_id not in case_ids:
        raise ValueError(f"{gate_id} mutation witnesses do not name suite cases")
    if set(atom_case_ids.values()) - set(case_ids):
        raise ValueError(f"{gate_id} has atomic references to absent cases")
    if len(set(atom_case_ids.values())) != len(atom_case_ids):
        raise ValueError(f"{gate_id} reuses one case for multiple atomic premises")
    entry = _ENTRY_BY_GATE[gate_id]
    required_atom_ids = set(isolatable_atom_ids(entry))
    if set(atom_case_ids) != required_atom_ids:
        missing = sorted(required_atom_ids - set(atom_case_ids))
        extra = sorted(set(atom_case_ids) - required_atom_ids)
        raise ValueError(
            f"{gate_id} isolatable atomic cases differ from the registry; "
            f"missing={missing!r}, extra={extra!r}"
        )
    required_ambiguities = set(entry.atom_isolation_ambiguities.values())
    if not required_ambiguities <= set(ambiguities):
        missing = sorted(required_ambiguities - set(ambiguities))
        raise ValueError(
            f"{gate_id} omitted registered atom isolation ambiguities: {missing!r}"
        )
    cases_by_id = {case.case_id: case for case in materialized}
    for atom_id, case_id in atom_case_ids.items():
        atom_case = cases_by_id[case_id]
        if atom_case.atom_ids != (atom_id,):
            raise ValueError(
                f"{gate_id} atomic case {case_id} does not isolate {atom_id}"
            )
        if atom_case.atom_relation is None:
            raise ValueError(f"{gate_id} atomic case {case_id} has no atom relation")
        validate_atom_relation(
            entry=entry,
            atom_id=atom_id,
            relation=atom_case.atom_relation,
        )
    return BoundarySuite(
        gate_id=gate_id,
        fixture_family=fixture_family,
        execution_class=execution_class,
        topology=topology,
        cases=materialized,
        atom_case_ids=atom_case_ids,
        tighten_case_id=tighten_case_id,
        loosen_case_id=loosen_case_id,
        ambiguities=ambiguities,
        omitted_unrepresentable_roles=omitted_unrepresentable_roles,
    )


__all__ = [
    "AtomBaseline",
    "AtomDependencyLogic",
    "AtomEvidence",
    "AtomPrerequisite",
    "AtomReducer",
    "AtomRelation",
    "AtomRelationKind",
    "AxisPosition",
    "AxisSample",
    "BoundaryCase",
    "BoundaryExecution",
    "BoundarySuite",
    "BoundaryTopology",
    "CaseReference",
    "CaseRunner",
    "DirectEvaluation",
    "ExecutionClass",
    "FixtureFamily",
    "GateSide",
    "OracleCheck",
    "PointRole",
    "RawObservation",
    "RealizedAxis",
    "RealizedPoint",
    "ThresholdPoint",
    "axis_samples",
    "axis_position_for_role",
    "capability_grid",
    "execute_case",
    "exact_grid",
    "fingerprint_realization",
    "float_grid",
    "freeze_suite",
    "integer_grid",
    "make_case",
    "make_atom_case",
    "make_grid_cases",
    "numerical_evaluation",
    "oracle_check",
    "realized_point",
    "source_alias_canonical",
    "source_ast_prerequisites",
    "validate_atom_evidence",
    "validate_atom_relation",
]
