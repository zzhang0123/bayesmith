"""Behavioral tests for the boundary harness's realized-input audit."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.numerical_gates.boundary_core import (
    AtomBaseline,
    AtomDependencyLogic,
    AtomEvidence,
    AtomPrerequisite,
    AtomReducer,
    AtomRelation,
    AtomRelationKind,
    AxisPosition,
    AxisSample,
    BoundaryCase,
    BoundarySuite,
    BoundaryTopology,
    ExecutionClass,
    FixtureFamily,
    GateSide,
    OracleCheck,
    PointRole,
    RawObservation,
    RealizedAxis,
    RealizedPoint,
    ThresholdPoint,
    _validate_dependency_truth,
    float_grid,
    realized_point,
    source_alias_canonical,
    source_ast_prerequisites,
    validate_atom_evidence,
    validate_atom_relation,
    validate_axis_position_values,
    validate_reachable_gap,
)
from tests.numerical_gates.registry import (
    GATE_REGISTRY,
    AxisRange,
    MutationMode,
    isolatable_atom_ids,
)

_TWO_SIDED_WITH_ATOMS = tuple(
    entry
    for entry in GATE_REGISTRY
    if entry.mutation_mode is MutationMode.TWO_SIDED and isolatable_atom_ids(entry)
)


def _alias_example() -> tuple[object, str, str]:
    for entry in _TWO_SIDED_WITH_ATOMS:
        for atom_id in isolatable_atom_ids(entry):
            canonical = source_alias_canonical(entry, atom_id)
            if canonical != atom_id:
                alias = atom_id
                return entry, canonical, alias
    raise AssertionError("the committed registry unexpectedly has no source aliases")


def _ast_dependency_example() -> tuple[object, str, tuple[str, ...]]:
    for entry in _TWO_SIDED_WITH_ATOMS:
        for atom_id in isolatable_atom_ids(entry):
            prerequisites = source_ast_prerequisites(entry, atom_id)
            if prerequisites:
                return entry, atom_id, prerequisites
    raise AssertionError("the committed registry unexpectedly has no nested atoms")


def _baselines(entry: object, *, outcome: bool = True) -> tuple[AtomBaseline, ...]:
    return tuple(
        AtomBaseline(atom_id=atom_id, outcome=outcome)
        for atom_id in entry.conjunction_atom_ids
        if source_alias_canonical(entry, atom_id) == atom_id
    )


def _atom_checks(evidence: tuple[AtomEvidence, ...]) -> tuple[OracleCheck, ...]:
    return tuple(
        OracleCheck(item.oracle, item.raw_actual, item.raw_actual)
        for item in evidence
    )


def test_oracle_check_derives_its_verdict_from_retained_values() -> None:
    """A provider cannot certify a contradiction by setting ``passed``."""
    contradiction = OracleCheck("literal equality", False, True)

    assert contradiction.actual is False
    assert contradiction.expected is True
    assert contradiction.passed is False


def test_raw_observation_retains_the_inputs_consumed_by_direct_calls() -> None:
    """Dropping the consumed-key audit would let metadata impersonate inputs."""
    point = RealizedPoint(
        quantity="minimum eigenvalue",
        input_key="matrix",
        value=2.0,
        threshold=2.0,
        dtype="float64",
        axes=(
            RealizedAxis(
                axis_name="scale",
                position=AxisPosition.ENDPOINT_LOW,
                input_key="matrix",
                value=2.0,
            ),
        ),
    )
    try:
        observation = RawObservation(
            observed_side=GateSide.ADMITTED,
            realized_point=point,
            realized_inputs={"matrix": 2.0},
            direct_calls=("direct_method",),
            direct_input_keys=("matrix",),
            direct_return_keys=(),
            oracle_checks=(OracleCheck("literal", 2.0, 2.0),),
        )
    except TypeError as error:
        pytest.fail(f"RawObservation lost the direct-input audit: {error}")

    assert observation.direct_input_keys == ("matrix",)
    assert observation.direct_return_keys == ()


def _case(
    observation: RawObservation,
    *,
    case_id: str = "audit::boundary::at",
    axis_names: tuple[str, ...] = ("scale",),
    active_axis: str | None = None,
    threshold_point: ThresholdPoint | None = None,
    execution_class: ExecutionClass = ExecutionClass.PAYLOAD_OR_REFUSAL,
) -> BoundaryCase:
    point = (
        ThresholdPoint(PointRole.AT, "2", "exact", GateSide.ADMITTED)
        if threshold_point is None
        else threshold_point
    )
    return BoundaryCase(
        case_id=case_id,
        gate_id="audit",
        atom_ids=(),
        fixture_family=FixtureFamily.SPD_SPECTRUM_CONDITION,
        execution_class=execution_class,
        topology=BoundaryTopology.FLOAT,
        threshold_point=point,
        axes=tuple(
            AxisSample(axis_name, AxisPosition.ENDPOINT_LOW, "2")
            for axis_name in axis_names
        ),
        direct_methods=("direct_method",),
        independent_oracles=("literal",),
        non_unit_scale=None,
        atom_relation=None,
        runner=lambda _point: observation,
        active_axis=active_axis,
    )


def test_axis_coverage_rejects_relabeling_one_input_as_five_positions() -> None:
    """Position names cannot make one frozen production input into an axis grid."""
    fixtures = (
        (PointRole.VERY_LOW, AxisPosition.VERY_LOW, 1.0),
        (PointRole.BELOW_ULP, AxisPosition.ENDPOINT_LOW, np.nextafter(2.0, -np.inf)),
        (PointRole.ABOVE_ULP, AxisPosition.ENDPOINT_HIGH, np.nextafter(2.0, np.inf)),
        (PointRole.VERY_HIGH, AxisPosition.VERY_HIGH, 3.0),
        (PointRole.EXTREME, AxisPosition.EXTREME, 4.0),
    )
    cases = []
    for role, position, quantity in fixtures:
        observation = _observation(
            point_input_key="quantity",
            point_value=quantity,
            axes=(RealizedAxis("scale", position, "matrix", 2.0),),
            realized_inputs={"matrix": 2.0, "quantity": quantity},
            direct_input_keys=("matrix",),
            direct_return_keys=("quantity",),
        )
        observation = replace(
            observation,
            realized_point=replace(observation.realized_point, threshold=2.0),
        )
        cases.append(
            _case(
                observation,
                case_id=f"audit::boundary::{role.value}",
                threshold_point=ThresholdPoint(
                    role, role.value, "audit", GateSide.ADMITTED
                ),
            )
        )
    suite = BoundarySuite(
        gate_id="audit",
        fixture_family=FixtureFamily.SPD_SPECTRUM_CONDITION,
        execution_class=ExecutionClass.PAYLOAD_OR_REFUSAL,
        topology=BoundaryTopology.FLOAT,
        cases=tuple(cases),
        atom_case_ids={},
        tighten_case_id=cases[0].case_id,
        loosen_case_id=cases[1].case_id,
    )

    with pytest.raises(AssertionError, match="reused.*axis position"):
        validate_axis_position_values(suite)


def test_compound_axis_sweep_rejects_a_drifting_companion_input() -> None:
    """Changing a companion can manufacture the active axis's gate crossing."""
    fixtures = (
        (PointRole.CAPABILITY_LOW, AxisPosition.VERY_LOW, 1.0),
        (PointRole.VALID_CAPABILITY, AxisPosition.ENDPOINT_LOW, 2.0),
        (PointRole.INVALID_CAPABILITY, AxisPosition.ENDPOINT_HIGH, 3.0),
        (PointRole.CAPABILITY_HIGH, AxisPosition.VERY_HIGH, 4.0),
        (PointRole.EXTREME, AxisPosition.EXTREME, 5.0),
    )
    cases = []
    for index, (role, position, active_value) in enumerate(fixtures):
        companion_value = 10.0 + index
        observation = _observation(
            point_input_key="active",
            point_value=active_value,
            axes=(
                RealizedAxis("active", position, "active", active_value),
                RealizedAxis(
                    "companion",
                    AxisPosition.INTERIOR,
                    "companion",
                    companion_value,
                ),
            ),
            realized_inputs={
                "active": active_value,
                "companion": companion_value,
            },
            direct_input_keys=("active", "companion"),
        )
        observation = replace(
            observation,
            realized_point=replace(
                observation.realized_point,
                threshold=3.0,
                active_axis="active",
            ),
        )
        cases.append(
            _case(
                observation,
                case_id=f"audit::compound::{role.value}",
                axis_names=("active", "companion"),
                active_axis="active",
                threshold_point=ThresholdPoint(
                    role,
                    role.value,
                    "audit",
                    GateSide.ADMITTED,
                ),
            )
        )
    suite = BoundarySuite(
        gate_id="audit",
        fixture_family=FixtureFamily.SPD_SPECTRUM_CONDITION,
        execution_class=ExecutionClass.PAYLOAD_OR_REFUSAL,
        topology=BoundaryTopology.CAPABILITY,
        cases=tuple(cases),
        atom_case_ids={},
        tighten_case_id=cases[0].case_id,
        loosen_case_id=cases[1].case_id,
    )

    with pytest.raises(AssertionError, match="companion.*changed"):
        validate_axis_position_values(suite)


def test_reachable_gap_rejects_far_points_labeled_as_nearest() -> None:
    """A prose ambiguity cannot certify that a representable input gap is exact."""
    cases = []
    for role, position, value in (
        (PointRole.REACHABLE_BELOW, AxisPosition.ENDPOINT_LOW, 1.0),
        (PointRole.REACHABLE_ABOVE, AxisPosition.ENDPOINT_HIGH, 3.0),
    ):
        observation = _observation(
            point_value=value,
            axes=(RealizedAxis("scale", position, "matrix", value),),
            realized_inputs={"matrix": value},
        )
        cases.append(
            _case(
                observation,
                case_id=f"audit::boundary::{role.value}",
                threshold_point=ThresholdPoint(
                    role, role.value, "claimed nearest", GateSide.ADMITTED
                ),
            )
        )
    suite = BoundarySuite(
        gate_id="audit",
        fixture_family=FixtureFamily.SPD_SPECTRUM_CONDITION,
        execution_class=ExecutionClass.PAYLOAD_OR_REFUSAL,
        topology=BoundaryTopology.FLOAT,
        cases=tuple(cases),
        atom_case_ids={},
        tighten_case_id=cases[0].case_id,
        loosen_case_id=cases[1].case_id,
        ambiguities=("T=0x1.0000000000000p+1; below=0x1p+0; above=0x1.8p+1",),
        omitted_unrepresentable_roles=frozenset(
            {PointRole.BELOW_ULP, PointRole.AT, PointRole.ABOVE_ULP}
        ),
    )

    with pytest.raises(AssertionError, match="adjacent"):
        validate_reachable_gap(suite)


def test_two_payload_case_requires_a_numerical_evaluation() -> None:
    """A selector cannot claim both payloads were checked with Boolean metadata."""
    with pytest.raises(AssertionError, match="two-payload"):
        _case(
            _observation(),
            execution_class=ExecutionClass.TWO_PAYLOAD,
        )()


def _observation(
    *,
    point_input_key: str = "matrix",
    point_value: object = 2.0,
    axes: tuple[RealizedAxis, ...] | None = None,
    realized_inputs: dict[str, object] | None = None,
    direct_input_keys: tuple[str, ...] = ("matrix",),
    direct_return_keys: tuple[str, ...] = (),
) -> RawObservation:
    if axes is None:
        axes = (
            RealizedAxis(
                axis_name="scale",
                position=AxisPosition.ENDPOINT_LOW,
                input_key="matrix",
                value=2.0,
            ),
        )
    if realized_inputs is None:
        realized_inputs = {"matrix": 2.0}
    return RawObservation(
        observed_side=GateSide.ADMITTED,
        realized_point=RealizedPoint(
            quantity="minimum eigenvalue",
            input_key=point_input_key,
            value=point_value,
            threshold=2.0,
            dtype="float64",
            axes=axes,
        ),
        realized_inputs=realized_inputs,
        direct_input_keys=direct_input_keys,
        direct_return_keys=direct_return_keys,
        direct_calls=("direct_method",),
        oracle_checks=(OracleCheck("literal", 2.0, 2.0),),
    )


def test_execute_case_rejects_more_than_one_realization_of_a_declared_axis() -> None:
    """One call cannot claim five axis positions by attaching five labels."""
    observation = _observation(
        axes=(
            RealizedAxis("scale", AxisPosition.ENDPOINT_LOW, "matrix", 2.0),
            RealizedAxis("scale", AxisPosition.ENDPOINT_HIGH, "matrix", 2.0),
        )
    )

    with pytest.raises(AssertionError, match="repeated|exactly one"):
        _case(observation)()


def test_each_realized_axis_position_must_match_the_boundary_role() -> None:
    """One valid axis label cannot conceal another axis at the wrong location."""
    observation = _observation(
        axes=(
            RealizedAxis("scale", AxisPosition.ENDPOINT_LOW, "matrix", 2.0),
            RealizedAxis("rank", AxisPosition.EXTREME, "rank", 3),
        ),
        realized_inputs={"matrix": 2.0, "rank": 3},
        direct_input_keys=("matrix", "rank"),
    )
    observation = replace(
        observation,
        realized_point=replace(observation.realized_point, active_axis="scale"),
    )

    with pytest.raises(AssertionError, match="companion axis"):
        _case(
            observation,
            axis_names=("scale", "rank"),
            active_axis="scale",
        )()


def test_compound_realized_point_requires_one_explicit_active_axis() -> None:
    """A bundle of real fields cannot all inherit one synthetic boundary role."""
    entry = replace(
        GATE_REGISTRY[0],
        axes=(
            AxisRange("margin", "low", ("below", "at"), "high", "extreme"),
            AxisRange(
                "tolerance",
                "low",
                ("below", "at"),
                "high",
                "extreme",
            ),
        ),
    )
    point = ThresholdPoint(PointRole.AT, "0", "zero", GateSide.ADMITTED)

    with pytest.raises(ValueError, match="compound.*active axis"):
        realized_point(
            entry=entry,
            point=point,
            quantity="constructor fields",
            input_key="margin",
            value=0.0,
            threshold=0.0,
            dtype="float64",
        )

    realized = realized_point(
        entry=entry,
        point=point,
        quantity="constructor fields",
        input_key="margin",
        value=0.0,
        threshold=0.0,
        dtype="float64",
        active_axis="margin",
        axis_bindings={
            "margin": ("margin", 0.0, AxisPosition.ENDPOINT_LOW),
            "tolerance": ("tolerance", 1e-3, AxisPosition.INTERIOR),
        },
    )

    assert realized.active_axis == "margin"
    assert {axis.axis_name: axis.position for axis in realized.axes} == {
        "margin": AxisPosition.ENDPOINT_LOW,
        "tolerance": AxisPosition.INTERIOR,
    }


def test_execute_case_rejects_synthetic_boundary_quantity_metadata() -> None:
    """A provider cannot substitute an unattached ordinal for a consumed value."""
    observation = _observation(
        point_input_key="boundary_quantity",
        realized_inputs={"matrix": 2.0, "boundary_quantity": 2.0},
    )

    with pytest.raises(AssertionError, match="boundary_quantity"):
        _case(observation)()


def test_axes_must_link_to_direct_inputs_not_derived_returns() -> None:
    """Axis coverage is proved by the fixture passed into the direct call."""
    observation = _observation(
        point_input_key="predicate_result",
        axes=(
            RealizedAxis(
                "scale",
                AxisPosition.ENDPOINT_LOW,
                "predicate_result",
                2.0,
            ),
        ),
        realized_inputs={"matrix": 2.0, "predicate_result": 2.0},
        direct_return_keys=("predicate_result",),
    )

    with pytest.raises(AssertionError, match="axis .* direct-call input"):
        _case(observation)()


def test_derived_gate_quantity_may_link_to_a_real_direct_return() -> None:
    """Condition/rho/error certificates are legitimate measured returns."""
    observation = _observation(
        point_input_key="predicate_result",
        realized_inputs={"matrix": 2.0, "predicate_result": 2.0},
        direct_return_keys=("predicate_result",),
    )

    execution = _case(observation)()

    assert execution.direct_input_keys == ("matrix",)
    assert execution.direct_return_keys == ("predicate_result",)


def test_nonzero_subnormal_relative_boundary_uses_the_threshold_scale() -> None:
    """The rel-error 1e-300 floor must not leak into neighbour construction."""
    threshold = np.float64(np.finfo(np.float64).tiny / 2.0)
    value = np.float64(threshold * (1.0 + 1e-6))
    point = ThresholdPoint(
        PointRole.ABOVE_RELATIVE_1E6,
        "subnormal T + relative 1e-6",
        "1e-6",
        GateSide.ADMITTED,
    )
    observation = _observation(
        point_input_key="matrix",
        point_value=value,
        axes=(RealizedAxis("scale", AxisPosition.ENDPOINT_HIGH, "matrix", value),),
        realized_inputs={"matrix": value},
    )
    observation = RawObservation(
        observed_side=observation.observed_side,
        realized_point=RealizedPoint(
            quantity=observation.realized_point.quantity,
            input_key=observation.realized_point.input_key,
            value=value,
            threshold=threshold,
            dtype=np.dtype(np.float64).str,
            axes=observation.realized_point.axes,
        ),
        realized_inputs=observation.realized_inputs,
        direct_input_keys=observation.direct_input_keys,
        direct_return_keys=observation.direct_return_keys,
        direct_calls=observation.direct_calls,
        oracle_checks=observation.oracle_checks,
    )

    execution = _case(observation, threshold_point=point)()

    assert execution.realized_point.value == value


def test_float_grid_can_explicitly_omit_unrepresentable_relative_roles() -> None:
    """A zero threshold uses exact/ULP neighbours instead of fake deltas."""
    try:
        points = float_grid(
            below=GateSide.REFUSED,
            at=GateSide.ADMITTED,
            above=GateSide.ADMITTED,
            threshold="0",
            include_relative=False,
        )
    except TypeError as error:
        pytest.fail(f"float_grid lacks an explicit relative-role policy: {error}")

    roles = {point.role for point in points}
    assert not roles & {
        PointRole.BELOW_RELATIVE_1E6,
        PointRole.BELOW_RELATIVE_1E12,
        PointRole.ABOVE_RELATIVE_1E12,
        PointRole.ABOVE_RELATIVE_1E6,
    }
    assert {PointRole.BELOW_ULP, PointRole.AT, PointRole.ABOVE_ULP} <= roles


def _reachable_gap_suite(
    *,
    ambiguities: tuple[str, ...],
    omitted: frozenset[PointRole],
    include_above: bool = True,
) -> BoundarySuite:
    below_point = ThresholdPoint(
        PointRole.REACHABLE_BELOW,
        "nearest reachable value below T",
        "-33554431.5 threshold ULPs",
        GateSide.REFUSED,
    )
    above_point = ThresholdPoint(
        PointRole.REACHABLE_ABOVE,
        "nearest reachable value above T",
        "+33554430 threshold ULPs",
        GateSide.ADMITTED,
    )
    below_case = replace(
        _case(_observation(), threshold_point=below_point),
        case_id="audit::reachable-gap::below",
    )
    cases = [below_case]
    if include_above:
        cases.append(
            replace(
                _case(_observation(), threshold_point=above_point),
                case_id="audit::reachable-gap::above",
            )
        )
    return BoundarySuite(
        gate_id="audit",
        fixture_family=FixtureFamily.FACTOR_CERTIFICATES,
        execution_class=ExecutionClass.PAYLOAD_OR_REFUSAL,
        topology=BoundaryTopology.FLOAT,
        cases=tuple(cases),
        atom_case_ids={},
        tighten_case_id=below_case.case_id,
        loosen_case_id=cases[-1].case_id,
        ambiguities=ambiguities,
        omitted_unrepresentable_roles=omitted,
    )


def test_quantized_output_gap_requires_measured_straddles_and_ambiguity() -> None:
    """A derived metric may omit exact T/ULP roles only with quantified evidence."""
    omitted = frozenset(
        {
            PointRole.BELOW_ULP,
            PointRole.AT,
            PointRole.ABOVE_ULP,
        }
    )

    suite = _reachable_gap_suite(
        ambiguities=(
            "nearest outputs are -33554431.5 and +33554430 threshold ULPs",
        ),
        omitted=omitted,
    )

    assert suite.omitted_unrepresentable_roles == omitted
    with pytest.raises(ValueError, match="quantified ambiguity"):
        _reachable_gap_suite(ambiguities=(), omitted=omitted)
    with pytest.raises(ValueError, match="reachable straddles"):
        _reachable_gap_suite(
            ambiguities=("nearest output gap measured in threshold ULPs",),
            omitted=omitted,
            include_above=False,
        )


@pytest.mark.parametrize(
    ("role", "value"),
    [
        (PointRole.REACHABLE_BELOW, np.float64(1.0 - 2.0**-20)),
        (PointRole.REACHABLE_ABOVE, np.float64(1.0 + 2.0**-20)),
    ],
)
def test_reachable_gap_points_retain_the_real_strict_order(
    role: PointRole, value: np.float64
) -> None:
    """Nearest reachable evidence is checked against T, not trusted as prose."""
    point = ThresholdPoint(role, role.value, "measured gap", GateSide.ADMITTED)
    observation = _observation(
        point_value=value,
        axes=(
            RealizedAxis(
                "scale",
                AxisPosition.ENDPOINT_LOW
                if role is PointRole.REACHABLE_BELOW
                else AxisPosition.ENDPOINT_HIGH,
                "matrix",
                value,
            ),
        ),
        realized_inputs={"matrix": value},
    )
    observation = replace(
        observation,
        realized_point=replace(
            observation.realized_point,
            value=value,
            threshold=np.float64(1.0),
            dtype=np.dtype(np.float64).str,
        ),
    )

    _case(observation, threshold_point=point)()

    equal_observation = replace(
        observation,
        realized_point=replace(observation.realized_point, value=np.float64(1.0)),
        realized_inputs={"matrix": np.float64(1.0)},
    )
    with pytest.raises(AssertionError, match="strictly"):
        _case(equal_observation, threshold_point=point)()


def test_alias_relation_is_bound_to_the_scanner_canonical_ast_digest() -> None:
    """A duplicate scanner family shares one real failure case, not a fake flip."""
    entry, canonical, alias = _alias_example()
    relation = AtomRelation(
        kind=AtomRelationKind.ALIAS,
        baselines=_baselines(entry),
        target_outcome=False,
        canonical_atom_id=canonical,
    )

    validate_atom_relation(entry=entry, atom_id=alias, relation=relation)

    wrong = next(
        candidate
        for candidate in entry.conjunction_atom_ids
        if source_alias_canonical(entry, candidate) != canonical
    )
    with pytest.raises(ValueError, match="canonical source identity"):
        validate_atom_relation(
            entry=entry,
            atom_id=alias,
            relation=AtomRelation(
                kind=AtomRelationKind.ALIAS,
                baselines=_baselines(entry),
                target_outcome=False,
                canonical_atom_id=wrong,
            ),
        )


def test_nested_ast_atom_must_declare_source_backed_dependency() -> None:
    """A parent expression cannot pretend its contained child stayed true."""
    entry, atom_id, prerequisites = _ast_dependency_example()

    with pytest.raises(ValueError, match="AST-containment"):
        validate_atom_relation(
            entry=entry,
            atom_id=atom_id,
            relation=AtomRelation(
                kind=AtomRelationKind.INDEPENDENT,
                baselines=_baselines(entry),
                target_outcome=False,
            ),
        )

    relation = AtomRelation(
        kind=AtomRelationKind.DEPENDENT,
        baselines=_baselines(entry),
        target_outcome=False,
        prerequisites=tuple(
            AtomPrerequisite(
                atom_id=prerequisite,
                expected_outcome=False,
            )
            for prerequisite in prerequisites
        ),
        logic=AtomDependencyLogic.ALL_ELEMENTS,
        rationale="the outer np.all truth is reduced from its registered child",
    )
    validate_atom_relation(entry=entry, atom_id=atom_id, relation=relation)


def test_logical_dependency_requires_a_reviewable_reason() -> None:
    """Non-AST co-failure is explicit evidence, never an unexplained exemption."""
    entry, atom_id, prerequisites = _ast_dependency_example()
    other = next(
        candidate
        for candidate in entry.conjunction_atom_ids
        if candidate not in {atom_id, *prerequisites}
        and source_alias_canonical(entry, candidate) == candidate
    )
    relation = AtomRelation(
        kind=AtomRelationKind.DEPENDENT,
        baselines=_baselines(entry),
        target_outcome=False,
        prerequisites=(
            *(
                AtomPrerequisite(
                    atom_id=prerequisite,
                    expected_outcome=False,
                )
                for prerequisite in prerequisites
            ),
            AtomPrerequisite(
                atom_id=other,
                expected_outcome=False,
            ),
        ),
        logic=AtomDependencyLogic.PREREQUISITES_IMPLY_TARGET,
    )

    with pytest.raises(ValueError, match="reason"):
        validate_atom_relation(entry=entry, atom_id=atom_id, relation=relation)


def test_relation_specific_evidence_accepts_real_alias_and_dependent_cofailure() -> (
    None
):
    """Every atom is named, while only genuinely independent siblings stay true."""
    alias_entry, canonical, alias = _alias_example()
    alias_relation = AtomRelation(
        kind=AtomRelationKind.ALIAS,
        baselines=_baselines(alias_entry),
        target_outcome=False,
        canonical_atom_id=canonical,
    )
    alias_group = {
        candidate
        for candidate in alias_entry.conjunction_atom_ids
        if source_alias_canonical(alias_entry, candidate) == canonical
    }
    alias_evidence = tuple(
        AtomEvidence(
            atom_id=candidate,
            raw_actual=candidate not in alias_group,
            truth=candidate not in alias_group,
            reducer=AtomReducer.SCALAR,
            realized_keys=("fixture",),
            oracle="independent atom oracle",
        )
        for candidate in alias_entry.conjunction_atom_ids
    )
    validate_atom_evidence(
        entry=alias_entry,
        atom_id=alias,
        relation=alias_relation,
        evidence=alias_evidence,
        available_realized_keys=frozenset({"fixture"}),
        independent_oracles=frozenset({"independent atom oracle"}),
        oracle_checks=_atom_checks(alias_evidence),
    )

    entry, atom_id, prerequisites = _ast_dependency_example()
    relation = AtomRelation(
        kind=AtomRelationKind.DEPENDENT,
        baselines=_baselines(entry),
        target_outcome=False,
        prerequisites=tuple(
            AtomPrerequisite(
                atom_id=prerequisite,
                expected_outcome=False,
            )
            for prerequisite in prerequisites
        ),
        logic=AtomDependencyLogic.ALL_ELEMENTS,
        rationale="the outer np.all truth is reduced from its registered child",
    )
    failed_canonicals = {
        source_alias_canonical(entry, candidate)
        for candidate in (atom_id, *prerequisites)
    }
    evidence = tuple(
        AtomEvidence(
            atom_id=candidate,
            raw_actual=source_alias_canonical(entry, candidate)
            not in failed_canonicals,
            truth=source_alias_canonical(entry, candidate) not in failed_canonicals,
            reducer=AtomReducer.SCALAR,
            realized_keys=("fixture",),
            oracle="independent atom oracle",
        )
        for candidate in entry.conjunction_atom_ids
    )
    validate_atom_evidence(
        entry=entry,
        atom_id=atom_id,
        relation=relation,
        evidence=evidence,
        available_realized_keys=frozenset({"fixture"}),
        independent_oracles=frozenset({"independent atom oracle"}),
        oracle_checks=_atom_checks(evidence),
    )


def test_dependent_evidence_rejects_an_undeclared_sibling_failure() -> None:
    entry, atom_id, prerequisites = _ast_dependency_example()
    relation = AtomRelation(
        kind=AtomRelationKind.DEPENDENT,
        baselines=_baselines(entry),
        target_outcome=False,
        prerequisites=tuple(
            AtomPrerequisite(
                atom_id=prerequisite,
                expected_outcome=False,
            )
            for prerequisite in prerequisites
        ),
        logic=AtomDependencyLogic.ALL_ELEMENTS,
        rationale="the outer np.all truth is reduced from its registered child",
    )
    evidence = tuple(
        AtomEvidence(
            atom_id=candidate,
            raw_actual=False,
            truth=False,
            reducer=AtomReducer.SCALAR,
            realized_keys=("fixture",),
            oracle="independent atom oracle",
        )
        for candidate in entry.conjunction_atom_ids
    )

    with pytest.raises(AssertionError, match="undeclared sibling"):
        validate_atom_evidence(
            entry=entry,
            atom_id=atom_id,
            relation=relation,
            evidence=evidence,
            available_realized_keys=frozenset({"fixture"}),
            independent_oracles=frozenset({"independent atom oracle"}),
            oracle_checks=_atom_checks(evidence),
        )


def test_refusal_predicate_preserves_a_true_raw_target_outcome() -> None:
    """A refusal condition is flipped to True, not relabelled as a false premise."""
    entry = next(
        candidate
        for candidate in GATE_REGISTRY
        if candidate.gate_id == "COUPLING:_condition_number:positive-spectrum"
    )
    (atom_id,) = entry.conjunction_atom_ids
    relation = AtomRelation(
        kind=AtomRelationKind.INDEPENDENT,
        baselines=(AtomBaseline(atom_id=atom_id, outcome=False),),
        target_outcome=True,
    )
    evidence = (
        AtomEvidence(
            atom_id=atom_id,
            raw_actual=True,
            truth=True,
            reducer=AtomReducer.SCALAR,
            realized_keys=("eigenvalues",),
            oracle="independent positive-spectrum oracle",
        ),
    )

    validate_atom_evidence(
        entry=entry,
        atom_id=atom_id,
        relation=relation,
        evidence=evidence,
        available_realized_keys=frozenset({"eigenvalues"}),
        independent_oracles=frozenset({"independent positive-spectrum oracle"}),
        oracle_checks=_atom_checks(evidence),
    )


def test_array_atom_evidence_retains_raw_values_and_contextual_reducer() -> None:
    """The child ``sigma > 0`` evidence stays an array consumed by ``np.all``."""
    entry = next(
        candidate
        for candidate in GATE_REGISTRY
        if candidate.gate_id == "LADDER:structure:compact-diagonal-positive"
    )
    outer, child = entry.conjunction_atom_ids
    relation = AtomRelation(
        kind=AtomRelationKind.DEPENDENT,
        baselines=_baselines(entry),
        target_outcome=False,
        prerequisites=(AtomPrerequisite(atom_id=outer, expected_outcome=False),),
        logic=AtomDependencyLogic.TARGET_IMPLIES_PREREQUISITES,
        rationale="a false element makes the enclosing np.all predicate false",
    )
    raw_child = np.array([True, False])
    evidence = (
        AtomEvidence(
            atom_id=outer,
            raw_actual=False,
            truth=False,
            reducer=AtomReducer.SCALAR,
            realized_keys=("sigma",),
            oracle="independent positivity oracle",
        ),
        AtomEvidence(
            atom_id=child,
            raw_actual=raw_child,
            truth=False,
            reducer=AtomReducer.ALL_ELEMENTS,
            realized_keys=("sigma",),
            oracle="independent positivity oracle",
        ),
    )

    validate_atom_evidence(
        entry=entry,
        atom_id=child,
        relation=relation,
        evidence=evidence,
        available_realized_keys=frozenset({"sigma"}),
        independent_oracles=frozenset({"independent positivity oracle"}),
        oracle_checks=_atom_checks(evidence),
    )

    with pytest.raises(AssertionError, match="lineage"):
        validate_atom_evidence(
            entry=entry,
            atom_id=child,
            relation=relation,
            evidence=evidence,
            available_realized_keys=frozenset(),
            independent_oracles=frozenset({"independent positivity oracle"}),
            oracle_checks=_atom_checks(evidence),
        )


def test_all_of_parent_can_select_one_false_ast_child_with_other_children_true() -> (
    None
):
    """AST containment records possible dependencies, not forced mass co-failure."""
    entry, atom_id, prerequisites = next(
        (entry, atom_id, prerequisites)
        for entry in _TWO_SIDED_WITH_ATOMS
        for atom_id in entry.conjunction_atom_ids
        if len(prerequisites := source_ast_prerequisites(entry, atom_id)) >= 3
    )
    selected = prerequisites[0]
    relation = AtomRelation(
        kind=AtomRelationKind.DEPENDENT,
        baselines=_baselines(entry),
        target_outcome=False,
        prerequisites=(AtomPrerequisite(selected, False),),
        logic=AtomDependencyLogic.ALL_OF,
        rationale="one false child is sufficient to make an all-of parent false",
    )

    validate_atom_relation(entry=entry, atom_id=atom_id, relation=relation)

    selected_group = source_alias_canonical(entry, selected)
    target_group = source_alias_canonical(entry, atom_id)
    evidence = tuple(
        AtomEvidence(
            atom_id=candidate,
            raw_actual=source_alias_canonical(entry, candidate)
            not in {selected_group, target_group},
            truth=source_alias_canonical(entry, candidate)
            not in {selected_group, target_group},
            reducer=AtomReducer.SCALAR,
            realized_keys=("fixture",),
            oracle="independent atom oracle",
        )
        for candidate in entry.conjunction_atom_ids
    )
    validate_atom_evidence(
        entry=entry,
        atom_id=atom_id,
        relation=relation,
        evidence=evidence,
        available_realized_keys=frozenset({"fixture"}),
        independent_oracles=frozenset({"independent atom oracle"}),
        oracle_checks=_atom_checks(evidence),
    )


def test_dependency_implication_uses_declared_outcome_events_not_raw_polarity() -> None:
    relation = AtomRelation(
        kind=AtomRelationKind.DEPENDENT,
        baselines=(
            AtomBaseline("target", True),
            AtomBaseline("prerequisite", True),
        ),
        target_outcome=False,
        prerequisites=(AtomPrerequisite("prerequisite", False),),
        logic=AtomDependencyLogic.PREREQUISITES_IMPLY_TARGET,
        rationale="the declared prerequisite failure forces the target failure",
    )
    evidence = {
        "target": AtomEvidence(
            "target", True, True, AtomReducer.SCALAR, ("fixture",), "oracle"
        ),
        "prerequisite": AtomEvidence(
            "prerequisite",
            False,
            False,
            AtomReducer.SCALAR,
            ("fixture",),
            "oracle",
        ),
    }

    with pytest.raises(AssertionError, match="dependency logic"):
        _validate_dependency_truth(
            atom_id="target",
            relation=relation,
            evidence=evidence,
            actual={"target": True, "prerequisite": False},
        )


def test_equivalent_dependency_rejects_mixed_prerequisite_truths() -> None:
    relation = AtomRelation(
        kind=AtomRelationKind.DEPENDENT,
        baselines=(
            AtomBaseline("target", True),
            AtomBaseline("left", True),
            AtomBaseline("right", False),
        ),
        target_outcome=False,
        prerequisites=(
            AtomPrerequisite("left", False),
            AtomPrerequisite("right", True),
        ),
        logic=AtomDependencyLogic.EQUIVALENT,
        rationale="every selected co-failure must have the target truth",
    )
    evidence = {
        atom_id: AtomEvidence(
            atom_id, truth, truth, AtomReducer.SCALAR, ("fixture",), "oracle"
        )
        for atom_id, truth in {"target": False, "left": False, "right": True}.items()
    }

    with pytest.raises(AssertionError, match="dependency logic"):
        _validate_dependency_truth(
            atom_id="target",
            relation=relation,
            evidence=evidence,
            actual={"target": False, "left": False, "right": True},
        )


def test_short_circuited_companion_is_not_falsely_reported_as_false() -> None:
    """An expression Python never evaluated retains an explicit third state."""
    relation = AtomRelation(
        kind=AtomRelationKind.DEPENDENT,
        baselines=(
            AtomBaseline("target", True),
            AtomBaseline("skipped", True),
        ),
        target_outcome=False,
        prerequisites=(AtomPrerequisite("skipped", None),),
        logic=AtomDependencyLogic.SHORT_CIRCUIT,
        rationale="target failure prevents the later comparison from executing",
    )
    evidence = {
        "target": AtomEvidence(
            "target", False, False, AtomReducer.SCALAR, ("fixture",), "oracle"
        ),
        "skipped": AtomEvidence(
            "skipped",
            None,
            None,
            AtomReducer.NOT_EVALUATED,
            ("fixture",),
            "oracle",
        ),
    }

    _validate_dependency_truth(
        atom_id="target",
        relation=relation,
        evidence=evidence,
        actual={"target": False, "skipped": None},
    )
