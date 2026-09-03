"""Freshness and schema tests for the numerical-gate registry."""

import ast
import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

import tests.numerical_gates.registry as registry_module
from tests.numerical_gates.registry import (
    GATE_REGISTRY,
    MutationMode,
    RegistryValidationError,
    SourceAnchor,
    validate_registry,
)
from tests.numerical_gates.source_manifest import (
    EXPECTED_CANDIDATE_IDS,
    EXPECTED_SOURCE_MANIFEST,
)
from tests.numerical_gates.source_scan import (
    CandidateClassification,
    CandidateFamily,
    RegistryFreshnessError,
    assert_manifest_fresh,
    locate_candidate_node,
    scan_repository,
    scan_source_text,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_expected_source_manifest_is_a_nonempty_checked_in_snapshot() -> None:
    """An empty snapshot would let the freshness check bless any source tree."""
    assert EXPECTED_CANDIDATE_IDS, (
        "the expected source manifest must be captured before the scanner can pass"
    )


def test_current_source_scan_exactly_matches_the_checked_in_manifest() -> None:
    candidates = scan_repository(REPOSITORY_ROOT)

    assert_manifest_fresh(candidates, EXPECTED_SOURCE_MANIFEST)
    assert tuple(candidate.candidate_id for candidate in candidates) == (
        EXPECTED_CANDIDATE_IDS
    )


def test_manifest_freshness_rejects_corrupted_stored_syntax() -> None:
    candidates = scan_repository(REPOSITORY_ROOT)
    corrupted = replace(EXPECTED_SOURCE_MANIFEST[0], syntax="not_the_source_predicate")

    with pytest.raises(RegistryFreshnessError, match="normalized syntax"):
        assert_manifest_fresh(
            candidates,
            (corrupted, *EXPECTED_SOURCE_MANIFEST[1:]),
        )


def test_raw_ast_baseline_is_explicit() -> None:
    candidates = scan_repository(REPOSITORY_ROOT)
    family_counts = {
        family: sum(candidate.family is family for candidate in candidates)
        for family in CandidateFamily
    }

    assert family_counts[CandidateFamily.RAISE] == 176
    assert family_counts[CandidateFamily.COMPARE] == 327


def test_every_raw_candidate_has_exactly_one_code_classification() -> None:
    candidate_ids = {
        candidate.candidate_id for candidate in scan_repository(REPOSITORY_ROOT)
    }
    classified_ids = [entry.candidate_id for entry in EXPECTED_SOURCE_MANIFEST]

    assert len(classified_ids) == len(set(classified_ids))
    assert set(classified_ids) == candidate_ids


def test_new_gate_failure_has_the_required_remediation() -> None:
    candidates = scan_source_text(
        "def select(value):\n"
        "    if value < 0.25:\n"
        "        raise ValueError('new gate')\n",
        "src/bayesmith/synthetic.py",
    )

    with pytest.raises(
        RegistryFreshnessError,
        match="add a registry entry and boundary grid",
    ):
        assert_manifest_fresh(candidates, ())


def test_stable_identity_ignores_leading_line_insertions() -> None:
    source = "def choose(value):\n    return value < 0.25\n"
    shifted = "# harmless heading\n\n" + source

    original = scan_source_text(source, "src/bayesmith/synthetic.py")
    inserted = scan_source_text(shifted, "src/bayesmith/synthetic.py")

    assert [candidate.candidate_id for candidate in original] == [
        candidate.candidate_id for candidate in inserted
    ]
    assert [candidate.lineno for candidate in original] != [
        candidate.lineno for candidate in inserted
    ]


def test_stable_identity_changes_with_the_comparison_premise() -> None:
    below = scan_source_text(
        "def choose(value):\n    return value < 0.25\n",
        "src/bayesmith/synthetic.py",
    )
    above = scan_source_text(
        "def choose(value):\n    return value <= 0.25\n",
        "src/bayesmith/synthetic.py",
    )

    assert below[0].candidate_id != above[0].candidate_id


def test_branch_identity_changes_when_a_helper_premise_changes() -> None:
    exact = scan_source_text(
        "def choose(value):\n    if exact(value):\n        return 1\n",
        "src/bayesmith/synthetic.py",
    )
    tolerant = scan_source_text(
        "def choose(value):\n    if tolerant(value):\n        return 1\n",
        "src/bayesmith/synthetic.py",
    )

    exact_decisions = [
        candidate
        for candidate in exact
        if candidate.family is CandidateFamily.DECISION_PREDICATE
    ]
    tolerant_decisions = [
        candidate
        for candidate in tolerant
        if candidate.family is CandidateFamily.DECISION_PREDICATE
    ]
    assert len(exact_decisions) == len(tolerant_decisions) == 1
    assert exact_decisions[0].candidate_id != tolerant_decisions[0].candidate_id


def test_decision_identity_detects_array_equal_becoming_allclose() -> None:
    exact = scan_source_text(
        "def matches(left, right):\n"
        "    if np.array_equal(left, right):\n"
        "        return True\n",
        "src/bayesmith/synthetic.py",
    )
    tolerant = scan_source_text(
        "def matches(left, right):\n"
        "    if np.allclose(left, right):\n"
        "        return True\n",
        "src/bayesmith/synthetic.py",
    )

    assert {item.candidate_id for item in exact} != {
        item.candidate_id for item in tolerant
    }


def test_decision_identity_detects_deleted_conjunction_atom() -> None:
    complete = scan_source_text(
        "def qualifies(a, b, c):\n    return a and b and c\n",
        "src/bayesmith/synthetic.py",
    )
    weakened = scan_source_text(
        "def qualifies(a, b, c):\n    return a and c\n",
        "src/bayesmith/synthetic.py",
    )

    assert {item.candidate_id for item in complete} != {
        item.candidate_id for item in weakened
    }


def test_boolean_binding_helper_call_is_a_decision_candidate() -> None:
    source = "def qualify(value):\n    sigma_spd = is_positive(value)\n    return sigma_spd\n"
    candidates = scan_source_text(source, "src/bayesmith/synthetic.py")
    decisions = [
        candidate
        for candidate in candidates
        if candidate.family is CandidateFamily.DECISION_PREDICATE
    ]

    assert [candidate.syntax for candidate in decisions] == ["is_positive(value)"]


def test_boolean_binding_is_tracked_by_use_not_variable_name() -> None:
    exact = scan_source_text(
        "def qualify(left, right):\n"
        "    admissible = np.array_equal(left, right)\n"
        "    if admissible:\n"
        "        return 1\n",
        "src/bayesmith/synthetic.py",
    )
    tolerant = scan_source_text(
        "def qualify(left, right):\n"
        "    admissible = np.allclose(left, right)\n"
        "    if admissible:\n"
        "        return 1\n",
        "src/bayesmith/synthetic.py",
    )

    assert {item.candidate_id for item in exact} != {
        item.candidate_id for item in tolerant
    }


def test_unknown_helper_assignment_feeding_a_branch_has_its_own_identity() -> None:
    exact = scan_source_text(
        "def qualify(value):\n"
        "    admissible = exact(value)\n"
        "    if admissible:\n"
        "        return 1\n",
        "src/bayesmith/synthetic.py",
    )
    tolerant = scan_source_text(
        "def qualify(value):\n"
        "    admissible = tolerant(value)\n"
        "    if admissible:\n"
        "        return 1\n",
        "src/bayesmith/synthetic.py",
    )

    exact_assignments = {
        item.syntax
        for item in exact
        if item.family is CandidateFamily.DECISION_PREDICATE
    }
    tolerant_assignments = {
        item.syntax
        for item in tolerant
        if item.family is CandidateFamily.DECISION_PREDICATE
    }
    assert "exact(value)" in exact_assignments
    assert "tolerant(value)" in tolerant_assignments


def test_boolean_alias_chain_propagates_to_the_original_assignment() -> None:
    candidates = scan_source_text(
        "def qualify(value):\n"
        "    initial = unknown_helper(value)\n"
        "    alias = initial\n"
        "    final = alias\n"
        "    if final:\n"
        "        return 1\n",
        "src/bayesmith/synthetic.py",
    )
    decisions = {
        item.syntax
        for item in candidates
        if item.family is CandidateFamily.DECISION_PREDICATE
    }

    assert {"unknown_helper(value)", "initial", "alias"} <= decisions


def test_nonboolean_payload_and_limit_bindings_are_not_decisions() -> None:
    candidates = scan_source_text(
        "def configure():\n"
        "    finite_max_n = 256\n"
        "    determinant_payload = object()\n"
        "    return finite_max_n\n",
        "src/bayesmith/synthetic.py",
    )

    assert not [
        candidate
        for candidate in candidates
        if candidate.family is CandidateFamily.DECISION_PREDICATE
    ]


def test_cholesky_identity_includes_the_exception_contract() -> None:
    linalg = scan_source_text(
        "def factor(matrix):\n"
        "    try:\n"
        "        np.linalg.cholesky(matrix)\n"
        "    except np.linalg.LinAlgError:\n"
        "        return False\n",
        "src/bayesmith/synthetic.py",
    )
    value_error = scan_source_text(
        "def factor(matrix):\n"
        "    try:\n"
        "        np.linalg.cholesky(matrix)\n"
        "    except ValueError:\n"
        "        return False\n",
        "src/bayesmith/synthetic.py",
    )
    linalg_ids = {
        item.candidate_id
        for item in linalg
        if item.family is CandidateFamily.LINALG_PREMISE
    }
    value_error_ids = {
        item.candidate_id
        for item in value_error
        if item.family is CandidateFamily.LINALG_PREMISE
    }

    assert linalg_ids != value_error_ids


def test_bare_validation_call_is_a_decision_candidate() -> None:
    source = "def qualify(value):\n    _validate_strict(value)\n    return value\n"
    candidates = scan_source_text(source, "src/bayesmith/synthetic.py")

    assert any(
        candidate.family is CandidateFamily.DECISION_PREDICATE
        and candidate.syntax == "_validate_strict(value)"
        for candidate in candidates
    )


def test_candidate_locator_uses_the_same_occurrence_identity() -> None:
    source = (
        "def choose(value):\n"
        "    if value < 1:\n"
        "        return 1\n"
        "    if value < 1:\n"
        "        return 2\n"
    )
    candidates = [
        candidate
        for candidate in scan_source_text(source, "src/bayesmith/synthetic.py")
        if candidate.family is CandidateFamily.COMPARE
    ]

    second = locate_candidate_node(source, candidates[1].candidate_id)
    assert isinstance(second, ast.Compare)
    assert second.lineno == 4


def test_every_scanner_family_round_trips_through_the_authoritative_locator() -> None:
    candidates = scan_repository(REPOSITORY_ROOT)
    seen = set()
    for candidate in candidates:
        if candidate.family in seen:
            continue
        source = (REPOSITORY_ROOT / candidate.module).read_text()
        node = locate_candidate_node(source, candidate.candidate_id)
        assert ast.unparse(node) == candidate.syntax
        seen.add(candidate.family)

    assert seen == set(CandidateFamily)


def test_registry_schema_and_source_anchors_are_complete() -> None:
    candidates = scan_repository(REPOSITORY_ROOT)

    validate_registry(GATE_REGISTRY, candidates, EXPECTED_SOURCE_MANIFEST)


def test_registered_atoms_are_descendants_of_their_declared_roots() -> None:
    candidates = scan_repository(REPOSITORY_ROOT)
    compound_index = next(
        index for index, entry in enumerate(GATE_REGISTRY) if entry.conjunction_atom_ids
    )
    compound = GATE_REGISTRY[compound_index]
    unrelated = next(
        item.candidate_id for item in candidates if item.module != compound.module
    )
    bad = replace(compound, conjunction_atom_ids=(unrelated,))
    entries = (
        *GATE_REGISTRY[:compound_index],
        bad,
        *GATE_REGISTRY[compound_index + 1 :],
    )

    with pytest.raises(RegistryValidationError, match="AST descendant"):
        validate_registry(entries, candidates, EXPECTED_SOURCE_MANIFEST)


def test_atom_containment_distinguishes_identical_syntax_occurrences() -> None:
    candidates = scan_repository(REPOSITORY_ROOT)
    rung6_index = next(
        index
        for index, entry in enumerate(GATE_REGISTRY)
        if entry.gate_id == "LADDER:rung6:trace"
    )
    rung6 = GATE_REGISTRY[rung6_index]
    wrong_occurrence = next(
        item.candidate_id
        for item in candidates
        if item.candidate_id.endswith("compare::6f6bf7e28f2b4e29::1")
    )
    bad = replace(rung6, conjunction_atom_ids=(wrong_occurrence,))
    entries = (*GATE_REGISTRY[:rung6_index], bad, *GATE_REGISTRY[rung6_index + 1 :])

    with pytest.raises(RegistryValidationError, match="AST descendant"):
        validate_registry(entries, candidates, EXPECTED_SOURCE_MANIFEST)


def test_entry_cannot_self_authenticate_a_noncanonical_mutation_target() -> None:
    candidates = scan_repository(REPOSITORY_ROOT)
    original = GATE_REGISTRY[0]
    unrelated_target = next(
        target
        for entry in GATE_REGISTRY[1:]
        for target in entry.mutation_target_ids
        if target not in original.mutation_target_ids
    )
    bad = replace(
        original,
        mutation_target_ids=(*original.mutation_target_ids, unrelated_target),
    )

    with pytest.raises(RegistryValidationError, match="canonical mutation targets"):
        validate_registry(
            (bad, *GATE_REGISTRY[1:]), candidates, EXPECTED_SOURCE_MANIFEST
        )


def test_entry_cannot_self_authenticate_a_noncanonical_root() -> None:
    candidates = scan_repository(REPOSITORY_ROOT)
    original = GATE_REGISTRY[0]
    replacement = GATE_REGISTRY[1]
    bad = replace(
        original,
        source_candidate_ids=replacement.source_candidate_ids,
        expected_source_syntax=replacement.expected_source_syntax,
        source_anchors=replacement.source_anchors,
        source_classifications=replacement.source_classifications,
        mutation_target_ids=replacement.mutation_target_ids,
        conjunction_atom_ids=replacement.conjunction_atom_ids,
    )

    with pytest.raises(RegistryValidationError, match="canonical source roots"):
        validate_registry(
            (bad, *GATE_REGISTRY[1:]), candidates, EXPECTED_SOURCE_MANIFEST
        )


def test_declared_anchor_cannot_be_replaced_by_an_unrelated_decision() -> None:
    candidates = scan_repository(REPOSITORY_ROOT)
    unrelated = next(item for item in candidates if item.syntax == "not copied")
    original = GATE_REGISTRY[0]
    bad = replace(
        original,
        source_candidate_ids=(unrelated.candidate_id,),
        mutation_target_ids=(unrelated.candidate_id,),
        conjunction_atom_ids=(),
        expected_source_syntax=(unrelated.syntax,),
        source_anchors=(
            SourceAnchor(unrelated.module, unrelated.qualname, unrelated.family),
        ),
        module=unrelated.module,
    )

    with pytest.raises(RegistryValidationError, match="semantically classified"):
        validate_registry(
            (bad, *GATE_REGISTRY[1:]), candidates, EXPECTED_SOURCE_MANIFEST
        )


def test_inventory_compounds_list_every_required_premise() -> None:
    registry = {entry.gate_id: entry for entry in GATE_REGISTRY}

    frozen = set(
        registry["EAGER:frozen-probes:identity-width-order"].expected_source_syntax
    )
    assert {
        "type(probes) is not FrozenProbes",
        "vectors.shape[1] != _n(lam)",
        "order < 0",
    } <= frozen

    symmetry = set(
        registry["EAGER:symmetry:tolerant-representative"].expected_source_syntax
    )
    assert any("_is_symmetric" in syntax for syntax in symmetry)
    assert any("eigenvalues > 0.0" in syntax for syntax in symmetry)

    ladder_sigma = set(
        registry["LADDER:sigma:symmetry-spd-condition"].expected_source_syntax
    )
    assert any("_is_symmetric" in syntax for syntax in ladder_sigma)
    assert any("_is_positive_definite" in syntax for syntax in ladder_sigma)
    assert any("condition_resolved" in syntax for syntax in ladder_sigma)

    rho = set(
        registry["PLAN:certificate:rho-domain-and-coverage"].expected_source_syntax
    )
    assert {
        "not 0.0 <= self.measured_max < 1.0",
        "not 0.0 <= self.certified_rho < 1.0",
        "self.certified_rho < self.measured_max",
    } <= rho

    optional = set(
        registry["PLAN:certificate:optional-scale-domain"].expected_source_syntax
    )
    assert any("max_abs_lambda_logdet" in syntax for syntax in optional)
    assert any("max_x_operator_norm" in syntax for syntax in optional)


def test_compound_atom_sets_are_complete_and_not_cross_contaminated() -> None:
    candidates = {item.candidate_id: item for item in scan_repository(REPOSITORY_ROOT)}
    registry = {entry.gate_id: entry for entry in GATE_REGISTRY}

    def atom_syntax(gate_id: str) -> set[str]:
        return {
            candidates[candidate_id].syntax
            for candidate_id in registry[gate_id].conjunction_atom_ids
        }

    assert atom_syntax("LADDER:determinant-lemma:payload") == {
        "problem.low_rank_factors is not None",
        "rank_evidence_valid",
        "sigma_formation_valid",
        "sigma_exactly_symmetric",
    }
    assert atom_syntax("LADDER:rung6:trace") == {
        "sigma_formation_valid",
        "traces_verified",
        "measured_rho_converges",
        "rho_covers_input",
        "0.0 <= rho < 1.0",
    }
    assert atom_syntax("LADDER:rung7:frozen") == {
        "sigma_formation_valid",
        "frozen_width_valid",
        "problem.trace_order is not None",
        "problem.trace_order >= 0",
        "measured_rho_converges",
        "rho_covers_input",
        "0.0 <= rho < 1.0",
    }
    condition_atoms = atom_syntax("EAGER:factor-base:condition-ceiling")
    error_atoms = atom_syntax("EAGER:factor-base:error-budget")
    assert condition_atoms == {
        "np.isfinite(base_condition)",
        "base_condition < base_condition_ceiling",
    }
    assert {
        "np.isfinite(base_log_error_bound)",
        "base_log_error_bound <= ceiling",
    } <= error_atoms


def test_non_gate_rows_have_reasons_and_no_fabricated_witnesses() -> None:
    non_gates = getattr(registry_module, "NON_GATE_REGISTRY", ())

    assert len(non_gates) == 4
    assert all(entry.reason for entry in non_gates)
    assert all(entry.tighten_witness is None for entry in non_gates)
    assert all(entry.loosen_witness is None for entry in non_gates)


def test_logically_dominated_or_reused_checks_are_static_not_fake_two_sided() -> (
    None
):
    """A check with no independent outcome delta cannot claim two mutation kills."""
    registry = {entry.gate_id: entry for entry in GATE_REGISTRY}
    static_ids = {
        "EAGER:factor-projection:error-budget",
        "EAGER:factor-projection:finite-qr-arithmetic",
        "EAGER:factor-projection:whitened-positive-spectrum",
        "EAGER:factor-base:condition-ceiling",
        "EAGER:factor-base:error-budget",
        "EAGER:factor-reduced:diagonal-certificate",
        "EAGER:factor-reduced:qr-certificate",
        "EAGER:factor-reduced:acceptance-budget",
        "EAGER:trace:actual-rho-strict",
        "PLAN:certificate:rho-domain-and-coverage",
        "PLAN:warmup:rho-inputs-and-margin",
        "PLAN:warmup:tail-fraction",
        "PLAN:warmup:rho-roundoff-ceiling",
        "PLAN:factory-certificate:strict-rho",
        "PLAN:measurement:lambda-logdet-finite",
        "PLAN:canonical-probes:runtime-finite",
    }

    assert {
        gate_id
        for gate_id, entry in registry.items()
        if entry.mutation_mode is MutationMode.STATIC_ONLY
    } == static_ids
    for gate_id in static_ids:
        entry = registry[gate_id]
        assert entry.static_reason
        assert not entry.tighten_witness
        assert not entry.loosen_witness


def test_unreachable_conjunction_atoms_are_static_not_fabricated_cases() -> None:
    """Capability-only or short-circuited atoms remain inventoried, not faked."""
    registry = {entry.gate_id: entry for entry in GATE_REGISTRY}
    balance_prefix = (
        "src/bayesmith/marginal/_logdet_eager.py::<module>."
        "_balanced_factor_columns::"
    )
    state_prefix = (
        "src/bayesmith/marginal/_logdet_eager.py::<module>."
        "state_space_logdet::"
    )
    frozen_prefix = (
        "src/bayesmith/marginal/_logdet_plan.py::<module>."
        "_frozen_probe_energy_bounds::"
    )
    expected = {
        "EAGER:factor-balance:exact-power-of-two-reversibility": {
            f"{balance_prefix}predicate_call_atom::508f419c8ad7bceb::0",
            f"{balance_prefix}predicate_call_atom::633c4ba7bc098a19::0",
            f"{balance_prefix}finite_predicate::633c4ba7bc098a19::0",
            f"{balance_prefix}predicate_call_atom::ff9f64daa7a7b618::0",
            f"{balance_prefix}predicate_call_atom::09a90b48970dbb22::0",
            f"{balance_prefix}finite_predicate::09a90b48970dbb22::0",
        },
        "EAGER:state-space:payload-domain": {
            f"{state_prefix}predicate_call_atom::5662962bf2f78843::0",
            f"{state_prefix}finite_predicate::5662962bf2f78843::0",
        },
        "PLAN:frozen:probe-energy-range": {
            f"{frozen_prefix}predicate_call_atom::ec9578c6de121206::0",
            f"{frozen_prefix}finite_predicate::ec9578c6de121206::0",
        },
    }

    assert {
        entry.gate_id: set(entry.static_atom_reasons)
        for entry in GATE_REGISTRY
        if entry.static_atom_reasons
    } == expected
    for gate_id, atom_ids in expected.items():
        entry = registry[gate_id]
        assert set(registry_module.dynamic_atom_ids(entry)) == (
            set(entry.conjunction_atom_ids) - atom_ids
        )
        assert all(entry.static_atom_reasons[atom_id].strip() for atom_id in atom_ids)


def test_static_atoms_must_be_owned_nonempty_conjunction_atoms() -> None:
    candidates = scan_repository(REPOSITORY_ROOT)
    index = next(
        index
        for index, entry in enumerate(GATE_REGISTRY)
        if entry.static_atom_reasons
    )
    original = GATE_REGISTRY[index]
    unknown = next(
        atom_id
        for entry in GATE_REGISTRY
        for atom_id in entry.conjunction_atom_ids
        if atom_id not in original.conjunction_atom_ids
    )
    for bad_reasons in (
        {unknown: "unrelated"},
        {next(iter(original.static_atom_reasons)): ""},
    ):
        bad = replace(original, static_atom_reasons=bad_reasons)
        entries = (*GATE_REGISTRY[:index], bad, *GATE_REGISTRY[index + 1 :])
        with pytest.raises(RegistryValidationError, match="static atom"):
            validate_registry(entries, candidates, EXPECTED_SOURCE_MANIFEST)


def test_resource_bound_atom_isolation_ambiguities_are_explicit_not_static() -> None:
    """Whole-gate domination must replace stale per-atom ambiguity claims."""
    assert not {
        entry.gate_id: set(entry.atom_isolation_ambiguities)
        for entry in GATE_REGISTRY
        if entry.atom_isolation_ambiguities
    }


def test_atom_isolation_ambiguities_must_be_owned_dynamic_atoms() -> None:
    candidates = scan_repository(REPOSITORY_ROOT)
    index = next(
        index
        for index, entry in enumerate(GATE_REGISTRY)
        if registry_module.dynamic_atom_ids(entry)
    )
    original = GATE_REGISTRY[index]
    dynamic_atom = registry_module.dynamic_atom_ids(original)[0]
    static_atom = next(
        atom_id
        for entry in GATE_REGISTRY
        for atom_id in entry.static_atom_reasons
    )
    for bad_reasons in (
        {static_atom: "wrong gate"},
        {dynamic_atom: ""},
    ):
        bad = replace(original, atom_isolation_ambiguities=bad_reasons)
        entries = (*GATE_REGISTRY[:index], bad, *GATE_REGISTRY[index + 1 :])
        with pytest.raises(RegistryValidationError, match="atom isolation ambiguity"):
            validate_registry(entries, candidates, EXPECTED_SOURCE_MANIFEST)


def test_finite_qr_postcheck_is_guarded_by_exception_first_arithmetic() -> None:
    """Lock the control flow that makes the postcheck capability-only."""
    path = REPOSITORY_ROOT / "src/bayesmith/marginal/_logdet_eager.py"
    tree = ast.parse(path.read_text())
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_factor_projection_certificate"
    )
    guarded = next(
        node
        for node in owner.body
        if isinstance(node, ast.Try)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "qr"
            for statement in node.body
            for call in ast.walk(statement)
        )
    )
    guarded_syntax = "\n".join(ast.unparse(statement) for statement in guarded.body)
    handlers = {ast.unparse(kind) for handler in guarded.handlers for kind in handler.type.elts}
    postcheck = next(
        node
        for node in owner.body[owner.body.index(guarded) + 1 :]
        if isinstance(node, ast.If)
    )

    assert "np.errstate(over='raise', invalid='raise')" in guarded_syntax
    assert "left_basis.T @ perturbation @ right_basis" in guarded_syntax
    assert "left_basis @ core @ right_basis.T" in guarded_syntax
    assert handlers == {"FloatingPointError", "np.linalg.LinAlgError"}
    assert ast.unparse(postcheck.test) == (
        "not (np.all(np.isfinite(left_basis)) and "
        "np.all(np.isfinite(right_basis)) and np.all(np.isfinite(core)) "
        "and np.all(np.isfinite(projected)))"
    )


def test_supported_finite_qr_outputs_do_not_supply_a_refused_postcheck_cell() -> None:
    largest = np.finfo(np.float64).max
    matrices = (
        np.array([[1.3, -0.7], [0.4, 2.1], [-1.1, 0.8]]),
        np.array([[largest, 0.0], [0.0, largest / 2.0]]),
        np.array([[1.0, 1.0], [1.0, math.nextafter(1.0, math.inf)]]),
    )

    for matrix in matrices:
        basis = np.linalg.qr(matrix, mode="reduced")[0]
        assert np.all(np.isfinite(basis))


def test_static_eigenvalue_atoms_need_a_solver_capability_fault() -> None:
    smallest_subnormal = math.nextafter(0.0, math.inf)
    matrices = (
        np.diag([smallest_subnormal, 2.0 * smallest_subnormal]),
        np.diag([1e-300, 1e300]),
        np.diag([-1.0, 1.0]),
        np.diag([0.0, 1.0]),
    )

    for matrix in matrices:
        smallest = float(np.min(np.linalg.eigvalsh(matrix)))
        assert smallest <= 0.0 or np.isfinite(smallest)


@pytest.mark.parametrize(
    "diagonal",
    (
        np.array([0.0]),
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0, -1.0]),
        np.array([1.0, 2.0]),
        np.array([-1.0, -2.0]),
    ),
)
def test_positive_reduced_qr_sign_dominates_nonzero_diagonal_atom(
    diagonal: np.ndarray,
) -> None:
    for q_sign in (-1.0, 1.0):
        reduced_sign = q_sign * float(np.prod(np.sign(diagonal)))
        assert not (reduced_sign > 0.0) or np.all(diagonal != 0.0)


@pytest.mark.parametrize(
    "eta",
    (
        0.0,
        math.nextafter(0.0, math.inf),
        0.25,
        math.nextafter(1.0, 0.0),
        1.0,
        math.inf,
        math.nan,
    ),
)
@pytest.mark.parametrize(
    "base_bound",
    (0.0, 1e-10, math.sqrt(2.0**-52), math.inf, math.nan),
)
def test_total_budget_logically_dominates_projection_and_base_budget_checks(
    eta: float, base_bound: float
) -> None:
    """No component-budget loosen can admit when the same-ceiling sum refuses."""
    ceiling = math.sqrt(2.0**-52)
    projection_bound = (
        -3.0 * math.log1p(-eta)
        if math.isfinite(eta) and 0.0 <= eta < 1.0
        else math.inf
    )
    reduced_bound = 2e-12
    total = math.fsum((projection_bound, base_bound, reduced_bound))
    total_valid = math.isfinite(total) and total <= ceiling
    projection_valid = (
        math.isfinite(eta)
        and eta < 1.0
        and math.isfinite(projection_bound)
        and projection_bound <= ceiling
    )
    base_budget_valid = math.isfinite(base_bound) and base_bound <= ceiling

    assert not total_valid or projection_valid
    assert not total_valid or base_budget_valid


def test_plan_strict_rho_call_is_an_unconsumed_validation_statement() -> None:
    """The PLAN row reuses the callee's gate; it has no return-value boundary."""
    path = REPOSITORY_ROOT / "src/bayesmith/marginal/_logdet_plan.py"
    tree = ast.parse(path.read_text())
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validate_plan_certificate"
    )
    calls = [
        statement
        for statement in owner.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "_validate_strict_rho"
    ]

    assert len(calls) == 1


@pytest.mark.parametrize(
    ("actual_rho", "certificate"),
    (
        (0.0, 0.0),
        (0.4, 0.4),
        (math.nextafter(1.0, 0.0), math.nextafter(1.0, 0.0)),
        (1.0, math.nextafter(1.0, 0.0)),
        (math.inf, 0.5),
    ),
)
def test_certificate_domain_and_coverage_dominate_actual_rho_strictness(
    actual_rho: float, certificate: float
) -> None:
    successful_following_checks = (
        0.0 <= certificate < 1.0 and actual_rho <= certificate
    )

    assert not successful_following_checks or actual_rho < 1.0


def test_non_gate_registry_is_bidirectionally_validated() -> None:
    validator = getattr(registry_module, "validate_non_gate_registry", None)
    assert validator is not None
    candidates = scan_repository(REPOSITORY_ROOT)
    validator(
        registry_module.NON_GATE_REGISTRY,
        candidates,
        EXPECTED_SOURCE_MANIFEST,
    )
    linked_policy = {
        candidate_id
        for entry in registry_module.NON_GATE_REGISTRY
        for candidate_id in entry.source_candidate_ids
        if entry.classification is CandidateClassification.POLICY_STATIC_REFUSAL
    }
    manifest_policy = {
        entry.candidate_id
        for entry in EXPECTED_SOURCE_MANIFEST
        if entry.classification is CandidateClassification.POLICY_STATIC_REFUSAL
    }
    assert linked_policy == manifest_policy


def test_gate_metadata_is_specific_not_module_template_text() -> None:
    assert all(
        "explicit source-domain boundary named by this gate" not in entry.threshold
        for entry in GATE_REGISTRY
    )
    assert all("named direct" not in entry.admitted_outcome for entry in GATE_REGISTRY)
    assert all(
        "named ladder rung" not in entry.admitted_outcome for entry in GATE_REGISTRY
    )
    assert len({entry.axes for entry in GATE_REGISTRY}) >= 12
    assert set(registry_module.GATE_METADATA) == {
        entry.gate_id for entry in GATE_REGISTRY
    }


def test_representative_gate_metadata_states_exact_live_semantics() -> None:
    registry = {entry.gate_id: entry for entry in GATE_REGISTRY}

    finite_rho = registry["LADDER:finite:payload-rho"]
    assert finite_rho.threshold == (
        "finite payload spectral radius rho <= 1.0 (closed endpoint; no certificate)"
    )
    assert finite_rho.dependencies == ("EAGER:finite:newton-stability-rho",)
    assert finite_rho.provenance is registry_module.ThresholdProvenance.BORROWED
    assert finite_rho.admitted_outcome == (
        "Keep the generic finite-polynomial payload eligible for L5"
    )
    assert finite_rho.refused_outcome == (
        "Exclude only the generic payload; L5 may still execute through "
        "determinant_lemma_payload"
    )
    assert finite_rho.oracle == (
        "Measure eigvals of the generic Lambda^-1 P payload independently; compare "
        "generic L5, determinant-lemma L5, and dense slogdet"
    )

    strict_rho = registry["EAGER:trace:actual-rho-strict"]
    assert strict_rho.threshold == "measured spectral radius rho < 1.0 (strict)"

    assert registry["MAP:map_estimate:stationarity-floor"].threshold == (
        "sqrt(eps(gradient dtype)) * mode.size * ||H||_2"
    )
    assert registry["MAP:map_estimate:relative-positive-curvature"].threshold == (
        "eps(H dtype) * abs(lambda_max) * max(mode.size, 1)"
    )
    dense = registry["LADDER:rung4:dense"]
    assert dense.quantity == (
        "n <= config.dense_max_n, condition_resolved, and sigma_spd."
    )
    assert dense.admitted_outcome == "L4 dispatches dense_cholesky_logdet"
    assert dense.refused_outcome == "falls through to L5 finite e-polynomial"
    assert dense.oracle == "NumPy slogdet with independent eigvalsh/condition checks"
    assert registry["LADDER:rung3:structured"].expected_source_syntax == ("structured",)
    assert registry["EAGER:factor-base:condition-ceiling"].threshold == (
        "base condition < 1/sqrt(eps(work dtype)) (strict)"
    )
    assert registry["PLAN:runtime:sigma-finite-and-positive"].threshold == (
        "compact Sigma entries > 0; dense slogdet sign > 0 after finite TwoSum formation"
    )


def test_finite_payload_rho_metadata_preserves_determinant_lemma_alternative() -> None:
    entry = {item.gate_id: item for item in GATE_REGISTRY}["LADDER:finite:payload-rho"]
    assert entry.quantity == (
        "The generic L5 payload symmetric_sigma-Lambda needs a resolved spectral "
        "radius rho <= 1; determinant_lemma_payload is an independent L5 alternative."
    )
    assert entry.provenance is registry_module.ThresholdProvenance.BORROWED
    assert entry.admitted_outcome == (
        "Keep the generic finite-polynomial payload eligible for L5"
    )
    assert entry.refused_outcome == (
        "Exclude only the generic payload; L5 may still execute through "
        "determinant_lemma_payload"
    )
    assert entry.oracle == (
        "Measure eigvals of the generic Lambda^-1 P payload independently; compare "
        "generic L5, determinant-lemma L5, and dense slogdet"
    )
    assert tuple(axis.name for axis in entry.axes) == (
        "lambda_scale",
        "perturbation_scale",
        "matrix_geometry",
        "computation_dtype",
        "determinant_alternative",
    )
    by_name = {axis.name: axis for axis in entry.axes}
    assert "resolved-rho" in by_name["lambda_scale"].low
    assert "resolved-rho" in by_name["perturbation_scale"].low
    assert "nonnormal" in by_name["matrix_geometry"].endpoints[1]
    assert "rescued only" in by_name["determinant_alternative"].endpoints[1]


def test_configured_rung_limits_own_api_contract_provenance() -> None:
    registry = {item.gate_id: item for item in GATE_REGISTRY}
    rung1 = registry["LADDER:rung1:low-rank-size"]
    assert rung1.quantity == (
        "Rank evidence valid AND (compact diagonal OR executable determinant lemma) "
        "AND Sigma SPD AND rank <= config.low_rank_max AND "
        "rank <= config.low_rank_fraction * n."
    )
    assert rung1.threshold == (
        "Inclusive configured rank and fraction limits; rank evidence, payload "
        "availability, and Sigma SPD are borrowed facts"
    )
    assert rung1.provenance is registry_module.ThresholdProvenance.API_CONTRACT

    rung4 = registry["LADDER:rung4:dense"]
    assert rung4.quantity == (
        "n <= config.dense_max_n, condition_resolved, and sigma_spd."
    )
    assert rung4.threshold == (
        "Inclusive configured n limit; condition resolution and Sigma SPD are "
        "borrowed facts"
    )
    assert rung4.provenance is registry_module.ThresholdProvenance.API_CONTRACT


def test_warmup_rho_metadata_preserves_rounded_binary64_sequence() -> None:
    entry = {item.gate_id: item for item in GATE_REGISTRY}[
        "PLAN:warmup:rho-roundoff-ceiling"
    ]
    assert entry.quantity == (
        "Binary64 evaluates raw_bound = float(measured_max + float(margin)); "
        "arithmetic_envelope = abs(raw_bound) * gamma_m; raw_certified = "
        "float(raw_bound + arithmetic_envelope); certified = "
        "nextafter(raw_certified, +inf); require certified < 1."
    )
    assert "(measured+margin)*(1+gamma" not in entry.quantity
    assert entry.admitted_outcome == (
        "Build RhoCertificate and choose its order from the outward certified rho"
    )
    assert entry.refused_outcome == (
        "Raise the warmup no-headroom ValueError when certified >= 1"
    )
    assert entry.oracle == (
        "Replay each binary64 addition and multiplication with float rounding, "
        "then apply math.nextafter; use Decimal only to check the enclosure"
    )
    assert tuple(axis.name for axis in entry.axes) == (
        "rho_value",
        "margin",
        "multiplicity",
    )
    rendered_axes = " | ".join(
        value
        for axis in entry.axes
        for value in (axis.low, *axis.endpoints, axis.high, axis.extreme)
    )
    assert "outward" in rendered_axes
    assert "one" in rendered_axes


def test_structure_gates_bind_exact_helper_internals() -> None:
    registry = {entry.gate_id: entry for entry in GATE_REGISTRY}
    symmetric = set(
        registry["EAGER:symmetry:tolerant-representative"].expected_source_syntax
    )
    assert "difference <= tolerance" in symmetric

    structured = set(
        registry["EAGER:structured:exact-shape-and-spectrum"].expected_source_syntax
    )
    assert {
        "np.array_equal(matrix, np.diag(np.diag(matrix)))",
        "np.array_equal(matrix, expected)",
        "diagonal == diagonal[0]",
        "eigenvalues <= 0.0",
        "reconstructed.shape != dense.shape or not np.array_equal(reconstructed, dense)",
    } <= structured


def test_semantic_links_name_the_actual_helper_and_compound_premises() -> None:
    registry = {entry.gate_id: entry for entry in GATE_REGISTRY}

    assert registry["EAGER:LogDetProblem:lambda-spd"].expected_source_syntax == (
        "not _is_positive_definite(lam)",
    )
    assert registry["EAGER:LogDetProblem:lambda-spd"].dependencies == (
        "EAGER:symmetry:tolerant-representative",
    )
    assert set(
        registry["EAGER:factor-reconstruction:layout-exactness"].expected_source_syntax
    ) == {
        "np.array_equal(canonical, value)",
        "np.array_equal(reconstructed, value)",
    }
    assert registry["PLAN:factory-certificate:strict-rho"].expected_source_syntax == (
        "_validate_strict_rho(problem.lambda_matrix, problem.perturbation, certificate.certified_rho)",
    )
    assert all(
        len(registry[gate_id].source_candidate_ids) == 1
        for gate_id in (
            "COUPLING:block_coupling:f-xx-spd",
            "COUPLING:block_coupling:f-tt-spd",
        )
    )
    assert (
        len(registry["MAP:map_estimate:finite-derivative-payload"].source_candidate_ids)
        == 4
    )


def test_each_within_block_cholesky_is_an_independent_reviewed_gate() -> None:
    """The second Cholesky cannot hide behind a witness for the first block."""
    registry = {entry.gate_id: entry for entry in GATE_REGISTRY}
    gates = {
        gate_id: registry[gate_id]
        for gate_id in (
            "COUPLING:block_coupling:f-xx-spd",
            "COUPLING:block_coupling:f-tt-spd",
        )
    }

    assert {
        entry.expected_source_syntax[0] for entry in gates.values()
    } == {
        "np.linalg.cholesky(f_xx)",
        "np.linalg.cholesky(f_tt)",
    }
    assert all(len(entry.source_candidate_ids) == 1 for entry in gates.values())
    assert all(entry.mutation_mode is MutationMode.TWO_SIDED for entry in gates.values())


def test_plan_coupling_map_graph_semantic_spot_checks() -> None:
    registry = {entry.gate_id: entry for entry in GATE_REGISTRY}

    rho = registry["PLAN:certificate:rho-domain-and-coverage"]
    assert set(rho.expected_source_syntax) == {
        "not 0.0 <= self.measured_max < 1.0",
        "not 0.0 <= self.certified_rho < 1.0",
        "self.certified_rho < self.measured_max",
    }
    optional = registry["PLAN:certificate:optional-scale-domain"]
    assert {"max_abs_lambda_logdet", "max_x_operator_norm"} <= {
        field
        for syntax in optional.expected_source_syntax
        for field in ("max_abs_lambda_logdet", "max_x_operator_norm")
        if field in syntax
    }

    coupling_xx = registry["COUPLING:block_coupling:f-xx-spd"]
    coupling_tt = registry["COUPLING:block_coupling:f-tt-spd"]
    assert len(coupling_xx.source_candidate_ids) == 1
    assert len(coupling_tt.source_candidate_ids) == 1
    assert not coupling_xx.conjunction_atom_ids
    assert not coupling_tt.conjunction_atom_ids
    assert any(
        "linalg_exception_premise" in item for item in coupling_xx.mutation_target_ids
    )

    map_finite = registry["MAP:map_estimate:finite-derivative-payload"]
    assert len(map_finite.conjunction_atom_ids) >= 3
    graph = registry["GRAPH:_names:duplicate-multiplicity"]
    assert graph.quantity == (
        "multiplicity names.count(name) for each requested node name; "
        "every name must occur exactly once (count <= 1)."
    )


def test_known_structural_candidates_are_not_labeled_numerical_gates() -> None:
    by_syntax = {}
    for entry in EXPECTED_SOURCE_MANIFEST:
        by_syntax.setdefault(entry.syntax, set()).add(entry.classification)

    for syntax in (
        "not copied",
        "outside",
        "wrong_remove",
        "downstream",
        "node.is_latent",
    ):
        assert CandidateClassification.NUMERICAL_GATE not in by_syntax[syntax]


def test_numerical_gate_classifications_are_semantically_linked() -> None:
    linked = {
        candidate_id
        for entry in GATE_REGISTRY
        for candidate_id in entry.mutation_target_ids
    }
    numerical = {
        entry.candidate_id
        for entry in EXPECTED_SOURCE_MANIFEST
        if entry.classification is CandidateClassification.NUMERICAL_GATE
    }

    assert numerical <= linked


def test_registry_entries_are_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        GATE_REGISTRY[0].gate_id = "changed"  # type: ignore[misc]


def test_pd_registry_binds_vector_and_dense_runtime_sign_branches() -> None:
    registry = {entry.gate_id: entry for entry in GATE_REGISTRY}
    pd_syntax = set(
        registry["EAGER:symmetry:tolerant-representative"].expected_source_syntax
    )
    runtime_syntax = set(
        registry["PLAN:runtime:sigma-finite-and-positive"].expected_source_syntax
    )

    assert "bool(np.all(value > 0.0))" in pd_syntax
    assert "value > 0.0" in pd_syntax
    assert "sign <= 0.0" in runtime_syntax


def test_boolean_annotated_unknown_helper_return_is_a_decision() -> None:
    candidates = scan_source_text(
        "def qualify(x) -> bool:\n    result = exact(x)\n    return result\n",
        "src/bayesmith/synthetic.py",
    )
    assert "exact(x)" in {
        item.syntax
        for item in candidates
        if item.family is CandidateFamily.DECISION_PREDICATE
    }


def test_boolean_annotated_aliased_unknown_helper_return_is_a_decision() -> None:
    candidates = scan_source_text(
        "def qualify(x) -> bool:\n"
        "    result = exact(x)\n"
        "    alias = result\n"
        "    return alias\n",
        "src/bayesmith/synthetic.py",
    )
    decisions = {
        item.syntax
        for item in candidates
        if item.family is CandidateFamily.DECISION_PREDICATE
    }
    assert {"exact(x)", "result"} <= decisions


@pytest.mark.parametrize("annotation", ["bool", "builtins.bool", "'bool'"])
def test_direct_annotated_boolean_return_tracks_unknown_helper(annotation: str) -> None:
    exact = scan_source_text(
        f"def qualify(x) -> {annotation}:\n    return exact(x)\n",
        "src/bayesmith/synthetic.py",
    )
    tolerant = scan_source_text(
        f"def qualify(x) -> {annotation}:\n    return tolerant(x)\n",
        "src/bayesmith/synthetic.py",
    )
    exact_decisions = {
        item.syntax
        for item in exact
        if item.family is CandidateFamily.DECISION_PREDICATE
    }
    tolerant_decisions = {
        item.syntax
        for item in tolerant
        if item.family is CandidateFamily.DECISION_PREDICATE
    }
    assert "exact(x)" in exact_decisions
    assert "tolerant(x)" in tolerant_decisions


def test_annotated_local_alias_return_tracks_unknown_helper() -> None:
    candidates = scan_source_text(
        "def qualify(x) -> bool:\n"
        "    result: bool = exact(x)\n"
        "    alias: bool = result\n"
        "    return alias\n",
        "src/bayesmith/synthetic.py",
    )
    decisions = {
        item.syntax
        for item in candidates
        if item.family is CandidateFamily.DECISION_PREDICATE
    }
    assert {"exact(x)", "result"} <= decisions


@pytest.mark.parametrize("annotation", ["bool", "builtins.bool", "'bool'"])
def test_annotated_local_seeds_boolean_flow_without_return_annotation(
    annotation: str,
) -> None:
    candidates = scan_source_text(
        "def qualify(x):\n"
        f"    verdict: {annotation} = exact(x)\n"
        "    alias = verdict\n"
        "    return alias\n",
        "src/bayesmith/synthetic.py",
    )
    decisions = {
        item.syntax
        for item in candidates
        if item.family is CandidateFamily.DECISION_PREDICATE
    }
    assert {"exact(x)", "verdict", "alias"} <= decisions


@pytest.mark.parametrize("annotation", ["bool", "builtins.bool", "'bool'"])
def test_declaration_only_boolean_annotation_seeds_later_assignment_flow(
    annotation: str,
) -> None:
    candidates = scan_source_text(
        "def qualify(x):\n"
        f"    verdict: {annotation}\n"
        "    verdict = exact(x)\n"
        "    alias = verdict\n"
        "    return alias\n",
        "src/bayesmith/synthetic.py",
    )
    decisions = {
        item.syntax
        for item in candidates
        if item.family is CandidateFamily.DECISION_PREDICATE
    }
    assert {"exact(x)", "verdict", "alias"} <= decisions


def test_ladder_structure_calls_are_canonical_numerical_targets() -> None:
    registry = {entry.gate_id: entry for entry in GATE_REGISTRY}
    manifest = {entry.candidate_id: entry for entry in EXPECTED_SOURCE_MANIFEST}

    diagonal = registry["LADDER:structure:diagonal-tolerance"]
    explicit_diagonal_ids = [
        candidate_id
        for candidate_id in diagonal.mutation_target_ids
        if "predicate_call_atom::3b501ceeac5159d6::1" in candidate_id
    ]
    assert len(explicit_diagonal_ids) == 1
    explicit_diagonal = explicit_diagonal_ids[0]
    assert manifest[explicit_diagonal].syntax == (
        "_is_diagonal(sigma, rtol=config.structure_rtol, atol=config.structure_atol)"
    )
    assert manifest[explicit_diagonal].classification is (
        CandidateClassification.NUMERICAL_GATE
    )

    kronecker = registry["LADDER:structure:kronecker-evidence"]
    target_syntax = {
        manifest[candidate_id].syntax for candidate_id in kronecker.mutation_target_ids
    }
    assert {
        "all((_is_positive_definite(factor) for factor in problem.structure.factors))",
        "_is_positive_definite(factor)",
        "not factors_spd",
    } <= target_syntax
    assert all(
        manifest[candidate_id].classification
        in {
            CandidateClassification.NUMERICAL_GATE,
            CandidateClassification.NUMERICAL_SAFETY,
        }
        for candidate_id in kronecker.mutation_target_ids
    )


def test_ladder_sigma_fact_group_borrows_both_eager_certificates() -> None:
    entry = {item.gate_id: item for item in GATE_REGISTRY}[
        "LADDER:sigma:symmetry-spd-condition"
    ]
    assert entry.dependencies == (
        "EAGER:symmetry:tolerant-representative",
        "EAGER:dense-condition:strict-dtype-ceiling",
    )
    assert entry.provenance is registry_module.ThresholdProvenance.BORROWED


def test_plan_margin_and_outward_postconditions_are_canonical_targets() -> None:
    registry = {entry.gate_id: entry for entry in GATE_REGISTRY}
    manifest = {entry.candidate_id: entry for entry in EXPECTED_SOURCE_MANIFEST}
    required = {
        "PLAN:warmup:lambda-scale-inputs": {
            "not np.isfinite(lambda_logdet_margin) or lambda_logdet_margin < 0.0",
            "np.isfinite(lambda_logdet_margin)",
            "lambda_logdet_margin < 0.0",
        },
        "PLAN:warmup:x-norm-inputs": {
            "not np.isfinite(x_operator_norm_margin) or x_operator_norm_margin < 0.0",
            "np.isfinite(x_operator_norm_margin)",
            "x_operator_norm_margin < 0.0",
        },
        "PLAN:runtime-range:product": {
            "not np.isfinite(result) or result > maximum",
            "np.isfinite(result)",
            "result > maximum",
        },
        "PLAN:runtime-range:sum": {
            "not np.isfinite(result) or result > maximum",
            "np.isfinite(result)",
            "result > maximum",
        },
    }
    for gate_id, expected in required.items():
        target_ids = registry[gate_id].mutation_target_ids
        target_syntax = {manifest[candidate_id].syntax for candidate_id in target_ids}
        assert expected <= target_syntax, gate_id
        assert all(
            manifest[candidate_id].classification
            in {
                CandidateClassification.NUMERICAL_GATE,
                CandidateClassification.NUMERICAL_SAFETY,
            }
            for candidate_id in target_ids
        ), gate_id


def test_ladder_structure_metadata_describes_live_exact_helpers() -> None:
    registry = {entry.gate_id: entry for entry in GATE_REGISTRY}
    exact_rows = {
        "LADDER:structure:diagonal-tolerance": (
            "Exact diagonal layout: every off-diagonal entry is bitwise zero.",
            "Exact equality to zero; rtol and atol are ignored by the live helper.",
            "off_diagonal",
        ),
        "LADDER:structure:circulant-tolerance-spectrum": (
            "Exact cyclic row shifts and a real, strictly positive FFT spectrum.",
            "Exact row-shift equality plus every real FFT eigenvalue > 0.",
            "circulant_layout",
        ),
        "LADDER:structure:toeplitz-tolerance": (
            "Exact equality along every matrix diagonal.",
            "Exact diagonal equality; rtol and atol are ignored by the live helper.",
            "toeplitz_layout",
        ),
    }
    for gate_id, (quantity, threshold, axis_name) in exact_rows.items():
        entry = registry[gate_id]
        assert entry.provenance is registry_module.ThresholdProvenance.EXACT_DOMAIN
        assert entry.quantity == quantity
        assert entry.threshold == threshold
        axis = next(axis for axis in entry.axes if axis.name == axis_name)
        rendered = " | ".join(
            (entry.quantity, entry.threshold, axis.low, *axis.endpoints, axis.extreme)
        )
        assert "tolerance" not in rendered.lower()
        assert "T±" not in rendered and "T-" not in rendered


def test_metadata_preserves_exact_formulas_and_complete_boundary_cells() -> None:
    registry = {entry.gate_id: entry for entry in GATE_REGISTRY}
    assert registry["EAGER:trace:tail-domain-and-order"].quantity == (
        "rho in [0,1), integer order >=0, multiplicity >=1, finite tolerance >0, "
        "and the smallest order satisfying "
        "multiplicity*rho**(m+1)/((m+1)*(1-rho)) <= tolerance."
    )
    assert "Tr(X**r)" in registry["EAGER:trace:exact-power-trace-evidence"].quantity
    assert (
        "sum(x_bound**p/p)"
        in registry["PLAN:runtime:frozen-prerequisites-and-series"].quantity
    )
    assert registry["PLAN:multiplicity:index-and-gamma-domain"].axes[0].extreme == (
        "bool, float, 10**1000"
    )
    rho_axes = registry["PLAN:certificate:rho-domain-and-coverage"].axes
    assert tuple(axis.name for axis in rho_axes) == (
        "measured_max",
        "certified_rho",
    )
    assert rho_axes[0].endpoints == (
        "0 and nextafter(0,-inf)",
        "nextafter(1,0) and 1",
    )
    assert rho_axes[1].endpoints == (
        "0 and nextafter(measured,0)",
        "equality/nextafter(measured,+inf)",
    )
    assert registry["EAGER:trace:certificate-upper-bound"].axes[0].endpoints == (
        "certificate equals actual rho (admit)",
        "certificate nextafter(actual, 0) refuses; nextafter(actual, +inf) admits",
    )
    assert registry["EAGER:factor-reduced:diagonal-certificate"].axes[0].endpoints == (
        "relative error nextafter(1, 0), 1, and nextafter(1, +inf)",
        "determinant sign +1, 0, and -1",
    )
    payload_axes = registry["LADDER:sigma:payload-symmetry"].axes
    assert tuple(axis.name for axis in payload_axes) == (
        "sigma_layout",
        "sigma_asymmetry",
        "structure_atol",
        "structure_rtol",
        "computation_dtype",
    )
    assert payload_axes[1].endpoints == (
        "asymmetry nextafter(atol + rtol*scale, 0)",
        "equality and nextafter(atol + rtol*scale, +inf)",
    )


def test_ladder_registry_enumerates_every_real_input_axis() -> None:
    """A bundled prose/derived/dead axis cannot replace the reviewed census."""
    expected = {
        "LADDER:sigma:payload-symmetry": (
            "sigma_layout",
            "sigma_asymmetry",
            "structure_atol",
            "structure_rtol",
            "computation_dtype",
        ),
        "LADDER:sigma:finite-two-sum": (
            "lambda_entry",
            "perturbation_entry",
            "computation_dtype",
        ),
        "LADDER:structure:compact-diagonal-positive": ("sigma_entry",),
        "LADDER:structure:diagonal-tolerance": (
            "structure_request",
            "off_diagonal",
        ),
        "LADDER:structure:circulant-tolerance-spectrum": (
            "circulant_layout",
            "spectrum_scale",
        ),
        "LADDER:structure:toeplitz-tolerance": ("toeplitz_layout",),
        "LADDER:structure:kronecker-evidence": (
            "structure_request",
            "structure_presence",
            "factor_spectrum",
            "factor_shape",
            "reconstruction_value",
        ),
        "LADDER:sigma:symmetry-spd-condition": (
            "sigma_layout",
            "sigma_symmetry",
            "smallest_eigenvalue",
            "condition_scale",
            "structure_rtol",
            "structure_atol",
            "computation_dtype",
        ),
        "LADDER:rank:evidence": (
            "factor_presence",
            "factor_reconstruction",
            "perturbation_rank",
            "lambda_scale",
            "factor_layout",
            "factor_gauge",
            "computation_dtype",
        ),
        "LADDER:rho:measurement": (
            "lambda_scale",
            "perturbation_scale",
            "matrix_geometry",
            "computation_dtype",
        ),
        "LADDER:finite:payload-rho": (
            "lambda_scale",
            "perturbation_scale",
            "matrix_geometry",
            "computation_dtype",
            "determinant_alternative",
        ),
        "LADDER:determinant-lemma:payload": (
            "factor_presence",
            "factor_reconstruction",
            "sigma_formation",
            "sigma_symmetry",
            "condition_scale",
        ),
        "LADDER:rung0:base": (
            "sigma_formation",
            "sigma_lambda_equality",
            "dense_condition",
        ),
        "LADDER:rung1:low-rank-size": (
            "rank_evidence",
            "payload_capability",
            "sigma_spd",
            "rank",
            "dimension",
            "low_rank_max",
            "low_rank_fraction",
        ),
        "LADDER:rung2:chain": (
            "chain_block_size",
            "chain_layout",
            "sigma_formation",
            "sigma_spd",
            "condition_scale",
            "computation_dtype",
        ),
        "LADDER:rung3:structured": (
            "structure_request",
            "structure_evidence",
            "sigma_formation",
            "sigma_spd",
            "condition_scale",
        ),
        "LADDER:rung4:dense": (
            "dimension",
            "dense_max_n",
            "condition_scale",
            "sigma_spd",
            "computation_dtype",
        ),
        "LADDER:rung5:finite-size": (
            "dimension",
            "finite_max_n",
            "payload_capability",
            "rank",
            "finite_max_rank",
        ),
        "LADDER:rung5:finite-executable": (
            "dimension",
            "finite_max_n",
            "payload_capability",
            "rank",
            "finite_max_rank",
            "lambda_scale",
            "perturbation_scale",
            "sigma_formation",
            "smallest_eigenvalue",
            "sigma_symmetry",
            "factor_presence",
            "factor_reconstruction",
            "dense_condition",
            "computation_dtype",
        ),
        "LADDER:rung6:trace": (
            "sigma_formation",
            "actual_rho",
            "trace_order",
            "trace_evidence",
            "certified_rho",
        ),
        "LADDER:rung7:frozen": (
            "sigma_formation",
            "trace_order",
            "probe_presence",
            "probe_width",
            "actual_rho",
            "certified_rho",
        ),
    }
    actual = {
        entry.gate_id: tuple(axis.name for axis in entry.axes)
        for entry in GATE_REGISTRY
        if entry.gate_id.startswith("LADDER:")
    }

    assert actual == expected


def test_metadata_literal_corruption_audit_covers_all_rows() -> None:
    forbidden = (
        "rho(m+1)",
        "Tr(Xr)",
        "x_boundp",
        "101000",
        "endpoints=",
    )
    for entry in GATE_REGISTRY:
        rendered = " | ".join(
            (
                entry.quantity,
                entry.threshold,
                entry.admitted_outcome,
                entry.refused_outcome,
                entry.oracle,
                *(axis.name for axis in entry.axes),
                *(axis.low for axis in entry.axes),
                *(value for axis in entry.axes for value in axis.endpoints),
                *(axis.high for axis in entry.axes),
                *(axis.extreme for axis in entry.axes),
            )
        )
        assert not any(text in rendered for text in forbidden), entry.gate_id
        assert all(
            not endpoint.rstrip().endswith(",")
            for axis in entry.axes
            for endpoint in axis.endpoints
        ), entry.gate_id


def test_registry_rejects_unlinked_manifest_numerical_relabel() -> None:
    candidates = scan_repository(REPOSITORY_ROOT)
    unlinked = next(
        item
        for item in EXPECTED_SOURCE_MANIFEST
        if item.classification is CandidateClassification.ORDINARY_VALIDATION
        and item.candidate_id
        not in {
            target for entry in GATE_REGISTRY for target in entry.mutation_target_ids
        }
    )
    relabeled = replace(unlinked, classification=CandidateClassification.NUMERICAL_GATE)
    manifest = tuple(
        relabeled if item.candidate_id == unlinked.candidate_id else item
        for item in EXPECTED_SOURCE_MANIFEST
    )
    with pytest.raises(RegistryValidationError, match="bidirectional"):
        validate_registry(GATE_REGISTRY, candidates, manifest)


def test_every_ladder_conjunction_has_all_live_atoms() -> None:
    registry = {entry.gate_id: entry for entry in GATE_REGISTRY}

    def atoms(gate_id: str) -> set[str]:
        ids = set(registry[gate_id].conjunction_atom_ids)
        return {
            item.syntax for item in EXPECTED_SOURCE_MANIFEST if item.candidate_id in ids
        }

    expected = {
        "LADDER:rung0:base": {
            "sigma_formation_valid",
            "bool(np.array_equal(sigma, lam))",
            "dense_arithmetic_resolved",
        },
        "LADDER:rung1:low-rank-size": {
            "rank_evidence_valid",
            "compact_diagonal_payload",
            "determinant_lemma_payload",
            "sigma_spd",
            "rank <= config.low_rank_max",
            "rank <= config.low_rank_fraction * n",
        },
        "LADDER:rung2:chain": {
            "chain_structure",
            "sigma_spd",
            "condition_resolved",
        },
        "LADDER:rung4:dense": {
            "n <= config.dense_max_n",
            "condition_resolved",
            "sigma_spd",
        },
        "LADDER:rung5:finite-executable": {
            "finite_size_qualified",
            "finite_payload_stable",
            "sigma_spd",
            "determinant_lemma_payload",
            "dense_arithmetic_resolved",
        },
    }
    for gate_id, wanted in expected.items():
        assert atoms(gate_id) == wanted


def test_plan_compound_groups_bind_every_live_premise() -> None:
    registry = {entry.gate_id: entry for entry in GATE_REGISTRY}
    syntax = {
        gate_id: set(registry[gate_id].expected_source_syntax)
        | {
            item.syntax
            for item in EXPECTED_SOURCE_MANIFEST
            if item.candidate_id in registry[gate_id].conjunction_atom_ids
        }
        for gate_id in (
            "PLAN:multiplicity:index-and-gamma-domain",
            "PLAN:factory-certificate:order-and-rank",
            "PLAN:frozen-factory:probe-presence-width",
            "PLAN:runtime:expected-and-ulp-finite",
            "PLAN:warmup:rho-inputs-and-margin",
        )
    }
    assert {
        "isinstance(value, (bool, np.bool_))",
        "multiplicity < 1",
        "multiplicity >= _RHO_MULTIPLICITY_LIMIT",
    } <= syntax["PLAN:multiplicity:index-and-gamma-domain"]
    assert (
        "problem.trace_order != certificate.order"
        in syntax["PLAN:factory-certificate:order-and-rank"]
    )
    assert {
        "problem.frozen_probes is None",
        "problem.frozen_probes.values.shape[1] != _n(problem.lambda_matrix)",
    } <= syntax["PLAN:frozen-factory:probe-presence-width"]
    assert {
        "not np.isfinite(expected)",
        "not np.isfinite(rounded) or not np.isfinite(ulp)",
        "ulp > certificate.tolerance",
    } <= syntax["PLAN:runtime:expected-and-ulp-finite"]


def test_structure_rows_depend_on_exact_helper_internals() -> None:
    registry = {entry.gate_id: entry for entry in GATE_REGISTRY}
    for gate_id in (
        "LADDER:structure:diagonal-tolerance",
        "LADDER:structure:circulant-tolerance-spectrum",
        "LADDER:structure:toeplitz-tolerance",
    ):
        assert (
            "EAGER:structured:exact-shape-and-spectrum"
            in registry[gate_id].dependencies
        )


def test_metadata_has_executable_gate_specific_semantics() -> None:
    forbidden = (
        "source premise evaluated by",
        "executable gate premise in",
        "literal predicate",
        "T-ulp for",
        "T+ulp for",
        "the guarded operation continues",
        "guard returns or raises its documented refusal",
        "contract oracle evaluated from the public domain formula",
    )
    for entry in GATE_REGISTRY:
        rendered = (
            f"{entry.quantity} | {entry.admitted_outcome} | "
            f"{entry.refused_outcome} | {entry.oracle}"
        )
        assert not any(phrase in rendered for phrase in forbidden), entry.gate_id
    assert len({entry.admitted_outcome for entry in GATE_REGISTRY}) == 109
    assert len({entry.refused_outcome for entry in GATE_REGISTRY}) == 109
    assert len({entry.oracle for entry in GATE_REGISTRY}) == 109
    intermediate = {entry.gate_id: entry for entry in GATE_REGISTRY}[
        "PLAN:frozen:intermediate-runtime-range"
    ]
    assert "raise" not in intermediate.quantity
    assert "raise" not in intermediate.threshold
    assert all(
        "raise" not in endpoint
        for axis in intermediate.axes
        for endpoint in axis.endpoints
    )


def test_metadata_uses_source_backed_boundary_cells_and_observable_routes() -> None:
    forbidden = (
        "nearest admissible",
        "nearest refused",
        "concrete source T is",
        "gate inputs checked by",
        "named executable branch",
        "named concrete refusal",
        "threshold T",
        "independent Python/NumPy evaluation of the literal domain formula",
    )
    for entry in GATE_REGISTRY:
        rendered = " | ".join(
            (
                entry.quantity,
                entry.threshold,
                entry.admitted_outcome,
                entry.refused_outcome,
                entry.oracle,
                *(axis.name for axis in entry.axes),
                *(axis.low for axis in entry.axes),
                *(value for axis in entry.axes for value in axis.endpoints),
                *(axis.high for axis in entry.axes),
                *(axis.extreme for axis in entry.axes),
            )
        )
        assert not any(phrase in rendered for phrase in forbidden), entry.gate_id

    registry = {entry.gate_id: entry for entry in GATE_REGISTRY}
    expected_cells = {
        "EAGER:LadderConfig:integer-threshold-domain": (
            "1",
            ("0 admits", "-1 refuses"),
            "sys.maxsize",
        ),
        "EAGER:factor-reconstruction:layout-exactness": (
            "exact non-unit rank-one product",
            ("exact bitwise-identical product", "one output entry moved by one ULP"),
            "multi-column cancellation",
        ),
        "EAGER:factor-projection:finite-qr-arithmetic": (
            "well-scaled independent columns near 1.3",
            ("largest finite QR product", "next exponent produces overflow"),
            "nearly dependent columns",
        ),
        "COUPLING:block_coupling:f-xx-spd": (
            "non-unit positive f_xx with valid f_tt",
            (
                "positive-subnormal minimum eigenvalue",
                "zero or negative minimum eigenvalue",
            ),
            "negative non-unit f_xx with valid f_tt",
        ),
        "COUPLING:block_coupling:f-tt-spd": (
            "non-unit positive f_tt with valid f_xx",
            (
                "positive-subnormal minimum eigenvalue",
                "zero or negative minimum eigenvalue",
            ),
            "negative non-unit f_tt with valid f_xx",
        ),
        "PLAN:runtime-range:product": (
            "zero or ordinary positive left operand",
            (
                "maximum/right one ULP below",
                "maximum/right and one ULP above",
            ),
            "left operand far above maximum/right",
        ),
        "MAP:map_estimate:stationarity-floor": (
            "gradient norm at half the derived floor",
            ("gradient norm equals the floor", "gradient norm one ULP above the floor"),
            "large nonstationary gradient",
        ),
        "GRAPH:_names:duplicate-multiplicity": (
            "empty list iterable, distinct from the empty tuple T-1 cell",
            ("count 1 admits", "count 2 refuses"),
            "many repetitions of several names",
        ),
    }
    for gate_id, (low, endpoints, high) in expected_cells.items():
        axis = registry[gate_id].axes[0]
        assert axis.low == low
        assert axis.endpoints == endpoints
        assert axis.high == high


def test_metadata_boundary_kinds_have_realizable_category_cells() -> None:
    registry = {entry.gate_id: entry for entry in GATE_REGISTRY}

    discrete_requirements = {
        "EAGER:LadderConfig:integer-threshold-domain": ("-1", "0", "1"),
        "GRAPH:_names:duplicate-multiplicity": ("count 1", "count 2"),
        "LADDER:rung1:low-rank-size": (
            "one integer below",
            "exactly",
            "one integer above",
        ),
        "LADDER:rung4:dense": (
            "one integer below",
            "exactly",
            "one integer above",
        ),
        "LADDER:rung5:finite-size": (
            "one integer below",
            "exactly",
            "one integer above",
        ),
        "PLAN:multiplicity:index-and-gamma-domain": ("-1", "0", "1"),
        "PLAN:certificate:order-is-derived": ("m-1", "m", "m+1"),
        "PLAN:factory-certificate:order-and-rank": ("T-1", "T", "T+1"),
        "PLAN:frozen-factory:probe-presence-width": ("n-1", "n", "n+1"),
    }
    for gate_id, required in discrete_requirements.items():
        rendered = " | ".join(
            value
            for axis in registry[gate_id].axes
            for value in (axis.low, *axis.endpoints, axis.high, axis.extreme)
        )
        assert all(token in rendered for token in required), gate_id
        assert "ULP" not in rendered or any(
            token in rendered for token in ("exact", "float", "rho", "condition")
        ), gate_id

    exact_evidence = (
        "EAGER:factor-reconstruction:layout-exactness",
        "EAGER:structured:exact-shape-and-spectrum",
        "EAGER:trace:exact-power-trace-evidence",
        "LADDER:structure:kronecker-evidence",
        "LADDER:rung0:base",
        "PLAN:audit:retained-trace-evidence",
        "PLAN:trace-factory:exact-evidence",
    )
    for gate_id in exact_evidence:
        entry = registry[gate_id]
        rendered = " | ".join(
            (
                entry.quantity,
                *(value for axis in entry.axes for value in axis.endpoints),
            )
        ).lower()
        assert "exact" in rendered, gate_id
        assert any(token in rendered for token in ("ulp", "subnormal", "mismatch")), (
            gate_id
        )

    spd_gates = (
        "EAGER:LogDetProblem:lambda-spd",
        "EAGER:factor-projection:whitened-positive-spectrum",
        "EAGER:symmetry:tolerant-representative",
        "COUPLING:_condition_number:positive-spectrum",
        "COUPLING:block_coupling:f-xx-spd",
        "COUPLING:block_coupling:f-tt-spd",
        "LADDER:structure:compact-diagonal-positive",
        "LADDER:sigma:symmetry-spd-condition",
        "LADDER:rung2:chain",
        "LADDER:rung4:dense",
        "PLAN:runtime:sigma-finite-and-positive",
    )
    for gate_id in spd_gates:
        rendered = " | ".join(
            value.lower()
            for axis in registry[gate_id].axes
            for value in (axis.low, *axis.endpoints, axis.high, axis.extreme)
        )
        assert "positive" in rendered, gate_id
        assert "zero" in rendered or "0" in rendered, gate_id
        assert "negative" in rendered or "-delta" in rendered, gate_id

    finite_gates = (
        "EAGER:array-normalization:shape-and-finiteness",
        "EAGER:factor-projection:finite-qr-arithmetic",
        "EAGER:spectral-radius:finite-measurement",
        "COUPLING:_classify_correlation:value-finite",
        "COUPLING:_classify_correlation:floor-finite",
        "COUPLING:_condition_number:finite-spectrum",
        "MAP:map_estimate:finite-derivative-payload",
        "LADDER:sigma:finite-two-sum",
        "LADDER:rho:measurement",
        "PLAN:certificate:optional-scale-domain",
        "PLAN:measurement:x-norm-finite",
        "PLAN:measurement:lambda-logdet-finite",
        "PLAN:canonical-probes:runtime-finite",
        "PLAN:runtime:expected-and-ulp-finite",
    )
    for gate_id in finite_gates:
        rendered = " | ".join(
            value.lower()
            for axis in registry[gate_id].axes
            for value in (axis.low, *axis.endpoints, axis.high, axis.extreme)
        )
        assert "finite" in rendered or "max" in rendered, gate_id
        assert "inf" in rendered or "nan" in rendered, gate_id


def test_compound_metadata_enumerates_every_semantic_premise() -> None:
    registry = {entry.gate_id: entry for entry in GATE_REGISTRY}
    required_tokens = {
        "EAGER:state-space:payload-domain": (
            "symmetric",
            "SPD",
            "condition",
            "Schur",
            "pivot",
            "finite",
        ),
        "EAGER:structured:exact-shape-and-spectrum": (
            "diagonal",
            "circulant",
            "Toeplitz",
            "FFT spectrum",
            "Kronecker",
        ),
        "EAGER:trace:tail-domain-and-order": (
            "rho",
            "order",
            "multiplicity",
            "tolerance",
            "smallest",
        ),
        "LADDER:sigma:symmetry-spd-condition": (
            "compact",
            "symmetric",
            "SPD",
            "condition",
        ),
        "LADDER:determinant-lemma:payload": (
            "Factors",
            "rank evidence",
            "formation finite",
            "bitwise symmetric",
            "condition",
        ),
        "LADDER:rung1:low-rank-size": (
            "Rank evidence",
            "compact diagonal",
            "determinant lemma",
            "Sigma SPD",
            "low_rank_max",
            "low_rank_fraction",
        ),
        "LADDER:rung5:finite-executable": (
            "Finite-size",
            "payload rho",
            "determinant payload",
            "Sigma SPD",
            "dense arithmetic",
        ),
        "LADDER:rung6:trace": (
            "Finite Sigma",
            "rho measurement",
            "traces/order",
            "exact trace",
            "actual rho <1",
            "certificate",
            "0<=rho<1",
        ),
        "LADDER:rung7:frozen": (
            "Finite Sigma",
            "FrozenProbes",
            "width n",
            "order",
            "actual rho <1",
            "certificate",
            "[0,1)",
        ),
        "PLAN:audit:retained-trace-evidence": (
            "nonempty",
            "order",
            "rank bound",
            "traces present",
            "exact arithmetic",
        ),
        "PLAN:runtime:expected-and-ulp-finite": (
            "Expected logdet finite",
            "rounded",
            "ulp",
            "tolerance",
        ),
        "PLAN:runtime:total-error-budget": (
            "analytic_tail",
            "gamma_operation_count",
            "base_scale",
            "series_scale",
            "tolerance",
        ),
    }
    for gate_id, tokens in required_tokens.items():
        entry = registry[gate_id]
        rendered = f"{entry.quantity} | {entry.threshold}"
        assert all(token in rendered for token in tokens), gate_id


def test_registry_validator_rejects_non_independent_oracle() -> None:
    candidates = scan_repository(REPOSITORY_ROOT)
    bad_entry = replace(GATE_REGISTRY[0], oracle="bayesmith sibling implementation")

    with pytest.raises(RegistryValidationError, match="independent oracle"):
        validate_registry(
            (bad_entry, *GATE_REGISTRY[1:]), candidates, EXPECTED_SOURCE_MANIFEST
        )


def test_registry_validator_requires_both_future_witness_names() -> None:
    candidates = scan_repository(REPOSITORY_ROOT)
    bad_entry = replace(GATE_REGISTRY[0], loosen_witness="")

    with pytest.raises(RegistryValidationError, match="loosen witness"):
        validate_registry(
            (bad_entry, *GATE_REGISTRY[1:]), candidates, EXPECTED_SOURCE_MANIFEST
        )
