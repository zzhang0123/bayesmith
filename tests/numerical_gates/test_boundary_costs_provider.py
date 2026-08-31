"""Contract tests for the cost-scoreboard boundary provider (P5)."""

from __future__ import annotations

import pytest

from tests.numerical_gates import oracles
from tests.numerical_gates.boundary_cases import BOUNDARY_SUITES
from tests.numerical_gates.mutation_harness import MutationDirection, run_mutation
from tests.numerical_gates.mutation_specs import COSTS_MUTATION_SPECS

_COSTS_GATES = (
    "COSTS:gap_is_contested:contested-bandwidth",
    "COSTS:timing_noise_in_domain:proper-fraction",
    "COSTS:cg_tol_positive:strictly-positive",
)


def test_costs_mutation_specs_exactly_cover_the_two_sided_costs_gates():
    expected = {
        (gate_id, direction) for gate_id in _COSTS_GATES for direction in MutationDirection
    }
    actual = {(spec.gate_id, spec.direction) for spec in COSTS_MUTATION_SPECS}
    assert actual == expected
    assert len(COSTS_MUTATION_SPECS) == 6


@pytest.mark.parametrize(
    "spec",
    COSTS_MUTATION_SPECS,
    ids=lambda spec: f"{spec.gate_id}-{spec.direction.value}",
)
def test_each_costs_mutation_turns_its_frozen_witness_bad(spec) -> None:
    suite = BOUNDARY_SUITES[spec.gate_id]
    case_id = (
        suite.tighten_case_id
        if spec.direction is MutationDirection.TIGHTEN
        else suite.loosen_case_id
    )
    case = next(c for c in suite.cases if c.case_id == case_id)
    result = run_mutation(spec, case)
    assert result.baseline.verdict is oracles.NumericalVerdict.OK
    assert result.hit_count > 0
    assert result.same_realization
    assert result.mutant.verdict is oracles.NumericalVerdict.BAD
    assert result.killed