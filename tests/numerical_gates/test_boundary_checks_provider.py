"""Contract tests for the predictive-check boundary provider (R3 Task 3)."""

from __future__ import annotations

import pytest

from tests.numerical_gates import oracles
from tests.numerical_gates.boundary_cases import BOUNDARY_SUITES
from tests.numerical_gates.mutation_harness import MutationDirection, run_mutation
from tests.numerical_gates.mutation_specs import CHECKS_MUTATION_SPECS

_CHECKS_GATES = (
    "CHECKS:tail_mass_within_rate:declared-false-positive-rate",
    "CHECKS:draws_resolve_the_band:p-value-draw-floor",
)


def test_checks_mutation_specs_exactly_cover_the_two_sided_checks_gates():
    expected = {
        (gate_id, direction)
        for gate_id in _CHECKS_GATES
        for direction in MutationDirection
    }
    actual = {(spec.gate_id, spec.direction) for spec in CHECKS_MUTATION_SPECS}
    assert actual == expected
    assert len(CHECKS_MUTATION_SPECS) == 4


@pytest.mark.parametrize(
    "spec",
    CHECKS_MUTATION_SPECS,
    ids=lambda spec: f"{spec.gate_id}-{spec.direction.value}",
)
def test_each_checks_mutation_turns_its_frozen_witness_bad(spec) -> None:
    """The kill each threshold exists to be able to fail.

    TIGHTEN moves the boundary one step onto the admitted side, so the cell
    AT the threshold -- a p-value with exactly the declared tail mass, or a
    result with exactly the floor's draws -- is refused where production
    admits it.  LOOSEN moves it the other way, so the cell one step below is
    admitted where production refuses it.  A gate that could no longer fail
    would survive both, which is the state this file exists to make visible.
    """
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
