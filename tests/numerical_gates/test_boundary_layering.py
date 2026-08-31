"""Fast-layer smoke cells and the layering invariant that keeps them honest.

The heavy numerical-gate grids (test_boundary_cases.py and the four
test_boundary_*_provider.py files) and the mutation harness are marked
"full" and run only in the nightly layer.  This module carries the fast
layer's share of the boundary contract: exactly one cheap, deterministic cell
per registered gate, plus a meta-test that fails if any gate is left without a
fast-layer cell.
"""

from __future__ import annotations

import pytest

from tests.numerical_gates import oracles
from tests.numerical_gates.boundary_cases import (
    BOUNDARY_CASES,
    BOUNDARY_SUITES,
    FAST_BOUNDARY_CASE_IDS,
)
from tests.numerical_gates.registry import GATE_REGISTRY, MutationMode

_TWO_SIDED = {
    entry.gate_id: entry
    for entry in GATE_REGISTRY
    if entry.mutation_mode is MutationMode.TWO_SIDED
}
_STATIC = {
    entry.gate_id: entry
    for entry in GATE_REGISTRY
    if entry.mutation_mode is MutationMode.STATIC_ONLY
}


def test_fast_layer_selects_exactly_one_cell_per_two_sided_gate() -> None:
    """The fast selection is one tighten witness per gate, with no orphans."""
    fast = tuple(FAST_BOUNDARY_CASE_IDS)
    assert len(fast) == len(set(fast)) == len(_TWO_SIDED)
    assert set(fast) <= set(BOUNDARY_CASES)
    assert {BOUNDARY_CASES[case_id].gate_id for case_id in fast} == set(_TWO_SIDED)


def test_every_registered_gate_has_a_fast_layer_boundary_cell() -> None:
    """The meta-test: every registered gate keeps at least one executable cell
    in the fast layer, so -m "not full" can never silently become "no
    numerical-gate coverage".
    """
    assert set(_TWO_SIDED) | set(_STATIC) == {
        entry.gate_id for entry in GATE_REGISTRY
    }
    fast = frozenset(FAST_BOUNDARY_CASE_IDS)
    for entry in GATE_REGISTRY:
        if entry.mutation_mode is MutationMode.STATIC_ONLY:
            # Static-only gates have no executable boundary grid by design; their
            # only coverage is the registry/metadata audit, which is never marked
            # full and so always runs in the fast layer.
            continue
        suite = BOUNDARY_SUITES[entry.gate_id]
        assert any(case.case_id in fast for case in suite.cases), entry.gate_id


@pytest.mark.parametrize("case_id", FAST_BOUNDARY_CASE_IDS)
def test_fast_boundary_cell_executes_and_is_admitted(case_id: str) -> None:
    """One live production cell per gate must run clean in the fast layer."""
    case = BOUNDARY_CASES[case_id]
    execution = case()
    assert execution.case_id == case_id
    assert execution.gate_id == case.gate_id
    assert execution.observed_gate_side is execution.expected_gate_side
    assert execution.verdict is oracles.NumericalVerdict.OK
    assert execution.realized_point.quantity
    assert execution.realized_point.axes
    assert execution.direct_calls
    assert execution.oracle_checks or execution.evaluations
