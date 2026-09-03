"""Unified executable boundary registry assembled from provider modules."""

from __future__ import annotations

from tests.numerical_gates.boundary_checks import CHECKS_SUITES
from tests.numerical_gates.boundary_collapse import COLLAPSE_SUITES
from tests.numerical_gates.boundary_core import (
    BoundaryCase,
    BoundarySuite,
    CaseReference,
)
from tests.numerical_gates.boundary_costs import COSTS_SUITES
from tests.numerical_gates.boundary_diagnose_graph import (
    DIAGNOSE_GRAPH_SUITES,
)
from tests.numerical_gates.boundary_eager import EAGER_SUITES
from tests.numerical_gates.boundary_ladder import LADDER_SUITES
from tests.numerical_gates.boundary_pilot import PILOT_SUITES
from tests.numerical_gates.boundary_plan import PLAN_SUITES
from tests.numerical_gates.boundary_sbc import SBC_SUITES
from tests.numerical_gates.registry import GATE_REGISTRY, MutationMode

_ALL_SUITES = (
    *EAGER_SUITES,
    *LADDER_SUITES,
    *PLAN_SUITES,
    *DIAGNOSE_GRAPH_SUITES,
    *COSTS_SUITES,
    *COLLAPSE_SUITES,
    *PILOT_SUITES,
    *CHECKS_SUITES,
    *SBC_SUITES,
)

BOUNDARY_SUITES: dict[str, BoundarySuite] = {
    suite.gate_id: suite for suite in _ALL_SUITES
}

BOUNDARY_CASES: dict[str, BoundaryCase] = {
    case.case_id: case for suite in _ALL_SUITES for case in suite.cases
}

# One fast-layer smoke cell per two-sided gate: the tighten witness (the nearest
# executable gate face).  It is cheap and deterministic -- it never uses the
# huge VERY_HIGH/EXTREME matrices -- and it is the exact cell the mutation
# harness mutates for the tighten direction, so the fast layer still exercises
# each gate's live production predicate once.  test_boundary_layering.py holds
# the meta-test that enforces this set covers every registered gate.
FAST_BOUNDARY_CASE_IDS: tuple[str, ...] = tuple(
    sorted(suite.tighten_case_id for suite in _ALL_SUITES)
)

_TWO_SIDED_ENTRIES = tuple(
    entry for entry in GATE_REGISTRY if entry.mutation_mode is MutationMode.TWO_SIDED
)

ATOM_CASES: dict[str, CaseReference] = {
    atom_id: CaseReference(
        name=f"atom::{atom_id}",
        gate_id=suite.gate_id,
        atom_id=atom_id,
        case_id=case_id,
        cases=BOUNDARY_CASES,
    )
    for suite in _ALL_SUITES
    for atom_id, case_id in suite.atom_case_ids.items()
}

WITNESS_CASES: dict[str, CaseReference] = {
    witness_name: CaseReference(
        name=witness_name,
        gate_id=entry.gate_id,
        atom_id=None,
        case_id=case_id,
        cases=BOUNDARY_CASES,
    )
    for entry in _TWO_SIDED_ENTRIES
    for witness_name, case_id in (
        (
            entry.tighten_witness,
            BOUNDARY_SUITES[entry.gate_id].tighten_case_id,
        ),
        (
            entry.loosen_witness,
            BOUNDARY_SUITES[entry.gate_id].loosen_case_id,
        ),
    )
}


__all__ = [
    "ATOM_CASES",
    "BOUNDARY_CASES",
    "BOUNDARY_SUITES",
    "FAST_BOUNDARY_CASE_IDS",
    "WITNESS_CASES",
]
