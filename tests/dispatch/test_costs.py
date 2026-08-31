"""Read-only cost scoreboard: the three cost expressions and their rules.

These tests pin the hard constraints the plan lists, plus the one guarantee
that keeps the scoreboard a pure addition: an abstained or declared plan
prints byte-identically to what it printed before ``costs.py`` existed.
"""

from __future__ import annotations

import inspect
import math

import jax
import pytest

from bayesmith import compile as compile_graph
from bayesmith.dispatch.costs import (
    K_CG_TOL,
    CostRow,
    LadderInputs,
    build_ladder,
    collapse_cost,
    k_cg,
    scoreboard,
    split_cost,
    tau,
    timing_reference,
)


def _inputs(**overrides):
    base = {
        "rho": 0.5,
        "kappa_cond": 100.0,
        "kappa_marg": 10.0,
        "kappa_joint": 500.0,
        "kappa_x": 100.0,
        "a": 1.0,
        "timing": timing_reference(),
        "m": 1,
    }
    base.update(overrides)
    return LadderInputs(**base)


def test_k_cg_rounds_up_and_never_fractional():
    assert k_cg(100.0, 1e-3) == 39
    assert isinstance(k_cg(4.0, 1e-2), int)


def test_tau_is_one_at_zero_coupling_and_blows_up_near_one():
    assert tau(0.0) == 1.0
    assert tau(0.99) == pytest.approx((1 + 0.99**2) / (1 - 0.99**2))
    assert tau(0.99) > 50.0


def test_the_cg_term_is_spelled_not_cancelled():
    """Constraint 1: m copies of the CG term in the split, never cancelled.

    The collapse and joint rows carry the CG term ONCE; the split carries it
    ``m`` times inside the brackets.  The difference between ``m=1`` and
    ``m=3`` is exactly ``2 * tau(c) * k_cg * c_A`` -- nothing is asserted
    to be negligible.
    """
    kappa_cond, kappa_x, c, a, c_gtheta, c_A = 100.0, 100.0, 0.5, 2.0, 5.2e-6, 6.8e-6
    one = split_cost(kappa_cond, kappa_x, c, a, c_gtheta, c_A, m=1)
    three = split_cost(kappa_cond, kappa_x, c, a, c_gtheta, c_A, m=3)
    assert three - one == pytest.approx(2 * tau(c) * k_cg(kappa_x, K_CG_TOL) * c_A)
    # collapse carries the CG term once, joint not at all
    assert collapse_cost(10.0, kappa_x, a, 43.2e-6, c_A) == pytest.approx(
        a * math.sqrt(10.0) * 43.2e-6 + k_cg(kappa_x, K_CG_TOL) * c_A
    )


def test_kappa_marg_is_an_input_not_a_function_of_c():
    """Constraint 2: no closed-form crossover in ``c``.

    ``collapse_cost`` takes ``kappa_marg`` and never ``c``, so the code cannot
    substitute ``kappa(F_tt)/(1-c^2)`` (which is measured to be wrong by a
    factor of 2525 at c=0.99).  Only ``split_cost`` reads ``c``.
    """
    split_params = set(inspect.signature(split_cost).parameters)
    collapse_params = set(inspect.signature(collapse_cost).parameters)
    assert "c" in split_params
    assert "c" not in collapse_params
    assert "kappa_marg" in collapse_params


def test_rows_are_cost_only_and_carry_a_chain_vs_whole_graph_kind():
    """Constraint 3 + the ESS trap: the argmin compares cost, never ESS.

    ``CostRow`` has no ESS field, so the scoreboard cannot mix Kish ESS and
    chain ESS.  The two row classes (chain vs whole-graph) are tagged, so a
    future ESS-based comparator can keep them in separate comparisons.
    """
    assert not hasattr(CostRow, "ess")
    record = build_ladder(_inputs(), strategy="cost")
    assert record is not None
    assert record.winner in {"split", "collapse", "joint"}
    assert "a is common to all HMC rows" in record.line()


def test_winner_is_the_minimum_cost_hi():
    rows = (
        CostRow("split", "chain", 1.0, 3.0, 1),
        CostRow("collapse", "whole-graph", 1.0, 2.0),
        CostRow("joint", "whole-graph", 1.0, 4.0),
    )
    verdict = scoreboard(rows)
    assert verdict.winner is not None and verdict.winner.strategy == "collapse"
    assert not verdict.abstained


def test_an_infinite_row_cannot_win_but_does_not_hide_the_finite_ones():
    rows = (
        CostRow("split", "chain", math.inf, math.inf, 1),
        CostRow("collapse", "whole-graph", 1.0, 2.0),
    )
    verdict = scoreboard(rows)
    assert verdict.winner is not None and verdict.winner.strategy == "collapse"


def test_all_infinite_rows_abstain():
    rows = (
        CostRow("split", "chain", math.inf, math.inf, 1),
        CostRow("collapse", "whole-graph", math.inf, math.inf),
        CostRow("joint", "whole-graph", math.inf, math.inf),
    )
    verdict = scoreboard(rows)
    assert verdict.abstained and verdict.winner is None


def test_an_overlapping_row_is_contested():
    rows = (
        CostRow("split", "chain", 1.0, 2.0, 1),
        CostRow("collapse", "whole-graph", 1.5, 1.9),
    )
    verdict = scoreboard(rows)
    assert verdict.winner is not None and verdict.winner.strategy == "collapse"
    assert {row.strategy for row in verdict.contested} == {"split"}


def test_declared_strategy_has_no_ladder():
    from tests.exact.models import radiometer

    plan = compile_graph(radiometer())
    assert plan.ladder is None


def test_cost_strategy_appends_the_scoreboard_on_a_real_mixed_graph():
    """The three cost expressions, on a real fixture, printed by str(plan)."""
    from tests.exact.models import indirect_ancestor

    with jax.enable_x64(True):
        plan = compile_graph(indirect_ancestor(), strategy="cost")
    assert plan.ladder is not None
    text = str(plan)
    assert "cost scoreboard" in text
    assert "split " in text and "collapse " in text and "joint " in text
    assert "winner " in text
    # the routing is untouched by strategy="cost"
    assert plan.exact is not None and plan.sampled is not None


def test_abstention_prints_byte_identically_to_the_declared_plan():
    """The plan's escape hatch: all-infinite costs abstain, str is unchanged.

    ``a=inf`` drives every row infinite, so the scoreboard abstains,
    ``ladder.line()`` is empty, and the plan prints exactly what the
    declared strategy prints -- the scoreboard added and removed nothing.
    """
    from tests.exact.models import indirect_ancestor

    with jax.enable_x64(True):
        graph = indirect_ancestor()
        declared = str(compile_graph(graph))
        cost = str(compile_graph(graph, strategy="cost", a=math.inf))
    assert declared == cost
