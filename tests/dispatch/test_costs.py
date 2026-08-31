"""Read-only cost scoreboard: the three cost expressions and their rules.

These tests pin the hard constraints the plan lists, plus the one guarantee
that keeps the scoreboard a pure addition: an abstained or declared plan
prints byte-identically to what it printed before ``costs.py`` existed.
"""

from __future__ import annotations

import inspect
import math

import jax
import numpy as np
import pytest

from bayesmith import compile as compile_graph
from bayesmith.dispatch.costs import (
    DOMINANCE_SHARE,
    K_CG_TOL,
    TIMING_NOISE_TOLERANCE,
    CostRow,
    LadderInputs,
    build_ladder,
    collapse_cost,
    cost_shares,
    dominant_input,
    k_cg,
    reconcile,
    recorded_inputs,
    scoreboard,
    share_is_dominant,
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

# --- P7: the reconciliation ledger -----------------------------------------


def _record(**overrides):
    """A built scoreboard record, which is what the ledger reconciles against."""
    record = build_ladder(_inputs(**overrides), strategy="cost")
    assert record is not None
    return record


def test_the_record_carries_every_input_it_was_built_from():
    """The ledger's whole claim: the inputs were written down.

    It does not depend on the three cost expressions being right. It depends
    on ``rho``, the four condition numbers, ``a``, the timing constants and ``m``
    surviving to the point where a measurement can be set beside them.
    """
    inputs = _inputs()
    record = _record()

    assert recorded_inputs(record) == inputs


def test_the_three_shares_partition_the_prediction():
    inputs = _inputs(rho=0.9)
    for strategy in ("split", "collapse", "joint"):
        shares = cost_shares(inputs, strategy)
        assert [name for name, _ in shares] == ["c", "a", "cg"]
        assert sum(share for _, share in shares) == pytest.approx(1.0)


def test_the_whole_graph_rows_pay_no_sweep_and_the_joint_row_is_all_gradient():
    """Attribution follows the expressions, and cannot be read off the winner.

    ``collapse`` carries no ``tau(c)`` at all, so its amplification share is
    exactly zero however strong the coupling; ``joint`` has neither an
    amplification nor a CG term, so it is all gradient and its miss can only
    ever be located at ``a`` (or at the conditioning that multiplies it).
    """
    inputs = _inputs(rho=0.999)

    assert dict(cost_shares(inputs, "collapse"))["c"] == 0.0
    assert dict(cost_shares(inputs, "joint")) == {"c": 0.0, "a": 1.0, "cg": 0.0}
    assert dominant_input(cost_shares(inputs, "joint")) == "a"


def test_strong_coupling_puts_the_split_prediction_on_c():
    """tau(0.999) is 1000, so the amplification is 99.9% of the split row."""
    shares = dict(cost_shares(_inputs(rho=0.999), "split"))

    assert shares["c"] > 0.99
    assert dominant_input(cost_shares(_inputs(rho=0.999), "split")) == "c"


def test_no_term_above_one_half_names_no_input():
    """A tie is reported as a tie. One half is not above one half.

    At or below the dominance share two terms can be equal largest, so naming
    one of them would be an invented attribution -- the one thing the ledger
    exists not to do.
    """
    assert dominant_input((("c", 0.5), ("a", 0.3), ("cg", 0.2))) == ""
    assert dominant_input((("c", 0.4), ("a", 0.4), ("cg", 0.2))) == ""
    assert dominant_input((("c", 0.0), ("a", 0.0), ("cg", 0.0))) == ""
    assert not share_is_dominant(DOMINANCE_SHARE)
    assert share_is_dominant(float(np.nextafter(DOMINANCE_SHARE, math.inf)))


def test_an_infinite_row_has_no_attribution_to_give():
    """A ``+inf`` prediction returns zero shares, never a nan.

    A nan share loses every comparison in silence, so the dominance test
    would skip it and the ledger would report "no input dominates" for a row
    that has no prediction at all.
    """
    shares = cost_shares(_inputs(a=math.inf), "joint")

    assert dict(shares) == {"c": 0.0, "a": 0.0, "cg": 0.0}
    assert dominant_input(shares) == ""


def test_the_predicted_interval_is_exactly_as_wide_as_the_declared_noise():
    """Every row is linear in the timing constants, so width is (1+t)/(1-t).

    Identical on all three rows, which is what makes a wider interval
    evidence that an undeclared uncertainty got in rather than a property of
    the row.
    """
    record = _record()
    expected = (1.0 + TIMING_NOISE_TOLERANCE) / (1.0 - TIMING_NOISE_TOLERANCE)
    for strategy in ("split", "collapse", "joint"):
        ledger = reconcile(record, strategy, seconds=1.0, ess=100.0)
        assert ledger.width == pytest.approx(expected)
        assert ledger.predicted_hi > ledger.predicted_lo


def test_the_ledger_records_the_inputs_even_when_the_prediction_missed():
    """A miss is the interesting row, and it must still carry its inputs."""
    record = _record(rho=0.9)
    ledger = reconcile(record, "split", seconds=1.0, ess=1.0)

    assert not ledger.within
    assert ledger.measured == pytest.approx(1.0)
    assert ledger.measured > ledger.predicted_hi
    assert ledger.rho == 0.9
    assert ledger.a == 1.0
    assert (ledger.kappa_cond, ledger.kappa_marg, ledger.kappa_joint) == (
        100.0,
        10.0,
        500.0,
    )
    assert ledger.kappa_x == 100.0 and ledger.m == 1
    assert ledger.fingerprint == record.fingerprint
    assert "predicted" in ledger.line() and "measured" in ledger.line()


def test_a_run_that_produced_no_effective_samples_is_priced_at_infinity():
    """Not a raise, and not a quietly finite number either."""
    ledger = reconcile(_record(), "split", seconds=1.0, ess=0.0)

    assert ledger.measured == math.inf
    assert not ledger.within


def test_reconciling_against_a_row_that_does_not_exist_refuses():
    with pytest.raises(KeyError):
        reconcile(_record(), "gibbs", seconds=1.0, ess=100.0)
    with pytest.raises(ValueError, match="no cost row called"):
        cost_shares(_inputs(), "gibbs")


def test_an_abstained_scoreboard_still_records_what_it_could_not_price():
    """All rows ``+inf``: the ledger is written anyway, and says so.

    An abstained scoreboard is not the same as no scoreboard. It measured the
    inputs and found them unpriceable, and that -- with the measurement beside
    it -- is exactly the row a calibration set wants: the prediction is
    ``+inf``, the run is not, and no term is named because there is no
    partition to name one from. Dropping it would lose the inputs, which are
    the only thing the ledger actually claims to have.
    """
    record = _record(a=math.inf)
    assert record.abstained and record.line() == ""

    ledger = reconcile(record, "split", seconds=1.0, ess=100.0)

    assert ledger.predicted_lo == math.inf and ledger.predicted_hi == math.inf
    assert ledger.measured == pytest.approx(0.01)
    assert not ledger.within
    assert ledger.dominant == ""
    assert ledger.a == math.inf
    assert ledger.rho == 0.5


def test_a_declared_plan_carries_no_ledger_and_a_cost_plan_does():
    """The published path is untouched; the ledger is opt-in with the scoreboard.

    Measured on this fixture: rho reads 0.0, so the split row's amplification
    share is exactly zero and 99.3% of its prediction is the inner CG solve.
    The run costs 12x the predicted interval -- and the ledger says WHERE to
    look, which is the whole point: not at the coupling, and not at ``a``.
    """
    from tests.exact.models import indirect_ancestor

    with jax.enable_x64(True):
        graph = indirect_ancestor()
        declared = compile_graph(graph)
        plain = declared.sample(jax.random.key(0), num_samples=80, num_warmup=80)
        assert plain.cost is None

        priced = compile_graph(graph, strategy="cost")
        posterior = priced.sample(jax.random.key(0), num_samples=80, num_warmup=80)

    assert posterior.cost is not None
    ledger = posterior.cost
    assert ledger.strategy == "split"
    assert ledger.ess == posterior.ess
    assert ledger.seconds > 0.0
    assert ledger.measured == pytest.approx(ledger.seconds / ledger.ess)
    assert dict(ledger.shares)["c"] == 0.0
    assert ledger.dominant == "cg"
    assert ledger.fingerprint == priced.ladder.fingerprint

