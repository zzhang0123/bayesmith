"""Read-only cost scoreboard for the split / collapse / joint tradeoff (P5).

This module is the independently useful half of Wave 2: it tells a user what
the canonical coupling ``rho`` and the three strategy condition numbers ARE,
and how much a collapsed sampler would save -- without changing any routing.
``compile(strategy="cost")`` builds it and prints it; the declared plan's
blocks and methods are untouched.

Three hard constraints, each one measured before it was written down:

1. **No elimination is asserted anywhere.** The CG term appears ``tau*m``
   times in the split numerator and once in the collapse/joint term, and it
   is never cancelled.  ``split_cost``, ``collapse_cost`` and ``joint_cost``
   each spell their CG term out independently.
2. **There is no closed-form crossover in ``c``.** ``kappa_marg`` is not a
   function of ``c`` -- measured: ``F_tt = diag(1,100)`` and ``diag(100,1)``
   give true ``kappa(Schur)`` of 5025 and 1.99 at the same ``c=0.99``, while
   ``kappa(F_tt)/(1-c^2)`` says 5025 both times, a 2525x miss.  The crossover
   is numeric-only; nothing here solves for ``c``.
3. **``c_gtheta``, ``c_gc``, ``c_A`` are measured, never modelled.**  The
   grad / (A + A^T) ratio moves in opposite directions with ``n`` on the two
   reference fixtures, so a formula in ``n`` or ``k`` would be a lie.  They
   are inputs (:class:`TimingConstants`), not functions of the graph size.

The ESS trap, also written down because rules rot: **never put Kish ESS and
chain ESS in the same argmin, and never compare a whole-graph row against a
chain row.**  ``plated_radiometer(n=25, kappa=0.4)`` at N=1200: SNIS is 1.40
posterior-sd from the truth, NUTS 18.5, yet NUTS's chain ESS (33) exceeds
the Kish ESS (14) -- an ESS-priced objective picks the answer 13x further
from the truth.  The scoreboard therefore compares COST, never ESS, and each
row carries a :attr:`CostRow.kind` so the two ESS currencies can be kept in
separate comparisons when the collapse arm lands.

Abstention: any ``+inf`` input makes its row ineligible to win; if every row
is ``+inf`` the scoreboard abstains, ``compile()`` falls back to today's
behaviour, and ``str(plan)`` is byte-identical to the declared plan.

**The reconciliation ledger (P7) is the part of this module that stays useful
even if every expression above is wrong.**  :class:`CostReconciliation` records
what was PREDICTED (a row's cost interval), what was MEASURED (seconds per
effective sample), and which of the cost expression's own terms dominated the
prediction -- so a miss can be located at ``c`` (the coupling: a funnel reads
0.0 to a Laplace measurement and the amplification term is then priced off a
number that describes nothing), at ``a`` (the leapfrog constant: a calibration
that did not transfer), or at an interval so wide that satisfying it proved
nothing.  None of that depends on the three cost expressions being right; it
depends only on their inputs having been written down.  Every predicted/measured
pair a run produces is one row of calibration data for a cost model that does
not have any yet.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

import equinox as eqx

# --- D93: the contested bandwidth ------------------------------------------
CONTESTED_BANDWIDTH: float = 0.25
"""Relative cost gap below which two strategy rows are "contested" rather
than a clear winner.

The cost inputs (condition numbers from a power-iteration bound, timing
constants from wall-clock probes) are uncertain enough that a gap under a
quarter of the winning cost is not evidence the winner would still win on a
re-measure.  A clear verdict needs headroom; a contested verdict names the
rows that overlap and defers to the declared default when one of them is it.
"""


# --- D94: the timing noise tolerance ---------------------------------------
TIMING_NOISE_TOLERANCE: float = 0.10
"""Relative wall-clock noise below which a cost gap is "within timing noise".

The timing constants are measured with ``time.perf_counter`` around a JAX
operation whose run-to-run spread is a fair fraction of the mean; a cost gap
smaller than this fraction of the row cost is indistinguishable from the
probe's own noise and is reported as contested rather than a decision.
"""


# --- D95: the CG tolerance inside ``k_cg`` ---------------------------------
K_CG_TOL: float = 1e-3
"""The CG tolerance ``tol_x`` in the cost model's ``k_cg(kappa_x, tol_x)``.

Not the solve's own ``tol`` (that is ``CONVERGENCE_TARGET / kappa``); this is
the tolerance the COST model prices one inner CG solve at, so the three rows
compare the same number of iterations for the same conditioning.
"""


def k_cg(kappa: float, tol: float) -> int:
    """``ceil(0.5 * sqrt(kappa) * log(2/tol))`` -- CG iterations for one solve.

The ``1/2 * sqrt(kappa)`` factor is the textbook CG convergence rate; the
``log(2/tol)`` is the number of halvings needed to reach ``tol``.  Rounded
UP because a solve cannot take a fraction of an iteration.
    """
    if not cg_tol_positive(tol):
        raise ValueError(f"the CG tolerance must be strictly positive, got {tol!r}")
    return math.ceil(0.5 * math.sqrt(kappa) * math.log(2.0 / tol))


def tau(c: float) -> float:
    """``(1 + c^2) / (1 - c^2)`` -- the Gibbs sweep amplification of coupling.

The number of outer sweeps the split route pays per independent sample,
growing without bound as the canonical correlation ``c`` approaches one.
    """
    return (1.0 + c * c) / (1.0 - c * c)


def split_cost(
    kappa_cond: float,
    kappa_x: float,
    c: float,
    a: float,
    c_gtheta: float,
    c_A: float,
    * ,
    tol_x: float = K_CG_TOL,
    m: int = 1,
) -> float:
    """``tau(c) * [ a*sqrt(kappa_cond)*c_gtheta + m*k_cg(kappa_x, tol_x)*c_A ]``.

``m`` is 1 for the plain GCR split and 3 for the GCR + Metropolis-Hastings
sweep, which re-solves for every proposed draw and its proposal.  The CG
term is spelled out inside the brackets and is NOT cancelled against the
collapse/joint CG terms -- see the module docstring.
    """
    cg = k_cg(kappa_x, tol_x) * c_A
    return tau(c) * (a * math.sqrt(kappa_cond) * c_gtheta + m * cg)


def collapse_cost(
    kappa_marg: float,
    kappa_x: float,
    a: float,
    c_gc: float,
    c_A: float,
    * ,
    tol_x: float = K_CG_TOL,
) -> float:
    """``a*sqrt(kappa_marg)*c_gc + k_cg(kappa_x, tol_x)*c_A``.

The collapsed route solves the marginal once per gradient; its CG term is
one ``k_cg`` rather than the split's ``tau*m`` copies.
    """
    return a * math.sqrt(kappa_marg) * c_gc + k_cg(kappa_x, tol_x) * c_A


def joint_cost(kappa_joint: float, a: float, c_gall: float) -> float:
    """``a*sqrt(kappa_joint)*c_g_all`` -- NUTS over the full joint graph.

No split and no marginal, so no CG term; the whole cost is one leapfrog
gradient against the joint conditioning.
    """
    return a * math.sqrt(kappa_joint) * c_gall


@dataclass(frozen=True, slots=True)
class TimingConstants:
    """Measured per-gradient wall-clock reference magnitudes, in seconds.

Measured with ``time.perf_counter`` around the actual operation, never
derived from ``n`` or ``k`` -- the grad / (A + A^T) ratio moves in opposite
directions with ``n``, so a formula would be wrong in one direction or the
other (see the module docstring).  ``c_g_all`` is the full-joint gradient.
    """

    c_gtheta: float
    c_gc: float
    c_A: float
    c_g_all: float


def timing_reference() -> TimingConstants:
    """The plan's measured float64 reference, ``mu = theta . B x``, n=100 k=8.

c_gc 43.2us, c_gtheta 5.2us, c_A 6.8us (ratio 8.3).  ``c_g_all`` is the
full-joint gradient, which spans both blocks; the reference pins it to the
collapse gradient plus the theta gradient -- the two gradients a full-joint
step must form -- and is the only entry here that is a composition rather
than a direct single-operation measurement.
    """
    c_gc = 43.2e-6
    c_gtheta = 5.2e-6
    c_A = 6.8e-6
    return TimingConstants(c_gtheta=c_gtheta, c_gc=c_gc, c_A=c_A, c_g_all=c_gc + c_gtheta)


def relative_gap(low: float, high: float) -> float:
    """``|high - low| / max(high, low)``, so the gap is a fraction of the row.

Used by both contested tests; spelled once so the two cannot drift.
    """
    if not (math.isfinite(low) and math.isfinite(high)):
        return math.inf
    return abs(high - low) / max(high, low)


def gap_is_contested(gap: float) -> bool:
    """D93: ``gap < CONTESTED_BANDWIDTH`` -- the argmin contested verdict.

    A relative cost gap below the contested bandwidth is not evidence that
    the winner would still win on a re-measure, so the scoreboard marks the
    row contested instead of a decision.
    """
    return gap < CONTESTED_BANDWIDTH


def timing_noise_in_domain(tol: float) -> bool:
    """D94: ``tol < 1.0`` -- a timing noise tolerance is a proper fraction.

    The spread on the measured timing constants must keep ``1.0 - tol``
    positive, or a row ``cost_lo`` would be negative and its cost interval
    meaningless.
    """
    return tol < 1.0


def cg_tol_positive(tol: float) -> bool:
    """D95: ``tol > 0.0`` -- the CG tolerance ``k_cg`` prices a solve at.

    ``log(2 / tol)`` is undefined for ``tol <= 0``, so a non-positive
    tolerance cannot name a finite CG iteration count.
    """
    return tol > 0.0


# --- D103: the ledger's dominance share ------------------------------------
DOMINANCE_SHARE: float = 0.5
"""Share of a row's predicted cost one term must carry to be named dominant.

**One half is the only value that needs no tie-break, and that is why it is
the value.**  The three terms of a row partition its predicted cost, so their
shares sum to one; at most one part of such a partition can exceed a half.
Above it, "which input dominates" has exactly one answer and the ledger can
name it without ranking near-equal terms against each other.  At or below it,
two terms can be tied for largest -- and a ledger that named one of them
anyway would be inventing an attribution, which is the one thing this record
exists not to do.  It abstains instead, with an empty
:attr:`CostReconciliation.dominant`, and the shares are still reported so a
reader can see the tie for themselves.
"""


def share_is_dominant(share: float) -> bool:
    """D103: ``share > DOMINANCE_SHARE`` -- this term owns the prediction.

    Open at the boundary: a term at exactly one half leaves the rest of the
    partition summing to one half too, so it is not uniquely the largest.
    """
    return share > DOMINANCE_SHARE


@dataclass(frozen=True, slots=True)
class CostRow:
    """One strategy row: its cost interval, and which ESS currency it owns.

Attributes:
    strategy: ``"split"``, ``"collapse"`` or ``"joint"``.
    kind: ``"chain"`` for the split (its samples come from a Gibbs chain) or
        ``"whole-graph"`` for collapse/joint (a reduced or full graph NUTS,
        whose diagnostic is chain ESS too).  Carried so that a future
        ESS-based comparator keeps Kish ESS and chain ESS in separate
        comparisons -- see the module docstring.
    cost_lo, cost_hi: the cost interval from input uncertainty (timing noise,
        conditioning bound).  ``cost_hi`` is what the argmin reads.
    m: the split's multiplier, ``None`` for rows that have none.
    """

    strategy: str
    kind: Literal["chain", "whole-graph"]
    cost_lo: float
    cost_hi: float
    m: int | None = None


@dataclass(frozen=True, slots=True)
class Scoreboard:
    """The argmin verdict over :class:`CostRow` rows, cost-only, never ESS.

    ``winner`` is the row with the smallest ``cost_hi``; ``contested`` lists
    every other row whose interval overlaps the winner's or whose gap is
    within :data:`CONTESTED_BANDWIDTH`; ``abstained`` is set when every
    row's ``cost_hi`` is non-finite (``+inf``), in which case there is no
    verdict to print and the caller falls back to the declared plan.
    """

    winner: CostRow | None
    contested: tuple[CostRow, ...]
    abstained: bool


def scoreboard(rows: tuple[CostRow, ...]) -> Scoreboard:
    """Pick the winner by minimum ``cost_hi``; mark overlapping rows contested.

A ``+inf`` ``cost_hi`` makes a row ineligible to win; if every row is
``+inf`` the scoreboard abstains.  A row is contested when its ``cost_lo``
is below the winner's ``cost_hi`` (the intervals overlap) or its gap to the
winner is within :data:`CONTESTED_BANDWIDTH` of the row cost.

Nothing here reads an ESS.  The winner is a cost winner; whether to trust it
as a sampling decision is a separate question the collapse arm answers.
    """
    if not rows:
        return Scoreboard(None, (), True)
    finite = [row for row in rows if math.isfinite(row.cost_hi)]
    if not finite:
        return Scoreboard(None, (), True)
    winner = min(finite, key=lambda row: row.cost_hi)
    contested: list[CostRow] = []
    for row in rows:
        if row is winner or row.cost_hi == math.inf:
            continue
        overlap = row.cost_lo < winner.cost_hi
        within_band = gap_is_contested(relative_gap(row.cost_hi, winner.cost_hi))
        if overlap or within_band:
            contested.append(row)
    return Scoreboard(winner, tuple(contested), False)


@dataclass(frozen=True, slots=True)
class LadderInputs:
    """The measured inputs one scoreboard is built from (all scalars)."""

    rho: float
    kappa_cond: float
    kappa_marg: float
    kappa_joint: float
    kappa_x: float
    a: float
    timing: TimingConstants
    m: int


def _rows(inputs: LadderInputs) -> tuple[CostRow, ...]:
    """The three strategy rows, with ``cost_lo``/``cost_hi`` from timing noise.

The timing constants are the one input with a measured spread; each row's
interval is the cost at the timing constants scaled down (``lo``) and up
    (``hi``) by :data:`TIMING_NOISE_TOLERANCE`.  The conditioning numbers are
    point measurements from the coupling report.
    """
    t = inputs.timing
    if not timing_noise_in_domain(TIMING_NOISE_TOLERANCE):
        raise ValueError(
            "the timing noise tolerance must keep the cost interval positive; "
            f"got {TIMING_NOISE_TOLERANCE!r}"
        )
    lo = 1.0 - TIMING_NOISE_TOLERANCE
    hi = 1.0 + TIMING_NOISE_TOLERANCE
    split_lo = split_cost(inputs.kappa_cond, inputs.kappa_x, inputs.rho, inputs.a, t.c_gtheta * lo, t.c_A * lo, m=inputs.m)
    split_hi = split_cost(inputs.kappa_cond, inputs.kappa_x, inputs.rho, inputs.a, t.c_gtheta * hi, t.c_A * hi, m=inputs.m)
    collapse_lo = collapse_cost(inputs.kappa_marg, inputs.kappa_x, inputs.a, t.c_gc * lo, t.c_A * lo)
    collapse_hi = collapse_cost(inputs.kappa_marg, inputs.kappa_x, inputs.a, t.c_gc * hi, t.c_A * hi)
    joint_lo = joint_cost(inputs.kappa_joint, inputs.a, t.c_g_all * lo)
    joint_hi = joint_cost(inputs.kappa_joint, inputs.a, t.c_g_all * hi)
    return (
        CostRow("split", "chain", split_lo, split_hi, inputs.m),
        CostRow("collapse", "whole-graph", collapse_lo, collapse_hi),
        CostRow("joint", "whole-graph", joint_lo, joint_hi),
    )


def _fingerprint(inputs: LadderInputs, rows: tuple[CostRow, ...]) -> str:
    """A stable scalar-only fingerprint so the record can say "these inputs"."""
    digest = hashlib.sha256()
    payload = (
        inputs.rho,
        inputs.kappa_cond,
        inputs.kappa_marg,
        inputs.kappa_joint,
        inputs.kappa_x,
        inputs.a,
        inputs.timing.c_gtheta,
        inputs.timing.c_gc,
        inputs.timing.c_A,
        inputs.timing.c_g_all,
        inputs.m,
        tuple((row.strategy, row.kind, row.cost_lo, row.cost_hi) for row in rows),
    )
    digest.update(repr(payload).encode())
    return digest.hexdigest()


class LadderRecord(eqx.Module):
    """The scalar-only, all-static scoreboard a plan can carry and print.

Every field is ``eqx.field(static=True)`` and holds a scalar, string or tuple
of them -- never a numpy array -- because an array-valued static field makes
a SECOND trace compare array pytree metadata and raise (it passes the first,
    so it is the kind of bug that only shows up in production).
    """

    strategy: str = eqx.field(static=True)
    winner: str = eqx.field(static=True)
    rho: float = eqx.field(static=True)
    kappa_cond: float = eqx.field(static=True)
    kappa_marg: float = eqx.field(static=True)
    kappa_joint: float = eqx.field(static=True)
    cost_split: float = eqx.field(static=True)
    cost_collapse: float = eqx.field(static=True)
    cost_joint: float = eqx.field(static=True)
    contested: tuple[str, ...] = eqx.field(static=True)
    abstained: bool = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)
    # The remaining inputs, carried so that a run can reconcile against them.
    # The scoreboard's printed line does not use them; the ledger does, and
    # "the inputs were written down" is the ledger's whole claim -- see
    # CostReconciliation. All scalars, for the same reason as the fields
    # above: an array in a static field survives the first trace and raises on
    # the second.
    kappa_x: float = eqx.field(static=True)
    a: float = eqx.field(static=True)
    c_gtheta: float = eqx.field(static=True)
    c_gc: float = eqx.field(static=True)
    c_A: float = eqx.field(static=True)
    c_g_all: float = eqx.field(static=True)
    m: int = eqx.field(static=True)

    def line(self) -> str:
        """The scoreboard's one printed line, or ``""`` when it abstained.

Empty on abstention on purpose: :meth:`InferencePlan.__str__` appends it
        under the same guard as the streaming line, so an abstained plan prints
        byte-identically to a declared one.
        """
        if self.abstained:
            return ""
        head = (
            f"cost scoreboard: rho={self.rho:.8g}, kappa_cond={self.kappa_cond:.8g}, "
            f"kappa_marg={self.kappa_marg:.8g}, kappa_joint={self.kappa_joint:.8g}"
        )
        rows = (
            f"split {self.cost_split:.8g}",
            f"collapse {self.cost_collapse:.8g}",
            f"joint {self.cost_joint:.8g}",
        )
        verdict = f"winner {self.winner} (a is common to all HMC rows, so it cannot tell them apart)"
        if self.contested:
            verdict += "; contested: " + ", ".join(self.contested)
        return head + " | " + " | ".join(rows) + " | " + verdict


def build_ladder(
    inputs: LadderInputs,
    * ,
    strategy: Literal["declared", "cost"] = "declared",
) -> LadderRecord | None:
    """Build the printable record, or ``None`` when the strategy is declared.

``strategy="declared"`` means "do not build a scoreboard at all", so the
    returned value is ``None`` and every declared plan stays byte-identical.
    ``strategy="cost"`` builds the three rows, runs the argmin, and returns a
    record whose :meth:`LadderRecord.line` is empty exactly when the
    scoreboard abstained.
    """
    if strategy == "declared":
        return None
    rows = _rows(inputs)
    verdict = scoreboard(rows)
    by_strategy = {row.strategy: row for row in rows}
    return LadderRecord(
        strategy=strategy,
        winner=(verdict.winner.strategy if verdict.winner is not None else ""),
        rho=float(inputs.rho),
        kappa_cond=float(inputs.kappa_cond),
        kappa_marg=float(inputs.kappa_marg),
        kappa_joint=float(inputs.kappa_joint),
        cost_split=float(by_strategy["split"].cost_hi),
        cost_collapse=float(by_strategy["collapse"].cost_hi),
        cost_joint=float(by_strategy["joint"].cost_hi),
        contested=tuple(row.strategy for row in verdict.contested),
        abstained=verdict.abstained,
        fingerprint=_fingerprint(inputs, rows),
        kappa_x=float(inputs.kappa_x),
        a=float(inputs.a),
        c_gtheta=float(inputs.timing.c_gtheta),
        c_gc=float(inputs.timing.c_gc),
        c_A=float(inputs.timing.c_A),
        c_g_all=float(inputs.timing.c_g_all),
        m=int(inputs.m),
    )


#: The three terms every row's predicted cost is partitioned into, in the
#: order the ledger reports them. Named for the INPUT a miss would be located
#: at, which is what the record is for: ``c`` is the Gibbs amplification
#: ``tau(c) - 1`` (zero on the two whole-graph rows, which pay no sweep),
#: ``a`` is the leapfrog-gradient term whose calibration constant is ``a``,
#: and ``cg`` is the inner CG solve. They sum to the row's cost at the nominal
#: timing constants, which :func:`cost_shares` normalises to one.
_TERM_NAMES = ("c", "a", "cg")


def _terms(inputs: LadderInputs, strategy: str) -> tuple[float, float, float]:
    """One row's cost, split into its amplification, gradient and solve terms.

    Spelled from the same three expressions as :func:`_rows` and NOT factored
    against them -- ``split_cost``'s ``tau(c) * (G + C)`` is written here as
    ``(tau(c) - 1) * (G + C) + G + C``, which is the same number and the only
    rearrangement this module performs. Nothing is cancelled; the CG term
    still appears once per row and ``m`` times inside the split's own solve.
    """
    t = inputs.timing
    if strategy == "split":
        gradient = inputs.a * math.sqrt(inputs.kappa_cond) * t.c_gtheta
        solve = inputs.m * k_cg(inputs.kappa_x, K_CG_TOL) * t.c_A
        return (tau(inputs.rho) - 1.0) * (gradient + solve), gradient, solve
    if strategy == "collapse":
        gradient = inputs.a * math.sqrt(inputs.kappa_marg) * t.c_gc
        solve = k_cg(inputs.kappa_x, K_CG_TOL) * t.c_A
        return 0.0, gradient, solve
    if strategy == "joint":
        return 0.0, inputs.a * math.sqrt(inputs.kappa_joint) * t.c_g_all, 0.0
    raise ValueError(
        f"there is no cost row called {strategy!r}; the scoreboard prices "
        "'split', 'collapse' and 'joint' and nothing else"
    )


def cost_shares(
    inputs: LadderInputs, strategy: str
) -> tuple[tuple[str, float], ...]:
    """Each term's share of one row's predicted cost, summing to one.

    A partition, not a sensitivity: it says which term the PREDICTION is made
    of, which is the question a missed prediction has to start from. The joint
    row is all gradient by construction (share 1.0 on ``a``), the collapse row
    has no amplification (share 0.0 on ``c``), and only the split row can put
    weight on all three.

    A non-finite or non-positive total yields all-zero shares rather than a
    nan: a row priced at ``+inf`` has no attribution to give, and a nan share
    would win no comparison and be skipped by the dominance test in silence.
    """
    amplification, gradient, solve = _terms(inputs, strategy)
    total = amplification + gradient + solve
    if not math.isfinite(total) or total <= 0.0:
        return tuple((name, 0.0) for name in _TERM_NAMES)
    parts = (amplification / total, gradient / total, solve / total)
    return tuple(zip(_TERM_NAMES, parts, strict=True))


def dominant_input(shares: tuple[tuple[str, float], ...]) -> str:
    """The one term above :data:`DOMINANCE_SHARE`, or ``""`` where none is.

    ``""`` is abstention and is the honest answer whenever two terms could be
    tied for largest -- see :data:`DOMINANCE_SHARE`. The caller still has the
    shares.
    """
    for name, share in shares:
        if share_is_dominant(share):
            return name
    return ""


def relative_width(low: float, high: float) -> float:
    """``high / low`` -- how much room a predicted interval left itself.

    Reported rather than gated. With only :data:`TIMING_NOISE_TOLERANCE`
    declared, every row of :func:`_rows` is linear in the timing constants, so
    this is ``(1 + t) / (1 - t)`` exactly and identically on all three rows; a
    wider one is evidence that an undeclared uncertainty got in, and a reader
    who sees it does not need a threshold to know what to think.
    """
    if low <= 0.0 or not math.isfinite(high):
        return math.inf
    return high / low


def recorded_inputs(record: LadderRecord) -> LadderInputs:
    """The :class:`LadderInputs` a :class:`LadderRecord` was built from.

    Every field round-trips as a scalar, which is why the record can carry
    them at all -- see :class:`LadderRecord`.
    """
    return LadderInputs(
        rho=record.rho,
        kappa_cond=record.kappa_cond,
        kappa_marg=record.kappa_marg,
        kappa_joint=record.kappa_joint,
        kappa_x=record.kappa_x,
        a=record.a,
        timing=TimingConstants(
            c_gtheta=record.c_gtheta,
            c_gc=record.c_gc,
            c_A=record.c_A,
            c_g_all=record.c_g_all,
        ),
        m=record.m,
    )


@dataclass(frozen=True, slots=True)
class CostReconciliation:
    """One run's predicted-versus-measured pair, and what it was made of.

    **This is the part of the module that is worth having even if every cost
    expression above is wrong.** It does not assert that the prediction was
    right; it records the prediction, the measurement, and the inputs both
    were computed from -- which is what a cost model needs to be calibrated
    against, and what this package has never collected.

    Attributes:
        strategy: which row this run is reconciled against -- the row for what
            RAN, not the row the scoreboard would have picked. The scoreboard
            changes no routing, so those are frequently different, and
            reconciling against the winner would compare a measurement of one
            thing with a prediction of another.
        predicted_lo, predicted_hi: the row's cost interval, in seconds per
            effective sample.
        measured: ``seconds / ess`` actually observed, same units.
        seconds, ess: the two numbers ``measured`` was formed from, kept apart
            because a miss caused by a slow wall clock and a miss caused by a
            chain that did not mix are different diagnoses.
        within: whether ``measured`` landed inside the predicted interval.
            **A ``True`` here is only as strong as ``width``**: an interval
            opened wide enough is satisfied by anything.
        width: ``predicted_hi / predicted_lo``. With only the timing noise
            declared as uncertainty this is exactly
            ``(1 + t) / (1 - t) = 1.2222`` at ``t = 0.10``; anything wider came
            from an input the model does not declare, and that is the third
            place a missed prediction can be located.
        shares: each cost term's share of the prediction, in ``("c", "a",
            "cg")`` order -- see :func:`cost_shares`.
        dominant: the term above :data:`DOMINANCE_SHARE`, or ``""``.
        rho, a, kappa_cond, kappa_marg, kappa_joint, kappa_x, m: the inputs,
            recorded verbatim, because "the inputs were written down" is the
            claim this record actually makes.
        fingerprint: the scoreboard's own input fingerprint, so a ledger row
            can be matched to the prediction it came from.
    """

    strategy: str
    predicted_lo: float
    predicted_hi: float
    measured: float
    seconds: float
    ess: float
    within: bool
    width: float
    shares: tuple[tuple[str, float], ...]
    dominant: str
    rho: float
    a: float
    kappa_cond: float
    kappa_marg: float
    kappa_joint: float
    kappa_x: float
    m: int
    fingerprint: str

    def line(self) -> str:
        """The ledger's one printed line: predicted, measured, attribution."""
        head = (
            f"cost ledger [{self.strategy}]: predicted "
            f"[{self.predicted_lo:.6g}, {self.predicted_hi:.6g}] s/ESS "
            f"(width {self.width:.4g}x), measured {self.measured:.6g} s/ESS "
            f"from {self.seconds:.6g} s at ESS {self.ess:.6g}"
        )
        verdict = "inside the interval" if self.within else "outside the interval"
        shares = ", ".join(f"{name} {share:.3f}" for name, share in self.shares)
        blame = (
            f"dominated by {self.dominant}"
            if self.dominant
            else "no term above one half, so no input is named"
        )
        return f"{head} | {verdict} | shares: {shares} | {blame}"


def reconcile(
    record: LadderRecord, strategy: str, *, seconds: float, ess: float
) -> CostReconciliation:
    """Build the ledger row for a run against the scoreboard that predicted it.

    ``ess`` is the posterior's own effective sample size, MIN over every site
    and coordinate. A zero makes ``measured`` ``+inf`` rather than raising: a run
    that produced no effective samples took infinitely long per one of them,
    which is both the true statement and the one that lands outside every
    finite interval.
    """
    inputs = recorded_inputs(record)
    rows = {row.strategy: row for row in _rows(inputs)}
    row = rows[strategy]
    measured = seconds / ess if ess else math.inf
    shares = cost_shares(inputs, strategy)
    return CostReconciliation(
        strategy=strategy,
        predicted_lo=float(row.cost_lo),
        predicted_hi=float(row.cost_hi),
        measured=float(measured),
        seconds=float(seconds),
        ess=float(ess),
        within=bool(row.cost_lo <= measured <= row.cost_hi),
        width=float(relative_width(row.cost_lo, row.cost_hi)),
        shares=shares,
        dominant=dominant_input(shares),
        rho=float(inputs.rho),
        a=float(inputs.a),
        kappa_cond=float(inputs.kappa_cond),
        kappa_marg=float(inputs.kappa_marg),
        kappa_joint=float(inputs.kappa_joint),
        kappa_x=float(inputs.kappa_x),
        m=int(inputs.m),
        fingerprint=record.fingerprint,
    )


__all__ = [
    "CONTESTED_BANDWIDTH",
    "DOMINANCE_SHARE",
    "TIMING_NOISE_TOLERANCE",
    "K_CG_TOL",
    "CostReconciliation",
    "CostRow",
    "LadderInputs",
    "LadderRecord",
    "Scoreboard",
    "TimingConstants",
    "build_ladder",
    "cg_tol_positive",
    "collapse_cost",
    "cost_shares",
    "dominant_input",
    "gap_is_contested",
    "joint_cost",
    "k_cg",
    "reconcile",
    "recorded_inputs",
    "relative_gap",
    "relative_width",
    "scoreboard",
    "share_is_dominant",
    "split_cost",
    "tau",
    "timing_noise_in_domain",
    "timing_reference",
]