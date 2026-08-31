"""The A4 pilot: one squared term against the published plan's coupling (P7).

**The pilot's job is not to hand back a better ``c``.**  Its ESS cannot
support one, and the number would not be usable if it could.  Measured in this
checkout on Neal's funnel (``v ~ N(0, 3)``, ``x ~ N(0, exp(v/2))``), where the
Laplace correlation is exactly 0.0 and the true dependence is entirely in the
second moment, at 200 000 iid draws:

* seed 0 reads linear cc **0.007992**, quadratic cc **0.115687** -- ratio
  **14.48**;
* over twenty seeds the quadratic reading spans 0.1018 to 0.2389 (a factor of
  2.35) and the linear one spans 0.00055 to 0.01218 (a factor of 22);
* a second, independently constructed feature set recorded in the plan reads
  **0.619** for the same phenomenon -- 6.08x the smallest reading here.

So the absolute value is estimator-dependent to within a factor of six on one
and the same funnel, and **the ratio is the signal**.  The null confirms it
from the other side: on a jointly Gaussian pair at rho=0.6, twenty seeds of
200 000 draws, the ratio reads between 1.0000001 and 1.0000508.  Squares add
nothing when there is nothing for them to add.

**The verdict is asymmetric, and that is the whole design.**  The pilot can
veto a switch; it can never order one.

* quadratic cc above its sampling floor ``sqrt(p_aug / N_eff)`` **and** above
  the linear cc by :data:`DECLARED_MULTIPLE` -> **veto the switch**, name the
  funnel, abstain from proposing anything;
* anything else -> **inconclusive**, and A3's decision stands exactly as it
  was, reported with ``blind_to=("gaussian-only",)`` so that the abstention is
  read as abstention rather than as endorsement.

**A stuck pilot therefore costs nothing, and it never quietly becomes a coin
flip.**  An inconclusive pilot returns the caller's own proposal unchanged --
:func:`resolve_switch` is the one place that is spelled, and
``tests/dispatch/test_pilot.py`` turns red if it ever abstains instead.

Only run it when there is something to veto: :func:`pilot_is_warranted` says
so, and it is a pair of booleans rather than a measurement, because "would A3
switch away from the published default" and "is the gap inside the timing
noise" are both already decided by the time the pilot is reachable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# --- D101: the declared multiple -------------------------------------------
DECLARED_MULTIPLE: float = 7.0
"""How far the quadratic canonical correlation must exceed the linear one.

**Bracketed by two measurements, and deliberately not fitted to the funnel.**
The lower end is the estimator's own arbitrariness: the same funnel's
quadratic canonical correlation reads 0.1018 under the feature construction in
this module and 0.619 under the independently constructed one recorded in the
plan, a factor of **6.08**.  A ratio below that cannot be told apart from a
change of features.  The upper end is the phenomenon: over twenty seeds of
200 000 funnel draws the smallest ratio is **8.36**, so a multiple at or above
that would fail to veto the very geometry it exists for.

7.0 is the round number inside ``(6.08, 8.36)``.  Its distance from the null is
not close: the Gaussian control's ratio never left ``[1.0000001, 1.0000508]``.
Twenty of twenty funnel draws veto at this multiple; twenty of twenty Gaussian
draws do not.
"""


def sampling_floor(p_aug: int, n_eff: float) -> float:
    """``sqrt(p_aug / n_eff)`` -- the size a null canonical correlation reaches.

    The largest canonical correlation between ``p_aug`` features and an
    independent target is not zero at finite sample size; it is of order
    ``sqrt(p_aug / N_eff)``, which is what a correlation has to clear before it
    is evidence of anything.  Measured on this module's funnel construction:
    ``p_aug=4`` at ``N_eff=200000`` gives 0.004472, and the funnel's LINEAR
    reading falls below it on 20-seed minimum (0.00055) -- the floor is doing
    work, not decorating the report.

    A pure formula, like :func:`~bayesmith.dispatch.costs.tau`, and defended
    the same way: ``n_eff`` is an effective sample size, so a non-positive
    value is a caller error rather than a boundary this module prices.
    """
    return math.sqrt(p_aug / n_eff)


def quadratic_cc_crosses_floor(quadratic_cc: float, floor: float) -> bool:
    """D102: ``quadratic_cc > floor`` -- the reading is above sampling noise.

    Open on purpose.  A correlation AT its sampling floor is what independent
    features produce, so admitting it would let the pilot veto a switch on
    nothing at all.
    """
    return quadratic_cc > floor


def ratio_exceeds_declared_multiple(ratio: float) -> bool:
    """D101: ``ratio > DECLARED_MULTIPLE`` -- the squares changed the answer.

    The quadratic-over-linear ratio, against the declared multiple.  Open for
    the same reason as the floor: a ratio exactly at the multiple is the
    boundary of what a change of features alone can produce.
    """
    return ratio > DECLARED_MULTIPLE


def pilot_is_warranted(*, switches_away: bool, contested: bool) -> bool:
    """Whether there is anything for a pilot to veto.

    Only two situations can be improved by spending draws on one: A3 would
    switch away from the published default, or the cost gap that decided it
    fell inside the timing noise.  Anywhere else the published default is
    already what runs, so a veto would change nothing and the pilot is not run
    at all -- which is the cheapest possible way for it to cost nothing.
    """
    return switches_away or contested


def augment(block: np.ndarray) -> np.ndarray:
    """``[block, block**2]`` -- the quadratic feature construction.

    The squares are appended, never substituted, so the quadratic canonical
    correlation is taken over a feature space that CONTAINS the linear one.
    That is what makes the ratio a lower bound of one over the other rather
    than two unrelated numbers: the quadratic reading can never be smaller
    than the linear one except by sampling noise.
    """
    values = np.asarray(block, dtype=float)
    return np.concatenate([values, values * values], axis=1)


def canonical_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Largest canonical correlation between two feature blocks of ONE sample.

    The same quantity :func:`~bayesmith.diagnose.coupling.block_coupling`
    reports, measured from draws instead of from a local precision, and by QR
    rather than by Cholesky: the canonical correlations are the singular values
    of ``Q_l.T @ Q_r`` for orthonormal bases of the two centred column spaces.
    No covariance is inverted and no factor has to exist, which matters here
    because the augmented block's second moment is heavy-tailed by
    construction -- the funnel's ``x**2`` is exactly the column a Cholesky
    would struggle on.

    Both blocks must come from the SAME draws, in the same row order; that is
    the whole point of the comparison, and nothing here can check it.

    **A feature column that never moved is not detected, and the failure is in
    the safe direction.**  A site held constant over the pilot's draws gives a
    zero column, whose QR basis vector is an arbitrary direction rather than a
    refusal, and the correlation against it is whatever that direction happens
    to hit.  Rank-detecting it would need a tolerance -- a threshold with no
    measurement behind it -- so it is named here instead.  What it can do is
    inflate a reading, and an inflated reading can only produce a VETO, which
    keeps the published default; the pilot has no power to switch anything on,
    so its one failure mode costs a missed optimisation rather than a wrong
    answer.  A site that did not move is also the larger problem.
    """
    l_centred = np.asarray(left, dtype=float)
    r_centred = np.asarray(right, dtype=float)
    l_centred = l_centred - l_centred.mean(axis=0)
    r_centred = r_centred - r_centred.mean(axis=0)
    q_l, _ = np.linalg.qr(l_centred)
    q_r, _ = np.linalg.qr(r_centred)
    return float(np.linalg.svd(q_l.T @ q_r, compute_uv=False)[0])


@dataclass(frozen=True, slots=True)
class PilotReport:
    """What one pilot measured, and the one thing it is allowed to conclude.

    Attributes:
        linear_cc: largest canonical correlation on the linear features.
        quadratic_cc: the same, with the squares appended.
        ratio: ``quadratic_cc / linear_cc``, ``inf`` where the linear reading
            is exactly zero.  **This is the signal**; the two absolute values
            are estimator-dependent by a factor of six on one fixture and are
            carried for the record, not for a threshold.
        floor: ``sampling_floor(p_aug, n_eff)``, the null size of a canonical
            correlation at this many features and this much information.
        p_aug: how many augmented features the two blocks contributed.
        n_eff: the effective sample size the floor was computed at.
        vetoed: whether both conditions fired.  ``False`` is inconclusive and
            is **not** an endorsement of anything -- see :attr:`blind_to`.
        blind_to: ``("gaussian-only",)`` on an inconclusive pilot, and empty on
            a veto.  An inconclusive pilot has not shown the geometry is
            Gaussian; it has failed to show that it is not, which is a
            different statement and the one this field carries.
        reason: the verdict in words, naming the funnel where it vetoed.
    """

    linear_cc: float
    quadratic_cc: float
    ratio: float
    floor: float
    p_aug: int
    n_eff: float
    vetoed: bool
    blind_to: tuple[str, ...]
    reason: str

    def line(self) -> str:
        """The pilot's one printed line, verdict last."""
        head = (
            f"A4 pilot: linear cc={self.linear_cc:.6g}, quadratic cc="
            f"{self.quadratic_cc:.6g}, ratio={self.ratio:.6g} against a "
            f"declared multiple of {DECLARED_MULTIPLE:g} and a sampling floor "
            f"of {self.floor:.6g} (p_aug={self.p_aug}, N_eff={self.n_eff:.6g})"
        )
        if self.blind_to:
            return head + f" | {self.reason} | blind_to={self.blind_to}"
        return head + f" | {self.reason}"


def pilot_report(
    first: np.ndarray, second: np.ndarray, *, n_eff: float
) -> PilotReport:
    """Two canonical correlations on ONE sample, and the asymmetric verdict.

    The linear and quadratic readings are taken from the same draws, in the
    same row order, so the ratio divides out everything the two constructions
    share -- which is the only reason a ratio is usable where neither absolute
    value is.
    """
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    linear = canonical_correlation(left, right)
    left_augmented = augment(left)
    right_augmented = augment(right)
    quadratic = canonical_correlation(left_augmented, right_augmented)
    p_aug = int(left_augmented.shape[1] + right_augmented.shape[1])
    floor = sampling_floor(p_aug, n_eff)
    ratio = quadratic / linear if linear else math.inf
    vetoed = quadratic_cc_crosses_floor(quadratic, floor) and (
        ratio_exceeds_declared_multiple(ratio)
    )
    if vetoed:
        reason = (
            f"veto: the quadratic canonical correlation {quadratic:.6g} is "
            f"above its {floor:.6g} sampling floor AND exceeds the linear "
            f"{linear:.6g} by {ratio:.6g}x, past the declared "
            f"{DECLARED_MULTIPLE:g}x. That is the funnel signature -- the "
            "dependence lives in the second moment, where a Laplace coupling "
            "reads it as zero -- so the switch this pilot was run for is "
            "refused and NOTHING is proposed in its place; the pilot's ESS "
            "cannot support a better c and its absolute readings move by a "
            "factor of six between feature constructions"
        )
        blind_to: tuple[str, ...] = ()
    else:
        reason = (
            f"inconclusive: the quadratic canonical correlation {quadratic:.6g} "
            f"did not clear both conditions (floor {floor:.6g}, "
            f"{DECLARED_MULTIPLE:g}x the linear {linear:.6g}), so this pilot "
            "changes nothing and the decision that was already taken stands "
            "unaltered. This is abstention, not endorsement: a pilot that "
            "found no curvature has not shown there is none"
        )
        blind_to = ("gaussian-only",)
    return PilotReport(
        linear_cc=float(linear),
        quadratic_cc=float(quadratic),
        ratio=float(ratio),
        floor=float(floor),
        p_aug=p_aug,
        n_eff=float(n_eff),
        vetoed=bool(vetoed),
        blind_to=blind_to,
        reason=reason,
    )


def stack_sites(samples: Mapping[str, Any], names: Sequence[str]) -> np.ndarray:
    """``{site: draws}`` for the named sites, flattened to ``(draws, features)``.

    The draw axis leads, as it does everywhere a :class:`Posterior` carries
    samples, and every other axis of a site is flattened into columns so that
    a plated site contributes one column per coordinate rather than being
    silently reduced.
    """
    columns = []
    for name in names:
        values = np.asarray(samples[name], dtype=float)
        columns.append(values.reshape(values.shape[0], -1))
    return np.concatenate(columns, axis=1)


def pilot_from_samples(
    samples: Mapping[str, Any],
    first: Sequence[str],
    second: Sequence[str],
    *,
    n_eff: float,
) -> PilotReport:
    """:func:`pilot_report` on two named blocks of one posterior sample."""
    return pilot_report(
        stack_sites(samples, first), stack_sites(samples, second), n_eff=n_eff
    )


def resolve_switch(declared: str, proposed: str, report: PilotReport) -> str:
    """``proposed`` unless the pilot vetoed it, in which case ``declared``.

    **The inconclusive branch returns the proposal unchanged**, and that is the
    behaviour a test in this package turns red on if it is ever weakened to an
    abstention.  A pilot that found nothing has not earned the right to
    override a decision that was taken on other evidence; only a veto has, and
    a veto returns the published default rather than a third answer of its own.
    """
    if report.vetoed:
        return declared
    return proposed


__all__ = [
    "DECLARED_MULTIPLE",
    "PilotReport",
    "augment",
    "canonical_correlation",
    "pilot_from_samples",
    "pilot_is_warranted",
    "pilot_report",
    "quadratic_cc_crosses_floor",
    "ratio_exceeds_declared_multiple",
    "resolve_switch",
    "sampling_floor",
    "stack_sites",
]
