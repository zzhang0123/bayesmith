"""Deterministic quadrature oracles for a residual evidence, with a certificate.

**Which side is the oracle, said once and in the module that defines both,
because getting it backwards is how R4's square-root route fooled its own
test.**

===================== ============================================ ======
name                  what it integrates                           tier
===================== ============================================ ======
``oracle_joint``      ``log_joint`` over **every** latent, the       1
                      residual block and the exact block together
``oracle_collapsed``  ``log_joint`` of the **reduced** graph over    5
                      the residual latents alone
===================== ============================================ ======

``oracle_joint`` shares only ``log_joint`` and ``evaluate`` with production --
the model's own definition -- so it is independent of the elimination and may
grade it. ``oracle_collapsed`` calls ``collapse_graph`` and
``marginal_log_density``, which ARE the elimination, so it is self-consistency
with respect to it: it may grade a sampler running on the collapsed problem and
it may never grade the collapse. In every comparison this module builds, the
uncollapsed side is the oracle and the collapsed side is what is being graded.

**Why a certificate, and why it can fail.** A gap between two quadratures where
one has not converged is indistinguishable from a defect in the other. The R5
plan walked into exactly that: a first grid on ``shared_ancestor`` reported a
gap of 6.55 nats and the uncollapsed side was simply drifting, not wrong.
``test_residual_oracle.py`` pins that span as a regression and requires an
ABSTAIN. So :func:`quadrature` never returns a bare number. It returns a
:class:`Quadrature` whose ``value`` is ``None`` unless **five** conditions
hold, each of which is a bound this module computes rather than a level it was
handed. (This sentence said "three" while listing four, from 2026-08 until
2026-09-06; the count is now asserted by
``test_residual_oracle.py::test_the_conditions_this_modules_docstring_lists_are_counted_not_typed``
rather than typed.)

1. **The increments shrink.** With the interval count doubled at each step, the
   last two movements give ``ratio = |delta_k| / |delta_(k-1)|``. ``ratio >= 1``
   means the value is not settling; on a span whose interior contains
   ``shared_ancestor``'s degenerate conditional the movement is a CONSTANT
   ``log 4 = 1.386`` per doubling, forever, which is what an unconverged
   trapezoid looks like from the outside.
2. **The geometric tail of those increments reaches the demanded resolution.**
   Summing the remaining movements as a geometric series -- the idiom
   ``dispatch/evidence.py::_tail_sum`` already uses for the prior audit's D109
   -- bounds what is left to gain by refining at
   ``|delta_k| * ratio / (1 - ratio)``, and the caller declares the level that
   bound must reach. **The resolution is declared and not inferred, because a
   band assembled from each side's own residual error otherwise WIDENS to
   contain whatever it is asked about.** An earlier draft of this module stopped
   at ``n_eval * eps * max(1, |value|)``, the naive-summation bound; in two
   dimensions ``n_eval`` is ``n**2``, so that floor QUADRUPLED with every
   refinement while the error it was chased with fell by four, and the test got
   easier the less work was done. The floor is now the pairwise-summation bound
   ``ceil(log2(n_eval)) * eps * max(1, |value|)`` -- numpy reduces with pairwise
   summation, whose relative error over ``N`` non-negative terms is
   ``O(log2 N) * eps``; a relative error on ``Z`` is an absolute error on
   ``log Z``, and ``max(1, |value|)`` carries the representation error of the
   logarithm itself -- and it enters the BAND rather than the stopping rule.
3. **The integrand is a number everywhere on the grid.** ``nan`` and ``+inf``
   are abstains with their own sentence, never a value that propagates.
   ``shared_ancestor`` is where this bites, and the reason is worth the line it
   takes: ``np.linspace(-3, 7, 101)`` lands EXACTLY on ``tau = 0`` -- where
   ``Normal(0, 0).log_prob`` is ``nan`` -- while ``jnp.linspace`` over the same
   interval misses it by ``5.55e-16`` and drifts instead. Two grids that read
   identically in a test, one ``nan`` and one a plausible number, chosen by
   which library built the axis.
4. **The integrand is decaying at every edge of every span.** Per axis and per
   side, ``rho = (outermost trapezoid cell) / (its neighbour)``. ``rho >= 1``
   says the mass is still growing as the span runs out, so the mass is outside
   the span and the number is an integral over the wrong region. ``rho < 1``
   models the tail past the edge as geometric and bounds the mass it holds at
   ``cell * rho / (1 - rho)``.
5. **The grid samples the integrand's own peak.** The only condition here that
   is not a statement about the refinement sequence, and the only one that
   could catch what it was added for. Conditions 1, 2 and 4 all ask whether the
   trapezoid is consistent with ITSELF, and a grid that steps over a narrow
   peak is perfectly consistent with itself: refining it keeps giving the same
   answer, so the increments look converged, the edges decay, and the value is
   wrong by whatever the peak was worth. Measured on ``high_snr_curvature``:
   ``sigma = 2e-6`` puts the curvature width at ``5.73e-07`` while the finest
   grid the budget affords is spaced ``5.63e-03``. This function returned
   **-2376535.508689067** with ``refused=None`` and a bound of ``9.54e-09``,
   against a Laplace estimate from the peak's own height and curvature of
   **+131.57** -- certified, and wrong by 2.4 million nats. The condition
   ascends THE INTEGRAND (not the graph's ``log_joint``: ``oracle_collapsed``
   integrates a different function over fewer axes), verifies the result is
   stationary by its gradient rather than by its claimed height, and requires
   the peak to sit within one curvature width of a grid point.

The three ERROR bounds -- refinement tail, float floor, truncation -- are also
the AGREEMENT BAND (**D111**): two quadratures agree
when the gap between them is no larger than the sum, over both sides, of
``refinement tail + float floor + truncation``, **and never less than**
:data:`AGREEMENT_FLOOR` ``* max(1, |log Z|)``. The FORM is derived -- each
summand is a named error source computed from the run -- and the one declared
LEVEL is :data:`AGREEMENT_FLOOR`, which is measured. That is D107's provenance
exactly: derived form, measured level.

**Why the band needs a floor, since it was green without one.** On
``indirect_ancestor`` both sides converge spectrally, so the refinement tail
falls to ``1e-16`` and the band collapses onto the arithmetic: a measured gap of
``1.95e-14`` against a band of ``5.45e-14``, which is 22 ULP of the answer
against a bound of 34. That is a comparison of one machine's rounding, and red
line 9 has cost this project four release tags. The floor makes the assertion
say what it means -- these two integrals are the same number -- at a level a
differently-rounding LAPACK cannot reach, and the price is quantified rather
than assumed:
``test_the_agreement_floor_is_far_below_the_defect_it_must_catch`` measures the
``dense_operator`` scaling against it and finds five to seven orders of margin
left. **A band assembled from
error bounds could in principle swallow a defect, so it is not asserted, it is
demonstrated**: ``test_the_oracle_is_not_blind_to_a_scaled_dense_operator``
scales ``dense_operator`` by 1.03 and the same comparison goes red with five to
seven orders of margin. R4's close-out records the square-root cross-route test
surviving exactly that mutation, and this module exists so that it does not
happen twice.

**What the geometric tail model is worth, measured rather than asserted in
either direction.** ``rho`` is estimated from the outermost pair of cells, so
the bound is exact for a geometric tail, high for anything decaying faster and
low for anything decaying slower. Measured on ``overflowing_outside_latent`` at
``|z| <= 60`` -- the span the R5 plan used while planning -- against the
``|z| <= 100`` value: the bound reads ``1.48e-03`` for an actual ``1.27e-03``,
so it BOUNDS the error, by 17 per cent, and the factor is stable to three
digits over six grids from ``n = 401`` to ``n = 12801``.
``test_the_truncation_bound_measures_the_mass_a_short_span_left_out`` pins that.
〔An earlier draft of this paragraph said the bound UNDERestimates, by 1.75x.
That came of comparing a bound this module reports for BOTH edges against a
measurement taken one edge at a time -- 7.45e-04 per side, and there are two.
The correction is recorded rather than swapped in silently, because the
direction of a bound's error is the only interesting thing about it.〕
**A polynomial tail is the case the geometric model does not cover, and it was
found by putting an integrand to it rather than by reading.** For a ``t**-p``
tail the geometric sum comes out ``p - 1`` over ``p`` of the truth, so on
``1 / (1 + z**2)`` it is exactly HALF -- measured against the closed form
``2 arctan(1/S) / pi`` at three spans in
``test_the_edge_bound_covers_a_polynomial_tail``. On a Cauchy pair whose
integrand goes as ``z**-4`` the shortfall reads 1.3333, 1.3334 and 1.3334 at
spans of 100, 1000 and 2000, so at ``|z| <= 2000`` the certificate published
4.09e-11 while the truth sat 5.45e-11 away, outside it. A bound that excludes
the truth is not a bound, so :func:`_power_tail` adds the power-law alternative
and the reported figure is the larger of the two.

**A span is a domain, not a claim about the support.** ``shared_ancestor``
declares ``tau ~ N(2, 0.5)`` and ``x ~ N(0, |tau|)``, so ``p(x | tau)`` is
degenerate at ``tau = 0``, which is INTERIOR to the support. A span that spans
it does not merely lose accuracy: with ``tau = 0`` on the grid,
``marginal_log_density`` aborts through ``eqx.error_if``.  So the caller
declares a finite span, :func:`excluded_prior_mass` records how much declared
prior mass that span leaves out, and the number is never reported without it.
The excluded PRIOR mass is recorded and is NOT the abstain criterion --
``overflowing_outside_latent`` on ``|z| <= 100`` excludes 99.99 per cent of a
``Cauchy(0, 1e6)`` prior and is right to four hundred digits of the integrand,
because the likelihood and not the prior decides where the integrand's mass is.

**The GAP is the quantity these comparisons pin; the values are not.** A
twelve-digit trapezoid sum is a platform-dependent reduction, and red line 9
forbids pinning one.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from bayesmith.dispatch.classify import prior_environment
from bayesmith.dispatch.collapse import collapse_graph
from bayesmith.exact.gaussian import apply_probabilistic
from bayesmith.graph.evaluate import log_joint
from bayesmith.graph.graph import Graph

#: Double precision is a premise of every number here, never an accident of the
#: ambient config.  The caller is expected to be inside ``jax.enable_x64(True)``
#: and :func:`quadrature` refuses if it is not.
EPS = float(np.finfo(np.float64).eps)

#: **D111.** The relative floor under the agreement band, and the resolution the
#: refinement is asked to reach. One number, used for both, because they are the
#: same statement: the oracle stops refining when what is left to gain is below
#: the level at which it is willing to claim two integrals are equal.
#:
#: Derived FORM -- relative to ``max(1, |log Z|)``, because the precision a log
#: evidence can be represented in scales with its magnitude. Measured LEVEL, and
#: both halves of the measurement are pinned as tests rather than asserted here:
#: it is about four orders ABOVE the float-level disagreement between the two
#: routes on this platform (worst measured ``1.95e-14`` on ``indirect_ancestor``,
#: 22 ULP), so a platform whose LAPACK rounds a thousand times worse still
#: certifies; and seven orders BELOW the smallest defect the comparison exists
#: to catch (scaling ``dense_operator`` by 1.03 moves the gap to ``2e-02``).
AGREEMENT_FLOOR = 1e-9

#: The largest product grid this oracle will build, in points. **A BUDGET, not
#: a threshold**: it decides how much arithmetic to spend and never what the
#: answer is, which is why it carries no D-number (R5 plan 0.7 rules the same way
#: about the likelihood-evaluation budget).
#:
#: It exists because the alternative is a hang. ``start=201`` is a sensible first
#: grid on one axis and 1.63e9 points on four, which does not fail -- it
#: allocates for a quarter of an hour and then dies on memory, telling the caller
#: nothing about their model. Above the budget the refinement ABSTAINS and says
#: what the next grid would have cost, which is exactly R5 Task 3.2's stop-rule:
#: the dimension at which quadrature stops being an oracle is the dimension at
#: which the grid stops fitting. 3e7 points is about 240 MB of float64 per array
#: and about a second per grid on this machine; a caller measuring the boundary
#: itself passes a larger one and pays for it.
MAX_POINTS = 50_000_000

#: How many grid points a single vmapped call evaluates.  Bounds peak memory at
#: a few megabytes whatever the dimension, so the cost of a high-dimensional
#: grid shows up as TIME -- which is what Task 3.2's stop-rule measures -- and
#: not as an allocation failure that says nothing about convergence.
_CHUNK = 1 << 15


@dataclasses.dataclass(frozen=True, slots=True)
class Span:
    """One integration axis: which latent, and the finite interval used."""

    name: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not self.upper > self.lower:
            raise ValueError(
                f"span for {self.name!r} is {(self.lower, self.upper)}, which "
                "is not an interval; an oracle over an empty or inverted span "
                "returns a number with no integrand under it"
            )

    def __str__(self) -> str:
        return f"{self.name} in ({self.lower:g}, {self.upper:g})"


@dataclasses.dataclass(frozen=True, slots=True)
class EdgeDecay:
    """What the integrand is doing at one end of one axis."""

    axis: str
    side: str
    ratio: float
    fraction: float

    @property
    def decaying(self) -> bool:
        return self.ratio < 1.0

    def __str__(self) -> str:
        return f"{self.axis}/{self.side}: rho={self.ratio:.4g} tail={self.fraction:.3e}"


@dataclasses.dataclass(frozen=True, slots=True)
class PeakResolution:
    """Whether the finest grid resolves the highest point the ascent reached.

    **Read that sentence rather than "the integrand's own maximum", which is
    what an earlier draft claimed and what this cannot deliver.** The starts are
    the grid's own argmax plus prior draws, so a feature the grid never sampled
    AND no start lands near is invisible to this condition as it is to the
    trapezoid. Built and measured: a background Gaussian with a spike of width
    1e-4 at ``x = 7.3137``, placed deliberately between grid nodes, certifies at
    ``resolved=True`` with the note naming ``x=0`` as the peak, and the value is
    9.95e-03 nats wrong -- **4.27e12 times its own bound**. Move the same spike
    onto a grid node and the ascent finds it and the condition fires. So this is
    sampling-luck dependent at the sub-spacing scale, which is the honest
    statement of what one grid can know.

    The one condition here that is not a statement about the refinement
    sequence, and the only one that could have caught what it was written for.
    A trapezoid that steps over a narrow peak converges beautifully -- to the
    integral of everything except the mass -- so every increment is tiny, every
    edge decays, the integrand is finite everywhere, and the certificate reads
    perfect. Measured on ``high_snr_curvature``: ``sigma = 2e-6`` puts the
    curvature width at ``5.73e-07`` while the finest grid the budget affords on
    the declared span is spaced ``5.63e-03``, 9811 times coarser.
    ``oracle_joint`` returned **-2376535.508689067** with ``refused=None`` and a
    bound of ``9.54e-09``, against a Laplace estimate from the peak's own height
    and curvature of **+131.57**. Nothing else in this certificate can see a
    2.4-million-nat error, because refining a grid that keeps missing the peak
    keeps giving the same answer.

    ``resolved`` is the verdict; ``note`` says which of the four ways it was
    reached -- located and resolved, located and too coarse, not located, or not
    stationary. **Declining is not passing** (red line 14): every way of failing
    to run is ``resolved=False`` with its own sentence, never a quiet True.
    """

    resolved: bool
    location: Mapping[str, float] | None
    widths: Mapping[str, float] | None
    height: float | None
    note: str

    def __str__(self) -> str:
        return f"{'resolved' if self.resolved else 'UNRESOLVED'}: {self.note}"


def _ascend(
    log_density: Callable[[dict[str, Any]], Any],
    names: Sequence[str],
    starts: Sequence[dict[str, float]],
    bounds: Sequence[tuple[float, float]],
) -> tuple[dict[str, float] | None, dict[str, float] | None, float | None, str]:
    """Gradient ascent on THE INTEGRAND, not on the graph's ``log_joint``.

    **Which function is ascended is the whole correctness of this check.**
    ``oracle_collapsed`` integrates ``log_joint(reduced, ...)`` over the
    residual latents only, so the peak of the full graph's ``log_joint`` is a
    different point with different curvature -- narrower, because collapsing a
    Gaussian block widens the residual marginal. Ascending the callable
    ``quadrature`` was actually handed makes the check right for both oracles
    and for any other integrand this module is ever pointed at.

    The starts come from the caller, and the grid's own argmax is always among
    them: whatever the grid found is within one spacing of the best point the
    grid can see, and it is free.

    **What is NOT claimed, because a mutant refuted it**: that starting from the
    argmax rather than anywhere else is load-bearing. Replacing it with the
    grid's ARGMIN -- the worst point on the grid -- changes no outcome anywhere
    in this suite. On the integrands here the optimiser reaches the peak from
    either end, which says the trapezoid is what fails on a narrow peak and the
    ascent is not close to its limits. It also means this start is coverage
    against integrands the suite does not contain.
    """
    from scipy import optimize

    def negative(x):
        return -log_density({name: value for name, value in zip(names, x, strict=True)})

    gradient = jax.jit(jax.grad(negative))
    best = None
    for start in starts:
        vector = np.array([start[name] for name in names], dtype=float)
        if not np.all(np.isfinite(vector)):
            continue
        try:
            found = optimize.minimize(
                lambda x: float(negative(jnp.asarray(x))),
                vector,
                jac=lambda x: np.asarray(gradient(jnp.asarray(x)), dtype=float),
                method="L-BFGS-B",
                # **Bounded to the spans.** Unbounded, L-BFGS-B walks off the
                # grid: measured, three fixtures in this suite located a peak
                # 1.81 SPAN-WIDTHS outside the span and still reported "every
                # axis has a grid point within one curvature width of the
                # peak". No wrong number shipped -- condition 4 catches those
                # spans -- but the field said something false, and a spacing
                # compared against a curvature width measured somewhere the
                # grid does not reach is not a statement about this grid.
                bounds=bounds,
            )
        except (ValueError, FloatingPointError, TypeError):
            continue
        if not np.isfinite(found.fun):
            continue
        if best is None or found.fun < best.fun:
            best = found
    if best is None:
        return None, None, None, "the ascent found no finite point from any start"
    # **The located peak must beat every start, or it is not a peak.**
    #
    # What this does NOT catch, corrected after a review measured it: a replaced
    # peak finder. That bypass -- returning the prior centre with a unit width --
    # is caught by `_stationary`, and only by it: with `complaint = None`
    # applied, `test_a_bogus_peak_is_refused_through_the_public_oracle` fails
    # and this guard says nothing. It could not be otherwise, since a guard
    # inside the function that gets replaced is replaced with it, which is the
    # design rule `_stationary`'s own docstring states and this comment used to
    # contradict.
    #
    # What it does catch is narrower and real: a multi-start whose BEST start's
    # optimisation raised and was skipped, leaving a lower mode reported as the
    # peak with that mode's curvature width.
    # `test_the_ascent_refuses_a_result_lower_than_its_own_starts` builds it.
    # **Filtered on the HEIGHT, not on the coordinates.** The finiteness test
    # used to read the start's position, so a start at a finite coordinate
    # whose log-density is `+inf` put `inf` into `heights`, made `max(heights)`
    # infinite, and refused every ascent unconditionally. Same non-finite-poison
    # shape as the NaN gradient one function down, one argument over.
    heights = [
        height
        for start in starts
        if np.all(np.isfinite([start[name] for name in names]))
        and np.isfinite(
            height := float(
                -negative(jnp.asarray([start[name] for name in names], dtype=float))
            )
        )
    ]
    # The slack is RELATIVE. An absolute `1e-9` is scale-blind, and this module
    # meets values where it means nothing: on `high_snr_curvature`,
    # `log_joint(w=0)` is -1.52e12, where one ULP is 2.44e-4 -- 244 000 times
    # the old slack. CLAUDE.md names a scale-blind absolute tolerance as one of
    # the sixteen repairs a Linux run forced.
    #
    # **This line and the finiteness filter above are REDUNDANT against the one
    # test that covers either, and that is measured rather than suspected.**
    # Mutating each alone SURVIVES; mutating both together is killed by
    # `test_a_start_whose_density_is_infinite_does_not_refuse_every_ascent`.
    # The reason is arithmetic: with an unfiltered `+inf` in `heights`, the
    # relative slack computes `inf - 1e-9 * inf = nan` and `x < nan` is False,
    # so the guard silently stops firing -- the absolute form gave `inf - 1e-9
    # = inf` and fired on everything. Neither is kept for that accident. The
    # filter is right because `heights` should hold heights; the slack is right
    # because a tolerance on a log-density cannot be absolute. Recorded so the
    # next reader does not delete one on the grounds that its mutant lives.
    if heights and -float(best.fun) < max(heights) - 1e-9 * max(
        1.0, abs(max(heights))
    ):
        return None, None, None, (
            "the ascent returned a point lower than one of its own starts, so "
            "it is not a maximum"
        )
    try:
        hessian = np.atleast_2d(
            np.asarray(
                jax.hessian(
                    lambda x: log_density(
                        {name: value for name, value in zip(names, x, strict=True)}
                    )
                )(jnp.asarray(best.x)),
                dtype=float,
            )
        )
    except Exception as error:  # noqa: BLE001 - a peak outside the integrand's domain
        return None, None, None, f"the Hessian raised {type(error).__name__}"
    with np.errstate(all="ignore"):
        curvature = -np.diag(hessian)
        widths = np.where(curvature > 0, 1.0 / np.sqrt(np.abs(curvature)), np.inf)
    return (
        dict(zip(names, np.asarray(best.x, dtype=float), strict=True)),
        dict(zip(names, np.asarray(widths, dtype=float), strict=True)),
        float(-best.fun),
        "located",
    )


def _stationary(
    log_density: Callable[[dict[str, Any]], Any],
    names: Sequence[str],
    location: Mapping[str, float],
    widths: Mapping[str, float],
) -> str | None:
    """Is the claimed peak a stationary point? Returns a complaint or ``None``.

    **Separate from :func:`_ascend` on purpose.** A guard living inside the
    function it guards goes with that function when the function is replaced,
    which is exactly how the probe's version was bypassed.

    **And it reads the GRADIENT, not the height.** Two weaker checks were tried
    in the probe and both let the bypass through: trusting the returned height
    trusts the liar, and comparing against a scan of the prior fails when the
    claimed peak IS the prior centre. At a real optimum the gradient vanishes;
    at ``high_snr_curvature``'s prior centre it is of order ``1e12``. The test
    is dimensionless -- one curvature width along the gradient must change the
    integrand by at most one nat -- so it introduces no tuned number.
    """
    finite = [name for name in names if np.isfinite(widths[name])]
    if not finite:
        return "no axis has a finite curvature width, so there is no peak to resolve"
    try:
        gradient = jax.grad(
            lambda values: log_density({name: values[name] for name in names})
        )({name: float(location[name]) for name in names})
    except Exception as error:  # noqa: BLE001 - a peak outside the integrand's domain
        return f"the gradient at the claimed peak raised {type(error).__name__}"
    worst, where = 0.0, ""
    for name in finite:
        step = abs(float(gradient[name]) * widths[name])
        # **A non-finite step must refuse HERE, not fall through.** `worst`
        # starts at 0.0 and `nan > 0.0` is False, so a NaN gradient never
        # becomes `worst`, leaves it at 0.0, and the `isfinite(worst)` clause
        # below then passes: a point that is not stationary reads as
        # stationary because its gradient was poison. Measured on
        # `-(x**2) + 0*nan`-style poisoning at x=5: the claimed peak passed
        # while x=4, whose gradient is finite, was correctly refused.
        if not np.isfinite(step):
            return (
                f"the gradient at the claimed peak is {gradient[name]!r} on "
                f"{name}, so stationarity could not be tested"
            )
        if step > worst:
            worst, where = step, name
    # No `isfinite(worst)` conjunct: the loop above returns on the first
    # non-finite step, so `worst` is finite here by construction. A mutant
    # deleting such a conjunct SURVIVES, which is the signature of a condition
    # with no consequence -- the same reading that removed one from the
    # refinement loop above.
    if worst > 1.0:
        return (
            f"the claimed peak is not stationary: one curvature width along the "
            f"gradient changes the integrand by {worst:.3g} nats on {where}, so "
            "it is a point the ascent did not reach rather than a peak"
        )
    return None


def _prior_starts(
    graph: Graph | None, names: Sequence[str], *, draws: int = 8
) -> list[dict[str, float]]:
    """Extra ascent starts from the prior, beyond the grid's own argmax.

    **What these are measured to buy on a real fixture, so far, is nothing.**
    No fixture in this suite has a prior start that beats the grid's argmax;
    what pins the best-not-first choice is a constructed test
    (``test_the_ascent_keeps_the_best_start_and_not_the_first``) rather than a
    fixture. They are kept because the
    grid argmax alone cannot see a mode the grid stepped over (see
    :class:`PeakResolution`), and a prior draw is the only other place a start
    can come from; they are a small fraction of what this check spends. Do not read
    the multi-start as demonstrated multimodality handling -- it is coverage
    whose value is not yet exercised by a fixture.

    Restricted to ``names`` on purpose: ``oracle_collapsed``'s spans are the
    RESIDUAL latents only, and a start carrying the exact block's coordinates
    would not be a point in the integrand's domain at all.

    Returns ``[]`` rather than raising when there is no sampler -- an improper
    prior has none -- because the grid's own argmax is always in the start set
    and this is coverage on top of it, not the thing being relied on.
    """
    if graph is None:
        return []
    try:
        from bayesmith.dispatch.evidence import compile_evidence_problem

        problem = compile_evidence_problem(graph)
    except Exception:  # noqa: BLE001 - no sampler, or not an evidence problem
        return []
    starts: list[dict[str, float]] = []
    for seed in range(draws):
        try:
            drawn = problem.prior_sample(jax.random.key(1000 + seed))
        except Exception:  # noqa: BLE001
            break
        if not all(name in drawn for name in names):
            break
        starts.append({name: float(np.ravel(np.asarray(drawn[name]))[0]) for name in names})
    return starts


def _peak_resolution(
    log_density: Callable[[dict[str, Any]], Any],
    spans: Sequence[Span],
    names: Sequence[str],
    starts: Sequence[dict[str, float]],
    count: int,
) -> PeakResolution:
    """Condition 5: does the finest grid sample the integrand's own peak?

    The rule is ``h / 2 <= width`` per axis: on a uniform grid of spacing ``h``
    the nearest point to any location is at most ``h / 2`` away, so that is
    "one grid point within one curvature width of the peak".

    **The half is the nearest-point geometry, and nearest-point distance is NOT
    what controls the trapezoid's error.** Aliasing is, and it falls as
    ``exp(-2 pi**2 (width / h)**2)``. Measured against an exact Gaussian with
    this module's own ``_log_trapezoid``, the dispatcher bypassed:

    ========  ==============
    h/width   error (nats)
    ========  ==============
    1.00      5.4e-09
    1.25      6.5e-06
    1.60      9.0e-04
    2.00      1.43e-02
    2.50      8.2e-02
    ========  ==============

    ``AGREEMENT_FLOOR`` is 1e-9, so **at its own threshold this rule admits
    about 1.4e7 band-widths**. It is a screen for the catastrophic case, not a
    bound on the error, and it must not be read as one.

    What keeps that from mattering here is measured, and the measurement is
    this file's own: instrumenting every one of the **117** calls the two
    dispatch test files make, the worst PASSING ratio is **1.3395** and the
    smallest REFUSING one is **2.3077**, with **no** passing row above 2.0. The
    threshold sits in that gap rather than on a cliff. Both sides are pinned by
    runs rather than by this paragraph:
    ``test_a_resolved_peak_still_certifies_and_says_so`` holds the passing side
    at 1.34 and
    ``test_a_grid_that_undersamples_by_between_two_and_four_is_still_refused``
    holds the refusing side at 3.00 and 2.31.

    〔Two numbers here were wrong until 2026-09-07, and the way they were wrong
    is the lesson: they were **copied from a review report rather than measured**
    -- 106 calls and a smallest-refusing ratio of 2.7955. The real figures are
    above, and the 2.3077 comes from a test added in the same commit that
    quoted 2.7955, so the commit refuted its own sentence. A borrowed number
    presented as a measurement is the defect this repository spends the most
    prose on, arriving through a door nobody had nailed shut.〕

    **Why not the stricter ``h <= width`` the probe this came from used?**
    ``undeclared_quartet`` has a closed-form Gaussian evidence computed by
    ``residual_models.gaussian_log_evidence`` -- pure numpy, one Cholesky,
    sharing no line with this module -- and the strict form refused a value
    that sits ``3.65e-10`` from it, inside its own band by 18.7x. That refusal
    was false. Its ``h/width = 1.34`` is itself inflated 6.6x, because the
    width reported here is the CONDITIONAL scale while aliasing depends on the
    MARGINAL: at ``h/marginal = 0.2031`` the quartet's true aliasing error is
    ``3.4e-208``. The threshold was calibrated against a ratio that overstates
    the risk, which is worth knowing before anyone moves it again.
    """
    location, widths, height, note = _ascend(
        log_density,
        names,
        starts,
        bounds=[(span.lower, span.upper) for span in spans],
    )
    if location is None or widths is None:
        return PeakResolution(False, None, None, None, note)
    complaint = _stationary(log_density, names, location, widths)
    if complaint is not None:
        return PeakResolution(False, None, None, height, complaint)
    for span in spans:
        width = widths[span.name]
        if not np.isfinite(width):
            continue
        spacing = (span.upper - span.lower) / (count - 1)
        if spacing / 2.0 > width:
            return PeakResolution(
                False,
                location,
                widths,
                height,
                f"the finest grid is spaced {spacing:.3e} on {span.name}, so the "
                f"peak can sit {spacing / 2.0:.3e} from the nearest point, and "
                f"the integrand's curvature width there is only {width:.3e}; a "
                "trapezoid that never samples the peak converges to the "
                "integral of everything except the mass",
            )
    return PeakResolution(
        True,
        location,
        widths,
        height,
        f"every axis has a grid point within one curvature width of the "
        f"highest point the ascent reached, {_at(location)}, height "
        f"{height:.6g}",
    )


def _at(values: Mapping[str, float]) -> str:
    """One-line coordinate, for a certificate sentence a reader has to act on."""
    return "{" + ", ".join(f"{name}={value:.6g}" for name, value in values.items()) + "}"


@dataclasses.dataclass(frozen=True, slots=True)
class Certificate:
    """Everything the quadrature knows about its own error, reported always.

    ``refused`` is the whole point: a certificate that cannot fail certifies
    nothing.  It carries the reason as a sentence, so a test asserting an
    ABSTAIN can assert WHICH abstain it got rather than merely that the value
    was ``None``.
    """

    history: tuple[tuple[int, float], ...]
    increment_ratio: float | None
    refinement_tail: float
    float_floor: float
    #: ``None`` means the edge test DID NOT RUN; ``inf`` means it ran and found
    #: an edge the mass is still growing towards; a float is the bound.
    #:
    #: **Three states in the field a consumer reads, which is what red line 14
    #: asks for.** It used to be two: the did-not-run case -- an integrand that
    #: is not a number, so there are no edges to measure -- reported ``inf``,
    #: the same value as a growing edge. They were distinguishable, but only by
    #: also consulting ``history == ()`` and ``edges == ()``, and requiring a
    #: reader to consult a second field to learn whether the first one means
    #: anything is exactly the indirection the rule exists to remove.
    truncation: float | None
    edges: tuple[EdgeDecay, ...]
    #: ``None`` means the peak test DID NOT RUN -- no grid was measured, so
    #: there is no finest spacing to compare a curvature width against. A
    #: :class:`PeakResolution` means it ran, and carries its own verdict.
    #: Three states in the field a consumer reads, as red line 14 asks, and for
    #: the same reason ``truncation`` has them one field up.
    peak: PeakResolution | None
    refused: str | None

    @property
    def bound(self) -> float:
        """This side's total error bound: refinement, arithmetic and truncation.

        ``inf`` when the edge test did not run: a bound that was never measured
        is not a bound of zero.
        """
        if self.truncation is None:
            return math.inf
        return self.refinement_tail + self.float_floor + self.truncation

    @property
    def increments(self) -> tuple[float, ...]:
        return tuple(
            self.history[index + 1][1] - self.history[index][1]
            for index in range(len(self.history) - 1)
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Quadrature:
    """A quadrature result. ``value is None`` means ABSTAIN, and says why."""

    value: float | None
    spans: tuple[Span, ...]
    certificate: Certificate
    excluded_prior_mass: Mapping[str, float | None]
    #: For :func:`oracle_collapsed` only: the largest ``|prior mean| / prior
    #: width`` the eliminated block takes anywhere on the declared span.
    #:
    #: **Recorded, and gating nothing.** The exact linear-Gaussian route carries
    #: two unbounded error laws -- measured, it is 0.28 nats out at
    #: ``|m| / s = 4e15`` -- and the R5 plan's 0.4 first asked this oracle to
    #: declare a covered range and abstain outside it. Measured, that remedy is
    #: wrong twice over: ``oracle_joint`` does NOT share the law (it reaches the
    #: model's own ``log_prob`` and never ``nuisance_prior``, so there is no
    #: ``m / s`` entry to cancel in a QR), so an abstain would DELETE a
    #: detection; and both corners the plan named sit at ``m = s``, where
    #: ``|m| / s`` is 1.0 either way, so a ceiling on it could not refuse the
    #: cells it was written about. 0.4 now says record and gate nothing, and
    #: this is the record. ``None`` where no block was eliminated.
    exact_block_ratio: float | None = None

    @property
    def certified(self) -> bool:
        return self.value is not None

    def describe(self) -> str:
        where = ", ".join(str(span) for span in self.spans)
        if self.value is None:
            return f"ABSTAIN over [{where}]: {self.certificate.refused}"
        return (
            f"{self.value!r} over [{where}], bound {self.certificate.bound:.3e} "
            f"(refinement {self.certificate.refinement_tail:.3e}, float "
            f"{self.certificate.float_floor:.3e}, truncation "
            f"{self.certificate.truncation:.3e}), excluded prior mass "
            f"{dict(self.excluded_prior_mass)}, eliminated block |m|/s "
            f"{self.exact_block_ratio!r}"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Agreement:
    """Two quadratures compared. ``gap`` is the quantity; the values are not."""

    left: Quadrature
    right: Quadrature
    gap: float | None
    band: float | None

    @property
    def agree(self) -> bool:
        return (
            self.gap is not None
            and self.band is not None
            and abs(self.gap) <= self.band
        )

    @property
    def margin(self) -> float:
        """How many times the band the gap is. Above 1 the comparison is red."""
        if self.gap is None or not self.band:
            return math.inf
        return abs(self.gap) / self.band

    def describe(self) -> str:
        if self.gap is None:
            return (
                "no comparison: "
                + ("left ABSTAINed; " if not self.left.certified else "")
                + ("right ABSTAINed; " if not self.right.certified else "")
                + f"left={self.left.describe()} right={self.right.describe()}"
            )
        return (
            f"gap {self.gap:+.6e} against band {self.band:.6e} "
            f"(margin {self.margin:.3e}); left {self.left.describe()}; "
            f"right {self.right.describe()}"
        )


def _require_x64() -> None:
    if jnp.zeros(()).dtype != jnp.float64:
        raise RuntimeError(
            "the residual oracle runs in double precision only; wrap the call "
            "in `with jax.enable_x64(True):`. A float32 trapezoid reaches its "
            "float floor four decades early and certifies a value that has "
            "not converged"
        )


def _evaluate(
    log_density: Callable[[dict[str, Any]], Any],
    names: Sequence[str],
    grids: Sequence[Any],
) -> np.ndarray:
    """``log_density`` on the full product grid, in memory-bounded chunks."""
    shape = tuple(int(grid.shape[0]) for grid in grids)
    total = int(np.prod(shape))
    stacked = jnp.stack(
        [jnp.asarray(grid, dtype=jnp.float64) for grid in grids]
    )  # (d, n)

    @jax.jit
    def block(indices):
        coordinates = []
        remainder = indices
        for size in reversed(shape):
            coordinates.append(remainder % size)
            remainder = remainder // size
        coordinates = coordinates[::-1]
        points = jnp.stack(
            [stacked[axis][coordinates[axis]] for axis in range(len(shape))], axis=1
        )
        return jax.vmap(
            lambda point: log_density(
                {name: point[axis] for axis, name in enumerate(names)}
            )
        )(points)

    pieces = [
        block(jnp.arange(start, min(start + _CHUNK, total)))
        for start in range(0, total, _CHUNK)
    ]
    return np.asarray(jnp.concatenate(pieces), dtype=float).reshape(shape)


def _log_trapezoid(
    density: np.ndarray, peak: float, axes: Sequence[np.ndarray]
) -> float:
    """``log`` of the trapezoid integral of ``exp(peak) * density``.

    Takes the peak-shifted density rather than the log values, because at the
    grids this oracle reaches -- ``6401**2`` on ``overflowing_outside_latent``
    -- one array of that shape is 328 MB and holding two of them at once was
    the whole memory cost.
    """
    if not math.isfinite(peak):
        return peak if peak == -math.inf else math.nan
    reduced = density
    for axis in range(len(axes) - 1, -1, -1):
        reduced = np.trapezoid(reduced, axes[axis], axis=axis)
    return peak + float(np.log(reduced))


def _marginal(density: np.ndarray, axes: Sequence[np.ndarray], keep: int) -> np.ndarray:
    """Integrate every axis except ``keep`` out of ``density``."""
    reduced = density
    for axis in range(len(axes) - 1, -1, -1):
        if axis == keep:
            continue
        reduced = np.trapezoid(reduced, axes[axis], axis=axis)
    return reduced


def _edges(
    density: np.ndarray, peak: float, axes: Sequence[np.ndarray], names: Sequence[str]
) -> tuple[EdgeDecay, ...]:
    """The decay of the integrand at both ends of every axis.

    Read off the per-axis MARGINAL rather than off a slice: a corner cell can be
    negligible while the edge of the axis as a whole is not, and it is the axis
    that the span is a statement about.
    """
    if not math.isfinite(peak):
        return tuple(
            EdgeDecay(name, side, math.inf, math.inf)
            for name in names
            for side in ("lower", "upper")
        )
    found: list[EdgeDecay] = []
    for axis, name in enumerate(names):
        marginal = np.asarray(_marginal(density, axes, axis), dtype=float)
        grid = np.asarray(axes[axis], dtype=float)
        cells = (marginal[1:] + marginal[:-1]) / 2.0 * np.diff(grid)
        centres = (grid[1:] + grid[:-1]) / 2.0
        width = float(grid[1] - grid[0])
        total = float(cells.sum())
        for side, (outer, inner) in (("lower", (0, 1)), ("upper", (-1, -2))):
            edge = float(cells[outer])
            neighbour = float(cells[inner])
            if edge <= 0.0:
                found.append(EdgeDecay(name, side, 0.0, 0.0))
                continue
            if neighbour <= 0.0 or total <= 0.0:
                found.append(EdgeDecay(name, side, math.inf, math.inf))
                continue
            ratio = edge / neighbour
            if ratio >= 1.0:
                found.append(EdgeDecay(name, side, ratio, math.inf))
                continue
            beyond = max(
                edge * ratio / (1.0 - ratio),
                _power_tail(edge, neighbour, centres[outer], centres[inner], width),
            )
            found.append(EdgeDecay(name, side, ratio, beyond / total))
    return tuple(found)


def _power_tail(
    edge: float, neighbour: float, at: float, before: float, width: float
) -> float:
    """The mass past the edge if the tail is a POWER of the distance, not a rate.

    **The geometric model is not a bound on a polynomial tail, and the shortfall
    is exactly ``p / (p - 1)``.** For a tail going as ``C t**-p`` the truth past
    ``S`` is ``C S**(1-p) / (p - 1)``; the outermost cell holds about
    ``C S**-p h``; and the local ratio is
    ``rho = ((S - h) / S)**p ~= 1 - p h / S``, so the geometric sum
    ``cell * rho / (1 - rho)`` comes out ``C S**(1-p) / p``. Measured on
    ``tests/exact/residual_models.py::cauchy_residual_pair``, whose integrand
    decays as ``z**-4`` and whose excluded mass is known in closed form: the
    ratio of the truth to the geometric bound reads 1.3333, 1.3334 and 1.3334 at
    spans of 100, 1000 and 2000 -- ``4 / 3`` to four digits, three times.

    That is not a wide miss and it is the wrong SIGN, which is what makes it
    worth the code: at ``|z| <= 2000`` the closed form sits 5.45e-11 from the
    oracle and the geometric bound reads 4.09e-11, so the certificate excluded
    the true answer. A bound that excludes the truth is not a bound.

    So the exponent is read off the same two cells --
    ``p = log(neighbour / edge) / log(|at| / |before|)`` -- and the power-law
    tail ``cell * |S| / (h (p - 1))`` is returned when it is the larger of the
    two.

    **It does not collapse on an exponentially cut tail, which is what a first
    draft of this paragraph claimed.** Measured on
    ``overflowing_outside_latent`` at ``|z| <= 60``: the fitted exponent is about
    15, the power-law term reads 9 per cent ABOVE the geometric one, and the
    reported bound moves from 1.485e-03 to 1.608e-03 against an actual error of
    1.268e-03. Still a bound, and a looser one -- the ``max`` buys soundness on
    the ``z**-4`` tail at the price of about 8 per cent of tightness on the
    exponential one, which is the trade this function is, stated rather than
    implied.

    Returns ``0.0`` -- deferring to the geometric bound -- when the fit has no
    meaning: an edge closer to the origin than its neighbour is an INNER edge
    and a power of the distance is not the tail there, and ``p <= 1`` is a tail
    with no finite mass at all, which the refinement will refuse on its own.
    """
    here, prior = abs(at), abs(before)
    if not (here > prior > 0.0) or width <= 0.0:
        return 0.0
    exponent = math.log(neighbour / edge) / math.log(here / prior)
    if not (exponent > 1.0) or not math.isfinite(exponent):
        return 0.0
    return edge * here / (width * (exponent - 1.0))


def _geometric_tail(increment: float, ratio: float) -> float:
    """What a geometric series of shrinking increments has left to give.

    ``increment * ratio / (1 - ratio)`` is the sum of everything after the last
    refinement, and it is a bound only while ``ratio < 1``. **Written as its own
    function because the inline form had a hole an adversarial review walked
    through**: relaxing the test to ``ratio < 2.0`` makes the expression
    NEGATIVE for a ratio in ``(1, 2)``, and a negative tail is finite and
    compares less than any demanded level, so a refinement that is DIVERGING
    certifies. The guard was in the right place and was doing two jobs at once
    -- deciding convergence and keeping the arithmetic meaningful -- and only
    one of them was pinned.

    So the sign is checked here rather than inferred from the ratio, and a tail
    that comes out negative is `inf`: not a bound, and never a pass.
    """
    if not ratio < 1.0:
        return math.inf
    tail = increment * ratio / (1.0 - ratio)
    return tail if tail >= 0.0 else math.inf


def excluded_prior_mass(graph: Graph, spans: Sequence[Span]) -> dict[str, float | None]:
    """How much declared prior mass each span leaves out.

    ``None`` where the question has no one-dimensional answer: a latent whose
    parents are latent has no marginal prior to integrate -- that is precisely
    what the R5 plan's 0.15 says makes a graph class (b) -- and a distribution
    without a ``cdf`` cannot be asked.  ``None`` is reported rather than a zero,
    because "no mass excluded" and "not asked" are the distinction this
    repository has paid for most often.
    """
    _require_x64()
    environment = prior_environment(graph)
    latents = frozenset(graph.latents)
    out: dict[str, float | None] = {}
    for span in spans:
        if _ancestors(graph, span.name) & latents:
            out[span.name] = None
            continue
        try:
            distribution = apply_probabilistic(
                graph, graph.node(span.name), dict(environment)
            )
            below = float(distribution.cdf(jnp.asarray(span.lower)))
            above = 1.0 - float(distribution.cdf(jnp.asarray(span.upper)))
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            out[span.name] = None
            continue
        out[span.name] = max(0.0, below) + max(0.0, above)
    return out


def _ancestors(graph: Graph, name: str) -> frozenset[str]:
    """Every node reachable upstream of ``name``, deduped."""
    seen: set[str] = set()
    stack = list(graph.node(name).parents)
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(graph.node(current).parents)
    return frozenset(seen)


def probe_points(spans: Sequence[Span]) -> list[dict[str, float]]:
    """Where :func:`block_prior_ratio` looks: each axis's two ends and its centre.

    ``2d + 1`` points, linear in the dimension. **Its own function because the
    inline version was ungraded in a way one fixture could not show.** The only
    fixture whose ratio varies is ``shifted_block_prior``, whose ``|tau| / 0.3``
    is monotone increasing, so its maximum sits exactly at the span's upper end
    -- and a probe that read the upper end and nothing else satisfied the test
    written to pin this. Two adversarial mutants lived there: "probe only the
    centre" and "probe only the upper end". A worst point in the INTERIOR is
    what separates them, and no graph in this package has one.
    """
    centre = {span.name: (span.lower + span.upper) / 2.0 for span in spans}
    points = [dict(centre)]
    for span in spans:
        for edge in (span.lower, span.upper):
            points.append({**centre, span.name: edge})
    return points


def worst_ratio(prior_mean: Any, prior_std: Any) -> float:
    """``max |mean| / min width`` over one block member's declared prior.

    Both reductions run over an ARRAY, and in every fixture this package ships
    that array has shape ``()``. Over one element ``max``, ``min``, ``first``
    and ``sum`` are the same function, so an adversarial review swapped the two
    reductions and swapped them again with each other and nothing moved. The
    reductions are the conservative choice -- the largest mean against the
    smallest width is the worst cell the member can be in -- and they are graded
    here directly rather than through a graph, because building a plated
    exact block to grade two calls to numpy is the wrong instrument.

    ``|mean|`` and not ``mean``: a negative prior mean is as far from zero as a
    positive one, and dropping the ``abs`` also survived, since the mean is
    non-negative at every probe point of every span this package can build.
    """
    mean = float(np.max(np.abs(np.asarray(prior_mean, dtype=float))))
    width = float(np.min(np.abs(np.asarray(prior_std, dtype=float))))
    return math.inf if width == 0.0 else mean / width


def block_prior_ratio(
    graph: Graph, exact_names: Sequence[str], spans: Sequence[Span]
) -> float | None:
    """The largest ``|prior mean| / prior width`` the eliminated block reaches.

    Probed ACROSS the span rather than at one point, because the eliminated
    block's prior is conditional on the residual parameters and moves with them
    -- ``shared_ancestor`` declares ``x ~ N(0, |tau|)``, so its width is a
    function of the very axis being integrated. Each axis is walked at its two
    ends and its centre with the others held at centre, which is linear in the
    dimension and enough to catch a ratio that runs away toward an edge.

    Returns ``None`` when the block cannot be built at those points, which is a
    fact about the span and is reported rather than raised: a span whose
    interior holds a degenerate conditional is exactly where this cannot be
    asked, and :func:`quadrature` will refuse it for its own reasons.
    """
    from bayesmith.exact.block import unchecked_operator

    _require_x64()
    names = tuple(exact_names)
    if not names:
        return None
    points = probe_points(spans)

    def built(at):
        """The block at one probe point, or ``None`` if it cannot be built there.

        Separated from the loop so the failure is a VALUE rather than an
        ``except: continue``. This repository has paid for that difference more
        than once -- a swallowed exception reports "nothing to see" and "never
        looked" as the same silence -- and the loop below counts the points it
        could not reach.
        """
        try:
            return unchecked_operator(graph, names, at=at, probe_gaussian=False)
        except Exception:  # noqa: BLE001 - any failure here is "not askable"
            return None

    worst: float | None = None
    unreachable = 0
    for at in points:
        block = built(at)
        if block is None:
            unreachable += 1
            continue
        for member in block.names:
            ratio = worst_ratio(block.prior_mean[member], block.prior_std[member])
            worst = ratio if worst is None else max(worst, ratio)
    if worst is None and unreachable:
        return None
    return worst


def quadrature(
    log_density: Callable[[dict[str, Any]], Any],
    spans: Sequence[Span],
    *,
    resolution: float,
    start: int = 201,
    refinements: int = 8,
    graph: Graph | None = None,
    max_points: int = MAX_POINTS,
) -> Quadrature:
    """Trapezoid over ``spans``, refined until it certifies or abstains.

    ``start`` is the point count of the first grid and each refinement doubles
    the intervals, ``n -> 2n - 1``, so every earlier grid is a subset of every
    later one and the movement between them is the discretisation error alone.

    ``resolution`` is relative: the refinement stops when the geometric tail of
    the increments is at or below ``resolution * max(1, |value|)``, and ABSTAINs
    if ``refinements`` grids do not get there. It has no default. A caller that
    could omit it would be choosing the band by choosing how long to run, which
    is the failure mode the module docstring's condition 2 describes.
    """
    _require_x64()
    if not 0.0 < resolution <= AGREEMENT_FLOOR:
        raise ValueError(
            f"resolution={resolution!r}; the refinement's level must be "
            f"positive and no looser than the declared AGREEMENT_FLOOR "
            f"({AGREEMENT_FLOOR:g}). A caller may demand MORE convergence than "
            "D111 declares and may not demand less, because the band is "
            "floored at D111 either way and a looser refinement only buys a "
            "certificate that means nothing.\n\n"
            "Two adversarial reviews walked through this guard in turn. The "
            "first found `not resolution > 0.0` admitting `inf`, which made "
            "`tail <= inf` vacuous at the first grid and certified a value "
            "2.5e-04 wrong. The second found the repair -- `0 < resolution < 1` "
            "-- admitting 0.999, which certifies `cauchy_residual_pair` "
            "**0.873 nats** wrong while reporting a refinement tail of 0.36: "
            "3500 times worse than the hole it closed. Nothing in the suite "
            "passed any level but the declared one, so the whole admitted "
            "range was unexercised both times. A bound is now the declared "
            "level itself, which leaves no range to sweep"
        )
    if start ** len(spans) > max_points:
        raise ValueError(
            f"the first grid is {start}**{len(spans)} = {start ** len(spans):.3e} "
            f"points, above the {max_points:.3e} budget. Pass a smaller `start` "
            "for this many axes, or a larger `max_points` and the memory to pay "
            "for it. Refusing here rather than allocating: a first grid too "
            "large to build is a hang, and a hang says nothing about the model"
        )
    names = tuple(span.name for span in spans)
    if len(set(names)) != len(names):
        raise ValueError(
            f"{names} names an axis twice; a product grid needs distinct axes"
        )
    if start < 3:
        raise ValueError(
            f"start={start}; a grid needs at least three points, because the "
            "edge test reads the outermost trapezoid cell AND its neighbour. "
            "Below three it raised IndexError from inside `_edges`, which "
            "names neither the argument nor the fix"
        )
    history: list[tuple[int, float]] = []
    edges: tuple[EdgeDecay, ...] = ()
    count = start
    ratio: float | None = None
    tail = math.inf
    floor = math.inf
    demanded = math.inf
    #: ``None`` until the first grid is measured, so a refinement budget of zero
    #: reports "did not run" rather than "ran and found an unbounded edge".
    truncation: float | None = None
    #: ``None`` until the first grid is measured, so a refinement budget of zero
    #: reports a peak test that did not run rather than one that passed.
    grid_argmax: dict[str, float] | None = None
    reasons: list[str] = []
    for _ in range(refinements):
        axes = [
            np.linspace(span.lower, span.upper, count, dtype=float) for span in spans
        ]
        log_values = _evaluate(log_density, names, [jnp.asarray(axis) for axis in axes])
        unusable = int(np.count_nonzero(np.isnan(log_values) | (log_values == np.inf)))
        if unusable:
            where = np.argwhere(np.isnan(log_values) | (log_values == np.inf))[0]
            at = ", ".join(
                f"{name}={axes[axis][index]!r}"
                for axis, (name, index) in enumerate(zip(names, where, strict=True))
            )
            return Quadrature(
                value=None,
                spans=tuple(spans),
                certificate=Certificate(
                    history=tuple(history),
                    increment_ratio=None,
                    refinement_tail=math.inf,
                    float_floor=math.inf,
                    truncation=None,
                    edges=(),
                    peak=None,
                    refused=(
                        f"the integrand is not a number at {unusable} of "
                        f"{log_values.size} grid points on the n={count} grid, "
                        f"the first at {at}; a span whose interior holds a "
                        "degenerate conditional is not a span this quadrature "
                        "can be asked about"
                    ),
                ),
                excluded_prior_mass=(
                    excluded_prior_mass(graph, spans) if graph is not None else {}
                ),
            )
        peak = float(np.max(log_values))
        # The grid's own best point, kept before the array is freed. It is the
        # start the peak condition ascends from, and it costs nothing: the
        # values are already computed, and whatever the grid found is within one
        # spacing of the best point the grid can see.
        best_cell = np.unravel_index(int(np.argmax(log_values)), log_values.shape)
        grid_argmax = {
            name: float(axes[axis][best_cell[axis]])
            for axis, name in enumerate(names)
        }
        density = np.exp(log_values - peak) if math.isfinite(peak) else log_values
        del log_values
        value = _log_trapezoid(density, peak, axes)
        history.append((count, value))
        edges = _edges(density, peak, axes, names)
        del density
        truncation = sum(edge.fraction for edge in edges)
        cells = int(np.prod([len(axis) for axis in axes]))
        scale = max(1.0, abs(value)) if math.isfinite(value) else math.inf
        floor = math.ceil(math.log2(cells)) * EPS * scale
        demanded = resolution * scale
        if len(history) >= 3:
            previous = history[-2][1] - history[-3][1]
            latest = history[-1][1] - history[-2][1]
            if abs(latest) <= floor:
                # The value has stopped moving as far as this arithmetic can
                # tell. Demanding more is demanding a movement smaller than the
                # sum's own representation, and what that produces is not
                # convergence but DITHER: measured on the mixture fixture, the
                # increments run 0.000e+00, +1.776e-15, -1.776e-15, whose ratio
                # is exactly 1.0 -- indistinguishable, to a rule that only looks
                # at ratios, from a quadrature drifting by a constant forever.
                ratio, tail = 0.0, abs(latest)
            elif previous == 0.0:
                ratio, tail = math.inf, math.inf
            else:
                ratio = abs(latest) / abs(previous)
                tail = _geometric_tail(abs(latest), ratio)
            # No `truncation` conjunct here, and it is not an oversight twice
            # over. The post-loop reason list re-tests exactly that, so an
            # infinite truncation abstains either way and the conjunct only
            # bought extra refinements before the same answer -- an adversarial
            # review scored it SURVIVED for that reason and was right, since a
            # condition with no consequence is not a condition. And `truncation`
            # cannot be `None` at this point: the loop body assigns it from the
            # edges before reaching here, and the one path that leaves it `None`
            # returns before the loop. Adding a `None` guard here would be
            # unreachable code guarding an unreachable state.
            if math.isfinite(tail) and tail <= demanded:
                break
        if (2 * count - 1) ** len(spans) > max_points:
            over = (2 * count - 1) ** len(spans)
            reasons.append(
                f"the next refinement would need {2 * count - 1}**{len(spans)} = "
                f"{over:.3e} points, above the {max_points:.3e} budget, and the "
                f"n={count} grid has not certified"
            )
            break
        count = 2 * count - 1
    if truncation is None or not math.isfinite(truncation):
        growing = [str(edge) for edge in edges if not edge.decaying]
        reasons.append(
            "the integrand is not decaying at "
            + ", ".join(growing)
            + " -- the mass is outside this span, so the value integrates the "
            "wrong region"
        )
    if not (math.isfinite(tail) and tail <= demanded):
        steps = [
            f"{history[index + 1][1] - history[index][1]:.3e}"
            for index in range(len(history) - 1)
        ]
        reasons.append(
            f"the value has not reached the demanded resolution: after "
            f"{len(history)} grids up to n={history[-1][0]} the increments are "
            f"{steps} (ratio {ratio!r}), whose geometric tail is {tail:.3e} "
            f"against a demanded {demanded:.3e}"
        )
    # **The one condition that is not about the refinement sequence**, and it
    # runs LAST because it needs the finest grid's spacing. The three above ask
    # whether the trapezoid is consistent with itself; a grid that steps over a
    # narrow peak is perfectly consistent with itself and wrong by 2.4 million
    # nats, so consistency was never going to catch it. Starts: the grid's own
    # argmax always, plus prior draws when a graph is available, because a
    # single start finds one mode of a multimodal integrand and reports its
    # width as the integrand's.
    peak_check: PeakResolution | None = None
    if grid_argmax is not None and history:
        starts = [grid_argmax, *_prior_starts(graph, names)]
        peak_check = _peak_resolution(
            log_density, spans, names, starts, history[-1][0]
        )
        if not peak_check.resolved:
            reasons.append(f"the grid does not resolve the peak: {peak_check.note}")
    certificate = Certificate(
        history=tuple(history),
        increment_ratio=ratio,
        refinement_tail=tail,
        float_floor=floor,
        truncation=truncation,
        edges=edges,
        peak=peak_check,
        refused="; ".join(reasons) if reasons else None,
    )
    return Quadrature(
        value=None if reasons else history[-1][1],
        spans=tuple(spans),
        certificate=certificate,
        excluded_prior_mass=(
            excluded_prior_mass(graph, spans) if graph is not None else {}
        ),
    )


def oracle_joint(graph: Graph, spans: Sequence[Span], **options: Any) -> Quadrature:
    """Tier 1. ``log_joint`` over EVERY latent -- residual and exact together.

    Refuses a subset, because integrating some of the latents and calling the
    answer an evidence is the error this whole module is arranged to prevent,
    and it would read as a perfectly ordinary call.
    """
    names = tuple(span.name for span in spans)
    if set(names) != set(graph.latents):
        raise ValueError(
            f"oracle_joint needs a span for every latent; got {sorted(names)} "
            f"for latents {sorted(graph.latents)}. Integrating a SUBSET yields "
            "a conditional, not an evidence, and nothing downstream of here "
            "could tell the difference"
        )
    return quadrature(
        lambda values: log_joint(graph, dict(values)), spans, graph=graph, **options
    )


def oracle_collapsed(
    graph: Graph,
    exact_names: Iterable[str],
    spans: Sequence[Span],
    **options: Any,
) -> Quadrature:
    """Tier 5. The REDUCED graph's ``log_joint`` over the residual latents.

    Self-consistency with respect to the elimination: it calls
    ``collapse_graph`` and ``marginal_log_density``, which are the production
    path.  It grades a sampler; it never grades the collapse.
    """
    exact = tuple(exact_names)
    residual = tuple(name for name in graph.latents if name not in frozenset(exact))
    names = tuple(span.name for span in spans)
    if set(names) != set(residual):
        raise ValueError(
            f"oracle_collapsed needs a span for every residual latent; got "
            f"{sorted(names)} for residual {sorted(residual)}"
        )
    reduced = collapse_graph(graph, exact, residual)
    found = quadrature(
        lambda values: log_joint(reduced, dict(values)), spans, graph=graph, **options
    )
    return dataclasses.replace(
        found, exact_block_ratio=block_prior_ratio(graph, exact, spans)
    )


def agreement(left: Quadrature, right: Quadrature) -> Agreement:
    """Compare two quadratures within the sum of what each says about itself."""
    if not (left.certified and right.certified):
        return Agreement(left=left, right=right, gap=None, band=None)
    scale = max(1.0, abs(left.value), abs(right.value))
    return Agreement(
        left=left,
        right=right,
        gap=left.value - right.value,
        band=max(
            left.certificate.bound + right.certificate.bound,
            AGREEMENT_FLOOR * scale,
        ),
    )


__all__ = [
    "AGREEMENT_FLOOR",
    "Agreement",
    "Certificate",
    "EdgeDecay",
    "PeakResolution",
    "Quadrature",
    "Span",
    "agreement",
    "block_prior_ratio",
    "probe_points",
    "excluded_prior_mass",
    "oracle_collapsed",
    "oracle_joint",
    "quadrature",
    "worst_ratio",
]
