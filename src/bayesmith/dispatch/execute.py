"""Running what :func:`~bayesmith.dispatch.plan.compile` decided.

:mod:`bayesmith.dispatch.plan` derives the partition, measures its
conditioning and prints it; nothing in that module draws a sample. This one
is the other half: given an :class:`~bayesmith.dispatch.plan.InferencePlan`,
it runs section 6.4's five shapes and section 6.5's two, reading ``method``,
``tol`` and ``sigma_needs_rebuild`` off the plan rather than re-deriving them,
so that what runs is what ``str(plan)`` printed.

The two entry points are :func:`run_sample` and :func:`run_estimate`.
:meth:`~bayesmith.dispatch.plan.InferencePlan.sample` and
:meth:`~bayesmith.dispatch.plan.InferencePlan.estimate` are the public spelling
of them and are where their defaults and their documentation live; the methods
delegate here and add nothing.

**The dependency runs one way only.** ``plan`` imports this module at module
scope; this module reaches ``InferencePlan`` for its annotations alone, under
``TYPE_CHECKING``, so there is no import cycle to break at runtime.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import numpy as np
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

from bayesmith.bridge.numpyro_bridge import nuts as nuts_draws
from bayesmith.dispatch.classify import block_at
from bayesmith.errors import ConvergenceError
from bayesmith.exact.block import domain_centre, unchecked_operator
from bayesmith.exact.correct import khat, log_weight, self_normalise, unreliable
from bayesmith.exact.gibbs import assemble
from bayesmith.exact.gls import iterative_gls, precision_from_graph
from bayesmith.exact.precision import per_sample_sigma
from bayesmith.exact.solve import gcr_sample
from bayesmith.graph.graph import Graph

if TYPE_CHECKING:  # annotations only -- see the module docstring
    from bayesmith.dispatch.plan import InferencePlan


SNIS_ESS_FLOOR: float = 0.1
"""Kish ESS/N below which the SNIS correction is declared collapsed.

Section 5.2's decision, and the reason it is a floor on the RATIO rather than
a flag on the object: self-normalised importance sampling degrades
**exponentially** in the number of mismatched coordinates and reports nothing
while it does. :func:`~bayesmith.exact.correct.self_normalise`'s own docstring
carries the reference measurement (n=1 -> ESS 2509, n=500 -> 1.00, at N=40000
throughout); sweeping only the plate of this package's ``plated_radiometer``
at N=2000 reproduces the shape -- 0.966 (n=1), 0.739 (n=6), 0.445 (n=12),
0.123 (n=50), 0.054 (n=100), 0.0039 (n=400).

**Crossing it ANNOTATES the sample; it does not replace it.** This floor used
to discard the weighted draws and run NUTS instead, and that was measured to
be the wrong way round. At the genuinely collapsed cell
``plated_radiometer(n=25, kappa=0.4)`` at N=1200, against exact
per-coordinate quadrature of that fixture's own factorised plate, worst
coordinate in units of its posterior sd: the SNIS answer that was being
discarded is out by **1.40** (keys 0 and 4: 1.398, 1.376) and the NUTS that
replaced it by **18.5** (18.507, 18.471) -- a factor of 13, in favour of the
estimator the floor threw away. Worse, the floor's own currency inverts the
ordering: NUTS's chain ESS on those two keys reads 33.3 and 51.3 against the
SNIS Kish ESS of 14.1 and 13.5, so the diagnostic that fires prefers the
answer 13x further from the truth. A floor that fires on a number cannot also
be trusted to choose the replacement by that number.

So what crossing it now produces is a :class:`Posterior` carrying its draws,
its weights, its k-hat, ``unreliable=True`` and the collapse named in
``reason``. Substituting NUTS is :meth:`InferencePlan.sample`'s
``nuts_on_collapse=True``, and it is opt-in rather than opt-out because the
caller who did not ask for it is exactly the caller who cannot tell it
happened.

**0.1 is the variance-inflation red line, not a crossover.** At ESS/N = 0.1
the estimator's variance is inflated 10x, i.e. the error bars are 3.16x wider
than the nominal ones and recovering a nominal bar costs ten times the draws.
It would be dishonest to claim either estimator is uniformly better: measured
at N=1000 against NUTS's own min-over-coordinates ESS/N on the same graph,
``plated_radiometer(n=6, kappa=0.4)`` reads SNIS 0.120 against NUTS 0.014 (the
weights win, 8x), while ``n=12, kappa=0.2`` reads SNIS 0.0065 against NUTS
0.332 (NUTS wins, 51x) and ``n=50, kappa=0.2`` reads 0.014 against 0.241. So
the crossover lies somewhere in 0.01-0.12 and is not sharp; 0.1 is its
conservative end. What the floor guarantees is not the better estimator, it is
that a ``Posterior`` carrying a thousand draws never comes back with four of
them real and only a boolean to say so -- and it now guarantees that by
setting the boolean, which is a claim it can actually support.

**It is read off the Kish ESS and never off k-hat**, which is not
interchangeable with it here. Measured on ``radiometer()`` -- a
ONE-dimensional block -- at N=1200 over six keys: Kish ESS/N is 0.9986-0.9988
every time while k-hat reads 0.91 to 1.80, past
:data:`~bayesmith.exact.correct.FINITE_MEAN_KHAT`. The log weights span under
one nat there, which is not a tail PSIS has any business fitting a
generalised Pareto to. Dispatch on k-hat and that graph is reported as a
collapse that did not happen.
"""


class Posterior(NamedTuple):
    """What :meth:`InferencePlan.sample` returns, on every path.

    Attributes:
        samples: ``{latent: draws}``, the leading axis being the draw. Only
            latents -- numpyro's ``get_samples`` also returns Deterministic
            sites (measured: ``mixed_radiometer``'s ``mu``, shape (400, 10)),
            and leaving them in would silently change ``ess``.
        log_weights: unnormalised SNIS log weights, or ``None`` on every path
            that produced an unweighted sample.
        ess: effective sample size, reduced by **MIN** over every site and
            every coordinate -- see :func:`chain_ess`. The Kish ESS on the
            SNIS path, and ``num_samples`` exactly on the iid one.
        khat: PSIS k-hat of ``log_weights``, or ``None`` where there are none.
        unreliable: k-hat past ``min(1 - 1/log10(N), 0.7)``, **or** a Kish
            ESS/N under ``ess_floor`` -- either is a reason not to trust the
            weights, and the field is the one place a caller looks for that.
            ``False`` wherever ``khat`` is ``None`` and nothing collapsed,
            which is **abstention, not endorsement** -- read ``ess``, which is
            always there.
        method: what actually RAN: ``"gcr"``, ``"gcr+snis"``, ``"gcr+mh"`` or
            ``"nuts"``. Not necessarily ``plan.exact.method``: a collapsed
            SNIS reports ``"nuts"`` when ``nuts_on_collapse=True`` asked for
            it, because that is what produced the draws, and ``"gcr+snis"``
            otherwise -- see :data:`SNIS_ESS_FLOOR` for why the default is
            to annotate rather than substitute.
        reason: why that, in the plan's own words, including the measured
            Kish ESS/N wherever the floor fired.
        diagnostics: per-site split r-hat and ESS -- see
            :func:`chain_diagnostics` -- on the paths that ran a CHAIN, and
            ``None`` on the paths that did not. ``None`` is abstention rather
            than endorsement, and here it is a stronger statement than usual:
            the ``gcr`` and ``gcr+snis`` paths draw independently, so r-hat
            has no referent at all on them. Reporting a number would invent
            one. Read :attr:`ess`, which is always there.
    """

    samples: dict[str, jax.Array]
    log_weights: jax.Array | None
    ess: float
    khat: float | None
    unreliable: bool
    method: str
    reason: str
    diagnostics: Mapping[str, SiteDiagnostic] | None = None


class Estimate(NamedTuple):
    """What :meth:`InferencePlan.estimate` returns: a point, and its covariance.

    Attributes:
        values: the GLS/Wiener solution, ``{latent: value}``.
        precision: the covariance it was solved at, as ``{observed: N^-1}``
            -- for a prediction-dependent model, the fixed point. This is what
            ``gcr_sample`` and ``log_weight`` read.
        converged: always ``True``. ``False`` is not returned; it is raised as
            :class:`~bayesmith.errors.ConvergenceError`, which is the whole
            point of this being the promotion site. Kept as a field because a
            caller reading a stored ``Estimate`` should not have to know that.
        residual: relative CG residual of the final solve. Not an accuracy --
            multiply by ``plan.exact.kappa`` for the error bound.
        iterations: reweighting steps taken, ``1`` for a constant sigma.
    """

    values: dict[str, jax.Array]
    precision: dict[str, Any]
    converged: bool
    residual: jax.Array
    iterations: jax.Array

    @property
    def noise_std(self) -> dict[str, jax.Array] | None:
        """:attr:`precision` as per-sample sigma VALUES, or ``None``.

        The same shape, and for the same reason, as
        :attr:`~bayesmith.exact.gls.GLSResult.noise_std`: one covariance is
        stored, and the spelling that does not always exist is the derived
        one. ``None`` for a correlated model, which has no per-sample sigma;
        for a diagonal one these are the arrays themselves, so
        ``noise_std_at(graph, values)`` reproduces them.
        """
        return per_sample_sigma(self.precision)


def chain_ess(samples: Mapping[str, Any], *, num_chains: int = 1) -> float:
    """MIN of numpyro's ESS over every site and every coordinate.

    **The reduction is MIN and that has to be written down.** Section 6.3's
    benchmark C has ``ESS(logw)=3.0`` and ``ESS(alm, min)=40.2`` in one run;
    :attr:`Posterior.ess` exists to make dividing by N a deliberate act, so it
    must report the worst of them. A mean lets a well-mixing site hide a stuck
    one, which is the exact shape of the bug the field is against.

    **A non-finite per-coordinate ESS becomes 1.0, not dropped.** A chain that
    never moved makes ``effective_sample_size`` divide 0 by 0 and return
    ``nan``, and ``nan`` is the one value ``min`` steps over silently --
    ``min(nan, 150.2)`` is either, depending on argument order. Measured:
    ``mixed_radiometer`` at key 2 leaves ``w`` bitwise constant over 400
    sweeps and reads exactly that. One draw's worth of information is the
    smallest a non-empty sample can carry, so 1.0 is both the honest value and
    the one that wins the ``min``.

    Args:
        samples: ``{site: draws}`` with the draw axis LEADING and chains
            already concatenated along it, which is what
            ``MCMC.get_samples()`` returns.
        num_chains: how many chains that axis holds, so it can be unstacked
            before the estimator sees it. Left at 1 the array is used as is.
    """
    worst = math.inf
    for draws in samples.values():
        values = np.asarray(draws)
        grouped = values.reshape((num_chains, -1, *values.shape[1:]))
        measured = np.asarray(effective_sample_size(grouped), dtype=float)
        worst = min(worst, float(np.where(np.isfinite(measured), measured, 1.0).min()))
    return worst


CHAIN_ESS_FLOOR: float = 100.0
"""Per-coordinate ESS below which convergence cannot be CERTIFIED at all.

Not the same object as :data:`SNIS_ESS_FLOOR`, which is a floor on a ratio for
importance weights. This is an absolute count, and it exists because of a
measurement that overturned the design it was added to.

**The r-hat ceiling has no power on its own, and the failure is systematic.**
``1 + C/ESS`` is calibrated under the null, where a low ESS means a chain that
mixes slowly while sampling the target. Under the alternative a low ESS means
a chain that is STUCK -- and a displaced chain drives ESS down faster than it
drives r-hat up, so the ceiling rises to meet the very statistic it should be
rejecting. Measured on two chains of 200 draws displaced by a separation, at
2000 coordinates, AR(1) rho in {0, 0.9, 0.99}: **all fifteen cells sneak
through**. At separation 8 and rho 0.99 the chains are visibly in different
places, r-hat is **9.25** -- and ESS is **1.02**, so the ceiling is **12.3**
and forgives it. The worse the failure, the more the ceiling forgives.

So the order matters and is not interchangeable: **ESS is gated first, and
r-hat is only asked where ESS can support an answer.** Every one of those
fifteen cells has ESS at most 19.9, so the floor catches all of them, and a
coordinate at ESS 100 has a ceiling of 1.115 -- a bound tight enough to be a
test. A converged chain below the floor is not slandered by this: it is
genuinely under-sampled, and "you do not have enough draws" is the true
statement about it.

100 is the field's usual recommendation (Vehtari et al. 2021) and is adopted
rather than invented; what is measured here is that it is sufficient for the
failures above, which is the part that could have been wrong.
"""

#: ``C`` in ``r_hat_ceiling``, at one coordinate: the measured maximum of
#: ``(r_hat_99 - 1) * ESS`` across 27 null cells, 11.46, rounded up.
_R_HAT_C1 = 11.5

#: How ``C`` grows per decade of coordinates, and where it stops. Measured
#: maxima: 11.46 at P=1, 14.44 at P=10, 18.76 at P=100, 22.61 at P=1000,
#: 22.99 at P=10000 -- a little under 4 per decade, saturating near 23.
_R_HAT_PER_DECADE = 3.9
_R_HAT_C_CAP = 23.0


def _constant(coordinates: int) -> float:
    """``C`` alone, so the scalar ceiling and the array path cannot disagree.

    Spelled once. Two copies of one formula is the defect this package has
    spent the most time repairing, and a constant is no exception to it.
    """
    return min(
        _R_HAT_C1 + _R_HAT_PER_DECADE * math.log10(max(coordinates, 1)),
        _R_HAT_C_CAP,
    )


class SiteDiagnostic(NamedTuple):
    """One site's convergence, judged at its worst coordinate.

    Attributes:
        r_hat: split r-hat at the coordinate that decided the verdict, or
            ``inf`` where numpyro returned a non-finite value. See
            :func:`chain_diagnostics` for why ``inf`` and not ``nan``.
        ess: effective sample size at that same coordinate. The pair is
            reported together because neither means anything alone -- the
            ceiling r-hat is judged against is a function of this number.
        ceiling: ``r_hat_ceiling(ess, total coordinates in the report)``.
        converged: whether ``r_hat`` is within ``ceiling``.
        worst: index of the deciding coordinate, ``()`` for a scalar site.
            The attribution is the point: the pre-existing scalar reported the
            worst number in the whole posterior and never said where it came
            from, and "which element of a 64-element site" is a different
            morning's work each time.
        reason: empty where ``converged``, and otherwise which of the two
            checks failed. They are different problems with different
            remedies -- too little information is answered by sampling
            longer, chains that disagree by finding out why -- and a single
            boolean cannot tell a caller which one they have.
    """

    r_hat: float
    ess: float
    ceiling: float
    converged: bool
    worst: tuple[int, ...]
    reason: str


def r_hat_ceiling(ess: float, *, coordinates: int = 1) -> float:
    """The largest split r-hat a CONVERGED chain of this ESS plausibly shows.

    **A constant threshold on r-hat is not a well-posed test**, which is the
    finding that shaped this function, and it is stronger than the migration
    spec's warning that the per-parameter form needs its own argument.
    Split r-hat's null distribution is governed by how much independent
    information the chain holds, so the same constant is a coin flip at one
    end of the ESS range and a no-op at the other.

    Measured with 4000 independent coordinates per cell over 27 cells --
    chains in {1, 2, 4}, draws in {200, 800, 3200}, AR(1) rho in
    {0, 0.5, 0.9}, so ESS from 14.8 to 12693:

    * ``r_hat_99 - 1`` scales as ``ESS ** -1.05``: a straight ``1 / ESS``.
    * ``(r_hat_99 - 1) * ESS`` stays within 4.87 to 11.46 while ESS moves by a
      factor of 860. That product is the stable quantity, so it is the one
      pinned, and the ceiling is derived from it.
    * A fixed **1.05** fires on **45.9 %** of converged chains at ESS 15, and
      **0.0 %** at ESS 12693. 1.01 fires on 69.9 % and 0.0 %. Neither constant
      is a test at both ends.

    To reproduce: draw ``(chains, draws, 4000)`` standard normals, optionally
    filtered to AR(1) with ``x[t] = rho x[t-1] + sqrt(1-rho^2) e[t]``, and take
    ``split_gelman_rubin`` and ``effective_sample_size`` over the coordinate
    axis. Coordinates are independent, so the family-wise quantile over ``P``
    of them is the per-coordinate quantile at ``0.99 ** (1/P)`` -- exact, and
    far cheaper than simulating each ``P``.

    Args:
        ess: the effective sample size AT the coordinate being judged, not a
            summary over the site. Mixing coordinates here would compare one
            coordinate's r-hat against another's information.
        coordinates: how many coordinates the report covers in total. Reporting
            the worst of ``P`` inflates the null, so the ceiling rises with
            ``P`` -- otherwise the alarm rate grows with the size of the model
            rather than with its health. Measured to saturate near 23, and
            capped there: an unbounded term would make a large model
            unfalsifiable.
    """
    return 1.0 + _constant(coordinates) / ess


def chain_diagnostics(
    samples: Mapping[str, Any], *, num_chains: int = 1
) -> dict[str, SiteDiagnostic]:
    """Per-parameter split r-hat and ESS, judged coordinate by coordinate.

    What B7 asks for. :func:`chain_ess` already reduced ESS to one number by
    MIN over everything, which is the right reduction for a scalar gate and
    the wrong one for a diagnosis: it reports the worst number in the whole
    posterior and cannot say which site, let alone which coordinate, produced
    it. This reports both, per site, and names the coordinate.

    **A non-finite r-hat becomes ``inf``, and that is load-bearing.** numpyro
    returns ``nan`` for a coordinate that never moved -- measured, on a site
    held bitwise constant. ``nan > ceiling`` is ``False``, so the most
    unconverged parameter it is possible to have would pass a naive comparison
    silently. ``inf`` is both the honest value for a chain carrying no
    information and the one that survives the comparison. Note the direction
    differs from :func:`chain_ess`, which maps a non-finite ESS to ``1.0``:
    both choices push the same way, toward the answer that does not flatter
    the chain.

    Args:
        samples: ``{site: draws}`` with the draw axis LEADING and chains
            already concatenated along it, as ``MCMC.get_samples()`` returns.
        num_chains: how many chains that axis holds. Unstacked before the
            estimator sees it -- handing concatenated chains to split r-hat as
            one long chain hides exactly the between-chain disagreement the
            statistic exists to find.

    Raises:
        ValueError: if any site carries fewer than 4 draws per chain, which
            numpyro's ``split_gelman_rubin`` refuses with a bare
            ``AssertionError``. A library assertion is not a diagnosis.
    """
    grouped: dict[str, Any] = {}
    for name, draws in samples.items():
        values = np.asarray(draws)
        block = values.reshape((num_chains, -1, *values.shape[1:]))
        if block.shape[1] < 4:
            raise ValueError(
                f"site {name!r} has {block.shape[1]} draws per chain; split "
                f"r-hat needs at least 4 draws. Sample for longer, or read "
                f"`ess` alone."
            )
        grouped[name] = block

    total = sum(
        int(np.prod(block.shape[2:], dtype=int)) for block in grouped.values()
    ) or 1

    report: dict[str, SiteDiagnostic] = {}
    for name, block in grouped.items():
        r = np.atleast_1d(np.asarray(split_gelman_rubin(block), dtype=float))
        e = np.atleast_1d(np.asarray(effective_sample_size(block), dtype=float))
        shape = block.shape[2:]
        # `inf` wherever either statistic is unusable, so the worst coordinate
        # is chosen on a comparison that a nan would have silently won.
        usable = np.isfinite(r) & np.isfinite(e) & (e > 0.0)
        safe_e = np.where(usable, e, 1.0)
        ceiling = np.where(usable, 1.0 + _constant(total) / safe_e, np.inf)
        # Two failures, ranked so the WORST coordinate is the one reported.
        # `starved` outranks `disagree` because it is checked first: where
        # there is too little information, r-hat's verdict is not evidence
        # either way -- see CHAIN_ESS_FLOOR for the fifteen cells that
        # measured what happens when it is trusted anyway.
        starved = ~usable | (e < CHAIN_ESS_FLOOR)
        excess = np.where(usable, (r - 1.0) / (ceiling - 1.0), np.inf)
        rank = np.where(starved, np.inf, excess)
        flat = int(np.argmax(rank))
        worst = tuple(int(i) for i in np.unravel_index(flat, shape)) if shape else ()
        if starved[flat]:
            reason = (
                f"effective sample size {float(safe_e[flat]) if usable[flat] else 0.0:.1f} "
                f"is below {CHAIN_ESS_FLOOR:.0f}; convergence cannot be "
                f"established at this coordinate, whatever r-hat says"
            )
        elif excess[flat] > 1.0:
            reason = (
                f"split r-hat {float(r[flat]):.4f} exceeds {float(ceiling[flat]):.4f}, "
                f"the ceiling for an effective sample size of {float(e[flat]):.1f}"
            )
        else:
            reason = ""
        report[name] = SiteDiagnostic(
            r_hat=float(r[flat]) if usable[flat] else math.inf,
            ess=float(e[flat]) if usable[flat] else 0.0,
            ceiling=float(ceiling[flat]),
            converged=not reason,
            worst=worst,
            reason=reason,
        )
    return report


def _diagnostics_or_none(
    samples: Mapping[str, Any], num_chains: int
) -> dict[str, SiteDiagnostic] | None:
    """:func:`chain_diagnostics`, or ``None`` where it cannot be computed.

    A run too short for split r-hat is not a reason to fail a sample that
    otherwise succeeded -- the draws are still the draws, and :attr:`ess` still
    describes them. So the refusal :func:`chain_diagnostics` raises is caught
    HERE and nowhere else: a caller asking for diagnostics directly gets the
    error and the diagnosis in it, while a caller who merely sampled gets
    ``None`` and the abstention that :class:`Posterior` documents.
    """
    try:
        return chain_diagnostics(samples, num_chains=num_chains)
    except ValueError:
        return None



def run_sample(
    plan: InferencePlan,
    key: jax.Array,
    *,
    num_samples: int,
    num_warmup: int,
    num_chains: int,
    chain_method: str,
    progress_bar: bool,
    nuts_options: Mapping[str, Any] | None,
    tol: float | None,
    maxiter: int | None,
    require_convergence: float | None,
    ess_floor: float,
    nuts_on_collapse: bool,
) -> Posterior:
    """:meth:`~bayesmith.dispatch.plan.InferencePlan.sample`, as a function.

    Every argument is required here and defaulted there: the method is the
    public spelling and owns the defaults, so there is one place a caller can
    read them off and one place they can drift from.
    """
    draw_key, fallback_key = jax.random.split(key)
    # Every chain setting, in ONE dict that all three sampling paths splat.
    # `chain_method` and `nuts_options` used to travel beside it as separate
    # arguments, reaching `assemble` on the mixed path and nowhere else -- so
    # both bare-NUTS shapes dropped them in silence while `sample`'s docstring
    # said they were "passed to whichever sampler runs". `gibbs.assemble` and
    # `bridge.nuts` take these six keywords under these six names precisely so
    # this dict can be handed to either without a translation step to get
    # wrong.
    chain = {
        "num_warmup": num_warmup,
        "num_samples": num_samples,
        "num_chains": num_chains,
        "chain_method": chain_method,
        "progress_bar": progress_bar,
        "nuts_options": nuts_options,
    }
    if plan.exact is None:
        return _nuts_posterior(plan.graph, fallback_key, plan.sampled.reason, chain)
    tol = plan.exact.tol if tol is None else tol
    if plan.sampled is not None:
        return _swept(plan, draw_key, tol, maxiter, chain)
    return _whole_graph(
        plan,
        draw_key,
        fallback_key,
        tol,
        maxiter,
        require_convergence,
        ess_floor,
        nuts_on_collapse,
        chain,
    )


def run_estimate(
    plan: InferencePlan,
    *,
    tol: float | None,
    maxiter: int | None,
    reweight_tol: float | None,
    min_reweights: int,
    max_reweights: int,
    require_convergence: float | None,
) -> Estimate:
    """:meth:`~bayesmith.dispatch.plan.InferencePlan.estimate`, as a function.

    Defaulted at the method for the same reason as :func:`run_sample`.
    """
    _refuse_unless_whole_graph_exact(plan)
    names = plan.exact.latents
    at = block_at(plan.graph, names)
    block = unchecked_operator(plan.graph, names, at)
    result = iterative_gls(
        block,
        # `precision_of`, the general spelling: a correlated node has no
        # per-sample sigma for `sigma_from_graph` to return, and this is the
        # entry the dispatcher promises an exact solve through.
        precision_of=precision_from_graph(plan.graph, at),
        depends_on_prediction=plan.exact.method != "gcr",
        tol=plan.exact.tol if tol is None else tol,
        maxiter=maxiter,
        reweight_tol=reweight_tol,
        min_reweights=min_reweights,
        max_reweights=max_reweights,
        require_convergence=require_convergence,
    )
    if not bool(result.converged):
        raise ConvergenceError(
            "the GLS reweighting did not reach a fixed point: the last "
            f"relative step was {float(result.delta):.6g} after "
            f"{int(result.iterations)} of at most {max_reweights} "
            "reweights, which is not below reweight_tol="
            f"{'its default, max(8*eps, tol)' if reweight_tol is None else reweight_tol}"
            ". The covariance that came back is therefore NOT a fixed "
            "point, and every moment conditioned on it inherits that. "
            "Raise max_reweights, or loosen reweight_tol -- but not below "
            "8 times the working epsilon or below tol, under either of "
            "which the step being measured is rounding rather than "
            "progress."
        )
    return Estimate(
        dict(result.solution),
        dict(result.precision),
        True,
        result.residual,
        result.iterations,
    )


def _latents_only(samples: Mapping[str, Any], graph: Graph) -> dict[str, jax.Array]:
    """``get_samples()`` minus the Deterministic sites numpyro adds to it."""
    return {name: samples[name] for name in graph.latents}


def _nuts_posterior(
    graph: Graph, key: jax.Array, reason: str, chain: dict[str, Any]
) -> Posterior:
    """Sample the whole graph with NUTS.

    Two callers: the graph with no exact block, and the SNIS collapse when
    ``nuts_on_collapse`` asked for the substitution. Both hand it the same
    ``chain`` dict, so the chain settings a caller passed reach the kernel on
    either -- which they did not before ``bridge.nuts`` took ``chain_method``
    and ``nuts_options``.
    """
    samples = _latents_only(nuts_draws(graph, key, **chain), graph)
    ess = chain_ess(samples, num_chains=chain["num_chains"])
    return Posterior(
        samples, None, ess, None, False, "nuts", reason,
        _diagnostics_or_none(samples, chain["num_chains"]),
    )


def _swept(
    plan: InferencePlan,
    key: jax.Array,
    tol: float,
    maxiter: int | None,
    chain: dict[str, Any],
) -> Posterior:
    """The mixed path: ``HMCGibbs``, assembled from the plan's own three numbers.

    ``method``, ``tol`` and ``sigma_needs_rebuild`` come off the plan rather
    than being re-derived, so what runs is what ``str(plan)`` printed. The
    reason is :meth:`InferencePlan._execution`'s own line for the same reason.
    """
    mcmc = assemble(
        plan.graph,
        plan.exact.latents,
        tol=tol,
        method=plan.exact.method,
        sigma_rebuild=plan.sigma_needs_rebuild,
        maxiter=maxiter,
        **chain,
    )
    mcmc.run(key)
    samples = _latents_only(mcmc.get_samples(), plan.graph)
    ess = chain_ess(samples, num_chains=chain["num_chains"])
    return Posterior(
        samples, None, ess, None, False, plan.exact.method, plan._execution(),
        _diagnostics_or_none(samples, chain["num_chains"]),
    )


def _iid_draws(
    block: Any,
    precision: dict[str, Any],
    key: jax.Array,
    count: int,
    *,
    tol: float,
    maxiter: int | None,
    require_convergence: float | None,
) -> dict[str, jax.Array]:
    """``count`` independent GCR draws at one frozen covariance.

    ``vmap`` over split keys rather than a loop: the fluctuation enters the
    right-hand side only, so every draw is the same solve at a different ``b``
    -- see :func:`~bayesmith.exact.solve.gcr_sample`. Nothing here is a chain,
    so there is no warmup and no ordering.
    """
    keys = jax.random.split(key, count)
    draws, _ = jax.vmap(
        lambda one: gcr_sample(
            block,
            precision=precision,
            key=one,
            tol=tol,
            maxiter=maxiter,
            require_convergence=require_convergence,
        )
    )(keys)
    return draws


def _whole_graph(
    plan: InferencePlan,
    draw_key: jax.Array,
    fallback_key: jax.Array,
    tol: float,
    maxiter: int | None,
    require_convergence: float | None,
    ess_floor: float,
    nuts_on_collapse: bool,
    chain: dict[str, Any],
) -> Posterior:
    """One block spanning every latent: iid draws, reweighted only if sigma moved.

    ``at`` is empty here by construction -- a block covering every latent has
    no outside latent to condition on -- which is what makes the draws
    unconditional posterior draws rather than one Gibbs step's worth.
    """
    graph, names = plan.graph, plan.exact.latents
    at = block_at(graph, names)
    block = unchecked_operator(graph, names, at)
    count = chain["num_samples"]
    settings = {
        "tol": tol,
        "maxiter": maxiter,
        "require_convergence": require_convergence,
    }
    if plan.exact.method == "gcr":
        # Operator-only: these draws never read a per-sample sigma.
        precision = precision_from_graph(graph, at)(domain_centre(block))
        draws = _iid_draws(block, precision, draw_key, count, **settings)
        return Posterior(
            draws,
            None,
            float(count),
            None,
            False,
            "gcr",
            f"exact block {list(names)}: sigma does not move with the block, so "
            "every draw is an independent posterior sample -- no chain, no "
            "warmup, and ESS is num_samples exactly",
        )
    fixed = iterative_gls(
        block,
        precision_of=precision_from_graph(graph, at),
        depends_on_prediction=True,
        tol=tol,
        maxiter=maxiter,
        require_convergence=require_convergence,
    )
    # `fixed.precision` at BOTH, from one GLSResult: the draws and the weight
    # must be at the same covariance or the estimator targets a tilted
    # distribution -- see `check_frozen_sigma`, which measures that.
    draws = _iid_draws(block, fixed.precision, draw_key, count, **settings)
    weights = jax.vmap(
        lambda x: log_weight(
            graph, block, x, at=at, precision=fixed.precision, mu=fixed.solution
        )
    )(draws)
    ess = float(self_normalise(weights)[1])
    collapsed = ess < ess_floor * count
    if collapsed and nuts_on_collapse:
        return _nuts_posterior(
            graph,
            fallback_key,
            _collapse_reason(names, ess, count, ess_floor, replaced=True),
            chain,
        )
    measured = khat(weights)
    reason = (
        _collapse_reason(names, ess, count, ess_floor, replaced=False)
        if collapsed
        else f"exact block {list(names)}: GCR proposal at the GLS fixed point, "
        "corrected by self-normalised importance weights; Kish ESS/N = "
        f"{ess / count:.3g} at N={count}, at or above ess_floor={ess_floor:g}"
    )
    # `unreliable` is an OR and not a replacement: k-hat and the Kish ESS
    # disagree in both directions on this package's own fixtures -- see
    # `SNIS_ESS_FLOOR` -- so a collapse must set the flag whatever k-hat
    # says, and a k-hat past its own threshold must set it whatever the
    # ratio says. The two live in one boolean because a caller checking
    # "may I use these weights" has one question, not two.
    return Posterior(
        draws,
        weights,
        ess,
        measured,
        collapsed or unreliable(measured, count),
        "gcr+snis",
        reason,
    )


def _collapse_reason(
    names: tuple[str, ...], ess: float, count: int, floor: float, *, replaced: bool
) -> str:
    """That the floor fired, the number that fired it, and what was done.

    Two endings for one event, because ``method`` alone cannot carry the
    difference: ``"gcr+snis"`` with ``unreliable=True`` and ``"nuts"`` are
    both honest labels for what ran, and neither says that the OTHER was
    available.

    **Where the fallback is named, it is named as forced rather than
    preferred**: ``gibbs.assemble`` refuses a block covering every latent, in
    those words, the inner NUTS kernel having no site left to sample -- so
    the Gibbs+MH correction is not a third option here.

    **Raising N is not the lever, and the old wording had the mechanism
    wrong.** It said "the Kish ESS of this proposal is bounded by the
    mismatch, not by N", and measurement contradicts that: on
    ``plated_radiometer(n=25, kappa=0.4)`` at key 0 the Kish ESS reads 5.65 at
    N=300, 14.14 at N=1200, 7.47 at N=5000, 55.84 at N=20000 and 119.54 at
    N=60000, so it does grow -- erratically, and 8.5x for 50x the draws. What
    is true is the conclusion: ESS/N, which is what this floor reads, FALLS
    over that range, 0.0188 -> 0.0118 -> 0.0015 -> 0.0028 -> 0.0020. Buying
    draws buys a worse ratio, so the collapse is not a sample-size problem
    and the message says which of those two statements it is making.
    """
    head = (
        f"exact block {list(names)}: the SNIS correction collapsed -- Kish "
        f"ESS/N = {ess / count:.3g} at N={count}, below ess_floor={floor:g} -- "
    )
    tail = (
        "so the weighted sample was discarded and the whole graph was sampled "
        "by NUTS instead, which is what nuts_on_collapse=True asks for. The "
        "Gibbs+MH correction is not available here: a block covering every "
        "latent leaves the inner NUTS kernel no site to sample, so there is no "
        "sweep to embed the Metropolis step in."
        if replaced
        else "so this Posterior carries unreliable=True, and its draws, "
        "weights and khat are handed back rather than replaced. Measured on "
        "plated_radiometer(n=25, kappa=0.4) at N=1200 against exact "
        "per-coordinate quadrature, worst coordinate in units of its own "
        "posterior sd: the weighted answer is out by 1.40 and the NUTS that "
        "used to replace it by 18.5, while NUTS's chain ESS of 33 exceeds "
        "this Kish ESS of 14 -- the diagnostic that fired prefers the answer "
        "13x further from the truth. Pass nuts_on_collapse=True to run NUTS "
        "here instead."
    )
    return (
        head
        + tail
        + " Raising num_samples does not rescue the weights: measured on that "
        "same cell the Kish ESS goes 14.1 at N=1200 to 119.5 at N=60000, i.e. "
        "8.5x for 50x the draws, so the ESS/N this floor reads FALLS from "
        "0.0118 to 0.0020."
    )


def _refuse_unless_whole_graph_exact(plan: InferencePlan) -> None:
    """:meth:`InferencePlan.estimate`'s two refusals, both pointing at ``sample``.

    A refusal that does not say what to do instead is where a user goes and
    writes their own alternating solve, which is rheplicant's motivating
    failure.
    """
    if plan.exact is not None and plan.sampled is None:
        return
    why = (
        "no subgraph of it qualifies for an exact solve, so there is no linear "
        "system to estimate"
        if plan.exact is None
        else f"its exact block {list(plan.exact.latents)} is solved CONDITIONAL "
        f"on {list(plan.sampled.latents)}, and those are only reachable by "
        "sampling"
    )
    raise NotImplementedError(
        f"estimate() has no point estimate for this graph: {why}. A point "
        "estimate of a partly-sampled graph is a MAP over the sampled "
        "latents, and the conditional mean of the exact block at some "
        "arbitrary value of the others is not it, however much it looks like "
        "a number. Two routes now exist and this one is still not either of "
        "them: `bayesmith.estimate_factors(graph, plan)` sweeps a factor "
        "partition -- exact blocks solved, the remainder stepped by `fit` -- "
        "or `bayesmith.fit(graph)` maximises the joint over every latent at "
        "once. Or use sample()."
    )
