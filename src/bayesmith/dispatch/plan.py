"""What :func:`compile` produces: a partition, its evidence, and how to run it.

This is the package's most important user experience: **a model says how it
will be fitted, and why, before it is fitted.** Everything printed here was
measured by :mod:`bayesmith.dispatch.classify` or by
:func:`~bayesmith.exact.solve.condition_bound`; nothing is a label a user
chose.

Two numbers travel together and must be read together. Section 4.2's rule is
that turning the in-sweep convergence guard OFF requires tightening ``tol`` in
the same breath -- rheplicant names "leave ``tol`` at its default and the
guard off" as the combination that returned a silently over-confident
posterior. So the plan prints ``kappa`` and the ``tol`` derived from it side by
side, at enough digits that a reader can check the division, together with
whether the guard is still running.

**Kappa is not one number when a latent outside the block moves it.** A block
is affine *given* the latents outside it, and its conditioning is a function
of where they sit -- ``indirect_ancestor``'s ``x`` has a prior width that is a
function of ``tau``, and ``tau`` moves every sweep. Pinning kappa at the prior
centre understates it by more than an order of magnitude at the edge of
``tau``'s own prior, and the error is in the dangerous direction: ``tol`` comes
out too LOOSE, CG stops early, the posterior comes back too narrow, and inside
a sweep the guard that would have noticed is hoisted out. So the block is
probed across the outside latents' own priors and the interval is what gets
printed, with ``tol`` derived from its UPPER end.

**Deriving the plan samples nothing and is not jittable**, for the same reason
:mod:`bayesmith.dispatch.classify` is not: every measurement :func:`compile`
takes runs on concrete values at compile time. :meth:`InferencePlan.sample`
and :meth:`InferencePlan.estimate` are the two places that do run, and they
run what the plan printed -- the same block, the same ``tol``, the same
method.
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Mapping
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.diagnostics import effective_sample_size

from bayesmith.bridge.numpyro_bridge import nuts as nuts_draws
from bayesmith.dispatch.classify import (
    Classification,
    block_at,
    partition,
    prior_environment,
)
from bayesmith.errors import BayesmithError, ConvergenceError
from bayesmith.exact.block import domain_centre, unchecked_operator
from bayesmith.exact.correct import khat, log_weight, self_normalise, unreliable
from bayesmith.exact.gaussian import gaussian_parts, node_shape
from bayesmith.exact.gibbs import assemble
from bayesmith.exact.gls import (
    MAX_REWEIGHTS,
    MIN_REWEIGHTS,
    iterative_gls,
    sigma_from_graph,
)
from bayesmith.exact.solve import condition_bound, gcr_sample
from bayesmith.graph.graph import Graph

CONVERGENCE_TARGET: float = 1e-3
"""Relative error the in-sweep ``tol`` is chosen to deliver: ``tol = target / kappa``.

The direction is the whole point. ``condition_bound``'s own docstring states
the relation -- "for a target relative accuracy ``a``, ask for roughly
``tol = a / condition_bound(...)``" -- because error and residual differ by
the condition number. Multiplying instead of dividing does not merely pick a
different number, it inverts the guard: the worse-conditioned the block, the
looser the tolerance it would be solved to.

A ``tol`` below the working precision's epsilon is not clipped here. It is a
true statement that this target is unreachable at this precision, and
:func:`~bayesmith.exact.solve.wiener_solve`'s own guard already says so in
those words (``bound * epsilon > require_convergence``). Clipping would
replace an honest "unreachable" with a quiet claim of an accuracy the
arithmetic cannot carry.
"""

KAPPA_PROBE_SIGMAS: tuple[float, ...] = (-3.0, -1.0, 1.0, 3.0)
"""Where each outside latent is probed, in units of its own prior width.

**Two-sided about the anchor, which is interior rather than at an end.** The
point the block was CLASSIFIED at is the prior centre, ``0.0``, and it is
swept too -- so the default sits in the middle of the scan and a one-sided
error in either direction is visible.

Symmetric and bounded by the prior rather than open-ended, because the prior
is the statement of where the latent is expected to be; +/-3 covers 99.7% of
a Gaussian one. The interior +/-1 points are not redundant with the ends:
nothing makes kappa monotone in an outside latent, so an interior maximum is
possible and endpoints alone would miss it.

Measured on ``indirect_ancestor`` (``tau ~ N(2.0, 0.5)``, kappa growing with
``|tau|``): the reachable interval is [21.4, 737] at +/-3, [69.7, 547] at
+/-2 and [146, 385] at +/-1, against 251 at the centre. Widening is the safe
direction -- a bigger upper end gives a tighter ``tol`` -- and the cost of it
is CG iterations, not correctness.
"""

_LINE_WIDTH: int = 88
"""Total width the printed plan wraps to, matching this project's line length."""

_LABELS: dict[str, str] = {
    "gcr": "GCR exact",
    "gcr+snis": "GCR + SNIS",
    "gcr+mh": "GCR + MH accept",
    "nuts": "NUTS",
}

_MEMBER_COLUMN: int = 16
_METHOD_COLUMN: int = 18
_BODY_INDENT: int = len("block 0  ")


def kappa_upper(kappa: float | tuple[float, float]) -> float:
    """The end of ``kappa`` that ``tol`` is derived from.

    An interval's UPPER end, never its lower one. Taking ``lo`` gives a
    ``tol`` that is too loose by the width of the interval -- 34x on
    ``indirect_ancestor`` -- and too loose is the direction that stops CG
    early, returns a posterior that is too narrow, and does it inside a sweep
    where the convergence guard has been hoisted out and nothing notices.
    """
    return float(kappa[1] if isinstance(kappa, tuple) else kappa)


def tol_for(kappa: float | tuple[float, float]) -> float:
    """``CONVERGENCE_TARGET / kappa`` -- the one place the division lives."""
    return CONVERGENCE_TARGET / kappa_upper(kappa)


def _kappa_at(
    graph: Graph, names: tuple[str, ...], at: dict[str, Any], key: jax.Array
) -> float:
    """``condition_bound`` for the block built at one value of the outside latents.

    Built through :func:`~bayesmith.exact.block.unchecked_operator` at an
    ``at`` derived from :func:`~bayesmith.dispatch.classify.block_at`, so the
    block whose conditioning is reported is the block that was CLASSIFIED,
    not a second, independently-spelled one. Sigma is taken through
    :func:`~bayesmith.exact.gls.sigma_from_graph` -- ``classify``'s own seam
    -- evaluated at the block's prior centre, which is where the solve starts.

    **The prior centre rather than zero is unguarded, and measured to be so.**
    Substituting ``domain_zero`` here leaves all 67 tests in
    ``tests/dispatch/test_plan.py`` green, because on all 28 fixtures with an
    exact block ``sigma(domain_centre) == sigma(domain_zero)`` bitwise: 26 have
    a prior mean of exactly zero, and the two that do not
    (``unconstrained_latent``, ``plated_latent_through_deterministic``) have a
    constant sigma. The distinction is real for a model that has both -- a
    noise-wave temperature near 250 K read by a radiometric sigma is exactly
    that shape, and it is the case
    :func:`~bayesmith.exact.block.domain_centre`'s own docstring exists for --
    so the centre is what is used, with no fixture here able to tell.
    """
    operator = unchecked_operator(graph, names, at)
    sigma = sigma_from_graph(graph, at)(domain_centre(operator))
    return float(condition_bound(operator, noise_std=sigma, key=key))


def _probe_values(
    graph: Graph, name: str, env: dict[str, Any]
) -> list[jax.Array] | None:
    """``loc + s * scale`` for every ``s``, or ``None`` for a latent with no width.

    A non-Gaussian outside latent has no ``scale`` to sweep by. ``Cauchy`` and
    ``ImproperUniform`` are both live fixtures here
    (``overflowing_outside_latent``, ``improper_outside_prior``) and both are
    LEGAL models, so the answer is to leave them at their centre and say so --
    the same policy, and for the same reason, as
    ``classify._at_points``' handling of the linearity check's own draws.
    """
    node = graph.node(name)
    try:
        loc, scale = gaussian_parts(graph, node, env)
    except BayesmithError:
        return None
    shape = node_shape(graph, node, env)
    loc = jnp.broadcast_to(loc, shape)
    scale = jnp.broadcast_to(scale, shape)
    return [loc + sigmas * scale for sigmas in KAPPA_PROBE_SIGMAS]


def _sweep_note(held: list[str], refused: list[str]) -> str:
    """What the interval could NOT see, in the block's own reason."""
    parts = []
    if held:
        parts.append(
            f"the kappa sweep held {sorted(set(held))} at their prior centre "
            "rather than moving them -- they are not Gaussian, so this module "
            "has no prior width to sweep them by, and the interval is not an "
            "interval in them"
        )
    if refused:
        parts.append(
            f"the kappa sweep could not build the block at {sorted(set(refused))}'s "
            "probe points, so those were dropped and the interval is narrower "
            "than the prior it was meant to cover"
        )
    return "".join("; " + part for part in parts)


def kappa_interval(
    graph: Graph,
    names: tuple[str, ...],
    *,
    env: dict[str, Any] | None = None,
    key: jax.Array | None = None,
) -> tuple[float, float, str]:
    """``(lo, hi, note)`` -- how far the block's conditioning moves, and what was missed.

    The centre is always evaluated, so ``lo == hi`` exactly when nothing
    moved kappa: a block spanning every latent has no outside latent to sweep,
    and one whose outside latents are all non-Gaussian has none this module
    can sweep. Both print as a single number rather than a degenerate
    interval, and the second says so in ``note``.

    **A sampled interval, not a bound.** Each outside latent is moved on its
    own, from the centre, along the UNIFORM direction (every element of a
    plated latent displaced together) -- ``1 + len(KAPPA_PROBE_SIGMAS) *
    outside`` evaluations rather than a product grid, and no contrast
    direction within a plate. Every outside latent in this package's fixture
    suite is scalar, so the contrast case is undocumented by measurement
    rather than covered.

    Args:
        graph: the model.
        names: the exact block's members.
        env: :func:`~bayesmith.dispatch.classify.prior_environment`, if it has
            already been built. Passed in rather than rebuilt so the plan
            anchors where ``partition`` classified.
        key: PRNG key for ``condition_bound``'s power iteration. Fixed by
            default, so the interval is reproducible.
    """
    key = jax.random.key(0) if key is None else key
    env = prior_environment(graph) if env is None else env
    at = block_at(graph, names, env=env)
    values = [_kappa_at(graph, names, at, key)]
    held: list[str] = []
    refused: list[str] = []
    for name in sorted(at):
        probes = _probe_values(graph, name, env)
        if probes is None:
            held.append(name)
            continue
        for probe in probes:
            try:
                measured = _kappa_at(graph, names, {**at, name: probe}, key)
            except BayesmithError:
                refused.append(name)
                continue
            if math.isfinite(measured):
                values.append(measured)
            else:
                refused.append(name)
    return min(values), max(values), _sweep_note(held, refused)


SNIS_ESS_FLOOR: float = 0.1
"""Kish ESS/N below which the SNIS correction is discarded and NUTS is run.

Section 5.2's decision, and the reason it is a floor on the RATIO rather than
a flag on the object: self-normalised importance sampling degrades
**exponentially** in the number of mismatched coordinates and reports nothing
while it does. :func:`~bayesmith.exact.correct.self_normalise`'s own docstring
carries the reference measurement (n=1 -> ESS 2509, n=500 -> 1.00, at N=40000
throughout); sweeping only the plate of this package's ``plated_radiometer``
at N=2000 reproduces the shape -- 0.966 (n=1), 0.739 (n=6), 0.445 (n=12),
0.123 (n=50), 0.054 (n=100), 0.0039 (n=400).

**0.1 is the variance-inflation red line, not a crossover.** At ESS/N = 0.1
the estimator's variance is inflated 10x, i.e. the error bars are 3.16x wider
than the nominal ones and recovering a nominal bar costs ten times the draws.
It would be dishonest to claim the fallback is uniformly better: measured at
N=1000 against NUTS's own min-over-coordinates ESS/N on the same graph,
``plated_radiometer(n=6, kappa=0.4)`` reads SNIS 0.120 against NUTS 0.014 (the
weights win, 8x), while ``n=12, kappa=0.2`` reads SNIS 0.0065 against NUTS
0.332 (NUTS wins, 51x) and ``n=50, kappa=0.2`` reads 0.014 against 0.241. So
the crossover lies somewhere in 0.01-0.12 and is not sharp; 0.1 is its
conservative end. What the floor guarantees is not the better estimator, it is
that a ``Posterior`` carrying a thousand draws never comes back with four of
them real and only a boolean to say so.

**It is read off the Kish ESS and never off k-hat**, which is not
interchangeable with it here. Measured on ``radiometer()`` -- a
ONE-dimensional block -- at N=1200 over six keys: Kish ESS/N is 0.9986-0.9988
every time while k-hat reads 0.91 to 1.80, past
:data:`~bayesmith.exact.correct.FINITE_MEAN_KHAT`. The log weights span under
one nat there, which is not a tail PSIS has any business fitting a
generalised Pareto to. Dispatch on k-hat and that graph goes to NUTS with a
collapse reported that did not happen.
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
        unreliable: k-hat past ``min(1 - 1/log10(N), 0.7)``. ``False`` wherever
            ``khat`` is ``None``, which is **abstention, not endorsement** --
            read ``ess``, which is always there.
        method: what actually RAN: ``"gcr"``, ``"gcr+snis"``, ``"gcr+mh"`` or
            ``"nuts"``. Not necessarily ``plan.exact.method``: a collapsed
            SNIS reports ``"nuts"``, because that is what produced the draws.
        reason: why that, in the plan's own words, including the measured
            Kish ESS/N wherever a fallback turned on it.
    """

    samples: dict[str, jax.Array]
    log_weights: jax.Array | None
    ess: float
    khat: float | None
    unreliable: bool
    method: str
    reason: str


class Estimate(NamedTuple):
    """What :meth:`InferencePlan.estimate` returns: a point, and its covariance.

    Attributes:
        values: the GLS/Wiener solution, ``{latent: value}``.
        noise_std: the covariance it was solved at -- for a
            prediction-dependent model, the fixed point, so
            ``noise_std_at(graph, values)`` reproduces it. Feed it back to
            ``gcr_sample`` to draw at the same covariance.
        converged: always ``True``. ``False`` is not returned; it is raised as
            :class:`~bayesmith.errors.ConvergenceError`, which is the whole
            point of this being the promotion site. Kept as a field because a
            caller reading a stored ``Estimate`` should not have to know that.
        residual: relative CG residual of the final solve. Not an accuracy --
            multiply by ``plan.exact.kappa`` for the error bound.
        iterations: reweighting steps taken, ``1`` for a constant sigma.
    """

    values: dict[str, jax.Array]
    noise_std: dict[str, jax.Array]
    converged: bool
    residual: jax.Array
    iterations: jax.Array


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


class Block(eqx.Module):
    """One group of latents and the method the graph selected for it.

    Attributes:
        latents: the members, sorted.
        method: ``"gcr"``, ``"gcr+snis"``, ``"gcr+mh"`` or ``"nuts"``, as
            :class:`~bayesmith.dispatch.classify.Classification` chose it.
        reason: why -- naming members on a refusal, plus whatever the kappa
            sweep could not reach.
        linearity: ``check_linearity``'s per-at-point errors, or ``None``.
        kappa: a single conditioning bound, or ``(lo, hi)`` when a latent
            outside the block moves it. ``None`` for a NUTS block, which is
            solved by no linear system at all.
        tol: ``CONVERGENCE_TARGET / kappa_upper(kappa)``.
    """

    latents: tuple[str, ...] = eqx.field(static=True)
    method: str = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    linearity: dict | None = eqx.field(static=True, default=None)
    kappa: float | tuple[float, float] | None = eqx.field(static=True, default=None)
    tol: float | None = eqx.field(static=True, default=None)


def _evidence(block: Block) -> str:
    """The linearity check's own numbers, on the block's first line.

    Both counts and the worst departure, because "checked" without them
    cannot be told apart from a single-point probe -- the exact failure mode
    ``check_linearity``'s ``at_points`` keyword exists to prevent.
    """
    if not block.linearity:
        return ""
    scales = len(next(iter(block.linearity.values())))
    worst = max(value for row in block.linearity.values() for value in row.values())
    return (
        f"linear_in ✓ {scales} scales x {len(block.linearity)} at-points "
        f"(max {worst:.2e})"
    )


def _wrapped(text: str, width: int) -> list[str]:
    """``textwrap.wrap`` with both of its word-splitting defaults turned off.

    Measured, not stylistic: with ``break_on_hyphens`` left at its default,
    ``check_linearity``'s refusal message comes out with ``sigma-weighted``
    split across two lines, and ``tests/dispatch/test_plan.py``'s check that
    the plan reproduces the classifier's reason -- whitespace collapsed on
    both sides -- fails on a word the plan silently rewrote. A reason quoting
    a hyphenated identifier or a negative exponent is worth keeping intact
    for the same reason: it is meant to be pasted back.
    """
    return textwrap.wrap(
        text, width=width, break_on_hyphens=False, break_long_words=False
    )


def _kappa_text(kappa: float | tuple[float, float]) -> str:
    """``kappa=...`` or ``kappa in [lo, hi]``.

    Eight significant digits, not the three a reader would prefer, because
    the printed pair is meant to be CHECKED: ``tol`` is asserted against
    ``CONVERGENCE_TARGET / kappa`` to a relative 1e-6, and a three-digit
    kappa cannot round-trip to better than 1e-3.
    """
    if isinstance(kappa, tuple):
        return f"kappa in [{kappa[0]:.8g}, {kappa[1]:.8g}]"
    return f"kappa={kappa:.8g}"


def _continuation(block: Block, *, hoisted: bool, width: int) -> list[str]:
    """The block's remaining lines: the kappa/tol pair, then the reason.

    ``hoisted`` is a property of the PLAN, not of the block -- the in-sweep
    guard is hoisted exactly when there is a sweep to hoist it out of -- so it
    is passed in rather than stored. It is printed next to ``tol`` and not
    somewhere else because section 4.2's rule is about the two together: a
    ``tol`` with no statement of the guard is half of the pair the rule
    constrains.
    """
    lines: list[str] = []
    if block.kappa is not None and block.tol is not None:
        guard = (
            "guard hoisted out of the sweep"
            if hoisted
            else "guard reachable, off by default (require_convergence=)"
        )
        pair = f"{_kappa_text(block.kappa)} -> tol={block.tol:.8g}, {guard}"
        lines.extend(_wrapped(pair, width))
    lines.extend(_wrapped(block.reason, width) or [""])
    return lines


class InferencePlan(eqx.Module):
    """What :func:`compile` produces: a partition, its reasons, and how to run it.

    Attributes:
        graph: the model this plan was derived from.
        blocks: the exact block first, if there is one, then the sampled one.
        sigma_needs_rebuild: whether an observed node's scale has a latent
            ancestor outside the exact block, in which case ``noise_std``
            must be recomputed every sweep rather than hoisted.
    """

    graph: Graph
    blocks: tuple[Block, ...]
    sigma_needs_rebuild: bool = eqx.field(static=True, default=False)

    @property
    def exact(self) -> Block | None:
        """The block an exact method applies to, or ``None``."""
        found = [block for block in self.blocks if block.method != "nuts"]
        return found[0] if found else None

    @property
    def sampled(self) -> Block | None:
        """The block NUTS samples, or ``None`` if the graph is fully exact."""
        found = [block for block in self.blocks if block.method == "nuts"]
        return found[0] if found else None

    @property
    def guard_hoisted(self) -> bool:
        """Whether the exact solve runs inside a sweep with its guard off.

        True exactly for a MIXED plan. The guard
        (:func:`~bayesmith.exact.solve.wiener_solve`'s ``require_convergence``)
        is a concrete-valued check and a Gibbs sweep runs under ``jit``, so
        inside a sweep it cannot run at all -- which is why ``tol`` has to
        carry the discipline on its own there.

        **False does not mean the guard is running**, and the printed line
        says so in those words. A fully exact plan runs its solve outside any
        trace, so the guard CAN run there -- but :meth:`sample` and
        :meth:`estimate` leave it off and expose it as ``require_convergence=``
        instead, because ``tol`` is already derived to deliver
        ``CONVERGENCE_TARGET`` and a guard set to that same target compares two
        numbers constructed to be equal. Measured at float32: ``straight_line``
        lands at ``1.078e-07 * 924.4 = 9.97e-04`` and is accepted by 0.3%,
        ``two_linear_latents`` at ``2.179e-07 * 5792 = 1.262e-03`` and is
        refused by 26%.
        """
        return self.exact is not None and self.sampled is not None

    def _execution(self) -> str:
        """The one line that says what will actually be run."""
        if self.exact is None:
            return "NUTS"
        if self.sampled is None:
            return "iid draws, no chain"
        sigma = (
            "noise_std rebuilt every sweep"
            if self.sigma_needs_rebuild
            else "noise_std hoisted out of the sweep"
        )
        sites = list(self.exact.latents)
        return f"HMCGibbs(inner=NUTS, gibbs_sites={sites}); {sigma}"

    def __str__(self) -> str:
        """One head line per block, its evidence indented under it, then execution.

        The continuation is indented to the ``block N`` column rather than
        under the evidence one -- which is where the plan for this task put it
        -- because at that depth (44 columns) a ``kappa in [lo, hi] -> tol=``
        pair printed to the eight digits it has to round-trip at wraps three
        times, and the one line a reader most needs to take in whole is the
        one that comes out least readable.
        """
        lines = []
        hoisted = self.guard_hoisted
        pad = " " * _BODY_INDENT
        width = _LINE_WIDTH - _BODY_INDENT
        for index, block in enumerate(self.blocks):
            members = "{" + ", ".join(block.latents) + "}"
            label = _LABELS.get(block.method, block.method)
            head = (
                f"block {index}  {members:<{_MEMBER_COLUMN}s} "
                f"{label:<{_METHOD_COLUMN}s}"
            )
            lines.append((head + _evidence(block)).rstrip())
            for extra in _continuation(block, hoisted=hoisted, width=width):
                lines.append(pad + extra)
        lines.append("execution: " + self._execution())
        return "\n".join(lines)

    def sample(
        self,
        key: jax.Array,
        *,
        num_samples: int = 2000,
        num_warmup: int = 1000,
        num_chains: int = 1,
        chain_method: str = "sequential",
        progress_bar: bool = False,
        nuts_options: Mapping[str, Any] | None = None,
        tol: float | None = None,
        maxiter: int | None = None,
        require_convergence: float | None = None,
        ess_floor: float = SNIS_ESS_FLOOR,
    ) -> Posterior:
        """Run the plan. Section 6.4's dispatch, and nothing else decides.

        Five shapes, in the order this reads them: no exact block -> NUTS;
        exact block plus a sampled one -> ``HMCGibbs`` with the sweep the plan
        printed; whole graph exact with a fixed sigma -> iid GCR draws, no
        chain; whole graph exact with a moving sigma -> GCR at the GLS fixed
        point corrected by SNIS; and that last one again when its Kish ESS/N
        falls under ``ess_floor`` -> discard the weights and run NUTS, saying
        so. See :data:`SNIS_ESS_FLOOR` for why the last row exists.

        Args:
            key: PRNG key. Split once, so the draws and any fallback chain do
                not share a stream.
            num_samples, num_warmup, num_chains, chain_method, progress_bar,
                nuts_options: passed to whichever sampler runs. ``num_warmup``
                is ignored on the iid path, there being nothing to adapt.
            tol: CG tolerance. Defaults to the plan's own
                ``exact.tol = CONVERGENCE_TARGET / kappa``; overriding it
                overrides the discipline that number carries.
            maxiter: CG iteration cap.
            require_convergence: bound on the relative ERROR of each solve.
                **Off by default, measured.** ``tol`` is derived so that a CG
                stopping at ``tol`` delivers exactly ``CONVERGENCE_TARGET``,
                so switching this on at that same target compares two numbers
                constructed to be equal and fires on rounding: at float32
                ``straight_line`` lands at ``1.078e-07 * 924.4 = 9.97e-04``
                and passes, ``two_linear_latents`` at ``2.179e-07 * 5792 =
                1.262e-03`` and does not -- two fixtures of the same shape,
                0.3% inside and 26% outside. It is also a property of the
                OPERATOR, so on the draw path it would be re-measured, at
                ``POWER_ITERATIONS`` operator applications, once per draw.
            ess_floor: the Kish ESS/N under which the SNIS path falls back.

        Returns:
            A :class:`Posterior`, whose ``method`` is what RAN.
        """
        draw_key, fallback_key = jax.random.split(key)
        chain = {
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "num_chains": num_chains,
            "progress_bar": progress_bar,
        }
        if self.exact is None:
            return _nuts_posterior(self.graph, fallback_key, self.sampled.reason, chain)
        tol = self.exact.tol if tol is None else tol
        if self.sampled is not None:
            return _swept(self, draw_key, tol, maxiter, chain_method,
                          nuts_options, chain)
        return _whole_graph(
            self, draw_key, fallback_key, tol, maxiter, require_convergence,
            ess_floor, chain,
        )

    def estimate(
        self,
        *,
        tol: float | None = None,
        maxiter: int | None = None,
        reweight_tol: float | None = None,
        min_reweights: int = MIN_REWEIGHTS,
        max_reweights: int = MAX_REWEIGHTS,
        require_convergence: float | None = None,
    ) -> Estimate:
        """The point estimate, where one exists. Section 6.5's dispatch.

        Whole graph exact with a fixed sigma is one
        :func:`~bayesmith.exact.solve.wiener_solve`; whole graph exact with a
        prediction-dependent sigma is
        :func:`~bayesmith.exact.gls.iterative_gls`'s fixed point. Anything
        mixed is REFUSED and pointed at :meth:`sample`.

        Both paths are spelled through ``iterative_gls``, whose
        ``depends_on_prediction=False`` branch *is* the single ``wiener_solve``
        -- so :class:`Estimate`'s five fields come from one place rather than
        two, and the difference between the branches stays visible in
        ``iterations`` (1 against several) rather than being invisible in
        which function was called.

        Args:
            tol, maxiter, require_convergence: as for :meth:`sample`; ``tol``
                defaults to the plan's own and the guard is off for the same
                measured reason.
            reweight_tol, min_reweights, max_reweights: the outer
                fixed-point iteration's own settings.

        Raises:
            ConvergenceError: if the reweighting did not reach a fixed point.
                P3a defined this class and left it unraised, leaving the
                promotion of ``GLSResult.converged`` to its caller; this is
                that caller.
            NotImplementedError: for a graph any part of which is only
                samplable. A point estimate there is a MAP, which is P5.
        """
        _refuse_unless_whole_graph_exact(self)
        names = self.exact.latents
        at = block_at(self.graph, names)
        block = unchecked_operator(self.graph, names, at)
        result = iterative_gls(
            block,
            sigma_from_graph(self.graph, at),
            depends_on_prediction=self.exact.method != "gcr",
            tol=self.exact.tol if tol is None else tol,
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
            dict(result.noise_std),
            True,
            result.residual,
            result.iterations,
        )


def _latents_only(
    samples: Mapping[str, Any], graph: Graph
) -> dict[str, jax.Array]:
    """``get_samples()`` minus the Deterministic sites numpyro adds to it."""
    return {name: samples[name] for name in graph.latents}


def _nuts_posterior(
    graph: Graph, key: jax.Array, reason: str, chain: dict[str, Any]
) -> Posterior:
    """Sample the whole graph with NUTS -- the no-structure path and the fallback."""
    samples = _latents_only(nuts_draws(graph, key, **chain), graph)
    ess = chain_ess(samples, num_chains=chain["num_chains"])
    return Posterior(samples, None, ess, None, False, "nuts", reason)


def _swept(
    plan: InferencePlan,
    key: jax.Array,
    tol: float,
    maxiter: int | None,
    chain_method: str,
    nuts_options: Mapping[str, Any] | None,
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
        chain_method=chain_method,
        nuts_options=nuts_options,
        **chain,
    )
    mcmc.run(key)
    samples = _latents_only(mcmc.get_samples(), plan.graph)
    ess = chain_ess(samples, num_chains=chain["num_chains"])
    return Posterior(
        samples, None, ess, None, False, plan.exact.method, plan._execution()
    )


def _iid_draws(
    block: Any,
    sigma: dict[str, Any],
    key: jax.Array,
    count: int,
    *,
    tol: float,
    maxiter: int | None,
    require_convergence: float | None,
) -> dict[str, jax.Array]:
    """``count`` independent GCR draws at one frozen sigma.

    ``vmap`` over split keys rather than a loop: the fluctuation enters the
    right-hand side only, so every draw is the same solve at a different ``b``
    -- see :func:`~bayesmith.exact.solve.gcr_sample`. Nothing here is a chain,
    so there is no warmup and no ordering.
    """
    keys = jax.random.split(key, count)
    draws, _ = jax.vmap(
        lambda one: gcr_sample(
            block,
            noise_std=sigma,
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
        sigma = sigma_from_graph(graph, at)(domain_centre(block))
        draws = _iid_draws(block, sigma, draw_key, count, **settings)
        return Posterior(
            draws, None, float(count), None, False, "gcr",
            f"exact block {list(names)}: sigma does not move with the block, so "
            "every draw is an independent posterior sample -- no chain, no "
            "warmup, and ESS is num_samples exactly",
        )
    fixed = iterative_gls(
        block,
        sigma_from_graph(graph, at),
        depends_on_prediction=True,
        tol=tol,
        maxiter=maxiter,
        require_convergence=require_convergence,
    )
    draws = _iid_draws(block, fixed.noise_std, draw_key, count, **settings)
    weights = jax.vmap(
        lambda x: log_weight(
            graph, block, x, at=at, noise_std=fixed.noise_std, mu=fixed.solution
        )
    )(draws)
    ess = float(self_normalise(weights)[1])
    if ess < ess_floor * count:
        return _nuts_posterior(
            graph, fallback_key, _collapse_reason(names, ess, count, ess_floor), chain
        )
    measured = khat(weights)
    return Posterior(
        draws, weights, ess, measured, unreliable(measured, count), "gcr+snis",
        f"exact block {list(names)}: GCR proposal at the GLS fixed point, "
        "corrected by self-normalised importance weights; Kish ESS/N = "
        f"{ess / count:.3g} at N={count}, at or above ess_floor={ess_floor:g}",
    )


def _collapse_reason(
    names: tuple[str, ...], ess: float, count: int, floor: float
) -> str:
    """Why the weights were thrown away, with the number that threw them.

    Names NUTS rather than the Gibbs+MH path because that choice is FORCED,
    not preferred: ``gibbs.assemble`` refuses a block covering every latent,
    in those words, the inner NUTS kernel having no site left to sample.
    """
    return (
        f"exact block {list(names)}: the SNIS correction collapsed -- Kish "
        f"ESS/N = {ess / count:.3g} at N={count}, below ess_floor={floor:g} -- "
        "so the weighted sample was discarded and the whole graph was sampled "
        "by NUTS instead. The Gibbs+MH correction is not available here: a "
        "block covering every latent leaves the inner NUTS kernel no site to "
        "sample, so there is no sweep to embed the Metropolis step in. Raising "
        "num_samples does not help -- the Kish ESS of this proposal is bounded "
        "by the mismatch, not by N."
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
        f"estimate() has no point estimate for this graph: {why}. Use sample() "
        "instead. A point estimate of a partly-sampled graph is a MAP over the "
        "sampled latents, which needs an optimiser this package does not ship "
        "(P5) -- and the conditional mean of the exact block at some arbitrary "
        "value of the others is not it, however much it looks like a number."
    )


def _sampled_reason(classification: Classification) -> str:
    """Why the remaining latents are sampled.

    With NO exact block, this is the whole verdict and the classifier's own
    reason is the only place the members that failed are named -- so it is
    passed through verbatim. ``bilinear_pair``'s "NUTS" with the ``gain``/
    ``t_ant`` refusal dropped would be indistinguishable from a graph that
    simply has no exact structure.

    With one, the classifier's reason is already printed against the exact
    block, and what is left to say is structural rather than per-latent.
    :class:`~bayesmith.dispatch.classify.Classification` keeps its ``why_not``
    map only when the block comes out EMPTY -- once a block is found, the
    per-latent grounds for ejection are gone. Re-deriving them here would be a
    second copy of the qualification rules, and two copies is how the plan
    comes to describe a partition the classifier did not make. Recorded as a
    gap rather than papered over.
    """
    if not classification.exact:
        return classification.reason
    return (
        f"outside the exact block {list(classification.exact)}: sampled by NUTS, "
        "and the exact block is solved CONDITIONAL on them, once per sweep"
    )


def compile(graph: Graph, *, key: jax.Array | None = None) -> InferencePlan:
    """Derive the plan for a graph: what runs, on which latents, and why.

    Runs :func:`~bayesmith.dispatch.classify.partition`, then measures the
    exact block's conditioning across the outside latents' own priors and
    derives ``tol`` from the worst of it. Takes no samples and forms no
    matrix; costs ``1 + len(KAPPA_PROBE_SIGMAS) * outside`` power iterations
    on top of what ``partition`` already spends.

    Shadows the builtin ``compile`` at module scope, which is the decided UX
    -- ``bayesmith.compile(graph)`` is the name this package wants -- so
    callers who need both spell it ``from bayesmith import compile as
    compile_graph``.

    Args:
        graph: the model.
        key: PRNG key, passed to both ``partition``'s linearity probes and
            ``condition_bound``'s power iteration. Fixed by default, so a
            plan is reproducible; and the default is
            ``condition_bound``'s own default, so a kappa printed here equals
            a kappa a caller measures directly.

    Returns:
        An :class:`InferencePlan`, whose ``str`` is the readable form.
    """
    key = jax.random.key(0) if key is None else key
    classification = partition(graph, key=key)
    env = prior_environment(graph)
    blocks: list[Block] = []
    if classification.exact:
        low, high, note = kappa_interval(graph, classification.exact, env=env, key=key)
        kappa: float | tuple[float, float] = high if high <= low else (low, high)
        blocks.append(
            Block(
                latents=classification.exact,
                method=classification.method,
                reason=classification.reason + note,
                linearity=classification.linearity,
                kappa=kappa,
                tol=tol_for(kappa),
            )
        )
    if classification.nuts:
        blocks.append(
            Block(
                latents=classification.nuts,
                method="nuts",
                reason=_sampled_reason(classification),
            )
        )
    return InferencePlan(graph, tuple(blocks), classification.sigma_needs_rebuild)
