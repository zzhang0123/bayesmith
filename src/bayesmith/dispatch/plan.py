"""What :func:`compile` produces: a partition, its reasons, and how to run it.

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

The running itself lives next door in :mod:`bayesmith.dispatch.execute`, which
this module imports; :meth:`InferencePlan.sample` and
:meth:`InferencePlan.estimate` keep their defaults and their documentation here
and delegate the body there.
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from bayesmith.dispatch.classify import (
    Classification,
    block_at,
    partition,
    prior_environment,
)
from bayesmith.dispatch.execute import (
    SNIS_ESS_FLOOR,
    Estimate,
    Posterior,
    # Re-exported, not used here: `from bayesmith.dispatch.plan import
    # chain_ess` is the path this name has had since it was written, and
    # `bayesmith.dispatch.execute` is where it now lives.
    chain_ess,  # noqa: F401
    run_estimate,
    run_sample,
)
from bayesmith.dispatch.streaming import StreamingRoute, streaming_route
from bayesmith.errors import NotGaussian
from bayesmith.exact.block import domain_centre, unchecked_operator
from bayesmith.exact.gaussian import gaussian_parts, node_shape, precision_at
from bayesmith.exact.gls import MAX_REWEIGHTS, MIN_REWEIGHTS
from bayesmith.exact.solve import condition_bound
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
    # `precision_at`, not `sigma_from_graph` + `diagonal_from`: a conditioning
    # number is a property of one normal operator, and this site only ever
    # wanted the operator. Reading it through the sigma producer made the
    # whole conditioning estimate refuse a correlated node.
    precision = precision_at(graph, {**at, **domain_centre(operator)})
    return float(condition_bound(operator, precision=precision, key=key))


def working_epsilon(graph: Graph, names: tuple[str, ...], at: dict[str, Any]) -> float:
    """Machine epsilon of the dtype the block's own arithmetic runs at.

    Read off ``block.offset``, which is the same leaf
    :func:`~bayesmith.exact.solve.wiener_solve`'s own precision guard reads
    (``jnp.result_type(*jax.tree.leaves(block.offset))``). Deliberately the
    same expression and not an equivalent one: the plan's verdict about
    whether ``tol`` is reachable and the solver's verdict about whether
    ``require_convergence`` is reachable must be verdicts about one number,
    or the plan can promise an accuracy the solve then refuses.

    It is a property of the GRAPH rather than of this module, because
    ``const`` and ``observe`` capture their arrays at trace time: a graph
    traced outside ``jax.enable_x64(True)`` stays float32 however it is later
    compiled.
    """
    block = unchecked_operator(graph, names, at)
    return float(jnp.finfo(jnp.result_type(*jax.tree.leaves(block.offset))).eps)


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

    ``NotGaussian`` and not ``BayesmithError``. They are siblings, not
    ancestor and descendant, and only one of them is a verdict about an
    ordinary model: :mod:`bayesmith.dispatch.classify`'s own docstring states
    that ``StructureError`` is caught at exactly one site in this package --
    ``check_linearity``'s -- and allowed through everywhere else, because from
    ``check_gaussian`` it means a node whose ``log_prob`` contradicts the
    ``loc``/``scale`` read off it. Held at its centre and reported as "not
    Gaussian, so this module has no prior width", a broken model would print
    as an ordinary unsweepable one.
    """
    node = graph.node(name)
    try:
        loc, scale = gaussian_parts(graph, node, env)
    except NotGaussian:
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
            except NotGaussian:
                # NOT `BayesmithError`, which is the base class of both this
                # and `StructureError`. `StructureError` is a SIBLING of this
                # class, not a subclass, and it is the one exception this
                # package insists must propagate: `check_gaussian` raises it
                # for a scale that is not strictly positive and finite, and
                # `gaussian_parts` for an integer `loc`. Both are faults in
                # the declaration, and dropping the probe printed a narrowed
                # interval and a note instead. Measured, on a graph whose
                # member prior width is an unfloored `tau ~ N(1.0, 0.5)`:
                # the -3-sigma probe puts that width at -0.5, and under
                # `except BayesmithError` `compile()` returned an interval of
                # [15.2, 356.0] built from the three probes that survived,
                # with nothing anywhere saying the model declares a negative
                # prior width over a fifth of its own prior.
                #
                # Nothing in this package's fixture suite reaches this arm as
                # narrowed, and the reason is worth writing down rather than
                # rediscovering: `_kappa_at` raises `NotGaussian` only from a
                # DISTRIBUTION TYPE -- a member's or an observed node's -- and
                # `unchecked_operator`'s internal-ancestry refusal, and both
                # are properties of the graph rather than of the values it is
                # evaluated at, so both fire at the ANCHOR evaluation that
                # seeds `values` -- which is outside this `try` -- before any
                # probe is taken. Only a
                # `dist_fn` returning a different distribution CLASS at
                # different parent values could reach it. The arm a fixture
                # does reach is the non-finite one below.
                refused.append(name)
                continue
            if math.isfinite(measured):
                values.append(measured)
            else:
                refused.append(name)
    return min(values), max(values), _sweep_note(held, refused)


class Block(eqx.Module):
    """One group of latents and the method the graph selected for it.

    Attributes:
        latents: the members, sorted.
        method: ``"gcr"``, ``"gcr+snis"``, ``"gcr+mh"``, ``"log-gcr"`` or
            ``"nuts"``. FIVE values, from two producers:
            :class:`~bayesmith.dispatch.classify.Classification` chooses the
            first four, and :func:`~bayesmith.dispatch.factor.factor_partition`
            also emits ``"log-gcr"`` without going through
            :class:`Classification` at all. Neither ``_LABELS`` here nor
            ``FACTOR_METHODS`` there enumerates all five, and that is
            deliberate on both sides -- each says in its own docstring which
            row it leaves out and why -- so this attribute is the one place
            the union is written down.
        reason: why -- naming members on a refusal, plus whatever the kappa
            sweep could not reach.
        linearity: ``check_linearity``'s per-at-point errors, or ``None``.
        kappa: a single conditioning bound, or ``(lo, hi)`` when a latent
            outside the block moves it. ``None`` for a NUTS block, which is
            solved by no linear system at all.
        tol: ``CONVERGENCE_TARGET / kappa_upper(kappa)``.
        epsilon: machine epsilon of the dtype this block's arithmetic runs at,
            from :func:`working_epsilon`. ``None`` for a NUTS block, which
            solves nothing. Stored rather than re-derived at print time
            because it is a property of the GRAPH -- ``const`` and ``observe``
            capture their arrays at trace time -- and a plan is meant to be a
            record of what was measured, not a live query.
    """

    latents: tuple[str, ...] = eqx.field(static=True)
    method: str = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    linearity: dict | None = eqx.field(static=True, default=None)
    kappa: float | tuple[float, float] | None = eqx.field(static=True, default=None)
    tol: float | None = eqx.field(static=True, default=None)
    epsilon: float | None = eqx.field(static=True, default=None)

    @property
    def tol_attainable(self) -> bool:
        """Whether CG can reach ``tol`` at all in this block's own arithmetic.

        ``False`` when ``kappa_upper(kappa) * epsilon >= CONVERGENCE_TARGET``:
        the error a solve delivers is ``kappa`` times its relative residual,
        the residual cannot go below the arithmetic's own noise, and past this
        product the target is arithmetically out of reach whatever ``tol``
        says. It is the same test
        :func:`~bayesmith.exact.solve.wiener_solve` makes on
        ``require_convergence`` (``bound * epsilon > require_convergence``),
        against the target ``tol`` was derived from, so the plan's verdict and
        the solver's cannot disagree.

        **A statement about the guarantee, not about the answer.** The bound
        is an UPPER bound on kappa and a loose one -- measured at 3676x the
        true conditioning on ``two_linear_latents``, and 232,000x between the
        prior centre and the GLS fixed point on ``radiometer`` -- so a
        ``False`` here is compatible with a solve that is in fact accurate.
        Ten of this package's own fixtures read ``False`` at float32 and
        ``tests/dispatch/test_acceptance.py`` measures two of them
        (``prior_held_direction``, ``straight_line(prior_std=1e3,
        sigma=1e-2)``) delivering inside ``CONVERGENCE_TARGET`` anyway. That
        is exactly why this is printed rather than raised.

        ``True`` where there is nothing to solve, which is abstention rather
        than endorsement -- the same convention
        :attr:`~bayesmith.dispatch.execute.Posterior.unreliable` uses for a
        missing k-hat.
        """
        if self.kappa is None or self.epsilon is None:
            return True
        return kappa_upper(self.kappa) * self.epsilon < CONVERGENCE_TARGET


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


def _precision_note(block: Block) -> str:
    """What to say when ``tol`` is below what the arithmetic can deliver.

    Empty where :attr:`Block.tol_attainable` holds. Where it does not, this
    REPLACES ``"guard reachable, off by default"`` rather than joining it,
    because that clause's justification is precisely that ``tol`` already
    delivers ``CONVERGENCE_TARGET`` -- see :attr:`InferencePlan.guard_hoisted`
    -- and here it does not, so offering the guard as the remedy points at a
    keyword that would refuse the solve rather than rescue it. It JOINS
    ``"guard hoisted out of the sweep"``, which is a different statement and
    still true: a sweep whose guard cannot run AND whose ``tol`` cannot
    deliver is section 4.2's forbidden combination, and both halves of it are
    worth naming.

    The wording follows :func:`~bayesmith.exact.solve.wiener_solve`'s own
    refusal, which reaches the same verdict from the same product and already
    says "either the condition bound times the machine epsilon already
    exceeds the target ... Run the solve inside ``with jax.enable_x64(True):``,
    or strengthen the prior".
    """
    if block.tol_attainable:
        return ""
    reach = kappa_upper(block.kappa) * block.epsilon
    return (
        f"tol UNATTAINABLE at this dtype: the condition bound times the machine "
        f"epsilon is {reach:.3g}, already past the {CONVERGENCE_TARGET:g} target, "
        "so no tol delivers it -- the error is kappa times whatever residual the "
        "arithmetic allows. Run inside `with jax.enable_x64(True):`, building "
        "the graph there too, or strengthen the prior"
    )


def _continuation(block: Block, *, hoisted: bool, width: int) -> list[str]:
    """The block's remaining lines: the kappa/tol pair, then the reason.

    ``hoisted`` is a property of the PLAN, not of the block -- the in-sweep
    guard is hoisted exactly when there is a sweep to hoist it out of -- so it
    is passed in rather than stored. It is printed next to ``tol`` and not
    somewhere else because section 4.2's rule is about the two together: a
    ``tol`` with no statement of the guard is half of the pair the rule
    constrains.

    Whether that ``tol`` is REACHABLE is the third thing the pair needs, and
    :func:`_precision_note` is where it comes from.
    """
    lines: list[str] = []
    if block.kappa is not None and block.tol is not None:
        unattainable = _precision_note(block)
        if hoisted:
            guard = "guard hoisted out of the sweep"
        elif unattainable:
            guard = ""
        else:
            guard = "guard reachable, off by default (require_convergence=)"
        clauses = "; ".join(part for part in (guard, unattainable) if part)
        pair = f"{_kappa_text(block.kappa)} -> tol={block.tol:.8g}, {clauses}"
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
        streaming: whether the campaign fold applies to this graph, and on
            which plate -- see
            :mod:`~bayesmith.dispatch.streaming`. MEASURED here rather than
            offered as a live query, because a plan is a record of what was
            measured; ``None`` only on a plan built by hand.
    """

    graph: Graph
    blocks: tuple[Block, ...]
    sigma_needs_rebuild: bool = eqx.field(static=True, default=False)
    streaming: StreamingRoute | None = eqx.field(static=True, default=None)

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
        # Only when there IS a route. A graph that is not a campaign prints
        # byte-identically to what it printed before `streaming.py` existed,
        # which is what keeps the twenty existing assertions on `str(plan)`
        # measuring what they were written to measure.
        if self.streaming is not None and (note := self.streaming.line()):
            lines.append(note)
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
        nuts_on_collapse: bool = False,
    ) -> Posterior:
        """Run the plan. Section 6.4's dispatch, and nothing else decides.

        Five shapes, in the order this reads them: no exact block -> NUTS;
        exact block plus a sampled one -> ``HMCGibbs`` with the sweep the plan
        printed; whole graph exact with a fixed sigma -> iid GCR draws, no
        chain; whole graph exact with a moving sigma -> GCR at the GLS fixed
        point corrected by SNIS; and that last one again when its Kish ESS/N
        falls under ``ess_floor`` -> the same weighted sample, marked
        ``unreliable=True`` and saying so, or NUTS instead if
        ``nuts_on_collapse``. See :data:`SNIS_ESS_FLOOR` for the measurement
        that decided which of those two is the default.

        Args:
            key: PRNG key. Split once, so the draws and any fallback chain do
                not share a stream.
            num_samples, num_warmup, num_chains, chain_method, progress_bar,
                nuts_options: passed to whichever sampler runs -- ``HMCGibbs``
                through :func:`~bayesmith.exact.gibbs.assemble` on the mixed
                path, ``NUTS`` through
                :func:`~bayesmith.bridge.numpyro_bridge.nuts` on both bare
                ones, under the same six names in both. ``num_warmup`` is
                ignored on the iid path, there being nothing to adapt.
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
            ess_floor: the Kish ESS/N under which the SNIS path declares the
                correction collapsed.
            nuts_on_collapse: on a collapse, discard the weighted sample and
                sample the whole graph with NUTS instead. **Off by default,
                measured.** At the collapsed cell ``plated_radiometer(n=25,
                kappa=0.4)``, N=1200, against exact per-coordinate quadrature,
                the weighted answer's worst coordinate is 1.40 posterior sd
                from the truth and the NUTS replacement's is 18.5 -- while
                NUTS's chain ESS (33) exceeds the Kish ESS (14), so the
                diagnostic that fires prefers the worse answer. Substituting
                is therefore something to ask for, with the collapse already
                reported in ``unreliable`` and ``reason`` for a caller who
                wants to decide for themselves.

        Returns:
            A :class:`Posterior`, whose ``method`` is what RAN.
        """
        return run_sample(
            self,
            key,
            num_samples=num_samples,
            num_warmup=num_warmup,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=progress_bar,
            nuts_options=nuts_options,
            tol=tol,
            maxiter=maxiter,
            require_convergence=require_convergence,
            ess_floor=ess_floor,
            nuts_on_collapse=nuts_on_collapse,
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
        return run_estimate(
            self,
            tol=tol,
            maxiter=maxiter,
            reweight_tol=reweight_tol,
            min_reweights=min_reweights,
            max_reweights=max_reweights,
            require_convergence=require_convergence,
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
    derives ``tol`` from the worst of it, and reads the block's working
    precision so the plan can say whether that ``tol`` is reachable at all.
    Takes no samples and forms no matrix; costs
    ``1 + len(KAPPA_PROBE_SIGMAS) * outside`` power iterations, plus one
    further block build for :func:`working_epsilon`, on top of what
    ``partition`` already spends.

    Then :func:`~bayesmith.dispatch.streaming.streaming_route`, which costs
    one leakage probe per PLATE and nothing at all on a graph with none.
    Measured on the scalar campaign: 0.278 s at four epochs and 0.342 s at
    512, against this function's own 1.0-1.3 s -- flat in the campaign,
    because the probe is one ``jacfwd`` of the whole plate rather than one
    per epoch.

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
        epsilon = working_epsilon(
            graph, classification.exact, block_at(graph, classification.exact, env=env)
        )
        blocks.append(
            Block(
                latents=classification.exact,
                method=classification.method,
                reason=classification.reason + note,
                linearity=classification.linearity,
                kappa=kappa,
                epsilon=epsilon,
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
    return InferencePlan(
        graph,
        tuple(blocks),
        classification.sigma_needs_rebuild,
        streaming_route(graph),
    )
