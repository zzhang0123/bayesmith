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

**Nothing here samples and nothing here is jittable**, for the same reason
:mod:`bayesmith.dispatch.classify` is not: every measurement it takes runs on
concrete values at compile time.
"""

from __future__ import annotations

import math
import textwrap
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
from bayesmith.errors import BayesmithError
from bayesmith.exact.block import domain_centre, unchecked_operator
from bayesmith.exact.gaussian import gaussian_parts, node_shape
from bayesmith.exact.gls import sigma_from_graph
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
        guard = "guard hoisted out of the sweep" if hoisted else "convergence guard on"
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
        carry the discipline on its own there. A fully exact plan runs one
        solve outside any trace and keeps the guard.
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
