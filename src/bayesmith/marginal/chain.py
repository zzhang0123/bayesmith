"""A nuisance that drifts across epochs, and the recursion that integrates it out.

An epoch-local nuisance is re-drawn every epoch and integrated away inside its
own. A LINKED one is not: it is a Markov chain, and treating it as epoch-local
marginalises a single physical fluctuation ``N`` times against independent
priors -- injecting information that is not there. This module is the exact
alternative (migration gap G3).

**The recursion.** Carry a joint square-root information factor over
``(theta, zeta_e)``. Fold in an epoch by stacking its rows and
re-triangularising. Advance by widening to ``(theta, zeta_e, zeta_{e+1})``,
appending the transition's rows, and marginalising ``zeta_e``: permute it
first, re-triangularise, drop row and column. **That drop IS the Schur
complement, in square root**, which is what keeps a thousand-epoch
accumulation inside float64 where the explicit ``(F, b)`` form goes indefinite.
``theta`` is never marginalised, so what comes back is ``log p(d_1:N | theta)``
exactly.

**Two sub-scopes, because "linear-Gaussian" is not enough.** An OU chain with
an INFERRED correlation time is still linear-Gaussian, so a caveat phrased
that way is satisfied while its claim fails: ``Q(theta)``, ``phi(theta)`` and
the Schur complement all become functions of theta, and a filter run once at
compression time pins them silently. The distinction lives in the TYPE. A
:class:`LinearGaussianTransition` holds numbers, and the theta posterior is
exact under filtering; a :class:`HyperTransition` holds a builder and is
resolved INSIDE the theta likelihood, so the whole recursion is a
differentiable ``lax.scan`` over the stored per-epoch blocks. One code path
serves both, because the recursion is traceable either way -- which is also
why the fixed case is validated by the same tests rather than by a second
implementation of the same arithmetic.

**The constant bookkeeping is not optional, and it is where this module can be
wrong while looking right.** Six constants reach the answer, and the
recursion's shape, gradient and curvature are all correct without any of them
-- so every test that checks a mean, a width or a derivative passes on a
version that has dropped one. Only a comparison against a dense joint density
notices. ``tests/marginal/test_chain.py`` builds that dense reference and
deletes each constant in turn to measure what it was worth.

Ported from ``rheplicant.inference.chain``. The containers a campaign stores
(a chain memory, its epoch bookkeeping) stay upstream, per the migration
ledger's D12 and the G6 enumeration; what is here is the recursion.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from bayesmith.errors import StructureError
from bayesmith.marginal.sqrtinfo import SqrtInfo, marginalise_arrays

__all__ = [
    "HyperTransition",
    "LinearGaussianTransition",
    "chain_log_likelihood",
    "chain_marginal",
    "ornstein_uhlenbeck",
    "smooth",
]


class LinearGaussianTransition(eqx.Module):
    """``zeta_{e+1} = phi zeta_e + w``, ``w ~ N(0, diag(process_std)^2)``.

    Attributes:
        phi: ``(n, n)``. A full matrix, because a multi-component drift can
            rotate; the process and initial spreads are diagonal because a
            correlated INNOVATION is a modelling claim nobody has made, and a
            silently-accepted full covariance would need its own Cholesky
            refusal.
        process_std: ``(n,)``, strictly positive.
        initial_std: ``(n,)``, strictly positive -- ``sd(zeta_1)``.
        initial_mean: ``(n,)``. Zero unless declared.
        hyper: empty for a fixed transition. Present so a caller can ask one
            question of either type.

    **Positivity is checked here and nowhere else, on purpose.** The rows this
    class contributes are what constrain every ``zeta_e``, so a strictly
    positive spread makes each marginalisation's block full-rank BY
    CONSTRUCTION -- and that is what lets the filter call
    :func:`~bayesmith.marginal.sqrtinfo.marginalise_arrays` inside a
    ``lax.scan`` instead of the checked
    :func:`~bayesmith.marginal.sqrtinfo.marginalise`, which concretises and
    therefore cannot be traced or differentiated. One eager check at
    declaration, not one traced check per epoch of a thousand.

    A TRACED spread is not checked, and cannot be: a :class:`HyperTransition`
    builds these from theta, and under NUTS theta goes wherever it likes.
    Parameterise the builder so positivity is structural -- return
    ``jnp.exp(log_sigma)``, never a raw sampled scale -- which is why this
    class takes standard deviations rather than a covariance.
    """

    phi: jax.Array
    process_std: jax.Array
    initial_std: jax.Array
    initial_mean: jax.Array
    hyper: tuple[str, ...] = eqx.field(static=True, default=())

    def __init__(
        self,
        phi: Any,
        process_std: Any,
        initial_std: Any,
        initial_mean: Any = None,
        hyper: Sequence[str] = (),
    ):
        self.process_std = jnp.atleast_1d(jnp.asarray(process_std))
        self.initial_std = jnp.atleast_1d(jnp.asarray(initial_std))
        self.phi = jnp.atleast_2d(jnp.asarray(phi))
        self.initial_mean = (
            jnp.zeros_like(self.initial_std)
            if initial_mean is None
            else jnp.broadcast_to(
                jnp.atleast_1d(jnp.asarray(initial_mean)), self.initial_std.shape
            )
        )
        self.hyper = tuple(hyper)

    def __check_init__(self) -> None:
        width = int(self.process_std.shape[0])
        if self.phi.shape != (width, width):
            raise StructureError(
                f"phi is {self.phi.shape} but process_std has {width} "
                "component(s), so the transition maps the chain into a space of "
                "a different size. Broadcasting one into the other would build a "
                "chain nobody declared."
            )
        if self.initial_std.shape != (width,):
            raise StructureError(
                f"initial_std is {self.initial_std.shape} but process_std is "
                f"{self.process_std.shape}; they describe the same chain."
            )
        for name, spread in (
            ("process_std", self.process_std),
            ("initial_std", self.initial_std),
        ):
            # A traced spread cannot be judged here and must not be pretended
            # about -- see the class docstring.
            if isinstance(spread, jax.core.Tracer):
                continue
            # `not (... > 0)` rather than `... <= 0`, because every comparison
            # against NaN is False and the second form waves a NaN spread
            # through. `isfinite` is separate for the same reason: `inf > 0` is
            # True, and an infinite spread makes `1 / process_std` zero, which
            # is a transition row of zeros -- the density comes back -inf from
            # the log-determinant, a thousand epochs after the declaration that
            # caused it.
            if not bool(jnp.all(jnp.isfinite(spread) & (spread > 0.0))):
                raise StructureError(
                    f"{name} must be finite and strictly positive; got {spread}. "
                    "These rows are what constrain zeta at each marginalisation, "
                    "so a zero leaves the Gaussian integral over that epoch "
                    "divergent -- and inside a lax.scan finite arithmetic returns "
                    "a large plausible number for it rather than an infinity "
                    "anyone would notice. A chain that genuinely does not move is "
                    "process_std=1e-9, not 0.0; one that is effectively unlinked "
                    "is 1e12, not inf, which would zero the transition rows and "
                    "send the whole campaign's density to -inf."
                )

    @property
    def width(self) -> int:
        """How many components the chain carries."""
        return int(self.process_std.shape[0])

    def at(self, values: dict[str, jax.Array]) -> LinearGaussianTransition:
        """Itself. A fixed transition does not depend on theta -- that is the
        claim it is making, and the reason its posterior is exact."""
        return self


def ornstein_uhlenbeck(
    tau: Any, sigma: Any, width: int = 1, hyper: Sequence[str] = ()
) -> LinearGaussianTransition:
    """A stationary OU chain: correlation time ``tau`` in epochs, spread ``sigma``.

    ``phi = exp(-1/tau)`` and ``process_std = sigma sqrt(1 - phi^2)``, so
    ``var(zeta_{e+1}) = phi^2 var + Q`` returns ``sigma^2`` when it starts
    there -- **stationarity is arithmetic here, not an assumption**, and a test
    pins it.

    A FUNCTION, not a class: the type the filter consumes -- and the type a
    :class:`HyperTransition` builder must return -- is
    :class:`LinearGaussianTransition`. An OU is a way of constructing one, and
    this package spells constructors in lower case.
    """
    phi_scalar = jnp.exp(-1.0 / jnp.asarray(tau))
    spread = jnp.asarray(sigma)
    return LinearGaussianTransition(
        phi=phi_scalar * jnp.eye(width),
        process_std=jnp.broadcast_to(
            spread * jnp.sqrt(1.0 - phi_scalar**2), (width,)
        ),
        initial_std=jnp.broadcast_to(spread, (width,)),
        hyper=hyper,
    )


class HyperTransition(eqx.Module):
    """A transition whose blocks are FUNCTIONS of theta.

    The distinction the module docstring argues for, as a type. ``build`` is
    resolved inside the theta likelihood, on every evaluation, so a correlation
    time that is inferred rather than pinned is differentiated on every
    leapfrog step. That is why the recursion is a ``lax.scan`` over stored
    blocks rather than a filter run once at compression time: the latter would
    pin ``Q(theta)`` and ``phi(theta)`` silently, and the posterior it produced
    would be exact for a model nobody declared.

    Attributes:
        build: ``{name: value} -> LinearGaussianTransition``. Positivity of
            the spreads it returns is the BUILDER's responsibility and cannot
            be checked here -- see :class:`LinearGaussianTransition`.
        width: how many components the chain carries. Declared rather than
            derived, because deriving it would mean calling ``build`` at
            construction time with values that may not exist yet.
        hyper: the latents ``build`` reads.
    """

    build: Callable[[dict[str, jax.Array]], LinearGaussianTransition] = eqx.field(
        static=True
    )
    width: int = eqx.field(static=True)
    hyper: tuple[str, ...] = eqx.field(static=True, default=())

    def at(self, values: dict[str, jax.Array]) -> LinearGaussianTransition:
        """The transition at these values."""
        return self.build(values)


def _initial_log_norm(transition: LinearGaussianTransition) -> jax.Array:
    """``-0.5 logdet(2 pi P0)`` -- the prior on ``zeta_1``.

    A module-level function rather than three inline terms so that a test can
    delete exactly this constant and measure what it was worth. It belongs to
    nobody else: the per-epoch blocks know nothing about the chain, and
    :func:`~bayesmith.marginal.sqrtinfo.marginalise_arrays` carries only the
    integral's own constant.
    """
    return -0.5 * transition.width * jnp.log(2.0 * jnp.pi) - jnp.sum(
        jnp.log(transition.initial_std)
    )


def _transition_log_norm(transition: LinearGaussianTransition) -> jax.Array:
    """``-0.5 logdet(2 pi Q)`` -- one per augmentation.

    **The ``2 pi`` half is not optional**, and the reason it reads as though it
    were is that it cancels against the marginalisation's ``+0.5 n log(2 pi)``.
    Keeping only the log-determinant while calling ``marginalise_arrays``
    leaves ``+0.5 log(2 pi)`` per transition -- which has no effect on any
    posterior mean, width or gradient, and is simply a wrong evidence.
    """
    return -0.5 * transition.width * jnp.log(2.0 * jnp.pi) - jnp.sum(
        jnp.log(transition.process_std)
    )


def _fold(
    factor: jax.Array,
    target: jax.Array,
    offset: jax.Array,
    block: tuple[jax.Array, jax.Array, jax.Array],
    width: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Add one epoch's evidence to the running joint form.

    ``SqrtInfo.combine``'s arithmetic on raw arrays, because a ``lax.scan``
    carry cannot afford a name lookup and a re-validation per epoch. **The
    corner is the largest of the six constants** and it grows with the
    campaign: it is the part of the stacked residual that no choice of the
    latents can remove, and dropping it loses a chi-square per epoch.
    """
    block_factor, block_target, block_offset = block
    upper = jnp.linalg.qr(
        jnp.concatenate(
            [
                jnp.concatenate([factor, target[:, None]], axis=1),
                jnp.concatenate([block_factor, block_target[:, None]], axis=1),
            ],
            axis=0,
        ),
        mode="r",
    )
    keep = min(upper.shape[0], width)
    corner = upper[keep:, width]
    return (
        upper[:keep, :width],
        upper[:keep, width],
        offset + block_offset - 0.5 * jnp.sum(corner**2),
    )


def _check_block_width(
    factors: jax.Array, names: tuple[str, ...], n_theta: int, n_zeta: int
) -> int:
    """How wide the stored blocks are, checked against what is claimed of them.

    One copy, called by the filter and by the smoother, because both slice at
    ``n_theta`` to separate theta's columns from the chain's: a block of the
    wrong width makes that slice take one for the other, and what comes back is
    finite, plausible, and a quantity nothing downstream re-derives.
    """
    width = n_theta + n_zeta
    if factors.shape[-1] != width:
        raise StructureError(
            f"The stored blocks are {factors.shape[-1]} columns wide but "
            f"{list(names)} plus a width-{n_zeta} chain is {width}. The blocks "
            "are a quadratic form in a specific ordered vector; reading them "
            "against a different one is not a rename, it is a different model."
        )
    return width


def _plan(
    blocks: tuple[jax.Array, jax.Array, jax.Array],
    transition: Any,
    values: dict[str, jax.Array],
    names: tuple[str, ...],
    shapes: tuple[tuple[int, ...], ...],
):
    """Everything both the filter and the smoother resolve the same way.

    One copy, because the two must agree about the block width, the column
    order and the transition's rows -- and a second spelling of any of those is
    a smoother describing a different chain from the filter's.
    """
    factors, _, _ = blocks
    resolved = transition.at(values)
    n_zeta = resolved.width
    n_theta = sum(int(jnp.zeros(shape).size) for shape in shapes)
    width = _check_block_width(factors, names, n_theta, n_zeta)
    inverse_process = 1.0 / resolved.process_std
    inverse_initial = 1.0 / resolved.initial_std
    # `Q^-1/2 (zeta_{e+1} - phi zeta_e)`, with zero response to theta: the
    # chain's locality written into the rows rather than assumed. The scaling
    # is `diag(1/q) @ phi`, NOT `phi @ diag(1/q)`; the two coincide for a
    # scalar chain, which is why a wide chain is what tests the ordering.
    transition_rows = jnp.concatenate(
        [
            jnp.zeros((n_zeta, n_theta)),
            -inverse_process[:, None] * resolved.phi,
            jnp.diag(inverse_process),
        ],
        axis=1,
    )
    return resolved, n_theta, n_zeta, width, inverse_initial, transition_rows


def chain_marginal(
    blocks: tuple[jax.Array, jax.Array, jax.Array],
    transition: Any,
    values: dict[str, jax.Array],
    names: tuple[str, ...],
    shapes: tuple[tuple[int, ...], ...],
) -> SqrtInfo:
    """``zeta_1:N`` integrated out exactly, leaving a quadratic form in theta.

    Args:
        blocks: ``(factor (N, w, w), target (N, w), offset (N,))`` -- one
            square per-epoch joint form over ``(*names, zeta)``, with
            ``zeta``'s columns **last**. Square rather than ragged because
            ``lax.scan`` needs one shape per iteration.
        transition: a :class:`LinearGaussianTransition` or a
            :class:`HyperTransition`. One code path serves both; ``.at(values)``
            is where they differ.
        values: theta, for a hyper transition to resolve against. A fixed one
            ignores it.
        names, shapes: theta's layout, as the returned :class:`SqrtInfo` will
            carry it.

    Returns:
        A :class:`SqrtInfo` over ``names`` alone. Its ``offset`` carries every
        constant of the integral, which is what makes it an EVIDENCE rather
        than a shape.

    **Why the scan stops one short.** Scanning over all ``N`` blocks and then
    marginalising once more integrates a ``zeta_{N+1}`` no data constrained,
    and comes back exact anyway -- the extra transition density integrates to
    one over its own argument, so its normalisation and the extra
    marginalisation's constant cancel term for term. The reason to stop one
    short is COST, one QR per call, not correctness. What it does change is the
    COUNT of each constant, which is what the constant tests notice and the
    exactness tests cannot.
    """
    factors, targets, offsets = blocks
    resolved, n_theta, n_zeta, width, inverse_initial, transition_rows = _plan(
        blocks, transition, values, names, shapes
    )

    # zeta_1's prior, and nothing else: theta's prior lives elsewhere, and a
    # stored per-epoch factor is prior-free in theta by construction.
    carry_factor = (
        jnp.zeros((width, width)).at[n_theta:, n_theta:].set(jnp.diag(inverse_initial))
    )
    carry_target = (
        jnp.zeros(width).at[n_theta:].set(inverse_initial * resolved.initial_mean)
    )
    carry_offset = _initial_log_norm(resolved)

    # `[zeta_e | theta | zeta_{e+1}]`: marginalise_arrays takes the block first.
    augment_order = jnp.asarray(
        list(range(n_theta, width))
        + list(range(n_theta))
        + list(range(width, width + n_zeta)),
        dtype=int,
    )
    final_order = jnp.asarray(
        list(range(n_theta, width)) + list(range(n_theta)), dtype=int
    )
    transition_constant = _transition_log_norm(resolved)

    def step(carry, block):
        factor, target, offset = _fold(*carry, block, width)
        widened = jnp.concatenate(
            [factor, jnp.zeros((factor.shape[0], n_zeta))], axis=1
        )
        joint = jnp.concatenate([widened, transition_rows], axis=0)[:, augment_order]
        joint_target = jnp.concatenate([target, jnp.zeros(n_zeta)])
        factor, target, offset, _ = marginalise_arrays(
            joint, joint_target, offset + transition_constant, n_zeta
        )
        return (factor, target, offset), None

    # Every epoch but the last is folded and then advanced; the last is folded
    # and then integrated out, because it has no successor to hand the chain to.
    (factor, target, offset), _ = jax.lax.scan(
        step,
        (carry_factor, carry_target, carry_offset),
        (factors[:-1], targets[:-1], offsets[:-1]),
    )
    factor, target, offset = _fold(
        factor, target, offset, (factors[-1], targets[-1], offsets[-1]), width
    )
    factor, target, offset, _ = marginalise_arrays(
        factor[:, final_order], target, offset, n_zeta
    )
    return SqrtInfo(
        factor=factor, target=target, offset=offset, names=names, shapes=shapes
    )


def chain_log_likelihood(
    blocks: tuple[jax.Array, jax.Array, jax.Array],
    transition: Any,
    values: dict[str, jax.Array],
    names: tuple[str, ...],
    shapes: tuple[tuple[int, ...], ...],
) -> jax.Array:
    """``log p(d_1:N | theta)``, the chain integrated out exactly. No prior."""
    return chain_marginal(blocks, transition, values, names, shapes).log_prob(values)


def _zeta_joint(
    blocks: tuple[jax.Array, jax.Array, jax.Array],
    transition: Any,
    values: dict[str, jax.Array],
    names: tuple[str, ...],
    shapes: tuple[tuple[int, ...], ...],
) -> tuple[jax.Array, jax.Array, int, int]:
    """The block-tridiagonal joint over ``zeta_1:N``, as a SQUARE ROOT.

    Three kinds of row and nothing else: each epoch's stored rows with ``theta``
    moved to the right-hand side, ``zeta_1``'s prior, and one coupling
    ``Q^-1/2 (zeta_{e+1} - phi zeta_e)`` per transition. The offsets are not
    read -- a constant cannot move a mean or a covariance -- which is why this
    returns no offset and :func:`smooth` reports no density.

    **Why rows and a QR rather than the precision and an inverse.** The two are
    the same quantity and not the same arithmetic. Assembling ``F = R^T R``
    squares the condition number, and this module's own header says what that
    costs: the square-root form is what keeps a long accumulation inside
    float64 where the explicit ``(F, b)`` form goes indefinite. The rows carry
    ``1 / process_std``; the precision would carry its square, so a chain at
    ``process_std = 1e-9`` asks float64 to hold ``1e18``, and ``1e18 * eps`` is
    ``220``.

    That was not hypothetical. Before this assembly existed, ``smooth`` inverted
    the precision, and on a frozen chain (``phi = 1``, ``process_std`` falling,
    so every epoch shares one latent and the answer must converge) it returned
    a smoothed mean that walked from ``-0.200652`` to ``-0.469638`` between
    ``1e-6`` and ``1e-8`` -- while the across-epoch spread read ``7.2e-16``, so
    the answer *looked* settled -- and ``nan`` at ``1e-9``.
    ``tests/marginal/test_chain_conditioning.py`` is that story as tests.

    The QR always has at least ``T + 1`` rows to work with: the assembled matrix
    has ``N n_theta + 2 N n_zeta`` of them against ``T = N n_zeta`` columns, so
    the slices below never under-run, for any ``N >= 1``.

    Returns:
        ``(triangular (T, T), rhs (T,), n_epochs, n_zeta)``.
    """
    factors, targets, _ = blocks
    resolved = transition.at(values)
    n_zeta = resolved.width
    n_theta = sum(int(jnp.zeros(shape).size) for shape in shapes)
    _check_block_width(factors, names, n_theta, n_zeta)
    n_epochs = int(factors.shape[0])
    total = n_epochs * n_zeta
    theta = (
        jnp.concatenate([jnp.ravel(jnp.asarray(values[name])) for name in names])
        if names
        else jnp.zeros(0)
    )

    rows, rhs = [], []
    # Each epoch's evidence. theta is CONDITIONED on, not marginalised: it moves
    # to the right-hand side rather than becoming more columns.
    for e in range(n_epochs):
        rows.append(
            jnp.zeros((factors.shape[1], total))
            .at[:, e * n_zeta : (e + 1) * n_zeta]
            .set(factors[e][:, n_theta:])
        )
        rhs.append(targets[e] - factors[e][:, :n_theta] @ theta)

    # zeta_1's prior.
    rows.append(
        jnp.zeros((n_zeta, total))
        .at[:, :n_zeta]
        .set(jnp.diag(1.0 / resolved.initial_std))
    )
    rhs.append(resolved.initial_mean / resolved.initial_std)

    # The couplings. `diag(1/q) @ phi`, NOT `phi @ diag(1/q)` -- the same line
    # the filter's transition rows are built from, and the same one that no
    # scalar and no equal-spread fixture can tell apart.
    inverse_process = 1.0 / resolved.process_std
    for e in range(n_epochs - 1):
        coupling = jnp.zeros((n_zeta, total))
        coupling = coupling.at[:, e * n_zeta : (e + 1) * n_zeta].set(
            -inverse_process[:, None] * resolved.phi
        )
        coupling = coupling.at[:, (e + 1) * n_zeta : (e + 2) * n_zeta].set(
            jnp.diag(inverse_process)
        )
        rows.append(coupling)
        rhs.append(jnp.zeros(n_zeta))

    upper = jnp.linalg.qr(
        jnp.concatenate(
            [jnp.concatenate(rows, axis=0), jnp.concatenate(rhs)[:, None]], axis=1
        ),
        mode="r",
    )
    return upper[:total, :total], upper[:total, total], n_epochs, n_zeta


def smooth(
    blocks: tuple[jax.Array, jax.Array, jax.Array],
    transition: Any,
    values: dict[str, jax.Array],
    names: tuple[str, ...],
    shapes: tuple[tuple[int, ...], ...],
) -> tuple[jax.Array, jax.Array]:
    """``p(zeta_e | d_1:N, theta)`` for every epoch -- mean and variance.

    ``theta`` is **conditioned on**, not marginalised: the question a smoother
    answers is "given this receiver model, what did the drift do?", and
    marginalising theta would answer a different one with the same shapes.

    **Not the classical backward pass, and the same quantity.** The joint form
    over ``zeta_1:N`` given theta is block-tridiagonal and small enough to
    assemble: ``N * width`` is the number of epochs times the chain's width, not
    the data size. Assembling and solving it gives the exact smoothed mean and
    marginal variances in one step, with no forward/backward pair to keep
    consistent -- and the two halves of an RTS smoother disagreeing is a defect
    that reads as a physical result.

    **Assembled as a square root, and :func:`_zeta_joint` says why at length.**
    In short: the precision would carry ``(1 / process_std) ** 2``, and a stiff
    chain then asks float64 to hold a number whose product with ``eps`` is not
    small. This function used to form it that way and returned confidently
    wrong answers on chains this package documents as supported.

    Returns:
        ``(mean (N, width), variance (N, width))`` -- the smoothed marginal of
        each epoch's chain state.
    """
    triangular, rhs, epochs, n_zeta = _zeta_joint(
        blocks, transition, values, names, shapes
    )
    size = epochs * n_zeta
    mean = jax.scipy.linalg.solve_triangular(triangular, rhs, lower=False)
    inverse = jax.scipy.linalg.solve_triangular(triangular, jnp.eye(size), lower=False)
    # var = diag((R^T R)^-1) = the row norms of R^-1, without forming R^-1 R^-T.
    variance = jnp.sum(inverse**2, axis=1)
    return jnp.reshape(mean, (epochs, n_zeta)), jnp.reshape(variance, (epochs, n_zeta))
