"""The square-root information form: a log-quadratic that cannot go indefinite.

A term is ``[R | z]`` with ``log L(x) = -0.5 ||R x - z||^2 + offset``, so the
information it carries is ``F = R^T R`` -- positive semi-definite by
construction rather than by hope. Three consequences the accumulation layer
depends on:

* **Rank deficiency is representable and cheap.** One epoch rarely constrains
  every global parameter; its ``R`` simply has fewer rows than columns. The
  equivalent statement in ``(F, b)`` form is ``F v = 0``, which survives
  addition but not the sequence of explicit Schur complements a filter needs.
* **The working condition number is the square root.**
  ``kappa(R) = sqrt(kappa(F))``, which is what keeps a thousand-epoch
  accumulation inside float64. Accumulating ``F`` directly and taking explicit
  Schur complements goes indefinite in float64 on a realistic near-degenerate
  campaign.
* **Accumulation is a QR.** Stacking two factors vertically and
  re-triangularising *is* the sum of the two quadratic forms, so
  order-invariance and associativity hold to roundoff by construction.

**Ported, not reinvented.** The migration spec's B11 row lists this
arithmetic among the kernels to preserve EXACTLY, having been checked line by
line by two reviews. ``tests/crosscheck/test_sqrtinfo_agrees.py`` compares
every operation here against ``rheplicant.inference.sqrtinfo`` on the same
inputs, so "preserved" is a measurement rather than an intention. What B11
rewrites is the layer ABOVE this -- which latents are global, per-epoch or
linked, and where that is decided; see this package's CHANGELOG,
which measures how much of it the graph can answer.

This module knows nothing about graphs, epochs or plans. That separation is
what lets the numerics be checked against a dense oracle without a model
anywhere near them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from bayesmith.errors import StructureError


def _size(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


class SqrtInfo(eqx.Module):
    """``log L(x) = -0.5 ||R x - z||^2 + offset`` over a flat, named vector.

    Attributes:
        factor: ``(r, n)`` array ``R``. ``r < n`` means the term constrains
            only an ``r``-dimensional subspace -- the normal case for a single
            epoch, not an error.
        target: ``(r,)`` array ``z``.
        offset: scalar; every part of the log-density that does not depend on
            the latents -- the residual chi-square about the storage origin
            and the (masked) Gaussian normalisation.
        names: the latents this term is over, in the order they are ravelled.
        shapes: each latent's shape, in the same order. ``()`` for a scalar.
    """

    factor: jax.Array
    target: jax.Array
    offset: jax.Array
    names: tuple[str, ...] = eqx.field(static=True)
    shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)

    def __check_init__(self) -> None:
        if self.factor.ndim != 2:
            raise StructureError(
                f"SqrtInfo.factor must be 2-D (r, n); got shape {self.factor.shape}."
            )
        if self.target.shape != (self.factor.shape[0],):
            raise StructureError(
                "SqrtInfo.target must have one entry per row of factor: factor is "
                f"{self.factor.shape}, target is {self.target.shape}."
            )
        if len(self.names) != len(self.shapes):
            raise StructureError(
                f"SqrtInfo has {len(self.names)} names but {len(self.shapes)} shapes."
            )
        if self.width != self.factor.shape[1]:
            raise StructureError(
                f"SqrtInfo.factor has {self.factor.shape[1]} columns but the named "
                f"latents {list(self.names)} ravel to {self.width} values."
            )
        complex_parts = [
            name
            for name, array in (
                ("factor", self.factor),
                ("target", self.target),
                ("offset", self.offset),
            )
            if jnp.iscomplexobj(array)
        ]
        if complex_parts:
            raise StructureError(
                f"SqrtInfo was given a complex {' and '.join(complex_parts)}, and "
                "this form is real by construction. Every quantity here is a "
                "BILINEAR form -- `log_prob` takes `sum(residual**2)`, which is "
                "`r^T r`, and `information()` takes `factor.T @ factor` with no "
                "conjugate -- while a complex QR's Q is UNITARY and preserves "
                "`r^H r` instead. So the two disagree, and they disagree "
                "silently: measured on one shared complex scalar with "
                "R_1 = [[1j]] and R_2 = [[1]], the summed information is exactly "
                "0 by hand and `combine` returns 2.0, an absolute error equal to "
                "the whole of the true value, with no exception raised. "
                "The way out is the one this package already takes everywhere "
                "else a complex latent meets real data: carry it as its REAL "
                "DEGREES OF FREEDOM. See `bayesmith.exact.block.real_parts`, "
                "whose docstring gives the reason -- every prediction here is "
                "real, so the map from complex coefficients to data is R-linear "
                "and not C-linear -- and `ComplexNormal`, which fixes the "
                "column convention for declaring one. Split before compressing; "
                "a term over 2n real columns is exact, and a term over n "
                "complex ones is not a term."
            )

    @property
    def width(self) -> int:
        """Number of columns the named latents ravel to."""
        return sum(_size(shape) for shape in self.shapes)

    def ravel(self, values: dict[str, Any]) -> jax.Array:
        """Flatten ``{name: array}`` into this term's column order."""
        parts = []
        for name, shape in zip(self.names, self.shapes, strict=True):
            if name not in values:
                raise StructureError(
                    f"This term is over {list(self.names)}; no value was given "
                    f"for {name!r}."
                )
            leaf = jnp.asarray(values[name])
            if leaf.shape != shape:
                raise StructureError(
                    f"Latent {name!r} has shape {shape} in this term but "
                    f"{leaf.shape} was supplied."
                )
            parts.append(jnp.ravel(leaf))
        return jnp.concatenate(parts) if parts else jnp.zeros(0)

    def log_prob(self, values: dict[str, Any]) -> jax.Array:
        """The log-density this term encodes, at the given latent values."""
        residual = self.factor @ self.ravel(values) - self.target
        return self.offset - 0.5 * jnp.sum(residual**2)

    def information(self) -> jax.Array:
        """``F = R^T R`` -- the information this term carries.

        May legitimately be singular: one epoch usually constrains only a
        subspace, and only the campaign total plus the prior need be positive
        definite. Named ``information`` rather than ``fisher`` because it is
        the observed information of a stored term, not a Fisher matrix of a
        model -- ``exact/fisher.py`` owns that word here.
        """
        return self.factor.T @ self.factor

    @classmethod
    def null(
        cls, names: tuple[str, ...], shapes: tuple[tuple[int, ...], ...]
    ) -> SqrtInfo:
        """A term that says nothing -- the identity of :meth:`combine`.

        Square rather than zero-row so an accumulator's pytree keeps a fixed
        treedef across a whole campaign, which is what stops ``jit`` retracing
        once per epoch.
        """
        width = sum(_size(shape) for shape in shapes)
        return cls(
            factor=jnp.zeros((width, width)),
            target=jnp.zeros(width),
            offset=jnp.zeros(()),
            names=names,
            shapes=shapes,
        )

    @classmethod
    def combine(cls, first: SqrtInfo, second: SqrtInfo) -> SqrtInfo:
        """The term whose log-density is the sum of the two given ones.

        Stack the augmented factors and re-triangularise. Writing
        ``y = [x; -1]`` so that ``[R | z] y = R x - z``, the stacked product
        has the same norm as its triangular factor because ``Q`` has
        orthonormal columns::

            ||R_a x - z_a||^2 + ||R_b x - z_b||^2 = ||R_tot x - z_tot||^2 + rho^2

        ``rho`` is the corner of the triangular factor -- the part of the two
        residuals that no single quadratic form in ``x`` can express. It is a
        constant, so it belongs in the offset; dropping it leaves every
        combined term wrong by an amount that grows with the campaign and is
        invisible in the posterior's SHAPE, which is why the tests here
        compare absolute log-densities.
        """
        if first.names != second.names or first.shapes != second.shapes:
            raise StructureError(
                "Cannot combine two terms over different latents: "
                f"{list(first.names)} vs {list(second.names)}. A ledger of terms "
                "declared against different parameter sets is not a likelihood."
            )
        width = first.factor.shape[1]
        stacked = jnp.concatenate(
            [
                jnp.concatenate([first.factor, first.target[:, None]], axis=1),
                jnp.concatenate([second.factor, second.target[:, None]], axis=1),
            ],
            axis=0,
        )
        upper = jnp.linalg.qr(stacked, mode="r")
        keep = min(upper.shape[0], width)
        corner = upper[keep:, width]
        return cls(
            factor=upper[:keep, :width],
            target=upper[:keep, width],
            offset=first.offset + second.offset - 0.5 * jnp.sum(corner**2),
            names=first.names,
            shapes=first.shapes,
        )


def marginalise_arrays(
    factor: jax.Array,
    target: jax.Array,
    offset: jax.Array,
    n_block: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The Schur complement in square-root form, with no Python control flow.

    The block is the **leading** ``n_block`` columns; permuting is the
    caller's job, because a caller that already knows its layout should not
    pay for a name lookup once per epoch inside a ``lax.scan``.

    This exists because :func:`marginalise` cannot be traced: it concretises
    twice, for the pivot scale and for the dtype's floor. A filter that
    evaluates this arithmetic inside a likelihood -- under ``scan``, and
    DIFFERENTIATED with respect to a transition's own parameters -- needs the
    half that traces. ``grad`` is the half that matters: a correlation time
    inferred rather than pinned is differentiated on every leapfrog step.

    **The refusal is not weakened, it is moved -- and this function cannot
    make it.** What :func:`marginalise` catches is a block that does not
    constrain itself, which makes the integral divergent. Under a trace that
    judgement is unavailable: it needs a comparison against a value, and there
    is no value. What this hands back instead is the evidence -- ``pivots``,
    as data -- so an eager caller can judge them and a traced one can carry
    them out.

    Args:
        factor: ``(r, n)`` array ``R``, block columns first.
        target: ``(r,)`` array ``z``.
        offset: scalar; the constant this term already carries.
        n_block: how many leading columns to integrate out. ``0`` is legal and
            is the identity on the density -- the re-triangularisation folds
            any excess rows into the corner and the offset absorbs it.

    Returns:
        ``(factor, target, offset, pivots)`` -- the retained form, the offset
        with the Gaussian integral's constant folded in, and ``|diag(R)|`` of
        the re-triangularisation so a checked caller can test it.
    """
    width = factor.shape[1]
    upper = jnp.linalg.qr(jnp.concatenate([factor, target[:, None]], axis=1), mode="r")
    keep = min(upper.shape[0], width)
    # The part of the residual no quadratic form in the retained columns can
    # express. A constant, so it belongs in the offset.
    corner = upper[keep:, width]
    pivots = jnp.abs(jnp.diag(upper))
    constant = (
        0.5 * n_block * math.log(2.0 * math.pi)
        - jnp.sum(jnp.log(pivots[:n_block]))
        - 0.5 * jnp.sum(corner**2)
    )
    return (
        upper[n_block:keep, n_block:width],
        upper[n_block:keep, width],
        offset + constant,
        pivots,
    )


def marginalise(info: SqrtInfo, block: Sequence[str]) -> SqrtInfo:
    """Integrate named latents out of a square-root information form, exactly.

    Permute the block's columns first, re-triangularise, and drop the leading
    rows and columns. That drop **is** the Schur complement, and the Gaussian
    integral over the block contributes exactly

    ``+ (n_block/2) log(2 pi)  -  sum log|R_bb,ii|  -  0.5 rho^2``

    and nothing else. In particular it does **not** contribute the block's own
    prior normalisation: whoever appended the prior rows owns
    ``-sum(log(std)) - (n/2) log(2 pi)``, and the two ``2 pi`` halves cancel
    while ``sum(log(std))`` has nothing to cancel against. rheplicant shipped
    that second term missing once -- 1.07 nats for three nuisances at
    ``std=0.7``, 27.47 for twenty-five at ``std=3``, and **exactly zero at**
    ``std=1``, which is how a probe built on unit priors passed. A constant is
    invisible in a posterior's shape, so the tests here compare ABSOLUTE
    log-densities and use a non-unit prior.

    Marginalising **every** name is legal and returns a zero-width term whose
    ``log_prob({})`` is the marginal likelihood; marginalising **nothing** is
    likewise legal and is the identity on the log-density.

    Args:
        info: the joint form, prior rows already appended by the caller.
        block: which names to integrate out.

    Returns:
        A term over ``info``'s remaining names, in their original relative
        order, whose log-density is the integral of ``info``'s over ``block``.

    Raises:
        StructureError: if a name is repeated or not in ``info``; if the
            re-triangularisation is not finite; or if the block is not
            constrained -- an unconstrained direction makes the integral
            divergent, and finite arithmetic returns a large plausible number
            for it rather than an infinity anyone would notice.
    """
    block = tuple(block)
    if len(set(block)) != len(block):
        raise StructureError(
            f"marginalise was given {list(block)}, which names a latent twice. "
            "Integrating the same block out twice is not defined. Pass each "
            "name once."
        )
    unknown = [name for name in block if name not in info.names]
    if unknown:
        raise StructureError(
            f"This term is not over {unknown}; it is over {list(info.names)}. "
            "Marginalise a name the term actually carries, or combine the terms "
            "that do carry it first."
        )

    spans: dict[str, range] = {}
    position = 0
    for name, shape in zip(info.names, info.shapes, strict=True):
        size = _size(shape)
        spans[name] = range(position, position + size)
        position += size
    kept = tuple(name for name in info.names if name not in block)
    columns = [column for name in block for column in spans[name]]
    columns += [column for name in kept for column in spans[name]]
    n_block = sum(len(spans[name]) for name in block)
    width = info.factor.shape[1]
    permuted = info.factor[:, jnp.asarray(columns, dtype=int)]

    if n_block and min(permuted.shape[0], width + 1) < n_block:
        # A shape fact, available before any arithmetic. `width + 1`, not
        # `width`, because the target column is part of the matrix being
        # factorised -- writing `width` would refuse a term with exactly
        # `width` independent rows and a full-rank block, which is the normal
        # case for a square padded epoch block.
        raise StructureError(
            f"This term has {min(permuted.shape[0], width + 1)} independent rows "
            f"but {n_block} columns in the block {list(block)}, so the block does "
            "not constrain itself and the integral over it diverges. A per-epoch "
            "latent is integrated exactly once, so its prior has to be part of "
            "the model rather than an optional regulariser: append the prior "
            "rows before marginalising."
        )

    factor, target, offset, pivots = marginalise_arrays(
        permuted, info.target, info.offset, n_block
    )

    if n_block:
        # Finiteness FIRST, and it is not defensive padding: the comparison
        # below is relative to `max(pivots)`, so a single `nan` or `inf`
        # anywhere in this term makes the threshold itself `nan` and every
        # comparison against it False. rheplicant measured that directly: a
        # `nan`-scaled block was ACCEPTED, with `offset` nan, past a
        # `__check_init__` that validates shapes only.
        #
        # BOTH ends, because a guard written NaN-safely can still be defeated
        # from the other one: `inf > 0` is True, so a threshold of `inf`
        # admits every pivot there is.
        if not bool(jnp.all(jnp.isfinite(pivots))):
            raise StructureError(
                "This term's re-triangularisation is not finite: its pivots are "
                f"{np.asarray(pivots)}, so the marginal over {list(block)} would "
                "come back as a SqrtInfo carrying nan -- which __check_init__ "
                "does not test for, and which loses every comparison a campaign "
                "audit could make about it. The stored factor or target already "
                "carried nan or inf before this call."
            )
        # Compared against the LARGEST pivot, not an absolute floor: the rows
        # are whitened data, so their scale is the epoch's 1/sigma and an
        # absolute threshold would refuse a well-constrained low-noise block
        # and wave through a badly-constrained high-noise one.
        #
        # These two lines are why this function cannot be traced, and why the
        # arithmetic above it can.
        scale = float(jnp.max(pivots))
        floor = float(np.sqrt(np.finfo(permuted.dtype).eps)) * scale
        # `not all(> floor)` rather than `any(<= floor)`: the two agree for
        # finite numbers and not for `nan`, which loses both comparisons.
        #
        # **This distinction is a SHAPE here, not a live guard, and the
        # mutation survives.** Measured: rewriting it as `any(<= floor)` is
        # killed by nothing in `tests/marginal` or `tests/crosscheck`, because
        # the finiteness check above already refuses every `nan` before this
        # line runs. It is written the strong way anyway -- this is the fourth
        # refusal in this family, and the previous three each needed it; a
        # weak one here is how the next reader copies the weak shape into a
        # place where it IS reachable. Recorded rather than papered over: a
        # test that reached this branch would have to defeat the check above
        # it, and a test of Python's `nan` comparison semantics would be
        # testing Python.
        if not bool(jnp.all(pivots[:n_block] > floor)):
            raise StructureError(
                f"The block {list(block)} does not constrain one of its own "
                "directions, so the Gaussian integral over it diverges and the "
                "marginal would come back as +inf -- finite arithmetic gives a "
                "large plausible number instead, which is worse, because nothing "
                "downstream tests for it. Give the block a proper prior and "
                "append its rows before marginalising."
            )

    return SqrtInfo(
        factor=factor,
        target=target,
        offset=offset,
        names=kept,
        shapes=tuple(info.shapes[info.names.index(name)] for name in kept),
    )
