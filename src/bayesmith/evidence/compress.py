"""One epoch of a linear-Gaussian model, compressed into a ``[R | z]`` term.

The bridge between B9's noise interface and B11's kernel. For

    d ~ N(A x + c, N)

the log-likelihood is exactly a square-root information term::

    R = N^-1/2 A        z = N^-1/2 (d - c)
    log p(d | x) = -1/2 ||R x - z||^2  -  1/2 log det (2 pi N)

so :func:`compress` is a whitening and a normalisation, and everything after
it is :mod:`bayesmith.evidence.sqrtinfo`'s arithmetic.

**This module owns the "unobserved sample" concept, and that is B11's first
design decision made rather than assumed.** ``sigma = inf`` means a sample was
not taken. :meth:`~bayesmith.exact.precision.Precision.whiten` already gives
it weight zero with no special case, so the QUADRATIC half needs nothing --
but ``log_normalizer()`` is ``+inf`` there, and it is right to be: a sample
with infinite variance has no density, which is the honest answer to the
question a ``Precision`` is asked. Reading it as ``0`` is a statement that the
sample is UNOBSERVED, which is a modelling concept this layer has and the
interface does not. So the mask lives here, and ``precision.py`` keeps a
normaliser that is never silently wrong.

**Masking is a DIAGONAL concept, and that is measured rather than assumed.**
For a stationary covariance the observed submatrix is neither circulant nor a
subset of the spectrum -- on a 6-point kernel with one sample dropped, its
log-determinant is ``-0.7084`` while the closest subset sum of log-eigenvalues
is ``0.47`` nats away. So an unobserved sample inside a correlated epoch is
refused rather than approximated; see :func:`compress`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from bayesmith.errors import StructureError
from bayesmith.evidence.sqrtinfo import SqrtInfo, marginalise
from bayesmith.exact.precision import per_sample_sigma


def observed_mask(precision: Any) -> jax.Array | None:
    """Which samples this covariance says were taken, or ``None``.

    ``None`` when the question does not apply -- a covariance with no
    per-sample sigma has no per-sample "was this taken", and
    :func:`compress` refuses to guess one. Otherwise a boolean array, ``True``
    where ``sigma`` is finite.

    Reads through :func:`~bayesmith.exact.precision.per_sample_sigma`, which
    is the one place in the package that asks which implementation a
    ``Precision`` is, rather than adding a second.
    """
    sigma = per_sample_sigma({"_": precision})
    if sigma is None:
        return None
    return jnp.isfinite(sigma["_"])


def compress(
    design: Mapping[str, jax.Array],
    data: jax.Array,
    precision: Any,
    shapes: Mapping[str, tuple[int, ...]],
    *,
    offset_prediction: jax.Array | None = None,
) -> SqrtInfo:
    """One epoch's likelihood as a sufficient statistic over the named latents.

    Args:
        design: ``{latent: (n_data, n_i)}`` blocks. **Column order follows
            this mapping's iteration order**, which becomes the term's latent
            order -- the same convention rheplicant's compressor uses, and the
            one :meth:`~bayesmith.evidence.sqrtinfo.SqrtInfo.combine` needs
            two terms to agree on.
        data: ``(n_data,)`` this epoch's observations.
        precision: the epoch's ``N^-1``, from
            :func:`~bayesmith.exact.gaussian.precision_at` or built directly.
        shapes: ``{latent: shape}``, so the term can ravel a value dict back.
        offset_prediction: the part of the prediction that does not depend on
            the latents, subtracted from ``data`` before whitening. ``None``
            means zero.

    Returns:
        A :class:`~bayesmith.evidence.sqrtinfo.SqrtInfo` whose ``log_prob``
        is this epoch's exact Gaussian log-likelihood -- **normalisation
        included**, because that constant is what a campaign's evidence is
        made of and dropping it is invisible in every posterior.

    Raises:
        StructureError: if a design block's rows do not match the data; if a
            block's columns do not match its declared shape; or if the
            covariance is correlated AND reports an unobserved sample, which
            has no exact meaning -- the observed submatrix of a stationary
            covariance is not stationary.
    """
    names = tuple(design)
    if not names:
        raise StructureError(
            "compress was given no design blocks, so the resulting term would "
            "be over no latents and could not be combined with anything. Pass "
            "at least one {latent: block}."
        )
    size = int(jnp.shape(data)[0])
    for name in names:
        block = jnp.asarray(design[name])
        if block.ndim != 2 or block.shape[0] != size:
            raise StructureError(
                f"design[{name!r}] has shape {block.shape}; it must be "
                f"(n_data, n_i) with n_data = {size} to match the data."
            )
        expected = 1
        for dim in shapes[name]:
            expected *= int(dim)
        if block.shape[1] != expected:
            raise StructureError(
                f"design[{name!r}] has {block.shape[1]} columns but its shape "
                f"{shapes[name]} ravels to {expected} values."
            )

    seen = observed_mask(precision)
    if seen is not None:
        # The masked normaliser, ALWAYS for a per-sample sigma -- not only
        # when something is actually masked. With every sample seen it is
        # BITWISE `log_normalizer()`, so there is nothing to gain by branching
        # on the values and something to lose: the branch was the one
        # concretisation in this function, and without it the whole thing
        # traces. `epoch_terms` vmaps it over a campaign.
        #
        # Only a per-sample sigma can say a sample was not taken, and only
        # then is the normaliser a subset sum, so the remaining branch is a
        # question about the TYPE -- static, and safe under a trace.
        sigma = per_sample_sigma({"_": precision})["_"]
        safe = jnp.where(seen, sigma, 1.0)
        offset = -0.5 * jnp.sum(
            jnp.where(seen, jnp.log(2.0 * math.pi * safe**2), 0.0)
        )
    else:
        offset = -0.5 * precision.log_normalizer()
        if not bool(jnp.isfinite(offset)):
            # A correlated covariance with a non-finite normaliser. Refused
            # rather than masked: there is no exact masked normaliser to fall
            # back on. Measured on a 6-point kernel with one sample dropped,
            # the observed submatrix's log-determinant is -0.7084 and the
            # closest subset sum of log-eigenvalues is 0.47 nats away -- so
            # any "mask the spectrum" rule would be an approximation wearing
            # an exact result's name, and this layer's whole job is the
            # constant.
            raise StructureError(
                "this epoch's covariance has a non-finite log-normaliser and no "
                "per-sample sigma to mask, so its evidence is not defined. An "
                "unobserved sample inside a CORRELATED epoch has no exact "
                "meaning: the observed submatrix of a stationary covariance is "
                "not itself stationary, and its log-determinant is not a subset "
                "sum of the spectrum. Split the epoch at the gap, or declare "
                "the covariance over the samples that were actually taken."
            )

    residual = jnp.asarray(data)
    if offset_prediction is not None:
        residual = residual - jnp.asarray(offset_prediction)

    # Whitened COLUMN BY COLUMN, not by broadcasting the whole matrix: a
    # diagonal implementation would be right by accident either way, while a
    # circulant one FFTs along the last axis and would silently whiten the
    # wrong direction. `exact/fisher.py::_weighted_design` makes the same
    # choice for the same reason.
    stacked = jnp.concatenate([jnp.asarray(design[name]) for name in names], axis=1)
    factor = jax.vmap(precision.whiten, in_axes=1, out_axes=1)(stacked)
    return SqrtInfo(
        factor=factor,
        target=precision.whiten(residual),
        offset=jnp.asarray(offset),
        names=names,
        shapes=tuple(tuple(shapes[name]) for name in names),
    )
def nuisance_prior(
    names: tuple[str, ...],
    shapes: Mapping[str, tuple[int, ...]],
    prior_std: Mapping[str, Any],
    prior_mean: Mapping[str, Any] | None,
    over: tuple[str, ...],
    over_shapes: Mapping[str, tuple[int, ...]],
) -> SqrtInfo:
    """``N(x_n | m_n, S_n)`` as a ``[R | z]`` term over the FULL column set.

    Rows ``I/std`` in the nuisance columns, zero everywhere else, target
    ``m_n/std``, and the offset the prior's own normalisation:
    ``-sum(log std) - (n/2) log 2 pi``.

    **That offset is the caller's, and this is the caller.**
    :func:`~bayesmith.evidence.sqrtinfo.marginalise` contributes
    ``+(n/2) log 2 pi - sum log|R_bb,ii| - 1/2 rho^2`` and deliberately not
    the prior's normalisation -- the two ``2 pi`` halves cancel and
    ``sum(log std)`` has nothing to cancel against, which is how rheplicant
    shipped it missing and why a unit-prior fixture could not see it.

    Emitted over the full column set rather than over the nuisances alone so
    it can be :meth:`~bayesmith.evidence.sqrtinfo.SqrtInfo.combine`d with the
    epoch's own term, which is what puts the prior rows in BEFORE the
    marginalisation rather than after.
    """
    order = names + over
    all_shapes = {**dict(shapes), **dict(over_shapes)}
    widths = {name: _ravelled(all_shapes[name]) for name in order}
    total = sum(widths.values())
    rows, targets = [], []
    position = 0
    for name in order:
        if name in names:
            std = jnp.broadcast_to(jnp.asarray(prior_std[name]), (widths[name],))
            mean = (
                jnp.zeros(widths[name])
                if prior_mean is None or name not in prior_mean
                else jnp.reshape(jnp.asarray(prior_mean[name]), (widths[name],))
            )
            block = jnp.zeros((widths[name], total))
            block = block.at[
                :, position : position + widths[name]
            ].set(jnp.diag(1.0 / std))
            rows.append(block)
            targets.append(mean / std)
        position += widths[name]
    factor = jnp.concatenate(rows, axis=0) if rows else jnp.zeros((0, total))
    target = jnp.concatenate(targets) if targets else jnp.zeros(0)
    size = int(factor.shape[0])
    offset = -jnp.sum(jnp.log(jnp.concatenate(
        [
            jnp.broadcast_to(jnp.asarray(prior_std[name]), (widths[name],))
            for name in names
        ]
    ))) - 0.5 * size * math.log(2.0 * math.pi)
    return SqrtInfo(
        factor=factor,
        target=target,
        offset=jnp.asarray(offset),
        names=order,
        shapes=tuple(tuple(all_shapes[name]) for name in order),
    )


def epoch_joint(
    global_design: Mapping[str, jax.Array],
    data: jax.Array,
    precision: Any,
    global_shapes: Mapping[str, tuple[int, ...]],
    *,
    nuisance_design: Mapping[str, jax.Array] | None = None,
    nuisance_shapes: Mapping[str, tuple[int, ...]] | None = None,
    nuisance_prior_std: Mapping[str, Any] | None = None,
    nuisance_prior_mean: Mapping[str, Any] | None = None,
    offset_prediction: jax.Array | None = None,
) -> tuple[SqrtInfo, tuple[str, ...]]:
    """The epoch's term over BOTH sets, prior rows appended -- before integrating.

    Split out of :func:`compress_epoch` so the assembly exists once and the
    two marginalisers can both use it. ``compress_epoch`` passes the result
    to :func:`~bayesmith.evidence.sqrtinfo.marginalise`, which checks and
    cannot be traced; a campaign passes it to
    :func:`~bayesmith.evidence.sqrtinfo.marginalise_arrays`, which traces and
    hands back ``pivots`` for the caller to check on every epoch at once.
    The same split as ``marginalise``/``marginalise_arrays`` one level down,
    and for the same reason.

    **The nuisances are the LEADING columns**, which is what lets a traced
    caller skip the name permutation entirely.

    Returns:
        ``(joint term, nuisance names)``.
    """
    nuisances = tuple(nuisance_design or ())
    joint = compress(
        {**dict(nuisance_design or {}), **dict(global_design)},
        data,
        precision,
        {**dict(nuisance_shapes or {}), **dict(global_shapes)},
        offset_prediction=offset_prediction,
    )
    if not nuisances:
        return joint, nuisances
    prior = nuisance_prior(
        nuisances,
        nuisance_shapes,
        nuisance_prior_std,
        nuisance_prior_mean,
        tuple(global_design),
        global_shapes,
    )
    return SqrtInfo.combine(joint, prior), nuisances


def compress_epoch(
    global_design: Mapping[str, jax.Array],
    data: jax.Array,
    precision: Any,
    global_shapes: Mapping[str, tuple[int, ...]],
    *,
    nuisance_design: Mapping[str, jax.Array] | None = None,
    nuisance_shapes: Mapping[str, tuple[int, ...]] | None = None,
    nuisance_prior_std: Mapping[str, Any] | None = None,
    nuisance_prior_mean: Mapping[str, Any] | None = None,
    offset_prediction: jax.Array | None = None,
) -> SqrtInfo:
    """One epoch, with its own nuisances integrated out, over the globals.

    The streaming analysis in one call: compress the epoch over BOTH sets,
    append the nuisances' prior rows, and marginalise them. What comes back is
    a term over the globals alone, and folding those across a campaign is
    :meth:`~bayesmith.evidence.sqrtinfo.SqrtInfo.combine`.

    **The prior rows go in before the marginalisation, not after**, and that
    is not a convenience. A per-epoch nuisance is integrated exactly once, so
    a prior that arrives later has nowhere to be applied -- and without one
    the block need not constrain itself, which makes the integral divergent.
    :func:`~bayesmith.evidence.sqrtinfo.marginalise` refuses that case rather
    than returning the large plausible number finite arithmetic would give.

    Args:
        global_design: ``{latent: (n_data, n_i)}`` for the latents that
            SURVIVE the epoch.
        data: ``(n_data,)`` this epoch's observations.
        precision: the epoch's ``N^-1``.
        global_shapes: ``{latent: shape}`` for the survivors.
        nuisance_design: ``{latent: (n_data, n_i)}`` for the latents
            integrated away inside the epoch. ``None`` means there are none,
            and this reduces to :func:`compress`.
        nuisance_shapes, nuisance_prior_std, nuisance_prior_mean: the
            nuisances' shapes, prior widths and prior centres.
            ``nuisance_prior_mean`` defaults to zero.
        offset_prediction: as for :func:`compress`.

    Returns:
        A term over ``global_design``'s latents whose ``log_prob`` is the
        epoch's marginal log-likelihood -- exactly
        ``log N(d | A_g x_g + A_n m_n + c, N + A_n S_n A_n^T)``, which is the
        independent formula the tests compare against.

    Raises:
        StructureError: from :func:`compress` or from
            :func:`~bayesmith.evidence.sqrtinfo.marginalise`; and if
            nuisances are named without a prior, which the marginalisation
            has no way to make convergent.
    """
    if not nuisance_design:
        return compress(
            global_design,
            data,
            precision,
            global_shapes,
            offset_prediction=offset_prediction,
        )
    if nuisance_shapes is None or nuisance_prior_std is None:
        raise StructureError(
            "compress_epoch was given nuisance_design but no nuisance_shapes "
            "or nuisance_prior_std. A per-epoch nuisance is integrated exactly "
            "once, so its prior has to be part of the model rather than an "
            "optional regulariser -- without one the block need not constrain "
            "itself and the integral over it diverges."
        )
    joint, nuisances = epoch_joint(
        global_design,
        data,
        precision,
        global_shapes,
        nuisance_design=nuisance_design,
        nuisance_shapes=nuisance_shapes,
        nuisance_prior_std=nuisance_prior_std,
        nuisance_prior_mean=nuisance_prior_mean,
        offset_prediction=offset_prediction,
    )
    return marginalise(joint, nuisances)


def _ravelled(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total
