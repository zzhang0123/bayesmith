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
from bayesmith.evidence.sqrtinfo import SqrtInfo
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
    if seen is not None and not bool(jnp.all(seen)):
        # The masked path. Only a per-sample sigma can say a sample was not
        # taken, and only then is the normaliser a subset sum -- which is
        # exactly why this branch is gated on `seen` existing at all.
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
