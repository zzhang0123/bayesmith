"""A campaign, compressed from the graph -- the bridge B11 is for.

Everything below this module works on matrices a caller assembled:
:func:`~bayesmith.evidence.compress.compress_epoch` takes designs, data and a
covariance and knows nothing about where they came from. This is where they
come from a :class:`~bayesmith.graph.graph.Graph`.

**Derived, not declared twice**, which is the migration spec's whole argument
for rewriting B11 rather than transplanting it. Every input
``compress_epoch`` needs is read off the graph:

===================  ======================================================
what                 where it comes from
===================  ======================================================
the partition        :func:`~bayesmith.evidence.factorize.factorize` --
                     plate membership, then ``epoch_leakage`` to test it
the designs          one ``jacfwd`` of the graph's own prediction map
the constant part    that map evaluated at zero
the data             the observed node's own ``obs``, sliced by epoch
the covariance       :func:`~bayesmith.exact.gaussian.precision_at`, sliced
the nuisance priors  the per-epoch latents' own ``dist_fn``
===================  ======================================================

The last row is the one rheplicant cannot have: its ``Factorization`` carries
per-epoch priors separately from the ``ParameterSpace`` that declares them,
and "the same space declared twice" is the error class this kills.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from bayesmith.errors import StructureError
from bayesmith.evidence.compress import compress_epoch
from bayesmith.evidence.factorize import Factorization, factorize
from bayesmith.evidence.sqrtinfo import SqrtInfo
from bayesmith.exact.block import isolate, unchecked_operator
from bayesmith.exact.gaussian import gaussian_parts, precision_at
from bayesmith.graph.evaluate import evaluate
from bayesmith.graph.graph import Graph


def epoch_terms(
    graph: Graph,
    epoch_plate: str,
    *,
    at: dict[str, Any] | None = None,
    factorization: Factorization | None = None,
) -> list[SqrtInfo]:
    """One ``[R | z]`` term per epoch, over the survivors.

    Each epoch's own per-epoch latents are integrated out inside it, at the
    prior the graph declares for them -- so what comes back is a list of terms
    all over the same latents, ready to fold.

    Args:
        graph: the model. Exactly one observed node, plated on ``epoch_plate``.
        epoch_plate: the plate whose axis indexes epochs.
        at: values for latents this holds fixed. Defaults to zeros, which is
            what an affine model's design does not depend on.
        factorization: a partition already derived and checked. ``None``
            derives one, **including the leakage check** -- pass one only if
            you have already run that.

    Raises:
        StructureError: from :func:`factorize`; or if the graph does not have
            exactly one observed node plated on ``epoch_plate``, which is what
            makes "one epoch's data" a thing that can be sliced.
    """
    found = factorization or factorize(graph, epoch_plate, at=at)
    observed = _the_epoch_observation(graph, epoch_plate)
    node = graph.node(observed)
    size = graph.plate_size(epoch_plate)

    names = found.survivors + found.per_epoch
    # The domain shapes come from a block rather than from the plate, because
    # a latent OUTSIDE the epoch plate can still have a shape of its own -- a
    # vector of weights is the ordinary case, and inferring `()` from "not
    # plated" gets it wrong.
    domain = unchecked_operator(graph, names, {})
    zeros = {name: jnp.zeros(domain.shape[name], domain.dtype[name]) for name in names}
    forward = isolate(graph, names, {})
    constant = forward(zeros)[observed]
    jacobian = jax.jacfwd(forward)(zeros)[observed]

    data = jnp.asarray(node.observed)
    per_epoch_rows = int(jnp.asarray(data[0]).size)
    precision = precision_at(graph, {**zeros, **(at or {})})[observed]
    env = evaluate(graph, {**zeros, **(at or {})})

    priors = {}
    for name in found.per_epoch:
        loc, scale = gaussian_parts(graph, graph.node(name), env)
        # the epoch-independent prior this latent declares. Broadcast to the
        # per-epoch width by `nuisance_prior`, so one entry is enough and a
        # per-epoch-varying prior would need more than this reads.
        priors[name] = (jnp.reshape(loc, (-1,))[0], jnp.reshape(scale, (-1,))[0])

    terms = []
    for epoch in range(size):
        global_design = {
            name: jnp.reshape(jacobian[name][epoch], (per_epoch_rows, -1))
            for name in found.survivors
        }
        nuisance_design = {
            name: jnp.reshape(
                jnp.take(jnp.take(jacobian[name], epoch, axis=0), epoch, axis=-1),
                (per_epoch_rows, -1),
            )
            for name in found.per_epoch
        }
        terms.append(
            compress_epoch(
                global_design,
                jnp.reshape(data[epoch], (per_epoch_rows,)),
                _slice_precision(precision, epoch, size),
                # the survivors' OWN shapes, so a scalar latent stays a
                # scalar and `SqrtInfo.ravel` can accept the value dict a
                # caller already has
                {name: domain.shape[name] for name in found.survivors},
                nuisance_design=nuisance_design or None,
                # a per-epoch latent's own per-epoch shape is what remains
                # after the epoch axis is indexed away
                nuisance_shapes={
                    name: domain.shape[name][1:] for name in found.per_epoch
                }
                or None,
                nuisance_prior_std={
                    name: priors[name][1] for name in found.per_epoch
                }
                or None,
                nuisance_prior_mean={
                    name: priors[name][0] for name in found.per_epoch
                }
                or None,
                offset_prediction=jnp.reshape(constant[epoch], (per_epoch_rows,)),
            )
        )
    return terms


def compress_campaign(
    graph: Graph,
    epoch_plate: str,
    *,
    at: dict[str, Any] | None = None,
    factorization: Factorization | None = None,
) -> SqrtInfo:
    """Fold every epoch into one term over the survivors.

    Holds one epoch's term at a time, which is the point of the exercise: the
    raw data and the forward evaluation are gone by the end, and what remains
    is a sufficient statistic whose size does not grow with the campaign.
    """
    total: SqrtInfo | None = None
    for term in epoch_terms(
        graph, epoch_plate, at=at, factorization=factorization
    ):
        total = term if total is None else SqrtInfo.combine(total, term)
    if total is None:  # pragma: no cover - factorize refuses an empty plate
        raise StructureError(f"plate {epoch_plate!r} has no epochs to fold.")
    return total


def _the_epoch_observation(graph: Graph, epoch_plate: str) -> str:
    plated = [
        name for name in graph.observed if epoch_plate in graph.node(name).plate
    ]
    if len(plated) != 1:
        raise StructureError(
            f"this graph has {len(plated)} observed nodes plated on "
            f"{epoch_plate!r} ({plated}); a campaign is compressed one epoch at "
            "a time, so exactly one node's data has to BE the epoch's data. "
            "Two would have to be sliced consistently and nothing here says "
            "they are; none means there is nothing to slice."
        )
    return plated[0]


def _slice_precision(precision: Any, epoch: int, size: int) -> Any:
    """This epoch's covariance out of the whole node's.

    A plate makes the covariance batched -- ``sigma`` of shape ``(E, ...)``
    for a diagonal, a ``(E, n)`` stacked kernel for a circulant -- so slicing
    is taking row ``epoch``. That the batched circulant is `E` INDEPENDENT
    covariances rather than one coupled one is what makes an epoch's noise
    separable at all, and it is a property of how a plate builds them.

    Refuses a covariance that does not batch over the epoch axis, because
    then the epochs' noise is coupled and folding them separately is wrong in
    the same way a leaky latent is.
    """
    from bayesmith.exact.precision import CirculantPrecision, DiagonalPrecision

    if isinstance(precision, DiagonalPrecision):
        sigma = precision.sigma
        if sigma.shape[:1] != (size,):
            raise StructureError(
                f"this node's sigma has shape {sigma.shape}, which does not "
                f"batch over the {size} epochs, so it cannot be split into one "
                "covariance per epoch. A campaign's noise has to be epoch-"
                "separable; one that couples epochs makes folding them "
                "separately wrong in the same way a leaky latent does."
            )
        return DiagonalPrecision(sigma=jnp.reshape(sigma[epoch], (-1,)))
    if isinstance(precision, CirculantPrecision):
        column = precision.first_column
        if column.shape[:1] != (size,):
            raise StructureError(
                f"this node's covariance kernel has shape {column.shape}, "
                f"which does not batch over the {size} epochs. A single "
                "circulant spanning the whole campaign COUPLES the epochs -- "
                "its observed submatrix per epoch is not itself circulant -- "
                "so it cannot be folded epoch by epoch."
            )
        return CirculantPrecision(first_column=column[epoch])
    raise StructureError(  # pragma: no cover - the gate admits two rows
        f"no rule for slicing a {type(precision).__name__} by epoch."
    )
