"""A campaign, compressed from the graph, one epoch at a time.

Everything below this module works on matrices a caller assembled:
:func:`~bayesmith.marginal.compress.compress_epoch` takes designs, data and a
covariance and knows nothing about where they came from. This is where they
come from a :class:`~bayesmith.graph.graph.Graph`.

**Derived, not declared twice**, which is the migration spec's whole argument
for rewriting B11 rather than transplanting it. Every input
``compress_epoch`` needs is read off the graph:

===================  ======================================================
what                 where it comes from
===================  ======================================================
the partition        :func:`~bayesmith.marginal.factorize.factorize` --
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
import numpy as np

from bayesmith.errors import StructureError
from bayesmith.exact.block import isolate, unchecked_operator
from bayesmith.exact.gaussian import gaussian_parts, precision_at
from bayesmith.graph.evaluate import evaluate
from bayesmith.graph.graph import Graph
from bayesmith.marginal.compress import epoch_joint
from bayesmith.marginal.factorize import Factorization, factorize
from bayesmith.marginal.sqrtinfo import SqrtInfo, marginalise_arrays


def epoch_terms(
    graph: Graph,
    epoch_plate: str,
    *,
    at: dict[str, Any] | None = None,
    factorization: Factorization | None = None,
    stacked_only: bool = False,
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
        stacked_only: return the terms as STACKED arrays rather than a list,
            so :func:`compress_campaign` can fold them under ``lax.scan``
            instead of unpacking E of them only to re-stack. Internal; the
            list is the shape a caller wants.

    Raises:
        StructureError: from :func:`factorize`; or if the graph does not have
            exactly one observed node plated on ``epoch_plate``, which is what
            makes "one epoch's data" a thing that can be sliced.
    """
    found = factorization or factorize(graph, epoch_plate, at=at)
    observed = epoch_observation(graph, epoch_plate)
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

    # VECTORISED, not looped. Measured on a 1000-epoch campaign: the Python
    # loop this replaces cost 2.82 ms per epoch, all of it eager QR, and an
    # unrolled `jit` of it compiled in 5.4 s. Every epoch does the same
    # arithmetic on the same shapes, which is what `vmap` is for -- and
    # `compress` traces now, so it can be.
    #
    # The per-epoch design is the jacobian's DIAGONAL in the two epoch axes:
    # entry `[e, ..., e, ...]`, i.e. how epoch e's data moves with epoch e's
    # own latent. `jnp.diagonal` takes all E at once.
    stacked_globals = {
        name: jnp.reshape(jacobian[name], (size, per_epoch_rows, -1))
        for name in found.survivors
    }
    stacked_nuisances = {
        name: jnp.moveaxis(
            jnp.diagonal(
                jnp.reshape(
                    jacobian[name], (size, per_epoch_rows, size, -1)
                ),
                axis1=0,
                axis2=2,
            ),
            -1,
            0,
        )
        for name in found.per_epoch
    }
    stacked_data = jnp.reshape(data, (size, per_epoch_rows))
    stacked_constant = jnp.reshape(constant, (size, per_epoch_rows))
    stacked_precision = _stack_precision(precision, size, per_epoch_rows)

    global_shapes = {name: domain.shape[name] for name in found.survivors}
    nuisance_shapes = {
        name: domain.shape[name][1:] for name in found.per_epoch
    }

    nuisance_width = sum(
        int(np.prod(nuisance_shapes[name], dtype=int)) for name in found.per_epoch
    )

    def one_epoch(globals_, nuisances, values, offset_prediction, noise):
        joint, _ = epoch_joint(
            globals_,
            values,
            noise,
            global_shapes,
            nuisance_design=nuisances or None,
            nuisance_shapes=nuisance_shapes or None,
            nuisance_prior_std={
                name: priors[name][1] for name in found.per_epoch
            }
            or None,
            nuisance_prior_mean={
                name: priors[name][0] for name in found.per_epoch
            }
            or None,
            offset_prediction=offset_prediction,
        )
        # `marginalise_arrays`, not `marginalise`: this runs under `vmap` and
        # the checked path concretises. The refusal is not weakened, it is
        # MOVED -- `pivots` comes back as data and every epoch's is judged
        # together, below, in one comparison instead of E.
        #
        # No permutation: `epoch_joint` puts the nuisances first, which is the
        # layout `marginalise_arrays` documents.
        return marginalise_arrays(
            joint.factor, joint.target, joint.offset, nuisance_width
        )

    factors, targets, offsets, pivots = jax.vmap(one_epoch)(
        stacked_globals,
        stacked_nuisances,
        stacked_data,
        stacked_constant,
        stacked_precision,
    )
    _refuse_unconstrained_epochs(pivots, nuisance_width, found.per_epoch)

    survivor_shapes = tuple(global_shapes[name] for name in found.survivors)
    if stacked_only:
        return factors, targets, offsets, found.survivors, survivor_shapes
    terms = [
        SqrtInfo(
            factor=factors[epoch],
            target=targets[epoch],
            offset=offsets[epoch],
            names=found.survivors,
            shapes=survivor_shapes,
        )
        for epoch in range(size)
    ]
    return terms


def fold_epochs(
    designs: jax.Array,
    targets: jax.Array,
    offsets: jax.Array,
    width: int,
) -> SqrtInfo | tuple[jax.Array, jax.Array, jax.Array]:
    """Fold ``E`` stacked ``[R | z]`` terms into one, under ``lax.scan``.

    Traceable, and that is the point. The Python loop it replaces costs one
    eager QR per epoch and an unrolled ``jit`` of it compiles in time
    proportional to ``E`` -- measured at ``E = 1000``, 5.4 s to compile a fold
    whose warm run is 2 ms. A scan compiles once whatever ``E`` is.

    The carry is the accumulated ``(factor, target, offset)``, and it keeps a
    FIXED shape across the whole campaign -- which is why
    :meth:`~bayesmith.marginal.sqrtinfo.SqrtInfo.null` is square rather than
    zero-row. A carry whose treedef moved would retrace once per epoch and
    defeat the exercise.

    Args:
        designs: ``(E, rows, width)`` stacked factors.
        targets: ``(E, rows)`` stacked targets.
        offsets: ``(E,)`` stacked constants.
        width: the column count, needed statically for the carry's shape.

    Returns:
        ``(factor, target, offset)`` of the folded term.
    """

    def step(carry, epoch):
        factor, target, offset = carry
        rows, row_target, row_offset = epoch
        stacked = jnp.concatenate(
            [
                jnp.concatenate([factor, target[:, None]], axis=1),
                jnp.concatenate([rows, row_target[:, None]], axis=1),
            ],
            axis=0,
        )
        upper = jnp.linalg.qr(stacked, mode="r")
        keep = min(upper.shape[0], width)
        corner = upper[keep:, width]
        return (
            upper[:keep, :width],
            upper[:keep, width],
            offset + row_offset - 0.5 * jnp.sum(corner**2),
        ), None

    start = (jnp.zeros((width, width)), jnp.zeros(width), jnp.zeros(()))
    (factor, target, offset), _ = jax.lax.scan(
        step, start, (designs, targets, offsets)
    )
    return factor, target, offset


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
    factors, targets, offsets, names, shapes = epoch_terms(
        graph, epoch_plate, at=at, factorization=factorization, stacked_only=True
    )
    width = int(factors.shape[-1])
    factor, target, offset = fold_epochs(factors, targets, offsets, width)
    return SqrtInfo(
        factor=factor, target=target, offset=offset, names=names, shapes=shapes
    )


def epoch_observation(graph: Graph, epoch_plate: str) -> str:
    """The one observed node whose data IS this plate's epochs, or a refusal.

    PUBLIC because a caller deciding *whether* to fold needs the same
    structural verdict the fold itself makes, and cannot get it from
    ``factorize`` alone -- see
    :func:`~bayesmith.dispatch.streaming.streaming_route`, which was the
    package's only cross-module private import until this name lost its
    underscore.
    """
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


def _refuse_unconstrained_epochs(
    pivots: jax.Array, n_block: int, names: tuple[str, ...]
) -> None:
    """Every epoch's pivots, judged together -- the moved refusal.

    :func:`~bayesmith.marginal.sqrtinfo.marginalise` makes this judgement one
    term at a time and therefore cannot be traced. A campaign vmaps the
    arithmetic and judges here instead, which is the same refusal at one
    comparison instead of E.

    Both halves, in the order that matters: FINITENESS first, because the
    degeneracy threshold is relative to the largest pivot, so one ``nan``
    anywhere makes the threshold ``nan`` and every comparison against it
    False -- and one ``inf`` makes it ``inf`` and admits every pivot there is.
    """
    if n_block == 0:
        return
    if not bool(jnp.all(jnp.isfinite(pivots))):
        bad = [int(e) for e in np.flatnonzero(~np.all(np.isfinite(np.asarray(pivots)), axis=1))]
        raise StructureError(
            f"epochs {bad} re-triangularise to non-finite pivots, so their "
            f"marginal over {list(names)} would carry nan -- which nothing "
            "downstream tests for. That epoch's design, data or covariance "
            "already carried nan or inf before this call."
        )
    scale = jnp.max(pivots, axis=1, keepdims=True)
    floor = float(np.sqrt(np.finfo(np.asarray(pivots).dtype).eps)) * scale
    ok = jnp.all(pivots[:, :n_block] > floor, axis=1)
    if not bool(jnp.all(ok)):
        bad = [int(e) for e in np.flatnonzero(~np.asarray(ok))]
        raise StructureError(
            f"in epochs {bad} the block {list(names)} does not constrain one "
            "of its own directions, so the Gaussian integral over it diverges "
            "and the marginal would come back as +inf -- finite arithmetic "
            "gives a large plausible number instead, which is worse. A "
            "per-epoch latent is integrated exactly once, so its prior has to "
            "be part of the model rather than an optional regulariser."
        )


def _stack_precision(precision: Any, size: int, rows: int) -> Any:
    """The whole node's covariance as ``size`` per-epoch ones, stacked.

    ``vmap`` maps over the leading axis of a pytree leaf, and a ``Precision``
    is an ``eqx.Module`` whose leaves are its arrays -- so a batched one IS
    the stack. This only has to check that the batch axis is the epoch axis
    and reshape the rest, which :func:`_slice_precision` states the reasons
    for one epoch at a time.
    """
    from bayesmith.exact.precision import CirculantPrecision, DiagonalPrecision

    first = _slice_precision(precision, 0, size)
    if isinstance(first, DiagonalPrecision):
        return DiagonalPrecision(
            sigma=jnp.reshape(precision.sigma, (size, rows))
        )
    if isinstance(first, CirculantPrecision):
        return CirculantPrecision(
            first_column=jnp.reshape(precision.first_column, (size, -1))
        )
    raise StructureError(  # pragma: no cover - the gate admits two rows
        f"no rule for stacking a {type(precision).__name__} by epoch."
    )


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
