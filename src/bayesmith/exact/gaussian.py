"""Reading a Gaussian off a node -- and checking the reading.

The exact solves need three numbers a graph does not hand over directly: the
prediction and its sigma (from an observed node's distribution), and the
prior's centre and width (from a latent node's). ``dist_fn`` returns an
opaque NumPyro distribution, so those numbers have to be *extracted* -- and,
this package's rule being that a claim is checked rather than trusted, the
extraction is checked too.

**Two functions, deliberately.** :func:`gaussian_parts` is the fast path: one
``isinstance``, two attribute reads, fully traceable, and it is what runs
inside ``jax.linearize`` on every solve. :func:`check_gaussian` is the guard:
it probes the node's own ``log_prob`` at several points and refuses if the
extracted ``(loc, scale)`` do not reproduce it. The guard uses Python floats
and raises, so it cannot run inside a trace -- it runs once, on concrete
values, when the block is built.

Splitting them is not an optimisation, it is a correctness requirement: a
guard that cannot run where the fast path runs must run *before* it, on the
values the fast path will see.

**Why the introspection is a fast path and not the answer.** Reading
``.loc``/``.scale`` off a ``Normal`` trusts the type. A ``Distribution``
subclass may override ``log_prob`` -- censored, tempered, or simply wrong --
and keep both attributes, so the type is evidence, not proof. The probe is
what raises the bar, at a cost of five ``log_prob`` evaluations per node per
block build.

**What the probe does and does not establish.** It establishes that the
extracted ``(loc, scale)`` reproduce the node's own density *at the probed
points, elementwise, at the shape the node's value actually takes*. It does
not establish agreement everywhere: a finite set of points cannot certify a
claim about a function, and a correction shaped to vanish at exactly these
offsets -- a quartic with roots there, say -- would pass while being wrong by
hundreds of nats at the mode. That is an adversarial construction, and the
threat this guard is placed against is accidental: a censored likelihood, a
tempered one, a hand-written approximation. Those do not have roots at the
probe points. Stated so the guarantee is not mistaken for a stronger one.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from bayesmith.errors import NotGaussian, StructureError
from bayesmith.graph.evaluate import apply_probabilistic, evaluate
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Node, Probabilistic

#: Probe offsets for :func:`check_gaussian`, in units of the node's own scale.
#: Spread over the bulk and into both tails: a wrong ``loc`` shifts every
#: probe, a wrong ``scale`` produces a mismatch that grows with ``|offset|``,
#: and a log-density that is not quadratic fails at the outer pair first.
#: Asymmetric on purpose -- a symmetric set cannot distinguish a sign error in
#: ``loc`` from a correct one. ``0.0`` is included because it is the mode:
#: the single point carrying the most posterior mass, and the point a
#: correction shaped to vanish at the other offsets is most likely to miss.
PROBE_OFFSETS: tuple[float, ...] = (-3.0, -1.0, 0.0, 0.5, 2.0)

_LOG_2PI = float(np.log(2.0 * np.pi))


def unwrap(distribution: Any) -> Any:
    """Strip wrappers that only change how ``log_prob`` is reduced.

    ``Independent`` is what ``.to_event(k)`` produces: the same per-element
    density with the last ``k`` batch dimensions summed. The elementwise
    reading underneath is what the exact solves want -- a diagonal covariance
    is diagonal whether or not its log-density was summed -- so the wrapper is
    removed rather than refused.
    """
    while isinstance(distribution, dist.Independent):
        distribution = distribution.base_dist
    return distribution


def gaussian_parts(
    graph: Graph, node: Node, env: dict[str, Any]
) -> tuple[jax.Array, jax.Array]:
    """``(loc, scale)`` of ``node``'s distribution, ``scale`` broadcast to ``loc``.

    Traceable and **unchecked** -- pair it with :func:`check_gaussian`, which
    runs once on concrete values before any trace is opened.

    Raises:
        NotGaussian: if the distribution is not a (possibly ``Independent``-
            wrapped) ``Normal``. The *type* is static under tracing, so this
            refusal is safe to make inside a trace. It is a **classification
            outcome, not a fault**: P3b's dispatcher catches it and routes the
            block to NUTS.
        StructureError: if ``loc`` has an integer dtype. That one IS a fault --
            a conjugate solve differentiates through ``loc``.
    """
    distribution = unwrap(apply_probabilistic(graph, node, env))
    if not isinstance(distribution, dist.Normal):
        raise NotGaussian(
            f"node {node.name!r} returns {type(distribution).__name__}; the exact "
            "linear-Gaussian path needs a diagonal Normal (a Normal, or one "
            "wrapped by .to_event(...)). A MultivariateNormal with a dense "
            "covariance is a different solve and is not implemented. This is a "
            "classification outcome, not a defect in the model."
        )
    return _checked_loc(node, distribution.loc), jnp.broadcast_to(
        jnp.asarray(distribution.scale), jnp.shape(jnp.asarray(distribution.loc))
    )


def _checked_loc(node: Node, loc: Any) -> jax.Array:
    """``loc`` as an array, refusing an integer dtype.

    Shared by :func:`gaussian_parts` and :func:`precision_parts` rather than
    written twice -- the two entry points must agree about what a usable loc
    is, and the second copy is the one that would drift.
    """
    found = jnp.asarray(loc)
    if not jnp.issubdtype(found.dtype, jnp.inexact):
        raise StructureError(
            f"node {node.name!r} has an integer loc (dtype {found.dtype}). A "
            "conjugate solve differentiates through loc, so it has to be a "
            "floating dtype -- pass a float to the distribution."
        )
    return found


def _loc_of(graph: Graph, node: Node, env: dict[str, Any]) -> jax.Array:
    """``node``'s ``loc``, with no opinion about its covariance.

    Split out of :func:`gaussian_parts` because :func:`node_shape` needs only
    this. Reading the shape through ``gaussian_parts`` made SHAPE refuse a
    correlated node -- measured: a ``CirculantNormal`` observation stopped in
    ``node_shape`` -> ``gaussian_parts``, so ``observation_parts``,
    ``noise_std_at``, ``check_linearity`` and ``linear_operator`` all raised
    ``NotGaussian`` before any covariance was read. A node's value has a shape
    whatever its samples' correlations are; the gate belongs where the
    covariance is read, which is :func:`gaussian_parts` and
    :func:`precision_parts`, and observed nodes still meet ``check_gaussian``.
    """
    return _checked_loc(node, unwrap(apply_probabilistic(graph, node, env)).loc)


def node_shape(graph: Graph, node: Node, env: dict[str, Any]) -> tuple[int, ...]:
    """Shape of ``node``'s VALUE.

    Three sources, broadcast together: the distribution's own batch shape, the
    plate the node sits in, and -- for an observed node -- the data. All three
    are needed. A plated latent whose ``dist_fn`` takes no plated parent has a
    scalar ``loc`` and a plate-shaped value (the ordinary "N iid draws from one
    shared prior"); an unplated observed node conditioned on a vector has a
    scalar ``loc`` and a vector value. Taking any one source alone gets one of
    those two wrong.

    This must agree with what ``to_numpyro`` opens the site at, or the block's
    domain is a different space from the one NUTS samples --
    ``test_node_shape_agrees_with_the_numpyro_bridge`` pins it.
    """
    loc = _loc_of(graph, node, env)
    shapes: list[tuple[int, ...]] = [jnp.shape(loc)]
    if node.plate:
        shapes.append((graph.plate_size(node.plate[0]),))
    if isinstance(node, Probabilistic) and node.observed is not None:
        shapes.append(jnp.shape(node.observed))
    try:
        return tuple(jnp.broadcast_shapes(*shapes))
    except ValueError as exc:
        raise StructureError(
            f"node {node.name!r} has shapes that cannot be reconciled: its "
            f"distribution's loc is {jnp.shape(loc)}"
            + (
                f", its plate is {(graph.plate_size(node.plate[0]),)}"
                if node.plate
                else ""
            )
            + (
                f", its data is {jnp.shape(node.observed)}"
                if isinstance(node, Probabilistic) and node.observed is not None
                else ""
            )
            + ". A node's value has one shape; these disagree. Raw broadcasting "
            "would report the same clash without naming the node."
        ) from exc


def check_gaussian(
    graph: Graph, node: Node, env: dict[str, Any], *, rtol: float | None = None
) -> dict[float, float]:
    """Verify the extracted ``(loc, scale)`` really reproduce ``node``'s log_prob.

    Costs ``len(PROBE_OFFSETS)`` evaluations of ``log_prob``. Runs on concrete
    values, **outside** any trace.

    **Probes at the shape the node's VALUE takes**, not at ``dist_fn``'s own
    batch shape, and compares **elementwise**. Both matter, and both were
    measured:

    * A plated latent whose ``dist_fn`` takes no plated parent has a scalar
      ``loc`` and a plate-shaped value; so does an unplated observed node
      conditioned on a vector. Probing at the scalar evaluates ``log_prob`` at
      a shape the consumer never uses. Measured on this package's own
      ``plated_latent`` fixture pattern: a ``Distribution`` subclass correct
      on a scalar and off by 1000 nats per element on an array passed the
      guard with every reported error exactly ``0.0``, against a real
      discrepancy of 2.0e6 nats. This is the same shape of defect P1 recorded
      -- the guard and the thing it guards looking at different shapes -- and
      it is fixed the same way.
    * A summed comparison dilutes a localised defect by the magnitudes of the
      correct entries. Measured: one wrong element out of 1e6, off by 50 nats,
      reports a summed relative error of 1.95e-5 -- under the default rtol,
      silently accepted -- and an elementwise error of 50.

    Args:
        graph, node, env: the node under test and the values its parents take.
        rtol: tolerance on the relative disagreement, per element. Default
            ``1e3 * eps`` of ``loc``'s dtype, which leaves room for
            accumulated roundoff without admitting a real difference in
            density.

    Returns:
        ``{offset: worst relative error over the node's entries}`` -- useful
        for reporting how Gaussian a node is, not only whether it passes.

    Raises:
        NotGaussian: propagated from :func:`gaussian_parts`.
        StructureError: if the scale is not strictly positive and finite, or
            if any probe disagrees by more than ``rtol`` at any entry.
    """
    distribution = unwrap(apply_probabilistic(graph, node, env))
    loc, scale = gaussian_parts(graph, node, env)
    shape = node_shape(graph, node, env)
    loc = jnp.broadcast_to(loc, shape)
    scale = jnp.broadcast_to(scale, shape)

    usable = jnp.isfinite(scale) & (scale > 0)
    if not bool(jnp.all(usable)):
        # The OFFENDING entry, not `min(scale)`. Measured: on
        # `[1, 1, inf, 1, inf, 1]` the minimum is 1, which is strictly
        # positive and finite -- so the message's own evidence exonerated
        # the scale it was refusing, and a reader checking it concludes the
        # guard is broken rather than the model. A number offered as
        # evidence has to point at the fault.
        broken = np.flatnonzero(~np.asarray(usable).reshape(-1))
        first = float(np.asarray(scale).reshape(-1)[broken[0]])
        raise StructureError(
            f"node {node.name!r} has a scale that is not strictly positive and "
            f"finite: {broken.size} of {int(np.asarray(scale).size)} entries, "
            f"the first being {first:g} at flat index {int(broken[0])}. A "
            "conjugate solve weights by 1/scale**2, so a zero or negative "
            "sigma is an infinite or negative weight rather than a tight "
            "constraint. Add a floor to the expression that produces it.\n\n"
            "An INFINITE sigma is a different intent and has a different "
            "home: it is how this package spells an unobserved (flagged) "
            "sample, and the mask that turns it into a clean zero "
            "contribution lives in bayesmith.evidence.compress, which owns "
            "that concept. A Precision is asked what density a covariance "
            "gives, and the honest answer for infinite variance is none, so "
            "it is not masked here."
        )
    if rtol is None:
        rtol = 1e3 * float(jnp.finfo(loc.dtype).eps)

    errors: dict[float, float] = {}
    for offset in PROBE_OFFSETS:
        probe = loc + offset * scale
        actual = jnp.broadcast_to(distribution.log_prob(probe), shape)
        predicted = (
            -0.5 * ((probe - loc) / scale) ** 2 - jnp.log(scale) - 0.5 * _LOG_2PI
        )
        # Elementwise, and floored at 1.0 so a probe landing where the
        # log-density happens to be ~0 does not divide by it.
        departure = jnp.abs(actual - predicted) / jnp.maximum(jnp.abs(predicted), 1.0)
        errors[offset] = float(jnp.max(departure))
        # NaN must count as a FAILURE: `nan > rtol` is False, so a naive
        # comparison treats an unusable probe as evidence of Gaussianity.
        if not np.isfinite(errors[offset]) or errors[offset] > rtol:
            detail = ", ".join(f"{k:+g}sigma -> {v:.3e}" for k, v in errors.items())
            raise StructureError(
                f"node {node.name!r} is a {type(distribution).__name__}, so its "
                "loc and scale were read off it directly -- but its own log_prob "
                f"does not agree with them (rtol={rtol:.2e}; worst entry per "
                f"probe: {detail}). A Distribution subclass that overrides "
                "log_prob keeps both attributes and changes the density, so the "
                "type is evidence and not proof. The exact path would solve the "
                "wrong posterior; it refuses instead."
            )
    return errors


def check_observed(
    graph: Graph, node: Node, env: dict[str, Any], *, rtol: float | None = None
) -> dict[Any, float]:
    """Probe an observed node's declared density, whichever covariance it has.

    The eager guard the build path runs, and the router between the two:

    ===================  ==================================================
    distribution         what is run
    ===================  ==================================================
    ``Normal``           :func:`check_gaussian` -- unchanged, five offsets
                         against the ``(loc, scale)`` read off it
    ``CirculantNormal``  :func:`~bayesmith.exact.precision.\
                         check_positive_definite`, THEN
                         :func:`~bayesmith.exact.precision.check_precision`
    ===================  ==================================================

    **The order is load-bearing, and it was measured.** ``check_precision``
    does refuse an indefinite kernel on its own -- but only through NaN
    propagation, and it names the wrong cause. Measured on three indefinite
    kernels, including one whose entries are all positive: numpyro's own
    ``CirculantNormal.log_prob`` returns ``nan``, so ``check_precision``
    reports ``linearity=nan, normalizer=nan``, and its message explains
    ``linearity`` as "the log-density is not quadratic, so it has no
    covariance to extract". The log-density IS quadratic; the kernel simply
    describes no realisable process. A user sent to look at their ``det``
    nodes and ``linear_in`` declarations would be looking in the wrong place.
    :func:`~bayesmith.exact.precision.check_positive_definite` says what is
    actually wrong, so it goes first.

    Refusing through a NaN is also fragile in a way that saying so is not: it
    holds only for as long as ``log_prob`` keeps returning ``nan`` rather
    than, say, clamping.

    Args:
        graph, node, env: the node under test and the values its parents take.
        rtol: passed to whichever guard runs. Both default it the same way,
            to ``1e3 * eps`` of ``loc``'s dtype.

    Returns:
        Whatever the guard that ran returns -- ``{offset: error}`` from
        :func:`check_gaussian`, ``{"operator"/"normalizer"/"linearity":
        error}`` from :func:`~bayesmith.exact.precision.check_precision`.
        The two are different measurements and are deliberately not squashed
        into one shape; every caller in this package discards them and reads
        the exception instead.

    Raises:
        NotGaussian: for a distribution neither row accepts, exactly as
            before -- still a classification outcome, not a fault.
        StructureError: from :func:`check_gaussian`.
        ValueError: from
            :func:`~bayesmith.exact.precision.check_positive_definite`, when
            the kernel is not a realisable autocovariance.
        PrecisionMismatch: from
            :func:`~bayesmith.exact.precision.check_precision`.
    """
    from bayesmith.exact.precision import (
        CirculantPrecision,
        check_positive_definite,
        check_precision,
    )

    distribution = unwrap(apply_probabilistic(graph, node, env))
    circulant = getattr(dist, "CirculantNormal", None)
    if circulant is None or not isinstance(distribution, circulant):
        return check_gaussian(graph, node, env, rtol=rtol)

    loc, precision = precision_parts(graph, node, env)
    assert isinstance(precision, CirculantPrecision)  # the only correlated row
    check_positive_definite(precision)
    return check_precision(
        distribution,
        precision,
        jnp.broadcast_to(loc, node_shape(graph, node, env)),
        rtol=rtol,
    )


def precision_parts(
    graph: Graph, node: Node, env: dict[str, Any]
) -> tuple[jax.Array, Any]:
    """``(loc, Precision)`` -- the ONE object the whole exact path should read.

    The widened counterpart of :func:`gaussian_parts`, which returns a
    per-sample ``scale`` and so can only ever describe a diagonal covariance.
    Traceable and **unchecked**; pair it with
    :func:`~bayesmith.exact.precision.check_precision`, which verifies the
    returned operator against the node's own density by the gradient identity
    and runs eagerly, outside any trace.

    The gate widens by exactly two rows and refuses everything else with the
    same :class:`NotGaussian`, so a model this cannot express is still a
    CLASSIFICATION outcome routed to NUTS rather than a fault:

    ==================  ====================================================
    distribution        Precision
    ==================  ====================================================
    ``Normal``          ``DiagonalPrecision`` -- unchanged behaviour
    ``CirculantNormal``  ``CirculantPrecision`` from ``.covariance_row``
    ==================  ====================================================

    ``CirculantNormal`` is not a new idea imported from outside: numpyro ships
    it, and its log-density agrees with ``CirculantPrecision``'s and with a
    dense reference to 1.8e-15 -- three independent routes. The exact path was
    the only thing refusing a correlated noise the rest of the stack could
    already handle.

    **``depends_on_prediction=True`` is accepted now.** ``45198f9`` refused
    it, because ``fisher.py``'s variance-information term was derived for a
    DIAGONAL ``N`` and nothing said it survived off-diagonal terms. It does,
    in a form that covers both rows at once:
    ``1/2 tr(N^-1 d_a N N^-1 d_b N) = 1/2 sum_k d_a log lam_k d_b log lam_k``
    whenever ``N``'s eigenbasis does not move with the parameters -- ``I`` for
    a ``Normal``, the DFT for a ``CirculantNormal``. Derived in
    ``docs/derivations/variance_information_spectral.wls``, measured against a
    dense finite-difference Fisher matrix in
    ``docs/probes/probe_9_correlated_variance_information.py``, and
    implemented as
    :func:`~bayesmith.exact.fisher._log_spectrum_curvature`, which
    ``fisher_information`` reaches through ``precision_of=``. The refusal was
    written to be deleted and this is its deletion.

    Raises:
        NotGaussian: for any other distribution, as before.
        StructureError: for an integer ``loc``, as before.
    """
    from bayesmith.exact.precision import CirculantPrecision, DiagonalPrecision

    distribution = unwrap(apply_probabilistic(graph, node, env))
    circulant = getattr(dist, "CirculantNormal", None)
    if circulant is not None and isinstance(distribution, circulant):
        loc = _checked_loc(node, distribution.loc)
        return loc, CirculantPrecision(
            first_column=jnp.asarray(distribution.covariance_row)
        )
    if not isinstance(distribution, dist.Normal):
        raise NotGaussian(
            f"node {node.name!r} returns {type(distribution).__name__}; the exact "
            "linear-Gaussian path reads a Normal (diagonal) or a "
            "CirculantNormal (stationary, periodic). A MultivariateNormal with "
            "a dense covariance is a different solve and is not implemented. "
            "This is a classification outcome, not a defect in the model."
        )
    loc = _checked_loc(node, distribution.loc)
    # To :func:`node_shape`, not to ``loc``'s shape. A plated node whose
    # ``dist_fn`` takes no plated parent has a SCALAR loc and a plate-shaped
    # value, and a ``DiagonalPrecision`` built at the scalar would still
    # ``apply`` correctly by broadcasting while its ``log_normalizer`` summed
    # ONE term instead of ``n`` -- a log-determinant short by a factor of the
    # plate size, which is defect B1's shape exactly.
    scale = jnp.broadcast_to(
        jnp.asarray(distribution.scale), node_shape(graph, node, env)
    )
    return loc, DiagonalPrecision(sigma=scale)


def observed_data_and_loc(
    graph: Graph, env: dict[str, Any]
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """``({obs: data}, {obs: loc})`` over every observed node, reading no covariance.

    The two halves of :func:`observation_parts` that have nothing to do with
    the noise. Split out because ``block.py`` wants exactly these -- ``isolate``
    takes the ``loc`` and the block build takes the ``data`` -- and reading
    them through ``observation_parts`` made both walk ``gaussian_parts``,
    which refuses a correlated node. A block's design matrix and its data do
    not depend on whether the samples are correlated.

    Both are broadcast to :func:`node_shape`, so the dicts align leaf for leaf
    and every reduction downstream is one ``jax.tree.map``.
    """
    data: dict[str, jax.Array] = {}
    loc: dict[str, jax.Array] = {}
    for name in graph.observed:
        node = graph.node(name)
        shape = node_shape(graph, node, env)
        data[name] = jnp.broadcast_to(node.observed, shape)
        loc[name] = jnp.broadcast_to(_loc_of(graph, node, env), shape)
    return data, loc


def observation_parts(
    graph: Graph, env: dict[str, Any]
) -> tuple[dict[str, jax.Array], dict[str, jax.Array], dict[str, jax.Array]]:
    """``({obs: data}, {obs: loc}, {obs: scale})`` over every observed node.

    All three are broadcast to :func:`node_shape`, so the three dicts align
    leaf for leaf and every reduction downstream is one ``jax.tree.map``.

    The DIAGONAL walk: ``scale`` comes from :func:`gaussian_parts`, so this
    refuses a correlated node as it always did. The first two dicts are
    :func:`observed_data_and_loc`, which does not, and callers wanting only
    those should take them from there.
    """
    data, loc = observed_data_and_loc(graph, env)
    scale: dict[str, jax.Array] = {}
    for name in graph.observed:
        node = graph.node(name)
        _, node_scale = gaussian_parts(graph, node, env)
        scale[name] = jnp.broadcast_to(node_scale, node_shape(graph, node, env))
    return data, loc, scale


def noise_std_at(graph: Graph, values: dict[str, Any]) -> dict[str, jax.Array]:
    """``{obs: scale}`` with the latents at ``values`` -- the GLS seam.

    ``iterative_gls`` iterates this: solve at the current sigma, recompute
    sigma at the new solution, repeat. Whether it moves at all is what
    separates a single Wiener solve from a reweighting loop, and
    ``test_noise_std_at_moves_with_the_latent_only_for_a_prediction_dependent_node``
    exercises both sides.
    """
    _, _, scale = observation_parts(graph, evaluate(graph, values))
    return scale


def precision_at(graph: Graph, values: dict[str, Any]) -> dict[str, Any]:
    """``{obs: Precision}`` with the latents at ``values`` -- the OPERATOR seam.

    The counterpart of :func:`noise_std_at`, and deliberately **not** its
    replacement. The seam SPLITS rather than renames, because the two sides
    are different quantities and only one of them generalises:

    ==============  ==============================  =========================
    seam            what it produces                who needs it
    ==============  ==============================  =========================
    ``precision_at``  ``N^-1`` as an operator       the solve, the density,
                                                    the information
    ``noise_std_at``  per-sample sigma VALUES       ``iterative_gls``'s
                                                    reweighting, the
                                                    linearity check's unit
    ==============  ==============================  =========================

    ``iterative_gls`` iterates sigma values -- solve at the current sigma,
    recompute sigma at the new solution, repeat -- and an operator has none to
    iterate. Renaming ``noise_std_at`` to this would have left that loop with
    nothing to say, which is why B9's step 4 is a split and not the rename its
    own text describes.

    A ``Precision`` is what a correlated node can supply and a sigma dict is
    not, so this is the producer a ``CirculantNormal`` observation reaches the
    solver through. For a diagonal model the two agree by construction, and
    ``test_precision_at_agrees_with_diagonal_from_noise_std_at`` pins that:
    this is not a second implementation of the diagonal case, it is
    :func:`precision_parts` over the same nodes ``observation_parts`` walks.

    Traceable and **unchecked**, like the parts it is built from. Pair it with
    :func:`check_gaussian` (diagonal) or
    :func:`~bayesmith.exact.precision.check_precision` (either), which run
    eagerly at the boundary.
    """
    env = evaluate(graph, values)
    return {
        name: precision_parts(graph, graph.node(name), env)[1]
        for name in graph.observed
    }
