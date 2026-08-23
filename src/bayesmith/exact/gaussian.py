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
and keep both attributes, so the type is evidence and not proof. The probe is
what turns it into proof, at a cost of four ``log_prob`` evaluations per node
per block build.
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
#: ``loc`` from a correct one.
PROBE_OFFSETS: tuple[float, ...] = (-3.0, -1.0, 0.5, 2.0)

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
    loc = jnp.asarray(distribution.loc)
    if not jnp.issubdtype(loc.dtype, jnp.inexact):
        raise StructureError(
            f"node {node.name!r} has an integer loc (dtype {loc.dtype}). A "
            "conjugate solve differentiates through loc, so it has to be a "
            "floating dtype -- pass a float to the distribution."
        )
    scale = jnp.broadcast_to(jnp.asarray(distribution.scale), jnp.shape(loc))
    return loc, scale


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
    loc, _ = gaussian_parts(graph, node, env)
    shapes: list[tuple[int, ...]] = [jnp.shape(loc)]
    if node.plate:
        shapes.append((graph.plate_size(node.plate[0]),))
    if isinstance(node, Probabilistic) and node.observed is not None:
        shapes.append(jnp.shape(node.observed))
    return tuple(jnp.broadcast_shapes(*shapes))


def check_gaussian(
    graph: Graph, node: Node, env: dict[str, Any], *, rtol: float | None = None
) -> dict[float, float]:
    """Verify the extracted ``(loc, scale)`` really reproduce ``node``'s log_prob.

    Costs ``len(PROBE_OFFSETS)`` evaluations of ``log_prob``. Runs on concrete
    values, **outside** any trace.

    Args:
        graph, node, env: the node under test and the values its parents take.
        rtol: tolerance on the relative disagreement. Default ``1e3 * eps`` of
            ``loc``'s dtype, which leaves room for accumulated roundoff in the
            reduction without admitting a real difference in density.

    Returns:
        ``{offset: relative error}`` -- useful for reporting how Gaussian a
        node is, not only whether it passes.

    Raises:
        NotGaussian: propagated from :func:`gaussian_parts`.
        StructureError: if the scale is not strictly positive and finite, or
            if any probe disagrees by more than ``rtol``.
    """
    distribution = unwrap(apply_probabilistic(graph, node, env))
    loc, scale = gaussian_parts(graph, node, env)
    if not bool(jnp.all(jnp.isfinite(scale) & (scale > 0))):
        raise StructureError(
            f"node {node.name!r} has a scale that is not strictly positive and "
            f"finite (min {float(jnp.min(scale)):g}). A conjugate solve weights "
            "by 1/scale**2, so a zero or negative sigma is an infinite or "
            "negative weight rather than a tight constraint. Add a floor to the "
            "expression that produces it."
        )
    if rtol is None:
        rtol = 1e3 * float(jnp.finfo(loc.dtype).eps)

    errors: dict[float, float] = {}
    for offset in PROBE_OFFSETS:
        probe = loc + offset * scale
        actual = float(jnp.sum(distribution.log_prob(probe)))
        predicted = float(
            jnp.sum(
                -0.5 * ((probe - loc) / scale) ** 2 - jnp.log(scale) - 0.5 * _LOG_2PI
            )
        )
        # Relative to the predicted magnitude, with a floor so a probe that
        # lands where the log-density happens to be ~0 does not divide by it.
        errors[offset] = abs(actual - predicted) / max(abs(predicted), 1.0)
        # NaN must count as a FAILURE: `nan > rtol` is False, so a naive
        # comparison treats an unusable probe as evidence of Gaussianity.
        if not np.isfinite(errors[offset]) or errors[offset] > rtol:
            detail = ", ".join(f"{k:+g}sigma -> {v:.3e}" for k, v in errors.items())
            raise StructureError(
                f"node {node.name!r} is a {type(distribution).__name__}, so its "
                "loc and scale were read off it directly -- but its own log_prob "
                f"does not agree with them (rtol={rtol:.2e}; {detail}). A "
                "Distribution subclass that overrides log_prob keeps both "
                "attributes and changes the density, so the type is evidence "
                "and not proof. The exact path would solve the wrong posterior; "
                "it refuses instead."
            )
    return errors


def observation_parts(
    graph: Graph, env: dict[str, Any]
) -> tuple[dict[str, jax.Array], dict[str, jax.Array], dict[str, jax.Array]]:
    """``({obs: data}, {obs: loc}, {obs: scale})`` over every observed node.

    All three are broadcast to :func:`node_shape`, so the three dicts align
    leaf for leaf and every reduction downstream is one ``jax.tree.map``.
    """
    data: dict[str, jax.Array] = {}
    loc: dict[str, jax.Array] = {}
    scale: dict[str, jax.Array] = {}
    for name in graph.observed:
        node = graph.node(name)
        shape = node_shape(graph, node, env)
        node_loc, node_scale = gaussian_parts(graph, node, env)
        data[name] = jnp.broadcast_to(node.observed, shape)
        loc[name] = jnp.broadcast_to(node_loc, shape)
        scale[name] = jnp.broadcast_to(node_scale, shape)
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
