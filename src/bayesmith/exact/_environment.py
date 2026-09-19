"""Shared prior anchors for exact analysis, planning and runtime adapters.

The anchor is numerical graph evaluation, independent of route selection.
Classification re-exports these functions for existing callers; every consumer
uses the same implementation, including non-Gaussian and dependent priors.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from bayesmith.errors import GraphError, NotGaussian
from bayesmith.exact.gaussian import gaussian_parts, node_shape
from bayesmith.graph.evaluate import apply_deterministic, apply_probabilistic
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Const, Deterministic, Probabilistic


def _latent_centre(graph: Graph, node: Probabilistic, env: dict[str, Any]) -> jax.Array:
    """One latent's centre, for an environment the classifier can run in.

    A Gaussian latent's centre is its ``loc``, read the same way the block
    machinery reads it. A latent that is not Gaussian has no ``loc`` at all --
    and the classifier still needs a value for it, because a DISQUALIFIED
    latent can sit upstream of an observed node whose Gaussianity is exactly
    what is being checked. Its own distribution's ``mean`` is the natural
    second choice, and zero is the third: ``Cauchy`` has no finite mean and
    ``ImproperUniform`` has no mean at all (measured -- ``NotImplementedError``
    from NumPyro 0.21). Both of those are live fixtures here
    (``overflowing_outside_latent``, ``improper_outside_prior``) and both are
    LEGAL models the classifier must not refuse.

    A NumPyro distribution does not know the plate its ``sample()`` named --
    ``dist.StudentT(6.0, 0.4, 0.9).shape()`` is ``()`` either way -- so that
    second choice has to be broadcast out to the plate's size by hand.
    ``plated_student_t_latent`` is the fixture; without the broadcast
    ``apply_deterministic`` raises out of ``vmap`` rather than returning a
    wrong centre, so nothing downstream papers over it.
    """
    try:
        loc, _ = gaussian_parts(graph, node, env)
        return jnp.broadcast_to(loc, node_shape(graph, node, env))
    except NotGaussian:
        pass
    distribution = apply_probabilistic(graph, node, env)
    shape = tuple(distribution.shape())
    if node.plate:
        shape = tuple(jnp.broadcast_shapes(shape, (graph.plate_size(node.plate[0]),)))
    try:
        centre = jnp.broadcast_to(jnp.asarray(distribution.mean), shape)
    except (NotImplementedError, AttributeError, TypeError, ValueError):
        return jnp.zeros(shape)
    return centre if bool(jnp.all(jnp.isfinite(centre))) else jnp.zeros(shape)


def prior_environment(graph: Graph) -> dict[str, Any]:
    """Every node's value, with each latent at the centre of its own prior.

    Repeats :func:`~bayesmith.graph.evaluate.evaluate`'s isinstance ladder for
    the same reason :func:`~bayesmith.exact.block._env_before` does: the
    values it substitutes have to be derived DURING the scan, from what is
    already in hand. A latent whose prior width is another latent's value --
    ``shared_ancestor``, ``mixed_radiometer`` -- cannot be centred before its
    ancestor has been.

    Public because :mod:`bayesmith.dispatch.plan` must anchor the block it
    BUILDS at exactly the point :func:`partition` classified it at. Two
    independent spellings of "the prior mean" would let the plan solve a
    different problem from the one that was checked.
    """
    env: dict[str, Any] = {}
    for node in graph.nodes:
        if isinstance(node, Const):
            env[node.name] = node.value
        elif isinstance(node, Deterministic):
            env[node.name] = apply_deterministic(graph, node, env)
        elif isinstance(node, Probabilistic):
            env[node.name] = (
                _latent_centre(graph, node, env) if node.is_latent else node.observed
            )
        else:  # pragma: no cover - defensive, mirrors evaluate()
            raise GraphError(f"unknown node type {type(node).__name__}")
    return env
