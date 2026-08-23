"""Running a graph forward, and reading its log-density.

Evaluation is one linear scan in topological order: every node's parents are
already in the environment by the time it is reached, which is what the
ordering rule in :class:`~bayesmith.graph.graph.Graph` buys.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from bayesmith.errors import GraphError
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Const, Deterministic, Probabilistic

Env = dict[str, Any]


def apply_deterministic(graph: Graph, node: Deterministic, env: Env) -> Any:
    """Call ``node.fn``, vmapped over the plate the node lives in.

    Public because the NumPyro bridge re-runs the same scan and must apply
    plates identically; a private name used across modules is not private.
    """
    args = [env[parent] for parent in node.parents]
    if not node.plate:
        return node.fn(*args)

    in_axes = tuple(
        0 if graph.node(parent).plate == node.plate else None
        for parent in node.parents
    )
    if all(axis is None for axis in in_axes):
        raise GraphError(
            f"deterministic node {node.name!r} is in plate {node.plate[0]!r} but "
            "none of its parents are, so there is nothing to map over. Either put "
            "a parent in the plate, or drop the plate and let broadcasting do it."
        )
    return jax.vmap(node.fn, in_axes=in_axes)(*args)


def evaluate(graph: Graph, values: Mapping[str, Any] | None = None) -> Env:
    """Compute every node's value.

    Args:
        graph: the graph to run.
        values: values for the latent nodes. Observed nodes take their own
            data and must not appear here; deterministic and constant nodes
            are computed.

    Returns:
        A mapping from node name to value, covering every node in the graph.

    Raises:
        GraphError: if a latent node has no value, or ``values`` names
            something that is not a latent node.
    """
    values = dict(values or {})
    unknown = set(values) - set(graph.latents)
    if unknown:
        raise GraphError(
            f"values names {sorted(unknown)[0]!r}, which is not a latent node of "
            f"this graph. Latents are {list(graph.latents)}."
        )

    env: Env = {}
    for node in graph.nodes:
        if isinstance(node, Const):
            env[node.name] = node.value
        elif isinstance(node, Deterministic):
            env[node.name] = apply_deterministic(graph, node, env)
        elif isinstance(node, Probabilistic):
            if not node.is_latent:
                env[node.name] = node.observed
            elif node.name in values:
                env[node.name] = values[node.name]
            else:
                raise GraphError(
                    f"latent node {node.name!r} has no value. Supply one in "
                    "`values`, or condition the node with observe(...)."
                )
        else:  # pragma: no cover - Node is abstract in practice
            raise GraphError(f"unknown node type {type(node).__name__}")
    return env


def log_joint(graph: Graph, values: Mapping[str, Any] | None = None) -> jax.Array:
    """The joint log-density of the graph at ``values``.

    Every probabilistic node contributes ``log_prob`` of its value under the
    distribution its parents parameterise; deterministic nodes contribute
    nothing but the dependence they carry.
    """
    env = evaluate(graph, values)
    total = jnp.zeros(())
    for node in graph.nodes:
        if isinstance(node, Probabilistic):
            distribution = node.dist_fn(*[env[p] for p in node.parents])
            total = total + jnp.sum(distribution.log_prob(env[node.name]))
    return total
