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
from bayesmith.graph.nodes import Const, Deterministic, Node, Probabilistic

Env = dict[str, Any]


def _plate_in_axes(graph: Graph, node: Node) -> tuple[int | None, ...]:
    """vmap ``in_axes`` for ``node``'s parents.

    A parent shares the mapped axis (``0``) iff its plate equals the node's
    own plate; every other parent is broadcast (``None``), unmapped. Shared
    by :func:`apply_deterministic` and :func:`apply_probabilistic` so this
    rule is defined in exactly one place instead of twice -- the failure
    mode Bug 1 hardens against was two independent scans agreeing with each
    other while both being wrong, and duplicated ``in_axes`` logic is
    exactly how that kind of drift starts.
    """
    return tuple(
        0 if graph.node(parent).plate == node.plate else None for parent in node.parents
    )


def apply_deterministic(graph: Graph, node: Deterministic, env: Env) -> Any:
    """Call ``node.fn``, vmapped over the plate the node lives in.

    Public because the NumPyro bridge re-runs the same scan and must apply
    plates identically; a private name used across modules is not private.
    """
    args = [env[parent] for parent in node.parents]
    if not node.plate:
        return node.fn(*args)

    in_axes = _plate_in_axes(graph, node)
    if all(axis is None for axis in in_axes):
        raise GraphError(
            f"deterministic node {node.name!r} is in plate {node.plate[0]!r} but "
            "none of its parents are, so there is nothing to map over. Either put "
            "a parent in the plate, or drop the plate and let broadcasting do it."
        )
    return jax.vmap(node.fn, in_axes=in_axes)(*args)


def apply_probabilistic(graph: Graph, node: Probabilistic, env: Env) -> Any:
    """Build ``node``'s distribution, vmapping ``dist_fn`` over its plate.

    Public for the same reason as :func:`apply_deterministic`: ``log_joint``
    and the NumPyro bridge must build the *same* distribution for the *same*
    node, so both call this one function instead of each independently
    calling ``node.dist_fn`` unmapped. Before this function existed, both
    call sites made that same independent (and wrong) choice, which is
    exactly why the bridge/log_joint cross-check could not catch it -- two
    scans with an identical blind spot agree with each other by
    construction, whether or not they agree with the truth.

    Deliberately does **not** raise when a plated node has no plated parent,
    unlike :func:`apply_deterministic`. That refusal exists there because
    ``fn``'s return value is an arbitrary array that must already be the
    right shape -- there is no fallback that is still correct, so refusing
    is the only safe option. A distribution has a real fallback: calling
    ``dist_fn`` once, unmapped, yields one shared distribution whose
    ``log_prob``/``sample`` already know how to broadcast themselves against
    a plate-shaped value -- plain array broadcasting in ``log_joint``, and
    ``numpyro.plate``'s own ``.expand()`` in ``to_numpyro``. That is not a
    workaround, it is the ordinary "N iid draws from one shared prior"
    pattern -- already relied on today by a plated node with *no* parents at
    all (see ``test_a_plated_latent_site_carries_the_plate_axis``). Refusing
    it here would break that already-correct, already-tested pattern for no
    safety benefit: there is no shape for dist_fn's output to get wrong,
    only a distribution object whose own broadcasting contract handles it.
    """
    args = [env[parent] for parent in node.parents]
    if not node.plate:
        return node.dist_fn(*args)

    in_axes = _plate_in_axes(graph, node)
    if all(axis is None for axis in in_axes):
        return node.dist_fn(*args)
    return jax.vmap(node.dist_fn, in_axes=in_axes)(*args)


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
        name = min(unknown)
        if name not in graph.names:
            raise GraphError(
                f"values names {name!r}, which does not name any node in "
                f"this graph. Latents are {list(graph.latents)}."
            )
        if isinstance(graph.node(name), Probabilistic):
            # In graph.names but excluded from graph.latents above, and
            # Probabilistic: the only way to be both is to be observed.
            raise GraphError(
                f"values names {name!r}, which is an observed node: its "
                "value comes from observe(..., obs=...), not from `values`. "
                f"Latents are {list(graph.latents)}."
            )
        raise GraphError(
            f"values names {name!r}, which is a deterministic or constant "
            "node: its value is computed, not supplied via `values`. "
            f"Latents are {list(graph.latents)}."
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
        else:
            # Live defensive code, not dead code: Node is a plain eqx.Module
            # (not an ABC) and Graph.__check_init__ does not check node
            # subtype, so a hand-built Graph containing a bare Node (or any
            # subclass other than Const/Deterministic/Probabilistic) reaches
            # this branch and raises correctly here.
            raise GraphError(
                f"unknown node type {type(node).__name__}"
            )  # pragma: no cover
    return env


def log_joint(graph: Graph, values: Mapping[str, Any] | None = None) -> jax.Array:
    """The joint log-density of the graph at ``values``.

    Every probabilistic node contributes ``log_prob`` of its value under the
    distribution its parents parameterise; deterministic nodes contribute
    nothing but the dependence they carry.

    A node's :attr:`~bayesmith.graph.nodes.Probabilistic.observed_mask` drops
    the samples it says were never taken. That is the whole of masking on THIS
    side -- there is no infinite scale to take a limit of, because a sample
    that was not observed contributes no term rather than a term whose width
    went to infinity, and the two differ by exactly the ``log sigma`` that
    would have sent the joint to ``-inf``.
    """
    env = evaluate(graph, values)
    total = jnp.zeros(())
    for node in graph.nodes:
        if isinstance(node, Probabilistic):
            distribution = apply_probabilistic(graph, node, env)
            term = distribution.log_prob(env[node.name])
            if node.observed_mask is not None:
                term = jnp.where(node.observed_mask, term, 0.0)
            total = total + jnp.sum(term)
    if graph.joint_prior is not None:
        # A density over SEVERAL latents, so it is a term of the joint and not
        # a node's own. Read HERE and in `to_numpyro`, from the one
        # declaration, because the two are two scans of one graph and a prior
        # honoured by one of them is a different posterior wearing the same
        # model's name.
        total = total + graph.joint_prior.log_density(
            graph, {name: env[name] for name in graph.latents}
        )
    return total
