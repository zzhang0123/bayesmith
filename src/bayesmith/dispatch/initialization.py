"""Bounded starting-point completion in the graph's model coordinates."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.distributions.transforms import biject_to

from bayesmith.artifacts.policies import InitializationPolicy
from bayesmith.graph.evaluate import apply_deterministic, apply_probabilistic, log_joint
from bayesmith.graph.nodes import Const, Deterministic, Probabilistic


def _supplied(policy, graph, chain, chains):
    masks = {x.name: x for x in policy.masks}
    supplied = {}
    for item in policy.values:
        if item.name not in graph.latents:
            raise ValueError(f"initial value {item.name!r} is not a latent")
        value = item.value
        mask = (
            masks[item.name].value if item.name in masks else np.ones(value.shape, bool)
        )
        if item.dims and item.dims[0] == "chain":
            if value.shape[0] != chains:
                raise ValueError(f"initial value {item.name}: expected {chains} chains")
            value, mask = value[chain], mask[chain]
        supplied[item.name] = (value, mask)
    return supplied


def _candidate(graph, key, supplied, radius):
    env, values = {}, {}
    for index, node in enumerate(graph.nodes):
        if isinstance(node, Const):
            env[node.name] = node.value
        elif isinstance(node, Deterministic):
            env[node.name] = apply_deterministic(graph, node, env)
        elif isinstance(node, Probabilistic):
            if not node.is_latent:
                env[node.name] = node.observed
                continue
            fn = apply_probabilistic(graph, node, env)
            batch_shape = tuple(fn.batch_shape)
            if node.plate:
                batch_shape = tuple(
                    jnp.broadcast_shapes(batch_shape, (graph.plate_size(node.plate[0]),))
                )
            shape = batch_shape + tuple(fn.event_shape)
            if fn.support.is_discrete:
                raise ValueError(
                    f"initialization of discrete latent {node.name} is unsupported"
                )
            seed = jax.random.fold_in(key, index)
            # A prior draw is only an initialization candidate. With a mask,
            # overwriting it makes no claim to exact conditional simulation.
            try:
                candidate = fn.expand(batch_shape).sample(seed)
            except NotImplementedError:
                transform = biject_to(fn.support)
                candidate = transform(
                    jax.random.uniform(
                        seed,
                        transform.inverse_shape(shape),
                        minval=-radius,
                        maxval=radius,
                    )
                )
            if node.name in supplied:
                value, mask = supplied[node.name]
                if value.shape != shape:
                    raise ValueError(
                        f"initial value {node.name}: shape {value.shape}, expected {shape}"
                    )
                candidate = jnp.where(mask, jnp.asarray(value), candidate)
            if not bool(jnp.all(fn.support(candidate))):
                raise ValueError(f"initial value {node.name} violates its support")
            values[node.name] = env[node.name] = candidate
    return values


def complete_initial_values(graph, key, policy: InitializationPolicy, chains: int):
    """Return validated values and attempts per chain, without altering inputs.

    A failed complete candidate may be retried because support/likelihoods
    can couple missing coordinates. Supplied values are never clipped.
    """
    points, attempts = [], []
    for chain in range(chains):
        supplied = _supplied(policy, graph, chain, chains)
        last_error = "non-finite joint or gradient"
        chain_key = jax.random.fold_in(key, chain)
        for attempt in range(policy.max_attempts):
            try:
                point = _candidate(
                    graph,
                    jax.random.fold_in(chain_key, attempt),
                    supplied,
                    policy.radius,
                )
                density, gradient = jax.value_and_grad(lambda v: log_joint(graph, v))(
                    point
                )
                valid = np.isfinite(np.asarray(density)).all() and all(
                    np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(gradient)
                )
                if valid:
                    points.append(point)
                    attempts.append(attempt + 1)
                    break
            except (ValueError, TypeError, NotImplementedError) as error:
                last_error = str(error)
                if "shape" in last_error or "unsupported" in last_error:
                    raise ValueError(last_error) from error
        else:
            raise ValueError(
                f"chain {chain}: no valid initial point after {policy.max_attempts} attempts "
                f"for {graph.latents}; {last_error}. Supply values or increase max_attempts."
            )
    return points, tuple(attempts)
