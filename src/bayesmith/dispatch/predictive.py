"""Predictive forward generation: push posterior draws onto observations.

The two primitives of the R2 predictive seam, both built on the SAME loc/scale
:func:`bayesmith.exact.gaussian.observation_parts` reads (§0.1): replay and
replicate differ only by log_prob vs sample, never by a second, parallel
forward model.  This module is part of the dispatch bridge -- it reads a Graph
and JAX, and returns a :class:`~bayesmith.artifacts.base.NamedArray` for the
pointwise log-likelihood -- and it never reaches the higher artifact protocol
(results, tasks, identity, refusal): those are `dispatch.task`'s business.

R3 §0.7 adds the two remaining forward primitives -- `prior_draws`, which
walks the whole graph from its priors, and `forward_draws`, which pushes ONE
parameter setting through it -- so that `SimulationTask`'s three parameter
sources share this module rather than growing a simulator each.  That sharing
is structural, not asserted: the fixed arm IS the replicate arm at a parameter
that happens not to vary, and `tests/dispatch/test_predictive_seam.py` compares
them at `rtol=0`.

**Generation law (§0.3).**  A replicated draw is `Normal(loc, scale).sample`
at the node's own observation distribution, one per source posterior draw
(draw axis one-to-one, no resampling).  This is the node's own distribution --
for a diagonal Gaussian it is exactly `apply_probabilistic(...).sample` -- so
density and generator are one object, not a hand-written simulator.

**Coverage domain (§0.4).**  `observation_parts` is a diagonal walk, so a
correlated or non-Gaussian observed node raises
:class:`~bayesmith.errors.NotGaussian` here rather than being silently
approximated.  The caller turns that into a typed Refusal.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from bayesmith.artifacts.base import NamedArray
from bayesmith.exact.gaussian import observation_parts
from bayesmith.graph.evaluate import apply_deterministic, apply_probabilistic, evaluate
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Const, Deterministic, Probabilistic

__all__ = [
    "replicated_draws",
    "pointwise_log_likelihood",
    "prior_draws",
    "forward_draws",
]


def _dims(graph: Graph, name: str, value: Any, *, draw: bool) -> tuple[str, ...]:
    """One name per axis: draw, the node's plate, then its own shape.

    The §0.2 axis-naming rule, shared with `dispatch.task`.  A plate has a
    NAME the model declared, so a plated node's second axis is called what the
    model calls it rather than `dim0`; axes the graph has no name for get one
    derived from the site.
    """
    remaining = int(np.ndim(value))
    names: list[str] = []
    if draw:
        names.append("draw")
        remaining -= 1
    for plate in graph.node(name).plate:
        if remaining <= 0:
            break
        names.append(plate)
        remaining -= 1
    names.extend(f"{name}_dim{index}" for index in range(max(remaining, 0)))
    return tuple(names)


def _draw_count(latent_draws: Mapping[str, Any]) -> int:
    """The leading draw count every latent shares, refused where they disagree."""
    counts = {int(np.shape(jnp.asarray(value))[0]) for value in latent_draws.values()}
    if len(counts) != 1:
        raise ValueError(
            "latent_draws disagree about the draw count: "
            + ", ".join(
                f"{name!r} has {np.shape(jnp.asarray(value))[0]}"
                for name, value in latent_draws.items()
            )
        )
    return counts.pop()


def _masked_log_prob(graph: Graph, name: str, loc: Any, scale: Any, data: Any) -> Any:
    """`Normal(loc, scale).log_prob(data)`, with never-taken samples zeroed.

    The mask is read off the NODE (as `log_joint` reads it), so the pointwise
    log-likelihood and the joint agree about which samples contribute a term.
    """
    node = graph.node(name)
    log_prob = dist.Normal(loc, scale).log_prob(data)
    if node.observed_mask is not None:
        log_prob = jnp.where(node.observed_mask, log_prob, 0.0)
    return log_prob


def replicated_draws(
    graph: Graph, latent_draws: Mapping[str, Any], key: jax.Array
) -> dict[str, jax.Array]:
    """`{obs: draws}` -- one replicated draw per source draw, per observed node.

    Args:
        graph: the model.
        latent_draws: every latent's posterior draws, each with the draw axis
            leading.  They must cover every latent `evaluate` needs.
        key: PRNG key, split once per source draw so the replicated draws do
            not share a stream.

    Returns:
        One array per observed node, shape `(draws, *node_shape)`, sampled from
        the node's own observation distribution at each source draw's loc/scale.

    Raises:
        NotGaussian: for a correlated or non-Gaussian observed node (§0.4) --
            a classification outcome the caller turns into a Refusal, not a
            silent approximation.
    """
    if not isinstance(graph, Graph):
        raise TypeError(f"replicated_draws' graph is a Graph; got {graph!r}")
    draws = dict(latent_draws)
    count = _draw_count(draws)
    keys = jax.random.split(key, count)

    def per_draw(draw_values: dict[str, Any], draw_key: jax.Array) -> dict[str, Any]:
        env = evaluate(graph, draw_values)
        _data, loc, scale = observation_parts(graph, env)
        return {
            name: dist.Normal(loc[name], scale[name]).sample(draw_key)
            for name in graph.observed
        }

    in_axes = (jax.tree.map(lambda _: 0, draws), 0)
    return jax.vmap(per_draw, in_axes=in_axes)(draws, keys)


def pointwise_log_likelihood(
    graph: Graph, latent_draws: Mapping[str, Any]
) -> NamedArray | None:
    """The observed-data replay: pointwise log-likelihood, indexed by draw first.

    For every source posterior draw, evaluates `Normal(loc, scale).log_prob(data)`
    at the observed node's own data -- the conditioning half, NOT a posterior
    predictive (§0.1) -- zeroing masked positions.  The leading axis is `draw`;
    the remaining axes are observation units named by `_dims` (a single
    observed node keeps its plate / `{name}_dim{i}` axes, several observed
    nodes are concatenated into one `observation` axis in declaration order).

    Returns:
        A `NamedArray` named `log_likelihood`, or `None` when the graph has no
        observed node -- there is no observation unit to score, so the caller
        abstains rather than fabricating a pointwise density.

    Raises:
        NotGaussian: for a correlated or non-Gaussian observed node (§0.4).
    """
    if not isinstance(graph, Graph):
        raise TypeError(f"pointwise_log_likelihood's graph is a Graph; got {graph!r}")
    draws = dict(latent_draws)
    _count = _draw_count(draws)
    names = tuple(graph.observed)
    if not names:
        return None

    def per_draw(draw_values: dict[str, Any]) -> dict[str, Any]:
        env = evaluate(graph, draw_values)
        data, loc, scale = observation_parts(graph, env)
        return {
            name: _masked_log_prob(graph, name, loc[name], scale[name], data[name])
            for name in names
        }

    per_node = jax.vmap(per_draw)(draws)

    if len(names) == 1:
        name = names[0]
        array = np.asarray(per_node[name])
        return NamedArray(
            name="log_likelihood", value=array, dims=_dims(graph, name, array, draw=True)
        )

    units = [jnp.reshape(per_node[name], (_count, -1)) for name in names]
    array = np.asarray(jnp.concatenate(units, axis=1))
    return NamedArray(name="log_likelihood", value=array, dims=("draw", "observation"))


# ------------------------------------------------------------ the §0.7 seam


def _plate_shape(graph: Graph, node: Probabilistic) -> tuple[int, ...]:
    """The declared sizes of the plates ``node`` sits under."""
    return tuple(graph.plate_size(name) for name in node.plate)


def _node_draw(
    graph: Graph, node: Probabilistic, env: Mapping[str, Any], key: jax.Array
) -> Any:
    """One draw from ``node``'s own distribution, expanded to its plate.

    :func:`~bayesmith.graph.evaluate.apply_probabilistic` deliberately returns
    an UNMAPPED distribution for a plated node whose parents are not plated --
    that is the "N iid draws from one shared prior" pattern, and its
    ``log_prob`` broadcasts across the plate correctly, which is why
    ``log_joint`` and the numpyro bridge are both happy with it.  Its
    ``.sample``, though, returns ONE value, so a forward walk that trusted it
    would write the same latent at every plate index and look entirely
    plausible doing it (probe_28 §8 refused to guess and left the decision
    here).  §0.7 expands with ``graph.plate_size``.

    Which case it is, is read off the distribution's own ``batch_shape``
    rather than by re-deriving whether ``apply_probabilistic`` vmapped: that
    decision has a home, and a second copy of it here would agree with the
    first until the day it did not.  A distribution already carrying the plate
    in its batch shape is left alone; anything else is drawn at
    ``sample_shape`` equal to the plate.
    """
    distribution = apply_probabilistic(graph, node, env)
    plate = _plate_shape(graph, node)
    batch = tuple(distribution.batch_shape)
    if plate and batch[: len(plate)] != plate:
        return distribution.sample(key, plate)
    return distribution.sample(key)


def prior_draws(graph: Graph, key: jax.Array, n: int) -> dict[str, jax.Array]:
    """`{node: draws}` for EVERY node, drawn from the model's priors forward.

    Observed nodes are drawn too, from their own distributions rather than
    carried as data -- so this is the prior PREDICTIVE, and the same primitive
    answers both of the questions R3 asks of it: the `(theta, y)` pairs SBC
    replicates over (§0.6) and the prior predictive check's replicated
    datasets (§0.2).  A version that stopped at the latents would need a
    second pass to reach the data, which is the parallel simulator invariant 1
    forbids.

    **Key discipline: one key per node, in ``graph.nodes`` order.**  Const and
    Deterministic nodes consume a key they never use.  That is not tidiness --
    it is what probe_28 §8 measured, and reproducing it bit for bit is what
    lets the plan's ``P(max|d_prior| >= max|d_obs|) = 0.210`` be CHECKED
    against this function rather than merely quoted beside it
    (``tests/dispatch/test_predictive_seam.py`` does exactly that).  The cost
    is that ``jax.random.split(k, len(nodes))`` reshuffles every node's stream
    when a node is added or removed; that is acceptable because such an edit
    already moves the GRAPH_STRUCTURE fingerprint, so by the protocol's own
    reckoning it is a different run and not a reproducibility claim that broke.

    Args:
        graph: the model.
        key: PRNG key, split once per draw so the draws do not share a stream.
        n: how many draws.  Never defaulted: a draw count is a budget, and
            §0.7 puts it in the task's ``budget.draws``.

    Returns:
        One array per node, shape `(n, *node_shape)`, keyed by node name.
    """
    if not isinstance(graph, Graph):
        raise TypeError(f"prior_draws' graph is a Graph; got {graph!r}")
    count = int(n)
    if count < 1:
        raise ValueError(f"prior_draws needs at least one draw; got {n!r}")
    nodes = graph.nodes

    def one(draw_key: jax.Array) -> dict[str, Any]:
        env: dict[str, Any] = {}
        for node, node_key in zip(
            nodes, jax.random.split(draw_key, len(nodes)), strict=True
        ):
            if isinstance(node, Const):
                env[node.name] = node.value
            elif isinstance(node, Deterministic):
                env[node.name] = apply_deterministic(graph, node, env)
            else:
                env[node.name] = _node_draw(graph, node, env, node_key)
        return env

    return jax.vmap(one)(jax.random.split(key, count))


def forward_draws(
    graph: Graph, values: Mapping[str, Any], key: jax.Array, n: int
) -> dict[str, jax.Array]:
    """`{obs: draws}` -- `n` observations generated at ONE parameter setting.

    §0.7's fixed arm.  Implemented by repeating the setting along a draw axis
    and calling :func:`replicated_draws`, which is the point rather than a
    shortcut: the fixed source and the posterior source then share ONE
    generation law by construction, and a future divergence between them is a
    change to one function instead of a drift between two that were never
    compared.  The price is `n` identical `evaluate` calls under the vmap,
    paid knowingly -- a second, cheaper path is exactly the parallel forward
    model §0.1 spent R2 removing.

    Args:
        graph: the model.
        values: the parameter setting, one entry per latent `evaluate` needs.
        key: PRNG key, split once per draw.
        n: how many observations to generate.

    Returns:
        One array per observed node, shape `(n, *node_shape)`.

    Raises:
        NotGaussian: for a correlated or non-Gaussian observed node (§0.4),
            through the same diagonal walk its sibling refuses on.
    """
    if not isinstance(graph, Graph):
        raise TypeError(f"forward_draws' graph is a Graph; got {graph!r}")
    count = int(n)
    if count < 1:
        raise ValueError(f"forward_draws needs at least one draw; got {n!r}")
    repeated = {
        name: jnp.broadcast_to(
            jnp.asarray(value), (count, *jnp.shape(jnp.asarray(value)))
        )
        for name, value in values.items()
    }
    return replicated_draws(graph, repeated, key)
