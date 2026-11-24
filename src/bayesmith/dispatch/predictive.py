"""Predictive forward generation: push posterior draws onto observations.

Prediction samples each observed site's declared distribution. Pointwise
replay scores the recorded observation under the same diagonal Gaussian law
at each posterior draw. The dispatch layer packages these arrays in artifacts.

R3 §0.7 adds the two remaining forward primitives -- `prior_draws`, which
walks the whole graph from its priors, and `forward_draws`, which pushes ONE
parameter setting through it -- so that `SimulationTask`'s three parameter
sources share this module rather than growing a simulator each.  That sharing
is structural, not asserted: the fixed arm IS the replicate arm at a parameter
that happens not to vary, and `tests/dispatch/test_predictive_seam.py` compares
them at `rtol=0`.

The predictive seam accepts diagonal Gaussian observations, including
independent complex components. Unsupported observation noise raises
:class:`~bayesmith.errors.NotGaussian`, which the task converts into a refusal.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from bayesmith.artifacts.base import NamedArray
from bayesmith.exact.gaussian import _diagonal_gaussian
from bayesmith.graph.evaluate import (
    _apply_deterministic,
    _law_builder,
    apply_probabilistic,
    distribution_batch_shape,
    evaluate,
)
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Const, Deterministic, Probabilistic

__all__ = [
    "replicated_draws",
    "pointwise_log_likelihood",
    "prior_draws",
    "forward_draws",
]


def _dims(
    graph: Graph, name: str, value: Any, *, draw: bool, environment: Callable
) -> tuple[str, ...]:
    """A declared plate occupies the rightmost batch axis, before the event."""
    axes = np.ndim(value) - int(draw)
    node = graph.node(name)
    names = [f"{name}_dim{i}" for i in range(axes - len(node.plate))]
    if node.plate and axes:
        event_rank = (
            apply_probabilistic(graph, node, environment()).event_dim if axes > 1 else 0
        )
        names.insert(axes - event_rank - 1, node.plate[0])
    if draw:
        names.insert(0, "draw")
    return tuple(names)


def _named_arrays(
    graph: Graph,
    values: Mapping[str, Any],
    latent_values: Mapping[str, Any],
    *,
    draw: bool,
    fixed: bool = False,
    generated: Mapping[str, Any] | None = None,
) -> tuple[NamedArray, ...]:
    @cache
    def environment():
        # Evaluate the graph only if a multi-axis plate needs its event rank.
        point = (
            {name: value[0] for name, value in latent_values.items()}
            if draw and not fixed else latent_values
        )
        if generated is None:
            return evaluate(graph, point)
        return evaluate(
            graph, point, observed_value=lambda node, _: (
                generated[node.name][0] if node.name in generated else node.observed
            ),
        )

    return tuple(
        NamedArray(name, np.asarray(value), _dims(graph, name, value, draw=draw, environment=environment))
        for name, value in values.items()
    )


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


def _masked_log_prob(graph: Graph, node: Probabilistic, env: dict[str, Any]) -> Any:
    """Score each observed scalar under the declared diagonal Gaussian law."""
    distribution = apply_probabilistic(graph, node, env)
    base = _diagonal_gaussian(node, distribution)
    shape = distribution_batch_shape(graph, node, distribution) + distribution.event_shape
    data = jnp.broadcast_to(node.observed, shape)
    log_prob = base.log_prob(data)
    if node.observed_mask is not None:
        log_prob = jnp.where(node.observed_mask, log_prob, 0.0)
    return log_prob


def replicated_draws(
    graph: Graph,
    latent_draws: Mapping[str, Any],
    key: jax.Array,
    *,
    conditioned_sites: frozenset[str] = frozenset(),
) -> dict[str, jax.Array]:
    """`{obs: draws}` -- one replicated draw per source draw, per observed node.

    Args:
        graph: the model.
        latent_draws: every latent's posterior draws, each with the draw axis
            leading.  They must cover every latent `evaluate` needs.
        key: PRNG key, split per draw and per observed site.
        conditioned_sites: sites held at their recorded values while generating
            descendants. All other observations are drawn in graph order.

    Returns:
        One array per generated site, shape `(draws, *node_shape)`, sampled from
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

    names = graph.observed
    unknown = conditioned_sites.difference(names)
    if unknown:
        raise ValueError(f"conditioned_sites names unknown observations {sorted(unknown)}")

    def per_draw(draw_values: dict[str, Any], draw_key: jax.Array) -> dict[str, Any]:
        # Preserve the single-site stream while giving every site in a joint
        # replication its own noise, including when only a subset is returned.
        site_keys = (
            (draw_key,) if len(names) == 1 else jax.random.split(draw_key, len(names))
        )
        keys_by_name = dict(zip(names, site_keys, strict=True))

        def observed_value(node: Probabilistic, env: dict[str, Any]) -> Any:
            if node.name in conditioned_sites:
                return node.observed
            distribution = apply_probabilistic(graph, node, env)
            _diagonal_gaussian(node, distribution)
            return _node_draw(graph, node, distribution, keys_by_name[node.name])

        env = evaluate(graph, draw_values, observed_value=observed_value)
        return {name: env[name] for name in names if name not in conditioned_sites}

    in_axes = (jax.tree.map(lambda _: 0, draws), 0)
    return jax.vmap(per_draw, in_axes=in_axes)(draws, keys)


def pointwise_log_likelihood(
    graph: Graph, latent_draws: Mapping[str, Any]
) -> NamedArray | None:
    """The observed-data replay: pointwise log-likelihood, indexed by draw first.

    For every source posterior draw, evaluates the declared scalar log density
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
        return {
            name: _masked_log_prob(graph, graph.node(name), env)
            for name in names
        }

    per_node = jax.vmap(per_draw)(draws)

    if len(names) == 1:
        name = names[0]
        array = np.asarray(per_node[name])
        dims = _dims(
            graph, name, array, draw=True,
            environment=lambda: evaluate(graph, {n: v[0] for n, v in draws.items()}),
        )
        return NamedArray("log_likelihood", array, dims)

    units = [jnp.reshape(per_node[name], (_count, -1)) for name in names]
    array = np.asarray(jnp.concatenate(units, axis=1))
    return NamedArray(name="log_likelihood", value=array, dims=("draw", "observation"))


# ------------------------------------------------------------ the §0.7 seam


def _node_draw(
    graph: Graph, node: Probabilistic, distribution: Any, key: jax.Array
) -> Any:
    """Sample independent batch entries, including recorded repetitions."""
    batch = distribution_batch_shape(graph, node, distribution)
    return distribution.expand(batch).sample(key)


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
        law_for = _law_builder(graph, env)
        for node, node_key in zip(
            nodes, jax.random.split(draw_key, len(nodes)), strict=True
        ):
            if isinstance(node, Const):
                env[node.name] = node.value
            elif isinstance(node, Deterministic):
                env[node.name] = _apply_deterministic(graph, node, env, law_for)
            else:
                distribution = law_for(node.name)
                env[node.name] = _node_draw(graph, node, distribution, node_key)
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
