"""Turning a graph into a NumPyro model, and running NUTS on it.

This is the last row of the dispatch table: whatever structure bayesmith
cannot solve exactly is handed to NumPyro. It is also the oracle every exact
path is checked against, because a graph that qualifies for an exact method
always also qualifies for NUTS.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax
import numpyro
from numpyro.infer import MCMC, NUTS

from bayesmith.graph.evaluate import apply_deterministic, apply_probabilistic
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Const, Deterministic, Probabilistic


def to_numpyro(graph: Graph) -> Callable[[], dict[str, Any]]:
    """Build a NumPyro model that declares the same joint distribution.

    Latent and observed nodes become ``numpyro.sample`` sites carrying the
    graph's own node names, so posterior samples come back keyed by them.
    Deterministic nodes are recorded with ``numpyro.deterministic`` so they
    appear in traces and predictives without contributing a density.
    """

    def model() -> dict[str, Any]:
        env: dict[str, Any] = {}
        for node in graph.nodes:
            if isinstance(node, Const):
                env[node.name] = node.value
            elif isinstance(node, Deterministic):
                env[node.name] = numpyro.deterministic(
                    node.name, apply_deterministic(graph, node, env)
                )
            elif isinstance(node, Probabilistic):
                distribution = apply_probabilistic(graph, node, env)
                if node.plate:
                    name = node.plate[0]
                    with numpyro.plate(name, graph.plate_size(name)):
                        env[node.name] = numpyro.sample(
                            node.name, distribution, obs=node.observed
                        )
                else:
                    env[node.name] = numpyro.sample(
                        node.name, distribution, obs=node.observed
                    )
        return env

    return model


def nuts(
    graph: Graph,
    key: jax.Array,
    *,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 1,
    chain_method: str = "sequential",
    progress_bar: bool = False,
    nuts_options: Mapping[str, Any] | None = None,
) -> dict[str, jax.Array]:
    """Sample the posterior of ``graph`` with NUTS.

    ``chain_method`` and ``nuts_options`` are here because
    :meth:`~bayesmith.dispatch.plan.InferencePlan.sample` promises them on
    every path, and two of its five shapes -- the graph with no exact block,
    and the SNIS collapse -- run through this function. Until they existed
    those two shapes silently ignored both keywords while the mixed shape,
    which reaches ``HMCGibbs`` through
    :func:`~bayesmith.exact.gibbs.assemble`, honoured them. The names and the
    defaults are ``assemble``'s, so the two spellings of "run a chain" take
    the same words.

    Args:
        graph: the model.
        key: a PRNG key.
        num_warmup: adaptation draws, discarded.
        num_samples: retained draws per chain.
        num_chains: independent chains.
        chain_method: how ``num_chains`` are run -- ``"sequential"``,
            ``"parallel"`` or ``"vectorized"``. All three are numpyro's own
            and all three are legal here; ``assemble`` refuses
            ``"vectorized"`` for a Gibbs sweep, but that refusal is about
            ``HMCGibbs.init`` and does not apply to a bare kernel.
        progress_bar: whether NumPyro prints progress.
        nuts_options: keywords for the ``NUTS`` kernel itself
            (``target_accept_prob``, ``max_tree_depth``, ``dense_mass``, ...).

    Returns:
        A mapping from latent node name to its draws.
    """
    mcmc = MCMC(
        NUTS(to_numpyro(graph), **dict(nuts_options or {})),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=progress_bar,
    )
    mcmc.run(key)
    return mcmc.get_samples()
