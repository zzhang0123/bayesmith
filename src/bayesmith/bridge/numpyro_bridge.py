"""Turning a graph into a NumPyro model, and running NUTS on it.

This is the last row of the dispatch table: whatever structure bayesmith
cannot solve exactly is handed to NumPyro. It is also the oracle every exact
path is checked against, because a graph that qualifies for an exact method
always also qualifies for NUTS.
"""

from __future__ import annotations

from collections.abc import Callable
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
    progress_bar: bool = False,
) -> dict[str, jax.Array]:
    """Sample the posterior of ``graph`` with NUTS.

    Args:
        graph: the model.
        key: a PRNG key.
        num_warmup: adaptation draws, discarded.
        num_samples: retained draws per chain.
        num_chains: independent chains.
        progress_bar: whether NumPyro prints progress.

    Returns:
        A mapping from latent node name to its draws.
    """
    mcmc = MCMC(
        NUTS(to_numpyro(graph)),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(key)
    return mcmc.get_samples()
