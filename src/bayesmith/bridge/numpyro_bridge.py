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
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from bayesmith.distributions import ComplexNormal
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
                if isinstance(distribution, ComplexNormal):
                    env[node.name] = _complex_site(node, distribution)
                elif node.plate:
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


def _complex_site(node: Any, distribution: ComplexNormal) -> jax.Array:
    """A complex latent as two real sites plus the deterministic that joins them.

    HMC's transforms are defined on real unconstrained space, so a sampler
    cannot step in C. Reparameterising here is what keeps this module's
    opening claim true -- *a graph that qualifies for an exact method always
    also qualifies for NUTS* -- rather than leaving one node type as a
    standing exception to it. The exact path for a complex latent splits into
    the same two real degrees of freedom
    (:func:`~bayesmith.exact.block.real_parts`), so the two routes are
    reparameterised the same way and stay comparable, which is the whole point
    of NUTS being the oracle.

    The graph's own name still carries the complex value, as a
    ``numpyro.deterministic``: posterior samples come back keyed by node name
    like every other node, and the two real sites are visible beside it under
    suffixed names for anyone reading diagnostics per degree of freedom.

    An observed complex node would need the value split too, and nothing
    declares one yet -- refused by name rather than silently sampled as a
    latent, which is what the ``obs=`` argument being dropped would amount to.
    """
    if node.observed is not None:
        raise NotImplementedError(
            f"node {node.name!r} is an OBSERVED ComplexNormal. The exact paths "
            "read complex LATENTS; a complex observation would have to split "
            "its data as well as its density, and nothing declares one yet. "
            "Refused here rather than sampled as if it were latent."
        )
    if node.plate:
        raise NotImplementedError(
            f"complex latent {node.name!r} is plated. The two real sites would "
            "each need the plate, and no fixture exercises it yet -- refused "
            "rather than emitted untested."
        )
    real = numpyro.sample(
        f"{node.name}__re", dist.Normal(jnp.real(distribution.loc), distribution.scale)
    )
    imag = numpyro.sample(
        f"{node.name}__im", dist.Normal(jnp.imag(distribution.loc), distribution.scale)
    )
    return numpyro.deterministic(node.name, real + 1j * imag)


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

            **``init_strategy`` goes here, and on a narrow posterior it is
            not a tuning knob.** NumPyro's default is ``init_to_uniform``,
            which draws in the UNCONSTRAINED space with no knowledge of
            where the graph's priors sit; a posterior far narrower than its
            prior is a needle, and warmup then adapts a step size for
            wherever it landed in the haystack. Measured on a power law
            whose amplitude has a prior of 1e4 and a posterior width of
            0.4, two chains of 400:

            ==============================================  ======  ======
            init                                            r_hat   ESS
            ==============================================  ======  ======
            default (``init_to_uniform``)                    1609     1.0
            ``init_to_value`` at the declared values          1.006  138.6
            ==============================================  ======  ======

            The declared point does not have to be good -- only somewhere a
            gradient can be followed. This is rheplicant's
            ``init_to_declared`` lesson, and the lesson is what was carried:
            the remedy already exists here as a keyword, so what was missing
            was the sentence saying so.

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


def predict(
    graph: Graph, samples: Mapping[str, Any], key: jax.Array | None = None
) -> dict[str, jax.Array]:
    """Posterior predictive: every node's value over a stack of draws.

    Args:
        graph: the model the draws came from.
        samples: ``{latent name: (n_draws, *latent shape)}`` -- what
            :func:`nuts` returns, unchanged.
        key: PRNG key for any node the draws do not fix. Fixed by default,
            so a predictive is reproducible.

    Returns:
        ``{node name: (n_draws, *node shape)}``, deterministic nodes
        included.

    Raises:
        GraphError: if a latent is missing from ``samples``; if a stack's
            PER-SAMPLE shape is not the latent's own; or if the stacks
            disagree about how many draws there are.

    Note:
        **Why the per-sample shape is checked and not just the name.** This
        is rheplicant's ``predict_from_samples`` guard, ported because the
        failure it prevents is reachable here and is silent. Measured on a
        length-3 latent with three draws: handing the stack in TRANSPOSED
        returns a finite, correctly-shaped ``(3, 3)`` predictive whose every
        entry is wrong --

            correct     [[0, 2, 6], [3, 8, 15], [6, 14, 24]]
            transposed  [[0, 6, 18], [1, 8, 21], [2, 10, 24]]

        -- because NumPyro's ``Predictive`` maps over the leading axis and
        has no independent statement of what the latent's shape is. The
        graph does have one: each latent's ``dist_fn``. A non-square
        transposition raises a broadcast error from three layers down that
        names neither the site nor the axis; a square one raises nothing at
        all.
    """
    from numpyro.infer import Predictive

    from bayesmith.dispatch.classify import prior_environment
    from bayesmith.errors import GraphError

    declared = prior_environment(graph)
    draws: set[int] = set()
    for name in graph.latents:
        if name not in samples:
            raise GraphError(
                f"samples is missing latent {name!r}; available: "
                f"{sorted(samples)}. A predictive needs every latent the "
                "graph declares, because the deterministic nodes read them."
            )
        expected = jnp.shape(declared[name])
        got = jnp.shape(samples[name])
        if got[1:] != expected:
            raise GraphError(
                f"samples[{name!r}] has per-sample shape {got[1:]}, but the "
                f"latent is {expected} (its full stack is {got}). The LEADING "
                "axis must be the draw axis. Checking only the name would let "
                "a transposed stack broadcast into the prediction and return a "
                "finite, correctly-shaped, wrong predictive -- silently, "
                "whenever the draw count happens to equal the latent's size."
            )
        draws.add(got[0])
    if len(draws) > 1:
        raise GraphError(
            f"the sample stacks disagree about the number of draws: "
            f"{sorted(draws)}. They must all come from one run."
        )
    predictive = Predictive(to_numpyro(graph), posterior_samples=dict(samples))
    return predictive(jax.random.key(0) if key is None else key)
