"""Exact marginalisation of declared discrete latents: P4's enumeration half.

:attr:`~bayesmith.graph.nodes.Probabilistic.support` has carried
``Discrete(n)`` since P1, and its docstring said plainly that nothing read it:
it was "recorded here so the declaration exists from the start". This module is
what reads it, which is what makes the declaration load-bearing rather than
decorative.

**What this computes.** For a graph whose latents are some discrete sites and
some others, the marginal ``log sum_z exp(log_joint(graph, {..., z}))``, summing
over every joint assignment of the declared discrete sites and conditioning on
whatever else the caller supplies. Exact, not approximate: no sampling, no
bound, and no dependence on a random key.

**Why only the exact half.** ``numpyro`` already ships
``DiscreteHMCGibbs`` and ``MixedHMC`` for sampling discrete sites, and the
migration spec is explicit that P4 need only write the exact-marginalisation
side. It could not have been borrowed either way: numpyro's enumeration route
goes through ``numpyro.contrib.funsor``, and ``funsor`` is not a dependency
here -- measured, it does not import in this venv.

**The cost is exponential and is therefore stated, not hidden.** A site with
``n`` states over ``c`` coordinates contributes ``n ** c`` assignments, and the
sites multiply. That is the price of an exact answer rather than a defect, and
the only dishonest thing to do with it is to begin the sum and hope: a caller
who sees a refusal naming 16777216 can raise the budget deliberately, while one
watching a hung process cannot tell it from a bug. So :func:`enumeration_states`
is computable before any arithmetic runs and :data:`ENUMERATION_BUDGET` bounds
it.

**What this does NOT do, and where the next piece goes.** A plated discrete
site is counted honestly at ``n ** plate_size`` and will therefore hit the
budget quickly -- for a mixture assignment per data point, almost immediately.
That is the correct cost *for this algorithm* and the wrong algorithm for that
model: conditionally independent assignments factorise, so the sum is
``size * n`` terms rather than ``n ** size``, and a chain of assignments is
``T * n**2`` by forward-backward rather than ``n ** T``. Both are structural
specialisations of exactly the kind this package dispatches on, and both are
P4's remaining half. Enumeration is the general fallback they must agree with,
and is deliberately built first so that they have an oracle to be checked
against.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Discrete, Probabilistic

__all__ = [
    "ENUMERATION_BUDGET",
    "discrete_latents",
    "enumeration_states",
    "marginal_log_likelihood",
    "posterior_marginals",
]

ENUMERATION_BUDGET: int = 65_536
"""Most joint assignments :func:`marginal_log_likelihood` will visit unasked.

Finite on purpose. An unbounded default would make the refusal unreachable in
any run that mattered, which is indistinguishable from not having one. The
value is a working ceiling rather than a measured threshold -- it is roughly a
second of ``log_joint`` calls on a small graph -- and a caller who wants more
passes ``budget=`` and has thereby said so.
"""


def discrete_latents(graph: Graph) -> tuple[str, ...]:
    """The latent nodes declaring :class:`~bayesmith.graph.nodes.Discrete`.

    In graph order, which is topological, so the enumeration below is
    reproducible rather than set-ordered.

    Three exclusions, each of which would be a different wrong answer:

    * **Observed nodes.** They carry data. Summing a node over states it does
      not occupy replaces the likelihood with something else entirely.
    * **Continuous latents.** Conditioned on, not summed.
    * **Latents with no declared support.** ``None`` is not a claim of
      continuity and not a claim of discreteness -- the field's own docstring
      says it must be ineligible for any support-specific method. So such a
      node is not enumerated, and, having no value either, it is refused by
      the graph's existing check, which names it.
    """
    return tuple(
        node.name
        for node in graph.nodes
        if isinstance(node, Probabilistic)
        and node.is_latent
        and isinstance(node.support, Discrete)
    )


def _coordinates(graph: Graph, name: str) -> int:
    """How many independent latents one site actually holds.

    A plated site is one node and many latents. Counting it as one would
    understate a 3-state site over 4 draws by a factor of 27, and the budget
    that guards the run is computed from this number.
    """
    node = graph.node(name)
    return math.prod(graph.plate_size(p) for p in node.plate)


def enumeration_states(graph: Graph) -> int:
    """How many joint assignments enumeration would visit. Exact, and cheap.

    One when there are no discrete latents -- the empty product. That is not a
    quibble: a sum over zero assignments returns ``-inf``, which is a silent
    and entirely wrong answer for a graph that simply has nothing to
    marginalise.
    """
    return math.prod(
        graph.node(name).support.n ** _coordinates(graph, name)
        for name in discrete_latents(graph)
    )


def _assignments(graph: Graph, names: tuple[str, ...]):
    """Every joint assignment, as ``{site: value}`` with the site's own shape."""
    shapes = [
        tuple(graph.plate_size(p) for p in graph.node(name).plate) for name in names
    ]
    ranges = [
        itertools.product(range(graph.node(name).support.n), repeat=_coordinates(graph, name))
        for name in names
    ]
    for combination in itertools.product(*ranges):
        yield {
            name: jnp.asarray(flat, dtype=jnp.int32).reshape(shape)
            if shape
            else jnp.asarray(flat[0], dtype=jnp.int32)
            for name, flat, shape in zip(names, combination, shapes, strict=True)
        }


def _terms(
    graph: Graph,
    values: Mapping[str, Any] | None,
    budget: int,
) -> tuple[tuple[str, ...], jax.Array]:
    """``log_joint`` at every assignment, and the sites they range over."""
    # Imported here rather than at module scope: `graph.evaluate` is the layer
    # below this one, and `exact` importing it at import time would put a
    # cycle one careless edit away.
    from bayesmith.graph.evaluate import log_joint

    names = discrete_latents(graph)
    states = enumeration_states(graph)
    if states > budget:
        raise ValueError(
            f"enumerating {list(names)} would visit {states} joint assignments, "
            f"past the budget of {budget}. Exact enumeration costs "
            f"n ** coordinates per site; pass `budget=` to allow it "
            f"deliberately, or marginalise a smaller set."
        )
    supplied = dict(values or {})
    stacked = [
        log_joint(graph, {**supplied, **assignment})
        for assignment in _assignments(graph, names)
    ]
    # No `if stacked` fallback, and that absence is load-bearing rather than
    # an oversight. `itertools.product()` over no iterables yields exactly ONE
    # empty tuple, so a graph with no discrete latents produces one assignment
    # -- the empty one -- whose term is the joint itself. That is the correct
    # mathematics (the empty product is one, not zero) and the library already
    # implements it, so a special case here would be dead code. It WAS dead
    # code: a mutation replacing the hand-written empty branch survived the
    # whole suite, which is how the branch was found to be unreachable.
    return names, jnp.stack(stacked)


def marginal_log_likelihood(
    graph: Graph,
    values: Mapping[str, Any] | None = None,
    *,
    budget: int = ENUMERATION_BUDGET,
) -> jax.Array:
    """``log sum_z exp log_joint``, summing every declared discrete latent out.

    Args:
        graph: the graph. Its discrete latents are summed over; every other
            latent must have a value in ``values``, exactly as ``log_joint``
            requires, and is conditioned on rather than marginalised.
        values: values for the latents that are NOT being marginalised.
        budget: most assignments to visit -- see :data:`ENUMERATION_BUDGET`.

    Raises:
        ValueError: if the assignment count exceeds ``budget``. Raised before
            any density is evaluated, so the cost is a refusal rather than a
            wait.
        GraphError: from ``log_joint``, if a latent that is not being
            marginalised has no value. Deliberately not re-raised here: the
            graph's own message already names the node and lists the latents,
            and a second wording would be a second thing to keep true.
    """
    _, terms = _terms(graph, values, budget)
    return logsumexp(terms)


def posterior_marginals(
    graph: Graph,
    values: Mapping[str, Any] | None = None,
    *,
    budget: int = ENUMERATION_BUDGET,
) -> dict[str, jax.Array]:
    """Per-site posterior probabilities over the states, exactly.

    ``{site: probabilities}`` where each array has the site's plate shape with
    the ``n`` states as its LAST axis, so ``result["z"][..., k]`` is
    ``P(z = k | data)`` and the axis a caller reduces over is the one numpy
    conventions put last.

    Computed from the same normalised terms as
    :func:`marginal_log_likelihood`, so the two cannot disagree about the
    normaliser -- the shape of defect this package records as B1, where two
    halves of one Gaussian disagreed about which covariance they described.
    """
    names, terms = _terms(graph, values, budget)
    weights = jnp.exp(terms - logsumexp(terms))
    layout = [
        (
            name,
            tuple(graph.plate_size(p) for p in graph.node(name).plate),
            graph.node(name).support.n,
        )
        for name in names
    ]
    out: dict[str, jax.Array] = {}
    for name, shape, n in layout:
        coordinates = math.prod(shape)
        totals = jnp.zeros((coordinates, n))
        for index, assignment in enumerate(_assignments(graph, names)):
            flat = jnp.reshape(assignment[name], (coordinates,))
            totals = totals.at[jnp.arange(coordinates), flat].add(weights[index])
        out[name] = jnp.reshape(totals, (*shape, n)) if shape else totals[0]
    return out
