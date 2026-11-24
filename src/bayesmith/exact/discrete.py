"""Exact marginalisation of declared discrete latents, by enumeration.

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
import numpy as np
from jax.scipy.special import logsumexp
from numpyro.distributions import constraints

from bayesmith.errors import StructureError
from bayesmith.graph.evaluate import (
    _apply_deterministic,
    _law_builder,
    distribution_batch_shape,
    log_joint,
)
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Const, Deterministic, Discrete, Probabilistic

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


def _check_state_labels(node: Probabilistic, distribution: Any) -> None:
    """Assignment indices must coincide with the distribution's actual scalar labels."""
    support = distribution.support
    if isinstance(support, constraints.independent):
        support = support.base_constraint
    n = node.support.n
    if support is constraints.boolean and n == 2:
        return
    if isinstance(support, constraints.integer_interval):
        bounds = ((support.lower_bound, 0), (support.upper_bound, n - 1))
        if all(
            not isinstance(bound, jax.core.Tracer) and np.all(np.asarray(bound) == expected)
            for bound, expected in bounds
        ):
            return
    raise StructureError(
        f"discrete site {node.name!r} requires a Cartesian product of scalar states "
        f"0 through {n - 1}; could not establish this for support {support}."
    )


def _site_shapes(graph: Graph, values: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """Trace the shapes of discrete sites and their ancestors without drawing or scoring."""
    names = discrete_latents(graph)

    def prototypes():
        class Ancestors(dict):
            def __missing__(self, name):
                self[name] = value(name)
                return self[name]

        env = Ancestors()
        law_for = _law_builder(graph, env)

        def value(name):
            node = graph.node(name)
            if isinstance(node, Const):
                return node.value
            if isinstance(node, Probabilistic):
                if not node.is_latent:
                    return node.observed
                if name not in names and name in values:
                    return values[name]
            if isinstance(node, Deterministic):
                return _apply_deterministic(graph, node, env, law_for)
            # Keep fixed integer bounds concrete under shape tracing. Bounds
            # that actually depend on an unknown parent still cannot certify support.
            with jax.ensure_compile_time_eval():
                distribution = law_for(name)
            batch = distribution_batch_shape(graph, node, distribution)
            shape = batch + distribution.event_shape
            if name not in names:
                # enumeration_states can be asked for a cost before any
                # conditioning values exist. Trace the parent's native dtype.
                try:
                    prototype = distribution.sample(jax.random.key(0))
                except NotImplementedError:
                    # Improper priors have a support and shape but no sampler.
                    prototype = distribution.support.feasible_like(jnp.zeros(shape))
                return jnp.broadcast_to(prototype, shape)
            _check_state_labels(node, distribution)
            return jnp.zeros(shape, dtype=jnp.int32)

        return tuple(env[name] for name in names)

    try:
        prototypes_by_site = jax.eval_shape(prototypes)
    except jax.errors.ConcretizationTypeError as exc:
        raise StructureError(
            f"enumerating {list(names)} requires a fixed coordinate layout; "
            "the graph requested a concrete latent value during shape inference."
        ) from exc
    return {name: item.shape for name, item in zip(names, prototypes_by_site, strict=True)}


def _state_count(graph: Graph, shapes: Mapping[str, tuple[int, ...]]) -> int:
    return math.prod(
        graph.node(name).support.n ** math.prod(shape) for name, shape in shapes.items()
    )


def enumeration_states(graph: Graph, values: Mapping[str, Any] | None = None) -> int:
    """How many joint assignments enumeration would visit before evaluating any density.

    Each scalar coordinate in a site's batch, plate and independent event
    dimensions contributes a factor of its declared cardinality. The empty
    product is one. Each coordinate must have the fixed integer support
    ``0, ..., n-1``. Supply conditioned ancestors in ``values`` if their
    shapes differ from the model's own sample shapes.
    """
    return _state_count(graph, _site_shapes(graph, values or {}))


def _assignments(graph: Graph, shapes: Mapping[str, tuple[int, ...]]):
    """Every joint assignment, as ``{site: value}`` with the site's own shape."""
    ranges = [
        itertools.product(range(graph.node(name).support.n), repeat=math.prod(shape))
        for name, shape in shapes.items()
    ]
    for combination in itertools.product(*ranges):
        yield {
            name: jnp.asarray(flat, dtype=jnp.int32).reshape(shape)
            for (name, shape), flat in zip(shapes.items(), combination, strict=True)
        }


def _terms(
    graph: Graph,
    values: Mapping[str, Any] | None,
    budget: int,
) -> tuple[dict[str, tuple[int, ...]], jax.Array]:
    """``log_joint`` at every assignment, and the sites they range over."""
    supplied = dict(values or {})
    shapes = _site_shapes(graph, supplied)
    states = _state_count(graph, shapes)
    if states > budget:
        # Very wide sites can exceed Python's limit on decimal integer conversion.
        count = str(states) if states.bit_length() <= 64 else f"at least 2**{states.bit_length() - 1}"
        raise ValueError(
            f"enumerating {list(shapes)} would visit {count} joint assignments, "
            f"past the budget of {budget}. Exact enumeration costs "
            f"n ** coordinates per site; pass `budget=` to allow it "
            f"deliberately, or marginalise a smaller set."
        )
    stacked = [
        log_joint(graph, {**supplied, **assignment})
        for assignment in _assignments(graph, shapes)
    ]
    # No `if stacked` fallback, and that absence is load-bearing rather than
    # an oversight. `itertools.product()` over no iterables yields exactly ONE
    # empty tuple, so a graph with no discrete latents produces one assignment
    # -- the empty one -- whose term is the joint itself. That is the correct
    # mathematics (the empty product is one, not zero) and the library already
    # implements it, so a special case here would be dead code. It WAS dead
    # code: a mutation replacing the hand-written empty branch survived the
    # whole suite, which is how the branch was found to be unreachable.
    return shapes, jnp.stack(stacked)


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

    ``{site: probabilities}`` where each array has the site's complete value shape with
    the ``n`` states as its LAST axis, so ``result["z"][..., k]`` is
    ``P(z = k | data)`` and the axis a caller reduces over is the one numpy
    conventions put last.

    Computed from the same normalised terms as
    :func:`marginal_log_likelihood`, so the two cannot disagree about the
    normaliser -- the shape of defect this package records as B1, where two
    halves of one Gaussian disagreed about which covariance they described.
    """
    shapes, terms = _terms(graph, values, budget)
    weights = jnp.exp(terms - logsumexp(terms))
    out: dict[str, jax.Array] = {}
    for name, shape in shapes.items():
        n = graph.node(name).support.n
        coordinates = math.prod(shape)
        totals = jnp.zeros((coordinates, n))
        for index, assignment in enumerate(_assignments(graph, shapes)):
            flat = jnp.reshape(assignment[name], (coordinates,))
            totals = totals.at[jnp.arange(coordinates), flat].add(weights[index])
        out[name] = jnp.reshape(totals, (*shape, n))
    return out
