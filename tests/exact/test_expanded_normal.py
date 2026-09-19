"""A Normal written with ``.expand`` is the diagonal Normal it expands to.

``dist.Normal(mu, s).expand((k,))`` is numpyro's ``ExpandedDistribution``:
its ``log_prob`` is the base Normal's ``log_prob`` broadcast to the expanded
batch shape, and its draws are independent copies. So its per-element
density is the base's, exactly as ``Independent`` (``.to_event``) changes
only how that density is summed. Until 2026-09-19 the Gaussian reader
stripped ``Independent`` and not ``ExpandedDistribution``, so the
hierarchical notebook's group prior -- the same law as
``Normal(mu * jnp.ones(8), s).to_event(1)`` -- lost its exact block and the
whole model went to NUTS.

The reading is still verified against the node's own ``log_prob``: the
wrapper is stripped to the base distribution, never replaced by a fresh
Normal, so a base that lies about its density is still caught.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import compile as compile_graph
from bayesmith import const, det, evaluate, observe, sample, trace
from bayesmith.errors import NotGaussian, StructureError
from bayesmith.exact.gaussian import check_gaussian, gaussian_parts, node_shape
from tests.exact.models import LyingNormal

GROUPS = 8
INDICES = np.repeat(np.arange(GROUPS), 5)
DATA = jnp.asarray(np.random.default_rng(0).normal(size=INDICES.size))


def _hierarchy(group_prior, population_prior=lambda: dist.Uniform(-3.0, 3.0)):
    def model(indices, data):
        index = const("group_index", indices)
        population = sample("population", population_prior)
        groups = sample("groups", group_prior, population)
        location = det("location", lambda u, i: u[i], groups, index)
        observe("obs", lambda loc: dist.Normal(loc, 0.25), location, obs=data)

    return trace(model, INDICES, DATA)


def _expanded(mu):
    return dist.Normal(mu, 0.6).expand((GROUPS,)).to_event(1)


def _broadcast(mu):
    return dist.Normal(mu * jnp.ones(GROUPS), 0.6).to_event(1)


def _env(graph):
    return evaluate(
        graph, {"population": jnp.asarray(0.4), "groups": jnp.zeros(GROUPS)}
    )


def test_an_expanded_normal_is_read_as_the_diagonal_normal_it_expands_to():
    with jax.enable_x64(True):
        expanded, broadcast = _hierarchy(_expanded), _hierarchy(_broadcast)
        shape = node_shape(expanded, expanded.node("groups"), _env(expanded))
        read = [
            [
                jnp.broadcast_to(part, shape)
                for part in gaussian_parts(g, g.node("groups"), _env(g))
            ]
            for g in (expanded, broadcast)
        ]
    assert shape == (GROUPS,)
    np.testing.assert_array_equal(read[0][0], read[1][0])
    np.testing.assert_array_equal(read[0][1], read[1][1])


def test_the_reading_is_verified_against_the_expanded_node_s_own_density():
    with jax.enable_x64(True):
        graph = _hierarchy(_expanded)
        errors = check_gaussian(graph, graph.node("groups"), _env(graph))
    assert errors and max(errors.values()) < 1e-12


def test_an_expanded_normal_that_lies_about_its_log_prob_is_still_refused():
    """Stripping the wrapper keeps the base, so the base's own density is probed."""
    with jax.enable_x64(True):
        graph = _hierarchy(
            lambda mu: LyingNormal(mu, 0.6).expand((GROUPS,)).to_event(1)
        )
        with pytest.raises(StructureError, match="log_prob"):
            check_gaussian(graph, graph.node("groups"), _env(graph))


def test_an_expanded_distribution_that_is_not_normal_is_still_not_gaussian():
    with jax.enable_x64(True):
        graph = _hierarchy(
            lambda mu: dist.Gamma(2.0 + mu**2, 1.0).expand((GROUPS,)).to_event(1)
        )
        with pytest.raises(NotGaussian, match="Gamma"):
            gaussian_parts(graph, graph.node("groups"), _env(graph))


@pytest.mark.parametrize(
    "population_prior",
    [lambda: dist.Uniform(-3.0, 3.0), lambda: dist.Normal(0.0, 2.0)],
    ids=["uniform-population", "normal-population"],
)
def test_both_spellings_of_the_group_prior_compile_to_the_same_plan(population_prior):
    with jax.enable_x64(True):
        plans = [
            compile_graph(_hierarchy(prior, population_prior))
            for prior in (_expanded, _broadcast)
        ]
    routes = [[(b.latents, b.method) for b in plan.blocks] for plan in plans]
    assert routes[0] == [(("groups",), "gcr"), (("population",), "nuts")]
    assert routes[0] == routes[1]
    assert plans[0].exact.kappa == pytest.approx(plans[1].exact.kappa, rel=1e-12)
