"""Graphs at the corners of what a graph can be.

Every one of these is a shape a user will eventually build by accident, and
each exercises a branch that a comfortable three-node example never reaches.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro.infer.util import log_density

from bayesmith.bridge.numpyro_bridge import to_numpyro
from bayesmith.errors import GraphError
from bayesmith.graph.evaluate import evaluate, log_joint
from bayesmith.graph.trace import const, det, observe, sample, trace


def test_a_single_node_graph_evaluates_and_has_zero_density():
    def model():
        const("X", jnp.array(1.0))

    graph = trace(model)
    assert graph.names == ("X",)
    assert graph.latents == ()
    assert evaluate(graph, {})["X"] == jnp.array(1.0)
    assert log_joint(graph, {}) == jnp.zeros(())


def test_a_graph_with_no_latents_still_has_a_density():
    """A fully observed model: the likelihood at fixed parameters."""

    def model():
        observe("d", lambda: dist.Normal(0.0, 1.0), obs=jnp.array([1.0, 2.0]))

    graph = trace(model)
    assert graph.latents == ()
    expected = jnp.sum(dist.Normal(0.0, 1.0).log_prob(jnp.array([1.0, 2.0])))
    assert jnp.allclose(log_joint(graph, {}), expected)


def test_a_graph_with_no_observations_is_the_prior():
    def model():
        sample("x", lambda: dist.Normal(0.0, 1.0))

    graph = trace(model)
    assert graph.observed == ()
    assert jnp.allclose(
        log_joint(graph, {"x": jnp.array(0.5)}),
        dist.Normal(0.0, 1.0).log_prob(0.5),
    )


def test_an_empty_graph_is_allowed_and_has_zero_density():
    """Nothing declared is a valid, if useless, graph -- not a crash."""

    def model():
        return None

    graph = trace(model)
    assert graph.names == ()
    assert log_joint(graph, {}) == jnp.zeros(())


def test_two_perfectly_collinear_parents_are_not_rejected_here():
    """Collinearity is an identifiability question, not a graph-shape one.

    P1 must build this graph without complaint; refusing it is P5's job, and
    doing it here would refuse legitimate over-parameterised models that a
    prior makes perfectly well posed.
    """

    def model():
        a = sample("a", lambda: dist.Normal(0.0, 1.0))
        b = sample("b", lambda: dist.Normal(0.0, 1.0))
        mu = det("mu", lambda x, y: x + y, a, b, linear_in=("a", "b"))
        observe("d", lambda m: dist.Normal(m, 1.0), mu, obs=jnp.array([1.0]))

    graph = trace(model)
    at = {"a": jnp.array(0.3), "b": jnp.array(0.4)}
    ours = log_joint(graph, at)
    theirs, _ = log_density(to_numpyro(graph), (), {}, at)
    assert jnp.allclose(ours, theirs, rtol=1e-6)


def test_a_deep_chain_evaluates_in_one_pass():
    """Topological order is declaration order, however long the chain."""

    def model():
        node = sample("x0", lambda: dist.Normal(0.0, 1.0))
        for i in range(1, 50):
            node = det(f"x{i}", lambda v: v + 1.0, node)

    graph = trace(model)
    env = evaluate(graph, {"x0": jnp.array(0.0)})
    assert env["x49"] == jnp.array(49.0)


def test_a_diamond_reaches_the_shared_parent_once():
    """A DAG, not a tree: two paths converge on one node."""

    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        left = det("left", lambda v: 2.0 * v, x)
        right = det("right", lambda v: 3.0 * v, x)
        det("join", lambda a, b: a + b, left, right)

    env = evaluate(trace(model), {"x": jnp.array(1.0)})
    assert env["join"] == jnp.array(5.0)


def test_a_cycle_cannot_be_expressed_by_tracing():
    """Tracing makes cycles unrepresentable: you cannot pass a handle you have
    not created yet. Hand-built cycles are caught by Graph.__check_init__,
    pinned in Task 2. Passing something that merely looks like a handle is
    refused by name."""

    def model():
        det("a", lambda v: v, object())

    with pytest.raises(GraphError, match="parents must be NodeRef"):
        trace(model)
