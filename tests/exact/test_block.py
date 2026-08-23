"""Exporting A, A^T and the offset from a graph -- and what the block refuses."""

import jax
import jax.numpy as jnp
import pytest

from bayesmith import evaluate
from bayesmith.errors import GraphError, NotGaussian
from bayesmith.exact.block import (
    _env_before,
    domain_centre,
    domain_zero,
    largest_variance,
    unchecked_operator,
    variance_parts,
)
from tests.exact.models import (
    plated_latent,
    shared_ancestor,
    straight_line,
    two_linear_latents,
    two_observations,
)


def test_offset_is_the_prediction_with_the_block_at_zero():
    """`b` sits outside the block, so it is what the offset carries.

    Held at 4.0, which is none of the model's own numbers (slope 1.5,
    intercept -3.0, sigma 0.4) -- a block that returned the wrong quantity
    could not land on 4.0 by coincidence.
    """
    graph = two_linear_latents()
    block = unchecked_operator(graph, ("a",), at={"b": jnp.asarray(4.0)})
    assert jnp.allclose(block.offset["d"], 4.0)


def test_forward_is_the_linear_action_of_the_block():
    graph = two_linear_latents(n=12)
    block = unchecked_operator(graph, ("a",), at={"b": jnp.asarray(4.0)})
    got = block.forward({"a": jnp.asarray(1.0)})["d"]
    assert jnp.allclose(got, jnp.linspace(-2.0, 2.0, 12))


def test_adjoint_is_the_transpose_under_the_real_inner_product():
    """`sum(x * adjoint(y)) == sum(forward(x) * y)`, across both observed nodes.

    An adjoint that dropped one observed node, or scaled one, breaks this
    identity; nothing else in the block would notice.
    """
    graph = two_observations()
    block = unchecked_operator(graph, ("w",), at={})
    x = {"w": jnp.asarray(0.37)}
    y = {
        "d1": jax.random.normal(jax.random.key(11), block.offset["d1"].shape),
        "d2": jax.random.normal(jax.random.key(12), block.offset["d2"].shape),
    }
    pulled = block.adjoint(y)
    pushed = block.forward(x)
    lhs = sum(float(jnp.sum(x[n] * pulled[n])) for n in block.names)
    rhs = sum(float(jnp.sum(pushed[o] * y[o])) for o in y)
    assert lhs == pytest.approx(rhs, rel=1e-5)


def test_the_block_spans_every_observed_node():
    graph = two_observations(n=7, m=5)
    block = unchecked_operator(graph, ("w",), at={})
    assert set(block.offset) == set(block.data) == {"d1", "d2"}
    assert block.offset["d1"].shape == (7,)
    assert block.offset["d2"].shape == (5,)


def test_the_prior_is_read_off_the_graph():
    """No prior_std keyword exists, so the graph cannot be contradicted."""
    graph = two_linear_latents()
    block = unchecked_operator(graph, ("a", "b"), at={})
    assert jnp.allclose(block.prior_std["a"], 5.0)
    assert jnp.allclose(block.prior_std["b"], 7.0)
    assert jnp.allclose(block.prior_mean["a"], 0.0)
    assert jnp.allclose(block.prior_mean["b"], 0.0)


def test_env_before_agrees_with_evaluate_on_every_node():
    """Pins the six lines `_env_before` duplicates from `evaluate`.

    `_env_before` cannot call `evaluate` (it runs before the block has any
    value to give it), so it repeats the isinstance ladder. That is exactly
    the kind of duplication P1 recorded as the start of a silent drift, so
    the two are compared node by node here.
    """
    graph = two_linear_latents()
    at = {"b": jnp.asarray(4.0)}
    env, domain = _env_before(graph, ("a",), at)
    full = evaluate(graph, {**at, "a": domain["a"][2]})
    assert set(env) == set(full) == set(graph.names)
    for name in graph.names:
        assert jnp.allclose(env[name], full[name]), name


def test_a_block_holding_a_latent_and_its_own_ancestor_is_refused():
    """`x`'s width IS `tau`, so the pair is not jointly Gaussian.

    Both nodes are individually Normal, so a classifier that checked only
    "is each node Gaussian" would put them in one block and solve a
    posterior nobody declared.
    """
    graph = shared_ancestor()
    with pytest.raises(NotGaussian, match="ancestor"):
        unchecked_operator(graph, ("tau", "x"), at={})


def test_each_of_the_two_is_a_legitimate_block_on_its_own():
    """The refusal above is about the PAIR, not about either member."""
    graph = shared_ancestor()
    block = unchecked_operator(graph, ("x",), at={"tau": jnp.asarray(2.0)})
    assert jnp.allclose(block.prior_std["x"], 2.0)


def test_a_plated_latent_block_carries_the_plate_shaped_domain():
    graph = plated_latent(n=6)
    block = unchecked_operator(graph, ("z",), at={})
    assert block.shape["z"] == (6,)
    assert domain_zero(block)["z"].shape == (6,)
    assert block.prior_std["z"].shape == (6,)


def test_naming_something_that_is_not_a_latent_is_refused():
    graph = straight_line()
    with pytest.raises(GraphError, match="mu"):
        unchecked_operator(graph, ("mu",), at={})
    with pytest.raises(GraphError, match="twice|repeat"):
        unchecked_operator(graph, ("w", "w"), at={})
    with pytest.raises(GraphError, match="at least one"):
        unchecked_operator(graph, (), at={})


def test_domain_centre_is_the_declared_prior_mean():
    graph = straight_line(prior_mean=1.75, prior_std=2.0)
    block = unchecked_operator(graph, ("w",), at={})
    assert jnp.allclose(domain_centre(block)["w"], 1.75)


def test_variance_parts_places_each_prior_on_its_own_leaf():
    graph = two_linear_latents()
    block = unchecked_operator(graph, ("a", "b"), at={})
    parts = variance_parts(block)
    assert jnp.allclose(parts["a"], 25.0)
    assert jnp.allclose(parts["b"], 49.0)


def test_largest_variance_takes_the_loosest_prior_not_the_tightest():
    """1/largest floors lambda_min of the normal operator.

    Taking the tightest instead would floor the estimate ABOVE the true
    lambda_min and report a condition number smaller than the real one --
    an over-confident guard, which is the direction that costs something.
    """
    graph = two_linear_latents()
    block = unchecked_operator(graph, ("a", "b"), at={})
    assert float(largest_variance(variance_parts(block))) == pytest.approx(49.0)


def test_ancestry_is_transitive_not_just_direct_parents():
    """`tau` reaches `x` through `width`, so a direct-parent check misses it."""
    from tests.exact.models import indirect_ancestor

    graph = indirect_ancestor()
    assert graph.node("x").parents == ("width",)  # tau is NOT a direct parent
    with pytest.raises(NotGaussian, match="ancestor"):
        unchecked_operator(graph, ("tau", "x"), at={})
