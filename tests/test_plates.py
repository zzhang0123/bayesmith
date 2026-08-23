import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from bayesmith.errors import GraphError
from bayesmith.graph.evaluate import evaluate, log_joint
from bayesmith.graph.trace import const, det, observe, plate, sample, trace


def test_a_plated_deterministic_node_is_vmapped_over_its_plated_parent():
    def model():
        obs = plate("obs", 3)
        X = const("X", jnp.array([1.0, 2.0, 3.0]), plate=obs)
        det("mu", lambda x: x**2, X, plate=obs)

    env = evaluate(trace(model), {})
    assert jnp.allclose(env["mu"], jnp.array([1.0, 4.0, 9.0]))


def test_an_unplated_parent_is_broadcast_not_mapped():
    def model():
        obs = plate("obs", 3)
        X = const("X", jnp.array([1.0, 2.0, 3.0]), plate=obs)
        a = const("a", jnp.array(10.0))
        det("mu", lambda x, a_: a_ * x, X, a, plate=obs)

    env = evaluate(trace(model), {})
    assert jnp.allclose(env["mu"], jnp.array([10.0, 20.0, 30.0]))


def test_vmap_agrees_with_an_explicit_python_loop():
    """The plate is an optimisation, so it must change nothing numerically."""
    xs = jnp.array([0.5, 1.5, 2.5, 3.5])

    def model():
        obs = plate("obs", 4)
        X = const("X", xs, plate=obs)
        det("mu", lambda x: jnp.sin(x) * 3.0, X, plate=obs)

    got = evaluate(trace(model), {})["mu"]
    expected = jnp.stack([jnp.sin(x) * 3.0 for x in xs])
    assert jnp.array_equal(got, expected)


def test_a_plated_node_with_no_plated_parent_is_refused_with_a_reason():
    def model():
        obs = plate("obs", 3)
        a = const("a", jnp.array(1.0))
        det("mu", lambda v: v, a, plate=obs)

    with pytest.raises(GraphError, match="nothing to map over"):
        evaluate(trace(model), {})


def test_a_plated_likelihood_sums_over_the_plate():
    def model():
        obs = plate("obs", 3)
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe("d", lambda v: dist.Normal(v, 1.0), x, obs=jnp.array([1.0, 2.0, 3.0]), plate=obs)

    got = log_joint(trace(model), {"x": jnp.array(0.5)})
    expected = dist.Normal(0.0, 1.0).log_prob(0.5) + jnp.sum(
        dist.Normal(0.5, 1.0).log_prob(jnp.array([1.0, 2.0, 3.0]))
    )
    assert jnp.allclose(got, expected, rtol=1e-6)


@pytest.mark.parametrize("size", [1, 1000])
def test_extreme_plate_sizes(size):
    """Failure modes are U-shaped; test both ends, not the comfortable middle."""

    def model():
        obs = plate("obs", size)
        X = const("X", jnp.arange(size, dtype=jnp.float32), plate=obs)
        det("mu", lambda x: x + 1.0, X, plate=obs)

    env = evaluate(trace(model), {})
    assert env["mu"].shape == (size,)
    assert jnp.allclose(env["mu"], jnp.arange(size, dtype=jnp.float32) + 1.0)
