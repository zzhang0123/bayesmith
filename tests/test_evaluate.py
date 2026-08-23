import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from bayesmith.errors import GraphError
from bayesmith.graph.evaluate import evaluate
from bayesmith.graph.trace import const, det, observe, sample, trace


class Scale(eqx.Module):
    w: jax.Array

    def __call__(self, x):
        return self.w * x


def _linear_model():
    def model():
        X = const("X", jnp.array([1.0, 2.0, 3.0]))
        a = sample("a", lambda: dist.Normal(0.0, 1.0))
        mu = det("mu", lambda a_, X_: a_ * X_, a, X, linear_in=("a",))
        observe("d", lambda m: dist.Normal(m, 0.1), mu, obs=jnp.zeros(3))

    return trace(model)


def test_evaluate_computes_every_node():
    env = evaluate(_linear_model(), {"a": jnp.array(2.0)})
    assert set(env) == {"X", "a", "mu", "d"}
    assert jnp.allclose(env["mu"], jnp.array([2.0, 4.0, 6.0]))


def test_observed_nodes_take_their_data_not_a_supplied_value():
    env = evaluate(_linear_model(), {"a": jnp.array(2.0)})
    assert jnp.array_equal(env["d"], jnp.zeros(3))


def test_a_latent_without_a_value_is_refused_by_name():
    with pytest.raises(GraphError, match="latent node 'a' has no value"):
        evaluate(_linear_model(), {})


def test_a_value_for_an_unknown_name_is_refused():
    with pytest.raises(GraphError, match="values names 'nope'"):
        evaluate(_linear_model(), {"a": jnp.array(1.0), "nope": jnp.array(0.0)})


def test_a_value_for_an_observed_node_explains_it_is_observed():
    """The refusal names the actual reason, not a generic "not latent".

    An observed node is the most likely real mistake -- someone who has
    not internalised the latent/observed split and tries to pass all their
    data through ``values`` -- so it gets its own explanation rather than
    sharing text with the deterministic/constant/unknown-name cases.
    """
    with pytest.raises(
        GraphError, match="values names 'd', which is an observed node"
    ):
        evaluate(_linear_model(), {"a": jnp.array(1.0), "d": jnp.zeros(3)})


def test_evaluate_is_differentiable_through_a_module_operator():
    """A parameterised operator inside a node stays differentiable.

    X and w are chosen so the true gradient (``sum(X)``) cannot coincide
    with w's own value, and the expectation is computed from X rather than
    written as a bare literal that could drift back into coincidence. Both
    guard against the same failure mode: equinox does not refuse a JAX
    array in a static field, it only warns -- so if ``fn`` were ever made
    static, the whole ``Scale`` module would be absorbed into pytree aux
    data and ``eqx.filter_grad`` would silently return each leaf's
    *original* value in place of a gradient. Nothing raises. If w happened
    to equal ``sum(X)``, that wrong answer would be indistinguishable from
    the right one.
    """
    X = jnp.array([1.0, 5.0])
    w = jnp.array(2.0)

    def model():
        Xc = const("X", X)
        det("mu", Scale(w=w), Xc)

    graph = trace(model)

    def total(g):
        return jnp.sum(evaluate(g, {})["mu"])

    grad = eqx.filter_grad(total)(graph)
    assert jnp.allclose(grad.nodes[1].fn.w, jnp.sum(X))


def test_evaluate_is_jittable():
    graph = _linear_model()
    jitted = eqx.filter_jit(lambda g, a: evaluate(g, {"a": a})["mu"])
    assert jnp.allclose(jitted(graph, jnp.array(2.0)), jnp.array([2.0, 4.0, 6.0]))
