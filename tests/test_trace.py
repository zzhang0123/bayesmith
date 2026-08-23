import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from bayesmith.errors import GraphError, TraceError
from bayesmith.graph.graph import Graph, Plate
from bayesmith.graph.nodes import Const, Deterministic, Probabilistic
from bayesmith.graph.trace import const, det, observe, plate, sample, trace


def test_trace_records_nodes_in_declaration_order():
    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        mu = det("mu", lambda v: 2.0 * v, x, linear_in=("x",))
        observe("d", lambda m: dist.Normal(m, 0.1), mu, obs=jnp.array([1.0]))

    g = trace(model)
    assert g.names == ("x", "mu", "d")
    assert isinstance(g.node("x"), Probabilistic)
    assert isinstance(g.node("mu"), Deterministic)


def test_passing_a_noderef_declares_the_edge():
    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        y = sample("y", lambda: dist.Normal(0.0, 1.0))
        det("mu", lambda a, b: a + b, x, y)

    g = trace(model)
    assert g.node("mu").parents == ("x", "y")


def test_const_becomes_a_node_carrying_its_value():
    def model():
        const("X", jnp.arange(3.0))

    g = trace(model)
    node = g.node("X")
    assert isinstance(node, Const)
    assert jnp.array_equal(node.value, jnp.arange(3.0))


def test_linear_in_is_recorded_as_declared():
    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        det("mu", lambda v: 2.0 * v, x, linear_in=("x",))

    assert trace(model).node("mu").linear_in == ("x",)


def test_observed_data_is_attached_to_the_node():
    def model():
        observe("d", lambda: dist.Normal(0.0, 1.0), obs=jnp.array([1.0, 2.0]))

    assert jnp.array_equal(trace(model).node("d").observed, jnp.array([1.0, 2.0]))


def test_plate_is_declared_and_attached():
    def model():
        obs = plate("obs", 4)
        const("X", jnp.arange(4.0), plate=obs)

    g = trace(model)
    assert g.plates == (Plate(name="obs", size=4),)
    assert g.node("X").plate == ("obs",)


def test_a_primitive_outside_trace_is_refused():
    with pytest.raises(TraceError, match="must be called inside trace"):
        sample("x", lambda: dist.Normal(0.0, 1.0))


def test_a_duplicate_name_is_refused_during_tracing():
    def model():
        sample("x", lambda: dist.Normal(0.0, 1.0))
        sample("x", lambda: dist.Normal(0.0, 1.0))

    with pytest.raises(GraphError, match="duplicate node name 'x'"):
        trace(model)


def test_trace_forwards_arguments_to_the_model():
    def model(data):
        observe("d", lambda: dist.Normal(0.0, 1.0), obs=data)

    g = trace(model, jnp.array([3.0]))
    assert jnp.array_equal(g.node("d").observed, jnp.array([3.0]))


def test_the_recorder_is_popped_even_when_the_model_raises():
    def model():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        trace(model)
    # The stack must be clean, or the next trace would inherit these nodes.
    with pytest.raises(TraceError):
        sample("x", lambda: dist.Normal(0.0, 1.0))


def test_tracing_twice_gives_isomorphic_graphs():
    """Structure must not depend on how many times the model has been traced."""

    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        det("mu", lambda v: 2.0 * v, x)

    a, b = trace(model), trace(model)
    assert a.names == b.names
    assert [n.parents for n in a.nodes] == [n.parents for n in b.nodes]
    assert [n.plate for n in a.nodes] == [n.plate for n in b.nodes]


def test_an_explicit_graph_and_a_traced_one_agree_on_structure():
    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        det("mu", lambda v: 2.0 * v, x, linear_in=("x",))

    traced = trace(model)
    built = Graph(
        nodes=(
            Probabilistic(
                name="x", parents=(), plate=(),
                dist_fn=lambda: dist.Normal(0.0, 1.0), observed=None,
            ),
            Deterministic(
                name="mu", parents=("x",), plate=(),
                fn=lambda v: 2.0 * v, linear_in=("x",),
            ),
        ),
        plates=(),
    )
    assert traced.names == built.names
    assert [n.parents for n in traced.nodes] == [n.parents for n in built.nodes]
    assert traced.node("mu").linear_in == built.node("mu").linear_in
