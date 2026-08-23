import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from bayesmith.errors import GraphError
from bayesmith.graph.graph import Graph, Plate
from bayesmith.graph.nodes import Const, Deterministic, Probabilistic


def _x():
    return Probabilistic(
        name="x", parents=(), plate=(), dist_fn=lambda: dist.Normal(0.0, 1.0),
        observed=None,
    )


def _mu():
    return Deterministic(
        name="mu", parents=("x",), plate=(), fn=lambda x: 2.0 * x, linear_in=("x",)
    )


def _d():
    return Probabilistic(
        name="d", parents=("mu",), plate=(),
        dist_fn=lambda m: dist.Normal(m, 0.1), observed=jnp.array([1.0]),
    )


def test_graph_exposes_names_in_declaration_order():
    g = Graph(nodes=(_x(), _mu(), _d()), plates=())
    assert g.names == ("x", "mu", "d")


def test_node_lookup_by_name():
    g = Graph(nodes=(_x(), _mu(), _d()), plates=())
    assert g.node("mu").parents == ("x",)


def test_unknown_node_name_is_refused_by_name():
    g = Graph(nodes=(_x(),), plates=())
    with pytest.raises(GraphError, match="no node named 'nope'"):
        g.node("nope")


def test_latents_and_observed_are_derived_not_stored():
    g = Graph(nodes=(_x(), _mu(), _d()), plates=())
    assert g.latents == ("x",)
    assert g.observed == ("d",)


def test_a_parent_declared_after_its_child_is_refused():
    with pytest.raises(GraphError, match="names parent 'x', which is not declared"):
        Graph(nodes=(_mu(), _x()), plates=())


def test_a_duplicate_node_name_is_refused():
    with pytest.raises(GraphError, match="duplicate node name 'x'"):
        Graph(nodes=(_x(), _x()), plates=())


def test_a_node_in_an_undeclared_plate_is_refused():
    n = Const(name="X", parents=(), plate=("obs",), value=jnp.arange(3.0))
    with pytest.raises(GraphError, match="plate 'obs', which the graph does not"):
        Graph(nodes=(n,), plates=())


def test_a_declared_plate_is_accepted():
    n = Const(name="X", parents=(), plate=("obs",), value=jnp.arange(3.0))
    g = Graph(nodes=(n,), plates=(Plate(name="obs", size=3),))
    assert g.plate_size("obs") == 3


def test_nested_plates_are_refused_with_a_reason():
    n = Const(name="X", parents=(), plate=("a", "b"), value=jnp.zeros((2, 3)))
    plates = (Plate(name="a", size=2), Plate(name="b", size=3))
    with pytest.raises(GraphError, match="nested plates are not supported yet"):
        Graph(nodes=(n,), plates=plates)


def test_unknown_plate_name_is_refused_by_name():
    g = Graph(nodes=(), plates=(Plate(name="obs", size=3),))
    with pytest.raises(GraphError, match="no plate named 'nope'"):
        g.plate_size("nope")


def test_a_duplicate_plate_name_is_refused():
    n = Const(name="X", parents=(), plate=("obs",), value=jnp.arange(3.0))
    plates = (Plate(name="obs", size=3), Plate(name="obs", size=5))
    with pytest.raises(GraphError, match="duplicate plate name 'obs'"):
        Graph(nodes=(n,), plates=plates)
