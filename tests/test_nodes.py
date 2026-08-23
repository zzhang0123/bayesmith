import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from bayesmith.graph.nodes import Const, Deterministic, Node, Probabilistic


class Scale(eqx.Module):
    """Stand-in for a rheplicant Pipeline: an operator carrying parameters."""

    w: jax.Array

    def __call__(self, x):
        return self.w * x


def test_node_identity_fields_are_static():
    n = Deterministic(
        name="a", parents=("x",), plate=(), fn=lambda x: x, linear_in=("x",)
    )
    assert n.name == "a"
    assert n.parents == ("x",)
    assert n.linear_in == ("x",)
    # name/parents/plate/linear_in are metadata, so they must NOT be leaves
    assert jax.tree.leaves(n) == [n.fn]


def test_a_lambda_fn_is_a_non_array_leaf():
    """filter_jit routes non-array leaves to the static side; that is the point."""
    n = Deterministic(name="a", parents=("x",), plate=(), fn=lambda x: 2.0 * x)
    (leaf,) = jax.tree.leaves(n)
    assert callable(leaf)
    assert not eqx.is_array(leaf)


def test_a_module_fn_exposes_its_parameters_as_traceable_leaves():
    """The rheplicant-compatibility property: gradients must reach into fn."""
    n = Deterministic(name="a", parents=("x",), plate=(), fn=Scale(w=jnp.array(3.0)))
    leaves = jax.tree.leaves(n)
    assert len(leaves) == 1
    assert eqx.is_inexact_array(leaves[0])

    grad = eqx.filter_grad(lambda node, x: jnp.sum(node.fn(x)))(n, jnp.array(5.0))
    assert grad.fn.w == jnp.array(5.0)


def test_const_holds_its_value_as_an_array_leaf():
    n = Const(name="X", parents=(), plate=(), value=jnp.arange(3.0))
    (leaf,) = jax.tree.leaves(n)
    assert jnp.array_equal(leaf, jnp.arange(3.0))


def test_probabilistic_is_latent_when_unobserved_and_observed_otherwise():
    latent = Probabilistic(
        name="x", parents=(), plate=(), dist_fn=lambda: dist.Normal(0.0, 1.0),
        observed=None,
    )
    seen = Probabilistic(
        name="d", parents=("x",), plate=(),
        dist_fn=lambda m: dist.Normal(m, 1.0),
        observed=jnp.array([1.0, 2.0]),
    )
    assert latent.is_latent
    assert not seen.is_latent


def test_every_node_type_is_a_node():
    for cls in (Const, Deterministic, Probabilistic):
        assert issubclass(cls, Node)
