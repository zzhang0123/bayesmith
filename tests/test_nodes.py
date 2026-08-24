import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from bayesmith.graph.nodes import (
    Const,
    Continuous,
    Deterministic,
    Discrete,
    Node,
    Probabilistic,
)


class Scale(eqx.Module):
    """Stand-in for a rheplicant Pipeline: an operator carrying parameters."""

    w: jax.Array

    def __call__(self, x):
        return self.w * x


class ScaledNormal(eqx.Module):
    """Stand-in for a noise model carrying its own parameters."""

    scale: jax.Array

    def __call__(self, loc):
        return dist.Normal(loc, self.scale)

    def as_normal(self, loc):
        """A second, *named* entry point with the same body as ``__call__``.

        Exists so a test can exercise "a bound method" (``model.as_normal``)
        as a spelling distinct from ``__call__`` itself -- equinox treats
        the two differently: see the module docstring's second
        gradient-loss trap. ``__call__`` and other dunder names are exempt
        from equinox's bound-method-to-pytree wrapping, so testing only
        ``model.__call__`` would not exercise the "bound method" case at
        all.
        """
        return dist.Normal(loc, self.scale)


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


def test_a_module_dist_fn_exposes_its_parameters_as_traceable_leaves():
    """Same rheplicant-compatibility property, for ``Probabilistic.dist_fn``.

    Latent (``observed=None``) so the node's only leaf is dist_fn's own
    parameter -- an observed array would add a second leaf and complicate
    the leaf-count assertion below.
    """
    n = Probabilistic(
        name="d",
        parents=("x",),
        plate=(),
        dist_fn=ScaledNormal(scale=jnp.array(2.0)),
        observed=None,
    )
    leaves = jax.tree.leaves(n)
    assert len(leaves) == 1
    assert eqx.is_inexact_array(leaves[0])

    grad = eqx.filter_grad(
        lambda node, loc: node.dist_fn(loc).log_prob(jnp.array(1.0))
    )(n, jnp.array(0.5))
    assert jnp.isfinite(grad.dist_fn.scale)
    assert grad.dist_fn.scale != 0.0


def test_a_bound_method_dist_fn_also_exposes_its_parameters_as_traceable_leaves():
    """The module docstring's second gradient-loss trap, first bullet.

    ``model.as_normal`` is not the same Python object as ``model``, but
    ``equinox.Module.__getattribute__`` wraps ordinary (non-dunder) method
    access in its own ``equinox.BoundMethod`` -- an ``eqx.Module`` that
    stores ``__self__`` (the model) as a genuine, non-static subnode. So the
    parameter stays reachable and differentiable exactly as it would through
    the model itself.

    Pinned explicitly because a previous reviewer claimed bound methods lose
    the gradient (mirroring the closure case below); measured here against
    this repository's actual, installed equinox version rather than
    assumed. If a future equinox upgrade changes this, this test -- not a
    surprise in production -- is where it will show up.
    """
    model = ScaledNormal(scale=jnp.array(2.0))
    n = Probabilistic(
        name="d",
        parents=("x",),
        plate=(),
        dist_fn=model.as_normal,
        observed=None,
    )

    # dist_fn is a BoundMethod, but it is a pytree: the scale parameter is
    # still the node's one and only leaf, same as the direct-module case.
    leaves = jax.tree.leaves(n)
    assert len(leaves) == 1
    assert eqx.is_inexact_array(leaves[0])

    grad = eqx.filter_grad(
        lambda node, loc: node.dist_fn(loc).log_prob(jnp.array(1.0))
    )(n, jnp.array(0.5))
    # grad.dist_fn is an equinox.BoundMethod, not a ScaledNormal -- dig
    # through __self__ the same way the forward call would.
    assert jnp.isfinite(grad.dist_fn.__self__.scale)
    assert grad.dist_fn.__self__.scale != 0.0

    # And it is the exact value the direct-module spelling produces --
    # the two are not just "both non-null", they agree.
    direct = Probabilistic(
        name="d",
        parents=("x",),
        plate=(),
        dist_fn=model,
        observed=None,
    )
    direct_grad = eqx.filter_grad(
        lambda node, loc: node.dist_fn(loc).log_prob(jnp.array(1.0))
    )(direct, jnp.array(0.5))
    assert grad.dist_fn.__self__.scale == direct_grad.dist_fn.scale


def test_a_closure_over_a_module_dist_fn_silently_loses_its_gradient():
    """The module docstring's second gradient-loss trap, second bullet.

    A plain closure over the module (here, over ``model`` via a ``def``
    that forwards to it) is an ordinary ``types.FunctionType`` -- an
    opaque, non-array leaf as far as JAX's pytree machinery is concerned.
    ``eqx.filter_grad`` therefore excludes it from differentiation
    entirely. Nothing raises: the forward value is byte-identical to
    calling the model directly, because the closure really does run with
    the model's real (correct) value -- only the gradient silently
    vanishes. This is the same shape as the static-field trap in the
    module docstring's first paragraph: a silent wrong (here, absent)
    answer, not a constructor error.
    """
    model = ScaledNormal(scale=jnp.array(2.0))

    def dist_fn(loc):
        return model(loc)

    n = Probabilistic(
        name="d",
        parents=("x",),
        plate=(),
        dist_fn=dist_fn,
        observed=None,
    )

    # The whole closure is one opaque leaf -- not an array, so not the
    # node's parameter, unlike the module and bound-method spellings above.
    leaves = jax.tree.leaves(n)
    assert len(leaves) == 1
    assert not eqx.is_inexact_array(leaves[0])

    forward = n.dist_fn(jnp.array(0.5)).log_prob(jnp.array(1.0))
    direct_forward = model(jnp.array(0.5)).log_prob(jnp.array(1.0))
    assert jnp.allclose(forward, direct_forward)

    grad = eqx.filter_grad(
        lambda node, loc: node.dist_fn(loc).log_prob(jnp.array(1.0))
    )(n, jnp.array(0.5))
    # Not an error, not zero -- absent. eqx.filter_grad reports "excluded
    # from differentiation" as None at this pytree position.
    assert grad.dist_fn is None


def test_const_holds_its_value_as_an_array_leaf():
    n = Const(name="X", parents=(), plate=(), value=jnp.arange(3.0))
    (leaf,) = jax.tree.leaves(n)
    assert jnp.array_equal(leaf, jnp.arange(3.0))


def test_probabilistic_is_latent_when_unobserved_and_observed_otherwise():
    latent = Probabilistic(
        name="x",
        parents=(),
        plate=(),
        dist_fn=lambda: dist.Normal(0.0, 1.0),
        observed=None,
    )
    seen = Probabilistic(
        name="d",
        parents=("x",),
        plate=(),
        dist_fn=lambda m: dist.Normal(m, 1.0),
        observed=jnp.array([1.0, 2.0]),
    )
    assert latent.is_latent
    assert not seen.is_latent


def test_probabilistic_support_and_depends_on_prediction_default_safely():
    """Dispatch-axis claims, like Deterministic.linear_in -- P1 does not
    read either, but the defaults matter for when a future dispatcher does:
    undeclared must never look like a claim that unlocks a shortcut. See
    Probabilistic's docstring for the full reasoning.
    """
    n = Probabilistic(
        name="d",
        parents=(),
        plate=(),
        dist_fn=lambda: dist.Normal(0.0, 1.0),
        observed=None,
    )
    assert n.support is None
    assert n.depends_on_prediction is True


def test_probabilistic_support_and_depends_on_prediction_are_static_not_leaves():
    n = Probabilistic(
        name="d",
        parents=("x",),
        plate=(),
        dist_fn=ScaledNormal(scale=jnp.array(3.0)),
        observed=None,
        support=Discrete(n=4),
        depends_on_prediction=False,
    )
    # Only ScaledNormal's own array parameter is a leaf; support and
    # depends_on_prediction are static metadata, like name/parents/plate.
    leaves = jax.tree.leaves(n)
    assert len(leaves) == 1
    assert eqx.is_inexact_array(leaves[0])
    assert n.support == Discrete(n=4)
    assert n.depends_on_prediction is False


def test_discrete_supports_of_different_n_are_distinct():
    """Discrete carries its state count, not just 'discrete-ness' -- two
    Discrete supports with different n must not compare or hash equal,
    since they are different claims about the model.
    """
    assert Discrete(n=3) != Discrete(n=4)
    assert Discrete(n=3) == Discrete(n=3)
    assert hash(Discrete(n=3)) == hash(Discrete(n=3))
    assert Continuous() == Continuous()


def test_every_node_type_is_a_node():
    for cls in (Const, Deterministic, Probabilistic):
        assert issubclass(cls, Node)
