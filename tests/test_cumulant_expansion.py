"""Independent polynomial and graph checks for the manual Edgeworth likelihood."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpy.polynomial.hermite_e import hermeval
from numpyro.infer.util import log_density

from bayesmith.bridge.numpyro_bridge import to_numpyro
from bayesmith.distributions import EdgeworthExpansion
from bayesmith.graph.evaluate import log_joint
from bayesmith.graph.trace import const, observe, plate, sample, trace


def _hermite(n, z):
    return hermeval(z, [0.0] * n + [1.0])


@pytest.mark.parametrize("order", [2, 3, 4, 5, 6])
def test_low_orders_against_explicit_independent_polynomials(order):
    z = np.array([-1.25, -0.25, 0.5, 1.5])
    a, b, c, d = 0.03, 0.004, -0.0002, 0.00003
    # a_r = standardized cumulant / r!, independent of implementation.
    terms = [
        np.ones_like(z),
        a * _hermite(3, z),
        b * _hermite(4, z) + a**2 / 2 * _hermite(6, z),
        c * _hermite(5, z) + a * b * _hermite(7, z) + a**3 / 6 * _hermite(9, z),
        d * _hermite(6, z)
        + (a * c + b**2 / 2) * _hermite(8, z)
        + a**2 * b / 2 * _hermite(10, z)
        + a**4 / 24 * _hermite(12, z),
    ]
    scale = 2.0
    cumulants = (
        6 * a * scale**3,
        24 * b * scale**4,
        120 * c * scale**5,
        720 * d * scale**6,
    )
    expansion = EdgeworthExpansion(0.75, scale, cumulants=cumulants, order=order)
    values = 0.75 + scale * z
    expected = sum(terms[: order - 1])
    np.testing.assert_allclose(expansion.correction(values), expected, rtol=3e-6)
    expected_log = dist.Normal(0.75, scale).log_prob(values) + np.log(expected)
    np.testing.assert_allclose(expansion.log_prob(values), expected_log, rtol=3e-6)


@pytest.mark.parametrize("order", [2, 4, 8])
def test_zero_higher_cumulants_recover_gaussian(order):
    normal = dist.Normal(jnp.array([1.0, -2.0]), jnp.array([0.5, 3.0]))
    expansion = EdgeworthExpansion(
        normal.loc, normal.scale, cumulants=(0.0,) * (order - 2), order=order
    )
    x = jnp.array([[0.0], [2.0]])
    np.testing.assert_array_equal(expansion.log_prob(x), normal.log_prob(x))


def test_negative_expansion_is_visible_and_never_clipped():
    expansion = EdgeworthExpansion(0.0, 1.0, cumulants=(6.0,), order=3)
    # At z=-3: 1 + H3(-3) = -17, deterministically negative.
    assert expansion.correction(-3.0) == -17.0
    assert jnp.isnan(expansion.log_prob(-3.0))
    assert jnp.isnan(jax.jit(expansion.log_prob)(-3.0))


def test_non_gaussian_sampling_is_explicitly_unimplemented():
    expansion = EdgeworthExpansion(0.0, 1.0, cumulants=(0.1,), order=3)
    with pytest.raises(NotImplementedError, match="likelihood"):
        expansion.sample(jax.random.key(0), (5,))


def test_order_two_sampling_uses_the_gaussian():
    expansion = EdgeworthExpansion(jnp.array([0.0, 2.0]), 0.5, order=2)
    key = jax.random.key(5)
    np.testing.assert_array_equal(
        expansion.sample(key, (3,)), dist.Normal(expansion.loc, 0.5).sample(key, (3,))
    )


@pytest.mark.parametrize("order", [True, 1, 0, -2, 3.5, "4"])
def test_bad_orders_are_rejected(order):
    with pytest.raises((TypeError, ValueError), match="order"):
        EdgeworthExpansion(0.0, 1.0, cumulants=(0.1, 0.2), order=order)


def test_missing_cumulants_are_not_silently_zeroed():
    with pytest.raises(ValueError, match="cumulants"):
        EdgeworthExpansion(0.0, 1.0, cumulants=(0.1,), order=4)


def test_jit_vmap_and_gradient_retain_cumulants_and_static_order():
    def build(k3):
        return EdgeworthExpansion(0.0, 2.0, cumulants=(k3, 0.2), order=4)

    batched = jax.jit(jax.vmap(build))(jnp.array([0.1, 0.3]))
    assert batched.order == 4
    np.testing.assert_allclose(
        batched.log_prob(1.0), jnp.stack([build(k).log_prob(1.0) for k in (0.1, 0.3)])
    )
    k3, z = 0.3, 0.5
    h3, h4, h6 = (_hermite(n, z) for n in (3, 4, 6))
    correction = 1 + k3 / 8 / 6 * h3 + 0.2 / 16 / 24 * h4 + (k3 / 8) ** 2 / 72 * h6
    derivative = (h3 / 48 + 2 * k3 / 64 / 72 * h6) / correction
    actual = jax.jit(jax.grad(lambda k: build(k).log_prob(1.0)))(k3)
    np.testing.assert_allclose(actual, derivative, rtol=3e-6)


def test_conditional_plated_graph_and_numpyro_bridge_match_explicit_density():
    data = jnp.array([0.2, 0.8, 1.5])
    centers = jnp.array([0.0, 0.5, 1.0])

    def model():
        p = plate("rows", 3)
        x = const("x", centers, plate=p)
        k3 = sample("k3", lambda: dist.Normal(0.0, 0.2))
        observe(
            "y",
            lambda m, k: EdgeworthExpansion(m, 1.0, cumulants=(k, 0.1), order=4),
            x,
            k3,
            obs=data,
            plate=p,
        )

    graph = trace(model)
    at = {"k3": jnp.array(0.15)}
    z = np.asarray(data - centers)
    correction = 1 + 0.15 / 6 * _hermite(3, z) + 0.1 / 24 * _hermite(4, z)
    correction += 0.15**2 / 72 * _hermite(6, z)
    expected = dist.Normal(0.0, 0.2).log_prob(at["k3"])
    expected += jnp.sum(dist.Normal(centers, 1.0).log_prob(data) + np.log(correction))
    bridged, _ = log_density(to_numpyro(graph), (), {}, at)
    np.testing.assert_allclose(log_joint(graph, at), expected, rtol=3e-6)
    np.testing.assert_allclose(bridged, expected, rtol=3e-6)
    derivative = jax.jit(jax.grad(lambda k: log_joint(graph, {"k3": k})))(at["k3"])
    assert jnp.isfinite(derivative)
