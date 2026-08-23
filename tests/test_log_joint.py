import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from bayesmith.graph.evaluate import log_joint
from bayesmith.graph.trace import const, det, observe, sample, trace


def _conjugate_graph(data, tau=2.0, sigma=0.5):
    """x ~ N(0, tau^2);  d_i ~ N(x, sigma^2)."""

    def model():
        x = sample("x", lambda: dist.Normal(0.0, tau))
        observe("d", lambda v: dist.Normal(v, sigma), x, obs=data)

    return trace(model)


def test_log_joint_matches_the_hand_written_density():
    data = jnp.array([1.0, 2.0, 3.0])
    tau, sigma, x = 2.0, 0.5, 0.7
    graph = _conjugate_graph(data, tau, sigma)

    got = log_joint(graph, {"x": jnp.array(x)})
    expected = dist.Normal(0.0, tau).log_prob(x) + jnp.sum(
        dist.Normal(x, sigma).log_prob(data)
    )
    assert jnp.allclose(got, expected, rtol=1e-6)


def test_deterministic_nodes_contribute_no_density():
    data = jnp.array([1.0])

    def with_det():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        mu = det("mu", lambda v: v, x)
        observe("d", lambda m: dist.Normal(m, 1.0), mu, obs=data)

    def without_det():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe("d", lambda v: dist.Normal(v, 1.0), x, obs=data)

    at = {"x": jnp.array(0.3)}
    assert jnp.allclose(log_joint(trace(with_det), at), log_joint(trace(without_det), at))


def test_const_nodes_contribute_no_density():
    def model():
        const("X", jnp.array([5.0, 6.0]))
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe("d", lambda v: dist.Normal(v, 1.0), x, obs=jnp.array([1.0]))

    at = {"x": jnp.array(0.3)}
    got = log_joint(trace(model), at)
    expected = dist.Normal(0.0, 1.0).log_prob(0.3) + dist.Normal(0.3, 1.0).log_prob(1.0)
    assert jnp.allclose(got, expected, rtol=1e-6)


def test_log_joint_is_differentiable_in_the_latent_value():
    """The gradient of a Gaussian log-density is available in closed form."""
    data = jnp.array([1.0, 2.0, 3.0])
    tau, sigma = 2.0, 0.5
    graph = _conjugate_graph(data, tau, sigma)

    grad = jax.grad(lambda x: log_joint(graph, {"x": x}))(jnp.array(0.7))
    expected = -0.7 / tau**2 + jnp.sum(data - 0.7) / sigma**2
    assert jnp.allclose(grad, expected, rtol=1e-5)


def test_log_joint_is_jittable():
    graph = _conjugate_graph(jnp.array([1.0, 2.0]))
    jitted = eqx.filter_jit(lambda g, x: log_joint(g, {"x": x}))
    assert jnp.isfinite(jitted(graph, jnp.array(0.4)))


def test_log_joint_is_a_scalar_whatever_the_data_shape():
    for shape in [(1,), (7,), (3, 4)]:
        graph = _conjugate_graph(jnp.ones(shape))
        assert log_joint(graph, {"x": jnp.array(0.0)}).shape == ()
