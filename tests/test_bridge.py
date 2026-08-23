import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer.util import log_density

from bayesmith.bridge.numpyro_bridge import to_numpyro
from bayesmith.graph.evaluate import log_joint
from bayesmith.graph.trace import const, det, observe, plate, sample, trace


def _graph(data):
    def model():
        X = const("X", jnp.array([1.0, 2.0, 3.0]))
        a = sample("a", lambda: dist.Normal(0.0, 2.0))
        mu = det("mu", lambda a_, X_: a_ * X_, a, X, linear_in=("a",))
        observe("d", lambda m: dist.Normal(m, 0.5), mu, obs=data)

    return trace(model)


def test_the_bridge_and_log_joint_agree_on_the_density():
    """Two independent readings of the same graph must give the same number."""
    graph = _graph(jnp.array([1.0, 2.0, 3.0]))
    at = {"a": jnp.array(0.7)}

    ours = log_joint(graph, at)
    theirs, _ = log_density(to_numpyro(graph), (), {}, at)
    assert jnp.allclose(ours, theirs, rtol=1e-6)


def test_latent_sites_carry_the_graph_node_names():
    graph = _graph(jnp.array([1.0, 2.0, 3.0]))
    trace_ = numpyro.handlers.trace(
        numpyro.handlers.seed(to_numpyro(graph), jax.random.key(0))
    ).get_trace()
    assert trace_["a"]["type"] == "sample"
    assert not trace_["a"]["is_observed"]
    assert trace_["d"]["is_observed"]


def test_deterministic_nodes_are_recorded_as_numpyro_deterministic():
    graph = _graph(jnp.array([1.0, 2.0, 3.0]))
    trace_ = numpyro.handlers.trace(
        numpyro.handlers.seed(to_numpyro(graph), jax.random.key(0))
    ).get_trace()
    assert trace_["mu"]["type"] == "deterministic"


def test_a_plated_graph_bridges_and_still_agrees():
    def model():
        obs = plate("obs", 3)
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe(
            "d", lambda v: dist.Normal(v, 1.0), x,
            obs=jnp.array([1.0, 2.0, 3.0]), plate=obs,
        )

    graph = trace(model)
    at = {"x": jnp.array(0.4)}
    ours = log_joint(graph, at)
    theirs, _ = log_density(to_numpyro(graph), (), {}, at)
    assert jnp.allclose(ours, theirs, rtol=1e-6)


def test_the_bridged_model_can_be_sampled_from_the_prior():
    graph = _graph(jnp.array([1.0, 2.0, 3.0]))
    predictive = numpyro.infer.Predictive(to_numpyro(graph), num_samples=8)
    draws = predictive(jax.random.key(1))
    assert draws["a"].shape == (8,)


def test_a_plated_latent_site_carries_the_plate_axis():
    """A dropped plate is invisible to the density-agreement check: without
    subsampling, ``numpyro.plate`` contributes no scale factor, and the
    plated test above only plates an *observed* site of fixed shape. Plate a
    *latent* instead: ``dist.Normal(0.0, 1.0)`` has batch shape ``()``, so a
    sampled value only picks up a leading axis if the plate was actually
    applied. Plate size 5 appears nowhere else in this file (the other
    sizes are 3 and 8), so a wrong-but-plausible shape can't pass by luck.
    """
    n = 5

    def model():
        obs = plate("obs", n)
        sample("x", lambda: dist.Normal(0.0, 1.0), plate=obs)

    graph = trace(model)
    trace_ = numpyro.handlers.trace(
        numpyro.handlers.seed(to_numpyro(graph), jax.random.key(0))
    ).get_trace()
    assert trace_["x"]["value"].shape == (n,)
