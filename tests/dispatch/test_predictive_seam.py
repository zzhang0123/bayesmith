"""The predictive seam primitives, pinned before they exist.

Task 1 writes these RED: :mod:`bayesmith.dispatch.predictive` does not exist
yet, so every test below that reaches it fails at import.  Task 2 creates the
module and turns them green.  The seam itself is three facts from §0.1/§0.3:

* `observation_parts` returns `(data, loc, scale)` broadcast to one shape, so
  replay and replicate share the same loc/scale and differ only by log_prob vs
  sample;
* `replicated_draws` samples the observation distribution per source draw --
  a NEW random result, elementwise different from the observed datum;
* `pointwise_log_likelihood` evaluates log_prob(observed) per source draw,
  zeroing masked positions, with the draw axis leading and the observation-unit
  axes named by the _dims rule (draw, plate, {name}_dim{i} fallback).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from bayesmith.dispatch.predictive import (
    pointwise_log_likelihood,
    replicated_draws,
)

from bayesmith import compile as compile_graph
from bayesmith import const, det, observe, sample, trace
from bayesmith.artifacts.base import NamedArray
from bayesmith.errors import NotGaussian
from bayesmith.exact.gaussian import observation_parts
from bayesmith.graph.evaluate import evaluate
from tests.exact.models import plated_latent, radiometer, straight_line


def _draws(graph, key=0, n=8):
    """Real posterior draws, small: the same seam the probe walks."""
    plan = compile_graph(graph)
    posterior = plan.sample(
        jax.random.key(key), num_samples=n, num_warmup=8, num_chains=1,
        nuts_on_collapse=False,
    )
    return {name: posterior.samples[name] for name in graph.latents}


def _correlated_graph(size=8, weight=2.0, decay=0.4):
    """A graph whose observed node declares correlated (CirculantNormal) noise."""
    lag = np.minimum(np.arange(size), size - np.arange(size))
    kernel = jnp.asarray(1.0 * decay**lag + 0.5)
    x = jnp.linspace(1.0, 4.0, size)
    data = weight * x

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 5.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.CirculantNormal(m, kernel), mu,
                depends_on_prediction=False, obs=data)

    return trace(model)


def _masked_graph():
    """d_i ~ N(w X_i, 0.5), the last observation flagged as not taken."""
    x = jnp.linspace(1.0, 4.0, 4)
    data = jnp.array([2.4, 5.1, 7.6, 10.2])
    mask = jnp.array([True, True, True, False])

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 2.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, 0.5), mu, obs=data, mask=mask)

    return trace(model)


# ------------------------------------------------------------ the §0.1 seam


def test_observation_parts_returns_three_dicts_with_aligned_leaves():
    """(data, loc, scale) keyed by the same observed nodes, broadcast to one
    shape -- so a downstream reduction is one tree-map, never a per-leaf guess."""
    for graph in (radiometer(), straight_line(), plated_latent()):
        env = evaluate(graph, {name: jnp.asarray(v)[0] for name, v in _draws(graph).items()})
        data, loc, scale = observation_parts(graph, env)
        assert set(data) == set(loc) == set(scale) == set(graph.observed)
        for name in graph.observed:
            assert np.shape(data[name]) == np.shape(loc[name]) == np.shape(scale[name])


def test_replicated_draws_samples_the_observation_distribution():
    """One NEW draw per source draw, elementwise different from the observed."""
    graph = radiometer()
    draws = _draws(graph)
    replicated = replicated_draws(graph, draws, jax.random.key(0))
    assert set(replicated) == set(graph.observed)
    observed = np.asarray(graph.node("d").observed)
    assert replicated["d"].shape == (8, 10)
    assert not bool(np.any(np.asarray(replicated["d"]) == observed))


def test_pointwise_log_likelihood_leads_with_the_draw_axis():
    """straight_line's unplated vector observation is one unit per element."""
    graph = straight_line()
    draws = _draws(graph)
    ll = pointwise_log_likelihood(graph, draws)
    assert isinstance(ll, NamedArray)
    assert ll.dims == ("draw", "d_dim0")
    assert ll.value.shape == (8, 8)
    # The value is Normal(loc, scale).log_prob(data), elementwise.
    env = evaluate(graph, {name: jnp.asarray(draws[name])[0] for name in graph.latents})
    data, loc, scale = observation_parts(graph, env)
    expected = dist.Normal(loc["d"], scale["d"]).log_prob(data["d"])
    np.testing.assert_allclose(ll.value[0], np.asarray(expected), rtol=1e-6)


def test_pointwise_log_likelihood_names_a_plated_axis():
    """A plated observed node's observation-unit axis is the plate's own name."""
    graph = plated_latent()
    draws = _draws(graph)
    ll = pointwise_log_likelihood(graph, draws)
    assert ll.dims == ("draw", "obs")
    assert ll.value.shape == (8, 6)


def test_pointwise_log_likelihood_zeroes_masked_positions():
    """A masked (never-taken) sample contributes no log-density term."""
    graph = _masked_graph()
    draws = _draws(graph)
    ll = pointwise_log_likelihood(graph, draws)
    assert ll.value.shape == (8, 4)
    assert bool(np.all(np.asarray(ll.value)[:, 3] == 0.0))
    assert bool(np.all(np.asarray(ll.value)[:, :3] != 0.0))


def test_a_correlated_node_raises_rather_than_silently_approximating():
    """§0.4: the diagonal walk refuses a correlated noise, and the refusal is
    loud -- a NotGaussian the caller can convert into a typed Refusal."""
    graph = _correlated_graph()
    draws = {name: jnp.full((4,), 2.0) for name in graph.latents}
    with pytest.raises(NotGaussian):
        replicated_draws(graph, draws, jax.random.key(0))
    with pytest.raises(NotGaussian):
        pointwise_log_likelihood(graph, draws)
