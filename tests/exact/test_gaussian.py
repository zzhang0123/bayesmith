"""The (loc, scale) extractor, and the log_prob probe that checks it."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro import handlers

from bayesmith import const, det, evaluate, observe, sample, to_numpyro, trace
from bayesmith.errors import NotGaussian, StructureError
from bayesmith.exact.gaussian import (
    check_gaussian,
    gaussian_parts,
    node_shape,
    noise_std_at,
    observation_parts,
)
from tests.exact.models import (
    LyingNormal,
    plated_latent,
    radiometer,
    straight_line,
    two_observations,
)

WEIGHT = 2.5
SIGMA = 0.5


def test_gaussian_parts_reads_loc_and_scale_off_a_plain_normal():
    graph = straight_line(weight=WEIGHT, sigma=SIGMA, prior_std=2.0)
    env = evaluate(graph, {"w": jnp.asarray(WEIGHT)})
    loc, scale = gaussian_parts(graph, graph.node("d"), env)
    assert jnp.allclose(loc, WEIGHT * graph.node("X").value)
    assert scale.shape == loc.shape
    assert jnp.allclose(scale, SIGMA)


def test_gaussian_parts_unwraps_a_to_event_wrapper():
    """`.to_event(1)` only changes how log_prob is reduced, not the density."""

    def model():
        xs = const("X", jnp.ones(4))
        w = sample("w", lambda: dist.Normal(0.0, 1.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, 0.25).to_event(1), mu, obs=jnp.zeros(4))

    graph = trace(model)
    loc, scale = gaussian_parts(graph, graph.node("d"), evaluate(graph, {"w": 1.0}))
    assert loc.shape == (4,)
    assert jnp.allclose(scale, 0.25)


def test_gaussian_parts_refuses_a_node_that_is_not_gaussian():
    def model():
        sample("k", lambda: dist.Gamma(2.0, 3.0))
        observe("d", lambda: dist.Normal(0.0, 1.0), obs=jnp.zeros(()))

    graph = trace(model)
    env = evaluate(graph, {"k": jnp.asarray(1.0)})
    with pytest.raises(NotGaussian, match="Gamma"):
        gaussian_parts(graph, graph.node("k"), env)


def test_check_gaussian_accepts_a_real_normal():
    graph = straight_line()
    env = evaluate(graph, {"w": jnp.asarray(WEIGHT)})
    errors = check_gaussian(graph, graph.node("d"), env)
    assert set(errors) and all(err < 1e-4 for err in errors.values())


def test_check_gaussian_catches_a_distribution_that_lies_about_its_log_prob():
    """The whole reason the probe exists.

    Introspection passes here -- LyingNormal IS a Normal, its `.loc` and
    `.scale` are exactly what the model meant -- and the density is still
    wrong. Delete the probe and this test is the one that goes red.
    """

    def model():
        w = sample("w", lambda: dist.Normal(0.0, 1.0))
        observe("d", lambda w_: LyingNormal(w_, 0.7), w, obs=jnp.zeros(3))

    graph = trace(model)
    env = evaluate(graph, {"w": jnp.asarray(0.3)})
    loc, scale = gaussian_parts(graph, graph.node("d"), env)  # introspection is happy
    assert jnp.allclose(loc, 0.3) and jnp.allclose(scale, 0.7)
    with pytest.raises(StructureError, match="log_prob"):
        check_gaussian(graph, graph.node("d"), env)


def test_check_gaussian_refuses_a_scale_that_is_not_strictly_positive():
    def model():
        w = sample("w", lambda: dist.Normal(0.0, 1.0))
        observe("d", lambda w_: dist.Normal(w_, 0.0), w, obs=jnp.zeros(()))

    graph = trace(model)
    with pytest.raises(StructureError, match="scale"):
        check_gaussian(graph, graph.node("d"), evaluate(graph, {"w": jnp.asarray(0.0)}))


def test_node_shape_agrees_with_the_numpyro_bridge():
    """An independent reading of the same question.

    The bridge builds the site through numpyro.plate; node_shape derives the
    shape from the distribution and the declared plate size. They must agree,
    or the block's domain is a different space from the one NUTS samples.
    """
    graph = plated_latent(n=6)
    traced = handlers.trace(
        handlers.seed(to_numpyro(graph), jax.random.key(0))
    ).get_trace()
    env = evaluate(graph, {"z": traced["z"]["value"]})
    assert node_shape(graph, graph.node("z"), env) == traced["z"]["value"].shape == (6,)
    assert node_shape(graph, graph.node("d"), env) == traced["d"]["value"].shape == (6,)


def test_observation_parts_covers_every_observed_node():
    graph = two_observations(n=7, m=5)
    env = evaluate(graph, {"w": jnp.asarray(1.25)})
    data, loc, scale = observation_parts(graph, env)
    assert set(data) == set(loc) == set(scale) == {"d1", "d2"}
    assert data["d1"].shape == loc["d1"].shape == scale["d1"].shape == (7,)
    assert data["d2"].shape == loc["d2"].shape == scale["d2"].shape == (5,)
    assert jnp.allclose(scale["d1"], 0.3) and jnp.allclose(scale["d2"], 0.9)


def test_noise_std_at_moves_with_the_latent_only_for_a_prediction_dependent_node():
    """The seam that decides Wiener vs GLS, exercised on both sides."""
    constant = straight_line()
    a = noise_std_at(constant, {"w": jnp.asarray(1.0)})["d"]
    b = noise_std_at(constant, {"w": jnp.asarray(9.0)})["d"]
    assert jnp.allclose(a, b)

    tracking = radiometer()
    c = noise_std_at(tracking, {"w": jnp.asarray(1.0)})["d"]
    d = noise_std_at(tracking, {"w": jnp.asarray(9.0)})["d"]
    assert not jnp.allclose(c, d)
    assert jnp.all(d > c)
