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

from bayesmith import compile as compile_graph
from bayesmith import const, det, observe, sample, trace
from bayesmith.artifacts.base import NamedArray
from bayesmith.dispatch.predictive import (
    forward_draws,
    pointwise_log_likelihood,
    prior_draws,
    replicated_draws,
)
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


# ---------------------------------------------------- the §0.7 forward seam


def test_prior_draws_cover_every_node_including_the_observed_one():
    """A prior draw is a walk of the WHOLE graph, observed nodes included.

    That is what makes it a prior PREDICTIVE rather than a prior sample: §0.7
    draws the observed node from its own distribution too, so the same
    primitive answers "what data would this model produce before seeing any"
    and "what parameters would it produce".  A version that stopped at the
    latents would need a second pass to reach the data, which is the parallel
    simulator invariant 1 forbids.
    """
    graph = straight_line()
    env = prior_draws(graph, jax.random.key(9), 16)

    assert set(env) == {node.name for node in graph.nodes}
    assert np.shape(env["w"]) == (16,)
    assert np.shape(env["d"]) == (16, 8)
    observed = np.asarray(graph.node("d").observed)
    assert not bool(np.any(np.asarray(env["d"]) == observed))
    # A Const is carried, not drawn: the same grid in every draw.
    assert np.shape(env["X"]) == (16, 8)
    assert bool(np.all(np.asarray(env["X"])[0] == np.asarray(env["X"])[15]))


def test_prior_draws_match_the_closed_form_moments_of_straight_line():
    """``w ~ N(0, 2)`` and ``d_i ~ N(0, sqrt(4 x_i^2 + 0.25))``, checked
    against the draws.

    The closed form is the whole reason this fixture is the one used: with
    ``mu = w x`` and ``d ~ N(mu, sigma)``, marginalising ``w`` gives a
    zero-mean Gaussian of variance ``prior_std^2 x^2 + sigma^2``, so the test
    knows the answer rather than comparing two implementations.

    **The band is DERIVED and the seed is fixed.**  A sample mean of n draws
    has standard error ``sd / sqrt(n)``; a sample standard deviation of a
    Gaussian has relative standard error ``1 / sqrt(2n)``.  Five of each is
    the band, so a correct implementation fails this roughly once in
    3.5 million per assertion.  Nothing here is a measured tolerance: the
    numbers 0 and ``sqrt(4 x^2 + 1/4)`` are the model's, and the width is the
    estimator's.

    **Measured margins at this seed** (derived band, measured occupancy):
    worst mean 2.073 sigma, worst standard deviation 1.319 sigma.  Recorded so
    that a later failure can be read as "this cell was never comfortable"
    rather than "something moved", which a bare pass/fail cannot answer.

    The eight ``d`` means are NOT eight independent 2-sigma coincidences: they
    all sit between 1.61 and 2.07 sigma and in the same direction, because
    ``d_i = w x_i + eps_i`` shares one ``w``, so their sample means are
    dominated by the single sample mean of ``w`` (1.748 sigma). The vector
    assertion is therefore closer to one test than to eight, which is worth
    knowing before anyone reads the row of passes as independent evidence.
    """
    n = 4000
    graph = straight_line()
    env = prior_draws(graph, jax.random.key(9), n)
    x = np.asarray(graph.node("X").value)
    expected_sd = np.sqrt(4.0 * x**2 + 0.25)  # prior_std=2.0, sigma=0.5

    w = np.asarray(env["w"])
    assert abs(w.mean()) <= 5.0 * 2.0 / np.sqrt(n)
    assert abs(w.std() / 2.0 - 1.0) <= 5.0 / np.sqrt(2 * n)

    d = np.asarray(env["d"])
    assert np.all(np.abs(d.mean(axis=0)) <= 5.0 * expected_sd / np.sqrt(n))
    assert np.all(np.abs(d.std(axis=0) / expected_sd - 1.0) <= 5.0 / np.sqrt(2 * n))


def test_prior_draws_expand_a_plated_node_to_its_plate_size():
    """§0.7's plate rule, which is the one thing the probe could not do.

    :func:`~bayesmith.graph.evaluate.apply_probabilistic` deliberately returns
    an UNMAPPED distribution for a plated node whose parents are not plated --
    that is the "N iid draws from one shared prior" pattern ``log_joint`` and
    the numpyro bridge both rely on, and its ``log_prob`` broadcasts correctly
    across the plate.  Its ``.sample``, however, is one shared value, so a
    forward walk that trusted it would put the SAME z at every plate index and
    look entirely plausible doing it.  The primitive reads
    ``graph.plate_size`` and expands.

    ``d`` is the control: its parent ``z`` IS plated, so it is already mapped
    and must not be expanded a second time.
    """
    graph = plated_latent()
    env = prior_draws(graph, jax.random.key(1), 5)

    assert np.shape(env["z"]) == (5, 6)
    assert np.shape(env["d"]) == (5, 6)
    z = np.asarray(env["z"])
    assert len(np.unique(z[0])) == 6, "one shared value repeated across the plate"


def test_prior_draws_reproduce_the_probe_measurement_of_section_8():
    """The number §0.2's ``prior_predictive_check`` row is built on, produced
    by the PRIMITIVE rather than by the probe that measured it.

    probe_28 §8 reports ``P(max|d_prior| >= max|d_obs|) = 0.210`` for
    ``straight_line`` at 4000 draws from ``jax.random.key(9)``, and the plan
    quotes it as the PASS end of the prior-scale check.  The probe has its own
    walk of the graph; this module has another.  They agree bit for bit
    because both split one key per node in ``graph.nodes`` order, which is the
    discipline recorded in ``prior_draws``' own docstring -- so this assertion
    is a genuine cross-implementation check, not a restatement.

    0.210 is 840 draws of 4000 exactly.  No linear solve is involved anywhere
    on this path: threefry bits, a scalar-vector multiply and an elementwise
    Normal sample, so the count is reproducible rather than merely stable.
    """
    graph = straight_line()
    drawn = np.asarray(prior_draws(graph, jax.random.key(9), 4000)["d"])
    observed = np.asarray(graph.node("d").observed)
    tail = float(np.mean(np.abs(drawn).max(axis=1) >= np.abs(observed).max()))
    assert tail == pytest.approx(0.210, abs=1.0 / 4000)


def test_forward_draws_generate_observations_at_one_parameter_setting():
    """The FIXED arm: n observations from ONE parameter value.

    The mean of the draws is the loc ``evaluate`` computes at that value, to
    within the estimator's own standard error -- again a derived band
    (``sigma / sqrt(n)``, five of them) rather than a measured tolerance.
    Measured occupancy at this seed: worst 1.134 sigma of the five.
    """
    n = 2000
    graph = straight_line()
    values = {"w": jnp.asarray(2.5)}
    drawn = forward_draws(graph, values, jax.random.key(4), n)["d"]

    assert np.shape(drawn) == (n, 8)
    loc = np.asarray(evaluate(graph, values)["mu"])
    assert np.all(np.abs(np.asarray(drawn).mean(axis=0) - loc) <= 5.0 * 0.5 / np.sqrt(n))


def test_forward_draws_are_replicated_draws_at_a_repeated_parameter():
    """One generation law, shown structurally rather than asserted.

    §0.7's three parameter sources must not become three simulators.  The
    fixed arm is the replicate arm at a parameter that happens not to vary, so
    the two agree BIT FOR BIT -- ``rtol=0`` -- and any future divergence
    between them is a divergence in one shared function rather than a drift
    between two.
    """
    graph = straight_line()
    n = 32
    key = jax.random.key(11)
    fixed = forward_draws(graph, {"w": jnp.asarray(2.5)}, key, n)["d"]
    broadcast = replicated_draws(graph, {"w": jnp.full((n,), 2.5)}, key)["d"]
    np.testing.assert_array_equal(np.asarray(fixed), np.asarray(broadcast))


def test_forward_draws_refuse_a_correlated_observation_like_their_sibling():
    """§0.4's coverage domain reaches the fixed arm too: it goes through the
    same ``observation_parts`` walk, so it refuses where that walk refuses
    rather than approximating a correlated node into a diagonal one."""
    graph = _correlated_graph()
    with pytest.raises(NotGaussian):
        forward_draws(graph, {"w": jnp.asarray(2.0)}, jax.random.key(0), 4)
