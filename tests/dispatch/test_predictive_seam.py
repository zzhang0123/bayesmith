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
from bayesmith import const, det, observe, plate, sample, trace
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


@pytest.mark.parametrize("kind", ["bernoulli", "poisson"])
def test_a_non_gaussian_without_loc_is_refused_before_gaussian_shape_access(kind):
    """Missing loc is a supported-domain refusal, not an AttributeError.

    StudentT has loc and never exposed the ordering bug. Bernoulli/Poisson
    must reach the same NotGaussian boundary so posterior result assembly
    can omit predictive metadata without discarding completed NUTS draws.
    """
    def model():
        theta = sample("theta", lambda: dist.Normal(0.0, 1.0))
        law = (lambda t: dist.Bernoulli(logits=t)) if kind == "bernoulli" else (
            lambda t: dist.Poisson(jnp.exp(t))
        )
        observe("obs", law, theta, obs=jnp.array([0, 1, 1]))

    graph = trace(model)
    values = {"theta": jnp.linspace(-0.2, 0.2, 8)}
    with pytest.raises(NotGaussian) as caught:
        pointwise_log_likelihood(graph, values)
    assert caught.value.reason == "not_normal"
    with pytest.raises(NotGaussian):
        replicated_draws(graph, values, jax.random.key(0))


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


@pytest.mark.parametrize("plated", [False, True])
@pytest.mark.parametrize("law", [
    lambda: dist.Bernoulli(0.5),
    lambda: dist.Normal(jnp.zeros(2), 1.0).to_event(1),
])
def test_prior_generation_broadcasts_recorded_batch_shape_and_preserves_events(law, plated):
    event = law().event_shape

    def model():
        group = plate("groups", 4) if plated else None
        observe("obs", law, obs=jnp.zeros((4, *event)), plate=group)

    obs = prior_draws(trace(model), jax.random.key(3), 32)["obs"]
    assert obs.shape == (32, 4, *event)
    assert not np.array_equal(obs[:, 0], obs[:, 1])


def test_unknown_conditioned_site_is_refused_before_drawing():
    with pytest.raises(ValueError, match="misspelt"):
        replicated_draws(
            straight_line(), {"w": jnp.zeros(2)}, jax.random.key(1),
            conditioned_sites=frozenset({"misspelt"}),
        )


def test_complex_pointwise_scores_both_parts_and_keeps_event_coordinates():
    from bayesmith import observe, sample, trace
    from bayesmith.distributions import ComplexNormal

    data = np.array([0.0 + 0.0j, 0.0 + 0.5j])
    scale = 0.1

    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe("z", lambda v: ComplexNormal(jnp.stack([v, v]) + 0j, scale).to_event(1),
                x, obs=data)

    draws = {"x": jnp.zeros(4)}
    density = pointwise_log_likelihood(trace(model), draws)
    expected = -np.abs(data) ** 2 / (2 * scale**2) - np.log(2 * np.pi * scale**2)
    assert expected[0] > 0 > expected[1]
    assert np.isrealobj(density.value)
    np.testing.assert_allclose(density.value, np.broadcast_to(expected, (4, 2)), rtol=2e-6)


def test_predictive_scores_and_samples_a_scalar_mean_with_different_observation_widths():
    widths = np.array([0.5, 2.0])
    datum = np.array([1.0, 1.0])

    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe("y", lambda v: dist.Normal(v, widths), x, obs=datum)

    graph = trace(model)
    draws = {"x": jnp.zeros(2048)}
    scores = pointwise_log_likelihood(graph, draws)
    expected = -0.5 * (datum / widths) ** 2 - np.log(widths * np.sqrt(2 * np.pi))
    np.testing.assert_allclose(scores.value, np.broadcast_to(expected, (2048, 2)), rtol=2e-6)
    replicated = replicated_draws(graph, draws, jax.random.key(31))["y"]
    np.testing.assert_allclose(np.std(replicated, axis=0), widths, rtol=0.08)


def test_predictive_scores_and_replicates_a_gaussian_with_a_discrete_parent():
    def model():
        k = sample("k", lambda: dist.Bernoulli(0.25))
        observe("y", lambda k: dist.Normal(k, 0.5), k, obs=1.0)

    graph = trace(model)
    draws = {"k": jnp.tile(jnp.array([0, 1]), 1024)}
    scores = pointwise_log_likelihood(graph, draws)
    expected = -2 * (1 - draws["k"]) ** 2 - np.log(0.5 * np.sqrt(2 * np.pi))
    np.testing.assert_allclose(scores.value, expected, atol=2e-6)
    y = replicated_draws(graph, draws, jax.random.key(18))["y"]
    np.testing.assert_allclose([y[::2].mean(), y[1::2].mean()], [0, 1], atol=0.06)


@pytest.mark.parametrize("rows", [2, 3])
def test_plated_children_receive_their_own_batch_and_event_coordinates(rows):
    groups, width = 3, 2

    def model():
        group = plate("group", groups)
        x = sample("x", lambda: dist.Normal(
            10 * jnp.arange(rows)[:, None, None] + jnp.arange(width), .1
        ).to_event(1), plate=group)
        total = det("total", jnp.sum, x, plate=group)
        observe("y", lambda v, t: dist.Normal(v[:, 0] + t, .1),
                x, total, obs=jnp.zeros((rows, groups)), plate=group)

    graph = trace(model)
    draw = prior_draws(graph, jax.random.key(0), 12)
    assert draw["x"].shape == (12, rows, groups, width)
    expected_total = draw["x"].sum(axis=(1, 3))
    np.testing.assert_allclose(draw["total"], expected_total, rtol=1e-6)
    expected_mean = draw["x"][..., 0] + expected_total[:, None, :]
    assert draw["y"].shape == expected_mean.shape
    assert np.max(np.abs(draw["y"] - expected_mean)) < .7
    scores = pointwise_log_likelihood(graph, {"x": draw["x"]})
    assert scores.dims == ("draw", "y_dim0", "group")
    np.testing.assert_allclose(scores.value, dist.Normal(expected_mean, .1).log_prob(0.), rtol=1e-6)


@pytest.mark.parametrize("matrix", [False, True])
def test_transformed_member_laws_keep_vector_and_matrix_parameters(matrix):
    from numpyro.infer.util import log_density

    from bayesmith.bridge.numpyro_bridge import to_numpyro
    from bayesmith.graph.evaluate import log_joint

    def model():
        group = plate("group", 3)
        centres = const("centres", 10. * jnp.arange(3), plate=group)

        def member(centre):
            loc = centre + jnp.arange(4)
            transform = (
                dist.transforms.LowerCholeskyAffine(loc, 0.1 * jnp.eye(4)) if matrix
                else dist.transforms.AffineTransform(loc, 0.1)
            )
            return dist.TransformedDistribution(dist.Normal(jnp.zeros((2, 4)), 1.).to_event(1),
                                                transform)

        sample("y", member, centres, plate=group)

    graph = trace(model)
    loc = 10. * jnp.arange(3)[:, None] + jnp.arange(4)
    y = prior_draws(graph, jax.random.key(7), 8)["y"]
    assert y.shape == (8, 2, 3, 4)
    assert np.max(np.abs(y - loc)) < .7
    expected = dist.Normal(loc, .1).log_prob(y[0]).sum()
    assert log_joint(graph, {"y": y[0]}) == pytest.approx(float(expected), abs=2e-5)
    actual, _ = log_density(to_numpyro(graph), (), {}, {"y": y[0]})
    assert actual == pytest.approx(float(expected), abs=2e-5)


def test_shared_plated_ancestors_are_built_once_per_forward_scan():
    from numpyro.infer.util import log_density

    from bayesmith.bridge.numpyro_bridge import to_numpyro
    from bayesmith.graph.evaluate import log_joint

    calls = []

    def member(*parents):
        calls.append(None)
        return dist.Normal(sum(parents, 0.), 1.)

    def model():
        group = plate("group", 3)
        nodes = []
        for i in range(8):
            nodes.append(sample(f"x{i}", member, *nodes[-2:], plate=group))

    graph = trace(model)
    values = {name: jnp.zeros(3) for name in graph.latents}
    for scan in (
        lambda: log_joint(graph, values),
        lambda: prior_draws(graph, jax.random.key(1), 2),
        lambda: log_density(to_numpyro(graph), (), {}, values),
    ):
        calls.clear()
        scan()
        assert len(calls) == len(values)
