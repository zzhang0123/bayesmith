"""Log space as a graph transform -- scenarios, refusals, and the solve.

The fixture is the model the capability exists for: a gain entered through an
exponential over a SUMMED sky, under multiplicative noise. ``log`` of a sum
is not affine in the summands, which is exactly the case that looks like it
should obstruct a log-linear gain and does not: conditional on the sky,
``log d = log_gain + log S`` with ``log S`` a known constant, and constants
live in the block's offset.

Every solve assertion here is against a CLOSED FORM, available because the
block is trivial in log space: ``y = log_gain + log S[f]`` with constant
sigma, so the posterior is Gaussian with precision ``n / f^2 + 1 / s0^2``.
``f`` is 4.05e-3 -- a 61 kHz channel at 1 s -- so the tests run where the
first-order equivalence is actually used, not at a round number.
"""

import math

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from bayesmith import det, joint_prior, observe, sample, trace
from bayesmith.errors import NotLogLinear
from bayesmith.exact.gaussian import precision_at
from bayesmith.exact.loglinear import (
    FIRST_ORDER_MAX_FRACTIONAL,
    LOG_DEFAULT_SCALES,
    check_log_linearity,
    log_linear_operator,
    log_space,
    multiplicative_log_data,
)
from bayesmith.exact.solve import gcr_sample, wiener_solve
from bayesmith.graph.evaluate import log_joint

N = 24
SKY = (
    2000.0 * jnp.linspace(1.0, 1.4, N) ** -2.5
    + 150.0 * jnp.exp(-jnp.linspace(0.0, 3.0, N))
    + jnp.zeros(N).at[7].set(400.0)
)
LOG_G = float(jnp.log(1.6))
PRIOR_STD = 10.0
F = 4.05e-3  # 1 / sqrt(61e3 * 1.0)


def _observed(kind: str, key=None) -> jax.Array:
    key = jax.random.key(0) if key is None else key
    draw = jax.random.normal(key, (N,))
    truth = jnp.exp(LOG_G) * SKY
    if kind == "multiplicative":
        return truth * (1.0 + F * draw)
    return truth * jnp.exp(F * draw)


def _graph(kind: str, observed=None, fractional: float = F):
    observed = _observed(kind) if observed is None else observed

    def model(data):
        lam = sample("log_gain", lambda: dist.Normal(0.0, PRIOR_STD))
        if kind == "multiplicative":
            mu = det("mu", lambda l: jnp.exp(l) * SKY, lam)
            observe("d", lambda m: dist.Normal(m, fractional * m), mu, obs=data)
        elif kind == "lognormal":
            ell = det(
                "ell", lambda l: l + jnp.log(SKY), lam, linear_in=("log_gain",)
            )
            observe("d", lambda e: dist.LogNormal(e, fractional), ell, obs=data)
        elif kind == "additive":
            mu = det("mu", lambda l: jnp.exp(l) * SKY, lam)
            observe("d", lambda m: dist.Normal(m, 5.0), mu, obs=data)
        else:
            raise AssertionError(kind)

    return trace(model, observed)


def _posterior(y: jax.Array, sigma: float) -> tuple[float, float]:
    """Closed form for ``y = log_gain + log SKY``, constant sigma."""
    precision = N / sigma**2 + 1.0 / PRIOR_STD**2
    mean = float(jnp.sum(y - jnp.log(SKY)) / sigma**2 / precision)
    return mean, float(1.0 / jnp.sqrt(precision))


class TestScenarioReading:
    def test_multiplicative_normal_is_read_with_its_fractional_level(self):
        ls = log_space(_graph("multiplicative"))
        assert ls.kind == {"d": "multiplicative"}
        assert jnp.allclose(ls.fractional["d"], F, rtol=1e-5)

    def test_lognormal_is_read_as_the_exact_scenario(self):
        ls = log_space(_graph("lognormal"))
        assert ls.kind == {"d": "lognormal"}
        assert "d" not in ls.fractional

    def test_additive_noise_has_no_log_route(self):
        with pytest.raises(NotLogLinear) as caught:
            log_space(_graph("additive"))
        assert "CONSTANT" in str(caught.value)
        # This graph has ONE observed node, so what escapes is the
        # graph-level verdict and the node's own reason is in `per_node`.
        # Asserted here rather than assumed: the outer refusal is all a
        # caller ever sees from `log_space`, so a per-node reason that
        # survived only inside the message would be a reason nobody can act
        # on -- the exact shape G11 removes.
        assert caught.value.reason == "no_node_qualifies"
        assert caught.value.per_node == {"d": "noise_additive"}

    def test_a_fractional_level_above_the_threshold_is_refused_by_name(self):
        with pytest.raises(NotLogLinear) as caught:
            log_space(_graph("multiplicative", fractional=0.3))
        message = str(caught.value)
        assert str(FIRST_ORDER_MAX_FRACTIONAL) in message
        assert "LogNormal" in message  # the remedy: the exact route has no threshold
        # The MEASURED level travels with the refusal. Without it a caller
        # cannot tell "just over the line, tighten the noise declaration"
        # from "nowhere near, this model is not multiplicative at all", and
        # the two call for opposite responses.
        assert caught.value.reason == "no_node_qualifies"
        assert caught.value.per_node == {"d": "fractional_too_large"}

    def test_the_lognormal_route_has_no_threshold(self):
        """The same fractional level that refused above is accepted here --
        which is the whole reason to declare log-Gaussian noise as such."""
        ls = log_space(_graph("lognormal", fractional=0.3))
        assert ls.kind == {"d": "lognormal"}

    def test_non_positive_data_on_a_multiplicative_node_is_refused(self):
        observed = _observed("multiplicative").at[3].set(-1.0)
        with pytest.raises(NotLogLinear) as caught:
            log_space(_graph("multiplicative", observed=observed))
        assert "non-positive" in str(caught.value)
        # Distinguished from `prediction_not_positive`, which reads much the
        # same in prose and blames the opposite thing: there the model is
        # wrong, here the data is. This one escapes un-wrapped, because the
        # data check runs AFTER the scenario was read successfully.
        assert caught.value.reason == "data_not_positive"
        assert caught.value.node == "d"

    def test_a_mixed_graph_transforms_per_node_and_records_the_skip(self):
        """One additive and one multiplicative observation: the transform
        keeps the additive node untouched and says so, rather than refusing
        the log route for the latent that never reaches it."""

        def model(d_cal, d_sky):
            c = sample("coef", lambda: dist.Normal(100.0, 30.0))
            observe("d_cal", lambda v: dist.Normal(v, 2.0), c, obs=d_cal)
            lam = sample("log_gain", lambda: dist.Normal(0.0, PRIOR_STD))
            mu = det("mu", lambda l: jnp.exp(l) * SKY, lam)
            observe("d_sky", lambda m: dist.Normal(m, F * m), mu, obs=d_sky)

        graph = trace(
            model,
            100.0 + 2.0 * jax.random.normal(jax.random.key(1), ()),
            _observed("multiplicative"),
        )
        ls = log_space(graph)
        assert ls.kind == {"d_sky": "multiplicative"}
        assert set(ls.skipped) == {"d_cal"}
        assert "CONSTANT" in ls.skipped["d_cal"]


class TestTheTransformedGraph:
    def test_the_offset_is_log_of_the_sky(self):
        """The claim that makes a SUMMED sky no obstacle to a log-linear gain."""
        block, _ = log_linear_operator(_graph("multiplicative"), ("log_gain",))
        assert jnp.allclose(block.offset["d"], jnp.log(SKY), rtol=1e-5)

    def test_the_transformed_data_carries_the_first_order_shift(self):
        graph = _graph("multiplicative")
        ls = log_space(graph)
        observed = _observed("multiplicative")
        node = {n.name: n for n in ls.graph.nodes}["d"]
        assert jnp.allclose(node.observed, jnp.log(observed) + F**2 / 2, rtol=1e-6)

    def test_the_lognormal_data_carries_no_shift(self):
        graph = _graph("lognormal")
        ls = log_space(graph)
        node = {n.name: n for n in ls.graph.nodes}["d"]
        assert jnp.allclose(node.observed, jnp.log(_observed("lognormal")), rtol=1e-6)

    def test_the_transformed_noise_does_not_depend_on_the_prediction(self):
        """The structural win: nothing left for a reweighting loop to iterate."""
        ls = log_space(_graph("multiplicative"))
        node = {n.name: n for n in ls.graph.nodes}["d"]
        assert node.depends_on_prediction is False


class TestTheChecks:
    def test_the_log_gain_passes_where_the_linear_check_cannot(self):
        """exp(log_gain) is not affine, so this latent has ONLY the log route."""
        check_log_linearity(_graph("multiplicative"), ("log_gain",))

    def test_curvature_in_log_space_is_refused_under_the_blameless_class(self):
        """An additive receiver term AFTER the gain: positive everywhere, and
        log of it is genuinely not affine -- the affinity check must be what
        catches it, and it must surface as NotLogLinear so a dispatcher can
        read it as an ordinary 'no'."""

        def model(data):
            lam = sample("gain", lambda: dist.Normal(1.6, 0.2))
            mu = det("mu", lambda l: l * SKY + 500.0, lam)
            observe("d", lambda m: dist.Normal(m, F * m), mu, obs=data)

        data = (1.6 * SKY + 500.0) * (
            1.0 + F * jax.random.normal(jax.random.key(2), (N,))
        )
        with pytest.raises(NotLogLinear) as caught:
            check_log_linearity(trace(model, data), ("gain",))
        assert "not affine" in str(caught.value)

    def test_the_default_scales_stay_inside_the_exponentials_range(self):
        """The linear default's 1e3 entry is deliberately absent: pushed
        through exp it overflows and the check would measure the dtype."""
        assert max(LOG_DEFAULT_SCALES) <= 1.0
        assert min(LOG_DEFAULT_SCALES) < 1e-2


class TestTheSolve:
    def test_wiener_matches_the_closed_form(self):
        graph = _graph("multiplicative")
        block, ls = log_linear_operator(graph, ("log_gain",))
        precision = precision_at(ls.graph, {"log_gain": jnp.array(0.0)})
        estimate, _ = wiener_solve(block, precision=precision)
        y = {n.name: n for n in ls.graph.nodes}["d"].observed
        mean, _ = _posterior(y, F)
        assert jnp.allclose(estimate["log_gain"], mean, rtol=1e-4)

    def test_the_estimate_recovers_the_truth_within_the_noise(self):
        graph = _graph("multiplicative")
        block, ls = log_linear_operator(graph, ("log_gain",))
        precision = precision_at(ls.graph, {"log_gain": jnp.array(0.0)})
        estimate, _ = wiener_solve(block, precision=precision)
        _, sd = _posterior(
            {n.name: n for n in ls.graph.nodes}["d"].observed, F
        )
        assert abs(float(estimate["log_gain"]) - LOG_G) < 4.0 * sd

    def test_the_draws_have_the_posterior_moments(self):
        """The check that separates a constrained realization from
        mean-plus-arbitrary-noise: the SECOND moment against the closed form."""
        graph = _graph("multiplicative")
        block, ls = log_linear_operator(graph, ("log_gain",))
        precision = precision_at(ls.graph, {"log_gain": jnp.array(0.0)})
        estimate, _ = wiener_solve(block, precision=precision)
        keys = jax.random.split(jax.random.key(3), 2000)
        draws = jnp.stack(
            [
                gcr_sample(block, precision=precision, key=k)[0]["log_gain"]
                for k in keys[:200]
            ]
        )
        _, sd = _posterior(
            {n.name: n for n in ls.graph.nodes}["d"].observed, F
        )
        assert jnp.allclose(jnp.std(draws), sd, rtol=0.25)
        assert abs(float(jnp.mean(draws) - estimate["log_gain"])) < 4.0 * sd / 200**0.5


class TestTheArraySeam:
    def test_multiplicative_log_data_is_the_shift_and_the_sigma_together(self):
        observed = _observed("multiplicative")
        y, sigma = multiplicative_log_data(observed, F)
        assert jnp.allclose(y, jnp.log(observed) + F**2 / 2, rtol=1e-6)
        assert jnp.shape(sigma) == jnp.shape(y)
        assert jnp.all(sigma == jnp.asarray(F, dtype=sigma.dtype))


#: A block prior's centre and width, deliberately off the evaluation point and
#: deliberately NOT 1.0. At ``std=1`` the ``-log s`` term of a Gaussian is
#: exactly zero, so half of a dropped prior's evidence vanishes for free; this
#: repository has had defects hide behind that number before.
BLOCK_MEAN = -1.3
BLOCK_STD = 0.35
BLOCK_AT = 0.5


class _BlockPrior:
    """A joint prior over ``log_gain``, standing in for ``JeffreysPrior``.

    ``Graph`` checks a joint prior structurally -- it must answer ``over`` and
    ``log_density``, and its block must name latents -- so the transform's
    obligation to CARRY the field is testable without also dragging in the
    Fisher arithmetic of the real prior, its ``ImproperUniform`` requirement on
    covered latents, and the float64 those need. What is under test here is
    the graph rebuild, and a double keeps it that way.
    """

    over = ("log_gain",)

    def log_density(self, graph, values):
        return dist.Normal(BLOCK_MEAN, BLOCK_STD).log_prob(values["log_gain"])


def _graph_with_block_prior():
    """The multiplicative fixture, carrying a joint prior over its latent."""
    data = _observed("multiplicative")

    def model(observed):
        joint_prior(_BlockPrior())
        lam = sample("log_gain", lambda: dist.Normal(0.0, PRIOR_STD))
        mu = det("mu", lambda l: jnp.exp(l) * SKY, lam)
        observe("d", lambda m: dist.Normal(m, F * m), mu, obs=observed)

    return trace(model, data), data


def _gaussian_log_prob(value, loc, scale):
    """``log N(value; loc, scale)`` written out, so the expectation is
    arithmetic this file did rather than arithmetic it asked the package for."""
    z = (value - loc) / scale
    return -0.5 * z**2 - jnp.log(scale) - 0.5 * math.log(2.0 * math.pi)


class TestTheJointPriorSurvivesTheTransform:
    """``log_space`` rebuilds the graph, and a rebuild may not lose a field.

    ``Graph.joint_prior`` defaults to ``None``, so ``Graph(nodes=..., plates=...)``
    is a legal call that silently drops it -- there is no construction-time
    complaint, because ``__check_init__`` only inspects the field when it is
    NOT ``None``. The loss is then invisible where anyone would look for it:
    a block prior shifts and re-curves the density, and every consumer of
    ``LogSpace`` in this file reports a MEAN and a WIDTH. So these assertions
    are on the absolute density, which is the only quantity that moves.
    """

    def test_the_transformed_graph_still_carries_the_declared_prior(self):
        graph, _ = _graph_with_block_prior()
        assert graph.joint_prior is not None
        ls = log_space(graph)
        assert ls.graph.joint_prior is graph.joint_prior

    def test_the_log_joint_of_the_transformed_graph_includes_the_prior_term(self):
        """The ABSOLUTE density, against a closed form written out here.

        Not the difference between two graphs, and not the posterior's shape:
        the whole number, so that a prior which failed to survive is a term
        missing from a value this test states in full. ``log_gain`` sits
        ``(0.5 + 1.3) / 0.35 = 5.14`` widths off the block prior's centre, so
        the term is about -13.1 nats against a total near -524 -- far outside
        any tolerance, and far outside the ~0.006 nats that separate this
        posterior's mean from the one without a prior at all.
        """
        graph, data = _graph_with_block_prior()
        ls = log_space(graph)
        at = {"log_gain": jnp.asarray(BLOCK_AT)}

        # Written from the model, not read back off the transform: the
        # log-space likelihood is Normal(log mu, F) at the shifted data, the
        # latent keeps its own Normal(0, PRIOR_STD), and the block prior is
        # the third term -- the one at issue.
        latent = _gaussian_log_prob(jnp.asarray(BLOCK_AT), 0.0, PRIOR_STD)
        y = jnp.log(data) + F**2 / 2.0
        likelihood = jnp.sum(
            _gaussian_log_prob(y, jnp.log(jnp.exp(BLOCK_AT) * SKY), F)
        )
        prior = _gaussian_log_prob(jnp.asarray(BLOCK_AT), BLOCK_MEAN, BLOCK_STD)
        expected = float(latent + likelihood + prior)

        assert float(prior) < -13.0  # the term is large; it cannot round away
        assert float(log_joint(ls.graph, at)) == pytest.approx(expected, rel=1e-4)

    def test_a_graph_without_a_joint_prior_still_has_none(self):
        """The repair carries the field; it does not invent one.

        Stated because 'preserve it' and 'default it' are one edit apart, and
        every other graph in this file predates the field.
        """
        ls = log_space(_graph("multiplicative"))
        assert ls.graph.joint_prior is None
