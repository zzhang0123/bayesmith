"""P4's first half: exact marginalisation of declared discrete latents.

``Probabilistic.support`` has carried ``Discrete(n)`` since P1, with its own
docstring saying that nothing reads it and that it is "recorded here so the
declaration exists from the start". This is what reads it.

**The declaration is a claim, so the refusals matter as much as the sum.**
That docstring is explicit that ``support=None`` must be treated as ineligible
for any support-specific method rather than as a claim of continuity, so an
undeclared latent is not enumerated and not quietly assumed away -- it simply
has no value, and the graph's own refusal names it. Those tests are here for
the same reason the arithmetic ones are.

**Cost is part of the contract.** Enumeration is exponential by construction:
``n ** coordinates`` per site, multiplied across sites. That is not a defect to
be hidden, it is the price of an exact answer, and the only dishonest thing to
do with it is to start the sum and hope. So the cost is computable before any
work happens and is refused past a budget, with the count in the message.
"""

import itertools
import math

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from jax.scipy.special import logsumexp

from bayesmith.exact.discrete import (
    ENUMERATION_BUDGET,
    discrete_latents,
    enumeration_states,
    marginal_log_likelihood,
    posterior_marginals,
)
from bayesmith.graph.evaluate import log_joint
from bayesmith.graph.nodes import Continuous, Discrete
from bayesmith.graph.trace import det, observe, plate, sample, trace

P, MU0, MU1, SIGMA = 0.3, -1.0, 2.0, 1.0


def _mixture(y):
    """z ~ Bernoulli(p);  y_i ~ N(mu_z, sigma).  One z, shared by all y."""

    def model():
        z = sample("z", lambda: dist.Bernoulli(probs=P), support=Discrete(2))
        mu = det("mu", lambda zz: jnp.where(zz == 1, MU1, MU0), z)
        observe("y", lambda m: dist.Normal(m, SIGMA), mu, obs=y)

    return trace(model)


def _closed_form(y):
    """The two-term sum, written out rather than enumerated."""
    terms = []
    for state, prior in ((0, 1.0 - P), (1, P)):
        mu = MU1 if state == 1 else MU0
        terms.append(
            math.log(prior) + float(jnp.sum(dist.Normal(mu, SIGMA).log_prob(y)))
        )
    return float(logsumexp(jnp.asarray(terms)))


class TestTheSumIsRight:
    def test_the_marginal_matches_the_closed_form(self):
        """An independent oracle: the two terms written out by hand, not
        enumerated by the thing under test."""
        y = jnp.array([1.5, 2.2, -0.3])
        got = float(marginal_log_likelihood(_mixture(y)))
        assert got == pytest.approx(_closed_form(y), rel=1e-6)

    def test_the_posterior_marginal_is_bayes_rule(self):
        y = jnp.array([1.8])
        lo = math.log(1 - P) + float(dist.Normal(MU0, SIGMA).log_prob(y).sum())
        hi = math.log(P) + float(dist.Normal(MU1, SIGMA).log_prob(y).sum())
        total = float(logsumexp(jnp.asarray([lo, hi])))
        got = posterior_marginals(_mixture(y))["z"]
        assert float(got[0]) == pytest.approx(math.exp(lo - total), rel=1e-6)
        assert float(got[1]) == pytest.approx(math.exp(hi - total), rel=1e-6)

    def test_the_marginals_sum_to_one(self):
        got = posterior_marginals(_mixture(jnp.array([0.4, 1.1])))["z"]
        assert float(jnp.sum(got)) == pytest.approx(1.0, rel=1e-6)

    def test_it_agrees_with_a_brute_force_sum_over_two_sites(self):
        """Two discrete latents, so the assignment is a PRODUCT and the order
        of the loop could plausibly be wrong. The oracle enumerates in the
        test, calling only ``log_joint`` -- the primitive being marginalised,
        not the marginalisation."""

        def model():
            a = sample("a", lambda: dist.Bernoulli(probs=0.4), support=Discrete(2))
            b = sample(
                "b", lambda: dist.Categorical(probs=jnp.array([0.2, 0.3, 0.5])),
                support=Discrete(3),
            )
            mu = det("mu", lambda x, w: 1.0 * x + 2.0 * w, a, b)
            observe("y", lambda m: dist.Normal(m, 1.0), mu, obs=jnp.array([2.4]))

        graph = trace(model)
        terms = [
            log_joint(graph, {"a": jnp.asarray(i), "b": jnp.asarray(j)})
            for i, j in itertools.product(range(2), range(3))
        ]
        assert float(marginal_log_likelihood(graph)) == pytest.approx(
            float(logsumexp(jnp.asarray(terms))), rel=1e-6
        )

    def test_a_graph_with_no_discrete_latents_is_just_the_joint(self):
        """The empty product is one term, not zero. A sum over no assignments
        that returned -inf would be an easy and silent way to be wrong."""

        def model():
            x = sample("x", lambda: dist.Normal(0.0, 1.0), support=Continuous())
            observe("y", lambda m: dist.Normal(m, 1.0), x, obs=jnp.array([0.5]))

        graph = trace(model)
        at = {"x": jnp.asarray(0.25)}
        assert float(marginal_log_likelihood(graph, at)) == pytest.approx(
            float(log_joint(graph, at)), rel=1e-9
        )


class TestItReadsTheDeclarationAndNothingElse:
    def test_only_declared_discrete_latents_are_enumerated(self):
        def model():
            x = sample("x", lambda: dist.Normal(0.0, 1.0), support=Continuous())
            z = sample("z", lambda: dist.Bernoulli(probs=0.5), support=Discrete(2))
            mu = det("mu", lambda a, b: a + b, x, z)
            observe("y", lambda m: dist.Normal(m, 1.0), mu, obs=jnp.array([1.0]))

        assert discrete_latents(trace(model)) == ("z",)

    def test_an_undeclared_latent_is_not_treated_as_continuous_nor_discrete(self):
        """``support=None`` is ineligible, by the declaration's own docstring.

        So it is neither enumerated nor assumed away: it stays a latent
        needing a value, and the graph's existing refusal names it. Asserted
        because the tempting shortcut -- "no support means continuous" -- is
        exactly the unverified claim the field was written to avoid.
        """

        def model():
            u = sample("u", lambda: dist.Bernoulli(probs=0.5))  # no support
            observe("y", lambda m: dist.Normal(m, 1.0), u, obs=jnp.array([1.0]))

        graph = trace(model)
        assert discrete_latents(graph) == ()
        with pytest.raises(Exception, match="u"):
            marginal_log_likelihood(graph)

    def test_a_continuous_latent_is_conditioned_on_rather_than_marginalised(self):
        """Its value must be supplied, and the answer must depend on it --
        a marginal that ignored the conditioning value would be constant."""

        def model():
            x = sample("x", lambda: dist.Normal(0.0, 1.0), support=Continuous())
            z = sample("z", lambda: dist.Bernoulli(probs=0.5), support=Discrete(2))
            mu = det("mu", lambda a, b: a + b, x, z)
            observe("y", lambda m: dist.Normal(m, 1.0), mu, obs=jnp.array([1.0]))

        graph = trace(model)
        near = float(marginal_log_likelihood(graph, {"x": jnp.asarray(1.0)}))
        far = float(marginal_log_likelihood(graph, {"x": jnp.asarray(-4.0)}))
        assert near > far

    def test_an_observed_discrete_node_is_not_enumerated(self):
        """It has data. Summing over states it does not occupy would replace
        the likelihood with something else entirely."""

        def model():
            p = sample("p", lambda: dist.Beta(2.0, 2.0), support=Continuous())
            observe(
                "k", lambda q: dist.Bernoulli(probs=q), p,
                obs=jnp.asarray(1), support=Discrete(2),
            )

        graph = trace(model)
        assert discrete_latents(graph) == ()
        at = {"p": jnp.asarray(0.6)}
        assert float(marginal_log_likelihood(graph, at)) == pytest.approx(
            float(log_joint(graph, at)), rel=1e-9
        )


class TestTheCostIsKnownBeforeTheWork:
    def test_the_state_count_is_the_product_over_sites(self):
        def model():
            a = sample("a", lambda: dist.Bernoulli(probs=0.5), support=Discrete(2))
            b = sample(
                "b", lambda: dist.Categorical(probs=jnp.ones(3) / 3),
                support=Discrete(3),
            )
            mu = det("mu", lambda x, w: x + w, a, b)
            observe("y", lambda m: dist.Normal(m, 1.0), mu, obs=jnp.array([1.0]))

        assert enumeration_states(trace(model)) == 6

    def test_a_plated_site_costs_n_to_the_plate_size(self):
        """Each coordinate is its own latent. Counting a plated site as ``n``
        would understate a 3-state site over 4 draws by a factor of 27, and
        the budget that guards the run is computed from this number."""

        def model():
            i = plate("i", 4)
            z = sample(
                "z", lambda: dist.Categorical(probs=jnp.ones(3) / 3),
                support=Discrete(3), plate=i,
            )
            observe(
                "y", lambda s: dist.Normal(1.0 * s, 1.0), z,
                obs=jnp.zeros(4), plate=i,
            )

        graph = trace(model)
        assert enumeration_states(graph) == 3**4

    def test_no_discrete_latents_is_one_state_and_not_zero(self):
        def model():
            x = sample("x", lambda: dist.Normal(0.0, 1.0), support=Continuous())
            observe("y", lambda m: dist.Normal(m, 1.0), x, obs=jnp.array([1.0]))

        assert enumeration_states(trace(model)) == 1

    def test_it_refuses_past_the_budget_and_says_what_it_would_have_cost(self):
        """Refuse before the sum, not during it. A caller who sees the number
        can raise the budget deliberately; one who sees a hung process cannot
        tell this from a bug."""

        def model():
            i = plate("i", 12)
            z = sample(
                "z", lambda: dist.Categorical(probs=jnp.ones(4) / 4),
                support=Discrete(4), plate=i,
            )
            observe(
                "y", lambda s: dist.Normal(1.0 * s, 1.0), z,
                obs=jnp.zeros(12), plate=i,
            )

        graph = trace(model)
        assert enumeration_states(graph) == 4**12
        with pytest.raises(ValueError, match=r"16777216"):
            marginal_log_likelihood(graph)

    def test_the_budget_can_be_raised_by_the_caller(self):
        def model():
            i = plate("i", 3)
            z = sample(
                "z", lambda: dist.Categorical(probs=jnp.ones(2) / 2),
                support=Discrete(2), plate=i,
            )
            observe(
                "y", lambda s: dist.Normal(1.0 * s, 1.0), z,
                obs=jnp.zeros(3), plate=i,
            )

        graph = trace(model)
        with pytest.raises(ValueError):
            marginal_log_likelihood(graph, budget=4)
        assert jnp.isfinite(marginal_log_likelihood(graph, budget=8))

    def test_the_default_budget_is_a_number_and_not_infinity(self):
        """An unbounded default would make the refusal above unreachable in
        practice, which is the same as not having it."""
        assert math.isfinite(ENUMERATION_BUDGET) and ENUMERATION_BUDGET > 1


@pytest.mark.parametrize("event", [False, True])
def test_vector_coordinates_share_one_layout_for_count_evidence_and_marginals(event):
    probs = jnp.array([0.2, 0.8])

    def model():
        sample("z", lambda: dist.Bernoulli(probs).to_event(int(event)), support=Discrete(2))

    graph = trace(model)
    assert enumeration_states(graph) == 4
    with pytest.raises(ValueError, match="4 joint assignments"):
        marginal_log_likelihood(graph, budget=3)
    assert float(marginal_log_likelihood(graph)) == pytest.approx(0, abs=2e-7)
    assert jnp.allclose(posterior_marginals(graph)["z"], jnp.stack((1 - probs, probs), axis=-1))


def test_independent_vector_events_replicated_in_a_plate():
    probs = jnp.array([0.2, 0.8])

    def model():
        i = plate("i", 3)
        sample("z", lambda: dist.Bernoulli(probs).to_event(1), support=Discrete(2), plate=i)

    graph = trace(model)
    assert enumeration_states(graph) == 2**6
    posterior = posterior_marginals(graph)["z"]
    assert posterior.shape == (3, 2, 2)
    assert jnp.allclose(posterior[..., 1], probs)


def test_discrete_children_inherit_discrete_and_conditioned_parent_shapes():
    probs = jnp.array([0.2, 0.8])

    def model():
        p = sample("p", lambda: dist.Uniform(jnp.zeros(2), 1).to_event(1))
        z = sample("z", lambda q: dist.Bernoulli(q), p, support=Discrete(2))
        q = det("q", lambda s: 0.25 + 0.5 * s, z)
        sample("k", lambda r: dist.Bernoulli(r).to_event(1), q, support=Discrete(2))

    graph = trace(model)
    assert enumeration_states(graph) == 16
    posterior = posterior_marginals(graph, {"p": probs})
    assert jnp.allclose(posterior["z"][..., 1], probs)
    assert jnp.allclose(posterior["k"][..., 1], 0.25 + 0.5 * probs)


def test_a_jointly_constrained_vector_support_is_refused_before_enumeration():
    def model():
        sample("z", lambda: dist.Multinomial(2, probs=jnp.array([0.2, 0.8])), support=Discrete(3))

    graph = trace(model)
    with pytest.raises(ValueError, match=r"z.*(Cartesian|coordinate)"):
        marginal_log_likelihood(graph)


@pytest.mark.parametrize("event_dims", [0, 1])
def test_enumeration_rejects_state_labels_outside_the_declared_coordinates(event_dims):
    def model():
        sample(
            "z",
            lambda: dist.DiscreteUniform(3, 4).expand((2,)).to_event(event_dims),
            support=Discrete(2),
        )

    graph = trace(model)
    with pytest.raises(ValueError, match="z.*state"):
        enumeration_states(graph)


def test_plated_distribution_batch_axis_aligns_with_the_plate():
    probs = jnp.array([[0.15, 0.4], [0.5, 0.75]])

    def model():
        i = plate("i", 2)
        sample("z", lambda: dist.Bernoulli(probs).to_event(1), support=Discrete(2), plate=i)

    graph = trace(model)
    assert enumeration_states(graph) == 16
    assert jnp.allclose(posterior_marginals(graph)["z"][..., 1], probs)


def test_conditioned_value_shapes_are_used_by_the_budget_and_the_sum():
    probs = jnp.array([0.2, 0.8])

    def model():
        p = sample("p", lambda: dist.Uniform(0, 1))
        sample("z", lambda q: dist.Bernoulli(q), p, support=Discrete(2))

    graph = trace(model)
    assert enumeration_states(graph) == 2
    assert enumeration_states(graph, {"p": probs}) == 4
    with pytest.raises(ValueError, match="4 joint assignments"):
        marginal_log_likelihood(graph, {"p": probs}, budget=3)
    assert jnp.allclose(posterior_marginals(graph, {"p": probs})["z"][..., 1], probs)


def test_fixed_binomial_bounds_are_concrete_during_shape_inference():
    def model():
        p = sample("p", lambda: dist.Uniform(0., 1.))
        sample("z", lambda prob: dist.Binomial(jnp.array([2, 2]), probs=prob).to_event(1),
               p, support=Discrete(3))

    graph = trace(model)
    assert enumeration_states(graph) == 9
    result = posterior_marginals(graph, {"p": jnp.array(0.25)})
    expected = jnp.array([0.75**2, 2 * 0.25 * 0.75, 0.25**2])
    assert jnp.allclose(result["z"], jnp.broadcast_to(expected, (2, 3)))


def test_enumeration_budget_needs_no_draw_from_an_improper_conditioning_prior():
    def model():
        x = sample("x", lambda: dist.ImproperUniform(dist.constraints.real, (), (2,)))
        sample("z", lambda v: dist.Bernoulli(logits=v), x, support=Discrete(2))

    graph = trace(model)
    assert enumeration_states(graph) == 4
    assert float(marginal_log_likelihood(graph, {"x": jnp.zeros(2)})) == pytest.approx(0.)


def test_large_vector_still_reports_the_node_and_budget():
    def model():
        sample("z", lambda: dist.Bernoulli(0.3).expand((15000,)).to_event(1),
               support=Discrete(2))

    with pytest.raises(ValueError, match=r"z.*2\*\*15000.*budget of 3"):
        marginal_log_likelihood(trace(model), budget=3)


@pytest.mark.parametrize("width", [2, 3])
def test_mapped_discrete_batch_and_plate_are_separate_coordinates(width):
    import jax
    from numpyro.infer.util import log_density

    from bayesmith.bridge.numpyro_bridge import to_numpyro

    def model():
        group = plate("group", 2)
        x = sample("x", lambda: dist.Normal(0., 1.), plate=group)
        sample("z", lambda v: dist.Bernoulli(logits=jnp.arange(width) + v), x,
               support=Discrete(2), plate=group)

    graph = trace(model)
    x = jnp.array([-2., 0.])
    logits = jnp.arange(width)[:, None] + x
    log_prior = dist.Normal(0., 1.).log_prob(x).sum()
    states = 2 ** (width * len(x))
    assert enumeration_states(graph, {"x": x}) == states
    with pytest.raises(ValueError, match=f"{states} joint assignments"):
        marginal_log_likelihood(graph, {"x": x}, budget=states - 1)
    assert marginal_log_likelihood(graph, {"x": x}) == pytest.approx(float(log_prior), abs=1e-6)
    marginals = posterior_marginals(graph, {"x": x})["z"]
    assert marginals.shape == (width, len(x), 2)
    assert jnp.allclose(marginals[..., 1], jax.nn.sigmoid(logits))

    z = jnp.arange(width * len(x)).reshape(width, len(x)) % 2
    log_mass = log_prior + dist.Bernoulli(logits=logits).log_prob(z).sum()
    assert log_joint(graph, {"x": x, "z": z}) == pytest.approx(float(log_mass))
    bridge_mass, _ = log_density(to_numpyro(graph), (), {}, {"x": x, "z": z})
    assert bridge_mass == pytest.approx(float(log_mass))
