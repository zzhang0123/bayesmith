"""R4 Task 9 -- the evidence of an all-discrete graph, and one naming trap.

An all-discrete model's evidence is a finite sum, so it needs no quadrature and
no square-root form: ``log Z = log sum_z p(z) p(y | z)`` over every assignment.
``exact/discrete.py`` already computes a logsumexp over ``log_joint``, and the
whole of this task is the distance between that quantity and this one.

**They are not the same quantity, and the function's name invites reading them
as one.** ``marginal_log_likelihood`` sums ``log_joint``, which includes every
CONTINUOUS latent's prior density, so what it returns is

    log sum_z p(theta) p(z) p(y | theta, z)  =  log p(theta) + log p(y | theta)

-- the joint ``log p(theta, y)``, not ``log p(y)``. The two coincide only when
there is no continuous latent to carry a prior. That is exactly the conflation
§1.4 invariant 3 forbids ("prior, likelihood, marginal likelihood term,
evidence and posterior must not be mixed"), and the name is one word away from
committing it, so it is pinned here rather than left to the next reader.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import const, det, observe, sample, trace
from bayesmith.exact.discrete import marginal_log_likelihood
from bayesmith.graph.evaluate import log_joint
from bayesmith.graph.nodes import Discrete

OBSERVED = (1.9, 0.4, 2.6)
SIGMA = 0.6
MEANS = (0.0, 1.0, 2.5)
WEIGHTS = (0.2, 0.5, 0.3)


def _all_discrete():
    """One categorical latent per observation, and nothing continuous."""
    data = jnp.asarray(OBSERVED)
    weights = jnp.asarray(WEIGHTS)
    means = jnp.asarray(MEANS)

    def model():
        picks = [
            sample(
                f"z{i}",
                lambda: dist.Categorical(probs=weights),
                support=Discrete(n=len(MEANS)),
            )
            for i in range(len(OBSERVED))
        ]
        m = const("means", means)
        mu = det(
            "mu",
            lambda m_, *z_: jnp.stack([m_[z] for z in z_]),
            m,
            *picks,
        )
        observe("d", lambda v: dist.Normal(v, SIGMA).to_event(1), mu, obs=data)

    return trace(model)


def _hand_summed_evidence():
    """``log sum_z p(z) p(y|z)``, by enumerating in Python.

    Shares nothing with the implementation: no logsumexp, no graph, no
    ``log_joint`` -- a loop over the product of the three supports, adding
    plain probabilities and taking one log at the end.
    """
    total = 0.0
    for assignment in itertools.product(range(len(MEANS)), repeat=len(OBSERVED)):
        prior = 1.0
        likelihood = 1.0
        for observation, pick in zip(OBSERVED, assignment, strict=True):
            prior *= WEIGHTS[pick]
            residual = observation - MEANS[pick]
            likelihood *= float(
                np.exp(-0.5 * (residual / SIGMA) ** 2)
                / (SIGMA * np.sqrt(2.0 * np.pi))
            )
        total += prior * likelihood
    return float(np.log(total))


def test_an_all_discrete_graph_has_an_evidence_that_is_a_finite_sum():
    with jax.enable_x64(True):
        found = float(marginal_log_likelihood(_all_discrete()))
        assert found == pytest.approx(_hand_summed_evidence(), abs=1e-9)


def test_the_enumeration_is_the_evidence_only_with_nothing_continuous():
    """The trap the function's name sets, pinned so it cannot be re-entered.

    Add ONE continuous latent with a prior and the same call returns
    ``log p(theta, y)``: the evidence plus that prior's log density at the
    value it was conditioned at. The gap is measured here rather than argued,
    and it is exactly the prior term.
    """
    with jax.enable_x64(True):
        data = jnp.asarray(OBSERVED)
        weights = jnp.asarray(WEIGHTS)
        means = jnp.asarray(MEANS)
        theta_at = 0.35
        prior_loc, prior_scale = 0.0, 1.4

        def model():
            theta = sample("theta", lambda: dist.Normal(prior_loc, prior_scale))
            picks = [
                sample(
                    f"z{i}",
                    lambda: dist.Categorical(probs=weights),
                    support=Discrete(n=len(MEANS)),
                )
                for i in range(len(OBSERVED))
            ]
            m = const("means", means)
            mu = det(
                "mu",
                lambda m_, t_, *z_: jnp.stack([m_[z] for z in z_]) + t_,
                m,
                theta,
                *picks,
            )
            observe("d", lambda v: dist.Normal(v, SIGMA).to_event(1), mu, obs=data)

        graph = trace(model)
        summed = float(
            marginal_log_likelihood(graph, {"theta": jnp.asarray(theta_at)})
        )

        conditional = 0.0
        for assignment in itertools.product(range(len(MEANS)), repeat=len(OBSERVED)):
            prior = 1.0
            likelihood = 1.0
            for observation, pick in zip(OBSERVED, assignment, strict=True):
                prior *= WEIGHTS[pick]
                residual = observation - (MEANS[pick] + theta_at)
                likelihood *= float(
                    np.exp(-0.5 * (residual / SIGMA) ** 2)
                    / (SIGMA * np.sqrt(2.0 * np.pi))
                )
            conditional += prior * likelihood
        log_p_y_given_theta = float(np.log(conditional))
        log_p_theta = float(
            -0.5 * ((theta_at - prior_loc) / prior_scale) ** 2
            - np.log(prior_scale)
            - 0.5 * np.log(2.0 * np.pi)
        )

        assert summed == pytest.approx(log_p_y_given_theta + log_p_theta, abs=1e-9)
        assert summed != pytest.approx(log_p_y_given_theta, abs=1e-3), (
            "the fixture must have a prior term big enough to tell the two "
            "quantities apart, or this test cannot fail"
        )
        assert abs(log_p_theta) > 0.5


def test_the_enumeration_agrees_with_log_joint_at_every_assignment():
    """The enumeration is a sum over ``log_joint`` and nothing else.

    Asserted because Task 9 does NOT open a discrete evidence route: it
    records what the existing enumeration is, so that a later release which
    does open one starts from a measured statement rather than from the
    function's name.
    """
    with jax.enable_x64(True):
        graph = _all_discrete()
        terms = []
        for assignment in itertools.product(range(len(MEANS)), repeat=len(OBSERVED)):
            values = {f"z{i}": jnp.asarray(pick) for i, pick in enumerate(assignment)}
            terms.append(float(log_joint(graph, values)))
        oracle = float(np.log(np.sum(np.exp(np.asarray(terms)))))
        assert float(marginal_log_likelihood(graph)) == pytest.approx(
            oracle, abs=1e-9
        )
