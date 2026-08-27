"""G7 -- carrying a parameter uncertainty onto a prediction, two ways.

``propagate_covariance`` linearises; ``push_forward`` samples. The pair is the
point: they agree exactly when the map is affine over the posterior's width
and diverge when it is not, and a reader arriving with "what is the error bar
on my prediction" is choosing between them whether or not they know it.

**The oracle for the linear one is a dense construction, not a second
implementation of the same formula.** On a graph whose prediction is
``X @ theta`` the delta method's answer IS ``sqrt(diag(X Sigma X^T))``, which
numpy can form from the design matrix directly, so the check runs against
linear algebra that shares no code with the function under test.

**The oracle for the Monte-Carlo one is the linear one**, on an affine model
where they must coincide -- plus a non-affine model where they must NOT, so
the agreement is a statement about the models and not a tautology about the
two functions being the same function.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import const, det, observe, sample, trace
from bayesmith.errors import StructureError
from bayesmith.exact.fisher import (
    FlatMatrix,
    fisher_information,
    parameter_covariance,
    propagate_covariance,
    push_forward,
)
from bayesmith.exact.linearity import linear_operator
from bayesmith.exact.precision import diagonal_from

N, SIGMA = 8, 0.4
#: The design's two columns are deliberately CORRELATED -- ``linspace(0.2,
#: 2.0)`` does not sum to zero, so ``X^T X`` is not diagonal and the parameter
#: covariance has real off-diagonal entries. With a symmetric ``linspace(-1,
#: 1)`` the columns are orthogonal, the covariance is nearly diagonal, and a
#: delta method that simply ignored parameter correlation would agree with the
#: correct one to well inside any tolerance. Measured: that fixture let the
#: "diagonal only" mutation survive the whole file.
X = jnp.stack([jnp.ones(N), jnp.linspace(0.2, 2.0, N)], axis=1)
TRUTH = jnp.array([2.0, -1.0])
#: Not zero, and that is the point. With ``Normal(0, 10)`` priors the declared
#: centre IS the origin, so `init_to_declared` returning zeros instead of the
#: declaration is indistinguishable from returning the declaration -- measured,
#: that mutation survived. A prior centre that is neither zero nor the truth
#: separates all three.
#: A tuple, converted inside the ``dist_fn``: a module-level ``jnp.array`` is
#: built at import, i.e. in float32, and a latent whose prior loc is float32
#: gives the block a float32 domain -- which then meets an x64 trace as a bare
#: "tangent aval float64 for primal aval float32" from inside JAX. The same
#: shape the joint-prior tests pay for, one layer down.
PRIOR_MEAN = (1.3, -0.7)
AT = {"theta": TRUTH}


def affine_graph(sigma=SIGMA):
    """``d ~ N(X theta, sigma)`` -- the prediction is linear in the latent."""
    data = X @ TRUTH

    def model():
        design = const("X", X)
        theta = sample("theta", lambda: dist.Normal(jnp.asarray(PRIOR_MEAN), 10.0))
        mu = det("mu", lambda a, t: a @ t, design, theta, linear_in=("theta",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def curved_graph():
    """``mu = exp(theta_0) + theta_1 x`` -- affine in one latent, not the other."""
    data = jnp.exp(TRUTH[0]) + TRUTH[1] * X[:, 1]

    def model():
        coordinate = const("x", X[:, 1])
        theta = sample("theta", lambda: dist.Normal(jnp.zeros(2), 1.0))
        mu = det("mu", lambda c, t: jnp.exp(t[0]) + t[1] * c, coordinate, theta)
        observe("d", lambda m: dist.Normal(m, SIGMA), mu, obs=data)

    return trace(model)


def covariance_of(graph, at=None):
    with jax.enable_x64(True):
        values = AT if at is None else at
        del values
        block = linear_operator(graph, ("theta",))
        return parameter_covariance(
            fisher_information(
                block,
                precision=diagonal_from({"d": jnp.full(N, SIGMA)}),
                depends_on_prediction=False,
            )
        )


# --------------------------------------------------------------------------
# The delta method against dense linear algebra
# --------------------------------------------------------------------------


def test_the_delta_method_matches_the_dense_construction():
    """``sqrt(diag(X Sigma X^T))`` -- numpy forms it from the design directly.

    Shares no code with the function under test: no ``jacfwd``, no graph, no
    ``FlatMatrix``. If the Jacobian were transposed, or the einsum contracted
    the wrong index, this is the assertion that notices.
    """
    with jax.enable_x64(True):
        graph = affine_graph()
        cov = covariance_of(graph)
        got = np.asarray(propagate_covariance(graph, cov, AT, node="mu"))
        design = np.asarray(X)
        expected = np.sqrt(np.diag(design @ np.asarray(cov.values) @ design.T))
        assert np.allclose(got, expected, rtol=1e-12)
        # And the fixture is not degenerate: the error bar has structure.
        assert got.max() / got.min() > 1.5
        # And the parameters really are correlated, so an implementation that
        # dropped the off-diagonal would be a different number rather than the
        # same one -- without this the assertion above holds for two reasons
        # and distinguishes neither.
        values = np.asarray(cov.values)
        diagonal_only = np.sqrt(np.diag(design @ np.diag(np.diag(values)) @ design.T))
        assert np.max(np.abs(diagonal_only / expected - 1.0)) > 0.05


def test_the_default_node_is_the_single_observed_one():
    """Defaulted only where the graph leaves no choice."""
    with jax.enable_x64(True):
        graph = affine_graph()
        cov = covariance_of(graph)
        assert np.allclose(
            np.asarray(propagate_covariance(graph, cov, AT)),
            np.asarray(propagate_covariance(graph, cov, AT, node="d")),
        )


# --------------------------------------------------------------------------
# The two methods against each other
# --------------------------------------------------------------------------


def test_the_two_methods_agree_on_an_affine_model():
    """A linear map carries a Gaussian to a Gaussian, so they must coincide.

    Many draws and a loose tolerance, because one side is Monte Carlo: the
    claim is that the two describe the same spread, not that a finite sample
    reproduces it to roundoff.
    """
    with jax.enable_x64(True):
        graph = affine_graph()
        cov = covariance_of(graph)
        chol = np.linalg.cholesky(np.asarray(cov.values))
        draws = np.asarray(TRUTH) + (
            np.random.default_rng(0).standard_normal((4000, 2)) @ chol.T
        )
        pushed = np.asarray(
            push_forward(graph, {"theta": jnp.asarray(draws)}, node="mu")
        )
        linear = np.asarray(propagate_covariance(graph, cov, AT, node="mu"))
        assert np.allclose(pushed.std(axis=0), linear, rtol=0.06)


def test_the_two_methods_disagree_where_the_map_curves():
    """The sibling: agreement above is about the MODEL, not about the functions.

    ``exp(theta_0)`` over a prior width of order one is not affine across the
    posterior, so the linearised spread is symmetric about the point and the
    pushed one is not. Without this, two functions that were secretly the same
    function would pass the test above.
    """
    with jax.enable_x64(True):
        graph = curved_graph()
        at = {"theta": jnp.array([1.0, -1.0])}
        wide = FlatMatrix(
            values=jnp.eye(2) * 0.25,
            names=("theta",),
            spans=((0, 2),),
            kind="covariance",
        )
        draws = np.asarray(at["theta"]) + 0.5 * np.random.default_rng(1).standard_normal(
            (4000, 2)
        )
        pushed = np.asarray(
            push_forward(graph, {"theta": jnp.asarray(draws)}, node="mu")
        )
        linear = np.asarray(propagate_covariance(graph, wide, at, node="mu"))
        # Same quantity, and they differ by more than Monte-Carlo error.
        assert np.max(np.abs(pushed.std(axis=0) / linear - 1.0)) > 0.1


def test_an_observed_node_pushes_its_prediction_not_its_data():
    """``evaluate`` gives a conditioned node its DATA, and data has no gradient.

    This is the defect the first draft of these functions had, found by the
    default-node test passing while comparing two zeros: propagating an
    observed node's VALUE reports a Jacobian of exactly zero and an error bar
    of exactly zero, on every entry, for any model -- the most confident
    possible wrong answer. So an observed node contributes its ``loc``, and
    that is what makes the default (`node=None` on a single-observed graph)
    mean what a caller reads it as.
    """
    with jax.enable_x64(True):
        graph = affine_graph()
        draws = jnp.asarray(
            np.asarray(TRUTH) + 0.1 * np.random.default_rng(2).standard_normal((64, 2))
        )
        at_d = np.asarray(push_forward(graph, {"theta": draws}, node="d"))
        at_mu = np.asarray(push_forward(graph, {"theta": draws}, node="mu"))
        assert np.allclose(at_d, at_mu)
        # And it moves: the data would not.
        assert at_d.std(axis=0).min() > 0.0
        assert np.allclose(
            np.asarray(graph.node("d").observed), np.asarray(X @ TRUTH)
        )


def test_predict_returns_the_data_for_a_conditioned_node():
    """Measured, because it is surprising and it is why ``push_forward`` exists.

    NumPyro's ``Predictive`` over a model whose ``obs=`` is baked in returns
    the observed node's data, identical in every draw. That is correct of
    NumPyro and is not what "posterior predictive" usually means -- so the
    unconditioned mode has to be reachable rather than assumed, and it is:
    calling the model with ``observed={}`` draws those sites instead.
    """
    from numpyro.infer import Predictive

    from bayesmith.bridge.numpyro_bridge import predict, to_numpyro

    with jax.enable_x64(True):
        graph = affine_graph()
        cov = covariance_of(graph)
        chol = np.linalg.cholesky(np.asarray(cov.values))
        draws = jnp.asarray(
            np.asarray(TRUTH)
            + np.random.default_rng(3).standard_normal((3000, 2)) @ chol.T
        )
        conditioned = np.asarray(predict(graph, {"theta": draws})["d"])
        # `ptp`, not `std`: every draw is BITWISE the datum, and a float32
        # `std` over 3000 identical values still reports ~5e-5 from its own
        # accumulation -- a number that reads as "nearly constant" when the
        # claim is "identical".
        assert np.all(np.ptp(conditioned, axis=0) == 0.0)

        quiet = np.asarray(push_forward(graph, {"theta": draws}, node="d")).std(axis=0)
        drawn = np.asarray(
            Predictive(to_numpyro(graph), posterior_samples={"theta": draws})(
                jax.random.key(0), observed={}
            )["d"]
        )
        assert np.allclose(drawn.std(axis=0), np.sqrt(quiet**2 + SIGMA**2), rtol=0.08)


# --------------------------------------------------------------------------
# What these refuse
# --------------------------------------------------------------------------


def test_a_precision_where_a_covariance_belongs_is_refused():
    """Same shape, wrong by the square of everything, nothing downstream notices."""
    with jax.enable_x64(True):
        graph = affine_graph()
        block = linear_operator(graph, ("theta",))
        fisher = fisher_information(
            block,
            precision=diagonal_from({"d": jnp.full(N, SIGMA)}),
            depends_on_prediction=False,
        )
        with pytest.raises(StructureError, match="not a covariance"):
            propagate_covariance(graph, fisher, AT, node="mu")


def test_a_covariance_over_a_different_shape_is_refused():
    """The gap rheplicant's ancestor documented and could not close.

    A pytree treedef over a dict encodes the KEY NAMES alone, so a covariance
    built for a length-2 latent and a graph whose latent is length 3 pass a
    structure check and give finite, wrong error bars. A ``FlatMatrix`` carries
    its spans, so the sizes are checked too.
    """
    with jax.enable_x64(True):
        graph = affine_graph()
        wrong = FlatMatrix(
            values=jnp.eye(3),
            names=("theta",),
            spans=((0, 3),),
            kind="covariance",
        )
        with pytest.raises(StructureError, match="3 entries but `at` has 2"):
            propagate_covariance(graph, wrong, AT, node="mu")


def test_a_covariance_over_a_latent_the_graph_does_not_declare_is_refused():
    with jax.enable_x64(True):
        graph = affine_graph()
        alien = FlatMatrix(
            values=jnp.eye(2), names=("phi",), spans=((0, 2),), kind="covariance"
        )
        with pytest.raises(StructureError, match="not a latent of this graph"):
            propagate_covariance(graph, alien, {"theta": TRUTH, "phi": TRUTH})


def test_an_ambiguous_node_is_refused_rather_than_guessed():
    """Two observed nodes, so there is no single prediction to default to."""

    def model():
        design = const("X", X)
        theta = sample("theta", lambda: dist.Normal(jnp.asarray(PRIOR_MEAN), 10.0))
        mu = det("mu", lambda a, t: a @ t, design, theta, linear_in=("theta",))
        observe("d1", lambda m: dist.Normal(m, SIGMA), mu, obs=X @ TRUTH)
        observe("d2", lambda m: dist.Normal(m, SIGMA), mu, obs=X @ TRUTH)

    with jax.enable_x64(True):
        graph = trace(model)
        cov = FlatMatrix(
            values=jnp.eye(2), names=("theta",), spans=((0, 2),), kind="covariance"
        )
        with pytest.raises(StructureError, match="needs `node=`"):
            propagate_covariance(graph, cov, AT)
        with pytest.raises(StructureError, match="needs `node=`"):
            push_forward(graph, {"theta": jnp.zeros((3, 2))})


def test_a_transposed_sample_stack_is_refused():
    """The leading axis IS the draw axis, and a square stack hides the swap."""
    with jax.enable_x64(True):
        graph = affine_graph()
        with pytest.raises(StructureError, match="LEADING axis"):
            push_forward(graph, {"theta": jnp.zeros((2, 5))}, node="mu")


def test_a_missing_latent_is_refused():
    with jax.enable_x64(True):
        graph = affine_graph()
        with pytest.raises(StructureError, match="missing latent"):
            push_forward(graph, {}, node="mu")


# --------------------------------------------------------------------------
# init_to_declared
# --------------------------------------------------------------------------


def test_init_to_declared_starts_at_the_declared_centres():
    """One statement of the prior mean, shared with the classifier.

    ``prior_environment`` is public precisely so the sampler starts where the
    partitioner looked; writing "the declared values" by hand at the call site
    is the second spelling that lets the two drift.
    """
    from bayesmith import init_to_declared
    from bayesmith.dispatch.classify import prior_environment

    graph = affine_graph()
    strategy = init_to_declared(graph)
    declared = prior_environment(graph)
    assert np.allclose(
        np.asarray(strategy.keywords["values"]["theta"]),
        np.asarray(declared["theta"]),
    )
    assert set(strategy.keywords["values"]) == set(graph.latents)
    # The declared centre is neither zero nor the truth, so "it returned the
    # declaration" is distinguishable from both of the plausible wrong answers.
    assert not np.allclose(np.asarray(declared["theta"]), 0.0)
    assert not np.allclose(np.asarray(declared["theta"]), np.asarray(TRUTH))


def test_init_to_declared_actually_starts_nuts_there():
    """Not a keyword that gets ignored: the chain begins at the declared point.

    A one-draw, zero-warmup chain, so the first sample is the initialisation.
    Without this the test above pins the strategy object's contents and says
    nothing about whether ``nuts`` reads it.
    """
    from bayesmith import init_to_declared
    from bayesmith.bridge.numpyro_bridge import nuts
    from bayesmith.dispatch.classify import prior_environment

    graph = affine_graph()
    samples = nuts(
        graph,
        jax.random.key(0),
        num_warmup=0,
        num_samples=1,
        nuts_options={"init_strategy": init_to_declared(graph)},
    )
    declared = np.asarray(prior_environment(graph)["theta"])
    assert np.allclose(np.asarray(samples["theta"])[0], declared, atol=1e-6)
