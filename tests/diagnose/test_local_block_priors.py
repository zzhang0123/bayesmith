"""G15 -- a local block that also carries the declared priors.

The gap, in one sentence: a NONLINEAR model's posterior precision at a point
needs the Jacobian from ``local_block`` and the prior from
``unchecked_operator``, and neither constructor had both. The second
linearizes at the domain's ZERO -- correct for an affine map, which has one
tangent everywhere, and simply a different matrix for a power law.

The oracle here is numpy on the analytic Jacobian of a power law, which is
written out by hand in this file and shares nothing with the graph machinery.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import const, det, observe, sample, trace
from bayesmith.diagnose.local import local_block
from bayesmith.errors import NotGaussian
from bayesmith.exact.block import unchecked_operator
from bayesmith.exact.fisher import dense_operator, fisher_information
from bayesmith.exact.precision import diagonal_from

X = jnp.linspace(1.0, 4.0, 8)
SIGMA = 0.3
PRIOR_MEAN, PRIOR_STD = 1.5, 0.8
AT = 2.0


def power_law(*, amplitude=2.0, index=1.4):
    """``mu = a * x**index`` -- nonlinear in ``a`` only through ``index``,
    and deliberately NOT affine in ``index``.

    ``d mu / d index = a x**index log x``, which depends on ``index``: the
    tangent at the domain's zero and the tangent at ``index = 2`` are
    different matrices, which is the whole subject.
    """
    data = amplitude * X**index + SIGMA * jax.random.normal(
        jax.random.key(0), X.shape
    )

    def model():
        xs = const("X", X)
        beta = sample("beta", lambda: dist.Normal(PRIOR_MEAN, PRIOR_STD))
        mu = det("mu", lambda b, x: amplitude * x**b, beta, xs)
        observe("d", lambda m: dist.Normal(m, SIGMA), mu, obs=data)

    return trace(model), amplitude


def _precision():
    return diagonal_from({"d": jnp.full(X.shape, SIGMA)})


class TestTheThirdConstructor:
    def test_by_default_it_still_carries_no_prior(self):
        """The property the module docstring argues for, and every existing
        caller depends on. It is asserted first because the keyword had to be
        added without disturbing it."""
        graph, _ = power_law()
        block = local_block(graph, ("beta",), {"beta": jnp.asarray(AT)})
        assert block.prior_mean == {} and block.prior_std == {}

    def test_with_priors_it_carries_the_DECLARED_ones(self):
        graph, _ = power_law()
        block = local_block(
            graph, ("beta",), {"beta": jnp.asarray(AT)}, priors=True
        )
        assert float(block.prior_mean["beta"]) == pytest.approx(PRIOR_MEAN)
        assert float(block.prior_std["beta"]) == pytest.approx(PRIOR_STD)

    def test_the_jacobian_is_still_the_one_at_the_CALLERS_point(self):
        """Adding the prior must not move the linearization point. Against the
        analytic derivative ``a x**b log x`` at ``b = 2``, in numpy."""
        graph, amplitude = power_law()
        block = local_block(
            graph, ("beta",), {"beta": jnp.asarray(AT)}, priors=True
        )
        design = np.asarray(dense_operator(block)).reshape(-1)
        expected = amplitude * np.asarray(X) ** AT * np.log(np.asarray(X))
        assert np.allclose(design, expected, rtol=1e-4), (design, expected)

    def test_the_other_constructor_gives_a_DIFFERENT_matrix_here(self):
        """The measurement that makes G15 a gap rather than a convenience.

        ``unchecked_operator`` carries the prior and linearizes at the
        domain's zero, so on a power law its design is
        ``a x**0 log x = log x`` -- a different matrix from the one above, by
        a factor of ``x**2`` per row. If the two agreed there would have been
        nothing to add.
        """
        graph, amplitude = power_law()
        affine = unchecked_operator(graph, ("beta",), at={})
        theirs = np.asarray(dense_operator(affine)).reshape(-1)
        expected_at_zero = amplitude * np.asarray(X) ** 0.0 * np.log(np.asarray(X))
        assert np.allclose(theirs, expected_at_zero, rtol=1e-4)
        mine = np.asarray(
            dense_operator(
                local_block(graph, ("beta",), {"beta": jnp.asarray(AT)}, priors=True)
            )
        ).reshape(-1)
        # ... and they are not close, on every row but the one where x = 1.
        assert not np.allclose(theirs[1:], mine[1:], rtol=0.5)

    def test_the_posterior_precision_is_the_dense_one(self):
        """End to end: ``J^T N^-1 J + 1/prior_std**2`` at the caller's point,
        against numpy on the analytic Jacobian."""
        graph, amplitude = power_law()
        block = local_block(
            graph, ("beta",), {"beta": jnp.asarray(AT)}, priors=True
        )
        found = fisher_information(
            block,
            precision=_precision(),
            include_prior=True,
            depends_on_prediction=False,
        )
        jacobian = amplitude * np.asarray(X) ** AT * np.log(np.asarray(X))
        expected = jacobian @ jacobian / SIGMA**2 + 1.0 / PRIOR_STD**2
        assert found.kind == "posterior_precision"
        assert float(found.values[0, 0]) == pytest.approx(expected, rel=1e-4)

    def test_without_the_prior_it_is_the_likelihood_alone(self):
        """The anti-vacuity clause: the two must differ by exactly the prior's
        curvature and nothing else, or "include_prior worked" is untested."""
        graph, _ = power_law()
        block = local_block(
            graph, ("beta",), {"beta": jnp.asarray(AT)}, priors=True
        )
        common = {"precision": _precision(), "depends_on_prediction": False}
        with_prior = float(
            fisher_information(block, include_prior=True, **common).values[0, 0]
        )
        without = float(
            fisher_information(block, include_prior=False, **common).values[0, 0]
        )
        assert with_prior - without == pytest.approx(1.0 / PRIOR_STD**2, rel=1e-4)

    def test_a_non_gaussian_prior_is_refused_when_the_priors_are_asked_for(self):
        """The refusal comes from ``_env_before``'s own ``check_gaussian`` --
        one spelling, not a second. A prior with no quadratic form must not
        contribute a silent zero to a posterior precision."""
        data = jnp.zeros(X.shape)

        def model():
            xs = const("X", X)
            beta = sample("beta", lambda: dist.Uniform(0.5, 3.0))
            mu = det("mu", lambda b, x: 2.0 * x**b, beta, xs)
            observe("d", lambda m: dist.Normal(m, SIGMA), mu, obs=data)

        graph = trace(model)
        with pytest.raises(NotGaussian):
            local_block(graph, ("beta",), {"beta": jnp.asarray(AT)}, priors=True)

    def test_the_same_graph_is_NOT_refused_without_the_priors(self):
        """The other direction, and it is what keeps the refusal about the
        priors rather than about the model: a Uniform prior is fine for a rank
        or a Jacobian, which is why the default reads unchanged."""
        data = jnp.zeros(X.shape)

        def model():
            xs = const("X", X)
            beta = sample("beta", lambda: dist.Uniform(0.5, 3.0))
            mu = det("mu", lambda b, x: 2.0 * x**b, beta, xs)
            observe("d", lambda m: dist.Normal(m, SIGMA), mu, obs=data)

        graph = trace(model)
        block = local_block(graph, ("beta",), {"beta": jnp.asarray(AT)})
        assert block.prior_std == {}
        assert dense_operator(block).shape == (8, 1)
