"""G9 in full -- the four faces beyond the minimal surface, and one refusal.

The minimal surface (``tests/exact/test_complex.py``) is the mean and draw
paths: ``wiener_solve`` and ``gcr_sample`` over a complex latent's real
degrees of freedom. G9 in full is the rest of the plan's list -- vmap, log
space, the Fisher's complex face -- plus the two items the ledger records
separately: ``diagnose``'s refusal, and ``exact.correct.log_weight``.

**Two of the four already worked and two did not, and that was measured
rather than assumed.** vmap and the log route were reachable and correct with
no change at all; what they lacked was anything asserting it, which is the
same as not having them. ``dense_operator`` and ``log_weight`` both failed
outright -- one with ``jacfwd requires real-valued inputs``, one inside
``jax.linearize`` -- and both failed for the same underlying reason: they
worked in the DOMAIN where the rest of the solver works in PARTS.

The oracle throughout is ``test_complex.py``'s ``dense_posterior()``, which
pushes real basis vectors through the block's own ``forward`` and forms the
4x4 normal equations in numpy. It shares no code with any solver.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as ndist
import pytest

from bayesmith import det, observe, sample, trace
from bayesmith.distributions import ComplexNormal
from bayesmith.exact.block import variance_parts
from bayesmith.exact.fisher import (
    _spans,
    dense_operator,
    fisher_information,
    parameter_covariance,
)
from bayesmith.exact.precision import diagonal_from
from bayesmith.exact.solve import gcr_sample, wiener_solve
from tests.exact.test_complex import (
    DATA,
    NOISE_STD,
    PRIOR_STD,
    complex_block,
    dense_posterior,
    mixed_block,
)


def _precision():
    return diagonal_from({"d": jnp.full(DATA.shape, NOISE_STD)})


# ------------------------------------------------- the Fisher's complex face --


class TestTheDenseRouteOverRealDegreesOfFreedom:
    """``dense_operator`` and everything on top of it used to raise
    ``jacfwd requires real-valued inputs``. The fix is the same one the
    iterative route made at G9's minimal surface: differentiate over
    ``(re, im)``, never over C, because the objective is not holomorphic and
    JAX's complex gradient is the CONJUGATE one."""

    def test_the_design_matrix_is_the_dense_reference_element_for_element(self):
        """The oracle pushes four real basis vectors through the block's own
        ``forward`` and shares nothing else. Equality rather than a tolerance:
        it is the same arithmetic in the same order."""
        _, _ = dense_posterior()  # (built for its side effects on nothing)
        design = dense_operator(complex_block())
        columns = []
        for index in range(4):
            parts = np.zeros(4)
            parts[index] = 1.0
            latent = jnp.asarray(parts[:2] + 1j * parts[2:], dtype=jnp.complex64)
            columns.append(
                np.asarray(complex_block().forward({"a": latent})["d"])
            )
        expected = np.stack(columns, axis=1)
        assert design.shape == (4, 4)
        assert np.allclose(np.asarray(design), expected, rtol=1e-6, atol=1e-7)

    def test_a_complex_latent_occupies_TWO_rows_per_entry(self):
        """The layout claim, stated where a caller reads it. Two complex
        entries are four real degrees of freedom, real half first."""
        spans, size = _spans(complex_block())
        assert (spans, size) == (((0, 4),), 4)

    def test_a_mixed_block_lays_the_real_member_after_the_complex_one(self):
        """The asymmetry `real_parts` keeps, in the flat spelling: a real
        member is NOT doubled, and a uniform rule that doubled everything
        would pass every all-complex test in this file."""
        spans, size = _spans(mixed_block())
        assert (spans, size) == (((0, 4), (4, 5)), 5)

    def test_the_fisher_is_the_dense_normal_matrix_without_the_prior(self):
        """``J^T N^-1 J`` over the real degrees of freedom, against numpy."""
        block = complex_block()
        found = fisher_information(
            block,
            precision=_precision(),
            include_prior=False,
            depends_on_prediction=False,
        )
        design = np.asarray(dense_operator(block), dtype=np.float64)
        expected = design.T @ (np.eye(4) / NOISE_STD**2) @ design
        assert found.kind == "fisher"
        # `atol` scaled to the matrix, because the two sides are not the same
        # dtype: `found.values` is float32 and `expected` is a float64 dense
        # assembly. numpy's default `atol=1e-8` is a float64-era constant, and
        # it was always too tight here -- float32's resolution at this matrix's
        # scale (entries run 0.16 to 7.08) is about 8e-7. It passed only while
        # the roundoff happened to land under it: on jax 0.11.1 / numpy 2.5.2
        # a structurally-zero off-diagonal came back as 2.68e-08 on one side
        # and exactly 0.0 on the other, and the comparison failed on arithmetic
        # noise rather than on anything about the Fisher.
        scale = float(np.abs(expected).max())
        assert np.allclose(
            np.asarray(found.values), expected, rtol=1e-5, atol=1e-5 * scale
        )

    def test_with_the_prior_it_inverts_to_the_dense_posterior_covariance(self):
        """End to end, and the strongest oracle here: the covariance a caller
        gets must be the one the dense reference computes by inverting the
        4x4 normal equations -- prior included, and each PART carrying
        ``prior_std**2``, which is the convention a factor of sqrt(2) would
        break.
        """
        block = complex_block()
        with jax.enable_x64(True):
            found = parameter_covariance(
                fisher_information(
                    block,
                    precision=_precision(),
                    include_prior=True,
                    depends_on_prediction=False,
                )
            )
        _, expected = dense_posterior()
        assert np.allclose(np.asarray(found.values), expected, rtol=1e-4)

    def test_it_is_not_the_prior_alone(self):
        """Anti-vacuity: with ``prior_std = 1`` the prior's own contribution is
        the identity, so a Fisher that had silently dropped the DATA would
        still invert to something plausible."""
        block = complex_block()
        found = fisher_information(
            block,
            precision=_precision(),
            include_prior=True,
            depends_on_prediction=False,
        )
        prior_only = np.eye(4) / PRIOR_STD**2
        assert not np.allclose(np.asarray(found.values), prior_only, rtol=1e-2)


# ---------------------------------------------------------------------- vmap --


class TestVmapOverAComplexDraw:
    """Already worked; nothing asserted it. ``n_draws > 1`` goes through
    ``jax.vmap`` over keys, and a complex latent has to survive the trace."""

    def test_the_stack_comes_back_complex_and_the_right_shape(self):
        block = complex_block()
        precision = _precision()

        def one(key):
            drawn, _ = gcr_sample(block, precision=precision, key=key)
            return drawn["a"]

        stack = jax.vmap(one)(jax.random.split(jax.random.key(0), 8))
        assert stack.shape == (8, 2)
        assert jnp.issubdtype(stack.dtype, jnp.complexfloating)

    def test_the_moments_match_the_dense_posterior(self):
        """Not just "it ran": the draws must be from the right Gaussian, and
        both halves of every entry are checked because a draw that dropped the
        imaginary part would still be complex, still be the right shape, and
        still have a plausible real part."""
        block = complex_block()
        precision = _precision()

        def one(key):
            drawn, _ = gcr_sample(block, precision=precision, key=key)
            return drawn["a"]

        stack = np.asarray(jax.vmap(one)(jax.random.split(jax.random.key(1), 4000)))
        parts = np.concatenate(
            [np.real(stack), np.imag(stack)], axis=1
        )  # (n, 4), same order as the reference
        mean, covariance = dense_posterior()
        error = parts.mean(axis=0) - mean
        sigma = np.sqrt(np.diag(covariance) / stack.shape[0])
        assert np.all(np.abs(error) < 4.0 * sigma), (error, sigma)
        assert np.allclose(np.cov(parts.T), covariance, atol=0.05)


# ----------------------------------------------------------------- log space --


def log_affine_complex_model(*, n=8, noise=0.05, seed=0):
    """``mu = exp(Re(D z))`` with multiplicative noise -- log-affine in a
    COMPLEX latent.

    ``log(mu) = Re(D z)`` is R-affine in ``z``, which is exactly the claim the
    log route probes rather than takes on declaration (there is deliberately
    no ``log_linear_in`` field). ``linear_in`` is NOT declared here, because
    ``mu`` itself is not affine in ``z`` -- only its log is, and saying
    otherwise would be a false declaration that the linear pass would catch.
    """
    design = jnp.array(
        [[0.30 + 0.10j], [0.10 - 0.20j], [0.25 + 0.05j], [-0.15 + 0.10j],
         [0.20 + 0.20j], [0.05 - 0.10j], [0.18 + 0.02j], [-0.08 - 0.12j]],
        dtype=jnp.complex64,
    )[:n]
    truth = jnp.array([1.0 + 0.5j], dtype=jnp.complex64)
    clean = jnp.exp(jnp.real(design @ truth))
    data = clean * jnp.exp(noise * jax.random.normal(jax.random.key(seed), (n,)))

    def model():
        z = sample("a", lambda: ComplexNormal(jnp.zeros(1, jnp.complex64), 2.0))
        mu = det("mu", lambda value: jnp.exp(jnp.real(design @ value)), z)
        observe(
            "d", lambda m: ndist.Normal(m, noise * m), mu,
            depends_on_prediction=True, obs=data,
        )

    return trace(model), truth


class TestTheLogRouteWithAComplexLatent:
    """Also already worked, and also unasserted. The transform, the probe and
    the sweep all had to survive a complex domain, and each is a separate
    place it could have failed."""

    def test_the_transform_reads_the_node_as_multiplicative(self):
        from bayesmith.exact.loglinear import log_space

        graph, _ = log_affine_complex_model()
        assert log_space(graph).kind == {"d": "multiplicative"}

    def test_the_probe_finds_log_of_the_prediction_affine_in_it(self):
        from bayesmith.exact.loglinear import check_log_linearity

        graph, _ = log_affine_complex_model()
        check_log_linearity(graph, ("a",), {})  # no refusal is the assertion

    def test_the_partition_gives_it_a_log_gcr_block(self):
        from bayesmith.dispatch.factor import factor_partition

        graph, _ = log_affine_complex_model()
        plan = factor_partition(graph)
        assert [(b.latents, b.method) for b in plan.blocks] == [(("a",), "log-gcr")]

    def test_the_sweep_recovers_the_truth_in_BOTH_halves(self):
        """Both halves named, because a route that dropped the imaginary part
        would recover the real one perfectly and look like a success."""
        from bayesmith.dispatch.factor import factor_partition, sample_factors

        graph, truth = log_affine_complex_model()
        plan = factor_partition(graph)
        drawn = sample_factors(
            graph, plan, jax.random.key(1), num_warmup=100, num_samples=200
        )
        found = complex(jnp.mean(drawn["a"]))
        assert found.real == pytest.approx(float(jnp.real(truth[0])), abs=0.35)
        assert found.imag == pytest.approx(float(jnp.imag(truth[0])), abs=0.35)
        assert abs(found.imag) > 0.2  # ... and it is not zero


# ------------------------------------------------------------- the SNIS weight --


class TestLogWeightOverAComplexBlock:
    """``log_weight`` used to raise inside ``jax.linearize``. The reason to
    split is not that it raised: ``x^T M x`` over C is not the quadratic form
    ``q`` was built from, so the day it stopped raising it would have been
    silently the wrong scalar."""

    def test_the_quadratic_term_is_the_dense_one(self):
        """The oracle is numpy on the dense normal matrix. ``log_weight``
        returns ``log_joint + quadratic``, so the quadratic is recovered by
        subtracting a ``log_joint`` this test computes itself."""
        from bayesmith.exact.correct import log_weight
        from bayesmith.graph.evaluate import log_joint
        from tests.exact.test_complex import sky_model

        graph = sky_model()
        from bayesmith.dispatch.classify import prior_environment
        from bayesmith.exact.gaussian import precision_at
        from bayesmith.exact.linearity import linear_operator

        centres = prior_environment(graph)
        at = {n: centres[n] for n in graph.latents}
        block = linear_operator(graph, ("alm",), at={})
        precision = precision_at(graph, at)
        mu, _ = wiener_solve(block, precision=precision)

        offset = jnp.asarray([0.15 - 0.05j, -0.1 + 0.2j], dtype=jnp.complex64)
        x = {"alm": mu["alm"] + offset}
        found = float(log_weight(graph, block, x, at=at, precision=precision, mu=mu))
        quadratic = found - float(log_joint(graph, {**at, **x}))

        _, covariance = dense_posterior()
        delta = np.concatenate([np.real(np.asarray(offset)), np.imag(np.asarray(offset))])
        expected = 0.5 * delta @ np.linalg.inv(covariance) @ delta
        assert quadratic == pytest.approx(expected, rel=2e-3)

    def test_the_imaginary_half_contributes(self):
        """Anti-vacuity: an implementation that summed only the real leaves
        would agree with the test above whenever the offset is real, and this
        file's offset is not -- but nothing SAYS so. A purely imaginary
        offset makes the point unambiguously."""
        from bayesmith.dispatch.classify import prior_environment
        from bayesmith.exact.correct import log_weight
        from bayesmith.exact.gaussian import precision_at
        from bayesmith.exact.linearity import linear_operator
        from bayesmith.graph.evaluate import log_joint
        from tests.exact.test_complex import sky_model

        graph = sky_model()
        centres = prior_environment(graph)
        at = {n: centres[n] for n in graph.latents}
        block = linear_operator(graph, ("alm",), at={})
        precision = precision_at(graph, at)
        mu, _ = wiener_solve(block, precision=precision)

        offset = jnp.asarray([0.0 + 0.3j, 0.0 - 0.4j], dtype=jnp.complex64)
        x = {"alm": mu["alm"] + offset}
        found = float(log_weight(graph, block, x, at=at, precision=precision, mu=mu))
        quadratic = found - float(log_joint(graph, {**at, **x}))
        assert quadratic > 0.05, quadratic

    def test_an_all_real_block_is_unchanged_by_the_split(self):
        """The regression the split had to not cause. For a real block
        ``split`` is the identity and the leaves are the same arrays in the
        same order, so the arithmetic must be bitwise what it was -- computed
        here in the OLD spelling, in this file, so the claim is checked and
        not merely asserted in a docstring."""
        from bayesmith.dispatch.classify import prior_environment
        from bayesmith.exact.correct import log_weight
        from bayesmith.exact.gaussian import precision_at
        from bayesmith.exact.linearity import linear_operator
        from bayesmith.exact.solve import normal_operator
        from bayesmith.graph.evaluate import log_joint
        from tests.exact.models import two_linear_latents

        graph = two_linear_latents()
        centres = prior_environment(graph)
        at = {n: centres[n] for n in graph.latents}
        block = linear_operator(graph, ("a", "b"), at={})
        precision = precision_at(graph, at)
        mu, _ = wiener_solve(block, precision=precision)
        x = {name: mu[name] + 0.1 for name in mu}

        found = float(log_weight(graph, block, x, at=at, precision=precision, mu=mu))

        # The old spelling, in the domain, with no split anywhere.
        operator = normal_operator(block, precision, variance_parts(block))
        delta = jax.tree.map(jnp.subtract, x, mu)
        pushed = operator(delta)
        old = float(
            log_joint(graph, {**at, **x})
            + 0.5 * sum(jnp.sum(delta[n] * pushed[n]) for n in delta)
        )
        assert found == old


# ------------------------------------------------ the refusal that STAYS ------


class TestTheDiagnoseRefusalIsAnIntendedDifference:
    """G9 in full does NOT lift ``diagnose``'s refusal of a complex latent,
    and that is a decision rather than an omission.

    A rank verdict over C is not the number a caller wants: a block with n
    complex coefficients has 2n real degrees of freedom, and the rank of the
    C-linear map is neither of those. The refusal says so by name. Splitting
    the DIAGNOSTIC would be a second semantic decision -- what a null
    direction in R^2n means when reported against a latent declared over C --
    and nothing in this programme has taken it.
    """

    def test_identifiability_still_refuses_and_names_the_reason(self):
        from bayesmith.diagnose.identifiability import identifiability
        from bayesmith.errors import GraphError
        from tests.exact.test_complex import sky_model

        with jax.enable_x64(True):
            graph = sky_model()
            with pytest.raises(GraphError, match="complex"):
                identifiability(graph, names=("alm",))

    def test_prior_sensitivity_still_refuses(self):
        from bayesmith.diagnose.sensitivity import prior_sensitivity
        from bayesmith.errors import GraphError
        from tests.exact.test_complex import sky_model

        with jax.enable_x64(True):
            graph = sky_model()
            with pytest.raises(GraphError, match="complex"):
                prior_sensitivity(graph, names=("alm",))

    def test_the_SOLVE_of_the_same_graph_is_not_refused(self):
        """The other direction, and it is what makes the refusal a boundary
        rather than a wall: the same graph solves, samples and now reports a
        Fisher. Only the two rank-style diagnostics stand down."""
        from bayesmith.dispatch.classify import prior_environment
        from bayesmith.exact.gaussian import precision_at
        from bayesmith.exact.linearity import linear_operator
        from tests.exact.test_complex import sky_model

        graph = sky_model()
        centres = prior_environment(graph)
        block = linear_operator(graph, ("alm",), at={})
        precision = precision_at(graph, {n: centres[n] for n in graph.latents})
        solved, _ = wiener_solve(block, precision=precision)
        assert jnp.issubdtype(solved["alm"].dtype, jnp.complexfloating)
        assert fisher_information(
            block, precision=precision, include_prior=False,
            depends_on_prediction=False,
        ).values.shape == (4, 4)
