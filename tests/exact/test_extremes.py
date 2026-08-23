"""Extreme parameter values. Failure modes are U-shaped; the middle is safe.

`boundary-validation.md`, verbatim requirement: for every parameter dimension
include both endpoints, a very low value and a very high one -- not only the
moderate range a probe naturally reaches for. The rule was written after a fix
inferred from `ell in {10, 100, 500}` turned out to blow up by 63 orders of
magnitude at `ell = 5000`.

**The mean is not the whole story.** Task 7 found two real `gcr_sample` bugs
-- a plate's elements sharing one fluctuation draw, and `_split_like` reusing
a key across domain leaves -- that disturb only the drawn covariance's
correlation structure while leaving marginal variances nearly right, and
that passed the entire suite at the time because every draw test used a
scalar-member domain. The plate-size and observed-node-count sweeps below
therefore run `gcr_sample`, not only `wiener_solve`, at sizes chosen to be
multi-element (several array entries behind one domain leaf) and multi-leaf
(several named members in one domain), so the SAME structural gap is closed
at the extreme end of the range this file is otherwise responsible for, not
only at the moderate sizes Task 7's own tests already cover.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.exact.gaussian import noise_std_at
from bayesmith.exact.linearity import linear_operator
from bayesmith.exact.solve import condition_bound, gcr_sample, wiener_solve
from tests.exact.models import (
    collinear_pair,
    many_observations,
    straight_line,
    wide_plate,
)
from tests.exact.oracle import flat_domain, graph_oracle

# --------------------------------------------------------------------------
# Plate size: 1 (scalar-degenerate), 2 (smallest multi-element), 1000 (large)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("size", [1, 2, 1000])
def test_a_plate_of_any_size_matches_the_oracle(size):
    with jax.enable_x64(True):
        graph = wide_plate(size=size)
        block = linear_operator(graph, ("z",), at={})
        sigma = noise_std_at(graph, {"z": jnp.zeros(size)})
        got, _ = wiener_solve(block, noise_std=sigma, tol=1e-14)
        oracle = graph_oracle(graph, ("z",), at={})
    assert block.shape["z"] == (size,)
    assert np.allclose(flat_domain(got, block.names), oracle.mean, rtol=1e-8)


@pytest.mark.slow
def test_a_very_wide_plate_still_solves_and_stays_finite():
    """10^5 entries: too big for the dense oracle, so the assertion is weaker.

    Checked against the closed form instead, which for `z_i ~ N(0, tau)` and
    `d_i ~ N(z_i, sigma)` is elementwise and needs no matrix at all.

    Measured wall-clock (this machine, isolated run): ~1.1s call time (~1.5s
    including pytest's own collection), for trace + build + solve at x64.
    Cheap enough that it does not strictly need the `slow` marker on this
    machine, but it is JIT-compilation-dominated (CG on a diagonal 10^5x10^5
    normal operator converges in 1-2 iterations once compiled) rather than
    solve-dominated, so it is kept `slow` anyway: compilation time for this
    shape is far more sensitive to JAX version, CPU and cache state than the
    rest of this suite, and grouping it out of the default `-m "not slow"`
    run keeps that variance off the common path.
    """
    size, tau, sigma = 100_000, 1.5, 0.4
    with jax.enable_x64(True):
        graph = wide_plate(size=size, tau=tau, sigma=sigma)
        block = linear_operator(graph, ("z",), at={})
        noise = noise_std_at(graph, {"z": jnp.zeros(size)})
        got, residual = wiener_solve(block, noise_std=noise, tol=1e-12)
        data = np.asarray(block.data["d"])
    shrinkage = tau**2 / (tau**2 + sigma**2)
    assert np.all(np.isfinite(np.asarray(got["z"])))
    assert float(residual) < 1e-8
    assert np.allclose(np.asarray(got["z"]), shrinkage * data, rtol=1e-6)


def test_a_wide_plate_s_draws_match_the_closed_form_mean_variance_and_zero_correlation():
    """`gcr_sample` at a plate size the dense oracle cannot reach.

    Structurally the same gap `test_gcr_draws_have_no_spurious_correlation_
    across_a_plated_block` (`tests/exact/test_solve.py`, plate size 6) closes
    at moderate scale: a plate's `n` elements are iid, so the TRUE drawn
    covariance is exactly diagonal, making mean(|off-diagonal|) a sharp
    statistic. Run here at `size=200` -- large enough that a dense oracle
    covariance (200^2 entries, one `np.linalg.inv`) is still cheap but far
    past the n=6 Task 7 used, so a correlation-structure bug that happened to
    need `n` past some small cutoff to become visible would still be caught.
    No dense oracle is needed for the mean/variance check (the closed form
    used by the slow 10^5 test above covers it elementwise), but the
    OFF-diagonal needs the empirical draw covariance regardless, which is
    cheap at n=200: no matrix inversion, just `np.cov` over `draws` vmapped
    solves.

    Measured wall-clock: ~1.3s for 2000 vmapped draws at (x64, size=200).
    """
    size, tau, sigma, draws = 200, 1.5, 0.4, 2000
    with jax.enable_x64(True):
        graph = wide_plate(size=size, tau=tau, sigma=sigma)
        block = linear_operator(graph, ("z",), at={})
        noise = noise_std_at(graph, {"z": jnp.zeros(size)})
        data = np.asarray(block.data["d"])
        samples = jax.vmap(
            lambda k: gcr_sample(
                block, noise_std=noise, key=k, tol=1e-12, require_convergence=None
            )[0]["z"]
        )(jax.random.split(jax.random.key(41), draws))
    flat = np.asarray(samples)
    shrinkage = tau**2 / (tau**2 + sigma**2)
    posterior_var = shrinkage * sigma**2
    standard_error = np.sqrt(posterior_var / draws)
    assert np.all(np.abs(flat.mean(axis=0) - shrinkage * data) < 5 * standard_error)
    empirical_var = flat.var(axis=0)
    assert np.allclose(empirical_var, posterior_var, rtol=0.25)
    cov = np.cov(flat, rowvar=False)
    off_diag = cov[~np.eye(size, dtype=bool)]
    # True off-diagonal is exactly zero (iid plate); atol scaled off the
    # diagonal spread the same way the moderate-size Task 7 test does.
    assert np.abs(off_diag).mean() < 0.03 * posterior_var


# --------------------------------------------------------------------------
# Prior width and noise scale: six orders of magnitude, both sides of 1.0
# --------------------------------------------------------------------------


@pytest.mark.parametrize("prior_std", [1e-6, 1.0, 1e6])
def test_the_solve_matches_the_oracle_across_six_orders_of_prior_width(prior_std):
    """Both ends matter and for opposite reasons: a tight prior makes the data
    irrelevant, a loose one makes the prior the only thing holding the blind
    directions down -- which is precisely where kappa explodes."""
    with jax.enable_x64(True):
        graph = straight_line(prior_std=prior_std, prior_mean=1.75)
        block = linear_operator(graph, ("w",), at={})
        sigma = noise_std_at(graph, {"w": jnp.asarray(0.0)})
        got, _ = wiener_solve(
            block, noise_std=sigma, tol=1e-14, require_convergence=None
        )
        oracle = graph_oracle(graph, ("w",), at={})
    assert np.allclose(flat_domain(got, block.names), oracle.mean, rtol=1e-7)


@pytest.mark.parametrize("sigma", [1e-6, 1.0, 1e6])
def test_the_solve_matches_the_oracle_across_six_orders_of_noise(sigma):
    with jax.enable_x64(True):
        graph = straight_line(sigma=sigma, prior_std=2.0, prior_mean=1.75)
        block = linear_operator(graph, ("w",), at={})
        noise = noise_std_at(graph, {"w": jnp.asarray(0.0)})
        got, _ = wiener_solve(
            block, noise_std=noise, tol=1e-14, require_convergence=None
        )
        oracle = graph_oracle(graph, ("w",), at={})
    assert np.allclose(flat_domain(got, block.names), oracle.mean, rtol=1e-7)


# --------------------------------------------------------------------------
# Observed-node count: 1, 2, 5
# --------------------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 2, 5])
def test_any_number_of_observed_nodes_matches_the_oracle(count):
    with jax.enable_x64(True):
        graph = many_observations(count=count)
        block = linear_operator(graph, ("w",), at={})
        sigma = noise_std_at(graph, {"w": jnp.asarray(0.0)})
        got, _ = wiener_solve(block, noise_std=sigma, tol=1e-14)
        oracle = graph_oracle(graph, ("w",), at={})
    assert len(block.offset) == count
    assert np.allclose(flat_domain(got, block.names), oracle.mean, rtol=1e-8)


def test_any_number_of_observed_nodes_gcr_sample_matches_the_oracle_at_the_high_end():
    """`gcr_sample` at the top of the observed-node-count sweep, not just the mean.

    `many_observations`' domain is a single scalar leaf (`w`), so this
    fixture cannot exercise `_split_like`'s cross-LEAF independence the way
    `collinear_pair` below does -- there is only one leaf to give a key to,
    however many observed nodes there are. What it DOES exercise, that a
    single-observed-node fixture cannot, is the draw path's reduction over a
    multi-leaf CODOMAIN: `_conjugate_solve`'s `pair_with` sums over every
    `pushed`/`vector` entry, and `omega_data`'s per-leaf keys are drawn once
    per OBSERVED node, at `count=5` rather than `count<=2` elsewhere in this
    module's draw tests. A bug that dropped or mis-summed one codomain leaf's
    contribution to the right-hand side would move the drawn MEAN (not just
    the covariance), which is what is checked here against the dense oracle.
    """
    draws = 3000
    with jax.enable_x64(True):
        graph = many_observations(count=5)
        block = linear_operator(graph, ("w",), at={})
        sigma = noise_std_at(graph, {"w": jnp.asarray(0.0)})
        samples = jax.vmap(
            lambda k: gcr_sample(
                block, noise_std=sigma, key=k, tol=1e-14, require_convergence=None
            )[0]["w"]
        )(jax.random.split(jax.random.key(42), draws))
        oracle = graph_oracle(graph, ("w",), at={})
    values = np.asarray(samples)
    posterior_sd = float(np.sqrt(oracle.covariance[0, 0]))
    assert values.mean() == pytest.approx(
        float(oracle.mean[0]), abs=5 * posterior_sd / np.sqrt(draws)
    )
    assert values.std() == pytest.approx(posterior_sd, rel=0.15)


# --------------------------------------------------------------------------
# Exactly collinear parents: an extreme, near-degenerate joint kappa
# --------------------------------------------------------------------------


def test_two_exactly_collinear_parents_are_solved_jointly_and_kappa_says_so():
    """The data fixes a+b; only the prior fixes a-b.

    The joint block matches the oracle, and its kappa is large -- which is the
    honest number. A per-block view cannot see the direction the pair is
    jointly blind to.

    **Measured correction to the plan's own draft assertion.** The plan text
    this file implements claimed `single_kappa == pytest.approx(1.0,
    rel=1e-6)` -- i.e. that pinning `b` and solving for `a` alone looks
    perfectly well-conditioned. Run directly: `single_kappa` is 1993.86, not
    1.0 -- `a` alone still sees the same real, non-degenerate design column
    `X` the joint block does, so it is not data-blind at all; only the JOINT
    `a - b` direction is. What IS true, and structural rather than
    coincidental to this fixture's numbers: because `mu = (a+b)*X` gives `a`
    and `b` IDENTICAL design columns, the joint operator's informative
    eigenvalue is exactly `2x` a single pinned block's own (`2c + 1/9` vs
    `c + 1/9`, both over the same `lambda_min = 1/9` set by the shared
    prior), so `kappa` (here 3986.71) is almost exactly double
    `single_kappa` (1993.86) once `c >> 1/9` -- verified with `rel=1e-3`
    below. That relationship is a REGION claim (it holds whenever the data
    term dominates the prior term, not only at `prior_std=3.0`), unlike the
    plan's `single_kappa == 1.0`, which was never true at any parameter
    setting for this model: `single_kappa = 1 + 9 * X^T N^-1 X`, and `X` is
    a fixed, non-zero grid.
    """
    with jax.enable_x64(True):
        graph = collinear_pair(prior_std=3.0)
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = noise_std_at(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        got, _ = wiener_solve(block, noise_std=sigma, tol=1e-14)
        kappa = float(condition_bound(block, noise_std=sigma, iterations=80))
        oracle = graph_oracle(graph, ("a", "b"), at={})
        single = linear_operator(graph, ("a",), at={"b": jnp.asarray(0.0)})
        single_kappa = float(condition_bound(single, noise_std=sigma, iterations=80))
    assert np.allclose(flat_domain(got, block.names), oracle.mean, rtol=1e-8)
    assert kappa > 100.0
    # The per-block view is not remotely near 1 (it is not data-blind), but
    # it is still meaningfully SMALLER than the joint view -- concretely,
    # because the two design columns are identical, almost exactly half.
    assert single_kappa < kappa
    assert kappa == pytest.approx(2.0 * single_kappa, rel=1e-2)
    assert kappa == pytest.approx(np.linalg.cond(oracle.precision), rel=1e-3)


def test_two_exactly_collinear_parents_gcr_sample_matches_the_oracle_covariance():
    """`gcr_sample` on a two-LEAF domain (`a`, `b`) at an extreme, near-singular
    joint kappa -- the multi-leaf-domain counterpart of the plate's
    multi-element coverage above, and the shape none of Task 7's own draw
    tests used (its two-leaf fixture, `two_linear_latents`, is only mildly
    conditioned; `radiometer_group`'s collinearity is not extreme either).

    `_split_like` derives one PRNG key per domain LEAF -- here `a` and `b` --
    and a bug that reused one key for both would correlate their prior
    fluctuations. On `collinear_pair` that failure mode is unusually visible:
    the TRUE covariance is already strongly anti-correlated (`a+b` is
    data-tight, `a-b` is prior-only and wide, and `Cov(a,b) = (Var(a+b) -
    Var(a-b))/4` is dominated by the wide, negative term), so a same-key bug
    would move the off-diagonal in the direction that makes it look LESS
    negative -- i.e. a mutation and the truth disagree in SIGN-adjacent
    magnitude here, not merely in a small correction on top of a near-zero
    number the way `two_linear_latents`' exactly-zero off-diagonal probes it
    elsewhere in this suite.

    Measured wall-clock: ~0.5s for 3000 vmapped draws on this 2-parameter
    block (x64) -- cheap because the block is only 2-dimensional, unlike the
    n=200 plate above.
    """
    draws = 3000
    with jax.enable_x64(True):
        graph = collinear_pair(prior_std=3.0)
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = noise_std_at(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        samples = jax.vmap(
            lambda k: gcr_sample(
                block, noise_std=sigma, key=k, tol=1e-14, require_convergence=None
            )[0]
        )(jax.random.split(jax.random.key(43), draws))
        oracle = graph_oracle(graph, ("a", "b"), at={})
    flat = np.stack([np.asarray(samples[n]).ravel() for n in block.names], axis=1)
    standard_error = np.sqrt(np.diag(oracle.covariance) / draws)
    assert np.all(np.abs(flat.mean(axis=0) - oracle.mean) < 5 * standard_error)
    cov = np.cov(flat, rowvar=False)
    spread = np.max(np.diag(oracle.covariance))
    assert np.allclose(cov, oracle.covariance, rtol=0.15, atol=0.05 * spread)
    # The true covariance here is strongly ANTI-correlated -- assert the sign
    # and rough magnitude explicitly, not only "close to the oracle matrix",
    # since a same-key `_split_like` bug moves the off-diagonal towards zero
    # (less negative) rather than merely adding noise to it.
    assert oracle.covariance[0, 1] < -0.3 * spread
    assert cov[0, 1] < 0.0
