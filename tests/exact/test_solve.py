"""The conjugate solves, against a dense oracle that shares none of them."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.exact.block import domain_zero, variance_parts
from bayesmith.exact.gaussian import noise_std_at
from bayesmith.exact.linearity import linear_operator
from bayesmith.exact.solve import condition_bound, normal_operator, wiener_solve
from tests.exact.models import (
    plated_latent,
    prior_held_direction,
    straight_line,
    two_linear_latents,
    two_observations,
    unconstrained_latent,
)
from tests.exact.oracle import flat_domain, graph_oracle


def _sigma(graph, at):
    return noise_std_at(graph, at)


def test_the_bound_is_lambda_max_times_the_loosest_prior_variance():
    """The bound's definition, checked against a dense eigendecomposition.

    lambda_max comes from the oracle's precision matrix -- which is built by
    probing g on a basis and never differentiates anything -- so this is the
    matrix-free power iteration against an independent route, not against
    itself.
    """
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = _sigma(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        bound = float(condition_bound(block, noise_std=sigma, iterations=80))
        oracle = graph_oracle(graph, ("a", "b"), at={})
    largest = float(np.linalg.eigvalsh(oracle.precision)[-1])
    loosest_variance = float(np.max(oracle.prior_std**2))
    assert bound == pytest.approx(largest * loosest_variance, rel=1e-3)


@pytest.mark.parametrize("loosened", [1e-4, 1e-2, 1e-1, 1.0, 1e2, 1e4])
def test_the_bound_is_never_below_the_true_condition_number(loosened):
    """The whole point: the bound may refuse a good solve, never accept a bad one.

    Swept across eight orders of magnitude of prior width, **in both
    directions**. Only widening was probed originally, and that half cannot
    exercise the guarantee at all on this fixture: `a`'s prior variance stays
    at 25 while the data alone pins `lambda_min` at 75.0 (set by the design
    and n/sigma**2, not by either prior), so `min(prior_variance) = 25` stays
    far above `1/lambda_min = 0.013` however far `b` is widened, and a
    tightest-prior aggregation still lands 1875x ABOVE the true kappa.

    Tightening is where the guarantee is actually load-bearing, and the
    crossover has a closed form: below

        L* = 1 / sqrt(S_b * lambda_min_data) = 1 / sqrt(49 * 75.02) = 0.0165

    `b`'s prior is tighter than what the data supplies in the weakest
    direction, so `min(prior_variance)` stops being a valid floor. Measured
    ratios of a min-based bound to the true kappa: 5.8e-5 at 1e-4, 0.579 at
    1e-2 (just inside the boundary), 37.8 at 1e-1 (just outside), 1875 at 1
    and above. The parametrisation deliberately straddles L* rather than
    only sampling far from it -- boundary-validation.md's rule is to evaluate
    at the threshold and on both sides of it.
    """
    import dataclasses

    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        widened = dataclasses.replace(
            block,
            prior_std={**block.prior_std, "b": block.prior_std["b"] * loosened},
        )
        sigma = _sigma(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        bound = float(condition_bound(widened, noise_std=sigma, iterations=80))
        oracle = graph_oracle(graph, ("a", "b"), at={})
    # `b`'s flat index depends on the oracle's own domain order, not on an
    # assumed position -- derive it rather than hardcoding 1, so a change to
    # that ordering cannot silently point this mutation at the wrong
    # diagonal entry of the dense precision.
    names = [name for name, _ in oracle.order]
    assert names == ["a", "b"], names
    b_index = names.index("b")
    precision = oracle.precision.copy()
    # Rebuild the dense precision with b's widened prior, so the comparison is
    # against the system the bound was actually computed for.
    precision[b_index, b_index] += (
        1.0 / (oracle.prior_std[b_index] * loosened) ** 2
        - 1.0 / oracle.prior_std[b_index] ** 2
    )
    true_kappa = float(np.linalg.cond(precision))
    assert bound >= true_kappa * (1.0 - 1e-6), (bound, true_kappa)


def test_the_bound_is_loose_when_the_data_constrains_every_direction():
    """Stated rather than hidden: this is the price of a one-sided guarantee.

    A one-parameter block has a true kappa of exactly 1 -- M is 1x1 -- and the
    bound reports lambda_max times the prior variance, which is hundreds. That
    is not a defect; it is what an upper bound derived from the prior must
    say when the data, not the prior, is what sets lambda_min. CG on such a
    block converges in one step, so the residual absorbs the slack.
    """
    with jax.enable_x64(True):
        graph = straight_line()
        block = linear_operator(graph, ("w",), at={})
        sigma = _sigma(graph, {"w": jnp.asarray(0.0)})
        bound = float(condition_bound(block, noise_std=sigma, iterations=40))
        oracle = graph_oracle(graph, ("w",), at={})
    assert float(np.linalg.cond(oracle.precision)) == pytest.approx(1.0, rel=1e-9)
    assert bound > 100.0
    assert bound == pytest.approx(
        float(oracle.precision[0, 0]) * float(oracle.prior_std[0] ** 2), rel=1e-3
    )


def test_the_bound_grows_in_proportion_to_the_loosest_prior_variance():
    """It is linear in max(prior_variance) by construction -- verify it is.

    A bound that ignored the prior would be flat across this pair, and a bound
    that took the TIGHTEST prior would move the wrong way.
    """
    import dataclasses

    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = _sigma(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        tight = float(condition_bound(block, noise_std=sigma, iterations=80))
        widened = dataclasses.replace(
            block, prior_std={**block.prior_std, "b": block.prior_std["b"] * 100.0}
        )
        loose = float(condition_bound(widened, noise_std=sigma, iterations=80))
    # b's prior variance grows by 1e4 and it was already the loosest (7 vs 5).
    assert loose == pytest.approx(1e4 * tight, rel=1e-2)


def test_the_normal_operator_is_symmetric():
    """`<u, M v> == <M u, v>` -- taking the curvature as a gradient of chi2
    makes this true by construction, and an adjoint assembled by hand from A
    and A^T is where the sign conventions would go wrong."""
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        weight = {
            o: 1.0 / jnp.asarray(s) ** 2
            for o, s in _sigma(graph, {"a": 0.0, "b": 0.0}).items()
        }
        normal = normal_operator(block, weight, variance_parts(block))
        keys = jax.random.split(jax.random.key(3), 2)
        u = {
            n: jax.random.normal(k, block.shape[n])
            for n, k in zip(block.names, keys, strict=True)
        }
        v = {
            n: jax.random.normal(jax.random.fold_in(k, 1), block.shape[n])
            for n, k in zip(block.names, keys, strict=True)
        }
        left = sum(float(jnp.sum(u[n] * normal(v)[n])) for n in block.names)
        right = sum(float(jnp.sum(normal(u)[n] * v[n])) for n in block.names)
    assert left == pytest.approx(right, rel=1e-10)


def test_the_normal_operator_reproduces_the_dense_precision_matrix():
    """Applying M to each basis vector must give the dense precision's columns.

    The dense matrix comes from the oracle, which never calls linearize or
    vjp; M comes from `jax.grad` of the block's own chi-squared. They agree
    or one of them is wrong.
    """
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        weight = {
            o: 1.0 / jnp.asarray(s) ** 2
            for o, s in _sigma(graph, {"a": 0.0, "b": 0.0}).items()
        }
        normal = normal_operator(block, weight, variance_parts(block))
        columns = []
        for name in block.names:
            probe = dict(domain_zero(block))
            probe[name] = jnp.asarray(1.0)
            image = normal(probe)
            columns.append(
                np.concatenate([np.asarray(image[n]).ravel() for n in block.names])
            )
        applied = np.stack(columns, axis=1)
        oracle = graph_oracle(graph, ("a", "b"), at={})
    assert np.allclose(applied, oracle.precision, rtol=1e-8)


def test_the_normal_operator_sums_over_every_observed_node():
    """Two observed nodes both contribute curvature; dropping one changes it."""
    with jax.enable_x64(True):
        graph = two_observations()
        block = linear_operator(graph, ("w",), at={})
        weight = {
            o: 1.0 / jnp.asarray(s) ** 2
            for o, s in _sigma(graph, {"w": jnp.asarray(0.0)}).items()
        }
        normal = normal_operator(block, weight, variance_parts(block))
        applied = float(normal({"w": jnp.asarray(1.0)})["w"])
        oracle = graph_oracle(graph, ("w",), at={})
    assert applied == pytest.approx(float(oracle.precision[0, 0]), rel=1e-8)


def test_the_bound_is_tight_when_the_prior_alone_holds_a_direction():
    """The regime `condition_bound` exists for, and the justification for its
    whole design -- which until now nothing tested.

    Every other test of this bound sits on the LOOSE side (3676x, 1e8x, 1e11x
    over the true kappa) because their fixtures let the data constrain every
    direction far better than the prior does. The design's defence is about
    the other side: where the data does not identify a direction, lambda_min
    IS the prior's curvature and the bound is exact. Measured here: ratio
    1.0000.
    """
    with jax.enable_x64(True):
        graph = prior_held_direction()
        block = linear_operator(graph, ("tight", "loose"), at={})
        sigma = _sigma(graph, {"tight": jnp.asarray(0.0), "loose": jnp.asarray(0.0)})
        bound = float(condition_bound(block, noise_std=sigma, iterations=80))
        oracle = graph_oracle(graph, ("tight", "loose"), at={})
    true_kappa = float(np.linalg.cond(oracle.precision))
    assert bound == pytest.approx(true_kappa, rel=1e-3)


def test_the_bound_uses_the_loosest_prior_not_the_tightest():
    """Taking the tightest prior would drop the bound BELOW the true kappa.

    That is not a smaller number, it is the loss of the only guarantee this
    function offers: an upper bound that a convergence guard can be built on.
    Measured on this fixture, a tightest-prior aggregation lands 1e-8 times
    the true condition number.

    `two_linear_latents` cannot show this -- there the mutated value is still
    1875x above the true kappa, so the mutation escapes entirely.
    """
    with jax.enable_x64(True):
        graph = prior_held_direction()
        block = linear_operator(graph, ("tight", "loose"), at={})
        sigma = _sigma(graph, {"tight": jnp.asarray(0.0), "loose": jnp.asarray(0.0)})
        bound = float(condition_bound(block, noise_std=sigma, iterations=80))
        oracle = graph_oracle(graph, ("tight", "loose"), at={})
        tightest = float(np.min(oracle.prior_std**2))
        loosest = float(np.max(oracle.prior_std**2))
    assert loosest / tightest > 1e6, "fixture no longer separates the two priors"
    assert bound >= float(np.linalg.cond(oracle.precision)) * (1.0 - 1e-6)
    # And the tightest-prior aggregation would be catastrophically below it.
    assert bound * (tightest / loosest) < float(np.linalg.cond(oracle.precision))


def test_wiener_solve_matches_the_dense_oracle():
    """R1 vs R2 -- the acceptance gate of this whole plan.

    Nothing is shared between the two but the model: the oracle has no
    linearize, no vjp, no cg, no tree_norm and no power iteration in it.
    """
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = _sigma(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        got, residual = wiener_solve(block, noise_std=sigma, tol=1e-14)
        oracle = graph_oracle(graph, ("a", "b"), at={})
    # The two sides flatten the domain independently -- `flat_domain` walks
    # `block.names`, the oracle walks the `names` it was handed. Nothing ties
    # them together, so a caller who passed a differently-ordered tuple to one
    # side would get a same-length, silently transposed comparison. Assert the
    # agreement rather than relying on both call sites staying in step.
    assert [name for name, _ in oracle.order] == list(block.names)
    assert float(residual) < 1e-10
    assert np.allclose(flat_domain(got, block.names), oracle.mean, rtol=1e-8)


def test_wiener_solve_matches_the_oracle_across_two_observed_nodes():
    """The pytree codomain: both observed nodes must enter A^T N^-1 (d-offset)."""
    with jax.enable_x64(True):
        graph = two_observations()
        block = linear_operator(graph, ("w",), at={})
        sigma = _sigma(graph, {"w": jnp.asarray(0.0)})
        got, _ = wiener_solve(block, noise_std=sigma, tol=1e-14)
        oracle = graph_oracle(graph, ("w",), at={})
    assert np.allclose(flat_domain(got, block.names), oracle.mean, rtol=1e-8)


def test_wiener_solve_matches_the_oracle_for_a_plated_block():
    """A six-dimensional domain, so the domain is a real vector space."""
    with jax.enable_x64(True):
        graph = plated_latent(n=6)
        block = linear_operator(graph, ("z",), at={})
        sigma = _sigma(graph, {"z": jnp.zeros(6)})
        got, _ = wiener_solve(block, noise_std=sigma, tol=1e-14)
        oracle = graph_oracle(graph, ("z",), at={})
    assert got["z"].shape == (6,)
    assert np.allclose(flat_domain(got, block.names), oracle.mean, rtol=1e-8)


def test_a_latent_the_data_never_reaches_comes_back_at_its_prior_mean():
    with jax.enable_x64(True):
        graph = unconstrained_latent()
        block = linear_operator(graph, ("w", "u"), at={})
        sigma = _sigma(graph, {"w": jnp.asarray(0.0), "u": jnp.asarray(0.0)})
        got, _ = wiener_solve(block, noise_std=sigma, tol=1e-14)
    assert float(got["u"]) == pytest.approx(1.25, rel=1e-6)


def test_the_convergence_guard_fires_on_a_deliberately_starved_solve():
    """maxiter=1 leaves a real residual, and the guard bounds the ERROR.

    equinox's runtime-error type varies between versions, so the assertion is
    on the message rather than the class.

    ``require_convergence`` has to land the guard in the "did not converge"
    branch rather than the "precision" branch, and the two branches are not
    interchangeable -- ``unreachable`` can fire first. Measured directly
    (x64, ``two_linear_latents``, ``maxiter=1``, ``tol=1e-14``):
    ``residual=0.2277``, ``bound`` (12 power iterations -- what the guard
    itself uses internally, not the 80 this test file uses elsewhere for a
    tighter estimate) ``=5793``, ``eps=2.220e-16``, so
    ``bound*eps=1.286e-12``. At the plan's original
    ``require_convergence=1e-12``, ``bound*eps`` (1.286e-12) is GREATER than
    the target, so ``unreachable`` is True and the guard raises the
    "precision" message instead -- ``match="did not converge"`` would not
    have matched. At ``require_convergence=1e-9`` used below, ``bound*eps``
    (1.286e-12) sits three orders of magnitude under the target
    (``unreachable`` is False) while ``error_bound = residual*bound = 1319``
    sits twelve orders of magnitude over it (``bad`` is True), landing
    cleanly in "did not converge" with wide margin on both sides.
    """
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = _sigma(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        with pytest.raises(Exception, match="did not converge"):
            wiener_solve(
                block, noise_std=sigma, tol=1e-14, maxiter=1, require_convergence=1e-9
            )


def test_disabling_the_guard_returns_the_unconverged_answer_instead():
    """The guard is what turns a bad answer into an error, nothing else does."""
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = _sigma(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        got, residual = wiener_solve(
            block, noise_std=sigma, tol=1e-14, maxiter=1, require_convergence=None
        )
        oracle = graph_oracle(graph, ("a", "b"), at={})
    assert float(residual) > 1e-12
    assert not np.allclose(flat_domain(got, block.names), oracle.mean, rtol=1e-8)


def test_the_guard_bounds_the_error_not_the_residual():
    """A tiny residual on an ill-conditioned block is not convergence.

    prior_std = 1e4 on `b` makes lambda_min ~ 1e-8 while the data sets
    lambda_max, so kappa is enormous. The residual at maxiter=1 looks fine
    relative to a loose target and the ERROR does not -- which is exactly the
    regime these solvers exist for.

    The second assertion is the bidirectional confirmation the plan asked
    for: solve the SAME loosened block with and without kappa in the error
    bound, and show the verdict differs. It does NOT use the plan's literal
    suggestion (``tol=1e-8, require_convergence=1e-3``) -- measured directly,
    those do not exercise the distinction. This is a 2-parameter block, and
    CG solves an n=2 SPD system EXACTLY within n steps: at any
    ``maxiter >= 2`` the residual is already 4.05e-16 (machine floor), so
    ``residual*kappa = 4.05e-16 * 1.18e10 = 4.79e-6`` -- still comfortably
    BELOW 1e-3, meaning the with-kappa guard would not fire there either at
    that target. ``maxiter=1`` is the only regime with a non-degenerate
    residual on a 2-dimensional block: measured ``residual=0.228``,
    ``kappa=1.18e10``, ``residual*kappa=2.69e9``.
    ``require_convergence=1.0`` sits between them with wide margin on each
    side -- the residual alone (0.228) would satisfy a target of 1.0 (no
    error from a residual-only guard), but the kappa-weighted error (2.69e9)
    misses it by nine orders of magnitude.
    """
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = _sigma(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        loosened = dataclasses.replace(
            block, prior_std={**block.prior_std, "b": jnp.asarray(1e4)}
        )
        kappa = float(condition_bound(loosened, noise_std=sigma, iterations=80))
        # Bidirectional confirmation: read the raw residual with the guard
        # OFF, then show the SAME solve, guard ON, raises -- so it is kappa,
        # not the residual, doing the work. A guard computing
        # `error_bound = residual` (mutation 3) could not have raised here,
        # because the residual alone never crosses 1.0.
        _, residual = wiener_solve(
            loosened, noise_std=sigma, tol=1e-8, maxiter=1, require_convergence=None
        )
        with pytest.raises(Exception, match="did not converge"):
            wiener_solve(
                loosened, noise_std=sigma, tol=1e-8, maxiter=1, require_convergence=1.0
            )
    assert kappa > 1e6
    assert float(residual) < 1.0


def test_the_guard_points_at_enable_x64_in_float32():
    """The `unreachable` branch, not just the `bad` branch, needs a guard.

    Deleting the ``unreachable`` split (mutation 4) leaves every OTHER test
    in this file green, because none of them checks which of the two
    messages comes back -- only that pytest.raises fires. Built at the
    default float32 (no x64 context at all) on the same widened-`b` block as
    `test_the_guard_bounds_the_error_not_the_residual`. Measured directly:
    ``bound`` (12 power iterations) blows up to ``~1.182e10`` with `b`'s
    prior widened to 1e4, and float32 `eps=1.192e-7`, so
    ``bound*eps=1409`` -- six orders of magnitude past the default
    ``require_convergence=1e-3`` on its own, independent of the precision
    floor. That is what should send this down the "precision" branch and
    name `enable_x64` as the remedy, not "did not converge".
    """
    graph = two_linear_latents()
    block = linear_operator(graph, ("a", "b"), at={})
    sigma = _sigma(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
    loosened = dataclasses.replace(
        block, prior_std={**block.prior_std, "b": jnp.asarray(1e4)}
    )
    with pytest.raises(Exception, match="enable_x64"):
        wiener_solve(loosened, noise_std=sigma)


def test_the_precision_floor_alone_makes_the_guard_unreachable():
    """``at_precision_floor`` isolated from the ``bound*eps`` clause.

    Without this test, `at_precision_floor` has no guard at all -- every
    other "unreachable" test in this file (see
    `test_the_guard_points_at_enable_x64_in_float32`) also has
    ``bound*eps > require_convergence`` true on its own, so a mutation
    deleting ONLY the `at_precision_floor` term would leave every existing
    test green.

    Measured directly (float32, plain -- NOT widened -- `two_linear_latents`,
    default ``tol``/``maxiter``, ``require_convergence=1e-3``):
    ``residual=1.731e-7``, ``bound`` (12 power iterations) ``=5792``,
    ``eps=1.192e-7``. ``bound*eps=6.905e-4`` is BELOW the 1e-3 target, so the
    `bound*eps` clause alone does not make this unreachable -- a guard
    carrying only rheplicant's original clause would take the "tighten tol"
    branch here, advice that cannot move a residual already 1.45x the
    float32 epsilon. ``at_precision_floor`` (``residual <= 10*eps=1.192e-6``)
    is True and is the ONLY reason this lands in the precision branch:
    ``error_bound = residual*bound = 1.0025e-3`` just clears the 1e-3
    target, so the guard fires at all only because ``bad`` is True, and
    lands in "precision" only because of the floor term.
    """
    graph = two_linear_latents()
    block = linear_operator(graph, ("a", "b"), at={})
    sigma = _sigma(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
    with pytest.raises(Exception, match="no tol or maxiter will help"):
        wiener_solve(block, noise_std=sigma, tol=1e-14, require_convergence=1e-3)
