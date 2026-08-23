"""The conjugate solves, against a dense oracle that shares none of them."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.exact.block import domain_zero, variance_parts
from bayesmith.exact.gaussian import noise_std_at
from bayesmith.exact.linearity import linear_operator
from bayesmith.exact.solve import (
    PRECISION_FLOOR,
    condition_bound,
    gcr_sample,
    normal_operator,
    wiener_solve,
)
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


def _assert_orderings_agree(oracle, block):
    """The two sides flatten the domain independently; nothing ties them together.

    `flat_domain` walks `block.names`, per member; the oracle records
    `order` per ELEMENT. A caller who passed a differently-ordered tuple to
    one side would get a same-length, silently transposed comparison, which
    is why this is asserted rather than trusted -- but the comparison has to
    be made at the same granularity. An earlier version of this check
    compared the per-element name list directly against `block.names` and so
    read `['z'] * 6 == ('z',)` for a plated member: False for a block that is
    perfectly well ordered. It passed only where every member happened to be
    scalar, which is the one place it could not fail.
    """
    per_member = list(dict.fromkeys(name for name, _ in oracle.order))
    assert per_member == list(block.names), (per_member, block.names)
    # And the flat lengths must agree, which is what the comparison below
    # actually depends on -- the name check alone would pass for a member
    # whose shape the two sides disagreed about.
    assert len(oracle.order) == oracle.mean.size


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

    **What this proves.** R1 (this matrix-free CG solve) and R2 (the dense
    oracle in ``tests/exact/oracle.py``) share no linear algebra: no
    ``jax.linearize``, no ``jax.vjp``, no ``cg``, no ``tree_norm``, no power
    iteration. Measured agreement, x64, across the three acceptance
    fixtures: 3.1e-16 / 1.8e-16 / 2.6e-16 relative difference -- the float64
    ULP floor. That is as strong a statement as this package can make about
    the CONJUGATE-GRADIENT SOLVE and the NORMAL-OPERATOR CONSTRUCTION: given
    a correct ``(A, offset, data, sigma, prior mean, prior std)``, the
    posterior mean these two completely independent routes compute agrees to
    machine precision.

    **What this does NOT prove**, stated precisely rather than left to be
    inferred from "nothing is shared but the model": R1 and R2 both read
    ``(A, offset, data, sigma, prior mean, prior std)`` off the graph through
    the SAME extraction functions -- ``_env_before``, ``observation_parts``
    and ``isolate`` -- because the oracle's ``g`` is exactly
    ``bayesmith.exact.block.isolate``'s output (see that module's own
    docstring) and both call ``observation_parts``/``_env_before`` directly,
    not an independent reimplementation. A bug in EXTRACTION shifts both
    sides together, and this gate cannot see it -- the P1 design record's own
    blind spot (two readings of one graph sharing an implementation share its
    blind spots; -225.65 agreed with -364.95). Measured, not argued: mutating
    each of the extraction paths below and re-running this gate leaves it
    GREEN every time. What actually catches each one lives elsewhere in the
    suite:

    ============================  ======  ================================
    mutated                        gate    what actually catches it
    ============================  ======  ================================
    ``_env_before`` prior_std      GREEN   test_block.py::
    -> variance                            test_the_prior_is_read_off_the_graph,
                                            test_each_of_the_two_is_a_legitimate_block_on_its_own,
                                            test_variance_parts_places_each_prior_on_its_own_leaf,
                                            test_largest_variance_takes_the_loosest_prior_not_the_tightest
    ``_env_before`` prior_mean     GREEN   test_block.py::
    + 1.0                                  test_the_prior_is_read_off_the_graph,
                                            test_domain_centre_is_the_declared_prior_mean;
                                            this file::
                                            test_a_latent_the_data_never_reaches_comes_back_at_its_prior_mean
    ``observation_parts`` sigma    GREEN   test_gaussian.py::
    x 2                                    test_observation_parts_covers_every_observed_node
                                            (checks ``scale``)
    ``observation_parts`` data     GREEN   test_gaussian.py::
    + 5.0                                  test_observation_parts_covers_every_observed_node
                                            (checks ``data`` -- ADDED for this
                                            finding; nothing caught it before)
    ``isolate``'s ``g``, first     GREEN   crashes (JAX pytree-structure
    observed node only                     ValueError) on the two-observed
                                            sibling: test_block.py::
                                            test_the_block_spans_every_observed_node,
                                            test_adjoint_is_the_transpose_under_the_real_inner_product;
                                            this file::
                                            test_the_normal_operator_sums_over_every_observed_node,
                                            test_wiener_solve_matches_the_oracle_across_two_observed_nodes
    ``isolate``'s ``g``, x 1.1      GREEN   test_block.py::
                                            test_offset_is_the_prediction_with_the_block_at_zero,
                                            test_forward_is_the_linear_action_of_the_block
    ============================  ======  ================================

    Every row is caught SOMEWHERE, by a test that reads the extracted value
    directly rather than through a second independent computation -- which is
    the only kind of check available for a value neither R1 nor R2
    re-derives. None of that coverage is this gate's; it is named here so a
    future reader does not mistake "R1 agrees with R2 to the ULP floor" for
    "the extraction layer is independently verified" -- it is verified, but
    by the tests in this table, not by this one.
    """
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = _sigma(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        got, residual = wiener_solve(block, noise_std=sigma, tol=1e-14)
        oracle = graph_oracle(graph, ("a", "b"), at={})
    # Assert the ordering agreement rather than relying on both call sites
    # staying in step -- see _assert_orderings_agree's own docstring for why
    # the naive per-element comparison this once was is wrong in general.
    _assert_orderings_agree(oracle, block)
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
    _assert_orderings_agree(oracle, block)
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
    _assert_orderings_agree(oracle, block)
    assert np.allclose(flat_domain(got, block.names), oracle.mean, rtol=1e-8)


def test_a_latent_the_data_never_reaches_comes_back_at_its_prior_mean():
    with jax.enable_x64(True):
        graph = unconstrained_latent()
        block = linear_operator(graph, ("w", "u"), at={})
        sigma = _sigma(graph, {"w": jnp.asarray(0.0), "u": jnp.asarray(0.0)})
        got, _ = wiener_solve(block, noise_std=sigma, tol=1e-14)
    assert float(got["u"]) == pytest.approx(1.37, rel=1e-6)


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

    **Why ``require_convergence`` is computed, not a hardcoded literal.** The
    first version of this test hardcoded ``require_convergence=1e-3`` against
    `two_linear_latents`'s plain float32 numbers: ``residual=1.731e-7``,
    ``bound=5792``, ``eps=1.192e-7``, so ``error_bound=1.0025e-3`` cleared the
    1e-3 target by 0.25%. That margin is not just thin, it is provably as
    good as a HARDCODED target can ever get here, for a structural reason:
    write ``r = residual/eps`` and ``beta = bound*eps``. Then
    ``error_bound = r*beta``, and ``at_precision_floor`` REQUIRES
    ``r <= PRECISION_FLOOR`` (10.0) by definition. For ``require_convergence``
    to sit strictly between ``beta`` (clause 1 false, isolating the floor
    term) and ``error_bound`` by a factor of ``K``, algebra gives
    ``r > K``, so ``K < 10`` is a hard ceiling -- "an order of magnitude"
    (``K=10``) is mathematically unreachable together with "isolates the
    floor disjunct", not merely hard to find. And empirically ``r`` never
    gets close to that ceiling: measured directly across `two_linear_latents`
    (9 seeds x 3 n x 4 sigma), a custom tight-prior/loose-prior pair (8
    tightness combinations) and a plated tight-prior model (12 n/tau
    combinations) -- 300+ fixture-and-seed combinations in total -- the
    largest baseline ``r`` found was ~2.55, and requiring the SAME
    ``require_convergence`` to also clear ``error_bound`` under three
    unrelated mutations (`observation_parts` sigma x2, `observation_parts`
    data +5.0, `_env_before` prior_mean +1.0 -- each shifts ``residual`` and
    ``bound`` by different, uncorrelated amounts) found not one feasible
    fixture: every candidate has at least one of those mutations pushing its
    own ``error_bound`` below the baseline's ``beta``, so no single hardcoded
    number survives all three AND isolates the floor term for the baseline.

    The fix computes ``require_convergence`` from THIS run's own measured
    ``error_bound`` (guard off, so the measurement cannot be circular) as
    ``error_bound / SAFETY``. That makes ``error_bound > require_convergence``
    -- and therefore ``bad=True`` -- true BY CONSTRUCTION for any ``bound``
    and any finite positive ``residual``, independent of whatever a mutation
    does to either one; the exact-arithmetic identity
    ``error_bound / (error_bound/SAFETY) = SAFETY > 1`` does not depend on
    the numeric value of ``error_bound`` at all. So this test's PASS/FAIL is
    robust to any mutation that leaves the block well-posed -- not merely the
    three checked below.

    **What is deliberately NOT a live assertion, and why.** Whether
    ``at_precision_floor`` specifically (rather than ``bound*eps``) is what
    makes ``unreachable`` true needs ``r > SAFETY``, which holds for the
    baseline (measured below) but NOT for every mutation -- an earlier
    version of this test asserted ``require_convergence > beta`` unconditionally
    to check that live, and it is exactly what broke: under `data +5.0` and
    `prior_mean +1.0`, that same run's own ``r`` drops under ``SAFETY``,
    ``require_convergence`` no longer clears ``beta``, and the assertion
    failed BEFORE ``wiener_solve`` was even called -- a second, self-inflicted
    instance of this test's own central lesson, caught by actually re-running
    the three mutations against the assertion rather than assuming it would
    survive them. Removed rather than guarded, because guarding it (e.g. only
    checking when mutation-free) is not something a test can detect about
    itself, and the property is not what PASS/FAIL depends on here -- the
    raised message is the same "cannot reach require_convergence at this
    precision" text whichever disjunct of ``unreachable`` is responsible, so
    the ``pytest.raises`` below is unaffected either way. The floor-isolation
    claim for the baseline is instead established by the measurement printed
    next, exactly as :func:`condition_bound`'s own docstring points at a
    named test rather than asserting its tightness inline.

    ``two_linear_latents(n=60, sigma=1.0)`` was the best of the swept
    fixtures for the baseline's own ``r`` (~2.55, against ~1.45 for the
    original n=12/sigma=0.4 defaults), and ``SAFETY=2`` sits comfortably
    below it: measured, ``residual=3.044e-7``, ``bound=4050``,
    ``beta=4.828e-4``, ``error_bound=1.233e-3``,
    ``require_convergence=6.165e-4`` -- a 1.28x margin over ``beta`` and an
    exact 2x margin under ``error_bound`` for THIS baseline, both real
    margins rather than a coincidence of one hardcoded number. Measured
    directly, re-running the three mutations above against this test (each
    independently, then reverted) all three leave it green -- not because
    ``at_precision_floor`` stays isolated in every mutated run (it does not:
    `data +5.0` and `prior_mean +1.0` push that run's own ``r`` below
    ``SAFETY``, so ``unreachable`` ends up true via ``bound*eps`` for those
    two instead) but because ``bad=True`` no longer depends on which clause
    fired.
    """
    graph = two_linear_latents(n=60, sigma=1.0)
    block = linear_operator(graph, ("a", "b"), at={})
    sigma = _sigma(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
    bound = float(condition_bound(block, noise_std=sigma))
    epsilon = float(jnp.finfo(jnp.result_type(*jax.tree.leaves(block.offset))).eps)
    _, residual = wiener_solve(
        block, noise_std=sigma, tol=1e-14, require_convergence=None
    )
    residual = float(residual)
    # Confirms the fixture reaches its claimed regime -- robust across the
    # three mutations too: measured, residual stayed under 10*eps in every
    # one of them, since none pushes CG's float32 floor up by an order of
    # magnitude.
    assert residual <= PRECISION_FLOOR * epsilon, (residual, epsilon)
    error_bound = residual * bound
    safety = 2.0
    require_convergence = error_bound / safety
    with pytest.raises(Exception, match="no tol or maxiter will help"):
        wiener_solve(
            block, noise_std=sigma, tol=1e-14, require_convergence=require_convergence
        )


def test_a_non_finite_residual_gets_its_own_message():
    """NaN/Inf is not a convergence problem, and the other two messages both
    give advice a non-finite residual cannot act on -- tightening `tol` or
    raising `maxiter` does nothing when the right-hand side was already
    non-finite before CG started.

    `b`'s prior_std forced to exactly 0.0 (via dataclasses.replace, same
    fixture-mutation pattern as the `loosened` block elsewhere in this file)
    makes S^-1 = 1/0**2 at that entry, so `rhs`'s `b` leaf is `0.0/0.0 = NaN`
    (`b`'s prior_mean is itself 0.0 on `two_linear_latents`) -- one of the
    causes named in the guard's own message. This cannot be reached through
    `linear_operator` on an ordinary graph: `check_gaussian` refuses a
    non-positive sigma at block-build time (see `gaussian.py`), so a real
    model can never hand `wiener_solve` a block like this one -- it has to
    be built the same way the `loosened` block above is, by mutating an
    already-checked block after the fact.

    Measured directly: with the guard OFF (`require_convergence=None`), the
    returned residual is exactly `nan` (`jnp.isfinite` is False), confirming
    the fixture reaches the regime this test claims before checking what the
    guard does with it.
    """
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = _sigma(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        broken = dataclasses.replace(
            block, prior_std={**block.prior_std, "b": jnp.asarray(0.0)}
        )
        _, residual = wiener_solve(
            broken, noise_std=sigma, tol=1e-14, require_convergence=None
        )
        assert not bool(jnp.isfinite(residual)), float(residual)
        with pytest.raises(Exception, match="non-finite residual"):
            wiener_solve(broken, noise_std=sigma, tol=1e-14)


@pytest.mark.slow
def test_gcr_draws_have_the_oracle_mean_and_covariance():
    """The draw is exact, so its first two moments are the oracle's.

    require_convergence=None inside the vmap on purpose: the guard costs
    POWER_ITERATIONS operator applications PER DRAW, and tol is set from
    the block's kappa instead -- which is the bargain wiener_solve's docstring
    recommends and this test is the demonstration of.

    **Measured power, and it is lopsided.** Scaling the DATA fluctuation
    term (``A^T N^-1/2 omega_1``) by a factor ``k`` and rerunning this
    exact assertion: red at ``k=1.10`` and ``k=0.90``, green at ``k=1.05``
    -- roughly the 10% its own ``rtol=0.1`` promises. Scaling the PRIOR
    fluctuation term (``S^-1/2 omega_2``) instead: green all the way to
    ``k=20``, first red at ``k=25`` -- this assertion has effectively NO
    power there. Why: ``two_linear_latents`` is data-dominated (prior
    variance 25/49 against posterior variance 0.0085/0.013, i.e.
    ``oracle.precision`` diag ~[118, 75] against ``S^-1`` diag
    [0.04, 0.0204]), and scaling ``omega_2`` by ``k`` turns the draw
    covariance from ``M^-1`` into ``M^-1(A^T N^-1 A + k^2 S^-1)M^-1``, so
    ``k^2 S^-1`` stays negligible next to ``A^T N^-1 A`` until
    ``k ~ sqrt(118/0.04) ~ 54`` -- same order as the measured crossover.
    The prior term is guarded instead by
    ``test_a_draw_with_uninformative_data_falls_back_to_the_prior``, whose
    fixture inverts this one's dominance. Read the two together: neither
    is redundant with the other, and deleting either one for looking like
    a duplicate removes real coverage silently.
    """
    draws = 4000
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = _sigma(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        samples = jax.vmap(
            lambda k: gcr_sample(
                block, noise_std=sigma, key=k, tol=1e-14, require_convergence=None
            )[0]
        )(jax.random.split(jax.random.key(20), draws))
        oracle = graph_oracle(graph, ("a", "b"), at={})
    flat = np.stack([np.asarray(samples[n]).ravel() for n in block.names], axis=1)
    standard_error = np.sqrt(np.diag(oracle.covariance) / draws)
    assert np.all(np.abs(flat.mean(axis=0) - oracle.mean) < 4 * standard_error)
    spread = np.max(np.diag(oracle.covariance))
    assert np.allclose(
        np.cov(flat, rowvar=False), oracle.covariance, rtol=0.1, atol=0.05 * spread
    )


@pytest.mark.slow
def test_a_draw_with_uninformative_data_falls_back_to_the_prior():
    """With sigma enormous the likelihood says nothing, so the draw is the prior.

    The check that the S^-1/2 fluctuation term is wired in at the right width:
    drop it and the draws collapse onto the prior MEAN with no scatter at all.

    **Measured power, complementary to
    test_gcr_draws_have_the_oracle_mean_and_covariance.** Scaling the PRIOR
    fluctuation term (``S^-1/2 omega_2``) by a factor ``k``: the std
    assertion goes red at ``k=1.10`` and ``k=0.85``, green at ``k=1.05``
    and ``k=0.90`` -- roughly the 10% its own ``rel=0.1`` promises. Scaling
    the DATA fluctuation term (``A^T N^-1/2 omega_1``) instead leaves mean
    and std unchanged even at ``k=1000`` -- zero power, which is exactly
    what "the likelihood says nothing" should look like when measured
    rather than assumed. This is the mirror image of the covariance
    test's power (~10% on ``omega_1``, none on ``omega_2`` there, on
    ``two_linear_latents`` where the data dominates instead): the two
    tests are not redundant, each is blind exactly where the other sees,
    and both are needed to cover both fluctuation terms.
    """
    draws = 3000
    with jax.enable_x64(True):
        graph = straight_line(sigma=1e6, prior_mean=1.75, prior_std=2.0)
        block = linear_operator(graph, ("w",), at={})
        sigma = _sigma(graph, {"w": jnp.asarray(0.0)})
        samples = jax.vmap(
            lambda k: gcr_sample(
                block, noise_std=sigma, key=k, tol=1e-14, require_convergence=None
            )[0]["w"]
        )(jax.random.split(jax.random.key(21), draws))
    values = np.asarray(samples)
    assert values.mean() == pytest.approx(1.75, abs=4 * 2.0 / np.sqrt(draws))
    assert values.std() == pytest.approx(2.0, rel=0.1)


@pytest.mark.slow
def test_the_mean_of_many_draws_is_the_wiener_solution():
    """The two exits share one solve, so they cannot disagree about the centre.

    Also, incidentally, an accidental covariance-width detector: Task 7's
    mutation 4 (dropping ``jnp.sqrt`` from ``sqrt(weight) * omega_data`` in
    the pairing) widened this fixture's drawn covariance ~6x and turned
    THIS test red too, alongside
    test_gcr_draws_have_the_oracle_mean_and_covariance, which it was not
    written to check -- the wider covariance inflates the Monte Carlo
    error of the sample MEAN past this test's tolerance, which is sized
    from the true, un-mutated ``posterior_sd``. That is a side effect of
    this test's own finite-``draws`` sampling noise, not a designed
    check: a red here is not by itself evidence of a mean bug in
    ``gcr_sample``, and whether the covariance test is ALSO red is what
    tells the two apart.
    """
    draws = 3000
    with jax.enable_x64(True):
        graph = two_observations()
        block = linear_operator(graph, ("w",), at={})
        sigma = _sigma(graph, {"w": jnp.asarray(0.0)})
        mean, _ = wiener_solve(block, noise_std=sigma, tol=1e-14)
        samples = jax.vmap(
            lambda k: gcr_sample(
                block, noise_std=sigma, key=k, tol=1e-14, require_convergence=None
            )[0]["w"]
        )(jax.random.split(jax.random.key(22), draws))
        oracle = graph_oracle(graph, ("w",), at={})
    values = np.asarray(samples)
    posterior_sd = float(np.sqrt(oracle.covariance[0, 0]))
    assert values.mean() == pytest.approx(
        float(mean["w"]), abs=4 * posterior_sd / np.sqrt(draws)
    )


@pytest.mark.slow
def test_gcr_draws_have_no_spurious_correlation_across_a_plated_block():
    """None of the other three GCR tests uses a plated (multi-element) member,
    so the draw's CORRELATION structure across one member's own elements was
    never checked -- only scalar members, where there is no such structure to
    get wrong. `_split_like` derives one key per LEAF (per member, not per
    element); correctness then depends on `jax.random.normal(key,
    leaf.shape)` fanning that single key out into `leaf.shape`-many
    INDEPENDENT values. Measured: mutating that call to draw one scalar per
    leaf and broadcast it across the leaf's shape -- so a plate's n entries
    get one perfectly correlated fluctuation instead of n independent ones --
    passes every one of the other 177 tests in this suite.

    `plated_latent`'s six `z_i` are iid (independent priors, independent
    likelihoods, no coupling between them), so the oracle's true covariance
    is EXACTLY diagonal -- the off-diagonal is not small, it is zero by
    construction. That makes mean(|off-diagonal|) a much sharper statistic
    than the full-matrix comparison below: the full comparison's margins are
    real but not generous (measured, N=4000, key(31): correct max|off-diag|
    0.00364 sits 2.05x under `atol`; the broadcast mutation's 0.01299 sits
    1.74x over it) because `atol` has to stay loose enough for MC noise on
    all 36 entries at once, diagonal included. The second assertion answers
    a narrower question -- is the off-diagonal centred on its true value of
    zero -- and separates correct (mean|off|=0.00194) from the mutation
    (mean|off|=0.00918) by 4.7x, measured with the same draws and key.
    """
    draws = 4000
    with jax.enable_x64(True):
        graph = plated_latent(n=6)
        block = linear_operator(graph, ("z",), at={})
        sigma = _sigma(graph, {"z": jnp.zeros(6)})
        samples = jax.vmap(
            lambda k: gcr_sample(
                block, noise_std=sigma, key=k, tol=1e-14, require_convergence=None
            )[0]["z"]
        )(jax.random.split(jax.random.key(31), draws))
        oracle = graph_oracle(graph, ("z",), at={})
    _assert_orderings_agree(oracle, block)
    flat = np.asarray(samples)
    cov = np.cov(flat, rowvar=False)
    spread = np.max(np.diag(oracle.covariance))
    assert np.allclose(cov, oracle.covariance, rtol=0.1, atol=0.05 * spread)
    off_diag = cov[~np.eye(cov.shape[0], dtype=bool)]
    assert np.abs(off_diag).mean() < 0.03 * spread
