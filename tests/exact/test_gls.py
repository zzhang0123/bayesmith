"""Finding the covariance a prediction-dependent sigma implies."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.errors import GraphError, StructureError
from bayesmith.exact.gls import (
    check_prediction_dependence,
    iterative_gls,
    sigma_from_graph,
)
from bayesmith.exact.linearity import linear_operator
from bayesmith.exact.solve import wiener_solve
from tests.exact.models import radiometer, straight_line
from tests.exact.oracle import flat_domain, graph_oracle

KAPPA = 0.05
FLOOR = 1e-3


def test_a_constant_sigma_converges_in_one_step():
    with jax.enable_x64(True):
        graph = straight_line()
        block = linear_operator(graph, ("w",), at={})
        result = iterative_gls(
            block, sigma_from_graph(graph, {}), depends_on_prediction=False, tol=1e-14
        )
        direct, _ = wiener_solve(
            block,
            noise_std=sigma_from_graph(graph, {})({"w": jnp.asarray(0.0)}),
            tol=1e-14,
        )
    assert int(result.iterations) == 1
    assert bool(result.converged)
    assert float(result.solution["w"]) == pytest.approx(float(direct["w"]), rel=1e-10)


def test_iterative_gls_finds_the_fixed_point_a_dense_iteration_finds():
    """A NumPy fixed-point loop, sharing nothing with the JAX while_loop."""
    with jax.enable_x64(True):
        graph = radiometer(kappa=KAPPA, floor=FLOOR)
        block = linear_operator(graph, ("w",), at={})
        result = iterative_gls(
            block, sigma_from_graph(graph, {}), tol=1e-14, reweight_tol=1e-10
        )
        oracle = graph_oracle(graph, ("w",), at={})

    design, offset, data = oracle.design, oracle.offset, oracle.data
    prior_precision = np.diag(1.0 / oracle.prior_std**2)
    x = oracle.prior_mean.copy()
    for _ in range(400):
        sigma = KAPPA * np.abs(design @ x + offset) + FLOOR
        noise_precision = np.diag(1.0 / sigma**2)
        x = np.linalg.solve(
            design.T @ noise_precision @ design + prior_precision,
            design.T @ noise_precision @ (data - offset)
            + prior_precision @ oracle.prior_mean,
        )

    assert bool(result.converged)
    assert np.allclose(flat_domain(result.solution, block.names), x, rtol=1e-6)
    # Mutation guard. Written for Task 8 mutation 1 (the `step` denominator
    # swapped from tree_norm(updated) to tree_norm(latent)), but measured NOT
    # to catch it there: at the default min_reweights=5, mutated and
    # unmutated code give identical iterations/delta/solution on this
    # fixture (agreeing to 10+ significant digits) -- see
    # test_iterative_gls_delta_denominator_uses_the_new_iterate for where
    # that mutation IS caught, at min_reweights=1. What this assertion DOES
    # catch is mutation 2 (the two connectives in `unfinished` swapped):
    # with a valid min<=max, that mutation ignores convergence and always
    # runs to max_reweights (100), well past this bound.
    assert int(result.iterations) < 50


def test_iterative_gls_delta_denominator_uses_the_new_iterate():
    """Regression pin for `step`'s ``tree_norm(updated)`` denominator.

    At the default ``min_reweights=5``, this choice is unguarded by any
    fixture in this module (see the comment above and the one on `step`
    itself) -- the sequence's magnitude stabilises within 1-2 real steps
    while MIN_REWEIGHTS keeps forcing solves regardless, so by the time
    ``delta`` is first consulted the two candidate denominators already
    agree to double precision. At ``min_reweights=1``, ``delta`` is
    consulted after a single reweighting step, comparing ``first`` (the
    naive solve at the sigma the prior mean implies) against ``updated``
    (the first genuinely reweighted solve) -- exactly where the two differ
    most.

    At radiometer()'s exact defaults (kappa=3.5, seed=6, prior_mean=0.0,
    weight=3.0): ``tree_norm(first)=6.390``, ``tree_norm(updated)=3.189``,
    giving ``delta=1.004`` normalised by the new iterate against ``0.501``
    normalised by the old one -- a factor of ~2 apart. ``reweight_tol=0.75``
    sits almost exactly halfway between them (0.249 below the wrong value,
    0.254 below the right one), so the correct denominator does not yet
    call step 2 converged and takes a 3rd step (``iterations=3``, solution
    ``w=3.9160``), while the swapped one stops at step 2 (``iterations=2``,
    solution ``w=4.2542``).

    **This is a POINT separation, not a region -- measured, not assumed.**
    Swept three things that "should not matter" before committing this pin:

    * seed 0-19 (kappa=3.5, prior_mean=0.0 fixed): only 6 of 20 preserve
      the split at reweight_tol=0.75 (3, 4, 6, 9, 10, 16). Several INVERT
      the ordering outright -- seed=8: correct delta=1.14 vs mutated=8.00;
      seed=11: correct=1.02 vs mutated=42.98 -- the wrong denominator
      sometimes reads MORE converged than the right one, not merely
      differently converged.
    * kappa 2.0-4.5 (seed=6, prior_mean=0.0 fixed): the split holds for
      every kappa from 3.25 through 4.5 but fails one grid point below, at
      3.0 -- a real but narrow band, and only demonstrated within this one
      seed.
    * prior_mean (kappa=3.5, seed=6 fixed, via a throwaway variant of
      ``radiometer`` with the prior mean exposed -- not a committed
      fixture): ONLY the exact default 0.0 works; 0.01 already breaks it.
      Mechanism: a zero-mean prior makes the prediction AT the prior mean
      exactly zero, so the FIRST sigma estimate (evaluated there) collapses
      to a uniform floor everywhere -- a degenerate, unusually large first
      step, which is what manufactures the separation this pin relies on.
      Measured: ``tree_norm(first)`` drops from 6.390 at prior_mean=0.0 to
      4.7-4.8 for every other prior_mean tried, from -2.0 to 2.0, and none
      of those split at reweight_tol=0.75.

    So this pin works as committed and will keep working -- but it
    certifies one point, not a class of inputs. If ``radiometer``'s default
    seed or weight is ever changed, or a prior mean parameter is ever added
    and defaulted away from 0.0, these numbers need RE-MEASURING, not just
    re-running.
    """
    with jax.enable_x64(True):
        graph = radiometer(kappa=3.5, floor=1e-3)
        block = linear_operator(graph, ("w",), at={})
        result = iterative_gls(
            block,
            sigma_from_graph(graph, {}),
            tol=1e-14,
            min_reweights=1,
            reweight_tol=0.75,
            max_reweights=300,
        )
    assert int(result.iterations) == 3
    assert bool(result.converged)
    assert float(result.solution["w"]) == pytest.approx(3.915959928929826, rel=1e-9)


def test_the_returned_sigma_really_is_a_fixed_point():
    with jax.enable_x64(True):
        graph = radiometer()
        block = linear_operator(graph, ("w",), at={})
        sigma_of = sigma_from_graph(graph, {})
        result = iterative_gls(block, sigma_of, tol=1e-14, reweight_tol=1e-10)
        recomputed = sigma_of(result.solution)
    assert np.allclose(
        np.asarray(recomputed["d"]), np.asarray(result.noise_std["d"]), rtol=1e-6
    )


def test_check_prediction_dependence_catches_a_false_declaration():
    """`depends_on_prediction=False` on a radiometer node is a claim, and false.

    Declared False, a dispatcher skips the reweighting loop entirely and
    solves at whatever sigma the prior mean implies -- a confident answer at
    the wrong covariance, with nothing to notice.
    """
    with jax.enable_x64(True):
        graph = radiometer()
        block = linear_operator(graph, ("w",), at={})
        with pytest.raises(StructureError, match="depends_on_prediction"):
            check_prediction_dependence(
                block, sigma_from_graph(graph, {}), declared=False
            )


def test_check_prediction_dependence_accepts_a_true_declaration():
    with jax.enable_x64(True):
        graph = straight_line()
        block = linear_operator(graph, ("w",), at={})
        movement = check_prediction_dependence(
            block, sigma_from_graph(graph, {}), declared=False
        )
    assert movement == pytest.approx(0.0, abs=1e-12)


def test_a_capped_run_reports_converged_false_rather_than_pretending():
    """converged=False means the returned covariance is NOT a fixed point.

    Everything conditioned on it inherits that, so it is returned as a flag
    rather than raised -- but it must never read True.
    """
    with jax.enable_x64(True):
        graph = radiometer()
        block = linear_operator(graph, ("w",), at={})
        result = iterative_gls(
            block,
            sigma_from_graph(graph, {}),
            tol=1e-14,
            min_reweights=1,
            max_reweights=1,
        )
    assert not bool(result.converged)


def test_a_one_sided_probe_would_miss_a_clipped_sigma():
    from tests.exact.models import one_sided_sigma

    with jax.enable_x64(True):
        graph = one_sided_sigma()
        block = linear_operator(graph, ("w",), at={})
        with pytest.raises(StructureError, match="depends_on_prediction"):
            check_prediction_dependence(
                block, sigma_from_graph(graph, {}), declared=False
            )


# Structural-dimension audit (not in the plan): every fixture above is a
# single SCALAR latent constrained by a single observed node. `sigma_of`,
# `check_prediction_dependence`'s probe construction, and the reweighting
# loop's `tree_norm`/`jax.tree.map` plumbing are generic dict/pytree code
# with no explicit branch on leaf count or element count -- and Task 7 found
# two real bugs (`_split_like` sharing a key across leaves, `omega_prior`
# collapsing a plate to one draw) hiding behind exactly that genericity on a
# test set that never varied those dimensions. The two fixtures below give
# "more than one latent leaf", "more than one observed leaf", and "one leaf
# with more than one element" each at least one exercise through the actual
# reweighting loop, checked against the same independent dense oracle the
# plan's own test uses.

N1_GROUP, M2_GROUP = 9, 6
KAPPA_GROUP, FLOOR_GROUP, S2_GROUP = 0.04, 2e-3, 0.25

N_PLATE, TAU_PLATE, KAPPA_PLATE, FLOOR_PLATE = 6, 3.0, 0.06, 2e-3


def test_iterative_gls_handles_a_multi_leaf_multi_observed_block():
    """Two latent leaves solved JOINTLY; two observed leaves, one
    prediction-dependent and one not.

    `radiometer_group`'s ``a``, ``b`` are solved together (like
    `two_linear_latents`), and its ``d1``/``d2`` are two observed nodes
    (like `two_observations`) -- but combined with a genuinely
    prediction-dependent sigma, which neither of those two constant-sigma
    fixtures has, so this is the first test to run the reweighting loop
    itself on more than one leaf on either side.

    `plated_radiometer` catches NEITHER of the leaf-count mutations this
    fixture exists for (see
    `test_iterative_gls_delta_denominator_ignores_extra_latent_leaves`
    below) -- correctly, not by omission: a plate is one leaf with several
    elements, not several leaves, so a mutation that restricts a
    computation to "the first leaf" is a no-op on a domain that has only
    one. The two fixtures exercise different structural dimensions and
    neither is redundant with the other.
    """
    from tests.exact.models import radiometer_group

    with jax.enable_x64(True):
        graph = radiometer_group(
            n=N1_GROUP, m=M2_GROUP, kappa=KAPPA_GROUP, floor=FLOOR_GROUP, s2=S2_GROUP
        )
        block = linear_operator(graph, ("a", "b"), at={})
        seam = sigma_from_graph(graph, {})
        result = iterative_gls(block, seam, tol=1e-14, reweight_tol=1e-10)
        oracle = graph_oracle(graph, ("a", "b"), at={})
        # The probe scans every observed leaf, not just the first -- d1
        # genuinely moves and d2 by construction does not.
        with pytest.raises(StructureError, match="depends_on_prediction"):
            check_prediction_dependence(block, seam, declared=False)

    design, offset, data = oracle.design, oracle.offset, oracle.data
    prior_precision = np.diag(1.0 / oracle.prior_std**2)
    x = oracle.prior_mean.copy()
    for _ in range(400):
        pred = design @ x + offset
        sigma = np.concatenate(
            [
                KAPPA_GROUP * np.abs(pred[:N1_GROUP]) + FLOOR_GROUP,
                np.full(pred.size - N1_GROUP, S2_GROUP),
            ]
        )
        noise_precision = np.diag(1.0 / sigma**2)
        x = np.linalg.solve(
            design.T @ noise_precision @ design + prior_precision,
            design.T @ noise_precision @ (data - offset)
            + prior_precision @ oracle.prior_mean,
        )

    assert bool(result.converged)
    assert np.allclose(flat_domain(result.solution, block.names), x, rtol=1e-6)


def test_iterative_gls_delta_denominator_ignores_extra_latent_leaves():
    """Regression pin for `step`'s ``change = jax.tree.map(jnp.subtract, ...)``.

    Restricting that tree_map to only the FIRST latent leaf (dropping every
    other leaf's contribution to `change` before it reaches `tree_norm`)
    leaves every other test in this file green, including
    `test_iterative_gls_handles_a_multi_leaf_multi_observed_block` -- same
    mechanism as the denominator pin above: at the default min_reweights=5,
    `radiometer_group` converges tightly enough that delta is never
    consulted while the mutation could still show.

    At ``min_reweights=1`` (delta consulted after one step, on
    `radiometer_group`'s exact defaults: n=9, m=6, a_true=1.5, b_true=-2.0,
    kappa=0.04, floor=2e-3, s2=0.25, seed=14): the FULL change norm gives
    delta_correct=0.01318; the change restricted to leaf ``a`` alone gives
    delta_mutated=0.00396. ``reweight_tol=0.008`` sits between them (margin
    0.0040 above the mutated value, 0.0052 below the correct one), so the
    correct code takes a 3rd step (``iterations=3``, ``a=1.50108,
    b=-2.00301``) while the mutated one stops at the 2nd (``iterations=2``,
    ``a=1.50112, b=-2.00307`` -- differing from the correct answer in the
    4th decimal).

    **Structurally stronger than the denominator pin, but still a point at
    this exact threshold -- measured, not assumed.** Unlike that pin, the
    ORDERING here (delta_mutated <= delta_correct) is not a numerical
    accident: `tree_norm` of a subset of a pytree's leaves cannot exceed
    `tree_norm` of the whole pytree (dropping a leaf only removes a
    non-negative term from the sum of squares), so `delta_mutated` can
    never exceed `delta_correct` for ANY input. Swept to confirm and to
    check whether reweight_tol=0.008 generalises past this one point:

    * seed 0-19 (kappa=0.04 fixed): the ordering holds at every seed (zero
      inversions, unlike the denominator pin's seed=8/11), but the fixed
      threshold reweight_tol=0.008 only splits 6 of 20 (3, 5, 12, 13, 14,
      15) -- the rest have BOTH deltas above or BOTH below 0.008, so the
      loop would stop at the same iteration count either way.
    * kappa 0.01-0.10 (seed=14 fixed): splits for kappa in roughly
      [0.03, 0.06] and fails outside that band in both directions (too
      small: both deltas land under the threshold; too large: both land
      over it) -- a real but narrow band, again only within one seed.

    So: the MECHANISM (restricting to one leaf can only understate
    convergence) is general and provably one-directional, but this specific
    ``reweight_tol=0.008`` pin certifies one fixture at its defaults, not a
    class of inputs. Re-measure if `radiometer_group`'s defaults change.
    """
    from tests.exact.models import radiometer_group

    with jax.enable_x64(True):
        graph = radiometer_group()
        block = linear_operator(graph, ("a", "b"), at={})
        result = iterative_gls(
            block,
            sigma_from_graph(graph, {}),
            tol=1e-14,
            min_reweights=1,
            reweight_tol=0.008,
            max_reweights=300,
        )
    assert int(result.iterations) == 3
    assert bool(result.converged)
    assert float(result.solution["a"]) == pytest.approx(1.5010813415358777, rel=1e-9)
    assert float(result.solution["b"]) == pytest.approx(-2.003009526038934, rel=1e-9)


def test_iterative_gls_handles_a_plated_member():
    """One latent leaf with several elements, rather than several scalar
    leaves.

    `radiometer`'s ``w`` is a scalar; `plated_radiometer`'s ``z`` is a plate
    of six, each element's own sigma tracking that SAME element's
    prediction -- so `tree_norm`'s reduction and the probe in
    `check_prediction_dependence` run on one array leaf here, never
    exercised by a fixture whose domain is a dict of scalars.
    """
    from tests.exact.models import plated_radiometer

    with jax.enable_x64(True):
        graph = plated_radiometer(
            n=N_PLATE, tau=TAU_PLATE, kappa=KAPPA_PLATE, floor=FLOOR_PLATE
        )
        block = linear_operator(graph, ("z",), at={})
        result = iterative_gls(
            block, sigma_from_graph(graph, {}), tol=1e-14, reweight_tol=1e-10
        )
        oracle = graph_oracle(graph, ("z",), at={})

    design, offset, data = oracle.design, oracle.offset, oracle.data
    prior_precision = np.diag(1.0 / oracle.prior_std**2)
    x = oracle.prior_mean.copy()
    for _ in range(400):
        sigma = KAPPA_PLATE * np.abs(design @ x + offset) + FLOOR_PLATE
        noise_precision = np.diag(1.0 / sigma**2)
        x = np.linalg.solve(
            design.T @ noise_precision @ design + prior_precision,
            design.T @ noise_precision @ (data - offset)
            + prior_precision @ oracle.prior_mean,
        )

    assert bool(result.converged)
    assert np.allclose(flat_domain(result.solution, block.names), x, rtol=1e-6)


def test_iterative_gls_refuses_min_reweights_above_max():
    """The loop caps at max_reweights either way, so min > max would
    silently deliver fewer steps than asked for -- refused up front rather
    than discovered from a suspiciously low `iterations`.
    """
    with jax.enable_x64(True):
        graph = radiometer()
        block = linear_operator(graph, ("w",), at={})
        with pytest.raises(GraphError, match="min_reweights"):
            iterative_gls(
                block,
                sigma_from_graph(graph, {}),
                min_reweights=5,
                max_reweights=2,
            )
