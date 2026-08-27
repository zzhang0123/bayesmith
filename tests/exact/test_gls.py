"""Finding the covariance a prediction-dependent sigma implies."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.errors import GraphError, StructureError
from bayesmith.exact.block import unchecked_operator
from bayesmith.exact.gls import (
    check_prediction_dependence,
    iterative_gls,
    sigma_from_graph,
)
from bayesmith.exact.linearity import linear_operator
from bayesmith.exact.precision import diagonal_from
from bayesmith.exact.solve import wiener_solve
from tests.exact.models import (
    contrast_sigma_pair,
    element_contrast_sigma_plate,
    one_sided_sigma,
    radiometer,
    sigma_functional_block,
    straight_line,
    sum_sigma_pair,
    two_linear_latents,
)
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
            precision=diagonal_from(
                sigma_from_graph(graph, {})({"w": jnp.asarray(0.0)})
            ),
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


def test_check_prediction_dependence_accepts_a_correct_false_declaration():
    """Renamed from `..._accepts_a_true_declaration`, which it never was.

    The old name claimed to exercise `declared=True`; the body has always
    called with `declared=False`, on `straight_line()` -- a fixture where
    that declaration happens to be accurate (a constant sigma, so movement
    is ~0). The name read as covering the OTHER half of the contract, and
    did not -- see `test_check_prediction_dependence_never_raises_for_a_
    declared_true_node` below for the test that actually does.
    """
    with jax.enable_x64(True):
        graph = straight_line()
        block = linear_operator(graph, ("w",), at={})
        movement = check_prediction_dependence(
            block, sigma_from_graph(graph, {}), declared=False
        )
    assert movement == pytest.approx(0.0, abs=1e-12)


def test_check_prediction_dependence_never_raises_for_a_declared_true_node():
    """The other half of the contract: `declared=True` must NEVER raise.

    Not "raises only above rtol" -- never, however large the measured
    movement is. That is what lets a caller invoke this unconditionally
    before deciding whether to run the reweighting loop: only `declared=
    False` combined with real movement is refused.

    Until this test existed, `declared=True` was never called anywhere in
    this suite -- every call site here passes `declared=False`, including
    the sibling above, whose old name
    (`..._accepts_a_true_declaration`) read as though it covered this and
    did not. Measured: deleting the `not declared and` clause of the guard
    (`if not declared and movement > rtol:` -> `if movement > rtol:`) left
    the entire pre-existing 240-test suite green -- a regression that made
    `declared=True` raise on a genuinely prediction-dependent node, exactly
    backwards from the contract, would have gone undetected.

    `radiometer()` rather than `straight_line()`: its sigma tracks the
    prediction directly, so the measured movement is genuinely large
    (2500, far above `rtol`'s default of 1e-8) -- this is not a vacuous
    zero-movement pass, unlike the sibling test above.
    """
    with jax.enable_x64(True):
        graph = radiometer()
        block = linear_operator(graph, ("w",), at={})
        movement = check_prediction_dependence(
            block, sigma_from_graph(graph, {}), declared=True
        )
    assert movement > 100.0


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


def test_sigma_depending_on_a_contrast_of_two_members_is_detected():
    """The probe must not travel one ray through the block's domain.

    Measured with the lockstep probe `centre + factor * prior_std`: movement
    came back exactly 0.0 -- not small, BITWISE zero -- because both members
    were displaced by the same signed multiple of equal prior widths, so
    `a - b` never changed and sigma is constant along that ray.

    No function here was crafted to have a root at the probe points. The ray
    simply never leaves the level set, which is why "try another magnitude"
    does not help: every magnitude is on the same ray.
    """
    graph = contrast_sigma_pair()
    block = unchecked_operator(graph, ["a", "b"])
    movement = check_prediction_dependence(
        block, sigma_from_graph(graph, {}), declared=True
    )
    assert movement > 1e-3, (
        f"sigma moves with a - b, but the probe measured {movement:.3e}"
    )


def test_a_genuinely_constant_sigma_still_measures_no_movement():
    """The two-sided half: richer probe directions must not invent movement.

    Without this, a 'fix' that reported a large movement unconditionally
    would pass the test above and route every model through the correction
    machinery it does not need.
    """
    graph = two_linear_latents()
    block = unchecked_operator(graph, ["a", "b"])
    movement = check_prediction_dependence(
        block, sigma_from_graph(graph, {}), declared=True
    )
    assert movement == pytest.approx(0.0, abs=1e-12)


def test_the_measured_movement_does_not_depend_on_the_member_order():
    """`_dependence_probe` sorts, so the same block described two ways agrees.

    The ``random`` pattern folds each member's sub-key in by its POSITION,
    so the positions have to come from somewhere stable. `block.names` is
    whatever order the caller passed -- `unchecked_operator` stores it
    verbatim -- so building the probe from it makes a yes/no guard's verdict
    depend on how the member list was typed. Measured with
    `sorted(block.names)` swapped for `block.names`: ``["a", "b"]`` reads
    3.639e+00 and ``["b", "a"]`` reads 7.844e-01 -- same block, same graph,
    two different numbers for a dispatcher to read. (Under the deterministic
    ``alternating`` pattern this fix replaced, the same mutation read
    6.389e+00 against 1.718e+00; the sorting is load-bearing for either
    reason, which is why this test outlived the pattern it was written for.)

    **The declaration order in the GRAPH is NOT the axis here**, unlike
    `two_observations_reverse_sorted_names` and `two_unusable_observed_scales`
    (whose dicts are built by comprehension over `graph.observed`). Nothing
    in this path consults it: `unchecked_operator` takes `names` from the
    caller and `_validated_names` returns `tuple(names)` unchanged, so a
    fixture that merely declared `b` before `a` would leave this mutation a
    no-op. Reversing the CALLER's list is what makes it visible -- measured
    both ways.
    """
    graph = contrast_sigma_pair()
    seam = sigma_from_graph(graph, {})
    declared = check_prediction_dependence(
        unchecked_operator(graph, ["a", "b"]), seam, declared=True
    )
    reversed_ = check_prediction_dependence(
        unchecked_operator(graph, ["b", "a"]), seam, declared=True
    )
    assert declared == pytest.approx(reversed_, rel=1e-12, abs=0.0)
    assert declared > 1e-3


def test_sigma_depending_on_a_sum_of_two_members_is_still_detected():
    """`uniform`'s own guard, and what it pins is a FLOOR, not a detection.

    Found by mutation rather than predicted, twice over.

    **First** (before the random directions existed): dropping ``uniform``
    from `DEPENDENCE_PATTERNS` left the ENTIRE suite green at 270 tests --
    every other prediction-dependent-sigma fixture here is either a
    ONE-member block, where the patterns build bitwise identical probes, or
    `radiometer_group`, whose sigma tracks ``mu = a*x_i + b`` elementwise so
    any displacement moves it. `sum_sigma_pair` was the missing region: sigma
    moves along the LOCKSTEP ray, which the then-second pattern held exactly
    constant.

    **Second** (after them): ``DEPENDENCE_PATTERNS = ("random",)`` ALSO left
    the whole suite green, this test included -- a random direction detects
    a sum with probability 1, so `assert movement > 1e-3` cannot separate
    them. That is a fixture-does-not-reach-the-region failure of the
    assertion, not of the fixture, and this is the assertion that fixes it.

    What ``uniform`` actually buys is the SIZE of the returned number, which
    is what a dispatcher thresholds. On a sum of equal-width members it reads
    exactly ``expm1(factor * members)`` with no key involved -- 6.389 here.
    ``random`` alone reads whatever ``sum(z_i)`` was: measured over 200 keys,
    **1.342e-01 to 4.419e+01** on this fixture (8.399e-01 at the default
    key, 7.6x below the anchor), and **1.565e-02 to 1.429e+02** on the
    three-member sum, the low end falling as the block widens and the draws
    cancel. So the assertion is a FLOOR at the anchor's key-free value: it
    passes on any probe set containing ``uniform`` and fails on any that
    drops it.
    """
    graph = sum_sigma_pair()
    block = unchecked_operator(graph, ["a", "b"])
    movement = check_prediction_dependence(
        block, sigma_from_graph(graph, {}), declared=True
    )
    # `>=` rather than `approx`: a random probe that happened to exceed the
    # anchor would be a better measurement, not a regression, and must not
    # turn this red.
    assert movement >= math.expm1(2.0) * (1.0 - 1e-6), (
        f"the uniform anchor guarantees {math.expm1(2.0):.6f} on a sum of two "
        f"equal-width members, but the probe measured {movement:.3e}"
    )


TRIPLE = ["a", "b", "c"]
QUAD = ["a", "b", "c", "d"]


def test_sigma_depending_on_a_contrast_of_two_same_parity_members_is_detected():
    """Three members, and the deterministic pair does not reach them.

    ``alternating`` flips sign with a member's POSITION, so it separates two
    positions only when their positions have DIFFERING PARITY. On three
    members with sigma tracking ``a - c`` -- sorted positions 0 and 2, the
    same sign under ``alternating`` and under ``uniform`` alike -- the
    contrast is constant along both rays and the probe reads bitwise 0.0.
    Same silent wrong answer as the two-member defect, one member up:
    whole-graph-one-block, so the dispatcher takes the iid-draws-no-chain
    row and there is no r-hat, no k-hat, no ESS to notice with.

    Measured on the two deterministic patterns, all four three-member rows
    (`sigma_functional_block`, float32):

    ==========================  =================
    sigma depends on            movement measured
    ==========================  =================
    ``a - b`` (positions 0,1)   6.389057e+00
    ``b - c`` (positions 1,2)   1.718282e+00
    ``a - c`` (positions 0,2)   **0.000000e+00**
    ``a + b + c`` (sum)         1.908554e+01
    ==========================  =================

    **This is the common case, not an edge case**: the dispatcher this guard
    feeds puts every qualified latent into ONE block, so three-or-more is the
    norm and two is the exception.
    """
    graph = sigma_functional_block(weights=(1.0, 0.0, -1.0))
    block = unchecked_operator(graph, TRIPLE)
    movement = check_prediction_dependence(
        block, sigma_from_graph(graph, {}), declared=True
    )
    assert movement > 1e-3, (
        f"sigma moves with a - c, but the probe measured {movement:.3e}"
    )


def test_sigma_depending_on_a_functional_no_sign_pattern_reaches_is_detected():
    """Four members, and NO deterministic sign pattern family reaches it.

    A probe pattern is a sign vector; sigma here is ``exp(f . theta)``, so
    what a pattern measures is ``|exp(factor * (signs . f)) - 1|`` and a
    pattern blind to ``f`` is exactly one whose sign vector is ORTHOGONAL to
    it. On ``f = a - b - c + d`` -- the third Hadamard row -- measured dot
    products against every deterministic pattern proposed for this guard:

    ==================================  =====
    pattern                             dot
    ==================================  =====
    ``uniform``   ``(+,+,+,+)``         0.0
    ``alternating`` = counter bit 0     0.0
    counter bit 1 ``(+,+,-,-)``         0.0
    ==================================  =====

    So this fixture read bitwise 0.0 on the shipped pair AND would read
    bitwise 0.0 on the binary-counter family the previous docstring named as
    "the deterministic family that closes it in general" -- **that claim is
    false, and this test is the measurement that refutes it**. The counter
    separates PAIRS of positions; it spans only ``1 + ceil(log2 members)``
    of the block's ``members`` directions, and any functional orthogonal to
    all of them is invisible to every member of the family. Four members is
    the smallest block where the counter has fewer patterns than dimensions,
    so it is the smallest block where such a functional exists at all.

    A random direction has no such subspace to miss: it is orthogonal to a
    fixed non-zero ``f`` with probability zero.
    """
    graph = sigma_functional_block(weights=(1.0, -1.0, -1.0, 1.0))
    block = unchecked_operator(graph, QUAD)
    movement = check_prediction_dependence(
        block, sigma_from_graph(graph, {}), declared=True
    )
    assert movement > 1e-3, (
        f"sigma moves with a - b - c + d, but the probe measured {movement:.3e}"
    )


def test_a_genuinely_constant_sigma_on_a_wide_block_still_measures_no_movement():
    """The two-sided arm, at the width where the random directions live.

    `test_a_genuinely_constant_sigma_still_measures_no_movement` above pins
    this on two members. Random probe directions are drawn per member, so a
    plumbing error that leaked a draw into the measured number would show up
    as block width grows while the two-member case stayed clean. Same
    fixture family as the two detection tests above, with the functional set
    to zero: sigma is ``base * exp(0)``, a constant, at every probe point.

    Without this arm, a "fix" that reported a large movement unconditionally
    would pass both detection tests and route every model through the
    correction machinery it does not need.
    """
    graph = sigma_functional_block(weights=(0.0, 0.0, 0.0, 0.0))
    block = unchecked_operator(graph, QUAD)
    movement = check_prediction_dependence(
        block, sigma_from_graph(graph, {}), declared=True
    )
    assert movement == pytest.approx(0.0, abs=0.0)


def test_the_dependence_probe_is_reproducible_and_steerable():
    """Default key: same number twice. Explicit key: a different probe.

    A yes/no guard whose verdict changes run to run is not a guard, so the
    default has to be a fixed key -- `check_linearity`'s contract, and the
    reason the random directions are drawn from ``jax.random.key(0)`` rather
    than from entropy. The second half is what stops the ``key`` argument
    from being decorative: a caller who wants a second opinion must get a
    genuinely different probe, and one that still detects.
    """
    graph = sigma_functional_block(weights=(1.0, 0.0, -1.0))
    seam = sigma_from_graph(graph, {})
    once = check_prediction_dependence(
        unchecked_operator(graph, TRIPLE), seam, declared=True
    )
    twice = check_prediction_dependence(
        unchecked_operator(graph, TRIPLE), seam, declared=True
    )
    assert once == twice
    other = check_prediction_dependence(
        unchecked_operator(graph, TRIPLE),
        seam,
        declared=True,
        key=jax.random.key(17),
    )
    assert other != once
    assert other > 1e-3


def test_the_measured_movement_ignores_member_order_at_three_members():
    """The `sorted` is load-bearing for the RANDOM directions too.

    Sub-keys are folded in by position in the SORTED names, exactly as
    `check_linearity` does, so the same block described two ways draws the
    same displacement for the same member. `block.names` is the CALLER's
    order -- `_validated_names` returns ``tuple(names)`` unchanged and
    `unchecked_operator` stores it verbatim -- so permuting the caller's
    list, not the graph's declaration order, is what exercises this;
    reversing a declaration would leave the mutation a no-op.

    Three members rather than two: with two, a reversal maps position 0 to
    position 1 and back, and a probe that folded in by the CALLER's position
    would still be caught -- but a bug that indexed the sub-keys by a
    stable-but-wrong key (say the same sub-key for every member) survives a
    two-member reversal in some layouts. Three members with a rotation
    exercises a permutation that is not an involution.
    """
    graph = sigma_functional_block(weights=(1.0, 0.0, -1.0))
    seam = sigma_from_graph(graph, {})
    declared = check_prediction_dependence(
        unchecked_operator(graph, TRIPLE), seam, declared=True
    )
    rotated = check_prediction_dependence(
        unchecked_operator(graph, ["c", "a", "b"]), seam, declared=True
    )
    reversed_ = check_prediction_dependence(
        unchecked_operator(graph, ["c", "b", "a"]), seam, declared=True
    )
    assert declared == pytest.approx(rotated, rel=1e-12, abs=0.0)
    assert declared == pytest.approx(reversed_, rel=1e-12, abs=0.0)
    assert declared > 1e-3


def test_sigma_depending_on_a_contrast_between_plate_elements_is_detected():
    """The block's dimension is its ELEMENT count, not its member count.

    `check_linearity` draws its probe per element and this guard now matches
    it, which is what puts a plate's interior inside the probed space. A
    per-MEMBER scalar draw would displace every entry of a leaf by the same
    amount, so the whole class of "sigma depends on a contrast between two
    entries of one array" stays on its level set -- the `contrast_sigma_pair`
    defect one structural level down, and equally silent: measured **bitwise
    0.0** per member against **1.730645e+01** per element.

    `plated_radiometer` does NOT reach this region and is not redundant with
    it: its ``sigma_i = kappa|z_i| + floor`` depends on the element it scales,
    so a uniform displacement already moves it. Only a dependence that is
    CONSTANT along the leaf's diagonal separates the two draws.
    """
    graph = element_contrast_sigma_plate()
    block = unchecked_operator(graph, ["z"])
    movement = check_prediction_dependence(
        block, sigma_from_graph(graph, {}), declared=True
    )
    assert movement > 1e-3, (
        f"sigma moves with z[0] - z[1], but the probe measured {movement:.3e}"
    )


ANCHOR_KEYS = 64


def test_a_clipped_sigma_is_detected_at_every_key_because_of_the_anchor():
    """`uniform` is load-bearing for DETECTION, not only for the number.

    `one_sided_sigma`'s ``sigma = kappa * max(mu, 0) + floor`` is exactly
    constant on the whole half-space ``mu <= 0``, so a probe set that lands
    entirely inside it reads bitwise 0.0 -- which is what
    :data:`~bayesmith.exact.gls.DEPENDENCE_PROBES`' two unequal SIGNED
    magnitudes exist to prevent. A random direction throws that guarantee
    away: it multiplies each magnitude by its own draw, so both probes land
    on the clipped side whenever the two draws have the wrong signs.

    Measured, ``DEPENDENCE_PATTERNS = ("random",)`` over 400 keys on this
    fixture: **105 of them -- 26% -- read bitwise 0.0**, i.e. one key in four
    would have let ``depends_on_prediction=False`` through on a genuinely
    prediction-dependent node. `test_a_one_sided_probe_would_miss_a_clipped_
    sigma` does not catch that; it runs at the default key, which happens to
    be one of the 74% that work.

    With ``uniform`` in the set the failure mode is gone by construction, and
    this is the measurement: over 64 keys the movement's MINIMUM is
    6.000000e+01 -- the anchor's own key-free reading, ``kappa * max`` at
    ``+1`` prior width over the floor -- and never once the random probes'
    contribution. That is what "deterministic anchor" buys.
    """
    graph = one_sided_sigma()
    block = unchecked_operator(graph, ["w"])
    seam = sigma_from_graph(graph, {})
    movements = [
        check_prediction_dependence(
            block, seam, declared=True, key=jax.random.key(seed)
        )
        for seed in range(ANCHOR_KEYS)
    ]
    assert min(movements) >= 60.0 * (1.0 - 1e-6), (
        "the uniform anchor guarantees 60.0 on this clipped sigma at every "
        f"key, but some key measured {min(movements):.3e}"
    )


@pytest.mark.parametrize("kappa", [0.05, 0.2, 0.5, 1.0])
def test_the_fixed_point_is_the_unbiased_estimator_not_the_gls_biased_one(kappa):
    """WHICH estimator the reweighting converges to, pinned by name.

    rheplicant's ``inference/noise.py`` documents a closed form for exactly
    this model: dropping the log-determinant from the Gaussian density gives
    "a *different estimator*", one that "returns ``sum d^2 / sum d``, biased
    high by ``(1 + f^2)``". It is easy to read that as a statement about
    generalized least squares in general, and therefore about this function,
    and it is not.

    The bias belongs to the objective that DIFFERENTIATES THROUGH sigma while
    dropping ``log sigma``. :func:`iterative_gls` does neither: it freezes
    sigma per inner solve and recomputes it afterwards, so each solve is an
    ordinary weighted least squares and the fixed point satisfies
    ``w = mean(u)``, ``u = d / x`` -- the same answer the FULL density gives,
    not the biased one. Reduce the linear model with ``u = d / x``: sigma
    ``= f w x`` makes the weighted normal equations collapse to ``sum(u) / n``
    with every weight identical.

    Measured here, distance from the fixed point to each candidate:

    =====  ==========  ==============  ==========
    kappa  |w-mean(u)|  |w-sum u^2/u|   ratio
    =====  ==========  ==============  ==========
    0.05   4.16e-05    5.32e-03        128x
    0.2    1.17e-03    8.32e-02        71x
    0.5    8.53e-03    4.97e-01        58x
    1.0    4.17e-02    1.85e+00        44x
    =====  ==========  ==============  ==========

    The residual gap to ``mean(u)`` is the ``w ~ Normal(0, 10)`` prior, and it
    grows with kappa because the data gets weaker -- direction and ordering
    both as they should be, which is why this asserts a RATIO of distances
    rather than an absolute tolerance that would have to absorb the prior.

    **Why this is worth a test rather than a comment.** The migration spec's
    first draft asked for the opposite: that this frozen-sigma path differ
    from a live-sigma path by ``(1 + f^2)``. It does not, and a test written
    to that specification would have been satisfied only by pulling a correct
    estimator onto the biased side. ``tests/crosscheck/test_noise_logdet.py``
    carries the other half -- that rheplicant's log-det-dropped likelihood
    really is ``(1 + f^2)`` high -- so the divergence between the two packages
    is recorded on both sides rather than in neither.
    """
    with jax.enable_x64(True):
        graph = radiometer(kappa=kappa, floor=FLOOR)
        block = linear_operator(graph, ("w",), at={})
        result = iterative_gls(
            block,
            sigma_from_graph(graph, {}),
            tol=1e-14,
            reweight_tol=1e-12,
            max_reweights=400,
        )
        oracle = graph_oracle(graph, ("w",), at={})
        found = float(result.solution["w"])
        u = (np.asarray(oracle.data) - np.asarray(oracle.offset)) / np.asarray(
            oracle.design
        )[:, 0]

    assert bool(result.converged)
    unbiased = float(u.mean())
    biased = float((u**2).sum() / u.sum())
    # The two candidates must actually be far apart, or "closer to one" is
    # not a claim. At kappa=0.05 they differ by only 0.17%, which is still
    # 128x the distance being resolved.
    assert biased > unbiased, (biased, unbiased)
    assert abs(found - unbiased) < 0.05 * abs(found - biased), (
        f"kappa={kappa}: fixed point {found} sits {abs(found - unbiased):.3e} "
        f"from mean(u)={unbiased} and {abs(found - biased):.3e} from "
        f"sum u^2/sum u={biased} -- it has moved toward the log-det-dropped "
        "estimator, which this function's freezing is what avoids."
    )


# ---------------------------------------------------------------------------
# B9 step 4: the boundary where sigma VALUES become the operator, once.
# ---------------------------------------------------------------------------


def test_gls_result_precision_solves_to_the_same_point_its_noise_std_does():
    """`.precision` is the operator `.noise_std` describes, and it is derived.

    The property exists so the conversion happens ONCE, where the answer is
    handed on, rather than at each of the four consumers that take an
    operator. Two things have to hold for that to be safe:

    * it agrees with converting `.noise_std` by hand -- bitwise, since it IS
      that conversion;
    * a solve at it lands where a solve at the sigma lands, which is the
      claim any caller actually relies on.

    Both, because the first alone would still pass if `diagonal_from` were
    wrong and the second alone would not notice a second stored copy drifting.
    """
    with jax.enable_x64(True):
        graph = radiometer(kappa=KAPPA, floor=FLOOR)
        block = linear_operator(graph, ("w",), at={})
        result = iterative_gls(
            block, sigma_from_graph(graph, {}), tol=1e-14, reweight_tol=1e-10
        )
        by_hand = diagonal_from(result.noise_std)
        from_property = result.precision
        assert set(by_hand) == set(from_property)
        for name in by_hand:
            assert jnp.array_equal(from_property[name].sigma, by_hand[name].sigma)
        direct, _ = wiener_solve(
            block, precision=from_property, tol=1e-14, require_convergence=None
        )
    assert float(direct["w"]) == pytest.approx(float(result.solution["w"]), rel=1e-10)


def test_gls_result_stores_one_covariance_and_derives_the_other_spelling():
    """ONE covariance in the result, and which one is stored is not arbitrary.

    This test asserted the opposite arrangement one increment ago --
    ``noise_std`` the field, ``precision`` a property -- for a reason that
    still holds: ``GLSResult`` is a ``NamedTuple`` and therefore a pytree
    whose leaves are its fields, so storing both would put two copies of one
    covariance inside every traced result, and two copies of one covariance is
    defect B1's shape.

    What changed is WHICH way round it can be. A correlated result has no
    per-sample sigma to derive an operator from, so ``noise_std`` cannot be
    the stored one; every result has an operator to derive a sigma from when
    one exists. The invariant survived the increment; the direction did not.

    Measured as a leaf COUNT rather than by inspecting the class, because what
    matters is what ``jax.tree`` sees: a second field would show up here
    whatever it was called.
    """
    with jax.enable_x64(True):
        graph = straight_line()
        block = linear_operator(graph, ("w",), at={})
        result = iterative_gls(
            block, sigma_from_graph(graph, {}), depends_on_prediction=False, tol=1e-14
        )
        assert "precision" in result._fields
        assert "noise_std" not in result._fields
        assert len(jax.tree.leaves(result)) == len(jax.tree.leaves(tuple(result)))

        # ...and the derived spelling is the arrays themselves, BITWISE --
        # `per_sample_sigma` returns what `diagonal_from` was handed, so a
        # caller reading `.noise_std` gets its own numbers back rather than a
        # round trip through the operator.
        from bayesmith.exact.block import domain_centre

        handed = sigma_from_graph(graph, {})(domain_centre(block))
        assert set(result.noise_std) == set(handed)
        for name in handed:
            assert jnp.array_equal(result.noise_std[name], handed[name])


def test_a_correlated_gls_result_reports_no_per_sample_sigma():
    """``None``, not per-mode amplitudes wearing the wrong name.

    A stationary covariance has an n-point kernel and no per-sample sigma.
    ``sqrt(lambda_k)`` exists and is the natural per-MODE amplitude, but
    reporting it as ``noise_std`` would be a lie by naming -- a caller feeding
    it to ``noise_std_at``-shaped code would get silence, not an error.
    """
    import numpy as np
    import numpyro.distributions as ndist

    from bayesmith import const, det, observe, sample, trace
    from bayesmith.exact.block import unchecked_operator
    from bayesmith.exact.gls import precision_from_graph
    from bayesmith.exact.precision import CirculantPrecision

    size = 8
    lag = np.minimum(np.arange(size), size - np.arange(size))
    kernel = jnp.asarray(1.0 * 0.4**lag + 0.5)
    grid = jnp.linspace(1.0, 4.0, size)

    def model():
        xs = const("X", grid)
        w = sample("w", lambda: ndist.Normal(0.0, 5.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe(
            "d",
            lambda m: ndist.CirculantNormal(m, kernel),
            mu,
            depends_on_prediction=False,
            obs=2.0 * grid,
        )

    with jax.enable_x64(True):
        graph = trace(model)
        block = unchecked_operator(graph, ("w",), {})
        result = iterative_gls(
            block,
            precision_of=precision_from_graph(graph, {}),
            depends_on_prediction=False,
            tol=1e-14,
        )
        assert result.noise_std is None
        assert isinstance(result.precision["d"], CirculantPrecision)
        assert jnp.allclose(result.precision["d"].first_column, kernel)


def test_iterative_gls_needs_exactly_one_spelling_of_the_rule():
    """Neither leaves it with no covariance; both is two chances to differ."""
    with jax.enable_x64(True):
        graph = straight_line()
        block = linear_operator(graph, ("w",), at={})
        with pytest.raises(ValueError, match="exactly one of"):
            iterative_gls(block, depends_on_prediction=False, tol=1e-14)
        with pytest.raises(ValueError, match="exactly one of"):
            iterative_gls(
                block,
                sigma_from_graph(graph, {}),
                precision_of=lambda x: diagonal_from(sigma_from_graph(graph, {})(x)),
                depends_on_prediction=False,
                tol=1e-14,
            )


def test_a_correlated_prediction_dependent_model_finds_the_same_fixed_point():
    """The composition of B9's last two increments, against a dense loop.

    `45198f9` refused a correlated node claiming `depends_on_prediction=True`;
    the spectral form of the variance-information term deleted that refusal,
    and increment 5 opened the graph path. This is the first fixture that
    needs BOTH, and it is checked against a NumPy fixed-point loop that shares
    no code with the JAX `while_loop`: solve at the current covariance,
    recompute the covariance, repeat.

    The kernel carries a FLOOR, and it is load-bearing rather than decorative.
    Without one the amplitude is `0.04 * mean(mu)**2`, which is exactly zero
    at the block's own zero -- so every eigenvalue vanishes there and
    `_refuse_unusable_noise` refuses the whole block with "smallest eigenvalue
    0". Measured: that model classifies to NUTS, and correctly. The suite's
    radiometer fixtures carry a floor for the same reason.

    Measured: 1.1e-16 relative, the JAX loop against seven NumPy iterations.
    """
    import numpy as np
    import numpyro.distributions as ndist

    from bayesmith import const, det, observe, sample, trace
    from bayesmith.dispatch.plan import compile as compile_graph

    size, prior_std, prior_mean, floor = 8, 5.0, 1.0, 0.05
    lag = np.minimum(np.arange(size), size - np.arange(size))
    grid = np.linspace(1.0, 4.0, size)
    data = 2.0 * grid

    def amplitude(mean_prediction):
        return 0.04 * mean_prediction**2 + floor

    def model():
        xs = const("X", jnp.asarray(grid))
        w = sample("w", lambda: ndist.Normal(prior_mean, prior_std))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))

        def kernel(prediction):
            amp = amplitude(jnp.mean(prediction))
            return amp * (0.4**lag) + 0.25 * amp

        observe(
            "d",
            lambda m: ndist.CirculantNormal(m, kernel(m)),
            mu,
            depends_on_prediction=True,
            obs=jnp.asarray(data),
        )

    with jax.enable_x64(True):
        plan = compile_graph(trace(model))
        assert plan.blocks[0].method == "gcr+snis", plan.blocks[0].method
        estimate = plan.estimate()
        got = float(np.asarray(estimate.values["w"]).reshape(()))
        # the reweighting really did converge to a CORRELATED covariance
        from bayesmith.exact.precision import CirculantPrecision

        assert isinstance(estimate.precision["d"], CirculantPrecision)
        assert estimate.noise_std is None

    def circulant(column):
        return np.array(
            [[column[(j - i) % size] for j in range(size)] for i in range(size)]
        )

    design = grid.reshape(-1, 1)
    point = prior_mean
    for iteration in range(200):
        amp = amplitude((point * grid).mean())
        inverse = np.linalg.inv(circulant(amp * (0.4**lag) + 0.25 * amp))
        normal = design.T @ inverse @ design + np.eye(1) / prior_std**2
        rhs = (
            design.T @ inverse @ data.reshape(-1, 1)
            + np.array([[prior_mean]]) / prior_std**2
        )
        moved = np.linalg.solve(normal, rhs).item()
        if abs(moved - point) < 1e-14 * max(abs(moved), 1e-30):
            point = moved
            break
        point = moved
    assert iteration < 199, "the NumPy reference did not converge"
    assert got == pytest.approx(point, rel=1e-12)


class TestD19sSubCaseIsREFUSEDRatherThanDegenerate:
    """The migration ledger's D19 fixture, measured -- and it does not do what
    D19 says it does.

    D19 records the premise this way: "bayesmith starts from sigma at the
    PRIOR CENTRE; with a zero prior centre and RadiometerNoise(floor=0) the
    starting sigma is 0 and the first solve is degenerate -- rheplicant, which
    starts from the data, has no such problem." It then rules for a
    data-anchored start, and adds that the zero-centre/floor-free sub-case
    "is a regression fixture from now on".

    **Measured here: the degenerate first solve never happens.** The block is
    refused before any solve, by ``_refuse_unusable_noise``, which names the
    scale expression as the fault and says to add a floor. And the refusal is
    not about the prior centre at all: ``floor=0`` is refused for a centre of
    2.5 as well, where sigma at the centre is 0.125, because
    ``check_linearity`` sweeps probes through the point where the prediction
    crosses zero and reads the covariance there.

    So the sub-case IS a regression fixture, as D19 asks -- but for the
    REFUSAL, not for a solve. Pinned here so Wave B reads a measurement rather
    than re-deriving one from the ledger's premise. Whether the starting point
    should move at all is then a pure numerical-continuity question about
    ``iterations``/``delta``/``converged``, answerable only against
    rheplicant's own pinned numbers, which do not exist on this side.

    (This package already knew: ``test_a_correlated_prediction_dependent_
    model_finds_the_same_fixed_point``'s docstring records the same mechanism
    for a correlated kernel, and says such a model "classifies to NUTS, and
    correctly".)
    """

    @staticmethod
    def _radiometer(*, floor, prior_mean, kappa=0.05, n=10):
        import numpyro.distributions as ndist

        from bayesmith import const, det, observe, sample, trace

        x = jnp.linspace(1.0, 5.0, n)
        data = 3.0 * x * (
            1.0 + kappa * jax.random.normal(jax.random.key(6), (n,))
        )

        def model():
            xs = const("X", x)
            w = sample("w", lambda: ndist.Normal(prior_mean, 10.0))
            mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
            observe(
                "d",
                lambda m: ndist.Normal(m, kappa * jnp.abs(m) + floor),
                mu,
                depends_on_prediction=True,
                obs=data,
            )

        return trace(model)

    @pytest.mark.parametrize("prior_mean", [0.0, 2.5])
    def test_a_floor_free_radiometer_is_refused_at_block_construction(
        self, prior_mean
    ):
        """Both centres, because the refusal is about the PROBE's sweep and
        not about where the prior sits -- which is the half of this that D19's
        wording does not have."""
        from bayesmith.errors import StructureError
        from bayesmith.exact.linearity import linear_operator

        graph = self._radiometer(floor=0.0, prior_mean=prior_mean)
        with pytest.raises(StructureError, match="positive definite"):
            linear_operator(graph, ("w",), at={})

    def test_the_refusal_names_the_scale_expression_and_the_remedy(self):
        """A refusal that did not say what to do is where a caller invents a
        starting point of their own, which is the failure D19 was reaching
        for."""
        from bayesmith.errors import StructureError
        from bayesmith.exact.linearity import linear_operator

        graph = self._radiometer(floor=0.0, prior_mean=0.0)
        with pytest.raises(StructureError) as caught:
            linear_operator(graph, ("w",), at={})
        message = str(caught.value)
        assert "scale expression" in message
        assert "floor" in message

    def test_the_SAME_model_with_a_floor_solves_from_the_prior_centre(self):
        """The other direction, and it is what makes the refusal a boundary
        rather than a wall: with a floor, the zero-centre case the ledger
        worried about converges from the prior centre with no trouble at all.

        Measured: w = 2.94668, converged, 5 reweights -- from a centre of 0.0,
        which is the exact configuration D19 describes as degenerate.
        """
        from bayesmith.exact.linearity import linear_operator

        graph = self._radiometer(floor=1e-3, prior_mean=0.0)
        block = linear_operator(graph, ("w",), at={})
        result = iterative_gls(
            block,
            sigma_of=sigma_from_graph(graph, {}),
            depends_on_prediction=True,
        )
        assert bool(result.converged)
        assert float(result.solution["w"]) == pytest.approx(2.94668, rel=1e-4)
