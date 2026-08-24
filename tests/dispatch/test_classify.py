"""The structural gate: what every fixture classifies as, and why.

Zero MCMC, zero statistics, fully deterministic. Every expectation below was
COMPUTED against the real `_ancestors` / `check_linearity` /
`check_prediction_dependence` before being written down -- the first draft of
this table had two wrong rows (`unconstrained_latent` and `shared_ancestor`),
both of which came from asserting rather than measuring. The derivation was
re-run independently of `bayesmith.dispatch.classify` for this commit, in a
throwaway script that calls those three primitives directly, and every row
the plan already carried came back unchanged.
"""

import jax
import pytest

from bayesmith.dispatch.classify import SIGMA_RTOL, partition
from bayesmith.errors import StructureError
from bayesmith.exact.linearity import DEFAULT_AT_POINTS
from tests.exact.models import (
    affine_only_at_zero,
    bilinear_pair,
    collinear_pair,
    cubic_tail,
    dangling_deterministic,
    diamond_ancestor,
    improper_outside_prior,
    indirect_ancestor,
    lying_observed_node,
    mixed_radiometer,
    observation_reused_downstream,
    orphaned_child_latent,
    overflowing_outside_latent,
    plated_and_scalar_latents,
    plated_latent,
    plated_radiometer,
    quadratic_claim,
    radiometer,
    radiometer_group,
    shared_ancestor,
    sigma_functional_block,
    student_t_likelihood,
    three_latent_chain,
    two_linear_latents,
    two_observations_reverse_sorted_names,
    unconstrained_latent,
)


def three_member_constant_sigma():
    """Three qualified members in ONE block, sigma bitwise constant.

    `sigma_functional_block` takes a required keyword, so it cannot be a
    parametrize entry directly. Wrapped rather than added to `models.py` as a
    fourth near-duplicate: the fixture already covers this shape, and the
    only thing missing was a zero-argument spelling of it.
    """
    return sigma_functional_block(weights=(0.0, 0.0, 0.0))


def three_member_moving_sigma():
    """Three qualified members in ONE block, sigma tracking a contrast."""
    return sigma_functional_block(weights=(1.0, -1.0, 0.0))


@pytest.mark.parametrize(
    "build, exact, nuts, method",
    [
        pytest.param(two_linear_latents, ("a", "b"), (), "gcr", id="two_linear"),
        pytest.param(radiometer, ("w",), (), "gcr+snis", id="radiometer"),
        pytest.param(radiometer_group, ("a", "b"), (), "gcr+snis", id="rad_group"),
        pytest.param(collinear_pair, ("a", "b"), (), "gcr", id="collinear"),
        pytest.param(plated_latent, ("z",), (), "gcr", id="plated"),
        # plate x sigma-dependence, and a block whose two members have
        # DIFFERENT leaf sizes (`z` is plate-shaped, `w` is scalar).
        pytest.param(plated_radiometer, ("z",), (), "gcr+snis", id="plated_rad"),
        pytest.param(
            plated_and_scalar_latents, ("w", "z"), (), "gcr", id="plate_plus_scalar"
        ),
        # Criterion 3 is VACUOUSLY true for `u`: it reaches no observed node
        # at all, so the universal quantifier ranges over an empty set. Its
        # column of A is exactly zero and the answer is its prior mean, which
        # the exact path already handles -- see
        # test_a_latent_the_data_never_reaches_comes_back_at_its_prior_mean.
        pytest.param(
            unconstrained_latent, ("u", "w"), (), "gcr", id="unconstrained"
        ),
        # The ancestor rule ejects tau in all three, and criterion 2 of the
        # first draft would have inverted every one of them.
        pytest.param(indirect_ancestor, ("x",), ("tau",), "gcr", id="indirect"),
        pytest.param(diamond_ancestor, ("x",), ("tau",), "gcr", id="diamond"),
        pytest.param(shared_ancestor, ("x",), ("tau",), "gcr", id="shared"),
        # Three latents in a CHAIN: tau is an ancestor of x and of y, x is an
        # ancestor of y. Ejection has to remove both, not stop at the first.
        pytest.param(
            three_latent_chain, ("y",), ("tau", "x"), "gcr", id="chain3"
        ),
        # `w` is affine and Gaussian, but `v`'s density depends on it and `v`
        # is disqualified -- so `w` must leave too, or p(v|w) is dropped.
        pytest.param(orphaned_child_latent, (), ("v", "w"), "nuts", id="orphaned"),
        # Joint refusal: each conditional is affine, the pair is not.
        pytest.param(bilinear_pair, (), ("gain", "t_ant"), "nuts", id="bilinear"),
        pytest.param(quadratic_claim, (), ("w",), "nuts", id="quadratic"),
        # `cubic_tail`'s DEFAULT prior_std=1.0 is refused in float32 and in
        # float64 alike (departure 7.45e-01 / 8.41e-01 at the 1e3 probe), so
        # this row does not ride on the dtype the suite happens to run in.
        # It is `cubic_tail(prior_std=1e-4)` -- not used here -- that is
        # dtype-dependent, passing in float32 and refused in float64.
        pytest.param(cubic_tail, (), ("w",), "nuts", id="cubic"),
        # Criterion 2: the OBSERVED node is not Gaussian.
        pytest.param(student_t_likelihood, (), ("w",), "nuts", id="student_t"),
        # `at` is built at the outside latents' PRIOR MEAN, so z sits at 3.0
        # where mu is quadratic in x -- refused at at-point 0. See
        # test_the_block_is_anchored_at_the_outside_latents_prior_mean.
        pytest.param(affine_only_at_zero, (), ("x", "z"), "nuts", id="affine_at_0"),
        # Observed nodes DECLARED in the reverse of their sorted order.
        pytest.param(
            two_observations_reverse_sorted_names, ("w",), (), "gcr", id="rev_sorted"
        ),
        # A Deterministic that reaches no observed node contributes nothing
        # to any prediction, so criterion 3 must not ask it for a
        # declaration.
        pytest.param(dangling_deterministic, ("w",), (), "gcr", id="dangling_det"),
        # A Deterministic whose PARENT is an observed node. `d1` is data, so
        # `w` does not reach `d2`'s location through it.
        pytest.param(
            observation_reused_downstream, ("w",), (), "gcr", id="obs_downstream"
        ),
        # A prediction-dependent sigma on a block that is a PROPER SUBSET --
        # the only row that reaches the Metropolis arm.
        pytest.param(mixed_radiometer, ("w",), ("tau",), "gcr+mh", id="mixed_rad"),
        # Block size 3, on both sides of the sigma branch.
        pytest.param(
            three_member_constant_sigma, ("a", "b", "c"), (), "gcr", id="three_const"
        ),
        pytest.param(
            three_member_moving_sigma,
            ("a", "b", "c"),
            (),
            "gcr+snis",
            id="three_moving",
        ),
        # Two LEGAL models that `check_linearity`'s default at-points refuse:
        # a Cauchy outside latent that overflows the prediction on ~99.7% of
        # prior draws, and an improper outside prior with no sampler at all.
        # `partition` must classify both, which is what building `at_points`
        # from the GAUSSIAN outside latents only buys.
        pytest.param(
            overflowing_outside_latent, ("w",), ("z",), "gcr", id="overflowing"
        ),
        pytest.param(improper_outside_prior, ("w",), ("z",), "gcr", id="improper"),
    ],
)
def test_partition_matches_the_hand_derived_answer(build, exact, nuts, method):
    result = partition(build())
    assert result.exact == exact
    assert result.nuts == nuts
    assert result.method == method


def test_a_refused_block_names_its_members_in_the_reason():
    """"Everything went to NUTS" is useless without saying which claim failed.

    A user one `linear_in` declaration away from an exact solve has to be able
    to see that from the plan, or the whole-block-falls-together policy is
    just an unexplained downgrade.
    """
    result = partition(bilinear_pair())
    assert "gain" in result.reason and "t_ant" in result.reason


def test_a_disqualified_latent_is_named_with_the_criterion_it_failed():
    """The empty-block reason must separate the two latents' two reasons.

    `orphaned_child_latent` fails for two DIFFERENT reasons at once -- `v` is
    Student-t (criterion 1) and `w` is an ancestor of `v` (the ejection rule)
    -- and a reason that collapsed them into one sentence would hide the fact
    that `w` itself is perfectly conjugate.
    """
    result = partition(orphaned_child_latent())
    assert "StudentT" in result.reason
    assert "ancestor" in result.reason
    assert "'w'" in result.reason and "'v'" in result.reason


def test_a_lying_observed_node_raises_rather_than_falling_back_to_nuts():
    """StructureError from check_gaussian must NOT be swallowed.

    `check_linearity` raises the same TYPE for a false linear_in, and that one
    must be caught and routed to NUTS -- so a classifier that discriminates by
    exception type instead of by raise site passes the bilinear_pair row above
    and silently downgrades this broken model. Both rows are needed; neither
    alone pins the distinction.
    """
    with pytest.raises(StructureError, match="log_prob"):
        partition(lying_observed_node())


def test_the_block_is_anchored_at_the_outside_latents_prior_mean():
    """`at` comes from the prior, not from zero -- and the message says where.

    `affine_only_at_zero`'s `mu` is affine in `x` exactly where `z == 0`, and
    `z ~ N(3, 1)`. Anchored at the prior mean the refusal comes from at-point
    0, "the caller's own `at`"; anchored at zero at-point 0 would PASS and the
    refusal would be attributed to "prior draw 1 of the outside latents"
    instead. Both spellings refuse this model, so the verdict alone cannot
    tell them apart -- the attribution can, and the attribution is what a
    user reads.
    """
    result = partition(affine_only_at_zero())
    assert result.method == "nuts"
    assert "the caller's own `at`" in result.reason


@pytest.mark.parametrize("point_count", [DEFAULT_AT_POINTS])
def test_the_linearity_claim_is_checked_at_more_than_one_outside_point(point_count):
    """A single at-point is the moderate-parameter probe `check_linearity` bans.

    `indirect_ancestor` has one latent outside the block (`tau`), so the
    at-points are genuinely distinct rather than repeated empty dicts. The
    count is parametrized off `DEFAULT_AT_POINTS` rather than written as `3`,
    so raising that constant does not silently leave this guard asserting the
    old number.
    """
    result = partition(indirect_ancestor())
    assert result.linearity is not None
    assert sorted(result.linearity) == list(range(point_count))


@pytest.mark.parametrize(
    "build, floor, ceiling",
    [
        # `radiometer`: sigma = kappa|mu| + floor with mu = 0 at the prior
        # mean, so the baseline is the floor alone and one prior width of
        # movement is three decades of it. Measured 2.500e+03.
        pytest.param(radiometer, 1e2, 1e4, id="radiometer_far_above"),
        pytest.param(mixed_radiometer, 1e2, 1e4, id="mixed_rad_far_above"),
        # A constant sigma does not move at all -- bitwise, not merely
        # smaller than the tolerance.
        pytest.param(two_linear_latents, 0.0, 0.0, id="two_linear_bitwise_zero"),
        pytest.param(collinear_pair, 0.0, 0.0, id="collinear_bitwise_zero"),
    ],
)
def test_sigma_movement_is_reported_and_lands_far_from_the_threshold(
    build, floor, ceiling
):
    """Both sides of `SIGMA_RTOL` are covered AT THE ENDS, not at the boundary.

    This is deliberately not a boundary validation. `SIGMA_RTOL` is 1e-8 and
    the four fixtures here read 2.50e+03, 8.46e+02, 0.0 and 0.0 -- eleven
    decades above it and bitwise zero, with nothing within a decade of the
    threshold itself. That matches what
    `check_prediction_dependence`'s own docstring says about its `rtol`: it
    is a coarse yes/no movement detector guarding a declaration, not a
    numeric dispatcher choosing between two methods that must agree at a
    threshold, so there is no requirement that the two sides produce the same
    answer there. A fixture landing near 1e-8 would have to be crafted to,
    and this suite does not craft fixtures to hit thresholds.
    """
    result = partition(build())
    assert result.sigma_movement is not None
    assert floor <= result.sigma_movement <= max(ceiling, floor)
    assert (result.sigma_movement > SIGMA_RTOL) == (result.method != "gcr")


@pytest.mark.parametrize(
    "build, expected",
    [
        # Every latent ancestor of `d` is a block member, so sigma is a
        # function of the block alone and the reweighting fixed point can be
        # hoisted out of the sweep.
        pytest.param(radiometer, False, id="whole_graph_block"),
        pytest.param(two_linear_latents, False, id="whole_graph_constant_sigma"),
        # `tau` is outside the block and reaches `d`, so nothing may be
        # hoisted: the sweep moves `tau`, which may move sigma.
        pytest.param(mixed_radiometer, True, id="latent_outside_the_block"),
        pytest.param(indirect_ancestor, True, id="outside_ancestor_of_the_prior"),
    ],
)
def test_sigma_needs_rebuild_tracks_latents_outside_the_block(build, expected):
    """Whether `noise_std` may be hoisted out of a Gibbs sweep.

    Distinct from `sigma_movement`, which only sees movement WITH the block.
    A conservative over-approximation on purpose: `loc` and `scale` come out
    of one `dist_fn` and cannot be separated structurally, so an outside
    latent that reaches the observed node only through its LOCATION -- which
    is exactly `indirect_ancestor` -- reports True as well. False is the
    dangerous verdict here (it authorises hoisting), so the over-approximation
    is on the safe side.
    """
    assert partition(build()).sigma_needs_rebuild is expected


@pytest.mark.parametrize("key_seed", [0, 1, 7, 2026])
@pytest.mark.parametrize(
    "build, method",
    [
        pytest.param(radiometer, "gcr+snis", id="radiometer"),
        pytest.param(two_linear_latents, "gcr", id="two_linear"),
        pytest.param(bilinear_pair, "nuts", id="bilinear"),
        pytest.param(mixed_radiometer, "gcr+mh", id="mixed_rad"),
    ],
)
def test_the_verdict_is_a_region_not_a_point_in_the_probe_key(build, method, key_seed):
    """The classification must not depend on which random probe was drawn.

    `check_linearity` and `check_prediction_dependence` both take a PRNG key
    and both default to `key(0)`; a verdict that held only at that key would
    be a coincidence rather than a property of the model. Swept over four
    keys here rather than asserted at the default one.
    """
    assert partition(build(), key=jax.random.key(key_seed)).method == method


@pytest.mark.parametrize("seed", [3, 11, 29])
@pytest.mark.parametrize("n", [4, 10, 40])
def test_the_verdict_is_a_region_not_a_point_in_the_data(seed, n):
    """Neither the data nor the sample size moves the structural verdict.

    The whole claim of the structural gate is that it runs before any data is
    fitted. `radiometer`'s sigma-dependence probe starts at the prior mean, so
    the data cannot reach it; sweeping `seed` and `n` is what turns that from
    an argument into a measurement.
    """
    result = partition(radiometer(n=n, seed=seed))
    assert result.exact == ("w",)
    assert result.method == "gcr+snis"
    assert result.sigma_movement > 1e2
