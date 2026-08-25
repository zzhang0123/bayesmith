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
import jax.numpy as jnp
import pytest

from bayesmith.dispatch.classify import (
    SIGMA_RTOL,
    block_at,
    partition,
    prior_environment,
)
from bayesmith.errors import StructureError
from bayesmith.exact.block import unchecked_operator
from bayesmith.exact.gls import check_prediction_dependence, sigma_from_graph
from bayesmith.exact.linearity import DEFAULT_AT_POINTS
from tests.exact.models import (
    affine_only_at_zero,
    bilinear_pair,
    collinear_pair,
    cubic_tail,
    dangling_deterministic,
    diamond_ancestor,
    hinged_sigma_beyond_the_probe,
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
    plated_student_t_latent,
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
        pytest.param(unconstrained_latent, ("u", "w"), (), "gcr", id="unconstrained"),
        # The ancestor rule ejects tau in all three, and criterion 2 of the
        # first draft would have inverted every one of them.
        pytest.param(indirect_ancestor, ("x",), ("tau",), "gcr", id="indirect"),
        pytest.param(diamond_ancestor, ("x",), ("tau",), "gcr", id="diamond"),
        pytest.param(shared_ancestor, ("x",), ("tau",), "gcr", id="shared"),
        # Three latents in a CHAIN: tau is an ancestor of x and of y, x is an
        # ancestor of y. Ejection has to remove both, not stop at the first.
        pytest.param(three_latent_chain, ("y",), ("tau", "x"), "gcr", id="chain3"),
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
        # A sigma whose hinge sits BEYOND every prior-scale probe, so the
        # movement measured from the prior centre alone is bitwise 0.0 and
        # this row reads `gcr` -- no correction at all -- unless sigma is
        # also probed where the data put the posterior. See
        # test_a_sigma_that_hinges_past_the_prior_probe_is_still_detected.
        pytest.param(
            hinged_sigma_beyond_the_probe, ("a",), (), "gcr+snis", id="hinged_sigma"
        ),
        # A latent that is plated AND not Gaussian: `u` is disqualified and
        # leaves, `w` stays. The classifier has to give `u` a centre anyway,
        # because `w`'s own check runs in an environment `u` is part of.
        pytest.param(
            plated_student_t_latent, ("w",), ("u",), "gcr", id="plated_student_t"
        ),
    ],
)
def test_partition_matches_the_hand_derived_answer(build, exact, nuts, method):
    result = partition(build())
    assert result.exact == exact
    assert result.nuts == nuts
    assert result.method == method


def test_a_refused_block_names_its_members_in_the_reason():
    """ "Everything went to NUTS" is useless without saying which claim failed.

    A user one `linear_in` declaration away from an exact solve has to be able
    to see that from the plan, or the whole-block-falls-together policy is
    just an unexplained downgrade.

    Asserted on the part of the reason that came from `check_linearity`, not
    on the whole string. `_all_to_nuts` is handed
    `f"exact block {list(block)} falls together: {exc}"`, and that PREFIX
    names every member unconditionally -- so "gain" and "t_ant" appear in the
    reason even if `: {exc}` is deleted outright and the user is told nothing
    about which claim failed. Measured: with the `: {exc}` removed, the
    earlier spelling of this test stayed green. Splitting the prefix off first
    is what makes the members and the diagnosis both load-bearing.
    """
    result = partition(bilinear_pair())
    _, marker, body = result.reason.partition("falls together: ")
    assert marker, f"the refusal carries no diagnosis at all: {result.reason!r}"
    assert "not JOINTLY affine" in body
    assert "gain" in body and "t_ant" in body


def test_a_non_gaussian_refusal_keeps_its_first_sentence_whole():
    """The truncation is at the first SENTENCE, and `.to_event` has a dot in it.

    `check_gaussian`'s `NotGaussian` message names, in its first sentence, the
    one wrapper that is nonetheless accepted -- `.to_event(...)`. Truncating
    the message at the first "." rather than at the first ". " cuts inside
    that wrapper's own leading dot, and what a user reads ends mid-clause on
    "or one wrapped by", having dropped the only actionable thing the sentence
    had to say. The plan this module was built from sketched the same `.split(".")`,
    so the implementation was faithful and the sketch was wrong.
    """
    reason = partition(student_t_likelihood()).reason
    assert "returns StudentT" in reason
    assert "wrapped by .to_event(...))" in reason
    # Still one sentence, not the whole message: the following two sentences
    # are about a solve that is not implemented and about this being a verdict
    # rather than a defect, and neither belongs in a one-line dispatch reason.
    assert "MultivariateNormal" not in reason


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
        # The only row whose number comes from the data-informed probe rather
        # than from the prior-scale ones: those read bitwise 0.0 here.
        # Measured 1.904e+01, key-free -- the Wiener solve that places the
        # probe takes no key.
        pytest.param(hinged_sigma_beyond_the_probe, 1e1, 1e2, id="hinged_far_above"),
    ],
)
def test_sigma_movement_is_reported_and_lands_far_from_the_threshold(
    build, floor, ceiling
):
    """Both sides of `SIGMA_RTOL` are covered AT THE ENDS, not at the boundary.

    This is deliberately not a boundary validation. `SIGMA_RTOL` is 1e-8 and
    the five fixtures here read 2.50e+03, 8.46e+02, 0.0, 0.0 and 1.90e+01 --
    nine to eleven decades above it and bitwise zero, with nothing within a
    decade of the threshold itself. That matches what
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


@pytest.mark.parametrize("key_seed", [0, 1, 7, 2026])
def test_a_sigma_that_hinges_past_the_prior_probe_is_still_detected(key_seed):
    """A block reaching `gcr` is a block getting NO correction at all.

    `gcr` is the one method that applies nothing on top of the Wiener solve:
    no importance weight, no Metropolis accept, `log_weights is None` and
    `ess == num_samples` exactly. So a sigma that moves and is not SEEN to
    move is not a degraded answer, it is a confident wrong one --
    `hinged_sigma_beyond_the_probe` came back 17.2x too narrow with
    `unreliable=False`.

    Swept over four keys because the fix must not be a coincidence of the
    default one, and because the prior-scale probes read bitwise 0.0 at every
    key here: a flat region is flat in every direction, so no amount of
    re-drawing the probe direction reaches this.
    """
    result = partition(hinged_sigma_beyond_the_probe(), key=jax.random.key(key_seed))
    assert result.exact == ("a",)
    assert result.sigma_movement is not None and result.sigma_movement > SIGMA_RTOL
    assert result.method == "gcr+snis"


def test_the_movement_that_catches_it_is_the_one_at_the_data_informed_point():
    """Names the mechanism, so the row above cannot pass for the wrong reason.

    Re-measures `check_prediction_dependence` directly, at the same block and
    the same anchor `partition` uses, and shows it reads **bitwise 0.0** --
    the prior-scale probes are `DEPENDENCE_PROBES`' 1.0 and -0.5 prior widths
    from the prior centre, and the hinge is at 3 prior widths. What
    `partition` reports instead is the movement between sigma at the prior
    centre and sigma at the block's own Wiener solution, which is where the
    posterior actually sits.

    This is the gap `DEPENDENCE_PROBES`' docstring names and declines to
    close from the prior side: "only a larger magnitude would, at the cost of
    probing where the posterior will never go". The data-informed point is
    the one place a larger magnitude costs nothing, because it is where the
    chain goes by construction.
    """
    graph = hinged_sigma_beyond_the_probe()
    result = partition(graph)
    at = block_at(graph, result.exact, env=prior_environment(graph))
    from_the_prior = check_prediction_dependence(
        unchecked_operator(graph, result.exact, at),
        sigma_from_graph(graph, at),
        declared=True,
        rtol=SIGMA_RTOL,
        key=jax.random.key(0),
    )
    assert from_the_prior == 0.0
    assert result.sigma_movement > 1e1


def test_a_plated_non_gaussian_latent_is_centred_over_its_whole_plate():
    """`_latent_centre`'s plate arm: a distribution does not know its plate.

    `dist.StudentT(6.0, 0.4, 0.9).shape()` is `()` whether or not the
    `sample()` that wrapped it named a plate, so the centre this returns has
    to be broadcast out to the plate's size by hand. Every other non-Gaussian
    latent in `models.py` is scalar and every plated one is Gaussian, so this
    arm had no fixture.

    The value is asserted as well as the shape. `_latent_centre`'s two
    `except` branches immediately below return `jnp.zeros(shape)`, so a mean
    of 0.0 would not tell the arms apart -- `u`'s prior centre is 0.4 for the
    same reason `plated_latent_through_deterministic`'s is 0.8.

    The shape is not cosmetic: measured, with the plate broadcast removed
    `partition` on this fixture does not merely misreport a centre, it raises
    from `apply_deterministic` -- "vmap was requested to map its argument
    along axis 0 ... but is only 0 (its shape is ())" -- so the table row
    above reds too.
    """
    graph = plated_student_t_latent(n=5)
    env = prior_environment(graph)
    assert jnp.shape(env["u"]) == (5,)
    assert jnp.allclose(env["u"], 0.4)


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


def test_a_correlated_graph_is_promised_an_exact_solve_and_gets_one():
    """The dispatcher must not promise a solve it cannot deliver.

    That property is what this test has always been for; its VERDICT has now
    changed twice, and both changes are the point.

    It was written when wiring `check_observed` into `_is_gaussian` was a
    REGRESSION: the density was sound but the block builder's data and loc
    walks were diagonal-only, so `compile()` stopped routing to NUTS and
    raised `NotGaussian` from deeper in. All 748 tests stayed green through
    that, because nothing else compiles a correlated graph.

    Increment 5 closed the gap, and this test then failed a second time --
    correctly -- because "routes to NUTS" had stopped being the right answer.
    The property did not move. What replaced the verdict is the stronger
    statement: the plan says exact, AND the exact path produces the dense
    Wiener filter's answer.

    Asserting the estimate and not merely the routing is what makes this a
    promise-and-delivery test rather than a label check. A `compile()` that
    said "gcr" while `estimate()` raised would pass the first assertion.
    """
    import numpy as np
    import numpyro.distributions as ndist

    from bayesmith import const, det, observe, sample, trace
    from bayesmith.dispatch.plan import compile as compile_graph
    from bayesmith.exact.precision import CirculantPrecision, dense

    size, prior_std = 8, 5.0
    lag = np.minimum(np.arange(size), size - np.arange(size))
    kernel = jnp.asarray(1.0 * 0.4**lag + 0.5)
    grid = jnp.linspace(1.0, 4.0, size)
    data = 2.0 * np.asarray(grid)

    def model():
        xs = const("X", grid)
        w = sample("w", lambda: ndist.Normal(0.0, prior_std))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe(
            "d",
            lambda m: ndist.CirculantNormal(m, kernel),
            mu,
            depends_on_prediction=False,
            obs=jnp.asarray(data),
        )

    with jax.enable_x64(True):
        plan = compile_graph(trace(model))
        assert plan.blocks[0].method == "gcr", plan.blocks[0].method
        estimate = plan.estimate()
        got = float(np.asarray(estimate.values["w"]).reshape(()))
        inverse = np.asarray(
            dense(CirculantPrecision(first_column=kernel), size, jnp.float64)
        )

    design = np.asarray(grid).reshape(-1, 1)
    normal = design.T @ inverse @ design + np.eye(1) / prior_std**2
    reference = np.linalg.solve(normal, design.T @ inverse @ data.reshape(-1, 1)).item()
    assert got == pytest.approx(reference, rel=1e-9)

    # A correlated model has no per-sample sigma, and the estimate says so
    # rather than reporting per-mode amplitudes under that name.
    assert estimate.noise_std is None
