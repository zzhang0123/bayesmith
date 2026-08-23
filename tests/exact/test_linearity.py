"""Checking the linear_in claim -- at several scales and at several at-points."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from bayesmith import sample, trace
from bayesmith.errors import GraphError, NotGaussian, StructureError
from bayesmith.exact.block import unchecked_operator
from bayesmith.exact.linearity import (
    DEFAULT_SCALES,
    affinity_errors,
    check_linearity,
    linear_operator,
    prior_at_points,
)
from tests.exact.models import (
    affine_only_at_zero,
    bilinear_pair,
    bright_and_faint_channels,
    bright_and_faint_observations,
    bright_and_faint_pair,
    cubic_tail,
    faint_alone,
    improper_outside_prior,
    nan_at_negative_probes,
    non_gaussian_observed_node,
    overflowing_outside_latent,
    quadratic_claim,
    roundoff_stress,
    straight_line,
    tunable_curvature,
    two_linear_latents,
    two_observations,
    two_unusable_observed_scales,
    unusable_observed_scale,
)


def test_a_genuinely_linear_block_passes_at_every_scale():
    graph = straight_line()
    errors = check_linearity(graph, ("w",), at={})
    assert errors  # one entry per at-point
    for per_point in errors.values():
        assert len(per_point) == 3
        assert all(err < 1e-3 for err in per_point.values())


def test_a_quadratic_claim_is_refused():
    graph = quadratic_claim()
    with pytest.raises(StructureError, match="affine"):
        check_linearity(graph, ("w",), at={})


def test_a_bilinear_pair_passes_singly_and_fails_jointly():
    """rheplicant's motivating failure, caught by three forward evaluations.

    Each conditional genuinely IS affine, which is why checking one latent at
    a time cannot see this and why a hand-rolled alternating solve reports a
    CG residual of 1e-7 and a per-block condition number of ~1.5 while landing
    thousands of kelvin away. The claim that is false is the JOINT one.
    """
    graph = bilinear_pair()
    check_linearity(graph, ("gain",), at={"t_ant": jnp.asarray(2.0)})
    check_linearity(graph, ("t_ant",), at={"gain": jnp.asarray(1.0)})
    with pytest.raises(StructureError, match="JOINTLY"):
        check_linearity(graph, ("gain", "t_ant"), at={})


def test_the_probe_magnitude_is_read_off_the_declared_prior():
    """One fn, two declared prior widths, opposite verdicts.

    Nothing about the model changes between these two calls except the width
    the graph declares for `w`. If the probe magnitude were a fixed constant
    the two would agree, whichever way.

    **This test's REFUSED half depends on `DEFAULT_SCALES` reaching 1e3.**
    Measured directly against `cubic_tail(prior_std=1.0)` through
    `check_linearity` itself (the exact call this test makes): the relative
    departure from affinity is 0.00e+00 at scale=1e-3, 4.58e-06 at scale=1.0,
    and 7.45e-01 only at scale=1e3 -- against this fixture's float32 `rtol`
    of 1.19e-3, the first two scales are indistinguishable from a genuinely
    linear model and only the widest probe catches the cubic term. A change
    that narrows `DEFAULT_SCALES` to drop 1e3 (e.g. to `(1e-3, 1.0)`) makes
    this test go RED -- not because the probe-magnitude-from-prior logic
    broke, but because the fixture's departure is invisible at the scales
    that remain. Recorded here so that a future edit to `DEFAULT_SCALES`
    meets this explanation instead of a silent, surprising failure.
    """
    with pytest.raises(StructureError, match="affine"):
        check_linearity(cubic_tail(prior_std=1.0), ("w",), at={})
    check_linearity(cubic_tail(prior_std=1e-4), ("w",), at={})


def test_check_linearity_probes_more_than_the_caller_s_at_point():
    """The claim holds where the caller pinned z, and nowhere the prior goes.

    Pinned explicitly to a single at-point the check passes; left to its
    default -- the caller's at PLUS draws from the graph's own prior -- it
    does not. An implementation that probed one point would pass both.
    """
    graph = affine_only_at_zero()
    pinned = {"z": jnp.asarray(0.0)}
    check_linearity(graph, ("x",), at=pinned, at_points=[pinned])
    with pytest.raises(StructureError, match="affine"):
        check_linearity(graph, ("x",), at=pinned)


def test_a_probe_that_returns_nan_counts_as_a_failure():
    """`nan_at_negative_probes` linearizes sqrt(w) AT w=0 -- itself a singular
    point of sqrt's derivative -- so this test's own raise is OVER-determined:
    a POSITIVE probe there produces a clean +inf departure that trips the
    ordinary `errors > rtol` comparison on its own (inf, unlike nan, compares
    fine), independent of whether NaN is separately masked. Measured
    directly: deleting the `not finite` branch this test is meant to guard
    still leaves it green, because at least one of its 9 (point, scale) grid
    cells lands on a positive probe under the fixed default key and the
    resulting +inf redundantly triggers the raise from a DIFFERENT point
    index than the one where every probe was negative. This test alone,
    therefore, does not prove the `not finite` branch is load-bearing --
    `test_affinity_errors_treats_nan_as_a_failure_in_isolation` below does,
    by construction, with no +inf anywhere to fall back on.
    """
    graph = nan_at_negative_probes()
    with pytest.raises(StructureError, match="affine"):
        check_linearity(graph, ("w",), at={})


def test_affinity_errors_treats_nan_as_a_failure_in_isolation():
    """Isolates the `not finite` branch from the redundant +inf pathway above.

    Linearizes sqrt(w) at w=1 -- an ORDINARY point, derivative 0.5, nothing
    singular -- then probes ENTIRELY on the negative side (w going to -0.5),
    so `actual` is NaN (sqrt of a negative number) while `predicted` stays
    finite (a regular derivative times a finite probe). The resulting
    departure is a clean NaN with no accompanying +inf anywhere: a naive
    `errors > rtol` comparison alone reads `nan > rtol` as False and would
    call this affine. Only the `not finite` branch catches it.

    The NaN reaches BOTH criteria here, and each is checked on its own
    finiteness, so this pins both branches at once: `sigma=1.0` is finite and
    strictly positive, so `departure / sigma` is NaN only because `departure`
    is.
    """

    def g(x):
        return {"y": jnp.sqrt(x["w"])}

    zero = {"w": jnp.asarray(1.0)}

    def probe_at(index, scale):
        del index
        return {"w": jnp.asarray(-1.5 * scale)}

    errors, failed, _, columns = affinity_errors(
        g, zero, probe_at, (1.0,), None, sigma={"y": jnp.asarray(1.0)}
    )
    assert not jnp.isfinite(errors[1.0])  # confirms this probe is the clean-NaN case
    assert all(not jnp.isfinite(column) for column in columns[1.0])
    assert failed == [1.0]


def test_linear_operator_checks_before_it_builds():
    """The safe name is the natural one; the unchecked primitive says so."""
    graph = quadratic_claim()
    with pytest.raises(StructureError):
        linear_operator(graph, ("w",), at={})
    unchecked_operator(graph, ("w",), at={})  # builds happily, and is wrong


def test_linear_operator_returns_the_block_the_primitive_would_have():
    graph = two_linear_latents()
    at = {"b": jnp.asarray(4.0)}
    checked = linear_operator(graph, ("a",), at=at)
    raw = unchecked_operator(graph, ("a",), at=at)
    assert jnp.allclose(checked.offset["d"], raw.offset["d"])
    probe = {"a": jnp.asarray(1.0)}
    assert jnp.allclose(checked.forward(probe)["d"], raw.forward(probe)["d"])
    assert jnp.allclose(checked.prior_std["a"], raw.prior_std["a"])


def test_a_graph_with_no_observed_node_is_refused():
    """There is nothing to condition on, so the posterior IS the prior.

    Refused by name rather than reaching affinity_errors, where an empty
    codomain has no dtype to take a tolerance from and the failure would
    arrive as an unrelated ValueError from two layers down -- measured
    directly: without the guard, `check_linearity` on this fixture raises
    ``ValueError: at least one array or dtype is required`` from
    `jnp.result_type()` seeing zero leaves.

    Checked through all three entry points, not just `unchecked_operator`.
    `linear_operator` calls `check_linearity` BEFORE `unchecked_operator`, so
    a guard living only in the latter is unreachable from the documented
    entry point -- `check_linearity` would hit the confusing ValueError
    above before `unchecked_operator`'s refusal is ever reached. That is why
    `_refuse_missing_observed` is a helper shared by both rather than a
    check inlined in `unchecked_operator` alone.
    """

    def model():
        sample("w", lambda: dist.Normal(0.0, 1.0))

    graph = trace(model)
    with pytest.raises(GraphError, match="observed"):
        unchecked_operator(graph, ("w",), at={})
    with pytest.raises(GraphError, match="observed"):
        check_linearity(graph, ("w",), at={})
    with pytest.raises(GraphError, match="observed"):
        linear_operator(graph, ("w",), at={})


def test_a_non_finite_baseline_is_attributed_to_the_outside_latent_not_the_member():
    """A genuinely affine `w` must not be blamed for `z` overflowing elsewhere.

    `z ~ Cauchy(0, 1e6)` feeds `exp(|z| / 50)`, which overflows float32 on
    about 99.7% of draws (measured: overflow needs `|z| > 4436`, and this
    Cauchy puts over 99% of its mass beyond that). Holding `z` fixed at ANY
    value, `mu` is affine in `w` -- so the default at-points, which draw `z`
    from its own prior, hit the overflow on (virtually certainly) at least
    one of the two draws, and the failure must name where `z` landed, not
    `w`'s declaration.
    """
    graph = overflowing_outside_latent()
    with pytest.raises(StructureError, match="non-finite at"):
        check_linearity(graph, ("w",), at={"z": jnp.asarray(0.0)})
    # Pinned at a `z` that does NOT overflow, the SAME block is accepted --
    # proof `w`'s declaration was true all along, and that the raise above
    # was never really about `w`.
    check_linearity(
        graph,
        ("w",),
        at={"z": jnp.asarray(0.0)},
        at_points=[{"z": jnp.asarray(0.0)}],
    )


def test_an_improper_outside_prior_gives_a_named_error_not_a_bare_one():
    """`ImproperUniform` has no sampler, so drawing default at-points must fail
    with a message pointing at `at_points=`, not a contextless NotImplementedError
    surfacing from three layers inside NumPyro.
    """
    graph = improper_outside_prior()
    with pytest.raises(StructureError, match="cannot be sampled"):
        check_linearity(graph, ("w",), at={"z": jnp.asarray(0.0)})
    # An explicit at_points sidesteps the sampler entirely and passes: `w`
    # really is affine, and pointing at `at_points=` in the message is
    # correct advice, not just an apology for the failure.
    check_linearity(
        graph,
        ("w",),
        at={"z": jnp.asarray(0.0)},
        at_points=[{"z": jnp.asarray(0.0)}],
    )


def test_prior_at_points_short_circuits_when_nothing_is_outside(monkeypatch):
    """No latent outside the block means every at-point is the same empty
    dict, so drawing it from the prior would cost a full NumPyro forward
    trace to learn nothing. Counts calls to `to_numpyro` rather than
    comparing the returned points (`{}` either way, with or without the
    short-circuit) -- a value-only assertion cannot see the wasted work the
    short-circuit removes. Same technique as
    `test_ancestry_dedups_a_diamond` in test_block.py.
    """
    import bayesmith.bridge.numpyro_bridge as bridge

    calls: list[object] = []
    original = bridge.to_numpyro

    def counting(g):
        calls.append(g)
        return original(g)

    monkeypatch.setattr(bridge, "to_numpyro", counting)

    graph = straight_line()  # single latent "w" -> nothing outside a ("w",) block
    points = prior_at_points(graph, ("w",), 2, jax.random.key(0))
    assert points == [{}, {}]
    assert calls == []


def test_the_default_scales_span_six_orders_of_magnitude():
    """The span is load-bearing, so it gets a test that says so by name.

    `cubic_tail(prior_std=1.0)` is caught ONLY at the largest scale: measured
    departures are 0.0 at 1e-3, 4.58e-06 at 1.0, and 7.45e-01 at 1e3, against
    an rtol of 1.19e-3. Narrow the sweep and that fixture passes silently,
    while the test that goes red is one whose name is about where the probe
    magnitude comes from -- not about the sweep at all.
    """
    assert max(DEFAULT_SCALES) / min(DEFAULT_SCALES) >= 1e6


def test_a_bright_component_does_not_mask_a_false_claim_on_a_faint_one():
    """The guard must not let one codomain leaf set another's yardstick.

    Measured before the fix: `check_linearity` returned a worst relative
    error of 2.57e-14 in float32 (3.45e-14 in float64) on this graph -- a
    clean PASS -- while the faint node ALONE was correctly refused at
    4.93e+00. The bright leaf supplied both the normalising `variation` and
    the roundoff `floor`, and each dilution was independent of the other.

    The consequence is not cosmetic: the "exact" posterior on this graph is
    +1.125 against a truth of +0.803, which is 202 true posterior standard
    deviations. `d2`'s sigma is 0.01 and `d1`'s is 1e19, so the faint node
    carries essentially all the information.

    `faint_alone` is asserted here too, in the same test, because the pair is
    the measurement: refusing the faint node is only evidence about the
    dilution if the SAME node is refused when its bright sibling is removed.
    """
    with pytest.raises(StructureError, match="not affine"):
        check_linearity(faint_alone(), ["w"])
    with pytest.raises(StructureError, match="not affine"):
        check_linearity(bright_and_faint_observations(), ["w"])


def test_the_dilution_is_caught_within_a_single_array_too():
    """Per-leaf is not enough -- the bright and faint entries share a leaf.

    Named by `test_a_bright_component_does_not_mask_a_false_claim_on_a_faint_one`,
    and it by this one: the two are the leaf-level and element-level halves of
    one defect, and a fix that only groups by leaf passes the first and fails
    this. `check_gaussian` made the elementwise choice for exactly this reason
    and says so in its own docstring.
    """
    with pytest.raises(StructureError, match="not affine"):
        check_linearity(bright_and_faint_channels(), ["w"])


def test_an_affine_model_with_the_same_dynamic_range_still_passes():
    """The two-sided half: huge dynamic range alone must NOT trip the guard.

    Without this, a "fix" that simply tightened the tolerance would pass both
    tests above while refusing every legitimate wide-dynamic-range model --
    which is most of this package's intended use.

    Only `mu`'s own fn is replaced, so the honest twin differs from the lying
    graph in exactly one node: same 1e17 first channel, same per-channel
    sigmas, same prior, same probes. `mu` is located by NAME rather than by
    position -- measured, `bright_and_faint_channels().nodes[2]` is `w`, a
    `Probabilistic` with no `fn` field at all, so a hard-coded index makes
    this test raise `AttributeError` instead of testing anything.
    """
    graph = bright_and_faint_channels(w_true=0.8)
    where = next(i for i, node in enumerate(graph.nodes) if node.name == "mu")
    honest = eqx.tree_at(
        lambda g: g.nodes[where].fn,
        graph,
        replace=lambda w_, x_, c_: w_ * x_,
    )
    assert check_linearity(honest, ["w"])


@pytest.mark.parametrize("bright", [1e-17, 1e-6, 1.0, 1e6, 1e17])
def test_the_dilution_is_caught_across_the_whole_brightness_range(bright):
    """Both ENDPOINTS, not only the one the defect was reported at.

    `boundary-validation.md`: a probe that only tests the parameter value the
    bug was found at cannot say whether the fix has a working range. 1e17 is
    an endpoint of the plausible range, so the low end is swept too -- at
    `bright=1e-17` the "bright" leaf is 1e17 times FAINTER than the faint
    one, which is the same dilution with the roles exchanged, and the false
    claim must still be refused.
    """
    with pytest.raises(StructureError, match="not affine"):
        check_linearity(bright_and_faint_channels(bright=bright), ["w"])


def test_the_dilution_is_caught_for_a_two_member_block():
    """The structural dimension the other bright/faint fixtures do not cover.

    Every other fixture here has a ONE-member block, so none of them says
    whether the per-element comparison survives the probe scheme's per-member
    random directions -- the probe is a different random draw per member, and
    a joint claim is what `bilinear_pair` shows a per-latent check cannot
    see. Here the two are combined: only the JOINT claim is false, and only
    on the faint channels.
    """
    with pytest.raises(StructureError, match="JOINTLY"):
        check_linearity(bright_and_faint_pair(), ["a", "b"])


def test_the_relative_criterion_is_load_bearing_where_sigma_hides_the_curvature():
    """The other two-sided half: `rtol` must still decide something on its own.

    The guard refuses on EITHER of two criteria -- a per-element relative
    departure against `rtol`, and a departure in units of sigma against
    `WEIGHTED_RTOL` -- so each needs a case the other cannot reach, or one of
    them is dead code that a future edit can delete without any test noticing.

    This is the relative half's case. `sigma=1e9` makes the departure worth
    1.5e-04 sigma at the widest probe, far under `WEIGHTED_RTOL`, while the
    prediction still differs from its own linearization by ~100% -- so the
    sigma-weighted criterion sees nothing and only `rtol` refuses. `rtol` is
    a documented public kwarg, and this is what it decides.
    """
    with pytest.raises(StructureError, match="not affine"):
        check_linearity(tunable_curvature(departure=1e-1, sigma=1e9), ["w"])


def test_the_refusal_reports_both_criteria_and_both_thresholds():
    """One number in the message would not say WHICH criterion refused.

    The guard is a disjunction, so a message quoting a single error against a
    single threshold is unreadable half the time: a reader sees a number that
    does not exceed the tolerance printed beside it and concludes the guard
    is broken. Both columns and both thresholds are reported per probe.
    """
    with pytest.raises(StructureError) as raised:
        check_linearity(bright_and_faint_channels(), ["w"])
    message = str(raised.value)
    assert "rtol=" in message
    assert "weighted_rtol=" in message
    assert "relative" in message and "sigma-weighted" in message


def test_the_per_element_denominator_sees_a_faint_lie_the_noise_cannot():
    """The two repairs are separable, and this separates them.

    The dilution is caught by two independent halves of the fix: the
    per-element roundoff FLOOR (which the sigma-weighted criterion is gated
    by) and the per-element normalising DENOMINATOR. Measured -- reverting
    the denominator alone to a global maximum over every leaf leaves every
    other test in this file green, because the weighted criterion catches
    those fixtures on the floor alone.

    `sigma_faint=1e13` is not a model of anything -- it is chosen to switch
    the weighted criterion off, and nothing else. Measured, the faint node's
    departure is then worth at most 9e-07 sigma across all three probes, four
    orders under `WEIGHTED_RTOL`, so only the relative criterion is left. Its
    per-element denominator is what this test pins: replace it with a global
    maximum and the bright leaf's 1e17 variation drives the ratio to ~1e-14,
    the lie passes, and every other test in this file stays green.
    """
    with pytest.raises(StructureError, match="not affine"):
        check_linearity(bright_and_faint_observations(sigma_faint=1e13), ["w"])


def test_the_sigma_weighted_criterion_survives_a_loosened_rtol():
    """`rtol` is a public kwarg, so it can be loosened until it decides nothing.

    `WEIGHTED_RTOL` has no kwarg and cannot be, which is the point: a caller
    who widens `rtol` to accommodate a noisy model has not thereby said that
    a departure worth 4.7e+07 noise widths is acceptable.

    Measured: with `rtol=1e6` the relative criterion cannot fire on any
    fixture in this file (its largest relative departure is 1.0 by
    construction, since `departure <= variation` whenever the linearization
    is the zero map), so this test fails the moment the sigma-weighted half
    is removed -- which no other test in this file does.

    `tunable_curvature(1e-6)` rather than a grosser lie because it also pins
    `WEIGHTED_RTOL`'s VALUE from above: its above-floor weighted departure is
    1.17e+01, so loosening the threshold by 1e6 makes this test go red, where
    a fixture sitting 4.7e+12 above it would survive any loosening a typo
    could plausibly introduce.
    """
    with pytest.raises(StructureError, match="not affine"):
        check_linearity(tunable_curvature(departure=1e-6), ["w"], rtol=1e6)


def test_a_lone_lying_channel_is_not_diluted_by_five_honest_ones():
    """The reduction over elements must be `any`, not an average.

    `bright_and_faint_channels()`'s default puts the false claim on five of
    six channels, so the mean of the per-element departures is within 6/5 of
    their maximum -- measured, an averaging reduction still refuses it, and
    the default fixture therefore cannot tell the two apart. With ONE lying
    channel the average is diluted by a factor of six by the honest entries,
    and by an unbounded factor as the array grows.

    This is `check_gaussian`'s own argument for its elementwise probe, and
    the reason `linear_operator` is allowed to be trusted on a spectrum whose
    defect lives in a single channel.
    """
    with pytest.raises(StructureError, match="not affine"):
        check_linearity(bright_and_faint_channels(lying=1), ["w"])


def test_a_covariate_grid_containing_an_exact_zero_is_not_a_failure():
    """An element the block cannot move at all is 0/0, and must not be a NaN.

    `two_observations`'s second covariate grid is `linspace(-1, 1, 5)`, whose
    third entry is exactly `0.0`, so `mu2[2]` is identically zero for every
    `w`: variation 0, departure 0. The per-element relative measure divides
    those, and the divisor is floored at `finfo(dtype).tiny` for exactly this
    reason. Measured: with the `1e-300` literal this guard was first drafted
    with, that floor UNDERFLOWS to 0.0 in float32, the ratio is `0/0 = NaN`,
    the finiteness branch reads NaN as a failure, and this entirely honest
    fixture is refused because its covariate grid contains a zero.

    float64 hides the whole thing -- 1e-300 is representable there -- so this
    test is only meaningful at the dtype the suite actually runs.
    """
    assert check_linearity(two_observations(), ["w"])


@pytest.mark.parametrize("big, sigma", [(1e0, 1e-2), (1e6, 1e-2), (1e15, 1e-2)])
def test_a_true_claim_with_real_roundoff_passes_at_any_offset_ratio(big, sigma):
    """The lower bound on the sigma-weighted criterion, and the only one.

    Every other honest fixture in `models.py` is bitwise exact -- the primal
    and the linearization evaluate the same expression in the same order, so
    the departure is identically zero and nothing bounds any tolerance from
    below. `roundoff_stress` has a REAL departure of order `eps * big * x`
    with a `linear_in` claim that is nonetheless exactly true.

    `departure / sigma` grows with the offset-to-noise ratio rather than with
    curvature, so ungated it reaches 2.44e-02 at a ratio of 1e2 in float32 --
    24 times `WEIGHTED_RTOL` -- for a model with no curvature at all. The
    per-element roundoff floor is what drives it to exactly 0. Without this
    test the floor on the weighted criterion can be deleted and the suite
    stays green, while every wide-dynamic-range model this package targets is
    refused.
    """
    assert check_linearity(roundoff_stress(big=big, sigma=sigma), ["w"])


@pytest.mark.parametrize("kind", ["zero", "one_zero", "negative", "nan"])
def test_an_unusable_observed_scale_is_blamed_on_the_node_not_the_latent(kind):
    """The scale expression is at fault, so the message must name the node.

    `mu = w * X` is exactly affine in this fixture, so every word of "latent
    'w' is declared linear, but the prediction is not affine in it" is
    wrong -- it sends the modeller to rewrite the one part of the model that
    is correct. Measured before the guard, for kind in {zero, one_zero, nan}:
    that is exactly the message they got, because departure/sigma goes inf or
    NaN and the finiteness branch reads an unreadable column as curvature.

    `kind="negative"` is the case that did not even mis-attribute: it PASSED.
    The weighted column takes `abs(sigma)`, so it cannot tell -0.5 from +0.5,
    while `check_gaussian` refuses a non-positive scale by name. A guard
    written only against non-finite values leaves that hole open, which is
    why the negative arm is parametrised separately rather than folded in.
    """
    graph = unusable_observed_scale(kind=kind)
    with pytest.raises(StructureError) as excinfo:
        check_linearity(graph, ["w"])
    message = str(excinfo.value)
    assert "'d'" in message, "the message must name the observed node at fault"
    assert "not affine" not in message, (
        "the model IS affine; blaming linear_in is the mis-attribution this "
        "guard exists to prevent"
    )


def test_a_usable_scale_of_the_same_shape_still_passes():
    """The other side. Without it, a guard that refused every array-valued
    sigma would pass all four arms above and break every real model.
    """
    graph = unusable_observed_scale(kind="one_zero")
    honest = eqx.tree_at(
        lambda g: g.nodes[3].dist_fn,
        graph,
        replace=lambda m: dist.Normal(m, jnp.full(5, 0.4)),
    )
    assert check_linearity(honest, ["w"])


def test_a_non_gaussian_observed_node_raises_not_gaussian_not_structure_error():
    """Pins a contract change that arrived silently, and the class of error.

    `check_linearity` used to read only each observed node's LOCATION, so a
    Student-t likelihood over an affine mean checked fine. Adding the
    sigma-weighted criterion made it call `noise_std_at`, which reaches
    `gaussian_parts` and raises `NotGaussian`. Nothing recorded that, and no
    fixture exercised it.

    WHICH error it is, is load-bearing rather than cosmetic. P3b's dispatcher
    catches `NotGaussian` and routes the block to NUTS -- an ordinary
    non-conjugate model, not a broken one -- while `StructureError` means a
    declaration was checked and found false and must never be swallowed. If
    this raise ever changes class, an ordinary Student-t likelihood starts
    looking like a broken model.
    """
    graph = non_gaussian_observed_node()
    with pytest.raises(NotGaussian) as excinfo:
        check_linearity(graph, ["w"])
    assert "'d'" in str(excinfo.value)
    assert not isinstance(excinfo.value, StructureError), (
        "NotGaussian and StructureError must stay siblings: a dispatcher "
        "doing `except NotGaussian` must not also swallow a broken model"
    )


def test_which_node_an_unusable_scale_names_does_not_depend_on_declaration_order():
    """`noise_std_at` returns declaration order, so the guard's `sorted` bears weight.

    With a single broken node the ordering is unobservable and the `sorted`
    could be deleted with nothing going red -- the shape P3a Task 9 named:
    a dict straight out of a JAX transform is already key-sorted, so sorting
    it again is provably a no-op, while a dict built by comprehension carries
    declaration order and sorting it is a guard. `noise_std_at` builds its
    result by comprehension over `graph.observed`, so it is the second kind.

    Here `z_first` is declared first and `a_second` sorts first. This pins the
    STABILITY of the message, not its correctness: either node is a fair thing
    to name and both are genuinely broken. The defect would be the name
    changing because somebody reordered their `observe()` calls.
    """
    graph = two_unusable_observed_scales()
    assert graph.observed == ("z_first", "a_second"), (
        "fixture must declare the nodes in reverse-sorted order, or this "
        "test cannot distinguish the two orderings"
    )
    with pytest.raises(StructureError) as excinfo:
        check_linearity(graph, ["w"])
    assert "'a_second'" in str(excinfo.value)
