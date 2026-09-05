"""R5 Task 2 -- the quadrature oracle, and the certificate it can fail.

**Which side is the oracle.** ``oracle_joint`` -- the trapezoid over EVERY
latent of the graph's own ``log_joint`` -- is the oracle, and the collapsed
route is what it grades.  Getting that backwards is not a stylistic choice: a
quadrature of the COLLAPSED density calls ``collapse_graph`` and
``marginal_log_density``, so it shares the elimination with the thing it would
be grading, and R4's close-out prices what that costs -- scaling
``dense_operator`` by 1.03 left the square-root cross-route test *completely
blind*, 5 passed, exit 0.  ``test_the_oracle_is_not_blind_to_a_scaled_dense_operator``
below applies that same mutation to this comparison and requires it to go red.

**What each test here is for**, since a file of quadratures reads as one idea
repeated:

* the four ``gcr`` class-(b) fixtures agree, each carrying its own span;
* the same comparison, with ``dense_operator`` scaled, goes red by five to
  seven orders -- an oracle that cannot show what it kills is not an oracle;
* the agreement floor is measured against that kill, so the one declared level
  in D111 is pinned between what it must tolerate and what it must catch;
* a span whose interior holds ``shared_ancestor``'s degenerate conditional
  ABSTAINs instead of reporting the several-nat gap it would otherwise print;
* a span placed off the mass ABSTAINs, and for the edge's reason, not because a
  refinement budget ran out;
* the geometric truncation bound UNDERESTIMATES a polynomial tail, by a factor
  this file measures rather than disclaims;
* every certified value carries its span and the prior mass that span excludes,
  and the excluded prior mass is NOT the abstain criterion;
* the structural class census of every shipped graph, so it cannot go stale.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import types
from collections import Counter
from collections.abc import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

import bayesmith
from bayesmith.artifacts.refusal import Refusal
from bayesmith.dispatch import collapse as collapse_module
from bayesmith.errors import BayesmithError
from bayesmith.graph.reduction import as_graph
from tests.dispatch import residual_oracle
from tests.dispatch.residual_oracle import (
    AGREEMENT_FLOOR,
    EPS,
    Span,
    agreement,
    oracle_collapsed,
    oracle_joint,
    quadrature,
)
from tests.exact import models, residual_models

#: The four ``gcr`` class-(b) fixtures: one exact latent, one residual latent,
#: and a span per axis.  **The spans are part of the fixture, not of the
#: implementation.**  Each one is a declared DOMAIN of the oracle (R5 plan
#: 0.15): ``shared_ancestor``'s excludes the neighbourhood of ``tau = 0``, where
#: ``x ~ N(0, |tau|)`` is a Dirac and ``marginal_log_density`` does not merely
#: lose accuracy but aborts -- see
#: ``test_a_span_across_the_degenerate_conditional_aborts_inside_the_collapse``.
#: ``overflowing_outside_latent``'s runs to ``|z| <= 100`` because its
#: ``Cauchy(0, 1e6)`` tail is 1.3e-03 short of converged at the ``|z| <= 60``
#: the R5 plan used while planning; that near miss is itself a test below.
CLASS_B = {
    "diamond_ancestor": (("x",), (("tau", -4.0, 8.0), ("x", -6.0, 6.0))),
    "indirect_ancestor": (("x",), (("tau", -4.0, 8.0), ("x", -6.0, 6.0))),
    "shared_ancestor": (("x",), (("tau", 0.15, 4.5), ("x", -1.5, 3.5))),
    "overflowing_outside_latent": (("w",), (("z", -100.0, 100.0), ("w", -8.0, 8.0))),
    # The two below are not shipped fixtures; they exist because an adversarial
    # review measured what the four above hold CONSTANT. Every one of them has
    # exactly one observed node, always a descendant of the exact block, and an
    # eliminated block whose prior mean is exactly zero -- so a whole class of
    # elimination defect and three mutants of `block_prior_ratio` were invisible
    # to the comparison. See tests/exact/residual_models.py for the counts.
    "outside_observation_pair": (("x",), (("tau", -1.5, 4.5), ("x", -7.0, 7.0))),
    # The same graph with its outside observation renamed so it sorts FIRST.
    # `marginal_log_density` walks `sorted(block.data)` with a row cursor, so a
    # filter defect indexed by POSITION is invisible while the unabsorbed
    # observation is always last -- which it was, in 6 of 6 graphs, including
    # the fixture written to close the previous gap.
    "outside_observation_first": (("x",), (("tau", -1.5, 4.5), ("x", -7.0, 7.0))),
    "shifted_block_prior": (("x",), (("tau", 0.4, 4.0), ("x", -1.0, 5.5))),
}


def _build(name):
    """The graph for one CLASS_B entry, from whichever module ships it."""
    if name == "outside_observation_first":
        return as_graph(residual_models.outside_observation_pair(outer_name="a"))
    source = models if hasattr(models, name) else residual_models
    return as_graph(getattr(source, name)())


def _sides(name):
    """``(collapsed, joint)`` for one class-(b) fixture, both certified or not.

    The starting grids differ because the two integrals have different
    dimensions and the same wall-clock buys different resolutions; the
    RESOLUTION each is held to is the same declared number, which is the half
    that matters.
    """
    exact, layout = CLASS_B[name]
    graph = _build(name)
    spans = tuple(Span(*entry) for entry in layout)
    residual = tuple(span for span in spans if span.name not in exact)
    collapsed = oracle_collapsed(
        graph, exact, residual, resolution=AGREEMENT_FLOOR, start=401, refinements=9
    )
    joint = oracle_joint(
        graph, spans, resolution=AGREEMENT_FLOOR, start=201, refinements=7
    )
    return collapsed, joint


@pytest.mark.parametrize("name", sorted(CLASS_B))
def test_the_collapsed_and_uncollapsed_evidences_agree_on_the_four_class_b_fixtures(
    name,
):
    """R5 G4. The GAP is the quantity; the twelve-digit sums are not.

    A trapezoid sum is a platform-dependent reduction and red line 9 forbids
    pinning one -- four release tags were spent learning that.  What is asserted
    is that the two routes answer the same number to within the sum of what each
    says about its own error, and that BOTH certified: an agreement between two
    values neither of which has converged is the failure this whole module
    exists to make impossible.
    """
    with jax.enable_x64(True):
        collapsed, joint = _sides(name)
    assert collapsed.certified, collapsed.describe()
    assert joint.certified, joint.describe()
    verdict = agreement(collapsed, joint)
    assert verdict.agree, verdict.describe()


@pytest.mark.parametrize("name", sorted(CLASS_B))
def test_the_oracle_is_not_blind_to_a_scaled_dense_operator(name, monkeypatch):
    """R5 Task 2.1b. The specific mutation that a previous oracle survived.

    R4's close-out records the square-root cross-route test passing with
    ``dense_operator``'s return scaled by 1.03, because both of its sides went
    through ``dense_operator``.  Here only the COLLAPSED side does; the
    uncollapsed side reaches the same model through ``log_joint``, which is the
    model's own definition.  So the mutation must move the gap, and it does --
    by four to six orders of the band, measured.
    """
    real = collapse_module.dense_operator
    monkeypatch.setattr(
        collapse_module, "dense_operator", lambda block: real(block) * 1.03
    )
    with jax.enable_x64(True):
        collapsed, joint = _sides(name)
    assert collapsed.certified, collapsed.describe()
    assert joint.certified, joint.describe()
    verdict = agreement(collapsed, joint)
    assert not verdict.agree, verdict.describe()
    # And not by a hair: a comparison that only just goes red would go green
    # again on a differently-rounding platform.
    assert verdict.margin > 1e4, verdict.describe()


def test_the_agreement_floor_is_far_below_the_defect_it_must_catch():
    """D111's one declared level, pinned between its two measured neighbours.

    A band assembled from each side's own error bounds could in principle widen
    until it swallowed a defect, and a FLOOR under that band could in principle
    be set so high it swallowed one outright.  Neither is asserted here; both
    are measured on the same fixture, in the same run, and the separation is
    what the assertion reads.
    """
    real = collapse_module.dense_operator
    with jax.enable_x64(True):
        clean = agreement(*_sides("indirect_ancestor"))
        try:
            collapse_module.dense_operator = lambda block: real(block) * 1.03
            scaled = agreement(*_sides("indirect_ancestor"))
        finally:
            collapse_module.dense_operator = real
    floor = AGREEMENT_FLOOR * max(1.0, abs(clean.left.value))
    # Below: the routes' float-level disagreement, which the floor must tolerate
    # so that a differently-rounding LAPACK still certifies.
    assert abs(clean.gap) < floor / 1e3, clean.describe()
    # Above: the smallest defect this comparison exists to call red.
    assert abs(scaled.gap) > floor * 1e4, scaled.describe()


def test_the_oracle_abstains_rather_than_reporting_a_gap_it_cannot_certify():
    """R5 Task 2.3. The near miss is a regression test, not an anecdote.

    While the R5 plan was being written, a first grid on ``shared_ancestor``
    reported a gap of **6.55 nats** and a reader given only that table would have
    opened a defect against ``collapse_graph``.  Nothing was wrong with the
    collapse.  The span ran across ``tau = 0``, where the declared
    ``x ~ N(0, |tau|)`` is a Dirac narrower than any x-grid, so a grid point that
    lands NEAR zero samples a spike of height ``1 / (tau sqrt(2 pi))`` and gives
    it a cell's width of weight.  Halve the grid and the spurious term halves:
    the value moves by a constant per doubling and never settles.

    **The stress is constructed, not found.**  Whether a span crossing zero
    produces this depends on how close the nearest grid point gets, and that is a
    coincidence of the axis: ``np.linspace(-3, 7, 101)`` lands EXACTLY on zero
    while ``jnp.linspace`` over the same interval misses by ``5.55e-16``.  Red
    line 9 forbids a fixture that pins such a coincidence, so the upper end is
    nudged by a declared amount and the closest approach comes out where the
    nudge puts it -- the same on any IEEE platform, because ``np.linspace`` is
    ``arange(n) * step + start`` and nothing else.

    Measured here, and it is the SEQUENCE that is the finding: as the closest
    approach moves from ``3.1e-15`` to ``3.0e-09``, the gap this span reports
    against a certified value moves from **5.9 nats to 4e-04**.  The number is a
    function of where the grid happened to land and of nothing else, which is why
    it must not be reported at all.
    """
    with jax.enable_x64(True):
        graph = as_graph(models.shared_ancestor())
        good = oracle_joint(
            graph,
            (Span("tau", 0.15, 4.5), Span("x", -1.5, 3.5)),
            resolution=AGREEMENT_FLOOR,
            start=201,
            refinements=6,
        )
        assert good.certified, good.describe()

        seen = []
        for nudge in (1e-14, 1e-12, 1e-10):
            axis = np.linspace(-3.0, 7.0 + nudge, 101)
            approach = float(np.min(np.abs(axis)))
            bad = oracle_joint(
                graph,
                (Span("tau", -3.0, 7.0 + nudge), Span("x", -6.0, 6.0)),
                resolution=AGREEMENT_FLOOR,
                start=101,
                refinements=4,
            )
            assert not bad.certified, (nudge, bad.describe())
            assert "has not reached the demanded resolution" in bad.certificate.refused
            apparent = [
                abs(value - good.value) for _n, value in bad.certificate.history
            ]
            seen.append((approach, max(apparent)))

        # The nudge does what it says, and by a derivation rather than by a
        # measurement: grid point 30 of 101 over (-3, 7 + nudge) sits at
        # 30 * (10 + nudge) / 100 - 3 = 0.3 * nudge, to within the rounding of
        # the intermediate 3.0.
        for (approach, _worst), nudge in zip(seen, (1e-14, 1e-12, 1e-10), strict=True):
            assert abs(approach - 0.3 * nudge) <= 8 * np.spacing(3.0), (approach, nudge)
        # And the reported gap tracks the approach rather than anything about the
        # model: nats when the grid nearly hits the singularity, and four
        # decimal places smaller when it stays a millionth away.
        worst = [value for _approach, value in seen]
        assert worst[0] > 5.0, seen
        assert worst == sorted(worst, reverse=True), seen
        assert worst[2] < worst[0] / 100.0, seen


def test_the_oracle_abstains_when_the_integrand_is_not_a_number():
    """The other half of the same span, and it is the half a grid actually hits.

    ``np.linspace(-3, 7, 101)`` puts a point EXACTLY on ``tau = 0``, where
    ``Normal(0, 0).log_prob`` is ``nan``.  A ``nan`` that reaches the trapezoid
    makes every later value ``nan`` and every increment ``nan``, so the
    refinement rule abstains -- but under the wrong sentence, blaming the
    convergence for something the integrand did.  The reason is named instead,
    with the offending point in it.
    """
    with jax.enable_x64(True):
        assert 0.0 in set(np.linspace(-3.0, 7.0, 101))
        graph = as_graph(models.shared_ancestor())
        bad = oracle_joint(
            graph,
            (Span("tau", -3.0, 7.0), Span("x", -6.0, 6.0)),
            resolution=AGREEMENT_FLOOR,
            start=101,
            refinements=4,
        )
    assert not bad.certified
    assert "not a number at 101 of 10201 grid points" in bad.certificate.refused
    assert "tau=np.float64(0.0)" in bad.certificate.refused
    assert bad.certificate.history == ()


def test_the_oracle_abstains_on_a_span_placed_off_the_mass():
    """A span far from the integrand's mass, refused for the EDGE's reason.

    Not for want of refinement: the increments on such a span shrink perfectly
    well -- it is a smooth integral over the wrong region -- so a rule that only
    watched them would certify a number that is not the evidence.  What says so
    is that the outermost trapezoid cell is LARGER than its neighbour, meaning
    the integrand is still climbing as the span runs out.
    """
    with jax.enable_x64(True):
        graph = as_graph(models.diamond_ancestor())
        away = oracle_joint(
            graph,
            (Span("tau", 20.0, 30.0), Span("x", -6.0, 6.0)),
            resolution=AGREEMENT_FLOOR,
            start=101,
            refinements=3,
        )
    assert not away.certified, away.describe()
    assert "not decaying" in away.certificate.refused, away.certificate.refused
    growing = [edge for edge in away.certificate.edges if not edge.decaying]
    assert growing, away.certificate.edges
    assert any(edge.axis == "tau" for edge in growing), growing
    assert away.certificate.increment_ratio < 1.0, away.certificate.increments


def test_a_span_across_the_degenerate_conditional_aborts_inside_the_collapse():
    """Why ``shared_ancestor``'s span is DECLARED and not extended.

    ``p(x | tau)`` is proper for almost every ``tau`` and undefined at
    ``tau = 0``, which is interior to the declared support.  Land a grid point
    there and ``marginal_log_density`` does not return a refusal -- it aborts
    through ``eqx.error_if``, from inside a traced computation.  Recorded here
    because the R5 plan's 0.15 turns exactly this into Task 7's stop-rule: a
    statistical boundary has to be a filable ``Refusal``, and this one is an
    exception.
    """
    with jax.enable_x64(True):
        graph = as_graph(models.shared_ancestor())
        assert 0.0 in set(np.linspace(-2.0, 6.0, 201))
        with pytest.raises(Exception) as raised:
            oracle_collapsed(
                graph,
                ("x",),
                (Span("tau", -2.0, 6.0),),
                resolution=AGREEMENT_FLOOR,
                start=201,
                refinements=3,
            )
    # The consequence, not the spelling: whatever this is, it is NOT a Refusal
    # the evidence gate could file, and it is not a bayesmith error at all. The
    # equinox abort surfaces as `eqx.EquinoxRuntimeError` when the density is
    # evaluated eagerly and as `jax.errors.JaxRuntimeError` from inside `jit`,
    # which is the shape a sampler would meet it in.
    assert not isinstance(raised.value, (Refusal, BayesmithError))
    assert isinstance(
        raised.value, (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError)
    )
    assert "collapse: the re-triangularisation" in str(raised.value)


def test_the_truncation_bound_measures_the_mass_a_short_span_left_out():
    """The instrument's own error, measured in a direction rather than asserted.

    ``overflowing_outside_latent`` at ``|z| <= 60`` is the R5 plan's own planning
    span and it is 1.3e-03 short of the ``|z| <= 100`` value.  The question the
    certificate exists to answer is whether the oracle KNOWS that, and it does:
    the geometric tail bound reads 1.48e-03, so it bounds the error, and the
    ratio is stable across grids -- the bound is a statement about the span, not
    about how finely the span was sampled.

    〔An earlier version of this test asserted the opposite, that the bound
    UNDERestimates by 1.75x.  That number came of comparing a bound reported for
    BOTH edges against a measurement taken one edge at a time.  Recorded, not
    swapped in quietly: the direction of a bound's error is the only interesting
    thing about it, and getting it backwards is how a certificate becomes a
    decoration.〕
    """
    with jax.enable_x64(True):
        graph = as_graph(models.overflowing_outside_latent())
        wide = oracle_collapsed(
            graph,
            ("w",),
            (Span("z", -100.0, 100.0),),
            resolution=AGREEMENT_FLOOR,
            start=401,
            refinements=9,
        )
        short = [
            oracle_collapsed(
                graph,
                ("w",),
                (Span("z", -60.0, 60.0),),
                resolution=AGREEMENT_FLOOR,
                start=grid,
                refinements=3,
            )
            for grid in (401, 3201)
        ]
    assert wide.certified, wide.describe()
    errors = [abs(one.certificate.history[0][1] - wide.value) for one in short]
    bounds = [one.certificate.truncation for one in short]
    for error, bound in zip(errors, bounds, strict=True):
        assert bound > error, (error, bound)
        assert bound < 1.5 * error, (error, bound)
    # A statement about the SPAN and not about how finely the span was sampled,
    # asserted as a comparison between two measured quantities rather than
    # against a number picked to pass: eight times the grid moves the bound by
    # LESS than the bound's own conservatism -- 1.95e-05 against 3.39e-04,
    # seventeen times smaller.
    assert abs(bounds[0] - bounds[1]) < abs(bounds[0] - errors[0]), (bounds, errors)
    # And the wide span is where the model stops mattering at all.
    assert wide.certificate.truncation < bounds[0] / 1e10


def test_every_certified_value_carries_its_span_and_the_mass_that_span_excludes():
    """R5 0.15: a pinned number without its span is an integral over a support
    nobody declared, and 3.4's eligibility list calls an implicit truncation a
    thing that makes a Bayes factor meaningless."""
    with jax.enable_x64(True):
        collapsed, _joint = _sides("shared_ancestor")
    assert collapsed.certified
    assert [str(span) for span in collapsed.spans] == ["tau in (0.15, 4.5)"]
    excluded = collapsed.excluded_prior_mass["tau"]
    # tau ~ N(2.0, 0.5); the span clears tau = 0 and pays for it in prior mass.
    assert 0.0 < excluded < 1e-3, excluded
    assert "excluded prior mass" in collapsed.describe()


def test_the_excluded_prior_mass_is_recorded_and_is_not_the_abstain_criterion():
    """R5 0.15 said the gate ABSTAINs when the excluded prior mass exceeds the
    band.  Measured, that rule refuses a fixture G4 requires.

    ``overflowing_outside_latent`` declares ``z ~ Cauchy(0, 1e6)``, so a span of
    ``|z| <= 100`` leaves out **99.99 per cent** of the declared prior mass --
    and the value is right, because ``mu`` carries ``exp(|z| / 50)`` and the
    likelihood, not the prior, decides where the integrand's mass is.  What the
    span check must ask about is the INTEGRAND at the edge, which is what
    :class:`~tests.dispatch.residual_oracle.EdgeDecay` measures.  The excluded
    prior mass is recorded beside the value and gates nothing.
    """
    with jax.enable_x64(True):
        collapsed, joint = _sides("overflowing_outside_latent")
    assert collapsed.excluded_prior_mass["z"] > 0.999
    assert collapsed.certified, collapsed.describe()
    assert agreement(collapsed, joint).agree
    assert collapsed.certificate.truncation < 1e-12


def test_oracle_joint_refuses_a_span_list_that_is_not_every_latent():
    """Integrating SOME of the latents gives a conditional, not an evidence, and
    the call site reads identically either way."""
    with jax.enable_x64(True):
        graph = as_graph(models.shared_ancestor())
        with pytest.raises(ValueError, match="every latent"):
            oracle_joint(graph, (Span("tau", 0.15, 4.5),), resolution=AGREEMENT_FLOOR)


def test_a_quadrature_outside_x64_is_refused_rather_than_certified():
    """A float32 trapezoid reaches its float floor four decades early, so the
    convergence rule would certify a value that has not converged."""
    with pytest.raises(RuntimeError, match="double precision"):
        quadrature(lambda values: values["a"], (Span("a", -1.0, 1.0),), resolution=1e-9)


# --------------------------------------------------------------- the census

#: The five fixtures that need constructor arguments, with the arguments
#: ``docs/probes/probe_34_residual_seams.py`` uses.  Duplicated there and here
#: on purpose: the probe is a thing a reader RUNS and this is a thing CI runs,
#: and ``test_the_two_denominators_are_the_same_set_plus_five`` below holds them
#: to the same total rather than to each other's spelling.
PARAMETERISED = {
    "cancelling_sum": {"cancel": 1e2},
    "many_observations": {"count": 3},
    "roundoff_stress": {"big": 1e6, "sigma": 1e-3},
    "sigma_functional_block": {"weights": (1.0, 0.0, -1.0)},
    "wide_plate": {"size": 4},
}


def _shipped_graphs():
    """Every shipped GRAPH, at a stated denominator.

    ``flagged_line`` returns three objects, two of which are graphs, so the
    denominator is GRAPHS and not fixture functions -- two independent censuses
    of this differed because neither said which, and the R5 plan's 0.15 records
    that a count whose denominator is unstated is one the next reader re-derives
    differently.
    """
    for name, function in sorted(vars(models).items()):
        if not inspect.isfunction(function) or name.startswith("_"):
            continue
        if function.__module__ != models.__name__:
            continue
        built = function(**PARAMETERISED.get(name, {}))
        for index, candidate in enumerate(
            built if isinstance(built, tuple) else (built,)
        ):
            try:
                graph = as_graph(candidate)
                graph.nodes  # noqa: B018 - reading it is the check
            except (AttributeError, TypeError):
                continue  # flagged_line's third return is an array
            label = name if not isinstance(built, tuple) else f"{name}[{index}]"
            yield label, graph


def _structural_class(plan) -> str:
    exact = tuple(plan.exact.latents) if plan.exact is not None else ()
    method = plan.exact.method if plan.exact is not None else None
    sampled = tuple(plan.sampled.latents) if plan.sampled is not None else ()
    if exact and not sampled and method == "gcr":
        return "(a) whole-graph exact"
    if exact and sampled and method == "gcr":
        return "(b) exact+residual gcr"
    if exact and sampled:
        return f"(b') exact+residual {method}"
    if not exact and sampled:
        return "(c) all-residual"
    if not exact and not sampled:
        return "(e) no latents"
    return f"(d) {method}"


#: Every shipped graph and the class it compiles to. **Membership, not just
#: counts.** An adversarial review hid one fixture behind a leading underscore
#: and added a new class-(c) graph in its place: the counts still matched and all
#: 36 tests passed, exit 0. A census that pins how MANY there are of each kind,
#: while the WHICH is free to move, is a census of the wrong thing.
CENSUS_MEMBERS = {
    "(a) whole-graph exact": (
        "cancelling_sum",
        "collinear_pair",
        "dangling_deterministic",
        "flagged_line[0]",
        "flagged_line[1]",
        "many_observations",
        "observation_reused_downstream",
        "plated_and_scalar_latents",
        "plated_latent",
        "plated_latent_through_deterministic",
        "prior_held_direction",
        "roundoff_stress",
        "straight_line",
        "tunable_curvature",
        "two_linear_latents",
        "two_observations",
        "two_observations_reverse_sorted_names",
        "unconstrained_latent",
        "wide_plate",
    ),
    "(b') exact+residual gcr+mh": ("mixed_radiometer",),
    "(b) exact+residual gcr": (
        "diamond_ancestor",
        "improper_outside_prior",
        "indirect_ancestor",
        "overflowing_outside_latent",
        "shared_ancestor",
        "three_latent_chain",
    ),
    "(c) all-residual": (
        "affine_only_at_zero",
        "bilinear_pair",
        "bright_and_faint_channels",
        "bright_and_faint_observations",
        "bright_and_faint_pair",
        "cubic_tail",
        "faint_alone",
        "high_snr_curvature",
        "nan_at_negative_probes",
        "non_gaussian_observed_node",
        "orphaned_child_latent",
        "quadratic_claim",
        "student_t_likelihood",
    ),
    "(d) gcr+snis": (
        "contrast_sigma_pair",
        "element_contrast_sigma_plate",
        "hinged_sigma_beyond_the_probe",
        "one_sided_sigma",
        "plated_radiometer",
        "radiometer",
        "radiometer_group",
        "sigma_functional_block",
        "steep_radiometer",
        "sum_sigma_pair",
    ),
    "compile-refused NotGaussian": ("plated_student_t_latent",),
    "compile-refused StructureError": (
        "lying_block_member",
        "lying_observed_node",
        "two_unusable_observed_scales",
        "unusable_observed_scale",
    ),
}


@pytest.fixture(scope="module")
def classified():
    """``label -> class`` for all 54 graphs, compiled ONCE.

    Both denominators are read off this one pass: the 49 are the 54 less the
    five that take constructor arguments, so compiling them twice would be
    twenty seconds spent proving that a subset is a subset.
    """
    import warnings

    out: dict[str, str] = {}
    with jax.enable_x64(True), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for label, graph in _shipped_graphs():
            try:
                plan = bayesmith.compile(graph)
            except BayesmithError as error:
                # NAMED, never swallowed.  "a fixture that legitimately refuses
                # to compile" and "a row the census never reached" are different
                # silences and an `except: continue` reports them as one.
                out[label] = f"compile-refused {type(error).__name__}"
            else:
                out[label] = _structural_class(plan)
    return out


def _census(classified, *, parameterised: bool):
    counts: Counter[str] = Counter()
    members: dict[str, list[str]] = {}
    for label, key in classified.items():
        if not parameterised and label.split("[")[0] in PARAMETERISED:
            continue
        counts[key] += 1
        members.setdefault(key, []).append(label)
    return counts, members


def test_the_structural_class_census_over_the_forty_nine_no_argument_graphs(
    classified,
):
    """R5 Task 2.4. A census run once is a census that goes stale.

    The denominator is stated in the assertion, not in a comment: these are the
    graphs whose fixture functions take no constructor arguments.  R5's whole
    subject is classes (b) and (c), so the two rows that must not drift
    unnoticed are named member by member rather than counted.
    """
    counts, members = _census(classified, parameterised=False)
    assert sum(counts.values()) == 49
    assert counts["(a) whole-graph exact"] == 15
    assert counts["(b) exact+residual gcr"] == 6
    assert counts["(b') exact+residual gcr+mh"] == 1
    assert counts["(c) all-residual"] == 13
    assert counts["(d) gcr+snis"] == 9
    assert (
        sum(value for key, value in counts.items() if key.startswith("compile-refused"))
        == 5
    )
    assert sorted(members["(b) exact+residual gcr"]) == [
        "diamond_ancestor",
        "improper_outside_prior",
        "indirect_ancestor",
        "overflowing_outside_latent",
        "shared_ancestor",
        "three_latent_chain",
    ]
    assert members["(b') exact+residual gcr+mh"] == ["mixed_radiometer"]
    # And the membership of every class, so a swap cannot hide inside a count.
    for key, expected in CENSUS_MEMBERS.items():
        got = tuple(
            label
            for label in sorted(members.get(key, ()))
            if label.split("[")[0] not in PARAMETERISED
        )
        assert got == tuple(
            label for label in expected if label.split("[")[0] not in PARAMETERISED
        ), (key, got, expected)
    # Row (e) -- no latents at all -- is REACHABLE and shipped by nothing.  The
    # R5 plan's 0.3 says it is the only graph that will still reach
    # `evidence_residual_integral_required` after Task 7 widens the gate, so
    # Task 7 has to build one; this row says it does not exist yet.
    assert "(e) no latents" not in counts


def test_the_five_parameterised_fixtures_move_only_classes_a_and_d(classified):
    """The same census at the other denominator, and what the difference is.

    Classes (b) and (c) -- R5's subject -- are identical at 49 and at 54, which
    is why the plan's 0.3 keeps both numbers on the page rather than restating
    one of them silently.
    """
    counts, members = _census(classified, parameterised=True)
    assert sum(counts.values()) == 54
    assert {key: tuple(sorted(value)) for key, value in members.items()} == {
        key: tuple(value) for key, value in CENSUS_MEMBERS.items()
    }
    assert counts["(a) whole-graph exact"] == 19
    assert counts["(b) exact+residual gcr"] == 6
    assert counts["(b') exact+residual gcr+mh"] == 1
    assert counts["(c) all-residual"] == 13
    assert counts["(d) gcr+snis"] == 10
    assert (
        sum(value for key, value in counts.items() if key.startswith("compile-refused"))
        == 5
    )


def test_only_the_collapsed_side_reaches_the_elimination():
    """Red line 7: the tier ordering asserted as a CALL, not as a spelling.

    ``oracle_collapsed`` is tier 5 because it goes through ``collapse_graph``
    and ``oracle_joint`` is tier 1 because it does not.  A test that read the
    two function bodies for the string ``collapse_graph`` would be walked past
    by ``import collapse_graph as _cg`` -- ``CLAUDE.md`` records two guards
    defeated in one day by exactly that move.  So the elimination is REPLACED
    and the two sides are asked which of them noticed.
    """
    called: list[tuple[str, ...]] = []
    real = residual_oracle.collapse_graph

    def watched(graph, exact, residual):
        called.append(tuple(exact))
        return real(graph, exact, residual)

    residual_oracle.collapse_graph = watched
    try:
        with jax.enable_x64(True):
            graph = as_graph(models.diamond_ancestor())
            spans = (Span("tau", -4.0, 8.0), Span("x", -6.0, 6.0))
            # The replacement is installed BEFORE this call, which is the whole
            # guard. An adversarial review defeated the earlier ordering -- it
            # ran oracle_joint first and installed the watcher afterwards, so a
            # call from oracle_joint was invisible by construction and the
            # bypass left all 36 tests green, exit 0. A guard that starts
            # watching after the act it forbids is not watching.
            oracle_joint(
                graph, spans, resolution=AGREEMENT_FLOOR, start=101, refinements=3
            )
            assert called == [], called
            oracle_collapsed(
                graph,
                ("x",),
                (spans[0],),
                resolution=AGREEMENT_FLOOR,
                start=201,
                refinements=3,
            )
    finally:
        residual_oracle.collapse_graph = real
    assert called == [("x",)]


def test_the_band_never_claims_to_be_tighter_than_the_declared_floor():
    """D111's floor is load-bearing, and this is the test that says so.

    Without it the comparison on ``indirect_ancestor`` still passes -- the two
    sides agree to 22 ULP against a bound of 34 -- so the floor is a fix that
    red line 13 would call REAL AND UNGRADED unless something asserts the band
    actually uses it.  Here the sum of the two sides' own bounds is measured to
    be BELOW the floor, and the band is measured to be the floor: delete the
    ``max`` in :func:`~tests.dispatch.residual_oracle.agreement` and this goes
    red where the comparison itself would not.
    """
    with jax.enable_x64(True):
        verdict = agreement(*_sides("indirect_ancestor"))
    floor = AGREEMENT_FLOOR * max(1.0, abs(verdict.left.value))
    own = verdict.left.certificate.bound + verdict.right.certificate.bound
    assert own < floor / 1e3, (own, floor)
    assert verdict.band == floor, (verdict.band, floor, own)
    assert math.isclose(AGREEMENT_FLOOR, 1e-9)


# ------------------------------------------- the two bounds' own failure modes


def _inverse_square(values):
    """``log 1 / (1 + z**2)`` -- a POLYNOMIAL tail whose truncation is exact.

    ``int 1/(1+z**2) dz`` over the whole line is ``pi`` and over ``(-S, S)`` is
    ``2 arctan(S)``, so the fraction of the mass a span of ``S`` leaves out is
    ``2 arctan(1/S) / pi`` with no quadrature in it anywhere. That makes it the
    one integrand here whose edge behaviour can be graded against a TRUTH rather
    than against another grid.
    """
    return -jnp.log1p(values["z"] ** 2)


def test_the_edge_bound_covers_a_polynomial_tail():
    """A geometric model of a ``z**-p`` tail is short by ``p / (p - 1)``.

    For ``C t**-p`` the truth past ``S`` is ``C S**(1-p) / (p - 1)`` while the
    geometric sum of the cells comes out ``C S**(1-p) / p``, so on this ``p = 2``
    integrand the geometric bound is exactly HALF the mass it is bounding. A
    bound that excludes the truth is not a bound, which is what
    :func:`~tests.dispatch.residual_oracle._power_tail` exists to fix.

    Graded against the closed form, at three spans, so the fix is measured
    rather than argued.
    """
    with jax.enable_x64(True):
        for span in (200.0, 1000.0, 5000.0):
            found = quadrature(
                _inverse_square,
                (Span("z", -span, span),),
                resolution=AGREEMENT_FLOOR,
                start=8001,
                refinements=4,
            )
            assert found.certified, found.describe()
            exact = 2.0 * math.atan(1.0 / span) / math.pi
            bound = found.certificate.truncation
            assert bound >= exact, (span, exact, bound)
            assert bound < 1.2 * exact, (span, exact, bound)
            # The geometric half of the same bound, recomputed from the ratio the
            # certificate reports: it is half the truth, and it is what the
            # oracle would have published without the power-law term.
            edge = max(one.ratio for one in found.certificate.edges)
            assert 0.99 < edge < 1.0, found.certificate.edges


def test_a_first_grid_above_the_budget_is_refused_rather_than_allocated():
    """``start=201`` is a sensible first grid on one axis and 1.63e9 points on
    four. Allocating it is a quarter of an hour and then a memory death, which
    says nothing about the model, so the budget is checked before the grid.

    Asserted at a SMALL budget rather than by handing the oracle the 1.63e9
    grid: with the guard deleted, this way the test fails in a second, and the
    real way it hangs -- and red line 13 asks whether restoring the old code
    goes red, which a run that never finishes cannot answer.
    """
    with jax.enable_x64(True), pytest.raises(ValueError, match="above the .* budget"):
        quadrature(
            lambda values: -sum(value**2 for value in values.values()),
            tuple(Span(name, -3.0, 3.0) for name in ("a", "b", "c")),
            resolution=AGREEMENT_FLOOR,
            start=101,
            max_points=1000,
        )


def test_the_refinement_abstains_when_the_next_grid_would_exceed_the_budget():
    """R5 Task 3.2's stop-rule, as a property of the oracle rather than of a
    script: the dimension at which quadrature stops being an oracle is the
    dimension at which its next grid stops fitting, and it says which grid."""
    with jax.enable_x64(True):
        found = quadrature(
            _inverse_square,
            (Span("z", -5.0, 5.0),),
            resolution=1e-18,  # unreachable, so only the budget can stop it
            start=101,
            refinements=8,
            max_points=1000,
        )
    assert not found.certified
    assert "above the 1.000e+03 budget" in found.certificate.refused
    assert found.certificate.history[-1][0] == 801


# ------------------------- the exact route's error law, and who shares it


def _scaled_prior_pair(mean, width, *, n=4, sigma=0.5):
    """``w ~ N(mean, width)``, ``mu = w X``, ``d ~ N(mu, sigma)``.

    One latent, one observed node, and a prior whose mean and width are the
    dial. The closed form is ``N(d ; A mean, width**2 A A^T + sigma**2 I)``, so
    all three routes -- shipped, closed form, quadrature -- can be put side by
    side at any point of the ``(mean, width)`` plane.
    """
    grid = np.linspace(1.0, 2.0, n)
    data = np.array([1.05, 1.35, 1.62, 1.94])[:n]

    def model():
        columns = bayesmith.const("X", jnp.asarray(grid))
        latent = bayesmith.sample("w", lambda: dist.Normal(mean, width))
        prediction = bayesmith.det(
            "mu", lambda w_, x_: w_ * x_, latent, columns, linear_in=("w",)
        )
        bayesmith.observe(
            "d", lambda mu_: dist.Normal(mu_, sigma), prediction, obs=jnp.asarray(data)
        )

    return as_graph(bayesmith.trace(model)), grid, data, sigma


def test_the_oracle_detects_the_exact_routes_error_law_rather_than_sharing_it():
    """Why ``oracle_joint`` is tier 1, asserted as a mechanism and not a label.

    The exact linear-Gaussian route carries an error law ``eps * |m| / s`` driven
    by the eliminated block's PRIOR MEAN. ``oracle_joint`` does not share it, and
    the reason is structural rather than lucky: ``log_joint`` evaluates each
    node's own ``log_prob`` and never reaches ``nuisance_prior``, so there is no
    ``m / s`` entry to cancel in a QR.

    Measured here at ``|m| / s = 4e15``: the shipped route is **0.28 nats** from
    the closed form and this oracle is **exactly** on it. That row is the reason
    the R5 plan's 0.4 no longer asks the oracle to abstain in this region -- an
    abstain would have deleted the detection -- and why the block's ratio is
    recorded instead.
    """
    with jax.enable_x64(True):
        rows = []
        for mean, width in ((0.4, 1.0), (0.4, 1e-8), (0.4, 1e-16)):
            graph, grid, data, sigma = _scaled_prior_pair(mean, width)
            shipped = float(collapse_module.marginal_log_density(graph, ("w",), {}))
            closed = residual_models.gaussian_log_evidence(
                grid.reshape(-1, 1),
                np.zeros(grid.size),
                data,
                np.array([mean]),
                np.array([width]),
                sigma,
            )
            found = oracle_joint(
                graph,
                (Span("w", mean - 9.0 * width, mean + 9.0 * width),),
                resolution=AGREEMENT_FLOOR,
                start=401,
                refinements=6,
            )
            assert found.certified, found.describe()
            rows.append((abs(mean) / width, shipped - closed, found.value - closed))

    benign, middling, extreme = rows
    # At |m|/s = 0.4 the two routes are indistinguishable.
    assert abs(benign[1]) < 1e-12 and abs(benign[2]) < 1e-12, rows
    # By 4e15 the shipped route is a quarter of a nat out...
    assert abs(extreme[1]) > 0.2, rows
    # ...and the quadrature is still on the closed form, by twelve orders more
    # than the route it grades. That ordering is the whole claim.
    assert abs(extreme[2]) < 1e-12, rows
    assert abs(extreme[1]) > 1e11 * max(abs(extreme[2]), EPS)
    # And the degradation is monotone in |m|/s, so it is one law and not noise.
    assert abs(benign[1]) <= abs(middling[1]) <= abs(extreme[1]), rows


def test_the_eliminated_blocks_ratio_travels_with_the_collapsed_value():
    """R5 plan 0.4, in its corrected form: record the ratio, gate nothing.

    A number without its domain is the defect 0.15 exists to prevent, and this
    is the domain a COLLAPSED value depends on -- the region of the exact
    route's two error laws. Measured over every fixture Wave B grades: the
    eliminated block's prior mean is exactly zero in all four class-(b) cases,
    and 0.5625 for the mixture fixture's ``b``. So nothing Wave B integrates
    goes near the region, and the record says so rather than a comment.
    """
    with jax.enable_x64(True):
        seen = {}
        for name in sorted(CLASS_B):
            collapsed, _joint = _sides(name)
            seen[name] = collapsed.exact_block_ratio
        joint = oracle_joint(
            as_graph(models.diamond_ancestor()),
            (Span("tau", -4.0, 8.0), Span("x", -6.0, 6.0)),
            resolution=AGREEMENT_FLOOR,
            start=101,
            refinements=3,
        )
        shared = _sides("shared_ancestor")[0]
    # Named per fixture, not collapsed to a set. Four are zero because their
    # eliminated block's prior is centred at the origin; `shifted_block_prior`
    # puts the residual latent IN that prior's mean, so its ratio is
    # |tau| / 0.3 and the recorded figure must be the largest the span reaches
    # -- 4.0 / 0.3, not the centre's 2.2 / 0.3 and not the low end's 0.4 / 0.3.
    # Three mutants survived the whole suite before this fixture existed:
    # forcing the mean to zero, taking a min over the probe points, and probing
    # only the centre.
    assert seen == {
        "diamond_ancestor": 0.0,
        "indirect_ancestor": 0.0,
        "outside_observation_first": 0.0,
        "outside_observation_pair": 0.0,
        "overflowing_outside_latent": 0.0,
        "shared_ancestor": 0.0,
        "shifted_block_prior": pytest.approx(4.0 / 0.3),
    }, seen
    assert seen["shifted_block_prior"] > 4.0 * (2.2 / 0.3) / 3.0, seen
    # oracle_joint eliminates nothing, so it has no such domain to declare and
    # says None rather than zero -- "not asked" and "asked, and it is zero" are
    # the distinction this repository has paid for most often.
    assert joint.exact_block_ratio is None
    assert "eliminated block |m|/s" in shared.describe()


def test_the_band_carries_the_truncated_mass_and_a_verdict_depends_on_it():
    """The truncation term is load-bearing, and this is what says so.

    An adversarial review dropped ``truncation`` from
    :attr:`~tests.dispatch.residual_oracle.Certificate.bound` and all 36 tests
    stayed green, exit 0. Every comparison in this file happened to put both
    sides on the SAME span, so their truncations were near-equal and cancelled
    out of the gap, leaving the term true but never decisive.

    Here the two sides are deliberately given DIFFERENT spans -- the collapsed
    route over ``|z| <= 60`` and the oracle over ``|z| <= 100`` -- so the gap IS
    the excluded mass. Measured: the gap is 1.27e-03, the band is 1.61e-03 and
    the truncated mass is 100.00 per cent of it. Strike that term and the band
    falls to 2.05e-08, which the gap exceeds by a factor of 6.2e+04.

    So the assertion is not "the band contains a truncation term" -- that would
    be a spelling. It is that this comparison AGREES, and that it could not have
    without the term.
    """
    with jax.enable_x64(True):
        graph = as_graph(models.overflowing_outside_latent())
        short = oracle_collapsed(
            graph,
            ("w",),
            (Span("z", -60.0, 60.0),),
            resolution=AGREEMENT_FLOOR,
            start=401,
            refinements=9,
        )
        wide = oracle_joint(
            graph,
            (Span("z", -100.0, 100.0), Span("w", -8.0, 8.0)),
            resolution=AGREEMENT_FLOOR,
            start=201,
            refinements=7,
        )
    verdict = agreement(short, wide)
    assert verdict.agree, verdict.describe()
    truncated = short.certificate.truncation + wide.certificate.truncation
    without = verdict.band - truncated
    assert without > 0.0, (verdict.band, truncated)
    assert abs(verdict.gap) > 1e3 * without, (verdict.gap, without)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"resolution": 0.0}, "no looser than"),
        ({"resolution": -1.0}, "no looser than"),
        ({"resolution": math.nan}, "no looser than"),
        ({"resolution": math.inf}, "no looser than"),
        ({"resolution": 1.0}, "no looser than"),
        # The one the FIRST repair still admitted, and the reason the bound is
        # now the declared level rather than an interval around it.
        ({"resolution": 0.999}, "no looser than"),
        ({"resolution": 1e-3}, "no looser than"),
        ({"resolution": AGREEMENT_FLOOR * 10}, "no looser than"),
        ({"resolution": AGREEMENT_FLOOR, "start": 2}, "at least three points"),
    ],
)
def test_the_declared_guards_refuse_what_they_name(kwargs, match):
    """Holes two adversarial reviews walked through in turn, and the bound that
    leaves no range to walk through.

    The first review found ``not resolution > 0.0`` admitting ``inf``: the tail
    test became ``tail <= inf``, vacuous at the very first grid, and the oracle
    CERTIFIED a value 2.5e-04 wrong. The repair was ``0 < resolution < 1``. The
    second review swept what THAT still admits and found ``resolution=0.999``
    certifying ``cauchy_residual_pair`` **0.873 nats** wrong while reporting a
    refinement tail of 0.36 -- 3500 times worse than the hole it closed.

    Both times the admitted range was entirely unexercised: nothing in the suite
    passed any level but the declared one. So the bound is now the declared
    level itself. A caller may demand MORE convergence than D111 and may not
    demand less, and there is no interval left to sweep.
    """
    with jax.enable_x64(True), pytest.raises(ValueError, match=match):
        quadrature(
            lambda values: -(values["z"] ** 2),
            (Span("z", -4.0, 4.0),),
            **{"resolution": AGREEMENT_FLOOR, **kwargs},
        )


def test_every_admitted_resolution_certifies_the_right_number():
    """The other half: what the guard lets through must be safe.

    Refusing 0.999 is only half an answer -- the levels still admitted have to
    be ones that actually converge. Swept against ``cauchy_residual_pair``'s
    closed form, which is exact.
    """
    with jax.enable_x64(True):
        exact = residual_models.cauchy_residual_pair_log_evidence()
        graph = as_graph(residual_models.cauchy_residual_pair())
        for level in (AGREEMENT_FLOOR, AGREEMENT_FLOOR / 10, AGREEMENT_FLOOR / 1e3):
            found = oracle_joint(
                graph,
                (Span("z", -1000.0, 1000.0),),
                resolution=level,
                start=4001,
                refinements=6,
            )
            assert found.certified, (level, found.describe())
            band = found.certificate.bound + AGREEMENT_FLOOR * max(1.0, abs(exact))
            assert abs(found.value - exact) <= band, (level, found.describe())


def test_the_recorded_domains_refuse_to_be_computed_in_single_precision():
    """``quadrature`` guarded x64 and the two helpers beside it did not.

    Called outside ``jax.enable_x64`` they returned float32-computed numbers
    with no complaint -- an excluded prior mass and a block ratio that look
    exactly like the double-precision ones and are not. Both now refuse.
    """
    graph = as_graph(models.shared_ancestor())
    spans = (Span("tau", 0.15, 4.5),)
    with pytest.raises(RuntimeError, match="double precision"):
        residual_oracle.excluded_prior_mass(graph, spans)
    with pytest.raises(RuntimeError, match="double precision"):
        residual_oracle.block_prior_ratio(graph, ("x",), spans)


# ------------------- what the second adversarial review found still ungraded


def test_only_the_elimination_itself_is_watched_not_one_name_for_it():
    """Red line 7 again, one level down, and the review walked through the first
    attempt.

    The previous guard replaced ``residual_oracle.collapse_graph`` -- ONE module
    attribute. A function-local ``from bayesmith.dispatch.collapse import
    collapse_graph as _cg`` inside ``oracle_joint`` binds from the source module
    at call time, never reads the patched name, and left the guard at 1 passed
    and the suite at 47 passed, exit 0. That is ``CLAUDE.md``'s
    ``ProducerRef as _PR`` verbatim, inside the guard written to avoid it.

    So all three doors are watched: the name ``residual_oracle`` imported, the
    name in the SOURCE module that a local import would bind, and
    ``marginal_log_density`` -- the half that actually computes, which
    ``CollapsedEvidence.log_density`` calls out of ``collapse.py``'s own globals
    and which no import into this module could avoid.
    """
    seen: list[str] = []
    real_graph = collapse_module.collapse_graph
    real_density = collapse_module.marginal_log_density

    def watch_graph(graph, exact, residual):
        seen.append("collapse_graph")
        return real_graph(graph, exact, residual)

    def watch_density(graph, exact, values):
        seen.append("marginal_log_density")
        return real_density(graph, exact, values)

    collapse_module.collapse_graph = watch_graph
    collapse_module.marginal_log_density = watch_density
    residual_oracle.collapse_graph = watch_graph
    try:
        with jax.enable_x64(True):
            graph = as_graph(models.diamond_ancestor())
            spans = (Span("tau", -4.0, 8.0), Span("x", -6.0, 6.0))
            oracle_joint(
                graph, spans, resolution=AGREEMENT_FLOOR, start=101, refinements=3
            )
            assert seen == [], seen
            oracle_collapsed(
                graph,
                ("x",),
                (spans[0],),
                resolution=AGREEMENT_FLOOR,
                start=201,
                refinements=3,
            )
    finally:
        collapse_module.collapse_graph = real_graph
        collapse_module.marginal_log_density = real_density
        residual_oracle.collapse_graph = real_graph
    assert "collapse_graph" in seen
    assert "marginal_log_density" in seen


#: Which latents each block-carrying graph puts on which side. **The class label
#: never names a latent**, so an adversarial review replaced the exact tuple with
#: a constant and both census tests still passed: the counts held, the membership
#: held, and nothing asked WHICH latents were eliminated.
BLOCK_SPLIT = {
    "diamond_ancestor": (("x",), ("tau",)),
    "improper_outside_prior": (("w",), ("z",)),
    "indirect_ancestor": (("x",), ("tau",)),
    "mixed_radiometer": (("w",), ("tau",)),
    "overflowing_outside_latent": (("w",), ("z",)),
    "shared_ancestor": (("x",), ("tau",)),
    "three_latent_chain": (("y",), ("tau", "x")),
}


def test_the_census_pins_which_latents_are_eliminated_not_only_how_many():
    """R5's whole subject is which side of the split a latent lands on.

    ``three_latent_chain`` is the row that makes this more than bookkeeping: it
    is the only shipped graph whose residual has TWO latents, and a classifier
    that ejected the wrong one would keep every count and every membership
    exactly as they are.
    """
    import warnings

    found = {}
    with jax.enable_x64(True), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for label, graph in _shipped_graphs():
            try:
                plan = bayesmith.compile(graph)
            except BayesmithError:
                continue
            exact = tuple(plan.exact.latents) if plan.exact is not None else ()
            sampled = tuple(plan.sampled.latents) if plan.sampled is not None else ()
            if exact and sampled:
                found[label] = (exact, sampled)
    assert found == BLOCK_SPLIT


def _two_gaussians(values):
    """A product of unit Gaussians centred at 0 and at 1 -- an EXACT reference.

    The excluded mass past any span is an ``erf``, so every edge the oracle
    reports can be graded against a truth rather than against another grid.
    """
    return -0.5 * values["z"] ** 2 - 0.5 * (values["w"] - 1.0) ** 2


def _gaussian_tails(lower, upper, mean):
    """``(below, above)`` -- the exact mass a span leaves out, per side."""
    below = 0.5 * (1.0 + math.erf((lower - mean) / math.sqrt(2.0)))
    above = 0.5 * (1.0 - math.erf((upper - mean) / math.sqrt(2.0)))
    return below, above


def test_every_edge_bounds_the_mass_beyond_it_and_the_axes_are_walked_separately():
    """Three survivors at once, and one exact reference settles all of them.

    The fixtures could not: in every graded comparison the second axis's tail
    runs 1e-33 to 1e-282, and the one fixture where truncation decides has
    rho = 0.9542 on BOTH edges, so an asymmetric or single-axis defect
    reproduces the right total. Measured, these survived the whole suite:
    ``_edges`` walking only axis 0; the truncation reported as twice the lower
    edges; and ``rho`` squared, which understates every tail.

    Here the spans are deliberately asymmetric and material on the SECOND axis
    only -- ``z`` cut at 6 sigma, ``w`` cut at 2.0 sigma below and 1.5 above --
    and the property asserted is the one that makes the bound a bound: **each
    edge's reported fraction is at least the exact mass beyond that edge.**
    Halving any of them, which is what squaring rho does, drops two of the four
    below their own truth.
    """
    spans = (Span("z", -6.0, 6.0), Span("w", -1.0, 2.5))
    with jax.enable_x64(True):
        found = quadrature(
            _two_gaussians, spans, resolution=AGREEMENT_FLOOR, start=201, refinements=6
        )
    edges = {(edge.axis, edge.side): edge for edge in found.certificate.edges}
    # Both axes are walked, not just the first.
    assert {axis for axis, _side in edges} == {"z", "w"}

    exact = {
        ("z", "lower"): _gaussian_tails(-6.0, 6.0, 0.0)[0],
        ("z", "upper"): _gaussian_tails(-6.0, 6.0, 0.0)[1],
        ("w", "lower"): _gaussian_tails(-1.0, 2.5, 1.0)[0],
        ("w", "upper"): _gaussian_tails(-1.0, 2.5, 1.0)[1],
    }
    for key, truth in exact.items():
        assert edges[key].fraction >= truth, (key, edges[key].fraction, truth)
        assert edges[key].fraction < 5.0 * truth, (key, edges[key].fraction, truth)

    # The two sides of `w` are NOT equal, so a total assembled as twice one of
    # them is a different number -- which is what the surviving mutant did.
    assert edges[("w", "upper")].fraction > 2.0 * edges[("w", "lower")].fraction
    # And the reported total is their sum, dominated by the axis that is cut.
    assert found.certificate.truncation == pytest.approx(
        sum(edge.fraction for edge in found.certificate.edges)
    )
    assert edges[("w", "upper")].fraction > 1e6 * edges[("z", "upper")].fraction


@pytest.mark.parametrize(
    ("mean", "width", "expected", "what"),
    [
        (-2.0, 0.5, 4.0, "a NEGATIVE mean is as far from zero as a positive one"),
        ((1.0, 3.0), 0.5, 6.0, "the LARGEST mean is the worst cell"),
        (1.0, (0.5, 2.0), 2.0, "the SMALLEST width is the worst cell"),
        ((-4.0, 1.0), (0.25, 2.0), 16.0, "both reductions at once"),
        (0.0, 1.0, 0.0, "a centred prior has no ratio to report"),
        (1.0, 0.0, math.inf, "a zero width is not a ratio, it is infinity"),
    ],
)
def test_the_block_ratio_reduces_to_the_worst_cell(mean, width, expected, what):
    """Graded directly, because through a graph it cannot be graded at all.

    ``block.prior_mean[member]`` and ``block.prior_std[member]`` are scalars of
    shape ``()`` in 6 of 6 fixtures, and over one element ``max``, ``min``,
    ``first`` and ``sum`` are the same function. An adversarial review swapped
    the two reductions, swapped them with each other, and dropped the ``abs``,
    and all three survived the whole suite. Building a plated exact block to
    grade two calls to numpy would be the wrong instrument; this is the right
    one, and the coverage statement is recorded beside it: **no graph in this
    package has a vector-valued or negative-mean exact block**, so the shipped
    fixtures cannot reach any of these rows.
    """
    assert residual_oracle.worst_ratio(mean, width) == pytest.approx(expected), what


def test_the_probe_visits_both_ends_and_the_centre_of_every_axis():
    """``2d + 1`` points, and which ones matters.

    The only fixture whose ratio varies is monotone increasing, so its maximum
    sits exactly at its span's upper end -- and a probe that read the upper end
    and nothing else satisfied the test written to pin this. Both "centre only"
    and "upper end only" survived.

    **What this does NOT establish, stated because the review asked for it:** a
    worst point in the INTERIOR of a span would be missed by all three probes,
    and no graph in this package has one, so the probe's adequacy for that case
    is unproven rather than proven.
    """
    spans = (Span("a", -2.0, 4.0), Span("b", 0.0, 10.0))
    points = residual_oracle.probe_points(spans)
    assert len(points) == 5
    assert points[0] == {"a": 1.0, "b": 5.0}
    assert {point["a"] for point in points} == {-2.0, 1.0, 4.0}
    assert {point["b"] for point in points} == {0.0, 5.0, 10.0}
    # Every point is a complete assignment: a probe missing an axis would build
    # the block at a default rather than where the caller declared.
    assert all(set(point) == {"a", "b"} for point in points)


def test_the_guards_beside_the_pinned_ones():
    """Four checks the repair batch left unreached, all of which survived.

    The batch pinned the guards the first review named and no others -- the
    duplicate-axis check sits three lines from the ``start >= 3`` check it did
    pin. ``oracle_joint``'s span-set guard was already pinned; its twin on the
    collapsed side was not.
    """
    with jax.enable_x64(True):
        with pytest.raises(ValueError, match="names an axis twice"):
            quadrature(
                lambda values: -(values["z"] ** 2),
                (Span("z", -1.0, 1.0), Span("z", -2.0, 2.0)),
                resolution=AGREEMENT_FLOOR,
            )
        with pytest.raises(ValueError, match="not an interval"):
            Span("z", 2.0, 2.0)
        with pytest.raises(ValueError, match="not an interval"):
            Span("z", 3.0, -3.0)
        with pytest.raises(ValueError, match="every residual latent"):
            oracle_collapsed(
                as_graph(models.shared_ancestor()),
                ("x",),
                (Span("x", -1.0, 1.0),),
                resolution=AGREEMENT_FLOOR,
            )


def test_an_abstaining_side_is_reported_as_no_comparison_and_never_as_agreement():
    """``agreement()`` on an ABSTAIN, which nothing had ever called.

    Every caller in this file asserts ``.certified`` before comparing, so the
    whole "no comparison" path was dead code -- and a mutant that made it return
    ``gap=0.0, band=inf``, so that ``agree`` reads True, survived the suite. A
    future comparison written against ``verdict.agree`` alone would then read an
    abstention as agreement, which is the one reading it must never have.
    """
    with jax.enable_x64(True):
        graph = as_graph(models.diamond_ancestor())
        good = oracle_joint(
            graph,
            (Span("tau", -4.0, 8.0), Span("x", -6.0, 6.0)),
            resolution=AGREEMENT_FLOOR,
            start=201,
            refinements=6,
        )
        away = oracle_joint(
            graph,
            (Span("tau", 20.0, 30.0), Span("x", -6.0, 6.0)),
            resolution=AGREEMENT_FLOOR,
            start=101,
            refinements=3,
        )
    assert good.certified and not away.certified
    verdict = agreement(good, away)
    assert verdict.gap is None and verdict.band is None
    assert not verdict.agree
    assert verdict.margin == math.inf
    assert "no comparison" in verdict.describe()
    assert "ABSTAIN" in away.describe()


def test_the_block_ratio_takes_the_worst_MEMBER_not_the_first(monkeypatch):
    """Every exact block in this package has exactly one member, 7 of 7.

    So ``for member in block.names`` and ``block.names[:1]`` are the same loop
    on everything that can be built here, and a mutant that reads only the first
    member is EQUIVALENT rather than surviving -- a distinction worth making,
    because the two have different remedies. The loop is graded by handing
    :func:`~tests.dispatch.residual_oracle.block_prior_ratio` a block that no
    graph in this package produces, with the worst member second.
    """

    @dataclasses.dataclass(frozen=True)
    class _Block:
        names: tuple[str, ...] = ("first", "second")
        prior_mean: Mapping[str, float] = types.MappingProxyType(
            {"first": 0.5, "second": 8.0}
        )
        prior_std: Mapping[str, float] = types.MappingProxyType(
            {"first": 1.0, "second": 0.25}
        )

    monkeypatch.setattr(
        "bayesmith.exact.block.unchecked_operator",
        lambda graph, names, at, probe_gaussian: _Block(),
    )
    with jax.enable_x64(True):
        worst = residual_oracle.block_prior_ratio(
            as_graph(models.shared_ancestor()), ("x",), (Span("tau", 0.15, 4.5),)
        )
    # 8.0 / 0.25 = 32, from the SECOND member; the first would give 0.5.
    assert worst == pytest.approx(32.0)


@pytest.mark.parametrize(
    ("increment", "ratio", "expected"),
    [
        (1e-6, 0.25, 1e-6 * 0.25 / 0.75),
        (1e-6, 0.5, 1e-6),
        (1e-6, 0.0, 0.0),
        # At and above 1 the series does not converge and there is no bound.
        (1e-6, 1.0, math.inf),
        (1e-6, 1.5, math.inf),
        (1e-6, 3.0, math.inf),
    ],
)
def test_the_geometric_tail_is_infinite_wherever_it_is_not_a_bound(
    increment, ratio, expected
):
    """A ratio in (1, 2) makes the closed form NEGATIVE, and negative passes.

    ``increment * ratio / (1 - ratio)`` is a bound only while ``ratio < 1``.
    Relaxing the test to ``ratio < 2.0`` -- which an adversarial review did --
    leaves the expression finite and NEGATIVE for a ratio in (1, 2), and a
    negative tail compares less than any demanded level, so a refinement that is
    DIVERGING certifies. The guard was doing two jobs, deciding convergence and
    keeping the arithmetic meaningful, and only one of them was pinned.

    ``ratio = 1.0`` is the row that separates the two spellings: under
    ``< 1.0`` it is `inf`, and under ``< 2.0`` it is a division by zero.
    """
    assert residual_oracle._geometric_tail(increment, ratio) == pytest.approx(expected)


def test_the_geometric_arm_decides_where_the_power_fit_has_no_meaning():
    """``max(geometric, power)`` is ``power`` on every edge this package reaches.

    Measured over 24 edges spanning three tail shapes -- a Gaussian, an
    exponentially cut Cauchy prior and a genuine ``z**-4`` tail -- the power arm
    wins **24 of 24**, by factors of 1.09 to 2.11. So the geometric arm never
    decides a reported bound in this package, and an adversarial review squared
    ``rho`` and nothing moved.

    It is not dead weight, though: :func:`~tests.dispatch.residual_oracle._power_tail`
    returns ``0.0`` when the fit has no meaning, and an INNER edge -- one closer
    to the origin than its neighbour -- is exactly that case. A span of
    ``(2, 10)`` around a mode at 8 has one: the geometric arm is the only bound
    there, and squaring ``rho`` drops it below the mass it is supposed to cover.
    """
    with jax.enable_x64(True):
        found = quadrature(
            lambda values: -0.5 * (values["z"] - 8.0) ** 2,
            (Span("z", 2.0, 10.0),),
            resolution=AGREEMENT_FLOOR,
            start=201,
            refinements=6,
        )
    edges = {(edge.axis, edge.side): edge for edge in found.certificate.edges}
    # The power fit is refused at an inner edge, so the geometric arm is alone.
    assert residual_oracle._power_tail(1e-3, 2e-3, 2.05, 2.15, 0.04) == 0.0
    lower_truth = 0.5 * (1.0 + math.erf((2.0 - 8.0) / math.sqrt(2.0)))
    upper_truth = 0.5 * (1.0 - math.erf((10.0 - 8.0) / math.sqrt(2.0)))
    assert edges[("z", "lower")].fraction >= lower_truth
    assert edges[("z", "lower")].fraction < 2.0 * lower_truth
    assert edges[("z", "upper")].fraction >= upper_truth


def test_an_edge_only_just_above_one_is_still_the_mass_leaving_the_span():
    """The off-the-mass ABSTAIN was pinned only far from its own boundary.

    ``diamond_ancestor`` over ``tau in (20, 30)`` has an edge rho of 2.5 to 24.6,
    so a threshold moved from ``rho >= 1.0`` to ``rho >= 1.5`` kept refusing it
    and survived. **No fixture has an edge rho in [1.0, 1.5)** -- the interval
    the move actually changes.

    A slowly rising integrand does: ``0.01 * z`` over ``(0, 10)`` puts rho at
    1.000125, and its mass is genuinely outside the span. The verdict has to be
    ABSTAIN there for the same reason it is at 24.6, and only the threshold's
    exact placement decides it.
    """
    with jax.enable_x64(True):
        found = quadrature(
            lambda values: 0.01 * values["z"],
            (Span("z", 0.0, 10.0),),
            resolution=AGREEMENT_FLOOR,
            start=201,
            refinements=3,
        )
    upper = next(edge for edge in found.certificate.edges if edge.side == "upper")
    assert 1.0 < upper.ratio < 1.1, upper
    assert not upper.decaying
    assert upper.fraction == math.inf
    assert not found.certified
    assert "not decaying" in found.certificate.refused


def test_the_certificate_says_did_not_run_apart_from_ran_and_found_nothing():
    """Red line 14, in the field a consumer actually reads.

    ``truncation`` used to have two states and three meanings. ``inf`` said both
    "an edge is growing, so the mass is outside this span" and "the integrand was
    not a number, so there were no edges to measure at all" -- distinguishable
    only by ALSO consulting ``history == ()``, which is the indirection the rule
    exists to remove. It is now ``None`` for the second.

    Measured across every outcome the oracle has:

    ========================  ======  =====  ============
    outcome                   grids   edges  truncation
    ========================  ======  =====  ============
    certified                      3      4  a float
    integrand is not a number      0      0  ``None``
    an edge is not decaying        3      4  ``inf``
    ========================  ======  =====  ============

    And ``bound`` is ``inf`` when the edge test did not run, because a bound that
    was never measured is not a bound of zero.
    """
    with jax.enable_x64(True):
        graph = as_graph(models.shared_ancestor())
        certified = oracle_joint(
            graph,
            (Span("tau", 0.15, 4.5), Span("x", -1.5, 3.5)),
            resolution=AGREEMENT_FLOOR,
            start=201,
            refinements=6,
        )
        not_a_number = oracle_joint(
            graph,
            (Span("tau", -3.0, 7.0), Span("x", -6.0, 6.0)),
            resolution=AGREEMENT_FLOOR,
            start=101,
            refinements=3,
        )
        growing = oracle_joint(
            as_graph(models.diamond_ancestor()),
            (Span("tau", 20.0, 30.0), Span("x", -6.0, 6.0)),
            resolution=AGREEMENT_FLOOR,
            start=101,
            refinements=3,
        )

    assert certified.certified
    assert isinstance(certified.certificate.truncation, float)
    assert math.isfinite(certified.certificate.truncation)

    # DID NOT RUN -- and the two other fields agree, but nobody has to read them.
    assert not_a_number.certificate.truncation is None
    assert not_a_number.certificate.history == ()
    assert not_a_number.certificate.edges == ()
    assert not_a_number.certificate.bound == math.inf

    # RAN, and found the mass leaving the span.
    assert growing.certificate.truncation == math.inf
    assert growing.certificate.edges != ()
    assert not growing.certified
