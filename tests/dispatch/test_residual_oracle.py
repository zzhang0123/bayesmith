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

import inspect
import math
from collections import Counter

import equinox as eqx
import jax
import numpy as np
import pytest

import bayesmith
from bayesmith.artifacts.refusal import Refusal
from bayesmith.dispatch import collapse as collapse_module
from bayesmith.errors import BayesmithError
from bayesmith.graph.reduction import as_graph
from tests.dispatch import residual_oracle
from tests.dispatch.residual_oracle import (
    AGREEMENT_FLOOR,
    Span,
    agreement,
    oracle_collapsed,
    oracle_joint,
    quadrature,
)
from tests.exact import models

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
}


def _sides(name):
    """``(collapsed, joint)`` for one class-(b) fixture, both certified or not.

    The starting grids differ because the two integrals have different
    dimensions and the same wall-clock buys different resolutions; the
    RESOLUTION each is held to is the same declared number, which is the half
    that matters.
    """
    exact, layout = CLASS_B[name]
    graph = as_graph(getattr(models, name)())
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
    # A statement about the SPAN: eight times the points move it by 0.3 per cent.
    assert abs(bounds[0] - bounds[1]) < 0.01 * bounds[0], bounds
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


def test_a_resolution_must_be_declared_and_positive():
    with jax.enable_x64(True), pytest.raises(ValueError, match="positive level"):
        quadrature(lambda values: values["a"], (Span("a", -1.0, 1.0),), resolution=0.0)


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
    counts, _members = _census(classified, parameterised=True)
    assert sum(counts.values()) == 54
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

    with jax.enable_x64(True):
        graph = as_graph(models.diamond_ancestor())
        spans = (Span("tau", -4.0, 8.0), Span("x", -6.0, 6.0))
        oracle_joint(graph, spans, resolution=AGREEMENT_FLOOR, start=101, refinements=3)
        assert called == []
        residual_oracle.collapse_graph = watched
        try:
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
