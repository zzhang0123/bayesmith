"""R5 Task 3 -- the three fixtures R5's gates need, each against a CONSTRUCTED
closed form, and the dimension at which the oracle stops being one.

R5's 8 gate asks for a non-Gaussian or multimodal fixture and this package ships
neither: every residual in ``tests/exact/models.py`` is unimodal and at most two
dimensional, so nothing there can tell a method that finds THE posterior from
one that finds A mode of it, and nothing there costs a product grid anything.
The three built here are graded twice over -- against a closed form that no
sampler and no quadrature contributed to, and against the Task 2 oracle, whose
own certificate has to hold before its number is used.

**The closed forms are asserted to DISCRIMINATE, not only to agree.** A closed
form that happened to be right by cancelling two wrong terms would agree with
the oracle and grade nothing, so each is re-run with a named part removed and
the distance measured in units of the same run's own band.
"""

from __future__ import annotations

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bayesmith
from bayesmith.artifacts.refusal import CAPABILITY_UNAVAILABLE_R1, Refusal
from bayesmith.artifacts.tasks import EvidenceTask, new_task_meta
from bayesmith.graph.reduction import as_graph
from tests.dispatch.residual_oracle import (
    AGREEMENT_FLOOR,
    MAX_POINTS,
    Span,
    oracle_joint,
)
from tests.dispatch.test_task_protocol import model_ref
from tests.exact import residual_models as rm

#: The spans each fixture is graded over. Declared beside the fixture rather
#: than inside the oracle, because a span is a DOMAIN and R5 plan 0.15 requires
#: every pinned number to carry the one it was integrated over.
MIXTURE_SPANS = (Span("w", -8.0, 9.5), Span("b", -7.5, 9.0))

#: Wide enough that the ``z**-4`` tail past it is 4.4e-10 of the mass, which is
#: what it takes for a Cauchy: at ``|z| <= 100`` the same integrand is 4.4e-07
#: short. The tail is the fixture's whole point.
CAUCHY_SPAN = (Span("z", -1000.0, 1000.0),)


def _quartet_spans(parts):
    """Six and a half posterior standard deviations either side of the mean.

    Placed from the closed-form posterior rather than guessed, so the span is a
    property of the model and not of whoever wrote the test. At 6.5 sd the mass
    outside is ``8e-11`` per axis for a Gaussian, below the band the comparison
    is held to.
    """
    mean, spread, _precision = rm.gaussian_posterior(
        parts["design"],
        parts["offset"],
        parts["data"],
        parts["prior_mean"],
        parts["prior_std"],
        parts["sigma"],
    )
    return tuple(
        Span(name, float(centre - 6.5 * width), float(centre + 6.5 * width))
        for name, centre, width in zip(parts["latents"], mean, spread, strict=True)
    )


# ------------------------------------------------------------ (i) multimodal


def _mixture_marginal(graph, *, along=801, across=401):
    """The ``w`` marginal of the mixture fixture's posterior, on a grid.

    **Vectorised, and that is not a micro-optimisation.** The first version of
    this helper was a nested Python loop over 1751 x 601 points, which is 1.05
    MILLION separate ``log_joint`` calls -- about nine minutes, in the fast
    layer. It went unnoticed because an assertion above it failed first, so the
    loop had never once run to completion; the cost appeared the moment the
    assertion started passing. A `vmap` over rows does the same grid in well
    under a second.
    """
    down = jnp.linspace(-8.0, 9.5, along)
    across_axis = jnp.linspace(-7.5, 9.0, across)
    row = jax.jit(
        jax.vmap(
            lambda w, b: bayesmith.graph.evaluate.log_joint(graph, {"w": w, "b": b}),
            in_axes=(None, 0),
        )
    )
    values = np.asarray(jnp.stack([row(w, across_axis) for w in down]), dtype=float)
    axis = np.asarray(down, dtype=float)
    marginal = np.trapezoid(
        np.exp(values - values.max()), np.asarray(across_axis, dtype=float), axis=1
    )
    return marginal / np.trapezoid(marginal, axis), axis


def test_the_mixture_fixture_has_two_modes_that_both_carry_mass():
    """Multimodal by measurement, not by construction-and-hope.

    A mixture PRIOR does not guarantee a mixture posterior -- a likelihood sharp
    enough kills all but one component -- so the ``w`` marginal is tabulated and
    the two peaks, the valley between them and the mass either side of it are
    read off it. The split is then checked against the closed form's own
    per-component weights, which is the part that matters: a closed form that
    got the TOTAL right by cancelling two wrong terms would disagree about the
    shape.
    """
    with jax.enable_x64(True):
        graph = as_graph(rm.mixture_prior_residual())
        found = oracle_joint(
            graph, MIXTURE_SPANS, resolution=AGREEMENT_FLOOR, start=201, refinements=4
        )
        assert found.certified, found.describe()
        total, terms = rm.mixture_prior_residual_log_evidence()

        marginal, axis = _mixture_marginal(graph)

    # Two interior maxima and one interior minimum between them: that is what
    # "bimodal" is, and it is checkable without naming where they are.
    rising = np.diff(marginal) > 0
    turns = np.flatnonzero(rising[:-1] != rising[1:]) + 1
    assert len(turns) == 3, axis[turns]
    lower, valley, upper = (int(index) for index in turns)
    assert marginal[valley] * 100 < min(marginal[lower], marginal[upper])
    assert axis[upper] - axis[lower] > 3.0, (axis[lower], axis[upper])

    # And the mass either side of the valley IS the closed form's component
    # weights, which is the check the total alone cannot make.
    below = float(np.trapezoid(marginal[: valley + 1], axis[: valley + 1]))
    weights = np.exp(terms - total)
    assert below == pytest.approx(float(weights[0]), abs=1e-4), (below, weights)
    assert 0.2 < below < 0.8, below


def test_the_mixture_closed_form_agrees_with_the_oracle_and_discriminates():
    """R5 G3's multimodal fixture, graded, and its closed form graded back.

    ``exp(terms - total)`` being the mode weights makes each summand separately
    falsifiable, so three named parts are removed in turn and the damage is
    reported in units of the same run's own band. A closed form that cannot be
    broken is a closed form nothing checked.
    """
    with jax.enable_x64(True):
        graph = as_graph(rm.mixture_prior_residual())
        found = oracle_joint(
            graph, MIXTURE_SPANS, resolution=AGREEMENT_FLOOR, start=201, refinements=4
        )
        assert found.certified, found.describe()
        total, terms = rm.mixture_prior_residual_log_evidence()
        band = found.certificate.bound + AGREEMENT_FLOOR * max(1.0, abs(total))
        assert abs(found.value - total) <= band, (found.describe(), total)

        # (a) flatten the mixture weights, (b) keep only the heavier component.
        flat, _ = rm.mixture_prior_residual_log_evidence(mode_weights=(0.5, 0.5))
        heavy = float(terms[1])
    # Measured: 5.23e-02 nats and 5.65e-01 nats, which is 3.9e+06 and 4.2e+07
    # times the band the agreement above is held to. The assertion asks for six
    # orders because that is the weakest statement that still says "not a near
    # miss"; the measured multiples are recorded so a drift toward the line is
    # visible rather than only a crossing of it.
    assert abs(flat - total) > 1e6 * band, (flat, total, band)
    assert abs(heavy - total) > 1e6 * band, (heavy, total, band)


# ----------------------------------------------------------- (ii) heavy tail


def test_the_cauchy_pair_closed_form_agrees_with_the_oracle():
    """A heavy tail in the LIKELIHOOD, which is the case quadrature finds hard.

    ``overflowing_outside_latent`` also declares a Cauchy, on a latent whose
    contribution to ``mu`` is ``exp(|z| / 50)``: the likelihood then cuts the
    tail off faster than a Gaussian and the integrand is easy. Here the
    integrand itself decays as ``z**-4``, and the closed form is exact because
    Cauchy is stable under convolution.
    """
    with jax.enable_x64(True):
        graph = as_graph(rm.cauchy_residual_pair())
        found = oracle_joint(
            graph, CAUCHY_SPAN, resolution=AGREEMENT_FLOOR, start=4001, refinements=5
        )
        assert found.certified, found.describe()
        exact = rm.cauchy_residual_pair_log_evidence()
        # INSIDE the x64 block, and that is not tidiness. These two calls sat
        # outside it until a second adversarial review noticed: the closed forms
        # in residual_models.py carry no `_require_x64` of their own, so at
        # float32 `cauchy_residual_pair_log_evidence` returns -2.2217936515808105
        # against the analytic -2.2217938383113070 -- an error of 1.87e-07, which
        # is 187 times AGREEMENT_FLOOR, returned silently. The assertion below
        # still passed, because the values it discriminates are orders away; it
        # was latent rather than failing, which is the only reason it survived
        # to be found.
        wrong_values = (
            rm.cauchy_residual_pair_log_evidence(sigma=0.0),
            rm.cauchy_residual_pair_log_evidence(gamma=0.0),
            float(np.log(math.sqrt(1.75**2 + 0.4**2))),
        )
    band = found.certificate.bound + AGREEMENT_FLOOR * max(1.0, abs(exact))
    assert abs(found.value - exact) <= band, (found.describe(), exact, band)

    # Discrimination: the prior's own width, the likelihood's own width, and a
    # Gaussian-style combination of the two are all far outside that band.
    for wrong in wrong_values:
        assert abs(wrong - exact) > 1e6 * band, (wrong, exact, band)


def test_the_edge_bound_is_measured_against_an_exactly_known_polynomial_tail():
    """The oracle's edge bound, graded against a truth on the hardest tail here.

    ``cauchy_tail_mass`` integrates ``gamma sigma / (pi**2 z**4)`` past the span
    in closed form, so this is not one quadrature checking another. The bound
    must COVER the mass -- a geometric model of a ``z**-p`` tail sums to
    ``(p - 1) / p`` of it and would not, which is why
    ``residual_oracle._power_tail`` exists -- and must not be extravagant, or it
    would swallow a real disagreement.
    """
    with jax.enable_x64(True):
        graph = as_graph(rm.cauchy_residual_pair())
        exact = rm.cauchy_residual_pair_log_evidence()
        for span in (100.0, 1000.0):
            found = oracle_joint(
                graph,
                (Span("z", -span, span),),
                resolution=AGREEMENT_FLOOR,
                start=4001,
                refinements=5,
            )
            assert found.certified, found.describe()
            error = abs(found.value - exact)
            truth = rm.cauchy_tail_mass(span=span)
            bound = found.certificate.truncation
            # The closed-form tail predicts the error to three digits...
            assert truth == pytest.approx(error, rel=1e-3), (span, truth, error)
            # ...and the oracle's own bound covers it without doubling it.
            assert bound >= truth, (span, bound, truth)
            assert bound < 1.2 * truth, (span, bound, truth)


# ------------------------------------------------------- (iii) four residual axes


def test_the_quartet_is_discovered_as_one_joint_exact_block():
    """Four undeclared affine axes are discovered jointly despite coupling.

    The coupling is the half that is easy to lose: four INDEPENDENT axes would
    make the evidence a product of four one-dimensional integrals, which a
    product-grid oracle gets right for a reason that has nothing to do with
    being four-dimensional.
    """
    with jax.enable_x64(True):
        graph = as_graph(rm.undeclared_quartet())
        plan = bayesmith.compile(graph)
        parts = rm.undeclared_quartet_parts(graph)
        _mean, spread, precision = rm.gaussian_posterior(
            parts["design"],
            parts["offset"],
            parts["data"],
            parts["prior_mean"],
            parts["prior_std"],
            parts["sigma"],
        )
    assert sorted(plan.exact.latents) == sorted(rm.QUARTET_NAMES)
    assert plan.exact.method == "gcr"
    assert plan.sampled is None
    off_diagonal = np.abs(precision - np.diag(np.diag(precision)))
    assert off_diagonal.max() > np.diag(precision).min(), precision
    covariance = np.linalg.inv(precision)
    correlation = covariance / np.outer(spread, spread)
    assert np.abs(correlation - np.eye(4)).max() > 0.5, correlation


def test_the_quartet_closed_form_agrees_with_the_four_dimensional_oracle():
    """R5 Task 3(iii): a residual of dimension four with a constructed answer."""
    with jax.enable_x64(True):
        graph = as_graph(rm.undeclared_quartet())
        parts = rm.undeclared_quartet_parts(graph)
        exact = rm.gaussian_log_evidence(
            parts["design"],
            parts["offset"],
            parts["data"],
            parts["prior_mean"],
            parts["prior_std"],
            parts["sigma"],
        )
        found = oracle_joint(
            graph,
            _quartet_spans(parts),
            resolution=AGREEMENT_FLOOR,
            start=9,
            refinements=5,
        )
        assert found.certified, found.describe()
        band = found.certificate.bound + AGREEMENT_FLOOR * max(1.0, abs(exact))
        assert abs(found.value - exact) <= band, (found.describe(), exact, band)

        # Discrimination: drop the prior's contribution to the data covariance,
        # which is the term a linear-Gaussian evidence is most often written
        # without, and the answer moves by nats.
        wrong = rm.gaussian_log_evidence(
            parts["design"],
            parts["offset"],
            parts["data"],
            parts["prior_mean"],
            np.zeros_like(parts["prior_std"]),
            parts["sigma"],
        )
    assert abs(wrong - exact) > 1e8 * band, (wrong, exact, band)


# ----------------------------------------------- Task 3.2: where the oracle stops


#: Measured on this tree, macOS/Accelerate, x64, at ``MAX_POINTS = 5e7``. The
#: numbers move with the budget and the machine; the SHAPE does not, and the
#: shape is the finding.
DIMENSION_SWEEP = (1, 2, 3, 4, 5, 6)


def test_the_dimension_at_which_the_oracle_stops_being_one():
    """R5 Task 3.2's stop-rule. **It fires, and it was expected to.**

    The boundary of R5's gradeable domain is the largest latent dimension at
    which a product-grid quadrature can certify inside its declared evaluation
    budget. It is not the residual dimension: ``oracle_joint`` integrates the
    residual block AND the exact block, so a residual of four beside any exact
    block is five axes, and this measurement is what says whether that is
    gradeable at all.

    Measured here, per dimension, at ``MAX_POINTS = 5e7``:

    ==  =========  ===  ============  ======  ==============================
    d   certified    n        n_eval    wall  value - closed form
    ==  =========  ===  ============  ======  ==============================
    1   yes         33            33   0.5 s  -1.2e-10
    2   yes         65         4,225   0.5 s  -1.7e-10
    3   yes         65       274,625   0.7 s  -2.6e-10
    4   yes         65    17,850,625   2.3 s  -3.6e-10
    5   NO          33    39,135,393   4.8 s  -6.1e-10
    6   NO          17    24,137,569   4.4 s  **+6.6e-03**
    ==  =========  ===  ============  ======  ==============================

    **So the answer is four, and the last column is why the certificate is not
    ceremony.** At five the uncertified value is still right to 6e-10 and at six
    it is wrong in the third decimal place -- and nothing about the two numbers
    says which is which. A reader who took the uncertified value because it
    looked reasonable would have been right once and 6.6e-03 nats wrong once.

    What stops it is the grid, not the arithmetic: five axes certify at
    ``n = 65``, which is 1.16e9 points and about 19 GB, and six do not certify
    at any grid this machine can hold. ``docs/probes/probe_35_oracle_dimension.py``
    runs the same sweep with the budget lifted so the two causes stay apart.
    """
    outcome = {}
    with jax.enable_x64(True):
        for dimension in DIMENSION_SWEEP:
            graph = as_graph(rm.undeclared_family(dimension=dimension))
            parts = rm.undeclared_family_parts(graph, dimension=dimension)
            exact = rm.gaussian_log_evidence(
                parts["design"],
                parts["offset"],
                parts["data"],
                parts["prior_mean"],
                parts["prior_std"],
                parts["sigma"],
            )
            started = time.perf_counter()
            found = oracle_joint(
                graph,
                _quartet_spans(parts),
                resolution=AGREEMENT_FLOOR,
                start=9,
                refinements=7,
            )
            outcome[dimension] = {
                "certified": found.certified,
                "n": found.certificate.history[-1][0],
                "error": found.certificate.history[-1][1] - exact,
                "seconds": time.perf_counter() - started,
                "why": found.certificate.refused,
            }

    certified = [d for d in DIMENSION_SWEEP if outcome[d]["certified"]]
    assert certified == [1, 2, 3, 4], outcome
    for dimension in certified:
        assert abs(outcome[dimension]["error"]) < 1e-8, outcome[dimension]
    # The two that abstain do so because of the BUDGET, named in the sentence,
    # and not because the integrand misbehaved.
    for dimension in (5, 6):
        assert "above the" in outcome[dimension]["why"], outcome[dimension]
        assert f"{MAX_POINTS:.3e} budget" in outcome[dimension]["why"]
    # And the abstain at six is necessary rather than cautious: the number it
    # would have published is wrong in the third decimal place, while five's is
    # right to ten. Nothing in the values separates them.
    assert abs(outcome[5]["error"]) < 1e-8, outcome[5]
    assert abs(outcome[6]["error"]) > 1e-4, outcome[6]


# ------------------------------------------------- what the gates say about them


def test_where_each_new_fixture_sits_in_the_structural_taxonomy():
    """The before-state, recorded so Task 7's widening has something to move.

    All three are admitted by no evidence gate today. Two are all-residual and
    one has an exact block beside its residual -- which makes the mixture
    fixture the first MULTIMODAL member of R5's headline class -- and each is
    refused under the premise that answers first, which is not the same as the
    premise that describes it.

    **TASK 7 TURNED THIS TEST RED, WHICH IS WHAT IT WAS FOR**, and the column
    below is the after-state. Task 7 widened ``_evidence_structure_refusal``
    and restated propriety, and all three now pass every premise about the
    MODEL: their priors are proper, and their conditionals are densities across
    the range the residual integral covers. What stops all three is the
    capability refusal -- the residual sampler is an optional extra and no
    adapter exists yet -- which is exactly where Task 7.4 requires the widened
    classes to route while the bake-off is still open, so that they are never
    admitted with nothing behind them.

    None of the three is stopped by anything structural any more, and the
    structural columns are unchanged, which is the part worth keeping: the
    widening moved the PREMISE and not the taxonomy.
    """
    reference = model_ref()
    seen = {}
    with jax.enable_x64(True):
        for name, builder in (
            ("mixture_prior_residual", rm.mixture_prior_residual),
            ("cauchy_residual_pair", rm.cauchy_residual_pair),
            ("undeclared_quartet", rm.undeclared_quartet),
        ):
            graph = as_graph(builder())
            plan = bayesmith.compile(graph)
            out = bayesmith.compile_task(
                graph,
                EvidenceTask(meta=new_task_meta(label="wave-b")),
                model_ref=reference,
            )
            seen[name] = (
                tuple(plan.exact.latents) if plan.exact is not None else (),
                plan.exact.method if plan.exact is not None else None,
                tuple(plan.sampled.latents) if plan.sampled is not None else (),
                out.failed_premise if isinstance(out, Refusal) else "(ADMITTED)",
            )
    # The failure message names its own remedy. A red test that says only
    # "expected to change" costs its reader the whole investigation; one that
    # says what to change it to costs them a diff.
    after_task_6 = (
        "Task 6 installs the residual adapter, at which point "
        "`residual_backend()` stops returning None and these three stop being "
        "refused for a capability the release now has. Replace the premise "
        "column with '(ADMITTED)' and keep the assertion -- it is the census "
        "of where these fixtures sit, and each wave moves one column of it."
    )
    assert seen["mixture_prior_residual"] == (
        ("b",),
        "gcr",
        ("w",),
        CAPABILITY_UNAVAILABLE_R1,
    ), (seen["mixture_prior_residual"], after_task_6)
    assert seen["cauchy_residual_pair"] == (
        (),
        None,
        ("z",),
        CAPABILITY_UNAVAILABLE_R1,
    ), (seen["cauchy_residual_pair"], after_task_6)
    assert seen["undeclared_quartet"][:3] == (
        ("alpha", "beta", "delta", "gamma"),
        "gcr",
        (),
    )
    assert seen["undeclared_quartet"][3] == "(ADMITTED)", (
        seen["undeclared_quartet"],
        after_task_6,
    )


#: The slope sweep ``mixture_prior_residual``'s docstring states, re-measured on
#: this tree. **It is here because two of the numbers it replaced were false and
#: nothing could have noticed** -- nothing in the suite read the sweep, so a row
#: transcribed from a design agent's report without being re-run sat in the
#: docstring for two commits. The 0.15 row claimed a 0.625/0.375 split; the true
#: split there is 0.5679/0.4321, and 0.625/0.375 is reached at slope 0.110894.
MIXTURE_SLOPE_SWEEP = (
    # slope, lower weight, valley/lower, valley/upper
    (0.05, 0.7066, 1144.6, 333.7),
    (0.15, 0.5679, 880.4, 470.3),
    (0.24, 0.4315, 697.8, 645.4),
    (0.36, 0.2663, 514.6, 995.6),
)


@pytest.mark.parametrize(("slope", "weight", "lower", "upper"), MIXTURE_SLOPE_SWEEP)
def test_the_slope_sweep_is_measured_rather_than_asserted(slope, weight, lower, upper):
    """``slope`` buys BALANCE, not the second mode -- and the sweep is now read.

    The claim the fixture rests on is that tuning ``slope`` moves how evenly the
    two modes are weighted while leaving both modes standing. That needs the
    whole sweep, not the default: at the default alone, a fixture that had only
    one mode for every other slope would look identical.

    Both valley ratios are checked, because "the valley ratio" is two numbers and
    only the smaller one bounds the multimodality. The tolerances are loose on
    purpose -- these are peak locations read off a discrete grid, and pinning
    them tighter would pin the grid rather than the model.
    """
    with jax.enable_x64(True):
        graph = as_graph(rm.mixture_prior_residual(slope=slope))
        marginal, axis = _mixture_marginal(graph)
        total, terms = rm.mixture_prior_residual_log_evidence(slope=slope)
    rising = np.diff(marginal) > 0
    turns = np.flatnonzero(rising[:-1] != rising[1:]) + 1
    assert len(turns) == 3, (slope, axis[turns])
    low, valley, high = (int(index) for index in turns)

    assert float(np.exp(terms - total)[0]) == pytest.approx(weight, abs=5e-5)
    assert marginal[low] / marginal[valley] == pytest.approx(lower, rel=1e-3)
    assert marginal[high] / marginal[valley] == pytest.approx(upper, rel=1e-3)
    # The second mode survives every row: the SMALLER ratio is what says so.
    assert min(lower, upper) > 300.0, (slope, lower, upper)


def test_the_quartet_parts_are_read_off_the_graph_and_not_recomputed():
    """The one crossing ``undeclared_family_parts``'s protection exists for.

    Its docstring says the parts are read off the graph "so the closed form
    cannot describe a different data vector from the one the graph carries".
    Nothing tested that: all nine graph builds in this module sit INSIDE
    ``jax.enable_x64``, so the graph and any recomputation agreed, and replacing
    the read with the identical recomputation left 47 passed, exit 0.

    Here the graph is built OUTSIDE the block and read INSIDE it -- the crossing
    the protection is about, because ``jax.random.normal`` on one key returns
    different draws at the two precisions and ``t``, ``t**2``, ``sin(t)`` and
    ``cos(t)`` are evaluated at the graph's own. Measured: the data read off the
    float32 graph is ``[2.387, 1.173, 1.406]`` and the same expressions
    recomputed at x64 give ``[2.024, 1.816, 2.231]``; the closed forms are
    -8.172355 and -5.987312, **2.185 nats apart**. The oracle, which sees only
    the graph, agrees with the first to 2.1e-08 and with the second to nothing.
    """
    outside = as_graph(rm.undeclared_quartet())
    assert np.asarray(outside.node("T").value).dtype == np.float32
    with jax.enable_x64(True):
        parts = rm.undeclared_family_parts(outside, dimension=4)
        exact = rm.gaussian_log_evidence(
            parts["design"],
            parts["offset"],
            parts["data"],
            parts["prior_mean"],
            parts["prior_std"],
            parts["sigma"],
        )
        found = oracle_joint(
            outside,
            _quartet_spans(parts),
            resolution=AGREEMENT_FLOOR,
            start=9,
            refinements=5,
        )
        assert found.certified, found.describe()
        # The closed form describes the graph in hand, at the precision in hand.
        band = found.certificate.bound + AGREEMENT_FLOOR * max(1.0, abs(exact))
        assert abs(found.value - exact) <= max(band, 1e-6), (found.describe(), exact)
        # And a recomputation would describe a different model entirely.
        rebuilt = rm.undeclared_family_parts(
            as_graph(rm.undeclared_quartet()), dimension=4
        )
    assert abs(rebuilt["data"][0] - parts["data"][0]) > 0.1


def test_the_shifted_block_prior_fixture_has_a_closed_form_of_its_own():
    """Both fixtures the first review's repairs added were graded only by
    collapsed-versus-uncollapsed agreement, which grades the ELIMINATION and
    cannot grade the FIXTURE: change the data vector and both sides move
    together, still agreeing. A second review pointed that out by changing one
    and watching 47 tests pass.

    ``shifted_block_prior`` is jointly linear-Gaussian -- ``tau ~ N(m, s)`` and
    ``x | tau ~ N(tau, w)`` make ``x`` marginally ``N(m, sqrt(s**2 + w**2))`` --
    so it has an exact closed form and needs no quadrature at all. Two
    independent derivations agree to fifteen digits: this one through the
    marginal collapse, and the reviewer's through Sherman-Morrison on the
    rank-one covariance with the remaining integral at mpmath 40 dps.
    """
    with jax.enable_x64(True):
        exact = rm.shifted_block_prior_log_evidence()
        found = oracle_joint(
            as_graph(rm.shifted_block_prior()),
            (Span("tau", 0.4, 4.0), Span("x", -1.0, 5.5)),
            resolution=AGREEMENT_FLOOR,
            start=201,
            refinements=6,
        )
        assert found.certified, found.describe()
        band = found.certificate.bound + AGREEMENT_FLOOR * max(1.0, abs(exact))
        assert abs(found.value - exact) <= band, (found.describe(), exact, band)
        # Discrimination: forget that the two widths add in quadrature -- the
        # single most likely way to write this closed form wrong.
        wrong = rm.shifted_block_prior_log_evidence(tau_scale=0.45, block_width=0.0)
    assert abs(wrong - exact) > 1e6 * band, (wrong, exact, band)


def test_the_closed_forms_refuse_to_describe_a_graph_they_are_not_looking_at():
    """MUT-X: the closed forms had no precision guard, and one call had drifted.

    ``cauchy_residual_pair_log_evidence`` is now written in plain float64 and is
    precision-blind -- it returns the same 16 digits inside and outside
    ``enable_x64``, matching the 50-digit analytic value. The other two cannot
    be: their grids are built with ``jnp``, so at float32 the GRAPH carries a
    different data vector and a float64 reference would be exact about a model
    nobody built. Those refuse instead.
    """
    outside = rm.cauchy_residual_pair_log_evidence()
    with jax.enable_x64(True):
        inside = rm.cauchy_residual_pair_log_evidence()
    assert outside == inside
    assert outside == pytest.approx(-2.221793838311306957321, abs=1e-15)

    for closed_form in (
        rm.mixture_prior_residual_log_evidence,
        rm.shifted_block_prior_log_evidence,
    ):
        with pytest.raises(RuntimeError, match="double precision"):
            closed_form()
