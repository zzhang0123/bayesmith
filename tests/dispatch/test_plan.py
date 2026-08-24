"""The printed plan: what it says, and whether a reader can check it.

Every assertion here is about text a user reads, so the failure mode this file
exists to prevent is a plan that is *reassuring* rather than *checkable* --
"NUTS" with no reason, a ``tol`` with no ``kappa`` beside it, or a ``kappa``
pinned at the prior centre while the latent that moves it wanders every sweep.

**Two traps, both hit while writing this file and both worth recording.**

The first is that ``bilinear_pair``'s members are printed in the block header
whatever the reason says, so the plan's own ``"gain" in text`` reads as a check
on the reason and is not one: delete the reason from the output entirely and
that assertion stays green. It is checked here against the classifier's own
``reason`` string instead, and against the body of the plan with the header
line removed.

The second is that the reason itself contains ``rtol=1.0e-08``, so a bare
``re.search("tol=...")`` can bind to it. The tolerance patterns below require a
word boundary, which ``rtol`` does not offer, so the check does not silently
depend on which line happens to come first.
"""

import itertools
import math
import re

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from bayesmith import compile as compile_graph
from bayesmith import const, det, observe, sample, trace
from bayesmith.dispatch import plan as plan_module
from bayesmith.dispatch.classify import block_at, partition
from bayesmith.dispatch.plan import (
    CONVERGENCE_TARGET,
    KAPPA_PROBE_SIGMAS,
    InferencePlan,
    kappa_interval,
)
from bayesmith.errors import StructureError
from bayesmith.exact.block import domain_centre, unchecked_operator
from bayesmith.exact.gls import sigma_from_graph
from bayesmith.exact.solve import condition_bound
from tests.dispatch.test_classify import (
    three_member_constant_sigma,
    three_member_moving_sigma,
)
from tests.dispatch.test_dispatch_entry import detached_ancestor
from tests.exact.models import (
    bilinear_pair,
    collinear_pair,
    diamond_ancestor,
    improper_outside_prior,
    indirect_ancestor,
    mixed_radiometer,
    orphaned_child_latent,
    overflowing_outside_latent,
    plated_and_scalar_latents,
    plated_latent,
    plated_radiometer,
    prior_held_direction,
    quadratic_claim,
    radiometer,
    shared_ancestor,
    straight_line,
    three_latent_chain,
    two_linear_latents,
    two_observations_reverse_sorted_names,
)

KAPPA_POINT = re.compile(r"kappa=([0-9.eE+-]+)")
KAPPA_RANGE = re.compile(r"kappa in \[([0-9.eE+-]+), ([0-9.eE+-]+)\]")
TOL = re.compile(r"\btol=([0-9.eE+-]+)")


def flat(text: str) -> str:
    """``text`` with every run of whitespace collapsed.

    The plan wraps its prose to the terminal, so a sentence it prints is not a
    substring of the plan. Collapsed on both sides it is.
    """
    return " ".join(text.split())


def execution_line(plan) -> str:
    """The plan's single ``execution:`` line."""
    rows = [row for row in str(plan).split("\n") if row.startswith("execution: ")]
    assert len(rows) == 1, rows
    return rows[0]


def measured_kappa(graph, names) -> float:
    """``condition_bound`` at the anchor, spelled independently of ``plan.py``.

    Deliberately re-derived through the public primitives rather than by
    calling ``plan._kappa_at``: the claim under test is that the number the
    plan prints is the number a caller measures for themselves, and calling
    the plan's own routine could not fail that claim however wrong it was.
    """
    at = block_at(graph, names)
    operator = unchecked_operator(graph, names, at)
    sigma = sigma_from_graph(graph, at)(domain_centre(operator))
    return float(condition_bound(operator, noise_std=sigma))


def _ancestor_width(width_of_tau, *, tau_loc, tau_scale, n=6, sigma=0.5, seed=8):
    """``indirect_ancestor``'s shape with the width function handed in. LOCAL.

    ``tau -> width -> x``'s prior, ``x -> mu -> d``, so the ancestor rule
    ejects ``tau`` and the exact block is ``{x}`` -- the partition
    ``indirect_ancestor`` already pins. What varies is the one function this
    module's kappa sweep evaluates at four displacements of ``tau``, which is
    the only thing the two fixtures below are about. Both are LOCAL to this
    file for the reason ``detached_ancestor``'s docstring gives: each exists
    for one dimension of one test, and neither is a shape any other module
    needs.
    """
    grid = jnp.linspace(1.0, 2.0, n)
    data = 1.0 * grid + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", grid)
        tau = sample("tau", lambda: dist.Normal(tau_loc, tau_scale))
        width = det("width", width_of_tau, tau)
        x = sample("x", lambda w: dist.Normal(0.0, w), width)
        mu = det("mu", lambda x_, g_: x_ * g_, x, xs, linear_in=("x",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def overflowing_probe_width(*, rate=15.0, **kwargs):
    """``x``'s prior width is ``exp(rate*tau)``, ``tau ~ N(0, 1)``. LOCAL.

    Legal, finite and strictly positive at every probe point -- a log-normal
    prior width is an ordinary way to write "this scale spans decades" -- but
    its SQUARE is not: at ``tau = +3`` the width is ``exp(45) = 3.5e+19`` and
    ``width**2`` overflows float32's 3.4e+38, while at ``tau = -3`` it is
    ``exp(-45) = 2.9e-20`` and the reciprocal variance overflows the same way.
    ``condition_bound`` is ``lambda_max * max(prior_variance)``, so it comes
    back ``inf`` at the upper probe and ``nan`` at the lower one, with
    ``check_gaussian`` raising nothing at either -- the scale itself is
    perfectly representable.

    That is the ONE route this package has to a non-finite kappa at a probe
    point, and it is what makes ``kappa_interval``'s ``refused`` list
    reachable at all.
    """
    return _ancestor_width(
        lambda t: jnp.exp(rate * t), tau_loc=0.0, tau_scale=1.0, **kwargs
    )


def unfloored_probe_width(**kwargs):
    """``x``'s prior width IS ``tau ~ N(1.0, 0.5)``, so it goes negative. LOCAL.

    A declaration that is true where the model was classified and false a
    fifth of the way into its own prior: at the centre the width is 1.0 and
    ``partition`` accepts it, at the ``-3``-sigma probe it is ``-0.5`` and
    ``check_gaussian`` raises ``StructureError`` -- "a zero or negative sigma
    is an infinite or negative weight rather than a tight constraint. Add a
    floor to the expression that produces it."

    That is a fault in the model and not a verdict about it, which is what
    ``test_a_broken_probe_point_is_raised_rather_than_noted`` is about.
    """
    return _ancestor_width(lambda t: t, tau_loc=1.0, tau_scale=0.5, **kwargs)


#: Graphs whose exact block spans every latent, so nothing outside it can move
#: kappa and the plan prints a single number. Chosen to reach three block
#: sizes (1, 2, 3), both sides of the sigma branch, a plate, and a graph with
#: two observed nodes.
POINT_KAPPA = [
    pytest.param(straight_line, 1, "gcr", id="one_member"),
    pytest.param(two_linear_latents, 2, "gcr", id="two_members"),
    pytest.param(three_member_constant_sigma, 3, "gcr", id="three_members"),
    pytest.param(radiometer, 1, "gcr+snis", id="snis_one_member"),
    pytest.param(three_member_moving_sigma, 3, "gcr+snis", id="snis_three_members"),
    pytest.param(plated_latent, 1, "gcr", id="plated"),
    pytest.param(plated_and_scalar_latents, 2, "gcr", id="plate_plus_scalar"),
    pytest.param(collinear_pair, 2, "gcr", id="collinear"),
    pytest.param(two_observations_reverse_sorted_names, 1, "gcr", id="two_observed"),
]

#: Graphs with a latent OUTSIDE the exact block, so kappa is a function of
#: where that latent sits. One and two outside latents, and both the
#: ``gcr`` and the ``gcr+mh`` arm.
INTERVAL_KAPPA = [
    pytest.param(indirect_ancestor, 1, id="indirect"),
    pytest.param(diamond_ancestor, 1, id="diamond"),
    pytest.param(shared_ancestor, 1, id="shared"),
    pytest.param(mixed_radiometer, 1, id="mixed_rad"),
    pytest.param(three_latent_chain, 2, id="chain3"),
]


@pytest.mark.parametrize("build, members, method", POINT_KAPPA)
def test_the_printed_plan_carries_kappa_and_the_tol_derived_from_it(
    build, members, method
):
    """Both numbers, or the discipline they encode cannot be checked.

    Section 4.2's rule is that turning the convergence guard off inside a
    sweep REQUIRES tightening `tol` in the same breath -- rheplicant names
    "leave tol at its default and the guard off" as the combination that
    returned a silently over-confident posterior. The plan is where a reader
    checks that the pair was actually chosen together, so printing one
    without the other makes the rule unverifiable.

    `rel=1e-6` is a round-trip tolerance, not a bug-catching one: it is what
    forces the plan to print enough digits of both numbers to be checked at
    all. Measured, the printed pair round-trips to better than 1e-8 -- the
    plan prints eight significant digits precisely so this assertion can be
    this tight. The mutations it is here to kill are far cruder (multiplying
    by kappa instead of dividing moves `tol` by kappa**2).
    """
    graph = build()
    plan = compile_graph(graph)
    text = str(plan)
    assert plan.exact is not None
    assert len(plan.exact.latents) == members
    assert plan.exact.method == method
    assert "kappa" in text
    assert "tol" in text
    kappa = float(KAPPA_POINT.search(text).group(1))
    tol = float(TOL.search(text).group(1))
    assert tol == pytest.approx(CONVERGENCE_TARGET / kappa, rel=1e-6, abs=0.0)


@pytest.mark.parametrize("build, members, method", POINT_KAPPA)
def test_the_printed_kappa_is_the_bound_a_caller_measures_for_themselves(
    build, members, method
):
    """The plan's number must be `condition_bound`, not a second estimate of it.

    A block spanning every latent has no outside latent to sweep, so the
    interval collapses to the anchor and the printed number IS
    `condition_bound` at the prior centre -- which makes this the one
    configuration where the two spellings can be compared exactly. `rel=1e-6`
    against float32 arithmetic that is bit-identical on both sides; the
    tolerance is for the eight-digit print, not for the computation.
    """
    graph = build()
    text = str(compile_graph(graph))
    printed = float(KAPPA_POINT.search(text).group(1))
    assert printed == pytest.approx(
        measured_kappa(graph, partition(graph).exact), rel=1e-6
    )


def test_a_kappa_that_moves_with_an_outside_latent_is_printed_as_an_interval():
    """A kappa pinned at the prior mean is too SMALL exactly where it matters.

    Section 3.2's ancestor rule creates this shape on purpose: `x`'s prior
    width is a function of `tau`, and `tau` moves every sweep. Re-measured for
    this task against `condition_bound` directly, and matching the number the
    plan for it recorded to three figures:

        tau  = 0.0   1.0    2.0 (prior mean)   4.0     6.0
        kappa= 1.568 69.728 251.488            955.808 2114.53

    so pinning at the prior mean understates it 8.4x at tau=6 -- and the
    error is in the dangerous direction: `tol` comes out too LOOSE, CG stops
    early, the posterior comes back too narrow, and the in-sweep guard is off
    so nothing notices. That is verbatim the rheplicant combination section
    4.2 quotes ("leave tol at its default and the guard off").

    `tol` is therefore derived from the interval's UPPER end, and the
    interval is what gets printed. Measured at this fixture's defaults the
    interval is [21.4, 737], a ratio of 34 against the 3 asserted here.
    """
    text = str(compile_graph(indirect_ancestor()))
    match = KAPPA_RANGE.search(text)
    assert match, f"expected an interval, got:\n{text}"
    lo, hi = (float(v) for v in match.groups())
    assert hi > 3 * lo, "the interval collapsed to a point; the sweep is unguarded"
    tol = float(TOL.search(text).group(1))
    assert tol == pytest.approx(CONVERGENCE_TARGET / hi, rel=1e-6)


@pytest.mark.parametrize("build, outside", INTERVAL_KAPPA)
@pytest.mark.parametrize("key_seed", [0, 7])
def test_tol_comes_from_the_upper_end_of_the_interval_on_every_mixed_graph(
    build, outside, key_seed
):
    """The `lo` spelling is the dangerous one, so it is refused by name.

    Two assertions, not one. `tol == target/hi` alone would pass a plan that
    printed the interval backwards; `tol != target/lo` alone would pass one
    that used some third number. The second is written as a *separation* --
    the two candidates differ by the interval's own width -- and the width is
    asserted first, so this cannot silently degrade into a comparison of two
    numbers that happen to be equal.

    Measured widths at key 0: indirect 34x, diamond 32x, shared 46x, mixed
    radiometer 8.0x, three-latent chain 468x. Invariant in the power
    iteration's key across {0, 1, 7} -- the starting vector moves lambda_max
    by less than the print's own last digit here.
    """
    graph = build()
    assert len(partition(graph).nuts) == outside
    text = str(compile_graph(graph, key=jax.random.key(key_seed)))
    match = KAPPA_RANGE.search(text)
    assert match, f"expected an interval, got:\n{text}"
    lo, hi = (float(v) for v in match.groups())
    assert hi > 4 * lo, "the interval is too narrow to tell the two ends apart"
    tol = float(TOL.search(text).group(1))
    assert tol == pytest.approx(CONVERGENCE_TARGET / hi, rel=1e-6, abs=0.0)
    assert tol != pytest.approx(CONVERGENCE_TARGET / lo, rel=0.5, abs=0.0)


@pytest.mark.parametrize(
    "n, sigma, seed",
    list(itertools.product((3, 6, 40), (0.05, 5.0), (0, 99))),
)
def test_the_interval_survives_the_dimensions_it_should_not_depend_on(n, sigma, seed):
    """A region, not a point: the interval is not an artefact of one fixture size.

    Swept over the three dimensions that must not decide whether kappa moves
    -- how many data points there are, how noisy they are, and which noise
    draw. All twelve cells print an interval, and the ratio hi/lo runs from
    4.31 to 34.4 over the grid (worst cell n=3, sigma=5.0: three very noisy
    points barely constrain `x` at all, so kappa is close to the prior's own
    conditioning everywhere and the interval genuinely narrows). 1.5 is
    asserted rather than the 3 the default cell clears by 11x, because 3
    would be within 1.4x of the measured worst cell and that is a constant
    sitting too close to the data.
    """
    text = str(compile_graph(indirect_ancestor(n=n, sigma=sigma, seed=seed)))
    match = KAPPA_RANGE.search(text)
    assert match, f"expected an interval, got:\n{text}"
    lo, hi = (float(v) for v in match.groups())
    assert hi > 1.5 * lo


@pytest.mark.parametrize(
    "build", [indirect_ancestor, diamond_ancestor, shared_ancestor, mixed_radiometer]
)
def test_the_kappa_sweep_reaches_both_sides_of_the_anchor(build):
    """A one-sided probe would leave the anchor at an end of its own interval.

    The point the block was CLASSIFIED at is the prior centre, so a sweep that
    only went up (or only down) would report an interval with the centre on
    its boundary -- and `tol` would then be derived from a bound that no
    displacement of the outside latent was ever checked against on one side.

    `three_latent_chain` is deliberately NOT in this list and is the measured
    reason the property is not universal: `y`'s prior width is `|x| + 0.35`
    with `x` centred at 0, so every displacement of `x` in either direction
    RAISES kappa and the centre really is the minimum (9.59, against a
    maximum of 4489). Two-sidedness is a property of the probe, not a
    guarantee about the function it probes.
    """
    graph = build()
    lo, hi, _ = kappa_interval(graph, partition(graph).exact)
    centre = measured_kappa(graph, partition(graph).exact)
    assert lo < centre < hi


def test_a_refused_block_prints_why_not_just_that_it_was_refused():
    """ "NUTS" with no reason is indistinguishable from "no exact structure".

    The member names alone cannot carry this: the block header prints
    `{gain, t_ant}` whatever the reason says, so `"gain" in text` is green
    even with the reason deleted. Checked instead against the classifier's own
    verdict string, and separately against the plan with its header lines
    stripped -- the two together red on any plan that drops the reason.
    """
    graph = bilinear_pair()
    text = str(compile_graph(graph))
    reason = partition(graph).reason
    assert "NUTS" in text
    assert "gain" in reason and "t_ant" in reason
    assert flat(reason) in flat(text)
    body = flat("\n".join(l for l in text.split("\n") if not l.startswith("block ")))
    assert "gain" in body and "t_ant" in body


@pytest.mark.parametrize(
    "build", [bilinear_pair, quadratic_claim, orphaned_child_latent]
)
def test_an_all_nuts_plan_shows_the_classifiers_verdict_verbatim(build):
    """Three different refusals, three different reasons, none of them dropped.

    A joint-affinity failure, a single false `linear_in`, and a latent ejected
    because another latent's density depends on it. Only the classifier knows
    which; the plan's job is to not lose it on the way to the terminal.
    """
    graph = build()
    text = str(compile_graph(graph))
    assert flat(partition(graph).reason) in flat(text)


@pytest.mark.parametrize(
    "build, execution, absent",
    [
        pytest.param(indirect_ancestor, "HMCGibbs", "no chain", id="mixed"),
        pytest.param(mixed_radiometer, "HMCGibbs", "no chain", id="mixed_mh"),
        pytest.param(two_linear_latents, "no chain", "HMCGibbs", id="fully_exact"),
        pytest.param(radiometer, "no chain", "HMCGibbs", id="fully_exact_snis"),
        pytest.param(bilinear_pair, "NUTS", "HMCGibbs", id="fully_sampled"),
    ],
)
def test_the_plan_names_the_execution_it_will_use(build, execution, absent):
    """A mixed graph runs HMCGibbs; a fully exact one runs no chain at all.

    The distinction is the product: 'iid draws, no chain' and 'HMCGibbs over
    these sites' are different enough that a user must not have to infer
    which one they got. Both directions are asserted on every row -- a plan
    that printed HMCGibbs unconditionally satisfies the positive half on
    three of the five rows.
    """
    rows = [
        row
        for row in str(compile_graph(build())).split("\n")
        if row.startswith("execution: ")
    ]
    assert len(rows) == 1
    assert execution in rows[0]
    assert absent not in rows[0]


def test_the_gibbs_sites_are_the_exact_block_and_not_the_sampled_one():
    """Naming the wrong half would run NUTS on the latents the solve handles.

    `HMCGibbs(gibbs_sites=...)` names the sites the Gibbs update owns -- here
    the exact block -- and NumPyro gives HMC everything else. Printing the
    NUTS latents there instead reads as a plan for the mirror image of the
    partition that was chosen.
    """
    line = execution_line(compile_graph(indirect_ancestor()))
    assert "gibbs_sites=['x']" in line
    assert "tau" not in line.split("gibbs_sites=")[1].split("]")[0]


@pytest.mark.parametrize(
    "build, rebuild",
    [
        pytest.param(indirect_ancestor, True, id="rebuild"),
        pytest.param(three_latent_chain, True, id="rebuild_chain"),
        pytest.param(detached_ancestor, False, id="hoisted"),
    ],
)
def test_the_execution_line_says_whether_noise_std_can_be_hoisted(build, rebuild):
    """`False` is the verdict that authorises hoisting, so it is the one to print.

    Recomputing sigma every sweep costs work; hoisting it when an outside
    latent moves it costs correctness. The plan states which one it chose so
    the choice is auditable rather than buried in the executor.

    **The `False` row is the whole test.** Every mixed fixture in
    `tests/exact/models.py` reports `sigma_needs_rebuild=True` -- measured
    over all seven -- so with only `True` rows the verdict is a constant and
    `("rebuilt every sweep" in line) is rebuild` cannot fail: the string
    `"noise_std hoisted out of the sweep"` (`plan.py`'s other arm) then
    appears in no assertion anywhere in the repository. Measured directly:
    with both arms of that ternary made to return `"noise_std rebuilt every
    sweep"`, the entire suite stays green without this row.

    `detached_ancestor` is imported from `test_dispatch_entry`, which is where
    it was written and where the other half of the `sigma_rebuild` wiring is
    checked, rather than copied: two spellings of a fixture whose one job is
    to be the `False` arm is two things to keep in step.
    """
    plan = compile_graph(build())
    assert plan.sigma_needs_rebuild is rebuild
    line = execution_line(plan)
    assert ("rebuilt every sweep" in line) is rebuild
    assert ("hoisted out of the sweep" in line) is not rebuild


#: ``kappa_upper(kappa) * eps`` at float32, measured over every zero-argument
#: fixture in ``tests/exact/models.py`` plus the parametrised cells
#: ``test_acceptance`` uses. Ten of the 36 with an exact block sit at or above
#: ``CONVERGENCE_TARGET``; the six rows below bracket the threshold from both
#: sides and span its whole range. The nearest pair straddles it by 1.5x,
#: which is the resolution this line is asserted at.
PRECISION_ROWS = [
    pytest.param(radiometer, {}, True, id="radiometer"),  # 1.267e+03
    pytest.param(prior_held_direction, {}, True, id="prior_held"),  # 1.203e+01
    pytest.param(plated_radiometer, {}, True, id="plated_rad"),  # 2.682e-01
    pytest.param(three_member_constant_sigma, {}, True, id="three_member"),  # 1.39e-3
    pytest.param(two_linear_latents, {}, False, id="two_linear"),  # 6.905e-04
    pytest.param(plated_latent, {}, False, id="plated"),  # 1.796e-06
]


@pytest.mark.parametrize("build, kwargs, unattainable", PRECISION_ROWS)
def test_a_tol_under_the_working_precision_says_so_rather_than_promising_a_guard(
    build, kwargs, unattainable
):
    """`tol = target / kappa` can land where the arithmetic cannot follow it.

    `CONVERGENCE_TARGET`'s own docstring already declines to CLIP such a
    `tol`, on the grounds that "it is a true statement that this target is
    unreachable at this precision" -- but the plan printed it beside
    `guard reachable, off by default`, and that pair reads as a promise. It is
    not one: the error CG delivers is `kappa * residual`, and when CG cannot
    get the residual below `tol` the delivered error is whatever the
    arithmetic allows times kappa, not `CONVERGENCE_TARGET`.

    Measured on a 200-element Gaussian-smoothing deconvolution -- `z ~ N(0,1)`
    of length 200, `mu = G z` with a width-6 row-normalised Gaussian kernel,
    `d ~ N(mu, 3e-4)` -- built and run at each dtype in turn:

        float32  kappa=1.089e+07 tol=9.19e-11  kappa*eps=1.30   UNATTAINABLE
                 CG stalls at residual 6.18e-07; delivered relative error of
                 the mean 1.10, i.e. 1100x the 1e-3 target, and the posterior
                 mean sits 1.05 posterior sd from the dense oracle.
        float64  kappa=1.084e+07 tol=9.22e-11  kappa*eps=2.41e-09  reachable
                 CG reaches residual 9.01e-11; delivered error 2.23e-04,
                 inside the target, mean 0.10 sd from the oracle.

    Same model, same printed `tol`, and the only thing separating a delivered
    1e-3 from a delivered 1.1 is the dtype -- which is what the line now says.

    The test is the WHOLE predicate `kappa_upper * eps >= CONVERGENCE_TARGET`
    and not one arm of it: `unattainable` rows must print the clause, and
    attainable rows must print the old one and NOT the clause. `two_linear_
    latents` at 6.905e-04 is the closest attainable row and `three_member_
    constant_sigma` at 1.392e-03 the closest unattainable one, so the pair
    straddles the threshold by 1.5x and neither arm can be satisfied by a
    constant.
    """
    plan = compile_graph(build(**kwargs))
    body = flat(str(plan))
    assert ("tol UNATTAINABLE at this dtype" in body) is unattainable
    assert ("guard reachable, off by default" in body) is not unattainable
    assert plan.exact.tol_attainable is not unattainable


def test_whether_the_tol_is_attainable_is_a_verdict_about_the_dtype():
    """The same graph, both precisions: the line has to move, and it is the
    only thing that does.

    `radiometer`'s condition bound is 1.063e+10, so `kappa * eps` is 1.27e+03
    at float32 and 2.36e-06 at float64 -- six decades either side of the 1e-3
    target. `tol` itself is 9.4e-14 at both, which is exactly the point: a
    reader comparing two plans cannot tell the two situations apart from
    `kappa` and `tol` alone, and those are the two numbers section 4.2 asks
    them to read together.

    The graph is rebuilt INSIDE the `enable_x64` block, not merely recompiled
    there: `const` and `observe` capture their arrays at trace time, so a
    graph traced at float32 stays float32 however it is later compiled, and a
    version of this test that hoisted the build would pass on a plan that
    ignored the dtype entirely.
    """
    at_float32 = str(compile_graph(radiometer()))
    with jax.enable_x64(True):
        at_float64 = str(compile_graph(radiometer()))
    assert "tol UNATTAINABLE at this dtype" in flat(at_float32)
    assert "tol UNATTAINABLE at this dtype" not in flat(at_float64)
    assert "guard reachable, off by default" in flat(at_float64)


@pytest.mark.parametrize("build", [overflowing_outside_latent, improper_outside_prior])
def test_an_unsweepable_outside_latent_is_named_rather_than_silently_pinned(build):
    """A point kappa on a graph that HAS an outside latent must say why.

    `Cauchy(0, 1e6)` and `ImproperUniform` are both legal priors and neither
    has a width this module can step by, so kappa is measured at the centre
    alone -- exactly the pinning this whole interval machinery exists to
    avoid. Printing a bare number there would hide the one case where the
    number really is pinned, so the block's reason names the latent.
    """
    graph = build()
    text = str(compile_graph(graph))
    assert partition(graph).nuts, "fixture no longer has an outside latent"
    assert KAPPA_RANGE.search(text) is None
    assert KAPPA_POINT.search(text) is not None
    assert "kappa sweep held ['z']" in flat(text)


def test_a_probe_point_the_bound_cannot_be_measured_at_is_named_too():
    """A narrowed interval with no note reads as a full one. It is not.

    ``kappa_interval`` drops any probe whose ``condition_bound`` is not
    finite, which is right -- ``tol = target / inf`` is 0.0 and ``/ nan`` is
    ``nan``, and neither is a tolerance -- but the interval that comes back
    then covers less of the outside latent's prior than it claims to, and
    ``tol`` is derived from the largest kappa the sweep could SEE rather than
    from the largest it went looking for. So the block's reason has to say
    which latent's probes were dropped.

    Until ``overflowing_probe_width`` existed nothing in the suite reached
    that branch: branch coverage reported ``refused.append(name)`` and the
    whole ``refused`` clause of ``_sweep_note`` unreached, and deleting the
    append reddened nothing. Measured here: the ``+3`` probe reads ``inf``
    and the ``-3`` probe ``nan``, both from ``width**2`` leaving float32 at
    ``exp(+/-45)``, while ``+/-1`` and the centre are finite -- so the
    interval that survives is [1.0, 6.07e+14] against a prior the sweep was
    asked to cover four points of.

    The complement of ``test_an_unsweepable_outside_latent_is_named_rather_
    than_silently_pinned``, which is the other half of ``_sweep_note``: there
    the latent had no width to step by at all, here it has one and the
    step lands where the bound cannot be measured.
    """
    graph = overflowing_probe_width()
    plan = compile_graph(graph)
    assert partition(graph).nuts == ("tau",)
    text = flat(str(plan))
    assert "kappa sweep could not build the block at ['tau']" in text
    assert "the interval is narrower than the prior it was meant to cover" in text
    # Whatever survived has to still be usable: a non-finite kappa that got
    # through would make `tol` 0.0 or nan, and both print without complaint.
    assert math.isfinite(plan_module.kappa_upper(plan.exact.kappa))
    assert plan.exact.tol > 0.0 and math.isfinite(plan.exact.tol)


def test_a_broken_probe_point_is_raised_rather_than_noted():
    """``StructureError`` is a fault. It must not arrive as a footnote.

    ``classify``'s own module docstring states the package's rule: a
    ``StructureError`` is caught at exactly one call site -- ``check_
    linearity``'s -- and allowed through everywhere else, because from
    ``check_gaussian`` it means a declaration the model contradicts. The two
    ``except BayesmithError`` handlers in ``plan.py`` were wider than their
    own docstrings, which say ``NotGaussian``, and ``StructureError`` is a
    SIBLING of that class rather than a subclass, so the wider spelling
    swallowed it.

    Measured, on this fixture: ``partition`` accepts the graph because at the
    prior centre ``tau`` is 1.0 and the width is 1.0, and the ``-3``-sigma
    probe puts the width at ``-0.5``. Under ``except BayesmithError``,
    ``compile()`` returned an ordinary-looking plan carrying ``kappa in
    [15.2, 356.0]`` -- built from the three probes that survived -- and the
    same "could not build the block" note the test above asserts, which is
    exactly the sentence that makes a broken model indistinguishable from a
    graph whose bound merely overflowed. Nothing else in the plan, and nothing
    in ``Posterior``, said the prior width goes negative.
    """
    graph = unfloored_probe_width()
    assert partition(graph).exact == ("x",), "the centre must still classify"
    with pytest.raises(StructureError) as caught:
        compile_graph(graph)
    assert "'x'" in str(caught.value)
    assert "not strictly positive" in str(caught.value)
    assert "Add a floor" in str(caught.value)


def test_the_plan_covers_exactly_the_partition_the_classifier_made():
    """No latent invented, none dropped, and the exact block first.

    Swept over every fixture the two tables above reach, so a plan that lost a
    block on one shape -- the empty exact block, the empty NUTS set, a block
    with three members -- is caught here rather than only where that shape
    happens to be printed.
    """
    builds = [param.values[0] for param in POINT_KAPPA + INTERVAL_KAPPA]
    builds += [bilinear_pair, orphaned_child_latent, overflowing_outside_latent]
    for build in builds:
        graph = build()
        classification = partition(graph)
        plan = compile_graph(graph)
        expected = [
            names for names in (classification.exact, classification.nuts) if names
        ]
        assert [block.latents for block in plan.blocks] == expected, build.__name__
        assert (plan.exact is None) is (not classification.exact), build.__name__
        assert (plan.sampled is None) is (not classification.nuts), build.__name__
        assert plan.guard_hoisted is bool(
            classification.exact and classification.nuts
        ), build.__name__


@pytest.mark.parametrize(
    "build", [two_linear_latents, indirect_ancestor, bilinear_pair]
)
def test_the_plan_is_reproducible(build):
    """Every number in it is measured, so two compiles must print one text.

    The linearity probes and the power iteration are both randomised and both
    take a fixed default key. A plan that varied run to run would make every
    number in it unquotable.
    """
    graph = build()
    assert str(compile_graph(graph)) == str(compile_graph(graph))


def test_kappa_and_tol_are_absent_where_there_is_no_linear_system():
    """A NUTS block solves nothing, so a tolerance for it would be fiction."""
    plan = compile_graph(bilinear_pair())
    assert plan.exact is None
    assert [block.kappa for block in plan.blocks] == [None]
    assert [block.tol for block in plan.blocks] == [None]
    assert KAPPA_POINT.search(str(plan)) is None
    assert TOL.search(str(plan)) is None


def test_the_evidence_line_reports_the_shape_of_the_linearity_check():
    """ "Checked" without the counts cannot be told from a single-point probe.

    A one-at-point check is the moderate-parameter probe `check_linearity`'s
    own docstring names as the failure mode to avoid, and its verdict is
    indistinguishable from a three-point one unless the plan says how many
    points there were.
    """
    graph = indirect_ancestor()
    plan = compile_graph(graph)
    linearity = partition(graph).linearity
    scales = len(next(iter(linearity.values())))
    text = str(plan)
    assert f"{scales} scales x {len(linearity)} at-points" in text
    assert scales >= 3 and len(linearity) >= 3


def test_bayesmith_compile_is_the_function_this_module_defines():
    """The lazy re-export must hand back the real function, not a shim.

    Mirrors `test_public_api`'s identity checks: a `__getattr__` that returned
    the module, or a wrapper, would pass a `hasattr` check and fail here.
    """
    import bayesmith

    assert bayesmith.compile is plan_module.compile
    assert "compile" in bayesmith.__all__
    assert isinstance(compile_graph(straight_line()), InferencePlan)


def test_the_probe_offsets_are_two_sided_about_the_anchor():
    """The anchor is swept too, so it must sit strictly inside the offsets.

    Not a restatement of the constant: what it rules out is the shape where
    every offset has one sign, which is how a "scan across magnitudes" comes
    to be one-sided without anybody noticing.
    """
    assert min(KAPPA_PROBE_SIGMAS) < 0.0 < max(KAPPA_PROBE_SIGMAS)
    assert 0.0 not in KAPPA_PROBE_SIGMAS
