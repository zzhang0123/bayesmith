"""What the SBC harness must decide, and what it must refuse to decide.

Layered on purpose (see the fast/full split in AGENTS.md). Everything that can
be decided from synthetic ranks, a closed-form sampler or a pure predicate runs
in the FAST layer, because those are the cells that catch a broken rank, a
mis-set floor or a dropped replicate. The two heavy grids -- a hundred real
compiled replicates on the exact route, and the same on NUTS -- are marked
``full``: they are the ones that show the harness works against the actual
dispatch, and they cost tens of seconds each.

**Every random cell here declares its seed, its replicate count and the level
it is judged at**, as §9.3 requires. Where a seed was chosen rather than
arbitrary, the measurement that chose it is quoted on the test.
"""

from __future__ import annotations

import dataclasses
import math
import uuid

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import const, det, observe, sample, trace
from bayesmith.artifacts.base import (
    ArtifactRef,
    ComputeBudget,
    TerminationReason,
    TerminationRecord,
)
from bayesmith.artifacts.identity import (
    ArtifactKind,
    FingerprintBundle,
    FingerprintKind,
    fingerprint,
)
from bayesmith.artifacts.refusal import Refusal
from bayesmith.artifacts.reports import Applicability, Conclusion
from bayesmith.artifacts.tasks import EvidenceTask, new_task_meta
from bayesmith.dispatch.task import compile_task
from bayesmith.evaluation import ALPHA
from bayesmith.evaluation import sbc as sbc_module
from bayesmith.evaluation.sbc import (
    REPLICATE_FLOOR,
    REPORT_KIND,
    SbcRanks,
    continuous_rank,
    ranks_are_uniform,
    replicates_meet_floor,
    sbc_ranks,
    sbc_report,
    simulation_based_calibration,
)
from tests.dispatch.test_task_protocol import model_ref

X = jnp.linspace(1.0, 4.0, 8)
SIGMA = 0.5
PRIOR_STD = 2.0

#: The exact route's per-replicate budget. `warmup=1` because the exact
#: (`gcr`) route has no chain to warm up; it is carried so the budget is stated
#: rather than defaulted.
EXACT_BUDGET = ComputeBudget(draws=100, warmup=1)


def line_with(data):
    """``straight_line``'s model at caller-supplied data -- the model callable
    §0.6 requires the replicate loop to re-trace, once per replicate."""

    def model():
        xs = const("X", X)
        w = sample("w", lambda: dist.Normal(0.0, PRIOR_STD))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, SIGMA), mu, obs=data)

    return trace(model)


LINE = line_with(2.5 * X + SIGMA * jax.random.normal(jax.random.key(0), X.shape))


def conjugate_sampler(width: float):
    """The EXACT posterior of ``w`` given ``d``, optionally at the wrong width.

    ``w ~ N(0, PRIOR_STD)`` with ``d ~ N(w X, SIGMA)`` is conjugate, so this is
    the posterior the route is trying to compute -- known in closed form,
    costing no inference at all. At ``width=1`` its ranks must be uniform by
    construction; at ``width=2`` it is the 2x-too-wide error D106 was measured
    against, in a form that needs no sampler to reproduce.
    """

    def sampler(datum, key, n):
        d = jnp.asarray(datum["d"])
        precision = 1.0 / PRIOR_STD**2 + jnp.sum(X**2) / SIGMA**2
        mean = jnp.sum(X * d) / SIGMA**2 / precision
        scale = width / jnp.sqrt(precision)
        return {"w": np.asarray(mean + scale * jax.random.normal(key, (n,)))}

    return sampler


def a_ref(kind: ArtifactKind = ArtifactKind.RESULT) -> ArtifactRef:
    return ArtifactRef(artifact_id=str(uuid.uuid4()), revision=0, artifact_type=kind)


def a_bundle() -> FingerprintBundle:
    return FingerprintBundle(
        **{
            name: fingerprint(FingerprintKind(name), value)
            for name, value in (
                ("model_source", "sbc-test"),
                ("graph_structure", "w -> d"),
                ("data", "simulated"),
                ("task", "simulation"),
            )
        }
    )


def synthetic(
    ranks_by_coordinate: dict[str, tuple[float, ...]],
    *,
    requested: int | None = None,
    refused: int = 0,
    unconverged: int = 0,
    undrawn: int = 0,
) -> SbcRanks:
    """An :class:`SbcRanks` built by hand, so a verdict can be tested exactly.

    The point of the split between :func:`sbc_ranks` and :func:`sbc_report`:
    what the report concludes from a given set of ranks is decidable without
    running a single replicate, so these cells are deterministic and instant.
    """
    coordinates = tuple(ranks_by_coordinate)
    usable = len(ranks_by_coordinate[coordinates[0]])
    return SbcRanks(
        coordinates=coordinates,
        ranks=tuple(ranks_by_coordinate[name] for name in coordinates),
        requested=requested if requested is not None else usable,
        usable=usable,
        refused=refused,
        unconverged=unconverged,
        undrawn=undrawn,
        route="synthetic",
        subject_ref=a_ref(),
        fingerprints=a_bundle(),
    )


def uniform_ranks(n: int) -> tuple[float, ...]:
    """Ranks a correct posterior would produce: the midpoints of n equal bins.

    Deterministic rather than drawn: a KS test against this sample has the
    largest p-value the sample size admits, so a PASS here is a statement
    about the check rather than about a lucky draw.
    """
    return tuple((index + 0.5) / n for index in range(n))


def piled_ranks(n: int) -> tuple[float, ...]:
    """Ranks a 2x-too-wide posterior produces: everything crowded into the
    middle third, which is what "the truth is never in the tails" looks like."""
    return tuple(1.0 / 3.0 + (index + 0.5) / (3.0 * n) for index in range(n))


def finding(report, code):
    matches = [item for item in report.findings if item.code == code]
    assert matches, f"no {code} finding in {[f.code for f in report.findings]}"
    return matches[0]


# ----------------------------------------------------- D106, at its two faces


@pytest.mark.parametrize(
    "usable, admitted",
    [
        (0, False),
        (1, False),
        (REPLICATE_FLOOR - 2, False),
        (REPLICATE_FLOOR - 1, False),
        (REPLICATE_FLOOR, True),
        (REPLICATE_FLOOR + 1, True),
        (10 * REPLICATE_FLOOR, True),
    ],
)
def test_replicate_floor_admits_exactly_the_counts_it_was_measured_for(
    usable, admitted
):
    """D106's boundary from both sides, including the two adjacent integers.

    The pair (99, 100) is the whole content of the threshold: the count below
    is refused and the count at is admitted. Moving the constant either way, or
    turning ``>=`` into ``>``, moves one of these two cells.
    """
    assert replicates_meet_floor(usable) is admitted


def test_the_replicate_floor_is_the_number_ten_seeds_measured():
    """D106 = 100, not the 50 the R3 plan proposed before Task 6.1 measured it.

    Pinned as a value because the docstring's provenance table is about THIS
    number; a floor that drifted to 50 again would leave the table describing a
    measurement nothing in the code answers to. The measurement itself is
    ``docs/probes/probe_30_sbc_replicate_floor.py``, and its own worst-case
    line is what this constant records.
    """
    assert REPLICATE_FLOOR == 100


# ------------------------------------------- the Bonferroni level's two faces


@pytest.mark.parametrize(
    "p_value, level, uniform",
    [
        (0.0, ALPHA, False),
        (float(np.nextafter(ALPHA, -math.inf)), ALPHA, False),
        (ALPHA, ALPHA, True),
        (float(np.nextafter(ALPHA, math.inf)), ALPHA, True),
        (1.0, ALPHA, True),
        (ALPHA / 2, ALPHA / 2, True),
        (float(np.nextafter(ALPHA / 2, -math.inf)), ALPHA / 2, False),
    ],
)
def test_rank_uniformity_admits_a_p_value_at_the_level_and_refuses_below_it(
    p_value, level, uniform
):
    """The rejection region is ``p < level``, so the level itself is admitted.

    The last two rows use ``ALPHA / 2``, the Bonferroni level for two latent
    coordinates: the boundary follows the derived level rather than sitting at
    D104, which is what makes the correction a derivation and not a second
    threshold.
    """
    assert ranks_are_uniform(p_value, level) is uniform


# ---------------------------------------------------------------- the rank


def test_the_rank_is_the_weighted_share_of_draws_below_the_truth():
    draws = np.array([0.0, 1.0, 2.0, 3.0])
    weights = np.full(4, 0.25)
    assert continuous_rank(draws, weights, 2.0) == pytest.approx(0.5)


def test_the_rank_uses_the_posterior_s_weights_and_does_not_resample():
    """R2 §0.5 in one cell: a weighted sample keeps its weights.

    Two draws with weights 0.7 / 0.3. An unweighted rank would answer 0.5 for
    a truth between them; the weighted one answers 0.7, and only the second
    is a statement about the posterior that was actually computed.
    """
    draws = np.array([0.0, 10.0])
    weights = np.array([0.7, 0.3])
    assert continuous_rank(draws, weights, 5.0) == pytest.approx(0.7)


@pytest.mark.parametrize("truth, expected", [(-1.0, 0.0), (99.0, 1.0)])
def test_the_rank_saturates_at_the_ends_of_its_interval(truth, expected):
    draws = np.array([0.0, 10.0])
    weights = np.array([0.7, 0.3])
    assert continuous_rank(draws, weights, truth) == pytest.approx(expected)


# ------------------------------------------- the verdict, from synthetic ranks


def test_uniform_ranks_at_the_floor_pass():
    report = sbc_report(synthetic({"w": uniform_ranks(REPLICATE_FLOOR)}))
    assert report.applicability is Applicability.APPLICABLE
    assert report.conclusion is Conclusion.PASS
    assert report.report_kind == REPORT_KIND


def test_ranks_piled_in_the_middle_fail():
    report = sbc_report(synthetic({"w": piled_ranks(REPLICATE_FLOOR)}))
    assert report.applicability is Applicability.APPLICABLE
    assert report.conclusion is Conclusion.FAIL
    observed = finding(report, "sbc_rank_uniformity").observed
    assert observed[0] == "w"
    assert observed[2] < ALPHA


def test_too_few_replicates_abstain_rather_than_passing():
    """The cell §0.6 exists for: a check with no power must not report PASS.

    Uniform ranks -- the most passing sample there is -- one short of the
    floor. A harness that judged them would say PASS on evidence it has
    already been measured to lack.
    """
    report = sbc_report(synthetic({"w": uniform_ranks(REPLICATE_FLOOR - 1)}))
    assert report.applicability is Applicability.APPLICABLE
    assert report.conclusion is Conclusion.ABSTAIN
    below = finding(report, "replicates_below_floor")
    assert below.observed == REPLICATE_FLOOR - 1
    assert below.expected == REPLICATE_FLOOR


@pytest.mark.parametrize(
    "bucket", ["refused", "unconverged", "undrawn"]
)
def test_one_replicate_that_produced_no_rank_abstains_the_whole_report(bucket):
    """§0.6's hard rule: no silently dropped replicate, whatever dropped it.

    All three buckets, one at a time, on top of a full floor's worth of
    perfectly uniform ranks. The report abstains anyway, because "99 of the
    100 that worked were uniform" is a different claim from the one SBC makes.
    """
    ranks = synthetic(
        {"w": uniform_ranks(REPLICATE_FLOOR)},
        requested=REPLICATE_FLOOR + 1,
        **{bucket: 1},
    )
    report = sbc_report(ranks)
    assert report.conclusion is Conclusion.ABSTAIN
    incomplete = finding(report, "replicates_not_completed")
    assert incomplete.observed == (
        ranks.refused,
        ranks.unconverged,
        ranks.undrawn,
    )
    assert incomplete.expected == 0
    assert finding(report, "sbc_replicate_accounting").observed[0] == (
        REPLICATE_FLOOR + 1
    )


def test_an_incomplete_replicate_outranks_a_short_budget_in_the_report():
    """Both faults at once: the report names the failures, not the count.

    "45 of 100 replicates diverged" and "you only asked for 45" are different
    facts, and reporting the second when the first is true would send a reader
    to raise a budget that was never the problem.
    """
    report = sbc_report(
        synthetic(
            {"w": uniform_ranks(10)}, requested=REPLICATE_FLOOR, unconverged=90
        )
    )
    assert report.conclusion is Conclusion.ABSTAIN
    assert next(item.code for item in report.findings) == "replicates_not_completed"


def test_every_coordinate_is_tested_at_the_bonferroni_corrected_level():
    """K coordinates share ALPHA; the level in the findings says so.

    Two fixtures pin the DIRECTION rather than one pinning a number: with one
    coordinate the level is ALPHA, with two it is ALPHA/2. A correction
    applied the wrong way round would divide by 1 and multiply by 2.
    """
    one = sbc_report(synthetic({"w": uniform_ranks(REPLICATE_FLOOR)}))
    two = sbc_report(
        synthetic(
            {
                "gain": uniform_ranks(REPLICATE_FLOOR),
                "t_ant": uniform_ranks(REPLICATE_FLOOR),
            }
        )
    )
    assert finding(one, "sbc_rank_uniformity").expected == pytest.approx(ALPHA)
    assert finding(two, "sbc_rank_uniformity").expected == pytest.approx(ALPHA / 2)
    assert len([f for f in two.findings if f.code == "sbc_rank_uniformity"]) == 2


def test_one_bad_coordinate_fails_the_whole_report():
    report = sbc_report(
        synthetic(
            {
                "gain": uniform_ranks(REPLICATE_FLOOR),
                "t_ant": piled_ranks(REPLICATE_FLOOR),
            }
        )
    )
    assert report.conclusion is Conclusion.FAIL


def test_the_verdict_can_be_recomputed_from_the_findings_alone():
    """G8: no verdict here needs a sampler log to audit.

    Each uniformity finding carries ``(coordinate, D, p, n)`` and the level it
    was compared against, so a consumer holding only the report reaches the
    same conclusion the report did.
    """
    for ranks, expected in (
        (synthetic({"w": uniform_ranks(REPLICATE_FLOOR)}), Conclusion.PASS),
        (synthetic({"w": piled_ranks(REPLICATE_FLOOR)}), Conclusion.FAIL),
    ):
        report = sbc_report(ranks)
        recomputed = all(
            ranks_are_uniform(item.observed[2], item.expected)
            for item in report.findings
            if item.code == "sbc_rank_uniformity"
        )
        assert (Conclusion.PASS if recomputed else Conclusion.FAIL) is expected
        assert report.conclusion is expected


# -------------------------------------------------- the sampler arm (§0.11)


def test_a_closed_form_posterior_passes_through_the_sampler_arm():
    """The exact conjugate posterior, judged by the same rank arithmetic.

    seed 0, N = REPLICATE_FLOOR replicates, 400 draws each, level = ALPHA.
    A closed-form posterior is uniform by construction, so a FAIL here is the
    harness's arithmetic and nothing else -- which is what makes this the
    cheapest honest calibration cell in the file.
    """
    report = simulation_based_calibration(
        LINE,
        key=jax.random.key(0),
        replicates=REPLICATE_FLOOR,
        model_ref=model_ref(),
        sampler=conjugate_sampler(1.0),
        sampler_draws=400,
        subject_ref=a_ref(),
    )
    assert report.conclusion is Conclusion.PASS, finding(
        report, "sbc_rank_uniformity"
    ).message
    assert finding(report, "sbc_replicate_accounting").observed[5] == "sampler"


def test_the_same_posterior_at_twice_its_width_fails():
    """The 2x error D106 was measured against, on the same seed and budget.

    Same seed, same replicate count, same draws: the ONLY difference from the
    cell above is the factor of two on the posterior scale. If this passed,
    the floor would be describing a power the harness does not have.
    """
    report = simulation_based_calibration(
        LINE,
        key=jax.random.key(0),
        replicates=REPLICATE_FLOOR,
        model_ref=model_ref(),
        sampler=conjugate_sampler(2.0),
        sampler_draws=400,
        subject_ref=a_ref(),
    )
    assert report.conclusion is Conclusion.FAIL
    assert finding(report, "sbc_rank_uniformity").observed[2] < ALPHA


def test_the_sampler_arm_says_what_it_is_about_or_refuses_to_run():
    with pytest.raises(TypeError, match="subject_ref"):
        sbc_ranks(
            LINE,
            key=jax.random.key(0),
            replicates=4,
            model_ref=model_ref(),
            sampler=conjugate_sampler(1.0),
            sampler_draws=10,
        )


def test_the_sampler_arm_needs_a_draw_count():
    with pytest.raises(ValueError, match="sampler_draws"):
        sbc_ranks(
            LINE,
            key=jax.random.key(0),
            replicates=4,
            model_ref=model_ref(),
            sampler=conjugate_sampler(1.0),
            subject_ref=a_ref(),
        )


# ------------------------------------------------- the route arm's plumbing


def test_the_route_arm_runs_the_real_dispatch_and_abstains_below_the_floor():
    """seed 3, N = 20 on the exact route: the plumbing case, fast layer.

    Twenty real replicates -- one ``SimulationTask(PRIOR)`` and twenty
    compiled, executed ``PosteriorTask``s -- so this cell fails if the wiring
    to dispatch breaks. It abstains because 20 is below D106, which is also
    the ABSTAIN-by-budget path on the route arm rather than on synthetic
    ranks.
    """
    report = simulation_based_calibration(
        LINE,
        key=jax.random.key(3),
        replicates=20,
        model_ref=model_ref(),
        build=lambda datum: line_with(datum["d"]),
        budget=EXACT_BUDGET,
    )
    assert report.conclusion is Conclusion.ABSTAIN
    accounting = finding(report, "sbc_replicate_accounting").observed
    assert accounting[:5] == (20, 20, 0, 0, 0)
    assert accounting[5] == "gcr"
    assert finding(report, "replicates_below_floor").observed == 20
    assert report.subject_ref.artifact_type is ArtifactKind.RESULT
    assert report.meta.parent_refs == (report.subject_ref,)


def test_the_route_arm_ranks_each_replicate_against_its_own_truth():
    """seed 3, N = 8: the ranks are in [0, 1] and are not all the same.

    A harness that ranked every replicate against ONE truth, or that reused
    one replicate's draws, produces a degenerate rank set; both faults survive
    a KS test at small N, so they are checked directly here.
    """
    ranks = sbc_ranks(
        LINE,
        key=jax.random.key(3),
        replicates=8,
        model_ref=model_ref(),
        build=lambda datum: line_with(datum["d"]),
        budget=EXACT_BUDGET,
    )
    assert ranks.coordinates == ("w",)
    values = ranks.ranks[0]
    assert len(values) == 8
    assert all(0.0 <= value <= 1.0 for value in values)
    assert len(set(values)) == 8


# --------------------------------------- replicates that did not produce ranks


def a_real_refusal() -> Refusal:
    """A genuine Refusal from ``compile_task``, not a hand-built stand-in."""
    refusal = compile_task(
        LINE,
        EvidenceTask(meta=new_task_meta(label="sbc-refusal")),
        model_ref=model_ref(),
    )
    assert isinstance(refusal, Refusal)
    return refusal


def test_a_refused_replicate_is_counted_and_never_dropped(monkeypatch):
    """Every replicate refuses: the report abstains and says so, twenty times.

    The alternative -- propagating the first refusal -- would replace a verdict
    about a calibration run with a verdict about its first bad draw, and §0.6
    asks for the count.
    """
    refusal = a_real_refusal()
    monkeypatch.setattr(
        sbc_module, "_route_replicate", lambda *a, **k: refusal
    )
    report = simulation_based_calibration(
        LINE,
        key=jax.random.key(3),
        replicates=20,
        model_ref=model_ref(),
        build=lambda datum: line_with(datum["d"]),
        budget=EXACT_BUDGET,
    )
    assert report.conclusion is Conclusion.ABSTAIN
    assert finding(report, "replicates_not_completed").observed == (20, 0, 0)
    assert finding(report, "sbc_replicate_accounting").observed[:5] == (
        20,
        0,
        20,
        0,
        0,
    )
    assert report.subject_ref.artifact_type is ArtifactKind.RESULT


def test_a_replicate_that_did_not_converge_is_counted_separately(monkeypatch):
    """A DIVERGED replicate is not a FAIL and not a silent PASS.

    Rewriting only the termination reason leaves draws that would have ranked
    perfectly well, which is the point: the abstention is about the run's own
    report of itself, not about the numbers it returned.
    """
    original = sbc_module._route_replicate

    def diverged(*args, **kwargs):
        result = original(*args, **kwargs)
        run = dataclasses.replace(
            result.run,
            termination=TerminationRecord(
                reason=TerminationReason.DIVERGED, message="forced in a test"
            ),
        )
        return dataclasses.replace(result, run=run)

    monkeypatch.setattr(sbc_module, "_route_replicate", diverged)
    report = simulation_based_calibration(
        LINE,
        key=jax.random.key(3),
        replicates=6,
        model_ref=model_ref(),
        build=lambda datum: line_with(datum["d"]),
        budget=EXACT_BUDGET,
    )
    assert report.conclusion is Conclusion.ABSTAIN
    assert finding(report, "replicates_not_completed").observed == (0, 6, 0)


def test_a_posterior_with_no_draws_is_counted_rather_than_ignored(monkeypatch):
    monkeypatch.setattr(sbc_module, "_posterior_draws", lambda result: None)
    report = simulation_based_calibration(
        LINE,
        key=jax.random.key(3),
        replicates=6,
        model_ref=model_ref(),
        build=lambda datum: line_with(datum["d"]),
        budget=EXACT_BUDGET,
    )
    assert report.conclusion is Conclusion.ABSTAIN
    assert finding(report, "replicates_not_completed").observed == (0, 0, 6)


def test_a_refused_prior_simulation_comes_back_as_the_refusal(monkeypatch):
    """The one Refusal the harness DOES propagate: nothing was simulated.

    A replicate's refusal is counted; a refusal of the simulation itself
    leaves no replicates to count, so there is no report to write.
    """
    refusal = a_real_refusal()
    monkeypatch.setattr(sbc_module, "compile_task", lambda *a, **k: refusal)
    answer = simulation_based_calibration(
        LINE,
        key=jax.random.key(3),
        replicates=20,
        model_ref=model_ref(),
        build=lambda datum: line_with(datum["d"]),
        budget=EXACT_BUDGET,
    )
    assert answer is refusal


# ------------------------------------------------------ argument validation


def test_a_harness_with_neither_arm_refuses_to_guess():
    with pytest.raises(TypeError, match="build=... "):
        sbc_ranks(LINE, key=jax.random.key(0), replicates=4, model_ref=model_ref())


def test_the_replicate_count_is_never_defaulted_or_zero():
    with pytest.raises(ValueError, match="at least one replicate"):
        sbc_ranks(
            LINE,
            key=jax.random.key(0),
            replicates=0,
            model_ref=model_ref(),
            build=lambda datum: line_with(datum["d"]),
        )


def test_the_graph_is_a_graph():
    with pytest.raises(TypeError, match="is a Graph"):
        sbc_ranks(
            "not a graph",
            key=jax.random.key(0),
            replicates=4,
            model_ref=model_ref(),
            build=lambda datum: line_with(datum["d"]),
        )


# ---------------------------------------------------------- the heavy grids


@pytest.mark.full
@pytest.mark.slow
def test_the_exact_route_at_the_floor_is_calibrated():
    """seed 1, N = 100 real replicates on the exact (gcr) route, level ALPHA.

    Measured before this test existed, over five seeds at N=100 (correct /
    2x-wide / 2x-narrow KS p): seed 0 gave 0.3706 / 0.0012 / 0.0000, seed 1
    0.6847 / 0.0006 / 0.0026, seed 2 0.9532 / 0.0105 / 0.0006, seed 3 0.0358 /
    0.0006 / 0.0000, seed 4 0.2527 / 0.0001 / 0.0001. Seed 1 is declared here
    because its correct-model p is far from the level on both sides of it;
    seed 3's 0.0358 is the declared false positive the level implies at
    ALPHA = 0.05, and it is why a seed had to be declared rather than left to
    whatever the file happened to use.

    Costs about 19 s.
    """
    report = simulation_based_calibration(
        LINE,
        key=jax.random.key(1),
        replicates=REPLICATE_FLOOR,
        model_ref=model_ref(),
        build=lambda datum: line_with(datum["d"]),
        budget=EXACT_BUDGET,
    )
    assert report.conclusion is Conclusion.PASS
    accounting = finding(report, "sbc_replicate_accounting").observed
    assert accounting[:5] == (REPLICATE_FLOOR, REPLICATE_FLOOR, 0, 0, 0)
    assert accounting[5] == "gcr"


@pytest.mark.full
@pytest.mark.slow
def test_the_exact_route_fails_when_its_draws_are_stretched(monkeypatch):
    """The same seed and budget, with every posterior doubled in width.

    ``_posterior_draws`` is the seam between the route and the rank, so
    stretching what it returns is a 2x-too-wide route and nothing else: the
    same data, the same truths, the same replicate accounting. Measured at
    p = 0.0006 against a level of 0.05.
    """
    original = sbc_module._posterior_draws

    def stretched(result):
        answer = original(result)
        draws, weights = answer
        return (
            {
                name: value.mean() + 2.0 * (value - value.mean())
                for name, value in draws.items()
            },
            weights,
        )

    monkeypatch.setattr(sbc_module, "_posterior_draws", stretched)
    report = simulation_based_calibration(
        LINE,
        key=jax.random.key(1),
        replicates=REPLICATE_FLOOR,
        model_ref=model_ref(),
        build=lambda datum: line_with(datum["d"]),
        budget=EXACT_BUDGET,
    )
    assert report.conclusion is Conclusion.FAIL
    assert finding(report, "sbc_rank_uniformity").observed[2] < ALPHA


def line_without_the_linear_hint(data):
    """The same straight line, with ``linear_in`` withheld.

    The exact route applies only to a declared-affine deterministic node, so
    dropping the declaration routes this identical model through NUTS instead.
    That is what makes it the sampled-route twin of ``line_with``: same prior,
    same noise, same truth -- a different route.
    """

    def model():
        xs = const("X", X)
        w = sample("w", lambda: dist.Normal(0.0, PRIOR_STD))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs)
        observe("d", lambda m: dist.Normal(m, SIGMA), mu, obs=data)

    return trace(model)


LINE_NUTS = line_without_the_linear_hint(
    2.5 * X + SIGMA * jax.random.normal(jax.random.key(0), X.shape)
)

#: 500 warmup and 500 draws per replicate. The warmup is what an adaptive
#: sampler needs and the exact route does not, which is why the two routes do
#: not share a budget.
NUTS_BUDGET = ComputeBudget(draws=500, warmup=500, chains=1)


def bilinear_with(data):
    """``mu = gain * t_ant * X``: affine in each latent, not in the pair.

    Its likelihood is invariant under ``(gain, t_ant) -> (-gain, -t_ant)``, so
    the posterior has two mirror-image modes separated by a barrier at
    ``gain = 0`` where ``mu`` vanishes.
    """
    grid = jnp.linspace(0.5, 3.0, 10)

    def model():
        xs = const("X", grid)
        gain = sample("gain", lambda: dist.Normal(1.0, 1.0))
        t_ant = sample("t_ant", lambda: dist.Normal(2.0, 3.0))
        mu = det(
            "mu",
            lambda g_, t_, x_: g_ * t_ * x_,
            gain,
            t_ant,
            xs,
            linear_in=("gain", "t_ant"),
        )
        observe("d", lambda m: dist.Normal(m, 0.3), mu, obs=data)

    return trace(model)


BILINEAR = bilinear_with(
    2.0 * jnp.linspace(0.5, 3.0, 10)
    + 0.3 * jax.random.normal(jax.random.key(0), (10,))
)


@pytest.mark.full
@pytest.mark.slow
def test_the_sampled_route_is_calibrated_through_the_same_harness():
    """G4's other half: NUTS, judged by the identical rank definition.

    seed 1, N = 100 replicates at 500 warmup + 500 draws. Measured over three
    seeds before this test existed: KS p = 0.3979 (seed 0), 0.6509 (seed 1),
    0.6509 (seed 2). Seed 1 is declared. The whole point of the cell is that
    NOTHING about the harness changes between here and the exact-route cell
    above -- same prior simulation, same continuous weighted rank, same
    Bonferroni level -- so a comparison between the two routes is a comparison
    rather than two differently-scored experiments.

    **The assertion is not the report's own verdict, and that is deliberate.**
    Passing at ALPHA is a random acceptance with a declared 5% false-positive
    rate; on a fixed seed that would normally be harmless, except that a NUTS
    trajectory is chaotic -- one ULP of difference in a gradient sends the
    chain somewhere else -- so a machine with a different BLAS redraws this
    p-value from its null and reds a cell nobody touched, one time in twenty.
    That is precisely the fixture this repository has burned release tags on.

    So the cell asserts the PROPERTY at a level of its own. The FORM is
    derived: the KS null makes P(p < L) = L for a correct route, so the
    platform-flake rate is exactly L. The CONSTANT 1e-3 is FITTED, and
    bracketed on both sides by measurements rather than chosen for roundness:
    it sits 650x below this fixture's own p (0.6509) and still refuses both
    miscalibrated cells in this file -- the 2x-stretched exact route at 6e-4
    and the sign-symmetric bilinear at 5e-19. The accounting assertions below
    are exact and carry no such caveat.

    Costs about 35 s.
    """
    report = simulation_based_calibration(
        LINE_NUTS,
        key=jax.random.key(1),
        replicates=REPLICATE_FLOOR,
        model_ref=model_ref(),
        build=lambda datum: line_without_the_linear_hint(datum["d"]),
        budget=NUTS_BUDGET,
    )
    assert report.applicability is Applicability.APPLICABLE
    assert report.conclusion is not Conclusion.ABSTAIN
    accounting = finding(report, "sbc_replicate_accounting").observed
    assert accounting[:5] == (REPLICATE_FLOOR, REPLICATE_FLOOR, 0, 0, 0)
    assert accounting[5] == "nuts"
    uniformity = finding(report, "sbc_rank_uniformity")
    assert uniformity.observed[0] == "w"
    assert uniformity.observed[3] == REPLICATE_FLOOR
    assert uniformity.observed[2] > 1e-3, uniformity.message


@pytest.mark.full
@pytest.mark.slow
def test_a_sign_symmetric_likelihood_is_caught_on_the_sampled_route():
    """A real inference failure, found by this harness rather than assumed.

    ``bilinear_pair`` was the fixture the R3 plan named for the NUTS SBC cell,
    on the strength of a cost measurement (probe_28 §5) that never asked
    whether it passes. It does not, and not marginally: measured KS p = 5.3e-19
    for ``gain`` and 3.3e-14 for ``t_ant`` at seed 0, N = 100, 100 warmup + 100
    draws, and unchanged at 2000 draws + 1000 warmup (D = 0.466 / 0.403), so it
    is not an under-budgeted chain.

    The cause is in the model rather than in the sampler: the likelihood is
    invariant under a joint sign flip, single-chain NUTS settles in whichever
    mirror mode it started in, and when the truth is in the other one the rank
    saturates. About half of the ranks land in the top fifth of the interval.

    Pinned as a FAIL because it is exactly what §3.3 says SBC is for -- "SBC
    checks the inference algorithm and also exposes model implementation and
    parameterisation errors" -- and because a fixture this far from calibrated
    must not be quoted anywhere as a passing one.

    Costs about 42 s.
    """
    report = simulation_based_calibration(
        BILINEAR,
        key=jax.random.key(0),
        replicates=REPLICATE_FLOOR,
        model_ref=model_ref(),
        build=bilinear_with_datum,
        budget=ComputeBudget(draws=100, warmup=100, chains=1),
    )
    assert report.conclusion is Conclusion.FAIL
    assert finding(report, "sbc_replicate_accounting").observed[:5] == (
        REPLICATE_FLOOR,
        REPLICATE_FLOOR,
        0,
        0,
        0,
    )
    uniformity = [
        item for item in report.findings if item.code == "sbc_rank_uniformity"
    ]
    assert {item.observed[0] for item in uniformity} == {"gain", "t_ant"}
    assert all(item.expected == pytest.approx(ALPHA / 2) for item in uniformity)
    assert all(item.observed[2] < 1e-10 for item in uniformity)


def bilinear_with_datum(datum):
    return bilinear_with(datum["d"])


def test_a_replicate_missing_one_latent_is_counted_rather_than_ranked():
    """A sampler that answers for only some latents cannot rank a replicate.

    The same hole a route that ELIMINATED a latent opens: draws for some
    coordinates and none for others. Counting it as undrawn keeps every
    coordinate's KS test at one shared N; ranking what came back would run
    them at different ones and report both against the same level.
    """
    report = simulation_based_calibration(
        LINE,
        key=jax.random.key(0),
        replicates=6,
        model_ref=model_ref(),
        sampler=lambda datum, key, n: {},
        sampler_draws=10,
        subject_ref=a_ref(),
    )
    assert report.conclusion is Conclusion.ABSTAIN
    assert finding(report, "replicates_not_completed").observed == (0, 0, 6)
