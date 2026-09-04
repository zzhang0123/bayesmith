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
import functools
import importlib.util
import math
import pathlib
import sys
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
    NamedArray,
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
from bayesmith.artifacts.results import WeightedDrawsPosterior
from bayesmith.artifacts.tasks import EvidenceTask, PosteriorTask, new_task_meta
from bayesmith.dispatch.task import compile_task, execute_task
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
from tests.exact.models import plated_latent

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

    **The verdict IS asserted, behind a conditional premise -- AGENTS.md's
    rule (c).** G4 asks a known-calibrated fixture to be reported PASS on the
    sampled route, so this cell says so in as many words. But passing at ALPHA
    is a random acceptance with a declared 5% false-positive rate, and a NUTS
    trajectory is chaotic -- one ULP of difference in a gradient sends the
    chain somewhere else -- so a machine with a different BLAS redraws this
    p-value from its null and reds a cell nobody touched, one time in twenty.
    That is precisely the fixture this repository has burned release tags on.

    So the ordering below is the one rule (c) prescribes. The CONTRACT
    assertions -- the accounting, the coordinate, the replicate count, and a
    band refusing gross miscalibration -- are unconditional and come first.
    Only then is the PREMISE tested: if this platform's p landed below the
    level, the cell SKIPS with ``THIS IS NOT A PASS`` and the measurement, and
    if it did not, ``conclusion is PASS`` is asserted outright. That assertion
    is not implied by the skip guard, which reads the FINDING while the
    assertion reads the report's own conclusion: a ``sbc_report`` that
    assembled FAIL or ABSTAIN out of uniform ranks passes the guard and reds
    on the assertion.

    The band's FORM is derived: the KS null makes P(p < L) = L for a correct
    route, so the rate at which the guard fires on a correct route is exactly
    L. The CONSTANT 1e-3 is FITTED, and bracketed on both sides by
    measurements rather than chosen for roundness: it sits 650x below this
    fixture's own p (0.6509) and still refuses both miscalibrated cells in this
    file -- the 2x-stretched exact route at 6e-4 and the sign-symmetric
    bilinear at 5e-19.

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

    # The PREMISE, and the only conditional line in the cell: everything above
    # ran unconditionally and holds on every platform.
    if not ranks_are_uniform(uniformity.observed[2], uniformity.expected):
        pytest.skip(
            "THIS IS NOT A PASS. G4 wants a known-calibrated fixture reported "
            f"PASS on the sampled route; this run measured KS "
            f"p={uniformity.observed[2]:.4g} against a level of "
            f"{uniformity.expected:.4g}, so the report says FAIL and this cell "
            "has no G4 evidence to give. Measured 0.3979 / 0.6509 / 0.6509 on "
            "seeds 0 / 1 / 2 of the laptop the fixture was written on, so a p "
            "below the level here is either the 5% the level declares or a "
            "NUTS trajectory this platform walks differently -- re-run, and "
            "read the CPU and BLAS lines suite.yml logs, before touching the "
            "harness."
        )
    assert report.conclusion is Conclusion.PASS


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


# ------------------------------ the weighted branch: R2 §0.5 on the result path

RAD_N = 10
RAD_X = jnp.linspace(1.0, 5.0, RAD_N)
#: ``tests.exact.models.radiometer``'s own numbers, restated here because that
#: fixture takes only keyword arguments and the route arm needs the model at
#: CALLER-supplied data, once per replicate.
RAD_KAPPA = 0.05
RAD_FLOOR = 1e-3
RAD_WEIGHT = 3.0


def radiometer_with(data):
    """``sigma_i = kappa |mu_i| + floor`` -- the plan's weighted fixture (§0.12).

    Classified ``gcr+snis``, so its posterior is a
    :class:`~bayesmith.artifacts.results.WeightedDrawsPosterior` and the rank
    it feeds is the weighted one R2 §0.5 froze. Everything else in this file
    routes to an equally weighted sample, which is why that ruling had no
    executable coverage on the result path until these two cells.
    """

    def model():
        xs = const("X", RAD_X)
        w = sample("w", lambda: dist.Normal(0.0, 10.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe(
            "d",
            lambda m: dist.Normal(m, RAD_KAPPA * jnp.abs(m) + RAD_FLOOR),
            mu,
            depends_on_prediction=True,
            obs=data,
        )

    return trace(model)


RADIOMETER = radiometer_with(
    RAD_WEIGHT * RAD_X
    + (RAD_KAPPA * jnp.abs(RAD_WEIGHT * RAD_X) + RAD_FLOOR)
    * jax.random.normal(jax.random.key(6), (RAD_N,))
)

#: Chosen rather than measured, and chosen FAR from uniform on purpose:
#: ``e^0 : e^-1 : e^-2 : e^-3`` normalises to 0.6439 / 0.2369 / 0.0871 /
#: 0.0321, so a harness that quietly used ``1/4`` for each is wrong by 2.6x on
#: the first draw and 7.8x on the last. A real SNIS run on this repository's
#: own weighted fixture is far flatter than that -- measured on ``radiometer``
#: at 64 draws, ESS 63.96 of 64, log-weight spread 0.026 -- so the real route
#: has almost no power against "uniform weights", and pinning the ARITHMETIC
#: needs numbers that were picked. The real route is exercised in the cell
#: below this one, for the thing IT can decide.
KNOWN_LOG_WEIGHTS = (0.0, -1.0, -2.0, -3.0)
KNOWN_DRAWS = (0.0, 1.0, 2.0, 3.0)


def an_exact_posterior_result():
    """One real ``PosteriorResult``, to carry a hand-chosen representation.

    Real rather than assembled field by field: the envelope -- meta, run
    record, fingerprints, ``latent_names`` -- is one an actual dispatch
    produced and one ``PosteriorResult.__post_init__`` accepted, so only the
    numbers under test are hand-written.

    The budget is ``len(KNOWN_DRAWS)`` and not this file's ``EXACT_BUDGET``
    because a result carries a per-draw ``pointwise_log_likelihood``, and
    ``PosteriorResult`` refuses a representation whose draw count disagrees
    with it. Making the envelope the right SIZE is the honest way past that;
    deleting the pointwise array to make room would be editing the real
    artifact to fit the test.
    """
    planned = compile_task(
        LINE,
        PosteriorTask(
            meta=new_task_meta(label="sbc weighted branch"),
            budget=ComputeBudget(draws=len(KNOWN_DRAWS), warmup=1),
        ),
        model_ref=model_ref(),
    )
    assert not isinstance(planned, Refusal)
    result = execute_task(planned, key=jax.random.key(0))
    assert not isinstance(result, Refusal)
    return result


def with_known_weights(result):
    """``result`` carrying a weighted representation whose weights are known."""
    return dataclasses.replace(
        result,
        representation=WeightedDrawsPosterior(
            draws=(
                NamedArray(
                    name="w",
                    value=np.asarray(KNOWN_DRAWS, dtype=float),
                    dims=("draw",),
                ),
            ),
            log_weights=NamedArray(
                name="log_weights",
                value=np.asarray(KNOWN_LOG_WEIGHTS, dtype=float),
                dims=("draw",),
            ),
            ess=None,
            khat=None,
            unreliable=False,
            method="gcr+snis",
        ),
    )


def test_the_weighted_branch_normalises_the_posterior_s_own_log_weights():
    """R2 §0.5 where the weights actually come from: the result, not the caller.

    ``continuous_rank`` is already tested against hand-supplied weights. This
    is the other half -- that ``_posterior_draws`` hands it the posterior's own
    ``log_weights``, shifted by their maximum and normalised to sum to one, and
    neither resamples them away nor replaces them with ``1/n``.

    Three ways of getting it wrong, each refused by its own line: the exact
    softmax refuses uniform weights, the sum refuses UNnormalised ones, and the
    rank ties both to the number SBC actually consumes. The expected weights
    are recomputed here from ``KNOWN_LOG_WEIGHTS`` rather than transcribed, so
    the cell states the FORMULA rather than a machine's arithmetic; the one
    transcribed number, ``0.96794``, is the rank that formula comes to, and it
    is pinned to five decimals so a reader can see the answer without running
    the file.
    """
    result = with_known_weights(an_exact_posterior_result())
    answer = sbc_module._posterior_draws(result)
    assert answer is not None
    draws, weights = answer

    shifted = np.exp(np.asarray(KNOWN_LOG_WEIGHTS) - max(KNOWN_LOG_WEIGHTS))
    expected = shifted / shifted.sum()
    assert weights == pytest.approx(expected)
    assert float(np.sum(weights)) == pytest.approx(1.0)
    # e^0 : e^-1 is 2.718 : 1, and no uniform weighting has that ratio.
    assert weights[0] == pytest.approx(math.e * weights[1])
    assert draws["w"] == pytest.approx(np.asarray(KNOWN_DRAWS))
    # Draws 0, 1 and 2 lie below 2.5; draw 3 does not.
    assert continuous_rank(draws["w"], weights, 2.5) == pytest.approx(
        float(expected[:3].sum())
    )
    assert continuous_rank(draws["w"], weights, 2.5) == pytest.approx(0.96794, abs=1e-5)


def test_the_weighted_route_ranks_with_the_weights_it_was_given():
    """``radiometer`` (gcr+snis) through the real replicate loop: seed 3, N = 8.

    The plan's own weighted fixture on the ROUTE arm, so the branch above is
    reached the way production reaches it -- a compiled, executed
    ``PosteriorTask`` whose posterior arrives weighted -- rather than through a
    representation a test wrote.

    **What is asserted here is structural, and deliberately not numeric.**
    SNIS's khat on this fixture sits near its own reliability threshold:
    measured on this checkout at seed 3, replicates 0..7 returned khat -0.87,
    -0.64, 0.79, 0.45, -0.67, 0.53, 0.83 and 0.44, and the three above the
    threshold stopped ``tolerance_unmet`` and were counted as unconverged
    rather than ranked. A khat is a tail fit over float32 log-weights; pinning
    which side of the threshold each replicate lands on would be pinning one
    machine's arithmetic, which is the fixture failure this repository has
    spent four release tags on. So the cell pins the route NAME, the closed
    accounting, and the interval a weighted rank must live in.

    That interval is the assertion with teeth. A rank is a SUM of weights, so
    it leaves ``[0, 1]`` the moment the weights are not normalised: measured on
    this fixture at 64 draws, the shifted weights sum to 61.66 before
    normalisation.
    """
    ranks = sbc_ranks(
        RADIOMETER,
        key=jax.random.key(3),
        replicates=8,
        model_ref=model_ref(),
        build=lambda datum: radiometer_with(datum["d"]),
        budget=ComputeBudget(draws=100, warmup=1),
    )
    assert not isinstance(ranks, Refusal)
    assert ranks.route == "gcr+snis"
    assert ranks.coordinates == ("w",)
    assert ranks.requested == 8
    assert ranks.usable + ranks.unusable == ranks.requested
    assert ranks.usable >= 1, "no weighted replicate was ranked at all"
    values = ranks.ranks[0]
    assert len(values) == ranks.usable
    assert all(0.0 <= value <= 1.0 for value in values), values


# --------------------------- a vector latent is K questions, at a level of α/K

PLATE_SIZE = 6
PLATE_SIGMA = 0.4
PLATE_TAU = 1.5
#: Stated rather than defaulted: ``plated_conjugate_sampler`` below is the
#: closed-form posterior for THESE numbers, so the graph and the sampler cannot
#: drift apart without the call site saying so.
PLATED = plated_latent(n=PLATE_SIZE, sigma=PLATE_SIGMA, tau=PLATE_TAU)


def plated_conjugate_sampler(width: float = 1.0):
    """``z_i | d_i`` in closed form, for the plate ``z_i ~ N(0, tau)``,
    ``d_i ~ N(z_i, sigma)``: mean ``d_i tau^2 / (tau^2 + sigma^2)`` and scale
    ``sqrt(tau^2 sigma^2 / (tau^2 + sigma^2))``, one independent coordinate per
    plate position."""
    total = PLATE_TAU**2 + PLATE_SIGMA**2
    shrink = PLATE_TAU**2 / total
    scale = width * math.sqrt(PLATE_TAU**2 * PLATE_SIGMA**2 / total)

    def sampler(datum, key, n):
        d = jnp.asarray(datum["d"])
        drawn = shrink * d + scale * jax.random.normal(key, (n,) + d.shape)
        return {"z": np.asarray(drawn)}

    return sampler


def test_a_vector_latent_is_ranked_coordinate_by_coordinate():
    """seed 1, N = REPLICATE_FLOOR, 400 draws: ``z[0]`` .. ``z[5]`` at α/6.

    ``_flat_coordinates``' whole reason for existing, and until this cell
    nothing ran it on a shape: a K-vector latent is K separate calibration
    questions, so it yields K coordinate names and a Bonferroni level of
    ``ALPHA / K`` rather than one pooled test at ``ALPHA``. Measured here:
    six names and a level of 0.008333.

    Deterministic by construction rather than by luck -- the sampler is the
    exact conjugate posterior, so its ranks are uniform whatever the seed
    happens to be. Seed 1's per-coordinate p values were measured at 0.7672,
    0.8765, 0.3078, 0.4049, 0.6004 and 0.3382, the worst of them 37x above the
    corrected level; the 2x-wide twin of this sampler measures 0.00123 down to
    1.16e-05 and fails, which is what makes the PASS above a statement.
    """
    report = simulation_based_calibration(
        PLATED,
        key=jax.random.key(1),
        replicates=REPLICATE_FLOOR,
        model_ref=model_ref(),
        sampler=plated_conjugate_sampler(1.0),
        sampler_draws=400,
        subject_ref=a_ref(),
    )
    uniformity = [
        item for item in report.findings if item.code == "sbc_rank_uniformity"
    ]
    assert tuple(item.observed[0] for item in uniformity) == tuple(
        f"z[{index}]" for index in range(PLATE_SIZE)
    )
    assert len(uniformity) == PLATE_SIZE
    assert all(
        item.expected == pytest.approx(ALPHA / PLATE_SIZE) for item in uniformity
    )
    assert all(item.observed[3] == REPLICATE_FLOOR for item in uniformity)
    assert report.applicability is Applicability.APPLICABLE
    assert report.conclusion is Conclusion.PASS, [
        item.message for item in uniformity
    ]


def test_a_vector_latent_at_twice_its_width_fails_every_coordinate():
    """The same six questions, asked of a posterior twice as wide.

    Same seed, same replicate count, same draw count: the only difference is
    the factor of two. Every coordinate falls below α/6 -- measured 0.00123,
    0.000367, 0.00378, 1.16e-05, 0.000193 and 2.43e-05 at seed 0 -- so the
    cell above is a PASS the harness could have refused.
    """
    report = simulation_based_calibration(
        PLATED,
        key=jax.random.key(0),
        replicates=REPLICATE_FLOOR,
        model_ref=model_ref(),
        sampler=plated_conjugate_sampler(2.0),
        sampler_draws=400,
        subject_ref=a_ref(),
    )
    assert report.conclusion is Conclusion.FAIL
    uniformity = [
        item for item in report.findings if item.code == "sbc_rank_uniformity"
    ]
    assert len(uniformity) == PLATE_SIZE
    assert all(item.observed[2] < ALPHA / PLATE_SIZE for item in uniformity)


# -------------------------------------------------- what a PASS does NOT say


def prior_ignoring_sampler(datum, key, n):
    """A "posterior" that never looks at ``datum``: draws from the prior."""
    return {"w": np.asarray(PRIOR_STD * jax.random.normal(key, (n,)))}


def test_a_posterior_that_ignores_the_data_is_still_calibrated():
    """The harness's blind spot, pinned rather than left to a docstring.

    Ranks of a prior draw among prior draws are uniform by construction, so a
    route that discards the observation entirely scores APPLICABLE x PASS.
    Measured at seed 0, N = REPLICATE_FLOOR, 400 draws: D = 0.0500, p = 0.9532;
    also PASS at seeds 1, 3 and 4 (p = 0.7265, 0.6004, 0.1842), with seed 2's
    0.0474 the false positive ALPHA declares.

    This cell is a PIN, not a guard: the behaviour is CORRECT and must not be
    "fixed". It exists because ``sbc.py``'s module docstring now says this in
    prose, and prose in this repository goes stale on days nobody edits it. The
    assertion is the conclusion plus a margin -- ``p > 0.5`` -- rather than the
    p-value itself, so a platform that reorders one floating-point sum does not
    red a cell about a statistical fact.
    """
    report = simulation_based_calibration(
        LINE,
        key=jax.random.key(0),
        replicates=REPLICATE_FLOOR,
        model_ref=model_ref(),
        sampler=prior_ignoring_sampler,
        sampler_draws=400,
        subject_ref=a_ref(),
    )
    assert report.applicability is Applicability.APPLICABLE
    assert report.conclusion is Conclusion.PASS
    uniformity = finding(report, "sbc_rank_uniformity")
    assert uniformity.observed[2] > 0.5, uniformity.message


# ------------------------------ Task 9: the local reference NPE, through here
#
# **Which rung of CLAUDE.md's ladder these three cells stand on, and why.**
# The subject of Task 9's pin is a 1500-step float32 Adam trajectory, and a
# trajectory is not a property.  Measured on this checkout, re-running the
# IDENTICAL recipe at 24 other init/train seed pairs -- same bank, same
# harness key, same budget, same machine -- fails this arm's own KS test in
# **6 of 24** runs, with ``best_step`` swinging 292 .. 1190.  A cell whose
# subject fails a quarter of the time under a re-run of its own recipe is the
# fixture shape that spent four release tags, so pinning that PASS
# unconditionally is not available.
#
# Rung (a) is not available either: the trajectory cannot be constructed
# deterministically without deleting the thing being measured.  Rung (b) --
# a derived band -- is not available for the mean bias, because over those 24
# retrains ``|bias|`` reaches **0.5468** exact sds, ABOVE the 0.4079 a
# 0.01-in-standardized-space mean shift produces; no band both survives the
# recipe and catches the shift.
#
# So: **rung (c), with rung (d) as its else-branch.**  The CONTRACT assertions
# -- that the sampler arm ran over every replicate, and that the estimator is
# not grossly wrong -- are unconditional and come FIRST, in their own cell.
# The trained network's recorded numbers -- the KS PASS, the width ratios, the
# mean biases -- are the PREMISE, and they are asserted only where a witness
# says this machine's arithmetic reproduced the recorded trajectory; where it
# did not, the cell skips saying ``THIS IS NOT A PASS`` and naming what it
# measured instead.
#
# The fragility is localised, not assumed.  It is NOT in the harness: nudging
# every posterior draw by a relative 1e-6 -- eight times float32 eps -- changes
# **0 of 300** ranks on both arms, and even a 1e-4 nudge (15 and 17 ranks
# moved) leaves KS D and p identical to four digits.  It is in the weights.
# Hence the third cell, which is new here: probe_29 §1's EXACT-posterior
# control has no optimisation in it at all, and it is the arm that can carry
# an unconditional PASS.

_PROBE_29 = (
    pathlib.Path(__file__).resolve().parents[2]
    / "docs"
    / "probes"
    / "probe_29_amortized_candidates.py"
)


@functools.cache
def probe_29():
    """probe_29, imported as a module rather than run as a script.

    The same loader ``tests/evaluation/test_probe28_pins.py`` uses, for the
    same reason: ``docs/probes/`` is a shelf of scripts, not a package, and
    the fixture below -- the graph, the bank, the budget, the seeds -- has to
    be the ONE the probe measured.  A test that rebuilt it here would agree
    with the probe by coincidence and stop agreeing on the day one of the two
    was edited, which is the failure this repository has spent the most time
    repairing.

    ``sys.argv`` is swapped for the import because probe_29's ``main`` reads
    ``sys.argv[1]`` as a replicate count; under pytest that argument is a node
    id.  The guard belongs on this side, which is the side doing something
    unusual.
    """
    spec = importlib.util.spec_from_file_location(_PROBE_29.stem, _PROBE_29)
    assert spec is not None and spec.loader is not None, _PROBE_29
    module = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = [str(_PROBE_29)]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


class _CountedSampler:
    """probe_29's §0.11 sampler, wrapped so the calls to it are countable.

    The replicate census reports which arm produced the ranks as the STRING
    ``"sampler"``, and a guard that reads a spelling can be walked past by a
    rename.  So the string is checked beside an object-level fact that no
    rename reaches: THIS object was the callable the harness drove, it was
    driven once per replicate, and every call asked it for the declared draw
    budget.
    """

    def __init__(self, inner):
        self.inner = inner
        self.draw_requests: list[int] = []

    def __call__(self, datum, key, n):
        self.draw_requests.append(int(n))
        return self.inner(datum, key, n)


@functools.cache
def _exact_posterior_arm():
    """probe_29 §1: the CLOSED-FORM posterior through ``sbc_ranks``' sampler arm."""
    probe = probe_29()
    b = probe.bayesmith()
    _theta, data = b.draw_bank(b.jax.random.key(probe.KEY_BANK), probe.BANK)
    sampler = _CountedSampler(probe.exact_sampler())
    ranks, report = probe.judged(probe.amortize_graph(np.asarray(data[0])), sampler)
    return ranks, report, sampler


@functools.cache
def _reference_npe_arm():
    """probe_29 §2: the trained reference through the same arm. Cached; ~7 s."""
    probe = probe_29()
    q, history, _seconds, (_theta, data) = probe.train_reference()
    sampler = _CountedSampler(probe.reference_sampler(q))
    ranks, report = probe.judged(probe.amortize_graph(np.asarray(data[0])), sampler)
    return q, history, ranks, report, sampler


def _assert_the_sampler_arm_ran_over_every_replicate(report, sampler):
    """The contract both arms owe, independent of how good the estimator is.

    A PASS computed from the replicates that happened to work is a different
    claim from a PASS over the replicates that were asked for, so the census
    is asserted whole; and the route label is checked beside the call count,
    because the label is a spelling and the count is not.
    """
    probe = probe_29()
    assert report.report_kind == REPORT_KIND
    assert report.applicability is Applicability.APPLICABLE
    assert finding(report, "sbc_replicate_accounting").observed == (
        probe.REPLICATES,
        probe.REPLICATES,
        0,
        0,
        0,
        "sampler",
    )
    assert sampler.draw_requests == [probe.DRAWS] * probe.REPLICATES
    uniformity = finding(report, "sbc_rank_uniformity")
    assert uniformity.observed[0] == "theta"
    assert uniformity.observed[3] == probe.REPLICATES
    assert uniformity.expected == pytest.approx(ALPHA)


#: The band the reference estimator's width must stay inside UNCONDITIONALLY,
#: as a multiple of the EXACT posterior width ``tests/test_amortize.py``'s
#: closed form gives for the same observation.
#:
#: **Which half is which.** The FORM -- a two-sided band on
#: ``width / exact_width`` -- is derived: the target is a closed form in numpy,
#: so 1.0 is where a correct estimator sits and no measurement of this machine
#: chose it.  The CONSTANTS 0.60 and 1.60 are FITTED, and fitted against a
#: sweep rather than against one run: re-training the identical recipe at 24
#: other seed pairs put all 72 measured ratios inside ``[0.8036, 1.1932]``, so
#: the binding margin here is 0.204 -- twelve times the 0.0170 Monte-Carlo
#: spread the same statistic shows over twelve draw keys at FIXED weights.
#:
#: **This is looser than the (0.80, 1.20) it replaces, deliberately, and the
#: reason is that (0.80, 1.20) was not an unconditional claim.** The same
#: 24-retrain sweep puts ratios at 0.8036 and 1.1932 -- inside, by 0.004 and
#: 0.007 out of a half-width of 0.20, i.e. 2-3% of it -- and ``JAX_ENABLE_X64=1``
#: on this machine moves one ratio from 0.9539 to 1.1103, which is 78% of that
#: half-width and nine times the Monte-Carlo spread above.  A band with 2% of
#: margin under a re-run of its own recipe records one trajectory; it does not
#: state a property.  What it did catch is not lost: it moves into
#: :func:`test_the_reference_npe_reproduces_its_recorded_calibration` as a
#: ``+-0.05`` PIN, four times tighter than the old band, behind a witness that
#: says the trajectory reproduced.
#:
#: It is deliberately NOT ``tests/test_amortize.py::test_the_posterior_width_matches``'
#: ``0.75 < ratio < 1.35``, which owns this criterion for that file at a LARGER
#: budget (bank 8192, 2000 steps).  Measured at Task 9's budget the retrain
#: cloud reaches 0.8036, only 0.054 above that floor -- three times the
#: Monte-Carlo spread, too thin to carry an unconditional guard here.
WIDTH_BAND = (0.60, 1.60)

#: probe_29 §2's recorded trajectory, and the witness that says this machine
#: reproduced it.  ``best_step`` is a 1500-way discrete fingerprint of the
#: optimisation: measured, it takes 24 DISTINCT values over the 24-seed
#: retrain sweep (292 .. 1190) and moves 322 -> 580 under ``JAX_ENABLE_X64=1``,
#: so it detects a diverged trajectory rather than merely a diverged bit.  The
#: validation minimum is checked beside it because a coincident argmin index
#: is not a coincident set of weights.
#:
#: ``TRAJECTORY_RTOL``'s FORM is derived -- a relative tolerance on a float32
#: loss -- and it is placed between two MEASURED scales rather than chosen:
#: float32 eps on this loss is 6e-8 (900x below), and the gap from this
#: minimum to the second-best validation value is 7.856e-4, i.e. 1.49e-3
#: relative (15x above), so no other step of this run can satisfy it.
RECORDED_BEST_STEP = 322
RECORDED_VALIDATION_MIN = -0.5286270976
TRAJECTORY_RTOL = 1e-4

#: probe_29 §2's recorded width ratios and mean biases at
#: ``probe_29.WIDTH_OBSERVATIONS``, and the tolerance they are pinned to
#: BEHIND the witness above.  At fixed weights and probe_29's fixed draw key
#: these are reproducible to ten digits in-process; the tolerances are three
#: times the Monte-Carlo spread the same statistics show over twelve draw keys
#: at fixed weights (0.0170 for the ratio, 0.0232 for the bias), rounded DOWN
#: to two digits.  So the FORM is derived -- a pin at the instrument's own
#: noise floor -- and the factor three is fitted.
RECORDED_WIDTH_RATIOS = (0.9660, 1.0287, 0.9539)
RECORDED_MEAN_BIASES = (0.2605, -0.0213, 0.0054)
WIDTH_PIN_TOLERANCE = 0.05
BIAS_PIN_TOLERANCE = 0.07


@pytest.mark.full
@pytest.mark.slow
def test_the_exact_posterior_is_calibrated_through_the_sampler_arm():
    """R3 Task 9.1, the boundary check: probe_29 §1, held here unconditionally.

    The amortize problem's EXACT posterior -- closed form, in numpy,
    ``tests/test_amortize.py::exact_posterior`` -- through
    :func:`~bayesmith.evaluation.sbc.sbc_ranks`' sampler arm on the graph
    probe_29 builds, at ``key(11)``, N = 300 replicates, 200 draws each,
    judged at ``ALPHA / K = 0.05``.

    **What it is for.** Everything probe_29 reports rests on that graph's
    forward law being the same joint ``draw_bank`` samples; if it were not,
    the exact posterior of one problem would be scored against replicates of
    another.  The page said so and nothing ran it.  Measured, a graph/bank
    divergence of 40% in the observation noise -- ``dist.Normal(m, NOISE)``
    -> ``dist.Normal(m, 1.4 * NOISE)`` in ``probe_29.amortize_graph`` -- turns
    this cell red at KS D = 0.1267, p = 1.179e-4.

    **Why THIS arm carries an unconditional PASS when the trained one does
    not.** There is no optimisation in it.  Its risk really is the declared
    ALPHA, and the two things that could move it were measured rather than
    argued:

    * the rank statistic is a COUNT, and counts do not move at rounding
      scale.  Nudging every draw by a relative 1e-7 or 1e-6 changes 0 of 300
      ranks; 1e-5 changes 1; even 1e-4 -- three orders above float32 eps, 15
      ranks moved -- leaves ``D = 0.0517`` and ``p = 0.3867`` unchanged.
    * ``JAX_ENABLE_X64=1``, which replaces the whole RNG stream as well as the
      arithmetic, still PASSES here: D = 0.0367, p = 0.8006, coverage 0.890.

    **Measured on this checkout** (``PYTHONPATH=. .venv/bin/python
    docs/probes/probe_29_amortized_candidates.py 300 1``): APPLICABLE x PASS,
    KS D = 0.0517, p = 0.3867, 90% coverage 0.910, 300 of 300 usable.  Over
    harness seeds ``key(11)`` .. ``key(20)`` all ten PASS, worst p = 0.1008,
    median 0.4071.

    **Expected false positives: 0.05**, by construction, because that is what
    ALPHA declares -- and for this arm that sentence is supported.
    """
    _ranks, report, sampler = _exact_posterior_arm()
    _assert_the_sampler_arm_ran_over_every_replicate(report, sampler)
    assert report.conclusion is Conclusion.PASS, finding(
        report, "sbc_rank_uniformity"
    ).message


@pytest.mark.full
@pytest.mark.slow
def test_the_reference_npe_goes_through_the_sampler_arm_over_every_replicate():
    """R3 Task 9.1, the unconditional half: the contract, and a gross-error floor.

    ``NeuralPosterior`` on ``tests/test_amortize.py``'s linear-Gaussian
    problem -- bank 2048, 1500 Adam steps at batch 256, lr 1e-3, 10% held out,
    seeds ``key(0)`` / ``key(1)`` / ``key(2)`` -- put through
    :func:`~bayesmith.evaluation.sbc.sbc_ranks`' sampler arm at ``key(11)``,
    N = 300 replicates, 200 draws each, judged at ``ALPHA / K = 0.05``.  Every
    one of those is probe_29's, imported rather than restated.

    Nothing asserted here depends on WHICH trained estimator came out of that
    recipe, which is the whole reason it is separate from
    :func:`test_the_reference_npe_reproduces_its_recorded_calibration`.  Two
    claims:

    * the harness drove the sampler arm over all 300 replicates and refused,
      lost or failed to draw none of them -- and the arm is identified by the
      call count on the sampler OBJECT as well as by the census's route
      string, so a rename cannot walk past it;
    * the estimator is not grossly wrong: its width against the closed form
      is inside :data:`WIDTH_BAND`, whose margin against a re-run of this
      recipe is measured on that constant.

    **What this cell catches.**  ``truth[index]`` -> ``truth[0]`` in
    ``sbc._accumulate`` (every replicate ranked against the first one's truth)
    and ``route = "sampler"`` -> ``route = "npe"`` in ``sbc_ranks`` both turn
    it red, as does ``sampler_draws`` -> ``sampler_draws // 2`` (which the
    census tuple and the KS verdict both survive: only the call record sees
    it), a ``NeuralPosterior`` whose scales are doubled (ratios 1.9321 /
    2.0575 / 1.9079), and one whose ``_mixture`` discards the datum -- which
    is ``sbc.py``'s own documented blind spot, PASSES the SBC verdict, and is
    caught here at ratios 15.1805.
    """
    _q, _history, _ranks, report, sampler = _reference_npe_arm()
    _assert_the_sampler_arm_ran_over_every_replicate(report, sampler)

    probe = probe_29()
    ratios = [
        probe.width_against_exact(_q, theta_true)[0]
        for theta_true in probe.WIDTH_OBSERVATIONS
    ]
    low, high = WIDTH_BAND
    assert all(low < ratio < high for ratio in ratios), ratios


@pytest.mark.full
@pytest.mark.slow
def test_the_reference_npe_reproduces_its_recorded_calibration():
    """R3 Task 9.1, the recorded half: probe_29 §2's numbers, behind a witness.

    **This is CLAUDE.md rung (c).**  The premise -- that the 1500-step float32
    Adam run this fixture describes came out the way it came out here on
    2026-09-04 -- is platform-dependent and is therefore conditional and
    recorded.  The contract assertions are unconditional and live in
    :func:`test_the_reference_npe_goes_through_the_sampler_arm_over_every_replicate`,
    ahead of this cell.  Where the premise does not hold this cell SKIPS
    saying ``THIS IS NOT A PASS`` and naming the two numbers it measured,
    rather than passing quietly (rung (d)).

    **Why the premise is not a property.**  Re-running the identical recipe at
    24 other init/train seed pairs on this machine -- same bank, same harness
    key, same budget -- gave 6 KS FAILs of 24 (p = 0.0025 / 0.0207 / 0.0016 /
    0.0037 / 0.0046 / 0.0037), ``best_step`` from 292 to 1190, and
    ``|bias|`` up to 0.5468 exact sds.  A quarter of the re-runs of its own
    recipe fail this arm's KS test; ALPHA is 0.05.  Under
    ``JAX_ENABLE_X64=1``, which is not a pure BLAS swap -- it changes the RNG
    stream as well -- ``best_step`` moves 322 -> 580 and the three width
    ratios move to 1.0255 / 0.9612 / 1.1103 and the biases to +0.0753 /
    -0.0816 / +0.0216.  Those movements are 8-9 times the 0.0170 / 0.0232
    Monte-Carlo spread of the same statistics over twelve draw keys at fixed
    weights, so they are the WEIGHTS moving, not the sampling.

    **Measured on this checkout** (``PYTHONPATH=. .venv/bin/python
    docs/probes/probe_29_amortized_candidates.py 300 2``): ``best_step`` 322,
    validation minimum -0.5286270976, APPLICABLE x PASS, KS D = 0.0683,
    p = 0.1159, 90% coverage 0.890, 300 of 300 usable, width ratios 0.9660 /
    1.0287 / 0.9539 and mean biases +0.2605 / -0.0213 / +0.0054 exact sds at
    ``theta_true`` = +0.5 / +1.6 / -0.9.

    **Expected false positives: 0.05** wherever the witness holds, which is
    what ALPHA declares; §9.3 asks that the number be declared rather than
    discovered.  Note what the seed sweep above says about that number: it
    bounds the risk of THIS trajectory, not of the recipe.

    **There is deliberately no assertion on the KS digits.**  At N = 300 a
    PASS already means ``D <= 0.077832``
    (``scipy.stats.kstwo.ppf(0.95, 300)``), so a band on D is implied by the
    PASS and can never fail on its own.  The same is true of the coverage:
    sweeping a width distortion at this seed, the three-sigma binomial band
    ``0.90 +- 3 * sqrt(0.9 * 0.1 / 300) = [0.848, 0.952]`` is crossed only
    where the KS verdict has already failed (1.3x gives p = 0.0046 and
    coverage 0.9567; 1.2x gives p = 0.0558 and coverage 0.9400, inside the
    band).

    **What each pin catches, and what killed it.**

    * the PASS -- an estimator whose stated uncertainty stops matching its
      prior.  Killed by ``scales[component]`` -> ``2.0 * scales[component]``
      in ``NeuralPosterior.sample`` (D = 0.1967, p = 1.2e-10).
    * the width pin.  Killed by ``1.2 * scales[component]`` -- ratios
      1.1592 / 1.2345 / 1.1447 -- which the PASS above SURVIVES, at a measured
      p = 0.05576.  That mutant is why a width instrument exists at all.
    * the bias pin.  Killed by ``means[component]`` ->
      ``means[component] + 0.01`` -- a constant shift in STANDARDIZED latent
      space, 0.147 exact sds, which moves the biases to +0.4079 / +0.1261 /
      +0.1528 while leaving the width ratios untouched and IMPROVING the KS
      statistic to D = 0.0533, p = 0.3486.  Neither the verdict nor the width
      can see it.  It cannot be caught unconditionally either: 6 of the 72
      bias cells in the 24-retrain sweep exceed 0.40 and one reaches 0.5468,
      so no band on ``|bias|`` both survives the recipe and catches a 0.147
      shift.  It is caught HERE, where the trajectory is pinned.
    """
    probe = probe_29()
    q, history, _ranks, report, _sampler = _reference_npe_arm()

    best_step = int(history.best_step)
    validation_min = float(np.asarray(history.validation).min())
    reproduced = best_step == RECORDED_BEST_STEP and validation_min == pytest.approx(
        RECORDED_VALIDATION_MIN, rel=TRAJECTORY_RTOL
    )
    if not reproduced:
        pytest.skip(
            "THIS IS NOT A PASS. probe_29 §2's recorded trajectory did not "
            f"reproduce here: best_step={best_step} (recorded "
            f"{RECORDED_BEST_STEP}), validation minimum={validation_min!r} "
            f"(recorded {RECORDED_VALIDATION_MIN} at rel={TRAJECTORY_RTOL}). "
            "The numbers below describe one 1500-step float32 Adam run and "
            "are not asserted against a different one; re-run probe_29 §2 on "
            "this machine and record what it gives."
        )

    assert report.conclusion is Conclusion.PASS, finding(
        report, "sbc_rank_uniformity"
    ).message

    measured = [
        probe.width_against_exact(q, theta_true)
        for theta_true in probe.WIDTH_OBSERVATIONS
    ]
    ratios = [pair[0] for pair in measured]
    biases = [pair[1] for pair in measured]
    assert ratios == pytest.approx(
        list(RECORDED_WIDTH_RATIOS), abs=WIDTH_PIN_TOLERANCE
    ), ratios
    assert biases == pytest.approx(
        list(RECORDED_MEAN_BIASES), abs=BIAS_PIN_TOLERANCE
    ), biases
