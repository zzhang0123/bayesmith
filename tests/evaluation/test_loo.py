"""PSIS-LOO through the optional ArviZ bridge (§0.9).

Three things are being held here, and only one of them is a number.

**The chain axis is a measured premise, not a style choice.** probe_28 §2
measured that ``az.loo`` on an export whose only sample axis is ``draw``
raises ``AttributeError: 'DataArray' object has no attribute 'chain'``. R2's
export is right to leave an iid result flat -- ``DrawsPosterior.chain_shape is
None`` means "no chain structure to diagnose", and inventing one there would
put a diagnosable chain into a result that never ran one. So the ``(1, n)``
axis is supplied HERE, by the consumer that needs it, and
:func:`test_a_flat_export_is_what_arviz_refuses` re-measures the premise: if a
later arviz stops needing the axis, that test says so rather than the
workaround silently outliving its reason.

**The verdict is arviz's, not ours.** §0.9: a threshold has one owner, so a
warning from arviz's own ``good_k`` rule becomes ABSTAIN -- the ESTIMATE is
unreliable, which is not the same claim as "the model is wrong" -- and this
package does not put a second Pareto-k cutoff beside it. There is no D-number
in this module for exactly that reason.

**WAIC is absent upstream and stays absent here.** ``hasattr(arviz, "waic")``
is False in 1.3.0, and §1.5 puts WAIC among the mature statistics this package
reuses rather than reimplements. :func:`test_the_upstream_still_has_no_top_level_waic`
is what turns "we checked once" into "we would be told".

**The one-owner claim is held by a STUB, because a fixture cannot hold it.**
§0.9 says the Pareto-k rule is arviz's and this package adds none of its own.
On every real fixture the two rules agree -- the healthy run sits at k=0.55
and the constructed one at k=0.95, either side of arviz's ``good_k`` of 0.697
-- so a locally-owned ``max_pareto_k > 0.7`` cutoff would route both exactly
as arviz does and no measurement could tell them apart.
:class:`TestTheParetoKRuleBelongsToArviZAlone` pulls them apart by stubbing
``az.loo`` to return the two combinations arviz never produces: warned at
k=0.10, and silent at k=0.95.

**A weighted sample is declined rather than averaged over.** R2's export
carries draws but no ``log_weights``, so ``arviz.loo`` on a
``WeightedDrawsPosterior`` cross-validates the proposal, and returns the same
numbers for a healthy importance sample and a collapsed one.
:class:`TestAWeightedSampleIsMoreThanThisExportCanCarry` holds §0.2's answer
to that: UNVERIFIABLE, not PASS. Those tests need no arviz, because the branch
runs before the import.

**Skipped at RUN time, not at collection.** A module-level
``pytest.importorskip`` would drop these tests from collection where arviz is
absent, which is the defect the R2 close-out recorded: a shrunken collection
read as a shrunken suite and broke ``tests/test_readme_count.py`` in the wheel
environment. The class-level ``skipif`` below keeps them collected.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import jax
import numpy as np
import pytest

from bayesmith.artifacts.base import (
    ArtifactKind,
    ArtifactRef,
    ComputeBudget,
    NamedArray,
)
from bayesmith.artifacts.refusal import Refusal
from bayesmith.artifacts.reports import Applicability, Conclusion, EvaluationReport
from bayesmith.artifacts.results import LogDensityAvailability
from bayesmith.artifacts.tasks import PosteriorTask, PredictiveTask, new_task_meta
from bayesmith.dispatch.task import compile_task, execute_task
from bayesmith.evaluation.loo import REPORT_KIND, loo_report
from tests.dispatch.test_task_protocol import model_ref
from tests.exact.models import bilinear_pair, radiometer, straight_line

try:
    import arviz as _arviz
except ImportError:  # the wheel venv omits the dev-only arviz extra
    _arviz = None

requires_arviz = pytest.mark.skipif(_arviz is None, reason="arviz is not installed")


# --------------------------------------------------------------- fixtures


def planned_for(graph, task):
    planned = compile_task(graph, task, model_ref=model_ref())
    assert not isinstance(planned, Refusal), planned
    return planned


def posterior_of(graph, *, key, draws, warmup, chains=1):
    task = PosteriorTask(
        meta=new_task_meta(label="loo"),
        budget=ComputeBudget(draws=draws, warmup=warmup, chains=chains),
        nuts_on_collapse=False,
    )
    result = execute_task(planned_for(graph, task), key=key)
    assert not isinstance(result, Refusal), result
    return result


def predictive_of(graph, posterior, *, key, latent_sites=("w",)):
    task = PredictiveTask(
        meta=new_task_meta(label="loo-ppc"),
        source_posterior_ref=ArtifactRef(
            artifact_id=posterior.meta.artifact_id,
            revision=posterior.meta.revision,
            artifact_type=ArtifactKind.RESULT,
        ),
        conditioned_sites=("d",),
        replicated_sites=("d",),
        latent_sites=latent_sites,
    )
    result = execute_task(planned_for(graph, task), key=key, source_posterior=posterior)
    assert not isinstance(result, Refusal), result
    return result


def iid_predictive():
    """probe_28 §2's first fixture, at its seeds and its budget.

    2000 gcr draws on ``straight_line``, keys 2 and 3 -- the same run the plan's
    §0.12 table quotes, so the numbers pinned below are that measurement rather
    than a fresh one that happens to look similar.
    """
    graph = straight_line()
    posterior = posterior_of(graph, key=jax.random.key(2), draws=2000, warmup=1000)
    predictive = predictive_of(graph, posterior, key=jax.random.key(3))
    return graph, predictive


def cheap_predictive():
    """A real result at a tenth the cost, for the tests that stub ``az.loo``.

    64 draws rather than :func:`iid_predictive`'s 2000: every number the
    stubbing tests read comes back from the stub, so the run's job here is
    only to be a genuine result the export accepts.
    """
    graph = straight_line()
    posterior = posterior_of(graph, key=jax.random.key(2), draws=64, warmup=8)
    return graph, predictive_of(graph, posterior, key=jax.random.key(3))


def weighted_posterior():
    """``radiometer`` -- the plan's own weighted fixture (§0.12, G1).

    ``gcr+snis``: a GCR proposal corrected by self-normalised importance
    weights, so the representation is a ``WeightedDrawsPosterior`` carrying a
    ``log_weights`` array that R2's ArviZ export does not project.
    """
    graph = radiometer()
    return graph, posterior_of(graph, key=jax.random.key(2), draws=64, warmup=8)


@dataclasses.dataclass(frozen=True)
class StubElpd:
    """Everything ``loo_report`` reads off ``az.loo``'s return, and nothing more.

    Every field is a value no real run of these fixtures produces, which is
    what makes the stub load-bearing: an implementation that recomputed any of
    them, or that hardcoded arviz's ``good_k`` or its ``scale``, would return
    its own number here instead of this one.
    """

    warning: bool
    pareto_k: tuple[float, ...]
    elpd: float = -2.5
    se: float = 0.75
    p: float = 1.25
    n_data_points: int = 3
    n_samples: int = 7
    good_k: float = 0.5
    scale: str = "stub-scale"


def measurements(report: EvaluationReport) -> dict:
    """The estimate finding's ``observed`` pairs, as a mapping."""
    for finding in report.findings:
        if finding.code == "loo_psis_estimate":
            return dict(finding.observed)
    raise AssertionError(f"no estimate finding in {[f.code for f in report.findings]}")


# ------------------------------------------------------- the two pin bands
#
# Two constants, because the two fixtures below are two different experiments
# and one band over both would be either too loose for the first or a fixture
# pinned to one machine's arithmetic for the second -- the shape that spent
# four release tags in this repository.
#
# Both were chosen AFTER measuring on two platforms: macOS/Accelerate and
# linux/amd64 scipy-openblas under OPENBLAS_CORETYPE=ZEN (CLAUDE.md's container
# recipe). What the two platforms differ by, and what a CHANGED SEAM moves the
# same quantity by, are in the commit message; the two numbers below sit
# between them.

#: The ``gcr`` fixture: exact linear algebra at a fixed key, so the run is
#: reproducible and only the arithmetic differs.
#:
#: FORM derived, CONSTANT measured. Above: the largest cross-platform
#: disagreement over the four pinned quantities was 5.9e-8 (max Pareto k;
#: elpd itself moved 2.0e-9). Below: the smallest seam change that can be
#: constructed here -- folding the 2000 iid draws as ``(2, 1000)`` instead of
#: ``(1, 2000)``, which changes arviz's relative-efficiency correction --
#: moves elpd by 4.78e-5 on macOS and 4.73e-5 on Linux. 1e-6 is 17x above the
#: first and 47x below the second.
IID_PIN_ATOL = 1e-6

#: The NUTS fixture: a sampled trajectory, and deliberately three orders
#: looser than the band above.
#:
#: FORM derived, CONSTANT measured, and the gap between them is the point. The
#: two platforms differed by 9.6e-7 on elpd here -- twenty times the gcr
#: fixture's worst -- because a leapfrog trajectory amplifies an arithmetic
#: difference that exact elimination does not. A 1e-6 band would have passed
#: on both machines measured with 4% to spare, and ``ubuntu-latest`` drew three
#: different CPUs across five runs of this repository's own workflow; a
#: U-turn decision that flips on one of them moves this value by far more than
#: any band worth having. So this pin claims only what a pin can claim about a
#: sampled run: the right ARRAY was cross-validated.
#:
#: What 1e-3 can and cannot see was then measured rather than assumed. Applying
#: the mutation "ignore the recorded chain shape" and running this file: the
#: elpd, se and p_loo pins all survive it (the fold moves them by 9.0e-5), and
#: the max-Pareto-k pin kills it (3.2e-2). So one of the four does hold the
#: chain axis -- but only by 32x, on a sampled quantity, which is a margin to
#: report and not one to rely on. The axis is held INDEPENDENTLY by
#: :func:`test_the_recorded_chain_shape_is_what_reached_arviz`, which compares
#: two calls on ONE machine and needs no band at all.
NUTS_PIN_ATOL = 1e-3

#: The constructed Pareto(1) column's k. Measured BITWISE EQUAL on both
#: platforms (0.9491763949885138), which is what an exact inverse-CDF
#: construction should give; the band is a floor under a comparison of doubles
#: rather than an absorbed difference, because none was observed.
CONSTRUCTED_K_ATOL = 1e-6


@requires_arviz
class TestAnIidResultGetsTheChainAxisArviZRequires:
    def test_the_report_carries_probe_28s_loo_and_passes(self):
        graph, predictive = iid_predictive()

        report = loo_report(predictive, graph=graph)

        assert report.report_kind == REPORT_KIND
        assert report.applicability is Applicability.APPLICABLE
        assert report.conclusion is Conclusion.PASS
        found = measurements(report)
        assert found["elpd"] == pytest.approx(-6.913729558630095, abs=IID_PIN_ATOL)
        assert found["se"] == pytest.approx(1.937490864323723, abs=IID_PIN_ATOL)
        assert found["p_loo"] == pytest.approx(0.6659671467972856, abs=IID_PIN_ATOL)
        assert found["max_pareto_k"] == pytest.approx(
            0.5525405453952401, abs=IID_PIN_ATOL
        )

    def test_the_report_points_at_the_result_it_judged(self):
        graph, predictive = iid_predictive()

        report = loo_report(predictive, graph=graph)

        assert report.subject_ref.artifact_id == predictive.meta.artifact_id
        assert report.subject_ref.revision == predictive.meta.revision
        assert report.subject_ref.artifact_type is ArtifactKind.RESULT
        assert report.subject_ref in report.meta.parent_refs

    def test_the_observation_unit_survives_into_n_data_points(self):
        """G7: the plate axis the export preserved is what arviz counted.

        ``straight_line`` observes eight points over one flat node, and the
        2000 draws became 1 chain x 2000. If either axis were folded into the
        other, one of these two counts would be wrong while the elpd still
        looked plausible.
        """
        graph, predictive = iid_predictive()

        found = measurements(loo_report(predictive, graph=graph))

        assert found["n_data_points"] == 8
        assert found["n_samples"] == 2000

    def test_a_flat_export_is_what_arviz_refuses(self):
        """The measured premise the ``(1, n)`` supply exists for (§0.9).

        Not a test of this module: a test of the reason this module has a line
        in it. R2's export leaves an iid result on one ``draw`` axis, and
        that is the export ``az.loo`` cannot read.
        """
        import arviz as az

        from bayesmith.bridge.arviz import to_inference_data

        graph, predictive = iid_predictive()
        flat = to_inference_data(predictive, graph=graph)
        assert flat["log_likelihood"]["log_likelihood"].dims == ("draw", "d_dim0")

        with pytest.raises(AttributeError, match="chain"):
            az.loo(flat)

    def test_the_upstream_still_has_no_top_level_waic(self):
        """§0.9's recorded absence, in the form that would tell us it changed.

        arviz 1.3.0 has no top-level ``waic``, so R3's "LOO/WAIC" lands as
        LOO-PSIS and the gap is recorded rather than filled by a local
        reimplementation (§1.5). If a later arviz grows one, this is where
        the decision to consume it gets made.
        """
        import arviz as az

        assert not hasattr(az, "waic"), (
            f"arviz {az.__version__} now exposes waic; §0.9's recorded absence "
            "is out of date, and the decision to bridge it belongs in the plan"
        )


@requires_arviz
class TestAChainedPosteriorKeepsItsOwnChainShape:
    def test_a_nuts_posterior_is_read_at_its_measured_chain_shape(self):
        """probe_28 §2's second fixture: bilinear_pair, NUTS, (2, 400).

        The source already knows it ran two chains of 400, so nothing is
        supplied here -- and the counts arviz reports back are what say the
        (2, 400) shape reached it rather than a flattened 800.
        """
        graph = bilinear_pair()
        posterior = posterior_of(
            graph, key=jax.random.key(4), draws=400, warmup=400, chains=2
        )
        assert posterior.representation.chain_shape == (2, 400)

        report = loo_report(posterior, graph=graph)

        assert report.applicability is Applicability.APPLICABLE
        assert report.conclusion is Conclusion.PASS
        found = measurements(report)
        assert found["n_samples"] == 800
        assert found["n_data_points"] == 10
        assert found["elpd"] == pytest.approx(-1.3121380017602773, abs=NUTS_PIN_ATOL)
        assert found["se"] == pytest.approx(0.8345653474153597, abs=NUTS_PIN_ATOL)
        assert found["p_loo"] == pytest.approx(0.6786355183923574, abs=NUTS_PIN_ATOL)
        assert found["max_pareto_k"] == pytest.approx(
            0.4263811057858868, abs=NUTS_PIN_ATOL
        )

    def test_the_recorded_chain_shape_is_what_reached_arviz(self):
        """The (2, 400) the source recorded, held without a tolerance.

        ``n_samples`` cannot see this: 2 x 400 and 1 x 800 are both 800 draws.
        What separates them is arviz's relative-efficiency correction, and the
        way to see it without pinning a sampled number is to ask for both on
        the SAME machine and require them to differ. A ``loo_report`` that
        ignored the recorded shape and always exported ``(1, n)`` would make
        these two calls identical, and this is the only test here that would
        notice.

        Measured on macOS: flattening moves max Pareto k by 3.2e-2 and elpd by
        9.0e-5. The assertion asks only that the difference is not zero, so it
        is a comparison of one machine with itself and no band is involved.
        """
        graph = bilinear_pair()
        posterior = posterior_of(
            graph, key=jax.random.key(4), draws=400, warmup=400, chains=2
        )

        recorded = measurements(loo_report(posterior, graph=graph))
        flattened = measurements(
            loo_report(posterior, graph=graph, chain_shape=(1, 800))
        )

        assert recorded["n_samples"] == flattened["n_samples"] == 800
        assert recorded["max_pareto_k"] != flattened["max_pareto_k"]
        assert recorded["elpd"] != flattened["elpd"]


@requires_arviz
class TestAnUnreliableEstimateAbstainsRatherThanFails:
    """§0.2's row for ``loo_psis``: a warning is about the ESTIMATE."""

    @staticmethod
    def pareto_tailed(predictive):
        """One observation whose importance ratios are EXACTLY Pareto(1).

        Constructed, not sampled. LOO's importance ratio for draw i at point j
        is ``exp(-loglik[i, j])``, so setting ``loglik[:, 0] = log(u_i)`` over
        the midpoint grid ``u_i = (i + 1/2)/n`` makes ``-loglik`` the exact
        inverse-CDF sample of Exp(1) and the ratios exactly Pareto with tail
        index 1 -- i.e. a shape parameter of 1, well above arviz's ``good_k``
        of 0.697 at 2000 draws, on every platform and with no RNG involved.
        The other seven columns are the run's own and stay under 0.553, so the
        warning has exactly one cause.
        """
        pointwise = predictive.pointwise_log_density
        draws = int(pointwise.value.shape[0])
        values = np.array(pointwise.value, dtype=float, copy=True)
        values[:, 0] = np.log((np.arange(draws) + 0.5) / draws)
        return dataclasses.replace(
            predictive,
            pointwise_log_density=NamedArray(
                name=pointwise.name, dims=pointwise.dims, value=values
            ),
        )

    def test_a_heavy_tailed_point_abstains_and_says_which_k(self):
        graph, predictive = iid_predictive()

        report = loo_report(self.pareto_tailed(predictive), graph=graph)

        assert report.applicability is Applicability.APPLICABLE
        assert report.conclusion is Conclusion.ABSTAIN, (
            "an unreliable PSIS estimate is not a failed model check"
        )
        found = measurements(report)
        assert found["max_pareto_k"] > found["good_k"]
        assert found["max_pareto_k"] == pytest.approx(
            0.9491763949885138, abs=CONSTRUCTED_K_ATOL
        )
        # good_k is a function of the draw count alone (2000 draws), so it is
        # the same double on both platforms and carries no fixture noise.
        assert found["good_k"] == pytest.approx(0.6970642492453765, abs=1e-12)

    def test_the_same_run_passes_when_the_tail_is_left_alone(self):
        """The control the fixture above needs: only the constructed column
        moved the verdict, not the budget or the seed."""
        graph, predictive = iid_predictive()

        assert loo_report(predictive, graph=graph).conclusion is Conclusion.PASS


@requires_arviz
class TestTheParetoKRuleBelongsToArviZAlone:
    """§0.9's central claim, in the only form that can fail.

    The module docstring, the commit message and the comments all say this
    package holds no Pareto-k cutoff of its own. Every OTHER test here is
    compatible with one: replace ``warned = bool(elpd.warning)`` with
    ``max_pareto_k > 0.7`` and all of them stay green, because the healthy
    fixture (k=0.55) and the constructed one (k=0.95) fall the same side of
    0.7 as they do of arviz's ``good_k``.

    So the two combinations arviz will not produce are manufactured here:
    warned at k=0.10, and silent at k=0.95. A verdict read off ``warning``
    answers ABSTAIN then PASS; a verdict read off a local cutoff answers PASS
    then ABSTAIN. Both cells, not one -- a single direction is satisfied by
    ``warned = True``.
    """

    @pytest.mark.parametrize(
        ("warning", "k", "expected"),
        [
            (True, 0.10, Conclusion.ABSTAIN),
            (False, 0.95, Conclusion.PASS),
        ],
    )
    def test_the_conclusion_follows_arvizs_warning_and_not_the_k(
        self, monkeypatch, warning, k, expected
    ):
        import arviz as az

        graph, predictive = cheap_predictive()
        monkeypatch.setattr(
            az, "loo", lambda idata: StubElpd(warning=warning, pareto_k=(k,))
        )

        report = loo_report(predictive, graph=graph)

        assert report.applicability is Applicability.APPLICABLE
        assert report.conclusion is expected
        found = measurements(report)
        # No band: the value went through float() unchanged, and a tolerance
        # here would be a tolerance on a number nothing computed.
        assert found["max_pareto_k"] == k
        assert found["good_k"] == 0.5

    def test_every_number_in_the_estimate_finding_is_the_one_arviz_returned(
        self, monkeypatch
    ):
        """The other way to plant a rule here: stop asking arviz.

        ``good_k`` hardcoded as arviz 1.3.0's 0.697, or ``scale`` as the
        literal "log", passes every fixture in this file -- the stub is what
        separates "read from the upstream" from "agrees with the upstream
        today". Exact equality throughout, because none of these values is
        computed.
        """
        import arviz as az

        graph, predictive = cheap_predictive()
        stub = StubElpd(warning=False, pareto_k=(0.95,))
        monkeypatch.setattr(az, "loo", lambda idata: stub)

        found = measurements(loo_report(predictive, graph=graph))

        assert found == {
            "elpd": stub.elpd,
            "se": stub.se,
            "p_loo": stub.p,
            "n_data_points": stub.n_data_points,
            "n_samples": stub.n_samples,
            "max_pareto_k": 0.95,
            "good_k": stub.good_k,
            "scale": stub.scale,
        }

    def test_an_absent_or_unfittable_number_is_recorded_as_absent(self, monkeypatch):
        """``None`` and NaN are both "no measurement", and NaN is the worse one.

        A NaN Pareto k would compare unequal to itself in every consumer that
        later read it back, so :func:`bayesmith.evaluation.loo._finite` maps it
        to None. arviz 1.3.0 hands these fixtures a finite k every time, so
        neither branch is reachable from a real run and the stub is the only
        way to exercise them.
        """
        import arviz as az

        graph, predictive = cheap_predictive()
        monkeypatch.setattr(
            az,
            "loo",
            lambda idata: StubElpd(warning=False, pareto_k=None, good_k=float("nan")),
        )

        found = measurements(loo_report(predictive, graph=graph))

        assert found["max_pareto_k"] is None
        assert found["good_k"] is None


@requires_arviz
class TestTheVerdictCanBeRecomputedFromTheFindings:
    def test_the_reliability_finding_is_what_decided_the_conclusion(self):
        """G8: no consumer has to read a sampler log to know why.

        ``observed`` is arviz's ``warning`` and ``expected`` is False, so the
        conclusion is a function of two fields the report carries.
        """
        graph, predictive = iid_predictive()
        heavy = TestAnUnreliableEstimateAbstainsRatherThanFails.pareto_tailed(
            predictive
        )

        for result, expected in (
            (predictive, Conclusion.PASS),
            (heavy, Conclusion.ABSTAIN),
        ):
            report = loo_report(result, graph=graph)
            (finding,) = [f for f in report.findings if f.code == "psis_reliability"]
            assert finding.expected is False
            recomputed = (
                Conclusion.PASS
                if finding.observed == finding.expected
                else Conclusion.ABSTAIN
            )
            assert recomputed is report.conclusion is expected

    def test_the_report_names_the_arviz_that_produced_it(self):
        import arviz as az

        graph, predictive = iid_predictive()

        report = loo_report(predictive, graph=graph)

        (finding,) = [f for f in report.findings if f.code == "arviz_version"]
        assert finding.observed == az.__version__


class TestAWeightedSampleIsMoreThanThisExportCanCarry:
    """§0.2 again: a check that cannot apply says so rather than passes.

    ``radiometer`` is routed ``gcr+snis``, so its representation is a
    ``WeightedDrawsPosterior``: draws, plus a ``log_weights`` array over the
    same draw axis. R2's export projects draws, replicated draws, pointwise
    log density and observed data -- and no weights; a grep for "weight" in
    ``src/bayesmith/bridge/arviz.py`` finds nothing. ``arviz.loo`` given that
    export therefore cross-validates the PROPOSAL.

    Measured before this branch existed, on a 2000-draw ``radiometer`` run:
    APPLICABLE / PASS with elpd -4.25675783332378, and the SAME report --
    equal in every field -- after collapsing ``log_weights`` so the weight ESS
    is 1.000 of 2000 rather than 1997.3. A verdict that cannot tell those two
    apart is not a verdict about either. The fixture below runs 64 draws,
    because after this branch nothing downstream of the weights is reached and
    a longer run would measure only itself.

    These tests are not skipped without arviz: the branch precedes the import,
    which is the point of
    :func:`test_the_answer_does_not_depend_on_whether_arviz_is_installed`.
    """

    def test_a_weighted_posterior_is_unverifiable_rather_than_a_pass(self):
        graph, posterior = weighted_posterior()
        assert type(posterior.representation).__name__ == "WeightedDrawsPosterior"

        report = loo_report(posterior, graph=graph)

        assert report.applicability is Applicability.UNVERIFIABLE
        assert report.conclusion is Conclusion.ABSTAIN
        assert [f.code for f in report.findings] == [
            "weights_not_carried_by_export",
            "weighted_sample",
        ]

    def test_the_finding_carries_the_pair_the_verdict_was_read_from(self):
        """G8: recomputable from the report, no sampler log involved.

        The three ``recorded_*`` pairs are the run's OWN diagnostics, copied
        into a report that declined to compute new ones, so each is checked
        against the representation it was copied from rather than against a
        number written here. That is what makes them guards: a report field
        that says something the sample's own record does not say is exactly
        the defect this branch exists to prevent, and an untested copy can
        drift to any value -- including one that reads healthy for a sample
        the run itself marked unreliable.

        Reading the representation is not a style preference here, it is the
        only version that survives a second machine. This fixture's ``khat``
        is 0.5209304017354495 on macOS/Accelerate and 0.5209322414779554 on
        linux/amd64 scipy-openblas under ``OPENBLAS_CORETYPE=ZEN`` -- a
        relative difference of 3.5e-6, which is 3.5x ``pytest.approx``'s
        default relative tolerance of 1e-6. So the same assertion written
        against either machine's LITERAL would be red on the other. Written
        against the representation, both sides are one in-process float and
        the platform moves them together; ``ess`` happens to agree bitwise on
        both, and is checked the same way for the same reason.
        """
        graph, posterior = weighted_posterior()

        report = loo_report(posterior, graph=graph)

        (loss,) = [
            f for f in report.findings if f.code == "weights_not_carried_by_export"
        ]
        assert loss.observed is False
        assert loss.expected is True
        (described,) = [f for f in report.findings if f.code == "weighted_sample"]
        described = dict(described.observed)
        assert described["representation"] == "WeightedDrawsPosterior"
        assert described["method"] == "gcr+snis"
        assert described["draws"] == 64
        recorded = posterior.representation
        assert described["recorded_unreliable"] is recorded.unreliable
        assert described["recorded_khat"] == pytest.approx(recorded.khat)
        assert described["recorded_ess"] == pytest.approx(recorded.ess)

    def test_the_draw_count_is_read_off_the_weights_rather_than_assumed(self):
        """One fixture cannot tell a reader from a constant equal to it.

        The pin above says ``draws == 64`` and 64 is the fixture's own draw
        count, so it holds just as well for ``draws = 64`` written into
        ``loo.py``; measured, that mutant survives the whole file. Asserting
        against the fixture's array does not separate them either -- the array
        is 64 long, so both sides move to the same constant. A SECOND draw
        count is what separates them, and the final line is what keeps it a
        second one: set the two budgets equal and this says so rather than
        quietly going back to proving nothing.
        """
        short_graph = radiometer()
        short = posterior_of(short_graph, key=jax.random.key(2), draws=32, warmup=8)
        full_graph, full = weighted_posterior()

        seen = []
        for graph, posterior in ((short_graph, short), (full_graph, full)):
            report = loo_report(posterior, graph=graph)
            (described,) = [f for f in report.findings if f.code == "weighted_sample"]
            reported = dict(described.observed)["draws"]
            assert reported == int(posterior.representation.log_weights.value.shape[0])
            seen.append(reported)
        assert seen[0] != seen[1]

    def test_the_answer_does_not_depend_on_whether_arviz_is_installed(
        self, monkeypatch
    ):
        """Two UNVERIFIABLE rows, and they are not the same row.

        "the export cannot carry this sample" is a property of the artifact;
        "arviz is missing" is a property of the machine. Hiding a weighted
        sample behind ``arviz_unavailable`` in a clone would make the report
        say the wrong true thing, so the weighted branch runs first and this
        pins that order.
        """
        graph, posterior = weighted_posterior()
        monkeypatch.setitem(sys.modules, "arviz", None)

        report = loo_report(posterior, graph=graph)

        assert report.applicability is Applicability.UNVERIFIABLE
        codes = [f.code for f in report.findings]
        assert codes == ["weights_not_carried_by_export", "weighted_sample"]

    def test_a_predictive_result_is_judged_by_the_source_it_is_handed(self):
        """A ``PredictiveResult`` carries a REF to its source, not the source.

        So the weights are invisible to it, and the caller that still holds
        the posterior is the one that can say. Both directions here, because
        a branch that fired on everything would be as wrong as one that fired
        on nothing.
        """
        graph, posterior = weighted_posterior()
        predictive = predictive_of(graph, posterior, key=jax.random.key(3))

        told = loo_report(predictive, graph=graph, source_posterior=posterior)

        assert told.applicability is Applicability.UNVERIFIABLE
        assert told.conclusion is Conclusion.ABSTAIN

    @requires_arviz
    def test_an_unweighted_source_changes_nothing_at_all(self):
        """The control a one-sided branch always needs.

        Asserted as an EQUALITY between the two calls rather than as a
        particular verdict: what has to be true is that naming an unweighted
        source is indistinguishable from not naming one, and pinning the
        verdict instead would make this test fail the day the fixture's own
        Pareto k moved -- which it does at 64 draws, where arviz's ``good_k``
        is 0.45 and this run trips it.
        """
        graph = straight_line()
        posterior = posterior_of(graph, key=jax.random.key(2), draws=64, warmup=8)
        predictive = predictive_of(graph, posterior, key=jax.random.key(3))

        silent = loo_report(predictive, graph=graph)
        told = loo_report(predictive, graph=graph, source_posterior=posterior)

        assert told.applicability is silent.applicability
        assert told.conclusion is silent.conclusion
        assert [f.code for f in told.findings] == [f.code for f in silent.findings]
        assert measurements(told) == measurements(silent)
        assert told.applicability is Applicability.APPLICABLE


class TestWhatLooCannotJudge:
    """The two non-APPLICABLE rows of §0.2, and they are different answers."""

    def test_a_result_with_no_pointwise_likelihood_is_inapplicable(self):
        """INAPPLICABLE: LOO is not a check this result can be asked about.

        Nothing is missing from the environment -- the result simply does not
        carry the per-observation density LOO is computed from.
        """
        graph = straight_line()
        posterior = posterior_of(graph, key=jax.random.key(2), draws=64, warmup=8)
        predictive = predictive_of(graph, posterior, key=jax.random.key(3))
        without = dataclasses.replace(predictive, pointwise_log_density=None)

        report = loo_report(without, graph=graph)

        assert report.applicability is Applicability.INAPPLICABLE
        assert report.conclusion is Conclusion.ABSTAIN
        assert [f.code for f in report.findings] == ["no_pointwise_log_likelihood"]

    def test_a_posterior_with_no_pointwise_likelihood_is_inapplicable_too(self):
        """The other half of ``_pointwise``: a posterior names the same array
        ``pointwise_log_likelihood``, and its absence is the same answer.

        ``log_density_availability`` moves with it, because the result protocol
        holds those two in step -- POINTWISE exactly when the array is there.
        """
        graph = straight_line()
        posterior = posterior_of(graph, key=jax.random.key(2), draws=64, warmup=8)
        without = dataclasses.replace(
            posterior,
            pointwise_log_likelihood=None,
            log_density_availability=LogDensityAvailability.NONE,
        )

        report = loo_report(without, graph=graph)

        assert report.applicability is Applicability.INAPPLICABLE
        assert report.conclusion is Conclusion.ABSTAIN
        assert [f.code for f in report.findings] == ["no_pointwise_log_likelihood"]

    def test_a_clone_without_arviz_gets_a_report_rather_than_an_importerror(
        self, monkeypatch
    ):
        """UNVERIFIABLE: the check applies, the tool to run it is absent.

        §7.3's graceful degradation, exercised by making ``import arviz``
        raise the way it would in a clone that never installed the dev extra.
        This test is deliberately NOT skipped when arviz is missing -- it is
        then measuring the very environment it describes.
        """
        graph = straight_line()
        posterior = posterior_of(graph, key=jax.random.key(2), draws=64, warmup=8)
        predictive = predictive_of(graph, posterior, key=jax.random.key(3))
        monkeypatch.setitem(sys.modules, "arviz", None)

        report = loo_report(predictive, graph=graph)

        assert report.applicability is Applicability.UNVERIFIABLE
        assert report.conclusion is Conclusion.ABSTAIN
        assert [f.code for f in report.findings] == ["arviz_unavailable"]

    def test_something_that_is_not_a_result_is_a_typeerror_not_a_report(self):
        """A report says something about a result; there is nothing here to
        say it about, and a verdict on the wrong type is worse than a raise."""
        with pytest.raises(TypeError, match="posterior or a predictive result"):
            loo_report(object(), graph=straight_line())


class TestTheModuleStaysCheapWithoutArviZ:
    def test_importing_the_loo_module_does_not_import_arviz(self):
        """ArviZ is optional only while nothing imports it on the way in.

        In a subprocess: by the time this test runs, arviz is already in this
        process's ``sys.modules`` because the tests above imported it, so an
        in-process assertion would be a statement about the test runner.
        """
        code = (
            "import sys; import bayesmith.evaluation.loo as m; "
            "assert m.loo_report; "
            "print(sorted({'arviz', 'xarray'} & set(sys.modules)))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "[]"

    def test_the_package_re_export_is_as_cheap_as_the_module(self):
        code = (
            "import sys; from bayesmith.evaluation import loo_report; "
            "assert loo_report; "
            "print(sorted({'arviz', 'xarray'} & set(sys.modules)))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "[]"
