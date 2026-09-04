"""``model_checking@1``: what the runner files, and what the gate then says.

The subject here is an ORCHESTRATOR, so almost nothing below asserts a number.
Every threshold these eight reports rest on is owned somewhere else --
:data:`bayesmith.evaluation.ALPHA` and its draw floor in ``checks``, the
replicate floor in ``sbc``, ``DEFAULT_RANK_RTOL`` and ``CRITERION_SHIFT`` in
``diagnose`` -- and the tests that hold those live beside them.  What is tested
here is the wiring: which slot each report lands in, what happens to a check
that raises, and whether the aggregated answer can be recomputed from the
reports it aggregated.

**The measurement this file was written around.**  Before a line of
``gate.py`` existed, :func:`~bayesmith.artifacts.gates.aggregate_gate` was
probed on this checkout with one required PASS slot beside one optional slot::

    optional slot ABSENT from slots        evaluated pass n_refs=1 findings=[]
    optional slot NOT_ATTEMPTED            evaluated pass n_refs=1 findings=[]
    optional slot UNVERIFIABLE x ABSTAIN   evaluated pass n_refs=2 findings=[]
    optional slot APPLICABLE x ABSTAIN     evaluated pass n_refs=2 findings=[]
    optional slot APPLICABLE x PASS        evaluated pass n_refs=2 findings=[]
    optional slot APPLICABLE x FAIL        evaluated pass n_refs=2 ['optional_report_failed']

    required ABSENT                evaluated abstain n_refs=0 ['required_report_missing']
    required NOT_ATTEMPTED         evaluated abstain n_refs=0 ['required_report_missing']
    required APPLICABLE x ABSTAIN  evaluated abstain n_refs=1 ['required_report_abstained']
    required UNVERIFIABLE          evaluated abstain n_refs=1 ['required_report_unverifiable']

On the REQUIRED axis a missing report and an abstaining one are separate
findings.  On the OPTIONAL axis they are the same status, the same verdict and
the same (empty) findings, and ``report_refs`` is the ONLY field that separates
them.  ``TestTheOptionalSlotTrap`` re-runs that comparison through the real
runner, because a runner that dropped an abstaining optional report would emit
a byte-identical clean PASS for a run where a check was attempted and could not
answer.

**Seeds, budgets and the declared false-positive rate (§9.3).**  Two seeds and
one budget, both module constants: ``POSTERIOR_SEED = 1`` fits every fixture's
posterior and ``GATE_SEED = 101`` drives the predictive replay and the prior
simulation inside :func:`~bayesmith.evaluation.gate.check_posterior`.  Nothing
is re-drawn and no assertion below compares a p-value against a band, so the
tolerated false-positive count of this file is **zero** -- but some of the
cells these fixtures produce sit at or over their own band, and that is worth
writing down, because a fixture that is one platform's arithmetic away from
flipping is what burned four release tags here:

* the calibrated fixture's ``prior_predictive_check`` on ``residual_sd`` has a
  tail mass of **0.0355** against a band edge of ``ALPHA / 2 = 0.025``, and
  that is not seed luck: swept over gate seeds 101/202/303/404/505 it reads
  0.0355 / 0.0340 / 0.0340 / 0.0335 / 0.0385.
* the masked fixture's same cell is not merely CLOSE to that edge -- **at one
  of those five seeds it is already over it.**  Swept the same way it reads
  0.0270 / 0.0260 / **0.0250** / 0.0285 / 0.0285, and the third of those is
  ``0.02499999999999999`` against ``0.025``: a margin of
  ``-1.0408340855860843e-17``, which makes that optional report APPLICABLE x
  FAIL at gate seed 303 on macOS/Accelerate TODAY.  So a Linux run that FAILS
  this cell is **not a regression** and not a platform difference worth
  chasing: it is a sub-ULP coin flip that has already landed on both sides
  here.  Nothing asserts on it and nothing should start to -- the file runs at
  ``GATE_SEED = 101`` only, where the same cell reads 0.0270, and the fixture
  is left exactly as it is rather than nudged away from an edge it does not
  decide anything from.  (An earlier draft of this list gave 0.1045 as the
  seed-303 value.  That number is real: it is this same MASKED fixture's
  ``largest`` cell, ``0.10449999999999997`` -- a transposed row, not a
  transposed fixture, and quoting it here hid the one sub-ULP margin in the
  file.  A first correction said CALIBRATED, which is also wrong: that
  fixture's ``largest`` at seed 303 reads ``0.10499999999999997``.  Both
  numbers measured here on 2026-09-04.  The second error is the more
  instructive one -- it was copied from a review rather than re-measured, one
  line below a sentence about exactly that.)
* everything the PASS VERDICTS below actually rest on is far from its edge:
  the required ``posterior_predictive_check``'s worst cell is 0.3205
  (calibrated) and 0.4175 (masked), twelve and sixteen times the edge, and
  ``identifiability`` decides on an integer nullity with no tolerance at all.

So no test here asserts that a PASS carries NO findings -- an optional
``prior_predictive_check`` that flipped on another BLAS would add one without
touching the verdict, and §0.6 is explicit that only required checks decide.
The assertions are on the required slots and on the verdict, which is the part
that is robust.  Measured on this checkout (macOS/Accelerate); the numbers
above are quoted so a Linux run that moves them can be compared rather than
guessed at.

**x64.**  ``identifiability`` and ``prior_sensitivity`` refuse a float32
ambient precision, and they refuse a graph whose constants were traced outside
the block separately (§0.10), so every fixture builds its graph and runs the
gate inside one ``jax.enable_x64(True)``.  The module carries the ``x64``
marker so a reader knows why; the context manager is opened per fixture, so
the file also passes in a default-precision session.  The one test that needs
float32 opens ``jax.enable_x64(False)`` explicitly, which holds in both.
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import const, det, observe, sample, trace
from bayesmith.artifacts.base import (
    ArtifactRef,
    ArtifactStatus,
    ComputeBudget,
    ProducerRef,
    invalidate_meta,
    new_artifact_meta,
)
from bayesmith.artifacts.gates import (
    AttemptStatus,
    GateResult,
    OperationalStatus,
    ReportSlot,
    aggregate_gate,
)
from bayesmith.artifacts.identity import (
    ArtifactKind,
    FingerprintKind,
    InvalidationPolicy,
    fingerprint,
)
from bayesmith.artifacts.refusal import Finding, Refusal
from bayesmith.artifacts.reports import Applicability, Conclusion, EvaluationReport
from bayesmith.artifacts.tasks import PosteriorTask, new_task_meta
from bayesmith.dispatch.execute import chain_diagnostics
from bayesmith.dispatch.task import PRODUCER as DISPATCH_PRODUCER
from bayesmith.dispatch.task import (
    _chain_diagnostics_report,
    compile_task,
    execute_task,
)
from bayesmith.evaluation import ALPHA
from bayesmith.evaluation import check_posterior as reexported_check_posterior
from bayesmith.evaluation.checks import DRAW_FLOOR
from bayesmith.evaluation.gate import (
    CARRIED_KINDS,
    CHAIN_DIAGNOSTICS,
    MODEL_CHECKING,
    POSTERIOR_PREDICTIVE_CHECK,
    PRIOR_PREDICTIVE_CHECK,
    _file,
    check_posterior,
    model_checking_slots,
)
from bayesmith.evaluation.sbc import REPORT_KIND as SBC
from bayesmith.evaluation.sbc import simulation_based_calibration
from tests.dispatch.test_task_protocol import model_ref
from tests.exact.models import bilinear_pair, straight_line

pytestmark = pytest.mark.x64

A = Applicability
C = Conclusion
S = OperationalStatus

#: §9.3, spelled once: the seed every posterior is fit at, the seed every gate
#: run splits for its predictive replay and its prior simulation, and the draw
#: budget both use.  2000 is comfortably above D105's floor; the starved
#: fixture below is the one that is not, and it says so in its own name.
POSTERIOR_SEED = 1
GATE_SEED = 101
DRAWS = 2000

#: ``straight_line``'s own shape, restated here so that the misspecified and
#: masked fixtures differ from the calibrated one in exactly one thing each --
#: a quadratic term the linear model cannot express, and two withheld points.
SIGMA = 0.5
PRIOR_STD = 2.0

#: Two of eight positions withheld.  Two rather than one because the held-out
#: band is Bonferroni-corrected by the NUMBER of held-out points, so a single
#: point cannot tell a correction from its absence.
MASK = jnp.array([True, True, True, False, True, True, False, True])

#: Where ``TestItDecidesNothing`` places an adversarial cell, as MULTIPLES of
#: the one band edge every check in this layer uses.
#:
#: Not a tolerance and not compared against anything -- the ladder exists
#: because a gate that re-decided a verdict would have to re-decide it against
#: SOME number, and the only number available to copy is the edge these
#: reports were already judged against.  So the ladder brackets that edge:
#: nothing, half of it, just under, exactly on it, just over, and then out to
#: twenty times it, which is past where the real fixtures' own cells sit.  A
#: rewrite rule tuned to any neighbourhood of the edge fires on at least one
#: rung, and the assertions on every rung are identity, not arithmetic.
EDGE_MULTIPLES = (0.0, 0.5, 0.9, 1.0, 1.1, 2.0, 20.0)


def line_at(curvature=0.0, *, mask=None, n=8, weight=2.5, seed=0):
    """probe_28's ``curved_line``, with every constant traced by the CALLER.

    The probe's own fixture reads a module-scope ``X`` built at import time, so
    a graph built from it inside ``jax.enable_x64(True)`` still carries float32
    constants and ``identifiability`` refuses it -- measured here, and it turns
    the misspecified fixture's required ``identifiability`` slot from
    APPLICABLE into UNVERIFIABLE.  Same model, same ``curvature`` dial, same
    ``SIGMA`` and prior; the only difference is where the arrays are made.
    """
    x = jnp.linspace(1.0, 4.0, n)
    noise = SIGMA * jax.random.normal(jax.random.key(seed), x.shape)
    data = weight * x + curvature * x**2 + noise

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, PRIOR_STD))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, SIGMA), mu, obs=data, mask=mask)

    return trace(model)


def correlated_line(size=8, weight=2.0, decay=0.4):
    """A ``CirculantNormal`` observation: outside R2 §0.4's coverage domain.

    Measured on this checkout: the posterior fits, and its
    ``predictive_ready`` is False, because ``pointwise_log_likelihood`` has no
    diagonal loc/scale to replay.  That is the ONE fixture here whose gate is
    BLOCKED rather than evaluated.
    """
    lag = np.minimum(np.arange(size), size - np.arange(size))
    kernel = jnp.asarray(1.0 * decay**lag + 0.5)
    x = jnp.linspace(1.0, 4.0, size)

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 5.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe(
            "d",
            lambda m: dist.CirculantNormal(m, kernel),
            mu,
            depends_on_prediction=False,
            obs=weight * x,
        )

    return trace(model)


def posterior_of(graph, *, draws=DRAWS, seed=POSTERIOR_SEED, chains=1, nuts=False):
    task = PosteriorTask(
        meta=new_task_meta(label="t8-subject"),
        budget=ComputeBudget(
            draws=draws, warmup=min(400, draws), chains=chains
        ),
        nuts_on_collapse=nuts,
    )
    planned = compile_task(graph, task, model_ref=model_ref())
    assert not isinstance(planned, Refusal), planned
    result = execute_task(planned, key=jax.random.key(seed))
    assert not isinstance(result, Refusal), result
    return result


@dataclasses.dataclass(frozen=True)
class Checked:
    """One fixture's three views of one run.

    ``slots`` and ``gate`` come from two SEPARATE calls, and the separation is
    visible rather than hidden: every check mints a fresh artifact id, so the
    reports in ``slots`` are not the ones ``gate.report_refs`` names -- they
    are a second set with the same content.  ``aggregated`` is
    :func:`~bayesmith.artifacts.gates.aggregate_gate` over ``slots``
    themselves, and it is the only one of the three whose references can be
    matched to a report object in hand.  ``check_posterior``'s own aggregation
    is held to it by
    ``test_the_gate_aggregates_the_slots_the_runner_built``.
    """

    graph: object
    posterior: object
    slots: tuple[ReportSlot, ...]
    gate: GateResult
    aggregated: GateResult

    def slot(self, name: str) -> ReportSlot:
        for item in self.slots:
            if item.requirement.name == name:
                return item
        raise AssertionError(f"{name!r} not among {[s.requirement.name for s in self.slots]}")

    def report(self, name: str) -> EvaluationReport:
        report = self.slot(name).report
        assert report is not None, f"{name!r} produced no report"
        return report

    def codes(self) -> list[str]:
        return [finding.code for finding in self.gate.findings]


def run(build, *, draws=DRAWS, x64=True, carried=(), posterior=None):
    """Build the graph, fit it and run the gate -- all inside one x64 block.

    ``model_checking_slots`` and ``check_posterior`` are called separately on
    purpose: the first is what the assertions read, the second is the function
    under test, and running both proves they see the same eight slots rather
    than assuming it (``test_the_gate_aggregates_the_slots_the_runner_built``).
    """
    ambient = jax.enable_x64(True) if x64 else jax.enable_x64(False)
    with ambient:
        graph = build()
        subject = posterior_of(graph, draws=draws) if posterior is None else posterior
        budget = ComputeBudget(draws=draws)
        slots = model_checking_slots(
            graph,
            subject,
            key=jax.random.key(GATE_SEED),
            budget=budget,
            model_ref=model_ref(),
            carried=carried,
        )
        gate = check_posterior(
            graph,
            subject,
            key=jax.random.key(GATE_SEED),
            budget=budget,
            model_ref=model_ref(),
            carried=carried,
        )
        aggregated = aggregate_gate(
            MODEL_CHECKING,
            meta=new_artifact_meta(
                artifact_type=ArtifactKind.EVALUATION_REPORT,
                fingerprints=subject.run.fingerprints,
                producer=DISPATCH_PRODUCER,
                summary="the same slots, aggregated by the test",
            ),
            prerequisites_ready=subject.predictive_ready,
            inputs_current=(
                subject.meta.lifecycle.status is ArtifactStatus.CURRENT
            ),
            slots=slots,
        )
    return Checked(
        graph=graph,
        posterior=subject,
        slots=slots,
        gate=gate,
        aggregated=aggregated,
    )


# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def calibrated():
    """``straight_line`` at its own data: §8.1's EVALUATED x PASS path."""
    return run(straight_line)


@pytest.fixture(scope="module")
def misspecified():
    """``curved_line(0.6)``: a quadratic the linear model cannot express."""
    return run(lambda: line_at(0.6))


@pytest.fixture(scope="module")
def starved():
    """Eight draws: below D105's floor, so the required check ABSTAINs."""
    return run(straight_line, draws=8)


@pytest.fixture(scope="module")
def masked():
    """Two withheld points: the only fixture whose held-out check APPLIES."""
    return run(lambda: line_at(0.0, mask=MASK))


@pytest.fixture(scope="module")
def in_float32():
    """The calibrated model with the ambient precision forced to float32."""
    return run(straight_line, draws=64, x64=False)


@pytest.fixture(scope="module")
def unpredictable():
    """A correlated observation: ``predictive_ready`` is False."""
    return run(correlated_line, draws=200)


@pytest.fixture(scope="module")
def abstaining_sbc():
    """A REAL ``sbc`` report that abstains, for the carried channel.

    Four replicates, which is below :data:`REPLICATE_FLOOR` by a wide margin,
    so the verdict is ABSTAIN by budget and costs four small fits rather than
    the hundred a PASS would need.  Its ``subject_ref`` is the first usable
    replicate's posterior -- §0.2's routing representative -- which is exactly
    why ``gate.py`` checks no provenance for this kind.
    """
    with jax.enable_x64(True):
        data = 2.5 * jnp.linspace(1.0, 4.0, 8) + SIGMA * jax.random.normal(
            jax.random.key(0), (8,)
        )
        report = simulation_based_calibration(
            _line_with(data),
            key=jax.random.key(3),
            replicates=4,
            model_ref=model_ref(),
            build=lambda datum: _line_with(datum["d"]),
            budget=ComputeBudget(draws=64),
        )
    assert isinstance(report, EvaluationReport), report
    return report


def _line_with(data):
    """``straight_line``'s model at caller-supplied data -- the callable
    ``sbc_ranks`` re-traces once per replicate."""
    x = jnp.linspace(1.0, 4.0, 8)

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, PRIOR_STD))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, SIGMA), mu, obs=data)

    return trace(model)


@pytest.fixture(scope="module")
def chained():
    """A NUTS posterior and the ``chain_diagnostics`` report about it.

    ``execute_task`` builds that report while it finishes the fit and keeps
    only its REFERENCE, so the object has to be rebuilt from the same inputs:
    :func:`~bayesmith.dispatch.execute.chain_diagnostics` over the result's own
    draws, filed against the result's own reference.  Rebuilt rather than
    invented -- ``report_refs`` is asserted non-empty below, so the kind this
    gate declares is the kind dispatch really produced here.
    """
    with jax.enable_x64(True):
        graph = bilinear_pair()
        posterior = posterior_of(graph, draws=200, chains=2, nuts=True, seed=4)
        samples = {
            array.name: np.asarray(array.value)
            for array in posterior.representation.draws
        }
        report = _chain_diagnostics_report(
            chain_diagnostics(samples, num_chains=2),
            subject_ref=ArtifactRef(
                artifact_id=posterior.meta.artifact_id,
                revision=posterior.meta.revision,
                artifact_type=ArtifactKind.RESULT,
            ),
            fingerprints=posterior.run.fingerprints,
        )
    return posterior, graph, report


@pytest.fixture(scope="module")
def errored(calibrated):
    """The calibrated run with its REQUIRED ``identifiability`` made to raise.

    Exists so §0.6's ERROR row has a witness that goes through
    :func:`~bayesmith.evaluation.gate.model_checking_slots` and
    ``aggregate_gate`` together.  ``TestACheckThatRaises`` pins the status, but
    it calls ``check_posterior`` alone, so ``_recompute``'s ERROR branch --
    part of the hand-written truth table G8 rests on -- had nothing to be
    compared against.
    """
    from bayesmith.evaluation import gate as gate_module

    def explode(*args, **kwargs):
        raise RuntimeError("the rank probe fell over")

    original = gate_module.identifiability_report
    gate_module.identifiability_report = explode
    try:
        return run(lambda: calibrated.graph, posterior=calibrated.posterior)
    finally:
        gate_module.identifiability_report = original


@pytest.fixture(scope="module")
def invalidated(calibrated):
    """The calibrated run over a subject whose data has since moved.

    The other row G8's truth table could not reach: ``INVALIDATED`` is pinned
    by ``TestStatusComesBeforeVerdict`` through ``check_posterior`` only.
    """
    stale_meta = invalidate_meta(
        calibrated.posterior.meta,
        before=calibrated.posterior.meta.fingerprints,
        after=dataclasses.replace(
            calibrated.posterior.meta.fingerprints,
            data=fingerprint(FingerprintKind.DATA, "the data moved"),
        ),
        policy=InvalidationPolicy.default(),
        at="2026-09-04T00:00:00Z",
    )
    stale = dataclasses.replace(calibrated.posterior, meta=stale_meta)
    assert stale.meta.lifecycle.status is ArtifactStatus.INVALIDATED
    return run(lambda: calibrated.graph, posterior=stale)


# ------------------------------------------------------------- the definition


class TestTheDefinition:
    """``model_checking@1``: eight requirements, two of them required."""

    def test_the_identity_is_the_spelling_a_task_carries(self):
        assert MODEL_CHECKING.identity == "model_checking@1"
        assert MODEL_CHECKING.name == "model_checking"
        assert MODEL_CHECKING.version == 1

    def test_the_required_pair_and_the_optional_six(self):
        """§8.1's list, in the definition's own order.

        Order matters here for a reason that is not taste: ``aggregate_gate``
        emits findings and report refs in DEFINITION order, so this tuple is
        what a reader of a ``GateResult`` sees.
        """
        assert [item.name for item in MODEL_CHECKING.requirements] == [
            "posterior_predictive_check",
            "identifiability",
            "prior_predictive_check",
            "held_out_prediction",
            "loo_psis",
            "sbc",
            "prior_sensitivity",
            "chain_diagnostics",
        ]
        required = {item.name for item in MODEL_CHECKING.requirements if item.required}
        assert required == {"posterior_predictive_check", "identifiability"}

    def test_no_optional_requirement_blocks_on_an_error(self):
        """Every optional slot's error is reported and none of them blocks.

        The field exists, so a reader is entitled to know which way it was
        set: an optional check that could not run says so in the slot, and
        §0.6 leaves the verdict to the required pair.  ``loo_psis`` is the one
        that makes this concrete -- ``arviz.loo`` raises outright on the
        starved fixture, and a blocking optional would turn a small budget
        into an ERROR gate.
        """
        assert not any(
            item.optional_error_blocks for item in MODEL_CHECKING.requirements
        )

    def test_it_governs_no_actions_because_r3_declares_none(self):
        """Empty on purpose: ``blocked_actions`` holds action CODES, and the
        action registry is R7's.  Three invented codes would look like a
        vocabulary this release can honour."""
        assert MODEL_CHECKING.blocked_actions == ()

    def test_the_remedies_point_at_constants_that_already_exist(self):
        """A remedy carrying a number of its own would be this module's fourth
        threshold.  The one number here is D105's floor, by identity."""
        actions = {remedy.action: remedy for remedy in MODEL_CHECKING.remedies}
        assert set(actions) == {
            "raise_draws_to_the_resolution_floor",
            "build_the_graph_inside_enable_x64",
        }
        assert dict(actions["raise_draws_to_the_resolution_floor"].parameters) == {
            "draws": DRAW_FLOOR
        }
        assert actions["build_the_graph_inside_enable_x64"].parameters == ()

    def test_the_two_carriable_kinds_are_the_two_the_runner_cannot_produce(self):
        assert set(CARRIED_KINDS) == {SBC, CHAIN_DIAGNOSTICS}

    def test_the_reexport_is_the_object_the_gate_module_defines(self):
        """Identity, not spelling: ``bayesmith.evaluation.check_posterior``
        must BE this function, not another one with the same name."""
        assert reexported_check_posterior is check_posterior


class TestEveryDeclaredNameIsAKindSomethingReallyProduces:
    """A requirement name that no report ever matches is a slot that silently
    never fills -- the failure §8.1 warns about when it says to read the codes
    off the modules.  Five of the eight ARE imported from the module that owns
    them; these tests cover the three that have no constant to import."""

    def test_the_union_over_the_fixtures_covers_all_eight(
        self, calibrated, masked, abstaining_sbc, chained
    ):
        produced = {
            slot.report.report_kind
            for fixture in (calibrated, masked)
            for slot in fixture.slots
            if slot.report is not None
        }
        produced.add(abstaining_sbc.report_kind)
        produced.add(chained[2].report_kind)
        assert produced == {item.name for item in MODEL_CHECKING.requirements}

    def test_the_two_checks_kinds_are_what_checks_actually_wrote(self, calibrated):
        """``checks.py`` passes its two kinds as literals at the call site, so
        ``gate.py`` spells them.  These are the reports it wrote."""
        assert calibrated.report(POSTERIOR_PREDICTIVE_CHECK).report_kind == (
            POSTERIOR_PREDICTIVE_CHECK
        )
        assert calibrated.report(PRIOR_PREDICTIVE_CHECK).report_kind == (
            PRIOR_PREDICTIVE_CHECK
        )

    def test_a_check_wired_to_the_wrong_slot_is_refused(self, calibrated, monkeypatch):
        """The slot is chosen by the REPORT's kind, never by which check was
        asked for.

        The failure this prevents is a refactor that swaps two entries in the
        runner's table: every slot still fills, every verdict still reads
        plausibly, and two checks have exchanged names.  Here
        ``identifiability_report`` is made to hand back a
        ``posterior_predictive_check``, which must be refused rather than
        filed under the name that was asked for.
        """
        from bayesmith.evaluation import gate as gate_module

        stray = calibrated.report(POSTERIOR_PREDICTIVE_CHECK)
        monkeypatch.setattr(
            gate_module, "identifiability_report", lambda *a, **k: stray
        )
        with jax.enable_x64(True), pytest.raises(ValueError, match="another check"):
            model_checking_slots(
                calibrated.graph,
                calibrated.posterior,
                key=jax.random.key(GATE_SEED),
                budget=ComputeBudget(draws=DRAWS),
                model_ref=model_ref(),
            )

    def test_the_chain_diagnostics_kind_is_what_dispatch_actually_wrote(self, chained):
        posterior, _graph, report = chained
        assert report.report_kind == CHAIN_DIAGNOSTICS
        # And dispatch really filed one for this posterior, so the kind above
        # is not a name only this test knows.
        assert len(posterior.report_refs) == 1
        assert posterior.report_refs[0].artifact_type is ArtifactKind.EVALUATION_REPORT


# --------------------------------------------------------------- three paths


class TestTheThreePaths:
    """§8.1's fixtures: PASS, FAIL, ABSTAIN, each through the required pair."""

    def test_a_calibrated_model_is_evaluated_and_passes(self, calibrated):
        assert calibrated.gate.status is S.EVALUATED
        assert calibrated.gate.verdict is C.PASS
        for name in ("posterior_predictive_check", "identifiability"):
            report = calibrated.report(name)
            assert report.applicability is A.APPLICABLE, name
            assert report.conclusion is C.PASS, name
        assert not [code for code in calibrated.codes() if code.startswith("required_")]

    def test_a_misspecified_model_fails_and_the_finding_names_the_check(
        self, misspecified
    ):
        """``curvature=0.6``: probe_28 §1 measures both of its discrepancies at
        p = 0.0000, so this FAIL is saturated rather than marginal."""
        assert misspecified.gate.status is S.EVALUATED
        assert misspecified.gate.verdict is C.FAIL
        assert misspecified.report(POSTERIOR_PREDICTIVE_CHECK).conclusion is C.FAIL
        failed = [
            finding
            for finding in misspecified.gate.findings
            if finding.code == "required_report_failed"
        ]
        assert len(failed) == 1
        assert "posterior_predictive_check" in failed[0].message
        assert (failed[0].observed, failed[0].expected) == ("fail", "pass")

    def test_the_misspecified_fixture_still_passes_the_other_required_check(
        self, misspecified
    ):
        """Otherwise the FAIL above would be reachable by breaking anything.

        This is also what ``line_at`` exists for: built from the probe's own
        module-scope arrays, the graph's constants stay float32 and this slot
        comes back UNVERIFIABLE, which would make the fixture a test of x64
        rather than of misspecification.
        """
        assert misspecified.report("identifiability").applicability is A.APPLICABLE
        assert misspecified.report("identifiability").conclusion is C.PASS

    def test_a_budget_below_the_draw_floor_abstains(self, starved):
        """Eight draws: a p-value can only be a multiple of 1/8, so D105 says
        ABSTAIN.  The gate must carry that through as ABSTAIN and not as the
        PASS an unchecked band would have produced."""
        assert starved.gate.status is S.EVALUATED
        assert starved.gate.verdict is C.ABSTAIN
        report = starved.report(POSTERIOR_PREDICTIVE_CHECK)
        assert report.applicability is A.APPLICABLE
        assert report.conclusion is C.ABSTAIN
        assert "required_report_abstained" in starved.codes()

    def test_the_three_paths_are_three_different_verdicts(
        self, calibrated, misspecified, starved
    ):
        """The sibling guard: if two fixtures ever collapsed onto one verdict,
        each test above would still pass on its own."""
        verdicts = [
            calibrated.gate.verdict,
            misspecified.gate.verdict,
            starved.gate.verdict,
        ]
        assert verdicts == [C.PASS, C.FAIL, C.ABSTAIN]


class TestARequiredCheckThatCannotRun:
    """§0.10: a gate that cannot run a required check abstains with a reason.

    Not an exception, and not a silent skip.  ``identifiability`` refuses a
    float32 ambient precision by name, so in that environment the report is
    UNVERIFIABLE and the gate is undecided -- which is a different finding
    from ``required_report_abstained`` and names a different thing to fix.
    """

    def test_float32_makes_the_gate_abstain_rather_than_raise(self, in_float32):
        assert in_float32.gate.status is S.EVALUATED
        assert in_float32.gate.verdict is C.ABSTAIN
        assert "required_report_unverifiable" in in_float32.codes()

    def test_the_refusal_reaches_the_report_rather_than_the_caller(self, in_float32):
        report = in_float32.report("identifiability")
        assert report.applicability is A.UNVERIFIABLE
        assert report.conclusion is C.ABSTAIN
        refused = [f for f in report.findings if f.code == "diagnostic_refused"]
        assert len(refused) == 1
        assert "enable_x64" in refused[0].message

    def test_the_checks_that_do_not_need_x64_still_ran(self, in_float32):
        """Otherwise "abstain" would be indistinguishable from "gave up"."""
        assert in_float32.report(POSTERIOR_PREDICTIVE_CHECK).applicability is A.APPLICABLE
        assert in_float32.report(PRIOR_PREDICTIVE_CHECK).applicability is A.APPLICABLE

    def test_the_unverifiable_finding_is_not_the_abstained_one(
        self, in_float32, starved
    ):
        """Two ways to be undecided, two codes.  A runner that folded them
        would send a reader to raise the draw count on a precision problem."""
        assert "required_report_abstained" not in in_float32.codes()
        assert "required_report_unverifiable" not in starved.codes()


# --------------------------------------------------------------- the trap


class TestTheOptionalSlotTrap:
    """A missing optional slot and an abstaining one are different facts."""

    def test_an_abstaining_optional_report_is_filed_not_dropped(self, calibrated):
        """``held_out_prediction`` on a graph that masks nothing is
        INAPPLICABLE x ABSTAIN, and its reference must still be in the result.

        This is the trap in its cheapest form: the slot contributes no
        finding, no status and no verdict, so ``report_refs`` is the whole
        record that the check was made.
        """
        slot = calibrated.slot("held_out_prediction")
        assert slot.attempt_status is AttemptStatus.ATTEMPTED
        assert slot.report is not None
        assert slot.report.applicability is A.INAPPLICABLE
        assert slot.report.conclusion is C.ABSTAIN
        assert _ref_of(slot.report) in calibrated.aggregated.report_refs
        assert len(calibrated.gate.report_refs) == len(
            calibrated.aggregated.report_refs
        )

    def test_a_check_nobody_ran_is_not_attempted_and_has_no_reference(
        self, calibrated
    ):
        for name in CARRIED_KINDS:
            slot = calibrated.slot(name)
            assert slot.attempt_status is AttemptStatus.NOT_ATTEMPTED, name
            assert slot.report is None and slot.error is None, name
        # Eight slots, and only the ones that produced a report contribute a
        # reference.  Derived rather than pinned at six: an optional check
        # that raised on another platform would move the count without
        # touching the property.
        produced = sum(1 for slot in calibrated.slots if slot.report is not None)
        assert produced == len(MODEL_CHECKING.requirements) - len(CARRIED_KINDS)
        assert len(calibrated.gate.report_refs) == produced

    def test_carrying_an_abstaining_report_changes_the_result(
        self, abstaining_sbc, calibrated
    ):
        """The measurement at the top of this file, through the real runner.

        Same posterior, same key, same everything -- except that one optional
        check was attempted and abstained.  Status, verdict and findings are
        identical; only ``report_refs`` is not.  A runner that dropped
        abstaining optional reports would make these two results equal, and a
        reader could not tell a campaign that ran and could not answer from
        one that never ran.
        """
        with_sbc = run(straight_line, carried=(abstaining_sbc,))
        assert abstaining_sbc.conclusion is C.ABSTAIN
        assert with_sbc.gate.status == calibrated.gate.status
        assert with_sbc.gate.verdict == calibrated.gate.verdict
        assert with_sbc.gate.findings == calibrated.gate.findings

        assert len(with_sbc.gate.report_refs) == len(calibrated.gate.report_refs) + 1
        assert _ref_of(abstaining_sbc) in with_sbc.gate.report_refs
        assert _ref_of(abstaining_sbc) not in calibrated.gate.report_refs

        # And the whole results differ in that one field, nothing else: the
        # normalisation below is the exact statement of what a dropped report
        # would have hidden.
        normalised = dataclasses.replace(
            with_sbc.gate,
            meta=calibrated.gate.meta,
            report_refs=calibrated.gate.report_refs,
        )
        assert normalised == calibrated.gate
        assert with_sbc.gate != dataclasses.replace(
            calibrated.gate, meta=with_sbc.gate.meta
        )

    def test_a_carried_sbc_report_judges_another_result_and_is_still_accepted(
        self, abstaining_sbc, calibrated
    ):
        """Why ``gate.py`` checks no provenance for ``sbc``: §0.2 makes its
        subject the first usable REPLICATE's posterior, so a rule demanding
        this posterior would refuse every correct report."""
        assert abstaining_sbc.subject_ref.artifact_id != (
            calibrated.posterior.meta.artifact_id
        )
        carried = run(straight_line, carried=(abstaining_sbc,))
        assert carried.slot(SBC).report is abstaining_sbc


class TestTheCarriedChannel:
    """``chain_diagnostics`` and ``sbc`` arrive from the caller, or not at all."""

    def test_a_carried_chain_diagnostics_report_fills_its_slot(self, chained):
        posterior, graph, report = chained
        with jax.enable_x64(True):
            gate = check_posterior(
                graph,
                posterior,
                key=jax.random.key(GATE_SEED),
                budget=ComputeBudget(draws=200),
                model_ref=model_ref(),
                carried=(report,),
            )
        assert _ref_of(report) in gate.report_refs

    def test_a_chain_diagnostics_report_about_another_result_is_refused(self, chained):
        """The hole this guard closes: another fit's convergence, filed as
        this one's.  Nothing downstream could tell, because the report is
        well-formed and the gate would count it faithfully."""
        posterior, graph, report = chained
        foreign = dataclasses.replace(
            report,
            subject_ref=ArtifactRef(
                artifact_id="00000000-0000-4000-8000-000000000000",
                revision=0,
                artifact_type=ArtifactKind.RESULT,
            ),
        )
        with jax.enable_x64(True), pytest.raises(ValueError, match="other run"):
            model_checking_slots(
                graph,
                posterior,
                key=jax.random.key(GATE_SEED),
                budget=ComputeBudget(draws=200),
                model_ref=model_ref(),
                carried=(foreign,),
            )

    def test_a_kind_the_runner_produces_itself_is_refused(self, calibrated):
        """Two slots for one requirement is what ``aggregate_gate`` calls the
        silent bug; refusing here names the kind instead."""
        with jax.enable_x64(True), pytest.raises(ValueError, match="not a kind"):
            model_checking_slots(
                calibrated.graph,
                calibrated.posterior,
                key=jax.random.key(GATE_SEED),
                budget=ComputeBudget(draws=DRAWS),
                model_ref=model_ref(),
                carried=(calibrated.report(POSTERIOR_PREDICTIVE_CHECK),),
            )

    def test_two_carried_reports_of_one_kind_are_refused(
        self, abstaining_sbc, calibrated
    ):
        with jax.enable_x64(True), pytest.raises(ValueError, match="two carried"):
            model_checking_slots(
                calibrated.graph,
                calibrated.posterior,
                key=jax.random.key(GATE_SEED),
                budget=ComputeBudget(draws=DRAWS),
                model_ref=model_ref(),
                carried=(abstaining_sbc, abstaining_sbc),
            )

    def test_something_that_is_not_a_report_is_refused_by_type(self, calibrated):
        with jax.enable_x64(True), pytest.raises(TypeError, match="EvaluationReport"):
            model_checking_slots(
                calibrated.graph,
                calibrated.posterior,
                key=jax.random.key(GATE_SEED),
                budget=ComputeBudget(draws=DRAWS),
                model_ref=model_ref(),
                carried=("sbc",),
            )


# ------------------------------------------------- a check that cannot run


class TestACheckThatRaises:
    """An exception out of a check becomes a record, never a traceback."""

    def test_arviz_raising_on_a_starved_posterior_is_filed_as_an_error(self, starved):
        """Measured on this checkout: ``arviz.loo`` raises
        ``ValueError: n_draws_tail must be at least 5`` on an 8-draw
        posterior.  Without the catch the ``draws=8`` fixture §8.1 asks for
        could not reach a verdict at all -- the call would raise before the
        aggregator saw a single slot.
        """
        slot = starved.slot("loo_psis")
        assert slot.attempt_status is AttemptStatus.ATTEMPTED
        assert slot.report is None
        assert slot.error is not None
        assert slot.error.code == "check_raised"
        assert slot.error.exception_type == "ValueError"

    def test_an_optional_error_does_not_decide_the_gate(self, starved):
        """It is reported and it does not block: the verdict is the required
        pair's ABSTAIN, not an ERROR status."""
        assert starved.gate.status is S.EVALUATED
        assert "blocking_optional_report_errored" not in starved.codes()

    def test_a_required_check_that_raises_errors_the_gate_and_does_not_raise(
        self, calibrated, monkeypatch
    ):
        """The other half of "not raise and not quietly skip".

        A required check that could not run leaves the gate with an ERROR
        status -- §0.6's own answer, computed by ``aggregate_gate`` and not
        second-guessed here -- and the call still returns a result.  The two
        outcomes this forbids are the traceback that reaches the caller with
        no record at all, and the slot quietly downgraded to NOT_ATTEMPTED,
        which would read as ABSTAIN and send a reader somewhere else.
        """
        from bayesmith.evaluation import gate as gate_module

        def explode(*args, **kwargs):
            raise RuntimeError("the rank probe fell over")

        monkeypatch.setattr(gate_module, "identifiability_report", explode)
        with jax.enable_x64(True):
            gate = check_posterior(
                calibrated.graph,
                calibrated.posterior,
                key=jax.random.key(GATE_SEED),
                budget=ComputeBudget(draws=DRAWS),
                model_ref=model_ref(),
            )
        assert gate.status is S.ERROR
        assert gate.verdict is None
        errored = [
            finding
            for finding in gate.findings
            if finding.code == "required_report_errored"
        ]
        assert len(errored) == 1
        assert "identifiability" in errored[0].message
        assert errored[0].observed == "check_raised"

    def test_a_refused_task_keeps_its_own_premise_code(self, unpredictable):
        """``execute_task`` refuses the predictive replay with
        ``predictive_noise_unsupported``, and the three checks that needed it
        carry that code rather than a generic one -- it is the word a caller
        can act on."""
        for name in ("posterior_predictive_check", "held_out_prediction", "loo_psis"):
            slot = unpredictable.slot(name)
            assert slot.attempt_status is AttemptStatus.ATTEMPTED, name
            assert slot.error is not None, name
            assert slot.error.code == "predictive_noise_unsupported", name
            assert slot.error.exception_type == "Refusal", name


# -------------------------------------------------- status before verdict


class TestStatusComesBeforeVerdict:
    """§0 ruling 7: three statuses say the gate never judged, and carry none."""

    def test_a_posterior_no_predictive_can_be_made_from_blocks_the_gate(
        self, unpredictable
    ):
        """``predictive_ready`` is READ, not decided: it is the execution
        layer's own statement that the artifact three of these checks need
        does not exist, which is §0.6's BLOCKED in as many words."""
        assert unpredictable.posterior.predictive_ready is False
        assert unpredictable.gate.status is S.BLOCKED
        assert unpredictable.gate.verdict is None
        assert unpredictable.codes() == ["prerequisite_missing"]

    def test_a_blocked_gate_still_records_the_reports_that_were_made(
        self, unpredictable
    ):
        """Blocked is not "nothing happened": ``identifiability`` and the two
        checks over the prior simulation ran, and their references survive."""
        assert len(unpredictable.gate.report_refs) == 3
        assert unpredictable.report("identifiability").conclusion is C.PASS

    def test_an_invalidated_posterior_invalidates_the_gate(self, calibrated):
        """The other READ field.  Reports about a subject that has moved judge
        something that is no longer there, and §0.6 gives that its own status
        rather than folding it into a FAIL."""
        stale_meta = invalidate_meta(
            calibrated.posterior.meta,
            before=calibrated.posterior.meta.fingerprints,
            after=dataclasses.replace(
                calibrated.posterior.meta.fingerprints,
                data=fingerprint(FingerprintKind.DATA, "the data moved"),
            ),
            policy=InvalidationPolicy.default(),
            at="2026-09-04T00:00:00Z",
        )
        stale = dataclasses.replace(calibrated.posterior, meta=stale_meta)
        assert stale.meta.lifecycle.status is ArtifactStatus.INVALIDATED

        with jax.enable_x64(True):
            gate = check_posterior(
                calibrated.graph,
                stale,
                key=jax.random.key(GATE_SEED),
                budget=ComputeBudget(draws=DRAWS),
                model_ref=model_ref(),
            )
        assert gate.status is S.INVALIDATED
        assert gate.verdict is None
        assert "inputs_invalidated" in [finding.code for finding in gate.findings]

    def test_the_same_posterior_current_reaches_a_verdict(self, calibrated):
        """The sibling of the test above: without it, INVALIDATED could be
        reached by anything at all."""
        assert calibrated.posterior.meta.lifecycle.status is ArtifactStatus.CURRENT
        assert calibrated.gate.status is S.EVALUATED


# ------------------------------------------------------ it decides nothing


class TestItDecidesNothing:
    """§8.2, as things that go red rather than as a claim in a docstring.

    **The two AST tests below are belt and braces, and they were walked past.**
    They read gate.py's SPELLING -- "no numeric constant", "no ordering
    operator" -- and a spelling is a thing a rename defeats.  Measured in this
    worktree on 2026-09-04, at the commit before this docstring: a ``gate.py``
    carrying ``_MY_DRAW_FLOOR = int("40")`` and ``_SLACK = float("0.30")``
    (string arguments, so no ``ast.Constant`` of numeric type appears) and
    deciding with ``operator.lt`` / ``operator.eq`` / ``math.isclose`` (calls,
    so no ``ast.Lt`` appears) passed all 54 tests in this file, ``PYTEST_EXIT=0``,
    with ``ruff check --no-cache src/ tests/`` clean -- **while rewriting a
    required ``posterior_predictive_check`` FAIL into a PASS inside**
    :func:`~bayesmith.evaluation.gate._file`.  Handed a report whose cell was
    0.020, for which ``checks.tail_mass_within_rate(0.020)`` is ``False``, the
    slot that came back said ``pass``.  Moving the constants into a sibling
    module would have escaped both tests as well, since both read gate.py's own
    file.  This is the same failure ``ProducerRef as _PR`` produced elsewhere in
    this repository the day before.

    **So the load-bearing assertions here are identity, not text.**  A report
    is FILED, never re-decided: the slot that comes back from ``_file`` holds
    the very object handed in, and every slot the runner builds holds the very
    object its check function returned.  That property is indifferent to how a
    threshold is spelled, where it lives and which operator applies it -- a
    rewrite has to produce a different object, and ``is`` sees that.  The AST
    tests stay because they are free and they name the file; they are no longer
    the thing standing between gate.py and a second opinion.
    """

    def test_the_module_writes_down_no_number_of_its_own(self):
        """One int -- the gate's version -- and no float at all.  A tolerance
        here would be a second copy of a threshold that already has an owner,
        which is the defect §0.10 exists to prevent, one layer up.

        Belt and braces only: see the class docstring for the working
        implementation that passed this test while rewriting a verdict.
        """
        source = _gate_source()
        floats = [
            node.value
            for node in ast.walk(source)
            if isinstance(node, ast.Constant) and isinstance(node.value, float)
        ]
        ints = [
            node.value
            for node in ast.walk(source)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
        ]
        assert floats == [], f"gate.py hard-codes {floats}"
        assert ints == [MODEL_CHECKING.version], f"gate.py hard-codes {ints}"

    def test_the_module_orders_nothing(self):
        """A gate that decides no number does not compare one.  ``<``, ``<=``,
        ``>`` and ``>=`` are absent; the comparisons that remain are identity,
        membership and equality of codes.

        Belt and braces only, and for the same reason: ``operator.lt`` is an
        ordering this test cannot see.
        """
        ordering = {ast.Lt, ast.LtE, ast.Gt, ast.GtE}
        found = [
            type(op).__name__
            for node in ast.walk(_gate_source())
            if isinstance(node, ast.Compare)
            for op in node.ops
            if type(op) in ordering
        ]
        assert found == [], f"gate.py orders something: {found}"

    # ------- the load-bearing pair: a report is filed, never re-decided -------

    @pytest.mark.parametrize(
        "kind", [item.name for item in MODEL_CHECKING.requirements]
    )
    def test_a_slot_holds_the_very_report_it_was_handed(self, calibrated, kind):
        """:func:`~bayesmith.evaluation.gate._file` FILES a report; it does not
        edit one.

        Swept over every kind this gate declares, both conclusions an
        applicable check can reach, and a ladder of cells around the one band
        edge in this layer -- because a runner that re-decided a verdict would
        re-decide it near that edge, and this is where such a rule fires.  The
        assertion on every rung is ``is``: the object that comes back must be
        the object that went in, so a rewrite fails whatever arithmetic it is
        spelled with, wherever that arithmetic lives, and under whatever name.

        Not a p-value comparison and not a tolerance -- ``EDGE_MULTIPLES`` is a
        list of places to stand, and nothing here asserts a verdict about any
        of them.
        """
        template = calibrated.report(POSTERIOR_PREDICTIVE_CHECK)
        requirement = MODEL_CHECKING.requirement(kind)
        for multiple in EDGE_MULTIPLES:
            cell = multiple * (ALPHA / 2.0)
            for conclusion in (C.FAIL, C.PASS):
                report = _at_cell(
                    template, kind=kind, conclusion=conclusion, cell=cell
                )
                for slot in (_file(report), _file(report, expected=requirement)):
                    assert slot.report is report, (
                        f"{kind} at {cell!r} concluding {conclusion.value} came "
                        "back as a different object: the gate re-decided it"
                    )
                    assert slot.requirement is requirement

    def test_every_slot_holds_the_object_its_check_returned(
        self, calibrated, monkeypatch
    ):
        """End to end: what the runner files is what the checks produced.

        The sibling of the test above, and the one that survives a rewrite
        moved OUT of ``_file`` -- into the runner, into a wrapper, into a
        sibling module.  Each check is wrapped in a recorder that keeps the
        object it returned, keyed by the ``report_kind`` that object CARRIES
        (read off the check's own output, so no kind is spelled here), and
        every slot must hold that identical object.

        Both directions are asserted, because each fails differently: a kind
        filed that no check produced is an invented report, and a kind produced
        that no slot holds is the dropped-report trap
        ``TestTheOptionalSlotTrap`` covers from the other side.
        """
        from bayesmith.evaluation import gate as gate_module

        produced: dict[str, EvaluationReport] = {}

        def recording(original):
            def wrapper(*args, **kwargs):
                report = original(*args, **kwargs)
                produced[report.report_kind] = report
                return report

            return wrapper

        for attribute in (
            "posterior_predictive_check",
            "prior_predictive_check",
            "held_out_report",
            "loo_report",
            "identifiability_report",
            "prior_sensitivity_report",
        ):
            monkeypatch.setattr(
                gate_module, attribute, recording(getattr(gate_module, attribute))
            )

        with jax.enable_x64(True):
            slots = model_checking_slots(
                calibrated.graph,
                calibrated.posterior,
                key=jax.random.key(GATE_SEED),
                budget=ComputeBudget(draws=DRAWS),
                model_ref=model_ref(),
            )

        filed = {
            slot.requirement.name: slot.report
            for slot in slots
            if slot.report is not None
        }
        assert set(filed) == set(produced)
        # Non-vacuous whatever an optional check does on another BLAS: the
        # required pair reaches for no optional dependency, so both always
        # produce a report on this fixture.  Without this an all-error run
        # would satisfy the identity assertion with two empty sets.
        required = {item.name for item in MODEL_CHECKING.requirements if item.required}
        assert required <= set(filed)
        for name, report in filed.items():
            assert report is produced[name], name

    @pytest.mark.parametrize("multiple", EDGE_MULTIPLES)
    def test_a_required_fail_reaches_the_verdict_wherever_its_cell_sits(
        self, calibrated, monkeypatch, multiple
    ):
        """The consequence, which is what a reader of a gate result cares about.

        A required check that FAILED must fail the gate -- at any cell, and in
        particular at cells sitting on the band edge, which is exactly where a
        runner that decided "close enough" would overturn it.  The canned
        report is asserted to arrive in its slot BY IDENTITY as well, so the
        two ways to break this (edit the report, or ignore it) are separated.
        """
        from bayesmith.evaluation import gate as gate_module

        cell = multiple * (ALPHA / 2.0)
        failed = _at_cell(
            calibrated.report(POSTERIOR_PREDICTIVE_CHECK),
            kind=POSTERIOR_PREDICTIVE_CHECK,
            conclusion=C.FAIL,
            cell=cell,
        )
        monkeypatch.setattr(
            gate_module, "posterior_predictive_check", lambda *a, **k: failed
        )

        # D105's own floor as the budget: the cheapest run the required check
        # admits, since the report it would have produced is replaced anyway.
        budget = ComputeBudget(draws=DRAW_FLOOR)
        with jax.enable_x64(True):
            slots = model_checking_slots(
                calibrated.graph,
                calibrated.posterior,
                key=jax.random.key(GATE_SEED),
                budget=budget,
                model_ref=model_ref(),
            )
            gate = check_posterior(
                calibrated.graph,
                calibrated.posterior,
                key=jax.random.key(GATE_SEED),
                budget=budget,
                model_ref=model_ref(),
            )

        filed = [
            slot
            for slot in slots
            if slot.requirement.name == POSTERIOR_PREDICTIVE_CHECK
        ]
        assert len(filed) == 1
        assert filed[0].report is failed
        assert gate.status is S.EVALUATED
        assert gate.verdict is C.FAIL, f"a required FAIL at {cell!r} was overturned"
        assert "required_report_failed" in [
            finding.code for finding in gate.findings
        ]

    def test_the_producer_it_stamps_is_the_dispatch_object(self):
        """One fact, one object: the gate result must be stamped by the same
        ``ProducerRef`` every other artifact in this package carries."""
        from bayesmith.evaluation import gate as gate_module

        assert gate_module.PRODUCER is DISPATCH_PRODUCER

    @pytest.mark.parametrize(
        "name",
        [
            # EVALUATED x {PASS, FAIL, ABSTAIN} and BLOCKED ...
            "calibrated",
            "misspecified",
            "starved",
            "masked",
            "in_float32",
            "unpredictable",
            # ... and the two rows of the truth table that had no witness
            # here: a required check that raised, and a subject that moved.
            "errored",
            "invalidated",
        ],
    )
    def test_the_verdict_is_recomputable_from_the_slots(self, name, request):
        """R3's G8, applied to the gate: §0.6's priority, re-derived here from
        the slots' own fields, must reproduce what ``aggregate_gate`` said.

        The rule is spelled out rather than imported.  A test that read the
        implementation's own table would agree with whatever that table says,
        including when it is wrong -- the same argument
        ``tests/artifacts/test_gates.py`` makes about its ``LEGAL_PAIRS``.
        """
        fixture = request.getfixturevalue(name)
        assert _recompute(fixture) == (fixture.gate.status, fixture.gate.verdict)


# ----------------------------------------------------------- determinism


class TestDeterminism:
    """Same inputs, same decision; and the order the slots arrive in reaches
    nothing."""

    def test_the_same_input_twice_decides_the_same_thing(self, calibrated):
        """Everything except the freshly minted identities, which cannot be
        equal and are asserted UNEQUAL so the comparison is not quietly
        reading one cached object twice.  ``new_artifact_meta`` says why in as
        many words: two artifacts with identical fingerprints are still two
        artifacts.
        """
        again = run(straight_line)
        first, second = calibrated.gate, again.gate

        assert first.meta.artifact_id != second.meta.artifact_id
        assert first.report_refs != second.report_refs
        assert len(first.report_refs) == len(second.report_refs)

        normalised = dataclasses.replace(
            second, meta=first.meta, report_refs=first.report_refs
        )
        assert normalised == first

    def test_the_same_input_twice_produces_the_same_reports(self, calibrated):
        """The half the comparison above normalises away.  Two runs must reach
        the same JUDGEMENTS -- kind, both axes, and every finding -- or the
        equality above would be comparing envelopes."""
        again = run(straight_line)
        assert _judgements(again.slots) == _judgements(calibrated.slots)

    def test_the_answer_does_not_depend_on_the_order_the_slots_arrived_in(
        self, calibrated
    ):
        """R1's sweep, re-run over this gate's own eight slots: every one of
        the 40320 permutations against one envelope, compared as a whole
        result.  The bug it forbids is the loop that keeps the last verdict it
        saw, which is right exactly when the inputs happen to arrive in a
        helpful order.
        """
        slots = calibrated.slots
        assert len(slots) == len(MODEL_CHECKING.requirements) == 8
        envelope = new_artifact_meta(
            artifact_type=ArtifactKind.EVALUATION_REPORT,
            fingerprints=calibrated.posterior.run.fingerprints,
            producer=ProducerRef(package="bayesmith", version="0.7.1"),
            summary="permutation sweep",
        )
        first = None
        count = 0
        for permutation in itertools.permutations(slots):
            result = aggregate_gate(
                MODEL_CHECKING,
                meta=envelope,
                prerequisites_ready=True,
                inputs_current=True,
                slots=permutation,
            )
            if first is None:
                first = result
            else:
                assert result == first
            count += 1
        assert count == 40320

    def test_the_runner_emits_one_slot_per_requirement_in_definition_order(
        self, calibrated
    ):
        """What makes the sweep above worth its run: eight distinct slots.
        With one slot per gate, permuting is the identity."""
        assert [slot.requirement.name for slot in calibrated.slots] == [
            item.name for item in MODEL_CHECKING.requirements
        ]
        assert all(
            slot.requirement is item
            for slot, item in zip(
                calibrated.slots, MODEL_CHECKING.requirements, strict=True
            )
        )

    def test_the_gate_aggregates_the_slots_the_runner_built(self, calibrated):
        """``check_posterior`` is ``model_checking_slots`` then the aggregator
        and nothing between them: aggregating the runner's own slots by hand
        must reach the same answer.  The two runs mint different artifact ids,
        so the references are compared by count."""
        assert calibrated.aggregated.status == calibrated.gate.status
        assert calibrated.aggregated.verdict == calibrated.gate.verdict
        assert calibrated.aggregated.findings == calibrated.gate.findings
        assert calibrated.aggregated.definition is calibrated.gate.definition
        assert len(calibrated.aggregated.report_refs) == len(
            calibrated.gate.report_refs
        )


class TestTheResultsLineage:
    """A verdict that cannot be retired is an opinion (§0.3)."""

    def test_the_gate_result_names_the_posterior_it_judged(self, calibrated):
        gate = calibrated.gate
        assert gate.meta.artifact_type is ArtifactKind.EVALUATION_REPORT
        assert gate.meta.fingerprints == calibrated.posterior.run.fingerprints
        assert gate.meta.parent_refs == (
            ArtifactRef(
                artifact_id=calibrated.posterior.meta.artifact_id,
                revision=calibrated.posterior.meta.revision,
                artifact_type=ArtifactKind.RESULT,
            ),
        )

    def test_every_report_ref_points_at_a_report_the_runner_holds(self, calibrated):
        """Read off ``aggregated``, which is the one view whose references
        name reports this test is holding; ``check_posterior``'s own run minted
        a second set of artifacts with the same content, so only its COUNT is
        comparable."""
        held = {
            _ref_of(slot.report) for slot in calibrated.slots if slot.report is not None
        }
        assert set(calibrated.aggregated.report_refs) == held
        assert len(calibrated.gate.report_refs) == len(held)

    def test_the_definition_reaches_the_result_unedited(self, calibrated):
        assert calibrated.gate.definition is MODEL_CHECKING
        assert calibrated.gate.blocked_actions == MODEL_CHECKING.blocked_actions
        assert calibrated.gate.remedies == MODEL_CHECKING.remedies


# ------------------------------------------------------------------ helpers


def _gate_source() -> ast.Module:
    from bayesmith.evaluation import gate as gate_module

    path = pathlib.Path(gate_module.__file__)
    return ast.parse(path.read_text(encoding="utf-8"))


def _at_cell(
    template: EvaluationReport, *, kind: str, conclusion: Conclusion, cell: float
) -> EvaluationReport:
    """``template``'s real envelope, carrying ONE finding at ``cell``.

    The finding is BUILT rather than edited from the template's own, so the
    reports these tests hand to the runner do not depend on what the template
    happens to contain -- a gate that stripped a finding would otherwise make
    this helper raise, and a crash is a red for the wrong reason.  The
    envelope is reused because an evaluation report needs a real fingerprint
    bundle and a subject that is a result.
    """
    return dataclasses.replace(
        template,
        report_kind=kind,
        applicability=A.APPLICABLE,
        conclusion=conclusion,
        findings=(
            Finding(
                code="discrepancy_outside_rate",
                message=f"one cell, placed at {cell!r}",
                observed=("d", "a.discrepancy", cell, cell),
                expected=ALPHA / 2.0,
            ),
        ),
    )


def _ref_of(report: EvaluationReport) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=report.meta.artifact_id,
        revision=report.meta.revision,
        artifact_type=report.meta.artifact_type,
    )


def _judgements(slots):
    """What every slot DECIDED, with the minted identities left out."""
    return [
        (
            slot.requirement,
            slot.attempt_status,
            slot.error,
            None
            if slot.report is None
            else (
                slot.report.report_kind,
                slot.report.applicability,
                slot.report.conclusion,
                slot.report.findings,
            ),
        )
        for slot in slots
    ]


def _recompute(fixture):
    """§0.6's priority, re-derived from the slots and the posterior's fields."""
    if not fixture.posterior.predictive_ready:
        return (S.BLOCKED, None)
    if fixture.posterior.meta.lifecycle.status is not ArtifactStatus.CURRENT:
        return (S.INVALIDATED, None)
    if any(slot.invalidated for slot in fixture.slots):
        return (S.INVALIDATED, None)
    blocking = [
        slot
        for slot in fixture.slots
        if slot.error is not None
        and (slot.requirement.required or slot.requirement.optional_error_blocks)
    ]
    if blocking:
        return (S.ERROR, None)

    required = [slot for slot in fixture.slots if slot.requirement.required]
    failed = any(
        slot.report is not None
        and slot.report.applicability is A.APPLICABLE
        and slot.report.conclusion is C.FAIL
        for slot in required
    )
    abstained = any(
        slot.report is None
        or slot.report.applicability is not A.APPLICABLE
        or slot.report.conclusion is C.ABSTAIN
        for slot in required
    )
    if failed:
        return (S.EVALUATED, C.FAIL)
    if abstained:
        return (S.EVALUATED, C.ABSTAIN)
    return (S.EVALUATED, C.PASS)
