"""``model_checking@1``: eight reports about one posterior, aggregated once.

This module is the only one in the layer that reaches no verdict of its own,
and that is its whole design (§8.2).  Every question it asks already has an
owner: the predictive band is :mod:`bayesmith.evaluation.checks`'s, the
replicate floor is :mod:`bayesmith.evaluation.sbc`'s, the rank tolerance and
the shift criterion are :mod:`bayesmith.diagnose`'s, and the truth table that
turns eight reports into one status and one verdict is
:func:`bayesmith.artifacts.gates.aggregate_gate`'s.  Nothing here compares a
number against a constant.  If a reader finds one, it is a bug: it means a
decision that lives somewhere else has been copied, and a copy agrees with its
original right up until somebody edits one of them.

**Two checks are required and six are optional, and the split is not about
cost.**  ``posterior_predictive_check`` asks whether the fitted model can
reproduce the data it was fitted to, and ``identifiability`` asks whether the
parameters were determined by that data at all; a posterior that fails either
has not been checked in any useful sense, so their absence must leave the gate
undecided.  The other six each answer a question that is real but conditional
-- there may be no held-out point, no ArviZ, no calibration campaign -- and
§0.6 lets only required checks decide.

**The trap this runner exists to avoid, measured before it was written.**
:func:`~bayesmith.artifacts.gates.aggregate_gate` was probed on this checkout
with one required PASS slot beside one optional slot in six states::

    optional slot ABSENT from slots        status=evaluated verdict=pass n_refs=1 findings=[]
    optional slot NOT_ATTEMPTED            status=evaluated verdict=pass n_refs=1 findings=[]
    optional slot UNVERIFIABLE x ABSTAIN   status=evaluated verdict=pass n_refs=2 findings=[]
    optional slot APPLICABLE x ABSTAIN     status=evaluated verdict=pass n_refs=2 findings=[]
    optional slot APPLICABLE x PASS        status=evaluated verdict=pass n_refs=2 findings=[]
    optional slot APPLICABLE x FAIL        status=evaluated verdict=pass n_refs=2 findings=['optional_report_failed']

So on the OPTIONAL axis the status, the verdict and the findings are identical
whether a check was never attempted or was attempted and could not answer:
``report_refs`` is the one field that separates them.  (On the REQUIRED axis
they are separate findings -- ``required_report_missing`` against
``required_report_abstained`` -- which is why the trap is an optional-slot
trap.)  **Therefore this runner never drops a report.**  A check that ran files
its report whatever the report says; only a check that nobody ran gets
``NOT_ATTEMPTED``.  A runner that skipped abstaining optional reports because
"they change nothing" would emit a byte-identical clean PASS for a run where a
check was attempted and failed to answer.

**Two of the eight are CARRIED rather than run**, and the reason is the same
ruling that keeps numbers out of this file.  ``sbc`` costs at least
:data:`~bayesmith.evaluation.sbc.REPLICATE_FLOOR` posterior fits, so running it
here would mean this module choosing a replicate count -- a number, and one
whose floor already has a home.  ``chain_diagnostics`` is decided by the
execution layer while it makes the posterior -- re-deciding convergence here
would put a second owner on it (§2.4, red line 3) -- and only its REFERENCE
survives on
:attr:`~bayesmith.artifacts.results.PosteriorResult.report_refs`, so a caller
who wants that slot filled supplies the report.  A caller with neither gets
two ``NOT_ATTEMPTED`` slots, which is the honest record.

**A check that raises becomes an ``ErrorRecord``, never an exception.**  Not a
defensive habit -- a measured requirement.  On this checkout ``arviz.loo``
raises ``ValueError: n_draws_tail must be at least 5`` on an 8-draw posterior
and ``ValueError: All tail values are the same`` on an 8-point masked line, so
the ``draws=8`` fixture §8.1 asks for cannot reach a verdict at all unless the
runner catches.  ``ReportSlot`` has the field for exactly this, and §0.6
decides what an error means: blocking for a required check, ignored for an
optional one unless the requirement says otherwise.  What this module must not
do is let the exception out, because then a gate that could not run a check
produces no record instead of a record saying so.

**``prerequisites_ready`` and ``inputs_current`` are READ, not decided.**  The
first is
:attr:`~bayesmith.artifacts.results.PosteriorResult.predictive_ready`, which is
the execution layer's own statement that a predictive result can be made from
this posterior -- and three of the eight checks need one, so when it is False
§0.6's BLOCKED ("needs an artifact that does not exist yet") is literally the
case.  The second is the posterior's lifecycle status: reports about an
invalidated subject judge something that has moved.

Layering: ``dispatch``, ``graph``, ``artifacts`` and this package's own
sibling modules (§0.1).  ArviZ stays optional -- it is reached only through
:func:`bayesmith.evaluation.loo.loo_report`, which imports it inside the
function that needs it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax

from bayesmith.artifacts.base import (
    ArtifactRef,
    ArtifactStatus,
    ComputeBudget,
    ErrorRecord,
    new_artifact_meta,
)
from bayesmith.artifacts.gates import (
    AttemptStatus,
    GateDefinition,
    GateResult,
    ReportRequirement,
    ReportSlot,
    aggregate_gate,
)
from bayesmith.artifacts.identity import ArtifactKind, ModelRef
from bayesmith.artifacts.refusal import Refusal, Remedy
from bayesmith.artifacts.reports import EvaluationReport
from bayesmith.artifacts.results import (
    PosteriorResult,
    PredictiveResult,
    SimulationResult,
)
from bayesmith.artifacts.tasks import (
    ParameterSource,
    PredictiveTask,
    SimulationTask,
    new_task_meta,
)
from bayesmith.dispatch.task import PRODUCER, compile_task, execute_task
from bayesmith.evaluation.checks import (
    DRAW_FLOOR,
    posterior_predictive_check,
    prior_predictive_check,
)
from bayesmith.evaluation.diagnostics import (
    IDENTIFIABILITY,
    PRIOR_SENSITIVITY,
    identifiability_report,
    prior_sensitivity_report,
)
from bayesmith.evaluation.heldout import REPORT_KIND as HELD_OUT_PREDICTION
from bayesmith.evaluation.heldout import held_out_report
from bayesmith.evaluation.loo import REPORT_KIND as LOO_PSIS
from bayesmith.evaluation.loo import loo_report
from bayesmith.evaluation.sbc import REPORT_KIND as SBC
from bayesmith.graph.graph import Graph

__all__ = [
    "CARRIED_KINDS",
    "CHAIN_DIAGNOSTICS",
    "MODEL_CHECKING",
    "POSTERIOR_PREDICTIVE_CHECK",
    "PRIOR_PREDICTIVE_CHECK",
    "check_posterior",
    "model_checking_slots",
]

#: The two ``report_kind`` codes :mod:`bayesmith.evaluation.checks` writes.
#:
#: Every other kind in this gate is IMPORTED from the module that owns it --
#: ``heldout.REPORT_KIND``, ``loo.REPORT_KIND``, ``sbc.REPORT_KIND``,
#: ``diagnostics.IDENTIFIABILITY``, ``diagnostics.PRIOR_SENSITIVITY``.  These
#: two and :data:`CHAIN_DIAGNOSTICS` have no constant to import: ``checks``
#: passes its kinds as literals at the call site and
#: ``dispatch.task._chain_diagnostics_report`` does the same.  Spelling them
#: here would be a typo waiting to become a slot that silently never fills, so
#: nothing in this module ROUTES on these constants: :func:`_file` looks a
#: report's slot up by the report's own ``report_kind``, and a name the gate
#: does not declare raises rather than being dropped.  ``test_gate.py`` fills
#: every one of the eight from a real report as well.
POSTERIOR_PREDICTIVE_CHECK = "posterior_predictive_check"
PRIOR_PREDICTIVE_CHECK = "prior_predictive_check"

#: The kind ``dispatch.task`` files beside a sampled posterior (R2).
CHAIN_DIAGNOSTICS = "chain_diagnostics"

#: The kinds :func:`check_posterior` does NOT produce, and therefore the only
#: two it will accept from a caller.  Anything else is either a kind this gate
#: does not declare or one the runner computes itself, and both would end as
#: two slots claiming one requirement.
CARRIED_KINDS: tuple[str, ...] = (SBC, CHAIN_DIAGNOSTICS)

#: Of the two carriable kinds, which one judges THIS posterior -- and so is
#: the one whose ``subject_ref`` this module can check.
#:
#: ``dispatch.task`` builds a ``chain_diagnostics`` report with the posterior
#: it is finishing as the subject, so a carried one pointing anywhere else is
#: another fit's convergence verdict, and a gate that filed it would report
#: that fit's r-hat as this one's.  ``sbc`` gets no counterpart and none is
#: invented: §0.2 makes its subject the first usable REPLICATE's posterior --
#: a different result by construction -- so there is nothing here to compare
#: it against, and a rule that guessed would refuse correct reports.
_SUBJECT_IS_THE_POSTERIOR: frozenset[str] = frozenset({CHAIN_DIAGNOSTICS})


#: ``model_checking@1``: §8.1's requirement list, in the order a reader meets
#: the questions -- can the model reproduce its data, were its parameters
#: determined, and then the six conditional checks.
#:
#: ``blocked_actions`` is empty ON PURPOSE.  The field carries action CODES,
#: and R3 declares no action registry (that is R7's); inventing three codes
#: here so the field looks used would create a vocabulary this release cannot
#: honour, which is the same species of mistake as inventing a threshold.
#: ``remedies`` is not empty, because both of its entries point at a constant
#: that already exists rather than at a new one.
MODEL_CHECKING = GateDefinition(
    name="model_checking",
    version=1,
    requirements=(
        ReportRequirement(name=POSTERIOR_PREDICTIVE_CHECK, required=True),
        ReportRequirement(name=IDENTIFIABILITY, required=True),
        ReportRequirement(name=PRIOR_PREDICTIVE_CHECK, required=False),
        ReportRequirement(name=HELD_OUT_PREDICTION, required=False),
        ReportRequirement(name=LOO_PSIS, required=False),
        ReportRequirement(name=SBC, required=False),
        ReportRequirement(name=PRIOR_SENSITIVITY, required=False),
        ReportRequirement(name=CHAIN_DIAGNOSTICS, required=False),
    ),
    blocked_actions=(),
    remedies=(
        Remedy(
            action="raise_draws_to_the_resolution_floor",
            message=(
                "A predictive check below D105's draw floor abstains rather "
                "than passing, because a p-value from that many draws cannot "
                "land in the tail the declared rate reserves. Re-run the "
                "posterior with at least this many draws."
            ),
            parameters=(("draws", DRAW_FLOOR),),
        ),
        Remedy(
            action="build_the_graph_inside_enable_x64",
            message=(
                "identifiability and prior_sensitivity refuse a float32 "
                "ambient precision, and they refuse a graph whose constants "
                "were traced outside the block as well (§0.10). Build the "
                "graph and run the check inside jax.enable_x64(True)."
            ),
        ),
    ),
)


def _ref_to(artifact: Any) -> ArtifactRef:
    """The reference to one VERSION of ``artifact``, from its own envelope."""
    return ArtifactRef(
        artifact_id=artifact.meta.artifact_id,
        revision=artifact.meta.revision,
        artifact_type=artifact.meta.artifact_type,
    )


def _from_exception(error: Exception) -> ErrorRecord:
    """A raised check, as the record §0.6 aggregates -- never the exception."""
    return ErrorRecord(
        code="check_raised",
        message=str(error) or type(error).__name__,
        exception_type=type(error).__name__,
    )


def _from_refusal(refusal: Refusal) -> ErrorRecord:
    """A refused task, keeping the refusal's OWN premise code as the code.

    A refusal already names why it declined, in the vocabulary
    :data:`bayesmith.artifacts.refusal.PREMISES` shares with the plan records,
    so re-coding it here would lose the one word a caller can act on --
    ``predictive_noise_unsupported`` says which coverage domain was left, and
    ``check_raised`` would not.
    """
    return ErrorRecord(
        code=refusal.failed_premise,
        message="; ".join(finding.message for finding in refusal.grounds),
        exception_type="Refusal",
    )


def _file(report: EvaluationReport, *, expected: ReportRequirement | None = None) -> ReportSlot:
    """The slot ``report`` belongs in, chosen by the report's OWN kind.

    Never by the caller's idea of which check it asked for.  A report whose
    ``report_kind`` this gate does not declare raises here, naming it, rather
    than being quietly dropped into a slot that then reads as unattempted --
    which is the failure the module docstring's measured table describes.
    """
    if not isinstance(report, EvaluationReport):
        raise TypeError(
            f"a check produced {type(report).__name__}, not an EvaluationReport"
        )
    declared = MODEL_CHECKING.requirement(report.report_kind)
    if expected is not None and declared is not expected:
        raise ValueError(
            f"the check asked to fill {expected.name!r} produced a "
            f"{report.report_kind!r} report; filing it would put one check's "
            "verdict under another check's name"
        )
    return ReportSlot(
        requirement=declared,
        report=report,
        attempt_status=AttemptStatus.ATTEMPTED,
    )


def _attempt(
    requirement: ReportRequirement, produce: Callable[[], EvaluationReport]
) -> ReportSlot:
    """Run ``produce`` and file whatever came back: the report, or the error."""
    try:
        report = produce()
    except Exception as error:  # noqa: BLE001 -- see the module docstring
        return ReportSlot(
            requirement=requirement,
            attempt_status=AttemptStatus.ATTEMPTED,
            error=_from_exception(error),
        )
    return _file(report, expected=requirement)


def _outcome(run: Callable[[], Any]) -> Any:
    """``run()``'s artifact, or the ``ErrorRecord`` for whatever stopped it."""
    try:
        outcome = run()
    except Exception as error:  # noqa: BLE001 -- see the module docstring
        return _from_exception(error)
    if isinstance(outcome, Refusal):
        return _from_refusal(outcome)
    return outcome


def _needs(
    subject: Any,
    requirement: ReportRequirement,
    produce: Callable[[Any], EvaluationReport],
) -> ReportSlot:
    """``produce(subject)``'s slot, or the error that stopped ``subject`` existing.

    A check whose input never arrived was still ATTEMPTED: something tried and
    the outcome is recorded.  ``NOT_ATTEMPTED`` is reserved for the two kinds
    nobody ran, and the difference is what §0.6's truth table turns on.
    """
    if isinstance(subject, ErrorRecord):
        return ReportSlot(
            requirement=requirement,
            attempt_status=AttemptStatus.ATTEMPTED,
            error=subject,
        )
    return _attempt(requirement, lambda: produce(subject))


def _predictive(
    graph: Graph,
    posterior: PosteriorResult,
    *,
    key: jax.Array,
    budget: ComputeBudget,
    model_ref: ModelRef,
) -> PredictiveResult | Refusal:
    """Replay the posterior forward: the subject of three of the eight checks."""
    task = PredictiveTask(
        meta=new_task_meta(label=f"{MODEL_CHECKING.identity} predictive"),
        source_posterior_ref=_ref_to(posterior),
        conditioned_sites=graph.observed,
        replicated_sites=graph.observed,
        latent_sites=posterior.latent_names,
        budget=budget,
        # The field exists to say which gate a task was run for; this one was
        # run for this gate.
        quality_gate=MODEL_CHECKING.identity,
    )
    planned = compile_task(graph, task, model_ref=model_ref)
    if isinstance(planned, Refusal):
        return planned
    return execute_task(planned, key=key, source_posterior=posterior)


def _simulation(
    graph: Graph,
    posterior: PosteriorResult,
    *,
    key: jax.Array,
    budget: ComputeBudget,
    model_ref: ModelRef,
) -> SimulationResult | Refusal:
    """Draw from the PRIOR: the subject of ``prior_predictive_check``.

    ``SimulationTask`` has no ``quality_gate`` field, and §0.4 says why: a
    simulation makes no statistical claim to check.  The CHECK over its output
    does, and that report is what this gate aggregates.
    """
    task = SimulationTask(
        meta=new_task_meta(label=f"{MODEL_CHECKING.identity} prior"),
        parameter_source=ParameterSource.prior(),
        latent_sites=posterior.latent_names,
        observed_sites=graph.observed,
        budget=budget,
    )
    planned = compile_task(graph, task, model_ref=model_ref)
    if isinstance(planned, Refusal):
        return planned
    return execute_task(planned, key=key)


def _carried_slots(
    posterior: PosteriorResult, carried: Sequence[EvaluationReport]
) -> dict[str, ReportSlot]:
    """The slots a caller supplied, checked against who may supply them.

    Three refusals, and each is a way a carried report would end up filed
    under a name that does not describe it: a kind this gate will not take
    from a caller, two reports of one kind, and -- for the kinds in
    :data:`_SUBJECT_IS_THE_POSTERIOR` -- a report about some other result.
    """
    slots: dict[str, ReportSlot] = {}
    for report in carried:
        if not isinstance(report, EvaluationReport):
            raise TypeError(
                "carried holds EvaluationReports; got "
                f"{type(report).__name__}"
            )
        kind = report.report_kind
        if kind not in CARRIED_KINDS:
            raise ValueError(
                f"{kind!r} is not a kind this gate accepts from a caller. "
                f"check_posterior produces every other slot itself, so a "
                f"carried one would be a second slot claiming one "
                f"requirement. Carriable: {sorted(CARRIED_KINDS)}"
            )
        if kind in slots:
            raise ValueError(
                f"two carried reports are of kind {kind!r}; the second would "
                "overwrite the first, and nothing says which one judged"
            )
        if kind in _SUBJECT_IS_THE_POSTERIOR and report.subject_ref != _ref_to(
            posterior
        ):
            raise ValueError(
                f"this {kind} report judges {report.subject_ref.artifact_id} "
                f"revision {report.subject_ref.revision}, and this gate is "
                f"about {posterior.meta.artifact_id} revision "
                f"{posterior.meta.revision}; filing it would report some "
                "other run's verdict as this one's"
            )
        slots[kind] = _file(report)
    return slots


def model_checking_slots(
    graph: Graph,
    posterior: PosteriorResult,
    *,
    key: jax.Array,
    budget: ComputeBudget,
    model_ref: ModelRef,
    carried: Sequence[EvaluationReport] = (),
) -> tuple[ReportSlot, ...]:
    """Run the checks that apply and file each outcome in its own slot.

    One slot per declared requirement, in the definition's order, always: a
    requirement with no slot is invisible to a reader of the result, and an
    optional one with no slot is invisible to
    :func:`~bayesmith.artifacts.gates.aggregate_gate` as well.

    Args:
        graph: the model.  READ -- nothing here modifies it (§2.4).  For the
            two ``diagnose`` projections it must have been BUILT inside
            ``jax.enable_x64(True)``; outside one they file an UNVERIFIABLE
            report carrying the refusal, which is a verdict the gate can
            aggregate rather than an exception it cannot.
        posterior: the subject.  Its ``latent_names`` name the sites carried
            forward, its ``predictive_ready`` says whether a predictive result
            can be made at all, and its own reference is what a carried
            ``chain_diagnostics`` report's ``subject_ref`` is checked against.
        key: PRNG key.  Split once, into one stream for the predictive replay
            and one for the prior simulation, so the same key gives the same
            eight reports and adding a check later cannot disturb the earlier
            streams.
        budget: what the two internal runs may spend.  Never defaulted: a draw
            count is a budget, and D105 is a floor on it rather than a value
            for it.
        model_ref: what the model callable is, for every compiled task.
        carried: reports produced elsewhere, of the kinds in
            :data:`CARRIED_KINDS`.  Anything else is refused by name.

    Returns:
        One :class:`~bayesmith.artifacts.gates.ReportSlot` per requirement of
        :data:`MODEL_CHECKING`, in that definition's order.

    Raises:
        TypeError: for the wrong kind of argument, or a carried entry that is
            not an evaluation report.
        ValueError: for a carried report this gate will not accept, two
            carried reports of one kind, or a carried ``chain_diagnostics``
            that judges some other result.
    """
    if not isinstance(graph, Graph):
        raise TypeError(f"model_checking_slots' graph is a Graph; got {graph!r}")
    if not isinstance(posterior, PosteriorResult):
        raise TypeError(
            "model_checking_slots judges a PosteriorResult; got "
            f"{type(posterior).__name__}"
        )
    if not isinstance(budget, ComputeBudget):
        raise TypeError(
            f"model_checking_slots' budget is a ComputeBudget; got {budget!r}"
        )
    # Checked HERE rather than left to compile_task, which is called inside
    # `_outcome`: a caller error swallowed into an ErrorRecord would come back
    # as a verdict about the model.
    if not isinstance(model_ref, ModelRef):
        raise TypeError(
            f"model_checking_slots' model_ref is a ModelRef; got {model_ref!r}"
        )

    supplied = _carried_slots(posterior, carried)
    predictive_key, prior_key = jax.random.split(key)
    predictive = _outcome(
        lambda: _predictive(
            graph, posterior, key=predictive_key, budget=budget, model_ref=model_ref
        )
    )
    simulation = _outcome(
        lambda: _simulation(
            graph, posterior, key=prior_key, budget=budget, model_ref=model_ref
        )
    )

    requirement = MODEL_CHECKING.requirement
    built: dict[str, ReportSlot] = {
        POSTERIOR_PREDICTIVE_CHECK: _needs(
            predictive,
            requirement(POSTERIOR_PREDICTIVE_CHECK),
            lambda result: posterior_predictive_check(
                graph, result, source_posterior=posterior
            ),
        ),
        IDENTIFIABILITY: _attempt(
            requirement(IDENTIFIABILITY),
            lambda: identifiability_report(graph, posterior),
        ),
        PRIOR_PREDICTIVE_CHECK: _needs(
            simulation,
            requirement(PRIOR_PREDICTIVE_CHECK),
            lambda result: prior_predictive_check(graph, result),
        ),
        HELD_OUT_PREDICTION: _needs(
            predictive,
            requirement(HELD_OUT_PREDICTION),
            lambda result: held_out_report(graph, result, source_posterior=posterior),
        ),
        LOO_PSIS: _needs(
            predictive,
            requirement(LOO_PSIS),
            lambda result: loo_report(result, graph=graph, source_posterior=posterior),
        ),
        PRIOR_SENSITIVITY: _attempt(
            requirement(PRIOR_SENSITIVITY),
            lambda: prior_sensitivity_report(graph, posterior),
        ),
    }
    for kind in CARRIED_KINDS:
        built[kind] = supplied.get(
            kind,
            ReportSlot(
                requirement=requirement(kind),
                attempt_status=AttemptStatus.NOT_ATTEMPTED,
            ),
        )

    # Walk the DEFINITION, and refuse a requirement nothing above produced.
    # A required one would surface as `required_report_missing`, but an
    # OPTIONAL one added to the definition without a producer here would be
    # silently absent -- which the module docstring's measured table shows is
    # indistinguishable from a check that ran and abstained.
    slots: list[ReportSlot] = []
    for item in MODEL_CHECKING.requirements:
        if item.name not in built:
            raise ValueError(
                f"gate {MODEL_CHECKING.identity} declares {item.name!r} and "
                "this runner produces no slot for it; an optional requirement "
                "with no producer is a slot that can never fill"
            )
        slots.append(built.pop(item.name))
    if built:
        raise ValueError(
            f"this runner built slots for {sorted(built)}, which gate "
            f"{MODEL_CHECKING.identity} does not declare"
        )
    return tuple(slots)


def check_posterior(
    graph: Graph,
    posterior: PosteriorResult,
    *,
    key: jax.Array,
    budget: ComputeBudget,
    model_ref: ModelRef,
    carried: Sequence[EvaluationReport] = (),
) -> GateResult:
    """Run ``model_checking@1`` over one posterior: slots, then the aggregator.

    :func:`model_checking_slots` then
    :func:`~bayesmith.artifacts.gates.aggregate_gate`, and nothing between them
    -- the same split as :func:`~bayesmith.evaluation.sbc.simulation_based_calibration`'s,
    for the same reason: everything about what ran is in the first, everything
    about what it means is in the second, and this function decides neither.
    The aggregator is not wrapped, extended or second-guessed here (§8.2).

    Args:
        graph: as :func:`model_checking_slots` takes it.
        posterior: the subject.  Two of its fields reach the aggregator
            directly: ``predictive_ready`` as ``prerequisites_ready``, and the
            lifecycle status of its envelope as ``inputs_current``.
        key: PRNG key for the two internal runs.
        budget: what those runs may spend.
        model_ref: what the model callable is.
        carried: ``sbc`` and ``chain_diagnostics`` reports the caller holds.

    Returns:
        The :class:`~bayesmith.artifacts.gates.GateResult`.  Its ``meta``
        carries the posterior's own fingerprints and names it as a parent, so
        §0.3 retires this verdict when the thing it judged changes.

        The result's identity is minted, so two calls on identical inputs give
        two DIFFERENT ``report_refs`` over the same decision: they are two
        artifacts, and ``new_artifact_meta`` says so in as many words.  What is
        deterministic is everything the gate decided, which is what
        ``test_gate.py`` compares.
    """
    slots = model_checking_slots(
        graph,
        posterior,
        key=key,
        budget=budget,
        model_ref=model_ref,
        carried=carried,
    )
    return aggregate_gate(
        MODEL_CHECKING,
        meta=new_artifact_meta(
            artifact_type=ArtifactKind.EVALUATION_REPORT,
            fingerprints=posterior.run.fingerprints,
            producer=PRODUCER,
            parent_refs=(_ref_to(posterior),),
            summary=f"{MODEL_CHECKING.identity} over one posterior result",
        ),
        prerequisites_ready=posterior.predictive_ready,
        inputs_current=posterior.meta.lifecycle.status is ArtifactStatus.CURRENT,
        slots=slots,
    )
