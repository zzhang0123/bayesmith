"""The bridge: a Graph's fingerprints, a compiled task, and typed refusals.

:mod:`bayesmith.dispatch.task` is the one place a runtime Graph and an
:class:`~bayesmith.dispatch.plan.InferencePlan` meet the artifact protocol, so
it is the one place §0 ruling 4 can be broken. What this module checks is that
it is not:

**The seven §0.3 slots move independently, or they are not slots.** Every pair
below differs in exactly one declared thing, and the assertion is which slot
moved -- ``changed_fingerprints`` answering "data" for a renamed callable, or
"graph_structure" for a flipped mask bit, would make the invalidation matrix a
table of coincidences.

**A callable has no canonical form, and ``repr`` is not a fallback.** A
function whose source cannot be identified is refused, in the typed shape §0
ruling 3 fixes, rather than digested through an address that changes on every
restart and stays put across a genuine edit.

**A verdict's code comes from a structured field, never from its prose.** The
exception-to-refusal adapters are pinned by rewriting the message and asserting
that ``failed_premise``, the finding's ``code`` and its ``observed`` do not move.
:class:`~bayesmith.errors.AffinityRefused` is deliberately absent from that
table: it says a declared affinity claim is FALSE, which is a graph-contract
fault, and adapting it into an ordinary method refusal is the downgrade §0.6
exists to prevent.
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib
import uuid

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import const, det, observe, plate, sample, trace
from bayesmith.artifacts.base import ArtifactRef, ComputeBudget, NamedArray
from bayesmith.artifacts.identity import (
    ArtifactKind,
    FingerprintKind,
    ModelRef,
    changed_fingerprints,
)
from bayesmith.artifacts.refusal import (
    CAPABILITY_UNAVAILABLE_R1,
    PREMISES,
    Refusal,
    ScopeKind,
)
from bayesmith.artifacts.reports import AnalysisReport, InferencePlanRecord
from bayesmith.artifacts.tasks import (
    Estimand,
    EvidenceTask,
    ParameterSource,
    PointEstimateTask,
    PosteriorTask,
    PredictiveTask,
    SimulationTask,
    TaskKind,
    new_task_meta,
)
from bayesmith.diagnose.coupling import Refused
from bayesmith.diagnose.map import NotApplicable
from bayesmith.dispatch import task as task_module
from bayesmith.dispatch.plan import InferencePlan
from bayesmith.dispatch.task import (
    PlannedTask,
    compile_task,
    data_manifest,
    graph_manifest,
    input_fingerprints,
    model_identity_gap,
    node_scope,
    refusal_from_verdict,
)
from bayesmith.errors import AffinityRefused, NotGaussian, NotLogLinear, StructureError
from bayesmith.graph.nodes import Continuous, Discrete
from tests.dispatch.test_plan import unfloored_probe_width
from tests.exact.models import mixed_radiometer, straight_line

K = FingerprintKind


# ------------------------------------------------------------------- fixtures


def _product(w_, x_):
    """``mu = w X``, as a module-level function with a stable qualname."""
    return w_ * x_


def _product_twin(w_, x_):
    """The same arithmetic under a different qualname -- see the pair test."""
    return w_ * x_


def _swapped(x_, w_):
    """``_product``'s two parents in the other order, same product."""
    return w_ * x_


def line(
    *,
    data=None,
    mask=None,
    design=None,
    linear_in=("w",),
    fn=_product,
    parents="w_first",
    support=None,
):
    """``d ~ N(w X, 0.5)`` with one knob per §0.3 slot the tests move.

    Written here rather than taken from ``tests/exact/models.py`` because every
    test below needs a PAIR of graphs differing in exactly one declared thing,
    and a fixture that cannot be perturbed one thing at a time cannot say
    which slot moved.
    """
    x = jnp.linspace(1.0, 4.0, 4) if design is None else design
    y = jnp.array([2.4, 5.1, 7.6, 10.2]) if data is None else data

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 2.0), support=support)
        mu = (
            det("mu", fn, w, xs, linear_in=linear_in)
            if parents == "w_first"
            else det("mu", fn, xs, w, linear_in=linear_in)
        )
        observe("d", lambda m: dist.Normal(m, 0.5), mu, obs=y, mask=mask)

    return trace(model)


def plated(*, size=4):
    """``z_i ~ N(0, 1.5)``, ``d_i ~ N(z_i, 0.4)`` on a plate of ``size``."""
    y = jnp.linspace(0.5, 2.0, size)

    def model():
        obs = plate("obs", size)
        z = sample("z", lambda: dist.Normal(0.0, 1.5), plate=obs)
        observe("d", lambda z_: dist.Normal(z_, 0.4), z, plate=obs, obs=y)

    return trace(model)


def unidentified_graph():
    """A graph whose deterministic node holds a callable with no module.

    Built through ``exec`` with globals that name no module, which is the shape
    :func:`~bayesmith.artifacts.identity.model_ref_from_callable` already
    refuses to digest: ``inspect.getsource`` has no file to read and ``repr``
    carries an address. A lambda typed into a REPL and a function built by a
    templating layer land in the same place.
    """
    namespace: dict = {}
    source = "def made(w_, x_):\n    return w_ * x_\n"
    exec(compile(source, "<made>", "exec"), namespace)  # noqa: S102
    return line(fn=namespace["made"])


def model_ref(**overrides) -> ModelRef:
    fields: dict = {"identifier": "line", "source_digest": "a" * 64}
    fields.update(overrides)
    return ModelRef(**fields)


def unpinned_ref() -> ModelRef:
    """A reference pinned by package rather than by source digest (§0.3)."""
    return model_ref(source_digest=None, package="atlas", package_version="1.2")


def posterior_task(**overrides) -> PosteriorTask:
    fields: dict = {
        "meta": new_task_meta(label="fit w"),
        "budget": ComputeBudget(draws=8, warmup=8, chains=1),
    }
    fields.update(overrides)
    return PosteriorTask(**fields)


def result_ref() -> ArtifactRef:
    return ArtifactRef(
        artifact_id=str(uuid.uuid4()), revision=0, artifact_type=ArtifactKind.RESULT
    )


def bundle_of(graph, *, task=None, ref=None, extra_data=()):
    return input_fingerprints(
        graph,
        task if task is not None else posterior_task(),
        model_ref=ref if ref is not None else model_ref(),
        extra_data=extra_data,
    )


def moved(first, second, **kwargs) -> set:
    """Which slots differ between two graphs' bundles, as a plain set."""
    return set(
        changed_fingerprints(bundle_of(first, **kwargs), bundle_of(second, **kwargs))
    )


# --------------------------------------------------- 6.1 the fingerprint seam


def test_the_same_model_traced_twice_fingerprints_the_same():
    """Identity is of the DECLARATION, not of the objects a trace produced.

    Two traces of one model build two Graphs, two closures and two arrays at
    two addresses. A fingerprint that moved between them would report every
    process restart as a changed model and make reuse impossible in exactly
    the situation it exists for.
    """
    assert moved(line(), line()) == set()


def test_an_observed_value_moves_the_data_slot_alone():
    assert moved(line(), line(data=jnp.array([2.4, 5.1, 7.6, 10.3]))) == {K.DATA}


def test_one_mask_bit_moves_the_data_slot_alone():
    """A mask says which samples were taken, so flipping a bit is a different
    data set -- and it is nothing else: the declaration is untouched."""
    every = jnp.array([True, True, True, True])
    one_dropped = jnp.array([True, False, True, True])
    assert moved(line(mask=every), line(mask=one_dropped)) == {K.DATA}


def test_a_const_value_moves_the_data_slot_alone():
    assert moved(line(), line(design=jnp.linspace(1.0, 4.5, 4))) == {K.DATA}


def test_extra_data_moves_the_data_slot_alone():
    """Data the graph does not carry is still data, and belongs in the slot
    the invalidation matrix reads for data."""
    graph = line()
    before = bundle_of(graph)
    after = bundle_of(graph, extra_data=(("holdout", np.array([1.0, 2.0])),))
    assert set(changed_fingerprints(before, after)) == {K.DATA}


def test_reordered_parents_move_the_graph_slot_alone():
    """``mu = w X`` and ``mu = X w`` are the same number and not the same
    declaration: ``parents`` is the order ``fn`` is called in."""
    assert moved(line(), line(fn=_swapped, parents="x_first")) == {K.GRAPH_STRUCTURE}


def test_a_changed_linear_in_declaration_moves_the_graph_slot_alone():
    """``linear_in`` decides whether an exact solve may run at all, so it is
    part of the structure whatever the arithmetic underneath it does."""
    assert moved(line(), line(linear_in=())) == {K.GRAPH_STRUCTURE}


def test_a_changed_support_declaration_moves_the_graph_slot_alone():
    assert moved(line(support=Continuous()), line(support=Discrete(n=3))) == {
        K.GRAPH_STRUCTURE
    }
    assert moved(line(), line(support=Continuous())) == {K.GRAPH_STRUCTURE}


def test_a_different_callable_moves_the_graph_slot_alone():
    """Same arithmetic, same data, a different function: the qualname is what
    identifies an operator, and two operators are two structures."""
    assert moved(line(), line(fn=_product_twin)) == {K.GRAPH_STRUCTURE}


def test_a_bigger_plate_moves_the_graph_slot():
    """The plate size is declared, so it is structural. Its data moves too --
    a plate of five holds five observations -- which is why this is the one
    pair in the group not asserted as moving a slot alone.
    """
    changed = moved(plated(size=4), plated(size=5))
    assert K.GRAPH_STRUCTURE in changed
    assert K.TASK not in changed and K.MODEL_SOURCE not in changed


def test_a_changed_model_source_digest_moves_the_model_slot_alone():
    """The graph is the same graph; what changed is the text it was built
    from. A digest that leaked into the structure slot would report a
    reformatted model as a restructured one."""
    graph = line()
    before = bundle_of(graph, ref=model_ref(source_digest="a" * 64))
    after = bundle_of(graph, ref=model_ref(source_digest="b" * 64))
    assert set(changed_fingerprints(before, after)) == {K.MODEL_SOURCE}


def test_the_task_slot_reads_the_question_and_not_the_task_id():
    """Two tasks asking the same thing under two minted ids are one input; a
    different budget is a different input."""
    graph = line()
    assert set(changed_fingerprints(bundle_of(graph), bundle_of(graph))) == set()
    bigger = posterior_task(budget=ComputeBudget(draws=9, warmup=8, chains=1))
    assert set(
        changed_fingerprints(bundle_of(graph), bundle_of(graph, task=bigger))
    ) == {K.TASK}


def test_a_manifest_holds_no_runtime_object():
    """§0 ruling 4 at the seam it is nearest: the manifests are what the
    artifacts layer hashes, so a Graph, a callable or a jax array reaching one
    would put a runtime handle inside an identity."""
    from bayesmith.artifacts._codec import canonical_dumps

    graph = line()
    for manifest in (graph_manifest(graph, model_ref()), data_manifest(graph)):
        canonical_dumps(manifest)  # refuses anything it cannot encode


# ------------------------------------- 6.1 a callable with no stable identity


def test_an_unidentifiable_callable_is_named_rather_than_addressed():
    assert model_identity_gap(unidentified_graph(), unpinned_ref()) == ("mu",)


def test_an_unidentifiable_callable_without_a_digest_is_a_typed_refusal():
    """``repr`` is not a fallback, and the refusal says so in structure: the
    node is named in ``grounds``, the scope is that node, and nothing in the
    payload carries an address."""
    refusal = compile_task(
        unidentified_graph(), posterior_task(), model_ref=unpinned_ref()
    )
    assert isinstance(refusal, Refusal)
    assert refusal.failed_premise == "model_source_identified"
    assert refusal.failed_premise in PREMISES
    assert refusal.scope == node_scope("mu")
    assert refusal.grounds and refusal.remedies
    assert refusal.grounds[0].observed == ("mu",)
    assert "0x" not in refusal.grounds[0].message


def test_an_explicit_digest_covers_a_callable_the_package_cannot_identify():
    """The way out §0.3 names: the caller pins the source themselves."""
    planned = compile_task(
        unidentified_graph(), posterior_task(), model_ref=model_ref()
    )
    assert isinstance(planned, PlannedTask)
    assert model_identity_gap(unidentified_graph(), model_ref()) == ()


# -------------------------------------------------- 6.2 the compile_task seam


def test_the_planned_task_shape_is_the_one_the_plan_froze():
    assert [f.name for f in dataclasses.fields(PlannedTask)] == [
        "task",
        "analysis",
        "record",
        "runtime_plan",
    ]


def test_a_posterior_task_compiles_into_records_that_hold_no_graph():
    planned = compile_task(straight_line(), posterior_task(), model_ref=model_ref())
    assert isinstance(planned, PlannedTask)
    assert isinstance(planned.analysis, AnalysisReport)
    assert isinstance(planned.record, InferencePlanRecord)
    assert isinstance(planned.runtime_plan, InferencePlan)

    assert planned.record.task_id == planned.task.meta.task_id
    assert planned.record.model_ref == model_ref()
    assert planned.record.analysis_report_ref in planned.record.meta.parent_refs
    assert (
        planned.record.analysis_report_ref.artifact_id
        == planned.analysis.meta.artifact_id
    )
    assert planned.analysis.graph_fingerprint.kind is K.GRAPH_STRUCTURE
    assert planned.analysis.meta.artifact_type is ArtifactKind.PLAN


def test_the_record_projects_the_plan_the_runtime_actually_holds():
    """The record is a projection, not a second decision: every block, its
    method and the ``tol`` derived from its kappa come off the runtime plan."""
    planned = compile_task(straight_line(), posterior_task(), model_ref=model_ref())
    runtime = planned.runtime_plan
    assert [block.names for block in planned.record.blocks] == [
        block.latents for block in runtime.blocks
    ]
    assert [block.method for block in planned.record.blocks] == [
        block.method for block in runtime.blocks
    ]
    assert planned.record.blocks[0].tolerance == runtime.exact.tol
    assert planned.analysis.candidate_routes == tuple(
        block.method for block in runtime.blocks
    )


def test_every_premise_a_record_lists_is_in_the_one_vocabulary():
    """``reports``'s docstring says a plan's premises and a refusal's
    ``failed_premise`` are one vocabulary read in both directions. Nothing
    checked it until here, and an unchecked claim about two lists is the
    defect this repository has spent the most time repairing."""
    planned = compile_task(straight_line(), posterior_task(), model_ref=model_ref())
    assert planned.record.premises
    assert set(planned.record.premises) <= PREMISES


def test_a_planned_task_is_identified_by_its_records_and_not_by_the_runtime():
    """``runtime_plan`` is ``compare=False``: it is the live object, and two
    plans compiled from one graph are the same PLANNED TASK even though the
    equinox modules underneath them are not the same object."""
    planned = compile_task(straight_line(), posterior_task(), model_ref=model_ref())
    other = compile_task(straight_line(), planned.task, model_ref=model_ref())
    assert dataclasses.replace(planned, runtime_plan=other.runtime_plan) == planned


def test_the_one_task_this_release_cannot_answer_is_refused_as_a_capability():
    """§0 ruling 1 keeps five tasks in the protocol; R3 answers the fourth.

    This test used to be parametrized over evidence AND simulation. R3 §0.7
    makes ``SimulationTask`` execute, so the simulation arm was removed and
    replaced by :func:`test_a_simulation_task_compiles_into_a_planned_task`
    below and by the execution tests in ``test_task_execution.py`` -- a
    deliberate test change, and the coverage it gives up here it gains there.

    What it still catches is more than the one refusal, because the set of
    unanswered kinds is now DERIVED rather than listed: adding a kind to
    ``SUPPORTED_TASK_KINDS`` without a story, or dropping one, turns the first
    assertion red. The old parametrized form could not have said that -- it
    named the two it already knew about, which is the shape of guard that let
    three submodules go missing from ``_LAZY_SUBMODULES`` at once.
    """
    unanswered = set(TaskKind) - task_module.SUPPORTED_TASK_KINDS
    assert unanswered == {TaskKind.EVIDENCE}, sorted(k.value for k in unanswered)

    task = EvidenceTask(meta=new_task_meta(label="Z"))
    refusal = compile_task(straight_line(), task, model_ref=model_ref())
    assert isinstance(refusal, Refusal)
    assert refusal.failed_premise == CAPABILITY_UNAVAILABLE_R1
    assert refusal.task is task
    assert refusal.grounds and refusal.remedies
    assert refusal.scope.kind is ScopeKind.TASK
    assert refusal.meta.artifact_type is ArtifactKind.PLAN
    # The refusal names what IS answered, read off the table rather than
    # restated -- so this stays true when the fifth kind lands in R4.
    assert refusal.grounds[0].expected == tuple(
        sorted(kind.value for kind in task_module.SUPPORTED_TASK_KINDS)
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(ParameterSource.prior(), id="prior"),
        pytest.param(
            ParameterSource.fixed(
                (NamedArray(name="w", value=np.asarray(2.5), dims=()),)
            ),
            id="fixed",
        ),
        pytest.param(
            ParameterSource.from_posterior_result(result_ref()),
            id="posterior_result",
        ),
    ],
)
def test_a_simulation_task_compiles_into_a_planned_task(source):
    """R3 answers simulation, and all three parameter sources compile.

    Compilation is where the source arm is NOT yet consulted -- the plan is of
    the graph, and which parameters get pushed through it is an execution-time
    question -- so all three reaching a ``PlannedTask`` is the claim. The
    posterior-source case in particular must compile against a reference it
    cannot resolve here: there is no artifact store, and refusing at compile
    time would make a plan depend on data it is not given.
    """
    task = SimulationTask(
        meta=new_task_meta(label="forward"),
        parameter_source=source,
        latent_sites=("w",),
        observed_sites=("d",),
        budget=ComputeBudget(draws=8),
    )
    planned = compile_task(straight_line(), task, model_ref=model_ref())
    assert isinstance(planned, PlannedTask)
    assert planned.record.quality_gate is None, (
        "a simulation makes no statistical claim, so it carries no gate (§0.4)"
    )


def test_a_predictive_task_compiles_into_a_planned_task():
    """R2 answers predictive: the task compiles against the graph it will push
    draws through, rather than being refused as a capability gap."""
    task = PredictiveTask(
        meta=new_task_meta(label="ppc"),
        source_posterior_ref=result_ref(),
        replicated_sites=("d",),
    )
    planned = compile_task(straight_line(), task, model_ref=model_ref())
    assert isinstance(planned, PlannedTask)


def test_an_unsupported_backend_is_a_typed_refusal():
    """R1 reads the backend off the graph's structure, so a task naming one
    cannot be honoured -- and a plan that ignored the field would honour it
    silently."""
    refusal = compile_task(
        straight_line(), posterior_task(backend="stan"), model_ref=model_ref()
    )
    assert isinstance(refusal, Refusal)
    assert refusal.failed_premise == "backend_supported"
    assert refusal.scope.kind is ScopeKind.BACKEND
    assert refusal.grounds[0].observed == "stan"
    assert refusal.grounds[0].expected == tuple(sorted(task_module.SUPPORTED_BACKENDS))


def test_an_unrecognised_option_is_refused_rather_than_ignored():
    refusal = compile_task(
        straight_line(),
        posterior_task(backend_options=(("temperature", 0.5),)),
        model_ref=model_ref(),
    )
    assert isinstance(refusal, Refusal)
    assert refusal.failed_premise == "task_options_recognised"
    assert refusal.grounds[0].observed == ("temperature",)


def test_a_posterior_mean_needs_a_whole_graph_exact_solve():
    """``estimate()`` refuses a partly-sampled graph, and the refusal belongs
    where the plan is read rather than where the solve would have run."""
    refusal = compile_task(
        mixed_radiometer(),
        PointEstimateTask(meta=new_task_meta(), estimand=Estimand.POSTERIOR_MEAN),
        model_ref=model_ref(),
    )
    assert isinstance(refusal, Refusal)
    assert refusal.failed_premise == "whole_graph_exact_solve"
    assert refusal.grounds[0].observed


def test_a_point_estimate_compiles_on_a_graph_that_has_the_solve():
    planned = compile_task(
        straight_line(),
        PointEstimateTask(meta=new_task_meta(), estimand=Estimand.POSTERIOR_MEAN),
        model_ref=model_ref(),
    )
    assert isinstance(planned, PlannedTask)
    assert "whole_graph_exact_solve" in planned.record.premises


def test_a_broken_declaration_is_an_exception_and_not_a_refusal():
    """§0.6's line: a model whose declaration is contradicted is a fault, and
    a fault arriving as a Refusal is a broken model wearing the schema of an
    ordinary "this route does not apply"."""
    with pytest.raises(StructureError):
        compile_task(unfloored_probe_width(), posterior_task(), model_ref=model_ref())


# ------------------------------------------- 6.3 exception-to-refusal adapters


def adapted(verdict) -> Refusal:
    return refusal_from_verdict(
        verdict,
        task=posterior_task(),
        model_ref=model_ref(),
        fingerprints=bundle_of(line()),
    )


def test_a_not_gaussian_verdict_reads_its_reason_and_not_its_sentence():
    first = adapted(
        NotGaussian(
            "the likelihood of d is Student-t",
            reason="not_normal",
            node="d",
            found="StudentT",
        )
    )
    second = adapted(
        NotGaussian(
            "d is not a diagonal normal, reworded entirely",
            reason="not_normal",
            node="d",
            found="StudentT",
        )
    )
    assert first.failed_premise == second.failed_premise == "gaussian_likelihood"
    assert first.grounds[0].code == second.grounds[0].code == "not_gaussian.not_normal"
    assert first.grounds[0].observed == second.grounds[0].observed == "StudentT"
    assert first.scope == second.scope == node_scope("d")
    assert first.grounds[0].message != second.grounds[0].message


def test_a_not_log_linear_verdict_reads_its_reason_and_not_its_sentence():
    payload = {"reason": "fractional_too_large", "node": "d", "fractional": 0.4}
    first = adapted(NotLogLinear("the fractional level is 0.4", **payload))
    second = adapted(NotLogLinear("reworded, and by a different author", **payload))
    assert first.failed_premise == second.failed_premise == "log_linear_route"
    assert first.grounds[0].code == second.grounds[0].code
    assert first.grounds[0].code == "not_log_linear.fractional_too_large"
    assert first.grounds[0].observed == second.grounds[0].observed == 0.4
    assert first.scope == second.scope == node_scope("d")


def test_a_map_refusal_carries_a_code_its_one_prose_field_cannot():
    """``diagnose.map.Refused`` has a ``reason`` that is a sentence and a
    ``verdict`` that is not. The adapter reads the second, which is why two
    differently worded refusals of one kind carry one code."""
    first = adapted(Refused("map_estimate did not converge, in these words"))
    second = adapted(Refused("map_estimate did not converge, in other words"))
    assert first.failed_premise == second.failed_premise == "local_mode_certified"
    assert first.grounds[0].code == second.grounds[0].code == "map_refused"
    assert first.grounds[0].observed == second.grounds[0].observed == "refused"
    assert first.grounds[0].message != second.grounds[0].message


def test_a_map_not_applicable_verdict_is_its_own_premise():
    refusal = adapted(NotApplicable("the graph has no latent nodes"))
    assert refusal.failed_premise == "graph_has_latents"
    assert refusal.grounds[0].observed == "not-applicable"


def test_an_affinity_refusal_is_not_adaptable_into_a_method_refusal():
    """``AffinityRefused`` is a ``StructureError``: the user's affinity claim
    was probed and is false. Downgrading it to "this route does not apply"
    would hide a broken declaration behind an ordinary fallback, so the
    adapter table does not know it and says so."""
    exc = AffinityRefused(
        "the prediction is not affine in w",
        names=("w",),
        at="prior centre",
        errors={1.0: 0.5},
        weighted={1.0: 12.0},
        rtol=1e-8,
        weighted_rtol=1e-2,
        failed=(1.0,),
    )
    with pytest.raises(TypeError, match="AffinityRefused"):
        adapted(exc)


def test_every_adapted_premise_is_in_the_one_vocabulary():
    verdicts = [
        NotGaussian("x", reason="jointly_dependent", node="a"),
        NotLogLinear("x", reason="noise_additive", node="d"),
        Refused("x"),
        NotApplicable("x"),
    ]
    assert {adapted(verdict).failed_premise for verdict in verdicts} <= PREMISES


def test_the_adapters_never_reach_for_a_regular_expression():
    """The rule §6.3 states, checked in the one way that cannot be satisfied
    by accident: a module that parses a message needs ``re`` or ``split``, and
    this one imports neither and calls neither."""
    source = pathlib.Path(task_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "re" not in imported
    attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "split" not in attributes and "rsplit" not in attributes
