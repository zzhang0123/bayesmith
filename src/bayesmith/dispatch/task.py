"""The bridge: a Graph and its plan, projected into artifacts (§0 ruling 5).

This module is the only place in the package where a runtime object -- a
:class:`~bayesmith.graph.graph.Graph`, an
:class:`~bayesmith.dispatch.plan.InferencePlan`, a jax array -- meets the
artifact protocol, and that is deliberate: :mod:`bayesmith.artifacts` imports
neither the graph layer nor JAX, so something has to know both, and one module
knowing both is auditable in a way that five would not be.

**A manifest is what the artifacts layer hashes; the Graph never crosses.**
:func:`graph_manifest` and :func:`data_manifest` live here because they have to
read a Graph, and they return canonical values -- strings, tuples, numpy
arrays -- which :func:`~bayesmith.artifacts.identity.fingerprint` then digests.
The split is the whole of §0 ruling 4 at this seam: the layer that knows what a
node is does not know what a digest is, and the layer that takes digests never
sees a node.

**The slots are independent or they are not slots.** An observed value, a mask
bit and a Const move the DATA slot; a parent order, a plate, a declared
``linear_in``, a support and the module-and-qualname of an operator move the
GRAPH_STRUCTURE slot; the model's own text moves MODEL_SOURCE alone. An array's
shape and dtype travel with its bytes, in the data slot, rather than being
restated in the structure one -- two slots answering for one input is how a
matrix of invalidation rules becomes a table of coincidences.

**A callable has no canonical form, and ``repr`` is not a fallback.** An
operator is identified by its module and qualname. Where those are missing --
a function built by ``exec``, one typed into a REPL -- the manifest writes a
placeholder rather than an address, and :func:`compile_task` refuses unless the
caller pinned the model themselves with a ``ModelRef`` source digest. An
address changes on every restart and stays put across a genuine edit, so a
digest that degraded to one would be wrong in both directions at once.

**A refusal's code comes from a structured field.** The adapters below read
``NotGaussian.reason``, ``NotLogLinear.reason`` and the ``verdict`` class
variable that :mod:`bayesmith.diagnose.map`'s results carry; a human message
is stored as a message and parsed by nothing.
:class:`~bayesmith.errors.AffinityRefused` is absent from that table on
purpose: it says a declared affinity claim is FALSE, which is a fault in the
model, and a fault adapted into "this route does not apply" is a broken
declaration wearing the schema of an ordinary fallback.

**Nothing here decides a number.** :func:`execute_task` calls
:meth:`~bayesmith.dispatch.plan.InferencePlan.sample`,
:meth:`~bayesmith.dispatch.plan.InferencePlan.estimate`,
:func:`~bayesmith.diagnose.map.map_estimate` or :func:`~bayesmith.optimize.fit`
exactly once, with the key it was given and no extra split, and projects what
comes back. The runtime defaults are READ from the signatures that own them
rather than restated here, so a default cannot drift into a second home.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import platform
import time
import uuid
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro

from bayesmith import __version__
from bayesmith.artifacts.base import (
    ApproximationClass,
    ApproximationRecord,
    ArtifactRef,
    BackendRef,
    ComputeBudget,
    DeviceRecord,
    NamedArray,
    ProducerRef,
    RunRecord,
    RunWarning,
    SeedRecord,
    TargetFidelity,
    TerminationReason,
    TerminationRecord,
    TimingRecord,
    new_artifact_meta,
    utc_timestamp,
)
from bayesmith.artifacts.identity import (
    ArtifactKind,
    FingerprintBundle,
    FingerprintKind,
    ModelRef,
    fingerprint,
)
from bayesmith.artifacts.refusal import (
    CAPABILITY_UNAVAILABLE_R1,
    Finding,
    Refusal,
    Remedy,
    ScopeKind,
    ScopeRef,
)
from bayesmith.artifacts.reports import (
    AnalysisFinding,
    AnalysisReport,
    Applicability,
    Conclusion,
    EvaluationReport,
    InferencePlanRecord,
    PlanBlockRecord,
)
from bayesmith.artifacts.results import (
    DrawsPosterior,
    LogDensityAvailability,
    PointEstimateResult,
    PosteriorResult,
    PredictiveResult,
    Result,
    SimulationResult,
    WeightedDrawsPosterior,
)
from bayesmith.artifacts.tasks import (
    Estimand,
    ParameterSourceKind,
    Task,
    TaskKind,
    task_fingerprint,
    task_kind,
)
from bayesmith.diagnose.coupling import Refused
from bayesmith.diagnose.map import MapEstimate, NotApplicable, map_estimate
from bayesmith.dispatch.execute import (
    Posterior,
    SiteDiagnostic,
    _refuse_unless_whole_graph_exact,
)
from bayesmith.dispatch.plan import Block, InferencePlan, kappa_upper
from bayesmith.dispatch.plan import compile as compile_plan
from bayesmith.dispatch.predictive import (
    _dims,
    forward_draws,
    pointwise_log_likelihood,
    prior_draws,
    replicated_draws,
)
from bayesmith.errors import NotGaussian, NotLogLinear
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Const, Deterministic, Node, Probabilistic
from bayesmith.optimize import fit

__all__ = [
    "PRODUCER",
    "SUPPORTED_BACKENDS",
    "SUPPORTED_TASK_KINDS",
    "MAP_METHODS",
    "graph_manifest",
    "data_manifest",
    "model_identity_gap",
    "input_fingerprints",
    "node_scope",
    "plan_ref",
    "refusal_from_verdict",
    "PlannedTask",
    "compile_task",
    "execute_task",
]

#: Who wrote the artifacts this module produces. The version is READ from the
#: installed metadata through the package's own ``__version__`` rather than
#: written here: a release number in a second place changes on exactly the
#: commit where nobody is looking at this file.
PRODUCER = ProducerRef(package="bayesmith", version=__version__)

#: The backends a task may ASK for. One member, and the reason is structural:
#: :func:`~bayesmith.dispatch.plan.compile` takes no backend argument at all --
#: the graph's own structure chooses the route -- so a task naming a backend is
#: asking for something R1 cannot honour. Refusing it is the alternative to
#: honouring it silently, which is what a field nobody reads amounts to.
SUPPORTED_BACKENDS: frozenset[str] = frozenset({"auto"})

#: The four of §0 ruling 1's five questions this release answers. The fifth
#: (evidence) is refused with
#: :data:`~bayesmith.artifacts.refusal.CAPABILITY_UNAVAILABLE_R1`, which is a
#: verdict a caller can branch on rather than a NotImplementedError.
#:
#: Simulation joined in R3 (§0.7). It had to: SBC needs `(theta, y)` pairs
#: from the prior and the prior predictive check needs datasets from it, and
#: writing either one outside this seam would have put a second forward model
#: beside the one R2 built.
SUPPORTED_TASK_KINDS: frozenset[TaskKind] = frozenset(
    {
        TaskKind.POSTERIOR,
        TaskKind.POINT_ESTIMATE,
        TaskKind.PREDICTIVE,
        TaskKind.SIMULATION,
    }
)

#: The MAP seams, named so a task can choose one. ``"newton"`` is
#: :func:`~bayesmith.diagnose.map.map_estimate`, which certifies stationarity
#: and curvature before it returns a point; the other two are
#: :func:`~bayesmith.optimize.fit`'s descents, which spend a fixed budget and
#: certify nothing. Chosen by the TASK rather than by this module, because
#: picking one on the caller's behalf -- by dtype, say -- would be a numerical
#: decision taken inside an adapter that is meant to take none.
MAP_METHODS: tuple[str, ...] = ("newton", "adam", "gradient")

_POSTERIOR_OPTIONS = frozenset({"collapse", "progress_bar"})
_MAP_OPTIONS = frozenset({"method", "learning_rate"})
_POSTERIOR_MEAN_OPTIONS = frozenset({"tolerance"})

#: What an unidentifiable operator is written as. A constant, never an address:
#: two unidentifiable callables hash the same here, which is precisely why
#: :func:`model_identity_gap` requires the caller's own source digest before a
#: graph holding one may be compiled.
_UNIDENTIFIED = ("unidentified",)

#: The runtime's own sampling defaults, read off the method that owns them.
#: :meth:`~bayesmith.dispatch.plan.InferencePlan.sample`'s docstring says it is
#: "the public spelling and owns the defaults, so there is one place a caller
#: can read them off"; this reads that place rather than becoming a second one.
_SAMPLE_DEFAULTS = {
    name: parameter.default
    for name, parameter in inspect.signature(InferencePlan.sample).parameters.items()
}


# --------------------------------------------------------------- small scopes


def node_scope(name: str) -> ScopeRef:
    """The scope of a verdict about one node."""
    return ScopeRef(kind=ScopeKind.NODE, name=name)


def _scope(kind: ScopeKind, name: str) -> ScopeRef:
    return ScopeRef(kind=kind, name=name)


def plan_ref(record: InferencePlanRecord) -> ArtifactRef:
    """The version reference to a plan record: id AND revision (§0.2)."""
    return _ref(record.meta.artifact_id, record.meta.revision, ArtifactKind.PLAN)


def _ref(artifact_id: str, revision: int, kind: ArtifactKind) -> ArtifactRef:
    return ArtifactRef(artifact_id=artifact_id, revision=revision, artifact_type=kind)


def _check(value: Any, kind: type, label: str) -> None:
    if not isinstance(value, kind):
        raise TypeError(f"{label} is a {kind.__name__}; got {value!r}")


# ------------------------------------------------------------------ manifests


def _type_name(kind: type) -> str:
    return f"{kind.__module__}.{kind.__qualname__}"


def _callable_identity(fn: Any) -> tuple[str, str, str] | None:
    """A stable name for a callable, or ``None`` where there is none.

    A function carries its module and qualname, and those are what identify an
    operator across processes. A function whose ``__module__`` is missing was
    built by ``exec`` or typed into a REPL: it has no source anyone can find
    again, so it gets ``None`` rather than a best-effort spelling.

    A callable that is not a function -- an ``equinox.Module`` instance, a
    ``functools.partial`` -- is identified by its CLASS, which is stable. The
    state inside it is not identified here and is not meant to be: that is what
    a ``ModelRef``'s ``build_arguments`` and source digest are for (§0.3).
    """
    if not callable(fn):
        return None
    qualname = getattr(fn, "__qualname__", None)
    module = getattr(fn, "__module__", None)
    if isinstance(qualname, str) and qualname:
        if isinstance(module, str) and module:
            return ("callable", module, qualname)
        return None
    kind = type(fn)
    if isinstance(kind.__module__, str) and isinstance(kind.__qualname__, str):
        return ("instance", kind.__module__, kind.__qualname__)
    return None


def _operator(fn: Any) -> tuple[str, ...]:
    identity = _callable_identity(fn)
    return _UNIDENTIFIED if identity is None else identity


def _support_manifest(support: Any) -> tuple[str, int | None] | None:
    """The support DECLARATION: its type, and the state count where it has one."""
    if support is None:
        return None
    count = getattr(support, "n", None)
    return (_type_name(type(support)), None if count is None else int(count))


def _density_manifest(term: Any) -> dict[str, Any] | None:
    """A graph-level prior or evidence term: its type, and the block it is over."""
    if term is None:
        return None
    return {"type": _operator(term), "over": tuple(getattr(term, "over", ()))}


def _node_manifest(node: Node) -> dict[str, Any]:
    common: dict[str, Any] = {
        "name": node.name,
        "parents": tuple(node.parents),
        "plate": tuple(node.plate),
        "type": _type_name(type(node)),
    }
    if isinstance(node, Const):
        return {**common, "kind": "const"}
    if isinstance(node, Deterministic):
        return {
            **common,
            "kind": "deterministic",
            "fn": _operator(node.fn),
            "linear_in": tuple(node.linear_in),
        }
    if isinstance(node, Probabilistic):
        return {
            **common,
            "kind": "probabilistic",
            "dist_fn": _operator(node.dist_fn),
            "support": _support_manifest(node.support),
            "depends_on_prediction": bool(node.depends_on_prediction),
            # WHETHER the node is observed and whether a mask was declared are
            # structure; the values and the bits themselves are data.
            "latent": node.observed is None,
            "masked": node.observed_mask is not None,
        }
    return {**common, "kind": "node"}


def graph_manifest(graph: Graph, model_ref: ModelRef) -> dict[str, Any]:
    """What the GRAPH_STRUCTURE slot is taken over: the declaration, no values.

    The model's identifier travels with it because a structure is somebody's
    structure -- two models that happen to declare the same shape are two
    models -- while the source digest and the build arguments stay in the
    MODEL_SOURCE slot, so reformatting a model does not read as restructuring
    it.
    """
    _check(graph, Graph, "graph_manifest's graph")
    _check(model_ref, ModelRef, "graph_manifest's model_ref")
    return {
        "model": model_ref.identifier,
        "plates": tuple((plate.name, int(plate.size)) for plate in graph.plates),
        "nodes": tuple(_node_manifest(node) for node in graph.nodes),
        "joint_prior": _density_manifest(graph.joint_prior),
        "evidence_terms": tuple(
            _density_manifest(term) for term in graph.evidence_terms
        ),
    }


def _canonical(value: Any) -> Any:
    """A canonical value for the codec: a scalar as it is, anything array-like
    as a numpy array.

    A jax array is a runtime handle and never enters an artifact (§0 ruling 4);
    anything the codec cannot encode raises there, where the offending type is
    still in hand and can be named.
    """
    if value is None or isinstance(value, (bool, int, float, str, bytes, tuple)):
        return value
    return np.asarray(value)


def _extra_pairs(pairs: Any) -> tuple[tuple[str, Any], ...]:
    """Sorted, key-unique ``(name, value)`` pairs.

    Sorting is normalisation -- the same two arrays in two orders are one data
    set -- and a repeated key is refused because one of the two values would be
    dropped without anything saying which.
    """
    collected: dict[str, Any] = {}
    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(f"extra_data holds (name, value) pairs; got {pair!r}")
        name, value = pair
        if not isinstance(name, str) or not name:
            raise TypeError(f"extra_data keys are non-empty strings; got {name!r}")
        if name in collected:
            raise ValueError(
                f"extra_data names {name!r} twice; one of the two values would "
                "be dropped without anything saying which"
            )
        collected[name] = _canonical(value)
    return tuple((name, collected[name]) for name in sorted(collected))


def data_manifest(
    graph: Graph, extra_data: tuple[tuple[str, Any], ...] = ()
) -> dict[str, Any]:
    """What the DATA slot is taken over: every Const, observation and mask.

    In declaration order, which is the graph's own topological order, plus any
    ``extra_data`` the caller conditions on that the graph does not carry. The
    bytes are what is hashed, so a single flipped bit is a different data set
    -- and so is the same array at a different dtype, which is why the bytes
    are hashed rather than the values.
    """
    _check(graph, Graph, "data_manifest's graph")
    entries: list[tuple[str, str, Any]] = []
    for node in graph.nodes:
        if isinstance(node, Const):
            entries.append(("const", node.name, _canonical(node.value)))
        elif isinstance(node, Probabilistic):
            if node.observed is not None:
                entries.append(("observed", node.name, _canonical(node.observed)))
            if node.observed_mask is not None:
                entries.append(("mask", node.name, _canonical(node.observed_mask)))
    return {"nodes": tuple(entries), "extra": _extra_pairs(extra_data)}


def model_identity_gap(graph: Graph, model_ref: ModelRef) -> tuple[str, ...]:
    """The nodes whose operator this package cannot identify, or ``()``.

    Empty whenever the caller pinned the model's own source: a ``ModelRef``
    carrying a ``source_digest`` identifies the whole text the callables were
    written in, which is the explicit answer §0.3 offers where
    ``inspect.getsource`` cannot serve.
    """
    _check(graph, Graph, "model_identity_gap's graph")
    _check(model_ref, ModelRef, "model_identity_gap's model_ref")
    if model_ref.source_digest is not None:
        return ()
    unnamed: list[str] = []
    for node in graph.nodes:
        operator = None
        if isinstance(node, Deterministic):
            operator = node.fn
        elif isinstance(node, Probabilistic):
            operator = node.dist_fn
        if operator is not None and _callable_identity(operator) is None:
            unnamed.append(node.name)
    return tuple(unnamed)


def input_fingerprints(
    graph: Graph,
    task: Task,
    *,
    model_ref: ModelRef,
    extra_data: tuple[tuple[str, Any], ...] = (),
) -> FingerprintBundle:
    """The four slots every artifact has, taken over this graph and this task.

    ``compilation``, ``evaluation`` and ``environment`` are left empty here:
    nothing has been compiled, judged or run yet, and a slot holding a digest
    of something that has not happened is worse than an empty one.
    """
    task_kind(task)  # TypeError for anything that is not one of the five
    return FingerprintBundle(
        model_source=fingerprint(FingerprintKind.MODEL_SOURCE, model_ref),
        graph_structure=fingerprint(
            FingerprintKind.GRAPH_STRUCTURE, graph_manifest(graph, model_ref)
        ),
        data=fingerprint(FingerprintKind.DATA, data_manifest(graph, extra_data)),
        task=task_fingerprint(task),
    )


# -------------------------------------------------- exception-to-refusal seam


#: What each refusal offers the caller instead. Non-empty by construction: a
#: refusal with nowhere to go is a dead end wearing a schema (§0 ruling 3), and
#: a table is where that is noticed -- a premise added without a remedy is a
#: missing row rather than an empty tuple somewhere inside a branch.
_REMEDIES: dict[str, tuple[Remedy, ...]] = {
    CAPABILITY_UNAVAILABLE_R1: (
        Remedy(
            action="ask_a_supported_task",
            message="R1 answers a posterior task and a point-estimate task. Ask "
            "one of those against this graph, or keep this task and run it "
            "against a later release that answers it.",
            parameters=(("supported", ("point_estimate", "posterior")),),
        ),
    ),
    "backend_supported": (
        Remedy(
            action="request_the_auto_backend",
            message="The graph's structure chooses the route in this release, "
            "so the backend field carries no request this package can honour. "
            "Leave it at auto and read the run record for what actually ran.",
            parameters=(("backend", "auto"),),
        ),
    ),
    "model_source_identified": (
        Remedy(
            action="pass_a_source_digest",
            message="Give the ModelRef a source_digest of the text the model "
            "was built from; that pins every operator in it at once.",
        ),
        Remedy(
            action="declare_the_operator_at_module_scope",
            message="Define the operator in a module rather than through exec "
            "or a REPL, so its module and qualname identify it.",
        ),
    ),
    "task_options_recognised": (
        Remedy(
            action="drop_the_unrecognised_option",
            message="An option this release does not read would change nothing "
            "while looking as though it had. Remove it, or use the field that "
            "carries the same request.",
        ),
    ),
    "whole_graph_exact_solve": (
        Remedy(
            action="ask_a_posterior_task",
            message="A point estimate of a partly-sampled graph is a MAP over "
            "the sampled latents, not the conditional mean of the exact block "
            "at some arbitrary value of the others. Ask a posterior task, or a "
            "point estimate whose estimand is the MAP.",
            parameters=(("task_kind", "posterior"),),
        ),
    ),
    "named_latents_declared": (
        Remedy(
            action="name_declared_latents",
            message="Name latents this graph declares, or leave names=None for "
            "every latent.",
        ),
    ),
    "gaussian_likelihood": (
        Remedy(
            action="use_nuts",
            message="The exact route needs a diagonal Gaussian it does not have "
            "here. Sample the graph instead; the dispatcher routes a "
            "non-Gaussian likelihood to NUTS on its own.",
        ),
    ),
    "log_linear_route": (
        Remedy(
            action="use_nuts",
            message="No log-linear route exists here. Sample the graph instead, "
            "or change the declaration the verdict names.",
        ),
    ),
    "local_mode_certified": (
        Remedy(
            action="start_from_another_point",
            message="The optimisation could not certify a mode. Try another "
            "starting basin, rescale or reparameterise the latent the verdict "
            "names, or sample the graph.",
        ),
    ),
    "graph_has_latents": (
        Remedy(
            action="evaluate_the_graph",
            message="This graph declares no latent, so there is nothing to "
            "estimate. Evaluate it directly.",
        ),
    ),
    "posterior_data_mismatch": (
        Remedy(
            action="run_the_source_posterior_on_the_same_data",
            message="A predictive task may only reuse a posterior drawn from the "
            "same model, graph and conditioning data. Run the posterior task "
            "against the same graph this predictive task names, then reference "
            "that result.",
        ),
    ),
    "predictive_noise_unsupported": (
        Remedy(
            action="use_a_diagonal_gaussian_observation",
            message="R2 generates predictive draws only for a diagonal-Gaussian "
            "observed node. A correlated or non-Gaussian observation has no "
            "per-sample loc/scale for this seam to draw from; keep the observed "
            "node diagonal Gaussian, or ask a later release.",
        ),
    ),
}


def _not_gaussian(verdict: NotGaussian) -> tuple[str, Finding, str | None]:
    return (
        "gaussian_likelihood",
        Finding(
            code=f"not_gaussian.{verdict.reason}",
            message=str(verdict),
            observed=verdict.found,
            expected="diagonal_normal",
        ),
        verdict.node,
    )


def _not_log_linear(verdict: NotLogLinear) -> tuple[str, Finding, str | None]:
    # The MEASURED number where the reason has one, the distribution type
    # where it has that instead. Both are public structured attributes; the
    # sentence in the message is read by nobody.
    measured = verdict.fractional if verdict.fractional is not None else verdict.found
    return (
        "log_linear_route",
        Finding(
            code=f"not_log_linear.{verdict.reason}",
            message=str(verdict),
            observed=measured,
            expected="log_linear_route",
        ),
        verdict.node,
    )


def _map_refused(verdict: Refused) -> tuple[str, Finding, str | None]:
    # A map refusal carries one prose field and one class variable. The class
    # variable is the structured half, and the call context -- a MAP that could
    # not be certified -- is what fixes the code, which is what §6.3 allows for
    # a verdict with no reason vocabulary of its own.
    return (
        "local_mode_certified",
        Finding(
            code="map_refused",
            message=verdict.reason,
            observed=verdict.verdict,
            expected=MapEstimate.verdict,
        ),
        None,
    )


def _map_not_applicable(verdict: NotApplicable) -> tuple[str, Finding, str | None]:
    return (
        "graph_has_latents",
        Finding(
            code="map_not_applicable",
            message=verdict.reason,
            observed=verdict.verdict,
            expected=MapEstimate.verdict,
        ),
        None,
    )


#: By EXACT type, and short on purpose. ``AffinityRefused`` is not here and must
#: not be: it is a ``StructureError`` saying a declared affinity claim is false,
#: and the difference between a fault and a verdict is precisely that one of
#: them keeps travelling as an exception (§0.6).
_ADAPTERS = {
    NotGaussian: _not_gaussian,
    NotLogLinear: _not_log_linear,
    Refused: _map_refused,
    NotApplicable: _map_not_applicable,
}


def refusal_from_verdict(
    verdict: Any,
    *,
    task: Task,
    model_ref: ModelRef,
    fingerprints: FingerprintBundle,
    artifact_type: ArtifactKind = ArtifactKind.PLAN,
) -> Refusal:
    """The typed :class:`~bayesmith.artifacts.refusal.Refusal` for a verdict.

    By exact type, never by ``isinstance``: a subclass carries state this table
    cannot see, and :class:`~bayesmith.errors.AffinityRefused` is exactly such a
    subclass in the direction that matters -- adapting it would downgrade a
    contradicted declaration into an ordinary "this route does not apply".
    """
    adapter = _ADAPTERS.get(type(verdict))
    if adapter is None:
        raise TypeError(
            f"{type(verdict).__name__} is not one of the verdicts this module "
            f"adapts ({sorted(kind.__name__ for kind in _ADAPTERS)}). A fault "
            "keeps travelling as an exception: a graph whose declaration is "
            "contradicted is not a route that does not apply."
        )
    premise, finding, node = adapter(verdict)
    scope = node_scope(node) if node else _scope(ScopeKind.MODEL, model_ref.identifier)
    return _refusal(
        task,
        artifact_type=artifact_type,
        fingerprints=fingerprints,
        failed_premise=premise,
        grounds=(finding,),
        scope=scope,
        summary=finding.message,
    )


def _refusal(
    task: Task,
    *,
    artifact_type: ArtifactKind,
    fingerprints: FingerprintBundle,
    failed_premise: str,
    grounds: tuple[Finding, ...],
    scope: ScopeRef,
    summary: str,
) -> Refusal:
    return Refusal(
        meta=new_artifact_meta(
            artifact_type=artifact_type,
            fingerprints=fingerprints,
            producer=PRODUCER,
            summary=summary,
        ),
        task=task,
        failed_premise=failed_premise,
        grounds=grounds,
        scope=scope,
        remedies=_REMEDIES[failed_premise],
    )


# --------------------------------------------------------------- compile_task


@dataclasses.dataclass(frozen=True, slots=True)
class PlannedTask:
    """A task, what compiling it concluded, and the plan that will run it.

    ``runtime_plan`` is out of the identity (``compare=False``) and out of the
    repr: it is the live equinox module, and two plans compiled from one graph
    are the same PLANNED TASK even though the objects underneath them are two.
    It is also the one field here that is not an artifact, which is why it is
    named for what it is rather than hidden behind a friendlier word.
    """

    task: Task
    analysis: AnalysisReport
    record: InferencePlanRecord
    runtime_plan: InferencePlan = dataclasses.field(compare=False, repr=False)


def _map_method(task: Task) -> Any:
    """Which MAP seam this task asked for, defaulting to the certified one."""
    return dict(task.optimizer_options).get("method", MAP_METHODS[0])


def _known_options(task: Task, kind: TaskKind) -> frozenset[str]:
    if kind is TaskKind.POSTERIOR:
        return _POSTERIOR_OPTIONS
    if kind in (TaskKind.PREDICTIVE, TaskKind.SIMULATION):
        return frozenset()
    return _MAP_OPTIONS if task.estimand is Estimand.MAP else _POSTERIOR_MEAN_OPTIONS


def _given_options(task: Task, kind: TaskKind) -> tuple[tuple[str, Any], ...]:
    if kind in (TaskKind.POSTERIOR, TaskKind.PREDICTIVE, TaskKind.SIMULATION):
        return task.backend_options
    return task.optimizer_options


def _option_refusal(
    task: Task, kind: TaskKind, fingerprints: FingerprintBundle
) -> Refusal | None:
    known = _known_options(task, kind)
    unknown = tuple(name for name, _ in _given_options(task, kind) if name not in known)
    if not unknown:
        return None
    return _refusal(
        task,
        artifact_type=ArtifactKind.PLAN,
        fingerprints=fingerprints,
        failed_premise="task_options_recognised",
        grounds=(
            Finding(
                code="unrecognised_option",
                message=f"this release reads none of {list(unknown)} on a "
                f"{kind.value} task, so passing them would change nothing while "
                "looking as though it had",
                observed=unknown,
                expected=tuple(sorted(known)),
            ),
        ),
        scope=_scope(ScopeKind.TASK, kind.value),
        summary=f"unrecognised option(s) {list(unknown)}",
    )


def _method_refusal(
    task: Task, method: Any, fingerprints: FingerprintBundle
) -> Refusal | None:
    """A MAP task naming an optimiser this package does not have.

    The value, where :func:`_option_refusal` checks the key. Both are the same
    premise: an option this release cannot read would otherwise be ignored,
    and a MAP produced by a silently substituted optimiser is a different
    number under the same name.
    """
    if method in MAP_METHODS:
        return None
    return _refusal(
        task,
        artifact_type=ArtifactKind.PLAN,
        fingerprints=fingerprints,
        failed_premise="task_options_recognised",
        grounds=(
            Finding(
                code="unrecognised_optimiser",
                message=f"{method!r} is not a MAP route this package has; it "
                f"knows {list(MAP_METHODS)}. Named rather than guessed, because "
                "a typo falling through to a default would change the algorithm "
                "silently",
                observed=method,
                expected=MAP_METHODS,
            ),
        ),
        scope=_scope(ScopeKind.TASK, TaskKind.POINT_ESTIMATE.value),
        summary=f"no {method!r} MAP route in this package",
    )


def _capability_refusal(
    task: Task, kind: TaskKind, fingerprints: FingerprintBundle, artifact: ArtifactKind
) -> Refusal:
    return _refusal(
        task,
        artifact_type=artifact,
        fingerprints=fingerprints,
        failed_premise=CAPABILITY_UNAVAILABLE_R1,
        grounds=(
            Finding(
                code="task_kind_unavailable",
                message=f"a {kind.value} task is part of the frozen protocol and "
                "is not answered in this release; the question is held, and "
                "nothing was computed for it",
                observed=kind.value,
                expected=tuple(sorted(item.value for item in SUPPORTED_TASK_KINDS)),
            ),
        ),
        scope=_scope(ScopeKind.TASK, kind.value),
        summary=f"a {kind.value} task is not answered in this release",
    )


def _backend_refusal(task: Task, fingerprints: FingerprintBundle) -> Refusal:
    return _refusal(
        task,
        artifact_type=ArtifactKind.PLAN,
        fingerprints=fingerprints,
        failed_premise="backend_supported",
        grounds=(
            Finding(
                code="backend_unavailable",
                message=f"this release has no {task.backend!r} backend to "
                "dispatch to; the graph's structure chooses the route, and the "
                "run record names the backend that actually ran",
                observed=task.backend,
                expected=tuple(sorted(SUPPORTED_BACKENDS)),
            ),
        ),
        scope=_scope(ScopeKind.BACKEND, task.backend),
        summary=f"no {task.backend!r} backend in this release",
    )


def _identity_refusal(
    task: Task, gap: tuple[str, ...], fingerprints: FingerprintBundle
) -> Refusal:
    return _refusal(
        task,
        artifact_type=ArtifactKind.PLAN,
        fingerprints=fingerprints,
        failed_premise="model_source_identified",
        grounds=(
            Finding(
                code="operator_not_identifiable",
                message=f"the operators at {list(gap)} carry no module and "
                "qualname, so this package cannot say which functions they are. "
                "There is no fallback: an address identifies a process rather "
                "than a model, changing on every restart and staying put across "
                "a genuine edit",
                observed=gap,
                expected="module_and_qualname",
            ),
        ),
        scope=node_scope(gap[0]),
        summary=f"the operator at {gap[0]!r} cannot be identified",
    )


def _undeclared_refusal(
    task: Task,
    unknown: tuple[str, ...],
    latents: tuple[str, ...],
    fingerprints: FingerprintBundle,
) -> Refusal:
    return _refusal(
        task,
        artifact_type=ArtifactKind.PLAN,
        fingerprints=fingerprints,
        failed_premise="named_latents_declared",
        grounds=(
            Finding(
                code="undeclared_latent",
                message=f"{list(unknown)} are not latents of this graph; its "
                f"latents are {list(latents)}",
                observed=unknown,
                expected=latents,
            ),
        ),
        scope=_scope(ScopeKind.TASK, TaskKind.POINT_ESTIMATE.value),
        summary=f"undeclared latent(s) {list(unknown)}",
    )


def _refuse_before_compiling(
    graph: Graph,
    task: Task,
    kind: TaskKind,
    model_ref: ModelRef,
    bundle: FingerprintBundle,
) -> Refusal | None:
    """Everything decidable by reading the task and the graph.

    All of it before :func:`~bayesmith.dispatch.plan.compile` runs: a task that
    is going to be refused should not first pay for the probes a plan costs.
    """
    if kind not in SUPPORTED_TASK_KINDS:
        return _capability_refusal(task, kind, bundle, ArtifactKind.PLAN)
    if task.backend not in SUPPORTED_BACKENDS:
        return _backend_refusal(task, bundle)
    option_refusal = _option_refusal(task, kind, bundle)
    if option_refusal is not None:
        return option_refusal
    if kind is TaskKind.POINT_ESTIMATE and task.estimand is Estimand.MAP:
        method_refusal = _method_refusal(task, _map_method(task), bundle)
        if method_refusal is not None:
            return method_refusal
    if kind is TaskKind.POINT_ESTIMATE and task.names is not None:
        unknown = tuple(name for name in task.names if name not in graph.latents)
        if unknown:
            return _undeclared_refusal(task, unknown, graph.latents, bundle)
    gap = model_identity_gap(graph, model_ref)
    if gap:
        return _identity_refusal(task, gap, bundle)
    return None


def _estimate_refusal(
    runtime: InferencePlan, task: Task, bundle: FingerprintBundle
) -> Refusal | None:
    """The one refusal that needs the compiled plan: a posterior mean has to
    have a whole-graph exact solve to be the mean OF.

    The condition is not restated here. The guard
    :meth:`~bayesmith.dispatch.plan.InferencePlan.estimate` itself runs is
    called, and its ``NotImplementedError`` -- a known method-inapplicability
    -- is what becomes the Refusal, so the compiler's verdict and the runtime's
    cannot drift apart.
    """
    try:
        _refuse_unless_whole_graph_exact(runtime)
    except NotImplementedError as exc:
        sampled = () if runtime.sampled is None else tuple(runtime.sampled.latents)
        return _refusal(
            task,
            artifact_type=ArtifactKind.PLAN,
            fingerprints=bundle,
            failed_premise="whole_graph_exact_solve",
            grounds=(
                Finding(
                    code="not_whole_graph_exact",
                    message=str(exc),
                    observed=sampled,
                    expected=(),
                ),
            ),
            scope=_scope(ScopeKind.MODEL, "posterior_mean"),
            summary="this graph has no whole-graph exact solve to take a mean of",
        )
    return None


def _finite_positive(value: float | None) -> float | None:
    """A measurement, or nothing.

    A record holds a number it can stand behind: a bound that overflowed
    measured nothing, and writing it down as a number would put an infinity
    where a conditioning estimate is read.
    """
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0.0 else None


def _reason_codes(runtime: InferencePlan, block: Block) -> tuple[str, ...]:
    """Why this method for these latents, as CODES read off structured fields.

    The block's own ``reason`` is prose written for a person, and it is kept as
    prose -- in the analysis report's summary. What a consumer branches on
    comes from the plan's structured fields instead: the method, whether a
    linearity check ran, whether a conditioning bound exists, and whether the
    tolerance derived from it is reachable at this dtype.
    """
    if block.method == "nuts":
        return (
            ("outside_exact_block",)
            if runtime.exact is not None
            else ("no_exact_structure",)
        )
    return (
        "linear_in_checked" if block.linearity else "linear_in_vacuous",
        "condition_bound_measured"
        if block.kappa is not None
        else "condition_unmeasured",
        "tol_attainable" if block.tol_attainable else "tol_unattainable_at_dtype",
        "sigma_rebuilt_per_sweep"
        if runtime.sigma_needs_rebuild
        else "sigma_hoisted_out_of_the_sweep",
    )


def _linearity_evidence(block: Block) -> tuple[int | None, float | None]:
    """``(at-points, worst departure)`` from the block's own linearity record.

    The same two numbers :func:`~bayesmith.dispatch.plan._evidence` prints, read
    off the same dict rather than off its rendering: both are derived where
    they are used, so there is no stored second copy to go stale.
    """
    if not block.linearity:
        return (None, None)
    worst = max(value for row in block.linearity.values() for value in row.values())
    return (len(block.linearity), float(worst))


def _kappa_ends(block: Block) -> tuple[float | None, float | None]:
    if block.kappa is None:
        return (None, None)
    if isinstance(block.kappa, tuple):
        return (block.kappa[0], block.kappa[1])
    return (block.kappa, block.kappa)


def _block_finding(runtime: InferencePlan, index: int, block: Block) -> AnalysisFinding:
    low, high = _kappa_ends(block)
    at_points, worst = _linearity_evidence(block)
    return AnalysisFinding(
        code="exact_block" if block.method != "nuts" else "sampled_block",
        conclusion=block.method,
        scope=_scope(ScopeKind.BLOCK, f"block_{index}"),
        measurements=(
            ("members", tuple(block.latents)),
            ("kappa_low", _finite_positive(low)),
            ("kappa_high", _finite_positive(high)),
            ("tolerance", block.tol),
            ("epsilon", block.epsilon),
            ("tol_attainable", block.tol_attainable),
            ("linearity_at_points", at_points),
            ("linearity_worst_departure", worst),
        ),
        grounds=_reason_codes(runtime, block),
    )


def _block_approximation(task: Task, method: str) -> ApproximationRecord | None:
    """How this block's answer is produced, on the two axes of §0.2.

    A posterior task draws: iid exact-linear draws and NUTS are both
    ``MONTE_CARLO`` of an ``EXACT`` target, which is the pairing
    :class:`~bayesmith.artifacts.base.ApproximationRecord`'s own docstring
    gives. A posterior mean solves rather than draws. A MAP is produced by an
    optimiser this plan's blocks do not schedule, so the block says nothing
    about it and the run record answers for it instead.
    """
    kind = task_kind(task)
    if kind is TaskKind.POSTERIOR:
        return ApproximationRecord(
            representation_class=ApproximationClass.MONTE_CARLO,
            target_fidelity=TargetFidelity.EXACT,
            details=(("method", method),),
        )
    if kind is TaskKind.POINT_ESTIMATE and task.estimand is Estimand.POSTERIOR_MEAN:
        return ApproximationRecord(
            representation_class=ApproximationClass.CERTIFIED_DETERMINISTIC,
            target_fidelity=TargetFidelity.EXACT,
            details=(("method", method),),
        )
    return None


def _block_record(runtime: InferencePlan, task: Task, block: Block) -> PlanBlockRecord:
    return PlanBlockRecord(
        names=tuple(block.latents),
        method=block.method,
        reason_codes=_reason_codes(runtime, block),
        kappa=_finite_positive(
            None if block.kappa is None else kappa_upper(block.kappa)
        ),
        tolerance=_finite_positive(block.tol),
        approximation=_block_approximation(task, block.method),
    )


def _premises(runtime: InferencePlan, task: Task, kind: TaskKind) -> tuple[str, ...]:
    """The codes this plan RESTS on, in the vocabulary a refusal names one of.

    :mod:`bayesmith.artifacts.reports` states the relation and
    ``tests/dispatch/test_task_protocol.py`` holds both directions to it: what a
    plan lists here is what a Refusal would name in ``failed_premise`` if it
    turned out to be false.
    """
    codes = ["backend_supported", "model_source_identified", "task_options_recognised"]
    if runtime.exact is not None:
        codes += ["gaussian_likelihood", "affine_prediction"]
    if kind is TaskKind.POINT_ESTIMATE:
        if task.estimand is Estimand.POSTERIOR_MEAN:
            codes.append("whole_graph_exact_solve")
        else:
            codes.append("graph_has_latents")
            if _map_method(task) == MAP_METHODS[0]:
                codes.append("local_mode_certified")
        if task.names is not None:
            codes.append("named_latents_declared")
    return tuple(codes)


def _fallback_policy(task: Task, kind: TaskKind) -> str | None:
    """What happens if the route collapses, as a code rather than as a habit."""
    if kind is not TaskKind.POSTERIOR:
        return None
    return "nuts_on_collapse" if task.nuts_on_collapse else "annotate_on_collapse"


def _analysis_report(
    runtime: InferencePlan, task: Task, model_ref: ModelRef, bundle: FingerprintBundle
) -> AnalysisReport:
    return AnalysisReport(
        meta=new_artifact_meta(
            artifact_type=ArtifactKind.PLAN,
            fingerprints=bundle,
            producer=PRODUCER,
            summary="; ".join(block.reason for block in runtime.blocks),
        ),
        model_ref=model_ref,
        graph_fingerprint=bundle.graph_structure,
        findings=tuple(
            _block_finding(runtime, index, block)
            for index, block in enumerate(runtime.blocks)
        ),
        candidate_routes=tuple(block.method for block in runtime.blocks),
    )


def _plan_record(
    runtime: InferencePlan,
    task: Task,
    kind: TaskKind,
    model_ref: ModelRef,
    bundle: FingerprintBundle,
    analysis: AnalysisReport,
) -> InferencePlanRecord:
    blocks = tuple(_block_record(runtime, task, block) for block in runtime.blocks)
    policy = _fallback_policy(task, kind)
    compiled = dataclasses.replace(
        bundle,
        compilation=fingerprint(
            FingerprintKind.COMPILATION,
            {
                "blocks": blocks,
                "fallback_policy": policy,
                "sigma_needs_rebuild": bool(runtime.sigma_needs_rebuild),
            },
        ),
    )
    reference = _ref(
        analysis.meta.artifact_id, analysis.meta.revision, ArtifactKind.PLAN
    )
    return InferencePlanRecord(
        meta=new_artifact_meta(
            artifact_type=ArtifactKind.PLAN,
            fingerprints=compiled,
            producer=PRODUCER,
            parent_refs=(reference,),
            summary=runtime._execution(),
        ),
        task_id=task.meta.task_id,
        model_ref=model_ref,
        analysis_report_ref=reference,
        blocks=blocks,
        backend=task.backend,
        premises=_premises(runtime, task, kind),
        budget=task.budget,
        # Four tasks carry a gate and simulation carries none -- §0.4's
        # deliberate omission, not an oversight: a simulation produced what
        # the model says, and whether that is a good model is what the other
        # four ask. Spelled as a branch rather than as a `getattr` default so
        # that a SIXTH task kind arriving without the field is a loud
        # AttributeError here rather than a silently ungated plan.
        quality_gate=None if kind is TaskKind.SIMULATION else task.quality_gate,
        fallback_policy=policy,
    )


def compile_task(
    graph: Graph,
    task: Task,
    *,
    model_ref: ModelRef,
    key: jax.Array | None = None,
    extra_data: tuple[tuple[str, Any], ...] = (),
) -> PlannedTask | Refusal:
    """Compile a task against a graph: a :class:`PlannedTask`, or a Refusal.

    The refusals are verdicts a caller can branch on -- a task kind this
    release does not answer, a backend it cannot dispatch to, an option it
    would otherwise ignore, an operator it cannot identify, a point estimate
    with no solve to be the estimate of. What is NOT a refusal is a fault: a
    :class:`~bayesmith.errors.GraphError` or
    :class:`~bayesmith.errors.StructureError` raised while the graph is read
    travels out of here as an exception, because a contradicted declaration is
    a broken model rather than a route that does not apply.

    Args:
        graph: the model.
        task: one of the five §0.4 tasks.
        model_ref: what the graph was built from. Never the callable itself (§0
            ruling 4); see :func:`model_identity_gap` for when its
            ``source_digest`` is required rather than optional.
        key: PRNG key for the compile-time probes, passed through to
            :func:`~bayesmith.dispatch.plan.compile`. ``None`` leaves that
            function's own default in place, so a task compiled without a key
            and a graph compiled directly agree block for block.
        extra_data: data the caller conditions on that the graph does not
            carry, hashed into the DATA slot.
    """
    _check(graph, Graph, "compile_task's graph")
    _check(model_ref, ModelRef, "compile_task's model_ref")
    kind = task_kind(task)
    bundle = input_fingerprints(graph, task, model_ref=model_ref, extra_data=extra_data)
    refusal = _refuse_before_compiling(graph, task, kind, model_ref, bundle)
    if refusal is not None:
        return refusal

    runtime = compile_plan(graph) if key is None else compile_plan(graph, key=key)

    if kind is TaskKind.POINT_ESTIMATE and task.estimand is Estimand.POSTERIOR_MEAN:
        refusal = _estimate_refusal(runtime, task, bundle)
        if refusal is not None:
            return refusal

    analysis = _analysis_report(runtime, task, model_ref, bundle)
    return PlannedTask(
        task=task,
        analysis=analysis,
        record=_plan_record(runtime, task, kind, model_ref, bundle, analysis),
        runtime_plan=runtime,
    )

# --------------------------------------------------------------- execute_task


def _named(graph: Graph, name: str, value: Any, *, draw: bool) -> NamedArray:
    return NamedArray(
        name=name, value=np.asarray(value), dims=_dims(graph, name, value, draw=draw)
    )


def _measured(value: float | None) -> float | None:
    """A diagnostic that was taken, or ``None`` where it could not be.

    PSIS answers ``inf`` when there is no tail for it to fit -- measured on
    ``radiometer`` at eight draws, with numpyro saying so in a warning -- and
    §0.5 keeps "was not computed" as ``None`` rather than as a number that
    compares like one. Nothing about whether to believe the weights is lost by
    this: ``unreliable`` is a stored verdict taken at the run's own threshold,
    and it stays ``True`` exactly where it was.
    """
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _seed_record(key: jax.Array | None) -> SeedRecord | None:
    """The entropy the run started from, read off the key itself.

    ``None`` where no key was consumed: a point estimate splits nothing, and a
    zero written into the field would read as a seed somebody chose.
    """
    if key is None:
        return None
    seed = 0
    for word in np.asarray(jax.random.key_data(key)).ravel().tolist():
        seed = (seed << 32) | int(word)
    return SeedRecord(seed=seed, key_algorithm=str(jax.random.key_impl(key)))


def _devices(values: list[Any]) -> tuple[DeviceRecord, ...]:
    """The devices the produced arrays actually live on.

    Measured off the arrays rather than taken from ``jax.devices()``, which
    lists what was AVAILABLE. They agree on a single-device host and stop
    agreeing on the first machine with two, which is the case the record
    exists for.
    """
    found: dict[tuple[str, str, int], None] = {}
    for value in values:
        reader = getattr(value, "devices", None)
        if reader is None:
            continue
        for device in reader():
            found[(device.platform, device.device_kind, int(device.id))] = None
    if not found:
        for device in jax.devices():
            found[(device.platform, device.device_kind, int(device.id))] = None
    return tuple(
        DeviceRecord(platform=platform_name, device_kind=kind, device_id=index)
        for platform_name, kind, index in sorted(found)
    )


def _dtype(values: list[Any]) -> str:
    """The dtype the produced values promote to, as one answer.

    Promotion rather than the first array's dtype: a run whose sites disagree
    has one working precision and it is the wider of them, and reporting the
    first would make the answer depend on a name's alphabetical position.
    """
    return str(np.result_type(*[np.asarray(value).dtype for value in values]))


def _jax_config() -> tuple[tuple[str, Any], ...]:
    return (
        ("jax_enable_x64", bool(jax.config.jax_enable_x64)),
        ("jax_default_backend", jax.default_backend()),
    )


def _environment_fingerprint():
    """The ENVIRONMENT slot: the interpreter and the stack that ran.

    In no row of the §0.3 invalidation matrix on purpose -- a backend patch
    leaves stored artifacts readable and gives the NEXT run new provenance --
    so this is recorded and never used to retire anything.
    """
    return fingerprint(
        FingerprintKind.ENVIRONMENT,
        {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpyro": numpyro.__version__,
            "bayesmith": __version__,
            "x64": bool(jax.config.jax_enable_x64),
            "backend": jax.default_backend(),
        },
    )


def _backend_ref(chained: bool) -> BackendRef:
    """Which package's machinery actually ran, never the policy value auto.

    A chain is numpyro's kernel -- bare NUTS, the Gibbs sweep and the collapse
    arm all run through it -- and everything else is this package's own exact
    solver or optimiser.
    """
    if chained:
        return BackendRef(name="numpyro", version=numpyro.__version__)
    return BackendRef(name="bayesmith", version=__version__)


def _ran_a_chain(runtime: InferencePlan, posterior: Posterior) -> bool:
    """Whether these draws came from a chain, read off the PLAN and not the
    method string.

    The method string cannot answer it: :func:`~bayesmith.dispatch.execute._swept`
    reports the exact block's own method, so a MIXED plan whose exact block is
    plain ``gcr`` comes back labelled ``"gcr"`` having run an HMCGibbs chain,
    while a whole-graph ``gcr`` draws iid under the same label. What separates
    them is the plan: a sampled block means a chain, and the one path that
    grows a chain without one is the collapse substitution, which does say so
    in its method.
    """
    return posterior.method == "nuts" or runtime.sampled is not None


def _planned_method(runtime: InferencePlan) -> str:
    return runtime.exact.method if runtime.exact is not None else "nuts"


def _or_default(name: str, value: Any) -> Any:
    return _SAMPLE_DEFAULTS[name] if value is None else value


def _sample_settings(task: Task) -> dict[str, Any]:
    """The task's budget and knobs as :meth:`InferencePlan.sample` keywords.

    A field the task left empty is filled from the runtime's own default and
    then passed explicitly, so the run record can say what actually ran
    without a second copy of the number: ``_SAMPLE_DEFAULTS`` reads the
    signature that owns it.
    """
    budget = task.budget
    settings: dict[str, Any] = {
        "num_samples": _or_default("num_samples", budget.draws),
        "num_warmup": _or_default("num_warmup", budget.warmup),
        "num_chains": _or_default("num_chains", budget.chains),
        "nuts_on_collapse": task.nuts_on_collapse,
    }
    if task.chain_method is not None:
        settings["chain_method"] = task.chain_method
    if task.solver_tolerance is not None:
        settings["tol"] = task.solver_tolerance
    if task.solver_maxiter is not None:
        settings["maxiter"] = task.solver_maxiter
    if task.ess_floor is not None:
        settings["ess_floor"] = task.ess_floor
    options = dict(task.backend_options)
    for name in sorted(_POSTERIOR_OPTIONS):
        if name in options:
            settings[name] = options[name]
    return settings


def _sample_termination(
    posterior: Posterior, chained: bool
) -> TerminationRecord:
    """Why the run stopped, which is not the same question as whether it worked.

    A weighted sample that crossed its floor is ``TOLERANCE_UNMET``: the
    correction collapsed and the draws are annotated rather than replaced. A
    chain is ``CONVERGED`` only where every site's own diagnostic says so --
    the diagnostic that was actually taken, not a re-judgement of it here --
    and ``COMPLETED`` otherwise, which is the honest word for "it ran to the
    end of its budget".
    """
    if posterior.log_weights is not None and posterior.unreliable:
        return TerminationRecord(
            reason=TerminationReason.TOLERANCE_UNMET, message=posterior.reason
        )
    if chained:
        report = posterior.diagnostics
        if report and all(site.converged for site in report.values()):
            return TerminationRecord(
                reason=TerminationReason.CONVERGED, message=posterior.reason
            )
    return TerminationRecord(
        reason=TerminationReason.COMPLETED, message=posterior.reason
    )


def _sample_warnings(
    task: Task,
    runtime: InferencePlan,
    posterior: Posterior,
    chained: bool,
    termination: TerminationRecord,
) -> tuple[RunWarning, ...]:
    """What the run noticed and did not fail over.

    Three separate statements, kept separate: the executed method is not the
    planned one; a chain's sites could not be certified; and the TASK asked for
    convergence and did not get a certificate. The last is the task's own
    requirement rather than a verdict about the draws, so it is a warning here
    and not a Refusal -- the draws exist and are handed back.
    """
    warnings: list[RunWarning] = []
    planned_method = _planned_method(runtime)
    if posterior.method != planned_method:
        warnings.append(
            RunWarning(
                code="executed_method_differs_from_plan",
                message=posterior.reason,
                scope=planned_method,
            )
        )
    if chained:
        if posterior.diagnostics is None:
            warnings.append(
                RunWarning(
                    code="chain_diagnostics_unavailable",
                    message="split r-hat needs at least four draws per chain, so "
                    "this run reports the effective sample size alone",
                )
            )
        else:
            for name, site in posterior.diagnostics.items():
                if not site.converged:
                    warnings.append(
                        RunWarning(
                            code="chain_not_converged",
                            message=site.reason,
                            scope=name,
                        )
                    )
    certified = termination.reason is TerminationReason.CONVERGED
    if task.require_convergence and not certified:
        warnings.append(
            RunWarning(
                code="convergence_not_certified",
                message="the task asked for convergence and this run stopped "
                f"{termination.reason.value}; the draws are handed back with "
                "that said rather than withheld",
            )
        )
    return tuple(warnings)


def _run_record(
    planned: PlannedTask,
    *,
    key: jax.Array | None,
    values: list[Any],
    budget: ComputeBudget,
    termination: TerminationRecord,
    timing: TimingRecord,
    approximation: ApproximationRecord,
    warnings: tuple[RunWarning, ...],
    chained: bool,
) -> RunRecord:
    return RunRecord(
        run_id=str(uuid.uuid4()),
        plan_ref=plan_ref(planned.record),
        fingerprints=dataclasses.replace(
            planned.record.meta.fingerprints, environment=_environment_fingerprint()
        ),
        seed=_seed_record(key),
        dtype=_dtype(values),
        devices=_devices(values),
        jax_config=_jax_config(),
        backend=_backend_ref(chained),
        budget=budget,
        termination=termination,
        timing=timing,
        approximation=approximation,
        warnings=warnings,
    )


def _result_meta(planned: PlannedTask, run: RunRecord, summary: str):
    """A Result's envelope: the plan it came from is its PARENT, by version.

    The same reference the run record carries, because a result whose lineage
    and whose run disagreed about which plan produced it could not be
    invalidated by a change to either.
    """
    return new_artifact_meta(
        artifact_type=ArtifactKind.RESULT,
        fingerprints=run.fingerprints,
        producer=PRODUCER,
        parent_refs=(plan_ref(planned.record),),
        summary=summary,
    )


def _chain_diagnostics_report(
    diagnostics: Mapping[str, SiteDiagnostic],
    *,
    subject_ref: ArtifactRef,
    fingerprints: FingerprintBundle,
) -> EvaluationReport:
    """The per-site chain diagnostics, projected into one evaluation report.

    This does NOT re-judge convergence: the verdict already lives in each
    SiteDiagnostic, and this function only projects it into a serialisable
    EvaluationReport. Re-deriving it here would make the verdict answer to two
    callers -- the one that measured it and the one that filed it -- which is
    the defect this package has spent the most time repairing.

    One Finding per site. The six values a consumer needs to audit the verdict
    are all carried:

    * message is the site reason, or "converged" where that is empty (a
      converged site has no reason, and Finding.message may not be empty).
    * observed is the 6-tuple (name, r_hat, ess, ceiling, worst, converged):
      the site name, the measured pair and ceiling, the deciding coordinate,
      and the stored verdict. worst is itself a tuple, so a scalar site
      empty coordinate survives the codec unchanged.
    * expected is True: the premise a convergence check requires.
    """
    findings = tuple(
        Finding(
            code="chain_diagnostics",
            message=site.reason or "converged",
            observed=(
                name,
                site.r_hat,
                site.ess,
                site.ceiling,
                tuple(site.worst),
                site.converged,
            ),
            expected=True,
        )
        for name, site in diagnostics.items()
    )
    converged = all(site.converged for site in diagnostics.values())
    return EvaluationReport(
        meta=new_artifact_meta(
            artifact_type=ArtifactKind.EVALUATION_REPORT,
            fingerprints=fingerprints,
            producer=PRODUCER,
            parent_refs=(subject_ref,),
            summary="split r-hat and ESS, one finding per sampled site",
        ),
        subject_ref=subject_ref,
        report_kind="chain_diagnostics",
        applicability=Applicability.APPLICABLE,
        conclusion=Conclusion.PASS if converged else Conclusion.FAIL,
        findings=findings,
    )

def _timing(started: str, finished: str, elapsed: float) -> TimingRecord:
    return TimingRecord(
        started_at=started, finished_at=finished, wall_clock_seconds=elapsed
    )


def _run_posterior(planned: PlannedTask, key: jax.Array | None) -> Result:
    task = planned.task
    if key is None:
        raise TypeError(
            "a posterior run needs a PRNG key: execute_task(planned, "
            "key=jax.random.key(0)). Minting one here would make the run "
            "irreproducible while looking exactly like one that was seeded."
        )
    runtime = planned.runtime_plan
    settings = _sample_settings(task)

    started = utc_timestamp()
    clock = time.perf_counter()
    posterior = runtime.sample(key, **settings)
    elapsed = time.perf_counter() - clock
    finished = utc_timestamp()

    graph = runtime.graph
    names = tuple(name for name in graph.latents if name in posterior.samples)
    values = [posterior.samples[name] for name in names]
    draws = tuple(
        _named(graph, name, value, draw=True)
        for name, value in zip(names, values, strict=True)
    )
    chained = _ran_a_chain(runtime, posterior)

    pointwise = None
    if graph.observed:
        try:
            pointwise = pointwise_log_likelihood(graph, posterior.samples)
        except NotGaussian:
            # A correlated or non-Gaussian observation has no diagonal loc/scale
            # to replay; the result ABSTAINs rather than fabricating a pointwise
            # density (§0.4).
            pointwise = None
    availability = (
        LogDensityAvailability.POINTWISE
        if pointwise is not None
        else LogDensityAvailability.NONE
    )
    predictive_ready = pointwise is not None

    if posterior.log_weights is None:
        representation = DrawsPosterior(
            draws=draws,
            chain_shape=(settings["num_chains"], settings["num_samples"])
            if chained
            else None,
            method=posterior.method,
        )
    else:
        representation = WeightedDrawsPosterior(
            draws=draws,
            log_weights=NamedArray(
                name="log_weights",
                value=np.asarray(posterior.log_weights),
                dims=("draw",),
            ),
            ess=float(posterior.ess),
            khat=_measured(posterior.khat),
            unreliable=bool(posterior.unreliable),
            method=posterior.method,
        )

    termination = _sample_termination(posterior, chained)
    run = _run_record(
        planned,
        key=key,
        values=values,
        budget=ComputeBudget(
            draws=settings["num_samples"],
            warmup=settings["num_warmup"],
            chains=settings["num_chains"],
            max_iterations=task.solver_maxiter,
        ),
        termination=termination,
        timing=_timing(started, finished, elapsed),
        approximation=ApproximationRecord(
            representation_class=ApproximationClass.MONTE_CARLO,
            target_fidelity=TargetFidelity.EXACT,
            details=(
                ("method", posterior.method),
                ("planned_method", _planned_method(runtime)),
                ("requested_backend", task.backend),
                # The one measurement §0.5 gives no field to on three of the
                # four representation arms. The weighted arm DOES have one, and
                # the two are held equal by test_task_execution.py rather than
                # left to agree by habit.
                ("effective_sample_size", float(posterior.ess)),
            ),
        ),
        warnings=_sample_warnings(task, runtime, posterior, chained, termination),
        chained=chained,
    )
    meta = _result_meta(planned, run, posterior.reason)
    report_refs = ()
    if posterior.diagnostics:
        subject_ref = _ref(meta.artifact_id, meta.revision, ArtifactKind.RESULT)
        report = _chain_diagnostics_report(
            posterior.diagnostics,
            subject_ref=subject_ref,
            fingerprints=run.fingerprints,
        )
        report_refs = (
            _ref(
                report.meta.artifact_id,
                report.meta.revision,
                ArtifactKind.EVALUATION_REPORT,
            ),
        )
    return PosteriorResult(
        meta=meta,
        run=run,
        representation=representation,
        latent_names=names,
        log_density_availability=availability,
        pointwise_log_likelihood=pointwise,
        predictive_ready=predictive_ready,
        report_refs=report_refs,
    )


def _selected(values: dict[str, Any], task: Task, graph: Graph) -> list[str]:
    """The latents this task asked to be reported, in declaration order."""
    wanted = set(values) if task.names is None else set(task.names)
    return [name for name in graph.latents if name in wanted]


def _run_posterior_mean(planned: PlannedTask) -> Result:
    task = planned.task
    options = dict(task.optimizer_options)
    settings: dict[str, Any] = {}
    if task.budget.max_iterations is not None:
        settings["maxiter"] = task.budget.max_iterations
    if "tolerance" in options:
        settings["tol"] = options["tolerance"]

    started = utc_timestamp()
    clock = time.perf_counter()
    estimate = planned.runtime_plan.estimate(**settings)
    elapsed = time.perf_counter() - clock
    finished = utc_timestamp()

    graph = planned.runtime_plan.graph
    names = _selected(estimate.values, task, graph)
    values = [estimate.values[name] for name in names]
    iterations = int(estimate.iterations)
    run = _run_record(
        planned,
        key=None,
        values=values,
        budget=ComputeBudget(max_iterations=task.budget.max_iterations),
        termination=TerminationRecord(
            reason=TerminationReason.CONVERGED
            if bool(estimate.converged)
            else TerminationReason.TOLERANCE_UNMET,
            iterations=iterations,
            message="the generalised least squares solve reached its fixed point",
        ),
        timing=_timing(started, finished, elapsed),
        approximation=ApproximationRecord(
            representation_class=ApproximationClass.CERTIFIED_DETERMINISTIC,
            target_fidelity=TargetFidelity.EXACT,
            details=(
                ("method", _planned_method(planned.runtime_plan)),
                ("requested_backend", task.backend),
            ),
        ),
        warnings=(),
        chained=False,
    )
    return PointEstimateResult(
        meta=_result_meta(planned, run, "the posterior mean of a whole-graph solve"),
        run=run,
        estimand=task.estimand,
        values=tuple(
            _named(graph, name, value, draw=False)
            for name, value in zip(names, values, strict=True)
        ),
        residual=float(estimate.residual),
        iterations=iterations,
    )


#: How each MAP seam's answer is produced, on §0.2's two axes. The Newton
#: route checks stationarity AND curvature before it returns anything, so its
#: point is a certified deterministic answer to the exact question; a descent
#: spends a fixed budget and certifies nothing, which is HEURISTIC of an
#: APPROXIMATE target however good the number turns out to be.
_MAP_APPROXIMATION = {
    "newton": (ApproximationClass.CERTIFIED_DETERMINISTIC, TargetFidelity.EXACT),
    "adam": (ApproximationClass.HEURISTIC, TargetFidelity.APPROXIMATE),
    "gradient": (ApproximationClass.HEURISTIC, TargetFidelity.APPROXIMATE),
}


def _run_map(planned: PlannedTask) -> Result | Refusal:
    task = planned.task
    options = dict(task.optimizer_options)
    method = _map_method(task)
    graph = planned.runtime_plan.graph

    started = utc_timestamp()
    clock = time.perf_counter()
    if method == MAP_METHODS[0]:
        outcome = map_estimate(graph)
    else:
        settings: dict[str, Any] = {"method": method}
        if task.budget.max_iterations is not None:
            settings["steps"] = task.budget.max_iterations
        if "learning_rate" in options:
            settings["learning_rate"] = options["learning_rate"]
        outcome = fit(graph, **settings)
    elapsed = time.perf_counter() - clock
    finished = utc_timestamp()

    if isinstance(outcome, (Refused, NotApplicable)):
        return refusal_from_verdict(
            outcome,
            task=task,
            model_ref=planned.record.model_ref,
            fingerprints=planned.record.meta.fingerprints,
            artifact_type=ArtifactKind.RESULT,
        )

    found = outcome.point if isinstance(outcome, MapEstimate) else outcome.values
    names = _selected(found, task, graph)
    values = [found[name] for name in names]
    iterations = (
        int(outcome.steps)
        if isinstance(outcome, MapEstimate)
        else int(np.shape(outcome.history)[0])
    )
    representation, fidelity = _MAP_APPROXIMATION[method]
    run = _run_record(
        planned,
        key=None,
        values=values,
        budget=ComputeBudget(max_iterations=task.budget.max_iterations),
        termination=TerminationRecord(
            reason=TerminationReason.CONVERGED
            if isinstance(outcome, MapEstimate)
            else TerminationReason.BUDGET_EXHAUSTED,
            iterations=iterations,
            message="a stationary point with positive curvature"
            if isinstance(outcome, MapEstimate)
            else "the descent spent its step budget, which is not a convergence "
            "claim",
        ),
        timing=_timing(started, finished, elapsed),
        approximation=ApproximationRecord(
            representation_class=representation,
            target_fidelity=fidelity,
            details=(("method", method), ("requested_backend", task.backend)),
        ),
        warnings=(),
        chained=False,
    )
    return PointEstimateResult(
        meta=_result_meta(planned, run, "a local mode of the graph's posterior"),
        run=run,
        estimand=task.estimand,
        values=tuple(
            _named(graph, name, value, draw=False)
            for name, value in zip(names, values, strict=True)
        ),
        objective=float(outcome.objective),
        gradient_norm=(
            float(outcome.gradient_norm) if isinstance(outcome, MapEstimate) else None
        ),
        iterations=iterations,
        # An optimiser reports a stationary point. Calling it THE mode would be
        # a global claim nothing here took.
        local_only=True,
    )


def _observation_unit(graph: Graph) -> str | None:
    """The observation-unit declaration: which observed nodes these are."""
    return ", ".join(graph.observed) if graph.observed else None


def _grouping(graph: Graph) -> str | None:
    """The plate/group name shared by the observed nodes, or ``None``."""
    plates: set[str] = set()
    for name in graph.observed:
        plates.update(graph.node(name).plate)
    return next(iter(plates)) if len(plates) == 1 else None


def _posterior_source_refusal(
    task: Task,
    kind: TaskKind,
    expected: FingerprintBundle,
    source_posterior: PosteriorResult,
) -> Refusal | None:
    """The §0.6 premise BOTH forward tasks rest on: this posterior is of this
    model, this graph and this conditioning data.

    Shared by :func:`_run_predictive` and :func:`_run_simulation` rather than
    copied into the second, because the two ask the same question and a second
    copy of a premise is how two answers to one question get written. Nothing
    about the predictive path's behaviour moves: the same three slots in the
    same order, and ``kind.value`` reads "predictive" there exactly as the
    literal it replaced did.
    """
    source = source_posterior.meta.fingerprints
    mismatched = tuple(
        slot.value
        for slot, left, right in (
            (FingerprintKind.DATA, source.data, expected.data),
            (
                FingerprintKind.GRAPH_STRUCTURE,
                source.graph_structure,
                expected.graph_structure,
            ),
            (FingerprintKind.MODEL_SOURCE, source.model_source, expected.model_source),
        )
        if left != right
    )
    if not mismatched:
        return None
    return _refusal(
        task,
        artifact_type=ArtifactKind.RESULT,
        fingerprints=expected,
        failed_premise="posterior_data_mismatch",
        grounds=(
            Finding(
                code="posterior_data_mismatch",
                message="the source posterior was drawn from a different "
                f"model, graph or conditioning data than this {kind.value} task "
                "names, so its draws cannot be pushed forward against this "
                "graph",
                observed=mismatched,
                expected=("data", "graph_structure", "model_source"),
            ),
        ),
        scope=_scope(ScopeKind.DATA, "source_posterior"),
        summary=f"source posterior fingerprints disagree on {list(mismatched)}",
    )


def _noise_refusal(
    task: Task, expected: FingerprintBundle, exc: NotGaussian, seam: str
) -> Refusal:
    """§0.4's coverage domain, adapted once for both forward tasks.

    ``observation_parts`` is a diagonal walk, so a correlated or non-Gaussian
    observed node raises out of the primitives rather than being quietly
    approximated. The code comes from the exception's structured ``reason``
    field, never from its prose.
    """
    return _refusal(
        task,
        artifact_type=ArtifactKind.RESULT,
        fingerprints=expected,
        failed_premise="predictive_noise_unsupported",
        grounds=(
            Finding(
                code=f"not_gaussian.{exc.reason}",
                message=str(exc),
                observed=exc.found,
                expected="diagonal_normal",
            ),
        ),
        scope=(node_scope(exc.node) if exc.node else _scope(ScopeKind.MODEL, seam)),
        summary="predictive generation needs a diagonal-Gaussian observation",
    )


def _run_predictive(
    planned: PlannedTask,
    key: jax.Array | None,
    source_posterior: PosteriorResult | None,
) -> Result | Refusal:
    """Push a source posterior's draws onto the graph's observations (§0.1).

    The source posterior is the caller's to supply -- there is no artifact store
    here -- so its id and revision are checked against the task's reference
    first, and its data/graph/model fingerprints against this task's own (§0.6)
    before any number is generated.  A correlated or non-Gaussian observed node
    raises :class:`~bayesmith.errors.NotGaussian` out of the primitives, which is
    adapted into the typed :data:`predictive_noise_unsupported` Refusal rather
    than a silent approximation.
    """
    task = planned.task
    if key is None:
        raise TypeError(
            "a predictive run needs a PRNG key: execute_task(planned, key=..., "
            "source_posterior=...). Minting one here would make the run "
            "irreproducible while looking exactly like one that was seeded."
        )
    if source_posterior is None:
        raise TypeError(
            "a predictive task names a source posterior and this release has no "
            "artifact store to load one from; pass it as source_posterior=..."
        )
    _check(source_posterior, PosteriorResult, "execute_task's source_posterior")

    reference = task.source_posterior_ref
    if (
        source_posterior.meta.artifact_id != reference.artifact_id
        or source_posterior.meta.revision != reference.revision
    ):
        raise TypeError(
            "the supplied source posterior is not the version the task's "
            "source_posterior_ref names; pass the posterior that reference "
            "points at rather than another one"
        )

    expected = planned.record.meta.fingerprints
    mismatch = _posterior_source_refusal(
        task, TaskKind.PREDICTIVE, expected, source_posterior
    )
    if mismatch is not None:
        return mismatch

    representation = source_posterior.representation
    if not isinstance(representation, (DrawsPosterior, WeightedDrawsPosterior)):
        raise TypeError(
            "a predictive task needs a source posterior that holds draws; this "
            f"one is a {type(representation).__name__}"
        )

    graph = planned.runtime_plan.graph
    unknown = tuple(name for name in task.replicated_sites if name not in graph.observed)
    if unknown:
        raise TypeError(
            f"replicated_sites names {list(unknown)}, which are not observed "
            f"nodes of this graph; its observed nodes are {list(graph.observed)}"
        )

    latent_values = {array.name: array.value for array in representation.draws}
    draw_count = len(representation.draws[0].value) if representation.draws else 0

    started = utc_timestamp()
    clock = time.perf_counter()
    try:
        replicated = replicated_draws(graph, latent_values, key)
        pointwise = pointwise_log_likelihood(graph, latent_values)
    except NotGaussian as exc:
        return _noise_refusal(task, expected, exc, "predictive")
    elapsed = time.perf_counter() - clock
    finished = utc_timestamp()

    replicated_named = tuple(
        _named(graph, name, replicated[name], draw=True)
        for name in task.replicated_sites
    )
    carried = tuple(
        array for array in representation.draws if array.name in set(task.latent_sites)
    )

    chained = source_posterior.run.backend.name == "numpyro"
    run = _run_record(
        planned,
        key=key,
        values=list(replicated.values()),
        budget=ComputeBudget(draws=draw_count),
        termination=TerminationRecord(
            reason=TerminationReason.COMPLETED,
            message="replicated draws generated from the source posterior's draws",
        ),
        timing=_timing(started, finished, elapsed),
        approximation=ApproximationRecord(
            representation_class=ApproximationClass.MONTE_CARLO,
            target_fidelity=TargetFidelity.EXACT,
            details=(
                ("method", representation.method),
                ("requested_backend", task.backend),
            ),
        ),
        warnings=(),
        chained=chained,
    )
    return PredictiveResult(
        meta=_result_meta(planned, run, "posterior predictive of the observations"),
        run=run,
        source_posterior_ref=task.source_posterior_ref,
        conditioning_data=task.conditioning_data,
        prediction_design=task.prediction_design,
        conditioned_sites=task.conditioned_sites,
        latent_draws=carried,
        replicated_draws=replicated_named,
        pointwise_log_density=pointwise,
        observation_unit=_observation_unit(graph),
        grouping=_grouping(graph),
        report_refs=(),
    )


def _check_simulation_sites(graph: Graph, task: Task) -> None:
    """The task's site names, checked against the graph before anything runs.

    A typo in ``observed_sites`` would otherwise come back as an empty result
    that validates, and a caller reading ``observation_draws`` would find the
    node they asked for simply absent rather than misspelled.
    """
    for field, declared in (
        ("latent_sites", graph.latents),
        ("observed_sites", graph.observed),
    ):
        unknown = tuple(
            name for name in getattr(task, field) if name not in declared
        )
        if unknown:
            raise TypeError(
                f"{field} names {list(unknown)}, which this graph does not "
                f"declare; its are {list(declared)}"
            )


def _run_simulation(
    planned: PlannedTask,
    key: jax.Array | None,
    source_posterior: PosteriorResult | None,
) -> Result | Refusal:
    """Generate forward from one of the three parameter sources (§0.7).

    The three arms differ only in where the parameters come from, and they
    reach the observations through the SAME two primitives R2 built:
    ``PRIOR`` walks the graph with :func:`~bayesmith.dispatch.predictive.prior_draws`
    (observed nodes included, so it is the prior predictive), ``FIXED`` pushes
    one setting through :func:`~bayesmith.dispatch.predictive.forward_draws`,
    and ``POSTERIOR_RESULT`` pushes a source posterior's draws through
    :func:`~bayesmith.dispatch.predictive.replicated_draws` -- the same call
    with the same key that :func:`_run_predictive` makes, which is why the two
    tasks answer with the same bits and why a test says so at ``rtol=0``.

    **``latent_draws`` holds latents that were DRAWN.** A fixed source draws
    none: its parameters have no draw axis to share with the observations, so
    they travel in ``parameters`` instead, and naming one in ``latent_sites``
    yields nothing rather than a repeated column pretending to be a sample.
    """
    task = planned.task
    if key is None:
        raise TypeError(
            "a simulation run needs a PRNG key: execute_task(planned, "
            "key=jax.random.key(0)). Minting one here would make the run "
            "irreproducible while looking exactly like one that was seeded."
        )
    graph = planned.runtime_plan.graph
    _check_simulation_sites(graph, task)
    source = task.parameter_source
    expected = planned.record.meta.fingerprints

    chained = False
    details: tuple[tuple[str, Any], ...] = (
        ("parameter_source", source.kind.value),
        ("requested_backend", task.backend),
    )
    latents: dict[str, Any] = {}

    if source.kind is ParameterSourceKind.POSTERIOR_RESULT:
        if source_posterior is None:
            raise TypeError(
                "a simulation task drawing from a posterior result names the "
                "artifact it draws from, and this release has no artifact "
                "store to load one from; pass it as source_posterior=..."
            )
        _check(source_posterior, PosteriorResult, "execute_task's source_posterior")
        reference = source.posterior_ref
        if (
            source_posterior.meta.artifact_id != reference.artifact_id
            or source_posterior.meta.revision != reference.revision
        ):
            raise TypeError(
                "the supplied source posterior is not the version the task's "
                "parameter source names; pass the posterior that reference "
                "points at rather than another one"
            )
        mismatch = _posterior_source_refusal(
            task, TaskKind.SIMULATION, expected, source_posterior
        )
        if mismatch is not None:
            return mismatch
        representation = source_posterior.representation
        if not isinstance(representation, (DrawsPosterior, WeightedDrawsPosterior)):
            raise TypeError(
                "a simulation task reading a posterior needs one that holds "
                f"draws; this one is a {type(representation).__name__}"
            )
        latents = {array.name: array.value for array in representation.draws}
        count = len(representation.draws[0].value) if representation.draws else 0
        chained = source_posterior.run.backend.name == "numpyro"
        details = (*details, ("source_method", representation.method))
    else:
        count = task.budget.draws
        if count is None:
            raise TypeError(
                "a simulation task's draw count is its budget.draws, and this "
                "task's budget names none. This module decides no number, and "
                "unlike a sampler's draw count there is no runtime signature "
                "here that owns a default to be read off -- so a value chosen "
                "here would be one nobody asked for, recorded as though they "
                "had."
            )

    started = utc_timestamp()
    clock = time.perf_counter()
    try:
        if source.kind is ParameterSourceKind.PRIOR:
            env = prior_draws(graph, key, count)
            latents = {name: env[name] for name in graph.latents}
            observations = {name: env[name] for name in graph.observed}
        elif source.kind is ParameterSourceKind.FIXED:
            observations = forward_draws(
                graph,
                {array.name: jnp.asarray(array.value) for array in source.values},
                key,
                count,
            )
        else:
            observations = replicated_draws(graph, latents, key)
    except NotGaussian as exc:
        return _noise_refusal(task, expected, exc, "simulation")
    elapsed = time.perf_counter() - clock
    finished = utc_timestamp()

    reported = [name for name in task.latent_sites if name in latents]
    latent_named = tuple(
        _named(graph, name, latents[name], draw=True) for name in reported
    )
    observation_named = tuple(
        _named(graph, name, observations[name], draw=True)
        for name in task.observed_sites
    )
    if not latent_named and not observation_named:
        # Reachable, and the artifact's own validation would say something
        # true but unhelpful here ("holding neither is a result with no
        # content") because `_dtype` runs first and fails on an empty list.
        # The combination that gets here is a FIXED source naming only
        # latent_sites, which is a request for draws of the values the caller
        # just fixed.
        raise TypeError(
            f"this simulation would hold no draws: a {source.kind.value} "
            f"parameter source draws {sorted(latents)}, and the task named "
            f"latent_sites={list(task.latent_sites)} and "
            f"observed_sites={list(task.observed_sites)}. A fixed source "
            "draws no latents -- the values it fixes travel in `parameters` "
            "-- so a fixed simulation names at least one observed site."
        )

    run = _run_record(
        planned,
        key=key,
        # The live arrays, not the NamedArrays' numpy copies: `_devices` reads
        # `.devices()` off each value and falls back to "what was available"
        # when nothing answers, which is the very distinction that record
        # exists to make.
        values=[latents[name] for name in reported]
        + [observations[name] for name in task.observed_sites],
        budget=ComputeBudget(draws=count),
        termination=TerminationRecord(
            reason=TerminationReason.COMPLETED,
            message="forward draws generated from the "
            f"{source.kind.value} parameter source",
        ),
        timing=_timing(started, finished, elapsed),
        approximation=ApproximationRecord(
            representation_class=ApproximationClass.MONTE_CARLO,
            target_fidelity=TargetFidelity.EXACT,
            details=details,
        ),
        warnings=(),
        chained=chained,
    )
    return SimulationResult(
        meta=_result_meta(
            planned, run, f"forward draws from the {source.kind.value} source"
        ),
        run=run,
        parameter_source=source,
        parameters=source.values,
        latent_draws=latent_named,
        observation_draws=observation_named,
        prediction_design=task.prediction_design,
        report_refs=(),
    )


def execute_task(
    planned: PlannedTask,
    *,
    key: jax.Array | None = None,
    source_posterior: PosteriorResult | None = None,
) -> Result | Refusal:
    """Run a compiled task and project what came back into a Result.

    One call to one seam, with the key it was given and no extra split, timed
    from just before to just after. Nothing here decides a number: the method,
    the tolerance, the budget and the fallback policy were all fixed by
    :func:`compile_task` and by the plan it holds.

    Args:
        planned: what :func:`compile_task` produced.
        key: the PRNG key a posterior, predictive or simulation run draws
            from. Required there, and unused by a point estimate -- which
            splits nothing, so its run record carries no seed rather than one
            it never consumed.
        source_posterior: the :class:`~bayesmith.artifacts.results.PosteriorResult`
            a predictive task pushes forward. Required for a predictive task
            and for a simulation task whose parameter source is a posterior
            result; in both, its id/revision and its data/graph/model
            fingerprints are checked against the task's before anything is
            generated.

    Returns:
        A :class:`~bayesmith.artifacts.results.PosteriorResult`,
        :class:`~bayesmith.artifacts.results.PointEstimateResult`,
        :class:`~bayesmith.artifacts.results.PredictiveResult` or
        :class:`~bayesmith.artifacts.results.SimulationResult`, or a
        :class:`~bayesmith.artifacts.refusal.Refusal` where the method turned
        out not to apply. A genuine execution failure --
        :class:`~bayesmith.errors.ConvergenceError` above all -- is raised, not
        refused: a run that broke is an ERROR for a workflow to mark, and
        filing it as "this method does not apply" would lose that distinction.
    """
    _check(planned, PlannedTask, "execute_task's planned")
    kind = task_kind(planned.task)
    if kind is TaskKind.POSTERIOR:
        return _run_posterior(planned, key)
    if kind is TaskKind.POINT_ESTIMATE:
        if planned.task.estimand is Estimand.POSTERIOR_MEAN:
            return _run_posterior_mean(planned)
        return _run_map(planned)
    if kind is TaskKind.PREDICTIVE:
        return _run_predictive(planned, key, source_posterior)
    if kind is TaskKind.SIMULATION:
        return _run_simulation(planned, key, source_posterior)
    # compile_task never produces one of these; a PlannedTask assembled by
    # hand gets the same verdict rather than an execution that half-works.
    return _capability_refusal(
        planned.task, kind, planned.record.meta.fingerprints, ArtifactKind.RESULT
    )
