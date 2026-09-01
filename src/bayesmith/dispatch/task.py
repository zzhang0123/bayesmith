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
from typing import Any

import jax
import numpy as np

from bayesmith import __version__
from bayesmith.artifacts.base import (
    ApproximationClass,
    ApproximationRecord,
    ArtifactRef,
    ProducerRef,
    TargetFidelity,
    new_artifact_meta,
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
    InferencePlanRecord,
    PlanBlockRecord,
)
from bayesmith.artifacts.tasks import (
    Estimand,
    Task,
    TaskKind,
    task_fingerprint,
    task_kind,
)
from bayesmith.diagnose.coupling import Refused
from bayesmith.diagnose.map import MapEstimate, NotApplicable
from bayesmith.dispatch.execute import _refuse_unless_whole_graph_exact
from bayesmith.dispatch.plan import Block, InferencePlan, kappa_upper
from bayesmith.dispatch.plan import compile as compile_plan
from bayesmith.errors import NotGaussian, NotLogLinear
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Const, Deterministic, Node, Probabilistic

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

#: The two of §0 ruling 1's five questions R1 answers. The other three are
#: refused with :data:`~bayesmith.artifacts.refusal.CAPABILITY_UNAVAILABLE_R1`,
#: which is a verdict a caller can branch on rather than a NotImplementedError.
SUPPORTED_TASK_KINDS: frozenset[TaskKind] = frozenset(
    {TaskKind.POSTERIOR, TaskKind.POINT_ESTIMATE}
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


def _known_options(task: Task, kind: TaskKind) -> frozenset[str]:
    if kind is TaskKind.POSTERIOR:
        return _POSTERIOR_OPTIONS
    return _MAP_OPTIONS if task.estimand is Estimand.MAP else _POSTERIOR_MEAN_OPTIONS


def _given_options(task: Task, kind: TaskKind) -> tuple[tuple[str, Any], ...]:
    if kind is TaskKind.POSTERIOR:
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
    if task_kind(task) is TaskKind.POSTERIOR:
        return ApproximationRecord(
            representation_class=ApproximationClass.MONTE_CARLO,
            target_fidelity=TargetFidelity.EXACT,
            details=(("method", method),),
        )
    if task.estimand is Estimand.POSTERIOR_MEAN:
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
            if dict(task.optimizer_options).get("method", "newton") == "newton":
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
        quality_gate=task.quality_gate,
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
