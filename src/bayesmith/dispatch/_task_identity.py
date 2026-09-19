"""Canonical Graph-to-artifact identity projection, without task execution.

This module owns structure/data manifests and their fingerprints. Public
imports remain available from ``dispatch.task``. No sampler, capability probe,
execution plan or run state participates in identifying the input model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bayesmith.artifacts.identity import (
    FingerprintBundle,
    FingerprintKind,
    ModelRef,
    fingerprint,
)
from bayesmith.artifacts.tasks import Task, task_fingerprint, task_kind
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Const, Deterministic, Node, Probabilistic

#: What an unidentifiable operator is written as. A constant, never an address:
#: two unidentifiable callables hash the same here, which is precisely why
#: :func:`model_identity_gap` requires the caller's own source digest before a
#: graph holding one may be compiled.
_UNIDENTIFIED = ("unidentified",)

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
