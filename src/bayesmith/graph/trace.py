"""Tracing a model function into an explicit graph.

The graph object is the ground truth; this module is sugar for producing one.
Tracing runs the model **once**, so the structure it records must not depend
on values -- which is exactly the static-DAG restriction the rest of the
package relies on.

Edges are declared by passing a :class:`NodeRef` returned by an earlier
primitive, not by inspecting argument names. That keeps ``parents`` explicit
and ordered, which is what ``fn`` and ``dist_fn`` are called with.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import jax.numpy as jnp

from bayesmith.errors import GraphError, TraceError
from bayesmith.graph.graph import Graph, Plate
from bayesmith.graph.nodes import Const, Deterministic, Node, Probabilistic


class NodeRef:
    """A handle to a declared node. Pass one to declare an edge."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"NodeRef({self.name!r})"


class _Recorder:
    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self.plates: list[Plate] = []
        self.names: set[str] = set()

    def add(self, node: Node) -> None:
        if node.name in self.names:
            raise GraphError(f"duplicate node name {node.name!r}")
        self.names.add(node.name)
        self.nodes.append(node)


_STACK: list[_Recorder] = []


def _active() -> _Recorder:
    if not _STACK:
        raise TraceError(
            "bayesmith tracing primitives must be called inside trace(model). "
            "Outside it there is no graph to record into."
        )
    return _STACK[-1]


def _parent_names(parents: Sequence[NodeRef]) -> tuple[str, ...]:
    for parent in parents:
        if not isinstance(parent, NodeRef):
            raise GraphError(
                f"parents must be NodeRef values returned by const/det/sample/"
                f"observe; got {type(parent).__name__}. Pass the handle, not the "
                "value."
            )
    return tuple(parent.name for parent in parents)


def _plate_names(plate_arg: str | Iterable[str] | None) -> tuple[str, ...]:
    if plate_arg is None:
        return ()
    if isinstance(plate_arg, str):
        return (plate_arg,)
    return tuple(plate_arg)


def plate(name: str, size: int) -> str:
    """Declare a repetition axis and return its name."""
    recorder = _active()
    for existing in recorder.plates:
        if existing.name == name:
            raise GraphError(f"duplicate plate name {name!r}")
    recorder.plates.append(Plate(name=name, size=int(size)))
    return name


def const(name: str, value: Any, *, plate: str | Iterable[str] | None = None) -> NodeRef:
    """Declare a fixed input node."""
    node = Const(
        name=name, parents=(), plate=_plate_names(plate), value=jnp.asarray(value)
    )
    _active().add(node)
    return NodeRef(name)


def det(
    name: str,
    fn: Callable[..., Any],
    *parents: NodeRef,
    linear_in: Iterable[str] = (),
    plate: str | Iterable[str] | None = None,
) -> NodeRef:
    """Declare a deterministic node: it propagates dependence, no density."""
    node = Deterministic(
        name=name,
        parents=_parent_names(parents),
        plate=_plate_names(plate),
        fn=fn,
        linear_in=tuple(linear_in),
    )
    _active().add(node)
    return NodeRef(name)


def sample(
    name: str,
    dist_fn: Callable[..., Any],
    *parents: NodeRef,
    plate: str | Iterable[str] | None = None,
) -> NodeRef:
    """Declare a latent probabilistic node."""
    node = Probabilistic(
        name=name,
        parents=_parent_names(parents),
        plate=_plate_names(plate),
        dist_fn=dist_fn,
        observed=None,
    )
    _active().add(node)
    return NodeRef(name)


def observe(
    name: str,
    dist_fn: Callable[..., Any],
    *parents: NodeRef,
    obs: Any,
    plate: str | Iterable[str] | None = None,
) -> NodeRef:
    """Declare a probabilistic node conditioned on data."""
    node = Probabilistic(
        name=name,
        parents=_parent_names(parents),
        plate=_plate_names(plate),
        dist_fn=dist_fn,
        observed=jnp.asarray(obs),
    )
    _active().add(node)
    return NodeRef(name)


def trace(model_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Graph:
    """Run ``model_fn`` once and return the graph it declared."""
    recorder = _Recorder()
    _STACK.append(recorder)
    try:
        model_fn(*args, **kwargs)
    finally:
        _STACK.pop()
    return Graph(nodes=tuple(recorder.nodes), plates=tuple(recorder.plates))
