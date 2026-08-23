"""Tracing a model function into an explicit graph.

The graph object is the ground truth; this module is sugar for producing one.
Tracing runs the model **once**, so the structure it records must not depend
on values -- which is exactly the static-DAG restriction the rest of the
package relies on.

Edges are declared by passing a :class:`NodeRef` returned by an earlier
primitive, not by inspecting argument names. That keeps ``parents`` explicit
and ordered, which is what ``fn`` and ``dist_fn`` are called with. A
``NodeRef`` (and likewise a ``PlateRef``) is bound to the recorder of the
``trace(...)`` call that created it, so a handle from a different trace is
refused rather than silently resolving by name against whatever the active
trace happens to have declared -- see ``_parent_names`` and ``_plate_names``.

**Not thread-safe.** ``_STACK`` is a single process-global list. It supports
nested ``trace(...)`` calls on one thread (an inner trace pushes and pops its
own recorder around the outer one), but concurrent ``trace(...)`` calls from
multiple threads race on the same stack and can attribute one thread's nodes
to another thread's graph. Confine tracing to a single thread at a time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import jax.numpy as jnp

from bayesmith.errors import GraphError, TraceError
from bayesmith.graph.graph import Graph, Plate
from bayesmith.graph.nodes import Const, Deterministic, Node, Probabilistic


class NodeRef:
    """A handle to a declared node. Pass one to declare an edge.

    Bound to the recorder of the ``trace(...)`` call that created it. A
    parent must be a ``NodeRef`` minted by the *currently active* trace --
    enforced in ``_parent_names`` -- so a handle leaked in from an outer or
    an earlier trace is refused instead of silently resolving against a
    same-named node the active trace happens to declare.
    """

    __slots__ = ("_owner", "name")

    def __init__(self, name: str, owner: _Recorder) -> None:
        self.name = name
        self._owner = owner

    def __repr__(self) -> str:
        return f"NodeRef({self.name!r})"


class PlateRef:
    """A handle to a declared plate. Pass one as ``plate=`` to place a node.

    Owner-checked the same way as :class:`NodeRef`, for the same reason:
    without it, a plate name that merely *coincides* with one declared by a
    different (or an outer) trace would bind silently -- enforced in
    ``_plate_names``.
    """

    __slots__ = ("_owner", "name")

    def __init__(self, name: str, owner: _Recorder) -> None:
        self.name = name
        self._owner = owner

    def __repr__(self) -> str:
        return f"PlateRef({self.name!r})"


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


def _parent_names(
    parents: Sequence[NodeRef], recorder: _Recorder
) -> tuple[str, ...]:
    for parent in parents:
        if not isinstance(parent, NodeRef):
            raise GraphError(
                "parents must be NodeRef values returned by const/det/sample/"
                f"observe; got {type(parent).__name__}. Pass the handle, not "
                "the value."
            )
        if parent._owner is not recorder:
            raise GraphError(
                f"parent {parent.name!r} is a NodeRef from a different "
                "trace(...) call. A handle is only valid inside the "
                "trace(...) that created it -- pass a NodeRef this model just "
                "declared, not one captured from an outer or an earlier "
                "trace."
            )
    return tuple(parent.name for parent in parents)


def _plate_names(
    plate_arg: PlateRef | Iterable[PlateRef] | None, recorder: _Recorder
) -> tuple[str, ...]:
    if plate_arg is None:
        return ()
    if isinstance(plate_arg, PlateRef):
        candidates: tuple[Any, ...] = (plate_arg,)
    elif isinstance(plate_arg, Iterable) and not isinstance(plate_arg, (str, bytes)):
        candidates = tuple(plate_arg)
    else:
        # Not a PlateRef and not iterable (a bare str, an int, ...) -- let the
        # loop below reject it with one uniform, actionable message instead
        # of a bare TypeError from trying to iterate it.
        candidates = (plate_arg,)

    names: list[str] = []
    for ref in candidates:
        if not isinstance(ref, PlateRef):
            raise GraphError(
                "plate must be a PlateRef returned by plate(...), or an "
                f"iterable of them; got {type(ref).__name__}. Pass the handle "
                "plate() returned, not a bare name."
            )
        if ref._owner is not recorder:
            raise GraphError(
                f"plate {ref.name!r} is a PlateRef from a different "
                "trace(...) call. A handle is only valid inside the "
                "trace(...) that created it -- pass a PlateRef this model "
                "just declared, not one captured from an outer or an earlier "
                "trace."
            )
        names.append(ref.name)
    return tuple(names)


# NB: const/det/sample/observe below each take a keyword parameter named
# "plate" (the node's plate placement), which shadows this module-level
# function for the duration of each body. None of those bodies call plate()
# internally, so it is not a bug today -- but if a future edit needs to,
# rename the parameter first (every call site passes it by keyword, so the
# rename is a mechanical `plate=` find-and-replace).
def plate(name: str, size: int) -> PlateRef:
    """Declare a repetition axis and return a handle to it."""
    recorder = _active()
    for existing in recorder.plates:
        if existing.name == name:
            raise GraphError(f"duplicate plate name {name!r}")
    recorder.plates.append(Plate(name=name, size=int(size)))
    return PlateRef(name, recorder)


def const(
    name: str, value: Any, *, plate: PlateRef | Iterable[PlateRef] | None = None
) -> NodeRef:
    """Declare a fixed input node."""
    recorder = _active()
    node = Const(
        name=name,
        parents=(),
        plate=_plate_names(plate, recorder),
        value=jnp.asarray(value),
    )
    recorder.add(node)
    return NodeRef(name, recorder)


def det(
    name: str,
    fn: Callable[..., Any],
    *parents: NodeRef,
    linear_in: Iterable[str] = (),
    plate: PlateRef | Iterable[PlateRef] | None = None,
) -> NodeRef:
    """Declare a deterministic node: it propagates dependence, no density."""
    recorder = _active()
    node = Deterministic(
        name=name,
        parents=_parent_names(parents, recorder),
        plate=_plate_names(plate, recorder),
        fn=fn,
        linear_in=tuple(linear_in),
    )
    recorder.add(node)
    return NodeRef(name, recorder)


def sample(
    name: str,
    dist_fn: Callable[..., Any],
    *parents: NodeRef,
    plate: PlateRef | Iterable[PlateRef] | None = None,
) -> NodeRef:
    """Declare a latent probabilistic node."""
    recorder = _active()
    node = Probabilistic(
        name=name,
        parents=_parent_names(parents, recorder),
        plate=_plate_names(plate, recorder),
        dist_fn=dist_fn,
        observed=None,
    )
    recorder.add(node)
    return NodeRef(name, recorder)


def observe(
    name: str,
    dist_fn: Callable[..., Any],
    *parents: NodeRef,
    obs: Any,
    plate: PlateRef | Iterable[PlateRef] | None = None,
) -> NodeRef:
    """Declare a probabilistic node conditioned on data."""
    recorder = _active()
    node = Probabilistic(
        name=name,
        parents=_parent_names(parents, recorder),
        plate=_plate_names(plate, recorder),
        dist_fn=dist_fn,
        observed=jnp.asarray(obs),
    )
    recorder.add(node)
    return NodeRef(name, recorder)


def trace(model_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Graph:
    """Run ``model_fn`` once and return the graph it declared."""
    recorder = _Recorder()
    _STACK.append(recorder)
    try:
        model_fn(*args, **kwargs)
    finally:
        _STACK.pop()
    return Graph(nodes=tuple(recorder.nodes), plates=tuple(recorder.plates))
