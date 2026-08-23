# bayesmith P1 图核 + P2 NumPyro 桥 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 bayesmith 能把一个模型函数追踪成显式静态图，在图上算出联合对数密度，并把任意图桥接到 NumPyro 跑 NUTS——形成一个最小可行、且能自我验证的闭环。

**Architecture:** 四种节点（`Const`/`Deterministic`/`Probabilistic`，共同基类 `Node`）都是 `eqx.Module`。`Graph` 按拓扑序持有节点元组，在 `__check_init__` 里验证。追踪器用一个模块级栈记录节点，四个原语（`const`/`det`/`sample`/`observe`）返回 `NodeRef`，把 `NodeRef` 传给下游即声明父子边。求值是一次线性扫描；`log_joint` 在其上累加概率节点的 `log_prob`。桥接器把同一次扫描重写成 `numpyro.sample` 调用。

**关键设计约束（已原型验证，勿改）：** `Deterministic.fn` 与 `Probabilistic.dist_fn` **必须是非静态字段**。lambda 会成为非数组叶子被 `filter_jit` 过滤；而当 `fn` 是 `eqx.Module`（例如一整条 rheplicant `Pipeline`）时，它的参数是可追踪叶子，`eqx.filter_grad` 能穿透到它们。改成 `static=True` 会让含数组的 Module 触发 equinox 的静态字段报错。

**Tech Stack:** Python ≥3.11、JAX、Equinox、NumPyro、pytest。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `src/bayesmith/errors.py` | 异常族。仅 stdlib，不导入 jax/numpy |
| `src/bayesmith/graph/nodes.py` | `Node`/`Const`/`Deterministic`/`Probabilistic` |
| `src/bayesmith/graph/graph.py` | `Plate`、`Graph` 容器与结构验证 |
| `src/bayesmith/graph/trace.py` | `NodeRef`、四个原语、`plate()`、`trace()` |
| `src/bayesmith/graph/evaluate.py` | `evaluate()` 前向扫描、`log_joint()` |
| `src/bayesmith/bridge/numpyro_bridge.py` | `to_numpyro()`、`nuts()` |
| `src/bayesmith/__init__.py` | 公共 API |
| `tests/*` | 每个模块一个测试文件，外加共轭预言机 |

---

## Task 0：环境与包骨架

**Files:**
- Create: `src/bayesmith/__init__.py`, `src/bayesmith/errors.py`, `src/bayesmith/graph/__init__.py`, `src/bayesmith/bridge/__init__.py`, `tests/__init__.py`
- Test: `tests/test_errors.py`

- [ ] **Step 1: 建虚拟环境并可编辑安装**

```bash
cd /Users/zzhang/projects/bayesmith
uv venv --python 3.12
uv pip install -e . --group dev
```

> 实施时踩到并已确认：`src/bayesmith/` 此刻还是空目录，而 hatchling 对空包目录会产出一个**不含任何包文件**的 wheel，于是 `import bayesmith` 报 `ModuleNotFoundError`，`.pth` 也不会写出。写完 Step 2–4 的源文件后**再跑一次同一条 `uv pip install -e . --group dev`** 即可（命令完全相同，只是要在源文件存在之后执行）。

- [ ] **Step 2: 写 errors.py**

```python
# src/bayesmith/errors.py
"""Exception family for bayesmith.

Deliberately stdlib-only: this module is imported by everything, including
from contexts that must not pay for jax, so it may never import numpy or jax.

Every concrete class also derives from the closest builtin, so generic
handlers keep working.
"""


class BayesmithError(Exception):
    """Base class for every error bayesmith raises."""


class GraphError(BayesmithError, ValueError):
    """A graph was declared inconsistently.

    Covers: a node naming a parent that was not declared before it, a
    duplicate node name, a latent node left without a value, and a plate
    used by a node whose parents are all outside it.
    """


class TraceError(BayesmithError, RuntimeError):
    """A tracing primitive was called outside ``trace(...)``."""
```

- [ ] **Step 3: 写 errors 的测试**

```python
# tests/test_errors.py
import pytest

from bayesmith.errors import BayesmithError, GraphError, TraceError


def test_graph_error_is_catchable_as_the_family_and_as_value_error():
    assert issubclass(GraphError, BayesmithError)
    assert issubclass(GraphError, ValueError)
    with pytest.raises(BayesmithError):
        raise GraphError("bad graph")


def test_trace_error_is_catchable_as_the_family_and_as_runtime_error():
    assert issubclass(TraceError, BayesmithError)
    assert issubclass(TraceError, RuntimeError)


def test_errors_module_imports_no_heavy_dependency():
    """errors.py is on every import path, so it must stay stdlib-only."""
    import subprocess
    import sys

    code = (
        "import bayesmith.errors, sys; "
        "print(sorted({'jax', 'numpy', 'numpyro'} & set(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"
```

- [ ] **Step 4: 写最小的 `__init__.py`**

```python
# src/bayesmith/__init__.py
"""bayesmith: a graph of operators is a Bayesian model."""

from bayesmith.errors import BayesmithError, GraphError, TraceError

__all__ = ["BayesmithError", "GraphError", "TraceError"]
```

`src/bayesmith/graph/__init__.py` 与 `src/bayesmith/bridge/__init__.py` 先写空文件（只放一行 docstring）：

```python
"""Graph representation: node types, container, tracer, evaluation."""
```

```python
"""Bridges to external inference engines."""
```

`tests/__init__.py` 写空文件。

- [ ] **Step 5: 跑测试**

Run: `.venv/bin/python -m pytest tests/test_errors.py -v`
Expected: 3 passed

- [ ] **Step 6: 提交**

```bash
git add -A
git commit -m "feat: error family and package skeleton"
```

---

## Task 1：节点类型

**Files:**
- Create: `src/bayesmith/graph/nodes.py`
- Test: `tests/test_nodes.py`

- [ ] **Step 1: 先写失败的测试**

```python
# tests/test_nodes.py
import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from bayesmith.graph.nodes import Const, Deterministic, Node, Probabilistic


class Scale(eqx.Module):
    """Stand-in for a rheplicant Pipeline: an operator carrying parameters."""

    w: jax.Array

    def __call__(self, x):
        return self.w * x


def test_node_identity_fields_are_static():
    n = Deterministic(
        name="a", parents=("x",), plate=(), fn=lambda x: x, linear_in=("x",)
    )
    assert n.name == "a"
    assert n.parents == ("x",)
    assert n.linear_in == ("x",)
    # name/parents/plate/linear_in are metadata, so they must NOT be leaves
    assert jax.tree.leaves(n) == [n.fn]


def test_a_lambda_fn_is_a_non_array_leaf():
    """filter_jit routes non-array leaves to the static side; that is the point."""
    n = Deterministic(name="a", parents=("x",), plate=(), fn=lambda x: 2.0 * x)
    (leaf,) = jax.tree.leaves(n)
    assert callable(leaf)
    assert not eqx.is_array(leaf)


def test_a_module_fn_exposes_its_parameters_as_traceable_leaves():
    """The rheplicant-compatibility property: gradients must reach into fn."""
    n = Deterministic(name="a", parents=("x",), plate=(), fn=Scale(w=jnp.array(3.0)))
    leaves = jax.tree.leaves(n)
    assert len(leaves) == 1
    assert eqx.is_inexact_array(leaves[0])

    grad = eqx.filter_grad(lambda node, x: jnp.sum(node.fn(x)))(n, jnp.array(5.0))
    assert grad.fn.w == jnp.array(5.0)


def test_const_holds_its_value_as_an_array_leaf():
    n = Const(name="X", parents=(), plate=(), value=jnp.arange(3.0))
    (leaf,) = jax.tree.leaves(n)
    assert jnp.array_equal(leaf, jnp.arange(3.0))


def test_probabilistic_is_latent_when_unobserved_and_observed_otherwise():
    latent = Probabilistic(
        name="x", parents=(), plate=(), dist_fn=lambda: dist.Normal(0.0, 1.0),
        observed=None,
    )
    seen = Probabilistic(
        name="d", parents=("x",), plate=(), dist_fn=lambda m: dist.Normal(m, 1.0),
        observed=jnp.array([1.0, 2.0]),
    )
    assert latent.is_latent
    assert not seen.is_latent


def test_every_node_type_is_a_node():
    for cls in (Const, Deterministic, Probabilistic):
        assert issubclass(cls, Node)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_nodes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bayesmith.graph.nodes'`

- [ ] **Step 3: 写实现**

```python
# src/bayesmith/graph/nodes.py
"""The three node types a bayesmith graph is built from.

A node is an :class:`equinox.Module`, so a graph is a pytree and the
parameters carried by its operators are differentiable leaves for free.

**Why ``fn`` and ``dist_fn`` are not static fields.** They must accept an
``eqx.Module`` -- a whole rheplicant ``Pipeline`` is the motivating case --
whose array leaves have to stay traceable so gradients reach the parameters
inside them. A plain lambda placed in the same field simply becomes a
non-array leaf, which ``eqx.filter_jit`` routes to the static side. Marking
these fields ``static=True`` would break the Module case outright: equinox
refuses a JAX array in a static field.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax


class Node(eqx.Module):
    """Identity and position of one node in the graph.

    Attributes:
        name: unique within a graph; how parents refer to it.
        parents: names of the nodes whose values this one consumes, in the
            order ``fn`` / ``dist_fn`` expects them.
        plate: names of the plates this node lives inside. Empty means the
            node is scalar with respect to every plate.
    """

    name: str = eqx.field(static=True)
    parents: tuple[str, ...] = eqx.field(static=True)
    plate: tuple[str, ...] = eqx.field(static=True)


class Const(Node):
    """A fixed input: data, coordinates, anything the model conditions on.

    The value is an ordinary array field rather than a closure capture, so it
    is one traced leaf of the graph instead of a constant baked into a
    compiled function.
    """

    value: jax.Array


class Deterministic(Node):
    """A mapping that propagates dependence but contributes no density.

    Attributes:
        fn: called with the parents' values, in ``parents`` order.
        linear_in: the parents this node claims to be linear in. This is a
            **claim about the model**, not a hint -- it decides whether an
            exact conjugate solve may be used -- so it is checked rather than
            trusted before any such solve runs. Nothing in P1 reads it; it is
            recorded here so the declaration exists from the start.
    """

    fn: Callable[..., Any]
    linear_in: tuple[str, ...] = eqx.field(static=True, default=())


class Probabilistic(Node):
    """A conditional distribution: contributes one term to the log-density.

    Attributes:
        dist_fn: called with the parents' values; returns a NumPyro
            distribution. Sampling and ``log_prob`` therefore come from one
            object and cannot disagree -- the failure mode of every
            hand-written ``data + sigma * normal`` sitting beside a likelihood
            that carries its own sigma.
        observed: the data this node is conditioned on, or ``None`` if the
            node is latent.
    """

    dist_fn: Callable[..., Any]
    observed: jax.Array | None

    @property
    def is_latent(self) -> bool:
        """Whether this node is inferred rather than conditioned on."""
        return self.observed is None
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_nodes.py -v`
Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
git add src/bayesmith/graph/nodes.py tests/test_nodes.py
git commit -m "feat: node types for the probabilistic graph"
```

---

## Task 2：Graph 容器与结构验证

**Files:**
- Create: `src/bayesmith/graph/graph.py`
- Test: `tests/test_graph.py`

- [ ] **Step 1: 先写失败的测试**

```python
# tests/test_graph.py
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from bayesmith.errors import GraphError
from bayesmith.graph.graph import Graph, Plate
from bayesmith.graph.nodes import Const, Deterministic, Probabilistic


def _x():
    return Probabilistic(
        name="x", parents=(), plate=(), dist_fn=lambda: dist.Normal(0.0, 1.0),
        observed=None,
    )


def _mu():
    return Deterministic(
        name="mu", parents=("x",), plate=(), fn=lambda x: 2.0 * x, linear_in=("x",)
    )


def _d():
    return Probabilistic(
        name="d", parents=("mu",), plate=(),
        dist_fn=lambda m: dist.Normal(m, 0.1), observed=jnp.array([1.0]),
    )


def test_graph_exposes_names_in_declaration_order():
    g = Graph(nodes=(_x(), _mu(), _d()), plates=())
    assert g.names == ("x", "mu", "d")


def test_node_lookup_by_name():
    g = Graph(nodes=(_x(), _mu(), _d()), plates=())
    assert g.node("mu").parents == ("x",)


def test_unknown_node_name_is_refused_by_name():
    g = Graph(nodes=(_x(),), plates=())
    with pytest.raises(GraphError, match="no node named 'nope'"):
        g.node("nope")


def test_latents_and_observed_are_derived_not_stored():
    g = Graph(nodes=(_x(), _mu(), _d()), plates=())
    assert g.latents == ("x",)
    assert g.observed == ("d",)


def test_a_parent_declared_after_its_child_is_refused():
    with pytest.raises(GraphError, match="names parent 'x', which is not declared"):
        Graph(nodes=(_mu(), _x()), plates=())


def test_a_duplicate_node_name_is_refused():
    with pytest.raises(GraphError, match="duplicate node name 'x'"):
        Graph(nodes=(_x(), _x()), plates=())


def test_a_node_in_an_undeclared_plate_is_refused():
    n = Const(name="X", parents=(), plate=("obs",), value=jnp.arange(3.0))
    with pytest.raises(GraphError, match="plate 'obs', which the graph does not"):
        Graph(nodes=(n,), plates=())


def test_a_declared_plate_is_accepted():
    n = Const(name="X", parents=(), plate=("obs",), value=jnp.arange(3.0))
    g = Graph(nodes=(n,), plates=(Plate(name="obs", size=3),))
    assert g.plate_size("obs") == 3


def test_nested_plates_are_refused_with_a_reason():
    n = Const(name="X", parents=(), plate=("a", "b"), value=jnp.zeros((2, 3)))
    plates = (Plate(name="a", size=2), Plate(name="b", size=3))
    with pytest.raises(GraphError, match="nested plates are not supported yet"):
        Graph(nodes=(n,), plates=plates)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_graph.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bayesmith.graph.graph'`

- [ ] **Step 3: 写实现**

```python
# src/bayesmith/graph/graph.py
"""The graph container, and the structural rules it refuses to violate.

Declaration order **is** topological order. A node may only name parents
already declared, which is automatic when the graph comes from ``trace`` --
you cannot pass a ``NodeRef`` you have not created -- and is checked here so
that a hand-built graph obeys the same rule.
"""

from __future__ import annotations

import equinox as eqx

from bayesmith.errors import GraphError
from bayesmith.graph.nodes import Node, Probabilistic


class Plate(eqx.Module):
    """A named repetition axis. ``size`` is fixed at construction."""

    name: str = eqx.field(static=True)
    size: int = eqx.field(static=True)


class Graph(eqx.Module):
    """A static DAG of nodes in topological order, plus its plates."""

    nodes: tuple[Node, ...]
    plates: tuple[Plate, ...]

    def __check_init__(self) -> None:
        plate_names = {p.name for p in self.plates}
        seen: set[str] = set()
        for node in self.nodes:
            if node.name in seen:
                raise GraphError(f"duplicate node name {node.name!r}")
            for parent in node.parents:
                if parent not in seen:
                    raise GraphError(
                        f"node {node.name!r} names parent {parent!r}, which is not "
                        "declared before it. Declaration order is topological "
                        "order: declare the parent first."
                    )
            if len(node.plate) > 1:
                raise GraphError(
                    f"node {node.name!r} is in plates {node.plate}: nested plates "
                    "are not supported yet. Flatten the two axes into one plate."
                )
            for plate in node.plate:
                if plate not in plate_names:
                    raise GraphError(
                        f"node {node.name!r} is in plate {plate!r}, which the graph "
                        f"does not declare. Known plates: {sorted(plate_names)}."
                    )
            seen.add(node.name)

    @property
    def names(self) -> tuple[str, ...]:
        """Every node name, in topological order."""
        return tuple(node.name for node in self.nodes)

    @property
    def latents(self) -> tuple[str, ...]:
        """Names of the probabilistic nodes that are inferred."""
        return tuple(
            n.name for n in self.nodes if isinstance(n, Probabilistic) and n.is_latent
        )

    @property
    def observed(self) -> tuple[str, ...]:
        """Names of the probabilistic nodes that are conditioned on."""
        return tuple(
            n.name
            for n in self.nodes
            if isinstance(n, Probabilistic) and not n.is_latent
        )

    def node(self, name: str) -> Node:
        """The node called ``name``.

        Raises:
            GraphError: if no node has that name.
        """
        for node in self.nodes:
            if node.name == name:
                return node
        raise GraphError(f"no node named {name!r}. Known: {list(self.names)}.")

    def plate_size(self, name: str) -> int:
        """The declared size of plate ``name``.

        Raises:
            GraphError: if no plate has that name.
        """
        for plate in self.plates:
            if plate.name == name:
                return plate.size
        raise GraphError(
            f"no plate named {name!r}. Known: {[p.name for p in self.plates]}."
        )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_graph.py -v`
Expected: 9 passed

- [ ] **Step 5: 提交**

```bash
git add src/bayesmith/graph/graph.py tests/test_graph.py
git commit -m "feat: graph container with structural validation"
```

---

## Task 3：追踪器

**Files:**
- Create: `src/bayesmith/graph/trace.py`
- Test: `tests/test_trace.py`

- [ ] **Step 1: 先写失败的测试**

```python
# tests/test_trace.py
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from bayesmith.errors import GraphError, TraceError
from bayesmith.graph.graph import Graph, Plate
from bayesmith.graph.nodes import Const, Deterministic, Probabilistic
from bayesmith.graph.trace import const, det, observe, plate, sample, trace


def test_trace_records_nodes_in_declaration_order():
    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        mu = det("mu", lambda v: 2.0 * v, x, linear_in=("x",))
        observe("d", lambda m: dist.Normal(m, 0.1), mu, obs=jnp.array([1.0]))

    g = trace(model)
    assert g.names == ("x", "mu", "d")
    assert isinstance(g.node("x"), Probabilistic)
    assert isinstance(g.node("mu"), Deterministic)


def test_passing_a_noderef_declares_the_edge():
    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        y = sample("y", lambda: dist.Normal(0.0, 1.0))
        det("mu", lambda a, b: a + b, x, y)

    g = trace(model)
    assert g.node("mu").parents == ("x", "y")


def test_const_becomes_a_node_carrying_its_value():
    def model():
        const("X", jnp.arange(3.0))

    g = trace(model)
    node = g.node("X")
    assert isinstance(node, Const)
    assert jnp.array_equal(node.value, jnp.arange(3.0))


def test_linear_in_is_recorded_as_declared():
    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        det("mu", lambda v: 2.0 * v, x, linear_in=("x",))

    assert trace(model).node("mu").linear_in == ("x",)


def test_observed_data_is_attached_to_the_node():
    def model():
        observe("d", lambda: dist.Normal(0.0, 1.0), obs=jnp.array([1.0, 2.0]))

    assert jnp.array_equal(trace(model).node("d").observed, jnp.array([1.0, 2.0]))


def test_plate_is_declared_and_attached():
    def model():
        obs = plate("obs", 4)
        const("X", jnp.arange(4.0), plate=obs)

    g = trace(model)
    assert g.plates == (Plate(name="obs", size=4),)
    assert g.node("X").plate == ("obs",)


def test_a_primitive_outside_trace_is_refused():
    with pytest.raises(TraceError, match="must be called inside trace"):
        sample("x", lambda: dist.Normal(0.0, 1.0))


def test_a_duplicate_name_is_refused_during_tracing():
    def model():
        sample("x", lambda: dist.Normal(0.0, 1.0))
        sample("x", lambda: dist.Normal(0.0, 1.0))

    with pytest.raises(GraphError, match="duplicate node name 'x'"):
        trace(model)


def test_trace_forwards_arguments_to_the_model():
    def model(data):
        observe("d", lambda: dist.Normal(0.0, 1.0), obs=data)

    g = trace(model, jnp.array([3.0]))
    assert jnp.array_equal(g.node("d").observed, jnp.array([3.0]))


def test_the_recorder_is_popped_even_when_the_model_raises():
    def model():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        trace(model)
    # The stack must be clean, or the next trace would inherit these nodes.
    with pytest.raises(TraceError):
        sample("x", lambda: dist.Normal(0.0, 1.0))


def test_tracing_twice_gives_isomorphic_graphs():
    """Structure must not depend on how many times the model has been traced."""

    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        det("mu", lambda v: 2.0 * v, x)

    a, b = trace(model), trace(model)
    assert a.names == b.names
    assert [n.parents for n in a.nodes] == [n.parents for n in b.nodes]
    assert [n.plate for n in a.nodes] == [n.plate for n in b.nodes]


def test_an_explicit_graph_and_a_traced_one_agree_on_structure():
    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        det("mu", lambda v: 2.0 * v, x, linear_in=("x",))

    traced = trace(model)
    built = Graph(
        nodes=(
            Probabilistic(
                name="x", parents=(), plate=(),
                dist_fn=lambda: dist.Normal(0.0, 1.0), observed=None,
            ),
            Deterministic(
                name="mu", parents=("x",), plate=(),
                fn=lambda v: 2.0 * v, linear_in=("x",),
            ),
        ),
        plates=(),
    )
    assert traced.names == built.names
    assert [n.parents for n in traced.nodes] == [n.parents for n in built.nodes]
    assert traced.node("mu").linear_in == built.node("mu").linear_in
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_trace.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bayesmith.graph.trace'`

- [ ] **Step 3: 写实现**

```python
# src/bayesmith/graph/trace.py
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_trace.py -v`
Expected: 12 passed

- [ ] **Step 5: 提交**

```bash
git add src/bayesmith/graph/trace.py tests/test_trace.py
git commit -m "feat: trace a model function into an explicit graph"
```

---

## Task 4：前向求值

**Files:**
- Create: `src/bayesmith/graph/evaluate.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: 先写失败的测试**

```python
# tests/test_evaluate.py
import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from bayesmith.errors import GraphError
from bayesmith.graph.evaluate import evaluate
from bayesmith.graph.trace import const, det, observe, sample, trace


class Scale(eqx.Module):
    w: jax.Array

    def __call__(self, x):
        return self.w * x


def _linear_model():
    def model():
        X = const("X", jnp.array([1.0, 2.0, 3.0]))
        a = sample("a", lambda: dist.Normal(0.0, 1.0))
        mu = det("mu", lambda a_, X_: a_ * X_, a, X, linear_in=("a",))
        observe("d", lambda m: dist.Normal(m, 0.1), mu, obs=jnp.zeros(3))

    return trace(model)


def test_evaluate_computes_every_node():
    env = evaluate(_linear_model(), {"a": jnp.array(2.0)})
    assert set(env) == {"X", "a", "mu", "d"}
    assert jnp.allclose(env["mu"], jnp.array([2.0, 4.0, 6.0]))


def test_observed_nodes_take_their_data_not_a_supplied_value():
    env = evaluate(_linear_model(), {"a": jnp.array(2.0)})
    assert jnp.array_equal(env["d"], jnp.zeros(3))


def test_a_latent_without_a_value_is_refused_by_name():
    with pytest.raises(GraphError, match="latent node 'a' has no value"):
        evaluate(_linear_model(), {})


def test_a_value_for_an_unknown_name_is_refused():
    with pytest.raises(GraphError, match="values names 'nope'"):
        evaluate(_linear_model(), {"a": jnp.array(1.0), "nope": jnp.array(0.0)})


def test_evaluate_is_differentiable_through_a_module_operator():
    """A parameterised operator inside a node stays differentiable."""

    def model():
        X = const("X", jnp.array([1.0, 2.0]))
        det("mu", Scale(w=jnp.array(3.0)), X)

    graph = trace(model)

    def total(g):
        return jnp.sum(evaluate(g, {})["mu"])

    grad = eqx.filter_grad(total)(graph)
    assert jnp.allclose(grad.nodes[1].fn.w, jnp.array(3.0))


def test_evaluate_is_jittable():
    graph = _linear_model()
    jitted = eqx.filter_jit(lambda g, a: evaluate(g, {"a": a})["mu"])
    assert jnp.allclose(jitted(graph, jnp.array(2.0)), jnp.array([2.0, 4.0, 6.0]))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bayesmith.graph.evaluate'`

- [ ] **Step 3: 写实现**

```python
# src/bayesmith/graph/evaluate.py
"""Running a graph forward, and reading its log-density.

Evaluation is one linear scan in topological order: every node's parents are
already in the environment by the time it is reached, which is what the
ordering rule in :class:`~bayesmith.graph.graph.Graph` buys.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from bayesmith.errors import GraphError
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Const, Deterministic, Probabilistic

Env = dict[str, Any]


def apply_deterministic(graph: Graph, node: Deterministic, env: Env) -> Any:
    """Call ``node.fn``, vmapped over the plate the node lives in.

    Public because the NumPyro bridge re-runs the same scan and must apply
    plates identically; a private name used across modules is not private.
    """
    args = [env[parent] for parent in node.parents]
    if not node.plate:
        return node.fn(*args)

    in_axes = tuple(
        0 if graph.node(parent).plate == node.plate else None
        for parent in node.parents
    )
    if all(axis is None for axis in in_axes):
        raise GraphError(
            f"deterministic node {node.name!r} is in plate {node.plate[0]!r} but "
            "none of its parents are, so there is nothing to map over. Either put "
            "a parent in the plate, or drop the plate and let broadcasting do it."
        )
    return jax.vmap(node.fn, in_axes=in_axes)(*args)


def evaluate(graph: Graph, values: Mapping[str, Any] | None = None) -> Env:
    """Compute every node's value.

    Args:
        graph: the graph to run.
        values: values for the latent nodes. Observed nodes take their own
            data and must not appear here; deterministic and constant nodes
            are computed.

    Returns:
        A mapping from node name to value, covering every node in the graph.

    Raises:
        GraphError: if a latent node has no value, or ``values`` names
            something that is not a latent node.
    """
    values = dict(values or {})
    unknown = set(values) - set(graph.latents)
    if unknown:
        raise GraphError(
            f"values names {sorted(unknown)[0]!r}, which is not a latent node of "
            f"this graph. Latents are {list(graph.latents)}."
        )

    env: Env = {}
    for node in graph.nodes:
        if isinstance(node, Const):
            env[node.name] = node.value
        elif isinstance(node, Deterministic):
            env[node.name] = apply_deterministic(graph, node, env)
        elif isinstance(node, Probabilistic):
            if not node.is_latent:
                env[node.name] = node.observed
            elif node.name in values:
                env[node.name] = values[node.name]
            else:
                raise GraphError(
                    f"latent node {node.name!r} has no value. Supply one in "
                    "`values`, or condition the node with observe(...)."
                )
        else:  # pragma: no cover - Node is abstract in practice
            raise GraphError(f"unknown node type {type(node).__name__}")
    return env


def log_joint(graph: Graph, values: Mapping[str, Any] | None = None) -> jax.Array:
    """The joint log-density of the graph at ``values``.

    Every probabilistic node contributes ``log_prob`` of its value under the
    distribution its parents parameterise; deterministic nodes contribute
    nothing but the dependence they carry.
    """
    env = evaluate(graph, values)
    total = jnp.zeros(())
    for node in graph.nodes:
        if isinstance(node, Probabilistic):
            distribution = node.dist_fn(*[env[p] for p in node.parents])
            total = total + jnp.sum(distribution.log_prob(env[node.name]))
    return total
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -v`
Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
git add src/bayesmith/graph/evaluate.py tests/test_evaluate.py
git commit -m "feat: forward evaluation of a graph"
```

---

## Task 5：联合对数密度

**Files:**
- Test: `tests/test_log_joint.py`（实现已在 Task 4 的 `evaluate.py` 内）

- [ ] **Step 1: 先写失败的测试**

```python
# tests/test_log_joint.py
import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from bayesmith.graph.evaluate import log_joint
from bayesmith.graph.trace import const, det, observe, sample, trace


def _conjugate_graph(data, tau=2.0, sigma=0.5):
    """x ~ N(0, tau^2);  d_i ~ N(x, sigma^2)."""

    def model():
        x = sample("x", lambda: dist.Normal(0.0, tau))
        observe("d", lambda v: dist.Normal(v, sigma), x, obs=data)

    return trace(model)


def test_log_joint_matches_the_hand_written_density():
    data = jnp.array([1.0, 2.0, 3.0])
    tau, sigma, x = 2.0, 0.5, 0.7
    graph = _conjugate_graph(data, tau, sigma)

    got = log_joint(graph, {"x": jnp.array(x)})
    expected = dist.Normal(0.0, tau).log_prob(x) + jnp.sum(
        dist.Normal(x, sigma).log_prob(data)
    )
    assert jnp.allclose(got, expected, rtol=1e-6)


def test_deterministic_nodes_contribute_no_density():
    data = jnp.array([1.0])

    def with_det():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        mu = det("mu", lambda v: v, x)
        observe("d", lambda m: dist.Normal(m, 1.0), mu, obs=data)

    def without_det():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe("d", lambda v: dist.Normal(v, 1.0), x, obs=data)

    at = {"x": jnp.array(0.3)}
    assert jnp.allclose(log_joint(trace(with_det), at), log_joint(trace(without_det), at))


def test_const_nodes_contribute_no_density():
    def model():
        const("X", jnp.array([5.0, 6.0]))
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe("d", lambda v: dist.Normal(v, 1.0), x, obs=jnp.array([1.0]))

    at = {"x": jnp.array(0.3)}
    got = log_joint(trace(model), at)
    expected = dist.Normal(0.0, 1.0).log_prob(0.3) + dist.Normal(0.3, 1.0).log_prob(1.0)
    assert jnp.allclose(got, expected, rtol=1e-6)


def test_log_joint_is_differentiable_in_the_latent_value():
    """The gradient of a Gaussian log-density is available in closed form."""
    data = jnp.array([1.0, 2.0, 3.0])
    tau, sigma = 2.0, 0.5
    graph = _conjugate_graph(data, tau, sigma)

    grad = jax.grad(lambda x: log_joint(graph, {"x": x}))(jnp.array(0.7))
    expected = -0.7 / tau**2 + jnp.sum(data - 0.7) / sigma**2
    assert jnp.allclose(grad, expected, rtol=1e-5)


def test_log_joint_is_jittable():
    graph = _conjugate_graph(jnp.array([1.0, 2.0]))
    jitted = eqx.filter_jit(lambda g, x: log_joint(g, {"x": x}))
    assert jnp.isfinite(jitted(graph, jnp.array(0.4)))


def test_log_joint_is_a_scalar_whatever_the_data_shape():
    for shape in [(1,), (7,), (3, 4)]:
        graph = _conjugate_graph(jnp.ones(shape))
        assert log_joint(graph, {"x": jnp.array(0.0)}).shape == ()
```

- [ ] **Step 2: 跑测试确认失败或通过**

Run: `.venv/bin/python -m pytest tests/test_log_joint.py -v`
Expected: 6 passed（`log_joint` 已在 Task 4 写好；若有失败，按报错修 `evaluate.py`，不要改测试）

- [ ] **Step 3: 提交**

```bash
git add tests/test_log_joint.py
git commit -m "test: pin log_joint against hand-written densities and closed-form gradients"
```

---

## Task 6：plate

**Files:**
- Test: `tests/test_plates.py`（实现已在 Task 4 的 `apply_deterministic` 内）

- [ ] **Step 1: 先写失败的测试**

```python
# tests/test_plates.py
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from bayesmith.errors import GraphError
from bayesmith.graph.evaluate import evaluate, log_joint
from bayesmith.graph.trace import const, det, observe, plate, sample, trace


def test_a_plated_deterministic_node_is_vmapped_over_its_plated_parent():
    def model():
        obs = plate("obs", 3)
        X = const("X", jnp.array([1.0, 2.0, 3.0]), plate=obs)
        det("mu", lambda x: x**2, X, plate=obs)

    env = evaluate(trace(model), {})
    assert jnp.allclose(env["mu"], jnp.array([1.0, 4.0, 9.0]))


def test_an_unplated_parent_is_broadcast_not_mapped():
    def model():
        obs = plate("obs", 3)
        X = const("X", jnp.array([1.0, 2.0, 3.0]), plate=obs)
        a = const("a", jnp.array(10.0))
        det("mu", lambda x, a_: a_ * x, X, a, plate=obs)

    env = evaluate(trace(model), {})
    assert jnp.allclose(env["mu"], jnp.array([10.0, 20.0, 30.0]))


def test_vmap_agrees_with_an_explicit_python_loop():
    """The plate is an optimisation, so it must change nothing numerically."""
    xs = jnp.array([0.5, 1.5, 2.5, 3.5])

    def model():
        obs = plate("obs", 4)
        X = const("X", xs, plate=obs)
        det("mu", lambda x: jnp.sin(x) * 3.0, X, plate=obs)

    got = evaluate(trace(model), {})["mu"]
    expected = jnp.stack([jnp.sin(x) * 3.0 for x in xs])
    assert jnp.array_equal(got, expected)


def test_a_plated_node_with_no_plated_parent_is_refused_with_a_reason():
    def model():
        obs = plate("obs", 3)
        a = const("a", jnp.array(1.0))
        det("mu", lambda v: v, a, plate=obs)

    with pytest.raises(GraphError, match="nothing to map over"):
        evaluate(trace(model), {})


def test_a_plated_likelihood_sums_over_the_plate():
    def model():
        obs = plate("obs", 3)
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe("d", lambda v: dist.Normal(v, 1.0), x, obs=jnp.array([1.0, 2.0, 3.0]), plate=obs)

    got = log_joint(trace(model), {"x": jnp.array(0.5)})
    expected = dist.Normal(0.0, 1.0).log_prob(0.5) + jnp.sum(
        dist.Normal(0.5, 1.0).log_prob(jnp.array([1.0, 2.0, 3.0]))
    )
    assert jnp.allclose(got, expected, rtol=1e-6)


@pytest.mark.parametrize("size", [1, 1000])
def test_extreme_plate_sizes(size):
    """Failure modes are U-shaped; test both ends, not the comfortable middle."""

    def model():
        obs = plate("obs", size)
        X = const("X", jnp.arange(size, dtype=jnp.float32), plate=obs)
        det("mu", lambda x: x + 1.0, X, plate=obs)

    env = evaluate(trace(model), {})
    assert env["mu"].shape == (size,)
    assert jnp.allclose(env["mu"], jnp.arange(size, dtype=jnp.float32) + 1.0)
```

- [ ] **Step 2: 跑测试**

Run: `.venv/bin/python -m pytest tests/test_plates.py -v`
Expected: 7 passed

- [ ] **Step 3: 提交**

```bash
git add tests/test_plates.py
git commit -m "test: plates map over their axis and agree with an explicit loop"
```

---

## Task 7：NumPyro 桥

**Files:**
- Create: `src/bayesmith/bridge/numpyro_bridge.py`
- Test: `tests/test_bridge.py`

- [ ] **Step 1: 先写失败的测试**

```python
# tests/test_bridge.py
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer.util import log_density

from bayesmith.bridge.numpyro_bridge import to_numpyro
from bayesmith.graph.evaluate import log_joint
from bayesmith.graph.trace import const, det, observe, plate, sample, trace


def _graph(data):
    def model():
        X = const("X", jnp.array([1.0, 2.0, 3.0]))
        a = sample("a", lambda: dist.Normal(0.0, 2.0))
        mu = det("mu", lambda a_, X_: a_ * X_, a, X, linear_in=("a",))
        observe("d", lambda m: dist.Normal(m, 0.5), mu, obs=data)

    return trace(model)


def test_the_bridge_and_log_joint_agree_on_the_density():
    """Two independent readings of the same graph must give the same number."""
    graph = _graph(jnp.array([1.0, 2.0, 3.0]))
    at = {"a": jnp.array(0.7)}

    ours = log_joint(graph, at)
    theirs, _ = log_density(to_numpyro(graph), (), {}, at)
    assert jnp.allclose(ours, theirs, rtol=1e-6)


def test_latent_sites_carry_the_graph_node_names():
    graph = _graph(jnp.array([1.0, 2.0, 3.0]))
    trace_ = numpyro.handlers.trace(
        numpyro.handlers.seed(to_numpyro(graph), jax.random.key(0))
    ).get_trace()
    assert trace_["a"]["type"] == "sample"
    assert not trace_["a"]["is_observed"]
    assert trace_["d"]["is_observed"]


def test_deterministic_nodes_are_recorded_as_numpyro_deterministic():
    graph = _graph(jnp.array([1.0, 2.0, 3.0]))
    trace_ = numpyro.handlers.trace(
        numpyro.handlers.seed(to_numpyro(graph), jax.random.key(0))
    ).get_trace()
    assert trace_["mu"]["type"] == "deterministic"


def test_a_plated_graph_bridges_and_still_agrees():
    def model():
        obs = plate("obs", 3)
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe(
            "d", lambda v: dist.Normal(v, 1.0), x,
            obs=jnp.array([1.0, 2.0, 3.0]), plate=obs,
        )

    graph = trace(model)
    at = {"x": jnp.array(0.4)}
    ours = log_joint(graph, at)
    theirs, _ = log_density(to_numpyro(graph), (), {}, at)
    assert jnp.allclose(ours, theirs, rtol=1e-6)


def test_the_bridged_model_can_be_sampled_from_the_prior():
    graph = _graph(jnp.array([1.0, 2.0, 3.0]))
    predictive = numpyro.infer.Predictive(to_numpyro(graph), num_samples=8)
    draws = predictive(jax.random.key(1))
    assert draws["a"].shape == (8,)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_bridge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bayesmith.bridge.numpyro_bridge'`

- [ ] **Step 3: 写实现**

```python
# src/bayesmith/bridge/numpyro_bridge.py
"""Turning a graph into a NumPyro model, and running NUTS on it.

This is the last row of the dispatch table: whatever structure bayesmith
cannot solve exactly is handed to NumPyro. It is also the oracle every exact
path is checked against, because a graph that qualifies for an exact method
always also qualifies for NUTS.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import numpyro
from numpyro.infer import MCMC, NUTS

from bayesmith.graph.evaluate import apply_deterministic
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Const, Deterministic, Probabilistic


def to_numpyro(graph: Graph) -> Callable[[], dict[str, Any]]:
    """Build a NumPyro model that declares the same joint distribution.

    Latent and observed nodes become ``numpyro.sample`` sites carrying the
    graph's own node names, so posterior samples come back keyed by them.
    Deterministic nodes are recorded with ``numpyro.deterministic`` so they
    appear in traces and predictives without contributing a density.
    """

    def model() -> dict[str, Any]:
        env: dict[str, Any] = {}
        for node in graph.nodes:
            if isinstance(node, Const):
                env[node.name] = node.value
            elif isinstance(node, Deterministic):
                env[node.name] = numpyro.deterministic(
                    node.name, apply_deterministic(graph, node, env)
                )
            elif isinstance(node, Probabilistic):
                distribution = node.dist_fn(*[env[p] for p in node.parents])
                if node.plate:
                    name = node.plate[0]
                    with numpyro.plate(name, graph.plate_size(name)):
                        env[node.name] = numpyro.sample(
                            node.name, distribution, obs=node.observed
                        )
                else:
                    env[node.name] = numpyro.sample(
                        node.name, distribution, obs=node.observed
                    )
        return env

    return model


def nuts(
    graph: Graph,
    key: jax.Array,
    *,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 1,
    progress_bar: bool = False,
) -> dict[str, jax.Array]:
    """Sample the posterior of ``graph`` with NUTS.

    Args:
        graph: the model.
        key: a PRNG key.
        num_warmup: adaptation draws, discarded.
        num_samples: retained draws per chain.
        num_chains: independent chains.
        progress_bar: whether NumPyro prints progress.

    Returns:
        A mapping from latent node name to its draws.
    """
    mcmc = MCMC(
        NUTS(to_numpyro(graph)),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(key)
    return mcmc.get_samples()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_bridge.py -v`
Expected: 5 passed

- [ ] **Step 5: 跑全套确认没有回归**

Run: `.venv/bin/python -m pytest -q`
Expected: 全部通过

- [ ] **Step 6: 提交**

```bash
git add -A
git commit -m "feat: bridge any graph to a NumPyro model and to NUTS"
```

---

## Task 8：退化图与极端情形

spec §四明确要求覆盖极端参数值，理由是失效模式常呈 U 形、只在角落出现。plate 尺寸的两端已在 Task 6 覆盖；这里补上图**形状**的角落。

**Files:**
- Create: `tests/test_degenerate_graphs.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_degenerate_graphs.py
"""Graphs at the corners of what a graph can be.

Every one of these is a shape a user will eventually build by accident, and
each exercises a branch that a comfortable three-node example never reaches.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro.infer.util import log_density

from bayesmith.bridge.numpyro_bridge import to_numpyro
from bayesmith.errors import GraphError
from bayesmith.graph.evaluate import evaluate, log_joint
from bayesmith.graph.trace import const, det, observe, sample, trace


def test_a_single_node_graph_evaluates_and_has_zero_density():
    def model():
        const("X", jnp.array(1.0))

    graph = trace(model)
    assert graph.names == ("X",)
    assert graph.latents == ()
    assert evaluate(graph, {})["X"] == jnp.array(1.0)
    assert log_joint(graph, {}) == jnp.zeros(())


def test_a_graph_with_no_latents_still_has_a_density():
    """A fully observed model: the likelihood at fixed parameters."""

    def model():
        observe("d", lambda: dist.Normal(0.0, 1.0), obs=jnp.array([1.0, 2.0]))

    graph = trace(model)
    assert graph.latents == ()
    expected = jnp.sum(dist.Normal(0.0, 1.0).log_prob(jnp.array([1.0, 2.0])))
    assert jnp.allclose(log_joint(graph, {}), expected)


def test_a_graph_with_no_observations_is_the_prior():
    def model():
        sample("x", lambda: dist.Normal(0.0, 1.0))

    graph = trace(model)
    assert graph.observed == ()
    assert jnp.allclose(
        log_joint(graph, {"x": jnp.array(0.5)}),
        dist.Normal(0.0, 1.0).log_prob(0.5),
    )


def test_an_empty_graph_is_allowed_and_has_zero_density():
    """Nothing declared is a valid, if useless, graph -- not a crash."""

    def model():
        return None

    graph = trace(model)
    assert graph.names == ()
    assert log_joint(graph, {}) == jnp.zeros(())


def test_two_perfectly_collinear_parents_are_not_rejected_here():
    """Collinearity is an identifiability question, not a graph-shape one.

    P1 must build this graph without complaint; refusing it is P5's job, and
    doing it here would refuse legitimate over-parameterised models that a
    prior makes perfectly well posed.
    """

    def model():
        a = sample("a", lambda: dist.Normal(0.0, 1.0))
        b = sample("b", lambda: dist.Normal(0.0, 1.0))
        mu = det("mu", lambda x, y: x + y, a, b, linear_in=("a", "b"))
        observe("d", lambda m: dist.Normal(m, 1.0), mu, obs=jnp.array([1.0]))

    graph = trace(model)
    at = {"a": jnp.array(0.3), "b": jnp.array(0.4)}
    ours = log_joint(graph, at)
    theirs, _ = log_density(to_numpyro(graph), (), {}, at)
    assert jnp.allclose(ours, theirs, rtol=1e-6)


def test_a_deep_chain_evaluates_in_one_pass():
    """Topological order is declaration order, however long the chain."""

    def model():
        node = sample("x0", lambda: dist.Normal(0.0, 1.0))
        for i in range(1, 50):
            node = det(f"x{i}", lambda v: v + 1.0, node)

    graph = trace(model)
    env = evaluate(graph, {"x0": jnp.array(0.0)})
    assert env["x49"] == jnp.array(49.0)


def test_a_diamond_reaches_the_shared_parent_once():
    """A DAG, not a tree: two paths converge on one node."""

    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        left = det("left", lambda v: 2.0 * v, x)
        right = det("right", lambda v: 3.0 * v, x)
        det("join", lambda a, b: a + b, left, right)

    env = evaluate(trace(model), {"x": jnp.array(1.0)})
    assert env["join"] == jnp.array(5.0)


def test_a_cycle_cannot_be_expressed_by_tracing():
    """Tracing makes cycles unrepresentable: you cannot pass a handle you have
    not created yet. Hand-built cycles are caught by Graph.__check_init__,
    pinned in Task 2. Passing something that merely looks like a handle is
    refused by name."""

    def model():
        det("a", lambda v: v, object())

    with pytest.raises(GraphError, match="parents must be NodeRef"):
        trace(model)
```

- [ ] **Step 2: 跑测试**

Run: `.venv/bin/python -m pytest tests/test_degenerate_graphs.py -v`
Expected: 8 passed

若 `test_an_empty_graph_is_allowed_and_has_zero_density` 失败，说明 `Graph` 不接受空节点元组——修 `graph.py`，不要改测试：空图是合法的。

- [ ] **Step 3: 提交**

```bash
git add tests/test_degenerate_graphs.py
git commit -m "test: graphs at the corners -- empty, latent-free, observation-free, diamond"
```

---

## Task 9：共轭预言机（本计划的验收关口）

这是 spec §四要求的第一道交叉验证：**桥接出来的 NUTS 必须打得中一个有闭式解的后验**。没有它，前面七个任务全都只是自洽而未必正确。

**Files:**
- Create: `tests/test_conjugate_oracle.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_conjugate_oracle.py
"""NUTS through the bridge, against a posterior that is known in closed form.

Normal-normal conjugacy: x ~ N(0, tau^2), d_i ~ N(x, sigma^2), i = 1..N gives

    var_post  = 1 / (1/tau^2 + N/sigma^2)
    mean_post = var_post * sum(d) / sigma^2

Everything upstream of this file is self-consistent by construction; this is
where the package first has to be *right*.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.diagnostics import effective_sample_size

from bayesmith.bridge.numpyro_bridge import nuts
from bayesmith.graph.trace import observe, sample, trace

TAU = 2.0
SIGMA = 0.5


def _graph(data):
    def model():
        x = sample("x", lambda: dist.Normal(0.0, TAU))
        observe("d", lambda v: dist.Normal(v, SIGMA), x, obs=data)

    return trace(model)


def _analytic_posterior(data):
    n = data.size
    var = 1.0 / (1.0 / TAU**2 + n / SIGMA**2)
    mean = var * jnp.sum(data) / SIGMA**2
    return float(mean), float(jnp.sqrt(var))


@pytest.mark.parametrize("n", [1, 20, 200])
def test_nuts_recovers_the_analytic_posterior(n):
    """Includes n=1, where the prior still dominates -- the awkward end."""
    data = jnp.linspace(0.5, 2.5, n)
    draws = nuts(_graph(data), jax.random.key(0), num_warmup=1000, num_samples=4000)

    mean_hat = float(jnp.mean(draws["x"]))
    sd_hat = float(jnp.std(draws["x"]))
    mean_true, sd_true = _analytic_posterior(data)

    # Compare the mean on the scale of its own Monte-Carlo error.
    ess = float(effective_sample_size(np.asarray(draws["x"])[None, :]))
    assert ess > 400, f"chain too autocorrelated to judge: ESS={ess:.0f}"
    z = abs(mean_hat - mean_true) / (sd_true / np.sqrt(ess))
    assert z < 4.0, f"posterior mean off by {z:.1f} sigma (n={n})"
    assert abs(sd_hat - sd_true) / sd_true < 0.1, (
        f"posterior sd {sd_hat:.4f} vs analytic {sd_true:.4f} (n={n})"
    )


def test_two_seeds_agree_within_monte_carlo_error():
    """A wrong graph often shows up as seed-dependent answers."""
    data = jnp.linspace(0.5, 2.5, 20)
    a = nuts(_graph(data), jax.random.key(0), num_warmup=1000, num_samples=4000)
    b = nuts(_graph(data), jax.random.key(1), num_warmup=1000, num_samples=4000)
    _, sd_true = _analytic_posterior(data)
    assert abs(float(jnp.mean(a["x"])) - float(jnp.mean(b["x"]))) < 0.2 * sd_true
```

- [ ] **Step 2: 跑测试**

Run: `.venv/bin/python -m pytest tests/test_conjugate_oracle.py -v`
Expected: 4 passed（约 1–2 分钟）

- [ ] **Step 3: 跑全套**

Run: `.venv/bin/python -m pytest -q`
Expected: 全部通过

- [ ] **Step 4: 导出公共 API**

```python
# src/bayesmith/__init__.py
"""bayesmith: a graph of operators is a Bayesian model.

Deterministic operators propagate dependence; probabilistic ones contribute a
conditional density. The graph's structure is what selects the inference
method -- exact where a subgraph permits one, NUTS where it does not.
"""

from bayesmith.bridge.numpyro_bridge import nuts, to_numpyro
from bayesmith.errors import BayesmithError, GraphError, TraceError
from bayesmith.graph.evaluate import evaluate, log_joint
from bayesmith.graph.graph import Graph, Plate
from bayesmith.graph.nodes import Const, Deterministic, Node, Probabilistic
from bayesmith.graph.trace import NodeRef, const, det, observe, plate, sample, trace

__all__ = [
    # tracing
    "trace",
    "const",
    "det",
    "sample",
    "observe",
    "plate",
    "NodeRef",
    # graph
    "Graph",
    "Plate",
    "Node",
    "Const",
    "Deterministic",
    "Probabilistic",
    # evaluation
    "evaluate",
    "log_joint",
    # inference
    "to_numpyro",
    "nuts",
    # errors
    "BayesmithError",
    "GraphError",
    "TraceError",
]
```

- [ ] **Step 5: 写公共 API 的冒烟测试**

```python
# tests/test_public_api.py
import bayesmith


def test_every_exported_name_resolves():
    for name in bayesmith.__all__:
        assert hasattr(bayesmith, name), name


def test_the_readme_example_runs():
    import jax
    import jax.numpy as jnp
    import numpyro.distributions as dist

    def model(data):
        x = bayesmith.sample("x", lambda: dist.Normal(0.0, 2.0))
        bayesmith.observe("d", lambda v: dist.Normal(v, 0.5), x, obs=data)

    graph = bayesmith.trace(model, jnp.array([1.0, 2.0]))
    assert graph.latents == ("x",)
    assert jnp.isfinite(bayesmith.log_joint(graph, {"x": jnp.array(0.0)}))
    draws = bayesmith.nuts(graph, jax.random.key(0), num_warmup=200, num_samples=200)
    assert draws["x"].shape == (200,)
```

Run: `.venv/bin/python -m pytest tests/test_public_api.py -v`
Expected: 2 passed

- [ ] **Step 6: 提交**

```bash
git add -A
git commit -m "feat: public API, and pin NUTS against the conjugate closed form"
```

---

## 验收（本计划完成的判据）

- [ ] `.venv/bin/python -m pytest -q` 全绿
- [ ] `tests/test_conjugate_oracle.py` 通过：NUTS 打中解析后验，且在 n=1 与 n=200 两端都成立
- [ ] `tests/test_bridge.py::test_the_bridge_and_log_joint_agree_on_the_density` 通过：**同一张图的两条独立读法给出同一个数**——这是图核与桥彼此的交叉检验
- [ ] `tests/test_plates.py::test_vmap_agrees_with_an_explicit_python_loop` 通过：plate 是优化，不得改变数值
- [ ] `tests/test_degenerate_graphs.py` 通过：空图 / 无隐变量 / 无观测 / 菱形 DAG / 共线父节点 / 50 节点长链全部成立
- [ ] `tests/test_errors.py::test_errors_module_imports_no_heavy_dependency` 通过：`import bayesmith.errors` 不拉 jax/numpy/numpyro
- [ ] `test_a_module_fn_exposes_its_parameters_as_traceable_leaves` 与 `test_evaluate_is_differentiable_through_a_module_operator` 通过：**梯度穿透节点内的算子参数**，即 rheplicant `Pipeline` 作为确定性节点时可微

## 明确不在本计划范围内

留给后续 spec，不要顺手做：

- 结构分派器与 `InferencePlan`（P3）——本计划里 `linear_in` 只被**记录**，没有任何东西读它，更没有做线性性检验
- 任何精确解（Wiener/GCR/GLS/RTS）（P3）
- 离散枚举与前向-后向（P4）
- 可辨识性、先验敏感度、线性性检验（P5）
- 流式证据层（P6）
- 嵌套 plate（`Graph.__check_init__` 会带理由拒绝）
- 序列化、GUI、config 层
