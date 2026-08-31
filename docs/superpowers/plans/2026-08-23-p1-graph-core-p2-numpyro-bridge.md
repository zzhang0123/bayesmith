# bayesmith P1 图核 + P2 NumPyro 桥 — 实施计划

> **文档状态：`record`** · 已落地批次/审计/测量的历史记录，写作当天为真，非当前权威。索引见 docs/README.md。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 bayesmith 能把一个模型函数追踪成显式静态图，在图上算出联合对数密度，并把任意图桥接到 NumPyro 跑 NUTS——形成一个最小可行、且能自我验证的闭环。

**Architecture:** 四种节点（`Const`/`Deterministic`/`Probabilistic`，共同基类 `Node`）都是 `eqx.Module`。`Graph` 按拓扑序持有节点元组，在 `__check_init__` 里验证。追踪器用一个模块级栈记录节点，四个原语（`const`/`det`/`sample`/`observe`）返回 `NodeRef`，把 `NodeRef` 传给下游即声明父子边。求值是一次线性扫描；`log_joint` 在其上累加概率节点的 `log_prob`。桥接器把同一次扫描重写成 `numpyro.sample` 调用。

**关键设计约束（已原型验证，勿改）：** `Deterministic.fn` 与 `Probabilistic.dist_fn` **必须是非静态字段**。lambda 会成为非数组叶子被 `filter_jit` 过滤；而当 `fn` 是 `eqx.Module`（例如一整条 rheplicant `Pipeline`）时，它的参数是可追踪叶子，`eqx.filter_grad` 能穿透到它们。改成 `static=True` 会让含数组的 Module 触发 equinox 的静态字段报错。

**Tech Stack:** Python ≥3.11、JAX、Equinox、NumPyro、pytest。

**测试纪律（Task 0 的质量审查发现后补入，适用于全部任务）：** 断言必须验证**行为**，不能是由 class 语句或类型签名本身保证为真的重言式。`assert issubclass(X, Y)` 在 `class X(Y)` 已经写死的情况下不证明任何东西——名字里说"可捕获"，就必须真的 `raise` 并 `pytest.raises` 捕获。每写一个测试，问一句：**如果实现是错的，这条断言会红吗？** 答不上来就重写。

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
    """A graph was declared or evaluated inconsistently.

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
    with pytest.raises(ValueError):
        raise GraphError("bad graph")


def test_trace_error_is_catchable_as_the_family_and_as_runtime_error():
    assert issubclass(TraceError, BayesmithError)
    assert issubclass(TraceError, RuntimeError)
    with pytest.raises(BayesmithError):
        raise TraceError("primitive called outside trace()")
    with pytest.raises(RuntimeError):
        raise TraceError("primitive called outside trace()")


def test_errors_module_imports_no_heavy_dependency():
    """errors.py is on every import path, so it must stay stdlib-only."""
    import subprocess
    import sys

    code = (
        "import bayesmith.errors, sys; "
        "print(sorted({'jax', 'numpy', 'numpyro'} & set(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
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


class ScaledNormal(eqx.Module):
    """Stand-in for a noise model carrying its own parameters."""

    scale: jax.Array

    def __call__(self, loc):
        return dist.Normal(loc, self.scale)


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


def test_a_module_dist_fn_exposes_its_parameters_as_traceable_leaves():
    """Same rheplicant-compatibility property, for ``Probabilistic.dist_fn``.

    Latent (``observed=None``) so the node's only leaf is dist_fn's own
    parameter -- an observed array would add a second leaf and complicate
    the leaf-count assertion below.
    """
    n = Probabilistic(
        name="d",
        parents=("x",),
        plate=(),
        dist_fn=ScaledNormal(scale=jnp.array(2.0)),
        observed=None,
    )
    leaves = jax.tree.leaves(n)
    assert len(leaves) == 1
    assert eqx.is_inexact_array(leaves[0])

    grad = eqx.filter_grad(
        lambda node, loc: node.dist_fn(loc).log_prob(jnp.array(1.0))
    )(n, jnp.array(0.5))
    assert jnp.isfinite(grad.dist_fn.scale)
    assert grad.dist_fn.scale != 0.0


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
        name="d", parents=("x",), plate=(),
        dist_fn=lambda m: dist.Normal(m, 1.0),
        observed=jnp.array([1.0, 2.0]),
    )
    assert latent.is_latent
    assert not seen.is_latent


def test_every_node_type_is_a_node():
    for cls in (Const, Deterministic, Probabilistic):
        assert issubclass(cls, Node)
```

> 这个 `dist_fn` 测试是 Task 1 的质量审查补上的：原来三个"不得为 static"的测试**只覆盖了 `Deterministic.fn`**，而 `nodes.py` 的 docstring 宣称这条理由同时适用于两个字段。若把 `dist_fn` 翻成 static，整套测试不会变红——一个有文档、无守卫的不变量。已实测：翻成 static 后正是这个测试变红。

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
these fields ``static=True`` would not raise on the Module case: equinox
does not refuse a JAX array in a static field, it only warns. Construction
succeeds, the whole module is absorbed into pytree aux data, and
``eqx.filter_grad`` then silently returns each parameter's *original* value
in place of a gradient -- nothing raises, the answer is simply wrong. That
is the stronger argument for keeping these fields non-static: not a
constructor error to catch, but a silent wrong answer.
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
Expected: 7 passed

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


def test_unknown_plate_name_is_refused_by_name():
    g = Graph(nodes=(), plates=(Plate(name="obs", size=3),))
    with pytest.raises(GraphError, match="no plate named 'nope'"):
        g.plate_size("nope")


def test_a_duplicate_plate_name_is_refused():
    n = Const(name="X", parents=(), plate=("obs",), value=jnp.arange(3.0))
    plates = (Plate(name="obs", size=3), Plate(name="obs", size=5))
    with pytest.raises(GraphError, match="duplicate plate name 'obs'"):
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
        plate_names: set[str] = set()
        for p in self.plates:
            if p.name in plate_names:
                raise GraphError(
                    f"duplicate plate name {p.name!r}: plate names must be "
                    "unique within a graph."
                )
            plate_names.add(p.name)

        seen: set[str] = set()
        for node in self.nodes:
            if node.name in seen:
                raise GraphError(
                    f"duplicate node name {node.name!r}: node names must be "
                    "unique within a graph."
                )
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
Expected: 11 passed

- [ ] **Step 5: 提交**

```bash
git add src/bayesmith/graph/graph.py tests/test_graph.py
git commit -m "feat: graph container with structural validation"
```

---

> Task 2 的质量审查用**变异测试**逐条禁用 `__check_init__` 的规则，确认每条规则恰好对应一个失败测试。它还补上了两处：`plate_size()` 的报错路径原本零覆盖（把 `raise` 换成 `return` 全绿），以及重复的 plate 名原本被静默吞掉——而 Task 6 的 vmap 正是从 `plate_size()` 取轴长，静默吞掉会在很远处以形状错误现身。

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


@pytest.mark.parametrize("bad_plate", [123, "obs", 3.5])
def test_a_non_plateref_plate_value_is_refused(bad_plate):
    """plate() returns a handle, not a name -- nothing else is accepted.

    Covers both the old silent bug (a bare string used to work, since
    plate() itself used to return one) and the old crash (a bare int used to
    escape as an uncaught TypeError instead of an actionable GraphError).
    """

    def model():
        const("X", jnp.arange(3.0), plate=bad_plate)

    with pytest.raises(GraphError, match="PlateRef"):
        trace(model)


def test_a_primitive_outside_trace_is_refused():
    with pytest.raises(TraceError, match="must be called inside trace"):
        sample("x", lambda: dist.Normal(0.0, 1.0))


def test_a_duplicate_name_is_refused_during_tracing():
    def model():
        sample("x", lambda: dist.Normal(0.0, 1.0))
        sample("x", lambda: dist.Normal(0.0, 1.0))

    with pytest.raises(GraphError, match="duplicate node name 'x'"):
        trace(model)


def test_a_duplicate_plate_name_is_refused_during_tracing():
    def model():
        plate("obs", 3)
        plate("obs", 5)

    with pytest.raises(GraphError, match="duplicate plate name 'obs'"):
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


def test_a_noderef_from_an_outer_trace_is_refused_by_an_inner_trace():
    """A handle must resolve by owner, not by name, across trace() calls.

    Reproduces the silent-misattachment failure mode this guards against:
    without an owner check, the inner "shared" Const would silently satisfy
    a parent reference actually meant for the outer Probabilistic of the
    same name, because parents were resolved by name alone.
    """
    outer = {}

    def inner_model():
        const("shared", jnp.array(42.0))
        det("y", lambda v: v, outer["handle"])

    def outer_model():
        outer["handle"] = sample("shared", lambda: dist.Normal(0.0, 1.0))
        trace(inner_model)

    with pytest.raises(GraphError, match="different trace"):
        trace(outer_model)


def test_a_plateref_from_a_different_trace_is_refused():
    """A plate handle must resolve by owner, not by name, across trace() calls."""
    outer = {}

    def inner_model():
        plate("obs", 10)  # same name as the outer plate, unrelated otherwise
        const("X", jnp.arange(3.0), plate=outer["handle"])

    def outer_model():
        outer["handle"] = plate("obs", 3)
        trace(inner_model)

    with pytest.raises(GraphError, match="different trace"):
        trace(outer_model)
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


def test_a_value_for_an_observed_node_explains_it_is_observed():
    """The refusal names the actual reason, not a generic "not latent".

    An observed node is the most likely real mistake -- someone who has
    not internalised the latent/observed split and tries to pass all their
    data through ``values`` -- so it gets its own explanation rather than
    sharing text with the deterministic/constant/unknown-name cases.
    """
    with pytest.raises(
        GraphError, match="values names 'd', which is an observed node"
    ):
        evaluate(_linear_model(), {"a": jnp.array(1.0), "d": jnp.zeros(3)})


def test_evaluate_is_differentiable_through_a_module_operator():
    """A parameterised operator inside a node stays differentiable.

    X and w are chosen so the true gradient (``sum(X)``) cannot coincide
    with w's own value, and the expectation is computed from X rather than
    written as a bare literal that could drift back into coincidence. Both
    guard against the same failure mode: equinox does not refuse a JAX
    array in a static field, it only warns -- so if ``fn`` were ever made
    static, the whole ``Scale`` module would be absorbed into pytree aux
    data and ``eqx.filter_grad`` would silently return each leaf's
    *original* value in place of a gradient. Nothing raises. If w happened
    to equal ``sum(X)``, that wrong answer would be indistinguishable from
    the right one.
    """
    X = jnp.array([1.0, 5.0])
    w = jnp.array(2.0)

    def model():
        Xc = const("X", X)
        det("mu", Scale(w=w), Xc)

    graph = trace(model)

    def total(g):
        return jnp.sum(evaluate(g, {})["mu"])

    grad = eqx.filter_grad(total)(graph)
    assert jnp.allclose(grad.nodes[1].fn.w, jnp.sum(X))


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
        name = min(unknown)
        if name not in graph.names:
            raise GraphError(
                f"values names {name!r}, which does not name any node in "
                f"this graph. Latents are {list(graph.latents)}."
            )
        if isinstance(graph.node(name), Probabilistic):
            # In graph.names but excluded from graph.latents above, and
            # Probabilistic: the only way to be both is to be observed.
            raise GraphError(
                f"values names {name!r}, which is an observed node: its "
                "value comes from observe(..., obs=...), not from `values`. "
                f"Latents are {list(graph.latents)}."
            )
        raise GraphError(
            f"values names {name!r}, which is a deterministic or constant "
            "node: its value is computed, not supplied via `values`. "
            f"Latents are {list(graph.latents)}."
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
        else:
            # Live defensive code, not dead code: Node is a plain eqx.Module
            # (not an ABC) and Graph.__check_init__ does not check node
            # subtype, so a hand-built Graph containing a bare Node (or any
            # subclass other than Const/Deterministic/Probabilistic) reaches
            # this branch and raises correctly here.
            raise GraphError(f"unknown node type {type(node).__name__}")  # pragma: no cover
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


def test_a_plated_latent_site_carries_the_plate_axis():
    """A dropped plate is invisible to the density-agreement check: without
    subsampling, ``numpyro.plate`` contributes no scale factor, and the
    plated test above only plates an *observed* site of fixed shape. Plate a
    *latent* instead: ``dist.Normal(0.0, 1.0)`` has batch shape ``()``, so a
    sampled value only picks up a leading axis if the plate was actually
    applied. Plate size 5 appears nowhere else in this file (the other
    sizes are 3 and 8), so a wrong-but-plausible shape can't pass by luck.
    """
    n = 5

    def model():
        obs = plate("obs", n)
        sample("x", lambda: dist.Normal(0.0, 1.0), plate=obs)

    graph = trace(model)
    trace_ = numpyro.handlers.trace(
        numpyro.handlers.seed(to_numpyro(graph), jax.random.key(0))
    ).get_trace()
    assert trace_["x"]["value"].shape == (n,)
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

TAU is deliberately smaller than SIGMA: the prior precision 1/tau^2 = 4.0 is
16x the per-observation precision 1/sigma^2 = 0.25, so the prior supplies
about 94% of the posterior precision at n=1, dropping to about 44% at n=20
and 7% at n=200. That gradient is what makes a silently-dropped prior term
observable at all: deleting it pulls the posterior mean toward the raw data
mean by an offset set by the prior, while NUTS keeps sampling the true
(correct) posterior regardless -- and because the posterior also concentrates
sharply as n grows, that fixed-looking offset shows up as an enormous
z-score at every n this file sweeps, not only at the small-n end where the
prior's precision share is largest. (An earlier version of these constants,
TAU=2.0 and SIGMA=0.5, had the ratio backwards: the likelihood so dominated
even at n=1 that dropping the prior term entirely still landed inside this
test's tolerances -- an oracle that cannot fail this way is not fit to be
the package's acceptance gate.)

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

TAU = 0.5
SIGMA = 2.0


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

from __future__ import annotations

import importlib
from typing import Any

from bayesmith.errors import BayesmithError, GraphError, TraceError

__all__ = [
    # tracing
    "trace",
    "const",
    "det",
    "sample",
    "observe",
    "plate",
    "NodeRef",
    "PlateRef",
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

# Every public name above except the three error classes is resolved lazily,
# on first attribute access, rather than imported here at module scope.
# Importing eagerly would make `import bayesmith` load numpyro (hence jax) as
# a side effect -- exactly the regression this module previously had: Python
# always runs a package's __init__.py before any of its submodules, so even
# `import bayesmith.errors` was dragging in the whole bridge, which broke the
# stdlib-only contract errors.py documents for itself and which
# test_errors_module_imports_no_heavy_dependency enforces. Only errors.py is
# cheap and stdlib-only, so it alone is still imported eagerly above.
#
# name -> (owning submodule, attribute name within it)
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "trace": ("bayesmith.graph.trace", "trace"),
    "const": ("bayesmith.graph.trace", "const"),
    "det": ("bayesmith.graph.trace", "det"),
    "sample": ("bayesmith.graph.trace", "sample"),
    "observe": ("bayesmith.graph.trace", "observe"),
    "plate": ("bayesmith.graph.trace", "plate"),
    "NodeRef": ("bayesmith.graph.trace", "NodeRef"),
    "PlateRef": ("bayesmith.graph.trace", "PlateRef"),
    "Graph": ("bayesmith.graph.graph", "Graph"),
    "Plate": ("bayesmith.graph.graph", "Plate"),
    "Node": ("bayesmith.graph.nodes", "Node"),
    "Const": ("bayesmith.graph.nodes", "Const"),
    "Deterministic": ("bayesmith.graph.nodes", "Deterministic"),
    "Probabilistic": ("bayesmith.graph.nodes", "Probabilistic"),
    "evaluate": ("bayesmith.graph.evaluate", "evaluate"),
    "log_joint": ("bayesmith.graph.evaluate", "log_joint"),
    "to_numpyro": ("bayesmith.bridge.numpyro_bridge", "to_numpyro"),
    "nuts": ("bayesmith.bridge.numpyro_bridge", "nuts"),
}

# Subpackages reachable as `bayesmith.<name>` after a bare `import bayesmith`,
# without eagerly importing any of them -- `bridge` in particular is what
# pulls in numpyro. `errors` is listed too for __dir__'s sake even though the
# eager import above already binds it as a real attribute, so __getattr__ is
# never actually consulted for it.
_LAZY_SUBMODULES = ("graph", "bridge", "errors")


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        value = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = value  # cache: later lookups skip __getattr__
        return value
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS) | set(_LAZY_SUBMODULES))
```

- [ ] **Step 5: 写公共 API 的冒烟测试**

```python
# tests/test_public_api.py
import bayesmith


def test_every_exported_name_resolves():
    for name in bayesmith.__all__:
        assert hasattr(bayesmith, name), name


def test_importing_bayesmith_stays_light():
    """``import bayesmith`` must not pull in jax or numpyro.

    A subprocess, not an in-process ``sys.modules`` check: by the time this
    test runs, pytest's own process has almost certainly already imported
    jax/numpyro via other test modules, which would make an in-process check
    pass regardless of what ``bayesmith/__init__.py`` actually does. Mirrors
    ``test_errors_module_imports_no_heavy_dependency`` in ``test_errors.py``
    -- and pins the more general claim that one relies on: Python always
    runs a package's ``__init__.py`` before any of its submodules, so a bare
    ``import bayesmith`` is the more direct thing to check, and the one that
    was actually broken (an eager ``__init__.py`` drags jax/numpyro in even
    for ``import bayesmith.errors`` alone).
    """
    import subprocess
    import sys

    code = (
        "import bayesmith, sys; "
        "print(sorted({'jax', 'numpy', 'numpyro'} & set(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


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

## 执行记录：质量审查发现了什么

本计划的代码块已回填为仓库的实际内容，与实现 AST 全量一致（Task 0 的 `__init__.py` 除外，那是它当时的正确产物，Task 9 替换了它）。

八轮代码质量审查发现了 **8 个真实缺陷**。值得记录的是它们的分布：**没有一个是实施错误**——每个任务的代码都逐字符符合计划、AST 比对全过。全部 8 个都源于计划本身，形状高度一致：**有文档、无守卫，或守卫看似存在实则无效**。

| # | 任务 | 缺陷 | 为何三道关里只有质量审查能发现 |
|---|---|---|---|
| 1 | 0 | 名字承诺"可捕获"的测试只有 `issubclass` 断言 | 规格比对通过——因为计划里写的就是这样 |
| 2 | 1 | docstring 称"两个字段都不能是 static"，但只有 `fn` 有测试保护 | 需要跨文件推理"这条不变量有守卫吗" |
| 3 | 2 | `plate_size()` 的 `Raises:` 契约零覆盖 | 需要跑覆盖率或变异测试 |
| 4 | 2 | 重复 plate 名被静默吞掉（重复节点名却会报错） | 需要注意到同一函数内的不对称 |
| 5 | 3 | `NodeRef` 只带名字 → 跨 trace 的 handle 静默绑到同名的别的节点 | 需要构造跨作用域场景 |
| 6 | 4 | 可微性测试的 fixture 数值巧合（`w=3.0`，`sum(X)=3.0`），使**本包最核心保证**的唯一守卫完全无效 | 只有变异测试能发现 |
| 7 | 7 | 删掉 bridge 的 plate 包装，5 个测试全绿 | 只有变异测试能发现 |
| 8 | 9 | 公共 API 的急切导入打破了 `errors.py` 的"仅 stdlib"不变量 | 全套测试运行时才暴露 |

另外两处是我在写计划时的事实性错误，由审查纠正：

- **equinox 0.13.8 不会拒绝静态字段里的 JAX 数组**，只发一条 `UserWarning`。后果比报错糟：模块被吸进 pytree aux 数据，`filter_grad` 静默返回参数原值而非梯度。
- **共轭预言机原本对"先验整个丢失"不敏感**：`TAU=2.0, SIGMA=0.5` 下似然精度是先验的 16 倍，即使 n=1 先验也只占不到 6% 的后验精度。改成 `TAU=0.5, SIGMA=2.0` 后，删除先验项在三个 n 上分别产生 10.4σ / 66.1σ / 34.6σ 的偏离。

**方法论上的两点收获**，可用于后续 P3–P7：

1. **规格合规应当机器化。** 把计划的代码块与提交的文件做 AST 比对，比让代理读两遍文件更强也更快（几秒钟），而且立刻能区分实质差异与排版差异。
2. **变异测试是唯一能发现"守卫无效"的手段。** 上表 8 个缺陷里有 3 个（#3、#6、#7）只有靠"故意破坏实现、看测试是否变红"才能暴露。应把它作为每个任务的标准动作，而不是审查者的自选项。

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
