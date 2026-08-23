# bayesmith P3a 精确解核心 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 bayesmith 能把一张图里的一组隐变量导出成无矩阵线性算子，检验它们被声明的线性性，并在其上求出线性高斯后验的精确均值与精确抽取——而正确性由一条**与被测者零共享代码**的 NumPy 稠密预言机判定。

**Architecture:** 图取代 rheplicant 的 `ParameterSpace`。观测节点分布的**位置参数**就是"预测"，其**尺度**就是 σ，隐变量分布的位置与尺度就是先验的 `(m, √S)`——三者由同一个提取器读出（内省做快路径，`log_prob` 探针做守卫）。块的定义域与陪域都是 dict（`{latent: x}` → `{obs: loc}`），`A` 来自 `jax.linearize`、`Aᵀ` 来自 `jax.vjp`，全程不形成矩阵。求解走 `jax.scipy.sparse.linalg.cg`，均值与抽取共用一个私有 `_conjugate_solve`，差一个 `key` 参数。

**上游 spec:** `docs/superpowers/specs/2026-08-23-p3-structural-dispatch-design.md`。本计划实现其 §一、§二（检验部分）、§三、§七的 R1/R2/R3。**分派器、`InferencePlan`、Gibbs、SNIS/MH 修正全部属于 P3b，不在本计划范围内**（§九）。

**移植源:** `/Users/zzhang/projects/e-RHINO/src/rheplicant/inference/{linear,gls,uncertainty,conditioning}.py`——**只读，不改动 e-RHINO 的任何文件**。

**Tech Stack:** Python ≥3.11、JAX 0.11.1、Equinox 0.13.8、NumPyro 0.21.0、NumPy 2.5.2、pytest。

---

## 三条纪律（P1 执行记录留下的，适用于每个任务）

1. **断言必须验证行为，不能是重言式。** `assert issubclass(X, Y)` 在 `class X(Y)` 已写死时不证明任何东西。每写一条断言问一句：**如果实现是错的，这条会红吗？** 答不上来就重写。
2. **每个任务收尾做变异测试。** 故意按任务指定的方式破坏实现，确认一条**具名**的测试变红，然后还原。P1 的 8 个缺陷里有 3 个只有它能发现。
3. **凡断言"结果等于某常数"的测试，该常数不得等于任何参数的当前值。** P1 的 Task 4 用 `w=3.0` 而真实梯度 `sum(X)=3.0`，使本包最核心的保证的唯一守卫完全无效。
4. **每个任务收尾做 AST 规格比对。** 把本计划该任务的代码块与提交的文件做 AST 比对，而不是让人读两遍——几秒钟，且立刻能区分实质差异与排版差异。P1 的执行记录把它列为两条方法论收获之一。脚本：

```bash
cat > /tmp/ast_compare.py << 'PY'
import ast, sys
a, b = (ast.dump(ast.parse(open(p).read())) for p in sys.argv[1:3])
print("IDENTICAL" if a == b else "DIFFERENT")
sys.exit(0 if a == b else 1)
PY
```

   用法：把计划里该任务的 python 代码块抠出来存成一个文件，再与提交的源文件比对。**实质差异是可以的**——发现了计划的缺陷就该改实现——但每一处都要在任务收尾时**具名说明**，而不是悄悄漂移。

## 实测更正：删掉 `extreme_eigenvalues`，守卫改用先验界（2026-08-23，Task 1 代码审查）

计划初稿沿用 rheplicant 的做法：用 `extreme_eigenvalues` 在 `λmax·I − M` 上做第二次幂迭代求 `λ_min`，再取 `κ = λmax / max(λ_min, floor)`。**实测证明这在梯度谱上原理性地不成立，且偏差在危险的一侧。**

移位算子的谱是 `λmax − eig`。对梯度谱，它的前若干个特征值全都挤在 `λmax` 附近、彼此间隙趋近于零——幂迭代收敛不了，**加迭代次数没用**。实测（float32，`jax.random.key(0)`）：

| 谱 | 真 κ | 12 次迭代测得 | 2000 次迭代 | `λmax × max(先验方差)` |
|---|---|---|---|---|
| 双簇 `3×1e6 + 3×1` | 1e6 | 9.4e5（×0.94） | — | 1e6（×1.00） |
| 双簇 `20×1e6 + 5×1` | 1e6 | 1e6（×1.00） | — | 1e6（×1.00） |
| **梯度 50 点几何，κ=1e7** | 1e7 | **179（×1.8e-5）** | 14025（×1.4e-3） | 1e7（×1.00） |
| 宽 3 点 `{2, 1e3, 1e6}` | 5e5 | 1068（×2.1e-3） | 4582（×9.2e-3） | 5e5（×1.00） |

**偏差方向是危险的那一侧**：λ_min 被高估 → κ 被低估 → `error_bound = residual × κ` 被低估 → **守卫在该报警时保持沉默**，而这正是它存在要防的事。原计划里的 `jnp.maximum(smallest, floor)` 护的是**另一个**方向（低估 λ_min），因此完全不咬。

**替代方案来自同一张表。** `AᵀN⁻¹A` 半正定，所以

    λ_min(AᵀN⁻¹A + S⁻¹) ≥ λ_min(S⁻¹) = 1 / max(先验方差)

这是**严格下界**，于是 `λmax × max(先验方差)` 是 κ 的**上界**（在 λmax 自身估计的精度之内——那一半收敛快且从下方逼近，`test_largest_eigenvalue_approaches_the_truth_from_below` 钉住这一点）。上表末列显示它在四种谱上都紧，包括幂迭代彻底失效的两种。

**决定：**

1. `conditioning.py` **只保留 `largest_eigenvalue`**，删掉 `extreme_eigenvalues`。它在 P3a 里没有使用者，且对其唯一用途已被实测证明不可用——留着是会误导人的死代码。rheplicant 里它仍在（服务于另一条 `identifiability` 路径）；此处记录，免得将来有人照着 rheplicant 又移植一遍。
2. Task 5 的出口改名 `condition_estimate` → **`condition_bound`**，实现为 `largest_eigenvalue(...) × largest_variance(prior_variance)`。
3. 成本**减半**：一次幂迭代而非两次。
4. 语义从"估计"变成"**界**"：守卫的失效方向从"可能静默过关"变成"最多虚报"。虚报只在数据把每个方向都约束得远好于先验时出现，而那恰是 CG 本来就轻松收敛、残差极小从而把松弛吸收掉的区域。

## 一条排版约定

计划里每个 python 代码块的**首行路径注释**（`# src/bayesmith/exact/conditioning.py`）是给读计划的人看的**元信息**，**不是文件内容**——文件的首行应当是它自己的 docstring。Task 1 首次实现时把它抄进了源文件，之后剥离；后续每个文件都适用这条。

同理，块里的 `# tests/exact/xxx.py` 也不抄。

## 一条精度纪律

**本包绝不在任何位置调用 `jax.config.update("jax_enable_x64", ...)`**——进程级全局，会静默改变宿主之后创建的每个数组的 dtype，且关不回去。需要 float64 时用 `with jax.enable_x64(True):`，并在块内转出到 NumPy。需要 x64 的测试打 `@pytest.mark.x64`（marker 已在 `pyproject.toml` 声明）。

---

## 文件结构

| 文件 | 职责 | 行数 |
|---|---|---|
| `src/bayesmith/errors.py` | 增补 `StructureError`、`ConvergenceError`、`NotGaussian`。仍仅 stdlib | +30 |
| `src/bayesmith/exact/__init__.py` | 空的包标记 | 1 |
| `src/bayesmith/exact/conditioning.py` | `tree_norm`、`largest_eigenvalue`。不认识图，也不认识块 | ~90 |
| `src/bayesmith/exact/gaussian.py` | `(loc, scale)` 提取器 + `log_prob` 探针守卫 + 形状规则 + 观测/先验接缝 | ~200 |
| `src/bayesmith/exact/block.py` | `LinearBlock`、定义域工具、`unchecked_operator(graph, names, at)` | ~240 |
| `src/bayesmith/exact/linearity.py` | `affinity_errors`、`check_linearity`（先验幅度、多 `at` 点） | ~190 |
| `src/bayesmith/exact/solve.py` | `condition_bound`、`wiener_solve`、`gcr_sample`、私有 `_conjugate_solve` | ~270 |
| `src/bayesmith/exact/gls.py` | `GLSResult`、`iterative_gls` | ~200 |
| `src/bayesmith/exact/fisher.py` | `FlatMatrix`、`dense_operator`、`fisher_information`、`parameter_covariance` | ~230 |
| `tests/exact/__init__.py` | 空 | 0 |
| `tests/exact/oracle.py` | **R2 预言机**：探基稠密 `A`，NumPy 解析后验。零 autodiff、零 jax 变换 | ~90 |
| `tests/exact/models.py` | 共享的玩具图构造器 | ~120 |
| `tests/exact/test_*.py` | 每模块一个，外加验收关口 | — |

**依赖方向**：`conditioning` → 无；`gaussian` → `graph`；`block` → `gaussian` + `graph`；`linearity` → `block` + `bridge`；`solve` → `block` + `conditioning`；`gls` → `solve`；`fisher` → `block`。无环。

---

## Task 0：三个新异常类

**Files:**
- Modify: `src/bayesmith/errors.py`
- Test: `tests/test_errors.py`

- [ ] **Step 1: 追加三个类**

在 `src/bayesmith/errors.py` 末尾追加：

```python
class StructureError(BayesmithError, ValueError):
    """A declared structural claim was checked and found false.

    Raised where a declaration and the model contradict each other: a
    ``Deterministic`` declaring ``linear_in`` whose prediction is not affine
    in that parent, a ``Probabilistic`` declaring
    ``depends_on_prediction=False`` whose scale does in fact move with the
    block, a ``dist_fn`` whose *type* says Normal but whose ``log_prob`` does
    not match the ``loc``/``scale`` read off it.

    **Not** raised when a graph merely fails to qualify for an exact method.
    "You did not declare it" is a dispatch outcome -- the block falls through
    to NUTS and the plan records why. "You declared it, and it is false" is
    this. Conflating the two would make an ordinary model that simply has no
    exact structure look like a broken one.
    """


class ConvergenceError(BayesmithError, RuntimeError):
    """An iterative procedure did not reach the accuracy it was asked for.

    For guards *inside* a jitted solve the mechanism is ``equinox.error_if``
    instead, because a Python ``if`` cannot branch on a traced value. This
    class is for the checks that run on concrete values, outside any trace --
    ``iterative_gls``'s ``converged=False`` promoted to a hard failure, say.
    """


class NotGaussian(BayesmithError, TypeError):
    """A node's distribution is not a diagonal Gaussian.

    Purely descriptive, and deliberately blameless: most models contain
    perfectly good non-Gaussian nodes. P3b's classifier catches this and
    routes the block to NUTS.

    **A sibling of** :class:`StructureError`. A dispatcher writing
    ``except NotGaussian`` must NOT also swallow a :class:`StructureError`: that one means a node's *type*
    says Normal while its own ``log_prob`` says otherwise, and silently
    downgrading it to NUTS would hide a broken model behind an
    ordinary-looking fallback.

    A subclass relationship, in either direction, is the ONE thing that would
    break this -- and it is worth being precise about which, because two
    classes that merely *share* a base are unaffected. ``except`` matches on
    the raised exception's MRO, not on a common ancestor: these two already
    share :class:`BayesmithError`, and neither catches the other. Measured,
    because an earlier draft of this docstring claimed otherwise.

    The differing builtin bases -- ``TypeError`` here against
    :class:`StructureError`'s ``ValueError`` -- therefore buy something
    narrower, and something real: a *generic* handler can tell the two apart,
    so an ``except ValueError`` around a modelling call sees the broken-model
    case and not the ordinary not-conjugate one.
    """
```

- [ ] **Step 2: 扩展 stdlib-only 测试**

`tests/test_errors.py` 里已有 `test_errors_module_imports_no_heavy_dependency`。**扩展它**，把三个新名字的存在性检查折进同一个子进程——导入模块这一件事同时证明两件事，另开一个子进程只会把前一半再证一遍：

```python
def test_errors_module_imports_no_heavy_dependency():
    """errors.py is on every import path, so it must stay stdlib-only.

    The name check for P3's three classes rides along in this same
    subprocess rather than getting one of its own: importing the module is
    what proves both, so a second spawn would only re-prove the first half.
    """
    import subprocess
    import sys

    code = (
        "import bayesmith.errors as e, sys; "
        "assert e.StructureError and e.ConvergenceError and e.NotGaussian; "
        "print(sorted({'jax', 'numpy', 'numpyro'} & set(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
```

再追加这一条。它是 Task 0 唯一的**行为**测试——`assert not issubclass(...)` 不算数，那只是把 class 语句重述一遍，正是 P1 第一条审查发现的重言式：

```python
def test_catching_not_gaussian_does_not_also_catch_structure_error():
    """The sibling relationship is load-bearing, so it is tested by behaviour.

    P3b's classifier writes `except NotGaussian` to mean "this block has no
    exact structure, route it to NUTS". A StructureError means something else
    entirely -- a node whose type says Normal while its own log_prob
    disagrees -- and must escape that clause.

    The two halves cover the two directions a hierarchy could collapse in,
    and each catches exactly one: half (a) goes red if StructureError is made
    a subclass of NotGaussian, half (b) if NotGaussian is made a subclass of
    StructureError. Giving the two the same builtin base changes nothing here
    and this test stays green -- correctly so, because `except` matches on
    the MRO and not on a shared ancestor.
    """
    from bayesmith.errors import NotGaussian, StructureError

    escaped = False
    try:
        try:
            raise StructureError("the log_prob probe disagreed")
        except NotGaussian:  # pragma: no cover - must not be taken
            pass
    except StructureError:
        escaped = True
    assert escaped

    with pytest.raises(NotGaussian):
        try:
            raise NotGaussian("this node is a Gamma")
        except StructureError:  # pragma: no cover - must not be taken
            pass
```

`tests/test_errors.py` 顶部需要 `import pytest`（若尚未导入）。

- [ ] **Step 3: 跑测试**

```bash
.venv/bin/python -m pytest tests/test_errors.py -v
```

Expected: 全部 PASS（原有 3 条，其中一条被扩展，加新增 1 条 = 4 条）。

- [ ] **Step 4: 变异测试**

在 `errors.py` 里把 `class StructureError(BayesmithError, ValueError):` 临时改成从 `numpy` 导入什么东西（例如在文件顶加 `import numpy`），重跑上面的测试。

Expected: `test_errors_module_imports_no_heavy_dependency` **变红**（打印出 `['numpy']` 而非 `[]`）。还原。

再把 `class NotGaussian(BayesmithError, TypeError):` 临时改成 `class NotGaussian(StructureError):`，重跑。
Expected: `test_catching_not_gaussian_does_not_also_catch_structure_error` **变红**。还原。

把 `NotGaussian` 改名为 `NotGaussianXYZ`（仅类名），重跑。
Expected: `test_errors_module_imports_no_heavy_dependency` **变红**——这条证明折进去的名字检查确实还在鉴别。还原。

**最后一条是用来验证 docstring 本身的**：把 `class NotGaussian(BayesmithError, TypeError):` 改成 `class NotGaussian(BayesmithError, ValueError):`（与 `StructureError` 同基类，但仍非父子），重跑。
Expected: `test_catching_not_gaussian_does_not_also_catch_structure_error` **保持绿**。若它变红，说明修正后的 docstring 也是错的。还原。

- [ ] **Step 5: 提交**

```bash
git add src/bayesmith/errors.py tests/test_errors.py
git commit -m "feat: add StructureError, ConvergenceError and NotGaussian"
```

---

## Task 1：谱诊断（`exact/conditioning.py`）

无矩阵求解器只能便宜地报告 `‖Mx−b‖`，而调用方真正想知道的是 `‖x−x*‖`。两者差一个条件数，所以任何诚实的收敛守卫都需要谱的两端——且不能形成矩阵。

本模块**不认识图，也不认识块**：算子是一个 callable，数据是 pytree。这让数值与模型机制可分离，依赖单向。

**只交付 `largest_eigenvalue`。** 谱的另一端不靠幂迭代求——见上文「实测更正」：那条路在梯度谱上原理性失效，且偏差在危险的一侧。`λ_min` 改由先验曲率给出严格下界，Task 5 的守卫因此用的是 κ 的**上界**而非估计。

**Files:**
- Create: `src/bayesmith/exact/__init__.py`, `src/bayesmith/exact/conditioning.py`
- Create: `tests/exact/__init__.py`, `tests/exact/test_conditioning.py`

- [ ] **Step 1: 建包标记**

```bash
mkdir -p src/bayesmith/exact tests/exact
printf '"""Structure-dispatched exact solves."""\n' > src/bayesmith/exact/__init__.py
: > tests/exact/__init__.py
```

- [ ] **Step 2: 写失败的测试**

```python
"""Spectral diagnostics: a known spectrum, and the float32 overflow it survives."""

import jax
import jax.numpy as jnp
import pytest

from bayesmith.errors import GraphError
from bayesmith.exact.conditioning import largest_eigenvalue, tree_norm


def _diagonal(diag):
    """A symmetric positive-definite operator over a one-leaf pytree."""
    return lambda parts: {"x": diag * parts["x"]}


def test_tree_norm_matches_the_flattened_euclidean_norm():
    parts = {"a": jnp.array([3.0, 4.0]), "b": jnp.array([[12.0]])}
    # sqrt(9 + 16 + 144) = 13 exactly, and 13 is not any input value.
    assert float(tree_norm(parts)) == pytest.approx(13.0)


def test_tree_norm_survives_a_leaf_whose_square_overflows_float32():
    """The naive sum-of-squares route really does overflow here.

    Asserting that first is what makes the second assertion mean something:
    without it this test would pass against an implementation that never
    needed the rescale.
    """
    big = jnp.array([3e19, 4e19], dtype=jnp.float32)
    assert not jnp.isfinite(jnp.sum(big**2))
    assert float(tree_norm({"x": big})) == pytest.approx(5e19, rel=1e-5)


def test_tree_norm_survives_a_leaf_small_enough_to_underflow_when_squared():
    """The other end of the same rescale, and the naive route fails here too.

    Entries at 1e-30 square to 1e-60, which is zero in float32 -- so the naive
    implementation returns exactly 0.0 for a vector that is emphatically not
    zero. Both ends matter: a normal operator's domain spans whatever units
    the model's latents happen to be in.
    """
    small = jnp.array([3e-30, 4e-30], dtype=jnp.float32)
    assert float(jnp.sqrt(jnp.sum(small**2))) == 0.0
    # abs=0.0 is load-bearing: pytest.approx applies a DEFAULT abs=1e-12 floor
    # and takes max(rel * expected, abs), so `approx(5e-30, rel=1e-5)` accepts
    # anything within 1e-12 of it -- including the exact 0.0 the naive
    # implementation returns, which is the bug this test exists to catch.
    assert float(tree_norm({"x": small})) == pytest.approx(5e-30, rel=1e-5, abs=0.0)


def test_tree_norm_of_an_all_zero_pytree_is_zero():
    assert float(tree_norm({"x": jnp.zeros(4)})) == 0.0


def test_largest_eigenvalue_finds_the_top_of_a_known_spectrum():
    diag = jnp.array([1.0, 1.0, 1.0, 100.0])
    got = largest_eigenvalue(_diagonal(diag), {"x": jnp.zeros(4)}, jax.random.key(0), 20)
    assert float(got) == pytest.approx(100.0, rel=1e-4)


@pytest.mark.parametrize(
    "spectrum", [[1.0, 1.0, 1.0, 100.0], [1.0, 99.9, 100.0]]
)
def test_largest_eigenvalue_approaches_the_truth_from_below(spectrum):
    """Power iteration underestimates, and the guard depends on knowing it does.

    `condition_bound` divides lambda_max by a prior-derived LOWER bound on
    lambda_min to get an UPPER bound on kappa. That bound is only as good as
    lambda_max, which must therefore never overshoot.

    Both a well-separated and a nearly-degenerate spectrum, because only the
    first can catch an overshoot. Measured: `[1, 99.9, 100]` plateaus at
    99.9396 and is still 0.0029 short after 2000 iterations, so a 0.01%
    overshoot hides inside its own shortfall; `[1, 1, 1, 100]` reaches exactly
    100.0 by ten iterations, where any overshoot at all is visible. An earlier
    version of this test used only the degenerate case and could not catch the
    mutation named in its own docstring.
    """
    diag = jnp.asarray(spectrum)
    truth = float(jnp.max(diag))
    template = {"x": jnp.zeros(len(spectrum))}
    for iterations in (1, 3, 10, 40):
        got = float(
            largest_eigenvalue(
                _diagonal(diag), template, jax.random.key(4), iterations
            )
        )
        assert got <= truth * (1.0 + 1e-6), (iterations, got)


def test_largest_eigenvalue_refuses_fewer_than_one_iteration():
    """Zero iterations returns the norm of an untouched random vector.

    That is a number with no relationship to the operator at all, and it would
    flow straight into a condition bound. Refused by name rather than
    returned.
    """
    with pytest.raises(GraphError, match="iterations"):
        largest_eigenvalue(
            _diagonal(jnp.ones(3)), {"x": jnp.zeros(3)}, jax.random.key(0), 0
        )


@pytest.mark.parametrize("top_in", ["a", "b"])
def test_largest_eigenvalue_spans_several_pytree_leaves(top_in):
    """The spectrum must be the JOINT one, not any single leaf's.

    Parametrised because whichever leaf holds the top reproduces it when
    restricted to that leaf -- unavoidable -- so one case cannot catch a
    single-leaf implementation. With the top in "b", restricting to the first
    leaf reports 20 instead of 100; with it in "a", restricting to the last
    leaf does. Their union is complete. An earlier single-case version of this
    idea silently missed one of the two, found by mutation testing, which is
    the only thing that finds a guard that does not guard.
    """
    top = jnp.array([2.0, 100.0])
    rest = jnp.array([10.0, 20.0])
    diagonals = {"a": top, "b": rest} if top_in == "a" else {"a": rest, "b": top}

    def operator(parts):
        return {name: diagonals[name] * parts[name] for name in diagonals}

    template = {"a": jnp.zeros(2), "b": jnp.zeros(2)}
    got = largest_eigenvalue(operator, template, jax.random.key(1), 60)
    assert float(got) == pytest.approx(100.0, rel=1e-3)
```

- [ ] **Step 3: 跑测试，确认失败**

```bash
.venv/bin/python -m pytest tests/exact/test_conditioning.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'bayesmith.exact.conditioning'`。

- [ ] **Step 4: 实现**

```python
# src/bayesmith/exact/conditioning.py
"""Spectral diagnostics for matrix-free symmetric operators.

An iterative solver can cheaply report ``‖M x - b‖``; what a caller actually
wants is ``‖x - x*‖``. The two differ by the condition number, so an honest
convergence guard over a matrix-free operator needs the ends of its spectrum,
and needs them without ever forming a matrix.

Everything here takes the operator as a callable and works on pytrees, so it
knows nothing about :mod:`bayesmith.exact.block`'s blocks and nothing about
graphs. That keeps the numerics separable from the model machinery and the
dependency pointing one way.

**Only the top of the spectrum is measured here.** rheplicant's
``extreme_eigenvalues`` finds ``lambda_min`` by a second power iteration on
``lambda_max * I - M``; that is deliberately not ported, because it was
measured to fail in principle on a graded spectrum -- the shifted operator's
leading eigenvalues all crowd against ``lambda_max`` with vanishing gaps, so
the iteration cannot separate them however long it runs (2000 steps still
left a factor of 700 on a 50-point geometric spectrum at kappa=1e7). Worse,
the bias is one-sided in the dangerous direction: ``lambda_min`` comes back
too large, so kappa comes back too small, so a convergence guard built on it
stays silent exactly when it should fire.

``lambda_min`` is instead bounded from below by the prior's own curvature:
``A^T N^-1 A`` is positive semi-definite, so
``lambda_min(A^T N^-1 A + S^-1) >= 1 / max(prior_variance)``. See
:func:`bayesmith.exact.solve.condition_bound`, which turns that into an
UPPER bound on kappa -- the direction a safety guard needs.

Ported from ``rheplicant.inference.conditioning``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp


def tree_norm(parts: Any) -> jax.Array:
    """Euclidean norm over a pytree, scaled so it survives float32.

    Squaring first overflows for entries beyond ~1.8e19, which turns the only
    convergence signal these solvers give into ``inf/inf = NaN`` exactly when
    the problem is badly scaled and the answer is most likely wrong.
    """
    leaves = [leaf for leaf in jax.tree.leaves(parts) if eqx.is_array(leaf)]
    if not leaves:  # pragma: no cover - defensive
        return jnp.array(0.0)
    biggest = jnp.max(jnp.stack([jnp.max(jnp.abs(leaf)) for leaf in leaves]))
    biggest = jnp.where(biggest > 0, biggest, 1.0)
    total = sum(jnp.sum(jnp.abs(leaf / biggest) ** 2) for leaf in leaves)
    return biggest * jnp.sqrt(total)


def _scaled(parts: Any, factor: jax.Array) -> Any:
    return jax.tree.map(lambda leaf: leaf / factor, parts)


def _random_like(template: Any, key: jax.Array) -> Any:
    leaves, treedef = jax.tree.flatten(template)
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(
        treedef,
        [
            jax.random.normal(subkey, leaf.shape, dtype=leaf.dtype)
            for leaf, subkey in zip(leaves, keys, strict=True)
        ],
    )


def largest_eigenvalue(
    operator: Callable[[Any], Any],
    template: Any,
    key: jax.Array,
    iterations: int,
) -> jax.Array:
    """Top eigenvalue of a symmetric positive-definite operator, by power iteration.

    Each step costs one application of ``operator`` -- for a normal operator
    that is the same JVP-plus-VJP a CG iteration costs, and no matrix is
    formed. The estimate approaches the true value from BELOW.

    Args:
        operator: the symmetric positive-definite map, pytree to pytree.
        template: a pytree of the operator's domain, for shapes and dtypes.
        key: PRNG key for the starting vector.
        iterations: number of steps.
    """
    vector = _random_like(template, key)
    largest = tree_norm(vector)
    vector = _scaled(vector, largest)
    for _ in range(iterations):
        image = operator(vector)
        largest = tree_norm(image)
        vector = _scaled(image, jnp.where(largest > 0, largest, 1.0))
    return largest
```

- [ ] **Step 5: 跑测试，确认通过**

```bash
.venv/bin/python -m pytest tests/exact/test_conditioning.py -v
```

Expected: 10 passed（两条参数化的各算两例）。

- [ ] **Step 6: 变异测试**

把 `tree_norm` 里的重标定去掉，改成朴素写法：

```python
    total = sum(jnp.sum(jnp.abs(leaf) ** 2) for leaf in leaves)
    return jnp.sqrt(total)
```

重跑。Expected: `test_tree_norm_survives_a_leaf_whose_square_overflows_float32` **变红**（返回 `inf`）。还原。

`largest_eigenvalue` 限制到首/末叶子，**两种形状都要跑**：

| 变异 | 必须在哪一例变红 |
|---|---|
| 只用首叶 | `top_in="b"` |
| 只用末叶 | `top_in="a"` |

**没有任何单一谱能同时抓住两种**——持有 λmax 的那个叶子按定义能独自复现它——所以必须参数化。还原。

再把 `largest_eigenvalue` 的循环体改成 `return tree_norm(operator(vector))`（只迭代一次，不论 `iterations`），重跑。
Expected: `test_largest_eigenvalue_finds_the_top_of_a_known_spectrum` **变红**。还原。

再让 `largest_eigenvalue` 超调：`return largest * 1.0001`。**这一行必须写在循环之外**——写在循环内（`largest = tree_norm(image) * 1.0001`）是个**无效变异**：归一化用的正是这个被放大的值，于是 `‖v‖` 缩小同样倍数、下一次 `‖Mv‖` 随之缩小，在收敛处**恰好抵消**，测试不会变红而实现其实没被破坏。实测确认过这一点。
Expected: `test_largest_eigenvalue_approaches_the_truth_from_below[spectrum0]` **变红**，`[spectrum1]` 保持绿（简并谱本就看不见 0.01% 的超调）。还原。

最后去掉 `iterations < 1` 守卫，重跑。
Expected: `test_largest_eigenvalue_refuses_fewer_than_one_iteration` **变红**。还原。

最后把 `tree_norm` 的重标定去掉、改成朴素的 `jnp.sqrt(sum(jnp.sum(leaf**2)))`，重跑。
Expected: 溢出与下溢两条测试**都变红**。还原。

- [ ] **Step 7: 提交**

```bash
git add src/bayesmith/exact tests/exact
git commit -m "feat: port spectral diagnostics for matrix-free operators"
```

---
## Task 2：高斯提取器与探针守卫（`exact/gaussian.py`）

精确解需要三个图不直接交出的数：预测与它的 σ（观测节点的分布），先验的中心与宽度（隐变量节点的分布）。`dist_fn` 返回的是不透明的 NumPyro 分布，所以这些数必须被**提取**——而按本包的纪律，提取要被**检验**。

**两个函数，刻意分开。** `gaussian_parts` 是快路径：一次 `isinstance`、两次属性读取，完全可追踪，每次求解都在 `jax.linearize` 里跑它。`check_gaussian` 是守卫：在若干点上探测节点自己的 `log_prob`，若提取出的 `(loc, scale)` 复现不了它就拒绝。守卫用 Python float 且会 `raise`，进不了 trace——它在建块时跑一次，跑在具体值上。

**分开不是优化，是正确性要求**：一个进不了快路径的守卫，必须在快路径**之前**、在快路径将看到的那些值上跑。

**Files:**
- Create: `src/bayesmith/exact/gaussian.py`
- Create: `tests/exact/models.py`, `tests/exact/test_gaussian.py`

- [ ] **Step 1: 建共享玩具图**

```python
# tests/exact/models.py
"""Toy graphs the exact-solve tests share.

Kept in one place so a change to a toy model cannot make two test modules
quietly disagree about what they are testing.

**Numbers are chosen to be pairwise distinct.** In every model below the true
value, the noise width, the prior width and the prior centre are all
different numbers. P1's Task 4 lost its only guard on this package's most
important guarantee to a fixture where `w = 3.0` and the true gradient
`sum(X) = 3.0` happened to agree, so a broken implementation returning the
parameter instead of its gradient passed anyway.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from bayesmith import const, det, observe, plate, sample, trace


class LyingNormal(dist.Normal):
    """A Normal whose ``log_prob`` is not the one its own loc/scale imply.

    Exists to make :func:`~bayesmith.exact.gaussian.check_gaussian`'s probe
    testable: introspection reads ``(loc, scale)`` off the type and is
    perfectly happy, and only evaluating ``log_prob`` reveals the
    disagreement. Not a contrived shape -- any ``Distribution`` subclass that
    overrides ``log_prob`` (a censored likelihood, a tempered one, a
    hand-written approximation) lands exactly here.
    """

    def log_prob(self, value):
        return 1.5 * super().log_prob(value)


def straight_line(*, n=8, weight=2.5, sigma=0.5, prior_std=2.0, prior_mean=0.0, seed=0):
    """``d ~ N(w X, sigma)``, ``w ~ N(prior_mean, prior_std)``.

    One linear latent, one observed node, no plate, sigma constant. The
    smallest graph the exact path applies to.
    """
    x = jnp.linspace(1.0, 4.0, n)
    data = weight * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(prior_mean, prior_std))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def two_linear_latents(*, n=12, slope=1.5, intercept=-3.0, sigma=0.4, seed=1):
    """``d ~ N(a X + b, sigma)``. Two latents that must be solved JOINTLY."""
    x = jnp.linspace(-2.0, 2.0, n)
    data = slope * x + intercept + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        a = sample("a", lambda: dist.Normal(0.0, 5.0))
        b = sample("b", lambda: dist.Normal(0.0, 7.0))
        mu = det("mu", lambda a_, b_, x_: a_ * x_ + b_, a, b, xs, linear_in=("a", "b"))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def bilinear_pair(*, n=10, sigma=0.3, seed=2):
    """``mu = gain * t_ant * X`` -- affine in EACH, not affine in the PAIR.

    The declaration ``linear_in=("gain", "t_ant")`` is therefore false, and
    `Graph.__check_init__` cannot see it: both names really are parents. Only
    the joint affinity probe can. This is rheplicant's motivating failure --
    a hand-rolled alternating solve here lands thousands of kelvin away while
    the CG residual reads 1e-7 and every per-block condition number reads ~1.5.
    """
    x = jnp.linspace(0.5, 3.0, n)
    data = 2.0 * 1.5 * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        g = sample("gain", lambda: dist.Normal(1.0, 1.0))
        t = sample("t_ant", lambda: dist.Normal(2.0, 3.0))
        mu = det(
            "mu", lambda g_, t_, x_: g_ * t_ * x_, g, t, xs, linear_in=("gain", "t_ant")
        )
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def quadratic_claim(*, n=6, sigma=0.5, seed=3):
    """``mu = w**2 X`` declared ``linear_in=("w",)`` -- a false single claim."""
    x = jnp.linspace(1.0, 2.0, n)
    data = 4.0 * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 2.0))
        mu = det("mu", lambda w_, x_: w_**2 * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def two_observations(*, n=7, m=5, weight=1.25, s1=0.3, s2=0.9, seed=4):
    """One latent constrained by TWO observed nodes -- the pytree codomain.

    rheplicant's codomain is a single array because a pipeline has one
    output. A graph can have several observed nodes, so every reduction in
    the solve runs over a dict of leaves rather than one array.
    """
    k1, k2 = jax.random.split(jax.random.key(seed))
    x1 = jnp.linspace(1.0, 3.0, n)
    x2 = jnp.linspace(-1.0, 1.0, m)
    d1 = weight * x1 + s1 * jax.random.normal(k1, (n,))
    d2 = 2.0 * weight * x2 + s2 * jax.random.normal(k2, (m,))

    def model():
        a = const("X1", x1)
        b = const("X2", x2)
        w = sample("w", lambda: dist.Normal(0.0, 4.0))
        m1 = det("mu1", lambda w_, x_: w_ * x_, w, a, linear_in=("w",))
        m2 = det("mu2", lambda w_, x_: 2.0 * w_ * x_, w, b, linear_in=("w",))
        observe("d1", lambda u: dist.Normal(u, s1), m1, obs=d1)
        observe("d2", lambda u: dist.Normal(u, s2), m2, obs=d2)

    return trace(model)


def plated_latent(*, n=6, sigma=0.4, tau=1.5, seed=5):
    """``z_i ~ N(0, tau)``, ``d_i ~ N(z_i, sigma)`` under one plate.

    The observed node's loc is the latent ITSELF, with no deterministic node
    on the path -- so the "every Deterministic on the path declares
    linear_in" rule holds vacuously, which the exact path must accept rather
    than trip over.
    """
    key = jax.random.key(seed)
    truth = tau * jax.random.normal(key, (n,))
    data = truth + sigma * jax.random.normal(jax.random.fold_in(key, 1), (n,))

    def model():
        obs = plate("obs", n)
        z = sample("z", lambda: dist.Normal(0.0, tau), plate=obs)
        observe("d", lambda z_: dist.Normal(z_, sigma), z, plate=obs, obs=data)

    return trace(model)


def radiometer(*, n=10, weight=3.0, kappa=0.05, floor=1e-3, seed=6):
    """``sigma_i = kappa |mu_i| + floor`` -- sigma tracks the prediction.

    The GLS / correction case. ``floor`` keeps sigma strictly positive where
    the prediction crosses zero, which the probe guard requires.
    """
    x = jnp.linspace(1.0, 5.0, n)
    truth = weight * x
    data = truth + (kappa * jnp.abs(truth) + floor) * jax.random.normal(
        jax.random.key(seed), (n,)
    )

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 10.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe(
            "d",
            lambda m: dist.Normal(m, kappa * jnp.abs(m) + floor),
            mu,
            depends_on_prediction=True,
            obs=data,
        )

    return trace(model)


def shared_ancestor(*, n=6, sigma=0.5, seed=7):
    """``tau`` is a latent AND an ancestor of the latent ``x``.

    Both nodes are Gaussian, so a naive classifier would put them in one
    block -- and the pair's joint distribution is not Gaussian, because x's
    own width is a function of tau. The block builder must refuse the pair.
    """
    x_grid = jnp.linspace(1.0, 2.0, n)
    data = 1.0 * x_grid + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x_grid)
        tau = sample("tau", lambda: dist.Normal(2.0, 0.5))
        x = sample("x", lambda t: dist.Normal(0.0, jnp.abs(t)), tau)
        mu = det("mu", lambda x_, g_: x_ * g_, x, xs, linear_in=("x",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)
```

- [ ] **Step 2: 写失败的测试**

```python
# tests/exact/test_gaussian.py
"""The (loc, scale) extractor, and the log_prob probe that checks it."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro import handlers

from bayesmith import const, det, evaluate, observe, sample, to_numpyro, trace
from bayesmith.errors import NotGaussian, StructureError
from bayesmith.exact.gaussian import (
    check_gaussian,
    gaussian_parts,
    node_shape,
    noise_std_at,
    observation_parts,
)
from tests.exact.models import (
    LyingNormal,
    plated_latent,
    radiometer,
    straight_line,
    two_observations,
)

WEIGHT = 2.5
SIGMA = 0.5


def test_gaussian_parts_reads_loc_and_scale_off_a_plain_normal():
    graph = straight_line(weight=WEIGHT, sigma=SIGMA, prior_std=2.0)
    env = evaluate(graph, {"w": jnp.asarray(WEIGHT)})
    loc, scale = gaussian_parts(graph, graph.node("d"), env)
    assert jnp.allclose(loc, WEIGHT * graph.node("X").value)
    assert scale.shape == loc.shape
    assert jnp.allclose(scale, SIGMA)


def test_gaussian_parts_unwraps_a_to_event_wrapper():
    """`.to_event(1)` only changes how log_prob is reduced, not the density."""

    def model():
        xs = const("X", jnp.ones(4))
        w = sample("w", lambda: dist.Normal(0.0, 1.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, 0.25).to_event(1), mu, obs=jnp.zeros(4))

    graph = trace(model)
    loc, scale = gaussian_parts(graph, graph.node("d"), evaluate(graph, {"w": 1.0}))
    assert loc.shape == (4,)
    assert jnp.allclose(scale, 0.25)


def test_gaussian_parts_refuses_a_node_that_is_not_gaussian():
    def model():
        sample("k", lambda: dist.Gamma(2.0, 3.0))
        observe("d", lambda: dist.Normal(0.0, 1.0), obs=jnp.zeros(()))

    graph = trace(model)
    env = evaluate(graph, {"k": jnp.asarray(1.0)})
    with pytest.raises(NotGaussian, match="Gamma"):
        gaussian_parts(graph, graph.node("k"), env)


def test_check_gaussian_accepts_a_real_normal():
    graph = straight_line()
    env = evaluate(graph, {"w": jnp.asarray(WEIGHT)})
    errors = check_gaussian(graph, graph.node("d"), env)
    assert set(errors) and all(err < 1e-4 for err in errors.values())


def test_check_gaussian_catches_a_distribution_that_lies_about_its_log_prob():
    """The whole reason the probe exists.

    Introspection passes here -- LyingNormal IS a Normal, its `.loc` and
    `.scale` are exactly what the model meant -- and the density is still
    wrong. Delete the probe and this test is the one that goes red.
    """

    def model():
        w = sample("w", lambda: dist.Normal(0.0, 1.0))
        observe("d", lambda w_: LyingNormal(w_, 0.7), w, obs=jnp.zeros(3))

    graph = trace(model)
    env = evaluate(graph, {"w": jnp.asarray(0.3)})
    loc, scale = gaussian_parts(graph, graph.node("d"), env)  # introspection is happy
    assert jnp.allclose(loc, 0.3) and jnp.allclose(scale, 0.7)
    with pytest.raises(StructureError, match="log_prob"):
        check_gaussian(graph, graph.node("d"), env)


def test_check_gaussian_refuses_a_scale_that_is_not_strictly_positive():
    def model():
        w = sample("w", lambda: dist.Normal(0.0, 1.0))
        observe("d", lambda w_: dist.Normal(w_, 0.0), w, obs=jnp.zeros(()))

    graph = trace(model)
    with pytest.raises(StructureError, match="scale"):
        check_gaussian(graph, graph.node("d"), evaluate(graph, {"w": jnp.asarray(0.0)}))


def test_node_shape_agrees_with_the_numpyro_bridge():
    """An independent reading of the same question.

    The bridge builds the site through numpyro.plate; node_shape derives the
    shape from the distribution and the declared plate size. They must agree,
    or the block's domain is a different space from the one NUTS samples.
    """
    graph = plated_latent(n=6)
    traced = handlers.trace(
        handlers.seed(to_numpyro(graph), jax.random.key(0))
    ).get_trace()
    env = evaluate(graph, {"z": traced["z"]["value"]})
    assert node_shape(graph, graph.node("z"), env) == traced["z"]["value"].shape == (6,)
    assert node_shape(graph, graph.node("d"), env) == traced["d"]["value"].shape == (6,)


def test_observation_parts_covers_every_observed_node():
    graph = two_observations(n=7, m=5)
    env = evaluate(graph, {"w": jnp.asarray(1.25)})
    data, loc, scale = observation_parts(graph, env)
    assert set(data) == set(loc) == set(scale) == {"d1", "d2"}
    assert data["d1"].shape == loc["d1"].shape == scale["d1"].shape == (7,)
    assert data["d2"].shape == loc["d2"].shape == scale["d2"].shape == (5,)
    assert jnp.allclose(scale["d1"], 0.3) and jnp.allclose(scale["d2"], 0.9)


def test_noise_std_at_moves_with_the_latent_only_for_a_prediction_dependent_node():
    """The seam that decides Wiener vs GLS, exercised on both sides."""
    constant = straight_line()
    a = noise_std_at(constant, {"w": jnp.asarray(1.0)})["d"]
    b = noise_std_at(constant, {"w": jnp.asarray(9.0)})["d"]
    assert jnp.allclose(a, b)

    tracking = radiometer()
    c = noise_std_at(tracking, {"w": jnp.asarray(1.0)})["d"]
    d = noise_std_at(tracking, {"w": jnp.asarray(9.0)})["d"]
    assert not jnp.allclose(c, d)
    assert jnp.all(d > c)
```

- [ ] **Step 3: 跑测试，确认失败**

```bash
.venv/bin/python -m pytest tests/exact/test_gaussian.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'bayesmith.exact.gaussian'`。

- [ ] **Step 4: 实现**

```python
# src/bayesmith/exact/gaussian.py
"""Reading a Gaussian off a node -- and checking the reading.

The exact solves need three numbers a graph does not hand over directly: the
prediction and its sigma (from an observed node's distribution), and the
prior's centre and width (from a latent node's). ``dist_fn`` returns an
opaque NumPyro distribution, so those numbers have to be *extracted* -- and,
this package's rule being that a claim is checked rather than trusted, the
extraction is checked too.

**Two functions, deliberately.** :func:`gaussian_parts` is the fast path: one
``isinstance``, two attribute reads, fully traceable, and it is what runs
inside ``jax.linearize`` on every solve. :func:`check_gaussian` is the guard:
it probes the node's own ``log_prob`` at several points and refuses if the
extracted ``(loc, scale)`` do not reproduce it. The guard uses Python floats
and raises, so it cannot run inside a trace -- it runs once, on concrete
values, when the block is built.

Splitting them is not an optimisation, it is a correctness requirement: a
guard that cannot run where the fast path runs must run *before* it, on the
values the fast path will see.

**Why the introspection is a fast path and not the answer.** Reading
``.loc``/``.scale`` off a ``Normal`` trusts the type. A ``Distribution``
subclass may override ``log_prob`` -- censored, tempered, or simply wrong --
and keep both attributes, so the type is evidence and not proof. The probe is
what turns it into proof, at a cost of four ``log_prob`` evaluations per node
per block build.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from bayesmith.errors import NotGaussian, StructureError
from bayesmith.graph.evaluate import apply_probabilistic, evaluate
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Node, Probabilistic

#: Probe offsets for :func:`check_gaussian`, in units of the node's own scale.
#: Spread over the bulk and into both tails: a wrong ``loc`` shifts every
#: probe, a wrong ``scale`` produces a mismatch that grows with ``|offset|``,
#: and a log-density that is not quadratic fails at the outer pair first.
#: Asymmetric on purpose -- a symmetric set cannot distinguish a sign error in
#: ``loc`` from a correct one.
PROBE_OFFSETS: tuple[float, ...] = (-3.0, -1.0, 0.5, 2.0)

_LOG_2PI = float(np.log(2.0 * np.pi))


def unwrap(distribution: Any) -> Any:
    """Strip wrappers that only change how ``log_prob`` is reduced.

    ``Independent`` is what ``.to_event(k)`` produces: the same per-element
    density with the last ``k`` batch dimensions summed. The elementwise
    reading underneath is what the exact solves want -- a diagonal covariance
    is diagonal whether or not its log-density was summed -- so the wrapper is
    removed rather than refused.
    """
    while isinstance(distribution, dist.Independent):
        distribution = distribution.base_dist
    return distribution


def gaussian_parts(
    graph: Graph, node: Node, env: dict[str, Any]
) -> tuple[jax.Array, jax.Array]:
    """``(loc, scale)`` of ``node``'s distribution, ``scale`` broadcast to ``loc``.

    Traceable and **unchecked** -- pair it with :func:`check_gaussian`, which
    runs once on concrete values before any trace is opened.

    Raises:
        NotGaussian: if the distribution is not a (possibly ``Independent``-
            wrapped) ``Normal``. The *type* is static under tracing, so this
            refusal is safe to make inside a trace. It is a **classification
            outcome, not a fault**: P3b's dispatcher catches it and routes the
            block to NUTS.
        StructureError: if ``loc`` has an integer dtype. That one IS a fault --
            a conjugate solve differentiates through ``loc``.
    """
    distribution = unwrap(apply_probabilistic(graph, node, env))
    if not isinstance(distribution, dist.Normal):
        raise NotGaussian(
            f"node {node.name!r} returns {type(distribution).__name__}; the exact "
            "linear-Gaussian path needs a diagonal Normal (a Normal, or one "
            "wrapped by .to_event(...)). A MultivariateNormal with a dense "
            "covariance is a different solve and is not implemented. This is a "
            "classification outcome, not a defect in the model."
        )
    loc = jnp.asarray(distribution.loc)
    if not jnp.issubdtype(loc.dtype, jnp.inexact):
        raise StructureError(
            f"node {node.name!r} has an integer loc (dtype {loc.dtype}). A "
            "conjugate solve differentiates through loc, so it has to be a "
            "floating dtype -- pass a float to the distribution."
        )
    scale = jnp.broadcast_to(jnp.asarray(distribution.scale), jnp.shape(loc))
    return loc, scale


def node_shape(graph: Graph, node: Node, env: dict[str, Any]) -> tuple[int, ...]:
    """Shape of ``node``'s VALUE.

    Three sources, broadcast together: the distribution's own batch shape, the
    plate the node sits in, and -- for an observed node -- the data. All three
    are needed. A plated latent whose ``dist_fn`` takes no plated parent has a
    scalar ``loc`` and a plate-shaped value (the ordinary "N iid draws from one
    shared prior"); an unplated observed node conditioned on a vector has a
    scalar ``loc`` and a vector value. Taking any one source alone gets one of
    those two wrong.

    This must agree with what ``to_numpyro`` opens the site at, or the block's
    domain is a different space from the one NUTS samples --
    ``test_node_shape_agrees_with_the_numpyro_bridge`` pins it.
    """
    loc, _ = gaussian_parts(graph, node, env)
    shapes: list[tuple[int, ...]] = [jnp.shape(loc)]
    if node.plate:
        shapes.append((graph.plate_size(node.plate[0]),))
    if isinstance(node, Probabilistic) and node.observed is not None:
        shapes.append(jnp.shape(node.observed))
    return tuple(jnp.broadcast_shapes(*shapes))


def check_gaussian(
    graph: Graph, node: Node, env: dict[str, Any], *, rtol: float | None = None
) -> dict[float, float]:
    """Verify the extracted ``(loc, scale)`` really reproduce ``node``'s log_prob.

    Costs ``len(PROBE_OFFSETS)`` evaluations of ``log_prob``. Runs on concrete
    values, **outside** any trace.

    Args:
        graph, node, env: the node under test and the values its parents take.
        rtol: tolerance on the relative disagreement. Default ``1e3 * eps`` of
            ``loc``'s dtype, which leaves room for accumulated roundoff in the
            reduction without admitting a real difference in density.

    Returns:
        ``{offset: relative error}`` -- useful for reporting how Gaussian a
        node is, not only whether it passes.

    Raises:
        NotGaussian: propagated from :func:`gaussian_parts`.
        StructureError: if the scale is not strictly positive and finite, or
            if any probe disagrees by more than ``rtol``.
    """
    distribution = unwrap(apply_probabilistic(graph, node, env))
    loc, scale = gaussian_parts(graph, node, env)
    if not bool(jnp.all(jnp.isfinite(scale) & (scale > 0))):
        raise StructureError(
            f"node {node.name!r} has a scale that is not strictly positive and "
            f"finite (min {float(jnp.min(scale)):g}). A conjugate solve weights "
            "by 1/scale**2, so a zero or negative sigma is an infinite or "
            "negative weight rather than a tight constraint. Add a floor to the "
            "expression that produces it."
        )
    if rtol is None:
        rtol = 1e3 * float(jnp.finfo(loc.dtype).eps)

    errors: dict[float, float] = {}
    for offset in PROBE_OFFSETS:
        probe = loc + offset * scale
        actual = float(jnp.sum(distribution.log_prob(probe)))
        predicted = float(
            jnp.sum(
                -0.5 * ((probe - loc) / scale) ** 2 - jnp.log(scale) - 0.5 * _LOG_2PI
            )
        )
        # Relative to the predicted magnitude, with a floor so a probe that
        # lands where the log-density happens to be ~0 does not divide by it.
        errors[offset] = abs(actual - predicted) / max(abs(predicted), 1.0)
        # NaN must count as a FAILURE: `nan > rtol` is False, so a naive
        # comparison treats an unusable probe as evidence of Gaussianity.
        if not np.isfinite(errors[offset]) or errors[offset] > rtol:
            detail = ", ".join(f"{k:+g}sigma -> {v:.3e}" for k, v in errors.items())
            raise StructureError(
                f"node {node.name!r} is a {type(distribution).__name__}, so its "
                "loc and scale were read off it directly -- but its own log_prob "
                f"does not agree with them (rtol={rtol:.2e}; {detail}). A "
                "Distribution subclass that overrides log_prob keeps both "
                "attributes and changes the density, so the type is evidence "
                "and not proof. The exact path would solve the wrong posterior; "
                "it refuses instead."
            )
    return errors


def observation_parts(
    graph: Graph, env: dict[str, Any]
) -> tuple[dict[str, jax.Array], dict[str, jax.Array], dict[str, jax.Array]]:
    """``({obs: data}, {obs: loc}, {obs: scale})`` over every observed node.

    All three are broadcast to :func:`node_shape`, so the three dicts align
    leaf for leaf and every reduction downstream is one ``jax.tree.map``.
    """
    data: dict[str, jax.Array] = {}
    loc: dict[str, jax.Array] = {}
    scale: dict[str, jax.Array] = {}
    for name in graph.observed:
        node = graph.node(name)
        shape = node_shape(graph, node, env)
        node_loc, node_scale = gaussian_parts(graph, node, env)
        data[name] = jnp.broadcast_to(node.observed, shape)
        loc[name] = jnp.broadcast_to(node_loc, shape)
        scale[name] = jnp.broadcast_to(node_scale, shape)
    return data, loc, scale


def noise_std_at(graph: Graph, values: dict[str, Any]) -> dict[str, jax.Array]:
    """``{obs: scale}`` with the latents at ``values`` -- the GLS seam.

    ``iterative_gls`` iterates this: solve at the current sigma, recompute
    sigma at the new solution, repeat. Whether it moves at all is what
    separates a single Wiener solve from a reweighting loop, and
    ``test_noise_std_at_moves_with_the_latent_only_for_a_prediction_dependent_node``
    exercises both sides.
    """
    _, _, scale = observation_parts(graph, evaluate(graph, values))
    return scale
```

- [ ] **Step 5: 跑测试，确认通过**

```bash
.venv/bin/python -m pytest tests/exact/test_gaussian.py -v
```

Expected: 9 passed。

- [ ] **Step 6: 变异测试**

1. 删掉 `check_gaussian` 里的整个 `for offset in PROBE_OFFSETS:` 循环（直接 `return {}`），重跑。
   Expected: `test_check_gaussian_catches_a_distribution_that_lies_about_its_log_prob` **变红**。还原。
2. 把 `node_shape` 改成只返回 `tuple(jnp.shape(loc))`，重跑。
   Expected: `test_node_shape_agrees_with_the_numpyro_bridge` **变红**（得到 `()` 而非 `(6,)`）。还原。
3. 把 `gaussian_parts` 的 `NotGaussian` 改成 `StructureError`，重跑。
   Expected: `test_gaussian_parts_refuses_a_node_that_is_not_gaussian` **变红**。还原。

- [ ] **Step 7: 提交**

```bash
git add src/bayesmith/exact/gaussian.py tests/exact/models.py tests/exact/test_gaussian.py
git commit -m "feat: extract and verify a node's Gaussian (loc, scale)"
```

---
## Task 3：线性块（`exact/block.py`）

把图里的一组隐变量导出成 `A x + offset`，其中 `A` 来自 `jax.linearize`、`Aᵀ` 来自 `jax.vjp`，**从不形成矩阵**。10⁶ 维的块因此与一次前向求值同价。

**定义域与陪域都是 dict，永远是。** rheplicant 有 `name=` / `names=` 双拼写和一个 `as_dict` 去调和它们；图让这件事变得不必要——隐变量有名字，观测节点有名字，两者各只有一种写法。一种拼写意味着没有 `as_dict`、没有 `grouped` 属性、也没有办法把一个裸数组交给按名索引的东西。

**先有鸡先有蛋，以及它逼出的规则。** 建块要知道每个成员的形状，形状来自它先验的 `loc`，而 `evaluate()` 要求所有隐变量先有值。解法是把 spec §2.1 的第 2 条**收紧为传递的**：**没有块成员可以是另一个块成员的祖先**。直接父检查看不见 `m1 → 某确定性节点 → m2 的先验` 这条路径，而联合分布同样不是高斯。规则一旦成立，一次拓扑扫描就能在读到任何成员的值**之前**算出它的形状。

**两个名字，不是一个开关。** rheplicant 用 `linear_operator(..., check=True)`，即最自然的调用名默认安全、而放弃检验只要改一个关键字。这里改成两个函数：本任务交付的 `unchecked_operator` 是**原语**；Task 4 交付的 `linear_operator` 先检验再建块，是**文档化的入口**。最自然的名字因此是安全的那个，而放弃检验必须写出一个自己说明了放弃什么的名字。这也是 rheplicant `plan.py` 对 `estimate()` / `sample()` 讲过的同一论证：*"Two methods, not a mode flag ... makes the invalid combinations unrepresentable rather than validated."*

**Files:**
- Create: `src/bayesmith/exact/block.py`
- Create: `tests/exact/test_block.py`

- [ ] **Step 1: 写失败的测试**

```python
# tests/exact/test_block.py
"""Exporting A, A^T and the offset from a graph -- and what the block refuses."""

import jax
import jax.numpy as jnp
import pytest

from bayesmith import evaluate
from bayesmith.errors import GraphError, NotGaussian
from bayesmith.exact.block import (
    _env_before,
    domain_centre,
    domain_zero,
    largest_variance,
    unchecked_operator,
    variance_parts,
)
from tests.exact.models import (
    plated_latent,
    shared_ancestor,
    straight_line,
    two_linear_latents,
    two_observations,
)


def test_offset_is_the_prediction_with_the_block_at_zero():
    """`b` sits outside the block, so it is what the offset carries.

    Held at 4.0, which is none of the model's own numbers (slope 1.5,
    intercept -3.0, sigma 0.4) -- a block that returned the wrong quantity
    could not land on 4.0 by coincidence.
    """
    graph = two_linear_latents()
    block = unchecked_operator(graph, ("a",), at={"b": jnp.asarray(4.0)})
    assert jnp.allclose(block.offset["d"], 4.0)


def test_forward_is_the_linear_action_of_the_block():
    graph = two_linear_latents(n=12)
    block = unchecked_operator(graph, ("a",), at={"b": jnp.asarray(4.0)})
    got = block.forward({"a": jnp.asarray(1.0)})["d"]
    assert jnp.allclose(got, jnp.linspace(-2.0, 2.0, 12))


def test_adjoint_is_the_transpose_under_the_real_inner_product():
    """`sum(x * adjoint(y)) == sum(forward(x) * y)`, across both observed nodes.

    An adjoint that dropped one observed node, or scaled one, breaks this
    identity; nothing else in the block would notice.
    """
    graph = two_observations()
    block = unchecked_operator(graph, ("w",), at={})
    x = {"w": jnp.asarray(0.37)}
    y = {
        "d1": jax.random.normal(jax.random.key(11), block.offset["d1"].shape),
        "d2": jax.random.normal(jax.random.key(12), block.offset["d2"].shape),
    }
    pulled = block.adjoint(y)
    pushed = block.forward(x)
    lhs = sum(float(jnp.sum(x[n] * pulled[n])) for n in block.names)
    rhs = sum(float(jnp.sum(pushed[o] * y[o])) for o in y)
    assert lhs == pytest.approx(rhs, rel=1e-5)


def test_the_block_spans_every_observed_node():
    graph = two_observations(n=7, m=5)
    block = unchecked_operator(graph, ("w",), at={})
    assert set(block.offset) == set(block.data) == {"d1", "d2"}
    assert block.offset["d1"].shape == (7,)
    assert block.offset["d2"].shape == (5,)


def test_the_prior_is_read_off_the_graph():
    """No prior_std keyword exists, so the graph cannot be contradicted."""
    graph = two_linear_latents()
    block = unchecked_operator(graph, ("a", "b"), at={})
    assert jnp.allclose(block.prior_std["a"], 5.0)
    assert jnp.allclose(block.prior_std["b"], 7.0)
    assert jnp.allclose(block.prior_mean["a"], 0.0)
    assert jnp.allclose(block.prior_mean["b"], 0.0)


def test_env_before_agrees_with_evaluate_on_every_node():
    """Pins the six lines `_env_before` duplicates from `evaluate`.

    `_env_before` cannot call `evaluate` (it runs before the block has any
    value to give it), so it repeats the isinstance ladder. That is exactly
    the kind of duplication P1 recorded as the start of a silent drift, so
    the two are compared node by node here.
    """
    graph = two_linear_latents()
    at = {"b": jnp.asarray(4.0)}
    env, domain = _env_before(graph, ("a",), at)
    full = evaluate(graph, {**at, "a": domain["a"][2]})
    assert set(env) == set(full) == set(graph.names)
    for name in graph.names:
        assert jnp.allclose(env[name], full[name]), name


def test_a_block_holding_a_latent_and_its_own_ancestor_is_refused():
    """`x`'s width IS `tau`, so the pair is not jointly Gaussian.

    Both nodes are individually Normal, so a classifier that checked only
    "is each node Gaussian" would put them in one block and solve a
    posterior nobody declared.
    """
    graph = shared_ancestor()
    with pytest.raises(NotGaussian, match="ancestor"):
        unchecked_operator(graph, ("tau", "x"), at={})


def test_each_of_the_two_is_a_legitimate_block_on_its_own():
    """The refusal above is about the PAIR, not about either member."""
    graph = shared_ancestor()
    block = unchecked_operator(graph, ("x",), at={"tau": jnp.asarray(2.0)})
    assert jnp.allclose(block.prior_std["x"], 2.0)


def test_a_plated_latent_block_carries_the_plate_shaped_domain():
    graph = plated_latent(n=6)
    block = unchecked_operator(graph, ("z",), at={})
    assert block.shape["z"] == (6,)
    assert domain_zero(block)["z"].shape == (6,)
    assert block.prior_std["z"].shape == (6,)


def test_naming_something_that_is_not_a_latent_is_refused():
    graph = straight_line()
    with pytest.raises(GraphError, match="mu"):
        unchecked_operator(graph, ("mu",), at={})
    with pytest.raises(GraphError, match="twice|repeat"):
        unchecked_operator(graph, ("w", "w"), at={})
    with pytest.raises(GraphError, match="at least one"):
        unchecked_operator(graph, (), at={})


def test_domain_centre_is_the_declared_prior_mean():
    graph = straight_line(prior_mean=1.75, prior_std=2.0)
    block = unchecked_operator(graph, ("w",), at={})
    assert jnp.allclose(domain_centre(block)["w"], 1.75)


def test_variance_parts_places_each_prior_on_its_own_leaf():
    graph = two_linear_latents()
    block = unchecked_operator(graph, ("a", "b"), at={})
    parts = variance_parts(block)
    assert jnp.allclose(parts["a"], 25.0)
    assert jnp.allclose(parts["b"], 49.0)


def test_largest_variance_takes_the_loosest_prior_not_the_tightest():
    """1/largest floors lambda_min of the normal operator.

    Taking the tightest instead would floor the estimate ABOVE the true
    lambda_min and report a condition number smaller than the real one --
    an over-confident guard, which is the direction that costs something.
    """
    graph = two_linear_latents()
    block = unchecked_operator(graph, ("a", "b"), at={})
    assert float(largest_variance(variance_parts(block))) == pytest.approx(49.0)
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
.venv/bin/python -m pytest tests/exact/test_block.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'bayesmith.exact.block'`。

- [ ] **Step 3: 实现**

```python
# src/bayesmith/exact/block.py
"""The affine action of a group of latents on every prediction.

No matrix is ever formed: ``A`` comes from ``jax.linearize`` and ``A^T`` from
``jax.vjp``, so a block with 10^6 degrees of freedom costs one forward
evaluation per application. That is what makes the conjugate-Gaussian solves
in :mod:`bayesmith.exact.solve` tractable at all.

**Where everything comes from.** rheplicant's ``unchecked_operator`` takes a
``ParameterSpace`` for the latents and their priors and a ``Pipeline`` for the
prediction. Here both come from the graph: the prediction is every observed
node's ``loc``, the prior is each member's own distribution, and sigma is
every observed node's ``scale``. There is consequently **no keyword to
override any of them** -- and so none of rheplicant's reconciliation
machinery (``_reconcile``, ``_agrees``, ``_resolve_prior``,
``_require_prior_std``) has anything to reconcile. One statement of the prior
means the exact solve and NUTS cannot target different posteriors.

**One spelling.** The domain is ``{latent: array}`` and the codomain is
``{observed: array}``, always, for one member or several. rheplicant carries
a ``name=``/``names=`` dual spelling plus ``as_dict`` to paper over it; the
graph makes that unnecessary because everything already has a name.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from typing import Any

import jax
import jax.numpy as jnp

from bayesmith.errors import GraphError, NotGaussian
from bayesmith.exact.gaussian import (
    check_gaussian,
    gaussian_parts,
    node_shape,
    observation_parts,
)
from bayesmith.graph.evaluate import apply_deterministic, evaluate
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Const, Deterministic, Probabilistic


@dataclasses.dataclass(frozen=True)
class LinearBlock:
    """``A x + offset``: what a group of latents does to every prediction.

    Deliberately a plain frozen dataclass rather than an ``eqx.Module``: this
    is a derived linear-algebra handle, not a differentiable model.
    ``forward`` and ``adjoint`` close over a traced computation, so a block is
    something you build where you need it, not a pytree to carry around.

    Attributes:
        names: the block's members, in the caller's order.
        shape, dtype: ``{name: ...}`` describing the domain.
        offset: ``{observed: prediction at x = 0}`` -- everything OUTSIDE the
            block contributes.
        forward: ``{name: x} -> {observed: A x}``, from ``jax.linearize``.
        adjoint: ``{observed: y} -> {name: A^T y}``, from ``jax.vjp``.
        data: ``{observed: value}``, broadcast to the prediction's shape.
        prior_mean, prior_std: ``{name: ...}``, read off each member's own
            distribution at ``at``.

    Adjoint convention: ``adjoint`` is exactly ``jax.vjp``, so the identity
    that holds is over the **real** inner product::

        sum(x * adjoint(y))  ==  sum(forward(x) * y)

    which is the pairing a Gaussian likelihood forms.
    ``test_adjoint_is_the_transpose_under_the_real_inner_product`` pins both
    halves so the distinction cannot rot into a silent factor.
    """

    names: tuple[str, ...]
    shape: dict[str, tuple[int, ...]]
    dtype: dict[str, Any]
    offset: dict[str, jax.Array]
    forward: Callable[[Any], dict[str, jax.Array]]
    adjoint: Callable[[Any], dict[str, jax.Array]]
    data: dict[str, jax.Array]
    prior_mean: dict[str, jax.Array]
    prior_std: dict[str, jax.Array]


def domain_zero(block: LinearBlock) -> dict[str, jax.Array]:
    """A zero of the block's domain."""
    return {n: jnp.zeros(block.shape[n], dtype=block.dtype[n]) for n in block.names}


def domain_centre(block: LinearBlock) -> dict[str, jax.Array]:
    """The prior's centre, laid out over the domain.

    A zero-mean prior is wrong for most physical quantities -- a noise-wave
    temperature sits near 250 K, not near zero -- and shifting the prior is
    not the same act as shifting the model even though the two give the same
    Gaussian. The graph states which one was meant, so this reads it.
    """
    return {n: block.prior_mean[n] for n in block.names}


def variance_parts(block: LinearBlock) -> dict[str, jax.Array]:
    """``S``'s diagonal, laid out over the domain.

    Block-diagonal assembly by **placement** rather than concatenation: each
    member's variance lands on the leaf its own parameters live on, so
    ``x / variance`` inside a ``jax.tree.map`` IS ``S^-1 x`` with no indices
    to get wrong.
    """
    return {n: jnp.asarray(block.prior_std[n]) ** 2 for n in block.names}


def largest_variance(prior_variance: dict[str, jax.Array]) -> jax.Array:
    """The biggest prior variance anywhere in the block.

    ``1/it`` floors ``lambda_min`` of the normal operator: ``A^T N^-1 A`` is
    positive semi-definite, so the LOOSEST prior in the block is what bounds
    the operator from below. Taking the tightest instead would floor the
    estimate above the true ``lambda_min`` and report a condition number
    smaller than the real one -- an over-confident guard, which is the
    direction that costs something.
    """
    leaves = jax.tree.leaves(prior_variance)
    return jnp.max(jnp.stack([jnp.max(jnp.asarray(leaf)) for leaf in leaves]))


def _ancestors(graph: Graph, name: str) -> set[str]:
    """Every node ``name`` transitively depends on."""
    seen: set[str] = set()
    stack = list(graph.node(name).parents)
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(graph.node(current).parents)
    return seen


def _validated_names(graph: Graph, names: Iterable[str]) -> tuple[str, ...]:
    names = tuple(names)
    if not names:
        raise GraphError(
            "a linear block needs at least one latent name. An empty block is "
            "solved every sweep and changes nothing, so a plan holding one runs, "
            "converges, and reports a partition that does not cover the space it "
            "claims to."
        )
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise GraphError(
            f"block {names} lists {repeated} twice. Two copies of one latent are "
            "exactly degenerate with each other, so the normal operator is "
            "singular in a direction that says nothing about the model."
        )
    latents = set(graph.latents)
    stray = [n for n in names if n not in latents]
    if stray:
        raise GraphError(
            f"block {names} names {stray}, which are not latent nodes of this "
            f"graph. Latents are {list(graph.latents)}. A deterministic or "
            "observed node has no posterior to solve for."
        )
    return names


def _refuse_internal_ancestry(graph: Graph, names: tuple[str, ...]) -> None:
    """Refuse a block one of whose members is an ancestor of another.

    ``x ~ N(0, |tau|)`` with ``tau`` a latent is perfectly good modelling, and
    both nodes are individually Normal -- but the PAIR is not jointly
    Gaussian, because one member's width is a function of the other. Solving
    them together would produce a finite, confident posterior for a model
    nobody declared.

    The test is **transitive**, not on direct parents: ``m1 -> some
    deterministic node -> m2``'s prior is the same situation with one more
    hop, and a direct-parent check cannot see it.

    It is also what makes :func:`_env_before` correct -- with no member
    ancestral to another, a topological scan reaches every member's prior
    before it needs any member's value.
    """
    members = set(names)
    for member in names:
        clash = sorted(members.intersection(_ancestors(graph, member)))
        if clash:
            raise NotGaussian(
                f"latent {member!r} has {clash} among its ancestors, and they are "
                "in the same block. One member's distribution is then a function "
                "of another's value, so the pair is not JOINTLY Gaussian however "
                "Normal each one is on its own -- and a conjugate solve over the "
                "pair would return a confident posterior for a model nobody "
                "declared. Put them in separate blocks and alternate."
            )


def _env_before(
    graph: Graph, names: tuple[str, ...], at: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, tuple[tuple[int, ...], Any, jax.Array, jax.Array]]]:
    """Every node's value, with the block members replaced by their prior means.

    Also returns ``{member: (shape, dtype, prior_mean, prior_std)}`` -- the
    block's domain, derived DURING the scan from values already in hand rather
    than from any member's own value. That is what breaks the circularity:
    building the domain needs the shapes, the shapes come from each member's
    prior, and the prior needs an environment that does not yet contain the
    member.

    Correct only because :func:`_refuse_internal_ancestry` has run: otherwise
    a member's prior could depend on a member not yet reached, and the
    placeholder inserted here would silently become part of it.

    Repeats ``evaluate``'s isinstance ladder rather than calling it, for the
    reason above. ``test_env_before_agrees_with_evaluate_on_every_node`` pins
    the two together so the duplication cannot drift.
    """
    members = set(names)
    env: dict[str, Any] = {}
    domain: dict[str, tuple[tuple[int, ...], Any, jax.Array, jax.Array]] = {}
    for node in graph.nodes:
        if node.name in members:
            check_gaussian(graph, node, env)
            loc, scale = gaussian_parts(graph, node, env)
            shape = node_shape(graph, node, env)
            mean = jnp.broadcast_to(loc, shape)
            std = jnp.broadcast_to(scale, shape)
            domain[node.name] = (shape, loc.dtype, mean, std)
            env[node.name] = mean
        elif isinstance(node, Const):
            env[node.name] = node.value
        elif isinstance(node, Deterministic):
            env[node.name] = apply_deterministic(graph, node, env)
        elif isinstance(node, Probabilistic):
            if not node.is_latent:
                env[node.name] = node.observed
            elif node.name in at:
                env[node.name] = at[node.name]
            else:
                raise GraphError(
                    f"latent node {node.name!r} is outside the block and has no "
                    f"value in `at`. A block is affine GIVEN the latents outside "
                    "it, so they must be somewhere -- pass the current values."
                )
        else:  # pragma: no cover - defensive, mirrors evaluate()
            raise GraphError(f"unknown node type {type(node).__name__}")
    return env, domain


def isolate(
    graph: Graph, names: tuple[str, ...], at: dict[str, Any]
) -> Callable[[dict[str, Any]], dict[str, jax.Array]]:
    """``g: {name: x} -> {observed: loc}`` -- the block's action on the prediction.

    Built on ``evaluate``, so there is exactly one forward scan in this package
    and the block cannot diverge from what ``log_joint`` and the NumPyro bridge
    read.
    """

    def g(x: dict[str, Any]) -> dict[str, jax.Array]:
        _, loc, _ = observation_parts(graph, evaluate(graph, {**at, **x}))
        return loc

    return g


def unchecked_operator(
    graph: Graph, names: Iterable[str], at: dict[str, Any] | None = None
) -> LinearBlock:
    """Export ``A``, ``A^T``, the offset, the data and the prior -- **unchecked**.

    Args:
        graph: the model.
        names: the latents in the block. Solving a group JOINTLY is not the
            same as alternating over its members: two latents the data barely
            tells apart are resolved in one CG here, where alternation
            converges at the rate of their correlation while reporting a
            converged residual and a condition number of ~1 at every step.
        at: values for the latents OUTSIDE the block. A block is affine
            *given* them, so this fixes where it is built -- which is what
            makes a Gibbs sweep possible: rebuild here every sweep at the
            current values.

    Raises:
        GraphError: if ``names`` is empty, repeats a latent, names something
            that is not a latent, or if a latent outside the block has no
            value in ``at``.
        NotGaussian: if a member or an observed node is not a diagonal
            Gaussian, or if a member is an ancestor of another member.
        StructureError: if a node's own ``log_prob`` disagrees with the
            ``loc``/``scale`` read off it.

    Note:
        The name is the warning. This does **not** check the ``linear_in``
        declaration, so it will happily export an operator for a block that is
        not affine at all and hand back a confident, wrong posterior.
        :func:`bayesmith.exact.linearity.linear_operator` is the entry point
        that checks first, and is what callers should reach for. This one is
        for inside a Gibbs sweep, where the check has been hoisted out of the
        loop deliberately.
    """
    names = _validated_names(graph, names)
    at = dict(at or {})
    _refuse_internal_ancestry(graph, names)

    env, domain = _env_before(graph, names, at)
    for observed in graph.observed:
        check_gaussian(graph, graph.node(observed), env)

    g = isolate(graph, names, at)
    zero = {n: jnp.zeros(domain[n][0], dtype=domain[n][1]) for n in names}
    offset, tangent = jax.linearize(g, zero)
    _, pullback = jax.vjp(g, zero)
    data, _, _ = observation_parts(graph, evaluate(graph, {**at, **zero}))

    return LinearBlock(
        names=names,
        shape={n: domain[n][0] for n in names},
        dtype={n: domain[n][1] for n in names},
        offset=offset,
        forward=tangent,
        adjoint=lambda y: pullback(y)[0],
        data=data,
        prior_mean={n: domain[n][2] for n in names},
        prior_std={n: domain[n][3] for n in names},
    )
```

- [ ] **Step 4: 跑测试，确认通过**

```bash
.venv/bin/python -m pytest tests/exact/test_block.py -v
```

Expected: 13 passed。

- [ ] **Step 5: 变异测试**

1. 把 `_refuse_internal_ancestry` 的 `_ancestors` 换成只看直接父（`set(graph.node(member).parents)`），重跑。
   Expected: `test_a_block_holding_a_latent_and_its_own_ancestor_is_refused` **仍绿**（`shared_ancestor` 里 `tau` 确实是 `x` 的直接父）。这说明这条测试**没有覆盖传递情形**——补一个模型：在 `tests/exact/models.py` 里加

```python
def indirect_ancestor(*, n=6, sigma=0.5, seed=8):
    """`tau` reaches `x`'s prior through a deterministic node, not directly.

    A direct-parent ancestry check passes this and is wrong: x's parents are
    ("width",), and `width` is a function of `tau`.
    """
    x_grid = jnp.linspace(1.0, 2.0, n)
    data = 1.0 * x_grid + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x_grid)
        tau = sample("tau", lambda: dist.Normal(2.0, 0.5))
        width = det("width", lambda t: jnp.abs(t) + 0.1, tau)
        x = sample("x", lambda w: dist.Normal(0.0, w), width)
        mu = det("mu", lambda x_, g_: x_ * g_, x, xs, linear_in=("x",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)
```

   并在 `tests/exact/test_block.py` 里加

```python
def test_ancestry_is_transitive_not_just_direct_parents():
    """`tau` reaches `x` through `width`, so a direct-parent check misses it."""
    from tests.exact.models import indirect_ancestor

    graph = indirect_ancestor()
    assert graph.node("x").parents == ("width",)  # tau is NOT a direct parent
    with pytest.raises(NotGaussian, match="ancestor"):
        unchecked_operator(graph, ("tau", "x"), at={})
```

   现在重做这条变异：直接父版本下新测试 **变红**。还原 `_ancestors`，确认两条都绿。

2. 把 `unchecked_operator` 里的 `adjoint=lambda y: pullback(y)[0]` 改成只保留第一个观测节点（`lambda y: pullback({k: v if k == "d1" else jnp.zeros_like(v) for k, v in y.items()})[0]`），重跑。
   Expected: `test_adjoint_is_the_transpose_under_the_real_inner_product` **变红**。还原。

3. 把 `largest_variance` 的 `jnp.max` 改成 `jnp.min`，重跑。
   Expected: `test_largest_variance_takes_the_loosest_prior_not_the_tightest` **变红**（得到 25 而非 49）。还原。

- [ ] **Step 6: 提交**

```bash
git add src/bayesmith/exact/block.py tests/exact/test_block.py tests/exact/models.py
git commit -m "feat: export a matrix-free linear block from a graph (unchecked)"
```

---
## Task 4：线性性检验与文档化入口（`exact/linearity.py`）

`linear_in` 是**关于模型的断言**，不是提示——它决定能否走精确解，所以在被利用之前必须被检验。一个假的声明否则会产出一个自信的、错误的后验，而不是一个错误。

**相对 rheplicant 的两处改动：**

1. **探针幅度取自声明的先验标准差**，不是 `max|init|`。rheplicant 的文档明写"全零 init 没有尺度可取，退化为绝对探针"这个坑；bayesmith 没有 `init`，而先验标准差正是"这个隐变量有多大"的正确陈述。还是逐元素的，所以形状不均匀的隐变量按各自的宽度被探测。
2. **在多个 `at` 点上重复探测。** `at` 是块外隐变量的取值，而 Gibbs sweep 里块会在不同 `at` 处反复重建。只在一个点验过就在所有点使用，是 `boundary-validation.md` 明令禁止的 moderate-parameter probe——那条规则记录的原案例（canoes，2026-05-10）里，一个只在 `ell ∈ {10,100,500}` 上探测得出的"修复"在 `ell=5000` 处炸了 63 个数量级。默认的额外 `at` 点从图自己的先验抽（走 P2 的桥），因此覆盖模型自己认为可能的取值范围。

**Files:**
- Create: `src/bayesmith/exact/linearity.py`
- Modify: `src/bayesmith/exact/block.py`（加一条"图必须有观测节点"的拒绝）
- Modify: `tests/exact/models.py`
- Create: `tests/exact/test_linearity.py`

- [ ] **Step 1: 加三个玩具模型**

追加到 `tests/exact/models.py`：

```python
def cubic_tail(*, n=6, curvature=1e-6, prior_std=1.0, sigma=0.5, seed=9):
    """``mu = (w + curvature w**3) X`` -- affine only for small ``|w|``.

    ``linear_in=("w",)`` is false, but detectably so only at probes large
    enough for the cubic term to matter -- and what sets that scale is the
    declared prior width. The SAME fn therefore passes with a narrow prior
    and fails with a wide one, which is the cleanest demonstration that the
    probe magnitude is read off the prior rather than fixed.
    """
    x = jnp.linspace(1.0, 2.0, n)
    data = 1.0 * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, prior_std))
        mu = det(
            "mu", lambda w_, x_: (w_ + curvature * w_**3) * x_, w, xs, linear_in=("w",)
        )
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def affine_only_at_zero(*, n=6, sigma=0.4, seed=10):
    """``mu = (x + z**2 x**2) X`` -- affine in ``x`` only where ``z == 0``.

    ``z ~ N(3, 1)``, so a prior draw lands nowhere near zero. A check that
    probes only at the caller's ``at`` (with z pinned to 0) passes; a check
    that also probes at prior draws does not. That gap is the entire reason
    check_linearity takes several at-points.
    """
    x_grid = jnp.linspace(1.0, 2.0, n)
    data = 1.0 * x_grid + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x_grid)
        z = sample("z", lambda: dist.Normal(3.0, 1.0))
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        mu = det(
            "mu",
            lambda x_, z_, g_: (x_ + z_**2 * x_**2) * g_,
            x,
            z,
            xs,
            linear_in=("x",),
        )
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def nan_at_negative_probes(*, n=4, sigma=0.5):
    """``mu = sqrt(w) X`` -- NaN wherever a probe goes negative.

    Half of every symmetric probe does. NaN must count as a FAILURE:
    `nan > rtol` is False, so a naive comparison reads an unusable probe as
    evidence of linearity.
    """
    x = jnp.linspace(1.0, 2.0, n)

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(4.0, 1.0))
        mu = det("mu", lambda w_, x_: jnp.sqrt(w_) * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=2.0 * x)

    return trace(model)
```

- [ ] **Step 2: 写失败的测试**

```python
# tests/exact/test_linearity.py
"""Checking the linear_in claim -- at several scales and at several at-points."""

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from bayesmith import sample, trace
from bayesmith.errors import GraphError, StructureError
from bayesmith.exact.block import unchecked_operator
from bayesmith.exact.linearity import check_linearity, linear_operator
from tests.exact.models import (
    affine_only_at_zero,
    bilinear_pair,
    cubic_tail,
    nan_at_negative_probes,
    quadratic_claim,
    straight_line,
    two_linear_latents,
)


def test_a_genuinely_linear_block_passes_at_every_scale():
    graph = straight_line()
    errors = check_linearity(graph, ("w",), at={})
    assert errors  # one entry per at-point
    for per_point in errors.values():
        assert len(per_point) == 3
        assert all(err < 1e-3 for err in per_point.values())


def test_a_quadratic_claim_is_refused():
    graph = quadratic_claim()
    with pytest.raises(StructureError, match="affine"):
        check_linearity(graph, ("w",), at={})


def test_a_bilinear_pair_passes_singly_and_fails_jointly():
    """rheplicant's motivating failure, caught by three forward evaluations.

    Each conditional genuinely IS affine, which is why checking one latent at
    a time cannot see this and why a hand-rolled alternating solve reports a
    CG residual of 1e-7 and a per-block condition number of ~1.5 while landing
    thousands of kelvin away. The claim that is false is the JOINT one.
    """
    graph = bilinear_pair()
    check_linearity(graph, ("gain",), at={"t_ant": jnp.asarray(2.0)})
    check_linearity(graph, ("t_ant",), at={"gain": jnp.asarray(1.0)})
    with pytest.raises(StructureError, match="JOINTLY"):
        check_linearity(graph, ("gain", "t_ant"), at={})


def test_the_probe_magnitude_is_read_off_the_declared_prior():
    """One fn, two declared prior widths, opposite verdicts.

    Nothing about the model changes between these two calls except the width
    the graph declares for `w`. If the probe magnitude were a fixed constant
    the two would agree, whichever way.
    """
    with pytest.raises(StructureError, match="affine"):
        check_linearity(cubic_tail(prior_std=1.0), ("w",), at={})
    check_linearity(cubic_tail(prior_std=1e-4), ("w",), at={})


def test_check_linearity_probes_more_than_the_caller_s_at_point():
    """The claim holds where the caller pinned z, and nowhere the prior goes.

    Pinned explicitly to a single at-point the check passes; left to its
    default -- the caller's at PLUS draws from the graph's own prior -- it
    does not. An implementation that probed one point would pass both.
    """
    graph = affine_only_at_zero()
    pinned = {"z": jnp.asarray(0.0)}
    check_linearity(graph, ("x",), at=pinned, at_points=[pinned])
    with pytest.raises(StructureError, match="affine"):
        check_linearity(graph, ("x",), at=pinned)


def test_a_probe_that_returns_nan_counts_as_a_failure():
    graph = nan_at_negative_probes()
    with pytest.raises(StructureError, match="affine"):
        check_linearity(graph, ("w",), at={})


def test_linear_operator_checks_before_it_builds():
    """The safe name is the natural one; the unchecked primitive says so."""
    graph = quadratic_claim()
    with pytest.raises(StructureError):
        linear_operator(graph, ("w",), at={})
    unchecked_operator(graph, ("w",), at={})  # builds happily, and is wrong


def test_linear_operator_returns_the_block_the_primitive_would_have():
    graph = two_linear_latents()
    at = {"b": jnp.asarray(4.0)}
    checked = linear_operator(graph, ("a",), at=at)
    raw = unchecked_operator(graph, ("a",), at=at)
    assert jnp.allclose(checked.offset["d"], raw.offset["d"])
    probe = {"a": jnp.asarray(1.0)}
    assert jnp.allclose(checked.forward(probe)["d"], raw.forward(probe)["d"])
    assert jnp.allclose(checked.prior_std["a"], raw.prior_std["a"])


def test_a_graph_with_no_observed_node_is_refused():
    """There is nothing to condition on, so the posterior IS the prior.

    Refused by name rather than reaching affinity_errors, where an empty
    codomain has no dtype to take a tolerance from and the failure would
    arrive as an unrelated TypeError from two layers down.
    """

    def model():
        sample("w", lambda: dist.Normal(0.0, 1.0))

    graph = trace(model)
    with pytest.raises(GraphError, match="observed"):
        unchecked_operator(graph, ("w",), at={})
```

- [ ] **Step 3: 跑测试，确认失败**

```bash
.venv/bin/python -m pytest tests/exact/test_linearity.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'bayesmith.exact.linearity'`。

- [ ] **Step 4: 给 `unchecked_operator` 加"必须有观测节点"的拒绝**

在 `src/bayesmith/exact/block.py` 的 `unchecked_operator` 里，`_refuse_internal_ancestry(graph, names)` 之后插入：

```python
    if not graph.observed:
        raise GraphError(
            "this graph has no observed node, so a linear block has nothing to "
            "condition on and its posterior is exactly its prior. Refused by "
            "name here rather than further down, where an empty codomain has no "
            "dtype for a tolerance to be taken from and the failure arrives as "
            "an unrelated TypeError from two layers away."
        )
```

- [ ] **Step 5: 实现 `exact/linearity.py`**

```python
# src/bayesmith/exact/linearity.py
"""Checking the linear_in claim before anything exploits it.

``Deterministic(linear_in=("w",))`` promises that, holding every other latent
fixed, every prediction is an **affine** function of ``w``::

    prediction(w) = A w + b

The promise is checkable, and this module checks it: :func:`check_linearity`
compares the model against its own linearization at zero, at several probe
magnitudes and at several values of the latents outside the block. A false
declaration would otherwise produce a confident, wrong posterior instead of
an error.

**Two entry points, not a flag.** :func:`linear_operator` checks and then
builds, and is what callers should reach for.
:func:`~bayesmith.exact.block.unchecked_operator` skips the check and says so
in its name -- for inside a Gibbs sweep, where the check is hoisted out of the
loop deliberately. rheplicant spells this as ``linear_operator(check=True)``,
which makes the most natural call name the one that is one keyword away from
unsafe.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from bayesmith.errors import StructureError
from bayesmith.exact.block import (
    LinearBlock,
    _env_before,
    _refuse_internal_ancestry,
    _validated_names,
    isolate,
    unchecked_operator,
)
from bayesmith.graph.graph import Graph

#: Probe magnitudes, as multiples of each latent's declared prior standard
#: deviation. Spans six orders of magnitude on purpose: curvature that is
#: invisible near the prior's centre is what a sampler wanders into.
DEFAULT_SCALES: tuple[float, ...] = (1e-3, 1.0, 1e3)

#: How many values of the OUTSIDE latents the claim is checked at, the
#: caller's own ``at`` included. Extras are drawn from the graph's own prior.
DEFAULT_AT_POINTS: int = 3


def _biggest(tree: Any) -> float:
    leaves = jax.tree.leaves(tree)
    return max(float(jnp.max(jnp.abs(leaf))) for leaf in leaves)


def affinity_errors(
    g: Callable[[Any], Any],
    zero: Any,
    probe_at: Callable[[int, float], Any],
    scales: Sequence[float],
    rtol: float | None,
) -> tuple[dict[float, float], list[float], float]:
    """Compare a map against its own linearization at zero, probe by probe.

    Every number below comes from ``g``, ``zero`` and the probe alone, so a
    single-latent and a grouped check cannot drift into measuring different
    things. Ported from rheplicant, generalised to a pytree codomain.
    """
    baseline, tangent = jax.linearize(g, zero)
    dtype = jnp.result_type(*jax.tree.leaves(baseline))
    if rtol is None:
        rtol = 1e4 * float(jnp.finfo(dtype).eps)
    epsilon = float(jnp.finfo(dtype).eps)

    errors: dict[float, float] = {}
    verdicts: dict[float, bool] = {}
    for index, scale in enumerate(scales):
        probe = probe_at(index, scale)
        actual = g(probe)
        predicted = jax.tree.map(lambda b, t: b + t, baseline, tangent(probe))
        # Measure against the VARIATION, not the total: a large constant
        # offset would otherwise hide a completely nonlinear response.
        variation = _biggest(jax.tree.map(jnp.subtract, actual, baseline))
        departure = _biggest(jax.tree.map(jnp.subtract, actual, predicted))
        errors[scale] = departure / max(variation, 1e-300)

        # A departure smaller than the arithmetic's OWN noise floor is not
        # evidence of curvature; without this the relative measure explodes at
        # small probes, where the variation is vanishing but roundoff is not,
        # and rejects perfectly linear blocks. The floor is set by the
        # magnitudes actually being differenced AT THIS PROBE -- not by a
        # constant, which would exempt every model whose prediction is small
        # in its own units, and not by the baseline alone, which would let an
        # unrelated bright component disable the check.
        floor = 1e4 * epsilon * max(_biggest(actual), _biggest(baseline))
        # NaN must count as a FAILURE: `nan > rtol` is False, so a naive
        # comparison reads an unusable probe as evidence of linearity.
        finite = np.isfinite(errors[scale]) and np.isfinite(departure)
        verdicts[scale] = (not finite) or (errors[scale] > rtol and departure > floor)

    failed = sorted(scale for scale, bad in verdicts.items() if bad)
    return errors, failed, rtol


def prior_at_points(
    graph: Graph, names: tuple[str, ...], count: int, key: jax.Array
) -> list[dict[str, Any]]:
    """``count`` alternative values for the latents OUTSIDE the block.

    Drawn from the graph's own prior, through the NumPyro bridge -- so they
    cover exactly the range the model itself considers plausible, and they
    work for a non-Gaussian outside latent (which is the usual case: a latent
    is outside the block precisely because it is not conjugate).
    """
    from numpyro import handlers

    from bayesmith.bridge.numpyro_bridge import to_numpyro

    model = to_numpyro(graph)
    outside = [name for name in graph.latents if name not in set(names)]
    points: list[dict[str, Any]] = []
    for index in range(count):
        traced = handlers.trace(
            handlers.seed(model, jax.random.fold_in(key, index))
        ).get_trace()
        points.append({name: traced[name]["value"] for name in outside})
    return points


def check_linearity(
    graph: Graph,
    names: Iterable[str],
    at: dict[str, Any] | None = None,
    *,
    scales: Sequence[float] = DEFAULT_SCALES,
    rtol: float | None = None,
    at_points: Sequence[dict[str, Any]] | None = None,
    key: jax.Array | None = None,
) -> dict[int, dict[float, float]]:
    """Verify every prediction really is affine in a block -- or in a group.

    Costs one linearization plus one forward evaluation per scale per
    at-point: with the defaults, three of each.

    Args:
        graph: the model under test.
        names: the latents in the block. Checked **jointly**, which is
            strictly stronger than checking each in turn. A gain and an
            antenna temperature are each affine given the other and their
            product is not affine in the pair, so a group holding both is
            refused here rather than solved as if it were linear.
        at: values for the latents OUTSIDE the block.
        scales: probe magnitudes, as multiples of each latent's own declared
            prior standard deviation -- per element, so a latent whose prior
            width varies across its entries is probed accordingly.
        rtol: tolerance on the relative departure from affinity. Default
            ``1e4 * eps`` of the prediction's dtype.
        at_points: values of the outside latents to check at. Defaults to
            ``at`` plus ``DEFAULT_AT_POINTS - 1`` draws from the graph's own
            prior. **Passing a single point is how a check becomes a
            moderate-parameter probe**, which is the failure mode
            ``boundary-validation.md`` exists to prevent; do it only when the
            model is used at exactly one outside value.
        key: PRNG key for probes and prior draws. Fixed by default, so the
            check is reproducible. Per-latent sub-keys are folded in by
            position in the SORTED names, so permuting ``names`` probes the
            same points and returns the same verdict.

    Returns:
        ``{at_point_index: {scale: relative error}}`` -- useful for reporting
        how linear a block is, not only whether it passes.

    Raises:
        StructureError: if any scale at any at-point departs from affinity by
            more than ``rtol`` while also exceeding the per-probe roundoff
            floor.
        GraphError, NotGaussian: propagated from the block machinery.
    """
    names = _validated_names(graph, names)
    at = dict(at or {})
    _refuse_internal_ancestry(graph, names)
    key = jax.random.key(0) if key is None else key

    if at_points is None:
        at_points = [
            at,
            *prior_at_points(
                graph, names, DEFAULT_AT_POINTS - 1, jax.random.fold_in(key, 7919)
            ),
        ]

    ordered = sorted(names)
    collected: dict[int, dict[float, float]] = {}
    for point_index, point in enumerate(at_points):
        _, domain = _env_before(graph, names, point)
        g = isolate(graph, names, point)
        zero = {n: jnp.zeros(domain[n][0], dtype=domain[n][1]) for n in names}
        point_key = jax.random.fold_in(key, point_index)

        def probe_at(index: int, scale: float, _domain=domain, _k=point_key):
            root = jax.random.fold_in(_k, index)
            return {
                member: _domain[member][3]
                * scale
                * jax.random.normal(
                    jax.random.fold_in(root, position),
                    _domain[member][0],
                    dtype=_domain[member][1],
                )
                for position, member in enumerate(ordered)
            }

        errors, failed, used_rtol = affinity_errors(g, zero, probe_at, scales, rtol)
        collected[point_index] = errors
        if failed:
            subject = (
                f"latent {names[0]!r} is declared linear, but the prediction is "
                "not affine in it"
                if len(names) == 1
                else f"latents {list(names)} are not JOINTLY affine -- each "
                "conditional may well be, which is exactly why this is not "
                "caught one latent at a time"
            )
            detail = ", ".join(f"{s:g}x -> {e:.2e}" for s, e in errors.items())
            where = (
                "the caller's own `at`"
                if point_index == 0
                else f"prior draw {point_index} of the outside latents"
            )
            raise StructureError(
                f"{subject}: departure from its own linearization exceeds "
                f"rtol={used_rtol:.2e} (above the per-probe roundoff floor) at "
                f"{failed} times each latent's declared prior width, evaluated at "
                f"{where} ({detail}). Either drop the linear_in declaration, or "
                "re-parameterize so the model really is affine there. For a group "
                "that is only pairwise affine, split it into separate blocks and "
                "alternate."
            )
    return collected


def linear_operator(
    graph: Graph,
    names: Iterable[str],
    at: dict[str, Any] | None = None,
    *,
    scales: Sequence[float] = DEFAULT_SCALES,
    rtol: float | None = None,
    at_points: Sequence[dict[str, Any]] | None = None,
    key: jax.Array | None = None,
) -> LinearBlock:
    """Check the linearity claim, then export the block. **The entry point.**

    Costs three forward evaluations per at-point more than
    :func:`~bayesmith.exact.block.unchecked_operator`, and buys the class of
    silent, confident errors that a false ``linear_in`` produces.

    In a Gibbs sweep, call this once outside the loop and
    ``unchecked_operator`` inside it: the claim is a property of the model,
    not of the sweep, so re-checking every sweep pays for the same answer
    repeatedly. The at-points this checks at are what make that safe.
    """
    check_linearity(
        graph, names, at, scales=scales, rtol=rtol, at_points=at_points, key=key
    )
    return unchecked_operator(graph, names, at)
```

- [ ] **Step 6: 跑测试，确认通过**

```bash
.venv/bin/python -m pytest tests/exact/test_linearity.py tests/exact/test_block.py -v
```

Expected: 全部 PASS（linearity 9 条 + block 14 条）。

- [ ] **Step 7: 变异测试**

1. 把 `check_linearity` 的默认 `at_points` 改成 `[at]`（只用调用方给的那个点），重跑。
   Expected: `test_check_linearity_probes_more_than_the_caller_s_at_point` **变红**。还原。
2. 把 `affinity_errors` 里的 `finite` 判断删掉（`verdicts[scale] = errors[scale] > rtol and departure > floor`），重跑。
   Expected: `test_a_probe_that_returns_nan_counts_as_a_failure` **变红**。还原。
3. 把 `probe_at` 里的 `_domain[member][3] *`（先验标准差）删掉，让探针成为绝对幅度，重跑。
   Expected: `test_the_probe_magnitude_is_read_off_the_declared_prior` **变红**（两个先验宽度给出同一个判决）。还原。
4. 把 `check_linearity` 改成逐个成员检查而非联合检查（对 `names` 里每个名字单独调用），重跑。
   Expected: `test_a_bilinear_pair_passes_singly_and_fails_jointly` **变红**。还原。

- [ ] **Step 8: 提交**

```bash
git add src/bayesmith/exact/linearity.py src/bayesmith/exact/block.py tests/exact/models.py tests/exact/test_linearity.py
git commit -m "feat: check the linear_in claim at several scales and at-points"
```

---
## Task 5：正规算子与条件数上界（`exact/solve.py` 之一）


`tol` 不是精度。残差与误差差一个条件数：

    ‖x̂ − x*‖ / ‖x*‖  ≤  κ(M) · ‖M x̂ − b‖ / ‖b‖

对数据无法完全辨识的块——一个定标负载对三个未知量、一个被标记的频道、一次短积分——**先验是唯一压住盲方向的东西**，于是 `λ_min(M)` 恰好是 `1/prior_std²`，`κ ≈ ‖AᵀN⁻¹A‖ · prior_std²` 轻易到 1e6 以上。κ=1e7 时默认 `tol=1e-6` 把相对误差界到 10：**一位有效数字都没有**。CG 停在一个看起来收敛的残差上，把先验主导的那些方向留在起点，抽出来的样本散度小了好几个量级。

这正是这些求解器存在的理由所在的区域，所以守卫默认开着，且精度目标以**误差**而非残差陈述。

**Files:**
- Create: `src/bayesmith/exact/solve.py`
- Create: `tests/exact/oracle.py`, `tests/exact/test_solve.py`

- [ ] **Step 1: 写 R2 预言机**

```python
# tests/exact/oracle.py
"""The independent dense oracle -- no autodiff, no JAX transformation.

Builds the same posterior the matrix-free CG path builds, by the most naive
route available: evaluate the block's own map on a basis of its domain,
assemble a dense ``A`` column by column, and solve the normal equations with
``numpy.linalg``.

**What it shares, and what it must not.** It shares the MODEL -- `isolate`'s
``g``, which is `evaluate` plus reading each observed node's loc -- because an
oracle has to evaluate the same model or it answers a different question. It
shares none of the LINEAR ALGEBRA: no `jax.linearize`, no `jax.vjp`, no `cg`,
no `tree_norm`, no power iteration. `A[:, j] = g(e_j) - g(0)` is the
definition of a linear map on a basis, exact for an affine ``g``, and the only
thing it trusts is that calling ``g`` twice gives the same answer twice.

That independence is the point. The P1 design record names the failure this
guards against: two readings of one graph that share an implementation share
its blind spots, and agreed on -225.65 while the truth was -364.95. "Exact vs
NUTS" is a self-consistency check for the same reason -- both go through
`apply_probabilistic`. This is not.

**Build the graph INSIDE the x64 context.** `jax.enable_x64(True)` is
thread-local and affects arrays created after it, so a graph traced at
float32 stays float32 no matter what context the solve runs in -- `const` and
`observe` call `jnp.asarray` at trace time. Every test below therefore traces
its model inside the `with` block, and the oracle would otherwise be the less
accurate of the two things being compared.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np


class Oracle(NamedTuple):
    """Everything the dense route computes, in one flat layout.

    ``order`` is ``[(latent_name, flat_index), ...]`` for the domain, so a
    caller can line a solve's ``{name: array}`` answer up against ``mean``
    without guessing offsets -- :func:`flat_domain` is that map.
    """

    mean: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray
    design: np.ndarray
    offset: np.ndarray
    data: np.ndarray
    sigma: np.ndarray
    prior_mean: np.ndarray
    prior_std: np.ndarray
    order: list[tuple[str, int]]


def _flatten(tree: dict[str, Any], order: list[str]) -> np.ndarray:
    return np.concatenate([np.asarray(tree[key]).ravel() for key in order])


def flat_domain(values: dict[str, Any], names: tuple[str, ...]) -> np.ndarray:
    """A solve's ``{name: array}`` answer, flattened the way ``Oracle`` is."""
    return _flatten(values, list(names))


def dense_design(g, shapes, dtypes, names, obs_order):
    """``(A, offset)`` with ``A[:, j] = g(e_j) - g(0)``."""
    zero = {n: jnp.zeros(shapes[n], dtype=dtypes[n]) for n in names}
    offset = _flatten(g(zero), obs_order)
    columns = []
    for name in names:
        size = int(np.prod(shapes[name], dtype=int))
        for index in range(size):
            flat = np.zeros(size)
            flat[index] = 1.0
            probe = dict(zero)
            probe[name] = jnp.asarray(
                flat.reshape(shapes[name] if shapes[name] else ()),
                dtype=dtypes[name],
            )
            columns.append(_flatten(g(probe), obs_order) - offset)
    return np.stack(columns, axis=1), offset


def analytic_posterior(design, offset, data, sigma, prior_mean, prior_std):
    """``(mean, covariance, precision)`` of the linear-Gaussian posterior."""
    noise_precision = np.diag(1.0 / np.asarray(sigma) ** 2)
    prior_precision = np.diag(1.0 / np.asarray(prior_std) ** 2)
    precision = design.T @ noise_precision @ design + prior_precision
    rhs = design.T @ noise_precision @ (data - offset) + prior_precision @ prior_mean
    covariance = np.linalg.inv(precision)
    return covariance @ rhs, covariance, precision


def graph_oracle(graph, names, at=None, sigma_at=None) -> Oracle:
    """The dense posterior of a block of ``graph``.

    Args:
        graph, names, at: the same three a block is built from.
        sigma_at: latent values to freeze sigma at, for a prediction-dependent
            noise model. Defaults to the block's zero, which is where a
            constant sigma is the same everywhere anyway.
    """
    from bayesmith.exact.block import _env_before, isolate
    from bayesmith.exact.gaussian import noise_std_at, observation_parts
    from bayesmith.graph.evaluate import evaluate

    names = tuple(names)
    at = dict(at or {})
    _, domain = _env_before(graph, names, at)
    shapes = {n: domain[n][0] for n in names}
    dtypes = {n: domain[n][1] for n in names}
    zero = {n: jnp.zeros(shapes[n], dtype=dtypes[n]) for n in names}

    obs_order = sorted(graph.observed)
    design, offset = dense_design(
        isolate(graph, names, at), shapes, dtypes, names, obs_order
    )
    data_tree, _, _ = observation_parts(graph, evaluate(graph, {**at, **zero}))
    sigma = noise_std_at(graph, {**at, **(sigma_at or zero)})

    prior_mean = _flatten({n: domain[n][2] for n in names}, list(names))
    prior_std = _flatten({n: domain[n][3] for n in names}, list(names))
    mean, covariance, precision = analytic_posterior(
        design,
        offset,
        _flatten(data_tree, obs_order),
        _flatten(sigma, obs_order),
        prior_mean,
        prior_std,
    )
    order = [
        (n, i) for n in names for i in range(int(np.prod(shapes[n], dtype=int)))
    ]
    return Oracle(
        mean=mean,
        covariance=covariance,
        precision=precision,
        design=design,
        offset=offset,
        data=_flatten(data_tree, obs_order),
        sigma=_flatten(sigma, obs_order),
        prior_mean=prior_mean,
        prior_std=prior_std,
        order=order,
    )
```

- [ ] **Step 2: 写失败的测试**

```python
# tests/exact/test_solve.py
"""The conjugate solves, against a dense oracle that shares none of them."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.exact.gaussian import noise_std_at
from bayesmith.exact.linearity import linear_operator
from bayesmith.exact.solve import condition_bound, normal_operator
from bayesmith.exact.block import domain_zero, variance_parts
from tests.exact.models import straight_line, two_linear_latents
from tests.exact.oracle import graph_oracle


def _sigma(graph, at):
    return noise_std_at(graph, at)


def test_the_bound_is_lambda_max_times_the_loosest_prior_variance():
    """The bound's definition, checked against a dense eigendecomposition.

    lambda_max comes from the oracle's precision matrix -- which is built by
    probing g on a basis and never differentiates anything -- so this is the
    matrix-free power iteration against an independent route, not against
    itself.
    """
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = _sigma(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        bound = float(condition_bound(block, noise_std=sigma, iterations=80))
        oracle = graph_oracle(graph, ("a", "b"), at={})
    largest = float(np.linalg.eigvalsh(oracle.precision)[-1])
    loosest_variance = float(np.max(oracle.prior_std**2))
    assert bound == pytest.approx(largest * loosest_variance, rel=1e-3)


@pytest.mark.parametrize("loosened", [1.0, 1e2, 1e4])
def test_the_bound_is_never_below_the_true_condition_number(loosened):
    """The whole point: it may refuse a good solve, never accept a bad one.

    Swept across four orders of magnitude of prior width, because the bound is
    tight at one end (the prior alone holds a direction, so lambda_min IS the
    prior curvature) and loose at the other. Both must stay on the safe side.
    """
    import dataclasses

    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        widened = dataclasses.replace(
            block,
            prior_std={**block.prior_std, "b": block.prior_std["b"] * loosened},
        )
        sigma = _sigma(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        bound = float(condition_bound(widened, noise_std=sigma, iterations=80))
        oracle = graph_oracle(graph, ("a", "b"), at={})
    precision = oracle.precision.copy()
    # Rebuild the dense precision with b's widened prior, so the comparison is
    # against the system the bound was actually computed for.
    precision[1, 1] += 1.0 / (oracle.prior_std[1] * loosened) ** 2 - 1.0 / oracle.prior_std[1] ** 2
    true_kappa = float(np.linalg.cond(precision))
    assert bound >= true_kappa * (1.0 - 1e-6), (bound, true_kappa)


def test_the_bound_is_loose_when_the_data_constrains_every_direction():
    """Stated rather than hidden: this is the price of a one-sided guarantee.

    A one-parameter block has a true kappa of exactly 1 -- M is 1x1 -- and the
    bound reports lambda_max times the prior variance, which is hundreds. That
    is not a defect; it is what an upper bound derived from the prior must
    say when the data, not the prior, is what sets lambda_min. CG on such a
    block converges in one step, so the residual absorbs the slack.
    """
    with jax.enable_x64(True):
        graph = straight_line()
        block = linear_operator(graph, ("w",), at={})
        sigma = _sigma(graph, {"w": jnp.asarray(0.0)})
        bound = float(condition_bound(block, noise_std=sigma, iterations=40))
        oracle = graph_oracle(graph, ("w",), at={})
    assert float(np.linalg.cond(oracle.precision)) == pytest.approx(1.0, rel=1e-9)
    assert bound > 100.0
    assert bound == pytest.approx(
        float(oracle.precision[0, 0]) * float(oracle.prior_std[0] ** 2), rel=1e-3
    )


def test_the_bound_grows_in_proportion_to_the_loosest_prior_variance():
    """It is linear in max(prior_variance) by construction -- verify it is.

    A bound that ignored the prior would be flat across this pair, and a bound
    that took the TIGHTEST prior would move the wrong way.
    """
    import dataclasses

    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = _sigma(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        tight = float(condition_bound(block, noise_std=sigma, iterations=80))
        widened = dataclasses.replace(
            block, prior_std={**block.prior_std, "b": block.prior_std["b"] * 100.0}
        )
        loose = float(condition_bound(widened, noise_std=sigma, iterations=80))
    # b's prior variance grows by 1e4 and it was already the loosest (7 vs 5).
    assert loose == pytest.approx(1e4 * tight, rel=1e-2)


def test_the_normal_operator_is_symmetric():
    """`<u, M v> == <M u, v>` -- taking the curvature as a gradient of chi2
    makes this true by construction, and an adjoint assembled by hand from A
    and A^T is where the sign conventions would go wrong."""
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        weight = {
            o: 1.0 / jnp.asarray(s) ** 2
            for o, s in _sigma(graph, {"a": 0.0, "b": 0.0}).items()
        }
        normal = normal_operator(block, weight, variance_parts(block))
        keys = jax.random.split(jax.random.key(3), 2)
        u = {
            n: jax.random.normal(k, block.shape[n])
            for n, k in zip(block.names, keys, strict=True)
        }
        v = {
            n: jax.random.normal(jax.random.fold_in(k, 1), block.shape[n])
            for n, k in zip(block.names, keys, strict=True)
        }
        left = sum(float(jnp.sum(u[n] * normal(v)[n])) for n in block.names)
        right = sum(float(jnp.sum(normal(u)[n] * v[n])) for n in block.names)
    assert left == pytest.approx(right, rel=1e-10)


def test_the_normal_operator_reproduces_the_dense_precision_matrix():
    """Applying M to each basis vector must give the dense precision's columns.

    The dense matrix comes from the oracle, which never calls linearize or
    vjp; M comes from `jax.grad` of the block's own chi-squared. They agree
    or one of them is wrong.
    """
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        weight = {
            o: 1.0 / jnp.asarray(s) ** 2
            for o, s in _sigma(graph, {"a": 0.0, "b": 0.0}).items()
        }
        normal = normal_operator(block, weight, variance_parts(block))
        columns = []
        for name in block.names:
            probe = dict(domain_zero(block))
            probe[name] = jnp.asarray(1.0)
            image = normal(probe)
            columns.append(np.concatenate([np.asarray(image[n]).ravel() for n in block.names]))
        applied = np.stack(columns, axis=1)
        oracle = graph_oracle(graph, ("a", "b"), at={})
    assert np.allclose(applied, oracle.precision, rtol=1e-8)
```

- [ ] **Step 3: 跑测试，确认失败**

```bash
.venv/bin/python -m pytest tests/exact/test_solve.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'bayesmith.exact.solve'`。

- [ ] **Step 4: 实现（本任务只写到 `condition_bound`）**

```python
# src/bayesmith/exact/solve.py
"""Conjugate-Gaussian solves over a matrix-free linear block.

The posterior mean (:func:`wiener_solve`) and an exact posterior draw
(:func:`gcr_sample`) are the same conjugate-gradient solve of

    (A^T N^-1 A + S^-1) x = b

differing only in ``b``. They share one private routine that takes ``key=None``
for the mean and a key for a draw, which is why the two can never drift apart.

**The normal operator and the right-hand side are obtained as gradients of the
objective itself**, never assembled from ``A`` and ``A^T`` by hand. That is
not a shortcut: it makes the operator symmetric positive definite *by
construction*, with no adjoint convention left to get wrong -- and, for a
group, no cross-block bookkeeping either, since ``jax.grad`` of the group's
own chi-squared produces the full operator, off-diagonal blocks included,
which is exactly the coupling an alternating solve throws away.

Ported from ``rheplicant.inference.linear``, with the codomain generalised
from one array to a dict of observed nodes, and with every prior/noise
keyword removed -- the graph is the only statement of either.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from bayesmith.exact.block import (
    LinearBlock,
    domain_zero,
    largest_variance,
    variance_parts,
)
from bayesmith.exact.conditioning import largest_eigenvalue, tree_norm

#: Power-iteration steps for the top of the spectrum. The estimate typically
#: settles within three; this leaves margin at a fixed cost of
#: ``POWER_ITERATIONS`` operator applications per guarded solve. Only the top
#: is measured -- see :func:`_condition_bound` for where the bottom comes from
#: and why it is not measured.
POWER_ITERATIONS: int = 12


def _weights(noise_std: dict[str, Any]) -> dict[str, jax.Array]:
    return {name: 1.0 / jnp.asarray(std) ** 2 for name, std in noise_std.items()}


def normal_operator(
    block: LinearBlock, weight: dict[str, Any], prior_variance: dict[str, Any]
):
    """``x -> (A^T N^-1 A + S^-1) x`` over the block's domain."""

    def half_chi2(parts):
        pushed = block.forward(parts)
        return 0.5 * sum(jnp.sum(weight[name] * pushed[name] ** 2) for name in pushed)

    def normal(parts):
        curvature = jax.grad(half_chi2)(parts)
        return jax.tree.map(
            lambda c, p, v: c + p / v, curvature, parts, prior_variance
        )

    return normal


def _condition_bound(
    block: LinearBlock,
    weight: dict[str, Any],
    prior_variance: dict[str, Any],
    key: jax.Array,
    iterations: int,
) -> jax.Array:
    """Upper bound on ``kappa`` of ``A^T N^-1 A + S^-1``.

    ``lambda_max`` is measured. ``lambda_min`` is **not**: it is bounded from
    below by the prior's own curvature, because ``A^T N^-1 A`` is positive
    semi-definite and therefore

        lambda_min(A^T N^-1 A + S^-1)  >=  lambda_min(S^-1)  =  1 / max(S)

    so the quotient is an upper bound rather than an estimate. That is the
    direction a safety guard needs -- an overestimate of kappa can only make
    the guard refuse a solve that was fine, while an underestimate makes it
    accept one that was not.

    Measuring ``lambda_min`` instead, by a second power iteration on
    ``lambda_max * I - M``, is what rheplicant does and what this plan
    originally specified. It was measured to fail *in principle* on a graded
    spectrum -- the shifted operator's leading eigenvalues all crowd against
    ``lambda_max`` with vanishing gaps -- and to fail one-sidedly in the
    dangerous direction. See :mod:`bayesmith.exact.conditioning`'s module
    docstring for the numbers.

    **The bound is tight exactly where it matters.** For a block the data does
    not fully identify, some direction is held by the prior alone and
    ``lambda_min`` IS the prior's curvature. It is loose in the opposite
    regime -- data far tighter than the prior in every direction -- where the
    guard may refuse a solve that was in fact accurate. That regime is also
    where CG converges in a handful of iterations and the residual is small
    enough to absorb the slack, which is why the trade is worth taking.

    For a group this is the JOINT bound, and it is the number a per-block
    guard cannot produce: two latents the data barely distinguishes give a
    well-conditioned operator each and a badly conditioned one together.
    """
    largest = largest_eigenvalue(
        normal_operator(block, weight, prior_variance),
        domain_zero(block),
        key,
        iterations,
    )
    return largest * largest_variance(prior_variance)


def condition_bound(
    block: LinearBlock,
    *,
    noise_std: dict[str, Any],
    iterations: int = POWER_ITERATIONS,
    key: jax.Array | None = None,
) -> jax.Array:
    """An upper bound on the conditioning of the system this block is solved with.

    Use it to pick ``tol``: for a target relative accuracy ``a``, ask for
    roughly ``tol = a / condition_bound(...)``.

    A large bound is not a defect, it is the design: for a block the data does
    not fully identify, ``lambda_min`` is exactly ``1/prior_std**2`` while
    ``lambda_max`` is set by the data, so it grows with how much better the
    data constrains one direction than the prior constrains another.

    Costs ``iterations`` applications of the normal operator -- each the same
    JVP-plus-VJP a CG iteration costs -- and forms no matrix.

    Args:
        block: from :func:`bayesmith.exact.linearity.linear_operator`.
        noise_std: ``{observed: sigma}``, as
            :func:`bayesmith.exact.gaussian.noise_std_at` returns. A decided
            sigma, not a rule for producing one: a conditioning number belongs
            to one particular normal operator.
        iterations: power-iteration steps for ``lambda_max``.
        key: PRNG key for the starting vector. Fixed by default, so the bound
            is reproducible.

    Returns:
        ``lambda_max * max(prior_variance)``, an upper bound on ``kappa`` --
        up to the accuracy of the ``lambda_max`` estimate, which converges
        geometrically and always from BELOW, so it can only make the bound
        smaller. ``test_largest_eigenvalue_approaches_the_truth_from_below``
        in ``tests/exact/test_conditioning.py`` pins that direction.
    """
    return _condition_bound(
        block,
        _weights(noise_std),
        variance_parts(block),
        jax.random.key(0) if key is None else key,
        iterations,
    )
```

- [ ] **Step 5: 跑测试，确认通过**

```bash
.venv/bin/python -m pytest tests/exact/test_solve.py -v
```

Expected: 9 passed（参数化的那条算三例）。

- [ ] **Step 6: 变异测试**

1. 把 `_condition_bound` 的 `largest_variance` 换成"最紧的先验方差"（`jnp.min` 而非 `jnp.max`），重跑。
   Expected: `test_the_bound_grows_in_proportion_to_the_loosest_prior_variance` 与 `test_the_bound_is_never_below_the_true_condition_number[10000.0]` **都变红**。第二条尤其重要——取最紧的先验会让这个"界"掉到真 κ 之下，也就是**失去它唯一的保证**。还原。
2. 把 `normal_operator` 里的 `+ p / v`（先验曲率项）删掉，重跑。
   Expected: `test_the_normal_operator_reproduces_the_dense_precision_matrix` **变红**。还原。
3. 把 `half_chi2` 的求和改成只对 `sorted(pushed)[0]` 一个观测节点求和，重跑。
   Expected: `test_the_normal_operator_sums_over_every_observed_node` **变红**。还原。
   （`two_linear_latents` 只有一个观测节点，所以这条变异必须由下面这条多观测测试来抓——把它一并写进 `tests/exact/test_solve.py`。）

```python
def test_the_normal_operator_sums_over_every_observed_node():
    """Two observed nodes both contribute curvature; dropping one changes it."""
    from tests.exact.models import two_observations

    with jax.enable_x64(True):
        graph = two_observations()
        block = linear_operator(graph, ("w",), at={})
        weight = {
            o: 1.0 / jnp.asarray(s) ** 2
            for o, s in _sigma(graph, {"w": jnp.asarray(0.0)}).items()
        }
        normal = normal_operator(block, weight, variance_parts(block))
        applied = float(normal({"w": jnp.asarray(1.0)})["w"])
        oracle = graph_oracle(graph, ("w",), at={})
    assert applied == pytest.approx(float(oracle.precision[0, 0]), rel=1e-8)
```

- [ ] **Step 7: 提交**

```bash
git add src/bayesmith/exact/solve.py tests/exact/oracle.py tests/exact/test_solve.py
git commit -m "feat: normal operator and matrix-free condition estimate"
```

---
## Task 6：后验均值（`exact/solve.py` 之二）

`d = A x + offset + n`，`n ~ N(0, N)`，`x ~ N(m, S)`：

    x̂ = (AᵀN⁻¹A + S⁻¹)⁻¹ [AᵀN⁻¹(d − offset) + S⁻¹m]

用共轭梯度解，算子只被**应用**、从不被形成。每次迭代一次 JVP 加一次 VJP——这正是 10⁶ 自由度的块可算的原因。

**Files:**
- Modify: `src/bayesmith/exact/solve.py`
- Modify: `tests/exact/models.py`, `tests/exact/test_solve.py`

- [ ] **Step 1: 加"无数据隐变量"玩具模型**

追加到 `tests/exact/models.py`：

```python
def unconstrained_latent(*, n=5, sigma=0.5, seed=11):
    """``u`` reaches no observed node, so its posterior IS its prior.

    An extreme corner the solve must handle rather than divide by: A's column
    for ``u`` is exactly zero, so the normal operator there is the prior
    curvature alone and the answer is the prior mean. ``u``'s centre (1.25)
    and width (0.75) are distinct from every other number in the model, so a
    solve that returned zero, or the other latent's prior, could not pass.
    """
    x = jnp.linspace(1.0, 2.0, n)
    data = 2.0 * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 3.0))
        sample("u", lambda: dist.Normal(1.25, 0.75))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)
```

- [ ] **Step 2: 写失败的测试**

追加到 `tests/exact/test_solve.py`（并在文件顶部补 `from bayesmith.exact.solve import wiener_solve`，以及 `from tests.exact.models import plated_latent, two_observations, unconstrained_latent`、`from tests.exact.oracle import flat_domain`）：

```python
def test_wiener_solve_matches_the_dense_oracle():
    """R1 vs R2 -- the acceptance gate of this whole plan.

    Nothing is shared between the two but the model: the oracle has no
    linearize, no vjp, no cg, no tree_norm and no power iteration in it.
    """
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = _sigma(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        got, residual = wiener_solve(block, noise_std=sigma, tol=1e-14)
        oracle = graph_oracle(graph, ("a", "b"), at={})
    assert float(residual) < 1e-10
    assert np.allclose(flat_domain(got, block.names), oracle.mean, rtol=1e-8)


def test_wiener_solve_matches_the_oracle_across_two_observed_nodes():
    """The pytree codomain: both observed nodes must enter A^T N^-1 (d-offset)."""
    with jax.enable_x64(True):
        graph = two_observations()
        block = linear_operator(graph, ("w",), at={})
        sigma = _sigma(graph, {"w": jnp.asarray(0.0)})
        got, _ = wiener_solve(block, noise_std=sigma, tol=1e-14)
        oracle = graph_oracle(graph, ("w",), at={})
    assert np.allclose(flat_domain(got, block.names), oracle.mean, rtol=1e-8)


def test_wiener_solve_matches_the_oracle_for_a_plated_block():
    """A six-dimensional domain, so the domain is a real vector space."""
    with jax.enable_x64(True):
        graph = plated_latent(n=6)
        block = linear_operator(graph, ("z",), at={})
        sigma = _sigma(graph, {"z": jnp.zeros(6)})
        got, _ = wiener_solve(block, noise_std=sigma, tol=1e-14)
        oracle = graph_oracle(graph, ("z",), at={})
    assert got["z"].shape == (6,)
    assert np.allclose(flat_domain(got, block.names), oracle.mean, rtol=1e-8)


def test_a_latent_the_data_never_reaches_comes_back_at_its_prior_mean():
    with jax.enable_x64(True):
        graph = unconstrained_latent()
        block = linear_operator(graph, ("w", "u"), at={})
        sigma = _sigma(graph, {"w": jnp.asarray(0.0), "u": jnp.asarray(0.0)})
        got, _ = wiener_solve(block, noise_std=sigma, tol=1e-14)
    assert float(got["u"]) == pytest.approx(1.25, rel=1e-6)


def test_the_convergence_guard_fires_on_a_deliberately_starved_solve():
    """maxiter=1 leaves a real residual, and the guard bounds the ERROR.

    equinox's runtime-error type varies between versions, so the assertion is
    on the message rather than the class.
    """
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = _sigma(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        with pytest.raises(Exception, match="did not converge"):
            wiener_solve(
                block, noise_std=sigma, tol=1e-14, maxiter=1, require_convergence=1e-12
            )


def test_disabling_the_guard_returns_the_unconverged_answer_instead():
    """The guard is what turns a bad answer into an error, nothing else does."""
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = _sigma(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        got, residual = wiener_solve(
            block, noise_std=sigma, tol=1e-14, maxiter=1, require_convergence=None
        )
        oracle = graph_oracle(graph, ("a", "b"), at={})
    assert float(residual) > 1e-12
    assert not np.allclose(flat_domain(got, block.names), oracle.mean, rtol=1e-8)
```

- [ ] **Step 3: 跑测试，确认失败**

```bash
.venv/bin/python -m pytest tests/exact/test_solve.py -v -k wiener
```

Expected: FAIL — `ImportError: cannot import name 'wiener_solve'`。

- [ ] **Step 4: 实现**

在 `src/bayesmith/exact/solve.py` 顶部的导入里加 `import equinox as eqx`，并在 `condition_bound` 之后追加：

```python
def _split_like(key: jax.Array, template: Any) -> Any:
    """One independent key per leaf of ``template``, same structure."""
    leaves, treedef = jax.tree.flatten(template)
    return jax.tree.unflatten(treedef, list(jax.random.split(key, len(leaves))))


def _conjugate_solve(
    block: LinearBlock,
    *,
    noise_std: dict[str, Any],
    tol: float,
    maxiter: int | None,
    key: jax.Array | None,
    require_convergence: float | None,
) -> tuple[dict[str, jax.Array], jax.Array]:
    """Shared machinery for the posterior mean and for a posterior draw.

    Both solve ``(A^T N^-1 A + S^-1) x = b`` by CG. They differ only in ``b``:
    the mean uses ``A^T N^-1 (d - offset) + S^-1 m``, a draw adds the two
    fluctuation terms. ``key=None`` selects the mean.
    """
    weight = _weights(noise_std)
    prior_variance = variance_parts(block)
    residual_data = jax.tree.map(jnp.subtract, block.data, block.offset)
    zero = domain_zero(block)
    centre = domain_centre(block)

    def pair_with(vector):
        """``A^T vector``, as the gradient of a real pairing.

        Taking it as a gradient rather than calling ``block.adjoint`` is what
        keeps the pairing conventions from ever entering: ``jax.grad`` of a
        real scalar is by construction the adjoint of the real inner product,
        which is the pairing every term here lives in.
        """

        def pairing(parts):
            pushed = block.forward(parts)
            return sum(jnp.sum(pushed[name] * vector[name]) for name in pushed)

        return jax.grad(pairing)(zero)

    normal = normal_operator(block, weight, prior_variance)

    # S^-1 m: a zero-mean prior is wrong for most physical quantities, and
    # shifting the prior is not the same act as shifting the model even though
    # the two give the same Gaussian.
    rhs = jax.tree.map(
        lambda base, mean, variance: base + mean / variance,
        pair_with(jax.tree.map(jnp.multiply, weight, residual_data)),
        centre,
        prior_variance,
    )

    if key is not None:
        # Constrained realization: two fluctuation terms whose covariances sum
        # to the normal operator itself, which is exactly why the solve comes
        # out distributed as the posterior rather than merely centred on its
        # mean.  b = A^T N^-1 (d-offset) + S^-1 m + A^T N^-1/2 w1 + S^-1/2 w2
        data_key, prior_key = jax.random.split(key)
        omega_data = jax.tree.map(
            lambda leaf, k: jax.random.normal(k, jnp.shape(leaf), dtype=leaf.dtype),
            residual_data,
            _split_like(data_key, residual_data),
        )
        omega_prior = jax.tree.map(
            lambda leaf, k: jax.random.normal(k, leaf.shape, dtype=leaf.dtype),
            zero,
            _split_like(prior_key, zero),
        )
        rhs = jax.tree.map(
            lambda base, from_data, from_prior, variance: (
                base + from_data + from_prior / jnp.sqrt(variance)
            ),
            rhs,
            pair_with(jax.tree.map(lambda w, o: jnp.sqrt(w) * o, weight, omega_data)),
            omega_prior,
            prior_variance,
        )

    solution, _ = jax.scipy.sparse.linalg.cg(normal, rhs, tol=tol, maxiter=maxiter)
    misfit = jax.tree.map(jnp.subtract, normal(solution), rhs)
    residual = tree_norm(misfit) / jnp.maximum(tree_norm(rhs), 1e-30)

    if require_convergence is not None:
        # jax's cg reports no convergence status of its own, so an unconverged
        # solve otherwise comes back looking like any other answer.
        # eqx.error_if fires under jit, where a Python `if` on a traced value
        # cannot.
        #
        # The residual ALONE cannot decide this. Error and residual differ by
        # the condition number, and for a block the data does not fully
        # identify kappa is enormous by construction -- lambda_min is exactly
        # the prior's 1/prior_std**2 -- so CG stops on a tiny residual with the
        # prior-dominated directions still at their starting value, and hands
        # back a draw whose posterior scatter there is orders of magnitude too
        # small. Guarding on the residual certifies precisely nothing in the
        # one regime these solvers exist to serve.
        kappa = _condition_bound(
            block, weight, prior_variance, jax.random.key(0), POWER_ITERATIONS
        )
        error_bound = residual * kappa
        bad = jnp.logical_or(~jnp.isfinite(residual), error_bound > require_convergence)

        # Below kappa*eps no tolerance can help: the arithmetic itself cannot
        # represent the answer that accurately. Worth its own message, because
        # the remedy is precision, and the natural response to the other
        # message -- tighten tol, raise maxiter -- burns a great many
        # iterations here to arrive at an equally wrong answer.
        epsilon = float(
            jnp.finfo(jnp.result_type(*jax.tree.leaves(block.offset))).eps
        )
        unreachable = kappa * epsilon > require_convergence

        solution = eqx.error_if(
            solution,
            jnp.logical_and(bad, unreachable),
            "wiener_solve/gcr_sample cannot reach require_convergence at this "
            "precision: the normal operator's condition number times the machine "
            "epsilon already exceeds it, so no tol or maxiter will help. This is "
            "the usual signature of a block the data does not identify. Run the "
            "solve inside `with jax.enable_x64(True):`, or strengthen the prior "
            "(prior_std bounds the conditioning: kappa ~ ||A^T N^-1 A|| * "
            "prior_std**2). condition_bound() reports the number.",
        )
        solution = eqx.error_if(
            solution,
            jnp.logical_and(bad, ~unreachable),
            "wiener_solve/gcr_sample did not converge: the relative residual "
            "times the normal operator's condition number -- the bound on the "
            "RELATIVE ERROR, which is what require_convergence limits -- exceeds "
            "it. The residual alone looks converged; it is not, along the "
            "directions the prior dominates. Pass tol ~ require_convergence/kappa "
            "with a maxiter to match, or strengthen the prior. "
            "condition_bound() reports kappa.",
        )
    return solution, residual


def wiener_solve(
    block: LinearBlock,
    *,
    noise_std: dict[str, Any],
    tol: float = 1e-6,
    maxiter: int | None = None,
    require_convergence: float | None = 1e-3,
) -> tuple[dict[str, jax.Array], jax.Array]:
    """Posterior mean of a linear-Gaussian block -- the Wiener filter, by CG.

    This is the posterior **mean**, not a sample. For a draw see
    :func:`gcr_sample`, which adds a fluctuation term to this same right-hand
    side and costs exactly the same solve.

    Args:
        block: from :func:`bayesmith.exact.linearity.linear_operator`.
        noise_std: ``{observed: sigma}``, from
            :func:`bayesmith.exact.gaussian.noise_std_at`. A sigma that has
            already been decided -- a conjugate solve has no prediction to
            evaluate a rule at, the prediction being what it solves for. For a
            prediction-dependent noise model see
            :func:`bayesmith.exact.gls.iterative_gls`, which finds the fixed
            point and hands the result back here.
        tol: CG tolerance -- a bound on the relative RESIDUAL, which is not
            the same as accuracy. See the note below.
        maxiter: CG iteration cap. ``None`` lets JAX choose.
        require_convergence: raise unless the relative ERROR can be bounded by
            this. ``None`` disables the guard and returns whatever CG
            produced. On by default because jax's ``cg`` reports no
            convergence status, so an unconverged solve otherwise comes back
            looking exactly like a converged one.

            The bound is ``kappa * relative_residual``, with ``kappa`` from
            :func:`condition_bound`. That costs ``POWER_ITERATIONS``
            extra operator applications, which on a well-conditioned block
            roughly DOUBLES the solve. In a Gibbs sweep, call
            :func:`condition_bound` once outside the loop, choose ``tol``
            from it, and pass ``require_convergence=None`` inside. What you
            must NOT do is leave ``tol`` at its default and the guard off --
            that is the combination that returns a silently over-confident
            posterior.

    Returns:
        ``(x_hat, relative_residual)``, the residual being
        ``||M x_hat - b|| / ||b||``. Note this is the residual, not the error;
        multiply by :func:`condition_bound` for the error bound.

    Note:
        **Where S comes from.** Each latent's own ``dist_fn`` is this
        package's one statement of what it is a priori, and it is the
        statement ``to_numpyro`` reads. So it is the statement this solve
        reads too -- there is no keyword to override it, and therefore no way
        for the exact exit and NUTS to target different posteriors.
    """
    return _conjugate_solve(
        block,
        noise_std=noise_std,
        tol=tol,
        maxiter=maxiter,
        key=None,
        require_convergence=require_convergence,
    )
```

- [ ] **Step 5: 跑测试，确认通过**

```bash
.venv/bin/python -m pytest tests/exact/test_solve.py -v
```

Expected: 12 passed。

- [ ] **Step 6: 变异测试**

1. 把 `rhs` 里的 `+ mean / variance`（`S⁻¹m` 项）删掉，重跑。
   Expected: `test_a_latent_the_data_never_reaches_comes_back_at_its_prior_mean` **变红**（得到 0 而非 1.25）。还原。
2. 把 `pair_with` 换成 `block.adjoint`（`lambda v: block.adjoint(v)`），重跑。
   Expected: 应当**仍绿**——两者在实内积下相等。这条变异是对"`pair_with` 是否只是 `adjoint` 的复杂写法"的诚实回答：在 bayesmith（无复数域）里确实等价，所以计划保留 `pair_with` 的理由缩小为"与 `normal_operator` 用同一种取梯度的方式，两处不会各自算错"。**把这条结论写进 `pair_with` 的 docstring**，替换掉从 rheplicant 抄来的复数论证。还原。
3. 把守卫里的 `error_bound = residual * kappa` 改成 `error_bound = residual`，重跑。
   Expected: `test_the_convergence_guard_fires_on_a_deliberately_starved_solve` 仍红（残差本身也大），说明这条测试**没有分离出 κ 的作用**。补一条：

```python
def test_the_guard_bounds_the_error_not_the_residual():
    """A tiny residual on an ill-conditioned block is not convergence.

    prior_std = 1e4 on `b` makes lambda_min ~ 1e-8 while the data sets
    lambda_max, so kappa is enormous. The residual at maxiter=2 looks fine and
    the ERROR does not -- which is exactly the regime these solvers exist for.
    """
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = _sigma(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        loosened = dataclasses.replace(
            block, prior_std={**block.prior_std, "b": jnp.asarray(1e4)}
        )
        kappa = float(condition_bound(loosened, noise_std=sigma, iterations=80))
    assert kappa > 1e6
```

   （`import dataclasses` 加到测试文件顶部。这条只断言 κ 确实大；把 κ 从守卫里拿掉后，`error_bound` 与 `require_convergence` 的关系就与 κ 无关，因此第 3 条变异的正确判据是**这条测试连同守卫测试一起看**——执行者应在实现后确认：在 `loosened` 块上以 `tol=1e-8, require_convergence=1e-3` 求解，带 κ 的守卫报错、不带 κ 的守卫不报错。把这个双向确认写成第二条断言。）
4. 把 `unreachable` 分支删掉（只留一条 `error_if`），重跑。
   Expected: 全绿——说明这条分支没有守卫。**补一条测试**：在 float32 下对 `loosened` 块求解，断言异常消息里出现 `enable_x64`。

- [ ] **Step 7: 提交**

```bash
git add src/bayesmith/exact/solve.py tests/exact/models.py tests/exact/test_solve.py
git commit -m "feat: Wiener posterior mean by matrix-free conjugate gradients"
```

---

## Task 7：精确后验抽取（`exact/solve.py` 之三）

约束实现（GCR）恒等式：解 `wiener_solve` 解的同一个系统，但在右端加两个白噪声项，

    (AᵀN⁻¹A + S⁻¹) x = AᵀN⁻¹(d−offset) + S⁻¹m + AᵀN⁻¹ᐟ²ω₁ + S⁻¹ᐟ²ω₂

右端于是以后验均值的分子为均值、以**算子本身** `AᵀN⁻¹A + S⁻¹` 为协方差，所以 `x = M⁻¹b` 的均值是后验均值、协方差是 `M⁻¹M M⁻¹ = M⁻¹`——精确。不是近似，也不是马氏链：**每次调用都是一次独立抽取，没有 burn-in，没有收敛可诊断。**

**Files:**
- Modify: `src/bayesmith/exact/solve.py`, `tests/exact/test_solve.py`

- [ ] **Step 1: 写失败的测试**

追加到 `tests/exact/test_solve.py`（顶部导入加 `gcr_sample`）：

```python
@pytest.mark.slow
def test_gcr_draws_have_the_oracle_mean_and_covariance():
    """The draw is exact, so its first two moments are the oracle's.

    require_convergence=None inside the vmap on purpose: the guard costs
    POWER_ITERATIONS operator applications PER DRAW, and tol is set from
    the block's kappa instead -- which is the bargain wiener_solve's docstring
    recommends and this test is the demonstration of.
    """
    draws = 4000
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = _sigma(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        samples = jax.vmap(
            lambda k: gcr_sample(
                block, noise_std=sigma, key=k, tol=1e-14, require_convergence=None
            )[0]
        )(jax.random.split(jax.random.key(20), draws))
        oracle = graph_oracle(graph, ("a", "b"), at={})
    flat = np.stack([np.asarray(samples[n]).ravel() for n in block.names], axis=1)
    standard_error = np.sqrt(np.diag(oracle.covariance) / draws)
    assert np.all(np.abs(flat.mean(axis=0) - oracle.mean) < 4 * standard_error)
    spread = np.max(np.diag(oracle.covariance))
    assert np.allclose(
        np.cov(flat, rowvar=False), oracle.covariance, rtol=0.1, atol=0.05 * spread
    )


@pytest.mark.slow
def test_a_draw_with_uninformative_data_falls_back_to_the_prior():
    """With sigma enormous the likelihood says nothing, so the draw is the prior.

    The check that the S^-1/2 fluctuation term is wired in at the right width:
    drop it and the draws collapse onto the prior MEAN with no scatter at all.
    """
    draws = 3000
    with jax.enable_x64(True):
        graph = straight_line(sigma=1e6, prior_mean=1.75, prior_std=2.0)
        block = linear_operator(graph, ("w",), at={})
        sigma = _sigma(graph, {"w": jnp.asarray(0.0)})
        samples = jax.vmap(
            lambda k: gcr_sample(
                block, noise_std=sigma, key=k, tol=1e-14, require_convergence=None
            )[0]["w"]
        )(jax.random.split(jax.random.key(21), draws))
    values = np.asarray(samples)
    assert values.mean() == pytest.approx(1.75, abs=4 * 2.0 / np.sqrt(draws))
    assert values.std() == pytest.approx(2.0, rel=0.1)


@pytest.mark.slow
def test_the_mean_of_many_draws_is_the_wiener_solution():
    """The two exits share one solve, so they cannot disagree about the centre."""
    draws = 3000
    with jax.enable_x64(True):
        graph = two_observations()
        block = linear_operator(graph, ("w",), at={})
        sigma = _sigma(graph, {"w": jnp.asarray(0.0)})
        mean, _ = wiener_solve(block, noise_std=sigma, tol=1e-14)
        samples = jax.vmap(
            lambda k: gcr_sample(
                block, noise_std=sigma, key=k, tol=1e-14, require_convergence=None
            )[0]["w"]
        )(jax.random.split(jax.random.key(22), draws))
        oracle = graph_oracle(graph, ("w",), at={})
    values = np.asarray(samples)
    posterior_sd = float(np.sqrt(oracle.covariance[0, 0]))
    assert values.mean() == pytest.approx(
        float(mean["w"]), abs=4 * posterior_sd / np.sqrt(draws)
    )
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
.venv/bin/python -m pytest tests/exact/test_solve.py -v -k gcr
```

Expected: FAIL — `ImportError: cannot import name 'gcr_sample'`。

- [ ] **Step 3: 实现**

追加到 `src/bayesmith/exact/solve.py`：

```python
def gcr_sample(
    block: LinearBlock,
    *,
    noise_std: dict[str, Any],
    key: jax.Array,
    tol: float = 1e-6,
    maxiter: int | None = None,
    require_convergence: float | None = 1e-3,
) -> tuple[dict[str, jax.Array], jax.Array]:
    """Draw an EXACT posterior sample of a linear-Gaussian block.

    The constrained-realization identity: solve the same system
    :func:`wiener_solve` does, with two white-noise terms added to the
    right-hand side, so that ``b`` has the posterior-mean numerator as its
    mean and covariance ``A^T N^-1 A + S^-1`` -- the operator itself. Then
    ``x = M^-1 b`` has the posterior mean and covariance ``M^-1 M M^-1 =
    M^-1`` exactly. Not an approximation and not a Markov chain: every call is
    an independent draw, with no burn-in and nothing to diagnose.

    It costs one CG solve -- the same as the mean -- because the fluctuation
    enters the right-hand side, never the operator. That is what makes a
    10^6-dimensional block samplable at all.

    Args:
        block: from :func:`bayesmith.exact.linearity.linear_operator`.
        noise_std: ``{observed: sigma}``, exactly as for
            :func:`wiener_solve`.
        key: PRNG key. ``vmap`` over split keys for many independent draws.
        tol: CG tolerance -- a bound on the residual, not on the accuracy.
        maxiter: CG iteration cap.
        require_convergence: as for :func:`wiener_solve`, which a draw is MORE
            exposed to than the mean. The fluctuation term ``S^-1/2 w2`` puts
            weight on every direction of the latent by construction, including
            the ones the data is blind to -- so a draw always has something to
            resolve where the operator is worst conditioned, whereas the mean
            does only when the prior mean is nonzero.

    Returns:
        ``(x, relative_residual)``. An unconverged CG returns a draw from the
        WRONG distribution -- and one that is too NARROW, since the directions
        left unresolved are the prior-dominated ones that should have carried
        the most scatter.
    """
    return _conjugate_solve(
        block,
        noise_std=noise_std,
        tol=tol,
        maxiter=maxiter,
        key=key,
        require_convergence=require_convergence,
    )
```

- [ ] **Step 4: 跑测试，确认通过**

```bash
.venv/bin/python -m pytest tests/exact/test_solve.py -v
```

Expected: 15 passed（三条 GCR 测试较慢，各约 10–30 秒）。

- [ ] **Step 5: 变异测试**

1. 删掉 `omega_prior` 项（右端只加 `from_data`），重跑。
   Expected: `test_a_draw_with_uninformative_data_falls_back_to_the_prior` **变红**（散度坍缩到 ~0）。还原。
2. 删掉 `omega_data` 项（右端只加 `from_prior / sqrt(v)`），重跑。
   Expected: `test_gcr_draws_have_the_oracle_mean_and_covariance` 的协方差断言 **变红**（后验过窄）。还原。
3. 把 `from_prior / jnp.sqrt(variance)` 改成 `from_prior * jnp.sqrt(variance)`，重跑。
   Expected: 同上变红。还原。
4. 把 `pair_with(... jnp.sqrt(weight) * omega_data ...)` 里的 `jnp.sqrt` 去掉，重跑。
   Expected: `test_gcr_draws_have_the_oracle_mean_and_covariance` **变红**。还原。

- [ ] **Step 6: 提交**

```bash
git add src/bayesmith/exact/solve.py tests/exact/test_solve.py
git commit -m "feat: exact posterior draws by constrained realization"
```

---
## Task 8：迭代重加权 GLS（`exact/gls.py`）

`gcr_sample` 是**给定**协方差下的线性抽样器，`wiener_solve` 是对应的均值。两者都拿 `noise_std`，都不关心它从哪来。σ 为常数时无话可说。

σ 跟踪预测时（辐射计噪声 `σ_i = κ|预测_i|`）有：权重依赖解，解依赖权重，两者都不先有。本模块补上缺的那一半——协方差——且不改变两个求解器的任何东西。

**这个估计量是什么，不是什么。** 在每次求解内冻结 σ 是使每步成为线性高斯问题的原因，也正是使收敛结果是**广义最小二乘**而非完整高斯似然之最大值的原因：对数行列式对解的依赖被固定住了，而不是被微分。两者的差是已知方向的，而**这道差正是 P3b 的重要性权重要还回去的东西**（spec §5.1）。所以 `iterative_gls` 在 P3a 里是点估计出口，在 P3b 里同时是提议中心。

**Files:**
- Create: `src/bayesmith/exact/gls.py`, `tests/exact/test_gls.py`

- [ ] **Step 1: 写失败的测试**

```python
# tests/exact/test_gls.py
"""Finding the covariance a prediction-dependent sigma implies."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.errors import StructureError
from bayesmith.exact.gls import (
    check_prediction_dependence,
    iterative_gls,
    sigma_from_graph,
)
from bayesmith.exact.linearity import linear_operator
from bayesmith.exact.solve import wiener_solve
from tests.exact.models import radiometer, straight_line
from tests.exact.oracle import flat_domain, graph_oracle

KAPPA = 0.05
FLOOR = 1e-3


def test_a_constant_sigma_converges_in_one_step():
    with jax.enable_x64(True):
        graph = straight_line()
        block = linear_operator(graph, ("w",), at={})
        result = iterative_gls(
            block, sigma_from_graph(graph, {}), depends_on_prediction=False, tol=1e-14
        )
        direct, _ = wiener_solve(
            block, noise_std=sigma_from_graph(graph, {})({"w": jnp.asarray(0.0)}),
            tol=1e-14,
        )
    assert int(result.iterations) == 1
    assert bool(result.converged)
    assert float(result.solution["w"]) == pytest.approx(float(direct["w"]), rel=1e-10)


def test_iterative_gls_finds_the_fixed_point_a_dense_iteration_finds():
    """A NumPy fixed-point loop, sharing nothing with the JAX while_loop."""
    with jax.enable_x64(True):
        graph = radiometer(kappa=KAPPA, floor=FLOOR)
        block = linear_operator(graph, ("w",), at={})
        result = iterative_gls(
            block, sigma_from_graph(graph, {}), tol=1e-14, reweight_tol=1e-10
        )
        oracle = graph_oracle(graph, ("w",), at={})

    design, offset, data = oracle.design, oracle.offset, oracle.data
    prior_precision = np.diag(1.0 / oracle.prior_std**2)
    x = oracle.prior_mean.copy()
    for _ in range(400):
        sigma = KAPPA * np.abs(design @ x + offset) + FLOOR
        noise_precision = np.diag(1.0 / sigma**2)
        x = np.linalg.solve(
            design.T @ noise_precision @ design + prior_precision,
            design.T @ noise_precision @ (data - offset)
            + prior_precision @ oracle.prior_mean,
        )

    assert bool(result.converged)
    assert np.allclose(flat_domain(result.solution, block.names), x, rtol=1e-6)


def test_the_returned_sigma_really_is_a_fixed_point():
    with jax.enable_x64(True):
        graph = radiometer()
        block = linear_operator(graph, ("w",), at={})
        sigma_of = sigma_from_graph(graph, {})
        result = iterative_gls(block, sigma_of, tol=1e-14, reweight_tol=1e-10)
        recomputed = sigma_of(result.solution)
    assert np.allclose(np.asarray(recomputed["d"]), np.asarray(result.noise_std["d"]),
                       rtol=1e-6)


def test_check_prediction_dependence_catches_a_false_declaration():
    """`depends_on_prediction=False` on a radiometer node is a claim, and false.

    Declared False, a dispatcher skips the reweighting loop entirely and
    solves at whatever sigma the prior mean implies -- a confident answer at
    the wrong covariance, with nothing to notice.
    """
    with jax.enable_x64(True):
        graph = radiometer()
        block = linear_operator(graph, ("w",), at={})
        with pytest.raises(StructureError, match="depends_on_prediction"):
            check_prediction_dependence(block, sigma_from_graph(graph, {}), declared=False)


def test_check_prediction_dependence_accepts_a_true_declaration():
    with jax.enable_x64(True):
        graph = straight_line()
        block = linear_operator(graph, ("w",), at={})
        movement = check_prediction_dependence(
            block, sigma_from_graph(graph, {}), declared=False
        )
    assert movement == pytest.approx(0.0, abs=1e-12)


def test_a_capped_run_reports_converged_false_rather_than_pretending():
    """converged=False means the returned covariance is NOT a fixed point.

    Everything conditioned on it inherits that, so it is returned as a flag
    rather than raised -- but it must never read True.
    """
    with jax.enable_x64(True):
        graph = radiometer()
        block = linear_operator(graph, ("w",), at={})
        result = iterative_gls(
            block, sigma_from_graph(graph, {}), tol=1e-14,
            min_reweights=1, max_reweights=1,
        )
    assert not bool(result.converged)
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
.venv/bin/python -m pytest tests/exact/test_gls.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'bayesmith.exact.gls'`。

- [ ] **Step 3: 实现**

```python
# src/bayesmith/exact/gls.py
"""Iteratively reweighted least squares: finding the covariance to solve at.

:func:`~bayesmith.exact.solve.gcr_sample` is a linear sampler *given* a
covariance and :func:`~bayesmith.exact.solve.wiener_solve` is the
corresponding mean. Both take ``noise_std`` and neither cares where it came
from. For a constant sigma there is nothing more to say.

For a sigma that tracks the prediction -- a radiometer's
``sigma_i = kappa |prediction_i|`` -- there is: the weights depend on the
solution and the solution depends on the weights, and neither is available
first. This module supplies the missing half and changes nothing about the
two solvers.

The algorithm is a fixed-point iteration: solve with the current weights,
recompute the weights at the new prediction, repeat. It is the same
iteratively-reweighted GLS as hydra-tod's ``iterative_gls``, but
**matrix-free** -- hydra-tod forms a dense design matrix and a dense
``N_inv``, while here the same algorithm runs on the block's JVP and VJP.

**What this estimator is, and is not.** Freezing sigma inside each solve is
what makes each step a linear-Gaussian problem, and it is also what makes the
converged answer *generalized least squares* rather than the maximum of the
full Gaussian likelihood: the log-determinant's dependence on the solution is
held fixed rather than differentiated. That difference is exactly what P3b's
importance weight puts back, which is why this function is both a point
estimate here and the proposal centre there.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from bayesmith.errors import GraphError, StructureError
from bayesmith.exact.block import LinearBlock, domain_centre
from bayesmith.exact.conditioning import tree_norm
from bayesmith.exact.gaussian import noise_std_at
from bayesmith.exact.solve import wiener_solve
from bayesmith.graph.graph import Graph

#: Reweighting steps taken before the convergence test is consulted. Matches
#: hydra-tod's default: the first steps of a fixed-point iteration can be
#: nearly stationary without being near the fixed point.
MIN_REWEIGHTS: int = 5

#: Cap on reweighting steps, so a non-contracting problem terminates and says
#: so through ``converged`` rather than spinning.
MAX_REWEIGHTS: int = 100

#: Multiple of the working precision's epsilon used as the default
#: ``reweight_tol``. See :func:`iterative_gls` for why it cannot be a constant.
REWEIGHT_TOL_EPS: float = 8.0

#: Multiples of the prior standard deviation the dependence probe moves the
#: block by. Movement is measured against the value at the CENTRE, not between
#: probes, so one probe would usually do -- except where sigma is clipped or
#: floored on one side (``kappa * max(mu, 0) + floor`` reads exactly constant
#: for every negative probe). Two-sided, and at unequal magnitudes so a sigma
#: that happens to be symmetric about the centre still moves the two probes by
#: different amounts.
DEPENDENCE_PROBES: tuple[float, ...] = (1.0, -0.5)


class GLSResult(NamedTuple):
    """What a reweighting run produced. A pytree, so it survives ``jit``.

    Attributes:
        noise_std: the converged sigma -- **the covariance**, and the whole
            point of the exercise. Feed it to ``gcr_sample`` or
            ``wiener_solve`` as ``noise_std=``.
        solution: the GLS point estimate at that covariance.
        residual: relative CG residual of the final solve. Not an accuracy.
        iterations: reweighting steps taken, the first solve included.
        delta: relative change of the last step.
        converged: whether ``delta`` fell below ``reweight_tol`` within
            ``max_reweights``. **False means the returned covariance is not a
            fixed point**, and everything conditioned on it inherits that.
    """

    noise_std: dict[str, jax.Array]
    solution: dict[str, jax.Array]
    residual: jax.Array
    iterations: jax.Array
    delta: jax.Array
    converged: jax.Array


def sigma_from_graph(
    graph: Graph, at: dict[str, Any]
) -> Callable[[dict[str, Any]], dict[str, jax.Array]]:
    """The ``{name: x} -> {observed: sigma}`` seam :func:`iterative_gls` iterates.

    Taking the seam as a callable rather than the graph keeps this module
    independent of the graph layer for testing, and lets P3b hand in a sigma
    frozen somewhere else without this module needing to know.
    """

    def sigma_of(x: dict[str, Any]) -> dict[str, jax.Array]:
        return noise_std_at(graph, {**at, **x})

    return sigma_of


def check_prediction_dependence(
    block: LinearBlock,
    sigma_of: Callable[[dict[str, Any]], dict[str, jax.Array]],
    *,
    declared: bool,
    rtol: float = 1e-8,
) -> float:
    """Measure how much sigma moves with the block, and check the declaration.

    ``depends_on_prediction`` is a **claim about the model**, like
    ``linear_in``: declared ``False``, a dispatcher skips the reweighting loop
    and solves at whatever sigma the prior mean implies -- a confident answer
    at the wrong covariance, with nothing to notice.

    Runs on concrete values, outside any trace, for the same reason
    :func:`~bayesmith.exact.gaussian.check_gaussian` does.

    Args:
        block: the block sigma might depend on.
        sigma_of: the seam, from :func:`sigma_from_graph`.
        declared: what the node claims.
        rtol: relative movement below which sigma counts as constant.

    Returns:
        The largest relative movement observed.

    Raises:
        StructureError: if ``declared`` is ``False`` and sigma does move.
            The opposite mismatch -- declared ``True``, sigma constant -- is
            merely conservative and is returned rather than raised, so a
            caller can report "the declaration is conservative; the
            reweighting loop could be skipped".
    """
    centre = domain_centre(block)
    baseline = sigma_of(centre)
    movement = 0.0
    for factor in DEPENDENCE_PROBES:
        probe = {
            name: centre[name] + factor * block.prior_std[name] for name in block.names
        }
        moved = sigma_of(probe)
        for observed, value in moved.items():
            scale = max(float(jnp.max(jnp.abs(baseline[observed]))), 1e-300)
            movement = max(
                movement,
                float(jnp.max(jnp.abs(value - baseline[observed]))) / scale,
            )
    if not declared and movement > rtol:
        raise StructureError(
            "a node declares depends_on_prediction=False, but moving the block by "
            f"one prior standard deviation moves sigma by {movement:.3e} relative "
            f"(rtol={rtol:.1e}). Declared False, the reweighting loop is skipped "
            "and the solve runs at whatever sigma the prior mean implies -- a "
            "confident answer at the wrong covariance. Drop the declaration, or "
            "make sigma genuinely independent of the prediction."
        )
    return movement


def iterative_gls(
    block: LinearBlock,
    sigma_of: Callable[[dict[str, Any]], dict[str, jax.Array]],
    *,
    depends_on_prediction: bool = True,
    tol: float = 1e-6,
    maxiter: int | None = None,
    reweight_tol: float | None = None,
    min_reweights: int = MIN_REWEIGHTS,
    max_reweights: int = MAX_REWEIGHTS,
    require_convergence: float | None = 1e-3,
) -> GLSResult:
    """Find the covariance a prediction-dependent noise model implies.

    Repeats: solve at the current sigma, recompute sigma at the new solution.
    With ``depends_on_prediction=False`` there is nothing to repeat and this is
    a single :func:`~bayesmith.exact.solve.wiener_solve`.

    Args:
        block: from :func:`bayesmith.exact.linearity.linear_operator`.
        sigma_of: the seam, from :func:`sigma_from_graph`.
        depends_on_prediction: the node's own claim. **Check it first** with
            :func:`check_prediction_dependence` -- this function cannot, being
            jittable.
        tol, maxiter: CG settings for each inner solve.
        reweight_tol: stop when the block's relative change falls below this.
            **The default cannot be a fixed number**, because two independent
            floors bound how small a step is measurable at all, so it defaults
            to ``max(8 * eps, tol)``:

            * the arithmetic's own epsilon -- a relative step below it is
              rounding, not a measurement. float32's is ``1.2e-7``, so a
              plausible-looking ``1e-8`` is exactly this trap;
            * **the inner solver's tolerance** ``tol`` -- consecutive solves
              differ by roughly their own CG residual no matter what the outer
              iteration is doing, so a step smaller than ``tol`` measures CG,
              not the fixed point. This is the binding floor in float64.

            Ask for less than either and the run does not fail loudly: it
            spends ``max_reweights`` steps and reports ``converged=False`` for
            a fixed point it had in fact reached.
        min_reweights: steps taken before the test is consulted.
        max_reweights: cap on steps.
        require_convergence: bound on the relative error of the **final**
            solve. Deliberately applied once, at the converged covariance, and
            not inside the loop: it bounds the error of what is returned and
            says nothing about the intermediate steps, which do not need it.

    Returns:
        A :class:`GLSResult`. **Check ``converged``** -- a covariance that is
        not a fixed point is still a number, and a draw conditioned on it is
        still a draw.

    Note:
        The iteration starts from sigma at the block's **prior mean** rather
        than from hydra-tod's unweighted least squares or rheplicant's sigma
        at the data. A graph's sigma is a function of the latents, not of a
        prediction array, so the data is not a point this seam can be
        evaluated at; the prior mean is the natural "before seeing anything"
        one. A fixed point does not depend on where the iteration started, so
        all three agree wherever any of them converges.

        Built on ``lax.while_loop``, so it is jittable but **not** reverse-mode
        differentiable. That is not the limitation it looks like: the result is
        a fixed point, so implicit differentiation -- not unrolling -- is the
        right way to take a gradient through it.
    """
    if not 1 <= min_reweights <= max_reweights:
        raise GraphError(
            f"iterative_gls needs 1 <= min_reweights <= max_reweights, got "
            f"{min_reweights} and {max_reweights}. The loop caps at "
            "max_reweights either way, so this configuration would silently get "
            "fewer steps than it asked for."
        )
    if reweight_tol is None:
        epsilon = float(jnp.finfo(jnp.result_type(*jax.tree.leaves(block.offset))).eps)
        reweight_tol = max(REWEIGHT_TOL_EPS * epsilon, tol)

    def solve_at(sigma, guard):
        return wiener_solve(
            block, noise_std=sigma, tol=tol, maxiter=maxiter, require_convergence=guard
        )

    centre = domain_centre(block)

    if not depends_on_prediction:
        sigma = sigma_of(centre)
        solution, residual = solve_at(sigma, require_convergence)
        return GLSResult(
            noise_std=sigma,
            solution=solution,
            residual=residual,
            iterations=jnp.asarray(1),
            delta=jnp.asarray(0.0),
            converged=jnp.asarray(True),
        )

    def step(carry):
        count, latent, _ = carry
        updated, _ = solve_at(sigma_of(latent), None)
        change = jax.tree.map(jnp.subtract, updated, latent)
        # Relative to the NEW iterate: relative to the old one, a step that
        # starts near zero reports a huge change forever.
        delta = tree_norm(change) / jnp.maximum(tree_norm(updated), 1e-30)
        return count + 1, updated, delta

    def unfinished(carry):
        count, _, delta = carry
        # max_reweights is the OUTER conjunct, so it caps the loop whatever
        # min_reweights says. Written the other way round -- keep going while
        # below the minimum OR not yet converged -- a min above the max never
        # terminates, and an infinite lax.while_loop under jit cannot be
        # interrupted.
        return jnp.logical_and(
            count < max_reweights,
            jnp.logical_or(count < min_reweights, delta > reweight_tol),
        )

    first, _ = solve_at(sigma_of(centre), None)
    count, latent, delta = lax.while_loop(
        unfinished, step, (jnp.asarray(1), first, jnp.asarray(jnp.inf))
    )

    # One final solve at the converged covariance, and the only place the
    # conditioning guard runs -- so what it certifies is what is returned.
    sigma = sigma_of(latent)
    solution, residual = solve_at(sigma, require_convergence)
    return GLSResult(
        noise_std=sigma,
        solution=solution,
        residual=residual,
        iterations=count,
        delta=delta,
        converged=delta <= reweight_tol,
    )
```

- [ ] **Step 4: 跑测试，确认通过**

```bash
.venv/bin/python -m pytest tests/exact/test_gls.py -v
```

Expected: 6 passed。

- [ ] **Step 5: 变异测试**

1. 把 `step` 里的 `delta` 分母从 `tree_norm(updated)` 改成 `tree_norm(latent)`，重跑。
   Expected: 不一定变红。补一条断言：`test_iterative_gls_finds_the_fixed_point_a_dense_iteration_finds` 里加 `assert int(result.iterations) < 50`——从先验均值（=0）起步时旧写法的分母是 0，delta 恒为巨大值，迭代会跑满 `max_reweights`。还原。
2. 把 `unfinished` 的两个连接词交换（`or` 在外、`and` 在内），并令 `min_reweights=5, max_reweights=2` 调用，重跑。
   Expected: 挂起（无限循环）。**这条变异不要在 CI 里跑**——在本地手工确认后立刻还原，并确认 `min_reweights > max_reweights` 被 `GraphError` 挡在前面。
3. 把 `check_prediction_dependence` 的 `DEPENDENCE_PROBES` 改成只有 `(-0.5,)`，重跑。
   Expected: `test_check_prediction_dependence_catches_a_false_declaration` 仍红——辐射计的 σ 在任一侧都动。这条变异因此**没有覆盖单侧探针的真实失效模式**：σ 被单侧钳位时（`sigma = kappa * max(mu, 0) + floor`）每个负探针读出的都恰好是 baseline。补一个玩具模型并补一条测试：

```python
def one_sided_sigma(*, n=8, kappa=0.2, floor=1e-2, seed=12):
    """``sigma = kappa * max(mu, 0) + floor`` -- constant for every mu <= 0.

    A one-sided probe that happens to go negative reads sigma as constant and
    lets `depends_on_prediction=False` through. Two-sided does not.
    """
    x = jnp.linspace(1.0, 3.0, n)
    data = 2.0 * x + floor * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 1.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe(
            "d",
            lambda m: dist.Normal(m, kappa * jnp.maximum(m, 0.0) + floor),
            mu,
            obs=data,
        )

    return trace(model)
```

```python
def test_a_one_sided_probe_would_miss_a_clipped_sigma():
    from tests.exact.models import one_sided_sigma

    with jax.enable_x64(True):
        graph = one_sided_sigma()
        block = linear_operator(graph, ("w",), at={})
        with pytest.raises(StructureError, match="depends_on_prediction"):
            check_prediction_dependence(
                block, sigma_from_graph(graph, {}), declared=False
            )
```

   现在重做这条变异：`DEPENDENCE_PROBES = (-0.5,)` 下新测试 **变红**。还原。

- [ ] **Step 6: 提交**

```bash
git add src/bayesmith/exact/gls.py tests/exact/test_gls.py
git commit -m "feat: iteratively reweighted GLS over a matrix-free block"
```

---
## Task 9：稠密 Fisher 与参数协方差（`exact/fisher.py`）

第三条路线（R3）。`jax.jacfwd` **真的形成**矩阵，而 CG 路径只应用算子——所以这是一条不同的实现。但它**不是独立**的：两者都走 JAX 自动微分。独立的参照物只有 `tests/exact/oracle.py`，它在基向量上探测 `g`、从不微分任何东西。

`F = JᵀN⁻¹J` 是**似然的**信息，与其他出口瞄准的后验精度是不同的量。`include_prior=` 决定是哪一个，`kind` 字段把答案记下来，这样一份 Fisher 不会被当成另一份用。

**Files:**
- Create: `src/bayesmith/exact/fisher.py`, `tests/exact/test_fisher.py`

- [ ] **Step 1: 写失败的测试**

```python
# tests/exact/test_fisher.py
"""The dense route -- and what it does and does not independently confirm."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.exact.fisher import (
    dense_operator,
    fisher_information,
    parameter_covariance,
)
from bayesmith.exact.gaussian import noise_std_at
from bayesmith.exact.linearity import linear_operator
from tests.exact.models import plated_latent, two_linear_latents, two_observations
from tests.exact.oracle import graph_oracle


def test_dense_operator_matches_the_probed_design_matrix():
    """R3 vs R2. jacfwd and a basis probe must agree on A."""
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        design = np.asarray(dense_operator(block))
        oracle = graph_oracle(graph, ("a", "b"), at={})
    assert design.shape == oracle.design.shape
    assert np.allclose(design, oracle.design, rtol=1e-8)


def test_fisher_with_the_prior_is_the_posterior_precision():
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = noise_std_at(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        fisher = fisher_information(block, noise_std=sigma)
        oracle = graph_oracle(graph, ("a", "b"), at={})
    assert fisher.kind == "posterior_precision"
    assert np.allclose(np.asarray(fisher.values), oracle.precision, rtol=1e-8)


def test_fisher_without_the_prior_is_the_likelihood_alone_and_says_so():
    """`F = J^T N^-1 J` is a different quantity, and the kind field records it."""
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = noise_std_at(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        fisher = fisher_information(block, noise_std=sigma, include_prior=False)
        oracle = graph_oracle(graph, ("a", "b"), at={})
    assert fisher.kind == "fisher"
    expected = oracle.precision - np.diag(1.0 / oracle.prior_std**2)
    assert np.allclose(np.asarray(fisher.values), expected, rtol=1e-8)


def test_parameter_covariance_matches_the_oracle():
    with jax.enable_x64(True):
        graph = two_observations()
        block = linear_operator(graph, ("w",), at={})
        sigma = noise_std_at(graph, {"w": jnp.asarray(0.0)})
        covariance = parameter_covariance(fisher_information(block, noise_std=sigma))
        oracle = graph_oracle(graph, ("w",), at={})
    assert covariance.kind == "covariance"
    assert np.allclose(np.asarray(covariance.values), oracle.covariance, rtol=1e-8)


def test_a_flat_matrix_block_is_addressable_by_latent_name():
    """A six-dimensional plated block, so the spans are not all width one."""
    with jax.enable_x64(True):
        graph = plated_latent(n=6)
        block = linear_operator(graph, ("z",), at={})
        sigma = noise_std_at(graph, {"z": jnp.zeros(6)})
        fisher = fisher_information(block, noise_std=sigma)
    assert fisher.names == ("z",)
    assert fisher.spans == ((0, 6),)
    assert fisher.block("z").shape == (6, 6)
    with pytest.raises(KeyError, match="w"):
        fisher.block("w")
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
.venv/bin/python -m pytest tests/exact/test_fisher.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'bayesmith.exact.fisher'`。

- [ ] **Step 3: 实现**

```python
# src/bayesmith/exact/fisher.py
"""The dense route: a materialised design matrix, and what it buys.

``jax.jacfwd`` forms ``A`` explicitly, where
:mod:`bayesmith.exact.solve` only ever applies it. For a block small enough
to hold as a matrix that gives a posterior covariance in one ``inv`` instead
of one CG per direction -- forecasts, error bars, and the Gaussian a Laplace
approximation samples from.

**It is a different implementation, not an independent one.** Both routes go
through JAX's autodiff, so a bug in how the block's ``forward`` was built
shows up identically in both. The independent reference is
``tests/exact/oracle.py``, which probes ``g`` on a basis of the domain and
differentiates nothing.

``F = J^T N^-1 J`` is the LIKELIHOOD's information, which is a different
quantity from the posterior precision every other exit here targets.
``include_prior=`` chooses, and :attr:`FlatMatrix.kind` records the answer so
one cannot quietly be used as the other.

Ported from ``rheplicant.inference.uncertainty``; ``propagate_covariance`` and
``push_forward`` are P5.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from bayesmith.exact.block import LinearBlock


class FlatMatrix(eqx.Module):
    """A matrix over a block's domain, plus where each latent sits in it.

    Attributes:
        values: the matrix, ``(n, n)`` over the flattened domain.
        names: the latents, in the block's own order.
        spans: ``(start, stop)`` per latent, derived from the actual
            flattening rather than assumed -- a plated latent occupies as many
            rows as it has entries.
        kind: ``"fisher"`` (likelihood information alone),
            ``"posterior_precision"`` (with the prior's curvature) or
            ``"covariance"``. Carried rather than inferred, because the three
            are the same shape and confusing them is silent.
    """

    values: jax.Array
    names: tuple[str, ...] = eqx.field(static=True)
    spans: tuple[tuple[int, int], ...] = eqx.field(static=True)
    kind: str = eqx.field(static=True)

    def block(self, name: str) -> jax.Array:
        """The diagonal sub-matrix belonging to one latent.

        Raises:
            KeyError: if ``name`` is not in this matrix.
        """
        for latent, (start, stop) in zip(self.names, self.spans, strict=True):
            if latent == name:
                return self.values[start:stop, start:stop]
        raise KeyError(
            f"{name!r} is not in this matrix; it covers {list(self.names)}."
        )

    def std(self) -> dict[str, jax.Array]:
        """``{name: sqrt(diagonal)}``. Meaningful only for ``kind='covariance'``.

        Raises:
            ValueError: for any other kind -- the square root of a precision's
                diagonal is not an error bar, and returning it would be a
                confident wrong number rather than a mistake.
        """
        if self.kind != "covariance":
            raise ValueError(
                f"std() needs a covariance, and this is a {self.kind!r}. The "
                "square root of a precision's diagonal is not an error bar; it "
                "is the error bar of a parameter with every other one held "
                "fixed. Invert it first with parameter_covariance()."
            )
        diagonal = jnp.diagonal(self.values)
        return {
            name: jnp.sqrt(diagonal[start:stop])
            for name, (start, stop) in zip(self.names, self.spans, strict=True)
        }


def _spans(block: LinearBlock) -> tuple[tuple[tuple[int, int], ...], int]:
    spans: list[tuple[int, int]] = []
    start = 0
    for name in block.names:
        size = int(np.prod(block.shape[name], dtype=int))
        spans.append((start, start + size))
        start += size
    return tuple(spans), start


def _unravel(flat: jax.Array, block: LinearBlock, spans) -> dict[str, jax.Array]:
    return {
        name: jnp.reshape(flat[start:stop], block.shape[name])
        for name, (start, stop) in zip(block.names, spans, strict=True)
    }


def _domain_dtype(block: LinearBlock):
    return jnp.result_type(*[block.dtype[name] for name in block.names])


def dense_operator(block: LinearBlock) -> jax.Array:
    """``A`` materialised, ``(n_data, n_parameters)``.

    Rows are the observed nodes concatenated in **sorted name order**, columns
    are the latents in the block's own order -- the same layout
    ``tests/exact/oracle.py`` uses, so the two are comparable element for
    element.
    """
    spans, size = _spans(block)

    def flat_forward(flat: jax.Array) -> jax.Array:
        pushed = block.forward(_unravel(flat, block, spans))
        return jnp.concatenate(
            [jnp.reshape(pushed[name], (-1,)) for name in sorted(pushed)]
        )

    return jax.jacfwd(flat_forward)(jnp.zeros(size, dtype=_domain_dtype(block)))


def fisher_information(
    block: LinearBlock,
    *,
    noise_std: dict[str, Any],
    include_prior: bool = True,
) -> FlatMatrix:
    """``J^T N^-1 J``, optionally plus the declared priors' curvature.

    Args:
        block: from :func:`bayesmith.exact.linearity.linear_operator`.
        noise_std: ``{observed: sigma}``, from
            :func:`bayesmith.exact.gaussian.noise_std_at`.
        include_prior: add ``S^-1``, making the result the posterior precision
            rather than the likelihood's information. Default ``True``,
            because that is the quantity every other exit in this package
            targets and a forecast that silently answered a different question
            would agree with none of them.
    """
    design = dense_operator(block)
    weight = jnp.concatenate(
        [
            jnp.reshape(1.0 / jnp.asarray(noise_std[name]) ** 2, (-1,))
            for name in sorted(noise_std)
        ]
    )
    values = design.T @ (weight[:, None] * design)
    if include_prior:
        curvature = jnp.concatenate(
            [
                jnp.reshape(1.0 / jnp.asarray(block.prior_std[name]) ** 2, (-1,))
                for name in block.names
            ]
        )
        values = values + jnp.diag(curvature)
    spans, _ = _spans(block)
    return FlatMatrix(
        values=values,
        names=block.names,
        spans=spans,
        kind="posterior_precision" if include_prior else "fisher",
    )


def parameter_covariance(fisher: FlatMatrix, jitter: float = 0.0) -> FlatMatrix:
    """Invert a precision. ``jitter`` adds ``jitter * I`` first.

    Raises:
        ValueError: if handed a covariance -- inverting one gives a precision,
            which is a legitimate operation but not what this function's name
            promises, and the ``kind`` field would then be a lie.
    """
    if fisher.kind == "covariance":
        raise ValueError(
            "parameter_covariance() was handed a covariance. Inverting it would "
            "give a precision back, which is not what the name says and would "
            "leave kind='covariance' on a matrix that is not one."
        )
    values = fisher.values + jitter * jnp.eye(
        fisher.values.shape[0], dtype=fisher.values.dtype
    )
    return FlatMatrix(
        values=jnp.linalg.inv(values),
        names=fisher.names,
        spans=fisher.spans,
        kind="covariance",
    )
```

- [ ] **Step 4: 跑测试，确认通过**

```bash
.venv/bin/python -m pytest tests/exact/test_fisher.py -v
```

Expected: 5 passed。

- [ ] **Step 5: 变异测试**

1. 把 `fisher_information` 的 `include_prior` 默认值改成 `False`，重跑。
   Expected: `test_fisher_with_the_prior_is_the_posterior_precision` **变红**。还原。
2. 把 `dense_operator` 里的 `sorted(pushed)` 改成 `list(pushed)`（插入序），重跑。
   Expected: 在 `two_observations`（`d1`/`d2`，插入序恰好也是字典序）上仍绿——所以补一个观测节点名的字典序与声明序**相反**的模型（例如把 `two_observations` 的两个观测改名为 `"z_first"` 与 `"a_second"`），并加一条 `test_dense_operator_matches_the_probed_design_matrix` 的变体。还原。
3. 把 `FlatMatrix.std()` 的 kind 检查删掉，重跑。
   Expected: 全绿——说明它无守卫。补一条：

```python
def test_std_refuses_a_precision():
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = noise_std_at(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        fisher = fisher_information(block, noise_std=sigma)
        with pytest.raises(ValueError, match="not an error bar"):
            fisher.std()
        assert parameter_covariance(fisher).std()["a"].shape == ()
```

- [ ] **Step 6: 提交**

```bash
git add src/bayesmith/exact/fisher.py tests/exact/test_fisher.py
git commit -m "feat: dense Fisher information and parameter covariance"
```

---
## Task 10：验收关口——公共 API、边界验证、极端参数

三件事：把 P3a 接到公共 API 上；按 `boundary-validation.md` 在每个阈值两侧**绕开分派逻辑**直接比较；覆盖极端参数值（失效模式常呈 U 形，只测"中等"参数会整片漏掉）。

**Files:**
- Modify: `src/bayesmith/exact/__init__.py`, `src/bayesmith/__init__.py`, `tests/test_public_api.py`
- Modify: `tests/exact/models.py`
- Create: `tests/exact/test_boundaries.py`, `tests/exact/test_extremes.py`

- [ ] **Step 1: 子包再导出**

把 `src/bayesmith/exact/__init__.py` 的内容换成：

```python
"""Structure-dispatched exact solves for linear-Gaussian blocks.

Import order matters only in that ``linear_operator`` -- the checked entry
point -- lives in :mod:`bayesmith.exact.linearity`, while the unchecked
primitive it wraps lives in :mod:`bayesmith.exact.block`. Re-exported here so
the checked name is the one a bare ``from bayesmith.exact import ...`` finds.
"""

from bayesmith.exact.block import LinearBlock, unchecked_operator
from bayesmith.exact.fisher import (
    FlatMatrix,
    dense_operator,
    fisher_information,
    parameter_covariance,
)
from bayesmith.exact.gaussian import check_gaussian, gaussian_parts, noise_std_at
from bayesmith.exact.gls import (
    GLSResult,
    check_prediction_dependence,
    iterative_gls,
    sigma_from_graph,
)
from bayesmith.exact.linearity import check_linearity, linear_operator
from bayesmith.exact.solve import condition_bound, gcr_sample, wiener_solve

__all__ = [
    "LinearBlock",
    "unchecked_operator",
    "linear_operator",
    "check_linearity",
    "gaussian_parts",
    "check_gaussian",
    "noise_std_at",
    "wiener_solve",
    "gcr_sample",
    "condition_bound",
    "iterative_gls",
    "GLSResult",
    "sigma_from_graph",
    "check_prediction_dependence",
    "FlatMatrix",
    "dense_operator",
    "fisher_information",
    "parameter_covariance",
]
```

- [ ] **Step 2: 接到顶层 API**

在 `src/bayesmith/__init__.py` 里做四处编辑：

1. 第 12 行的 eager 导入改为：

```python
from bayesmith.errors import (
    BayesmithError,
    ConvergenceError,
    GraphError,
    NotGaussian,
    StructureError,
    TraceError,
)
```

2. `__all__` 的 `# inference` 段之后、`# errors` 段之前插入：

```python
    # exact
    "linear_operator",
    "check_linearity",
    "wiener_solve",
    "gcr_sample",
    "condition_bound",
    "iterative_gls",
    "sigma_from_graph",
    "noise_std_at",
    "fisher_information",
    "parameter_covariance",
```

3. `__all__` 的 `# errors` 段改为：

```python
    # errors
    "BayesmithError",
    "GraphError",
    "TraceError",
    "StructureError",
    "ConvergenceError",
    "NotGaussian",
```

4. `_LAZY_ATTRS` 的末尾（`"nuts"` 之后）插入：

```python
    "linear_operator": ("bayesmith.exact.linearity", "linear_operator"),
    "check_linearity": ("bayesmith.exact.linearity", "check_linearity"),
    "wiener_solve": ("bayesmith.exact.solve", "wiener_solve"),
    "gcr_sample": ("bayesmith.exact.solve", "gcr_sample"),
    "condition_bound": ("bayesmith.exact.solve", "condition_bound"),
    "iterative_gls": ("bayesmith.exact.gls", "iterative_gls"),
    "sigma_from_graph": ("bayesmith.exact.gls", "sigma_from_graph"),
    "noise_std_at": ("bayesmith.exact.gaussian", "noise_std_at"),
    "fisher_information": ("bayesmith.exact.fisher", "fisher_information"),
    "parameter_covariance": ("bayesmith.exact.fisher", "parameter_covariance"),
```

5. `_LAZY_SUBMODULES` 改为 `("graph", "bridge", "exact", "errors")`。

**懒解析在这里不是风格。** `exact/` 顶到 numpyro（`gaussian.py` 用 `numpyro.distributions`），而 `errors.py` 的"仅 stdlib"契约由 `test_errors_module_imports_no_heavy_dependency` 强制着——Python 在导入任何子模块前都会先跑包的 `__init__.py`，所以急切导入 `exact` 会**直接破坏那条已有的测试**。P1 的第 8 号缺陷就是这条路径。

- [ ] **Step 3: 扩展公共 API 测试**

追加到 `tests/test_public_api.py`：

```python
def test_every_exact_name_resolves_and_is_the_same_object_as_its_module_s():
    """Lazy resolution must hand back the real function, not a shim.

    Checked by identity rather than by `hasattr`: a __getattr__ that returned
    the module, or a wrapper, would pass a name check and fail here.
    """
    import bayesmith
    from bayesmith.exact import gaussian, gls, linearity, solve
    from bayesmith.exact import fisher

    expected = {
        "linear_operator": linearity.linear_operator,
        "check_linearity": linearity.check_linearity,
        "wiener_solve": solve.wiener_solve,
        "gcr_sample": solve.gcr_sample,
        "condition_bound": solve.condition_bound,
        "iterative_gls": gls.iterative_gls,
        "sigma_from_graph": gls.sigma_from_graph,
        "noise_std_at": gaussian.noise_std_at,
        "fisher_information": fisher.fisher_information,
        "parameter_covariance": fisher.parameter_covariance,
    }
    for name, target in expected.items():
        assert getattr(bayesmith, name) is target, name
        assert name in bayesmith.__all__


def test_importing_bayesmith_still_does_not_import_numpyro():
    """The exact subpackage reaches numpyro, so it must stay lazy.

    Eagerly importing it here would break errors.py's stdlib-only contract,
    because Python runs a package's __init__.py before any submodule of it.
    """
    import subprocess
    import sys

    code = (
        "import bayesmith, sys;"
        "assert 'numpyro' not in sys.modules;"
        "assert 'bayesmith.exact' not in sys.modules;"
        "assert bayesmith.wiener_solve is not None;"
        "assert 'numpyro' in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 4: 加边界验证与极端值所需的玩具模型**

追加到 `tests/exact/models.py`：

```python
def tunable_curvature(*, n=8, departure=0.0, sigma=0.5, prior_std=1.0, seed=14):
    """``mu = (w + departure * w**2 / prior_std) X``.

    ``departure`` is, to first order, the relative departure from affinity a
    one-sigma probe sees -- so sweeping it across check_linearity's rtol walks
    the accept/reject boundary directly, which is what
    `boundary-validation.md` asks for: evaluate BOTH sides at the threshold
    rather than trusting the dispatcher's own verdict.
    """
    x = jnp.linspace(1.0, 2.0, n)
    data = 1.0 * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, prior_std))
        mu = det(
            "mu",
            lambda w_, x_: (w_ + departure * w_**2 / prior_std) * x_,
            w,
            xs,
            linear_in=("w",),
        )
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def collinear_pair(*, n=8, sigma=0.4, prior_std=3.0, seed=13):
    """``mu = (a + b) X`` -- the data cannot tell ``a`` from ``b`` at all.

    Jointly affine, so check_linearity passes and the JOINT block is the right
    thing: the data fixes ``a + b``, the prior alone fixes ``a - b``, and the
    joint kappa reports honestly how much worse one direction is determined
    than the other. Alternating over two one-latent blocks instead would
    report a converged residual and a condition number of ~1 forever, which is
    rheplicant's recorded failure in its purest form.
    """
    x = jnp.linspace(1.0, 3.0, n)
    data = 2.0 * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        a = sample("a", lambda: dist.Normal(0.0, prior_std))
        b = sample("b", lambda: dist.Normal(0.0, prior_std))
        mu = det("mu", lambda a_, b_, x_: (a_ + b_) * x_, a, b, xs, linear_in=("a", "b"))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def wide_plate(*, size, sigma=0.4, tau=1.5, seed=15):
    """``plated_latent`` at an arbitrary plate size, for the size sweep."""
    return plated_latent(n=size, sigma=sigma, tau=tau, seed=seed)


def many_observations(*, count, n=6, weight=1.5, sigma=0.4, seed=16):
    """One latent constrained by ``count`` observed nodes.

    Names are ``obs_0 ... obs_{count-1}``, whose sorted order is their
    declaration order only while ``count <= 10`` -- deliberately, so the
    codomain ordering is exercised rather than assumed.
    """
    key = jax.random.key(seed)
    grids = [jnp.linspace(1.0, 2.0 + index, n) for index in range(count)]
    data = [
        weight * grid + sigma * jax.random.normal(jax.random.fold_in(key, index), (n,))
        for index, grid in enumerate(grids)
    ]

    def model():
        w = sample("w", lambda: dist.Normal(0.0, 4.0))
        for index, (grid, values) in enumerate(zip(grids, data, strict=True)):
            xs = const(f"X_{index}", grid)
            mu = det(
                f"mu_{index}", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",)
            )
            observe(
                f"obs_{index}", lambda m: dist.Normal(m, sigma), mu, obs=values
            )

    return trace(model)
```

- [ ] **Step 5: 写边界测试**

```python
# tests/exact/test_boundaries.py
"""Threshold behaviour, checked by bypassing the dispatch and testing both sides.

`boundary-validation.md`: for a dispatcher that routes on a threshold T, the
useful check is not "method A gives the right answer at point P" but "A and B
agree at T". Every threshold P3a introduces is swept here, at values on both
sides of it, and each sweep asserts what the OTHER side does too -- a test
that only confirms the failing side would pass against a check that always
fails.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.errors import StructureError
from bayesmith.exact.gaussian import check_gaussian, noise_std_at
from bayesmith.exact.gls import iterative_gls, sigma_from_graph
from bayesmith.exact.linearity import check_linearity
from bayesmith.exact.block import unchecked_operator
from bayesmith.exact.solve import condition_bound, wiener_solve
from bayesmith import evaluate, observe, sample, trace
import numpyro.distributions as dist
from tests.exact.models import (
    radiometer,
    tunable_curvature,
    two_linear_latents,
)


@pytest.mark.parametrize("departure", [0.0, 1e-8, 1e-6])
def test_check_linearity_accepts_below_its_rtol(departure):
    """The quiet side of the threshold: a departure this small is roundoff."""
    graph = tunable_curvature(departure=departure)
    check_linearity(graph, ("w",), at={}, at_points=[{}])


@pytest.mark.parametrize("departure", [1e-2, 1.0, 10.0])
def test_check_linearity_refuses_above_its_rtol(departure):
    graph = tunable_curvature(departure=departure)
    with pytest.raises(StructureError, match="affine"):
        check_linearity(graph, ("w",), at={}, at_points=[{}])


def test_the_roundoff_floor_does_not_reject_a_perfectly_linear_block():
    """The small-probe end, where the relative measure would explode.

    scales down to 1e-9 of the prior width: the variation there is vanishing
    but roundoff is not, so without the per-probe floor the relative departure
    blows up and a genuinely linear block is refused.
    """
    graph = tunable_curvature(departure=0.0)
    check_linearity(
        graph, ("w",), at={}, at_points=[{}], scales=(1e-9, 1e-6, 1e-3, 1.0)
    )


def _probe_graph(distribution_fn):
    def model():
        w = sample("w", lambda: dist.Normal(0.0, 1.0))
        observe("d", distribution_fn, w, obs=jnp.zeros(3))

    return trace(model)


@pytest.mark.parametrize("scaling", [1.0, 1.0 + 1e-12, 1.0 - 1e-12])
def test_the_gaussian_probe_accepts_a_log_prob_that_agrees(scaling):
    """The ACCEPT side of the probe's rtol, which is the half easy to forget.

    A guard that always raised would pass every refusal test in this suite;
    only this one distinguishes it from a working guard. `scaling` sits far
    below `1e3 * eps(float32)`, so all three must be accepted.
    """

    class NearlyExact(dist.Normal):
        def log_prob(self, value):
            return scaling * super().log_prob(value)

    graph = _probe_graph(lambda w_: NearlyExact(w_, 0.7))
    env = evaluate(graph, {"w": jnp.asarray(0.3)})
    check_gaussian(graph, graph.node("d"), env)


@pytest.mark.parametrize("scaling", [1.5, 0.5, 1.01])
def test_the_gaussian_probe_refuses_a_log_prob_that_does_not(scaling):
    class Off(dist.Normal):
        def log_prob(self, value):
            return scaling * super().log_prob(value)

    graph = _probe_graph(lambda w_: Off(w_, 0.7))
    env = evaluate(graph, {"w": jnp.asarray(0.3)})
    with pytest.raises(StructureError, match="log_prob"):
        check_gaussian(graph, graph.node("d"), env)


def test_the_convergence_guard_flips_at_require_convergence_over_kappa():
    """tol just above and just below the value the guard's own algebra implies.

    Bypasses the guard to compute kappa first, then asks for a tol on each
    side of `require_convergence / kappa`. Both sides are asserted: a guard
    that always raised would pass the strict half alone.
    """
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = unchecked_operator(graph, ("a", "b"), at={})
        sigma = noise_std_at(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        kappa = float(condition_bound(block, noise_std=sigma, iterations=80))

        # CG on a 2-parameter system reaches machine precision in 2 steps, so
        # maxiter -- not tol -- is what sets the residual here. Measure the
        # bound the guard will compute, with the guard itself switched off.
        _, residual = wiener_solve(
            block, noise_std=sigma, tol=1e-14, maxiter=1, require_convergence=None
        )
        bound = float(residual) * kappa
        assert bound > 0.0, "fixture no longer leaves a measurable residual"

        # Strict side: ask for half the bound the guard will find.
        with pytest.raises(Exception, match="did not converge"):
            wiener_solve(
                block,
                noise_std=sigma,
                tol=1e-14,
                maxiter=1,
                require_convergence=bound / 2.0,
            )
        # Permissive side: ask for twice it, and the SAME solve is accepted.
        # Asserting only the strict half would pass against a guard that
        # always raised.
        wiener_solve(
            block,
            noise_std=sigma,
            tol=1e-14,
            maxiter=1,
            require_convergence=bound * 2.0,
        )


def test_the_unreachable_branch_names_precision_rather_than_tolerance():
    """kappa * eps above require_convergence -- no tol or maxiter can help.

    Run in float32 on purpose: eps is 1.2e-7, so a kappa of 1e6 already puts
    the product above a 1e-3 target and the remedy is x64, not iterations.
    """
    graph = two_linear_latents()
    block = unchecked_operator(graph, ("a", "b"), at={})
    sigma = noise_std_at(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
    import dataclasses

    starved = dataclasses.replace(
        block, prior_std={**block.prior_std, "b": jnp.asarray(1e4, dtype=jnp.float32)}
    )
    kappa = float(condition_bound(starved, noise_std=sigma, iterations=80))
    epsilon = float(jnp.finfo(jnp.float32).eps)
    assert kappa * epsilon > 1e-3, "fixture no longer reaches the unreachable branch"
    with pytest.raises(Exception, match="enable_x64"):
        wiener_solve(
            starved, noise_std=sigma, tol=1e-8, maxiter=1, require_convergence=1e-3
        )


def test_asking_for_a_reweight_tol_below_the_floor_reports_not_converged():
    """The trap the default exists to avoid, demonstrated on both sides.

    Consecutive solves differ by roughly the inner CG's own residual whatever
    the outer iteration does, so a reweight_tol below `tol` measures CG rather
    than the fixed point: the run spends max_reweights steps and reports
    converged=False for a fixed point it did reach. The same model with the
    default tolerance converges.
    """
    with jax.enable_x64(True):
        graph = radiometer()
        block = unchecked_operator(graph, ("w",), at={})
        seam = sigma_from_graph(graph, {})
        honest = iterative_gls(block, seam, tol=1e-10, max_reweights=40)
        starved = iterative_gls(
            block, seam, tol=1e-10, reweight_tol=1e-16, max_reweights=40
        )
    assert bool(honest.converged)
    assert not bool(starved.converged)
    # And the answers agree: the fixture reached the fixed point either way.
    assert np.allclose(
        np.asarray(honest.solution["w"]), np.asarray(starved.solution["w"]), rtol=1e-6
    )
```

- [ ] **Step 6: 写极端参数测试**

```python
# tests/exact/test_extremes.py
"""Extreme parameter values. Failure modes are U-shaped; the middle is safe.

`boundary-validation.md`, verbatim requirement: for every parameter dimension
include both endpoints, a very low value and a very high one -- not only the
moderate range a probe naturally reaches for. The rule was written after a fix
inferred from `ell in {10, 100, 500}` turned out to blow up by 63 orders of
magnitude at `ell = 5000`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.exact.gaussian import noise_std_at
from bayesmith.exact.linearity import linear_operator
from bayesmith.exact.solve import condition_bound, wiener_solve
from tests.exact.models import (
    collinear_pair,
    many_observations,
    straight_line,
    wide_plate,
)
from tests.exact.oracle import flat_domain, graph_oracle


@pytest.mark.parametrize("size", [1, 2, 1000])
def test_a_plate_of_any_size_matches_the_oracle(size):
    with jax.enable_x64(True):
        graph = wide_plate(size=size)
        block = linear_operator(graph, ("z",), at={})
        sigma = noise_std_at(graph, {"z": jnp.zeros(size)})
        got, _ = wiener_solve(block, noise_std=sigma, tol=1e-14)
        oracle = graph_oracle(graph, ("z",), at={})
    assert block.shape["z"] == (size,)
    assert np.allclose(flat_domain(got, block.names), oracle.mean, rtol=1e-8)


@pytest.mark.slow
def test_a_very_wide_plate_still_solves_and_stays_finite():
    """10^5 entries: too big for the dense oracle, so the assertion is weaker.

    Checked against the closed form instead, which for `z_i ~ N(0, tau)` and
    `d_i ~ N(z_i, sigma)` is elementwise and needs no matrix at all.
    """
    size, tau, sigma = 100_000, 1.5, 0.4
    with jax.enable_x64(True):
        graph = wide_plate(size=size, tau=tau, sigma=sigma)
        block = linear_operator(graph, ("z",), at={})
        noise = noise_std_at(graph, {"z": jnp.zeros(size)})
        got, residual = wiener_solve(block, noise_std=noise, tol=1e-12)
        data = np.asarray(block.data["d"])
    shrinkage = tau**2 / (tau**2 + sigma**2)
    assert np.all(np.isfinite(np.asarray(got["z"])))
    assert float(residual) < 1e-8
    assert np.allclose(np.asarray(got["z"]), shrinkage * data, rtol=1e-6)


@pytest.mark.parametrize("prior_std", [1e-6, 1.0, 1e6])
def test_the_solve_matches_the_oracle_across_six_orders_of_prior_width(prior_std):
    """Both ends matter and for opposite reasons: a tight prior makes the data
    irrelevant, a loose one makes the prior the only thing holding the blind
    directions down -- which is precisely where kappa explodes."""
    with jax.enable_x64(True):
        graph = straight_line(prior_std=prior_std, prior_mean=1.75)
        block = linear_operator(graph, ("w",), at={})
        sigma = noise_std_at(graph, {"w": jnp.asarray(0.0)})
        got, _ = wiener_solve(block, noise_std=sigma, tol=1e-14, require_convergence=None)
        oracle = graph_oracle(graph, ("w",), at={})
    assert np.allclose(flat_domain(got, block.names), oracle.mean, rtol=1e-7)


@pytest.mark.parametrize("sigma", [1e-6, 1.0, 1e6])
def test_the_solve_matches_the_oracle_across_six_orders_of_noise(sigma):
    with jax.enable_x64(True):
        graph = straight_line(sigma=sigma, prior_std=2.0, prior_mean=1.75)
        block = linear_operator(graph, ("w",), at={})
        noise = noise_std_at(graph, {"w": jnp.asarray(0.0)})
        got, _ = wiener_solve(block, noise_std=noise, tol=1e-14, require_convergence=None)
        oracle = graph_oracle(graph, ("w",), at={})
    assert np.allclose(flat_domain(got, block.names), oracle.mean, rtol=1e-7)


@pytest.mark.parametrize("count", [1, 2, 5])
def test_any_number_of_observed_nodes_matches_the_oracle(count):
    with jax.enable_x64(True):
        graph = many_observations(count=count)
        block = linear_operator(graph, ("w",), at={})
        sigma = noise_std_at(graph, {"w": jnp.asarray(0.0)})
        got, _ = wiener_solve(block, noise_std=sigma, tol=1e-14)
        oracle = graph_oracle(graph, ("w",), at={})
    assert len(block.offset) == count
    assert np.allclose(flat_domain(got, block.names), oracle.mean, rtol=1e-8)


def test_two_exactly_collinear_parents_are_solved_jointly_and_kappa_says_so():
    """The data fixes a+b; only the prior fixes a-b.

    The joint block matches the oracle, and its kappa is large -- which is the
    honest number. Two alternating one-latent blocks would each report a
    converged residual and a kappa near 1 forever, and neither could see the
    direction they are jointly blind to.
    """
    with jax.enable_x64(True):
        graph = collinear_pair(prior_std=3.0)
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = noise_std_at(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        got, _ = wiener_solve(block, noise_std=sigma, tol=1e-14)
        kappa = float(condition_bound(block, noise_std=sigma, iterations=80))
        oracle = graph_oracle(graph, ("a", "b"), at={})
        single = linear_operator(graph, ("a",), at={"b": jnp.asarray(0.0)})
        single_kappa = float(
            condition_bound(single, noise_std=sigma, iterations=80)
        )
    assert np.allclose(flat_domain(got, block.names), oracle.mean, rtol=1e-8)
    assert kappa > 100.0
    assert single_kappa == pytest.approx(1.0, rel=1e-6)
    assert kappa == pytest.approx(np.linalg.cond(oracle.precision), rel=1e-3)
```

- [ ] **Step 7: 跑全套**

```bash
.venv/bin/python -m pytest -q
```

Expected: 全绿。P1/P2 的 92 条 + P3a 新增约 60 条。

```bash
.venv/bin/python -m pytest --cov=bayesmith --cov-report=term-missing -q -m "not slow"
```

Expected: `src/bayesmith/exact/` 每个文件覆盖率 ≥ 80%（`common/testing.md` 的最低要求）。未覆盖行只应是标了 `# pragma: no cover` 的防御分支。

- [ ] **Step 8: ruff**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format --check src tests
```

Expected: 无问题。

- [ ] **Step 9: 提交**

```bash
git add -A src tests
git commit -m "feat: wire the exact solves into the public API, with boundary and extreme-value coverage"
```

---

## 验收（本计划完成的判据）

- [ ] `.venv/bin/python -m pytest -q` 全绿
- [ ] **`tests/exact/test_solve.py::test_wiener_solve_matches_the_dense_oracle` 通过，rtol 1e-8**——R1 与 R2 在 x64 下一致，而两者除模型外不共享任何一行
- [ ] `test_gcr_draws_have_the_oracle_mean_and_covariance` 通过：抽取的一、二阶矩落在 MC 误差内
- [ ] `test_a_bilinear_pair_passes_singly_and_fails_jointly` 通过：rheplicant 那个"所有守卫全绿而答案错几千开尔文"的分区失效被三次前向求值抓住
- [ ] `test_check_linearity_probes_more_than_the_caller_s_at_point` 通过：多 `at` 点不是装饰
- [ ] `test_check_gaussian_catches_a_distribution_that_lies_about_its_log_prob` 通过：内省是证据、不是证明
- [ ] `test_iterative_gls_finds_the_fixed_point_a_dense_iteration_finds` 通过
- [ ] `test_dense_operator_matches_the_probed_design_matrix` 通过：R3 与 R2 一致
- [ ] `tests/exact/test_extremes.py` 全通过：plate 1/2/1000（及 slow 的 1e5）、`prior_std` 与 σ 各跨六个量级、观测节点 1/2/5、完全共线的父节点
- [ ] `tests/exact/test_boundaries.py` 全通过：每个阈值的**两侧**都被断言
- [ ] `test_importing_bayesmith_still_does_not_import_numpyro` 通过：`exact/` 保持懒加载，`errors.py` 的仅-stdlib 契约未破
- [ ] 每个任务的变异测试都做过，且每条都指名了一条**变红的具体测试**
- [ ] 每个任务的 AST 规格比对都跑过；每一处实质差异都在任务收尾时具名说明了
- [ ] `ruff check` 与 `ruff format --check` 干净
- [ ] `exact/` 每个文件覆盖率 ≥ 80%

## 明确不在本计划范围内

留给 P3b，不要顺手做：

- 合格性分类与分区推导（`compile/dispatch.py`）
- `InferencePlan`、它的可打印表、`sample()` / `estimate()`
- `HMCGibbs` 装配与 `gibbs_fn` 工厂
- SNIS、Kish ESS、PSIS k̂、独立-MH 的反向密度（`exact/correct.py`）
- `depends_on_prediction` 的**自动**探测与分派（P3a 只提供 `check_prediction_dependence`，由调用方显式调用）
- 读 `linear_in` 的**路径**规则（P3a 检验的是仿射性这一事实，不检查沿路径的声明）
- `propagate_covariance` / `push_forward`（P5）
- `MultivariateNormal` 稠密协方差观测噪声
