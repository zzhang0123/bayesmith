# bayesmith P3b 分派与执行 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `bayesmith.compile(graph)` 从图推导出分区与方法、把理由打印出来，并按分区执行——精确块走 GCR，σ 依赖的块走带修正的提议，其余交给 NUTS——而在此之前先修好两个已被证明会给出静默错误答案的守卫。

**Architecture:** 分派器读三条已存在但从未被读过的结构轴（`linear_in`、`depends_on_prediction`、`support`），用 P3a 的检验器验证它们，产出一个 `InferencePlan`。执行层把精确块装进 `numpyro.infer.HMCGibbs` 的 `gibbs_sites`，`gibbs_fn` 每 sweep 重建无矩阵算子并抽取；σ 依赖块自身时提议不再是条件抽取，用**独立提议 + MH 接受步**精确修正。

**上游 spec:** `docs/superpowers/specs/2026-08-23-p3b-dispatch-execution-design.md`。**该 spec 经四路独立审查修订过，其 §〇 记录了被推翻的初稿主张——读计划前先读 §〇 与 §一。**

**Tech Stack:** Python ≥3.11、JAX 0.11.1、Equinox 0.13.8、NumPyro 0.21.0、NumPy 2.5.2、pytest。

---

## 五条纪律（P1 三条 + P3a 实测追加两条，适用于每个任务）

1. **断言必须验证行为，不能是重言式。** 每写一条断言问：**如果实现是错的，这条会红吗？** 答不上来就重写。
2. **每个任务收尾做变异测试。** 故意按任务指定的方式破坏实现，确认一条**具名**的测试变红，然后还原。P1 的 8 个缺陷里有 3 个只有它能发现；P3a 的约 30 个里绝大多数同样。
3. **数值巧合会让守卫失效，两个面都要防。**
   - **断言面**：断言的常数不得等于任何参数的当前值。
   - **变异面**：**变异写入的值不得等于正确值。**
4. **变异不变红时，按「fixture 到不了它声称的区域」→「变异是 no-op」→「测试写错了」的顺序诊断。**
5. **每个任务收尾做 AST 规格比对。** 把本计划该任务的代码块与提交的文件做 AST 比对。**实质差异是可以的**——发现了计划的缺陷就该改实现——但每一处都要在任务收尾时**具名说明**。

```bash
cat > /tmp/ast_compare.py << 'PY'
import ast, sys
a, b = (ast.dump(ast.parse(open(p).read())) for p in sys.argv[1:3])
print("IDENTICAL" if a == b else "DIFFERENT")
sys.exit(0 if a == b else 1)
PY
```

## 四条子判据（P3a 用约 30 个缺陷换来，每个任务都要过一遍）

1. **双侧扫描**。跨越量级 ≠ 双侧。判据：**默认值落在扫描范围的哪一端？** 在端点就是单侧。
2. **点 vs 区域**。找到一个能分开两种实现的参数点不够——扫「不该有影响」的维度（seed、n、先验中心与宽度、噪声尺度），报告分离成立的范围。只在一个点上成立，就在 docstring 里写明**「这是点不是区域」**及其依赖的机制。
3. **结构维度**。列出被测代码在哪些维度上分支（标量 vs 数组、单叶 vs 多叶、单观测 vs 多观测、有 plate vs 无 plate），检查测试集合在每一维取到**至少两个值**。
4. **`slow` 标记会把守卫从 `-m "not slow"` 里拿掉**。打标记前问：它是不是某个 bug 的唯一守卫。

## 三条精度与排版纪律

- **绝不调用 `jax.config.update("jax_enable_x64", ...)`**——进程级全局，关不回去。用 `with jax.enable_x64(True):`。**且图必须在 `with` 块内构造**（`const`/`observe` 在 `trace()` 时就调 `jnp.asarray`）。不要打 `@pytest.mark.x64`。
- **`pytest.approx(x, rel=r)` 有默认 `abs=1e-12`**，对小于约 1e-12 的期望值完全失效，需显式 `abs=0.0`。
- 计划里每个 python 代码块的**首行路径注释**是给读计划的人看的**元信息**，**不抄进源文件**。

## 一条本计划特有的纪律：数字必须带参数化

spec §1.3 与 §二 的教训——初稿把 `5.98x`、`0.24x`、`ESS=3.0`、`khat=0.184` 当成事实写下，实测它们全是 **float32、单种子、短跑**的点估计，且多数落在自身分布的极端（`0.184` 是 20 个种子里的最大值；`5.98x` 换个种子变 `4.72x`）。

> **判据**：任何进入测试套件的数字，**`seed`、`N`、dtype、`tol`、`maxiter` 必须写进参数化**，docstring 里写**实测区间**而非单点。做不到就不要断言那个数字。

---

## 文件结构

| 文件 | 职责 | 变化 |
|---|---|---|
| `src/bayesmith/exact/linearity.py` | **B1**：`affinity_errors` 逐叶/逐元素 + `1/sigma` 加权 | 改 ~+35 |
| `src/bayesmith/exact/gls.py` | **B2**：逐成员独立探测方向（含符号混合） | 改 ~+25 |
| `src/bayesmith/exact/block.py` | `probe_gaussian` 关键字（`unchecked_operator` 与 `_env_before` 两处） | 改 ~+40 |
| `src/bayesmith/exact/correct.py` | `log_weights`、SNIS、Kish ESS、k̂、MH 接受步 | 新写 ~210 |
| `src/bayesmith/exact/gibbs.py` | `gibbs_fn` 工厂 + `HMCGibbs` 装配 | 新写 ~180 |
| `src/bayesmith/dispatch/__init__.py` | 包标记 | 新写 ~1 |
| `src/bayesmith/dispatch/classify.py` | 合格性分类 + 分区推导 | 新写 ~280 |
| `src/bayesmith/dispatch/plan.py` | `Block` / `InferencePlan` / `__str__` / `sample` / `estimate` | 新写 ~270 |
| `src/bayesmith/__init__.py` | `compile` 进 `_LAZY_ATTRS`，`dispatch` 进 `_LAZY_SUBMODULES` | 改 |
| `tests/exact/models.py` | 新 fixture（见各任务） | 改 |
| `tests/dispatch/` | 新测试目录 | 新写 |

**依赖方向**：`exact/correct` → `exact/{block,solve}` + `graph/evaluate`；`exact/gibbs` → `exact/*` + `bridge`；`dispatch/classify` → `exact/{gaussian,linearity,gls}` + `graph`；`dispatch/plan` → `dispatch/classify` + `exact/*` + `bridge`。无环。

## 任务依赖

```
Task 1 (B1 判决对比表)  →  Task 2 (B1 修复)  ─┐
                           Task 3 (B2 修复)  ─┤
                           Task 4 (probe_gaussian) ─┤
                                                    ├→ Task 5 (classify)
                                                    │        ↓
                                                    │  Task 6 (InferencePlan)
                                                    │        ↓
                           Task 7 (correct.py) ─────┤  Task 8 (gibbs.py)
                                                    │        ↓
                                                    └→ Task 9 (sample/estimate)
                                                             ↓
                                                       Task 10 (验收关口)
```

**Task 1 必须先于 Task 2**：B1 的修复会改变 `check_linearity` 的判决，而 Task 5 的分类表以那些判决为前提。判决翻转必须在改代码**之前**知道。

---

## Task 1：B1 修复前的判决对比表（**纯测量，不改产品代码**）

spec §1.4 的「实施前必测」。加权版本可能让今天通过的合法 fixture 变红。**先量，再改。**

> **已完成**（提交 `f18536c`），结果见 `docs/superpowers/plans/2026-08-23-p3b-task1-verdicts.md`。它推翻了本节初稿的两个说法：套件跑在 **float32** 而非 float64；`cubic_tail` 在默认参数下**不是**「曲率可忽略」而是一个真实失败。四处更正见 Task 2 开头。

**Files:**
- Create: `/tmp/p3b_task1_verdicts.py`（脚本，不进仓库）
- Create: `docs/superpowers/plans/2026-08-23-p3b-task1-verdicts.md`（结果表，进仓库）

- [ ] **Step 1: 写测量脚本**

```python
# /tmp/p3b_task1_verdicts.py
"""B1: every fixture's verdict under the current guard vs two candidate fixes.

Does NOT modify product code. It drives the REAL `isolate` / `_env_before` /
`probe_at` scheme -- random per-member directions at `zero`, scaled by each
member's declared prior width, exactly as `check_linearity` builds them -- and
only re-does the three candidate NORMALISATIONS on the same numbers. Any other
probe scheme would compare a different thing.
"""
import jax, jax.numpy as jnp, numpy as np
import tests.exact.models as M
from bayesmith.exact.block import _env_before, isolate
from bayesmith.exact.gaussian import noise_std_at
from bayesmith.exact.linearity import DEFAULT_SCALES

FIXTURES = [
    ("straight_line", ("w",), ()), ("two_linear_latents", ("a", "b"), ()),
    ("quadratic_claim", ("w",), ()), ("cubic_tail", ("w",), ()),
    ("affine_only_at_zero", ("x",), ()), ("collinear_pair", ("a", "b"), ()),
    ("plated_latent", ("z",), ()), ("plated_latent_through_deterministic", ("z",), ()),
    ("two_observations", ("w",), ()), ("radiometer", ("w",), ()),
    ("radiometer_group", ("a", "b"), ()), ("prior_held_direction", ("a", "b"), ()),
    ("plated_and_scalar_latents", ("w", "z"), ()),
    ("indirect_ancestor", ("x",), ("tau",)),
]

def three_normalisations(graph, names, at, key=jax.random.key(0)):
    """{scale: (global_rel, per_element_rel, weighted_by_sigma)}."""
    _, domain = _env_before(graph, names, at)
    g = isolate(graph, names, at)
    zero = {n: jnp.zeros(domain[n][0], dtype=domain[n][1]) for n in names}
    baseline, tangent = jax.linearize(g, zero)
    sigma = noise_std_at(graph, {**at, **zero})
    ordered = sorted(names)
    out = {}
    for index, scale in enumerate(DEFAULT_SCALES):
        root = jax.random.fold_in(key, index)
        probe = {
            m: domain[m][3] * scale * jax.random.normal(
                jax.random.fold_in(root, pos), domain[m][0], dtype=domain[m][1])
            for pos, m in enumerate(ordered)
        }
        actual = g(probe)
        predicted = jax.tree.map(lambda b, t: b + t, baseline, tangent(probe))
        keys = sorted(baseline)
        big = lambda tree: max(float(jnp.max(jnp.abs(tree[k]))) for k in keys)
        var_g = max(float(jnp.max(jnp.abs(actual[k] - baseline[k]))) for k in keys)
        dep_g = max(float(jnp.max(jnp.abs(actual[k] - predicted[k]))) for k in keys)
        per, wt = 0.0, 0.0
        for k in keys:
            v = jnp.abs(actual[k] - baseline[k])
            d = jnp.abs(actual[k] - predicted[k])
            per = max(per, float(jnp.max(d / jnp.maximum(v, 1e-300))))
            wt = max(wt, float(jnp.max(d / jnp.abs(sigma[k]))))
        out[scale] = (dep_g / max(var_g, 1e-300), per, wt)
    return out

if __name__ == "__main__":
    with jax.enable_x64(True):
        hdr = f"{'fixture':40s} {'scale':>8s} {'global':>11s} {'per-elem':>11s} {'weighted':>11s}"
        print(hdr); print("-" * len(hdr))
        for name, names, outside in FIXTURES:
            g = getattr(M, name)()
            at = {}
            for o in outside:
                _, dom = _env_before(g, (o,), {n: jnp.asarray(0.0) for n in g.latents if n != o})
                at[o] = dom[o][2]                      # its prior mean
            try:
                res = three_normalisations(g, names, at)
            except Exception as e:
                print(f"{name:40s} FAILED {type(e).__name__}: {str(e).splitlines()[0][:44]}")
                continue
            for s_, (a, b, c) in res.items():
                print(f"{name:40s} {s_:8.1e} {a:11.3e} {b:11.3e} {c:11.3e}")
```

- [ ] **Step 2: 跑它**

```bash
PYTHONPATH=/Users/zzhang/projects/bayesmith .venv/bin/python /tmp/p3b_task1_verdicts.py
```

Expected: 每个 fixture × 每个 scale 三列数字。`quadratic_claim` / `tunable_curvature(departure>0)` / `affine_only_at_zero` 三列都应远大于 `1e4*eps`；`straight_line` / `two_linear_latents` 三列都应接近 0。

- [ ] **Step 3: 判定阈值下的翻转**

对每一列，用 `rtol = 1e4 * eps(float64) = 2.22e-12` 判 pass/fail（加权列的阈值另定，见 Step 4），列出**在三种归一化下判决不同**的 fixture。

- [ ] **Step 4: 决定加权列的阈值**

加权列的量纲是「偏离 / σ」，不是相对误差，所以 `1e4*eps` 不适用。判据应当是**偏离相对于噪声可忽略**：建议 `WEIGHTED_RTOL = 1e-3`（即仿射偏离小于噪声的千分之一）。**用 Step 2 的输出验证这个数**：合法 fixture 的加权列必须远低于它，`quadratic_claim` 必须远高于它。若不成立，按实测调整并记录理由。

- [ ] **Step 5: 写下结果表**

创建 `docs/superpowers/plans/2026-08-23-p3b-task1-verdicts.md`，含：完整的三列数字表；**判决翻转清单**（fixture、哪一列翻、为什么）；加权阈值的选定值与依据；以及一句结论——**Task 2 采用哪种归一化，以及 Task 5 的分类表是否需要跟着改**。

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/plans/2026-08-23-p3b-task1-verdicts.md
git commit -m "docs: measure every fixture's linearity verdict under three normalisations

B1's fix changes what check_linearity accepts, and the dispatcher reads
those verdicts as its criteria. Measured before changing anything so a
flipped verdict is a decision rather than a surprise."
```

---

## Task 2：B1 修复——`affinity_errors` 逐元素 + `1/sigma` 加权

> ## Task 1 实测带来的四处更正——**写 Task 2 之前必读**
>
> 结果表：`docs/superpowers/plans/2026-08-23-p3b-task1-verdicts.md`（提交 `f18536c`）。
>
> **(1) 套件跑在 float32，不是 float64。** `tests/exact/test_linearity.py` 里没有任何
> `enable_x64`，整个仓库也没有 `conftest.py`——所以 B1 守卫每天实际工作的 `rtol` 是
> **1.19e-3**，不是 2.22e-12。本计划此前默认 x64 是工作区间，**那是错的**。修复必须在
> **两种 dtype 下都验证**，且以 float32 为主。实测有四个 fixture **只因 dtype 就翻转判决**。
>
> **(2) `1e-300` 在 float32 下下溢，把诚实模型判成失败。** 实测
> `jnp.maximum(jnp.float32(0.0), 1e-300) == 0.0`，于是块动不了的那个陪域元素给出
> `0/0 = nan`，而 `not finite` 分支把 NaN 读成**失败**。`two_observations` 的
> `x2 = linspace(-1, 1, 5)` 第三项恰好是 `0.0`——**一个诚实的 fixture 会因为协变量网格里
> 有个零而被拒绝**。改用 `jnp.finfo(dtype).tiny`（实测 `0/tiny == 0.0`）。
>
> **(3) σ 加权判据必须由同一个逐元素舍入地板把关，并且要有自己的有限性检查。** 不加地板时
> 它测的是**动态范围而不是曲率**：在**恰好仿射**的模型上（`mu=(w+big)*X`），加权列在
> float32 下 offset/noise=1e2 处已达 2.44e-02，float64 下 1e17 处达 2.50e+01。加了地板后
> 这些全部变成 `0.000e+00`。
>
> **实测记分板（48 个 fixture 行，判错的个数）**：
>
> | 判据 | float32 判错 | float64 判错 |
> |---|---|---|
> | 全局（已发布） | 3 | 3 |
> | 逐元素（本计划初稿） | 12 | 1 |
> | σ 加权（本计划初稿） | 12 | 4 |
> | **σ 加权 + 逐元素地板** | **1** | **0** |
>
> 唯一漏掉的是 `tunable_curvature(1e-9)`，其偏离本就在 float32 舍入之下——**当前守卫同样漏掉它**。
>
> **(4) `cubic_tail` 在默认参数下不是「曲率真实但可忽略」。** 本计划开头那句话是错的：
> `prior_std=1.0` 时它是一个**真实的失败**，当前守卫已经拒绝它（scale 1e3 处 0.841）。
> 只有 `prior_std=1e-4` 才是可忽略的那一档，**而它在 float64 下会被当前已发布的守卫拒绝**
> ——`test_the_probe_magnitude_is_read_off_the_declared_prior` 断言它通过，那条测试的绿色
> 因此依赖于套件跑在 float32 上，而这件事没有写在任何地方。**Task 5 的分类表要记下这一条是
> dtype 相关的。**
>
> **`WEIGHTED_RTOL = 1e-3` 已验证**，但**只在加了地板之后**：float64 下窗口
> `(4.87e-08, 2.12e-02]`，比最差的诚实 fixture 低 **2.05e+04 倍**，比最小的假声明高
> **2.12e+01 倍**，比每一个**具名**的假声明高 ≥ **4.9e+07 倍**。**不加地板时这个窗口是空的**
> （最差诚实 1.28e+04 > 最小假声明 1.17e-02），即 1e-3 这个数字在没有地板时毫无意义。

**Files:**
- Modify: `src/bayesmith/exact/linearity.py`（`affinity_errors`、`_refuse_affinity`、`check_linearity`）
- Modify: `tests/exact/models.py`（新增两个 fixture）
- Test: `tests/exact/test_linearity.py`

- [ ] **Step 1: 加两个 fixture**

```python
# tests/exact/models.py  （追加）
def bright_and_faint_observations(*, n=6, bright=1e17, sigma_faint=0.01, w_true=0.8, seed=22):
    """An honest BRIGHT node beside a lying FAINT one that dominates the posterior.

    `affinity_errors` normalised `departure` by a `variation` taken as a max
    over EVERY codomain leaf, so the bright node set the yardstick -- and the
    roundoff floor -- for the faint one. Measured before the fix:
    `check_linearity` returned 3.45e-14 and PASSED, while `mu2` alone was
    correctly refused, and the "exact" answer was off by 202 true posterior
    standard deviations.

    `bright=1e17` is not adversarial engineering; it is the dynamic range
    this package targets -- a foreground in K beside a signal in mK is 1e6,
    and an interferometric visibility against a monopole is far more.
    """
    x1 = jnp.linspace(1.0, 2.0, n)
    x2 = jnp.linspace(1.0, 2.0, n)
    d1 = bright * w_true * x1
    d2 = (w_true + 0.5 * w_true**2) * x2

    def model():
        a = const("X1", x1)
        b = const("X2", x2)
        w = sample("w", lambda: dist.Normal(0.0, 3.0))
        m1 = det("mu1", lambda w_, x_: bright * w_ * x_, w, a, linear_in=("w",))
        m2 = det("mu2", lambda w_, x_: (w_ + 0.5 * w_**2) * x_, w, b, linear_in=("w",))
        observe("d1", lambda u: dist.Normal(u, bright * 100.0), m1, obs=d1)
        observe("d2", lambda u: dist.Normal(u, sigma_faint), m2, obs=d2)

    return trace(model)


def bright_and_faint_channels(*, n=6, bright=1e17, sigma=0.01, w_true=0.8, seed=23):
    """The same dilution WITHIN ONE ARRAY -- one bright channel, five faint.

    Sibling of `bright_and_faint_observations`, and the more realistic of the
    two: a spectrum whose first channel carries a bright foreground and whose
    remaining channels carry the signal is one observed node, not two. A
    per-LEAF fix is not enough here -- only a per-ELEMENT comparison sees it,
    which is the same argument `check_gaussian`'s docstring already makes for
    its own elementwise probe.
    """
    x = jnp.concatenate([jnp.array([bright]), jnp.linspace(1.0, 2.0, n - 1)])
    curvature = jnp.concatenate([jnp.zeros(1), jnp.full(n - 1, 0.5)])
    truth = (w_true + curvature * w_true**2) * x
    scale = jnp.concatenate([jnp.array([bright * 100.0]), jnp.full(n - 1, sigma)])

    def model():
        xs = const("X", x)
        cs = const("C", curvature)
        w = sample("w", lambda: dist.Normal(0.0, 3.0))
        mu = det("mu", lambda w_, x_, c_: (w_ + c_ * w_**2) * x_, w, xs, cs,
                 linear_in=("w",))
        observe("d", lambda u: dist.Normal(u, scale), mu, obs=truth)

    return trace(model)
```

- [ ] **Step 2: 写失败的测试**

```python
# tests/exact/test_linearity.py  （追加）
def test_a_bright_component_does_not_mask_a_false_claim_on_a_faint_one():
    """The guard must not let one codomain leaf set another's yardstick.

    Measured before the fix: `check_linearity` returned a worst relative
    error of 3.45e-14 on this graph -- a clean PASS -- while the faint node
    ALONE was correctly refused. The bright leaf supplied both the
    normalising `variation` and the roundoff `floor`, and each dilution was
    independent of the other.

    The consequence is not cosmetic: the "exact" posterior on this graph is
    +1.125 against a truth of +0.803, which is 202 true posterior standard
    deviations. `d2`'s sigma is 0.01 and `d1`'s is 1e19, so the faint node
    carries essentially all the information.
    """
    graph = bright_and_faint_observations()
    with pytest.raises(StructureError, match="not affine"):
        check_linearity(graph, ["w"])


def test_the_dilution_is_caught_within_a_single_array_too():
    """Per-leaf is not enough -- the bright and faint entries share a leaf.

    Named by `test_a_bright_component_does_not_mask_a_false_claim_on_a_faint_one`,
    and it by this one: the two are the leaf-level and element-level halves of
    one defect, and a fix that only groups by leaf passes the first and fails
    this. `check_gaussian` made the elementwise choice for exactly this reason
    and says so in its own docstring.
    """
    graph = bright_and_faint_channels()
    with pytest.raises(StructureError, match="not affine"):
        check_linearity(graph, ["w"])


def test_an_affine_model_with_the_same_dynamic_range_still_passes():
    """The two-sided half: huge dynamic range alone must NOT trip the guard.

    Without this, a "fix" that simply tightened the tolerance would pass both
    tests above while refusing every legitimate wide-dynamic-range model --
    which is most of this package's intended use.
    """
    graph = bright_and_faint_channels(w_true=0.8)
    honest = eqx.tree_at(
        lambda g: g.nodes[2].fn,          # 'mu'
        graph,
        replace=lambda w_, x_, c_: w_ * x_,
    )
    assert check_linearity(honest, ["w"])
```

- [ ] **Step 3: 跑测试确认它红**

```bash
.venv/bin/python -m pytest tests/exact/test_linearity.py -k "bright or dilution or dynamic_range" -v
```

Expected: 前两条 **FAIL**（`DID NOT RAISE StructureError`），第三条 PASS。**若第一条已经绿，停下来诊断**——按纪律 4，第一个假设是 fixture 没到达它声称的区域（检查 `bright` 是否真的进了同一个 `check_linearity` 调用）。

- [ ] **Step 4: 改 `affinity_errors`——逐元素相对 + σ 加权，两者任一失败即失败**

**不替换 `rtol` 的语义**（它是公开 kwarg 且有文档），而是把全局归约改成逐元素，**并新增**一个以 σ 为单位的独立判据。两者任一失败即拒绝。理由：逐元素相对量修掉亮/暗掩盖，但在 `variation_i ≈ 0` 的元素上是 0/0；σ 加权没有这个问题（`check_gaussian` 已保证 σ 严格为正），且它才是似然真正关心的量。两条一起是**双侧**的——单靠任何一条都能被一个方向的「修复」绕过。

```python
# src/bayesmith/exact/linearity.py
WEIGHTED_RTOL = 1e-3
"""Departure from affinity measured in units of the noise sigma, above which
a ``linear_in`` claim is refused.

NOT a relative error, so ``1e4 * eps`` would be meaningless for it. The
likelihood divides every residual by sigma, so sigma is the unit in which
"this departure cannot change the posterior" is a statement with content.
Pinned by Task 1's measured table: every legitimate fixture sits far below
it, every false claim far above.
"""


def affinity_errors(
    g: Callable[[Any], Any],
    zero: Any,
    probe_at: Callable[[int, float], Any],
    scales: Sequence[float],
    rtol: float | None,
    *,
    sigma: dict[str, jax.Array],
    at_description: str = "the linearisation point",
) -> tuple[dict[float, float], list[float], float]:
    """Compare a map against its own linearization at zero, **per element**.

    Args:
        sigma: ``{observed: scale}`` at the same point ``zero`` is anchored
            at. The second, unit-ful half of the criterion below.
        at_description: names the point ``zero`` is anchored at.

    Two changes from the original, both forced by measurement:

    * **Per element, not a max over the whole codomain.** ``variation`` and
      ``floor`` were each a ``_biggest`` over EVERY leaf, so a bright leaf
      supplied both the yardstick and the roundoff floor for a faint one.
      Measured: an honest 1e17 component beside a false ``linear_in`` claim
      on a 1e-2 one reported 3.45e-14 and PASSED, while the faint node alone
      was correctly refused -- and the resulting "exact" posterior was 202
      true posterior standard deviations wrong. The same dilution happens
      between ELEMENTS of one leaf, which is the realistic case: a spectrum
      with one bright foreground channel and five faint signal ones.
    * **A second criterion in units of sigma.** The relative measure is 0/0
      on an element the block does not move at all; the weighted one is not,
      and it is what the likelihood actually cares about.

    The original already suspected half of this. Its own floor comment says
    the floor must not be set "by the baseline alone, which would let an
    unrelated bright component disable the check" -- and then took a global
    max anyway. This finishes that thought.

    ``check_gaussian`` made the elementwise choice first, and argues it in
    its docstring: a summed comparison dilutes a localised defect by the
    magnitudes of the correct entries.
    """
    baseline, tangent = jax.linearize(g, zero)
    if not all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(baseline)):
        raise StructureError(
            f"the prediction is already non-finite at {at_description}, before "
            "any probe is taken -- so nothing here is a statement about any "
            "linear_in declaration. Some part of the graph overflows at the "
            "values the latents outside the block are sitting at. If those "
            "values came from the default prior draws, a heavy-tailed or very "
            "wide prior on an outside latent is the usual cause; pass "
            "`at_points=` explicitly to check where the model is actually used."
        )
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
        worst = 0.0
        bad = False
        # `baseline` comes out of `jax.linearize`, and JAX's dict pytree sorts
        # keys unconditionally on flatten -- so it is ALREADY sorted and a
        # `sorted()` here would be a provable no-op (P3a Task 9). Iterating it
        # directly says that, instead of implying a guarantee this loop makes.
        for key in baseline:
            variation = jnp.abs(actual[key] - baseline[key])
            departure = jnp.abs(actual[key] - predicted[key])
            # PER ELEMENT: a bright entry no longer sets a faint one's scale.
            # `finfo(dtype).tiny`, NOT 1e-300 -- measured, 1e-300 underflows to
            # 0.0 in float32, so an element the block cannot move gives 0/0 =
            # NaN and the `not finite` branch below reads that as a FAILURE.
            # `two_observations`'s covariate grid contains an exact zero, so
            # that alone refuses an honest model.
            relative = departure / jnp.maximum(variation, jnp.finfo(dtype).tiny)
            # A departure below the arithmetic's own noise floor is not
            # curvature. Per element for the same reason as above: a global
            # floor let one bright component disable the check everywhere.
            floor = 1e4 * epsilon * jnp.maximum(
                jnp.abs(actual[key]), jnp.abs(baseline[key])
            )
            # In units of the noise the likelihood divides by -- and gated by
            # the SAME floor. Ungated it measures DYNAMIC RANGE, not
            # curvature: measured on exactly-affine `mu = (w + big) * X`, the
            # weighted column reaches 2.44e-02 at an offset/noise ratio of
            # 1e2 in float32 and 2.50e+01 at 1e17 in float64. Gated, every one
            # of those is exactly 0.000e+00.
            weighted = departure / jnp.abs(sigma[key])
            above_floor = departure > floor
            worst = max(worst, float(jnp.max(relative)), float(jnp.max(weighted)))
            # NaN must count as a FAILURE: `nan > rtol` is False, so a naive
            # comparison reads an unusable probe as evidence of linearity.
            # Both criteria get their own check -- one can be NaN while the
            # other is finite.
            finite = bool(jnp.all(jnp.isfinite(relative) & jnp.isfinite(weighted)))
            over_relative = bool(jnp.any((relative > rtol) & above_floor))
            over_weighted = bool(jnp.any((weighted > WEIGHTED_RTOL) & above_floor))
            bad = bad or (not finite) or over_relative or over_weighted
        errors[scale] = worst
        verdicts[scale] = bad

    failed = sorted(scale for scale, is_bad in verdicts.items() if is_bad)
    return errors, failed, rtol
```

**调用点**：`check_linearity` 在循环里加 `sigma = noise_std_at(graph, {**point, **zero})` 并传 `sigma=sigma`。`_refuse_affinity` 的消息要**同时报出两个判据与各自的阈值**，否则用户看到一个数字而不知道是哪一条触发的。

- [ ] **Step 4b: 两种 dtype 都要验证**

Run: `.venv/bin/python -m pytest tests/exact/ -q`（float32，套件的真实区间）
Then 用 Task 1 的脚本在**两种 dtype** 下重跑记分板，确认 `σ 加权 + 逐元素地板` 这一列是 f32 判错 1 个、f64 判错 0 个。

**若某个诚实 fixture 在 float32 下变红，第一个假设不是「阈值要放松」，而是地板没有生效**——Task 1 实测的四种组合里，只有「加权 + 地板」这一种把诚实 fixture 全部保住。

- [ ] **Step 5: 跑全套**

```bash
.venv/bin/python -m pytest tests/exact/ -q
```

Expected: 全绿。**任何原有测试翻红都必须对照 Task 1 的表**——若表预测了这次翻转，按表里的理由更新那条测试的期望值并具名说明；若表没预测到，**停下来**，是新缺陷。

- [ ] **Step 6: 变异测试**

| 变异 | 必须变红 |
|---|---|
| `departure` 的分母改回全局 `max(_biggest(...))` | `test_a_bright_component_does_not_mask_a_false_claim_on_a_faint_one` |
| 逐元素 `jnp.max` 改成 `jnp.mean` | `test_the_dilution_is_caught_within_a_single_array_too` |
| `WEIGHTED_RTOL` 放大 1e6 倍 | 前两条 |
| `WEIGHTED_RTOL` 缩小 1e6 倍 | `test_an_affine_model_with_the_same_dynamic_range_still_passes`（双侧） |

每条先跑确认它红再还原。**第四条是双侧判据**——没有它，一个把阈值收到零的「修复」会通过前三条。

- [ ] **Step 7: 子判据检查**

- **双侧**：`bright` 从 1e-17 扫到 1e17，报告分离成立的区间。默认 1e17 在端点 ⟹ 必须补低端。
- **点 vs 区域**：扫 `n`、`sigma_faint`、`w_true`、seed，报告区间。
- **结构维度**：叶子数 {1, 2}（两个 fixture 各一）、叶内元素数 {1, 6}、成员数 {1, 2}——**成员数 2 今天没有 fixture，补一个**。

- [ ] **Step 8: AST 比对 + Commit**

```bash
git add src/bayesmith/exact/linearity.py tests/exact/test_linearity.py tests/exact/models.py
git commit -m "fix: affinity_errors compared per element in units of sigma

A bright honest codomain component supplied both the normalising variation
and the roundoff floor for a faint one, so a false linear_in claim on the
faint node passed at 3.45e-14 while the same node alone was correctly
refused. Measured consequence: 202 true posterior sd. The same dilution
happens between elements of one array, which is the realistic case -- a
spectrum with a bright foreground channel -- so the comparison is per
element, not per leaf.

Weighting by 1/sigma rather than by the prediction's own magnitude follows
from what the guard is for: the likelihood divides every residual by sigma,
so that is the unit in which 'negligible departure from affine' is a
meaningful claim at all.

check_gaussian already made this choice and argued it in its docstring.
This generalises it to the guard next door."
```

---


### 实测记录：Task 2 收尾时两条变异未被杀死（2026-08-24）

Task 2 已完成（`2de46d1`），外加一条后续修复（`23e6ccd`）。spec 合规审查判定 **ISSUES FOUND，一条 MISSING**。两条记在这里，**明确不在 Task 2 的范围内关闭**。

**(1) Step 6 变异行 2「逐元素 `jnp.any` 判决归约改成 above-floor 元素的均值」——全套仍绿。**

计划为它点名的 `test_the_dilution_is_caught_within_a_single_array_too`，以及实现者为它专门加的
`test_a_lone_lying_channel_is_not_diluted_by_five_honest_ones`，**两条都不变红**。

实测原因：在 `bright_and_faint_channels(lying=1)` 上两列分别是 `relative = 1.454e+00` 与
`weighted = 2.634e+09`，各自超阈值 **1.2e3 倍**与 **2.6e12 倍**。除以 6 一个都跨不过去。
fixture docstring 里「被六个诚实项稀释六倍」这句话**算术上是对的、但不充分**，而紧跟的
「随数组增长稀释无界」那一句**从未被执行到**——`n` 固定在 6。

要真正杀死它，需要**同时**关掉加权那一半**且** `n ≳ 1.2e3`（例如
`bright_and_faint_observations(sigma_faint=1e13, n≈2000)` 带单个撒谎元素）。加权那一半活着时，
没有任何可行的 `n` 做得到——需要 `n > 2.6e12`。

> **这一条是本计划自己第三条子判据的实例**：计划的变异行写下时没有算它的功效，于是写了一个
> 在这个 fixture 上**不可能变红**的目标。与 Task 6「一个不可能的目标」同形——两个约束都是我写的，
> 冲突摆在纸面上，两行算术即可看出。

**(2) Step 6 变异行 4「`WEIGHTED_RTOL` ÷ 1e6」——知情地不可达。**

Task 1 §5 实测：float32 下每个诚实 fixture 的 above-floor 加权偏离**恰好是 `0.000e+00`**，
所以在套件实际运行的 dtype 上**没有任何东西从下方约束这个常数**。诚实一侧是被**地板**保护的，
不是被阈值的取值保护的。`WEIGHTED_RTOL` 的 docstring 已如实写下这一点，并把双侧性转交给
`test_a_true_claim_with_real_roundoff_passes_at_any_offset_ratio`（它钉的是地板，且实测在地板被
移除时确实会死）。**记录为具名替代，不是静默空洞。**

---

## Task 3：B2 修复——`check_prediction_dependence` 的探测方向

> ## 实测更正：本任务的设计被 Task 3b 推翻并替换（2026-08-24）
>
> **下面 Step 4 起的正文是历史记录，不要照着实现。** 实际落地的是
> `ec7e142`：`DEPENDENCE_PATTERNS = ("uniform", "random")`，逐成员**独立随机方向**，
> `key` 参数默认 `jax.random.key(0)`，子 key 按 `sorted(block.names)` 的位置 fold in
> ——即 spec §1.5 原本的处方，也是 `check_linearity` 已有的构造。
>
> **为什么推翻**：`888cc8b`（两个确定性模式）在**两成员**块上正确，在**三成员**块上
> 失效——`a−c` 落在 sorted 位置 0 与 2，`uniform` 与 `alternating` 给它们**相同符号**，
> 于是读出 bitwise 0.0，与原缺陷同形，只是多一个成员。而分派器把**所有**合格隐变量
> 放进一个块，所以 ≥3 成员是常态。实测 `a−c`：`0.000000e+00` → `3.816786e-01`。
>
> **本任务正文里被证伪的三处推理**：
>
> 1. 「随机会让 yes/no 守卫的判决依赖 key」——**站不住**。`check_linearity` 就用随机
>    探针，固定默认 key 即确定可复现。这条顾虑换掉了一个覆盖更完整的设计。
> 2. 「`alternating` 是可证分离两成员对比的确定性方向」——两成员上对，但**不推广**。
>    Task 3b 把它作为第三个模式加回全部 13 个 fixture 行，**没有一个判决改变**，故删除。
> 3. `888cc8b` 的 docstring 声称二进制计数器族「一般地关掉它」——**假的，已实测**：
>    4 成员时 `uniform` 与两个计数位与 `a−b−c+d` 的内积**恰好为 0**。计数器只分离
>    **位置对**，张成 `1+ceil(log2 m)` 个方向中的 m 个。
>
> **一处比原设计更强的理由，是实测出来的**：`uniform` 不是「便宜的确定性锚点」。
> 「随机方向对任何非零**线性**泛函以概率 1 检出」这句话是对的、且几乎不相关——真实的
> σ 不是线性的。`one_sided_sigma` 的 σ 是单边截断的，只用 `random` 时两个探针在
> **400 个 key 中的 105 个（26%）**上双双落进平坦半空间，读出 bitwise 0.0——这是
> **检出失败**，不是幅度不足。加上 `uniform` 后 400 个 key 全部检出，最小值恰为其
> 无 key 的 `6.000000e+01`。
>
> **仍然存在的洞（方向已覆盖，幅度未覆盖）**：σ 在 O(1) 先验宽度内平坦、在其外才拐折
> 的情形对所有方向都不可见——`base + max(a−b−offset, 0)` 在 offset 0/1 处读
> 5.115e+00 / 1.782e+00，在 offset 3 与 10 处读 **0.0**。关掉它属于
> `DEPENDENCE_PROBES`（更大的倍数，或先验预测式探针）的决定，不是模式的决定。
>
> **一条未被杀死的变异，已在代码里具名记录**：`fold_in(key, index)` 改成常数索引
> （两个探针幅度共用一个方向）——数值变化最大 36 倍，但**没有任何判决改变**。诊断 (a)：
> 一个随机方向已以概率 1 检出每个非零线性泛函，第二个买的是余量不是检出。

**Files:**
- Modify: `src/bayesmith/exact/gls.py`（`DEPENDENCE_PROBES`、`check_prediction_dependence`）
- Modify: `tests/exact/models.py`（新增 `contrast_sigma_pair`）
- Test: `tests/exact/test_gls.py`

- [ ] **Step 1: 加 fixture**

```python
# tests/exact/models.py  （追加）
def contrast_sigma_pair(*, n=200, a_true=1.0, b_true=-0.5, base=0.3, seed=24):
    """sigma depends on a CONTRAST of two members that the mean cannot separate.

    `check_prediction_dependence` moved every member by the same signed
    multiple of its own prior width, so its probe never left the level set of
    `a - b` and it measured a movement of exactly 0.0 -- bitwise. The
    dispatcher then reads "sigma is constant" and picks plain `gcr`.

    Both latents sit on the SAME regressor, so `a + b` is all the mean knows
    and `a - b` is determined entirely by sigma. Measured with the lockstep
    probe: the contrast came back +0.0000 (sd 1.4142) against a long-NUTS
    +1.6038 (sd 0.0486) -- 33 NUTS sd out, width inflated 29x.

    The worst part is not the size of the error. This graph is
    whole-graph-one-block, so it takes the iid-draws-no-chain row of the
    dispatch table: no r-hat, no k-hat, no ESS, nothing to diagnose.
    """
    x = jnp.linspace(0.5, 2.0, n)
    truth = (a_true + b_true) * x
    noise = base * jnp.exp(a_true - b_true)
    data = truth + noise * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        a = sample("a", lambda: dist.Normal(0.0, 1.0))
        b = sample("b", lambda: dist.Normal(0.0, 1.0))
        mu = det("mu", lambda a_, b_, x_: (a_ + b_) * x_, a, b, xs, linear_in=("a", "b"))
        observe(
            "d",
            lambda m, a_, b_: dist.Normal(m, base * jnp.exp(a_ - b_)),
            mu, a, b, obs=data,
        )

    return trace(model)
```

- [ ] **Step 2: 写失败的测试**

```python
# tests/exact/test_gls.py  （追加）
def test_sigma_depending_on_a_contrast_of_two_members_is_detected():
    """The probe must not travel one ray through the block's domain.

    Measured with the lockstep probe `centre + factor * prior_std`: movement
    came back exactly 0.0 -- not small, BITWISE zero -- because both members
    were displaced by the same signed multiple of equal prior widths, so
    `a - b` never changed and sigma is constant along that ray.

    No function here was crafted to have a root at the probe points. The ray
    simply never leaves the level set, which is why "try another magnitude"
    does not help: every magnitude is on the same ray.
    """
    graph = contrast_sigma_pair()
    block = unchecked_operator(graph, ["a", "b"])
    movement = check_prediction_dependence(
        block, sigma_from_graph(graph, {}), declared=True
    )
    assert movement > 1e-3, (
        f"sigma moves with a - b, but the probe measured {movement:.3e}"
    )


def test_a_genuinely_constant_sigma_still_measures_no_movement():
    """The two-sided half: richer probe directions must not invent movement.

    Without this, a 'fix' that reported a large movement unconditionally
    would pass the test above and route every model through the correction
    machinery it does not need.
    """
    graph = two_linear_latents()
    block = unchecked_operator(graph, ["a", "b"])
    movement = check_prediction_dependence(
        block, sigma_from_graph(graph, {}), declared=True
    )
    assert movement == pytest.approx(0.0, abs=1e-12)
```

- [ ] **Step 3: 跑测试确认第一条红**

Run: `.venv/bin/python -m pytest tests/exact/test_gls.py -k "contrast or genuinely_constant" -v`

Expected: 第一条 **FAIL**（`movement` 为 `0.000e+00`），第二条 PASS。

- [ ] **Step 4: 改探测方向**

```python
# src/bayesmith/exact/gls.py
DEPENDENCE_PROBES: tuple[float, ...] = (1.0, -0.5)
"""Probe magnitudes, in units of each member's own prior width."""

DEPENDENCE_PATTERNS: tuple[str, ...] = ("uniform", "alternating")
"""Directions the magnitudes are applied along.

``uniform`` is the original single ray -- every member displaced by the same
signed multiple. ``alternating`` flips the sign with each member's position
in the SORTED names, which is what makes a sigma depending on a CONTRAST
visible: on the uniform ray the contrast is exactly constant, so the original
probe measured bitwise 0.0 and the dispatcher read "sigma does not move".

Two deterministic patterns rather than one random direction: a random
direction would almost surely work, but it makes a yes/no guard's verdict
key-dependent, and ``alternating`` is the direction that provably separates
the two-member contrast case. Cost goes from 2 to 4 ``sigma_of`` calls,
negligible beside the CG solves this guard protects.
"""


def _dependence_probe(block, centre, factor, pattern):
    """One displacement of the whole block, in units of each prior width."""
    ordered = sorted(block.names)
    return {
        name: centre[name]
        + factor
        * block.prior_std[name]
        * (-1.0 if (pattern == "alternating" and position % 2) else 1.0)
        for position, name in enumerate(ordered)
    }
```

`check_prediction_dependence` 的循环改成对 `(factor, pattern)` 的乘积迭代，其余不动。

> **为什么按 `sorted(names)` 而不是 `block.names`**：`block.names` 是调用方给的顺序，`sorted` 让同一个块无论成员怎么写都得到同一组探针——与 `check_linearity` 的 `ordered = sorted(names)` 一致。这个 dict 是**普通推导式**建的，不是 JAX 变换的产物，所以这个 `sorted` **承重**（P3a Task 9 的判据）。

- [ ] **Step 5: 跑测试与全套**

Run: `.venv/bin/python -m pytest tests/exact/test_gls.py -k "contrast or genuinely_constant" -v`
Then: `.venv/bin/python -m pytest tests/exact/ -q`

Expected: 两条都 PASS，全套全绿。

- [ ] **Step 6: 变异测试**

| 变异 | 必须变红 |
|---|---|
| `DEPENDENCE_PATTERNS` 改回 `("uniform",)` | `test_sigma_depending_on_a_contrast_of_two_members_is_detected` |
| `alternating` 的符号翻转去掉（恒为 `1.0`） | 同上 |
| `sorted(block.names)` 改成 `block.names` | 见 Step 7——**需要一个成员声明序 ≠ 排序的 fixture，否则这个变异是 no-op** |
| `movement` 的 `max` 改成 `min` | `test_sigma_depending_on_a_contrast_of_two_members_is_detected` |

- [ ] **Step 7: 子判据检查**

- **双侧**：`test_a_genuinely_constant_sigma_still_measures_no_movement` 是另一侧。
- **点 vs 区域**：扫 `n`、`base`、`a_true - b_true` 的大小、seed，报告 `movement > 1e-3` 成立的区间。**特别地：`a_true - b_true = 0` 时会怎样？** 那时真值处对比为零但 σ 仍随对比变化——探针从 `centre`（先验均值 0）出发，所以仍应检出。**实测确认**，并把结果写进 docstring。
- **结构维度**：成员数 {1, 2}（`radiometer` 是 1，`contrast_sigma_pair` 是 2）；成员声明序 {排序, 逆序}——**逆序今天没有，补一个**，否则 `sorted` 的变异抓不住。

- [ ] **Step 8: AST 比对 + 提交**

暂存 `src/bayesmith/exact/gls.py`、`tests/exact/test_gls.py`、`tests/exact/models.py`，提交信息：

```
fix: probe sigma dependence along more than one ray

check_prediction_dependence displaced every block member by the same signed
multiple of its own prior width, so its probe never left the level set of a
contrast between two equal-width members. Measured: bitwise 0.0 movement on
a graph whose sigma depends on exactly that contrast, so the dispatcher
reads "sigma is constant" and picks plain gcr -- contrast 33 NUTS sd out,
width inflated 29x, and because the graph is whole-graph-one-block it
returns iid draws with no r-hat, k-hat or ESS to notice with.

Adds an alternating-sign pattern rather than a random direction: random
would almost surely work but makes a yes/no guard's verdict key-dependent,
and alternating is the deterministic direction that provably separates the
two-member contrast. 2 -> 4 sigma_of calls.
```

---

## Task 4：`probe_gaussian` 关键字（两个调用点，三个调用方）

spec §1.2。`gibbs_fn` 在 trace 下运行，而 `check_gaussian` 用 `bool(jnp.all(...))` 与 `raise`。

**Files:**
- Modify: `src/bayesmith/exact/block.py`（`_env_before`、`unchecked_operator`）
- Test: `tests/exact/test_block.py`

- [ ] **Step 1: 写失败的测试**

```python
# tests/exact/test_block.py  （追加）
def test_unchecked_operator_refuses_to_trace_with_the_gaussian_probe_live():
    """The probe is concrete-valued by construction, so it cannot be traced.

    `check_gaussian` does `bool(jnp.all(...))` and `float(jnp.min(...))` and
    raises -- its own docstring says it runs outside any trace. This pins the
    fact rather than leaving a future reader to discover it from a stack
    trace inside a Gibbs sweep, which is where it would otherwise surface.
    """
    graph = two_linear_latents()

    def build(a_value):
        return unchecked_operator(graph, ["b"], at={"a": a_value}).offset

    with pytest.raises(jax.errors.TracerBoolConversionError):
        jax.jit(build)(jnp.asarray(0.5))


def test_probe_gaussian_false_makes_the_operator_traceable():
    """The other side, and the whole reason the keyword exists.

    A Gibbs sweep rebuilds the operator at the current `hmc_sites` every
    sweep, under jit. The Gaussian probe is a statement about the MODEL, not
    about the sweep, so it is checked once at compile time and disabled here.

    Both halves are asserted because only the pair proves the keyword is
    connected: a `probe_gaussian` that is accepted and ignored passes this
    test and fails the one above, and one that is honoured everywhere but
    inside `_env_before` passes neither.
    """
    graph = two_linear_latents()

    def build(a_value):
        return unchecked_operator(
            graph, ["b"], at={"a": a_value}, probe_gaussian=False
        ).offset

    offset = jax.jit(build)(jnp.asarray(0.5))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(offset))


def test_probe_gaussian_false_also_silences_the_per_member_probe():
    """`_env_before`'s per-MEMBER probe is a second call site, easily missed.

    Gating only `unchecked_operator`'s per-observed-node call leaves
    `block.py`'s `_env_before` calling `check_gaussian` for every block
    member, and the trace still dies. A PLATED member is used here so the two
    call sites cannot be confused by a fixture where the member and the
    observed node happen to be the same shape -- with a scalar member and a
    scalar observed node, a partial fix can look complete.
    """
    graph = plated_latent_through_deterministic()

    def build(scale):
        block = unchecked_operator(graph, ["z"], at={}, probe_gaussian=False)
        return block.offset["d"] * scale

    assert bool(jnp.all(jnp.isfinite(jax.jit(build)(jnp.asarray(2.0)))))
```

- [ ] **Step 2: 跑测试**

Run: `.venv/bin/python -m pytest tests/exact/test_block.py -k "probe_gaussian or refuses_to_trace" -v`

Expected: 第一条 PASS（今天就该抛），后两条 **FAIL**（`TypeError: unexpected keyword argument 'probe_gaussian'`）。

- [ ] **Step 3: 加关键字**

```python
# src/bayesmith/exact/block.py
def _env_before(
    graph: Graph,
    names: tuple[str, ...],
    at: dict[str, Any],
    *,
    probe_gaussian: bool = True,
) -> tuple[dict[str, Any], dict[str, tuple]]:
    # ... unchanged body, except the per-member probe:
    #         if probe_gaussian:
    #             check_gaussian(graph, node, env)
    ...


def unchecked_operator(
    graph: Graph,
    names: Iterable[str],
    at: dict[str, Any] | None = None,
    *,
    probe_gaussian: bool = True,
) -> LinearBlock:
    """Export ``A``, ``A^T``, the offset, the data and the prior -- **unchecked**.

    Args:
        probe_gaussian: run :func:`~bayesmith.exact.gaussian.check_gaussian`
            on every block member and every observed node. Default ``True``,
            which is what every P3a call site already gets.

            **Pass ``False`` only inside a traced Gibbs sweep, and only after
            the same check has run once at compile time.** The probe evaluates
            ``log_prob`` at concrete offsets and raises on a mismatch, so it
            does ``bool(jnp.all(...))`` -- a ``TracerBoolConversionError``
            under ``jit``, not a slow path. There are TWO call sites (per
            member in ``_env_before``, per observed node here) and disabling
            only one still dies.

            What each of this module's three entry points checks:

            ==================================  ==========  ===========
            entry point                         linearity   gaussianity
            ==================================  ==========  ===========
            ``linear_operator``                 yes         yes
            ``unchecked_operator``              no          yes
            ``unchecked_operator(..., False)``  no          no
            ==================================  ==========  ===========

    Note:
        Trace-safety here is a claim about THIS library's code, not about the
        graph it is handed. A ``Deterministic`` whose ``fn`` does
        ``float(x) > 0`` raises ``ConcretizationTypeError`` under ``jit`` no
        matter what this keyword says -- and a dispatcher runs arbitrary user
        ``fn`` under trace.
    """
    names = _validated_names(graph, names)
    at = _validated_at(graph, names, at)
    _refuse_internal_ancestry(graph, names)
    _refuse_missing_observed(graph)

    env, domain = _env_before(graph, names, at, probe_gaussian=probe_gaussian)
    if probe_gaussian:
        for observed in graph.observed:
            check_gaussian(graph, graph.node(observed), env)
    # ... rest unchanged
```

- [ ] **Step 4: 跑全套**

Run: `.venv/bin/python -m pytest tests/exact/ -q`

Expected: 全绿，三条新测试 PASS。**其余两个 `_env_before` 调用方（`linearity.py:295`、`tests/exact/oracle.py:111`）不传该关键字，拿到默认 `True`，行为逐位不变**——有意为之，它们都在具体值上跑。

- [ ] **Step 5: 变异测试**

| 变异 | 必须变红 |
|---|---|
| `probe_gaussian` 被接受但忽略（两处照跑 `check_gaussian`） | `test_probe_gaussian_false_makes_the_operator_traceable` |
| 只关掉 `unchecked_operator` 里那一处，不关 `_env_before` | `test_probe_gaussian_false_also_silences_the_per_member_probe` |
| 默认值改成 `False` | `test_unchecked_operator_refuses_to_trace_with_the_gaussian_probe_live`。~~以及 `test_gaussian.py` 里依赖默认探测的测试~~ —— **后半句是假的，已实测**：`test_gaussian.py` 对 `unchecked_operator` / `_env_before` / `linear_operator` **零引用**，十二条测试全部直接在手搭的 `env` 上调 `check_gaussian`，默认值根本不在它们的调用路径上 |
| **`_env_before` 的逐成员探针整段删掉** | `test_the_per_member_probe_catches_a_member_whose_log_prob_lies`（提交 `4fa9e94`）。**这条是补上的**：直到那次提交之前，删掉它整套仍 581 绿。jit 下每个 `check_gaussian` 都抛，所以 trace 那条测试由先跑的那个调用点满足（就是 `_env_before`）；真正非高斯的成员由下一行 `gaussian_parts` 抓。逐成员探针唯一独有的职责，是**类型读起来是 Normal 而 `log_prob` 不是**的成员——而 `LyingNormal` 此前只出现在观测节点上 |

- [ ] **Step 6: AST 比对 + 提交**

暂存 `src/bayesmith/exact/block.py`、`tests/exact/test_block.py`，提交信息：

```
feat: probe_gaussian keyword so a block can be built under trace

A Gibbs sweep rebuilds the operator at the current hmc_sites every sweep,
under jit, and check_gaussian is concrete-valued by construction -- its own
docstring says so. Measured: unchecked_operator under jax.jit raises
TracerBoolConversionError.

Two call sites, not one: per observed node here and per block member inside
_env_before. Gating only the first still dies, which is what the plated test
pins. The other two _env_before callers keep the default and are unchanged.

Trace-safety is a claim about this library's code only. A Deterministic whose
fn does float(x) > 0 still raises under jit, and the dispatcher runs
arbitrary user fn under trace -- said in the docstring rather than left to be
discovered from a stack trace inside a sweep.
```

---

## Task 5：`dispatch/classify.py` —— 合格性与分区

spec §三。**这是 P3b 的核心算法**，也是三个静默错误答案里两个的所在地。

**Files:**
- Create: `src/bayesmith/dispatch/__init__.py`
- Create: `src/bayesmith/dispatch/classify.py`
- Create: `tests/dispatch/__init__.py`
- Create: `tests/dispatch/test_classify.py`
- Modify: `tests/exact/models.py`（新增 `orphaned_child_latent`、`student_t_likelihood`、`lying_observed_node`）

- [ ] **Step 1: 加三个 fixture**

```python
# tests/exact/models.py  （追加）
def orphaned_child_latent(*, n=6, sigma=0.5, w_true=2.0, seed=25):
    """`w` is Gaussian and affine, but a DISQUALIFIED latent's density needs it.

    The partition rule's ejection clause originally read "z leaves if it is an
    ancestor of another QUALIFIED latent". `v` here is Student-t, so it fails
    criterion 1 and is not qualified -- and `w` therefore stayed in the block
    while the factor `p(v | w)` was dropped on the floor. `unchecked_operator`
    reads exactly two things, the block members' own priors and the observed
    nodes; every other density term in the graph is invisible to it.

    Measured with the qualified-only rule: mean(w) +0.4106 sd 1.7723 against
    a truth of +1.9759 / 0.4816 and a long-NUTS +2.0004 / 0.4809 -- 3.2 true
    sd out, width inflated 3.7x.

    `tests/exact/oracle.py::graph_oracle` reproduces the SAME wrong answer,
    because it reads the same two sources. The dense oracle cannot see this
    class of defect at all, which is why the guard has to be structural.
    """
    x = jnp.linspace(1.0, 2.0, n)
    key = jax.random.key(seed)
    data = w_true * x + sigma * jax.random.normal(key, (n,))
    v_data = jnp.asarray(2.0)

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 3.0))
        sample("v", lambda w_: dist.StudentT(3.0, w_, 0.4), w)
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def student_t_likelihood(*, n=6, sigma=0.5, w_true=1.5, seed=26):
    """The OBSERVED node is not Gaussian -- criterion 2.

    Every one of the 29 `observe()` calls in this module used `dist.Normal`
    before this fixture existed, so criterion 2 had no fixture and a
    classifier that simply never checked observed nodes would have passed the
    entire table. The latent-side criterion is already covered twice, by
    `overflowing_outside_latent`'s Cauchy and `improper_outside_prior`'s
    ImproperUniform.
    """
    x = jnp.linspace(1.0, 2.0, n)
    data = w_true * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 3.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.StudentT(4.0, m, sigma), mu, obs=data)

    return trace(model)


def lying_observed_node(*, n=6, sigma=0.5, w_true=1.5, seed=27):
    """The observed node's TYPE says Normal while its own log_prob does not.

    `LyingNormal` keeps `loc` and `scale` and changes the density, so
    introspection passes and the probe does not. The classifier must let the
    resulting StructureError THROUGH -- routing it to NUTS would hide a
    broken model behind an ordinary-looking fallback, which is precisely the
    distinction `errors.py` exists to preserve.

    Note this is the same exception TYPE that `check_linearity` raises for a
    false `linear_in`, and that one MUST be caught. The classifier therefore
    discriminates by raise SITE, not by exception type.
    """
    x = jnp.linspace(1.0, 2.0, n)
    data = w_true * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 3.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: LyingNormal(m, sigma), mu, obs=data)

    return trace(model)
```

- [ ] **Step 2: 写失败的测试（分类表）**

```python
# tests/dispatch/test_classify.py
"""The structural gate: what every fixture classifies as, and why.

Zero MCMC, zero statistics, fully deterministic. Every expectation below was
COMPUTED against the real `_ancestors` / `check_linearity` /
`check_prediction_dependence` before being written down -- the first draft of
this table had two wrong rows (`unconstrained_latent` and `shared_ancestor`),
both of which came from asserting rather than measuring.
"""
import jax.numpy as jnp
import pytest

from bayesmith.dispatch.classify import partition
from bayesmith.errors import StructureError
from tests.exact.models import (
    bilinear_pair, collinear_pair, cubic_tail, diamond_ancestor,
    indirect_ancestor, lying_observed_node, orphaned_child_latent,
    plated_latent, quadratic_claim, radiometer, radiometer_group,
    shared_ancestor, student_t_likelihood, two_linear_latents,
    unconstrained_latent,
)


@pytest.mark.parametrize(
    "build, exact, nuts, method",
    [
        (two_linear_latents, ("a", "b"), (), "gcr"),
        (radiometer, ("w",), (), "gcr+snis"),
        (radiometer_group, ("a", "b"), (), "gcr+snis"),
        (collinear_pair, ("a", "b"), (), "gcr"),
        (plated_latent, ("z",), (), "gcr"),
        # Criterion 3 is VACUOUSLY true for `u`: it reaches no observed node
        # at all, so the universal quantifier ranges over an empty set. Its
        # column of A is exactly zero and the answer is its prior mean, which
        # the exact path already handles -- see
        # test_a_latent_the_data_never_reaches_comes_back_at_its_prior_mean.
        (unconstrained_latent, ("u", "w"), (), "gcr"),
        # The ancestor rule ejects tau in all three, and criterion 2 of the
        # first draft would have inverted every one of them.
        (indirect_ancestor, ("x",), ("tau",), "gcr"),
        (diamond_ancestor, ("x",), ("tau",), "gcr"),
        (shared_ancestor, ("x",), ("tau",), "gcr"),
        # `w` is affine and Gaussian, but `v`'s density depends on it and `v`
        # is disqualified -- so `w` must leave too, or p(v|w) is dropped.
        (orphaned_child_latent, (), ("v", "w"), "nuts"),
        # Joint refusal: each conditional is affine, the pair is not.
        (bilinear_pair, (), ("gain", "t_ant"), "nuts"),
        (quadratic_claim, (), ("w",), "nuts"),
        (cubic_tail, (), ("w",), "nuts"),
        # Criterion 2: the OBSERVED node is not Gaussian.
        (student_t_likelihood, (), ("w",), "nuts"),
    ],
)
def test_partition_matches_the_hand_derived_answer(build, exact, nuts, method):
    result = partition(build())
    assert result.exact == exact
    assert result.nuts == nuts
    assert result.method == method


def test_a_refused_block_names_its_members_in_the_reason():
    """"Everything went to NUTS" is useless without saying which claim failed.

    A user one `linear_in` declaration away from an exact solve has to be able
    to see that from the plan, or the whole-block-falls-together policy is
    just an unexplained downgrade.
    """
    result = partition(bilinear_pair())
    assert "gain" in result.reason and "t_ant" in result.reason


def test_a_lying_observed_node_raises_rather_than_falling_back_to_nuts():
    """StructureError from check_gaussian must NOT be swallowed.

    `check_linearity` raises the same TYPE for a false linear_in, and that one
    must be caught and routed to NUTS -- so a classifier that discriminates by
    exception type instead of by raise site passes the bilinear_pair row above
    and silently downgrades this broken model. Both rows are needed; neither
    alone pins the distinction.
    """
    with pytest.raises(StructureError, match="log_prob"):
        partition(lying_observed_node())
```

- [ ] **Step 3: 跑测试确认它红**

Run: `.venv/bin/python -m pytest tests/dispatch/test_classify.py -v`
Expected: 全部 **FAIL**，`ModuleNotFoundError: No module named 'bayesmith.dispatch'`。

- [ ] **Step 4: 写 `classify.py`**

```python
# src/bayesmith/dispatch/classify.py
"""Which latents an exact method applies to, and how they group.

The three structural axes P1 recorded -- ``linear_in``, ``support``,
``depends_on_prediction`` -- are read here for the first time. P3a *verified*
``linear_in`` (it measured affinity); this module *reads* it, which is a
different thing: it walks the declaration along every path from a latent to
every observed node's location parameter.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from bayesmith.errors import NotGaussian, StructureError
from bayesmith.exact.block import _ancestors, _env_before, unchecked_operator
from bayesmith.exact.gaussian import check_gaussian
from bayesmith.exact.gls import check_prediction_dependence, sigma_from_graph
from bayesmith.exact.linearity import check_linearity
from bayesmith.graph.evaluate import evaluate
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Deterministic, Probabilistic

SIGMA_RTOL = 1e-8
"""Relative sigma movement above which a block counts as prediction-dependent."""


@dataclasses.dataclass(frozen=True)
class Classification:
    """The partition of one graph, and the evidence behind it.

    Attributes:
        exact: the single exact block, sorted. Empty if there is none.
        nuts: every other latent, sorted.
        method: ``"gcr"``, ``"gcr+snis"``, ``"gcr+mh"`` or ``"nuts"``.
        reason: why -- named members on a refusal, so a user one declaration
            away from an exact solve can see that from the plan.
        linearity: ``check_linearity``'s per-at-point errors, or ``None``.
        sigma_movement: the largest relative movement of sigma with the
            block, or ``None`` if there is no block.
        sigma_needs_rebuild: whether any observed node's scale has a latent
            ANCESTOR, in which case ``noise_std`` must be recomputed every
            sweep rather than hoisted. Distinct from ``sigma_movement``,
            which only sees movement with the BLOCK.
    """

    exact: tuple[str, ...]
    nuts: tuple[str, ...]
    method: str
    reason: str
    linearity: dict | None
    sigma_movement: float | None
    sigma_needs_rebuild: bool


def _observed_ancestors(graph: Graph) -> set[str]:
    """Every node that some observed node depends on."""
    out: set[str] = set()
    for name in graph.observed:
        out |= _ancestors(graph, name) | {name}
    return out


def _relevant_deterministics(graph: Graph, name: str) -> dict[str, set[str]]:
    """``{det_name: parents the path arrives by}`` for criterion 3.

    Walks FORWARD from ``name`` through ``Deterministic`` nodes only, and
    stops at every ``Probabilistic`` one. Stopping there is the point: a path
    ``x -> v -> mu -> d`` with ``v`` latent does not make ``x`` reach ``d``'s
    location deterministically -- ``isolate`` holds ``v`` fixed, so ``x`` does
    not reach ``d`` that way at all. (That path is not harmless, but it is
    :func:`partition`'s ancestor rule that handles it, not this.)

    Restricted to deterministics that are themselves ancestors of some
    observed node. One that leads nowhere contributes nothing to any
    prediction, so requiring a declaration from it would refuse a model for a
    node the solve never evaluates.
    """
    matters = _observed_ancestors(graph)
    reached = {name}
    arrivals: dict[str, set[str]] = {}
    for node in graph.nodes:  # declaration order IS topological order
        if not isinstance(node, Deterministic):
            continue
        incoming = {p for p in node.parents if p in reached}
        if not incoming:
            continue
        reached.add(node.name)
        if node.name in matters:
            arrivals[node.name] = incoming
    return arrivals


def _declares_linear_in(graph: Graph, name: str) -> tuple[bool, str]:
    """Criterion 3: EVERY Deterministic on EVERY path declares its in-edges."""
    for det_name, incoming in _relevant_deterministics(graph, name).items():
        undeclared = sorted(incoming - set(graph.node(det_name).linear_in))
        if undeclared:
            return False, (
                f"{det_name!r} declares linear_in="
                f"{graph.node(det_name).linear_in!r}, which does not name "
                f"{undeclared} -- and that node is on a path from {name!r} to "
                "an observed node's location"
            )
    return True, ""


def _is_gaussian(graph: Graph, name: str, env: dict[str, Any]) -> tuple[bool, str]:
    """Criterion 1 and 2. ``NotGaussian`` is a verdict; ``StructureError`` is not.

    ``NotGaussian`` means "this node is simply not a diagonal Gaussian",
    which is an ordinary property of an ordinary model and routes the node to
    NUTS. ``StructureError`` from :func:`check_gaussian` means the node's own
    ``log_prob`` contradicts the ``loc``/``scale`` read off it -- a broken
    model -- and is deliberately NOT caught here.
    """
    try:
        check_gaussian(graph, graph.node(name), env)
    except NotGaussian as exc:
        return False, str(exc).split(".")[0]
    return True, ""


def partition(graph: Graph, *, key: jax.Array | None = None) -> Classification:
    """Derive the exact block and its method from the graph's structure."""
    latents = list(graph.latents)
    env = evaluate(graph, {n: _prior_mean(graph, n) for n in latents})

    for observed in graph.observed:
        ok, why = _is_gaussian(graph, observed, env)
        if not ok:
            return Classification(
                (), tuple(sorted(latents)), "nuts",
                f"observed node {observed!r} is not a diagonal Gaussian: {why}",
                None, None, False,
            )

    qualified, why_not = [], {}
    for name in latents:
        ok, why = _is_gaussian(graph, name, env)
        if not ok:
            why_not[name] = why
            continue
        ok, why = _declares_linear_in(graph, name)
        if not ok:
            why_not[name] = why
            continue
        qualified.append(name)

    # Ejection: ancestor of ANY latent, qualified or not. The "qualified"
    # reading drops the factor p(child | member) silently -- see
    # `orphaned_child_latent`, and note the dense oracle reproduces the same
    # wrong answer because it reads the same two sources the operator does.
    ejected = {z for z in qualified if any(
        z in _ancestors(graph, other) for other in latents if other != z
    )}
    for z in ejected:
        why_not[z] = f"{z!r} is an ancestor of another latent's distribution"
    block = tuple(sorted(set(qualified) - ejected))

    if not block:
        return Classification(
            (), tuple(sorted(latents)), "nuts",
            "; ".join(f"{n}: {w}" for n, w in sorted(why_not.items())),
            None, None, False,
        )
    return _classify_block(graph, block, latents, why_not, key)
```

`_classify_block` 负责剩下的部分（`check_linearity` → 失败则整块 NUTS；`check_prediction_dependence` → 选方法；`sigma_needs_rebuild` 判定），拆出来是为了让 `partition` 保持在 50 行以内。`_prior_mean(graph, name)` 从 `_env_before(graph, (name,), {其余: 0})` 取第 3 项。

**`at` 与 `at_points`（spec §3.6）**：块是真子集时用块外隐变量的**先验均值**建 `at`；`at_points` 只从**高斯**块外隐变量的先验抽，其余用先验均值，退化写进 `reason`。**绝不传 `at_points=[at]`**。

- [ ] **Step 5: 跑测试**

Run: `.venv/bin/python -m pytest tests/dispatch/test_classify.py -v`
Expected: 全部 PASS。**任何一行不符，先按纪律 4 诊断**——这张表的每一行都是算出来的，不符意味着实现与算出来的判据有出入，而不是表错了。

- [ ] **Step 6: 变异测试**

| 变异 | 必须变红 |
|---|---|
| 弹出规则改回「另一个**合格**隐变量」 | `orphaned_child_latent` 那一行 |
| `_declares_linear_in` 改成「存在某个 det 点了 x 的名字」 | `plated_latent` 与 `unconstrained_latent` 两行（空真） |
| `_relevant_deterministics` 不在 `Probabilistic` 处停下 | ~~`orphaned_child_latent` 行~~ —— **这一行是假的，已实测**：该变异在 `orphaned_child_latent` 上、以及扫过的九个 fixture 上，**一个判决都不改**。经**隐**节点离开的路径使其源头成为另一个隐变量的祖先，弹出规则已经把它移走，两种读法殊途同归。只有**观测**中间节点能分开两者（数据不是隐变量）——`observation_reused_downstream`（提交 `ca66750`）是那个图，加上它变异红一行、去掉它红零行 |
| 去掉 `matters` 交集（不限定观测节点的祖先） | **今天没有 fixture 能抓**——补一个带「悬空 Deterministic」的图 |
| 观测节点的高斯检查整段删掉 | `student_t_likelihood` 行 |
| `except NotGaussian` 改成 `except (NotGaussian, StructureError)` | `test_a_lying_observed_node_raises_rather_than_falling_back_to_nuts` |
| `check_linearity` 的 `StructureError` 不捕获 | `bilinear_pair` / `quadratic_claim` / `cubic_tail` 三行 |

- [ ] **Step 7: 子判据检查**

- **结构维度**：观测节点数 {1, 2}（`radiometer_group` 是 2）；plate {无, 有}；块大小 {0, 1, 2}——**≥3 今天没有，补一个 `tau → x → y` 的三隐变量链**，它同时检验弹出规则是跑一遍还是跑到不动点；观测节点声明序 {排序, 逆序}——**补 `two_observations_reverse_sorted_names` 一行**。
- **点 vs 区域**：`SIGMA_RTOL` 两侧各要一个 fixture，且报告 `radiometer` 的 `sigma_movement`（实测 2.50e+03）与 `two_linear_latents` 的（0.0）之间没有 fixture 落在 `1e-8` 附近——**如实写明这是覆盖两端而非边界验证**，与 `check_prediction_dependence` 自己的 docstring 一致。

- [ ] **Step 8: AST 比对 + 提交**

提交信息：

```
feat: derive the exact block and its method from the graph

Reads linear_in along every path rather than merely verifying affinity,
which is what P3a did. Three things the first draft got wrong, all of them
found by adversarial review and re-verified before being changed:

- The qualification criterion "x's prior's parents contain no other
  CANDIDATE member" was circular, and on shared_ancestor produced the exact
  inverse of the intended partition: block={tau}, whose A is identically
  zero, so the "exact" answer was tau's prior forever. Deleted.
- The ejection rule must range over ANY latent, not any QUALIFIED one.
  Disqualify a latent whose density depends on a block member and the factor
  p(child|member) is dropped silently -- measured 3.2 true sd out, width
  inflated 3.7x -- and graph_oracle reproduces the same wrong answer because
  it reads the same two sources.
- Criterion 3 is a universal quantifier over paths, so it is vacuously true
  both for a latent with no Deterministic on its path and for one with no
  path at all. Both must be ACCEPTED; the obvious spelling rejects both.

StructureError is discriminated by raise site, not by type: check_linearity
raises it for a false linear_in and that must route to NUTS, while
check_gaussian raises it for a node whose log_prob contradicts its own type
and that must propagate.
```

---

## Task 6：`dispatch/plan.py` —— `Block` 与可打印的 `InferencePlan`

> **测试里的导入约定**：公开名是 `bayesmith.compile`，它遮蔽内置 `compile`，所以每个测试文件顶部写 `from bayesmith import compile as compile_graph`，正文一律用 `compile_graph`。不要在测试里直接写 `compile(...)`——遮蔽内置函数在模块作用域是既定 UX，在测试文件里只是可读性损失。

spec §六。**这是这个包最重要的用户体验**：模型在被拟合之前，先说明它将如何被拟合，以及为什么。

**Files:**
- Create: `src/bayesmith/dispatch/plan.py`
- Test: `tests/dispatch/test_plan.py`

- [ ] **Step 1: 写失败的测试**

```python
# tests/dispatch/test_plan.py
def test_the_printed_plan_carries_kappa_and_the_tol_derived_from_it():
    """Both numbers, or the discipline they encode cannot be checked.

    Section 4.2's rule is that turning the convergence guard off inside a
    sweep REQUIRES tightening `tol` in the same breath -- rheplicant names
    "leave tol at its default and the guard off" as the combination that
    returned a silently over-confident posterior. The plan is where a reader
    checks that the pair was actually chosen together, so printing one
    without the other makes the rule unverifiable.
    """
    text = str(compile_graph(two_linear_latents()))
    assert "kappa" in text
    assert "tol" in text
    kappa = float(re.search(r"kappa=([0-9.e+-]+)", text).group(1))
    tol = float(re.search(r"tol=([0-9.e+-]+)", text).group(1))
    assert tol == pytest.approx(1e-3 / kappa, rel=1e-6)


def test_a_refused_block_prints_why_not_just_that_it_was_refused():
    """"NUTS" with no reason is indistinguishable from "no exact structure"."""
    text = str(compile_graph(bilinear_pair()))
    assert "NUTS" in text
    assert "gain" in text and "t_ant" in text


def test_the_plan_names_the_execution_it_will_use():
    """A mixed graph runs HMCGibbs; a fully exact one runs no chain at all.

    The distinction is the product: 'iid draws, no chain' and 'HMCGibbs over
    these sites' are different enough that a user must not have to infer
    which one they got.
    """
    assert "HMCGibbs" in str(compile_graph(indirect_ancestor()))
    assert "no chain" in str(compile_graph(two_linear_latents()))
```

- [ ] **Step 2: 跑，确认红**

Run: `.venv/bin/python -m pytest tests/dispatch/test_plan.py -v`
Expected: `ImportError` / `ModuleNotFoundError`。

- [ ] **Step 3: 写 `plan.py`**

```python
# src/bayesmith/dispatch/plan.py
CONVERGENCE_TARGET = 1e-3
"""Relative error the in-sweep `tol` is chosen to deliver: tol = target / kappa."""


class Block(eqx.Module):
    """One group of latents and the method the graph selected for it."""

    latents: tuple[str, ...] = eqx.field(static=True)
    method: str = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    linearity: dict | None = eqx.field(static=True, default=None)
    kappa: float | tuple[float, float] | None = eqx.field(static=True, default=None)
    tol: float | None = eqx.field(static=True, default=None)


class InferencePlan(eqx.Module):
    """What `compile` produces: a partition, its reasons, and how to run it."""

    graph: Graph
    blocks: tuple[Block, ...]
    sigma_needs_rebuild: bool = eqx.field(static=True, default=False)

    def __str__(self) -> str:
        lines = []
        for index, block in enumerate(self.blocks):
            members = "{" + ", ".join(block.latents) + "}"
            head = f"block {index}  {members:<16s} {_LABELS[block.method]:<18s}"
            lines.append(head + _evidence(block))
            for extra in _continuation(block):
                lines.append(" " * len(head) + extra)
        lines.append("execution: " + self._execution())
        return "\n".join(lines)
```

`_LABELS` 把 `"gcr"` 映到 `"GCR exact"`、`"gcr+snis"` 到 `"GCR + SNIS"`、`"gcr+mh"` 到 `"GCR + MH accept"`、`"nuts"` 到 `"NUTS"`。`_evidence(block)` 打印 `linear_in ✓ N scales x M at-points (max ...)`；`_continuation` 打印 `kappa=... -> tol=..., guard hoisted` 与 `reason` 的换行部分。`_execution()` 在全精确时返回 `"iid draws, no chain"`，混合时返回 `"HMCGibbs(inner=NUTS, gibbs_sites=[...])"`，全 NUTS 时返回 `"NUTS"`。

**κ 是区间时**（spec §4.2：块外隐变量影响 κ）打印 `kappa in [lo, hi]`，且 `tol` 用 **`hi`** 导出——用 `lo` 会给出一个太松的 `tol`，方向恰是让 CG 早停、后验变窄而守卫沉默的那一侧。

- [ ] **Step 3b: 再加一条——κ 随块外隐变量漂移时必须打印区间**

```python
def test_a_kappa_that_moves_with_an_outside_latent_is_printed_as_an_interval():
    """A kappa pinned at the prior mean is too SMALL exactly where it matters.

    Section 3.2's ancestor rule creates this shape on purpose: `x`'s prior
    width is a function of `tau`, and `tau` moves every sweep. Measured on
    `indirect_ancestor`, condition_bound runs

        tau  = 0.0   1.0    2.0 (prior mean)   4.0     6.0
        kappa= 1.57  69.7   2.51e2             9.56e2  2.11e3

    so pinning at the prior mean understates it 8.4x at tau=6 -- and the
    error is in the dangerous direction: `tol` comes out too LOOSE, CG stops
    early, the posterior comes back too narrow, and the in-sweep guard is off
    so nothing notices. That is verbatim the rheplicant combination section
    4.2 quotes ("leave tol at its default and the guard off").

    `tol` is therefore derived from the interval's UPPER end, and the
    interval is what gets printed.
    """
    text = str(compile_graph(indirect_ancestor()))
    match = re.search(r"kappa in \[([0-9.eE+-]+), ([0-9.eE+-]+)\]", text)
    assert match, f"expected an interval, got:\n{text}"
    lo, hi = (float(v) for v in match.groups())
    assert hi > 3 * lo, "the interval collapsed to a point; the sweep is unguarded"
    tol = float(re.search(r"tol=([0-9.eE+-]+)", text).group(1))
    assert tol == pytest.approx(1e-3 / hi, rel=1e-6)
```

- [ ] **Step 4-5: 实现到测试通过，跑全套**

Run: `.venv/bin/python -m pytest tests/dispatch/ tests/exact/ -q`

- [ ] **Step 6: 变异测试**

| 变异 | 必须变红 |
|---|---|
| `tol` 用 `CONVERGENCE_TARGET * kappa` 而非除 | `test_the_printed_plan_carries_kappa_and_the_tol_derived_from_it` |
| κ 区间用 `lo` 导出 `tol` | `test_a_kappa_that_moves_with_an_outside_latent_is_printed_as_an_interval` |
| `reason` 不打印成员名 | `test_a_refused_block_prints_why_not_just_that_it_was_refused` |
| 全精确图也打印 `HMCGibbs` | `test_the_plan_names_the_execution_it_will_use` |

- [ ] **Step 7: AST 比对 + 提交**（`feat: InferencePlan, printable`）

---

## Task 7：`exact/correct.py` —— 重要性权重、SNIS、Kish ESS、k̂

spec §5.1、§5.4。**符号在这里最容易写反，而 P3 spec 与 P3b 初稿都写反了。**

**Files:**
- Create: `src/bayesmith/exact/correct.py`
- Test: `tests/exact/test_correct.py`

- [ ] **Step 1: 写失败的测试**

```python
# tests/exact/test_correct.py
def test_log_weight_equals_log_p_minus_log_q_against_a_dense_gaussian():
    """The sign of the quadratic term, pinned against scipy.

    Both P3's spec and P3b's first draft wrote
    `C = ½ log det M − (n/2) log 2π`, which has BOTH signs inverted:
    `log q(y) = −½rᵀMr + ½ log det M − (n/2) log 2π`, so in
    `log w = log p − log q` the quadratic and the log-det must carry
    OPPOSITE signs. Measured with the draft's C: error +2.396e-01. With the
    corrected one: bitwise zero.

    This test compares the DIFFERENCE between two draws' log weights, which
    is the only thing either consumer uses -- self-normalisation and the MH
    ratio both cancel any constant. That makes it insensitive to C by
    construction, so `test_the_weight_constant_cancels_between_draws` below
    is what pins the cancellation itself, and this one pins the quadratic.
    """
    with jax.enable_x64(True):
        graph = radiometer()
        block = unchecked_operator(graph, ["w"])
        sigma = noise_std_at(graph, {"w": jnp.asarray(0.0)})
        oracle = graph_oracle(graph, ["w"])
        mu = {"w": jnp.asarray(oracle.mean[0])}
        draws = {"w": jnp.asarray([0.4, 1.1, 2.7])}

        got = jax.vmap(
            lambda x: log_weight(graph, block, {"w": x}, at={},
                                 noise_std=sigma, mu=mu)
        )(draws["w"])

        precision = oracle.precision[0, 0]
        expect = []
        for x in [0.4, 1.1, 2.7]:
            lp = float(log_joint(graph, {"w": jnp.asarray(x)}))
            lq = (-0.5 * precision * (x - oracle.mean[0]) ** 2
                  + 0.5 * np.log(precision) - 0.5 * np.log(2 * np.pi))
            expect.append(lp - lq)
        expect = np.asarray(expect)
        assert np.allclose(got - got[0], expect - expect[0], rtol=1e-10)


def test_kish_ess_is_one_when_a_single_draw_carries_all_the_weight():
    """The degenerate end, which is where SNIS actually fails.

    Measured in the spec's dimension sweep: a mild radiometer at n=500 gives
    Kish ESS 1.00 out of 40000 draws. This is not a pathological input, it is
    what self-normalised importance sampling does as the number of mismatched
    coordinates grows, and the dispatcher has to be able to see it.
    """
    log_w = jnp.asarray([0.0, -400.0, -400.0, -400.0])
    weights, ess = self_normalise(log_w)
    assert float(ess) == pytest.approx(1.0, abs=1e-6)
    assert float(jnp.max(weights)) == pytest.approx(1.0, abs=1e-9)


def test_kish_ess_is_n_when_every_weight_is_equal():
    """The other end. Without it, `ess = 1.0` unconditionally passes above."""
    weights, ess = self_normalise(jnp.zeros(7))
    assert float(ess) == pytest.approx(7.0, rel=1e-9)


@pytest.mark.parametrize("seed, n, lo, hi", [(0, 2000, -0.05, 0.20),
                                             (1, 2000, -0.05, 0.20)])
def test_khat_pins_the_private_numpyro_entry_point(seed, n, lo, hi):
    """`_psis_khat` is private, so its existence is pinned deliberately.

    The RANGE, not a point: measured over 20 seeds the Gaussian case has mean
    0.076, sd 0.066 and spans -0.017..0.184, and it is strongly N-dependent
    (N=200 -> 0.143, N=20000 -> 0.0008). An earlier draft pinned 0.184, which
    is the MAXIMUM over those 20 seeds -- a test that would fail on almost
    any other seed. seed and N are parameters here for exactly that reason.
    """
    log_w = jax.random.normal(jax.random.key(seed), (n,)) * 0.5
    assert lo < khat(log_w) < hi


def test_khat_is_none_rather_than_an_exception_when_the_private_entry_is_gone():
    """A private upstream name disappearing must degrade, not crash."""
    with mock.patch.dict(sys.modules, {"numpyro.infer.importance": None}):
        assert khat(jnp.zeros(100)) is None
```

- [ ] **Step 2: 跑，确认红**

- [ ] **Step 3: 写 `correct.py`**

```python
# src/bayesmith/exact/correct.py
"""Correcting a frozen-sigma proposal back to the exact conditional.

Freezing sigma makes each step a linear-Gaussian problem, which is the whole
reason GCR applies -- and it is also exactly the step `gls.py` names as the
reason its fixed point is the GLS optimum rather than the likelihood's:
"the log-determinant's dependence on the solution is held fixed rather than
differentiated". The importance weight below contains precisely the two terms
that were frozen. This is not a patch on an approximation; it is putting a
recorded discrepancy back.
"""


def log_weight(graph, block, x, *, at, noise_std, mu):
    """``log p(x, z, d) - log q(x)``, up to a constant common to every draw.

    ``q = N(mu, M^-1)`` with ``M = A^T N^-1 A + S^-1`` at the FROZEN sigma, so

        log w = log_joint(graph, {**at, **x}) + ½ (x-mu)^T M (x-mu) + C ,
        C = -½ log det M + (n/2) log 2pi .

    ``C`` is dropped. It is identical for every draw **because sigma is frozen
    at a value that does not depend on x** -- see :func:`mh_log_ratio`, where
    that independence is what makes the cancellation real rather than assumed.
    Both consumers (self-normalisation, the MH ratio) cancel it.

    The sign matters and is easy to get backwards: the quadratic term and the
    log-det term carry OPPOSITE signs, so a ``C`` with ``+½ log det M`` is
    wrong under every convention, not merely a different one. Measured against
    scipy: the inverted form is off by 2.396e-01 on a 4-dimensional case.

    Costs one application of ``M`` (a JVP plus a VJP) and one graph scan per
    draw. Both `vmap` cleanly.
    """
    operator = normal_operator(block, _weights(noise_std), variance_parts(block))
    delta = jax.tree.map(jnp.subtract, x, mu)
    quadratic = 0.5 * sum(
        jnp.sum(delta[name] * operator(delta)[name]) for name in delta
    )
    return log_joint(graph, {**at, **x}) + quadratic


def self_normalise(log_weights):
    """``(normalised weights, Kish ESS)``. ESS is ``1 / sum(w^2)``."""
    weights = jax.nn.softmax(log_weights)
    return weights, 1.0 / jnp.sum(weights**2)


def khat(log_weights):
    """PSIS k-hat, or ``None`` if numpyro's private entry point is gone.

    ``numpyro.infer.importance._psis_khat`` is private, so this is pinned by
    a test rather than trusted. It cannot be jitted -- not because it returns
    a Python float (that is a symptom) but because it does ``np.sort`` and
    then indexes with a boolean mask whose output shape depends on the data,
    which has no jit expression at all. The whole diagnostic therefore runs
    outside any trace, which costs nothing: SNIS has no loop to jit.
    """
    try:
        from numpyro.infer.importance import _psis_khat
    except (ImportError, AttributeError):
        return None
    return float(_psis_khat(np.asarray(log_weights)))


def unreliable(khat_value, n):
    """PSIS's reliability threshold, which is sample-size dependent.

    ``min(1 - 1/log10(N), 0.7)`` (Vehtari et al. 2024) -- BELOW 0.7 for
    N < 1e4, so a hard-wired 0.7 is optimistic exactly where SNIS is run
    without a chain and N is the caller's choice. Note also that the moment
    bands are ``k < 0.5`` for finite variance and ``k < 1`` for a finite
    MEAN; 0.7 is an empirical reliability threshold, not where the mean
    stops existing, and an earlier draft conflated the two.
    """
    if khat_value is None:
        return False
    return khat_value >= min(1.0 - 1.0 / np.log10(max(n, 11)), 0.7)
```

- [ ] **Step 4-5: 实现到测试通过，跑全套**

- [ ] **Step 6: 变异测试**

| 变异 | 必须变红 |
|---|---|
| `quadratic` 的 `0.5` 改成 `-0.5` | `test_log_weight_equals_log_p_minus_log_q_against_a_dense_gaussian` |
| `quadratic` 整项删掉 | 同上 |
| `self_normalise` 的 ESS 改成 `len(log_weights)` | `test_kish_ess_is_one_when_a_single_draw_carries_all_the_weight` |
| `unreliable` 的阈值写死 `0.7` | **需要一个 N < 1e4 且 k̂ 落在两阈值之间的参数化**——补一条 |
| `khat` 的 `except` 去掉 | `test_khat_is_none_rather_than_an_exception_when_the_private_entry_is_gone` |

- [ ] **Step 7: 子判据 + AST 比对 + 提交**（`feat: importance weights, Kish ESS and PSIS k-hat`）

**结构维度**：`log_weight` 在「块成员数」「叶内元素数」两维上都走 `jax.tree.map` 与求和——两维都要取到两个值，否则单成员标量 fixture 会让一个只处理首个叶子的实现全绿（P3a Task 7 的原样重演）。

---

## Task 8：`exact/gibbs.py` —— `gibbs_fn` 工厂、HMCGibbs 装配、独立提议 MH

spec §四、§5.3。**§5.3 是整份 spec 里最大的一处更正**，实现必须照更正后的写。

**Files:**
- Create: `src/bayesmith/exact/gibbs.py`
- Modify: `tests/exact/models.py`（新增 `steep_radiometer`、`mixed_radiometer`）
- Test: `tests/exact/test_gibbs.py`

- [ ] **Step 1: 加两个 fixture**

```python
# tests/exact/models.py  （追加）
def steep_radiometer(*, n=6, kappa=0.5, floor=1e-2, prior_std=10.0, w_true=1.5, seed=28):
    """A sigma steep enough that the MH and SNIS corrections are CHEAP to test.

    Not a cosmetic choice. On the default `radiometer`, the subtle mutation
    "evaluate the reverse density at x instead of rebuilding it at x'" shifts
    the posterior mean by 0.018 posterior sd at an acceptance rate of 0.955
    against a correct 0.950 -- exactly the "acceptance rate looks normal"
    failure §5.3 warns about -- and needs ~12,000 draws to reach 2 sigma,
    ~75,000 for 5 sigma, at 3 CG solves each. That is unavoidably `slow`, and
    it would be the ONLY guard for that bug, which is the fourth
    sub-criterion firing.

    Measured here instead: bias -0.2415 posterior sd, sd ratio 0.738,
    acceptance 0.64, and 2 sigma at **66 draws** -- about 430 for 5 sigma.
    Not slow.

    The bare-GCR/SNIS gap is measured on the same fixture at 16.8 sigma,
    against 2.1 sigma on `radiometer`'s defaults and 0.2 sigma one parameter
    away (n=40) -- so one fixture closes both gaps, and the "otherwise the
    correction is empty" clause stops being vacuous.
    """
    x = jnp.linspace(1.0, 3.0, n)
    truth = w_true * x
    scale = kappa * jnp.abs(truth) + floor
    data = truth + scale * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, prior_std))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda u: dist.Normal(u, kappa * jnp.abs(u) + floor), mu,
                depends_on_prediction=True, obs=data)

    return trace(model)


def mixed_radiometer(*, n=8, kappa=0.4, floor=1e-2, w_true=1.5, beam_true=0.3, seed=29):
    """A sigma-dependent EXACT block beside a latent that must go to NUTS.

    The only fixture in this module that can produce `gcr+mh`. Every other
    sigma-dependent graph here -- `radiometer`, `radiometer_group`,
    `plated_radiometer`, `one_sided_sigma` -- is whole-graph-one-block, so
    the dispatch table routes all of them to `gcr+snis`. Without this
    fixture, DELETING the entire MH branch leaves every structural row green.

    `beam` enters through `exp`, so `mu` is not affine in it and `linear_in`
    names only `w`.
    """
    x = jnp.linspace(1.0, 3.0, n)
    truth = jnp.exp(beam_true) * w_true * x
    scale = kappa * jnp.abs(truth) + floor
    data = truth + scale * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        beam = sample("beam", lambda: dist.Normal(0.0, 0.5))
        w = sample("w", lambda: dist.Normal(0.0, 6.0))
        mu = det("mu", lambda w_, b_, x_: jnp.exp(b_) * w_ * x_, w, beam, xs,
                 linear_in=("w",))
        observe("d", lambda u: dist.Normal(u, kappa * jnp.abs(u) + floor), mu,
                depends_on_prediction=True, obs=data)

    return trace(model)
```

- [ ] **Step 2: 写失败的测试**

```python
# tests/exact/test_gibbs.py
def test_gibbs_fn_uses_the_hmc_sites_it_was_handed():
    """Pins `at`, without the oracle sharing the choice.

    §7.2(a) inherits P3a's shared-layer blind spot -- `graph_oracle` and
    `unchecked_operator` both read `_env_before`, `isolate` and
    `observation_parts`, so all six P3a mutations to that layer move BOTH
    sides and the comparison never moves. It also adds a NEW shared
    parameter: the test must hand the oracle the same `at` it fixed the
    `hmc_sites` to.

    So this asserts a DIFFERENCE instead: run `gibbs_fn` at two different
    hmc_sites, NEITHER equal to any prior mean, and check the draw's mean
    moves by the amount the oracle predicts for that shift. A `gibbs_fn`
    that ignores `hmc_sites` and rebuilds `at` from prior means -- which is
    what §3.6 tells the implementer to do at COMPILE time, so the confusion
    is pre-installed -- produces no shift at all and fails here.
    """
    with jax.enable_x64(True):
        graph = mixed_radiometer()
        fn = gibbs_factory(graph, ("w",), tol=1e-10)
        means = []
        for beam in (0.35, -0.42):          # neither is beam's prior mean (0.0)
            keys = jax.random.split(jax.random.key(0), 4000)
            draws = jax.vmap(
                lambda k, b=beam: fn(
                    rng_key=k, gibbs_sites={"w": jnp.asarray(0.0)},
                    hmc_sites={"beam": jnp.asarray(b)},
                )["w"]
            )(keys)
            means.append(float(jnp.mean(draws)))
        predicted = [
            graph_oracle(graph, ["w"], at={"beam": jnp.asarray(b)}).mean[0]
            for b in (0.35, -0.42)
        ]
        assert means[0] - means[1] == pytest.approx(
            predicted[0] - predicted[1], rel=2e-2
        )


def test_gibbs_fn_freezes_sigma_where_it_says_it_does():
    """Pins the freeze point, with BOTH arms named.

    `graph_oracle`'s `sigma_at` defaults to the block's ZERO while
    `_env_before` centres members at their PRIOR MEAN -- two different freeze
    points inside one file. Whatever the test author passes for `sigma_at` is
    what makes the comparison agree, so a mutation that freezes sigma at the
    wrong point is absorbed unless the WRONG arm is asserted too.

    The wrong arm is taken at 3.0, NOT at zero: `steep_radiometer`'s prior
    mean IS 0.0, so a zero arm would coincide with the correct one and
    assert nothing -- discipline 3's mutation face.
    """
    with jax.enable_x64(True):
        graph = steep_radiometer()
        fn = gibbs_factory(graph, ("w",), tol=1e-10)
        keys = jax.random.split(jax.random.key(1), 6000)
        draws = jax.vmap(
            lambda k: fn(rng_key=k, gibbs_sites={"w": jnp.asarray(0.0)},
                         hmc_sites={})["w"]
        )(keys)
        # The freeze point the factory documents: sigma at the block's own
        # PRIOR MEAN given `at` -- an x-INDEPENDENT choice, which is what
        # makes the proposal a genuine independence proposal. Written out
        # here rather than imported from the implementation, so a mutation
        # that moves the freeze point cannot move the expectation with it.
        declared = graph_oracle(graph, ["w"], sigma_at={"w": jnp.asarray(0.0)})
        wrong = graph_oracle(graph, ["w"], sigma_at={"w": jnp.asarray(3.0)})
        got = float(jnp.std(draws))
        assert got == pytest.approx(float(np.sqrt(declared.covariance[0, 0])), rel=3e-2)
        assert got != pytest.approx(float(np.sqrt(wrong.covariance[0, 0])), rel=3e-2)


def test_gibbs_fn_survives_a_trace_and_the_probe_still_bites():
    """§1.2's whole reason, exercised. Two-sided, milliseconds, no MCMC.

    §7.2(a) most likely runs on concrete values, and then a `probe_gaussian`
    that is accepted and ignored is GREEN -- concrete `check_gaussian` works
    fine. Only opening a trace tells the two apart.
    """
    graph = mixed_radiometer()
    fn = gibbs_factory(graph, ("w",), tol=1e-8)
    out = jax.jit(lambda k, b: fn(rng_key=k, gibbs_sites={"w": jnp.asarray(0.0)},
                                  hmc_sites={"beam": b}))(
        jax.random.key(0), jnp.asarray(0.2))
    assert bool(jnp.isfinite(out["w"]))


def test_gibbs_fn_is_called_by_keyword_and_returns_exactly_the_block():
    """Pins numpyro's contract, which is a keyword one.

    `hmc_gibbs.py:168` calls `self._gibbs_fn(rng_key=..., gibbs_sites=...,
    hmc_sites=...)`, so the PARAMETER NAMES are the contract and the order is
    free -- and a positional-only `/` in the signature raises TypeError.
    Returning a subset raises an AssertionError with an EMPTY message that
    names nothing, and an extra key naming a NUTS latent is SILENTLY ignored,
    so 'exactly' is asserted rather than 'at least'.

    [EXPERIMENTAL INTERFACE] upstream: this makes a change there a deliberate
    decision instead of a surprise.
    """
    graph = mixed_radiometer()
    fn = gibbs_factory(graph, ("w",), tol=1e-8)
    params = inspect.signature(fn).parameters
    assert set(params) == {"rng_key", "gibbs_sites", "hmc_sites"}
    assert all(p.kind is not p.POSITIONAL_ONLY for p in params.values())
    out = fn(rng_key=jax.random.key(0), gibbs_sites={"w": jnp.asarray(0.0)},
             hmc_sites={"beam": jnp.asarray(0.2)})
    assert set(out) == {"w"}
```

- [ ] **Step 3: 跑，确认红**

- [ ] **Step 4: 写 `gibbs.py`**

```python
# src/bayesmith/exact/gibbs.py
"""The `gibbs_fn` numpyro's HMCGibbs calls, and the assembly around it.

`HMCGibbs`'s own docstring says "it is the user's responsibility to provide a
correct implementation of `gibbs_fn` that samples from the corresponding
posterior conditional". **That sentence is this package's product.** NumPyro
supplies the scaffolding and cannot know which conditionals are exactly
solvable, nor solve them.
"""


def gibbs_factory(graph, names, *, tol, method="gcr", sigma_rebuild=False,
                  maxiter=None):
    """Build the callable `HMCGibbs` invokes once per sweep.

    Args:
        tol: CG tolerance, chosen at compile time as
            ``CONVERGENCE_TARGET / kappa``. The in-sweep convergence guard is
            OFF -- it costs 12 extra operator applications and cannot raise
            under trace anyway -- so this number is the only thing standing
            between the sweep and a silently over-narrow posterior. Never
            leave it at a default with the guard off.
        method: ``"gcr"`` for a genuine conditional draw, ``"gcr+mh"`` when
            sigma depends on the block itself and the frozen-sigma draw is
            only a PROPOSAL.
        sigma_rebuild: recompute ``noise_std`` inside the sweep. Required
            whenever any observed node's scale has a latent ancestor --
            including one OUTSIDE the block, which
            ``check_prediction_dependence`` cannot see because it only moves
            block members. Measured: freezing sigma on such a graph gave a
            posterior 17x too narrow.

    The returned function takes ``rng_key``, ``gibbs_sites`` and
    ``hmc_sites`` **by keyword** -- that is numpyro's actual calling
    convention -- and returns exactly ``names``, in constrained space.
    """

    def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        at = {k: v for k, v in hmc_sites.items() if k in graph.latents}
        block = unchecked_operator(graph, names, at=at, probe_gaussian=False)
        sigma = _frozen_sigma(graph, block, at, sigma_rebuild)
        if method == "gcr":
            draw, _ = gcr_sample(block, noise_std=sigma, key=rng_key, tol=tol,
                                 maxiter=maxiter, require_convergence=None)
            return {name: draw[name] for name in names}
        return _mh_step(graph, block, gibbs_sites, at, sigma, rng_key, tol,
                        maxiter, names)

    return gibbs_fn
```

**`_mh_step` 必须按 spec §5.3 更正后的形态写**：

```python
def _mh_step(graph, block, current, at, sigma, key, tol, maxiter, names):
    """Independence-proposal Metropolis, so that log det M genuinely cancels.

    The spec's first draft froze sigma at ``sigma(m(x))`` -- the CURRENT
    state -- and rebuilt the proposal at ``x'`` to get a reverse density.
    That is where it went wrong: rebuilding makes ``M' != M``, so the ratio
    carries ``½(log det M' - log det M)``, which is nonzero exactly when
    sigma moved, i.e. in every case path (B) exists for. And ``log det`` of an
    implicit operator is the ONE quantity a matrix-free method cannot
    produce. Measured against 1-D quadrature: dropping it biases the mean by
    27 standard errors (reproduced on a second seed), and the stationary
    distribution has a closed form, ``p(x) * det M(x)^(-1/2)`` -- at an
    acceptance rate that looks HEALTHIER than the correct chain's.

    Freezing sigma at a function of the OUTER state alone fixes it: the
    proposal is then genuinely independent of ``x``, ``M' = M`` exactly, the
    constant cancels for real, and the cost drops from 3 CG solves to 2 --
    forward mu and the draw, with both densities being quadratic forms that
    cost one operator application each. Measured error: -0.00046.

    A consequence worth stating: **correctness does not depend on sigma-hat
    being any good.** Any x-independent choice gives a valid chain; the
    quality of sigma-hat sets only the acceptance rate. That converts a
    correctness risk into a performance knob.

    Sigma must NOT be taken from the previously accepted x -- that depends on
    the chain's history, making this an adaptive chain valid only under
    diminishing adaptation.
    """
    propose_key, accept_key = jax.random.split(key)
    mu, _ = wiener_solve(block, noise_std=sigma, tol=tol, maxiter=maxiter,
                         require_convergence=None)
    draw, _ = gcr_sample(block, noise_std=sigma, key=propose_key, tol=tol,
                         maxiter=maxiter, require_convergence=None)
    proposed = {name: draw[name] for name in names}
    now = {name: current[name] for name in names}
    log_alpha = (log_weight(graph, block, proposed, at=at, noise_std=sigma, mu=mu)
                 - log_weight(graph, block, now, at=at, noise_std=sigma, mu=mu))
    take = jnp.log(jax.random.uniform(accept_key)) < log_alpha
    return {n: jnp.where(take, proposed[n], now[n]) for n in names}
```

`assemble(plan, ...)` 用 `HMCGibbs(NUTS(to_numpyro(graph)), gibbs_fn=..., gibbs_sites=[...])` 装配，并**拒绝 `chain_method="vectorized"`**（§二实测：`HMCGibbs.init` 无条件 `random.split`，vectorized 下递批量 key，在 `gibbs_fn` 被调用之前就抛 `ValueError`）。

- [ ] **Step 4b: 再加两条测试——σ 依赖块外隐变量，以及 vectorized 多链**

```python
def test_noise_std_is_rebuilt_when_sigma_depends_on_a_latent_outside_the_block():
    """`check_prediction_dependence` is structurally blind to this case.

    It only ever moves BLOCK members, so on a graph whose sigma depends on an
    OUTSIDE latent it reports 0.000e+00 -- and the plan prints "sigma is
    constant", which licenses hoisting `noise_std` out of the sweep. Measured
    on `d ~ N(w*X, exp(lognoise))`: the block's own sigma movement is exactly
    0.0 while `lognoise` over +/-2 moves sigma by 639%, and freezing it gives
    a posterior 4.4x too narrow at lognoise=+0.5 and 17x too narrow at +2.0.

    The criterion is structural, not numeric: any observed node whose scale
    has a LATENT ANCESTOR forces the rebuild, whether or not that ancestor is
    in this block. A movement probe cannot be the criterion here, because the
    probe is exactly what cannot see it.
    """
    with jax.enable_x64(True):
        graph = outside_sigma_latent()
        fn = gibbs_factory(graph, ("w",), tol=1e-10, sigma_rebuild=True)
        for value in (0.5, 2.0):
            keys = jax.random.split(jax.random.key(2), 4000)
            draws = jax.vmap(lambda k, v=value: fn(
                rng_key=k, gibbs_sites={"w": jnp.asarray(0.0)},
                hmc_sites={"lognoise": jnp.asarray(v)})["w"])(keys)
            oracle = graph_oracle(graph, ["w"],
                                  at={"lognoise": jnp.asarray(value)})
            assert float(jnp.std(draws)) == pytest.approx(
                float(np.sqrt(oracle.covariance[0, 0])), rel=5e-2)


def test_vectorized_chains_are_refused_with_a_reason():
    """numpyro 0.21.0's HMCGibbs.init splits rng_key unconditionally.

    `HMC.init` has an `rng_key.ndim` branch; `HMCGibbs.init` does not, so
    under `chain_method="vectorized"` MCMC hands it a batched key and it
    raises `ValueError: split accepts a single key...` before `gibbs_fn` is
    ever reached. Refusing it here turns an upstream stack trace into a
    sentence, and pins the limitation so that a numpyro upgrade fixing it
    shows up as a deliberate decision rather than a silent behaviour change.
    """
    plan = compile_graph(mixed_radiometer())
    with pytest.raises(NotImplementedError, match="vectorized"):
        plan.sample(jax.random.key(0), num_chains=2, chain_method="vectorized")
```

新 fixture `outside_sigma_latent()`：`lognoise ~ N(-1, 1)`、`w ~ N(0, 3)`、`mu = w*X`（`linear_in=("w",)`）、`d ~ N(mu, exp(lognoise))`。`lognoise` 因判据 3 落 NUTS（`mu` 不声明对它线性，且它根本不在 `mu` 的路径上——它进的是 `d` 的 scale），块是 `{w}`。

- [ ] **Step 5-6: 实现到通过、跑全套、变异测试**

| 变异 | 必须变红 |
|---|---|
| `at` 从先验均值建而非 `hmc_sites` | `test_gibbs_fn_uses_the_hmc_sites_it_was_handed` |
| σ̂ 冻在零点而非声明点 | `test_gibbs_fn_freezes_sigma_where_it_says_it_does`（两臂） |
| `probe_gaussian` 传 `True` | `test_gibbs_fn_survives_a_trace_and_the_probe_still_bites` |
| `_mh_step` 的 `log_alpha` 去掉 `now` 那一项 | Task 10 的 MH 不变性测试 |
| σ̂ 改回依赖 `current` | 同上（且这是 spec §5.3 的原始错误） |
| 返回 `{**now, **proposed}` 之外的键 | `test_gibbs_fn_is_called_by_keyword_and_returns_exactly_the_block` |
| `sigma_rebuild=True` 被接受但忽略 | `test_noise_std_is_rebuilt_when_sigma_depends_on_a_latent_outside_the_block` |
| 不拒绝 `chain_method="vectorized"` | `test_vectorized_chains_are_refused_with_a_reason` |

- [ ] **Step 7: AST 比对 + 提交**

---

## Task 9：`sample()` / `estimate()` 与公开的 `bayesmith.compile`

**Files:**
- Modify: `src/bayesmith/dispatch/plan.py`（加 `sample` / `estimate` / `Posterior` / `Estimate`）
- Modify: `src/bayesmith/__init__.py`
- Test: `tests/dispatch/test_dispatch_entry.py`、`tests/test_public_api.py`

- [ ] **Step 1: 写失败的测试**

```python
def test_a_fully_exact_graph_samples_without_a_chain():
    """The unambiguous win: iid draws, ESS = N, no warmup, nothing to diagnose."""
    post = compile_graph(two_linear_latents()).sample(jax.random.key(0), num_samples=500)
    assert post.method == "gcr"
    assert post.log_weights is None
    assert post.ess == pytest.approx(500.0, rel=1e-9)
    assert not post.unreliable


def test_ess_reduces_over_every_site_and_coordinate():
    """`ess` is one float, so the reduction has to be written down: MIN.

    Measured in the spec's benchmark C: ESS(logw)=3.0 and ESS(alm,min)=40.2
    coexist in one run. This field exists to make dividing by N a deliberate
    act, so it must report the WORST of them -- a mean would let a
    well-mixing site hide a stuck one.
    """
    post = compile_graph(mixed_radiometer()).sample(
        jax.random.key(0), num_warmup=200, num_samples=400)
    per_site = {k: float(jnp.min(effective_sample_size(np.asarray(v)[None])))
                for k, v in post.samples.items()}
    assert post.ess == pytest.approx(min(per_site.values()), rel=1e-6)


def test_estimate_raises_convergence_error_rather_than_returning_a_number():
    """P3a defined ConvergenceError and left it unraised, by design.

    `iterative_gls` returns `converged` as a field and leaves promotion to
    its caller. This is that caller -- the first one in the package.
    """
    with pytest.raises(ConvergenceError):
        compile_graph(radiometer()).estimate(max_reweights=1, reweight_tol=1e-14)


def test_estimate_refuses_a_mixed_graph_and_says_where_to_go():
    with pytest.raises(NotImplementedError, match="sample"):
        compile_graph(mixed_radiometer()).estimate()


def test_importing_bayesmith_still_does_not_import_numpyro():
    """The lazy-import contract, unchanged by a new subpackage."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import bayesmith, sys; print('numpyro' in sys.modules)"],
        capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_compile_is_the_function_not_the_subpackage():
    """`dispatch/`, not `compile/` -- and this is the test that would have
    caught the collision the first draft of the spec proposed.

    Measured on an isomorphic package: with the dispatcher living in
    `compile/`, one test file doing `from bayesmith.compile.partition import
    ...` rebinds `bayesmith.compile` to a module for the whole session, so
    this assertion passes alone and fails in the suite -- and under xdist.
    """
    import bayesmith
    import bayesmith.dispatch.classify  # noqa: F401  (the poisoning import)
    assert callable(bayesmith.compile)
```

- [ ] **Step 2-4: 实现，跑通**

`compile(graph)` 进 `_LAZY_ATTRS`（指向 `bayesmith.dispatch.plan`），`dispatch` 进 `_LAZY_SUBMODULES`。`Posterior` / `Estimate` 按 spec §6.3 定义。**`sample()` 在 SNIS 路径上必须检查 Kish ESS/N 是否塌缩并按 §6.4 回退**，而不是对一个大块返回 `unreliable=True`。

- [ ] **Step 5-6: 变异测试 + 提交**

| 变异 | 必须变红 |
|---|---|
| `ess` 取 `mean` 而非 `min` | `test_ess_reduces_over_every_site_and_coordinate` |
| `estimate` 忽略 `converged` | `test_estimate_raises_convergence_error_rather_than_returning_a_number` |
| `compile` 放进 `compile/` 子包 | `test_compile_is_the_function_not_the_subpackage` |
| `dispatch` 在 `__init__` 里 eager import | `test_importing_bayesmith_still_does_not_import_numpyro` |

---

## Task 10：验收关口二 —— 统计

spec §7.2。**每一条的功效都已实测**，参数写进参数化。

**Files:** `tests/dispatch/test_acceptance.py`、`tests/exact/test_correct.py`（补）

- [ ] **Step 1: (b) MH 不变性——在 `steep_radiometer` 上，不标 `slow`**

```python
@pytest.mark.parametrize("draws, seed", [(2000, 0), (2000, 1)])
def test_dropping_the_reverse_density_term_moves_the_moments(draws, seed):
    """The single guard for §5.3's silent-wrong-stationary-distribution.

    Deliberately NOT on `radiometer`: measured there, the subtle mutation
    shifts the mean by 0.018 posterior sd at acceptance 0.955 against a
    correct 0.950, needing ~12,000 draws for 2 sigma and ~75,000 for 5 --
    unavoidably `slow`, and `slow` would remove the only guard this bug has.

    On `steep_radiometer`: bias -0.2415 posterior sd, sd ratio 0.738,
    acceptance 0.64, 2 sigma at 66 draws. 2000 draws is ~8 sigma.

    The positive control on acceptance matters as much as the bias: if a
    future parameter change flattens this fixture, acceptance drifts toward
    0.95 and the test quietly loses its power while staying green.
    """
    ...
    assert accept_rate == pytest.approx(0.64, abs=0.08)   # positive control
    assert abs(biased_mean - correct_mean) / correct_sd > 0.15
```

- [ ] **Step 2: (c) SNIS —— 预言机换成求积，另加 ESS 守卫**

`radiometer` 是标量、`radiometer_group` 是二维，都能精确积到 ~1e-10。**不要用长跑 NUTS**——P3a 的记录说那是自洽检查。

外加 `test_a_non_converged_gls_sigma_shows_up_as_ess_not_as_bias`：**矩对 σ̂ 的冻结点是瞎的**（用不同 σ̂ 的 SNIS 仍是合法重要性采样，矩照样收敛，只有 ESS 掉），所以这一条断言 Kish ESS/N 掉一个钉死的倍数而加权均值仍对。

- [ ] **Step 3: (d) CG 容差——三点，不是两点**

```python
@pytest.mark.parametrize("tol", [1e-1, 1e-6, 1e-12])
def test_weighted_moments_agree_at_tight_tolerances_and_move_at_a_loose_one(tol):
    """Three points, because two would prove nothing in either direction.

    The draft used {1e-6, 1e-12}. Two problems, both fatal:
    - the DEFAULT is 1e-6, i.e. the LOOSE end of that pair, while §5.5's
      failure mode lives where CG has not converged -- looser still. The whole
      sweep ran in the direction where nothing can happen (P3a Task 5 verbatim);
    - two arms asserted to AGREE cannot detect a `tol` that is ignored
      entirely: both become bit-identical and the test is green.

    1e-1 is the third point, and it must measurably DISAGREE. That is what
    proves the knob is connected. x64 throughout: in float32 CG on an
    ill-conditioned block plateaus at ~4e-3 relative residual and never
    reaches any requested tolerance, so `maxiter` is the real stopping rule
    and this test would measure nothing.
    """
```

- [ ] **Step 4: (e) 复合 vs 纯 NUTS —— 拆两条**

条件矩走 Task 8 的预言机对比（`hmc_sites` 固定、确定性）；边缘矩**只在先实测过外参数确实混合的 fixture 上**比较，`ESS` 与容差一起写进 docstring。

- [ ] **Step 5: (f) ESS/秒 —— 不做数值断言**

```python
@pytest.mark.benchmark
def test_report_ess_per_second_without_asserting_a_threshold():
    """A report, not a guard, and the docstring says why.

    Measured across five seeds on the spec's benchmark C: in-block ratio
    4.72, 5.97, 10.71, 16.75, 19.39 -- so an assertion of >=5.98x is already
    red at seed 30. Under x64 the same configuration gives 24-43x, and at
    tol=1e-4 it falls to 1.58x because the draws become WRONG (in-block ESS
    collapses 286.8 -> 9.0).

    Worse than fragile, it is monotone in the wrong direction: a `gibbs_fn`
    that returns a cheap wrong draw is FASTER, so the ratio goes UP. A
    performance assertion sitting beside a correctness question rewards
    breaking the correctness.

    Two directions do NOT improve and are printed too: the out-of-block
    parameter (0.17x steady state) and every well-conditioned block
    (0.13-0.91x). tol, maxiter, dtype and seed are pinned so the numbers mean
    something when they are read.
    """
```

- [ ] **Step 6: §7.3 边界验证——只剩两个真阈值**

`condition_bound` 导出的 `tol`（守卫开/关）与 Kish ESS/N 的塌缩阈值（SNIS/回退）。**初稿列的另两个是范畴错误**：k̂=0.7 两侧是同一个计算；「MH 接受率｜反向密度在场/缺席」实测两侧 0.950 vs 0.955——**它们一致，而这正是 bug**。

- [ ] **Step 7: 极端参数**

块大小 1 与 1e4（**受 §5.2 的 SNIS 维度上限约束**——n≈500 时 Kish ESS 已经是 1.00，所以 1e4 的块只在非 SNIS 路径上有意义）、观测节点数 1 与 5、κ 从 1 到 1e10、σ 跨六量级、完全共线的父节点、零观测的隐变量。

- [ ] **Step 8: 全套 + `ruff check` + 覆盖率 + 提交**

Run: `.venv/bin/python -m pytest -q && .venv/bin/python -m pytest -m "not slow" -q && ruff check`

**两种跑法都要跑**——P3a 的第四条子判据正是「测试写对了、fixture 到位、变异会变红，但它在 `-m "not slow"` 下不执行」。

---

## 验收（本计划完成的判据）

- [ ] `.venv/bin/python -m pytest -q` 全绿，且 `-m "not slow"` 也全绿
- [ ] **Task 1 的判决对比表已提交**，任何翻转都具名解释
- [ ] `test_a_bright_component_does_not_mask_a_false_claim_on_a_faint_one` 通过（B1）
- [ ] `test_sigma_depending_on_a_contrast_of_two_members_is_detected` 通过（B2）
- [ ] `tests/dispatch/test_classify.py` 的分类表全绿，含 `orphaned_child_latent`、`student_t_likelihood`、`shared_ancestor` 三行
- [ ] `test_a_lying_observed_node_raises_rather_than_falling_back_to_nuts` 通过
- [ ] `test_gibbs_fn_survives_a_trace_and_the_probe_still_bites` 通过（两侧）
- [ ] MH 修正的守卫通过，且**未标 `slow`** —— 实际名为
      `test_the_frozen_sigma_proposal_is_wrong_and_the_accept_step_is_what_fixes_it`
      （`tests/dispatch/test_acceptance.py`，提交 `6960268`）。**改名是实测的结果**：
      「丢掉反向密度项」这条变异**不会移动矩**——实测 `log w(x')` 在 20000 次抽取里
      中位数 −14.39、最大 −8.204，从不 ≥0，于是接受率塌到 0/20000，核变成恒等映射，
      而恒等映射**完美通过任何不变性断言**。杀死它的是接受率带，所以那一条断言排在最前。
- [ ] `test_log_weight_equals_log_p_minus_log_q_against_a_dense_gaussian` 通过
- [ ] `InferencePlan.__str__` 打印 κ（或区间）与 `tol`，且拒绝理由点名成员
- [ ] `test_importing_bayesmith_still_does_not_import_numpyro` 与 `test_compile_is_the_function_not_the_subpackage` 都通过
- [ ] 每个任务的变异测试都做过，每条都指名了一条**变红的具体测试**
- [ ] 每个任务的 AST 规格比对都跑过，每处实质差异都具名说明
- [ ] `ruff check` 干净
- [ ] 新文件覆盖率 ≥ 80%

## 执行结果（2026-08-24 完成）

Tasks 4–10 全部落地，随后经四路只读审查（规格合规 / 数值正确性 / 测试质量 / 代码质量与分层）
审了整个 `main..HEAD`，审查发现的 1 条 CRITICAL、5 条 HIGH 与若干 MEDIUM 已在同一轮修掉。

**基线迁移**：278 → **637** 通过；`-m "not slow"` 632 通过、5 条既有 `slow` 被 deselect；
`ruff check` 干净；五个新模块行覆盖 97–100%。

### 审查改掉的东西（都不在原计划里，都是实测驱动）

| 提交 | 严重度 | 内容 |
|---|---|---|
| `73b5d3a` | **CRITICAL** | float32 下 σ 加权判据**根本不运行**（SNR>0.84 即失效），假 `linear_in` 被接受、后验偏 **801σ**，而计划打印 `linear_in ✓ max 0.00e+00`。拆成 `WEIGHTED_FLOOR_FACTOR=1e2` 与 `RELATIVE_FLOOR_FACTOR=1e4`；未判定的列现在返回 `Unresolved` 并具名告警，不再报成「测得零」 |
| `b3ab244` | HIGH | 裸 `gcr` 不做任何修正，而移动探针够不到拐折的 σ。改为对先验尺度探针与**块自身 Wiener 解**处的 σ 取 max——后者正是后验所在。判决表零变化 |
| `c0caaa6` | HIGH ×4 | `chain_method`/`nuts_options` 在两条 NUTS 路径上被静默丢弃；`tol` 低于工作精度时不再谎称 guard reachable；SNIS 塌缩改为**标注而非替换**（被丢掉的那个实测更准：1.40σ vs 18.46σ）；`except BayesmithError` 收窄为 `NotGaussian` |
| `4fa9e94` | MED | 逐成员高斯探针此前可整段删除而套件全绿——`LyingNormal` 只出现在观测节点上 |
| `15fff84` | — | 本分支造成的 12 个文件格式漂移（豁免只覆盖既有的 9 个） |

### 与 spec §4.2 的一处**已知偏离**，不是缺陷

§4.2 要求「块外隐变量影响 κ 时**在 sweep 内重算** `condition_bound`」。实际落地的是 Task 6
Step 3 授权的**采样区间**：在每个块外隐变量的 ±1σ/±3σ（外加锚点）处各测一次，`tol` 取区间**上端**。
理由是安全方向——上端给出更紧的 `tol`，代价是 CG 迭代数而不是精度。**残余风险已具名**：
两个块外隐变量的**联合**极值点、plate 内的对比方向、以及 ±3σ 之外，都没有被采样到，
那里 `tol` 会偏松而 sweep 内的守卫是关着的。`kappa_interval` 的 docstring 与提交 `cd50fd2`
都写着「这是采样区间，不是上界」。**不要把这一条当成新发现的缺陷重新报一遍。**

### 仍然开着的口子（已记录，不要当成没发现）

- `LinearBlock` 不记录自己是在哪个 `at` 上建的，`log_weight` 因此无法校验传进来的 `at`。
  经 `compile()/sample()` 不可达（每个调用点都在同一个表达式里建 `block` 与 `at`），
  直接调 `log_weight` 的人才碰得到。
- `depends_on_prediction` **无人读**（不是「无人查」）：默认值就是 `True`，
  分派完全由实测 `movement` 决定。仓库里没有 fixture 声明 `False`。
- 混合计划不说明**每个** NUTS 隐变量为何被弹出——`why_not` 只在块为空时保留。
- `_ancestors`（跨包）与 `_weights`（跨模块）是私有名被外部读取；
  `plan._execution()` 在拆分后成了跨模块的私有调用。三处都建议提升为公开名。
- 分派方法名是裸 `str`，四个文件里比较与查表，只有 `gibbs_factory` 一处校验。
- `num_chains > 1` 接通但无测试。

## 明确不在本计划范围内

- 混合图的 MAP（P5）；`fn` 非线性块的修正（P5）；`identifiability` 的稠密 SVD（P5）
- RTS / Kalman（P4）；离散枚举（P4）
- `MultivariateNormal` 稠密协方差观测噪声；嵌套 plate
- 失败块的极大合格子集搜索（spec §3.5）
- `chain_method='vectorized'` 的多链（numpyro 0.21.0 下坏掉；用 `sequential`/`parallel`）
- `ruff format --check` 在 9 个 P1/P2 文件上的漂移——单独一轮
