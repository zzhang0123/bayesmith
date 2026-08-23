# bayesmith P3b — 分派器、InferencePlan、Gibbs 与近似块的修正

> 本文是设计（spec），不是实施计划。
> 上游：`docs/superpowers/specs/2026-08-23-p3-structural-dispatch-design.md`（下称「P3 spec」）的 §四、§五、§六、§七。
> 前置：P1 图核 + P2 NumPyro 桥 + **P3a 精确解核心**（已合入 main，242 测试全绿，HEAD `0c03ea5`）。
>
> **本文相对 P3 spec 的地位**：P3 spec 的 §四/§五/§六 是 P3b 的设计草案，写于 P3a 开工之前。本文在其上做三处**实测更正**（§一）并补齐它留白的三处（分区算法、返回类型、验收关口的可达性）。**冲突处以本文为准**，P3 spec 对应位置已插入指针。

## Context

P3a 交付了无矩阵线性算子、线性性检验、Wiener 解、GCR 精确抽取、迭代 GLS 与稠密 Fisher——**但没有任何东西读它们**。今天调用方必须自己知道哪些隐变量合格、自己组块、自己决定用 `wiener_solve` 还是 `iterative_gls`、自己把守卫提到循环外。

P3b 是这些决定第一次由**图**做出。P1 记录的三条结构轴（`linear_in`、`support`、`depends_on_prediction`）中，`linear_in` 在 P3a 被**检验**过但没有被**读**过——P3a 检验的是仿射性这一事实，不检查沿路径的声明。P3b 是它第一次决定计算路径。

---

## 一、三处实测更正

本节的每一条都是在**写下这份 spec 的过程中实测**的，而不是回忆或推理。P3a 的执行记录量化了为什么必须这样做：约三分之二的真实缺陷源自计划文本而非实现，且失效集中在三类断言上——**数值区域、通用性、移植/上游来源的行为**。下面三条恰好一类一个。

### 1.1 `compile/` 目录与 `bayesmith.compile()` 会静默互相覆盖

P3 spec §八 把分派器放在 `src/bayesmith/compile/`，而设计文档 §二 的既定 UX 是 `plan = compile(graph)`。**两者不能共存**，且失效是顺序依赖且静默的。

实测（构造同形的最小包，`__init__.py` 用与 `bayesmith/__init__.py` 相同的 `__getattr__` 惰性解析）：

| 顺序 | 结果 |
|---|---|
| 先访问属性 `pkg.compile('g')` | **可用**，返回 `PLAN(g)` |
| 其后 `import pkg.compile.dispatch` | `pkg.compile` **被静默重绑为 module** |
| 先 `import pkg.compile.dispatch`，再 `pkg.compile('g')` | `TypeError: 'module' object is not callable` |
| `from pkg import compile` | 拿到 **module**，不是函数 |

危险的是第二行：程序在第 1 行可用，在第 50 行因为一句看上去无关的 `import` 而失效，**而出错的位置不在那句 import 上**。`bayesmith/__init__.py` 现有的 `_LAZY_SUBMODULES` 机制会让这件事更容易发生——它就是为了让 `bayesmith.exact` 这类子包可达而写的。

**决定**：目录改名 **`src/bayesmith/dispatch/`**。公开函数名 `bayesmith.compile(graph)` **不动**（设计文档 §二 的既定 UX，且遮蔽内置 `compile` 已由 docstring 点名接受）。目录名从不出现在用户代码里，改它零成本。

### 1.2 `unchecked_operator` 在 trace 下不可用

P3 spec §4.1 的 `gibbs_fn` 正文写的是：

```python
block = linear_operator(graph, names, at=at, check=False)   # 每 sweep 重建
```

**两处与实现不符**，都实测：

1. `linear_operator` 没有 `check=` 关键字。实测签名是 `(graph, names, at=None, *, scales, rtol, at_points, key)`。
2. 更要紧的：`gibbs_fn` **在 trace 下运行**。实测 `gibbs_sites` 与 `hmc_sites` 的值都是 `DynamicJaxprTracer`。而 `unchecked_operator` 内部对每个观测节点调 `check_gaussian`，后者做 `bool(jnp.all(...))`、`float(jnp.min(scale))` 与 `raise`——它自己的 docstring 就写着 "Runs on concrete values, **outside** any trace"。

实测把 `unchecked_operator` 放进 `jax.jit`：

```
TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[].
```

**同时实测了充分性**：把 `check_gaussian` 替换为 no-op 后，`unchecked_operator` + `gcr_sample` 在 `HMCGibbs` 的 `gibbs_fn` 里完整跑通了三组不同规模的基准（§1.3）。**高斯探针是唯一的 trace 不安全处**，其余部分（`_validated_names`、`_env_before`、`isolate`、`jax.linearize`、`jax.vjp`、`observation_parts`）都可 trace。

**决定**：`unchecked_operator` 增加关键字 `probe_gaussian: bool = True`。

- 默认 `True`：行为与今天逐位相同，P3a 的既有调用方与测试全部不受影响。
- sweep 内传 `False`，且**只允许**在编译期已经跑过 `check_gaussian` 之后传。

不新增第三个名字。本包已有的两级命名（`linear_operator` 检验线性性 / `unchecked_operator` 不检验）再加第三级会让「哪一级检验什么」变成需要背的东西；一个具名关键字把它写在调用点上。docstring 必须逐条列出三级各检验什么，以及 `False` 的唯一合法用途。

### 1.3 ESS/秒 不是一条可以无条件断言的关口

P3 spec §7.2 把这一条写成了 pass/fail：

> 复合 vs 纯 NUTS：……且 HMCGibbs 的 **ESS/秒更高**——否则这条路径没有存在理由。

**实测：这条断言是随被测量的量而变的，不能作为无条件关口。**

三组基准，每组都用 P3a 的真实 `unchecked_operator` + `gcr_sample` 装进 `HMCGibbs`，与同一张图上的纯 NUTS 比。所有比值为 `(ESS/秒)_Gibbs / (ESS/秒)_NUTS`：

**基准 A**——`mu_i = exp(beam)·alm_i`（逐元素，plated），`alm ~ N(0, 1.5)`，`sigma=0.4`，warmup 400 / samples 800：

| n | beam | alm |
|---|---|---|
| 20 | 0.16x | 0.15x |
| 100 | 0.60x | 0.91x |
| 500 | 0.27x | 0.27x |

**基准 B**——`mu = exp(beam)·(K alm)`，`K` 为行归一化高斯平滑核，`alm ~ N(0, 1.5)`，warmup 500 / samples 1000。实测 `condition_bound ≈ 15`——**块仍然良态**：

| n | ell | κ 上界 | beam | alm |
|---|---|---|---|---|
| 100 | 6 | 15.0 | 0.13x | 0.79x |
| 300 | 10 | 14.6 | 0.15x | 0.70x |
| 600 | 20 | 14.9 | 0.20x | 0.78x |

**基准 C**——把先验放宽到 `alm ~ N(0, 100)` 使块**病态**，外参数改为平滑宽度 `logw`（非线性，且不是纯乘性简并），n=120，warmup 250 / samples 500，实测 `condition_bound = 6.203e4`：

| | 墙钟 | ESS(logw) | ESS(alm, min) | ESS/秒 logw | ESS/秒 alm |
|---|---|---|---|---|---|
| NUTS | 20.38s | 296.3 | 159.1 | 14.54 | 7.81 |
| Gibbs | **0.86s** | **3.0** | 40.2 | 3.45 | 46.66 |
| 比值 | 24x 更快 | | | **0.24x** | **5.98x** |

**结论，以及它的机制。** 优势方向相反且都是真的，成因是同一个：

* **块内（`alm`）**：GCR 是 iid 抽取，块内自相关被完全消掉。病态时 NUTS 需要极小步长，Gibbs 赢 **5.98x**。
* **块外（`beam` / `logw`）**：条件于 `alm` 使外参数被数据钉死，Gibbs 在它上面爬行——`ESS(logw) = 3.0`（共 500 抽取），基本没在动。这是强耦合下 Gibbs 混合失败的经典形态，不是实现缺陷。

良态时（基准 A、B）**两个方向都输**：块内没有自相关可消，而块外的爬行照旧。

**决定**：

1. **正确性是关口**，性能不是。HMCGibbs 的存在理由改述为「**它让混合图可解**，并且在病态块上把块内 ESS/秒提高一个可测的倍数」，而不是「它更快」。
2. ESS/秒 降为一条**具名区域**的基准测试：在 κ ≳ 1e4 的 fixture 上断言**块内** ESS/秒 改善，并在同一条测试的 docstring 里**如实写下块外不改善**及其机制。
3. 这条更正直接改写了另一条测试的设计——见 §7.2(e)。

> **本节三条各属 P3a 记录的一个失效类别**：§1.1 是结构性断言（写下时两行代码即可证伪），§1.2 是「移植/上游来源的行为」（在回忆 spec 草案而不是读 P3a 的实现），§1.3 是「数值区域」（凭直觉给了一个方向）。P3a 的记录说后两类"写下的当时就可以验证，而我没有"——本文验证了。

---

## 二、上游 API 事实（本次实测）

在 `/Users/zzhang/projects/bayesmith/.venv` 实测（2026-08-23，numpyro 0.21.0 / jax 0.11.1）：

| 事实 | 值 |
|---|---|
| `HMCGibbs.__init__` | `(self, inner_kernel, gibbs_fn, gibbs_sites)` |
| `gibbs_fn` 调用约定 | `(rng_key, gibbs_sites, hmc_sites) -> dict` |
| `gibbs_sites` 的内容 | `{块成员名: 值}`，值是 **`DynamicJaxprTracer`** |
| `hmc_sites` 的内容 | `{块外隐变量名: 值}`，**不含观测节点** |
| `rng_key` 的类型 | 带类型的 key（`key<fry>`），不是 uint32 对 |
| 返回值 | 只需含 gibbs 成员；实测返回常数后该 site 的抽取全部等于该常数 |
| `_psis_khat` | `numpyro.infer.importance._psis_khat(log_weights: np.ndarray) -> float` |
| `_psis_khat` 的返回 | **具体 Python `float`**——因此**不能在 jit 下调用** |
| `_psis_khat` 标定 | 高斯 log 权重 → 0.184；指数(×5) log 权重 → 4.398 |
| `log_joint` | `(graph, values) -> jax.Array`（标量），SNIS 权重公式直接可用 |
| `exact → bridge` 依赖 | 已存在（`linearity.py:157` 函数内惰性 import），`gibbs.py` 沿用同一方向，无环 |

`_psis_khat` 的具体返回值是本文新增的一条约束：**SNIS 的诊断必须在 jit 之外算**。这不构成限制——SNIS 没有循环要 jit，权重与自归一化是一次 `vmap` 加一次 `softmax`。

---

## 三、分区推导

P3 spec §2.3 只说「合格隐变量按图的耦合自动成块」，没有给规则。本节给出。

### 3.1 合格性（沿用 P3 spec §2.1，四条全部成立）

1. `x` 自己的分布是对角高斯（`gaussian_parts` + `check_gaussian` 通过）；
2. `x` 的先验的父节点不含另一个候选块成员；
3. 每个观测节点都是对角高斯；
4. 从 `x` 到每个观测节点的 `loc` 的**每一条**路径，沿途每个 `Deterministic` 节点都在 `linear_in` 里声明了其入边父节点。

第 4 条是 P3b 第一次**读** `linear_in`——P3a 只检验仿射性这一事实，不检查沿路径的声明。**未声明 → 不合格 → NUTS**，与 `support=None` 的既定原则一致。

`NotGaussian` 捕获后落 NUTS；`StructureError` **绝不捕获**——它意味着节点自称高斯而它自己的密度不是，静默降级会把坏模型藏在一个看起来正常的回退后面。P3a 的 `errors.py` 已把这条区分做成行为契约。

### 3.2 分区规则

1. 按 §3.1 分出合格集合 `Q`。
2. **祖先出块**：若 `Q` 中的 `z` 是 `Q` 中另一个 `x` 的祖先，则 `z` 离开 `Q`，走 NUTS。
   分层模型 `tau → x` 因此得到标准且正确的切法：**`x` 精确、`tau` 走 NUTS**。P3a 的 `_refuse_internal_ancestry` 已经在算子层强制这一点（抛 `NotGaussian`），分派器在它之前就把这种组合排除掉，使那条 raise 保持为防御性的。
3. **剩下的 `Q` 构成唯一一个块**——联合求解，不交替。
4. 对这个块跑 `check_linearity`（多 `at` 点）。**失败则整块落 NUTS**，理由里点出成员名。
5. σ 依赖探测（`check_prediction_dependence`）决定方法：`gcr` / `gcr+snis` / `gcr+mh`。

> **实现要求（实测）**：当块是隐变量集合的**真子集**时（§3.2 第 2 条出块之后），
> `check_linearity` / `unchecked_operator` 会对块外隐变量要求 `at` 值，缺了就抛
> `GraphError`「latents [...] are outside the block and have no value in `at`」。
> 所以分派器必须为块外隐变量构造 `at`——用它们各自的**先验均值**，与
> `_env_before` 对块成员的做法一致。`indirect_ancestor` / `diamond_ancestor` /
> `shared_ancestor` 三个 fixture 都会走这条路径。

### 3.3 为什么是「一个块」而不是连通分量

两个合格隐变量只要共同影响同一个观测节点，给定数据就**后验相关**。把它们分到不同块靠交替推进，正是 rheplicant `plan.py` 用 60 行 docstring 记录的那个失败：双线性 `gain × T_ant` 手工交替，**CG 残差 1e-7、逐块 κ≈1.47、`check_linearity` 每轮都过，答案错几千开尔文**——每个条件分布确实是仿射的，错的是分区，而任何逐块的数字都无权发现它。

联合成块使这件事**不可能发生**：`gain` 与 `t_ant` 落在同一个块里，联合 `check_linearity` 探针（三次前向求值）直接抓住假断言，整块落 NUTS。

### 3.4 判据 4 的两种「空真」，都必须被接受

实测发现两个 fixture 使合格性判据第 4 条**空真**，而**两个都应当合格**：

| fixture | 空真的成因 | 为什么合格是对的 |
|---|---|---|
| `plated_latent` | `d` 的 `loc` **就是** `z` 本身，路径上**没有 `Deterministic` 节点** | 「沿途每个 `Deterministic` 都声明了」对空集成立 |
| `unconstrained_latent` | `u` **到任何观测节点都没有路径** | `A` 对 `u` 的列恰为零，正规算子在该方向只剩先验曲率，答案就是先验均值——P3a 的 `test_a_latent_the_data_never_reaches_comes_back_at_its_prior_mean` 已经钉住 |

**一个朴素的实现会把两个都错误地拒绝**：把判据 4 写成「存在某个 `Deterministic`
的 `linear_in` 点了 `x` 的名字」，对两者都为假。正确的写法是**对路径集合做全称量
词**，而全称量词对空集为真。

这也是 P3a「结构维度」判据的一个实例：判据 4 在「路径条数」这一维上分支，而 0 是
一个必须取到的值——**取两次**，因为 0 条 `Deterministic` 与 0 条路径是不同的零。

### 3.5 「失败全落」的代价，明确写下

五个合格隐变量里有一对双线性，五个一起落 NUTS。这是**故意的**：

* 安全方向正确——落 NUTS 永远给对的答案，只是慢；
* 搜索极大合格子集是指数的，且子集选择不唯一（去掉 `gain` 还是去掉 `t_ant`？两者都能让剩下的通过，而它们给出**不同的**分区）；
* 拒绝时点出成员名，用户改一个 `linear_in` 声明就能自己得到更好的分区——这个决定应该由知道模型含义的人做，不是由一个不唯一的搜索做。

`InferencePlan` 的可打印表必须把这条理由印出来，否则用户只会看到「全部 NUTS」而不知道差一个声明。

---

## 四、Gibbs 装配

### 4.1 形态

```python
kernel = HMCGibbs(NUTS(to_numpyro(graph)), gibbs_fn=..., gibbs_sites=[*block.latents])
mcmc   = MCMC(kernel, num_warmup=..., num_samples=..., num_chains=...)
```

`to_numpyro(graph)`（P2）已把每个节点开成同名 site，`gibbs_sites` 直接用节点名。**P2 全量复用，零适配层。**

**不移植** `plan.py` 的 sweep 循环、`split_rhat`、`PlanDiagnostics`、chi² 监控，也不移植 `engines.py`——NumPyro 的 `MCMC` 全部提供，r̂ / ESS 走 `numpyro.diagnostics`。

### 4.2 三个必须外提的守卫（补一个）

`gibbs_fn` 每 sweep 在 trace 下运行，所以：

| 守卫 | 编译期 | sweep 内 |
|---|---|---|
| `check_gaussian` | 跑一次 | `probe_gaussian=False` ← **§1.2 新增** |
| `check_linearity` | 多 `at` 点跑一次 | 不跑（`unchecked_operator`） |
| `condition_bound` | 跑一次，`tol = require_convergence / κ` | `require_convergence=None` |

**关掉守卫必须同时收紧 `tol`。** rheplicant 明写"leave `tol` at its default and the guard off"正是它返回过静默过窄后验的组合。**`InferencePlan` 必须把 κ 与导出的 `tol` 两个数都打印出来**，否则这条纪律无法被检查。

### 4.3 上游接口风险

`HMCGibbs` 标着 `[EXPERIMENTAL INTERFACE]`。按本包对 equinox `BoundMethod` 的既有做法：写一条测试**钉住 `gibbs_fn(rng_key, gibbs_sites, hmc_sites)` 的调用约定**——包括 §二 实测的那四条内容约定（gibbs_sites 是块成员、hmc_sites 是块外隐变量且不含观测、值是 tracer、返回值只需含 gibbs 成员）——让上游改动变成一个自觉的决定而不是意外。numpyro 版本在 pyproject 里下界收紧。

---

## 五、近似块的修正

沿用 P3 spec §五 的数学，本文不重述，只记录两处本次确定的细节。

### 5.1 数学（摘要）

块 `x`，其余固定在 `z`，冻结 `σ̂` 后 `q = N(μ, M⁻¹)`，GCR 给出精确 iid 抽取。

```
log w = log p(x, z, d) + ½ (x−μ)ᵀ M (x−μ) + C
        └ log_joint(graph) ┘   └ 一次 M 的应用 ┘
```

`C = ½ log det M − (n/2) log 2π` 对每个抽取相同，自归一化与 MH 比值中整项抵消——**唯一一个无矩阵方法算不动的量，恰好不需要算**。

### 5.2 (A) 整图即一个块 → SNIS

`iterative_gls` 求不动点 σ̂（**同一份计算也是 `estimate()` 的答案**）→ `wiener_solve` 得 μ → `gcr_sample` 抽 N 个 iid → 逐样本 `log w` → `w̃ = softmax(log w)`。**无链、无 burn-in、无 r̂。**

### 5.3 (B) 块嵌在 Gibbs sweep 里 → 带反向密度的 MH

`σ̂ = σ(m(x))` 依赖当前状态，所以提议**不是**独立提议：

```
α = min( 1,  [ p(x') q(x | x') ] / [ p(x) q(x' | x) ] )
```

即在 `x'` 处**再建一次**提议来算 `q(x|x')`。**省掉反向项会得到一条接受率看起来正常、平稳分布是错的链**——§六有一条专门的测试守这个。

每 Gibbs 步 3 次 CG。`A` 不随 σ 改变，反向重建无需重新线性化。

### 5.4 诊断随结果返回

- **Kish ESS** `= 1/Σ w̃²` 与 `ESS/N`
- **k̂**：`<0.5` 有限方差；`0.5–0.7` 有限均值；**`≥0.7` 不可靠**
- 最大权重占比；(B) 的接受率

**不确定度除以 ESS，不除以 N。** k̂ ≥ 0.7 时结果对象带显式 `unreliable=True`。

`_psis_khat` 是私有的：调它 + 一条钉住它的测试；导入失败时降级为只报 Kish ESS，`khat=None`。**且它返回具体 float（§二），所以整个诊断在 jit 之外算。**

### 5.5 一个 k̂ 看不见的失效模式

`log q` 的正确性依赖 CG 真的收敛。CG 返回 `x̃ = x + e` 时实际抽样分布不是 `N(μ, M⁻¹)`，而权重公式假定它是 → 有偏。**k̂ 抓不到**：它测的是权重的尾部行为，不是 q 是否是真实的抽样分布。

两条缓解都要：`tol` 由编译期 κ 定；一条测试断言 x64 下 `tol=1e-6` 与 `tol=1e-12` 的**加权矩一致**。

### 5.6 不做的推广

同一套零件也能修正 `fn` 非线性的块，但那需要一个 MAP 点作线性化中心，P3 没有 MAP 求解器。**P3b 只对 σ 依赖的块自动分派修正；`fn` 非线性的块仍然拒绝，理由具名。**

---

## 六、InferencePlan 与返回类型

### 6.1 对象

```python
class Block(eqx.Module):            # 全静态字段
    latents: tuple[str, ...]
    method: str                     # "gcr" | "gcr+snis" | "gcr+mh" | "gls" | "nuts"
    reason: str
    linearity: dict | None          # {at_point: {scale: rel_err}}，或 None（未检验）
    kappa: float | None
    tol: float | None

class InferencePlan(eqx.Module):
    graph: Graph
    blocks: tuple[Block, ...]
    def __str__(self) -> str
    def sample(self, key, **kw) -> Posterior
    def estimate(self, **kw) -> Estimate
```

### 6.2 可打印——这个包最重要的用户体验

```
block 0  {alm}         GCR exact          linear_in ✓ 3 scales x 3 at-points (max 4.1e-08)
                                          kappa=3.4e+05 -> tol=2.9e-09, guard hoisted
block 1  {t_ant}       GCR + MH accept    sigma depends on this block (radiometer);
                                          proposal is frozen-sigma GCR, corrected exactly
block 2  {beam_fwhm}   NUTS               mu declares no linear_in for 'beam_fwhm'
execution: HMCGibbs(inner=NUTS, gibbs_sites=['alm', 't_ant'])
```

模型在被拟合之前，先告诉你它将如何被拟合，以及**为什么**。§4.2 的 κ 与 `tol` 必须在这张表里。§3.4 的拒绝理由也必须在这张表里。

### 6.3 返回类型

P3 spec 提到 `Posterior` 与 `Estimate` 但从未定义。现有 `nuts()` 返回裸 dict。

```python
class Posterior(NamedTuple):
    samples: dict[str, jax.Array]
    log_weights: jax.Array | None     # None 除非 SNIS
    ess: float                        # SNIS 走 Kish，其余走 numpyro
    khat: float | None                # None 表示 _psis_khat 不可得
    unreliable: bool                  # khat >= 0.7
    method: str

class Estimate(NamedTuple):
    values: dict[str, jax.Array]
    noise_std: dict[str, jax.Array]
    converged: bool
    residual: jax.Array
    iterations: jax.Array
```

**每条路径返回同一个类型**，加权与否由字段说了算。理由：设计文档附录二记录的 mcpost 三个缺陷之一，正是 `monte_carlo_integral` 除以 N 而非它自己算出的 ESS。一个**总是**带 `ess` 字段的返回类型让「除以 N」成为一件需要主动写出来的事，而一个有时是 dict、有时是富类型的 API 让它成为默认。

`estimate()` 在 `converged=False` 时抛 **`ConvergenceError`**——P3a 定义了它、测过兄弟关系，但无处 raise，这是设计中预期的第一个调用点。

### 6.4 `sample()` 的分派

| 图的形状 | 走什么 |
|---|---|
| 整图一个精确块，σ 不依赖块 | 直接 GCR——**iid 抽取，无链** |
| 整图一个块，σ 依赖块 | GCR 提议 + **SNIS**，返回加权 iid 样本 + k̂/ESS |
| 无精确结构 | NUTS |
| 混合 | **HMCGibbs**；精确块作 `gibbs_sites`，σ 依赖的块用 **MH 接受步** |

### 6.5 `estimate()` 的分派

| 图的形状 | 走什么 |
|---|---|
| 整图精确，σ 常数 | `wiener_solve` |
| 整图精确，σ 依赖预测 | `iterative_gls` |
| 混合 | **P3b 拒绝**，理由具名，指向 `sample()`。混合图 MAP 需块坐标下降 + optax，推到 P5 |

---

## 七、验收：两道关口

按 §1.3 的更正，验收分成两段——**它们的失败模式完全不同**，混在一批任务里诊断会互相掩盖。

### 7.1 关口一：结构，零 MCMC

分派器的分类与分区，对**手工推导**的答案，在 P3a 已有的 `tests/exact/models.py` fixture 上。全部确定性，无统计涨落，无 MCMC：

**下表的每一行都是算出来的，不是断言的**——用 P3a 的真实 `_ancestors` /
`check_linearity` / `check_prediction_dependence` 对每个 fixture 跑了一遍
（脚本见附录）。写这张表的初稿时有两行是错的，见表下的注。

| fixture | 候选集 | 联合检验 | σ | 必须得到 |
|---|---|---|---|---|
| `two_linear_latents` | `{a, b}` | ok (0.0) | 常数 | 一个块 `{a,b}`，`gcr` |
| `bilinear_pair` | `{gain, t_ant}` | **`StructureError`**「not JOINTLY affine」 | — | **整块 NUTS**，理由点名两个成员 |
| `radiometer` | `{w}` | ok (0.0) | 移动 2.50e+03 | `gcr+snis` |
| `radiometer_group` | `{a, b}` | ok (0.0) | 移动 4.40e+02 | 一个块 `{a,b}`，`gcr+snis` |
| `indirect_ancestor` | `{x}`（`tau` 出块） | ok | 常数 | `x` 精确，`tau` NUTS |
| `diamond_ancestor` | `{x}`（`tau` 出块） | ok | 常数 | 同上；`tau` 经**两条**路径到达 |
| `shared_ancestor` | `{x}`（`tau` 出块） | ok | 常数 | 同上 |
| `quadratic_claim` / `cubic_tail` | `{w}` | **`StructureError`** | — | NUTS，理由具名 |
| `collinear_pair` | `{a, b}` | ok (0.0) | 常数 | **共线父节点同块**（§3.3 的实测确认） |
| `unconstrained_latent` | `{w, u}` | ok (0.0) | 常数 | 一个块 `{w,u}`，`gcr` |
| `plated_latent` | `{z}` | ok (0.0) | 常数 | `gcr` |

> **实测更正（写本文时）**：本表初稿把 `unconstrained_latent` 写成「无 `linear_in`
> 声明 → NUTS」。**错的。** 实测 `u` 的候选判定为**合格**，且那是正确行为——
> 见 §3.4。

外加 `__str__` 的输出——κ 与 `tol` 在场，拒绝理由在场。

**一处已知的 fixture 空缺**：`models.py` 今天**没有**一个「隐变量分布本身不是高斯」
的图（`LyingNormal` 是给 `check_gaussian` 用的，它自称高斯）。合格性判据第 1 条
（`NotGaussian` → NUTS）因此没有 fixture 可测，P3b 必须**新增**一个。

外加 `__str__` 的输出——κ 与 `tol` 在场，拒绝理由在场。

### 7.2 关口二：统计

**（a）`gibbs_fn` 的条件正确性——最强的一条，且不需要 NUTS。**
固定 `hmc_sites`，`gibbs_fn` 的大量抽取的样本矩 vs **R2 稠密预言机**的解析条件后验。直接检验 `HMCGibbs` 契约里那句 "it is the user's responsibility to provide a correct implementation of `gibbs_fn`"——**那句 responsibility 就是 bayesmith 的产品**。确定性预言机、锐利判据、无 MCMC 混合问题。

**（b）MH 不变性。** 故意省掉反向密度项，样本矩必须**变红**。§5.3 的静默错误路径的唯一守卫。

**（c）SNIS 对 σ 依赖的修正。** 与长跑 NUTS 的后验矩一致；且**未修正的裸 GCR 必须显著偏离**——否则这个修正是空的。

**（d）CG 收敛对权重的影响。** x64 下 `tol=1e-6` vs `1e-12` 的加权矩一致（§5.5）。

**（e）复合 vs 纯 NUTS 的边缘矩。** ——**本条按 §1.3 重新设计**：

> 实测基准 C 里 `ESS(logw) = 3.0`（共 500 抽取）。在这种情况下比较 HMCGibbs 与纯 NUTS 的**边缘**后验矩，会因为外参数根本没混合而变红，**而原因与正确性无关**；接着自然的动作是放松容差直到它变绿——于是这条测试从此不再检验任何东西。
>
> 所以拆成两条：条件矩走 (a)（`hmc_sites` **固定**，对稠密预言机，确定性且锐利）；边缘矩只在一条**先实测过外参数确实混合**的 fixture 上比较，且该 fixture 的 `ESS` 与所用容差一起写进 docstring。

**（f）ESS/秒——具名区域的基准，不是 pass/fail 关口。**
在 κ ≳ 1e4 的 fixture 上断言**块内** ESS/秒 改善（实测 5.98x），并在同一条测试的 docstring 里如实写下**块外不改善**（实测 0.24x）及其机制。良态块上两个方向都不改善（基准 A、B），这也要写下来——否则将来有人在良态 fixture 上加一条同类断言，会得到一个无法通过的目标。

### 7.3 边界验证

按 `boundary-validation.md`：**绕开分派器**，在阈值两侧直接比较两侧方法。P3b 新增的阈值：

| 阈值 | 两侧 |
|---|---|
| k̂ = 0.7 | reliable / unreliable |
| `check_prediction_dependence` 的 `rtol` | `gcr` / `gcr+snis` 的分派边界 |
| MH 接受率 | 反向密度在场 / 缺席 |
| `condition_bound` 导出的 `tol` | 守卫开 / 关时的一致性 |

**极端参数值必须覆盖**：块大小 1 与 1e4；观测节点数 1 与 5；κ 从 1 到 1e10；σ 跨越六个量级；完全共线的两个父节点；零观测的隐变量。

### 7.4 每个任务的标准动作（沿用 P3a）

1. **变异测试**——故意破坏实现、确认一条**具名的**测试变红。
2. **AST 规格比对**——计划的代码块 vs 提交的文件，机器化。
3. **四条子判据**（P3a 用约 30 个缺陷换来的）：双侧扫描 / 点 vs 区域 / 结构维度 / `slow` 标记会拿掉守卫。
4. **变异不变红时**，按「fixture 到不了它声称的区域」→「变异是 no-op」→「测试写错了」的顺序诊断。

---

## 八、文件布局

```
src/bayesmith/exact/correct.py       log q / SNIS / Kish ESS / khat / MH 接受步   新写   ~200
src/bayesmith/exact/gibbs.py         gibbs_fn 工厂 + HMCGibbs 装配                新写   ~170
src/bayesmith/exact/block.py         +probe_gaussian 关键字（§1.2）                改     ~+15
src/bayesmith/dispatch/__init__.py   包标记                                        新写   ~1
src/bayesmith/dispatch/classify.py   合格性分类 + 分区推导                          新写   ~260
src/bayesmith/dispatch/plan.py       Block / InferencePlan / __str__ / sample / estimate  新写 ~260
src/bayesmith/__init__.py            +compile / InferencePlan / Posterior / Estimate 的惰性项  改
```

**依赖方向**：`exact/correct` → `exact/{block,solve}` + `graph/evaluate`；`exact/gibbs` → `exact/{block,solve,correct,gaussian}` + `bridge`；`dispatch/classify` → `exact/{gaussian,linearity,gls}` + `graph`；`dispatch/plan` → `dispatch/classify` + `exact/*` + `bridge`。无环（`exact → bridge` 已存在于 `linearity.py:157`）。

**惰性导入契约不破**：`bayesmith.compile` 走 `_LAZY_ATTRS`，`dispatch` 进 `_LAZY_SUBMODULES`。`test_importing_bayesmith_still_does_not_import_numpyro` 与 `errors.py` 的仅-stdlib 契约都必须保持绿。

---

## 九、明确不在本 spec 范围内

- **混合图的 MAP**（块坐标下降 + optax）——P5
- **`fn` 非线性块的修正**（需 MAP 作线性化中心）——P5
- **`identifiability` 的稠密 SVD**——P5
- **`propagate_covariance` / `push_forward`**——P5
- **RTS / Kalman 平滑器**——P4
- **离散枚举与前向-后向**——P4（`DiscreteHMCGibbs` 与 `MixedHMC` 已在 numpyro 里）
- **`MultivariateNormal` 稠密协方差的观测噪声**
- **嵌套 plate**（`Graph.__check_init__` 已带理由拒绝）
- **失败块的极大合格子集搜索**（§3.4，理由已写下）
- **`ruff format --check` 在 9 个 P1/P2 文件上的漂移**——值得单独一轮，不要顺手做

---

## 附录：本文所有数值的复现方式

§7.1 的分类表由一个脚本对每个 fixture 实算得出（`_ancestors` 求隐变量间祖先关系 → 去掉「是另一合格者的祖先」的 → 对余下集合跑 `check_linearity` → 通过者再跑 `check_prediction_dependence`）。P3b 计划的第一个任务应当把这个脚本变成 `tests/dispatch/test_classify.py` 的参数化表。

§1.3 的三组基准与 §一/§二 的所有 API 事实，都由这次会话在 `.venv` 里实测。基准的构造要点（写进 P3b 计划的对应任务，作为 fixture 的来源）：

* **基准 A**：`alm` plated `N(0, 1.5)`，`mu_i = exp(beam)·alm_i` 逐元素，`d ~ N(mu, 0.4)`。块的条件后验是**对角**的——NUTS 最容易的形态，也是精确抽取最没有优势的形态。
* **基准 B**：`mu = exp(beam)·(K alm)`，`K` 行归一化高斯平滑核。相关但**仍良态**（κ≈15），因为 `alm` 的先验 `N(0, 1.5)` 相对数据是紧的。
* **基准 C**：`alm ~ N(0, 100)`（宽先验 → 病态，κ=6.2e4），外参数是平滑宽度 `logw ~ N(log 4, 0.3)`，非线性且非纯乘性简并。**这是唯一一组块内 ESS/秒 改善的配置。**

三组的差别只在**先验宽度**与**外参数如何进入**——这两个维度决定了 HMCGibbs 是赢是输，而 P3 spec 的原始断言在两个维度上都没有取到两个值。这正是 P3a 记录的「结构维度」判据的又一个实例。
