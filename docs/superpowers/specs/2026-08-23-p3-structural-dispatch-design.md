# bayesmith P3 — 结构分派器 + InferencePlan + 线性高斯精确解

> 本文是设计（spec），不是实施计划。
> 上游：`docs/superpowers/specs/2026-08-23-bayesmith-design.md` §五 的 P3。
> 前置：P1 图核 + P2 NumPyro 桥（已合入 main，92 测试全绿）。
> 移植源：`/Users/zzhang/projects/e-RHINO/src/rheplicant/inference/`——**只读**。

## Context

P1 给了图与 `log_joint`，P2 给了 NUTS。两者合起来是一个可用但**没有任何东西是别人做不到的**包：追踪式 PPL 都能做到这一步。

P3 是第一个"别人没有"的能力，而它的全部依据是一件事：**显式图暴露了结构，于是可以按结构分派**。P1 里 `linear_in`、`support`、`depends_on_prediction` 三条结构轴被记录下来但没有任何东西读它们；P3 是它们第一次被读、第一次被检验、第一次决定计算路径。

### 与 NumPyro 的分工，逐条

设计文档立的纪律是"每写一行都要能回答 NumPyro 为什么做不了这个"。P3 的答案：

| 这一层 | 谁做 | 为什么 |
|---|---|---|
| 分区推导（哪些隐变量可精确、该分几块） | **bayesmith** | 需要图。追踪式 PPL 事后拿不到结构 |
| 线性高斯条件的精确解（Wiener / GCR） | **bayesmith**（移植 rheplicant） | NumPyro 无共轭分派 |
| 精确条件抽取的正确性守卫（线性性、κ、收敛） | **bayesmith** | 断言必须被检验 |
| 近似提议的修正（SNIS / MH 接受步） | **bayesmith** | 需要 `log_joint` 与 `log q` 两者，只有图两者都有 |
| Gibbs sweep 循环、warmup、多链、r̂、ESS、summary | **NumPyro**（`HMCGibbs`） | 已经存在且成熟 |
| 非精确块的转移核 | **NumPyro**（内层 NUTS） | 已经存在且成熟 |
| 分布库、HMC 内核、变量变换 | **NumPyro** | 设计文档 §三"不做的东西" |

`numpyro.infer.HMCGibbs` 的 docstring 写着 *"it is the user's responsibility to provide a correct implementation of `gibbs_fn` that samples from the corresponding posterior conditional."* ——**那句"responsibility"就是 bayesmith 的产品**。NumPyro 提供脚手架，无法知道哪个条件分布可精确求解，也无法求解它。

## 已定决策

| 项 | 决策 | 来源 |
|---|---|---|
| 高斯识别 | **内省做快路径，数值探测做守卫** | 用户 2026-08-23 |
| 执行边界 | **Gibbs 进 P3**，借 `numpyro.infer.HMCGibbs` | 用户 2026-08-23（取代先前的"混合图降级 NUTS"） |
| 验收预言机 | **NumPy 探基稠密解**（零 autodiff）+ 移植 Fisher 作第三条路径 | 用户 2026-08-23 |
| 近似块修正 | **(A) SNIS 与 (B) 带反向密度的 MH 都进**，按图的形状分派 | 用户 2026-08-23 |
| `linear_in` 未声明 | **不合格 → NUTS**（不声明就不解锁捷径） | 沿用 `nodes.py` 对 `support=None` 已立的原则 |
| 陪域 | 推广为 `{obs_name: array}` pytree（图可有多个观测节点） | 本文 §三 |
| 复数隐变量 | `_real_parts` **照移**（天空 alm 是复数） | 本文 §三 |
| 公开名 | `bayesmith.compile(graph)`，遮蔽内置 `compile` 由 docstring 点名 | 设计文档 §二 的既定 UX |

---

## 一、接缝：图里的"预测"是什么

rheplicant 的 `linear_operator(space, pipeline, state_template, name=...)` 从 `ParameterSpace` 取"隐变量与先验"、从 `pipeline` 取"预测"。bayesmith 没有 `ParameterSpace`，只有图，所以两者都必须由图结构定义：

```
g : {latent_name: x}  ->  {obs_name: loc}
```

即**每个观测节点的分布的位置参数**，作为一个 pytree。

### 1.1 唯一的提取器

```python
def gaussian_parts(graph, node, env) -> tuple[Array, Array]:
    """(loc, scale) of a Probabilistic node's distribution, verified."""
```

**快路径（内省）**：`apply_probabilistic(graph, node, env)`（复用 P1 的共享扫描）拿到 `dist`；剥掉 `Independent` 包装（`.to_event()` 与 plate 都可能产生它）；要求 `isinstance(dist, numpyro.distributions.Normal)`；取 `loc = dist.loc`、`scale = jnp.broadcast_to(dist.scale, jnp.shape(dist.loc))`。

**守卫（数值探测）**：在至少 3 个跨越 `scale` 的探针 y 上验证

```
dist.log_prob(y)  ==  -0.5*((y-loc)/scale)**2 - log(scale) - 0.5*log(2*pi)
```

相对容差取 `1e3 * eps(dtype)`。不通过则抛 `StructureError`（§1.4），**不降级为"大概是高斯"**。

`MultivariateNormal`（稠密协方差）P3 拒绝，理由具名，指向后续阶段。

### 1.2 同一个提取器读三样东西

| 读什么 | 从哪个节点 | 得到 |
|---|---|---|
| 预测与噪声 | 观测 `Probabilistic` 节点 | `loc` = 预测，`scale` = σ |
| 先验 | 隐变量 `Probabilistic` 节点 | `loc` = 先验均值 m，`scale` = 先验标准差 √S |

**只有一个提取器。** P1 记录的教训是"两条扫描共享实现就共享盲点"；这里反过来利用它——守卫比对的是 `log_prob` 本身，所以即使提取器有 bug，先验与噪声两处会被**各自独立地**抓住。

### 1.4 两个新异常类

`errors.py` 今天只有 `BayesmithError` / `GraphError` / `TraceError`。P3 加两个，**仍然只依赖 stdlib**（该模块的"不导入 jax/numpy"契约由 `test_errors_module_imports_no_heavy_dependency` 强制）：

| 类 | 何时抛 | 基类 |
|---|---|---|
| `StructureError` | **一个被声明的结构断言经检验为假**——`linear_in` 声明了但线性性探针不过；`depends_on_prediction=False` 但 σ 确实依赖；`dist_fn` 自称高斯（内省通过）但 `log_prob` 探针不吻合 | `BayesmithError, ValueError` |
| `ConvergenceError` | 一个迭代过程在**非 jit** 路径上未达到要求的精度（`iterative_gls` 的 `converged=False` 被要求为硬失败时） | `BayesmithError, RuntimeError` |

**"图不合格"不是错误**，是分派结果——落到 NUTS，理由写进 plan。`StructureError` 只在**声明与事实矛盾**时抛，这是两件不同的事：前者是"你没说"，后者是"你说错了"。

jit 内的数值守卫仍走 `eqx.error_if`（`_conjugate_solve` 的既有做法）——Python 的 `if` 对 traced 值无效。

### 1.3 图使一整批 rheplicant 守卫变得无意义

rheplicant `linear.py` 约 250 行在处理"`prior_std=` / `noise_std=` 关键字与 `Latent(prior=...)` 声明并存时谁赢"，以及"一维 sigma 该沿哪个轴广播"。

bayesmith 里先验只有一个来源（隐变量自己的 `dist_fn`），σ 也只有一个来源（观测节点的 `.scale`，已由 numpyro 广播到 `loc` 的形状）。**没有第二个来源就没有冲突可仲裁，没有轴可猜。** 不移植清单见 §八。

同理：rheplicant 的探针幅度取自 `max|init|`，其文档明写"全零 init 没有尺度可取，退化为绝对探针"这个坑。bayesmith 没有 `init`，探针幅度取自**声明的先验标准差**——那正是"这个隐变量有多大"的正确陈述。坑随之消失。

---

## 二、合格性分类

### 2.1 判据

隐变量 `x` 合格走精确路径，当且仅当**四条全部成立**：

1. `x` 自己的分布是对角高斯（§1.1 提取器通过）；
2. `x` 的先验的父节点**不含另一个候选块成员**——块内分层（`x ~ N(m(z), s(z))` 且 `z` 也在块里）不是线性高斯，联合分布不再是高斯；
3. 每个观测节点都是对角高斯；
4. 从 `x` 到每个观测节点的 `loc` 的**每一条**路径，沿途每个 `Deterministic` 节点都在 `linear_in` 里声明了其入边父节点。

**"每一条"是必需的**：`x` 经两条路径到达 `mu`，其一线性其一不线性，合成就不仿射。

**未声明 `linear_in` → 不合格 → NUTS。** 与 `nodes.py` 已写下的原则一致（`support=None` 视为不合格而非"声明连续"）。

### 2.2 声明必须被检验

第 4 条是**断言**。检验它的是移植自 rheplicant 的 `check_linearity`：把块隔离成 `g: x -> {obs: loc}`，与其在零点的线性化比较，在三个尺度上（默认 `1e-3, 1.0, 1e3` 倍先验标准差）测量相对偏离。

**两处相对 rheplicant 的改动：**

- 探针幅度取自先验标准差而非 `max|init|`（§1.3）；
- **在多个 `at` 点上重复探测，而非只在一个点。** `at` 是块外隐变量的取值，而 Gibbs sweep 里块会在不同 `at` 处反复重建。只在一个点验过就在所有点使用，是 `boundary-validation.md` 明令禁止的"moderate-parameter probe"。编译期从块外隐变量的先验抽 3 个 `at` 点，逐点探测。

### 2.3 分区

P3 的分区是**从图推导**的，不是用户声明的。这是 bayesmith 相对 rheplicant 的核心增量：

| rheplicant 的手段 | bayesmith 的对应 |
|---|---|
| 用户手写 `Block("t_nw", "t_ant")` | 合格隐变量按图的耦合自动成块 |
| 稠密 Jacobian + 稠密 SVD 的 `identifiability` 抓双线性分区 | **联合 `check_linearity` 探针**（3 次前向求值）——`gain × T_ant` 声明 `linear_in=("gain","t_ant")` 是假断言，联合探针直接抓住 |
| SVD 抓完全共线 | 共线的父节点**被自动分到同一块**（图看得见它们是同一子节点的父），联合 κ 因而是诚实的 |
| 分区覆盖性检查（漏掉 / 重复隐变量） | 图的隐变量集合是权威的，分区按构造穷尽且不重叠 |

rheplicant 的 `plan.py` 用 60 行 docstring 记录了分区错误的代价：双线性 `gain × T_ant` 手工交替，**CG 残差 1e-7、逐块 κ≈1.47、`check_linearity` 每轮都过，答案错几千开尔文**——因为每个条件分布确实是仿射的，错的是分区，而任何逐块的数字都无权发现它。

**把 Gibbs 留在应用端，意味着每个应用手工重新推导一次这个分区。** 分区能被推导的地方只有一个：有图的地方。

`identifiability` 的 SVD 仍有独立价值（它能说出病态方向的**名字**），但那是 P5 的诊断，不是 P3 的前置条件。

### 2.4 σ 依赖块自身

若观测节点的 `scale` 依赖任一块成员（判据：`jax.jacfwd` 对块成员的导数非零），条件分布**不是高斯的**：

```
p(x | rest, d)  ∝  exp( -½ Σ (d-m(x))²/σ(x)²  -  Σ log σ(x) ) · π(x)
```

裸 GCR 因此**不是**精确条件抽取。处理见 §五。

节点上的 `depends_on_prediction` 轴在这里第一次被读：
- 声明 `False` 却探测出依赖 → 抛错（声明是假的）；
- 声明 `True` 而探测出无依赖 → 走直接路径，plan 注明"声明保守"。

**语义定义**（与 rheplicant 有别，需记录）：rheplicant 的 `NoiseModel.depends_on_prediction` 意为"σ 依赖它自己预测的那个量"。bayesmith 读作"**σ 依赖任一块成员**"。单观测节点时二者等价；多观测节点时 bayesmith 的读法更宽，即更保守。

---

## 三、精确解（移植）

### 3.1 移植的核心

| 从 | 到 | 性质 |
|---|---|---|
| `linear.py::LinearBlock` | `exact/block.py` | 改写：`linear_operator(graph, names, at)` 取代 `(space, pipeline, state_template, name)` |
| `linear.py::check_linearity` + `_affinity_errors` | `exact/linearity.py` | 改写：探针幅度取自先验；多 `at` 点 |
| `linear.py::wiener_solve / gcr_sample / _conjugate_solve / _normal_operator / condition_estimate` | `exact/solve.py` | 移植，陪域推广为 pytree |
| `linear.py::_real_parts / _domain_zero / _domain_centre / _variance_parts / _largest_variance` | `exact/block.py` | 逐字 |
| `conditioning.py`（全部） | `exact/conditioning.py` | 逐字（`tree_norm`、`largest_eigenvalue`、`extreme_eigenvalues`） |
| `gls.py`（全部） | `exact/gls.py` | 移植 |
| `uncertainty.py::FlatMatrix / _named_spans / fisher_information / parameter_covariance` | `exact/fisher.py` | 移植子集 |

### 3.2 一处必要推广：陪域是 pytree

rheplicant 的陪域是单个数组（一次观测）。bayesmith 的图可以有多个观测节点，所以 `observed`、`offset`、`noise_std` 都变成 `{obs_name: array}`：

- `residual_data = tree_map(sub, observed, offset)`
- `weight = tree_map(lambda s: 1/s**2, noise_std)`
- `half_chi2` 跨叶求和
- `pair_with(v)` 的配对跨叶求和
- `omega_data` 逐叶抽取

约 10 行改动。定义域早已是 pytree（分组块），所以 `jax.tree.map` 与 `cg` 都不需要改。

### 3.3 复数隐变量

`_real_parts` 照移。天空 alm 系数是复数，而预测是实的，所以映射是 **ℝ-线性而非 ℂ-线性**，Krylov 方法必须跑在实自由度上。现在不移植将来要痛苦回填。

### 3.4 x64

**bayesmith 绝不在任何位置调用 `jax.config.update("jax_enable_x64", ...)`**（设计文档 §六的硬约束——进程级全局，关不回去）。需要 float64 的地方用 `with jax.enable_x64(True):`，并在块内转出到 NumPy。

`solve.py` 的 docstring 必须说明：κ·eps 超过 `require_convergence` 时**只有精度能救**，这是 rheplicant `_conjugate_solve` 已经写好的那条错误消息。测试用 pyproject 已声明的 `x64` marker 单独跑。

---

## 四、Gibbs：借 `HMCGibbs`

### 4.1 装配

```python
kernel = HMCGibbs(NUTS(to_numpyro(graph)), gibbs_fn=..., gibbs_sites=[...块成员...])
mcmc   = MCMC(kernel, num_warmup=..., num_samples=..., num_chains=...)
```

`to_numpyro(graph)`（P2）已经把每个节点开成同名的 `numpyro.sample` site，所以 `gibbs_sites` 就是块成员的节点名。**P2 全量复用，零适配层。**

bayesmith 只写 `gibbs_fn`：

```python
def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
    at    = {**hmc_sites}                       # 块外隐变量的当前取值
    block = linear_operator(graph, names, at=at, check=False)   # 每 sweep 重建
    return gcr_sample(block, observed, noise_std=σ, prior_std=..., key=rng_key)
```

**不移植** `plan.py` 的 sweep 循环、`split_rhat`、`PlanDiagnostics`、chi² 监控，也不移植 `engines.py` 的梯度引擎与 Adam——NumPyro 的 `MCMC` 全部提供，且 r̂ / ESS 走 `numpyro.diagnostics`。

### 4.2 三个必须外提的守卫

`gibbs_fn` 在 jit 下每 sweep 运行：

1. **`check_linearity` 不能进循环**——它用 Python float 和 `raise`。编译期检查（多 `at` 点，§2.2），循环内 `check=False`。
2. **`condition_estimate` 不能进循环**——每次 `2 × POWER_ITERATIONS = 24` 次算子应用。编译期算一次 κ，**据此设 `tol ≈ require_convergence / κ`**，循环内 `require_convergence=None`。
3. **关掉守卫必须同时收紧 `tol`。** rheplicant 明写"leave `tol` at its default and the guard off"正是它返回过静默过窄后验的组合。plan 必须把这两个数打印出来。

### 4.3 上游接口风险

`HMCGibbs` 标着 `[EXPERIMENTAL INTERFACE]`。按本包对 equinox `BoundMethod` 的既有做法：写一条测试**钉住 `gibbs_fn(rng_key, gibbs_sites, hmc_sites)` 的调用约定**，让上游改动变成一个自觉的决定而不是意外。numpyro 版本在 pyproject 里下界收紧。

---

## 五、近似块的修正

### 5.1 为什么需要

§2.4：σ 依赖块自身时，冻结 σ 的 GCR 是一个**合法的提议**，不是条件抽取。

而这正好是 rheplicant `gls.py` 自己点名的缺口：冻结 σ 使每步成为线性高斯问题，也正是这一步让收敛结果是 GLS 而不是完整似然的最大值——*"the log-determinant's dependence on the solution is held fixed rather than differentiated"*。

**重要性权重里包含的恰好就是被冻结的那两项**：χ² 的重加权差，以及 `−Σ log σ(x)` 这个被 q 换成常数的对数行列式项。这不是给近似打补丁，是把那道被明确记录的差精确还回去。

### 5.2 数学

块 `x`，其余固定在 `z`。冻结 `σ̂` 后：

```
M = AᵀN̂⁻¹A + S⁻¹ ,   μ = M⁻¹[AᵀN̂⁻¹(d−offset) + S⁻¹m₀]
q = N(μ, M⁻¹)          ← GCR 给出精确 iid 抽取
```

```
log w = log p(x, z, d) + ½ (x−μ)ᵀ M (x−μ) + C
        └ log_joint(graph) ┘   └ 一次 M 的应用 ┘
```

**`C = ½ log det M − (n/2) log 2π` 对每个抽取都相同，自归一化与 MH 比值中整项抵消。** 我们从来不需要 `log det M`——那是唯一一个无矩阵方法算不动的量，而它恰好不需要算。

每抽取的边际成本：一次 `M` 应用（一 JVP + 一 VJP）+ 一次图扫描。两者都可 `vmap`。

### 5.3 (A) 整图即一个块 → SNIS

1. `iterative_gls` 求不动点 `σ̂` 与中心（**同一份计算也是 `estimate()` 的答案**）；
2. `wiener_solve` 得 `μ`（一次 CG）；
3. `gcr_sample` 抽 N 个 iid 样本；
4. `log w` 逐样本；`w̃ = softmax(log w)`；
5. 返回加权样本 + 诊断。

**无链、无 burn-in、无 r̂。** 这是大线性块（10⁶ 维天空 alm）最想要的形态。

### 5.4 (B) 块嵌在 Gibbs sweep 里 → 带反向密度的 MH

`gibbs_fn` 改为：提议 + 接受步。

**关键细节，写错会得到一个静默错误的平稳分布：** `σ̂ = σ(m(x))` 依赖当前状态，所以提议**不是**独立提议，接受比必须带反向密度：

```
α = min( 1,  [ p(x') q(x | x') ] / [ p(x) q(x' | x) ] )
```

即在 `x'` 处**再建一次**提议来算 `q(x|x')`。省掉反向项会得到一条接受率看起来正常、平稳分布是错的链。

每 Gibbs 步的成本：正向 `μ`（1 CG）+ GCR 抽取（1 CG）+ 反向 `μ'`（1 CG）+ 2 次算子应用 = **3 次 CG**。相对一条 NUTS 轨迹（32–1000 次前向+反向）微不足道。`A` 不随 σ 改变，反向重建无需重新线性化。

**正当性**：rheplicant `Block(steps=)` 的 docstring 已经写过——有限步 NUTS *"is a transition that merely leaves the conditional invariant. The scheme is then Metropolis-within-Gibbs — valid"*。MH 步是同一句话的另一个实例，且其不变性是**精确的**，不像有限步 NUTS 还要论证。

`HMCGibbs` 的 `gibbs_fn(rng_key, gibbs_sites, hmc_sites)` 里 `gibbs_sites` 就是当前 `x`——契约天然吻合。

### 5.5 诊断，随结果返回而非可选

- **Kish ESS** `= 1/Σ w̃²`，以及 `ESS/N`
- **k̂**（PSIS，Vehtari 等）：`<0.5` 有限方差；`0.5–0.7` 有限均值；**`≥0.7` 不可靠**
- 最大权重占比
- (B) 的接受率

**不确定度除以 ESS，不除以 N。** 设计文档附录二记录的 mcpost 三个缺陷之一正是这个（`monte_carlo_integral` 除以 N 而非它自己算出的 ESS）。bayesmith 从第一天起就要对。

k̂ ≥ 0.7 时结果对象带**显式 `unreliable=True` 标记**，而不是静默返回数字。

**k̂ 的 API 风险**：公开的 `numpyro.infer.psis_diagnostic` 只接受 model/guide 对，不适用；私有的 `numpyro.infer.importance._psis_khat(log_weights)` 正好吃 log 权重。决策：**调私有函数 + 一条钉住它的测试**（本包对 equinox `BoundMethod` 就是这么做的）；导入失败时降级为只报 Kish ESS，并在结果里注明 k̂ 不可得。

### 5.6 一个 k̂ 看不见的失效模式

**`log q` 的正确性依赖 CG 真的收敛。** 若 CG 返回 `x̃ = x + e`，实际抽样分布不是 `N(μ, M⁻¹)`，而权重公式假定它是 → 有偏。

**k̂ 抓不到这个**：k̂ 测的是权重的尾部行为，不是 q 是否是真实的抽样分布。未收敛的 GCR + 健康的 k̂ = 这条路径唯一的静默错误。

两条缓解都要：
1. `tol` 由编译期 κ 定（§4.2），不留在默认值；
2. 一条测试：x64 下 `tol=1e-6` 与 `tol=1e-12` 的**加权矩必须一致**。

### 5.7 不做的推广

同一套零件也能修正 `fn` 非线性的块（`check_linearity` 失败者）——代码一行不改。但那需要一个 MAP 点作线性化中心，P3 没有 MAP 求解器（块坐标下降 + Adam 需要 optax，是新依赖）。**P3 只对 σ 依赖的块自动分派修正；`fn` 非线性的块仍然拒绝，理由具名。**

---

## 六、InferencePlan

### 6.1 对象

```python
class Block(eqx.Module):        # 全静态字段
    latents: tuple[str, ...]
    method: str        # "gcr" | "gcr+snis" | "gcr+mh" | "gls" | "nuts"
    reason: str
    linearity: dict | None      # {scale: rel_err} per at-point，或 None（未检验）
    kappa: float | None
    tol: float | None

class InferencePlan(eqx.Module):
    graph: Graph
    blocks: tuple[Block, ...]
    def __str__(self) -> str
    def sample(self, key, **kw) -> Posterior
    def estimate(self, **kw) -> Estimate
```

### 6.2 可打印

```
block 0  {alm}         GCR exact          linear_in ✓ 3 scales x 3 at-points (max 4.1e-08)
                                          kappa=3.4e+05 -> tol=2.9e-09, guard hoisted
block 1  {t_ant}       GCR + MH accept    sigma depends on this block (radiometer);
                                          proposal is frozen-sigma GCR, corrected exactly
block 2  {beam_fwhm}   NUTS               mu declares no linear_in for 'beam_fwhm'
execution: HMCGibbs(inner=NUTS, gibbs_sites=['alm', 't_ant'])
```

**这是这个包最重要的用户体验**（设计文档 §二）：模型在被拟合之前，先告诉你它将如何被拟合，以及**为什么**。

### 6.3 `sample()` 的分派

| 图的形状 | 走什么 |
|---|---|
| 整图一个精确块，σ 不依赖块 | 直接 GCR——**iid 抽取，无链** |
| 整图一个块，σ 依赖块 | GCR 提议 + **SNIS**（§5.3），返回加权 iid 样本 + k̂/ESS |
| 无精确结构 | NUTS |
| 混合 | **HMCGibbs**；精确块作 `gibbs_sites`，σ 依赖的块用 **MH 接受步**（§5.4） |

### 6.4 `estimate()` 的分派

| 图的形状 | 走什么 |
|---|---|
| 整图精确，σ 常数 | `wiener_solve` |
| 整图精确，σ 依赖预测 | `iterative_gls` |
| 混合 | **P3 拒绝**，指向 `sample()`。混合图 MAP 需块坐标下降 + optax，推到 P5 |

---

## 七、验收

### 7.1 四条路线，其中一条与被测者零共享

| 路线 | 实现 | 与 R1 共享 |
|---|---|---|
| **R1** 无矩阵 CG | `linearize`/`vjp` + `jax.scipy.sparse.linalg.cg` | 被测者 |
| **R2** NumPy 探基稠密 | `A[:,j] = g(e_j) − g(0)`，`numpy.linalg.solve` | **零**——无 autodiff、无 jax |
| **R3** 稠密 Fisher | `jacfwd` + `parameter_covariance` | JAX autodiff |
| **R4** NUTS | numpyro | `apply_probabilistic` |

**R2 是主预言机**，因为它与 R1 一行代码都不共享。设计文档记录的"自洽性盲区"（两条扫描共享实现因此共享盲点，实测 −225.65 与正确值 −364.95 彼此一致地都错）在这里被正面回应：R1 vs R4 是自洽检验（共用 `apply_probabilistic`），R1 vs R2 不是。

判据：
- R1 vs R2：x64 下**均值与协方差**都 rtol 1e-8；
- GCR 抽样的样本协方差 vs R2 解析协方差：MC 误差内；
- R1 vs R4：后验矩 ESS≥400 时 z<4；
- R3 与前两者一致。

### 7.2 修正路径的验收

- **条件正确性**：固定 `hmc_sites`，`gibbs_fn` 的大量抽取的样本矩 vs R2 的解析条件后验——直接检验 `HMCGibbs` 契约里那句 "user's responsibility"。
- **MH 不变性**：故意省掉反向密度项，样本矩必须**变红**（§5.4 的静默错误路径的守卫）。
- **复合 vs 纯 NUTS**：同一张混合图，HMCGibbs 与纯 NUTS 的后验矩在 MC 误差内一致，且 HMCGibbs 的 **ESS/秒更高**——否则这条路径没有存在理由。
- **SNIS 对 σ 依赖的修正**：与长跑 NUTS 的后验矩一致；且**未修正**的裸 GCR 必须显著偏离（否则这个修正是空的）。
- **CG 收敛对权重的影响**：x64 下 `tol=1e-6` vs `1e-12` 的加权矩一致（§5.6）。

### 7.3 边界验证

按 `boundary-validation.md`：**绕开分派器**，在阈值两侧直接比较两侧方法。阈值清单：

| 阈值 | 两侧 |
|---|---|
| `check_linearity` 的 `rtol`（默认 `1e4·eps`） | 接受 / 拒绝，在恰好该偏离处 |
| 逐探针 roundoff `floor` | 真曲率 / 舍入噪声 |
| 高斯守卫的 rtol | 高斯 / 非高斯 |
| `require_convergence=1e-3` 与 κ 的乘积 | 通过 / 报错，含 `κ·eps` 那条独立分支 |
| GLS 的 `reweight_tol = max(8·eps, tol)` | 收敛 / 未收敛 |
| `min_reweights` | 早停 / 未早停 |
| k̂ = 0.7 | reliable / unreliable |

**极端参数值必须覆盖**（方法论硬要求，失效模式常呈 U 形）：plate 大小 1 与 1e5；`prior_std` 1e-6 与 1e6；κ 从 1 到 1e10；完全共线的两个父节点；零观测的隐变量；σ 跨越六个量级；块大小 1 与 1e4；观测节点数 1 与 5。

### 7.4 每个任务的标准动作

P1 的执行记录留下两条方法论，P3 原样采用：

1. **变异测试**——故意破坏实现、确认一条**具名的**测试变红。P1 的 8 个缺陷里有 3 个只有它能发现。
2. **AST 规格比对**——计划的代码块 vs 提交的文件，机器化，几秒钟，能区分实质差异与排版差异。

---

## 八、文件布局与规模

```
src/bayesmith/exact/
  gaussian.py      提取器 + 守卫                                  新写   ~150
  conditioning.py  tree_norm / largest / extreme_eigenvalues      逐字   ~120
  block.py         LinearBlock、linear_operator(graph,...)、域工具 改写   ~240
  linearity.py     check_linearity（先验幅度、多 at 点）          改写   ~170
  solve.py         wiener_solve / gcr_sample / condition_estimate 移植   ~330
  gls.py           iterative_gls / GLSResult                      移植   ~200
  fisher.py        FlatMatrix / fisher_information / param_cov    移植   ~250
  correct.py       log q / SNIS / Kish ESS / khat / MH 接受步     新写   ~170
  gibbs.py         gibbs_fn 工厂 + HMCGibbs 装配                  新写   ~150
src/bayesmith/compile/
  dispatch.py      合格性分类 + 分区推导                          新写   ~260
  plan.py          Block / InferencePlan / __str__ / sample / estimate  新写 ~230
```

外加 `errors.py` 增补两个类（§1.4，约 +20 行）。源码约 2290 行。每个文件都在 200–400 典型区间内，无一超 800（`coding-style.md`）。

### 分两个实施计划

本 spec 是**一个**设计，但体量应拆成**两个** plan 文档：

- **P3a 精确解核心**：`gaussian` / `conditioning` / `block` / `linearity` / `solve` / `gls` / `fisher`（~1460 行）。验收关口：**R1 vs R2 在 x64 下 rtol 1e-8**。这一半可独立验证，不依赖分派器。
- **P3b 分派与执行**：`dispatch` / `plan` / `gibbs` / `correct`（~810 行）。验收关口：**HMCGibbs 与纯 NUTS 的后验矩一致且 ESS/秒 更高**，以及 §7.2 全部。

P3b 依赖 P3a。

---

## 九、明确不在本 spec 范围内

留给后续，不要顺手做：

- **Gibbs 的梯度引擎与 Adam**（`engines.py` 的另一半）——内层 NUTS 由 NumPyro 提供
- **混合图的 MAP**（块坐标下降 + optax，新依赖）——P5
- **`fn` 非线性块的修正**（需 MAP 作线性化中心）——P5
- **`identifiability` 的稠密 SVD**——P5
- **`propagate_covariance` / `push_forward`**（delta 法、后验推前）——P5
- **RTS / Kalman 平滑器**（`chain.py`）——P4
- **离散枚举与前向-后向**——P4（注：`DiscreteHMCGibbs` 与 `MixedHMC` 已在 numpyro 里，P4 同样只需写"精确边际化"那一半）
- **`MultivariateNormal` 稠密协方差的观测噪声**
- **嵌套 plate**（`Graph.__check_init__` 已带理由拒绝）
- **流式证据层**——P6
- **mcpost 融合**——P7（bayesmith 在 P3 就把 SNIS + ESS + k̂ 做对，P7 只剩"把 GSA 作用在加权样本上"）

---

## 附录一：本次会话核实的 API 事实

在 `/Users/zzhang/projects/bayesmith/.venv` 实测（2026-08-23）：

| 事实 | 值 |
|---|---|
| 版本 | jax 0.11.1 / equinox 0.13.8 / numpyro 0.21.0 / numpy 2.5.2 |
| `jax.scipy.sparse.linalg.cg` 签名 | `(A, b, x0=None, *, tol=1e-05, atol=0.0, maxiter=None, M=None)`——**仍是 `tol=`**，移植可逐字 |
| `equinox.error_if` | 存在 |
| `jax.enable_x64` 上下文管理器 | 存在（设计文档已验证：线程局部、计入 jit key、可嵌套、退出完全还原） |
| 默认 dtype | float32（未开 x64） |
| `numpyro.infer.HMCGibbs` | `(inner_kernel, gibbs_fn, gibbs_sites)`，`[EXPERIMENTAL INTERFACE]` |
| `gibbs_fn` 签名 | `(rng_key, gibbs_sites, hmc_sites) -> dict` |
| `numpyro.infer.DiscreteHMCGibbs` / `MixedHMC` | 存在——P4 的离散行 |
| `numpyro.infer.psis_diagnostic` | 只接受 model/guide 对，**不适用** |
| `numpyro.infer.importance._psis_khat(log_weights)` | 存在但**私有**——见 §5.5 的决策 |
| `numpyro.diagnostics` | `effective_sample_size` / `split_gelman_rubin` / `summary` / `print_summary` |
| rheplicant 的 `Likelihood` | `eqx.Module` 实现的 Protocol，**不是** numpyro 分布——接入时写成 `dist.Normal(mu, noise.std(mu))` |

## 附录二：不移植清单及理由

以下 rheplicant 代码**不进 bayesmith**，因为图使它们无意义（§1.3）：

| 不移植 | 理由 |
|---|---|
| `_reconcile` / `_agrees` / `_resolve_prior` / `_per_member` | 仲裁"关键字 vs 声明"的冲突。bayesmith 的先验只有一个来源 |
| `_require_prior_std` | 同上 |
| `_refuse_a_noise_model_at_the_conjugate_seam` | σ 从分布对象导出，调用方无从传入 `NoiseModel` |
| `check_noise_std_axis` | σ 已由 numpyro 广播到 `loc` 的形状，一维轴歧义不可能出现 |
| `_resolve_name` / `_resolve_names` / `_BOTH_SPELLINGS` | 图的隐变量名是权威的；无 `name=` / `names=` 双拼写 |
| `ParameterSpace` / `Latent` / `State` / `AbstractOperator` 相关全部 | 图取代之 |
| `plan.py` 的 sweep 循环 / `split_rhat` / `PlanDiagnostics` / chi² 监控 | NumPyro 的 `MCMC` + `numpyro.diagnostics` 提供 |
| `plan.py::Block(steps=/learning_rate=/engine=)` | 引擎从图推导；NUTS 自适应步长 |
| `engines.py` 的梯度引擎与 `_adam` | 内层 NUTS 由 NumPyro 提供 |
