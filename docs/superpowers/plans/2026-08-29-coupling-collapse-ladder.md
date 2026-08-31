# 计划：耦合度量、logdet 阶梯、图归约

> **文档状态：`record`** · 已落地批次/审计/测量的历史记录，写作当天为真，非当前权威。索引见 docs/README.md。

**日期**：2026-08-29
**来源**：一次 12-agent 的设计审计，全部结论在本 checkout 实测（M1–M10，见 §0）
**形态**：Wave 1 的四个包**文件互不重叠**，可在四个 worktree 并行；Wave 2 有依赖。

---

## 0. 这个计划要解决的问题，以及支撑它的实测

`compile()` 今天对一个混合图（线性块 + NUTS 块）只有一条路：`HMCGibbs`，线性块进
`gibbs_sites`。两块 Gibbs 的收敛率由两块之间的**最大典型相关** `c` 决定，积分自相关
时间 `τ(c) = (1+c²)/(1−c²)` 是**线性于条件数**的；而联合 HMC 只付 `√κ`。所以当两块
相关时，分块是平方对一次方的亏损。

线性块和非线性块**通有地**相关，因为线性算子是在当前非线性值处构造的
（`exact/gibbs.py`: `unchecked_operator(graph, names, at=at)`）。

出路有二：**边缘化**（把线性块从 NUTS 目标里积掉，之后精确回抽）——脊不再存在；
**重参数化**——脊被扶正。把线性块**并进** NUTS 永远是错的：那是拿一个精确独立抽样
换一个自相关抽样。

### 实测结论（每条都是本 checkout 的实读，不是推理）

* **M1 — 廉价的「可分性证书」测的是错的量。** 在 `mu = x·1₂₀ + exp(th)` 上
  （`x ~ N(0,10)` 声明 `linear_in`，`th ~ N(0,1)`，`sigma=0.5`），`compile()` 给出两块
  混合计划；混合二阶导 `max|∂²μ/∂x∂th|` = **0.0 bitwise**，`sigma_movement` =
  **0.000e+00**，而众数处真实后验相关 = **−0.9942708**（τ = 174）。
  **加性进入的非线性项——前景、offset、非线性基线——全落在这个洞里。**
  正确的判据不是「th 是否进入 A」，而是 **`∂μ/∂th` 落在 `range(A)` 里多少**。
  **本计划因此没有可分性阶段。**

* **M2 — Laplace 路把它看对了。** `local_block(graph, ('x','th'), mode, priors=True)`
  → `fisher_information(include_prior=True)` 给相关 **−0.9940992**（真值 −0.9942708）。

* **M3 — traced 路径必须用 `unchecked_operator`，不能用 `local_block`。**
  `local_block(..., priors=True)` 在 `diagnose/local.py:293` 以 `probe_gaussian` 默认
  True 调 `_env_before`，jit 下从 `exact/gaussian.py:249` 抛 `TracerBoolConversionError`。
  `unchecked_operator(graph, block, at=theta, probe_gaussian=False)` 可 jit 可 grad，
  两路同值 32.974425。注意 `unchecked_operator` 的 `at` 是**块外**的 latent，它在定义域
  零点线性化块本身——对「给定 `at` 后对块仿射」的块，这正是**正确**的矩阵，而这恰是
  collapse 的前提。「对会移动的 theta 而言是错的矩阵」这个说法是错的：theta 不在块里。

* **M4 — collapse 的算术端到端成立，且 theta 在设计矩阵内。**
  `mu = th·(B x)`，B 12×4，`x ~ N(0, 2²I₄)`（**故意非单位宽度**），`sigma=0.5`。链路
  `unchecked_operator(probe_gaussian=False)` → `dense_operator` → `compress` →
  `SqrtInfo.combine(nuisance_prior)` → `marginalise_arrays`，对稠密 `slogdet` oracle：

  ```
  th=0.7  −20.972299055165  vs  −20.972299055165   (1.8e−14)
  th=1.0  −22.306147021139  vs  −22.306147021139   (7.1e−15)
  th=2.5  −25.895172785665  vs  −25.895172785665   (3.6e−15)
  梯度 −2.9944287322  vs oracle 中心差分 −2.9944287583  (2.6e−08)
  jit: clean.  hessian: clean.
  ```

  **log-determinant 没有被假设掉。** `sqrtinfo.py:277` 把 `−Σ log pivots[:n_block]`
  折进 offset，那个和**就是** `0.5·logdet(F_bb)`，由执行消元的同一个 QR 产出。

* **M5 — `kappa_marginal = kappa(F_tt)/(1−c²)` 是错的，禁止使用。**
  `F_tt = diag(1,100)`、c=0.99 时真实 κ(Schur) = 5025，公式吻合；
  `F_tt = diag(100,1)`、**同一个 c=0.99** 时真实 κ(Schur) = **1.99**，公式仍说 5025
  ——同样的耦合差 2525 倍。平方根形式 `L_t (I − MᵀM) L_tᵀ` 精确给出 5025 和 1.99。
  **量它，不要推它。**

* **M6 — 精度矩阵 Cholesky 路的典型相关是精确的**，不是标量特例：对三个随机 6×6
  精度矩阵按 3/3 切分，与协方差路最大偏差 **1.1e−16**。

* **M7 — Laplace 估计器对 funnel 全盲，而二次特征 pilot 检测的是这个盲，不是修它。**
  Neal's funnel 众数处 Laplace 相关 **精确 0.0**；20 万 iid 抽样上线性特征 cc
  **0.0080**，二次特征 cc **0.1157** —— 比值 14.5，但**绝对值远低于任何合理的
  「高耦合」阈值**。所以 pilot 的职责不是给一个更好的 `c`，而是给一个比值，说
  **「高斯那个数没在描述这个几何」**。

* **M8 — collapse-vs-split 的比较是真实的、有区分度的，且被块大小主导。**
  本 checkout 每梯度墙钟，float64，`mu = th·(B x)`：

  | n | k | c_gc（collapse 梯度） | c_gθ（条件梯度） | c_A（一次 A·Aᵀ） | r = c_gc/c_gθ |
  |---|---|---|---|---|---|
  | 100 | 8 | 43.2 µs | 5.2 µs | 6.8 µs | 8.3 |
  | 100 | 64 | 332.8 µs | 5.7 µs | 7.7 µs | 58.6 |
  | 400 | 256 | 5 312 µs | 9.5 µs | 37.5 µs | 559 |
  | 1000 | 512 | 28 012 µs | 13.0 µs | 110.6 µs | 2163 |

  τ(c) 有界（c=0.99 时 ≈100，c=0.999 时 ≈1000）。所以 **k=8 时 c ≳ 0.95 collapse 就赢，
  k=512 时要 c ≳ 0.9995**。这正是「可能但付不起」，**而且它是一个数，不是一道门**。
  推论：collapse 把这个包推到了它自己那条中心不对称的**稠密**一侧——`exact/solve.py`
  存在的理由是永不成矩阵（docstring 点了 10⁶ 自由度），而 collapse 要物化设计矩阵和 QR。

* **M9 — 图归约不存在。** `grep 'Graph(' src/` 只有两个构造点：`graph/trace.py:323` 和
  `exact/loglinear.py:444`。`numpyro.factor` 只有一处，`bridge/numpyro_bridge.py:93`，
  硬接到 `graph.joint_prior`。（`loglinear.py:444` 静默丢 `joint_prior` 的缺陷已由
  **另一个 session 处理中**，本计划不碰。）

* **M10 — 分层。** 模块层面 `diagnose → {errors, exact, graph}`，
  `dispatch → {bridge, errors, exact, graph, marginal}`。新增模块层面的
  `dispatch → diagnose` 边保持无环且 `in_degree['dispatch'] == 0`；
  `tests/test_layering.py` 四条断言全部成立。

### 一条贯穿全局的度量陷阱（必须写进每个包）

`execute.py:754` 的 `_collapse_reason` 记录：`plated_radiometer(n=25, κ=0.4)`、N=1200 时，
SNIS 的答案偏离真值 **1.40 个后验 sd**，NUTS 偏离 **18.5**，而 NUTS 的 chain ESS（33）
**超过**了 Kish ESS（14）。**一个以 ESS 计价的目标函数会选中那个离真相远 13 倍的答案。**
缓解只能是一条规则：**永不把 Kish ESS 与 chain ESS 放进同一个 argmin，全图行不与链行
同场比较。** 规则会腐烂，这是本设计公开承认的最弱一环。

---

## 1. 编号分配（避免并行冲突）

| 包 | D 编号 | probe 编号 | spec 文件 |
|---|---|---|---|
| P1 耦合度量 | D72–D75 | probe_19 | `2026-08-29-p1-coupling.md` |
| P2 MAP | D76–D78 | probe_20 | `2026-08-29-p2-map-estimate.md` |
| P3 logdet 阶梯 | D79–D84 | probe_21, probe_22 | `2026-08-29-p3-logdet-ladder.md` |
| P4 图归约 | D85–D89 | probe_23 | `2026-08-29-p4-graph-reduction.md` |
| Wave 2 | D90+ | probe_24+ | 各自 |

**决策的家是本包自己的 spec 文件**，不是 `2026-08-26-one-implementation.md`——四个包同时
往同一个文件追加 D 行必然冲突，而 CLAUDE.md 的规则是「一个决策一个家」，不是「一个文件」。
合并之后再补指针行。

## 2. 文件边界（Wave 1 四包零重叠）

| 包 | 允许写 | **禁止碰** |
|---|---|---|
| P1 | `src/bayesmith/diagnose/coupling.py`（新）、`tests/diagnose/test_coupling.py`（新）、自己的 probe/spec | `sensitivity.py`、`graph/`、`bridge/`、`dispatch/` |
| P2 | `src/bayesmith/diagnose/map.py`（新）、`diagnose/sensitivity.py`（**唯一有权改它的包**）、自己的测试/probe/spec | `coupling.py`、`graph/`、`bridge/`、`marginal/` |
| P3 | `src/bayesmith/marginal/logdet.py`（新）、自己的测试/probe/spec | `diagnose/`、`graph/`、`bridge/`、`dispatch/` |
| P4 | `src/bayesmith/graph/*`、`bridge/numpyro_bridge.py`、自己的测试/probe/spec | `diagnose/`、`marginal/`、`dispatch/` |

---

## 3. Wave 1 — 四个并行包

以下每一节是一个**自包含 prompt**，可直接交给一个新 session。

---

# 【P1】`diagnose/coupling.py` — 两块之间的耦合与三个条件数

## 你的任务

在 `/Users/zzhang/projects/bayesmith` 新建 `src/bayesmith/diagnose/coupling.py`，
提供一个函数，度量一个图的两组 latent 之间的**耦合**，以及三个采样策略各自要付的
**条件数**。这是一个纯诊断模块：**不改任何采样路径，不碰 `dispatch/`**。

## 为什么（理论）

两块 Gibbs 采样器在高斯目标上的转移算子，其谱**恰好是两块之间典型相关的平方**
`{c_i²}`（Amit 1991；Liu, Wong & Kong 1994；Roberts & Sahu 1997）。因此谱隙是
`1 − c_max²`，最慢线性泛函的积分自相关时间是

```
τ(c) = (1 + c²) / (1 − c²)
```

而 HMC 在同一个目标上只付 `√κ`（步长被最小方向卡住 `ε ≲ √λ_min`，最慢模式周期
`∝ √λ_max`）。**Gibbs 线性收费，HMC 平方根收费**——这是整个重设计的根据。

典型相关 `c_1 = max_{a,b} corr(aᵀu, bᵀv)`：**在 A 块里能凑出的所有线性组合和 B 块里
所有线性组合之间，最大的那个相关**。它不是相关矩阵里的某个元素——脊的方向可能不对着
任何单个参数（「所有增益一起浮 + 所有像素一起沉」是典型情形）。

更精确地说，控制两块 Gibbs 的量是 HGR **最大相关** `ρ* = sup{corr(f(u), g(v))}`（对所有
平方可积的 f、g）。Lancaster 定理：**联合高斯时 `ρ* = c_1`**，sup 在线性函数上取到。
非高斯时 `ρ* ≥ c_1`，可以严格更大——这就是 funnel（M7）漏网的理论原因，也说明
**线性 CCA 只会漏报，绝不会误报**（噪声地板以上）。

## 具体实现

### 核心算法：走**精度矩阵**，全程不成协方差

```
F = [[F_xx, F_xθ], [F_θx, F_θθ]]        # Fisher + 先验，按 spans 分块
L_x = chol(F_xx);  L_θ = chol(F_θθ)
M   = L_x⁻¹ F_xθ L_θ⁻ᵀ                   # 两次三角求解
c   = svdvals(M)                          # 典型相关
```

**为什么走精度不走协方差**：代价是 `√κ` 而不是 `κ`（与 D64 为 `smooth` 定下的
「平方根优于求逆」同一论证），并且 `fisher.condition_ceiling` 那道拒绝**根本不用付**。
M6 已实测这条路与协方差路最大偏差 1.1e−16。

### 三个条件数

```
κ_cond  = κ(F_θθ)                        # Gibbs sweep 里内层 NUTS 付的
κ_marg  = κ(L_θ (I − MᵀM) L_θᵀ)          # collapsed 目标付的 —— 一次 eigvalsh
κ_joint = κ(F)                           # 联合 NUTS 付的 —— 一次 eigvalsh
```

**`κ_marg` 必须这样量，绝不能写成 `κ(F_θθ)/(1−c²)`。** M5 实测那个公式在同一个 c=0.99
下可以差 2525 倍（`diag(1,100)` → 真值 5025，`diag(100,1)` → 真值 1.99，公式两次都说
5025）。**κ_marg 不是 c 的函数。**

### 噪声地板与三值裁决

```python
floor_c = sqrt(kappa(F_xx) * kappa(F_theta_theta)) * eps
```

**低于或等于地板必须裁决为 `refused`，不是 `low`。** 「病态白化里掉出来的一个舒服的
小数」正是这个模块要禁止的失败模式。裁决类型三值，不允许返回裸 float：

```
Measured(value, floor, n_eff)  |  Refused(reason)  |  NotApplicable(reason)
```

`blind_to` 是一个**字段**，不是注释：Laplace 路对尺度型依赖（funnel）全盲，报告必须
自己说出来，形如 `blind_to=("gaussian-only",)`。

### 接口

```python
def block_coupling(graph, first, second, *, at) -> CouplingReport
```

`at` 是**必需参数**，不是可选：本模块不求 MAP（那是 P2 的事），它在你给的点上线性化。
这样本模块可以完全独立开发和测试，用手写的点对 oracle。`CouplingReport` 用 frozen
dataclass 持 numpy（不要 `eqx.Module` 持 numpy——嵌进 static 字段会在**第二次** trace
上炸 treedef 比较）。

### jit 陷阱（M3）

若日后有 traced 消费者：**必须**用 `unchecked_operator(graph, block, at=theta,
probe_gaussian=False)`。`local_block(..., priors=True)` 在 `diagnose/local.py:293` 以
`probe_gaussian` 默认 True 调 `_env_before`，jit 下从 `exact/gaussian.py:249` 抛
`TracerBoolConversionError`。本模块自身在 trace 外运行，但把这条写进 docstring。

### 复用

`diagnose/local.py:230 local_block(priors=True)`（G15，已建已测但**没有生产消费者**——
这是它的第一个）、`exact/fisher.py` 的 `fisher_information` / `FlatMatrix.spans`、
`exact/block.py` 的 `unchecked_operator`。

## 测试（对本仓的标准）

* **oracle 必须与实现无共享**：稠密 NumPy 高斯 + `np.linalg.inv`/`svd`，
  **绝不拿我们的一个例程去验另一个**。
* **钉住 M1 的加性可分 fixture**，值 **−0.9942708**。这是廉价证书会搞错的那个案例，
  它必须以回归测试的形式活着。
* **钉住 M5 的一对**：`F_tt = diag(1,100)` 与 `diag(100,1)`，同一个 c=0.99，断言
  `κ_marg` 分别回来 **5025** 和 **1.99**。只测第一格的测试分辨不出实测路和错公式。
* **地板纪律要双向断言**：等于或低于 `floor_c` 的值必须 `verdict="refused"`，
  **且测试必须在它报 `"low"` 时失败**。
* **每个 fixture 都用非单位先验宽度。** `−Σ log std` 在 std=1 时精确为零——上游的
  log-det 缺陷就是这样藏进去的。
* 与协方差路对拍（M6，容差 1e-15）。

## 文档（必交，不是可选）

新建 `docs/superpowers/specs/2026-08-29-p1-coupling.md`，包含：

1. **理论**：为什么是典型相关而不是逐元素相关；`τ(c)` 与 `√κ` 的线性/平方根不对称
   的推导；HGR 最大相关与 Lancaster 定理，以及由此得出的**「只漏报不误报」**这一
   单边性质。
2. **决策 D72–D75**（编号已为你保留），每条给出**它解决的问题**和**支撑它的实测**：
   建议 D72 = 走精度不走协方差；D73 = `κ_marg` 实测而非推导（引 M5 两格）；
   D74 = 地板以下裁 `refused` 不裁 `low`；D75 = `at` 为必需参数（模块独立性）。
3. **M1 那个洞**：为什么本模块**没有**廉价可分性证书，加性反例的完整数字。
4. **盲区**：Laplace 是局部量，多峰下错；对 funnel 全盲（M7 的数）；这两条都要
   明写，不要藏。

`docs/probes/probe_19_coupling_oracles.py`：把上述 oracle 对拍写成可重跑的 probe，
输出追加到 `docs/probes/OUTPUT.txt` 的惯例格式。

## 边界

**不做**：MAP 求解（P2）、成本模型（Wave 2）、任何 `dispatch/` 改动、任何采样路径改动。
**不碰**：`sensitivity.py`、`graph/`、`bridge/`、`dispatch/`。

## 仓库惯例

```bash
.venv/bin/python -m pytest -n 4 > run.log 2>&1; echo "PYTEST_EXIT=$?" > run.exit
cat run.exit
```

**绝不加 `-q`**——`pyproject.toml` 已有 `addopts = "-q"`，第二个会变成 `-qq` 而
**汇总行整个消失**。计数取自 `--junit-xml`，裁决取自写进独立文件的退出码。
只有退出码 **1** 才是测试失败（2 中断，3 内部错误，4 用法错，5 未收集到，143 被杀）。
基线：**1269 passed, 0 skipped, 0 failed**，`-n 4` 下约 207 s。

---

# 【P2】`map_estimate` — 图原生的 MAP，带自己的拒绝

## 你的任务

在 `/Users/zzhang/projects/bayesmith` 提供一个公开的 `map_estimate(graph, ...)`，
返回后验众数，供 Laplace 类诊断使用。

## 为什么

`diagnose/coupling.py`（P1，并行开发中）需要一个线性化点。今天没有公开的 MAP 入口。
`diagnose/sensitivity.py:230` 有 `_newton` 和 `_descent_direction`，但：

* `_newton` 接受的是一个**裸 callable**，而「图 → 目标函数」的构造与
  `prior_sensitivity` 的选择逻辑纠缠在 `sensitivity.py:721-736`。
  **这不是一行提升，是一个新的图原生目标构造器。**
* **它必须写自己的拒绝，绝不能继承 `prior_sensitivity` 的三条**——那三条会拒掉
  funnel、拒掉每一个层级模型、并且恰好拒掉那个你想边缘化的线性 nuisance 块。
  这一点是本包最容易出错的地方，请在实现前先把那三条读出来并在 spec 里列明。

## 具体要求

* 新建 `src/bayesmith/diagnose/map.py`。**你是唯一有权改 `sensitivity.py` 的包**——
  若要抽取共享的 `_newton`/`_descent_direction`，在这里做，并保证
  `prior_sensitivity` 的行为逐位不变（现有测试是你的护栏）。
* float64 是前置条件（D9），**按名拒绝**，不要静默降精度。
* 拒绝是三值的，与 P1 同构：`Refused(reason)` 必须带**出路**，不只是原因。
* 返回值要能直接喂给 `local_block(graph, names, <此处>, priors=True)`。

## 测试

* 对解析已知众数的图对拍（线性高斯：众数 = Wiener 解；确认与 `wiener_solve` 一致）。
* **funnel 必须被接受**，不被拒绝——这是 `prior_sensitivity` 的拒绝会误伤的头号案例，
  写一个测试明确断言它通过。
* 一个层级模型必须被接受。
* 一个真正病态的图必须被**拒绝且给出出路**，断言拒绝文本里含出路。
* 收敛失败必须是 `Refused`，**不是**一个「看起来合理」的返回值——不能分辨
  「没收敛」和「收敛到这里」的返回值是本仓反复吃亏的那一类。

## 文档

`docs/superpowers/specs/2026-08-29-p2-map-estimate.md`：**D76–D78**。
必须包含一节，逐条列出 `prior_sensitivity` 的三条拒绝、为什么它们对 MAP 是错的、
以及新写的拒绝分别替代了什么。`docs/probes/probe_20_map_refusals.py` 记录对拍。

## 边界 / 仓库惯例

**不碰**：`coupling.py`、`graph/`、`bridge/`、`marginal/`。
测试命令与 `-q` 禁令同 P1（见本文件 §P1 末尾，一字不差地适用）。

---

# 【P3】`marginal/logdet.py` — 从特殊到一般的 logdet 阶梯

## 你的任务

在 `/Users/zzhang/projects/bayesmith` 新建 `src/bayesmith/marginal/logdet.py`：一个
**有序**的 log-determinant 方法阶梯，每一级带**可检验的前提**，外加一个只做判定不做
计算的前提检查器。

## 为什么（这是本计划风险最高、潜在收益最大的包）

边缘化需要 `log det(Σ)`。一个**无矩阵**算子（`exact/solve.py` 的整个设计前提是「只作用，
从不组装」）**无法**从作用中得到行列式。M8 实测显示：collapse 的每梯度成本随线性块维数
k 急剧增长（k=8 时是条件路的 8.3 倍，k=512 时是 2163 倍），因为它走了物化设计矩阵 + QR
的稠密路。

**因此本包的战场不是「让不可能变可能」，而是「让 k 大时 c_gc 不要炸」。**
截断 trace-log 只需要 m 次**算子作用**，不物化 A，不做 QR——它可能是唯一能把 M8 那张表
的 r 从 2163 拉回来的东西。**本包最重要的一个交付物，是在 k=512 那一行上用截断
trace-log 替代 QR 后 c_gc 的实测值。**

## 关键性质：确定性 vs 随机性

**带噪声的 logdet 会破坏 HMC**（leapfrog 不再可逆）。**但确定性的近似 logdet 完全没问题**
——你只是在采一个略微不同、但定义明确的目标。这条分界必须硬编进接口：

* 截断展开是 θ 的确定性函数 → **HMC 安全**。
* Hutchinson 随机迹估计**只要冻结探针向量**（common random numbers）也是确定性的
  → 安全。每次调用重抽探针 → **不安全，必须拒绝**。

## 阶梯（从特殊到一般）

| # | 方法 | 前提（**必须验证，不得假设**） | 精度 | 现状 |
|---|---|---|---|---|
| 0 | Λ 本身（对角/平凡） | Σ = Λ | 精确 | — |
| 1 | 行列式引理 / 平方根形式 | rank(P) = k ≪ n | **精确** | `marginal/sqrtinfo.py` ✅ |
| 2 | 状态空间递推 | 图是链 | **精确** | `marginal/chain.py` ✅ |
| 3 | 结构化（循环/Toeplitz/Kronecker） | 结构成立 | **精确** | 缺 |
| 4 | 稠密 Cholesky | n ≤ 阈值 且 κ < ceiling | **精确** | `exact/fisher.py` ✅ |
| 5 | 有限精确微扰（Newton 恒等式） | n 小，或 rank(P) 小 | **精确** | **本包新建** |
| 6 | 截断 trace-log | **ρ(X) < 1** | 确定性近似，误差有解析上界 | **本包新建** |
| 7 | 冻结探针 SLQ | 无（偏差不解析） | 确定性近似 | 可选 |
| 8 | 每次重抽的随机估计 | — | **HMC 不安全** | **拒绝** |

**调度规则**：从上往下，第一个前提成立的赢。

## 第 5、6 级的数学（用户提供的推导，`~/Downloads/logdet.tex`）

设 `Σ = Λ + P`，`X ≡ P Λ⁻¹`，`Σ = (I + X) Λ`。

**第 5 级——有限精确展开**：

```
det(Σ) = det(Λ) · Σ_{k=0}^{n} e_k(X)
```

`e_k` 是 X 特征值的第 k 个初等对称多项式，由 Newton 恒等式从迹 `p_j = Tr(X^j)` 得到：

```
k·e_k(X) = Σ_{j=1}^{k} (−1)^{j−1} e_{k−j}(X) p_j
e_1 = Tr(X)
e_2 = ½{[Tr X]² − Tr X²}
e_3 = ⅙{[Tr X]³ − 3 Tr X Tr X² + 2 Tr X³}
```

**这既是恒等式也是在 n 阶终止的微扰展开。**

**一个必须写进文档的结构观察**：当 P 低秩（秩 k）时 X 也是秩 k，于是 `j > k` 的所有
`e_j(X)` **精确为零**，级数自动在 k 处终止——而这**就是**行列式引理。
**所以阶梯的第 1 级是第 5 级的低秩特化，不是另一个方法。** 请在实现中体现这一点
（同一份代码路径，靠 `e_j` 的稀疏性分流），并在测试中断言两路逐位一致。

**第 6 级——截断 trace-log**：

```
ln det(Σ) = ln det(Λ) + Σ_{r=1}^{∞} ((−1)^{r+1}/r) Tr(X^r)          [ρ(X) < 1]
```

截断到 m 阶，余项有界：

```
|余项| ≤ ρ^(m+1) / [(m+1)(1−ρ)]          ρ ≡ ρ(X)
```

**因此 m 由测量决定，不由拍脑袋决定**：用幂迭代（无矩阵）量 ρ(X)，给定目标精度反解 m。
本仓已有 `POWER_ITERATIONS` 的先例（见 `exact/gibbs.py` 的 docstring）。

**梯度免费**：截断级数是矩阵乘积与迹的有限复合，**JAX 直接微分它对截断后的对象是精确的**。
不要另推 `∂ log det` 的公式。

## 两个设计要点

**(i) Λ 的选择就是预条件子设计。** Λ 不必是噪声协方差 N。收敛条件是 `ρ(PΛ⁻¹) < 1`，
所以你**有权选** Λ：Σ 的对角、块对角近似、循环近似……Λ 越好 ρ(X) 越小，需要的 m 越少。
这与 CG 的预条件是同一件事，检查本仓是否已有可复用的机器。
**这对前景主导的场景是决定性的**：`P = ASAᵀ` 相对 `N` 往往**不小**，直接展开会发散，
换个 Λ 才可能收敛。

**(ii) ρ(X) 依赖 θ，而你无法在 trace 里检查。** 收敛前提必须在采样器走过的**整个区域**
成立。本仓已就同类问题定过调（`exact/gibbs.py`：「in-sweep convergence guard 是**关掉**的，
`eqx.error_if` 帮不了一个它无法中断的 sweep」）。因此唯一自洽的做法是：

```
warmup   在探测点集上测 ρ(X) 最大值，加安全裕度，保守定 m   ← 在 trace 外
运行     不检查（检查不了）
事后     在保留样本上重算 ρ(X)，看有没有越界        ← 便宜，只有几千点
```

并且**像 `tol` 一样，把「这个数是唯一挡在你和静默错误之间的东西」写进 docstring**。

## 测试

* 每一级对稠密 `np.linalg.slogdet` oracle 对拍，**oracle 与实现无共享**。
* **第 1 级与第 5 级在低秩输入上必须逐位一致**（上面那个结构观察的可执行形式）。
* **边界验证**（`~/.claude/rules/common/boundary-validation.md`）：第 5/6 级之间、
  第 6 级的 m 选择、`ρ(X) < 1` 的判据——每一处**绕过调度器，直接在阈值两侧求值两种方法**，
  断言 `rel_err`。**务必包含极端参数**：ρ(X) ∈ {0.01, 0.5, 0.9, 0.99, 1.01（必须拒绝）}，
  n 与 rank(P) 各跨四个数量级。
* 截断误差的**解析上界必须被实测检验**：断言实际误差 ≤ 上界，在每一格。
* **第 8 级（重抽探针）必须被拒绝**，且测试断言拒绝发生——一个不会变红的守卫不是守卫。
* 所有 fixture 用**非单位**先验宽度。

## 交付物中最重要的一项

`docs/probes/probe_22_logdet_cost_at_scale.py`：**复现 M8 那张表，并加一列**——
在同样的 (n, k) 上用第 6 级替代 QR 之后的 `c_gc`。这个数决定 collapse 在大块上到底
有没有救，也决定 Wave 2 的 P6 该怎么写。**即使本包其余部分都推迟，这个测量也要做。**

## 文档

`docs/superpowers/specs/2026-08-29-p3-logdet-ladder.md`：**D79–D84**。
必须含：完整的阶梯表与每级前提；上面那段数学（含「第 1 级是第 5 级的低秩特化」的证明）；
确定性/随机性那条分界及其对 HMC 的后果；Λ-as-preconditioner 的讨论；ρ(X) 依赖 θ 这个
守卫难题及其三段式处理；以及 M8 复现表 + 新增列的实测结果与解读。
`docs/probes/probe_21_logdet_ladder_oracles.py` 记录对拍。

## 边界 / 仓库惯例

**不碰**：`diagnose/`、`graph/`、`bridge/`、`dispatch/`。
测试命令与 `-q` 禁令同 P1。

---

# 【P4】图归约与第二个 factor 槽 —— **这是难的那一步**

## 你的任务

在 `/Users/zzhang/projects/bayesmith` 建立把一个图**归约**（删掉一组 latent 及其孤立
后代与被吸收的观测节点）并给归约后的图**挂上一个证据项**的机器。

## 为什么

边缘化要求 NUTS 在一个**已删掉线性块、但携带该块边缘似然**的图上跑。M4 已实测边缘似然
的算术端到端成立（对稠密 oracle 1.8e−14，梯度 2.6e−08，jit 与 hessian 干净）——
**缺的不是算术，是图这一侧的管道。** M9：`grep 'Graph(' src/` 只有两个构造点；
`numpyro.factor` 只有一处（`bridge/numpyro_bridge.py:93`），硬接到 `graph.joint_prior`。

## 必须做到的事

1. **`Graph` 新增字段 `evidence_terms: tuple[Any, ...] = ()`**，像 `joint_prior` 一样做
   结构校验，**并且拒绝与 `joint_prior` 重叠**——`JeffreysPrior.information` 要求它覆盖的
   latent 是 `ImproperUniform`，这与一个似然因子的要求正好相反。
2. **`log_joint` 在读 `joint_prior` 的同一处求和**，这样一个图的两次扫描不可能不一致。
3. **`to_numpyro` 把它作为第二个 factor 发出，且必须在任何 plate 之外**——
   **plate 内的 factor 会被静默乘上 plate 大小。**
4. **硬拒绝**：某个项的 `over` 里的任何 latent 落进非 NUTS 块时必须拒绝。
   `grep 'joint_prior' src/bayesmith/exact/ src/bayesmith/dispatch/` 返回零——
   `exact/` 和 `dispatch/` 里**没有任何东西读图级密度项**，于是一个省略了该项的条件分布
   抽出来的 `gcr_sample` 是**静默错误，且所有诊断都健康**。
5. **把结构规则变成类型**：**归约与挂载必须是同一个返回对象的两个字段，不提供任何
   单独产出其一的 API。** 否则数据被算两遍，而且没有症状。
6. `Graph.__check_init__` 要求每个父节点先于子节点声明——归约必须保持这个不变量。

## 测试：那个真正重要的测试

**双计数测试必须是绝对密度，不是形状：**

```
log_joint(归约图 ⊕ 证据项)   vs   log_joint(原图) 对块做稠密求积分
```

在 K 个点上，**包含每个参数的两个端点**，用**非单位先验宽度**。
**一次双计数在均值、宽度和梯度上全都不可见，只在密度值上可见。**

再加一个**变异测试**：把证据项挂到**未归约**的图上，断言测试变红。
**一个在缺陷被引入时不变红的守卫不是守卫。**

其余：每条拒绝一对 accept/refuse，且断言拒绝文本给出**出路**。
**删掉任何不可能失败的检查**——D23 的规则「构造成功不等于可达」在这里适用。

## 文档

`docs/superpowers/specs/2026-08-29-p4-graph-reduction.md`：**D85–D89**。
必须含：为什么 `evidence_terms` 与 `joint_prior` 必须互斥（`JeffreysPrior` 的相反要求）；
为什么 factor 必须在 plate 外（含 plate 内会被乘 plate 大小的实测演示）；
为什么归约与挂载不能有独立 API（双计数不可见性的论证）；以及第 4 条那个静默错误的完整
论证——`exact/`、`dispatch/` 都不读图级密度项，这是本包存在的核心风险。
`docs/probes/probe_23_double_count_visibility.py`：演示双计数在均值/宽度/梯度上**不可见**、
在密度值上**可见**——这是把上述论证变成可重跑证据。

## 边界 / 仓库惯例

**不碰**：`diagnose/`、`marginal/`、`dispatch/`。
`exact/loglinear.py:444` 丢 `joint_prior` 的缺陷**由另一个 session 处理，本包不要碰**
——但它用的是同一套重建惯用法，合并后注意对齐。
测试命令与 `-q` 禁令同 P1。

---

## 4. Wave 2 — 有依赖，等 Wave 1 落地后再开

| 包 | 依赖 | 内容 |
|---|---|---|
| **P5** `dispatch/costs.py` + 只读 ladder | P1, P2 | 三个成本表达式 + 计时探针；`strategy="cost"` **只打印记分板，不改任何路由**。独立有用：它告诉用户模型为什么混合得差、collapse 能帮多少——即使 collapse 还不可建。`C_split = τ(c)[a√κ_cond·c_gθ + m·k_cg·c_A]`，`C_collapse = a√κ_marg·c_gc + k_cg·c_A`，`C_joint = a√κ_joint·c_g_all`。**不得断言任何消去**：CG 项在分子出现 τ·m 次、分母一次，不约掉；**c 上没有闭式交叉点**（M5：κ_marg 不是 c 的函数），交叉点只能数值求解。 |
| **P6** collapsed target + `collapse` 臂 | P4，且 **P3 的 probe_22 结论** | `execute.py:503-518` 的 ladder 加一臂。M4 保证算术已成；这是接线。两处都用 `unchecked_operator(..., probe_gaussian=False)`（M3），traced pivots 上加 `eqx.error_if`，使一个在链会到达的 theta 处无约束的块**抛出**而不是返回一个有限的、看起来合理的数。 |
| **P7** A4 pilot + 对账账本 | P1, P5 | pilot 的裁决是**非对称**的：二次 cc 同时越过 `sqrt(p_aug/N_eff)` **且**超线性 cc 一个声明倍数 → **否决切换**并点名 funnel；没结论 → A3 决定照旧，打印 `blind_to=("gaussian-only",)`。**一个卡住的 pilot 代价为零，且不会静默变成抛硬币。** 账本记录预测区间、实测 秒/ESS、以及**哪个输入主导**，使一次落空能定位到 c、a 还是区间过宽。 |

**两个被明确排除的策略**：

* **非中心化**：D23 实测本声明层里一个 `Latent` **不能**参数化另一个 `Latent` 的先验
  （`dist.Normal(<Latent>, 0.5)` 构造得出、`.sample()` 抛 `TypeError`）。重参数化暂时
  是**用户的义务**，解锁条件写在 D23。
* **campaign fold**：`campaign.py:91-93` 以 `isolate(graph, names, {})` 构造设计，空 `at`
  与零硬编码，而 `at` 只到达 `precision_at`(:97) 与 nuisance 先验(:98)——一个双线性图
  会拿到逐位相同的 `[[0,0],[0,0]]`。**修这个是前置条件，不是本设计的一部分。**

## 5. 全局提醒（每个包都适用）

* **ESS 陷阱**（见 §0 末）：绝不把 Kish ESS 与 chain ESS 放进同一个 argmin。
* **三值裁决**：任何驱动决策的量都不许是裸 float；`Refused` 必须回落到默认并**说出来**，
  绝不能当成 0。
* **突变测试**：先提交批次再变异（`git checkout -- src/` 恢复到 HEAD，未提交的工作会被
  静默连带回滚）；变异与恢复之间 `rm -rf __pycache__`；**修好之后先提交再重跑**，
  否则第二次运行开头的恢复会把修复本身撤掉，而输出里没有任何东西会告诉你。
* `ruff check src/ tests/` 保持干净。`ruff format` 有 31 文件 / 517 行的既有漂移，
  **不要顺手扫**；真要扫就传那 31 个文件名，不要传 `src/ tests/`。
