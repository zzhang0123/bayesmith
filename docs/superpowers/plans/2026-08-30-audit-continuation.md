# 数值门审计修复 + Wave 2 执行（P5/P6/P7）—— 合并交接

> **文档状态：`record`** · 已落地批次/审计/测量的历史记录，写作当天为真，非当前权威。索引见 docs/README.md。

**日期**：2026-08-30
**性质**：两份交接合并为一份，交给一个 session 执行。前半是**已复现的审计问题**（F1–F4，
实测坐实），中间是**套件分层**，后半是 **Wave 2 的 P5/P6/P7**（含硬性设计约束与实测表）。
每个阶段结束必须提交；上下文变紧时**在阶段边界停下并报告停在哪**，绝不打开一个完不成的阶段。

**必读背景（按顺序）**：
1. `~/.claude/rules/common/boundary-validation.md`（方法论）
2. `docs/superpowers/plans/2026-08-29-coupling-collapse-ladder.md`（§0 实测 M1–M10、§4 Wave 2）
3. `tests/numerical_gates/registry.py` 的 `GateEntry` 结构（要往里加条目）
4. `docs/superpowers/specs/2026-08-29-p{1,2,3,4}-*.md`（P1–P4 已实现）
5. `docs/superpowers/plans/2026-08-30-r1-task-artifact-provenance.md`（R1，本计划**不**包含，仅作后续）
6. `CLAUDE.md`

---

## 贯穿全部阶段的规矩（比任何单条任务都重要）

> **你这次引入的每一个数值阈值，都必须登记进 `tests/numerical_gates/registry.py`，
> 并带上两侧（tighten + loosen）的边界格。** 填对 `provenance`：
> derived / exact_or_domain / api_contract / borrowed / magic。只能填 `magic` 就是在
> 告诉未来的人这个数没有依据——先想清楚再写。

**现状**：4690 passed / 0 failed / exit 0，ruff 干净，README 与收集数一致。架子在
`tests/numerical_gates/`：**100 个门**（84 个两向带 tighten+loosen witness，16 个
static-only，0 遗留歧义）。

---

# 硬阶段边界与停止规则（最高优先级）

五个阶段，**按顺序，每个阶段结束必须提交**，每个阶段单独有用：

| 阶段 | 内容 | 改动路由？ |
|---|---|---|
| **0** | 审计修复（F1 误拒 + MAP 尺度 + F3/F4 裁决） | 是（F1） |
| **1** | 套件分层（fast/full 两层） | 否 |
| **2** | P5 `dispatch/costs.py` 只读记分板 | 否（纯加法） |
| **3** | P6 collapsed target + `collapse` 臂 | **是** |
| **4** | P7 A4 pilot + 对账账本 | 否（观测） |

**停止规则**：上下文变紧就在阶段边界停下并报告；半个阶段 3 不如完整的阶段 0+1。每个阶段
跑全套 + ruff，提交后再开下一个。README 计数留到最后一个阶段写。

---

# 阶段 0：审计修复（先做，一次提交）

## 0.1 F1 —— determinant-lemma rung 被 κ(Σ) < 1/eps 误拒（正确性，已复现）

**一句话**：低秩 rung 被一道与 spec、docstring、以及"自己加了两次"互相矛盾的门卡住，
对"低秩、SPD、非对角、但 preconditioner Λ 病态"的输入，本应精确的 logdet 被整梯拒绝。

**复现**（本 checkout 实测，新 session 先重跑确认仍红）：

```python
import numpy as np
from bayesmith.marginal import _logdet_eager as eager
from bayesmith.marginal import _logdet_ladder as ladder

lam = np.diag([1e8, 1e-8, 1.0, 1.0])     # κ(Λ)=1e16，对角但病态
L   = np.array([[0.0],[0.0],[0.5],[0.5]])
P   = L @ L.T                             # 秩 1，带非对角 → σ 非对角（避开 level3 对角 fallback）
factors = eager.LowRankFactors(L)

print(float(np.linalg.slogdet(lam + P)[1]))   # 0.4054651081081644 == log(1.5)，精确
problem = eager.LogDetProblem(lam, P, low_rank_factors=factors)
verdicts = ladder.check_logdet_premises(problem)
print([v.level for v in verdicts if v.satisfied])   # [] —— 一整个阶梯都不满足
print(verdicts[1].details["rank_evidence_valid"])   # True —— eta 证书本身通过
print(verdicts[1].details["condition"])             # 1e16 —— 唯一卡点 κ(Σ)<1/eps
ladder.dispatch_logdet(problem)                     # ResamplingRefused
```

**根因链（三处互相矛盾）**：
- `_logdet_ladder.py:279-285`：`determinant_lemma_payload = ... and sigma_exactly_symmetric
  and condition_resolved`。
- `_logdet_eager.py:1193`：`_newton_logdet` 对 `factors is not None` 再执行
  `_require_resolved_dense_condition(sigma, "determinant lemma")` —— 同一道门两次。
- `_logdet_eager.py:1066-1067`（`_condition_certificate` docstring）："determinant-lemma
  payloads **do not use it**"。
- spec p3"条件数策略"段："第 0/1/4/5 级不以它拒绝，condition 仅作诊断"，D 表 row 0/1/5 不列 condition。

**为什么是误拒**：determinant lemma 走 `logdet(Λ) + logdet(I + RᵀΛ⁻¹L)`，只分解 k×k 小矩阵，
精度由 `κ(Λ)`（对角 Λ 时 ceiling=inf 不设门）与 k×k 因子的 `reduced_condition`（证书里已有
独立门）决定，与 n×n 的 κ(Σ) 无关。

**修法**：
1. 从 determinant-lemma 路径去掉 κ(Σ) 门：`determinant_lemma_payload` 里的 `condition_resolved`
   与 `_newton_logdet` 里的 `_require_resolved_dense_condition(sigma, "determinant lemma")`
   都去掉（后者只应留在"通用稠密 Cholesky"分支，不应在 `factors is not None` 分支）。
2. `base` 的 `dense_arithmetic_resolved` 只对 2D 稠密 Cholesky 保留（1D 已
   `sigma.ndim==1 → True`）；`dense`（level 4）的 `condition_resolved` 保留——那是真正的
   稠密消元分辨率门（B2 修的）。
3. 同步 `_condition_certificate` docstring 与 spec p3 的"条件数策略"段 + D 表 row 0/1/5，
   写清"1/eps 门只属于 2D 稠密消元/递推/变换载荷，determinant lemma 与对角结构化载荷不用"。
4. 两条边界格：`κ(Σ)≥1/eps 但 lemma 精确` 必 ADMITTED（钉"去掉门后不误拒"）；
   `κ(Σ)<1/eps 但因子证据坏` 必 REFUSED（钉"eta 证书仍在把关，不因删门放水"）。

## 0.2 F2 + 最后一个魔数 —— MAP 收敛/曲率守卫的尺度不变性

这是同一家族的两个问题，一起做：

**（A）还剩一个曲率侧 clamp**（`map.py:212` `max(abs(largest), 1.0)`），注册表标成整批
100 个门里**唯一**的 `magic`（gate_id `MAP:map_estimate:curvature-scale-clamp`）。它是两个
已被 B1 删除的梯度侧 clamp 的曲率侧兄弟。已实测：同一后验两次**精确单位换算**（宽度
30/28/32，两边 max|mode|>1 所以另一个 clamp 不活跃），‖H‖=1.276e-03 接受 4.02e-05 的相对
误差，‖H‖=1.000 接受 5.13e-08——比值 **783.7 = 1/‖H‖**，四位有效数字，四个扰动方向一致。
**同一物理点在一个单位制下返回 MapEstimate、在另一个下 Refused。**

二选一并说明理由：(a) 删掉它，与梯度侧两个 clamp 一致；(b) 保留，把 provenance 从
`magic` 改成有依据的那档并在 spec 写下依据。无论哪条，加一个把"同一物理点在不同单位制下
裁决相同"钉住的边界格。

**（B）P2 spec 描述了已删除的公式**：`docs/superpowers/specs/2026-08-29-p2-map-estimate.md`
§三.4 与 §4.7 描述的梯度地板是 `sqrt(eps)·n·max(|λ_max(H)|,1)·max(‖x‖∞,1)`（两个 clamp），
但 B1（`2dd988e`）已删，现行是 `sqrt(eps)·n·‖H‖₂`。**注意**：删掉 clamp 后，梯度地板对
**目标尺度**不变（‖H‖ 随目标缩放），但对**坐标尺度**不再保证——`x→c·x` 时 ‖H‖₂∝c² 而
梯度∝c，裁决比随 `1/c` 漂移。spec 里"缺任一项都会让纯单位换算改变裁决"与 §4.7 的坐标不变
主张**不再被公式背书**。修法：更新 spec 把"目标尺度不变、坐标尺度不保证（B1 用大坐标拒绝
钉子替代）"写成明确边界；若 owner 要恢复坐标不变，重引入 `‖H‖₂·max(‖x‖,1)` 项（**不能
回退到 B1 修掉的逃逸点放水版**）——这是策略裁决，不要拍板。

## 0.3 F3 / F4 —— 裁决并记录（各写一行结论即可，不强制改代码）

- **F3** `_power_traces_match`（`_logdet_eager.py:1662-1675`）用 `np.array_equal` 逐位比较，
  Decimal 精确迹差 1 ULP 就被 `truncated_trace_logdet` 拒绝（调用方算得完全正确也会被拒）。
  裁决：放宽到 ULP 邻接 + 两侧测试，或记录为何容忍。
- **F4** eta 证书对"巨大+微小"SPD 模式过保守（`Λ=I, P=diag(1e16,0,0), L=[[1e8],[0],[0]]`
  → `eta=1` 拒绝，但真实 logdet 误差仅 ~1e-16）。这是**保守方向的假拒绝（安全侧）**，
  可记录、不急着改，除非有真实用例被误伤。

**阶段 0 做完：跑全套 + ruff，提交。**

---

# 阶段 1：套件分层（fast / full 两层）

**问题**：套件从 ~4 min/1843 tests 涨到 **19:22 / 4690 tests**。4 分钟人们提交前会跑，
19 分钟不会；失效方式可预测——人们开始跑子集，而"跑子集"正是这个仓库被咬过一次的
junit 截断陷阱入口（`CLAUDE.md` 印的命令没有 `--junit-xml`，pytest 在 sessionfinish 以
mode="w" 截断 junit 路径且不发警告，一次窄运行会静默覆盖上次计数而 log/exit 保持旧的）。

**做法**：给重的边界/变异格打 pytest marker，分成快层和全层。**必须用元测试强制**：每个
登记在册的门**至少有一格在快层**，否则"快"会悄悄变成"没有"。把两层跑法写进 `CLAUDE.md`，
把"Running the tests"里过时的配方换成每次一个目录的形式（见文末"测试产物"）。

**回报**：快层的测试数与耗时；全层耗时。目标是快层能进提交前习惯，全层进夜间。

**阶段 1 做完：跑全套，提交。**

---

# 阶段 2：P5 —— `dispatch/costs.py` 与只读记分板

Wave 2 里唯一**不改任何路由**的一步，也最先该有：让用户第一次看见模型里 ρ 和三个条件数
是多少、collapse 能帮多少。**纯加法，不破坏任何已发布路径。**

## 2.1 三个成本表达式

```
k_cg(κ, tol) = ceil(0.5 · sqrt(κ) · log(2/tol))
τ(c)         = (1 + c²) / (1 − c²)

C_split    = τ(c) · [ a·√κ_cond · c_gθ  +  m · k_cg(κ_x, tol_x) · c_A ]   m = 1 (gcr), 3 (gcr+mh)
C_collapse =          a·√κ_marg · c_gc  +      k_cg(κ_x, tol_x) · c_A
C_joint    =          a·√κ_joint · c_g_all
```

**三条硬性约束（每条都被实测逼出来）**：
1. **不得断言任何消去。** CG 项在分子出现 τ·m 次、分母一次，**不约掉**。
2. **c 上没有闭式交叉点。** κ_marg **不是 c 的函数**——实测：`F_θθ = diag(1,100)` 与
   `diag(100,1)` 在**同一个 c=0.99** 下真实 κ(Schur) 分别是 **5025** 和 **1.99**，而
   `κ(F_θθ)/(1−c²)` 两次都说 5025，差 2525 倍。交叉点只能数值求解。κ_marg 由
   `diagnose/coupling.py` **实测**给出（`κ(L_θ (I − MᵀM) L_θᵀ)`），不要推导。
3. **c_gθ、c_gc、c_A 直接计时测，不要建模。** grad/(A+Aᵀ) 的比值在两个既有 fixture 上随
   n **朝相反方向**变化。每梯度墙钟参考量级（float64，μ=θ·Bx）：

```
 n=100  k=8    c_gc 43.2µs   c_gθ 5.2µs   c_A 6.8µs    r=8.3
 n=100  k=64   c_gc 332.8µs  c_gθ 5.7µs   c_A 7.7µs    r=58.6
 n=400  k=256  c_gc 5312µs   c_gθ 9.5µs   c_A 37.5µs   r=559
 n=1000 k=512  c_gc 28012µs  c_gθ 13.0µs  c_A 110.6µs  r=2163
```

## 2.2 argmin 规则

赢家 = 最小 `cost_hi`；任何 `cost_lo < winner.cost_hi` 的行、或差距小于计时噪声的行，标为
**contested**；争议里若含 `partition` 本来会选的那一行，**优先用已发布的默认**。

`a`（HMC leapfrog 常数）进入全部三个 HMC 行，所以它**无法在它们之间区分**，只能区分 HMC
行与精确行——这句话要印在记分板上。

**无法回答时**：任何输入 `+inf` → 那一行不能赢。全部 `+inf` → **弃权**，`compile()` 落到
今天的行为，`str(plan)` **逐字节不变**。加一条断言这一点的测试。

## 2.3 那个必须写进设计的度量陷阱

`execute.py:754` 的 `_collapse_reason` 记录：`plated_radiometer(n=25, κ=0.4)`、N=1200 时 SNIS
偏真值 **1.40 个后验 sd**，NUTS 偏 **18.5**，而 NUTS 的 chain ESS（33）**超过** Kish ESS（14）。
**一个以 ESS 计价的目标函数会选中那个离真相远 13 倍的答案。**

规则（写进代码注释和 spec，因为规则会腐烂）：**绝不把 Kish ESS 与 chain ESS 放进同一个
argmin；全图行绝不与链行同场比较。** 加测试断言两类行不会进同一次比较。

## 2.4 接口

`compile(graph, *, strategy: Literal["declared","cost"] = "declared", ...)`。
`strategy="cost"` **只计算并打印记分板，不改任何路由**。
`InferencePlan` 加 `ladder: LadderRecord | None`，`eqx.field(static=True)`，**所有字段 static、
不持有任何 numpy 数组**——把 numpy 塞进 static 的 eqx 字段会在**第二次** trace 上炸 treedef
比较（不是第一次，所以很难发现）。存标量摘要 + 指纹。

`InferencePlan.__str__` 在 `self.streaming.line()` 用的同一个 guard 下追加 `self.ladder.line()`，
这样被弃权的图**逐字节不变**，二十条既有 `str(plan)` 断言仍在测它们原本要测的东西。

新增阈值（争议带宽、计时噪声容差、`k_cg` 里的 tol）→ **全部进注册表，两侧格子。**

D 编号：**D93–D96**。probe：**probe_24**。spec：`docs/superpowers/specs/2026-08-31-p5-costs.md`。

**阶段 2 做完：跑全套，提交。**

---

# 阶段 3：P6 —— collapsed target 与 collapse 臂

## 3.1 范围先收窄（最重要的一条）

**collapse 是小块专用特性，不是通用加速器。** 实测结论，不要重建成通用加速器：

* k=8 时 c ≳ 0.95 collapse 就赢；k=512 时要 c ≳ 0.9995 才赢。
* 截断 trace-log **不救大 k**：按标量阶数比 QR 慢 25–33%，按**认证阶数**慢 **95–130%**
  （梯度误差 2.803e-06）。原因不是数值问题：ρ(X) 在每个 fixture 上都是 0.85–0.93，m 要取
  135–212 阶，上百次算子作用把"不物化矩阵"省下的钱全花掉。根因是 `P = ASAᵀ` 相对
  `Λ = N` **根本不小**，而没有有物理依据的通用 Λ selector（D83）。

把这一段写进 P6 spec 开头，否则下一个人会照通用加速器去建。

## 3.2 实现（算术已验过，不要重建）

`unchecked_operator(probe_gaussian=False)` → `dense_operator` → `compress` →
`SqrtInfo.combine(nuisance_prior)` → `marginalise_arrays`，对稠密 slogdet oracle **1.8e-14**，
梯度 **2.6e-08**，jit 与 hessian 干净。`sqrtinfo.py:277` 折进 offset 的 `−Σ log pivots[:n_block]`
**就是** `0.5·logdet(F_bb)`。

`execute.py:503-518` 的 ladder 加**一个**臂 `collapse`：NUTS 打归约图上的边缘目标，然后对每个
保留的 θ 做一次 vmapped `gcr_sample` 回抽。

两处都必须用 `unchecked_operator(..., probe_gaussian=False)`——`local_block(..., priors=True)`
在 `diagnose/local.py:293` 以 `probe_gaussian` 默认 True 调 `_env_before`，jit 下从
`exact/gaussian.py:249` 抛 `TracerBoolConversionError`。

traced pivots 上加 `eqx.error_if`，使一个在链会到达的 θ 处**无约束**的块**抛出**，而不是
返回一个有限的、看起来合理的数。

`Posterior.diagnostics` 对被精确回抽的那个块保持 `None`（遵守已有弃权语义）。

**顺手修一个潜伏陷阱**：`execute.py:545` 的 `depends_on_prediction = plan.exact.method != "gcr"`
是**字符串比较**，任何新方法都会被读成 prediction-dependent。改成能力查表。

## 3.3 两个明确排除的东西

* **非中心化**：D23 实测这个声明层里一个 `Latent` **不能**参数化另一个 `Latent` 的先验
  （`dist.Normal(<Latent>, 0.5)` 构造得出、`.sample()` 抛 `TypeError`）。重参数化是用户义务，
  解锁条件写在 D23。**不要实现它。**
* **campaign fold**：`campaign.py:91-93` 以 `isolate(graph, names, {})` 构造设计，空 `at` 与
  零硬编码，而 `at` 只到达 `precision_at`(:97) 与 nuisance 先验(:98)——一个双线性图会拿到
  **逐位相同**的 `[[0,0],[0,0]]`，加不加 `at` 一样。**修它是前置条件，不是本阶段一部分**；
  若发现 P6 需要它，停下来报告，不要顺手改。

## 3.4 测试

那条**绝对密度**双计数守卫已经存在且是硬的（误差 7.105e-15 对 atol 2e-9；丢掉 slogdet 项的
变异给 2.203272，高九个数量级）。**不要削弱它。** 新的 collapse 臂要有自己的等价物：
`log_joint(归约图 ⊕ 证据项)` 对 `log_joint(原图)` 在块上稠密积分，K 个点含每个参数的两个端点，
**非单位**先验宽度。

边界验证：在 c 的交叉点上**绕过调度器**，把 split 和 collapse **两条都跑到固定 ESS**（不是跑
调度器——通过调度器测只能看到函数自身连续性，那是规则点名的反模式）。c ∈ {0, 0.5, 0.99, 0.999}，
k 跨四个数量级，κ 跨四个数量级。

新增阈值 → 注册表，两侧格子。D 编号：**D97–D100**。probe：**probe_25**。
spec：`docs/superpowers/specs/2026-08-31-p6-collapse.md`。

**阶段 3 做完：跑全套，提交。**

---

# 阶段 4：P7 —— A4 pilot 与对账账本

## 4.1 pilot 的裁决是非对称的

**只在 A3 会切换离开已发布默认、或差距落在噪声里时才跑。** 从已发布计划抽一个短 pilot，
在**同一批抽样**上算两次典型相关——一次线性特征、一次加平方项——**报告比值**。

它的职责**不是**给一个更好的 c。pilot 的 ESS 支撑不了，而且那个数**不可用**：Neal's funnel
上 Laplace 相关精确 0.0，20 万 iid 抽样上线性 cc 0.0080、二次 cc 0.1157（比值 14.5，但绝对值
远低于任何合理阈值）；另一次独立运行不同特征构造得 0.619——**同一现象量级差 5 倍**。绝对值
是估计器相关的，**比值才是信号**。

裁决语义：
* 二次 cc 同时越过 `sqrt(p_aug / N_eff)` **且**超线性 cc 一个**声明的**倍数 → **否决切换**，
  点名 funnel，弃权；
* 没结论 → A3 的决定照旧生效，打印 `blind_to=("gaussian-only",)`。

**一个卡住的 pilot 代价为零，也不会静默变成抛硬币。** 加测试断言这两条，包括"没结论时必须
保持 A3 的决定"——若它弃权，测试要红。那个"声明的倍数"是新阈值 → **注册表，两侧格子，
provenance 写清楚。**

## 4.2 对账账本

`Posterior` 加 `cost: CostReconciliation | None`，记录：预测区间、实测 秒/ESS、以及**哪个输入
主导**——使一次落空能定位到 `c`（funnel）、`a`（标定不迁移）、还是区间开得太宽。

**这个账本是即使其它每一部分都错了也仍然有价值的那一块**：它不依赖成本表达式是对的，只依赖
它们的输入被记下来了。每次运行的"预测/实测"对就是成本模型的标定数据。

D 编号：**D101–D103**。probe：**probe_26**。spec：`docs/superpowers/specs/2026-08-31-p7-pilot-ledger.md`。

**阶段 4 做完：跑全套，写 README 计数，提交。**

---

# 之后（本计划不包含，仅指路）

- **R1**（artifact/provenance 协议）：`docs/superpowers/plans/2026-08-30-r1-task-artifact-provenance.md`
  Task 1–9。红线"只建最小协议、不改变已有数值结果"；Task 1.4 绿灯条件 = 同一提交把
  `"artifacts"` 加入 `_LAZY_SUBMODULES`。
- **R2–R8**：顶层设计 §8；§14.3 明示"下一块最重要的缺口是通用 model checking"（R3）。

---

# 仓库惯例（贯穿）

先写会失败的测试，跑红，再改实现。

**测试产物每次一个目录**（现有配方有实测陷阱）：

```bash
RUN=$(date +%Y%m%dT%H%M%S)-$$; D=runs/$RUN; mkdir -p "$D"
.venv/bin/python -m pytest -n 4 --junit-xml="$D/junit.xml" > "$D/log" 2>&1
echo "PYTEST_EXIT=$?" > "$D/exit"
```

三个产物同一次调用、同一个目录。**绝不加 -q**（pyproject 已有，第二个会让汇总行消失）。
计数取自 junit，裁决取自退出码文件；只有 exit **1** 才是测试失败（2 中断，3 内部错误，4
用法错，5 未收集到，143 被杀）。

**基线：4690 passed / 0 failed / exit 0。** 不得引入任何失败。README 计数留到最后一次写。

变异验证只用 monkeypatch / pytest 插件 / AST——**绝不用 git checkout / stash / restore**（可能有
并行 session 在改这个 checkout）。`ruff check src/ tests/` 保持干净；**不要跑 ruff format**（31 文件
/ 517 行刻意漂移；真要扫就传那 31 个文件名）。

不要碰 `docs/superpowers/specs/` 下的顶层设计或架构文档——那些属于仓库所有者。

---

# 回报

- **阶段 0**：F1 去掉门后"κ(Σ)≥1/eps 但 lemma 精确"格子 ADMITTED、"证据坏"格子 REFUSED；
  魔数选了 (a) 还是 (b) 及理由；F3/F4 各自一行结论。
- **阶段 1**：快层测试数与耗时；全层耗时；元测试是否强制"每门至少一格在快层"。
- **阶段 2**：三个成本表达式在一个真实 fixture 上的记分板输出；弃权路径 `str(plan)` 是否
  逐字节不变；新增了几个阈值、是否都登记。
- **阶段 3**：collapse 臂在 c 交叉点两侧跑到固定 ESS 的实测对比；双计数守卫灵敏度是否保持。
- **阶段 4**：pilot 在 funnel 上的线性/二次 cc 比值与裁决；账本在一次真实运行定位到哪个输入。
- **全程**：新增阈值总数、其中 `magic` 的个数（**目标 0**）、每个新门两侧变异是否都能变红。
