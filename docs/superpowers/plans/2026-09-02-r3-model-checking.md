# R3 执行计划：model checking 与 calibration

> **文档状态：`plan-active`** · 尚未执行完的计划，仍指导后续工作。索引见 docs/README.md。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **本计划只写到计划为止，任何 task 都要等 owner 过目后再在单独的执行 session 里开跑。**

**Goal:** 把 R2 造出来的 `PredictiveResult`、pointwise log-likelihood 与 ArviZ export 提升为一层**通用而非 campaign-specific 的模型检验**：prior / posterior predictive check、held-out prediction、SBC、经 ArviZ 的 LOO、identifiability 与 prior-sensitivity 报告——每一份都是引用 Result 的 typed `EvaluationReport`，走 R1 冻结的 applicability × conclusion 两轴，并由一个版本化的 gate 定义确定性地聚合；已知校准的 fixture 要 PASS、已知误设的要 FAIL、预算不足的要 ABSTAIN；随机验收固定 seed、预算与预先声明的误报率。同时把 SBC 所需的 prior simulation 接成 `SimulationTask` 的真执行，让本地 reference NPE 第一次拿到一个可比的校准数。

**Architecture:** 新增一个 **evaluation 层**（`bayesmith.evaluation`，顶层设计 §7.1 点名的概念边界之一），依赖方向严格向下：`evaluation → dispatch / graph / artifacts / bridge.arviz`；`artifacts` 叶层与 `dispatch` 桥**都不** import 它（`tests/test_layering.py` 加断言）。evaluation **只评价 Result、不改 Result、不选算法**（§2.4）：每个检查读 `PosteriorResult` / `PredictiveResult` / `SimulationResult` 的字段，产出 `EvaluationReport`，`subject_ref` 指向被评价的 Result。前向生成**全部复用** R2 的 `dispatch.predictive`（`observation_parts` 一次读 loc/scale；replay 与 replicate 只差 log_prob vs sample），SBC 与 prior predictive 所需的 prior 抽样作为同一模块的第三个原语落地，不另写平行的 simulator（不变量 1）。ArviZ 继续是 optional、export-only 的上游：LOO 由 `arviz.loo` 算，bayesmith 只拥有 observation grouping、适用性判断与 artifact（§2.4、§0.8 of R2）。

**Tech Stack:** Python 3.11 frozen/slots dataclasses、NumPy、SciPy（`scipy.stats.kstest`，jax 的既有传递依赖）、JAX；NumPyro 只在 dispatch/bridge 层；ArviZ optional（`skipif`，不 `importorskip`，见 R2 close-out 的 wheel 教训）；pytest、ruff；`tests/numerical_gates/registry.py` 登记新阈值。

**Spec:** [Bayesmith 顶层设计](../specs/2026-08-30-bayesmith-top-level-design.md) §2.4（Evaluation 只评价 Result）、§3.3（predictive 与模型检验）、§3.6（SimulationTask 承载 prior simulation 与 SBC data generation）、§4.3（gate 真值表）、§8 R3（目标、九条主要工作、八条门槛）、§9.3（随机验收纪律）、§9.5（模型检验专项测试）；[R2 close-out](../specs/2026-08-31-r2-close-out.md)；[R2 计划](2026-08-31-r2-predictive-seam.md) §0（conditioning vs replicated、observation unit、ESTIMATOR）；[R1 计划](2026-08-30-r1-task-artifact-provenance.md)（结构样板）。

---

## 0. 冻结裁决与执行边界

这些是 R3 的实现输入，不在执行过程中重新发明。每一条都写「为什么」，magic 数一个不留；每个数字来自 §0.12 的 probe_28 实测。

### 0.1 evaluation 层的家：`bayesmith.evaluation`，依赖只向下

**裁决：** 新建子包 `src/bayesmith/evaluation/`：`checks.py`（prior / posterior predictive check）、`sbc.py`、`heldout.py`、`loo.py`（ArviZ 桥的消费面）、`diagnostics.py`（identifiability / prior-sensitivity 投影）、`gate.py`（`model_checking@1` 定义与 `check_posterior`）。它 import `dispatch.predictive`、`dispatch.task`、`graph.evaluate`、`artifacts.*`、`bridge.arviz`；**反向一律不许**。`"evaluation"` 在 Task 1 的同一提交里进 `bayesmith.__init__._LAZY_SUBMODULES`（R1 教训：`test_public_api` 从文件系统推导期望集合，晚登记就是七个提交的红）。

**为什么：** 顶层设计 §7.2 的依赖链是 `model/graph → analysis/compiler → execution adapters → evaluation → workflow`；把检查塞进 `dispatch.task`（已 2097 行）会让执行层反过来评价自己的产物，正是 §2.4「Evaluation 只评价 Result，不修改 posterior，也不替执行层选择新算法」要防的。`EvaluationReport` 的 `report_kind` 是 code 字符串（R1 冻结），新报告种类**不改 schema**。

### 0.2 报告种类与 verdict 语义，一表定死

| `report_kind` | subject | APPLICABLE 时的 PASS / FAIL | ABSTAIN / UNVERIFIABLE |
|---|---|---|---|
| `posterior_predictive_check` | `PredictiveResult` | p 在 D104 带内 PASS，带外 FAIL | draws < D105 → ABSTAIN；discrepancy 无定义 → UNVERIFIABLE |
| `prior_predictive_check` | `SimulationResult`(PRIOR) | 同上，p 对 prior predictive 算 | 同上 |
| `held_out_prediction` | `PredictiveResult`（图带 mask） | 任一 held-out 点的 PIT 落在 Bonferroni 校正的 D104 尾部外 FAIL，否则 PASS | 无 held-out 点（mask 全 True）→ INAPPLICABLE；predictive 不可用 → UNVERIFIABLE |
| `loo_psis` | `PredictiveResult` / `PosteriorResult`（有 pointwise） | arviz 自己的 Pareto-k 规则无警告 PASS | arviz 报 warning → ABSTAIN（估计不可靠，不是模型坏）；arviz 未装 → UNVERIFIABLE；无 pointwise → INAPPLICABLE |
| `sbc` | `PosteriorResult`（作为路由代表）| 每个 latent 坐标的 rank KS p ≥ α/K PASS，任一 < α/K FAIL | 复制数 N < D106 → ABSTAIN；任一复制被 Refusal / 未收敛 → ABSTAIN 并计数 |
| `identifiability` | `PosteriorResult` 的图 | nullity == 0 PASS，> 0 FAIL（finding 带 participation） | 环境 float32 或诊断拒绝 → UNVERIFIABLE |
| `prior_sensitivity` | 同上 | worst \|shift_sigma\| < `CRITERION_SHIFT` PASS，≥ FAIL | refit 未收敛 / `verified` 有 False → ABSTAIN；float32 → UNVERIFIABLE |

**为什么：** §2.4 要求「运行失败应单独表达，不能伪装为 FAIL；统计方法不适用也不能伪装为运行错误」。表里每一格都能从 artifact 字段算出，不读日志（§8 R3 门槛 8）。`identifiability` 与 `prior_sensitivity` 的阈值**已经住在** `diagnose/` 里（`DEFAULT_RANK_RTOL`、`CRITERION_SHIFT`，各有登记），投影层只读 verdict 字段，不再判一次（一个决定一个家）。

### 0.3 discrepancy 的契约：值进 artifact，callable 不进

**裁决：** discrepancy 是 `(y, loc) -> 每 draw 一个标量` 的具名 callable。默认集合五个、稳定且不依赖协变量：`mean`、`sd`、`min`、`max`（只读 `y`）与 `residual_sd`（读 `y - loc`）。用户自定义的 callable 在评价时传入，报告只记它的 **importable identity**（`module.qualname`，与 `dispatch.amortized._embed_identity` 同一规则；lambda / REPL 函数拒绝）和逐 draw 的测量值。p 值按源 posterior 的权重算：`p = Σ_i w_i · 1[T(rep_i, loc_i) ≥ T(obs, loc_i)]`，draw 轴一一对应（R2 §0.5：不重采样、不丢权重）。

**为什么：** §3.3「框架应支持用户提供 discrepancy function，同时提供少量通用、稳定的默认统计量」；§0 ruling 4（R1）：artifact 里没有 callable。实测 §0.12：正确模型的 p 在 0.33–0.91 之间，`curvature=0.6` 的误设两条统计量都是 **0.0000**，而 `curvature=0.15` 的轻误设 p=0.43 / 0.34——**PPC 的功效是有限的**，这条要写进 `docs/evaluation.md`，不许把「PASS」讲成「模型正确」。

### 0.4 D104：所有检查共用一个声明的误报率

**裁决：** 一个声明的两侧误报率 **α = 0.05**（`evaluation.ALPHA`，D104，provenance **borrowed**：统计学的常规默认，不是从本仓库推导的）。PPC / prior predictive 的通过带是 `[α/2, 1 − α/2]`；held-out 对 m 个点做 Bonferroni（`α/(2m)` 每尾）；SBC 对 K 个 latent 坐标做 Bonferroni（`α/K`）。Bonferroni 是推导不是新阈值，不另编号。整套 R3 测试套件的期望误报数在 close-out 里**算出来写下**（每条随机验收一行：seed、N、α、期望误报）。

**为什么：** §9.3「必须固定 seed 与计算预算，并预先声明统计量、容忍区间和可接受误报率」；一个 α 一个家，否则每个检查各挑一个数就是五个 magic。

### 0.5 D105：PPC 的 draw 地板，从 D104 推导

**裁决：** p 值的分辨率是 `1/N_draws`，要能落进 `α/2` 宽的尾部，`N_draws ≥ ceil(1/(α/2)) = 40`（D105，provenance **derived**，公式写进 registry）。少于 40 draws 的 `PredictiveResult` 得到 ABSTAIN（`finding: draws_below_resolution`），不是 PASS。

**为什么：** 现有测试的 `BUDGET = ComputeBudget(draws=8, ...)` 会让 p 只能取 0, 1/8, …——一个 8 draw 的检查永远「通过」或永远「失败」，两种都不是判断。

### 0.6 SBC：连续加权 rank、KS、D106 复制数地板

**裁决：** 每个 latent 标量坐标的 rank 是连续加权量 `r = Σ_i w_i · 1[θ_i < θ_true] ∈ [0, 1]`（加权 posterior 用权重，iid 用 1/N），复制间用 `scipy.stats.kstest(ranks, "uniform")`。复制数地板 **D106 = 50**（provenance **derived**，probe_28 §4：2 倍过宽 / 过窄两个 mutant 在 N=50 时 p=0.013 / 0.0003，N=20 时过宽只到 0.033——离 α 不到 1.5 倍；N=10 时过宽 p=0.389，检不出）。**这是一组 seed 的测量**，Task 6 登记前必须换 10 组 seed 重测并把最差值写进 registry 的 provenance。N < 50 → ABSTAIN（`finding: replicates_below_floor`）。

复制循环 = `SimulationTask(PRIOR)` 生成 `(θ_true, y)` → 同一 model callable 在 `y` 上 `compile_task` + `PosteriorTask` → rank。**同一 model callable**，不是同一 Graph：每个复制的 data 不同，Graph 由 `trace(model, data)` 重建，`model_ref` 相同。任一复制返回 Refusal 或 `termination != COMPLETED/CONVERGED`，整份报告 ABSTAIN 并在 finding 里计数——SBC 检验的是「算法 + 实现」，一个静默丢掉失败复制的 SBC 检验的是别的东西。

**为什么：** §3.3「SBC 既检验推断算法，也暴露模型实现与参数化错误」；§8 R3 门槛「SBC 能同时覆盖 exact 与 sampled 路由」。实测 §0.12：exact 路由 N=100 正确 KS p=0.68，2 倍过宽 p=0.0012、过窄 p=0.0001；成本 0.42 s / 复制；NUTS 路由（bilinear_pair，100 warmup + 100 draws）0.59 s / 复制。

### 0.7 SimulationTask 真执行，prior 抽样是 `dispatch.predictive` 的第三个原语

**裁决：** `SUPPORTED_TASK_KINDS` 加 `SIMULATION`；`execute_task` 对三种 `ParameterSource` 各走一条：`PRIOR` → 新原语 `prior_draws(graph, key, n)`（逐节点 `apply_probabilistic(...).sample`，观测节点也抽，即 prior predictive）；`FIXED` → `evaluate` + 观测节点抽样；`POSTERIOR_RESULT` → 源 posterior 的 latent draws + `replicated_draws`（**与 PredictiveTask 同一生成律，同 key 逐位相同**，Task 2 用测试钉住）。plated 节点：`apply_probabilistic` 对「plated 但无 plated parent」的节点返回未映射的分布，其 `.sample` 是一个共享值不是一个 plate（probe_28 §8 实测拒绝），原语用 `graph.plate_size(name)` 展开 `sample_shape`。`EvidenceTask` 继续 `capability_unavailable_r1`。

**为什么：** §3.6「SimulationTask 统一承载 prior simulation、SBC data generation、synthetic oracle construction」；R2 §0.3 已裁决观测生成律 = 节点自己的分布。没有它，SBC 与 prior predictive 各写一个 simulator，就是不变量 1 的反例。

### 0.8 held-out 由 `observe(mask=...)` 定义，不另造 held-out 参数

**裁决：** held-out 点 = 观测节点 `observed_mask` 为 False 的位置。conditioning 只用 mask 为 True 的点（`log_joint` 与 `pointwise_log_likelihood` 已如此，实测 masked 位置 pointwise LL 恒为 0），prediction 覆盖所有点（`replicated_draws` 的 loc/scale 在 masked 位置照样存在）。报告携带每个 held-out 点的 PIT `F_pred(y_j)` 与 `elpd_heldout = Σ_j log Σ_i w_i p(y_j | θ_i)`；verdict 按 §0.2。R2 §0.4 的覆盖域不变：correlated / 非 Gaussian 观测节点 → `predictive_noise_unsupported`。

**为什么：** §3.3「held-out prediction：对未参与 conditioning 的 observation…计算 predictive performance」。mask 是图上已有的声明（G1 wiring），复用它就没有第二个「哪些点被条件了」的家。实测 §0.12：正确直线 elpd_heldout = −1.019，`curvature=0.6` 的误设 = −24.047，2 个 held-out 点。

### 0.9 LOO 走 `arviz.loo`；WAIC 本包不提供

**裁决：** `loo_psis` 报告由 `bayesmith.bridge.arviz.to_inference_data` + `arviz.loo` 产生。**iid 结果必须以 `chain_shape=(1, n)` 导出**——实测 `az.loo` 对只有 `draw` 轴的 export 抛 `AttributeError: 'DataArray' object has no attribute 'chain'`；这是 arviz 1.x 的 sample_dims 约定，`loo.py` 负责补 chain 轴而**不改** R2 的 export 语义（`DrawsPosterior.chain_shape is None` 仍表示「无链结构可诊断」）。verdict 用 arviz 自己的 `warning` 字段（其 `good_k` 规则），**不在本包再放一个 0.7**；finding 记 `elpd / se / p / n_data_points / max_pareto_k / arviz 版本`。**WAIC：arviz 1.3.0 顶层无 `waic`**（实测 `hasattr(az, "waic") is False`），按 §1.5「R-hat、ESS、PSIS、LOO、WAIC 等成熟通用统计量」归上游，本包**不自实现** WAIC，R3 的「LOO/WAIC」项落地为 LOO-PSIS，WAIC 在 close-out 记为「所选上游不提供」。

**为什么：** §2.4「ESS、R̂、PSIS、Pareto k̂ 等成熟的低层统计量应优先复用 BlackJAX 或 ArviZ」；§7.3 optional 缺失优雅退化（未装 → UNVERIFIABLE 报告，不是异常）。实测 §0.12：straight_line 2000 draws elpd −6.914 ± 1.937、p_loo 0.666、max k 0.553；bilinear_pair NUTS (2, 400) elpd −1.312 ± 0.835、max k 0.426。

### 0.10 identifiability / prior_sensitivity 的投影要在 x64 里建图

**裁决：** 两条诊断都拒绝 float32 环境（实测 `refuse_ambient_float32`，且要求**图在 `jax.enable_x64(True)` 块内构建**）。`diagnostics.py` 接收 graph；环境是 float32 时返回 UNVERIFIABLE 报告并把拒绝文本放进 finding，不抛。测试标 `x64` marker（pyproject 已有：单独 session 跑）。verdict 只读 `IdentifiabilityReport.nullity / participation` 与 `PriorSensitivityReport.worst / verified / refit_converged / criterion_std`。

**为什么：** §3.3「静态结构诊断、局部 curvature、posterior geometry 和有计划的 prior perturbation 应形成相互区分的报告」；阈值住在 `diagnose/`。实测 §0.12：`collinear_pair` nullity=1、参与度 a:0.5 / b:0.5；`straight_line` worst shift −0.042σ，criterion_std 1.297 < prior_std 2。

### 0.11 amortized backend 评估：同一 harness，先量本地 reference，候选另记一页

**裁决：** SBC harness 接受任何 `sampler(datum, key, n) -> draws`，与 `PosteriorTask` 路由同一套 rank / KS 判据。Task 9 先用它给本地 `NeuralPosterior` 一个数（实测 §0.12：bank 2048、1500 步、4.0 s 训练，300 次 prior 复制 KS p=0.88，90% 区间覆盖 0.900）。BayesFlow / sbiJAX **不加进依赖**：在 scratch venv 里跑同一 harness、同一 bank、同一预算，结果写成一页 `record`；按 §1.5 六条门槛裁决；**没有候选通过就记录结论，不伪造 production winner，本地 reference NPE 不退役**（§8 R3 门槛 6）。

### 0.12 现状实测（probe_28，本 checkout，2026-09-02）

命令：`PYTHONPATH=. .venv/bin/python docs/probes/probe_28_model_checking_seams.py 100`（全部九段 331 s；第 9 段单独 `… 100 9`）。probe 随本计划提交，Task 1 只把它接成 pin。

| 段 | 量 | 值 |
|---|---|---|
| 1 PPC | straight_line (gcr) p_curvature / p_scale | **0.9120 / 0.3270** |
| 1 | radiometer (gcr+snis，加权 p) | 0.3228 / 0.7463 |
| 1 | curved_line curvature=0.6（误设） | **0.0000 / 0.0000** |
| 1 | curved_line curvature=0.15（轻误设） | 0.4300 / 0.3410 |
| 2 LOO | flat draw 轴的 export → `az.loo` | `AttributeError: … no attribute 'chain'` |
| 2 | `chain_shape=(1, 2000)`：elpd / se / p_loo / max k | −6.914 / 1.937 / 0.666 / 0.553 |
| 2 | bilinear_pair NUTS (2, 400)：elpd / se / p_loo / max k | −1.312 / 0.835 / 0.679 / 0.426 |
| 2 | `hasattr(arviz, "waic")` (1.3.0) | False |
| 3 SBC exact | N=100 正确：KS D / p，5 桶 | 0.070 / **0.6847**，[21, 22, 21, 21, 15] |
| 3 | 2 倍过宽 / 过窄：KS p | **0.0012 / 0.0001** |
| 3 | 成本 / 复制（compile + 100 gcr draws） | 0.42 s |
| 4 功效 | N = 10 / 20 / 50 / 100：过宽 p | 0.389 / 0.033 / **0.013** / 0.005 |
| 4 | N = 10 / 20 / 50 / 100：过窄 p | 0.034 / 0.019 / 0.0003 / 0.003 |
| 4 | N = 10 / 20 / 50 / 100：正确 p | 0.346 / 0.414 / 0.337 / 0.843 |
| 5 SBC NUTS | bilinear_pair 100+100，成本 / 复制 | 0.59 s（首次 0.66 s） |
| 6 NPE | 2048 bank、1500 步：训练时间 / best_step | 4.0 s / 322 |
| 6 | 300 次 prior 复制：KS D / p；90% 覆盖 | 0.033 / **0.8815**；**0.900** |
| 7 held-out | 2 点；正确 / 误设 elpd_heldout | **−1.019 / −24.047** |
| 7 | masked 位置的 pointwise LL | 恒 0（exact 路由接受 mask） |
| 8 prior pred. | straight_line：sd(d[-1]) prior / observed d[-1] | 8.075 / 10.247 |
| 8 | P(max\|d_prior\| ≥ max\|d_obs\|)：N(0,2) 先验 / N(0,1e6) 先验 | **0.210 / 1.0000** |
| 8 | plated 节点（plated_latent）的 prior 抽样 | 需按 `plate_size` 展开（probe 拒绝） |
| 9 oracle | gain：bayesmith r_hat / ess vs arviz rhat / ess_bulk | 1.0083 / 118.5 vs 1.0173 / 110.9 |
| 9 | t_ant：同上 | 1.0199 / 111.2 vs 1.0262 / 109.3 |
| 9 | identifiability：straight_line / collinear_pair nullity | 0 / **1**（参与度 a 0.5、b 0.5） |
| 9 | prior_sensitivity worst shift：straight_line / radiometer | −0.0421σ / −0.0014σ |
| 9 | 两条诊断在 float32 环境 | `GraphError`（要求 x64 内建图） |

第 9 段的 R̂ 差异是 arviz 用 rank-normalized split-R̂、本包用 split-R̂ 所致；R3 **不**用 arviz 替换 `SiteDiagnostic`，只加一条两侧 verdict 一致的 cross-check 测试（Task 7）。

---

## Task 1：evaluation 骨架、分层守卫与 probe_28 的 pin

**Files:**

- Create: `src/bayesmith/evaluation/__init__.py`（`ALPHA`、公开面清单，先空实现）
- Modify: `src/bayesmith/__init__.py`（`_LAZY_SUBMODULES` 加 `"evaluation"`）
- Modify: `tests/test_layering.py`（`artifacts`、`dispatch`、`graph` 不 import `evaluation`；subprocess 断言 `import bayesmith.evaluation` 不拉 arviz）
- Create: `tests/evaluation/__init__.py`、`tests/evaluation/test_probe28_pins.py`

- [ ] **1.1 红灯**：写 layering 测试与 `test_public_api` 期望（子包存在即被要求登记）；写 pin 测试：probe_28 §1 的 straight_line p_curvature=0.9120（`rel_tol=1e-3`）、§3 的 N=20 正确 KS p 与过宽 p 分居 α 两侧、§8 的 0.210 / 1.0000。运行 `.venv/bin/python -m pytest tests/evaluation tests/test_layering.py tests/test_public_api.py`，预期 FAIL。
- [ ] **1.2 实现**：骨架 + 登记；pin 测试只调用 probe 里已有的原语（`replicated_draws`、`prior_draws` 暂由 probe 提供）。
- [ ] **1.3 绿灯**：同上命令 PASS；`.venv/bin/ruff check src/bayesmith/evaluation tests/evaluation tests/test_layering.py` 干净。
- [ ] **1.4 提交**：`git add src/bayesmith/evaluation/__init__.py src/bayesmith/__init__.py tests/test_layering.py tests/evaluation/__init__.py tests/evaluation/test_probe28_pins.py`，commit `"feat: open the evaluation layer and pin the model-checking seams"`。

---

## Task 2：SimulationTask 真执行（prior / fixed / posterior_result）

**Files:**

- Modify: `src/bayesmith/dispatch/predictive.py`（新原语 `prior_draws`、`forward_draws`，plate 展开）
- Modify: `src/bayesmith/dispatch/task.py`（`SUPPORTED_TASK_KINDS` += SIMULATION；`_run_simulation`）
- Modify: `tests/dispatch/test_predictive_seam.py`、`tests/dispatch/test_task_execution.py`、`tests/dispatch/test_task_protocol.py`

- [ ] **2.1 红灯**：`compile_task(SimulationTask)` 返回 `PlannedTask`；`execute_task` 对 PRIOR 返回 `SimulationResult`（`latent_draws` + `observation_draws` 共享 draw 轴，`parameter_source=PRIOR`）；prior 抽样的一阶二阶矩对 straight_line 闭式（w 的 N(0,2)、d 的 N(0, 4x²+0.25)）在 MC 误差内；plated_latent 的 `z` 抽出 `(n, 6)`；FIXED 的 `observation_draws` 均值等于 `evaluate` 的 loc；POSTERIOR_RESULT 的 `observation_draws` 与同 key 的 `PredictiveResult.replicated_draws` **逐位相同**。`EvidenceTask` 仍 `capability_unavailable_r1`。预期 FAIL。
- [ ] **2.2 实现**：原语只依赖 `graph.evaluate` / `graph.nodes` / `graph.plate_size`；`_run_simulation` 机械投影，`RunRecord.approximation = MONTE_CARLO / EXACT`，`budget.draws` 记 n。
- [ ] **2.3 绿灯**：`.venv/bin/python -m pytest tests/dispatch tests/artifacts` PASS；ruff 干净。旧测试 `test_the_three_tasks_r1_cannot_answer_are_refused_as_a_capability`（或其 R2 后继）的 simulation 分支改成断言真执行，只留 evidence——刻意的测试变更。
- [ ] **2.4 提交**：`git add src/bayesmith/dispatch/predictive.py src/bayesmith/dispatch/task.py tests/dispatch/test_predictive_seam.py tests/dispatch/test_task_execution.py tests/dispatch/test_task_protocol.py`，commit `"feat: execute simulation tasks from prior, fixed and posterior sources"`。

---

## Task 3：posterior predictive 与 prior predictive check（D104、D105）

**Files:**

- Create: `src/bayesmith/evaluation/checks.py`
- Modify: `src/bayesmith/evaluation/__init__.py`
- Create: `tests/evaluation/test_checks.py`
- Modify: `tests/numerical_gates/registry.py`、`tests/numerical_gates/source_scan.py`（`SOURCE_PATHS` += `checks.py`；D104 borrowed、D105 derived，两侧格子 + tighten/loosen 变异）

- [ ] **3.1 红灯**：默认五个 discrepancy 对 straight_line 的 `PredictiveResult`（≥ 40 draws）报 APPLICABLE × PASS；`curved_line(0.6)` 用 `residual_sd` 报 FAIL（p=0.0000）；用户 callable 的 identity 进 finding、lambda 拒绝、callable 进不了 artifact（codec 拒绝）；draws=8 → ABSTAIN `draws_below_resolution`；加权源（radiometer）的 p 用权重（与 probe 0.3228 / 0.7463 对上）；prior predictive：straight_line PASS（0.210）、N(0, 1e6) 先验 FAIL（1.0000）；report 的 `subject_ref` 指向被评价的 Result；round-trip 不丢 finding。预期 FAIL。
- [ ] **3.2 实现**：p 值、带、地板都读 `ALPHA` 与推导式；registry 登记 D104 / D105 并让 meta-test 认出 `checks.py`。
- [ ] **3.3 绿灯**：`.venv/bin/python -m pytest tests/evaluation tests/numerical_gates -m "not full"` PASS；ruff 干净。
- [ ] **3.4 提交**：`git add src/bayesmith/evaluation/checks.py src/bayesmith/evaluation/__init__.py tests/evaluation/test_checks.py tests/numerical_gates/registry.py tests/numerical_gates/source_scan.py tests/numerical_gates/source_manifest.py`，commit `"feat: add prior and posterior predictive checks with a declared false-positive rate"`。

---

## Task 4：held-out prediction 报告

**Files:**

- Create: `src/bayesmith/evaluation/heldout.py`
- Create: `tests/evaluation/test_heldout.py`

- [ ] **4.1 红灯**：mask 掉 2 点的直线：PIT 全在带内 PASS，`elpd_heldout` 与 probe 的 −1.019 对上（`abs_tol=1e-2`，同 key）；`curved_line(0.6)` FAIL 且 elpd −24.047；mask 全 True → INAPPLICABLE；correlated 观测 → UNVERIFIABLE（承 `predictive_noise_unsupported`）；Bonferroni 因子等于 held-out 点数（用 m=1 与 m=2 两个 fixture 钉住方向）。预期 FAIL。
- [ ] **4.2 实现**：只读 `PredictiveResult.replicated_draws` 的 loc/scale（经 `observation_parts` 在 latent draws 上重算，不重抽）与 `observed_mask`。
- [ ] **4.3 绿灯**：`.venv/bin/python -m pytest tests/evaluation/test_heldout.py` PASS；ruff 干净。
- [ ] **4.4 提交**：commit `"feat: score held-out observations declared by the graph's mask"`。

---

## Task 5：LOO 经 ArviZ（optional）

**Files:**

- Create: `src/bayesmith/evaluation/loo.py`
- Create: `tests/evaluation/test_loo.py`（`requires_arviz = pytest.mark.skipif(...)`，照 `tests/bridge/test_arviz_export.py`）

- [ ] **5.1 红灯**：iid `PredictiveResult` → 报告 elpd/se/p/max k 与 probe（−6.914 / 1.937 / 0.666 / 0.553）对上；NUTS (2, 400) 走源 posterior 的 `chain_shape`；`warning=False` → PASS；构造一个 arviz 会警告的 fixture（极重尾权重）→ ABSTAIN；`sys.modules` 里塞 `arviz=None` 的 monkeypatch → UNVERIFIABLE 报告而非异常；无 pointwise 的源 → INAPPLICABLE；`import bayesmith.evaluation.loo` 不拉 arviz。预期 FAIL。
- [ ] **5.2 实现**：`chain_shape=(1, n)` 只在 `loo.py` 内补，finding 记 `arviz.__version__`。
- [ ] **5.3 绿灯**：`.venv/bin/python -m pytest tests/evaluation/test_loo.py tests/bridge` PASS（有 arviz）/ 声明 skip（无）；ruff 干净。
- [ ] **5.4 提交**：commit `"feat: bridge PSIS-LOO through the optional ArviZ export"`。

---

## Task 6：SBC harness（exact + NUTS + 任意 sampler；D106）

**Files:**

- Create: `src/bayesmith/evaluation/sbc.py`
- Create: `tests/evaluation/test_sbc.py`
- Modify: `tests/numerical_gates/registry.py`、`source_scan.py`（`SOURCE_PATHS` += `sbc.py`；D106 derived）

- [ ] **6.1 先重测**：用 probe_28 §4 的扫描换 10 组 seed（`jax.random.key(23+k)`），把 2 倍过宽 / 过窄在 N ∈ {20, 50, 100} 的最差 p 写进 registry 的 provenance；若 N=50 的最差 p 越过 α/3，把 D106 抬到 100 并记录。
- [ ] **6.2 红灯**：合成 rank 的单元测试（确定性：均匀 rank PASS、堆中间 FAIL、N<地板 ABSTAIN、任一复制 Refusal → ABSTAIN 计数、K 个坐标的 Bonferroni）；exact 路由 N=50 正确 PASS、2 倍过宽 mutant（monkeypatch 拉伸 draws）FAIL——**full 层**；NUTS 路由（bilinear_pair，100+100）N=50 —— **full 层**（预计 30 s）；fast 层只跑 N=20 的 plumbing 用例并断言 ABSTAIN-by-budget。任意 sampler 接口：一个闭式 Gaussian sampler PASS、其 2 倍宽版本 FAIL。预期 FAIL。
- [ ] **6.3 实现**：复制循环走 `SimulationTask(PRIOR)` + `compile_task` / `execute_task`；每复制记 termination；seed 由 task 的 key 派生（`fold_in`），报告 finding 记 N、α、seed、每坐标 KS D / p。
- [ ] **6.4 绿灯**：`.venv/bin/python -m pytest tests/evaluation/test_sbc.py`（含 `-m full` 一次）PASS；ruff 干净。
- [ ] **6.5 提交**：commit `"feat: add the simulation-based calibration harness"`。

---

## Task 7：identifiability / prior-sensitivity 投影 + chain diagnostics 的 ArviZ 对照

**Files:**

- Create: `src/bayesmith/evaluation/diagnostics.py`
- Create: `tests/evaluation/test_diagnostics.py`（`x64` marker）、`tests/evaluation/test_chain_diagnostics_oracle.py`（`requires_arviz`）

- [ ] **7.1 红灯**：x64 内：straight_line identifiability PASS、collinear_pair FAIL 且 finding 带 participation {a: 0.5, b: 0.5}；prior_sensitivity straight_line PASS（−0.0421σ）、一个先验极紧的 fixture FAIL（shift ≥ 0.1σ）；refit 未收敛 fixture → ABSTAIN；float32 环境 → UNVERIFIABLE，finding 含拒绝文本，不抛。ArviZ 对照：bilinear_pair (2, 400) 上本包 `converged` 与 arviz `rhat < 1.01`/`ess_bulk ≥ 100` 的 verdict 一致（两侧各一 fixture：收敛的与冻结坐标的）。预期 FAIL。
- [ ] **7.2 实现**：投影只读报告字段；不重算 rank / shift。
- [ ] **7.3 绿灯**：`JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/evaluation/test_diagnostics.py -m x64` 与常规 session 下的其余测试都 PASS；ruff 干净。
- [ ] **7.4 提交**：commit `"feat: project identifiability and prior sensitivity into evaluation reports"`。

---

## Task 8：`model_checking@1` gate 与 `check_posterior`

**Files:**

- Create: `src/bayesmith/evaluation/gate.py`
- Create: `tests/evaluation/test_gate.py`

- [ ] **8.1 红灯**：`GateDefinition("model_checking", 1, requirements=…)`：required = `posterior_predictive_check`、`identifiability`；optional = `prior_predictive_check`、`held_out_prediction`、`loo_psis`、`sbc`、`prior_sensitivity`、`chain_diagnostics`。`check_posterior(graph, posterior, *, key, budget)` 跑适用的检查、按 slot 组装、`aggregate_gate`；三条路径各一 fixture：straight_line → EVALUATED × PASS；curved_line(0.6) → FAIL；draws=8 → ABSTAIN（`required_report_abstained`）；同一输入两次调用 `GateResult` 相等（确定性）；slot 顺序打乱结果不变（复用 R1 的排列测试）。预期 FAIL。
- [ ] **8.2 实现**：不改 `aggregate_gate`；`check_posterior` 不决定任何数，只编排。
- [ ] **8.3 绿灯**：`.venv/bin/python -m pytest tests/evaluation tests/artifacts/test_gates.py` PASS；ruff 干净。
- [ ] **8.4 提交**：commit `"feat: define the model_checking gate and its deterministic runner"`。

---

## Task 9：amortized 校准记录（本地 reference NPE + 候选协议）

**Files:**

- Create: `docs/superpowers/specs/2026-09-XX-amortized-calibration.md`（`record`）
- Create: `docs/probes/probe_29_amortized_candidates.py`
- Modify: `tests/evaluation/test_sbc.py`（本地 NPE 经 harness 的 pin，full 层）

- [ ] **9.1** 用 Task 6 的 harness 给 `NeuralPosterior`（tests/test_amortize 的问题，bank 2048、1500 步、seed 固定）一个 SBC 数，与 probe §6（KS p=0.88、覆盖 0.900）对上，pin 进 full 层。
- [ ] **9.2** 在 scratch venv 里尝试安装 BayesFlow 与 sbiJAX；能装的用 probe_29 跑**同一 bank、同一 harness、同一预算**；不能装的记录版本与失败原因。
- [ ] **9.3** 记录页按 §1.5 六条门槛逐条填；结论只能是「候选 X 通过 → 进入 §7.3 的替换评估」或「无候选通过 → reference NPE 保留」，不得是第三种。
- [ ] **9.4 提交**：commit `"docs: record the amortized calibration baseline and candidate protocol"`。

---

## Task 10：文档同步、registry 收口与 R3 close-out

**Files:**

- Create: `docs/evaluation.md`（`module-spec`：报告种类表、verdict 表、α 与地板、PPC 功效的诚实说明）
- Modify: `docs/artifacts.md`（R2 之后已过期的三句：PredictiveTask 已执行、`ArtifactKind.ESTIMATOR`、amortized 不再 reserved；加 R3 报告种类）
- Modify: `docs/ownership.md`（`bayesmith.evaluation.*` first-party core；`bayesmith.bridge.arviz` thin adapter 一行——R2 漏登；amortized 行改「pending R3 → 已校准」）
- Modify: `README.md`（模型检验一段 + 测试数）
- Create: `docs/superpowers/specs/2026-09-XX-r3-close-out.md`

- [ ] **10.1** 建验收矩阵：G1–G8 + source / lint / wheel / consumer 四共享 gate，先列空 measured cell。
- [ ] **10.2** 跑 source full suite（含 `-m full` 与 x64 session）、`ruff check src/ tests/`、built-wheel suite、rheplicant consumer gate（照 R2 close-out 命令）。
- [ ] **10.3** 算出并写下整套随机验收的期望误报数（每条 seed / N / α 一行）。
- [ ] **10.4** 用本次真实 SHA / JUnit / consumer revision 写 `R3 closed` 或 `R3 open` + blocker；`tools/sync_doc_index.py`；README 计数。
- [ ] **10.5 提交**：commit `"docs: close the R3 model-checking layer"`。

---

## 红线（每个 task 提交前自查，violate 即回滚该 task）

1. **不改 R1/R2 已冻结 schema。** `EvaluationReport` 的 `report_kind` 是 code，新种类不是 schema 变更；五进五出、四种 posterior representation、`Refusal.grounds`、指纹七槽、失效矩阵一律不动。
2. **不改旧 API 数值。** `compile` / `sample` / `estimate` / `Posterior` / `fit` / `map_estimate` 与 R2 的 `_run_posterior` / `_run_predictive` 逐字节不变；Task 2 只在 `dispatch/task.py` 加 `_run_simulation` 一条分支。
3. **evaluation 不改 Result、不选算法、不重判。** 任何 verdict 只从 Result / diagnose 报告的字段算；`SiteDiagnostic`、`IdentifiabilityReport`、`PriorSensitivityReport` 的阈值住在原处。
4. **依赖单向。** `artifacts`、`graph`、`dispatch` 不 import `evaluation`；`evaluation` 不 import NumPyro 内部对象；`test_layering.py` 钉住。
5. **随机验收纪律（§9.3）。** 每条随机测试写死 seed、N、α；红了不换 seed，改的是实现或 fixture 并写下理由；误报率不合适 blocking CI 的降为 full 层。
6. **每个 task 顺序执行**，一个提交，红灯→实现→绿灯→lint；不 `git add -A`；只 stage 本 task 列出的文件。
7. **lint gate 是 `ruff check src/ tests/`**，不清扫 format drift。
8. **先测后写。** 新阈值只能是 §0 列出的 D104–D106，provenance 已定；执行中若冒出第四个数，先停下登记再写代码。
9. **一个决定一个家。** 本计划的裁决写在本计划；被执行推翻的，在本计划原行写回。

---

## 完成门槛：§8 R3 的 8 条 → 可跑、可数的 gate

| # | §8 R3 门槛 | 可数 gate |
|---|---|---|
| G1 | 已知校准的 synthetic fixtures 通过 | straight_line / radiometer / bilinear_pair(NUTS) 各一：PPC PASS、prior predictive PASS、held-out PASS、LOO PASS、SBC PASS（full）、identifiability PASS。可数：≥3 fixtures × ≥5 报告 |
| G2 | 已知错误或 misspecified fixtures 能失败 | curved_line(0.6) PPC FAIL 与 held-out FAIL；2 倍过宽 / 过窄 mutant SBC FAIL；collinear_pair identifiability FAIL；N(0,1e6) 先验 prior predictive FAIL；极紧先验 prior_sensitivity FAIL。可数：≥6 FAIL 路径 |
| G3 | 不足以判断的 fixtures 能 ABSTAIN | draws=8 PPC ABSTAIN；N=20 SBC ABSTAIN；refit 未收敛 ABSTAIN；arviz 未装 / float32 → UNVERIFIABLE；mask 全 True → INAPPLICABLE。可数：≥5 |
| G4 | SBC 同时覆盖 exact 与 sampled 路由 | exact（gcr）与 NUTS 各一 full 层用例，同一 harness、同一 rank 定义 |
| G5 | 随机校准验收固定 seed、预算、误报率 | close-out 表：每条随机测试的 seed / N / α / 期望误报；CI 无 rerun |
| G6 | 本地 reference NPE 退役规则 | Task 9 记录页：候选逐条过 §1.5 六门槛或「无候选通过」 |
| G7 | ArviZ round-trip 不丢 observation 或 chain 语义 | R2 的 export 测试继续绿 + Task 5 的 LOO 消费同一 export，plate 轴与 chain 轴在 `az.loo` 的 `n_data_points` / `n_samples` 上对得上 |
| G8 | report 不依赖人工阅读 sampler 日志 | 每个 verdict 由 finding 的 observed/expected 复算得出（Task 8 一条测试：从 report 字段重算 verdict） |

外加与 R1/R2 相同的四条共享 gate（source full suite、ruff、built-wheel suite、rheplicant consumer gate），全绿才写 `R3 closed`。

---

## D 编号与 probe 编号

- **probe：** probe_28 随本计划提交（`docs/probes/probe_28_model_checking_seams.py`）；Task 9 用 **probe_29**；后续独立实测接 probe_30+。
- **D 编号：** 现有最高 D103。R3 声明三个：**D104** `ALPHA = 0.05`（borrowed）、**D105** PPC draw 地板 `ceil(1/(α/2)) = 40`（derived，公式登记）、**D106** SBC 复制数地板 50（derived，Task 6.1 换 10 组 seed 重测后登记最差值）。**D107–D108 预留**，预计消费 0 个；Bonferroni 因子、PIT 与 KS 都是推导，不编号。magic 目标 0。

---

## R3 明确不做的事

- 不做模型比较 / `Compare` action（R7 的 Action registry），`loo_psis` 只报单模型的 elpd 与可靠性。
- 不做 evidence（`EvidenceTask` 继续 `capability_unavailable_r1`，R4）、nested sampling（R5）、BlackJAX。
- 不自实现 WAIC、R-hat、ESS、PSIS（§1.5）；不用 arviz 替换 `SiteDiagnostic`。
- 不做 correlated / 非 Gaussian 观测节点的 predictive、held-out 与 PPC（承 R2 §0.4，→ Refusal / UNVERIFIABLE，R4）。
- 不训练新的 NPE 架构；候选 SBI 上游只评估、不加依赖。
- 不让 LLM / agent 参与 verdict；aggregator 仍是确定性纯函数。
- 不收复 `bayesmith.evidence`。

---

## 完成定义

只有同时满足以下条件才可写「R3 closed」：

1. G1–G8 八个 gate 全部有本次实测证据，source / lint / wheel / consumer 四共享 gate 全绿；
2. `SimulationTask` 三种 source 真执行，POSTERIOR_RESULT 的观测抽样与 R2 predictive 逐位一致；
3. 七种报告各有 PASS、FAIL 与 ABSTAIN / UNVERIFIABLE 的至少一条测试路径，verdict 可从 finding 复算；
4. 随机验收全部固定 seed 与预算，期望误报数已算出并写在 close-out；
5. D104–D106 登记完毕，两侧格子与 tighten / loosen 变异都红过；
6. 本地 reference NPE 有 harness 给出的校准数，候选记录页结论只取两种之一；
7. 旧 API 数值、R1/R2 schema、依赖单向三者逐字节未动；
8. close-out 使用本次真实 SHA、JUnit 计数与 consumer revision，不复用 R2 的绿色结论。
