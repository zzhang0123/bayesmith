# R2 执行计划：完整 posterior 与 predictive seam

> **文档状态：`record`** · 已落地批次/审计/测量的历史记录，写作当天为真，非当前权威。索引见 docs/README.md。
>
> R2 已 closed，证据见 `../specs/2026-08-31-r2-close-out.md`；本文保留为 §0 九条冻结裁决（conditioning vs replicated、observation unit、ESTIMATOR 唯一 schema 例外等）的完整出处。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **本计划只写到计划为止，任何 task 都要等 owner 过目后再在单独的执行 session 里开跑。**

**Goal:** 把 PredictiveTask 从「schema 有、执行无」（R1 统一返回 `capability_unavailable_r1` 的 typed Refusal）变成「真能跑」，产出可独立持久化、可被多个 EvaluationReport 消费的 PredictiveResult；同时把 amortized posterior 明确编码为 heuristic 表示、统一多链 diagnostics、提供可选 ArviZ export——都在不改变旧 API 数值、不改动 R1 已冻结 schema（除被证明确实缺的字段）的前提下。

**Architecture:** 保持 R1 的依赖单向：`bayesmith.artifacts` 叶层不 import graph/JAX/Equinox/NumPyro；`bayesmith.dispatch.task` 仍是唯一把 runtime Graph 接到 artifact 的桥。新增的 predictive 生成原语放进 `bayesmith.dispatch.predictive`（它 import Graph/JAX，是桥的一部分，不是 artifacts 的一部分）。predictive 的前向生成**复用** `graph.evaluate` + `graph.evaluate.apply_probabilistic` + `exact.gaussian.observation_parts`，不另写一个平行的前向模型（不变量 1「一个事实只有一个声明位置」）。

**Tech Stack:** Python 3.11 frozen/slots dataclasses、`enum.StrEnum`、NumPy、标准库 `json`/`hashlib`/`base64`/`uuid`；JAX/Equinox/NumPyro 只出现在 dispatch/bridge 层；pytest、ruff；ArviZ 为 optional dependency（`importorskip`）。

**Spec:** [Bayesmith 顶层设计](../specs/2026-08-30-bayesmith-top-level-design.md) §1.5、§3.3、§7.3、§8 R2；[R1 close-out](../specs/2026-08-30-r1-close-out.md)；[R1 计划](../plans/2026-08-30-r1-task-artifact-provenance.md)（结构样板）。

---

## 0. 冻结裁决与执行边界

这些是 R2 的实现输入，不在执行过程中重新发明。每一条都写「为什么」，magic 数一个不留。

### 0.1 conditioning vs replicated：语义定在字段，不是 prose

**裁决：** PredictiveResult 里 `conditioned_sites` 是 conditioning（observed data 属于它）；`replicated_draws` 是新的随机结果。二者在**生成律**上分开，不许用一个数组装两个意思。

**为什么：** §3.3 明写「observed data 属于 conditioning；replicated data 是新的随机结果」。把观测值回放进模型（observed-data replay）得到的是 pointwise log-likelihood（conditioning 那一半），不是 posterior predictive。R1 已把 `PredictiveResult.conditioning_data`/`prediction_design`/`conditioned_sites`/`replicated_draws`/`pointwise_log_density` 字段冻结，R2 只接执行、不改语义。§8 R2 门槛 2 就是这条的反面：replay 不得被误当 predictive，测试见 G2。

**接缝（实测，见 §0.10）：** `observation_parts(graph, evaluate(graph, values))` 对每个 observed node 给出 `(data, loc, scale)`。`loc` 是该 latent 取值下的预测均值，`scale` 是噪声尺度。replay = `dist.Normal(loc, scale).log_prob(data)`；replicate = `dist.Normal(loc, scale).sample(key)`。两者用同一个 loc/scale，只有 log_prob vs sample 之别。

### 0.2 pointwise log-likelihood 的 observation unit

**裁决：** pointwise log-likelihood 的 observation unit 是**observed node × plate 元素**；plated node 的每个 plate 元素是一个独立 unit。`PosteriorResult.pointwise_log_likelihood`（`NamedArray`）与 `PredictiveResult.pointwise_log_density` 的 leading 轴恒为 draw，其余轴是 observation unit；`observation_unit` 记录「node 或 node/plate 元素」的声明，`grouping` 记录 plate/group 名。`log_density_availability == POINTWISE ⇔ pointwise_log_likelihood is not None`（这条双向断言 R1 已冻结，R2 只是让它从恒 None 变真）。

**为什么：** §3.3 明写「plate 和 grouping 语义必须进入定义，不能事后猜测」；§8 R2 门槛 5 要「observation unit 可审计」。dim 轴名来自 `dispatch.task._dims`（draw 轴、plate 轴、`{name}_dim{i}` 兜底），是 R1 已冻结的轴命名规则，R2 复用而不另造。对非 plate 的 flat graph，一个 node 就是一个 scalar unit。

### 0.3 生成律：replicated draws 走 observation 分布，不手写 simulator

**裁决：** replicated draw 通过对 observed node 调 `apply_probabilistic(graph, node, env)` 得到的分布 `.sample(key)` 生成（对 diagonal-Gaussian 的 exact 路由等价于 `Normal(loc, scale).sample`）。**不**把 `dist_fn` 之外的某个手写「simulator」塞进来，也**不**复用 amortize.py 里 D10(2)/D42 那条「simulator 不在这里」的银行生成律。

**为什么：** D10(2)/D42 的 ruling 是「生成 `(theta, x)` 联合对（simulation bank）的生成律 ≠ 密度律，对 multiplicative instrument 差一个 |value|+floor」。但 R2 的 replicated draw 生成的是 **observed node 自己的观测值**，其生成律就是该 node 的 distribution（`log_joint` 对同一 node 读的就是 `dist.log_prob`，所以 density 与 generator 在这里是同一个对象）。对 Normal/Poisson 这类 node 这是唯一正确的定义。correlated-noise node（`gaussian_parts` 会拒绝）不在 R2 覆盖域，见 0.4。

### 0.4 R2 的 predictive 覆盖域与拒绝边界

**裁决：** R2 第一遍只覆盖 **diagonal-Gaussian observed nodes** 的 predictive（exact 路由的域）+ **NUTS/混合路由采样出的 latent**。correlated-noise observed node、非 Gaussian observed node 的 predictive → 返回 typed Refusal（`failed_premise` 固定为 `predictive_noise_unsupported`，非空 grounds/remedies），**不**静默退回某个近似。

**为什么：** `observation_parts`/`gaussian_parts` 是 diagonal walk，对 correlated node 会 raise，这正好是「缺信息就拒绝」的自然边界；correlated 的 predictive 需要 precision-operator 上的前向采样，属于 R4 evidence 那侧的能力，不在 R2 抢跑。§8 R2 门槛 6 要「缺信息时 Refusal/ABSTAIN」，这条给出第一个缺信息场景。

### 0.5 加权 posterior → predictive 的规则

**裁决：** 每个源 posterior draw 产出一个 replicated draw（draw 轴一一对应，不重采样、不按权重丢弃）。源 posterior 若是 `WeightedDrawsPosterior`，其权重、Kish ESS、khat、unreliable 留在源 `PosteriorResult.representation` 里，由 `source_posterior_ref` 的 lineage 可读，**不**在 PredictiveResult 里复制第二份权重。

**为什么：** `PredictiveResult` 的 schema（R1 冻结）没有权重字段，只有共享 draw 轴的 `latent_draws`/`replicated_draws`/`pointwise_log_density`。一个 predictive 结果不自行「决定」该信哪些 draw——那是源 posterior 的 verdict（`unreliable`、`ess`）和 R3 的 PPC/LOO 的事。复制权重到 predictive 里会造出两个家（不变量 1）。

### 0.6 源 posterior 兼容性检查（§3.1）

**裁决：** execute_task 解析 `source_posterior_ref` 后，必须核对其 data/graph/model 指纹与本次 PredictiveTask 的图、`conditioning_data` 指纹一致；不一致返回 typed Refusal（`failed_premise="posterior_data_mismatch"`），**不**拿一个来自不同数据条件的 posterior 做 predictive。

**为什么：** §3.1「复用必须明确证明兼容，例如 posterior draws 是否来自与 PredictiveTask 相同的数据条件和参数化」。这是 fingerprint 语义相等的检查（不是数值阈值），没有 magic number。

### 0.7 amortized posterior → FittedConditionalPosterior（含唯一的 schema 例外）

**裁决：** `NeuralPosterior`（eqx.Module + static callable + standardization arrays）**不得**以 callable/对象形式进 artifact。它编码成 `FittedConditionalPosterior` 的引用三件套：

- `simulation_bank_ref` → 一个 `SimulationResult`：银行就是 prior simulation，`latent_draws=thetas`、`observation_draws=data`、`parameter_source=PRIOR`。**零 schema 变更**。
- `training_run_id` → 真实训练运行的 `run_id`；`validation_report_refs` → 训练 validation 曲线的 `EvaluationReport`。**零 schema 变更**。
- `estimator_ref` → **R1 冻结的 `_result_ref` 要求它指向 `ArtifactKind.RESULT`，但五种 Result 没有一种装得下「训练好的估计器权重」——这是 R2 证明确实缺的那个字段。** R2 新增 `ArtifactKind.ESTIMATOR`（失效矩阵加一行：对 model/graph/data/task 敏感，与 RESULT 同），把 `FittedConditionalPosterior.estimator_ref` 的校验从 `_result_ref` 放宽为指向 `ArtifactKind.ESTIMATOR` 的引用。估计器 artifact 是 canonical JSON envelope（同 `ArtifactFile`）包一个 equinox leaf-serialization blob + 一份 static manifest（embed 的 callable identity、n_components、n_params、min_scale、standardization 数组）；blob 对 codec 是不透明 bytes，`artifacts` 层依旧不 import JAX/equinox（serialize/deserialize 在 `dispatch.amortized` 桥里做）。

**为什么：** §8 R2 主要工作明写「将现有 amortized posterior 明确编码为 heuristic PosteriorResult representation，记录 training-bank、validation、backend 和 calibration provenance」，§14 决策 10「amortized inference 是 heuristic PosteriorResult representation」。银行/训练 lineage 已有现成 schema 家；唯独「估计器权重」没有。这是红线「除非某字段被证明确实缺」的唯一一次启用，且不动五进五出的 bijection（`ResultKind` 仍五个，`ArtifactKind` 才是三个变四个——失效矩阵的 taxonomy，不是结果种类）。

### 0.8 ArviZ 是 optional、export-only，不是 core

**裁决：** ArviZ 是 **optional dependency**，core 不 import 它；R2 只做 **export seam**（`PosteriorResult`/`PredictiveResult` + 多链 diagnostics → `arviz.InferenceData`，round-trip 不丢 observation unit 与 chain 语义），**不做** LOO/WAIC 计算——那是 R3。测试用 `importorskip`。

**为什么：** §3.3「第一阶段可以通过可选 ArviZ bridge 获得成熟的 LOO、WAIC 和标准可视化生态，但 Graph、observation grouping、适用性判断和最终 artifact 仍由 bayesmith 定义」；§7.3 backend policy 九个问题（optional 缺失优雅退化、backend 对象不泄漏进公共 API、升级可检测语义漂移）都要照答。R3 门槛「ArviZ round-trip 不丢失 observation 或 chain 语义」依赖 R2 先把 export 做对；把 LOO/WAIC 拉进 R2 等于抢 R3 的 ownership。

### 0.9 多链 diagnostics 统一成一个 EvaluationReport

**裁决：** 统一的多链 diagnostics = 现有 `dispatch.execute.chain_diagnostics` 的 `SiteDiagnostic`（`r_hat`/`ess`/`ceiling`/`converged`/`worst`/`reason`，worst-coordinate 归因），逐 site 投影为 `EvaluationReport` 的 `Finding`，由 `PosteriorResult.report_refs` 引用。**不**给 `PosteriorResult` 加新字段。

**为什么：** R1 的 `_run_posterior` 把 `posterior.diagnostics` 只折成 `RunWarning` prose（`chain_not_converged`），丢掉了 per-site r_hat/ESS/worst 归因。§8 R2 主要工作「统一多链 sampling diagnostics」。EvaluationReport 是 R1 已冻结的 artifact，引用它不碰 schema；worst-coordinate 归因是 `SiteDiagnostic` 已经算好的，R2 只做「结构化投影 + 序列化」，不重新判收敛（不变量：verdict 属于算它的那一层）。

### 0.10 现状实测（写进计划的数字都来自本 checkout）

命令：`.venv/bin/python /tmp/probe_r2.py`（本计划写作当天；执行期把它固化为 `docs/probes/probe_27_predictive_seam.py`）。源码现状（读过，不是背）：

- **`amortize.py`**：`NeuralPosterior`（conditional Gaussian mixture，MLP + Adam 手写，无 optax），`create`/`log_prob`/`sample`/`train_posterior`，`MIN_SCALE=1e-3`、`n_components=4`、`width=64`、`depth=3`、`n_steps=3000`、`batch_size=256`、`learning_rate=1e-3`、`validation_fraction=0.1`——这些是已冻结的默认，R2 只把它们记进 `FittedConditionalPosterior` 的 training lineage，不重造。`TrainingHistory` 有 `train`/`validation`/`best_step`。
- **`optimize.py`**：`fit`（graph 入口，全密度 joint MAP）与 `minimize`（adam/gradient），无收敛 verdict（`steps` 步照跑），`Fit(values, objective, history)`。R2 不动它；它是 PointEstimate 路径，R1 已接。
- **`exact`/`dispatch.execute` 路由**：`Posterior.method` ∈ `{"gcr","gcr+snis","gcr+mh","nuts","collapse"}`；`Posterior.samples` 是 `{latent: draws}`（draw 轴 leading）；`chain_diagnostics` 产出 `SiteDiagnostic`；`observation_parts`/`noise_std_at` 在 `exact.gaussian`。

实测数字（radiometer，prediction-dependent sigma，gcr+snis，2000 draws，key 0/1）：

| 量 | 值 |
|---|---|
| `execute_task` 结果类型 | `PosteriorResult`，method `gcr+snis` |
| `predictive_ready` / `log_density_availability` / `pointwise_log_likelihood` | `False` / `none` / `None`（R2 要接的缝） |
| `compile_task(PredictiveTask)` | `Refusal`，`failed_premise=capability_unavailable_r1`（确认现状） |
| observed-data replay mean log-lik（10 obs 求和） | **-3.8373** |
| replicated draw grand mean d / observed grand mean d | **8.8379 / 8.8621**（均值接近但逐元素不同——replay≠replicate 的数值形态） |
| posterior mean w / true w | **2.9483 / 3.0** |
| `observation_parts` 在 posterior mean 处 | loc `(10,)`、scale `(10,)`、scale mean **0.4432** |

这些数字是 §0.1/§0.3/§0.4 的地基：loc/scale 就是 predictive 接缝，replay 的 -3.8373 与 replicated 的逐元素差异就是 G2/G4 的 oracle。

---

## Task 1：predictive seam 原语与独立 oracle（probe_27）

**Files:**

- Create: `docs/probes/probe_27_predictive_seam.py`
- Create: `tests/dispatch/test_predictive_seam.py`

- [ ] **1.1 先写 probe 脚本，把 §0.10 的实测固化**：对 radiometer（prediction-dependent）与 straight_line（constant sigma）分别打印 `observation_parts` 的 loc/scale、replay 的 pointwise log-lik 均值、replicated draw 的逐元素值与 grand mean，断言 replicated 的每个元素与 observed 对应元素不同、replay 均值等于 §0.10 记录的数。
- [ ] **1.2 写红灯测试** `tests/dispatch/test_predictive_seam.py`：`observation_parts` 返回 `(data, loc, scale)` 三 dict 且 leaf 对齐；replay≠replicate；correlated node（若 fixture 有）raise 而非静默。运行 `.venv/bin/python -m pytest tests/dispatch/test_predictive_seam.py`，预期 FAIL（实现原语尚不存在）。
- [ ] **1.3 实现**：若 `observation_parts`/`evaluate` 已足够（源码已确认），本 task 主要落 probe + 测试，不动 src；若发现缺一个「在 latent draws 上 vmap 生成 replicated」的小 helper，放进 `dispatch/predictive.py`（见 Task 2），本 task 只 import 它。
- [ ] **1.4 绿灯**：`.venv/bin/python -m pytest tests/dispatch/test_predictive_seam.py` PASS；`.venv/bin/ruff check tests/dispatch/test_predictive_seam.py docs/probes/probe_27_predictive_seam.py` 干净。
- [ ] **1.5 提交**：`git add docs/probes/probe_27_predictive_seam.py tests/dispatch/test_predictive_seam.py`，commit `"probe: fix the predictive seam oracle (probe 27)"`。

---

## Task 2：`dispatch.predictive` 前向生成原语

**Files:**

- Create: `src/bayesmith/dispatch/predictive.py`
- Modify: `tests/dispatch/test_predictive_seam.py`

- [ ] **2.1 红灯**：写测试钉住两个原语——`replicated_draws(graph, latent_draws, key) -> {obs: draws}`（对每个 observed node 用 `apply_probabilistic(...).sample`，vmap 过 draw 轴）与 `pointwise_log_likelihood(graph, latent_draws) -> NamedArray`（`apply_probabilistic(...).log_prob(observed)`，mask 位归零，observation unit 轴名来自 `_dims` 规则）；correlated node → 明确异常。预期 FAIL。
- [ ] **2.2 实现**：`predictive.py` 只依赖 `graph.evaluate`/`graph.nodes`/`exact.gaussian.observation_parts`，不 import artifacts（保持桥内依赖单向）；diagonal-Gaussian 用 loc/scale，其它 Gaussian 类 node 用 `apply_probabilistic` 的 distribution；`observation_unit` 轴名与 `grouping` 规则照 §0.2。
- [ ] **2.3 绿灯**：`.venv/bin/python -m pytest tests/dispatch/test_predictive_seam.py` PASS；`.venv/bin/ruff check src/bayesmith/dispatch/predictive.py tests/dispatch/test_predictive_seam.py` 干净。
- [ ] **2.4 提交**：`git add src/bayesmith/dispatch/predictive.py tests/dispatch/test_predictive_seam.py`，commit `"feat: add the predictive forward-generation primitive"`。

---

## Task 3：PredictiveTask 接线 compile/execute

**Files:**

- Modify: `src/bayesmith/dispatch/task.py`
- Modify: `tests/dispatch/test_task_protocol.py`
- Create: `tests/dispatch/test_task_execution.py`（若 R1 未含 predictive 用例则新建，否则 extend）

- [ ] **3.1 红灯**：写测试——`compile_task(predictive_task)` 返回 `PlannedTask`（不再是 `capability_unavailable_r1`）；`execute_task` 对合法源 posterior 返回 `PredictiveResult`，`latent_draws`/`replicated_draws` 共享 draw 轴、`source_posterior_ref` 精确指向源 id/revision；源 posterior 指纹不符 → `Refusal(posterior_data_mismatch)`；correlated node → `Refusal(predictive_noise_unsupported)`。预期 FAIL。
- [ ] **3.2 实现**：把 `TaskKind.PREDICTIVE` 加进 `SUPPORTED_TASK_KINDS`；`compile_task` 对 PredictiveTask 走正常计划（图要编译，用于前向评估）；`execute_task` 解析 `source_posterior_ref` → 加载源 `PosteriorResult` → §0.6 兼容检查 → 调 `dispatch.predictive` 两原语 → 组装 `PredictiveResult`。`run` 记录 backend（`bayesmith`，或 NUTS 时 `numpyro`）、seed、budget、termination、timing、approximation（`MONTE_CARLO/EXACT` 或按源表示类）。
- [ ] **3.3 绿灯**：`.venv/bin/python -m pytest tests/dispatch/test_task_execution.py tests/dispatch/test_task_protocol.py` PASS；`.venv/bin/ruff check src/bayesmith/dispatch/task.py tests/dispatch` 干净。旧测试 `test_the_three_tasks_r1_cannot_answer_are_refused_as_a_capability` 的 predictive 分支**要改成**断言真执行，仅 evidence/simulation 保留 capability refusal——这是刻意的测试变更，不是绕测试。
- [ ] **3.4 提交**：`git add src/bayesmith/dispatch/task.py tests/dispatch/test_task_protocol.py tests/dispatch/test_task_execution.py`，commit `"feat: execute predictive tasks into typed predictive results"`。

---

## Task 4：PosteriorResult 的 pointwise LL 与 predictive_ready

**Files:**

- Modify: `src/bayesmith/dispatch/task.py`
- Modify: `tests/dispatch/test_task_execution.py`

- [ ] **4.1 红灯**：测试——`_run_posterior` 对 exact diagonal 路由把 `pointwise_log_likelihood` 填成 `NamedArray`（dims 含 draw + observation unit 轴）、`log_density_availability=POINTWISE`、`predictive_ready=True`；对纯 NUTS 路由按 §0.2 填 POINTWISE（NUTS 也能算 pointwise LL）；对 correlated/非 Gaussian 保持 `NONE`/`None` 且 `predictive_ready=False`。预期 FAIL。
- [ ] **4.2 实现**：`_run_posterior` 在拿到 `posterior.samples` 后调 `dispatch.predictive.pointwise_log_likelihood`；只有计算成功且 observation unit 明确时才置 POINTWISE/True，否则保持 NONE/None（ABSTAIN，不伪造）。
- [ ] **4.3 绿灯**：`.venv/bin/python -m pytest tests/dispatch/test_task_execution.py tests/artifacts` PASS（`test_tasks_results.py` 里 `log_density_availability` 的双向断言现在覆盖真值）；`.venv/bin/ruff check src/bayesmith/dispatch/task.py tests/dispatch/test_task_execution.py` 干净。
- [ ] **4.4 提交**：`git add src/bayesmith/dispatch/task.py tests/dispatch/test_task_execution.py`，commit `"feat: record pointwise log-likelihood and predictive readiness"`。

---

## Task 5：多链 diagnostics → EvaluationReport

**Files:**

- Modify: `src/bayesmith/dispatch/task.py`
- Create: `tests/dispatch/test_chain_diagnostics_report.py`

- [ ] **5.1 红灯**：测试——NUTS/混合/collapse 路由的 `posterior.diagnostics`（`SiteDiagnostic`）逐 site 投影成 `EvaluationReport`（`Finding` 带 `r_hat`/`ess`/`ceiling`/`converged`/`worst`/`reason`），被 `PosteriorResult.report_refs` 引用；round-trip 后 worst-coordinate 归因不丢；gcr/gcr+snis iid 路由 diagnostics 为 `None` → 不产生 report（r-hat 无 referent）。预期 FAIL。
- [ ] **5.2 实现**：复用 `chain_diagnostics` 已算的 verdict，只做结构化投影 + `EvaluationReport` 组装；不重新判收敛；`applicability`/`conclusion` 用 R1 已冻结的合法 pair（APPLICABLE×PASS/FAIL、UNVERIFIABLE×ABSTAIN）。
- [ ] **5.3 绿灯**：`.venv/bin/python -m pytest tests/dispatch/test_chain_diagnostics_report.py tests/artifacts/test_gates.py` PASS；`.venv/bin/ruff check src/bayesmith/dispatch/task.py tests/dispatch/test_chain_diagnostics_report.py` 干净。
- [ ] **5.4 提交**：`git add src/bayesmith/dispatch/task.py tests/dispatch/test_chain_diagnostics_report.py`，commit `"feat: record multi-chain diagnostics as evaluation reports"`。

---

## Task 6：ArviZ optional export

**Files:**

- Create: `src/bayesmith/bridge/arviz.py`
- Create: `tests/bridge/test_arviz_export.py`（`importorskip("arviz")`）

- [ ] **6.1 红灯**：测试——`PosteriorResult`/`PredictiveResult` + 多链 diagnostics → `arviz.InferenceData`，round-trip 后 observation unit（plate 轴/grouping）与 chain 语义（draw/chains 轴）不丢；ArviZ 未安装时模块 import 不崩、测试 skip。预期 FAIL。
- [ ] **6.2 实现**：薄 export 层，把 `NamedArray` 映射成 `posterior`/`posterior_predictive`/`log_likelihood` group，`observed_data` 从 `observation_parts` 的 data 取。core 不 import arviz；`bridge/arviz.py` 里才 import。
- [ ] **6.3 绿灯**：`.venv/bin/python -m pytest tests/bridge/test_arviz_export.py` PASS（有 arviz 时）/ skip（无时，声明预期）；`.venv/bin/ruff check src/bayesmith/bridge/arviz.py tests/bridge/test_arviz_export.py` 干净。
- [ ] **6.4 提交**：`git add src/bayesmith/bridge/arviz.py tests/bridge/test_arviz_export.py`，commit `"feat: add optional ArviZ export for predictive results"`。

---

## Task 7：amortized posterior → FittedConditionalPosterior

**Files:**

- Modify: `src/bayesmith/artifacts/identity.py`（`ArtifactKind.ESTIMATOR` + 失效矩阵行）
- Modify: `src/bayesmith/artifacts/results.py`（`estimator_ref` 校验放宽）
- Create: `src/bayesmith/dispatch/amortized.py`（equinox serialize/deserialize 桥）
- Create: `tests/dispatch/test_amortized_encoding.py`

- [ ] **7.1 红灯**：测试——`NeuralPosterior` 训练后编码成 `FittedConditionalPosterior`（`simulation_bank_ref` 是 `SimulationResult`、`training_run_id`/`validation_report_refs` 真实、`estimator_ref` 指向 `ArtifactKind.ESTIMATOR` 且 round-trip 后 `sample` 数值一致）；callable/模块对象无法进入 artifact（codec 拒绝）；`ArtifactKind.ESTIMATOR` 的失效矩阵行按 model/graph/data/task 失效。预期 FAIL。
- [ ] **7.2 实现**：`identity.py` 加 `ArtifactKind.ESTIMATOR` 并补 `InvalidationPolicy` 行（同 RESULT 敏感集）；`results.py` 把 `estimator_ref` 的校验改成「ArtifactKind.ESTIMATOR」；`dispatch/amortized.py` 用 `eqx.tree_serialise_leaves` 序列化参数叶子、static manifest 记录 embed identity/n_components/n_params/min_scale/standardization，写进 `ArtifactFile` 式 envelope；deserialize 还原 `NeuralPosterior`。`artifacts` 层只见 opaque bytes，不 import JAX/equinox。
- [ ] **7.3 绿灯**：`.venv/bin/python -m pytest tests/dispatch/test_amortized_encoding.py tests/artifacts tests/test_layering.py` PASS（layering 断言 `artifacts` 仍不 import graph/JAX）；`.venv/bin/ruff check src/bayesmith/artifacts src/bayesmith/dispatch/amortized.py tests/dispatch/test_amortized_encoding.py` 干净。
- [ ] **7.4 提交**：`git add src/bayesmith/artifacts/identity.py src/bayesmith/artifacts/results.py src/bayesmith/dispatch/amortized.py tests/dispatch/test_amortized_encoding.py`，commit `"feat: encode the amortized posterior as a fitted-conditional reference"`。

---

## Task 8：R2 完成门槛 close-out

**Files:**

- Create: `docs/superpowers/specs/2026-08-31-r2-close-out.md`
- Modify only if a measured failure requires it: files owned by Tasks 1–7

- [ ] **8.1 建立验收矩阵**（§9 门槛 + 下面 6 个 gate），先列空 measured cell，再逐项实测填入。
- [ ] **8.2 跑 source full suite**：`.venv/bin/python -m pytest -n 4 --junit-xml=runs/<run>/junit.xml`，独立 exit code 与 JUnit 计数。
- [ ] **8.3 跑 lint**：`.venv/bin/ruff check src/ tests/`，不跑 format sweep。
- [ ] **8.4 跑 built-wheel suite 与 rheplicant consumer gate**（照 R1 §9.4/§9.5 命令，wheel 里 sibling 未装 → crosscheck 预期 skip）。
- [ ] **8.5 用本次真实 SHA/JUnit/consumer revision 完成 close-out**，写 `R2 closed` 或 `R2 open` + blocker。
- [ ] **8.6 提交**：`git add docs/superpowers/specs/2026-08-31-r2-close-out.md`，commit `"docs: close the R2 predictive seam"`。

---

## 红线（每个 task 提交前自查，violate 即回滚该 task）

1. **不改 R1 已冻结 schema，除 §0.7 那一个被证明确实缺的字段**（`ArtifactKind.ESTIMATOR` + `estimator_ref` 校验）。五进五出 bijection（`PRIMARY_RESULT_BY_TASK`）、五类 Result、四种 posterior representation、`Refusal.grounds`、fingerprint 七槽、失效矩阵既有行，一律不动。
2. **不改旧 API 数值。** `compile`/`sample`/`estimate`/`Posterior`/`Estimate`/`fit`/`map_estimate` 逐字节不变；R2 只在 `dispatch/task.py` 与新建的 `dispatch/predictive.py`/`dispatch/amortized.py`/`bridge/arviz.py` 上加适配层，不重写数值内核。
3. **`artifacts` 层依旧不 import graph/JAX/Equinox/NumPyro。** 新桥的 equinox serialize 只发生在 `dispatch.amortized`；`tests/test_layering.py` 的 subprocess 断言继续绿。
4. **每个 task 顺序执行**，一个提交，红灯→实现→绿灯→lint；不并行，不 `git add -A`，只 stage 本 task 列出的文件。
5. **lint gate 是 `ruff check src/ tests/`**，不是 `ruff format --check`；不清扫 31 个文件的 format drift。
6. **先测后写**：任何新「成本/边界/门槛」数字都来自本 checkout 实测，写清命令与结果；拿不准写「待测」，不编。
7. **一个决定一个家**：本计划的裁决写在本计划，不在顶层设计或别的 spec 另开 roadmap。

---

## 完成门槛：§8 R2 的 6 条 → 可跑、可数的 gate

（照 R1 close-out 验收矩阵：每条门槛一行，写清「要求」与「measured 形式」。）

| # | §8 R2 门槛 | 可数 gate |
|---|---|---|
| G1 | exact、factorized 和 NUTS 路径可进入同一 predictive API | 一个参数化测试集：gcr / gcr+snis / gcr+mh / collapse / 纯 NUTS / 混合(factorized) 六条路由 × 同一 PredictiveTask，每条断言返回 typed `PredictiveResult`（非 Refusal），`latent_draws`/`replicated_draws` 共享 draw 轴。可数：≥6 路由 × 1 断言 |
| G2 | observed-data replay 不会被误当 posterior predictive | 表驱动测试：每 fixture 同时算 replay（`log_prob(observed)` 求和）与 replicate（`sample`），断言 replicated 逐元素 ≠ observed 对应元素，且 replay 均值 = probe_27 实测值（radiometer **-3.8373** 为锚）。可数：≥3 fixtures × 双向断言 |
| G3 | PredictiveResult 可独立持久化并被多个 EvaluationReport 消费 | `dump_artifact`/`load_artifact` byte-stable round-trip（字段逐项）；两个不同 `EvaluationReport` 引用同一 `PredictiveResult` 后各自 `aggregate_gate` 不互相覆盖。可数：round-trip 全字段 + 2 reports |
| G4 | analytic posterior 与 predictive moments 有独立 oracle | linear-Gaussian（straight_line 等）闭式 posterior mean/sd 与 replicated 的 loc/scale 手算（不调 runtime），`assert_allclose` 对比 runtime 值。可数：≥2 fixtures × moments 闭式 vs runtime |
| G5 | pointwise likelihood 的 observation unit 可审计 | 逐 fixture 断言 `pointwise_log_likelihood` 的 `NamedArray.dims`（flat 图 = draw 轴；plated 图含 plate 轴）与 `observation_unit`/`grouping` 字符串一致；`log_density_availability==POINTWISE ⇔ pointwise is not None`（R1 已冻结的双向断言复用）。可数：≥3 fixtures（flat/plated/grouped）|
| G6 | 缺少 predictive 所需信息时返回 Refusal 或 ABSTAIN | ≥3 个缺信息场景各一断言：源 posterior 指纹不符 → `Refusal(posterior_data_mismatch)`；correlated/noise 不支持 → `Refusal(predictive_noise_unsupported)`；源 posterior 无 pointwise 能力 → ABSTAIN 的 `EvaluationReport`（不伪造）。可数：≥3 场景 × 1 断言 |

外加与 R1 相同的四条共享 gate（source full suite、ruff、built-wheel suite、rheplicant consumer gate），全绿才写 `R2 closed`。

---

## D 编号与 probe 编号（预留）

- **probe：** 下一个是 **probe_27**（Task 1 的 `docs/probes/probe_27_predictive_seam.py`）。后续若拆出独立实测，接 probe_28+。
- **D 编号：** 现有最高是 D103（P7）。R2 **预期不引入数值阈值**（predictive 的 compat check 是 fingerprint 语义相等，不是数值门槛；replay≠replicate 是结构性断言；pointwise unit 是轴名审计，均无 magic number）。若执行中真的冒出数值门槛，从 **D104** 起登记进 `tests/numerical_gates/registry.py`，provenance 必须 `derived`/`exact_or_domain`，magic 目标 0；计划层面先声明「D104–D106 预留给 R2，预计消费 0 个」。

---

## R2 明确不做的事

- 不做 R3 的 PPC / held-out / SBC / LOO / WAIC / calibration（那些是引用 PredictiveResult 的 EvaluationReport，R3 才建）；R2 只把 PredictiveResult 造出来并让它可以被消费。
- 不做 R4 的 evidence（`EvidenceTask` 继续 `capability_unavailable_r1`）、R5 的 nested sampling。
- 不做 correlated-noise / 非 Gaussian observed node 的 predictive 生成（→ Refusal，见 §0.4）。
- 不做 `SimulationTask` 的执行（R1 状态保留；它只是 PredictiveTask 的底层 primitive 边界，R2 不抢）。
- 不扩张本地 NPE/flow 算法 zoo（§14 决策 10）：amortized 只做「现有 NeuralPosterior 的 reference 编码」，不训练新架构。
- 不让 LLM/agent 参与 gate truth value；aggregator 仍是确定性纯函数。
- 不收复 deprecated `bayesmith.evidence` namespace。

---

## 完成定义

只有同时满足以下条件才可写「R2 closed」：

1. G1–G6 六个 gate 全部有本次实测证据（source/lint/wheel/consumer 四个共享 gate 也全绿）；
2. PredictiveTask 真执行、PredictiveResult 可独立持久化并被多个 EvaluationReport 消费；
3. observed-data replay 与 posterior predictive 在测试中被数值性地区分（probe_27 为 oracle）；
4. pointwise log-likelihood 的 observation unit 逐 fixture 审计通过；
5. 缺信息场景（指纹不符 / noise 不支持 / 无 pointwise 能力）逐一返回 typed Refusal 或 ABSTAIN；
6. amortized posterior 编码为 `FittedConditionalPosterior` 且 `artifacts` 依旧不 import JAX；
7. 旧 API 数值、R1 schema（除 §0.7 唯一例外）、`artifacts` 依赖单向，三者逐字节未动；
8. close-out 使用本次真实 SHA、JUnit 计数与 consumer revision，不复用 R1 的绿色结论。
