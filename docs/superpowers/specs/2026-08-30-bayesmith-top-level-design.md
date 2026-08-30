# bayesmith 顶层设计：从结构化推断到可审计的 Bayesian workflow

**状态：** 面向未来的规范性设计  
**日期：** 2026-08-30  
**采用方案：** A — compiler-first  
**适用范围：** bayesmith 的长期产品定位、公共概念模型、能力边界和建设顺序

## 0. 文档地位

本文是 bayesmith 面向未来的顶层设计和主路线图。已有的专题设计、阶段计划和实现记录仍然保留其历史价值，但如果它们与本文在产品定位、分层、公共概念或建设顺序上冲突，以本文为准。

本文只规定：

- bayesmith 要成为什么；
- 哪些语义由它拥有；
- 各层之间如何传递可审计的 artifact；
- posterior、evidence、model checking 和未来 agent workflow 如何形成一个整体；
- 各阶段完成的客观门槛。

本文不规定具体版本号、发布日期或所有 Python 模块名。每个实现阶段仍应有独立、较短的执行计划；执行计划不得成为第二份长期 roadmap。

---

## 1. 北极星、边界与原则

### 1.1 北极星

bayesmith 的目标是：

> 将显式 Bayesian graph 编译为一条可审计的推断与检验过程：先识别并验证结构，精确消元可解析部分，为剩余问题选择执行后端，再产出带适用条件、诊断和 provenance 的结果。

它不是另一个 probabilistic programming language，也不是 sampler 的集合。它是位于模型声明与数值后端之间的 **graph-aware、task-aware Bayesian compiler and workflow layer**。

“利用结构”本身不是足够精确的差异化表述。BlackJAX 等 inference library 已经提供 latent-Gaussian、Laplace-marginalized 等利用特定结构的算法；bayesmith 的独特职责是从完整 Graph 中发现并验证结构，跨 subgraph 组合 exact 与 numerical routes，并说明为什么该组合对当前 Task 合法。

这里的“通用 Bayesian 工具”指 workflow 的通用性，而不是声称实现所有分布、所有 sampler、所有图结构或所有可视化。对任何处于支持域内的 Graph，bayesmith 应逐步能够回答五个问题：

1. 这个模型实际声明的联合分布是什么？
2. 数据能够识别哪些参数、方向或组合？
3. 哪些部分可以精确处理，剩余部分应走什么推断路径，为什么？
4. posterior、evidence 或 predictive 结论是否可信，适用条件是什么？
5. 当前结果之后有哪些合法且有信息增益的下一步？

### 1.2 bayesmith 拥有的语义

bayesmith 应拥有：

- Graph 及其结构语义；
- 结构检查、精确消元和 residual problem 编译；
- task-aware 的 InferencePlan，以及选择或拒绝某条路径的原因；
- 跨 exact、sampling、optimization 和外部 backend 的统一结果协议；
- applicability、refusal、diagnostic、provenance 和 quality gate；
- 从声明、分析、推断、检验到下一步决策的 epistemic state transition。

最简洁的边界表述是：

> bayesmith owns epistemic state transitions, not conversational agency.

也就是说，bayesmith 负责定义“基于哪些证据，可以从什么状态进入什么状态”；未来的 LLM agent 只负责在合法动作中选择、解释和编排。

### 1.3 明确不做什么

bayesmith 不应：

- 重新实现一个完整 PPL、distribution library 或通用 sampler zoo；
- 取代 NumPyro、BlackJAX、JAXNS、ArviZ 等成熟项目已有且维护良好的数值能力；
- 变成通用 DAG scheduler、任务队列、聊天框架或 agent framework；
- 允许 LLM 改写统计真值、降低诊断标准或绕过 quality gate；
- 仅为了接更多 backend 而引入缺乏统计语义的抽象层；
- 默认自动修改用户的科学模型。

### 1.4 十条不变量

1. **一个事实只有一个声明位置。** Graph 是模型真值的唯一来源。
2. **先 compile，后 execute。** 优化必须验证前提；不满足时应拒绝或降级，不能静默套用。
3. **严格区分统计对象。** prior、likelihood、marginal likelihood term、evidence 和 posterior 不得混名。
4. **结论总有适用性。** 缺少必要诊断时应 ABSTAIN，而不是默认为可信。
5. **近似级别必须可见。** exact、certified deterministic、Monte Carlo 和 heuristic 不能混成同一种结果。
6. **运行条件必须可追溯。** seed、dtype、device、backend、版本、数据身份和预算属于结果的一部分。
7. **外部 backend 不定义核心 API。** adapter 服从 bayesmith 的任务和结果协议。
8. **agent 不能绕过 gate。** 它只能选择当前状态公开的合法动作。
9. **失败是一等 artifact。** Refusal 应说明证据、失败前提和可采取的修复，不只是异常文本。
10. **先消元再蛮力。** 能可靠识别并精确消除的结构，优先于通用高成本算法。

---

## 2. 五层架构

### 2.1 Layer 1 — Model Graph

Graph 是模型的唯一语义来源。它表达：

- 随机变量和确定性节点；
- 节点角色、support、shape、plate 和依赖关系；
- prior、likelihood 与 graph-level terms；
- 数据绑定与模型构造所需的显式元数据。

Graph 不承载：

- sampler 配置；
- 执行状态；
- 诊断结论；
- workflow 历史；
- agent 对话。

不应过早承诺任意 Python callable 都能稳定序列化。需要持久化和引用模型时，使用 ModelRef：

- model identifier；
- graph fingerprint；
- 源码或 package 版本；
- 输入数据 identity；
- graph build arguments；
- 必要的环境信息。

Plan、Result、Report 和 Refusal 应可序列化；Graph 可以仍是 runtime object。持久化的是可重建身份和产物链，而不是假装所有 Python 对象都有跨版本稳定表示。

### 2.2 Layer 2 — Analysis and Compilation

编译必须是 task-aware 的。同一个 Graph 对不同任务可能有不同合法性，因此至少要区分：

- PosteriorTask；
- EvidenceTask；
- PredictiveTask；
- PointEstimateTask；
- SimulationTask。

例如，某个 improper prior 可能产生形式上可用的 posterior，却使 absolute evidence 未定义。若只有一个无任务上下文的 compile 接口，就很容易把 posterior 的合法性错误地传播到 evidence。

这一层产出四类 artifact：

**AnalysisReport**

记录结构事实和验证结果，例如：

- conditional independence；
- linear、Gaussian、log-linear、log-Gaussian 或 state-space 结构；
- discrete subgraph；
- rank、curvature、support 和 coupling 证据；
- properness、normalization 和 differentiability 条件；
- 静态结论与数值 probe 的证据等级。

**InferencePlan**

记录获准执行的推断策略：

- task；
- graph 与 data fingerprint；
- 精确消元的变量和顺序；
- residual parameterization；
- 选用 backend 及理由；
- 必须满足的前提；
- 运行预算与 quality gate；
- fallback 或 refusal policy。

**CompiledProblem**

是特定 backend 可执行的、语义已经锁定的问题表示。它可以包含 residual log density、变换、常数项、reconstruction map 和 backend adapter 所需信息，但不能重新解释 Graph。

**Refusal**

当结构、前提或资源不允许可靠执行时，返回机器可读的拒绝：

- task；
- failed premise；
- evidence；
- scope；
- suggested remedy；
- 是否存在更保守的 fallback。

bayesmith 已有面向天文观测问题的结构识别能力，例如 linear block、log-linear、log-Gaussian 和相关特例。这些能力应继续作为 first-party compiler passes 发展，但输出通用的 Plan 和 CompiledProblem，而不是让顶层 API 被某一应用领域限定。

早期不建立完全开放的第三方 rewrite registry。任何新增 compiler pass 都必须说明并测试：

- 它识别的结构；
- 语义保持条件；
- 失败边界；
- 数值稳定策略；
- 与原问题独立对照的 oracle。

### 2.3 Layer 3 — Inference Execution

执行层负责执行已经批准的 Plan，不负责重新分析 Graph，也不应在运行中静默换算法。

执行可以来自：

- bayesmith 自有 exact kernels；
- NumPyro posterior inference；
- 可选的 BlackJAX MCMC、SMC、VI 或 nested sampling；
- optimization 或 Laplace 路径；
- 未来其他明确适配的 backend。

NumPyro 保持当前 Graph 的通用 posterior fallback 和既有 bridge；引入 BlackJAX 不意味着替换 NumPyro。BlackJAX 更适合作为可选的 inference runtime：接收 compiler 已经锁定语义的 log density 或分离的 log-prior/log-likelihood，提供 bayesmith 不应自行重写的算法。

结果使用 tagged union，而不是一个含义随路径变化的大字典：

- PosteriorResult；
- EvidenceResult；
- PointEstimateResult；
- SimulationResult。

所有结果都引用一个共同的 RunRecord：

- plan identity；
- model、graph 和 data fingerprint；
- seed；
- dtype、device 和 JAX 配置；
- backend 及版本；
- compute budget；
- termination reason；
- timing；
- warnings；
- approximation class。

PosteriorResult 至少区分：

- draws 或 analytic posterior representation；
- reconstructed eliminated variables；
- chain/sample geometry；
- log density 或 pointwise log-likelihood 的可用性；
- 是否可直接进入 predictive evaluation。

EvidenceResult 不能只是一个浮点数。它至少应包含：

- log evidence；
- 估计不确定度；
- weighted samples 或 posterior reconstruction capability；
- termination information；
- repeated-run consistency；
- prior 和 likelihood normalization audit 的引用；
- 精确消元项与 residual evidence 的组合说明。

### 2.4 Layer 4 — Evaluation and Comparison

Evaluation 只评价 Result，不修改 posterior，也不替执行层选择新算法。

它产出 typed EvaluationReport，覆盖：

- sampling diagnostics；
- prior predictive 与 posterior predictive checks；
- identifiability 和 prior sensitivity；
- simulation-based calibration；
- pointwise log-likelihood、LOO 和 WAIC；
- evidence 与 Bayes factor；
- 多 epoch、多通道或 campaign-level diagnostics。

每份报告必须同时表达两组状态：

**适用性**

- APPLICABLE；
- INAPPLICABLE；
- UNVERIFIABLE。

**结论**

- PASS；
- FAIL；
- ABSTAIN。

运行失败应单独表达，不能伪装为 FAIL；统计方法不适用也不能伪装为运行错误。

Predictive comparison 与 evidence comparison 是不同问题：

- LOO、WAIC 和 held-out prediction 衡量预测能力；
- evidence 衡量在完整 prior 与 likelihood 定义下的 prior predictive mass；
- posterior predictive check 衡量模型能否生成与观察数据在相关方面相似的重复数据。

三者都重要，但不能互相替代。

ESS、R̂、PSIS、Pareto k̂ 等成熟的低层统计量应优先复用 BlackJAX 或 ArviZ，而不是在 bayesmith 中重复实现。bayesmith 拥有的是 observation grouping、方法适用性、结果聚合和 gate 语义，以及 Graph 驱动的 PPC、SBC、identifiability 和 prior-sensitivity workflow。

### 2.5 Layer 5 — Workflow and Agent Control Plane

控制面建立在前四层已经产生的 typed artifacts 之上。它至少包含：

- append-only artifact ledger；
- artifact identity 和 lineage；
- gate status；
- invalidation rules；
- 当前合法的 Actions；
- 每个 Action 的 prerequisites、outputs、成本和副作用。

典型状态流为：

~~~text
Graph
  -> analyze(task)
  -> compile(task)
  -> execute(plan)
  -> evaluate(result)
  -> decide next action
~~~

Action 应具有：

- 稳定名称和版本；
- 输入 artifact 类型；
- 前置 gate；
- 输出 artifact 类型；
- 估计成本；
- 幂等性或显式副作用说明；
- failure 和 retry policy。

CLI、Python 用户、自动化脚本和未来 LLM agent 都使用同一个控制面。LLM 不是核心依赖，也不拥有特殊后门。它不能伪造 PASS、忽略 evidence 的 proper-prior 要求，或把一个不可验证的结果升级为可信结论。

### 2.6 跨层约束

- 下层不得依赖上层。
- InferencePlan 是 Graph 进入 backend 的唯一授权边界。
- Result 必须与生成它的 Plan 一起解释。
- EvaluationReport 必须引用被评价的 Result。
- 上游 fingerprint 变化时，下游 artifact 必须失效。
- 编程错误使用 exception；统计方法不适用使用 Refusal。
- backend 的原生对象可以保留为附属信息，但不得取代公共 artifact。

---

## 3. 完整 Bayesian 工作流

### 3.1 生命周期

bayesmith 的完整生命周期分为六步：

1. **Declare**：构造 Graph，绑定数据和任务所需的上下文。
2. **Analyze**：发现结构、验证前提、识别不可辨识或不合法部分。
3. **Compile**：选择精确消元、参数化和 residual backend，形成 Plan。
4. **Execute**：在预算内执行 Plan，形成 Result。
5. **Evaluate**：检验数值质量、校准、预测能力或模型比较条件。
6. **Decide**：根据 gate 给出合法下一步，或正式停止并说明原因。

同一个 Graph 可以产生多个 task-specific Plan，但 artifact 不能仅因来自同一个 Graph 就跨任务复用。复用必须明确证明兼容，例如 posterior draws 是否来自与 PredictiveTask 相同的数据条件和参数化。

### 3.2 Posterior inference

posterior 路径的目标不是“拿到 samples”，而是得到可解释、可重建、可检验的 posterior artifact。

推荐顺序：

1. 验证 support、shape、coupling、rank 和 differentiability；
2. 识别并消除 exact blocks；
3. 为 residual problem 选择 analytic、NUTS、optimization 或其他合法路径；
4. 重建被消元变量或保留条件分布表示；
5. 统一封装为 PosteriorResult；
6. 运行数值诊断和 posterior predictive evaluation。

exact 与 sampled posterior 不应成为两套互不相容的产品。它们只是同一 PosteriorResult 协议的不同 representation。

### 3.3 Predictive inference 与模型检验

predictive 能力需要从“在已有 posterior samples 上调用模型”提升为显式任务。关键区别是：

- observed data 属于 conditioning；
- replicated data 是新的随机结果；
- prediction 可以是已有观测点的复现，也可以是新设计点、未来 epoch 或 held-out data。

PredictiveTask 应明确：

- conditioning dataset；
- prediction design 或 future covariates；
- 哪些 observed sites 需要解除条件；
- replicate 数量；
- discrepancy 或 summary；
- 是否保留 latent predictive quantities。

模型检验的基础能力包括：

**Prior predictive checks**

在观察 posterior 之前检查先验是否生成物理上或观测上合理的数据尺度。

**Posterior predictive checks**

比较 replicated data 与 observed data。框架应支持用户提供 discrepancy function，同时提供少量通用、稳定的默认统计量。

**Held-out prediction**

对未参与 conditioning 的 observation、epoch、channel 或 group 计算 predictive performance。

**Pointwise log-likelihood**

为 LOO、WAIC 和诊断保留明确的 observation unit。plate 和 grouping 语义必须进入定义，不能事后猜测。

**Simulation-based calibration**

从 prior 生成真值和数据，重复推断，并检查 posterior ranks 或等价校准统计量。SBC 既检验推断算法，也暴露模型实现与参数化错误。

**Identifiability and prior sensitivity**

静态结构诊断、局部 curvature、posterior geometry 和有计划的 prior perturbation 应形成相互区分的报告。

第一阶段可以通过可选 ArviZ bridge 获得成熟的 LOO、WAIC 和标准可视化生态，但 Graph、observation grouping、适用性判断和最终 artifact 仍由 bayesmith 定义。

### 3.4 Evidence 与模型比较

evidence 是独立 task，不是 posterior inference 的附带标量。对模型 M 和数据 y：

~~~text
Z = p(y | M) = integral p(y | theta, M) p(theta | M) d theta
~~~

一个 EvidenceTask 在执行前必须通过 eligibility audit：

- 所有参与 evidence 的 prior 是否 proper 且 normalized；
- likelihood 是否包含 absolute normalization constants；
- 数据、base measure 和 conditioning 是否定义完整；
- 参数变换的 Jacobian 是否纳入；
- 精确消元是否保留全部常数项；
- 被比较模型是否使用相同数据语义；
- 是否存在会让 Bayes factor 无意义的 data-dependent prior 或隐式截断。

EvidenceTask 的首选编译路径是：

1. 精确积分 Gaussian、linear、discrete 或其他已验证 block；
2. 将所有 normalization 与 determinant terms 记入 compiled evidence expression；
3. 仅对低维 residual problem 使用数值积分；
4. 合并 exact contribution 与 residual log evidence；
5. 运行重复估计、误差评估和 prior sensitivity。

这正是 bayesmith 最有差异化潜力的 evidence 路径：它不与 nested sampler 竞争，而是给 nested sampler 一个更小、更平滑、更符合前提的 residual problem。

### 3.5 是否接 nested sampling

结论是：**需要接，但作为可选 backend，不作为核心依赖，也不自己实现通用 nested sampler。**

接入策略：

- 先定义 backend-neutral CompiledEvidenceProblem；它是 CompiledProblem 的 evidence-specific variant，不是新的顶层 artifact，至少包含分离的 log-prior、log-likelihood、prior sampling 或 transform、参数表示、normalization audit 和 reconstruction information；
- 第一轮 backend evaluation 同时考察 BlackJAX nested sampling 与 JAXNS，不预先指定胜者；
- 使用相同的 analytic oracles、代表性的 astronomy-shaped residual problems 和运行预算比较 correctness、bias、uncertainty、termination、multimodal behavior、JIT/compile cost、memory 和 API stability；
- 至少选择一个 production adapter；只有在第二实现能提供实质性的独立 cross-check 或不同适用域时才长期维护两个；
- adapter 只接收 EvidenceTask 编译出的 CompiledEvidenceProblem；
- 未安装可选依赖时返回明确的 capability Refusal；
- backend 的 termination、logZ uncertainty、weighted samples 和 diagnostics 全部进入 EvidenceResult；
- 默认至少提供独立重复运行或等价稳定性检查；
- x64 与数值尺度检查属于 evidence gate；
- dynesty 等非 JAX backend 只有在明确补足适用域或提供独立验证价值时才进入后续评估。

BlackJAX 是特别自然的候选，因为它直接以 JAX log density 为边界，nested sampling API 还要求显式分离 log-prior 与 log-likelihood，并且同一 optional dependency 可进一步提供 SMC、MCLMC、Pathfinder 和其他 execution routes。它仍然只是执行层：Graph 解释、exact collapse、evidence eligibility、prior normalization、结果 gate 和 backend 选择继续由 bayesmith 拥有。

不应把“nested sampling 成功返回”当作 evidence 可信。可信还依赖：

- task eligibility；
- normalization audit；
- prior sensitivity；
- residual dimensionality 与 geometry；
- termination；
- repeated-run consistency；
- 必要时与 analytic oracle 或独立算法交叉验证。

Nested sampling 也不等于 model checking。它解决的是积分和 posterior weighted samples；PPC、SBC、LOO 和科学 discrepancy 仍需单独建立。

### 3.6 Point estimate 与 simulation

MAP、Laplace、profile 或其他 point-estimate 路径必须明确其统计含义。PointEstimateResult 不得冒充 PosteriorResult。

SimulationTask 则统一承载：

- prior simulation；
- posterior predictive simulation；
- SBC data generation；
- synthetic oracle construction；
- experimental design 所需的 forward simulation。

这样 simulation 不再只是测试辅助函数，而成为 model criticism、校准和未来 agent loop 的基础动作。

---

## 4. Artifact、gate 与状态机

### 4.1 Artifact envelope

每个公共 artifact 都应有共同 envelope：

- artifact type 与 schema version；
- stable identifier；
- created-at；
- producer 与 package version；
- parent artifact identifiers；
- model、graph、data 和 task fingerprint；
- status；
- warnings；
- human-readable summary；
- machine-readable payload。

核心统计对象应使用强类型 Python representation；JSON-compatible schema 用于记录、CLI、agent 和跨进程传递。不能为了 JSON 简单而把所有内容退化成无约束字典。

### 4.2 Fingerprint 与 invalidation

至少区分：

- model source identity；
- graph structure fingerprint；
- data identity；
- task configuration；
- compilation configuration；
- execution environment。

变更传播规则必须可预测：

- 模型或 Graph 改变：所有下游失效；
- conditioning data 改变：Plan、Result 和 Report 失效；
- evaluation threshold 改变：Result 可复用，相关 Report 失效；
- 仅显示选项改变：统计 artifact 不失效；
- backend patch version 改变：旧结果仍可读取，但新运行有新 provenance；
- agent prompt 改变：不改变统计 artifact，只改变 action selection record。

### 4.3 Gate

建议的 gate 顺序：

1. Graph validity；
2. task eligibility；
3. compilation validity；
4. execution completion；
5. numerical quality；
6. statistical evaluation；
7. comparison eligibility；
8. publication or decision readiness。

Gate 必须：

- 由明确的 artifact 证据计算；
- 输出 PASS、FAIL 或 ABSTAIN；
- 说明阻止哪些 action；
- 说明哪些修复动作仍合法；
- 在依赖失效时自动回退。

### 4.4 Refusal 与 exception

Refusal 是预期内的统计或能力边界，例如：

- evidence 使用 improper prior；
- model structure 不在某 exact compiler pass 的支持域；
- pointwise likelihood 无法定义 observation unit；
- optional backend 未安装；
- 诊断所需信息未记录。

Exception 是实现错误、违反内部 invariant 或环境损坏。将两者分开后，agent 才不会把一个软件 bug 当作科学结论，也不会把合法的“不适用”当作需要无限重试的故障。

---

## 5. Agent-ready workflow

### 5.1 设计目标

未来 agent loop 的价值不在于让 LLM 直接写 sampler 参数，而在于让它：

- 读取结构化 artifact；
- 解释当前可信结论；
- 在 allowed actions 中选择下一步；
- 比较成本、风险与信息增益；
- 请求必要的人类决策；
- 保持完整 provenance；
- 在满足停止条件时停止。

因此先建设 typed workflow，再接 LLM。没有 typed workflow 的 agent 只能围绕日志和字符串做脆弱的 prompt orchestration。

### 5.2 WorkflowState

WorkflowState 是某一 model/data/task lineage 在一个时刻的投影，包括：

- latest valid artifacts；
- passed、failed 和 abstained gates；
- invalidated artifacts；
- active budgets；
- outstanding approvals；
- allowed actions；
- terminal conditions。

ledger 应是 append-only；WorkflowState 可以从 ledger 重建。这样 agent 的每次选择都可审计、回放和比较。

### 5.3 Action protocol

最小 Action 类型包括：

- Analyze；
- Compile；
- Execute；
- Evaluate；
- Compare；
- Simulate；
- Escalate；
- ProposeRevision；
- Stop。

ProposeRevision 只产生候选改动和理由。它不默认修改科学模型。真正修改 Graph、data policy 或 prior 需要用户明确授权或预先声明的受控规则。

每个 Action 应声明：

- consumes；
- produces；
- prerequisite gates；
- invalidates；
- estimated cost；
- side-effect class；
- retryability；
- human approval requirement。

### 5.4 预算与停止

agent workflow 必须有显式预算：

- wall-clock；
- number of model evaluations；
- accelerator or memory budget；
- maximum retries；
- maximum model revisions；
- evidence precision target；
- diagnostic improvement threshold。

建议的停止条件包括：

- 目标 gate 全部通过；
- 当前结论为 ABSTAIN 且没有合法的信息增益动作；
- 后续动作超过预算；
- 需要人类科学判断；
- 多次合法尝试没有改善；
- 模型或数据需要实质性重定义。

### 5.5 安全边界

LLM 可以：

- 选择已注册 Action；
- 填写允许的配置；
- 总结 artifact；
- 提议额外诊断；
- 生成供人审阅的模型修订建议。

LLM 不可以：

- 改写 artifact 内容；
- 把 FAIL 或 ABSTAIN 解释成 PASS；
- 删除 provenance；
- 跳过 task eligibility；
- 自动扩大预算；
- 未经授权改变 prior、likelihood 或 observation selection；
- 仅凭自然语言声称模型已被验证。

---

## 6. 目标能力地图

### 6.1 Structure and compilation

- 显式 Graph 与 task-aware analysis；
- linear-Gaussian、log-linear、log-Gaussian 和相关 block；
- state-space 与可消元 chain；
- finite discrete subgraph；
- rank、support、coupling、normalization 和 curvature probes；
- exact-first residualization；
- 可解释 cost model 与 fallback。

### 6.2 Posterior

- analytic posterior；
- hybrid exact plus sampled posterior；
- NUTS/MCMC；
- optimization 和 Laplace；
- eliminated variable reconstruction；
- unified PosteriorResult；
- sampling geometry 与 quality diagnostics。

### 6.3 Predictive and criticism

- prior predictive；
- posterior predictive；
- future or held-out prediction；
- user-defined discrepancy；
- pointwise log-likelihood；
- LOO/WAIC bridge；
- SBC；
- identifiability 和 prior sensitivity。

### 6.4 Evidence and comparison

- task eligibility 与 normalization audit；
- exact evidence components；
- graph-level exact evidence compilation；
- optional nested residual integration；
- EvidenceResult；
- repeated-run stability；
- Bayes factor eligibility；
- prior sensitivity；
- 与 predictive comparison 并列而不混淆。

### 6.5 Workflow

- artifact schema；
- lineage 和 invalidation；
- gates；
- action registry；
- budget 和 stop conditions；
- deterministic replay；
- CLI/API；
- external agent example。

---

## 7. 包边界与演进方式

### 7.1 保留现有强项

现有 graph、dispatch、exact、marginal、diagnose、bridge、optimize 和 amortize 等职责应作为演进起点，不进行仅为“看起来更整齐”的大规模搬家。

顶层设计需要新增的是概念边界，而不是立即新增许多 package：

- tasks；
- artifacts；
- result protocols；
- evaluation；
- workflow state；
- backend adapters。

这些可以先在少量模块中成熟，再根据依赖方向和代码量拆分。

### 7.2 Namespace 约束

当前若已有 deprecated namespace 或 compatibility alias，不应在其上建设新的长期实现。特别是 evidence 相关名称，应先核对现有兼容承诺，再选择不会与退役路径冲突的新归属。

具体模块命名留给每阶段实现计划决定，但依赖方向必须保持：

~~~text
model/graph
    -> analysis/compiler
        -> execution adapters
            -> evaluation
                -> workflow
~~~

共享 artifact 类型可以位于低耦合模块，但不能形成一个所有层都反向依赖的无边界 common bucket。

### 7.3 Backend policy

引入 backend 前必须回答：

1. 它补足了什么 bayesmith 自己不应重写的能力？
2. 它能否服从现有 Task、Plan、Result 和 Refusal 协议？
3. optional dependency 缺失时是否优雅退化？
4. 是否有独立 oracle 或 cross-check？
5. backend-specific object 泄漏到公共 API 的范围是否最小？
6. 升级 backend 时如何检测语义漂移？
7. 它要求的目标表示是否能由 CompiledProblem 无损提供，而不让 backend 重新解释 Graph？
8. 如果已有 adapter 能完成任务，第二个 adapter 是否带来独立验证或新的适用域，而不只是增加算法数量？

不以 backend 数量作为成熟度指标。

backend 评估必须针对具体 Task 和 compiled problem family，不能笼统宣布某个 library 为全局首选。对 posterior，NumPyro 可以继续是默认通用 fallback；对 residual evidence，BlackJAX NS 与 JAXNS 通过同一 oracle suite 竞争；对某些 geometry，未来也可以由 Plan 选择 BlackJAX 的专门算法。无论选择哪一个，公共 Result、Report 和 gate 不随 backend 改变。

---

## 8. 路线图

路线图按能力门槛推进，不绑定日历时间。编号表示依赖顺序，不要求每个阶段对应一个 release。

### R0 — 稳定现有核心与基线

**目标**

把当前 Graph、exact inference、marginal terms、diagnostics、graph reduction 和 backend bridge 固化为可信基线。

**主要工作**

- 完成当前在途的 graph reduction 与 log-determinant 稳定性工作；
- 明确现有公共对象的实际语义；
- 修正 README 和文档中 posterior、marginal term 与 evidence 的措辞；
- 记录现有 route、fallback 和 refusal；
- 建立可重复的 full-suite、lint 和 wheel-level 验证；
- 不在此阶段追求新的 API 广度。

**完成门槛**

- 全量测试、lint 和 package 验证可复现；
- exact 和 fallback 路径有独立 oracle；
- 公共文档与实际行为一致；
- 当前核心对象有清晰 owner 和兼容性策略。

### R1 — Task、artifact 与 provenance 基础

**目标**

为所有后续能力建立共同语言，而不改变已有数值结果。

**主要工作**

- 定义 Task family；
- 定义 AnalysisReport、InferencePlan、Refusal；
- 定义 tagged Result 和 RunRecord；
- 建立 schema version、fingerprint 和 lineage；
- 建立最小 invalidation rules；
- 将现有 posterior routes 适配到新协议；
- 保留兼容层并制定 deprecation 路径。

**完成门槛**

- analytic、MCMC 和 optimization 路径都能返回统一协议；
- 同一运行可被稳定记录和读取；
- 数据或 Graph 改动会使相关 artifact 明确失效；
- method inapplicability 不再依赖解析异常字符串；
- 数值结果与迁移前保持一致。

### R2 — 完整 posterior 与 predictive seam

**目标**

把“posterior 能跑”提升为“posterior 可检验、可预测、可比较”。

**主要工作**

- 明确 PosteriorResult 的 analytic 与 sample representations；
- 统一 eliminated variable reconstruction；
- 建立显式 PredictiveTask；
- 正确区分 conditioning observations 与 replicated observations；
- 记录 pointwise log-likelihood 和 observation grouping；
- 提供可选 ArviZ export；
- 统一多链 sampling diagnostics。

**完成门槛**

- exact、factorized 和 NUTS 路径可进入同一 predictive API；
- observed-data replay 不会被误当 posterior predictive；
- analytic posterior 与 predictive moments 有独立 oracle；
- pointwise likelihood 的 observation unit 可审计；
- 缺少 predictive 所需信息时返回 Refusal 或 ABSTAIN。

### R3 — Model checking 与 calibration

**目标**

建立通用而非仅 campaign-specific 的模型检验层。

**主要工作**

- prior predictive checks；
- posterior predictive discrepancy framework；
- held-out prediction；
- SBC harness；
- LOO/WAIC integration；
- prior sensitivity；
- identifiability reports；
- PASS、FAIL、ABSTAIN 和 applicability 协议。

**完成门槛**

- 已知校准的 synthetic fixtures 通过；
- 已知错误或 misspecified fixtures 能失败；
- 不足以判断的 fixtures 能 ABSTAIN；
- SBC 能同时覆盖 exact 与 sampled routes；
- ArviZ round-trip 不丢失 observation 或 chain 语义；
- report 不依赖人工阅读原始 sampler 日志。

### R4 — Evidence foundation

**目标**

把已有 marginal-likelihood 数学部件提升为 graph-level EvidenceTask。

**主要工作**

- proper-prior 和 absolute-normalization audit；
- EvidenceTask eligibility；
- 精确 Gaussian、linear、discrete 和 chain evidence assembly；
- 将 graph reduction 的常数项与 determinant terms 纳入同一表达；
- 定义 EvidenceResult；
- 定义 Bayes factor comparability；
- 建立 prior sensitivity report。

**完成门槛**

- 多组 prior scale 下与 dense analytic log evidence 一致；
- 常数项、Jacobian 或 determinant 的 mutation 会被测试杀死；
- improper prior 或不完整 likelihood 会被正式拒绝；
- 比较不同 data semantics 的模型会被拒绝；
- exact evidence 的 provenance 可追溯到各个编译项。

### R5 — Nested residual evidence

**目标**

为不可完全解析的低维 residual problem 提供可靠的数值 evidence。

**主要工作**

- 定义 backend-neutral CompiledEvidenceProblem 与 adapter contract，并保持它是 CompiledProblem 的 evidence-specific variant；
- 对 BlackJAX nested sampling 和 JAXNS 做同预算、同 oracle 的 backend evaluation；
- 根据 correctness、bias、uncertainty calibration、termination、multimodal behavior、JIT/compile cost、memory、API stability 和维护成本记录选择决策；
- 建立至少一个可选 production adapter；
- exact collapse 后再调用 nested sampling；
- 记录 termination、uncertainty 和 weighted samples；
- 独立重复运行或等价 stability check；
- 加入 x64、尺度和 prior transform 检查；
- 仅在有独立验证价值或不同适用域时保留第二 backend。

**完成门槛**

- backend 决策有可复现 benchmark、oracle 结果和明确适用域，而不是预设偏好；
- 在 analytic oracle 上 log evidence 与声明误差一致；
- 至少覆盖一个非高斯或多模态 fixture；
- exact collapse 与未 collapse 的小问题对照一致；
- optional dependency 缺失时清晰拒绝，不破坏核心安装；
- 不稳定的重复估计不能通过 evidence gate；
- EvidenceResult 可用于 posterior reconstruction 和比较报告。

### R6 — Compiler breadth

**目标**

在核心协议稳定后扩展结构识别覆盖面。

**候选方向**

- 更完整的 finite discrete dispatch；
- forward-backward 或 message passing；
- nested plates；
- structured covariance；
- 更多 log-linear、log-Gaussian 变体；
- nonlinear residual 的局部或混合编译；
- 更精确的 cost model；
- exact、Laplace、MCMC 和 nested 的混合计划。

**完成门槛**

每个新增 pass 都必须具备：

- 明确支持域；
- 语义保持说明；
- 独立 oracle；
- 边界与拒绝测试；
- 数值稳定测试；
- 保守 fallback；
- 至少一个真实科学问题上的收益证据。

### R7 — Workflow 与 agent seam

**目标**

将已有统计能力组合成可回放、可约束的闭环。

**主要工作**

- append-only artifact ledger；
- WorkflowState；
- gate evaluation；
- Action registry；
- cost、budget 和 stop conditions；
- deterministic replay；
- CLI 或 JSON protocol；
- 一个不依赖特定 LLM 的外部 agent 示例；
- human approval points。

**完成门槛**

- 同一 ledger 可重建相同 state 和 allowed actions；
- agent 无法调用未通过 prerequisite gate 的 action；
- Graph 或 data 改动会正确失效下游；
- workflow 在没有 LLM 时完全可运行；
- mock policy、规则 policy 和 LLM policy 使用相同 Action API；
- 达到预算或需要科学判断时会正式停止。

### R8 — 1.0 与生态成熟

**目标**

把内部一致的系统变成可由外部用户长期依赖的产品。

**主要工作**

- 稳定公共 API 和 schema；
- 完成 deprecated aliases 的退役；
- 定义兼容性和 migration policy；
- 完整用户文档和端到端教程；
- posterior、model checking、evidence 和 agent loop 示例；
- benchmark 与性能基线；
- wheel、optional extras 和 release verification；
- 收集独立用户反馈。

**完成门槛**

- 至少两个非仓库作者控制的实际 consumer；
- 关键 workflow 有版本化 artifact compatibility；
- optional backends 的失败不会影响基础安装；
- 每类核心 Task 都有一条完整、可复现的 tutorial；
- 一次模型或数据修改的 invalidation 行为有稳定保证；
- 1.0 承诺与实际维护能力匹配。

---

## 9. 测试与验收哲学

### 9.1 Oracle 层级

可信度从高到低建议为：

1. 独立 analytic 或 dense oracle；
2. 与不同数学实现的交叉验证；
3. 跨 backend 对照；
4. property 或 metamorphic test；
5. self-consistency；
6. snapshot 或日志匹配。

不能用同一实现的两个入口互相证明正确。尤其 evidence 中的 normalization constants、Jacobian 和 log determinant 必须有独立来源。

### 9.2 三类质量门

**Semantic correctness**

- Graph 与 task 语义一致；
- compiler transformation 保持目标分布；
- 常数项和变换完整；
- refusal boundary 正确。

**Statistical calibration**

- posterior coverage；
- SBC ranks；
- Monte Carlo uncertainty；
- PPC behavior；
- evidence estimator calibration。

**Operational reproducibility**

- seed 与 environment 被记录；
- artifact 可重建；
- wheel 与 source 行为一致；
- optional dependency 行为明确；
- replay 产生相同状态转换。

### 9.3 必须覆盖的测试类型

- exact oracle tests；
- boundary 和 refusal tests；
- mutation tests；
- randomized property tests；
- dtype 与 x64 tests；
- multi-device 或 backend provenance tests；
- serialization round-trip；
- invalidation tests；
- agent gate bypass tests；
- performance regression measurements。

性能门槛应先基于测量建立，再决定是否硬性阻断。不能把未经基准证明的“更快”写成设计事实。

### 9.4 Evidence 专项

Evidence 测试至少覆盖：

- prior normalization；
- likelihood constants；
- change of variables；
- prior volume sensitivity；
- analytic Gaussian evidence；
- exact collapse plus residual evidence；
- repeated nested runs；
- Bayes factor comparability；
- undefined evidence refusal。

### 9.5 Model checking 专项

模型检验测试至少覆盖：

- 正确模型与正确推断；
- 正确模型与错误实现；
- misspecified model；
- weakly identified model；
- insufficient simulation budget；
- 无法定义 discrepancy 或 observation unit；
- PASS、FAIL 和 ABSTAIN 三条路径。

---

## 10. 治理与演进规则

### 10.1 一个决定一个家

长期产品决定只在本文维护。专题 spec 可以引用本文并记录局部实现决定，但不复制整份 roadmap。

每个待决问题的答案应写回提出问题的同一位置。一个已经解决的问题不能在别处继续以开放问题存在。

### 10.2 新功能准入

任何显著新功能先回答：

1. 它解决的是 bayesmith 应拥有的统计语义，还是成熟依赖已经解决的数值问题？
2. 它是否利用了 Graph 和 graph-aware、task-aware compilation 的差异化价值？
3. 它的适用条件能否被检查？
4. 它失败时能否正式 Refuse 或 ABSTAIN？
5. 它是否有独立 oracle？
6. 它是否增加了长期公共 API 表面积？
7. 如果只是 adapter，为什么不能停留在外部示例？

### 10.3 兼容性

- artifact schema 必须版本化；
- 公共语义变化需要 migration note；
- backend 原生对象不享受同等稳定承诺；
- deprecated API 有明确替代和退役点；
- alpha 阶段可以调整设计，但不能静默改变统计含义；
- 1.0 前应通过真实 consumer 验证，而不只看内部测试数量。

### 10.4 文档

文档应按用户问题组织，而不只按模块组织：

- 我声明了什么模型？
- 为什么走这条推断路径？
- 结果是否可信？
- evidence 为什么可定义或不可定义？
- 模型在哪些方面失败？
- 下一步有哪些选择？

每个高层示例都应展示 Plan、Result 和至少一个 Report，而不只展示最终数组或图。

---

## 11. 主要风险与缓解

### 11.1 Graph 序列化过度承诺

**风险：** 任意 callable、closure 和外部状态无法稳定跨版本序列化。  
**缓解：** 持久化 ModelRef、fingerprint、构造参数和 artifacts；Graph 保持可重建的 runtime object。

### 11.2 数值 probe 被误当证明

**风险：** coupling、rank 或 linearity probe 在有限点上通过，但全局前提不成立。  
**缓解：** 标明证据等级；关键 transformation 要有结构证明或受限支持域；不确定时拒绝或走保守 backend。

### 11.3 自动分块改变模型语义

**风险：** 错误的 conditional independence、plate 或 normalization 处理导致 silent bias。  
**缓解：** compiler pass 有 independent oracle、reconstruction check、mutation test 和显式 fallback。

### 11.4 Evidence 给出虚假确定性

**风险：** logZ 数字看似精确，却依赖任意 prior volume、不完整常数项或不稳定 estimator。  
**缓解：** EvidenceTask eligibility、prior sensitivity、uncertainty、重复运行和 comparability report 强制进入 gate。

### 11.5 Agent 放大错误

**风险：** 自动化快速重复一个错误假设，或把诊断警告解释为成功。  
**缓解：** typed artifacts、closed action set、gate enforcement、budget、human approval 和 append-only audit。

### 11.6 范围膨胀

**风险：** 同时建设 PPL、sampler、visualization、workflow engine 和 LLM framework。  
**缓解：** 坚持 compiler-first；优先拥有统计语义，复用数值与展示生态。

### 11.7 Adapter 漂移

**风险：** NumPyro、BlackJAX、JAXNS 或 ArviZ 升级改变行为。

**缓解：** versioned provenance、contract tests、wheel-level integration tests 和最小 adapter surface。

### 11.8 Artifact 体系先于真实需求而过度设计

**风险：** 为未来 agent 构建庞大框架，却没有真实统计 workflow 验证。  
**缓解：** R1 只建立最小协议；R2–R5 用 posterior、model checking 和 evidence 实际驱动；完整控制面推迟到 R7。

---

## 12. 成熟度判据

bayesmith 达到“通用 Bayesian workflow 工具”的成熟状态，不以支持多少 sampler 或画多少图判断，而以以下事实判断：

- 支持域内的 Graph 能回答第一节列出的五个问题；
- exact、approximate 和 heuristic 路径不会静默混淆；
- posterior 结果能自然进入 predictive checks 和 calibration；
- evidence 有完整 eligibility、normalization、uncertainty 和 sensitivity；
- predictive comparison、evidence comparison 与 posterior predictive criticism 各自清晰；
- inapplicability、failure 与 inconclusive 有不同 artifact；
- 所有重要结论都有 provenance 和适用条件；
- backend 可替换而核心语义稳定；
- model/data 变化能可靠 invalidation；
- 无 LLM 时 workflow 完整可用；
- 有 LLM 时 agent 只能在同一受控 Action API 上工作；
- 至少有独立外部 consumer 证明这些抽象不是只为仓库内部服务。

---

## 13. 明确推迟的事项

以下事项不是当前路线图的前置条件：

- 自研通用 nested sampler；
- 通用分布语言；
- 任意 Python Graph 的完全序列化；
- 开放式第三方 compiler rewrite marketplace；
- 分布式任务调度平台；
- 内置聊天 UI；
- 自动修改科学模型；
- 让 LLM 决定统计 gate；
- 为覆盖所有 PPL 建立庞大 backend matrix。

如果未来要引入其中任何一项，必须先证明它解决了已经出现的真实瓶颈，并且不会模糊 bayesmith 对结构、任务和可信结论的所有权。

---

## 14. 当前总决策

1. bayesmith 采用 **compiler-first** 的长期定位。
2. 它的通用性来自完整 Bayesian workflow，而不是复制整个 PPL 生态。
3. 当前 posterior 与结构化 exact inference 是基础优势；下一块最重要的缺口是通用 model checking。
4. Evidence 应建设为独立、task-aware 的产品能力。
5. Nested sampling 应接入现成工具但不进入核心依赖；BlackJAX NS 与 JAXNS 先通过同一 oracle-driven evaluation，再选择至少一个 production adapter。
6. bayesmith 的独特 evidence 优势应是先精确消元，再对 residual problem 做 nested sampling。
7. astronomy-oriented structural optimizations 继续作为 first-party compiler capabilities，但不限制顶层定位。
8. agent 是未来外部控制策略；typed artifact、gate 和 Action protocol 才是核心。
9. 先完成 posterior predictive、SBC、LOO/WAIC 和 evidence contracts，再建设完整 agent loop。
10. 以后所有阶段性计划以本文为北极星，并用客观 gate 而不是日期或功能数量定义完成。
