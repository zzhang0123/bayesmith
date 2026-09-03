# bayesmith ← rheplicant 贝叶斯层迁移 — 规格与验收

> **文档状态：`decision-home`** · 某类决定的唯一登记处，仍在更新；决定的答案写回提出它的那一行。索引见 docs/README.md。

> 本文是**规格**，不是实施计划：它定义「什么必须为真」，逐阶段的实施计划另开
> （沿用 `plans/` 的任务化写法）。
>
> **本文自足。** 执行 bayesmith 侧工作的 session 只需读本仓库；对 rheplicant 的
> 引用一律给出 `file:line`，需要原文时再去读那个仓库。
>
> **上游证据**（在 e-RHINO 仓库，`/Users/zzhang/projects/e-RHINO`）：
> - `CODE_REVIEW_REPORT.md` — 评审 R1（英文），bootstrap/审计层与 config 代理完整性最全
> - `REVIEW_REPORT.md` — 评审 R2（中文），第一性原理重推最深，逐条标注核实状态
> - `PROPOSAL_MERGED.md` — 两份评审的整合裁决 + Track A（rheplicant 非贝叶斯半边）
>
> **分工边界：** 凡 `src/rheplicant/inference/` 的东西在本文；其余在 e-RHINO 的
> `PROPOSAL_MERGED.md` Track A。两份文档**不重复内容**，只互相指针。
>
> **日期：** 2026-08-24。**标注约定：** `[实测]` = 整合阶段亲自跑过命令；
> `[互证]` = 两份独立评审各自发现；`[R1]`/`[R2]` = 单一来源、未复核，动手前先验证。

---

## 〇、先读这一节：三个实测事实

### 0.1 cross-check harness 现在无处可跑 `[实测]`

两份评审方案都把 cross-check harness 称作「最高价值的第一个 commit」，但都没
发现它跑不起来：

```
bayesmith/.venv:  python -c "import rheplicant"  → ModuleNotFoundError
e-RHINO/.venv:    python -c "import bayesmith"   → ModuleNotFoundError
```

两边同为 Python 3.12.9，各自 editable 安装在自己的 venv 里。

**决定（本文采纳，理由见 §五 D6）：harness 住在 bayesmith 的 venv，
rheplicant 作为「仅测试」依赖。** bayesmith 绝不获得对 rheplicant 的运行时依赖；
rheplicant 更不该获得一个对 0.0.0 版包的依赖。落地：

```bash
cd /Users/zzhang/projects/bayesmith
uv pip install --python .venv/bin/python --no-deps -e /Users/zzhang/projects/e-RHINO
```

**已落地（2026-08-24），并且上面这条命令和本文原先写的那条不是同一条。**
原文写的是 `.venv/bin/pip install -e ...`，**跑不起来**：这个 venv 由 uv 管理，
里面根本没有 `pip`（`ls .venv/bin` 只有 `python*`）。

`--no-deps` 也不是图省事，是实测后的决定：rheplicant 声明的依赖里有 `limTOD`，
带进来约七十八个传递依赖，还会拿第二套约束去重解 jax/numpy。而实测
`rheplicant`、`rheplicant.inference`、甚至 `rheplicant.config` 在
**limTOD 与 PyYAML 都被屏蔽时照样 import 成功**（天空引擎是懒加载 limTOD 的），
所以 cross-check 要比对的那一层只需要 jax / equinox / numpy —— 这个 venv 本来
就有。装前装后实测：**26 个包变 27 个，本套件 637 passed → 637 passed**。

`pyproject.toml` 的 `[dependency-groups]` 下已新增 `crosscheck` 组，把上面这些
连同「本地路径依赖、不进 wheel、缺席时 skip 并说明原因」一并写在注释里；
`[tool.pytest.ini_options].markers` 下新增 `crosscheck` 标记。
第一个 harness 是 `tests/crosscheck/`（见 §0.1a）。

**一个必须提前记下的技术皱褶：精度模型两边不同。** rheplicant 用**进程全局**
`jax_enable_x64`（这正是它有两个 pytest session 的原因：证据层要 float64，
其余测试断言的拒绝只有 float32 才触发）；bayesmith 的纪律是库内**绝不**调
`jax.config.update`，一律 `with jax.enable_x64(True):` 上下文。同时 import 两个
包的 harness 必须尊重这个差异——**证据层的比对得走 rheplicant 那个 float64
子进程**（见 e-RHINO `tests/test_evidence_session.py`），不能指望在一个进程里
同时满足两边。

**这条差异还有第二半，实测补记（2026-08-24）：同一个 key，x64 开与不开抽出来的
是完全不同的数。**

```
jax.random.normal(jax.random.key(6), (4,))
  float32 : [ 0.386472, -0.570797, -1.678261, -1.203193]
  float64 : [ 0.040150,  0.968112, -1.136482,  0.389514]
```

所以 §四 4.1 要求的「同一 `gain × T_ant` fixture」**不能靠同一个 seed 复现**——
两边精度模型不同，同一行构造代码会生成两组不同的数据。必须把**数据数组本身**
传过去（或从一个与 dtype 无关的来源读）。否则比对失败会以「数值不一致」的面目
出现，指向求解器而不是指向 fixture。

同样的坑在**单个进程内**也会咬人：`with jax.enable_x64(True):` 管的是**运算**，
不是数组。一个 float64 数组在 with 块**外**做 `jnp.mean` 会被截回 float32。
写这份 harness 时实测踩到一次，表现为估计量在第八位「错」了。

### 0.1a harness 的第一个结果：rheplicant 的 kappa 偏向不安全的一侧 `[实测]`

`tests/crosscheck/test_conditioning.py`（9 个用例，全部变异测试过）比对
`bayesmith.exact.conditioning` 与 `rheplicant.core.conditioning`（**注意**：
上游今天把它从 `inference/` 移到了 `core/`，因为 `radio` 需要它而又不许 import
`inference`；本文其余处提到 `rheplicant.inference.conditioning` 的都要按此改读）。

已移植的部分**逐位一致**：`tree_norm`（含两边 docstring 都在论证的 1e20 溢出行）
与 `largest_eigenvalue`（同算子、同 template、同 key、同迭代数）。

真正的收获是**没有移植的那个**。本仓库 `conditioning.py` 的模块 docstring 早已
写明：`extreme_eigenvalues` 用「在 `lambda_max*I - M` 上再跑一次幂迭代」求
`lambda_min`，在梯度谱上**原理上就不成立**，而且偏差是**单侧、朝危险方向**的。
harness 现在把这句话变成一个活的数字。在 20 点几何谱、真实 kappa = 1e4 上实测：

```
true  lambda_max = 9.999999e-01   lambda_min = 9.997481e-05
est   lambda_max = 9.998552e-01   lambda_min = 3.392756e-03   <- 偏大 33.9 倍
true  kappa = 1.000e+04           reported kappa = 2.947e+02   <- 偏小 33.9 倍
```

**后果落在 rheplicant 那边，不在这边。** 上游两处守卫都算
`kappa = lambda_max / max(lambda_min_measured, floor)`，其中 `floor` 正是
`1/max(prior_variance)`（`inference/linear.py::_condition_number`）或 ridge
（`radio/filters/skyspace.py::_checked`，2026-08-24 新增）。也就是说**严格下界
已经在手里，却在测量值更大时被丢弃**——而测量值偏大恰恰是常态。于是
`require_convergence` 限制的那个 `kappa * residual` 误差上界被系统性低报，守卫
在最该开口的时候沉默。本仓库 `exact/solve.py::condition_bound` 的做法
（`lambda_max * max(prior_variance)`，配合 `lambda_max` 只会从下方逼近）给出的是
**kappa 的上界**，才是安全守卫要的方向。

**这是 Track A 的条目，不是本迁移的。** 记在这里，因为它是 harness 存在的理由
第一次兑现：一次手工比对只能在写下的那天成立，之后两边就在沉默里各走各的。
`test_rheplicant_still_carries_it_and_still_leans_the_unsafe_way` 会在上游修好的
那天变红，届时连同它守的这一段一起删掉。

### 0.2 e-RHINO 的完整测试套件当前是绿的，除了一个过期数字 `[实测]`

```
.venv/bin/python -m pytest -n 8
1 failed, 9673 passed, 501 skipped, 712 warnings in 280.77s
Total coverage: 88.29%
FAILED tests/test_readme_counts.py::test_every_test_count_in_the_readme_is_the_real_one
```

唯一的失败是 README 里写着 10153 而实收 10174。**这对本文的意义是：把
rheplicant 当作 cross-check 基准是安全的**——它的数值层确实还在做它声称的事，
两份评审逐条重推的结论有一个绿套件背书。（那个红是 Track A 的 A0，与本文无关。）

### 0.3 rheplicant 的两条已知缺陷，会污染 cross-check `[见 §三]`

见 §二铁律 5 与 §三的 B1、B2。**这是本文最重要的一条工程后果**，先读。

---

## 一、迁移的五条铁律

前四条由两位评审独立提出，第五条是整合阶段追加的。

1. **先 cross-check 一致，再挪动。** 任何模块从 rheplicant 退休之前，其
   bayesmith 替代物必须在共享 fixture 上**被证明**与之一致。
2. **新功能直接在 bayesmith 实现，绝不回补 rheplicant。** 相关噪声、非高斯
   似然、流式证据的 config 面、`prior_sensitivity` 面，全部只在这边做。
3. **迁移是重新表达，不是 `cp`。** 两个包的抽象不同（`State → State` +
   `ParameterSpace` + `SamplingPlan` ↔ 图节点 + 结构分派）。逐文件拷贝会静默
   丢掉那些活在 docstring 与测试里的正确性保证（offset 记账、秩下限、
   NaN 安全守卫）。
4. **两个实现一致不算证据。** 本仓库设计文档 §一已记录该教训：两条共享实现的
   路径一致地给出 −225.65，而真值是 −364.95。裁判只能是解析真值、第三方计算
   （手写 NumPyro 模型、scipy）、或变异测试。
5. **【新增】拿新实现去对齐旧实现，会把旧实现的缺陷固化成基准。**
   §三里每一条缺陷，必须在其模块 cross-check **之前**在 bayesmith 侧修好；
   或者把已知差异写成**带符号、带量级**的期望值断言。B1 与 B2 是两个活例：
   若不先修，两边会一致地错，而 cross-check 会为此发绿灯。

---

## 二、cross-check 协议

> **`docs/migration/` 在 2026-08-25 之前一直不存在**（上游交接 §5.14 记下的
> 那条）。P5 的三页是第一批：`identifiability.md`、`sensitivity.md`、
> `priors.md`。§六 被本节第 6 步卡住的那些模块，缺的正是这种页面。

每个模块按序执行并留痕，一模块一页，记在 `docs/migration/<module>.md`：

1. **选 fixture**：优先复用 rheplicant 自己的测试 fixture——它们是钉过的实测点，
   不是新猜的。每模块至少一个「健康」fixture + 一个**必须被拒绝**的 fixture。
2. **数值一致**：确定性出口断言到 float64 roundoff；采样出口在 MC 误差内
   （ESS ≥ 400 时 |z| < 4，沿用本仓库设计 §四的阈值）。
3. **拒绝一致**：rheplicant 侧每个响亮拒绝，这边有同形拒绝。异常类 identity
   按设计文档附录一的共享方案（rheplicant 有 52 处测试 import
   `ParameterSpaceError` / `StateValidationError`，identity 不能断）。
4. **独立预言机**（铁律 4）：解析真值 / 手写 NumPyro / scipy / 变异。
5. **记录有意的差异**：写明理由与等价性论证（例如 GLS 起点改变，用「固定点
   不依赖起点」证等价）。
6. **全部通过后**，才动 rheplicant 侧对应模块（§六）。

**追加硬项（铁律 5）：** §三中属于本模块的缺陷，在第 2 步之前已修，或已写成
带符号的期望差异。

---

## 三、必须在 cross-check 之前修掉的缺陷

| # | 缺陷 | 落点与做法 | 状态 |
|---|---|---|---|
| **B1** ✅ | **【已闭合 2026-08-28,e-RHINO `74fac09`。】**`Conditioning.neg_log_likelihood` = `0.5*chi2 + log_determinant`,**两个 potential 构造器都拿到它**——只修单参数那个会把同一个「两目标」缺陷在下一层重建(该变异实测 KILLED)。梯度块从 **6.2483** 移到 **5.0041**(无偏闭式 5.1046;余下 2.0% 是先验的雅可比,因为 fixture 用 `mu = exp(w) x` 声明 `w`,所以 `w` 上的 Normal 是尺度上的 `1/scale`)。`chi2` 本身**故意不动**:它是收敛监视器。附带一条裁决 **D55**:`include_logdet: false` 被 plan 两个出口**拒绝**而非静默覆盖。守卫:`tests/inference/test_potential_carries_the_logdet.py`(12 例,含跨缝对比 `to_numpyro_model` 的 `log_density`)、`test_noise_log_determinant.py`(11 例,钉住与 `NoiseModelLikelihood` 的精确关系)。以下为闭合前的原始记录。<br><br>**log-determinant 缺口。** rheplicant `inference/engines.py:118-140` 的 `chi2` 是 `Σ r²/σ²`，σ 在当前预测处求值；`conditional_potential`（`:164-179`）= `0.5·chi2 − log_prior`，**不含 `Σ log σ`**。而 `numpyro_bridge.py:279` 的观测 site 是 `dist.Normal(prediction, sigma)`，`log_prob` 自带 `−log σ`——桥自己的 docstring（`:188-196`）写着「a prediction-dependent sigma brings its log-determinant with it, and that is the point of routing it through here … The two answers differ」。于是 `depends_on_prediction=True`（RadiometerNoise）时，同一份模型：`nuts` 出口采全密度，`plan.sample` 的 gradient 块采 GLS 型目标，无守卫、无说明。该包自己在证据层把 full/GLS 之分上升到拒绝混存的高度（`compressed.py` 的 `estimator` 字段），引擎之间却沉默。 | **设计已在 P3b 就位，无需新设计**：`plans/2026-08-23-p3b-dispatch-execution.md:1630+` 的 `exact/correct.py`——把冻结项放回去的重要性/MH 权重，符号陷阱已对 scipy 实测（写反差 `+2.396e-01`）。本文追加的是**验收**，而本节初稿把它写错了对象，已实测更正（2026-08-24）：**`(1+f²)` 不是「冻结 σ」与「活 σ」之差，是「丢 log-det」与「留 log-det」之差。** 实测（κ=0.5，n=2e5，同一 fixture）：`argmax NoiseModelLikelihood(include_logdet=False)` = 4.9993169984，等于闭式 `Σd²/Σd` = 4.9993169980（九位）；`include_logdet=True` = 4.0004027234，等于闭式正根 `(−Σd+√((Σd)²+4nf²Σd²))/(2nf²)` = 4.0004027155（八位）；比值 **1.24970 对 (1+f²)=1.25，差 0.024%**。该正根代入大 n 矩恰好化为 μ₀，所以完整密度是**精确**无偏而非「偏得少一些」。而 bayesmith 的 `iterative_gls` 是**冻结 σ 的 IRLS**（每次内解 σ 不动，解完再更新），其不动点满足 `w = mean(u)`，`u = d/x`——**与完整密度同侧，不带这个偏差**。实测不动点到 `mean(u)` 的距离比到 `Σu²/Σu` 近 44–128 倍（κ∈{0.05,0.2,0.5,1.0}）。<br><br>所以验收是**两条**，分别落在两个仓库该落的地方：（i）`tests/crosscheck/test_noise_logdet.py` 钉住 rheplicant 的丢-log-det 估计量确实高偏 `(1+f²)`，含一条 `HomoscedasticNoise` 的反空洞对照（σ 不依赖预测时两者必须重合），全部闭式、不抽样；（ii）`tests/exact/test_gls.py::test_the_fixed_point_is_the_unbiased_estimator_not_the_gls_biased_one` 钉住 bayesmith 这边本来就在无偏侧。**按初稿那句写，测试会红，而最自然的「修法」是去掰一个本来正确的估计量——正是下面这句警告的事，只是枪口对错了方向。** **不这么做，`plan`/`engines` 的 cross-check 会把 GLS 型目标当真值固化。** | `[R2]` `[实测确认]` |
| **B2** | **Fisher/协方差通路不强制 float64。** rheplicant `inference/uncertainty.py:378-506`，`jnp.linalg.inv` 在 `:500`。与证据层严格的 x64 纪律不对称；`F = JᵀN⁻¹J` 平方了条件数，所以一个轻度病态的 float32 模型会给出静默错误的 Cramér–Rao。 | **(1) 已实测更正（2026-08-24）：把求逆放进 `with jax.enable_x64(True):` 是个空操作，而且是会伪装成已修好的那种空操作。** 两层原因，都实测过：① 上下文管的是在它**之下被 trace 的**东西，不是已经存在的数组 —— `jnp.linalg.inv(float32 数组)` 在上下文里返回的仍是 float32；② 就算强行 `astype(float64)` 也救不回来，因为 `F = JᵀN⁻¹J` **已经把条件数平方了**，位数在**构造 F 时**就花掉了。实测 κ(J)=1e3（即 κ(F)=1e6）：全 float32 误差 **2.41e-02**，只在求逆处上调 **2.45e-02**（无法区分，甚至略差），全 float64 **1.08e-12**。所以包在求逆处加一层 x64 会「报告缺陷已修」而误差棒仍然错 2.4%。<br><br>**正解，且它本来就是这个仓库的既有纪律**：`src/` 从不自己开 x64 上下文 —— `plan.py:189/459/469`、`linearity.py:498/541`、`solve.py:355` 全部是**拒绝并告诉调用者**去 `with jax.enable_x64(True):` 里**建图**。fisher 照此办理。<br><br>(2) 条件数门槛已落地，并且**按 dtype 推导而非写死**：`1/√eps`（float32 **2.90e+03**、float64 **6.71e+07**），即「求逆已花掉一半位数」那一点。写死 float64 的天花板会把 float32 那个**正是缺陷本身**的情形放过去。API：`max_condition="auto"|float|None`，`None` 关闭；jitter **先加再量**（jitter 正是本函数提供的唯一解药，量未加 jitter 的矩阵会拒绝一个已经把问题修好的调用者）。判据写成 `not measured <= ceiling` 而非 `>`，因为 NaN 条件数（矩阵含 NaN 或 inf 时 `cond` 返回 NaN）在 `>` 下会静默通过 —— inf 尤其危险：它反演成干净的 `0.0` 方差，看起来像一个被精确测定的参数，对输出做有限性检查也抓不到。<br><br>落点：`exact/fisher.py::condition_ceiling` + `parameter_covariance`，测试 `tests/exact/test_fisher.py`（9 条，含 `test_widening_only_the_inverse_does_not_recover_the_bound` —— 它把上面两层实测钉死，使得「按初稿把 inv 包进 x64」不能作为改进被重新引入）。**不要回补 rheplicant**（双写），那边已加 docstring 指过来（`inference/uncertainty.py::parameter_covariance` 的 Note）。**必须先于 §四「已在移植」表的 Fisher 行比对。** | `[互证]` |
| **B3** | `prior_sensitivity`（`inference/sensitivity.py:207`）的 damped Newton 对可能不定的似然 Hessian 无退路；模块 docstring `:24` 对 Δ 的方向表述有歧义（公式给 `Δ = θ̂ − μ`，句子字面读作 `μ − θ̂`）。恒等式本身正确（R2 独立重推过：由 `∇ℓ(θ̂)=P(θ̂−m)` 与 `∇ℓ(θ̂)=−H(θ̂−μ)` 联立即得）。 | 移植到 P5 时：`eigvalsh` 检查 + Cholesky-with-jitter 退路 + 一个不定 Hessian 的玩具测试；表述改写为无歧义方向陈述。 | `[互证]` |
| **B4** | `inference/npe.py:136-140` 的 `simulate_pairs` 用 `std()` 做**加性**散布而非调 `noise.realise()`，与自己 docstring 的「multiplicative, exactly as in the data」字面相悖。高斯对称下分布等价，但 `RadiometerNoise.floor > 0` 时两者分叉（`std` 施加 floor，`realise` 不施加）。 | 一行：`observed = noise.realise(prediction, key=…)`。**此项不依赖分离，rheplicant 侧现在就能修**——已列入 Track A 的 Batch 1。NPE 若将来进 bayesmith，先确认这一行已修。 | `[互证]` |
| **B5** | `1/σ²` 权重公式有两个家：`inference/linear.py:1421` 与 `inference/noise.py:339-349`，在 NaN σ 上分叉（共轭路径传播 NaN，`inverse_variance` 映射为 0）。文档称之为刻意，但它是「derive, don't re-spell」唯一一处刻意的破例。 | **已完成，但不是按本行开的方子——2026-08-26 实测更正。** bayesmith 确实收敛为**一个**家：`exact/precision.py` 的 `DiagonalPrecision.apply`，全仓库唯一一处做 `r/σ²`。但它**没有 `mask_nan` 形参，而且不应该有**。<br><br>那两种行为不是一个可以由标志选择的策略，是两种**不同的输入**：σ=inf 意为「该样本未被观测」，`1/inf²` 恰好是 `0.0`，`log_normalizer` 是 `+inf`，于是 log-density 是 −inf——无穷方差的样本没有密度，这是对的；σ=NaN 意为「你的 σ 坏了」，NaN 传播，这也是对的。实测（float64，r=[1,2,3]，σ=[1,x,0.5]）：x=inf 给 `apply=[1,0,12]`、`log_normalizer=inf`；x=nan 给 `apply=[1,nan,12]`、`log_normalizer=nan`。<br><br>**加一个 `mask_nan=True` 会把这两者合并**，让调用者能把「坏掉的 σ」当「未观测」静默处理——而那正是上游漏出来的一个 NaN 变得不可见的路径。B5 要修的分叉会以更难发现的形式回来：不再是两个家各有各的答案，而是一个家在一个布尔量下给两个答案。本行保留而非删除，是为了不被当成遗漏重开。 | `[已完成 2026-08-26]` |
| **B6** | `validate()` 在 jit 编译期重复跑（`engines.py:302` → `linear.py:637,657` → `parameters.py:822`）：与 θ 无关的 `eval_shape` 追踪与管线遍历在编译期执行。是编译成本，不是错数。 | **在新架构里自然消失**——图只 trace 一次、验证一次。无代码可迁，只需不要重新引入。 | `[互证]` |
| **B7** | Gibbs 诊断只有联合 χ² 单标量的 split-r̂，无逐参数 r̂/ESS。 | **已完成 2026-08-26**：`dispatch/execute.py` 的 `chain_diagnostics` + `SiteDiagnostic` + `r_hat_ceiling`，21 个测试在 `tests/dispatch/test_chain_diagnostics.py`，`Posterior.diagnostics` 承载。<br><br>**本行要求的「自己的阈值论证」做了，而结论比本行预期的强：****常数阈值根本不是一个良定义的检验**，无论取哪个常数。split r̂ 的零分布由链携带的独立信息量决定。实测（每格 4000 个独立坐标，27 格：chains∈{1,2,4}、draws∈{200,800,3200}、AR(1) ρ∈{0,0.5,0.9}，ESS 跨 14.8–12693）：`r̂₉₉−1` 按 `ESS^-1.05` 标度，`(r̂₉₉−1)·ESS` 在 ESS 变化 860 倍时只在 4.87–11.46 之间。于是固定 **1.05** 在 ESS=15 处对**已收敛**的链误报 **45.9 %**（1.01 是 69.9 %），在 ESS=12693 处误报 **0.0 %**——一端是抛硬币，另一端根本触发不了。所以天花板写成 `1 + C/ESS`，`C` 是被实测钉住的量；多重性并进 `C`（坐标独立，P 个的族错误率就是单坐标的 `0.99^(1/P)` 分位），实测 C 上界 11.46(P=1)→14.44(10)→18.76(100)→22.61(1000)→22.99(10⁴)，饱和于 23。<br><br>**第二个发现推翻了本实现的第一版，值得单独记：**「天花板」单独用是**几乎没有功效**的。它按零假设标定，而零假设下低 ESS 意味着「混得慢但在采对的分布」；备择假设下低 ESS 意味着「卡住了」——而错位的链把 ESS 压低的速度快过把 r̂ 抬高的速度，于是天花板升上去迎接它本该拒绝的那个统计量。实测两条相距若干 σ 的链、200 draws、2000 坐标、ρ∈{0,0.9,0.99}：**十五格全部溜过去**；最极端一格 r̂=**9.25** 而 ESS=**1.02**，天花板 **12.3**。**所以顺序是不可交换的：先卡 ESS 下限（100，Vehtari 等 2021 的惯例，此处只实测它对上述十五格足够），只有在 ESS 撑得住结论的地方才问 r̂。** 那十五格 ESS 最大 19.9，全部被下限拦下。<br><br>五个变异全部被具名测试杀死（退出码 1，每次清 `__pycache__`）：去掉 ESS 下限、非有限 r̂ 放行、去掉多重性项、天花板改成常数 1.05、不按 `num_chains` 拆分。 | `[已完成 2026-08-26]` |
| **B8** | `inference/chain.py:398,504,963-966,1080-1081` 用 `int(jnp.zeros(shape).size)` 分配一个数组去读静态尺寸。 | 重写证据层时用 `math.prod(shape)`。 | `[R1]` |

---

## 四、迁移台账

### 4.1 已在移植（`exact/` 已有对应物）

| rheplicant | bayesmith | 必须一致的内容 |
|---|---|---|
| `linear.py`（`LinearBlock`、`check_linearity`、`wiener_solve`、`gcr_sample`、`condition_estimate`） | `exact/block.py` `linearity.py` `solve.py` `conditioning.py` | 同一 `gain × T_ant` fixture（`tests/inference/test_degenerate_partition.py`）下 CG 解逐元素到 float64 roundoff；GCR 的均值**与协方差**在 MC 误差内；`condition_estimate` 同键同数；**全部守卫同形**：`κ·residual` 判据、`κ·eps > require_convergence` 时「收紧 tol 无用」的单独判决、1-D sigma 轴歧义拒绝、共轭缝上的 NoiseModel 拒绝 |
| `gls.py`（`iterative_gls`） | `exact/gls.py` | 同 fixture 下固定点 σ 一致；`converged=False` 触发条件一致；起点若改变，用固定点的起点无关性证等价 |
| `uncertainty.py` 的 Fisher 半 | `exact/fisher.py` | `F = JᵀN⁻¹J` 逐元素一致；`(1+2f²)` 辐射计修正。**先修 B2，否则两边一致地错** —— B2 已修（`aa644e0`）。**本行已实测，一半通过一半未移植（2026-08-24，`tests/crosscheck/test_noise_logdet.py`）：**<br>① `F = JᵀN⁻¹J` 在常数 σ 模型上**逐元素一致到 rtol=1e-12** ✓。（附注：Fisher **不读数据**，只由 Jacobian 和 σ 构成，所以 §0.1 那个 PRNG 坑够不着本行；要一致的是**设计矩阵**，测试里它是从 bayesmith 的图上读出来的，不是第二次拼写。）<br>② **`(1+2f²)` 已移植并已比对（2026-08-24）**：两边在预测依赖模型上**逐值一致到 rel=1e-7**，并带反空洞条款：一致的值同时必须是第一项的 `(1+2f²)` 倍 —— 两个都漏掉第二项的包会完美一致，那条测试就会在它本该抓的缺陷上变绿。<br>rheplicant 侧的因子实测**精确**（f∈{0.05,0.5,1.0}，十位吻合）。bayesmith 侧新增 `_log_sigma_curvature` 与 `fisher_information` 的三个参数，按 `iterative_gls` 的先例：`depends_on_prediction: bool = True`（**默认取安全侧**，且**只管 `sigma_of` 是否必需、不管该项是否施加** —— 因为常数 σ 下该项实测精确为 0，让标志同时管算术会给出两种拼法和一个无从裁决的矛盾），`sigma_of=`（`sigma_from_graph` 那个既有接缝；已决定的 dict 没有导数），`centre=`（`noise_std` 读取处）。<br>**`centre` 与 `noise_std` 互相核对而非互相信任** —— 二者按构造冗余，而未被比较的冗余正是「协方差在一处加权、在另一处取曲率」的来路。`exact/correct.py` 把同一危险记为 **UNVERIFIABLE**（`LinearBlock` 不记得自己建于哪个 `at`）；这里可验，所以验了。<br>方向仍值得记住：漏掉大于 1 的因子使 `F` 偏小、误差棒偏**宽** `√(1+2f²)`（f=0.5 时 22%，f=1 时 73%），**过于保守的预报读起来像安全的**。<br>代价：默认 True 使 **8 个既有调用点**必须显式声明 `depends_on_prediction=False`（它们本来就是常数 σ）。这是有意的：新调用点不可能默默拿到偏宽的误差棒。`src/` 无调用点。 |
| `likelihood.py` / `noise.py` 的高斯密度 | `exact/gaussian.py` | `−½Σ[r²/σ² + log 2πσ²]`；flagged σ=∞ 的干净零贡献，且以 **mask** 实现而非让 `inf` 传播 |

### 4.2 尚未开始（P3b 后半与 P5）

> **⚠️【标题已过期，保留而不改写；答案写在下面每一行的第三栏。实测 2026-09-03。】**
> 本节**七行全部已落地**。三行（`identifiability`、`sensitivity`、`priors`）
> 从 2026-08-25 起就带着「已落地」字样；另外四行同一天也做完了，只是没有人
> 回来写——现在写上了。最后一行是 `numpyro_bridge`（`8199377`，2026-08-25），
> 它的记录页第一句就是「**这是 §四 的最后一行，闭合它就打开 §六 的门**」。
>
> 不改标题，是因为「尚未开始」曾经为真，而本仓库的规矩是把过期的话留在原处、
> 在旁边写清它为什么不再适用。但**只读标题就决定要不要往下读**的人会得到一个
> 反的结论，所以答案放在这里，而不是留给读者去数七行。
>
> 一条不必读七行就能核的旁证，而且它是可执行的：
> `tests/test_migration_records.py` 的
> `test_the_gate_on_section_six_is_open_and_every_module_has_a_page`
> 在 2026-08-25 **把断言翻了向**——它原先断言「只要还有 §四 模块没有记录页，
> §六 就不许开始」，现在断言「每个 §四 模块都必须有记录页」。那次翻向就是本
> 节归零的那一刻。

| rheplicant | 目的地 | 必须一致的内容 |
|---|---|---|
| `parameters.py`（`Latent`/`Bind`/`ParameterSpace`/`validate`/`refuse_stochastic_stages`） | 节点声明层（`linear_in`） | 语义映射而非逐行移植。最小集：三种绑定形态（derived/tied/direct）在同一 toy pipeline 上给出相同预测；`refuse_stochastic_stages` 有等价物（这边表述为「无密度的随机节点不能进联合分布」）——**理由改写，行为不得变**。<br><br>**已落地（2026-08-25，`d58a079`）**：三种绑定形态在同一 toy pipeline（4×5 网格、两级乘性）上给出的预测**逐位相同**；tied 的可观测后果单独钉住（增益进平方，3 → 9，而这个 9 是两个包都没算过的独立算术——防的是「tied 只是把一次乘法改了个名」这种空洞通过）；FAN fixture 直接用 rheplicant 自己的 `TestFanOut`（`t_physical = 0` 使效率成为纯乘法，向量取故意不对称的 `[2, 5]`，因为对称向量会让两种读法一致从而致盲整个比对），两侧读数 4.0 与 10.0 逐一复现。<br><br>**本行真正的工作落在第二个分句上，而它比本行预期的重：那个行为这边当时还不存在。** `refuse_stochastic_stages` 的等价物是为本行写的——`Graph.__check_init__` 拒绝持有 PRNG key 的 `Const`，异常映射 `ParameterSpaceError` → `GraphError`。三个变异全部被具名测试杀死。记录：`docs/migration/parameters.md`；测试 `tests/crosscheck/test_parameters.py`。 |
| `noise.py`（协议、`RadiometerNoise`、`FlaggedNoise`、`NoiseModelLikelihood`、`check_noise_std_axis`、`inverse_variance`） | 概率节点（`dist_fn`） | 三个噪声模型 × 有/无 flags：log-density 与抽样分布一致。`FlaggedNoise` 的 σ=∞→零权重必须是 **mask**。**顺序：§五 B9（相关噪声）会改这一层的接口形态，先定接口再定稿本模块**。<br><br>**已落地（2026-08-25，`e971e53`）**，且这条排序条件当时已经满足——B9 先落，协方差以 `Precision` 到达每一个消费者，对角是那个退化情形而不是一条平行通路。fixture 的预测**故意穿过零**（`[3, −2, 5, 1, −4, 2.5]`）：那是辐射计的乘性生成器与一个加性生成器唯一分叉、`RadiometerNoise.std` 的 `abs` 唯一承重的区间。常数 σ 密度**五种拼法末位相同**（−5.846065603244213）；flagged 密度**四种拼法相同**（−3.5202942825891324）；辐射计密度三种拼法差一个 ULP（`rel=1e-15`——两侧以不同表达式到同一个 σ、求和次序也不同，所以不是逐位，而且记录页把这句诚实地写了出来）；`inverse_variance` 对 `Precision.apply` **逐位相同**，flagged 项正好 `0.0`，即本行要求的「σ=∞→零权重必须是 **mask**」；`HomoscedasticNoise.realise` 与图上节点的抽样在同一 key 下**逐位相同**，即本行要求的「抽样分布一致」。<br><br>**反空洞条款：**`include_logdet=False` 必须是**另一个数**，差额等于 `−½Σ log 2πσ²` 且符号固定——没有这一条，那张密度对照根本没有在测 log-determinant，而 log-determinant 正是 B1 的全部主题。记录：`docs/migration/noise.md`（与 4.1 的 `likelihood`/`noise` 行同页，因为两行都点名同一个模块）；测试 `tests/crosscheck/test_gaussian.py`。 |
| `plan.py` + `engines.py` 的 Gibbs | P3b 分派执行 | 同 partition 同 toy 模型下 `plan.estimate` 逐值一致；`plan.sample` 比后验矩（χ² 迹线跨 NUTS 实现不可比）。**先落 B1**。<br><br>**已落地（2026-08-25，`d2ca7fe`）**：`plan.estimate` 的三个系数**绝对差 8.9e-15**（相对 ~1e-15），两侧各自对稠密预言机 ~1e-14；`plan.sample` 均值 \|z\| = 0.89（上游）与 0.72（本侧）；`plan.sample` 方差落在预言机的 1.6% 与 6.3% 内，对照的是 5.0% 的抽样误差；预测依赖 σ 上的共轭块两包差 **9e-12**。**不是逐位，而且记录页把理由写了出来**：两侧跑的是收敛到同一个固定点的不同迭代格式（块坐标下降 对 单次重加权求解），float64 roundoff 才是能立住的断言，「逐位」这个说法在这里根本不可用。抽样那一栏比的是**稠密预言机**而不是另一个包——同一个 key 下两次 Gibbs 扫过的状态序列本就不同，能要求一致的是它们保持不变的那个分布。<br><br>**本行的排序条件「先落 B1」在本行做不到，而查清为什么做不到，是本行的主要结果**（记录页 §5(a)）。**B1 自己在 2026-08-28 闭合**（e-RHINO `74fac09`，见 §三 B1），比本行晚三天——本行落不了它，但本行是定位到它的地方。记录：`docs/migration/plan.md`；测试 `tests/crosscheck/test_dispatch.py`。 |
| `identifiability.py` | P5 `diagnose/` | 同一退化模型的 rank/nullity 相同；`IdentifiabilityReport.direction` 逐分量一致。**必须重测 `rtol=1e-8` 的谱隙论证**——rheplicant 的论证建立在进程全局 x64 上（实测：null 方向 6.6e-17、最弱可辨识方向 4.8e-5、SVD 噪声底 ~1e-14；float32 下 null 方向浮到 3.1e-8），上下文管理器的 dtype 边界不同，谱可能略变。**已落地（2026-08-25）**：常数重测而非搬运——本侧机制下 null 方向 **7.479e-17**（谱确实动了，判决没动）、最弱可辨识 **4.822138e-5**（逐位相同）、float32 null **3.116759e-8**；1e-8 仍成立，理由文字已换成本侧实测。四行表逐格一致；方向按符号固定后逐分量 1e-9，八维 null 空间按投影算子比对。记录：`docs/migration/identifiability.md` |
| `sensitivity.py` | P5 `diagnose/` | 闭式 `Δ = H⁻¹P(m−θ̂)` 与重拟合两条路线在 tour 模型上复现 rheplicant 的实测表（0.0069σ 等）；同一 pass 修 B3。**已落地（2026-08-25）**：tour 表逐项复现（mode、shift、sigma_post、criterion 0.0795、七级 s-ladder），并与 rheplicant 活体报告逐字段比对；B3 修法为 eigvalsh 判定 + Cholesky-with-jitter（shift 取 2·\|λ_min\| 反射式——只加过零的 shift 会返回 2e7 长度的方向）。**一个移植暴露的语义发现**：秩拒绝的裁决对象从观测 Jacobian 挪到 rest 项自身曲率——图里被选 latent 可以只被下游 latent 密度约束（`child ~ Normal(parent, s)`），似然 mode 存在而观测 Jacobian rank 0，rheplicant 的平坦结构表达不出这种情形。记录：`docs/migration/sensitivity.md` |
| `priors.py`（`JeffreysPrior`） | P5 `diagnose/` 或桥层 | 复现 RadiometerNoise 下 Jeffreys 退化为平坦先验的实测常数 **+15.80169853**——既是回归测试，也是对「先验形状由噪声模型选择」这一语义的 cross-check。保留 `eigh` + 秩下限；**不得**换回 `slogdet`/Cholesky（病态块上它们给出貌似合理的有限值）。**已落地（2026-08-25），取 `diagnose/`**：常数九格逐位复现（对 rheplicant 活体、对 numpy 独立闭式各 1e-8）；奇异块的两个"貌似合理"数也逐位复现（slogdet +6.420496、cholesky pivot 9.755e-05），eigh+floor 给 −338；噪声改从图读取，行序保留 over= 顺序（sorted 之疣不移植）；今日消费者是 `numpyro.factor` 模式（套件演示），图上声明 + 桥集成留给 numpyro_bridge 行。记录：`docs/migration/priors.md` |
| `numpyro_bridge.py` | `bridge/numpyro_bridge.py`（已有） | 补三条 rheplicant 特有的：`init_to_declared` 等价物（实测 r_hat 840 vs 1.002——**带过去的是教训不是代码**）、`predict_from_samples` 的形状守卫、Jeffreys factor site 的「密度只加一次」。<br><br>**已落地（2026-08-25，`8199377`）——这是 §四 的最后一行，闭合它就打开了 §六 的门。** 本行**不做数值对照，而且记录页把为什么写清楚了**：两个包各自构造一个 NumPyro 模型、交给同一个 NUTS，逐值比对等于拿 NumPyro 和 NumPyro 比。比的是上游点名的那三种失败模式下的**行为**。三条里**有两条在本侧是可达的失败，本次一并修掉**：`predict` 的每样本形状守卫——漏网的恰好是方阵（长度 3 的 latent 抽 3 次，转置后形状不变，守卫看不见它），以及 Jeffreys 的「密度只加一次」，由 `JeffreysPrior._check_against` 接住。唯一被比的那个数是 Jeffreys factor 的贡献：联合 log-density 只按先验自己的那一项移动、不多不少，对独立求值的 `JeffreysPrior.log_density` 到 `rel=1e-12`——是一次减法，不是去 trace 里看一眼。预言机是手写算术（三次抽样下 `c ⊙ x` 的 `[[0, 2, 6], [3, 8, 15], [6, 14, 24]]`），**连转置后的答案也一并写出**，那是「静默的那一格确实静默」的证据。三个变异全部被针对它们具名的测试杀死。记录：`docs/migration/numpyro_bridge.md`；测试 `tests/crosscheck/test_bridge.py`。 |

### 4.3 不迁移

> **⚠️【本节部分作废,2026-08-29;而这段话本身先写错过一次,见下。】**
>
> 本节有**三条**,状态各不相同:
>
> | 条目 | 状态 |
> |---|---|
> | `calibrate.py` | **已作废**,由 **D11** 点名推翻(owner 2026-08-27);已切 |
> | **整个流式证据层**(八个模块,含 `reduced_basis.py`) | **仍然有效**——它说的不是「不做」,是「**不移植,在 bayesmith 侧通用化重写**」,归 **§五 B11** |
> | `npe.py` | **已作废**,由 **D10** 点名推翻(owner 2026-08-27);已切 |
>
> 上位是 2026-08-26 的「未迁移的全部迁移」,但它推翻的是「**不迁移**」,
> **不是**「不移植而重写」——后者本来就是迁移的一种形态,`sqrtinfo` 那一页
> 正是这么记的。
>
> **【2026-08-29 再次收窄,这次靠的是一个已执行的先例而不是论证。】**
> 上面这句原本写作「**不要按 `calibrate`/`npe` 的先例直接切**」,**那说过头了**。
> 本条名下的八个模块里,**`reduced_basis.py` 已经切了**——它同时出现在本条的清单里
> **和** §五 Wave C 那一行,做成**半切**(5 个数组级名字委托,3 个函数 + 2 个容器
> 留守),理由逐条登记在 **D60**。所以:
>
> * 本条**不禁止**近端委托到那个「重写出来的」远端面——`reduced_basis` 就是这么做的;
> * 本条**确实**排除把它们当成 `calibrate`/`npe` 那样的**整体**切换:
>   这一族以容器为主、受 **D12** 管辖,**大部分留守**,而能过去的是数组级内核。
>
> **判据仍是 D42/D31 的那一把尺**(签名碰不碰 `ParameterSpace`/pipeline/`State`/
> `NoiseModel`/D12 容器),逐名量,不逐模块猜。Wave D 的开波实测见
> `2026-08-29-wave-D-opening.md`。
>
> ---
>
> **这段标注的第一版说「本节只有两条,两条都已作废」,那是错的**,而错法是
> 本会话第三次犯同一个病:我的自动核查找**加粗的模块名**,而证据层那一条
> 加粗的是「**整个流式证据层**」这个中文短语,八个文件名在里面用普通反引号写着。
> **匹配器只认一种写法,其余读成「不存在」,而「不存在」读起来像「查过了」。**
> 前两次是私名普查的假阳性、和漏掉大写的 D10;**这一次它进了一条已发布的断言**,
> 而那条断言会让人去切证据层的模块。
>
> **给下一位的三条(第 3 条是这次新踩的):**
> 1. **搜推翻要不分大小写**(`grep -rin`),并同时试模块名、去 `.py` 的词干、
>    大写/英文别名。D10 标题写作「NPE 迁移」,按 `npe.py` 搜会漏。
> 2. **一节可以整体过期,也可以只过期一部分。** 查栏位时数一下这一栏还剩几条。
> 3. **数条目要用眼睛,不要只用正则。** 一个条目可以把模块名藏在一句话里、
>    藏在括号里、或者一条讲八个模块。**先把整节读完,再写「一共几条」。**


- **`calibrate.py`** ⚠️ **【本条已作废。2026-08-26 owner 裁决「未迁移的全部迁移」,
  2026-08-27 **D11** 点名授权迁移本模块;已于 2026-08-29 切换(e-RHINO `2c18744`)。
  下面整段保留而不删除,因为读到它的人需要看到它为什么不再适用。】**
  它的第一句前提写于 2026-08-24,**今天为假**:`optimize.py` 在 P2 落地并随 **0.5.0**
  发布。D11 给的迁移理由正好相反:「把 calibrate 留在 rheplicant 会造出**两个梯度
  MAP 实现**,直接违反本计划的一份实现法则」。
  **这一条曾被当作现行依据用过一次,并据此回滚掉了已授权的工作——见 D58。**

- ~~**`calibrate.py`**~~（`GradientCalibrator`/`AdamCalibrator`）：bayesmith 没有独立
  calibrator 层，估计出口是分派表的产物。rheplicant 侧在分离完成前原样保留
  （config 的 `optimize` kind 依赖它）。去留见 §五 D1。
  *附注：rheplicant `calibrate.py:45-80` 的入口守卫——先读声明再在完美预测处
  **实测**判定 loss 的方向（误差函数取最小、对数密度取最大，实测 g=−30.7 vs
  真值 1.0）——是对「minimize/maximize 签名相同」这一 Python 痼疾我见过最干净的
  解法。即使不迁移这个模块，这个手法值得在 bayesmith 任何接受用户 callable 的
  地方复用。*
- **整个流式证据层**（`archive.py`、`memory.py`、`compress.py`、`compressed.py`、
  `chain.py`、`factorize.py`、`reduced_basis.py`、`diagnostics.py`）：按铁律 2
  在这边**通用化重写**，见 §五 B11。
- **`npe.py`**：不在既定路线图。若要做，先确认 B4 已修。

---

## 五、新能力（只在 bayesmith 做）

### B9 — 相关 / 非对角噪声 `[互证]`，**排在流式证据之前**

rheplicant 全部噪声模型是逐样本对角：协议就是 `std(prediction) -> sigma`，
形状同预测。没有任何相关噪声通路——无 1/f 增益漂移、无大气相关、无通道间
协方差。而 Fisher、GCR、NUTS、证据压缩全都消费同一份 `1/σ²` 权重，于是**共享
同一个独立性假设**。对真实辐射计而言这是全包最显著的物理通用性缺口，比任何
引擎选择都重要。

**接口推广**（R2 的形式，采纳）：把 `std → σ` 换成两个操作——

- `apply(residual) -> N⁻¹r`
  （对角退化为逐样本除法；**circulant** 为 FFT；1/f 为频域滤波）
- 推广的 `log_normalizer()`（对角即现有的 `Σ log 2πσ²`；**circulant** 用
  FFT 本征值求和）

**实测更正（2026-08-24）：上面原写「Toeplitz 用 FFT」，那是 circulant 的性质。**
DFT 精确对角化的是 **circulant** 矩阵，不是 Toeplitz。同一个 8 点对称核实测：
circulant 的首列 FFT 逐位复现其本征值、`C⁻¹r` 与 log-det；用同一核构造的
Toeplitz 矩阵，FFT **根本不给出它的本征值**。所以落地的类叫
`CirculantPrecision`，名副其实；真正非周期的 Toeplitz 是另一个对象，用 FFT 处理
它是一个**必须自我声明的近似**，因此**不提供**，而不是用一个暗示精确的名字提供。

circulant 也正是推动这件事的物理的正确模型：1/f 漂移与大气相关都是平稳的，
周期边界就是「在这一段上平稳」的诚实表述。

**接口取 `at(prediction) -> Precision` 的形态**（而非方法带 `prediction` 形参）：
这样「一个对象喂三个消费者」是**构造上**成立的，不是靠调用纪律 —— 见下条。

`depends_on_prediction` 与 `realise` 语义不变。

**R1 补上的纪律，是真正承重的那条**：**一个对象必须同时喂给 `log_joint`、
`wiener_solve` 与 `fisher_information`**，使协方差与似然不可能各说各话——B1
正是这条纪律被违反的实例。

**连锁影响**（每环都有 rheplicant 的既有答案可借）：正规算子 `AᵀN⁻¹A + S⁻¹`
变为算子作用（CG 本来就不成形矩阵，天然兼容）；GLS 固定点不变；NUTS 经节点
`log_prob` 不受影响；证据层的白化行由 `r/σ` 变为 `L⁻¹r`，QR 体系不变。

**验收**：对角特例必须**逐数值退化**为现有实现（这就是它的 cross-check）；外加一个小尺寸 circulant 对直接矩阵求逆的预言机。

**增量 1 已落地（2026-08-24，`exact/precision.py` + `tests/exact/test_precision.py`，19 个测试）：**
接口（`Precision` 协议、`quadratic`、`log_density`）、`DiagonalPrecision`、`CirculantPrecision`、`dense()`（按**作用**materialise，供预言机用）。
- 对角退化对照的是**旧的字面表达式**（`r/σ²`、`Σ log 2πσ²`），不是同一想法的第二次实现 ——一个公式的两种拼法即使都错也会一致。
- circulant 预言机与实现**零共享**：scipy 造矩阵、numpy 稠密求逆、numpy 取 slogdet。
- 两个实现在重叠处必须相遇：核为 `[s,0,0,…]` 的 circulant **就是** `sI`，两条路必须给同一个数。
- 反空洞：另有一条断言 fixture 的**非对角项确实存在**，否则整个 circulant 类在一个退化成对角的核上会全绿而什么都没测。
- 构造时拒绝非正定核（FFT 对不定协方差会**给出有限答案**：负本征值让 log 出 NaN、让该模上的残差被推向**相反方向**，下游任何有限性检查都抓不到）。
变异测试：`apply` 乘而非除杀 5、对角除 σ 而非 σ² 杀 5、不定核放行杀 1、`quadratic` 忽略 precision 杀 3。`Σlog(2πλ)` ↔ `n log2π + Σlogλ` 是**等价变异**（同一恒等式）。

**增量 2 已落地**（本行原写「未做」，2026-08-26 实测更正）：12 个既有消费点已经接进。`fisher_information` 直接收 `precision: dict[str, Any]`（`exact/fisher.py:237`）；`normal_operator` 的文档字符串指向 `precision.quadratic`（`exact/solve.py:58`）；证据层的白化行经 `evidence/campaign.py` 的 `precision_at` 走同一接口；`diagonal_from` 在 `dispatch/classify.py`、`exact/gls.py`、`exact/fisher.py` 均有调用点。

> `docs/correlated-noise-proposal.md` 的 5.1b–5.1f 记的是增量 4 与 5，也就是本行之后两步的事。两份文档就此打架了一天，而**打架的方向恰好是本文更保守**：读到「未做」的人会去重做一件已经做完的事，并且在动手之前不会有任何东西告诉他。这正是「一个事实只有一个家」那条规则要防的形状——增量进度住在提案文档里，本文引用它，不再各记一份。

**为什么排在证据层之前**：压缩格式的白化行形态取决于这个接口。

### B10 — 非高斯后验似然 `[R1]`

`Likelihood` 协议 `(prediction, observed) -> scalar` 本身完全通用，
`calibrate`/`optimize` 接受任意 loss——但**后验**引擎
（`Conditioning`/`conditional_potential`、`gcr_sample`、`to_numpyro_model`）
全部硬编码了高斯观测 site。Poisson / Student-t 似然能驱动优化，却不能产出
抽样，也进不了 `SamplingPlan`。bayesmith 的 `Probabilistic` 节点已经是对的缝：
加密度类型，让分派决定「能精确就精确，否则 NUTS」。
**验收**：一个非高斯节点能产出**抽样**，而不只是一个 loss。

> **【已闭合 2026-08-25，`074848a`。答案写在这一行下面，因为问它的是这一行。】**
> 两个具名测试，都在 `tests/dispatch/test_acceptance.py`：
> `test_a_poisson_observation_produces_draws_and_they_match_quadrature` 与
> `test_a_student_t_observation_produces_draws_on_both_of_its_latents`。
>
> Poisson 那条挑的是**方差跟着预测走**的形状——Poisson 的均值就是它的方差——
> 所以它不只是「一个非高斯密度」，而是精确通路装不出来的那一种；分派对它给出
> `plan.exact is None`，路由到 NUTS。预言机是那个文件自己的 `log_joint` 网格
> 求积，与被测对象共享模型、不共享别的；比的是后验标准差，那是「采样器找到了
> 对的分布」唯一有意义的尺度。
>
> **闭合前它的状态不是「没做」，而是「做了但从没有人跑过」，这个区别才是本条
> 留下的东西。** 分派表自分类器存在的那天起就已经这么路由，但**没有任何测试把
> 一个非高斯观测节点一路带到抽样**，于是这条验收靠的是**读**分派表而不是**跑**
> 它。那个测试文件的注释把这句原话记了下来——「Assumed to hold, with no guard」
> ——而那正是本仓库反复记录的那种绿灯。

### B11 — 流式证据，从图重写 `[互证]`

README 的第四条头条能力，也是 config 完全够不着的子系统（`campaign:` 保留并
拒绝）。两位评审一致认为应重写而非搬运。

- **输入升级**：rheplicant 靠手工声明的 `Factorization` scope
  （global/per_epoch/linked）；bayesmith 的图自带 plate 与依赖结构，per-epoch
  nuisance 应表达为 plate 轴上的链节点，factorization **由图推导**——这消灭了
  `factorize.py` 存在理由的那整个错误类别（同一个空间被声明两次）。
- **必须原样保留的数值内核**（两份评审逐条核过全部正确）：SqrtInfo 的 `[R|z]`
  形式；QR 合并的 `−½Σρ²` corner；边缘化常数
  `+½ n log 2π − Σ log|R_ii| − ½ρ²`；σ=∞ 样本的 masked 归一化；千 epoch 的
  opaque-leaf 归档（实测 12,007 → 8 leaves）；x64 上下文纪律。
- **整套照搬的预言机**：`tests/evidence/test_streaming_equals_batch.py`
  的「流式 == 批处理到 roundoff」是总预言机。逐常数的 nat 成本表——丢一项差多少
  nat，实测值：初始先验归一化 +0.9189、逐转移 `−½logdet(2πQ)` +2.8618、
  边缘化常数 +7.2619、fold corner +45.9502、masked 归一化 −6.8408——正是铁律 4
  要求的独立裁判。全部复现。
- **保住诊断的诚实形状**：`diagnostics.py` 对 in-span 相干误差的拒绝式设计
  （「这类误差数据永远看不见，所以 §9.4/9.5 是声明式拒绝而非统计量」）是设计
  哲学，不是实现细节。

> **D90（2026-08-30）：本条的「原样保留」已由「两份一致」变为「单处存在」，
> 就地改判。** e-RHINO `b87e44f`（2026-08-28）把 `marginalise_arrays` 的
> Schur complement 委托给 `bayesmith.marginal.sqrtinfo`（切换前实测四个返回
> 数组 bitwise 相同；五个 refusal 连同上游自己的异常类与措辞留在近侧），
> `marginalise` 自此是壳（命名→置换、offset 穿线、pivot 读取）套共享内核。
> 按铁律 2 后半句，`tests/crosscheck/test_sqrtinfo_agrees.py::
> test_marginalise_arrays_agrees_bitwise` 应同批退役——实际晚了两天
> （2026-08-30 才执行），这两天它一直在拿本包比本包，而文件级 SWITCHED 登记
> （`tests/test_migration_records.py`）按构造看不见符号级切换：`sqrtinfo.py`
> 只切换了一个函数，文件还在，文件级断言就还绿。空窗由新守卫补上：
> `tests/crosscheck/test_provenance.py` 以同模块 AST 可达性**双向**断言每个
> 比较对象的归属——OWN 表不得触达 bayesmith（允许共享的阈值常量按实测
> exact set 逐名列出），SHARED_KERNEL 表必须继续触达；e-RHINO 若撤销某项
> 委托，失败信息指回本条与相应 route-comparison docstring，而不是让空转的
> 比较悄悄复活。内核的独立预言机本来就在这边：`tests/marginal/
> test_sqrtinfo.py`、`tests/marginal/test_streaming_equals_batch.py`
> （上一条 bullet 写的 `tests/evidence/` 是 0.5.0 改名前的旧路径，原文不动）。
> `combine`/`null`/`log_prob` 两侧仍各有实现，比较继续，且现在有守卫钉住
> 「仍各有实现」这个前提本身。

### B12 — `prior_sensitivity` 的 config 面 `[R1]`

rheplicant 无对应 run kind，只有 Python API。连同 B3 的修正，在这边实现，
含其 config 面。

> **本条上半句被实测推翻，下半句无法按字面执行（2026-08-25 重定范围）。**
> `tests/crosscheck/test_spec_claims.py` 把这条留作台账，所以上面的原文
> **不动**；下面是执行时该读的版本。
>
> - **"只有 Python API" 是半假的。** rheplicant 侧 `prior_sensitivity`
>   已有完整 config 面，形态是**受控检查**而非 run kind：四个模式经真实
>   gate 生效，非法值给出 `Finding(check='A1', severity='refuse')`，默认
>   `off`，id **C19**，接线在 `config/postflight/fitting.py:469`；GUI 以
>   通配 `inference.checks.*.mode` 承载，所以按 `prior_sensitivity` 去 grep
>   catalog 什么都搜不到、读起来像不存在。真正为假的只有 "无 run kind"
>   这一半——那一半是真的。
> - **"含其 config 面" 在这边没有可执行对象**：bayesmith 根本没有 config
>   层，而 D1 的建议是 config 语法留在 rheplicant。
> - **B3 已经做完了**（§八 第 5 步，`diagnose/sensitivity.py`），所以本条
>   剩下的实质工作只有"决定要哪种形状"。三个选项：(a) 什么都不做——Python
>   API 就是这边的面，rheplicant 的 C19 检查改调 bayesmith；(b) 在
>   rheplicant 的 config 层加一个真正的 run kind，消费 bayesmith 的
>   `prior_sensitivity`；(c) 等 bayesmith 有第二个消费者时再谈 config 层。
>
> **【owner 已拍板 2026-08-25：取 (c)。】** 暂不动 config 层。与 D1 的建议
> 一致（config 语法留在 `rheplicant.config`），理由也一样：把语法搬进一个
> 通用推断库，要么被迫通用化、要么两处重复，而现在**只有一个消费者**——
> 没有任何东西能告诉我们哪些部分是通用的。等 bayesmith 有第二个消费者时
> 重估。
>
> 因此 **B12 到此为止，无代码可写**：B3 已完成（§八 第 5 步），rheplicant
> 侧的受控检查 C19 保持原样，bayesmith 侧的面就是 `diagnose/sensitivity.py`
> 的 Python API。

---

## 六、rheplicant 侧收尾（cross-check 全绿之后）

1. 在此之前，**`src/rheplicant/inference/` 一行不动**——例外只有两项，且都已
   列入 Track A 的 Batch 1：B1 的 `plan.py` docstring 补写（它今天只写了
   conjugate 块的冻结-σ 情形，应补上 gradient 块同样不含 logdet），以及 B4 的
   一行修复。任何指向 bayesmith 的 docstring 指针也算例外。

   > **【本步骤已被 2026-08-26 的 owner 裁决整条取代，原文保留而非删除。实测
   > 2026-09-03，e-RHINO `27e621b`。】**「一行不动」的有效期止于当日的「未迁移
   > 的全部迁移」：Wave A–D 与 D10/D11 授权了成批切换，今天
   > `src/rheplicant/inference/` 的 **28 个 `.py` 文件里有 16 个** import
   > bayesmith：**顶层 12 个**（`calibrate`、`chain`、`diagnostics`、`gls`、
   > `identifiability`、`linear`、`loglinear`、`npe`、`partition`、
   > `reduced_basis`、`sensitivity`、`sqrtinfo`），另有 **4 个把 import 推迟到
   > 函数体内**（`graph_bridge`、`numpyro_bridge`、`priors`、`uncertainty`）。
   > 分开数是必要的：只报顶层会得出 12，而漏掉的 `graph_bridge` 正是 §六 步骤 3
   > 所依赖的那个 adapter。那条执行计划的家是 `2026-08-26-one-implementation.md`。
   >
   > **本步骤点名的两项例外都已兑现，答案写在这里而不是留在别处：**
   > B4 的一行修复已落地（e-RHINO `d499171`，2026-08-24：`npe.py:165` 改用
   > `noise.realise(prediction, key=…)`，docstring 同步改写成「the draw is taken
   > with the model's own ``realise``, so the simulator and the likelihood cannot
   > disagree about the law」）；B1 的 `plan.py` docstring 补写随 B1 整条闭合，
   > 今天那份 docstring 明写 gradient 块从 **6.248269** 移到 **5.004059**、
   > 并载着「**It is the BLOCK TYPE that decides, not the exit**」那张四行表——
   > 它同时更正了本文与它自己都曾把这件事说成「`plan.sample` 做的事」。
2. **【owner 已拍板 2026-08-26：本步骤重写为「两个包各自成立、互相比对」。】**

   原文是「`inference/` 转薄壳（重导出 + `DeprecationWarning`，或设计文档附录
   一验证过的 `sys.modules` 别名），config 的 18 个 run kind 改指 bayesmith」。
   **两半都不做**，各有各的理由：18 个 run kind 是 D1 取 (a) 已经决定不动；
   薄壳是实测做不了，且在**能**做的地方有害——证据保留在下面的实测记录里。

   **新形态，三条：**

   - **(a)** `src/rheplicant/inference/` 保持为**这台仪器自己的**贝叶斯层。
     不转薄壳、不重导出、不加 `DeprecationWarning`、**不废弃**。
   - **(b)** bayesmith 保持为**不含射电天文的通用包**。
   - **(c)** 两边重叠的能力靠 **cross-check 保持一致**，而不是靠共用实现。

   **这条决定最重要的后果是它改变了 cross-check 的地位，而这一点很容易被
   漏掉。** 原本 cross-check 是**过渡性**的——「验证移植是对的，然后删掉原
   件」；现在原件不删，于是它成了两份实现之间**唯一的长期保证**。过渡性的
   检查可以手工跑一次就算数，长期保证不行。

   **由此产生步骤 2 剩下的唯一实质工作：让 cross-check 能自动跑。** 今天跑
   不了——`tests/crosscheck/` 在没有 rheplicant checkout 时**静默 skip**（实
   测：publish workflow 那次运行里 120 个 skip），而**会 skip 的守卫不是会
   通过的守卫**。这笔学费 rheplicant 已经付过：`test_readme_counts.py` 在缺
   两个包的机器上 skip 了好几周，后面藏着三个真失败。

   能解决它的条件恰好是这次发布带来的：两个包现在都能按 URL 装上，所以
   cross-check 可以在 CI 里真的跑起来，而这在 2026-08-26 之前不可能。

   **已落地（2026-08-26，`.github/workflows/crosscheck.yml`）：123 passed / 0
   skipped / 0 failed**，在 Linux runner 上、对着 rheplicant 的 **main**。三
   条设计决定各自有实测理由：

   - **装 main 而不是装 PyPI。** rheplicant 发布的 0.2.0 落后它自己的 main
     **385 个提交、七万行**，而版本号一模一样——按名字装等于对着一个没有开发
     者在跑的东西比对。workflow 因此打印解析出的 commit，让日志永远能回答
     「这次比的是哪个 rheplicant」。
   - **`--no-deps`**，沿用 §0.1 的实测决定。
   - **必须证明它真的比过东西。** `tests/crosscheck/conftest.py` 的
     `importorskip` 在 rheplicant 缺席时让整个目录站下来而 pytest 仍退出 0，
     所以 job 从 junit XML 读计数，比过 0 个就红。

   第一次在笔记本以外跑就抓到两件事：一条 cross-check 在 runner 上拿到
   `nan` 的 ESS（链卡得比断言所能表达的还彻底，而 `nan < 10` 是 False），以
   及 11 条因缺 `rhino-cal-jax` 而 skip 的 sensitivity 行。两者都已修，后者
   靠把那个可选依赖也装上。

   - **(d)** 双向 docstring 指针。rheplicant 侧已加在
     `inference/__init__.py` 的模块 docstring（e-RHINO `7acf995`）：写明有
     这个兄弟包、重叠部分是**被比对**而不是被取代、这边没有任何东西被废弃、
     以及该拿哪一个（模型是这台仪器就用这边，不是就用那边）。

   **什么会推翻这条决定**：bayesmith 出现**第二个消费者**。理由与 D1、B12
   一字不差——只有一个消费者时，没有任何东西能告诉我们哪些部分是通用的。
   届时「一份实现」重新变成一个值得问的问题，而那时要造的是**适配层**
   （pipeline + `ParameterSpace` → `Graph`），不是薄壳，并且它会以同样的方式
   让 cross-check 失效，所以需要单独拍板。

   > **【已触发并拍板 2026-08-26：owner 亲自引用本条件。】** 当日 rheplicant
   > 成为 bayesmith 的第一个包级消费者（`pyproject.toml` 声明
   > `bayesmith>=0.2`，`partition.py`/`loglinear.py` 运行时 import），owner
   > 裁决走「一份实现」路线：造适配层、未迁移的全部迁移、cross-check 按预告
   > 的方式换防。执行计划与后续裁决的家：
   > `2026-08-26-one-implementation.md`（本目录）。上面的实测记录继续有效
   > ——它说明的是薄壳为什么不行，而新计划做的正是它点名的适配器。

   ---

   **以下是促成这次重写的实测记录，保留而非删除**，因为「薄壳做不到」这件事
   看起来完全像是没人动手而已，下一个读到原文的人会重新开始做它：

   > **【实测 2026-08-26。发布阻塞已解除（bayesmith 0.1.0 已上 PyPI），所以
   > 卡住薄壳的不再是依赖，是下面这两条，而且都是结构性的、不是工作量。】**
   >
   > **(一) 两边的 API 根本不对应。** `rheplicant.inference.__all__` 有 **99**
   > 个名字，bayesmith 全仓库（顶层 + 七个子模块的 `__all__`）合起来 **77**
   > 个，**交集只有 24 个（24 %）**，另外 **75 个 bayesmith 连同名的东西都
   > 没有**（`ParameterSpace`、`Latent`、`Bind`、`Likelihood`、`NoiseModel`、
   > 两个 calibrator、NPE 那一族……）。而且**连那 24 个的签名也不兼容**，因为
   > 两边是两套范式——rheplicant 是 pipeline + `ParameterSpace`，bayesmith 是
   > `Graph`：
   >
   > | | rheplicant | bayesmith |
   > |---|---|---|
   > | `fisher_information` | `(forward: Callable, params, noise_std, …)` | `(block: LinearBlock, *, precision, …)` |
   > | `identifiability` | `(space, pipeline, state_template, …)` | `(graph, *, names, at, rtol)` |
   > | `wiener_solve` | `(block, observed, *, noise_std, prior_std, …)` | `(block, *, precision, tol, …)` |
   >
   > 重导出这些等于把每一个调用点都打断。**「薄壳」这个词假设 bayesmith 是
   > rheplicant 的超集，而本文其余部分从来没打算造那个东西**：§四 4.1 的表头
   > 写的是「已有**对应物**」，那一列的标题是「**必须一致的内容**」，里面每
   > 一条都是数值一致（逐元素到 roundoff、同样的守卫、同一个不动点）。
   > `docs/migration/` 十三页里没有一页声称过 API 兼容。
   >
   > **(二) 少数真能对上的地方，重导出会让 cross-check 变成永远不会失败的
   > 测试——这条比 (一) 更要命。** `SqrtInfo`、`marginalise`、
   > `marginalise_arrays` 三个的签名和字段两边**完全一致**，看起来正是可以
   > 薄壳的那一小块。但 `tests/crosscheck/test_sqrtinfo_agrees.py` 干的事
   > 就是 `SqrtInfo(**kwargs)` 和 `TheirSqrtInfo(**kwargs)` 各造一个然后比。
   > 一旦 rheplicant 那个变成本仓库这个的重导出，`TheirSqrtInfo is SqrtInfo`,
   > 这条测试就是拿 X 和 X 比，**再也不可能红**。
   >
   > 也就是说：**薄壳越是容易做的地方，它毁掉的东西越多**。迁移的最后一步
   > 会把迁移自己的质量机制悄悄拆掉，而拆掉的方式恰好是这个项目反复记录的
   > 那一种——一个不再能失败的守卫。
   >
   > **建议把步骤 2 重写成它实际该是的样子。** 证据指向的终局不是「一个包
   > 吸收另一个」，而是**两个包各自成立、互相比对**——这与 D1 取 (a)、B12 取
   > (c) 是同一个判断（只有一个消费者时不要为了搬而搬），只是那两条是对
   > config 层说的，这条是对 `inference/` 说的。真要走「一份实现」，需要的是
   > 一层**适配器**（pipeline + `ParameterSpace` → `Graph`），那既不是薄壳，
   > 也会同样让 cross-check 失效，且需要 owner 单独拍板。
   >
   > 步骤 1 已经允许的「指向 bayesmith 的 docstring 指针」不受影响，已加在
   > `rheplicant/inference/__init__.py` 的模块 docstring 上。
3. **一个可能的红利**：rheplicant 的两个 pytest session 之所以存在，是因为证据层
   要 float64 而其余测试断言的拒绝只有 float32 才触发，且 `jax_enable_x64` 是
   进程全局的。证据层迁出后两个 session 可能重新合一——届时重估
   `tests/test_evidence_session.py` 的存在理由，并改写 README 关于覆盖率的叙事
   （88.2 % vs ~99.7 % 那一段）。

   > **【答案：红利不会到，而且数字朝反方向走了——不是两个 session 合成一个，
   > 是变成了三个。实测 2026-09-03，e-RHINO `27e621b`。】**
   >
   > **(一) 前提永久为假：证据层不迁出。** `src/rheplicant/inference/` 里那
   > 八个模块一个不少，共 6199 行——`archive.py` 444、`memory.py` 1045、
   > `compress.py` 1029、`compressed.py` 594、`chain.py` 1255、
   > `factorize.py` 205、`reduced_basis.py` 807、`diagnostics.py` 820；
   > `tests/evidence/` 仍有 40 个测试文件，仍带 x64 采集门。这不是拖延，是
   > 下面步骤 2 那条 owner 裁决（2026-08-26）的直接后果：两个包各自成立、
   > `inference/` 不废弃，于是证据层没有「迁出」这件事可做。四个模块确实已经
   > 在委托远端（`chain.py`、`reduced_basis.py`、`diagnostics.py`、
   > `sqrtinfo.py` 各自 import bayesmith），但**委托不是迁出**：文件还在、
   > 测试还在、x64 还在，第二个 session 一天也没有少跑。
   >
   > **(二) 「两个 session」这个数在 2026-08-27 就不对了。** `647a2ed` 加了
   > `tests/seam/`（适配器的确定性层，断言是对稠密法方程的 `rtol <= 1e-12`，
   > float32 说不出这句话），它带自己的 x64 采集门和自己的驱动
   > `tests/test_seam_session.py`。今天一次 `pytest` 调用跑**三个** session。
   >
   > **(三) 「重估 `tests/test_evidence_session.py` 的存在理由」这件事已经做
   > 了，而且做在它自己的 docstring 里**，比本行早得多，两条实测：
   > 2026-08-25 量到 `tests/evidence/` 当时是套件里唯一带 x64 采集门的目录，
   > 所以「合一」的真实含义是「删掉这个文件和那个 conftest」而不是「给全套件
   > 打开 x64」；2026-08-27 就地更正——seam 的 x64 需求**不随迁移离开**，因为
   > 适配器是 rheplicant **留下**的东西而不是交出去的东西，所以「两个 session
   > 合一」最好的情形也只是「两个里走掉一个」。那份 docstring 还把代价量了出
   > 来：22 个只在 float32 下才成立的拒绝里，只有 6 个在 `tests/inference/`、
   > 会随迁移走；另外 16 个在 `tests/core/` 与 `tests/radio/`，不迁移，而且
   > 承重——float32 是这台仪器的生产 dtype，与推断层做什么无关。
   >
   > **(四) README 的覆盖率叙事不必改，因为它讲的那件事没有变。** e-RHINO 的
   > `README.md` 仍写着「the 99.7 % it was before the evidence layer landed」，
   > 理由仍然是第二个 session 在自己的进程里跑 `--no-cov`。**但同一段把 session
   > 数写成了 two**，那在 2026-08-27 之后少了一个。那是 e-RHINO 的行，不是本
   > 仓库能改的，记在这里等那边的人来读。
4. README 的「四能力」叙事改写：贝叶斯推断与流式证据指向 bayesmith；
   `docs/inference.md`、`docs/inference-*.md`、`docs/evidence.md` 改为迁移指南
   或移除。

   > **【答案：已做，2026-08-26；而做出来的东西与本行预设的方向相反，所以本行
   > 的原文一个字不动，答案写在这里。实测 2026-09-03，e-RHINO `27e621b`。】**
   >
   > 本行预设「贝叶斯推断与流式证据**指向** bayesmith」、那几页「改为迁移指南
   > 或移除」。下面步骤 2 的 owner 裁决取消了这个前提（`inference/` 不废弃、不
   > 转薄壳），于是执行出来的是另一件事：
   >
   > - **README（`575c07c`，2026-08-26）**：「四能力」那张表**原样保留**，四条
   >   仍然都是 rheplicant 自己的能力；表下新增一段，标题式的一句是「**2 和 4
   >   有一个兄弟，而它是一个独立的包而不是继任者**」，并写明重叠部分靠
   >   cross-check 套件而不是共享代码维持一致、这边没有任何东西在迁走或被废弃、
   >   以及该拿哪一个（有 RHINO twin 就在这边，模型不是这台仪器就去那边）。
   > - **`docs/inference.md`、`docs/inference-linear.md`、
   >   `docs/inference-plans.md`、`docs/inference-spaces.md`、`docs/evidence.md`
   >   （`ce282dd`，2026-08-26）**：五页**都没有**改成迁移指南、**都没有**移除，
   >   正文原样保留；五页各在第 3 行 `{include}` 同一个片段
   >   `docs/_migration-to-bayesmith.md`，而那个片段的第一句就是答案——
   >   「There is a sibling package, and this layer is not moving to it」。
   >   片段列在 `docs/conf.py` 的 `exclude_patterns` 里，不另外成页。
   >
   > **为什么是一处 include 而不是五页各写一遍，那个片段自己在注释里说了**，
   > 理由正是本仓库的常客：一句话拼五遍，就是五个要更新的地方和一个真会被更新
   > 的地方。而这不是假想——那份片段**在写下后几小时之内就过期了一次**（当天
   > owner 做了裁决），一处源正是「修那一次只花一次编辑」的原因。
5. **跨仓库已发布契约**——rheplicant 改动其中任何一条都是对 bayesmith 的
   breaking change，值得在 rheplicant 侧加一个具名测试把它们标出来
   （散文挡不住，守卫才行）：
   - `build_forward_fn` 接缝（bayesmith 的「pipeline = 确定性节点」经此零适配接入）
   - `core` 异常类的共享 identity（52 处测试 import）
   - `AbstractOperator` 的「`__call__` 内只做结构校验」契约（函数追踪安全性的前提）

   > **【答案：已做，2026-08-26，e-RHINO `eed1357`，
   > `tests/test_published_contracts.py`——295 行、9 个测试函数、实收 27 passed
   > （`pytest tests/test_published_contracts.py --no-cov`，2026-09-03）。】** 三条各有自己的具名类，与上面三个 bullet 一一对应：
   >
   > - `build_forward_fn` 接缝 → `TestTheForwardSeamKeepsItsPublishedSurface`：
   >   发布路径与定义模块持有同一个对象；形参名、顺序与默认值；返回消费者要
   >   解包的那个 pair。**行为**不在这里重测（那是 `tests/inference/test_forward.py`
   >   的题目），这个文件守的是**表面**，而表面正是那个文件不钉的东西。
   > - `core` 异常类的共享 identity → `TestTheExceptionClassesKeepTheirIdentity`：
   >   先断言这一遍**真的看见了那个模块**（否则一次扫空的普查会读起来像全过），
   >   再逐名断言每个类既能按族捕获也能按其内建基类捕获，且根导出是**同一个
   >   对象**而不是一个副本。
   > - `AbstractOperator.__call__` 只做结构校验 → `TestAPipelineStaysTraceableAsANode`：
   >   jit 复现 eager 的结果、梯度到得了 pipeline 内部的参数，外加一个**阳性
   >   对照**——`__call__` 里放一个取值检查，必须在这里被抓到。
   >
   > 它的模块 docstring 把本行那句「散文挡不住，守卫才行」接了下去，并说清了
   > 为什么这个守卫**必须长在上游**：**每一种破坏方式的症状都落在另一个仓库。**
   > 改一个关键字名、或者把异常类**重建**而不是**重导出**，rheplicant 自己的
   > 套件照样是绿的——因为它内部没有任何一处像外部消费者那样 import 这些名字；
   > 而 `except ParameterSpaceError` 会在跨仓库的另一侧安静地不再触发。
   >
   > **三条里今天真正被消费的是哪几条，那份 docstring 也是量出来的而不是假设
   > 的**：契约 2 与 3 是活的（bayesmith 从 `rheplicant.core.errors` import
   > `ParameterSpaceError` 与 `StateValidationError`；`graph/nodes.py` 把一个
   > `eqx.Module` 当作节点的 `fn`、经 `__call__` 调用）；契约 1 **被设计针对
   > 但今天尚未被 import**——`build_forward_fn` 出现在 bayesmith 的文档里，
   > `.py` 文件里一次都没有。它照样被钉住，理由写在那里：「published」说的是
   > 外部消费者**可以**依赖什么，不是今天谁在依赖。

---

## 七、需要 owner 拍板的决策（跨仓库项归本文所有）

> **归属规则：跨仓库的决策住这里，纯 rheplicant 的决策住 e-RHINO 的
> `PROPOSAL_MERGED.md` §5（那边是 D3/D4/D5）。一个决策只有一个家。**

**D1 — 分离后谁拥有 fitting exits？** rheplicant 的 config 今天经
`config/sections/{inference,noise,observed,parameters,transforms,npe,nuts,conjugate,exit_support,exits,diagnostics}.py`
与 `config/{preflight,postflight}/{fitting,gated,noise}.py` 代理
`rheplicant.inference`。二选一：**(a)** 这些 section 留在 `rheplicant.config`，
把 bayesmith 当运行时依赖调用；**(b)** 迁到一个 bayesmith 的 config 层。
同一问题在文档 schema 层重复一次（`inference.*`、`runs[].kind`）。附带子问题：
`calibrate.py` 的去留（config 的 `optimize` 依赖它，bayesmith 无对应物）。

*建议：先取 (a)。* `rheplicant.config` 的价值在于它是**这台仪器**的 YAML 面；
把语法搬进一个通用推断库，要么被迫通用化（丢掉 C18 这类射电专属拒绝），要么
两处重复。等 bayesmith 有第二个消费者时再重估。

> **【owner 已拍板 2026-08-26：取 (a)。】** config section 留在
> `rheplicant.config`，bayesmith 作**运行时依赖**被调用。与 B12 的 (c) 同源，
> 两条裁决方向一致：只有一个消费者时，无从知道语法的哪些部分是通用的。
>
> **随之而来的是 §六 步骤 2 的一条新前提，它不是决策问题而是发布问题。**
> D1(a) 定了依赖方向，但 `rheplicant` 是 PyPI 上的已发布包（README 写的就是
> `pip install rheplicant`），而 bayesmith 今天是 **0.0.0 且不在索引上**。把
> `inference/` 转成重导出 bayesmith 的薄壳，等于让每一次 `pip install
> rheplicant` 依赖一个装不上的包。D6 已经就 harness 说过同一句话（「那会让
> rheplicant 依赖一个 0.0.0 版包，不推荐」）；这里是同一事实落在步骤 2 上，
> 而步骤 2 的后果比 harness 大得多——一个是开发环境，一个是所有用户。
>
> 本仓库对这件事有**同一形状的先例**：rheplicant 为 limTOD 保留了 `>=1.10`
> 的下限，而当时索引上只有 `<=1.8.0`，`pyproject.toml` 的注释把它记为「拒绝
> 假装 sky engine 是可选的」，并在 limTOD 上架后自然兑现。区别在于 limTOD
> 当时**已在索引上**、只是版本不够；bayesmith 则整个不在。先例支持「声明真
> 实依赖并等它兑现」，不支持「现在就把用户的安装挂在上面」。
>
> 所以步骤 2 有两条路，**先答走哪条**：(i) 先发布 bayesmith（哪怕 0.1.0 的
> 预发布），再转薄壳；(ii) 薄壳做成**软依赖**——bayesmith 不在场时
> `inference/` 仍以今天的实现工作，在场时改指它。(ii) 意味着两份实现并存一
> 段时间，正是 §一 铁律想避免的；(i) 意味着步骤 2 的排期取决于一次发布。
> **在此之前不要开始步骤 2。**
>
> 一条与这个选择有关的事实，只记不判（2026-08-26 实测）：本仓库 `README.md` 的
> §Status 仍写「Early development」，版本仍是 `0.0.0`，而四条头条能力的模块都在、
> 都不小——`exact/precision.py` 451 行、`evidence/campaign.py` 408 行、
> `diagnose/sensitivity.py` 816 行、`bridge/numpyro_bridge.py` 205 行，1127 个测试
> 全绿。README 低报了现状，而「要不要发布」恰好卡在这一点上。发布与否是 owner 的
> 决定，这里不替它做；写下来只是让做这个决定的人手边有这个数。

**D2 — `conditioning.py` 是否随贝叶斯层迁走？** e-RHINO 的
`radio/filters/skyspace.py` 需要幂迭代式条件数估计来实现 Track A 的 A8.2
（`SkySpaceFilter` 的 CG 目前无收敛通道，且实测当前 JAX 的 `cg` 返回
`(x, None)`——不是忘了检查，是无从读起）。若 `conditioning.py` 迁走，radio 侧
需要一个 core 级等价物或自含实现。**A8.2 动手前必须先定。**

> **【更正 2026-08-26，在同日的裁决之后：本条问题本身是坏的，而且早就有人
> 发现了。】** 上面这条 D2 问「`conditioning.py` 是否随贝叶斯层迁走」，前提是
> `SkySpaceFilter` 用得到它。用不到，而且不可能用到：分层是 `core ← radio ←
> inference`，由 `tests/core/test_layering.py` 强制，`radio/` 一行也不能 import
> `inference/`。而且 `conditioning.py` 根本不在 `inference/` 里——它在
> **`core/conditioning.py`**，贝叶斯层迁走时压根不动它。所以 **A8.2 从来没有被
> D2 卡住**，本行「A8.2 动手前必须先定」是空的。
>
> **A8.2 也已经做完了**：`radio/filters/skyspace.py` 带 `require_convergence:
> float | None`，在 `__check_init__` 里校验，模块 docstring 写明「残差不是精度」
> 并经 `rheplicant.core.conditioning` 以 `kappa * residual` 定界。2026-08-26 用
> 变异确认了这个守卫仍然会红：把判据从 `residual * kappa` 改成 `residual`，
> `test_a_small_residual_does_not_certify_the_answer` 与
> `test_the_two_verdicts_are_different_verdicts` 立刻失败（退出码 1）。那个
> `kappa *` 是承重的，不是装饰。
>
> **这条被重复裁决，是「一个决策只有一个家」这条规则自己的漏洞。** e-RHINO 的
> `PROPOSAL_MERGED.md` 已在 **2026-08-25** 记下「D2 — RESOLVED: the question was
> malformed」，理由与上面一致。但 D2 的「家」被宣告在本文，于是本文继续挂着一个
> 已死的问题，而更正住在另一个仓库里。**规则说了决策住哪儿，没说它的解答住
> 哪儿**——于是家里留着问题，答案在外面，下一个人（本次就是）照着家里那份重新
> 裁决了一遍。补法不是再加一条规则，是让解答回到问题的那一行去。
>
> 下面这条同日的裁决保留，因为它问的是另一件事，与前提无关：
>
> 条件是这条裁决里真正新的部分：**若其中的能力对贝叶斯问题是普遍适用的，
> bayesmith 实现自己的一份。** 注意这不是「迁移」的另一种拼法，两者的验收
> 完全不同——迁移要求一份实现搬家，而这里要求**两份共存并互证**。所以它落在
> §二 cross-check 协议下，不落在 §四 迁移台账下：两边各自实现，同一输入必须
> 给出同一答案，这正是 §四 4.1 Fisher 行、`linear` 行走过的形状。
>
> 也因此，动手前要先答的是「哪一部分是普遍的」，而不是「怎么搬」。幂迭代式
> 条件数估计本身与射电无关，是候选；而它在 `skyspace.py` 里的用法（CG 的收敛
> 通道、`cg` 返回 `(x, None)` 这个实测事实）是**这台仪器的**，不是。把后者
> 一起搬过去，就是 D1 拒绝的那个错误换一层皮：为了一个消费者而通用化。

**D6 — cross-check harness 的宿主环境。** 已在 §0.1 按「bayesmith 的 venv +
rheplicant 作仅测试依赖」提出并给出落地命令。若 owner 反对，替代方案是在
e-RHINO 侧装 bayesmith——但那会让 rheplicant 依赖一个 0.0.0 版包，不推荐。

> **【已定，而且是被执行定的，不是被裁决定的；答案写回提问的这一行。实测
> 2026-09-03。】** 取本文提的那条：harness 住在 bayesmith 这边，rheplicant
> 作仅测试依赖。落地在 §0.1（2026-08-24），CI 形态在
> `.github/workflows/crosscheck.yml`（2026-08-26）——按 URL 从 rheplicant 的
> **main** 装、`--no-deps`、打印解析出的 commit，并从 junit XML 读计数，比过
> 0 个就红。**没有 owner 的反对记录**，所以这条是「已执行且无人反对」，不是
> 「已裁决」；两者的区别在于后者会挡住重开，前者不会。
>
> **而那句「不推荐」的理由今天已经不成立，这一点必须一起写下**——否则下一个
> 读到本行的人会以为替代方案仍被同一个论据挡着，并据此重新论证一遍。当时的
> 论据是「那会让 rheplicant 依赖一个 0.0.0 版包」。2026-08-26 起 rheplicant
> **本来就**声明了 `bayesmith>=0.5`（e-RHINO `pyproject.toml:82`）并在运行时
> import 它（`partition.py`、`loglinear.py`），而 bayesmith 已经在 PyPI 上——
> 实测 `pypi.org/simple/bayesmith/`（照 `CLAUDE.md` 的规矩问 `/simple/`，
> 不问 JSON API）：0.1.0、0.2.0、0.3.0、0.4.0、0.5.0、0.7.1。所以两条路今天
> 都装得上；本条继续取第一条的理由变成了「它已经在跑，而且跑的就是 CI 里那
> 一份」，而不再是「另一条装不上」。

---

## 八、执行顺序

1. **§0.1 的 harness 宿主**（一条 pip 命令 + pyproject 的 crosscheck 组）。
   这是两份原方案都称作「最高价值第一个 commit」却跑不起来的那件事。
2. **P3b 收尾**：确认 B1 的设计规则已落在 `exact/correct.py`；然后按 §四 4.2
   比对 `plan`/`engines`（**B1 之后**）与 `noise` 层。
3. **B2**，然后比对 §四 4.1 的 Fisher 行。
4. **B9 相关噪声**（它决定证据层的压缩格式）。
5. **P5 诊断移植**：`identifiability`、`sensitivity`（+B3）、`priors`。
   **已完成 2026-08-25**（顺序上被跳过、在第 6 步之后补做）：`diagnose/`
   包 + 120 个本地测试 + 三份 cross-check 页；rtol 常数重测、B3 修复、
   `fisher._weighted_design` 顺带修了一个真缺陷（二维观测节点上广播失败）。
6. **P6 = B11 流式证据重写**。
7. **B10**，然后 §六的 rheplicant 收尾。**B12 已于 2026-08-25 由
   owner 拍板取 (c)（不动 config 层），无代码可写**——见 §五 B12 的批注。
   **B10 也已于 2026-08-25 闭合**（`074848a`，两个具名验收测试）——见 §五 B10
   的批注；证据不在这里复述，这一行只是指针。§六 的收尾状态同样写在 §六 各步
   骤自己的行上：步骤 1 被取代、2 已重写并落地、3 的红利不会到、4 与 5 已做。

---

*规格完。配套：e-RHINO 的 `PROPOSAL_MERGED.md`（Track A + 两份评审的裁决）；
证据：e-RHINO 的 `CODE_REVIEW_REPORT.md`、`REVIEW_REPORT.md`。*
