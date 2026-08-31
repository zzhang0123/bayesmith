# P2 — 图原生 MAP 与它自己的拒绝

> **文档状态：`module-spec`** · 已发布模块/能力的当前设计文档，从属于顶层设计；改动对应代码须同步本页。索引见 docs/README.md。

**日期**：2026-08-29  
**状态**：已实现并验证  
**实现**：`src/bayesmith/diagnose/map.py`  
**探针**：`docs/probes/probe_20_map_refusals.py`

## 一、问题与接口

`map_estimate(graph, *, at=None)` 优化图自己的完整 `log_joint`，返回后验众数，供
`local_block(graph, names, result, priors=True)` 等 Laplace 诊断直接使用。成功值
`MapEstimate` 实现只读 `Mapping[str, jax.Array]`，同时携带迭代数、目标值和梯度范数；
优化层面的失败返回与 P1 **身份相同**的 `Refused(reason)`，无 latent 的图返回
`NotApplicable(reason)`。

这不是把 `prior_sensitivity` 的一行提升成公开函数。后者的问题是“选中先验把众数推了
多远”，所以它构造一个删掉那些先验的 `neg_log_rest`，并要求那个反事实本身可解；MAP
的问题只是“完整声明的后验众数在哪里”。两个目标不同，准入条件也必须不同。

## 二、先读出的三条旧拒绝，以及它们为什么对 MAP 都错

### 1. `_refuse_entangled_selection`：选中先验不能依赖选中 latent

`prior_sensitivity` 要把每个选中先验写成常数对角矩阵 `P=diag(s⁻²)`。若
`child ~ Normal(parent, s)` 且 parent、child 同时选中，`m` 随参数移动，闭式位移恒等式
就不成立，所以旧拒绝是对的。

对 MAP，这个依赖正是图的联合密度的一部分，JAX 会通过 `log_joint` 完整求导；拒绝它
反而是在删模型。它会拒掉每一个通常的层级模型，也会拒掉 Neal funnel 的
`x ~ Normal(0, exp(y/2))`。**替代**：不做祖先纠缠拒绝；只要完整目标可微、有限且优化
收敛，就接受。

### 2. `_prior_moments`：每个选中先验必须是固定参数的对角 Gaussian

`prior_sensitivity` 需要从先验读出 `(loc, scale)` 才能组 `P`；Uniform、
ImproperUniform、LogNormal 等没有它需要的那个二次型，旧拒绝同样是对的。

MAP 不组 `P`，它读取分布自己的 `log_prob`。任何在当前坐标中可微且有有限众数的密度
都可以合法优化；把“没有 sensitivity 的闭式公式”误写成“没有 MAP”会拒掉可解模型。
**替代**：不分类分布名字；目标或导数非有限、或迭代无法找到驻点时，才返回带出路的
`Refused`。

### 3. `_refuse_unanchored_selection`：likelihood-only 曲率必须正定且不过条件数门

`prior_sensitivity` 定义的是从 likelihood-only mode 到 posterior mode 的位移。若删掉
选中先验后有一条 ray，就没有 likelihood-only mode，距离也不是一个数；因此它检查
`H_lik` 的最小特征值和 `condition_ceiling`。

MAP 要找的是**完整后验**众数。由声明先验、下游层级密度或 `joint_prior` 锚定一条
likelihood 不识别的方向完全合法。特别是要边缘化的线性 nuisance 块，常常正是“数据
只认某个组合、先验使积分有限”的块；沿用此门会恰好拒掉 collapse 想处理的对象。
**替代**：检查完整后验在返回点是否是有限驻点和局部极小（负 log joint 的 Hessian
正定），但**不设条件数上限**；这里不成协方差，高条件数本身不是 MAP 不存在。

三条拒绝都留在 `prior_sensitivity` 原处且行为逐位不变；P2 不调用它们。

## 三、目标、求解与拒绝

latent 按 `graph.latents` 的声明顺序扁平化；`at` 覆盖声明中心并作为 Newton 起点。
唯一目标是

```text
objective(x) = -log_joint(graph, unflatten(x)).
```

因此每个 latent 自己的密度、每个 observed density、层级依赖、mask 和
`graph.joint_prior` 都从同一个图扫描进入，不另写第二份 posterior。

求解复用 `sensitivity._newton` 的阻尼 Newton 算术，但它返回的相对步长布尔只是一条
**预算信号**，不是准入裁决。原因是同一阈值会在病态二次目标的舍入抖动上永不满足，
也会被一个巨大坐标分母提前满足。P2 总是对最后一点重算完整目标、梯度和 Hessian，
再按自己的证据解释：

1. 环境不是 float64：`Refused`，出路是把**构图也放进** `jax.enable_x64(True)`；
2. 构图时已固定为 float32 的图值或分布参数：在 `log_joint` 被 float64 零提升
   **之前**按名 `Refused`；累加后的标量 dtype 不是图精度的证据。这个门与 P1 共用
   同一份原始图叶子扫描。纯 dtype 无法区分一个无损的 float32 `0/1` 掩码与一个已截断
   的 float32 设计（二者上转后都精确），所以诊断家族选择保守一致：两者都拒绝，而不让
   MAP 拒绝、coupling 测量同一张图；
3. 目标、梯度或 Hessian 非有限：`Refused`，出路是改起点、尺度/坐标，或用 NUTS；
4. 先量 Hessian 谱，再把梯度与 `sqrt(eps)·n_parameter·||H||₂` 比较；高于这个
   **局部目标舍入尺度**就 `Refused`，绝不把相对小步泄漏成“已收敛”。现行公式没有
   clamp（B1 删除了 `max(|lambda_max(H)|,1)` 与 `max(||x||_inf,1)` 两个），所以它
   **对目标尺度不变**（`||H||` 随目标缩放、梯度同向缩放），但**对坐标尺度不保证**：
   `x→c·x` 时 `||H||₂∝c²` 而梯度 `∝c`，裁决比随 `1/c` 漂移。坐标不变由
   `test_a_large_candidate_cannot_buy_its_own_gradient_allowance` 的大坐标拒绝钉子
   替代；若 owner 要恢复坐标不变，需重引入 `||H||₂·max(||x||,1)` 项（不能回退到 B1
   修掉的逃逸点放水版）——这是策略裁决，本 spec 不拍板；
5. 驻点的负 log joint Hessian 非正定，或最大曲率本身不高于 `sqrt(eps)`：`Refused`。
   后一条专门区分“真实驻点”和“尾部目标/梯度下溢成零”；
6. 无 latent：`NotApplicable`，而不是空字典冒充估计。

每个 `Refused.reason` 同时说原因和至少一条可执行出路。这里故意**没有**：先验必须
Gaussian、先验参数不得层级依赖、likelihood-only 必须可逆、posterior 条件数不得过门。

## 四、测试与独立参照

### 4.1 线性高斯

使用非单位先验宽度；MAP 与 `wiener_solve` 对拍，且成功值直接传给
`local_block(..., priors=True)`。目标值、非零梯度范数和步数都由独立稠密表达式读取，
因此把三个证据字段任一硬写成零都会红。

### 4.2 Funnel

一维 Neal funnel 接受并落在解析众数 `(y,x)=(-4.5,0)`；这直接杀掉借用
`_refuse_entangled_selection` 的实现。

### 4.3 层级模型

`child ~ Normal(parent, 0.6)` 的层级似然图接受。这里的 `0.6` 与 fixture 和 probe
逐字一致，不另写一个文档模型。

### 4.4 真平脊与非驻点

两个 ImproperUniform latent 只以和进入数据，完整 Hessian 有真零方向：拒绝文本点明
proper prior/删冗余参数等出路。Vandermonde 有效点即使用满 100 步仍与
`np.linalg.solve` oracle 在 `5e-8` 内，必须接受。反方向用一个唯一众数为 `x=0.3`
的高偶次目标：从 `x=1.3` 出发，100 步预算结束时梯度仍远高于舍入尺度，必须拒绝；从
解析 oracle `x=0.3` 出发则逐位得到零梯度并接受。这样 oracle 比被裁决的量更准，而不再
用“大坐标本身”触发要测试的拒绝。

### 4.5 Logistic 尾部下溢不是有限 MAP

ImproperUniform 上放三份 `Bernoulli(logits=x)` 成功观测，其 log likelihood 的上确界
只在 `x→∞` 取得。旧裁决走到 `x≈37` 后得到 `objective=-0.0`、`gradient=0.0`，而
Hessian 仍是略高于裸 `eps` 的退化正数，于是把算术下溢读成了众数。测试明确断言：
**零梯度配退化 Hessian 不是收敛证据**，结果必须是含出路的 `Refused`，不得泄漏尾部
iterate 给 Laplace。

### 4.6 精度与空图

ambient float32、显式 float32 起点、以及 x64 外构造而在 x64 内调用的截断图分别由
独有拒绝短语钉住；无 latent 返回 `NotApplicable`。

### 4.7 尺度与单位不变性

40×3 ridge 固定 `cond(H)=1.59644`，只把观测标准差从 `1e-2` 缩到 `1e-6`，使
`||H||` 从 `3.680e5` 放大到 `3.680e13`；五格全部 `MapEstimate`，对独立 NumPy
正规方程的相对误差不超过 `3.71e-16`。同一模型再做坐标换算
`x_physical=c x_coordinate`，`c in {1e-2,1,1e2,1e4}`，四格均测量且物理众数在打印的
九位上逐分量相同：`[1.200038709, -0.549966357, 0.280401865]`。删掉 Hessian 尺度
后，前一组的后三格会变红；删掉坐标尺度对现行公式的裁决不变（现行公式本来就不保证
坐标不变，由大坐标拒绝钉子覆盖）。

oracle 不使用 MAP 实现的任何例程。线性图的一侧是矩阵自由 `wiener_solve`；funnel
众数由手写联合密度 `y²/18 + y/2 + x²/(2 exp(y))` 的导数直接得到。

## 五、决策 D76–D78

| 编号 | 解决的问题 | 一句话结论 |
|---|---|---|
| **D76** | MAP 的目标从哪里来？ | 只优化图原生完整 `-log_joint`，不从 sensitivity 的 Gaussian moments 重建第二份 posterior。 |
| **D77** | 沿用哪套拒绝？ | 一条也不沿用 prior-sensitivity 的三条模型分类；P2 只拒绝精度、非有限、梯度未到曲率与坐标共同定标的舍入尺度和 Hessian 非极小/退化这些 MAP 自身失败。 |
| **D78** | 失败如何暴露？ | 成功返回可直接当 latent mapping 的 `MapEstimate`；优化失败返回带原因和出路的 `Refused`，无 latent 返回 `NotApplicable`。 |

## 六、盲区与度量纪律

MAP 是一个局部优化结果。多峰后验有多个局部 mode，`at` 决定落在哪个盆；一个成功的
MAP 不证明全局最高，也不描述尾部质量。随后以它为中心的 Laplace 更是局部量，对尺度
几何可全盲：Neal funnel 在众数处的线性 Laplace 相关为 **0.0**，而 20 万 iid 样本的
线性/二次特征 CCA 分别为 **0.0080/0.1157**。P2 接受 funnel 只表示“找到一个合法
mode”，绝不表示“funnel 几何已被修好”。

本包也不把 MAP 或 Hessian 条件数塞进采样成本 argmin。以后若进入 Wave 2，Kish ESS
与 chain ESS 仍不得同场：计划实测 SNIS 偏 1.40 个 posterior sd、NUTS 偏 18.5，
而 chain ESS 33 反而高于 Kish ESS 14；共同 argmin 会选离真相远约 13 倍的答案。

## 审计续（2026-08-30）：曲率 clamp 裁决

**F2(A)**：删除 `curvature_floor` 里的 `max(abs(lambda_max), 1.0)` clamp（选 (a)，与 B1 已删的
两个梯度侧 clamp 一致）。实测：同一后验两次精确单位换算（宽度 30/28/32），`‖H‖=1.276e-03`
时相对误差 `4.02e-05`、`‖H‖=1.000` 时 `5.13e-08`，比值 `783.7 = 1/‖H‖`；同一物理点在
一个单位制下返回 `MapEstimate`、在另一个下 `Refused`。clamp 的 `1.0` 是单位尺度锚、无
推导依据（registry 里唯一的 `magic`），删掉后曲率地板退化为 `eps·|lambda_max|·n`，随目标
尺度缩放，裁决在单位换算下不变。边界格：`test_subunit_curvature_floor_is_invariant_under_a_unit_conversion`。
