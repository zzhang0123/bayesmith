# P4 — 图归约与第二个 graph-level factor 槽

> **文档状态：`module-spec`** · 已发布模块/能力的当前设计文档，从属于顶层设计；改动对应代码须同步本页。索引见 docs/README.md。

**日期**：2026-08-29  
**状态**：已实现  
**实现**：`src/bayesmith/graph/{graph,evaluate,reduction}.py`、`src/bayesmith/bridge/numpyro_bridge.py`  
**探针**：`docs/probes/probe_23_double_count_visibility.py`

## 一、公开形状：一个图、两个互斥槽、一个原子入口

`Graph` 的最终字段是：

```python
nodes: tuple[Node, ...]
plates: tuple[Plate, ...]
joint_prior: Any = None
evidence_terms: tuple[Any, ...] = ()
```

每个 evidence term 结构上必须提供：

```python
over: tuple[str, ...]
log_density(graph: Graph, values: dict[str, Any]) -> scalar
```

`over` 只能命名归约后仍存在的 latent。一个图可携带多个 evidence term，因为多个已吸收
数据集的 likelihood 可以相乘；它们可以覆盖同一个 retained latent。`joint_prior` 仍只有
一个，而且它与每个 evidence term 的 `over` 必须不相交。

归约只有一个入口：

```python
reduce_with_evidence(
    graph,
    *,
    remove_latents,
    absorb_observed,
    evidence_term,
    nuts_latents,
) -> ReducedGraph
```

返回对象的 `nodes` 已删掉被积分 latent、显式吸收的 observed node 与其孤立的确定性后代，
同一个对象的 `evidence_terms` 已追加替代 likelihood。`ReducedGraph` 是 NUTS-only 包装：
`log_joint`、`to_numpyro`、`nuts` 明确解包；通用 `compile` 一读取 `latents` 就给出带出路的
硬拒绝。模块不提供 `reduce_graph(...)` 或 `attach_evidence(...)`；调用者不能通过两个独立
成功的步骤得到一个只有半边的中间态，且不能把完整的归约结果误送进不消费 evidence 的
conditional exact 路径。

底层 `Graph(...)` 仍能手工构造故意的未归约 mutant；这是绝对密度变异测试所需的缺陷注入
点，不是生产归约 API。生产入口返回的类型不同，因而“删图”和“挂项”在类型上不可拆开。

## 二、为什么 evidence 与 joint prior 必须互斥

这不是两个名字表示同一类 factor。`JeffreysPrior.information` 要求其覆盖 latent 自身是
`ImproperUniform`；否则 node 自己的 proper prior 加 Jeffreys density 是同一数量上的两个
先验，已有路径按名拒绝。边缘 likelihood 的要求正相反：它对 surviving theta 条件化，
theta 自己的 proper prior 必须保留，最终密度是

```text
p(theta) * p(absorbed data | theta).
```

若两槽在 theta 上重叠，同一个 node 不可能同时满足“必须 flat”与“必须保留 proper prior”。
因此构图时拒绝交集，并明确建议保持两个 block 不相交；这比等到第一次 leapfrog 才由某个
term 自己发现冲突更早，也不会让 bridge 与 `log_joint` 在不同时间失败。

## 三、两次扫描读同一声明，factor 永远在 plate 外

`log_joint` 在读 `joint_prior` 的同一个 `graph_terms` 循环里依次加全部
`evidence_terms`。它不是另开的一次节点扫描；新增 evaluator 时可复制的单位是“全部图级
密度项”，不是其中一个槽。

两个扫描共用 `graph_density`，要求每项返回 shape `()` 的单个标量；若 term 返回向量，
`log_joint` 原本会广播而 NumPyro 会求和，二者会悄悄得到不同语义，所以两边现在以同一句
拒绝文本失败。evidence term 只收到其 `over` 声明的值，不能从完整 latent dict 暗读未声明
依赖。

`to_numpyro` 先完成所有 node/sample 的循环，离开每一个 `numpyro.plate`，再依次发出
`evidence_0`、`evidence_1`……；若图节点或 plate 占用该 NumPyro site 名，就逐个加前导
`_`，直至不碰撞。测试分别覆盖 node/plate 冲突，既检查 factor site 的
`cond_indep_stack == ()`，也把 bridge 的绝对密度与 `log_joint` 比较。

位置是数值语义，不是缩进风格。独立 characterization 把同一个 `-1.7` factor 放在大小
5 的 plate 外与 plate 内：`inside / outside == 5.0`。NumPyro 不报错；plate 内的标量被
广播成五项再求和。因此一个写在 observed sample 旁边、只差一级缩进的 factor 会把完整
边缘 likelihood 静默乘以 plate 大小。

## 四、归约规则与拓扑不变量

`remove_latents` 只能命名 latent，`absorb_observed` 只能命名 observed probabilistic node。
删除前沿只从 `remove_latents` 开始；原图按原声明顺序扫描，任何父节点已落入前沿的
deterministic node 也落入前沿。一个 `absorb_observed` 名字只有在它是该前沿（或另一个已落入
前沿的节点）的后代时才可加入。若这种依赖到达一个没有显式列入相应集合的 probabilistic
node，归约拒绝：自动删掉一个 latent distribution 或 likelihood 等于声称 evidence term
包含一项它从未声明的密度。

“absorbed 必须在 remove 的后代闭包内”是刻意收窄的**结构代理**，不是对不透明
`log_density` 内容的完备证明。一个 term 在数学上可能合法地同时携带与 removed block 独立的
观测因子；图层看不见 term 的内部，因而会把这种合法写法当作假阳性拒绝。出路是让独立
likelihood 保持显式，或把它放入有自己删除前沿、明确 evidence 的另一归约；相比静默删掉整条
`p(z_data | z)`，这个保守拒绝是有意选择。

保留节点是原 `nodes` 的稳定子序列，绝不重排。因此 `Graph.__check_init__` 的“父先于子”
不变量原样保留。`plates` 同样按原声明顺序过滤，只保留至少被一个存活节点引用的 plate；
删掉最后一个引用者不会留下孤儿 plate 声明。测试不是只断言构造成功；它在一个未受影响的 `z -> z_mu -> z_data`
分支上实际求 `log_joint`，使错误剪枝或拓扑破坏可达。

## 五、核心硬拒绝：evidence 只能覆盖 NUTS block

`nuts_latents` 是调用计划随原子归约一起给出的结构见证，但不存进 `ReducedGraph`：返回图由
`to_numpyro`/`nuts` 采全部 retained latent，保存一个可能是真子集的公开字段只会伪装成可供
消费者读取的真实分区。graph 层不导入 dispatch，而是校验新旧**每一个** evidence term 的
`over` 都是该见证的子集。

这是本包最重要的拒绝。当前可复现的
`rg 'evidence_terms' src/bayesmith/exact src/bayesmith/dispatch` 只有 exact 下两处语义：
`exact/block.py` 只检查是否需要 NUTS 见证，`exact/loglinear.py` 在图重建时**携带字段**；
dispatch 没有密度读取点。`joint_prior` 的旧 M9 “零命中”已不成立：同一重建现在显式携带
`joint_prior`。这些命中都不是条件密度求值点；
`gcr_sample` 仍不消费任何图级 likelihood。若 evidence 覆盖某个非 NUTS latent，条件 GCR
会从一个省略该项的条件分布抽样；该分布仍归一、CG residual 仍健康、R-hat/ESS 仍有数，
所有诊断都可能是绿色，错误只存在于目标密度。

因此防线落在**数据和执行边界**上：归约时检查 `term.over ⊆ nuts_latents`；
`partition`/`factor_partition`/`declared_partition` 在得到真实 NUTS block 后对裸 `Graph`
重查；`sample_factors`/`estimate_factors` 对缓存或手写 plan 再查；`ReducedGraph` 仍以类型
阻止通用 `compile`；公开 `linear_operator`、内部/高级 `unchecked_operator` 在形成任一
exact block 时必须接收完整 plan 的 `nuts_latents` 见证，不能把“本块之外”猜成 NUTS；
带 term 而不提供见证直接拒绝，因此即使显式解包 `ReducedGraph.as_graph()` 也不能绕过。
factor plan 尚在形成时还不存在这个见证，其私有 operator 只供结构/条件数探测，完整 plan
形成后必须先以 `plan.nuts` 统一裁决才能返回或执行；执行入口再把同一见证传给每个 exact
block。这样第二个 exact block 不会被第一个 block 的局部补集误称为 NUTS。
拒绝文本给两条出路：把该 latent 放入 NUTS block，或保持原 likelihood 显式存在且不要吸收
其 observation。`log_space` 的合法图重建同时携带 `joint_prior` 与 `evidence_terms`，所以
受支持的 `log_joint(log_space(graph).graph, ...)` 不会丢绝对密度。

核心拒绝及其合法邻居是：

| 拒绝 | 触发条件 | 合法邻居 / 出路 |
|---|---|---|
| term 协议/域错误 | 缺 `over`/`log_density`，或 `over` 命名非 surviving latent | 提供完整协议，并先把被删变量从 term 中边缘化 |
| prior/likelihood 冲突 | `evidence_term.over ∩ joint_prior.over != ∅` | 两块保持不相交；需要别的语义就声明另一模型 |
| 删除前沿未封闭 | 未吸收的 probabilistic descendant 仍依赖被删区域 | 把其完整密度纳入 term 后显式列入对应集合，或不归约 |
| 无关 observation 被吸收 | `absorb_observed` 不是 remove/drop 前沿的后代 | 保持 likelihood 显式，或单独归约；独立因子虽可能合法但结构层保守拒绝 |
| 非 NUTS 覆盖 | 任一新旧 term 的 `over` 不含于 `nuts_latents` | 放入 NUTS，或保留显式 likelihood |

重复名、把 observed 写进 `remove_latents` 等纯参数角色错误也按名拒绝；它们不是上述五条
会产生静默错误的数学拒绝。

## 六、绝对密度守卫与可重跑反例

关键测试用

```text
x ~ Normal(0.35, 1.7²)
gain ~ Normal(0.1, 1.3²)
offset ~ Normal(-0.2, 0.8²)
mu = offset * trend + gain * basis * x
d ~ Normal(mu, 0.55²)
```

三个 prior 宽度都非单位。对 `(gain, offset)` 的五个点求值：四个角
`{-1.2, 1.4} x {-1.1, 0.75}`（所以每个参数的两个端点都出现）加 prior centre。oracle
在 `x_loc ± 12 x_scale` 上用 60,001 点梯形积分直接积
`exp(log_joint(original))`；被测侧直接求 `log_joint(reduced)`。判据是绝对 log density，
`rtol=0, atol=2e-9`。当前最大差为浮点舍入量级，而不是只比较归一化后的形状。

同一个 assertion 随后对故意 mutant 重跑：把 evidence term 挂到**未归约**的原图，再对 x
积分。测试用 `pytest.raises(AssertionError)` 证明守卫确实转红，并另断言五点最小绝对 gap
大于 1 nat；这不是一个只有“正确实现会绿”而没有“目标缺陷会红”的守卫。

`probe_23_double_count_visibility.py` 给更尖锐的反例：保留的 `theta ~ Normal(-0.4,2.3²)`
与被积分的 `(x,d)` 独立。错误图比正确图多一个常数 `log p(d)`，所以归一化 posterior
均值、宽度及任一点梯度逐数相同；绝对密度在每一点都偏同一个非零 nat 数。探针用稠密 x
积分验证错误图，而不是只打印手写公式。它把“只看均值/宽度/梯度会漏掉双计数”变成可重跑
证据。

## 七、决策 D85–D89

| 编号 | 一句话结论 |
|---|---|
| **D85** | `Graph.evidence_terms` 是 `tuple[Any, ...] = ()`，term 以 `over + log_density(graph, values)` 声明；它与 `joint_prior` 按 latent 强制互斥。 |
| **D86** | `log_joint` 在一个循环中求和全部图级项，bridge 在所有 node/plate 结束后以独立 factor 发 evidence；plate 内实测会乘 plate size。 |
| **D87** | 唯一入口 `reduce_with_evidence` 返回 NUTS-only `ReducedGraph`，同一对象同时暴露归约 `nodes` 与已追加 `evidence_terms`，保留拓扑、过滤孤儿 plate 且没有单边 API。 |
| **D88** | 新旧 term 必须落在真实 NUTS block；归约、完整 plan、显式 exact witness、执行入口与返回类型共同拒绝条件 exact 路遗漏该项。 |
| **D89** | 正确性以五点绝对密度对稠密积分、未归约 mutant 必红为准；均值、宽度和梯度不足以守住 theta-independent 的双计数。 |
