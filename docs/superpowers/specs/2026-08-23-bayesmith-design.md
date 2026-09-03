# bayesmith — 设计文档

> **文档状态：`superseded`** · 已被 `2026-08-30-bayesmith-top-level-design.md` 取代；仅保留为历史记录。索引见 docs/README.md。

> 本文是设计（spec），不是实施计划。已于 2026-08-23 经用户批准。
> 实施计划见 `docs/superpowers/plans/`，按 §五 分解的 P1…P7 逐个展开。
>
> **历史状态：** 产品定位、公共概念模型与长期路线图已由
> [`2026-08-30-bayesmith-top-level-design.md`](2026-08-30-bayesmith-top-level-design.md)
> 取代；本文仅保留为早期设计与既有实现决策的历史记录。

## Context

rheplicant 里的信号路径由一组 operator 组成的图表示。这个抽象比射电天文更通用：**任何贝叶斯模型都可以表示成一张 operator 图**，其中每个 operator 是两类之一——**确定性**的传递依赖，**概率性**的定义条件分布——两者合起来构成完整的联合分布。

这正是概率图模型（PGM）/ 概率编程（PPL）的核心抽象，Pyro、NumPyro、PyMC、Gen、RxInfer 都建立在它之上。所以**新颖之处不在图本身**。真正只有这里有、通用 PPL 做不到的，是三样东西：

1. **结构化精确推断**——共轭 / Wiener / GCR / GLS 解，按子图结构分派
2. **流式证据层**——SqrtInfo 跨数据集、跨观测轮次精确合并
3. **图上的诊断**——可辨识性、先验敏感度、线性性检验

而**显式图正是能按子图结构分派到这些方法的前提**。追踪式 PPL 把结构藏在 Python 调用栈里，事后拿不到，因此做不了这件事。

bayesmith 的定位由此确定：**它是分派层，不是又一个 PPL。** 通用 MCMC/VI 直接借 NumPyro（rheplicant 的 `numpyro_bridge.py` 已经是这个模式的先例）。

## 已定决策

| 项 | 决策 |
|---|---|
| **实现位置** | **`/Users/zzhang/projects/bayesmith/`——全新的独立 git 仓库（已确认未占用）。绝不在 `~/Workspace/MCPost` 内实现，也不改动 MCPost 的任何文件。** |
| 定位 | 分派层；NumPyro 作兵库 |
| 与 rheplicant `core/` 的关系 | 松耦合：**一整条 pipeline = 图中的一个确定性节点** |
| 表达力 | 静态 DAG + plate + **离散/混合** |
| 作图方式 | 函数追踪 → **显式图对象为基底真相** |
| 名字 | `bayesmith`（PyPI 实测可用） |
| 与 mcpost 的关系 | 独立新包；mcpost 单独修正后再谈融合 |

---

## 一、图模型

### 1.1 节点的三个正交轴

原草图里 deterministic/probabilistic 的二分不够。每个节点有**三个正交属性**：

| 轴 | 取值 | 作用 |
|---|---|---|
| `kind` | deterministic / probabilistic | 是否向 log-density 贡献一项 |
| `role` | input / latent / observed | 固定输入、待推断、被条件化（由 `observed is not None` 导出，不单独存） |
| `structure` | `linear_in`、`support`、`depends_on_prediction` | **分派判据** |

第三轴是 bayesmith 的核心，而它在 rheplicant 里已有雏形：

- `Latent(linear=True)` → 广义化为 `linear_in: tuple[str, ...]`（对哪些父节点线性）
- `NoiseModel.depends_on_prediction` → 原样沿用（`False` 意味着可跳过重加权循环）

**这些是关于模型的断言，不是提示**——所以必须**被检验而不是被信任**。rheplicant 的 D36 已经立了这条纪律（`check_linearity` 默认运行，唯一豁免是写下理由），bayesmith 原样继承：任何声明为线性的节点，在被用于精确解之前，都要在三个尺度上探测其线性性。

### 1.2 结构上的两处必要校正

- **是 DAG，不是树。** 草图里 A0/A1/A2 → B 就是汇聚结构，`P(d|x,y,z,X,Y)` 也是多父依赖。树表达不了共享父节点。
- **必须有 plate。** 任何分层模型都要 `for i in 1..N`。plate 是节点上的静态注解 `plate=("obs",)`，编译成对该轴的 `vmap`。形状在构造期固定——这是精确分派与预分配的前提。

### 1.3 接口

```python
class Node(eqx.Module):
    name: str                = eqx.field(static=True)
    parents: tuple[str, ...] = eqx.field(static=True)
    plate: tuple[str, ...]   = eqx.field(static=True)

class Deterministic(Node):
    fn: Callable                              # (parent values) -> value，可以是 eqx.Module
    linear_in: tuple[str, ...] = eqx.field(static=True)   # 声明，且必须被检验

class Probabilistic(Node):
    dist_fn: Callable[..., Any]  # 传入父节点值，返回一个 NumPyro 分布
                                 # sample 与 log_prob 因此出自同一对象
    observed: jax.Array | None # None 即隐变量
    support: Support = eqx.field(static=True)   # Continuous | Discrete(n)
```

> ⚠ **实测的陷阱（2026-08-23，equinox 0.13.8）**：`fn`/`dist_fn` 必须是**非静态**字段，但把它们误设为 `static=True` **不会报错**——equinox 只发一条 `UserWarning`。后果是整个 Module 被吸收进 pytree 的 aux 数据，`filter_grad` 于是**静默返回每个参数的原值而非其梯度**。不抛异常，只是答案错了。
>
> 这个陷阱还会污染测试：Task 4 原本的可微性测试用 `w=3.0`、`X=[1.0,2.0]`，而真实梯度 `sum(X)=3.0` 恰好等于 `w` 本身，所以即使 `fn` 被改成 static、返回的是原值，断言照样通过。**凡是断言"梯度等于某常数"的测试，都要确保该常数不等于任何参数的当前值。**

概率节点的接口要求 **sample 与 log_prob 出自同一个对象**。这不是风格偏好——rheplicant 的 `NoiseModel` docstring 已经写下了理由：*"A caller that draws with this and weights with `std` cannot have the two disagree, which is the failure mode of every hand-written `data + sigma * normal` line beside a likelihood carrying its own sigma."*

> ### ⚠ 自洽性检查的固有盲区（P1 终审实测发现）
>
> §四把"同一张图的两种独立读法给出同一个数"（`log_joint` vs NumPyro 的 `log_density`）当作核心交叉检验。**它有一个必须写下来的局限：两条扫描共享实现，因此共享盲点。**
>
> 实测：`Deterministic.fn` 会被 `vmap` 到 plate 轴上，而 `Probabilistic.dist_fn` 起初不会。对原生广播的 NumPyro 分布这不可见；但对一个**未向量化的外部对象**（正是 rheplicant `NoiseModel` 的情形），两条扫描给出 **-225.65，而正确值是 -364.95**——**彼此完全一致，同时都是错的**。那道交叉检验给出的保护是零。
>
> 已修：抽出共享的 `apply_probabilistic`，两条扫描都走它。但结论要留下：**自洽性检验不能发现两边都有的错误。** 真正的独立判据只有解析真值（Task 9 的共轭预言机）和变异测试。P3 引入精确解时，"精确解 vs NUTS"同样是自洽检验——两者若共用同一个 `log_joint`，就共享同一批盲点。

> ### ⚠ 第二条梯度静默丢失的路径（实测，equinox 0.13.8）
>
> 把参数化对象交给 `fn`/`dist_fn` 有三种自然写法，**前向值完全相同**，梯度却不同：
>
> | 写法 | 梯度 |
> |---|---|
> | 直接传模块 | ✅ 正确 |
> | 绑定方法（`obj.method`） | ✅ 正确——equinox 把具名方法访问包成 `BoundMethod`，它本身是 Module |
> | 普通闭包（`lambda x: obj.method(x)`） | ❌ **静默为 None** |
>
> 闭包是不透明的 `FunctionType` 叶子，`filter_grad` 看不进去。参数永远不更新，且不报错。注意 `__call__` 不享受 `BoundMethod` 待遇（equinox 排除 dunder）。
>
> **附带的方法论教训**：核验这条时，我用 `isinstance(grad.dist_fn, NoiseModel)` 判断，而绑定方法的梯度包在 `BoundMethod` 里，于是我的检查恒返回 None——我因此错误地报告"绑定方法也丢梯度"。**测量工具本身也会有盲点；内省要解开包装层再断言。**

### 1.4 与 rheplicant 的吻合（已验证，非断言）

| rheplicant 的东西 | 在 bayesmith 里是 | 验证 |
|---|---|---|
| 一整条 `Pipeline` | 一个**确定性节点** | 经 `build_forward_fn` 的既有接缝，零适配层 |
| `NoiseModel` | 一个**概率节点** | `realise(prediction, key)` = sample；`std` + `NoiseModelLikelihood` = log_prob。两半都已存在 |
| `Latent(linear=True)` | `linear_in` | 语义相同 |
| `NoiseModel.depends_on_prediction` | 分派判据 | 原样 |

**函数追踪对 rheplicant 是安全的**，理由是它自己的契约：`AbstractOperator` 明文规定 *"Only structural (shape/dtype) validation inside `__call__` — value checks would break under jit"*，所以用 tracer 跑一遍是安全的。pipeline 作为节点字段时，其数组叶子仍是图的 pytree 叶子，`eqx.partition` / `filter_grad` / `tree_at` 全部照常工作。

**一个必须记录的真实缺口**：rheplicant 的随机算子（`requires=("key",)`）是**只有采样器、没有密度**的——它经 `state.next_key()` 抽样并返回推进后的 state，不暴露 `log_prob`。因此它**不能**自动升格为概率节点。规则是：**pipeline → 确定性节点；NoiseModel → 概率节点**。

有意思的是，这恰好让 rheplicant 今天的一条"拒绝"归位：`refuse_stochastic_stages` 拒绝含随机算子的推断模型（因为闭包在模板 state 上会冻结一次噪声实现，造成实测 10.6σ 偏差且所有诊断都看不见）。在 bayesmith 里，随机性不再是"被冻结的抽样"，而是**一个密度因子**——但只有当它带着密度一起来时才成立。所以那条拒绝**仍然正确**，只是理由从"随机性有害"变成了"没有密度的随机性无法进入联合分布"。

---

## 二、作图与编译

**基底真相是显式图对象**；函数追踪是获得它的糖：

```python
def model(X, Y, Z, d_obs):
    x  = sample("x", Normal(0, 1), plate="obs")
    y  = sample("y", Normal(Y, 1))
    mu = det("mu", lambda x, y: pipeline_op(x, y, Z), linear_in=("x",))
    observe("d", Normal(mu, sigma), d_obs)

graph = trace(model, X=..., Y=..., Z=..., d_obs=...)   # 追踪一次
plan  = compile(graph)                                  # 结构分派
```

追踪只跑**一次**，产出一个静态图对象；之后所有分派、诊断、序列化都在图上做。静态结构（已定决策）保证追踪结果确定，这正是能这么做的前提。

`compile` 的产物 `InferencePlan` 是**可打印的**：

```
block 0  {x}          Wiener exact        (linear_in checked ✓, 3 scales)
block 1  {z}          enumerate 4 states
block 2  {sigma, nu}  NUTS (numpyro)      no exact structure found
```

**这是这个包最重要的用户体验**：模型在被拟合之前，先告诉你它将如何被拟合。没有任何追踪式 PPL 能做到。

---

## 三、分派表（真正的产品）

编译器遍历图，切成按推断方法划分的极大子图：

| 子图结构 | 方法 | 来源 |
|---|---|---|
| 隐变量线性高斯 + 共轭先验 | Wiener 后验均值 / GCR 精确抽样 | rheplicant `linear.py` |
| 线性高斯、协方差未知 | 迭代重加权 GLS | rheplicant `gls.py` |
| plate 上的线性高斯链 | RTS 平滑器（Kalman） | rheplicant `chain.py` |
| 有限支撑的离散隐变量 | 精确枚举边际化 | 新写 |
| 离散链 | 前向-后向 | 新写 |
| 上述若干块交替 | Gibbs | rheplicant `plan.py` |
| 其余一切 | NUTS | NumPyro |

最后两行的组合值得单独指出：**线性高斯链的 RTS 平滑器 + 离散链的前向-后向 = 切换状态空间模型**，一类真正困难且有价值的模型，而两半的一半已经在 rheplicant 里写好了。

### 不做的东西

分布库、MCMC/HMC/NUTS 内核、VI、变量变换、平板代数——全部借 NumPyro / distrax。**每写一行都要能回答"NumPyro 为什么做不了这个"**，答不上来就不写。

---

## 四、验收策略

分派器就是一个**阈值分派器**，因此适用 `boundary-validation.md` 的方法论：在分派边界处绕开分派器、直接比较两侧方法。

这里有一个罕见的好性质：**凡是够格走精确解的图，也一定够格走 NUTS**。所以每一条精确路径都有一个永远可用的参照物。

| 层 | 判据 |
|---|---|
| 精确 vs NUTS | 同一张图，精确解的后验矩 vs NUTS 的后验矩，在 MC 误差内一致（ESS≥400 时 z<4） |
| 精确 vs 闭式 | 线性高斯玩具模型对解析后验，rtol 1e-8 |
| 声明检验 | 每个 `linear_in` 声明在三个尺度上被探测；未通过则**拒绝走精确路径**并说明 |
| 追踪 vs 显式 | 同一模型两种写法产出**同构的图**（节点集、边集、静态字段全等） |
| plate vs 展开 | `vmap` 结果 vs Python 循环展开，rtol 0 |
| 离散枚举 vs 采样 | 小支撑集上枚举的精确边际 vs 大样本采样估计 |

**极端参数值必须覆盖**（方法论的硬要求，失效模式常呈 U 形）：单节点图、无隐变量图、无观测图、plate 大小 1 与 10⁶、离散支撑 2 与 1000、完全共线的两个父节点。

---

## 五、分解为子项目

这个设计包含多个独立子系统，不应写成一个 spec。建议顺序，每一步自成一个 spec → plan → 实施循环：

| 阶段 | 内容 | 为什么是这个顺序 |
|---|---|---|
| **P1 图核** | 节点类型、追踪、plate、log-density 组装 | 最小可算联合密度的东西 |
| **P2 NumPyro 桥** | 任意图 → NumPyro 模型 → NUTS | 让包**立刻可用**，同时提供交叉验证的预言机 |
| **P3 结构分派 + 线性高斯精确解** | 编译器 + 从 rheplicant 移植 `linear`/`gls`/`uncertainty` | 第一个"别人没有"的能力 |
| **P4 离散** | 枚举 + 前向-后向 | 与 P3 合成切换状态空间模型。**枚举半边已完成 2026-08-26**（`exact/discrete.py`，15 个测试，7 个变异中 6 个被具名测试杀死；第 7 个存活并因此揪出一段死代码——见下）；前向-后向仍未做 |
| **P5 图上诊断** | 可辨识性、先验敏感度、线性性检验（移植） | 依赖 P3 的图结构 |
| **P6 流式证据** | SqrtInfo / memory / chain（移植） | 最独立，可随时插入 |
| ~~**P7 mcpost 融合**~~ | GSA、重要性权重作用于图的输出 | **【owner 已拍板 2026-08-26：永久搁置。】** 不是推迟，是不做了——见下方批注 |

> **P4 枚举半边的三条实测记录（2026-08-26）：**
>
> 1. **`support` 从此是承重的。** `Probabilistic.support` 自 P1 起就带
>    `Discrete(n)`，而它自己的 docstring 写着「nothing in P1 reads it」。
>    `exact/discrete.py` 是第一个读它的东西。三条排除各自对应一种不同的错法：
>    observed 节点带着数据（对它求和等于把似然换成别的东西）、连续潜变量是被
>    条件化而不是被求和、而 `support=None` **既不当连续也不当离散**——按该字段
>    docstring 的原话，它对任何 support-specific 方法都不合格。
> 2. **代价先算再做。** 一个站点是 `n ** 坐标数`，站点之间相乘；plate 里的站点
>    是 `n ** plate_size`。这不是缺陷而是精确解的价格，唯一不诚实的做法是开始
>    求和然后祈祷。所以 `enumeration_states` 在任何算术之前就可算，越过预算就
>    拒绝并把数字写进消息里。
> 3. **一个存活的变异找出了一段死代码。** 原实现为「没有离散潜变量」写了一条
>    特判分支，把它换掉之后整个套件仍然全绿。原因是 `itertools.product()` 对空
>    输入产出**恰好一个空元组**——空积是 1 而不是 0，正是正确的数学，标准库已
>    经实现了它。那条特判从来没有被执行过。已删除，并在原处写明为什么不需要
>    特判。
>
> **仍未做的是前向-后向**，且它需要的不是新算法而是新的**结构识别**：链式的
> 离散潜变量今天可以表达（T 个节点，每个以前一个为父），但没有任何东西认得
> 出那是一条链。枚举是它的预言机——两者在小 T 上必须逐值相等，这正是 §二
> cross-check 协议的形状，也是枚举先做的理由。

> **P7 永久搁置（owner，2026-08-26）。** 记在这里而不是删掉，是因为一条被
> 删掉的阶段会被下一个读路线图的人当成漏写而重新提出——本仓库今天已经在
> A8.2 和 D2 上各吃过一次这个亏。
>
> 搁置的是**融合**，不是它想要的能力。P3 已经把 SNIS + ESS + k̂ 做对了（见
> `SNIS_ESS_FLOOR` 及其实测），所以「加权样本」这一半本来就在。剩下的
> 「把 GSA 作用上去」需要先修 mcpost 自身的正确性——附录二记的那个实测缺陷
> （十个 ARD 长度尺度全钉在下界、GP 坍成常数、S1=ST=0，而现有测试因为只断言
> `not isna().all()` 全绿）。为一个还不正确的东西造融合层，是把两个包绑在
> 一起而不是让其中一个变好。
>
> **什么会重新打开它**：mcpost 那个缺陷被修掉，并且有人真的需要在图的输出上
> 跑 GSA。在那之前这一行是记录，不是待办。
>
**P1+P2 是最小可行且自我验证的闭环**，应作为第一个 spec 的全部范围。

---

## 六、包的形态

全新仓库 `/Users/zzhang/projects/bayesmith/`，`git init` 起步，与 MCPost、rheplicant 都是独立仓库、独立发布节奏。MCPost 在本项目中**只读**（附录二记录的缺陷另案处理），rheplicant 在移植 P3/P5/P6 时**只读源码**。

```
bayesmith/                 ← 仓库根
├── pyproject.toml
├── src/bayesmith/
│   ├── graph/        节点类型、Graph 对象、追踪器、plate
│   ├── compile/      结构分派器、InferencePlan
│   ├── exact/        线性高斯（移植）、离散枚举、前向-后向
│   ├── bridge/       NumPyro 桥接与回退
│   ├── diagnose/     可辨识性、先验敏感度、线性性检验（移植）
│   ├── evidence/     SqrtInfo / memory / chain（移植）
│   └── errors.py     仅 stdlib
└── tests/
```

子包名用 `bridge/` 而非 `numpyro/`——同名子包会遮蔽顶层 `import numpyro`，是个现成的坑。

依赖：`jax`、`equinox`、`numpy`；`numpyro` 为**必需**（它是分派表的最后一行，不是可选）。Python ≥3.11。

**精度策略**（本次会话实测得出的硬约束）：bayesmith 作为库**绝不在任何位置调用 `jax.config.update("jax_enable_x64", ...)`**——它是进程级全局的，会静默改变宿主之后创建的每个数组的 dtype，而且关不回去（已存在的 float64 数组会静默截断）。需要 float64 时用 `with jax.enable_x64(True):`（已在 jax 0.10.0 与 0.11.0 上验证：线程局部、计入 jit key、可嵌套、退出完全还原），并在块内就转出到 NumPy。证据层的算术**必须**在 x64 下跑（其 offset 约 7e11 对差值约 1e5，float32 下零位有效数字）。

---

## 附录一：本次会话验证过的迁移事实（P3/P5/P6 仍要用）

从 rheplicant 移植推断层的工程结论不受本次转向影响，只是目标从 mcpost 改成 bayesmith：

- **边界极窄**：`inference/` 只从 `core` 导入 5 个符号（两个异常类、`AbstractOperator`、`State`），外加 `parameters.py` 独有的 `RANDOMNESS/describe_stages/stages_requiring` 与私有的 `core.graph._aliased_leaf_paths`。对 `radio/`、`config/`、`gui/` 的导入为 **0**。
- **异常类不对称**：`ParameterSpaceError` 在 core/radio 中 raise **0 次**；`StateValidationError` raise **150 次**。52 处测试导入全部来自 `rheplicant.core.errors` → 必须共享 identity。
- **`except DirtError` 在生产代码中 0 处**，`pytest.raises(DirtError)` 全仓库仅 3 处，其中只有 `tests/core/test_basis.py:420` 会被打破。
- **测试耦合实测**：广义耦合下 `tests/inference/` 只有 **6 个**文件完全干净；`tests/evidence/` 的 32 处 core 导入里 29 处是异常类，其余 3 处各 1 次（coordinates/operator/state）——三个 toy 替身即可整体搬走。
- **薄壳机制已原型验证**：sys.modules 别名对私有名、fresh 进程首次深导入、任意嵌套均可用，零磁盘 stub；但 `python -m pkg.sub` 会坏（已核实 rheplicant 无此入口），别名列表须用 `pkgutil.walk_packages` 程序化生成。
- **钩子机制已原型验证**：调用时读取的模块级全局量，8/8 组合触发，`from X import func` 不破坏迟到安装；但必须导出**函数**而非全局量的**值**，且 jit 会击穿"调用时"语义（trace 缓存）。
- **`jax.scipy.optimize.minimize(method="BFGS")` 不可用于 ML-II**：8/8 报 `success=False`，6/8 未达最优。改用 scipy 的 L-BFGS-B 驱动 `jax.jit(jax.value_and_grad(...))`。

## 附录二：mcpost 的独立问题（不进 bayesmith，但需处理）

本次调研实测发现 mcpost 现有 GSA 在它自己的主 fixture 上是坏的：`large_sample_data`（1000×10）下十个 ARD 长度尺度全部钉死在下界 0.01、GP 坍塌成常数、**S1=ST=0**；`sample_data`（100×5）下 **ST=4.61**（必须 ≤1）。根因是 `gsa/kernels.py:24` 的 `length_scale_bounds=(1e-2,1e2)` 对 minmax 缩放数据下界太高。现有测试全绿，因为只断言 `not isna().all()`。

另有三个独立缺陷：Sobol 置信区间**从未被播种**（SALib 用 `if seed:`，而 mcpost 默认 `gp_random_state=0` 是 falsy）；`qmc_integral_importance` **数学错误**（均匀抽样后又除以 q，实测返回 287.8 而解析真值 2.0）；`monte_carlo_integral` 的不确定度除以 N 而非它自己算出的 ESS。主分支还有 4–5 个红测试（间歇性）。

**这些应作为 mcpost 自己的修正版发布处理，与 bayesmith 无关**，且优先级高——它们会给出错误的科学结论。建议核对 Zhang et al. (2026) 中用到 mcpost 的那次分析：查其 `ARD_LS` 是否贴着 0.01 或 100 的边界。
