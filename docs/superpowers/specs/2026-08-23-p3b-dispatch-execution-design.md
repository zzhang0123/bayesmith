# bayesmith P3b — 分派器、InferencePlan、Gibbs 与近似块的修正

> 本文是设计（spec），不是实施计划。
> 上游：`docs/superpowers/specs/2026-08-23-p3-structural-dispatch-design.md`（下称「P3 spec」）的 §四、§五、§六、§七。
> 前置：P1 图核 + P2 NumPyro 桥 + P3a 精确解核心（已合入 main，HEAD `0c03ea5`，242 测试全绿）。
> **P3a 已发布代码里的两个守卫缺陷（§1.4、§1.5）并入本计划，作为最前面的两个任务**——它们必须先于分派器完成，因为分派器正是把这两个守卫的输出当作判据来读的。
>
> **本文相对 P3 spec 的地位**：P3 spec 的 §四/§五/§六 写于 P3a 开工之前。本文更正其中数处并补齐它留白的四处。**冲突处以本文为准**，P3 spec 对应位置已插入指针。
>
> **修订历史**：本文初稿经四路独立审查——第一性原理数学、对抗性反例构造、实测审计、验收关口覆盖度——发现约 30 处缺陷，**其中若干是初稿自己的实测错误**。§〇 记录被推翻的初稿主张：「哪些说法曾经是错的」与「现在的说法是什么」同样承重。

## Context

P3a 交付了无矩阵线性算子、线性性检验、Wiener 解、GCR 精确抽取、迭代 GLS 与稠密 Fisher——**但没有任何东西读它们**。P3b 是这些决定第一次由**图**做出。P1 记录的 `linear_in` 在 P3a 被**检验**过但没有被**读**过；P3b 是它第一次决定计算路径。

---

## 〇、初稿被推翻的主张

| 初稿主张 | 实测 | 现状 |
|---|---|---|
| 「`compile/` 与 `bayesmith.compile()` **不能共存**」 | **假**。两种变通经端到端验证 | 结论（改名）不变，**理由换掉**，§1.1 |
| 「冷启动 `from pkg import compile` 拿到 module」 | **假**，拿到**函数**。初稿那次测量是在同进程已 `import pkg.compile.dispatch` **之后**跑的 | 表格行已删 |
| 「`_psis_khat` 不能 jit **因为**返回具体 float」 | 理由错。真阻断是 `np.sort` 与**数据依赖形状的布尔掩码索引** | §二。写成「返回 float」会诱使人「修」成 jnp 标量 |
| `gibbs_fn(rng_key, gibbs_sites, hmc_sites)` 的元组记法 | 误导。numpyro **纯按关键字**调用。**参数名是契约，顺序自由** | §二 |
| 「返回值只需含 gibbs 成员」 | 必须**恰好**是。少一个 → **空消息 `AssertionError`**；多一个指向 NUTS 隐变量 → **静默忽略** | §二 |
| 「`check_gaussian` 是唯一 trace 不安全处」 | 结论对（经七次证伪），但**调用点有两个**：`block.py:296`（`_env_before` 内，逐成员）与 `:385` | §1.2 |
| 「`exact → bridge` 的惰性 import 在破环」 | 假。`bridge` 不 import `exact`，AST 扫描零环 | 标注为**导入开销**选择 |
| §7.1 的 `shared_ancestor` 行 | **与判据算出来的相反** | §3.1 删除判据 2 |
| 「`models.py` 没有非高斯**隐变量**」 | 假。`dist.Cauchy` 与 `dist.ImproperUniform` 都在。真空缺是**观测节点**非高斯 | §7.1 |
| §1.3 的三组基准 | 全部 **float32、单种子、短跑**，多数落在自身分布的极端 | §1.3 |

**唯一没有被推翻的核心**：§1.3 的结论不是建在 bug 上。审计者固定外参数、抽 8000 个 iid 样本对 `tests/exact/oracle.py::graph_oracle` 比对，均值 max|z| = 2.6–3.0，六个变异全部变红（σ×2 → z=34.8；错 `at` → 234；重用 key → 162.5）。全流程长跑收敛（Gibbs N=50000 → +1.3759 对 NUTS +1.3751）。

---

## 一、五处实测更正——三处在 spec，两处在 P3a 已发布的代码里

### 1.1 目录改名 `dispatch/`，但理由是 pytest

**两者可以共存**（初稿说不能，错了）——可调用子包、父包 `__setattr__` 守卫，两种变通经验证。但代价真实：可调用子包让 `inspect.signature` **静默丢掉第一个位置参数**、不可序列化、强制 eager import。

**真正的危险**：**先导入者获胜，而 pytest 的收集顺序让它变成确定性的。** 一个测试文件写 `from bayesmith.compile.partition import ...`，就会让**同一 session 里**的公共 API 测试失败——单独跑通过，全套跑失败，`xdist` 下同样。`exact/__init__.py` eager 拉进全部八个子模块，会放大它。

这条比初稿编的「第 1 行能用、第 50 行坏掉」更值得写下，因为它**注定发生在这个仓库的测试布局里**。

**决定**：目录 **`src/bayesmith/dispatch/`**，公开名 `bayesmith.compile(graph)` 不动。

### 1.2 `unchecked_operator` 在 trace 下不可用

P3 spec §4.1 写 `linear_operator(graph, names, at=at, check=False)`。**两处与实现不符**：没有 `check=` 关键字；且 `gibbs_fn` **在 trace 下运行**，而 `check_gaussian` 做 `bool(jnp.all(...))` 与 `raise`——实测 `TracerBoolConversionError`。

**调用点有两个**（初稿只点了一个）：`block.py:385`（逐观测节点）与 **`block.py:296`（`_env_before` 内，逐块成员）**。只关掉前者仍抛错。而 `_env_before` 另有三个调用方（`linearity.py:295`、`oracle.py:111`、`test_block.py:120`），**§八「+15 行」的估计偏低**。

**「唯一的 trace 不安全处」经七次证伪尝试仍成立**：两观测节点不同形状、双成员块、σ 依赖预测、成员先验尺度依赖被 trace 的块外隐变量、两外层隐变量、`numpyro.enable_validation(True)`，全部可 trace。

**但作用域仅限库代码**：`fn` 里写 `float(b_) > 0` 的 `Deterministic` 会抛 `ConcretizationTypeError`。分派器在 trace 下跑**任意用户 `fn`**，这句必须进 docstring。

**决定**：`unchecked_operator` 与 `_env_before` 都加 `probe_gaussian: bool = True`。

### 1.3 ESS/秒不是关口，且初稿数字全是 float32 单种子

优势**随被测量的量反向**：块内 GCR 消掉自相关 → 赢；块外条件于块使外参数被钉死 → 输。良态块两个方向都输。

**具体数字不可作为阈值**：

| | 初稿 | 实测范围 |
|---|---|---|
| 块内比值（float32，5 种子） | 5.98x | **4.72 – 19.39x** |
| 块内比值（**x64**） | — | **24 – 43x** |
| 块外比值 | 0.24x | 稳态 **0.17x**；x64 下 0.60x |

**断言 `≥5.98x` 换个种子（4.72x）就红。** 另两条：

1. **三组基准全跑在 float32**，而 NUTS 自己的答案从 +1.374（f32）移到 +1.441（x64）——**float32 的「真值」本身差 0.065**。
2. **块内胜出完全取决于 `tol`/`maxiter`/dtype**：x64 下 tol=1e-6 → 44.6x，**tol=1e-4 → 1.58x**（抽取变错，块内 ESS 从 286.8 塌到 9.0）。float32 下 CG 在块 C 停在相对残差 ~4e-3、**永远达不到任何请求的容差**，`maxiter` 才是真停止规则。

**决定**：正确性是关口；ESS/秒 降为**不做数值断言的报告**（§7.2(f)）。

### 1.4 B1：`affinity_errors` 在整个陪域上归一化（**P3a 代码缺陷**）

`exact/linearity.py` 的 `_biggest` 是 `max(|leaf|)` **跨所有叶子**：

```python
variation = _biggest(jax.tree.map(jnp.subtract, actual, baseline))   # 全局 max
departure = _biggest(jax.tree.map(jnp.subtract, actual, predicted))  # 全局 max
errors[scale] = departure / max(variation, 1e-300)
floor = 1e4 * epsilon * max(_biggest(actual), _biggest(baseline))    # 也是全局
```

一个**亮**的陪域分量因此同时决定了一个**暗**分量的**标尺**与**舍入地板**。两个稀释独立发生。

实测——诚实的亮分量 + 谎报 `linear_in` 的暗分量，暗分量主导后验：

| | 结果 |
|---|---|
| 暗分量**单独** | `StructureError: ... not affine in it` ← 正确拒绝 |
| 亮 + 暗**同图** | `check_linearity` 最大相对误差 **3.451e-14** → **通过** |
| 「精确」答案 vs 真值 | `w = +1.125` vs `+0.803` ⟹ 偏 **202 个真后验 sd** |

**并且不需要两个节点**——同一个稀释发生在**一个数组的元素之间**（一个亮前景通道 + 五个暗信号通道：报 1.610e-14 通过，偏 **154 个真 sd**）。**这正是本包的目标动态范围**（前景 K、信号 mK）。两个 `fn` 都不是对抗性构造的，只有幅度比是。

**修法**：`errors` 与 `floor` **逐叶**计算（理想逐元素），任一叶失败即失败。**更好的一步**是按 `1/sigma` 加权后再比较——要紧的不是「偏离相对预测幅度多大」，而是「偏离相对似然要除的那个噪声多大」。

> **实施前必测**：加权版本会不会让今天通过的合法 fixture 变红？`cubic_tail`（curvature=1e-6）设计成「曲率真实但可忽略」，加权后可能翻红。**必须先出一张全部现有 fixture 在新旧两版守卫下的判决对比表**，任何翻转都要具名解释。这是本计划的第一个任务。

### 1.5 B2：`check_prediction_dependence` 沿一条射线探测（**P3a 代码缺陷**）

`exact/gls.py`，`DEPENDENCE_PROBES = (1.0, -0.5)`：

```python
probe = {name: centre[name] + factor * block.prior_std[name] for name in block.names}
```

**每个成员按同一个带符号的倍数移动。** 依赖两个等宽成员**对比量**的 σ 在这条射线上恰好是常数——探针从未离开它的水平集。

实测（两成员同回归量，即 §7.1 要求同块的 `collinear_pair` 形状）：

| | a | b | a−b | 隐含 σ |
|---|---|---|---|---|
| 「精确」 | +0.2549 | +0.2549 | **+0.0000**（sd 1.4142） | 0.3000 |
| 长跑 NUTS | +1.0561 | −0.5477 | +1.6038（sd 0.0486） | 1.4915 |
| 真值 | +1.0000 | −0.5000 | +1.5000 | 1.3445 |

σ 移动测得 **0.000e+00** → 判为 `gcr`。对比量偏 **33 个 NUTS sd**，sd 涨 **29 倍**。**最坏的部分**：整图一个块 ⟹ 走 §6.4 第一行 ⟹ **直接 GCR、iid、无链** ⟹ **没有 r̂、k̂、ESS，什么都没有可诊断**。

声明救不了它：`d` 带默认 `depends_on_prediction=True`，`check_prediction_dependence(declared=True)` 只返回移动量。分派**必须**由探针决定。

**修法**：沿**逐成员独立**的方向探测，且至少含一个**符号混合**的方向。可复用 `check_linearity` 已有的 `fold_in(root, position)` 逐成员随机探针。`sigma_of` 的调用次数不变。

### 1.6 三个 break 的共同根因

§1.4、§1.5 与 §3.2 的孤儿密度因子是**同一个形状：用全局归约代替逐部分检查**。§1.4 把逐叶相对误差归约成陪域上的 max；§1.5 把多维依赖探测归约成一条射线；§3.2 把「模型」归约成 `{成员先验} ∪ {观测节点}`。

而**本包已经为完全相同的理由做过相反的选择**，并把理由写进了 `exact/gaussian.py::check_gaussian` 的 docstring——「一个 1e6 里错一个元素、差 50 nats，求和形式报 1.95e-5 静默接受，逐元素报 50」。**这不是一条新洞察，是一条已经写下来却没有被推广的洞察。**

---

## 二、上游 API 事实（实测，经审计修正）

| 事实 | 值 |
|---|---|
| `HMCGibbs.__init__` | `(self, inner_kernel, gibbs_fn, gibbs_sites)` |
| **`gibbs_fn` 调用方式** | **纯关键字**（`hmc_gibbs.py:168`）。**参数名是契约，顺序自由**；positional-only `/` 抛 `TypeError` |
| `gibbs_sites` | `{块成员名: 值}`，trace 期为 `DynamicJaxprTracer` |
| **`hmc_sites`** | 块外隐变量**加 `deterministic` 站点**，且值在**约束空间**（`hmc_gibbs.py:166` 先过 `postprocess_fn`） |
| **返回值** | 必须**恰好**是 gibbs 成员。少 → **空消息 `AssertionError`**；多且指向 NUTS 隐变量 → **静默忽略**；多且指向别的 → 响亮 `ValueError`。**返回约束空间外的值静默接受**，链冻在 −inf |
| 调用次数 | **整个运行恰好一次**（trace 期）。`jax.disable_jit()` 下每迭代一次、值具体 |
| `rng_key` | 是 `mcmc.run` 传什么就是什么，**非上游保证**。`PRNGKey(0)` 原样是 `uint32[2]` |
| **`chain_method='vectorized'`** | **在 0.21.0 下坏掉**：`HMCGibbs.init` 无条件 `random.split`，而 vectorized 递批量 key。`sequential`/`parallel` 可用 |
| `_psis_khat` | `importance._psis_khat(log_weights) -> float`，**私有** |
| **不可 jit 的原因** | `np.sort`（`:158`）与**数据依赖输出形状的布尔掩码索引**（`:168`）。返回 `float` 只是症状 |
| `_psis_khat` 标定 | 初稿的 0.184 是 20 种子里的**最大值**（均值 0.076、sd 0.066），且强依赖 N。**钉决策阈值，或把 `(seed, N)` 与 ≥0.05 容差一起钉死** |
| `log_joint` | `(graph, values) -> jax.Array` 标量 |
| `exact → bridge` | 惰性 import **不在破环**——是**导入开销**选择 |

---

## 三、分区推导

### 3.1 合格性判据（**初稿的判据 2 已删除**）

隐变量 `x` 合格，当且仅当：

1. `x` 自己的分布是对角高斯；
2. 每个观测节点都是对角高斯；
3. 从 `x` 到每个观测节点的 `loc` 的**每一条**路径，沿途每个 `Deterministic` 都在 `linear_in` 里声明了其入边父节点。

> **实测更正：删除初稿的判据 2**（「`x` 的先验的父节点不含另一个**候选**块成员」）。它**自我循环**——候选集正是它要算的东西——且按字面实现给出**错误分区**：
>
> `shared_ancestor` 里 `x ~ Normal(0, |tau|)`，`x` 的 parents 是 `('tau',)`，`tau` 是候选 ⟹ `x` 不合格 ⟹ `Q = {tau}` ⟹ 块 = `{tau}`，而 `tau` 的 `A` 恒为零 ⟹「精确」答案永远是 `tau` 的先验。**与初稿 §7.1 写的恰好相反。**
>
> 删掉它，`x` 与 `tau` 都合格，§3.2 的祖先规则弹出 `tau`，得到正确的 `{x}`。**判据 2 是主动有害的。**

**异常处理按 raise 点具名**（初稿在此自相矛盾）：

| raise 点 | 怎么办 |
|---|---|
| `gaussian_parts` / `check_gaussian` 的 `NotGaussian` | **捕获** → 不合格 → NUTS |
| `check_gaussian` 的 `StructureError`（自称高斯而 `log_prob` 不符） | **绝不捕获** |
| **`check_linearity` 的 `StructureError`**（声明线性而不仿射） | **在该调用点单独捕获** → 整块 NUTS |

> **实测更正**：初稿 §3.1 写「`StructureError` 绝不捕获」，而 §3.2 规则 4 要求联合检验失败落 NUTS——`check_linearity` **正是**抛 `StructureError`（`linearity.py:83/167/201`）。两条直接矛盾。修法是**按 raise 点而非按类型**决定；宽 `except` 会让「自称高斯而密度不是」被静默降级，正是 `errors.py` 明令禁止的。

### 3.2 分区规则

1. 按 §3.1 分出合格集合 `Q`。
2. **祖先出块**：若 `z ∈ Q` 是**任何其它隐变量**的祖先——**合格与否都算**——则 `z` 离开 `Q`。
3. 剩下的 `Q` 构成唯一一个块。
4. 跑 `check_linearity`（多 `at` 点）；失败 → 整块 NUTS，点名成员。
5. σ 依赖探测决定方法。

> **实测更正：规则 2 的「任何隐变量」是承重的。** 初稿写「另一个**合格**隐变量」，留下一个静默错误答案的洞。
>
> `unchecked_operator` 只读**块成员的先验**与**观测节点**。一个块外隐变量 `v`，其密度依赖块成员 `w`，因子 `p(v|w)` 被**静默丢弃**。把 `v` 用任一判据判为不合格（如 `v ~ StudentT(3, w, 0.4)`），初稿的规则 2 就不弹出 `w`：
>
> | | mean(w) | sd(w) |
> |---|---|---|
> | 真值 | +1.9759 | 0.4816 |
> | 长跑 NUTS | +2.0004 | 0.4809 |
> | **P3b 链** | **+0.4106** | **1.7723** |
>
> 偏 3.2 真 sd，sd 涨 3.7 倍。**而 `graph_oracle` 复现同样的错误答案**——它从 `isolate` 与 `observation_parts` 取材，同一个盲区。现有稠密预言机看不见这一类。
>
> 判据 3 也救不了：把 `w` 经 `v` 通到第二个观测节点、路径上不放 `Deterministic`，判据 3 就**空真**。

### 3.3 为什么是「一个块」

> **实测更正：初稿此节的论证是误诊，结论对。** 初稿说交替推进给出错误答案。**实测证否**——双线性图上两块**精确** Gibbs 交替，60 万 sweep 对二维求积：
>
> ```
> 交替 2 块精确 Gibbs : E[g]=1.65226 sd=0.53256  E[t]=1.27484
> 二维求积（真值）    : E[g]=1.66033 sd=0.53473  E[t]=1.26904
> ```
>
> 四位有效数字。**交替的精确 Gibbs 渐近正确**；失效是**混合**（积分自相关时间 ≈ 50，随块间相关趋于 1 而无界），不是平稳分布。rheplicant 引的失败是**块坐标下降**在脊上游走的**估计**失败，被搬进了**采样**论证里。
>
> 另外「共同影响同一观测节点 ⟹ 后验相关」作为全称命题**是假的**：正交设计（`A=[[1,1],[1,−1]]`、等 σ、等宽独立先验）给出对角 `M` 与恰好独立的后验。正确说法是「一般相关」。

**联合成块的正确理由**：

1. **联合 `check_linearity` 探针是唯一能证伪 `linear_in` 声明的东西**——`gain × t_ant` 各自条件仿射、联合不仿射。
2. 交替虽渐近正确，但**按块间相关的速率混合**，而块间相关正是图看得见、用户看不见的东西。

**这条更正承重**：它改变 §7.2 允许断言什么——**不得断言「交替给出错误答案」**。

### 3.4 判据 3 的两种「空真」，都必须被接受

| fixture | 成因 | 为什么对 |
|---|---|---|
| `plated_latent` | 路径上**没有 `Deterministic`** | 全称量词对空集成立 |
| `unconstrained_latent` | **到任何观测节点都没有路径** | `A` 的列恰为零，答案是先验均值 |

朴素实现（「存在某个 `Deterministic` 的 `linear_in` 点了 `x`」）会**把两个都错误拒绝**。判据 3 在「路径条数」上分支，**0 必须取到两次**——0 条 `Deterministic` 与 0 条路径是不同的零。

> **限定条件**：`unconstrained_latent` 合格，**当且仅当它同时不是任何隐变量的祖先**（§3.2 规则 2）。

### 3.5 「失败全落」的代价

搜索极大合格子集是指数的且**子集不唯一**（去掉 `gain` 还是 `t_ant`？两者都能让剩下的通过，分区不同）。拒绝时点名成员，让知道模型含义的人来定。

### 3.6 `at` 与 `at_points`（实测）

块是真子集时必须为块外隐变量构造 `at`——用**先验均值**。

**`at_points` 是另一件事，且有陷阱**：`check_linearity` 默认从块外隐变量的先验**抽样**，实测两个**合法**模型因此抛错：

```
overflowing_outside_latent -> StructureError: prediction already non-finite at prior draw 1
improper_outside_prior     -> StructureError: outside latent cannot be sampled from its own prior
```

**决定**：`at_points` 从**高斯**块外隐变量的先验抽，其余用先验均值/零，退化在 `Block.reason` 里具名。

> **不得**为绕开这两个错误而传 `at_points=[at]`——那会把 `check_linearity` 降格成它自己 docstring 点名禁止的 moderate-parameter probe。`affine_only_at_zero` 是抓这个的 fixture。

---

## 四、Gibbs 装配

### 4.1 形态

`gibbs_fn` 必须用**参数名** `rng_key` / `gibbs_sites` / `hmc_sites`（§二），返回**恰好**块成员，值在**约束空间**。

### 4.2 四个必须外提的守卫（初稿写三个）

| 守卫 | 编译期 | sweep 内 |
|---|---|---|
| `check_gaussian` | 跑一次 | `probe_gaussian=False`（**两个调用点**） |
| `check_linearity` | 多 `at` 点跑一次 | 不跑 |
| `condition_bound` | 见下 | 见下 |
| **`noise_std`** | **不可外提** | **每 sweep 重算** |

> **`noise_std` 必须进表**（初稿漏了）。σ 可依赖**块外**隐变量，而 `check_prediction_dependence` 只移动块成员，对这种图报 0.0。实测 `d ~ N(w·X, exp(lognoise))`：块 σ 移动 0.000e+00，而 `lognoise` 在 ±2 内使 σ 动 **639%**；冻结则后验窄 **17 倍**。**判据：任一观测节点的 scale 有隐变量祖先，`noise_std` 就必须在 sweep 内重算。**

> **`condition_bound` 不能无条件外提**（初稿说可以）。κ 是块外隐变量的函数——正是祖先规则造出来的那种图。实测 `indirect_ancestor`：
>
> | τ | 0.0 | 1.0 | **2.0（先验均值）** | 4.0 | 6.0 |
> |---|---|---|---|---|---|
> | `condition_bound` | 1.57 | 69.7 | **2.51e2** | 9.56e2 | 2.11e3 |
>
> 钉在先验均值的 κ 在 τ=6 时小 8.4 倍，方向是**让 `tol` 太松** → CG 早停 → 后验太窄 → 守卫关着无人察觉。正是 §4.2 引的 rheplicant 组合。
>
> **决定**：`_condition_bound` 是 12 次幂迭代的纯 `jnp`，trace 安全。块外隐变量影响 κ 时**在 sweep 内重算**；否则外提。`InferencePlan` 打印**区间**而非点值。

**关掉守卫必须同时收紧 `tol`。** `InferencePlan` 必须把 κ（或区间）与 `tol` 都打印出来。

### 4.3 上游接口风险

一条测试钉住 §二 的**全部**契约：关键字调用、`hmc_sites` 含 deterministic 且在约束空间、返回值必须恰好是成员、`vectorized` 不可用。

---

## 五、近似块的修正

### 5.1 重要性权重（**符号已更正**）

对 `q = N(μ, M⁻¹)`，`log q(y) = −½rᵀMr + ½ log det M − (n/2) log 2π`。于是

```
log w = log p − log q = log p(x, z, d) + ½ (x−μ)ᵀ M (x−μ) + C ,
        C = −½ log det M + (n/2) log 2π
```

> **实测更正**：P3 spec 与本文初稿都写 `C = ½ log det M − (n/2) log 2π`——**两个符号都反了**。二次项与 log-det 项必须**反号**，任何约定都救不了同号。实测（4 维，对 `scipy.stats.multivariate_normal` 校验）：初稿的 `C` 误差 **+2.396e-01**，更正后**逐位为零**。
>
> 路径 (A) 里无害（每次运行的常数，自归一化杀掉）；路径 (B) 里不无害。

### 5.2 (A) 整图即一个块 → SNIS

`iterative_gls` 求 σ̂ → `wiener_solve` 得 μ → `gcr_sample` 抽 N 个 iid → `w̃ = softmax(log w)`。

> **实测更正：路径 (A) 有维度上限**，初稿无条件推荐它。自归一化 IS 随失配坐标数**指数退化**。实测（`A = I`，σᵢ = 0.3|xᵢ|+0.05，温和的 radiometer，提议在 GLS 不动点，N=40000）：
>
> | n | 1 | 25 | 50 | 100 | **500** | 4000 |
> |---|---|---|---|---|---|---|
> | Kish ESS | 2509 | 124 | 4.5 | 6.0 | **1.00** | **1.00** |
>
> n≈500 时一个抽取吃掉全部权重，而 §7.3 要求块大小到 1e4。
>
> **决定**：§6.4 的分派必须在**实测 Kish ESS/N 塌缩**时回退（到 (B) 或 NUTS），而不是对 1e4 的块返回 `unreliable=True` 的对象。

### 5.3 (B) 块嵌在 Gibbs sweep 里 → **独立提议** + MH

> **实测更正：本文最大的一处更正。** P3 spec 与初稿都写「σ̂ = σ(m(x)) 依赖当前状态，所以提议不是独立提议，必须在 `x'` 处**再建一次**提议」。
>
> **那样写 `log det` 不抵消。** 前向 `M = M(x)`、反向 `M' = M(x')`，残留 `½(log det M' − log det M)`，恰在 σ̂ ≠ σ̂′ 时非零——正是路径 (B) 存在的情形。而 `log det` 是无矩阵方法**唯一算不动**的量。
>
> 实测（1-D，对求积真值；两个独立实现一致）：
>
> | 链 | 均值误差 | 接受率 |
> |---|---|---|
> | 按初稿（状态依赖 σ̂，无 log det） | **+0.0357（≈27 SE）**，另一种子 +0.0385 | 0.777 |
> | 带 log det（正确但算不动） | −0.0014（≈1 SE） | 0.725 |
> | **独立提议，σ̂ 冻在 GLS 不动点** | **−0.00046** | 0.754 |
>
> 略去 log det 的链的平稳分布有闭式：**π ∝ p(x)·det M(x)^{−1/2}**（经求积确认）。而**接受率看起来比正确的链还健康**——正是 §5.3 自己警告的静默错误。

**决定：σ̂ 改为只依赖块外状态 `z` 的确定性函数**（GLS 不动点给定 `z`，或更便宜的块先验均值处的 σ）。于是：

* 提议**真的**独立，`M' = M`，`log det` **真正抵消**；
* `α = min(1, w(x')/w(x))`，用 §5.1 改正符号后的 `log w`；
* **成本从 3 次 CG 降到 2 次**（μ 一次 + 抽取一次；两个密度都只是二次型，各一次算子应用，无需反向重建）；
* **正确性不依赖 σ̂ 有多好**——任何 x-无关的 σ̂ 都给出有效的链，σ̂ 的质量只影响**接受率**。正确性风险变成性能旋钮。

**反向密度项仍然必需**（`q(x)/q(x')` 不消失），§7.2(b) 的守卫依旧有效。

**σ̂ 绝不可取自「上一次接受的 x」**——那依赖链的历史，是自适应链，只在 diminishing adaptation 下有效。

### 5.4 诊断

- **Kish ESS** `= 1/Σw̃²`；**不确定度是 `variance / ESS`，即 `sd / √ESS`**（初稿写「除以 ESS」有歧义，正是它要防的那个 bug 的形状）
- **k̂ 分带（已更正）**：`k̂ < 0.5` 有限方差；**`k̂ < 1` 有限均值**（初稿写 0.7——那是经验可靠性阈值，不是矩存在的界）；不可靠阈值取 **`min(1 − 1/log₁₀N, 0.7)`**（Vehtari 等 2024，N < 1e4 时低于 0.7）
- 最大权重占比；(B) 的接受率

`_psis_khat` 私有：调它 + 钉住测试；导入失败降级为只报 Kish ESS。**整个诊断在 jit 之外算**（§二）。

### 5.5 一个 k̂ 看不见的失效模式

`log q` 的正确性依赖 CG 真的收敛；k̂ 测的是权重尾部，不是 q 是否是真实的抽样分布。缓解：`tol` 由 κ 定；§7.2(d) 的**三点**测试。

**float32 下 CG 可能根本达不到任何请求的容差**——实测块 C 停在 ~4e-3，`maxiter` 才是真停止规则。**修正路径的测试必须在 x64 下跑。**

### 5.6 不做的推广

`fn` 非线性的块仍然拒绝，理由具名（需 MAP 作线性化中心，P5）。

---

## 六、InferencePlan 与返回类型

### 6.1 对象

```python
class Block(eqx.Module):        # 全静态字段（实测：静态 dict 字段可构造、可 flatten、treedef 可 hash、可 jit）
    latents: tuple[str, ...]
    method: str        # "gcr" | "gcr+snis" | "gcr+mh" | "gls" | "nuts"
    reason: str
    linearity: dict | None
    kappa: float | tuple[float, float] | None    # 点值，或 §4.2 的区间
    tol: float | None
```

### 6.2 可打印

κ（或区间）与 `tol` 必须在表里；§3.5 的拒绝理由与 §3.6 的 `at_points` 退化也必须在。

### 6.3 返回类型

```python
class Posterior(NamedTuple):
    samples: dict[str, jax.Array]
    log_weights: jax.Array | None
    ess: float          # SNIS: Kish。链: 对所有 site 与坐标取 MIN
    khat: float | None
    unreliable: bool    # khat >= min(1 - 1/log10(N), 0.7)
    method: str

class Estimate(NamedTuple):
    values: dict[str, jax.Array]
    noise_std: dict[str, jax.Array]
    converged: bool
    residual: jax.Array
    iterations: jax.Array
```

> **实测更正：`ess` 的归约必须写下来。** SNIS 下 Kish ESS 是一个数；链下 ESS 逐 site 逐坐标。§1.3 的基准 C 正是例子：`ESS(logw)=3.0` 与 `ESS(alm,min)=40.2` 并存。**取 min**——这个字段存在的全部理由是让「除以 N」变成要主动写出来的事，那它必须报最坏的那个。

`estimate()` 在 `converged=False` 时抛 `ConvergenceError`——P3a 预期的第一个调用点。

### 6.4 `sample()` 的分派

| 图的形状 | 走什么 |
|---|---|
| 整图一个精确块，σ 不依赖块 | 直接 GCR——iid，无链 |
| 整图一个块，σ 依赖块，**Kish ESS/N 未塌缩** | GCR + SNIS |
| 整图一个块，σ 依赖块，**ESS/N 塌缩**（§5.2） | 回退：(B) 或 NUTS，理由具名 |
| **σ 依赖块外隐变量** | 精确块照常，`noise_std` **每 sweep 重算**（§4.2） |
| 无精确结构 | NUTS |
| 混合 | HMCGibbs；σ 依赖的块用 §5.3 的**独立提议 MH** |

### 6.5 `estimate()` 的分派

整图精确 σ 常数 → `wiener_solve`；整图精确 σ 依赖预测 → `iterative_gls`；混合 → **拒绝**，指向 `sample()`（P5）。

---

## 七、验收：两道关口

### 7.1 关口一：结构，零 MCMC

**下表每一行都是算出来的。** 初稿有两行错——`unconstrained_latent`（§3.4）与 `shared_ancestor`（§3.1）。

| fixture | 候选集 | 结果 |
|---|---|---|
| `two_linear_latents` | `{a, b}` | `gcr` |
| `bilinear_pair` | `{gain, t_ant}` | 联合 `StructureError` → **整块 NUTS**，点名两成员 |
| `radiometer` | `{w}` | σ 移动 2.50e+03 → `gcr+snis` |
| `radiometer_group` | `{a, b}` | σ 移动 4.40e+02 → `gcr+snis` |
| `indirect_ancestor` / `diamond_ancestor` / **`shared_ancestor`** | `{x}`（`tau` 出块） | `x` 精确、`tau` NUTS。**判据 2 删除后才对** |
| `quadratic_claim` / `cubic_tail` | `{w}` | `StructureError` → NUTS |
| `collinear_pair` | `{a, b}` | 共线父节点**同块** |
| `unconstrained_latent` | `{w, u}` | `gcr`（判据 3 空真，且 `u` 非任何隐变量的祖先） |
| `plated_latent` | `{z}` | `gcr` |

**必须新增的 fixture**，每个关掉一个今天零覆盖的分支：

| 新 fixture | 关掉什么 |
|---|---|
| `mixed_radiometer` | **`gcr+mh` 今天零覆盖**——所有 σ 依赖的 fixture 都是整图块，全部路由到 `gcr+snis`。**删掉整个 MH 分支，12 行全绿** |
| `student_t_likelihood` | **判据 2（观测节点非高斯）零覆盖**——29 个 `observe()` 全是 `dist.Normal` |
| `orphaned_child_latent` | §3.2 规则 2 的「任何隐变量」 |
| `lying_observed_node`（用已有的 `LyingNormal`） | `StructureError` 必须**穿透**而非降级 |
| `lying_noise_declaration` | `depends_on_prediction=False` 而 σ 确实动——今天无 fixture 声明 `False` |
| `two_observations_reverse_sorted_names`（已存在） | 观测节点**声明序 ≠ 排序**——12 行里两者恒等 |
| `plated_radiometer` / `plated_and_scalar_latents`（已存在） | plate × σ 依赖；块内**叶大小不等** |
| 三隐变量链 `tau → x → y` | 祖先弹出跑一遍 vs 跑到不动点；块大小 ≥ 3 |
| `overflowing_outside_latent` / `improper_outside_prior`（已存在） | §3.6：`compile()` 对**合法**模型不得抛错 |
| `affine_only_at_zero`（已存在） | 多 `at` 点确实跑了 |

### 7.1bis 关口一之前：两个守卫的修复（§1.4、§1.5）

每条修复由一条**具名**测试守住，且该测试**在修复前必须先跑一遍确认它红**——否则它守的是别的东西：

| 变异 | 必须变红的测试 |
|---|---|
| `affinity_errors` 改回全局 `_biggest` | `test_a_bright_component_does_not_mask_a_false_claim_on_a_faint_one` |
| `floor` 改回全局 | `test_the_roundoff_floor_is_per_leaf`（两侧：真曲率 / 真舍入） |
| 逐元素改回逐叶 | 单数组内亮/暗通道的那条 |
| `DEPENDENCE_PROBES` 改回锁步 | `test_sigma_depending_on_a_contrast_of_two_members_is_detected` |

新 fixture：`bright_and_faint_observations`（两观测节点）、`bright_and_faint_channels`（**一个数组内**，更接近真实场景）、`contrast_sigma_pair`（σ = f(a−b)）。

**外加一张全部现有 fixture 在新旧两版守卫下的判决对比表**（§1.4 的实施前必测），任何判决翻转都要具名解释。

四条子判据在这里的落法：亮/暗幅度比要**双侧**扫（默认比值 1 在扫描的哪一端？）；报告分离成立的**区间**而非一个点；叶子个数 {1,2}、叶内元素数 {1,多}、成员数 {1,2} 每维至少两值；这几条都是廉价确定性测试，**不应有任何一条需要 `slow`**。

### 7.2 关口二：统计

**(a) `gibbs_fn` 的条件正确性**——对 `graph_oracle`，**但必须写明它证明什么、不证明什么**。

> P3a 记录量过：R1 与 R2 共享 `_env_before`/`isolate`/`observation_parts`，**六个针对共享层的变异一个都没抓住**。§7.2(a) 继承同一盲区并**新增两个共享参数**：
> * **`at`**——变异「`gibbs_fn` 忽略 `hmc_sites`，从先验均值建 `at`」在 `hmc_sites` 恰等于先验均值时不可见。而 §3.6 正好叫实现者在编译期这么做——**混淆是预装的**。
> * **`sigma_at`**——`graph_oracle` 默认块的**零**，而 `_env_before` 用**先验均值**。变异「σ̂ 冻错点」被测试作者选的 `sigma_at` 吸收。
>
> 拆成三条：
> 1. `test_gibbs_fn_uses_the_hmc_sites_it_was_handed`——两个**都不等于任何先验均值**的 `hmc_sites`，断言抽取均值按预言机预测的量移动。
> 2. `test_gibbs_fn_freezes_sigma_at_the_declared_point`——断言协方差匹配 `sigma_at=<声明点>` **且明确不等于** `sigma_at=zero`。两侧具名。
> 3. `test_gibbs_fn_conditional_agrees_with_log_joint`——用 `log_joint`（桥那侧）给抽取打分。**唯一跨越 exact/bridge 接缝的检验**，而 HMCGibbs 正骑在那道缝上。

**(b) MH 不变性**——**不要写在 `radiometer` 上**。实测「反向密度不在 `x'` 处重建」这个**微妙**变异（接受率 0.955 vs 正确的 0.950——正是 §5.3 说的「看起来正常」）需 **~12,000 抽取**到 2σ、~75,000 到 5σ，每步 3 CG ⟹ 必然 `slow` ⟹ **子判据 4 触发**（它是这个 bug 的唯一守卫）。

新 fixture `steep_radiometer(n=6, kappa=0.5, floor=1e-2, prior_std=10)` 实测：偏 −0.2415 后验 sd、sd 比 0.738、接受率 0.64、**2σ 只要 66 个抽取**，≈430 到 5σ。**不标 slow。** 另断言正确链的接受率 ≈0.64 作正控制。

**(c) SNIS 的修正**——实测 `radiometer` 默认参数下裸 GCR 只偏 2.1σ（N=500），`n=40` 时只剩 0.2σ——**「否则这个修正是空的」已经在一个参数之外成为空话**。用同一个 `steep_radiometer`（16.8σ）。

**预言机从长跑 NUTS 换成求积**：`radiometer` 标量、`radiometer_group` 二维，都能精确积到 ~1e-10。P3a 记录说「Exact vs NUTS 是自洽检查」。

**矩对 σ̂ 的冻结点是瞎的**——不同 σ̂ 的 SNIS 仍是合法重要性采样，矩照样收敛，只有 ESS 掉。另加 `test_a_non_converged_gls_sigma_shows_up_as_ess_not_as_bias`：Kish ESS/N 掉一个钉死的倍数，而加权均值仍对。

**(d) CG 收敛对权重的影响**——**三点，不是两点**。初稿的 `{1e-6, 1e-12}` 里默认值 `1e-6` 在**松的那一端**，而 §5.5 的失效模式在**更松**的一侧——整条扫描跑在什么都不会发生的方向（P3a Task 5 原样重演）。且两点**互相一致**的测试无法发现「`tol` 被忽略」——两臂逐位相同，绿。

改为 `tol ∈ {1e-1, 1e-6, 1e-12}`，断言后两者一致**且 `1e-1` 可测地不一致**，偏离量钉死。第三点证明旋钮接上了。**x64 下跑**。

**(e) 复合 vs 纯 NUTS**——拆两条：条件矩走 (a)（`hmc_sites` 固定、确定性）；边缘矩只在**先实测过外参数确实混合**的 fixture 上比较，`ESS` 与容差一起进 docstring。

**(f) ESS/秒**——**不做数值断言**。实测 5 种子块内 4.72–19.39x、x64 下 24–43x、`tol=1e-4` 时 1.58x。**`≥5.98x` 换个种子就红。** 更糟：它在**错误方向上单调**——返回便宜错误抽取的 `gibbs_fn` 更**快**、比值更**高**。改为不断言的报告，`tol`/`maxiter`/dtype/seed 全钉死，docstring 写实测区间与两个不改善方向。

**(g) trace 安全**——`jax.jit(gibbs_fn)(...)` 不得抛 `TracerBoolConversionError`，且 `probe_gaussian=True` 下**必须**抛。两侧、毫秒级、无 MCMC。否则 §1.2 的整个理由从未被执行——(a) 若跑在具体值上，`probe_gaussian` 被接受并忽略也是绿的。

### 7.3 边界验证

| 阈值 | 两侧 |
|---|---|
| `condition_bound` 导出的 `tol` | 守卫开 / 关 |
| Kish ESS/N 的塌缩阈值（§5.2 的回退） | SNIS / 回退 |

> **实测更正：初稿列的四个阈值有两个是范畴错误。**
> * **k̂ = 0.7** 是可靠性**标签**，两侧同一个计算，没有「两方法在阈值处一致」可言。
> * **「MH 接受率 | 反向密度在场/缺席」**——实测两侧 0.950 vs 0.955，**它们一致，而这正是 bug**。是正确性检验，不是分派阈值。
> * `check_prediction_dependence` 的 `rtol`——它自己的 docstring 已论证边界验证**不适用**。初稿与它矛盾。

**极端参数**：块大小 1 与 1e4（受 §5.2 的维度上限约束）、观测节点数 1 与 5、κ 从 1 到 1e10、σ 跨六量级、完全共线的父节点、零观测的隐变量。

### 7.4 每个任务的标准动作

变异测试；AST 规格比对；四条子判据；「变异不变红时」的三步诊断。

---

## 八、文件布局

```
src/bayesmith/exact/correct.py     log q / SNIS / Kish ESS / khat / MH 接受步   新写 ~210
src/bayesmith/exact/gibbs.py       gibbs_fn 工厂 + HMCGibbs 装配                新写 ~180
src/bayesmith/exact/block.py       +probe_gaussian（两个调用点，三个调用方）     改   ~+40
src/bayesmith/exact/linearity.py   B1：affinity_errors 逐叶/逐元素 + 1/sigma 权  改   ~+35
src/bayesmith/exact/gls.py         B2：逐成员独立探测方向（含符号混合）           改   ~+25
src/bayesmith/dispatch/classify.py 合格性 + 分区推导                             新写 ~280
src/bayesmith/dispatch/plan.py     Block / InferencePlan / sample / estimate    新写 ~270
```

**依赖方向**：`exact/correct` → `exact/{block,solve}` + `graph/evaluate`；`exact/gibbs` → `exact/*` + `bridge`；`dispatch/classify` → `exact/{gaussian,linearity,gls}` + `graph`；`dispatch/plan` → `dispatch/classify` + `exact/*` + `bridge`。无环。

`bayesmith.compile` 走 `_LAZY_ATTRS`，`dispatch` 进 `_LAZY_SUBMODULES`。`test_importing_bayesmith_still_does_not_import_numpyro` 与 `errors.py` 的仅-stdlib 契约保持绿。

---

## 九、明确不在本 spec 范围内

- 混合图的 MAP（P5）；`fn` 非线性块的修正（P5）；`identifiability` 的稠密 SVD（P5）
- RTS / Kalman（P4）；离散枚举（P4）
- `MultivariateNormal` 稠密协方差；嵌套 plate
- 失败块的极大合格子集搜索（§3.5）
- `chain_method='vectorized'` 的多链（§二：0.21.0 下坏掉；用 `sequential`/`parallel`）
- `ruff format --check` 在 9 个 P1/P2 文件上的漂移

---

## 附录：复现

§1.3 的三组基准、§5.3 的 1-D MH 实验、§5.2 的 SNIS 维度扫描、§4.2 的 κ 扫描、§3.2 的孤儿密度反例、§7.2 的功效计算，脚本均在本次会话的 scratchpad。

**每一个进入测试套件的数字都必须把 `seed`、`N`、dtype、`tol`、`maxiter` 写进参数化**，并把**实测区间**而非单点写进 docstring——§1.3 与 §二 的教训。
