# 执行页 **G15** — 带先验的非线性局部块

> 计划:§四 **G15**(切 `uncertainty` 时新开的缺口,带解除条件)。
> 前一批次:`2026-08-27-p2-g9-full.md`(P2 余项清空)。
> **日期**:2026-08-27 · 实现全在 bayesmith 一侧,rheplicant **一行未动**——
> 解除那条延期要等发布(见 §五)。

## 〇、缺口是一句话,而那句话是可以量的

一个**非线性**模型在某点的后验精度,需要 `local_block` 的**雅可比**与
`unchecked_operator` 的**先验**,而两个构造器都只有一半:

* `local_block` 在**调用者的点**取切线(非线性模型的正确雅可比),
  但 `prior_mean` / `prior_std` **是空字典**——它自己的模块 docstring 有一整段
  论证为什么。
* `unchecked_operator` 带先验,但在**域的零点**取切线——仿射映射处处一个切线,
  所以那对仿射模型是对的。

**量出来的**,在 `mu = a·x^b` 上:`unchecked_operator` 的设计矩阵是 `a·log x`
(即 `b=0` 处的切线),而 `b=2` 处的是 `a·x²·log x`。**除了 x=1 那一行,每一行都不同**。
两者都没错,只是**还没有第三个**。

## 一、做成关键字,而不是第二个函数,也不是改主意

`local_block(graph, names, values, *, priors=False)`。

**默认一字未变**,因为 `diagnose/local.py` 模块 docstring 里那段论证**每个字都还
成立**:建在它上面的诊断要么自己就拒绝非高斯先验(sensitivity),要么**根本不能
看见先验**(一个 Jeffreys 先验若由含先验曲率的矩阵造出来,就坐进了自己的定义里);
而空字典正是让 `fisher_information(include_prior=True)` **响亮失败**而不是悄悄折进
一个没人声明的曲率的东西。

关键字把「调用方要了」这件事写在**调用点**上。

**先验经 `_env_before` 读**——那是本包里**唯一**把一个 latent 的声明变成
`(shape, dtype, prior_mean, prior_std)` 的地方。这一点是双重故意的:没有第二份
拼写可以漂;而且它的 `check_gaussian` 一起来了,于是一个**没有二次型**的先验在
**这里**被指名拒绝,而不是给后验精度贡献一个静默的零。**`priors=False` 时没有这条
拒绝**,因为那时根本没有东西被读——两个方向都钉住了。

## 二、oracle 是手写的解析雅可比

`d(a x^b)/db = a x^b log x`,在测试文件里用 numpy 写出来,与图机器**不共享任何代码**。

| 断言 | 对照 |
|---|---|
| 设计矩阵 | `a x^2 log x`(b=2 处),rel 1e-4 |
| 后验精度 | `Jᵀ N⁻¹ J + 1/prior_std²`,rel 1e-4 |
| 带/不带先验之差 | **恰好** `1/prior_std²`,rel 1e-4(反空洞:否则「include_prior 生效了」没被测) |
| 另一个构造器 | 给出 `a log x`,**且与上面不接近**(除 x=1 那一行) |

## 三、变异集:4 条 **4 杀**

基线前后各一次绿。

| # | 变异 | 判决 | 红数 |
|---|---|---|---|
| Z1 | `priors=True` 被忽略,永不携带 | KILLED | 3 |
| Z2 | 先验**总是**携带,默认也带 | KILLED | **36** |
| Z3 | mean 与 std 读的 domain 槽位互换 | KILLED | 2 |
| Z4 | 切线移到域的零点(即变成 `unchecked_operator` 的那个) | KILLED | **43** |

**Z2 与 Z4 的红面很宽,而那本身是一条读数**:36 与 43 条,遍及 `test_identifiability`、
`test_graph_joint_prior` 等。默认路径被**大量**既有诊断依赖着——这正是为什么 G15
必须是第三个构造器而不是改掉第一个,而这句话现在有数字。

## 四、铁律 4 四件套(按 G 项的形态)

| | 项 | 结果 |
|---|---|---|
| (i) | 该批测试全绿 | bayesmith **1380 passed** exit 0(241.2 s);本批新增 **8** 条(1372 → 1380) |
| (ii) | 接缝变异红 | 4 条 **4 杀**(§三);基线前后各一次绿 |
| (iii) | 旧实现删除、计数守卫刷新 | **本批不能删** rheplicant 的 `_prior_precision`,见 §五 |
| (iv) | 文档实测数字重测 | CHANGELOG;G15 行回填;模块 docstring 那一段改写为「默认不带」并说明第三个构造器 |

## 五、解除条件**还没有被兑现**,而它被发布门挡着——这是一致的

`uncertainty._prior_precision` 的 docstring 逐字写着解除条件:G15 落地并**发布**
之后删掉该函数、调用改成 `include_prior=space is not None`,**只有那一行会变**。

**「并发布」是这句话里承重的两个字。** 铁律 5 说 rheplicant 的 main 不得依赖一个
不在索引上的远端表面,而 `local_block(..., priors=True)` 现在只在 `Unreleased` 里。
所以本批**只做 bayesmith 一侧**,rheplicant 一行未动。

> 这不是把工作推给下一位:延期本来就写着「落地**并发布**之后」,而本批把「落地」
> 那一半做完了。剩下的那一半是**一行**加一次数字重测,并且它现在与收尾发布排在
> 一起,而不是各等各的。

## 六、留给下一位

1. **发布之后**:删 `uncertainty._prior_precision`,调用改成
   `include_prior=space is not None`,重测被钉的数字。
   **同一批要重跑 e-RHINO 全套并重新量一件事**:`2026-08-27-p2-g9-full.md` §六 记着
   「G9 全量修掉的那个先验广播缺陷,门面今天够不到,因为它永远传
   `include_prior=False`」。**这条改动正是让它够得到的那条。**
2. **D12 的前置**(读档 fixture)不受发布门约束,是现在就能做完的解锁工作。
3. **D23** 仍是唯一一条已登记、未裁决、无守卫的语义差。
