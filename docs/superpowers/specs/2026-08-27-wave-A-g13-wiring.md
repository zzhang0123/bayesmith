# 执行页 Wave A · G13 接线 — 联合先验作为声明的 factor site 过缝

> **文档状态：`record`** · 已落地批次/审计/测量的历史记录，写作当天为真，非当前权威。索引见 docs/README.md。

> 计划:§四 G13 / 铁律 3、5。前一批次:`2026-08-27-wave-A-s6-widened.md`。
> **日期**:2026-08-27 · **本页状态:完成。这是 `priors` 与 `numpyro_bridge` 的先决,
> 不是可选的收尾——同 G1 接线之于 `sensitivity`。**

## 〇、为什么它是先决

`to_graph` 至今**拒绝**任何声明了 `joint_prior` 的空间(`_refuse_a_joint_prior`),
措辞里逐字写着「the G13 gap」。而 `priors.JeffreysPrior` 的算术要委托出去,就必须
有一张**带着这条声明**的图——远端的 `JeffreysPrior.information(graph, values)`
第一件事就是 `_check_against(graph, values)`。

G13 的实现已在 bayesmith **0.4.0**(已在索引上,`git log v0.4.0..HEAD -- src/` 为空,
且 `v0.4.0` 的 `__init__.py` 里 `joint_prior` 与 `JeffreysPrior` 都在),所以铁律 5
满足:本批次不依赖任何未发布的远端表面,地板也不需要动(已是 `>=0.4`)。

## 一、翻译的三个方向,以及为什么它们各有一条守卫

* **被覆盖的 latent 必须声明成 `ImproperUniform`。** 远端 `_check_against` **按类型**
  拒绝一个既被 `over=` 覆盖、又带自己密度的 latent(「两个先验在一个量上」)。
  rheplicant 在 `ParameterSpace.__check_init__` 用**同样的措辞**拒同一件事,所以两侧
  一致;缝要做的是把 `prior=None` 翻成那个具体的类型。这与 `to_numpyro_model` 十几个
  月来给同一批 latent 的站点是**同一个拼写**,所以图和手写桥声明的是同一件事。
* **未被覆盖的 latent 必须保留自己的先验。** 一个「所有 latent 都声明成 flat」的接线
  会通过前一条的每一项检查,并**静默删掉**块外每一个先验。
* **`rank_rtol` 必须过缝。** 两侧 `None` 都表示同一个默认值,所以只有一个**非默认**值
  说得出话:显式的 `rank_rtol` 是调用方关于「零特征值从哪里开始」的决定,丢掉它会留下
  一个在没人要求的容差上取的秩判决——有限、可信、另一个先验,而图的形状与 dtype
  一个字都不会说。

## 二、本批次的核心守卫:**丢掉声明是看不见的**

一张没有这条声明的图**节点对、形状对、dtype 对**;它验证通过、可以抽样,而且每一条
收敛诊断都干净。它只是**另一个后验**——只有似然的那个。这正是被退役的那条拒绝原本
大声说出来的话。

所以守卫不是「字段被设上了」,而是**「势能动了,而且动了这么多」**:

```
log_joint(有声明) - log_joint(无声明) = +15.8016985265
0.5 log det I(远端自己算)              = +15.8016985265
```

那个数不是随便一个数:它是**幂律 + radiometer 声明下 Jeffreys 先验恰好是平先验**的
那个常数,两个包各自在自己的 module docstring 里引用它,并且**各自用闭式导出**——
`sigma = f|mu|` 给出 `N^-1 = 1/(mu^2 f^2)`,而 `J_{k,i} = mu_k g_i(nu_k)`,于是每个 `mu`
都抵消,`I_ij = (1 + 2 f^2)/f^2 · sum_k g_i g_j`,一个常数矩阵。

## 三、数值验收住 `tests/seam/`,而这是被套件教的

第一版把「势能移动了多少」写进了 `tests/inference/test_graph_bridge.py`。**红了**,
理由指名:`JeffreysPrior.information` 拒绝 ambient float32(D9 补的那条),而那个文件
**故意是 float32 的**——它自己的 header 就写着「adapter 的数值验收住 `tests/seam/`,
本文件是 float32、讲拒绝与形状,所以它能在这里」。

于是新增 `tests/seam/test_g13_joint_prior.py`(x64 门),**12 例**:

| 测试 | 钉什么 |
|---|---|
| `test_the_closed_form_is_the_constant_both_packages_quote` | **先钉 oracle 本身**。它若漂了,下面每一条都在量漂移而不是量接缝——而这个常数被引用在两处 module docstring 加一份契约页里,过期会同时静默三处 |
| `test_the_potential_moves_by_the_closed_form_at_every_grid_point`(9 例) | 九个格点、一个数:两张除声明外完全相同的图,`log_joint` 之差就是那个 factor,别无他物 |
| `test_the_prior_really_is_flat_across_the_grid` | **兄弟断言**:九例都对着**同一个**数,所以一个「完全无视 values、返回一个缓存 factor」的接缝会全数通过。这条断言**似然项在同样九个点上不是常数**(跨度 > 1),于是「差是平的」说的是先验平,不是什么都没变 |
| `test_a_block_named_in_the_other_order_is_the_same_prior` | `over=` 反序是一次对称置换,行列式不动,势能必须一字不动——因为**行序两边并不相同**(D24),这条钉的是那条分歧**停在行上、够不到密度** |

`tests/inference/test_graph_bridge.py` 留下 **5 条结构守卫**(声明过缝、`rank_rtol`
过缝、被覆盖的声明成 flat、未覆盖的保留自己的、`priors=` 与 `over=` 撞名被拒),
并在原地留一句注释说明数值那条为什么不在这里。

## 四、退役一条,理由与 G1 那次同源

`test_a_joint_prior_because_it_needs_a_factor_site` **退役**——分诊第三列,需要理由:
它钉的是一个 GAP,缺口补上了,而**钉住「不存在」的测试不比它的主题活得久**。

**拒绝普查净变化为零,而内容变了两处**:退役这一条、新增
`test_a_supplied_prior_for_a_covered_latent_is_refused`(`priors=` 与 `over=` 同时点到
一个 latent——声明期那条检查够不到它,因为那本字典要到 `to_graph` 被调用才存在)。
该文件 13 → **13**,总数 240 → **240**,`BY_CLASS` 一字未动。**计数守卫因此保持绿。**

> **而这暴露了附录 B 已经过期,过期了一整批。** G1 接线退役 `sigma = inf` 那一条时
> 把 14 → 13 写进了记录页与 `CENSUS`,**没有改附录 B**;本批次净变化为零,普查测试
> 更不会说话。**计数抓不到这种,只有内容能。** 附录 B 的这一段自此**由
> `test_refusal_census._sites()` 重生成**,不再手抄——本批次已重生成并改正。

## 五、两条语义差,当场量出来,登记为 D24 与 D25

**这两条都是计划没有预见的裁决点**,按「委托不是空白支票」的规则自选保守侧并入簿。

### D24 — 行序

rheplicant 的 `information` 按 `sorted(over)` 返回,并在自己的 docstring 里把这条写成
一个 **wart**(「读第 0 行的调用方在 tour 自己的块上错 7.4e+1」);bayesmith **故意
没有移植**(「the wart does not port」)。行列式不受影响,所以**先验相同**;变的是
返回矩阵的行序,而它是被测试读到的。**取:门面置换回 `sorted(over)`**,理由是
tour 自己的块 sorted 与原序**不同**,所以这条差**有 fixture 分得开**——是能被守卫
钉住的那一侧。落地在 `priors` 批次。

### D25 — 一个 float32 会话里的 Jeffreys 先验

远端 0.4.0 起**指名拒绝** ambient float32(D9 的第二个洞:310 nat,在 NUTS 会取指数的
那一项上)。rheplicant 今天没有这条守卫。

**范围是量出来的,不是估的**:把该守卫临时装进 rheplicant 的 `information`,跑
`tests/inference` + `tests/config`,红的**恰好三条**,全在
`tests/config/test_config_exits_npe.py::TestThePriorGate`。而
`tests/inference/test_jeffreys_prior.py` 的 **49 条一条不红**——它有一条 module 级
autouse 的 x64 fixture,**和 S6 那个文件同一形状**。第二次了。

**取:拒绝,并提到 `to_numpyro_model` 的构造期。** 另外两条路**不存在**而不是更差:
`jax_enable_x64` 是追踪期的全局开关,而 NUTS 在门面返回之后才追踪 `model`,所以
identifiability/sensitivity 用的「门面内部开 x64」在这里用不了——float32 追踪出来的
模型里放不进 float64 的因子。给远端开一个 `allow_single_precision=` 的口子则要重开 D9
并发一版,还把「静默产出错答案」重新变成可达的。

## 六、变异集:5 条,而**第一轮的两条幸存都是洞,不是归因**

| # | 变异 | 仓 | 第一轮 | 修好后 |
|---|---|---|---|---|
| N1 | 声明根本不进图 | rheplicant | KILLED(11 红) | KILLED(13 红) |
| N2 | 所有 latent 一律声明成 flat(`joint.covers(name)` → `joint is not None`) | rheplicant | **SURVIVED** | KILLED(1 红) |
| N3 | 翻译时丢掉 `rank_rtol` | rheplicant | KILLED(1 红) | KILLED(1 红) |
| N4 | 块**反序**过缝 | rheplicant | **SURVIVED** | KILLED(2 红) |
| N5 | 远端 `graph/trace.py` 不再记录声明 | **bayesmith** | KILLED(11 红) | KILLED(13 红) |

基线前后各一次绿。**N5 是一条真跨仓击杀。**

### N2:守卫在正确的地方,时机是错的

`test_an_uncovered_latent_keeps_its_own_prior` 用的是一个**根本没有 joint prior**
的空间。在那里 `joint is None`,于是变异前后**两种读法都说「未覆盖」**——测试从未
走到它要讲的那个分支。这正是交接页 §七.3 那条形状(rheplicant 的 `auto_blocks`
在求解期问了一个分区期的问题),换了一个身子出现。

改法:条件必须由一个**覆盖两个 latent 中的一个**的空间制造,两个节点都读。

### N4:`over` 是一个**一元组**,而一元组是自己的反序

反序在**密度上完全不可见**——对称置换不动行列式,而 `tests/seam/` 的
`test_a_block_named_in_the_other_order_is_the_same_prior` 是**故意**断言两者相等的。
所以只有**声明本身**看得见它,而当时每一条结构测试用的都是 `over=("gains",)`。

改法:两 latent 的块,**两种顺序各跑一次**,各自对着自己声明的那一个断言。
这条轴正是 D24 的主题,所以「接缝悄悄自己选了一个顺序」恰恰是不能没人发现的。

### 第一轮与第二轮之间踩的坑,值得单写

修好之后重跑,**报的还是同样两条幸存**。原因不是修得不对:变异脚本的 `restore()`
第一件事就是 `git checkout -- src/ tests/`,而修补**还没提交**——它在任何一个变异被
应用之前就被回退了。

**五行协议第 (0) 条因此有第二半**:「先提交再变异」不是「批次开始时提交一次」,而是
**每一次跑变异集之前,HEAD 都必须已经是你想要回的东西**。而且输出里没有任何东西说
得出这件事——一个被回退的修补和一个不起作用的修补长得一模一样。

已写进两仓工作笔记(rheplicant 的 `CLAUDE.md`/`AGENTS.md` 成对改,`cmp` 逐字节一致;
bayesmith 的 `CLAUDE.md`)。附带一条:变异脚本恢复的路径不要宽于变异点本身——这一份
为了撤销只在 `src/` 里的变异而恢复了整个 `tests/`。

## 七、铁律 4 四件套

| | 项 | 结果 |
|---|---|---|
| (i) | 该批测试全绿 | rheplicant **10070 passed / 534 skipped** exit 0(329.2 s)加 **21 passed** exit 0(e2e)加 **31 passed / 1 xfailed** exit 0(`JAX_ENABLE_X64=1 pytest tests/seam`,58.9 s);bayesmith **1280 passed / 0 skipped** exit 0(215.4 s)。**修补后重跑:rheplicant 10073 passed / 534 skipped exit 0(349.3 s)** |
| (ii) | 接缝变异红 | **5 条全杀**(第一轮 3/5,两条幸存均为真洞,已修),§六 |
| (iii) | 旧实现删除、计数守卫刷新 | `_refuse_a_joint_prior` 删除;退役测试 1 条;拒绝普查 13 → **13**(内容变两处,附录 B 重生成并改正一批的过期);README 计数 10608 → 10624 → **10627**(修补加了 3 条) |
| (iv) | 文档实测数字重测 | 上述;计划 §四 G13 行标记落地;登记簿标题 D7–D19 → **D7–D25**;附录 A 补 N1–N5;两仓工作笔记补第 (0) 条的第二半 |

## 八、留给下一位

1. **`priors` 现在可以切了**,先决已清。它要**同批**处置 D24(置换回 sorted)与
   D25(构造期拒绝 + 那三条 config 测试按第二列改写)。
2. **`numpyro_bridge` 与 `priors` 仍是一批**:`log_density` 在 rheplicant 里只有
   `to_numpyro_model` 一个消费者,而门面签名 `log_density(forward, values, noise_std,
   flags)` **拿不到图**——`forward` 是一个闭包。所以要么门面自己造一张图(D22 的形状,
   合法性必须**重新测量**而不是继承),要么由 `to_numpyro_model` 造一次再传下去。
   这是那一批的第一个设计问题。
3. **那一批还欠一条真链验收**(交接页 §三.4):G13 至今只验了势能,没跑过 `nuts()`。
4. **`to_numpyro_model` 的站点名是保持面**:`"prediction"`、`"obs"`(可由 `obs_name=`
   改),而 `to_graph` 的内部节点叫 `__mu__` / `__data__`。直接委托给
   `bayesmith.to_numpyro` 会改站点名,那是一条**要先量再决定**的事。
5. **D23** 仍是已登记、未裁决、无守卫。
