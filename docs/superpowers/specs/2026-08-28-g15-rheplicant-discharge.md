# 执行页 — G15 的 rheplicant 一半:解除那条有条件的延期

> **文档状态：`record`** · 已落地批次/审计/测量的历史记录，写作当天为真，非当前权威。索引见 docs/README.md。

> 计划:§五 G15 / 铁律 1–5;新增 **D48**(裁决)。
> 前一批次:`2026-08-28-architecture-narrative.md`(bayesmith 侧 0.5.0 发布)。
> **日期**:2026-08-28 · **本页状态:已落地。** `uncertainty._prior_precision`
> 的算术已删,五条准入原地留守并改名 `_declared_gaussian_priors`;地板
> `bayesmith>=0.5`。

## 〇、这一批为什么值得单独写一页

交接页把它列成「**最便宜的一件**」,计划正文和被延期函数自己的 docstring
**都**说解除只改一行。**两处都错**,而错的方向是同一种:**从 bayesmith 那一头
看过去,`priors=True` 确实是一个关键字**;从 rheplicant 这一头看,有三个地方
各自独立地决定过「这里不带先验」,而**每一个决定在它自己的位置上都是对的**。

这就是**一条延期从站着的那一端看总是比实际小**的形状,值得留一页。

## 一、先量,再改:计划的一行是几行

探针 `docs/probes/probe_16_g15_discharge.py` §1。把 `include_prior` 翻成
`space is not None`、**其他一字不动**:

| 改了什么 | 结果 |
|---|---|
| 只翻 `include_prior` | `KeyError: 'a_vec'` |
| 再让 `local_block(priors=True)`,图仍是 flat | `with` 块空手而归 |
| 再让 `declared` 由 `space` 建 | **对** |

三处,不是一处:

1. `_bayesmith_fisher` 建块时 `local_block(graph, block_names, values)`
   ——**没有** `priors=`。先验曲率是从**块的** `prior_std` 读的,所以带着
   `include_prior=True` 去问一个无先验的块,第一个成员上就 `KeyError`。
2. `fisher_information` 的**两个分支都**硬写 `declared = None`,于是
   `graph_for_information` 给每个 latent 挂 improper flat。
   (`graph_for_information` 的 `priors=` 参数**早就在**,docstring 甚至写着
   它「正是让这张图既能算似然信息也能算后验精度的东西」——只是没有人传。)
3. `include_prior=space is not None` 本身。

**没有一个失败长得像「答案错了」**:一个 `KeyError`,一个什么都不返回。

## 二、算术确实是一份 —— 逐比特,且 oracle 不是被测代码

接对之后,与被删的拼写比:

```
the deleted spelling, in numpy   diag = [37.2562  53.4876  42.23347]
the delegation                   diag = [37.2562  53.4876  42.23347]
max |difference| = 0.000000e+00    relative = 0.000000e+00
```

**不是「在容差内」,是相同**。两边都归约到同样跨度上的 `diag(1/sigma^2)`。

> **这一节自己踩过一次要防的坑,记下来。** 探针最初拿
> `fisher_information(space=...)` 当参照——写的时候它还是本地拼写,量得没错。
> 委托一落地,**比较的两边成了同一次调用**,`0.0` 依旧,而它已经什么都不证明了。
> 现在探针自带一份 numpy 的独立拼写(手写扁平排布,不走 jax)。往 oracle 里
> 注入 1% 误差,差值变 `1.600000e-01` ——**它还能红**。

## 三、真正的发现:第四处,以及它为什么不是「多改一行」

原函数里有**五条拒绝**,三条被测试逐名钉住:

| 拒绝 | 钉在哪 | 远端有没有对应物 |
|---|---|---|
| 非高斯先验(Uniform) | `test_fisher_prior.py`,`match="z_scalar"` + `"Uniform"` | 有,但**到不了**(见下) |
| `LogNormal` 不许鸭子类型 | 同上,`match="LogNormal"` | 同上 |
| latent 无先验 | 同上,`match="z_scalar"` | 同上 |
| params 无名 | 同上,`StateValidationError`,`match="not named"` | **没有** |
| 声明了 `joint_prior` | `test_jeffreys_prior.py`,`match="inside its own definition"` | **没有** |

前三条交给远端会**全部消失**,而且不是变成另一个异常——**根本不到达**。

`graph_bridge.translate` 把 bayesmith 的拒绝分三族,`NotGaussian` 是**无责裁定**
那一族:**捕获、不外抛**,记在让出的 `Seam` 上。它自己的 docstring 讲得很清楚,
理由也对:一个为了**分支**而问「这里有没有精确路线」的调用者,不该被迫在一个
**问题**外面写 `except`。

`fisher_information(space=...)` 不是那个调用者。对它,一个 Uniform 先验是**错误**。

实测(探针 §3),三例都一样:

```
Uniform     seam.blameless = NotGaussian;  block ended early = True
LogNormal   seam.blameless = NotGaussian;  block ended early = True
prior=None  seam.blameless = NotGaussian;  block ended early = True
```

`with` 块提前结束,下一行在一个**从未赋值的名字**上读 `.values`。调用者被承诺
的是一个点名的 `ParameterSpaceError`,拿到的是一个谁也没点名的
`UnboundLocalError`。

**裁决的依据不是「保守起见」,是计划自己已经有的规则**:P1 总原则——

> 凡图缝会抹掉证据的拒绝,住在 `to_graph` 前的预验证。

首例是 `check_noise_std_axis`,形状完全一样。G15 的正文只是没有引用它。

于是:`_prior_precision` → `_declared_gaussian_priors`。**算术删掉,五条准入
原地不动**,返回 `{name: prior}` 交给 `graph_for_information` 的 `priors=`
——那本来就是它的单一入口。

## 四、顺带量出的第二处分歧,是本批更有意思的一个

第一次全量跑,**只有一条红**,而它不在上面任何一格里:
`test_an_expanded_normal_is_recognised`,`UnboundLocalError`。

两包对「什么算高斯先验」的判据差**恰好一个拼写**(探针 §4):

| 写法 | rheplicant | bayesmith |
|---|---|---|
| `Normal(zeros(2), full(2, .5))` | accept | accept |
| `Normal(0, .5).expand([2])` | accept | **refuse (NotGaussian)** |
| `Normal(zeros(2), .5).to_event(1)` | accept | accept |
| `LogNormal(0, 1)` | refuse | refuse |

`_gaussian_parameters` 拆 `Independent` **和** `ExpandedDistribution`,理由写在
它自己的 docstring 里:两者都只是给基分布**换个形状**。`check_gaussian` 只认
前者。**两条规则都站得住**——但经过一条会吞掉「不」的缝,这个分歧到用户手里
就是 `UnboundLocalError`。

修法不是去改任一条规则(改 bayesmith 要发版,铁律 5 挡着),而是让过缝的东西
是**规范形**:`Normal(loc, scale)` 广播到 latent 自己的形状,**一种写法覆盖
所有拼写**。`Latent` 里用户写的东西一字不动;这就是构图表里早写着的
「经 `priors=` 单入口合成 `Normal`」。

## 五、G15 点名要重测的那件:答案是「仍然够不到」,理由比原来的好

计划说:「G9 全量修掉的先验广播缺陷今天够不到门面(它永远传
`include_prior=False`),而这条改动**正是让它够得到的**。」

那个缺陷的入口形状是**标量 `prior_std` 配向量 latent**。实测:

```
ParameterSpaceError: Latent 'a_vec': prior has shape () but init has shape (2,).
The prior describes the latent, so the two must agree.
```

`Latent.__check_init__` 在**构造期**就按名字拒绝了。所以它**从 `ParameterSpace`
根本到不了** Fisher,现在也一样——但理由从「门面不传那个标志」换成了
「声明层不接受那个形状」,后者稳固得多。

`.expand([2])` 是同一个先验**补上了形状**,它到得了,而规范形正是为它准备的。

## 六、变异

五行协议,规则 (0) 两半都守:批次先提交,且**每一轮开跑时 HEAD 就是想要
回来的东西**。击杀 = pytest 退出码**恰好 1** 且登记的测试在红名单里。

| # | 变异 | 判决 | 由谁杀 |
|---|---|---|---|
| M1 | `local_block(..., priors=include_prior)` → `priors=False` | KILLED | 6+ 条,含 `test_the_prior_precision_lands_on_the_right_spans` |
| M2 | `include_prior=space is not None` → `False` | KILLED | 6+ 条,含 `test_a_space_relabels_the_matrix` |
| M3 | 预验证仍跑但**丢弃结果**(`declared = None`) | KILLED | 6+ 条 |
| M4 | 规范形去掉:`declared[name] = prior` | KILLED | 4 条,全在新模块与 `test_an_expanded_normal_is_recognised` |
| M5 | 只去掉广播,保留 `Normal(...)` | KILLED | 3 条 |
| M6 | `kind=found.kind` → 本地重新拼 `if space is not None` | KILLED | **仅** `test_the_kind_is_the_far_sides_own_tag_and_not_a_second_rule` |
| M7 | `joint_prior` 那条拒绝短路 | KILLED | **仅** `test_fisher_information_with_a_space_is_refused_while_one_is_declared` |
| S1 | **测试变异**:删掉顺序守卫的 `monkeypatch.setattr` 那一行 | **第一轮 SURVIVED**,修好后 KILLED | `test_a_refused_prior_never_reaches_graph_construction` |

**S1 是本批唯一的收获,也是收盘问题的现场答案。** 那条守卫原本写成两个测试:
一个装 monkeypatch 断言拒绝先到,一个断言 monkeypatch 确实会响。删掉第一个里的
装配行——**套件全绿**。因为没有 patch 时它只是
`test_a_prior_with_no_quadratic_form_is_refused_by_name` 的副本,而它被添加时
要立的那个论点(**准入跑在图之前**)已经不再被任何东西断言了。

patch 在的时候顺序确实钉住了;它做不到的是**被删掉时变红**。修法是把兄弟测试
并进来:先断言一个**被接纳的**先验**确实**到达被 patch 的构造器,于是 patch 成了
这个测试能通过的原因,删掉就红。重跑,KILLED。

> 这正是本仓反复付学费的那个形状——不是代码错,是**一条守卫失去了失败的能力**。
> 而它只有在变异测试**测试本身**时才看得见:M1–M7 全部一击即中,没有一个能说出
> S1 说的话。

## 七、四件套

- **分诊**:`test_fisher_prior.py` 21 条 + `test_jeffreys_prior.py` + `test_uncertainty.py`
  ——**一条未改**。全量:**10700 passed / 553 skipped**(README 计数与
  `test_refusal_census` 的 253 同批重测;两者都由守卫报数,不手抄)。这就是铁律 1 的验收:门面的公开名、签名、异常身份、钉住的文案
  全部不变,而实现换了一侧。
- **接缝变异**:见 §六。
- **旧实现已删**:`_prior_precision` 的 `jnp.diag(precision)` 与整个跨 span 的
  循环算术不再存在。
- **文档数字重测**:地板 `>=0.4 → >=0.5`(`pyproject.toml` + CLAUDE.md +
  AGENTS.md,后两者成对,`tests/test_docs_claims.py` 逐字节钉着)。

## 八、留给下一位

- 本批新增 `tests/inference/test_g15_prior_delegation.py`(7 条)。它钉的是
  **没有数字的那一半**:准入跑在图存在**之前**,以及那条使它必要的缝行为
  **仍然是它**。若 `translate` 哪天改成外抛 `NotGaussian`,
  `test_the_seam_still_files_not_gaussian_as_a_blameless_verdict` 会红,并把人
  指回 `_declared_gaussian_priors`。
- **两包的高斯判据仍然不同**,只是现在有一层规范形挡着。若 bayesmith 哪天收了
  `ExpandedDistribution`,规范形就成了纯粹的多余——那时**先量再删**,因为它同时
  在做广播。
