# 执行页 Wave A · G1 接线 — `FlaggedNoise` 作为声明的掩码过缝

> 计划:§四 G1 / §三 构图表 `FlaggedNoise` 行 / 铁律 3、5。
> 前一批次:`2026-08-27-wave-A-identifiability.md`。
> **日期**:2026-08-27 · **本页状态:完成。这是 `sensitivity` 的先决,不是可选的
> 收尾。**

## 〇、为什么它是先决而不是顺手

开工切 `sensitivity` 时读它的签名,发现与 `identifiability` 有决定性的不同:

```python
prior_sensitivity(space, pipeline, state_template, observed, noise_std, flags=None, *, names=...)
```

它**自带**数据、噪声与 flags——所以 D22 的「合成三样并证明它们够不到答案」在这里
**根本不适用**(交接页上一版把这条写成了「要重新测量不变性」,那也不对:不是要
重测,是**不需要合成**)。

但 `noise_std` 可以是 `FlaggedNoise`,而 `to_graph` **拒绝**它。
`config/postflight/fitting.py` 的 C19 路径把文档声明的噪声原样传进去,而
`config/sections/noise.py` 会在有 flags 时构造 `FlaggedNoise`。

**所以先切 `sensitivity` 会把一个今天能跑的 config 检查变成一条拒绝**——铁律 3
不允许这样静默地改语义。G1 的接线因此是先决。

而它现在**可以**做:G1 在 bayesmith 0.4.0 里,发布已上索引(本会话早些时候)。

## 一、翻译的两个方向,都对每一种形状检查隐形

* **极性**:`FlaggedNoise.flags` 是「被 flag」为真;图的 `mask` 是「被取到」为真。
  两者互为**否定**,都是数据形状的布尔数组——交换它们**不被任何 shape/dtype 检查
  发现**。守卫因此钉**被 flag 的那个 channel**,而不是钉一个计数:计数只在
  flagged 比例不等于一半时才抓得住。
* **被 flag 处声明的 σ**:必须**有限**。远端 `check_gaussian` 指名拒绝非有限 scale,
  理由是「sigma 表达式溢出了」与「这一路被 flag 了」需要不同的修法——后者现在由
  `mask` 说,所以 scale 只说前者。放进去的值是**基模型自己的 σ**,不是占位的 1.0:
  它够不到任何答案(远端已量过被掩样本 σ 改 1e9 倍不动答案),但用仪器自己的数字
  意味着这张图读起来是模型而不是编码。

## 二、退役一条,理由是它钉的是一个**缺口**

`test_flagged_noise_because_masking_has_not_landed` **退役**——分诊表第三列,
需要理由:它钉的是一个 GAP,缺口补上了,而**钉住「不存在」的测试不比它的主题
活得久**。它保护的东西由 `TestFlaggedNoiseCrossesAsADeclaredMask` 三条**正面**钉住。

拒绝普查随之从 14 → 13(该文件)、241 → 240(总数),**数字取自守卫自己报的**。

## 三、地板升到 `>=0.4`,这次是真需要

模块 1 的门面只用 0.3.0 就有的名字,当时**没有**升地板并写明了理由。本批次是
rheplicant 里第一处真正需要 0.4 表面(`observed_mask`)的地方,所以地板在同一个
提交里升。地板声明的是**真实需要**——这条规则两批次里各用了一次,方向相反。

## 四、变异集(4 条,全部击杀)

| # | 变异 | 在哪个仓 | 指名红 | 判决 |
|---|---|---|---|---|
| X1 | 掩码保留 flags 的极性(不取反) | e-RHINO | `test_the_mask_is_the_negation_of_the_flags` | KILLED |
| X2 | 声明的 scale 保留 `inf` | e-RHINO | `test_the_declared_scale_is_finite_where_the_flags_are` | KILLED |
| X3 | 每张图都给一个满掩码 | e-RHINO | `test_an_unflagged_noise_model_declares_no_mask_at_all`(共 6 红) | KILLED |
| X4 | **远端**把节点的 mask 丢掉(`graph/trace.py`) | **bayesmith** | `test_the_mask_is_the_negation_of_the_flags` | KILLED |

X4 是又一条真跨仓击杀:改 bayesmith 的 `observe(mask=)`,e-RHINO 红。

## 五、本批次最贵的一课:一个守卫连续两次被读成噪声,而它一直是对的

`tests/test_docs_claims.py` 的两条路径检查在本批次中途开始红,而且**在一次全量
运行里绿、单独跑却红**——读起来像 flaky。它不是。

**真正的原因是我上一批的错误**:`identifiability` 那次提交用了 `git add -A`,
把根目录**九份未跟踪评审草稿**一并入了库——而计划附录 C 明确写着它们**不入库**
(「另行裁决去留,不随本程序入库」)。那些草稿里有早已过期的引用(一个 bayesmith
的文件、两个 GUI 助手、一个测试类)。**草稿未跟踪时守卫看不见它们;一旦入库,
守卫立刻看见并报红。**

于是:那次全量运行(在错误提交**之前**)是绿的,之后每一次都是红的。这不是
flakiness,是**守卫如实报告仓库内容变了**。

两次归因都归错了地方:先归给未提交的改动(`git stash` 没有 stash 未跟踪文件,
所以那次对照没有意义),再归给一个「干净的 HEAD worktree」(而那时草稿**已经**
被跟踪在 HEAD 上,所以 worktree 里也有)。

处置三件:

1. `git rm --cached` 那九份,提交 `f8a73eb`。**内容仍在 `860703d` 的历史里**;
   把它从历史里去掉需要**强推**,而强推是本程序授权明写要停下来问的三件事之一,
   所以留给 owner 决定。
2. **`.gitignore` 收下这九个文件名。** 靠「它们是未跟踪的」不成立——`git add -A`
   一天之内把它们扫进去**两次**。忽略让附录 C 的排除**机械化**,而不是依赖谁记得;
   它**不**决定它们的去留,文件仍在盘上。
3. 写在这里,因为「守卫红了但看起来无关」这件事本身是要付学费的形状。

> 教训一句话:**一个守卫在全量运行里绿、单独跑却红,不要先怀疑守卫;先问仓库的
> 内容在这两次之间变了什么。**

## 六、铁律 4 四件套

| | 项 | 结果 |
|---|---|---|
| (i) | 该批测试全绿 | e-RHINO **10062 passed / 522 skipped** exit 0(354.7 s);`test_docs_claims`、`test_refusal_census`、`test_readme_counts` 全绿 |
| (ii) | 接缝变异红 | **4 条全杀**,含一条真跨仓(X4) |
| (iii) | 旧实现删除、计数守卫刷新 | `_refuse_flagged_noise` 删除;退役测试 1 条;拒绝普查 14→13、241→240;README 10603 → **10605** |
| (iv) | 文档实测数字重测 | 上述全部;`pyproject.toml` 与两份工作笔记的地板段(0.3 → 0.4)成对改 |

**分诊表(本批次)**:原样重放 37 / 改写对适配器 0 / **带理由退役 1**(§二)。

## 七、留给下一位

1. **`sensitivity` 现在可以切了**,先决已清。它自带 `observed`/`noise_std`/`flags`,
   所以**不要**照抄模块 1 的 `_graph_for_rank` 合成——把调用方给的三样如实传下去。
   `flags` 现在能过缝了(本批次)。
2. 切它才能删掉 `_flat_view`(它是最后一个消费者)。
3. **那九份草稿的内容仍在 `860703d` 的历史里。** 要不要重写历史把它去掉,是 owner
   的决定(强推)。现状:已从索引移除、已 gitignore、文件在盘上。
