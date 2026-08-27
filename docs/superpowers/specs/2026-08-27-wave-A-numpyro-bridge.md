# 执行页 Wave A · 模块 4 — `numpyro_bridge` 的 `to_numpyro_model` 切换

> 计划:§五 Wave A / 铁律 1–5;登记簿 **D26**、**D27**、**D28**。
> 前一批次:`2026-08-27-numpyro-bridge-measurements.md`(先决测量 + 其清单第 1 项)。
> **日期**:2026-08-27 · **本页状态:清单第 2–5 项做完。但第 4 项的结论与清单
> 写的相反,理由是实测——见 §五。**

## 〇、分诊表

| 列 | 数 | 内容 |
|---|---|---|
| **原样重放** | **18** | `tests/inference/test_numpyro_bridge.py` 全数,一条未改 |
| **改写对适配器** | **1** | `test_jeffreys_prior.py` 的那条平先验链比较,重钉理由见 §六 |
| 带理由退役 | 0 | (cross-check **不退役**,见 §五) |

## 一、切了什么

`to_numpyro_model` 的 model body **整个走了**:逐 latent 的 sample 站点、带 mask 的
高斯似然、手写的 `numpyro.factor("joint_prior", ...)`,现在都由
`bayesmith.to_numpyro` 从本模块声明的图里发出来。

留下的是**图拼不出来的东西**:

| 留守物 | 归类 | 理由 |
|---|---|---|
| 五条构造期拒绝 | 第 4 类预验证 | 措辞被钉;图缝会抹掉证据 |
| 站点名 `"prediction"` / `obs_name` | 第 5 类保持面 | D26 |
| `observed=None` 的含义 | 第 5 类保持面 | 两个包**方向相反**,见 §六 |
| `init_to_declared` | 声明层 | 契约页 §5(a):图的 latent 有先验没有 `init`,**没有东西可移**——「带过去的是教训不是代码」 |
| `predict_from_samples` | 第 4 类 + 声明层 | 三条形状检查加一次 `eqx.filter_vmap`,**里面没有任何贝叶斯数值**;它 vmap 的是 rheplicant 自己的前向模型 |

## 二、D26 落地:节点名归调用方,**拒绝也要跟着走**

`to_graph` 新增 `prediction_name` / `observation_name`。`to_numpyro_model` 传
`"prediction"` 与 `obs_name`。

**关键的一半是那条碰撞拒绝**:`_refuse_internal_names` 从钉 `INTERNAL_NAMES` 改成钉
**本次调用选定的**那一组。钉默认名的版本会把两个方向都做反——在**需要**
`prediction` 这个名字的那一次调用上放一个叫 `prediction` 的 latent 过去,
又在不需要它的调用上拒一个叫 `__mu__` 的。所以守卫是**两条**,一条一个方向;
变异 R2 只被这两条杀死。

## 三、D27 落地:被抽样的 sigma 成为图的一个 latent

`to_graph` 新增 `scale_prior=(name, distribution)`;scale 作为观测节点的**第二个
parent** 抵达。0.4.0 就够,先决测量已经验过(`observe` 收多个 parent)。

**同来的是一个占位**:`to_numpyro_model` 仍要给 `to_graph` 一个 `NoiseModel`
(flags 住在里面),而那个 sigma 是假的。于是:

* `_refuse_a_scale_prior_the_seam_cannot_read` 把这个组合限制在**homoscedastic**
  (可 flagged)上。没有它,一个 radiometer 的、跟着预测走的 sigma 会被一个常数
  latent**静默替换**——形状一样、dtype 一样、链一样健康、模型不一样。
* 占位的够不到答案是**量出来的**:1.0 换成 1e9,`log_joint` **一个比特不动**。
  这条正是让占位合法而不只是「没人读」的东西。

## 四、D28:一条**新拒绝**,而它的代价实测为零

`to_graph` 的 `_prevalidate` 调 `check_noise_std_axis`,`to_numpyro_model` 从前不调。
实测:8×8 方格网格上 `noise_std=jnp.linspace(0.4, 0.6, 8)` **今天被接受**,沿最后
一个轴广播成 `(8, 8)`。委托之后被拒。

**按铁律 4(iv) 的「接受为修正」收下**,并把检查**提到 `to_numpyro_model` 自己名下**
(P1 §三:`to_graph` 的同一条检查会自称 `to_graph`,对一个调用了本函数的人是真话
而无用)。**代价实测为零**——全套没有一条测试、config 也没有一条路径向这个出口传过
一个有歧义的 1-D sigma。

## 五、清单第 4 项:**cross-check 不退役**,而这是与清单相反的结论

清单写着「`tests/crosscheck/test_bridge.py` 同批退役,`SWITCHED` 加
`numpyro_bridge.py`」。**逐条读过它的六条之后,结论是不退役。**

铁律 2 的判据是「参照是 rheplicant 的随文件退役」。六条里:

| # | 测试 | 参照谁 | 判决 |
|---|---|---|---|
| 1 | `test_the_init_strategy_lesson_transfers_and_the_remedy_is_reachable` | 只测 bayesmith 自己的 `nuts()`(r_hat 1609/ESS 1.0 对 1.006/138.6) | 单侧,仍有效 |
| 2 | `test_rheplicant_still_ships_the_strategy_this_side_reaches_by_keyword` | rheplicant 的 `init_to_declared`,**它留守** | 仍有效 |
| 3 | `test_a_transposed_sample_stack_is_refused_here_as_it_is_upstream` | 两侧的 `predict` 形状守卫,**两侧都在** | 仍有效 |
| 4 | `test_a_square_transposition_is_invisible_to_both_guards` | 同上,记的是一条**有意的差异** | 仍有效 |
| 5 | `test_a_joint_prior_over_a_latent_that_already_has_one_is_refused` | 只测 bayesmith | 单侧,仍有效 |
| 6 | `test_the_factor_site_adds_the_jeffreys_term_exactly_once` | 只测 bayesmith | 单侧,仍有效 |

**没有一条因为本批次变成「本包与本包比」。** 而模块**只切了一半**——
`init_to_declared` 与 `predict_from_samples` 按契约留守——所以
`numpyro_bridge.py` **不进 `SWITCHED`**,而这与那条守卫的另一个方向一致
(「不在 SWITCHED 里却丢了 cross-check → 红」)。

> **按「委托不是空白支票」处置**:照清单执行会删掉六条仍然能失败的测试,并让一条
> 簿记守卫声称一个只切了一半的模块已经切完。按事实选,并在这里点名。

**顺手改掉一句过期的**:第 6 条的 docstring 写着「`numpyro.factor` 是今天先验抵达
NUTS 的方式(把它声明**在图上**属于 `bridge/` 那一行)」。那个括号已经兑现(G13
接线 + 本批次),所以改写成:那条测试**故意**保留手写形式,因为它是被发出来的那个
factor 的**独立对照**——用发出来的 factor 去检查发出来的 factor 什么也没检查。

## 六、`observed=None`:两个包**方向相反**,而这是唯一逃过第一轮变异的一条

| | `model()` 无参 | `model(x)` |
|---|---|---|
| rheplicant(一直如此) | **先验预测**,不条件化 | 条件化在 `x`(一个数组)上 |
| bayesmith `to_numpyro` | **用每个节点自己声明的数据** | 一个 mapping,逐节点覆盖 |

门面传 `{}` 而不是 `None`。

**为什么这需要它自己的守卫**:图是拿一张**零占位**建的(`to_graph` 建图时要数据,
而 `to_numpyro_model` 拿不到——数据是运行时才来的)。所以一个把 `None` 直接透传的
翻译,会把**每一次先验预测调用条件在那张占位上**,交回一堆形状与 dtype 全对的零。

**实测:它通过了本文件里的每一条测试。** 形状断言两边一样、站点都在、别的什么都不看。
补两条(每个方向一条),读的是 `is_observed` 而不是值本身。

## 七、一条被重钉的测试,而它不是放宽容差

`priors` 批加的 `test_a_flat_prior_leaves_the_trajectory_bitwise_unchanged`(已更名为 `..._where_it_was`)断言
两条链**逐比特相同**;现在差 **1.877e-10**。

**原因是真的,而且值得写下来**:委托之前,那个 factor 的信息矩阵是从一张**合成图**
装配的,`sigma = f|mu|` 对上 `J = mu·g` 是**一个值除以它自己**,于是每个 `mu`
**逐比特**抵消,factor 在每一点上是同一个 float,轨迹因此完全相同。现在信息矩阵
来自**模型图**、装配顺序不同,抵消只精确到舍入。

实测:factor 仍然是那个平常数,精确到 **5.7e-9**(与印出来的 `15.80169853` 对到
八位小数,且在同文件既有的 1e-7 之内);漏出来的轨迹发散是 **1.4e-6 个后验 sd**。

所以判据改成**相对后验**而不是一个绝对 epsilon:一个在采样器能在意的意义上不再
常数的 factor 会把链移动**一个 sigma 的可观份额**,而兄弟测试正好展示了那是什么样子
——**4 到 5 个 sigma**。中间隔着六个数量级。

## 八、变异集:6 条,**第一轮 5/6**,幸存的那条是真洞

见附录 A 的表。**R5** 就是 §六 那条;修好后重跑 6/6。**R6 是跨仓击杀**
(改 bayesmith 的 `_masked`,e-RHINO 的 `test_flags_masked_likelihood` 红)。

两轮之间**先提交了修补再重跑**——第 (0) 条的第二半,今天已经踩过一次。

## 九、铁律 4 四件套

| | 项 | 结果 |
|---|---|---|
| (i) | 该批测试全绿 | e-RHINO **10093 passed / 534 skipped** exit 0(359.8 s)加 **31 passed / 1 xfailed** exit 0(x64 seam,50.4 s)加 **21 passed** exit 0(e2e,63.5 s);bayesmith **1269 passed** exit 0(222.8 s)。**加 R5 两条守卫后重测:e-RHINO 10095 passed / 534 skipped exit 0** |
| (ii) | 接缝变异红 | 6 条,**第一轮 5 杀**,幸存者是真洞,补守卫后 **6/6** |
| (iii) | 旧实现删除、计数守卫刷新 | model body 的站点/似然/factor 删除;拒绝普查 13 → **16**(该文件)、241 → **244**;`ParameterSpaceError` 175 → **178**;README 计数 10638 → **10649** |
| (iv) | 文档实测数字重测 | 上述;D26/D27 回填、**D28 新增**;附录 A 补 Q1–Q3 与 R1–R6;附录 B 的 `test_graph_bridge.py` 清单重生成、表头总数 241 → 244 |

## 十、留给下一位

1. **Wave A 只剩 `uncertainty`,而它最重**:`FlatMatrix` **永久**保持(config
   products 逐字段读)、`as_noise_model` 留守、`_named_spans` 随 Wave C/D 退役,
   **文件不整删**。
2. **它必须同批改写一条测试**:`test_a_vector_latent_permutes_by_ITS_SPAN_and_not_
   as_one_row` 拿 `uncertainty.fisher_information` 当**回归** oracle,而那条路随
   `uncertainty` 一起走。不改它,D24 的布局声明会失去它唯一的对照。
3. **它还欠 D9 的第二项功课**:`parameter_covariance` 的 `1/√eps` 天花板拒绝逐
   fixture 冒烟(float64 下是 **6.71e7** 而不是 float32 的 2.90e3,范围比 D9 原文
   小得多,但不为零)。
4. **`numpyro_bridge` 是本程序第一个「只切了一半」的模块**,而簿记如实反映了这件事
   (不进 `SWITCHED`,cross-check 保留)。Wave B/C/D 若再遇到同样形状,这是先例。
5. **D23** 仍是已登记、未裁决、无守卫。
