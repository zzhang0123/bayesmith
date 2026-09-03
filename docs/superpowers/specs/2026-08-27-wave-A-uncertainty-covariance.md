# 执行页 Wave A · 模块 5 第二步 — `parameter_covariance` 与 `propagate_covariance` 切换

> **文档状态：`record`** · 已落地批次/审计/测量的历史记录，写作当天为真，非当前权威。索引见 docs/README.md。

> 计划:§五 Wave A / 铁律 1–5;新增裁决 **D30**、**D31**;**D29 落地**。
> 前一批次:`2026-08-27-wave-A-uncertainty-fisher.md`(**它的 §八 是本批的开工清单**)。
> **日期**:2026-08-27 · **本页状态:Wave A 的模块全部切完。`uncertainty.py`
> 与 `numpyro_bridge.py` 一样**不进 `SWITCHED`**,理由在 §六,而这次是量出来的。**

## 〇、分诊表

| 列 | 数 | 内容 |
|---|---|---|
| **原样重放** | **83** | `test_uncertainty.py` 27 + `test_fisher_prior.py` 17 + `test_noise_std_axis.py` 39,一字未动;下游消费者(`test_inference_unpinned_refusals.py` 等)同样未动 |
| **改写对适配器** | **3** | D29 点名的三条,一文件一条,逐条见 §二 |
| 带理由退役 | **0** | cross-check **不退役**,见 §六 |
| (新增守卫) | **12** | 条件数天花板 7、合成图 2、精度拒绝 1、posterior 协方差仍可传播 1、加宽真的生效 1 |

> 三个文件的 `def test_` 计数,切换前后:`test_uncertainty.py` 27 → 38、
> `test_fisher_prior.py` 18 → 18、`test_noise_std_axis.py` 40 → 41。**数出来的,
> 不是估的**——本页第一版把「原样重放」写成 20,那是只看了一个文件的一次心算。

## 一、开工先复核了 D29 的范围,而不是继承它

上一批把天花板临时装进本包量过一次:65 次求逆里 3 次会被拒。**本批重量了一遍**
——不是因为怀疑,而是因为「委托之后仍然是这三条」和「委托改变了别的什么」在输出里
长得一模一样。探针装在 `parameter_covariance` 里,跑 `tests/inference` + `tests/config`:

```
total calls: 65
kinds:   {'fisher': 49, 'posterior_precision': 16}
dtypes:  {'float32': 65}
refused: {False: 62, True: 3}
```

三条被拒的 κ = **6.5e5 / 1.0e4 / 3.2e6**,与上一批逐个数字相同。切换之后实跑,
`tests/inference` 恰好红两条(第三条住 `tests/config`),**没有一条附带损伤**。

**顺带量到一件上一批没问的事**:65 次里**没有一次**的 `kind` 是 `covariance` 或
`matrix`。这个数字后来决定了两件事——一条新拒绝的代价(§四),以及一张表里一行的
去留(§七)。

## 二、三条改写,而三条的「病灶」都是它们自己的主题

天花板给出的出路只有一条:**在建图那一层加宽算术**。三条都照它改,而值得写下来的是
**每一条的条件数就是它自己要展示的那个量**——不是一个碍事的数值细节:

| 测试 | κ | 它要展示的 |
|---|---|---|
| `test_the_two_explicit_readings_give_visibly_different_answers` | 1.0e4 | `sigma` 跨 **100 倍**的两种读法 |
| `test_tightening_the_prior_tightens_the_error_bar` | 3.2e6 | 一个比数据紧 **四个量级**的先验 |
| `test_width_fisher_over_a_whole_multi_latent_space_is_allowed` | 6.5e5 | 两个 sigma 相隔 **五个量级** |

`F = J^T N^-1 J` 把这些跨度**平方**。所以「加宽算术」在这里不是绕过守卫,而是让
断言重新落在它本来声称的量上:第二条原先用 `rtol=1e-4` 去比一个已经花掉一半以上
有效位的逆。

**三处各带一条「块真的生效了」的兄弟断言**(`dtype == float64`)。没有它,加宽哪天
悄悄失效,三条会以**完全相同的方式**通过——走的正是天花板要拒绝的那条路,而
`rel=1e-3` 松到看不出来。

**config 那条的声明写在测试里而不是 builder 里**,照 `priors` 批 `joint_results()`
的先例:`runtime.jax_enable_x64` 在构建期就核对进程,而 fixture 普查无参驱动每一个
`*_document`。

**一次实测的边界,记下来因为它反直觉**:一个 float64 的 `FlatMatrix` 在 x64 块
**之外**读 `.sigma()` 会抛
`lax.mul requires arguments to have the same dtypes, got float32, float64`,
**不是**悄悄变窄。加宽是**算术所处上下文**的性质,不是数组自己的性质——这正是远端
那句「around the CONSTRUCTION」从另一头说的同一件事。所以 config 那条的两个 sigma
也在块内读完再出来。

## 三、`propagate_covariance`:先量再决定,答案是切

上一批写的是「先量再决定,大概率与 `predict_from_samples` 同一结论(留守)」。
**量完之后结论相反。**

### 量到的

探针装进函数,跑 `tests/inference` + `tests/config` + 两个 x64 会话,**23 次调用**:

| 形状 | 次数 |
|---|---|
| 具名 dict + `FlatMatrix` | 19 |
| 无名 pytree + 裸数组 | 3 |
| 无名 pytree + `FlatMatrix` | 1 |

**每一个 `FlatMatrix` 的 `kind` 都是 `"covariance"`**;一次 `posterior_covariance`
都没有,一次精度也没有。

### 为什么切,而 `push_forward` 不切

判据不是「有没有 jax 调用」,是**里面有没有一个有名字的统计方法**:

* `push_forward` 是 `jax.vmap(forward)`。一个 map,零贝叶斯数值。**留守**,与
  `predict_from_samples` 同一条理由。委托它反而要为一次纯映射合成噪声与数据。
* `propagate_covariance` 是 **delta 方法**:一次雅可比、一个二次型、一个线性化
  近似,而远端实现的是**同一个**。留着它就是把一个有名字的算法留下第二份实现。

### 逐比特对照(写代码之后,判决之前)

旧公式与委托后的实现,同模型同点,两种 dtype、两条路线:

```
float32:  unnamed  max rel diff = 0.000e+00  bitwise=True
          named 2D max rel diff = 0.000e+00  bitwise=True
float64:  unnamed  max rel diff = 0.000e+00  bitwise=True
          named 2D max rel diff = 0.000e+00  bitwise=True
```

## 四、合成的两样,以及为什么这次也**量**而不是**论证**

图要数据和噪声,delta 方法两样都读不到——`sqrt(diag(J Σ J^T))` 里既没有残差也没有
权重。这是**整条委托赖以成立的假设**,与 D22 对秩测试说的是同一句话,所以照 D22 的
规矩办:**造三遍**(σ=1 / σ=1e4 / 数据加 1e3),报告**逐比特**比较,外加一条兄弟断言
基线有限且非零——三个相同的报告也可能是三个零或三个 NaN。

变异 **U7**(让远端的 delta 方法真的按精度加权)把这两条打红,所以它们不是空的。

## 五、两条新拒绝,代价都实测为零

| 拒绝 | 今天的行为 | 实测代价 |
|---|---|---|
| `parameter_covariance` 收到一个协方差 | 再求一次逆,回来贴着 `kind='covariance'` 而它已经不是 | 65 次调用里 **0** 次 |
| `propagate_covariance` 收到一个**精度** | 返回一个有限、形状正确、**错了整整一个平方**的误差棒 | 23 次调用里 **0** 次 |

第二条**复用 `FlatMatrix.sigma` 已经在用的那张表**(`_PRECISION_KINDS`),同一个措辞、
同一条出路——一张表两个调用方,不是第二份实现。它住在**缝前**:图缝之后这句话会
穿上 `ParameterSpaceError` 到达,而本模块承诺的是 `StateValidationError`。

## 六、cross-check 不退役,而这次「能不能退」是**量**出来的

`tests/crosscheck/test_noise_logdet.py` 七条,逐条读(照 `numpyro_bridge` 的先例):

| # | 测试 | 参照谁 | 判决 |
|---|---|---|---|
| 1–4 | 对数行列式估计量四条 | `rheplicant.inference.noise.NoiseModelLikelihood`,**未切**(Wave B) | 仍是跨包 |
| 5 | `test_constant_sigma_gives_the_same_matrix_to_roundoff` | 两侧都是 bayesmith 的算术了 | **见下** |
| 6 | `test_the_two_packages_now_agree_on_the_full_information` | 同上 | **见下** |
| 7 | `test_agreement_is_not_because_both_dropped_the_term` | 断言 `mine / first == 1+2f²`,**只读 bayesmith** | 单侧,仍有效 |

第 5、6 条按字面读是「本包与本包比」,而那正是 `SWITCHED` 守卫说会「永远绿」的形状。
**但那是一个可以检验的断言,不是一个可以推理的断言。** 两侧的**构造路线不同**:一侧
`linear_operator` 拿测试里写的图建块,另一侧 `local_block` 拿适配器从一个裸 forward
与一个 `NoiseModel` **合成**的图建块。

**变异 U6**:让 `graph_for_information` 忽略调用方的噪声模型。**四条红**(常数 σ 那条
加三个 f 的 radiometer)。所以它们**还能失败**,而且失败在适配器上——这个文件现在是
一条接缝守卫。

`uncertainty.py` 因此**不进 `SWITCHED`**,与 `numpyro_bridge.py` 同一形状:
`as_noise_model`(5 个包内消费者)、`FlatMatrix`(config products **永久**保持面)、
`_named_spans`(随 Wave C/D)、`push_forward` 全部留守,**文件不整删**。这与守卫的
另一个方向也一致(不在 `SWITCHED` 里就必须有 cross-check)。此外该文件**本就与 `gls`
共用**(`test_migration_records.py` 的 aliases 里 `uncertainty` 与 `gls` 都映到
`noise_logdet`),而 `gls` 是 Wave B。

**顺手改掉两句已经过期的**:第 5、6 条的类 docstring 逐字写着「the two packages」。
改写成它们现在测的东西(两条构造路线),并把 U6 写在旁边——一个能失败的守卫,理由要
和它一起放着。

## 七、变异集:7 条 **7 杀**,两条跨仓,一条是本页的判据本身

基线前后各一次绿(两仓各一次)。

| # | 变异 | 仓 | 判决 | 指名红 |
|---|---|---|---|---|
| U1 | 远端的 `ValueError` 不再翻译成本包的类 | rheplicant | KILLED(5) | `test_the_refusal_wears_this_packages_class_and_keeps_the_original` 等 |
| U2 | `_REMOTE_KIND` 忘掉 `posterior_covariance` 也是协方差 | rheplicant | KILLED(1) | `test_a_posterior_covariance_is_refused_by_the_same_rule` |
| U3 | 过缝的矩阵用**协方差自己的**布局而不是 params 的 | rheplicant | KILLED(1) | `test_the_matching_covariance_propagates`(**既有测试**) |
| U4 | `propagate_covariance` 不再拒绝精度 | rheplicant | KILLED(1) | `test_a_precision_is_refused_rather_than_propagated` |
| U5 | 远端在**加 jitter 之前**量条件数 | **bayesmith** | KILLED(1) | `test_jitter_is_measured_after_it_is_applied` |
| U6 | 信息图忽略调用方的噪声模型 | rheplicant | KILLED(4) | **bayesmith 的 cross-check 四条**(§六) |
| U7 | 远端的 delta 方法按精度加权 | **bayesmith** | KILLED(6) | 含 `test_the_synthetic_sigma_and_data_do_not_move_the_report` |

**U3 是被一条既有测试杀掉的**,不是被本批新写的。追下去是对的:该变异让过缝矩阵声明
成一个 `__flat__` latent,而图有逐 latent 的节点,于是远端 `StructureError`——
`test_the_matching_covariance_propagates` 早就守着这条。按铁律 2 的「指认既有等价物」
处置,记在这里而不是再写一条。

**U6 与 U7 是本页两条判据各自的检验**:U6 检验「cross-check 还能失败吗」,U7 检验
「合成的两样真的够不到答案吗」。两条判据都不是论证。

## 八、一条被交接页点名的 oracle,重看之后的结论

`test_a_vector_latent_permutes_by_ITS_SPAN_and_not_as_one_row`。上一批已经量过:
它作为**数值** oracle 死了,作为**布局**断言还活着(把 `priors` 批的 P2 变异装回去,
它照样红),因为两侧到达同一布局的**路线不同**。

本批把 `uncertainty` 切完之后**再看一次**——两条路线**没有合并**:`priors` 仍是
拿到 `over` 序的矩阵再置换,`fisher_information` 仍是按 sorted 序索取块。所以结论
不变,**这次是在两条路都切完之后确认的**。

## 九、铁律 4 四件套

| | 项 | 结果 |
|---|---|---|
| (i) | 该批测试全绿 | rheplicant **10109 passed / 534 skipped** exit 0(341.2 s)加 **21 passed** exit 0(e2e,`-n 2`);两个 x64 会话由各自的驱动带过(全套内);bayesmith **1269 passed** exit 0(203.1 s) |
| (ii) | 接缝变异红 | 7 条 **7 杀**,两条跨仓,基线前后各一次绿(§七) |
| (iii) | 旧实现删除、计数守卫刷新 | `jnp.linalg.inv`、`jax.jacfwd`+`einsum` 两处删除;拒绝普查 244 → **250**,`StateValidationError` 58 → **64**,`test_uncertainty.py` 8 → **14**;README 计数 10651 → **10663** |
| (iv) | 文档实测数字重测 | 上述;**D30**、**D31** 入簿(登记簿到 **D7–D31**),**D29 回填「已落地」**;附录 A 补 U1–U7;附录 B 表头 244 → 250、`test_uncertainty.py` 清单由 `_sites()` 重生成 |

## 十、本页动了什么

- **rheplicant 源码**:`parameter_covariance` 与 `propagate_covariance` 成为门面;
  新增 `_REMOTE_KIND` 与 `_remote_flat`;`propagate_covariance` 新增一条缝前拒绝。
- **rheplicant 测试**:三条按分诊第二列改写(各带一条 dtype 兄弟断言),十条新守卫,
  两个 fixture 的函数体提出来供直接调用(一份拼写、两个调用方),普查 pin 三处刷新。
- **bayesmith**:cross-check 两处类 docstring 改写(§六);**顺手修掉本仓唯一一条
  ruff 错误**——它随 `0c5ca10` 进来,而 CLAUDE.md 与交接页 §二 都还写着
  「`ruff check src/ tests/` 干净」,所以那句话已经假了一天。两仓现在都是
  `All checks passed`(rheplicant 侧 13 条与本批无关,见 §十一)。
- **登记簿**:D29 回填、D30/D31 新增。

## 十一、留给下一位

1. **Wave A 完成。** 五个模块全部处置:`identifiability`、`sensitivity`、`priors`
   已切且已进 `SWITCHED`;`numpyro_bridge`、`uncertainty` **各切了一半**,按契约
   留守其余,两者都**不在** `SWITCHED` 里且 cross-check 保留。下一步是 **P2 余项**
   的五项(G2、G9 全量、G10、G12、G14),并行候选仍是 **G2**。
2. **G15 未动**,解除条件仍逐字写在 `uncertainty._prior_precision` 的 docstring 里:
   远端有了**带先验的非线性局部块**并发布之后,删掉该函数、调用改成
   `include_prior=space is not None`,**只有那一行会变**。
3. **D23 仍是唯一一条已登记、未裁决、无守卫的语义差。**
4. **rheplicant 的 `ruff check src/ tests/` 有 13 条**,全部与本程序无关
   (`I001` 导入排序 **9** 条、`F401` 未用导入 3 条、`E501` 1 条,散在
   `identifiability.py`/`sensitivity.py`/`gui/`/几个测试)。**没有 CI 在跑它**,
   所以它不会自己红;bayesmith 侧相反,那边一直干净而本批把它修回干净。
   要不要把 rheplicant 也扫干净是一条独立的决定,不属于本程序。
5. **一次新的「结果分不清『是 X』与『查询没发生』」**(计划 §七 第 1 条那一族,
   本会话又见一次,记下来因为形状是新的)。e2e 那一趟被写成一条复合命令,前半
   `cd` 去了 bayesmith,**`cd` 在同一条复合命令内是存活的**,于是 pytest 在
   bayesmith 里找 `tests/gui/e2e`,收集到零条。`PYTEST_EXIT=5`,日志里两行
   `bringing up nodes...` 和**零个点**。若当时只看「没有 FAILED」,那就是一次
   假绿。已在 rheplicant 里重跑:**21 passed exit 0**。
   > 教训的具体形状:两仓工作笔记写的是「`cd` 不跨回合存活,git 命令都用 `-C`」。
   > 反过来的一半同样咬人——**`cd` 在一条命令之内太存活了**,而它把一次跑测试
   > 送进了另一个仓,退出码说了实话(5,不是 0 也不是 1)。

6. **铁律 4(i) 的「查一次 CI」仍未入簿**(见 `2026-08-27-ci-flat-chain.md` §六)。
   本批同样**没有自作主张改铁律**;而 owner 已把推送改成攒到最后一起推,所以
   「收尾清单里加一行」这个提法要和新的推送时机一起想:CI 只在推送时跑,而推送
   现在只发生一次。
