# 执行页 Wave A · 模块 3 — `priors`(`JeffreysPrior`)切换

> **文档状态：`record`** · 已落地批次/审计/测量的历史记录，写作当天为真，非当前权威。索引见 docs/README.md。

> 计划:§五 Wave A / 铁律 1–5;登记簿 **D24**、**D25**。
> 前一批次:`2026-08-27-wave-A-g13-wiring.md`。
> **日期**:2026-08-27 · **本页状态:切换完成。两条登记在案的语义差同批落地;
> 交接页 §三.4 欠的「真链验收」在此结清,但**结清的方式和当初设想的不一样**。**

## 〇、分诊表

| 列 | 数 | 内容 |
|---|---|---|
| **原样重放** | **49** | `tests/inference/test_jeffreys_prior.py` 全数,一条未改 |
| **改写对适配器** | **3** | `tests/config/test_config_exits_npe.py::TestThePriorGate` 的三条,D25 的代价(§三) |
| 带理由退役 | 0 | (cross-check 文件的退役归铁律 2,不进分诊表) |

49 条**一字未动**就是这一批的验收:签名、异常类、被钉文案、返回布局全都保持。

## 一、门面拿不到图,而这是本批次的第一个设计问题

远端的接口是 `JeffreysPrior.information(graph, values)`。本包的是
`information(forward, values, noise_std, flags)`——**一个闭包,不是一个空间**,
因为它的调用方是 NumPyro 的 model body,手里只有一个绑好的 forward。
`to_graph` 要的是 `ParameterSpace` + pipeline,给不出。

**处置:把「从裸 forward 造图」这件事放进适配器**,不是放进 `priors.py`。
新增 `graph_bridge.graph_for_information(forward, values, noise)`,复用本模块已有的
节点函数(`_prediction_fn`、`_observation_fn`、`_observed_mask`),于是观测节点
**只有一个拼写**。适配器是本计划 §〇 第 3 类,造图是它的活。

### 合成两样,合法性**被测量**而不是被论证(D22 的形状)

* **数据**,取预测形状的零。Fisher 是**期望**信息——`J^T N^-1 J` 加上方差自己那项
  ——两边都不出现残差。
* **每个 latent 的密度**,取 improper flat。远端的块的先验字段「deliberately
  empty」,而被覆盖的那些**必须**是 flat 否则 `_check_against` 按类型拒绝。

**两条都量过**(`probe_prior.py`,并已提成
`TestTheSynthesisedInformationGraph`):数据从 0 换成 1e4,矩阵 `max|diff| = 0.0`;
未覆盖 latent 的先验从 flat 换成 `Normal(0, 1e6)`,`max|diff| = 0.0`。

**没有合成的是噪声**:它是调用方的,因为这个先验的**全部主题**就是噪声模型决定它的
形状。

## 二、切换后的第一次读数:**逐比特相同**

同一个块、同一组值,本包切换前后:

```
rheplicant(sorted 行序) [[  882378.53452251 -1833513.22295522]
                          [-1833513.22295522 64000128.        ]]
bayesmith(over 行序)    [[64000128.         -1833513.22295522]
                          [-1833513.22295522   882378.53452251]]
置换后 max|diff|          0.0
half-log-det 两侧          15.801698526453313 / 15.801698526453313
```

注意左上角那两个数:**882378 对 64000128**,比值 **72.5**——正是本包 docstring 里
写的「读第 0 行的调用方在 tour 自己的块上错 **7.4e+1**」。D24 的差异不是理论,
它就是这两个数。

## 三、D24 落地:置换住在门面里

`_rows_in_sorted_order(matrix, over, values)`。**按 span 置换,不按名字**:一个
三元向量 latent 和一个标量不能当作两行对调。已有的
`test_information_rows_are_in_sorted_order_not_declaration_order` 钉住结果;本批次
补一条 `test_a_vector_latent_permutes_by_ITS_SPAN_and_not_as_one_row`,因为
两个标量的块**说不出**这条区别——一个按名字对调的版本会返回形状正确、对称、正定、
乱掉的矩阵。

那条新测试的 oracle 是 `uncertainty.fisher_information`,即**切换前的那条路**,
所以它是**回归** oracle 不是独立 oracle,而且**只到 `uncertainty` 被切为止**。
这一点写在测试自己的 docstring 里,因为一个会随别的批次消失的 oracle 必须自己说出来。

## 四、D25 落地:拒绝提到构造期,而三条 config 测试被**文档**修好

远端 0.4.0 起拒绝 ambient float32 的 Jeffreys 信息矩阵。切换后这条拒绝会到达
`to_numpyro_model`——但**到得太晚、说得太少**:它在追踪期到达,措辞是「a Jeffreys
information matrix」,既不提 `joint_prior` 也不提是哪份文档声明的。

于是 `_refuse_a_joint_prior_in_single_precision` 放在 `to_numpyro_model` 的**构造
期**,措辞里带上声明、带上 310 nat、带上**两条出路**。P1 §三 的原则:图缝会抹掉
证据的拒绝住在缝前。

### 那三条 config 测试的处置,比预想的好

原以为要按「期望一条拒绝」改写。**实测发现 config 层早就有这条路**:
`runtime.jax_enable_x64`,而且它是 delivery 层自己那条 float64 拒绝里点名的
「remedy 2」。所以改写是**给文档补上它本来就需要的一句声明**:

`joint_results()` 在 x64 会话里**构建并运行**文档,并在那里给它写上
`runtime.jax_enable_x64: true`。两半都要:`build_runtime` 是**核对**而不是应用,
它把声明与进程实际的精度对起来,任一半缺席都被拒。

三条测试的断言**一字未改**,包括那条真跑 NUTS 并要求 `d ≈ 1.2`、`a ≈ 12.0` 的。
float64 下它们照样成立。

**这是「拒绝」这一侧付得起的证明**:代价不是三条测试的语义,是一行声明。

### 而那行声明**不能**写进 builder,这是套件教的

第一版写进了 `joint_prior_document()`。**两条毫不相干的 config 测试红了**
(`test_the_built_rule_agrees_with_t2c_generated_on_every_shipped_builder`、
`test_every_builder_in_every_helper_module_keeps_the_repair`):fixture 普查会
**无参地驱动每一个 `*_document` builder**,而 `runtime.jax_enable_x64` 在**构建期**
就对着进程核对,于是一个声明 float64 的 builder 在这套 float32 会话里根本**造不出来**。

这不是意外,是本仓已有的形状:projector 的 helper 写 `acknowledge_float32_sky` 是
同一个理由——**一份文档能声明什么,不只取决于它的含义,还取决于谁会去构建它**。
声明因此住在**运行它的那个 helper** 里,builder 一字未动。

## 五、真链验收(交接页 §三.4),以及它为什么不能是「均值移动了多少」

第一版写的是:同一个种子跑两条链(带 factor / 不带),断言 `fg_log_amp` 的后验均值
被先验往上推。**它红了,而红得对**:移动量 **-2.5e-6**。

换成饥饿一点的 fixture(`sigma = 1000`)之后移动量是 **+0.00512**,而一阶预测
`Cov @ grad(log prior)` 给 +0.0057——量级和符号都对。但**跨三个种子重复**:

| seed | 实测(log_amp, beta) | 预测 | 比值 |
|---|---|---|---|
| 20260827 | +0.00314, +0.01440 | +0.00573, -0.01660 | 0.55, **-0.87** |
| 7 | +0.00805, -0.01243 | +0.00560, -0.01352 | 1.44, 0.92 |
| 99 | +0.00602, -0.03345 | +0.00551, -0.01260 | 1.09, **2.66** |

beta 那一列**换了符号**。一条钉在这个数上的断言是一枚**穿着容差的硬币**。

**而这不是本文件的局限,是 `prior_sensitivity` 自己 module docstring 里的算术**:
从 `n_eff` 个抽样得到的后验均值,其 MC 标准误是 `1/sqrt(n_eff)` 个 sigma,而两条链的
噪声**相加**。那个模块存在的理由就是这件事。

### 改用共同随机数——**而这一半后来被证明是错的,更正记在这里**

当时的推理是:在势能上加一个**常数**,NUTS 的轨迹完全不变(梯度相同,常数从每一个
Metropolis 比值里约掉),所以同一个 key 上的两条链应当逐比特相同。实测确实如此:

| 噪声声明 | 这个先验 | 当时实测 `max|路径差|` |
|---|---|---|
| radiometer | **恰好是平的**(+15.80169853) | `0.000e+00` / `0.000e+00` |
| homoscedastic σ=1000 | `p(log A) ∝ A²`,不平 | **3.99** / **5.09** 个后验 sd |

**但第一行只在这台机器上成立。** radiometer 下每个 `mu` 的抵消**只精确到舍入**,
而一条 leapfrog 轨迹是**混沌的**——最后几位会指数放大。CI runner 上同一条断言给出
**1.89e-2 个后验 sd**,比本机大四个数量级。`numpyro_bridge` 批把它重钉成
「小于 1e-5 个后验 sd」,那是**同一个错误加了一个容差**,在 CI 上照样红。

> **本页当时写着「判据里一个容差都没有」,那句话是错的**:它把一台机器上的运气
> 当成了代码的性质。**Seam CI 因此从本批次(2026-08-27 14:50)起连红五次**,五次
> 全是这一条断言。
>
> 处置见 `2026-08-27-ci-flat-chain.md`:平先验那一半**不该由链来测**——它由
> `test_under_radiometer_noise_switching_it_on_only_shifts_the_posterior` 在**势能**
> 上确定性地测,三个相距很远的点差到 1e-7。留给链的只剩**大效应**的两条:链能跑且
> 会动;不平的先验把它挪开 4–5 个 sd。

## 六、铁律 2:cross-check 同批退役

`tests/crosscheck/test_diagnose_jeffreys.py` **删除**。它把 rheplicant 的**实时**求值
与本包对照,而 rheplicant 现在**就是**本包——三条断言一次性变成「本包与本包比」。

三个主题**都有既有等价物**(铁律 2 的另一分支,指认而非改籍),全在
`tests/diagnose/test_jeffreys.py`:平常数对 numpy 闭式
(`test_the_flat_constant_equals_a_numpy_closed_form_no_autodiff_touched`)、
两种噪声下的梯度(`test_the_noise_model_chooses_the_priors_shape`、
`test_the_radiometer_jeffreys_prior_has_a_zero_gradient`)、退化块的地板
(`test_the_eigh_route_floors_the_singular_block_to_effectively_zero`)。
`SWITCHED` 加 `priors.py`,理由逐条写在那一行的注释里。

## 七、删除了什么,留下了什么

**离开 rheplicant 的**:Fisher 的装配(经 `fisher_information` 的那条路)、
`half_log_determinant` 的 `eigvalsh` + 秩地板。

**留下的**,逐项有理由:

| 留守物 | 归类 | 理由 |
|---|---|---|
| `JeffreysPrior` 类本身、`over`/`rank_rtol` 字段 | 第 5 类容器 | 它是 `ParameterSpace.joint_prior` 的字段,config 逐字段读 |
| `validate_against` | 第 4 类预验证 | 声明期拒绝(两个先验在一个量上),措辞被钉 |
| `check_identified` 的**措辞** | 第 4 类 | 判决早已远端(经 `identifiability` 门面);本包的消息带着实测数字(`+6.420496`、`9.755e-05`),远端的是通用版 |
| `rank_tolerance` | 机制 | 读的是 `identifiability.DEFAULT_RANK_RTOL`,而那是 bayesmith 的再导出——值只有一份 |
| `_rows_in_sorted_order` | 第 5 类 | D24 的执行机制,是布局不是算术 |

`priors.py` 里现在**唯一**的 `jnp` 算术是那个置换的下标簿记。

## 八、变异集:6 条,5 杀 1 存,而**那一条的幸存是结论不是缺口**

变异集要跑**两个会话**:`test_jeffreys_prior.py` 要 x64(它自己的 module fixture),
而 `test_numpyro_bridge.py` 的 D25 两条要 float32 才制造得出条件。合在一条命令里跑
**基线就是红的**——一个装成失败的用法错误,恰好是本程序反复付学费的那个形状。

| # | 变异 | 仓 | 指名红 | 判决 |
|---|---|---|---|---|
| P1 | 不做 D24 置换,行序按 `over` | e-RHINO | `test_a_vector_latent_permutes_by_ITS_SPAN...`、`test_information_rows_are_in_sorted_order...` | KILLED(2) |
| P2 | 置换按**名字**而不是按 span | e-RHINO | `test_a_vector_latent_permutes_by_ITS_SPAN...` | KILLED(**1**) |
| P3 | 合成数据从 0 换成 1e4 | e-RHINO | — | **SURVIVED,见下** |
| P4 | 远端丢掉方差自己那一项(`(1 + 2f²)`) | **bayesmith** | 平常数九点全红等 | KILLED(11) |
| P5 | 去掉 D25 的构造期拒绝 | e-RHINO | `test_a_declared_joint_prior_is_refused_at_construction` | KILLED(1) |
| P6 | 远端不再应用秩地板 | **bayesmith** | `test_the_eigh_route_floors_the_singular_block_to_effectively_zero` | KILLED(1) |

基线前后各一次绿。**P4 与 P6 是跨仓击杀。**

**P2 只红一条**,而且正是那条向量 latent 的——两个标量的块对「按名字还是按 span」
一言不发,这是 §三 那条测试存在的全部理由,变异把它证成了。

### P3:一条**必须**幸存的变异

合成数据从 0 换成 1e4,整套绿。按通常读法这是「没有守卫」。**这里不是**:
`TestTheSynthesisedInformationGraph::test_the_synthesised_data_cannot_reach_the_answer`
**断言的就是这件事为真**——Fisher 是期望信息,残差不出现在里面,所以四个量级的数据
变化一个比特都不动答案。**一条在这里变红的守卫会是在宣称一件关于期望信息的假话。**

所以 P3 的幸存是被证明的性质,不是缺口。**但「幸存要追到底」这条仍然适用**,追下去
发现一个真的可以出错的地方:那三条测量是拿**手搭的图**做的,而适配器是另一段代码,
它完全可以造出一张**不同的图**,于是那些测量对它一言不发。已补上闭环——同一条测试
最后断言 `graph_for_information` 造出的图给出的矩阵**就是**基线那一个。这一句同样
杀不掉 P3(两边一起变),但它把「测量的对象」和「发货的对象」钉成了一个。

## 九、铁律 4 四件套

| | 项 | 结果 |
|---|---|---|
| (i) | 该批测试全绿 | e-RHINO **10082 passed / 534 skipped** exit 0(350.1 s)加 **31 passed / 1 xfailed** exit 0(x64 seam,51.7 s)加 **21 passed** exit 0(e2e,62.3 s);bayesmith **1269 passed / 0 skipped** exit 0(212.0 s,1280 − 11 条随 cross-check 退役) |
| (ii) | 接缝变异红 | 6 条 **5 杀**,唯一幸存已证明为必须幸存(§八) |
| (iii) | 旧实现删除、计数守卫刷新 | Fisher 装配与 `eigvalsh` 地板删除;cross-check 文件删除并记入 `SWITCHED`;README 计数 10627 → **10636** |
| (iv) | 文档实测数字重测 | 上述全部;两条登记项 D24/D25 的裁决回填 |

## 十、留给下一位

1. **Wave A 还剩两个模块**:`numpyro_bridge`、`uncertainty`。
2. **`numpyro_bridge` 的第一个问题是站点名**:本包用 `"prediction"` 与 `"obs"`
   (可由 `obs_name=` 改),而 `to_graph` 的内部节点是 `__mu__` / `__data__`。
   直接委托给 `bayesmith.to_numpyro` 会改站点名,而站点名是保持面
   (`predict_from_samples` 读它们)。**先量再决定**,大概率是一条新登记项。
3. **`uncertainty` 最重**:`FlatMatrix` 永久保持、`as_noise_model` 留守、
   `_named_spans` 随 Wave C/D 退役,**文件不整删**。它切的时候
   `fisher_information` 会走,而本批次有一条测试拿它当回归 oracle(§三),
   那条测试要同批改写。
4. **D23** 仍是已登记、未裁决、无守卫。
5. 九份草稿仍在 `860703d` 的历史里,是否重写历史需强推,是 owner 的决定。
   > **【已处置 2026-08-28】** owner 授权重写历史;九份草稿已从 e-RHINO 的历史中移除(`860703d~1..HEAD` 22 个提交重写为 21,`f8a73eb` 因变空被剪掉),**重写后 HEAD 的 tree 与重写前逐字节相同**,九份未跟踪的工作副本原样保留。本行提到的两个 SHA 自此不再存在。

