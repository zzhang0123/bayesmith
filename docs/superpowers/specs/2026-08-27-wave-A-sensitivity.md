# 执行页 Wave A · 模块 2 — `sensitivity` 切换

> **文档状态：`record`** · 已落地批次/审计/测量的历史记录，写作当天为真，非当前权威。索引见 docs/README.md。

> 计划:§五 Wave A / 铁律 1–3。前一批次:`2026-08-27-wave-A-g1-wiring.md`。
> **日期**:2026-08-27 · **本页状态:切换完成,套件全绿;变异集 6 条里 2 条幸存,
> 两条都是真发现,其中一条是**未结清的开放项**(见 §五)。**

## 〇、分诊表(第一次三列都非空)

| 列 | 数 | 内容 |
|---|---|---|
| 原样重放 | **62** | |
| **改写对适配器** | **2** | 两条 monkeypatch 本模块的 `MAX_NEWTON_STEPS` 来逼出不收敛;它们钉的那个求解已经在远端,所以改 patch bayesmith 的常量。**patch 这边被 re-export 的名字什么都不会改,却读起来像改了**——那正是「不会失败的测试」 |
| 带理由退役 | 0 | |

## 一、与模块 1 的决定性差别:**没有任何东西是合成的**

`prior_sensitivity(space, pipeline, state, observed, noise_std, flags=None, ...)`
**自带**数据、噪声与 flags。所以模块 1 的 D22(合成三样 + 证明它们够不到答案)在这里
**不适用**——不是「要重新测量不变性」(上一版交接页那句也不对),而是**根本不需要
合成**:调用方给什么就传什么。

只摘掉 `joint_prior`,只因为 `to_graph` 拒绝它(图一个节点一个分布)。被选中 latent
自己的高斯先验**原样过缝**——它们正是这个函数要问的东西。

`flags` 能过缝,是因为 G1 接线上一批已落地。**这正是那一批必须先做的理由**:
`config/postflight/fitting.py` 把文档声明的噪声原样传进来,而那可以是 `FlaggedNoise`。

## 二、三条先验拒绝留在缝前

其中两条**否则永远到不了**:`to_graph` 会先拒绝一个无先验的 latent,而它的措辞
既不提 `prior_std` 也不提 `linear`,对一个持 `wiener_solve` 式声明的调用方等于没说。
第三条(非高斯先验)远端也拒绝、措辞还是本仓移植过去的——**仍然留下**,因为
三条里放一条到远端去发,会让调用方看到哪条消息取决于声明顺序。

## 三、删除(铁律 2)

`_newton`、先验矩的装配、`_refuse_rank_deficient` 随切换删除。
**`_flat_view` 删除**——模块 1 为它留了一批,因为 `sensitivity` 是它最后一个消费者;
铁律 1 的这条欠账在此结清。`MAX_NEWTON_STEPS` 改为 re-export(与模块 1 的
`DEFAULT_RANK_RTOL` 同一处理)。

## 四、一条**存在但无 fixture 能分辨**的语义差(D23)

rheplicant 在**观测雅可比的秩**上拒绝;bayesmith 在 **rest 项自身的曲率**上拒绝。
bayesmith 的 docstring 逐字说明为什么它的判据更对:一个被**下游** latent 的密度
持住的被选 latent(`child ~ Normal(parent, s)`,child 在选择之外),其
likelihood-only 的众数**完全良定义**,而雅可比秩检验会拒绝这个合法问题。

两个方向都存在:bayesmith **接受**一些 rheplicant 拒绝的(上述情形),也**拒绝**
一些 rheplicant 接受的(雅可比满秩但曲率条件数超过 `1/sqrt(eps)` 天花板)。

**本批次没有采纳也没有回避:两侧的拒绝消息措辞相同**(移植时保留了),所以门面
只翻译异常类,64 条测试无一分辨得出这条差。**登记为 D23**,并写明:它是一条
**已知的、当前无守卫的**语义差;要正式采纳 bayesmith 的判据,需要一条能分辨它的
fixture(下游密度持住的被选 latent),那是一次语义升级,走登记簿。

## 五、变异集:6 条,4 杀 2 存,**两条幸存都是真发现**

| # | 变异 | 仓 | 判决 |
|---|---|---|---|
| S1 | shift 丢掉后验 sigma 缩放 | bayesmith | KILLED(3 红) |
| S2 | 先验的二次拉力减半 | bayesmith | KILLED(**20** 红) |
| S3 | 远端去掉图侧精度守卫 | bayesmith | **SURVIVED** |
| S4 | 门面改吃 bayesmith 的默认展开点 | e-RHINO | KILLED(8 红) |
| S5 | 跳过先验预检查 | e-RHINO | KILLED(1 红) |
| S6 | 不把值加宽到 float64 | e-RHINO | **SURVIVED** |

### S3:守卫有家,但**这个文件到不了它**

D9 给 `prior_sensitivity` 补的图侧精度守卫,在本文件的 fixture 下不触发。它**是**
被钉住的——在 bayesmith 的 `tests/diagnose/test_precision_policy.py` 里,由那条普查
测试。所以这不是缺口,是**这个文件不制造那个条件**。记下来而不是假装它红了。

对照:模块 1 的同一条变异(W6)**是**红的。两个模块对同一条远端守卫的承重不同,
这本身是「谁承重」的一次读数。

### S6:`_widened` 在这个文件里**没有被行使**,而我无法在本批次内结清它

实测:`Latent("a", init=7.8)` 在声明期(float32 环境)存下的就是 **float32**,
在 x64 块里读出来仍是 float32,只有与 float64 相乘才提升。所以 `_widened` 看起来
应当是承重的——模块 1 里它就是(去掉它 45 条全红)。

但这里去掉它,**64 条全绿**。两种可能,本批次未能分辨:

1. 它在这里确实**不需要**(`data = jnp.asarray(observed, dtype=jnp.float64)` 的显式
   转换已经让整条链提升),那它是死代码,应当删掉;
2. 它**需要**,只是本文件没有一个 fixture 制造那个条件,那应当补一条**声明期
   float32 init** 的 fixture 把它钉住。

**留作开放项,写进交接页。** 不猜、不删、不加断言——这两种处置方向相反,选错任何
一个都会留下一个说不出自己在防什么的东西。

## 六、铁律 4 四件套

| | 项 | 结果 |
|---|---|---|
| (i) | 该批测试全绿 | e-RHINO **10063 passed / 522 skipped** exit 0(353.3 s);bayesmith **1280 passed / 0 skipped** exit 0(208.3 s) |
| (ii) | 接缝变异红 | 6 条,**4 杀**;2 条幸存已逐条归因(§五),其中 S6 是开放项 |
| (iii) | 旧实现删除、计数守卫刷新 | `_newton` / 先验矩装配 / `_refuse_rank_deficient` / **`_flat_view`** 删除;cross-check `test_diagnose_sensitivity.py` 退役并记入 `SWITCHED`;bayesmith 计数 1291 → **1280** |
| (iv) | 文档实测数字重测 | 上述;e-RHINO README 计数未动(用例总数不变:2 条改写、0 条增删) |

## 七、留给下一位

1. **S6 是第一件事**:决定 `_widened` 在 `sensitivity` 里是死代码还是缺 fixture。
   两个方向的处置相反,别猜。
2. **Wave A 还剩三个模块**:`priors`、`numpyro_bridge`、`uncertainty`。
   `uncertainty` 最重(`FlatMatrix` 永久保持、`as_noise_model` 留守、
   `_named_spans` 随 Wave C/D),文件**不整删**。
3. **D23** 是一条已登记但无守卫的语义差;采纳 bayesmith 的曲率判据是一次语义升级,
   需要先造出能分辨它的 fixture。
4. 九份草稿仍在 `860703d` 的历史里,是否重写历史是 owner 的决定(需强推)。
   > **【已处置 2026-08-28】** owner 授权重写历史;九份草稿已从 e-RHINO 的历史中移除(`860703d~1..HEAD` 22 个提交重写为 21,`f8a73eb` 因变空被剪掉),**重写后 HEAD 的 tree 与重写前逐字节相同**,九份未跟踪的工作副本原样保留。本行提到的两个 SHA 自此不再存在。

