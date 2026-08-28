# 执行页 **G6** — 证据消费面:一个 campaign 对自己说得出什么

> 计划:§四 **G6 证据消费面**(Wave D 先决)。结清登记页的那条待判项
> (`memory.reject_bad_term`),新增裁决 **D44**(`compress_reduced_basis` 的波次)、
> **D45**(`template_modes` 不是 `coherent_mode` 的改名)。
> 前一批次:`2026-08-28-g5-amortize.md`。
> **日期**:2026-08-28 · 实现全在 bayesmith 一侧,rheplicant **一行未动**
> (接线在 Wave D,受发布门约束)。

## 一、开工第一件事:`reject_bad_term`,判为**留守**

登记页 §三 写的判据是:查**结构**就是构造期契约(留守),查**统计**就是数值(归 G6),
并且写明「本页不猜,做 G6 那一批开工的第一件事是读它」。**读完之后,那条判据本身
不成立**——一条准入规则天然两样都查。它的六条:

| # | 检查 | 哪一类 |
|---|---|---|
| 1 | `REQUIRED_TERM_MEMBERS` 都在 | 结构 |
| 2 | `latents_ok(term)`(bag 与 chain 答得不同) | 结构 |
| 3 | `prior_share[0] != 0` —— 加温过的项会把先验用两次 | **算术** |
| 4 | estimator 不一致 —— GLS 与全高斯似然「其和两者都不是」 | **算术** |
| 5 | `epoch_id` 重复 —— 数据算两遍 | 簿记(后果是数值) |
| 6 | `_reject_shared_inputs`(§9.5)—— 共享输入产品即非条件独立,**52.6σ** | **统计** |

**改用三条可测的事实**,而它们一致:

1. **它守的两个容器都留守。** 唯一的两个调用方是 `memory.py:459` 的
   `BayesMemory.remember` 与 `chain.py:994` 的 `ChainMemory.remember`,两个类都在
   登记页 §2.2。
2. **远端没有它要回答的那个问题。** `src/bayesmith/` 里 `remember` / `admit` /
   `add_term` / `append` **一个都没有**——`compress_campaign` 是**一张图一次调用**
   折完所有 epoch。「这是别人建的一个项,能不能加进我已经有的?」在那边从不被问,
   所以迁这条规则**要先发明它所守的那个累加器**。
3. **它读的每一个字段远端都没有。** `epoch_id`、`prior_share`、`estimator`、
   `inputs`、`represents` 在 `bayesmith/evidence/` 七个模块里出现 **0 次**(实测
   逐文件计数);远端 `Factorization` 只有 `epoch_plate`/`survivors`/`per_epoch`。

**不拆。** 第 3、4、6 条确实是算术与统计,§9.5 更是通用贝叶斯;但每一条读的都是容器
字段,而该函数自己的 docstring 给了不拆的理由——「a rule enforced in one accumulator
and not the other is a rule with a way round it」。跨两个包拆是同一缺陷高一层。

**结果:G6 = 7,不是 7+1;STAY 从 12 变 13,OPEN 归零。** 数字由
`probe_14_g6_enumeration.py` 重跑打印,本页只转录(登记页自己立的规矩)。

## 二、交付 7 里的 6,第 7 个有了**波次**而不是「跟着走」(D44)

`compress_reduced_basis` 在登记页上写的是「依赖 G4;**排期上跟 G4 走**」,而
**G4 的「跟着走」在那页上没有终点**——一条只写着「跟另一件走」的推迟,正是 §八
「限制需要解除条件」付过四次学费的形状。**归 Wave C**,与 G4 余下三个名字同批,
理由是量出来的:

* **它的数值核心本包已经有了。** 它自己的 docstring 写着这是「`compress_linear`'s QR
  with `S_e^T` in place of the design matrix」。**实跑核实**:40 个样本、5 维基,
  `compress({"c": rows.T}, ...)` 的 `log_prob` 对着 numpy 写的
  `d ~ N(S^T c, s² I)` 精确似然,六个探点最坏相对误差 **1.76e-16**。
* **它剩下的每一件都读容器**:`basis.rows`、`c_ref`、`coefficients(values)`、
  `reference_values`、`support` 五个字段,而 `ReducedBasis` 是留守容器(D12)。
* **它要的 `c(theta)` 由 `build_reduced_basis` 造**,而那是 G4 归 Wave C 的三个名字
  之一。在造它的东西存在之前先写吃它的东西,是对着一个没人定过的接口写实现。

## 三、**D45**:`coherent_mode` 同名不同题,而登记页自己警告过

登记页 §2.1 把 `coherent_mode` 列进「已有对应物」,并**同时**写着「同名不等于同题」。
本批把两边并排读了,**四处不同**:

| | bayesmith 的 | rheplicant 的 |
|---|---|---|
| 输入 | 存下的项 **+ 一个点** | 压缩时存下的**逐 epoch 摘要** |
| 残差 | 该点上的 `‖Rx − z‖²` | **出张成**的 `‖(I−P)z‖²`,过了本 epoch 自己的最佳拟合 |
| 自由度 | **第一个项的行数** | **逐 epoch 各自的,求和** |
| 命名模板 | **没有** | `{name: {mean, scatter, z}}` |

两者只在**逐 epoch 的极小点**上重合,而那不是 campaign 取值的点。所以**新增
`template_modes`,不改 `coherent_mode`**:后者是 0.4.0 已发布的表面,改签名要一次
破坏性发版(铁律 5),而这里没有任何东西需要那个代价。**两处 docstring 互指**。

## 四、oracle:**三种,互不相关**

| 仪器 | 对什么 | 实测 |
|---|---|---|
| `held_out_z` | **从头重拟**的留一后验(numpy,不共享一行代码) | 12 个 epoch,最坏相对误差 **1.5e-16** |
| `residual_summary` 的 χ² | **它自己的零分布**(400 epoch 的均值,不是公式) | 均值 **18.55** 对自由度 **19** |
| `shrinkage_power` | **整数比**的幂律 | −0.5 与 −1.0,**15 位** |

外加两条**大效应**的判别 fixture(按 `ci-flat-chain` §三 的判别法):

* **漏掉 nuisance 列的投影仪**:含 nuisance 自由度 19、均值 **18.55**;不含自由度 22、
  均值 **1227.4**——**60 倍**,在一份没有任何毛病的数据上。
* **最紧方向 vs 最紧坐标**:近共线设计上两个坐标宽度都是 **702**,最紧方向 **0.0583**
  ——**一万两千倍**。0.1 的地板下,按坐标读会说「没过线」,按方向读说「早就过了」。

命名模板那一组同样是倍数而不是容差:注入后 `right` 的 z 从 **+0.59** 到 **+23.28**,
而正交的 `wrong` 从 −0.76 到 −0.04(哑火),在张成内的 `in_span` 两次都是**恰好 0.000**。

## 五、顺带查出并修掉一个 0.4.0 就带着的缺陷:`compress` 信任了一个零权重

`compress` 不在 `seen` 上做**选择**,而是靠 `sigma = inf` 给出的零权重。但
**`0.0 * nan` 是 `nan`**,而一个被旗标的样本**通常正是因为它是 NaN 才被旗标的**。
实测,四个样本、index 2 处 `sigma = inf`:

| 放毒的地方 | `factor` | `target` | `offset` |
|---|---|---|---|
| 干净 | 有限 | 有限 | 有限 |
| **数据**里那一个样本 | 有限 | **NaN** | 有限 |
| **design** 里那一个样本 | **NaN** | 有限 | 有限 |

**第一种是安静的那一种**:`information()` 读 `factor.T @ factor`,依然有限且良态,
所以 campaign 审计报健康,而它产出的每一个密度是 NaN——一旦折进累加器就不可逆。
这正是上游 `compress_linear` 的注释记着自己关掉过的那个洞(「every other masked path
in the package already selects; this was the one copy that did not」),而本包这一份
就是那个 did not。**已修,两半都选择,没有旗标的 epoch 逐比特不变。**

## 六、「模板在张成内」不能写成 `norm > 0.0`

上游 docstring 说一个完全落在设计列空间里的模板「projects to exactly zero」。
**在精确算术里对,在浮点里不对**,而这条差别是危险的那一侧。实测:拿一个**本身就是
设计列**的模板,投影后剩下的是舍入——float32 下是它自己范数的 **6.0e-07**——而
`norm > 0.0` 为真,于是代码**除以那个舍入范数**,交回一个**任意单位向量**与残差的内积,
实测 **−0.2517**。一个看起来平平无奇的投影,代表的却是「这条模板被完全解释了」;
更糟的是那个方向是 SVD 的舍入挑的,**换台机器就换一个数,而两台都像在测量**。

切点改成 `sqrt(eps)`(手上算术的)。**与 `numerical_rank` 是同一个公式、不同的规则**
——那条宽是因为二次型平方了条件数,这条宽是因为判成「在张成内」会让模板**更安静**,
而对一个检测统计量来说安静是安全的一侧。(D41 立的规矩:两个不同理由不是一个事实的
两份拼写,所以**不**共用一个常量。)

## 七、变异集:12 条,**10 杀 2 必存**,而第一轮 3 条幸存里只有 1 条是缺口

登记先于实跑。第一轮 **9 杀 3 存**,三条追到底之后**是三件不同的事**。

| # | 变异 | 第一轮 | 结论 | 修后 |
|---|---|---|---|---|
| M1 | `compress` 不再选择 `seen` | KILLED(2) | | KILLED |
| M2 | 张成内的切点回到 `0.0` | KILLED(2) | | KILLED |
| M3 | 残差不减投影 | KILLED(3) | | KILLED |
| M4 | 自由度不减秩 | KILLED(5) | | KILLED |
| M5 | 饱和时 reduced χ² 返回 0 而非 nan | KILLED(1) | | KILLED |
| M6 | 总自由度用第一个 epoch 的乘以 N | **SURVIVED** | **fixture 分不出对与貌似对** | KILLED(1) |
| M7 | 模板 z 不乘 `sqrt(N)` | KILLED(1) | | KILLED |
| M8 | 留一不减那一项 | KILLED(1) | | KILLED |
| M9 | spread 丢掉 `R Σ Rᵀ` | KILLED(1) | | KILLED |
| M10 | 最紧方向改读最紧**坐标** | KILLED(2) | | KILLED |
| M11 | `not (σ > floor)` → `σ <= floor` | SURVIVED | **变异的是够不到的代码** | **SURVIVED(必然)** |
| M12 | OLS 斜率不中心化 `log σ` | SURVIVED | **等价变异** | **SURVIVED(必然)** |

### 七.1 M6:一个只有 **1.7 %** 散兵的「散兵 campaign」

那条测试的名字叫「uses each epoch's OWN degrees of freedom」,fixture 是 **60 个
自由度 19 的 epoch 加一个自由度 0 的**。求和 1140,按第一个乘以 61 得 1159——**差 19,
1.7 %**。`chi2_z` 从 −0.57 变成 −0.96,舒舒服服待在一条为噪声写的带子里。

**与 W8、X4、G3 的常数类、G5 的 Z9 同一族**:守卫够不到它要守的那个条件。这次的
成因是**量的尺度**,不是位置——判据写对了,fixture 没有把两条规则分开。
修法:**一半 campaign 用三分之一的样本**(自由度 3 对 19),求和 **660** 对相乘
**1140**,是「z 约等于 0」与「z 约等于 −10」的差别;并且**先断言这个分离**再断言 z,
所以 fixture 不能悄悄地不再散兵。

### 七.2 M11:变异的是**够不到的代码**,说出来比编一个 fixture 好

`not (σ > floor)` 与 `σ <= floor` **只在 NaN 上不同**,而实测 **NaN 到不了那一行**:
非有限的信息矩阵会先让 `cholesky` 抛,而且它抛得**远早于**求逆可能溢出到 inf——
特征值分离 **1e-160 就已经被拒**,而 **1e-12** 还能给出量级 1e12 的有限协方差。

所以这条 NaN 安全写法是**对「协方差怎么形成」将来可能改动的防御**,不是对输入的守卫。
**两条都做**:写法保留(`tightest_direction` 是公开的、它确实返回 nan,而且规则本身
是对的),但注释不再暗示存在一条守卫;**够得到的那一半**——非正定信息矩阵的拒绝——
补上了守卫。**编一个假 fixture 去杀它,才是更贵的错误。**

### 七.3 M12:等价变异,**必须**幸存

`centred @ (log_sigma − mean)` 与 `centred @ log_sigma` 是**同一个 OLS 斜率**,因为
`centred` 均值为零,`centred @ 常数 = 0`。实测三组随机输入:两组逐比特相同,一组差
一个末位。**登记错的是期望值,不是守卫**——同 G14 的 X1、G8 的形状。

> **第 (0) 条的第二半又兑现一次**:M6 的修补先**提交**(`3c53ed9`)再重跑整套。

## 八、铁律 4 四件套(按 G 项的形态)

| | 项 | 结果 |
|---|---|---|
| (i) | 该批测试全绿 | bayesmith **1523 passed / 0 skipped** exit 0;本批新增 **49** 条(1474 → 1523) |
| (ii) | 接缝变异红 | 12 条 **10 杀 2 必存**;第一轮 3 条幸存逐条追到底(1 个缺口、1 条够不到的代码、1 条等价变异),修补提交后重跑;基线前后各一次绿 |
| (iii) | 旧实现删除、计数守卫刷新 | **没有旧实现可删**;rheplicant 的证据族随 **Wave D** 退役。README 计数守卫红了两次,两次都照它自己报的数字改(1474 → 1522 → 1523) |
| (iv) | 文档实测数字重测 | CHANGELOG(一条 `Fixed` 加 G6);**D44/D45** 入簿(登记簿到 **D7–D45**);G6 行回填;登记页 §2.2/§2.5/§三/§五 按探针重生成 |

> **退出码又骗了一次,而这次有记录**:全套跑完 harness 的通知说「exited with code
> 0」,`run.exit` 说 `PYTEST_EXIT=1`,真有一条红的(README 计数守卫,它该红)。
> 两份 CLAUDE.md 记的那条,今天又值一次。

## 九、留给下一位

1. **不受发布门约束的项到此清空。** (甲)两件(G5、G6)都做完了,程序下一步是
   **收尾发布(D13)**,而**发新 PyPI 版本号是授权明写要停下来问 owner 的三件事之
   一**,它天然落在最后那次推送旁边。
2. **`compress_reduced_basis` 归 Wave C**(D44),与 G4 的
   `score_directions`/`build_reduced_basis`/`basis_fidelity` 同批。
3. **Wave D 接线时要量的两件,本批没量**:(a) rheplicant 的
   `EpochResidual`/`HeldOut` 类身份有没有被测试钉住——若钉住,门面要用本批返回的
   dict/NamedTuple **构造**它们而不是替换,那是 D12 在证据容器上的同一形状;
   (b) `systematic_floor` 上游读 `memory` 与其 `factorization` 的先验并**微分**它们
   (`_prior_curvature`),而本批的入口收一个现成的 `prior_fisher`——**那一层由谁算**
   是接线那一批的第一个问题。
4. **`diagnostics.py` 现在 754 行**,项目自定的上限是 800。下一次往里加东西之前
   先拆,不要等它越线。
