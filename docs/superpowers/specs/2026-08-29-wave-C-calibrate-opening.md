# 执行页 Wave C / `calibrate` 开波 —— 8 条拒绝里的第 8 条

> 计划:§五 Wave C / 铁律 1、3、7;**新增裁决 D57**(已自裁并落地)。
> 前一批次:`2026-08-28-wave-B-plan-opening.md`(D56 待裁决)。
> **日期**:2026-08-29 · **本页状态:开波已做,D57 已落远端,切换未做。**

## 〇、为什么先开这一波

Wave B 剩下的两行各有拦路:`plan`/`engines` 卡在 **D56**(待 owner),
`noise`/`likelihood` 卡在 **floor 换不过去**。`calibrate` 是**最可切的一行**:
**252 行、2 个公开类、9 条拒绝、私名普查零借用**,
而且**数值逐位相同**(§二之三)。注意「9 条」是修正后的数字——初稿写 8 条,
漏掉的那条见 §二之二。

## 一、铁律 1 私名普查 —— 干净

`from rheplicant.inference.calibrate import _*`:**全仓 0 处**。
两个私名(`_refuse_mis_shaped_observed`、
`_refuse_a_score_the_optimizer_would_walk_away_from`)都只在本模块内用。

## 二、铁律 7 + D48 式逐条问(**初稿,下一节更正**)

| # | 近端 | 远端 |
|---|---|---|
| 1 | `loss_fn` 声明 `sense=MAXIMIZE` | `check_loss_sense:139` **逐字相同** |
| 2 | 入口处 loss 非有限 | `check_loss_sense:149` **逐字相同** |
| 3 | 对完美拟合打分更差 | `check_loss_sense:156` **逐字相同** |
| 4,6 | `learning_rate must be > 0` | `_checked_settings:179` |
| 5,7 | `n_steps must be a positive int` | `_checked_settings:174`(`steps`) |
| **8** | **`beta1/beta2 must be in [0, 1)`** | **无对应物 → D57** |

前三条逐字相同说明 `check_loss_sense` 就是
`_refuse_a_score_the_optimizer_would_walk_away_from` 迁过去的那一份。

**这张表漏了一条,而且把「过得去」和「应该过去」混为一谈了——见下一节。**

**异常类身份不过缝**:近端是 `ParameterSpaceError` × 3 +
`StateValidationError` × 5,远端**一律** `StructureError` × 13。
初稿由此推论「门面要按调用点决定重抛成哪一个」——**下一节说明为什么不必**:
那条分界线正好和「哪些拒绝真的过缝」重合。

## 二之二、**上面那张表数错了,而且是我数的** —— 真结构是 9 条,6 条留守

**先把结论写在前面**:`calibrate` 的拒绝是 **9 条不是 8 条**,其中 **6 条必须留守**,
**0 条文案 pin 会断**。本节的初稿说「23 条 pin 里 12 条会断」,**那是错的**,
错法比结论有用。

**两处漏数,两种形状:**

1. **`check_observed_shape` 不是 `raise`,是调用一个帮手。** 我的 AST 普查扫的是
   `ast.Raise`,于是 `_refuse_mis_shaped_observed` 整条看不见。
   **与「按名字 grep 私名普查」是同一族**:匹配器只认一种写法,其余读成「不存在」。
2. **5 条构造守卫在 `__check_init__` 里,是**构造时**触发的。** 而远端
   `minimize` **是个函数,不是类**——**根本没有构造这一步可以镜像**。
   `AdamCalibrator(learning_rate=-1)` 近端当场拒;委托下去就要等到 `.fit()`。
   **一条拒绝什么时候响,是它契约的一部分**,而且被
   `test_inference_construction_guards.py` 钉着——文件名就是它的主张。

**真结构,而且分界线干净得出奇:**

| 组 | 条数 | 异常类 | 何时响 | 判定 |
|---|---|---|---|---|
| `check_observed_shape` | 1 | 帮手的 | `fit()` 入口 | **留守**——远端从不见 `observed` |
| `_refuse_a_score_...` | 3 | `ParameterSpaceError` | `fit()` 入口 | **可委托**,远端逐字相同 |
| `__check_init__` × 2 类 | 5 | `StateValidationError` | **构造时** | **留守**——远端没有对象可构造 |

**分界线正好落在异常类上**:3 条 `ParameterSpaceError` 全部委托,
5 条 `StateValidationError` 全部留守。于是:

* **异常翻译不是「按调用点」的,是一条映射**:`StructureError → ParameterSpaceError`,
  只对 loss-sense 那一条。§四 初稿把它说成本行最不平凡的工作,**也是高估了**。
* **文案 pin 一条都不断**:`n_steps`(8 条)、`beta1/beta2`(4 条)、
  `learning_rate`(7 条)全在 `__check_init__` 里,**留守则消息不变**;
  剩下 4 条 pin 的三句在远端**逐字相同**。

**为什么初稿会错**:我比对了缝两侧的**文案**,却**没有先问哪些拒绝真的过缝**。
于是把一批**根本不会被替换**的句子拿去和远端比,比出了 12 条「会断」。
**顺序错了:先定哪些过缝,再比文案。** 反过来做,每一句都能比出差异,而差异
全是假的。

## 二之三、数值验收:**逐位相同**

切换前先量了(纯 numpy/jax,两侧同一 `forward` 与 `loss_fn`,`y = a x + b`,32 点):

| 方法 | `|Δa|` | `|Δb|` | 120 步 loss 历史 |
|---|---|---|---|
| `gradient` | **0.000e+00** | **0.000e+00** | `max|Δ| = 0.0` |
| `adam` | **0.000e+00** | **0.000e+00** | `max|Δ| = 0.0` |

**逐位相同,连整条 loss 历史都是。** 所以这一行的数值风险为零,
剩下的全部是拒绝与文案的适配。

## 三、D57:第 8 条是最坏的一种

见登记簿 D57 全文。要点:远端把 beta 当普通浮点收下,而 `beta1=1.5` 在
`(x-3)**2` 上返回 **15.384941**——有限、无警告、真极小值的 5 倍。这是三种结局里的
**(c) 远端静默作答**,不是消息损失而是**正确性损失**。

**已落在远端**,理由是远端**自己的调用方**今天拿得到一个静默的错答案。
**近端那条不撤**——它在 `__check_init__` 里、构造时触发,而远端 `minimize` 是函数,
没有构造这一步。两边各一份守的是**两个不同的入口**,不是一条规则的两份拷贝。
(D57 初稿的第二条理由说反了,已在登记簿里更正。)

**顺带收窄了一次范围,而逼出它的是一条自相矛盾的测试。** 守卫最初放在
`_checked_settings` 公共段,`method="gradient"` 也会被拒;而 `optimize.py:288`
的契约写着 beta「**ignored by** `"gradient"`」。我写的
`test_the_gradient_method_is_unaffected` **docstring 说不该拒、断言写成会拒,
而且通过了**。改成 Adam-only。规则:**改变答案的就拒绝,契约说忽略的就真忽略。**

## 四、切换这一批要做的(未做)

1. 两个 calibrator 的 `fit` 转调 `bayesmith.optimize.minimize`,
   `GradientCalibrator → method="gradient"`、`AdamCalibrator → method="adam"`。
2. **6 条拒绝留守**(1 条 shape + 5 条构造),**照原样不动**——它们的消息、类、
   触发时机都不变,所以那 19 条文案 pin 也不动。
3. **一条异常映射**:近端自己先调 `check_loss_sense`,把它的 `StructureError`
   翻成 `ParameterSpaceError`,**然后**再调 `minimize`。
   比「在 `minimize` 外面包一个 `except StructureError`」好,因为后者是**靠论证
   安全的**(近端构造守卫已经把 `steps`/`learning_rate`/beta 都挡在前面了,所以
   `minimize` 里只剩 loss-sense 能响)——**而那个论证会随远端新增拒绝而失效**。
4. **D33 的 1 条分诊**已量(见 D33):
   `test_the_fixed_step_descent_really_does_diverge_on_the_unscaled_negation`
   改成 `pytest.raises`,主张不变、观测渠道变。
5. 四件套:分诊绿 / 接缝变异红 / 旧实现删除 + 计数刷新 / 文档数字重量。

## 五、留给下一位

**不要在近端补 beta 检查**——它已经在远端了(D57),补一份就是两份。
**要写异常翻译层**,而且要为它写变异体:一条把 `ParameterSpaceError` 换成
`StateValidationError` 的变异,如果没人红,说明类身份其实没被测到,那才是要修的东西。
