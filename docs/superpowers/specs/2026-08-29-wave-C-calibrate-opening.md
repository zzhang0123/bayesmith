# 执行页 Wave C / `calibrate` 开波 —— 8 条拒绝里的第 8 条

> 计划:§五 Wave C / 铁律 1、3、7;**新增裁决 D57**(已自裁并落地)。
> 前一批次:`2026-08-28-wave-B-plan-opening.md`(D56 待裁决)。
> **日期**:2026-08-29 · **本页状态:开波已做,D57 已落远端,切换未做。**

## 〇、为什么先开这一波

Wave B 剩下的两行各有拦路:`plan`/`engines` 卡在 **D56**(待 owner),
`noise`/`likelihood` 卡在 **floor 换不过去**。`calibrate` 是**最可切的一行**:
**252 行、2 个公开类、8 条拒绝、私名普查零借用**。

## 一、铁律 1 私名普查 —— 干净

`from rheplicant.inference.calibrate import _*`:**全仓 0 处**。
两个私名(`_refuse_mis_shaped_observed`、
`_refuse_a_score_the_optimizer_would_walk_away_from`)都只在本模块内用。

## 二、铁律 7 + D48 式逐条问:8 条拒绝,7 条过得去

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

**但异常类身份不过缝**,这是切换时的真约束:近端是
`ParameterSpaceError` × 3 + `StateValidationError` × 5,远端**一律**
`StructureError` × 13。**近端在远端只用一个类的地方分了两个类**,所以门面不能
一条 `except StructureError: raise X` 了事——**要按是哪一个调用点决定重抛成哪一个**。
这与 `linear` 那一批的形状相同。

## 二之二、文案 pin:23 条,**12 条会断**,而断的是两句

这一条把本行的估工改了,所以单独一节。逐句数(`grep` 全 `tests/`):

| 近端句子 | pin 数 | 远端 |
|---|---|---|
| `declares sense=` | 1 | **逐字相同** |
| `not finite at entry` | 1 | **逐字相同** |
| `scores a PERFECT` | 2 | **逐字相同** |
| `learning_rate must be > 0` | 7 | **逐字相同** |
| **`n_steps must be a positive int`** | **8** | 远端说 **`steps`** ✗ |
| **`beta1/beta2 must be in`** | **4** | 远端说 **`beta1 must be in`** ✗ |

**注意 `learning_rate` 那句逐字相同**——所以不是远端整体另起炉灶,断的**只是
参数名和消息形状**这两处。而且 `n_steps` 那 8 条里有一条在 `tests/config/`,
不在 `tests/inference/`。

**两句的处置不一样,而且理由不同——不要一刀切:**

* **`n_steps` 必须由适配层翻译回来。** 近端的公开参数**就叫** `n_steps`;
  用户写 `n_steps=0`,却被告知「`steps` must be a positive int」,
  **消息指了一个他没用过的参数**。这不是口味问题,是消息的正确性。
* **beta 那句应当改判 pin,不是保留。** 近端说
  「`beta1/beta2 must be in [0, 1), got 1.5, 0.999`」(**两个值都报**),
  远端说「`beta1 must be in [0, 1), got 1.5`」(**只报出错的那个**)。
  **远端那句更好**——它指名了是哪一个。按 D52 的先例改写测试并记下理由。

这正是 `linear` 那批 D52/D53 的同一形状:**先问「这条测试到底在主张什么」,
再决定保留还是改判**,而不是按「哪边先写的」决定。

## 三、D57:第 8 条是最坏的一种

见登记簿 D57 全文。要点:远端把 beta 当普通浮点收下,而 `beta1=1.5` 在
`(x-3)**2` 上返回 **15.384941**——有限、无警告、真极小值的 5 倍。这是三种结局里的
**(c) 远端静默作答**,不是消息损失而是**正确性损失**。

**已落在远端而不是留守近端**,因为 `calibrate` 一切,近端 calibrator 就成了门面,
只在近端留这条等于把同一条规则写两遍。

**顺带收窄了一次范围,而逼出它的是一条自相矛盾的测试。** 守卫最初放在
`_checked_settings` 公共段,`method="gradient"` 也会被拒;而 `optimize.py:288`
的契约写着 beta「**ignored by** `"gradient"`」。我写的
`test_the_gradient_method_is_unaffected` **docstring 说不该拒、断言写成会拒,
而且通过了**。改成 Adam-only。规则:**改变答案的就拒绝,契约说忽略的就真忽略。**

## 四、切换这一批要做的(未做)

1. 两个 calibrator 的 `fit` 转调 `bayesmith.optimize.minimize`,
   `GradientCalibrator → method="gradient"`、`AdamCalibrator → method="adam"`。
2. **异常类翻译**:按调用点把 `StructureError` 重抛成
   `ParameterSpaceError`(前 3 条)或 `StateValidationError`(后 5 条)。
   铁律 1 要求类身份保持。
3. **文案:`n_steps` 那句翻译回来(8 条 pin),beta 那句改判 pin(4 条)**——
   理由分别见 §二之二。这两件加起来是本行**真正的适配工作量**,
   而开波之前它是看不见的:8 条拒绝里 7 条「过得去」会让人以为这一行几乎是免费的。
4. **D33 的 1 条分诊**已量(见 D33):
   `test_the_fixed_step_descent_really_does_diverge_on_the_unscaled_negation`
   改成 `pytest.raises`,主张不变、观测渠道变。
5. 四件套:分诊绿 / 接缝变异红 / 旧实现删除 + 计数刷新 / 文档数字重量。

## 五、留给下一位

**不要在近端补 beta 检查**——它已经在远端了(D57),补一份就是两份。
**要写异常翻译层**,而且要为它写变异体:一条把 `ParameterSpaceError` 换成
`StateValidationError` 的变异,如果没人红,说明类身份其实没被测到,那才是要修的东西。
