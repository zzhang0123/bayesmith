# 执行页 Wave C / `calibrate` 开波 —— 8 条拒绝里的第 8 条

> **文档状态：`record`** · 已落地批次/审计/测量的历史记录，写作当天为真，非当前权威。索引见 docs/README.md。

> 计划:§五 Wave C / 铁律 1、3、7;**新增裁决 D57**(已自裁并落地)。
> 前一批次:`2026-08-28-wave-B-plan-opening.md`(D56 待裁决)。
> **日期**:2026-08-29 · **本页状态:开波与切换都已做,四件套齐备。**
> 上游提交:e-RHINO `7f28efd`。

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

## 四、切换(已做,e-RHINO `7f28efd`)

两个 `fit` 的循环体转调 `bayesmith.optimize.minimize`,两段手写 `lax.scan` 删除。
**六条拒绝留守,三条随迁。** 留守的六条理由是量出来的:5 条构造守卫无处可托
(远端 `minimize` 是函数,没有构造这一步),1 条 shape 守卫远端见不到 `observed`。

**三条 loss-sense 随迁,依 D11——而这一条我第一版做反了。** 初版把它们也留在近端,
理由是「逻辑只有 13 行、包在 21 行指名近端路线的文案里,委托会增加代码并让消息变坏」。
那个测量是真的,**但 D11 早就写了答案**:「loss 方向守卫**随迁**,`test_loss_sense`
**经包装重放**」——「经包装」三个字正是为这个问题准备的。
现在:检测委托给 `bayesmith.optimize.check_loss_sense`,包装层做两件缝抹掉的事——
翻译异常类(远端 `StructureError` → 近端被钉住的 `ParameterSpaceError`),
以及替换那句补救(远端指向 `fit`/`nuts`,**近端没有这两个名字**;把用户指向一个
他的包里不存在的函数,比不给建议更糟)。

**替换按远端原文匹配,所以它会静默失效**:上游一改措辞,消息就悄悄退回
`fit`/`nuts`,而异常照抛、类照对、本文件其余用例全绿。因此两个方向都钉住了
(近端路线**在**、远端路线**不在**),外加一条反空洞钉住远端自己的开头分句——
证明文本确实来自那边,而不是包装层偷偷改成了本地重写。

**D33 如期落地**:切完之后正好 1 条测试红,就是分诊点名的那条,
改成 `pytest.raises`,主张不变。

### 4.1 验收:逐位那条只能在删除之前取,而它取到了

见 §二之三。切完之后近端调的就是远端,同样的比较是循环的。
**永久可跑的换成非循环 oracle**:直线最小二乘的闭式解,NumPy 写自正规方程。

### 4.2 接缝变异:9 条,9 杀 —— 其中 3 条只有新写的守卫能杀

| 变异 | 杀它的 |
|---|---|
| M1 gradient 送 `method="adam"` | **spy** |
| M2 漏传 `beta1` | **spy,而且只有 spy** |
| M3 漏传 `eps` | spy + `test_config_exits_estimators`(它用了一个巨大的 eps) |
| M4 `steps` 写死 1 | 闭式 oracle |
| M5 history 槽里返回 values | 闭式 oracle |
| M6 删 shape 守卫 | `test_observed_shape` |
| M7 删 loss-sense 守卫 | `test_loss_sense` |
| **M8 包装层不再替换补救**(模拟上游改措辞) | **`TestTheWrapperSaysThisPackagesRoutes`,专为它写的** |
| M9 包装层吞掉远端的拒绝 | `test_loss_sense::TestRefusedByDeclaration` |

**M2 是这一批最有价值的一条,而它是被「问出来」的而不是被想出来的**:切完之后我问
「门面漏传一个旋钮,谁会红?」——实测**没人**。全套里唯一设非默认 beta 的测试用
`beta1=0.0, beta2=0.0`,只断言「有限」和「动过了」,**两条在远端默认 0.9 下都成立**。
补上 spy(D50 的同一补救,往下一层)之后 M2 才有人杀。
**没问那一句,这一批会带着一个漏传的旋钮通过全部四件套。**

## 五、留给下一位

**不要在近端补 beta 检查**——它已经在远端了(D57),补一份就是两份。
**要写异常翻译层**,而且要为它写变异体:一条把 `ParameterSpaceError` 换成
`StateValidationError` 的变异,如果没人红,说明类身份其实没被测到,那才是要修的东西。
