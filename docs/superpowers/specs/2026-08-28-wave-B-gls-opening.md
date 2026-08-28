# 执行页 Wave B / `gls` 开波 —— 一次契约误读,和量它的那三行

> 计划:§五 Wave B / 铁律 1、3、7。**没有新裁决项。**
> 前一批次:`2026-08-28-wave-B-linear.md`。
> **日期**:2026-08-28 · **本页状态:开波仪式已做,模块未切。**

## 〇、为什么这一页存在

`gls` 是 Wave B 的下一批。开波仪式(铁律 1 普查 + 铁律 7 契约阅读)做完之后,
**第二件读出了一个看起来会挡住整批的东西,而它是我读错了**。把误读和纠正它的
那三行测量一起写下来,因为**下一位会读同一页**,而那一页的措辞会把人往同一个
方向带。

## 一、铁律 1 私名普查 —— 干净

`gls` 的公开面只有两个名字:`GLSResult` 与 `iterative_gls`,两个都经
`inference/__init__.py` 导出。

`grep -rn` 全仓:**没有任何模块借 `gls` 的私名**。唯一的真实代码消费者是
`src/rheplicant/config/sections/conjugate.py`(`noise_from: gls`),其余十来处
全是 docstring 交叉引用。

反过来则不然:**`gls` 借着 `linear._check_solve_arguments`**,那是上一批的
保持面,**随本批到期**。

## 二、铁律 7 契约阅读 —— 以及我把 §5 读反了

`docs/migration/gls.md` §5 的标题是「Intended differences — the whole point of
the row」,开头是:

> **bayesmith's `iterative_gls` does NOT carry this bias, and must not be made
> to.** It is frozen-sigma IRLS ... so its fixed point satisfies `w = mean(u)`

我读成了「**近端带这个偏差、远端不带**」,于是以为这一批会**改变答案**——
把 rheplicant 的不动点从 `Σu²/Σu` 挪到 `mean(u)`,那是一次语义迁移,按铁律 3
要先裁决后切换。

**实测三行,结论相反。** 常数均值模型 + `RadiometerNoise`,`f ∈ {0.05, 0.2, 0.5}`,
`n = 2000`,直接量 **近端** `iterative_gls` 的不动点落在哪一边:

| f | 近端 w | `mean(u)` | `Σu²/Σu` | 离哪个近 |
|---|---|---|---|---|
| 0.05 | 99.872284 | 99.872299 | 100.128395 | `mean(u)`,**16 785×** |
| 0.2 | 99.489212 | 99.489166 | 103.602737 | `mean(u)`,**89 861×** |
| 0.5 | 98.722847 | 98.722908 | 124.632271 | `mean(u)`,**424 500×** |

**近端也是 frozen-σ IRLS**,也落在 `mean(u)` 上,三个 f 一致。所以 §5 那句
「does NOT carry this bias」讲的是**远端相对于 B1 的位置**,不是「近端带、
远端不带」的对比。

回头再读 §3 就清楚了:**B1 不是 `iterative_gls` 的缺陷**,是
「rheplicant 的 `nuts` 路线(它的 `dist.Normal` 自带 `−log σ`)与它的
`plan.sample` 梯度块(不带)在同一个模型上瞄准两个不同的估计量,**中间没有
守卫**」。B1 住在**似然的对数行列式**里,归 `plan`/`engines` 那一行,
不归 `gls`。

§5 自己的最后一句其实也这么说:「Consequence for the pending `plan`/`engines`
row (§四 4.2): **B1 must land first**」。

**这一页不改契约页。** 契约页没有说错,是我读错了——而这个区别重要:
铁律 7 要求契约**当下为真**,而它是真的。会把人带偏的是**标题**
(「Intended differences」)加上第一句的主语,读快了就成了两侧对比。
所以纠正写在这里,并在契约页 §5 前面加一句指路,而不是改写它的论断。

## 三、这一批的形状(未做)

两侧签名的差和 `linear` 那一批**同一个形状**:

```
近端:  iterative_gls(block, observed, *, noise: NoiseModel, prior_std, prior_mean, ...)
远端:  iterative_gls(block, sigma_of=None, *, precision_of=None,
                     depends_on_prediction=True, ...)
```

即数据与先验从**参数**变成 block 的**字段**,而 `NoiseModel` 变成一个
**可调用的 `sigma_of`**。所以门面仍然是**形状转换 + 拒绝前置**,外加一件
`linear` 那批没有的:**把 `NoiseModel` 包成 `sigma_of`**。

开工要做的三件,按 D48 的先例:

1. 逐条问 `gls` 的每一条拒绝**过缝之后还到不到得了**——它借的
   `linear._check_solve_arguments` 已经是门面里的了,所以这一条的答案
   多半是「已经在正确的一侧」,但要量。
2. **远端有 `check_prediction_dependence` 和 `_dependence_probe`,近端没有。**
   那是新能力,不是差异;要问的是门面暴不暴露它(保守侧:不暴露,
   因为公开面必须保持)。
3. ~~跨仓 cross-check `test_noise_logdet.py` 的 17 条要按铁律 2 处置~~
   —— **实测:不必,而这是本页第二处「台账指着的东西不是它说的那个」。**
   交接页与契约页都把这个文件列为 `gls` 的 cross-check(「17 条,与 Fisher
   行共享」),于是切 `gls` 看起来要连带处置它。**数一下**:该文件有
   **4 个 test 函数**(参数化后 17 例),而 `iterative_gls` 在里面出现
   **0 次**。四条全部是对数行列式估计量的比较——`Σd²/Σd`、二次式的根、
   两者之比 `1+f²`、以及 σ 不依赖预测时那个 gap 归零的反空转控制。
   **那是似然,是 B1,归 `plan`/`engines` 行**,和 `iterative_gls` 无关。
   
   于是 `gls` **没有专属的 cross-check 文件可退役**:契约页写的「Test」是
   B1 那一行的测试,加上远端自己的
   `tests/exact/test_gls.py::test_the_fixed_point_is_the_unbiased_estimator_not_the_gls_biased_one`。
   铁律 2 在这一批**无事可做**,而这把批次的规模显著缩小了。
   
   **两处误导同一个形状**:§二 那处是「标题 + 主语让人读成两侧对比」,
   这一处是「台账把一个文件挂在这一行下,而文件内容属于另一行」。
   两处都不是错的记录,都是**会被读错的记录**,而分辨它们的办法一样:
   **数它,不要读它**。

## 三点五、切换前要裁决的那一条(已量,不是问题,是发现)

两侧的循环**结构逐行相同**——同一个 `solve_at`、同一个 `step`、同一个
`unfinished` 的两个合取、同一个 `not depends_on_prediction` 短路,连三个常量都
一样(`MIN_REWEIGHTS=5`、`MAX_REWEIGHTS=100`、`REWEIGHT_TOL_EPS=8.0`)。

**差的是种子。**

| | 循环的第一个 σ |
|---|---|
| 近端 | `noise.std(observed)` —— 在**数据**上求值 |
| 远端 | `rule(centre)`,`centre = domain_centre(block)` —— 在**先验均值**上求值 |

后果分三层,按严重性:

1. **不动点相同。** 已量(本页 §二 那张表 + 契约页 §5):两侧都是 frozen-σ IRLS,
   都落在 `mean(u)`。所以**收敛时的 `solution` 与 `noise_std` 不受影响**。
2. **`iterations` 会变。** 它是 `GLSResult` 的一个字段,而按铁律 1
   「产品容器字段布局保持」它是公开面。**任何钉住步数的断言都要重读**
   ——上一批(D53)已经因为不动点位置吃过一次这个亏。
3. **不收敛时答案不同。** 循环被 `max_reweights` 截断时返回的是**路径上的某一点**,
   而两条路径从不同的地方出发。`converged=False` 的那些用例因此**不是**
   「同一个答案,不同的步数」。

**保守侧(建议,未裁决)**:门面自己算种子,即把 `noise.std(observed)` 的结果
作为第一个 σ 递过去,而不是让远端从先验中心起步。远端的 `sigma_of` 是一个
可调用对象,所以这做得到;代价是门面里多一行,收益是 `iterations` 与不收敛
用例都逐位不变。**若采纳,登记为 D54。**

## 四、留给下一位的一句

**契约页会把人带偏,而它没有说错。** 上一批(`linear`)读出的是两处**过期**,
处置是带日期的更正;这一批读出的是一处**会被读反的措辞**,处置是指路而不是
改写。两者都是铁律 7 的产出,但它们要求的动作相反——**先分清是哪一种**,
再决定动不动那一页。
