# 执行页 Wave B / `gls` 开波 —— 一次契约误读,和量它的那三行

> **文档状态：`record`** · 已落地批次/审计/测量的历史记录，写作当天为真，非当前权威。索引见 docs/README.md。

> 计划:§五 Wave B / 铁律 1、3、4、7。**没有新裁决项——D19 的延期部分在此结清。**
> 前一批次:`2026-08-28-wave-B-linear.md`。
> **日期**:2026-08-28 · **本页状态:模块已切,四件套齐备。**(开波部分见 §一–§三,切换见 §五起。)

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

## 五、切换(本页第二部分,2026-08-28 同日)

### 5.1 D19 的延期部分:结清,而**第一次测量一文不值**

D19 是 owner 在 2026-08-26 拍的「取数据锚定起步」,2026-08-27 的实测修正推翻了
它的**前提**(零中心 + `floor=0` 的退化首解不会发生,块根本建不起来),并把
第 (1)(2) 条**改为待 Wave B 实测**:

> 起点该不该动,现在是一个纯粹的**数值连续性**问题(`iterations`/`delta`/
> `converged` 三个可观测量会随起点动),而它**只能**对着 rheplicant 自己钉住的
> 数字量。

**本批就是那个 Wave B。** 逐条量:

**第一次量,结论是错的。** 把近端的种子换成远端的(`sigma_at(centre)` 代替
`noise.std(observed)`),44 条测试**全绿**。看起来「起点无所谓」。

**不是。** 两条硬钉的步数**都是地板**:`iterations == 5` 是 `MIN_REWEIGHTS`,
`iterations == 8` 是显式的 `min_reweights: 8`(那条测试自己的注释就写着
「The fixed point is reached in 5 steps, so a floor of 8 is visible only if
the declaration arrived」)。**地板看不见种子。** 44 条全绿因此什么都没证明。

**把地板降到 1**,`iterations` 才是真正的收敛步数:

| | 种子 | iterations | delta | converged |
|---|---|---|---|---|
| 近端 | σ 在**数据**上 | **4** | 1.637e-07 | True |
| 远端 | σ 在**先验均值**上 | **4** | 3.092e-07 | True |

而两个种子**确实不同**(反空转已查):`[2.940, 3.074, 2.938, 3.230]` 对
`[3, 3, 3, 3]`,相对差 **9.2%**。

**结论:起点不必动。** `iterations` 在敏感配置下也相同;`delta` 变了但两个都在
舍入量级、都远低于容差,所以 `converged` 不受影响。**D19 的延期部分至此结清,
不新增 D54。**

方向也查了(上一批的教训):对着 float64 跑到不动点的 IRLS,
**委托版 9.61e-08、被删的本地循环 1.055e-07** —— 委托版更接近。
常数 σ 那条路径**逐比特相同**。

### 5.2 两条拒绝都留守,理由不同

| 拒绝 | 为什么留守 |
|---|---|
| `需要 NoiseModel,不是裸 sigma` | 远端收的是 `sigma_of` **可调用对象**,从没听说过 `NoiseModel`。过缝之后**没有东西可认**。这是 `linear` 那条「拒绝 NoiseModel」的**镜像**:一个在该给定值的地方拒绝规则,一个在该给规则的地方要求规则,**同一个抹除,相反的极性**。 |
| `1 <= min_reweights <= max_reweights` | 远端**逐字**带着同一句话,但抛 `GraphError`,而**它自己的注释**说那个类在这里是 misfit。铁律 1 保异常类身份,前置一条比在一个站点翻译一个类便宜。 |

### 5.3 分诊表:空的

**44 条 gls 测试第一次运行就全绿,一条没改**;全套 10136 passed,与切换前**同一个数**。
拒绝普查不动(两条文案逐字保留)。这是本波第一个**分诊表为空**的批次,原因是
上一批已经把 `_check_solve_arguments` 搬进门面了——`gls` 借的正是它。

### 5.4 接缝变异集:14 条,**14 杀**(第一轮 13 杀)

靶子集这次**先回答「谁在测这段代码」**(`grep -rl iterative_gls tests/` 给出
11 个文件,11 个全列上),而不是凭印象——这正是 `linear` 批 M12 栽的地方。

第一轮唯一的幸存是 **N13:`reweight_tol` 没有转发**,而它的成因值得写清楚,
因为**不是「没有测试」**:远端用**同一个公式**算自己的默认值
(`max(REWEIGHT_TOL_EPS * eps, tol)`),所以在调用方不声明时,丢掉转发
**恰好等价**——而这一族的每一个 fixture 都不声明。缺口只在**显式**给定容差时
张开,而那个数值后果 **D50 已经证明在这里无法用括号夹住**。

于是照 D50 自己的办法处理:**直接钉接缝**。D50 钉的是 config→inference 那一跳;
本批**新造了 inference→bayesmith 这一跳**,而没有任何东西看着它。补了
`TestTheKnobsReachTheFarSideExactlyAsWritten`(7 条),其余五个旋钮一并直接钉住
——它们被变异集杀了,但杀它们的是**数值后果**,是间接的。

**复查过红是不是自己那一条**:重跑 N13,击杀者正是
`test_an_explicit_reweight_tol_arrives_unchanged`。

### 5.5 铁律 1:新增一条私名依赖

`gls.py` 现在借 `linear` 的 **`_OBSERVED`、`_as_far_block`、`_from_far_domain`**。
两个文件都已是门面,所以这是**适配器对适配器**,不是跨层借用;记在这里而不是
留给下一波去发现——上一批的开波页正是因为这条纪律才抓到 `uncertainty` 借
`_numpyro_distributions`。

### 5.6 计数

README 计数 **10722 → 10729**(+7,取自守卫自己的失败消息)。
拒绝普查**不动**(254),因为两条拒绝的 `pytest.raises` 站点一个没变。
cross-check **无事可做**(见 §三 第 3 条)。

