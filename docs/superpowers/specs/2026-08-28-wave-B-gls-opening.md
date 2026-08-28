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
3. **跨仓 cross-check `test_noise_logdet.py` 的 17 条与 Fisher 行共享**
   (`uncertainty.md` 也指着它)。按铁律 2 逐条改籍或指认,**不能随文件
   消失**——而且因为它同时属于两行,退役它需要 `uncertainty` 那一行也同意。

## 四、留给下一位的一句

**契约页会把人带偏,而它没有说错。** 上一批(`linear`)读出的是两处**过期**,
处置是带日期的更正;这一批读出的是一处**会被读反的措辞**,处置是指路而不是
改写。两者都是铁律 7 的产出,但它们要求的动作相反——**先分清是哪一种**,
再决定动不动那一页。
