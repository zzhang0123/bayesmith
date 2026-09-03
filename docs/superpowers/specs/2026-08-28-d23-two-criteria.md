# 执行页 **D23** — 两条拒绝判据,以及一条已经在跑却没有守卫的语义差

> **文档状态：`record`** · 已落地批次/审计/测量的历史记录，写作当天为真，非当前权威。索引见 docs/README.md。

> 计划:§二 **D23**(`prior_sensitivity` 的拒绝判据:雅可比秩 vs 曲率)。
> 本页把那一行写的「先得造出能分辨它的 fixture」兑现,并**拍板**。
> 前一批次:`2026-08-28-g6-consumption.md`。
> **日期**:2026-08-28 · 本批改动在 **rheplicant 的 tests/ 与一处 docstring**,
> 两包的 `src/` 数值面**一行未动**——所以它不受发布门约束。

## 〇、为什么这一项现在做,以及做的时候框架变了

D23 是程序里**唯一一条已登记、未裁决、当前无守卫**的语义差(D39 已登记未裁决,但
两个方向都钉住了)。原行写的下一步是:「要正式采纳曲率判据,先得造出能分辨它的
fixture……那是一次**语义升级**,届时在本行拍板。」

**造出来之后,「升级」这个词就不对了。** Wave A 切换 `prior_sensitivity` 的那一刻,
**曲率判据就已经是正在跑的那一条**——`sensitivity.py` 委托给
`bayesmith.diagnose.sensitivity`,而两侧拒绝的措辞相同、门面只翻译异常类,所以 64 条
测试一条也分辨不出来。于是唯一还开着的问题不是「要不要换」,是「**它有没有守卫**」,
而答案是没有。

**拍板的内容因此是「采纳 + 钉住」,不是「改变」。** 这是本程序第四次出现「照建议做
与已实测的事实冲突」的形状(前三次是 D13、D9、D19),只是这一次冲突的对象是那一行
**对自己处境的描述**,而不是它的建议。

## 一、两个方向,而**可达性不同**——这是本次的新事实

原行把这条差写成「双向」,并把论证建在方向 1 上。实测之后,两个方向不是对称的。

### 方向 1 — 远端**接受**,秩判据**拒绝**(原行的论证依据)

`child ~ Normal(parent, s)`,`parent` 被选中而 `child` 在选择之外。`parent` 只经
`child` 到达数据,所以在固定的 `child` 上观测雅可比对 `parent` **恒为零**;但
`child` 自己的密度属于 rest 项,所以 likelihood-only 众数**完全良定义**——它就是
`child` 的值。**在图里实测**:

```
observed-Jacobian rank : 0 of 1  (nullity 1)   -> 秩判据会拒绝
bayesmith prior_sensitivity(names=['parent'])   -> 接受,shift = -0.1096
```

**远端的论证一个字没错。** 但——

**本包声明不出这个形状。** `Latent` 的 prior 是**声明期**建好的 numpyro 分布,参数
是具体数组,那时还没有 latent 可以指。实测:

```
dist.Normal(<一个 Latent>, 0.5)   构造成功       <- numpyro 什么都收
      .sample(key)                TypeError: unsupported operand type(s)
                                  for +: 'Latent' and ArrayImpl
```

> **构造成功不是可达性,而探针第一版正是这么读的。** 它检查的是构造函数抛不抛,
> 而 numpyro 从不抛,于是它印出「direction A IS reachable」——**一条不会失败的
> 检查**。判据要看的是这条声明**被用起来**之后会怎样。同一族的第五次:G5 的 Z9、
> G6 的 M6、W8、X4、G3 的常数类。

### 方向 2 — 远端**拒绝**,秩判据**接受**(人人可达,而且今天就在跑)

一个**普通的两参数近共线设计**。实测**经本包自己的公开 API**
(`data = a·g + b·(g + sep·g²)`,noise 0.05,float64,`1/sqrt(eps)` 天花板 6.71e7):

| sep | `identifiability` 秩 | κ(H) | `prior_sensitivity` | 秩判据会怎么说 |
|---|---|---|---|---|
| 1e-1 | **2 / 2** | 1.18e4 | 接受 | 接受(一致) |
| **1e-3** | **2 / 2 满秩** | **1.01e8** | **拒绝** | **接受** ← 判别用例 |
| **1e-5** | **2 / 2 满秩** | **1.01e12** | **拒绝** | **接受** ← 判别用例 |
| 1e-7 | 1 / 2 | 6.39e15 | 拒绝 | 拒绝(一致,**不能**用作判别) |

中间两行就是这条语义差**可达的全部**。

> **fixture 的第一版是错的,而错法值得写下来**:用的是
> `columns = [g, g·(1 + sep)]`。那是第一列的**标量倍数**,在**任何** sep 下都精确
> 共线,于是四行全部秩 1/2——**方向 1 的秩亏穿着方向 2 的名字**,四行都「拒绝」,
> 看起来像一次成功的测量。第二列必须真的离开第一列的张成:`g + sep·g²`。

## 二、拍板:采纳曲率判据,补守卫

**【本次委托下拍板,2026-08-28】** 内容:

1. **采纳**——它已经在跑,而且它更对(远端 docstring 的论证不曾被驳倒);
2. **钉住**——`tests/inference/test_d23_refusal_criterion.py`,10 例;
3. **方向 1 保持登记**,解除条件写进那一行:若声明层哪天长出层级先验,那个方向就
   可达,届时需要它自己的守卫。

**守卫怎么写才不是白写**:它**先断言雅可比满秩**,再断言拒绝。少了前一半,那条拒绝
就是两个判据都同意的一次拒绝,这份文件什么也没钉住——**这正是「守卫够不到它要守的
那个条件」在设计阶段的样子**,而本会话已经因为它红过三次。配套的另外两条:一条
`sep=1e-1` 接受的基线(否则每一条拒绝都可能只是 fixture 不能用),一条断言**拒绝
信息说了雅可比不是理由**(远端在两个判据分歧时报测得的谱,而不是借一个不成立的秩
判决)。

## 三、顺带更正一句被切换落在后面的 docstring

`inference/sensitivity.py` 的模块 docstring 到今天为止还写着:

> 「…so a rank-deficient selection is refused, and the rank comes from
> `identifiability`, which already knows how to say which latents the
> degeneracy mixes.」

那描述的是**被换掉的**判据。切换发生在 Wave A,而这句话留在原地——**同一形状本程序
已付过学费**:D21 的契约页 §5.2、附录 B 过期一整批、本页 §〇 的 D17 那次差六十秒。
已改写:说清判据是曲率、两个方向各是什么、以及**命名仍然委托给 `identifiability`**
(那一半没有变,变的是判决)。

## 四、铁律 4 四件套

| | 项 | 结果 |
|---|---|---|
| (i) | 该批测试全绿 | rheplicant **10119 passed / 553 skipped** exit 0(352.9 s)加 **21 passed**(e2e,67.2 s);本批新增 **10** 条 |
| (ii) | 接缝变异红 | 5 条 **5 杀**,每一条都由**登记的**那条杀死;**四条跨仓**(改 bayesmith,看 rheplicant 红),基线前后各一次绿 |
| (iii) | 旧实现删除、计数守卫刷新 | **无实现改动**——本批只加守卫与改一句 docstring。拒绝普查 **250 → 252**,附录 B 逐文件清单由 `_sites()` 重生成 |
| (iv) | 文档实测数字重测 | D23 回填;README 计数;附录 B 总数与分类 |

## 五、变异集:5 条 **5 杀**,四条跨仓

登记先于实跑。

| # | 变异 | 仓 | **登记的**指名红 | 实跑 |
|---|---|---|---|---|
| C1 | 远端不再设曲率天花板(`healthy` 恒真) | **bayesmith** | `test_the_curvature_criterion_refuses_it_anyway` | KILLED(6) |
| C2 | 远端改回按**观测雅可比的秩**判决 | **bayesmith** | 同上 | KILLED(5) |
| C3 | 判别 fixture 的第二列改回 `g·(1+sep)`(精确共线) | rheplicant | `test_the_observed_jacobian_is_full_rank_there` | KILLED(6) |
| C4 | 天花板从 dtype 推导改成写死 float32 的值 | **bayesmith** | `test_a_well_conditioned_design_is_reported_on` | KILLED(2) |
| C5 | 分歧时借用秩判决的措辞 | **bayesmith** | `test_the_refusal_says_the_jacobian_was_not_the_reason` | KILLED(2) |

> **C1 顺带打红了一条既有测试**:`test_prior_sensitivity.py::
> test_a_rank_deficient_selection_is_refused_and_identifiability_is_named`。按铁律 2
> 「指认既有等价物」记账——那条测试一直在读这条拒绝,只是读不出它是**哪一个**判据
> 做的,这正是 D23 的处境。

**C3 是这一组里最该跑的一条**,因为它变异的是 **fixture 自己**:它把判别用例换成
两个判据都同意的那种,而**如果没有那条「雅可比满秩」的兄弟断言,它会活下来**——
拒绝照旧发生,守卫照旧绿。跑它就是问「这份文件钉住的是分歧,还是只是一次拒绝」。

## 六、留给下一位

1. **D23 结清,登记簿现在只剩 D39 一条已登记未裁决**,而它两个方向都已钉住、解除
   条件写在行内、归 Wave D。
2. **方向 1 的解除条件**:声明层长出层级先验的那一天。
3. **`identifiability` 与 `prior_sensitivity` 在分歧时会说不同的话**,而本批只钉了
   `prior_sensitivity` 一侧。哪一批要报「秩满而曲率不足」这件事给用户看,那是一次
   新的报告面,不是本行的事。
