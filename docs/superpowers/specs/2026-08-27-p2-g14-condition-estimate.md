# 执行页 P2 余项 · **G14 measured-κ 诊断**(D15(a))

> **文档状态：`record`** · 已落地批次/审计/测量的历史记录，写作当天为真，非当前权威。索引见 docs/README.md。

> 计划:§四 G14;登记簿 **D15**(取 (a):`condition_estimate` 移植为显式标注
> 「measured-κ,不可作守卫」的诊断)。新增裁决 **D37**。
> 前一批次:`2026-08-27-p2-g10-g12.md`(G10+G12)。
> **日期**:2026-08-27 · **本页是一个 G 项批次**,实现全在 bayesmith 一侧,
> rheplicant **一行未动**。

## 〇、这一项落在一条**相反的既有决定**上,而那条决定没有被推翻

bayesmith 里有三处白纸黑字写着「这东西**故意没有移植**」:

1. `exact/conditioning.py` 的模块 docstring;
2. `tests/crosscheck/test_conditioning.py::TestWhatWasRejectedIsStillAbsent`
   (断言 `not hasattr(ours, "extreme_eigenvalues")`);
3. `tests/crosscheck/test_linear.py::test_bayesmith_does_not_carry_condition_estimate`;
   外加 `docs/migration/conditioning.md` §5 整节。

第 2 条的 docstring 甚至逐字写着:「**这条会在有人最终移植它时变红——那时他应该
去读那份拒绝了它的模块 docstring,而不是把一个绿套件当成同意。**」

**读过了。论证一个字没错,一个字也没有撤回。** 变的不是论证,是那条论证**管的
范围**:它说的是**守卫**,而这个例程还有第二个用途,那个用途不是守卫。

* 一个**界**把 λ_min 用先验曲率**垫住**,所以它**结构上**报不出一个近简并的分区
  ——而那件事整个住在 λ_min 里。
* 一个**诊断**看得见它。

D15(a) 裁的就是后者。所以本批**不是**推翻那条决定,而是把它从「这个包里没有这个
函数」收窄回它本来的形状:「**没有守卫读这个函数**」。

## 一、交付了什么

| 名字 | 在哪 | 是什么 |
|---|---|---|
| `extreme_eigenvalues(operator, template, key, iterations)` | `exact/conditioning.py` | `(λ_max, λ_min)`;λ_min 由 `λ_max·I − M` 上的第二次幂迭代得到。**复用 `largest_eigenvalue` 取顶**,所以幂迭代只有一份 |
| `condition_estimate(block, *, precision, iterations, key)` | `exact/solve.py` | 测得的 κ,**用先验曲率垫住** λ_min。docstring 第一段就写「不是界、不要拿来当守卫」 |

两者都经 `bayesmith.__init__` 的惰性表导出。

## 二、偏差是**数字**而不是主张

`geomspace(1, 1e7, 50)`,真 λ_min = 1.0:

| 迭代 | λ_min | 报出的 κ(真值 1e7) |
|---|---|---|
| 50 | 10210.8 | 979 |
| 200 | 2351.3 | 4.25e3 |
| 800 | 805.9 | 1.24e4 |
| **2000** | **501.2** | **2.00e4** |

**四十倍的功,还差五百倍。** 这与 rheplicant 与本仓 docstring 里记的
「2000 步仍留 ~700 倍」同量级(具体因子随起始向量变,所以测试断的是**方向与量级**
而不是一个钉死的比值——钉住它就是在测 PRNG)。

移位算子的领头特征值全部挤在 λ_max 上、间隙趋零,所以**任何**迭代次数都分不开它们。
这不是预算问题,有一条测试专门说这句话。

## 三、它能做的那件事:**看见简并**

`collinear_pair`——数据只定住 `a+b`,`a−b` 只由先验持住。断言写成**比值的比值**,
所以不依赖任何一个绝对数:**联合块的测得 κ 超过单个成员的倍数,必须远大于两者的
界之间的倍数**(测试里要求 >10×)。界做不到这件事**不是因为它不准**,而是因为它把
λ_min 换成了先验地板——那正是简并住的地方。

## 四、三条站在「不存在」上的守卫,搬家而不是删除(**D37**)

- 两条 cross-check 的「不存在」断言改成**一致性**断言:两个包现在都有
  `extreme_eigenvalues`,同算子同模板同 key 同迭代数下**逐位相等**;再加一条
  「两者在梯度谱上都错、且错在同一个方向」——**同意一个错数字仍然是同意**,所以
  这条比原来的「不存在」强。
- 「不存在」真正代表的那条规则**直接钉住**:`tests/exact/test_condition_estimate.py::
  TestNoGuardReadsTheMeasuredRoute`,**AST 扫描**(不是文本扫描,所以 docstring 里
  的名字不算),**双向**:`extreme_eigenvalues` 只许被 `condition_estimate` 调用,
  `condition_estimate` 在包内**一处也不许**被调用;外加一条**自检**
  (`largest_eigenvalue` 确实有调用点),因为一个扫不到任何东西的扫描在空仓库上也绿。
- `docs/migration/conditioning.md` §5 整节重写:论证原样保留,结论改成
  「差异现在是一条**规则**而不是一处**缺席**」。

**为什么这算一条裁决(D37)而不是一次实现**:铁律说「改判据 = 新裁决项」。三条守卫
的判据从「缺席」变成「无守卫读它」,那是判据的改动,即使它是在执行 D15(a)。

## 五、变异集:6 条,**5 杀 1 必存**,两条幸存各带出一件该做的事

基线前后各一次绿。

| # | 变异 | 第一轮 | 修好后 | 说明 |
|---|---|---|---|---|
| X1 | 移位反向(`image − λ_max·leaf`) | **SURVIVED** | **SURVIVED(必然)** | 见 §五.1 |
| X2 | λ_min 直接返回 spread | KILLED(5) | | |
| X3 | 第二次迭代复用第一个 key | KILLED(1) | | 跨包一致那条 |
| X4 | 去掉 λ_min 的先验地板 | **SURVIVED** | **KILLED(1)** | 见 §五.2,**真洞** |
| X5 | `condition_estimate` 改返回那个**界** | KILLED(1) | | 简并那条 |
| X6 | `condition_bound` 开始读测得路线 | KILLED(2) | | AST 规则那两条 |

### 五.1 X1 **必须**幸存,而理由是一条没人断言过的不变量

`largest_eigenvalue` 用 `tree_norm` 归一化,所以它收敛到的是**最大模**;把算子取负
不改变任何模。实测 `diag(9,3,1)`:两个方向的 spread **都是 8,逐位相同**。

追到底的收获:那条不变量属于这个例程而**没有任何东西断言它**。补了一条,所以
下一个看到这条幸存的人不用重新发现;而一个把 `largest_eigenvalue` 改成对符号敏感
的实现(比如换成 Rayleigh 商)会以一个**有名字**的失败出现。

### 五.2 X4 是真洞,而**第一次修补没有杀掉它**——这一条值得写两遍

去掉 `jnp.maximum(smallest, floor)`。**第一轮幸存**,因为本文件所有 fixture 用的
迭代数都太大,地板从不生效。

**追下去发现的不是「地板用不上」,是一个负的条件数。** `iterations=1`、
`geomspace(1, 1e7, 50)`:测得 λ_min = **−2182035.5**,真值 1.0。机理值得写下来:
`largest` 自己也是一次迭代的估计、因而**偏小**,于是 `largest·I − M` **根本不是**
半正定的,它的最大模可以超过 `largest`,`largest − spread` 就成了负数。除进
`largest` 里,那是一个**符号错了**的条件数——有限、量级看着正常、纯粹是胡说。

**第一次修补写的是对着那个手造谱的断言,而变异在 `condition_estimate` 里**——所以
补上了守卫,X4 **照样幸存**。第二次改成对着仓库自己的一个块量:`collinear_pair`
在 `iterations=1` 时测得 λ_min = **−198.07**,先验地板 0.1111;两次迭代后落在
**0.11111**,恰好就是地板——对一个最弱方向只由先验持住的块,那是诚实的答案。
两半钉在**同一个真实块**上之后,X4 才被杀掉。

> **同一个形状,一批里出现了两次**(G10 的 W8 与本批的 X4):**守卫的 fixture
> 够不到它存在的那个条件**。第二次还多一层——修补本身也够不到,因为它写在了
> 另一个函数上。

## 六、铁律 4 四件套(按 G 项的形态)

| | 项 | 结果 |
|---|---|---|
| (i) | 该批测试全绿 | bayesmith **1350 passed** exit 0(249.7 s);本批新增 **21** 条(1329 → 1350)。e-RHINO **未动** |
| (ii) | 接缝变异红 | 6 条,**5 杀 1 必存**,两条幸存各追到底并各补一件事(§五);基线前后各一次绿 |
| (iii) | 旧实现删除、计数守卫刷新 | **没有旧实现可删**;rheplicant 的 `condition_estimate` 随 **Wave B** 退役 |
| (iv) | 文档实测数字重测 | CHANGELOG `Unreleased`;**D37** 入簿(登记簿到 **D7–D37**);G14 行回填;`conditioning.py` 模块 docstring 与 `docs/migration/conditioning.md` §5 重写 |

## 七、留给下一位

1. **P2 余项只剩 G9 全量**(vmap / log 空间 / Fisher 的复数面;另两项:`diagnose`
   仍拒绝复 latent、`exact.correct.log_weight` 仍在域里索引)。做完就到**收尾发布**
   (D13)。
2. **铁律 5**:`Unreleased` 现在有 **G2 + G10 + G12 + G14** 四项。
3. **Wave B 接线时的一件事,而它已经量掉了。** 本页第一版在这里写着两侧的默认
   迭代数「不同」——**是我没量就写的,而且错了**。实测:两边都是
   `POWER_ITERATIONS: int = 12`(rheplicant `core/conditioning.py:35`,bayesmith
   `exact/solve.py`)。所以接线时**默认值对得上**,不需要登记为有意的差异。
   仍然要记住的是 §二 那张表说的事:**这个数改变答案**,所以任何一侧改了它都是一次
   数字重测,不是一个调参。
   > 记在这里而不是悄悄改掉:一个没人核过的数字正是本程序反复付学费的形状,
   > 而这次它出现在一页**专门讲测量**的记录里。
4. **D23** 仍是唯一一条已登记、未裁决、无守卫的语义差。
