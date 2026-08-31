# 执行页 Wave B / `linear` 求解面 —— 十二条拒绝,一条会变成错答案

> **文档状态：`record`** · 已落地批次/审计/测量的历史记录，写作当天为真，非当前权威。索引见 docs/README.md。

> 计划:§五 Wave B / 铁律 1、3、4;新增 **D52**、**D53**(裁决)。
> 前一批次:`2026-08-28-wave-B-opening.md`。
> **日期**:2026-08-28 · **本页状态:四件套齐备。**
> 探针:`docs/probes/probe_17_linear_solve_seam.py`。

## 〇、这一批切了什么

`rheplicant.inference.linear` 的**四个公开求解名**——`wiener_solve`、
`gcr_sample`、`condition_bound`、`condition_estimate`——改为委托
`bayesmith.exact.solve` 的同名函数。**签名一个字没动**,所以碰这四个名字的
**29 个测试文件**里绝大多数原样重放。

删除的本地实现:`_conjugate_solve`、`_normal_operator`、`_condition_bound`、
`_condition_estimate`、`_real_parts`、`_domain_zero`、`_variance_parts`、
`_largest_variance`、`_split_like`(**2097 → 1911 行**)。新增的是一个转换器
(`_as_far_block`、`_from_far_domain`、`_far_precision`)和一条内部常量
`_OBSERVED`。`_domain_centre` **留守**——远端的 `domain_centre` 直接读字段,
而 `None → zeros` 与广播这一层是门面的活。

## 一、开工第一件事:逐条问每一条拒绝「过缝之后还到不到得了」

D48 立的规矩是「凡图缝会抹掉证据的拒绝,住在建图之前」。这一批把它作为**开工
第一件事**执行,而且**枚举是派生的**:从四个公开名沿调用图做传递闭包,数到
`linear.py` 里 **10 个 `raise` 站点**,外加 **2 个外部拒绝助手**
(`check_observed_shape`、`check_noise_std_axis`,住在兄弟模块里,grep 这个
文件是看不见的)。

**十二条全部被缝抹掉,而且原因是结构性的,不是谁漏了检查**:

```
近端:  wiener_solve(block, observed, *, noise_std, prior_std, prior_mean)
远端:  wiener_solve(block, *, precision)
```

`observed`/`noise_std`/`prior_std`/`prior_mean` 在近端是**参数**,在远端是
**block 的字段**——因为那边的 block 是从图上切下来的。所以凡是检查这四个参数
的拒绝,过缝之后**没有东西可检查**。

### 1.1 三种后果,只有一种是危险的

| | 远端的行为 | 结论 |
|---|---|---|
| (a) | 抛出等价的拒绝 | 可以委托 |
| (b) | 抛出,但指错层 | 留守;损失是消息 |
| (c) | **静默给出答案** | 留守;损失是**正确性** |

实测(probe_17 §1):**没有一条落在 (a)**。R1/R2(`NoiseModel`)、R4(复数
offset)、R5(observed 形状)在远端都是 (b) ——`TypeError` 从 trace 内部抛出。
其中 R1 值得单记一笔:`_refuse_a_noise_model_at_the_conjugate_seam` 的
docstring **预言过**远端的消息(「comes back as ``TypeError: Value
'HomoscedasticNoise(...)' with dtype object is not a valid JAX array type``」),
而实测**逐字就是那一句**。一条写了很久的推测,这次有了测量。

### 1.2 落在 (c) 的那一条:`check_noise_std_axis`

一个长度 `n` 的 sigma 对着 `(n, n)` 的预测,「每行一个」与「每列一个」读法
同样成立,NumPy 按尾轴对齐悄悄选一个。近端**拒绝**;远端收的是
`Precision`,歧义在构造它的时候就已经被解掉了。实测:

```
upstream, ambiguous       StateValidationError: ... whose axes [0, 1] are ALL of length 8
downstream, bare (n,)     NO REFUSAL -> [0.999999 1.142856 1.285713 1.42857 ]
downstream, explicit (n,1) NO REFUSAL -> [1.       1.142825 1.285702 1.428235]
两种读法差 2.5e-03,都有限、形状都对
```

**下游没有任何东西分得出这两个。** 这是本批唯一一条「不留守就变成错答案」的
拒绝,也是计划没点名的那一条(计划点的是
`_check_solve_arguments`、`_require_prior_std`、
`_refuse_a_noise_model_at_the_conjugate_seam` 三条)。

**但它不是本页发现的。** 写完探针之后才读到:跨仓 cross-check
`tests/crosscheck/test_linear.py::test_the_ambiguous_1d_sigma_is_resolved_before_bayesmith_can_see_it`
**早就记着这一条**,而且给的理由比探针深一层——不是「`Precision` 抹掉了
sigma」,而是 **numpyro 的 `promote_shapes` 在用户的 `dist_fn` 内部就把
`(8,)` 变成了 `(1, 8)`**,与一个明确无歧义的 `(1, 8)` 再也分不开,所以
「信息在到达本包之前就没了」,守卫**写不出来**而不是没写。

两件事都成立,而且是**两条不同的路**:那条记录讲的是**过图**的路,探针走的是
**直接递 `Precision`** 的路——门面走的正是后者,所以这一条在门面上也必须留守。
把它写下来是因为**重新推导出一件已经记着的事,和发现它,不是一回事**,而本
程序 §七.5 记的正是「一个事实两份记录,更新了一份」的税。这里是第三份,所以
两份都指向对方。

### 1.3 顺带量到:`_require_prior_std` 守的是**拼写**,不是条件数

它的消息说没有先验会让 `AᵀN⁻¹A` 可能奇异、「CG would return a finite,
arbitrary answer rather than fail」,然后**自己邀请**「Pass a large prior_std
for an effectively flat prior」。两半指的方向相反,所以两半都量了:

* 被邀请的那个平先验(`prior_std=1e8`)在**两侧都到得了求解**,没有任何拒绝。
* 而答案**不是任意的**:它是最小范数解,与 prior_mean 无关——CG 从零起步留在
  `range(Aᵀ)` 里。`prior_mean` 取 `(5,-5,0)` 或 `(0,0,7)`(都在零空间里),
  答案都是 `[1, 1, 0]`。

所以这条守卫是对的、该留守,但**它写的理由比代码宽**:这里**均值**是良定义的,
不良定义的是**宽度**,于是真正会给出胡话的出口是 `gcr_sample` 而不是
`wiener_solve`。措辞在重新拼写它的地方更正。

## 二、门面的形状:形状转换,不是重写数值

**前置条件先量,再谈别的**:手工转换的 block 与近端解出的后验均值
**逐比特相同**(probe_17 §0,`max abs difference: 0.0`)。没有这一条,底下每
一条拒绝比对都是在比两个不同的模型——这是本程序 §七.2 记的那个形状。

两侧 `LinearBlock` 的差异全是**位置**而不是**内容**:名字元组化、域字典化、
预测按观测节点键化、数据与先验从参数变成字段。所以一个 group 靠改标签转换,
一个单 latent block 靠包一层转换。

**一条转换期的缺陷,值得写下来**:`_domain_centre` 的**唯一**旧调用点是
`_conjugate_solve`,而那里 `prior_mean` 已经被 `_resolve_prior` 归一成
per-member 字典了。`condition_bound`/`condition_estimate` 没有先验均值要解
(条件数不依赖它),直接把裸 `None` 递进来,于是 group 路径上
`prior_mean[member]` 打在 `None` 上。**换调用点就换了前置条件**,而旧调用点
把前置条件维持得太好,以至于函数自己从没写过它需要什么。

## 三、分诊表 —— 五条红,三个原因

`tests/inference/` + `tests/core/test_basis.py` 全跑,五条红:

| 测试 | 原因 | 处置 |
|---|---|---|
| `test_linear_blocks` ×2 | 文案 pin `condition number` | **改写对适配器**(D52) |
| `test_linear_blocks` ×1 | 单 key 钉住 key 相关的拒绝 | **改写对适配器**(D53) |
| `test_conjugate_transition` ×1 | 同上 | **改写对适配器**(D53) |
| `test_gls` ×1 | 钉住不动点的**退出方式** | **改写对适配器**(D53) |

外加 `test_degenerate_partition` 的 12 条 + `test_linear_groups` 的 3 条,
那些是上面那条转换期缺陷,**修的是实现不是测试**。

三个原因逐条见 D52/D53。这里只记**方法**:每一条都先问「这个测试到底在主张
什么」,再问「它现在还能不能因为那个主张而失败」。第二问抓到的东西比第一问多。

## 四、「大效应」不等于缺陷 —— 本页最贵的一段

`test_the_conjugate_convergence_guard_still_raises_equinox` 的守卫在切换前
**每一个**容差都触发、切换后**一个都不触发**(1e-8 到 1e-14 全试过)。按
`ci-flat-chain.md` §三 这是**大效应**,值得先查——查下去是:

```
maxiter=1 时  近端 residual=3.35e-07 × bound=4.69e+07 = 15.7  -> 守卫开火
             远端 residual=0.0                              -> 守卫沉默
```

这个 block 只有**一个参数**。CG 在 1 维系统上**按构造一步收敛**,所以
「`maxiter=1` 因此不收敛」从来就不成立;近端只是因为多一次舍入而没落到不动点
上,守卫是**对着一个正确答案开火**。这正是交接页 §三.2 本会话刚补的那条兄弟
命题,第一次被独立撞上。

**而真正的发现在再下一层。** 把它扫 20 个 key:

| | 切换前 | 切换后 |
|---|---|---|
| `test_float32_is_refused_however_tight_the_tolerance` | **15/20** 被拒 | **12/20** |
| `test_the_conjugate_convergence_guard_still_raises_equinox` | **14/20** | **10/20** |

**这两条拒绝一直是 key 相关的**,而两条测试都用**一个 key** 把它钉成了性质。
`key(0)`/`key(1)` 只是换了边。所以这不是本次切换制造的脆弱,是本次切换
**照出来的**——而且它属于交接页 §三.1 那个「守卫够不到它要守的条件」家族的
一个新变种:守卫够得到,但**够不够得到是随机的,而测试假装它不是**。

两条都改成断言与 key 无关的东西:地板本身(κ·eps = 3.81 > 1e-3,两侧相同)、
**每一条**拒绝的**种类**(实测两侧都是 100% 的 "at this precision"),外加
一句「至少有一个 key 被拒」的自检——那一句就是「这条守卫还能失败吗」写成了
断言。

## 五、残差不是误差,这次量出来了

D53 的第 1 条附带一个值得单独讲的测量。σ 是数组时两侧的均值差 ~1 ulp,拿
float64 稠密解做基准:

| | 相对误差 | 报告的残差 |
|---|---|---|
| 近端(旧) | 1.57e-07 | **7.02e-08** |
| 远端(新) | **1.12e-07** | 1.78e-07 |

**报告残差更小的那一个,离真解更远。** 两包的 docstring 都写着「residual is
not the error」,而这是它第一次以一个具体的数对出现在本程序的记录里。

## 六、计数守卫与文档(铁律 4 第三、四件)

* **拒绝普查 253 → 252**,`RuntimeError` **4 → 3**。少掉的是一个
  `pytest.raises` 站点(改成了 20-key 扫描),**不是一条断言**;类没有丢,
  另外三个文件还钉着 `RuntimeError`。`CENSUS`、`BY_CLASS`、总数 pin 与
  **附录 B** 同批刷新,附录 B 的 `test_linear_blocks.py` 一节由 `_sites()`
  **重新生成**而不是手改。
* **顺带清掉一批过期引用**,全部是「一个名字在两个地方,其中一份过期了」:
  `_condition_number` 被 **5 处**引用,而它在本批之前**就已经不存在**了
  (早先改名为 `_condition_bound`);`_conjugate_solve`、`_condition_bound`
  各 3 处。全部改为指向现在承载那个意思的**公开名**或远端函数。
  `conjugate_support.py` 里两处**行号引用**改为**函数名引用**——其中一处写着
  `linear.py:963-973`,而那个函数在 1321,已经搬过两次。

## 七、接缝变异集(铁律 4 第二件)—— 12 条,11 杀

靶子是转换器的每一条分支加求解面的四条拒绝。基线 exit **0**,每一轮清
`__pycache__`(**只扫 `src/rheplicant`,不扫 `.venv`**),恢复用
`git checkout --` **只针对被变异的那一个文件**,**只有退出码 1 记为击杀**。

| | 变异 | 结果 |
|---|---|---|
| M1 | `data` 完全忽略 `observed` | KILLED `test_matches_a_dense_solve` |
| M2 | 去掉 `observed=None` 的处理 | **SURVIVED —— 等价变异**,见下 |
| M3 | 去掉 group 的 `prior_mean=None` 归一 | KILLED `test_the_condition_estimate_matches_a_dense_eigenvalue_computation` |
| M4 | group 的 `prior_std` 改读 `prior_mean` | KILLED `test_a_group_of_ONE_is_legal_and_answers_in_a_dict` |
| M5 | 单 latent 的 `forward` 忽略名字 | KILLED `test_the_default_tolerance_is_refused_rather_than_trusted` |
| M6 | `_from_far_domain` 对裸 block 也返回字典 | KILLED `test_different_keys_give_different_draws` |
| M7 | `_from_far_domain` 两条分支对调 | KILLED 同上 |
| M8 | `_far_precision` 递方差而不是 sigma | KILLED `test_recovers_a_noiseless_signal_under_a_weak_prior` |
| M9 | `condition_bound` 去掉 `_require_prior_std` | **SURVIVED → 真缺口,已补** |
| M10 | `condition_estimate` 去掉 `check_noise_std_axis` | KILLED `test_the_ambiguous_vector_is_refused_and_names_this_exit` |
| M11 | 去掉 `NoiseModel` 拒绝 | KILLED `test_a_prediction_dependent_model_gets_the_longer_refusal` |
| M12 | 去掉复数 offset 拒绝 | KILLED(见下) |

**三条幸存,三种成因,而这一次三种都不是「补个 fixture」。**

* **M12 根本不是幸存**,是**我的靶子集选错了**。杀它的测试
  (`test_inference_unpinned_refusals.py::test_a_complex_offset_is_refused_and_names_the_exit`)
  一直存在,只是不在我列的九个文件里。**这是 `CLAUDE.md` 记的那个陷阱的镜像**
  ——那一条讲「变异被三个函数之外的守卫先杀了,你写的那条根本没被评估」,
  这一条讲「**测试在,但你没跑它**」,而输出长得一模一样:都是一行 SURVIVED。
  把靶子集扩到拒绝普查涉及的文件之后立刻击杀。
* **M9 是真缺口。** 删掉 `condition_bound` 的 `_require_prior_std`,**整个定向
  套件仍然全绿**。它不是静默错答案——`prior_std=None` 之后仍会在
  `jnp.asarray(None) ** 2` 处炸——所以**没被守住的是消息**,而消息正是这条
  拒绝存在而不是交给数组层的**唯一**理由。补了
  `TestBothConditioningExitsRunTheSamePreconditions`,两个出口各一条,并各自
  钉住消息**点自己的名**。**复查过红是不是自己那一条**:重跑 M9,击杀者正是
  `test_a_missing_prior_is_refused_by_name[condition_bound]`。
* **M2 是等价变异,而且是**能被验证**的那种。** 门面在没有 `observed` 时递一个
  预测形状的零;换成裸 `None` 之后一切照绿——**正确地**,因为 bayesmith 的
  整条条件数路径**从不读 `block.data`**(逐个走过 `condition_bound`、
  `condition_estimate`、`_condition_bound`、`normal_operator`,四个都不碰)。
  所以为它编一个 fixture 会钉一个 mock。**改为钉住让这条分支有意义的性质**:
  条件数不依赖数据。远端哪天开始读 `data`,它就红。

计数守卫因此第二次移动:拒绝普查 **252 → 254**(`ParameterSpaceError`
180 → 182),`CENSUS`、`BY_CLASS`、总数 pin 与附录 B 同批再刷一次。

## 八、还欠的(下一位接手处)

1. **`gls` 是下一批**,它的 `_check_solve_arguments` 借用随之到期;跨仓
   cross-check `test_noise_logdet.py` 的 17 条按铁律 2 逐条改籍或指认。
2. **cross-check 已按铁律 2 处置**:14 条退役 8 条,文件**保留**,
   `linear.py` **不进 `SWITCHED`**(只切了求解面)。退役的每一条都在
   `tests/exact/test_solve.py` 里**指认**了既有的家,而不是改籍重写。
   **退役的触发者是它自己**:`test_the_gcr_draws_reproduce_the_oracle_mean_and_covariance`
   自带的反空转守卫(「the draws came out bitwise identical, which means the
   two packages are sharing a key stream」)就是切换当天变红的那一条——一条
   cross-check 能**自己报告自己过时了**,这是本程序第一次见到。
