# 执行页 Wave B 开波 —— 普查、契约、以及契约本身的两处过期

> **文档状态：`record`** · 已落地批次/审计/测量的历史记录，写作当天为真，非当前权威。索引见 docs/README.md。

> 计划:§五 Wave B / 铁律 1、4、7;新增 **D51**(裁决)。
> 前一批次:`2026-08-28-ci-triage.md`。
> **日期**:2026-08-28 · **本页状态:开波仪式三件已做两件半。**
> 模块尚未切换——`linear` 求解面是下一批,理由见 §五。

## 〇、开波仪式是三件事,而第二件这次有产出

计划写着:**每波开工做铁律 1 私名普查 + 铁律 7 契约阅读 + 测试分诊表**。

前两件通常是读。这次第二件**读出了两处过期**,而且两处都是 Wave B 被要求据以
行动的行。

## 一、铁律 1 私名普查

`grep -rn 'from rheplicant.inference.<mod> import' src/`,六个模块逐个:

| 私名 | 消费者 | 归宿 |
|---|---|---|
| `linear._magnitude` | `engines.py` | **留守**(交接页 §八 已定) |
| `linear._gaussian_parameters` | `sensitivity.py`、`uncertainty.py` | 至两者全切 |
| `linear._numpyro_distributions` | `uncertainty.py` | **本会话新增**,见下 |
| `linear._resolve_one_prior` | `graph_bridge.py` | 适配器自身,近乎永久 |
| `linear._check_solve_arguments` | `gls.py` | 与 `gls` 同波 |
| `linear.{_affinity_errors,_group_probe,_isolate,_isolate_group,_require_inexact,_resolve_names,_single_probe}` | `loglinear.py` | **七个探针助手**,铁律 1 已点名:D17 落定前不删 |

**普查抓到了本会话自己制造的一条。** G15 的解除让
`uncertainty._declared_gaussian_priors` 从 `linear` 借了
`_numpyro_distributions`(规范形要一个 `Normal` 构造器)。按铁律 1,这个名字
从此是保持面,直到 `uncertainty` 的最后一个消费者切换。**写在这里而不是留给
Wave B 的 `linear` 批次自己发现**,因为一条上一批新增的依赖,正是下一批最不会
去找的东西。

`noise` 被 **12 个模块**导入,是消费面最宽的一个;`likelihood.check_observed_shape`
被 6 个。两者的「工厂化」因此是本波风险最高的改动面,尽管它们的行数最少。

## 二、铁律 7 契约阅读 —— 两处过期,一处承重

### 2.1 `linear.md` §5(a):一条已经不成立的「有意差异」

它开头写着 **「`condition_estimate` is not ported」**、**「bayesmith has
`condition_bound` only」**。

实测:`bayesmith/src/bayesmith/exact/solve.py:204` 就是 `condition_estimate`。
**G14 于 2026-08-27 落地**,提前于 Wave B,而 `conditioning.md` 的
「What is new」一节**已经**为它更新过——`linear.md` 没有。

于是 Wave B 关于**台账里最大的那个模块**读到的第一件事,是一个早已从「有意
差异」变成「带警告的一致」的行。第一句之后的内容全部仍然成立(偏差是真的,
方向是**认证**而非拒绝),变的是这一波欠的比较从「有没有」变成了**逐值**。

### 2.2 `plan.md` §5(a) 与两条守卫:分支依赖,而它是那个「结构性盲点」的样本

它写着 rheplicant 的 docstring 更正在 `track-a-tail` 上、**未合并未推送**,所以
checkout `main` 拿不到,并说**守卫可能因为「不是任何人的缺陷」而变红**。

**按远端实测**(先 `git ls-remote` 取尖端,再
`merge-base --is-ancestor 7f03af1 origin/main` 与
`git show origin/main:src/rheplicant/inference/plan.py`):`7f03af1` **是**
`origin/main` 的祖先,句子就在远端自己的文件里,`track-a-tail` 已不存在。
Seam CI checkout 的正是 `rheplicant@main`,所以它读得到。

**这一条不是文书工作。** 依赖成立的那段时间里,那两条守卫是绿的,而绿的原因
是**可编辑安装恰好 checkout 在哪个分支上**;对 docstring 文本做任何变异都不可能
把它照出来,因为**那个 ref 从来不在变量集里**。rheplicant 的 `CLAUDE.md` 把它记作
变异测试**唯一的结构性盲点**,而这两条守卫就是它的样本。

**实例已闭,教训未变。** 三处记录同批更正(两份契约页、两条守卫自己的
docstring 与失败消息),并在 `CLAUDE.md`/`AGENTS.md` 里补了一句方法:
**问一条绿守卫依赖着什么你从未变动过的东西,并且用「量远端」来回答它**
——因为本地的 `origin/main` 只是一个「上次 fetch 时正确」的文件。

> 两处都以**带日期的更正**写入,而不是重写。契约页与执行页不同:契约要当下为真,
> 所以必须改;但**改成什么**与**原来是什么**同样有用,而且这两处过期的形状
> 恰好是本程序反复付学费的那一个——一个事实两份记录,更新了一份。

## 三、分诊表(种子已铺,未完)

计划给了种子:`test_plan_compiles_once.py`、`test_conjugate_transition.py`、
`test_block_learning_rate.py`、`test_magnitude_is_build_time.py`、
`test_inference_construction_guards.py` 部分行。

本次做完的是**其中被点名要「显式重谈」的三条**(D51),因为它们是分诊表里
唯一**现在就必须裁决**的格子——其余各行的处置取决于模块怎么切,写在切换之前
是预测而不是分诊。

三条不是同一件事,逐条见 D51。要点:

* 一条是**现在就该修的缺陷,与 Wave B 无关**:`MIN_SWEEPS`/`DEFAULT_MAX_ITER`
  这两个值在整个仓库里**仅有的 pin** 住在 config 层。已搬到
  `tests/inference/test_plan.py`,并顺带把 `MIN_DRAWS` 钉成**派生**的
  (经包自己的 `_halves`:「两半各二」)而不是字面 4。
* 一条**本来就是派生**,原样重放。
* 一条是**真正的源文本 pin**,留到 `plan` 切换那批,**解除条件与替代形态写死在
  登记簿里**——行为等价扫描,外加一句「加一个模式就要加一个候选」。

## 四、开波顺带量到的两件小事

* `MIN_SWEEPS` 的注释说它是 `MIN_REWEIGHTS` 的「a third the count」,两者是
  **3 对 5**。改成陈述成立的关系而不是换一个比例数字——比例本来就不是设计。
* `gls` 的公开面被 **12 个测试文件**碰到,而 `tests/inference/test_gls.py` 只有
  **20 条**;跨仓 cross-check `test_noise_logdet.py` 有 **17 条**且与 Fisher 行
  共享。切 `gls` 时那 17 条按铁律 2 逐条改籍或指认,不能随文件一起消失。

## 五、下一批是 `linear` 求解面,以及为什么不是 `gls`

`gls` 行数最少(268)、消费者最少(只有 `inference/__init__.py`),看着是最便宜的
开头。**不是。** 它的内层解就是 `wiener_solve`:把 `gls` 切到
`bayesmith.exact.gls` 而 `linear` 未切,会让同一个包里存在**两个求解器**——
门面的 `wiener_solve` 是 rheplicant 的,而 `gls` 内部用的是远端的。计划把
「`linear` 求解面」写在 `gls` 前面,量过之后这个顺序是对的。

两侧 `LinearBlock` 的**形状不同**,这是 `linear` 那一批的核心工作:

| rheplicant | bayesmith |
|---|---|
| `name: str \| tuple` | `names: tuple` |
| `shape: tuple \| dict` | `shape: dict` |
| `offset: Array` | `offset: dict` |
| `forward: x -> A x` | `forward: {name: x} -> {obs: A x}` |
| `prior: Any = None` | `prior_mean` + `prior_std` |
| —— | `data: {obs: value}` |

即上游是**单观测节点、数组形**,远端是**多名字多节点、字典形**。计划里
「适配器把两层降一层」指的就是这里。四个公开求解名两侧同名
(`wiener_solve`、`gcr_sample`、`condition_bound`、`condition_estimate`),
所以门面是**形状转换 + 拒绝前置**,不是重写数值。

**拒绝前置**这一条现在有 D48 的先例了:凡图缝会抹掉证据的拒绝,住在建图之前。
`linear` 的求解面拒绝很多(`_check_solve_arguments`、`_require_prior_std`、
`_refuse_a_noise_model_at_the_conjugate_seam`……),开工第一件事是逐条问它们
**过缝之后还到不到得了**。
