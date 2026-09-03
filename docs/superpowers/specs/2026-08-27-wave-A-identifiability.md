# 执行页 Wave A · 模块 1 — `identifiability` 切换

> **文档状态：`record`** · 已落地批次/审计/测量的历史记录，写作当天为真，非当前权威。索引见 docs/README.md。

> 计划:`2026-08-26-one-implementation.md` §五 Wave A / 铁律 1–5。
> 前一批次:`2026-08-27-wave-A-opening.md`。
> **日期**:2026-08-27 · **本页状态:`identifiability` 已切换。这是本程序第一个
> 真正被切掉的模块,也是第一次跑真正的跨仓接缝变异。**

## 〇、切了什么,留了什么

**离开 rheplicant 的**:雅可比、列归一化、SVD、秩判决——即那句「贝叶斯数值实现」。

**留在 rheplicant 的**(逐项都有理由,不是省事):

| 留守物 | 归类(终局形态) | 理由 |
|---|---|---|
| 签名 `(space, pipeline, state_template, *, names, at, rtol)` | 第 5 类薄包装 | 铁律 1;两侧签名有意不同(旧 spec §六 整节论证过不做重导出) |
| `IdentifiabilityReport` 及其三个方法 | 第 5 类**容器** | 它的方法抛 rheplicant 的异常类;字段布局是被读取的保持面。`FlatMatrix` 是同类先例 |
| `_resolve_names` / `_check_at` / `_check_differentiable` | 第 4 类探针/预验证 | 13 条被钉文案里的 7 条住这里。P1 §三 的原则:**凡图缝会抹掉证据的拒绝,住在 `to_graph` 之前** |
| `_in_float64()` | 机制 | D9 取 (a) 的执行机制 |
| `_flat_view` | **临时**保持面 | `sensitivity.py` 还在 import 它。铁律 1:最后一个消费者切换前不删——它随本波稍后的 `sensitivity` 一起走 |
| `DEFAULT_RANK_RTOL` | **改为 re-export** | 不留第二份(D16 先例)。值的全部论证——8.7 个量级的窗口、D9 的族扫描——住在用它的算术旁边 |

**分诊表**:45 条**全部原样重放**,一条未改、一条未退役。三列里另外两列**空**。
这不是运气,是铁律 1 的验收:签名、异常类、被钉文案、报告字段全都没动。

## 一、门面必须补三样、又必须不继承一样

`identifiability` 不带数据、不带噪声、不带先验——秩是**前向模型自己的**性质。
而 `to_graph` 三样都要,因为**图的节点就是它的分布**。

### 补的三样,以及为什么合法

合成 `observed=zeros`、`noise=HomoscedasticNoise(1.0)`、以及给**自由 latent**
(`prior=None`)合成先验。

**合法性只来自一件事:这三样都够不到答案。** bayesmith 从 `local_block` 取雅可比,
而那个函数的 docstring 逐字写着先验字段是「**deliberately empty**」;
`dense_operator` 微分的是观测节点的 `loc`,也就是预测。

**但 docstring 不是测量。** 这是整个委托赖以成立的假设,所以它被**测**了:
`TestTheSynthesisedGraph` 把图造三遍(数据换成 1e4、σ 换成 1e-3、先验宽度换成 1e6),
逐字段比较报告——`nullity`、`rank`、`n_data`、`singular_values`、`jacobian`、
`column_norms` 全部相等。并配一条**兄弟**测试断言基线本身非退化(`n_par > 1`、
`s[0] > 0`、谱不是常数),否则「三份全零报告」也能满足上面每一条。

### 不继承的那一样:声明了的 `joint_prior`

`to_graph` **拒绝**一个声明了 `joint_prior` 的空间,而且拒得对:图一个节点一个分布,
不声明它就等于**静默丢掉先验**,交回一个只是似然的后验。

**但那条拒绝对求解是对的,对秩测试是错的。** 继承它就是门面从一个它根本不用的层
里进口了一条约束——实测代价:`test_jeffreys_prior.py` **8 条**红,它们经
`JeffreysPrior.check_identified` 走到这里。

所以 `_without_joint_prior` 把它摘掉,合法性与上面同源(秩判决不读先验),
**并且这里有一个值得点名的循环**:`JeffreysPrior` 本身就是**由信息矩阵定义的**,
一个去咨询它的秩测试等于拿一个从模型导出的先验去问那个模型。

## 二、D21 落地:展开点是声明的 `init`

开工批实测出的那条(见 `2026-08-27-wave-A-opening.md` §三)在这里落地:
`values0 = {**space.initial_values(), **(at or {})}`,**显式传给 bayesmith**,
不吃它 `prior_environment` 的默认。

守卫两条,一条**必须**跟着另一条:

* `test_the_two_points_really_do_give_different_answers`——先证明这个 fixture
  **分得开**两个点(gain=0 处模型退化,gain=1 处可辨识)。没有它,下一条是空的。
* `test_the_default_point_is_the_init_not_the_prior_centre`——先验中心两边都是
  1.0,**只有 init 不同**,而判决跟着 init 走。

## 三、第一版错的两处,都是套件抓的不是评审抓的

1. **开 x64 上下文不够,还要 cast 摄入的数组。** `Latent.init` 是声明空间时造的,
   在块**外**,所以是 float32,雅可比也就是 float32——bayesmith **指名拒绝**,
   45 条一次全红。旧实现在**同一个位置**做过这次 cast(`jnp.asarray(...,
   dtype=jnp.float64)`),委托保留了上下文管理器却丢掉了 cast。
   **这正是 D9 对 (a) 的顾虑的实物**:「必须上下文内重建图并 cast 摄入数组」——
   而它**失败得响亮**,不是静默,这也正是 D9 记录页判定 (a) 可用的那条实测。
2. **自由的复 latent 需要 `ComplexNormal` 而不是 `Normal`。** 适配器指名拒绝这个
   替换,且拒得对:`ComplexNormal` 两半独立、各带 `scale**2`,拿实 `Normal` 的
   scale 当它读会**静默地把声明的方差翻倍**。这是 45 条里最后一条转绿的。

## 四、铁律 2:cross-check 同批退役,而守卫抓住了它

`tests/crosscheck/test_diagnose_identifiability.py` **删除**。四条断言全部对照
rheplicant,而 rheplicant 现在**委托给这里**——它们已经变成「本包与本包比」,
四条一次性变成**不会失败的测试**。

**其中唯一的独立 oracle 已有既有等价物**(铁律 2 的另一分支):四行表
(free/basis × tone on/off 的 `n_par`/`rank`/`nullity`)由
`tests/diagnose/test_identifiability.py::TestTheMotivatingCase::test_the_four_row_table`
本侧独立钉住。其余三条参照的是 rheplicant 自己的数字,随文件退役。

### 守卫红了,这是它该做的

`test_every_paged_module_actually_has_a_cross_check_test` 在文件删掉的一刻就红。
它写在切换阶段**之前**,断言的不变量是「每个有页的模块都有一个 cross-check」,
而铁律 2 让这条**故意变假**。

改法是加一个 `SWITCHED` 集合并**双向**读:

* 记在 `SWITCHED` 里却**还有** cross-check → 红(忘了退役);
* 不在 `SWITCHED` 里却**丢了** cross-check → 红(退役没记录)。

每一条 `SWITCHED` 条目必须写明退役记录在哪、以及它带的独立 oracle 去了哪。

> **这算不算「改判据」?** 判断是**不算**:铁律 2 早就写着 cross-check 随模块删除,
> 这只是让守卫编码一条已存在的法则,而不是新立一条。所以没上登记簿。
> **写下来是为了让下一位能不同意**——如果读成改判据,那它该走登记簿。

## 五、接缝变异:本程序第一次是**真的**跨仓

此前每一组变异都在**一个**仓里跑,因为 rheplicant 侧还没有消费者。现在有了。
**改 bayesmith,看 rheplicant 红。**

**7 条,全部击杀**,基线前后各一次绿(49 例)。

| # | 变异(在 bayesmith) | rheplicant 红 | 判决 |
|---|---|---|---|
| W1 | 秩的切点 `>` 改成 `>=` | 1 条 | KILLED |
| W2 | 雅可比不做列归一化 | **9** 条 | KILLED |
| W3 | 零列除以自己的零范数 | 2 条 | KILLED |
| W4 | 谱不补齐到 `n_par` | 1 条 | KILLED |
| W5 | SVD 永不索取完整左因子 | 4 条 | KILLED |
| W6 | 去掉图侧精度拒绝 | 2 条 | KILLED |
| W7 | 切换了却不记进 `SWITCHED`(在 rheplicant 侧的簿记守卫上,对 bayesmith 套件跑) | 1 条 | KILLED |

**W6 值得单说**:它证明门面对 D9 那条精度拒绝的依赖是**实的**——去掉 bayesmith
的守卫,rheplicant 的 `test_a_model_pinned_to_float32_is_refused` 立刻红。跨仓的
「谁承重」在这一行上是可读的。

附录 A 新增这七行。

## 六、铁律 4 四件套

| | 项 | 结果 |
|---|---|---|
| (i) | 该批测试全绿 | rheplicant **10061 passed / 522 skipped** exit 0(359.9 s,`-n 4 --ignore=tests/gui/e2e`)加 **21 passed** exit 0(`tests/gui/e2e -n 2`);bayesmith **1291 passed / 0 skipped** exit 0(208.0 s) |
| (ii) | 接缝变异红 | **7 条全杀**,§五;**第一次是真跨仓** |
| (iii) | 旧实现删除、计数守卫刷新 | rheplicant 的秩算术删除;cross-check 文件删除;rheplicant README 计数 10599 → **10603**(由守卫报数,不自己加);bayesmith README/CLAUDE.md 1295 → **1291**;crosscheck 123 → **119** |
| (iv) | 文档实测数字重测 | 上述全部;`docs/migration/identifiability.md` §5.2 已在开工批改写 |

**coverage floor 未动**(`fail_under = 89`):本批次删掉的是 rheplicant 的算术
**并同时**加了 4 条测试,覆盖率未跌破地板(全套 exit 0,而 floor 由 coverage 自己的
job 判)。

**bayesmith 地板仍是 `>=0.3`,这是有意的。** 本门面只用 `identifiability`、
`DEFAULT_RANK_RTOL`、`BayesmithError`、`ComplexNormal`——**四个在 0.3.0 里都有**。
0.4.0 已在索引上(发布门已开),但地板声明的是**真实需要**;它随第一个真正需要
0.4 表面的模块(`sensitivity` 要 D9 修好的守卫,`priors` 要 JeffreysPrior 的那条)
一起升。

## 七、留给下一位

1. **Wave A 还剩四个模块**:`sensitivity`、`priors`、`numpyro_bridge`、
   `uncertainty`。分诊总量见开工批 §四(265 条,本批已处置 45+4)。
2. **`sensitivity` 是下一个自然选择**,理由有二:它与 `identifiability` 共用
   `_flat_view` 与 `_in_float64`,所以切它才能把 `_flat_view` 真正删掉;而且它
   **需要 0.4** 的 `prior_sensitivity` 守卫修复,所以那一批同时升地板。
3. **`_graph_for_rank` 的三样合成 + 摘 `joint_prior`,`sensitivity` 大概率照抄。**
   若照抄,把它提成一个共用 helper——但**先看它的不变性是否同样成立**:
   `prior_sensitivity` **确实读先验**(它算的就是先验位移),所以合成先验对它
   **不是**中性的,`_flat_prior` 那条路对它无效。这是一条必须重新测量而不是继承的
   假设,写在这里以免被顺手复制。
4. **附录 A 需要把 W1–W7 补进去**(本页 §五 已是登记形式)。
