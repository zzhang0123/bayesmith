> **历史状态：** 产品定位、公共概念模型与长期路线图已由
> [`2026-08-30-bayesmith-top-level-design.md`](2026-08-30-bayesmith-top-level-design.md)
> 取代；本文仅保留为测量记录与历史审计。
>
> **测量页 — bayesmith 架构叙事审计**。本页是**评估**,不是一个实现批次。
> **日期**:2026-08-28 · 起因是 owner:「整体 check 一下 bayesmith 的架构叙事,
> 把 evidence 模块改名也一起考虑一下。」
>
> **方法**:四阶段多智能体(5 份分片测绘,含**用 AST 重建的依赖图** → 5 条横切
> 分析 → **对抗评审**,默认立场是「公开包的改名不值得,让提案自己挣」,且要求
> **去数**真实成本 → 报告)。**14 条建议里对抗者只放行 5 条。**
>
> **本页与 `2026-08-28-multi-dataset-joint-posterior.md` §0.3 在改名这一点上
> 不一致**,而这一点没有被抹平:那一页建议**不改名**(成本高、收益小),本页
> 倾向改名到 `marginal/`(信心约 0.6),并把那条裁决的两侧都量了出来
> ——成本侧 106 处 / 29 文件 / 两次目录移动,收益侧**今天为零**(下游 0 import)
> 但 **P6 之后不为零**,且窗口在 0.5.0 upload 关闭。**这是 owner 的决定。**
>
> **本页只改了两处 `src/`**(R1 与 R3,都非破坏性且已实跑核实),其余全部是登记。

---

HEAD 在审计期间前进了一步（`ffb46d5`，比卷宗里的 `e854a1f` 新一个提交），这一步恰好与本次问题直接相关；下文第 6 节会说明。以下所有数字都是我本次在 `/Users/zzhang/projects/bayesmith`（`ffb46d5`，工作树干净）与 `/Users/zzhang/projects/e-RHINO` 上跑命令得到的。**没有修改任何被跟踪的文件。**

---

# 1. 结论先行

叙事**基本成立**：模块级依赖图是干净的无环 DAG（我自己用 AST 重建：`graph` 入度 6、`errors` 入度 7、`dispatch` 入度 0 出度 5，Tarjan 无多节点 SCC），这正是顶层 docstring 与 `pyproject.toml:8` 宣称的"图的结构选择推断方法"。这一半不需要修，只需要写下来——因为它**目前没有写在任何读者够得着的地方**。

最大的问题不是 `evidence/` 这个名字，而是**三个公开名字在裸 `import bayesmith` 之后抛 AttributeError，且是顺序相关的**：`bayesmith.optimize`、`bayesmith.amortize`、`bayesmith.distributions`。这是 `__init__.py:243-249` 自己叙述过、为 `evidence` 修过一次的同一个缺陷，守卫（`tests/test_public_api.py:297`、`:371` 的逐名白名单）在结构上不可能发现它。这是**唯一的 blocking 缺陷**，修复是纯增量的。

关于 `evidence/`：**名字确实过度承诺，我倾向改名，但这是一个接近的判断（信心约 0.6），而不是一个显然的结论**——而且必须先说：**你自己在今天的 `ffb46d5` 里已经就同一问题裁决过，结论是不改名**。我在第 6 节把两边的账都算出来，包括那条裁决的成本侧（可测：106 处 / 29 个文件 / 两个目录）与收益侧（今天为零，P6 切换之后不为零），并给出一个**实测可用的**弃用垫片。如果你维持原裁决，那条裁决是可接受的，但需要补两件事才够（见 6.6）。

---

# 2. 叙事是否成立

核心主张（`pyproject.toml:8`）：*"A graph of operators is a Bayesian model; its structure chooses the inference"*。

**代码兑现了它。** 我的 AST 遍历（44 个 `.py`）给出的模块级子包边：

```
amortize -> errors
graph    -> errors
optimize -> errors, graph
bridge   -> distributions, graph
exact    -> bridge, distributions, errors, graph
evidence -> errors, exact, graph
diagnose -> errors, exact, graph
dispatch -> bridge, errors, evidence, exact, graph
```

无环。分层是真的：`errors`/`distributions` 在底，`graph` 之上是 `exact`，`evidence`/`diagnose` 是叶子应用（`evidence` 入度 1、出度 3；`diagnose` 入度 0、出度 3），`dispatch` 在顶且没有人依赖它。`dispatch/classify.py:9`（"Nothing here samples, and nothing here is jittable"）这类不变量是真的、被遵守的、且自解释。`README.md:59` 写的 1523 个测试，我 `--collect-only` 数出来正好 **1523**，而且有守卫（`tests/test_readme_count.py:64`）。

**但这个分层在任何用户够得着的地方都没有写下来。** `README.md` 的五个标题是 `bayesmith / What bayesmith is not / Worked examples / Status / License`——没有一条是架构。四份架构性文档全部在 `docs/superpowers/`（本仓库跟踪，45 个文件），而 `pyproject.toml` 的 sdist `include` 只列 `docs/migration`，wheel 是 `packages = ["src/bayesmith"]`，都不发。且它们是中文的。

**三处 DEFECT，叙事说了假话：**

| 位置 | 声明 | 实际 |
|---|---|---|
| `evidence/__init__.py:3` | "Nothing here reaches the graph yet." | `campaign.py:43,44` 和 `factorize.py:64` 在模块作用域 import `bayesmith.graph`；`dispatch/plan.py:842` 对**每个** graph 调 `streaming_route(graph)`，后者在 `dispatch/streaming.py:40,41` import 本层 |
| `exact/gibbs.py:19-24` | "`exact` never reads `dispatch` … a layering decision, not an oversight" | `exact/fisher.py:717` 就是 `from bayesmith.dispatch.classify import prior_environment`，函数作用域、**无注释**。无任何测试守卫这条规则 |
| `README.md:32-33` | diagnose/ 拥有 "identifiability, prior sensitivity, and **linearity checking**" | `check_linearity` 在 `exact/linearity.py:641`；`README` 提到 `diagnose`/`optimize`/`amortize` 的次数各为 **0**（`grep -oic`） |

`README.md:49` 写 `0.4.0`，`pyproject.toml:7` 是 `0.5.0`；`README.md:50-51` 说下游"in two places"，实测 e-RHINO 有 **9 个生产模块** import bayesmith；`README.md:55` 说"the 0.3 floor"，e-RHINO `pyproject.toml:73` 钉的是 `bayesmith>=0.4`。**测试数有守卫，版本号没有。**

---

# 3. 词汇审计

按"读者被误导的概率"排序。**每一条都标注 DEFECT 还是 TASTE。**

### 3.1 `evidence` —— DEFECT，四义（子包内）+ 一义（子包外），且贝叶斯义为零

`grep -rniE '\bevidence\b' src/bayesmith --include='*.py'`：**104 处，子包内 52、子包外 52**。

子包**外**，剔除模块路径引用后剩 **15 行**，全部是普通英文的"依据/佐证"义：

```
errors.py:9              "carries its evidence as ATTRIBUTES"
dispatch/plan.py:1       "a partition, its evidence, and how to run it"
dispatch/plan.py:579,582 "its evidence indented under it"
dispatch/classify.py:78  "the evidence behind it"
dispatch/execute.py:413  "r-hat's verdict is not evidence"
exact/gaussian.py:25,252,255,302,311  "the type is evidence and not proof"
exact/linearity.py:254,281
exact/gls.py:382         "evidence the movement is at the edge of rtol"
optimize.py:118          "evidence a caller has"
```

**贝叶斯 `p(d)` 义在子包外出现 0 次。** `grep -rniE 'bayes_factor|model_comparison|log_evidence|marginal_likelihood' src/` 只命中 `exact/correct.py:196`——一个关于 SNIS 权重常数的**测试名**。种子观察成立。

子包**内**四义：

| 义 | 位置 |
|---|---|
| (a) 层的名字 | `evidence/__init__.py:1` |
| (b) `p(d)` / 归一化常数 | `compress.py:94`、`:158`、`factorize.py:46`、`:270`、`chain.py:258` |
| **(c) 一个 epoch 的单个似然因子** | **`chain.py:272` "Add one epoch's evidence to the running joint form."** |
| (d) 取证义（呈堂证供） | `sqrtinfo.py:222`、`diagnostics.py:188` |

(c) 是最直接的矛盾：单个似然因子恰恰是"evidence"**不**指的东西。

**两个内部证人说明作者已经知道更好的词：**
- `dispatch/plan.py:1` "a partition, its **evidence**, and how to run it" 对 `dispatch/plan.py:509`（这个模块为之存在的那个类的 docstring）"a partition, its **reasons**, and how to run it"。同一句话，两个词，相隔 508 行。而 `reason` 是 `Block` 上的真实字段名（`plan.py:343`）。
- `README.md:30-31` 自己把这条 bullet 注释成 "**Streaming evidence** — square-root information **factors** combined exactly"。解释的本能伸手就抓 factor。

### 3.2 `block` —— DEFECT，四义，其中三个是公开拼写

793 处（`grep -rniw 'blocks\?'`），是包里最高频的名词。
- `exact/block.py:47` `class LinearBlock` —— 仿射映射
- `dispatch/plan.py:336` `class Block` —— 计划的一行
- `exact/fisher.py:68` `FlatMatrix.block()` —— 子矩阵
- `diagnose/priors.py` —— "a named block of latents"（一组名字）

顺带一个真缺陷：`dispatch/plan.py:341-342` 声明 `method` 是 `"gcr"/"gcr+snis"/"gcr+mh"/"nuts"` 四值、"as `Classification` chose it"；`dispatch/factor.py:391` 构造 `Block(method="log-gcr")`，而 `factor_partition` 根本不经过 `Classification`。`_LABELS`（`plan.py:119-124`）4 行、`FACTOR_METHODS`（`factor.py:81`）3 行，并集 5，无一处枚举全部。今天不崩（`plan.py:594` 是 `_LABELS.get(m, m)`），但文档是错的，而 e-RHINO 的 `tests/seam/test_p1_ten_examples.py` 就是手工构造 `Block` 的调用方。

### 3.3 `operator` —— DEFECT，四义，其中一义是包自己的主题句

112 处。`__init__.py:1` 与 `README.md:3` 的 "a graph of **operators** is a Bayesian model" 指图节点；而全部五个 `*_operator` 公开函数指线性映射：`exact/linearity.py:796`、`exact/block.py:443`、`exact/solve.py:63`、`exact/loglinear.py:496`、`exact/fisher.py:162`。**主题句用的那个义在代码里再也没出现过。** 但 `linear_operator`/`log_linear_operator` 在顶层 `__all__`（91 名）里，改代码是破坏性的且无正确性收益——所以这条**只应改散文**。

### 3.4 `factor` —— DEFECT，三义，且这是 `factors/` 作为改名候选被封死的原因

126 处。(i) 乘性标量（`gls.py` 里就有一个 `factor: float` 参数）；(ii) 一个 Gibbs 块（`FactorPlan`、`FACTOR_METHODS`、`factor_partition`、`sample_factors`、`estimate_factors`，其中后三者在顶层 `__all__`）；(iii) 密度因子（`evidence/factorize.py`）。

### 3.5 `information` —— **不是缺陷，是本包应该照抄的范本**

`SqrtInfo.information()` 的 docstring 明说：*named `information` rather than `fisher` because it is the observed information of a stored term, not a Fisher matrix of a model — `exact/fisher.py` owns that word here.* 这是包里唯一一处**显式协商词界**的地方。`evidence` 这个目录名没有做同样的事。

### 3.6 TASTE，不建议动

- `Graph.joint_prior` —— `graph/graph.py:34` 把它精确定义为"a density over SEVERAL latents at once"，用法一致。名字**欠具体**而非歧义。（顺带：`ffb46d5` 的新规范 §0.1 已经明确禁止重用这个词，这条已经被你自己处理了。）
- `partition` —— 我读了 `exact/solve.py:231,234` 与 `exact/conditioning.py:32,106` 四处，它们指的就是"一起解的那组 latent"，与 `factor_partition` 同义，**没有第三义**。卷宗里"把它换成 degeneracy"的提案是错的：`diagnose/identifiability.py:325` 正在同一句里把 partition 和 degenerate 当两个概念用。真正的同音异义是 `eqx.partition`（`amortize.py:404`、`graph/nodes.py:46`），那是 Equinox 的东西，无解也无需解。

---

# 4. 分层与归属

**布局在中间说真话，在两端说假话。**

**真话**：`graph → exact → dispatch` 这条脊柱与实测依赖图一致，且各模块 docstring 自己论证得比多数已发布包都好。

**归属错误（DEFECT）：**

1. **三个顶层 `.py` 模块不在 `_LAZY_SUBMODULES`（`__init__.py:251`）里。** 逐个新进程实测：

   ```
   graph exact dispatch diagnose evidence bridge errors -> OK
   optimize amortize distributions                      -> AttributeError
   ```
   且顺序相关：`bayesmith.fit` 之后 `bayesmith.optimize` 就有了；`bayesmith.exact` 之后 `bayesmith.distributions` 就有了。守卫是逐名白名单（`tests/test_public_api.py:297` `assert "evidence" in ...`、`:371` 同形），每修一个 bug 长一行，**结构上不可能覆盖没人加过的名字**。

2. **门面缺口 40/81。** 我从 `_LAZY_ATTRS` 反推归属：顶层 81 个由子包提供的名字里，**40 个在自己的子包上取不到**——`graph` 17/17、`bridge` 4/4、`dispatch` 7/9（**包括 `bayesmith.dispatch.compile`，那是头号入口**）、`evidence` 6/14（就是 `chain.py` 的六个）、`exact` 6/23。这一点很重要：**`evidence.chain` 的"孤儿"不是孤例，而是全包惯例**，所以单独给 `evidence` 补齐反而会让叙事更不一致（详见第 7 节 R7 的裁决）。

3. **`diagnose/priors.py::JeffreysPrior` 是建模构件放在诊断包里（DEFECT，但不建议动）。** 它的全部消费者都在它**下面**：`graph/trace.py:297`、`graph/graph.py:49`、`:179`（错误消息里硬编码类名，因为 `joint_prior: Any`，`graph/graph.py:55`）。`diagnose/` 内部无人用它。但 e-RHINO 生产代码钉了 `bayesmith.diagnose.identifiability`（`identifiability.py:93,94`）和 `bayesmith.diagnose.local`（`uncertainty.py:449`），改这个包是真破坏。见第 8 节。

4. **`diagnose/local.py`（409 行）在任何 `__all__` 里都不存在**（`bayesmith.diagnose.__all__` 是 7 个名字，`local_block` 在 `bayesmith` 顶层也取不到），却是 e-RHINO **生产代码**的 import 目标（`src/rheplicant/inference/uncertainty.py:449`）。它承载的是一个**没有名字的概念**：graph + 一个点 → `LinearBlock`，两个构造器分居两个子包（`exact/block.py:443` 在零点取切；`diagnose/local.py:230` 在调用方给的点取切）。

**TASTE，不建议动**：`bridge/` 只有 360 行 2 个文件却是子包，`optimize.py` 437 行却不是——这个区分编码的是到达顺序而非类别。但一个单文件包本身也是谎言。**说清规则，不要重排目录。**

---

# 5. 私有语境泄漏到公开包

**先更正种子**：这些代码**不**在另一个仓库里。`git ls-files docs/superpowers | wc -l` = **45**，就在 bayesmith 里。结论反而更强，因为原因换成了两条本仓库自己写下的：

- `pyproject.toml` 的 sdist `include` 只有 `src`、`tests`、`docs/migration`、README、LICENSE、CHANGELOG、pyproject，注释明说 `docs/` 是 *"internal working material … **not user documentation**"*；wheel 是 `packages = ["src/bayesmith"]`，**一份文档都不发**。
- 键值是中文的：`B11` → `docs/superpowers/specs/2026-08-24-rheplicant-migration.md:311` `### B11 — 流式证据，从图重写`。

**即：构建配置裁定这些规范不是用户文档，然后 44 个源文件中的 35 个引用它们。**

**实测计数**（`grep -rhoE '\b(B[0-9]{1,2}|G[0-9]{1,2}|P[0-9]{1,2}[a-z]?|D[0-9]{1,2})\b' src/bayesmith --include='*.py'`）：

- **91 处原始命中，36 个不同代码，35/44 个文件**。恰好一个误报：`evidence/chain.py:238` 的 `P0` 是 `-0.5 logdet(2 pi P0)` 里的初始协方差。→ **90 处真实，35 个代码**。
- 前六名：`B11`×10、`P1`×9、`P3b`×8、`P3a`×7、`B9`×7、`P4`×4。
- **6 个文件的 docstring 第一行**就是代码：`optimize.py`（G2）、`amortize.py`（G5）、`evidence/__init__.py`（B11）、`evidence/campaign.py`（B11）、`exact/reduced_basis.py`（G4）、`exact/discrete.py`（P4）。`help()` 打印的模块摘要就是这一行。
- **一处在运行时到达用户**：`dispatch/factor.py:528`，在一个 `StructureError` 的**消息正文**里——"declare it 'gcr' and read this function's note on G12"。注意这条其实是**可解的**：那条 note 就在同一函数的 docstring 里（`factor.py:454`），随 wheel 发布。所以缺陷是**标签不透明**，不是悬空引用。

**哪些真的解不了**：把 src 里的代码集与"sdist 会发的文档（CHANGELOG + docs/migration + README）里提到的代码集"求差，得到 **9 个代码 / 33 处**：`D10 D16 D33 D41 P1 P3 P3a P3b P4`。反过来说，`CHANGELOG.md` 已经**定义式地**解释了不少（`:65` "G5: `bayesmith.amortize` — a posterior fitted to simulations…"、`:113` G6、`:169` G3、`:197` G4），而且 CHANGELOG 在 sdist 里。**但 wheel 不发 CHANGELOG**，所以对 `pip install` 的读者，90 处全部不可解。全仓库无术语表（`git ls-files | grep -iE "glossar|plan-code"` → 0）。

**第二套引用体系更糟（DEFECT）**：`Section N.N`，11 处（`dispatch/plan.py:9,458,485,627,707`、`dispatch/execute.py:6,49,191`、`evidence/compress.py:408`、`evidence/diagnostics.py:195,271`），其中两处是 `InferencePlan.sample` 和 `.estimate` 的**第一行**。它们看起来像是引用本包自己的文档。

`evidence/` 的三处全是 `Section 9.3`，指向 e-RHINO 的 `docs/superpowers/specs/2026-08-04-streaming-evidence-design.md`，那份文档 `## 9. Diagnostics that can actually fire`（第 420 行）**没有 9.3 这个标题**，检索不到。**而且今天变得更糟**：`ffb46d5` 引入的 `docs/superpowers/specs/2026-08-28-multi-dataset-joint-posterior.md:1050` 现在有一个 `### 9.3 第二个 graph 级 likelihood factor slot`——**主题完全不同**。一个在 bayesmith 里 grep "9.3" 的读者现在会得到一个自信的错误答案。

> **R9 已落地,2026-08-28。** 三处都改成**引用内容而不是引用编号**。
> 追下去发现「Section 9.3」指的不是一个 `### 9.3` 标题,而是那一节**编号列表的
> 第 3 项**(「a fixed-size (~100 byte) per-epoch residual summary: chi-square,
> DOF, and the residual projected onto a handful of NAMED systematic
> templates」)——所以任何 grep 都解不到它,不是因为文档改了,是因为那个编号
> 从来就不是一个可检索的锚点。**而且那份文档在 e-RHINO 里是未跟踪的**
> (`docs/superpowers` 被 gitignore),所以 clone 这个仓库的人根本够不到它。
> 现在那句话**被引在 `ResidualSummary` 的 docstring 里**,另外两处指向它;
> 本仓那个主题无关的 `### 9.3` 造成的「自信的错误答案」也在同一处点名了。
> **顺带一笔**:`marginal/diagnostics.py` 从 775 行到 **779**(上限 800)。
> 交接页说的「往里加东西之前先拆」**仍然欠着**——这次是一处必须做的更正,
> 加了四行,不是新功能。

**应该换成什么**：代码本身在正文里多数是**有用的**（`exact/precision.py:24` "exactly defect B1, where one engine dropped `sum log sigma`" 自带解释）。所以不建议全清。建议：(a) 清掉 6 处第一行；(b) 把 `factor.py:528` 的 `G12` 改成"（G12 note：sigma frozen at the block's current value）"这种**保留锚点、补上含义**的形式；(c) 修掉 3 处 `Section 9.3`；(d) 其余用一份随 sdist 发布的 `docs/plan-codes.md` 兜底。注意**不能盲清**：`tests/exact/test_condition_estimate.py:243` 有 `assert "G14" in text or "D15" in text`，断言的是 `bayesmith.exact.conditioning.__doc__`，而这个测试在 sdist 里。

---

# 6. `evidence/` 要不要改名

## 6.1 这个子包实际上是什么

一句涵盖全部七个模块的真话：**一个 epoch / 一个数据集的线性-高斯模型压缩成的、定尺寸的平方根信息项，及其全部簿记**——形成前对图的结构检查（`factorize`）、项的算术（`sqrtinfo`、`compress`）、从图形成与整场折叠（`campaign`）、跨 epoch 漂移的 nuisance 的精确积出（`chain`）、以及一场折叠完之后它能对自己说什么（`diagnostics`）。

结构上它是**叶子应用**，不是基础设施：入度 1（唯一消费者是 `dispatch/streaming.py:40,41`，取 27 个导出名里的 2 个）、出度 3、位置上与 `diagnose/` 完全对称。

**34.5% 根本不是密度**：`diagnostics.py` 775 行 + `factorize.py` 281 行 = 1056 / 3057。z 分数、χ² 尾、`log σ` 对 `log N` 的 OLS 斜率、一个条件独立性检查，以及一个**不返回任何数字**、只负责拒绝的导出（`refuse_undeclared_coherent_error`）。这是 TASTE 层面的观察，但它是"没有任何量词能诚实命名这个目录"的原因。

## 6.2 名字是假的，还是只是不精确？——三者都有，这本身就是发现

**(a) 它宣称了包不做的事（假）。** `p(d)` 无入口、无消费者、无 Bayes factor（第 3.1 节的 grep）。

**(b) 但算术是完整的——我自己重跑过，不是转抄。** 只用三个已发布的公开函数（`compress` → `nuisance_prior` → `SqrtInfo.combine` → `marginalise(全部名字)` → `log_prob({})`），`JAX_ENABLE_X64=1`，n=12、p=3、σ=0.7、非单位先验标准差 `[2.0,0.5,1.3]`、非零先验均值：

```
machinery  : -16.555096522587395
closed form: -16.555096522587405     # N(d | A m, A S Aᵀ + σ²I)
abs diff   : 1.07e-14
```

`sqrtinfo.py:279-280` 的 docstring 自己就写着 *"Marginalising **every** name is legal and returns a zero-width term whose `log_prob({})` is the marginal likelihood."* 所以名字是**过早**，不是**无据**。

**(c) 但图级 API 故意不这么做。** `campaign.py:102` 只在 `for name in found.per_epoch` 里调 `gaussian_parts`，survivors 的先验从不从图里读；我上面那次能算出来，是因为我拿 **nuisance** 的先验发射器传了 `over=()`、`over_shapes={}`（`nuisance_prior` 的签名可查）。也就是说：**数组层能算 `p(d)` 且文档说了；图层刻意只算 `p(d|θ)`，且参数名对你在做什么撒谎。**

**(d) 词被包自己占着，而且用得对。** 15 处普通英文义在 `errors.py`、`dispatch/`、`exact/` 里，其中 `errors.py:9` 是唯一被 eager import 的模块、是用户读到的第一段 bayesmith 散文。这 15 处每一处都因为一个子包占了这个词而瞬间产生歧义，而**任何写在 `evidence/` 里面的免责声明都够不到 `evidence/` 外面的散文**。

**(e) 这个词的最终主人在别处。** 若真做模型比较，`p(d)` 的一般情形落在 `exact/correct.py:174-182` 的 `log_weight` 旁边（那里丢掉的常数补回来就是），不在一个线性-高斯 epoch 压缩器里。届时 `bayesmith.evidence` 会是错的 import path。

**诚实的反方（我认真对待）**：滤波传统（Durbin & Koopman、Särkkä、GP marginal likelihood）确实把新息累积的 `log p(d|θ)` 叫 marginal likelihood 或 evidence，`chain.py:380` 的论证在那个文献里站得住。但那个传统**固定 θ**；本包的标语是"一个算子图就是一个贝叶斯模型"，带先验、带图、带 NUTS，在模型空间里 "evidence" 只有一个意思，而那不是这个。

## 6.3 候选名，各自宣称什么，实测冲突

| 候选 | 宣称 | 实测冲突 | 覆盖 |
|---|---|---|---|
| `evidence`（现状） | `p(d)`、模型比较 | 语义冲突：包内 15 处正确英文义 + 4 处包内歧义 | — |
| `factors/` | 似然因子 | **封死**：`dispatch/factor.py`，且 `factor_partition`/`sample_factors`/`estimate_factors` 在顶层 `__all__` | — |
| `streaming/` | 流式处理 | **封死**：`dispatch/streaming.py`（正是决定要不要走进本层的那个模块） | — |
| `compress/` / `campaign/` / `sqrtinfo/` | 动词/领域名词/形式 | **需同批改内部模块名**：三者都与子包内模块同名，而且这个陷阱是活的——`type(bayesmith.evidence.compress).__name__` 实测是 `'function'`，重导出的函数**永久遮蔽**同名模块（`factorize` 同理；`campaign` 目前解析为 module） | `compress` 覆盖 6/7，`campaign` 覆盖 7/7 |
| `information/` | 信息形式 | **封死**：60 处，`fisher_information` 在顶层 `__all__`，且 `SqrtInfo.information()` 专门写了它不是 fisher | — |
| **`marginal/`** | 每数据集的边缘似然 `L_i` | **无冲突**（无同名模块、无顶层同名、`marginal` 全包 11 处） | 密度那 65% 精确；诊断那 34.5% 不覆盖 |
| `likelihood/` | 似然因子及其精确簿记 | **无冲突**（53 处词频，无模块、无顶层名） | 钝，但无假声明 |
| `sufficient/` | 充分统计量 | 无冲突（3 处），但形容词做包名读着别扭 | 覆盖 7/7 |

## 6.4 实测成本

**仓库内**：`bayesmith.evidence` 字符串 **106 处 / 29 个文件**（`src` 58、`tests` 32、`docs` 16；README 与 CHANGELOG 各 0）。其中 `src/bayesmith/__init__.py` 里 16 行（14 行 `_LAZY_ATTRS` + 1 行 `_LAZY_SUBMODULES`）。**子包外的运行时 import 只有 2 行**（`dispatch/streaming.py:40,41`）。**一处用户可见的错误消息**：`exact/gaussian.py:268`。加上两次目录移动：`src/bayesmith/evidence/`（7 文件 3057 行）与 `tests/evidence/`（11 文件 3493 行，且 `tests` 在 sdist include 里，所以这是已发布的路径）。**一天，不是一小时。**

**已发布面**：`git show v0.4.0:src/bayesmith/evidence/__init__.py` 的 `__all__` 是 **17 个名字**；与顶层 `__all__`（91 名）求交是 **空集**——**全部 17 个只能从深路径到达**。这与卷宗里"14 个已在顶层、改名不触及"的说法**方向相反**，我实测的是空交集。

**下游**：`grep -rn "bayesmith\.evidence"` 遍历 e-RHINO 的 `src` 与 `tests` → **0**。唯一已知消费者一次都没 import 过这个子包（它 import 的是 `graph.evaluate`×6、`exact.fisher`×4、`errors`×4、`dispatch.factor`×3、`diagnose.identifiability`×3 …，共 9 个生产模块），且钉 `bayesmith>=0.4` 无上限。

**窗口，向远端量的**：

```
PyPI:  releases ['0.1.0','0.2.0','0.3.0','0.4.0']  latest 0.4.0
       上传时间 2026-08-26T08:18 → 2026-08-27T10:40（两天内，Alpha 分类）
git tag                 → 止于 v0.4.0
git ls-remote --tags    → 止于 v0.4.0
origin/main 54c90f1     vs  本地 HEAD ffb46d5
pyproject.toml          → version = "0.5.0"
```

**0.5.0 已切、未打 tag、未推送。** 这个窗口关在 upload 那一刻，而且下游成本会在 **P6 之后**从零变正：`docs/superpowers/specs/2026-08-24-rheplicant-migration.md:614` 第 6 步是 "P6 = B11 流式证据重写"，届时 rheplicant 自己那份 `inference/{sqrtinfo,compress,chain,factorize,diagnostics,compressed,archive,memory}.py` 加 40 个文件的 `tests/evidence/` 才会真正切过来，那时才第一次出现真实 import 点。

**垫片，实测**：卷宗里有人断言 `__getattr__` 垫片不成立。我在 scratchpad 里搭了一个最小复现验证：

```
包级 __getattr__ 垫片:
  from pkg.old import Thing        -> OK
  from pkg.old.kernel import helper-> ModuleNotFoundError
  import pkg.old.kernel            -> ModuleNotFoundError
```

**这条断言是对的**，而且致命——因为 17 个已发布名全部只在深路径上，而深路径 import 正是本包自己的写法（`dispatch/streaming.py:40-41`）。但**可修**：加 `sys.modules` 别名注册后，同样四种写法我实测**全部通过**。所以"非破坏性垫片"是可行的，只是不是 30 行的那一版。

## 6.5 你自己今天已经裁决过

`ffb46d5` 引入的 `docs/superpowers/specs/2026-08-28-multi-dataset-joint-posterior.md` §0 把词汇钉死了（§0.1：`L_i` = 边缘似然、`p(d_1..d_N)` = evidence、**范围外**），§0.3 列出"三处名字说谎"，第一处就是 `bayesmith/evidence/` 整个模块，并给出裁决：

> **建议**：不重命名模块（成本高、收益小），但在 `evidence/__init__.py` 顶部加一段词汇声明，并逐条修掉上面五处 docstring 的理由。

我把这条裁决的两侧都量了：**成本侧**可测为 106 处 / 29 文件 / 两次目录移动（"高"成立，但不是数量级）；**收益侧**"小"在**今天**成立（下游 0 import、已发布面 17 名、两天龄 alpha），在 **P6 之后不成立**，且窗口在 upload 关闭。

## 6.6 建议

**我倾向改名，目标 `marginal/`，在 `v0.5.0` 打 tag 之前，并保留 `bayesmith/evidence/` 作为带 `sys.modules` 别名的弃用垫片（发 `DeprecationWarning`，1.0 退役）。信心约 0.6——这是一个接近的判断，不是显然的结论。**

选 `marginal/` 而不是卷宗里的 `campaign/` 或 `compress/`，理由是三条实测：(i) 它是**你自己今天写下的词汇表**里 `SqrtInfo` 项所存的那个量（§0.1"一个 `SqrtInfo` 项存的就是这个东西"）；(ii) 它是唯一**零冲突**的候选——`campaign`/`compress`/`sqrtinfo` 都与内部模块同名，而 `type(E.compress) == function` 已经证明这个遮蔽陷阱是活的；(iii) 它留门——`p(d)` 也是一个 marginal，将来若真做模型比较，不必再改一次。代价是它不覆盖那 34.5% 的诊断，这一点我承认，并把它标为 TASTE。

**如果你维持 `ffb46d5` 的裁决（这完全站得住），那份裁决还不够，必须补两件：**
1. **`dispatch/plan.py:1`**：把 "its evidence" 改成 "its **reasons**"——`plan.py:509` 已经这么写了，`reason` 是真实字段名。这是一个词，免费，且它是 `evidence/` 之外那 15 处歧义里最刺眼的一处。
2. **`evidence/__init__.py` 的那段词汇声明必须显式写出"本层不计算 `p(d)`；这里每一个 marginal 都是对 θ 的条件量"**，而不只是解释五处 docstring 的理由。因为读者会在 import 行、traceback、`pip show` 级别的目录浏览里读到这个名字，而那些地方读不到 docstring。

---

# 7. 建议清单

对抗审阅者对卷宗里 13 条建议的裁决我全部保留并标注。**被驳回的没有被静默删除，也没有被静默提拔。**

| # | 建议 | 对抗裁决 | 成本（实测） | 破坏性 | 优先级 |
|---|---|---|---|---|---|
| **R1** | 把 `optimize`/`amortize`/`distributions` 加进 `_LAZY_SUBMODULES`（`__init__.py:251`）；把 `tests/test_public_api.py:297,371` 的逐名断言换成从文件系统派生（`pkgutil.iter_modules`，过滤 `_` 前缀，双向相等） | **SURVIVES** | 3 行源码 + ~15 行测试 | **否**（纯放宽：原本抛错的属性访问变成成功） | **1 · 应挡住 0.5.0 的 tag** |
| **R2** | 删掉 `evidence/__init__.py:3` 的 "Nothing here reaches the graph yet"，删掉 `-- B11`，改成一段自足的说明 + 明确的 `p(d)` 免责 + 覆盖 `chain` 与 `diagnostics` | **SURVIVES（须修正）**：原提案说"两条悬空 docs 路径"，实测只有一条（`docs/evidence-layer-readiness.md`），且它**存在于仓库**、只是不随 sdist/wheel 发布 | 一个 docstring | 否 | **2** |
| **R3** | `dispatch/plan.py:1` "its evidence" → "its reasons" | 卷宗内未被单独攻击 | 一个词 | 否 | **2** |
| **R4** | **改名 `evidence/` → `marginal/`**，保留 `evidence/` 作 `sys.modules` 别名垫片，1.0 退役；同批修 `exact/gaussian.py:268` 的错误消息 | 卷宗里 `campaign/` + `__getattr__` 垫片的版本被 **REJECTED**，三条理由：垫片不支持深路径（**我实测确认，成立**）、成本低估（**成立**：实测 106/29 而非 58/23）、新名与内部模块冲突（**成立**）。我以 `marginal/` + `sys.modules` 垫片重新提出，三条逐一被处理；但**你自己在 `ffb46d5` 已裁决不改名**，见 6.5–6.6 | 106 处 / 29 文件；2 行运行时 import；16 行懒加载表；两次目录移动（3057 + 3493 行）；1 处用户可见字符串。约一天 | **是——`bayesmith.evidence` 是 0.4.0 已发布的 import path，17 个名字只能从深路径到达；垫片可把破坏降为零** | **4（窗口在 0.5.0 upload 关闭）** |
| **R5** | README：`0.4.0`→`0.5.0`、"in two places"→9 个模块、"0.3 floor"→`>=0.4`、补 `diagnose`/`optimize`/`amortize`；加版本守卫 | **SURVIVES（须修正）**：守卫**不能**读 `importlib.metadata.version("bayesmith")`——本 checkout 实测返回 **0.2.0**（可编辑安装的 dist-info 陈旧），守卫会对着正确的 README 变红并指示写 0.2.0。改读 `pyproject.toml`（`tomllib`）或 git tag | 一段 + 4 行测试 | 否 | **3** |
| **R6** | `dispatch/factor.py:528`：把 "read this function's note on G12" 改成"保留 G12 锚点 + 补上含义"，**不是删除** | **SURVIVES（须修正）**：原理由"指向未发布的中文文档"是**假的**——那条 note 就在同一函数 docstring（`factor.py:454`），随 wheel 发布。缺陷是标签不透明，不是悬空引用。删除会切断 `factor.py:454` / `CHANGELOG.md` / `TestG12...` 四处交叉索引 | 一行 | 否 | **3** |
| **R7** | 把 `chain.py` 的六个名字补进 `evidence/__init__.py` | **REJECTED**，我的实测支持驳回：门面缺口是**全包惯例**（40/81：`graph` 17/17、`bridge` 4/4、`dispatch` 7/9 含 `compile`），单修 `evidence` 会让它成为唯一穷尽的子包，制造一条在一处为真、四处为假的新声明；且它在改名决定之前把广告面从 27 扩到 33，其中 `chain_marginal`/`smooth` 正是与"evidence"冲突最狠的两个 | — | — | **改为**：R2 的 docstring 里写明 `chain` 在 `bayesmith` 顶层导出；门面规则若要立，须一次覆盖全部六个子包，且在改名之后 |
| **R8** | 修 `exact/gibbs.py:19-24` 的散文（"never reads dispatch" → "no module in `exact` imports `dispatch` at module scope；`fisher.py` 在调用内借一个函数"）；给四处无注释的延迟 import 各加一行注释；加一个断言模块级子包 DAG 无环、且 `exact/` 不在模块级 import `dispatch/` 的测试 | 卷宗里那条"把 `prior_environment` 搬到 `graph/evaluate.py`"的版本被 **REJECTED**，我独立确认它**必然失败**：`_latent_centre`（`classify.py:108,130`）用 `exact.gaussian` 的 `gaussian_parts`/`node_shape`（`classify.py:45-46`），而 `exact/gaussian.py:52` 在模块作用域 import `graph.evaluate`——搬过去就是硬循环。这里保留的是被驳回提案里**唯一有价值的那一半**：那个测试在 HEAD 上原样就通过 | ~20 行测试 + 5 处注释/散文 | 否 | **3** |
| **R9** ✅ **已做 2026-08-28** | ~~修 3 处 `Section 9.3`~~（`evidence/compress.py:408`、`diagnostics.py:195,271`）；其余 8 处 `Section N.N` 至少注明指向哪份规范 | 卷宗里"11 处全解不了"被我推翻：8 处能解到 `docs/superpowers/specs/2026-08-23-p3b-dispatch-execution-design.md`；只有 `9.3` 解不了，**而且今天变糟了**——`ffb46d5` 新增的 `2026-08-28-multi-dataset-joint-posterior.md:1050` 有一个主题完全不同的 `### 9.3` | 3–11 行 | 否 | **3** |
| **R10** | plan codes：清 6 处第一行 + `factor.py:528`；其余用随 sdist 发布的 `docs/plan-codes.md` 兜底 | 卷宗里"全清 90 处"被 **REJECTED**，我确认了那条驳回的核心证据：`tests/exact/test_condition_estimate.py:243` 断言 `"G14" in text or "D15" in text`，断言对象是 `bayesmith.exact.conditioning.__doc__`，而这个测试在 sdist 里，盲清会让"sdist 自带测试跑绿"这条性质失效 | 6 处编辑 + 一份文档 | 否 | **4** |
| **R11** | `Block.method` 的 docstring（`plan.py:341-342`）：补上 `"log-gcr"`，去掉"由 `Classification` 选择"的错误归因 | 卷宗里"抽一个共享五方法模块"被 **REJECTED**（`factor.py:80` 与 `gibbs.py:53-55` 各自写明某一行**故意缺席**，分歧本身就是规格；且 `FACTOR_METHODS[:2]` 的顺序是承重的） | 3 行 | 否 | **3** |
| — | 把 `JeffreysPrior` 搬到顶层 `bayesmith/priors.py` | **REJECTED**：`bayesmith.priors` 会成为读者去找 `joint_prior`（在 `graph/trace.py`）和逐节点先验（节点的 `dist_fn`）的地方，而两者都不在里面——用一个窄的错架子换一个宽的假承诺；且垫片会让 `tests/test_public_api.py` 的身份断言变成恒真 | — | — | **不做** |
| — | `Block` → `PlanRow` | **REJECTED**：`row` 在 src 里已有三义，`gibbs.py:395` 明说"一个 block **有**一个 row"；且 `str(plan)` 打印 "block 0"（`plan.py:596`），README 的头号示例就是它 | — | — | **不做** |
| — | `partition` → `degeneracy`（`solve.py:231,234`、`conditioning.py:32,106`） | **REJECTED**：前提为假，那四处就是 sense 1；替换后 "a near-degenerate degeneracy" 不成句 | — | — | **不做** |
| — | `operator` 的散文消歧 | **REJECTED（放置错误）**：两义只在 `src/bayesmith/__init__.py` 同时出现（`:1` 主题句 vs `:62` `linear_operator`），`exact/` 里 0 处节点义 | — | — | **若做，放顶层 `__init__.py`，一句** |

---

# 8. 不建议动的

1. **不要改 `diagnose/`。** 它有和 `evidence/` 完全同型的缺陷（名字记录的是迁移批次，不是代码类别；`JeffreysPrior` 是会改变后验的建模构件），但成本侧相反：e-RHINO **生产代码**钉了 `bayesmith.diagnose.identifiability`（`identifiability.py:93,94`）和 `bayesmith.diagnose.local`（`uncertainty.py:449`）。这个不对称正是"`evidence/` 能改而 `diagnose/` 不能"的全部理由。改它的 docstring 和 README 那条 bullet 就够了。

2. **不要把 `optimize.py`/`amortize.py` 提升为子包。** 单文件包本身是谎言，而且 `bridge/` 只有 360 行 2 个文件却是包——尺寸规则早已不成立。写下规则，不要重排树。顺带值得在 `amortize.py` 的 docstring 里加一句：它入度 0、只 import `errors`、不接受 `Graph`（`Graph` 在该文件出现一次，就在解释它为何缺席的那句里），是包的主题句**结构上无法选择**的那条推断路径。

3. **不要动 `evidence/sqrtinfo.py` 的位置或它的 `information()` 命名。** `sqrtinfo.py:31-33` 明说它不认识 graph/epoch/plan，那条分离正是它能被 dense oracle 对拍、能与 `rheplicant.inference.sqrtinfo` 交叉校验的原因；`information()` 的词界协商是全包应该照抄的范本。（这也是 `campaign/sqrtinfo.py` 作为改名结果不可接受的原因之一。）

4. **不要为了"全包一致"去补齐门面。** 40/81 的缺口是真的，但它是**惯例**而不是**事故**；一次性补齐是一个覆盖六个子包的更大提案，且必须在名字定下来之后做。眼下唯一值得单独处理的是 `bayesmith.dispatch.compile` 取不到——那是头号入口，值得一行。

5. **不要碰 `Graph.joint_prior: Any`。** `graph/graph.py:35-50` 写清楚了这是**刻意的结构化鸭子类型**（"这个模块是核心，不 import diagnose 里的任何东西"），不是意外的循环。搬 `JeffreysPrior` 不会消除它。

6. **`tests/test_readme_count.py` 这个守卫是对的，照它的样子加版本守卫。** 它是本仓库"derive, do not re-spell"落实得最好的一处；README 的测试数今天准确（1523 = 1523）。缺的只是同样待遇给版本号，且数据源必须是 `pyproject.toml` 而不是 `importlib.metadata`（实测 0.2.0）。

7. **`evidence/` 是一个子包还是两个（密度算术 vs 诊断/结构检查），是 TASTE。** 34.5% 不是密度，这是"没有量词能诚实命名这个目录"的原因；但 `SqrtInfo` 确实被两半共享，"按领域对象内聚"是站得住的组织方式。**命名这个目录之难，可能是症状而不是病。**
