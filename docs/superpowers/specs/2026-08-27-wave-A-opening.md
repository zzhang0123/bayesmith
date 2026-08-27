# 执行页 Wave A(开工批)— 私名普查、契约阅读、分诊表,以及 D21

> 计划:`2026-08-26-one-implementation.md` §五 Wave A。
> 前一批次:`2026-08-27-wave-P2-D9.md`。
> **日期**:2026-08-27 · **本页状态:Wave A 已开工;本批只做「每波开工」那三件
> 事,外加一条实测出来的新裁决 D21。没有任何 rheplicant 模块被切换。**

## 〇、这一批为什么不切模块

计划 §五 写着每波开工要做三件事:**铁律 1 私名普查**、**铁律 7 契约阅读**、
**测试分诊表**。这三件在 Wave A 上不是形式——普查改变了「哪些名字是保持面」,
契约阅读**推翻了契约页上的一句话**(见 §三),而分诊表决定这一波有多大。

先把它们做完再切第一个模块,是因为其中任何一件都可能改变第一个模块该怎么切。
实际上确实改变了:**D21**。

## 一、铁律 1:私名普查

`grep -rn 'inference\.<mod>' --include='*.py' src/`,滤掉 docstring 引用后,
**真正的 import**:

| 模块 | 被谁 import 了什么 | 处置 |
|---|---|---|
| `uncertainty` | `memory.py` ← `FlatMatrix, _named_spans`;`reduced_basis.py` ← `FlatMatrix, _named_spans`;`plan.py` / `numpyro_bridge.py` / `compress.py` / `sensitivity.py` ← `as_noise_model` | **文件不整删**。`as_noise_model` 留守(计划已写);`FlatMatrix` **永久**保持(config products 逐字段读它);`_named_spans` 随 Wave C/D 的最后一个消费者退役 |
| `sensitivity` | `config/postflight/fitting.py`(C19)、`__init__.py` | 公开面保持 |
| `identifiability` | `config/postflight/fitting.py`(C13)、`plan.py` | 公开面保持;`IdentifiabilityReport.participation` 被 `plan.py` 用 |
| `priors` | `__init__.py`、`parameters.py` ← `JeffreysPrior` | `JeffreysPrior` 是 `ParameterSpace.joint_prior` 的类型,保持 |
| `numpyro_bridge` | `__init__.py`、`uncertainty.py` ← `predict_from_samples` | 公开面保持 |

**普查确认了计划 §一 的三条预告,没有推翻任何一条**,并补上一条它没点名的:
`plan.py` 用 `IdentifiabilityReport.participation`,所以那个方法在 `plan` 切换前
(Wave B)是保持面。

## 二、铁律 7:契约阅读

契约 = 旧 spec §四 台账行 + `docs/migration/<mod>.md`。五页都读了。要点:

* **签名两侧不同,且这是有意的**(旧 spec §六 用整节论证过为什么不做「薄壳」
  重导出):`identifiability(space, pipeline, state_template, …)` 对
  `identifiability(graph, *, names, at, rtol)`。所以门面必须**建图**,而不是转发。
* **异常映射**:`ParameterSpaceError → GraphError`、`StateValidationError →
  StructureError`。门面要走**反方向**,而 P1 的 `translate` 今天只覆盖仿射类与
  `GraphError`;诊断族的拒绝形状不同,`translate` 要不要扩、还是各模块自己翻,
  是切第一个模块那一批的事。
* **x64 机制两侧不同**,已由 D9 结清:bayesmith 拒绝,门面按 (a) 在 x64 上下文
  **内**建图。rheplicant 现成的 `_in_float64()` 正是这个机制,位置改成包住
  「`to_graph` + bayesmith 调用」即可。
* **`DEFAULT_RANK_RTOL` 已在本侧重测**(7.479266e-17 / 4.822138e-05 /
  3.116759e-08),常数与窗口都成立;D9 又把它从「一个模型」扩到了「一个族」。

## 三、契约阅读推翻了契约页上的一句话 → D21

`docs/migration/identifiability.md` §5.2 原文:

> `at` defaults to prior centres (`prior_environment` …) rather than declared
> inits — **the same point, one spelling**.

**实测为假。** `Latent` 的 `init` 与 `prior` 是**互相独立**的字段——
`parameters.py` 自己的例子就是
`Latent("fwhm_deg", init=12.0, prior=dist.Uniform(5.0, 30.0))`。
而 identifiability 是**非线性模型的局部性质**(它自己的 docstring 这么写),
所以展开点不是记号问题。

同一个模型、同一组先验,`mu = a·exp(b·x)`:

| 展开点 | nullity | `s_min/s_max` |
|---|---|---|
| `a = 0.0` | **1** | 0.000e+00 |
| `a = 1.0` | **0** | 2.406e-01 |

**判决翻转。**

**处置(D21,委托下自定)**:门面**显式传** `at=space.initial_values()`,不吃
bayesmith 的默认。三条理由与完整论证在计划 §二 的 D21 行;契约页 §5.2 已改写,
并写明它错过、什么时候改的、以及是什么让它暴露(有人要在它上面建门面了)。

> 这是本程序里「一个事实两份拼写,其中一份悄悄过期」的又一例,而这次过期的是
> **契约页**——铁律 7 让人读它,所以它错了就会被照着实现。开工三件事里,
> 「读契约」这件的产出正是这个。

## 四、分诊表(种子)

Wave A 触及的 e-RHINO 测试文件与用例数(`--collect-only` 实测):

| 文件 | 用例 | 属 |
|---|---|---|
| `test_identifiability.py` | 45 | identifiability |
| `test_prior_sensitivity.py` | 64 | sensitivity |
| `test_jeffreys_prior.py` | 49 | priors |
| `test_fisher_prior.py` | 18 | priors / uncertainty |
| `test_declared_prior.py` | 40 | priors / parameters(**多数留守**) |
| `test_numpyro_bridge.py` | 13 | numpyro_bridge |
| `test_numpyro_noise_model.py` | 11 | numpyro_bridge |
| `test_uncertainty.py` | 25 | uncertainty |
| **合计** | **265** | |

三列分诊(原样重放 / 改写对适配器 / 带理由退役)**逐文件的判定留给各模块自己
那一批**——本批不切模块,所以现在填会是猜测。这里只钉住**总量**:Wave A 要处置
**265** 条,是本程序至今最大的一批,计划估的 2–3 个会话是合理的下限。

**`test_declared_prior.py` 的 40 条大部分留守**:`ParameterSpace`/`Latent`/`Bind`
是声明层(终局形态第 1 类,不迁移),那个文件测的多是声明与校验而非诊断。切
`priors` 那一批要先把它逐类分开。

## 五、铁律 4 四件套

| | 项 | 结果 |
|---|---|---|
| (i) | 该批测试全绿 | bayesmith **1295 passed / 0 skipped** exit 0;e-RHINO 未动 |
| (ii) | 接缝变异红 | **本批不适用**:没有新代码。D21 的守卫随第一个模块切换落地(见 §六) |
| (iii) | 旧实现删除、计数守卫刷新 | 无删除 |
| (iv) | 文档实测数字重测 | `docs/migration/identifiability.md` §5.2 改写;计划 §二 新增 D21 |

**(ii) 写成「不适用」而不是省略**:一个只做普查、阅读与登记的批次没有可变异的
新判据,而「变异集为空」和「忘了跑变异」在记录页上长得一样。D21 的守卫**欠着**,
欠条写在 §六。

## 六、下一批(Wave A 的第一个模块)

**切 `identifiability`。** 它最小、契约最清楚,而且 D9 刚把它的先决结清。要做的:

1. 门面 `identifiability(space, pipeline, state_template, names=, at=, rtol=)`
   保持签名与异常类身份不变,内部:`_in_float64()` → `to_graph(...)`(**在块内**)
   → bayesmith `identifiability(graph, names=, at=space.initial_values() | at,
   rtol=)` → 报告字段逐个搬回 rheplicant 的 `IdentifiabilityReport`。
2. **D21 的守卫**:一个 `init != prior 中心` 的 fixture,断言门面给出的是
   **init 处**的判决。没有这条 fixture,D21 是一句话而不是一条裁决——两个默认
   碰巧相同的模型分不开它。
3. `IdentifiabilityReport.participation` 是保持面(§一),`plan.py` 在用。
4. 分诊 `test_identifiability.py` 的 45 条;cross-check 文件同批删除,其中参照
   非 rheplicant 的 oracle 断言**逐条改籍**进 bayesmith(铁律 2)。
5. **这是本程序第一次动 rheplicant 的 main**,所以也是第一次真正用到铁律 5 的
   发布门:0.4.0 必须已在索引上,e-RHINO 的 bayesmith 地板同批升到 `>=0.4`。
6. **接缝变异这一批起是真的跨仓变异**(改 bayesmith、看 e-RHINO 红),不再是
   库内的替代品。附录 A 要新增行。
