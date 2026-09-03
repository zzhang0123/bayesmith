# 执行页 Wave A · 模块 5 第一步 — `fisher_information` 的似然一半切换

> **文档状态：`record`** · 已落地批次/审计/测量的历史记录，写作当天为真，非当前权威。索引见 docs/README.md。

> 计划:§五 Wave A / 铁律 1–5;新增 **G15**(缺口)与 **D29**(裁决)。
> 前一批次:`2026-08-27-wave-A-numpyro-bridge.md`。
> **日期**:2026-08-27 · **本页状态:`fisher_information` 已切,`parameter_covariance`
> 与 `propagate_covariance` 留给下一步(§八)。`uncertainty` 是 Wave A 唯一被拆成
> 多步的模块,理由在 §三。**

## 〇、分诊表(本步)

| 列 | 数 | 内容 |
|---|---|---|
| **原样重放** | **43** | `test_uncertainty.py`(25)+ `test_fisher_prior.py`(18),一条未改 |
| 改写对适配器 | 0 | |
| 带理由退役 | 0 | (cross-check 归下一步一并处置) |

**43 条一字未动**就是这一步的验收。

## 一、开工前先量了两件事,两件都推翻了一个「显然」的假设

### `params` 有两种形状,而只有一种有名字

`fisher_information(forward, params, ...)` 的 `params` 是一个**任意 pytree**。
把一条探针装进它、跑遍所有调用点(10 个文件),**78 次调用**:

| 形状 | 次数 |
|---|---|
| 扁平 `{name: array}` dict | **63** |
| 裸 `ArrayImpl` | 14 |
| `Assembly`(pipeline pytree) | 1 |

于是:dict 走**一个 latent 一个节点**;其余没有名字可给(`_named_spans` 自己就
返回 `(None, None, None)`),走**一个** latent、域是 `ravel_pytree` 自己的向量
——而那**正是** `FlatMatrix` 文档里写的排布,所以两条路的布局是同一个。

### `space=` 会不会撞上没有名字的那一种?会,但**已经被拒**

第二次探针把 `space is not None` 也记下来:**78 次里恰好 1 次**是
`space=YES` + 裸数组。查下去,那是 `_prior_precision` 里**既有的**一条拒绝的测试
(「这些 params 没有名字……按位置放先验就是猜」)。所以它到不了图,映射是干净的。

> 两次探针都值得记:第一次得出的「非 dict 也要支持 `space=`」是错的,第二次才把它
> 排除掉。**先量再设计**在这里省掉了一整条不需要存在的代码路径。

## 二、切了什么,以及一次逐比特的对照

雅可比、加权设计、方差自己那一项(`2 (dlogσ/dx)^T(dlogσ/dx)`)**全部过缝**,
经 `graph_bridge.graph_for_information`(`priors` 批建的那个 helper,本批次让它
也能携带声明的先验)。

**写代码之前先量过**:同一模型、同一点,两种噪声各一次:

```
homoscedastic: rheplicant[0,0]=1.231060261e+06  bayesmith[0,0]=1.231060261e+06  max rel diff=0.000e+00
radiometer   : rheplicant[0,0]=8.823785345e+05  bayesmith[0,0]=8.823785345e+05  max rel diff=0.000e+00
```

块按 **sorted(names)** 索取,因为那就是本包的扁平排布——远端按给它的顺序铺,
所以**事后不需要置换**(D24 是同一件事从唯一一个能自己选顺序的调用点看过去)。

## 三、G15:一条真缺口,而它**不是**被绕开的

`fisher_information(space=...)` 要加上声明先验的曲率。远端 `include_prior=True`
正是这件事——但它从**块**上读先验,而两个块构造器各缺一半:

| 构造器 | 雅可比 | 先验 |
|---|---|---|
| `local_block` | 在**调用者的点**取切线,对非线性模型是对的 | **故意为空**——它自己的 module docstring 写着:交给 `include_prior=True` **必须**在空字典上响亮失败,而不是折进一个没人声明的曲率 |
| `unchecked_operator` | 在**域的零点**取切线(仿射映射处处一个切线) | 有 |

对一条幂律,第二个的雅可比是错的。**两者都没错,只是还没有第三个**;而铁律 5
禁止依赖一个没有发布的远端表面。

**处置:登记为 G15,先验曲率暂时留守,并给它写上解除条件。**
`_prior_precision` 的 docstring 里逐字写着:G15 落地并发布之后,该函数删除、
调用改为 `include_prior=space is not None`——**委托已经照那个形状写好,只有那一行会变**。

> 这是「限制要有解除条件」那条教训的一次正面兑现:一个只写在某人记忆里的延期,
> 就是四次被继承的 §8。

## 四、一条拒绝必须搬回缝前,而搬法是**调用**而不是重写

`FlaggedNoise.std` 用**本包的措辞和异常类**拒绝一个形状不对的 flags 数组。
非图出口都是通过调 `std` 撞上它的;**图出口不调 `std`**——它读 `noise.flags` 去做
节点的 `mask`,于是形状不符落到远端,那里拒得对,但说的是 `observed_mask`
(调用方从未用过的词),穿的是 `ParameterSpaceError`(本包承诺的是
`StateValidationError`)。

搬回来的方式是**在 `jax.eval_shape` 下调用那条已经存在的检查**:形状比较是一个普通
的 Python `if`,照样触发,而什么都不算。**一句话、一个类、一处写下。**
`to_graph` 同批受益——它有同样的洞。

## 五、变异集:5 条 3 杀,**两条幸存都是构造上不可能被杀的**

| # | 变异 | 仓 | 判决 |
|---|---|---|---|
| S1 | 块按声明序而非 sorted 索取 | rheplicant | **SURVIVED,构造上必然** |
| S2 | `depends_on_prediction` 传 `False` | rheplicant | **SURVIVED,构造上必然** |
| S3 | 不再加上延期的先验曲率 | rheplicant | KILLED(10 红) |
| S4 | 拿掉缝前的 flags 检查 | rheplicant | KILLED(1 红) |
| S5 | 远端设计矩阵不再被精度加权 | **bayesmith** | KILLED(12 红) |

基线前后各一次绿。**S5 是跨仓击杀。**

### 两条幸存,逐条追到底

**S1**:`_named_spans` 的名字**永远已经是排好序的**——jax 的 dict 展平按键排序。
实测:`{"zeta":…, "mu":…, "alpha":…}` 出来是 `("alpha","mu","zeta")`。所以
`sorted()` 今天是个 **no-op**,杀不掉。

**但这条追下去有收获**:那个不变量属于 **jax**,不属于本仓,而**没有任何东西断言它**。
补两条:一条直接钉 `_named_spans` 对乱序 dict 返回排好序的名字与连续的 span;
一条端到端,在一个两种顺序**看得出差别**的模型上钉住第 0 行是谁的。
`sorted()` 保留——两个顺序不能自由漂移,而漂移会是静默的:每个数都对,每一行都错位。

**S2**:远端的 `depends_on_prediction` **只管「要不要一条规则」,不管「加不加那一项」**
(它的契约页 §5(1) 逐字这么写)。本门面**总是**传 `precision_of`,所以那一项照加。
实测:传 `True` 与传 `False` 的矩阵**逐比特相同**。所以这个参数在本路径上是**惰性的**。

那一项本身**是**被钉住的——`test_prediction_dependent_noise_adds_the_variance_term`
钉的就是 `(1 + 2 f^2)` 那个因子;S2 只是够不到它。

> 与 `priors` 批的 P3 同一形状:**一条必须幸存的变异**。但这两条各带出一件该做的事
> (S1 → 补不变量守卫;S2 → 把「这个参数是惰性的」写进代码而不是留给下一位重新发现),
> 所以「追到底」仍然付了钱。

## 六、D29:天花板的范围已实测,**下一步落地**

远端 `parameter_covariance` 有 `max_condition="auto"`(从 dtype 推 `1/sqrt(eps)`:
float32 **2.90e3**、float64 **6.71e7**);本包**没有门**。D9 把「逐 fixture 冒烟」
列为功课,本步把它量掉:

把天花板临时装进本包的 `parameter_covariance`,跑
`tests/inference` + `tests/config` + `tests/evidence`:**65 次求逆,3 次会被拒**,
全是 float32,κ = **1.0e4 / 6.5e5 / 3.2e6**:

| 测试 | κ |
|---|---|
| `test_config_exits_conjugate.py::TestWidth::test_width_fisher_over_a_whole_multi_latent_space_is_allowed` | 6.5e5 |
| `test_noise_std_axis.py::TestFisherInformation::test_the_two_explicit_readings_give_visibly_different_answers` | 1.0e4 |
| `test_fisher_prior.py::TestPriorEntersTheMatrix::test_tightening_the_prior_tightens_the_error_bar` | 3.2e6 |

**裁决:收下天花板**(D29)。那三条今天拿到的是**静默错误的**数字,而一个错了却
不说的 Cramér–Rao 界比不给更糟——这正是 D9 把它列为功课的理由。三条按分诊第二列
改写,**落地在下一步**。

## 七、铁律 4 四件套(本步)

| | 项 | 结果 |
|---|---|---|
| (i) | 该批测试全绿 | rheplicant **10097 passed / 534 skipped** exit 0(359.0 s)加 **31 passed / 1 xfailed** exit 0(x64 seam,49.8 s);bayesmith **1269 passed** exit 0(205.0 s) |
| (ii) | 接缝变异红 | 5 条 **3 杀**,两条幸存构造上必然、已逐条追到底并各补了一件事(§五) |
| (iii) | 旧实现删除、计数守卫刷新 | 雅可比 / 加权 / `_log_sigma_curvature` / `inverse_variance` 导入删除;README 计数 10649 → **10651**;拒绝普查未动 |
| (iv) | 文档实测数字重测 | 上述;**G15** 与 **D29** 入簿,登记簿标题到 **D7–D29** |

## 八、留给下一步(仍是 `uncertainty`)

1. **`parameter_covariance`**:委托 + 收下天花板(D29 已裁决,范围已量,三条测试
   已点名)。
2. **`propagate_covariance`**:远端是 `propagate_covariance(graph, covariance, at,
   node=)`,本包是 `(forward, params, param_cov)`。**先量再决定**,大概率与
   `predict_from_samples` 同一结论。
3. **`push_forward` 留守**:它是 `jax.vmap(forward)`,里面没有任何贝叶斯数值
   ——与 `predict_from_samples` 同一理由。
4. **cross-check 的处置**:照 `numpyro_bridge` 的先例**逐条读**,不要照清单退役。
   `uncertainty` 大概率也只切一半(`FlatMatrix` 永久保持、`as_noise_model` 留守、
   `_named_spans` 随 Wave C/D)。
5. **一条被交接页点名的 oracle,状态变了但**没有**死——量过才知道。**
   `test_a_vector_latent_permutes_by_ITS_SPAN_and_not_as_one_row` 拿
   `fisher_information` 当回归 oracle,而它**本步之后也是门面了**,所以作为
   **数值** oracle 它确实变成了「本包与本包比」。

   **但它作为布局断言仍然能失败**,而这是量出来的而不是推出来的:把 `priors` 批的
   P2 变异(按名字而非按 span 置换)重新装上再跑,**那条测试照样红**。原因是两侧
   到达同一个布局的**路线不同**——`priors` 是拿到 `over` 序的矩阵再置换,
   `fisher_information` 是直接按 sorted 序索取块——所以置换里的错误仍然被抓。

   交接页上一版把这条写成「必须同批改写,否则 D24 失去唯一对照」。**那句话现在
   过度了**:失去的是数值对照,布局对照还在。**下一步仍应重看它**(`uncertainty`
   全切之后两条路线可能合并),但它不是一条已经死掉的守卫。
