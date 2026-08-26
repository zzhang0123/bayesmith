# 全面移交:rheplicant 的贝叶斯层归于 bayesmith(一份实现计划)

> **状态:v3,定稿。** v1 → 七视角对抗评审(~60 发现,8 BLOCKER)→ v2 →
> 三视角新鲜验证(自洽 / 语义 / 可执行性,~20 发现,2 BLOCKER)→ v3。
> 主要裁决项标注了来源发现;两轮评审的完整结构化结果存于会话工作流记录。
> **日期**:2026-08-26
>
> v2→v3 的两个决定性修正,点名以防回退:(一)v2 的 D14 前提为假——
> bayesmith **已有**多块分区执行器(`dispatch.factor.sample_factors` 对
> `FactorPlan` 逐块扫描,`Block` 可手工构造,docstring 明邀外给分区);
> G10 因此是**扩展它**,另起炉灶将在 bayesmith 内部违反本计划自己的一份
> 实现法则。(二)v2 的 G12 选址不当——「冻结在块当前值」的语义在
> `sample_factors` 的 rebuild 分支**已经存在**(`precision_at(source,
> current)`,current 含块自身最新值);G12 移址,且该模式对 `gcr+mh` 必须
> 构造期拒绝(x-免签名正是 `_mh_step` 正确性的执行机制,实测违反代价是
> -16σ 的均值漂移)。
>
> **裁决来源**:owner 于 2026-08-26 亲自引用并触发了
> `2026-08-24-rheplicant-migration.md` §六 步骤 2 写明的推翻条件——
> 「bayesmith 出现第二个消费者」。rheplicant 已于同日成为 bayesmith 的第一
> 个包级消费者(**注意:该形态今日只在两个工作树里,尚未提交**——P0 第一
> 批动作落盘)。owner 的原话:「bayesmith 是通用贝叶斯工具,rheplicant
> 应该完全可以把所有贝叶斯任务移交给 bayesmith……适配器(Pipeline →
> Graph)是个好方案。还没迁移的都要迁移。」本文取代旧 spec §六 步骤 2 的
> 「两个包各自成立」形态,但继承促成那次重写的两条实测发现——它们由本计
> 划**正面解决**而非推翻。「一条决策一个家」:本程序全部裁决住本文。

---

## 〇、终局形态

- **bayesmith**:通用贝叶斯推断包,持有**全部**数值算法。零射电天文。
- **rheplicant.inference** 由五种东西组成:
  1. **声明层**:`ParameterSpace`/`Latent`/`Bind`(parameters.py)与
     `forward.py`(`build_forward_fn`,三条已发布跨仓契约之一,适配器
     消费的既有下半)——不迁移、不删除;
  2. **噪声物理**:`RadiometerNoise` 等;对图的呈现是分布工厂;
  3. **适配器**(`graph_bridge.py`):构图 + 图缝前验证 + 错误翻译;
  4. **探针层**(D16/D17 取留守侧时):`check_linearity` 的探测机器、
     `partition.py`(auto_blocks 的现居所)、`loglinear.py` 的 Pipeline
     探针——检查非求解,其 `jax.linearize` 用途在机房守卫中按文件豁免;
  5. **薄包装**:公开 API、异常类身份、被钉文案、**产品容器字段布局**
     (config/products 逐字段读取的名字;`FlatMatrix` 因 config 不迁移而
     为**永久**保持面)。**逐参数显式转发,禁止 `**kwargs` 直通**;
     bayesmith 独有能力(circulant、discrete、SNIS 旋钮)经门面不可达,
     开放走登记簿。`loglinear.py` 已是此形态(算术委托 bayesmith)。
- **机械验收**(`test_engine_room.py`,规格 §六):AST 级、双向允许名单;
  名单枚举:`graph_bridge.py`、`linear.py` 探针段、`partition.py`、
  `loglinear.py` 探针段。
- `rheplicant.config`/GUI 语法留守(D1(a) 仍立);**18 个 run kind**
  (`runs.py` `_KINDS` 实数,14 个触及 inference)全入冒烟矩阵——
  `predict`/`benchmark`/`compare` 消费他 kind 的 product,字段一变先红
  在它们身上。

## 一、铁律

1. **适配器与探针层是仅有的新/留实现面。** 公开名字、签名、异常类身份、
   被钉文案、产品容器字段布局保持;做不到的进登记簿。**每波开工做包内
   私名普查**(`grep -rn 'from rheplicant.inference.<mod> import' src/`):
   被未切模块 import 的名字在其最后一个消费者切换前都是保持面。今日已知:
   linear 的七个探针助手(loglinear 在用,D17 落定前不删)、`_magnitude`
   (engines 在用)、`_named_spans`(uncertainty→reduced_basis/memory,
   随 Wave C/D 退役)、`FlatMatrix`(uncertainty→config products,
   **永久**保持,归 §〇 第 5 类)。**Wave A 切 uncertainty 时容器与
   `_named_spans` 留守,文件不整删。**
2. **切换即删除,守卫同批退役**:模块的 cross-check 文件同批删除;其中
   参照非 rheplicant 的 oracle 断言**逐条改籍**进 bayesmith `tests/`
   或指认既有等价物;参照是 rheplicant 的随文件退役。
3. **无声明不迁移语义。** 语义差先裁决后切换,结果由 rheplicant 测试钉住。
4. **每批次四件套**(批次 = 一个会话切换的模块集):(i) 分诊表处置后
   该批测试全绿(分诊三列:原样重放 / 改写对适配器 / 带理由退役——退役
   即裁决项);(ii) 接缝变异红,按 §六 五行协议;(iii) 旧实现删除、计数
   守卫(README 计数、coverage floor——**每批**核对)与 cross-check 计数
   pin 同批刷新;(iv) 文档实测数字重测。
5. **发布门**:任何使 rheplicant **main** 依赖新 bayesmith 表面的提交,
   之前必须有承载该表面的 bayesmith 0.2.x 发布在索引上。(今日状态合规
   的原因是 floor 尚未上 main——P0 的动作顺序因此固定:先发 0.2.0,再推
   rheplicant。)
6. **记录页计时**:遵守 docs/migration/README 既有规则——模块未到最后
   一波不写用例计数。
7. **绑定契约不在本文重述**:模块契约 = 旧 spec §四 台账行 + 其
   docs/migration 页,开工先读。

## 二、裁决登记簿(D7–D19;拍板后回填本行)

- **D7 — gradient 块两个出口的目标密度。** 差异属**块类型**(rheplicant
  plan.py 自己的警告框架):gradient 块的 sample 与 estimate 都在 GLS 味
  一侧(无 `Σ log σ`);bayesmith NUTS 走全密度,且**无** gradient-MAP
  对应物(G2 承接)。(a) 两出口采全密度(正确侧;数字重测+changelog);
  (b) 适配器复刻 GLS 味势。*建议 (a)*;Wave B 先决因此含 G2。
- **D8 — 预测依赖 σ 的共轭扫描语义。** 实测事实:`gibbs_factory` 的
  `method="gcr"` 把 σ 冻在**先验中心**(`_precision_at` 无 x 可传——该
  签名正是 `_mh_step` 正确性的执行机制),不是 rheplicant 的逐 sweep 当前
  值;而**当前值冻结语义已存在于 `sample_factors` 的 rebuild 分支**。
  选项:(i') **G12**:把该语义经声明分区路径显式暴露(见 G12 移址),
  精确复刻今日行为——注记写为**近似声明**(历史依赖核,非严格不变),
  不是正确性证明;(ii) 多块 gcr+mh(G8);(iii) NUTS;(iv) gcr+snis
  (bayesmith 的既有设计答案,目标全密度,顺带解决此路径的 D7;`Draws`
  面须决定长 `log_weights/ess/khat` 或包装重采样)。*建议:切换期 (i')
  保数值连续;(iv) 与 D7 同批裁决为语义升级项。*
- **D9 — float32 政策(diagnose 族 + 条件数天花板)。**
  `refuse_ambient_float32` 同门管 identifiability 与 prior_sensitivity;
  config 消费者:run kind `identifiability` 与检查 C13/C19(float32 主
  会话)。`parameter_covariance` 切换后新获 `1/√eps` 天花板拒绝。选项:
  (a) 适配器局部 x64——**必须上下文内重建图并 cast 摄入数组**(仅包调用
  是 B2 定罪的 no-op),拍板前在该配置下重跑谱隙表;(b) bayesmith 放宽为
  dtype 推导 rtol(fisher 天花板先例);(c) 保留 rheplicant 实现(违背
  一份实现)。*建议 (b) 主、(a) 兜底*;C19 四 gate 模式与新拒绝逐
  fixture 冒烟;天花板拒绝按「接受为修正」入铁律 4(iv)。
- **D10 — NPE 迁移(`bayesmith.amortize`)。** 前提已定:**B4 已修**
  (e-RHINO `d499171`,simulate_pairs 以 `noise.realise` 生成、`std()`
  仅判 flagged——实测核实)。剩两个子裁决:(2) 生成器忠实性——§三噪声
  映射是**密度侧**;simulate_pairs 迁移后*建议*继续用 rheplicant 的
  `NoiseModel.realise`(噪声物理本就留守),graph dist_fn 不承担生成器
  忠实义务;(3) 薄包装保持三名与 `NeuralPosterior` 的 `__all__` 导出
  (config surface 测试钉住)。
- **D11 — calibrate 与 `InferencePlan.estimate`。** *建议迁为
  `bayesmith.fit`(G2)*,联合 MAP 与块坐标并存;loss 方向守卫随迁,
  `test_loss_sense` 经包装重放。
- **D12 — 证据族 API 与在盘数据。** rheplicant 容器保持自有类
  (`__check_init__` 异常身份在构造期,基类先行,**子类化无法翻译**),
  委托在算术调用处逐调用互转。**前置**:切换前用今日代码写出并**提交**
  读档 fixture(`.eqx`+manifest,`template_projections` 有/无两形态),
  Wave D 读档回归以提交文件为输入,x64 会话中逐字段断言。
- **D13 — 发布列车。** P0 即发 bayesmith 0.2.0(表面已在;机制:
  `publish.yml` 门 tag==pyproject 版本、测试构建轮;动作序列见 §九 P0);
  0.3.0 承载 P2;程序结束前 rheplicant 发版清 385 提交旧账。
- **D14 — 分区执行面的完形(G10 的范围)。**(v2 前提已修正。)
  bayesmith **已有**执行器:`sample_factors(graph, plan, key)` 逐块扫描
  `FactorPlan`,块可手工构造。缺的是三件:(i) **每 sweep 诊断钩子**
  (联合 χ² 轨迹、`each_sweep` identifiability、块残差——喂
  `PlanDiagnostics`);(ii) **sweep 形 estimate**(块坐标下降;今日
  `run_estimate` 拒绝部分精确图);(iii) **声明分区入口**(绕过
  factor_partition 的探测与 movement 政策门,接受外给块表,伴随文档化的
  「你声明你负责」语义)。G10 = 在 `sample_factors` 上补齐这三件,
  **不另起执行器**。
- **D15 — `condition_estimate` 与 `condition` kind。** (a) 依一份实现
  裁决重访 conditioning.md 的拒绝,移植为显式标注「measured-κ,不可作
  守卫」的诊断(**G14**);(b) kind 换 condition_bound 语义(数字动,
  `iterations:` 旋钮失对象);(c) 机房守卫挂名豁免。*建议 (a)*。
- **D16 — `check_linearity` 探针契约的家。** 两边探针语义**五轴**不同:
  锚点(max|init| vs 先验宽)、at 点数(1 vs 3 含先验抽取)、判据数
  (单 vs 双含 Unresolved)、返回/异常形、**聚合粒度**(整输出 max vs
  逐元素逐列 floor——bayesmith 实测过整域 max 放行逐元素拒绝的假声明,
  1e17 亮叶旁的 1e-2 假 linear_in 报 2.57e-14 通过)。选项:(a) 探针
  机器留守(豁免其 linearize);(b) bayesmith 加兼容旋钮;(c) 采纳强
  判据重钉 fixtures(部分今日通过者将新拒)。*建议 (a) 切换期、(c)
  升级项。*
- **D17 — 分区/对数发现探针(auto_blocks)的家。** Pipeline 探针 vs 图侧
  探针。**裁决协议**:同 fixture 集(含极端 f、边界仿射、及 D16 第五轴的
  亮暗混合模型)双跑 diff 判决,逐例一致才换,不一致逐个裁决。
- **D18 — G9(复数域)的家,P1 开工前拍板。** (a) bayesmith exact 原生
  支持复 latent(移植 `_real_parts` 拆分与实内积伴随约定及两半恒等式
  测试);(b) 适配器侧 re/im 重参数化为两个实节点 + det 重组,bayesmith
  不动。*建议 (a)*——复数是通用贝叶斯能力,不是仪器特性。
  **【owner 已拍板 2026-08-26:取 (a)。】** 复数域是 bayesmith 的原生
  能力。P2a 移植 `_real_parts` 拆分、实内积伴随约定与两半恒等式测试;
  适配器**不**承担复→实的重参数化,§三 构图表的「复数 `Latent`」行因此
  直通 `sample(name, prior)`,不再是先决占位。
- **D19 — `iterative_gls` 退化起点,P1 例 5 运行前拍板。** bayesmith 从
  **先验中心 σ** 起步;零先验中心 + `RadiometerNoise(floor=0)` 时起点
  σ=0、首解退化——rheplicant 从数据起步无此问题。选项:数据锚定起步 /
  要求 floor>0 / 显式拒绝。**例 5 的零中心子例在本项拍板前不触发 R7**
  (它是裁决 fixture,不是回归 fixture)。
  **【owner 已拍板 2026-08-26:取数据锚定起步。】** bayesmith 的
  `iterative_gls` 起点改为由数据估计 σ,复刻 rheplicant 今日行为;不取
  「要求 floor>0」,因为那会新拒今日通过的 fixture,而退化的来路是起点、
  不是配置。三个后果,写明以免各自漂移:(1) 实现落在 Wave B 的 `gls`
  工作里,不在 P1——P1 例 5 只**验收**它;(2) bayesmith 侧既有 gls 测试
  的数字随实现重测(铁律 4(iv)),起点变了的行按新值重钉,不放宽容差;
  (3) 零中心 + `floor=0` 子例**从此是回归 fixture**,R7 对它恢复生效。
  旧 spec §四 `gls.py` 行的「起点若改变,用不动点的起点无关性证等价」在
  本裁决下不必动用——两侧起点自此同源。

## 三、P1 — 适配器基石

`rheplicant/inference/graph_bridge.py`:

```python
def to_graph(space, pipeline, state_template, observed, noise,
             *, priors: dict | None = None) -> Graph
translate(site: str)   # 上下文管理器
```

**`translate` 的形态(定死,不留发明空间)**:上下文管理器,包住每次
bayesmith 调用(**trace 外**):`GraphError→ParameterSpaceError`(附录 B
行的原文案重述,`raise … from e`);仿射类 `StructureError`(读 G11 载荷)
`→LinearityRefused`;`NotGaussian/NotLogLinear` 捕获不外抛。

**先验覆盖单入口**:薄包装把调用点 `prior_mean=/prior_std=` 经
`_reconcile` 同款矛盾检查折成 `priors={name: dist}` 传入 `to_graph`;
构图表中「合成」即指此路径,不存在第二条。

**总原则:凡图缝会抹掉证据的拒绝,住在 to_graph 前的预验证。** 首例
σ 轴歧义(`check_noise_std_axis`,47 条测试钉住——
`tests/inference/test_noise_std_axis.py` `--collect-only` 实测,另有
config 侧引用)。

构图规则:

| rheplicant 侧 | Graph 侧 | 注 |
|---|---|---|
| `Latent(name, prior=…)` | `sample(name, prior)` | |
| `Latent(prior=None)`+调用点关键字 | 经 `priors=` 单入口合成 `Normal` | bayesmith 无先验关键字,图是唯一声明 |
| 复数 `Latent` | `sample(name, prior)`,复 latent 直通 | **D18 已拍 (a)**:复数住 bayesmith;G9 最小面是 P1 先决 |
| `forward_fn` | `det("__mu__", forward, *latents, linear_in=…)` | 内部名不漏进 samples 键空间(G7) |
| `HomoscedasticNoise(σ)` | `Normal(mu, σ)` | 密度侧 |
| `RadiometerNoise(Δν,τ,floor)` | `Normal(mu, f·max(\|mu\|,floor))`,dop=True | **密度侧;生成侧走 `realise` 留守(D10-2)** |
| `FlaggedNoise` | **G1 先决** | |
| `scope="per_epoch"` | `Plate` | Wave D |
| `scope="linked"` | **G3 先决** | |

**P1 交付物**:适配器 + `translate` + 预验证 + **拒绝文案清单**(census
`tests/inference` 全部 `match=` 站点→模式→今日抛出处→切换后抛出侧,入
附录 B)+ **接缝 CI 工作流的建立**(规格 §六——它从 Wave A 起就要双岗,
不能等 P4)。

**P1 验收(钉名十例,分层判据)**:

- *确定性层(float64;R7 只由本层触发)*。**CG 背书的行(1/2/3/5)在钉
  死的机器级 tol 下双跑,或在可稠密分解的小 fixture 上对 dense 参照;
  rtol ≤ 1e-12 指该配置下的收敛解**——两个正确实现可以因收敛结构在远高
  于 1e-12 处逐元素分歧,那不触发 R7,写进各行验收:
  1 线性 wiener;2 复数 alm wiener/GCR 均值(G9 最小面验收);3
  `conjugate.wiener` 带先验覆盖;4 identifiability 健康+拒绝对(x64 子
  进程会话,D9 摘出 P1 关键路径);5 GLS 不动点(零中心子例受 D19 保护;
  `iterations/delta` 排除断言);6 `plan.estimate` **限全共轭、整图形态**
  (今日 bayesmith 可执行;梯度块 estimate 变体移 Wave B 验收)——值到
  求解容差,诊断字段除外。
- *抽样层(固定 key,矩对 dense oracle;ESS ≥ 400,|z| < 4,均值与协方
  差)*:7 GCR 单次路径(n_draws=1 调用,**≥400 独立 key 聚成矩**——与
  例 8 的对比即标量路径 vs vmap 路径);8 GCR 经 config 的 `jax.vmap`
  `n_draws > 1` 全链路(vmap 可 trace 为硬约束,校验与翻译在 Python 侧);
  9 log 空间 GCR;10 `plan.sample` 小分区——**限今日 `sample_factors`
  可执行形态(显式 FactorPlan)**,诊断字段除外;含梯度块与每 sweep 诊断
  的完整形态是 Wave B 验收。

## 四、P2 — bayesmith 能力缺口

- **G1 掩码/旗标**:观测掩码贯通 exact/precision(inf-σ = 零权)。
- **G2 `bayesmith.fit`**:联合 MAP(Adam/朴素梯度)+ loss 方向守卫;
  承接 D7 的 gradient-MAP 出口。
- **G3 `exact.chain`**:RTS/Kalman + `linked` 转移;自含于
  bayesmith.evidence.sqrtinfo 之上(chain→sqrtinfo 依赖边,故与证据族
  同波)。
- **G4 `exact.reduced_basis`**。
- **G5 `bayesmith.amortize`**:NPE 族(D10 子裁决先决)。
- **G6 证据消费面**:D12 包装所需缺口,逐项登记。
- **G7 bridge 补齐**:`init_to_declared`、`predict_from_samples`、
  `propagate_covariance`/`push_forward`(config `predict` kind 逐 fisher
  product 调用);约定:单参数 `model(observed=None)`、latent 同名 site、
  内部节点名不漏出。
- **G8(后置)**:多块 gcr+mh 论证与实现。
- **G9 复数域**(家由 D18 定)。**最小面(P1 先决,定义)**:
  `wiener_solve`/`gcr_sample` 接受复 latent,均值路径经 `_real_parts`
  约定与两半恒等式测试;矩验收随 P1 例 2。全量(vmap、log 空间、Fisher
  的复数面)随 Wave B。
- **G10 分区执行面完形**(D14 范围):在 `sample_factors` 上加
  (i) per-sweep 回调(χ² 轨迹、identifiability 节奏、块残差),
  (ii) sweep 形 estimate,(iii) 声明分区入口(绕探测与政策门,文档化
  「声明者负责」)。**不另起执行器。**
- **G11 结构化拒绝载荷**:仿射类异常带 errors=/rtol=/failed;
  NotGaussian/NotLogLinear 带判别字段。先于任何委托检查的波次。
- **G12 冻结在当前值的 gcr(移址)**:经 G10(iii) 的声明分区路径暴露
  `sample_factors` rebuild 分支的既有语义;若保留 gibbs.py 侧模式,对
  `method="gcr+mh"` **构造期拒绝**;注记为近似声明,非正确性证明。
- **G13 图级联合先验**:`JeffreysPrior(over=…)` 的图侧声明与
  `to_numpyro` factor site 读取。Wave A 的 priors/numpyro_bridge 之门。
- **G14 measured-κ 诊断**(D15(a)):`condition_estimate` 的对应物,
  显式标注不可作守卫;随 Wave B 的 linear 工作落地。

每 G 项 = 实现 + 独立 oracle 测试 + §四式记录页(铁律 6 计时)。

## 五、P3 — 逐模块切换

> 先决列语义(全文统一):**先决 = 该波首个受其约束的模块切换前完成;
> 标「(+X 实现)」的项在波首会话内完成;「D17 若换探针」为条件先决。**
> 每波开工:铁律 1 私名普查 + 铁律 7 契约阅读 + 测试分诊表。分诊表种子:
> `test_plan_compiles_once.py`(→R3 门)、`test_conjugate_transition.py`
> (程序缓存键)、`test_block_learning_rate.py`(→G2 API 要求)、
> `test_magnitude_is_build_time.py`、`test_inference_construction_guards.py`
> 部分行。

- **Wave A(检查与报告面;先决 P1+D9+D16+G7+G11+G13)**:
  `identifiability`、`sensitivity`、`priors`、`numpyro_bridge`、
  `uncertainty` 全模块(propagate/push_forward 经 G7;`as_noise_model`
  留守;容器与 `_named_spans` 留守,文件不整删;`parameter_covariance`
  新拒绝按 D9 附带处理)。
- **Wave B(求解与计划;先决 D7+D8+D14+G1+**G2**+G9+G10+G12
  (+D17 若换探针))**:`linear` 求解面、`gls`(D19 已拍)、
  `plan`+`engines`(验证与词汇留守,执行经 G10)、`noise`/`likelihood`
  工厂化。linear 探针助手在 D17 落定前不删。config 侧钉内部件的三个测试
  (`MIN_DRAWS` 等常量、`SamplingPlan` 源文本 pin)本波显式重谈,走登记
  簿。G14 随本波。P1 例 6/10 的完整形态(梯度块、每 sweep 诊断)在本波
  验收。
- **Wave C(先决 G2+G4+G5+R2 清单——`reduced_basis` 测试族在
  tests/evidence 的 x64 会话里)**:`calibrate`、`npe`、`reduced_basis`。
- **Wave D(先决 D12+G3+G6)**:`chain` + 证据族七模块。会话合并按 R2
  清单执行。
- **已成形态**:`partition.py` 与 `loglinear.py` 今日已是门面形态
  (§〇 第 4/5 类),D17 裁决其探针的最终家,无单独切换波。

**适配器不经 `dispatch.classify`**:分块由 SamplingPlan/auto_blocks 决定,
逐块显式映射到 G10 执行面;bayesmith 独有路径经门面不可达。

## 六、P4 — 质量机制换防

- **oracle 改籍**(铁律 2 细则)与 **双岗**:接缝 CI 与 crosscheck 是
  **两个并行 workflow**(接缝 CI 于 **P1 建立**,住 bayesmith 侧
  crosscheck.yml 旁),两者并跑直到最后一个 cross-check 文件退役;
  cross-check 文件与其模块同批删除。
- **接缝 CI 机械规格**:`actions/checkout` e-RHINO@main(打印 commit)→
  `pip install --no-deps -e` + bayesmith `-e .` + **全部 importorskip
  依赖(h5py、rhino-cal-jax by URL)** → `tests/inference --junit-xml` →
  XML 读数;**通过地板 = 上次绿跑的 `tests - skipped` 记录值**,按理由的
  skip 允许名单(rhino-cal-jax 缺席吞掉哪些行,名单写明)。
- **接缝变异五行协议**:(1) scratch clone 变异,`git checkout` 恢复,
  不用 `cp`;(2) 变异与恢复间 `rm -rf __pycache__`;(3) 击杀 = 退出码
  **恰为 1** 且 junit 中**指名测试**在红名单(2/4/5/143 记「未运行」并
  打红 job);(4) 变异集前后各一次基线绿;(5) 变异集是登记清单
  (附录 A 起步),每波扩充。
- **`test_engine_room.py`**:AST 解析、import 别名解析、FORBIDDEN 集
  (§〇)、允许名单逐文件列许可子集(§〇 第 3/4 类四个文件)、双向断言
  (名单外含禁 token 红;名单内不再含其许可 token 也红)。
- **R3 门 = 编译计数**:适配路径 XLA 编译数 O(blocks) 非 O(sweeps),
  用 `_count_compiles` 模式;墙钟 ≤1.2× 仅次级。

## 七、P5–P7

- **P5**:18 kind 逐 `_KINDS` 冒烟;每批四件套附 extractor 往返。
  `optimize` 随 D11;`condition` 随 D15。
- **P6**:0.2.0(P0)→ 0.3.0(P2 完)→ rheplicant floor 与发版(D13)。
- **P7 具名清单**:(a) e-RHINO CLAUDE.md **与** AGENTS.md 成对改(两会话
  段、bayesmith-floor 段、完整环境段);(b) README 计数 pin;(c) 覆盖率
  截断值+`fail_under`(每批核对);(d) `_migration-to-bayesmith.md` 整篇
  (门面与机房);(e) `rheplicant/inference/__init__.py` 模块 docstring
  (今日三论点在终局全假,而它是 106 名单门户);(f)
  `tests/test_published_contracts.py`(27 条实测)与其 docstring;
  (g) bayesmith README/CHANGELOG。

## 八、风险登记

- **R1 文档数字**:铁律 4(iv);范围含 D7/D8/D9/D15 的全部页面。
- **R2 x64 会话(清单项,非调查)**:测量已在
  `tests/test_evidence_session.py:66-80`——合并 = 删该文件与那个
  conftest,永不全套 x64;22 个 float32-only 拒绝中 16 个住 core/radio,
  不迁移。清单:Wave C 前确认 reduced_basis/chain 测试族会话归属;
  Wave D 末按既定条件执行删除。
- **R3 性能**:编译计数主门;vmap/jit 可 trace 硬约束(P1 例 8)。
- **R4 拒绝文案**:附录 B 清单驱动,逐波核销。
- **R5 published contracts**:`tests/test_published_contracts.py`
  (27 条),P7(f)。
- **R6 循环依赖**:运行时单向,无环。
- **R7 计划失效条件**:P1 确定性层在**钉死求解容差/dense 参照**配置下
  出现无法裁决的分歧 → 回 owner;抽样层、诊断字段、D19 保护中的子例
  不触发。
- **R8 门面拓宽**:铁律 1 禁 `**kwargs` + 机房守卫双向名单。

## 九、排期(会话计)

| 阶段 | 会话数(估) | 先决 |
|---|---|---|
| **P0(有序清单)**:(1) bayesmith 提交推送(清单见附录 C)→ (2) `git tag v0.2.0 && push`,publish.yml 走完,**确认 0.2.0 上索引** → (3) e-RHINO 提交推送(floor 此刻才合规上 main)→ (4) `git ls-remote` 双仓核实 → 本文定稿提交 | 1 | 评审完 |
| **P2a(先于 P1)**:G9 最小面 + G11 | 1–2 | ~~D18~~ 已拍 (a) |
| P1 适配器 + 十例 + 文案清单 + 接缝 CI 建立 | 2–3 | P2a、~~D19~~ 已拍 |
| P2 余项:G1/G2/G7/G9 全量/G10/G12/G13/G14 | 7–11 | 各 D 项 |
| P3 Wave A | 2–3 | P1+D9+D16+G7+G11+G13 |
| P3 Wave B | 3–5 | D7/D8/D14(+D17 若换探针)+G1/G2/G9/G10/G12 |
| P3 Wave C(+G4/G5 实现) | 3–4 | G2/G4/G5+R2 清单 |
| P3 Wave D(+G3/G6 实现) | 4–6 | D12+G3+G6 |
| P4–P7 收尾 | 3–4 | 各波 |
| **合计** | **约 27–40** | |

## 十、本计划自身的验收与记录纪律

- 旧 spec §六 推翻条件行已加注(随 P0 提交)。
- 每 D 项拍板后回填;每波证据链写进本目录 tracked 执行页
  (`2026-XX-XX-wave-X.md`;e-RHINO 的 docs/superpowers 是 gitignored,
  八份计划死在那里过——执行页只住本目录)。

---

## 附录 A — 接缝变异探针(起始清单,2026-08-26 会话手工示范)

1. **first_fit**:bayesmith `dispatch/factor.py`
   `if all(compatible(name, member) ...)` → `if True:`;指名红:e-RHINO
   `test_auto_partition.py::TestTheMultilinearSplit::test_the_coupled_factor_gets_a_block_of_its_own`
   与 `::test_the_derived_partition_recovers_the_truth`;观测退出码 1。
2. **log 位移**:bayesmith `exact/loglinear.py`
   `y = jnp.log(observed) + fractional**2 / 2.0` → 去位移;指名红:
   e-RHINO `test_loglinear.py::TestTheNoiseTransform::test_the_leading_order_mean_shift_is_added_back`;
   观测退出码 1。

## 附录 B — 拒绝文案清单

P1 交付物,census 后回填(格式:match= 模式 | 今日抛出处 | 切换后抛出侧)。

## 附录 C — P0 提交清单

- **bayesmith**:全部已跟踪修改 + 新文件
  {`src/bayesmith/dispatch/factor.py`, `src/bayesmith/exact/loglinear.py`,
  `tests/dispatch/test_factor.py`, `tests/exact/test_loglinear.py`,
  `tests/test_examples.py`, `examples/`(**五**文件,见下),
  `docs/factor-partition-examples.md`, 本 spec 与旧 spec 加注};
  **排除** `run.log`/`run.exit`/`dist/`。
  - **`examples/` 是五个文件,不是六个。** 实测 `find examples -type f
    -not -path '*__pycache__*'`:`README.md`、`models.py`、
    `three_routes.py`、`hierarchy.py`、`validate_sampling.py`。定稿时把
    `tests/test_examples.py` 一并数了进去,而它住 `tests/`。改正而非沿用
    ——一个没人核过的计数正是本程序反复付学费的形状。
  - **`AGENTS.md`:【owner 已拍板 2026-08-26:删除,不入库。】** 它当时
    与 CLAUDE.md 逐字节一致,而 CLAUDE.md 正文写着「刻意无第二份」——
    文件因存在而使那句话为假。删除让声明重新为真,且不欠一个没人写的
    一致性测试。bayesmith 侧不设 `AGENTS.md`;e-RHINO 侧两份继续由
    `tests/test_docs_claims.py` 钉着。两仓规矩不同是**有意**的,各自的
    理由写在各自的 CLAUDE.md 里。
- **e-RHINO**:全部已跟踪修改 + 新文件
  {`src/rheplicant/inference/partition.py`,
  `src/rheplicant/inference/loglinear.py`,
  `tests/inference/test_auto_partition.py`,
  `tests/inference/test_loglinear.py`};**排除**根目录八份未跟踪评审/交接
  草稿(另行裁决去留,不随本程序入库)。
