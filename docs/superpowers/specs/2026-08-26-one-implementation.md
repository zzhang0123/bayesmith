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

## 二、裁决登记簿(D7–D54;拍板后回填本行)

- **D7 — gradient 块两个出口的目标密度。** 差异属**块类型**(rheplicant
  plan.py 自己的警告框架):gradient 块的 sample 与 estimate 都在 GLS 味
  一侧(无 `Σ log σ`);bayesmith NUTS 走全密度,且**无** gradient-MAP
  对应物(G2 承接)。(a) 两出口采全密度(正确侧;数字重测+changelog);
  (b) 适配器复刻 GLS 味势。*建议 (a)*;Wave B 先决因此含 G2。
  **【owner 已拍板 2026-08-26:取 (a),两出口采全密度。】** gradient 块的
  sample 与 estimate 都带 `Σ log σ`。三个后果:(1) rheplicant 侧钉住这两个
  出口的数字**全部重测**并写 changelog——按铁律 4(iv) 的「接受为修正」,不
  放宽容差;(2) Wave B 的先决因此**确实含 G2**(`bayesmith.fit` 承接
  gradient-MAP 出口,今日 bayesmith 无对应物);(3) 与 D8 的 (iv) 同批落地
  ——两者都是「目标密度改为全密度」的同一件事,分开做会让同一批数字重测两次。
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
  **【owner 已拍板 2026-08-26:分期,先 (i') 后 (iv)。】** 切换期走 G12,
  经声明分区路径暴露 `sample_factors` rebuild 分支的既有语义,精确复刻今日
  行为,保数值连续;注记**必须**写成「近似声明:历史依赖核,非严格不变」,
  不得写成正确性证明。随后 (iv) gcr+snis 作为语义升级项**与 D7 同批**落地
  ——目标全密度,顺带解决此路径的 D7。升级批开工前需回答一个未决子项:
  `Draws` 面是长出 `log_weights/ess/khat`,还是包装重采样。**分期的退场
  条件写在这里,以免「切换期」变成永久**:(i') 的近似声明在 (iv) 落地当批
  撤除;若 (iv) 被推迟,推迟本身进登记簿而不是留在这一行。
- **D9 — float32 政策(diagnose 族 + 条件数天花板)。**
  `refuse_ambient_float32` 同门管 identifiability 与 prior_sensitivity;
  config 消费者:run kind `identifiability` 与检查 C13/C19(float32 主
  会话)。`parameter_covariance` 切换后新获 `1/√eps` 天花板拒绝。选项:
  (a) 适配器局部 x64——**必须上下文内重建图并 cast 摄入数组**(仅包调用
  是 B2 定罪的 no-op),拍板前在该配置下重跑谱隙表;(b) bayesmith 放宽为
  dtype 推导 rtol(fisher 天花板先例);(c) 保留 rheplicant 实现(违背
  一份实现)。*建议 (b) 主、(a) 兜底*;C19 四 gate 模式与新拒绝逐
  fixture 冒烟;天花板拒绝按「接受为修正」入铁律 4(iv)。
  **【owner 已授权委托,2026-08-27:取 (b) 主、(a) 兜底,依 2026-08-27 的
  一次性委托。】** 读成 (b) 的理由:float32 政策的争点是**一条 rtol 的来路**,
  不是**谁算这件事**。(b) 让 bayesmith 从 dtype 推导 rtol——它已经为 fisher
  天花板这么做过,所以这是**扩展一条既有惯例**而不是新开一条;(a) 则要求适配器
  在上下文内重建图并 cast 摄入数组,而那正是 B2 定罪过的形状:**只包一层调用是
  no-op**,做对的代价是把「哪些数组进过 cast」变成适配器的长期义务。**兜底的
  退场条件写在这里,以免它变成主路**:(a) 只在 Wave A 实测出「dtype 推导的 rtol
  仍拒绝一个今日通过的 C13/C19 fixture」时启用,且启用当批必须附上那次实测。
  拍板不豁免该行原有的两项功课:C19 四 gate 模式与新天花板拒绝逐 fixture 冒烟,
  谱隙表在最终采用的配置下重跑。
  **【实测修正,2026-08-27:谱隙表跑完,(b) 不成立;改取 (a),refuse 保持。】**
  这是本次委托下的**第二次事实修正**(第一次是 D13),按「委托不是空白支票」那条
  规则处置:照建议做与实测冲突,按事实选,冲突写进本行。
  证据:`docs/probes/probe_13_d9_precision_policy.py`——两分量幂律,`delta` 把真实
  条件数拉过**十个量级**,对每一个候选容差(`1e-8` 到 float32 的 `sqrt(eps)`=3.45e-4)
  问它能否复现 float64 的判决。**没有一个能。**
  **原因不是「切点难放」,而是被切的那个量没了**:float64 的最小奇异值跟着模型走
  (5.1e-6 → 5.2e-16),float32 的从第二个量级起就坐在自己的舍入地板 ~1e-7 上,并且
  **非单调**——那是噪声。float64 分得开两个量级的两个模型,在 float32 里回来时无法
  分辨,所以没有任何切点能跟上它们:推导的、调过的、逐模型选的,都不行。
  **(b) 的类比在这里断掉,值得写清楚**:条件数天花板可以从 dtype 推导,因为它是
  「算术还剩几位数」的陈述;秩的切点不能,因为它是「谱从哪里开始不再描述模型」的
  陈述,而 float32 里谱**在任何切点之上**就已经不描述模型了。
  rheplicant 早已独立得出同一结论并写在自己的 `inference/identifiability.py` 里:
  「A per-precision retune of rtol would therefore recover *this* model. It would
  not recover one a few decades worse conditioned... Forcing float64 is what lets
  one default be right for both.」——这一行原先的 (b) 建议没有读到那段。
  **改取 (a)**,并且实测它**失败得响亮**:只把调用包进 x64 而图仍在外面建,会被
  既有的 `refuse_single_precision` 指名拒绝,不会给出一个穿着 float64 外衣的
  float32 判决。所以 (a) 的「适配器长期义务」是有守卫的义务,不是记忆力。
  **顺带查出并已修的两个洞**(找反证时找到的,不是本来要做的):
  `refuse_ambient_float32` 有**三**个调用方而本行只点了两个;`JeffreysPrior.
  information` 根本没有图侧守卫——实测同一个精确退化块给出 **-27.52** 对 **-338.05**
  的半对数行列式,310 nat 的静默误差,且在 NUTS 会取指数的那一项上;
  `prior_sensitivity` 的守卫钉在一个**会自动提升**的标量上,因而形同虚设。
  证据链:`2026-08-27-wave-P2-D9.md`。
- **D10 — NPE 迁移(`bayesmith.amortize`)。** 前提已定:**B4 已修**
  (e-RHINO `d499171`,simulate_pairs 以 `noise.realise` 生成、`std()`
  仅判 flagged——实测核实)。剩两个子裁决:(2) 生成器忠实性——§三噪声
  映射是**密度侧**;simulate_pairs 迁移后*建议*继续用 rheplicant 的
  `NoiseModel.realise`(噪声物理本就留守),graph dist_fn 不承担生成器
  忠实义务;(3) 薄包装保持三名与 `NeuralPosterior` 的 `__all__` 导出
  (config surface 测试钉住)。
  **【owner 已授权委托,2026-08-27:子裁决 (2) 续用 `NoiseModel.realise`;
  (3) 薄包装保持三名,依 2026-08-27 的一次性委托。】** 读成这两个选项的理由:
  §三 的噪声映射表整张都是**密度侧**(`Normal(mu, noise.std(mu))`),而
  `RadiometerNoise` 的生成侧与密度侧**故意不同**——`realise` 是乘性的
  `d(1+fw)`,`std` 取了绝对值且可带 floor,两者在预测过零处符号不同,该模块
  自己的 docstring 逐字写着这一点。让 graph 的 `dist_fn` 兼任生成器,等于
  静默地把生成侧换成密度侧;而噪声物理本就是留守的第 2 类。(3) 同理是铁律 1:
  三个名字与 `NeuralPosterior` 的 `__all__` 导出被 config surface 测试钉着,
  是保持面。
- **D11 — calibrate 与 `InferencePlan.estimate`。** *建议迁为
  `bayesmith.fit`(G2)*,联合 MAP 与块坐标并存;loss 方向守卫随迁,
  `test_loss_sense` 经包装重放。
  **【owner 已授权委托,2026-08-27:取「迁为 `bayesmith.fit`(G2)」,依
  2026-08-27 的一次性委托。】** 读成这个选项的理由:D7 已拍 (a),而 (a) 的
  第二个后果**逐字**写着「Wave B 的先决因此确实含 G2(`bayesmith.fit` 承接
  gradient-MAP 出口)」。G2 因此无论如何都要存在;把 calibrate 留在 rheplicant
  会造出两个梯度 MAP 实现,直接违反本计划的一份实现法则。loss 方向守卫随迁,
  `test_loss_sense` 经包装重放(该测试问的是符号,而符号是 G2 必须保持的东西)。
- **D12 — 证据族 API 与在盘数据。** rheplicant 容器保持自有类
  (`__check_init__` 异常身份在构造期,基类先行,**子类化无法翻译**),
  委托在算术调用处逐调用互转。**前置**:切换前用今日代码写出并**提交**
  读档 fixture(`.eqx`+manifest,`template_projections` 有/无两形态),
  Wave D 读档回归以提交文件为输入,x64 会话中逐字段断言。
  **【owner 已授权委托,2026-08-27:取「rheplicant 容器保持自有类,算术调用处
  逐调用互转」,依 2026-08-27 的一次性委托。】** 读成这个选项的理由是该行自己
  括号里那句实测:`__check_init__` 的异常身份在**构造期**抛,基类先行,所以
  **子类化无法翻译**——把容器换成 bayesmith 的类会在构造期抛出 bayesmith 的
  异常,而那是被钉住的保持面。**前置条件不因委托而松动**:读档 fixture
  (`.eqx`+manifest,`template_projections` 有/无两形态)必须在切换**之前**用
  今日代码写出并**提交**;Wave D 若发现该 fixture 未提交,那一波不得开工。
  **【前置已满足 2026-08-27】** 两份归档加两份 manifest 已写出并**提交**在
  e-RHINO 的 `tests/evidence/fixtures/`(生成器 `make_d12_archives.py` 与之同住,
  并在非 x64 下拒绝运行);读档回归是 `tests/evidence/test_d12_read_back.py`,
  **逐字段对着写在该文件里的值**断言而不是对着一个现场构造的 term——后者只能说明
  新写手与新读手一致,而问题是**切换后的代码还读不读得懂切换前写的字节**。
  **本行两处措辞按实测更正**:扩展名是 **`.rhep`** 不是 `.eqx`;manifest 是
  **`x.json`**(后缀**替换**)不是 `x.rhep.json`。**顺带发现并登记 D39**。
- **D13 — 发布列车。** P0 即发 bayesmith 0.2.0(表面已在;机制:
  `publish.yml` 门 tag==pyproject 版本、测试构建轮;动作序列见 §九 P0);
  0.3.0 承载 P2;程序结束前 rheplicant 发版清 385 提交旧账。
  **【owner 已授权委托,2026-08-27:发布列车照原样走,但版本号的归属**修正**:
  0.3.0 承载 **P2a**(P1 的先决表面),P2 余项完成后另发一版;依 2026-08-27 的
  一次性委托。】** **这是委托下的一次事实修正,不是照抄建议**,理由是一次实测
  与一条铁律的冲突:铁律 5 要求「任何使 rheplicant main 依赖新 bayesmith 表面
  的提交,之前必须有承载该表面的发布在索引上」,而 **P1 的适配器依赖 P2a 的两个
  表面**(`AffinityRefused` 的载荷、`ComplexNormal`),它们今天只在 Unreleased
  段里。原文「0.3.0 承载 P2」要求 P2 **全部**做完才发版,那会让 P1 无法在铁律 5
  下落地——两条规矩不可能同时满足。**为什么是 0.3.0 而不是铁律 5 字面写的
  「0.2.x」**:CHANGELOG 的 Unreleased 段**自己**写着 `Breaking`(三个异常类的
  `reason` 变必填),而 0.x 下破坏性变更属于 minor 位。把它发成 0.2.1 会让一个
  破坏性变更藏在 patch 位下,这是「静默产出错答案」的那一侧;铁律 5 关心的是
  **发布先于依赖**,版本号的选择本就是本行的职责。
  **【版本号已定 2026-08-28:那一版是 `0.5.0`;owner 于 2026-08-28 授权「你 propose
  一个新版本号,我都 OK」。】** 取 minor 而非 patch 的理由与 0.3.0 / 0.4.0 同一条:
  `Unreleased` 里**没有** `Breaking` 段(每个新名字都是加法,每个新关键字默认不变,
  没有签名改动),但两条 `Fixed` **改变了调用方可能一直在读的数值**——先验曲率广播
  (实测在 `prior_std = 2.0` 的 (3,) 块上每个非对角大 0.25)与 `compress` 的掩码缺陷
  (`information()` 有限良态而 `target` 是 NaN)。0.x 下这属于 minor 位。
  **`pyproject.toml` 与 CHANGELOG 已经改好并提交**(套件 1523 passed exit 0),
  所以发布日只剩 `git tag v0.5.0 && push` 加索引核实——**而推 tag 仍然要 owner 在场**。
- **D14 — 分区执行面的完形(G10 的范围)。**(v2 前提已修正。)
  bayesmith **已有**执行器:`sample_factors(graph, plan, key)` 逐块扫描
  `FactorPlan`,块可手工构造。缺的是三件:(i) **每 sweep 诊断钩子**
  (联合 χ² 轨迹、`each_sweep` identifiability、块残差——喂
  `PlanDiagnostics`);(ii) **sweep 形 estimate**(块坐标下降;今日
  `run_estimate` 拒绝部分精确图);(iii) **声明分区入口**(绕过
  factor_partition 的探测与 movement 政策门,接受外给块表,伴随文档化的
  「你声明你负责」语义)。G10 = 在 `sample_factors` 上补齐这三件,
  **不另起执行器**。
  **【owner 已拍板 2026-08-26:三件全做,范围如上。】** (i)(ii)(iii) 都在
  `sample_factors` 上补齐,不另起执行器,也不缩到子集。因此 P1 例 6/10 的
  **完整形态**(梯度块 estimate、每 sweep 诊断)确实在 Wave B 验收,而 P1
  只验今日 `sample_factors` 可执行的那部分——两处已如此写,此处确认。
- **D15 — `condition_estimate` 与 `condition` kind。** (a) 依一份实现
  裁决重访 conditioning.md 的拒绝,移植为显式标注「measured-κ,不可作
  守卫」的诊断(**G14**);(b) kind 换 condition_bound 语义(数字动,
  `iterations:` 旋钮失对象);(c) 机房守卫挂名豁免。*建议 (a)*。
  **【owner 已授权委托,2026-08-27:取 (a),依 2026-08-27 的一次性委托。】**
  读成 (a) 的理由:(b) 换语义会**动数字**并让 `iterations:` 旋钮失去对象,那是
  config 面的破坏;(c) 挂名豁免会在机房守卫的允许名单上留一个**没有理由的条目**,
  而那份名单刚刚才被改成**双向**断言(未用的豁免自己就是红),所以 (c) 与已落地的
  守卫形状直接冲突。(a) 把 `condition_estimate` 移植为 **G14** 的诊断,并显式
  标注「measured-κ,不可作守卫」——保住数字、保住旋钮,且把「它不是守卫」这句话
  写进代码而不是留在某个人的记忆里。
- **D16 — `check_linearity` 探针契约的家。** 两边探针语义**五轴**不同:
  锚点(max|init| vs 先验宽)、at 点数(1 vs 3 含先验抽取)、判据数
  (单 vs 双含 Unresolved)、返回/异常形、**聚合粒度**(整输出 max vs
  逐元素逐列 floor——bayesmith 实测过整域 max 放行逐元素拒绝的假声明,
  1e17 亮叶旁的 1e-2 假 linear_in 报 2.57e-14 通过)。选项:(a) 探针
  机器留守(豁免其 linearize);(b) bayesmith 加兼容旋钮;(c) 采纳强
  判据重钉 fixtures(部分今日通过者将新拒)。*建议 (a) 切换期、(c)
  升级项。*
  **【owner 已拍板 2026-08-27:逐轴而非整体表决。①锚点/②at 点数/③判据数/
  ⑤聚合粒度取 (c)——采纳强判据;④返回与异常形取 (a)——rheplicant 的公开
  形状是保持面(铁律 1),适配器负责把两层降一层,合并时取逐 scale 最坏的
  at 点。】** 探针机器**留守 rheplicant**,变的是判据不是家。四条轴分四次
  落地并**分别**量代价:聚合 1 条、锚点 2 条、at 点数 2 条、判据数 **0** 条,
  合计 5 条测试,无一是放宽判据。**轴 1 不是单调更严**——先验宽于 init 时更
  严、窄时更松,那条反方向的实测记在页上。守卫:e-RHINO
  `tests/inference/test_linearity_contract.py`(19 例,每条并写旧判决)。
  常数与 `Unresolved` 类型自 bayesmith 导入,不留第二份。
  证据与逐轴实测:`2026-08-27-d16-five-axes.md`。
- **D17 — 分区/对数发现探针(auto_blocks)的家。** Pipeline 探针 vs 图侧
  探针。**裁决协议**:同 fixture 集(含极端 f、边界仿射、及 D16 第五轴的
  亮暗混合模型)双跑 diff 判决,逐例一致才换,不一致逐个裁决。
  **【owner 已拍板 2026-08-26:先跑裁决协议,再按结果定家。】** 不直接留守
  也不直接换。协议是**先决**而非事后核对:双跑 diff 必须在 Wave B 切
  `linear`/`plan` 之前跑完,因为 D17 是那一波的条件先决(「若换探针」)。
  逐例一致才换;任何不一致都是一条新裁决,逐个上登记簿,不得整体表决。
  协议的证据链写进该波执行页,格式照 §四 五节。
  **【协议已跑完 2026-08-27,结果见 `2026-08-27-d17-protocol.md`;换家的条件
  未满足,四条分歧待逐个拍板。】** 8 例中 **4 例一致**(加性对、多线性对、
  线性+非线性、乘性 f=0.004 的对数发现)。四条分歧归成**两族**,各有一个成因:
  - **第一族(加性噪声下的对数发现、f=0.3 的对数发现)**:rheplicant 的
    **分区探针看不见噪声**——`auto_blocks` 的签名里没有噪声模型。同样的两条
    拒绝在 rheplicant **存在**,常数也一样(`FIRST_ORDER_MAX_FRACTIONAL`
    两边都是 **0.06**),只是住在 `to_log_space`,即**求解期**。实测后果:
    `auto_blocks` 对加性噪声模型产出 `log_conjugate` 块,而同一模型交给
    `to_log_space` 被 `ParameterSpaceError` 拒绝——**产出了自己的求解器会拒绝
    的分区**,正是 partition.py 模块 docstring 声明本模块要避免的事。
    选项:(a) 该轴换图侧探针;(b) 给 `auto_blocks` 传噪声,把两条拒绝提到分区
    期(判据不变,只提前);(c) 接受首个 sweep 才拒,并**撤掉** partition.py
    那句承诺。*建议 (b) 或 (a);(c) 要显式改掉一句已写下的承诺,故是裁决。*
  - **第二族(边界仿射、亮暗混合)**:是 **D16 的第一轴(锚点)与第五轴
    (聚合粒度)**,不是 D17 的新问题。*建议把这两例挂在 D16 名下,D16 拍板后
    自动结清,不在 D17 里单独表决。* 先定 D17 只会把同一个决定做两遍。
  **【D17 已结清 2026-08-27,而本行到今天为止是过期的。】** 记录页
  `2026-08-27-d17-protocol.md`(比本行只晚一分钟提交)写着 owner 已就两族拍板:
  **第一族取 (b)**——给 `auto_blocks` 传噪声,把两条拒绝提到分区期,判据与常数
  不变,**探针留守 rheplicant**;实现已落地(e-RHINO `a04410e`),不传 `noise=`
  时**不声明任何 log 块**并发 `UncheckedLogRouteWarning`,判据抽成
  `loglinear.log_route_refusal(noise)` 一个谓词供两个消费者共读,并有防漂移守卫。
  修完后 **6/8**。本行「四条分歧待逐个拍板」因此**从未成立过**,是本文与记录页
  之间的一次一分钟的过期——「一个事实两份拼写」的又一例,而这次两份只差六十秒。
  **【第二族的「自动结清」是假的,已实测,登记为 D40。】** D16 四条轴落地之后
  **重跑了该协议**(它是 tracked 且可重跑的):仍然 **6/8**,`boundary_affine` 与
  `bright_and_faint` 仍分歧。但**分歧的形状变了**,而那才是结论——见 D40。
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
  **【实测修正,2026-08-27:本行的前提不成立,退化的首解在 bayesmith 里不会发生。】**
  这是本次委托下的**第三次事实修正**(前两次是 D13 与 D9),按「委托不是空白支票」
  处置:照建议做与实测冲突,按事实选,冲突写进本行。
  证据:`tests/exact/test_gls.py::TestD19sSubCaseIsREFUSEDRatherThanDegenerate`。
  1. **零中心 + `floor=0` 不会产生一个退化的首解**,因为**块根本建不起来**:
     `check_linearity` 的 `_refuse_unusable_noise` 在任何求解之前就以
     「smallest eigenvalue 0」拒绝,并**指名 scale 表达式**、给出出路(加一个 floor)。
  2. **而且这条拒绝与先验中心无关**:`floor=0` 在中心 **2.5**(那里 σ=0.125)时**同样**
     被拒,因为仿射探针的 scale 扫描会经过预测过零的那个点并在那里读协方差。
     所以 `floor=0` 不是「起点不巧」,是**本包的精确路线不接受的模型**。
  3. **本包早就知道**:`test_a_correlated_prediction_dependent_model_finds_the_same_
     fixed_point` 的 docstring 记着同一机理的相关核版本,并写着这种模型
     「classifies to NUTS, and correctly」。
  **因此本行的第 (3) 条按事实改写**:零中心 + `floor=0` 子例**确实**从此是回归
  fixture,但它是**拒绝**的回归 fixture,不是求解的——已按此形态落地(四条断言,
  含两个中心与「同一模型加了 floor 就从零中心正常收敛」的反向)。
  **第 (1)(2) 条改为待 Wave B 实测**:起点该不该动,现在是一个纯粹的**数值连续性**
  问题(`iterations`/`delta`/`converged` 三个可观测量会随起点动),而它**只能**对着
  rheplicant 自己钉住的数字量,那些数字不在本包这一侧。**本批因此没有改起点**,
  也没有把这条推迟藏在某人的记忆里。
  **【延期部分结清,2026-08-28,Wave B 的 `gls` 批:起点不动。】**
  按本行写死的判别条件逐条量过,而**第一次测量一文不值,值得记下来**:把近端
  的种子换成远端的,44 条测试全绿——但两条硬钉的步数**都是地板**
  (`MIN_REWEIGHTS=5`,以及一条显式的 `min_reweights: 8`),**地板看不见种子**。
  把地板降到 1、让 `iterations` 成为真正的收敛步数之后再量:两个种子确实不同
  (σ 在数据上 `[2.940, 3.074, 2.938, 3.230]` 对 σ 在先验均值上 `[3, 3, 3, 3]`,
  相对差 **9.2%**),而 `iterations` **两侧都是 4**;`delta` 从 1.637e-07 变到
  3.092e-07,两个都在舍入量级、都远低于容差,`converged` 不受影响。
  方向另查:对着 float64 跑到不动点的 IRLS,**委托版 9.61e-08、被删的本地循环
  1.055e-07**,委托版更接近;常数 σ 路径逐比特相同。
  **因此不新增裁决项**——这一条从 2026-08-26 拍板、2026-08-27 修正前提、
  2026-08-28 结清数值半边,三次都写在本行里。
  证据链:`2026-08-28-wave-B-gls-opening.md` §5.1。
- **D20 — 掩码的声明面(G1 落地时新增)。** §四 G1 只写了「观测掩码贯通
  exact/precision(inf-σ = 零权)」,没写**谁声明**。两个读法:(a) inf-σ 本身
  即声明——`precision_parts` 见到非有限 scale 就产出掩码协方差;(b) 节点上显式
  声明 `observed_mask`,scale 保持有限,inf-σ 只作为**回报**存在于
  `per_sample_sigma` 这一个接缝。
  **【本次委托下自定,2026-08-27:取 (b),依 2026-08-27 的一次性委托;计划未
  预见此点,按「保守的一侧」规则自选。】** 三条理由,都可核:
  1. **inf-σ 只对记得去看的消费者是掩码。** 对其余消费者它不是「没有信息」而是
     **-inf**:`Normal(mu, inf).log_prob` 处处 -inf,`log_joint` 与 `to_numpyro`
     的整个势能随之消失。rheplicant 自己就不把 inf 放进 numpyro site——
     `numpyro_bridge.to_numpyro_model` 写的是 `where(seen, sigma, 1.0)` 加
     `handlers.mask`,并在 docstring 里逐字说明为什么。所以 inf-σ 是**上游编码**,
     每个消费者在自己的接缝上翻译它;(b) 只是把那句翻译写成一处。
  2. **(a) 要删掉一条已写下论证的守卫的对象。** bayesmith 的 `check_gaussian`
     拒绝非有限 scale,消息里逐字写着「一个 `Precision` 被问的是这个协方差给出
     什么密度,而无限方差的诚实答案是没有」。走 (a) 会把「sigma 表达式溢出了」
     与「这一路被 flag 了」两种需要不同修法的故障合并成一种。
  3. **声明可枚举,inf 不能。** 一个 inf 既可能是 flag 也可能是除零;一个字段
     可以被守卫双向钉住。
  **「inf-σ = 零权」在 (b) 下仍然为真**,只是位置从输入编码移到输出回报:
  `per_sample_sigma` 对被掩样本报 `inf`,于是 `evidence.compress` 的既有掩码
  路径一行未改就对上了,两处掩码归一化器**逐比特相等**(守卫钉住)。
  **两条实测边界写在这里以免被读成遗漏**:(i) 相关(circulant)协方差 + 掩码
  在构造期 `StructureError` 拒绝——compress 已量过近似代价 0.47 nat;
  (ii) `evidence.campaign` 的分期切片没有 `MaskedPrecision` 规则,会以
  「no rule for slicing」明确报错,证据层是 Wave D,补齐归那一波。
  证据链:`2026-08-27-wave-P2-G1.md`。

- **D21 — 诊断的展开点(Wave A 开工时新增)。** `identifiability` 是**非线性模型的
  局部性质**(它自己的 docstring 这么写),所以「在哪一点展开」是判据的一部分。
  两侧默认**不同**:rheplicant 展开在 `space.initial_values()`(每个 `Latent` 声明的
  `init`),bayesmith 展开在 `prior_environment(graph)`(先验中心)。
  `docs/migration/identifiability.md` §5.2 把这条差异记作「the same point, one
  spelling」——**实测为假**。`Latent` 的 `init` 与 `prior` 是**互相独立**的字段,该
  模块自己的例子就是 `Latent("fwhm_deg", init=12.0, prior=dist.Uniform(5.0, 30.0))`。
  实测(`mu = a·exp(b·x)`,同一模型同一先验):展开在 `a=0.0` 得 **nullity 1**
  (`s_min/s_max = 0`),展开在 `a=1.0` 得 **nullity 0**(`2.406e-01`)——**判决翻转**。
  **【本次委托下自定,2026-08-27:门面显式传 `at=space.initial_values()`,不吃
  bayesmith 的默认。】** 取这一侧的三条理由:(1) 铁律 1 要的是**数值一致**,而展开点
  一动数字就动;(2) 它**不需要改 bayesmith**——`at=` 本来就是参数,默认值只是默认值;
  (3) 它可被守卫钉住,而「两个默认碰巧相同」不能——只有 `init != prior 中心` 的
  fixture 才分得开,所以那条 fixture 是这条裁决的守卫。
  **同时更正** `docs/migration/identifiability.md` §5.2 的那句话:它是本程序里
  「一个事实两份拼写、其中一份悄悄过期」的又一例,而这次过期的那份是**契约页**。
  证据链:`2026-08-27-wave-A-identifiability.md`。
- **D22 — 秩测试合成图的三样,以及摘掉 `joint_prior`(Wave A 模块 1 时新增)。**
  `identifiability` 不带数据、噪声、先验;`to_graph` 三样都要。门面因此**合成**
  `observed=zeros`、`noise=HomoscedasticNoise(1.0)`、以及自由 latent 的先验,并
  **摘掉**声明了的 `joint_prior`(否则继承 `to_graph` 对它的拒绝,实测打红
  `test_jeffreys_prior.py` 八条)。
  **【本次委托下自定,2026-08-27:合成三样并摘 `joint_prior`,且把「它们够不到
  答案」作为**被测量的**前提而不是论证。】** 合法性只有一条:秩判决读不到它们
  ——bayesmith 的 `local_block` 的 docstring 逐字写着先验字段「deliberately
  empty」,`dense_operator` 微分的是观测节点的 `loc`。**但 docstring 不是测量**,
  而这是整个委托赖以成立的假设,所以 `TestTheSynthesisedGraph` 把图造三遍(数据、
  σ、先验宽度各变一次)逐字段比较报告,并配一条断言基线非退化的兄弟测试。
  **一条必须重新测量而不能继承的假设**:`prior_sensitivity` **确实读先验**
  (它算的就是先验位移),所以本行对它**无效**,`sensitivity` 那一批要自己量。
  证据链:`2026-08-27-wave-A-identifiability.md` §一。

- **D23 — `prior_sensitivity` 的拒绝判据:雅可比秩 vs 曲率(Wave A 模块 2 新增)。**
  rheplicant 在**观测雅可比的秩**上拒绝;bayesmith 在 **rest 项自身的曲率**上拒绝。
  bayesmith 的判据更对,理由写在它自己的 docstring 里:一个被**下游** latent 的密度
  持住的被选 latent(`child ~ Normal(parent, s)`,child 在选择之外),其
  likelihood-only 众数**完全良定义**,而雅可比秩检验会拒绝这个合法问题。
  差异**双向**:bayesmith 接受一些 rheplicant 拒绝的(上述),也拒绝一些 rheplicant
  接受的(雅可比满秩、曲率条件数超 `1/sqrt(eps)` 天花板)。
  **【登记,未裁决,2026-08-27】** 本批次**既未采纳也未回避**:两侧拒绝的**措辞相同**
  (移植时保留),门面只翻译异常类,64 条测试**无一能分辨**这条差。所以它是一条
  **已知的、当前无守卫的**语义差。要正式采纳曲率判据,先得造出能分辨它的 fixture
  (下游密度持住的被选 latent),那是一次**语义升级**,届时在本行拍板。
  证据链:`2026-08-27-wave-A-sensitivity.md` §四。
  **【本次委托下拍板,2026-08-28:采纳曲率判据,并补上守卫。】** fixture 造出来了,
  而**造出来之后本行的框架就变了**:这不是一次「要不要升级」的选择,因为 Wave A 切换
  `prior_sensitivity` 的那一刻**曲率判据就已经在跑了**。唯一还开着的问题是**它有没有
  守卫**,而答案是没有。所以拍板的内容是「采纳」加「钉住」,不是「改变」。
  **两个方向逐个量过,而它们的**可达性**不同**——这一点原行没有区分,是本次的新事实:
  1. **远端接受、秩判据拒绝**(原行论证所依据的那个方向):在图里是真的
     ——`child ~ Normal(parent, s)`,`child` 在选择之外,实测观测雅可比秩 **0 / 1**
     而 `prior_sensitivity` 交回 shift **−0.1096**。**但本包声明不出来**:`Latent`
     的 prior 是**声明期**建好的 numpyro 分布,参数是具体数组,没有 latent 可指。
     实测 `dist.Normal(<一个 Latent>, 0.5)` **构造得出来**(numpyro 什么都收),
     而 `.sample()` 抛 `TypeError: unsupported operand type(s) for +: 'Latent' and
     ArrayImpl`。**构造成功不是可达性**——探针第一版正是这么读的,那是一条不会失败的
     检查。
  2. **远端拒绝、秩判据接受**:一个**普通的两参数近共线设计**,人人可达,而且**今天
     就在跑**。实测经**本包自己的公开 API**:
     `data = a·g + b·(g + sep·g²)`,noise 0.05,float64——

     | sep | `identifiability` 秩 | `prior_sensitivity` |
     |---|---|---|
     | 1e-1 | **2 / 2** | 接受 |
     | 1e-3 | **2 / 2**(满秩) | **拒绝** |
     | 1e-5 | **2 / 2**(满秩) | **拒绝** |
     | 1e-7 | 1 / 2 | 拒绝(两个判据一致,**不是**判别用例) |

     中间两行就是这条语义差可达的全部:**秩判据会接受,而正在跑的判据拒绝。**
  **守卫**:e-RHINO `tests/inference/test_d23_refusal_criterion.py`(10 例)。它**先
  断言雅可比满秩**再断言拒绝——否则那条拒绝会是两个判据都同意的一次拒绝,这份文件就
  什么也没钉住;并配一条 `sep=1e-1` 接受的基线,和一条「拒绝信息里说了雅可比不是理由」
  (远端在两个判据分歧时报测得的谱,而不是借一个不成立的秩判决)。
  **方向 1 保持登记,解除条件写在这里**:若本包的声明层哪天长出层级先验(一个 latent
  的 prior 由另一个 latent 参数化),那个方向就变得可达,届时它需要自己的守卫。
  **顺带更正一句被切换落在后面的模块 docstring**:`inference/sensitivity.py` 到今天
  为止还写着「a rank-deficient selection is refused, and the rank comes from
  `identifiability`」——那描述的是**被换掉的**判据。同一形状本程序已付过学费
  (D21 的契约页 §5.2、附录 B 过期一整批)。
  证据链:`2026-08-28-d23-two-criteria.md`;探针
  `docs/probes/probe_15_d23_two_criteria.py`。

- **D24 — `JeffreysPrior.information` 的行序:`sorted(over)` 还是 `over`(G13 接线时新增)。**
  rheplicant 按 **`sorted(over)`** 返回行列,并在自己的 docstring 里把这条写成一个
  **wart**:「`over=("b","a")` 与 `over=("a","b")` 返回**同一个**矩阵……一个把第 0 行读成
  『我传的第一个名字』的调用方,在 tour 自己的块上错 **7.4e+1**」。bayesmith **故意没有
  移植这条 wart**——它的 docstring 逐字写着「the graph machinery preserves the caller's
  order, so the wart does not port」。行列式不受影响(对称置换),所以**先验本身两边
  相同**;变的只是**返回矩阵的行序**,而那是一条被测试读到的可观测输出。
  **【本次委托下自定,2026-08-27:门面把远端的输出置换回 `sorted(over)`,登记本行;
  采纳 `over` 序是一次语义升级,另行拍板。】** 取保守侧的三条理由:(1) 铁律 1 说的是
  可观测面保持,而 tour 自己的块 `("fg_log_amp","fg_beta")` 的 sorted 与原序**不同**,
  所以这条差**有 fixture 分得开**——它是能被守卫钉住的一侧;(2) 置换发生在薄包装里,
  不是第二份实现——§〇 第 5 类明写「产品容器字段布局」属保持面;(3) 升级路径是干净的:
  哪天采纳 `over` 序,改的是门面的一行加一批重测的数字,而不是一个算法。
  **落地在 `priors` 那一批**(本行随 G13 接线登记,因为差异是在那里量到的)。
  **【已落地 2026-08-27】** `priors._rows_in_sorted_order`,**按 span 置换而非按名字**
  ——两个标量的块说不出这条区别,一条向量 latent 的测试说得出,而按名字置换的变异
  (P2)恰好只被它一条杀死。证据链:`2026-08-27-wave-A-priors.md` §三、§八。

- **D25 — 一个 float32 会话里的 Jeffreys 先验:新拒绝(G13 接线时新增)。**
  bayesmith 的 `JeffreysPrior.information` 在 0.4.0 里**指名拒绝** ambient float32
  (D9 顺带查出的第二个洞:同一个精确退化块给出 **-27.52** 对 **-338.05**,310 nat,
  且在 NUTS 会取指数的那一项上)。rheplicant 今天**没有**这条守卫,于是一个 float32
  的 NUTS 运行照跑。`priors` 切换后这条拒绝会**新拒今天通过的路径**。
  **范围已实测,不是估计**:把该守卫临时装进 rheplicant 的 `information` 再跑
  `tests/inference` + `tests/config`,红的**恰好三条**,全在
  `tests/config/test_config_exits_npe.py::TestThePriorGate`
  (`test_one_document_is_refused_by_npe_and_run_by_nuts`、
  `test_the_refusal_is_not_the_missing_section_one`、
  `test_the_advice_the_gate_gives_depends_on_the_document`)。
  `tests/inference/test_jeffreys_prior.py` 的 49 条**一条不红**——它有一条 module 级
  autouse 的 x64 fixture,和 S6 那个文件同一形状。
  **【本次委托下自定,2026-08-27:取「拒绝」,并把拒绝提到 `to_numpyro_model` 的
  构造期;依 2026-08-27 的一次性委托。】** 取这一侧的理由,以及为什么另外两条路
  **不存在**:
  1. `jax_enable_x64` 是**追踪期**的全局开关,而 NUTS 在门面返回之后才追踪 `model`。
     所以「门面内部开 x64」这条(identifiability/sensitivity 用的那条)在这里**用不了**
     ——一个 float32 追踪出来的模型里放不进一个 float64 的因子。
  2. 让 bayesmith 长一个 `allow_single_precision=` 的口子,等于重开 D9 并要一次发版,
     且它把「静默产出错答案」重新变成可达的。
  3. 于是只剩「拒绝」。**提到构造期**是 P1 §三 的原则:图缝会抹掉证据的拒绝住在缝前
     ——构造期拒绝能说出「这份文档声明了 joint_prior」,追踪期只能说出一个 dtype。
  **代价与义务**:那三条 config 测试按分诊第二列**改写对适配器**,并且拒绝必须说出
  出路(在 x64 会话里跑该文档),不能只说「被拒绝了」。**落地在 `priors` 那一批。**
  **【已落地 2026-08-27,而代价比预计小】** 拒绝住
  `numpyro_bridge._refuse_a_joint_prior_in_single_precision`(构造期)。那三条测试
  **一条断言未改**——config 层早有 `runtime.jax_enable_x64`(delivery 层那条 float64
  拒绝自己点名的「remedy 2」),所以改写只是给文档补上它本来就需要的一句声明,连那条
  真跑 NUTS 的也照样绿。**声明写在运行它的 helper 里而不是 builder 里**:fixture 普查
  无参驱动每一个 `*_document`,而这一行在构建期就核对进程,写进 builder 会打红两条
  毫不相干的普查测试(实测)。证据链:`2026-08-27-wave-A-priors.md` §四。

- **D26 — `to_numpyro_model` 的站点名(切 `numpyro_bridge` 前新增)。**
  本包的站点是 `"prediction"`(deterministic)与 `obs_name`(默认 `"obs"`);
  `to_graph` 的内部节点是 `__mu__` / `__data__`。直接委托给 `bayesmith.to_numpyro`
  会改名。**代价已实测,而且比预期小得多**:把两者改名后跑 `tests/inference` +
  `tests/config`,**只红一条**(`test_sites_are_named_after_their_latents`)——
  config 层把 `get_samples()` 过滤成 `space.names`,所以那个站点**无论叫什么**都
  到不了 product。
  **顺带查出一句写反了的注释**:`test_config_exits_npe.py` 逐字写着「按名字断言缺席
  才是有分辨力的一半」,实测**相反**——旁边的集合断言才是(它在改名后仍然看得见
  新名字),按名字那一行改名后就死了。四处里有注释的两处已改正,断言一行未动。
  **【本次委托下自定,2026-08-27:取「保持站点名」;`to_graph` 增加可选节点命名
  参数,只由 `to_numpyro_model` 使用。】** 理由:名字出现在 `to_numpyro_model` 的
  docstring 与 `examples/tutorial_nuts.py` 里,用户读 `get_samples()["prediction"]`
  是被文档邀请的用法,铁律 1 要公开名字保持;而保住它便宜(适配器给两个内部节点
  命名),没有理由用一次可见的破坏换一点简洁。
  证据链:`2026-08-27-numpyro-bridge-measurements.md` §一。

- **D27 — 被抽样的 `noise_std`(切 `numpyro_bridge` 前新增)。**
  `to_numpyro_model(noise_std=<numpyro 分布>)` 产出站点 `"noise_std"`,而 `to_graph`
  只收具体的 `NoiseModel`。乍看像一个发布门问题。**实测不是**:
  `bayesmith.observe(name, dist_fn, *parents, ...)` 收多个 parent,所以 scale 可以是
  另一个节点;端到端探针一次跑通(声明 / 求值 / `log_joint` / `to_numpyro` 四步),
  所需能力**已在 0.4.0 里**,铁律 5 满足,工作全在适配器一侧。
  用量:全仓**两个**消费者,均在 `tests/inference/`;**config 层够不到**
  (`config/sections/` 里没有 `Distribution`)。但它是签名的一部分(连同
  `allow_sampled_noise_std`),铁律 1 要签名保持。
  **【本次委托下自定,2026-08-27:取「适配器把它声明成一个名为 `"noise_std"` 的图
  latent」。】** 理由:它本来就是**一个声明**(σ 的先验)而不是数值,换成图的词汇
  不新增语义;另一条路是为这一个分支保留手写的似然,那就把第二份实现留在最容易
  被忘记的分支上;而且它可被守卫钉住——`"noise_std"` 会进 `graph.latents` 而空间
  没有声明它,所以**空间里一个同名 latent 就是一次碰撞**。
  **那次碰撞今天是什么行为尚未量过**,实现批的第一件事是量它并按结果补拒绝。
  证据链:`2026-08-27-numpyro-bridge-measurements.md` §二。

- **D28 — `to_numpyro_model` 下一个被拒的 1-D sigma(切 `numpyro_bridge` 时新增)。**
  `to_graph` 的 `_prevalidate` 调 `check_noise_std_axis`,而 `to_numpyro_model` 从前
  不调。**实测**:在一个方格网格(8×8)上,`noise_std=jnp.linspace(0.4, 0.6, 8)`
  今天**被接受**,并沿最后一个轴广播成 `(8, 8)`——两种读法是两个不同的模型,而
  兼容的那一种正是危险的那一种(`to_graph` 的 docstring 自己写着「过了这一点,
  sigma 已经被广播进一个分布,歧义被静默地、错误地 settle 掉了」)。委托之后这条路
  **新拒**它。
  **【本次委托下自定,2026-08-27:接受这条新拒绝,按铁律 4(iv) 的「接受为修正」;
  并把它提到 `to_numpyro_model` 自己名下。】** 理由:(1) 保守的一侧是**拒绝**——
  被接受的那一侧静默产出另一个模型;(2) 它可被守卫钉住,而「碰巧广播对了」不能;
  (3) 提到本函数名下是 P1 §三 的原则——`to_graph` 的同一条检查会自称 `to_graph`,
  那对一个调用了 `to_numpyro_model` 的人是真话而无用。**代价实测为零**:全套没有
  一条测试、config 也没有一条路径向这个出口传过一个有歧义的 1-D sigma。
  证据链:`2026-08-27-wave-A-numpyro-bridge.md`。

- **D29 — `parameter_covariance` 的条件数天花板(切 `uncertainty` 时新增;D9 的第二项
  功课在此结清测量)。** 本包的 `parameter_covariance` **不对条件数设门**,而
  `F = J^T N^-1 J` **平方**了设计矩阵的条件数,所以一个普通模型就能触到算术的极限
  (本函数自己的 docstring:κ(J)=1e3 时 float32 的协方差错 2.4%,float64 错 1.08e-12,
  **两者都不说话**)。远端 `bayesmith.exact.fisher.parameter_covariance` 有
  `max_condition="auto"`,从 dtype 推出 `1/sqrt(eps)`(float32 **2.90e3**、
  float64 **6.71e7**)。
  **范围已实测,不是估计**:把该天花板临时装进本包的 `parameter_covariance`,跑
  `tests/inference` + `tests/config` + `tests/evidence`,**65 次求逆里 3 次会被拒**,
  全是 float32,κ 分别为 **1.0e4 / 6.5e5 / 3.2e6**,逐条点名:
  `tests/config/test_config_exits_conjugate.py::TestWidth::test_width_fisher_over_a_whole_multi_latent_space_is_allowed`、
  `tests/inference/test_noise_std_axis.py::TestFisherInformation::test_the_two_explicit_readings_give_visibly_different_answers`、
  `tests/inference/test_fisher_prior.py::TestPriorEntersTheMatrix::test_tightening_the_prior_tightens_the_error_bar`。
  **【本次委托下自定,2026-08-27:接受天花板,按铁律 4(iv) 的「接受为修正」;
  三条按分诊第二列改写。】** 理由:那三条今天拿到的是**静默错误的**数字——一个
  Cramér–Rao 界错了却不说,比不给更糟,而这正是 D9 原文把它列为功课的原因。
  **落地在 `uncertainty` 的下一步**(本步只做 `fisher_information`)。
  证据链:`2026-08-27-wave-A-uncertainty-fisher.md`。
  **【已落地 2026-08-27】** 范围**重量了一遍**而不是继承:65 次求逆、3 次被拒、
  κ 逐个相同,切换后实跑恰好红那三条,无附带损伤。三条按分诊第二列改写为 float64,
  各带一条 `dtype == float64` 的兄弟断言——**没有它,加宽哪天悄悄失效,三条会以完全
  相同的方式通过**。证据链:`2026-08-27-wave-A-uncertainty-covariance.md` §一、§二。

- **D30 — 远端 `parameter_covariance` 的拒绝穿什么异常类(切 `uncertainty` 后半时新增)。**
  远端对**两件事**抛**裸 `ValueError`**:条件数超天花板,以及「你给的已经是一个协方差」。
  本模块承诺的是 `StateValidationError`(它自己的 8 条既有拒绝、`FlatMatrix.sigma`、
  `propagate_covariance` 的两条来源守卫都是)。一个穿着远端词汇的 `ValueError` 到达
  rheplicant 的出口,正是 D27 对那个裸 `AssertionError` 说过的形状。
  **【本次委托下自定,2026-08-27:门面按类捕获并以本包的类重抛,`from` 原异常;
  计划未预见此点,按「保守的一侧」规则自选。】** 三条理由:
  1. **窄而不宽。** `parameter_covariance` 是本模块**唯一到不了图**的委托——它只调
     `jnp.linalg.cond`/`inv` 与 `condition_ceiling`,所以 `translate` 认识的三族
     `BayesmithError` **在这条路上一条都不可能出现**,`except ValueError` 因此不会
     吞掉一个结构性拒绝。**不**扩宽 `translate` 去认 `ValueError`:那会把远端内部
     真正的 bug 变成一条拒绝。
  2. **信息不丢。** `from refusal` 保留原异常,守卫**两半都钉**——只钉类的守卫会被
     一个丢掉测量的实现满足,只钉消息的会被一个穿错类的实现满足。
  3. **另外两条路不存在。** 在门面里自己算一遍条件数就是第二份实现;改远端抛
     `BayesmithError` 是一次破坏性的已发布表面变更,要一次发版(铁律 5)。
  **同批收下的一条新拒绝**(铁律 4(iv)「接受为修正」):**再求一次逆**今天回来贴着
  `kind='covariance'` 而它已经是一个精度;**代价实测为零**(65 次调用里 0 次)。
  证据链:`2026-08-27-wave-A-uncertainty-covariance.md` §五。

- **D31 — `propagate_covariance` 切还是留,以及它合成的两样(同上批次新增)。**
  上一批写的是「先量再决定,大概率与 `predict_from_samples` 同一结论(留守)」。
  **量完之后结论相反**,按「委托不是空白支票」处置并在此点名。
  **【本次委托下自定,2026-08-27:`propagate_covariance` 切,`push_forward` 留守。】**
  判据不是「有没有 jax 调用」,是**里面有没有一个有名字的统计方法**:
  `push_forward` 是 `jax.vmap(forward)`,一个 map,零贝叶斯数值;
  `propagate_covariance` 是 **delta 方法**——一次雅可比、一个二次型、一个线性化近似,
  而远端实现的是同一个,留着它就是把一个有名字的算法留下第二份实现。
  **数值连续性**:旧公式与委托后逐比特相同,float32/float64 各两条路线,`rel diff = 0`。
  **合成的两样(数据 + 噪声)**:图要它们,delta 方法两样都读不到。合法性照 D22 的规矩
  **量而不是论证**——造三遍(σ=1 / σ=1e4 / 数据 +1e3)逐比特比较,外加一条基线非退化的
  兄弟断言;变异 U7(让远端真的按精度加权)把这两条打红。
  **同批收下的一条新拒绝**:交给它一个**精度**今天返回一个有限、形状正确、**错了整整
  一个平方**的误差棒;**代价实测为零**(23 次调用里 0 次)。该拒绝**复用
  `FlatMatrix.sigma` 已在用的 `_PRECISION_KINDS`**——一张表两个调用方,不是第二份
  实现——并住在**缝前**,因为图缝之后它会穿上 `ParameterSpaceError` 到达。
  证据链:`2026-08-27-wave-A-uncertainty-covariance.md` §三、§四、§五。

- **D32 — `bayesmith.fit` 是一个入口还是两个(做 G2 时新增)。**
  两个消费者打分的**不是同一样东西**:Wave B 的梯度块降的是一个**条件后验**
  (似然 + 先验),Wave C 的 calibrator 降的是 `loss_fn(forward(params), observed)`
  ——一个只看得见预测与数据的目标。rheplicant 的 `engines.py` 自己的 docstring 逐字
  记着这条区别,并说明为什么 `AdamCalibrator` 不能用作梯度块:「这个接口没有办法
  传先验」。
  **【本次委托下自定,2026-08-27:两个入口(`fit` 图侧 / `minimize` 任意目标),
  共用一个优化器;计划未预见此点,按「保守的一侧」规则自选。】** 另外两条路各自
  破一条法则:只给 `fit` 会逼 Wave C 为一次最小二乘**发明一个模型**;只给
  `minimize` 会让 Wave B 的**每个**梯度块自己拼 `-log_joint`,把一句声明抄进每个
  调用点(而「一份实现」的对偶正是「一句声明一处」)。两个入口共用一个优化器,
  所以不是两份实现——`fit` 是 `minimize` 的一个调用方,九行。
  证据链:`2026-08-27-p2-g2-fit.md` §五。

- **D33 — 一次发散的下降:返回还是拒绝(同上批次新增)。**
  朴素梯度下降在步长超过 `2/L` 时发散,Adam 不会——**实测**:单 latent 高斯
  `L = 231.1`,界 `0.00865`,`"gradient"` 在 0.006 收敛、0.02 给 **NaN**,
  `"adam"` 两个都收敛。所以一个只改 `method=` 不改 `learning_rate=` 的调用方正好
  落在这里,而 rheplicant 的两个 calibrator 今天会**把 NaN 交回去**。
  **【本次委托下自定,2026-08-27:拒绝,用 `eqx.error_if`。】** 取拒绝一侧的理由:
  一个 NaN 的「拟合」是本程序反复拒绝的那种**静默错答案**,而它有一条现成的、
  jit 下也能用的机制(`wiener_solve` 的同一条)。挂在**点**上而不是 objective 上,
  因为调用方先读的是点,而一个没人用的检查可能被优化掉。
  **这条改变了 Wave C 的分诊**:`calibrate` 的测试里若有依赖「NaN 也照样返回」的,
  接线那一批要按分诊第二列处置——**今天没有量过**,量它是那一批的第一件事。
  证据链:`2026-08-27-p2-g2-fit.md` §六.2。

- **D34 — `declared_partition` 遇到不完整覆盖:补一个 NUTS 块还是拒绝(做 G10 时新增)。**
  `factor_partition` 把解不了的 latent 扫进一个 NUTS 块;声明入口照做会**替调用方
  作一个他没作的决定**,而那正好是「你声明你负责」的反面——一个漏掉的 latent 与一个
  被判给 NUTS 的 latent 在返回的 plan 里长得一模一样。
  **【本次委托下自定,2026-08-27:拒绝,并点名未覆盖的 latent;计划未预见此点,
  按「保守的一侧」规则自选。】** 同一条规则决定了这个入口**还**拒绝什么:名字不是
  latent、一个 latent 在两个块里、空块、未知方法、第二个 `nuts` 块——全部是**簿记**
  错误而不是建模主张。建模主张(这个块真的仿射吗)恰恰是这个入口跳过的东西,拒绝
  它们会让入口失去意义;簿记错误放过去则会让它变成一个陷阱。
  证据链:`2026-08-27-p2-g10-g12.md` §一。

- **D35 — 每 sweep 钩子在 NUTS 余项存在时:调用还是拒绝(同上批次新增)。**
  **实测而非推测**:带 NUTS 余项时 sweep 变成 `HMCGibbs` 的 `gibbs_fn`,而 numpyro
  **追踪**它——两块 plan 跑五个 sweep,Python 层只进入 **2 次**。一个 Python 回调
  因此只在**追踪期**触发一次,交回一个从未发生过的 sweep,而里面的值形状与 dtype
  全对。
  **【本次委托下自定,2026-08-27:拒绝,并在消息里写出那次计数。】** 另一条路
  (`jax.debug.callback`)能在 jit 下触发,但顺序不定且无法交回 Python 对象,
  于是「诊断」会变成一串不知道属于哪个 sweep 的打印。**拒绝的对象是钩子,不是混合
  plan**——后者一直能跑,兄弟测试钉住这一条。
  证据链:`2026-08-27-p2-g10-g12.md` §二。

- **D36 — sweep 形 estimate:解还是抽(同上批次新增)。**
  D14 (ii) 写的是「块坐标下降」,没写精确块用 `wiener_solve`(条件**均值**)还是
  `gcr_sample`(条件**抽样**)。
  **【本次委托下自定,2026-08-27:解,因此不收 key,因此确定性。】** 三条理由:
  (1) 一个 estimate 的名字承诺的是点而不是样本;(2) 抽样版本也会落在众数附近,
  **在 docstring 里两者一模一样**而每次运行的数字不同——那是「静默的错答案」的形状;
  (3) 高斯条件的**均值即众数**,所以解出来的 sweep 是对联合密度的**精确坐标上升**,
  于是 `history` 单调不降**按构造成立**,而这正是一条「用了过期环境」的实现会红的
  断言。余项块交给 **`fit`**(G2),目标同一个联合——所以 `history` 是一条轨迹而不是
  两条拼起来的。
  证据链:`2026-08-27-p2-g10-g12.md` §三。

- **D37 — 三条站在「不存在」上的守卫,在 D15(a) 落地时怎么办(做 G14 时新增)。**
  bayesmith 有三处白纸黑字断言它**故意没有**移植 `extreme_eigenvalues` /
  `condition_estimate`(模块 docstring、两条 cross-check、契约页 §5 整节),其中一条
  的 docstring 逐字写着「这条会在有人最终移植它时变红——那时他应该去读那份拒绝了它的
  模块 docstring,而不是把一个绿套件当成同意」。
  **【本次委托下自定,2026-08-27:论证保留、判据搬家;计划未预见此点。】**
  读过那份论证:**一个字没错,一个字也没有撤回**——它说的是**守卫**,而 D15(a) 要的是
  **诊断**。所以:(1) 两条 cross-check 的「不存在」断言改成**一致性**断言(两个包
  逐位相等,外加「两者在梯度谱上都错在同一方向」);(2)「不存在」真正代表的那条规则
  **直接钉住**——**AST 双向扫描**:`extreme_eigenvalues` 只许被 `condition_estimate`
  调用,`condition_estimate` 在包内一处也不许被调用,外加一条自检;(3) 契约页 §5
  重写为「差异是一条**规则**而不是一处**缺席**」。
  **为什么这是一条裁决**:铁律说「改判据 = 新裁决项」,而三条守卫的判据从「缺席」
  变成了「无守卫读它」——即使这是在执行 D15(a)。
  证据链:`2026-08-27-p2-g14-condition-estimate.md` §〇、§四。

- **D38 — G9 全量要不要解除 `diagnose` 对复 latent 的拒绝(做 G9 全量时新增)。**
  计划把「`diagnose` 仍拒绝复 latent」列为 G9 全量下**另登记在案**的一项,但没写
  该拒绝是留是走。
  **【本次委托下自定,2026-08-27:保留拒绝;计划未预见此点,按「保守的一侧」规则自选。】**
  理由:一个秩判决在 C 上**既不是 n 也不是 2n**,要让它有意义得先回答一个**语义**
  问题——一个 R^2n 里的零方向,对一个在 C 上声明的 latent 意味着什么、按什么名字报
  出来?D18 只裁了「复数是 bayesmith 的原生能力」,没有裁这个,而**本程序没有人
  回答过它**。保留拒绝是保守的一侧:它指名、给理由、且不阻塞任何别的东西——同一张图
  照样解、照样抽、现在还能出 Fisher。守卫钉**两个方向**(拒绝在;求解不受影响),
  因为没有第二条,一个把整张图都拒掉的实现同样会绿。
  **解除条件**:哪天有人要在 C 上报秩,先定义 R^2n 的零方向怎么报,那是一次语义
  升级,届时在本行拍板。
  证据链:`2026-08-27-p2-g9-full.md` §三。

- **D39 — 归档的 manifest 与它的二进制**没有绑定**(做 D12 前置时发现)。**
  `archive.py` 的第一句是「把 memory 写到盘上,使得读回来时它不能对它撒谎」,而
  manifest 是这个格式的**重建规格**。但**没有任何东西把一份 manifest 绑到它对应的
  那个二进制上**:把 `x.rhep` 配上另一份归档的 `x.json`,两个方向行为不同,而且
  **只有一个方向会抱怨**。
  **实测**(用本次提交的两份 fixture):

  | 配对 | 结果 |
  |---|---|
  | 带模板的二进制 + 不带模板的规格 | **读得进去**,`template_projections` 回来是 `None`——文件里那两个值被**静默丢弃**,其余字段全对 |
  | 不带模板的二进制 + 带模板的规格 | `TreePathError`,指名 `template_projections` |

  不对称有成因:`template_projections` 是**最后一个**动态叶子,所以少期待一个叶子的
  规格只是**提前停下**,多期待一个的才会读过头。模块 docstring 里那句「后面每一个
  叶子都会从错误的偏移读」说的是**后面还有东西**的字段;这里后面没有东西,于是后果
  从「响亮的错乱」变成「安静的丢失」。
  **【登记,未裁决,2026-08-27】** **本批不修**,理由是范围而不是难度:把两者绑起来
  (manifest 里放二进制的摘要或长度)是一次**在盘格式**的改动,而证据层是 **Wave D**
  的;在一条**前置**的名义下改掉那条前置要保护的东西,是自相矛盾的。
  **改为把当前行为两个方向都钉住**(`tests/evidence/test_d12_read_back.py::
  TestTheManifestIsNotBOUNDToItsBinary`),所以切换不能在不出声的情况下改变它。
  **解除条件**:Wave D 决定是否绑定;若绑定,那是一次 `_FORMAT_VERSION` 提升,
  本次提交的两份 fixture 要按新版本重写并**保留旧版本的那两份**,因为「旧写新读」
  正是这条 fixture 存在的理由。

- **D40 — D17 余下两例的分歧**不是探针的分歧**,而是一条假的 `linear=True` 声明
  该被怎么处置(D16 落地后重跑协议时新增)。**
  D16 的四条轴落地后重跑 `probe_11_d17_dual_run.py`:仍 6/8,仍是 `boundary_affine`
  与 `bright_and_faint`。**但两侧现在都判定那条 `linear=True` 是假的**;不同的是
  接下来做什么:
  - **rheplicant 抛** `LinearityRefused`(`auto_blocks`),而它自己的 docstring 写着
    理由:这条检查跑在任何一对之前,「so that 'these two may not share a block'
    always means a coupling between two sound declarations and never one broken
    declaration poisoning every pair it appears in」。
  - **bayesmith 把它归进 NUTS 块**并写上 `reason`,因为 `factor_partition` 是**推导**
    一个分区,而一条被证伪的声明是那次推导的**输入**,不是它的故障。
  **【本次委托下自定,2026-08-27:登记为有意的差异,两侧都不改;计划未预见此点。】**
  两种处置对各自的包都是对的,而且**都不是探针**——把 `auto_blocks` 换成图侧探针
  (D17 问的那件事)不会改变这两例中的任何一例。**这正是结清 D17 的那句话。**
  **它现在有守卫了**,而之前没有:协议本身**故意**不论一致与否都退出 0(以免下一位
  被诱去把它弄绿),所以这条差异一直是 D23 那种「已登记、无守卫」的形状。守卫是
  `tests/crosscheck/test_dispatch.py::TestAFalseLinearDeclarationIsDisposedOfDIFFERENTLY`,
  两个方向各一条,外加一条「两侧对**模型**的判断是一致的」。
  **守卫刻意用一条毫不含糊为假的声明**(`0.1 s²`)而不是协议那条边界值(`1e-7`):
  被钉的是**处置**,而用边界 fixture 会让这条守卫对一个它不该管的阈值敏感——
  实测,`1e-7` 配先验宽 10 时 rheplicant **不拒绝**,所以那样写会把阈值钉成一次意外。
  证据链:`2026-08-27-d17-protocol.md` §四点五 + 本次重跑。

- **D41 — `exact.reduced_basis` 对 ambient float32 的态度(做 G4 时新增)。**
  上游 `rheplicant.inference.reduced_basis` 不拒绝——它住在一个 x64 会话里。
  本包的规矩是「`src/` 从不自己开 x64,它拒绝并说出出路」,所以这里要一个决定。
  **【本次委托下自定,2026-08-27:每个入口拒绝;计划未预见此点,按「保守的一侧」
  规则自选。】** 理由是一个**数**:本模块产出的是调用方要拿去建投影算子的行,
  `c^T G c` 平方条件数,保留切点是**手上算术的** `sqrt(eps)`——float32 **3.4e-04**
  对 float64 **1.5e-08**。前景主导的 bank 上,比最大方向小一万倍以下的方向会被
  **静默丢掉**,而那正是这个基存在的理由(RHINO 前景 ~200 K 对 21 cm ~0.2 K)。
  **它与 `diagnose/local.refuse_ambient_float32` 是两条守卫而不是一条抄两遍**:
  那条的论证是秩判决住在舍入之下,这条的是 Gram 矩阵。两个不同理由不是一个事实的
  两份拼写。消息里给出两个数并写明**在哪里**加宽(建 bank 的那一层,不是这次调用)。
  **顺带把一条 R2 清单项答了一半**:`reduced_basis` 测试族的会话归属**不再是选择**
  ——它必须在 x64 里。
  证据链:`2026-08-27-g4-reduced-basis.md` §三。

- **D42 — G5 的**范围**:`simulate_pairs` 迁不迁(做 G5 时新增)。**
  D10 的子裁决 (2) 写的是「simulate_pairs **迁移后**续用 rheplicant 的
  `NoiseModel.realise`」,而这句话有两个读法,且**字面那个读法做不到**:
  bayesmith 里没有 `NoiseModel`,也不能有(§〇「零射电天文」,且两仓的依赖边是
  单向的)。所以「迁移」只能是「bayesmith 收一个可调用对象,rheplicant 传一个
  闭包」。
  **【本次委托下自定,2026-08-28:`simulate_pairs` **留守**;`bayesmith.amortize`
  是**数组级**的,只收 `(thetas, data)`,不收 `Graph`、不收模拟器;计划未预见此点,
  按「保守的一侧」规则自选。】** 三条理由,前两条是**读签名读出来的**而不是判断:
  1. **缝在哪里是可以读出来的,不是选出来的。** `npe.py` 的四个公开件里,
     `NeuralPosterior`、`train_posterior`、`TrainingHistory` 的签名**只收数组**——
     它们从不碰 `ParameterSpace`、pipeline 或 `NoiseModel`。只有 `simulate_pairs`
     碰,而且三样全碰。
  2. **D10(2) 已经裁掉了另一条路。** 一个 bayesmith 一侧的模拟器只能从图的
     `dist_fn` 生成,而那正是 D10(2) 说不承担的义务——`RadiometerNoise` 的生成侧
     (`d(1+fw)`,符号跟着预测走)与密度侧(`|d|f`,取绝对值并施 floor)是**两条
     不同的律**,该模块自己的 docstring 逐字写着这一点。
  3. **「收一个闭包」不是一份实现,是一层间接。** 真要迁,能迁的只有拆 key、
     `vmap`、和把各 latent 拉平拼接这三件簿记;两个闭包(先验抽样、模拟一次)仍
     由 rheplicant 写,而那里面才是全部内容。按 **D31** 立下的判据——「里面有没有
     一个**有名字的统计方法**」——`simulate_pairs` 的内容是噪声物理加绑定,归 §〇
     第 2 类留守。
  **后果**:Wave C 的 `npe` 接线是**三名委托、一名留守**;`simulate_pairs` 不进
  分诊表的第三列(退役),它连切都不切。
  证据链:`2026-08-28-g5-amortize.md` §一。

- **D43 — 一次发散的训练:两个结果,只拒绝其中一个(同上批次新增)。**
  D33 已为 `minimize` 立下规矩:一次发散的下降**拒绝**而不是把 NaN 交回去。NPE
  的训练是同一个 Adam,但它多了一个 D33 没有的东西——**留出集选择**,而它把同一次
  发散分成了两个结果。**实测**(工作学习率的一千倍,`lr=1e3`,60 步):

  | `validation_fraction` | 返回的估计器 | 历史 |
  |---|---|---|
  | `0.1` | **有限、可用**(`log_prob` = -45.5) | `best_step` 塌到 **1**,`validation` 带 NaN |
  | `0.0` | **参数就是 NaN**,此后每一次 `log_prob`/`sample` 都是 NaN,形状与 dtype 全对 | `train` 带 NaN |

  **【本次委托下自定,2026-08-28:只拒绝第二种,用 `eqx.error_if`,挂在**参数**上;
  第一种**报告**而不拒绝;计划未预见此点,按「保守的一侧」规则自选。】**
  1. **拒绝第二种**,理由同 D33 一字不差:一个穿着答案外衣到达的 NaN,是本程序花
     力气最多要消灭的形状。挂在**参数**上而不是历史上,因为参数是返回物、是后面
     每一次查询读的东西;用 `eqx.error_if` 而不是 Python `if`,因为守卫要在 jit 下
     也能响(和 `wiener_solve`、`minimize` 同一条机制)。
  2. **不拒绝第一种**,因为那里**没有错答案**:留出集选择留下的是最后一批有限参数,
     它是一个真的估计器。发散这件事的唯一记录在 `history` 里,所以历史里的非有限项
     **不清洗**——清洗掉它,这次运行就再没有任何东西说得出发生过什么。
  3. **发散不总是走到 NaN**:再高一千倍(`lr=1e6`)两条路都回来有限,损失 88.8。
     所以本模块**不拒绝一个差的拟合,只拒绝一个不存在的**。
  **同批**「量了之后**没有**新增」的一条拒绝,写在这里因为它比新增的那条更容易被
  下一位重新发明:`min_scale = 0`。上游 config 层有一条针对它的语法拒绝,读起来像
  是数值层也该有一条。**实测没有**:scale 是 `softplus(raw) + min_scale`,而
  `softplus` **严格为正**,所以零地板碰不到零;一个刻意可塌缩的 bank(八个不同
  theta、八个分量、1500 步)在 `min_scale=0` 与 `1e-3` 下都训练到有限密度
  (max log-density 3.71 对 3.63)。**为一个造不出来的故障加一条拒绝,就是造一条
  永远不会失败的守卫**——本程序反复付学费的那条。
  **但 `min_scale < 0` 是另一回事,而且它新增了**:负地板从一个严格为正的 softplus
  里减出来,足够负的分量 scale 就是**负数**,密度里的 `log(scale)` 于是对每一次
  查询都是 NaN,而网络参数**完全有限**——所以上面那条 `eqx.error_if` 也够不到它。
  实测:`min_scale=-1.0` 在一个**未训练**的估计器上就返回 NaN。构造期拒绝,
  代价实测为零。
  证据链:`2026-08-28-g5-amortize.md` §三、§四。

- **D44 — `compress_reduced_basis` 的归属波次(做 G6 时新增)。**
  G6 的登记页把它列进七个确定缺口,并注「依赖 G4;**排期上跟 G4 走**,登记在 G6」。
  但 **G4 的「跟着走」在那一页上没有终点**:G4 的另外三个名字(`score_directions`、
  `build_reduced_basis`、`basis_fidelity`)归 Wave C,而本行原来没说这一个也归。
  一条只写着「跟另一件走」的推迟,是本程序 §八「限制需要解除条件」那条已经付过四次
  学费的形状。
  **【本次委托下自定,2026-08-28:归 Wave C,与 G4 余下三个名字同批;计划未预见
  此点,按「保守的一侧」规则自选。】** 理由是量出来的,不是排期偏好:
  1. **它的数值核心本包**已经有了**。该函数自己的 docstring 逐字写着它是
     「`compress_linear`'s QR with `S_e^T` in place of the design matrix ... the
     same three lines rather than a second implementation of them」——把基的行当作
     design 传给 `bayesmith.evidence.compress` 就是它,**本批实跑核实**:40 个样本、
     5 维基,`compress({"c": rows.T}, ...)` 的 `log_prob` 对着 numpy 写的
     `d ~ N(S^T c, s^2 I)` 精确似然,六个探点最坏相对误差 **1.76e-16**。
  2. **它剩下的每一件都读容器。** `basis.rows`、`basis.c_ref`、
     `basis.coefficients(values)`、`basis.reference_values`、`basis.support` 五个字段,
     而 `ReducedBasis` 是 §2.2 的留守容器(D12)。冻结噪声代价、support 扫描、
     偏差梯度三件都挂在这五个字段上。
  3. **它要的 `c(theta)` 由 `build_reduced_basis` 造**,而那是 G4 归 Wave C 的三个
     名字之一。**在造它的东西存在之前先写吃它的东西**,是对着一个没人定过的接口写实现。
  **所以 G6 本批交付 7 个里的 6 个**,第 7 个有了确定的波次而不是一句「跟着走」。
  证据链:`2026-08-28-g6-consumption.md` §二。

- **D45 — `template_modes` 与既有 `coherent_mode`:两条统计量,还是一份实现的两次
  拼写(同上批次新增)。**
  G6 登记页 §2.1 把 `coherent_mode` 列进「已有对应物」,并**同时警告**「同名不等于
  同题……哪些需要 cross-check 由那一批按铁律 2 定」。本批读了两边,**它们不是同一个
  统计量**,四处不同,逐条实测:

  | | bayesmith 的 `coherent_mode` | rheplicant 的 |
  |---|---|---|
  | 输入 | 存下的项 **+ 一个点** | 压缩时存下的**逐 epoch 摘要** |
  | 残差 | 该点上的 `‖Rx − z‖²` | **出张成**的 `‖(I−P)z‖²`,过了本 epoch 自己的最佳拟合 |
  | 自由度 | **第一个项的行数** | **逐 epoch 各自的,求和** |
  | 命名模板 | **没有** | `{name: {mean, scatter, z}}` |

  两者只在**逐 epoch 的极小点**上重合,而那不是一个 campaign 会去取值的点。
  **【本次委托下自定,2026-08-28:新增 `template_modes`,**不改** `coherent_mode`;
  计划未预见此点,按「保守的一侧」规则自选。】** 三条理由:
  1. **它们回答的问题不同。** 一个可以在任意 theta 上重问,另一个**在原始数据还在的
     那一刻**才算得出来,此后任何调用都重算不了。把后者写成前者的一个关键字,等于让
     一个函数按参数变成两条统计量。
  2. **`coherent_mode` 是 0.4.0 已发布的表面**,改签名要一次破坏性发版(铁律 5),
     而这里没有任何东西需要那个代价。
  3. **散兵自由度那条差异是实测的**,不是风格:`Var(Σχ²_k) = 2Σk`,所以求和才是
     χ² 分布的那个量。读第一个项的行数——一个基于项的统计量只能这么做——在各 epoch
     旗标不同的那一刻就把零假设算错了。守卫是一条**故意散兵**的 campaign
     (`test_the_chi_square_z_uses_each_epochs_own_degrees_of_freedom`),它报
     `chi2_dof is None` 而不是一个数。
  **两处 docstring 互指**,写明各自读的是什么,因为「同名不同题」的代价正是没人并排
  看过它们。
  证据链:`2026-08-28-g6-consumption.md` §三。

- **D46 — `evidence/` 对复 latent 既不处理也不拒绝(推多数据集联合后验时发现)。**
  本包的目标域是 visibility,而 visibility 是复的。`exact/` 与 `diagnose/` 早就
  处理或拒绝复数——`exact/block.py::real_parts` 存在的理由逐字是「the map from
  complex coefficients to data is R-linear but not C-linear」,`optimize.py` 有
  `_refuse_complex`,`exact/fisher.py` 三处分支于 `is_complex`,`diagnose` 按 **D38**
  保留拒绝。**`evidence/` 是唯一两样都不做的 kernel family**:实测
  `grep -c complex src/bayesmith/evidence/*.py` 每个文件都是 **0**。
  **后果是静默的错答案,不是异常。** `log_prob` 算 `-0.5*sum(residual**2)`——双线性型
  `r^T r`;而复 QR 的 `Q` 是**酉**的,保 `r^H r` **不保** `r^T r`。`information()` 是
  `factor.T @ factor`,**没有共轭**。最小复现(本次独立实跑核实,x64):
  一个共享复标量,`R_1=[[1j]]`、`R_2=[[1]]`、`z=0`——

  | 量 | 手算真值 | 实际 |
  |---|---|---|
  | `sum_i R_i^T R_i` | **0**(`(1j)^2 + 1^2`,无任何舍入) | `combine(...).information()` = **2.0** |
  | `t1.log_prob + t2.log_prob` | `0j` | `combine(...).log_prob` = **-1.0** |

  **绝对误差等于真值的全部**,没有 `StructureError`,没有警告。
  **而且不是构造器滥用**:`compress` 本身接受复设计块,返回复 `factor` 配 float
  `offset`(「常数都对」恰是读起来正常的那一半);`compress_epoch` → `marginalise`
  全程也不拒绝,一个圆对称复高斯返回 `-8.33 + 3.17j` 对真值 `-26.18`——**实部错
  17.85 nat,外加一个对数概率里的虚部**。realify 路线给 `-26.18`,吻合 `1e-12`。
  **【登记,未裁决,2026-08-28】** 修法有两半,而**选哪一半是一次语义裁决,推导里
  没有人替它做**:(a) 在 `SqrtInfo.__check_init__` **拒绝**复 `factor`/`target`,
  另配一个 realify 层;(b) 在 `log_prob` 与 `information` 里加共轭
  (`jnp.abs(residual)**2` 与 `factor.conj().T @ factor`)。
  推导页倾向 **(a) + realify**,理由是 realify 的**列序本身是设计决策**,藏进
  `log_prob` 会让它不可声明——而 `ComplexNormal`(0.3.0)已经为复 latent 规定过
  一次列约定,两处约定必须是同一句话。**取 (b) 则要回答**:`information()` 返回的
  Hermitian 矩阵与 `exact/fisher.py` 按实自由度铺开的 `2n x 2n` 布局(**G9 全量**)
  是不是同一个对象。
  **这条与 0.5.0 的关系要 owner 定**:它是一个**已存在**的缺陷(不是本轮引入),
  加拒绝会**新拒今天被接受的输入**,按铁律 4(iv) 属「接受为修正」;但发版内容是
  owner 的决定。
  证据链:`2026-08-28-multi-dataset-joint-posterior.md` §7.1。
  **【owner 拍板并已落地 2026-08-28:取 (a),在 `SqrtInfo.__check_init__` 拒绝。】**
  拒绝住**构造期的唯一咽喉**——每一个项都经过它,包括 `compress` 产出的那些,所以
  一条检查覆盖全部到达路径。消息点名出路(按实自由度携带,见 `exact/block.py::
  real_parts` 与 `ComplexNormal` 的列约定),而不只是说「被拒绝了」。
  **代价实测为零**:`tests/marginal` + `tests/crosscheck` **289 条**一条未改。
  **realify 层不在本批**——那是让复数**能用**的新表面,归需要它的那一波;本批做的
  是把**静默的错答案**变成一次拒绝。守卫含一条**兄弟断言**,把拒绝所保护的那条算术
  单独钉住(双线性和恰为 0,酉形式为 2),因为拒绝落地之后那个行为就不可达了。
  变异 S1 击杀 5 条。

- **D47 — `evidence/` 改名为 `marginal/`(架构叙事审计后新增)。**
  **【owner 拍板并已落地 2026-08-28:改名,旧路径保留为弃用垫片,1.0 退役。】**
  两个 workflow 在这一点上**结论相反**,而拍板取了审计那一侧:数学推导页 §0.3 建议
  **不改**(成本高、收益小),架构审计 §6.6 倾向**改**(信心 0.6)。owner 裁决改。
  **理由的要害不在子包内部,而在外部**:`evidence` 在本包另有 **15 处**普通英文
  「依据」义且用得对——包括 `errors.py`,**唯一 eager import 的模块**,用户读到的第一段
  bayesmith 散文。一个占着这个词的子包让那 15 处全部瞬间歧义,而**任何写在子包里面
  的免责声明都够不到子包外面的散文**。子包**内部**该词有四义,最刺眼的是
  `chain.py` 的「add one epoch's **evidence**」——单个似然因子恰恰是这个词**不**指的东西。
  **`marginal` 是唯一零冲突的候选**:`factors/`、`streaming/`、`information/` 分别被
  `dispatch/factor.py`、`dispatch/streaming.py`、`fisher_information` 封死;
  `compress/`、`campaign/`、`sqrtinfo/` 与子包内模块同名,而**这个遮蔽陷阱是活的**
  ——`compress` 被重导出之后 `type(bayesmith.marginal.compress).__name__` 实测就是
  `'function'`,同名模块**永久被遮蔽**。
  **垫片的形状是承重的,不是实现细节**:模块级 `__getattr__` **不支持深路径**
  (`from pkg.old.kernel import helper` 抛 `ModuleNotFoundError`),而 0.4.0 从这个
  子包发布的 **17 个名字与顶层 `__all__` 的交集是空集**——**每一个都只能走深路径**。
  所以垫片走 `sys.modules` 别名,别名清单**由新包派生**而非手抄。四种 import 形式
  逐一钉住,外加「是同一个对象而不是副本」「旧路径警告、新路径不警告」两条。
  **成本实测**:仓库内 106 处 / 29 文件,子包外的运行时 import **只有 2 行**,
  下游 e-RHINO **0 次 import**。变异 S2/S6 击杀。
  证据链:`2026-08-28-architecture-narrative.md` §6。
- **D48 — G15 的解除是几行?以及先验准入归谁(做 G15 的 rheplicant 一半时新增)。**
  **【自裁于委托之下,2026-08-28:解除是四处,准入留守 rheplicant。】**
  **本条是「委托不是空白支票」的第四例**,而且冲突的是本页自己:G15 与
  `uncertainty._prior_precision` 的 docstring **都**写着解除只改一行
  (`include_prior=space is not None`,「委托已经照那个形状写好」)。**实测三处**
  都得改:`_bayesmith_fisher` 建块时 `local_block(...)` **不带** `priors=`;
  `fisher_information` 两个分支**都**硬写 `declared = None`,所以图上的 latent
  一律是 improper flat。任一处不改,翻 `include_prior` 得到的是 `KeyError:
  'a_vec'` 或一个空手而归的 `with` 块——**没有一个是「答案错了」的样子**。
  两侧的**算术**确实是一份:接对之后与被删的拼写对**独立 numpy oracle**
  逐比特相同(`0.000000e+00`,不是「在容差内」)。
  **要害不是数目,是第四处**。原函数里有**五条拒绝**,其中三条被测试逐名钉住。
  把它们交给远端会**全部消失**——不是变成另一个异常,是根本不到达:
  `graph_bridge.translate` 把 `NotGaussian` 归为**无责裁定**(捕获、不外抛、
  记在 `Seam` 上),这对「这里有没有精确路线」式的分支调用者是对的设计,而
  `fisher_information(space=...)` 不是那个调用者。实测:Uniform / LogNormal /
  无先验三例,`with` 块提前结束,下一行在一个从未赋值的名字上读 `.values`,
  调用者拿到 `UnboundLocalError`。另两条(声明了 `joint_prior`、params 无名)
  在远端**根本没有对应物**。
  **裁决依据不是「保守」,是计划自己已有的规则**:P1 总原则——**凡图缝会抹掉证据
  的拒绝,住在 to_graph 前的预验证**(首例 `check_noise_std_axis`)。G15 的正文
  只是没有引用它。于是 `_prior_precision` 改名 `_declared_gaussian_priors`:
  **算术删掉,五条准入原地不动**,返回给 `graph_for_information` 的 `priors=`。
  **顺带量出第二处分歧,是本批更有意思的一个**:两包对「什么算高斯先验」的判据
  差**恰好一个拼写**——bayesmith 的 `check_gaussian` 收 `Normal` 与
  `Independent`(`.to_event(1)`)、拒 `ExpandedDistribution`(`.expand([2])`);
  rheplicant 的 `_gaussian_parameters` 三个都拆。两条规则都站得住,但经过一条会
  吞掉「不」的缝,分歧到用户手里就是 `UnboundLocalError`。故过缝的是**规范形**:
  `Normal(loc, scale)` 广播到 latent 自己的形状,一种写法覆盖所有拼写。
  **G15 点名要重测的那件,答案是「仍然够不到」,理由比原来的好**:标量
  `Normal(0, .5)` 配 `(2,)` latent(G9 广播缺陷的入口形状)**根本到不了**门面——
  `Latent.__check_init__` 在构造期就按名字拒绝形状不符。`.expand([2])` 是同一个
  先验补上形状,它到得了,而规范形正是为它。
  地板 `bayesmith>=0.4 → >=0.5`(CLAUDE.md 与 AGENTS.md 成对同改)。
  证据链:`2026-08-28-g15-rheplicant-discharge.md`,探针
  `docs/probes/probe_16_g15_discharge.py`。
- **D49 — CI 连红六条,哪些是环境、哪些是真缺陷(查 CI 时新增)。**
  **【自裁于委托之下,2026-08-28:一条真缺陷、四条守卫写错、一条 workflow 漏配。】**
  交接页把 CI 列为下一位第一件事,判别法是 `2026-08-27-ci-flat-chain.md` §三:
  **问这条断言钉的是大效应还是小差值**。六条逐一过判别法,结论不是「都是抖动」:
  1. **`_reject_a_foreign_block` 是真缺陷**,而且是最反直觉的一种——守卫比较的
     第三个量 `z.z + offset` **不是 `combine` 的不变量**。`combine` 把 x 的任何
     二次型都表达不了的那部分残差移进角落 `rho`:`z.z` 掉 `rho^2`,`offset` 付
     `-rho^2/2`,两者之**和**掉 `1.5*rho^2`。真正的不变量是对数密度自己的常数项
     `offset - z.z/2`,实测两边差 **5.6e-09**,而旧写法差 **21.93**。
     旧写法只在 `rho` 为空时守恒——也就是每一个满秩 epoch,以及**每一个到过这条
     代码路径的 fixture**。它以平台差异的形式浮出:精确共线的设计其 QR 有一个零
     主元,**LAPACK 把它放在哪里**因机器而异(arm64 上第 1 列约到 1.36e-16,零方向
     留在保留块内,`rho` 为空;x86-64 上精确约到 0,质量进角落,`rho^2 = 14.62`)。
     块在两台机器上**都是对的**;守卫在其中一台上以 `StateValidationError` 指控
     「战役被打乱了」——**对正确数据发出的数据完整性指控**。已修,并补
     `tests/evidence/test_chain_block_invariants.py`(12 条),其中三条 fixture 用
     手搭的 `SqrtInfo` 把非空角落造出来(`rho^2` 恰为 25 / 16 / 0),**不依赖平台**。
  2. **`test_the_ratio_is_marginalised_...` 断言的是一条不成立的普遍命题。**
     它在**种子基**上断言「原始梯度与边缘化比值的排序相反」,而那里两个量都是
     ~5e-12——邻近测试自己的注释已经把这个区间称作「一个已经与零无法区分的相干
     误差,其绝对大小由算术在哪里舍入决定」。在偏差**真实**的 fixture 上扫描,
     反转随随机 key 变号(n=5:key 50 反转、key 7 与 key 123 都不反转)。
     **那不是方法的性质。** 换成真正成立的性质:**边缘化比值对参数化(单位)不变,
     原始梯度不变**——把某个 latent 缩放 1e6,报告的比值一个不动,原始梯度的
     argmax 翻到那一维。大效应,不是小差值。
  3–5. **三条绝对壁钟预算**(0.09 ms / 2 ms / 0.05 s)。它们是**刻意收紧**的
     (x6 / x2.6 / 「余量」),各自配着一张变异标定表,而 runner 每次操作约慢四倍,
     余量整个被吃掉。不动那三个数字,也不放宽:`tests/config/inflight_helpers.py`
     新增 `machine_factor()`——一段纯 CPU、与各 pass 同类工作的参照负载,除以本机
     标定值。**每一张标定表都还成立**,因为表里每一行都与干净数随机器同比例移动;
     而代码本身慢 10 倍仍然红,因为参照不跟着动。
  6. **A13 那条**:`A13_JUST_UNDER_THE_FLOOR_MESSAGE` 把 336 Hz 写死在文案里,而
     该 band 真实间距 333.33 Hz、float32 在 7e7 处的 ulp **恰好是 8 Hz**,所以每个
     gap 只能是 328 或 336,中位数落在哪边由 `linspace` 往哪边舍入决定。
     **本模块早已诊断过这件事并建好了 `assert_a13_narrow` / `assert_a13_wide`
     两个派生器**——这一条是**唯一没有被改造的断言**,于是它红而兄弟们全绿,
     读起来像检查坏了而不是 fixture 坏了。已并入同一个派生器(一个生成器带 width
     参数,不是第二份拼写),并补一条**扫描本模块命名空间**的守卫:任何 upper-case
     字符串常量都不得把 channel spacing 写成字面量。
  7. **`tsc` 找不到**:`Coverage (serial)` 跑的是同一套测试,却没有 Suite 那三步
     Node 配置。`8ed385f`(标题正是「the toolchain CI never installed」)只给 Suite
     加了,**Coverage 被漏掉**——与第 6 条同一个形状:一条策略逐处应用,漏了一处。
  **本条的共同教训**:六条里**五条**的根因是「一条规则/一次修复只应用到了它被
  发现的那一处」。判别法本身是对的,但它只在**有人把它拿到每一处**时起作用。
  证据链:`2026-08-28-ci-triage.md`。
- **D50 — `reweight_tol` 那条守卫用的是一个不存在的 fixture(同上批次新增)。**
  **【自裁于委托之下,2026-08-28:改用直接观察转发,放弃数值括号。】**
  `test_reweight_tol_alone_decides_converged` 用「一个容差在步长下、一个在上」
  来证明这个键被读了。它上一次已经被硬化过一轮(不再写死步数,改成搜索一个
  「还在动」的步长),而这次 CI 上**搜索区间内每个步数都精确收敛**,测试用它自己的
  诊断消息失败:「pick a fixture whose reweighting is still moving」。
  **实测:这个 fixture 家族里没有那样的 fixture。** 块是一个 latent 且对它精确
  仿射,数据是经 fit twin 的确定性前向,所以 GLS 解精确重现数据,RADIOMETER 那个
  正比于预测的 sigma 一步之后就不再变。逐个量:`gls_product` 4.7684e-07、
  `gls_pair_product` 8.9391e-07、`at={"g":0.5}` **恰好 0.0**,全部 **2 次迭代**;
  用 `realise:` 往数据里加散射也没用(seed 7 是 1.4304e-06,仍是 2 次)。
  即 `delta` 在 1 次重加权时是 `inf`、在 ≥2 次时是舍入量,**中间没有稳定值**;
  一个「低于 delta」的容差在精确收敛的机器上必须低于零。
  这与 `SQUEEZED` 常量注释记的是同一件事,只是从另一侧看。
  **转发本来就是这条测试自称要立的论点**,于是直接钉转发:monkeypatch
  `inference.iterative_gls`,断言运行声明的 `3.7e-09` 原样到达;兄弟测试断言不声明
  时该键**根本不出现**。这比数值括号**更强**——一个恰好与默认值相同的声明骗不过它。
  变异验证:从 `_GLS_KNOB_SPECS` 删掉 `reweight_tol` 一行,该测试逐名击杀。
  证据链:`2026-08-28-ci-triage.md`。
- **D51 — config 侧钉 inference 内部件的三个测试(Wave B 开波,计划点名要「显式重谈」)。**
  **【自裁于委托之下,2026-08-28:一条现在就搬,两条给出解除条件、留到 `plan` 切换那批。】**
  计划 §五 Wave B 行写着这三条「本波显式重谈,走登记簿」。**逐条量过之后,三条不是
  同一件事**:
  1. **`test_the_defaults_come_from_the_package_and_not_from_here`
     (`test_preflight_fitting.py`)。** 它带着
     `assert (MIN_SWEEPS, DEFAULT_MAX_ITER) == (3, 100)`,而**实测:这是这两个值
     在整个仓库里仅有的 pin**——`tests/inference/` 一处也没有。也就是说,唯一钉住
     这两个数的东西,住在**消费它们的那一层**,而不是声明它们的那一层。
     **这与 Wave B 无关也是缺陷**,所以现在就搬:新增
     `tests/inference/test_plan.py::TestTheDefaultsTheConfigLayerQUOTES`(3 条),
     config 侧只留「消息由常量派生」那一半(那本来就是它的主题),并留下指路注释。
     顺带把 `MIN_DRAWS` 也钉成**派生**的而不是字面 4:经包自己的 `_halves`,
     「两半各二」是方差存在的最小切分,少一个就不是二。
  2. **`test_check_identifiability_is_a_closed_enum_read_from_the_package`。**
     它比较 `_A25_CHECK_MODES == (False, CHECK_ONCE, CHECK_EACH_SWEEP)`——**这是
     派生,不是 pin**,两边读同一批常量。`plan` 切换后只要这两个名字仍从门面导出
     (铁律 1 要求公开名保持),它原样重放。**判定:留守,归「原样重放」列。**
  3. **`test_the_package_guard_this_enum_mirrors_is_still_that_guard`——真正的
     源文本 pin。** 它 `inspect.getsource(SamplingPlan._prepare)`,断言一行**逐字**
     的守卫表达式。它存在的理由是对的,而且第 2 条做不到:值相等看不见 `plan.py`
     里到来的**第四个**模式,因为两边比的是同一批名字。
     **判定:留到 `plan` 切换那一批,解除条件写死在这里**——`_prepare` 是私名,
     切换后它可能不存在、或那条判断住到远端去。届时的保守替代**不是删掉**,而是
     换成**行为等价扫描**:对一串候选值(`False`/`True`/`0`/`1`/`'once'`/
     `'each_sweep'`/`'banana'`/大小写变体/`None`/`2`)逐个比较「config 拒不拒」与
     「包拒不拒」,两个判定必须一致。那比源文本对重排版稳健,对「第四个模式」的
     强度则取决于候选表——所以候选表本身要写明:**加一个模式就要加一个候选**。
  **顺带量到的一件小事**:`MIN_SWEEPS` 的 `#:` 注释说它是 `MIN_REWEIGHTS` 的
  「a third the count」,而两者是 **3 对 5**(0.60)。改成陈述**成立的关系**
  (一次 sweep 比一次 reweight 贵,所以问之前花得更少),而不是改一个比例数字
  ——比例本来就不是设计。
  证据链:`2026-08-28-wave-B-opening.md`。

- **D52 — 两条收敛守卫的文案在缝的两侧措辞不同。**
  **【自裁于委托之下,2026-08-28:采纳远端措辞,两条 `match=` 改籍。】**
  `require_convergence` 的两条消息在两侧**前缀相同**——`wiener_solve/gcr_sample
  cannot reach require_convergence at this precision` 与 `... did not converge`
  ——所以按前缀匹配的五处 pin 全部原样通过。差的是细节:近端写
  「the normal operator's **condition number**」,远端写「the **condition bound**」。
  **远端是对的。** 守卫读的是 `condition_bound()`(上界),不是
  `condition_estimate()`(测量值,偏低),而近端**自己的 docstring** 早就写着
  「Do not divide an accuracy target by this number」并指向 bound。近端的消息
  因此一直在用一个它自己文档里判为错误的名字。
  **实测:`src/` 里没有任何东西解析这两条消息**(config 层也不解析),所以文案
  只被测试钉着,不是机器契约。两条 `match="condition number"` 改为
  `match="condition bound"`;那条 docstring 里指向已不存在的
  `inference/linear.py::_condition_number` 的引用同批更正。
  证据链:`2026-08-28-wave-B-linear.md`。
- **D53 — 同一段线性代数,两侧拼写不同:均值差 ~1 ulp,抽样对同一个 key 不同。**
  **【自裁于委托之下,2026-08-28:采纳远端拼写;三条按单 key 钉住的守卫改断言
  与 key 无关的性质。】**
  一个原因,两个后果。远端把噪声当作 `Precision` **算子**收下,近端收的是一个
  权重数组,于是同一个量在两侧的写法不同:
  1. **均值(`wiener_solve`)。** 近端 `weight = 1/σ²` 再 `weight * r`(两次舍入),
     远端 `r / σ²`(一次)。**σ 是标量时两者逐比特相同**(实测:
     `test_linear_blocks` 的 `LOAD_NOISE=0.1` 一路相同);**σ 是数组时差 ~1 ulp**。
     方向不是中性的:拿 float64 稠密解做基准,GLS 那个 fixture 上
     **远端 1.12e-07、近端 1.57e-07**——远端更接近真解,而它**报告的残差更大**
     (1.78e-07 对 7.02e-08)。这正是两包 docstring 都在讲的那句「残差不是误差」,
     第一次被量出来:近端报了一个更小的残差,同时离真解更远。
  2. **抽样(`gcr_sample`)。** 涨落项的白化,近端 `sqrt(weight) * ω`,远端
     `ω / σ`(`Precision.whiten`)。于是**同一个 key 给出不同的抽样**——统计上等价,
     逐值不等。
  第 2 条的后果不是数值而是**守卫可测性**:三条测试用**一个 key** 钉住「这个解
  必然被拒」,而那个结果**一直是 key 相关的**,只是没人量过。逐条实测(20 个 key):
  `test_float32_is_refused_however_tight_the_tolerance` 切换前 **15/20** 被拒、
  切换后 **12/20**;`test_the_conjugate_convergence_guard_still_raises_equinox`
  前 **14/20**、后 **10/20**。`key(0)`/`key(1)` 只是换了边。
  **所以这不是本次切换制造的脆弱,是本次切换照出来的。** 三条改为断言与 key
  无关的东西:地板本身(κ·eps = **3.81** > 1e-3,两侧相同)、**每一条**拒绝的
  **种类**(实测两侧都是 100% 的 "at this precision",一条 "did not converge"
  都没有——因为在精度地板之下,「调紧 tol」是错的建议),以及「至少有一个 key
  被拒」这条「守卫还能失败吗」的自检。
  第三条是 `test_gls.py::test_a_tolerance_below_the_epsilon_never_converges`,
  它钉的是**退出方式**(`not converged`、`iterations == 100`),而那是 float32
  迭代停在不动点**上方一个 ulp** 还是**恰好落在其上**的产物:实测同一个 fixture
  一侧 `delta = 6.98e-08`、另一侧 `delta = 0.0`,而 `0.0` 让任何容差都被满足,
  于是第 8 步就"收敛"。**两个数都低于 float32 的 eps**,而那才是这条测试一直
  在讲的事(「a relative step of 1e-8 is rounding rather than a measurement」)。
  改为断言 `delta < eps`,并保留「两条路走到同一个解」那两行。
  **这与 2026-08-28 CI 分诊在 arm64/x86-64 上遇到的是同一个形状**,读法也相同。
  证据链:`2026-08-28-wave-B-linear.md`、`probe_17_linear_solve_seam.py`。

- **D54 — `SWITCHED` 的别名表把 `test_noise_logdet.py` 记成了 `gls` 的 cross-check。**
  **【自裁于委托之下,2026-08-28:改别名,`gls.py` 进 `SWITCHED`。这是一次判据改动,
  所以走登记簿。】**
  `tests/test_migration_records.py` 的别名表写着 `"gls": ("gls", "noise_logdet")`。
  **实测该文件与 `gls` 无关**:4 个 test 函数,`iterative_gls` 出现 **0 次**,
  四条全部比较对数行列式估计量——那是 **B1**,归 `plan`/`engines` 行;
  `uncertainty` 也别名同一个文件(Fisher 那半),**那一条是对的**。
  **这条别名在两个分支上都承重**,而两个方向都错:
  1. `gls.py` 未切时,它断言「这个模块有 cross-check」——**用一个不测它的文件
     满足了这个断言**,也就是说这条守卫对 `gls` 从来就是绿的、且绿得没有理由。
  2. `gls.py` 一旦进 `SWITCHED`,同一条别名会反过来要求**删掉那个文件**
     (「a switched module that still has a cross-check ... would go green
     forever」)。**为了记录一次它从未守护过的切换,把 B1 的台账删掉。**
  **选项**:(a) 不把 `gls.py` 记进 `SWITCHED`,让台账漏记一个已切模块;
  (b) 改别名为 `("gls",)`,并把 `gls.py` 记进 `SWITCHED`。
  **取 (b)**,理由有三:漏记正是风险登记里 **W7** 点名的失效形态;
  别名的错误是**已实测的**,不是判断题;而且 (a) 会让第 1 条那个「绿得没有理由」
  的断言继续存在。改后 `tests/crosscheck/` 里没有任何文件名含 `gls`,
  于是 SWITCHED 分支的 `assert not found` 成立,`uncertainty` 那一条不受影响
  (它仍然找得到 `noise_logdet`,而它**确实**该找到)。
  **`gls` 的独立 oracle 不随之消失**:远端自己的
  `tests/exact/test_gls.py::test_the_fixed_point_is_the_unbiased_estimator_not_the_gls_biased_one`
  就是它,按铁律 2 的另一支**指认而非改籍**。
  证据链:`2026-08-28-wave-B-gls-opening.md` §三 第 3 条与 §5。

- **D55 — B1 修好之后,`inference.noise.include_logdet: false` 无处安放。**
  **【自裁于委托之下,2026-08-28:plan 两个出口**拒绝**该声明,不向下透传。】**
  B1 的修法本身不是判断题(缺的项就是缺的),但它照出一件计划没预见的事:
  config 的 **A49** 把 `include_logdet` 定为**必须显式声明**的键,理由写在拒绝
  文案里——`false` 是「a DIFFERENT estimator, biased high by (1 + f^2)」。
  而这个声明**今天只到 `chi2` 诊断**(`docs/config-inference.md` 自己写着
  「`chi2` is the **first** consumer」),从不到 plan。
  于是修好 B1 之前和之后,各有一格是错的,方向相反:

  | 声明 | 修前的 plan | 只修 B1 之后 |
  |---|---|---|
  | `true` | GLS ✗(**这就是 B1**) | 全密度 ✓ |
  | `false` | GLS ✓(碰巧对) | 全密度 ✗(**新的静默覆盖**) |

  **只修 B1 是拿一个静默的不一致换另一个**,而后者更糟:它静默地推翻了一条
  用户**明确写下**的声明,而 A49 这条检查存在的全部理由就是防止这件事。
  **选项**:(a) 只修 B1,`false` 被静默覆盖;(b) 把 flag 透传进 `Conditioning`
  与 `SamplingPlan`;(c) plan 两个出口**拒绝** `false`,并指出仍然honour它的路。
  **取 (c)**,理由有三:
  1. (b) 会让同一个词在两个出口意思不同——GLS 是个**合法的点估计**,但**不是
     后验**,所以 `plan.sample` 对 `false` 没有连贯的解释。「同一个词在两个出口
     意思不同」**正是 B1 本身**,拿它当修法是把病灶搬了个家。
  2. (c) 不增加公共签名,也不新造 schema 的 A 码——A 码是 schema 层的东西,
     为一个出口的不连贯造一个新码是把 B1 的范围撑大。拒绝写在
     `sections/exits.py`,与 `_a29_estimate_takes_no_seed` 同处,那是先例。
  3. (c) **可逆**:将来若真要透传,替换掉这条拒绝即可,而被拒绝过的人不会
     因此拿到过错的答案。
  **实测该组合在测试与文档里都没有被走过**(`include_logdet` 为 false 的
  32 处引用里,config 侧只有 `exit_helpers.GCR_RADIOMETER` 与
  `test_config_section_noise.py`,前者只喂 `conjugate.gcr`),所以这条拒绝
  不推翻任何现存文档。反空洞对照:同一份文档写 `true` 时两个出口都照跑。
  证据链:`src/rheplicant/config/sections/exits.py::_a49_is_not_honourable_by_a_plan`、
  `tests/config/test_config_exits_plan.py::TestAPlanRefusesTheGLSDeclaration`。

- **D56 —(待裁决,不自裁)`plan`/`engines` 这一行的数值早已迁完,剩下的不是迁移。**
  **【状态:待 owner 裁决。本会话只提交测量与建议,没有改 §四 的表,也没有动代码。】**
  开波普查与形状测量见 `2026-08-28-wave-B-plan-opening.md`。三条实测:
  1. **`engines.py` 没有任何自有线性代数**——`solve`/`cholesky`/`inv`/`lstsq`/
     `eigh`/`svd`/`qr` 全文件零出现。它建 `linear_operator` 然后调
     `wiener_solve`/`gcr_sample`,**而这两个名字自 `linear` 批次起已是
     `bayesmith.exact.solve` 的门面**(`linear.py:158`、`:95`)。
     **也就是说这一行的数值部分随上上批已经过去了。**
  2. **26 条拒绝里有 16 条**(声明语法 8 + 分区合法性 3 + engine 指派 5)守的是
     **「模改者自己声明分区」这个面**;而远端
     `dispatch.partition(graph, *, key=None)` **只吃图**,分区是推导的。
     这 16 条**在远端无处可托**,不是远端漏了,是远端的模改者犯不了这些错。
  3. 该行的 cross-check 已存在且通过(`test_dispatch.py`,10 例),**并且正是
     它发现了 B1**——这一行已经交付过它的价值。
  **建议**(可推翻):把 §四 4.2 这一行从「迁移」改判为「互证」。
  **不建议**:把 `SamplingPlan` 的声明面转成图再交给 `dispatch`——那会有
  **两个分区器**,一个声明一个推导,不一致时以谁为准没有答案(铁律 3)。
  **为什么不自裁**:与 D55 不同,这里**没有被迫的选择**。D55 是「修了 B1 就必然
  有一格变错,必须挑一个」;这一条**两条路都不坏,而现状是稳定的**——不改判,
  代码一行不动,什么也不会坏。改判则会改变 §六 对这两个模块的处置,那是 owner
  该看见的。**推翻它需要的一条测量我没有**:远端若将来接受**声明的**分区,
  §三那 16 条就有了托付对象,判定应当重来。

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
  **【已落地 2026-08-27,bayesmith 侧;rheplicant 一行未动】** `bayesmith.optimize`
  给出六个名字:`fit`(图入口,联合 MAP 或 `names=` 的块坐标)、`minimize`(任意
  标量目标)、`Fit`、`check_loss_sense`、`MINIMIZE`/`MAXIMIZE`。目标是**全密度**
  (D7),逐 latent 步长,无收敛判据。两条新裁决 **D32**(两个入口)与
  **D33**(发散即拒绝)。**接线在 Wave B/C,且铁律 5 要求先发一版**——
  CHANGELOG 的 `Unreleased` 段自本批起不再为空。
  证据链:`2026-08-27-p2-g2-fit.md`。
- **G3 `exact.chain`**:RTS/Kalman + `linked` 转移;自含于
  bayesmith.evidence.sqrtinfo 之上(chain→sqrtinfo 依赖边,故与证据族
  同波)。
  **【已落地 2026-08-27,bayesmith 侧;落址 `evidence/chain.py`】** 名字写的是
  `exact.chain`,但同一行写着「自含于 `evidence.sqrtinfo` 之上」,而 `SqrtInfo` 与
  `marginalise_arrays` 都住 `evidence/`——**跟依赖走,不跟名字走**。六个名字全数,
  容器 `ChainMemory` 按 G6 §2.2 留守。oracle 是 numpy 装配的**稠密联合**,四个探点、
  一条 width-3 旋转链、20 epoch,rel **1e-9**。**六个常数逐个删掉再量**,外加一条
  证明这一整类必须存在的测试(删掉常数后 `factor`/`target` 逐比特不变)。
  `smooth` 用稠密块三对角解而非经典 RTS 后向传递,理由在其 docstring。
  变异 6/6。证据链:`2026-08-27-g3-chain.md`。
- **G4 `exact.reduced_basis`**。
  **【数组级五个名字已落地 2026-08-27,bayesmith 侧】** `orthonormal_transform` /
  `orthonormalise` / `numerical_rank` / `select_svd` / `select_greedy`。范围由 G6
  登记页 §2.4 定:两个容器留守(D12),八个属于 G4,本批做五个**数组级**的;
  另外三个(`score_directions`、`build_reduced_basis`、`basis_fidelity`)都要先有
  容器那一侧的接线,**跟 Wave C 走**。每个入口**拒绝 ambient float32**(**D41**)。
  oracle 全是 numpy 且大多算**性质**(Gram = I、张成逐前缀嵌套、残差投影),
  唯一的过程复算是「重正交化那一遍值不值」,而那条问的就是两个过程的差。
  证据链:`2026-08-27-g4-reduced-basis.md`。
- **G5 `bayesmith.amortize`**:NPE 族(D10 子裁决先决)。
  **【已落地 2026-08-28,bayesmith 一侧;rheplicant 一行未动】** 四个名字:
  `NeuralPosterior`(条件高斯混合)、`train_posterior`、`TrainingHistory`、
  `MIN_SCALE`。**范围是读签名读出来的,不是划出来的**——上游四件公开件里三件只收
  数组,一件收 `ParameterSpace`+pipeline+`NoiseModel`,所以 `simulate_pairs`
  **留守**(**D42**),本模块**数组级**、不收 `Graph`。
  oracle 是**闭式的 numpy 线性高斯后验**,外加两条互不相关的仪器:格点求积
  (密度是不是密度,且对标准化的雅可比敏感)与采样器(与求积的前两阶矩比)。
  **接受 float32**,与 `reduced_basis`(D41)/`diagnose.local` 相反,理由写在
  模块 docstring:那两条底下压着秩判决与 Gram 矩阵,这里没有判决。
  新裁决 **D42**(范围)与 **D43**(发散的两个结果,只拒一个)。
  **顺带修掉一处与本项无关的过期**:README 的 `1269 tests` 已经漂了 **204** 条,
  横跨六个批次,而**没有守卫**;补了 `tests/test_readme_count.py`。
  证据链:`2026-08-28-g5-amortize.md`。
- **G6 证据消费面**:D12 包装所需缺口,逐项登记。
  **【7 个里的 6 个已落地 2026-08-28,bayesmith 一侧;rheplicant 一行未动】**
  `residual_summary` + `ResidualSummary`(§9.3 的 χ²、自由度、命名模板投影,压缩期算)、
  `epoch_residuals`、`refuse_mixed_templates`、`template_modes`(**D45**)、
  `held_out_z`、`shrinkage_power`/`shrinkage_report`、`systematic_floor` +
  `tightest_direction`。全部**数组级**:容器按 D12 留守,所以返回的是 `SqrtInfo`、
  NamedTuple 与 dict,不是上游的 `EpochResidual`/`HeldOut`。
  第 7 个 `compress_reduced_basis` 归 **Wave C**(**D44**),而它的数值核心
  **本包已经有了**——把基的行当 design 传给 `compress`,实测 1.76e-16。
  oracle 三种且互不相关:`held_out_z` 对**从头重拟**的留一后验(最坏 1.5e-16)、
  χ² 对**它自己的零分布**(400 epoch 的均值)、`shrinkage_power` 对**整数比**的幂律
  (−0.5 与 −1.0,15 位)。
  **顺带查出并修掉一个 0.4.0 就带着的缺陷**:`compress` 不在 `seen` 上做选择,
  于是一个被旗标的样本上的 NaN 会污染 `target`(或 design 里的污染 `factor`),
  而 `information()` 依然有限且良态——campaign 审计报健康,每一个密度是 NaN。
  证据链:`2026-08-28-g6-consumption.md`。
  **【已逐项登记 2026-08-27;实现仍属 Wave D】** 证据族九个模块共 **40** 个公开名,
  逐个判决(探针 `docs/probes/probe_14_g6_enumeration.py`,判决表在探针里、总数由它
  求和):**HAVE 6**(远端已有同名对应物——**线索不是判决**,只有 `SqrtInfo` 有
  cross-check)、**STAY 12**(容器/格式/声明,按 D12 与 §〇)、**G3 6**、**G4 8**
  (各归自己的条目,**不重复登记进 G6**——不加区分的「34 个缺口」会让两处都以为
  对方在做)、**G6 7**、**OPEN 1**(`memory.reject_bad_term`:准入判据还是构造期
  契约,要读代码才知道,那一批的第一件事)。
  **所以 G6 = 7 个确定 + 1 个待判,不是 34。**
  **【待判项已结清 2026-08-28:`memory.reject_bad_term` 判为**留守**,G6 = 7,
  STAY 从 12 变 13,OPEN 归零。】** 判据不是那一页原本写的「结构还是统计」——读完
  发现它六条检查里两类都有,所以那条判据判不了它。改用三条**可测**的事实:
  (1) 唯一的两个调用方 `BayesMemory.remember` 与 `ChainMemory.remember` 都是留守
  容器;(2) 远端**没有增量累加器**(`src/bayesmith/` 里没有任何 `remember`/`admit`/
  `add_term`/`append`,`compress_campaign` 是一张图一次调用),所以「这个别人建的项
  能不能加进来」在那边从不被问——迁这条规则要**先发明它所守的累加器**;(3) 它读的
  `epoch_id`/`prior_share`/`estimator`/`inputs`/`represents` 在 `bayesmith/evidence/`
  七个模块里出现 **0 次**。**不拆**,理由是它自己的 docstring:「一条只在一个累加器
  里生效的规则,是一条有办法绕过去的规则」——跨两个包拆是同一缺陷高一层。
  证据链:`2026-08-28-g6-consumption.md` §一。
  **两条限制写在页上以免被读成完备**:按公开名数,**私名普查仍要做**;同名不等于
  同题。证据链:`2026-08-27-g6-enumeration.md`。
- **G7 bridge 补齐**:`init_to_declared`、`predict_from_samples`、
  `propagate_covariance`/`push_forward`(config `predict` kind 逐 fisher
  product 调用);约定:单参数 `model(observed=None)`、latent 同名 site、
  内部节点名不漏出。
- **G8(后置)**:多块 gcr+mh 论证与实现。
- **G9 复数域**(家由 D18 定)。
  **【全量已落地 2026-08-27,bayesmith 侧】** 四面里**两面本来就是好的**(vmap、
  log 空间——实测,补了守卫),两面坏在同一件事上:**在 DOMAIN 里做了 PARTS 的事**。
  `dense_operator` 改为按实自由度铺(`n` 个复条目占 **2n** 行,实部半段在前),
  `log_weight` 改为在 parts 空间取二次型且**按叶子求和**。`diagnose` 的拒绝**保留**
  (**D38**)。**顺带查出并修掉一个 0.4.0 就带着的缺陷**:`include_prior=True` 的
  先验曲率对标量 `prior_std` 广播到了**每一个非对角元**——图路线够不到它,复数块
  永远够得到。证据链:`2026-08-27-p2-g9-full.md`。
  **最小面(P1 先决,定义)**:
  `wiener_solve`/`gcr_sample` 接受复 latent,均值路径经 `_real_parts`
  约定与两半恒等式测试;矩验收随 P1 例 2。全量(vmap、log 空间、Fisher
  的复数面)随 Wave B。
  **【owner 已拍板 2026-08-26:声明路径提进最小面。】** 起因是实测而非推测:
  求解面做完后发现**复 latent 在 Graph 里根本无法声明**——`block.dtype` 取自
  线性化域、域取自先验抽样,而 numpyro 无任何分布抽出复数。于是本行原本的
  「最小面」与它自己的验收标准(P1 例 2:**经适配器**解出复数 alm)之间缺一段
  路,因为适配器交出的是 `Graph`。裁决取 (a):把声明路径**提进最小面**,而不是
  归 G9 全量、让一个先决的验收晚于依赖它的波次。**范围随之明确为三件**:
  (1) 复 latent 可声明(其先验按 re/im 两个实分量解释,每半携带
  `scale**2`,与 `variance_parts` 的复制约定同一句话);(2) 建块路径接受它,
  产出复 dtype、复 `prior_mean`、实 `prior_std`;(3) 线性检查在复域上可跑。
  **仍属 G9 全量(Wave B)**:`diagnose` 族对复 latent 的拒绝、`exact.correct`
  的 SNIS 权重、vmap、log 空间、Fisher 的复数面。
- **G10 分区执行面完形**(D14 范围):在 `sample_factors` 上加
  (i) per-sweep 回调(χ² 轨迹、identifiability 节奏、块残差),
  (ii) sweep 形 estimate,(iii) 声明分区入口(绕探测与政策门,文档化
  「声明者负责」)。**不另起执行器。**
  **【已落地 2026-08-27,bayesmith 侧】** 三件都落在 `sample_factors` 上,没有第二个
  执行器。(i) `on_sweep` + `SweepReport`(index / warmup / values / **log_joint** /
  逐块 CG 残差),**带 NUTS 余项时拒绝**(D35);χ² 用 `log_joint` 拼写——高斯模型下
  `-2 log_joint` 就是 χ² 差一个不随轨迹动的常数,而非高斯模型下只有前者存在。
  (ii) `estimate_factors` + `SweepEstimate`,精确块 `wiener_solve`、余项交 `fit`
  (D36);三种块(gcr / log-gcr / nuts)同一个 plan 里一起估出来,而
  `InferencePlan.estimate` 对这张图**直接拒绝**。(iii) `declared_partition`,
  探测数实测为 **0**,不完整覆盖**拒绝**(D34)。
  证据链:`2026-08-27-p2-g10-g12.md`。
- **G11 结构化拒绝载荷**:仿射类异常带 errors=/rtol=/failed;
  NotGaussian/NotLogLinear 带判别字段。先于任何委托检查的波次。
- **G12 冻结在当前值的 gcr(移址)**:经 G10(iii) 的声明分区路径暴露
  `sample_factors` rebuild 分支的既有语义;若保留 gibbs.py 侧模式,对
  `method="gcr+mh"` **构造期拒绝**;注记为近似声明,非正确性证明。
  **【已落地 2026-08-27,随 G10】** 三件都按本行做了,而**触发条件与本行的读法
  差一层,值得写下来**:rebuild 分支由 `_sigma_needs_rebuild` 决定,而它问的是
  「有没有**块外**的 latent 到达观测节点」。所以**两块以上**的 plan 走 rebuild
  (sigma 冻在块自身当前值),**整图一块**的 plan 走 hoist(sigma 冻在**先验中心**)
  ——后者是另一个近似,误差也不同。实测:单 latent radiometer 的
  `_sigma_needs_rebuild` 返回 **False**。两条都不是正确性证明,注记照本行写成
  「近似声明:历史依赖核,非严格不变」。
  证据链:`2026-08-27-p2-g10-g12.md` §四。
- **G13 图级联合先验**:`JeffreysPrior(over=…)` 的图侧声明与
  `to_numpyro` factor site 读取。Wave A 的 priors/numpyro_bridge 之门。
  **【实现已落地 0.4.0;e-RHINO 侧接线已落地 2026-08-27】** `to_graph` 不再拒绝
  `joint_prior`,而是把被覆盖的 latent 声明成 `ImproperUniform` 并调
  `bayesmith.joint_prior(...)`。接线批次同时量出**两条语义差**并登记为 **D24**
  (行序)与 **D25**(float32 下的新拒绝)。证据链:
  `2026-08-27-wave-A-g13-wiring.md`。
- **G14 measured-κ 诊断**(D15(a)):`condition_estimate` 的对应物,
  显式标注不可作守卫;~~随 Wave B 的 linear 工作落地~~。
  **【已落地 2026-08-27,bayesmith 侧,提前于 Wave B】** `exact/solve.py::
  condition_estimate` 与 `exact/conditioning.py::extreme_eigenvalues`。
  「不可作守卫」不是一句注记而是一条 **AST 双向扫描**(D37)。偏差重测:
  `geomspace(1, 1e7, 50)` 上 2000 次迭代后 λ_min 仍是 **501.2**(真值 1.0),
  报出 κ = 2.00e4(真值 1e7)。**提前做**是因为它自足、不依赖 Wave B 的任何东西,
  而 P2 余项本来就在它前面。证据链:`2026-08-27-p2-g14-condition-estimate.md`。
- **G15 带先验的局部块**(切 `uncertainty` 时新增):一个**非线性**模型在某点的
  局部块,**并且携带各 latent 声明的先验**。今日两个构造器各缺一半——
  `local_block` 给对雅可比而**故意不带先验**(它自己的 module docstring 写着:
  把它交给 `fisher_information(include_prior=True)` **必须**在空字典上响亮失败,
  而不是悄悄折进一个没人声明的曲率);`unchecked_operator` 带先验,却在**域的零点**
  取切线(仿射映射处处一个切线),对一条幂律那是错的雅可比。两者都没错,只是还
  没有第三个。
  **后果**:rheplicant 的 `fisher_information(space=...)` 的先验曲率**暂时留守**,
  作为一条**有解除条件的**延期(见 `uncertainty._prior_precision` 的 docstring):
  G15 落地并发布之后,该函数删除、调用改为 `include_prior=space is not None`——
  委托已经照那个形状写好,**只有那一行会变**。
  证据链:`2026-08-27-wave-A-uncertainty-fisher.md`。
  **【bayesmith 一侧已落地 2026-08-27;rheplicant 一侧仍待发布】**
  第三个构造器是 `local_block(..., priors=True)`——**关键字而非第二个函数**,默认
  一字未变,因为 `diagnose/local.py` 那段「本块不带先验」的论证每个字都还成立。
  先验经 `_env_before` 读(全包唯一一处把声明变成 `(shape, dtype, prior_mean,
  prior_std)` 的地方),所以没有第二份拼写,且它的 `check_gaussian` 一起来了。
  缺口是量出来的:`mu = a x^b` 上 `unchecked_operator` 给 `a log x`(零点切线),
  `b=2` 处的是 `a x² log x`,除 x=1 外每行都不同。
  **解除条件的另一半仍未兑现**,而它被铁律 5 挡着——`priors=True` 只在 `Unreleased`
  里。发布之后改那一行、重测数字,**并重跑 e-RHINO 全套**:G9 全量修掉的先验广播
  缺陷今天够不到门面(它永远传 `include_prior=False`),而这条改动正是让它够得到的。
  证据链:`2026-08-27-g15-local-block-priors.md`。

每 G 项 = 实现 + 独立 oracle 测试 + §四式记录页(铁律 6 计时)。

> **记录页的家,实测后更正(2026-08-26,G11 落地时)**:**不是**
> `docs/migration/`。那个目录是个**闭集**——`tests/test_migration_records.py`
> 从旧 spec §四 的表格推导出模块名、按 rheplicant 模块名索引、并**双向**断言
> (`pages == PAGED_TODAY`、`pages <= spec_names | OUT_OF_LEDGER`),外加五节
> 标题检查。往里放一个 G 项页面会同时打红三条守卫。G 项记录页住本目录的波次
> 执行页(§十 已为它定的家),但**保留 §四 那五节的形式**:fixture / 数值一致
> / 拒绝一致 / 独立 oracle / 有意的差异。先例见 `2026-08-26-wave-P2a.md` §二。
> 若要给 G 项单开目录,那是一条新裁决。

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
  **【已完成 2026-08-27,本行两处按实测更正】**(1) `push_forward` **不经 G7**,
  留守——它是 `jax.vmap(forward)`,零贝叶斯数值,与 `predict_from_samples` 同一
  理由(**D31**);`propagate_covariance` 照本行经 G7 切了。(2) `numpyro_bridge`
  与 `uncertainty` **各只切了一半**,按契约留守其余,因此两者都**不在 `SWITCHED`
  里**且 cross-check **保留**——先例见 `2026-08-27-wave-A-numpyro-bridge.md` §五,
  而 `uncertainty` 这一侧「还能不能失败」是**变异量出来的**(U6),不是推的。
  证据链:`2026-08-27-wave-A-uncertainty-covariance.md`。
- **Wave B(求解与计划;先决 ~~D7+D8+D14~~ **四门已拍 2026-08-26** +G1+
  **G2**+G9+G10+G12,外加 **D17 的双跑 diff 协议先跑完**)**:`linear` 求解面、`gls`(D19 已拍)、
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
  **(0) 先把这一批提交,再跑变异集。** 第 (1) 条用 `git checkout` 而不用
  `cp` 是对的(`cp` 在同一秒内恢复会留下 Python 会复用的字节码,记下一个假
  SURVIVED),但 `git checkout` 以 **HEAD** 为准:在一棵带着未提交改动的树上,
  它回退的不是变异,是那一批还没提交的全部工作。2026-08-27 实测:一次变异集
  超时被杀、树里留着变异,随后的 `git checkout -- src/` 把整批未提交的 G1
  源码改动一并抹掉(`tests/` 侥幸幸存,因为变异点都在 `src/`)。这条不削弱
  协议——`git checkout` 之所以更好正是因为 HEAD 是参照,那就要求 HEAD 已经是
  你想要回的东西。证据:`2026-08-27-wave-P2-G1.md` §五。
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
| P3 Wave B | 3–5 | ~~D7/D8/D14~~ 已拍 + D17 协议先跑 + G1/G2/G9/G10/G12 |
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

> **两条都已按 §六 五行协议实跑,双双击杀**(P0 批次,证据与逐条判据见
> `2026-08-26-wave-P0.md` §四(ii))。下面各行的「指名红」是登记的期望值,
> 不是跑过的记录——每波扩充本清单时,新增行同样先登记后实跑。

1. **first_fit**:bayesmith `dispatch/factor.py`
   `if all(compatible(name, member) ...)` → `if True:`;指名红:e-RHINO
   `test_auto_partition.py::TestTheMultilinearSplit::test_the_coupled_factor_gets_a_block_of_its_own`
   与 `::test_the_derived_partition_recovers_the_truth`;观测退出码 1。
2. **log 位移**:bayesmith `exact/loglinear.py`
   `y = jnp.log(observed) + fractional**2 / 2.0` → 去位移;指名红:
   e-RHINO `test_loglinear.py::TestTheNoiseTransform::test_the_leading_order_mean_shift_is_added_back`;
   观测退出码 1。
3. **G11 载荷**(P1 批次新增,**已实跑,KILLED**):bayesmith `errors.py`
   `AffinityRefused.__init__` 的 `self.failed = tuple(failed)` → `self.failed = ()`;
   指名红:e-RHINO
   `tests/inference/test_graph_bridge.py::TestTranslateBringsRefusalsBackInThisPackagesClasses::test_the_translated_refusal_carries_the_probe_numbers`;
   观测退出码 **1**,且该条是**唯一**的红。这一行正是 P2a 记录页 §四 预告的那条
   ——当时无跨仓消费者,`translate` 尚不存在,故只登记不实跑;今日两者都在了。
4. **复数 join**(P1 批次新增,**已实跑,KILLED**):bayesmith `exact/block.py`
   `real_parts` 的 join,`parts[n][0] + 1j * parts[n][1]` → `+ 0j *`;指名红:
   e-RHINO `tests/seam/test_p1_ten_examples.py::TestExample2ComplexAlm` 的**三条**
   (稠密均值、虚部被数据约束、GCR 矩);观测退出码 **1**(x64 会话),
   float32 会话退出码 0——因为复数面只在 `tests/seam/` 被跨仓消费,这本身就是
   「哪一侧承重」的一次读数。

### Wave A / `identifiability`(2026-08-27,**第一组真正的跨仓变异**,7/7 击杀)

此前每组变异都在一个仓内跑,因为 rheplicant 侧没有消费者。这七条是**改 bayesmith、
看 e-RHINO 红**,详情与逐条红名单见 `2026-08-27-wave-A-identifiability.md` §五。

| # | 变异(bayesmith) | e-RHINO 红 |
|---|---|---|
| W1 | `diagnose/identifiability.py` 秩切点 `>` → `>=` | 1 |
| W2 | 同上,雅可比不做列归一化 | 9 |
| W3 | 同上,零列除以自己的零范数 | 2 |
| W4 | 同上,谱不补齐到 `n_par` | 1 |
| W5 | 同上,SVD 永不索取完整左因子 | 4 |
| W6 | `diagnose/local.py` 去掉图侧精度拒绝 | 2(含 `test_a_model_pinned_to_float32_is_refused`) |
| W7 | `tests/test_migration_records.py` 的 `SWITCHED` 漏记一个已切模块 | 1(本条对 bayesmith 套件跑) |

### Wave A / G1 接线(2026-08-27,4/4 击杀)

详情见 `2026-08-27-wave-A-g1-wiring.md` §四。

| # | 变异 | 仓 | 指名红 |
|---|---|---|---|
| X1 | 掩码保留 flags 的极性(不取反) | e-RHINO | `test_the_mask_is_the_negation_of_the_flags` |
| X2 | 声明的 scale 保留 `inf` | e-RHINO | `test_the_declared_scale_is_finite_where_the_flags_are` |
| X3 | 每张图都给一个满掩码 | e-RHINO | `test_an_unflagged_noise_model_declares_no_mask_at_all`(6 红) |
| X4 | `graph/trace.py` 把节点的 mask 丢掉 | **bayesmith** | `test_the_mask_is_the_negation_of_the_flags` |

### Wave A / `sensitivity`(2026-08-27,6 条 4 杀 2 存)

详情见 `2026-08-27-wave-A-sensitivity.md` §五。**两条幸存都逐条归因**,不是缺口:
S3 是「这个文件到不了那条远端守卫」(模块 1 的同一条 W6 是红的),S6 当时未能分辨,
留成开放项,已由下一批结清(见下)。

| # | 变异 | 仓 | 判决 |
|---|---|---|---|
| S1 | shift 丢掉后验 sigma 缩放 | bayesmith | KILLED(3 红) |
| S2 | 先验的二次拉力减半 | bayesmith | KILLED(20 红) |
| S3 | 远端去掉图侧精度守卫 | bayesmith | SURVIVED(已归因) |
| S4 | 门面改吃 bayesmith 的默认展开点 | e-RHINO | KILLED(8 红) |
| S5 | 跳过先验预检查 | e-RHINO | KILLED(1 红) |
| S6 | 不把值加宽到 float64 | e-RHINO | SURVIVED → 见下,已结清 |

### Wave A / S6 结清(2026-08-27,4/4 击杀)

详情见 `2026-08-27-wave-A-s6-widened.md` §七。S6 的幸存是**fixture 缺失**,不是死代码;
追到底之后发现承重的是 `astype` **剥掉弱类型**那一半,而不是加宽那一半。

| # | 变异 | 仓 | 指名红 |
|---|---|---|---|
| M1 | `_widened` → 恒等(S6 原条) | e-RHINO | `test_the_verdict_comes_back_in_double`、`test_a_weak_float64_init_does_not_carry_a_float32_model`;**旧 64 条一条不红** |
| M2 | 只 cast 真 float32,放过弱 float64 | e-RHINO | `test_a_weak_float64_init_does_not_carry_a_float32_model`(唯一红) |
| M3 | 给新 fixture 加上 x64 | e-RHINO | `test_the_fixture_really_is_declared_in_single_precision`(兄弟断言) |
| M4 | M1 + 远端 `refuse_single_precision` 一并去掉 | 两仓 | `test_the_verdict_comes_back_in_double`,红在**数值一致**断言而非 dtype 断言 |

### Wave A / G13 接线(2026-08-27,5/5 击杀,**第一轮 3/5**)

详情见 `2026-08-27-wave-A-g13-wiring.md` §六。**第一轮的两条幸存都是真洞**,
修好后重跑全杀;两轮之间还踩了一次第 (0) 条的第二半(修补未提交,被变异脚本自己的
开场 `git checkout` 回退,输出与「修补不起作用」不可分辨)。

| # | 变异 | 仓 | 第一轮 | 修好后 |
|---|---|---|---|---|
| N1 | 声明根本不进图 | e-RHINO | KILLED(11) | KILLED(13) |
| N2 | 所有 latent 一律声明成 flat | e-RHINO | **SURVIVED**(守卫用了一个没有 joint prior 的空间,两种读法都说「未覆盖」) | KILLED(1) |
| N3 | 翻译时丢掉 `rank_rtol` | e-RHINO | KILLED(1) | KILLED(1) |
| N4 | 块**反序**过缝 | e-RHINO | **SURVIVED**(`over` 是一元组,而一元组是自己的反序) | KILLED(2) |
| N5 | 远端 `graph/trace.py` 不再记录声明 | **bayesmith** | KILLED(11) | KILLED(13) |

### Wave A / `priors`(2026-08-27,6 条 5 杀,唯一幸存**必须**幸存)

详情见 `2026-08-27-wave-A-priors.md` §八。变异集要跑**两个会话**(x64 与 float32),
合在一起跑基线就是红的。

| # | 变异 | 仓 | 判决 |
|---|---|---|---|
| P1 | 不做 D24 置换 | e-RHINO | KILLED(2) |
| P2 | 置换按名字而非按 span | e-RHINO | KILLED(1,只有向量 latent 那条) |
| P3 | 合成数据 0 → 1e4 | e-RHINO | **SURVIVED,且必须如此**——一条测试正断言它够不到答案 |
| P4 | 远端丢掉方差自己那一项 | **bayesmith** | KILLED(11) |
| P5 | 去掉 D25 的构造期拒绝 | e-RHINO | KILLED(1) |
| P6 | 远端不再应用秩地板 | **bayesmith** | KILLED(1) |

### Wave A / D27 碰撞拒绝(2026-08-27,3/3 击杀)

详情见 `2026-08-27-numpyro-bridge-measurements.md` §四。

| # | 变异 | 仓 | 指名红 |
|---|---|---|---|
| Q1 | 拒绝不触发 | e-RHINO | `test_the_collision_is_refused_by_name` |
| Q2 | 只看名字,不看 sigma 是否被抽样 | e-RHINO | `test_a_fixed_sigma_beside_that_latent_is_left_alone` |
| Q3 | 只看 sigma 被抽样,不看有没有碰撞 | e-RHINO | `test_sampled_noise_std`(**既有测试**) |

### Wave A / `numpyro_bridge` 委托(2026-08-27,6/6 击杀,**第一轮 5/6**)

详情见 `2026-08-27-wave-A-numpyro-bridge.md`。同 `priors` 批,变异集要跑**两个会话**。

| # | 变异 | 仓 | 第一轮 | 修好后 |
|---|---|---|---|---|
| R1 | 忽略调用方选的节点名 | e-RHINO | KILLED(3) | KILLED(3) |
| R2 | 碰撞拒绝回到钉默认名 | e-RHINO | KILLED(2) | KILLED(2) |
| R3 | 丢掉声明的 scale,占位 sigma 变成真的 | e-RHINO | KILLED(4) | KILLED(4) |
| R4 | scale latent 的名字不再被保留 | e-RHINO | KILLED(1) | KILLED(1) |
| R5 | `observed=None` 不再翻译(`{}` → `None`) | e-RHINO | **SURVIVED** | KILLED(1) |
| R6 | 远端不再 honour 观测节点的 mask | **bayesmith** | KILLED(1) | KILLED(1) |

**R5 是一个真洞**:两个包对那个参数的读法**方向相反**,而图是拿一张零占位建的,
所以未翻译的版本会把每一次先验预测调用**条件在占位上**,交回一堆形状与 dtype 都对的
零。文件里没有任何东西分得出来。补了两条(每个方向一条),读的是 `is_observed`
而不是值本身。

### Wave A / `uncertainty` 后半(2026-08-27,7/7 击杀,**两条跨仓**)

详情见 `2026-08-27-wave-A-uncertainty-covariance.md` §七。**U6 与 U7 不是普通的变异,
它们是本批两条判据各自的检验**:U6 问「一个门面在一侧的 cross-check 还能不能失败」
(能——所以文件不退役),U7 问「合成的噪声与数据真的够不到答案吗」(够不到——所以
D31 的合法性是量出来的)。

| # | 变异 | 仓 | 判决 | 指名红 |
|---|---|---|---|---|
| U1 | 远端的 `ValueError` 不再翻译成本包的类 | e-RHINO | KILLED(5) | `test_the_refusal_wears_this_packages_class_and_keeps_the_original` 等五条 |
| U2 | `_REMOTE_KIND` 忘掉 `posterior_covariance` 也是协方差 | e-RHINO | KILLED(1) | `test_a_posterior_covariance_is_refused_by_the_same_rule` |
| U3 | 过缝矩阵用协方差自己的布局而非 params 的 | e-RHINO | KILLED(1) | `test_the_matching_covariance_propagates`(**既有测试**,按铁律 2「指认既有等价物」记账) |
| U4 | `propagate_covariance` 不再拒绝精度 | e-RHINO | KILLED(1) | `test_a_precision_is_refused_rather_than_propagated` |
| U5 | 远端在**加 jitter 之前**量条件数 | **bayesmith** | KILLED(1) | `test_jitter_is_measured_after_it_is_applied` |
| U6 | 信息图忽略调用方的噪声模型 | e-RHINO | KILLED(4) | **bayesmith** `test_noise_logdet.py` 的常数 σ 一条 + 三个 f 的 radiometer |
| U7 | 远端的 delta 方法按精度加权 | **bayesmith** | KILLED(6) | 含 `test_the_synthetic_sigma_and_data_do_not_move_the_report` |

### P2 余项 / **G2 `fit`**(2026-08-27,6/6 击杀,**本仓内**)

详情见 `2026-08-27-p2-g2-fit.md` §八。**这一组没有跨仓接缝**——rheplicant 尚未接线
——所以变异打在 bayesmith 自己的守卫上。六条的「指名红」**先登记后实跑**,六条全中。

| # | 变异(bayesmith) | 判决 | 指名红 |
|---|---|---|---|
| V1 | `fit` 丢掉 `Σ log σ` | KILLED(3) | `test_the_fit_lands_on_the_grid_minimum` |
| V2 | `names=` 被忽略 | KILLED(4) | `test_the_held_latents_come_back_untouched` |
| V3 | `Fit.objective` 返回 `history[-1]` | KILLED(1) | `test_the_reported_objective_is_at_the_reported_point` |
| V4 | `step_sizes` 被忽略 | KILLED(2) | `test_per_latent_rates_reach_what_a_single_rate_does_not` |
| V5 | 方向守卫只留声明的一半 | KILLED(1) | `test_an_undeclared_maximiser_is_caught_by_measurement` |
| V6 | 非有限结果不再拒绝 | KILLED(1) | `test_plain_gradient_above_its_stability_limit_is_refused_not_returned` |

### P2 余项 / **G10 + G12**(2026-08-27,9 条,**第一轮 8 杀**,本仓内)

详情见 `2026-08-27-p2-g10-g12.md` §八。**W8 是真洞**,形状是「fixture 分不出对与
貌似对」;**W5 的红不是登记的那条**,而登记错的是期望值不是守卫。

| # | 变异(bayesmith) | 第一轮 | 修好后 |
|---|---|---|---|
| W1 | 不完整覆盖补一个 NUTS 块 | KILLED(1) | |
| W2 | 声明块的 `reason` 换成派生块的 | KILLED(1) | |
| W3 | 钩子拿到 sweep 之前的值 | KILLED(1) | |
| W4 | NUTS 余项下不再拒绝 `on_sweep` | KILLED(1) | |
| W5 | 估计条件在 sweep 开始时的值上 | KILLED(1,**非登记的那条**) | |
| W6 | 估计**抽**而不是解 | KILLED(4) | |
| W7 | 余项块从不被 `fit` 步进 | KILLED(3) | |
| W8 | 重建的精度丢掉块自身的值 | **SURVIVED** | KILLED(1) |
| W9 | `gcr+mh` 落到通用拒绝 | KILLED(1) | |

### P2 余项 / **G14**(2026-08-27,6 条,**5 杀 1 必存**,本仓内)

详情见 `2026-08-27-p2-g14-condition-estimate.md` §五。**X1 必须幸存**(幂迭代收敛到
最大**模**,取负不改变模——实测两个方向的 spread 逐位都是 8);**X4 是真洞,而第一次
修补没杀掉它**(修补写在了另一个函数上)。

| # | 变异(bayesmith) | 第一轮 | 修好后 |
|---|---|---|---|
| X1 | 移位反向 | SURVIVED | **SURVIVED(必然)** |
| X2 | λ_min 直接返回 spread | KILLED(5) | |
| X3 | 第二次迭代复用第一个 key | KILLED(1) | |
| X4 | 去掉 λ_min 的先验地板 | **SURVIVED** | KILLED(1) |
| X5 | `condition_estimate` 改返回那个界 | KILLED(1) | |
| X6 | `condition_bound` 开始读测得路线 | KILLED(2) | |

### P2 余项 / **G9 全量**(2026-08-27,6/6 击杀,外加一条忠实回退的复查)

详情见 `2026-08-27-p2-g9-full.md` §四。**Y4 写得不忠实**(把旧写法**加**在新写法
前面,先验被加了两次),所以补跑 **Y4b**:把新写法整段换回 0.4.0 那三行。

| # | 变异(bayesmith) | 判决 | 指名红 |
|---|---|---|---|
| Y1 | 复 latent 只给 n 行 | KILLED(7) | 布局 + 四条数值 |
| Y2 | 两个半段读反 | KILLED(2) | 设计矩阵逐元素 |
| Y3 | 实成员也翻倍 | KILLED(27) | 混合块 + 既有 `test_fisher.py` 一大片 |
| Y4 | 先验曲率再加一遍(**不忠实**) | KILLED(9) | 见 Y4b |
| **Y4b** | 先验曲率**忠实回退**到 0.4.0 写法 | KILLED(4) | **四条全是本批新写的;既有 `test_fisher.py` 一条都没红** |
| Y5 | `log_weight` 只求和第一个叶子 | KILLED(2) | 虚部有贡献 |
| Y6 | `log_weight` 回到 DOMAIN 写法 | KILLED(2) | 同上 |

### **G4**(2026-08-27,6/6 击杀;R2 追出一条自比测试)

详情见 `2026-08-27-g4-reduced-basis.md` §四、§五。**R2 第一轮的五条红里没有登记
的那条**,追下去发现那条测试的两边是**同一个表达式**(`orthonormalise` 的定义就是
`transform @ candidates`)。修好后 R2 被指名的那条杀掉。

| # | 变异(bayesmith) | 判决 |
|---|---|---|
| R1 | 去掉重正交化那一遍 | KILLED(1) |
| R2 | 变换不再随向量累加 | KILLED(5);**第一轮不含登记的那条** |
| R3 | 秩切在 `eps` 而非 `sqrt(eps)` | KILLED(1) |
| R4 | greedy 改成按范数排序 | KILLED(1) |
| R5 | 秩亏丢弃从不触发 | KILLED(4) |
| R6 | 去掉 float32 拒绝 | KILLED(6) |

## 附录 B — 拒绝文案清单

> **P1 交付物,已回填(2026-08-27;2026-08-28 重测两次)。** 实测:
> `tests/inference/` 下共 **254** 个 `pytest.raises(..., match=...)` 站点。
> 按抛出的异常类:`ParameterSpaceError` **182**、`StateValidationError` **64**、
> `RuntimeError` **3**、`Exception` **3**、`TypeError` **2**。
> (252 → 254 随 **Wave B 的变异集**:它发现**没有任何测试**断言
> `condition_bound`/`condition_estimate` 的 `_require_prior_std`——删掉那一行,
> 整个定向套件仍然全绿。不是静默错答案(`prior_std=None` 之后仍会在
> `jnp.asarray(None) ** 2` 处炸),**没被守住的是消息**,而消息正是这条拒绝
> 存在而不是交给数组层的**唯一**理由。两个出口各补一条,并各自钉住消息
> **点自己的名**——三个出口共用这句话,只有 caller 参数分得开。)
> (253 → 252 随 **Wave B 的 `linear` 切换**:`test_linear_blocks.py` 的
> float32 地板那条不再用 `pytest.raises` 钉**一个 key** 的拒绝,改为扫二十个。
> 那条拒绝**一直是 key 相关的**——切换前 20 个 key 里 15 个被拒,切换后 12 个
> ——而单 key 的 pin 把一个随机结果写成了性质。它仍然断言拒绝,而且更强:
> 地板本身(κ·eps = 3.81 > 1e-3,与 key 无关)加上**每一条**拒绝的**种类**。
> 少掉的是 `pytest.raises` 站点,不是断言。类没有丢,另外三个文件还钉着
> `RuntimeError`。)
> (252 → 253 随 **D48**,G15 的解除:新增的一条钉的是**顺序**——先验准入跑在
> 图存在之前。它的类**没有**变,而那正是 D48 的全部内容:交给远端的话,这条
> 拒绝不是换一个类到达,是**根本不到达**,因为 `translate` 把 `NotGaussian`
> 归为无责裁定。这是本清单第一次记录一条**因为缝会吞掉它而留守**的拒绝。)
> (250 → 252 随 **D23** 的判别 fixture:曲率拒绝一条,加一条钉住
> 「一个由别的 `Latent` 参数化的先验不是本层能表达的声明」的 `TypeError`。)
> (244 → 250 随 `uncertainty` 后半:D29 的天花板 + D30 的类 + D31 的精度拒绝,
> 全部由守卫报数。)
>
> **这一行在 2026-08-27 一整天里是错的,而它今天又变对了,这比一直错更值得写下来。**
> G1 接线退役了一条拒绝(241 → 240)并改了 `CENSUS` 的 pin,**没有改这里**;D27
> 的碰撞拒绝今天又加回一条(240 → 241),于是这个数**碰巧**重新等于纸面上的它。
> 中间那段时间,这一行与守卫的 pin 不一致,而没有任何东西会说。同一形状在
> bayesmith 的工作笔记里已经写过一次:「那个数碰巧是对的,而这正是坏结果,因为
> 那次运行里没有任何东西能说出相反的话。」**逐文件清单自 2026-08-27 起由
> `test_refusal_census._sites()` 重生成;这个总数也应当照做,而不是手抄。**
>
> **这张表不由人手维护,也不该由人手维护。** 它由
> e-RHINO `tests/inference/test_refusal_census.py` 逐文件计数并**钉住**,
> 所以新增或删除一个被钉的拒绝会先让那个守卫红,并**报出要写进这里的数字**。
> 本附录与那个守卫是**同一次测量的两份呈现**,必须同批更新——计划反复付学费
> 的形状正是「一个事实两份拼写,其中一份悄悄过期」。
>
> **「今日抛出处」与「切换后抛出侧」为什么不是两列。** 定稿时设想的是三列表,
> 而实测后改了形态,理由写下来以免被读成偷工:227(当时)个站点里,今天
> **每一条都由 rheplicant 抛出**——这是铁律 1 的直接推论,门面保持异常类身份
> 与被钉文案。所以「今日抛出处」整列是同一个值,而「切换后抛出侧」在切换**之前**
> 填只能是预测。二者都不是测量。
>
> 因此本附录记录的是**总体与逐文件的分布**(可测、会漂、被守卫钉住),而
> 「哪一条切换后由 bayesmith 抛出并经 `translate` 回来」**逐波核销**:每一波
> 切换的模块,其文件在下表的计数会变,守卫会红,那一波在自己的执行页里写明
> 它接手了哪些句子、哪些改由 `translate` 产生。这正是计划 §六「R4 拒绝文案:
> 附录 B 清单驱动,逐波核销」说的动作,只是把「清单」定义成了一个**能失败的
> 守卫**而不是一张静态表。
>
> **今日已知的两条例外**,也就是清单里唯一已经跨过缝的部分:
> `AffinityRefused` → `LinearityRefused`(载荷同数,不重算)与
> 其余 `BayesmithError` → `SeamRefusal`(`ParameterSpaceError` 子类,点名 site)。
> 两条都在 e-RHINO `tests/inference/test_graph_bridge.py` 里被钉住,并各有一个
> 已实跑的接缝变异(附录 A 第 3、4 行)。

### 逐文件清单(2026-08-27 实测)

<details><summary><code>test_block_learning_rate.py</code> — 2 条</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 62 | `Exception` | `learning_rate must be > 0` |
| 69 | `Exception` | `conjugate` |

</details>

<details><summary><code>test_d23_refusal_criterion.py</code> — 2 条(2026-08-28 由 <code>_sites()</code> 生成)</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 161 | `ParameterSpaceError` | `curvature` |
| 217 | `TypeError` | `unsupported operand` |

</details>

<details><summary><code>test_declared_prior.py</code> — 14 条</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 222 | `ParameterSpaceError` | `prior_std` |
| 284 | `ParameterSpaceError` | `prior_std` |
| 296 | `ParameterSpaceError` | `conjugate` |
| 396 | `ParameterSpaceError` | `amp_b` |
| 409 | `ParameterSpaceError` | `prior_std` |
| 416 | `ParameterSpaceError` | `prior_mean` |
| 423 | `ParameterSpaceError` | `prior_std` |
| 451 | `ParameterSpaceError` | `prior_std= you passed is a traced value` |
| 544 | `ParameterSpaceError` | `traced` |
| 558 | `ParameterSpaceError` | `prior_std` |
| 592 | `ParameterSpaceError` | `conjugate` |
| 613 | `ParameterSpaceError` | `LogNormal` |
| 635 | `ParameterSpaceError` | `needs prior_std` |
| 642 | `ParameterSpaceError` | `different problem` |

</details>

<details><summary><code>test_fisher_prior.py</code> — 7 条</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 222 | `StateValidationError` | `parameter_covariance` |
| 233 | `StateValidationError` | `parameter_covariance` |
| 273 | `ParameterSpaceError` | `z_scalar` |
| 295 | `ParameterSpaceError` | `LogNormal` |
| 314 | `ParameterSpaceError` | `z_scalar` |
| 321 | `StateValidationError` | `not named` |
| 331 | `ParameterSpaceError` | `do not match` |

</details>

<details><summary><code>test_forward.py</code> — 1 条</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 53 | `ParameterSpaceError` | `NoiseOperator at 'noise'` |

</details>

<details><summary><code>test_gls.py</code> — 3 条</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 217 | `ParameterSpaceError` | `prior_std` |
| 223 | `ParameterSpaceError` | `min_reweights` |
| 438 | `ParameterSpaceError` | `needs a NoiseModel` |

</details>

<details><summary><code>test_graph_bridge.py</code> — 16 条(2026-08-27 由 `_sites()` 重生成)</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 162 | `ParameterSpaceError` | `building the block` |
| 339 | `StateValidationError` | `more than one legitimate reading` |
| 349 | `StateValidationError` | `complex `observed`` |
| 361 | `ParameterSpaceError` | `HomoscedasticNoise` |
| 385 | `ParameterSpaceError` | `<computed>` |
| 398 | `ParameterSpaceError` | `internal node names` |
| 404 | `ParameterSpaceError` | `declares no prior` |
| 410 | `ParameterSpaceError` | `also declares` |
| 423 | `ParameterSpaceError` | `which this space does not declare` |
| 458 | `ParameterSpaceError` | `ComplexNormal` |
| 495 | `ParameterSpaceError` | `declares` |
| 499 | `ParameterSpaceError` | `no prior for latent` |
| 823 | `ParameterSpaceError` | `two priors on one quantity` |
| 884 | `ParameterSpaceError` | `internal node names` |
| 982 | `ParameterSpaceError` | `scale_prior` |
| 1010 | `ParameterSpaceError` | `internal node names` |

</details>

<details><summary><code>test_identifiability.py</code> — 13 条</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 366 | `StateValidationError` | `null direction` |
| 368 | `StateValidationError` | `null direction` |
| 375 | `StateValidationError` | `null direction` |
| 449 | `StateValidationError` | `Inconsistent report` |
| 451 | `StateValidationError` | `Inconsistent report` |
| 458 | `StateValidationError` | `Inconsistent report` |
| 560 | `ParameterSpaceError` | `not a latent` |
| 918 | `StateValidationError` | `float32\|single precision` |
| 962 | `ParameterSpaceError` | `not a latent` |
| 969 | `ParameterSpaceError` | `more than once\|repeated` |
| 977 | `ParameterSpaceError` | `at least one` |
| 1004 | `ParameterSpaceError` | `R-linear but not C-linear` |
| 1024 | `ParameterSpaceError` | `not a continuous parameter` |

</details>

<details><summary><code>test_inference_construction_guards.py</code> — 10 条</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 166 | `StateValidationError` | `learning_rate must be > 0` |
| 168 | `StateValidationError` | `n_steps must be a positive int` |
| 657 | `StateValidationError` | `learning_rate must be > 0` |
| 661 | `StateValidationError` | `floor must be >= 0` |
| 676 | `StateValidationError` | `channel_width` |
| 681 | `StateValidationError` | `integration_time` |
| 692 | `StateValidationError` | `n_steps must be a positive int` |
| 698 | `StateValidationError` | `beta1/beta2 must be in \[0, 1\)` |
| 704 | `StateValidationError` | `must be in \[0, 1\)` |
| 722 | `StateValidationError` | `n_components must be positive` |

</details>

<details><summary><code>test_inference_unpinned_refusals.py</code> — 5 条</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 88 | `ParameterSpaceError` | `'amp' is not declared linear=True` |
| 94 | `ParameterSpaceError` | `Declare it` |
| 114 | `ParameterSpaceError` | `No latent in this space is declared` |
| 144 | `ParameterSpaceError` | `wiener_solve expects a real-valued` |
| 206 | `StateValidationError` | `was computed for \{'amp'` |

</details>

<details><summary><code>test_jeffreys_prior.py</code> — 13 条</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 445 | `ParameterSpaceError` | `over no latents` |
| 450 | `ParameterSpaceError` | `more than once` |
| 455 | `ParameterSpaceError` | `takes latent NAMES` |
| 460 | `ParameterSpaceError` | `positive relative cut` |
| 466 | `ParameterSpaceError` | `names \['fg_index'\]` |
| 472 | `ParameterSpaceError` | `AND declare their own` |
| 521 | `ParameterSpaceError` | `no entry for \['fg_beta'\]` |
| 531 | `ParameterSpaceError` | `inside its own definition` |
| 545 | `ParameterSpaceError` | `splits it across blocks` |
| 562 | `ParameterSpaceError` | `does not evaluate a joint prior` |
| 606 | `ParameterSpaceError` | `\['t_floor'\] have no prior` |
| 672 | `ParameterSpaceError` | `sqrt.det I. is not a density` |
| 687 | `ParameterSpaceError` | `sigma\^-2` |

</details>

<details><summary><code>test_linear_block_as_dict.py</code> — 2 条</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 92 | `ParameterSpaceError` | `groups \['gain'\]` |
| 97 | `ParameterSpaceError` | `keyed by \['sky'\]` |

</details>

<details><summary><code>test_linear_blocks.py</code> — 20 条(2026-08-28 由 <code>_sites()</code> 重新生成)</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 105 | `ParameterSpaceError` | `linear=True` |
| 113 | `ParameterSpaceError` | `not affine` |
| 320 | `ParameterSpaceError` | `different` |
| 326 | `ParameterSpaceError` | `prior_std` |
| 346 | `ParameterSpaceError` | `not affine` |
| 455 | `ParameterSpaceError` | `not affine` |
| 488 | `ParameterSpaceError` | `not affine` |
| 503 | `ParameterSpaceError` | `No latent named` |
| 517 | `ParameterSpaceError` | `which latent` |
| 545 | `ParameterSpaceError` | `not affine` |
| 764 | `ParameterSpaceError` | `prior_std` |
| 838 | `ParameterSpaceError` | `not a latent` |
| 1023 | `RuntimeError` | `condition bound` |
| 1041 | `RuntimeError` | `condition bound` |
| 1233 | `RuntimeError` | `precision|condition number` |
| 1303 | `ParameterSpaceError` | `needs prior_std` |
| 1307 | `ParameterSpaceError` | `<computed>` |
| 381 | `ParameterSpaceError` | `not affine` |
| 399 | `ParameterSpaceError` | `not affine` |
| 478 | `ParameterSpaceError` | `not affine` |

</details>

<details><summary><code>test_linear_groups.py</code> — 21 条</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 274 | `ParameterSpaceError` | `name= OR names=` |
| 278 | `ParameterSpaceError` | `at least one latent name` |
| 282 | `ParameterSpaceError` | `not a latent of this space` |
| 286 | `ParameterSpaceError` | `more than once` |
| 303 | `ParameterSpaceError` | `not declared linear=True` |
| 314 | `ParameterSpaceError` | `not a latent` |
| 619 | `ParameterSpaceError` | `different` |
| 631 | `ParameterSpaceError` | `one prior PER LATENT` |
| 635 | `ParameterSpaceError` | `one prior PER LATENT` |
| 639 | `ParameterSpaceError` | `one prior PER LATENT` |
| 643 | `ParameterSpaceError` | `does not group` |
| 648 | `ParameterSpaceError` | `prior_std for \['t_ant'\]` |
| 668 | `ParameterSpaceError` | `latent 't_nw' declares a Uniform` |
| 682 | `ParameterSpaceError` | `latent 't_ant' declares` |
| 715 | `ParameterSpaceError` | `one entry per member` |
| 729 | `ParameterSpaceError` | `one entry per member` |
| 753 | `ParameterSpaceError` | `not affine in them JOINTLY` |
| 759 | `ParameterSpaceError` | `not affine` |
| 789 | `ParameterSpaceError` | `floating-point or complex` |
| 799 | `ParameterSpaceError` | `floating-point or complex` |
| 803 | `ParameterSpaceError` | `name= OR names=` |

</details>

<details><summary><code>test_loss_sense.py</code> — 5 条</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 121 | `ParameterSpaceError` | `declares sense='maximize'` |
| 138 | `ParameterSpaceError` | `must be 'maximize'` |
| 151 | `ParameterSpaceError` | `scores a PERFECT prediction` |
| 170 | `ParameterSpaceError` | `scores a PERFECT prediction` |
| 185 | `ParameterSpaceError` | `not finite at entry` |

</details>

<details><summary><code>test_noise_model.py</code> — 3 条</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 104 | `StateValidationError` | `positive` |
| 106 | `StateValidationError` | `positive` |
| 163 | `StateValidationError` | `shape` |

</details>

<details><summary><code>test_noise_std_axis.py</code> — 18 条</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 107 | `StateValidationError` | `more than one` |
| 145 | `StateValidationError` | `more than one` |
| 156 | `StateValidationError` | `more than one` |
| 163 | `StateValidationError` | `more than one` |
| 177 | `StateValidationError` | `more than one` |
| 195 | `StateValidationError` | `more than one` |
| 211 | `StateValidationError` | `more than one` |
| 279 | `StateValidationError` | `more than one` |
| 286 | `StateValidationError` | `wiener_solve` |
| 378 | `StateValidationError` | `more than one` |
| 382 | `StateValidationError` | `<computed>` |
| 491 | `ParameterSpaceError` | `takes a plain sigma array` |
| 509 | `ParameterSpaceError` | `no prediction to evaluate it` |
| 516 | `ParameterSpaceError` | `wiener_solve` |
| 518 | `ParameterSpaceError` | `gcr_sample` |
| 554 | `StateValidationError` | `condition_estimate` |
| 558 | `ParameterSpaceError` | `takes a plain sigma array` |
| 615 | `StateValidationError` | `<computed>` |

</details>

<details><summary><code>test_npe.py</code> — 4 条</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 236 | `ParameterSpaceError` | `no prior` |
| 243 | `StateValidationError` | `positive` |
| 253 | `StateValidationError` | `same pairs` |
| 258 | `StateValidationError` | `n_params` |

</details>

<details><summary><code>test_numpyro_bridge.py</code> — 6 条(2026-08-27 由 `_sites()` 重生成)</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 149 | `ParameterSpaceError` | `no prior` |
| 157 | `ParameterSpaceError` | `shape` |
| 329 | `StateValidationError` | `per-sample shape` |
| 344 | `StateValidationError` | `differing numbers of draws` |
| 349 | `StateValidationError` | `missing site` |
| 382 | `ParameterSpaceError` | `noise_std` |

</details>

<details><summary><code>test_parameters.py</code> — 29 条</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 180 | `ParameterSpaceError` | `unique` |
| 187 | `ParameterSpaceError` | `undeclared` |
| 195 | `ParameterSpaceError` | `never bound` |
| 205 | `ParameterSpaceError` | ``into` selector` |
| 219 | `ParameterSpaceError` | `returned 3 values` |
| 223 | `ParameterSpaceError` | `exactly one latent` |
| 228 | `ParameterSpaceError` | `shape` |
| 235 | `ParameterSpaceError` | `scope` |
| 316 | `ParameterSpaceError` | `fan='broadcast'` |
| 333 | `ParameterSpaceError` | `fan='distribute'` |
| 347 | `ParameterSpaceError` | `returned 3 values` |
| 355 | `ParameterSpaceError` | `returned 3 values` |
| 395 | `ParameterSpaceError` | `fan='broadcast'` |
| 410 | `ParameterSpaceError` | `fan='tie'` |
| 436 | `ParameterSpaceError` | `fan='broadcast'` |
| 456 | `ParameterSpaceError` | `array leaf` |
| 467 | `ParameterSpaceError` | `written by more than one` |
| 475 | `ParameterSpaceError` | `shape` |
| 484 | `ParameterSpaceError` | `complex` |
| 494 | `ParameterSpaceError` | `structure` |
| 533 | `ParameterSpaceError` | `shape` |
| 607 | `ParameterSpaceError` | `INSTEAD of bindings` |
| 627 | `ParameterSpaceError` | `does not reach the pipeline` |
| 640 | `ParameterSpaceError` | `shape` |
| 650 | `ParameterSpaceError` | `complex` |
| 663 | `Exception` | `bindings` |
| 780 | `ParameterSpaceError` | `'x'` |
| 794 | `ParameterSpaceError` | `'x'` |
| 812 | `ParameterSpaceError` | `'x'` |

</details>

<details><summary><code>test_plan.py</code> — 32 条</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 271 | `ParameterSpaceError` | `at least one latent name` |
| 277 | `ParameterSpaceError` | `latent NAMES` |
| 281 | `ParameterSpaceError` | `more than once` |
| 285 | `ParameterSpaceError` | `the engines are` |
| 293 | `ParameterSpaceError` | `positive int` |
| 302 | `ParameterSpaceError` | `at least one Block` |
| 306 | `ParameterSpaceError` | `does not declare` |
| 310 | `ParameterSpaceError` | `\['gain', 't_coeff'\]` |
| 316 | `ParameterSpaceError` | `'gain' is in more than one block` |
| 322 | `ParameterSpaceError` | `does not cover latent\(s\) \['t_coeff'\]` |
| 359 | `ParameterSpaceError` | `mixes declared-linear` |
| 363 | `ParameterSpaceError` | `\['amp'\].*\['centre'\]` |
| 381 | `ParameterSpaceError` | `not declared linear=True` |
| 387 | `ParameterSpaceError` | `no inner ` |
| 427 | `ParameterSpaceError` | `nullity 6` |
| 535 | `ParameterSpaceError` | `check_identifiability` |
| 539 | `ParameterSpaceError` | `check_identifiability` |
| 620 | `ParameterSpaceError` | `nullity 12` |
| 736 | `ParameterSpaceError` | `min_sweeps <= max_iter` |
| 745 | `ParameterSpaceError` | `max_iter >= 1` |
| 761 | `ParameterSpaceError` | `n_sweeps >= 1` |
| 771 | `ParameterSpaceError` | `warmup >= 0` |
| 783 | `ParameterSpaceError` | `<computed>` |
| 807 | `ParameterSpaceError` | `\['centre'\] have none` |
| 843 | `ParameterSpaceError` | `this plan's model predicts` |
| 848 | `ParameterSpaceError` | `this plan's model predicts` |
| 871 | `ParameterSpaceError` | `this plan's model predicts` |
| 876 | `ParameterSpaceError` | `this plan's model predicts` |
| 889 | `ParameterSpaceError` | `not affine in them JOINTLY` |
| 894 | `ParameterSpaceError` | `not affine in them JOINTLY` |
| 906 | `TypeError` | `key` |
| 1071 | `ParameterSpaceError` | `<computed>` |

</details>

<details><summary><code>test_prior_sensitivity.py</code> — 9 条</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 818 | `StateValidationError` | `fg_gamma` |
| 820 | `StateValidationError` | `fg_gamma` |
| 826 | `StateValidationError` | `positive` |
| 828 | `StateValidationError` | `positive` |
| 832 | `StateValidationError` | `broadcast` |
| 963 | `ParameterSpaceError` | `fg_beta` |
| 999 | `ParameterSpaceError` | `broadcast\|shape` |
| 1006 | `ParameterSpaceError` | `fg_gamma` |
| 1011 | `ParameterSpaceError` | `fg_gamma` |

</details>

<details><summary><code>test_stochastic_twin.py</code> — 4 条</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 123 | `ParameterSpaceError` | `NoiseOperator at 'noise'` |
| 127 | `ParameterSpaceError` | `NoiseOperator at 'noise'` |
| 131 | `ParameterSpaceError` | `Assembly\.without\(node_id\)` |
| 142 | `ParameterSpaceError` | `RFIOperator at 'field_sum/rfi_field'` |

</details>

<details><summary><code>test_uncertainty.py</code> — 14 条(2026-08-27 由 `_sites()` 重生成)</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 51 | `StateValidationError` | `flags` |
| 123 | `StateValidationError` | `no trainable` |
| 154 | `StateValidationError` | `param_cov` |
| 167 | `StateValidationError` | `structure` |
| 181 | `StateValidationError` | `not a covariance` |
| 364 | `StateValidationError` | `parameter_covariance` |
| 370 | `StateValidationError` | `no parameter named` |
| 378 | `StateValidationError` | `not named` |
| 392 | `StateValidationError` | `Complex parameters` |
| 429 | `StateValidationError` | `condition number` |
| 450 | `StateValidationError` | `condition number` |
| 465 | `StateValidationError` | `enable_x64` |
| 517 | `StateValidationError` | `covariance` |
| 545 | `StateValidationError` | `covariance` |

</details>


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
  `tests/inference/test_loglinear.py`};**排除**根目录**九**份未跟踪评审/
  交接草稿(另行裁决去留,不随本程序入库)。
  - **九份,不是八份**(实测 `git status --porcelain | grep -c '^??'`):
    `CODE_REVIEW_REPORT.md`、`HANDOVER.md`、`NEXT_SESSION_PROMPT.md`、
    `PROPOSAL_BAYESMITH.md`、`PROPOSAL_MERGED.md`、`PROPOSAL_RHEPLICANT.md`、
    `PROPOSAL_bayesmith_bayesian.md`、`PROPOSAL_rheplicant_nonbayesian.md`、
    `REVIEW_REPORT.md`。与上一条一样是「没人核过的计数」——本附录两个数字
    都错了同一种错,所以下一波开工时,这里的每个数字都当待测量处理。
