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
  证据链:Wave A 的执行页。

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

## 附录 B — 拒绝文案清单

> **P1 交付物,已回填(2026-08-27)。** 实测:`tests/inference/` 下共
> **241** 个 `pytest.raises(..., match=...)` 站点。按抛出的异常类:
> `ParameterSpaceError` **175**、`StateValidationError` **58**、
> `RuntimeError` **4**、`Exception` **3**、`TypeError` **1**。
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

<details><summary><code>test_graph_bridge.py</code> — 14 条</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 161 | `ParameterSpaceError` | `building the block` |
| 338 | `StateValidationError` | `more than one legitimate reading` |
| 348 | `StateValidationError` | `complex `observed`` |
| 358 | `ParameterSpaceError` | `sigma = inf` |
| 370 | `ParameterSpaceError` | `HomoscedasticNoise` |
| 394 | `ParameterSpaceError` | `<computed>` |
| 407 | `ParameterSpaceError` | `joint_prior` |
| 420 | `ParameterSpaceError` | `internal node names` |
| 426 | `ParameterSpaceError` | `declares no prior` |
| 432 | `ParameterSpaceError` | `also declares` |
| 445 | `ParameterSpaceError` | `which this space does not declare` |
| 480 | `ParameterSpaceError` | `ComplexNormal` |
| 517 | `ParameterSpaceError` | `declares` |
| 521 | `ParameterSpaceError` | `no prior for latent` |

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

<details><summary><code>test_linear_blocks.py</code> — 19 条</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 104 | `ParameterSpaceError` | `linear=True` |
| 112 | `ParameterSpaceError` | `not affine` |
| 319 | `ParameterSpaceError` | `different` |
| 325 | `ParameterSpaceError` | `prior_std` |
| 345 | `ParameterSpaceError` | `not affine` |
| 380 | `ParameterSpaceError` | `not affine` |
| 398 | `ParameterSpaceError` | `not affine` |
| 454 | `ParameterSpaceError` | `not affine` |
| 477 | `ParameterSpaceError` | `not affine` |
| 487 | `ParameterSpaceError` | `not affine` |
| 502 | `ParameterSpaceError` | `No latent named` |
| 516 | `ParameterSpaceError` | `which latent` |
| 544 | `ParameterSpaceError` | `not affine` |
| 763 | `ParameterSpaceError` | `prior_std` |
| 837 | `ParameterSpaceError` | `not a latent` |
| 1018 | `RuntimeError` | `condition number` |
| 1036 | `RuntimeError` | `condition number` |
| 1063 | `RuntimeError` | `precision\|condition number` |
| 1190 | `RuntimeError` | `precision\|condition number` |

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

<details><summary><code>test_numpyro_bridge.py</code> — 5 条</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 62 | `ParameterSpaceError` | `no prior` |
| 70 | `ParameterSpaceError` | `shape` |
| 242 | `StateValidationError` | `per-sample shape` |
| 257 | `StateValidationError` | `differing numbers of draws` |
| 262 | `StateValidationError` | `missing site` |

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

<details><summary><code>test_uncertainty.py</code> — 8 条</summary>

| 行 | 类 | `match=` |
|---|---|---|
| 49 | `StateValidationError` | `flags` |
| 121 | `StateValidationError` | `no trainable` |
| 152 | `StateValidationError` | `param_cov` |
| 165 | `StateValidationError` | `structure` |
| 262 | `StateValidationError` | `parameter_covariance` |
| 268 | `StateValidationError` | `no parameter named` |
| 276 | `StateValidationError` | `not named` |
| 290 | `StateValidationError` | `Complex parameters` |

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
