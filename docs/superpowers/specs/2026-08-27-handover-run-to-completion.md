# 交接 prompt — 把「全面移交」程序跑到完全结束

> 与 kickoff 同目录、同被跟踪。使用方式:把 `---` 以下整段粘贴给新 session。
> **日期**:2026-08-28(第十九次改写)· 交接自 **G5、G6 本体、D23、多数据集联合后验
> 的数学推导、架构叙事审计、`evidence/`→`marginal/` 改名、D46 复数拒绝** 之后的会话。
> **0.5.0 已发布并核实上索引;历史重写已完成并强推;两仓已同步。**
> 本页 §四 是**剩余工作的完整盘点**,为 compact 之后接着做而写。

---

你要把「全面移交」程序**跑到完全结束**:把 rheplicant 的全部贝叶斯数值实现
移交给 bayesmith,`rheplicant.inference` 收缩为门面。计划已定稿,不要重新
评审、不要重新设计。

## 〇、owner 的授权(最重要,先读)

**owner 已授权:不要停下来问,一直做到程序结束。** 这条改变了裁决纪律的形态,
所以要写清楚它**具体**是什么意思:

- 计划 §二 登记簿里**仍未拍板**的 D 项,owner **委托给计划自己的建议**。对每一项:
  采纳该行 `*建议*` 所写的选项,并把裁决回填进那一行,格式为
  `**【owner 已授权委托,YYYY-MM-DD:取 (x),依 2026-08-27 的一次性委托】**`,
  后接一句你为什么读成这个选项。
- **委托不是空白支票。** 如果照建议执行会与**已实测的事实**冲突,**不要**照做:
  把冲突写进那一行,按事实选择,并在批次记录页里点名说明。**这已经发生过四次**
  ——D13 的版本号、D9 的整条建议、D21 的契约页那句、以及本页 §三 早先那条关于
  CLAUDE.md 的过期断言。所以这条不是理论。
- 遇到**计划完全没有预见**的新裁决点:自己选,选保守的一侧(能保住正确性、
  能被守卫钉住的那一侧),把它作为新条目写进登记簿并注明是本次委托下自定的。
  **D20、D21、D22、D23、D24、D25 都是这样来的。**
- **唯一仍要停下来的情况**:一个动作会**不可逆地对外**(发一个新 PyPI 版本号、
  强推覆盖远端、删除未跟踪的用户文件)**且**计划没有明写该动作。

## 一、第一步永远是读(按此顺序)

1. **计划本体**:`docs/superpowers/specs/2026-08-26-one-implementation.md`
   ——终局形态、七条铁律、登记簿 **D7–D45**、缺口 G1–G15、四波切换、附录 A/B/C。
2. 本程序至今的执行记录,同目录。**最近三份最要紧,按序读**:
   `2026-08-28-g5-amortize.md`、`2026-08-28-g6-consumption.md`、
   `2026-08-28-d23-two-criteria.md`。再往前:
   `2026-08-27-g6-enumeration.md`(G6 的逐项登记,§三 已结清)、
   `2026-08-27-g4-reduced-basis.md`、`2026-08-27-g3-chain.md`、
   `2026-08-27-wave-A-uncertainty-covariance.md`、
   `2026-08-27-p2-g2-fit.md`、`2026-08-27-p2-g10-g12.md`、
   `2026-08-27-p2-g14-condition-estimate.md`、
   `2026-08-27-ci-flat-chain.md`(**本机绿而远端红的唯一一次,读它的 §三 判别方法**)、
   `2026-08-27-wave-A-uncertainty-fisher.md`、
   `2026-08-27-wave-A-numpyro-bridge.md`、`2026-08-27-numpyro-bridge-measurements.md`、
   `2026-08-27-wave-A-priors.md`。其余:`2026-08-26-wave-P0.md`、
   `2026-08-26-wave-P2a.md`、`2026-08-27-d17-protocol.md`、`2026-08-27-d16-five-axes.md`、
   `2026-08-27-wave-P1.md`、`2026-08-27-wave-P2-G1.md`、`2026-08-27-wave-P2-G13.md`、
   `2026-08-27-wave-P2-G7.md`、`2026-08-27-wave-P2-D9.md`、
   `2026-08-27-wave-A-opening.md`、`2026-08-27-wave-A-identifiability.md`、
   `2026-08-27-wave-A-g1-wiring.md`、`2026-08-27-wave-A-sensitivity.md`、
   `2026-08-27-wave-A-s6-widened.md`、`2026-08-27-wave-A-g13-wiring.md`。
3. 前代 spec 的 §四(模块绑定契约台账)与 §六(本程序的由来):
   `2026-08-24-rheplicant-migration.md`。
4. 两仓工作笔记(操作规则,条条是学费):`/Users/zzhang/projects/e-RHINO/CLAUDE.md`
   (与 AGENTS.md 逐字节一致,有测试钉着)与 `/Users/zzhang/projects/bayesmith/CLAUDE.md`。
   **两份都在 2026-08-27 补了第 (0) 条的第二半**,见 §六。

## 二、当前状态(2026-08-28 实测,先复核再信)

- **bayesmith** 套件 **1544 passed / 0 skipped** exit 0(298.3 s);`ruff` 干净。
  **PyPI 上仍是 0.4.0**;**`pyproject.toml` 已是 `0.5.0`,CHANGELOG 已切好**,
  发布日只差 `git tag v0.5.0 && push` 与索引核实。
- **e-RHINO** **10120 passed / 553 skipped** exit 0(349.1 s)。README 计数
  **10693**;拒绝普查 **252**;coverage floor **89**;bayesmith 地板 `>=0.4`。
- **两仓提交未推送**。用 `git ls-remote` 数,不要读本地 `origin/main`。
  实测于 2026-08-28、**在改写本页的提交之前**:e-RHINO 领先 **6**,
  bayesmith 领先 **48**;两棵树干净。
- 执行页共 **41** 份(含本页)。

**已完成**:P0、P2a、D16、D17、P1、P2 余项全部、0.4.0 发布、Wave A 全部五模块、
G15/G4/G3、D12 前置、G6 逐项登记、**G5**、**G6 本体(7 里的 6)**、**D23**、
**多数据集联合后验的数学推导**、**架构叙事审计**、**`evidence/`→`marginal/`**、
**D46**。**登记簿 D7–D47;未裁决只剩 D39**,两个方向都已钉住,归 Wave D。

**0.5.0 的内容已经定稿并全部落地。**

## 三、下一位会先撞到的几件事

1. **一条缺陷形状,本程序至今出现八次,当成收尾检查项**:**守卫够不到它要守的那个
   条件**。G10 的 W8、G14 的 X4、X4 的第一次修补、G3 的常数类、**G5 的 Z9**(断言读的
   是**正在被优化的那个量**,所以符号翻转之后它通过得更漂亮)、**G6 的 M6**(「散兵
   campaign」只散了 **1.7 %**)、**D23 探针第一版**(检查构造函数抛不抛,而 numpyro
   从不抛)、以及 **D23 判别 fixture 的第一版**(第二列是第一列的标量倍数,于是四行
   都是另一个方向的秩亏)。收工前问:**「这条守卫真的制造出它要防的那个情形了吗?」**
2. **另一条**:**手加的总数**。能派生就不要手抄。G6 页第一版三个总数错了两个;
   本会话给 bayesmith 补了 `tests/test_readme_count.py`,因为那个 README 的计数
   **漂了 204 条、横跨六个批次**而没有任何东西会说。
3. **一个幸存的变异有三种成因,不要当成一种**。本会话实测:**真缺口**(M6)、
   **变异的是够不到的代码**(M11:`cholesky` 先抛,NaN 到不了那一行)、**等价变异**
   (M12:`centred` 均值为零,两个写法是同一个斜率)。给后两种编 fixture 是更贵的错误。
4. **退出码,又值了两次。** 本会话两次:harness 的完成通知说「exited with code 0」,
   而 `run.exit` 说 `PYTEST_EXIT=1`,两次都真有一条红的。**把退出码写进自己的文件再读。**
5. **两个模块只切了一半**(`numpyro_bridge`、`uncertainty`),都不在 `SWITCHED` 里,
   cross-check 都保留。
6. **D39** 已登记未裁决,但两个方向都已钉住,归 Wave D。**D23 已结清**(§四)。

## 四、剩余工作(2026-08-28 全量盘点)

### 〇、本次会话结束时的状态,先复核再信

- **bayesmith `0.5.0` 已在 PyPI**,`/simple/` 有 wheel 与 sdist;**干净 venv 装
  `bayesmith>=0.5` 实测**:版本 0.5.0、`bayesmith.marginal` 可用、
  `bayesmith.evidence` 可用且发 `DeprecationWarning`、深路径 `from
  bayesmith.evidence.compress import compress` 可用、D46 拒绝在已发布的 wheel 里。
  (装的时候用 `--no-cache-dir`/`--refresh`:pip 的缓存在发布日会说 0.4.0 是最新,
  这是 CLAUDE.md 记的那条。)
- **两仓已推送并核实**:`git ls-remote` 双仓 `ahead=0`。
- **e-RHINO 历史已重写**:九份草稿从**远端**消失(浅克隆实测 0 commits),
  重写后 tree 逐字节相同,九份未跟踪工作副本原样保留。
- 套件:bayesmith **1544 passed / 0 skipped**;e-RHINO **10120 passed / 553 skipped**。
- 登记簿 **D7–D47**;未裁决只剩 **D39**(两个方向已钉住,归 Wave D)。
- **`SWITCHED` 只有三个模块**:`identifiability.py`、`sensitivity.py`、`priors.py`。
  `numpyro_bridge` 与 `uncertainty` **各只切了一半**,故意不在里面,cross-check 保留。

### 一、CI —— 唯一还没确认绿的东西,**下一位第一件事**

- **bayesmith `Cross-check`:绿**(2026-08-28)。
- **bayesmith `Seam`:2026-08-27 与 08-28 两次红,原因已查明并已修**——它 checkout
  的是 `e-RHINO@main`,而修 flat-prior 断言的那个提交(`5ab926d`,「the flat-prior
  chain assertion was a property of one machine」)当时还没推。e-RHINO 推送之后已
  **手动重跑**,结果**本会话结束时尚未出来**。**去看它。**
- **e-RHINO `Tests`:2026-08-27 两次红**,本次推送触发的那次结束时仍在跑。
  **去看它**;若仍红,大概率是同一条 flat-prior(现已修)或另一条,按
  `2026-08-27-ci-flat-chain.md` §三 的判别法处理:**问这条断言钉的是大效应还是
  小差值**。
- 清理:`git tag -d backup/pre-draft-purge-2026-08-28` 与
  `git update-ref -d refs/original/refs/heads/main`(**满意之后再删**;删掉之前
  旧历史仍可从这两个 ref 找回)。

### 二、G15 的 rheplicant 一行 —— **现在解锁了,最便宜的一件**

0.5.0 载着 `local_block(..., priors=True)`,所以那条有解除条件的延期到期了:
删 `uncertainty._prior_precision`(`src/rheplicant/inference/uncertainty.py:284`),
把 `:584` 改成 `include_prior=space is not None`,**并把 e-RHINO 的地板从
`bayesmith>=0.4` 提到 `>=0.5`**(`pyproject.toml:73`)。
**那一批必须重跑 e-RHINO 全套并重新量一件事**:G9 全量修掉的先验广播缺陷今天
到不了门面(门面永远传 `include_prior=False`),这条改动**正是让它够得到的**。

### 三、Wave B —— 先决**全部满足**,是下一个大波

`linear` 求解面、`gls`、`plan`+`engines`(验证与词汇留守,执行经 G10)、
`noise`/`likelihood` 工厂化。D7/D8/D14 已拍,G1/G2/G9/G10/G12/G14 已落地,
D17 协议已跑完。**P1 例 6/10 的完整形态(梯度块 estimate、每 sweep 诊断)在本波
验收。** config 侧钉内部件的三个测试(`MIN_DRAWS` 等常量、`SamplingPlan` 源文本
pin)本波**显式重谈**,走登记簿。

### 四、Wave C

`calibrate`、`npe`、`reduced_basis`。另有:
1. **`compress_reduced_basis`**(**D44**)与 G4 余下三名(`score_directions`、
   `build_reduced_basis`、`basis_fidelity`)同批。
2. **`npe` 接线按 D42**:三名委托、`simulate_pairs` 留守。**那一批要量的一件**:
   rheplicant 的 `NeuralPosterior` **类身份**有没有被测试钉住——若钉住,门面要包装
   而不是重导出(D12 在证据容器上的同一形状)。
3. **D33 的分诊**:`fit` 拒绝发散的下降,而 rheplicant 的 calibrator 今天**交回
   NaN**;有没有测试依赖它,**今天没有量过**,是那一批的第一件事。
4. **`min_scale` 的两层**:上游 config 拒绝 `min_scale: 0`,而 bayesmith 只拒绝负值。
   两者**不冲突**(上游更严),但要有人写下来。
5. R2 清单:`reduced_basis` 的测试族**必须**在 x64 会话(D41 已经把这半个答案定死)。

### 五、Wave D

`chain` + 证据族七模块。另有:
1. **D39 拍板**(归档 manifest 与二进制不绑定)。两个方向已钉住,解除条件写在行内;
   若绑定,是一次 `_FORMAT_VERSION` 提升,两份 fixture 要按新版本重写并**保留旧的**。
2. **要量的两件**(`2026-08-28-g6-consumption.md` §十):(a) `EpochResidual`/`HeldOut`
   的类身份有没有被钉住;(b) `systematic_floor` 上游读 `memory` 并**微分**其先验
   (`_prior_curvature`),而 bayesmith 的入口收现成的 `prior_fisher`——**那一层由谁算**。
3. `marginal/diagnostics.py` 现在 **775 行**,项目上限 800。**下次往里加东西之前先拆。**

### 六、P4 – P7

- **P4**:双岗已建好(`seam.yml` + `crosscheck.yml`),剩下的是 **cross-check 随模块
  退役**——每切一个模块,同批删它的 cross-check 文件,oracle 按铁律 2 改籍或指认。
- **P5**:18 个 run kind 逐 `_KINDS` 冒烟;每批四件套附 extractor 往返。
- **P6**:bayesmith 的发布列车**已走完** 0.1→0.5;**rheplicant 自己的发版还欠着**
  (计划 §七:「程序结束前 rheplicant 发版清 385 提交旧账」)。
- **P7 具名清单**:(a) e-RHINO 的 CLAUDE.md **与** AGENTS.md **成对**改(有测试逐字节
  钉着);(b) README 计数 pin;(c) coverage 截断值 + `fail_under`;(d)
  `_migration-to-bayesmith.md` 整篇;(e) `rheplicant/inference/__init__.py` 的模块
  docstring(**今日三论点在终局全假**,而它是 106 名单门户);(f)
  `tests/test_published_contracts.py`(27 条实测)与其 docstring;(g) bayesmith
  README/CHANGELOG。

### 七、两份新推导页留下的工作

**`2026-08-28-multi-dataset-joint-posterior.md`**:
- §9 是**真正要新建的东西**,最上游是 **`iota` —— 一个被声明的跨 graph 身份映射**。
  理由是本包已经做过的决定:`graph/trace.py` 的 `NodeRef._owner` 拒绝来自另一次
  `trace()` 的 handle,所以**从名字相等推断共享 latent 会反转它**。其余:列见证、
  第二个 graph 级 likelihood factor slot、**realify 层(D46 的另一半)**、
  从 rheplicant 移植的准入闸、nuisance 先验的类型分裂、Tier 1.5 的存储契约。
- §7 是**拒绝清单**(12 条),每条都写了它保护哪个数学陈述。**7.1 已落地(D46)**,
  其余 11 条未落地。
- §10 是**未决问题 + 能了断它们的测量**,9 条。
- §11 明确标了**哪些数字未验证**——引用它们之前先量。

**`2026-08-28-architecture-narrative.md`** §7 表里放行但未做的:
- **R9**:3 处 `Section 9.3` 解不到,而且**本会话让它变糟了**——新推导页有一个主题
  完全不同的 `### 9.3`,所以在 bayesmith 里 grep "9.3" 会得到一个**自信的错误答案**。
- **R10** 的兜底:一份随 sdist 发布的 `docs/plan-codes.md`(`src/` 里还有 ~80 处
  计划代号;**不能盲清**——`tests/exact/test_condition_estimate.py:243` 断言
  `"G14" in text or "D15" in text`,而那个测试在 sdist 里)。
- §8 的「不建议动的」**7 条是结论,不是待办**:不要改 `diagnose/`(e-RHINO 生产代码
  钉着它)、不要把 `optimize.py`/`amortize.py` 提升为子包、不要动 `sqrtinfo.py` 的
  位置与 `information()` 的命名、不要为「全包一致」去补齐门面(40/81 是惯例不是事故)、
  不要碰 `Graph.joint_prior: Any`。

### 八、接线时已经量好、不要重量的七件

- `fit` 收算好的 `step_sizes`;`_magnitude` 留 rheplicant。**两侧默认步长是否等价
  没有量过**。
- `condition_estimate` 两侧默认迭代数**相同**(都是 12),但这个数**改变答案**。
- **D33**:见 §四.3。
- **D19 的前提不成立**(实测):`floor=0` 的块在任何求解之前就被拒。
- **G15 的那一行**:见 §二。
- **Wave C 的 `npe`**:见 §四.2。
- **Wave D 的证据族**:见 §五.2。

## 五、批次纪律(铁律 4,每批四件套)

一个批次 = 一个模块集。完成 = 四件套齐备:

1. 分诊后**该批测试全绿**(分诊三列:原样重放 / 改写对适配器 / 带理由退役);
2. **接缝变异红**,按 §六 五行协议;
3. 旧实现删除,**计数守卫同批刷新**(README 计数、coverage floor);
4. **文档实测数字重测**。

每批把证据链写进本目录的 tracked 执行页(`2026-XX-XX-*.md`)。
**不要**把计划性文件放进 e-RHINO 的 `docs/superpowers`(gitignored,八份计划死在
那里过)。

**改判据 = 新裁决项**,要走登记簿并写进记录页——即使在本次委托下也是。

## 六、这台机器上的操作陷阱(全部实测于 2026-08-26/27/28)

**跑测试**

```bash
# e-RHINO(共享机器上两阶段跑,见其 CLAUDE.md)
.venv/bin/python -m pytest -n 4 --ignore=tests/gui/e2e > run.log 2>&1; echo "PYTEST_EXIT=$?" > run.exit
# bayesmith
.venv/bin/python -m pytest -n 4 > run.log 2>&1; echo "PYTEST_EXIT=$?" > run.exit
```

- **退出码写进自己的文件再读。** 只有 **1** 是测试失败;**5** 是一条都没收集到。
- **harness 的完成通知报的是复合命令的退出码,不是 pytest 的。** 2026-08-27 见了
  一次(通知说 0,`run.exit` 说 1,真有三条红的);**2026-08-28 又见了两次**,
  两次都真有一条红的(都是 README 计数守卫,它该红)。**这条不是偶发。**
- **不要再加 `-q`**:bayesmith 的 `addopts` 已有一个,叠成 `-qq` 会吃掉摘要行。
- e-RHINO 的**部分运行不需要 `--no-cov`**(那条规则早已反转)。

**变异集**

- **五行协议第 (0) 条有两半。** 先提交再变异;**而且每一次跑之前 HEAD 都必须已经
  是你想要回的东西**。修补写在第一轮和第二轮之间、还没提交,会被第二轮自己的开场
  `git checkout` 回退——**输出里没有任何东西说得出这件事**,一个被回退的修补和一个
  不起作用的修补长得一模一样。本会话实测,两条幸存因此被读了两遍。
- **恢复的路径不要宽于变异点本身**:一份脚本为了撤销只在 `src/` 里的变异而恢复了
  整个 `tests/`,那就是上一条的成因。
- **一个幸存的变异有三种成因,而它们要三种不同的处置**(2026-08-28 实测,一组里
  三种全见到了):**真缺口**(补 fixture)、**变异的是够不到的代码**(说出来,并钉住
  够得到的那一半——给它编 fixture 是更贵的错误)、**等价变异**(登记为必存;错的是
  期望值,不是守卫)。
- **一条变异可能必须幸存。** `priors` 批的 P3(合成数据 0 → 1e4)全绿,而这**正是**
  一条测试断言为真的性质。通常读法「没有守卫」在这里是错的;但**追到底仍然适用**
  ——追下去发现测量是拿手搭的图做的,适配器可能造出另一张,于是补了闭环。
- 变异集**可能要跑两个会话**:`priors` 批的两个靶子文件精度需求相反,合在一条命令
  里跑**基线就是红的**。

**git**

- **提交信息里出现 `-n 4` 之类会被 `block-no-verify` 钩子拦下**。把信息写进
  scratchpad 文件再 `git commit -F <文件>`,并且**不要在同一条复合命令里既跑
  `pytest -n 4` 又 `git commit`**。
- `cd` 不跨回合存活:每条 git 命令都用 `git -C <绝对路径>`。
- **推送后用 `git ls-remote` 核实**,不要读本地记录。
- **不要 `git add -A`**(它一天之内把九份草稿扫进库两次);逐路径 `git add`。

**计数与文档**

- e-RHINO 的 `tests/test_readme_counts.py` **按等号钉住** README 的计数,并且
  会**先红**。让它红,然后用**它自己报的数字**改 README——不要自己加。
- **拒绝普查的计数可以在内容变了两处时保持不变**(退役一条、新增一条)。
  `priors`/`G13` 那两批各发生过一次,而附录 B 因此**过期了一整批**没人发现。
  附录 B 的逐文件清单现在**由 `test_refusal_census._sites()` 重生成**,不要手抄。
- coverage floor 只住在 `[tool.coverage.report] fail_under`。
- **fixture 普查会无参驱动每一个 `*_document` builder**,所以一份文档能声明什么
  受限于**谁会去构建它**——`runtime.jax_enable_x64: true` 写进 builder 会打红两条
  毫不相干的普查测试(实测)。这类声明写进**运行它的 helper**。

**探针**

```bash
cd /Users/zzhang/projects/bayesmith
/Users/zzhang/projects/e-RHINO/.venv/bin/python docs/probes/probe_11_d17_dual_run.py
/Users/zzhang/projects/e-RHINO/.venv/bin/python docs/probes/probe_12_d16_five_axes.py
```

```bash
cd /Users/zzhang/projects/e-RHINO
JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/seam
```

**全套里有两个 x64 会话**——`tests/evidence/` 与 `tests/seam/`,各有自己的门
conftest 与驱动。**另有两个 module 级 autouse x64 fixture**
(`test_prior_sensitivity.py`、`test_jeffreys_prior.py`),它们**各自制造过一次
「守卫不会失败了」**(S6,以及 D25 的范围只有三条而非四十九条)。**遇到一个
module 级 x64 fixture,先问它拿掉了什么条件。**

## 七、本程序反复付学费的形状(检查表)

1. **分不清「是 X」与「这次查询根本没发生」的结果。** 退出码、zsh glob、PyPI 索引、
   被回退的修补——同一个家族。
2. **比对悄悄变成「比两个不同模型」。** 每个 fixture 的数学只写一次,两侧各包一层。
3. **判据在正确的地方、时机却是错的。** `auto_blocks` 在求解期问了一个分区期的
   问题;G13 那批的守卫用了一个**没有 joint prior 的空间**去测「未覆盖的 latent
   保留先验」,于是变异前后两种读法都说「未覆盖」。**问「这个检查拿到它需要的全部
   输入了吗」。**
4. **一条从另一个库搬来的「常识」。** JAX 的 `bool * float` 是选择,NumPy 是相乘。
   跨库搬事实要重测;**一个幸存的变异要追到底**——真正的缺陷常在第二层。
   S6 是这条的第二次兑现:`astype` 做了两件事,承重的是**剥掉弱类型**那一半,
   而不是加宽那一半。
5. **一个数字在两个地方拼写,其中一份会过期。** 附录 B 过期了一整批;`identifiability`
   的契约页 §5.2 过期过;本页 §三 关于 CLAUDE.md 的那条过期过。**能推导就不要重抄。**
6. **一条链量不出的东西,不要让它去量。** `priors` 批第一版拿两条链的均值差去钉
   先验位移,跨三个种子比值 0.55/1.44/1.09 且有一个分量换了符号——那是
   `prior_sensitivity` 自己 docstring 里的算术。改用**共同随机数**之后判据里一个
   容差都没有:平先验必须让轨迹**逐比特不变**,不平的必须让它变。

## 八、上下文用尽时怎么收尾(不要硬撑)

程序全长估计 27–40 个会话,一个会话跑不完。**不要为了「跑到结束」而把批次做
半截。** 当上下文接近用尽:

1. 把当前批次收到一个**四件套齐备**的状态(或干净回退到批次开始);
2. **两仓提交,但不要推送。** owner 于 2026-08-27 新增:推送攒到全部任务做完时
   两仓一起推,因为每次推送触发两个 workflow、失败邮件太多。**这条只改推送时机,
   不改提交纪律**——每批仍必须收到四件套齐备并本地提交,否则一次上下文用尽就会丢掉;
3. 更新本文件(它是交接页):改写 §二 当前状态、§三 已处置的项、§四 剩余工作,
   让下一个会话能从这里直接开工;
4. 提交这次更新,然后停。

**没跑完不是失败;把一个半截批次留在工作树里才是。**

> **推送攒着有一个已知代价,写在这里以免它变成一次意外**:CI 只在推送时跑,而
> `2026-08-27-ci-flat-chain.md` 记的那次是「本机连绿五次而远端连红五次」。所以
> **最后那一次推送之后必须去看两个 workflow 的结论**,而且要预期它可能红——积压
> 的批次越多,一次推送要一起验证的东西就越多。那份记录页 §六 提的「把查一次 CI
> 写进收尾清单」**仍未入登记簿**;它是一条判据改动,要走登记簿,而**没有哪一批
> 自作主张改过铁律**。
