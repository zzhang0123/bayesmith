# 交接 prompt — 把「全面移交」程序跑到完全结束

> 与 kickoff 同目录、同被跟踪。使用方式:把 `---` 以下整段粘贴给新 session。
> **日期**:2026-08-27(第十四次改写)· 交接自 **Wave A 全部完成 + P2 余项的
> G2 / G10 / G12 / G14 四项完成**之后的会话。

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
   ——终局形态、七条铁律、登记簿 **D7–D37**、缺口 G1–G15、四波切换、附录 A/B/C。
2. 本程序至今的**二十四份**执行记录,同目录。**最近四份最要紧,按序读**:
   `2026-08-27-wave-A-uncertainty-covariance.md`(Wave A 收尾)、
   `2026-08-27-p2-g2-fit.md`、`2026-08-27-p2-g10-g12.md`、
   `2026-08-27-p2-g14-condition-estimate.md`(**它的 §七 是下一步的开工清单**)。
   再往前:`2026-08-27-ci-flat-chain.md`(**本机绿而远端红的唯一一次,读它的 §三
   判别方法**)、`2026-08-27-wave-A-uncertainty-fisher.md`、
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

## 二、当前状态(2026-08-27 实测,先复核再信)

- **bayesmith** 套件 **1350 passed / 0 skipped** exit 0(249.7 s);
  `ruff check src/ tests/` **干净**——**上一版说它干净时它已经不是了**(一条 `ISC004`
  随 `0c5ca10` 进来,本会话修掉;这个数没有 CI 在跑,所以每次都要重量)。
  **0.4.0 在 PyPI 上**(tag `v0.4.0`)。
- **CHANGELOG 的 `Unreleased` 段不再为空**:**G2、G10、G12、G14** 四项未发布表面。
  **铁律 5 因此变成一件要主动做的事**——Wave B / Wave C 接线之前**必须先发一版**。
  今天满足铁律 5 的原因只是 rheplicant 一行都还没依赖它们。
- **两仓各有提交未推送**(按 §〇 owner 新增的那条攒着):
  **e-RHINO 领先 4 个,bayesmith 领先 15 个**。
- **e-RHINO** 两阶段套件 **10109 passed / 534 skipped** exit 0(341.2 s,
  `-n 4 --ignore=tests/gui/e2e`)加 **21 passed** exit 0(`tests/gui/e2e -n 2`)。
  README 计数 **10663**(由守卫报数);拒绝普查 **250**
  (`StateValidationError` **64**、`ParameterSpaceError` 178、`RuntimeError` 4、
  `Exception` 3、`TypeError` 1);coverage floor `fail_under = 89` 未动;
  bayesmith 地板 **`>=0.4`**。
  **e-RHINO 的 `ruff check src/ tests/` 有 13 条**(`I001` 9、`F401` 3、`E501` 1),
  全部与本程序无关且**没有 CI 在跑它**;扫不扫是一条独立的决定。
- 两仓互为 editable 安装;两个跨仓 workflow 并行(`crosscheck.yml`、`seam.yml`)。
- e-RHINO 根目录九份未跟踪评审/交接草稿:**不动**(附录 C 明令,且已 gitignore)。

**已完成**:P0、P2a、D17 协议、D16 四条轴、P1、**P2 余项的 G1/G13/G7/D9**、
**0.4.0 发布**、**Wave A 全部五个模块**(`identifiability`、`sensitivity`、
`priors`、`numpyro_bridge`、`uncertainty`)、以及 **P2 余项的 G2 / G10 / G12 / G14**。

**登记簿 D7–D37 全部有裁决,只有 D23 例外**(见 §三)。**缺口:G15 新开且未做。**

## 三、至今各批留下的几件事(下一位会先撞到)

1. **D23 是唯一一条已登记、未裁决、当前无守卫的语义差。**
   `prior_sensitivity` 的拒绝判据:rheplicant 在**观测雅可比的秩**上拒绝,
   bayesmith 在 **rest 项自身的曲率**上拒绝,而两侧措辞相同、64 条测试无一分辨得出。
   要正式采纳曲率判据,先得造出能分辨它的 fixture(**下游密度持住的被选 latent**),
   那是一次语义升级,届时在那一行拍板。
2. **`evidence/campaign.py` 的分期切片对 `MaskedPrecision` 无规则**,会以
   「no rule for slicing a MaskedPrecision」明确报错。证据层是 Wave D,补齐还是
   保留为限制由那一波定,并上登记簿。
3. **两个模块「只切了一半」,而簿记如实反映了这件事。**
   `numpyro_bridge.py` 与 `uncertainty.py` **都不在 `SWITCHED` 里**,两者的
   cross-check **都保留**。`uncertainty` 这一侧「cross-check 还能不能失败」是
   **变异量出来的**(U6:改适配器的图 → bayesmith 的四条红),不是推的。
   **Wave B/C/D 若再遇到同样形状,这是先例。**
4. **G15 未做,而它的解除条件逐字写在代码里。**
   `fisher_information(space=...)` 的先验曲率暂时留守,因为远端没有一个**带先验的
   非线性局部块**。解除条件在 `uncertainty._prior_precision` 的 docstring:G15 发布
   之后删掉该函数、调用改成 `include_prior=space is not None`,**只有那一行会变**。
5. **D8 的 (iv)(gcr+snis 语义升级)未做。** 分期的退场条件写在 D8 那一行:
   (i') 的近似声明在 (iv) 落地当批撤除;若推迟,推迟本身进登记簿。G12 做的是 (i')。
6. **一条同样的缺陷形状,在最后两批里各出现一次,值得当成检查项。**
   G10 的 **W8** 与 G14 的 **X4**:**守卫的 fixture 够不到它存在的那个条件**,
   两条都是变异幸存才发现的。X4 还多一层——**第一次修补也够不到**,因为它写在了
   另一个函数上。收工前问一句「这条守卫的 fixture 真的制造出它要防的那个情形了吗」。

## 四、剩余工作(按计划 §九 的顺序)

1. **P2 余项只剩 G9 全量**:vmap、log 空间、Fisher 的复数面;另登记在案的两项
   ——`diagnose` 族对复 latent 的拒绝、`exact.correct.log_weight` 仍在域里索引。
   > 口径差,写明以免两处都读、两处都信:早先版本把 G3/G4/G5/G6 也列进「P2 余项」;
   > 计划 §九 的排期表把它们分别排在 **Wave C(+G4/G5 实现)** 与
   > **Wave D(+G3/G6 实现)**。**以 §九 为准。**
2. **收尾发布(D13)**:G9 全量做完后发一版 bayesmith。**这一步现在是硬门**
   ——Unreleased 里已经有四项,Wave B/C 接线依赖它们。
   **发一个新 PyPI 版本号是授权里明写要停下来问的三件事之一**(§〇 最后一条),
   所以到这一步要停下来问 owner,除非计划已明写该动作(D13 明写了发布列车,
   但没有明写这一版的号)。
3. **P3 剩三波**:Wave B(求解与计划)、Wave C(calibrate/npe/reduced_basis)、
   Wave D(chain + 证据族)。**Wave A 已完成。**
4. **P4** 质量机制换防;**P5–P7**。
5. **G15**:带先验的**非线性**局部块,在 bayesmith 一侧实现并发布,然后拆掉
   rheplicant 里那条有解除条件的延期。排期由做它的那一批定。

**Wave B 开工时已经量好、不要重量的三件**:
- `fit` 的逐 latent 步长收的是**算好的** `step_sizes`;`_magnitude` 留在 rheplicant
  (声明层的知识)。**两侧默认步长是否等价没有量过**,那是接线批的第一件事。
- `condition_estimate` 两侧的默认迭代数**相同**(都是 `POWER_ITERATIONS = 12`,实测),
  但这个数**改变答案**,所以任一侧改它都是一次数字重测。
- **D33 改变了 Wave C 的分诊**:`fit` 拒绝一次发散的下降,而 rheplicant 的两个
  calibrator 今天**把 NaN 交回去**。依赖那个行为的测试要走分诊第二列——**今天没有
  量过有没有这样的测试**。

**一件仍然悬着的事,是 owner 的决定**:`860703d` 的历史里含有九份评审草稿
(`git add -A` 误提交)。已从索引移除(`f8a73eb`)并 gitignore,文件在盘上。
**把它从历史里去掉需要强推**,而强推是授权明写要停下来问的三件事之一,所以没做。

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

## 六、这台机器上的操作陷阱(全部实测于 2026-08-26/27)

**跑测试**

```bash
# e-RHINO(共享机器上两阶段跑,见其 CLAUDE.md)
.venv/bin/python -m pytest -n 4 --ignore=tests/gui/e2e > run.log 2>&1; echo "PYTEST_EXIT=$?" > run.exit
# bayesmith
.venv/bin/python -m pytest -n 4 > run.log 2>&1; echo "PYTEST_EXIT=$?" > run.exit
```

- **退出码写进自己的文件再读。** 只有 **1** 是测试失败;**5** 是一条都没收集到。
- **harness 的完成通知报的是复合命令的退出码,不是 pytest 的。** 本会话又见了
  一次:通知说「exited with code 0」,`run.exit` 说 `PYTEST_EXIT=1`,而那次真有
  三条红的。
- **不要再加 `-q`**:bayesmith 的 `addopts` 已有一个,叠成 `-qq` 会吃掉摘要行。
- e-RHINO 的**部分运行不需要 `--no-cov`**(那条规则早已反转)。

**变异集**

- **五行协议第 (0) 条有两半。** 先提交再变异;**而且每一次跑之前 HEAD 都必须已经
  是你想要回的东西**。修补写在第一轮和第二轮之间、还没提交,会被第二轮自己的开场
  `git checkout` 回退——**输出里没有任何东西说得出这件事**,一个被回退的修补和一个
  不起作用的修补长得一模一样。本会话实测,两条幸存因此被读了两遍。
- **恢复的路径不要宽于变异点本身**:一份脚本为了撤销只在 `src/` 里的变异而恢复了
  整个 `tests/`,那就是上一条的成因。
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
