# 交接 prompt — 把「全面移交」程序跑到完全结束

> 与 kickoff 同目录、同被跟踪。使用方式:把 `---` 以下整段粘贴给新 session。
> **日期**:2026-08-27(第七次改写)· 交接自 **0.4.0 发布、Wave A 开工、
> 第一个模块 `identifiability` 切换完成**之后的会话。

---

你要把「全面移交」程序**跑到完全结束**:把 rheplicant 的全部贝叶斯数值实现
移交给 bayesmith,`rheplicant.inference` 收缩为门面。计划已定稿,不要重新
评审、不要重新设计。

## 〇、owner 的授权(最重要,先读)

**owner 已授权:不要停下来问,一直做到程序结束。** 这条改变了裁决纪律的形态,
所以要写清楚它**具体**是什么意思:

- 计划 §二 登记簿里**仍未拍板**的 D 项(见 §三),owner **委托给计划自己的
  建议**。对每一项:采纳该行 `*建议*` 所写的选项,并把裁决回填进那一行,格式
  为 `**【owner 已授权委托,YYYY-MM-DD:取 (x),依 2026-08-27 的一次性委托】**`,
  后接一句你为什么读成这个选项。
- **委托不是空白支票。** 如果照建议执行会与**已实测的事实**冲突(例如建议
  的前提被一次测量推翻,像 v2 的 D14 前提那样),**不要**照做:把冲突写进那一行,
  按事实选择,并在批次记录页里点名说明。计划自己就是这样被改过两次的。
- 遇到**计划完全没有预见**的新裁决点:自己选,选保守的一侧(能保住正确性、
  能被守卫钉住的那一侧),把它作为新条目写进登记簿并注明是本次委托下自定的。
- **唯一仍要停下来的情况**:一个动作会**不可逆地对外**(发一个新 PyPI 版本号、
  强推覆盖远端、删除未跟踪的用户文件)**且**计划没有明写该动作。发布本身是
  计划明写的(D13),照做即可。

## 一、第一步永远是读(按此顺序)

1. **计划本体**:`docs/superpowers/specs/2026-08-26-one-implementation.md`
   ——终局形态、七条铁律、登记簿 D7–D19、缺口 G1–G14、四波切换、附录 A/B/C。
2. 本程序至今的**十一份**执行记录,同目录:
   `2026-08-26-wave-P0.md`、`2026-08-26-wave-P2a.md`、
   `2026-08-27-d17-protocol.md`、`2026-08-27-d16-five-axes.md`、
   `2026-08-27-wave-P1.md`、`2026-08-27-wave-P2-G1.md`、
   `2026-08-27-wave-P2-G13.md`、`2026-08-27-wave-P2-G7.md`、
   `2026-08-27-wave-P2-D9.md`、`2026-08-27-wave-A-opening.md`、`2026-08-27-wave-A-identifiability.md`。
   **G1 那份最值得先看**:它记的三处「守卫已经不会失败了」是三种不同的形状,
   而三种都只有变异能看见。
3. 前代 spec 的 §四(模块绑定契约台账)与 §六(本程序的由来):
   `2026-08-24-rheplicant-migration.md`。
4. 两仓工作笔记(操作规则,条条是学费):`/Users/zzhang/projects/e-RHINO/CLAUDE.md`
   (与 AGENTS.md 逐字节一致,有测试钉着)与 `/Users/zzhang/projects/bayesmith/CLAUDE.md`。

## 二、当前状态(2026-08-27 实测,先复核再信)

- **bayesmith** `95d82ff`,已推送;**0.4.0 已在 PyPI 上**(tag `v0.4.0`,
  run 33063195570);套件 **1291 passed / 0 skipped**,exit 0,208 s;`ruff` 干净。
  crosscheck **119**(identifiability 那份已随模块退役)。CHANGELOG `Unreleased`
  段为空——G1/G13/G7/D9 都在 0.4.0 里。
  **0.4.0 的确认方式照 P0 的教训做**:绿的 publish workflow 是**记录不是索引**;
  在 `pypi.org/simple/bayesmith/` 上见到 whl 与 sdist(**JSON API 当时仍报
  0.3.0**,滞后如 P0 所记),再用 `uv pip install --target … bayesmith==0.4.0`
  真解析一次,并在装出来的 `dist-info/METADATA` 里核实版本号。
- **e-RHINO** `647a2ed`,已推送,**本会话未动一行源码**。上次全量:
  **10057 passed / 522 skipped**(`-n 4 --ignore=tests/gui/e2e`)加 **21 passed**
  (`tests/gui/e2e -n 2`);README 计数 **10599**,coverage floor
  `fail_under = 89`(均未动)。
- **e-RHINO** `860703d`,已推送。**源码已动**:`inference/identifiability.py`
  是本程序第一个被切换的模块。两阶段套件 **10061 passed / 522 skipped** exit 0
  加 **21 passed** exit 0;README 计数 **10603**(由守卫报数);coverage floor
  未动(`fail_under = 89`)。
- **跨仓实测(2026-08-27)**:e-RHINO `tests/inference` **996 passed** exit 0;
  `tests/seam`(x64)**19 passed / 1 xfailed** exit 0。四个批次各跑过一次,一致。
- **bayesmith 地板仍是 `>=0.3`,这是有意的。** e-RHINO 今天**没有用**任何 0.4.0
  的表面:适配器仍拒绝 `FlaggedNoise`,没有任何图声明 `joint_prior`,诊断走的还是
  rheplicant 自己那份。地板声明的是**真实需要**(`pyproject.toml` 那段注释自己
  这么写),所以它**随 Wave A 第一个模块切换同批升到 `>=0.4`**,不是现在。
  那个 xfail 是 D19 的闹钟,**strict**,Wave B 落地当天它会因为「意外地绿」而红。
- 两仓互为 editable 安装;e-RHINO 的 bayesmith 地板 **`>=0.3`**。
- 两个跨仓 workflow 并行:`crosscheck.yml` 与 `seam.yml`。
- e-RHINO 根目录九份未跟踪评审/交接草稿:**不动**(附录 C 明令)。

**已完成**:P0、P2a、D17 协议、D16 四条轴、P1、**P2 余项的 G1、G13、G7**、
**D9**、**0.4.0 发布**、**Wave A 开工批 + 模块 1 `identifiability`**(记录页 `-G1.md`、`-G13.md`、`-G7.md`、
`-D9.md`、`2026-08-27-wave-A-opening.md`)。

**登记簿新增 D20**(掩码的声明面)、**D21**(诊断的展开点)、**D22**(秩测试合成图的三样)。
**两次事实修正**(委托下按「不是空白支票」那条处置):**D13**(发布号
归属)与 **D9**(float32 政策)。D9 的建议 (b) 被谱隙表推翻,改取 (a);两条的
冲突都写在各自那一行里。

**登记簿**:D7–D19 全部有裁决;**D20 为本次委托下自定的新条目**(掩码的声明面
取节点声明,不取 inf-σ),理由回填在计划 §二。

## 三、四批留下的几件事(下一位会先撞到)

1. **`Unreleased` 段不为空,而适配器在等它。** e-RHINO 的
   `graph_bridge.py::_refuse_flagged_noise` 仍原样拒绝 `FlaggedNoise`,文案里
   写着「Masking is the G1 gap on the bayesmith side」。按铁律 5,接线要等承载
   G1 的 bayesmith 发布上索引;按 D13,那一版在 **P2 余项做完之后**发。所以
   接线是 P2 收尾那一批的事,不是现在。届时
   `tests/inference/test_refusal_census.py` 的逐文件计数会红并**报出**要写进
   附录 B 的新数字——按它报的改,不要自己加。
2. **`evidence/campaign.py` 的分期切片对 `MaskedPrecision` 无规则**,会以
   「no rule for slicing a MaskedPrecision」明确报错。证据层是 Wave D,补齐还是
   保留为限制由那一波定,并上登记簿。
3. **`exact/fisher.py` 的模块 docstring 里写着 `propagate_covariance` 与
   `push_forward`「are P5」**,而计划 §四 把它们放在 **G7**。那是一句过期的排期
   注记,不是判据,所以开工 G7 时**顺手改掉**即可,不必上登记簿——但要在 G7 的
   记录页里点名说明这次更正。
4. **G13 只验了势能,没跑真链。** `nuts()` 现在会自动带上 `joint_prior` 的
   factor site(它走 `to_numpyro`);本批次用 `numpyro.infer.util.log_density`
   一次求值验的,理由是丢了 factor 的 handler 会改变**每一点**的势能。真链验收
   归 Wave A 切 `priors` 那一批。
5. **D9 已完成,但结论与裁决相反,而这正是要读的那一条。** (b)「从 dtype 推导
   rtol」被谱隙表推翻——不是难实现,是**不存在**:float32 下最小奇异值坐在自己的
   舍入地板上,float64 分得开两个量级的模型在那里无法分辨。改取 (a),
   `refuse_ambient_float32` **一行未松**。**Wave A 切 diagnose 三模块时,适配器
   必须在 x64 上下文内建图**;做错会被指名拒绝(实测),所以这是一条有守卫的义务。
6. **D9 还剩第二项功课**:`parameter_covariance` 的 `1/√eps` 天花板拒绝逐 fixture
   冒烟。在 (a) 下消费者遇到的是 float64 的 **6.71e7** 而不是 float32 的 2.90e3,
   所以范围比 D9 原文小得多——但不为零,归 Wave A 的 `uncertainty` 批次。
7. **五行协议现在有第 (0) 条**:先提交本批次,再跑变异集。`git checkout -- src/`
   以 HEAD 为准,在一棵带未提交改动的树上它回退的是**工作**而不是变异——本会话
   实测,它吃掉了 G1 的第一版源码。已写进计划 §六 与 bayesmith 的 CLAUDE.md;
   **e-RHINO 的 CLAUDE.md/AGENTS.md 还没写**(它们逐字节一致,由
   `tests/test_docs_claims.py` 钉着,要成对改),留给下一个动 e-RHINO 的批次。

## 四、剩余工作(按计划 §九 的顺序)

1. ~~P1 适配器~~ 已完成;~~G1 掩码~~、~~G13 图级联合先验~~、~~G7 bridge 补齐~~、
   ~~D9 float32 政策~~ **均已完成 2026-08-27**。
2. **P2 余项(按计划 §九 的口径,还剩五项)**:**G2** `bayesmith.fit`、**G9 全量**(vmap / log 空间 / Fisher 的复数面;另登记在案的两项:`diagnose`
   仍拒绝复 latent、`exact.correct.log_weight` 仍在域里索引)、**G10** 分区执行面
   (三件全做,D14 已拍)、**G12**、**G14**。
   > 口径差,写明以免下一位两处都读、两处都信:本页早先版本把 G3/G4/G5/G6 也
   > 列进「P2 余项」;计划 §九 的排期表把它们分别排在 **Wave C(+G4/G5 实现)**
   > 与 **Wave D(+G3/G6 实现)**。**以 §九 为准。**
3. **收尾发布**:P2 余项做完后发一版 bayesmith(D13),然后 e-RHINO 侧接线
   (含 §三 第 1 条)。
4. **P3 四波**:Wave A(先决 P1+D9+D16+**G7**+G11+**G13**)、Wave B、Wave C、
   Wave D。
5. **P4** 质量机制换防;**P5–P7**。

**Wave A 已开工,先决全部就位,发布门已开**(0.4.0 在索引上)。

**下一批是 Wave A 的模块 2:`sensitivity`。** 理由与陷阱写在
`2026-08-27-wave-A-identifiability.md` §七,三条,其中第 3 条最要紧:

* 切它才能把 `_flat_view` 真正删掉(它是最后一个消费者);
* 它**需要 0.4**(D9 修好的 `prior_sensitivity` 守卫),所以**那一批同时把
  e-RHINO 的 bayesmith 地板升到 `>=0.4`**;
* **不要照抄 `_graph_for_rank` 的合成。** D22 的不变性对 `identifiability` 成立
  是因为秩判决读不到先验;而 `prior_sensitivity` **算的就是先验位移**,所以合成
  先验对它**不是**中性的。这是一条必须重新测量而不能继承的假设。

**bayesmith 地板仍是 `>=0.3`**:模块 1 的门面只用到 0.3.0 就有的四个名字。地板
声明的是真实需要,随模块 2 升。

**并行候选**:**G2 `bayesmith.fit`**——Wave B 的先决,D7/D11 共同指向的 gradient-MAP
出口,在 bayesmith 一侧、不受发布门约束。

## 五、批次纪律(铁律 4,每批四件套)

一个批次 = 一个模块集。完成 = 四件套齐备:

1. 分诊后**该批测试全绿**(分诊三列:原样重放 / 改写对适配器 / 带理由退役);
2. **接缝变异红**,按 §六 五行协议(scratch 变异、`git checkout` 恢复不用 `cp`、
   两者之间 `rm -rf __pycache__`、击杀 = 退出码**恰为 1** 且**指名测试**在红名单、
   变异集前后各一次基线绿);
3. 旧实现删除,**计数守卫同批刷新**(README 计数、coverage floor);
4. **文档实测数字重测**。

每批把证据链写进本目录的 tracked 执行页(`2026-XX-XX-*.md`)。
**不要**把计划性文件放进 e-RHINO 的 `docs/superpowers`(gitignored,八份计划
死在那里过)。

**改判据 = 新裁决项**,要走登记簿并写进记录页——即使在本次委托下也是。

## 六、这台机器上的操作陷阱(全部实测于 2026-08-26/27)

**跑测试**

```bash
# e-RHINO(共享机器上两阶段跑,见其 CLAUDE.md)
.venv/bin/python -m pytest -n 4 --ignore=tests/gui/e2e > run.log 2>&1; echo "PYTEST_EXIT=$?" > run.exit
# bayesmith
.venv/bin/python -m pytest -n 4 > run.log 2>&1; echo "PYTEST_EXIT=$?" > run.exit
```

- **退出码写进自己的文件再读。** 只有 **1** 是测试失败;**5** 是一条都没收集到
  (本会话踩过:在 bayesmith 目录里跑 e-RHINO 的路径,读成「没有失败」)。
- **harness 的完成通知报的是复合命令的退出码,不是 pytest 的。** 本会话见过
  「1 failed」与通知里的「exit code 0」并存。
- **不要再加 `-q`**:bayesmith 的 `addopts` 已有一个,叠成 `-qq` 会**整个吃掉
  摘要行**。计数从 `--junit-xml` 取。
- e-RHINO 的**部分运行不需要 `--no-cov`**(那条规则早已反转)。

**git**

- **提交信息里出现 `-n 4` 之类会被 `block-no-verify` 钩子拦下**(它把 `-n` 读成
  绕过标志)。把信息写进 scratchpad 文件再 `git commit -F <文件>`,并且**不要
  在同一条复合命令里既跑 `pytest -n 4` 又 `git commit`**——本会话两种都踩过。
- `cd` 不跨回合存活:每条 git 命令都用 `git -C <绝对路径>`。
- **推送后用 `git ls-remote` 核实**,不要读本地记录。
- 子代理一律被拒 `EnterWorktree`;编排者自建 worktree + `git -C` 可行。
- **跑变异集之前先把批次提交。** `git checkout -- src/` 以 HEAD 为准;在一棵带
  未提交改动的树上它是一次静默的全量回退。实测吃掉过一整批未提交的源码改动。
  另外两条同源的脚本教训:变异日志要 `flush`(被杀的运行否则只留下一个 0 字节
  文件),`rglob("__pycache__")` 不要从仓根走(它连 `.venv` 一起删,把 15 秒的
  变异拖成 2 分钟)。

**计数与文档**

- e-RHINO 的 `tests/test_readme_counts.py` **按等号钉住** README 的计数,并且
  会**先红**。让它红,然后用**它自己报的数字**改 README——不要自己加。
- 每加/删测试都要过这一关;coverage floor 只住在 `[tool.coverage.report]
  fail_under`。
- bayesmith 的 README 与 CLAUDE.md 各有一个测试计数,每批同步。

**两个可重跑的探针**(它们是 D16/D17 的回归证据)

```bash
cd /Users/zzhang/projects/bayesmith
/Users/zzhang/projects/e-RHINO/.venv/bin/python docs/probes/probe_11_d17_dual_run.py
/Users/zzhang/projects/e-RHINO/.venv/bin/python docs/probes/probe_12_d16_five_axes.py
```

改动任一侧探针语义后重跑它们;`probe_11` 现在应报 **6/8 一致**(余下两例是
D16 轴 1/轴 5,已随 D16 落地结清——若它现在报 8/8,把那两行的期望更新掉)。

**第三条,P1 起可跑**(适配器的验收层,必须带 x64):

```bash
cd /Users/zzhang/projects/e-RHINO
JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/seam
```

默认(float32)会话里它整目录 **skip**,那是设计:`tests/test_seam_session.py`
以子进程跑上面这条命令,并在它红时红。**全套里现在有两个 x64 会话,不是一个**
——`tests/evidence/` 与 `tests/seam/`,各有自己的门 conftest 与驱动。

## 七、本会话反复付学费的三种形状(值得当成检查表)

1. **分不清「是 X」与「这次查询根本没发生」的结果。** D17 协议第一次跑报
   「1/6 一致」,分组全对、种类全是 gradient——因为 `.get(engine, "gradient")`
   的默认分支吃掉了 `engine is None`。它**没有报错**,只给出一张完整、自洽、
   全错的判决表。查询表一律写成**没有默认分支**的显式字典。
2. **比对悄悄变成「比两个不同模型」。** 一个图 fixture 漏了加性 offset,与手造
   孪生体差 0.16,读起来完全像求解器 bug。**每个 fixture 的数学只写一次,两侧
   各包一层。**
3. **判据在正确的地方、时机却是错的。** rheplicant 的 `auto_blocks` 有和
   bayesmith 一模一样的两条拒绝、同一个常数 0.06,只是住在求解期,于是它产出
   了自己的求解器会拒绝的分区。**问「这个检查拿到它需要的全部输入了吗」。**

另外:**一个刚写完的守卫要先证明它还能失败**。对每个新判据都跑变异
(D16 四条轴、G9 六个变异、G11 的 AST 普查、G1 十四条),每次都问「红的是不是
**我的**断言」。

4. **一条从另一个库搬来的「常识」。** G1 的 `MaskedPrecision` 有一段 docstring
   论证「`0 * nan` 是 `nan`,所以要在输入侧也掩」——为它写的变异**幸存**了。
   实测:JAX 里 `bool 数组 * float 数组` 是**选择**(`[T,F,T] * [1,nan,3]` 得
   `[1,0,3]`,eager 与 jit 一致),而 NumPy 是相乘(得 `[1,nan,3]`)。那句
   「常识」是 NumPy 的,搬进来防的是一个不存在的东西。**顺着这条查下去才发现
   真的洞**:乘法在 `quadratic` 里,`sum(r * apply(r))`,于是求解干净而密度是
   `nan`。教训是两层:跨库搬事实要重测;以及**一个幸存的变异要追到底,不要
   补一条断言就算了结**——真正的缺陷在第二层。

## 八、上下文用尽时怎么收尾(不要硬撑)

程序全长估计 27–40 个会话,一个会话跑不完。**不要为了「跑到结束」而把批次做
半截。** 当上下文接近用尽:

1. 把当前批次收到一个**四件套齐备**的状态(或干净回退到批次开始);
2. 两仓提交并推送,`ls-remote` 核实;
3. 更新本文件(它是交接页):改写 §二 当前状态、§三 已处置的 D 项、§四 剩余
   工作,让下一个会话能从这里直接开工;
4. 提交这次更新,然后停。

**没跑完不是失败;把一个半截批次留在工作树里才是。**
