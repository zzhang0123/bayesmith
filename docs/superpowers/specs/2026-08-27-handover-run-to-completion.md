# 交接 prompt — 把「全面移交」程序跑到完全结束

> 与 kickoff 同目录、同被跟踪。使用方式:把 `---` 以下整段粘贴给新 session。
> **日期**:2026-08-27 · 交接自 P0 / P2a / D16 / D17 完成后的会话。

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
2. 本程序至今的四份执行记录,同目录:
   `2026-08-26-wave-P0.md`、`2026-08-26-wave-P2a.md`、
   `2026-08-27-d17-protocol.md`、`2026-08-27-d16-five-axes.md`。
3. 前代 spec 的 §四(模块绑定契约台账)与 §六(本程序的由来):
   `2026-08-24-rheplicant-migration.md`。
4. 两仓工作笔记(操作规则,条条是学费):`/Users/zzhang/projects/e-RHINO/CLAUDE.md`
   (与 AGENTS.md 逐字节一致,有测试钉着)与 `/Users/zzhang/projects/bayesmith/CLAUDE.md`。

## 二、当前状态(2026-08-27 实测,先复核再信)

- **bayesmith** `eb69414`,已推送;**0.3.0 在 PyPI 上**(simple 索引已见 wheel
  与 sdist);套件 **1235 passed / 0 skipped**;`ruff check src/ tests/ examples/`
  干净。
- **e-RHINO** `647a2ed`,已推送;套件两阶段跑
  **10057 passed / 522 skipped**(`-n 4 --ignore=tests/gui/e2e`)加
  **21 passed**(`tests/gui/e2e -n 2`),两阶段退出码均 0;README 计数 **10599**,
  coverage floor `fail_under = 89`(未动)。
- 两仓互为 editable 安装;e-RHINO 的 bayesmith 地板已升到 **`>=0.3`**。
- **两个跨仓 workflow 现在并行**:`crosscheck.yml`(两实现是否仍一致)与
  **新增的 `seam.yml`**(rheplicant 的 inference 层与适配器验收层在这份
  bayesmith 上是否还能跑)。crosscheck 对 e-RHINO main 仍 **123 passed /
  0 skipped**。
- e-RHINO 根目录九份未跟踪评审/交接草稿:**不动**(计划附录 C 明令)。
  `run.log`/`run.exit`/`run2.log`/`run2.exit` 已入 `.gitignore`。

**已完成**:P0(落盘 + 发布 0.2.0)、P2a(G11 + G9 最小面含 `ComplexNormal`)、
D17 裁决协议、D16 四条轴落地、**P1(适配器 + 钉名十例 + 拒绝文案清单 +
接缝 CI + 发布 0.3.0)**——记录页 `2026-08-27-wave-P1.md`。

**已拍板的 D 项**:D7、D8、D14、D16、D17、D18、D19(owner 亲拍);
**D9、D10、D11、D12、D13、D15(2026-08-27 的一次性委托下已处置,理由回填在
各自那一行)**。**登记簿至此全部有裁决。**

## 三、委托下已处置的六项(2026-08-27),以及一条被实测收紧的

| 项 | 结果 |
|---|---|
| **D9** float32 政策 | 取 (b) 主、(a) 兜底;(a) 的**启用条件**写在那一行,以免兜底变主路 |
| **D10** NPE 迁移 | (2) 续用 `NoiseModel.realise`;(3) 薄包装保持三名 |
| **D11** calibrate | 迁为 `bayesmith.fit`(G2) |
| **D12** 证据族 API | 容器保持自有类,逐调用互转;**读档 fixture 的前置条件不因委托而松动** |
| **D13** 发布列车 | **事实修正**:0.3.0 承载 **P2a**(已发),P2 余项完成后另发一版;理由是铁律 5 与原文冲突 |
| **D15** condition_estimate | 取 (a),移植为 G14 的 measured-κ 诊断 |

**D19 被一次实测收紧(裁决不变,范围变)**:登记簿写「首解退化」,实测退化发生
在**更早一层**——`linear_operator` 自己的线性化点就取在先验中心,σ=0,
`check_linearity` 先抛 `StructureError`。所以 Wave B 的数据锚定起步要**同时**
覆盖 block 的 `at` 与 GLS 的起点。已编码成
`tests/seam/test_p1_ten_examples.py` 里一条 `xfail(strict=True)` 的回归 fixture
——**strict**,所以 Wave B 落地那天它会因为「意外地绿」而红。

## 四、剩余工作(按计划 §九 的顺序)

1. ~~**P1 适配器**~~ **已完成 2026-08-27**,见 `2026-08-27-wave-P1.md`。
   落点:e-RHINO `src/rheplicant/inference/graph_bridge.py`、
   `tests/seam/`(x64 子会话,19 passed + 1 xfail)、
   `tests/inference/test_graph_bridge.py`(36)、
   `tests/inference/test_refusal_census.py`(附录 B 的守卫)、
   bayesmith `.github/workflows/seam.yml`。**两条接缝变异实跑,双双击杀**
   (附录 A 第 3、4 行)。**三件事留给下一位**:(a) 接缝 CI 的两个 floor
   (`FLOOR`、`ADAPTER_FLOOR`)是本地实测的种子,按 runner 第一次绿跑打印的
   数字往上写;(b) `to_graph`/`translate` 要不要经门面公开,是 Wave A 前的
   一条新裁决(今天**没有**导出,门面 106 名单未动);(c) 例 5b 的
   `xfail(strict=True)` 是 Wave B 的闹钟,不要顺手删。
2. **P2 余项**:G1 掩码、G2 `bayesmith.fit`、G3 `exact.chain`、G4
   `exact.reduced_basis`、G5 `bayesmith.amortize`、G6 证据消费面、G7 bridge 补齐、
   **G9 全量**(vmap/log 空间/Fisher 的复数面;另**登记在案的两项**:`diagnose`
   仍拒绝复 latent、`exact.correct.log_weight` 仍在域里索引)、G10 分区执行面
   (三件全做,D14 已拍)、G12、G13、G14。
3. **P3 四波**:Wave A(检查与报告面)、Wave B(求解与计划)、Wave C、Wave D。
4. **P4** 质量机制换防(oracle 改籍、接缝 CI 与 crosscheck 双岗、
   `test_engine_room.py`、R3 编译计数门)。
5. **P5–P7**:18 个 run kind 冒烟、发布列车、P7 具名文档清单。

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

另外:**一个刚写完的守卫要先证明它还能失败**。本会话对每个新判据都跑了变异
(D16 四条轴、G9 六个变异、G11 的 AST 普查),每次都问「红的是不是**我的**断言」。

## 八、上下文用尽时怎么收尾(不要硬撑)

程序全长估计 27–40 个会话,一个会话跑不完。**不要为了「跑到结束」而把批次做
半截。** 当上下文接近用尽:

1. 把当前批次收到一个**四件套齐备**的状态(或干净回退到批次开始);
2. 两仓提交并推送,`ls-remote` 核实;
3. 更新本文件(它是交接页):改写 §二 当前状态、§三 已处置的 D 项、§四 剩余
   工作,让下一个会话能从这里直接开工;
4. 提交这次更新,然后停。

**没跑完不是失败;把一个半截批次留在工作树里才是。**
