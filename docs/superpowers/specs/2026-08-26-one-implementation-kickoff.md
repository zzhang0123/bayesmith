# Kickoff prompt — 全面移交程序的执行会话

> 本文件是交给执行会话的开工 prompt 的存档副本,与计划同目录、同被跟踪。
> 使用方式:把下面整段(含 owner 拍板行)粘贴给新 session。

---

你要执行「全面移交」程序:把 rheplicant 的全部贝叶斯数值实现移交给
bayesmith,rheplicant.inference 收缩为门面(声明层 + 噪声物理 + 适配器 +
探针层 + 薄包装)。**计划已定稿,不要重新评审、不要重新设计**——它经过
七视角对抗评审与三视角验证,每条结论都有源码取证。

## 第一步永远是读(按此顺序,不可跳)

1. **计划本体(本程序的宪法)**:
   `/Users/zzhang/projects/bayesmith/docs/superpowers/specs/2026-08-26-one-implementation.md`
   ——终局形态、七条铁律、裁决登记簿 D7–D19、缺口 G1–G14、四波切换、
   附录 A(接缝变异探针)/ B(文案清单)/ C(P0 提交清单)。
2. 前代 spec 的 §四(各模块绑定契约的台账)与 §六(本程序的由来):
   `/Users/zzhang/projects/bayesmith/docs/superpowers/specs/2026-08-24-rheplicant-migration.md`
3. 两仓的工作笔记(操作规则,条条是学费):
   `/Users/zzhang/projects/e-RHINO/CLAUDE.md`(与 AGENTS.md 逐字节一致,
   有测试钉着)与 `/Users/zzhang/projects/bayesmith/CLAUDE.md`。

## 当前状态(2026-08-26 会话结束时,先核实再信)

- **两个工作树都有大量未提交改动,程序的触发前提本身尚未落盘。**
  P0 因此是第一批工作,且顺序固定(铁律 5):bayesmith 提交推送 →
  `git tag v0.2.0 && git push origin v0.2.0`(publish.yml 门 tag==版本、
  测构建轮)→ **确认 0.2.0 上 PyPI** → 这之后才提交推送 e-RHINO
  (`bayesmith>=0.2` 的 floor 此刻才合规上 main)→ `git ls-remote`
  双仓核实(家规:测远端,不测本地记录)。提交范围严格按计划附录 C;
  e-RHINO 根目录的八份未跟踪评审/交接草稿**不动**,bayesmith 的未跟踪
  `AGENTS.md`(与其 CLAUDE.md 的「刻意无第二份」声明矛盾)去留问 owner。
- 环境:e-RHINO venv 里 bayesmith 是 editable(`uv pip install --python
  .venv/bin/python --no-deps -e ../bayesmith`);bayesmith venv 里
  rheplicant 同样 editable。两套测试:e-RHINO 用
  `.venv/bin/python -m pytest -n 4 --ignore=tests/gui/e2e`(共享机器上
  两阶段跑,见其 CLAUDE.md),bayesmith 用 `.venv/bin/python -m pytest
  -n 4`。**退出码写到自己的文件再读,只有 1 是测试失败**;计数从
  `--junit-xml` 取。
- 基线(开工先复测):e-RHINO 9962 passed / 502 skipped;bayesmith
  1205 passed;两边 ruff 干净;bayesmith 版本号已是 0.2.0。

## 裁决纪律(最重要的一条规矩)

计划里的 **D 项是 owner 的决定,不是你的**。每条已附建议与理由,但你
只在 owner 拍板后执行,并把裁决回填进计划该行(格式照 D1/B12 的先例:
`**【owner 已拍板 YYYY-MM-DD:…】**`)。到达一个未拍板的门就停下来问,
不要用「建议」自我授权。P1 开工前必须拍板的是 **D18**(复数域的家)与
**D19**(iterative_gls 退化起点);Wave B 前是 D7/D8/D14(+D17 若换探针)。

> **Owner 拍板行(由 owner 在交付 prompt 时填写)**:
> **【2026-08-26 交付时留空;执行会话开工即在门口停下并逐条问出,
> owner 当场拍板如下——D18 取 (a)(复数住 bayesmith);D19 取数据锚定
> 起步;bayesmith 的 `AGENTS.md` 删除、不入库;P0 发布动作一路做完
> 不中停。四条已回填计划本体的对应行。】**
>
> 留空本身值得记下来:**空行不等于「按建议执行」**。执行会话把它当作
> 未拍板处理、停下来问,是本程序的裁决纪律要求的动作——而不是自我授权
> 的例外。下一次交付若想让执行会话直接开跑,把这一行填掉。

## 本会话的工作量与节奏

- 一个会话 = 一个批次,批次完成 = 铁律 4 四件套齐备(分诊后测试全绿、
  接缝变异红——按 §六 五行协议、旧实现删除且计数守卫刷新、文档数字重测)。
- 首个会话做完 **P0**;有余力则在 D18 拍板后开 **P2a**(G9 最小面 + G11)。
- 每批证据链写进计划同目录的 tracked 执行页(`2026-XX-XX-wave-X.md`);
  **不要**把任何计划性文件放进 e-RHINO 的 docs/superpowers(gitignored,
  有八份计划死在那里的前科)。
- 需要并行时记住:子代理一律被拒绝 EnterWorktree,编排者自建 worktree +
  `git -C <绝对路径>` 可行;`cd` 不跨回合存活。

## 不要做的事

- 不重跑七视角评审;不改判据来迁就结果(判据改动 = 新裁决项)。
- 不动 partition.py/loglinear.py 的既有门面形态(它们已完成,D17 只裁
  探针的最终家)。
- 不让任何守卫静默 skip(会 skip 的守卫不是会通过的守卫);不在模块
  未到最后一波时往记录页写用例计数。
- 不把 bayesmith 独有能力(SNIS/discrete/circulant)经门面漏出。
