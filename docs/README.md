# bayesmith 文档索引

这一页回答「这份文档还算数吗」。每份文档在自己的开头声明 **文档状态**，本页只是把
它们汇总起来；两侧由 `tests/test_document_status.py` 双向校验，任何一边漂移都会红。
改动后用 `tools/sync_doc_index.py` 重新生成本页。

| 状态 | 含义 |
|---|---|
| `normative` | 唯一的顶层设计，冲突时以它为准 |
| `module-spec` | 已发布模块/能力的当前设计文档，从属于顶层设计 |
| `decision-home` | 某类决定的唯一登记处，仍在更新 |
| `plan-active` | 尚未执行完的计划，仍指导后续工作 |
| `record` | 已落地批次/审计/测量的历史记录，非当前权威 |
| `superseded` | 已被指名的后继文档取代 |

另外三处有自己的规矩，不在本索引内：`docs/migration/`（自带 README 与
`tests/test_migration_records.py`）、`docs/probes/` 与 `docs/derivations/`（可执行探针
与推导，不是文档页）。

## 先读这三份

1. `docs/superpowers/specs/2026-08-30-bayesmith-top-level-design.md` — 顶层设计，唯一 `normative`。
2. `CLAUDE.md`（= `AGENTS.md`，一份文件两个名字）— 本仓库的实测工作笔记：跑法、退出码语义、变异纪律。
3. `docs/ownership.md` — 哪些实现由 bayesmith 拥有，哪些应归上游。

## 全部文档

| 文档 | 状态 | 标题 |
|---|---|---|
| `docs/correlated-noise-proposal.md` | `record` | Declaring a correlated noise on a graph node |
| `docs/evidence-layer-readiness.md` | `record` | What B11 will find here |
| `docs/factor-partition-examples.md` | `module-spec` | From a model to an auto-partitioned sampler: two worked examples |
| `docs/ownership.md` | `decision-home` | Implementation ownership |
| `docs/superpowers/plans/2026-08-23-p1-graph-core-p2-numpyro-bridge.md` | `record` | bayesmith P1 图核 + P2 NumPyro 桥 — 实施计划 |
| `docs/superpowers/plans/2026-08-23-p3a-exact-core.md` | `record` | bayesmith P3a 精确解核心 — 实施计划 |
| `docs/superpowers/plans/2026-08-23-p3b-dispatch-execution.md` | `record` | bayesmith P3b 分派与执行 — 实施计划 |
| `docs/superpowers/plans/2026-08-23-p3b-task1-verdicts.md` | `record` | P3b Task 1 — linearity verdicts under three normalisations, measured before B1's fix |
| `docs/superpowers/plans/2026-08-29-coupling-collapse-ladder.md` | `record` | 计划：耦合度量、logdet 阶梯、图归约 |
| `docs/superpowers/plans/2026-08-30-audit-continuation.md` | `record` | 数值门审计修复 + Wave 2 执行（P5/P6/P7）—— 合并交接 |
| `docs/superpowers/plans/2026-08-30-r1-task-artifact-provenance.md` | `plan-active` | R1 Task、artifact 与 provenance 执行计划 |
| `docs/superpowers/plans/HANDOFF-p3b-tasks-4-10.md` | `record` | 交接：bayesmith P3b Tasks 4–10 —— **已全部完成（2026-08-24）** |
| `docs/superpowers/specs/2026-08-23-bayesmith-design.md` | `superseded` | bayesmith — 设计文档 |
| `docs/superpowers/specs/2026-08-23-p3-structural-dispatch-design.md` | `superseded` | bayesmith P3 — 结构分派器 + InferencePlan + 线性高斯精确解 |
| `docs/superpowers/specs/2026-08-23-p3b-dispatch-execution-design.md` | `module-spec` | bayesmith P3b — 分派器、InferencePlan、Gibbs 与近似块的修正 |
| `docs/superpowers/specs/2026-08-24-rheplicant-migration.md` | `decision-home` | bayesmith ← rheplicant 贝叶斯层迁移 — 规格与验收 |
| `docs/superpowers/specs/2026-08-26-one-implementation-kickoff.md` | `record` | Kickoff prompt — 全面移交程序的执行会话 |
| `docs/superpowers/specs/2026-08-26-one-implementation.md` | `record` | 全面移交:rheplicant 的贝叶斯层归于 bayesmith(一份实现计划) |
| `docs/superpowers/specs/2026-08-26-wave-P0.md` | `record` | 执行页 P0 — 落盘与发布列车 |
| `docs/superpowers/specs/2026-08-26-wave-P2a.md` | `record` | 执行页 P2a — G11 结构化拒绝载荷 + G9 复数域最小面 |
| `docs/superpowers/specs/2026-08-27-ci-flat-chain.md` | `record` | 执行页 — Seam CI 连红五次,原因是一条我写的断言把一台机器的运气当成了性质 |
| `docs/superpowers/specs/2026-08-27-d16-five-axes.md` | `record` | D16 — 五条轴,逐条实测,逐条已拍并落地 |
| `docs/superpowers/specs/2026-08-27-d17-protocol.md` | `record` | D17 裁决协议的执行结果 — 双跑 diff |
| `docs/superpowers/specs/2026-08-27-g15-local-block-priors.md` | `record` | 执行页 **G15** — 带先验的非线性局部块 |
| `docs/superpowers/specs/2026-08-27-g3-chain.md` | `record` | 执行页 **G3** — `evidence.chain`:把 linked 干扰精确积掉的那条递推 |
| `docs/superpowers/specs/2026-08-27-g4-reduced-basis.md` | `record` | 执行页 **G4** — `exact.reduced_basis`:选择与正交化 |
| `docs/superpowers/specs/2026-08-27-g6-enumeration.md` | `record` | 测量页 — **G6 的逐项登记**:证据消费面,一行一个判决 |
| `docs/superpowers/specs/2026-08-27-handover-run-to-completion.md` | `record` | 交接 prompt — 把「全面移交」程序跑到完全结束 |
| `docs/superpowers/specs/2026-08-27-numpyro-bridge-measurements.md` | `record` | 测量页 — 切 `numpyro_bridge` 之前必须先量的两件事(D26、D27) |
| `docs/superpowers/specs/2026-08-27-p2-g10-g12.md` | `record` | 执行页 P2 余项 · **G10 分区执行面完形** 与 **G12 冻结在当前值的 gcr** |
| `docs/superpowers/specs/2026-08-27-p2-g14-condition-estimate.md` | `record` | 执行页 P2 余项 · **G14 measured-κ 诊断**(D15(a)) |
| `docs/superpowers/specs/2026-08-27-p2-g2-fit.md` | `record` | 执行页 P2 余项 · **G2 `bayesmith.fit`** — 梯度 MAP,以及 D7 的那个出口 |
| `docs/superpowers/specs/2026-08-27-p2-g9-full.md` | `record` | 执行页 P2 余项 · **G9 全量**(复数域的其余四面),外加一个 0.4.0 就带着的缺陷 |
| `docs/superpowers/specs/2026-08-27-wave-A-g1-wiring.md` | `record` | 执行页 Wave A · G1 接线 — `FlaggedNoise` 作为声明的掩码过缝 |
| `docs/superpowers/specs/2026-08-27-wave-A-g13-wiring.md` | `record` | 执行页 Wave A · G13 接线 — 联合先验作为声明的 factor site 过缝 |
| `docs/superpowers/specs/2026-08-27-wave-A-identifiability.md` | `record` | 执行页 Wave A · 模块 1 — `identifiability` 切换 |
| `docs/superpowers/specs/2026-08-27-wave-A-numpyro-bridge.md` | `record` | 执行页 Wave A · 模块 4 — `numpyro_bridge` 的 `to_numpyro_model` 切换 |
| `docs/superpowers/specs/2026-08-27-wave-A-opening.md` | `record` | 执行页 Wave A(开工批)— 私名普查、契约阅读、分诊表,以及 D21 |
| `docs/superpowers/specs/2026-08-27-wave-A-priors.md` | `record` | 执行页 Wave A · 模块 3 — `priors`(`JeffreysPrior`)切换 |
| `docs/superpowers/specs/2026-08-27-wave-A-s6-widened.md` | `record` | 执行页 Wave A · 开放项 S6 结清 — `_widened` 是承重的,缺的是 fixture |
| `docs/superpowers/specs/2026-08-27-wave-A-sensitivity.md` | `record` | 执行页 Wave A · 模块 2 — `sensitivity` 切换 |
| `docs/superpowers/specs/2026-08-27-wave-A-uncertainty-covariance.md` | `record` | 执行页 Wave A · 模块 5 第二步 — `parameter_covariance` 与 `propagate_covariance` 切换 |
| `docs/superpowers/specs/2026-08-27-wave-A-uncertainty-fisher.md` | `record` | 执行页 Wave A · 模块 5 第一步 — `fisher_information` 的似然一半切换 |
| `docs/superpowers/specs/2026-08-27-wave-P1.md` | `record` | 执行页 P1 — 适配器、钉名十例、拒绝文案清单、接缝 CI |
| `docs/superpowers/specs/2026-08-27-wave-P2-D9.md` | `record` | 执行页 D9 — float32 政策:谱隙表跑完,建议不成立 |
| `docs/superpowers/specs/2026-08-27-wave-P2-G1.md` | `record` | 执行页 P2 余项 · G1 — 掩码/旗标贯通 exact/precision |
| `docs/superpowers/specs/2026-08-27-wave-P2-G13.md` | `record` | 执行页 P2 余项 · G13 — 图级联合先验 |
| `docs/superpowers/specs/2026-08-27-wave-P2-G7.md` | `record` | 执行页 P2 余项 · G7 — bridge 补齐 |
| `docs/superpowers/specs/2026-08-28-architecture-narrative.md` | `superseded` | bayesmith 架构叙事审计（2026-08-28 测量页） |
| `docs/superpowers/specs/2026-08-28-ci-triage.md` | `record` | 执行页 — CI 连红六条,逐条过判别法 |
| `docs/superpowers/specs/2026-08-28-d23-two-criteria.md` | `record` | 执行页 **D23** — 两条拒绝判据,以及一条已经在跑却没有守卫的语义差 |
| `docs/superpowers/specs/2026-08-28-g15-rheplicant-discharge.md` | `record` | 执行页 — G15 的 rheplicant 一半:解除那条有条件的延期 |
| `docs/superpowers/specs/2026-08-28-g5-amortize.md` | `record` | 执行页 **G5** — `bayesmith.amortize`:不写似然的推断 |
| `docs/superpowers/specs/2026-08-28-g6-consumption.md` | `record` | 执行页 **G6** — 证据消费面:一个 campaign 对自己说得出什么 |
| `docs/superpowers/specs/2026-08-28-multi-dataset-joint-posterior.md` | `record` | 多数据集联合后验：不同 graph、共享参数的累积推断 |
| `docs/superpowers/specs/2026-08-28-wave-B-b1.md` | `record` | 执行页 Wave B / B1 —— 本程序第一条真缺陷,和修它时照出的第二个格子 |
| `docs/superpowers/specs/2026-08-28-wave-B-gls-opening.md` | `record` | 执行页 Wave B / `gls` 开波 —— 一次契约误读,和量它的那三行 |
| `docs/superpowers/specs/2026-08-28-wave-B-linear.md` | `record` | 执行页 Wave B / `linear` 求解面 —— 十二条拒绝,一条会变成错答案 |
| `docs/superpowers/specs/2026-08-28-wave-B-opening.md` | `record` | 执行页 Wave B 开波 —— 普查、契约、以及契约本身的两处过期 |
| `docs/superpowers/specs/2026-08-28-wave-B-plan-opening.md` | `record` | 执行页 Wave B / `plan` + `engines` 开波 —— 这一行不是门面切换,原因可测 |
| `docs/superpowers/specs/2026-08-29-p1-coupling.md` | `module-spec` | P1 — 两块 latent 的局部耦合 |
| `docs/superpowers/specs/2026-08-29-p2-map-estimate.md` | `module-spec` | P2 — 图原生 MAP 与它自己的拒绝 |
| `docs/superpowers/specs/2026-08-29-p3-logdet-ladder.md` | `module-spec` | P3 — logdet 阶梯：从可证明的特例到有证书的近似 |
| `docs/superpowers/specs/2026-08-29-p4-graph-reduction.md` | `module-spec` | P4 — 图归约与第二个 graph-level factor 槽 |
| `docs/superpowers/specs/2026-08-29-wave-C-calibrate-opening.md` | `record` | 执行页 Wave C / `calibrate` 开波 —— 8 条拒绝里的第 8 条 |
| `docs/superpowers/specs/2026-08-29-wave-C-npe-opening.md` | `record` | 执行页 Wave C / `npe` 开波 —— 全缝逐位相同,而形状由 7 条异常类 pin 决定 |
| `docs/superpowers/specs/2026-08-29-wave-C-reduced-basis.md` | `record` | 执行页 Wave C / `reduced_basis` —— 半切,以及一次「找全了」错三遍 |
| `docs/superpowers/specs/2026-08-29-wave-D-opening.md` | `record` | 执行页 Wave D 开波 —— 八个模块,而「切」的其实很少 |
| `docs/superpowers/specs/2026-08-30-bayesmith-top-level-design.md` | `normative` | bayesmith 顶层设计：从结构化推断到可审计的 Bayesian workflow |
| `docs/superpowers/specs/2026-08-30-r0-close-out.md` | `record` | R0 close-out — 稳定现有核心与基线 |
| `docs/superpowers/specs/2026-08-31-p5-costs.md` | `module-spec` | P5 — 只读成本记分板（dispatch/costs.py） |
| `docs/superpowers/specs/2026-08-31-p6-collapse.md` | `module-spec` | P6 — collapsed target 与 collapse 臂（dispatch/collapse.py） |
| `docs/superpowers/specs/2026-08-31-p7-pilot-ledger.md` | `module-spec` | P7 — A4 pilot 与对账账本（dispatch/pilot.py、dispatch/costs.py） |
