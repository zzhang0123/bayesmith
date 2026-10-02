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

另外三处有自己的规矩，不在本索引内：`docs/migration/`（自带 README，历史记录）、`docs/probes/` 与 `docs/derivations/`（可执行探针
与推导，不是文档页）。`docs/superpowers/` 是本地目录，不随仓库发布，也不在本索引内。

## 先读这三份

1. `docs/design.md` — 顶层设计，唯一 `normative`。
2. `CLAUDE.md`（= `AGENTS.md`，一份文件两个名字）— 本仓库的实测工作笔记：跑法、退出码语义、变异纪律。
3. `docs/ownership.md` — 哪些实现由 bayesmith 拥有，哪些应归上游。

## 全部文档

| 文档 | 状态 | 标题 |
|---|---|---|
| `docs/artifacts.md` | `module-spec` | The artifact protocol: Tasks, Results, provenance and gates |
| `docs/automatic-affinity.md` | `module-spec` | Automatic affine discovery and conditional blocks |
| `docs/campbell-sky-validation.md` | `record` | Campbell sky validation: GCR reweighting and analytic cumulants |
| `docs/correlated-noise-proposal.md` | `record` | Declaring a correlated noise on a graph node |
| `docs/cumulant-expansion.md` | `module-spec` | Cumulant likelihoods for array-valued fields |
| `docs/design.md` | `normative` | bayesmith 顶层设计：从结构化推断到可审计的 Bayesian workflow |
| `docs/evaluation.md` | `module-spec` | Model checking: eight report kinds, two axes, and what a PASS does not mean |
| `docs/evidence-layer-readiness.md` | `record` | What B11 will find here |
| `docs/evidence.md` | `module-spec` | The evidence layer: one structure class, five terms, and what a PASS does not mean |
| `docs/factor-partition-examples.md` | `module-spec` | From a model to an auto-partitioned sampler: two worked examples |
| `docs/mutation/2026-09-04-r5-wave-a.md` | `record` | R5 Wave A mutation table |
| `docs/mutation/2026-09-05-r5-wave-b.md` | `record` | R5 Wave B mutation table |
| `docs/mutation/2026-09-05-r5-wave-c.md` | `record` | R5 Wave C mutation table |
| `docs/mutation/2026-09-05-r5-wave-d.md` | `record` | R5 Wave D mutation table |
| `docs/ownership.md` | `decision-home` | Implementation ownership |
| `docs/reweight.md` | `module-spec` | Gaussian reference sampling and non-Gaussian hyperparameter MAP |
| `docs/stability.md` | `module-spec` | The 0.9 stable baseline |
| `docs/stabilization-090.md` | `record` | 0.9.0 stabilization review — T-002 |
