# 交接：bayesmith P3b Tasks 4–10

仓库 `/Users/zzhang/projects/bayesmith`，用 `.venv/bin/python`。
分支 **`p3b-dispatch-execution`**（不要在 `main` 上做实现）。
导入测试 fixture 需要 `PYTHONPATH=/Users/zzhang/projects/bayesmith`。

## 起手：确认基线

```bash
git log --oneline -1 && git status --short
.venv/bin/python -m pytest -q | tail -1
.venv/bin/ruff check src/ tests/
```

三样都干净再往前走。**基线数字以实际输出为准**，不要照抄本文里的任何数字。

> Tasks 1–3b 全部落地后应为 **278 测试全绿、ruff 干净**，HEAD 是 `ec7e142` 之后。
> 唯一预期的未跟踪文件是 owner 的 `specs/2026-08-24-rheplicant-migration.md`（见下）。
>
> **若树意外不干净**：不要 `git checkout` / `git stash` / `git restore` —— 那会毁掉
> 未提交的工作，正是本文教训第 1 条讲的事故。先看 `git diff` 是什么，再决定。

## 先读这三处，按这个顺序

1. **`docs/superpowers/plans/2026-08-23-p3b-dispatch-execution.md`** —— 本计划。开头的「五条纪律」「四条子判据」「三条精度与排版纪律」「数字必须带参数化」**适用于每个任务**，不是背景。Task 2 开头有一段「Task 1 实测带来的四处更正」，Task 3 之前有一段「实测记录：两条变异未被杀死」——**那两段是执行期改掉/记下的东西，不是可跳过的注释。**
2. **`docs/superpowers/specs/2026-08-23-p3b-dispatch-execution-design.md`** —— 上游 spec。**先读它的 §〇「初稿被推翻的主张」**：这份 spec 的初稿经四路对抗性审查发现约 30 个缺陷，其中十条是初稿自己的实测错误，§〇 逐条记着。不读 §〇 就会把已经被推翻的说法当成设计。
3. **`docs/superpowers/plans/2026-08-23-p3b-task1-verdicts.md`** —— Task 1 的实测表（681 行）。Tasks 4–10 里任何涉及 `check_linearity` 判决的地方以它为准。

## 已完成（不要重做）

| 任务 | 提交 | 内容 |
|---|---|---|
| Task 1 | `f18536c` | 三种归一化下全部 fixture 的判决对比表（纯测量） |
| Task 2 | `2de46d1` | **B1**：`affinity_errors` 逐元素 + σ 加权，两判据共用逐元素舍入地板 |
| Task 2 后续 | `23e6ccd` | 观测节点 scale 不可用时具名报错，而不是诬告 `linear_in` |
| — | `b5f9509` | 记录两条杀不死的变异（见下「已知开口」） |
| Task 3 | `888cc8b` | **B2**：`check_prediction_dependence` 不再只走一条射线 |
| Task 3b | `ec7e142` | 改回 spec §1.5 的逐成员随机方向。`888cc8b` 在**三成员**块上仍失效（`a−c` 读 0.0）；现 `("uniform", "random")`，逐**元素**随机，`key` 默认 `jax.random.key(0)` |

## 还剩什么

**Tasks 4–10**，正文在计划里。抽取每个任务的完整文本：

```bash
python3 - << 'PY'
import pathlib
P = pathlib.Path("docs/superpowers/plans/2026-08-23-p3b-dispatch-execution.md")
lines = P.read_text().splitlines()
marks = [i for i, l in enumerate(lines)
         if l.startswith("## Task ") or l.startswith("## 验收（本计划完成的判据）")]
out = pathlib.Path("/tmp/p3b_tasks"); out.mkdir(exist_ok=True)
for n, (a, b) in enumerate(zip(marks, marks[1:]), start=1):
    (out / f"task{n}.txt").write_text("\n".join(lines[a:b]).rstrip() + "\n")
    print(n, lines[a][:70])
PY
```

依赖关系：

```
[已完成 1,2,3,3b]  →  Task 4 (probe_gaussian) ─┐
                                                ├→ Task 5 (classify) → Task 6 (InferencePlan)
                          Task 7 (correct.py) ──┤                             ↓
                                                └→ Task 8 (gibbs.py) → Task 9 (compile/sample/estimate)
                                                                              ↓
                                                                        Task 10 (统计验收)
```

**Task 7 与 Tasks 5/6 文件零重叠，可以并行**：Task 7 只碰
`src/bayesmith/exact/correct.py` 与 `tests/exact/test_correct.py`，Tasks 5/6 只碰
`src/bayesmith/dispatch/` 与 `tests/dispatch/`。

> subagent-driven 那条「不要并行派实现者」防的是**同一批文件上的写冲突**，不是
> 并行本身。判据是**文件集合是否相交**，不是任务编号是否相邻。本轮全程串行，其中
> Task 7 这一段是白等的。
>
> 但**审查者也算写者**——见下面第 2 条教训。派并行实现者时，两个都跑完再派审查。

## 流程

`superpowers:subagent-driven-development`：每个任务派一个**全新的** subagent，给它**完整的任务正文**（不要让它自己去读计划文件），任务之间做两道审查——**先 spec 合规，再代码质量**。审查发现问题 → 同一个实现者修 → 再审。

**model 选择**：纯机械、1–2 个文件、规格完整的任务可以用便宜模型；涉及数值判断或设计取舍的用默认模型。Tasks 5、8、10 属于后者。

## 这一轮用真实代价换来的教训——必须带走

### 1. 变异测试的恢复，绝不能读 git

这条已经害掉一次工作。一个 reviewer 做了七轮 mutate→restore，它的备份是在树干净时拍的，所以每次「恢复」写回的都是 **HEAD 的内容**，把此后产生的未提交改动全部抹掉。

> **规矩**：备份取自**开始那一刻文件的磁盘内容**，恢复只从那份备份读。**永远不要用 `git checkout` / `git stash` / `git restore` / 从 HEAD 复制来「恢复」**——git 只知道最后一次提交，而变异测试的使用场景恰恰是有未提交改动的时候。恢复后逐字节校验。

### 2. 不要让 reviewer 和你自己同时碰同一个文件

同上那次事故的另一半原因：我把 reviewer 当成只读的，同时自己在编辑同一个文件。**reviewer 要么被明确禁止写任何文件（让它把「需要变异才能验证」的项报告出来，由你事后跑），要么你在它跑完之前不碰那些文件。**

### 3. 写下变异行之前，先算它的功效

计划里 Task 2 的一条变异行（`jnp.any` 判决归约改成均值）**在它点名的 fixture 上不可能变红**——两列各自超阈值 1.2e3 倍和 2.6e12 倍，除以 6 一个都跨不过去。要杀死它需要 `n > 2.6e12`。这是「一个不可能的目标」的又一次重演：约束都是我自己写的，两行算术就能看出冲突，而我没算。

> **规矩**：每写一条「这个变异必须让 X 变红」，先估 X 的功效。做不到就换 fixture 或换目标，不要留一条只会绿的行。

### 4. 套件跑在 float32，且没有 `conftest.py`

`tests/` 里没有任何 `enable_x64`，整个仓库没有 `conftest.py`。所以守卫日常工作的 `rtol` 是 **1.19e-3**，不是 float64 的 2.22e-12。**四个 fixture 只因 dtype 就翻转判决**，其中 `cubic_tail(prior_std=1e-4)` 在 float64 下会被当前守卫拒绝，而一条测试断言它通过——**那条测试的绿色依赖于套件跑在 float32 上，而这件事没写在任何地方。**

需要 float64 时用 `with jax.enable_x64(True):`，且**图必须在 `with` 块内构造**（`const`/`observe` 在 `trace()` 时就 `jnp.asarray`）。**绝不调 `jax.config.update("jax_enable_x64", ...)`**——进程级全局，关不回去。

### 5. 「它在 N 上成立」不等于「它在 N+1 上成立」

Task 3 的修复在两成员块上完美，在**三成员**块上 `a−c` 读出 bitwise 0.0——同一个缺陷，一个成员之上。而**分派器把所有合格隐变量放进一个块**，所以 ≥3 成员是常态。

> **规矩**：任何按「成员数 / 叶子数 / 观测节点数」分支的代码，测试集合在那一维上至少取三个值，且必须包含一个**大于修复所针对的那个数**的值。

### 6. `sorted()` 是守卫还是文档，取决于 dict 从哪来

JAX 变换（`linearize`/`vjp`）出来的 dict **已经按键排序**，再 `sorted()` 是**可证的 no-op**，针对它的变异不可能被观测到。普通推导式建的 dict（`noise_std_at`、`observation_parts`）带**声明序**，`sorted()` **承重**。

另外注意：`block.names` 是**调用方给的顺序**，不是图的声明顺序（`_validated_names` 原样返回 `tuple(names)`）。所以要让针对它的 `sorted` 变异可见，要置换的是**调用方的列表**，不是图里 `observe()` 的声明顺序。

## 已知开口（已记录，**不要当成没发现**）

1. **Task 2 变异行 2**（逐元素 `jnp.any` 判决归约 → 均值）杀不死。原因与所需条件记在计划里 Task 3 之前那一节。
2. **Task 2 变异行 4**（`WEIGHTED_RTOL` ÷ 1e6）知情地不可达：float32 下每个诚实 fixture 的 above-floor 加权偏离**恰好是 0.0**，所以没有东西从下方约束这个常数；双侧性由地板测试承担，已在 docstring 里具名替代。

## 不属于本计划

- **`docs/superpowers/specs/2026-08-24-rheplicant-migration.md`**（313 行，未跟踪）是 owner 新增的下一阶段规格。**Tasks 4–10 全部完成之后**再考虑它。它的 §三「必须在 cross-check 之前修掉的缺陷」可能与本轮已修的 B1/B2 重叠——接手时先把 §三 的清单与 `2de46d1` / `23e6ccd` / `888cc8b` 对一遍，别重做。
- `ruff format --check` 在 9 个 P1/P2 时期的文件上仍有漂移。不属于本计划，单独一轮。
- `chain_method='vectorized'` 的多链在 numpyro 0.21.0 下坏掉（`HMCGibbs.init` 无条件 `random.split`）。Task 8 只需拒绝它并具名。

## 全部完成后

走 `superpowers:finishing-a-development-branch`，并派一个 reviewer 审整个 `main..HEAD` 的 diff，而不只是最后一个任务。
