#!/usr/bin/env python3
"""Regenerate `docs/README.md` from the status lines the pages declare.

The status is written by hand, in the page, once -- declaring what authority a
document has is a deliberate act and should look like one. This script only
collects those declarations and rewrites the index table, so the two can never
be edited into disagreement by accident.

    PYTHONPATH=. .venv/bin/python tools/sync_doc_index.py

`tests/test_document_status.py` checks the result in both directions; running
this is how you make that test green after adding, renaming or reclassifying a
page.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
INDEX = DOCS / "README.md"

# Kept identical to tests/test_document_status.py on purpose: this script and
# that test are the two halves of one convention, and the test is the one that
# fails loudly if they drift.
EXEMPT_DIRS = ("migration", "probes", "derivations")
STATUS = re.compile(r"^> \*\*文档状态：`(?P<status>[a-z-]+)`\*\*", re.MULTILINE)
H1 = re.compile(r"^# (.+)$", re.MULTILINE)

HEADER = """# bayesmith 文档索引

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
"""


def pages() -> list[pathlib.Path]:
    return sorted(
        path
        for path in DOCS.rglob("*.md")
        if path != INDEX
        and not any(part in EXEMPT_DIRS for part in path.relative_to(DOCS).parts)
    )


def main() -> int:
    rows = []
    unstamped = []
    for path in pages():
        text = path.read_text(encoding="utf-8")
        found = STATUS.findall(text)
        if len(found) != 1:
            unstamped.append((path.relative_to(ROOT).as_posix(), len(found)))
            continue
        title = H1.search(text)
        rows.append(
            (
                path.relative_to(ROOT).as_posix(),
                found[0],
                title.group(1).strip() if title else path.stem,
            )
        )

    if unstamped:
        for relative, count in unstamped:
            print(
                f"{relative}: {count} status lines, expected 1. Add one, e.g.\n"
                "  > **文档状态：`record`** · 已落地批次/审计/测量的历史记录，"
                "写作当天为真，非当前权威。索引见 docs/README.md。",
                file=sys.stderr,
            )
        return 1

    body = "".join(f"| `{p}` | `{s}` | {t} |\n" for p, s, t in rows)
    INDEX.write_text(HEADER + body, encoding="utf-8")
    print(f"docs/README.md: {len(rows)} pages indexed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
