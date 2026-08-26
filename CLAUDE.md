# Working notes for coding agents

Repo-specific facts that are expensive to rediscover. Everything here was
measured in this checkout on 2026-08-26, not assumed or carried over.

**There is deliberately no `AGENTS.md` beside this file.** The sibling
repository (e-RHINO) keeps both and holds them byte-identical with a test. No
such test exists here, and an unenforced second copy is the defect this
project has spent the most time repairing — six copies of one measurement went
stale on a day none of them was edited. One file, or a test; not two files and
a hope.

## Running the tests

```bash
.venv/bin/python -m pytest -n 4 > run.log 2>&1; echo "PYTEST_EXIT=$?" > run.exit
cat run.exit
```

Measured: **1127 passed, 0 skipped, 0 failed** in about 112 s at `-n 4`.

**Do not add `-q`.** `pyproject.toml` already carries `addopts = "-q"`, so a
second one makes it `-qq` and **the summary line disappears entirely** — no
count, nothing to read but the exit code. This is not hypothetical: a commit
message in this repository asserted "1127 passed" from a run whose log had no
such line. The number happened to be right, which is the bad outcome, because
nothing in the run could have said otherwise. Take counts from
`--junit-xml`, and the verdict from the exit code written to its own file.

Only exit **1** means a test failed. **2** interrupted, **3** internal error,
**4** usage error, **5** nothing collected, **143** killed. That distinction is
load-bearing in mutation testing, where scoring any non-zero as a kill turns a
typo into a KILLED for every mutant.

## The test subject moves when the sibling checkout does

The cross-checks under `tests/crosscheck/` import `rheplicant` from an
**editable install**, so they test whatever e-RHINO currently has checked out.
Switching branches over there silently changes what passes here. Re-run this
suite after any e-RHINO checkout change, and before pushing either repo.

The importable module of the second editable install is **`rhino_cal_jax`**,
not `rhino_cal`; checking the wrong name reads exactly like "never installed".

## Linting

`ruff check src/ tests/` is clean. `ruff format` reports **31 files / 517
lines** of drift, left there on purpose: nothing enforces it (no CI, no
pre-commit), and `c2a0605` shows formatting here has been applied per file
behind a waiver rather than swept. If it is ever swept, **pass the 31 file
names, not `src/ tests/`** — `ruff format --check` prints them.

## Two habits this repository rewards

**A decision's answer belongs on the line that asks the question.** Cross-repo
decisions live in `docs/superpowers/specs/2026-08-24-rheplicant-migration.md`
§七 under a stated rule that a decision has one home. That rule has a hole:
it says where a decision lives and nothing about where its *resolution* lives.
D2 was resolved as malformed in the OTHER repository, so the home kept a dead
question for a day and the next reader ruled on it again. Put the resolution
back on the question's line.

**Read the source before believing a ledger row.** A8.2 was marked as needing
a decision before it could start. It was already shipped. The row for it in
the other repo now says so, and says the same thing happened to a row called
A3. When a table says "not done", check.
