# Working notes for coding agents

Repo-specific facts that are expensive to rediscover. Everything here was
measured in this checkout on 2026-08-26, not assumed or carried over.

**There is deliberately no `AGENTS.md` beside this file.** The sibling
repository (e-RHINO) keeps both and holds them byte-identical with a test. No
such test exists here, and an unenforced second copy is the defect this
project has spent the most time repairing — six copies of one measurement went
stale on a day none of them was edited. One file, or a test; not two files and
a hope.

One did appear, byte-identical to this file, and made the sentence above false
by existing. The owner ruled on 2026-08-26 that it be deleted rather than
committed. If a tool here needs `AGENTS.md`, the ruling to revisit is that one
-- and the price of keeping it is the identity test, not a good intention.

## Running the tests

```bash
.venv/bin/python -m pytest -n 4 > run.log 2>&1; echo "PYTEST_EXIT=$?" > run.exit
cat run.exit
```

Measured: **1254 passed, 0 skipped, 0 failed** in about 171 s at `-n 4`.

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

## A zsh glob can turn "it ran and found nothing" into "it never ran"

`ls LICENSE* COPYING* 2>/dev/null || echo "(none)"` prints `(none)` **whenever
either pattern fails to match**, because zsh's default `nomatch` aborts the
whole command rather than passing the pattern through as bash would. The `||`
branch then fires and reads exactly like a finding.

Measured, and it cost something: a release audit reported that this repository
had no LICENSE file. It has had one since `ba8c7b5`. The file was then
overwritten with byte-identical text, so nothing was lost -- by luck, not by
care. A differently worded licence would have been destroyed by a check that
was reporting on a command that never ran.

Same shape as the two traps above: a result that cannot distinguish "absent"
from "the command did not happen". Glob one pattern per `ls`, or use `find`,
which has no opinion about patterns that match nothing.

## On release day the index has three answers, and two are stale

A green `publish.yml` run is a record, not the index. Measured within one
minute of tagging v0.2.0, all three asked about the same package:

| asked | answered | what it actually is |
|---|---|---|
| `pypi.org/pypi/bayesmith/json` | only 0.1.0 | the JSON API's cache, minutes behind |
| `pypi.org/simple/bayesmith/` | 0.1.0 **and** 0.2.0 | the table pip resolves against |
| `uv pip install 'bayesmith>=0.2'` | "unsatisfiable" | uv's LOCAL index cache |

Two of the three said the release had not happened. It had -- the upload step
had already logged both files. `--refresh` made uv install 0.2.0 immediately,
and the JSON API caught up on its own.

Same family as the zsh glob above: a result that cannot distinguish "absent"
from "this lookup did not really happen". The cost here is worse than a wrong
audit line, because the obvious reaction to "the release did not appear" is
to re-push the tag -- and a tag that has already published is immutable, so
that road only adds damage.

**Ask `/simple/`, or resolve once with `--refresh`.** The strongest check is
the one a consumer performs: install the floor into a throwaway venv, then
look inside the installed package for the module the floor exists for. A
source-tree run passes happily when packaging has excluded a file, which is
why `publish.yml` tests the built wheel and why this check reads the wheel's
files rather than the repository's.

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
