# Working notes for coding agents

Repo-specific facts that are expensive to rediscover. Everything here was
measured in this checkout on 2026-08-26, not assumed or carried over.

**`AGENTS.md` is a byte-identical copy of this file, and a test holds it
there.** That is a reversal of the 2026-08-26 ruling, and the reason it was
reversed is the failure the ruling was meant to prevent, arriving anyway.

The old ruling said: no second copy, because an unenforced second copy is the
defect this project has spent the most time repairing — six copies of one
measurement went stale on a day none of them was edited. One file, or a test;
not two files and a hope. It also named the exit: *if a tool here needs
`AGENTS.md`, the ruling to revisit is that one, and the price of keeping it is
the identity test.*

Measured on 2026-08-31: an untracked `AGENTS.md` was sitting in this checkout,
dated 2026-08-27, **not** identical to this file. It still printed the
pre-layering `1280 passed` recipe and knew nothing about the fast/full split.
An agent session read it as workspace instructions — because the tools that
look for a repository's working notes look for `AGENTS.md` first. A file that
is deleted rather than committed is a file nothing checks, so it came back
stale and nobody could have noticed.

So the price is now paid rather than avoided: both files are tracked,
`tests/test_agent_notes_are_one_file.py` compares them byte for byte, and
editing either one alone turns the suite red.

**Where the documents are.** `docs/README.md` is the index, and every page
under `docs/` declares its own **文档状态** — `normative` (exactly one: the
top-level design), `module-spec`, `decision-home`, `plan-active`, `record`,
`superseded`. `tests/test_document_status.py` checks the pages and the index
against each other in both directions, so a new document with no status, or an
index row pointing at nothing, fails rather than joining the pile. Before
citing a page, read its status line: fifty-seven of the seventy-three are
`record`, true of the day they were written and authoritative over nothing.

## Running the tests

The suite is split into a **fast layer** (the pre-commit habit) and a **full
layer** (nightly). The heavy numerical-gate boundary and mutation grids are
marked `full`; everything else is fast.

Fast layer:

```bash
.venv/bin/python -m pytest -n 4 -m "not full"
```

Full layer (nightly — everything, including the `full` grids):

```bash
.venv/bin/python -m pytest -n 4
```

A meta-test in `tests/numerical_gates/test_boundary_layering.py` fails if any
registered gate loses its one fast-layer cell, so "fast" cannot silently
collapse to "no numerical-gate coverage".

Test artifacts go in one directory per run — three products from one
invocation, all in the same directory (add `-m "not full"` for the fast
layer):

```bash
RUN=$(date +%Y%m%dT%H%M%S)-$$; D=runs/$RUN; mkdir -p "$D"
.venv/bin/python -m pytest -n 4 --junit-xml="$D/junit.xml" > "$D/log" 2>&1
echo "PYTEST_EXIT=$?" > "$D/exit"
```

`runs/` is gitignored.

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

**Commit the batch before you mutate it.** The protocol restores with
`git checkout -- src/` rather than `cp`, and that is right -- `cp` inside the
same second leaves bytecode Python reuses, which records a false SURVIVED. But
`git checkout` restores to HEAD, so on a tree carrying uncommitted work it is a
silent full revert of that work, not of the mutant. Measured on 2026-08-27: a
mutation run was killed on a timeout, leaving a mutant in the tree; the
`git checkout -- src/` that followed took the whole unfinished G1 change with
it. Only `tests/` survived, because the mutants were all under `src/`. Commit
first, then mutate, then restore -- `git checkout` is the better tool exactly
because HEAD is the reference, which means HEAD has to be the thing you want
back.

**Rule (0) has a second half, and it is the half that bites twice.** "Commit
the batch before you mutate it" is not "commit once when the batch starts" --
it is **HEAD has to be what you want back, every time you run the set**. The
protocol's `git checkout -- src/ tests/` restores to HEAD, so a fix written
*after* the first mutation run and *before* the second is reverted by that
second run's own opening restore, silently and before any mutant is applied.
Measured 2026-08-27: two survivors were diagnosed correctly, the guards were
repaired, the set was re-run to confirm -- and it reported the same two
survivors, because the repair no longer existed. Nothing in the output says
so; a reverted fix and a fix that did not work look identical.

Commit the repair, then re-run. And if a mutation script restores paths beyond
the mutants' own, narrow it: that one restored all of `tests/` to undo mutants
that were only ever in `src/`.

`rm -rf __pycache__` between the mutation and the restore stays required
either way.

## The test subject moves when the sibling checkout does

The cross-checks under `tests/crosscheck/` import `rheplicant` from an
**editable install**, so they test whatever e-RHINO currently has checked out.
Switching branches over there silently changes what passes here. Re-run this
suite after any e-RHINO checkout change, and before pushing either repo.

The importable module of the second editable install is **`rhino_cal_jax`**,
not `rhino_cal`; checking the wrong name reads exactly like "never installed".

## Linting

**Pass `--no-cache`, or the check can report on a run it did not make.**
Measured 2026-09-03: `ruff check src/ tests/` printed `All checks passed!` in
this checkout while the identical command, same binary, in a fresh worktree of
the same commit found an `I001` in `tests/dispatch/test_predictive_seam.py`.
The difference was `.ruff_cache/`, which is gitignored and therefore exists
only where someone has run ruff before. `ruff check --no-cache src/ tests/`
finds it in both. Three separate agents reported the error and this checkout
denied it, which is how long a stale cache can hold a lie.

Same family as the zsh glob and the PyPI index below: a result that cannot
distinguish "clean" from "the check did not really run". And take the exit code
from ruff itself -- `ruff check ... | tail -3; echo $?` reports on `tail`.

`ruff check --no-cache src/ tests/` is clean. `ruff format` reports **31 files / 517
lines** of drift, left there on purpose: nothing enforces it (no CI, no
pre-commit), and `c2a0605` shows formatting here has been applied per file
behind a waiver rather than swept. If it is ever swept, **pass the 31 file
names, not `src/ tests/`** — `ruff format --check` prints them.

## A numerical fixture that pins one machine's arithmetic will burn a release tag

It burned four. `v0.6.0`, `v0.6.1`, `v0.6.2` and `v0.7.0` were tagged and none
reached the index, because until 2026-09-02 the only place this suite ever met
Linux was `publish.yml` — which runs *after* the tag is pushed, and a PyPI
version number is spent once used. `v0.7.0` failed there with 16 failures and
2091 errors while the same wheel passed 5375/5375 on the development laptop.
`suite.yml` now runs the suite on ubuntu before any tag exists; that is the
mechanism, and the rest of this section is the discipline it enforces.

**None of the sixteen was a typo.** Every one was a fixture that had written
down what one machine's arithmetic happened to produce and called it a
property. macOS numpy uses **Accelerate**; Linux wheels use **scipy-openblas**;
the versions are otherwise identical. What differed:

* **Layout invariance.** Accelerate returns C-order and F-order products of the
  same factors one ULP apart; OpenBLAS returns them bitwise equal, and
  separates none of 96 swept shapes. `_matching_factor_reconstruction`'s
  cross-layout fallback is therefore unreachable code on Linux.
* **Subnormal resolution.** A singular value at the bottom of the subnormal
  range can only be reported as a whole multiple of `2**-1074`. Accelerate said
  two units, OpenBLAS said one — opposite sides of the `1/eps` ceiling for a
  matrix whose exact condition is knowable and sits below it.
* **`dpotrf` on a numerically indefinite matrix.** The same bits, the same
  `eigvalsh` verdict, and one LAPACK factorises while the other refuses.
* **FMA contraction.** OpenBLAS picks its dgemm microkernel from the CPU at
  runtime. At condition 3.6e15 one fused multiply-add moved an exact-Decimal
  logdet by 0.66 nats, against an assertion pinning it to `2e-15`.
* **A scale-blind absolute tolerance**, which is not a platform fact at all:
  `atol=2e-17` over fixtures that deliberately scale the factors by 16x. It
  had simply never been exercised, because on Accelerate the two sides agreed
  exactly.

**The rule, in the order to try it.** (a) Construct the stress deterministically
— hex floats, exact integers, `Fraction`/`Decimal`, an explicit ULP nudge — so
it holds everywhere. (b) Failing that, assert the PROPERTY the fixture needs
with a band whose FORM is derived, and say which part is derived and which is
measured; do not dress the second as the first. (c) Failing that, make the
platform-dependent PREMISE conditional and recorded while keeping the CONTRACT
assertions unconditional and ahead of it. (d) Failing that, skip loudly — `THIS
IS NOT A PASS`, naming the measurement — rather than passing quietly.

**And do not widen a tolerance to get to green.** Measured, twice, in this
repair: a two-sided band around an exact value ADMITS `[1.0, exact + band]`
where `<= 1.0` refused, and two eigensolve-bias mutants that died before
survived it; and a fixture change that fixed a platform failure silently
released a ceiling mutant the old one killed. Both were caught only because
each repair was required to carry a mutation table that a second reader re-ran.
If a repair cannot show what it still kills, it is not finished.

**Reproducing CI locally.** A `linux/amd64` container resolves the same numpy,
scipy, jax and scipy-openblas as the runner, but QEMU's CPU is unrecognised so
OpenBLAS falls back to a non-FMA kernel. **Set `OPENBLAS_CORETYPE=ZEN`** and it
reproduces one runner bit for bit — 14 of 16 failures without it, 16 of 16
with it.

**But there is no such thing as "the runner".** GitHub allocates a fresh VM per
job and the `ubuntu-latest` pool is heterogeneous: measured across three runs
it served an AMD EPYC 7763 (`avx avx2 fma sse4_2`, no AVX-512), an AMD EPYC
9V74 and an Intel Xeon Platinum 8370C (both with the AVX-512 family). Two jobs
of ONE workflow ran on the last two and a third run put the same job on the
first. So a boundary cell can pass in one job and fail in the other on the same
commit — that happened, and it is what identified the sixteenth repair. Do not
tune a fixture to a CPU; make it insensitive to one. `suite.yml` logs the CPU
and the BLAS in both jobs, because when a cell moves that is the first
question. Mind that **zsh does not word-split unquoted
variables**, so a `-e VAR=x` held in a shell variable reaches `docker` as one
argument and is silently ignored — that alone produced a confident and wrong
"the coretype makes no difference".

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
