# Cross-check records — and what §六 is still waiting for

Migration spec §二 requires one page per module and gates **all** of §六
(rheplicant's wind-down) on step 6: *"全部通过后，才动 rheplicant 侧对应模
块"*. This directory is where those pages live. It did not exist until
2026-08-25, which is why §六 had never started.

**Do not read the table below as authority.** It is prose, and prose has no
test. `tests/test_migration_records.py` is the authority: it
derives the module list from the spec's own §四 tables, the page list from
this directory, and the test list from `tests/crosscheck/`, and fails when
they disagree. If the table and that test ever disagree, the test is right.

| §四 row | module | cross-check test | page |
|---|---|---|---|
| 4.1 | `linear.py` → `exact/block,linearity,solve` | `test_linear.py` | ✅ |
| 4.1* | `conditioning.py` → `exact/conditioning` | `test_conditioning.py` | ✅ |
| 4.1 | `gls.py` → `exact/gls` | `test_noise_logdet.py` (B1 half) | ✅ |
| 4.1 | `uncertainty.py` (Fisher) → `exact/fisher` | `test_noise_logdet.py` | ✅ |
| 4.1 | `likelihood.py`/`noise.py` → `exact/gaussian` | `test_gaussian.py` | ✅ `noise.md` |
| 4.2 | `parameters.py` → node declarations | `test_parameters.py` | ✅ |
| 4.2 | `noise.py` → probabilistic nodes | `test_gaussian.py` | ✅ `noise.md` |
| 4.2 | `plan.py`+`engines.py` → dispatch | `test_dispatch.py` | ✅ `plan.md` |
| 4.2 | `identifiability.py` → `diagnose/` | `test_diagnose_identifiability.py` | ✅ |
| 4.2 | `sensitivity.py` → `diagnose/` | `test_diagnose_sensitivity.py` | ✅ |
| 4.2 | `priors.py` → `diagnose/` | `test_diagnose_jeffreys.py` | ✅ |
| 4.2 | `numpyro_bridge.py` → `bridge/` | — | — |
| 4.3* | `sqrtinfo` (rewritten; kernel preserved per B11) | `test_sqrtinfo_agrees.py` | ✅ |
| 4.3* | `calibrate.py` → `optimize` (**switched** per D11; 4.3 superseded) | — (retired: never existed) | ✅ `calibrate.md` |

`*` — has a page but **no source row of its own** in §四, and the test
records why. `calibrate.py` is the third such entry and the first to reach
the table through the SWITCHED branch rather than by naming a cross-check
test: a switched module is asserted to have *no* cross-check, since one would
compare this package with itself. §4.3 listed it under 不迁移; that entry was
superseded by the owner's 2026-08-26 ruling and by D11, and **D58 records what
reading the stale entry as live authority cost**. `conditioning.py` appears only in the `linear.py` row's
DESTINATION cell (upstream moved it to `rheplicant.core` so `radio` could
use it without importing `inference`); `sqrtinfo` belongs to the evidence
layer, which §四 4.3 marks 不迁移 — rewritten under iron law 2, with its
numerical kernel required to be preserved exactly and a bitwise
cross-check enforcing it.

## Where the gate actually stands

**Open, since 2026-08-25.** Twelve pages exist and **every §四 module has
one**. `tests/test_migration_records.py::test_the_gate_on_section_six_is_open_and_every_module_has_a_page`
now asserts that in the other direction: while the list was non-empty it
blocked §六, and now that §六's steps are being taken against these pages,
a page may not go missing.

| §四 row | closed |
|---|---|
| `linear.py` → `exact/{block,linearity,solve}` | `linear.md` |
| `likelihood.py`/`noise.py` → `exact/gaussian` | `noise.md` |
| `gls.py`, `uncertainty.py` (Fisher) | `gls.md`, `uncertainty.md` |
| `parameters.py` → node declarations | `parameters.md` |
| `noise.py` → probabilistic nodes | `noise.md` |
| `plan.py`+`engines.py` → dispatch | `plan.md` |
| `identifiability.py`, `sensitivity.py`, `priors.py` → `diagnose/` | three pages |
| `numpyro_bridge.py` → `bridge/` | `numpyro_bridge.md` |
| *(out of ledger)* `conditioning.py`, `sqrtinfo` | two pages, with their reasons |

**What that does and does not authorise.** §六 step 1 still governs:
nothing in `src/rheplicant/inference/` moves except the two exceptions
already in e-RHINO's Track A Batch 1 (B1's `plan.py` docstring and B4's
one-line fix), plus docstring pointers to bayesmith. Read §六's five steps
before starting, and note step 3's possible dividend — the two pytest
sessions may be able to merge once the evidence layer is out.

## What a page must contain

The five headings §二 names, in its order. The P5 pages are the worked
examples.

**One thing NOT to write: a case count, unless the module is finished.**
Some pages here carry one and some do not, and the asymmetry is a
measurement rather than an oversight. The P5 pages' counts (17, 17, 10)
were written in session 3 and are still exact, because nothing touched
those modules afterwards. `plan.md`'s said 6 and was 7 within hours,
because its author kept working on the module after writing the page. So
the rule is about timing, not taste: a count is a safe thing to record only
once the thing counted has stopped moving, and the four pages written
during active work say `pytest --collect-only -q` instead. Nothing in this
repository reads a count out of a page, so none of them can go red.

The five headings:

1. **Fixtures** — reused from rheplicant where possible (they are pinned
   measurements, not fresh guesses), at least one healthy and one that must
   be refused.
2. **Numerical agreement** — deterministic exits to float64 roundoff,
   sampled exits within MC error. Say which tolerance and why; where
   exactness is not mathematically available (a null direction is a ray, a
   null space is basis-dependent), say what is compared instead.
3. **Refusal agreement** — every loud rheplicant refusal has a same-shape
   refusal here, with the exception-class mapping recorded.
4. **Independent oracle** (iron law 4) — analytic truth, a hand-written
   NumPyro model, scipy, or mutation. *Two implementations agreeing is not
   evidence.*
5. **Intended differences** — with the reason and the equivalence argument.
   A difference that is deliberate and unrecorded is indistinguishable from
   a bug six months later.

Iron law 5 adds one hard item: any §三 defect belonging to the module is
fixed **before** step 2, or written as a signed, sized expected difference.
