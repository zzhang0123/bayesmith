# Cross-check records — and what §六 is still waiting for

Migration spec §二 requires one page per module and gates **all** of §六
(rheplicant's wind-down) on step 6: *"全部通过后，才动 rheplicant 侧对应模
块"*. This directory is where those pages live. It did not exist until
2026-08-25, which is why §六 had never started.

**Do not read the table below as authority.** It is prose, and prose has no
test. `tests/crosscheck/test_migration_records.py` is the authority: it
derives the module list from the spec's own §四 tables, the page list from
this directory, and the test list from `tests/crosscheck/`, and fails when
they disagree. If the table and that test ever disagree, the test is right.

| §四 row | module | cross-check test | page |
|---|---|---|---|
| 4.1 | `linear.py` → `exact/block,linearity,solve` | — | — |
| 4.1* | `conditioning.py` → `exact/conditioning` | `test_conditioning.py` | ✅ |
| 4.1 | `gls.py` → `exact/gls` | `test_noise_logdet.py` (B1 half) | ✅ |
| 4.1 | `uncertainty.py` (Fisher) → `exact/fisher` | `test_noise_logdet.py` | ✅ |
| 4.1 | `likelihood.py`/`noise.py` → `exact/gaussian` | — | — |
| 4.2 | `parameters.py` → node declarations | — | — |
| 4.2 | `noise.py` → probabilistic nodes | — | — |
| 4.2 | `plan.py`+`engines.py` → dispatch | — | — |
| 4.2 | `identifiability.py` → `diagnose/` | `test_diagnose_identifiability.py` | ✅ |
| 4.2 | `sensitivity.py` → `diagnose/` | `test_diagnose_sensitivity.py` | ✅ |
| 4.2 | `priors.py` → `diagnose/` | `test_diagnose_jeffreys.py` | ✅ |
| 4.2 | `numpyro_bridge.py` → `bridge/` | — | — |
| 4.3* | `sqrtinfo` (rewritten; kernel preserved per B11) | `test_sqrtinfo_agrees.py` | ✅ |

`*` — has a page but **no source row of its own** in §四, and the test
records why. `conditioning.py` appears only in the `linear.py` row's
DESTINATION cell (upstream moved it to `rheplicant.core` so `radio` could
use it without importing `inference`); `sqrtinfo` belongs to the evidence
layer, which §四 4.3 marks 不迁移 — rewritten under iron law 2, with its
numerical kernel required to be preserved exactly and a bitwise
cross-check enforcing it.

## Where the gate actually stands

Seven pages exist; **five §四 rows have neither a cross-check test nor a
page**, and they are the blocker:

- `linear.py` → `exact/{block,linearity,solve}` (§四 4.1)
- `likelihood.py`/`noise.py` → `exact/gaussian` (§四 4.1)
- `parameters.py` → node declarations (§四 4.2)
- `noise.py` → probabilistic nodes (§四 4.2) — **ordering**: §五 B9 changes
  this interface, so settle the interface before finalising the module
- `plan.py`+`engines.py` → dispatch (§四 4.2) — **B1 first**, or the
  comparison fixes the GLS-type target as the reference
- `numpyro_bridge.py` → `bridge/` (§四 4.2)

Those are measurement jobs, not writing jobs: each needs fixtures, an
independent oracle and a refusal comparison before its page can exist.

So §六 is still blocked, on a list rather than a category. Nothing in
`src/rheplicant/inference/` may move until that list is empty — the two
exceptions §六 step 1 names are already in e-RHINO's Track A Batch 1.

## What a page must contain

The five headings §二 names, in its order. The P5 pages are the worked
examples:

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
