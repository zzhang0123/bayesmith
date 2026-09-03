# Cross-check: `sqrtinfo`

`rheplicant.inference.sqrtinfo` → `bayesmith.marginal.sqrtinfo` (spelled
`bayesmith.evidence.sqrtinfo` when this page was written; that name is a
deprecated alias since 0.5.0, retiring at 1.0). Test:
`tests/crosscheck/test_sqrtinfo_agrees.py` — 6 cases since 2026-08-30,
10 before D90 retired `marginalise_arrays`'s four. Page written 2026-08-25
from that test's assertions, re-run on that date; the cross-check itself
predates this page. **Updated 2026-08-30 for D90** (migration spec §五
B11): the Schur-complement kernel is single-implementation since rheplicant
`b87e44f`, and `tests/crosscheck/test_provenance.py` now pins which side
each remaining subject is on.

Not a §四 row — the whole streaming-evidence layer is §四 4.3 "不迁移",
generalised and rewritten here per iron law 2 (B11). The numerical KERNEL
is nevertheless required to be preserved exactly (§五 B11: "必须原样保留
的数值内核"), and that is what this compares.

## 1. Fixtures

Random `[R|z]` factors at fixed seeds, in `float64` throughout — the
evidence layer's own precision, and a float32 comparison would agree to a
tolerance that hides a real difference in the constant.

- `combine`: two factors of different row counts (4×3, 5×3), the second
  with a non-zero offset.
- `null`: two latents of different shapes, `((2,), ())`.
- `marginalise_arrays`: ~~a 9×5 factor swept over `n_block ∈ {0, 1, 2, 5}`~~
  — retired 2026-08-30 (D90): their side is a wrapper over our kernel since
  rheplicant `b87e44f`, so these four cases compared this package with itself.
- `marginalise`: the checked path, with the prior scale swept over
  `{0.7, 1.0, 3.0}` — since D90 a SHELL comparison (their permutation,
  offset threading and pivot reading against ours, over the one shared
  kernel); the sweep stays because a non-unit prior drives a non-zero
  offset through both shells.

## 2. Numerical agreement

**Bitwise** (`rtol=0, atol=0`) on `factor`, `target`, `offset` and
`log_prob`, for every case above. Bitwise is the right bar here: it is the
same arithmetic in the same order on the same library, so a tolerance would
let a genuinely different fold pass — which is the thing this file exists
to notice.

**Why the prior scale is swept** was the load-bearing fixture decision: the
marginalisation constant rheplicant once shipped *missing* is exactly zero
at `std = 1`, so a cross-check run only at unit prior would agree with a
port that had dropped the same term. Since D90 that rationale is dead for
the KERNEL — both sides compute the constant in the same place, so a
dropped term would be dropped for both and stay green here. The constant is
pinned one-sided by `tests/marginal/test_sqrtinfo.py` and §五 B11's
nat-cost table; the sweep survives only as offset traffic through the two
shells.

## 3. Refusal agreement — deliberately NOT compared

bayesmith raises `StructureError` where rheplicant raises
`StateValidationError`; the two packages have their own error families and
the port is allowed to differ there on purpose. The refusals themselves are
owned by `tests/marginal/test_sqrtinfo.py` on this side (the `tests/
evidence/` spelling predates the 0.5.0 rename).

## 4. Independent oracle

The evidence layer's total oracle is `tests/marginal/test_streaming_equals_
batch.py` ("streaming == batch to roundoff") plus the per-constant nat cost
table (§五 B11 lists all five measured values), both on the bayesmith side.
What *this* file adds is the anti-vacuity guard:
`test_the_comparison_can_still_fail` perturbs one entry of our factor by
1e-9 and asserts the comparison notices — so a `_same` that stopped reading
a field, or a `_pair` that stopped building both sides, is caught here
rather than by nobody.

## 5. Intended differences

1. **Exception families** (above).
2. **The layer around the kernel is a rewrite, not a port** — the
   factorization is DERIVED from the graph's plates instead of declared
   (§五 B11), which eliminates the "same space declared twice" error class
   `factorize.py` existed to police. That rewrite is checked by its own
   oracles, not by comparison with rheplicant, precisely because iron law 4
   says two implementations agreeing is not evidence.
