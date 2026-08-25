# Cross-check: `conditioning`

`rheplicant.core.conditioning` (moved there from `inference/` upstream when
`radio` needed it and could not import `inference`) →
`bayesmith.exact.conditioning`. Test: `tests/crosscheck/test_conditioning.py`
(9 cases). Page written 2026-08-25 from that test's assertions, re-run on
that date; the cross-check itself predates this page.

## 1. Fixtures

A symmetric operator with an EXACTLY known spectrum, built as
`Q diag(λ) Qᵀ` from a QR of a fixed-seed normal matrix and handed to both
packages as the same callable — so a difference is the algorithm and never
the setup. Spectra: `[1,2,3]`, `[1e-6,1,5,5.5]`, six-fold degenerate
`[1]*6`, and a 20-point geometric spectrum with κ = 1e4.

For `tree_norm`, three pytrees including the **overflow** case both module
docstrings argue about: `{a: [1e20, 1e20], b: [0]}`. Squaring first turns
1e20 into `inf`, and the scaling that avoids it is the entire content of
the function — a port that dropped it agrees on the ordinary row and
disagrees here.

## 2. Numerical agreement

| quantity | agreement |
|---|---|
| `tree_norm`, all three trees | **exact float equality** |
| `largest_eigenvalue` (same operator, template, key, 12 iterations) | **exact float equality** |

Equality rather than a tolerance, because it is the same arithmetic in the
same order on the same library; a tolerance would let a genuinely different
iteration pass, which is what this file exists to notice.

## 3. Refusal agreement

Neither module refuses anything — these are numerical estimators, not
gates. The refusal that *consumes* them (`condition_bound`'s κ ceiling)
belongs to `solve.py` and is checked in `tests/exact/`.

## 4. Independent oracle

`jnp.linalg.eigvalsh` of the materialised matrix. Both packages'
`largest_eigenvalue` must approach the true λ_max **from below** (within
1e-5 above, and within 1e-3 relative) — "agreeing on a wrong number is
still agreement", so the shared claim is checked against truth as well as
against each other.

## 5. Intended differences — one, and it is the reason the harness exists

**`extreme_eigenvalues` was deliberately NOT ported.** rheplicant estimates
λ_min by running a second power iteration on `λ_max·I − M`, which on a
graded spectrum cannot separate the eigenvalues crowded against λ_max — and
the error is **one-sided and toward danger**: λ_min comes back too LARGE,
so κ too SMALL, so a convergence guard built on it is silent exactly when
it should fire. Measured upstream at κ = 1e4: λ_min over-large by 33.9×,
reported κ = 2.947e+02 against a true 1.000e+04.

bayesmith bounds λ_min below by the prior's own curvature instead
(`exact/solve.py::condition_bound`), giving an UPPER bound on κ — the
direction a safety guard needs.

Both halves are live tests, not prose:

- `test_bayesmith_does_not_carry_extreme_eigenvalues` goes red if someone
  ports it after all, and points at the docstring that rejected it.
- `test_rheplicant_still_carries_it_and_still_leans_the_unsafe_way`
  asserts the bias as an **inequality against the truth** (λ_min > 10×
  true), so it survives an iteration-count retune. When rheplicant fixes
  this, that test goes red and should be deleted together with the
  paragraph it guards.

This is a **Track A item for e-RHINO, not this migration** — recorded here
because it is the first time the harness's reason for existing paid out: a
manual comparison holds only on the day it is written.
