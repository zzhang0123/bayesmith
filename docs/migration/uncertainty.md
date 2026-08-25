# Cross-check: `uncertainty` (the Fisher half) — and B1's ledger row

`rheplicant.inference.uncertainty`'s Fisher half → `bayesmith.exact.fisher`
(§四 4.1). Test: `tests/crosscheck/test_noise_logdet.py` (17 cases, which
also carries the B1 log-determinant ledger for `gls`/`engines` — see
[gls.md](gls.md)). Page written 2026-08-25 from that test's assertions,
re-run on that date; the cross-check itself predates this page.

**§三 B2 was fixed before this comparison**, per iron law 5 (commit
`aa644e0`): the condition ceiling is derived from the dtype rather than
hard-wired, and the *natural* wrong fix — widening only the inverse — is
itself pinned as ineffective (`test_widening_only_the_inverse_does_not_
recover_the_bound`), so it cannot be reintroduced as an improvement.

## 1. Fixtures

- **Constant sigma**: `straight_line` from `tests/exact/models.py`.
- **Prediction-dependent sigma**: `radiometer(kappa=f, floor=1e-9)`, swept
  over `f ∈ {0.05, 0.5, 1.0}` — the fractional levels at which the missing
  term costs 0.25 %, 22 % and 73 % of the error bar.

**The design matrix is read off bayesmith's own graph** and handed to
rheplicant's function, rather than written out a second time, so the two
sides cannot be given different models by a typo in the test file.

Note what needs no guarding here, and why it is worth saying: a Fisher
matrix **does not read the data**. It is built from the Jacobian and sigma
alone, so §0.1's PRNG trap — the same key drawing different numbers under
x64 — cannot reach this comparison. What must match is the *design*.

## 2. Numerical agreement

| quantity | agreement |
|---|---|
| `F = JᵀN⁻¹J`, constant sigma | elementwise, rel **1e-12** |
| full information incl. `(1+2f²)`, all three f | rel **1e-7** |

`kind` is asserted `"fisher"` on both sides — the same shape can be a
likelihood information or a posterior precision, and confusing them is
silent.

## 3. Refusal agreement

The refusal this module owns is bayesmith's own (`parameter_covariance`'s
condition ceiling); rheplicant has no counterpart because B2 was fixed
*here* and deliberately not back-ported — double-writing was refused, and
rheplicant's docstring points across instead. Checked in
`tests/exact/test_fisher.py`, not here.

## 4. Independent oracle, and the anti-vacuity clause that matters most

**Two packages that had both omitted the second term would agree
perfectly**, and a bare agreement test would be green on the exact defect
it exists to catch. So the agreed value is additionally required to be the
`(1 + 2f²)` **factor above the first term** — a number neither package can
reach by omission — to rel 1e-6 at all three f.

For the constant-sigma row the oracle is `tests/exact/oracle.py`, which
probes `g` on a basis of the domain and differentiates nothing.

## 5. Intended differences

1. **`depends_on_prediction` is declared, not defaulted.** It governs only
   whether a RULE is required, never whether the term is applied — under a
   constant sigma the term is exactly 0.0, so letting one flag do both
   would give two spellings of one arithmetic and no way to adjudicate.
   Default `True` (the safe side) costs 8 existing call sites an explicit
   `False`; that is deliberate, so a new call site cannot silently obtain a
   too-wide error bar.
2. **The variance term is written on the SPECTRUM**, not on per-sample
   sigmas, so it covers a fixed-basis correlated node. The diagonal case is
   bitwise the old rule (`log_spectrum` returns `2 log σ`, halving is
   exact). What the per-sample form got wrong on a correlated node is not
   small: a kernel whose shape moves while its diagonal does not registered
   as **exactly 0.0** information against a true 3.44 — blind, not
   inaccurate.
3. **`centre` and the sigma rule are cross-checked rather than trusted** —
   they are redundant by construction, and an unchecked redundancy is how a
   covariance ends up weighted at one point and curved at another.
4. **A design column is reshaped to the node's own shape before the
   precision sees it** (fixed 2026-08-25, commit `afb0f79`): the design is
   flattened while a `Precision` keeps `node_shape`, so every observed node
   with more than one axis broadcast-failed. Found by the P5 port, whose
   Jeffreys evaluation is the first caller to weight a two-dimensional
   node.
