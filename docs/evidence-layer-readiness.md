# What B11 will find here

The migration spec's step 6 and its ledger row B11 describe the streaming
evidence layer as **rewritten from the graph** rather than transplanted, and
say it was "waiting on this interface's shape". B9 is finished, so this
records what the interface actually offers — measured, on this tree, with
re-runnable probes.

Nothing here is a plan. It is the set of facts a plan would otherwise assume.

---

## 1. The whitening row is ready. The layer is not.

Step 6 reads: *"the whitening row becomes `L^-1 r`; `Precision.whiten` is the
operation it needs and already exists."* True of the row, false of the layer.

B11's must-preserve list names "the masked normalisation of `sigma = inf`
samples" among its numerical kernels. rheplicant does it with
`weight = where(seen, 1/sigma, 0)` (`inference/compress.py:111`) and a
normaliser summed over finite-sigma samples only (`compress.py:422`).
Measured on a fixture with one unobserved sample:

| operation | result |
|---|---|
| `apply(r)` at the `inf` entry | `0.0` — exactly the mask, no special case |
| `whiten(r)` at that entry | `0.0` — likewise |
| `quadratic(precision, r)` | finite; the unseen sample contributes nothing |
| `log_normalizer()` | `+inf` |
| `log_spectrum()[2]` | `+inf` |
| rheplicant's masked figure | `5.513631199` |

**Not a defect to fix in `precision.py`.** A `sigma = inf` sample has no
density, so `+inf` is the honest answer to the question a `Precision` is
asked. Reading it as `0` is a statement that the sample is UNOBSERVED — a
modelling concept the evidence layer has and this interface does not. Masking
it here would put a silently-wrong normaliser one import away from every
consumer, which is defect B1's shape.

So it is B11's first design decision.
`TestWhatTheEvidenceLayerWillFindHere` in `tests/exact/test_precision.py`
pins both halves so nobody assumes `log_normalizer` already made it.

---

## 2. Two of the three scopes are derivable. The third is a convention.

The spec's argument for rewriting rather than transplanting is that
rheplicant declares each latent's extent by hand (`Factorization`,
`scope="global" | "per_epoch" | "linked"`) while bayesmith's graph carries
plates, so the factorization can be DERIVED — "which kills the whole error
class `factorize.py` exists for (the same space declared twice)".

`docs/probes/probe_10_b11_scope_derivation.py` measures that claim.

| scope | expressible? | reaches the exact path? |
|---|---|---|
| `global` | yes | yes |
| `per_epoch` | yes | yes |
| `linked`, non-centred | yes | **yes — `gcr`** |
| `linked`, centred | yes | **no — `nuts`** |

**`global` and `per_epoch` really are a plate-membership question.**
`node.plate` is empty for one and names the epoch plate for the other. That
half of the spec's argument holds.

**`linked` is expressible and exact — non-centred.** An AR(1) chain written as
iid innovations through a deterministic linear recursion is affine in the
innovations, so the block stays linear. Measured against a dense posterior
mean: `2.5e-16` at `tol=1e-14`. At the plan's own default tolerance the error
is `1.2e-05`, which is CG tolerance and not disagreement — worth knowing
before someone reads the default as a discrepancy.

**But it is not a scope tag the graph can read off.** In that spelling the
innovations are per-epoch and the chain lives in a `Deterministic` node, so
`node.plate` alone cannot tell a linked latent from an ordinary per-epoch one.
The factorization has to know the modelling convention. **The spec's
derivation claim is two-thirds true**, and the remaining third is a naming
question rather than a structural one.

**The centred spelling needs a third `Precision` row**, and its cost is not
the QR. `fisher._log_spectrum_curvature` rests on the identity

    1/2 tr(N^-1 d_a N N^-1 d_b N) = 1/2 sum_k d_a log lam_k d_b log lam_k

which holds only where the covariance's eigenBASIS does not move with the
parameters — true of `I` and of the DFT, not of a general band. **A third row
must bring its own variance-information term.** Both spellings are pinned in
`tests/dispatch/test_classify.py` so the routing changing reads as a decision.

---

## 3. The kernel is in. `bayesmith.evidence.sqrtinfo`

Three of the five numerical kernels the ledger row names are ported and
cross-checked **bitwise** against `rheplicant.inference.sqrtinfo`
(`tests/crosscheck/test_sqrtinfo_agrees.py`, `rtol=0 atol=0`): the `[R|z]`
form, the QR fold's `-1/2 sum rho^2` corner, and the marginalisation constant
`+1/2 n log 2pi - sum log|R_ii| - 1/2 rho^2`.

The oracle throughout is dense NumPy — a log-density as
`-1/2 (x-m)^T F (x-m)`, an integral by `slogdet` — never one of our own
routines against another. Every comparison is on the **absolute**
log-density, because every constant here is invisible in a posterior's shape
and visible only in the evidence. That is how rheplicant shipped the
marginalisation constant with `-sum(log std)` missing: the probe that passed
used unit priors, where the term is exactly zero. Every fixture sweeps the
prior scale, and one test asserts the gap **is** `n_block * log(std)` at three
widths and three scales.

Two things measured that rheplicant's own docstring does not say:

* **`marginalise` raises different errors by transform.** `jit` trips the
  `bool(...)` in the pivot guard and gives `TracerBoolConversionError`;
  `grad` trips `float(max(pivots))` and gives `ConcretizationTypeError`.
  Both pinned. `marginalise_arrays` does both cleanly, which is the whole
  reason the two exist.
* **The NaN-safe spelling of the pivot guard is a SHAPE, not a live guard.**
  Rewriting `not all(> floor)` as `any(<= floor)` survives every test,
  because the finiteness check above refuses every `nan` first. Recorded in
  the comment with the measurement. The other two constants are convicted:
  dropping the corner fails 4 tests, dropping the log-pivot term fails 16.

## 4. What is still untouched

The masked normalisation (§1 — B11's first design decision, unmade), the
thousand-epoch opaque-leaf archive, the streaming layer that would USE this
kernel, the oracle suite that runs a whole campaign through it, and
`diagnostics.py`'s declarative refusal of in-span coherent errors, which the
spec calls design philosophy rather than implementation.

Nothing is blocked. The kernel is no longer the reason.
