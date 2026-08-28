# Cross-check: `noise` (with `likelihood`)

`rheplicant.inference.noise` (`NoiseModel`, `HomoscedasticNoise`,
`RadiometerNoise`, `FlaggedNoise`, `inverse_variance`,
`NoiseModelLikelihood`, `check_noise_std_axis`) and
`rheplicant.inference.likelihood` (`GaussianLikelihood`,
`MaskedGaussianLikelihood`) → `bayesmith.exact.gaussian` +
`exact.precision`, and probabilistic nodes on the graph.

Test: `tests/crosscheck/test_gaussian.py`. Measured 2026-08-25.

**Two §四 rows, one page**, because both name `noise.py`: 4.1's
`likelihood.py`/`noise.py` → `exact/gaussian` (the density and the flagged
mask) and 4.2's `noise.py` → probabilistic nodes (three models × flags, in
log-density **and** in sampling distribution). §五 B9 was already landed
when this row ran, so 4.2's ordering constraint was satisfied — the
covariance reaches every consumer as a `Precision`, and the diagonal case
is the degenerate one rather than a parallel path.

## 1. Fixtures

Six samples, a per-sample sigma `[1, 0.5, 2, 1, 0.8, 1.5]`, and a
prediction `[3, −2, 5, 1, −4, 2.5]` that **crosses zero on purpose**: it is
the only regime where the radiometer's multiplicative generator and an
additive one differ, and where `RadiometerNoise.std`'s `abs` is
load-bearing.

`delta_nu · tau = 400`, so the fractional level is exactly `0.05` — the two
sides reach it by different expressions and a non-representable
intermediate would put a ULP between them for no reason of substance.

Healthy: constant sigma, and the radiometer with `floor = 0`. Must be
refused: a sigma with a zero, a negative and a NaN entry; a `flags` array
of the wrong shape.

Flagged: samples 2 and 4, so the mask is neither empty nor everything and
neither end of the array.

## 2. Numerical agreement

| quantity | agreement |
|---|---|
| constant-sigma density, **five spellings** | **identical to the last digit**, −5.846065603244213 |
| radiometer density (`σ = f·\|μ\|`), three spellings | one ULP (`rel=1e-15`) |
| flagged density, **four spellings** | **identical**, −3.5202942825891324 |
| `inverse_variance` against `Precision.apply` | **bitwise**, flagged entries exactly `0.0` |
| `HomoscedasticNoise.realise` against the node's draw | **bitwise** at the same key |

The five spellings of the first row are rheplicant's `GaussianLikelihood`
and `NoiseModelLikelihood(HomoscedasticNoise)`, bayesmith's `log_density`
on a `Precision` and `log_joint` walking the graph, and NumPy. Only the
last is evidence — see §4.

The radiometer row is **not** bitwise, and honestly so: the two sides reach
the same sigma by different expressions and sum in different orders.

`include_logdet=False` is checked to be a **different** number, with the
gap equal to `−½Σ log 2πσ²` and its sign fixed. Without that, the density
comparison would not be testing the log-determinant at all — it is the
anti-vacuity clause for B1's whole subject.

## 3. Refusal agreement

| rheplicant | here | mapping |
|---|---|---|
| zero / negative / NaN sigma | `check_gaussian` | `StateValidationError` → `StructureError` |
| `FlaggedNoise.std` with mis-shaped `flags` | JAX broadcasting, inside `dist_fn` | `StateValidationError` → raw `ValueError` — **names the shapes, not the node** |
| data that cannot reconcile with `loc` | `node_shape` | `StructureError`, naming the node and all three sources |
| `check_noise_std_axis` on a 1-D sigma | **no counterpart** | recorded in `linear.md` §5(b) — numpyro resolves it inside the user's `dist_fn` |

**A gap rather than a difference**, recorded because it is the state and
not the design: a mis-shaped sigma is refused by JAX inside the user's own
`dist_fn`, before this package reads the node, so the message names `(6,)`
and `(7,)` but no node. On a one-node graph that is perfectly usable; on a
twenty-node graph it is not. Closing it means wrapping
`apply_probabilistic`, which is the hot path `log_joint` and the bridge
both run under trace — not this row's to do. Pinned in both directions so
that closing it shows up as a red test rather than silently.

Also measured, after a first draft asserted otherwise: `trace()` does not
evaluate `dist_fn`, so neither shape fault is caught at declaration.

NaN is in the sweep deliberately: `nan > 0` and `nan < 0` are both False,
so a guard written as a single inequality lets it through. Both sides
refuse it.

**A defect this row found and fixed.** `check_gaussian`'s refusal read *"a
scale that is not strictly positive and finite (min 1)"* on
`[1, 1, inf, 1, inf, 1]` — and `1` is strictly positive and finite. The
sentence was right and the number beside it contradicted the sentence,
which is worse than no number: a reader checks the evidence, sees it
exonerate the scale, and concludes the guard is broken rather than the
model. It now names how many entries offend, the first offending value and
its flat index, and — because an infinite sigma reaching this guard is not
hypothetical — where the flagging concept actually lives.

## 4. Independent oracle

**NumPy, written differently on purpose.** `−log σ − ½ log 2π` rather than
`−½ log(2πσ²)`: a different grouping of the same expression, so a shared
algebraic slip in the two packages does not survive into the oracle.

For the flagged case the oracle is stronger than a tolerance: it is the
density of the **four unflagged samples alone**, computed without the
flagged ones existing. That is what makes "contributes exactly zero" an
assertion rather than "contributes something small".

**Mutation.** See §6.

## 5. Intended differences

**(a) The mask lives one layer away, deliberately.** `DiagonalPrecision.apply`
is `r / σ²`, so an infinite sigma gives weight zero with no special case and
`quadratic` is already the four-sample sum. But `log_normalizer` is `+inf`,
so `log_density` is `−inf` — which looks **exactly** like the "let `inf`
propagate" defect §四 4.1 warns against.

It is not. `evidence/compress.py`'s module docstring argues it: *"a sample
with infinite variance has no density, which is the honest answer to the
question a `Precision` is asked. Reading it as 0 is a statement that the
sample is UNOBSERVED, which is a modelling concept this layer has and the
interface does not."* So the mask lives where "unobserved" is a concept,
and `precision.py` keeps a normaliser that is never silently wrong.

The masked number is identical to rheplicant's. What differs is which
object you ask.

*This was nearly "fixed" while writing this row.* The `−inf` was measured,
matched against 4.1's sentence, and read as a defect; the module docstring
one directory away is the only thing that stopped a deliberate,
argued choice from being patched out.
`test_the_quadratic_half_masks_and_the_normaliser_deliberately_does_not`
pins **both** directions for the next reader who arrives at `−inf` with
4.1's sentence in hand.

**(b) The radiometer's generator is multiplicative there and additive
here.** rheplicant's `RadiometerNoise.realise` is `d(1 + f w)`, and its
docstring says why: `σ = |prediction|·f` uses an absolute value that a
*generator* must not, "and the two forms differ in sign wherever the
prediction does". A node declaring `Normal(μ, f|μ|)` draws `μ + f|μ| w`.

Measured: identical wherever the prediction is positive, and the deviation
exactly **negated** wherever it is negative. Same distribution — `w` is
symmetric, so `−f|μ|w` and `+f|μ|w` are equal in law — and a different
realisation at a fixed key. The test asserts the exact reflection rather
than "they differ", so a generator broken some other way would not pass.

**(c) `floor` means different things.** rheplicant applies it to the
magnitude *before* scaling: `max(|μ|, floor)·f`. bayesmith's own radiometer
fixture writes `κ|μ| + floor`, an additive floor. The cross-check runs at
`floor = 0`, where both reduce to `f|μ|`, so the comparison is of the noise
law and not of two spellings of a regularisation. A migration that carries
a non-zero floor across must convert it.

> **Measured 2026-08-28, and the sentence above is too mild: it cannot be
> converted.** There is no `(κ, c)` with `κ|μ| + c = f·max(|μ|, floor)` for
> all `μ`. Large `|μ|` forces `κ = f` and `c = 0`, which then gives `0` at
> `μ = 0` where rheplicant gives `f·floor`. At `f = 0.2, floor = 1.5` the two
> stand in ratios from **1.75 to 6.00** across `μ ∈ [0, 10]` — not a scale
> factor, a different **functional form**: rheplicant's floor is a floor on
> the *magnitude*, bayesmith's fixture's is a floor on the *sigma*.
>
> So a switch of this row needs the far side to be able to express
> `f·max(|μ|, floor)` — a `det` node over `μ` feeding the scale, rather than
> the affine `κ|μ| + floor` the fixture uses. That is a real piece of work and
> it is invisible from the cross-check, **because the cross-check runs at
> `floor = 0`, the one value where the difference vanishes**. `floor` is a
> live config key (`inference.noise.floor`, dimension `prediction`), so this
> is reachable by a user rather than hypothetical.
>
> This is the anti-vacuity question asked of a whole row: *what does this
> comparison hold fixed that a user can vary?*

**(d) `depends_on_prediction` is a declaration here, an attribute there.**
rheplicant's models carry it as a `ClassVar`; a graph node takes it as a
keyword the dispatcher checks against a probe. The claim is the same and
the direction of trust is not: here it is checked.

## 6. Mutation

Applied to `exact/precision.py` and `exact/gaussian.py`,
`__pycache__` cleared before each run, judged on **exit code 1 only** —
exit 4 is pytest's usage error and reads as a kill — and on the failing
test's own NAME, because a guard you did not know existed can kill a
mutation first and leave your assertion unevaluated. All four below were
killed by the test named against them. See `linear.md` §4.

| mutation | caught by |
|---|---|
| `DiagonalPrecision.apply` uses `sigma` instead of `sigma**2` | the density and `inverse_variance` comparisons |
| `log_normalizer` drops the `2π` | the five-spelling identity |
| `log_normalizer` masks non-finite sigma (the "fix" §5(a) describes) | the both-directions pin |
| `check_gaussian` reports `min(scale)` again | the refusal-evidence test — **after a fix; see below** |

The last one **survived on the first attempt**, and the reason is worth
more than the four kills. The test asserted `"min 1" not in text` — a
literal carried over from an earlier probe whose sigma was all ones. This
row's fixture is `[1, 0.5, 2, 1, 0.8, 1.5]`, so the reverted message reads
`min 0.5`, and a test written specifically to kill that mutation passed it.
The assertion now names properties the fixture derives (`2 of 6 entries`,
`at flat index 2`) and the absence of the word `min `, not one spelling of
one value. **Assert the number you measured, not the number you remember
measuring.**

## 7. What this row does NOT cover

`NoiseModel.realise` for `RadiometerNoise` is compared as a *formula*, not
as an empirical distribution — no two-sample test over many draws. The
`floor > 0` regime is untested on both sides here. And `check_noise_std_axis`
is covered only in `linear.md`, where its consumer is.
