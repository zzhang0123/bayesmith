# Cross-check: `linear`

`rheplicant.inference.linear` (`LinearBlock`, `check_linearity`,
`linear_operator`, `wiener_solve`, `gcr_sample`, `condition_bound`,
`condition_estimate`) → `bayesmith.exact.block` + `linearity` + `solve`.
Test: `tests/crosscheck/test_linear.py`. Case count and runtime are not written
here: `test_dispatch.py`'s said 6 within hours of being 7, and nothing in
this repository reads a count out of a page. Ask pytest —
`pytest tests/crosscheck/test_linear.py --collect-only -q`. Measured
2026-08-25 against e-RHINO `0c49cae`.

This is §四 4.1's first row and the largest single module in the ledger
(1788 lines against three files here).

## 1. Fixtures

The **bilinear `gain × T_ant` model** from rheplicant's own
`tests/inference/test_degenerate_partition.py` — a pinned measurement,
reused rather than re-guessed (§二 step 1), including its `PRIOR_STD = 1e6`
and its 8×8 grid. Each conditional block is genuinely affine and the pair is
not, which is the whole subject: an alternating solve reads every per-block
guard green while sitting thousands of kelvin from the truth.

Healthy: the `gain` block (8 parameters, κ bound 1.27e+20) and the `t_ant`
block (64 parameters, κ bound 3.37e+12) — two very different conditioning
regimes on one fixture, which is what makes the two convergence verdicts
reachable without a second model.

Must be refused: the **joint** group `("gain", "t_ant")`, which is the
block an alternating solve silently pretends it has.

**Everything is built inside `with jax.enable_x64(True):`, arrays
included.** Measured while writing the test: a fixture built outside the
block arrives as float32 into a float64 graph, which moved the two
packages' `condition_bound` apart by **4.096e-09** and made
`check_linearity` report unresolved departures. Both went to bitwise
agreement once the arrays moved inside. `enable_x64` governs the
operation, not the array.

## 2. Numerical agreement

| quantity | block | agreement |
|---|---|---|
| posterior mean (`wiener_solve`) | `gain`, `t_ant` | **bitwise identical** |
| relative residual | `gain`, `t_ant` | **bitwise identical** |
| `condition_bound` (same key, 12 iterations) | `gain`, `t_ant` | **bitwise identical** |
| GCR mean, 256 fixed keys | `gain` | \|z\| < 4 (measured max 1.9) |
| GCR variance and full covariance | `gain` | \|z\| < 4 (measured max 2.9) |

**Equality, not a tolerance**, for the first three. Same operator, same
right-hand side, same `jax.scipy.sparse.linalg.cg`, so the two walk the
same arithmetic in the same order; `rel=1e-12` would have admitted the
float32 fixture bug above, which showed at 4e-09.

The GCR row is the one comparison here stated in MC error, and it has to
be: the draws are **not** bitwise identical and cannot be, because each
package splits the PRNG key over its own pytree. The threshold is the
design document's `|z| < 4` at ESS ≥ 400. The keys are fixed
(`jax.random.key(s)`, `s < 256`), so every z-score is deterministic — a
failure is a change in the arithmetic, never a bad roll.

## 3. Refusal agreement

| rheplicant refusal | here | mapping |
|---|---|---|
| `κ·residual > require_convergence` | same branch | `ParameterSpaceError` → `EquinoxRuntimeError` (both from `eqx.error_if`) |
| `κ·eps > require_convergence` — the separate "tightening `tol` is useless" verdict | same branch | same, and this side adds a **second** unreachability reason (`residual` already at the precision floor) and a **third** branch for a non-finite residual |
| a `NoiseModel` at the conjugate seam | refused by the `Precision` protocol | `ParameterSpaceError` → `TypeError`; §5 |
| a 1-D `noise_std` whose axis the prediction cannot settle | **no counterpart** | §5 — the ambiguity is resolved before this package is reached |
| joint block that is not affine | `check_linearity` refuses | `ParameterSpaceError` → `StructureError` |

The two convergence verdicts are compared on **which branch fires**, not on
wording — the packages phrase the same decision differently ("the normal
operator's condition number" against "the condition bound"), and pinning
wording pins prose. The criterion itself is pinned separately: both refuse
`require_convergence=1e5` on `t_ant` and both accept `1e6`, so a guard
reading a different κ or a different residual would flip somewhere else.

## 4. Independent oracle

Iron law 4, twice over.

**Dense linear algebra.** `(AᵀN⁻¹A + S⁻¹)⁻¹(AᵀN⁻¹(d − offset) + S⁻¹m)` in
NumPy, with `A` and the offset read out of the **full model** at basis
vectors rather than hand-written. Both packages agree with it to CG's own
`tol`: 1.24e-06 on `gain`, 1.50e-07 on `t_ant` — the solver bounds the
residual, not the error, so that is the right comparison. The posterior
covariance `M⁻¹` from the same construction is what the GCR moments are
measured against.

*The first draft of this oracle wrote `gain * x` for the `t_ant` block and
silently dropped the tone, which is the offset. It then disagreed with both
packages by 64% and looked like a finding.* Recorded because an oracle is
only independent if it is also right.

**Mutation.** Four mutations applied to `exact/solve.py`, `__pycache__`
cleared before each, **all four killed**:

| mutation | caught by |
|---|---|
| `condition_bound` divides by `max(prior_variance)` instead of multiplying | the bitwise κ test |
| the GCR prior fluctuation scaled `* sqrt(variance)` instead of `/` | the covariance z-score (the MEAN is untouched by it) |
| the convergence guard reads the residual without the bound | the branch and flip tests |
| the "unreachable" branch collapsed into "did not converge" | the branch test |

*Three of these were first recorded as KILLED on a nonzero exit code that
was pytest's **usage** error (exit 4), because the runner passed `-k not
gcr` as three argv items.* Only exit 1 is a test failure. The same trap
`CLAUDE.md` records for `| tail`, in the other direction.

And a kill is not the same as *your* guard working. Each row above was
re-run with a runner that parses the junit XML for the failing test NAMES
and requires the intended one among them — a guard you did not know
existed can kill the mutation first, leaving the assertion you just wrote
unevaluated, and no exit code separates those two. All four here were
killed by the test named against them.

## 5. Intended differences

**(a) `condition_estimate` was not ported, and its docstring was wrong
upstream.**

> **Correction, 2026-08-28 (Wave B opening, iron law 7).** The first
> sentence of this row is **no longer true**: **G14** landed
> `exact/solve.py::condition_estimate` on 2026-08-27, ahead of Wave B,
> and `conditioning.md` §"What is new" was updated for it. This page was
> not, so the row that Wave B is required to read *before touching
> `linear`* still said the counterpart did not exist. Everything after the
> first sentence stands: the bias is real, the direction is certification
> rather than refusal, and bayesmith's own docstring now carries the
> warning in the first person (*"This is a diagnostic and not a bound.
> Never divide an accuracy target by it, and never guard on it"*), backed
> by an AST scan (**D37**) rather than by prose. What changed is that the
> intended DIFFERENCE became an intended AGREEMENT-with-a-warning, and the
> comparison Wave B owes is now value-for-value rather than presence.
>
> Recorded rather than rewritten, because the shape is the one this
> programme keeps paying for: one fact in two pages, one of them updated.

bayesmith had `condition_bound` only; the argument is already
in `conditioning.md` (measuring λ_min by a second power iteration errs
one-sidedly toward danger). Measured on this fixture, rheplicant's
`condition_estimate / condition_bound` is **8.38e-21** on `gain` and
**4.43e-13** on `t_ant`.

While this row was being written, rheplicant's public `condition_estimate`
opened with *"An upper bound on the conditioning"* and stated *"The number
here is now `λ_max · max(prior_variance)`"* — what `condition_bound`
returns — while its implementation called the private, measured,
biased-low one, whose own docstring calls it *"not a bound … unsafe to
guard on"*.

The e-RHINO side then measured it further, and it is worse than a
paragraph in the wrong function: **the docstring contradicted itself
internally.** Its `Returns:` line already said *"the measured condition
number"*, and its cost paragraph already said it spends `2 · iterations`
while *"`condition_bound` costs half that"* — both consistent with the
body. Only the opening line and the "A BOUND" paragraph were not. The
correct description had never been deleted; the wrong one was laid over
it. Those two faults recur differently: a paragraph in the wrong function
recurs at the next rename, an overlay recurs at the next "clarification".

The consumer makes it concrete rather than editorial. `condition_estimate`
is **the number rheplicant's `condition` run kind hands a document author**
(`config/sections/conjugate.py:873`), together with the instruction
`tol = a / κ`. A κ biased low by 33.9× (at a true 1e4) to ~700× (at 1e7)
divides an accuracy target into a `tol` too loose by the same factor, and
the direction is **certification, not refusal**.

**Docstring fixed upstream in e-RHINO `0c49cae` after this row reported
it; the numbers were deliberately not touched**, because §四 4.1 lists
`condition_estimate` as a value-for-value comparison item and
`conditioning.md`'s
`test_rheplicant_still_carries_it_and_still_leans_the_unsafe_way` exists to
pin the present behaviour. So: **docstring fixed, arithmetic unchanged,
bias still there.**

**Where the upstream text lives.** The corrected sentences are on
e-RHINO's `main`.

> **Correction, 2026-08-28 (Wave B opening, iron law 7).** This said they
> were on `track-a-tail` and **not on `main`**, so that the two guards
> asserting them read whatever the editable install had checked out.
> **Measured against the remote** (`git ls-remote` for the tip, then
> `git show origin/main:src/rheplicant/inference/linear.py`): they are on
> `origin/main`, and `track-a-tail` no longer exists. The paragraph that
> followed — that `main...track-a-tail` was docstring-only, so no numeric
> comparison depended on the checkout — described a branch that is gone;
> its conclusion survives it, since there is now one ref.
>
> The dependency mattered for a reason bigger than tidiness: while it
> stood, these two guards were green because of which branch was checked
> out, and mutating the docstring could never have surfaced that, because
> the ref was not in the variable set. e-RHINO's `CLAUDE.md` records it as
> mutation testing's structural blind spot with these two as the example.
> The lesson stands; this instance is closed.

The two guards here and in `plan.md` are still the only ones in
`tests/crosscheck/` that read upstream prose at all — worth knowing
whenever that prose is edited.

Three guards here, none of which duplicates e-RHINO's own
`TestTheTwoConditionNumbersDivideTheLabour` — that one holds the API
contract upstream, these hold this page from going stale:
`test_bayesmith_does_not_carry_condition_estimate` (red if it is ported
after all), `test_rheplicants_condition_estimate_is_orders_below_its_own_bound`
(the ratios above, as bounds so an iteration-count retune does not misfire
and a vanished bias does), and
`test_rheplicants_condition_estimate_no_longer_claims_to_be_the_bound`.

**(b) The 1-D sigma axis ambiguity cannot be refused at this seam.** This
is the row's one missing refusal, and the reason is numpyro rather than an
omission: `dist.Normal(loc, scale)` runs `promote_shapes` in its own
constructor, **inside the user's `dist_fn`**, so a bare `(8,)` has already
become `(1, 8)` before anything in this package reads
`distribution.scale` — indistinguishable from an explicit, unambiguous
`(1, 8)`. A guard written here would either miss the ambiguous case or
refuse the honest one. The information is gone.

Written as a **signed, sized** expected difference, per iron law 5.
Posterior standard deviations of the eight `gain` samples, dense oracle,
same `linspace(0.01, 1.0, 8)` read two ways:

| reading | min | max | spread |
|---|---|---|---|
| per-time `(8, 1)` — what was meant | 9.165e-07 | 8.690e-05 | 94.8× |
| per-freq `(1, 8)` — what is silently taken | 3.055e-06 | 3.228e-06 | 1.1× |

The sign is the dangerous one: the sample the data constrains **worst**
comes back **26.9× narrower** than it is, and the ~95× structure the sigma
vector describes is averaged away without a word. An over-confident error
bar reads exactly like a well-measured parameter. Both halves are live
tests, so if numpyro ever stops promoting, the first goes red and the
decision becomes writable again.

**(c) The linearity claim lives in a different place, on purpose.**
rheplicant declares `linear=True` on the **latent**; bayesmith declares
`linear_in=(...)` on the **deterministic node**. The graph form scopes the
claim to the node making it, so a model with two predictions can be affine
in a latent at one and not at the other and say so. The flat form has one
place to put the word. Same verdict on this fixture, both ways, including
the joint refusal.

**(d) bayesmith checks at more at-points, and says when it cannot resolve
one.** rheplicant's `check_linearity` probes at a single at-point and
sweeps probe magnitudes there; bayesmith adds draws from the graph's own
prior, because a single at-point is the moderate-parameter probe
`boundary-validation` exists to prevent. On **this** fixture the extra
at-points cannot be resolved — `PRIOR_STD` is 1e6 by design, so a prior
draw of `gain` is ~1e6, the prediction reaches ~1e9, and the departure
falls under the per-element roundoff floor even in float64. bayesmith
returns an `Unresolved` and warns, rather than reporting a floor as a
measured zero. The warning's own advice (open `jax.enable_x64`) does not
apply here, because the call already is inside one.

## 6. What this row does NOT cover

`LinearBlock.grouped` / `as_dict`, the complex-domain split
(`_real_parts`), and `linear_operator`'s prior reconciliation
(`_reconcile` / `_agrees`) have no comparison here. The fixture is real
throughout and declares its priors on the graph, so the complex path and
the "supplied prior disagrees with the declared one" path are both
unexercised. Named rather than left implied: a page that lists only what
it did check reads as complete six months later.
