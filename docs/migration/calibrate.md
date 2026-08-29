# Cross-check: `calibrate` — the row §4.3 said not to migrate

`rheplicant.inference.calibrate` (`GradientCalibrator`, `AdamCalibrator`) →
`bayesmith.optimize` (`minimize`, `check_loss_sense`).

**This page exists for a module the old spec's §4.3 lists under 不迁移.** That
entry was superseded twice — by the owner's 2026-08-26 ruling *"未迁移的全部
迁移"*, and by **D11** on 2026-08-27, which names this module and gives the
reason: keeping it upstream would create two gradient-MAP implementations, in
direct violation of the plan's one-implementation law. §4.3 keeps its original
text with a marker, because it was once read as live authority and used to
revert authorised work — see **D58**.

`*` in the README table, for the same reason `conditioning.py` and `sqrtinfo`
carry one: a page with no source row of its own in §四.

**No cross-check test, and that is not an omission.** A do-not-migrate module
never had one written, so iron law 2's retirement clause is vacuous here and
its oracle clause has nothing to re-home. What that law's second branch asks
for instead — identifying an existing equivalent — is §4 below.

## 1. Fixtures

`y = a·x + b` over 32 points on `[0, 1]`, truth `a = 2.0`, `b = −0.5`, started
from `a = b = 0`. Deliberately healthy: this row's subject is whether one
descent reproduces another, not whether either survives a hard problem.

For the loss-sense guard, a `GaussianLikelihood` passed where an error
function belongs — the case that type-checks, runs, and descends a function
unbounded below while the loss history looks like textbook convergence.

## 2. Numerical agreement

**Bitwise, and it could only be measured before the deletion.**

| method | `\|Δa\|` | `\|Δb\|` | 120-step loss history |
|---|---|---|---|
| `gradient` | **0.0** | **0.0** | `max\|Δ\| = 0.0` |
| `adam` | **0.0** | **0.0** | `max\|Δ\| = 0.0` |

Not "to roundoff" — identical, including every entry of the history. Both
sides ran the same `lax.scan` arithmetic, so this is the one row in the
programme where the bit was available.

**It cannot be re-run.** Once the near side calls the far side, the same
comparison is a facade against the thing it calls. Recorded in
`docs/superpowers/specs/2026-08-29-wave-C-calibrate-opening.md` §二之三, and
that ordering — measure, then delete — is the whole reason it exists.

## 3. Refusal agreement

**Nine refusals, six near-side and three migrated.** The count was eight in
the opening's first draft: an AST scan for `ast.Raise` cannot see
`_refuse_mis_shaped_observed`, which refuses by calling `check_observed_shape`.

| group | n | class | fires | disposition |
|---|---|---|---|---|
| `check_observed_shape` | 1 | helper's | at `fit()` | **stays** — the far side never sees `observed` |
| `_refuse_a_score_…` | 3 | `ParameterSpaceError` | at `fit()` | **migrates**, wrapped (D11) |
| `__check_init__` ×2 | 5 | `StateValidationError` | **at construction** | **stays** |

The five construction guards stay for a structural reason rather than a
cautious one: `minimize` is a function, so there is no construction step to
mirror and nothing to delegate them to. `AdamCalibrator(learning_rate=-1)`
refuses immediately; delegating would defer that to `.fit()`, and *when* a
refusal fires is part of its contract —
`tests/inference/test_inference_construction_guards.py` is named for it.

The line falls exactly on the exception classes: all three
`ParameterSpaceError` cross, all five `StateValidationError` stay. So the
translation is one mapping rather than per-call-site work.

**Message pins: 23, and none break.** `n_steps`, `beta1/beta2` and
`learning_rate` are all `__check_init__` messages, which stay untouched. The
four pins on the migrated sentences match the invariant part of each — tested
by running each pin's regex against the far side's real message, which is the
only way to answer that question. Reading two messages side by side cannot.

## 4. Independent oracle

**Least squares' closed form**, written in NumPy from the normal equations in
`tests/inference/test_calibrate_against_closed_form.py`, which neither package
supplies. It replaces the bitwise comparison as the permanent, non-circular
check.

**For the descent itself, an equivalent already lives here and is identified
rather than re-homed** (iron law 2's second branch):
`tests/test_optimize.py::TestAgainstTheClosedFormPosterior` — the conjugate
formula in numpy, "differentiating nothing", plus a second case whose oracle
is a *different algorithm* (a direct linear solve against an iterative
descent).

**Mutation.** Nine seam mutants, nine killed. Three are worth naming because
nothing else in either suite catches them: sending `method="adam"` from the
gradient calibrator, dropping the `beta1` wiring, and the wrapper silently
ceasing to substitute the remedy sentence. The `beta1` one had **no killer at
all** until the knob-spy test was written — the only case in the suite that
sets a non-default beta uses `beta1=0.0, beta2=0.0` and asserts finiteness and
that a step was taken, both of which survive the far side's default of 0.9.

## 5. Intended differences

**(a) A diverged descent is refused here and returned there — deliberately,
and it is D33.** rheplicant's calibrators handed back a NaN fit; `minimize`
refuses a point whose objective is not finite. The one test that depended on
the old behaviour was named in advance by D33's triage and rewritten to catch
a raise instead of a NaN. Its claim is unchanged: a NaN returned and a refusal
raised measure the same fact.

**(b) The remedy sentence is substituted, because the far side's names
routes that do not exist upstream.** `check_loss_sense` ends with *"use a
density-aware route (`fit`, `nuts`)"*; rheplicant has neither name, and advice
pointing a user at a function their package lacks is worse than none. The
wrapper swaps it.

That substitution is keyed on the far side's exact wording, so an upstream
rewording stops it applying — **silently**, with the message going back to
naming `fit` and `nuts` while still raising, still the right class, still
passing every other case. It is therefore asserted in both directions, with an
anti-vacuity case pinning the far side's own opening clause so a wrapper that
had quietly become a local rewrite would fail.

**(c) The betas are guarded on both sides, at different entries.** rheplicant
refuses `[0, 1)` at construction; bayesmith refuses it inside `minimize`
(**D57**, added when this row found the far side had no such guard and would
return 15.38 for a minimum of 3.0). Not one rule written twice — one
constraint held at two genuinely different doors, the way an API boundary and
a database constraint both check.
