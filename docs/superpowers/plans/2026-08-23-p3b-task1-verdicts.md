# P3b Task 1 — linearity verdicts under three normalisations, measured before B1's fix

**Date:** 2026-08-23   **Branch:** `p3b-dispatch-execution`   **Status:** measurement only, no product code changed.

B1's fix changes *which models `check_linearity` accepts*, and Task 5's classifier
reads those verdicts as its dispatch criteria. This table exists so a flipped
verdict is a recorded decision rather than a surprise discovered three tasks later.

Every number below was produced by running
`/tmp/p3b_task1_verdicts.py` (not committed). Nothing here is inferred.

---

## 1. What was measured, and how it was validated

The script drives the **real** `_env_before` / `isolate` / `noise_std_at` machinery
and reproduces `check_linearity`'s own probe scheme exactly — random per-member
directions at the domain zero, scaled by each member's declared prior width,
sub-keys folded in by position in the *sorted* names, at the same at-point list
(`at` plus `prior_at_points` draws with `fold_in(key, 7919)`). Only the three
candidate **normalisations** are re-done on those same numbers. A different probe
scheme would measure a different thing.

Two self-checks, because an unvalidated measurement is the failure this task exists
to stop repeating:

1. **The `global` column is cross-checked against `check_linearity` itself**, called
   with `rtol=inf` so it returns instead of raising. Max relative difference `0.0e+00`
   on every fixture at both dtypes. The one exception is `nan_at_negative_probes`,
   which raises through the finiteness branch regardless of `rtol` — noted, not hidden.
2. **Everything is run at both float32 and float64.** This turned out to matter more
   than the normalisation choice; see §2.

### The six criteria compared

| code | criterion |
|---|---|
| `G` | **global** — current shipped code: `_biggest(departure)/_biggest(variation)`, global floor |
| `P` | **per-elem** — Task 2's draft: per-element relative, per-element floor, `jnp.maximum(variation, 1e-300)` |
| `W` | **weighted** — Task 2's draft: `departure/sigma > WEIGHTED_RTOL`, finiteness *shared* with `P` |
| `w` | weighted judged on its **own** finiteness, still no roundoff floor |
| `p` | per-element relative with `jnp.finfo(dtype).tiny` instead of the `1e-300` literal |
| `f` | **weighted gated by the same per-element roundoff floor** `departure > 1e4·eps·max(\|actual\|,\|baseline\|)` |

`rtol = 1e4·eps` throughout (1.192e-03 in float32, 2.220e-12 in float64);
`WEIGHTED_RTOL = 1e-3`.

### Fixtures

All 26 block-forming fixtures in `tests/exact/models.py`, plus:

* **`bright_and_faint_observations` / `bright_and_faint_channels` / `faint_alone`** —
  the motivating defect, defined inside the script (Task 2 will add them to
  `models.py`). Without them the table cannot say whether a fix works.
* **`roundoff_stress(big, sigma)`** — `mu = (w + big)·X`, an **exactly true**
  `linear_in=("w",)` claim with *real* roundoff: the primal computes `(probe+big)·x`,
  the linearization computes `big·x + probe·x`, and the two differ by ordinary float
  rounding of order `eps·big·x`. Every other honest fixture in the suite is bitwise
  exact (departure identically 0), so without this family **nothing bounds any
  threshold from below at all**. Swept over offset/noise from 1e2 to 1e17, per
  `boundary-validation.md`'s "include extreme parameter values".

---

## 2. The finding that reframes the rest: the suite is float32

`tests/exact/test_linearity.py` contains **zero** occurrences of `enable_x64`, and
there is no `conftest.py` anywhere in the repo. Every shipped linearity verdict is a
**float32** verdict at `rtol = 1.19e-03`. The plan's Step 3 asks for verdicts at
`rtol = 1e4·eps(float64) = 2.22e-12` — **nine orders of magnitude apart**.

Measured dtype-induced flips at a *fixed* normalisation (no fix applied):

| fixture | criterion | float32 | float64 |
|---|---|---|---|
| `cubic_tail(prior_std=1e-4)` | global | pass | **REFUSE** |
| `cubic_tail(prior_std=1e-4)` | per-elem | pass | **REFUSE** |
| `tunable_curvature(1e-9)` | global | pass | **REFUSE** |
| `two_observations` | per-elem, weighted | **REFUSE** | pass |

`cubic_tail(prior_std=1e-4)` is asserted to PASS by
`test_the_probe_magnitude_is_read_off_the_declared_prior`. It passes in float32 and
is refused in float64 **by the current shipped guard**, with no fix applied. A flip
list quoted at one dtype alone would not describe the other, so both are given below.

---

## 3. Verdict-flip list

### 3a. The shipped defect is reproduced exactly

| fixture | global | per-elem | weighted | truth |
|---|---|---|---|---|
| `bright_and_faint_observations` | **3.451e-14 → pass** | 4.931e+00 → REFUSE | 4.763e+09 → REFUSE | FALSE |
| `bright_and_faint_channels(lying)` | **6.902e-14 → pass** | 4.931e+00 → REFUSE | 4.763e+09 → REFUSE | FALSE |
| `faint_alone` (same node, no bright sibling) | 4.931e+00 → REFUSE | REFUSE | REFUSE | FALSE |

(float64; float32 gives 2.566e-14 / 5.133e-14 with the same verdicts.)

The **3.451e-14** matches the adversarial review's reported 3.45e-14 to three
significant figures, confirming the reproduction is faithful and that the review's
number was taken in float64. The same faint node **alone** reports 4.931e+00 and is
correctly refused — the dilution is entirely the bright sibling's doing.

**Both proposed normalisations catch it, at both dtypes.** This is the flip the fix
is for, and it is real.

### 3b. Flips that are *not* wanted — measured consequences of Task 2's draft

| fixture | truth | global | Task 2 `P` | Task 2 `W` | dtype |
|---|---|---|---|---|---|
| `two_observations` | honest | pass | **REFUSE** | **REFUSE** | f32 |
| `two_observations_reverse_sorted_names` | honest | pass | **REFUSE** | **REFUSE** | f32 |
| `roundoff_stress(big=1e0, sig=1e-2)` | honest | pass | pass | **REFUSE** | f32 |
| `roundoff_stress(big≥1e3)` (9 cases) | honest | pass | **REFUSE** | **REFUSE** | f32 |
| `roundoff_stress(big≥1e13)` (4 cases) | honest | pass | pass | **REFUSE** | f64 |
| `cubic_tail(prior_std=1e-4)` | honest | REFUSE | **REFUSE** | pass | f64 |

These are **false refusals of true `linear_in` claims**, and they trace to two
concrete defects in the drafted code, both measured:

**Defect A — `1e-300` underflows in float32.**
`jnp.maximum(variation, 1e-300)` evaluates to `jnp.maximum(0.0, 0.0) = 0.0` in
float32, because 1e-300 is not representable there (measured:
`float(jnp.maximum(jnp.float32(0.0), 1e-300)) == 0.0`). Any codomain element the
block cannot move at all then gives `0/0 = NaN`, and Task 2's `not finite` branch
counts NaN as a FAILURE. `two_observations`'s covariate grid is
`jnp.linspace(-1, 1, 5)`, whose third entry is exactly `0.0`, so `mu2[2]` is
identically zero for every `w`:

```
d2: variation=[2. 1. 0. 1. 2.]   relative=[ 0.  0. nan  0.  0.]
```

An entirely honest fixture is refused because its covariate grid contains a zero.
In float64 `1e-300` is representable and the same fixture passes — hence the
dtype flip above. Replacing the literal with `jnp.finfo(dtype).tiny` removes all
of these (criterion `p`: 12 wrong → 1 wrong in float32).

> Note the `1e-300` guard does not fully rescue float64 either: at
> `roundoff_stress(big=1e15)` the per-element ratio reaches **2.500e+299** — finite,
> so the finiteness branch never fires, but far above `rtol`. What saves it is the
> *per-element floor*, not the guard.

**Defect B — the weighted criterion has no roundoff floor.**
`departure/sigma` grows without bound as the offset-to-noise ratio grows, so it
measures dynamic range rather than curvature. Measured on exactly-affine models:

| offset/noise | weighted, float32 | weighted, float64 | weighted **above the floor**, both |
|---|---|---|---|
| 1e2 | 2.441e-02 | 2.274e-11 | **0.000e+00** |
| 1e8 | 1.250e+01 | 2.328e-08 | **0.000e+00** |
| 1e11 | 1.280e+04 | 2.384e-05 | **0.000e+00** |
| 1e17 | (below f32 resolution) | 2.500e+01 | **0.000e+00** |

In float32 an honest model needs only offset/noise ≈ 1e2 to breach
`WEIGHTED_RTOL = 1e-3`; in float64 it takes ≈ 1e13, still well inside the range this
package states it targets ("a foreground in K beside a signal in mK is 1e6, and an
interferometric visibility against a monopole is far more"). **Gating the weighted
criterion by the same per-element roundoff floor the relative measure already uses
drives every one of these to exactly 0.000e+00.**

---

## 4. Scoreboard — misclassifications over all 48 fixture rows

| criterion | float32 wrong | float64 wrong |
|---|---|---|
| `G` global (**current shipped**) | 3 (all false ACCEPT, incl. both defect cases) | 3 (2 false accept, 1 false refuse) |
| `P` per-elem, Task 2 draft | 12 | 1 |
| `W` weighted, Task 2 draft (shared finiteness) | 12 | 4 |
| `w` weighted alone, no floor | 5 | 4 |
| `p` per-elem with `finfo.tiny` | 1 | 1 |
| **`f` weighted + per-element roundoff floor** | **1** | **0 — perfect** |

The single float32 miss of criterion `f` is `tunable_curvature(departure=1e-9)`,
whose above-floor weighted departure is exactly **0.000e+00** in float32: its
curvature is genuinely below float32 roundoff and no criterion can see it at that
dtype. The current shipped guard misses it too. In float64 it is caught
(2.117e-02 → REFUSE).

Criterion `f` is also the only one that gets `cubic_tail(prior_std=1e-4)` right in
float64 — both relative criteria refuse it there, contradicting
`test_the_probe_magnitude_is_read_off_the_declared_prior`'s assertion (which holds
only because the suite runs float32).

---

## 5. `WEIGHTED_RTOL` — validated at 1e-3

Measured separation **on the criterion being recommended** (weighted, restricted to
elements clearing the roundoff floor):

| | float64 | float32 |
|---|---|---|
| worst **honest** | **4.870e-08** (`cubic_tail(prior_std=1e-4)`) | **0.000e+00** (all honest fixtures) |
| smallest **false** | 2.117e-02 (`tunable_curvature(1e-9)`) | 1.171e+01 (`tunable_curvature(1e-6)`) [^f32] |
| usable window | (4.87e-08, 2.12e-02] — span 4.35e+05x | (0, 1.17e+01] |
| **margin below at 1e-3** | **2.05e+04x** | unbounded |
| **margin above at 1e-3** | **2.12e+01x** | 1.17e+04x |

Margins against the false claims the task names, float64, above-floor weighted:

| fixture | value | margin above 1e-3 |
|---|---|---|
| `quadratic_claim` | 8.468e+07 | 8.5e+10x |
| `affine_only_at_zero` | 4.336e+08 | 4.3e+11x |
| `cubic_tail(prior_std=1.0)` | 4.870e+04 | 4.9e+07x |
| `bilinear_pair(joint)` | 3.395e+07 | 3.4e+10x |
| `bright_and_faint_observations` | 4.763e+09 | 4.8e+12x |

[^f32]: `tunable_curvature(1e-9)`'s above-floor weighted departure is exactly
    `0.000e+00` in float32 — genuinely below float32 roundoff, and therefore
    undetectable at that dtype by any criterion, the current one included. It is
    excluded from the float32 column because it bounds nothing.

**Verdict: keep `WEIGHTED_RTOL = 1e-3`.** Every legitimate fixture sits at least
2.05e+04x below it and every *named* false claim at least 4.9e+07x above it. The only
thing within 21x is `tunable_curvature(departure=1e-9)`, a synthetic dial built to sit
near a threshold; letting a knob set the design would be backwards.

The choice is also principled rather than fitted: `WEIGHTED_RTOL = 1e-3` means
"a departure that could move any residual by more than 0.001 sigma is refused", so a
claim that slips through moves the posterior by less than 0.001 sigma — which is what
the criterion is meant to tolerate. Lowering it toward the window's geometric centre
(≈3.2e-05) would buy margin only against that synthetic dial while spending margin
against real honest models.

**This number is only meaningful once the roundoff floor is added.** Without the
floor, `1e-3` is breached by an honest model at offset/noise ≈ 1e2 in float32 (§3b),
and no threshold in the window survives — the unfloored window is *empty*
(worst honest 1.280e+04 > smallest false 1.172e-02).

---

## 6. Conclusion

**Task 2 should adopt the per-element normalisation — with two repairs the draft is
missing — and gate the sigma-weighted criterion by the same per-element roundoff
floor:**

1. Replace the `1e-300` literal with `jnp.finfo(dtype).tiny` (Defect A).
2. Require `departure > 1e4·eps·max(|actual|, |baseline|)` for the **weighted**
   criterion too, not only the relative one (Defect B).
3. Give the weighted criterion its **own** finiteness check rather than sharing one
   with the relative measure, so a 0/0 in the relative column cannot poison it.

With all three, the disjunction "either criterion fails → refuse" misclassifies 1 of
48 rows in float32 (`tunable_curvature(1e-9)`, below float32 resolution — the current
guard misses it too) and 1 of 48 in float64 (`cubic_tail(prior_std=1e-4)`, a false
*refusal* from the relative half). Without them it misclassifies 12.

**Task 5's classification table:** no change is needed for any fixture it is likely
to use. Every honest fixture in `models.py` keeps its `pass` verdict under the
repaired criteria at both dtypes, and the three fixtures whose verdicts change
(`bright_and_faint_observations`, `bright_and_faint_channels`, `faint_alone`) all
change from a wrong `pass` to a correct `REFUSE`. The one entry to write down
explicitly is that **`cubic_tail(prior_std=1e-4)` is dtype-dependent**: it passes in
float32 (as the suite asserts) and is refused in float64 by either relative
criterion, including the current shipped one.

---

## 7. Full measured tables

Values are the max over the three at-points at that scale; `F` = REFUSE, `.` = pass.
The per-scale tables carry four verdict marks (`G P W w`); the per-fixture overall
tables below carry all six criteria of §1 (`global`, `per-elem`, `weighted`,
`w-alone`, `per-fix`, `w+floor`).

### float32 — what `tests/exact/test_linearity.py` actually runs

```

fixture                                   scale     global    per-raw    per-flr   weighted | G P W w
-----------------------------------------------------------------------------------------------------
straight_line                           1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
straight_line                           1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
straight_line                           1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
two_linear_latents                      1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
two_linear_latents                      1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
two_linear_latents                      1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
collinear_pair                          1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
collinear_pair                          1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
collinear_pair                          1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
plated_latent                           1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
plated_latent                           1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
plated_latent                           1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
plated_latent_through_deterministic     1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
plated_latent_through_deterministic     1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
plated_latent_through_deterministic     1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
two_observations                        1.0e-03  0.00e+00       nan  0.00e+00  0.00e+00 | . F F .
two_observations                        1.0e+00  0.00e+00       nan  0.00e+00  0.00e+00 | . F F .
two_observations                        1.0e+03  0.00e+00       nan  0.00e+00  0.00e+00 | . F F .
two_observations_reverse_sorted_names   1.0e-03  0.00e+00       nan  0.00e+00  0.00e+00 | . F F .
two_observations_reverse_sorted_names   1.0e+00  0.00e+00       nan  0.00e+00  0.00e+00 | . F F .
two_observations_reverse_sorted_names   1.0e+03  0.00e+00       nan  0.00e+00  0.00e+00 | . F F .
plated_and_scalar_latents               1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
plated_and_scalar_latents               1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
plated_and_scalar_latents               1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
prior_held_direction                    1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
prior_held_direction                    1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
prior_held_direction                    1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
unconstrained_latent                    1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
unconstrained_latent                    1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
unconstrained_latent                    1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
radiometer                              1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
radiometer                              1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
radiometer                              1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
plated_radiometer                       1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
plated_radiometer                       1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
plated_radiometer                       1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
radiometer_group                        1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
radiometer_group                        1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
radiometer_group                        1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
one_sided_sigma                         1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
one_sided_sigma                         1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
one_sided_sigma                         1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
many_observations(12)                   1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
many_observations(12)                   1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
many_observations(12)                   1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
wide_plate(256)                         1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
wide_plate(256)                         1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
wide_plate(256)                         1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
indirect_ancestor                       1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
indirect_ancestor                       1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
indirect_ancestor                       1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
diamond_ancestor                        1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
diamond_ancestor                        1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
diamond_ancestor                        1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
shared_ancestor                         1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
shared_ancestor                         1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
shared_ancestor                         1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
overflowing_outside_latent              1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
overflowing_outside_latent              1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
overflowing_outside_latent              1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
improper_outside_prior                  1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
improper_outside_prior                  1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
improper_outside_prior                  1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
tunable_curvature(0.0)                  1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
tunable_curvature(0.0)                  1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
tunable_curvature(0.0)                  1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
cubic_tail(prior_std=1e-4)              1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
cubic_tail(prior_std=1e-4)              1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
cubic_tail(prior_std=1e-4)              1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
affine_only_at_zero @z=0 pinned         1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
affine_only_at_zero @z=0 pinned         1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
affine_only_at_zero @z=0 pinned         1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
bright_and_faint_channels(honest)       1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
bright_and_faint_channels(honest)       1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
bright_and_faint_channels(honest)       1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
roundoff_stress(big=1e+00,sig=1e-02)    1.0e-03  1.68e-03  2.79e-03  0.00e+00  1.19e-05 | . . . .
roundoff_stress(big=1e+00,sig=1e-02)    1.0e+00  1.40e-06  1.75e-06  0.00e+00  1.19e-05 | . . . .
roundoff_stress(big=1e+00,sig=1e-02)    1.0e+03  7.62e-08  1.19e-07  0.00e+00  2.44e-02 | . . F F
roundoff_stress(big=1e+03,sig=1e-02)    1.0e-03  1.00e+00       nan  0.00e+00  1.22e-02 | . F F F
roundoff_stress(big=1e+03,sig=1e-02)    1.0e+00  1.43e-03  2.04e-03  0.00e+00  1.22e-02 | . . F F
roundoff_stress(big=1e+03,sig=1e-02)    1.0e+03  3.05e-07  5.08e-07  0.00e+00  2.44e-02 | . . F F
roundoff_stress(big=1e+06,sig=1e-02)    1.0e-03  0.00e+00       nan  0.00e+00  0.00e+00 | . F F .
roundoff_stress(big=1e+06,sig=1e-02)    1.0e+00  1.00e+00       nan  0.00e+00  1.25e+01 | . F F F
roundoff_stress(big=1e+06,sig=1e-02)    1.0e+03  3.12e-04  5.20e-04  0.00e+00  1.25e+01 | . . F F
roundoff_stress(big=1e+09,sig=1e-02)    1.0e-03  0.00e+00       nan  0.00e+00  0.00e+00 | . F F .
roundoff_stress(big=1e+09,sig=1e-02)    1.0e+00  0.00e+00       nan  0.00e+00  0.00e+00 | . F F .
roundoff_stress(big=1e+09,sig=1e-02)    1.0e+03  3.33e-01  5.00e-01  0.00e+00  1.28e+04 | . . F F
roundoff_stress(big=1e+06,sig=1e+00)    1.0e-03  0.00e+00       nan  0.00e+00  0.00e+00 | . F F .
roundoff_stress(big=1e+06,sig=1e+00)    1.0e+00  1.00e+00       nan  0.00e+00  1.25e-01 | . F F F
roundoff_stress(big=1e+06,sig=1e+00)    1.0e+03  3.12e-04  5.20e-04  0.00e+00  1.25e-01 | . . F F
roundoff_stress(big=1e+12,sig=1e+00)    1.0e-03  0.00e+00       nan  0.00e+00  0.00e+00 | . F F .
roundoff_stress(big=1e+12,sig=1e+00)    1.0e+00  0.00e+00       nan  0.00e+00  0.00e+00 | . F F .
roundoff_stress(big=1e+12,sig=1e+00)    1.0e+03  0.00e+00       nan  0.00e+00  0.00e+00 | . F F .
roundoff_stress(big=1e+13,sig=1e+00)    1.0e-03  0.00e+00       nan  0.00e+00  0.00e+00 | . F F .
roundoff_stress(big=1e+13,sig=1e+00)    1.0e+00  0.00e+00       nan  0.00e+00  0.00e+00 | . F F .
roundoff_stress(big=1e+13,sig=1e+00)    1.0e+03  0.00e+00       nan  0.00e+00  0.00e+00 | . F F .
roundoff_stress(big=1e+15,sig=1e+00)    1.0e-03  0.00e+00       nan  0.00e+00  0.00e+00 | . F F .
roundoff_stress(big=1e+15,sig=1e+00)    1.0e+00  0.00e+00       nan  0.00e+00  0.00e+00 | . F F .
roundoff_stress(big=1e+15,sig=1e+00)    1.0e+03  0.00e+00       nan  0.00e+00  0.00e+00 | . F F .
roundoff_stress(big=1e+12,sig=1e-02)    1.0e-03  0.00e+00       nan  0.00e+00  0.00e+00 | . F F .
roundoff_stress(big=1e+12,sig=1e-02)    1.0e+00  0.00e+00       nan  0.00e+00  0.00e+00 | . F F .
roundoff_stress(big=1e+12,sig=1e-02)    1.0e+03  0.00e+00       nan  0.00e+00  0.00e+00 | . F F .
roundoff_stress(big=1e+15,sig=1e-02)    1.0e-03  0.00e+00       nan  0.00e+00  0.00e+00 | . F F .
roundoff_stress(big=1e+15,sig=1e-02)    1.0e+00  0.00e+00       nan  0.00e+00  0.00e+00 | . F F .
roundoff_stress(big=1e+15,sig=1e-02)    1.0e+03  0.00e+00       nan  0.00e+00  0.00e+00 | . F F .
quadratic_claim                         1.0e-03  1.00e+00  1.00e+00  1.00e+00  3.42e-05 | F F . .
quadratic_claim                         1.0e+00  1.00e+00  1.00e+00  1.00e+00  7.30e+01 | F F F F
quadratic_claim                         1.0e+03  1.00e+00  1.00e+00  1.00e+00  4.68e+07 | F F F F
bilinear_pair(joint)                    1.0e-03  1.00e+00  1.00e+00  1.00e+00  6.09e-05 | F F . .
bilinear_pair(joint)                    1.0e+00  1.00e+00  1.00e+00  1.00e+00  1.25e+02 | F F F F
bilinear_pair(joint)                    1.0e+03  1.00e+00  1.00e+00  1.00e+00  5.02e+07 | F F F F
cubic_tail(prior_std=1.0)               1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
cubic_tail(prior_std=1.0)               1.0e+00  4.58e-06  4.62e-06  0.00e+00  3.91e-05 | . . . .
cubic_tail(prior_std=1.0)               1.0e+03  7.45e-01  7.45e-01  7.45e-01  2.00e+04 | F F F F
affine_only_at_zero                     1.0e-03  2.89e-03  2.89e-03  2.89e-03  4.63e-06 | F F . .
affine_only_at_zero                     1.0e+00  1.17e+00  1.17e+00  1.17e+00  1.60e+02 | F F F F
affine_only_at_zero                     1.0e+03  1.00e+00  1.00e+00  1.00e+00  2.37e+07 | F F F F
tunable_curvature(1e-9)                 1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
tunable_curvature(1e-9)                 1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
tunable_curvature(1e-9)                 1.0e+03  1.71e-06  1.78e-06  0.00e+00  1.17e-02 | . . F F
tunable_curvature(1e-6)                 1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
tunable_curvature(1e-6)                 1.0e+00  2.12e-06  2.15e-06  0.00e+00  1.81e-05 | . . . .
tunable_curvature(1e-6)                 1.0e+03  1.71e-03  1.71e-03  1.71e-03  1.17e+01 | F F F F
tunable_curvature(1e-3)                 1.0e-03  1.43e-06  1.52e-06  0.00e+00  8.38e-09 | . . . .
tunable_curvature(1e-3)                 1.0e+00  2.14e-03  2.14e-03  2.14e-03  1.83e-02 | F F F F
tunable_curvature(1e-3)                 1.0e+03  2.41e+00  2.41e+00  2.41e+00  1.17e+04 | F F F F
tunable_curvature(1e-1)                 1.0e-03  1.46e-04  1.46e-04  0.00e+00  8.56e-07 | . . . .
tunable_curvature(1e-1)                 1.0e+00  2.72e-01  2.72e-01  2.72e-01  1.83e+00 | F F F F
tunable_curvature(1e-1)                 1.0e+03  1.01e+00  1.01e+00  1.01e+00  1.17e+06 | F F F F
bright_and_faint_observations           1.0e-03  2.19e-20  2.20e-03  2.20e-03  1.93e-03 | . F F F
bright_and_faint_observations           1.0e+00  3.20e-17  1.55e+00  1.55e+00  4.11e+03 | . F F F
bright_and_faint_observations           1.0e+03  2.57e-14  1.00e+00  1.00e+00  2.63e+09 | . F F F
faint_alone                             1.0e-03  2.20e-03  2.20e-03  2.20e-03  1.93e-03 | F F F F
faint_alone                             1.0e+00  1.55e+00  1.55e+00  1.55e+00  4.11e+03 | F F F F
faint_alone                             1.0e+03  1.00e+00  1.00e+00  1.00e+00  2.63e+09 | F F F F
bright_and_faint_channels(lying)        1.0e-03  4.39e-20  2.20e-03  2.20e-03  1.93e-03 | . F F F
bright_and_faint_channels(lying)        1.0e+00  6.41e-17  1.55e+00  1.55e+00  4.11e+03 | . F F F
bright_and_faint_channels(lying)        1.0e+03  5.13e-14  1.00e+00  1.00e+00  2.63e+09 | . F F F
nan_at_negative_probes                  1.0e-03       nan       nan       inf       nan | F F F F
nan_at_negative_probes                  1.0e+00       nan       nan  0.00e+00       nan | F F F F
nan_at_negative_probes                  1.0e+03       nan       nan       inf       nan | F F F F

```

### float64 — what the plan's Step 3 asks for

```

fixture                                   scale     global    per-raw    per-flr   weighted | G P W w
-----------------------------------------------------------------------------------------------------
straight_line                           1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
straight_line                           1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
straight_line                           1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
two_linear_latents                      1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
two_linear_latents                      1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
two_linear_latents                      1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
collinear_pair                          1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
collinear_pair                          1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
collinear_pair                          1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
plated_latent                           1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
plated_latent                           1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
plated_latent                           1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
plated_latent_through_deterministic     1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
plated_latent_through_deterministic     1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
plated_latent_through_deterministic     1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
two_observations                        1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
two_observations                        1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
two_observations                        1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
two_observations_reverse_sorted_names   1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
two_observations_reverse_sorted_names   1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
two_observations_reverse_sorted_names   1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
plated_and_scalar_latents               1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
plated_and_scalar_latents               1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
plated_and_scalar_latents               1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
prior_held_direction                    1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
prior_held_direction                    1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
prior_held_direction                    1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
unconstrained_latent                    1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
unconstrained_latent                    1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
unconstrained_latent                    1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
radiometer                              1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
radiometer                              1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
radiometer                              1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
plated_radiometer                       1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
plated_radiometer                       1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
plated_radiometer                       1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
radiometer_group                        1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
radiometer_group                        1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
radiometer_group                        1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
one_sided_sigma                         1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
one_sided_sigma                         1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
one_sided_sigma                         1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
many_observations(12)                   1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
many_observations(12)                   1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
many_observations(12)                   1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
wide_plate(256)                         1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
wide_plate(256)                         1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
wide_plate(256)                         1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
indirect_ancestor                       1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
indirect_ancestor                       1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
indirect_ancestor                       1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
diamond_ancestor                        1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
diamond_ancestor                        1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
diamond_ancestor                        1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
shared_ancestor                         1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
shared_ancestor                         1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
shared_ancestor                         1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
overflowing_outside_latent              1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
overflowing_outside_latent              1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
overflowing_outside_latent              1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
improper_outside_prior                  1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
improper_outside_prior                  1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
improper_outside_prior                  1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
tunable_curvature(0.0)                  1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
tunable_curvature(0.0)                  1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
tunable_curvature(0.0)                  1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
cubic_tail(prior_std=1e-4)              1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
cubic_tail(prior_std=1e-4)              1.0e+00  6.97e-15  7.02e-15  0.00e+00  2.33e-18 | . . . .
cubic_tail(prior_std=1e-4)              1.0e+03  5.29e-08  5.29e-08  5.29e-08  4.87e-08 | F F . .
affine_only_at_zero @z=0 pinned         1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
affine_only_at_zero @z=0 pinned         1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
affine_only_at_zero @z=0 pinned         1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
bright_and_faint_channels(honest)       1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
bright_and_faint_channels(honest)       1.0e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
bright_and_faint_channels(honest)       1.0e+03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
roundoff_stress(big=1e+00,sig=1e-02)    1.0e-03  2.14e-13  3.06e-13  0.00e+00  2.22e-14 | . . . .
roundoff_stress(big=1e+00,sig=1e-02)    1.0e+00  1.97e-15  2.82e-15  0.00e+00  2.22e-14 | . . . .
roundoff_stress(big=1e+00,sig=1e-02)    1.0e+03  1.39e-16  1.91e-16  0.00e+00  2.27e-11 | . . . .
roundoff_stress(big=1e+03,sig=1e-02)    1.0e-03  1.76e-10  2.32e-10  0.00e+00  2.27e-11 | . . . .
roundoff_stress(big=1e+03,sig=1e-02)    1.0e+00  2.02e-12  2.52e-12  0.00e+00  2.27e-11 | . . . .
roundoff_stress(big=1e+03,sig=1e-02)    1.0e+03  1.15e-16  1.91e-16  0.00e+00  1.71e-11 | . . . .
roundoff_stress(big=1e+06,sig=1e-02)    1.0e-03  2.24e-07  3.74e-07  0.00e+00  2.33e-08 | . . . .
roundoff_stress(big=1e+06,sig=1e-02)    1.0e+00  2.07e-09  2.59e-09  0.00e+00  2.33e-08 | . . . .
roundoff_stress(big=1e+06,sig=1e-02)    1.0e+03  2.35e-13  2.94e-13  0.00e+00  2.33e-08 | . . . .
roundoff_stress(big=1e+09,sig=1e-02)    1.0e-03  1.85e-04  2.64e-04  0.00e+00  2.38e-05 | . . . .
roundoff_stress(big=1e+09,sig=1e-02)    1.0e+00  1.21e-06  2.02e-06  0.00e+00  2.38e-05 | . . . .
roundoff_stress(big=1e+09,sig=1e-02)    1.0e+03  1.46e-10  2.43e-10  0.00e+00  2.38e-05 | . . . .
roundoff_stress(big=1e+06,sig=1e+00)    1.0e-03  2.24e-07  3.74e-07  0.00e+00  2.33e-10 | . . . .
roundoff_stress(big=1e+06,sig=1e+00)    1.0e+00  2.07e-09  2.59e-09  0.00e+00  2.33e-10 | . . . .
roundoff_stress(big=1e+06,sig=1e+00)    1.0e+03  2.35e-13  2.94e-13  0.00e+00  2.33e-10 | . . . .
roundoff_stress(big=1e+12,sig=1e+00)    1.0e-03  2.50e-01  5.00e-01  0.00e+00  2.44e-04 | . . . .
roundoff_stress(big=1e+12,sig=1e+00)    1.0e+00  2.17e-03  2.72e-03  0.00e+00  2.44e-04 | . . . .
roundoff_stress(big=1e+12,sig=1e+00)    1.0e+03  1.49e-07  1.66e-07  0.00e+00  2.44e-04 | . . . .
roundoff_stress(big=1e+13,sig=1e+00)    1.0e-03 1.95e+297 1.95e+297  0.00e+00  1.95e-03 | . . F F
roundoff_stress(big=1e+13,sig=1e+00)    1.0e+00  1.72e-02  2.44e-02  0.00e+00  1.95e-03 | . . F F
roundoff_stress(big=1e+13,sig=1e+00)    1.0e+03  1.97e-06  2.82e-06  0.00e+00  1.95e-03 | . . F F
roundoff_stress(big=1e+15,sig=1e+00)    1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
roundoff_stress(big=1e+15,sig=1e+00)    1.0e+00  1.00e+00 2.50e+299  0.00e+00  2.50e-01 | . . F F
roundoff_stress(big=1e+15,sig=1e+00)    1.0e+03  1.53e-04  2.18e-04  0.00e+00  2.50e-01 | . . F F
roundoff_stress(big=1e+12,sig=1e-02)    1.0e-03  2.50e-01  5.00e-01  0.00e+00  2.44e-02 | . . F F
roundoff_stress(big=1e+12,sig=1e-02)    1.0e+00  2.17e-03  2.72e-03  0.00e+00  2.44e-02 | . . F F
roundoff_stress(big=1e+12,sig=1e-02)    1.0e+03  1.49e-07  1.66e-07  0.00e+00  2.44e-02 | . . F F
roundoff_stress(big=1e+15,sig=1e-02)    1.0e-03  0.00e+00  0.00e+00  0.00e+00  0.00e+00 | . . . .
roundoff_stress(big=1e+15,sig=1e-02)    1.0e+00  1.00e+00 2.50e+299  0.00e+00  2.50e+01 | . . F F
roundoff_stress(big=1e+15,sig=1e-02)    1.0e+03  1.53e-04  2.18e-04  0.00e+00  2.50e+01 | . . F F
quadratic_claim                         1.0e-03  1.00e+00  1.00e+00  1.00e+00  7.82e-06 | F F . .
quadratic_claim                         1.0e+00  1.00e+00  1.00e+00  1.00e+00  1.12e+01 | F F F F
quadratic_claim                         1.0e+03  1.00e+00  1.00e+00  1.00e+00  8.47e+07 | F F F F
bilinear_pair(joint)                    1.0e-03  1.00e+00  1.00e+00  1.00e+00  7.37e-06 | F F . .
bilinear_pair(joint)                    1.0e+00  1.00e+00  1.00e+00  1.00e+00  1.90e+01 | F F F F
bilinear_pair(joint)                    1.0e+03  1.00e+00  1.00e+00  1.00e+00  3.39e+07 | F F F F
cubic_tail(prior_std=1.0)               1.0e-03  4.89e-13  4.89e-13  0.00e+00  1.37e-15 | . . . .
cubic_tail(prior_std=1.0)               1.0e+00  6.99e-07  6.99e-07  6.99e-07  2.34e-06 | F F . .
cubic_tail(prior_std=1.0)               1.0e+03  8.41e-01  8.41e-01  8.41e-01  4.87e+04 | F F F F
affine_only_at_zero                     1.0e-03  1.08e-02  1.08e-02  1.08e-02  3.74e-05 | F F . .
affine_only_at_zero                     1.0e+00  6.25e+00  6.25e+00  6.25e+00  7.95e-01 | F F F F
affine_only_at_zero                     1.0e+03  1.00e+00  1.00e+00  1.00e+00  4.34e+08 | F F F F
tunable_curvature(1e-9)                 1.0e-03  6.99e-13  6.99e-13  0.00e+00  1.96e-15 | . . . .
tunable_curvature(1e-9)                 1.0e+00  8.36e-10  8.36e-10  8.36e-10  2.80e-09 | F F . .
tunable_curvature(1e-9)                 1.0e+03  2.30e-06  2.30e-06  2.30e-06  2.12e-02 | F F F F
tunable_curvature(1e-6)                 1.0e-03  6.99e-10  6.99e-10  6.99e-10  1.96e-12 | F F . .
tunable_curvature(1e-6)                 1.0e+00  8.36e-07  8.36e-07  8.36e-07  2.80e-06 | F F . .
tunable_curvature(1e-6)                 1.0e+03  2.31e-03  2.31e-03  2.31e-03  2.12e+01 | F F F F
tunable_curvature(1e-3)                 1.0e-03  6.99e-07  6.99e-07  6.99e-07  1.96e-09 | F F . .
tunable_curvature(1e-3)                 1.0e+00  8.37e-04  8.37e-04  8.37e-04  2.80e-03 | F F F F
tunable_curvature(1e-3)                 1.0e+03  4.50e+00  4.50e+00  4.50e+00  2.12e+04 | F F F F
tunable_curvature(1e-1)                 1.0e-03  6.99e-05  6.99e-05  6.99e-05  1.96e-07 | F F . .
tunable_curvature(1e-1)                 1.0e+00  9.13e-02  9.13e-02  9.13e-02  2.80e-01 | F F F F
tunable_curvature(1e-1)                 1.0e+03  1.02e+00  1.02e+00  1.02e+00  2.12e+06 | F F F F
bright_and_faint_observations           1.0e-03  1.05e-20  1.05e-03  1.05e-03  4.40e-04 | . F . .
bright_and_faint_observations           1.0e+00  1.25e-17  4.93e+00  4.93e+00  6.29e+02 | . F F F
bright_and_faint_observations           1.0e+03  3.45e-14  1.00e+00  1.00e+00  4.76e+09 | . F F F
faint_alone                             1.0e-03  1.05e-03  1.05e-03  1.05e-03  4.40e-04 | F F . .
faint_alone                             1.0e+00  4.93e+00  4.93e+00  4.93e+00  6.29e+02 | F F F F
faint_alone                             1.0e+03  1.00e+00  1.00e+00  1.00e+00  4.76e+09 | F F F F
bright_and_faint_channels(lying)        1.0e-03  2.10e-20  1.05e-03  1.05e-03  4.40e-04 | . F . .
bright_and_faint_channels(lying)        1.0e+00  2.51e-17  4.93e+00  4.93e+00  6.29e+02 | . F F F
bright_and_faint_channels(lying)        1.0e+03  6.90e-14  1.00e+00  1.00e+00  4.76e+09 | . F F F
nan_at_negative_probes                  1.0e-03       nan       nan       inf       nan | F F F F
nan_at_negative_probes                  1.0e+00       nan       nan  0.00e+00       nan | F F F F
nan_at_negative_probes                  1.0e+03       nan       nan  0.00e+00       nan | F F F F

```

### Per-fixture overall verdicts

float32:

```
---- OVERALL VERDICTS (float32  (what tests/exact/test_linearity.py actually runs)) ----
fixture                                  truth   global  per-elem  weighted   w-alone   per-fix   w+floor  note
straight_line                           honest     pass      pass      pass      pass      pass      pass  
two_linear_latents                      honest     pass      pass      pass      pass      pass      pass  
collinear_pair                          honest     pass      pass      pass      pass      pass      pass  
plated_latent                           honest     pass      pass      pass      pass      pass      pass  
plated_latent_through_deterministic     honest     pass      pass      pass      pass      pass      pass  
two_observations                        honest     pass    REFUSE    REFUSE      pass      pass      pass  2 observed leaves
two_observations_reverse_sorted_names   honest     pass    REFUSE    REFUSE      pass      pass      pass  
plated_and_scalar_latents               honest     pass      pass      pass      pass      pass      pass  heterogeneous member sizes
prior_held_direction                    honest     pass      pass      pass      pass      pass      pass  `loose` reaches no observed node
unconstrained_latent                    honest     pass      pass      pass      pass      pass      pass  `u` reaches no observed node
radiometer                              honest     pass      pass      pass      pass      pass      pass  sigma(mu)
plated_radiometer                       honest     pass      pass      pass      pass      pass      pass  sigma(mu) elementwise
radiometer_group                        honest     pass      pass      pass      pass      pass      pass  one obs sigma(mu), one constant
one_sided_sigma                         honest     pass      pass      pass      pass      pass      pass  
many_observations(12)                   honest     pass      pass      pass      pass      pass      pass  12 observed leaves
wide_plate(256)                         honest     pass      pass      pass      pass      pass      pass  256-element plate
indirect_ancestor                       honest     pass      pass      pass      pass      pass      pass  tau outside at its prior mean
diamond_ancestor                        honest     pass      pass      pass      pass      pass      pass  tau outside at its prior mean
shared_ancestor                         honest     pass      pass      pass      pass      pass      pass  only the PAIR is refused, not this block
overflowing_outside_latent              honest     pass      pass      pass      pass      pass      pass  at_points PINNED to z=0 (Cauchy draws overflow)
improper_outside_prior                  honest     pass      pass      pass      pass      pass      pass  at_points PINNED (improper prior has no sampler)
tunable_curvature(0.0)                  honest     pass      pass      pass      pass      pass      pass  departure exactly 0
cubic_tail(prior_std=1e-4)              honest     pass      pass      pass      pass      pass      pass  same fn, narrow prior -- suite asserts it PASSES
affine_only_at_zero @z=0 pinned         honest     pass      pass      pass      pass      pass      pass  genuinely affine at z=0; suite asserts it PASSES
bright_and_faint_channels(honest)       honest     pass      pass      pass      pass      pass      pass  1e17 dynamic range, NO curvature -- must still pass
roundoff_stress(big=1e+00,sig=1e-02)    honest     pass      pass    REFUSE    REFUSE      pass      pass  TRUE linear_in; offset/noise = 1e+02
roundoff_stress(big=1e+03,sig=1e-02)    honest     pass    REFUSE    REFUSE    REFUSE      pass      pass  TRUE linear_in; offset/noise = 1e+05
roundoff_stress(big=1e+06,sig=1e-02)    honest     pass    REFUSE    REFUSE    REFUSE      pass      pass  TRUE linear_in; offset/noise = 1e+08
roundoff_stress(big=1e+09,sig=1e-02)    honest     pass    REFUSE    REFUSE    REFUSE      pass      pass  TRUE linear_in; offset/noise = 1e+11
roundoff_stress(big=1e+06,sig=1e+00)    honest     pass    REFUSE    REFUSE    REFUSE      pass      pass  TRUE linear_in; offset/noise = 1e+06
roundoff_stress(big=1e+12,sig=1e+00)    honest     pass    REFUSE    REFUSE      pass      pass      pass  TRUE linear_in; offset/noise = 1e+12
roundoff_stress(big=1e+13,sig=1e+00)    honest     pass    REFUSE    REFUSE      pass      pass      pass  TRUE linear_in; offset/noise = 1e+13
roundoff_stress(big=1e+15,sig=1e+00)    honest     pass    REFUSE    REFUSE      pass      pass      pass  TRUE linear_in; offset/noise = 1e+15
roundoff_stress(big=1e+12,sig=1e-02)    honest     pass    REFUSE    REFUSE      pass      pass      pass  TRUE linear_in; offset/noise = 1e+14
roundoff_stress(big=1e+15,sig=1e-02)    honest     pass    REFUSE    REFUSE      pass      pass      pass  TRUE linear_in; offset/noise = 1e+17
quadratic_claim                          FALSE   REFUSE    REFUSE    REFUSE    REFUSE    REFUSE    REFUSE  
bilinear_pair(joint)                     FALSE   REFUSE    REFUSE    REFUSE    REFUSE    REFUSE    REFUSE  joint claim
cubic_tail(prior_std=1.0)                FALSE   REFUSE    REFUSE    REFUSE    REFUSE    REFUSE    REFUSE  curvature=1e-6
affine_only_at_zero                      FALSE   REFUSE    REFUSE    REFUSE    REFUSE    REFUSE    REFUSE  false away from z=0
tunable_curvature(1e-9)                  FALSE     pass      pass    REFUSE    REFUSE      pass      pass  boundary walk
tunable_curvature(1e-6)                  FALSE   REFUSE    REFUSE    REFUSE    REFUSE    REFUSE    REFUSE  boundary walk
tunable_curvature(1e-3)                  FALSE   REFUSE    REFUSE    REFUSE    REFUSE    REFUSE    REFUSE  boundary walk
tunable_curvature(1e-1)                  FALSE   REFUSE    REFUSE    REFUSE    REFUSE    REFUSE    REFUSE  boundary walk
bright_and_faint_observations            FALSE     pass    REFUSE    REFUSE    REFUSE    REFUSE    REFUSE  THE DEFECT: honest 1e17 leaf beside a lying faint one
faint_alone                              FALSE   REFUSE    REFUSE    REFUSE    REFUSE    REFUSE    REFUSE  the same faint node with no bright sibling
bright_and_faint_channels(lying)         FALSE     pass    REFUSE    REFUSE    REFUSE    REFUSE    REFUSE  THE DEFECT, within ONE array
nan_at_negative_probes                   FALSE   REFUSE    REFUSE    REFUSE    REFUSE    REFUSE    REFUSE  probes go NaN; caught by the finiteness branch

```

float64:

```
---- OVERALL VERDICTS (float64  (what the plan's Step 3 asks for)) ----
fixture                                  truth   global  per-elem  weighted   w-alone   per-fix   w+floor  note
straight_line                           honest     pass      pass      pass      pass      pass      pass  
two_linear_latents                      honest     pass      pass      pass      pass      pass      pass  
collinear_pair                          honest     pass      pass      pass      pass      pass      pass  
plated_latent                           honest     pass      pass      pass      pass      pass      pass  
plated_latent_through_deterministic     honest     pass      pass      pass      pass      pass      pass  
two_observations                        honest     pass      pass      pass      pass      pass      pass  2 observed leaves
two_observations_reverse_sorted_names   honest     pass      pass      pass      pass      pass      pass  
plated_and_scalar_latents               honest     pass      pass      pass      pass      pass      pass  heterogeneous member sizes
prior_held_direction                    honest     pass      pass      pass      pass      pass      pass  `loose` reaches no observed node
unconstrained_latent                    honest     pass      pass      pass      pass      pass      pass  `u` reaches no observed node
radiometer                              honest     pass      pass      pass      pass      pass      pass  sigma(mu)
plated_radiometer                       honest     pass      pass      pass      pass      pass      pass  sigma(mu) elementwise
radiometer_group                        honest     pass      pass      pass      pass      pass      pass  one obs sigma(mu), one constant
one_sided_sigma                         honest     pass      pass      pass      pass      pass      pass  
many_observations(12)                   honest     pass      pass      pass      pass      pass      pass  12 observed leaves
wide_plate(256)                         honest     pass      pass      pass      pass      pass      pass  256-element plate
indirect_ancestor                       honest     pass      pass      pass      pass      pass      pass  tau outside at its prior mean
diamond_ancestor                        honest     pass      pass      pass      pass      pass      pass  tau outside at its prior mean
shared_ancestor                         honest     pass      pass      pass      pass      pass      pass  only the PAIR is refused, not this block
overflowing_outside_latent              honest     pass      pass      pass      pass      pass      pass  at_points PINNED to z=0 (Cauchy draws overflow)
improper_outside_prior                  honest     pass      pass      pass      pass      pass      pass  at_points PINNED (improper prior has no sampler)
tunable_curvature(0.0)                  honest     pass      pass      pass      pass      pass      pass  departure exactly 0
cubic_tail(prior_std=1e-4)              honest   REFUSE    REFUSE      pass      pass    REFUSE      pass  same fn, narrow prior -- suite asserts it PASSES
affine_only_at_zero @z=0 pinned         honest     pass      pass      pass      pass      pass      pass  genuinely affine at z=0; suite asserts it PASSES
bright_and_faint_channels(honest)       honest     pass      pass      pass      pass      pass      pass  1e17 dynamic range, NO curvature -- must still pass
roundoff_stress(big=1e+00,sig=1e-02)    honest     pass      pass      pass      pass      pass      pass  TRUE linear_in; offset/noise = 1e+02
roundoff_stress(big=1e+03,sig=1e-02)    honest     pass      pass      pass      pass      pass      pass  TRUE linear_in; offset/noise = 1e+05
roundoff_stress(big=1e+06,sig=1e-02)    honest     pass      pass      pass      pass      pass      pass  TRUE linear_in; offset/noise = 1e+08
roundoff_stress(big=1e+09,sig=1e-02)    honest     pass      pass      pass      pass      pass      pass  TRUE linear_in; offset/noise = 1e+11
roundoff_stress(big=1e+06,sig=1e+00)    honest     pass      pass      pass      pass      pass      pass  TRUE linear_in; offset/noise = 1e+06
roundoff_stress(big=1e+12,sig=1e+00)    honest     pass      pass      pass      pass      pass      pass  TRUE linear_in; offset/noise = 1e+12
roundoff_stress(big=1e+13,sig=1e+00)    honest     pass      pass    REFUSE    REFUSE      pass      pass  TRUE linear_in; offset/noise = 1e+13
roundoff_stress(big=1e+15,sig=1e+00)    honest     pass      pass    REFUSE    REFUSE      pass      pass  TRUE linear_in; offset/noise = 1e+15
roundoff_stress(big=1e+12,sig=1e-02)    honest     pass      pass    REFUSE    REFUSE      pass      pass  TRUE linear_in; offset/noise = 1e+14
roundoff_stress(big=1e+15,sig=1e-02)    honest     pass      pass    REFUSE    REFUSE      pass      pass  TRUE linear_in; offset/noise = 1e+17
quadratic_claim                          FALSE   REFUSE    REFUSE    REFUSE    REFUSE    REFUSE    REFUSE  
bilinear_pair(joint)                     FALSE   REFUSE    REFUSE    REFUSE    REFUSE    REFUSE    REFUSE  joint claim
cubic_tail(prior_std=1.0)                FALSE   REFUSE    REFUSE    REFUSE    REFUSE    REFUSE    REFUSE  curvature=1e-6
affine_only_at_zero                      FALSE   REFUSE    REFUSE    REFUSE    REFUSE    REFUSE    REFUSE  false away from z=0
tunable_curvature(1e-9)                  FALSE   REFUSE    REFUSE    REFUSE    REFUSE    REFUSE    REFUSE  boundary walk
tunable_curvature(1e-6)                  FALSE   REFUSE    REFUSE    REFUSE    REFUSE    REFUSE    REFUSE  boundary walk
tunable_curvature(1e-3)                  FALSE   REFUSE    REFUSE    REFUSE    REFUSE    REFUSE    REFUSE  boundary walk
tunable_curvature(1e-1)                  FALSE   REFUSE    REFUSE    REFUSE    REFUSE    REFUSE    REFUSE  boundary walk
bright_and_faint_observations            FALSE     pass    REFUSE    REFUSE    REFUSE    REFUSE    REFUSE  THE DEFECT: honest 1e17 leaf beside a lying faint one
faint_alone                              FALSE   REFUSE    REFUSE    REFUSE    REFUSE    REFUSE    REFUSE  the same faint node with no bright sibling
bright_and_faint_channels(lying)         FALSE     pass    REFUSE    REFUSE    REFUSE    REFUSE    REFUSE  THE DEFECT, within ONE array
nan_at_negative_probes                   FALSE   REFUSE    REFUSE    REFUSE    REFUSE    REFUSE    REFUSE  probes go NaN; caught by the finiteness branch

```
