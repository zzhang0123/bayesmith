# Cross-check: `gls` — and the B1 log-determinant ledger

`rheplicant.inference.gls` (`iterative_gls`) → `bayesmith.exact.gls`
(§四 4.1). Test: `tests/crosscheck/test_noise_logdet.py` (17 cases, shared
with the Fisher row — see [uncertainty.md](uncertainty.md)) plus
`tests/exact/test_gls.py::test_the_fixed_point_is_the_unbiased_estimator_
not_the_gls_biased_one` on this side. Page written 2026-08-25 from those
tests' assertions, re-run on that date; the cross-checks predate this page.

> **CORRECTION, 2026-08-28 (Wave B).** `rheplicant.inference.gls.iterative_gls`
> **now delegates to `bayesmith.exact.gls`**, so the reweighting loop is no
> longer two implementations. Two refusals stayed upstream and did not move:
> the `NoiseModel` requirement, because the far side takes a `sigma_of`
> callable and cannot recognise one past the seam, and the
> `min_reweights <= max_reweights` bound, because the far side carries that
> sentence word for word but raises `GraphError` where iron law 1 keeps
> `ParameterSpaceError`.
>
> **D19's deferred half discharged with it: the start does not move.** Measured
> against rheplicant's own pinned observables, the two seeds -- sigma at the
> data against sigma at the prior centre, a 9.2% gap -- give the same
> `iterations` (4) once the floor is lowered far enough for that number to be
> a convergence count rather than `MIN_REWEIGHTS`. Evidence:
> `docs/superpowers/specs/2026-08-28-wave-B-gls-opening.md` §5.
>
> §5 below is unaffected: it is about B1 and the likelihood, not about this
> loop -- see the reading note there.

**This page is unusual: its subject is a difference, not an agreement.**
§三 B1 is a defect in rheplicant, and iron law 5 forbids aligning a new
implementation to a defective one. So the acceptance is a *signed, sized*
expected difference on one side and a proof of correctness on the other.

## 1. Fixtures

A constant-mean model under `RadiometerNoise`, so both estimators have a
closed form and nothing samples: `f ∈ {0.05, 0.2, 0.5}`, `n = 2000` for the
closed-form rows and `n = 200 000` for the asymptotic one (measured: at
f=0.5 the ratio lands 0.024 % from `1+f²` at n=200 000, against 2.5 % at
n=2000). A `HomoscedasticNoise` row is the anti-vacuity control.

## 2. Numerical agreement — of each estimator with its own closed form

| claim | agreement |
|---|---|
| `include_logdet=False` maximiser = `Σd²/Σd` | rel **1e-8** |
| `include_logdet=True` maximiser = positive root of `n f² μ² + μ Σd − Σd² = 0` | rel **1e-8** |
| ratio of the two = `1 + f²` | rel **2e-3**, at three f |
| the full density's estimate = the truth | rel **5e-3** |

The quadratic's root is derived in the test rather than taken from
rheplicant's docstring, because the asymptotic claim rests on it: it is
what makes the full density **exactly unbiased** rather than differently
biased, and a test asserting only "closer to the truth" would pass on an
estimator that was merely less wrong.

The maximiser is found by **scipy on a Python closure** over rheplicant's
likelihood — no JAX gradient, no algebra of ours. A closed form checked
against its own rearrangement would be checking arithmetic, not the
estimator.

## 3. Refusal agreement

None to compare: neither package refuses the mixture. That is the defect —
B1's point is that rheplicant's `nuts` route (whose `dist.Normal` carries
its own `−log σ`) and its `plan.sample` gradient block (which does not)
target different estimators on one model, **with no guard between them**,
while the same package's evidence layer raises full-versus-GLS to a
refusal.

## 4. Independent oracles

- The two closed forms above (algebra, not either implementation).
- scipy's optimiser, sharing nothing with either package's gradients.
- **Anti-vacuity**: under `HomoscedasticNoise` the log-determinant is an
  additive constant and the two estimators must **coincide** — so the gap
  is attributed to the prediction-dependence and not to some other
  difference between the two likelihood spellings.
- The ratio is checked across an f that varies the answer by a factor of
  25 between its ends, so a constant fudge cannot satisfy it; and the SIGN
  is asserted separately, because it is the half that says which engine is
  the optimistic one (`gls > full > 0`).

## 5. Intended differences — the whole point of the row

> **Reading note, 2026-08-28.** The subject of the first sentence below is
> **bayesmith**, and it is positioning bayesmith against B1 -- it is *not*
> a contrast in which the near side carries the bias and the far side does
> not. Measured on the Wave B `gls` opening: rheplicant's `iterative_gls` is
> **also** frozen-sigma IRLS and also lands on `mean(u)`, at f = 0.05/0.2/0.5
> by factors of 16785x/89861x/424500x nearer `mean(u)` than `sum(u^2)/sum(u)`.
> B1 lives in the LIKELIHOOD's log-determinant -- the `nuts` route against the
> `plan.sample` gradient block, §3 above -- and belongs to the `plan`/`engines`
> row, which is what this section's own last sentence says. Nothing here is
> wrong; the note exists because the heading plus that subject read as a
> two-sided comparison at speed, and one reader has already lost time to it.
> Evidence: `docs/superpowers/specs/2026-08-28-wave-B-gls-opening.md`.

**bayesmith's `iterative_gls` does NOT carry this bias, and must not be
made to.** It is frozen-sigma IRLS: each inner solve holds σ fixed and
recomputes it afterwards, so its fixed point satisfies `w = mean(u)`,
`u = d/x` — the same side as the full density. Measured: the fixed point is
44–128× closer to `mean(u)` than to `Σu²/Σu`, at `κ ∈ {0.05, 0.2, 0.5, 1.0}`.

The spec's first draft asked for the opposite — that bayesmith's frozen-σ
path differ from a live-σ path by `(1+f²)` — and building that test would
have pulled a correct estimator toward a bias it does not have. **The
initial statement of the acceptance was wrong, and measurement is what
found it**; the correction is recorded in §三 B1 itself, marked
`[实测确认]`.

Consequence for the pending `plan`/`engines` row (§四 4.2): B1 must land
first, or that comparison will fix the GLS-type target as the reference.

> **B1 is closed, 2026-08-28** — and the ordering constraint above was
> discharged twice over, in opposite ways, which is worth separating.
>
> The `plan`/`engines` row ran **before** B1 landed and its
> `docs/migration/plan.md` §5(a) records why that was safe: a non-linear
> graph has no exact subgraph here, so `estimate()` refuses and `sample()`
> goes through NumPyro's own `-log sigma`. There is no second place in this
> package that could drop the term, so the comparison could not have fixed
> the wrong reference — *not because it was careful, but because one side of
> it does not exist*. That argument never depended on B1 and does not expire
> with it.
>
> B1 then landed in e-RHINO's `74fac09`:
> `Conditioning.neg_log_likelihood` is `0.5 * chi2 + log_determinant`, given
> to **both** potential builders. The gradient block moved from **6.2483** to
> **5.0041** against an unbiased closed form of 5.1046. What this page says
> about `iterative_gls` is unchanged — it was never the biased side, and
> nothing was done to it.
>
> One adjudication came with the fix and is registered as **D55**: a plan
> exit refuses `inference.noise.include_logdet: false` rather than silently
> overriding it, because GLS is a point estimator and is not a posterior.
