# Cross-check: `JeffreysPrior`

`rheplicant.inference.priors` → `bayesmith.diagnose.priors` (migration spec
§八 step 5, acceptance §四 4.2). Executed 2026-08-25.

## 1. Fixtures

- **Healthy**: the bare power law `mu = A (nu/nu0)^-beta` over
  `(log A, beta)`, 8×8, radiometer-style noise `Normal(mu, 1e-3·|mu|)` —
  matching `RadiometerNoise(channel_width=1e6, integration_time=1)` — and
  its floored (300 K) and constant-sigma variants.
- **Must-refuse**: the doubled amplitude `exp(a+b)` (exactly singular
  block); a covered latent carrying a proper density; `over` naming a
  non-latent; missing values entries; empty/repeated/non-string `over`.

## 2. Numerical agreement (`tests/crosscheck/test_diagnose_jeffreys.py`)

| quantity | value | agreement |
|---|---|---|
| flat constant, nine grid points | +15.80169853 | vs rheplicant live abs 1e-8, vs the pinned constant abs 1e-8, vs a numpy-only closed form abs 1e-8 |
| floored `d/dβ`, radiometer | −1.366854e-02 | vs live rel 1e-9 |
| floored `d/dβ`, constant σ | +8.052944e-03 | vs live rel 1e-9, opposite sign confirmed |
| singular block, eigh+floor | ≈ −338 | vs live abs 0.5 (both effectively zero density) |
| slope in log A, constant σ | +2.000000 | abs 5e-7, midpoint on the line |
| reparameterisation invariance (β ↔ u = log(β−2)) | — | abs 1e-11 |

The singular block's *plausible-liar* numbers also reproduce to the digit
(`slogdet` +6.420496, `cholesky` min pivot 9.755e-05): the reason the
`eigh` + rank-floor discipline is kept, and must not be replaced by either
routine — on ill-conditioned blocks they are precisely the routines that
cannot say no.

## 3. Refusal agreement

Same-shape: empty/repeated/non-string `over`, non-positive `rank_rtol`,
unknown latent (block would silently shrink), missing values entry,
rank-deficient block (delegated to identifiability, direction named
0.50/0.50). The "two priors on one quantity" refusal changes spelling:
rheplicant checks `Latent(prior=...)` against `over`; here every latent
carries a density, so the flat spelling is `ImproperUniform` and anything
else on a covered latent is the double-count — checked inside
`information()`, where distribution types are static under tracing.

## 4. Independent oracles

- The numpy-only closed form `(1+2f²)/f² · GᵀG` (no autodiff, no shared
  code) equals the flat constant.
- The `depends_on_prediction` gate is pinned by arithmetic: skipping the
  variance term costs exactly `log(1+2f²)` — a gate that skipped nothing,
  or the wrong half, cannot produce that number.
- Reparameterisation invariance is a property of the mathematics, not of
  either implementation.
- 4 of the 17 mutation-pass mutations targeted this module; all killed
  (unfloored determinant, dropped variance term, vacuous double-prior
  check, ignored `rank_rtol`).

## 5. Intended differences

1. **The noise comes from the graph**, not from an exit argument — the
   "prior inherits the exit's noise" contract is now structural, and the
   likelihood/prior noise-mismatch class is unbuildable. Consequence: the
   "same prior, two noise models" tests compare two GRAPHS differing only
   in the observed node's `dist_fn`.
2. **`information()` keeps `over`'s own row order.** rheplicant's fisher
   flattens sorted, and its docstring must warn that row 0 is not the
   first name passed; the graph machinery preserves caller order, so the
   wart does not port. Pinned: reversing `over` transposes the matrix, and
   row 0's diagonal is the first name's `(1+2f²)/f²·n_data`.
3. **The assembly bypasses `fisher_information`'s eager centre check.**
   That check reconciles a decided precision against a rule at a claimed
   centre — redundancy that exists because its callers hold the noise in
   two places. Here both the weighting and the curvature are read from the
   one graph at the one values dict, so there is no second spelling to
   reconcile, and the check's concrete-value comparison would break the
   jit path NUTS differentiates through. The pieces themselves
   (`dense_operator`, `_weighted_design`, `_log_spectrum_curvature`) are
   imported, not respelled.
4. **The consumer, today**: a NumPyro model with flat sites plus
   `numpyro.factor("joint_prior", prior.log_density(graph, values))`,
   demonstrated in the suite with the density difference pinned at two
   parameter points. Declaring a joint prior ON the graph and honouring it
   in `to_numpyro` is the `numpyro_bridge` row of §四 4.2, deliberately
   not part of step 5; when that lands, `check_identified` becomes its
   build-time gate the way `to_numpyro_model` uses it upstream.
5. rheplicant's `validate_against` (space-declaration-time check) has no
   hook here until the bridge integration exists; its two refusals live in
   `information()`/`check_identified` instead, so they run at first
   evaluation rather than at declaration. Recorded as a difference, not an
   omission: there is no declaration site to hang them on yet.
