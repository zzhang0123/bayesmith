# Cross-check: `prior_sensitivity`

`rheplicant.inference.sensitivity` → `bayesmith.diagnose.sensitivity`
(migration spec §八 step 5; §三 B3 fixed in the same pass, as ordered).
Executed 2026-08-25.

## 1. Fixtures

- **Healthy**: the tour — rheplicant's full radio twin (noise waves,
  bandpass, time-varying gain, 12-bit ADC) with the nonlinear
  `(fg_log_amp, fg_beta)` pair. The bayesmith side wraps rheplicant's own
  `space.forward_fn` in a `det` node and conditions on the same observed
  ARRAY, so the two packages share the model bit for bit and every
  disagreement is the diagnostic's. Requires `rhino-cal-jax` (see
  pyproject's crosscheck group); the fixture skips loudly without it, and
  a skip is not a pass.
- **Must-refuse**: ImproperUniform/Uniform selected priors; a selection
  whose priors move with it (`child ~ Normal(parent, s)`, both selected);
  the `exp(a+b)` rank-deficient pair; unknown names; ambient float32.
- Native (`tests/diagnose/`): a noisy 8-channel power law, a starved
  affine model, the θ² saddle model, a vector latent, the hierarchy.

## 2. Numerical agreement (`tests/crosscheck/test_diagnose_sensitivity.py`)

Pinned both against the design-phase constants AND against rheplicant's
live report — the constant catches the pair drifting together, the live
comparison catches either drifting alone:

| quantity | value | agreement |
|---|---|---|
| mode | (7.824320989, 2.553069844) | rel 1e-8, and field-for-field vs live rel 1e-6 |
| shift_sigma | (+0.0024711038, −0.0069239167) | rel 1e-5 |
| sigma_post | (2.9775575e-4, 2.4990616e-3) | rel 1e-6 |
| criterion_std (beta) | 0.0795 | rel 1e-3 |
| the seven-row s-ladder via `shift_at` | design column | rel 1e-3, abs 5e-7; last rung rel 3e-3 (the tour's own nonlinearity over six sigma, measured upstream at 1.8e-3) |
| `verified` | all true | — |

## 3. Refusal agreement, and the two that die structurally

Same-shape refusals: non-quadratic prior (named, with the NUTS/`names=`
way out — `NotGaussian` here), rank-deficiency (named via
identifiability's shares), non-convergent MAP (**`ConvergenceError`**
here — the precise class exists), report-query errors (`GraphError`).

Two rheplicant refusals are UNBUILDABLE here and exist only as a sentence
in the docstring: "no prior at all" (every `sample` node carries a
density) and "`linear=True` with a call-site `prior_std=`" (the graph is
the single statement of the model; there is no call-site prior to
diverge). This is iron law 3 working as intended — the error class was
removed by construction, not ported.

## 4. Independent oracles

- An in-file plain undamped Newton (no backtracking, no eigvalsh, no code
  shared with the module) lands on both modes to rel 1e-9 and reproduces
  the refit column to rel 1e-7.
- `shift_at(name, declared)` collapses onto `shift_sigma` to rel 1e-12 —
  an algebraic identity of the anchored counterfactual; the dropped-anchor
  mutation is killed by exactly this.
- `shift_at` vs an actual re-run at the hypothesised width (rebuilt graph,
  second solve): rel 1e-3 at s=0.1, 1e-2 at 0.03.
- The starved affine fixture separates `H^{-1}` from `(H+P)^{-1}` by
  **57%** (`diag((H+P)^{-1}P)` = 0.57), against a closed-vs-refit floor of
  ~4e-16 — the wrong-matrix mutation dies loudly, where a data-rich
  fixture would hide it.
- 7 of the 17 mutation-pass mutations targeted this module; all killed.

## 5. B3, fixed here

`_descent_direction`: `eigvalsh` decides; a positive-definite Hessian
keeps the exact Newton step (pinned bitwise); otherwise
Cholesky-with-jitter, with the shift `2|λ_min| + 1e-8·scale` — reflecting
the most negative eigenvalue rather than parking it just above zero, which
returns a 2e7-length lunge the line search must tame (measured on the 2×2
pin). The θ² saddle model (likelihood curvature negative at the start)
walks out to its mode where the raw solve steps toward the saddle. The Δ
direction statement is rewritten unambiguously in the module docstring:
`theta_hat − theta_L = H^{-1} P (m − theta_hat)`, positive = pulled UP.

## 6. Intended differences

1. **Data and noise come from the graph** — no `observed`/`noise_std`/
   `flags` parameters. Nothing separate can disagree with what a sampler
   reads.
2. **The rank refusal's verdict moved from the observed Jacobian to the
   rest term's own curvature**, because a graph can hold a selected latent
   through a DOWNSTREAM latent's density (`child ~ Normal(parent, s)`,
   child unselected) — the likelihood-only mode then exists although the
   observed Jacobian is rank 0. rheplicant's flat structure cannot express
   this, so its Jacobian test was exact there and wrong here. The ceiling
   is `condition_ceiling`'s (fisher's own constant); identifiability still
   names the direction when the two verdicts agree. The hierarchy fixture
   pins both halves.
3. **"The likelihood"** generalises to "every probabilistic term except
   the selected latents' own densities" — downstream latent densities
   included, verified by an independent spelling of the objective's
   curvature in the suite.
4. A selection entangled with its own priors (parent and child both
   selected) is a NEW refusal with no rheplicant counterpart — the
   structure that makes it possible does not exist there.
