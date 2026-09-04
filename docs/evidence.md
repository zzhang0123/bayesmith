# The evidence layer: one structure class, five terms, and what a PASS does not mean

> **文档状态：`module-spec`** · 已发布模块/能力的当前设计文档，从属于顶层设计。索引见 docs/README.md。

`Z = p(d | M) = ∫ p(d | θ, M) p(θ | M) dθ` is what `EvidenceTask` answers. R4
answers it for one structure class and refuses everything else by name.

The whole difference between an evidence and a posterior is that every
θ-independent constant is invisible in the second and load-bearing in the
first. A dropped constant does not make an evidence obviously wrong; it makes
it finite, plausible, and wrong by a fixed number of nats. Every design
decision below follows from that one sentence.

---

## What it answers, and what it refuses

The admitted class is a **whole-graph-exact linear-Gaussian model**:
`classification.nuts == ()` and `classification.exact.method == "gcr"`.

| refusal | when |
|---|---|
| `evidence_residual_integral_required` | anything outside that class — the model leaves a residual problem needing numerical integration, which is R5's subject |
| `evidence_prior_proper` | a prior whose mass diverges, or whose mass this release cannot resolve |
| `evidence_prior_normalised` | a prior with finite mass that is not one — `Z` converges, to that factor away from the number a Bayes factor compares |
| `evidence_prior_undeclared` | a latent covered by the graph-level reference prior, which is a declaration rather than a normalised density |
| `evidence_requires_x64` | a float32 environment |
| `task_options_recognised` | `repeat_count` or `reconstruct_posterior`, which this release reads and does not honour |

**Every one of these is an asymmetry, and each is asserted beside its refusal:
the same graph still compiles a posterior task unchanged.** §2.2 of the
top-level design names that asymmetry as the reason compilation is task-aware
at all — an improper prior leaves the posterior perfectly well defined and the
evidence undefined. A refusal that also broke the posterior would be a
regression wearing a boundary's name, and only the second assertion tells them
apart.

---

## The answer is five terms, not a number

`EvidenceResult.exact_components` carries five `EvidenceComponent`s. The
decomposition is the determinant lemma written out, with `C = A S Aᵀ + N` and
`F = Aᵀ N⁻¹ A + S⁻¹`:

| component | value |
|---|---|
| `data_log_normaliser` | `−½ logdet(2π N)` |
| `residual_quadratic` | `−½ rᵀ C⁻¹ r` |
| `prior_log_normaliser` | `−½ logdet(2π S)` |
| `integral_log_two_pi` | `+k/2 log 2π` |
| `block_log_determinant` | `−½ logdet F` |

The third and fourth do **not** cancel. Only their `2π` halves do; what remains
is `−Σ log sⱼ`, the declared prior widths, which is exactly zero at unit width
and nowhere else — `−0.5306` at `s = 1.7`, `−12.2830` at `k = 3, s = 60`. An
earlier draft of this page's source said they cancel exactly, which is what
unit priors look like, and it contradicted a test in its own commit.

**`log_evidence` is not defined as the sum of the components.** That identity
survives a dropped term, because dropping one removes it from both sides at
once. The total comes from the closed form and the equality is an assertion.

---

## What checks it

Three routes, and the distinction between them is the point of §9.1.

1. **The hand derivation** (`_by_hand` in `tests/dispatch/test_evidence.py`)
   rebuilds the design from the model's own parameters. This is the check §9.1
   asks for.
2. **The square-root information route** (`marginal_log_density`) shares every
   upstream seam with the assembler — `unchecked_operator`, `precision_at`,
   `dense_operator`, `observed_descendants` — so agreeing with it is **not** an
   independent derivation. Measured: scaling `dense_operator`'s return by 1.03
   leaves the cross-route test completely blind while the hand derivation fails
   five cells. It is kept because it catches everything downstream of the
   shared design, which is where the constants live.
3. **Quadrature of the graph's own `log_joint`** is the genuinely third route.

Every sweep covers `{0.05, 0.25, 1.0, 4.0, 60.0}` prior widths, and that is not
decoration: deleting `prior_log_normaliser` kills four of those five cells and
**leaves `1.0` passing**, because `−Σ log s` is identically zero there. A
fixture family that never leaves unit priors reports a missing constant as
green, which is how it shipped missing once already.

---

## The prior audit, and what it cannot decide

`audit_prior` integrates the declared density over a sequence of widening
windows and reads the **ratio of successive increments**. A flat density
doubles its mass with its window (ratio exactly 2); a `1/x` tail adds a
constant `log 2` (ratio exactly 1); every convergent density falls below 1.
That is a property of the density and carries no units.

A ratio rather than a difference, because a difference does. An earlier version
compared `|mass(2W) − mass(W)|` against `1e-6 · max(mass, 1)`, and the absolute
floor made the verdict depend on what the parameter was measured in: the same
flat improper prior was IMPROPER in metres and PROPER in nanometres.

**It abstains rather than guessing**, and the list is not short:

| UNVERIFIABLE when | why |
|---|---|
| the prior has non-scalar shape | the rule is one-dimensional |
| the support is discrete | the mass of a PMF is a sum, not an integral |
| the quadrature has not converged | `Gamma(0.5, 1)`'s `x^-0.5` pole is still 1.2% unresolved at 64× the panels |
| the window finds essentially no mass | "zero here" and "the window is elsewhere" look identical from inside the integral |
| the latent has parents | a hierarchical prior is not a fixed density; `p(w)` exists only once `s` is integrated out |

That last row is why six of this package's own shipped fixtures used to crash
the audit with `KeyError`. A standing test now runs it over every fixture in
`tests/exact/models.py` and requires that it RETURNS for each — a crash is not
a verdict.

---

## What a PASS does not mean

* **`evidence_normalization_audit` PASS does not mean the model is right.** It
  means every declared prior has finite mass and that mass is one. A perfectly
  normalised prior on a badly wrong model gives a perfectly well-defined
  evidence for that wrong model.
* **`evidence_prior_sensitivity` PASS does not mean the evidence is
  insensitive.** R4 **reports** `d log Z / d log s` and does not threshold it:
  no derivation for a criterion exists, and the only prior-sensitivity number
  this package has is derived from a chain's MCSE of a posterior mean, which
  has nothing to do with nats of log evidence. A PASS here means the slope was
  measured. **Read it.**
* **`evidence_comparability` PASS does not mean the models are comparable in
  any wider sense.** It means the DATA fingerprint is unchanged, so their
  difference is a log Bayes factor rather than `p(d₁|M₁)/p(d₂|M₂)`, which
  compares nothing. Whether the two models are the two you meant to compare is
  not a question a fingerprint can answer.
* **An exact assembly reports `standard_error = None`, not `0.0`.** None was
  computed; that is different from one measured and found to be zero, and a
  Bayes-factor consumer has to branch on which.

`evidence_prior_sensitivity` is a **different report kind** from R3's
`prior_sensitivity` and the difference is measured, not bookkeeping. R3's is a
posterior mode displacement in posterior sigmas: perturb a prior's width and
leave its centre alone, and R3's report reads zero while `log Z` moves half a
nat. Two perturbations, and only one of them is the one an evidence is
sensitive to.

---

## The numbers

| ID | Name | Value | Provenance |
|---|---|---|---|
| D107 | dense-agreement band | `C · n · eps · max(κ₂, \|log Z\|)`, `C = 1.1336` | **derived form, measured coefficient** |
| D109 | `_CONVERGENT_INCREMENT_RATIO` | `0.995` | **derived** |
| D110 | `_NORMALISED_TOLERANCE` | `1e-6` floor, plus the extrapolation's measured error | **derived** |

**D107's form changed because the measurement said so.** The R4 plan derived
`C · n · eps · κ₂` from backward stability of the QR route. Measured over a
60-cell grid, the worst cell is well-conditioned — `κ₂ = 1.04` — where the
bound is `9.2e-16` and the gap is `1.2e-14`, about thirteen ULP of a log
evidence whose *magnitude* is 10. Conditioning explains none of it, because a
sum of logs carries absolute error proportional to its largest term. Four forms
were measured; `n · eps · max(κ₂, |log Z|)` is the only one whose coefficient
is `O(1)` (1.134, spread 60×) rather than 13 to 165 with spreads of 400–700×,
and a form needing a coefficient of 165 is a form doing the wrong thing with a
fudge factor on top.

**Both platforms measure `C = 1.1336`.** macOS/Accelerate and
`linux/amd64` + `OPENBLAS_CORETYPE=ZEN` (numpy 2.5.2, scipy-openblas, QEMU TCG,
`fma avx avx2`, no `avx512f`) agree to five digits, well inside the
factor-of-four rule the plan set. Re-measure with
`docs/probes/probe_33_d107_agreement_band.py`.

**D108 was reserved and not consumed.** The declared-prior-scale domain became
a `NotImplementedError` at the point the width is read, not a registered
threshold, so it needs no gate. **D109 and D110 were reserved with expected
consumption zero and were consumed** — both are the prior audit's, and both
appeared during execution rather than in the plan.
