# Campbell sky validation: GCR reweighting and analytic cumulants

> **文档状态：`record`** · 2026-09-15 的可复现实验：小型相关天空图，解析累积量、完整后验重加权与 Edgeworth 失效反例。

## Question and model

Can cheap Gaussian posterior draws recover the marginal likelihood and
non-Gaussian hyperparameter of a sky with independently known cumulants?

The example uses a 12 x 12 periodic Cartesian map. At each source-grid cell,

\[
N_j\sim\mathrm{Poisson}(\lambda),\quad
x_j=\sqrt{v_s/\lambda}(N_j-\lambda)+g_j,\quad
g_j\sim\mathcal N(0,v_g),\quad s_i=\sum_j B_{ij}x_j.
\]

Here `v_s=0.8`, `v_g=0.2`, and the positive, unit-sum beam has center weight
0.6 and weight 0.1 on each of four periodic neighbors. Its eigenvalues are
strictly positive, so the coordinate transform loses no modes. This is a
mean-subtracted sky of identical positive-amplitude sources on grid sites,
plus an independent Gaussian diffuse component. The latter gives a continuous
prior: a discrete Poisson count law alone cannot be reweighted from a
continuous Gaussian proposal by an ordinary density ratio.

**Every candidate rate has the same mean and covariance.** The per-source
amplitude changes as `sqrt(v_s/lambda)`, so `E[x]=0`, `Cov(x)=I`, and
`Cov(s)=B B^T` for every rate. Recovery therefore uses higher-order information;
it cannot be obtained from a rate-dependent mean or power spectrum.

The Poisson cumulant generating function is

\[
\log E[e^{t^Ts}]=\lambda\sum_j
\{e^{a(B^Tt)_j}-1-a(B^Tt)_j\}
+\tfrac12v_g\lVert B^Tt\rVert^2,\qquad a=\sqrt{v_s/\lambda}.
\]

Differentiating gives the discrete Campbell formula, for `r>=3`,

\[
K_{i_1\cdots i_r}=\lambda a^r\sum_j B_{i_1j}\cdots B_{i_rj}.
\]

At order two the multiplier is `lambda*a^2+v_g=1`; order one vanishes.
Neighboring map pixels share nonzero higher connected cumulants. The general
Poisson-functional and shot-noise construction is described in
[Last & Penrose, *Lectures on the Poisson Process*, sections 3.3 and 15.4](https://stoch.math.kit.edu/img/Last/lastpenrose2017.pdf).

## Independent oracle and deliberate simplifications

The observation is `d=B(x+epsilon)`, with independent coefficient noise
`epsilon ~ N(0,0.1^2 I)`. Thus the **map noise is correlated**, with covariance
`0.01 B B^T`. Applying the known inverse beam gives independent coefficient
observations `y=B^-1 d=x+epsilon`. This convenient noise law is chosen for a
controlled oracle, not as a claim about a realistic instrument.

Conditional on a source count, both the coefficient and its observation are
Gaussian. Therefore the evidence is a product of analytically convolved
Poisson–Gaussian sums:

\[
p(y\mid\lambda)=\prod_j\sum_{k=0}^\infty
\mathrm{Pois}(k;\lambda)\,
\mathcal N\!\left(y_j;\sqrt{0.8/\lambda}(k-\lambda),\,0.2+0.01\right).
\]

The independent oracle uses NumPy/SciPy; the target graph uses NumPyro and
JAX. Direct numerical integration over a latent coefficient independently
checks the convolution and conditional posterior mean. In map coordinates
there is also a fixed `-log|det B|` term; it cancels in every evidence ratio
and changes neither the hyperparameter mode nor its interval.

The source generator uses the full Poisson law. Density evaluation conditions
the count on `k<=48` and normalizes the finite mixture. For all allowed rates
`[0.5,8]`, omitted count probability is at most `1.171e-22` per cell, or
`1.686e-20` by a union bound over the whole map. Doubling the cutoff to 96
changed none of the evaluated oracle log evidences at reported double precision.
Analytic cumulant formulas refer to the untruncated generative law.

## Recorded run

Run:

```bash
.venv/bin/python -m examples.inference.campbell_sky_validation
```

Defaults: simulation seed `20260915`, true rate `2.0`, Gaussian draw seeds
`71` and `72`, 8192 draws per bank, 220 Adam steps from rate `3.0`. The
reference graph compiles to fixed-covariance **GCR**, with no NUTS or
importance correction inside the reference sampler. The final objective is
`logmeanexp(full target joint - full reference joint)`, with a uniform
hyperprior in rate on `[0.5,8]`. Optimization uses a logit coordinate without
adding a density Jacobian. Inference ran with ambient JAX float32; NumPy/SciPy
oracle calculations used float64.

| Quantity | Bank A | Independent bank B | Analytic oracle |
|---|---:|---:|---:|
| Rate MAP | 1.30044 | 1.30010 | 1.30145 |
| Kish ESS / 8192 | 5113 / 8192 | 5058 / 8192 | — |
| Largest normalized weight | 0.000881 | 0.001328 | — |
| Log evidence ratio at each fitted rate | 3.77861 | 3.79072 | 3.78440 |
| Coefficient posterior-mean RMSE against its analytic target | 0.001302 | 0.001284 | — |

**This verifies the numerical inference, not precise recovery of the injected
rate from one small sky.** The analytic 95% equal-tail interval is
`[0.9589, 7.6140]`, which contains the injection `2.0`. Even exact integration
has a MAP near 1.3 on this realization. Its broad upper tail depends on the
stated finite hyperprior bounds. Across 24 separately generated skies (seeds
1000–1023), 22 oracle intervals contain the fixed injected rate and some MAPs
hit the upper bound. This auxiliary run measures finite-sky variation only;
it is not a calibration test of the reweighting algorithm or a guarantee of
95% frequentist coverage.

## Campbell cumulants

The separate simulation uses 98,304 independent local sky realizations,
divided into 48 batches, to estimate unbiased single- and mixed-pixel
cumulants. `0` and `1` denote adjacent pixels. Reported errors are one
Monte Carlo standard error from independent batches.

| Cumulant | Campbell prediction | Simulation | Standard error | Difference / SE |
|---|---:|---:|---:|---:|
| `K00` | 0.400000 | 0.398607 | 0.002096 | -0.665 |
| `K000` | 0.111312 | 0.107919 | 0.002391 | -1.419 |
| `K0000` | 0.041600 | 0.039021 | 0.004307 | -0.599 |
| `K001` | 0.021251 | 0.020245 | 0.001334 | -0.754 |
| `K0001` | 0.007104 | 0.005448 | 0.001881 | -0.881 |

These checks include connected spatial dependence, not just one-pixel
skewness. A separate derivative test differentiates the cumulant generating
function through order four.

## What happens to Edgeworth?

The successful reweighting above uses the **complete positive Poisson–Gaussian
mixture**, not a truncated cumulant density. The same known cumulants expose
a failure of the order-four approximation.

For one coefficient at the true rate, the scalar Edgeworth correction reaches
`-0.52524` at standardized coefficient `-3.18093`. It is not a proper prior.
For the 144-dimensional **joint** order-four expansion, a still simpler
counterexample is the zero field. With variance one, `H3(0)=0`, `H4(0)=3`,
and `H6(0)=-15`, so the full correction is

\[
1+144\left(\frac{\kappa_4}{8}-\frac{5\kappa_3^2}{24}\right)
=1-\frac{3.84}{\lambda}=-0.92\quad (\lambda=2).
\]

The executable audit evaluates this witness using the actual `FieldEdgeworth`
and `LowRankCumulants` implementation. The fast independent-coordinate formula
also includes cross-cell `K3*K3` terms and is tested against `FieldEdgeworth` on
a smaller map; it is **not** the product of scalar truncated expansions.

On bank A the joint corrections at the injected rate happen to all be
positive. That does not rescue the approximation: the explicit zero-field
witness lies elsewhere. No signed weights were clipped, dropped, absolutized
or used in the fitted evidence. This experiment supports the full-density
reweighting method and shows why density validity must be checked separately
from importance-weight ESS.

## Files and scope

- Model and oracles: `examples/inference/campbell_sky.py`.
- Runner, diagnostics and plots: `examples/inference/campbell_sky_validation.py`.
- Independent tests: `tests/test_campbell_sky.py`.
- Generated report, figure and arrays: `runs/campbell-sky/results.json`,
  `campbell_sky.png`, `campbell_sky.pdf`, `maps_and_profiles.npz`.

The original map above is small, all beam modes are retained, and observation noise is low.
A broader beam with independent detector-pixel noise, missing modes, weaker
observations, or a larger correlated field would need a new overlap and
performance measurement. The current result does not establish those cases.


## Large-field follow-up and notebook chapter (2026-09-15)

The follow-up uses **128 × 128 = 16,384 source coefficients**, with two banks
of **2,048 independent Gaussian conditional draws**. The generative model,
noise level and rate bounds are unchanged. Separately, the draw benchmark
reaches **256 × 256 = 65,536 coefficients**, 1,024 draws at a time.

This benchmark uses the diagonal closed-form Gaussian conditional directly:
`mean=y/1.01`, `variance=0.01/1.01`. It is a special case of the linear/GCR
sampler, not a claim about large-graph compiler performance. The previous
small-field experiment still exercises the actual compiler and graph-based
`PosteriorReweighting` API. Regression tests equate the streamed whole-map
score to the graph API's full-posterior score on the same draws.

### Two estimators, explicitly distinguished

Write `r_mj = p_lambda(x_mj)/p_0(x_mj)`. Whole-map reweighting estimates
`logmeanexp_m(sum_j log r_mj)`. Since **both** q and the target factorize in
source coordinates for this observation model, the same evidence can also
be estimated by `sum_j logmeanexp_m(log r_mj)`. These estimators differ at
finite sample size. The second exploits exact independence, retains a
finite-sample log bias, and cannot be used for arbitrary coupled fields.
The implementation streams 64 cells per mixture evaluation, without an
n-by-n covariance or a dense cumulant tensor. Selected beam rows also avoid
an n-by-n allocation in the independent Campbell cumulant audit.

### Recorded results

Injected rate **2.0**; analytic marginal MAP **2.001527**; analytic 95% equal-tail interval **[1.803994, 2.273959]**.

| Bank | Factorized MAP | Whole-map MAP | Whole-map ESS at injection | Largest whole-map weight | Minimum local ESS at fit | Factorized log evidence error at fit |
|---|---:|---:|---:|---:|---:|---:|
| 71 | 2.001428 | 2.063131 | 6.235 / 2048 | 32.038% | 1614.468 / 2048 | +0.106941 |
| 72 | 2.001053 | 2.054367 | 2.421 / 2048 | 61.308% | 1605.198 / 2048 | +0.123581 |

| Map | Scalar draws | First call (s) | Warm median (s) | Bank memory (MiB) |
|---|---:|---:|---:|---:|
| 16 × 16 | 262,144 | 0.1615 | 0.0019 | 1 |
| 64 × 64 | 4,194,304 | 0.1258 | 0.0073 | 16 |
| 128 × 128 | 16,777,216 | 0.1626 | 0.0265 | 64 |
| 256 × 256 | 67,108,864 | 0.2296 | 0.1050 | 256 |

Every timing waits for JAX completion. Cold includes compilation, warm is the
median of three calls with different keys; the source coefficient draws are
timed, excluding beam convolution of sample maps. The saved environment is
JAX CPU with float32 computation. No speedup ratio against NUTS was measured.
The main experiment takes 0.224 s for its first bank (including compilation) and 0.056 s for its second. Evaluating the 61-point profile plus refining both MAPs takes 62.82 / 62.77 s, plus 1.47 / 1.55 s for first score compilation/evaluation. The independent analytic 401-point profile and mode take 4.71 s; for this special model the oracle is the simplest inference method.

The conclusion is specific: Gaussian draws are cheap, but the whole-map
importance estimator has poor overlap at this dimension. Exact model
factorization restores accurate marginal MAP here. Neither the small local
ESS losses nor agreement between two banks certify arbitrary-field accuracy.
Campbell's five local cumulants are again checked against 98,304 independent
local realizations; all discrepancies are below 1.42 estimated standard errors.

### Reproduction and presentation

```bash
.venv/bin/python -m examples.inference.campbell_sky_scaling --output runs/campbell-large-128
.venv/bin/python -m examples.inference.campbell_notebook --input runs/campbell-large-128 --notebook runs/inference-demo-verified
```

The existing inference notebook now has **07 · Poisson sky & Gaussian draws**,
with the same five stages, language switching and hash navigation as the six
original demos. It explains Campbell cumulants, Gaussian conditioning, joint
density ratios, marginal hyper-MAP, weight ESS and exact factorization.
The source model/runner hashes and measurements live in `result.json`;
matching executed-source snapshots are retained in the run’s `source/` directory.
The plotting code is separately maintained in `campbell_plot.py`;
`presentation.json` separately hashes the renderer and saved input bundle.
Map figures label the analytic posterior mean explicitly. Other demos'
numerical artifacts are reused, not recomputed.

- Numerical runner: `examples/inference/campbell_sky_scaling.py`.
- Bilingual chapter: `examples/inference/campbell_presentation.py`.
- Notebook installer: `examples/inference/campbell_notebook.py`.
- Additional tests: `tests/test_campbell_scaling.py`.
- Results and figures: `runs/campbell-large-128/`.
- Notebook: `runs/inference-demo-verified/index.html#case=campbell_sky&step=model`.

### Follow-up verification

The targeted suite (`test_campbell_scaling`, `test_campbell_sky`,
`test_reweight`, `test_inference_navigation`, `test_document_status`) completed
with **60 passed, 0 failures/errors/skips**, recorded in
`runs/campbell-notebook-final-tests/{junit.xml,log,exit}`. This is a targeted
run, not the full repository suite. Project Ruff with `--no-cache` passes for
`src/`, `tests/` and the changed example modules.

Browser checks exercise English/中文, the model, sampling, recovery and method
views, Home/ArrowRight navigation, and switching to the original linear
Gaussian case while preserving the active stage. Campbell images load, math
has no KaTeX errors, and the inspected views have no page-level horizontal
overflow or browser warnings/errors. A direct-script renderer regression also
verifies that the new case remains usable from `plot_results.py`.

## Continuous discussion and restored distributions (2026-09-15)

The presentation now uses a continuous eight-section argument, with an anchor
contents list and bilingual text, rather than the shared five-step panel
layout. Existing `step=model/methods/diagnostics/sampling/recovery` bookmarks
remain valid; new `section=distributions/campbell/edgeworth` anchors expose
the added discussion. Other cases keep their original controls.

New distribution figures read the saved 128×128 map. The source histogram
pools independent source cells; the beam histogram pools correlated pixels
and carries no fictitious independent-bin error bars. Histogram normalization
uses all cells, including those outside the display range. An independent
NumPy/SciPy mixture supplies both PDFs: the central count is Poisson(λ), the
sum of the four neighbour counts is independently Poisson(4λ). Numerical
quadrature checks normalization and the first three moments against Campbell.

The signed Edgeworth tail is plotted on a symmetric log axis, linear near
zero. Negative probability density stays visible. A separate family plot
compares λ=0.5, 2 and 8 at fixed mean and covariance. Two additional conditional
posterior illustrations use 40,000 fresh Gaussian draws per coordinate at
the injection, seed 9017. These are explicitly distinct from the full-field
inference banks. Their full target posterior is normalized using the analytic
noisy evidence, and a quadrature test checks this normalization.

The added figures and `distribution-comparison.json` live beside the saved
notebook case. Their renderer is `campbell_distributions.py`, recorded in
`presentation.json`; the existing numerical inference report is unchanged.
The downloadable figures retain the horizontal layout, while narrow notebook
panels use stacked plots with the same data.

Validation for this presentation revision: 23 targeted tests passed
(`runs/campbell-essay-final-tests/{junit.xml,log,exit}`). The mixed-layout
JavaScript regression exercises continuous sections without step controls,
new anchors, legacy bookmarks, language changes, and return to a classic
five-panel demo. Browser inspection confirms eight visible sections and zero
KaTeX errors; narrow views use stacked scientific figures.

## Prior choice separated from inference (2026-09-15)

The fourth section now groups mixture, normalising flow and Edgeworth as
alternative representations of a non-Gaussian prior. It explicitly identifies
the Poisson–Gaussian mixture as the only target used in the recorded experiment.
The discussion distinguishes flow as a prior from flow as a proposal, and
explains why fitting a prior from observations does not inherently require
simulated training maps. Edgeworth is presented with its weak-non-Gaussian
validity limits, rather than as an unrestricted family of proper densities.

Edgeworth formulas, signed-tail figures and the joint negative-density witness
are in a closed optional expansion within that section. The general histogram
figure compares only the proper mixture and its matched Gaussian. The sampling,
weights, ESS, scaling and recovery sections contain no Edgeworth discussion.
The new anchor is `section=priors`; the old `section=edgeworth` bookmark still
lands at the prior-choice section.

The installer regenerated the notebook from saved data. Its inference
`result.json` is byte-identical to `runs/campbell-large-128/result.json`.
Validation: 23 targeted tests passed with no failures, errors or skips
(`runs/campbell-prior-choice-tests/{junit.xml,log,exit}`), and project Ruff
passed for the changed Python files. Browser checks confirmed both languages,
the new directory link, all eight visible sections, the optional expansion,
loaded figures, zero KaTeX errors and no page-level horizontal overflow.

## Show the useful Edgeworth regime first (2026-09-15)

The prior-choice section now opens its Edgeworth illustrations with linear-scale
PDF comparisons at λ=8 and 32, at fixed mean and variance. These are analytic
examples, not additional inference runs. The same expansion through order 1/λ,
including the squared-skewness term, nearly overlaps the exact mixture and
improves on the matched Gaussian. `edgeworth_accuracy.{png,pdf}` and its stacked
PNG appear before the optional λ=2 tail stress case. All Edgeworth discussion
remains within the prior-choice section.

`distribution-comparison.json` records the parameters, integrated absolute
errors, integration interval, grid size and negative mass on that interval.
On [−10σ, 16σ], using 26,001 points, the Gaussian/Edgeworth errors are
0.0643341/0.00118056 at λ=8 and 0.0319151/0.000141821 at λ=32 (roughly 54× and
225× improvement). Independent adaptive quadrature reproduced all four errors
to absolute differences below 5.3e−9. The text distinguishes density validity
from approximation accuracy and does not claim global positivity from a plot.

Validation: 23 targeted tests passed (`runs/campbell-edgeworth-accuracy-tests/`),
and project Ruff passed for the changed Python files. Browser checks confirmed
the successful examples are visible before the collapsed stress case, both
languages render, the narrow view selects stacked figures, and there are no
KaTeX errors or page-level horizontal overflow. The saved inference result
remains byte-identical to the original large-map run.
