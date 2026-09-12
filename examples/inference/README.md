# From operators to recovered parameters

## Real observations: TRIS × Haslam

The notebook has a separate **Real observations / 真实观测** section. Its TRIS
case uses actual LAMBDA archive data, with no simulated input or recovery score.
Run it separately from the six demos:

```bash
# Preparation environment: bayesmith plus requirements-tris.txt and
# requirements-presentation.txt. All inputs are explicit local files.
python examples/inference/tris_prepare.py \
  --archive /path/to/limTOD/downloads/TRIS \
  --haslam /path/to/haslam408_dsds_Remazeilles2014.fits \
  --output runs/tris-input

# The standard bayesmith environment can sample the prepared archive offline.
.venv/bin/python -m examples.inference.tris_case \
  --input runs/tris-input --output runs/inference-demo-verified/tris_haslam

# Render alongside the existing recorded demos; astronomy dependencies required.
python examples/inference/plot_results.py --input runs/inference-demo-verified
```

Open `runs/inference-demo-verified/index.html#case=tris_haslam&step=model`.
The real-data section also remains reachable from the Jeffreys/proposal pages
when those are rendered again. English / 中文, chapter keyboard navigation,
a frequency selector and PNG/SVG figure exports work offline.

Inputs come from the [LAMBDA TRIS archive](https://lambda.gsfc.nasa.gov/product/tris/tris_prod_table.html)
and the [Remazeilles Haslam 408-MHz map](https://lambda.gsfc.nasa.gov/product/foreground/fg_2014_haslam_408_info.html).
Preparation never downloads implicitly. The archive directory must contain
`TRIS_absolute_600.txt`, `TRIS_absolute_820.txt`,
`TRIS_absolute_2500MHz.txt`, and `TRIS_Beam_Profile.txt`.
`manifest.json` records SHA-256 of every observation file, the Haslam input,
limTOD source, preparation code and the resulting `maps.npz`.

The forward equation is an independent JAX implementation of
[`bayesian_skymap`](https://github.com/BellaNasirudin/bayesian_skymap), revision
`e7b8cb34e1a872d11a219f6215e38791058f4e14`, specifically `apply_beam` and
`calc_model_spectral`: reference amplitude times `(nu/408 MHz)**beta`, followed
by the instrument response. **This is a restricted three-region version**:
amplitude and spectral index each have three coordinates for |b| <10°, 10–30°,
and ≥30°. It does not reproduce the upstream free per-pixel amplitude field or
fit a second Haslam gain; Haslam defines the fixed morphology/calibration
reference. The reference map has only the frequency-correct RJ CMB removed,
and the RJ CMB is restored at the TRIS frequencies. Other foreground components
and Haslam zero-level/calibration errors remain unmodelled.

The 600.5/817.8 MHz rings each contain 120 samples. The measured TRIS beam cuts,
pointing conventions and a 0.004-K floor for the one zero-error row per ring
come through limTOD. The six 2427.8-MHz measurements are displayed separately:
their supplied uncertainty is a shared zero level, and no statistical errors
are published. They do not enter the fit. A shared nuisance correction per
fitted frequency carries the 0.066-K and −0.300/+0.430-K zero-level scales;
the asymmetric range already uses astrophysical constraints (TRIS I §5).
Its probability density is an explicit equal-side-mass half-normal assumption.

**A reconstructed map needs its response as well as its errors.** The Wiener
map is `m = b + W d`. Inference fits `b + W (A sky(theta) - correction)` with
observational covariance `W N W.T`, whitened on its supported SVD modes. The
full posterior map covariance is saved and visualized separately. This avoids
treating prior-filled sky pixels as independent observations and avoids
counting the mapmaking prior twice. Tests change that prior and verify
likelihood invariance; the real run also compares likelihood differences with
the original rings before sampling. `--nside 8` is the default coarse grid;
`--nside 16` is supported for resolution checks. Beam interpolation and
pixelization errors are not marginalized in this case.

The recorded case distinguishes **sampler convergence** from **model
adequacy**. It reports posterior predictive residual checks, conditional
95% intervals, correlations and full chain traces. A structured residual much
larger than the supplied noise is a scientific finding, not permission to
inflate the errors. The CLI saves that finding and returns nonzero for failed
sampling diagnostics; it does not label model mismatch as sampling failure.
Interpret parameter intervals conditional on the stated, possibly inadequate,
model. No claim of precise physical spectral-index measurement follows from
converged chains alone.

The **Findings / 发现与解释** chapter displays full-sky posterior mean and SD
maps of `a`, `beta`, and recalibrated Haslam, plus a before/after Haslam comparison
with a common color scale. Recalibrated Haslam is the 408-MHz sky,
`a * (Haslam - CMB_RJ(408)) + CMB_RJ(408)`, before any beam/mapmaking response.
The CMB is not multiplied by `a`; the TRIS instrumental zero-level corrections
are not added to this map. At 408 MHz, beta drops out. These are **three-region
model extrapolations**, with conditional parameter uncertainty, not independent
pixel measurements or a treatment of model discrepancy.

`predictions.npz` includes `amplitude_{mean,sd,lower,upper}`,
`beta_{mean,sd,lower,upper}`, `haslam_{mean,sd,lower,upper}`,
`haslam_change_{mean,sd,lower,upper}` and `haslam_reference_k`. Haslam arrays are
in RJ kelvin; a and beta are dimensionless; lower/upper are 95% equal-tail
limits. All arrays use the same equatorial HEALPix RING grid as `maps.npz`.
Preparation now records `reference_cmb_k` explicitly; older inputs without it
must be prepared again before generating these products.

Methods, diagnostics and sampling reuse the demo presentation components:
actual compiled parameter blocks and method references, a compile-only
alternative strategy, declared priors and bounds, scoped Fisher/Jeffreys
findings, validated per-chain initial states, fixed-budget policy and recorded
checkpoints. Priors remain Uniform for a/beta and Normal for the standardized
corrections. No alternative-prior fit or prior-bound sensitivity study has been
performed for this case.

The case folder contains the full map/covariance/response bundle, input
manifest and four source text files, unweighted posterior samples, predictive
summaries, task/analysis/posterior artifacts, actual DAG and execution plan.
Generated products stay in gitignored `runs/`. The code is in `tris_*.py`;
tests in `tests/test_tris_case.py` need no astronomy data downloads.

## Simulation examples

Six executable examples follow the same sequence: **declare operators → trace a
DAG → simulate data → infer parameters**. Simulation uses known truth; inference
receives only the model, priors and observations. Truth is revealed again for
recovery checks. Each model file follows this sequence, with shared task calls
and checks in `common.py`.

| Model | Graph and operators | What it demonstrates |
|---|---|---|
| `linear_gaussian.py` | Four basis coefficients with bounded Uniform priors | Truncated Gaussian posterior; NUTS, independent Gaussian reference with a negligible-truncation bound |
| `power_law.py` | Two curves `y = A x^alpha + ε`, known additive Gaussian white noise | Four bounded parameters, automatic block plan, refined 2D posterior reference, paired MAP/mean bias experiment |
| `hierarchical.py` | population → random group locations → indexing → observations | Random-node dependencies, a vector latent, hierarchical recovery |
| `bernoulli.py` | Two input features + interaction → four-coefficient logits → Bernoulli | NUTS with a two-dimensional probability surface |
| `multiplicative_noise.py` | `mu = exp(U @ p_g) * (A @ p_n)`; `sigma = 0.001 * mu` | Automatic log-linear + MH for `p_g`, iterative GLS + MH for `p_n`; recorded data-only initialization |
| `composed_process.py` | Random Fourier instance → linear response → nonlinear + linear additions → log-linear gain → unknown-scale multiplicative white noise | 20 coordinates; automatic joint instance/background GLS + MH block (14 coordinates), NUTS remainder (six); explicit data-only initialization |

The Gaussian-observation examples generate data with
`SimulationTask(ParameterSource.fixed(...))`, then run `PosteriorTask`.
The Bernoulli example samples **the distribution returned by its declared
observation operator** directly. Unified SimulationTask / PredictiveTask do not
yet support these families; the examples do not claim to extend that interface.

## Run

From the repository root, with bayesmith and its dependencies installed:

```bash
.venv/bin/python -m examples.inference --case all --output runs/inference-demo
```

Run the new multiplicative example on its own:

```bash
.venv/bin/python -m examples.inference --case multiplicative_noise
```

Defaults are float64, seed 0, 2,000 posterior draws per chain and 1,000 warmup
steps. MCMC routes run two sequential chains; whole-graph GCR draws
independent samples. Graphs, constants and data are created inside an x64 context
that restores the caller's JAX configuration on exit.

`--seed`, `--draws`, `--warmup` and `--output` control the run. Different seeds or
short chains can fail recovery or diagnostics. The program saves the report and
exits nonzero; it never retries seeds or relaxes the acceptance criteria.

## The multiplicative model

The observation distribution is `Normal(mu, sigma)`, where **sigma is a standard
deviation**, recomputed from the current parameters at every likelihood call:

```python
mu = exp(U @ p_g) * (A @ p_n)
sigma = 0.001 * mu
```

The demo uses 128 inputs `x` in `[0, 1]` and four unknown scalar coordinates:

- `U = column_stack([sin(4*pi*x), cos(4*pi*x)])`, shape `(128, 2)`.
- `A = column_stack([ones_like(x), x])`, shape `(128, 2)`.
- `p_g`: independent `Uniform(-0.5, 0.5)` coordinates; generating values `[0.12, -0.08]`.
- `p_n`: independent `Uniform(0.1, 3.0)` coordinates; generating values `[1.2, 0.7]`.

The positive prior for `p_n`, together with this nonnegative `A`, keeps `mu > 0`
and hence `sigma > 0`. There is no `abs`, clipping, or fixed-at-truth noise.
U has no constant column: a constant gain mode could otherwise cancel an overall
rescaling of `p_n` exactly. The tests independently verify full local Jacobian
rank for this design; arbitrary choices of U and A need their own check.

The default task automatically selects **bias-corrected log-linear + MH for
`p_g` and iterative GLS + MH for `p_n`**. The two-dimensional Gaussian proposal
uses dense Cholesky sampling, implementing the Gaussian draw in the
GCR + iterative GLS + MH construction without invoking the matrix-free executor.
Finite Uniform boundaries and the full changing noise covariance enter MH.
Initialization fits the observed data without using the generating parameters.

This bounded automatic extension applies to otherwise all-NUTS posterior plans
with supported parameter-dependent Gaussian noise. It can also enlarge a small
moving-noise GCR block with jointly linear parameters. Each union is rechecked
for joint affinity and aggregate budgets; individually linear products are not
merged. If enlargement fails, the existing GCR plan is retained intact.
Fixed-covariance GCR and explicit schedules retain priority; `auto_proposals=False` in
backend options retains the legacy plan. Unsupported or over-budget candidates
stay in NUTS, with reasons recorded in the analysis artifact.

The composed process samples its instance and hyperparameters jointly, including
the instance's conditional prior. This route does not require marginalisation
or a marginal Fisher calculation. The latter is needed only for the separately
optional marginal-Jeffreys prior construction. A Gaussian-prior sensitivity
test is marked inapplicable to Uniform priors; it does not assess their bounds.

Baseline and Jeffreys/proposal comparison pages share the complete example
catalogue, so opening a counterpart never hides unrelated baseline examples.

## Saved outputs and presentation

Each case directory contains:

- `README.md`: recovery table and the actual DAG.
- `dag.mmd`: Mermaid nodes and edges exported from the traced Graph.
- `plan.txt`: actual execution plan and dispatch reasons.
- `result.json`: checks, chain diagnostics, seed, budget, versions and source hashes.
- `posterior.npz`: full unweighted posterior samples.
- `posterior.artifact.json`: native bayesmith PosteriorResult.
- `analysis.artifact.json` and `task.artifact.json`: original preflight findings and task configuration.
- `result.json` also retains actual chain initial values, checkpoint metrics, stopping rules and reasons.
- `simulation.artifact.json`: native SimulationResult for Gaussian-observation cases.

Plotting is separate; running inference and its tests does not require matplotlib.
Install the optional presentation dependencies and render a saved run:

```bash
uv pip install --python .venv/bin/python -r examples/inference/requirements-presentation.txt
.venv/bin/python examples/inference/plot_results.py --input runs/inference-demo
```

Open **`index.html`**. Both the design guide and example walkthroughs have an
English / 中文 switch and the same chapter layout. Each recorded example shows:

1. **Model & simulation**: equation, dimensions, traced DAG, operators, truth and data.
2. **Basic methods**: the same numbered methods as the guide, marking those actually used; block members, dimensions, conditions and compiler reasons.
3. **Diagnostics & priors**: every saved preflight status, measurements and bounds; supplied priors and Jeffreys assessments.
4. **Sampling**: actual order, initial values, warmup, retained counts, ESS/R̂/MCSE/divergence checks and stop reason.
5. **Recovery**: all-coordinate standardized intervals, posterior correlations, a coordinate selector, histograms and chain traces.

Two-channel models use facets; two-input Bernoulli uses heatmaps. No line is drawn
across unrelated channels or through the two-dimensional input grid. Both languages
share all numerical records and plots; raw compiler evidence retains its original text.

The page works offline, including direct file opening, and has keyboard controls.
Arrow keys move between steps. With JavaScript disabled, all steps are expanded.
View controls never rerun simulation or inference. Narrow intervals receive
additional decimal places so rounding does not hide their width.

Choose **Design & diagnostics** for the compact methodology guide, in this order:
**Basic methods / 基础方法** with numbered cards → parameter grouping and diagnostics → prior checks →
sampling and summaries → six model summaries. **English / 中文** switches paired text without changing
the chapter, formulas, open explanations, statistical goal or loaded evidence.
The four derivation panels under **Details & proofs** are also bilingual.
Cards **5 / 6 / 7 / 8** share a teal **Methods with proposals** badge:
5 / 6 / 7 can construct a drawable, evaluable proposal; 8 applies correction.
The sidebar groups the recorded examples below **Verified with demos**, above
the design and assistance credits. Proposal badges share the method-title row;
importance reweighting remains available on eligible routes with retained weights. MH rejection behavior, target invariance,
and the distinction between blockwise MH and SNIS summaries are explained in
the sampling/proof panels in both languages.
Sampling covers order, partial initialization, fixed/checkpoint stopping and
Rao–Blackwell versus SNIS summaries. Automatic preflight, initialization, stopping and
explicit bounded proposal/MH composition are implemented. The six automatic
recordings remain separate from explicit proposal comparisons at `proposals/index.html`.
Arrow keys navigate chapters, goals, stopping policies and languages. Share a view such as
`?lang=zh#design/diagnostics`.

The complete original English design, including detailed diagnostic contracts,
the implementation inventory and proofs, is preserved in `design-reference.html`,
rendered from the canonical Markdown; `design.md` is a byte-identical copy.
The compact paired guide is maintained in `methodology_guide.py`.
The 500-coordinate panel reads the loaded probe; absent probes are labeled
as absent. Historical measurements are identified separately. Guide controls
illustrate design choices without executing them. KaTeX is bundled locally,
so equations need no network access.

Each case also gets `simulation.png/svg`, `inference.png/svg`, individual
`parameter-<index>.png/svg` plots, `recovery.png/svg` for all coordinates and correlations,
`trace-<index>.png/svg` for MCMC chains, and an exportable `demo.png/svg` histogram figure.
`presentation.json` records rendering sources and input hashes separately from
the numerical run, including the design document and bundled assets. Rendering
leaves `result.json` and `posterior.npz` unchanged. The generated folder includes
`design.md`, `design-reference.html`, referenced local source files, and the offline math assets; keep
them alongside `index.html` when copying the notebook.
**Mean-function intervals exclude new observation noise; they are not posterior
predictive intervals.**

## What counts as parameter recovery?

The criteria are declared before a run in `common.py`:

1. Every generating coordinate lies inside its **99% marginal equal-tail posterior interval**.
2. Each posterior SD is less than **half its marginal prior SD**. This demo
   informativeness requirement rejects an unchanged broad prior that covers truth;
   it is not a general statistical theorem.
3. MCMC reuses bayesmith's ESS / split-r-hat diagnostics and verdicts. Independent
   exact draws have no MCMC convergence check.
4. The linear Gaussian case independently reconstructs the textbook posterior
   from raw x and data. A Gaussian marginal-tail bound verifies negligible prior-box truncation. Sample moments are compared using six ESS-adjusted error bands, without calling the generating operator or graph.

These are **fixed-truth recovery regression tests**, not an SBC study or proof of
99% repeated-experiment coverage. Marginal intervals are not a simultaneous 99%
region. Hierarchical group effects are drawn from their conditional prior; the
population mean is less precisely determined because only eight groups inform it.

## Inference tests

```bash
tools/pytest_gate.py && .venv/bin/python -m pytest tests/test_inference_examples.py
```

The tests execute the CLI examples and check recovery, routes and artifacts.
They also reject biased, uninformative, missing or nonfinite draws and compare
operators with independent numerical values. The multiplicative tests check the
actual graph likelihood at different parameter values, including its changing
standard deviation and normalisation, and check the chosen design's local rank.
Run artifacts and figures live in gitignored `runs/`; example code and tests are
kept in the repository.

## Automatic blocking: strategy, reasons and results

The [block-methodology design](../../docs/superpowers/specs/2026-09-08-block-methodology-design.md)
explains the implemented explicit iterative-GLS, bias-corrected log-linear and
Gauss–Newton proposal builders, their statistical targets, and the remaining
automatic-selection work. The saved demo plans below report what actually ran.

The multiplicative example also simulates a graph with **500 positive `p_n`
coordinates**, two `p_g` coordinates and 1,500 observations. It runs both
partition inspections on that graph and saves them under `blocking_probes`.
The page shows its actual returned blocks and refusal reasons separately.
This larger probe stops at compilation: its sampler is not executed, so the
posterior plots and recovery verdict remain those of the four-coordinate demo.
Its `A` matrix repeats the 500-dimensional identity three times; its `U` matrix
uses the same sine/cosine gain modes. Call `inspect_500_parameters(seed=0)` in
`multiplicative_noise.py` to reproduce this inspection without fitting the demo.

The **Basic methods** step includes an **Automatic blocking** panel for every example.
It reads structured records stored in `result.json` under `blocking`:

- **Executed strategy:** the actual `PosteriorTask` runtime plan, including block
  members, scalar dimension, selected method, variables conditioned on, and the
  compiler's reason. Numerical affinity evidence, condition bounds and solver
  tolerances remain available when the plan returned them.
- **Alternative automatic strategy:** the result of calling `factor_partition`
  on the **same full graph**. The page shows its blocks or the actual exception
  and reason if no partition was returned. This comparison does not generate
  posterior draws or change the executed strategy.
- **Input priors:** distribution families and dimensions read from the graph.

The default compiler's `strategy='declared'` reads structural declarations and
probes them automatically; it is not a hand-written `declared_partition` block
list. The separate factor strategy searches individual linear and log-linear
candidates and groups compatible pairs. Without explicit proposal options,
PosteriorTask uses the default compiler, not `factor_partition`.

For the multiplicative demo as declared here, `compile_task` extends the legacy
NUTS plan with automatically validated `p_g` log-linear/MH and `p_n` GLS/MH
blocks. The separate factor comparison still raises `NotGaussian` for the
bounded Uniform `p_n` prior during the full-graph log-space probe, so it returns **no
factor plan**. The actual plan and the alternative strategy are recorded separately.

Mathematically, `log(mu)` is linear with a fixed offset in `p_g` given `p_n`, and `mu` is linear in
`p_n` given `p_g`. That structure alone does not state which path an executor
selected. For Gaussian relative noise, the log-space transform is a small-noise
approximation. Iterative GLS is a reweighted point estimator; it is not by itself
a positive-parameter posterior sampler or the full heteroscedastic MAP. The
factor sampler currently rejects prediction-dependent-noise GCR blocks rather
than sweeping a corrected `gcr+mh` update.

## Automatic sampling policies

`PosteriorTask` now defaults to bounded automatic preflight and validated per-chain
initialization. Use `StoppingPolicy(mode="checkpoints", ...)` for continuing
checkpoint sampling; fixed-budget sampling remains the default. Inspect
`plan.analysis.findings`, `result.run.initial_values` and
`result.run.sampling_details` for the actual decisions and stop reason.

```bash
.venv/bin/python -m examples.inference.policy_demo
```

This separate simulation/recovery demo saves task, analysis, simulation and
posterior artifacts in `runs/inference-policy-demo`. It does not overwrite the
six gallery recordings. The design notebook's sampling page documents the API
and current boundaries. Explicit bounded proposal/MH composition is configured
separately with the API below.

## Explicit proposal comparisons

Run the three opt-in comparisons, render their separate gallery, then refresh
the main notebook navigation:

```bash
.venv/bin/python -m examples.inference.proposal_demo --output runs/inference-demo/proposals
python3 examples/inference/plot_results.py --input runs/inference-demo/proposals
python3 examples/inference/plot_results.py --input runs/inference-demo
```

`ProposalBlockPolicy` selects `iterative_gls`, `bias_corrected_log_linear` or
`gauss_newton`. Pass `proposal_options(...)` through `PosteriorTask.backend_options`:

```python
from bayesmith.artifacts import ProposalBlockPolicy, proposal_options

options = proposal_options(
    ProposalBlockPolicy(names=("p_g",), method="bias_corrected_log_linear"),
    ProposalBlockPolicy(names=("p_n",), method="iterative_gls"),
)
# PosteriorTask(..., backend_options=options)
```

MH is optional per explicit block and defaults to ON:
`ProposalBlockPolicy(..., mh_correction=False)` disables correction. The CLI
equivalent is `python -m examples.inference.proposal_demo --mh off --output ...`.
An OFF block makes the whole schedule approximate: its unadjusted updates need
not preserve the original posterior or any common surrogate posterior. Support,
finite-target and numerical checks remain active. Counters distinguish applied
proposals from MH acceptances; ESS/R-hat cannot establish absence of approximation
bias. The switch does not alter the existing specialized GCR/SNIS executors.

These blocks run in declaration order; any uncovered parameters form a NUTS
remainder. An all-proposal schedule needs no dummy latent. The default dense
budget per block is 64 parameter coordinates and 200,000 matrix elements.
Members must be real continuous blocks with supported Normal or Uniform priors.
The builders require the supported Gaussian observation/precision interface;
the log builder also accepts canonical LogNormal observations and checks the
positive-data/small-fractional-noise conditions for multiplicative Normal data.
Other factors remain in the full MH target. Complex/discrete proposal blocks and
general non-Gaussian proposal fits remain outside this route.
Fixed-covariance GCR retains priority. The bounded automatic extension may enlarge
a small moving-noise GCR block when proposal options are absent. A 500-coordinate compilation example does not
establish that this bounded dense route fits that problem's budget.

The proposal's draw law and log density match. MH retains the original likelihood,
all prior factors, support, and both forward/reverse normalizers. A rejected step
retains the current block, counts as a step, and continues the sweep; no clipping
or retry-until-accepted rule is used. Saved comparison panels identify the explicit
policies and display attempted/accepted, support-rejected and numerical-failure
counts aggregated over production draws only, excluding warmup. Linear regression
starts at zero; the two nonlinear comparisons explicitly use a bounded L-BFGS
fit from prior centers, using data and priors only. Both chains start at the fit
with independent random streams. This is demo setup, not automatic library
initialization or posterior sampling by the optimizer.

The initial zero/one-start pilot passed linear regression but stalled in the
Gauss–Newton and multiplicative examples. Its failed reports remain in
`runs/inference-jeffreys-mh-verified/proposals`; reproduce that setup with
`--initialization simple`. The data-fit comparisons use the same seed, dataset,
prior bounds and 99% recovery/chain criteria. All three pass at 2,000 retained
draws per chain in `runs/inference-jeffreys-mh-final/proposals`. Good starts help
these narrow proposals; the full convergence checks are still required.

The existing moving-noise route is **GCR + iterative GLS + MH**: iterative GLS
builds frozen noise weights, GCR draws the resulting Gaussian proposal, and MH
corrects it against the original target. With a genuinely Gaussian full conditional
and fixed covariance, GCR is a direct conditional draw and needs no MH correction.

The linear Gaussian diagnostic now uses a primal Jaxpr certificate independently
of prior or sampler. Supported affine means, block-independent covariance, full
float64 numerical rank and consistent probes establish flat Jeffreys density in the block's
model coordinates at the recorded complement. Unsupported derivative semantics,
rank deficiency and numerical contradictions remain visible in the report.

Jeffreys demo pages and comparisons have been removed from the notebook.
Likelihood Jeffreys/flatness diagnostics and the mathematical prior helpers
remain. The retired research runner requires explicit `--case`; historical
results are archived outside the displayed gallery.

The notebook’s example chapter is **Methods and blocks**. Shared symbols identify
parameters throughout formulas, DAGs, block cards and update order. In example 6,
`s` is a random operator and `h(x;c,w)=0.7 exp(-(x-c)^2/(2w^2))` is deterministic.
The probability reading is `p(y,s|phi)=p(y|s,phi)p(s|phi)`; deterministic stages
propagate values between those conditional distributions.

## Finite flat priors and the composed process

The six baseline demos’ **root parameters** use finite Uniform priors declared in `priors.py`; the
notebook prints their actual bounds. Gaussian group/process conditionals remain
part of their hierarchy. These priors are flat in the named model coordinates,
not in arbitrary transformed coordinates, and do not imply a flat Jeffreys prior.
An unbounded flat prior is unsuitable for every parameter by default: for example,
an unbounded decay rate has a nonzero limiting likelihood as the rate tends to
infinity. Bounds express real modeling assumptions and are not inferred from data.

The new process model has 12 sine/cosine coefficients at frequencies 4–9, with
`P_k(a)=(a/k)^2` and one Uniform power amplitude. Its linear response attenuates
high frequencies. A Gaussian bump has unknown center/width; a two-coefficient
background and two gain modes complete the mean. Observations have
`Normal(mu, abs(mu)*sigma_w)` law: this exactly integrates out `w ~ Normal(0,sigma_w)`
in `y=mu*(1+w)`. The factor f=1 is fixed because f and sigma_w cannot separately
be identified from this likelihood.

The registered seed-0 run with generic random starts failed mixing. The demo now
explicitly supplies a bounded L-BFGS fit of the data/model joint density from prior
centers, perturbed in support coordinates for two chains. It never receives truth.
This preserves the compiler's automatic partition; it is an initial-value setup,
not an automatic sampler fallback. Sensible supplied values or a data-only fit are
recommended for coupled models; convergence checks remain necessary.

The current 3,840-observation seed-0 run passes all 20 recovery checks and chain
diagnostics. Its observed sigma_w posterior mean is 0.01499812 for truth 0.015.
The same 99% recovery criterion is retained.

Automatic Fisher supports Gaussian observations and canonical supported Bernoulli
factors. JAX differentiation alone cannot establish a Fisher expectation or global
flatness. Matching local probes on a generic bounded-prior NUTS route remain
unresolved; the linear example separately shows its analytic constant Fisher.
Hierarchical direct-observation rank excludes latent-density information; it is
not a posterior-identifiability or marginal-Fisher verdict.


## Power-law regression and finite-sample bias

Example 02 uses `y_ji = A_j x_i**alpha_j + ε_ji`, with independent
`ε_ji ~ Normal(0, 0.2**2)` and 32 fixed inputs per curve on `[0.5, 2]`.
Both amplitudes and indices are inferred: `A_j ~ Uniform(0.2,3)` and
`alpha_j ~ Uniform(-2,2)`. The registered truth is `A=[1.4,0.9]`,
`alpha=[-1.1,0.8]`; the noise SD is known. Forward data come from the same
declared Gaussian operator through `SimulationTask`. Inference uses the
original observation-space likelihood, without a logarithmic data transform.

Under the baseline prior, the mean is conditionally linear in amplitude and
finite Uniform bounds truncate its Gaussian conditional. Its covariance is
fixed at fixed alpha, so iterative GLS is unnecessary. The exponent enters
nonlinearly; taking log y changes the additive noise and can be undefined.
The current fixed-noise GCR path requires diagonal Normal priors. The actual
automatic block plan therefore uses joint NUTS for these four coordinates.

The retired Jeffreys research comparison retained the same bounds and data.
It is not displayed in the notebook; its mathematical implementation remains.
Writing `S_k = sum_i x_i**(2 alpha_j) (log x_i)**k`, its joint density is
`product_j A_j sqrt(S_0 S_2 - S_1**2) / sigma**2`. It includes amplitude–index
cross information. JAX evaluates the determinant using weighted log-input
variance, avoiding subtraction of nearly equal Gram products.
At fixed alpha, this prior weights the amplitude conditional by A, so that
conditional is no longer a truncated Gaussian. The Jeffreys run also uses NUTS.

`power_law_reference.py` independently integrates each two-parameter posterior
with 128- and 256-point Gauss–Legendre rules per axis. Refinement checks means,
SDs and log normalizers; this is a numerical reference, not a closed-form
posterior. Joint MAP is computed by profiling the conditional amplitude mode,
refining exponent peaks and edge cells, and comparing the boundaries.
The 512 paired forward datasets report MAP/mean bias, Monte Carlo SE, RMSE
and paired prior differences. They measure this registered truth; no check
requires Jeffreys to improve bias. Integration and bias checks run on the host,
outside the JAX inference graph.

The earlier Pareto density model and its Gamma posterior are different from
this regression. For that reference model only, `r=alpha-1`,
`T=sum(log(x))`, and Jeffreys gives `r|x ~ Gamma(n,T)` without extra bounds.
For `n > 1`, its MAP is unbiased, while its posterior mean has bias `r/(n-1)`.
These identities do not apply to `A x**alpha + Gaussian noise`.
The library's canonical Pareto Fisher support and its tests remain available.
The separate `exponential_decay.py` example and its derivations also remain
available as source reference material; the retired “Earlier exponential-decay”
entry and its saved pages are omitted from the active notebook.

Switching examples or the selected prior comparison preserves the current
walkthrough step and language. The notebook omits optional counterpart-status
labels for other cases. Explicitly requesting the unimplemented 06 marginal
Jeffreys comparison with the CLI returns exit 2 and writes status.json;
no posterior result is constructed.


## Demo 06: one dataset with 3,840 observations

```bash
.venv/bin/python -m examples.inference --case composed_process \
  --output runs/inference-demo --seed 0 --draws 2000 --warmup 1000
```

Demo 06 uses one 3,840-point dataset on [0, 1), with the original bounded Uniform
root priors and a 12-coefficient Gaussian Fourier instance. Automatic blocking
jointly updates the instance and linear background using iterative GLS + MH,
then uses NUTS for the remaining parameters. Initialization uses data only.
Two chains each retain 2,000 draws after 1,000 warmup steps.

The saved seed-0 run passes recovery for all 20 parameter coordinates and passes
chain diagnostics with zero divergences. For sigma_w (truth 0.015), the posterior
mean is 0.01499812, SD 0.00017499 and 99% interval [0.01453308, 0.01546047].
The power-amplitude SD is 0.14598: densely measuring one process instance does
not add independent modes. A single recovery pass does not establish unbiasedness
or repeated-experiment coverage.

This is the sole displayed version of demo 06. The notebook defaults to English;
explicit language selection is preserved across examples and steps. Previous
study outputs are archived outside the notebook. Research utilities remain
available separately; they are not linked or embedded as demo cases.
