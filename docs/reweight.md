# Gaussian reference sampling and non-Gaussian hyperparameter MAP

> **文档状态：`module-spec`** · 固定参考后验样本、完整目标密度重加权与边缘超参数 MAP；flow 后续事项见根目录 TODO.md。

## Statistical target

The input is a fixed bank of unweighted draws from a reference posterior,
usually a linear-Gaussian graph sampled cheaply with GCR. Let

    x_i ~ p_0(x | d) = p_0(x, d) / Z_0
    log w_i(eta) = log p_eta(x_i, d) - log p_0(x_i, d)

where `eta` contains the non-Gaussian hyperparameters. Then

    Z_eta / Z_0 = E_{p_0(x | d)}[w(eta)]
    objective(eta) = logsumexp(log w_i(eta)) - log N + log p(eta).

Optimizing this objective estimates a **marginal hyperparameter MAP**. The
latent field `x` is integrated out using importance sampling. It is not a
joint field/hyperparameter MAP, an average of conditional MAPs, or the EM
objective `mean(log w)`. Without a hyperprior it is marginal maximum
likelihood (empirical Bayes). The logarithm of the Monte Carlo estimate is
biased at finite N; optimizing it can exploit noise in the fixed bank.

When only the prior changes, the common likelihood cancels and
`w = p_eta(x) / p_0(x)`. The graph entry evaluates **both complete joint
densities**, so a changed noise model or graph-level factor is included too.
Every hyperparameter-dependent normalizing constant must be retained.
Normalized posterior weights are only used for posterior expectations and
diagnostics; summing them cannot recover the marginal likelihood objective.

The returned `log_mean_weight` is a **log evidence ratio**, not an absolute
log evidence. Add a separately established `log Z_0` to obtain the latter.
The low-level callback entry also allows a normalized proposal denominator;
then its log mean weights estimate absolute target evidence. In particular,
draws from the Gaussian **prior** require the likelihood in the numerator.
The caller must specify the density of the distribution actually sampled.

This is a numerical composition primitive, like `optimize.minimize`, not a
new certified `EvidenceTask` route. It does not create an `EvidenceResult`
or attach an exactness/convergence verdict to Monte Carlo output.

## API and sampling workflow

```python
import jax
import jax.numpy as jnp
from bayesmith import compile
from bayesmith.reweight import PosteriorReweighting

# reference_graph has fixed Gaussian priors and a linear, fixed-noise model.
# target_graph(eta) declares the full non-Gaussian model of the SAME data.
plan = compile(reference_graph)
assert plan.sampled is None and plan.exact is not None
assert plan.exact.method == "gcr" and not plan.sigma_needs_rebuild
reference = plan.sample(jax.random.key(1), num_samples=10000)
assert reference.log_weights is None

problem = PosteriorReweighting.from_graphs(
    reference_graph, reference.samples, target_graph,
)
fit = problem.fit(
    {"shape": jnp.array(0.0)},
    log_hyperprior=log_hyperprior,
    steps=500, learning_rate=0.01,
    min_ess=1000,  # optional, caller-chosen FINAL Kish-ESS floor
)
eta_map = fit.parameters
weights = fit.importance.weights
```

`from_graphs` requires all latent names, including nuisance latents, and
checks that target and reference condition on the same data and masks.
Each sample leaf has one common leading draw axis. Combine chain/draw axes
explicitly; remaining axes describe each latent's value. A held-nuisance
conditional draw does not estimate the fully marginalized hyperposterior.
The target factory is traced by JAX: graph structure/shapes must remain
static as parameter values change.

For precomputed or specialized densities, construct
`PosteriorReweighting(samples, reference_log_density=..., target_log_density=...)`.
Callbacks return one real scalar per full draw; there is no automatic sum of
an accidentally un-reduced event or batch dimension. The denominator is
cached and fixed, and gradients do not pass through the reference samples.
`estimate`, `log_weights` and `log_objective` support JIT and differentiation.
Use `eqx.filter_jit(problem.estimate)` for an Equinox bound method, or
`jax.jit(lambda eta: problem.estimate(eta))`.

`fit` reuses `optimize.minimize` and accepts its settings, including method,
step budget and learning rate. The returned objective/history have the
maximization sign. Weights and gradient norm are evaluated at the final
parameters. It reports where the budget ended, without claiming a global
optimum or automatic convergence.

Use explicit parameterizations for constrained hyperparameters. If
`eta = exp(u)`, optimizing `log Z(exp(u)) + log p_eta(exp(u))` finds a mode in
`eta`. Adding a Jacobian instead finds a mode of the density in `u`, generally
a different point. This API adds no automatic coordinate Jacobian.

## Reliability and limits

- GCR requires a genuinely linear-Gaussian reference, not merely a Gaussian
  prior. Prediction-dependent noise, nonlinear forward models and conditioned
  blocks need their proper sampling density; scoring another reference joint
  silently changes the estimator. No reference-sampling provenance can be
  inferred from a raw array bank.
- The proposal must cover the target support. Kish ESS, ESS/N and largest
  normalized weight describe concentration on **seen** samples. They cannot
  discover missed modes, establish finite weight variance, or account for
  MCMC autocorrelation. High-dimensional or heavy-tailed corrections can fail
  despite a smooth optimization trace. An optional `min_ess` refusal is only
  a concentration check; it neither certifies overlap nor adapts the proposal.
- Validate the fitted parameters and objective on an independently generated
  reference bank, and against direct integration or independent target
  inference where possible. Weight uncertainty and hyperparameter uncertainty
  are different: the returned weighted samples use a plug-in hyperparameter
  estimate, not integration over the hyperparameter posterior.
- Zero weights are allowed. NaN, positive infinity and no finite weights are
  refused, including under JIT. A negative Edgeworth correction is never
  clipped, dropped or replaced by its absolute value. Finite scores at the
  sampled points do not establish that a truncated Edgeworth density is
  globally nonnegative or suitable as a proper prior. The recovery example
  uses a normalized Gaussian-mixture prior for that reason.

## Executable validation

Run `examples/inference/non_gaussian_reweight.py` to draw from a Gaussian
reference using GCR, fit a non-Gaussian mixture fraction with a Beta
hyperprior, compare against the analytic convolution, and evaluate on a fresh
reference bank. This small example demonstrates parameter recovery; it is
not a claim about high-dimensional cosmological-field performance.

`tests/test_reweight.py` checks independent mixture integrals and gradients,
hyper-MAP against an integrated grid, actual GCR-to-fit composition, changed
likelihood terms, shape semantics, runtime failures and weight concentration.

Background: [prior sensitivity by importance sampling](https://bookdown.org/rdpeng/advstatcomp/importance-sampling.html)
derives the posterior prior-ratio identity. The log-mean objective above
follows directly by integrating the complete joint ratio.
