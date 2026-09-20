# bayesmith

bayesmith compiles a Bayesian model, written as a graph of operators, into an
inference procedure that can be audited. It finds and checks the structure of
the graph, solves exactly the parts that admit a closed form, hands the
remainder to a numerical backend, and returns results that carry their
conditions, diagnostics and provenance. It sits between the model declaration
and the sampler: not a probabilistic programming language, and not a collection
of samplers.

**A model says how it will be fitted, and why, before it is fitted.** For a
model whose prediction is affine in `x` once `nu` is fixed, `bayesmith.compile`
prints:

```
block 0  {x}              GCR exact         structurally certified; numerical check 3 scales x 3 at-points (max 0.00e+00)
block 1  {nu}             NUTS
execution: HMCGibbs(inner=NUTS, gibbs_sites=['x']); noise_std rebuilt every sweep
```

`x` has a Gaussian prior and an affine prediction, so its conditional posterior
is Gaussian and is drawn exactly. `nu` is not, so NUTS moves it and every sweep
draws `x` exactly given the current `nu`. Nothing was declared: the affine block
was discovered from the traced program and then checked numerically at three
scales. The script is
[`site/snippets/overview_plan.py`](site/snippets/overview_plan.py), the plan
above is the recorded output in
[`site/assets/plans/overview.json`](site/assets/plans/overview.json), and
`tests/test_site_plans.py` re-measures both.

**A flowchart is a hierarchical model.** The arrows express dependencies:
shared population parameters govern latent variables, deterministic operators
turn them into signals, and observation laws connect those signals to data. The
conditional factors define one joint probability model, and that factorization
is what the compiler reads. The longer-term direction is a task-aware Bayesian
workflow layer whose posterior, predictive, model-checking and evidence results
share explicit provenance and quality gates; the approved boundary and roadmap
live in the
[top-level design](docs/superpowers/specs/2026-08-30-bayesmith-top-level-design.md).
The complete workflow-control layer remains a roadmap item.

The [English documentation](site/index.html) opens with what the package does
and why, then works through examples ordered by how much of the model is solved
exactly, the concepts a plan and a result are made of, and the design and its
verification. Build and verify it with `python tools/build_docs.py --check`. The
0.10.0 candidate is local and unpublished; all documentation pages request
`noindex`, and Pages requires explicit manual opt-in. Source fragments and
generated output are described in [site/README.md](site/README.md).

## What bayesmith is not

It is **not another probabilistic programming language or a sampler zoo**.
[NumPyro](https://github.com/pyro-ppl/numpyro) is the current general posterior
backend; mature upstream libraries should continue to own generic MCMC,
optimization and neural-estimation kernels. bayesmith owns the graph-aware
statistical semantics around those kernels, plus exact routes whose structure
the graph can certify. Its current ownership inventory is
[recorded explicitly](docs/ownership.md).

What it owns today:

- **Graph analysis and structural dispatch** — automatic affine discovery, Gaussianity,
  support, coupling and conditioning claims, followed by an inspectable plan.
- **Structural exact inference** — first-party conjugate / Wiener / GCR / GLS
  solves and exact posterior sampling, selected per subgraph. Exact
  enumeration of declared discrete latents is a separate primitive in
  `bayesmith.exact.discrete`; the compiler does not yet select it.
- **Streamed marginal likelihoods** (`bayesmith.marginal`) — each epoch or
  dataset compressed to a square-root information term, combined exactly. Not
  by itself the graph-level Bayesian evidence `p(d)`: a term is a function of
  the surviving parameters, with that dataset's own nuisances integrated away.
  The subpackage was called `evidence` through 0.4.0 and that path still works,
  with a `DeprecationWarning`, until 1.0.
- **Diagnostics on the graph** (`bayesmith.diagnose`) — identifiability and
  prior sensitivity. Linearity checking lives with the solvers that exploit
  it, in `bayesmith.exact.linearity`, because the declaration it checks is
  what those solvers rely on.
- **Current non-exact exits** — a thin NumPyro bridge, `bayesmith.optimize` for
  gradient MAP on a graph or scalar objective, and `bayesmith.amortize` for a
  posterior fitted to simulations. Their graph-facing contracts belong here;
  their generic optimizer and neural-estimator algorithms are reference or
  upstream-candidate implementations rather than a commitment to grow local
  algorithm families.

`linear_in` is optional. The compiler discovers supported affine structure
from the primal computation, with all complementary latents kept symbolic.
It checks numerical behavior separately and distinguishes structural proof
from finite probes. Unsupported structure falls back conservatively.
Multiple certified Gaussian conditional blocks run as a Gibbs chain; they
are not reported as one joint Gaussian solve or independent draws. See
[automatic affine discovery](docs/automatic-affinity.md) for examples and
the proof's scope.

## Worked examples

[`docs/factor-partition-examples.md`](docs/factor-partition-examples.md) walks
two models from declaration to auto-partitioned sampling -- three factors
three routes, then a hierarchy where the ancestry rule earns its keep. Every
printout there was produced by running the code shown, and the partitions are
pinned by ``tests/dispatch/test_factor.py``.
### The typed task workflow (R1)

R1 publishes a serialisable, invalidatable, evaluable protocol in
`bayesmith.artifacts`, reached from the root through two lazy entry points.
The five Tasks map one-to-one onto five Results; a task that cannot be
compiled or executed returns a typed `Refusal` rather than an exception in
disguise. The legacy entry points (`compile()`, `InferencePlan.sample()` /
`.estimate()`, `Posterior`, `Estimate`, `fit`) are unchanged and remain
supported -- the typed entry points wrap them, they do not replace them.

```python
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

import bayesmith
from bayesmith.artifacts import PosteriorTask, model_ref_from_callable, new_task_meta


def model(data):
    x = bayesmith.sample("x", lambda: dist.Normal(0.0, 2.0))
    bayesmith.observe("d", lambda v: dist.Normal(v, 0.5), x, obs=data)


graph = bayesmith.trace(model, jnp.array([1.0, 2.0]))
task = PosteriorTask(meta=new_task_meta(label="radiometer posterior"))
ref = model_ref_from_callable(model, identifier="radiometer")

planned = bayesmith.compile_task(graph, task, model_ref=ref, key=jax.random.key(0))
result = bayesmith.execute_task(planned, key=jax.random.key(0))
# result is a PosteriorResult: a DrawsPosterior or WeightedDrawsPosterior,
# with run provenance, fingerprints and a frozen Refusal.grounds on refusal.
```

`model_ref_from_callable` derives the source digest automatically when the
model is defined in an importable module. A model typed into a REPL or built
by `exec` has no recoverable source, so pass `source_digest=...` explicitly --
`repr()` is never a fallback, because it carries a memory address.

The full protocol -- the five-in/five-out table, fingerprint boundaries, the
invalidation matrix, `Refusal.grounds` and the gate truth table -- is
documented in [`docs/artifacts.md`](docs/artifacts.md).

Since 0.7.0 the `PredictiveTask` executes as well (R2): `execute_task(planned,
key=..., source_posterior=...)` pushes a `PosteriorResult`'s draws onto the
observed nodes and returns a `PredictiveResult` whose replicated draws and
pointwise log-likelihood come from one loc/scale read, so an observed-data
replay is never mistaken for a posterior predictive. A source posterior drawn
under different data, graph or model is refused as `posterior_data_mismatch`;
a correlated or non-Gaussian observed node is refused as
`predictive_noise_unsupported`. An optional, export-only ArviZ seam lives in
`bayesmith.bridge.arviz`.

`SimulationTask` executes too, since R3 opened the evaluation layer: prior,
fixed and posterior-sourced parameter sources all run through the same forward
primitives the predictive seam uses, so there is one forward model rather than
a simulator beside it.

`EvidenceTask` executes since R4, and it is the last of the five. It answers
`log p(d)` for one structure class -- a whole-graph-exact linear-Gaussian
model -- and the answer arrives as five separately derived
`EvidenceComponent`s rather than as a scalar, because every theta-independent
constant that a posterior never sees is load-bearing in an evidence. Anything
outside that class is refused as `evidence_residual_integral_required`, which
names the numerical integral R5 supplies rather than returning a number
nothing graded.

Its refusals are the interesting half. A prior with infinite mass is refused
as `evidence_prior_proper`; one whose mass is finite but not one as
`evidence_prior_normalised`; a latent covered by the graph-level reference
prior as `evidence_prior_undeclared`; a float32 environment as
`evidence_requires_x64`, because the same assembly agrees with a dense
analytic value to 3.4e-07 at that precision and 8.9e-16 at float64. In every
case the SAME graph still compiles a posterior task unchanged -- that
asymmetry is why compilation is task-aware at all.

R3's own surface is model checking. `bayesmith.evaluation.check_posterior(graph,
posterior, key=..., budget=..., model_ref=...)` runs the checks that apply to
one fitted posterior — posterior and prior predictive checks, held-out
prediction on the points the graph's mask withheld, PSIS-LOO through the
optional ArviZ seam, identifiability and prior sensitivity — files each as a
typed `EvaluationReport`, and aggregates them under the versioned
`model_checking@1` gate. Every report answers on **two** axes rather than one:
an applicability (`APPLICABLE`, `INAPPLICABLE`, `UNVERIFIABLE`) beside a
conclusion (`PASS`, `FAIL`, `ABSTAIN`), so that "this check does not apply
here", "it applies but its inputs were missing" and "it ran and the model
failed" are three answers instead of one word. Only an `APPLICABLE` check may
pass or fail, which is what keeps a run failure from being dressed as a verdict
about the model. One false-positive rate is declared in advance for the whole
layer — `ALPHA = 0.05` — and the draw and replicate floors are derived from it
rather than chosen beside it. The page is
[`docs/evaluation.md`](docs/evaluation.md).

**What a model-checking PASS does not promise** — stated here for the same
reason the Status section below names what this release does not do yet: a
front page is a claim, and finding out afterwards is worse than reading it now.
A predictive check bounds the statistics it computed and nothing wider:
`curved_line(0.15)` — a straight line fitted to data with a real quadratic
term, a quarter the size of one the same check catches at p = 0.0000 — passes
all five default discrepancies, and a green test in the suite exists to pin
that it does. An SBC pass says a route's stated
uncertainty is consistent with its stated prior, which a "posterior" that
ignores the data and returns prior draws satisfies by construction. Neither is
a defect being disclosed; both are what these checks measure. Every predictive
check writes the first caveat into its own report's `meta.summary` — "a pass
bounds these statistics and nothing wider" — so that one travels with the
artifact rather than living only on a page.

## Status

**0.10.0.** The 0.9.0 stable baseline plus a convergence certificate for the
gradient route: `minimize(..., certify=<limit>)` and `fit(..., certify=<limit>)`
return a `Fit` whose `converged` is a proof that the point is within `limit` of
the minimum, in units of the posterior's own sigma, and a stated refusal where
the arithmetic cannot support the claim. The package remains pre-1.0:
experimental APIs and unavailable routes are explicitly separated from the
maintained public surface. See the
[user documentation](https://zzhang0123.github.io/bayesmith/) and
[implementation ownership](docs/ownership.md).

The version in this checkout is not proof of publication. Consumers should
check the [package index](https://pypi.org/simple/bayesmith/) for installable
releases. rheplicant
uses bayesmith across its production inference layer: its auto-partition and
log-space seams import `dispatch.factor.first_fit` and `exact.loglinear`; its
adapter presents a pipeline as a `Graph`, reads `AffinityRefused`'s payload and
declares complex latents with `ComplexNormal`; and its diagnostics delegate to
`diagnose.identifiability`, `diagnose.sensitivity` and `diagnose.local`. It pins
`bayesmith>=0.5`. The consumer contract is guarded by running rheplicant's own
inference and seam suites against the candidate bayesmith checkout, not by a
hard-coded count of importing modules.

The 0.9 baseline is pre-1.0: existing supported interfaces and architecture
boundaries are maintained under the [stability policy](docs/stability.md).
Correctness repairs may reject previously accepted inputs that produced a wrong
answer; such changes require release notes and consumer regression checks.

Implemented and tested, 7,212 tests: the graph core with plates and joint
log-density, with flagged samples declared per node and honoured by every
route; the NumPyro bridge, so any graph is runnable through NUTS;
structural dispatch with the linear-Gaussian exact solves; the FACTOR
partition -- as many exact blocks as the model has factors, grouped by
pairwise probe, with log-space blocks discovered rather than declared
(`factor_partition`, `sample_factors`, `log_space`); exact enumeration of
discrete latents; streamed marginal-likelihood terms as square-root information
factors; and graph diagnostics for identifiability, prior sensitivity and
linearity. A graph-level `EvidenceTask` executes whole-graph linear-Gaussian evidence
with proper normalized priors and x64. Residual numerical evidence and general
model comparison remain outside the completed public execution routes.

Posterior tasks also support automatic bounded Jeffreys diagnostics and explicit
proposal/MH schedules. `ProposalBlockPolicy` selects iterative GLS, bias-corrected
log-linear or Gauss–Newton proposals; `proposal_options(...)` passes their ordered
blocks through `PosteriorTask.backend_options`. The full model corrects every
proposal, with an optional NUTS remainder. These dense builders have independent
size limits. Automatic discovery also handles supported parameter-dependent
Gaussian noise and can enlarge a small moving-noise GCR block with jointly
linear parameters. Fixed-covariance GCR retains priority; failed enlargement
preserves the original plan. See
[`examples/inference/README.md`](examples/inference/README.md) for the bilingual
notebook and executable comparisons.

**Two things the page above describes that this release does not do yet.** Stated here
because a front page is a claim, and finding out afterwards is worse than
reading it now:

- **Enumeration is not dispatcher-selected.** `bayesmith.exact.discrete`
  computes the exact marginal and the posterior marginals over declared
  discrete latents, and reads the `Discrete(n)` support declaration to do it —
  but `classify` does not yet route a discrete subgraph to it. The
  `block 1  {z}  enumerate 4 states` line above is therefore a design sketch
  rather than a transcript; call the module directly.
- **Forward-backward is not implemented**, so a chain of `T` discrete latents
  costs `n ** T` by enumeration rather than `T * n**2`. Enumeration refuses
  past a budget rather than hanging, and names the count it would have visited.

## License

MIT
