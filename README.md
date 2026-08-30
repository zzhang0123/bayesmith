# bayesmith

A Bayesian model is a graph of operators. Deterministic operators propagate
dependence; probabilistic operators contribute a conditional density. Together
they *are* the joint distribution.

bayesmith makes that graph **explicit and inspectable**, analyzes the structure,
and compiles an inference route — an exact solve where the structure permits
one, NumPyro NUTS as the current general fallback where it does not.

```
block 0  {x}          Wiener exact        (linear_in checked, 3 scales)
block 1  {z}          enumerate 4 states
block 2  {sigma, nu}  NUTS (numpyro)      no exact structure found
```

The model tells you how it will be fitted, before it is fitted. The longer-term
direction is a task-aware Bayesian workflow layer whose posterior, predictive,
model-checking and evidence results share explicit provenance and quality gates;
the approved boundary and roadmap live in the
[top-level design](docs/superpowers/specs/2026-08-30-bayesmith-top-level-design.md).
Those future protocols are not claimed as current API here.

## What bayesmith is not

It is **not another probabilistic programming language or a sampler zoo**.
[NumPyro](https://github.com/pyro-ppl/numpyro) is the current general posterior
backend; mature upstream libraries should continue to own generic MCMC,
optimization and neural-estimation kernels. bayesmith owns the graph-aware
statistical semantics around those kernels, plus exact routes whose structure
the graph can certify. Its current ownership inventory is
[recorded explicitly](docs/ownership.md).

What it owns today:

- **Graph analysis and structural dispatch** — checked linearity, Gaussianity,
  support, coupling and conditioning claims, followed by an inspectable plan.
- **Structural exact inference** — first-party conjugate / Wiener / GCR / GLS
  solves and exact posterior sampling, plus exact enumeration of discrete
  latents, selected per subgraph.
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

Declarations such as `linear_in` are *claims about the model*, not hints, so
they are **checked rather than trusted**: a node declared linear is probed at
three scales before any exact solve is allowed to use it.

## Worked examples

[`docs/factor-partition-examples.md`](docs/factor-partition-examples.md) walks
two models from declaration to auto-partitioned sampling -- three factors
three routes, then a hierarchy where the ancestry rule earns its keep. Every
printout there was produced by running the code shown, and the partitions are
pinned by ``tests/dispatch/test_factor.py``.

## Status

**0.6.2.** Published so other packages can depend on it by name. rheplicant
uses bayesmith across its production inference layer: its auto-partition and
log-space seams import `dispatch.factor.first_fit` and `exact.loglinear`; its
adapter presents a pipeline as a `Graph`, reads `AffinityRefused`'s payload and
declares complex latents with `ComplexNormal`; and its diagnostics delegate to
`diagnose.identifiability`, `diagnose.sensitivity` and `diagnose.local`. It pins
`bayesmith>=0.4`. The consumer contract is guarded by running rheplicant's own
inference and seam suites against the candidate bayesmith checkout, not by a
hard-coded count of importing modules.

Alpha in the classifier's sense: the API may
still move -- 0.3.0 made `reason` required on `NotGaussian` and
`NotLogLinear`, and 0.4.0 tightens two precision refusals, each breaking for
a caller who was relying on the wrong answer.

Implemented and tested, 1793 tests: the graph core with plates and joint
log-density, with flagged samples declared per node and honoured by every
route; the NumPyro bridge, so any graph is runnable through NUTS;
structural dispatch with the linear-Gaussian exact solves; the FACTOR
partition -- as many exact blocks as the model has factors, grouped by
pairwise probe, with log-space blocks discovered rather than declared
(`factor_partition`, `sample_factors`, `log_space`); exact enumeration of
discrete latents; streamed marginal-likelihood terms as square-root information
factors; and graph diagnostics for identifiability, prior sensitivity and
linearity. A graph-level `EvidenceTask`, Bayes factors and general model
comparison are roadmap work, not present capabilities.

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
