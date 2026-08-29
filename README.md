# bayesmith

A Bayesian model is a graph of operators. Deterministic operators propagate
dependence; probabilistic operators contribute a conditional density. Together
they *are* the joint distribution.

bayesmith makes that graph **explicit and inspectable**, and then uses its
structure to choose how the model is fitted — an exact solve where the structure
permits one, NUTS where it does not.

```
block 0  {x}          Wiener exact        (linear_in checked, 3 scales)
block 1  {z}          enumerate 4 states
block 2  {sigma, nu}  NUTS (numpyro)      no exact structure found
```

The model tells you how it will be fitted, before it is fitted.

## What bayesmith is not

It is **not another probabilistic programming language**. Distributions, MCMC
kernels, variational inference and transforms all come from
[NumPyro](https://github.com/pyro-ppl/numpyro). bayesmith is the dispatch layer
above them, and every line in it must answer *"why can NumPyro not do this?"*

What it owns, because a trace-based PPL structurally cannot:

- **Structural exact inference** — conjugate / Wiener / GCR / GLS solves, and
  exact enumeration of discrete latents, selected per subgraph.
- **Streamed marginal likelihoods** (`bayesmith.marginal`) — each epoch or
  dataset compressed to a square-root information term, combined exactly. Not
  the Bayesian evidence `p(d)`: a term is a function of the parameters, with
  that dataset's own nuisances integrated away. The subpackage was called
  `evidence` through 0.4.0 and that path still works, with a
  `DeprecationWarning`, until 1.0.
- **Diagnostics on the graph** (`bayesmith.diagnose`) — identifiability and
  prior sensitivity. Linearity checking lives with the solvers that exploit
  it, in `bayesmith.exact.linearity`, because the declaration it checks is
  what those solvers rely on.
- **Two exits an exact solve does not have** — `bayesmith.optimize` for
  gradient MAP on a graph or on any scalar objective, and `bayesmith.amortize`
  for a posterior fitted to simulations rather than evaluated.

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

**0.6.1.** Published so other packages can depend on it by name -- and one
now does, from **nine** of its production modules. rheplicant's auto-partition
and log-space seams import `dispatch.factor.first_fit` and `exact.loglinear`
from here; its adapter, which presents a pipeline as a `Graph`, reads
`AffinityRefused`'s payload and declares complex latents with `ComplexNormal`;
and its diagnostics delegate to `diagnose.identifiability`,
`diagnose.sensitivity` and `diagnose.local`. It pins `bayesmith>=0.4`.

Alpha in the classifier's sense: the API may
still move -- 0.3.0 made `reason` required on `NotGaussian` and
`NotLogLinear`, and 0.4.0 tightens two precision refusals, each breaking for
a caller who was relying on the wrong answer.

Implemented and tested, 1557 tests: the graph core with plates and joint
log-density, with flagged samples declared per node and honoured by every
route; the NumPyro bridge, so any graph is runnable through NUTS;
structural dispatch with the linear-Gaussian exact solves; the FACTOR
partition -- as many exact blocks as the model has factors, grouped by
pairwise probe, with log-space blocks discovered rather than declared
(`factor_partition`, `sample_factors`, `log_space`); exact enumeration of
discrete latents; streaming evidence as square-root information factors; and
graph diagnostics for identifiability, prior sensitivity and linearity.

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
