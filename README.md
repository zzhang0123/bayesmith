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
- **Streaming evidence** — square-root information factors combined exactly
  across datasets and observing epochs.
- **Diagnostics on the graph** — identifiability, prior sensitivity, and
  linearity checking of the declarations the dispatcher relies on.

Declarations such as `linear_in` are *claims about the model*, not hints, so
they are **checked rather than trusted**: a node declared linear is probed at
three scales before any exact solve is allowed to use it.

## Status

Early development. Design: [`docs/superpowers/specs/2026-08-23-bayesmith-design.md`](docs/superpowers/specs/2026-08-23-bayesmith-design.md).

## License

MIT
