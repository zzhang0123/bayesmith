# Changelog

## 0.2.0 — 2026-08-26

The factor partition and log space. `dispatch.factor` derives as many exact
blocks as the model has factors — pairwise joint-linearity probes, coloured by
`first_fit` — where `partition` finds one block by declaration and drops a
multilinear model whole to NUTS; `sample_factors` sweeps the result, pure
Gibbs when everything is closed-form and through `HMCGibbs` when a remainder
needs NUTS. `exact.loglinear` takes a graph to log space as a graph-to-graph
transform, reading each observed node's scenario by probe — multiplicative
Gaussian (`Normal(mu, f mu)`, first-order with the `f^2/2` shift and a
measured `f <= 0.06` refusal) or log-Gaussian (`LogNormal`, exact, no
threshold) — after which every stock consumer runs unchanged and the noise
genuinely stops depending on the prediction. `NotLogLinear` joins the error
family as the blameless "no log route here" verdict. The grouping rule and
the transform arithmetic are imported by rheplicant rather than re-spelled
there, which is what the 0.2 floor in its pyproject names.

## 0.1.0 — 2026-08-26

First release. Published so that downstream packages can declare a dependency
on it by name rather than by path; until now the version was `0.0.0` and the
package was not on any index.

### What it does

- **Graph core.** Deterministic and probabilistic nodes, plates, and the joint
  log-density assembled from them.
- **NumPyro bridge.** Any graph is runnable through NUTS, which is also the
  oracle every exact path is verified against.
- **Structural dispatch** with the linear-Gaussian exact solves: conjugate,
  Wiener, GCR and GLS, selected per subgraph, with declarations such as
  `linear_in` checked at three scales rather than trusted.
- **Exact enumeration of discrete latents** (`bayesmith.exact.discrete`),
  reading the `Discrete(n)` support declaration. Not yet dispatcher-selected —
  see the README's Status section.
- **Streaming evidence** as square-root information factors, combined exactly
  across epochs.
- **Graph diagnostics**: identifiability, prior sensitivity, linearity.
- **Per-parameter convergence diagnostics** on chain paths: split r-hat and ESS
  per coordinate, gated ESS-first because a fixed r-hat threshold is not a
  well-posed test — see `r_hat_ceiling`'s docstring for the measurements.

### Known limits

- Forward-backward for chain-structured discrete latents is not implemented.
- Discrete enumeration is not yet chosen by the dispatcher.
- The API may move; this is an alpha release.
