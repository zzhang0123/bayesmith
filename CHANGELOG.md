# Changelog

## Unreleased

Nothing yet.

## 0.3.0 — 2026-08-27

Released ahead of the rest of P2, and the reason is a rule rather than a
milestone: rheplicant's adapter (`graph_bridge.py`) depends on the two
surfaces below, and the migration plan forbids rheplicant's main from
depending on a bayesmith surface that is not on the index. The plan's
release row said 0.3.0 would carry all of P2; the two rules cannot both
hold, so the row was corrected and this number carries P2a. A patch
release was the other option and was refused: the change below is
breaking for anyone constructing the three error classes directly, and a
breaking change hidden in the patch position is the silently-wrong side.

**A refusal a caller must act on now carries its evidence as data (G11).**
`AffinityRefused` is new -- a subclass of `StructureError`, so every existing
`except StructureError` keeps catching it -- and it carries the probe's own
numbers: `errors` and `weighted` per scale with the passing probes included
(the trend across scales is the diagnostic), the two tolerances actually used,
and the scales that failed. `NotGaussian` and `NotLogLinear` now name their
verdict in a required `reason` field, drawn from the closed vocabularies
`NOT_GAUSSIAN_REASONS` and `NOT_LOG_LINEAR_REASONS`, alongside `node`, `found`,
the measured `fractional` level, and per-node reasons for a graph-level
refusal. Before this, the numbers were rendered into a sentence and dropped,
so the only way for a consumer to read them was to parse prose.

Breaking for anyone constructing these three directly: `reason` is required,
and an unknown one is refused where it is written. Raising them from library
code is unaffected in behaviour -- same classes, same messages, same catch
semantics.

**A complex latent solves in its real degrees of freedom (G9, minimal
surface).** `wiener_solve` and `gcr_sample` accept a block with complex
members: `exact.block.real_parts` splits each into `(re, im)` and the solve
runs there, joining back at the boundary so a caller never learns the
representation. Sky `alm` coefficients are complex while the data they
predict is real, so the map between them is R-linear and not C-linear -- CG
over C would minimise a different objective, the objective having no complex
derivative to descend.

`normal_operator`, `domain_zero` and `variance_parts` now speak that
real-degrees-of-freedom space. For an all-real block it IS the domain, so
every existing caller reads unchanged. A complex member's prior variance is
duplicated across its two halves: each carries `prior_std**2`, so the latent's
total prior variance is `2 * prior_std**2`.

`ComplexNormal` is how a graph declares one, because every numpyro
distribution samples real and a block reads its dtype off the prior's `loc`.
Its two parts are independent and equally wide, so `scale` is the width of
EACH -- the same statement `variance_parts` makes by duplicating it.
`to_numpyro` emits a complex latent as two real sites plus the deterministic
that recombines them, so NUTS still runs on every graph an exact method
accepts, which is what those paths are checked against; the graph's own name
still carries the complex value. An observed or plated complex node is refused
by name rather than emitted untested.

Not yet, and registered rather than left in a docstring: the diagnose family
still refuses complex latents, `exact.correct.log_weight` (SNIS) still indexes
in the domain, and vmap, log space and the Fisher surface are untouched.

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
