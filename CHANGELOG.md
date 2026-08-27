# Changelog

## Unreleased

Nothing yet.

## 0.4.0 -- 2026-08-27

Cut ahead of the rest of P2, and for the rule rather than the milestone. The
migration plan's release row said the next number would carry all of P2; its
release GATE says only that rheplicant's main may not depend on a bayesmith
surface that is not on the index. Wave A -- the first wave that switches
rheplicant modules onto this package -- needs the four items below, and there
is no reason for it to wait on the five P2 gaps that remain (G2, G9 in full,
G10, G12, G14). The gate is satisfied by publishing before the dependency, not
by publishing last.

Minor rather than patch: two refusals are TIGHTENED below. Anyone handing
`prior_sensitivity` or `JeffreysPrior` a graph whose own arithmetic is single
precision was getting a wrong number and now gets an exception, which is a fix
and is also a behaviour change. Under 0.x that belongs in the minor position,
the same reading 0.3.0 took.


**Two diagnostics could be handed a graph that truncates, and only one of them
said so (D9).** `JeffreysPrior.information` and `prior_sensitivity` now refuse a
graph whose own arithmetic is single precision, the way `identifiability`
always has. The hole was not cosmetic: on an exactly degenerate block, a graph
whose constants were traced OUTSIDE the x64 block gave a half-log-determinant
of **-27.52** where the same block honestly gives **-338.05** -- a 310-nat error
in a log-prior, silent, in a term NUTS exponentiates. The eigenvalue floor was
working; it substitutes the DTYPE's smallest positive number, and float32's is
`log`-87 against float64's -708.

`prior_sensitivity`'s existing guard was on the scalar log-posterior, which
accumulates into a float64 zero and so comes back float64 whatever the graph
did. It now guards the observed nodes' predictions -- what `identifiability`
guards its Jacobian of, and where a truncation actually is. Before, such a
graph reached a `ConvergenceError` thirty lines later that named float32 as one
of three candidate causes.

**And the float32 refusal stays, which is the measured answer to a question the
migration expected to answer the other way.** `docs/probes/probe_13_d9_precision
_policy.py` sweeps a two-component power law across ten decades of conditioning
and asks whether any float32 rank tolerance -- from `1e-8` up to `sqrt(eps)` --
reproduces float64's verdicts. None does, and not because the cut is hard to
place: float64's smallest singular value tracks the model while float32's sits
on its own roundoff floor near 1e-7 and wanders non-monotonically. Two models
float64 separates by two decades come back indistinguishable. A condition-number
ceiling derives from the dtype because it is a statement about available digits;
a rank cut cannot, because in float32 the spectrum stops describing the model
above any cut one might pick.

**Carrying a parameter uncertainty onto a prediction, two ways (G7).**
`propagate_covariance(graph, covariance, at, node=)` is the delta method --
`sqrt(diag(J Sigma J^T))` from one `jacfwd` -- and `push_forward(graph,
samples, node=)` is the Monte-Carlo pushforward of the same quantity. Both are
next to `parameter_covariance` because choosing between them is the decision a
reader arrives with: they agree exactly when the map is affine over the
posterior's width and diverge when it is not.

An OBSERVED node contributes its `loc`. `evaluate` gives such a node its DATA
-- that is what conditioning means -- and data has no parameter dependence, so
propagating the value would report a Jacobian of exactly zero and an error bar
of exactly zero, on every entry, for any model. That was this feature's first
draft, and it was caught by the default-node test passing while comparing two
zeros.

`FlatMatrix` carries `names` and `spans`, so the covariance is checked against
the graph by SHAPE as well as by name. rheplicant's ancestor could only compare
pytree structures, and its own docstring records what that missed: a dict
treedef encodes the key names alone, so two spaces with the same latent names
and different per-latent shapes pass and give finite, wrong error bars. A
matrix whose `kind` is not `"covariance"` is refused outright -- a precision is
the same shape and is wrong by the square of everything.

**`init_to_declared(graph)`** returns the init strategy that starts NUTS at the
graph's declared prior centres, reading them through the public
`prior_environment` so the sampler starts where the classifier looked. `nuts`'s
docstring already carried the measurement (r_hat 1609 and ESS 1.0 from the
default `init_to_uniform` against 1.006 and 138.6 from the declared point);
what was missing was the remedy spelled once.

**The model a graph compiles to now takes `observed=`.** `None` (the default)
conditions on each node's own data, unchanged; a mapping overrides per node;
`{}` conditions on nothing, which is the prior predictive. Needed because
`Predictive` over a model with `obs=` baked in returns the observed node's data
identical in every draw -- measured, a standard deviation of 0 across 3000
draws against a prediction spread of 0.26 -- which is correct of NumPyro and is
not what "posterior predictive" usually means.

**A joint prior is declared ON the graph, so both readers of the model find it
(G13).** `Graph` carries a `joint_prior`, `trace` records one through
`joint_prior(...)`, and BOTH `log_joint` and `to_numpyro` evaluate it -- the
latter as a `numpyro.factor("joint_prior", ...)` site. `JeffreysPrior` could
always evaluate itself; what it could not do was be part of a model. A caller
wrote the factor line beside the model by hand, which meant the graph's own
`log_joint` did not know about it: the exact paths were being checked against a
NUTS potential missing a term, and a model that simply forgot the line sampled
a different posterior with every diagnostic healthy.

One per graph, and the refusal is mathematical rather than clerical: two
Jeffreys blocks are not two independent factors. Each is the CONDITIONAL prior
of its block given the other latents, and a product of conditionals is in
general the joint density of nothing. A model that wants both blocks covered
declares one prior over their union, which is a different -- and often
non-existent -- density.

`graph.py` is the core and imports nothing from `diagnose`, so what it checks
is structural: the object answers `over` and `log_density`, and its block names
latents this graph declares. That last one runs at CONSTRUCTION, earlier than
the prior's own check, which needs values and so cannot run until the potential
does -- a typo in a block name should not survive to the first leapfrog step.
Identifiability and the double-prior refusal stay with `JeffreysPrior` and fire
where it is evaluated.

Not breaking: `joint_prior` defaults to `None` and every existing graph reads
unchanged.

**A sample that was not observed informs nothing, on every exact route (G1).**
An observed node may now declare `observed_mask` -- boolean, shaped like its
data, `True` where the sample was actually taken -- and `precision_at` builds a
`MaskedPrecision` from it. A masked sample gets zero weight in the normal
equations, no term in the log-determinant, no contribution to the
variance-information term and no term in a GCR draw, so the posterior is
exactly the posterior of the model over the samples that were taken. No solver
changed: masking is a new implementation of the existing `Precision` protocol,
which is why every consumer of a covariance inherits it at once and none can
honour it differently from another. `log_joint` and `to_numpyro` honour the
same declaration, the latter through `numpyro.handlers.mask`, so NUTS stays the
oracle the exact paths are checked against.

The mask is declared on the NODE and not spelled as an infinite scale.
`Normal(mu, inf).log_prob` is `-inf` everywhere, so an inf inside the scale is
not a statement that a sample carries no information -- it takes the whole
joint with it. The `sigma = inf` encoding survives at exactly one seam,
`per_sample_sigma`, which reports it back so `GLSResult.noise_std` and
`Estimate.noise_std` say "not observed" in the word their caller used, and so
`evidence.compress` masks a masked covariance without knowing the class exists.
`check_gaussian` still refuses a non-finite sigma, unchanged: "the expression
that produces sigma has an infinity in it" and "this channel was flagged" need
different fixes, and only a declaration tells them apart.

`quadratic` -- and so `log_density`, and so the whole density route -- now
says that a sample whose weight is exactly zero contributes exactly zero,
whatever its residual. The multiplication `residual * apply(residual)` lives
there rather than in `apply`, and `nan * 0` is `nan`, so a masked model whose
solve was clean still handed back a `nan` log-density: measured, `[1, 2, nan,
3]` under a mask `[T, T, F, T]` gave `apply` a clean `[4, 8, 0, 12]` and
`quadratic` a `nan`. For every finite residual this is bitwise the expression
it replaces. The guard is on the WEIGHT, not on the residual: a non-zero
weight times a `nan` is still `nan`, so a poisoned datum that WAS observed
stays loud, and the gradient is checked against its closed form rather than
only for finiteness.

Masking a CORRELATED covariance is refused rather than approximated. The
observed submatrix of a stationary covariance is not itself stationary and its
log-determinant is not a subset sum of the spectrum -- measured on a 6-point
kernel with one sample dropped, `-0.7084` against a closest subset sum 0.47
nats away.

Not breaking: `observed_mask` defaults to `None`, and every existing graph
reads unchanged.

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
