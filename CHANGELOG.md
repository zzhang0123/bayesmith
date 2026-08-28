# Changelog

## 0.5.0 -- 2026-08-28

Carries everything the migration had left that did not itself need a release:
the last of P2 (`optimize`, the complex face of the Fisher surface, the
partition executor's three completions, the measured-kappa diagnostic), plus
`evidence.chain`, `exact.reduced_basis`, `amortize`, the evidence consumption
surface, and `local_block(priors=True)`. The release gate is what makes this a
number rather than a milestone: rheplicant's Wave B, C and D cannot wire onto
any of it until it is on the index, and `local_block(priors=True)` in
particular is holding a one-line change in rheplicant's `uncertainty` that has
been written and waiting since Wave A.

Minor rather than patch, and for the reason 0.3.0 and 0.4.0 took the same
position: two defects below CHANGE VALUES a caller could have been reading.
The prior-curvature broadcast returned a symmetric, finite, plausible matrix
whose off-diagonals were wrong -- measured 0.25 too large on a (3,) block at
`prior_std = 2.0` -- and `compress` returned a term whose `information()` was
finite and well conditioned while its `target` was NaN. Both are fixes; both
are behaviour changes; under 0.x that belongs in the minor position.

Nothing here is breaking in the API sense. Every new name is additive, every
new keyword defaults to the old behaviour, and no existing signature changed.

### Changed

**`bayesmith.evidence` is now `bayesmith.marginal`.** The old path still works,
with a `DeprecationWarning`, and retires at 1.0.

The old name claimed something the package does not do. The Bayesian evidence
is `p(d) = INT p(theta) PROD_i L_i(theta) d theta` -- the parameters integrated
OUT, a single number, what model comparison needs -- and nothing here computes
or consumes one. What these modules build is each dataset's marginal likelihood
`L_i(theta)`, its own nuisances integrated away: a function of the parameters.

The cost of keeping it was not confined to the subpackage. `evidence` is a word
this package uses correctly in fifteen other places in its ordinary English
sense -- "the grounds for a verdict" -- including in `errors.py`, the only
eagerly imported module and so the first bayesmith prose anyone reads. A
subpackage holding the term made every one of those ambiguous, and a disclaimer
written inside the subpackage cannot reach the prose outside it. Inside, the
word had four meanings at once, the sharpest being `chain.py`'s "add one
epoch's evidence to the running joint form" -- a single likelihood factor is
exactly what the word does not mean.

`marginal` was the only candidate with no collision: `factors/`, `streaming/`
and `information/` are blocked by `dispatch/factor.py`, `dispatch/streaming.py`
and `fisher_information`; `compress/`, `campaign/` and `sqrtinfo/` collide with
modules inside the subpackage, and that shadowing trap is live -- with
`compress` re-exported, `type(bayesmith.marginal.compress).__name__` is
`'function'`, so the module of the same name is permanently hidden.

Deep imports are aliased through `sys.modules` rather than a module-level
`__getattr__`, because `__getattr__` does not support them:
`from pkg.old.kernel import helper` raises `ModuleNotFoundError` against one.
That matters here -- of the seventeen names 0.4.0 published from this
subpackage, the intersection with the top-level `__all__` was EMPTY, so every
one was reachable only by the deep path. All four import forms are pinned.

### Fixed

**`SqrtInfo` accepted a complex term and returned a silently wrong number
(D46).** Every quantity in this form is BILINEAR -- `log_prob` takes
`sum(residual**2)`, which is `r^T r`, and `information()` takes
`factor.T @ factor` with no conjugate -- while a complex QR's `Q` is unitary
and preserves `r^H r` instead. Measured on one shared complex scalar with
`R_1 = [[1j]]` and `R_2 = [[1]]`: the summed information is exactly 0 by hand,
`combine` returns 2.0, and nothing was raised. An absolute error equal to the
whole of the true value.

This was reachable rather than hypothetical -- `compress` accepts a complex
design block, and the package's target domain is visibilities. `marginal/` was
the only kernel family that neither handled nor refused one: `exact/block.py`
has `real_parts`, `optimize.py` has `_refuse_complex`, `exact/fisher.py`
branches on `is_complex`, and `diagnose` refuses by decision. Now refused at
`SqrtInfo.__check_init__`, which every term passes through, with the route out
named: carry the latent as its real degrees of freedom. Cost measured at zero
-- 289 tests across `tests/marginal` and `tests/crosscheck` pass unchanged.


**Three submodules were unreachable after a bare `import bayesmith`, and which
ones worked depended on what you had touched first.** `bayesmith.optimize`,
`bayesmith.amortize` and `bayesmith.distributions` were missing from
`_LAZY_SUBMODULES`, so attribute access raised `AttributeError` -- unless
something had already resolved a name that imports the module as a side
effect. Measured: `bayesmith.optimize` raised, then `bayesmith.fit` succeeded,
then `bayesmith.optimize` succeeded. An attribute whose presence depends on
access order is worse than one that is simply absent.

This is the third time the same hole has been repaired, and the reason it
recurred is the guard: the tests pinned the table one name at a time
(`assert "evidence" in _LAZY_SUBMODULES`), so each fix covered exactly the name
someone had already thought of. The check now DERIVES the expected set from the
package directory with `pkgutil.iter_modules` and asserts both directions, and
a second test resolves each submodule in a fresh interpreter so the
order-dependence itself is pinned rather than the symptom.


**`fisher_information(include_prior=True)` added the prior's curvature to
every OFF-DIAGONAL entry as well, for a scalar `prior_std` against a vector
latent.** The curvature was built as `jnp.diag(reshape(1 / prior_std**2,
(-1,)))`; a scalar reshapes to length 1, `jnp.diag` of that is a **1x1**
matrix, and adding a 1x1 to an `n x n` broadcasts it into every element. The
diagonal came out right, the matrix stayed symmetric, finite and plausible,
and nothing said anything. Measured on a hand-built `(3,)` block at
`prior_std = 2.0`: every off-diagonal 0.25 too large.

Why it survived 0.4.0: the GRAPH route never reaches it. numpyro broadcasts a
`Normal`'s scale to its batch shape, so `linear_operator` hands over a
full-shaped `prior_std` and the concatenation happened to be the right length.
A hand-built block reaches it, and a COMPLEX block reaches it always -- there
the real degrees of freedom outnumber the declared entries two to one, which
is how it was found.

The curvature now comes from `variance_parts`, which is the same spelling
`normal_operator` reads, so the dense route and the iterative one cannot
disagree about it; and it is broadcast per span, so a width that is neither 1
nor the latent's own size raises instead of wrapping.

**`compress` multiplied by a zero weight where it should have SELECTED on
`seen`.** A flagged sample carries `sigma = inf`, so `whiten` gives it weight
zero and it contributes nothing -- but `0.0 * nan` is `nan`, and a flagged
sample is usually flagged precisely because it holds one. Measured on a
four-sample epoch with `sigma = inf` at index 2: a NaN in the DATA there left
`factor` clean and poisoned `target`; a NaN in the DESIGN there did the
reverse.

The first is the quiet one. `offset` stays finite and `information()` reads
`factor.T @ factor`, which stays finite and well conditioned -- so a campaign
audits as healthy while every density it produces is NaN, and once the term is
folded into an accumulator that is irreversible. Both halves now select before
whitening, and an epoch with nothing flagged is bitwise unchanged.

### Added

**G5: `bayesmith.amortize` -- a posterior fitted to simulations rather than
evaluated from a likelihood.** `NeuralPosterior`, `train_posterior`,
`TrainingHistory`, `MIN_SCALE`. A conditional Gaussian mixture over the latent
vector, an MLP from a summary of the data to its weights, means and scales,
and a hand-rolled Adam that maximises the mean `log q(theta | x)` over a bank
of pairs drawn from the joint.

**The simulator is not here, and that is the decision rather than the omission
(D42).** Of the four public pieces upstream, three take only arrays and one
takes a parameter space, a pipeline and a noise model; the seam is readable
off the signatures rather than chosen. Drawing `(theta, x)` means generating
data, and for a multiplicative instrument model the generative law and the
density differ by an absolute value and a floor -- so a node's `dist_fn`
pressed into service as a simulator would silently swap one for the other.
Every entry point here is array-level and takes no `Graph`.

**Single precision is accepted, unlike `reduced_basis` and `diagnose.local`.**
Those refuse it because a rank verdict or a Gram matrix sits underneath the
rounding. Nothing here returns a verdict, and float32 is the precision such a
network is normally trained in.

**A diverged run has two outcomes and only one is refused (D43).** Measured at
a thousand times the working rate: with a held-out split the best-step
selection keeps the last finite parameters, so a usable estimator comes back
and the divergence is recorded in `history.validation` and in a `best_step`
that has collapsed to 1. With `validation_fraction=0.0` the parameters
themselves are NaN, and every later `log_prob` and `sample` returns NaN,
correctly shaped, from an object that looks like any other posterior. That one
is refused, by `eqx.error_if` on the parameters, on the same reasoning as
`minimize`'s. Non-finite entries in the history are left in place: they are the
only record that anything happened.

A **negative** `min_scale` is refused at construction. It is subtracted from a
strictly positive softplus, so a sufficiently negative component gets a
negative scale and `log(scale)` is NaN for every query while the network's
parameters stay finite -- which is why the guard above cannot reach it. A
**zero** floor is NOT refused: softplus never reaches zero, and a deliberately
collapsible bank trains to a finite density with or without it. There is no
failure there to refuse.

**The over-fitting numbers this module documents were re-measured rather than
carried over, and the mechanism changed.** The upstream implementation blames
the step count -- 0.88 of the exact width at 1500 steps, 0.60 at 4000. Measured
here on the linear-Gaussian problem the tests use: 8192 pairs at 4000 steps
comes back at **1.002**, while 512 pairs at 8000 steps comes back at **0.271**
with its training loss still improving. Over-fitting arrives when capacity
outruns the bank, not when the step count rises.

**G6: the evidence consumption surface -- what a campaign can say about itself
from stored terms.** `residual_summary` and `ResidualSummary`,
`epoch_residuals`, `refuse_mixed_templates`, `template_modes`, `held_out_z`,
`shrinkage_power`, `shrinkage_report`, `systematic_floor` and
`tightest_direction`. Array-level throughout: the containers stay upstream
under D12, so what comes back is a `SqrtInfo`, a NamedTuple or a dict.

`residual_summary` is the half of an epoch's compression that `compress_epoch`
did not have -- the chi-square left AFTER the epoch's own best fit, its
degrees of freedom, and the projections of named systematic templates. It has
to be computed while the raw data exists, because no later call can recover
it. **It must be given every column the epoch fits, nuisances included**: leave
them out and their contribution stays in the residual while the dof is
over-counted by their rank. Measured over 400 clean epochs of a two-survivor
design with a three-column nuisance -- including it gives dof 19 and a mean
chi-square of 18.55, excluding it gives dof 22 against **1227**, a sixty-fold
detection of nothing at all on data with no fault in it.

**"This template lies inside the design's span" is a relative test, and that is
arithmetic rather than taste.** In exact arithmetic such a template leaves
nothing and `norm > 0.0` would be the whole check. In floating point it leaves
roundoff -- measured at 6.0e-07 of its own norm in float32 for a template that
IS a design column -- and the projection then divides by that roundoff norm and
returns an arbitrary unit vector's dot with the residual, measured -0.2517: an
ordinary-looking projection standing for "fully explained", along a direction
the SVD's rounding picked and no other machine would pick. The cut is
`sqrt(eps)` of the arithmetic in hand -- the same formula `numerical_rank` uses
and a different rule, because declaring a template in-span makes it quieter and
quieter is the safe direction for a detection statistic.

`template_modes` is **not** `coherent_mode` under another name (D45). That one
evaluates the residual at a point you choose and can be re-asked anywhere; this
one reads what compression recorded, carries the named templates, and sums each
epoch's OWN degrees of freedom -- `Var(sum chi2_k) = 2 sum k`, so a campaign
whose epochs differ in flagging has a null the first term's row count cannot
express. The two docstrings point at each other.

`held_out_z` scores each epoch against the rest by subtracting one
positive-semidefinite summand from a total formed once, never by downdating a
QR, which cannot be un-summed stably. Checked against a leave-one-out posterior
refitted from scratch in numpy: worst relative error 1.5e-16 over twelve
epochs. A single rogue epoch scores +457.6 against a largest-other +8.0.

`systematic_floor` watches the tightest DIRECTION of a latent's posterior
block, not its tightest coordinate, and the difference is a basis rotation
wide. Measured on a near-collinear campaign: both coordinate widths are 702
while the tightest direction is 0.0583 -- twelve thousand times narrower --
so a coordinate reading reports an error bar comfortably above a 0.1 floor
while the campaign is well under it. The crossing epoch is computed from the
observed width and the test reaches it rather than quoting it.

`shrinkage_power` is a sanity check and says so in `shrinkage_report`'s own
fields: `sigma_N` is data-independent for a Gaussian model, so the power is
-0.5 by construction and a deterministic error shared across epochs cannot
move it at all.

**G3: `evidence.chain` -- the recursion that integrates a linked nuisance out
exactly.** `LinearGaussianTransition`, `HyperTransition`,
`ornstein_uhlenbeck`, `chain_marginal`, `chain_log_likelihood`, `smooth`.

Carry a joint square-root information factor over `(theta, zeta_e)`; fold an
epoch in by re-triangularising, advance by widening, appending the
transition's rows and marginalising `zeta_e`. That drop IS the Schur
complement in square root, which is what keeps a thousand-epoch accumulation
inside float64 where the explicit `(F, b)` form goes indefinite. `theta` is
never marginalised, so the result is `log p(d_1:N | theta)` exactly -- checked
against a dense joint assembled in numpy at four probes, on a width-3 chain
with a rotating `phi`, and over a 20-epoch campaign, to 1e-9.

The two sub-scopes are a TYPE rather than a caveat. An OU with an inferred
correlation time is still linear-Gaussian, so a caveat phrased that way is
satisfied while its claim fails: a filter run once at compression time pins
`Q(theta)` and `phi(theta)` silently. A `LinearGaussianTransition` holds
numbers; a `HyperTransition` holds a builder resolved INSIDE the likelihood,
so the recursion is differentiated on every leapfrog step -- pinned against a
finite difference.

**Six constants reach the answer and the recursion's shape, gradient and
curvature are correct without any of them**, so every test that checks a mean,
a width or a derivative passes on a version that has dropped one. They are
deleted one at a time and measured, and a further test asserts that deleting
them leaves the `SqrtInfo`'s factor and target bitwise unchanged -- which is
why only a dense comparison can notice.

**G4: `exact.reduced_basis` -- selection and orthonormalisation.**
`orthonormal_transform`, `orthonormalise`, `numerical_rank`, `select_svd`,
`select_greedy`. The ARRAY-LEVEL linear algebra only: the containers and the
declaration layer that builds a bank from a parameter space stay upstream, per
D12 and the G6 enumeration.

Selection and basis are different things and the separation is load-bearing.
The selectors choose CANDIDATES; `orthonormalise` turns candidates into a
basis. Storing raw candidates gives a Gram matrix no float64 quadratic form
survives, which is also why `numerical_rank` cuts at `sqrt(eps)` and not at
`eps` -- the quadratic form squares the conditioning, so a set that is merely
invertible is not usable. `orthonormal_transform` returns the TRANSFORM rather
than the rows, because a basis has to be applicable to the raw rows too: the
whitened copy is infinite wherever the reference could not see, and that is
what a zero weight means rather than a limitation.

**Every entry point refuses ambient float32, by name.** A separate guard from
`diagnose.local`'s, with a separate argument: that one is about a rank verdict
below float32's roundoff, this one is about the Gram matrix. The retention cut
is `sqrt(eps)` of the arithmetic in hand -- 3.4e-04 in float32 against 1.5e-08
in float64 -- so on a foreground-dominated bank every direction below a
ten-thousandth of the largest is silently dropped, which is precisely the
direction a reduced basis exists to keep. The message says to build the BANK
inside the block, because widening only the call recovers nothing.

**G15: `local_block(..., priors=True)` -- a nonlinear model's local block that
also carries the declared priors.** The gap in one sentence: a nonlinear
model's posterior precision at a point needs the Jacobian from `local_block`
and the prior from `unchecked_operator`, and neither had both. The second
linearizes at the domain's ZERO, which is the same tangent everywhere only
when the map is affine -- on `mu = a x**b` its design is `a log x` where the
one at `b = 2` is `a x**2 log x`, a different matrix on every row but `x = 1`.

A third constructor rather than a change of mind about the first: the default
still carries empty prior fields, so `fisher_information(include_prior=True)`
still fails loudly on it, and every argument in `diagnose.local`'s module
docstring still holds word for word. The priors are read through
`_env_before`, the one place that turns a latent's declaration into
`(shape, dtype, prior_mean, prior_std)` -- so there is no second spelling, and
its `check_gaussian` comes along: a member whose prior has no quadratic form
is refused by name instead of contributing a silent zero to a posterior
precision. With `priors=False` there is no such refusal, because nothing is
being read off it.

**G9 in full: the complex latent reaches the dense route, the SNIS weight, vmap
and log space.** Two of those four already worked and two did not, measured
rather than assumed.

`dense_operator` -- and therefore `fisher_information`, `parameter_covariance`
and `propagate_covariance` -- used to raise `jacfwd requires real-valued
inputs`. It now lays a complex latent out over its **real degrees of freedom**,
real half first, `2n` rows for `n` entries: the same statement the iterative
route made at G9's minimal surface, because the map from complex coefficients
to a real prediction is R-linear and not C-linear, and JAX's complex gradient
is the conjugate one. Checked against the dense reference that pushes real
basis vectors through the block's own `forward`, element for element, and end
to end against the 4x4 posterior covariance it inverts in numpy.

`exact.correct.log_weight` used to raise inside `jax.linearize`. It now takes
the quadratic form in PARTS space, where `normal_operator` lives. The reason
is not that it raised: `x^T M x` over C is not the form `q` was built from, so
the day it stopped raising it would have been silently the wrong scalar. For an
all-real block the value is bitwise what it was, and a test computes the old
spelling to say so.

`jax.vmap` over `gcr_sample` with a complex latent, and the whole log route
(`log_space`, `check_log_linearity`, a `log-gcr` block, `sample_factors`),
both already worked and now have guards -- including one on a model whose
`log(mu)` is R-affine in a complex latent, where the sweep recovers both
halves of the truth.

**`diagnose`'s refusal of a complex latent STAYS**, and that is a decision. A
rank verdict over C is neither `n` nor `2n`, and splitting the diagnostic
would be a second semantic decision -- what a null direction in R^2n means for
a latent declared over C -- that nobody has taken. The same graph still
solves, samples and reports a Fisher; only the two rank-style diagnostics
stand down, and a test pins both directions.

**`condition_estimate` -- the MEASURED kappa, as a diagnostic and never a
guard (G14, ledger D15(a)).** `condition_bound` measures only the top of the
spectrum and replaces `lambda_min` with the prior's own curvature, which makes
it an upper bound -- the direction a safety guard needs. This one measures
both ends, and it is biased in the other direction, so it is not
interchangeable with it and the docstring says so first.

The bias is not a budget problem and the numbers are in the tests rather than
in a claim. On `geomspace(1, 1e7, 50)`, whose true `lambda_min` is 1.0:
50 steps give 10210.8, 200 give 2351.3, 800 give 805.9, and 2000 give 501.2 --
so the kappa it reports is 2.00e4 against a true 1e7 even after forty times
the work. The shifted operator's leading eigenvalues crowd against
`lambda_max` with vanishing gaps, and no iteration count separates them.

What it can do is what a bound structurally cannot: SEE a degeneracy. A
near-degenerate partition lives entirely in `lambda_min`, which the bound
floors. On `collinear_pair` -- the data fixes `a + b`, the prior alone fixes
`a - b` -- the joint block's measured kappa exceeds a single member's by more
than an order of magnitude beyond what their bounds differ by. That is the
question this answers: "how badly conditioned is this partition?", never "is
this solve accurate enough?".

`extreme_eigenvalues` comes with it, in `exact.conditioning`, reusing
`largest_eigenvalue` for the top so there is one power iteration and not two.
That module's docstring used to say the routine was "deliberately not ported";
the argument it gave is unchanged and is still why it must never be a guard,
but the sentence was about the guard and had become false of the package.

**The factor execution surface gains the three things D14 named (G10), and
G12 with them.** All three land ON `sample_factors` rather than beside it: the
migration plan's v2 believed this package had no multi-block executor and was
wrong, and starting a second one would break the one-implementation rule from
the inside.

`declared_partition(graph, blocks)` builds a `FactorPlan` from a block table
the caller decided. No affinity probe runs -- counted, not asserted:
`check_linearity` is entered zero times -- and no movement gate. You declare,
you are responsible: a `"gcr"` block whose prediction is not affine in its
members gives a draw from a linearisation, silently, with a converged residual
and a healthy chain, and nothing here can tell you so because the check that
would have is what the entry skips. Every block records `declared` and `not
probed` in its `reason`, so a plan read later cannot be mistaken for a derived
one. What is still refused is bookkeeping rather than modelling: a name that
is not a latent, a latent in two blocks or in none, an empty block, an unknown
method, a second `"nuts"` block. An incomplete cover is refused rather than
swept into a NUTS block, because inventing that decision is the opposite of
what the entry is for.

`sample_factors(..., on_sweep=)` calls back with a `SweepReport` after every
sweep -- index, warmup flag, the values kept, the joint log-density there, and
a relative CG residual per exact block, which this executor used to drop on the
floor. The joint is the chi-square trajectory in the spelling that survives a
non-Gaussian model: for a Gaussian one `-2 log_joint` IS the chi-square up to a
constant that does not move along a trajectory. **Refused on a plan with a
NUTS remainder**, and the reason is measured rather than defensive: the sweep
is then HMCGibbs's `gibbs_fn`, which numpyro traces -- entered twice at the
Python level for five sweeps of a two-block plan -- so a callback there fires
once, at trace time, and reports a sweep that never happened.

`estimate_factors(graph, plan, sweeps=)` is a POINT by block coordinate
ascent: exact blocks solved for their conditional mean by `wiener_solve`, and
the remainder, if any, stepped by `fit`. It takes no key and is deterministic;
a sweep that drew would land near the mode too and differ run to run. It
answers what `InferencePlan.estimate` refuses -- a graph that is not exact
throughout -- and on a plan carrying a `gcr` block, a `log-gcr` block and a
`nuts` remainder it recovers `log_gain` 0.469 against a simulated 0.470 and
`centre` 0.1033 against 0.100.

Because a Gaussian conditional's mean IS its mode, a Wiener sweep is exact
coordinate ascent on the joint, so `SweepEstimate.history` is non-decreasing
by construction on an all-exact plan -- the assertion a stale-environment bug
fails. What it does not fix is the partition: alternating one-latent blocks on
`collinear_pair` is still 0.758 from the joint answer after 300 sweeps, with
the joint log-density monotone the whole way and moving five hundredths of a
nat. Every number a caller could look at says converged. The remedy is the
block, not the sweep count.

**G12 -- sigma frozen at the block's current value** is reachable through that
declared path, and it is the existing rebuild branch rather than new
arithmetic: `precision_at(source, current)` where `current` already holds the
block's own latest draw. The condition is about an OUTSIDE latent, which is
worth stating because it is easy to get backwards -- two blocks over a
prediction-dependent sigma rebuild, one block over the whole model is hoisted
at the prior centre instead, a different approximation with a different error.
Both are approximations with a name and neither is a correctness proof; the
transition is history-dependent and the chain is not in general invariant for
the declared posterior. `"gcr+mh"`, the corrected version, is refused at
construction with its own message rather than the generic unknown-method one:
`_mh_step`'s argument is a single-block one and is enforced by a signature
that has no `x` to pass.

**`fit` -- gradient MAP, the exit an exact solve does not have (G2).**
`wiener_solve` and `iterative_gls` answer a model that is affine in its
latents; everything else has only a gradient, and until now this package had
no exit for it. `bayesmith.fit(graph, at=None, names=None, ...)` maximises the
graph's joint log-density over every latent, or over `names=` with the rest
held, which is block coordinate. `bayesmith.minimize` is the same optimiser on
any scalar objective, for a caller scoring a PREDICTION against DATA rather
than a joint. Adam and plain gradient descent, per-latent step sizes.

**The objective is the FULL density**, `sum log sigma` and any declared
`joint_prior` included, and that is a decision rather than an implementation
detail (D7 in the migration ledger). Under a prediction-dependent sigma the
GLS-flavoured potential a point estimate might otherwise descend has a
DIFFERENT optimum, so the estimate and the draw from one declaration would
target two distributions. `fit` reads `log_joint`, which is also what
`to_numpyro` samples.

`check_loss_sense` comes with it: a log-density has an error's signature and
the opposite optimum, so a minimiser handed one descends a function unbounded
below while the loss history looks like textbook convergence. Measured on a
one-parameter gain fit with truth `g = 1.0`: an error reaches `+0.9999`, a
log-density reaches `-30.7349` with a loss going `-3.2e7 -> -1.3e11`. The
guard has a declared half (`sense`) and a measured one (score the PERFECT
prediction), because a declaration alone is a whitelist and a whitelist is
wrong about exactly the code it has not met.

Two things it refuses rather than returns. A **non-finite** result: plain
gradient descent diverges above `2/L` for curvature `L` while Adam's step is
bounded by its rate, so a `method=` changed without revisiting
`learning_rate=` lands there -- measured at `L = 231`, rate 0.006 converges
and 0.02 gives NaN. And a **complex** starting value: `jax.grad` of a real
objective at a complex point gives the CONJUGATE gradient, so a descent using
it walks the wrong way without erroring.

There is no convergence verdict: `steps` steps are taken and the objective
reached is reported. The starting point is load-bearing and `at=` is how it is
supplied -- measured on a fractional-sigma model, starting from the prior
centre puts the objective at 4.3e8 against 24.2 at the optimum, and 6000 Adam
steps from there travel 0.08 with a monotonically decreasing loss history the
whole way.

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
